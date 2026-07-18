#!/usr/bin/env python3
"""6-hourly rainfall isoband polygons from NOAA CMORPH hourly precipitation.

Higher-temporal-resolution companion to precip_isobands.py (which is CPC
daily). Source: NOAA CDR CMORPH, bias-corrected, HOURLY, 0.25 deg GLOBAL,
one netCDF per hour, anonymous:

    https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/
        access/hourly/0.25deg/<YYYY>/<MM>/<DD>/CMORPH_V1.0_ADJ_0.25deg-HLY_<YYYYMMDDHH>.nc

`cmorph` is a rain RATE in mm/hr, so a 6-hour accumulation is the sum of six
consecutive hourly files. We clip to CONUS, sum each 00/06/12/18 UTC window,
contour into filled isobands, and emit exploded WKB polygons that `stt-build`
consumes directly (same schema as the daily pipeline):

    geometry (WKB)   timestamp (window start, unix ms)   end_timestamp (+6h)
    precip_band (mm/6h, non-numeric RANGE label -> categorical GPU fill)

Raw hourly files are cached under data/cmorph/<YYYY>/<MM>/<DD>/ (resumable:
re-runs skip files already on disk).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from contourpy import FillType, contour_generator
from shapely import Polygon, to_wkb
from shapely.geometry import MultiPolygon

BASE = (
    "https://www.ncei.noaa.gov/data/"
    "cmorph-high-resolution-global-precipitation-estimates/access/hourly/0.25deg"
)
# 6-hour accumulation bands (mm). Lower than the daily bands since each window
# holds a quarter-day of rain. Non-numeric labels keep the column categorical.
DEFAULT_THRESHOLDS = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
# CONUS clip box (lon in 0..360 → 235..294 = -125..-66; lat 24..50).
CONUS_LON = (235.0, 294.0)
CONUS_LAT = (24.0, 50.0)
HOUR_MS = 3_600_000
WINDOW_HOURS = 6


def _fmt(v: float) -> str:
    # Drop the trailing ".0" on integer bins ("1"), keep fractional ones ("0.5")
    # so sub-6h thresholds get sensible band labels.
    return f"{v:g}"


def band_label(lo: float, hi: float) -> str:
    return f"{_fmt(lo)}-{_fmt(hi)}" if hi < 1.0e8 else f"{_fmt(lo)}+"


def _rings_to_polygons(points: np.ndarray, offsets: np.ndarray) -> list[Polygon]:
    rings = [points[offsets[r] : offsets[r + 1]] for r in range(len(offsets) - 1)]
    if not rings or len(rings[0]) < 4:
        return []
    poly = Polygon(rings[0], rings[1:])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return []
    if isinstance(poly, MultiPolygon):
        return [g for g in poly.geoms if not g.is_empty]
    return [poly]


def hour_url_and_path(dt: datetime, cache: Path) -> tuple[str, Path]:
    stamp = dt.strftime("%Y%m%d%H")
    name = f"CMORPH_V1.0_ADJ_0.25deg-HLY_{stamp}.nc"
    url = f"{BASE}/{dt:%Y}/{dt:%m}/{dt:%d}/{name}"
    path = cache / f"{dt:%Y}" / f"{dt:%m}" / f"{dt:%d}" / name
    return url, path


def fetch_hour(dt: datetime, cache: Path, retries: int = 3) -> Path | None:
    url, path = hour_url_and_path(dt, cache)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=60) as r:
                data = r.read()
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)
            return path
        except Exception:
            if attempt == retries - 1:
                return None  # missing hour → treated as dry in aggregation
    return None


def prefetch_year(year: int, cache: Path, workers: int) -> None:
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    hours = [start + timedelta(hours=h) for h in range(365 * 24)]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_hour, dt, cache): dt for dt in hours}
        for f in as_completed(futs):
            done += 1
            if done % 500 == 0 or done == len(hours):
                print(f"  fetched {done}/{len(hours)} hourly files")


def load_window_sum(
    window_start: datetime, cache: Path, window_hours: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sum `window_hours` hourly CONUS grids (mm/hr → mm accumulated). Returns lon,lat,z."""
    total = None
    lon = lat = None
    for h in range(window_hours):
        _, path = hour_url_and_path(window_start + timedelta(hours=h), cache)
        if not path.exists():
            continue
        ds = xr.open_dataset(path)
        sub = ds["cmorph"].isel(time=0).sel(
            lon=slice(*CONUS_LON), lat=slice(*CONUS_LAT)
        )
        vals = np.nan_to_num(sub.values.astype("float64"), nan=0.0)
        if total is None:
            total = vals
            lon = ((sub["lon"].values.astype("float64") + 180.0) % 360.0) - 180.0
            lat = sub["lat"].values.astype("float64")
        else:
            total = total + vals
    if total is None:
        return None
    order = np.argsort(lon)
    return lon[order], lat, total[:, order]


def build(
    year: int,
    out_parquet: Path,
    cache: Path,
    thresholds: list[float],
    simplify_deg: float,
    min_area_deg2: float,
    workers: int,
    skip_fetch: bool,
    window_hours: int,
) -> None:
    if not skip_fetch:
        print(f"Prefetching {year} hourly CMORPH into {cache} ...")
        prefetch_year(year, cache, workers)

    upper = thresholds[1:] + [1.0e9]
    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    epoch_ms = int(year_start.timestamp() * 1000)
    n_windows = (365 * 24) // window_hours

    geoms: list[bytes] = []
    ts_col: list[int] = []
    end_col: list[int] = []
    band_col: list[str] = []

    for w in range(n_windows):
        wstart = year_start + timedelta(hours=w * window_hours)
        loaded = load_window_sum(wstart, cache, window_hours)
        if loaded is None:
            continue
        lon, lat, z = loaded
        if float(z.max()) < thresholds[0]:
            continue
        win_ms = epoch_ms + w * window_hours * HOUR_MS
        cg = contour_generator(x=lon, y=lat, z=z, fill_type=FillType.OuterOffset)
        for lo, hi in zip(thresholds, upper):
            if float(z.max()) < lo:
                continue
            points_list, offsets_list = cg.filled(lo, hi)
            for points, offsets in zip(points_list, offsets_list):
                for poly in _rings_to_polygons(points, offsets):
                    poly = poly.simplify(simplify_deg, preserve_topology=True)
                    if poly.is_empty or poly.area < min_area_deg2:
                        continue
                    geoms.append(to_wkb(poly))
                    ts_col.append(win_ms)
                    end_col.append(win_ms + window_hours * HOUR_MS)
                    band_col.append(band_label(lo, hi))
        if (w + 1) % 120 == 0 or w + 1 == n_windows:
            print(f"  ...window {w + 1}/{n_windows}  ({len(geoms)} polygons)")

    table = pa.table(
        {
            "geometry": pa.array(geoms, type=pa.binary()),
            "timestamp": pa.array(ts_col, type=pa.int64()),
            "end_timestamp": pa.array(end_col, type=pa.int64()),
            "precip_band": pa.array(band_col, type=pa.string()),
        }
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_parquet)
    bands, counts = np.unique(np.array(band_col), return_counts=True)
    print(f"\nWrote {table.num_rows} polygons -> {out_parquet}")
    print("  per-band:", dict(zip(bands.tolist(), counts.tolist())))
    print(f"  windows with rain: {len(set(ts_col))}/{n_windows}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cmorph"))
    ap.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS
    )
    ap.add_argument("--simplify", type=float, default=0.03)
    ap.add_argument("--min-area", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--window-hours",
        type=int,
        default=WINDOW_HOURS,
        help="Hours accumulated per bucket (default 6). Use 2 for 2-hourly.",
    )
    ap.add_argument(
        "--skip-fetch", action="store_true", help="use cached files only"
    )
    args = ap.parse_args()
    build(
        args.year,
        args.out,
        args.cache,
        args.thresholds,
        args.simplify,
        args.min_area,
        args.workers,
        args.skip_fetch,
        args.window_hours,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
