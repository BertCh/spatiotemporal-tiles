#!/usr/bin/env python3
"""Turn a year of gridded daily precipitation into filled isoband polygons.

Source: NOAA CPC US Unified Gauge-Based Analysis of Daily Precipitation
(0.25 deg CONUS grid, mm/day). One netCDF per year, anonymous:

    https://downloads.psl.noaa.gov/Datasets/cpc_us_precip/RT/precip.V1.0.<year>.nc

For each day and each precip threshold band we run a filled marching-squares
contour (contourpy) and emit the resulting (multi)polygon rings, exploded to
individual Polygons so they tile cleanly. Output is a WKB GeoParquet that
`stt-build` consumes directly:

    geometry (WKB)   timestamp (unix ms, day 00:00 UTC)   end_timestamp
    precip_band (mm, the band's lower bound -> categorical GPU fill)

This is a deliberately SIMPLE, light representation: ~a few thousand polygons
for the whole year (hundreds of polygons on the wettest days, a handful of
bands each), animated over 365 daily buckets. The continental rainfall field
pulses day to day as the companion forcing to the NWM river-discharge demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from contourpy import FillType, contour_generator
from shapely import Polygon, to_wkb
from shapely.geometry import MultiPolygon

# Precip band lower bounds in mm/day. Each band is [lo, next_lo); the last is
# [100, inf). Chosen for a clean rainfall legend: drizzle (<2mm) is dropped so
# the field reads as weather systems, not gauge noise.
DEFAULT_THRESHOLDS = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0]

# Milliseconds in a day; daily buckets align with the NWM river demo.
DAY_MS = 86_400_000
# 2019-01-01 00:00:00 UTC in unix ms (matches nwm-rivers-2019 bucket 0).
EPOCH_2019_MS = 1_546_300_800_000


def _rings_to_polygons(points: np.ndarray, offsets: np.ndarray) -> list[Polygon]:
    """One contourpy OuterOffset entry -> shapely Polygons (exterior + holes)."""
    rings = [points[offsets[r] : offsets[r + 1]] for r in range(len(offsets) - 1)]
    if not rings:
        return []
    shell = rings[0]
    holes = rings[1:]
    if len(shell) < 4:
        return []
    poly = Polygon(shell, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)  # fix self-touching rings
    if poly.is_empty:
        return []
    if isinstance(poly, MultiPolygon):
        return [g for g in poly.geoms if not g.is_empty]
    return [poly]


def build(
    src: Path,
    out_parquet: Path,
    thresholds: list[float],
    simplify_deg: float,
    min_area_deg2: float,
    max_days: int | None,
) -> None:
    ds = xr.open_dataset(src)
    lat = ds["lat"].values.astype("float64")
    # CPC lon is 0..360; fold to -180..180 (CONUS stays monotonic increasing).
    lon = ((ds["lon"].values.astype("float64") + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    lon = lon[order]
    precip = ds["precip"].values.astype("float64")[:, :, order]
    times = ds["time"].values  # datetime64[ns]
    n_days = precip.shape[0] if max_days is None else min(max_days, precip.shape[0])

    upper = thresholds[1:] + [1.0e9]  # each band's exclusive upper bound

    # Non-numeric RANGE labels ("2-5" … "100+") keep precip_band a CATEGORICAL
    # column in stt-build (a bare-integer label would be sniffed as Numeric,
    # defeating the renderer's per-band colorMapping — same rule as dbz_band).
    def band_label(lo: float, hi: float) -> str:
        return f"{int(lo)}-{int(hi)}" if hi < 1.0e8 else f"{int(lo)}+"

    geoms: list[bytes] = []
    ts_col: list[int] = []
    end_col: list[int] = []
    band_col: list[str] = []

    for d in range(n_days):
        z = np.nan_to_num(precip[d], nan=0.0)  # ocean / no-gauge -> dry
        if float(z.max()) < thresholds[0]:
            continue  # bone-dry day, no bands at all
        day_ms = int(EPOCH_2019_MS + d * DAY_MS)
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
                    ts_col.append(day_ms)
                    end_col.append(day_ms + DAY_MS)
                    band_col.append(band_label(lo, hi))
        if (d + 1) % 30 == 0 or d + 1 == n_days:
            print(f"  ...day {d + 1}/{n_days}  ({len(geoms)} polygons so far)")

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
    print("  per-band polygon counts:", dict(zip(bands.tolist(), counts.tolist())))
    print(f"  days with rain: {len(set(ts_col))}/{n_days}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="CPC precip netCDF")
    ap.add_argument("--out", type=Path, required=True, help="output GeoParquet")
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="precip band lower bounds (mm/day), ascending",
    )
    ap.add_argument(
        "--simplify",
        type=float,
        default=0.03,
        help="Douglas-Peucker tolerance in degrees (~3km)",
    )
    ap.add_argument(
        "--min-area",
        type=float,
        default=0.02,
        help="drop polygons smaller than this (deg^2) as slivers",
    )
    ap.add_argument("--max-days", type=int, default=None, help="cap for quick tests")
    args = ap.parse_args()
    build(
        args.src, args.out, args.thresholds, args.simplify, args.min_area, args.max_days
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
