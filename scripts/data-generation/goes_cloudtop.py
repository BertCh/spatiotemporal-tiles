#!/usr/bin/env python3
"""GOES-16 C13 cloud-top "anvil canopy" isoband polygons (storm4d-cloudtop).

Cloud-top layer of the Storm-4D Greenfield campaign (§9.1 of
docs/roadmap/storm-4d-greenfield-2026-07.md). Source: GOES-16 ABI **L2 CMIP
CONUS channel 13** (clean longwave IR, brightness temperature in K) — one
NetCDF per 5-minute scan on the anonymous NOAA S3 bucket:

    https://noaa-goes16.s3.amazonaws.com/ABI-L2-CMIPC/<YYYY>/<DDD>/<HH>/
        OR_ABI-L2-CMIPC-M6C13_G16_s<YYYYDDDHHMMSSf>_e<...>_c<...>.nc

Per scan we read CMI + the GOES fixed-grid scan angles (radians) + projection
attrs, run filled marching-squares isobands (contourpy) every 10 K over
280→190 K directly in fixed-grid space, convert every contour vertex to
geodetic lon/lat with the analytic GOES-R inverse navigation (PUG L1b §5.1.2.8
formula, pure numpy — no pyproj), clip to the storm bbox, and emit exploded
WKB polygons that `stt-build` consumes directly:

    geometry (WKB)        timestamp (scan start, unix ms)
    end_timestamp (+5 min, cross-dissolve)
    bt_band  (str, contract labels "270-280","260-270",…,"200-210","<200")
    top_alt_m (f32, standard-atmosphere height of the band-mid BT; piecewise
              linear 288.15 K→0 m .. 216.65 K→11 km, then +500 m per −10 K,
              capped at 16 km — the FE lifts each band to this altitude)

Knobs (size levers, in order of power — target archive ≤ 300 MB):
    --simplify   Douglas-Peucker tolerance in degrees applied per polygon
                 (default 0.01 ≈ 1 km; native pixel is ~2 km, so this is
                 near-lossless. Coarsen first if over the size gate.)
    --min-area   sliver drop threshold in deg² (default 0.0005 ≈ 5 km² —
                 keeps 1-2-pixel overshooting-top cores in the cold bands)
    --quantize-coords  fixed-point coord storage in meters (default 100 m,
                 invisible vs the 2 km pixel; 0 = raw Float64)

Raw scans are cached under <cache>/ABI-L2-CMIPC/<YYYY>/<DDD>/<HH>/ (resumable:
re-runs skip files already on disk). Use --skip-build to stop at the parquet,
or --stt-build /path/to/stt-build to point at the binary.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
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
from shapely.geometry import MultiPolygon, box

BASE = "https://noaa-goes16.s3.amazonaws.com"
PRODUCT = "ABI-L2-CMIPC"
# Channel filter: mode-agnostic (M3/M4/M6 all appear historically).
_C13 = re.compile(r"-M\dC13_G16_")
# GOES filename scan-start token: s<YYYY><DDD><HH><MM><SS><f> (f = tenths).
_S_TOKEN = re.compile(r"_s(\d{14})_")

# Contract §9.1 storm4d-cloudtop: crop bbox (min_lon, min_lat, max_lon, max_lat).
STORM_BBOX = (-102.0, 37.0, -87.0, 46.0)
# Scan cadence → end_timestamp pad for cross-dissolve.
SCAN_MS = 5 * 60 * 1000

# Isobands every 10 K over 280→190 K. Non-numeric RANGE labels keep bt_band a
# CATEGORICAL column in stt-build (same rule as dbz_band / precip_band). Labels
# are the §9.1 contract byte-for-byte. The "<200" catch-all is contoured from a
# 150 K floor (real cloud tops never reach it); its nominal band is 190-200 so
# its band-mid BT for the height lift is 195 K.
BT_BANDS: list[tuple[float, float, str, float]] = [
    (270.0, 280.0, "270-280", 275.0),
    (260.0, 270.0, "260-270", 265.0),
    (250.0, 260.0, "250-260", 255.0),
    (240.0, 250.0, "240-250", 245.0),
    (230.0, 240.0, "230-240", 235.0),
    (220.0, 230.0, "220-230", 225.0),
    (210.0, 220.0, "210-220", 215.0),
    (200.0, 210.0, "200-210", 205.0),
    (150.0, 200.0, "<200", 195.0),
]


def bt_to_alt_m(bt_k: float) -> float:
    """Contract piecewise standard-atmosphere BT→height (m).

    288.15 K→0 m linear to 216.65 K→11,000 m (6.5 K/km troposphere), then
    +500 m per −10 K below the tropopause, capped at 16,000 m.
    """
    if bt_k >= 216.65:
        return max((288.15 - bt_k) * (11000.0 / (288.15 - 216.65)), 0.0)
    return min(11000.0 + (216.65 - bt_k) * 50.0, 16000.0)


def parse_when(spec: str) -> datetime:
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_bbox(spec: str) -> tuple[float, float, float, float]:
    parts = [float(v) for v in spec.split(",")]
    if len(parts) != 4:
        sys.exit("--bbox must be min_lon,min_lat,max_lon,max_lat")
    return parts[0], parts[1], parts[2], parts[3]


def scan_start(name: str) -> datetime | None:
    """Extract the scan-start time encoded in an ABI filename."""
    m = _S_TOKEN.search(name)
    if not m:
        return None
    tok = m.group(1)  # YYYYDDDHHMMSSf
    year, doy = int(tok[0:4]), int(tok[4:7])
    hh, mm, ss, tenths = int(tok[7:9]), int(tok[9:11]), int(tok[11:13]), int(tok[13:14])
    base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    return base.replace(hour=hh, minute=mm, second=ss, microsecond=tenths * 100_000)


# ── S3 listing (anonymous, ListObjectsV2 REST → XML) ──────────────────────────
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def list_prefix(prefix: str, retries: int = 4) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        url = f"{BASE}/?list-type=2&prefix={prefix}"
        if token:
            from urllib.parse import quote

            url += f"&continuation-token={quote(token, safe='')}"
        body = None
        for attempt in range(retries):
            try:
                with urlopen(url, timeout=60) as r:
                    body = r.read()
                break
            except Exception:
                if attempt == retries - 1:
                    return keys
        root = ET.fromstring(body)
        token = None
        for el in root:
            tag = _local(el.tag)
            if tag == "Contents":
                for child in el:
                    if _local(child.tag) == "Key" and child.text:
                        keys.append(child.text)
            elif tag == "NextContinuationToken" and el.text:
                token = el.text
        if not token:
            return keys


def hour_prefixes(start: datetime, end: datetime) -> list[str]:
    out: list[str] = []
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur <= end:
        doy = cur.timetuple().tm_yday
        out.append(f"{PRODUCT}/{cur:%Y}/{doy:03d}/{cur:%H}/")
        cur += timedelta(hours=1)
    return out


def gather_keys(start: datetime, end: datetime) -> list[str]:
    prefixes = hour_prefixes(start, end)
    print(f"Listing {len(prefixes)} hour-prefix(es) on {BASE} …")
    keys: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for got in ex.map(list_prefix, prefixes):
            keys.extend(got)
    in_window = []
    for k in keys:
        if not _C13.search(Path(k).name):
            continue
        t = scan_start(Path(k).name)
        if t is not None and start <= t < end:
            in_window.append(k)
    in_window.sort()  # filename order == chronological
    print(f"  {len(in_window)} C13 scan(s) in window")
    return in_window


# ── download (resumable, atomic) ──────────────────────────────────────────────
def fetch(key: str, cache: Path, retries: int = 3) -> Path | None:
    path = cache / key
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{key}"
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=120) as r:
                data = r.read()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)
            return path
        except Exception:
            if attempt == retries - 1:
                return None
    return None


def download_all(keys: list[str], cache: Path, workers: int) -> list[Path]:
    paths: list[Path] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, k, cache): k for k in keys}
        for f in as_completed(futs):
            p = f.result()
            if p is not None:
                paths.append(p)
            done += 1
            if done % 20 == 0 or done == len(keys):
                print(f"  fetched {done}/{len(keys)} scan(s)")
    paths.sort()
    return paths


# ── GOES-R fixed grid → geodetic lon/lat (PUG L1b §5.1.2.8, analytic) ─────────
def fixed_grid_to_lonlat(
    x: np.ndarray, y: np.ndarray, proj: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """x/y are E-W/N-S scan angles in radians; returns (lon_deg, lat_deg).

    proj = (r_eq, r_pol, H, lam0_rad) with H = perspective height + r_eq.
    Off-disk points (negative discriminant) come back NaN.
    """
    r_eq, r_pol, big_h, lam0 = proj
    sin_x, cos_x = np.sin(x), np.cos(x)
    sin_y, cos_y = np.sin(y), np.cos(y)
    ratio = (r_eq * r_eq) / (r_pol * r_pol)
    a = sin_x * sin_x + cos_x * cos_x * (cos_y * cos_y + ratio * sin_y * sin_y)
    b = -2.0 * big_h * cos_x * cos_y
    c = big_h * big_h - r_eq * r_eq
    disc = b * b - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        r_s = (-b - np.sqrt(disc)) / (2.0 * a)  # NaN off-disk
    s_x = r_s * cos_x * cos_y
    s_y = -r_s * sin_x
    s_z = r_s * cos_x * sin_y
    lat = np.degrees(np.arctan(ratio * s_z / np.hypot(big_h - s_x, s_y)))
    lon = np.degrees(lam0 - np.arctan(s_y / (big_h - s_x)))
    return lon, lat


def read_proj(ds: xr.Dataset) -> tuple[float, float, float, float]:
    p = ds["goes_imager_projection"].attrs
    r_eq = float(p["semi_major_axis"])
    r_pol = float(p["semi_minor_axis"])
    big_h = float(p["perspective_point_height"]) + r_eq
    lam0 = np.radians(float(p["longitude_of_projection_origin"]))
    return r_eq, r_pol, big_h, lam0


def compute_slices(
    x: np.ndarray,
    y: np.ndarray,
    proj: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
    margin: float,
) -> tuple[slice, slice]:
    """Index bounding box of fixed-grid pixels whose lon/lat fall in bbox+margin."""
    grid_x, grid_y = np.meshgrid(x, y)
    lon, lat = fixed_grid_to_lonlat(grid_x, grid_y, proj)
    with np.errstate(invalid="ignore"):
        mask = (
            (lon >= bbox[0] - margin)
            & (lon <= bbox[2] + margin)
            & (lat >= bbox[1] - margin)
            & (lat <= bbox[3] + margin)
        )
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        sys.exit("bbox does not intersect the CONUS fixed grid — check --bbox")
    r0, r1 = max(int(rows[0]) - 2, 0), min(int(rows[-1]) + 3, y.size)
    c0, c1 = max(int(cols[0]) - 2, 0), min(int(cols[-1]) + 3, x.size)
    return slice(r0, r1), slice(c0, c1)


# ── isoband extraction ────────────────────────────────────────────────────────
def _rings_to_polygons(points: np.ndarray, offsets: np.ndarray) -> list[Polygon]:
    """One contourpy OuterOffset entry (already lon/lat) → shapely Polygons."""
    rings = [points[offsets[r] : offsets[r + 1]] for r in range(len(offsets) - 1)]
    if not rings or len(rings[0]) < 4:
        return []
    poly = Polygon(rings[0], rings[1:])
    if not poly.is_valid:
        poly = poly.buffer(0)  # fix self-touching rings
    if poly.is_empty:
        return []
    if isinstance(poly, MultiPolygon):
        return [g for g in poly.geoms if not g.is_empty]
    return [poly]


class ScanGrid:
    """Per-run cache: the CONUS fixed grid is constant across scans, so the
    bbox index slice (the expensive full-grid navigation) is computed once."""

    def __init__(self) -> None:
        self.key: tuple | None = None
        self.slices: tuple[slice, slice] | None = None

    def get(self, x, y, proj, bbox, margin) -> tuple[slice, slice]:
        key = (x.size, y.size, round(float(x[0]), 9), round(float(y[0]), 9))
        if key != self.key:
            self.slices = compute_slices(x, y, proj, bbox, margin)
            self.key = key
        return self.slices  # type: ignore[return-value]


def process_scan(
    path: Path,
    grid: ScanGrid,
    bbox: tuple[float, float, float, float],
    margin: float,
    simplify_deg: float,
    min_area_deg2: float,
) -> tuple[list[bytes], list[str], list[float], int] | None:
    """One scan → (wkb_geoms, band_labels, top_alts, n_vertices)."""
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
    except Exception:
        return None
    try:
        if "CMI" not in ds:
            return None
        proj = read_proj(ds)
        x = ds["x"].values.astype("float64")
        y = ds["y"].values.astype("float64")
        row_sl, col_sl = grid.get(x, y, proj, bbox, margin)
        z = ds["CMI"].values.astype("float64")[row_sl, col_sl]
    finally:
        ds.close()

    xs = x[col_sl]
    ys = y[row_sl]
    # contourpy wants ascending axes; GOES y scans north→south (descending).
    if ys[0] > ys[-1]:
        ys = ys[::-1]
        z = z[::-1, :]
    if xs[0] > xs[-1]:
        xs = xs[::-1]
        z = z[:, ::-1]
    z = np.nan_to_num(z, nan=999.0)  # fill/space → warm, outside every band
    z_min = float(z.min())

    clip = box(*bbox)
    geoms: list[bytes] = []
    bands: list[str] = []
    alts: list[float] = []
    n_verts = 0
    cg = contour_generator(x=xs, y=ys, z=z, fill_type=FillType.OuterOffset)
    for lo, hi, label, mid_bt in BT_BANDS:
        if z_min >= hi:
            continue  # nothing colder than this band's top anywhere in crop
        top_alt = bt_to_alt_m(mid_bt)
        points_list, offsets_list = cg.filled(lo, hi)
        for points, offsets in zip(points_list, offsets_list):
            # Contoured in fixed-grid radians → navigate every vertex to lon/lat.
            lon_v, lat_v = fixed_grid_to_lonlat(points[:, 0], points[:, 1], proj)
            pts_ll = np.column_stack([lon_v, lat_v])
            for poly in _rings_to_polygons(pts_ll, offsets):
                poly = poly.simplify(simplify_deg, preserve_topology=True)
                if poly.is_empty:
                    continue
                clipped = poly.intersection(clip)
                if clipped.is_empty:
                    continue
                parts = (
                    list(clipped.geoms)
                    if hasattr(clipped, "geoms")
                    else [clipped]
                )
                for part in parts:
                    if part.geom_type != "Polygon" or part.area < min_area_deg2:
                        continue
                    geoms.append(to_wkb(part))
                    bands.append(label)
                    alts.append(top_alt)
                    n_verts += len(part.exterior.coords) + sum(
                        len(r.coords) for r in part.interiors
                    )
    return geoms, bands, alts, n_verts


def build_parquet(
    paths: list[Path],
    bbox: tuple[float, float, float, float],
    margin: float,
    simplify_deg: float,
    min_area_deg2: float,
    out: Path,
) -> int:
    grid = ScanGrid()
    geoms: list[bytes] = []
    ts_col: list[int] = []
    end_col: list[int] = []
    band_col: list[str] = []
    alt_col: list[float] = []
    total_verts = 0

    for i, p in enumerate(paths):
        t = scan_start(p.name)
        if t is None:
            continue
        t_ms = int(t.timestamp() * 1000)
        got = process_scan(p, grid, bbox, margin, simplify_deg, min_area_deg2)
        if got is not None:
            g, b, a, nv = got
            geoms.extend(g)
            ts_col.extend([t_ms] * len(g))
            end_col.extend([t_ms + SCAN_MS] * len(g))
            band_col.extend(b)
            alt_col.extend(a)
            total_verts += nv
        print(
            f"  scan {i + 1}/{len(paths)}  {t:%Y-%m-%d %H:%M}Z  "
            f"({len(geoms)} polygons, {total_verts} vertices so far)"
        )

    if not geoms:
        sys.exit("No isoband polygons in window/bbox — nothing to build.")

    table = pa.table(
        {
            "geometry": pa.array(geoms, type=pa.binary()),
            "timestamp": pa.array(ts_col, type=pa.int64()),
            "end_timestamp": pa.array(end_col, type=pa.int64()),
            "bt_band": pa.array(band_col, type=pa.string()),
            "top_alt_m": pa.array(alt_col, type=pa.float32()),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy")
    labels, counts = np.unique(np.array(band_col), return_counts=True)
    print(f"\nWrote {table.num_rows} polygons ({total_verts} vertices) → {out}")
    print("  per-band:", dict(zip(labels.tolist(), counts.tolist())))
    print(f"  scans with cloud: {len(set(ts_col))}/{len(paths)}")
    return table.num_rows


# ── stt-build ─────────────────────────────────────────────────────────────────
def build_cmd(parquet: Path, stt_dir: Path, args) -> list[str]:
    cmd = [
        args.stt_build,
        "--input", str(parquet),
        "--output", str(stt_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        "--end-time-field", "end_timestamp",
        "--min-zoom", str(args.min_zoom),
        "--max-zoom", str(args.max_zoom),
        "--temporal-bucket", args.temporal_bucket,
        "--compression", "zstd",
        # Multi-cell time PLAYBACK requires time-major ordering — `auto` can pick
        # spatial and silently stall the playhead (blob-ordering gotcha).
        "--blob-ordering", "time-major",
    ]
    if args.quantize_coords > 0:
        cmd += ["--quantize-coords", str(args.quantize_coords)]
    if args.publish:
        cmd += ["--publish"]
    return cmd


def run_stt_build(parquet: Path, stt_dir: Path, args) -> None:
    import subprocess

    cmd = build_cmd(parquet, stt_dir, args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {stt_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", required=True, help="window start, ISO-8601 UTC (e.g. 2024-05-21T17:30Z)")
    ap.add_argument("--end", required=True, help="window end, ISO-8601 UTC (exclusive)")
    ap.add_argument("--bbox", default=",".join(str(v) for v in STORM_BBOX),
                    help="min_lon,min_lat,max_lon,max_lat crop (default storm4d contract bbox)")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="extra degrees contoured beyond bbox before clipping")
    ap.add_argument("--simplify", type=float, default=0.01,
                    help="Douglas-Peucker tolerance in degrees (~1 km; pixel is ~2 km)")
    ap.add_argument("--min-area", type=float, default=0.0005,
                    help="drop clipped polygons smaller than this (deg²; 0.0005 ≈ 5 km²)")
    ap.add_argument("--cache", type=Path, default=Path("data/storm4d/goes"))
    ap.add_argument("--out", type=Path, required=True,
                    help="output .stt directory (or .parquet with --skip-build)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-zoom", type=int, default=3)
    ap.add_argument("--max-zoom", type=int, default=8)
    ap.add_argument("--temporal-bucket", default="5m", help="matches the 5-min scan cadence")
    ap.add_argument("--quantize-coords", type=int, default=100,
                    help="coord quantization in meters (0 = raw Float64)")
    ap.add_argument("--publish", action="store_true", help="zstd-19 deploy build")
    ap.add_argument("--skip-fetch", action="store_true", help="use cached scans only")
    ap.add_argument("--skip-build", action="store_true", help="stop at parquet")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = parse_when(args.start), parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    bbox = parse_bbox(args.bbox)

    keys = gather_keys(start, end)
    if not keys:
        sys.exit("No C13 scans listed for that window — check --start/--end.")

    if args.skip_fetch:
        paths = sorted(p for p in (args.cache / k for k in keys) if p.exists())
        print(f"Using {len(paths)} cached scan(s).")
    else:
        print(f"Downloading into {args.cache} …")
        paths = download_all(keys, args.cache, args.workers)
    if not paths:
        sys.exit("No scans on disk — nothing to read.")

    pq_path = args.out if args.out.suffix == ".parquet" else args.out.with_suffix(".parquet")
    build_parquet(paths, bbox, args.margin, args.simplify, args.min_area, pq_path)

    if args.skip_build or args.out.suffix == ".parquet":
        stt_dir = args.out.with_suffix("")
        print("\nGeoParquet only. Build with:\n  " + " ".join(build_cmd(pq_path, stt_dir, args)))
        return 0

    run_stt_build(pq_path, args.out.with_suffix(""), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
