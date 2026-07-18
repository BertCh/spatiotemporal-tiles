#!/usr/bin/env python3
"""Continental precipitation + storm cells + tracks from NOAA MRMS reflectivity.

The precipitation member of the weather suite. Source: NOAA MRMS
`MergedReflectivityQCComposite` — a national 0.01° (~1 km) radar-reflectivity
mosaic every ~2 minutes on the anonymous `noaa-mrms-pds` S3 bucket, GRIB2 (.gz):

    https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MergedReflectivityQCComposite_00.50/
        <YYYYMMDD>/MRMS_MergedReflectivityQCComposite_00.50_<YYYYMMDD>-<HHMMSS>.grib2.gz

For a time window it lists + downloads the composite grids (anonymous HTTPS,
stdlib `urllib` — same pattern as glm_lightning.py), reads each with cfgrib,
masks the no-data sentinels, and from one frame per `--step-min` bakes THREE
archives that mirror the `storms` (NEXRAD) generator so the showcase's `radar`
composite render works unchanged:

    <out>-field    filled dBZ isoband polygons  (categorical `dbz_band` RANGE label)
    <out>-cells    storm-cell centroids (points; `max_dbz`, `area_km2`)
    <out>-tracks   cells linked across frames (LineStrings; per-vertex `max_dbz`)

All contouring (marching squares via contourpy), cell detection (connected
components via scipy.ndimage), and nearest-centroid tracking happen here; the
browser renders finished vector tiles. Grids cache under <cache>/<YYYYMMDD>/
(resumable). Use --skip-build to stop at the parquet, --skip-fetch to reuse the
cache, or --stt-build /path to point at the binary.
"""

from __future__ import annotations

import argparse
import gzip
import math
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = "noaa-mrms-pds"
BASE = f"https://{BUCKET}.s3.amazonaws.com"
PRODUCT = "MergedReflectivityQCComposite_00.50"
PREFIX = f"CONUS/{PRODUCT}"

# Default analysis box (min_lat, min_lon, max_lat, max_lon): CONUS land.
CONUS = (24.0, -122.0, 50.0, -67.0)
# 10-dBZ bands read cleanly at continental scale (fewer, bolder polygons than the
# 5-dBZ NEXRAD bands). Non-numeric RANGE labels keep the column categorical.
DEFAULT_THRESHOLDS = [20.0, 30.0, 40.0, 50.0, 60.0]
M_PER_DEG_LAT = math.pi * 6_371_000.0 / 180.0

_TIME = None  # set lazily to avoid importing at module load


def parse_bounds(spec: str) -> tuple[float, float, float, float]:
    p = [float(x) for x in spec.split(",")]
    if len(p) != 4:
        sys.exit("--bounds must be min_lat,min_lon,max_lat,max_lon")
    return p[0], p[1], p[2], p[3]


def parse_when(spec: str) -> datetime:
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def key_time(name: str) -> datetime | None:
    """Parse the <YYYYMMDD>-<HHMMSS> stamp out of an MRMS filename."""
    try:
        stem = name.split("_")[-1]  # <YYYYMMDD>-<HHMMSS>.grib2.gz
        stamp = stem.split(".")[0]
        return datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ── S3 listing + download (anonymous, stdlib) ─────────────────────────────────
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def list_prefix(prefix: str, retries: int = 4) -> list[str]:
    keys: list[str] = []
    token = None
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
            t = _local(el.tag)
            if t == "Contents":
                for c in el:
                    if _local(c.tag) == "Key" and c.text:
                        keys.append(c.text)
            elif t == "NextContinuationToken" and el.text:
                token = el.text
        if not token:
            return keys


def select_keys(start: datetime, end: datetime, step_min: int) -> list[tuple[datetime, str]]:
    """One key per `step_min` slot: the granule nearest each slot's target time."""
    days = []
    d = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    all_keys: list[tuple[datetime, str]] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for keys in ex.map(lambda dd: list_prefix(f"{PREFIX}/{dd:%Y%m%d}/"), days):
            for k in keys:
                t = key_time(Path(k).name)
                if t is not None and start <= t <= end:
                    all_keys.append((t, k))
    all_keys.sort()
    if not all_keys:
        return []
    # Bucket to step slots, keep the granule closest to each slot centre.
    step = timedelta(minutes=step_min)
    chosen: dict[int, tuple[float, datetime, str]] = {}
    for t, k in all_keys:
        slot = int((t - start) / step)
        target = start + slot * step
        d = abs((t - target).total_seconds())
        if slot not in chosen or d < chosen[slot][0]:
            chosen[slot] = (d, t, k)
    out = [(t, k) for _, t, k in (chosen[s] for s in sorted(chosen))]
    print(f"  {len(all_keys)} granules → {len(out)} frames at {step_min}-min cadence")
    return out


def fetch(key: str, cache: Path, retries: int = 3) -> Path | None:
    path = cache / Path(key).name
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urlopen(f"{BASE}/{key}", timeout=120) as r:
                data = r.read()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)
            return path
        except Exception:
            if attempt == retries - 1:
                return None
    return None


# ── grid read (gz → cfgrib) ───────────────────────────────────────────────────
def read_grid(path: Path, bounds, grid_step: float):
    """Return (lon1d, lat1d ascending, z2d) cropped to bounds and coarsened to
    ~grid_step degrees. No-data sentinels become NaN. z is dBZ."""
    import xarray as xr

    raw = gzip.decompress(path.read_bytes())
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    try:
        tmp.write(raw)
        tmp.close()
        ds = xr.open_dataset(
            tmp.name, engine="cfgrib", backend_kwargs={"indexpath": ""}
        )
        var = list(ds.data_vars)[0]
        da = ds[var]
        lat = ds["latitude"].values.astype("float64")
        lon = ds["longitude"].values.astype("float64")
        lon = ((lon + 180.0) % 360.0) - 180.0  # 0..360 → -180..180
        z = da.values.astype("float32")
        ds.close()
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    # Orient ascending in both axes.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        z = z[::-1, :]
    order = np.argsort(lon)
    lon = lon[order]
    z = z[:, order]

    # Coarsen to ~grid_step by striding (native ~0.01°).
    native = abs(lon[1] - lon[0])
    stride = max(1, int(round(grid_step / native)))
    if stride > 1:
        lon = lon[::stride]
        lat = lat[::stride]
        z = z[::stride, ::stride]

    # Crop to bounds.
    min_lat, min_lon, max_lat, max_lon = bounds
    yi = (lat >= min_lat) & (lat <= max_lat)
    xi = (lon >= min_lon) & (lon <= max_lon)
    lat, lon, z = lat[yi], lon[xi], z[np.ix_(yi, xi)]

    # Mask no-data (-999 / -99) and anything below detection to NaN.
    z = np.where(z <= -30.0, np.nan, z)
    return lon, lat, z


# ── band labels + polygon helpers (from cmorph_isobands.py) ───────────────────
def band_label(lo: float, hi: float) -> str:
    f = lambda v: f"{v:g}"
    return f"{f(lo)}-{f(hi)}" if hi < 1.0e8 else f"{f(lo)}+"


def rings_to_polys(points, offsets):
    from shapely import Polygon
    from shapely.geometry import MultiPolygon

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


def contour_bands(lon, lat, z, thresholds, simplify_deg, min_area_deg2):
    """Filled dBZ isobands → list of (dbz_band, shapely Polygon)."""
    from contourpy import FillType, contour_generator
    from shapely import to_wkb

    zf = np.nan_to_num(z, nan=thresholds[0] - 40.0)
    if float(zf.max()) < thresholds[0]:
        return []
    uppers = thresholds[1:] + [1.0e9]
    cg = contour_generator(x=lon, y=lat, z=zf, fill_type=FillType.OuterOffset)
    out = []
    for lo, hi in zip(thresholds, uppers):
        if float(zf.max()) < lo:
            continue
        pts_list, off_list = cg.filled(lo, hi)
        for pts, off in zip(pts_list, off_list):
            for poly in rings_to_polys(pts, off):
                poly = poly.simplify(simplify_deg, preserve_topology=True)
                if poly.is_empty or poly.area < min_area_deg2:
                    continue
                out.append((band_label(lo, hi), to_wkb(poly)))
    return out


# ── storm-cell detection (scipy connected components) ─────────────────────────
def detect_cells(lon, lat, z, cell_dbz, min_km2):
    """Connected components on z>=cell_dbz → reflectivity-weighted centroids."""
    from scipy import ndimage

    mask = np.nan_to_num(z, nan=-99.0) >= cell_dbz
    if not mask.any():
        return []
    lbl, n = ndimage.label(mask)
    if n == 0:
        return []
    idx = range(1, n + 1)
    zpos = np.nan_to_num(np.maximum(z, 0.0), nan=0.0)
    coms = ndimage.center_of_mass(zpos, lbl, idx)
    maxd = ndimage.maximum(np.nan_to_num(z, nan=-99.0), lbl, idx)
    sizes = ndimage.sum(np.ones_like(lbl, dtype="float64"), lbl, idx)

    dlon = abs(lon[1] - lon[0])
    dlat = abs(lat[1] - lat[0])
    mid = float(lat.mean())
    cell_km2 = (dlat * M_PER_DEG_LAT / 1000.0) * (
        dlon * M_PER_DEG_LAT * math.cos(math.radians(mid)) / 1000.0
    )
    cells = []
    for k in range(n):
        area = float(sizes[k]) * cell_km2
        if area < min_km2:
            continue
        iy, ix = coms[k]
        clon = float(lon[0] + ix * dlon)
        clat = float(lat[0] + iy * dlat)
        cells.append((clon, clat, float(maxd[k]), area))
    return cells


def haversine(lon1, lat1, lon2, lat2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def track_cells(frames, max_speed_mps):
    """Greedy nearest-centroid linking across frames (SCIT-style)."""
    tracks: list[dict] = []
    open_ids: list[int] = []
    for t_ms, cells in frames:
        used = set()
        still_open = []
        for ti in open_ids:
            tr = tracks[ti]
            lt, llon, llat = tr["t"][-1], tr["lon"][-1], tr["lat"][-1]
            gate = max_speed_mps * max(1.0, (t_ms - lt) / 1000.0)
            best, bestd = None, 1e18
            for ci, (clon, clat, md, ar) in enumerate(cells):
                if ci in used:
                    continue
                d = haversine(llon, llat, clon, clat)
                if d <= gate and d < bestd:
                    best, bestd = ci, d
            if best is not None:
                clon, clat, md, ar = cells[best]
                tr["lon"].append(clon)
                tr["lat"].append(clat)
                tr["t"].append(t_ms)
                tr["v"].append(md)
                used.add(best)
                still_open.append(ti)
            # else: track ends (dropped from open set)
        for ci, (clon, clat, md, ar) in enumerate(cells):
            if ci in used:
                continue
            tracks.append({"lon": [clon], "lat": [clat], "t": [t_ms], "v": [md]})
            still_open.append(len(tracks) - 1)
        open_ids = still_open
    return [tr for tr in tracks if len(tr["lon"]) >= 2]


# ── parquet writers ───────────────────────────────────────────────────────────
def write_field(rows, out: Path):
    geom, ts, ets, band = [], [], [], []
    for t, et, b, wkb in rows:
        geom.append(wkb)
        ts.append(t)
        ets.append(et)
        band.append(b)
    tbl = pa.table(
        {
            "geometry": pa.array(geom, type=pa.binary()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "end_timestamp": pa.array(ets, type=pa.int64()),
            "dbz_band": pa.array(band, type=pa.string()),
        }
    )
    pq.write_table(tbl, out, compression="snappy")
    return tbl.num_rows


def write_cells(rows, out: Path):
    lon, lat, ts, md, ar = [], [], [], [], []
    for t, clon, clat, maxd, area in rows:
        lon.append(clon)
        lat.append(clat)
        ts.append(t)
        md.append(maxd)
        ar.append(area)
    tbl = pa.table(
        {
            "lon": pa.array(lon, type=pa.float64()),
            "lat": pa.array(lat, type=pa.float64()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "max_dbz": pa.array(md, type=pa.float64()),
            "area_km2": pa.array(ar, type=pa.float64()),
        }
    )
    pq.write_table(tbl, out, compression="snappy")
    return tbl.num_rows


def write_tracks(tracks, out: Path):
    from shapely import wkb as shp_wkb
    from shapely.geometry import LineString

    geom, ts, ets, vts, vvals, peak = [], [], [], [], [], []
    for tr in tracks:
        coords = list(zip(tr["lon"], tr["lat"]))
        geom.append(shp_wkb.dumps(LineString(coords)))
        t = [int(x) for x in tr["t"]]
        ts.append(t[0])
        ets.append(t[-1])
        vts.append(t)
        vvals.append([np.float32(x) for x in tr["v"]])
        peak.append(float(max(tr["v"])))
    tbl = pa.table(
        {
            "geometry": pa.array(geom, type=pa.binary()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "end_timestamp": pa.array(ets, type=pa.int64()),
            "vertex_timestamps": pa.array(vts, type=pa.list_(pa.int64())),
            "vertex_values": pa.array(vvals, type=pa.list_(pa.float32())),
            "peak_dbz": pa.array(peak, type=pa.float64()),
        }
    )
    pq.write_table(tbl, out, compression="snappy")
    return tbl.num_rows


# ── stt-build ─────────────────────────────────────────────────────────────────
def build(parquet: Path, out_dir: Path, args, *, end_time: bool, bucket: str):
    cmd = [
        args.stt_build,
        "--input", str(parquet),
        "--output", str(out_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", "0",
        "--max-zoom", str(args.max_zoom),
        "--temporal-bucket", bucket,
        "--blob-ordering", "time-major",
        "--compression", "zstd",
    ]
    if end_time:
        cmd += ["--end-time-field", "end_timestamp"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", required=True, help="ISO-8601 UTC")
    ap.add_argument("--end", required=True, help="ISO-8601 UTC")
    ap.add_argument("--bounds", default=",".join(str(x) for x in CONUS))
    ap.add_argument("--step-min", type=int, default=10, help="frame cadence (min)")
    ap.add_argument("--grid-step", type=float, default=0.02,
                    help="coarsen native 0.01° to this (deg); 0.02 ≈ 2 km")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--simplify", type=float, default=0.02)
    # Minimum band-polygon area in DEGREES² (poly coords are lon/lat). 0.01 deg²
    # ≈ 95 km² at mid-CONUS — drops speckle, keeps storm-scale bands.
    ap.add_argument("--min-band-deg2", type=float, default=0.01)
    ap.add_argument("--cell-dbz", type=float, default=50.0)
    ap.add_argument("--min-cell-km2", type=float, default=25.0)
    ap.add_argument("--max-cell-speed", type=float, default=40.0, help="m/s")
    ap.add_argument("--cache", type=Path, default=Path("data/mrms"))
    ap.add_argument("--out", type=Path, required=True,
                    help="base path → <out>-field/-cells/-tracks")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-zoom", type=int, default=8)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = parse_when(args.start), parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    bounds = parse_bounds(args.bounds)
    step_ms = args.step_min * 60_000

    print(f"MRMS {PRODUCT}: {start:%Y-%m-%d %H:%M} … {end:%Y-%m-%d %H:%M}Z")
    frames_keys = select_keys(start, end, args.step_min)
    if not frames_keys:
        sys.exit("No MRMS granules listed for that window.")

    # Download (parallel, resumable).
    if not args.skip_fetch:
        print(f"Downloading {len(frames_keys)} frames into {args.cache} …")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch, k, args.cache): (t, k) for t, k in frames_keys}
            done = 0
            for _ in as_completed(futs):
                done += 1
                if done % 100 == 0 or done == len(frames_keys):
                    print(f"  fetched {done}/{len(frames_keys)}")

    # Process each frame → field polygons + cells; collect cells per frame for tracking.
    field_rows: list[tuple[int, int, str, bytes]] = []
    cell_rows: list[tuple[int, float, float, float, float]] = []
    frames_for_tracking: list[tuple[int, list]] = []
    n_ok = 0
    for i, (t, key) in enumerate(frames_keys):
        path = args.cache / Path(key).name
        if not path.exists():
            continue
        try:
            lon, lat, z = read_grid(path, bounds, args.grid_step)
        except Exception as e:
            print(f"    ⚠️  {path.name}: {e}")
            continue
        if z.size == 0 or not np.isfinite(z).any():
            continue
        t_ms = int(t.timestamp() * 1000)
        for band, wkb in contour_bands(
            lon, lat, z, args.thresholds, args.simplify, args.min_band_deg2
        ):
            field_rows.append((t_ms, t_ms + step_ms, band, wkb))
        cells = detect_cells(lon, lat, z, args.cell_dbz, args.min_cell_km2)
        for clon, clat, maxd, area in cells:
            cell_rows.append((t_ms, clon, clat, maxd, area))
        frames_for_tracking.append((t_ms, cells))
        n_ok += 1
        if (i + 1) % 25 == 0 or i + 1 == len(frames_keys):
            print(f"  frame {i + 1}/{len(frames_keys)}  "
                  f"({len(field_rows)} polys, {len(cell_rows)} cells)")

    if n_ok == 0:
        sys.exit("No usable MRMS frames decoded.")
    tracks = track_cells(frames_for_tracking, args.max_cell_speed)
    print(f"\nBaked {len(field_rows)} band polygons, {len(cell_rows)} cell centroids, "
          f"{len(tracks)} tracks from {n_ok} frames.")

    base = args.out
    # Parquet intermediates live in the cache dir, not next to the served archives.
    args.cache.mkdir(parents=True, exist_ok=True)
    field_pq = args.cache / (base.name + "-field.parquet")
    cells_pq = args.cache / (base.name + "-cells.parquet")
    tracks_pq = args.cache / (base.name + "-tracks.parquet")
    write_field(field_rows, field_pq)
    write_cells(cell_rows, cells_pq)
    write_tracks(tracks, tracks_pq)
    print(f"Wrote parquet: {field_pq.name}, {cells_pq.name}, {tracks_pq.name}")

    if args.skip_build:
        print("--skip-build set; stopping at parquet.")
        return 0

    bucket = f"{args.step_min}m"
    build(field_pq, base.with_name(base.name + "-field"), args, end_time=True, bucket=bucket)
    build(cells_pq, base.with_name(base.name + "-cells"), args, end_time=False, bucket=bucket)
    build(tracks_pq, base.with_name(base.name + "-tracks"), args, end_time=True, bucket=bucket)
    print(f"\n✅ Built {base.name}-field / -cells / -tracks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
