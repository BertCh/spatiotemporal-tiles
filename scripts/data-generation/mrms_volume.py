#!/usr/bin/env python3
"""NOAA MRMS 3D reflectivity mosaic → national STT 3D point volume.

The country-scale companion to `nexrad_volume.py`: instead of decoding one
NEXRAD site with Py-ART, this reads the pre-mosaicked MRMS 3D reflectivity
product — `MergedReflectivityQC_<height>` at 33 vertical levels (0.5–19 km,
1 km horizontal, ~2-min cadence) on the anonymous `noaa-mrms-pds` bucket — and
emits one 3D point per grid cell above a dBZ floor, at that level's altitude.
A whole continent of storms becomes one time-animated 4D reflectivity object.

Reuses the MRMS S3 front-half of `mrms_weather.py` (list/select/fetch/grid-read,
cfgrib) and the point-schema + dBZ-band contract + stratified LOD + stt-build
back-half of `nexrad_volume.py`. No Py-ART, no radar geometry (MRMS is already
QC'd, dealiased-N/A, Cartesian) — reflectivity ONLY (national velocity would
need a ~160-site Level-II mosaic).

SCALE WALL (measured): at ≥10 dBZ, native 1 km, one CONUS frame ≈ 17.6 M points
— roughly the *entire* single-site Greenfield archive, per 5-min frame. So a
national volume trades among extent / grid step / dBZ floor / window length to
stay shippable. Defaults here target a SHORT high-fidelity window; widen the
window only as the byte budget allows. The stratified LOD (deepest zoom lossless)
keeps the *framing* load bounded regardless of total size.

Pipeline:  MRMS 3D (S3, cached) → cfgrib decode ×33 heights/frame → threshold →
           GeoParquet (lon,lat,alt_m,timestamp,dbz,dbz_band,min_zoom) → stt-build

Height product → altitude:  alt_m = height_km × 1000  (MSL, matches the FE's
shared STORM4D_ELEVATION_SCALE; alt_m stays a PROPERTY column, LiDAR pattern).
"""
from __future__ import annotations

import argparse
import gzip
import sys
import tempfile
import time as time_mod
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BUCKET = "noaa-mrms-pds"
BASE = f"https://{BUCKET}.s3.amazonaws.com"

# The 33 MRMS 3D reflectivity height levels (km MSL) — dir names AND altitudes.
HEIGHTS_KM = [
    0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50,
    4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 10.00,
    11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.00,
]
def product_name(h_km: float) -> str:
    return f"MergedReflectivityQC_{h_km:05.2f}"

# CONUS land box (min_lat, min_lon, max_lat, max_lon) — same as mrms_weather.
CONUS = (24.0, -122.0, 50.0, -67.0)

# dBZ band contract — MUST match nexrad_volume.DBZ_LABELS + FE STORM4D_DBZ_COLORS.
DBZ_EDGES = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], dtype=np.float64)
DBZ_LABELS = [
    "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45",
    "45-50", "50-55", "55-60", "60-65", "65-70", "70+",
]


def parse_when(spec: str) -> datetime:
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_bounds(spec: str) -> tuple[float, float, float, float]:
    p = [float(x) for x in spec.split(",")]
    if len(p) != 4:
        sys.exit("--bounds must be min_lat,min_lon,max_lat,max_lon")
    return p[0], p[1], p[2], p[3]


def key_time(name: str) -> datetime | None:
    try:
        stamp = name.split("_")[-1].split(".")[0]  # <YYYYMMDD>-<HHMMSS>
        return datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def dbz_band_column(dbz: np.ndarray) -> pa.Array:
    idx = np.digitize(dbz, DBZ_EDGES).astype(np.int32)
    return pc.take(pa.array(DBZ_LABELS), pa.array(idx))


# ── S3 listing + download (anonymous, stdlib — from mrms_weather.py) ──────────
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


def frame_times(start: datetime, end: datetime, step_min: int) -> list[datetime]:
    out, t = [], start
    while t <= end:
        out.append(t)
        t += timedelta(minutes=step_min)
    return out


def product_key_map(product: str, start: datetime, end: datetime) -> list[tuple[datetime, str]]:
    """All (time, key) granules for one height product across the window's days."""
    days, d = [], start.replace(hour=0, minute=0, second=0, microsecond=0)
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    pairs: list[tuple[datetime, str]] = []
    for dd in days:
        for k in list_prefix(f"CONUS/{product}/{dd:%Y%m%d}/"):
            if not k.endswith(".grib2.gz"):
                continue
            t = key_time(Path(k).name)
            if t is not None and start - timedelta(minutes=10) <= t <= end + timedelta(minutes=10):
                pairs.append((t, k))
    pairs.sort()
    return pairs


def nearest(pairs: list[tuple[datetime, str]], when: datetime) -> str | None:
    if not pairs:
        return None
    return min(pairs, key=lambda tk: abs((tk[0] - when).total_seconds()))[1]


def fetch(key: str, cache: Path, retries: int = 3) -> Path | None:
    path = cache / Path(key).name
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urlopen(f"{BASE}/{key}", timeout=180) as r:
                data = r.read()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)
            return path
        except Exception:
            if attempt == retries - 1:
                return None
    return None


# ── grid read (gz → cfgrib), returns lon1d, lat1d ascending, z2d dBZ ──────────
def read_grid(path: Path, bounds, stride: int):
    import xarray as xr
    raw = gzip.decompress(path.read_bytes())
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    try:
        tmp.write(raw)
        tmp.close()
        ds = xr.open_dataset(tmp.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = list(ds.data_vars)[0]
        lat = ds["latitude"].values.astype("float64")
        lon = ds["longitude"].values.astype("float64")
        lon = ((lon + 180.0) % 360.0) - 180.0
        z = ds[var].values.astype("float32")
        ds.close()
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1, :]
    order = np.argsort(lon)
    lon, z = lon[order], z[:, order]
    if stride > 1:
        lon, lat, z = lon[::stride], lat[::stride], z[::stride, ::stride]
    min_lat, min_lon, max_lat, max_lon = bounds
    yi = (lat >= min_lat) & (lat <= max_lat)
    xi = (lon >= min_lon) & (lon <= max_lon)
    return lon[xi], lat[yi], z[np.ix_(yi, xi)]


# ── stratified LOD min-zoom (from nexrad_volume.lod_min_zoom) ─────────────────
def lod_min_zoom(lon, lat, alt, dbz, min_zoom, max_zoom, cell_deg_top, cell_alt_top):
    n = len(lon)
    alt = np.nan_to_num(alt.astype(np.float64), nan=0.0)
    dbz = np.nan_to_num(dbz.astype(np.float64), nan=-99.0)
    lon0, lat0, alt0 = lon.min(), lat.min(), alt.min()
    order = np.argsort(-dbz, kind="stable")
    min_z = np.full(n, max_zoom, dtype=np.int16)
    for z in range(min_zoom, max_zoom):
        scale = 2 ** (max_zoom - 1 - z)
        cd, ca = cell_deg_top * scale, cell_alt_top * scale
        cx = ((lon - lon0) / cd).astype(np.int64)
        cy = ((lat - lat0) / cd).astype(np.int64)
        cz = ((alt - alt0) / ca).astype(np.int64)
        code = (cx << 40) ^ (cy << 20) ^ cz
        _, first = np.unique(code[order], return_index=True)
        np.minimum.at(min_z, order[first], z)
    return min_z


def run_stt_build(parquet, out_dir, stt_build, min_zoom, max_zoom, publish, min_zoom_field):
    cmd = [
        stt_build, "--input", str(parquet), "--output", str(out_dir),
        "--time-field", "timestamp", "--time-format", "unix-ms",
        "--min-zoom", str(min_zoom), "--max-zoom", str(max_zoom),
        "--temporal-bucket", "5m", "--compression", "zstd",
        "--blob-ordering", "time-major",
        "--quantize-coords", "100", "--quantize-attrs-auto",
    ]
    if min_zoom_field:
        cmd += ["--min-zoom-field", min_zoom_field]
    if publish:
        cmd += ["--publish"]
    print("Running:", " ".join(cmd))
    import subprocess
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="window start, ISO-8601 UTC")
    ap.add_argument("--end", required=True, help="window end, ISO-8601 UTC")
    ap.add_argument("--bounds", default=",".join(str(x) for x in CONUS),
                    help="min_lat,min_lon,max_lat,max_lon (default CONUS)")
    ap.add_argument("--step-min", type=int, default=5, help="frame cadence (min)")
    ap.add_argument("--grid-km", type=float, default=2.0,
                    help="horizontal grid target (km); native ~1 km")
    ap.add_argument("--dbz-floor", type=float, default=10.0,
                    help="keep cells ≥ this dBZ (10 = every detectable echo)")
    ap.add_argument("--hmax-km", type=float, default=19.0, help="drop levels above this")
    ap.add_argument("--no-lod", action="store_true")
    ap.add_argument("--lod-cell-deg", type=float, default=0.02)
    ap.add_argument("--lod-cell-alt", type=float, default=1000.0)
    ap.add_argument("--min-zoom", type=int, default=2)
    ap.add_argument("--max-zoom", type=int, default=8)
    ap.add_argument("--cache", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "mrms3d")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12, help="parallel height fetch+decode")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = parse_when(args.start), parse_when(args.end)
    bounds = parse_bounds(args.bounds)
    heights = [h for h in HEIGHTS_KM if h <= args.hmax_km + 1e-6]
    args.cache.mkdir(parents=True, exist_ok=True)

    print(f"MRMS 3D volume  {start:%Y-%m-%d %H:%M}→{end:%H:%M}Z  bounds={bounds}")
    print(f"  {len(heights)} levels ≤{args.hmax_km}km  grid~{args.grid_km}km  "
          f"floor≥{args.dbz_floor:g}dBZ  cadence {args.step_min}min")

    frames = frame_times(start, end, args.step_min)
    print(f"  {len(frames)} frames; listing {len(heights)} height products…")
    t0 = time_mod.time()
    with ThreadPoolExecutor(max_workers=min(len(heights), 12)) as ex:
        maps = list(ex.map(lambda h: product_key_map(product_name(h), start, end), heights))
    hmaps = dict(zip([product_name(h) for h in heights], maps))

    # native grid stride: MRMS native ~0.01° ≈ 1 km → stride = round(grid_km)
    stride = max(1, int(round(args.grid_km)))

    def decode_one(args_tuple):
        h_km, when = args_tuple
        key = nearest(hmaps[product_name(h_km)], when)
        if not key:
            return None
        p = fetch(key, args.cache)
        if not p:
            return None
        lon, lat, z = read_grid(p, bounds, stride)
        yy, xx = np.nonzero(z >= args.dbz_floor)  # NaN >= floor is False
        if yy.size == 0:
            return None
        return (
            lon[xx].astype(np.float64),
            lat[yy].astype(np.float64),
            np.full(yy.size, h_km * 1000.0, dtype=np.float32),
            np.full(yy.size, int(when.timestamp() * 1000), dtype=np.int64),
            z[yy, xx].astype(np.float32),
        )

    jobs = [(h, f) for f in frames for h in heights]
    chunks = {"lon": [], "lat": [], "alt": [], "t": [], "dbz": []}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(decode_one, jobs):
            done += 1
            if done % 66 == 0 or done == len(jobs):
                pts = sum(len(c) for c in chunks["lon"])
                print(f"    {done}/{len(jobs)} height-frames  {pts:,} pts  "
                      f"{time_mod.time()-t0:.0f}s", flush=True)
            if res is None:
                continue
            for k, v in zip(("lon", "lat", "alt", "t", "dbz"), res):
                chunks[k].append(v)

    if not chunks["lon"]:
        sys.exit("No gates above floor in window — check bounds/time/floor.")
    g = {k: np.concatenate(v) for k, v in chunks.items()}
    order = np.argsort(g["t"], kind="stable")
    g = {k: v[order] for k, v in g.items()}
    n = len(g["lon"])
    print(f"\n  {n:,} 3D points total ({n/max(len(frames),1):,.0f}/frame avg)")

    cols = {
        "lon": pa.array(g["lon"], type=pa.float64()),
        "lat": pa.array(g["lat"], type=pa.float64()),
        "timestamp": pa.array(g["t"], type=pa.int64()),
        "alt_m": pa.array(g["alt"], type=pa.float32()),
        "dbz": pa.array(g["dbz"], type=pa.float32()),
        "dbz_band": dbz_band_column(g["dbz"]),
    }
    if not args.no_lod:
        mz = lod_min_zoom(g["lon"], g["lat"], g["alt"], g["dbz"],
                          args.min_zoom, args.max_zoom, args.lod_cell_deg, args.lod_cell_alt)
        cols["min_zoom"] = pa.array(mz, type=pa.int16())
        cum = np.cumsum(np.bincount(mz - args.min_zoom,
                                    minlength=args.max_zoom - args.min_zoom + 1))
        print("  LOD min-zoom pyramid (loaded at each zoom / % full):")
        for i, z in enumerate(range(args.min_zoom, args.max_zoom + 1)):
            print(f"    z{z}: {cum[i]:>11,}  ({100.0*cum[i]/max(n,1):5.1f}%)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vol_pq = args.out_dir / "mrms-storm3d-volume.parquet"
    pq.write_table(pa.table(cols), vol_pq, compression="snappy")
    print(f"  wrote {vol_pq} ({time_mod.time()-t0:.0f}s total)")

    if args.skip_build:
        print("  --skip-build: parquet only.")
        return 0
    run_stt_build(vol_pq, args.out_dir / "mrms-storm3d-volume", args.stt_build,
                  args.min_zoom, args.max_zoom, args.publish,
                  None if args.no_lod else "min_zoom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
