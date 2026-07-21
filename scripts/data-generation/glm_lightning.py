#!/usr/bin/env python3
"""Lightning flash points from NOAA GOES GLM (Geostationary Lightning Mapper).

The lightning layer of the weather suite. Source: GOES-16/18/19 GLM **L2 LCFA**
(Lightning Cluster-Filter Algorithm) — one NetCDF granule every 20 seconds on
the anonymous NOAA S3 buckets, e.g.:

    https://noaa-goes16.s3.amazonaws.com/GLM-L2-LCFA/<YYYY>/<DDD>/<HH>/
        OR_GLM-L2-LCFA_G16_s<YYYYDDDHHMMSSf>_e<...>_c<...>.nc

(DDD = zero-padded day-of-year.) Each granule carries detected **flashes** with
a geodetic lon/lat, an optical energy, a footprint area, and a quality flag. We
list + download the granules for a time window (anonymous HTTPS, no boto3/s3fs —
same stdlib `urllib` pattern as cmorph_isobands.py), read each with xarray, keep
the good-quality flashes inside a bounding box, and emit one point per flash that
`stt-build` consumes directly:

    lon (Float64)   lat (Float64)   timestamp (Int64, flash time, unix ms)
    energy_fj (Float64, optical energy in femtojoules — friendlier than the
              raw ~1e-15 J)   area_km2 (Float64, flash footprint)

One archive drives BOTH demo layers: AnimatedPointLayer (individual flashes,
`wakeLength` bright-then-fade) and AnimatedHeatmapLayer (live strike density).

Raw granules are cached under <cache>/<G>/<YYYY>/<DDD>/<HH>/ (resumable: re-runs
skip files already on disk). Use --skip-build to stop at the parquet, or
--stt-build /path/to/stt-build to point at the binary.
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

# GLM is flown on GOES-East (G16, 75.2°W) and GOES-West (G18, 137°W); G19 became
# GOES-East operational in 2025. Pick with --satellite; G16 covers CONUS best for
# events through early 2025.
SAT_BUCKET = {
    "goes16": "noaa-goes16",
    "goes18": "noaa-goes18",
    "goes19": "noaa-goes19",
}
SAT_TAG = {"goes16": "G16", "goes18": "G18", "goes19": "G19"}
PRODUCT = "GLM-L2-LCFA"

# CONUS clip box (min_lat, min_lon, max_lat, max_lon).
CONUS = (24.0, -125.0, 50.0, -66.0)

# GOES filename scan-start token: s<YYYY><DDD><HH><MM><SS><f> (f = tenths).
_S_TOKEN = re.compile(r"_s(\d{14})_")


def s3_base(satellite: str) -> str:
    return f"https://{SAT_BUCKET[satellite]}.s3.amazonaws.com"


def parse_bounds(spec: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 4:
        sys.exit("--bounds must be min_lat,min_lon,max_lat,max_lon")
    return parts[0], parts[1], parts[2], parts[3]


def parse_when(spec: str) -> datetime:
    """Parse an ISO-8601 UTC instant (with or without trailing Z)."""
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def scan_start(name: str) -> datetime | None:
    """Extract the scan-start time encoded in a GLM granule filename."""
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
    return tag.rsplit("}", 1)[-1]  # strip the {namespace}


def list_prefix(base: str, prefix: str, retries: int = 4) -> list[str]:
    """Return every object key under `prefix` (follows continuation tokens)."""
    keys: list[str] = []
    token: str | None = None
    while True:
        url = f"{base}/?list-type=2&prefix={prefix}"
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
                    return keys  # give up on this prefix; treat as empty
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
    """GLM prefixes GLM-L2-LCFA/<YYYY>/<DDD>/<HH>/ spanning [start, end]."""
    out: list[str] = []
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur <= end:
        doy = cur.timetuple().tm_yday
        out.append(f"{PRODUCT}/{cur:%Y}/{doy:03d}/{cur:%H}/")
        cur += timedelta(hours=1)
    return out


def gather_keys(base: str, start: datetime, end: datetime, stride: int) -> list[str]:
    """List granules across the window, filter to [start, end], apply stride."""
    prefixes = hour_prefixes(start, end)
    print(f"Listing {len(prefixes)} hour-prefix(es) on {base} …")
    keys: list[str] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for got in ex.map(lambda p: list_prefix(base, p), prefixes):
            keys.extend(got)
    in_window = []
    for k in keys:
        t = scan_start(Path(k).name)
        if t is not None and start <= t <= end:
            in_window.append(k)
    in_window.sort()  # filename order == chronological
    if stride > 1:
        in_window = in_window[::stride]
    print(f"  {len(in_window)} granule(s) in window (stride {stride})")
    return in_window


# ── download (resumable, atomic) ──────────────────────────────────────────────
def fetch(base: str, key: str, cache: Path, retries: int = 3) -> Path | None:
    path = cache / key
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base}/{key}"
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


def download_all(base: str, keys: list[str], cache: Path, workers: int) -> list[Path]:
    paths: list[Path] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, base, k, cache): k for k in keys}
        for f in as_completed(futs):
            p = f.result()
            if p is not None:
                paths.append(p)
            done += 1
            if done % 200 == 0 or done == len(keys):
                print(f"  fetched {done}/{len(keys)} granule(s)")
    paths.sort()
    return paths


# ── flash extraction ──────────────────────────────────────────────────────────
def read_flashes(path: Path, bounds, min_quality: bool):
    """Return (lon, lat, t_ms, energy_fj, area_km2) arrays for one granule."""
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
    except Exception:
        return None
    try:
        if "flash_lat" not in ds or ds.sizes.get("number_of_flashes", 0) == 0:
            return None
        lat = ds["flash_lat"].values.astype("float64")
        lon = ds["flash_lon"].values.astype("float64")
        # flash_time_offset_of_first_event is CF "seconds since <epoch>" and is
        # decoded to datetime64 by xarray → straight to unix-ms.
        t = ds["flash_time_offset_of_first_event"].values
        t_ms = t.astype("datetime64[ms]").astype("int64")
        energy = ds["flash_energy"].values.astype("float64") if "flash_energy" in ds else np.zeros_like(lat)
        area = ds["flash_area"].values.astype("float64") if "flash_area" in ds else np.zeros_like(lat)
        qc = ds["flash_quality_flag"].values.astype("int64") if "flash_quality_flag" in ds else np.zeros_like(lat, dtype="int64")
    finally:
        ds.close()

    min_lat, min_lon, max_lat, max_lon = bounds
    keep = (
        (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
        & np.isfinite(lat) & np.isfinite(lon)
    )
    if min_quality:
        keep &= qc == 0  # 0 = good_quality_qf
    if not keep.any():
        return None
    return (
        lon[keep],
        lat[keep],
        t_ms[keep],
        energy[keep] * 1e15,   # J → femtojoules
        area[keep] / 1e6,      # m² → km²
    )


def build_parquet(paths: list[Path], bounds, min_quality: bool, out: Path) -> int:
    lon_a, lat_a, t_a, e_a, ar_a = [], [], [], [], []
    for i, p in enumerate(paths):
        got = read_flashes(p, bounds, min_quality)
        if got is not None:
            lon, lat, t_ms, ener, area = got
            lon_a.append(lon)
            lat_a.append(lat)
            t_a.append(t_ms)
            e_a.append(ener)
            ar_a.append(area)
        if (i + 1) % 500 == 0 or i + 1 == len(paths):
            flat = sum(len(x) for x in t_a)
            print(f"  read {i + 1}/{len(paths)} granule(s)  ({flat} flashes)")
    if not t_a:
        sys.exit("No flashes found in window/bounds — nothing to build.")

    lon = np.concatenate(lon_a)
    lat = np.concatenate(lat_a)
    t_ms = np.concatenate(t_a)
    ener = np.concatenate(e_a)
    area = np.concatenate(ar_a)

    # Sort by time so the archive lays down chronologically (also helps the
    # time-major blob ordering downstream).
    order = np.argsort(t_ms, kind="stable")
    table = pa.table(
        {
            "lon": pa.array(lon[order], type=pa.float64()),
            "lat": pa.array(lat[order], type=pa.float64()),
            "timestamp": pa.array(t_ms[order], type=pa.int64()),
            "energy_fj": pa.array(ener[order], type=pa.float64()),
            "area_km2": pa.array(area[order], type=pa.float64()),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy")
    span0 = datetime.fromtimestamp(int(t_ms.min()) / 1000, tz=timezone.utc)
    span1 = datetime.fromtimestamp(int(t_ms.max()) / 1000, tz=timezone.utc)
    print(f"\nWrote {table.num_rows} flashes → {out}")
    print(f"  time span: {span0:%Y-%m-%d %H:%M} … {span1:%Y-%m-%d %H:%M} UTC")
    return table.num_rows


# ── stt-build ─────────────────────────────────────────────────────────────────
def run_stt_build(parquet: Path, stt_dir: Path, stt_build: str, args) -> None:
    import subprocess

    cmd = [
        stt_build,
        "--input", str(parquet),
        "--output", str(stt_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", "0",
        "--max-zoom", str(args.max_zoom),
        "--temporal-bucket", args.temporal_bucket,
        "--compression", "zstd",
        # Multi-cell time PLAYBACK requires time-major ordering — `auto` can pick
        # spatial and silently stall the playhead (blob-ordering gotcha).
        "--blob-ordering", "time-major",
        "--quantize-coords", "100",
        "--quantize-attrs-auto",
    ]
    if args.summary:
        # Continental zoom-out density LOD: H3 hexes carrying flash count + energy.
        cmd += ["--summary-tier", "h3", "--summary-columns", "count,energy_fj:sum"]
    if args.publish:
        cmd += ["--publish"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {stt_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", required=True, help="window start, ISO-8601 UTC (e.g. 2024-05-07T18:00Z)")
    ap.add_argument("--end", required=True, help="window end, ISO-8601 UTC")
    ap.add_argument("--satellite", choices=list(SAT_BUCKET), default="goes16")
    ap.add_argument("--bounds", default=",".join(str(x) for x in CONUS),
                    help="min_lat,min_lon,max_lat,max_lon (default CONUS)")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth granule (1 = every 20 s scan)")
    ap.add_argument("--all-quality", action="store_true",
                    help="keep all flashes (default keeps quality_flag==0 only)")
    ap.add_argument("--cache", type=Path, default=Path("data/glm"))
    ap.add_argument("--out", type=Path, required=True,
                    help="output .stt directory (or .parquet with --skip-build)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-zoom", type=int, default=7)
    # 15m, NOT the granule cadence: flashes carry exact per-feature timestamps
    # regardless of bucket size, and 1m buckets shattered the archive into
    # ~287k ~1KB tiles — a 30-min render window then kept ~380 tiles resident
    # and churned hundreds of loads/sec at playback speed (2-6 fps).
    ap.add_argument("--temporal-bucket", default="15m")
    ap.add_argument("--summary", action="store_true",
                    help="also emit an H3 summary tier (continental density LOD)")
    ap.add_argument("--publish", action="store_true", help="zstd-19 deploy build")
    ap.add_argument("--skip-fetch", action="store_true", help="use cached granules only")
    ap.add_argument("--skip-build", action="store_true", help="stop at parquet")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = parse_when(args.start), parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    bounds = parse_bounds(args.bounds)
    base = s3_base(args.satellite)
    sat_cache = args.cache / SAT_TAG[args.satellite]

    keys = gather_keys(base, start, end, args.stride)
    if not keys:
        sys.exit("No granules listed for that window — check --start/--end/--satellite.")

    if args.skip_fetch:
        paths = sorted(p for p in (sat_cache / k for k in keys) if p.exists())
        print(f"Using {len(paths)} cached granule(s).")
    else:
        print(f"Downloading into {sat_cache} …")
        paths = download_all(base, keys, sat_cache, args.workers)
    if not paths:
        sys.exit("No granules on disk — nothing to read.")

    pq_path = args.out if args.out.suffix == ".parquet" else args.out.with_suffix(".parquet")
    n = build_parquet(paths, bounds, not args.all_quality, pq_path)
    if n == 0:
        sys.exit("No flashes after filtering.")

    if args.skip_build or args.out.suffix == ".parquet":
        stt_dir = args.out.with_suffix("")
        print(f"\nGeoParquet only. Build with:\n"
              f"  {args.stt_build} --input {pq_path} --output {stt_dir} "
              f"--time-field timestamp --time-format unix-ms --min-zoom 0 "
              f"--max-zoom {args.max_zoom} --temporal-bucket {args.temporal_bucket} "
              f"--blob-ordering time-major --quantize-coords 100 --quantize-attrs-auto")
        return 0

    run_stt_build(pq_path, args.out.with_suffix(""), args.stt_build, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
