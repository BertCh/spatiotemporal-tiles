#!/usr/bin/env python3
"""Storm-based warning polygon phases for the Storm-4D Greenfield demo.

`storm4d-warnings` archive (storm-4d-greenfield campaign, contract §9.1):
POLYGONS, 1-minute bucket, zoom 3-9. One feature per SBW *phase* — every SVS
follow-up that shrinks the polygon is its own feature with its own time span,
so the prism visibly tightens around the storm as the warning is trimmed.

Source: IEM `watchwarn.py` shapefile service with `limit1=yes` (storm-based
polygons only, GTYPE=P) and `addsvs=yes` (SVS shrink phases included):

    https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py
        ?accept=shapefile&sts=...&ets=...&limit1=yes&addsvs=yes

Each DBF record carries `POLY_BEG`/`POLY_END` (YYYYMMDDHHMM UTC — the span this
exact polygon was in force), `PHENOM`, `SIG`, `ETN`, `STATUS` (NEW/CON/EXT/EXP).
NOTE the REST alternative `/api/1/vtec/sbw_interval.geojson` returns only the
NEW-issue polygon per event (no SVS phases) — do not switch back to it.

We keep PHENOM in {TO, SV, FF} with SIG=W (per contract), drop zero-length
phases, validate rings with shapely (make_valid), explode multi-part geometries
into individual Polygons so they tile cleanly, and keep any polygon that
intersects the scene bbox. Output GeoParquet → `stt-build`:

    geometry (WKB Polygon)   timestamp (Int64 unix ms, phase start)
    end_timestamp (Int64 unix ms, phase end)   phenom (str: TO|SV|FF)
    etn (str, VTEC event tracking number)

The raw zip is cached under --cache (idempotent: re-runs skip the download).
Use --skip-build to stop at the parquet.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq
import shapefile
from shapely import to_wkb
from shapely.geometry import box as shp_box
from shapely.geometry import shape as shp_shape
from shapely.validation import make_valid

IEM_WATCHWARN = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"

# Scene bbox (min_lon, min_lat, max_lon, max_lat) — Iowa + border states,
# matches the storm4d-reports crop.
DEFAULT_BBOX = (-98.0, 39.0, -90.0, 44.0)

# Contract §9.1: TO | SV | FF warnings only (SIG=W).
KEEP_PHENOM = ("TO", "SV", "FF")


def parse_when(spec: str) -> datetime:
    """Parse an ISO-8601 UTC instant (with or without trailing Z)."""
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_bbox(spec: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 4:
        sys.exit("--bbox must be min_lon,min_lat,max_lon,max_lat")
    return parts[0], parts[1], parts[2], parts[3]


def poly_time_ms(tok: str) -> int:
    """`YYYYMMDDHHMM` (UTC) DBF token → unix ms."""
    dt = datetime.strptime(tok, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ── fetch (cached, atomic) ────────────────────────────────────────────────────
def fetch_zip(start: datetime, end: datetime, cache: Path) -> Path:
    path = cache / f"wwa_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.zip"
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached {path}")
        return path
    url = (
        f"{IEM_WATCHWARN}?accept=shapefile&sts={start:%Y-%m-%dT%H:%M}Z"
        f"&ets={end:%Y-%m-%dT%H:%M}Z&limit1=yes&addsvs=yes"
    )
    print(f"Fetching {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=300) as r:
        data = r.read()
    tmp = path.with_suffix(".zip.tmp")
    tmp.write_bytes(data)
    tmp.rename(path)
    print(f"  {len(data)} bytes → {path}")
    return path


def extract_shp(zip_path: Path) -> Path:
    out_dir = zip_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    shps = sorted(out_dir.glob("*.shp"))
    if not shps:
        sys.exit(f"No .shp inside {zip_path}")
    return shps[0]


# ── phase extraction ──────────────────────────────────────────────────────────
def read_phases(shp_path: Path, bbox, start: datetime, end: datetime):
    """Return rows (wkb, t_ms, et_ms, phenom, etn) — one per Polygon per phase."""
    clip = shp_box(*bbox)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    sf = shapefile.Reader(str(shp_path))
    names = [f[0] for f in sf.fields[1:]]
    rows = []
    n_dropped_zero = 0
    n_fixed = 0
    status_counter: Counter[str] = Counter()
    for sr in sf.iterShapeRecords():
        rec = dict(zip(names, sr.record))
        if rec.get("GTYPE") != "P" or rec.get("SIG") != "W":
            continue
        phenom = rec.get("PHENOM")
        if phenom not in KEEP_PHENOM:
            continue
        t_ms = poly_time_ms(rec["POLY_BEG"])
        et_ms = poly_time_ms(rec["POLY_END"])
        if et_ms <= t_ms:
            n_dropped_zero += 1  # EXP bookkeeping rows etc.
            continue
        if et_ms <= start_ms or t_ms >= end_ms:
            continue  # phase entirely outside the demo window
        geom = shp_shape(sr.shape.__geo_interface__)
        if not geom.is_valid:
            geom = make_valid(geom)
            n_fixed += 1
        # Explode multi-part / collection results into individual Polygons.
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        etn = str(rec["ETN"])
        kept_any = False
        for part in parts:
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            if not part.intersects(clip):
                continue
            rows.append((to_wkb(part), t_ms, et_ms, phenom, etn))
            kept_any = True
        if kept_any:
            status_counter[rec.get("STATUS", "?")] += 1
    print(
        f"  kept {len(rows)} polygon phase(s) "
        f"(status {dict(status_counter)}; dropped {n_dropped_zero} zero-length, "
        f"repaired {n_fixed} invalid ring(s))"
    )
    return rows


def build_parquet(rows, out: Path) -> int:
    if not rows:
        sys.exit("No warning phases in window/bbox — nothing to build.")
    rows.sort(key=lambda r: (r[1], r[3], r[4]))
    table = pa.table(
        {
            "geometry": pa.array([r[0] for r in rows], type=pa.binary()),
            "timestamp": pa.array([r[1] for r in rows], type=pa.int64()),
            "end_timestamp": pa.array([r[2] for r in rows], type=pa.int64()),
            "phenom": pa.array([r[3] for r in rows], type=pa.string()),
            "etn": pa.array([r[4] for r in rows], type=pa.string()),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy")
    t0 = datetime.fromtimestamp(rows[0][1] / 1000, tz=timezone.utc)
    t1 = datetime.fromtimestamp(max(r[2] for r in rows) / 1000, tz=timezone.utc)
    by_phenom = Counter(r[3] for r in rows)
    print(f"\nWrote {table.num_rows} phase polygon(s) → {out}")
    print(f"  by phenom: {dict(by_phenom)}")
    print(f"  time span: {t0:%Y-%m-%d %H:%M} … {t1:%Y-%m-%d %H:%M} UTC")
    return table.num_rows


# ── stt-build ─────────────────────────────────────────────────────────────────
def run_stt_build(parquet: Path, stt_dir: Path, args) -> None:
    cmd = [
        args.stt_build,
        "--input", str(parquet),
        "--output", str(stt_dir),
        "--time-field", "timestamp",
        "--end-time-field", "end_timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", str(args.min_zoom),
        "--max-zoom", str(args.max_zoom),
        "--temporal-bucket", args.temporal_bucket,
        "--compression", "zstd",
        # Playback-safe ordering (blob-ordering gotcha: `auto` may pick spatial).
        "--blob-ordering", "time-major",
    ]
    if args.publish:
        cmd += ["--publish"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {stt_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", default="2024-05-21T17:30Z",
                    help="window start, ISO-8601 UTC (contract §9.0 default)")
    ap.add_argument("--end", default="2024-05-22T03:00Z",
                    help="window end, ISO-8601 UTC")
    ap.add_argument("--bbox", default=",".join(str(x) for x in DEFAULT_BBOX),
                    help="scene crop min_lon,min_lat,max_lon,max_lat")
    ap.add_argument("--cache", type=Path, default=Path("data/storm4d/warnings"))
    ap.add_argument("--out", type=Path, required=True,
                    help="output .stt directory (or .parquet with --skip-build)")
    ap.add_argument("--min-zoom", type=int, default=3)
    ap.add_argument("--max-zoom", type=int, default=9)
    ap.add_argument("--temporal-bucket", default="1m",
                    help="event-driven phases; contract says 1m")
    ap.add_argument("--publish", action="store_true", help="zstd-19 deploy build")
    ap.add_argument("--skip-build", action="store_true", help="stop at parquet")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = parse_when(args.start), parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    bbox = parse_bbox(args.bbox)

    zip_path = fetch_zip(start, end, args.cache)
    shp_path = extract_shp(zip_path)
    print(f"Reading phases from {shp_path.name} …")
    rows = read_phases(shp_path, bbox, start, end)

    pq_path = args.out if args.out.suffix == ".parquet" else args.out.with_suffix(".parquet")
    build_parquet(rows, pq_path)

    if args.skip_build or args.out.suffix == ".parquet":
        print(f"\nGeoParquet only. Build with:\n"
              f"  {args.stt_build} --input {pq_path} --output {args.out.with_suffix('')} "
              f"--time-field timestamp --end-time-field end_timestamp "
              f"--time-format unix-ms --min-zoom {args.min_zoom} "
              f"--max-zoom {args.max_zoom} --temporal-bucket {args.temporal_bucket} "
              f"--blob-ordering time-major")
        return 0

    run_stt_build(pq_path, args.out.with_suffix(""), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
