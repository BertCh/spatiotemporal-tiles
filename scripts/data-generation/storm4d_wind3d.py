#!/usr/bin/env python3
"""Multi-level HRRR wind particles for the Storm-4D Greenfield demo → STT trips.

`storm4d-wind3d` (§9.1 of docs/roadmap/storm-4d-greenfield-2026-07.md): the
storm environment as FOUR stacked particle decks threading the radar volume —
850 / 700 / 500 / 250 mb, each advected independently through its own
time-varying HRRR pressure-level wind and emitted as trips with per-vertex
timestamps + wind speed. Each feature carries `level_mb` (i32) and
`level_alt_m` (f32, standard-atmosphere height of the level: 1457 / 3012 /
5574 / 10363 m); the FE lifts each deck with `elevationProperty:'level_alt_m'`
under the shared STORM4D_ELEVATION_SCALE.

This is a thin multi-level driver over `hrrr_advect.py` — the HRRR `.idx`
byte-range fetch (UGRD/VGRD only, ~1-5 MB/cycle/level), LCC→lat/lon regrid,
regional bilinear sampler and RK4 advection core are IMPORTED from it
unchanged; only the seeding bbox, the per-level loop and the parquet schema
(level columns) live here.

Knobs (defaults = the ratified §9.1 build):
- window 2024-05-21 17:00Z → 05-22 03:00Z (HRRR f00 analyses, 11 cycles)
- seed bbox [-102, 37, -87, 46] (lon/lat; also the regrid domain — particles
  leaving it retire and respawn, exactly like the CONUS build)
- ~2,500 particles per level, 15-min vertex cadence (RK4 ×4 substeps),
  4 h lifetime
- stt-build: trips, `--temporal-bucket 2h`, zoom 3-6, time-major, --simplify

Pipeline:  HRRR GRIB2 (.idx subset)  →  regrid  →  RK4 advection ×4 levels
           →  GeoParquet  →  stt-build

Full build (exactly the §9.1 archive):

    venv/bin/python storm4d_wind3d.py \
        --out ../../examples/showcase/public/data/storm4d-wind3d

Use --skip-build to stop at the GeoParquet; --levels to subset for smoke runs.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import hrrr_advect as ha

# §9.1: level → standard-atmosphere altitude (m) baked per feature.
LEVEL_ALT_M = {850: 1457.0, 700: 3012.0, 500: 5574.0, 250: 10363.0}

# Seed/regrid bbox around the Greenfield storm corridor, §9.1 order lon/lat →
# hrrr_advect bounds order (min_lat, min_lon, max_lat, max_lon).
STORM4D_BOUNDS = (37.0, -102.0, 46.0, -87.0)


def write_geoparquet(level_tracks, out_path: Path) -> int:
    """Trips GeoParquet: hrrr_advect vertex schema + per-feature level cols."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import wkb
    from shapely.geometry import LineString

    geom, ts, ets, vts, vvals = [], [], [], [], []
    level_mb, level_alt, speed = [], [], []
    for mb, alt_m, tracks in level_tracks:
        for tr in tracks:
            coords = list(zip(tr["lon"], tr["lat"]))
            if len(coords) < 2:
                continue
            geom.append(wkb.dumps(LineString(coords)))
            t = [int(x) for x in tr["t"]]
            ts.append(t[0])
            ets.append(t[-1])
            vts.append(t)
            sp = [float(x) for x in tr["v"]]
            vvals.append([np.float32(x) for x in sp])
            finite = [x for x in sp if x == x]
            speed.append(float(np.mean(finite)) if finite else float("nan"))
            level_mb.append(mb)
            level_alt.append(alt_m)
    table = pa.table(
        {
            "geometry": pa.array(geom, type=pa.binary()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "end_timestamp": pa.array(ets, type=pa.int64()),
            "vertex_timestamps": pa.array(vts, type=pa.list_(pa.int64())),
            "vertex_values": pa.array(vvals, type=pa.list_(pa.float32())),
            "level_mb": pa.array(level_mb, type=pa.int32()),
            "level_alt_m": pa.array(level_alt, type=pa.float32()),
            "speed": pa.array(speed, type=pa.float64()),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"Wrote {table.num_rows} particle trips → {out_path}")
    return table.num_rows


def run_stt_build(parquet: Path, stt_dir: Path, stt_build: str,
                  min_zoom: int, max_zoom: int, bucket: str) -> None:
    cmd = [
        stt_build, "--input", str(parquet), "--output", str(stt_dir),
        "--time-field", "timestamp", "--end-time-field", "end_timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", str(min_zoom), "--max-zoom", str(max_zoom),
        "--temporal-bucket", bucket, "--blob-ordering", "time-major",
        "--compression", "zstd", "--simplify",
    ]
    print("Running:", " ".join(cmd))
    import subprocess

    subprocess.run(cmd, check=True)
    print(f"Built {stt_dir}")


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-05-21T17:00Z", help="ISO-8601 UTC")
    ap.add_argument("--end", default="2024-05-22T03:00Z", help="ISO-8601 UTC")
    ap.add_argument("--bounds", default=",".join(str(x) for x in STORM4D_BOUNDS),
                    help="min_lat,min_lon,max_lat,max_lon (hrrr_advect order)")
    ap.add_argument("--levels", default="850,700,500,250",
                    help="comma list of pressure levels (mb) to advect")
    ap.add_argument("--particles", type=int, default=2500, help="per level")
    ap.add_argument("--grid-step", type=float, default=0.08)
    ap.add_argument("--native-stride", type=int, default=3)
    ap.add_argument("--step-hours", type=float, default=0.25,
                    help="output vertex cadence (h)")
    ap.add_argument("--substeps", type=int, default=4)
    # Shorter than the continental build's 8 h: the window is only 10 h and the
    # 250 mb jet crosses the 15° bbox in ~5 h — 4 h keeps tracks turning over.
    ap.add_argument("--lifetime-hours", type=float, default=4.0)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path,
                    default=here / "data" / "storm4d" / "hrrr")
    ap.add_argument("--out", type=Path, required=True,
                    help="output packed dataset dir (or .parquet to stop there)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-zoom", type=int, default=3)
    ap.add_argument("--max-zoom", type=int, default=6)
    ap.add_argument("--temporal-bucket", default="2h")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--stt-build", default=str(
        here.parents[1] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = ha.parse_when(args.start), ha.parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    bounds = ha.parse_bounds(args.bounds)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    bad = [mb for mb in levels if mb not in LEVEL_ALT_M]
    if bad:
        sys.exit(f"No level_alt_m mapping for {bad}; known: {sorted(LEVEL_ALT_M)}")

    adv = Namespace(particles=args.particles, step_hours=args.step_hours,
                    substeps=args.substeps, lifetime_hours=args.lifetime_hours,
                    min_points=args.min_points)
    level_tracks = []
    for mb in levels:
        print(f"\n── {mb} mb (level_alt_m={LEVEL_ALT_M[mb]:.0f}) " + "─" * 40)
        # Per-level rng → deterministic AND decorrelated seeding across decks.
        rng = np.random.default_rng(args.seed + mb)
        times_ms, lats, lons, U, V, _ = ha.load_hrrr_wind(
            start, end, bounds, args.grid_step, args.cache, args.workers,
            args.skip_fetch, args.native_stride, f"{mb}mb")
        tracks = ha.advect(times_ms, lats, lons, U, V, adv, rng, bounds)
        level_tracks.append((mb, LEVEL_ALT_M[mb], tracks))

    if args.out.suffix == ".parquet":
        pq_path = args.out
    else:
        args.cache.mkdir(parents=True, exist_ok=True)
        pq_path = args.cache / (args.out.name + ".parquet")
    write_geoparquet(level_tracks, pq_path)
    if args.skip_build or args.out.suffix == ".parquet":
        print("GeoParquet only (--skip-build).")
        return 0
    run_stt_build(pq_path, args.out.with_suffix(""), args.stt_build,
                  args.min_zoom, args.max_zoom, args.temporal_bucket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
