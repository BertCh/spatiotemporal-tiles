#!/usr/bin/env python3
"""NEXRAD Level II → CAPPI reflectivity ISO-LINES (storm4d-isolines).

The contour-sheet sibling of `nexrad_volume.py`: instead of one 3D point per
radar gate, the same KDMX volumes are gridded to a true CARTESIAN grid and
contoured at fixed reflectivity levels on every constant-altitude slice, so the
storm reads as a STACK of nested iso-line sheets terracing up through the
troposphere — the classic 3D contour plot, drawn from weather radar.

    per volume scan
      → Py-ART `grid_from_radars`  (polar sweeps → x/y/z Cartesian)
      → for each CAPPI level (1 km … 15 km MSL, `--alt-*`)
          → marching-squares contours at each dBZ level (`--dbz-levels`)
          → grid metres → lon/lat (Py-ART's own aeqd inverse, same projection)
          → simplify + drop specks
      → GeoParquet LineStrings → stt-build (paths)

Each contour is ONE feature riding at ONE altitude, which is exactly what
`AnimatedPathLayer.elevationProperty` wants (`alt_m` numeric column × the shared
`STORM4D_ELEVATION_SCALE`) — the whole ring lifts to its CAPPI height and the
nested rings terrace into a hill. Emitted columns:

    geometry       WKB LineString (lon/lat degrees, 2D — the tiler drops line-
                   vertex Z, so the height rides as the `alt_m` COLUMN)
    timestamp      volume-scan start (unix ms)
    end_timestamp  the NEXT scan's start + --fade-ms (see `scan_validity`)
    alt_m          f32, CAPPI level altitude in metres MSL
    dbz            f32, the contour's reflectivity level (also the GPU
                   `filterProperty` the demo's threshold slider drives)
    dbz_level      str, contract label of the level ("20","25",…,"60")
    alt_band       str, contract label of the CAPPI level ("1km","2km",…) —
                   the second render mode (color by HEIGHT instead of echo)

DATA POLICY. This is a DERIVED product (contours of a gridded field), exactly
like `storm4d-cloudtop`'s GOES isobands — the citable lossless base stays
`storm4d-volume` (every gate above the semantic dBZ floor). Gridding
resolution, contour levels, smoothing and simplify tolerance are presentation
knobs of the derived geometry, NOT thinning of source features.

Knobs (all measured in the smoke run before the full window):
  --crop-km        150   half-width of the Cartesian grid around Greenfield
  --grid-res-km    1.0   horizontal grid spacing (the 0.5° beam is already
                         ~1.3 km wide over Greenfield, so a finer grid would
                         invent detail the radar never resolved)
  --alt-min/max/step     CAPPI levels in metres MSL (default 1–15 km @ 1 km)
  --dbz-levels     20,25,…,60   contour levels (dBZ); the archive's numeric
                         `dbz` column is the same value, so the demo's GPU
                         threshold peels the stack down to the cores
  --weighting      nearest  gate→grid weighting (see process_volume)
  --smooth-cells   0.8   gaussian pre-smooth (grid cells) — kills marching-
                         squares staircase without moving the isopleth
  --simplify-deg   0.002 Douglas-Peucker tolerance (~200 m ≪ 1 km cell)
  --min-ring-km    1.5   drop specks smaller than this (bbox diagonal)

Pipeline: Level II (S3, cached by nexrad_volume.py) → Py-ART grid → contourpy
→ GeoParquet → stt-build. The raw-volume cache is SHARED with nexrad_volume.py
(`data/storm4d/nexrad/`), so with the volume archive already built this runs
`--skip-fetch` off local files.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import subprocess
import sys
import time as time_mod
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Shared with the volume generator: S3 listing/fetch, the storm reference point,
# the smoke window, and the time parser. One cache, one contract.
from nexrad_volume import (
    REF_LAT,
    REF_LON,
    SMOKE_END,
    SMOKE_START,
    download_all,
    gather_keys,
    parse_when,
)

# Contour levels → contract label. The FE colorMapping keys match these
# byte-for-byte (STORM4D_ISO_DBZ_COLORS in datasets.ts).
DEFAULT_DBZ_LEVELS = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

# Value written into grid cells the radar never sampled (above the top tilt,
# beyond range, inside the cone of silence). Far below every contour level, so
# a data void reads as "no echo" and the contour closes along the echo edge.
VOID_DBZ = -30.0

# Minimum vertices for a ring to survive (a triangle is a marching-squares
# artefact, not an isopleth).
MIN_RING_VERTICES = 5


def dbz_level_label(level: float) -> str:
    """Contract label for a reflectivity contour level (FE colorMapping key)."""
    return f"{level:g}"


def alt_band_label(alt_m: float) -> str:
    """Contract label for a CAPPI level (FE colorMapping key): "1km", "2km", …"""
    return f"{alt_m / 1000.0:g}km"


# ── per-volume worker ─────────────────────────────────────────────────────────
def process_volume(task: tuple) -> dict:
    """Grid one Level II volume to CAPPI slices and contour each one.

    Returns {"stats": …, "rows": [(wkb, t_ms, alt_m, level)]}. `end_timestamp`
    is NOT set here — it needs the NEXT volume's start time, which only the
    parent process knows (see `scan_validity`).
    """
    (path_str, crop_km, grid_res_km, alt_levels, dbz_levels, smooth_cells,
     simplify_deg, min_ring_km, dbz_floor, weighting) = task
    import warnings

    warnings.filterwarnings("ignore")
    import pyart
    from contourpy import LineType, contour_generator
    from pyart.core.transforms import cartesian_to_geographic_aeqd
    from scipy import ndimage as ndi
    from shapely import LineString, to_wkb

    t_start = time_mod.perf_counter()
    name = Path(path_str).name
    stats = dict(name=name, rings=0, vertices=0, dropped=0, error=None)
    try:
        radar = pyart.io.read_nexrad_archive(path_str)
    except Exception as e:  # truncated/corrupt volume: report, don't kill the run
        stats["error"] = f"{type(e).__name__}: {e}"
        stats["seconds"] = time_mod.perf_counter() - t_start
        return {"stats": stats, "rows": []}

    base_time = parse_when(radar.time["units"].split("since", 1)[1])
    t_ms = int((base_time + timedelta(seconds=float(radar.time["data"][0]))
                ).timestamp() * 1000)

    try:
        # Gates below the volume archive's semantic dBZ floor are "no
        # meteorological echo" — excluding them keeps the interpolation from
        # smearing noise into the analysed field (and makes gridding faster).
        gf = pyart.filters.GateFilter(radar)
        gf.exclude_transition()
        gf.exclude_masked("reflectivity")
        gf.exclude_invalid("reflectivity")
        gf.exclude_below("reflectivity", dbz_floor)

        crop_m = crop_km * 1000.0
        res_m = grid_res_km * 1000.0
        n_xy = int(round(2 * crop_m / res_m)) + 1
        nz = len(alt_levels)
        grid = pyart.map.grid_from_radars(
            (radar,),
            grid_shape=(nz, n_xy, n_xy),
            grid_limits=(
                (float(alt_levels[0]), float(alt_levels[-1])),
                (-crop_m, crop_m),
                (-crop_m, crop_m),
            ),
            # Grid origin = the STORM (Greenfield), altitude datum = MSL, so
            # `alt_m` here means the same thing it does in storm4d-volume
            # (Py-ART's gate altitudes are MSL too).
            grid_origin=(REF_LAT, REF_LON),
            grid_origin_alt=0.0,
            fields=["reflectivity"],
            gatefilters=(gf,),
            # `nearest` (not Py-ART's Barnes2 default) on purpose: Barnes
            # distance-weighting averages the beam's own ~1.3 km footprint a
            # SECOND time and clipped this storm's core from 66 dBZ to 62 —
            # the 60 dBZ isopleth all but vanished. Nearest keeps the peaks
            # (grid max 65.5 vs a 66.5 dBZ raw max), is ~2× faster, and the
            # blockiness it leaves is exactly what `--smooth-cells` is for.
            weighting_function=weighting,
            roi_func="dist_beam",
            min_radius=res_m,
            # Nothing meteorological above the top CAPPI level — cap the
            # gate scan there instead of Py-ART's 17 km default.
            toa=float(alt_levels[-1]) + res_m,
            map_roi=False,
        )
    except Exception as e:
        stats["error"] = f"grid: {type(e).__name__}: {e}"
        stats["seconds"] = time_mod.perf_counter() - t_start
        return {"stats": stats, "rows": []}

    xs = np.asarray(grid.x["data"], dtype=np.float64)
    ys = np.asarray(grid.y["data"], dtype=np.float64)
    field = np.ma.filled(
        np.ma.masked_invalid(grid.fields["reflectivity"]["data"]), VOID_DBZ
    ).astype(np.float64)

    # Speck threshold in grid units → the contour's own bbox diagonal in metres.
    min_ring_m = min_ring_km * 1000.0
    rows: list[tuple] = []
    for k, alt_m in enumerate(alt_levels):
        slab = field[k]
        if not np.isfinite(slab).any() or slab.max() < dbz_levels[0]:
            continue
        if smooth_cells > 0:
            # Smooth in the "no echo" floor as well, so the analysed edge moves
            # smoothly instead of stair-stepping along grid cells.
            slab = ndi.gaussian_filter(slab, sigma=smooth_cells, mode="nearest")
        cg = contour_generator(
            x=xs, y=ys, z=slab, name="serial", line_type=LineType.Separate,
            corner_mask=False,
        )
        for level in dbz_levels:
            if slab.max() < level:
                continue
            for ring in cg.lines(float(level)):
                if len(ring) < MIN_RING_VERTICES:
                    stats["dropped"] += 1
                    continue
                rx, ry = ring[:, 0], ring[:, 1]
                if max(np.ptp(rx), np.ptp(ry)) < min_ring_m:
                    stats["dropped"] += 1
                    continue
                lon, lat = cartesian_to_geographic_aeqd(rx, ry, REF_LON, REF_LAT)
                line = LineString(np.column_stack((lon, lat)))
                if simplify_deg > 0:
                    line = line.simplify(simplify_deg, preserve_topology=False)
                    if len(line.coords) < MIN_RING_VERTICES:
                        stats["dropped"] += 1
                        continue
                rows.append((
                    to_wkb(line), t_ms, float(alt_m), float(level),
                ))
                stats["rings"] += 1
                stats["vertices"] += len(line.coords)

    stats["seconds"] = time_mod.perf_counter() - t_start
    return {"stats": stats, "rows": rows}


def scan_validity(scan_times: np.ndarray, fade_ms: int,
                  max_hold_ms: int = 0) -> tuple[dict[int, int], int]:
    """Map each volume-scan start time → the `end_timestamp` its contours get.

    A contour must stay valid until its SUCCESSOR arrives, or the scene blinks
    out between scans. KDMX runs VCP 212 at ~6.9 min here (NOT the 5 min the
    temporal bucket suggests) and the cadence drifts, so the validity is
    derived from the actual next scan rather than a fixed pad:

        end = next_scan_start + fade_ms

    The `fade_ms` overhang is the WPC-fronts cross-dissolve trick: with the FE
    running `fadeInDuration == fadeOutDuration == fade_ms`, the outgoing scan
    ramps 1→0 over exactly the span its successor ramps 0→1, so the handoff
    holds constant alpha instead of dipping through a gap. `max_hold_ms`
    (default 2× the median cadence) caps how long a scan survives a DATA gap —
    a missing hour of volumes should read as missing, not as one frozen storm.

    Returns (end_by_start, median_cadence_ms).
    """
    scan_times = np.asarray(sorted(scan_times), dtype=np.int64)
    gaps = np.diff(scan_times)
    cadence = int(np.median(gaps)) if len(gaps) else 300_000
    hold = max_hold_ms if max_hold_ms > 0 else 2 * cadence
    ends: dict[int, int] = {}
    for i, t in enumerate(scan_times):
        nxt = int(scan_times[i + 1]) if i + 1 < len(scan_times) else int(t) + cadence
        ends[int(t)] = min(nxt, int(t) + hold) + fade_ms
    return ends, cadence


# ── stt-build (the AV iso-line build shape: windowed LineStrings) ─────────────
def run_stt_build(parquet: Path, out_dir: Path, stt_build: str,
                  min_zoom: int, max_zoom: int, bucket: str,
                  publish: bool) -> None:
    cmd = [
        stt_build,
        "--input", str(parquet),
        "--output", str(out_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        # Each contour is valid for its own scan window, so consecutive volumes
        # cross-dissolve instead of popping (same shape as storm4d-cloudtop).
        "--end-time-field", "end_timestamp",
        "--min-zoom", str(min_zoom),
        "--max-zoom", str(max_zoom),
        "--temporal-bucket", bucket,
        "--compression", "zstd",
        # Keep every ring WHOLE (a closed isopleth clipped at a tile edge would
        # be re-timed by the trajectory clipper's per-vertex interpolation —
        # the contour would animate as if it were being drawn). Same reason the
        # AV density iso-lines build with --no-clip.
        "--no-clip",
        # Multi-cell time PLAYBACK requires time-major ordering (blob-ordering
        # gotcha: `auto` can pick spatial and silently stall the playhead).
        "--blob-ordering", "time-major",
        # NOTE deliberately no --quantize-coords: quantizing multi-vertex
        # LineStrings mis-sizes PathLayer's instanced draw. Lines are cheap.
        # And no --simplify: it would distort the isopleths (we already
        # simplified in lon/lat space with a measured, sub-cell tolerance).
    ]
    if publish:
        cmd += ["--publish"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {out_dir}")


def parse_levels(spec: str) -> list[float]:
    try:
        levels = sorted({float(v) for v in spec.split(",") if v.strip()})
    except ValueError:
        sys.exit(f"--dbz-levels: not a comma-separated number list: {spec!r}")
    if not levels:
        sys.exit("--dbz-levels must list at least one level")
    return levels


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", help="window start, ISO-8601 UTC (e.g. 2024-05-21T17:30Z)")
    ap.add_argument("--end", help="window end, ISO-8601 UTC")
    ap.add_argument("--smoke", action="store_true",
                    help=f"pipeline-validation window {SMOKE_START} → {SMOKE_END}")
    ap.add_argument("--site", default="KDMX", help="radar site (default KDMX)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="directory receiving storm4d-isolines/ (and its .parquet)")
    ap.add_argument("--cache", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "storm4d" / "nexrad",
                    help="Level II cache — SHARED with nexrad_volume.py")
    # ── grid + contour knobs ─────────────────────────────────────────────────
    ap.add_argument("--crop-km", type=float, default=150.0)
    ap.add_argument("--grid-res-km", type=float, default=1.0)
    ap.add_argument("--alt-min", type=float, default=1000.0,
                    help="lowest CAPPI level, metres MSL")
    ap.add_argument("--alt-max", type=float, default=15000.0,
                    help="highest CAPPI level, metres MSL")
    ap.add_argument("--alt-step", type=float, default=1000.0,
                    help="CAPPI level spacing, metres")
    ap.add_argument("--dbz-levels", default=",".join(
        dbz_level_label(v) for v in DEFAULT_DBZ_LEVELS))
    ap.add_argument("--dbz-floor", type=float, default=10.0,
                    help="gates below this are excluded before gridding "
                         "(same semantic floor as storm4d-volume)")
    ap.add_argument("--weighting", default="nearest",
                    choices=["nearest", "Barnes2", "Cressman"],
                    help="Py-ART gate→grid weighting (see process_volume)")
    ap.add_argument("--smooth-cells", type=float, default=0.8)
    ap.add_argument("--simplify-deg", type=float, default=0.002)
    ap.add_argument("--min-ring-km", type=float, default=1.5)
    ap.add_argument("--fade-ms", type=int, default=90000,
                    help="cross-dissolve overhang past the next scan (see "
                         "scan_validity); the FE must use the SAME value for "
                         "fadeInDuration/fadeOutDuration")
    ap.add_argument("--max-hold-ms", type=int, default=0,
                    help="cap on how long one scan stays valid across a DATA "
                         "gap (0 = 2× the measured median cadence)")
    # ── build knobs ──────────────────────────────────────────────────────────
    ap.add_argument("--temporal-bucket", default="5m")
    # Whole-ring placement (--no-clip) means the zoom pyramid is pure
    # DUPLICATION — deeper levels add NO detail (a ring is never cut), they
    # only re-store every contour in a finer tile grid, and a ring whose
    # centroid tile falls outside a tight viewport would wink out. So the
    # pyramid stops at z7 (~236 km tiles here — the 300 km scene is 2×2 of
    # them); deck overzooms that level on the dive. Same reasoning as §9.4's
    # overlay-pyramid clamp.
    ap.add_argument("--min-zoom", type=int, default=5)
    ap.add_argument("--max-zoom", type=int, default=7)
    ap.add_argument("--workers", type=int, default=5,
                    help="volume-grid processes (each ~2 GB peak)")
    ap.add_argument("--skip-fetch", action="store_true", help="use cached volumes only")
    ap.add_argument("--skip-build", action="store_true", help="stop at parquet")
    ap.add_argument("--publish", action="store_true", help="zstd-19 deploy build")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    if args.smoke:
        start, end = parse_when(SMOKE_START), parse_when(SMOKE_END)
    elif args.start and args.end:
        start, end = parse_when(args.start), parse_when(args.end)
    else:
        sys.exit("Pass --start and --end (or --smoke).")
    if end <= start:
        sys.exit("--end must be after --start")

    alt_levels = list(np.arange(args.alt_min, args.alt_max + 1e-6, args.alt_step))
    if len(alt_levels) < 2:
        sys.exit("--alt-min/--alt-max/--alt-step must yield at least 2 CAPPI levels")
    dbz_levels = parse_levels(args.dbz_levels)

    print(f"Listing {args.site} volumes {start:%Y-%m-%dT%H:%M}Z → {end:%Y-%m-%dT%H:%M}Z …")
    keys = gather_keys(args.site, start, end)
    if not keys:
        sys.exit("No volumes listed for that window — check --start/--end/--site.")
    print(f"  {len(keys)} volume(s) in window")

    if args.skip_fetch:
        paths = sorted(p for p in (args.cache / Path(k).name for k in keys) if p.exists())
        print(f"Using {len(paths)} cached volume(s).")
    else:
        print(f"Downloading into {args.cache} …")
        paths = download_all(keys, args.cache, workers=8)
    if not paths:
        sys.exit("No volumes on disk — nothing to grid.")

    print(f"CAPPI levels: {len(alt_levels)} × {args.alt_step:g} m "
          f"({alt_levels[0]:g} → {alt_levels[-1]:g} m MSL)")
    print(f"Contour levels: {', '.join(dbz_level_label(v) for v in dbz_levels)} dBZ")
    print(f"Grid: {int(round(2 * args.crop_km / args.grid_res_km)) + 1}² × "
          f"{len(alt_levels)} @ {args.grid_res_km:g} km")

    tasks = [(str(p), args.crop_km, args.grid_res_km, alt_levels, dbz_levels,
              args.smooth_cells, args.simplify_deg, args.min_ring_km,
              args.dbz_floor, args.weighting) for p in paths]
    print(f"Gridding + contouring {len(tasks)} volume(s) on {args.workers} worker(s) …")
    t_all = time_mod.perf_counter()
    results = []
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=2) as pool:
        for res in pool.imap_unordered(process_volume, tasks):
            st = res["stats"]
            if st["error"]:
                print(f"  [{st['name']}] FAILED: {st['error']}")
            else:
                print(f"  [{st['name']}] {st['seconds']:6.1f}s  "
                      f"rings={st['rings']:>7,} vertices={st['vertices']:>9,} "
                      f"dropped={st['dropped']:>6,}")
            results.append(res)
    wall = time_mod.perf_counter() - t_all

    good = [r for r in results if r["stats"]["error"] is None]
    if not good:
        sys.exit("Every volume failed to grid.")
    rings = sum(r["stats"]["rings"] for r in good)
    verts = sum(r["stats"]["vertices"] for r in good)
    secs = [r["stats"]["seconds"] for r in good]
    print(f"\nContour funnel over {len(good)} volume(s):")
    print(f"  iso-line rings emitted: {rings:>12,}")
    print(f"  vertices:               {verts:>12,}  "
          f"(mean {verts / max(rings, 1):.1f}/ring)")
    print(f"  dropped specks:         {sum(r['stats']['dropped'] for r in good):>12,}")
    print(f"  per-volume: mean {np.mean(secs):.1f}s  max {np.max(secs):.1f}s  "
          f"(pool wall {wall:.1f}s)")
    if rings == 0:
        sys.exit("No contours produced — lower --dbz-levels or check the window.")

    # ── parquet ──────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = sorted((row for r in good for row in r["rows"]), key=lambda r: r[1])
    geom, t0, alt, lvl = (list(c) for c in zip(*all_rows))
    ends, cadence = scan_validity(np.unique(np.asarray(t0, dtype=np.int64)),
                                  args.fade_ms, args.max_hold_ms)
    t1 = [ends[int(t)] for t in t0]
    print(f"  scan cadence: median {cadence / 1000:.0f}s → each scan valid to "
          f"the next + {args.fade_ms / 1000:.0f}s cross-dissolve "
          f"(FE fadeIn/fadeOut MUST match)")
    alt_arr = np.asarray(alt, dtype=np.float32)
    lvl_arr = np.asarray(lvl, dtype=np.float32)
    table = pa.table({
        "geometry": pa.array(geom, type=pa.binary()),
        "timestamp": pa.array(t0, type=pa.int64()),
        "end_timestamp": pa.array(t1, type=pa.int64()),
        "alt_m": pa.array(alt_arr, type=pa.float32()),
        "dbz": pa.array(lvl_arr, type=pa.float32()),
        "dbz_level": pa.array([dbz_level_label(v) for v in lvl], type=pa.string()),
        "alt_band": pa.array([alt_band_label(v) for v in alt], type=pa.string()),
    })
    iso_pq = args.out_dir / "storm4d-isolines.parquet"
    pq.write_table(table, iso_pq, compression="snappy")
    print(f"\nWrote {table.num_rows:,} iso-line(s) → {iso_pq} "
          f"({iso_pq.stat().st_size / 1e6:.1f} MB parquet)")
    for level in dbz_levels:
        n = int((lvl_arr == np.float32(level)).sum())
        print(f"    {dbz_level_label(level):>3} dBZ: {n:>8,} rings")

    if args.skip_build:
        print("\nGeoParquet only (--skip-build).")
        return 0

    # Packs are content-addressed, so rebuilding INTO a populated directory
    # leaves the previous run's packs orphaned beside the new ones (the archive
    # reads fine; the directory is ~2× on disk and on any r2-sync).
    out_archive = args.out_dir / "storm4d-isolines"
    shutil.rmtree(out_archive, ignore_errors=True)
    run_stt_build(iso_pq, out_archive, args.stt_build,
                  min_zoom=args.min_zoom, max_zoom=args.max_zoom,
                  bucket=args.temporal_bucket, publish=args.publish)
    return 0


if __name__ == "__main__":
    sys.exit(main())
