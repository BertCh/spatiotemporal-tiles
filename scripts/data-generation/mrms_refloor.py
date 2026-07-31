#!/usr/bin/env python3
"""Re-floor a master MRMS 3D parquet and RECOMPUTE its stratified LOD.

`mrms_volume.py` decodes once at the lowest dBZ floor worth keeping, which for a
national 9.5 h window is far more points than `stt-build` can tile. This trims
that master to a shippable floor without re-decoding a single GRIB granule
(the decode is ~45 min; this is ~1 min).

WHY THIS IS NOT JUST A `WHERE dbz >= floor`: `min_zoom` is a *stratified LOD*
column — for each 4D (lon, lat, alt, time-bucket) cell it marks the strongest
gate as the one that represents that cell at coarse zooms. Filtering rows
afterwards keeps whichever representatives happen to clear the new floor and
silently drops the rest, so the coarse tiers come out sparser than the ladder
claims — exactly the failure documented in storm-4d-greenfield-2026-07.md §11.4,
where a mis-keyed LOD left the framing zoom showing a median 13% of the visible
bucket. So the LOD is recomputed over the SURVIVING points.

Frame-by-frame, because `--temporal-bucket` equals the frame cadence and the LOD
cell carries the bucket: each frame already thins against itself, so with a
pinned grid corner the per-frame result equals the whole-window one. That keeps
peak memory at one frame regardless of window length.
"""
from __future__ import annotations

import argparse
import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrms_volume import CONUS, LOD_BUCKET_MS, dbz_band_column, lod_min_zoom  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="master parquet")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dbz-floor", type=float, required=True)
    ap.add_argument("--step-min", type=int, default=0,
                    help="keep only frames on this cadence (0 = keep all)")
    ap.add_argument("--bounds", default=",".join(str(x) for x in CONUS))
    ap.add_argument("--lod-cell-deg", type=float, default=0.0085)
    ap.add_argument("--lod-cell-alt", type=float, default=60.0)
    ap.add_argument("--min-zoom", type=int, default=2)
    ap.add_argument("--max-zoom", type=int, default=8)
    ap.add_argument("--histogram", action="store_true",
                    help="report survivor counts per candidate floor and exit")
    args = ap.parse_args()

    b = [float(x) for x in args.bounds.split(",")]
    lod_origins = (b[1], b[0], 0.0)
    n_zooms = args.max_zoom - args.min_zoom + 1
    pf = pq.ParquetFile(args.input)
    t0 = time_mod.time()
    print(f"{args.input}  {pf.metadata.num_rows:,} rows  "
          f"{pf.metadata.num_row_groups} row groups")

    if args.histogram:
        # One pass, dbz column only — cheap enough to answer "which floor lands
        # under the tiler's ceiling" exactly instead of extrapolating.
        edges = [10, 15, 20, 25, 27, 30, 35, 40]
        counts = np.zeros(len(edges), dtype=np.int64)
        total = 0
        for rg in range(pf.metadata.num_row_groups):
            d = pf.read_row_group(rg, columns=["dbz"])["dbz"].to_numpy()
            total += d.size
            for i, e in enumerate(edges):
                counts[i] += int(np.count_nonzero(d >= e))
        print(f"\n  {total:,} rows total")
        for e, c in zip(edges, counts):
            print(f"    >={e:>2} dBZ: {c:>13,}  ({100.0*c/max(total,1):5.1f}%)")
        return 0

    schema = pa.schema([
        ("lon", pa.float64()), ("lat", pa.float64()), ("timestamp", pa.int64()),
        ("alt_m", pa.float32()), ("dbz", pa.float32()), ("dbz_band", pa.string()),
        ("min_zoom", pa.int16()),
    ])
    lod_totals = np.zeros(n_zooms, dtype=np.int64)
    frame_shares: list[np.ndarray] = []
    kept = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # A row group is NOT a frame: pyarrow splits a written table at ~1 M rows,
    # so this file's 115 frames arrive as 432 groups. The LOD must see a WHOLE
    # frame at once (its cell carries the temporal bucket, so a frame split
    # across two calls would thin each half against itself and keep twice the
    # representatives). Rows are time-sorted and each frame is contiguous, so
    # buffer by timestamp and flush when the timestamp changes.
    frames_done = 0

    def flush(buf: dict[str, list[np.ndarray]], stamp: int) -> int:
        nonlocal kept, frames_done
        if not buf["lon"]:
            return 0  # frame had nothing above the floor
        if args.step_min and ((stamp // 60_000) % 1440) % args.step_min != 0:
            return 0
        lon = np.concatenate(buf["lon"])
        lat = np.concatenate(buf["lat"])
        alt = np.concatenate(buf["alt"])
        ts = np.concatenate(buf["ts"])
        dbz = np.concatenate(buf["dbz"])
        nf = lon.size
        if nf == 0:
            return 0
        mz = lod_min_zoom(lon, lat, alt, dbz, ts, LOD_BUCKET_MS,
                          args.min_zoom, args.max_zoom,
                          args.lod_cell_deg, args.lod_cell_alt,
                          origins=lod_origins)
        lod_totals[:] += np.bincount(mz - args.min_zoom, minlength=n_zooms)
        frame_shares.append(np.array(
            [100.0 * int(np.count_nonzero(mz <= z)) / nf
             for z in range(args.min_zoom, args.max_zoom + 1)]))
        w.write_table(pa.table({
            "lon": pa.array(lon, type=pa.float64()),
            "lat": pa.array(lat, type=pa.float64()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "alt_m": pa.array(alt, type=pa.float32()),
            "dbz": pa.array(dbz, type=pa.float32()),
            "dbz_band": dbz_band_column(dbz),
            "min_zoom": pa.array(mz, type=pa.int16()),
        }, schema=schema))
        kept += nf
        frames_done += 1
        if frames_done % 10 == 1:
            print(f"    frame {frames_done}  {nf:,} kept  ({kept:,} total)  "
                  f"{time_mod.time()-t0:.0f}s", flush=True)
        return nf

    with pq.ParquetWriter(args.output, schema, compression="snappy") as w:
        buf: dict[str, list[np.ndarray]] = {k: [] for k in ("lon", "lat", "alt", "ts", "dbz")}
        cur: int | None = None
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["lon", "lat", "timestamp", "alt_m", "dbz"])
            ts_all = t["timestamp"].to_numpy()
            dbz_all = t["dbz"].to_numpy()
            lon_all = t["lon"].to_numpy()
            lat_all = t["lat"].to_numpy()
            alt_all = t["alt_m"].to_numpy()
            # Split this group wherever the timestamp changes.
            cuts = np.flatnonzero(np.diff(ts_all)) + 1
            for seg in np.split(np.arange(ts_all.size), cuts):
                if seg.size == 0:
                    continue
                stamp = int(ts_all[seg[0]])
                if cur is not None and stamp != cur:
                    flush(buf, cur)
                    buf = {k: [] for k in buf}
                cur = stamp
                keep = dbz_all[seg] >= args.dbz_floor
                if not keep.any():
                    continue
                idx = seg[keep]
                buf["lon"].append(lon_all[idx])
                buf["lat"].append(lat_all[idx])
                buf["alt"].append(alt_all[idx])
                buf["ts"].append(ts_all[idx])
                buf["dbz"].append(dbz_all[idx])
        if cur is not None and buf["lon"]:
            flush(buf, cur)

    cum = np.cumsum(lod_totals)
    shares = np.vstack(frame_shares)
    print(f"\n  {kept:,} points kept at >={args.dbz_floor:g} dBZ "
          f"over {len(frame_shares)} frames")
    print("  LOD min-zoom pyramid (dataset-wide / median share of ONE bucket):")
    for i, z in enumerate(range(args.min_zoom, args.max_zoom + 1)):
        print(f"    z{z}: {cum[i]:>12,}  ({100.0*cum[i]/max(kept,1):5.1f}% of all)"
              f"   bucket median {np.median(shares[:, i]):5.1f}%"
              f"  worst {shares[:, i].min():5.1f}%")
    print(f"  wrote {args.output} ({args.output.stat().st_size/1e9:.2f} GB, "
          f"{time_mod.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
