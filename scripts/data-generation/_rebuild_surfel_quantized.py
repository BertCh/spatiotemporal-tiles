"""One-off: rebuild a stale surfel lidar archive THROUGH the already-wired
quantization path (av_common.run_stt_build with lidar_quantize_attrs), from the
cached lidar.parquet — no Waymo/AV2 raw needed. Drops the dead `intensity`
column first (current extractors no longer write it). Builds into <dir>/lidar.q
so the live <dir>/lidar is untouched until verified.

Usage:
  venv-waymo/bin/python _rebuild_surfel_quantized.py <scene_dir> [--zmax N]
"""
import sys
import shutil
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av_common as avc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STT_BUILD = str(REPO / "target" / "release" / "stt-build")
LIDAR_QUANTIZE_M = 0.05
LIDAR_MIN_ZOOM = 14


def main() -> None:
    scene_dir = Path(sys.argv[1])
    zmax = 18
    if "--zmax" in sys.argv:
        zmax = int(sys.argv[sys.argv.index("--zmax") + 1])
    src_pq = scene_dir / "lidar.parquet"
    assert src_pq.exists(), f"missing {src_pq}"

    # Strip the dead `intensity` column (a fresh extraction omits it).
    t = pq.read_table(src_pq)
    if "intensity" in t.column_names:
        t = t.drop(["intensity"])
        print(f"  dropped intensity → {t.num_rows} rows, cols {t.column_names}")
    stripped = scene_dir / "lidar.noint.parquet"
    pq.write_table(t, stripped)

    out_dir = scene_dir / "lidar.q"
    shutil.rmtree(out_dir, ignore_errors=True)
    avc.run_stt_build(
        stripped, out_dir, "point",
        stt_build=STT_BUILD,
        temporal_bucket="100ms",
        min_zoom=LIDAR_MIN_ZOOM,
        max_zoom=zmax,
        quantize_coords=LIDAR_QUANTIZE_M,
        quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
        publish=True,
    )
    stripped.unlink(missing_ok=True)
    print(f"  DONE → {out_dir}")


if __name__ == "__main__":
    main()
