#!/usr/bin/env bash
# Build the oriented-surfel (`--surfel`) cockpit bundles for the Waymo scenes,
# from the already-cached waymo-raw/ components (NO gcloud/download). Mirrors
# waymo_batch.sh but adds --surfel and writes to <id>-surfel. The surfel build
# auto-applies the canonical quantization (av_common.lidar_quantize_attrs) via
# run_stt_build, so each bundle is the "optimized" form.
#
#   bash waymo_surfel_batch.sh            # build any not-yet-built -surfel scenes
#   FORCE=1 bash waymo_surfel_batch.sh    # rebuild all
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUT="../../data-fleet"
STT="${STT_BUILD:-$HERE/../../target/release/stt-build}"
PY="venv-waymo/bin/python"

# sf-day already has a -surfel bundle; build the other four.
SCENES=(
  "waymo-phx-day:8956556778987472864_3404_790_3424_790"
  "waymo-phx-night:18024188333634186656_1566_600_1586_600"
  "waymo-sf-night:8679184381783013073_7740_000_7760_000"
  "waymo-phx-dusk-rain:13415985003725220451_6163_000_6183_000"
)

for entry in "${SCENES[@]}"; do
  id="${entry%%:*}"; seg="${entry#*:}"
  out="$OUT/$id-surfel"
  echo "=== $id-surfel  ($seg) ==="
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  "$PY" waymo_extract.py --seg "$seg" --out "$out" --stt-build "$STT" --surfel
  echo "  built $id-surfel"
done
echo "ALL WAYMO SURFEL SCENES DONE"
