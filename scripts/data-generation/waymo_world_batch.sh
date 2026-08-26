#!/usr/bin/env bash
# Build the WORLDBUILD (`--worldbuild`) cockpit bundles for the Waymo scenes
# (Feature 1: one consolidated cross-sweep-merged cloud — static returns
# voxel-deduped over the whole drive, dynamic actors passed through un-merged),
# from the already-cached waymo-raw/ components (NO gcloud/download). Mirrors
# waymo_surfel_batch.sh but passes --worldbuild and writes to <id>-world. The
# build auto-applies the canonical quantization (incl. is_dynamic) via
# run_stt_build, so each bundle is the "optimized" form.
#
#   bash waymo_world_batch.sh            # build any not-yet-built -world scenes
#   FORCE=1 bash waymo_world_batch.sh    # rebuild all
#   WORLDBUILD_VOXEL=0.12 WORLDBUILD_STATIC_SPEED=0.8 bash waymo_world_batch.sh
#
# Waymo Dataset License Agreement: NON-COMMERCIAL, NO redistribution. These
# bundles are LOCAL-ONLY — do NOT R2-sync them without confirming the license.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUT="../../data-fleet"
STT="${STT_BUILD:-$HERE/../../target/release/stt-build}"
PY="venv-waymo/bin/python"
WB_VOXEL="${WORLDBUILD_VOXEL:-0.15}"
WB_SPEED="${WORLDBUILD_STATIC_SPEED:-0.5}"

# Same five curated scenes as waymo_batch.sh / waymo_surfel_batch.sh.
SCENES=(
  "waymo-sf-day:1943605865180232897_680_000_700_000"
  "waymo-phx-day:8956556778987472864_3404_790_3424_790"
  "waymo-phx-night:18024188333634186656_1566_600_1586_600"
  "waymo-sf-night:8679184381783013073_7740_000_7760_000"
  "waymo-phx-dusk-rain:13415985003725220451_6163_000_6183_000"
)

for entry in "${SCENES[@]}"; do
  id="${entry%%:*}"; seg="${entry#*:}"
  out="$OUT/$id-world"
  echo "=== $id-world  ($seg) ==="
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  "$PY" waymo_extract.py --seg "$seg" --out "$out" --stt-build "$STT" \
        --worldbuild --worldbuild-voxel "$WB_VOXEL" \
        --worldbuild-static-speed "$WB_SPEED"
  echo "  built $id-world (LOCAL-ONLY — do not R2-sync)"
done
echo "ALL WAYMO WORLDBUILD SCENES DONE (LOCAL-ONLY)"
