#!/usr/bin/env bash
# Download + build the curated Waymo Open Dataset (Perception v2.0.1) scenes for
# the /drive AV cockpit. Mirrors argoverse_batch.sh: per scene it pulls only the
# components the cockpit needs (skipping any already on disk), then runs
# waymo_extract.py. Requires: an authed `gcloud` (you accepted the Waymo license
# + `gcloud auth login`), the venv-waymo python, and a release stt-build.
#
#   bash waymo_batch.sh            # build any not-yet-built scenes
#   FORCE=1 bash waymo_batch.sh    # rebuild all
#
# Waymo Dataset License Agreement: NON-COMMERCIAL, no redistribution. Do not push
# the raw waymo-raw/ components or derived bundles anywhere public without
# confirming the license permits it.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

BUCKET="gs://waymo_open_dataset_v_2_0_1"
SPLIT="validation"
RAW="waymo-raw"
OUT="../../examples/showcase/public/data"
STT="${STT_BUILD:-$HERE/../../target/release/stt-build}"
PY="venv-waymo/bin/python"
COMPONENTS="lidar lidar_calibration lidar_box vehicle_pose camera_image stats"

# scene id -> Waymo segment context name (curated for day/night/location/weather
# spread; see the data-generation README).
SCENES=(
  "waymo-sf-day:1943605865180232897_680_000_700_000"
  "waymo-phx-day:8956556778987472864_3404_790_3424_790"
  "waymo-phx-night:18024188333634186656_1566_600_1586_600"
  "waymo-sf-night:8679184381783013073_7740_000_7760_000"
  "waymo-phx-dusk-rain:13415985003725220451_6163_000_6183_000"
)

for entry in "${SCENES[@]}"; do
  id="${entry%%:*}"; seg="${entry#*:}"
  echo "=== $id  ($seg) ==="
  if [[ -f "$OUT/$id/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  for c in $COMPONENTS; do
    dst="$RAW/$SPLIT/$c/$seg.parquet"
    if [[ -f "$dst" ]]; then
      echo "  have $c"
    else
      mkdir -p "$RAW/$SPLIT/$c"
      echo "  downloading $c …"
      gcloud storage cp "$BUCKET/$SPLIT/$c/$seg.parquet" "$RAW/$SPLIT/$c/"
    fi
  done
  "$PY" waymo_extract.py --seg "$seg" --out "$OUT/$id" --stt-build "$STT"
done
echo "ALL WAYMO SCENES DONE"
