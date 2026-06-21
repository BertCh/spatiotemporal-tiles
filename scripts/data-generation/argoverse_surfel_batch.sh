#!/usr/bin/env bash
# Build the oriented-surfel (`--surfel`) cockpit bundles for the 5 Argoverse 2
# cities that don't yet have one (Miami already does). Public AWS open-data,
# no auth. Mirrors argoverse_batch.sh's selective download → extract → cleanup,
# but: (1) targets the EXACT logs already shipped (so the surfel scene is the
# same place as the height-ramp one), (2) downloads all 7 RING cameras (the
# surfel colorizer projects into all of them), and (3) passes --surfel, writing
# to <id>-surfel. Quantization is auto-applied by run_stt_build.
#
#   bash argoverse_surfel_batch.sh           # build any not-yet-built -surfel scenes
#   FORCE=1 bash argoverse_surfel_batch.sh   # rebuild all
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

BUCKET="${BUCKET:-s3://argoverse/datasets/av2/sensor}"
SPLIT="${SPLIT:-val}"
PY="${PY:-$HERE/venv-av2/bin/python}"
STT="${STT_BUILD:-$REPO/target/release/stt-build}"
OUT="${OUT_DIR:-$REPO/examples/showcase/public/data}"
WORK="${WORK:-$(mktemp -d)}"
RING=(ring_front_center ring_front_left ring_front_right \
      ring_side_left ring_side_right ring_rear_left ring_rear_right)

aws_cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }

# log UUID : 3-letter city (recovered via `aws s3 ls .../val/`).
# Miami (02a00399) is included so a full re-roll (FORCE=1) rebuilds every city
# with the current pipeline; without FORCE an already-built city is skipped.
SCENES=(
  "02a00399-3857-444e-8db3-a8f58489c394:MIA"
  "02678d04-cc9f-3148-9f95-1ba66347dff9:PIT"
  "0b5142c1-420b-3fea-9e98-b87327ae22c6:WDC"
  "0bae3b5e-417d-3b03-abaa-806b433233b8:DTW"
  "25e5c600-36fe-3245-9cc0-40ef91620c22:PAO"
  "92b900b1-ac4a-3d41-b118-e42c66382c91:ATX"
)

for entry in "${SCENES[@]}"; do
  log="${entry%%:*}"; city="${entry#*:}"; short="${log:0:8}"
  out="$OUT/argoverse-$short-surfel"
  raw="$WORK/$log"
  echo "── $city · $log → argoverse-$short-surfel ──"
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  rm -rf "$out" "$raw"; mkdir -p "$raw/sensors/cameras"
  aws_cp "$BUCKET/$SPLIT/$log/annotations.feather"         "$raw/"
  aws_cp "$BUCKET/$SPLIT/$log/city_SE3_egovehicle.feather" "$raw/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/map/"            "$raw/map/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/lidar/"  "$raw/sensors/lidar/"
  # Camera calibration (intrinsics + egovehicle_SE3_sensor) — REQUIRED for the
  # surfel colorizer (PinholeCamera.from_feather). Without it every ring camera
  # fails to load → 0% colour coverage → the cloud renders as fallback slate.
  aws_cp --recursive "$BUCKET/$SPLIT/$log/calibration/"   "$raw/calibration/"
  for cam in "${RING[@]}"; do
    aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/cameras/$cam/" \
                       "$raw/sensors/cameras/$cam/"
  done
  "$PY" "$HERE/argoverse_extract.py" --log-dir "$raw" --city "$city" \
        --out "$out" --camera ring_front_center --surfel --stt-build "$STT"
  rm -rf "$raw"
  echo "  built argoverse-$short-surfel ($city); raw removed"
done
[ -n "${KEEP_WORK:-}" ] || rm -rf "$WORK"
echo "ALL ARGOVERSE SURFEL SCENES DONE"
