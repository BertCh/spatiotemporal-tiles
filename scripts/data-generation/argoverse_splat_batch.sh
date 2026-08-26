#!/usr/bin/env bash
# Build the camera-coloured POINT-SPLAT (`--colorize`, no `--surfel`) cockpit
# bundles for the 6 Argoverse 2 cities — the `mode=splat` bundles
# (AnimatedPointLayer with `splat:true`, `<id>-splat`). Public AWS open-data, no
# auth. Mirrors argoverse_surfel_batch.sh exactly (same logs, all 7 RING cameras
# for the colorizer, calibration), but passes `--colorize` instead of `--surfel`
# and writes to <id>-splat. Colour rides an interleaved `point_rgba` vector
# column so the client binds it zero-copy (no per-point re-pack on the render
# thread); elevation rides the quantized `z` COLUMN, which the layer lifts via
# `elevationProperty` — never folded into the geometry, see the note in
# av_common.run_stt_build.
#
#   bash argoverse_splat_batch.sh           # build any not-yet-built -splat scenes
#   FORCE=1 bash argoverse_splat_batch.sh   # rebuild all
#   CITIES="MIA PIT" FORCE=1 bash argoverse_splat_batch.sh   # subset (one at a time)
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

BUCKET="${BUCKET:-s3://argoverse/datasets/av2/sensor}"
SPLIT="${SPLIT:-val}"
PY="${PY:-$HERE/venv-av2/bin/python}"
STT="${STT_BUILD:-$REPO/target/release/stt-build}"
OUT="${OUT_DIR:-$REPO/data-fleet}"
WORK="${WORK:-$(mktemp -d)}"
CITIES="${CITIES:-MIA PIT WDC DTW PAO ATX}"
RING=(ring_front_center ring_front_left ring_front_right \
      ring_side_left ring_side_right ring_rear_left ring_rear_right)

aws_cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }

# log UUID : 3-letter city (mirrors argoverse_surfel_batch.sh).
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
  case " $CITIES " in *" $city "*) ;; *) continue ;; esac   # not a target city
  out="$OUT/argoverse-$short-splat"
  raw="$WORK/$log"
  echo "── $city · $log → argoverse-$short-splat ──"
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  rm -rf "$out" "$raw"; mkdir -p "$raw/sensors/cameras"
  aws_cp "$BUCKET/$SPLIT/$log/annotations.feather"         "$raw/"
  aws_cp "$BUCKET/$SPLIT/$log/city_SE3_egovehicle.feather" "$raw/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/map/"            "$raw/map/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/lidar/"  "$raw/sensors/lidar/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/calibration/"   "$raw/calibration/"
  for cam in "${RING[@]}"; do
    aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/cameras/$cam/" \
                       "$raw/sensors/cameras/$cam/"
  done
  "$PY" "$HERE/argoverse_extract.py" --log-dir "$raw" --city "$city" \
        --out "$out" --camera ring_front_center --colorize --stt-build "$STT"
  rm -rf "$raw"
  echo "  built argoverse-$short-splat ($city); raw removed"
done
[ -n "${KEEP_WORK:-}" ] || rm -rf "$WORK"
echo "ALL ARGOVERSE SPLAT SCENES DONE"
