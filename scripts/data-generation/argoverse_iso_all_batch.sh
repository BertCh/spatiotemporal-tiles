#!/usr/bin/env bash
# Build BOTH density iso-line cockpit bundles — flat (<id>-iso) AND true-3D
# (<id>-iso3d) — for all 6 Argoverse 2 cities, downloading each log from public
# AWS open-data ONCE and building both before deleting the raw (vs two separate
# batches that would re-download 6 logs each). Both prioritize XY (spatial)
# resolution: the high-XY-res grid/decimate/sigma/simplify defaults live in
# av_common.add_contour_args; the 3D one additionally keeps height layers
# (--contour-z-step, default 1.0 — height is a key factor, just secondary to XY).
#
#   bash argoverse_iso_all_batch.sh            # build any not-yet-built bundles
#   FORCE=1 bash argoverse_iso_all_batch.sh    # rebuild all (use after a param change)
#   Z_STEP=1.0 bash argoverse_iso_all_batch.sh # 3D height-slab spacing (m)
#
# AV2 license: CC-BY-NC-SA 4.0 (non-commercial) — these CAN go to R2.
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
Z_STEP="${Z_STEP:-1.0}"

aws_cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }

# log UUID : 3-letter city — the EXACT 6 logs the cockpit ships.
SCENES=(
  "02678d04-cc9f-3148-9f95-1ba66347dff9:PIT"
  "02a00399-3857-444e-8db3-a8f58489c394:MIA"
  "0b5142c1-420b-3fea-9e98-b87327ae22c6:WDC"
  "0bae3b5e-417d-3b03-abaa-806b433233b8:DTW"
  "25e5c600-36fe-3245-9cc0-40ef91620c22:PAO"
  "92b900b1-ac4a-3d41-b118-e42c66382c91:ATX"
)

for entry in "${SCENES[@]}"; do
  log="${entry%%:*}"; city="${entry#*:}"; short="${log:0:8}"
  flat="$OUT/argoverse-$short-iso"; d3="$OUT/argoverse-$short-iso3d"
  raw="$WORK/$log"
  echo "══ $city · $log → argoverse-$short-{iso,iso3d} ══"
  if [[ -f "$flat/scene.json" && -f "$d3/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  both already built — skip (FORCE=1 to rebuild)"; continue
  fi
  rm -rf "$raw"; mkdir -p "$raw/sensors/cameras"
  aws_cp "$BUCKET/$SPLIT/$log/annotations.feather"         "$raw/"
  aws_cp "$BUCKET/$SPLIT/$log/city_SE3_egovehicle.feather" "$raw/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/map/"            "$raw/map/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/lidar/"  "$raw/sensors/lidar/"
  # Front ring camera ONLY (cockpit inset); no calibration / colorize for iso.
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/cameras/ring_front_center/" \
                     "$raw/sensors/cameras/ring_front_center/"
  echo "  -- flat (argoverse-$short-iso) --"
  rm -rf "$flat"
  "$PY" "$HERE/argoverse_extract.py" --log-dir "$raw" --city "$city" \
        --out "$flat" --camera ring_front_center --contours --stt-build "$STT"
  echo "  -- 3D (argoverse-$short-iso3d) --"
  rm -rf "$d3"
  "$PY" "$HERE/argoverse_extract.py" --log-dir "$raw" --city "$city" \
        --out "$d3" --camera ring_front_center \
        --contours --contour-z-step "$Z_STEP" --stt-build "$STT"
  rm -rf "$raw"
  echo "  built argoverse-$short-iso + -iso3d ($city); raw removed"
done
[ -n "${KEEP_WORK:-}" ] || rm -rf "$WORK"
echo "ALL ARGOVERSE ISO + ISO3D SCENES DONE"
