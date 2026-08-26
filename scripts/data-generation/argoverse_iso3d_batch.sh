#!/usr/bin/env bash
# Build the TRUE-3D density iso-line (`--contours --contour-z-step`) cockpit
# bundles for all 6 Argoverse 2 cities. Public AWS open-data, no auth. Mirrors
# argoverse_surfel_batch.sh's selective download → extract → cleanup, but: it does
# NOT colorize (iso-lines are density-colored, not camera-colored), so it SKIPS the
# calibration/ dir and downloads only the FRONT ring camera (for the cockpit inset)
# instead of all 7 — a much lighter pull. Passes --contours --contour-z-step,
# writing to <id>-iso3d (the cockpit's "Iso 3D" render mode).
#
#   bash argoverse_iso3d_batch.sh           # build any not-yet-built -iso3d scenes
#   FORCE=1 bash argoverse_iso3d_batch.sh   # rebuild all
#   Z_STEP=0.8 bash argoverse_iso3d_batch.sh # finer height layers
#
# Tune the relief with waymo/argoverse --contour-z-step / --contour-* (see
# av_common.add_contour_args). The showcase applies a vertical exaggeration
# (AV_ISO_DENSITY_ELEVATION_SCALE) at render time.
#
# AV2 license: CC-BY-NC-SA 4.0 (non-commercial). These bundles CAN go to R2 (the
# height-ramp / surfel AV2 bundles already do); r2-sync them with the rest.
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
# Tuned "HD-XY / coarse-Z" iso3d recipe (matches the Miami reference build):
# fine horizontal grid (CELL) over ALL returns (DECIMATE 1) with smoothing scaled
# to the finer cell (SIGMA), and a deliberately COARSE height step (Z_STEP) so the
# stack reads as a few clean readable bands (the showcase fades the upper slabs
# translucent at render time). Override any via env.
Z_STEP="${Z_STEP:-1.2}"
CELL="${CELL:-0.15}"
DECIMATE="${DECIMATE:-1}"
SIGMA="${SIGMA:-2.6}"

aws_cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }

# log UUID : 3-letter city — the EXACT 6 logs the cockpit ships (so the iso3d
# scene is the same place as the height-ramp / surfel one).
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
  out="$OUT/argoverse-$short-iso3d"
  raw="$WORK/$log"
  echo "── $city · $log → argoverse-$short-iso3d ──"
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  rm -rf "$out" "$raw"; mkdir -p "$raw/sensors/cameras"
  aws_cp "$BUCKET/$SPLIT/$log/annotations.feather"         "$raw/"
  aws_cp "$BUCKET/$SPLIT/$log/city_SE3_egovehicle.feather" "$raw/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/map/"            "$raw/map/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/lidar/"  "$raw/sensors/lidar/"
  # Front ring camera ONLY (the cockpit inset); no calibration / colorize.
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/cameras/ring_front_center/" \
                     "$raw/sensors/cameras/ring_front_center/"
  "$PY" "$HERE/argoverse_extract.py" --log-dir "$raw" --city "$city" \
        --out "$out" --camera ring_front_center \
        --contours --contour-cell "$CELL" --contour-decimate "$DECIMATE" \
        --contour-z-step "$Z_STEP" --contour-sigma "$SIGMA" --stt-build "$STT"
  rm -rf "$raw"
  echo "  built argoverse-$short-iso3d ($city); raw removed"
done
[ -n "${KEEP_WORK:-}" ] || rm -rf "$WORK"
echo "ALL ARGOVERSE ISO3D SCENES DONE"
