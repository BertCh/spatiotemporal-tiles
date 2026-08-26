#!/usr/bin/env bash
# Build the scene-split ("stage + actors", `--scene-split`) cockpit bundle(s) for
# Argoverse 2. Mirrors argoverse_surfel_batch.sh (selective public-AWS download →
# extract → cleanup, all 7 ring cameras for the colorizer) but passes
# --scene-split, writing the TWO-archive `<id>-stage` bundle (static/ + dynamic/).
#
# Builds the scene-split `-stage` bundle for ALL 6 AV2 cities (auto-derived stage
# scenes in datasets.ts, gated LOCAL-ONLY until R2-synced). Eyeball each in the
# cockpit (/drive/<id> → Stage), then sync + drop the STAGE_LOCAL_ONLY gate.
#
#   bash argoverse_stage_batch.sh           # build the canary -stage scene(s)
#   FORCE=1 bash argoverse_stage_batch.sh   # rebuild
#   STAGE_VOXEL=0.25 bash argoverse_stage_batch.sh   # coarser stage (smaller)
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
# Stage density is kept LOW enough that the densest (concentrated-scene) tile stays
# well under the browser's per-tile decode limit — the full-range stage can put the
# whole cloud in one tile, so a 62 MB tile flickered/dropped. base 0.3 / flat-pct 80
# → ~300k pts even on dense downtown scenes → biggest tile ~16-19 MB.
STAGE_VOXEL="${STAGE_VOXEL:-0.3}"      # BASE voxel = detail kept in COMPLEX regions
STAGE_MAX_VOXEL="${STAGE_MAX_VOXEL:-0.6}"  # voxel for the flattest tier (big disks)
STAGE_FLAT_PCT="${STAGE_FLAT_PCT:-80}" # σ pct treated as fully flat (higher=more compact)
RING=(ring_front_center ring_front_left ring_front_right \
      ring_side_left ring_side_right ring_rear_left ring_rear_right)

aws_cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }

# log UUID : 3-letter city — all 6 AV2 cities (stage scenes auto-derived in datasets.ts).
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
  out="$OUT/argoverse-$short-stage"
  raw="$WORK/$log"
  echo "── $city · $log → argoverse-$short-stage (stage-voxel $STAGE_VOXEL) ──"
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  rm -rf "$out" "$raw"; mkdir -p "$raw/sensors/cameras"
  aws_cp "$BUCKET/$SPLIT/$log/annotations.feather"         "$raw/"
  aws_cp "$BUCKET/$SPLIT/$log/city_SE3_egovehicle.feather" "$raw/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/map/"            "$raw/map/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/lidar/"  "$raw/sensors/lidar/"
  # Camera calibration + all 7 ring cameras — REQUIRED: the stage/actors surfels
  # are camera-colored (scene-split forces the colorizer on).
  aws_cp --recursive "$BUCKET/$SPLIT/$log/calibration/"   "$raw/calibration/"
  for cam in "${RING[@]}"; do
    aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/cameras/$cam/" \
                       "$raw/sensors/cameras/$cam/"
  done
  "$PY" "$HERE/argoverse_extract.py" --log-dir "$raw" --city "$city" \
        --out "$out" --camera ring_front_center --scene-split \
        --stage-voxel "$STAGE_VOXEL" --stage-max-voxel "$STAGE_MAX_VOXEL" \
        --stage-flat-pct "$STAGE_FLAT_PCT" --stt-build "$STT"
  rm -rf "$raw"
  rm -f "$out"/*.parquet   # drop the intermediate GeoParquets (served bundle = packs)
  echo "  built argoverse-$short-stage ($city); raw removed"
  echo "  static/ pack bytes: $(du -sh "$out/static/packs" 2>/dev/null | cut -f1)"
  echo "  dynamic/ pack bytes: $(du -sh "$out/dynamic/packs" 2>/dev/null | cut -f1)"
done
[ -n "${KEEP_WORK:-}" ] || rm -rf "$WORK"
echo "ALL ARGOVERSE STAGE SCENES DONE"
