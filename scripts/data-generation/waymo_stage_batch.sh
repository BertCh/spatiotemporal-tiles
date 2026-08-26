#!/usr/bin/env bash
# Build the scene-split ("stage + actors", --scene-split) bundle for ALL 5 Waymo
# scenes. Mirrors waymo_batch.sh (pull only the needed components, skip those on
# disk) but passes --scene-split, writing the two-archive `<id>-stage` bundle
# (static/ + dynamic/). Stage scenes are auto-derived in datasets.ts; Waymo is
# LOCAL-ONLY (license: non-commercial, no redistribution) AND `-stage` is gated
# local-only, so these never reach the public deploy.
#
#   bash waymo_stage_batch.sh           # build any not-yet-built -stage scenes
#   FORCE=1 bash waymo_stage_batch.sh   # rebuild all
#
# Actor denoise is OFF by default (most scenes want every moving return); only the
# rain scene enables it (heavy rain scatter) — see RAIN_EXTRA below.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

BUCKET="gs://waymo_open_dataset_v_2_0_1"
SPLIT="validation"
RAW="waymo-raw"
OUT="../../data-fleet"
STT="${STT_BUILD:-$HERE/../../target/release/stt-build}"
PY="venv-waymo/bin/python"
COMPONENTS="lidar lidar_calibration lidar_box vehicle_pose camera_image camera_calibration stats"
# Keep the stage SIMPLE enough that the densest (concentrated-scene) tile stays
# well under the browser's per-tile decode limit (a full-range stage can pack the
# whole cloud into one tile; a 62 MB tile flickered/dropped). base 0.3 / flat-pct 80
# → ~300k pts even on dense scenes → biggest tile ~16-19 MB.
STAGE_VOXEL="${STAGE_VOXEL:-0.3}"
STAGE_MAX_VOXEL="${STAGE_MAX_VOXEL:-0.6}"
STAGE_FLAT_PCT="${STAGE_FLAT_PCT:-80}"

SCENES=(
  "waymo-sf-day:1943605865180232897_680_000_700_000"
  "waymo-phx-day:8956556778987472864_3404_790_3424_790"
  "waymo-phx-night:18024188333634186656_1566_600_1586_600"
  "waymo-sf-night:8679184381783013073_7740_000_7760_000"
  "waymo-phx-dusk-rain:13415985003725220451_6163_000_6183_000"
)
# Per-scene actor handling. The rain scene is the noise outlier → principled
# denoise. The two SF scenes have dense traffic that overplots → a per-sweep voxel
# dedup caps the redundant density (NOT thinning the agents). The rest keep every
# moving return.
RAIN_SEG="13415985003725220451_6163_000_6183_000"
SF_DAY_SEG="1943605865180232897_680_000_700_000"
SF_NIGHT_SEG="8679184381783013073_7740_000_7760_000"

for entry in "${SCENES[@]}"; do
  id="${entry%%:*}"; seg="${entry#*:}"
  out="$OUT/$id-stage"
  echo "=== $id-stage  ($seg) ==="
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  for c in $COMPONENTS; do
    dst="$RAW/$SPLIT/$c/$seg.parquet"
    if [[ -f "$dst" ]]; then echo "  have $c"; else
      mkdir -p "$RAW/$SPLIT/$c"; echo "  downloading $c …"
      gcloud storage cp "$BUCKET/$SPLIT/$c/$seg.parquet" "$RAW/$SPLIT/$c/"
    fi
  done
  EXTRA=""
  case "$seg" in
    "$RAIN_SEG") EXTRA="--actor-denoise --actor-voxel 0.1" ;;   # rain scatter
    "$SF_DAY_SEG"|"$SF_NIGHT_SEG") EXTRA="--actor-voxel 0.1" ;; # dense SF traffic
  esac
  "$PY" waymo_extract.py --seg "$seg" --out "$out" --scene-split \
        --stage-voxel "$STAGE_VOXEL" --stage-max-voxel "$STAGE_MAX_VOXEL" \
        --stage-flat-pct "$STAGE_FLAT_PCT" $EXTRA --stt-build "$STT"
  rm -f "$out"/*.parquet   # drop the intermediate GeoParquets (served bundle = packs)
  echo "  static/ $(du -sh "$out/static/packs" 2>/dev/null | cut -f1) · dynamic/ $(du -sh "$out/dynamic/packs" 2>/dev/null | cut -f1)"
done
echo "ALL WAYMO STAGE SCENES DONE"
