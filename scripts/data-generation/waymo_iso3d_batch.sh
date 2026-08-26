#!/usr/bin/env bash
# Build the TRUE-3D density iso-line (`--contours --contour-z-step`) cockpit
# bundle(s) for the Waymo scenes, from the already-cached waymo-raw/ components
# (NO gcloud/download). Mirrors waymo_iso_batch.sh but slices the returns into
# HEIGHT LAYERS and contours each layer's XY density independently, tagging every
# contour with its real altitude (`z_layer`). The frontend's "Iso 3D" render mode
# lifts the iso-lines to that height, so the vertical axis carries genuine LIDAR
# structure — a wall contours up its full height, a parked car only near the
# ground — rather than an artificial band→height map. Writes to <id>-iso3d.
#
#   bash waymo_iso3d_batch.sh            # build any not-yet-built -iso3d scenes
#   FORCE=1 bash waymo_iso3d_batch.sh    # rebuild all
#
# Tune the relief with waymo_extract.py --contour-z-step / --contour-z-thickness /
# --contour-z-min / --contour-z-max / --contour-min-pts-layer (+ the shared
# --contour-cell / --contour-levels / etc.). NB: this Waymo scene is vertically
# shallow (top-LIDAR FOV ~a few m), so the real z span is small — the showcase
# applies a vertical exaggeration (AV_ISO_DENSITY_ELEVATION_SCALE) at render time.
#
# Waymo Dataset License Agreement: NON-COMMERCIAL, no redistribution. The iso3d
# bundles stay LOCAL-ONLY until the license is confirmed to permit R2 hosting.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUT="../../data-fleet"
STT="${STT_BUILD:-$HERE/../../target/release/stt-build}"
PY="venv-waymo/bin/python"

# Tuned "HD-XY / coarse-Z" iso3d recipe (mirrors argoverse_iso3d_batch.sh / the
# Miami reference): fine horizontal grid (CELL) over ALL returns (DECIMATE 1),
# smoothing scaled to the cell (SIGMA), coarse height step (Z_STEP). Waymo's z
# span is shallow (top-LIDAR FOV ~few m), so 1.2 m gives only ~3 readable bands —
# lower Z_STEP for more vertical layers on these scenes. Override any via env.
Z_STEP="${Z_STEP:-1.2}"
CELL="${CELL:-0.15}"
DECIMATE="${DECIMATE:-1}"
SIGMA="${SIGMA:-2.6}"

# All 5 Waymo cockpit scenes (same set as waymo_surfel_batch.sh).
SCENES=(
  "waymo-sf-day:1943605865180232897_680_000_700_000"
  "waymo-phx-day:8956556778987472864_3404_790_3424_790"
  "waymo-phx-night:18024188333634186656_1566_600_1586_600"
  "waymo-sf-night:8679184381783013073_7740_000_7760_000"
  "waymo-phx-dusk-rain:13415985003725220451_6163_000_6183_000"
)

for entry in "${SCENES[@]}"; do
  id="${entry%%:*}"; seg="${entry#*:}"
  out="$OUT/$id-iso3d"
  echo "=== $id-iso3d  ($seg) ==="
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  "$PY" waymo_extract.py --seg "$seg" --out "$out" --stt-build "$STT" \
    --contours --contour-cell "$CELL" --contour-decimate "$DECIMATE" \
    --contour-z-step "$Z_STEP" --contour-sigma "$SIGMA"
  echo "  built $id-iso3d"
done
echo "ALL WAYMO ISO3D SCENES DONE"
