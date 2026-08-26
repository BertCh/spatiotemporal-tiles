#!/usr/bin/env bash
# Build the density iso-line (`--contours`) cockpit bundle(s) for the Waymo
# scenes, from the already-cached waymo-raw/ components (NO gcloud/download).
# Mirrors waymo_surfel_batch.sh but adds --contours and writes to <id>-iso. The
# iso build draws the LIDAR returns as live topographic density contours (an
# alternate `/drive` render mode), tuned by the waymo_extract.py --contour-*
# knobs (defaults: 0.5 m grid cell, decimate 4 — see that file for the rest).
#
#   bash waymo_iso_batch.sh            # build any not-yet-built -iso scenes
#   FORCE=1 bash waymo_iso_batch.sh    # rebuild all
#
# Waymo Dataset License Agreement: NON-COMMERCIAL, no redistribution. The iso
# bundles stay LOCAL-ONLY until the license is confirmed to permit R2 hosting.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUT="../../data-fleet"
STT="${STT_BUILD:-$HERE/../../target/release/stt-build}"
PY="venv-waymo/bin/python"

# High-XY FLAT density overview recipe (matches argoverse_iso_all_batch.sh's flat
# pass): with no height layers, flat spends its whole budget on a FINE horizontal
# grid over ALL returns. The 3D sibling is waymo_iso3d_batch.sh. Override via env.
CELL="${CELL:-0.10}"
DECIMATE="${DECIMATE:-1}"
SIGMA="${SIGMA:-3.0}"

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
  out="$OUT/$id-iso"
  echo "=== $id-iso  ($seg) ==="
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  "$PY" waymo_extract.py --seg "$seg" --out "$out" --stt-build "$STT" --contours \
    --contour-cell "$CELL" --contour-decimate "$DECIMATE" --contour-sigma "$SIGMA"
  echo "  built $id-iso"
done
echo "ALL WAYMO ISO SCENES DONE"
