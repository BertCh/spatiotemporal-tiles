#!/usr/bin/env bash
# Build the SWEEP (`--scan`) cockpit bundles for the Argoverse 2 cities
# (Feature 5, AV2 only: per-return TRUE scan time = sweep_ts + offset_ns→ms, so
# the playhead reveals the LiDAR beam sweeping the scene, coloured by scan_phase
# as a rotating rainbow). A separate '<id>-scan' bundle (single mid-density tier,
# NOT a base-archive change). Public AWS open-data, no auth. Mirrors
# argoverse_batch.sh's selective download → extract → cleanup, targeting the EXACT
# logs already shipped, but: (1) NO cameras (the rainbow is from scan_phase, not
# imagery — saves the camera download), and (2) passes --scan, writing <id>-scan.
#
#   bash argoverse_scan_batch.sh           # build any not-yet-built -scan scenes
#   FORCE=1 bash argoverse_scan_batch.sh   # rebuild all
#   SCAN_DECIMATE=2 bash argoverse_scan_batch.sh
#
# Argoverse 2 = CC BY-NC-SA 4.0 (non-commercial); the scan bundles are R2-OK
# (same license posture as the existing AV2 scenes).
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
SCAN_DEC="${SCAN_DECIMATE:-4}"

aws_cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }

# log UUID : 3-letter city (same six logs as the surfel / world batches).
SCENES=(
  "02678d04-cc9f-3148-9f95-1ba66347dff9:PIT"
  "0b5142c1-420b-3fea-9e98-b87327ae22c6:WDC"
  "0bae3b5e-417d-3b03-abaa-806b433233b8:DTW"
  "25e5c600-36fe-3245-9cc0-40ef91620c22:PAO"
  "92b900b1-ac4a-3d41-b118-e42c66382c91:ATX"
)

[ -x "$PY" ]  || { echo "ERROR: python not found at $PY (set PY=...)"; exit 1; }
[ -x "$STT" ] || { echo "ERROR: stt-build not found at $STT (cargo build -p stt-build --release)"; exit 1; }

for entry in "${SCENES[@]}"; do
  log="${entry%%:*}"; city="${entry#*:}"; short="${log:0:8}"
  out="$OUT/argoverse-$short-scan"
  raw="$WORK/$log"
  echo "── $city · $log → argoverse-$short-scan ──"
  if [[ -f "$out/scene.json" && "${FORCE:-0}" != "1" ]]; then
    echo "  already built — skip (FORCE=1 to rebuild)"; continue
  fi
  rm -rf "$out" "$raw"; mkdir -p "$raw/sensors"
  aws_cp "$BUCKET/$SPLIT/$log/annotations.feather"         "$raw/"
  aws_cp "$BUCKET/$SPLIT/$log/city_SE3_egovehicle.feather" "$raw/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/map/"            "$raw/map/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/lidar/"  "$raw/sensors/lidar/"
  # --scan needs NO cameras (the rainbow is computed from scan_phase). Build with
  # --no-camera so a missing cam dir doesn't warn.
  "$PY" "$HERE/argoverse_extract.py" --log-dir "$raw" --city "$city" \
        --out "$out" --scan --scan-decimate "$SCAN_DEC" --no-camera \
        --stt-build "$STT"
  rm -rf "$raw"
  echo "  built argoverse-$short-scan ($city); raw removed"
done
[ -n "${KEEP_WORK:-}" ] || rm -rf "$WORK"
echo "ALL ARGOVERSE SCAN SCENES DONE"
