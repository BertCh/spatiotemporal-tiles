#!/usr/bin/env bash
# argoverse_batch.sh — build one (or more) AV-cockpit scene bundle(s) per AV2 city.
#
# Argoverse 2's Sensor dataset is public, multi-city, and login-free
# (s3://argoverse, --no-sign-request). This driver gives the cockpit broad
# coverage with no manual log hunting:
#
#   1. Pick logs — scan only each log's tiny `map/` listing (NO sensor download)
#      and read the city from the map filename (`...____PIT_city_71109.json`),
#      taking the first PER_CITY logs for each target city.
#   2. Download selectively — only `annotations.feather`,
#      `city_SE3_egovehicle.feather`, `map/`, `sensors/lidar/`, and ONE ring
#      camera (skips the other 8 cameras + stereo): ~285 MB/log, not ~1.5 GB.
#   3. Extract — run `argoverse_extract.py` → a packed scene bundle under
#      `examples/showcase/public/data/argoverse-<log8>/`, then delete the raw log.
#
# Each bundle gets the full cockpit stream set (lidar / ego / objects / HD-map
# with lane centerlines / camera inset / ego-derived telemetry).
#
# Usage:
#   bash argoverse_batch.sh                       # 1 scene per city (6 total)
#   PER_CITY=2 bash argoverse_batch.sh            # 2 per city (~12 total)
#   CITIES="PIT MIA" bash argoverse_batch.sh      # just two cities
#   KEEP_WORK=1 bash argoverse_batch.sh           # keep the temp download dir
#
# Requires: aws CLI, the venv-av2 python (av2/pyproj/pyarrow/shapely), and a
# release `stt-build` (cargo build -p stt-build --release).
set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
BUCKET="${BUCKET:-s3://argoverse/datasets/av2/sensor}"
SPLIT="${SPLIT:-val}"
CITIES="${CITIES:-PIT MIA ATX DTW PAO WDC}"
PER_CITY="${PER_CITY:-1}"
CAMERA="${CAMERA:-ring_front_center}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-$HERE/venv-av2/bin/python}"
STT_BUILD="${STT_BUILD:-$REPO/target/release/stt-build}"
OUT_DIR="${OUT_DIR:-$REPO/examples/showcase/public/data}"
WORK="${WORK:-$(mktemp -d)}"

aws_ls() { aws s3 ls --no-sign-request "$@"; }
aws_cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }

echo "Argoverse 2 batch: $PER_CITY scene(s)/city over [$CITIES] from $BUCKET/$SPLIT"
echo "  python=$PY  stt-build=$STT_BUILD"
echo "  out=$OUT_DIR  work=$WORK"
[ -x "$PY" ]        || { echo "ERROR: python not found at $PY (set PY=...)"; exit 1; }
[ -x "$STT_BUILD" ] || { echo "ERROR: stt-build not found at $STT_BUILD (cargo build -p stt-build --release)"; exit 1; }

# ── 1. discover one-(or N-)per-city logs (probe only map/, no sensor data) ────
city_of() {  # $1 = log id  →  3-letter city code on stdout (empty if unknown)
  aws_ls "$BUCKET/$SPLIT/$1/map/" 2>/dev/null \
    | awk '/log_map_archive_/{print $4}' \
    | grep -oE '(ATX|DTW|MIA|PAO|PIT|WDC)_city_' | head -1 | cut -d_ -f1
}

PICKS="$WORK/picks.txt"; : > "$PICKS"
count_for() { grep -c " $1\$" "$PICKS" 2>/dev/null || true; }   # picks for a city

NEED=0; for _ in $CITIES; do NEED=$((NEED + PER_CITY)); done
GOT=0

echo "Scanning $SPLIT logs for city coverage (need $NEED)…"
for log in $(aws_ls "$BUCKET/$SPLIT/" | awk '/PRE/{print $2}' | tr -d '/'); do
  [ "$GOT" -ge "$NEED" ] && break
  c="$(city_of "$log")"
  [ -z "$c" ] && continue
  case " $CITIES " in *" $c "*) ;; *) continue ;; esac          # not a target city
  [ "$(count_for "$c")" -ge "$PER_CITY" ] && continue           # city already full
  echo "$log $c" >> "$PICKS"
  GOT=$((GOT + 1))
  echo "  picked $log → $c"
done

echo "Selected $GOT log(s):"; sed 's/^/  /' "$PICKS"
[ "$GOT" -lt "$NEED" ] && echo "  (warning: only $GOT/$NEED found in $SPLIT — some city may be short)"

# ── 2. download (selective) → extract → cleanup, per pick ────────────────────
while read -r log city; do
  [ -z "${log:-}" ] && continue
  short="${log:0:8}"
  out="$OUT_DIR/argoverse-$short"
  raw="$WORK/$log"
  echo "── $city · $log → argoverse-$short ──"
  rm -rf "$out"          # clean rebuild — avoid orphaned content-addressed packs
  mkdir -p "$raw/sensors"
  aws_cp "$BUCKET/$SPLIT/$log/annotations.feather"             "$raw/"
  aws_cp "$BUCKET/$SPLIT/$log/city_SE3_egovehicle.feather"     "$raw/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/map/"                "$raw/map/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/lidar/"      "$raw/sensors/lidar/"
  aws_cp --recursive "$BUCKET/$SPLIT/$log/sensors/cameras/$CAMERA/" \
                                                               "$raw/sensors/cameras/$CAMERA/"
  "$PY" "$HERE/argoverse_extract.py" --log-dir "$raw" --city "$city" \
        --out "$out" --camera "$CAMERA" --stt-build "$STT_BUILD"
  rm -rf "$raw"
  echo "  done argoverse-$short ($city); raw log removed"
done < "$PICKS"

echo "All done. Bundles under $OUT_DIR/argoverse-*"
[ -n "${KEEP_WORK:-}" ] || rm -rf "$WORK"
