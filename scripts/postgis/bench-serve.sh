#!/usr/bin/env bash
# Serve benchmark: dynamic on-the-fly PostGIS tile generation (stt-serve) vs the
# SAME tiles served pre-baked as static files. Measures the live-DB trade-off.
#
# Steps:
#   1. sample real points -> a list of (z,x,y,t) tile paths that contain data
#   2. materialize each tile once from stt-serve -> a static dir (keep only 200s)
#   3. load-test stt-serve (cold pass, then warm pass) and the static file server
#
# Prereqs: stt-serve already running at $SERVE_URL against the loaded table.
# Usage: scripts/postgis/bench-serve.sh
# Env: PGURL, SERVE_URL, STATIC_PORT, CONC, ZOOMS, BUCKET_MS, SAMPLE, CONTAINER
set -euo pipefail

PGURL="${PGURL:-postgresql://postgres:postgres@localhost:5432/stt}"
SERVE_URL="${SERVE_URL:-http://127.0.0.1:8088}"
STATIC_PORT="${STATIC_PORT:-8099}"
CONC="${CONC:-16}"
ZOOMS="${ZOOMS:-4,5,6,7,8}"
BUCKET_MS="${BUCKET_MS:-604800000}"   # 7d, must match the running server
SAMPLE="${SAMPLE:-400}"
CONTAINER="${CONTAINER:-stt-postgis}"
SCRATCH="${SCRATCH:-$(pwd)/scratch-postgis}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$SCRATCH/serve"
mkdir -p "$WORK"

echo "==> Sampling $SAMPLE points -> tile request list (zooms $ZOOMS)"
docker exec "$CONTAINER" psql -U postgres -d stt -tAc \
  "\copy (SELECT lon, lat, (extract(epoch from iso_time)*1000)::bigint AS ms
          FROM hurricane_obs WHERE iso_time >= '1970-01-01'
          ORDER BY md5(sid || iso_time::text) LIMIT $SAMPLE)
   TO STDOUT WITH (FORMAT csv, HEADER true)" > "$WORK/sample.csv"
python3 "$HERE/gen_tile_urls.py" "$WORK/sample.csv" "$BUCKET_MS" "$ZOOMS" > "$WORK/urls.all.txt"
echo "    $(wc -l < "$WORK/urls.all.txt") candidate tiles"

echo "==> Materializing tiles from stt-serve (keeping non-empty 200s)"
rm -rf "$WORK/static"; : > "$WORK/urls.txt"
empty=0
while IFS= read -r path; do
  code=$(curl -s -o "$WORK/static$path" --create-dirs -w '%{http_code}' "$SERVE_URL$path")
  if [ "$code" = "200" ]; then
    echo "$path" >> "$WORK/urls.txt"
  else
    rm -f "$WORK/static$path"; empty=$((empty+1))
  fi
done < "$WORK/urls.all.txt"
echo "    $(wc -l < "$WORK/urls.txt") non-empty tiles materialized ($empty empty/204)"
echo "    total pre-baked size: $(du -sh "$WORK/static" | cut -f1)"

echo "==> Static file server on :$STATIC_PORT"
( cd "$WORK/static" && exec python3 -m http.server "$STATIC_PORT" --bind 127.0.0.1 ) \
  > "$WORK/static-server.log" 2>&1 &
STATIC_PID=$!
trap 'kill $STATIC_PID 2>/dev/null || true' EXIT
sleep 1

echo ""
echo "================= SERVE BENCHMARK ================="
echo "(N tiles = $(wc -l < "$WORK/urls.txt"), concurrency = $CONC)"
echo ""
# Dynamic: first pass is cold-ish (PG plan/cache warming), second is warm.
python3 "$HERE/bench_serve.py" "$SERVE_URL" "$WORK/urls.txt" "$CONC" "DYNAMIC PostGIS (cold pass)"
echo ""
python3 "$HERE/bench_serve.py" "$SERVE_URL" "$WORK/urls.txt" "$CONC" "DYNAMIC PostGIS (warm pass)"
echo ""
python3 "$HERE/bench_serve.py" "http://127.0.0.1:$STATIC_PORT" "$WORK/urls.txt" "$CONC" "STATIC pre-baked files"
echo "==================================================="
