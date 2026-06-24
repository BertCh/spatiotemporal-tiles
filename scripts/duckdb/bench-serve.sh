#!/usr/bin/env bash
# Serve benchmark: dynamic on-the-fly DuckDB tile generation (stt-serve) vs the
# SAME tiles served pre-baked as static files. Measures the live-source trade-off.
#
# Steps:
#   1. sample real points (from the baseline parquet) -> (z,x,y,t) tile paths
#   2. materialize each non-empty tile once from stt-serve -> a static dir
#   3. load-test stt-serve (cold pass, then warm pass) and the static file server
#
# Prereqs: stt-serve already running at $SERVE_URL against hurricane.duckdb, e.g.
#   target/release/stt-serve --duckdb scratch-duckdb/hurricane.duckdb \
#       --table hurricane_obs --geom-column geom --time-field iso_time \
#       --temporal-bucket 7d --min-zoom 3 --max-zoom 8
#
# Usage: scripts/duckdb/bench-serve.sh
# Env: SERVE_URL, STATIC_PORT, CONC, ZOOMS, BUCKET_MS, SAMPLE, PARQUET
set -euo pipefail

SERVE_URL="${SERVE_URL:-http://127.0.0.1:8088}"
STATIC_PORT="${STATIC_PORT:-8099}"
CONC="${CONC:-16}"
ZOOMS="${ZOOMS:-4,5,6,7,8}"
BUCKET_MS="${BUCKET_MS:-604800000}"   # 7d, must match the running server
SAMPLE="${SAMPLE:-400}"
SCRATCH="${SCRATCH:-$(pwd)/scratch-duckdb}"
# Sample real points from the IBTrACS CSV with the stdlib only (no pyarrow /
# duckdb python needed) — post-1970 so the t buckets match servable data.
CSV="${CSV:-data/ibtracs.csv}"
# The serve-side helpers are backend-agnostic — reuse the PostGIS ones.
PG="$(cd "$(dirname "$0")/../postgis" && pwd)"
WORK="$SCRATCH/serve"
mkdir -p "$WORK"

echo "==> Sampling $SAMPLE points from $CSV -> tile request list (zooms $ZOOMS)"
python3 - "$CSV" "$SAMPLE" > "$WORK/sample.csv" <<'PY'
import sys, csv, datetime as dt
src, n = sys.argv[1], int(sys.argv[2])
out = []
with open(src, newline="") as fh:
    r = csv.DictReader(fh)
    next(r, None)  # IBTrACS has a units row right after the header
    for row in r:
        try:
            lon, lat = float(row["LON"]), float(row["LAT"])
            t = dt.datetime.strptime(row["ISO_TIME"].strip(), "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            continue
        if t.year < 1970:
            continue
        out.append((lon, lat, int(t.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)))
print("lon,lat,ms")
step = max(1, len(out) // n)
for i in range(0, len(out), step):
    lon, lat, ms = out[i]
    print(f"{lon},{lat},{ms}")
PY
python3 "$PG/gen_tile_urls.py" "$WORK/sample.csv" "$BUCKET_MS" "$ZOOMS" > "$WORK/urls.all.txt"
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
python3 "$PG/bench_serve.py" "$SERVE_URL" "$WORK/urls.txt" "$CONC" "DYNAMIC DuckDB (cold pass)"
echo ""
python3 "$PG/bench_serve.py" "$SERVE_URL" "$WORK/urls.txt" "$CONC" "DYNAMIC DuckDB (warm pass)"
echo ""
python3 "$PG/bench_serve.py" "http://127.0.0.1:$STATIC_PORT" "$WORK/urls.txt" "$CONC" "STATIC pre-baked files"
echo "==================================================="
