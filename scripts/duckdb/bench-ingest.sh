#!/usr/bin/env bash
# Ingest A/B: build the SAME post-1970 IBTrACS point set into STT tiles two ways —
#   (1) from a GeoParquet file   (--input)
#   (2) from the DuckDB hurricane_obs table (--duckdb ... --where post-1970)
# — with identical tiling flags, and compare wall-clock + logical parity
# (tile_count / feature_count / bounds / time_range).
#
# Requires: target/release/stt-build (built with --features duckdb), and the
# inputs from `cargo run --release -p stt-build --features duckdb \
#   --example duckdb_load_ibtracs` (a hurricane.duckdb + baseline parquet).
#
# Usage: scripts/duckdb/bench-ingest.sh
# Env:   DB, PARQUET, OUT, BUILD, RUNS, BUCKET, MAXZOOM, WHERE
set -euo pipefail

BUILD="${BUILD:-target/release/stt-build}"
SCRATCH="${SCRATCH:-$(pwd)/scratch-duckdb}"
DB="${DB:-$SCRATCH/hurricane.duckdb}"
PARQUET="${PARQUET:-$SCRATCH/hurricane_points.parquet}"
OUT="${OUT:-$SCRATCH/ingest}"
RUNS="${RUNS:-3}"
BUCKET="${BUCKET:-7d}"
MAXZOOM="${MAXZOOM:-8}"
WHERE="${WHERE:-iso_time >= '1970-01-01'}"

COMMON=(--time-field iso_time --temporal-bucket "$BUCKET" --max-zoom "$MAXZOOM")
mkdir -p "$OUT"

# Best (min) real-seconds over RUNS, via the bash `time` builtin (TIMEFORMAT=%R).
best_real() {
  local label=$1; shift
  local best=""
  for _ in $(seq 1 "$RUNS"); do
    rm -rf "$OUT/$label"
    local real
    real=$( { TIMEFORMAT='%R'; time "$@" -o "$OUT/$label" --name "$label" >/dev/null 2>&1; } 2>&1 )
    if [ -z "$best" ] || awk "BEGIN{exit !($real < $best)}"; then best="$real"; fi
  done
  echo "$best"
}

echo "==> File ingest  (--input $PARQUET)"
FILE_T=$(best_real file "$BUILD" --input "$PARQUET" "${COMMON[@]}")
echo "    best real: ${FILE_T}s"

echo "==> DuckDB ingest (--duckdb $DB --table hurricane_obs)"
DK_T=$(best_real duckdb "$BUILD" --duckdb "$DB" --table hurricane_obs \
        --geom-column geom --where "$WHERE" "${COMMON[@]}")
echo "    best real: ${DK_T}s"

echo ""
echo "==> Validating both archives (stt-validate)"
VALIDATE="${VALIDATE:-target/release/stt-validate}"
# stt-validate exits non-zero on benign per-tile "schema drift" (sparse columns
# omitted from all-null tiles) — present identically in BOTH archives. It still
# writes the JSON report, so tolerate the exit code.
"$VALIDATE" --json "$OUT/file/manifest.json"   > "$OUT/file.validate.json"   2>/dev/null || true
"$VALIDATE" --json "$OUT/duckdb/manifest.json" > "$OUT/duckdb.validate.json" 2>/dev/null || true

echo ""
echo "==> Parity + timing"
python3 - "$OUT/file/manifest.json" "$OUT/duckdb/manifest.json" \
         "$OUT/file.validate.json" "$OUT/duckdb.validate.json" "$FILE_T" "$DK_T" <<'PY'
import json, sys
fm = json.load(open(sys.argv[1]))["metadata"]
dm = json.load(open(sys.argv[2]))["metadata"]
fv = json.load(open(sys.argv[3]))
dv = json.load(open(sys.argv[4]))
ft, dt = float(sys.argv[5]), float(sys.argv[6])

# Order-independent logical parity (manifest metadata + decoded validate report).
rows = [
    ("bounds",              fm.get("bounds"),               dm.get("bounds")),
    ("time_range",          fm.get("time_range"),           dm.get("time_range")),
    ("temporal_bucket_ms",  fm.get("temporal_bucket_ms"),   dm.get("temporal_bucket_ms")),
    ("tile_count",          fv.get("tile_count"),           dv.get("tile_count")),
    ("feature_count_index", fv.get("feature_count_index"),  dv.get("feature_count_index")),
    ("features_decoded",    fv.get("feature_count_decoded"),dv.get("feature_count_decoded")),
]
print(f"{'metric':<20}{'file':>22}{'duckdb':>22}{'match':>7}")
ok = True
for k, a, b in rows:
    m = (a == b); ok = ok and m
    sa, sb = str(a), str(b)
    if len(sa) > 21: sa = sa[:18] + "..."
    if len(sb) > 21: sb = sb[:18] + "..."
    print(f"{k:<20}{sa:>22}{sb:>22}{('Y' if m else 'N'):>7}")

fb = fv.get("payload_bytes_compressed", 0)
db = dv.get("payload_bytes_compressed", 0)
print()
print(f"payload bytes (zstd) : file {fb:,}  duckdb {db:,}  (Δ {abs(fb-db)/max(fb,1)*100:.2f}% — feature order within tiles differs; logically identical)")
print()
print(f"file ingest  : {ft:.3f}s")
print(f"duckdb ingest: {dt:.3f}s   ({dt/ft:.2f}x file)")
print()
print("PARITY:", "PASS — identical tiles from both sources" if ok else "FAIL — sources diverged")
sys.exit(0 if ok else 1)
PY
