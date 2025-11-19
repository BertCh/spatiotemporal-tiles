#!/bin/bash
# Download and process real AIS data from NOAA Marine Cadastre
# Usage: ./download-ais.sh YYYY MM DD

set -e

YEAR=${1:-2024}
MONTH=${2:-01}
DAY=${3:-01}

# Pad with zeros
MONTH=$(printf "%02d" $MONTH)
DAY=$(printf "%02d" $DAY)

DATE="${YEAR}-${MONTH}-${DAY}"
FILENAME="AIS_${YEAR}_${MONTH}_${DAY}.zip"
URL="https://coast.noaa.gov/htdata/CMSP/AISDataHandler/${YEAR}/${FILENAME}"

echo "🚢 NOAA Marine Cadastre AIS Data Downloader"
echo "==========================================="
echo ""
echo "📅 Date: ${DATE}"
echo "📂 File: ${FILENAME}"
echo "🌐 URL: ${URL}"
echo ""

# Create data directory
mkdir -p data

# Download if not exists
if [ ! -f "data/${FILENAME}" ]; then
    echo "⬇️  Downloading AIS data..."
    curl -L -o "data/${FILENAME}" "${URL}" --progress-bar
    echo "✅ Download complete"
else
    echo "✅ File already exists: data/${FILENAME}"
fi

# Unzip
echo ""
echo "📦 Extracting CSV..."
CSV_FILE="data/AIS_${YEAR}_${MONTH}_${DAY}.csv"

if [ ! -f "${CSV_FILE}" ]; then
    unzip -o "data/${FILENAME}" -d data/
    echo "✅ Extracted to: ${CSV_FILE}"
else
    echo "✅ CSV already extracted: ${CSV_FILE}"
fi

# Check file size
FILE_SIZE=$(du -h "${CSV_FILE}" | cut -f1)
echo ""
echo "📊 CSV file size: ${FILE_SIZE}"
echo ""

# Process with geographic bounds
echo "🔄 Processing AIS data..."
echo ""
echo "Choose a region:"
echo "  1) US East Coast (25°N-45°N, 80°W-65°W)"
echo "  2) Gulf of Mexico (25°N-30°N, 97°W-80°W)"
echo "  3) San Francisco Bay (37°N-38°N, 123°W-122°W)"
echo "  4) Entire dataset (no bounds)"
echo ""
read -p "Select region [1-4]: " REGION

case $REGION in
    1)
        BOUNDS="25.0,-80.0,45.0,-65.0"
        REGION_NAME="east-coast"
        ;;
    2)
        BOUNDS="25.0,-97.0,30.0,-80.0"
        REGION_NAME="gulf"
        ;;
    3)
        BOUNDS="37.0,-123.0,38.0,-122.0"
        REGION_NAME="sf-bay"
        ;;
    4)
        BOUNDS=""
        REGION_NAME="all"
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac

OUTPUT_FILE="data/ais-${YEAR}-${MONTH}-${DAY}-${REGION_NAME}.geojson"

if [ -z "$BOUNDS" ]; then
    ../../target/release/generate-ais-data \
        --input "${CSV_FILE}" \
        --output "${OUTPUT_FILE}" \
        --sample-minutes 10 \
        --max-vessels 5000
else
    ../../target/release/generate-ais-data \
        --input "${CSV_FILE}" \
        --output "${OUTPUT_FILE}" \
        --sample-minutes 10 \
        --bounds "${BOUNDS}" \
        --max-vessels 5000
fi

echo ""
echo "✅ GeoJSON created: ${OUTPUT_FILE}"
echo ""
echo "📦 Building STT archive..."

../../target/release/stt-build \
    --input "${OUTPUT_FILE}" \
    --output "../../examples/showcase/public/data/ais-${REGION_NAME}.stt" \
    --time-field timestamp \
    --temporal-resolution daily-aggregates \
    --min-zoom 0 \
    --max-zoom 14 \
    --compression gzip

echo ""
echo "✅ Complete! STT file: ../../examples/showcase/public/data/ais-${REGION_NAME}.stt"
echo ""
echo "📝 Update examples/showcase/src/datasets.ts:"
echo "   - Change ship-traffic URL to: /data/ais-${REGION_NAME}.stt"
echo "   - Update time range to: ${DATE}"
echo ""

