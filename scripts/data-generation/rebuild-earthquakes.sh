#!/bin/bash
# Rebuild earthquake data

set -e

echo "🌍 Rebuilding Earthquake Data"
echo "=============================="
echo ""

cd "$(dirname "$0")"

# Check if earthquakes.geojson exists, generate if not
if [ ! -f "data/earthquakes.geojson" ]; then
    echo "⚠️  data/earthquakes.geojson not found, generating..."
    cargo run --release --bin generate-earthquake-data -- --output data/earthquakes.geojson
fi

echo "📦 Building STT archive..."
echo ""

stt-build \
    --input data/earthquakes.geojson \
    --output ../../examples/showcase/public/data/earthquakes.stt \
    --time-field timestamp \
    --min-zoom 0 \
    --max-zoom 10 \
    --compression gzip

echo ""
echo "✅ Earthquake data rebuilt successfully!"
echo ""
echo "📁 Output: ../../examples/showcase/public/data/earthquakes.stt"
echo ""
echo "🎯 Next: Refresh your browser to see the updated earthquakes!"
