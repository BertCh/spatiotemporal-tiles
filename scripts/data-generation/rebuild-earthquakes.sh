#!/bin/bash
# Rebuild earthquake data with proper temporal resolution

set -e

echo "🌍 Rebuilding Earthquake Data with Temporal Resolution"
echo "========================================================="
echo ""

cd "$(dirname "$0")"

# Check if earthquakes.geojson exists, generate if not
if [ ! -f "data/earthquakes.geojson" ]; then
    echo "⚠️  data/earthquakes.geojson not found, generating..."
    cargo run --release --bin generate-earthquake-data -- --output data/earthquakes.geojson
fi

echo "📦 Building STT archive with sparse-events profile..."
echo ""

stt-build \
    --input data/earthquakes.geojson \
    --output ../../examples/showcase/public/data/earthquakes.stt \
    --time-field timestamp \
    --temporal-resolution sparse-events \
    --min-zoom 0 \
    --max-zoom 10 \
    --compression gzip

echo ""
echo "✅ Earthquake data rebuilt successfully!"
echo ""
echo "Temporal resolution profile: sparse-events"
echo "  • Zoom 0-4: Monthly buckets"
echo "  • Zoom 5-8: Weekly buckets"  
echo "  • Zoom 9+: Daily buckets"
echo ""
echo "📁 Output: ../../examples/showcase/public/data/earthquakes.stt"
echo ""
echo "🎯 Next: Refresh your browser to see earthquakes with auto-configuration!"

