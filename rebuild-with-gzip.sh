#!/bin/bash

# Rebuild all datasets with gzip compression for browser compatibility
# NOTE: This script is deprecated. Use stt-generate instead:
#   stt-generate all --output-dir examples/showcase/public/data

cd /Users/robertchristie/Documents/GitHub/spatiotemporal-tiles

echo "🔄 Rebuilding earthquake data with gzip..."
if [ -f "examples/showcase/public/data/earthquakes.geojson" ]; then
  ./target/release/stt-build \
    --input examples/showcase/public/data/earthquakes.geojson \
    --output examples/showcase/public/data/earthquakes.stt \
    --time-field timestamp \
    --temporal-resolution sparse-events \
    --min-zoom 0 \
    --max-zoom 10 \
    --compression gzip
  echo "✅ Earthquakes rebuilt"
else
  echo "⚠️  Skipping earthquakes - no source GeoJSON"
fi

echo -e "\n✅ Rebuild complete! All datasets now use gzip compression."

