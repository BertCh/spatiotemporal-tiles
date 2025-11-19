#!/bin/bash

# Rebuild all datasets with gzip compression for browser compatibility

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

echo -e "\n🔄 Rebuilding COVID data with gzip..."
if [ -f "examples/showcase/public/data/covid-cases.geojson" ]; then
  ./target/release/stt-build \
    --input examples/showcase/public/data/covid-cases.geojson \
    --output examples/showcase/public/data/covid-cases.stt \
    --time-field timestamp \
    --temporal-resolution daily-aggregates \
    --min-zoom 0 \
    --max-zoom 14 \
    --compression gzip
  echo "✅ COVID cases rebuilt"
else
  echo "⚠️  Skipping COVID - no source GeoJSON"
fi

echo -e "\n✅ Rebuild complete! All datasets now use gzip compression."

