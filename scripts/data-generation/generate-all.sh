#!/bin/bash
# Generate all showcase datasets

set -e

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 STT Data Generation Pipeline"
echo "================================"
echo ""

# Create output directories (relative to this script)
mkdir -p data
mkdir -p metadata
mkdir -p ../../examples/showcase/public/data

# COVID-19 Cases
echo "📊 1/6 Generating COVID-19 data..."
cargo run --release --bin generate-covid-data -- --output data/covid-cases.geojson
cargo run -p stt-build --release --bin stt-build -- --input data/covid-cases.geojson \
          --output ../../examples/showcase/public/data/covid-cases.stt \
          --time-field timestamp \
          --min-zoom 0 \
          --max-zoom 14 \
          --compression gzip \
          --metadata-output metadata/covid-cases.meta.json
echo "✓ COVID-19 data complete"
echo ""

# Earthquakes
echo "🌍 2/6 Generating earthquake data..."
cargo run --release --bin generate-earthquake-data -- \
  --output data/earthquakes.geojson \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --min-magnitude 4.0
cargo run -p stt-build --release --bin stt-build -- --input data/earthquakes.geojson \
          --output ../../examples/showcase/public/data/earthquakes.stt \
          --time-field timestamp \
          --min-zoom 0 \
          --max-zoom 10 \
          --compression gzip \
          --metadata-output metadata/earthquake-activity.meta.json
echo "✓ Earthquake data complete"
echo ""

# Ships / AIS Maritime Traffic
echo "🚢 3/6 Generating ship traffic data..."
cargo run --release --bin generate-ship-data -- \
  --output data/ships.geojson \
  --start-date 2024-01-01 \
  --days 7 \
  --num-ships 500
cargo run -p stt-build --release --bin stt-build -- --input data/ships.geojson \
          --output ../../examples/showcase/public/data/ships.stt \
          --time-field timestamp \
          --min-zoom 0 \
          --max-zoom 12 \
          --compression gzip \
          --metadata-output metadata/ship-traffic.meta.json
echo "✓ Ship traffic data complete"
echo ""

# Hurricanes
echo "🌀 4/6 Generating hurricane track data..."
cargo run --release --bin generate-hurricane-data -- \
  --output data/hurricanes.geojson
cargo run -p stt-build --release --bin stt-build -- --input data/hurricanes.geojson \
          --output ../../examples/showcase/public/data/hurricanes.stt \
          --time-field timestamp \
          --min-zoom 0 \
          --max-zoom 8 \
          --compression gzip \
          --metadata-output metadata/hurricanes.meta.json
echo "✓ Hurricane data complete"
echo ""

# Flights
echo "✈️  5/6 Generating flight traffic data..."
cargo run --release --bin generate-flight-data -- \
  --output data/flights.geojson
cargo run -p stt-build --release --bin stt-build -- --input data/flights.geojson \
          --output ../../examples/showcase/public/data/flights.stt \
          --time-field timestamp \
          --min-zoom 0 \
          --max-zoom 10 \
          --compression gzip \
          --metadata-output metadata/flights.meta.json
echo "✓ Flight data complete"
echo ""

# San Francisco Taxis
echo "🚕 6/6 Generating taxi trajectory data..."
cargo run --release --bin generate-taxi-data -- \
  --output data/sf-taxis.geojson \
  --num-taxis 100 \
  --date 2024-01-15 \
  --interval 60
cargo run -p stt-build --release --bin stt-build -- --input data/sf-taxis.geojson \
          --output ../../examples/showcase/public/data/sf-taxis.stt \
          --time-field timestamp \
          --min-zoom 10 \
          --max-zoom 16 \
          --compression gzip \
          --metadata-output metadata/sf-taxis.meta.json
echo "✓ Taxi data complete"
echo ""

echo "✅ All datasets generated successfully!"
echo ""
echo "📝 Generating datasets.ts configuration..."
node generate-datasets-config.js metadata ../../examples/showcase/src/datasets.ts
echo ""
echo "📁 Output location: ../../examples/showcase/public/data/"
echo ""
echo "Next steps:"
echo "  cd ../../examples/showcase"
echo "  npm install"
echo "  npm run dev"
