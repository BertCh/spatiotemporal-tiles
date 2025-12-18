#!/bin/bash
# Generate all showcase datasets using stt-generate
#
# This script wraps the unified stt-generate CLI tool.
# For individual datasets, use stt-generate directly:
#   stt-generate earthquakes --output earthquakes.stt
#   stt-generate ais --input ais.csv --output ais.stt
#   stt-generate flights --date 2020-01-06 --output flights.stt

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "🚀 STT Data Generation Pipeline"
echo "================================"
echo ""

# Build stt-generate if not already built
echo "🔨 Building stt-generate..."
cargo build --release -p stt-generate

OUTPUT_DIR="examples/showcase/public/data"

echo ""
echo "📊 Generating datasets..."
echo ""

# Run the unified tool
./target/release/stt-generate all --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ All datasets generated successfully!"
echo ""
echo "📁 Output location: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  cd examples/showcase"
echo "  npm install"
echo "  npm run dev"
echo ""
echo "For individual datasets or custom options, use stt-generate directly:"
echo "  stt-generate --help"
echo "  stt-generate earthquakes --help"
echo "  stt-generate ais --help"
