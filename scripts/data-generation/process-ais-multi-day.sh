#!/bin/bash
# Process multiple days of AIS data from NOAA Marine Cadastre
# This script processes all available CSV files and merges them into a comprehensive dataset

set -e

echo "🚢 Multi-Day AIS Data Processor"
echo "==============================="
echo ""

# Navigate to the data directory
DATA_DIR="data/marineCadastre"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Directory $DATA_DIR not found"
    exit 1
fi

# Count available CSV files
CSV_COUNT=$(find "$DATA_DIR" -name "AIS_*.csv" | wc -l | tr -d ' ')
echo "📊 Found $CSV_COUNT AIS CSV files"
echo ""

# Ask for region
echo "Choose a region to filter by:"
echo "  1) San Francisco Bay (37°N-38°N, 123°W-122°W)"
echo "  2) Los Angeles/Long Beach (33°N-34°N, 119°W-117.5°W)"
echo "  3) Seattle/Puget Sound (47°N-48.5°N, 123°W-122°W)"
echo "  4) Gulf of Mexico (25°N-30°N, 97°W-80°W)"
echo "  5) US East Coast (25°N-45°N, 80°W-65°W)"
echo "  6) Entire US Waters (no bounds, may be very large)"
echo ""
read -p "Select region [1-6]: " REGION

case $REGION in
    1)
        BOUNDS="37.0,-123.0,38.0,-122.0"
        REGION_NAME="sf-bay"
        MAX_VESSELS=2000
        ;;
    2)
        BOUNDS="33.0,-119.0,34.0,-117.5"
        REGION_NAME="la-port"
        MAX_VESSELS=3000
        ;;
    3)
        BOUNDS="47.0,-123.0,48.5,-122.0"
        REGION_NAME="seattle"
        MAX_VESSELS=2000
        ;;
    4)
        BOUNDS="25.0,-97.0,30.0,-80.0"
        REGION_NAME="gulf"
        MAX_VESSELS=5000
        ;;
    5)
        BOUNDS="25.0,-80.0,45.0,-65.0"
        REGION_NAME="east-coast"
        MAX_VESSELS=8000
        ;;
    6)
        BOUNDS=""
        REGION_NAME="all"
        MAX_VESSELS=10000
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac

echo ""
echo "📍 Region: $REGION_NAME"
if [ -n "$BOUNDS" ]; then
    echo "📐 Bounds: $BOUNDS"
fi
echo "🚢 Max vessels: $MAX_VESSELS"
echo ""

# Ask how many days to process
echo "How many days of data to process?"
echo "  (Found $CSV_COUNT CSV files available)"
echo ""
read -p "Number of days [1-$CSV_COUNT]: " NUM_DAYS

if [ "$NUM_DAYS" -lt 1 ] || [ "$NUM_DAYS" -gt "$CSV_COUNT" ]; then
    echo "Invalid number of days"
    exit 1
fi

echo ""
echo "⏱️  Processing $NUM_DAYS days of AIS data for region: $REGION_NAME"
echo ""

# Create temporary directory for intermediate GeoJSON files
TEMP_DIR="data/temp_ais"
mkdir -p "$TEMP_DIR"

# Build the project first
echo "🔨 Building AIS processor..."
cd ../..
cargo build --release --bin generate-ais-data
cd scripts/data-generation

# Process each CSV file
COUNTER=0
PROCESSED=0

# Sort CSV files by date
for CSV_FILE in $(find "$DATA_DIR" -name "AIS_*.csv" | sort); do
    if [ "$COUNTER" -ge "$NUM_DAYS" ]; then
        break
    fi
    
    COUNTER=$((COUNTER + 1))
    FILENAME=$(basename "$CSV_FILE")
    
    echo "[$COUNTER/$NUM_DAYS] Processing $FILENAME..."
    
    # Extract date from filename (AIS_2023_01_01.csv)
    DATE_STR=$(echo "$FILENAME" | sed 's/AIS_\([0-9]*\)_\([0-9]*\)_\([0-9]*\)\.csv/\1-\2-\3/')
    OUTPUT_FILE="$TEMP_DIR/ais-$DATE_STR-$REGION_NAME.geojson"
    
    # Run the processor
    if [ -z "$BOUNDS" ]; then
        ../../target/release/generate-ais-data \
            --input "$CSV_FILE" \
            --output "$OUTPUT_FILE" \
            --sample-minutes 15 \
            --max-vessels "$MAX_VESSELS"
    else
        ../../target/release/generate-ais-data \
            --input "$CSV_FILE" \
            --output "$OUTPUT_FILE" \
            --sample-minutes 15 \
            --bounds "$BOUNDS" \
            --max-vessels "$MAX_VESSELS"
    fi
    
    if [ $? -eq 0 ]; then
        PROCESSED=$((PROCESSED + 1))
        echo "  ✅ Created $OUTPUT_FILE"
    else
        echo "  ⚠️  Failed to process $FILENAME"
    fi
    
    echo ""
done

echo "✅ Processed $PROCESSED files"
echo ""

# Merge all GeoJSON files
echo "🔗 Merging GeoJSON files..."
MERGED_FILE="data/ais-multi-day-$REGION_NAME.geojson"

# Use Node.js to merge the GeoJSON files
node -e "
const fs = require('fs');
const glob = require('glob');

console.log('Reading GeoJSON files...');
const files = glob.sync('$TEMP_DIR/ais-*.geojson');
console.log(\`Found \${files.length} files to merge\`);

let allFeatures = [];
let minTime = Infinity;
let maxTime = -Infinity;

files.forEach((file, idx) => {
    console.log(\`  [\${idx+1}/\${files.length}] Reading \${file}...\`);
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    
    // Track time range
    data.features.forEach(f => {
        const ts = f.properties.timestamp;
        if (ts) {
            const time = new Date(ts).getTime();
            if (time < minTime) minTime = time;
            if (time > maxTime) maxTime = time;
        }
    });
    
    allFeatures = allFeatures.concat(data.features);
});

console.log(\`\nTotal features: \${allFeatures.length}\`);
console.log(\`Time range: \${new Date(minTime).toISOString()} to \${new Date(maxTime).toISOString()}\`);

// Sort features by timestamp
console.log('Sorting by timestamp...');
allFeatures.sort((a, b) => {
    const ta = new Date(a.properties.timestamp).getTime();
    const tb = new Date(b.properties.timestamp).getTime();
    return ta - tb;
});

const merged = {
    type: 'FeatureCollection',
    features: allFeatures
};

console.log('Writing merged file...');
fs.writeFileSync('$MERGED_FILE', JSON.stringify(merged));
console.log('✅ Merged file created: $MERGED_FILE');
" 2>/dev/null || {
    # Fallback: simple Python merger if Node.js fails
    python3 -c "
import json
import glob
from pathlib import Path

print('Reading GeoJSON files...')
files = sorted(glob.glob('$TEMP_DIR/ais-*.geojson'))
print(f'Found {len(files)} files to merge')

all_features = []
for idx, file in enumerate(files):
    print(f'  [{idx+1}/{len(files)}] Reading {file}...')
    with open(file) as f:
        data = json.load(f)
        all_features.extend(data['features'])

print(f'\nTotal features: {len(all_features)}')

# Sort by timestamp
print('Sorting by timestamp...')
all_features.sort(key=lambda f: f['properties'].get('timestamp', ''))

merged = {
    'type': 'FeatureCollection',
    'features': all_features
}

print('Writing merged file...')
with open('$MERGED_FILE', 'w') as f:
    json.dump(merged, f)

print('✅ Merged file created: $MERGED_FILE')
"
}

echo ""
echo "📊 File size:"
du -h "$MERGED_FILE"
echo ""

# Clean up temporary files
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo ""

# Build STT archive
echo "📦 Building STT archive..."
STT_OUTPUT="../../examples/showcase/public/data/ais-$REGION_NAME.stt"

../../target/release/stt-build \
    --input "$MERGED_FILE" \
    --output "$STT_OUTPUT" \
    --time-field timestamp \
    --min-zoom 0 \
    --max-zoom 14 \
    --compression gzip \
    --metadata-output "metadata/ais-$REGION_NAME.meta.json"

echo ""
echo "✅ Complete!"
echo ""
echo "📁 Output files:"
echo "   GeoJSON: $MERGED_FILE"
echo "   STT: $STT_OUTPUT"
echo ""
echo "📊 To view the data:"
echo "   cd ../../examples/showcase"
echo "   npm run dev"
echo ""
echo "💡 Don't forget to update examples/showcase/src/datasets.ts:"
echo "   - Add or update the AIS dataset entry"
echo "   - URL: /data/ais-$REGION_NAME.stt"
echo ""




