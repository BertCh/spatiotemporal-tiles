#!/bin/bash
# Generate a week of AIS data for San Francisco Bay
# Non-interactive script for automated processing

set -e

echo "🚢 AIS Data Generator - San Francisco Bay (7 Days)"
echo "=================================================="
echo ""

# Configuration
REGION_NAME="sf-bay"
BOUNDS="36.5,-123.5,38.5,-121.5"  # San Francisco Bay + approaches
MAX_VESSELS=3000
SAMPLE_MINUTES=10
NUM_DAYS=7

DATA_DIR="data/marineCadastre"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Directory $DATA_DIR not found"
    exit 1
fi

# Count available CSV files
CSV_FILES=($(find "$DATA_DIR" -name "AIS_2023_01_*.csv" -type f | sort))
CSV_COUNT=${#CSV_FILES[@]}

echo "📊 Found $CSV_COUNT AIS CSV files in $DATA_DIR"
echo "📍 Region: San Francisco Bay"
echo "📐 Bounds: $BOUNDS"
echo "🚢 Max vessels: $MAX_VESSELS"
echo "⏱️  Sampling: 1 position per vessel per $SAMPLE_MINUTES minutes"
echo "📅 Days to process: $NUM_DAYS"
echo ""

if [ "$CSV_COUNT" -lt "$NUM_DAYS" ]; then
    echo "⚠️  Only $CSV_COUNT files available, processing all of them"
    NUM_DAYS=$CSV_COUNT
fi

# Create temporary directory for intermediate GeoJSON files
TEMP_DIR="data/temp_ais_sf"
mkdir -p "$TEMP_DIR"
mkdir -p "metadata"

# Process each CSV file
PROCESSED=0
for i in $(seq 0 $((NUM_DAYS - 1))); do
    CSV_FILE="${CSV_FILES[$i]}"
    FILENAME=$(basename "$CSV_FILE")
    
    echo "[$((i + 1))/$NUM_DAYS] Processing $FILENAME..."
    
    # Extract date from filename (AIS_2023_01_01.csv)
    if [[ $FILENAME =~ AIS_([0-9]{4})_([0-9]{2})_([0-9]{2})\.csv ]]; then
        YEAR="${BASH_REMATCH[1]}"
        MONTH="${BASH_REMATCH[2]}"
        DAY="${BASH_REMATCH[3]}"
        DATE_STR="$YEAR-$MONTH-$DAY"
        OUTPUT_FILE="$TEMP_DIR/ais-$DATE_STR-$REGION_NAME.geojson"
        
        # Run the processor
        ../../target/release/generate-ais-data \
            --input "$CSV_FILE" \
            --output "$OUTPUT_FILE" \
            --sample-minutes "$SAMPLE_MINUTES" \
            --bounds "$BOUNDS" \
            --max-vessels "$MAX_VESSELS"
        
        if [ $? -eq 0 ]; then
            PROCESSED=$((PROCESSED + 1))
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            FEATURE_COUNT=$(grep -o '"type": "Feature"' "$OUTPUT_FILE" | wc -l | tr -d ' ')
            echo "  ✅ Created $OUTPUT_FILE ($FILE_SIZE, $FEATURE_COUNT features)"
        else
            echo "  ⚠️  Failed to process $FILENAME"
        fi
    else
        echo "  ⚠️  Skipping $FILENAME (invalid format)"
    fi
    
    echo ""
done

echo "✅ Processed $PROCESSED files"
echo ""

if [ "$PROCESSED" -eq 0 ]; then
    echo "❌ No files were processed successfully"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Merge all GeoJSON files using Python
echo "🔗 Merging GeoJSON files..."
MERGED_FILE="data/ais-week-$REGION_NAME.geojson"

python3 << 'PYTHON_SCRIPT'
import json
import glob
from datetime import datetime
from pathlib import Path

temp_dir = "data/temp_ais_sf"
output_file = "data/ais-week-sf-bay.geojson"

print(f"Reading GeoJSON files from {temp_dir}...")
files = sorted(glob.glob(f"{temp_dir}/ais-*.geojson"))
print(f"Found {len(files)} files to merge")

all_features = []
min_time = None
max_time = None
vessel_ids = set()

for idx, file_path in enumerate(files, 1):
    print(f"  [{idx}/{len(files)}] Reading {Path(file_path).name}...", end="")
    with open(file_path) as f:
        data = json.load(f)
        features = data.get('features', [])
        
        # Track statistics
        for feature in features:
            props = feature.get('properties', {})
            timestamp = props.get('timestamp')
            mmsi = props.get('mmsi')
            
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if min_time is None or ts < min_time:
                        min_time = ts
                    if max_time is None or ts > max_time:
                        max_time = ts
                except:
                    pass
            
            if mmsi:
                vessel_ids.add(mmsi)
        
        all_features.extend(features)
        print(f" {len(features)} features")

print(f"\n📊 Statistics:")
print(f"  Total features: {len(all_features)}")
print(f"  Unique vessels: {len(vessel_ids)}")
if min_time and max_time:
    print(f"  Time range: {min_time.isoformat()} to {max_time.isoformat()}")
    duration = max_time - min_time
    print(f"  Duration: {duration.days} days, {duration.seconds // 3600} hours")

# Sort features by timestamp
print("\nSorting by timestamp...")
all_features.sort(key=lambda f: f.get('properties', {}).get('timestamp', ''))

merged = {
    'type': 'FeatureCollection',
    'features': all_features
}

print(f"\nWriting merged file to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(merged, f)

print(f"✅ Merged file created: {output_file}")

# Write statistics to metadata file
metadata = {
    'dataset': 'AIS Maritime Traffic',
    'region': 'San Francisco Bay',
    'source': 'NOAA Marine Cadastre',
    'feature_count': len(all_features),
    'unique_vessels': len(vessel_ids),
    'time_range': {
        'start': min_time.isoformat() if min_time else None,
        'end': max_time.isoformat() if max_time else None
    },
    'bounds': [36.5, -123.5, 38.5, -121.5]
}

with open('metadata/ais-week-sf-bay.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata written to metadata/ais-week-sf-bay.json")
PYTHON_SCRIPT

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

# Ensure stt-build is built
if [ ! -f "../../target/release/stt-build" ]; then
    echo "Building stt-build..."
    cd ../..
    cargo build --release --bin stt-build
    cd scripts/data-generation
fi

../../target/release/stt-build \
    --input "$MERGED_FILE" \
    --output "$STT_OUTPUT" \
    --time-field timestamp \
    --temporal-resolution high-frequency \
    --min-zoom 0 \
    --max-zoom 14 \
    --compression gzip \
    --metadata-output "metadata/ais-$REGION_NAME.stt.meta.json"

echo ""
echo "✅ Complete!"
echo ""
echo "📁 Output files:"
echo "   GeoJSON: $MERGED_FILE"
echo "   STT: $STT_OUTPUT"
echo "   Metadata: metadata/ais-week-sf-bay.json"
echo ""
echo "📊 File sizes:"
ls -lh "$MERGED_FILE" "$STT_OUTPUT" 2>/dev/null | awk '{print "   " $9 ": " $5}'
echo ""
echo "🎉 Success! You can now view the AIS data in the showcase:"
echo "   cd ../../examples/showcase"
echo "   npm run dev"
echo ""




