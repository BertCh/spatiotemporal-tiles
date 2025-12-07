#!/bin/bash
# Generate comprehensive AIS dataset using CSV format (memory-efficient)
# Processes all available CSV files without geographic filtering

set -e

echo "🚢 AIS Data Generator - All US Coastal Waters (CSV Mode)"
echo "========================================================="
echo ""

DATA_DIR="data/marineCadastre"
TEMP_DIR="data/temp_ais_csv"
OUTPUT_CSV="data/ais-all-us.csv"
STT_OUTPUT="../../examples/showcase/public/data/ais-all-us.stt"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Directory $DATA_DIR not found"
    exit 1
fi

# Configuration
SAMPLE_MINUTES=10
MAX_VESSELS=0  # Unlimited

# Count available CSV files
CSV_FILES=($(find "$DATA_DIR" -name "AIS_2023_*.csv" -type f | sort))
CSV_COUNT=${#CSV_FILES[@]}

echo "📊 Found $CSV_COUNT AIS CSV files in $DATA_DIR"
echo "🌎 Region: All US Coastal Waters (no geographic filtering)"
echo "🚢 Max vessels: Unlimited"
echo "⏱️  Sampling: 1 position per vessel per $SAMPLE_MINUTES minutes"
echo "📝 Output format: CSV (streaming, memory-efficient)"
echo ""

# Create temp directory
mkdir -p "$TEMP_DIR"
mkdir -p "metadata"

# Build the project first
echo "🔨 Building AIS processor..."
cd ../..
cargo build --release --bin generate-ais-data
cd scripts/data-generation

# Process each CSV file
PROCESSED=0
for i in $(seq 0 $((CSV_COUNT - 1))); do
    CSV_FILE="${CSV_FILES[$i]}"
    FILENAME=$(basename "$CSV_FILE")
    
    echo "[$((i + 1))/$CSV_COUNT] Processing $FILENAME..."
    
    # Extract date from filename (AIS_2023_01_01.csv)
    if [[ $FILENAME =~ AIS_([0-9]{4})_([0-9]{2})_([0-9]{2})\.csv ]]; then
        YEAR="${BASH_REMATCH[1]}"
        MONTH="${BASH_REMATCH[2]}"
        DAY="${BASH_REMATCH[3]}"
        DATE_STR="$YEAR-$MONTH-$DAY"
        OUTPUT_FILE="$TEMP_DIR/ais-$DATE_STR.csv"
        
        # Run the processor with CSV output
        ../../target/release/generate-ais-data \
            --input "$CSV_FILE" \
            --output "$OUTPUT_FILE" \
            --sample-minutes "$SAMPLE_MINUTES" \
            --max-vessels "$MAX_VESSELS"
        
        if [ $? -eq 0 ]; then
            PROCESSED=$((PROCESSED + 1))
            ROW_COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
            echo "  ✅ Created $OUTPUT_FILE ($ROW_COUNT rows)"
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

# Merge all CSV files
echo "🔗 Merging CSV files..."

# Get header from first file
FIRST_CSV=$(find "$TEMP_DIR" -name "ais-*.csv" | sort | head -1)
head -1 "$FIRST_CSV" > "$OUTPUT_CSV"

# Concatenate all files (skip header for each)
TOTAL_ROWS=0
for csv in $(find "$TEMP_DIR" -name "ais-*.csv" | sort); do
    ROWS=$(tail -n +2 "$csv" | wc -l | tr -d ' ')
    TOTAL_ROWS=$((TOTAL_ROWS + ROWS))
    tail -n +2 "$csv" >> "$OUTPUT_CSV"
    echo "  Added $(basename $csv): $ROWS rows"
done

echo ""
echo "📊 Total rows: $TOTAL_ROWS"
echo "📊 File size:"
ls -lh "$OUTPUT_CSV"
echo ""

# Clean up temporary files
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo ""

# Build STT archive
echo "📦 Building STT archive..."

../../target/release/stt-build \
    --input "$OUTPUT_CSV" \
    --output "$STT_OUTPUT" \
    --time-field timestamp \
    --min-zoom 0 \
    --max-zoom 10 \
    --compression gzip \
    --metadata-output "metadata/ais-all-us.stt.meta.json"

echo ""
echo "✅ Complete!"
echo ""
echo "📁 Output files:"
echo "   CSV: $OUTPUT_CSV"
echo "   STT: $STT_OUTPUT"
echo "   Metadata: metadata/ais-all-us.stt.meta.json"
echo ""
echo "📊 File sizes:"
ls -lh "$OUTPUT_CSV" "$STT_OUTPUT" 2>/dev/null | awk '{print "   " $9 ": " $5}'
echo ""



