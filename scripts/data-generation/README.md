# Data Generation Scripts

Rust-based scripts for downloading and processing real-world datasets into STT format.

## Overview

These scripts download public datasets, process them into GeoJSON with temporal metadata, and prepare them for conversion to STT archives using `stt-build`.

## Available Scripts

### ✅ Implemented

#### 1. COVID-19 Cases (`generate-covid-data`)

Downloads county-level COVID-19 case data from the New York Times.

**Data Source**: [NYT COVID-19 Data](https://github.com/nytimes/covid-19-data)

```bash
cargo run --bin generate-covid-data -- --output covid-cases.geojson

# Convert to STT
stt-build \
  --input covid-cases.geojson \
  --output ../examples/showcase/public/data/covid-cases.stt \
  --time-field timestamp \
  --min-zoom 0 \
  --max-zoom 14 \
  --compression brotli
```

**Output**: ~1.2M features, 730 time steps (2020-2021)

#### 2. Earthquake Activity (`generate-earthquake-data`)

Fetches global earthquake data from USGS.

**Data Source**: [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/)

```bash
cargo run --bin generate-earthquake-data -- \
  --output earthquakes.geojson \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --min-magnitude 4.0

# Convert to STT
stt-build \
  --input earthquakes.geojson \
  --output ../examples/showcase/public/data/earthquakes.stt \
  --time-field timestamp \
  --min-zoom 0 \
  --max-zoom 10 \
  --compression brotli
```

**Output**: ~35K features, 5 years

#### 3. San Francisco Taxis (`generate-taxi-data`)

Generates synthetic taxi trajectories (realistic simulation).

```bash
cargo run --bin generate-taxi-data -- \
  --output sf-taxis.geojson \
  --num-taxis 100 \
  --date 2024-01-15 \
  --interval 60

# Convert to STT
stt-build \
  --input sf-taxis.geojson \
  --output ../examples/showcase/public/data/sf-taxis.stt \
  --time-field timestamp \
  --min-zoom 10 \
  --max-zoom 16 \
  --compression brotli
```

**Output**: ~144K trajectory points, 24 hours

### 🚧 To Be Implemented

#### Maritime Traffic (`generate-ship-data`)

**Status:** Script exists with synthetic data

**Real Data Source:** NOAA Marine Cadastre (Free, public access)

- **URL:** https://coast.noaa.gov/htdata/CMSP/AISDataHandler
- **Format:** ZIP compressed CSV (AIS_YYYY_MM_DD.zip)
- **Coverage:** US coastal waters, 2009-Present
- **Size:** ~200-300 MB compressed per day (~800 MB uncompressed)
- **Fields:** MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, VesselType, etc.

```bash
# Download AIS data for January 1, 2023
curl -o AIS_2023_01_01.zip \
  https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip

# Unzip
unzip AIS_2023_01_01.zip

# TODO: Add CSV to GeoJSON conversion script
# Then convert to STT
stt-build \
  --input ship-traffic.geojson \
  --output ../examples/showcase/public/data/ships.stt \
  --time-field timestamp \
  --min-zoom 0 \
  --max-zoom 8 \
  --compression brotli
```

**Current Workaround:** Use the synthetic data generator:

```bash
cargo run --bin generate-ship-data -- \
  --output ship-traffic.geojson \
  --start-date 2023-01-01 \
  --days 7 \
  --num-ships 500
```

#### Flight Density (`generate-flight-data`)

**Status:** Script exists with synthetic data

**Real Data Source:** ADSBExchange (Free for 1st of each month)

- **URL:** https://samples.adsbexchange.com/hires-traces
- **Format:** Gzip-compressed JSON traces (per aircraft)
- **Coverage:** Global, 2020-Present
- **Organization:** Files bucketed by ICAO prefix (00-ff)
- **Size:** 50-500 KB per aircraft per day
- **Availability:** Only the 1st of each month is freely available

```bash
# Download trace for aircraft a12345 on January 1, 2025
# Files are organized in buckets by ICAO prefix (first 2 hex digits)
curl -o trace_a12345.json \
  https://samples.adsbexchange.com/hires-traces/2025/01/01/a1/trace_full_~a12345.json

# Note: You must know ICAO codes in advance
# Use ADSBExchange index files to get lists of active aircraft

# TODO: Add JSON trace to GeoJSON conversion script
# Then convert to STT
stt-build \
  --input flight-density.geojson \
  --output ../examples/showcase/public/data/flights.stt \
  --time-field timestamp \
  --min-zoom 3 \
  --max-zoom 8 \
  --compression brotli
```

**Current Workaround:** Use the synthetic data generator:

```bash
cargo run --bin generate-flight-data -- \
  --output flight-density.geojson \
  --date 2025-01-01 \
  --num-flights 1000
```

#### Other Datasets

- `generate-hurricane-data`: NOAA hurricane tracks (partially implemented)
- `generate-wildfire-data`: California wildfire perimeters
- `generate-bikeshare-data`: NYC Citi Bike trips

## Quick Start

### Build All Scripts

```bash
cd scripts/data-generation
cargo build --release
```

### Generate All Datasets

```bash
# Run the complete pipeline
./generate-all.sh
```

Or individually:

```bash
# COVID-19 data
cargo run --release --bin generate-covid-data

# Earthquake data
cargo run --release --bin generate-earthquake-data

# Taxi trajectories
cargo run --release --bin generate-taxi-data
```

### Convert to STT Archives

```bash
# After generating GeoJSON files, convert them
cd ../../

# COVID-19
stt-build --input scripts/data-generation/covid-cases.geojson \
          --output examples/showcase/public/data/covid-cases.stt \
          --time-field timestamp --compression brotli

# Earthquakes
stt-build --input scripts/data-generation/earthquakes.geojson \
          --output examples/showcase/public/data/earthquakes.stt \
          --time-field timestamp --compression brotli

# Taxis
stt-build --input scripts/data-generation/sf-taxis.geojson \
          --output examples/showcase/public/data/sf-taxis.stt \
          --time-field timestamp --compression brotli
```

## Architecture

### Common Utilities (`src/common.rs`)

Shared functionality:

- `download_file()`: HTTP downloads with progress bars
- `create_point_feature()`: GeoJSON point creation
- `create_linestring_feature()`: GeoJSON line creation
- `write_geojson()`: Write FeatureCollection to file
- `parse_date()`: Date parsing utilities
- `unzip_file()`: ZIP extraction

### Script Structure

Each script follows a consistent pattern:

```rust
// 1. Parse command-line arguments
let args = Args::parse();

// 2. Download raw data
download_file(source_url, &raw_data_path)?;

// 3. Process into GeoJSON features
let features = process_data(&raw_data_path)?;

// 4. Write GeoJSON output
write_geojson(features, &args.output)?;

// 5. Show stt-build command
println!("Now run: stt-build --input {} ...", args.output);
```

## Data Sources

### COVID-19

- **Source**: [New York Times COVID-19 Data](https://github.com/nytimes/covid-19-data)
- **License**: CC BY-NC 4.0
- **Format**: CSV (date, county, state, fips, cases, deaths)
- **Updates**: Daily
- **Note**: Requires separate county coordinate file

### Earthquakes

- **Source**: [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/)
- **License**: Public Domain
- **Format**: GeoJSON via API
- **Updates**: Real-time
- **Note**: API rate limits apply (10K records per request)

### Maritime Traffic (AIS)

- **Source**: [NOAA Marine Cadastre](https://marinecadastre.gov/ais/)
- **License**: Public Domain (US Government)
- **Format**: ZIP compressed CSV files
- **Coverage**: US coastal waters, 2009-Present
- **URL Pattern**: `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{YEAR}/AIS_{YEAR}_{MM}_{DD}.zip`
- **File Size**: ~200-300 MB compressed per day (~800 MB uncompressed)
- **Update Frequency**: Daily files
- **Records per Day**: ~17,000-20,000 vessels

**Key Fields:**

- `MMSI` - Maritime Mobile Service Identity (vessel ID)
- `BaseDateTime` - UTC timestamp
- `LAT`, `LON` - Position (decimal degrees)
- `SOG` - Speed Over Ground (knots)
- `COG` - Course Over Ground (degrees)
- `Heading` - True heading (degrees)
- `VesselName`, `VesselType` - Vessel information
- `Length`, `Width`, `Draft` - Vessel dimensions (meters)
- `Status` - Navigation status

**Advantages:**

- ✅ Free, unrestricted access
- ✅ Complete daily coverage
- ✅ Historical data back to 2009
- ✅ High-quality government data

**Limitations:**

- ⚠️ US coastal waters only
- ⚠️ Large file sizes
- ⚠️ Some vessels disable transponders

### Flight Density (ADS-B)

- **Source**: [ADSBExchange](https://www.adsbexchange.com/data-samples/)
- **License**: Community-driven (check terms)
- **Format**: Gzip-compressed JSON traces (per aircraft)
- **Coverage**: Global, 2020-Present
- **URL Pattern**: `https://samples.adsbexchange.com/hires-traces/{YEAR}/{MM}/{DD}/{BUCKET}/trace_full_~{ICAO}.json`
- **File Organization**: 256 buckets (00-ff) by ICAO prefix
- **File Size**: 50-500 KB per aircraft per day
- **Availability**: Only the 1st of each month is freely available
- **Aircraft per Day**: ~5,000-8,000 (varies by date)

**Trace Structure:**

```json
{
  "icao": "a12345",      // 6-character hex ICAO code
  "r": "N12345",         // Registration
  "t": "B738",           // Aircraft type
  "trace": [
    [timestamp, lat, lon, alt, gs, track, ...],
    ...
  ]
}
```

**Trace Fields:**

- Index 0: Unix timestamp (seconds)
- Index 1: Latitude (decimal degrees)
- Index 2: Longitude (decimal degrees)
- Index 3: Altitude (feet, barometric)
- Index 4: Ground speed (knots)
- Index 5: Track angle (degrees)

**Advantages:**

- ✅ Global coverage
- ✅ High precision tracking
- ✅ Historical data back to 2020
- ✅ Per-aircraft trace files

**Limitations:**

- ⚠️ Only 1st of each month free
- ⚠️ Must download per-aircraft traces (requires ICAO code list)
- ⚠️ Coverage varies by region
- ⚠️ More complex data structure

### Taxis (Synthetic)

- **Source**: Generated algorithmically
- **Method**: Random walk simulation with realistic constraints
- **Parameters**: Configurable taxi count, date, update interval
- **Note**: For demonstration purposes; use real AIS/GPS data in production

## Data Processing Pipeline

### AIS (Maritime) Processing

```bash
# 1. Download raw data
curl -o AIS_2023_01_01.zip \
  https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip

# 2. Unzip
unzip AIS_2023_01_01.zip

# 3. Convert CSV to GeoJSON (custom script needed)
# Process ~800 MB CSV into GeoJSON with temporal metadata
# Expected processing: ~45 seconds (M1 Max, ~180K records/sec)

# 4. Build STT archive
stt-build \
  --input ship-traffic.geojson \
  --output ships.stt \
  --time-field timestamp \
  --min-zoom 0 \
  --max-zoom 8 \
  --compression brotli

# Result: ~80 MB compressed (10:1 ratio)
```

### ADS-B (Aviation) Processing

```bash
# 1. Download traces for multiple aircraft
# Example: Download 100 aircraft from bucket "a1" on Jan 1, 2025
for icao in $(cat icao_list.txt); do
  bucket=${icao:0:2}
  curl -o "trace_${icao}.json" \
    "https://samples.adsbexchange.com/hires-traces/2025/01/01/${bucket}/trace_full_~${icao}.json"
done

# 2. Convert JSON traces to GeoJSON (custom script needed)
# Aggregate multiple trace files into unified GeoJSON
# Expected processing: ~60-90 seconds for 1 day (~200K positions/sec)

# 3. Build STT archive
stt-build \
  --input flight-density.geojson \
  --output flights.stt \
  --time-field timestamp \
  --min-zoom 3 \
  --max-zoom 8 \
  --compression brotli

# Result: ~120 MB compressed (15:1+ ratio)
```

### Storage Requirements

**Raw Data (1 month):**

- AIS: ~24 GB compressed, ~240 GB uncompressed
- ADS-B: ~60-150 GB (varies by coverage)

**Processed STT Archives (1 month):**

- AIS: ~2.4 GB (H3-tiled, compressed)
- ADS-B: ~3.6 GB (H3-tiled, compressed)

**Recommended Storage:**

- Development: 500 GB (raw + processed for several months)
- Production: 50 GB per year (processed data only)

## Adding New Datasets

### 1. Create New Script File

```bash
touch src/your-dataset.rs
```

### 2. Implement the Script

```rust
mod common;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Download data
    common::download_file(source_url, &temp_file)?;

    // Process into features
    let features = process_data(&temp_file)?;

    // Write GeoJSON
    common::write_geojson(features, &args.output)?;

    Ok(())
}

fn process_data(path: &PathBuf) -> Result<Vec<Feature>> {
    // Your processing logic here
    Ok(vec![])
}
```

### 3. Add Binary to Cargo.toml

```toml
[[bin]]
name = "generate-your-dataset"
path = "src/your-dataset.rs"
```

### 4. Test It

```bash
cargo run --bin generate-your-dataset -- --output test.geojson
```

## Performance Tips

### Parallel Processing

Use Rayon for CPU-intensive operations:

```rust
use rayon::prelude::*;

let features: Vec<Feature> = records
    .par_iter()
    .map(|record| process_record(record))
    .collect();
```

### Memory Management

For large datasets, use streaming:

```rust
// Instead of loading all into memory
let mut features = Vec::new();
for record in csv_reader.deserialize() {
    features.push(process_record(record?));

    // Write in batches
    if features.len() >= 10000 {
        write_batch(&features)?;
        features.clear();
    }
}
```

### Progress Indicators

All scripts use `indicatif` for progress bars:

```rust
use indicatif::{ProgressBar, ProgressStyle};

let pb = ProgressBar::new(total_count);
pb.set_style(
    ProgressStyle::default_bar()
        .template("[{bar:40}] {pos}/{len} ({eta})")?
);

for item in items {
    process(item);
    pb.inc(1);
}

pb.finish_with_message("Complete!");
```

## Troubleshooting

### Download Fails

```bash
# Use cached data
cargo run --bin generate-covid-data -- --cached
```

### Out of Memory

```bash
# Process in smaller chunks
cargo run --bin generate-covid-data -- --chunk-size 10000
```

### Slow Processing

```bash
# Build with optimizations
cargo build --release
cargo run --release --bin generate-covid-data
```

## License

Scripts are MIT licensed. Individual datasets have their own licenses:

- COVID-19 data: CC BY-NC 4.0 (NYT)
- Earthquake data: Public Domain (USGS)
- Synthetic data: MIT

---

**Questions?** Open an issue or check the [main documentation](../../README.md).
