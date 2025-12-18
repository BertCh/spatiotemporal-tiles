# Data Generation Scripts

> **Note:** This directory contains legacy Rust scripts that have been consolidated into the unified `stt-generate` CLI tool.

## Recommended: Use stt-generate

The `stt-generate` tool provides a unified interface for generating all showcase datasets:

```bash
# Install
cargo install --path ../../crates/stt-generate

# Generate all datasets
stt-generate all --output-dir ../../examples/showcase/public/data

# Generate individual datasets
stt-generate earthquakes --output earthquakes.stt
stt-generate hurricanes --output hurricanes.stt
stt-generate wildfires --output wildfires.stt
stt-generate ais --input ais.csv --output ais.stt
stt-generate flights --date 2020-01-06 --output flights.stt
```

See the [Data Generation Guide](../../docs/guides/data-generation.md) for full documentation.

## Legacy Scripts (Deprecated)

The Rust binaries in `src/` are still available but deprecated:

```bash
# Build legacy scripts
cargo build --release

# Run legacy scripts
cargo run --release --bin generate-earthquake-data -- --output earthquakes.geojson
cargo run --release --bin generate-ais-data -- --input ais.csv --output ais.geojson
```

These scripts output GeoJSON/CSV files that then need to be processed with `stt-build`.

## Available Datasets

| Dataset | Source | Command |
|---------|--------|---------|
| Earthquakes | USGS API | `stt-generate earthquakes` |
| AIS Maritime | NOAA Marine Cadastre | `stt-generate ais --input <csv>` |
| Flight Traffic | OpenSky Network | `stt-generate flights` |
| Hurricanes | NOAA IBTrACS | `stt-generate hurricanes` |
| Wildfires | NIFC | `stt-generate wildfires` |
| NYC Rideshare | TLC + OSRM | `stt-generate nyc-rideshare` |

## Custom Data

For data not covered by built-in datasets, use `stt-build` directly:

```bash
stt-build --input my-data.geojson --output my-data.stt --time-field timestamp
```

## Setup Files

- `setup-osrm.sh` - Set up OSRM server for NYC routing (required for nyc-rideshare with real routing)
- `generate-all.sh` - Wrapper script to generate all datasets using stt-generate
