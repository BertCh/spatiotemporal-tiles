# Data Generation

Use the unified `stt-generate` CLI tool to download and process datasets.

## Quick Start

```bash
# Install the tool
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

## Utility Scripts

- `generate-all.sh` - Convenience wrapper to generate all datasets
- `setup-osrm.sh` - Set up OSRM server for NYC routing (required for nyc-rideshare)
- `generate-datasets-config.js` - Generate TypeScript config from metadata
- `validate-ais-coords.js` - Validate AIS coordinate data

## Data Files

- `data/` - Downloaded and processed data files (gitignored)
- `metadata/` - Dataset metadata JSON files
