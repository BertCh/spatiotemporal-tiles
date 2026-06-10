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
stt-generate ais --date 2024-01-01 --output ais.stt
stt-generate flights --date 2020-01-06 --output flights.stt
stt-generate satellites --output satellites.stt
```

See the [Data Generation Guide](../../docs/guides/data-generation.md) for full documentation.

## Available Datasets

All datasets support automatic downloading from their respective sources:

| Dataset | Source | Command |
|---------|--------|---------|
| Earthquakes | USGS API | `stt-generate earthquakes` |
| AIS Maritime | NOAA Marine Cadastre | `stt-generate ais --date 2024-01-01` |
| Flight Traffic | OpenSky Network | `stt-generate flights --date 2020-01-06` |
| Hurricanes | NOAA IBTrACS | `stt-generate hurricanes` |
| Wildfires | NIFC | `stt-generate wildfires` |
| NYC Rideshare | TLC + OSRM | `stt-generate nyc-rideshare --download 2016-01` |
| Satellites | CelesTrak TLE | `stt-generate satellites` |

## Date Range Examples

Several datasets support downloading specific dates or date ranges:

```bash
# AIS: Download a single day
stt-generate ais --date 2024-01-01 --output ais-day.stt

# AIS: Download a week of data
stt-generate ais --start-date 2024-01-01 --end-date 2024-01-07 --output ais-week.stt

# AIS: Use existing CSV file
stt-generate ais --input data/AIS_2024_01_01.csv --output ais.stt

# Flights: Download specific date (Mondays from 2017-2020)
stt-generate flights --date 2020-01-06 --hours 0-12 --output flights.stt

# NYC Rideshare: Download TLC data (pre-July 2016 for lat/long)
stt-generate nyc-rideshare --download 2016-01 --max-trips 10000 --output nyc.stt

# NYC Taxi Flow (pre-aggregated overview): route once keeping the paths
# intermediate, then aggregate into time-binned road-segment flow corridors.
# Re-run step 2 alone to re-tune --flow-bin without re-routing through OSRM.
stt-generate nyc-rideshare --input yellow_tripdata_2015-01.csv --paths \
  --chronological --max-trips 500000 --output nyc-taxi-paths-2015-01.parquet
stt-generate nyc-rideshare --flows --flow-bin 15m \
  --from-intermediate nyc-taxi-paths-2015-01.parquet --output nyc-taxi-flows.stt
```

## Custom Data

For data not covered by built-in datasets, use `stt-build` directly.
`stt-build` accepts **GeoParquet only** — convert other formats first
(`ogr2ogr -f Parquet out.parquet in.geojson`, or see the
[Python guide](../../docs/guides/python.md)):

```bash
stt-build \
  --input my-data.parquet --output my-data.stt \
  --time-field timestamp --time-format unix-ms \
  --auto
```

## Data Files

Downloaded data is cached locally and gitignored:

- `data/ais/` - AIS data from NOAA Marine Cadastre
- `data/opensky-historical/` - Flight data from OpenSky Network  
- `data/` - Other cached downloads

## Utility Scripts

- `generate-all.sh` - Convenience wrapper to generate all datasets
- `setup-osrm.sh` - Set up OSRM server for NYC routing (optional for nyc-rideshare)
- `generate-datasets-config.js` - Generate TypeScript config from metadata
- `validate-ais-coords.js` - Validate AIS coordinate data

## Notes

- NYC Rideshare: TLC data after June 2016 uses location IDs instead of coordinates. Use `--download 2015-01` through `2016-06` for actual lat/long data.
- Flights: OpenSky historical data is only available for Mondays from 2017-2020.
- AIS: Data files are ~500MB-2GB per day (compressed).
