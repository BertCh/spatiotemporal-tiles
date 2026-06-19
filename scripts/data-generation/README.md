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

Most datasets auto-download from their source; a few take a local `--input`
(e.g. BIXI). Run `stt-generate <subcommand> --help` for per-dataset flags.

| Dataset | Source | Command |
|---------|--------|---------|
| Earthquakes | USGS API | `stt-generate earthquakes` |
| AIS Maritime | NOAA Marine Cadastre | `stt-generate ais --date 2024-01-01` |
| Flight Traffic | OpenSky Network | `stt-generate flights --date 2020-01-06` |
| Hurricanes | NOAA IBTrACS | `stt-generate hurricanes` |
| Wildfires | NIFC | `stt-generate wildfires` |
| NYC Rideshare | TLC + OSRM | `stt-generate nyc-rideshare --download 2016-01` |
| NYC Taxi Points | derived from rideshare | `stt-generate nyc-taxi-points` |
| BIXI Flowmap | BIXI Montréal open data | `stt-generate bixi --input <csv>` |
| Satellites | CelesTrak TLE | `stt-generate satellites` |
| Ocean Drifters | NOAA GDP | `stt-generate drifters` (or `drifters-hourly`) |
| Animal Migration | Movebank | `stt-generate animals` |
| OSM Edits | OSM changesets | `stt-generate osm-edits` |
| Storm Radar | NEXRAD Level II | `stt-generate storms` |

Run `stt-generate --help` for the authoritative subcommand list.

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

### Python adapters (research / domain pipelines)

These produce GeoParquet for `stt-build` outside the Rust generators:

- `download_ecco.py` / `ecco_advect.py` - ECCO ocean-current particle advection
  (see [ECCO.md](./ECCO.md))
- `av_common.py`, `av_synthetic.py`, `nuscenes_extract.py`, `comma_extract.py`,
  `argoverse_extract.py`, `waymo_extract.py` - AV-cockpit scene adapters
  (see [docs/roadmap/av-cockpit.md](../../docs/roadmap/av-cockpit.md))
- `argoverse_batch.sh` - builds one (or `PER_CITY=N`) Argoverse 2 cockpit scene
  per AV2 city straight from the public `s3://argoverse` bucket (no login):
  selectively downloads each log (lidar + ego + annotations + map + one ring
  camera), runs `argoverse_extract.py`, and deletes the raw log. Needs the
  `venv-av2` python + a release `stt-build`. Run `bash argoverse_batch.sh`.
- `waymo_extract.py` / `waymo_batch.sh` - builds the curated Waymo Open Dataset
  (Perception **v2.0.1**, the *modular Parquet* release) cockpit scenes. Reads the
  Parquet components with pyarrow and decodes the 5-laser LIDAR range images in
  pure numpy — **no TensorFlow / waymo-open-dataset lib** (its wheels are
  Linux-only). Deviations from the contract: no HD map (v2.0.1 ships none) and no
  Waymo georeferencing (poses are an arbitrary local frame), so each scene is
  anchored to an APPROXIMATE local lat/lon by metro and rides the cockpit's dark
  basemap. License-gated: accept the Waymo Dataset License Agreement
  (**non-commercial, no redistribution**) at waymo.com/open, then `gcloud auth
  login`. Needs the `venv-waymo` python (`pyarrow numpy shapely pandas`) + a
  release `stt-build`. Run `bash waymo_batch.sh` (`FORCE=1` to rebuild).

## Notes

- NYC Rideshare: TLC data after June 2016 uses location IDs instead of coordinates. Use `--download 2015-01` through `2016-06` for actual lat/long data.
- Flights: OpenSky historical data is only available for Mondays from 2017-2020.
- AIS: Data files are ~500MB-2GB per day (compressed).
