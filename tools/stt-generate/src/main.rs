//! stt-generate - Unified CLI for generating spatiotemporal tile datasets
//!
//! This tool downloads, processes, and builds STT archives for showcase datasets.
//! Each dataset has its own subcommand with sensible defaults.
//!
//! For custom data, use `stt-build` directly.

mod common;
mod datasets;
mod edge_bundle;
mod radar;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "stt-generate")]
#[command(about = "Generate spatiotemporal tile datasets", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate earthquake data from USGS API
    Earthquakes(datasets::earthquakes::Args),

    /// Generate AIS maritime traffic data from NOAA Marine Cadastre
    Ais(datasets::ais::Args),

    /// Generate flight traffic data from OpenSky Network
    Flights(datasets::flights::Args),

    /// Generate hurricane track data from NOAA IBTrACS
    Hurricanes(datasets::hurricanes::Args),

    /// Generate wildfire perimeter data from NIFC
    Wildfires(datasets::wildfires::Args),

    /// Generate NYC rideshare trajectory data from TLC records + OSRM routing
    NycRideshare(datasets::nyc_rideshare::Args),

    /// Generate a Montreal BIXI origin→destination flowmap from open-data trips
    Bixi(datasets::bixi::Args),

    /// Generate a country-scale transit "ballet" from a static GTFS feed (one service date)
    Gtfs(datasets::gtfs::Args),

    /// Generate NWM river-discharge corridors on the NHDPlus CONUS network
    Nwm(datasets::nwm::Args),

    /// Derive a NYC taxi POINT dataset by interpolating an existing path .stt
    NycTaxiPoints(datasets::nyc_taxi_points::Args),

    /// Generate satellite orbit data from CelesTrak TLE
    Satellites(datasets::satellites::Args),

    /// Generate ocean-current trajectories from NOAA's Global Drifter Program
    Drifters(datasets::drifters::Args),

    /// EXPERIMENTAL: GDP HOURLY ocean-current trajectories (drifter_hourly_qc)
    DriftersHourly(datasets::drifters_hourly::Args),

    /// Generate animal-migration trajectories from GBIF tracking datasets
    Animals(datasets::animals::Args),

    /// Generate an OSM editing-history dataset (node creations or changesets)
    OsmEdits(datasets::osm_edits::Args),

    /// Generate NEXRAD storm-radar tiles for the 2020-08-10 Iowa derecho
    Storms(datasets::storms::Args),
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Earthquakes(args) => datasets::earthquakes::run(args),
        Commands::Ais(args) => datasets::ais::run(args),
        Commands::Flights(args) => datasets::flights::run(args),
        Commands::Hurricanes(args) => datasets::hurricanes::run(args),
        Commands::Wildfires(args) => datasets::wildfires::run(args),
        Commands::NycRideshare(args) => datasets::nyc_rideshare::run(args),
        Commands::Bixi(args) => datasets::bixi::run(args),
        Commands::Gtfs(args) => datasets::gtfs::run(args),
        Commands::Nwm(args) => datasets::nwm::run(args),
        Commands::NycTaxiPoints(args) => datasets::nyc_taxi_points::run(args),
        Commands::Satellites(args) => datasets::satellites::run(args),
        Commands::Drifters(args) => datasets::drifters::run(args),
        Commands::DriftersHourly(args) => datasets::drifters_hourly::run(args),
        Commands::Animals(args) => datasets::animals::run(args),
        Commands::OsmEdits(args) => datasets::osm_edits::run(args),
        Commands::Storms(args) => datasets::storms::run(args),
    }
}
