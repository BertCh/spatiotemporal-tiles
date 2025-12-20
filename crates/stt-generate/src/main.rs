//! stt-generate - Unified CLI for generating spatiotemporal tile datasets
//!
//! This tool downloads, processes, and builds STT archives for showcase datasets.
//! Each dataset has its own subcommand with sensible defaults.
//!
//! For custom data, use `stt-build` directly.

mod common;
mod datasets;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    /// Generate all showcase datasets
    All {
        /// Output directory for STT files
        #[arg(short, long, default_value = "examples/showcase/public/data")]
        output_dir: PathBuf,

        /// Skip datasets that already exist
        #[arg(long)]
        skip_existing: bool,
    },

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

    /// Generate satellite orbit data from CelesTrak TLE
    Satellites(datasets::satellites::Args),
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::All { output_dir, skip_existing } => {
            run_all(&output_dir, skip_existing)
        }
        Commands::Earthquakes(args) => datasets::earthquakes::run(args),
        Commands::Ais(args) => datasets::ais::run(args),
        Commands::Flights(args) => datasets::flights::run(args),
        Commands::Hurricanes(args) => datasets::hurricanes::run(args),
        Commands::Wildfires(args) => datasets::wildfires::run(args),
        Commands::NycRideshare(args) => datasets::nyc_rideshare::run(args),
        Commands::Satellites(args) => datasets::satellites::run(args),
    }
}

/// Run all dataset generators
fn run_all(output_dir: &PathBuf, skip_existing: bool) -> Result<()> {
    use std::process::Command;

    println!("🚀 STT Data Generation Pipeline");
    println!("================================\n");

    // Ensure output directory exists
    std::fs::create_dir_all(output_dir)?;

    let datasets = [
        ("earthquakes", "earthquakes.stt"),
        ("hurricanes", "hurricanes.stt"),
        ("wildfires", "wildfires.stt"),
    ];

    for (i, (name, filename)) in datasets.iter().enumerate() {
        let output_path = output_dir.join(filename);

        if skip_existing && output_path.exists() {
            println!("[{}/{}] ⏭️  Skipping {} (already exists)", i + 1, datasets.len(), name);
            continue;
        }

        println!("[{}/{}] 📊 Generating {}...", i + 1, datasets.len(), name);

        // Run the subcommand
        let status = Command::new(std::env::current_exe()?)
            .arg(name)
            .arg("--output")
            .arg(&output_path)
            .status()?;

        if !status.success() {
            println!("⚠️  {} generation failed, continuing...", name);
        } else {
            println!("✅ {} complete\n", name);
        }
    }

    // Note: Some datasets require specific dates or additional setup
    println!("\n📝 Note: Some datasets require additional parameters:");
    println!("   - ais: Specify --date YYYY-MM-DD to download from NOAA Marine Cadastre");
    println!("   - flights: Specify --date YYYY-MM-DD (Mondays 2017-2020 only)");
    println!("   - nyc-rideshare: Use --download YYYY-MM or --synthetic");
    println!("   - satellites: Uses CelesTrak TLE data");
    println!("\nRun individual commands with --help for options.");

    println!("\n✅ Dataset generation complete!");
    println!("📁 Output directory: {}", output_dir.display());

    Ok(())
}

