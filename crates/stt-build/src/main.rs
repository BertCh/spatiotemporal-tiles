//! stt-build - CLI tool for building spatiotemporal tile archives
//!
//! This tool converts GeoJSON, CSV, or other spatial data formats into
//! optimized STT archives for web visualization.

mod input;
mod tiler;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "stt-build")]
#[command(about = "Build spatiotemporal tile archives from geospatial data", long_about = None)]
#[command(version)]
struct Args {
    /// Input file path (GeoJSON, CSV, etc.)
    #[arg(short, long)]
    input: PathBuf,

    /// Output archive path (.stt file)
    #[arg(short, long)]
    output: PathBuf,

    /// Field name containing timestamps (Unix ms or ISO 8601)
    #[arg(short, long, default_value = "timestamp")]
    time_field: String,

    /// Time format: "unix-ms", "unix-sec", or "iso8601"
    #[arg(long, default_value = "iso8601")]
    time_format: String,

    /// Temporal resolution profile: 
    /// - Profiles: "high-frequency", "sparse-events", "daily-aggregates"
    /// - Or fixed bucket: "none", "second", "minute", "hour", "day", "week", "month", "year"
    #[arg(long, default_value = "sparse-events")]
    temporal_resolution: String,

    /// Minimum zoom level
    #[arg(long, default_value = "0")]
    min_zoom: u8,

    /// Maximum zoom level
    #[arg(long, default_value = "14")]
    max_zoom: u8,

    /// Tile extent (coordinate precision)
    #[arg(long, default_value = "4096")]
    extent: u32,

    /// Compression method: none, gzip
    #[arg(long, default_value = "gzip")]
    compression: String,

    /// Simplification tolerance (in degrees, 0 = no simplification)
    #[arg(long, default_value = "0.0001")]
    simplification: f64,

    /// Maximum tile size in bytes (features will be dropped if exceeded)
    #[arg(long, default_value = "500000")]
    max_tile_size: usize,

    /// Number of parallel workers
    #[arg(short, long, default_value = "4")]
    workers: usize,

    /// Archive name (metadata)
    #[arg(long)]
    name: Option<String>,

    /// Archive description (metadata)
    #[arg(long)]
    description: Option<String>,

    /// Attribution text (metadata)
    #[arg(long)]
    attribution: Option<String>,

    /// Layer name
    #[arg(long, default_value = "default")]
    layer: String,

    /// Enable delta encoding (reduces file size for repetitive features)
    #[arg(long)]
    delta_encoding: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Output metadata JSON file (for generating datasets config)
    #[arg(long)]
    metadata_output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(if args.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set tracing subscriber")?;

    info!("Starting stt-build");
    info!("Input: {}", args.input.display());
    info!("Output: {}", args.output.display());

    // Parse compression
    let compression = parse_compression(&args.compression)?;

    // Parse temporal resolution profile
    let temporal_resolution = tiler::TemporalResolutionProfile::from_str(&args.temporal_resolution)
        .ok_or_else(|| anyhow::anyhow!(
            "Invalid temporal resolution: {}.\nProfiles: high-frequency, sparse-events, daily-aggregates\nFixed buckets: none, second, minute, hour, day, week, month, year",
            args.temporal_resolution
        ))?;

    info!("Temporal resolution: {}", temporal_resolution.description());

    // Step 1: Load and parse input
    info!("Loading input data...");
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message("Reading input file...");

    let features = input::load_features(
        &args.input,
        &args.time_field,
        &args.time_format,
    )?;

    pb.finish_with_message(format!("Loaded {} features", features.len()));
    info!("Loaded {} features", features.len());

    if features.is_empty() {
        warn!("No features found in input file");
        return Ok(());
    }

    // Step 2: Analyze data bounds
    info!("Analyzing data bounds...");
    let (bounds, time_range) = input::calculate_bounds(&features)?;
    info!("Spatial bounds: {:?}", bounds);
    info!("Time range: {} to {}", time_range.start, time_range.end);

    // Step 3: Generate tiles
    info!("Generating tiles...");
    if args.delta_encoding {
        info!("Delta encoding: enabled");
    }
    let tile_config = tiler::TileConfig {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        extent: args.extent,
        simplification: args.simplification,
        max_tile_size: args.max_tile_size,
        layer_name: args.layer.clone(),
        temporal_resolution,
        use_delta_encoding: args.delta_encoding,
        use_quantization: false,   // TODO: Make configurable
        budget: stt_core::budget::TileBudget::new(args.max_tile_size, 10000, 1000),
        target_chunk_size: args.max_tile_size,
    };

    let tiles = tiler::generate_tiles(&features, &tile_config, args.workers)?;
    info!("Generated {} tiles", tiles.len());

    // Step 4: Write archive
    info!("Writing archive...");
    let pb = ProgressBar::new(tiles.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut writer = stt_core::Archive::create(&args.output)?;

    for tile in &tiles {
        writer.add_tile(&tile.id, &tile.proto, compression)?;
        pb.inc(1);
    }

    pb.finish_with_message("Tiles written");

    // Step 5: Build metadata
    let metadata = stt_core::metadata::Metadata::new(
        args.name.unwrap_or_else(|| args.input.file_stem().unwrap().to_string_lossy().to_string())
    )
    .with_description(args.description.unwrap_or_default())
    .with_attribution(args.attribution.unwrap_or_default())
    .with_bounds(bounds)
    .with_time_range(time_range)
    .with_zoom_levels(args.min_zoom, args.max_zoom);

    let proto_metadata = metadata.to_proto();
    writer.finalize(&proto_metadata)?;

    // Step 6: Write metadata JSON if requested
    if let Some(metadata_path) = args.metadata_output {
        info!("Writing metadata JSON to {}...", metadata_path.display());
        let metadata_json = serde_json::json!({
            "filename": args.output.file_name().unwrap().to_string_lossy(),
            "name": metadata.name,
            "description": metadata.description,
            "attribution": metadata.attribution,
            "timeRange": {
                "start": time_range.start,
                "end": time_range.end,
            },
            "zoomLevels": {
                "min": args.min_zoom,
                "max": args.max_zoom,
            },
            "bounds": {
                "minLon": bounds.min_lon,
                "minLat": bounds.min_lat,
                "maxLon": bounds.max_lon,
                "maxLat": bounds.max_lat,
            },
            "tileCount": tiles.len(),
            "compression": args.compression,
        });
        std::fs::write(metadata_path, serde_json::to_string_pretty(&metadata_json)?)?;
    }

    info!("Archive written successfully to {}", args.output.display());
    info!("Total tiles: {}", tiles.len());
    info!("Total features: {}", features.len());

    Ok(())
}

fn parse_compression(s: &str) -> Result<stt_core::types::Compression> {
    match s.to_lowercase().as_str() {
        "none" => Ok(stt_core::types::Compression::None),
        "gzip" => Ok(stt_core::types::Compression::Gzip),
        _ => anyhow::bail!("Invalid compression method: {}. Use 'none' or 'gzip'", s),
    }
}

