//! stt-build - CLI tool for building spatiotemporal tile archives
//!
//! This tool converts GeoParquet files into optimized STT archives for web visualization.

mod clip;
mod columnar;
mod input;
mod simplify;
mod tiler;

use tiler::TileWriter;

use anyhow::{Context, Result};
use clap::{parser::ValueSource, ArgMatches, CommandFactory, FromArgMatches, Parser};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "stt-build")]
#[command(about = "Build spatiotemporal tile archives from GeoParquet data", long_about = None)]
#[command(version)]
struct Args {
    /// Input GeoParquet file path (.parquet or .geoparquet)
    #[arg(short, long)]
    input: PathBuf,

    /// Output archive path (.stt file)
    #[arg(short, long)]
    output: PathBuf,

    /// Field name containing timestamps (Unix ms or ISO 8601)
    #[arg(short, long, default_value = "timestamp")]
    time_field: String,

    /// Field name containing end timestamps for features with time ranges (optional)
    /// If provided, features will have valid_from (time_field) and valid_to (end_time_field)
    #[arg(long)]
    end_time_field: Option<String>,

    /// Time format: "unix-ms", "unix-sec", or "iso8601"
    #[arg(long, default_value = "iso8601")]
    time_format: String,

    /// Minimum zoom level
    #[arg(long, default_value = "0")]
    min_zoom: u8,

    /// Maximum zoom level
    #[arg(long, default_value = "14")]
    max_zoom: u8,

    /// Compression method: none, gzip
    #[arg(long, default_value = "gzip")]
    compression: String,

    /// Temporal bucket size for chunking tiles (e.g., "1h", "6h", "1d", "30m")
    /// Features are grouped into fixed temporal intervals, creating predictable tile boundaries
    /// that align with natural time units for efficient animation and prefetching.
    #[arg(long, default_value = "1h")]
    temporal_bucket: String,

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

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Output metadata JSON file (for generating datasets config)
    #[arg(long)]
    metadata_output: Option<PathBuf>,

    // --- Trajectory Clipping Options ---
    /// Disable trajectory clipping (store entire trajectories in centroid tile)
    /// By default, LineStrings with duration are clipped at tile boundaries
    #[arg(long)]
    no_clip: bool,

    /// Minimum vertices required to clip a trajectory (skip short paths)
    #[arg(long, default_value = "2")]
    clip_min_vertices: usize,

    // --- Memory/Performance Options ---
    /// Enable streaming mode (write tiles as each zoom level completes)
    /// Reduces peak memory usage for large datasets at cost of some parallelism
    #[arg(long)]
    streaming: bool,

    /// Enable line simplification for lower zoom levels (reduces memory and improves performance)
    #[arg(long)]
    simplify: bool,

    /// Maximum zoom level to apply simplification (higher zooms keep full detail)
    #[arg(long, default_value = "14")]
    simplify_max_zoom: u8,

    /// Fail the build on any row with a null or unparseable timestamp,
    /// rather than coercing it to Unix epoch 0 with a warning.
    #[arg(long)]
    strict_times: bool,

    /// Auto-tune zoom range, temporal bucket size, and compression by
    /// running stt-optimize's analyzer over the input before building.
    /// Any flag the user passes explicitly still wins; only unset/default
    /// values are filled in from the recommendation.
    #[arg(long)]
    auto: bool,
}

fn main() -> Result<()> {
    let matches = Args::command().get_matches();
    let mut args = Args::from_arg_matches(&matches)
        .context("failed to parse stt-build arguments")?;

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

    // Validate input file is GeoParquet
    let extension = args.input.extension().and_then(|s| s.to_str()).unwrap_or("");
    if !matches!(extension.to_lowercase().as_str(), "parquet" | "geoparquet") {
        anyhow::bail!(
            "Input must be a GeoParquet file (.parquet or .geoparquet), got: .{}",
            extension
        );
    }

    if args.auto {
        apply_auto_recommendations(&matches, &mut args)?;
    }

    // Parse compression
    let compression = parse_compression(&args.compression)?;

    // Step 1: Load all features into memory
    info!("Loading input data...");
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message("Reading input file...");

    let strictness = if args.strict_times {
        input::InputStrictness::Strict
    } else {
        input::InputStrictness::Warn
    };

    let features = input::load_features(
        &args.input,
        &args.time_field,
        args.end_time_field.as_deref(),
        &args.time_format,
        strictness,
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

    // Parse temporal bucket size
    let temporal_bucket_ms = parse_duration(&args.temporal_bucket)?;
    info!("Temporal bucket size: {} ms ({}))", temporal_bucket_ms, args.temporal_bucket);

    if !args.no_clip {
        info!("Trajectory clipping enabled (min {} vertices)", args.clip_min_vertices);
    } else {
        info!("Trajectory clipping disabled (--no-clip)");
    }

    if args.simplify {
        info!(
            "Line simplification enabled (max zoom {})",
            args.simplify_max_zoom
        );
    }

    let tile_config = tiler::TileConfig {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        layer_name: args.layer.clone(),
        temporal_bucket_ms,
        clip_trajectories: !args.no_clip,
        clip_min_vertices: args.clip_min_vertices,
        simplify: args.simplify,
        simplify_max_zoom: args.simplify_max_zoom,
    };

    let mut writer = stt_core::Archive::create(&args.output, compression)?;

    let tile_count = if args.streaming {
        // Streaming mode: write tiles as each zoom level completes
        info!("Using streaming mode (lower memory usage)...");
        let stats = tiler::generate_tiles_streaming(
            &features,
            &tile_config,
            &mut writer,
            args.workers,
        )?;
        info!(
            "Generated {} tiles ({} clipped segments, {} original features)",
            stats.total_tiles, stats.clipped_segments, stats.original_features
        );
        stats.total_tiles
    } else {
        // Standard mode: generate all tiles then write
        let tiles = tiler::generate_tiles(&features, &tile_config, args.workers)?;
        info!("Generated {} tiles", tiles.len());

        // Write archive
        info!("Writing archive...");
        let pb = ProgressBar::new(tiles.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );

        for tile in &tiles {
            writer.write_tile(tile)?;
            pb.inc(1);
        }

        pb.finish_with_message("Tiles written");
        tiles.len()
    };

    // Step 5: Build metadata
    let metadata = stt_core::metadata::Metadata::new(args.name.unwrap_or_else(|| {
        args.input
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string()
    }))
    .with_description(args.description.unwrap_or_default())
    .with_attribution(args.attribution.unwrap_or_default())
    .with_bounds(bounds)
    .with_time_range(time_range)
    .with_zoom_levels(args.min_zoom, args.max_zoom)
    .with_temporal_bucket_ms(temporal_bucket_ms);

    writer.finalize(&metadata)?;

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
            "tileCount": tile_count,
            "compression": args.compression,
            "temporalBucketMs": temporal_bucket_ms,
        });
        std::fs::write(metadata_path, serde_json::to_string_pretty(&metadata_json)?)?;
    }

    info!("Archive written successfully to {}", args.output.display());
    info!("Total tiles: {}", tile_count);
    info!("Total features: {}", features.len());

    Ok(())
}

/// Run stt-optimize over the input and fold its recommendations into `args`
/// for any flag the user did NOT pass explicitly.
fn apply_auto_recommendations(matches: &ArgMatches, args: &mut Args) -> Result<()> {
    info!("--auto: analyzing input for build recommendations...");
    let source = stt_optimize::DataSource::GeoParquet {
        path: args.input.clone(),
        time_field: args.time_field.clone(),
        time_format: args.time_format.clone(),
    };
    let rec = stt_optimize::recommend_for(&source)
        .context("stt-optimize analyzer failed")?;

    let user_set = |name: &str| {
        matches!(matches.value_source(name), Some(ValueSource::CommandLine))
    };

    if !user_set("min_zoom") {
        info!("  min-zoom: {} (auto)", rec.min_zoom);
        args.min_zoom = rec.min_zoom;
    }
    if !user_set("max_zoom") {
        info!("  max-zoom: {} (auto)", rec.max_zoom);
        args.max_zoom = rec.max_zoom;
    }
    if !user_set("compression") {
        info!("  compression: {} (auto)", rec.compression);
        args.compression = rec.compression.clone();
    }
    if !user_set("temporal_bucket") && rec.temporal_bucket_ms > 0 {
        let human = rec.temporal_bucket_human.clone();
        info!("  temporal-bucket: {} (auto)", human);
        args.temporal_bucket = human;
    }

    info!(
        "  confidence: {}% — {} reasons",
        rec.confidence,
        rec.explanations.len()
    );
    for line in &rec.explanations {
        info!("    - {}", line);
    }
    Ok(())
}

fn parse_compression(s: &str) -> Result<stt_core::types::Compression> {
    match s.to_lowercase().as_str() {
        "none" => Ok(stt_core::types::Compression::None),
        "gzip" => Ok(stt_core::types::Compression::Gzip),
        "zstd" | "zstandard" => Ok(stt_core::types::Compression::Zstd),
        _ => anyhow::bail!(
            "Invalid compression method: {}. Use 'none', 'gzip', or 'zstd'",
            s
        ),
    }
}

/// Parse a duration string like "1h", "6h", "1d", "30m" into milliseconds
fn parse_duration(s: &str) -> Result<u64> {
    let s = s.trim().to_lowercase();

    // Try to extract number and unit
    let mut num_str = String::new();
    let mut unit = String::new();

    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            num_str.push(c);
        } else {
            unit.push(c);
        }
    }

    let value: f64 = num_str
        .parse()
        .context(format!("Invalid duration value: {}", s))?;

    let multiplier: u64 = match unit.as_str() {
        "ms" | "" => 1,
        "s" | "sec" => 1000,
        "m" | "min" => 60 * 1000,
        "h" | "hr" | "hour" => 60 * 60 * 1000,
        "d" | "day" => 24 * 60 * 60 * 1000,
        _ => anyhow::bail!(
            "Invalid duration unit '{}'. Use ms, s, m, h, or d (e.g., '1h', '30m', '6h')",
            unit
        ),
    };

    Ok((value * multiplier as f64) as u64)
}
