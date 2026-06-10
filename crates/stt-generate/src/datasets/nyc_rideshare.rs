//! Generate NYC rideshare trajectory data using real TLC trip records + OSRM routing
//!
//! Uses historical NYC TLC data (pre-2017 with actual lat/long coordinates)
//! and routes trips through OSRM for realistic trajectories.

use crate::common::{self, LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter};
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::Deserialize;
use serde_json::{json, Map};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Generate NYC rideshare trajectory data from TLC records + OSRM routing")]
pub struct Args {
    /// Output file (.stt, .geojson, or .csv)
    #[arg(short, long, default_value = "nyc-rideshare.stt")]
    pub output: PathBuf,

    /// Input TLC trip data CSV file (must have lat/lon columns)
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Download TLC data for the specified month (format: YYYY-MM, e.g., 2016-01)
    /// Uses pre-2017 data which has actual lat/long coordinates
    #[arg(long)]
    pub download: Option<String>,

    /// Generate synthetic trips (random pickup/dropoff in Manhattan, routed through OSRM)
    #[arg(long)]
    pub synthetic: bool,

    /// Number of synthetic trips to generate
    #[arg(long, default_value = "1000")]
    pub num_trips: usize,

    /// Date for synthetic trips (format: YYYY-MM-DD)
    #[arg(long, default_value = "2024-01-15")]
    pub date: String,

    /// OSRM server URL
    #[arg(long, default_value = "http://localhost:5000")]
    pub osrm_url: String,

    /// Maximum number of trips to process
    #[arg(long)]
    pub max_trips: Option<usize>,

    /// Trajectory point interval in seconds
    #[arg(long, default_value = "30")]
    pub interval: u32,

    /// Skip trips longer than this duration in minutes
    #[arg(long, default_value = "60")]
    pub max_duration_minutes: u32,

    /// Skip trips shorter than this distance in meters
    #[arg(long, default_value = "500")]
    pub min_distance_meters: f64,

    /// Skip OSRM routing - just output pickup/dropoff points
    #[arg(long)]
    pub skip_routing: bool,

    /// Output LineString paths instead of individual points
    #[arg(long)]
    pub paths: bool,

    /// Aggregate routed trips into time-binned road-segment flow corridors —
    /// the pre-aggregated *overview* layer (one feature per corridor per
    /// `--flow-bin`, per-vertex `vertex_values` carry traversal counts).
    /// With `--from-intermediate <paths.parquet>` re-aggregates a kept
    /// `--paths` intermediate without re-routing through OSRM.
    #[arg(long)]
    pub flows: bool,

    /// Time bin duration for --flows aggregation (e.g. "15m", "1h")
    #[arg(long, default_value = "15m")]
    pub flow_bin: String,

    /// Skip stt-build step
    #[arg(long)]
    pub skip_build: bool,

    /// Build STT from existing intermediate file (skip download/generation)
    #[arg(long)]
    pub from_intermediate: Option<PathBuf>,

    /// Keep intermediate file after building STT
    #[arg(long)]
    pub keep_intermediate: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Take first N trips chronologically instead of random sampling (use with --max-trips)
    #[arg(long)]
    pub chronological: bool,

    /// Temporal bucket size for tile chunking (e.g., "1h", "6h", "1d", "30m")
    /// Uses fixed time intervals for predictable tile boundaries during animation.
    #[arg(long, default_value = "1h")]
    pub temporal_bucket: String,
}

#[derive(Debug, Clone)]
struct Trip {
    pickup_time: DateTime<Utc>,
    dropoff_time: DateTime<Utc>,
    pickup_lon: f64,
    pickup_lat: f64,
    dropoff_lon: f64,
    dropoff_lat: f64,
    passenger_count: u8,
    trip_distance: f64,
    fare_amount: f64,
}

#[derive(Debug, Deserialize)]
struct OsrmRouteResponse {
    code: String,
    routes: Option<Vec<OsrmRoute>>,
}

#[derive(Debug, Deserialize)]
struct OsrmRoute {
    geometry: OsrmGeometry,
    /// Present when the request includes `annotations=duration,distance`.
    /// Contains per-leg (segment) durations/distances along the route.
    /// `legs[i].annotation.duration[j]` is the OSRM-estimated seconds to
    /// traverse the j-th edge in leg i — reflects road class / speed limit,
    /// so highway runs are short and urban-grid blocks are long.
    #[serde(default)]
    legs: Vec<OsrmLeg>,
}

#[derive(Debug, Deserialize)]
struct OsrmGeometry {
    coordinates: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize, Default)]
struct OsrmLeg {
    #[serde(default)]
    annotation: Option<OsrmAnnotation>,
}

#[derive(Debug, Deserialize, Default)]
struct OsrmAnnotation {
    #[serde(default)]
    duration: Vec<f64>,
    #[allow(dead_code)]
    #[serde(default)]
    distance: Vec<f64>,
}

const NYC_MIN_LON: f64 = -74.05;
const NYC_MAX_LON: f64 = -73.70;
const NYC_MIN_LAT: f64 = 40.60;
const NYC_MAX_LAT: f64 = 40.90;

const MANHATTAN_HOTSPOTS: &[(f64, f64, &str)] = &[
    (-73.9855, 40.7580, "Times Square"),
    (-73.9712, 40.7527, "Grand Central"),
    (-73.9857, 40.7484, "Penn Station"),
    (-73.9776, 40.7614, "Rockefeller Center"),
    (-74.0060, 40.7128, "Wall Street"),
    (-74.0134, 40.7046, "Battery Park"),
    (-74.0000, 40.7200, "City Hall"),
    (-73.9654, 40.7829, "Upper West Side"),
    (-73.9589, 40.7736, "Central Park South"),
    (-73.9496, 40.7831, "Upper East Side"),
    (-73.9700, 40.7614, "5th Avenue"),
    (-73.9590, 40.7589, "Lexington Ave"),
    (-74.0020, 40.7410, "Chelsea"),
    (-74.0044, 40.7283, "West Village"),
    (-73.8740, 40.6413, "JFK Airport"),
    (-73.8726, 40.7769, "LaGuardia Airport"),
];

/// Helper macro for verbose logging
macro_rules! log_verbose {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!("   [verbose] {}", format!($($arg)*));
        }
    };
}

pub fn run(args: Args) -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           🚕 NYC Rideshare Trajectory Generator              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Handle --from-intermediate option (skip to STT build)
    if let Some(ref intermediate_file) = args.from_intermediate {
        println!("📂 Using existing intermediate file: {}", intermediate_file.display());
        if args.flows {
            // The intermediate is a --paths parquet: re-aggregate it into
            // flow corridors (no OSRM round-trip), then build.
            return flows_from_paths_intermediate(&args, intermediate_file);
        }
        return build_stt_from_intermediate(&args, intermediate_file);
    }

    // Determine intermediate output format. stt-build only accepts
    // GeoParquet (.parquet/.geoparquet), so both paths-mode and points-mode
    // write Parquet. The GeoJSON intermediate is kept around only for
    // `--output foo.geojson` invocations where the user explicitly asks for
    // GeoJSON (no stt-build step).
    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    let use_parquet = common::is_parquet_output(&intermediate_path);
    
    println!("📋 Configuration:");
    println!("   Output:        {}", args.output.display());
    println!("   Intermediate:  {}", intermediate_path.display());
    println!("   Format:        {}", if use_parquet { "GeoParquet (streaming)" } else { "GeoJSON" });
    let mode = if args.flows {
        "Flow corridors (time-binned segment aggregation)"
    } else if args.paths {
        "LineString paths"
    } else {
        "Point trajectories"
    };
    println!("   Mode:          {}", mode);
    println!("   Routing:       {}", if args.skip_routing { "Disabled (straight lines)" } else { &args.osrm_url });
    if let Some(max) = args.max_trips {
        println!("   Max trips:     {}", max);
    }
    println!();

    // Get trips
    let trips = if args.synthetic {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📝 STEP 1: Generate synthetic trips");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("🎲 Generating {} synthetic trips...", args.num_trips);
        generate_synthetic_trips(&args)?
    } else if let Some(ref download_month) = args.download {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📝 STEP 1: Download TLC data");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        let input_path = download_tlc_data(download_month, args.verbose)?;
        
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📝 STEP 2: Parse trip records");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📂 Reading trips from: {}", input_path.display());
        let trips = parse_tlc_csv(&input_path, &args)?;
        
        // Clean up downloaded file after parsing (unless keeping)
        if !args.keep_intermediate {
            log_verbose!(args.verbose, "Cleaning up downloaded CSV: {}", input_path.display());
            let _ = std::fs::remove_file(&input_path);
        }
        trips
    } else if let Some(ref input_path) = args.input {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📝 STEP 1: Parse trip records");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📂 Reading trips from: {}", input_path.display());
        parse_tlc_csv(input_path, &args)?
    } else {
        return Err(anyhow!(
            "Either --input, --download, --synthetic, or --from-intermediate must be specified.\n\n\
             Examples:\n\
             --download 2015-01            Download TLC data for January 2015 (with coordinates)\n\
             --synthetic                   Generate synthetic trips\n\
             --input data.csv              Use a CSV file with lat/lon coordinates\n\
             --from-intermediate data.parquet  Build STT from existing intermediate file"
        ));
    };

    if trips.is_empty() {
        return Err(anyhow!("No valid trips found"));
    }

    println!("   ✓ Loaded {} valid trips", trips.len());

    // Sample trips if max_trips is set
    let trips = if let Some(max) = args.max_trips {
        let mut sampled: Vec<_> = trips.clone();
        if args.chronological {
            // Sort by pickup time and take first N
            sampled.sort_by_key(|t| t.pickup_time);
            sampled.truncate(max);
            if let (Some(first), Some(last)) = (sampled.first(), sampled.last()) {
                println!("   📊 Selected first {} trips chronologically (from {})", sampled.len(), trips.len());
                println!("   📅 Time range: {} to {}", 
                    first.pickup_time.format("%Y-%m-%d %H:%M"),
                    last.pickup_time.format("%Y-%m-%d %H:%M"));
            }
        } else {
            // Random sampling
            let mut rng = rand::thread_rng();
            sampled.shuffle(&mut rng);
            sampled.truncate(max);
            println!("   📊 Sampled {} trips randomly (from {})", sampled.len(), trips.len());
        }
        sampled
    } else {
        trips
    };

    // Check OSRM connectivity if routing is enabled
    if !args.skip_routing {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📝 STEP 3: Verify OSRM routing server");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        check_osrm_connectivity(&args.osrm_url)?;
    }

    // Generate trajectories
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 STEP 4: Generate {} with OSRM routing", if args.paths { "paths" } else { "trajectories" });
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Processing {} trips...", trips.len());
    println!("   Output: {}", intermediate_path.display());
    
    if args.flows {
        if !use_parquet {
            return Err(anyhow!(
                "--flows requires a .stt or .parquet output (GeoJSON not supported)"
            ));
        }
        generate_flows_parquet(&trips, &args, &intermediate_path)?;
    } else if args.paths {
        if use_parquet {
            generate_paths_parquet(&trips, &args, &intermediate_path)?;
        } else {
            generate_paths_geojson(&trips, &args, &intermediate_path)?;
        }
    } else if use_parquet {
        generate_trajectories_parquet(&trips, &args, &intermediate_path)?;
    } else {
        // Points-mode GeoJSON output was removed alongside the points→parquet
        // rewrite: stt-build only consumes GeoParquet now, and the legacy
        // GeoJSON path silently produced unbuilt intermediates. Surface that
        // clearly rather than writing a file no downstream tool reads.
        return Err(anyhow!(
            "Points-mode (--output {}) only supports .stt or .parquet outputs. \
             Use `--output …{}.parquet` then run stt-build, or `--output …{}.stt` \
             to get both in one step.",
            args.output.display(),
            args.output.file_stem().and_then(|s| s.to_str()).unwrap_or("nyc-rideshare"),
            args.output.file_stem().and_then(|s| s.to_str()).unwrap_or("nyc-rideshare"),
        ));
    }

    // Verify intermediate file was created
    if !intermediate_path.exists() {
        return Err(anyhow!("Intermediate file was not created: {}", intermediate_path.display()));
    }
    let file_size = std::fs::metadata(&intermediate_path)?.len();
    println!("   ✓ Intermediate file created: {:.1} MB", file_size as f64 / 1_000_000.0);

    // Build STT if output is .stt
    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        build_stt_from_intermediate(&args, &intermediate_path)?;
        
        // Clean up intermediate file
        if !args.keep_intermediate {
            log_verbose!(args.verbose, "Cleaning up intermediate file: {}", intermediate_path.display());
            let _ = std::fs::remove_file(&intermediate_path);
        } else {
            println!("   📁 Keeping intermediate file: {}", intermediate_path.display());
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    ✅ Generation Complete                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("   Output: {}", args.output.display());
    if let Ok(meta) = std::fs::metadata(&args.output) {
        println!("   Size:   {:.1} MB", meta.len() as f64 / 1_000_000.0);
    }
    println!();

    Ok(())
}

fn build_stt_from_intermediate(args: &Args, intermediate_path: &PathBuf) -> Result<()> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 STEP 5: Build STT archive");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let time_field = "timestamp";
    let end_time_field = if args.paths || args.flows {
        Some("end_timestamp")
    } else {
        None
    };

    // Flow corridors are an overview layer: shallower zooms, and the tile
    // temporal bucket matches the aggregation bin so one tile holds exactly
    // one animation frame's features.
    let (min_zoom, max_zoom) = if args.flows { (8, 14) } else { (10, 16) };
    let temporal_bucket = if args.flows {
        &args.flow_bin
    } else {
        &args.temporal_bucket
    };

    println!("   Input:      {}", intermediate_path.display());
    println!("   Output:     {}", args.output.display());
    println!("   Time field: {}", time_field);
    if let Some(end_field) = end_time_field {
        println!("   End time:   {}", end_field);
    }
    println!("   Zoom:       {}-{}", min_zoom, max_zoom);
    println!("   Temporal bucket: {}", temporal_bucket);
    println!();

    // Single shared entry point for both the points (no end-time) and paths
    // (end_timestamp) builds. Previously the paths branch hand-rolled its own
    // std::process::Command with a duplicated flag list — which could silently
    // drift from the helper (and from the rest of the generators) as flags
    // evolved. `end_time_field` is Some("end_timestamp") exactly when
    // `args.paths`, so this is behaviour-identical to the old fork. The v3
    // zstd default compresses ~5× faster than gzip-6 for an equivalent ratio
    // and the v3 reader decodes ~3.3× faster (Sprint 2026-05 phase-1).
    common::run_stt_build_with_options(
        intermediate_path,
        &args.output,
        time_field,
        end_time_field,
        min_zoom,
        max_zoom,
        "zstd",
        Some(temporal_bucket),
    )?;

    Ok(())
}

/// `--flows --from-intermediate <paths.parquet>`: re-aggregate a kept
/// `--paths` intermediate into flow corridors, write the flows parquet next
/// to the output, then build the archive. Routing is skipped entirely.
fn flows_from_paths_intermediate(args: &Args, paths_parquet: &PathBuf) -> Result<()> {
    use crate::datasets::nyc_rideshare_flows::{parse_bin_ms, FlowAggregator};

    let bin_ms = parse_bin_ms(&args.flow_bin)?;
    let flows_parquet = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };
    if flows_parquet == *paths_parquet {
        return Err(anyhow!(
            "flows output parquet would overwrite the paths intermediate ({}); \
             choose a different --output",
            flows_parquet.display()
        ));
    }

    println!("\n🔁 Aggregating paths intermediate into {} flow bins", args.flow_bin);
    let mut agg = FlowAggregator::new(bin_ms);
    let rows = crate::datasets::nyc_rideshare_flows::aggregate_paths_parquet(paths_parquet, &mut agg)?;
    let (added, skipped, entries) = agg.stats();
    println!("   ✓ {} rows read ({} aggregated, {} skipped)", rows, added, skipped);
    println!("   ✓ {} (bin, segment) entries", entries);

    let (features, bins) = agg.write_parquet(&flows_parquet)?;
    println!("   ✓ {} corridor features across {} bins → {}", features, bins, flows_parquet.display());

    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        build_stt_from_intermediate(args, &flows_parquet)?;
        if !args.keep_intermediate {
            let _ = std::fs::remove_file(&flows_parquet);
        }
    }
    Ok(())
}

/// `--flows` from freshly routed trips: route in parallel (same pipeline as
/// `--paths`), aggregate in memory, write the flows parquet.
fn generate_flows_parquet(trips: &[Trip], args: &Args, output: &PathBuf) -> Result<()> {
    use crate::datasets::nyc_rideshare_flows::{parse_bin_ms, FlowAggregator};

    let bin_ms = parse_bin_ms(&args.flow_bin)?;
    let (results, annotated, failed) = route_trips_parallel(trips, args);

    println!("✓ Routing complete, aggregating into {} flow bins...", args.flow_bin);
    println!(
        "📊 OSRM annotations captured for {}/{} trips ({:.1}%), {} fallback routes",
        annotated,
        trips.len(),
        100.0 * annotated as f64 / trips.len().max(1) as f64,
        failed
    );

    let mut agg = FlowAggregator::new(bin_ms);
    for (_, trip, coords, vertex_times) in &results {
        agg.add_trip(
            coords,
            vertex_times.as_deref(),
            trip.pickup_time.timestamp_millis(),
            trip.dropoff_time.timestamp_millis(),
        );
    }
    let (added, skipped, entries) = agg.stats();
    println!("   ✓ {} trips aggregated ({} skipped), {} (bin, segment) entries", added, skipped, entries);

    let (features, bins) = agg.write_parquet(output)?;
    println!("✓ GeoParquet (flow corridors) written: {} features across {} bins", features, bins);
    Ok(())
}

fn generate_synthetic_trips(args: &Args) -> Result<Vec<Trip>> {
    let mut rng = rand::thread_rng();
    let mut trips = Vec::with_capacity(args.num_trips);

    let base_date = chrono::NaiveDate::parse_from_str(&args.date, "%Y-%m-%d")
        .context("Invalid date format. Use YYYY-MM-DD")?;
    let base_datetime = base_date.and_hms_opt(0, 0, 0).unwrap();
    let base_time = DateTime::from_naive_utc_and_offset(base_datetime, Utc);

    let pb = ProgressBar::new(args.num_trips as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} trips")?
            .progress_chars("=>-"),
    );

    for _ in 0..args.num_trips {
        let (pickup_lon, pickup_lat) = if rng.gen_bool(0.7) {
            let hotspot = &MANHATTAN_HOTSPOTS[rng.gen_range(0..MANHATTAN_HOTSPOTS.len())];
            let jitter_lon = rng.gen_range(-0.005..0.005);
            let jitter_lat = rng.gen_range(-0.003..0.003);
            (hotspot.0 + jitter_lon, hotspot.1 + jitter_lat)
        } else {
            (rng.gen_range(-74.02..-73.93), rng.gen_range(40.70..40.82))
        };

        let (dropoff_lon, dropoff_lat) = if rng.gen_bool(0.7) {
            let hotspot = &MANHATTAN_HOTSPOTS[rng.gen_range(0..MANHATTAN_HOTSPOTS.len())];
            let jitter_lon = rng.gen_range(-0.005..0.005);
            let jitter_lat = rng.gen_range(-0.003..0.003);
            (hotspot.0 + jitter_lon, hotspot.1 + jitter_lat)
        } else {
            (rng.gen_range(-74.02..-73.93), rng.gen_range(40.70..40.82))
        };

        let distance = common::haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon);
        if distance < args.min_distance_meters {
            continue;
        }

        let hour: u32 = if rng.gen_bool(0.4) {
            if rng.gen_bool(0.5) {
                rng.gen_range(7..10)
            } else {
                rng.gen_range(17..20)
            }
        } else {
            rng.gen_range(0..24)
        };
        let minute: u32 = rng.gen_range(0..60);
        let second: u32 = rng.gen_range(0..60);

        let pickup_time = base_time + Duration::hours(hour as i64)
            + Duration::minutes(minute as i64)
            + Duration::seconds(second as i64);

        let avg_speed_mps = 6.7;
        let est_duration_secs = (distance / avg_speed_mps) as i64;
        let duration_jitter = rng.gen_range(-60..60);
        let trip_duration = (est_duration_secs + duration_jitter).max(120);

        let dropoff_time = pickup_time + Duration::seconds(trip_duration);

        let passenger_count = rng.gen_range(1..5) as u8;
        let fare_amount = (distance / 1000.0) * 2.5 + rng.gen_range(2.0..5.0);

        trips.push(Trip {
            pickup_time,
            dropoff_time,
            pickup_lon,
            pickup_lat,
            dropoff_lon,
            dropoff_lat,
            passenger_count,
            trip_distance: distance / 1000.0,
            fare_amount,
        });

        pb.inc(1);
    }

    pb.finish_and_clear();
    println!("✓ Generated {} synthetic trips", trips.len());

    Ok(trips)
}

/// Download TLC yellow taxi data for a specific month (pre-2017 data has lat/long coordinates)
/// Uses Internet Archive to get the original CSV files with actual coordinates
fn download_tlc_data(month: &str, verbose: bool) -> Result<PathBuf> {
    // Validate month format (YYYY-MM)
    let parts: Vec<&str> = month.split('-').collect();
    if parts.len() != 2 {
        return Err(anyhow!("Invalid month format. Use YYYY-MM (e.g., 2016-01)"));
    }
    let year: i32 = parts[0].parse().context("Invalid year")?;
    let month_num: u32 = parts[1].parse().context("Invalid month")?;
    
    if year > 2016 || (year == 2016 && month_num >= 7) {
        return Err(anyhow!(
            "Data after June 2016 uses location IDs instead of coordinates.\n\
             Use --download 2015-01 through 2016-06 for lat/long data."
        ));
    }
    
    // Use Internet Archive to get the original CSV with lat/long coordinates
    // The official TLC site has retroactively converted all data to use LocationIDs
    let url = format!(
        "https://web.archive.org/web/20200904232527/https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_{}.csv",
        month
    );
    
    println!("   📥 Downloading from Internet Archive");
    println!("   Source: NYC TLC Yellow Taxi Data {}", month);
    log_verbose!(verbose, "URL: {}", url);
    println!();
    
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(1800)) // 30 min timeout for large files
        .build()?;
    
    log_verbose!(verbose, "Sending HTTP request...");
    let response = client
        .get(&url)
        .send()
        .context("Failed to download TLC data from Internet Archive")?;
    
    if !response.status().is_success() {
        return Err(anyhow!("Failed to download TLC data: HTTP {}", response.status()));
    }
    
    let total_size = response.content_length().unwrap_or(0);
    if total_size > 0 {
        println!("   📦 File size: {:.1} MB ({} bytes)", total_size as f64 / 1_000_000.0, total_size);
    } else {
        println!("   📦 Downloading (streaming, size unknown)...");
    }
    
    // Save to temp file as CSV
    let output_path = PathBuf::from(format!("tlc_yellow_{}.csv", month));
    log_verbose!(verbose, "Saving to: {}", output_path.display());
    
    let mut file = File::create(&output_path)?;
    
    // Stream the download with progress bar
    use std::io::Read;
    let pb = ProgressBar::new(if total_size > 0 { total_size } else { 2_000_000_000 }); // 2GB estimate if unknown
    pb.set_style(
        ProgressStyle::default_bar()
            .template("   [{bar:50.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
            .progress_chars("█▓░"),
    );
    
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 8192];
    let mut reader = response;
    
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(n) => {
                file.write_all(&buffer[..n])?;
                downloaded += n as u64;
                pb.set_position(downloaded);
            }
            Err(e) => {
                pb.finish_and_clear();
                return Err(anyhow!("Download failed after {} MB: {}", downloaded / 1_000_000, e));
            }
        }
    }
    
    pb.finish_and_clear();
    println!("   ✓ Downloaded: {:.1} MB", downloaded as f64 / 1_000_000.0);
    println!("   📁 Saved to: {}", output_path.display());
    
    Ok(output_path)
}

fn parse_tlc_csv(path: &PathBuf, args: &Args) -> Result<Vec<Trip>> {
    #[derive(Debug, Deserialize)]
    struct TlcTripRecord {
        #[serde(alias = "tpep_pickup_datetime", alias = "pickup_datetime")]
        pickup_datetime: String,
        #[serde(alias = "tpep_dropoff_datetime", alias = "dropoff_datetime")]
        dropoff_datetime: String,
        #[serde(alias = "pickup_longitude")]
        pickup_longitude: Option<f64>,
        #[serde(alias = "pickup_latitude")]
        pickup_latitude: Option<f64>,
        #[serde(alias = "dropoff_longitude")]
        dropoff_longitude: Option<f64>,
        #[serde(alias = "dropoff_latitude")]
        dropoff_latitude: Option<f64>,
        #[serde(alias = "passenger_count")]
        passenger_count: Option<u8>,
        #[serde(alias = "trip_distance")]
        trip_distance: Option<f64>,
        #[serde(alias = "fare_amount")]
        fare_amount: Option<f64>,
    }

    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut csv_reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(reader);

    let mut trips = Vec::new();

    for result in csv_reader.deserialize() {
        let record: TlcTripRecord = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let (pickup_lon, pickup_lat, dropoff_lon, dropoff_lat) = match (
            record.pickup_longitude,
            record.pickup_latitude,
            record.dropoff_longitude,
            record.dropoff_latitude,
        ) {
            (Some(plon), Some(plat), Some(dlon), Some(dlat)) => {
                if plon == 0.0 || plat == 0.0 || dlon == 0.0 || dlat == 0.0 {
                    continue;
                }
                (plon, plat, dlon, dlat)
            }
            _ => continue,
        };

        if !is_in_nyc(pickup_lon, pickup_lat) || !is_in_nyc(dropoff_lon, dropoff_lat) {
            continue;
        }

        let pickup_time = match parse_tlc_datetime(&record.pickup_datetime) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let dropoff_time = match parse_tlc_datetime(&record.dropoff_datetime) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let duration = dropoff_time.signed_duration_since(pickup_time);
        if duration.num_minutes() > args.max_duration_minutes as i64 || duration.num_seconds() < 60 {
            continue;
        }

        let estimated_distance = common::haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon);
        if estimated_distance < args.min_distance_meters {
            continue;
        }

        trips.push(Trip {
            pickup_time,
            dropoff_time,
            pickup_lon,
            pickup_lat,
            dropoff_lon,
            dropoff_lat,
            passenger_count: record.passenger_count.unwrap_or(1),
            trip_distance: record.trip_distance.unwrap_or(0.0),
            fare_amount: record.fare_amount.unwrap_or(0.0),
        });
    }

    Ok(trips)
}

fn is_in_nyc(lon: f64, lat: f64) -> bool {
    lon >= NYC_MIN_LON && lon <= NYC_MAX_LON && lat >= NYC_MIN_LAT && lat <= NYC_MAX_LAT
}

fn parse_tlc_datetime(s: &str) -> Result<DateTime<Utc>> {
    let formats = [
        "%Y-%m-%dT%H:%M:%S%.3f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%m/%d/%Y %H:%M:%S",
    ];

    for fmt in &formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s.trim(), fmt) {
            return Ok(DateTime::from_naive_utc_and_offset(dt, Utc));
        }
    }

    Err(anyhow!("Could not parse datetime: {}", s))
}

fn check_osrm_connectivity(osrm_url: &str) -> Result<()> {
    println!("🔍 Checking OSRM connectivity...");

    let test_url = format!(
        "{}/route/v1/driving/-73.99,40.73;-73.98,40.74?overview=full&geometries=geojson",
        osrm_url
    );

    let response = reqwest::blocking::get(&test_url)
        .context("Failed to connect to OSRM server. Is it running?")?;

    if !response.status().is_success() {
        return Err(anyhow!("OSRM server returned error status: {}", response.status()));
    }

    let body: serde_json::Value = response.json()?;
    let code = body.get("code").and_then(|c| c.as_str()).unwrap_or("");
    if code != "Ok" {
        return Err(anyhow!("OSRM returned code: {}", code));
    }

    println!("✓ OSRM server is ready");
    Ok(())
}

fn get_osrm_route(osrm_url: &str, from_lon: f64, from_lat: f64, to_lon: f64, to_lat: f64) -> Result<Option<OsrmRoute>> {
    // `annotations=duration,distance` returns per-leg per-edge durations so
    // we can compute real per-vertex timestamps that reflect street class
    // (highway vs. urban grid). Without this OSRM still returns the full
    // polyline but with no per-edge timing, forcing us to distribute the
    // trip duration uniformly by distance → "flash" artifacts on long
    // segments in the animated-trips renderer.
    let url = format!(
        "{}/route/v1/driving/{:.6},{:.6};{:.6},{:.6}?overview=full&geometries=geojson&annotations=duration,distance",
        osrm_url, from_lon, from_lat, to_lon, to_lat
    );

    let response = reqwest::blocking::get(&url)?;
    let body: OsrmRouteResponse = response.json()?;

    if body.code != "Ok" {
        return Ok(None);
    }

    Ok(body.routes.and_then(|mut r| r.pop()))
}

/// Compute true per-vertex absolute timestamps from OSRM annotation data.
///
/// OSRM's `annotations=duration` returns one duration per edge (between
/// consecutive overview coordinates). We accumulate them to get a per-vertex
/// absolute timestamp, then rescale to land exactly on the trip's recorded
/// pickup→dropoff window — OSRM's predicted total duration usually differs
/// from the actual TLC trip duration by ±10-30%, but the *shape* of the
/// per-edge distribution (which edges are fast, which are slow) is what
/// drives the per-segment-speed effect we want.
///
/// Returns `None` if the annotation arrays are missing or have the wrong
/// length, in which case the caller falls back to even-distance
/// interpolation in stt-build.
fn vertex_timestamps_from_route(
    route: &OsrmRoute,
    start_ms: i64,
    end_ms: i64,
) -> Option<Vec<i64>> {
    let coords = &route.geometry.coordinates;
    if coords.len() < 2 || end_ms <= start_ms {
        return None;
    }

    // Collect per-edge durations from all legs. For a single-leg request
    // (one source → one destination) there is exactly one leg whose
    // annotation.duration has `coords.len() - 1` entries.
    let mut edge_durations: Vec<f64> = Vec::with_capacity(coords.len().saturating_sub(1));
    for leg in &route.legs {
        if let Some(ann) = &leg.annotation {
            edge_durations.extend(ann.duration.iter().copied());
        }
    }
    if edge_durations.len() != coords.len() - 1 {
        return None;
    }
    let total: f64 = edge_durations.iter().sum();
    if total <= 0.0 {
        return None;
    }

    let trip_duration_ms = (end_ms - start_ms) as f64;
    let mut acc = 0.0f64;
    let mut out: Vec<i64> = Vec::with_capacity(coords.len());
    out.push(start_ms);
    for &d in &edge_durations {
        acc += d;
        let frac = acc / total;
        // Rescale onto the trip's recorded time window so endpoints are exact.
        out.push(start_ms + (frac * trip_duration_ms) as i64);
    }
    // Replace last to guarantee exact end_ms (avoids cumulative rounding drift).
    if let Some(last) = out.last_mut() {
        *last = end_ms;
    }
    Some(out)
}

fn interpolate_route(
    route: &OsrmRoute,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    interval_secs: u32,
) -> Vec<(f64, f64, DateTime<Utc>)> {
    let coords = &route.geometry.coordinates;
    if coords.is_empty() {
        return vec![];
    }
    let total_duration = end_time.signed_duration_since(start_time).num_seconds() as f64;
    if total_duration <= 0.0 {
        return vec![];
    }

    let start_ms = start_time.timestamp_millis();
    let end_ms = end_time.timestamp_millis();

    // Prefer per-segment timing from OSRM `annotations=duration` when present
    // — places sample points using the real street-class speed distribution
    // (highway sweeps fast, urban grid creeps slow). Falls back to the legacy
    // uniform-by-distance behaviour when annotations aren't available.
    if let Some(vertex_times) = vertex_timestamps_from_route(route, start_ms, end_ms) {
        return interpolate_route_with_vertex_times(coords, &vertex_times, interval_secs);
    }

    // Fallback: uniform-by-distance interpolation (constant average speed
    // across the full polyline). Preserves the prior behaviour for any
    // future OSRM build that omits annotations.
    let mut cumulative_distances: Vec<f64> = vec![0.0];
    let mut total_distance = 0.0;
    for i in 1..coords.len() {
        let dist = common::haversine_distance(
            coords[i - 1][1], coords[i - 1][0],
            coords[i][1], coords[i][0],
        );
        total_distance += dist;
        cumulative_distances.push(total_distance);
    }
    if total_distance == 0.0 {
        return vec![];
    }

    let mut points = Vec::new();
    let num_intervals = (total_duration / interval_secs as f64).ceil() as i64;

    for i in 0..=num_intervals {
        let t = (i as f64 * interval_secs as f64).min(total_duration);
        let fraction = t / total_duration;
        let target_distance = fraction * total_distance;

        let segment_idx = cumulative_distances
            .iter()
            .position(|&d| d >= target_distance)
            .unwrap_or(coords.len() - 1)
            .saturating_sub(1);

        let segment_start_dist = cumulative_distances[segment_idx];
        let segment_end_dist = cumulative_distances.get(segment_idx + 1).copied().unwrap_or(total_distance);
        let segment_length = segment_end_dist - segment_start_dist;

        let segment_fraction = if segment_length > 0.0 {
            (target_distance - segment_start_dist) / segment_length
        } else {
            0.0
        };

        let start_coord = &coords[segment_idx];
        let end_coord = coords.get(segment_idx + 1).unwrap_or(start_coord);

        let lon = start_coord[0] + (end_coord[0] - start_coord[0]) * segment_fraction;
        let lat = start_coord[1] + (end_coord[1] - start_coord[1]) * segment_fraction;
        let timestamp = start_time + Duration::seconds(t as i64);

        points.push((lon, lat, timestamp));
    }

    points
}

/// Walk the polyline in TIME (one sample per `interval_secs`), using
/// `vertex_times` (absolute Unix-ms, one per coord) as the time→space
/// mapping. Per-segment speed implied by OSRM annotations determines how
/// far each tick moves.
fn interpolate_route_with_vertex_times(
    coords: &[Vec<f64>],
    vertex_times: &[i64],
    interval_secs: u32,
) -> Vec<(f64, f64, DateTime<Utc>)> {
    let n = coords.len();
    if n < 2 || vertex_times.len() != n || interval_secs == 0 {
        return vec![];
    }
    let start_ms = vertex_times[0];
    let end_ms = vertex_times[n - 1];
    if end_ms <= start_ms {
        return vec![];
    }
    let total_ms = (end_ms - start_ms) as f64;
    let interval_ms = (interval_secs as i64) * 1_000;
    let n_steps = ((total_ms / interval_ms as f64).ceil() as i64).max(1);

    let mut out = Vec::with_capacity(n_steps as usize + 1);
    // Two-pointer walk: segment cursor advances monotonically with ticks.
    let mut seg = 0usize;
    for k in 0..=n_steps {
        let dt = ((k * interval_ms) as f64).min(total_ms);
        let ts = start_ms + dt as i64;
        while seg + 1 < n && vertex_times[seg + 1] < ts {
            seg += 1;
        }
        if seg + 1 >= n {
            seg = n - 2;
        }
        let t0 = vertex_times[seg];
        let t1 = vertex_times[seg + 1];
        let span = (t1 - t0) as f64;
        let local = if span > 0.0 { (ts - t0) as f64 / span } else { 0.0 };
        let a = &coords[seg];
        let b = &coords[seg + 1];
        let lon = a[0] + (b[0] - a[0]) * local;
        let lat = a[1] + (b[1] - a[1]) * local;
        let ts_dt = chrono::DateTime::<Utc>::from_timestamp_millis(ts).unwrap_or(start_time_anchor(start_ms));
        out.push((lon, lat, ts_dt));
    }
    out
}

/// `from_timestamp_millis` returns `Option`; unwrap it via a tiny anchor
/// fallback so the interpolator stays `Vec<(...)>` rather than `Result`.
fn start_time_anchor(start_ms: i64) -> DateTime<Utc> {
    chrono::DateTime::<Utc>::from_timestamp_millis(start_ms).unwrap_or_else(Utc::now)
}

fn generate_trajectories_parquet(trips: &[Trip], args: &Args, output: &PathBuf) -> Result<()> {
    use common::{PointRecord, PropertyColumn, PropertyKind, StreamingParquetWriter};

    println!("\n🚀 Generating trajectories (Parquet)...");

    let pb = ProgressBar::new(trips.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} trips ({eta})")?
            .progress_chars("=>-"),
    );

    // Typed property columns so stt-build's downstream property inference
    // sees numerics as numerics (passenger_count/trip_distance/fare_amount)
    // and status as a string category. Mirrors the schema the old GeoJSON
    // path produced after going through stt-build's type inference.
    let property_columns = vec![
        PropertyColumn { name: "trip_id".into(), kind: PropertyKind::Numeric },
        PropertyColumn { name: "passenger_count".into(), kind: PropertyKind::Numeric },
        PropertyColumn { name: "trip_distance".into(), kind: PropertyKind::Numeric },
        PropertyColumn { name: "fare_amount".into(), kind: PropertyKind::Numeric },
        PropertyColumn { name: "status".into(), kind: PropertyKind::String },
    ];
    let mut writer = StreamingParquetWriter::with_columns(output, property_columns)?;

    let mut failed_routes = 0usize;
    let mut total_points = 0usize;

    for (trip_id, trip) in trips.iter().enumerate() {
        let points = if args.skip_routing {
            vec![
                (trip.pickup_lon, trip.pickup_lat, trip.pickup_time),
                (trip.dropoff_lon, trip.dropoff_lat, trip.dropoff_time),
            ]
        } else {
            match get_osrm_route(
                &args.osrm_url,
                trip.pickup_lon, trip.pickup_lat,
                trip.dropoff_lon, trip.dropoff_lat,
            ) {
                Ok(Some(route)) => {
                    interpolate_route(&route, trip.pickup_time, trip.dropoff_time, args.interval)
                }
                Ok(None) | Err(_) => {
                    failed_routes += 1;
                    vec![
                        (trip.pickup_lon, trip.pickup_lat, trip.pickup_time),
                        (trip.dropoff_lon, trip.dropoff_lat, trip.dropoff_time),
                    ]
                }
            }
        };

        let last_idx = points.len().saturating_sub(1);
        for (i, (lon, lat, timestamp)) in points.iter().enumerate() {
            let status = if i == 0 {
                "pickup"
            } else if i == last_idx {
                "dropoff"
            } else {
                "enroute"
            };
            let mut properties = Map::new();
            properties.insert("trip_id".into(), json!(trip_id));
            properties.insert("passenger_count".into(), json!(trip.passenger_count));
            properties.insert("trip_distance".into(), json!(trip.trip_distance));
            properties.insert("fare_amount".into(), json!(trip.fare_amount));
            properties.insert("status".into(), json!(status));

            writer.write_point(&PointRecord {
                lon: *lon,
                lat: *lat,
                timestamp_ms: timestamp.timestamp_millis(),
                properties,
            })?;
            total_points += 1;
        }
        pb.inc(1);
    }

    let rows = writer.finish()?;
    pb.finish_and_clear();
    println!("📊 Generated {} trajectory points (wrote {} rows)", total_points, rows);
    if failed_routes > 0 {
        println!("⚠️  {} trips used fallback routing", failed_routes);
    }

    Ok(())
}

fn generate_paths_geojson(trips: &[Trip], args: &Args, output: &PathBuf) -> Result<()> {
    println!("\n🚀 Generating paths...");

    let pb = ProgressBar::new(trips.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} trips ({eta})")?
            .progress_chars("=>-"),
    );

    let mut features = Vec::new();
    let mut failed_routes = 0;
    let mut total_coords = 0;

    for (trip_id, trip) in trips.iter().enumerate() {
        let coords: Vec<[f64; 2]> = if args.skip_routing {
            vec![
                [trip.pickup_lon, trip.pickup_lat],
                [trip.dropoff_lon, trip.dropoff_lat],
            ]
        } else {
            match get_osrm_route(
                &args.osrm_url,
                trip.pickup_lon, trip.pickup_lat,
                trip.dropoff_lon, trip.dropoff_lat,
            ) {
                Ok(Some(route)) => {
                    route.geometry.coordinates.iter().map(|c| [c[0], c[1]]).collect()
                }
                Ok(None) | Err(_) => {
                    failed_routes += 1;
                    vec![
                        [trip.pickup_lon, trip.pickup_lat],
                        [trip.dropoff_lon, trip.dropoff_lat],
                    ]
                }
            }
        };

        if coords.len() < 2 {
            pb.inc(1);
            continue;
        }

        total_coords += coords.len();

        let mut properties = Map::new();
        properties.insert("trip_id".to_string(), json!(trip_id));
        properties.insert("passenger_count".to_string(), json!(trip.passenger_count));
        properties.insert("trip_distance".to_string(), json!(trip.trip_distance));
        properties.insert("fare_amount".to_string(), json!(trip.fare_amount));

        let feature = common::create_linestring_feature_with_time_range(
            coords,
            trip.pickup_time,
            trip.dropoff_time,
            properties,
        );
        features.push(feature);

        pb.inc(1);
    }

    pb.finish_and_clear();
    println!("📊 Generated {} path features with {} total coordinates", features.len(), total_coords);
    if failed_routes > 0 {
        println!("⚠️  {} trips used fallback routing", failed_routes);
    }

    common::write_geojson(features, output)?;
    Ok(())
}

/// Per-trip routing result: (trip index, trip, polyline, optional per-vertex
/// absolute timestamps from OSRM's `annotations=duration`). `None` timestamps
/// mean stt-build falls back to uniform-by-distance interpolation.
type TripResult = (usize, Trip, Vec<[f64; 2]>, Option<Vec<i64>>);

/// Route trips through OSRM in parallel batches. Returns (results,
/// annotated-route count, failed-route count). Shared by `--paths` (which
/// writes one feature per trip) and `--flows` (which aggregates).
fn route_trips_parallel(trips: &[Trip], args: &Args) -> (Vec<TripResult>, usize, usize) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    // Determine parallelism - use fewer threads to avoid overwhelming OSRM
    let num_threads = std::cmp::min(rayon::current_num_threads(), 16);
    println!("   Using {} parallel workers", num_threads);

    let failed_routes = AtomicUsize::new(0);
    let processed = AtomicUsize::new(0);
    let osrm_url = args.osrm_url.clone();
    let skip_routing = args.skip_routing;
    let total_trips = trips.len();

    let batch_size = 5000;
    let mut all_results: Vec<TripResult> = Vec::with_capacity(trips.len());
    let start_time = Instant::now();
    let mut annotated_routes: usize = 0;

    let num_batches = (trips.len() + batch_size - 1) / batch_size;
    println!("   Processing {} batches of {} trips each\n", num_batches, batch_size);

    for (batch_idx, batch_start) in (0..trips.len()).step_by(batch_size).enumerate() {
        let batch_end = std::cmp::min(batch_start + batch_size, trips.len());
        let batch = &trips[batch_start..batch_end];

        // Process batch in parallel
        let batch_results: Vec<TripResult> = batch
            .par_iter()
            .enumerate()
            .map(|(i, trip)| {
                let trip_id = batch_start + i;
                let (coords, vertex_times): (Vec<[f64; 2]>, Option<Vec<i64>>) = if skip_routing {
                    (
                        vec![
                            [trip.pickup_lon, trip.pickup_lat],
                            [trip.dropoff_lon, trip.dropoff_lat],
                        ],
                        None,
                    )
                } else {
                    match get_osrm_route_pooled(
                        &osrm_url,
                        trip.pickup_lon, trip.pickup_lat,
                        trip.dropoff_lon, trip.dropoff_lat,
                    ) {
                        Ok(Some(route)) => {
                            let coords: Vec<[f64; 2]> = route
                                .geometry
                                .coordinates
                                .iter()
                                .map(|c| [c[0], c[1]])
                                .collect();
                            let vt = vertex_timestamps_from_route(
                                &route,
                                trip.pickup_time.timestamp_millis(),
                                trip.dropoff_time.timestamp_millis(),
                            );
                            (coords, vt)
                        }
                        Ok(None) | Err(_) => {
                            failed_routes.fetch_add(1, Ordering::Relaxed);
                            (
                                vec![
                                    [trip.pickup_lon, trip.pickup_lat],
                                    [trip.dropoff_lon, trip.dropoff_lat],
                                ],
                                None,
                            )
                        }
                    }
                };
                processed.fetch_add(1, Ordering::Relaxed);
                (trip_id, trip.clone(), coords, vertex_times)
            })
            .collect();

        annotated_routes += batch_results.iter().filter(|r| r.3.is_some()).count();
        
        all_results.extend(batch_results);
        
        // Print progress after each batch
        let done = processed.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed().as_secs_f64();
        let rate = done as f64 / elapsed;
        let remaining = (total_trips - done) as f64 / rate;

        eprintln!(
            "   [{:>3}/{}] Batch {} complete: {}/{} trips ({:.0}/s) - ETA: {:.0}m {:.0}s",
            batch_idx + 1,
            num_batches,
            batch_idx + 1,
            done,
            total_trips,
            rate,
            remaining / 60.0,
            remaining % 60.0
        );
    }

    let failed = failed_routes.load(Ordering::Relaxed);
    (all_results, annotated_routes, failed)
}

fn generate_paths_parquet(trips: &[Trip], args: &Args, output: &PathBuf) -> Result<()> {
    println!("\n🚀 Generating paths (Parquet) - PARALLEL MODE");

    let (all_results, annotated_routes, failed) = route_trips_parallel(trips, args);
    println!("✓ Routing complete, writing to Parquet...");

    // Create streaming Parquet writer and write results sequentially
    let property_columns = vec![
        PropertyColumn::numeric("trip_id"),
        PropertyColumn::numeric("passenger_count"),
        PropertyColumn::numeric("trip_distance"),
        PropertyColumn::numeric("fare_amount"),
    ];
    let mut writer = StreamingLineStringParquetWriter::with_columns(output, property_columns)?;
    
    let mut total_coords = 0;
    for (trip_id, trip, coords, vertex_times) in all_results {
        if coords.len() < 2 {
            continue;
        }

        total_coords += coords.len();

        let mut properties = Map::new();
        properties.insert("trip_id".to_string(), json!(trip_id));
        properties.insert("passenger_count".to_string(), json!(trip.passenger_count));
        properties.insert("trip_distance".to_string(), json!(trip.trip_distance));
        properties.insert("fare_amount".to_string(), json!(trip.fare_amount));

        let mut record = LineStringRecord::new(
            coords,
            trip.pickup_time,
            Some(trip.dropoff_time),
            properties,
        );
        // Only attach when length matches; defensive — vertex_timestamps_from_route
        // already enforces this, but a mismatch would silently corrupt timing.
        if let Some(vt) = vertex_times {
            if vt.len() == record.coordinates.len() {
                record.vertex_timestamps_ms = Some(vt);
            }
        }

        writer.write_linestring(&record)?;
    }

    let count = writer.finish()?;
    println!("✓ GeoParquet (LineString) written successfully ({} rows)", count);
    println!("📊 Generated {} path features with {} total coordinates", count, total_coords);
    println!(
        "📊 OSRM annotations captured for {}/{} trips ({:.1}%)",
        annotated_routes,
        trips.len(),
        100.0 * annotated_routes as f64 / trips.len().max(1) as f64
    );
    if failed > 0 {
        println!("⚠️  {} trips used fallback routing", failed);
    }

    Ok(())
}

/// Thread-local HTTP client for connection pooling
fn get_osrm_route_pooled(osrm_url: &str, from_lon: f64, from_lat: f64, to_lon: f64, to_lat: f64) -> Result<Option<OsrmRoute>> {
    use std::cell::RefCell;
    
    thread_local! {
        static CLIENT: RefCell<reqwest::blocking::Client> = RefCell::new(
            reqwest::blocking::Client::builder()
                .pool_max_idle_per_host(10)
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap()
        );
    }
    
    let url = format!(
        "{}/route/v1/driving/{:.6},{:.6};{:.6},{:.6}?overview=full&geometries=geojson&annotations=duration,distance",
        osrm_url, from_lon, from_lat, to_lon, to_lat
    );

    CLIENT.with(|client| {
        let response = client.borrow().get(&url).send()?;
        let body: OsrmRouteResponse = response.json()?;

        if body.code != "Ok" {
            return Ok(None);
        }

        Ok(body.routes.and_then(|mut r| r.pop()))
    })
}


