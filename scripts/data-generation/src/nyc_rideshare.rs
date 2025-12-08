//! Generate NYC rideshare trajectory data using real TLC trip records + OSRM routing
//!
//! Uses historical NYC TLC data (pre-2017 with actual lat/long coordinates)
//! and routes trips through OSRM for realistic trajectories.

mod common;

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray};
use chrono::{DateTime, Duration, NaiveDateTime, TimeZone, Utc};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::seq::SliceRandom;
use serde::Deserialize;
use serde_json::{json, Map};
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-nyc-rideshare")]
#[command(about = "Generate NYC rideshare trajectory data from TLC records + OSRM routing")]
struct Args {
    /// Output file path (.csv recommended for large datasets)
    #[arg(short, long, default_value = "nyc-rideshare.csv")]
    output: PathBuf,

    /// Input TLC trip data CSV file (must have lat/lon columns, not zone IDs)
    #[arg(long)]
    input: Option<PathBuf>,

    /// Download TLC data for this year-month (format: YYYY-MM, e.g., 2015-01)
    /// Note: Modern TLC parquet files use zone IDs, not coordinates.
    /// For coordinates, use --synthetic mode or provide a Kaggle CSV with --input
    #[arg(long)]
    download: Option<String>,

    /// Generate synthetic trips (random pickup/dropoff in Manhattan, routed through OSRM)
    #[arg(long)]
    synthetic: bool,

    /// Number of synthetic trips to generate (only with --synthetic)
    #[arg(long, default_value = "1000")]
    num_trips: usize,

    /// Date for synthetic trips (format: YYYY-MM-DD)
    #[arg(long, default_value = "2024-01-15")]
    date: String,

    /// OSRM server URL
    #[arg(long, default_value = "http://localhost:5000")]
    osrm_url: String,

    /// Maximum number of trips to process (for testing/sampling)
    #[arg(long)]
    max_trips: Option<usize>,

    /// Trajectory point interval in seconds (how often to emit a point along route)
    #[arg(long, default_value = "30")]
    interval: u32,

    /// Skip trips longer than this duration in minutes
    #[arg(long, default_value = "60")]
    max_duration_minutes: u32,

    /// Skip trips shorter than this distance in meters
    #[arg(long, default_value = "500")]
    min_distance_meters: f64,

    /// Number of parallel OSRM requests
    #[arg(long, default_value = "4")]
    parallelism: usize,

    /// Skip OSRM routing - just output pickup/dropoff points (for testing)
    #[arg(long)]
    skip_routing: bool,
}

/// NYC TLC Yellow Taxi Trip Record (for CSV parsing, pre-2017 format with coordinates)
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct TlcTripRecord {
    // Different CSV headers exist across years, handle common variations
    #[serde(alias = "tpep_pickup_datetime", alias = "Trip_Pickup_DateTime", alias = "pickup_datetime")]
    pickup_datetime: String,

    #[serde(alias = "tpep_dropoff_datetime", alias = "Trip_Dropoff_DateTime", alias = "dropoff_datetime")]
    dropoff_datetime: String,

    #[serde(alias = "pickup_longitude", alias = "Start_Lon")]
    pickup_longitude: Option<f64>,

    #[serde(alias = "pickup_latitude", alias = "Start_Lat")]
    pickup_latitude: Option<f64>,

    #[serde(alias = "dropoff_longitude", alias = "End_Lon")]
    dropoff_longitude: Option<f64>,

    #[serde(alias = "dropoff_latitude", alias = "End_Lat")]
    dropoff_latitude: Option<f64>,

    #[serde(alias = "passenger_count", alias = "Passenger_Count")]
    passenger_count: Option<u8>,

    #[serde(alias = "trip_distance", alias = "Trip_Distance")]
    trip_distance: Option<f64>,

    #[serde(alias = "fare_amount", alias = "Fare_Amt")]
    fare_amount: Option<f64>,
}

/// FiveThirtyEight Uber FOIL data format (pickup-only, 2014)
/// Format: "Date/Time","Lat","Lon","Base"
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct UberFoilRecord {
    #[serde(alias = "Date/Time")]
    datetime: String,
    
    #[serde(alias = "Lat")]
    lat: f64,
    
    #[serde(alias = "Lon")]
    lon: f64,
    
    #[serde(alias = "Base")]
    base: String,
}

/// Parsed and validated trip
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

/// OSRM Route Response
#[derive(Debug, Deserialize)]
struct OsrmRouteResponse {
    code: String,
    routes: Option<Vec<OsrmRoute>>,
}

#[derive(Debug, Deserialize)]
struct OsrmRoute {
    geometry: OsrmGeometry,
    duration: f64,  // seconds
    distance: f64,  // meters
    legs: Vec<OsrmLeg>,
}

#[derive(Debug, Deserialize)]
struct OsrmGeometry {
    coordinates: Vec<Vec<f64>>,  // [[lon, lat], ...]
}

#[derive(Debug, Deserialize)]
struct OsrmLeg {
    steps: Option<Vec<OsrmStep>>,
    duration: f64,
    distance: f64,
}

#[derive(Debug, Deserialize)]
struct OsrmStep {
    duration: f64,
    distance: f64,
    geometry: OsrmGeometry,
}

/// NYC bounding box (Manhattan and surrounding areas)
const NYC_MIN_LON: f64 = -74.05;
const NYC_MAX_LON: f64 = -73.70;
const NYC_MIN_LAT: f64 = 40.60;
const NYC_MAX_LAT: f64 = 40.90;

/// Manhattan hotspots for realistic pickup/dropoff distribution
const MANHATTAN_HOTSPOTS: &[(f64, f64, &str)] = &[
    // Midtown
    (-73.9855, 40.7580, "Times Square"),
    (-73.9712, 40.7527, "Grand Central"),
    (-73.9857, 40.7484, "Penn Station"),
    (-73.9776, 40.7614, "Rockefeller Center"),
    // Downtown
    (-74.0060, 40.7128, "Wall Street"),
    (-74.0134, 40.7046, "Battery Park"),
    (-74.0000, 40.7200, "City Hall"),
    // Uptown
    (-73.9654, 40.7829, "Upper West Side"),
    (-73.9589, 40.7736, "Central Park South"),
    (-73.9496, 40.7831, "Upper East Side"),
    // East Side
    (-73.9700, 40.7614, "5th Avenue"),
    (-73.9590, 40.7589, "Lexington Ave"),
    // West Side
    (-74.0020, 40.7410, "Chelsea"),
    (-74.0044, 40.7283, "West Village"),
    // Airports (some trips go there)
    (-73.8740, 40.6413, "JFK Airport"),
    (-73.8726, 40.7769, "LaGuardia Airport"),
];

use rand::Rng;

/// Generate synthetic trips with random pickup/dropoff in Manhattan
fn generate_synthetic_trips(args: &Args) -> Result<Vec<Trip>> {
    let mut rng = rand::thread_rng();
    let mut trips = Vec::with_capacity(args.num_trips);
    
    // Parse base date
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
        // Pick random pickup location (weighted towards hotspots)
        let (pickup_lon, pickup_lat) = if rng.gen_bool(0.7) {
            // 70% chance: near a hotspot
            let hotspot = &MANHATTAN_HOTSPOTS[rng.gen_range(0..MANHATTAN_HOTSPOTS.len())];
            let jitter_lon = rng.gen_range(-0.005..0.005);
            let jitter_lat = rng.gen_range(-0.003..0.003);
            (hotspot.0 + jitter_lon, hotspot.1 + jitter_lat)
        } else {
            // 30% chance: random location in Manhattan
            (
                rng.gen_range(-74.02..-73.93),
                rng.gen_range(40.70..40.82),
            )
        };
        
        // Pick random dropoff location (different from pickup)
        let (dropoff_lon, dropoff_lat) = if rng.gen_bool(0.7) {
            let hotspot = &MANHATTAN_HOTSPOTS[rng.gen_range(0..MANHATTAN_HOTSPOTS.len())];
            let jitter_lon = rng.gen_range(-0.005..0.005);
            let jitter_lat = rng.gen_range(-0.003..0.003);
            (hotspot.0 + jitter_lon, hotspot.1 + jitter_lat)
        } else {
            (
                rng.gen_range(-74.02..-73.93),
                rng.gen_range(40.70..40.82),
            )
        };
        
        // Skip if too close
        let distance = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon);
        if distance < args.min_distance_meters {
            continue;
        }
        
        // Random time during the day (weighted towards rush hours)
        let hour: u32 = if rng.gen_bool(0.4) {
            // 40% chance: rush hour (7-9 AM or 5-7 PM)
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
        
        // Estimate trip duration based on distance (avg 15 mph in NYC traffic)
        let avg_speed_mps = 6.7; // ~15 mph
        let est_duration_secs = (distance / avg_speed_mps) as i64;
        let duration_jitter = rng.gen_range(-60..60);
        let trip_duration = (est_duration_secs + duration_jitter).max(120); // minimum 2 minutes
        
        let dropoff_time = pickup_time + Duration::seconds(trip_duration);
        
        // Random passengers and fare
        let passenger_count = rng.gen_range(1..5) as u8;
        let fare_amount = (distance / 1000.0) * 2.5 + rng.gen_range(2.0..5.0); // ~$2.50/km base
        
        trips.push(Trip {
            pickup_time,
            dropoff_time,
            pickup_lon,
            pickup_lat,
            dropoff_lon,
            dropoff_lat,
            passenger_count,
            trip_distance: distance / 1000.0, // km
            fare_amount,
        });
        
        pb.inc(1);
    }
    
    pb.finish_and_clear();
    println!("✓ Generated {} synthetic trips", trips.len());
    
    Ok(trips)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🚕 NYC Rideshare Trajectory Generator");
    println!("======================================\n");

    // Get trips - either from file, download, or synthetic generation
    let trips = if args.synthetic {
        println!("🎲 Generating {} synthetic trips...", args.num_trips);
        generate_synthetic_trips(&args)?
    } else {
        // Get input file - either provided or download
        let input_path = match (&args.input, &args.download) {
            (Some(path), _) => path.clone(),
            (None, Some(year_month)) => download_tlc_data(year_month)?,
            (None, None) => {
                return Err(anyhow!(
                    "Either --input, --download, or --synthetic must be specified.\n\
                     Examples:\n\
                     --synthetic                    Generate synthetic trips routed through OSRM\n\
                     --input data.csv              Use a CSV file with lat/lon coordinates\n\
                     --download 2015-01            Download TLC data (note: may not have coordinates)"
                ));
            }
        };

        println!("📂 Reading trips from: {:?}", input_path);

        // Parse trips from Parquet (or CSV for legacy files)
        let trips = if input_path.extension().map(|e| e == "parquet").unwrap_or(false) {
            parse_tlc_parquet(&input_path, &args)?
        } else {
            parse_tlc_csv(&input_path, &args)?
        };
        
        if trips.is_empty() {
            return Err(anyhow!(
                "No valid trips found in input file.\n\
                 The file may use zone IDs instead of coordinates.\n\
                 Try using --synthetic mode instead, or provide a CSV with lat/lon columns."
            ));
        }
        
        trips
    };

    println!("✓ Loaded {} valid trips", trips.len());

    // Sample trips if max_trips is set
    let trips = if let Some(max) = args.max_trips {
        let mut rng = rand::thread_rng();
        let mut sampled: Vec<_> = trips.clone();
        sampled.shuffle(&mut rng);
        sampled.truncate(max);
        println!("📊 Sampled {} trips", sampled.len());
        sampled
    } else {
        trips
    };

    // Check OSRM connectivity if routing is enabled
    if !args.skip_routing {
        check_osrm_connectivity(&args.osrm_url)?;
    }

    // Generate trajectories
    let use_csv = common::is_csv_output(&args.output);
    if use_csv {
        generate_trajectories_csv(&trips, &args)?;
    } else {
        generate_trajectories_geojson(&trips, &args)?;
    }

    println!("\n✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output nyc-rideshare.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 10 \\");
    println!("             --max-zoom 16 \\");
    println!("             --compression gzip");

    Ok(())
}

/// Download NYC TLC Yellow Taxi data from NYC Open Data (2016 data with coordinates)
fn download_tlc_data(year_month: &str) -> Result<PathBuf> {
    // Parse year-month
    let parts: Vec<&str> = year_month.split('-').collect();
    if parts.len() != 2 {
        return Err(anyhow!("Invalid year-month format. Use YYYY-MM (e.g., 2016-01)"));
    }
    let year: u32 = parts[0].parse()?;
    let _month: u32 = parts[1].parse()?;

    // NYC Open Data 2016 dataset has actual coordinates (not zone IDs)
    // Dataset ID: uacg-pexx
    // API: https://data.cityofnewyork.us/resource/uacg-pexx.csv
    
    if year != 2016 {
        println!("⚠️  Note: Only 2016 data is available with coordinates on NYC Open Data.");
        println!("   Using 2016 Yellow Taxi Trip Data from NYC Open Data.");
    }

    let filename = "nyc-opendata-2016-taxi.csv".to_string();
    let output_path = PathBuf::from("data").join(&filename);

    if output_path.exists() {
        println!("📂 Using cached file: {:?}", output_path);
        return Ok(output_path);
    }

    // Download from NYC Open Data Socrata API
    // Note: Free API has 50,000 row limit without app token
    // For larger datasets, use $limit and $offset
    let limit = 50000;
    let url = format!(
        "https://data.cityofnewyork.us/resource/uacg-pexx.csv?$limit={}",
        limit
    );

    std::fs::create_dir_all("data")?;

    println!("📥 Downloading {} rows from NYC Open Data (2016 Yellow Taxi)...", limit);
    println!("   Source: https://data.cityofnewyork.us/d/uacg-pexx");
    common::download_file(&url, &output_path)?;
    
    println!("✓ Downloaded to {:?}", output_path);
    println!("   Note: For more data, download directly from NYC Open Data portal");
    
    Ok(output_path)
}

/// Parse TLC Parquet file and extract valid trips
fn parse_tlc_parquet(path: &PathBuf, args: &Args) -> Result<Vec<Trip>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    
    println!("📊 Parquet schema: {:?}", builder.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>());
    
    let reader = builder.build()?;

    let mut trips = Vec::new();
    let mut skipped_no_coords = 0;
    let mut skipped_out_of_bounds = 0;
    let mut skipped_duration = 0;
    let mut skipped_distance = 0;
    let mut total_rows = 0;

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} Reading trips: {pos} valid, {msg}")?,
    );

    for batch_result in reader {
        let batch = batch_result?;
        total_rows += batch.num_rows();

        // Try to get columns by various names (TLC schema changed over time)
        let pickup_time_col = get_timestamp_column(&batch, &["tpep_pickup_datetime", "pickup_datetime", "Trip_Pickup_DateTime"]);
        let dropoff_time_col = get_timestamp_column(&batch, &["tpep_dropoff_datetime", "dropoff_datetime", "Trip_Dropoff_DateTime"]);
        let pickup_lon_col = get_float_column(&batch, &["pickup_longitude", "Start_Lon"]);
        let pickup_lat_col = get_float_column(&batch, &["pickup_latitude", "Start_Lat"]);
        let dropoff_lon_col = get_float_column(&batch, &["dropoff_longitude", "End_Lon"]);
        let dropoff_lat_col = get_float_column(&batch, &["dropoff_latitude", "End_Lat"]);
        let passenger_col = get_int_column(&batch, &["passenger_count", "Passenger_Count"]);
        let distance_col = get_float_column(&batch, &["trip_distance", "Trip_Distance"]);
        let fare_col = get_float_column(&batch, &["fare_amount", "Fare_Amt"]);

        // Check if we have coordinate columns
        let (pickup_lon_col, pickup_lat_col, dropoff_lon_col, dropoff_lat_col) = match (pickup_lon_col, pickup_lat_col, dropoff_lon_col, dropoff_lat_col) {
            (Some(plon), Some(plat), Some(dlon), Some(dlat)) => (plon, plat, dlon, dlat),
            _ => {
                println!("⚠️  No coordinate columns found in this file.");
                println!("   This parquet file may use zone IDs instead of lat/lon coordinates.");
                println!("   Try using data from 2015 or earlier.");
                continue;
            }
        };

        let (pickup_time_col, dropoff_time_col) = match (pickup_time_col, dropoff_time_col) {
            (Some(p), Some(d)) => (p, d),
            _ => {
                println!("⚠️  No timestamp columns found");
                continue;
            }
        };

        for i in 0..batch.num_rows() {
            // Get coordinates
            let pickup_lon = pickup_lon_col.value(i);
            let pickup_lat = pickup_lat_col.value(i);
            let dropoff_lon = dropoff_lon_col.value(i);
            let dropoff_lat = dropoff_lat_col.value(i);

            // Skip null/zero coordinates
            if pickup_lon == 0.0 || pickup_lat == 0.0 || dropoff_lon == 0.0 || dropoff_lat == 0.0 {
                skipped_no_coords += 1;
                continue;
            }

            // Validate coordinates are in NYC area
            if !is_in_nyc(pickup_lon, pickup_lat) || !is_in_nyc(dropoff_lon, dropoff_lat) {
                skipped_out_of_bounds += 1;
                continue;
            }

            // Get timestamps (microseconds since epoch)
            let pickup_us = pickup_time_col.value(i);
            let dropoff_us = dropoff_time_col.value(i);
            
            let pickup_time = Utc.timestamp_micros(pickup_us).single()
                .ok_or_else(|| anyhow!("Invalid pickup timestamp"))?;
            let dropoff_time = Utc.timestamp_micros(dropoff_us).single()
                .ok_or_else(|| anyhow!("Invalid dropoff timestamp"))?;

            // Validate duration
            let duration = dropoff_time.signed_duration_since(pickup_time);
            if duration.num_minutes() > args.max_duration_minutes as i64 || duration.num_seconds() < 60 {
                skipped_duration += 1;
                continue;
            }

            // Validate distance
            let trip_distance = distance_col.as_ref().map(|c| c.value(i)).unwrap_or(0.0);
            let estimated_distance = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon);
            if estimated_distance < args.min_distance_meters {
                skipped_distance += 1;
                continue;
            }

            trips.push(Trip {
                pickup_time,
                dropoff_time,
                pickup_lon,
                pickup_lat,
                dropoff_lon,
                dropoff_lat,
                passenger_count: passenger_col.as_ref().map(|c| c.value(i) as u8).unwrap_or(1),
                trip_distance,
                fare_amount: fare_col.as_ref().map(|c| c.value(i)).unwrap_or(0.0),
            });

            if trips.len() % 10000 == 0 {
                pb.set_position(trips.len() as u64);
                pb.set_message(format!(
                    "skipped: {} no coords, {} out of bounds, {} duration, {} distance",
                    skipped_no_coords, skipped_out_of_bounds, skipped_duration, skipped_distance
                ));
            }
        }
    }

    pb.finish_and_clear();
    println!(
        "📊 Parsed {} trips from {} total rows, skipped: {} no coords, {} out of bounds, {} duration, {} distance",
        trips.len(),
        total_rows,
        skipped_no_coords,
        skipped_out_of_bounds,
        skipped_duration,
        skipped_distance
    );

    Ok(trips)
}

/// Helper to get a timestamp column by trying multiple names
fn get_timestamp_column<'a>(batch: &'a arrow::record_batch::RecordBatch, names: &[&str]) -> Option<&'a TimestampMicrosecondArray> {
    for name in names {
        if let Ok(col) = batch.column_by_name(name)
            .ok_or(())
            .and_then(|c| c.as_any().downcast_ref::<TimestampMicrosecondArray>().ok_or(())) 
        {
            return Some(col);
        }
    }
    None
}

/// Helper to get a float column by trying multiple names
fn get_float_column<'a>(batch: &'a arrow::record_batch::RecordBatch, names: &[&str]) -> Option<&'a Float64Array> {
    for name in names {
        if let Ok(col) = batch.column_by_name(name)
            .ok_or(())
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>().ok_or(())) 
        {
            return Some(col);
        }
    }
    None
}

/// Helper to get an int column by trying multiple names
fn get_int_column<'a>(batch: &'a arrow::record_batch::RecordBatch, names: &[&str]) -> Option<&'a Int64Array> {
    for name in names {
        if let Ok(col) = batch.column_by_name(name)
            .ok_or(())
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>().ok_or(())) 
        {
            return Some(col);
        }
    }
    None
}

/// Parse TLC CSV file (legacy format, pre-2017)
fn parse_tlc_csv(path: &PathBuf, args: &Args) -> Result<Vec<Trip>> {
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut csv_reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(reader);

    let mut trips = Vec::new();
    let mut skipped_no_coords = 0;
    let mut skipped_out_of_bounds = 0;
    let mut skipped_duration = 0;
    let mut skipped_distance = 0;
    let mut skipped_parse_error = 0;

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} Reading trips: {pos} valid, {msg}")?,
    );

    for result in csv_reader.deserialize() {
        let record: TlcTripRecord = match result {
            Ok(r) => r,
            Err(_) => {
                skipped_parse_error += 1;
                continue;
            }
        };

        // Validate coordinates exist
        let (pickup_lon, pickup_lat, dropoff_lon, dropoff_lat) = match (
            record.pickup_longitude,
            record.pickup_latitude,
            record.dropoff_longitude,
            record.dropoff_latitude,
        ) {
            (Some(plon), Some(plat), Some(dlon), Some(dlat)) => {
                if plon == 0.0 || plat == 0.0 || dlon == 0.0 || dlat == 0.0 {
                    skipped_no_coords += 1;
                    continue;
                }
                (plon, plat, dlon, dlat)
            }
            _ => {
                skipped_no_coords += 1;
                continue;
            }
        };

        // Validate coordinates are in NYC area
        if !is_in_nyc(pickup_lon, pickup_lat) || !is_in_nyc(dropoff_lon, dropoff_lat) {
            skipped_out_of_bounds += 1;
            continue;
        }

        // Parse timestamps
        let pickup_time = match parse_tlc_datetime(&record.pickup_datetime) {
            Ok(t) => t,
            Err(_) => {
                skipped_parse_error += 1;
                continue;
            }
        };
        let dropoff_time = match parse_tlc_datetime(&record.dropoff_datetime) {
            Ok(t) => t,
            Err(_) => {
                skipped_parse_error += 1;
                continue;
            }
        };

        // Validate duration
        let duration = dropoff_time.signed_duration_since(pickup_time);
        if duration.num_minutes() > args.max_duration_minutes as i64 || duration.num_seconds() < 60 {
            skipped_duration += 1;
            continue;
        }

        // Validate distance
        let trip_distance = record.trip_distance.unwrap_or(0.0);
        let estimated_distance = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon);
        if estimated_distance < args.min_distance_meters {
            skipped_distance += 1;
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
            trip_distance,
            fare_amount: record.fare_amount.unwrap_or(0.0),
        });

        if trips.len() % 10000 == 0 {
            pb.set_position(trips.len() as u64);
            pb.set_message(format!(
                "skipped: {} no coords, {} out of bounds, {} duration, {} distance",
                skipped_no_coords, skipped_out_of_bounds, skipped_duration, skipped_distance
            ));
        }
    }

    pb.finish_and_clear();
    println!(
        "📊 Parsed {} trips, skipped: {} no coords, {} out of bounds, {} duration, {} distance, {} parse errors",
        trips.len(),
        skipped_no_coords,
        skipped_out_of_bounds,
        skipped_duration,
        skipped_distance,
        skipped_parse_error
    );

    Ok(trips)
}

fn is_in_nyc(lon: f64, lat: f64) -> bool {
    lon >= NYC_MIN_LON && lon <= NYC_MAX_LON && lat >= NYC_MIN_LAT && lat <= NYC_MAX_LAT
}

/// Parse TLC datetime strings (various formats)
fn parse_tlc_datetime(s: &str) -> Result<DateTime<Utc>> {
    // Try different formats
    let formats = [
        "%Y-%m-%dT%H:%M:%S%.3f",  // ISO 8601 with milliseconds (NYC Open Data format)
        "%Y-%m-%dT%H:%M:%S",      // ISO 8601 without milliseconds
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
    ];

    for fmt in &formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s.trim(), fmt) {
            return Ok(DateTime::from_naive_utc_and_offset(dt, Utc));
        }
    }

    Err(anyhow!("Could not parse datetime: {}", s))
}

/// Check OSRM server connectivity
fn check_osrm_connectivity(osrm_url: &str) -> Result<()> {
    println!("🔍 Checking OSRM connectivity at {}...", osrm_url);

    // Try a simple route request with geometry
    let test_url = format!(
        "{}/route/v1/driving/-73.99,40.73;-73.98,40.74?overview=full&geometries=geojson",
        osrm_url
    );

    let response = reqwest::blocking::get(&test_url)
        .context("Failed to connect to OSRM server. Is it running?")?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "OSRM server returned error status: {}. \
             Make sure OSRM is running with NYC data.\n\
             See setup-osrm.sh for instructions.",
            response.status()
        ));
    }

    // Just check for valid JSON with "Ok" code
    let body: serde_json::Value = response.json()
        .context("Invalid JSON response from OSRM server")?;

    let code = body.get("code").and_then(|c| c.as_str()).unwrap_or("");
    if code != "Ok" {
        return Err(anyhow!("OSRM returned code: {}", code));
    }

    println!("✓ OSRM server is ready");
    Ok(())
}

/// Query OSRM for a route between two points
fn get_osrm_route(
    osrm_url: &str,
    from_lon: f64,
    from_lat: f64,
    to_lon: f64,
    to_lat: f64,
) -> Result<Option<OsrmRoute>> {
    let url = format!(
        "{}/route/v1/driving/{:.6},{:.6};{:.6},{:.6}?overview=full&geometries=geojson&steps=true",
        osrm_url, from_lon, from_lat, to_lon, to_lat
    );

    let response = reqwest::blocking::get(&url)?;
    let body: OsrmRouteResponse = response.json()?;

    if body.code != "Ok" {
        return Ok(None);
    }

    Ok(body.routes.and_then(|mut r| r.pop()))
}

/// Generate trajectory points along a route
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
    let route_duration = route.duration;

    // Calculate cumulative distances for each segment
    let mut cumulative_distances: Vec<f64> = vec![0.0];
    let mut total_distance = 0.0;

    for i in 1..coords.len() {
        let dist = haversine_distance(
            coords[i - 1][1], coords[i - 1][0],
            coords[i][1], coords[i][0],
        );
        total_distance += dist;
        cumulative_distances.push(total_distance);
    }

    if total_distance == 0.0 {
        return vec![];
    }

    // Generate points at regular time intervals
    let mut points = Vec::new();
    let num_intervals = (total_duration / interval_secs as f64).ceil() as i64;

    for i in 0..=num_intervals {
        let t = (i as f64 * interval_secs as f64).min(total_duration);
        let fraction = t / total_duration;
        let target_distance = fraction * total_distance;

        // Find the segment containing this distance
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

        // Interpolate position within segment
        let start_coord = &coords[segment_idx];
        let end_coord = coords.get(segment_idx + 1).unwrap_or(start_coord);

        let lon = start_coord[0] + (end_coord[0] - start_coord[0]) * segment_fraction;
        let lat = start_coord[1] + (end_coord[1] - start_coord[1]) * segment_fraction;
        let timestamp = start_time + Duration::seconds(t as i64);

        points.push((lon, lat, timestamp));
    }

    points
}

/// Generate trajectories and write to CSV
fn generate_trajectories_csv(trips: &[Trip], args: &Args) -> Result<()> {
    println!("\n🚀 Generating trajectories (CSV output)...");

    let property_columns = vec![
        "trip_id".to_string(),
        "passenger_count".to_string(),
        "trip_distance".to_string(),
        "fare_amount".to_string(),
        "status".to_string(),
    ];

    let mut csv_writer = common::StreamingCsvWriter::new(&args.output, property_columns)?;

    let pb = ProgressBar::new(trips.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} trips ({eta})")?
            .progress_chars("=>-"),
    );

    let mut total_points = 0;
    let mut failed_routes = 0;

    for (trip_id, trip) in trips.iter().enumerate() {
        let points = if args.skip_routing {
            // Just emit pickup and dropoff points
            vec![
                (trip.pickup_lon, trip.pickup_lat, trip.pickup_time),
                (trip.dropoff_lon, trip.dropoff_lat, trip.dropoff_time),
            ]
        } else {
            // Get route from OSRM
            match get_osrm_route(
                &args.osrm_url,
                trip.pickup_lon,
                trip.pickup_lat,
                trip.dropoff_lon,
                trip.dropoff_lat,
            ) {
                Ok(Some(route)) => {
                    interpolate_route(&route, trip.pickup_time, trip.dropoff_time, args.interval)
                }
                Ok(None) => {
                    failed_routes += 1;
                    // Fallback to straight line
                    vec![
                        (trip.pickup_lon, trip.pickup_lat, trip.pickup_time),
                        (trip.dropoff_lon, trip.dropoff_lat, trip.dropoff_time),
                    ]
                }
                Err(_) => {
                    failed_routes += 1;
                    vec![
                        (trip.pickup_lon, trip.pickup_lat, trip.pickup_time),
                        (trip.dropoff_lon, trip.dropoff_lat, trip.dropoff_time),
                    ]
                }
            }
        };

        // Write trajectory points
        for (i, (lon, lat, timestamp)) in points.iter().enumerate() {
            let mut properties = Map::new();
            properties.insert("trip_id".to_string(), json!(trip_id));
            properties.insert("passenger_count".to_string(), json!(trip.passenger_count));
            properties.insert("trip_distance".to_string(), json!(trip.trip_distance));
            properties.insert("fare_amount".to_string(), json!(trip.fare_amount));

            // Status: pickup, enroute, dropoff
            let status = if i == 0 {
                "pickup"
            } else if i == points.len() - 1 {
                "dropoff"
            } else {
                "enroute"
            };
            properties.insert("status".to_string(), json!(status));

            csv_writer.write_point(*lon, *lat, *timestamp, &properties)?;
            total_points += 1;
        }

        pb.inc(1);
    }

    pb.finish_and_clear();
    let row_count = csv_writer.finish()?;

    println!("📊 Generated {} trajectory points from {} trips", total_points, trips.len());
    if failed_routes > 0 {
        println!("⚠️  {} trips used fallback routing (OSRM failures)", failed_routes);
    }

    Ok(())
}

/// Generate trajectories and write to GeoJSON
fn generate_trajectories_geojson(trips: &[Trip], args: &Args) -> Result<()> {
    println!("\n🚀 Generating trajectories (GeoJSON output)...");

    let pb = ProgressBar::new(trips.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} trips ({eta})")?
            .progress_chars("=>-"),
    );

    let mut features = Vec::new();
    let mut failed_routes = 0;

    for (trip_id, trip) in trips.iter().enumerate() {
        let points = if args.skip_routing {
            vec![
                (trip.pickup_lon, trip.pickup_lat, trip.pickup_time),
                (trip.dropoff_lon, trip.dropoff_lat, trip.dropoff_time),
            ]
        } else {
            match get_osrm_route(
                &args.osrm_url,
                trip.pickup_lon,
                trip.pickup_lat,
                trip.dropoff_lon,
                trip.dropoff_lat,
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

        for (i, (lon, lat, timestamp)) in points.iter().enumerate() {
            let mut properties = Map::new();
            properties.insert("trip_id".to_string(), json!(trip_id));
            properties.insert("passenger_count".to_string(), json!(trip.passenger_count));
            properties.insert("trip_distance".to_string(), json!(trip.trip_distance));
            properties.insert("fare_amount".to_string(), json!(trip.fare_amount));

            let status = if i == 0 {
                "pickup"
            } else if i == points.len() - 1 {
                "dropoff"
            } else {
                "enroute"
            };
            properties.insert("status".to_string(), json!(status));

            let feature = common::create_point_feature(*lon, *lat, *timestamp, properties);
            features.push(feature);
        }

        pb.inc(1);
    }

    pb.finish_and_clear();
    println!("📊 Generated {} trajectory points from {} trips", features.len(), trips.len());
    if failed_routes > 0 {
        println!("⚠️  {} trips used fallback routing (OSRM failures)", failed_routes);
    }

    common::write_geojson(features, &args.output)?;
    Ok(())
}

/// Calculate haversine distance between two points in meters
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS: f64 = 6371000.0; // meters

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS * c
}

