//! Generate NYC rideshare trajectory data using real TLC trip records + OSRM routing
//!
//! Uses historical NYC TLC data (pre-2017 with actual lat/long coordinates)
//! and routes trips through OSRM for realistic trajectories.

use crate::common;
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::Deserialize;
use serde_json::{json, Map};
use std::fs::File;
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

    /// Skip stt-build step
    #[arg(long)]
    pub skip_build: bool,
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
    duration: f64,
    #[allow(dead_code)]
    distance: f64,
}

#[derive(Debug, Deserialize)]
struct OsrmGeometry {
    coordinates: Vec<Vec<f64>>,
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

pub fn run(args: Args) -> Result<()> {
    println!("🚕 NYC Rideshare Trajectory Generator");
    println!("======================================\n");

    // Determine intermediate output format
    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("geojson")
    } else {
        args.output.clone()
    };

    // Get trips
    let trips = if args.synthetic {
        println!("🎲 Generating {} synthetic trips...", args.num_trips);
        generate_synthetic_trips(&args)?
    } else if let Some(ref input_path) = args.input {
        println!("📂 Reading trips from: {:?}", input_path);
        parse_tlc_csv(input_path, &args)?
    } else {
        return Err(anyhow!(
            "Either --input or --synthetic must be specified.\n\
             Examples:\n\
             --synthetic                    Generate synthetic trips\n\
             --input data.csv              Use a CSV file with lat/lon coordinates"
        ));
    };

    if trips.is_empty() {
        return Err(anyhow!("No valid trips found"));
    }

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
    if args.paths {
        generate_paths_geojson(&trips, &args, &intermediate_path)?;
    } else {
        generate_trajectories_geojson(&trips, &args, &intermediate_path)?;
    }

    // Build STT if output is .stt
    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        let time_field = "timestamp";
        let end_time_field = if args.paths { Some("end_time") } else { None };

        if end_time_field.is_some() {
            // For paths, we need to use stt-build with end-time-field
            use std::process::Command;
            println!("\n📦 Building STT archive...");
            let mut cmd = Command::new("stt-build");
            cmd.arg("--input").arg(&intermediate_path)
                .arg("--output").arg(&args.output)
                .arg("--time-field").arg(time_field)
                .arg("--end-time-field").arg("end_time")
                .arg("--min-zoom").arg("10")
                .arg("--max-zoom").arg("16")
                .arg("--compression").arg("gzip");

            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("stt-build failed");
            }
            println!("✅ STT archive created: {}", args.output.display());
        } else {
            common::run_stt_build(
                &intermediate_path,
                &args.output,
                time_field,
                10,
                16,
                "gzip",
            )?;
        }

        // Clean up intermediate file
        let _ = std::fs::remove_file(&intermediate_path);
    }

    println!("\n✅ NYC rideshare data generation complete!");

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
    let url = format!(
        "{}/route/v1/driving/{:.6},{:.6};{:.6},{:.6}?overview=full&geometries=geojson",
        osrm_url, from_lon, from_lat, to_lon, to_lat
    );

    let response = reqwest::blocking::get(&url)?;
    let body: OsrmRouteResponse = response.json()?;

    if body.code != "Ok" {
        return Ok(None);
    }

    Ok(body.routes.and_then(|mut r| r.pop()))
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

fn generate_trajectories_geojson(trips: &[Trip], args: &Args, output: &PathBuf) -> Result<()> {
    println!("\n🚀 Generating trajectories...");

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
    println!("📊 Generated {} trajectory points", features.len());
    if failed_routes > 0 {
        println!("⚠️  {} trips used fallback routing", failed_routes);
    }

    common::write_geojson(features, output)?;
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


