//! Generate synthetic taxi trajectory data
//!
//! Simulates realistic taxi movements in San Francisco

mod common;

use anyhow::Result;
use chrono::{Duration, NaiveDate};
use clap::Parser;
use rand::Rng;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-taxi-data")]
#[command(about = "Generate synthetic taxi trajectory data")]
struct Args {
    /// Output file (use .csv for streaming output, .geojson for JSON)
    #[arg(short, long, default_value = "sf-taxis.csv")]
    output: PathBuf,

    /// Number of taxis to simulate
    #[arg(long, default_value = "100")]
    num_taxis: usize,

    /// Simulation date
    #[arg(long, default_value = "2024-01-15")]
    date: String,

    /// Update interval in seconds
    #[arg(long, default_value = "60")]
    interval: i64,
}

// San Francisco bounding box
const SF_MIN_LON: f64 = -122.52;
const SF_MAX_LON: f64 = -122.35;
const SF_MIN_LAT: f64 = 37.70;
const SF_MAX_LAT: f64 = 37.81;

#[derive(Clone, Copy, Debug)]
enum TaxiStatus {
    Available,
    Occupied,
    EnRoute,
}

impl TaxiStatus {
    fn to_string(&self) -> &str {
        match self {
            TaxiStatus::Available => "available",
            TaxiStatus::Occupied => "occupied",
            TaxiStatus::EnRoute => "enroute",
        }
    }
}

struct TaxiState {
    id: usize,
    lon: f64,
    lat: f64,
    status: TaxiStatus,
    speed: f64, // km/h
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🚕 Taxi Trajectory Generator");
    println!("============================\n");

    println!("🏙️  Simulating {} taxis in San Francisco", args.num_taxis);
    println!("📅 Date: {}", args.date);
    println!("⏱️  Update interval: {} seconds", args.interval);

    let date = NaiveDate::parse_from_str(&args.date, "%Y-%m-%d")?;
    let start_time = common::date_to_datetime(date);

    let mut taxis = initialize_taxis(args.num_taxis);
    let mut rng = rand::thread_rng();

    // Simulate 24 hours
    let num_steps = (24 * 60 * 60) / args.interval;
    println!("\n🔄 Simulating {} time steps...", num_steps);

    let use_csv = common::is_csv_output(&args.output);
    if use_csv {
        println!("📄 Using streaming CSV output (memory-efficient)");
    }

    // Create streaming CSV writer or collect features for GeoJSON
    let property_columns = vec![
        "taxi_id".to_string(),
        "status".to_string(),
        "speed".to_string(),
    ];

    if use_csv {
        // Streaming CSV mode - write directly to disk
        let mut csv_writer = common::StreamingCsvWriter::new(&args.output, property_columns)?;

        for step in 0..num_steps {
            let timestamp = start_time + Duration::seconds(step * args.interval);

            for taxi in &mut taxis {
                update_taxi_position(taxi, &mut rng, args.interval);

                let mut properties = Map::new();
                properties.insert("taxi_id".to_string(), json!(taxi.id));
                properties.insert("status".to_string(), json!(taxi.status.to_string()));
                properties.insert("speed".to_string(), json!(taxi.speed));

                csv_writer.write_point(taxi.lon, taxi.lat, timestamp, &properties)?;
            }

            if step % 60 == 0 {
                println!(
                    "  Step {}/{} ({:.1}%) - {} rows written",
                    step,
                    num_steps,
                    (step as f64 / num_steps as f64) * 100.0,
                    csv_writer.row_count()
                );
            }
        }

        let row_count = csv_writer.finish()?;
        println!("\n📊 Generated {} trajectory points", row_count);
    } else {
        // GeoJSON mode - collect all features in memory
        let mut features = Vec::new();

        for step in 0..num_steps {
            let timestamp = start_time + Duration::seconds(step * args.interval);

            for taxi in &mut taxis {
                update_taxi_position(taxi, &mut rng, args.interval);

                let mut properties = Map::new();
                properties.insert("taxi_id".to_string(), json!(taxi.id));
                properties.insert("status".to_string(), json!(taxi.status.to_string()));
                properties.insert("speed".to_string(), json!(taxi.speed));

                let feature = common::create_point_feature(taxi.lon, taxi.lat, timestamp, properties);
                features.push(feature);
            }

            if step % 60 == 0 {
                println!(
                    "  Step {}/{} ({:.1}%)",
                    step,
                    num_steps,
                    (step as f64 / num_steps as f64) * 100.0
                );
            }
        }

        println!("\n📊 Generated {} trajectory points", features.len());
        println!("\n💾 Writing output...");
        common::write_geojson(features, &args.output)?;
    }

    println!("\n✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output sf-taxis.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 10 \\");
    println!("             --max-zoom 16 \\");
    println!("             --compression gzip");

    Ok(())
}

fn initialize_taxis(count: usize) -> Vec<TaxiState> {
    let mut rng = rand::thread_rng();
    let mut taxis = Vec::new();

    for id in 0..count {
        taxis.push(TaxiState {
            id,
            lon: rng.gen_range(SF_MIN_LON..SF_MAX_LON),
            lat: rng.gen_range(SF_MIN_LAT..SF_MAX_LAT),
            status: if rng.gen_bool(0.7) {
                TaxiStatus::Available
            } else {
                TaxiStatus::Occupied
            },
            speed: rng.gen_range(0.0..50.0),
        });
    }

    taxis
}

fn update_taxi_position(taxi: &mut TaxiState, rng: &mut impl Rng, interval: i64) {
    // Use proper coordinate math instead of rough approximation
    let speed_mps = (taxi.speed * 1000.0) / 3600.0; // Convert km/h to m/s
    let distance_m = speed_mps * interval as f64;

    // Convert to degrees using proper Web Mercator math
    // At the equator, 1 degree ≈ 111.32 km
    // Adjust for latitude: distance in lon degrees = distance_m / (111320 * cos(lat))
    let lat_rad = taxi.lat.to_radians();
    let meters_per_degree_lon = 111320.0 * lat_rad.cos();
    let meters_per_degree_lat = 111320.0;

    let distance_deg_lon = distance_m / meters_per_degree_lon;
    let distance_deg_lat = distance_m / meters_per_degree_lat;

    // Random direction
    let angle = rng.gen_range(0.0..std::f64::consts::TAU);
    let dx = angle.cos() * distance_deg_lon;
    let dy = angle.sin() * distance_deg_lat;

    taxi.lon += dx;
    taxi.lat += dy;

    // Keep within San Francisco bounds
    taxi.lon = taxi.lon.clamp(SF_MIN_LON, SF_MAX_LON);
    taxi.lat = taxi.lat.clamp(SF_MIN_LAT, SF_MAX_LAT);

    // Randomly change status and speed
    if rng.gen_bool(0.05) {
        taxi.status = match rng.gen_range(0..3) {
            0 => TaxiStatus::Available,
            1 => TaxiStatus::Occupied,
            _ => TaxiStatus::EnRoute,
        };
        taxi.speed = rng.gen_range(0.0..50.0);
    }
}
