//! Generate bike share trip data
//!
//! Uses synthetic data based on NYC Citi Bike patterns

mod common;

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, Utc};
use clap::Parser;
use geojson::Feature;
use rand::Rng;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-bikeshare-data")]
#[command(about = "Generate bike share trip data")]
struct Args {
    /// Output GeoJSON file
    #[arg(short, long, default_value = "bike-share.geojson")]
    output: PathBuf,

    /// Date to simulate
    #[arg(long, default_value = "2024-01-01")]
    date: String,

    /// Number of trips
    #[arg(long, default_value = "10000")]
    num_trips: usize,
}

// NYC bike share station locations (subset)
const STATIONS: &[(f64, f64, &str)] = &[
    (-73.9919, 40.7262, "Union Square"),
    (-73.9807, 40.7673, "Central Park S"),
    (-74.0059, 40.7406, "Chelsea"),
    (-73.9964, 40.7298, "Flatiron"),
    (-74.0127, 40.7205, "West Village"),
    (-73.9813, 40.7527, "Times Square"),
    (-73.9871, 40.7589, "Columbus Circle"),
    (-74.0061, 40.7195, "Tribeca"),
    (-73.9972, 40.7614, "Hell's Kitchen"),
    (-73.9629, 40.7644, "Upper East Side"),
];

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🚴 Bike Share Data Generator");
    println!("============================\n");
    println!("⚠️  Note: Using synthetic data based on NYC patterns");
    println!();

    let date = NaiveDate::parse_from_str(&args.date, "%Y-%m-%d")?;
    let start_time = common::date_to_datetime(date);

    println!("📊 Generating {} trips...", args.num_trips);
    let features = generate_bikeshare_data(start_time, args.num_trips)?;

    println!("\n💾 Writing output...");
    common::write_geojson(features, &args.output)?;

    println!("\n✅ Success! Now run:");
    println!("   stt-build --input {} --output bike-share.stt \\", args.output.display());
    println!("             --time-field timestamp \\");
    println!("             --temporal-bucket hour \\");
    println!("             --min-zoom 11 \\");
    println!("             --max-zoom 16 \\");
    println!("             --compression gzip");
    println!("\n💡 Temporal bucketing: hour (for usage patterns) or day (for demand analysis)");

    Ok(())
}

fn generate_bikeshare_data(start_time: DateTime<Utc>, num_trips: usize) -> Result<Vec<Feature>> {
    let mut rng = rand::thread_rng();
    let mut features = Vec::new();

    // Generate trips throughout the day
    let intervals = 24 * 12; // 5-minute intervals
    let trips_per_interval = num_trips / intervals;

    for interval in 0..intervals {
        let timestamp = start_time + Duration::minutes((interval * 5) as i64);
        let hour = interval / 12;

        // More trips during rush hours
        let multiplier = match hour {
            7..=9 => 2.0,   // Morning rush
            17..=19 => 2.5, // Evening rush
            11..=14 => 1.5, // Lunch
            _ => 0.5,
        };

        let num_this_interval = (trips_per_interval as f64 * multiplier) as usize;

        for _ in 0..num_this_interval {
            // Pick random origin and destination
            let origin_idx = rng.gen_range(0..STATIONS.len());
            let mut dest_idx = rng.gen_range(0..STATIONS.len());
            while dest_idx == origin_idx {
                dest_idx = rng.gen_range(0..STATIONS.len());
            }

            let origin = STATIONS[origin_idx];
            let dest = STATIONS[dest_idx];

            // Trip duration (5-30 minutes)
            let duration_min = rng.gen_range(5..30);

            // Create origin point
            let mut properties = Map::new();
            properties.insert("station".to_string(), json!(origin.2));
            properties.insert("trip_type".to_string(), json!("origin"));
            properties.insert("duration".to_string(), json!(duration_min));
            properties.insert("weight".to_string(), json!(1));

            let feature = common::create_point_feature(origin.0, origin.1, timestamp, properties);
            features.push(feature);

            // Create destination point
            let end_time = timestamp + Duration::minutes(duration_min);
            let mut properties = Map::new();
            properties.insert("station".to_string(), json!(dest.2));
            properties.insert("trip_type".to_string(), json!("destination"));
            properties.insert("duration".to_string(), json!(duration_min));
            properties.insert("weight".to_string(), json!(1));

            let feature = common::create_point_feature(dest.0, dest.1, end_time, properties);
            features.push(feature);
        }

        if interval % 24 == 0 {
            println!("  Generated hour {}/24...", interval / 12);
        }
    }

    Ok(features)
}

