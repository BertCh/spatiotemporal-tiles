//! Generate flight density data
//!
//! ## Real ADS-B Data Source
//!
//! **Provider:** ADSBExchange (Free for 1st of each month)
//! **Base URL:** https://samples.adsbexchange.com/hires-traces
//!
//! ### Coverage
//! - **Geographic:** Global coverage
//! - **Temporal:** 2020 - Present (hourly snapshots)
//! - **Availability:** Only 1st of each month freely available
//! - **File Format:** Gzip-compressed JSON traces
//! - **Organization:** Files bucketed by ICAO hex prefix (00-ff)
//!
//! ### JSON Structure
//! Each trace file contains one aircraft's full day of positions:
//! ```json
//! {
//!   "icao": "a12345",
//!   "r": "N12345",      // Registration
//!   "t": "B738",        // Aircraft type
//!   "trace": [
//!     [timestamp, lat, lon, alt, gs, track, ...],
//!     ...
//!   ]
//! }
//! ```
//!
//! ### Trace Fields
//! - Index 0: Unix timestamp (seconds)
//! - Index 1: Latitude (decimal degrees)
//! - Index 2: Longitude (decimal degrees)
//! - Index 3: Altitude (feet, barometric)
//! - Index 4: Ground speed (knots)
//! - Index 5: Track angle (degrees)
//!
//! ### Download Example
//! ```bash
//! # Download specific aircraft trace for January 1, 2025
//! # Files are organized by ICAO prefix (first 2 hex digits)
//! # Example: aircraft a12345 is in bucket "a1"
//! curl -o trace_a12345.json \
//!   https://samples.adsbexchange.com/hires-traces/2025/01/01/a1/trace_full_~a12345.json
//!
//! # Note: You must know the ICAO codes in advance
//! # Use ADSBExchange index files or the API to get ICAO lists
//! ```
//!
//! ### Important Notes
//! - Only the 1st of each month is freely available
//! - Must know ICAO codes in advance (use index files)
//! - Files organized in 256 buckets (00-ff) based on ICAO prefix
//! - Global coverage but density varies by region
//!
//! ### References
//! - ADS-B Exchange: https://www.adsbexchange.com/data-samples/
//! - Data Format: https://www.adsbexchange.com/version-2-api-wip/
//!
//! ---
//!
//! This script generates synthetic flight data for demonstration purposes.
//! For production use, download and process real ADS-B traces from ADSBExchange.

mod common;

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, Utc};
use clap::Parser;
use geojson::Feature;
use rand::Rng;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-flight-data")]
#[command(about = "Generate synthetic flight density data")]
#[command(long_about = "
Generate synthetic flight density data for demonstration.

REAL DATA SOURCE:
  Provider: ADSBExchange (Free for 1st of each month)
  URL:      https://samples.adsbexchange.com/hires-traces
  Format:   Gzip-compressed JSON traces (per aircraft)
  Coverage: Global, 2020-Present
  Files:    Organized by ICAO prefix buckets (00-ff)

DOWNLOAD EXAMPLE:
  # Aircraft a12345 on January 1, 2025 (bucket \"a1\")
  curl -o trace_a12345.json \\
    https://samples.adsbexchange.com/hires-traces/2025/01/01/a1/trace_full_~a12345.json

NOTE: Only the 1st of each month is freely available.
      You must know ICAO codes in advance.

For production use, download and process real ADS-B traces.
")]
struct Args {
    /// Output GeoJSON file
    #[arg(short, long, default_value = "flight-density.geojson")]
    output: PathBuf,

    /// Date to simulate (YYYY-MM-DD)
    #[arg(long, default_value = "2024-01-01")]
    date: String,

    /// Number of flights to simulate
    #[arg(long, default_value = "1000")]
    num_flights: usize,
}

// Major North American airports
const AIRPORTS: &[(&str, f64, f64)] = &[
    ("ATL", 33.6367, -84.4281),
    ("LAX", 33.9425, -118.4081),
    ("ORD", 41.9786, -87.9047),
    ("DFW", 32.8968, -97.0380),
    ("DEN", 39.8561, -104.6737),
    ("JFK", 40.6413, -73.7781),
    ("SFO", 37.6213, -122.3790),
    ("SEA", 47.4502, -122.3088),
    ("LAS", 36.0840, -115.1537),
    ("MCO", 28.4312, -81.3081),
];

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("✈️  Flight Density Generator");
    println!("============================\n");
    println!("⚠️  Note: Using SYNTHETIC data for demonstration");
    println!();
    println!("📍 Real ADS-B Data Available From:");
    println!("   Provider: ADSBExchange (Free for 1st of each month)");
    println!("   URL: https://samples.adsbexchange.com/hires-traces");
    println!("   Coverage: Global, 2020-Present");
    println!("   Format: Gzip JSON traces (per aircraft, 50-500 KB/day)");
    println!();
    println!("   Example download (aircraft a12345 on Jan 1, 2025):");
    println!("   curl -o trace_a12345.json \\");
    println!(
        "     https://samples.adsbexchange.com/hires-traces/2025/01/01/a1/trace_full_~a12345.json"
    );
    println!();
    println!("   Note: Files organized in 256 buckets (00-ff) by ICAO prefix");
    println!("         Only 1st of each month freely available");
    println!();

    let date = NaiveDate::parse_from_str(&args.date, "%Y-%m-%d")?;
    let start_time = common::date_to_datetime(date);

    println!("📊 Generating {} flights...", args.num_flights);
    let features = generate_flight_data(start_time, args.num_flights)?;

    println!("\n💾 Writing output...");
    common::write_geojson(features, &args.output)?;

    println!("\n✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output flight-density.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 3 \\");
    println!("             --max-zoom 10 \\");
    println!("             --compression gzip");

    Ok(())
}

fn generate_flight_data(start_time: DateTime<Utc>, num_flights: usize) -> Result<Vec<Feature>> {
    let mut rng = rand::thread_rng();
    let mut features = Vec::new();

    // Generate 24 hours of data in 5-minute intervals
    let intervals = (24 * 60) / 5;

    for interval in 0..intervals {
        let timestamp = start_time + Duration::minutes(interval * 5);

        // More flights during day, fewer at night
        let hour = interval / 12;
        let flight_multiplier = if hour >= 6 && hour <= 22 { 1.5 } else { 0.5 };
        let num_this_interval =
            (num_flights as f64 * flight_multiplier / intervals as f64) as usize;

        for _ in 0..num_this_interval {
            // Pick random route
            let origin_idx = rng.gen_range(0..AIRPORTS.len());
            let mut dest_idx = rng.gen_range(0..AIRPORTS.len());
            while dest_idx == origin_idx {
                dest_idx = rng.gen_range(0..AIRPORTS.len());
            }

            let origin = AIRPORTS[origin_idx];
            let dest = AIRPORTS[dest_idx];

            // Random position along route
            let progress = rng.gen_range(0.0..1.0);
            let lon = origin.1 + (dest.1 - origin.1) * progress;
            let lat = origin.2 + (dest.2 - origin.2) * progress;

            let mut properties = Map::new();
            properties.insert("origin".to_string(), json!(origin.0));
            properties.insert("destination".to_string(), json!(dest.0));
            properties.insert("altitude".to_string(), json!(rng.gen_range(30000..42000)));
            properties.insert("speed".to_string(), json!(rng.gen_range(450..550)));
            properties.insert("weight".to_string(), json!(1));

            let feature = common::create_point_feature(lon, lat, timestamp, properties);
            features.push(feature);
        }

        if interval % 60 == 0 {
            println!("  Generated data for hour {}...", interval / 12);
        }
    }

    Ok(features)
}
