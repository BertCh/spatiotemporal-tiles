//! Download and process real flight data from OpenSky Network
//!
//! ## Data Sources
//!
//! ### OpenSky Network (Primary - FREE)
//!
//! **Provider:** OpenSky Network (https://opensky-network.org)
//! **API:** REST API with anonymous access
//! **Rate Limits:**
//!   - Anonymous: 10 requests/minute, data updated every 10 seconds
//!   - Registered (free): 100 requests/minute, data updated every 5 seconds
//!
//! ### API Endpoint
//! ```
//! GET https://opensky-network.org/api/states/all
//! ```
//!
//! ### Response Fields (state vector array)
//! - Index 0: icao24 - ICAO24 transponder address (hex)
//! - Index 1: callsign - Callsign (8 chars max)
//! - Index 2: origin_country - Country of origin
//! - Index 3: time_position - Unix timestamp of last position update
//! - Index 4: last_contact - Unix timestamp of last message received
//! - Index 5: longitude - WGS-84 longitude
//! - Index 6: latitude - WGS-84 latitude
//! - Index 7: baro_altitude - Barometric altitude in meters
//! - Index 8: on_ground - Boolean indicating if on ground
//! - Index 9: velocity - Velocity in m/s
//! - Index 10: true_track - True track in degrees (clockwise from north)
//! - Index 11: vertical_rate - Vertical rate in m/s
//! - Index 12: sensors - IDs of sensors contributing to state
//! - Index 13: geo_altitude - Geometric altitude in meters
//! - Index 14: squawk - Transponder code
//! - Index 15: spi - Special purpose indicator
//! - Index 16: position_source - Origin of position (0=ADS-B, 1=ASTERIX, 2=MLAT, 3=FLARM)
//!
//! ### Geographic Filtering
//! ```
//! GET https://opensky-network.org/api/states/all?lamin=25&lomin=-125&lamax=50&lomax=-65
//! ```
//!
//! ### ADSBExchange (Alternative - Historical)
//!
//! **Provider:** ADSBExchange (https://www.adsbexchange.com/data-samples/)
//! **Availability:** Only 1st of each month freely available
//! **Format:** Gzip-compressed JSON traces per aircraft
//! **URL Pattern:** https://samples.adsbexchange.com/hires-traces/YYYY/MM/DD/XX/trace_full_~ICAO.json
//!
//! ---

mod common;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use clap::{Parser, ValueEnum};
use geojson::Feature;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::thread;
use std::time::Duration as StdDuration;

#[derive(Debug, Clone, ValueEnum)]
enum DataSource {
    /// Download real-time data from OpenSky Network API
    Opensky,
    /// Use synthetic data (fallback for demos)
    Synthetic,
}

#[derive(Parser, Debug)]
#[command(name = "generate-flight-data")]
#[command(about = "Download real flight data from OpenSky Network")]
#[command(long_about = "
Download and process real flight data from OpenSky Network.

DATA SOURCES:
  OpenSky Network - Free real-time ADS-B data
  API: https://opensky-network.org/api/states/all
  
  Rate limits (anonymous):
    - 10 requests/minute
    - Data updates every 10 seconds

USAGE EXAMPLES:

  # Collect 1 hour of US flight data (120 samples at 30-second intervals)
  cargo run --release --bin generate-flight-data -- \\
    --source opensky \\
    --duration 3600 \\
    --interval 30 \\
    --bounds 25,-125,50,-65 \\
    --output flights-us.geojson

  # Collect 30 minutes of global flight data
  cargo run --release --bin generate-flight-data -- \\
    --source opensky \\
    --duration 1800 \\
    --output flights-global.geojson

  # Use synthetic data for testing
  cargo run --release --bin generate-flight-data -- \\
    --source synthetic \\
    --num-flights 1000 \\
    --output flights-synthetic.geojson

BUILDING STT FILE:
  stt-build --input flights-us.geojson --output flights.stt \\
            --time-field timestamp --min-zoom 0 --max-zoom 10 --compression gzip
")]
struct Args {
    /// Data source to use
    #[arg(long, value_enum, default_value = "opensky")]
    source: DataSource,

    /// Output GeoJSON file
    #[arg(short, long, default_value = "flights.geojson")]
    output: PathBuf,

    /// Collection duration in seconds (for opensky source)
    #[arg(long, default_value = "3600")]
    duration: u64,

    /// Sample interval in seconds (for opensky source)
    /// Anonymous rate limit: 10 req/min = 6 second minimum
    #[arg(long, default_value = "30")]
    interval: u64,

    /// Geographic bounds: min_lat,min_lon,max_lat,max_lon
    /// Example: 25,-125,50,-65 (Continental US)
    #[arg(long)]
    bounds: Option<String>,

    /// Number of flights (synthetic source only)
    #[arg(long, default_value = "1000")]
    num_flights: usize,

    /// Date to simulate (synthetic source only, YYYY-MM-DD)
    #[arg(long, default_value = "2024-01-01")]
    date: String,
}

/// OpenSky API response
#[derive(Debug, Deserialize)]
struct OpenSkyResponse {
    time: i64,
    states: Option<Vec<StateVector>>,
}

/// Individual state vector from OpenSky
/// Fields are positional in the JSON array
#[derive(Debug)]
struct StateVector {
    icao24: String,
    callsign: Option<String>,
    origin_country: String,
    time_position: Option<i64>,
    last_contact: i64,
    longitude: Option<f64>,
    latitude: Option<f64>,
    baro_altitude: Option<f64>,
    on_ground: bool,
    velocity: Option<f64>,
    true_track: Option<f64>,
    vertical_rate: Option<f64>,
    geo_altitude: Option<f64>,
    squawk: Option<String>,
}

impl<'de> Deserialize<'de> for StateVector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let arr: Vec<Value> = Vec::deserialize(deserializer)?;

        Ok(StateVector {
            icao24: arr
                .get(0)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            callsign: arr
                .get(1)
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            origin_country: arr
                .get(2)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            time_position: arr.get(3).and_then(|v| v.as_i64()),
            last_contact: arr.get(4).and_then(|v| v.as_i64()).unwrap_or(0),
            longitude: arr.get(5).and_then(|v| v.as_f64()),
            latitude: arr.get(6).and_then(|v| v.as_f64()),
            baro_altitude: arr.get(7).and_then(|v| v.as_f64()),
            on_ground: arr.get(8).and_then(|v| v.as_bool()).unwrap_or(false),
            velocity: arr.get(9).and_then(|v| v.as_f64()),
            true_track: arr.get(10).and_then(|v| v.as_f64()),
            vertical_rate: arr.get(11).and_then(|v| v.as_f64()),
            geo_altitude: arr.get(13).and_then(|v| v.as_f64()),
            squawk: arr
                .get(14)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("✈️  Flight Data Generator");
    println!("=========================\n");

    match args.source {
        DataSource::Opensky => {
            collect_opensky_data(&args)?;
        }
        DataSource::Synthetic => {
            generate_synthetic_data(&args)?;
        }
    }

    Ok(())
}

fn collect_opensky_data(args: &Args) -> Result<()> {
    println!("📡 Data Source: OpenSky Network (Real-time ADS-B)");
    println!("⏱️  Duration: {} seconds", args.duration);
    println!("📊 Sample interval: {} seconds", args.interval);

    // Parse bounds if provided
    let bounds = args.bounds.as_ref().map(|s| parse_bounds(s)).transpose()?;
    if let Some((min_lat, min_lon, max_lat, max_lon)) = bounds {
        println!(
            "📍 Geographic filter: [{}, {}] to [{}, {}]",
            min_lat, min_lon, max_lat, max_lon
        );
    } else {
        println!("📍 Geographic filter: Global (all aircraft)");
    }

    println!();
    println!("⚠️  Note: Anonymous rate limit is 10 requests/minute.");
    println!("   Using {} second interval to stay within limits.", args.interval);
    println!();

    let client = reqwest::blocking::Client::builder()
        .timeout(StdDuration::from_secs(30))
        .build()?;

    let mut all_features: Vec<Feature> = Vec::new();
    let mut aircraft_seen: HashMap<String, usize> = HashMap::new();
    let samples = args.duration / args.interval;
    let start_time = Utc::now();

    println!("🔄 Collecting {} samples over {} seconds...", samples, args.duration);
    println!();

    for i in 0..samples {
        let sample_start = std::time::Instant::now();

        // Build URL with optional bounds
        let url = if let Some((min_lat, min_lon, max_lat, max_lon)) = bounds {
            format!(
                "https://opensky-network.org/api/states/all?lamin={}&lomin={}&lamax={}&lomax={}",
                min_lat, min_lon, max_lat, max_lon
            )
        } else {
            "https://opensky-network.org/api/states/all".to_string()
        };

        print!("  Sample {}/{}: ", i + 1, samples);
        std::io::Write::flush(&mut std::io::stdout())?;

        match fetch_opensky_states(&client, &url) {
            Ok(response) => {
                let timestamp =
                    DateTime::from_timestamp(response.time, 0).unwrap_or(Utc::now());

                if let Some(states) = response.states {
                    let valid_states: Vec<_> = states
                        .into_iter()
                        .filter(|s| !s.on_ground && s.longitude.is_some() && s.latitude.is_some())
                        .collect();

                    println!(
                        "✅ {} aircraft in flight",
                        valid_states.len()
                    );

                    for state in valid_states {
                        *aircraft_seen.entry(state.icao24.clone()).or_insert(0) += 1;

                        let feature = state_to_feature(&state, timestamp);
                        all_features.push(feature);
                    }
                } else {
                    println!("⚠️  No aircraft data");
                }
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }

        // Wait for next sample (minus time spent fetching)
        if i < samples - 1 {
            let elapsed = sample_start.elapsed();
            let wait_time = StdDuration::from_secs(args.interval).saturating_sub(elapsed);
            if !wait_time.is_zero() {
                thread::sleep(wait_time);
            }
        }
    }

    let end_time = Utc::now();
    let collection_duration = end_time - start_time;

    println!();
    println!("📊 Collection Summary:");
    println!("  Total points: {}", all_features.len());
    println!("  Unique aircraft: {}", aircraft_seen.len());
    println!("  Duration: {} seconds", collection_duration.num_seconds());
    println!(
        "  Time range: {} to {}",
        start_time.format("%H:%M:%S"),
        end_time.format("%H:%M:%S")
    );

    println!();
    println!("💾 Writing GeoJSON...");
    common::write_geojson(all_features, &args.output)?;

    println!();
    println!("✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output flights.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 0 \\");
    println!("             --max-zoom 10 \\");
    println!("             --compression gzip");

    Ok(())
}

fn fetch_opensky_states(client: &reqwest::blocking::Client, url: &str) -> Result<OpenSkyResponse> {
    let response = client
        .get(url)
        .send()
        .context("Failed to fetch from OpenSky API")?;

    if !response.status().is_success() {
        anyhow::bail!("OpenSky API returned status: {}", response.status());
    }

    let data: OpenSkyResponse = response.json().context("Failed to parse OpenSky response")?;
    Ok(data)
}

fn state_to_feature(state: &StateVector, timestamp: DateTime<Utc>) -> Feature {
    let mut properties = Map::new();

    properties.insert("icao24".to_string(), json!(state.icao24));
    properties.insert("country".to_string(), json!(state.origin_country));

    if let Some(ref callsign) = state.callsign {
        properties.insert("callsign".to_string(), json!(callsign));
    }

    if let Some(altitude) = state.baro_altitude {
        // Convert meters to feet for display
        properties.insert("altitude".to_string(), json!((altitude * 3.28084) as i32));
    }

    if let Some(velocity) = state.velocity {
        // Convert m/s to knots
        properties.insert("speed".to_string(), json!((velocity * 1.94384) as i32));
    }

    if let Some(track) = state.true_track {
        properties.insert("heading".to_string(), json!(track as i32));
    }

    if let Some(vrate) = state.vertical_rate {
        // Convert m/s to ft/min
        properties.insert("vertical_rate".to_string(), json!((vrate * 196.85) as i32));
    }

    if let Some(ref squawk) = state.squawk {
        properties.insert("squawk".to_string(), json!(squawk));
    }

    common::create_point_feature(
        state.longitude.unwrap(),
        state.latitude.unwrap(),
        timestamp,
        properties,
    )
}

fn parse_bounds(s: &str) -> Result<(f64, f64, f64, f64)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        anyhow::bail!("Bounds must be: min_lat,min_lon,max_lat,max_lon");
    }

    Ok((
        parts[0].parse()?,
        parts[1].parse()?,
        parts[2].parse()?,
        parts[3].parse()?,
    ))
}

// ============================================================================
// Synthetic Data Generation (Fallback)
// ============================================================================

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

fn generate_synthetic_data(args: &Args) -> Result<()> {
    use rand::Rng;

    println!("⚠️  Using SYNTHETIC data for demonstration");
    println!();
    println!("📍 For real data, use: --source opensky");
    println!();

    let date = NaiveDate::parse_from_str(&args.date, "%Y-%m-%d")?;
    let start_time = common::date_to_datetime(date);

    println!("📊 Generating {} flights...", args.num_flights);

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
            (args.num_flights as f64 * flight_multiplier / intervals as f64) as usize;

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
            let lat = origin.1 + (dest.1 - origin.1) * progress;
            let lon = origin.2 + (dest.2 - origin.2) * progress;

            let mut properties = Map::new();
            properties.insert("origin".to_string(), json!(origin.0));
            properties.insert("destination".to_string(), json!(dest.0));
            properties.insert("altitude".to_string(), json!(rng.gen_range(30000..42000)));
            properties.insert("speed".to_string(), json!(rng.gen_range(450..550)));
            properties.insert("heading".to_string(), json!(rng.gen_range(0..360)));
            properties.insert("country".to_string(), json!("United States"));

            let feature = common::create_point_feature(lon, lat, timestamp, properties);
            features.push(feature);
        }

        if interval % 60 == 0 {
            println!("  Generated data for hour {}...", interval / 12);
        }
    }

    println!();
    println!("💾 Writing {} features...", features.len());
    common::write_geojson(features, &args.output)?;

    println!();
    println!("✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output flights.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 3 \\");
    println!("             --max-zoom 10 \\");
    println!("             --compression gzip");

    Ok(())
}
