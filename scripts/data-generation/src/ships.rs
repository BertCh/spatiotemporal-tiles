//! Generate maritime traffic data
//!
//! ## Real AIS Data Source
//! 
//! **Provider:** NOAA Marine Cadastre (Free, public access)
//! **Base URL:** https://coast.noaa.gov/htdata/CMSP/AISDataHandler
//! 
//! ### Coverage
//! - **Geographic:** US coastal waters
//! - **Temporal:** 2009 - Present (daily files)
//! - **File Format:** ZIP compressed CSV files
//! - **Naming:** `AIS_YYYY_MM_DD.zip`
//! - **Size:** ~200-300 MB compressed per day (~800 MB uncompressed)
//! 
//! ### CSV Fields
//! - `MMSI` - Maritime Mobile Service Identity (vessel ID)
//! - `BaseDateTime` - UTC timestamp
//! - `LAT`, `LON` - Position (decimal degrees)
//! - `SOG` - Speed Over Ground (knots)
//! - `COG` - Course Over Ground (degrees)
//! - `Heading` - True heading (degrees)
//! - `VesselName`, `VesselType` - Vessel information
//! - `Length`, `Width`, `Draft` - Vessel dimensions (meters)
//! 
//! ### Download Example
//! ```bash
//! # Download a single day
//! curl -o AIS_2023_01_01.zip \
//!   https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip
//! 
//! # Unzip and process
//! unzip AIS_2023_01_01.zip
//! 
//! # Convert to GeoJSON (custom script needed)
//! # Then use stt-build to create STT archive
//! ```
//! 
//! ### References
//! - NOAA AIS Portal: https://marinecadastre.gov/ais/
//! - Data Dictionary: https://coast.noaa.gov/data/marinecadastre/ais/AIS-FAQ.pdf
//! 
//! ---
//! 
//! This script generates synthetic ship movements for demonstration purposes.
//! For production use, process real AIS data from NOAA.

mod common;

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, Utc};
use clap::Parser;
use geojson::Feature;
use rand::Rng;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-ship-data")]
#[command(about = "Generate synthetic maritime traffic data")]
#[command(long_about = "
Generate synthetic maritime traffic data for demonstration.

REAL DATA SOURCE:
  Provider: NOAA Marine Cadastre (Free, public access)
  URL:      https://coast.noaa.gov/htdata/CMSP/AISDataHandler
  Format:   ZIP compressed CSV (AIS_YYYY_MM_DD.zip)
  Coverage: US coastal waters, 2009-Present
  Size:     ~200-300 MB compressed per day

DOWNLOAD EXAMPLE:
  curl -o AIS_2023_01_01.zip \\
    https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip

For production use, download and process real AIS data from NOAA.
")]
struct Args {
    /// Output GeoJSON file
    #[arg(short, long, default_value = "ship-traffic.geojson")]
    output: PathBuf,

    /// Start date (YYYY-MM-DD)
    #[arg(long, default_value = "2024-01-01")]
    start_date: String,

    /// Number of days to simulate
    #[arg(long, default_value = "7")]
    days: i64,

    /// Number of ships to simulate
    #[arg(long, default_value = "500")]
    num_ships: usize,
}

#[derive(Clone, Copy)]
enum VesselType {
    Cargo,
    Tanker,
    Passenger,
    Fishing,
}

impl VesselType {
    fn to_string(&self) -> &str {
        match self {
            VesselType::Cargo => "cargo",
            VesselType::Tanker => "tanker",
            VesselType::Passenger => "passenger",
            VesselType::Fishing => "fishing",
        }
    }

    fn typical_speed(&self) -> f64 {
        match self {
            VesselType::Cargo => 15.0,
            VesselType::Tanker => 12.0,
            VesselType::Passenger => 20.0,
            VesselType::Fishing => 8.0,
        }
    }
}

// Major shipping routes (simplified)
const ROUTES: &[(f64, f64, f64, f64, &str)] = &[
    (-122.0, 37.0, -75.0, 40.0, "Pacific-Atlantic"),
    (0.0, 51.0, -74.0, 40.0, "Europe-NY"),
    (103.0, 1.0, -118.0, 34.0, "Asia-LA"),
    (-5.0, 36.0, -80.0, 25.0, "Med-Caribbean"),
];

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🚢 Maritime Traffic Generator");
    println!("============================\n");
    println!("⚠️  Note: Using SYNTHETIC data for demonstration");
    println!();
    println!("📍 Real AIS Data Available From:");
    println!("   Provider: NOAA Marine Cadastre (Free, public)");
    println!("   URL: https://coast.noaa.gov/htdata/CMSP/AISDataHandler");
    println!("   Coverage: US coastal waters, 2009-Present");
    println!("   Format: ZIP compressed CSV (~200-300 MB/day)");
    println!();
    println!("   Example download:");
    println!("   curl -o AIS_2023_01_01.zip \\");
    println!("     https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip");
    println!();

    let date = NaiveDate::parse_from_str(&args.start_date, "%Y-%m-%d")?;
    let start_time = common::date_to_datetime(date);

    println!("📊 Generating {} ships over {} days...", args.num_ships, args.days);
    let features = generate_ship_data(start_time, args.days, args.num_ships)?;

    println!("\n💾 Writing output...");
    common::write_geojson(features, &args.output)?;

    println!("\n✅ Success! Now run:");
    println!("   stt-build --input {} --output ship-traffic.stt \\", args.output.display());
    println!("             --time-field timestamp \\");
    println!("             --temporal-resolution high-frequency \\");
    println!("             --min-zoom 0 \\");
    println!("             --max-zoom 12 \\");
    println!("             --compression gzip");
    println!("\n💡 Temporal resolution: high-frequency profile");
    println!("   Zoom 0-3: daily → Zoom 4-6: hourly → Zoom 7-9: minute → Zoom 10+: second");

    Ok(())
}

fn generate_ship_data(start_time: DateTime<Utc>, days: i64, num_ships: usize) -> Result<Vec<Feature>> {
    let mut rng = rand::thread_rng();
    let mut features = Vec::new();

    let mut ships: Vec<(usize, VesselType, f64, f64, f64, f64)> = Vec::new();

    // Initialize ships on routes
    for ship_id in 0..num_ships {
        let vessel_type = match rng.gen_range(0..4) {
            0 => VesselType::Cargo,
            1 => VesselType::Tanker,
            2 => VesselType::Passenger,
            _ => VesselType::Fishing,
        };

        let route = ROUTES[rng.gen_range(0..ROUTES.len())];
        let progress = rng.gen_range(0.0..1.0);
        
        let lon = route.0 + (route.2 - route.0) * progress;
        let lat = route.1 + (route.3 - route.1) * progress;
        
        ships.push((ship_id, vessel_type, lon, lat, route.2, route.3));
    }

    // Simulate movement
    let hours = days * 24;
    for hour in 0..hours {
        let timestamp = start_time + Duration::hours(hour);

        for ship in &mut ships {
            let (id, vessel_type, lon, lat, dest_lon, dest_lat) = *ship;

            // Calculate movement
            let speed = vessel_type.typical_speed();
            let distance_deg = (speed * 1.852) / 111.0; // knots to degrees

            let dx = dest_lon - lon;
            let dy = dest_lat - lat;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist > 0.1 {
                ship.2 += (dx / dist) * distance_deg;
                ship.3 += (dy / dist) * distance_deg;
            }

            let mut properties = Map::new();
            properties.insert("ship_id".to_string(), json!(id));
            properties.insert("vessel_type".to_string(), json!(vessel_type.to_string()));
            properties.insert("speed".to_string(), json!(speed));
            properties.insert("heading".to_string(), json!((dx.atan2(dy) * 180.0 / std::f64::consts::PI) as i32));

            let feature = common::create_point_feature(ship.2, ship.3, timestamp, properties);
            features.push(feature);
        }

        if hour % 24 == 0 {
            println!("  Generated day {}/{}...", hour / 24 + 1, days);
        }
    }

    Ok(features)
}

