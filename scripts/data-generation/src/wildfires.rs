//! Generate wildfire perimeter data
//!
//! Uses synthetic data based on California wildfire patterns

mod common;

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, Utc};
use clap::Parser;
use geojson::Feature;
use rand::Rng;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-wildfire-data")]
#[command(about = "Generate wildfire perimeter data")]
struct Args {
    /// Output GeoJSON file
    #[arg(short, long, default_value = "wildfires.geojson")]
    output: PathBuf,

    /// Start year
    #[arg(long, default_value = "2020")]
    start_year: u32,

    /// End year
    #[arg(long, default_value = "2023")]
    end_year: u32,

    /// Number of fires
    #[arg(long, default_value = "50")]
    num_fires: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🔥 Wildfire Data Generator");
    println!("============================\n");
    println!("⚠️  Note: Using synthetic data based on CA fire patterns");
    println!();

    println!("📊 Generating {} fires from {} to {}...", 
             args.num_fires, args.start_year, args.end_year);

    let features = generate_wildfire_data(args.start_year, args.end_year, args.num_fires)?;

    println!("\n💾 Writing output...");
    common::write_geojson(features, &args.output)?;

    println!("\n✅ Success! Now run:");
    println!("   stt-build --input {} --output wildfires.stt \\", args.output.display());
    println!("             --time-field timestamp \\");
    println!("             --temporal-bucket day \\");
    println!("             --min-zoom 5 \\");
    println!("             --max-zoom 12 \\");
    println!("             --compression gzip");
    println!("\n💡 Temporal bucketing: day (wildfire progression is typically tracked daily)");

    Ok(())
}

fn generate_wildfire_data(start_year: u32, end_year: u32, num_fires: usize) -> Result<Vec<Feature>> {
    let mut rng = rand::thread_rng();
    let mut features = Vec::new();

    // California fire-prone areas
    let fire_zones = vec![
        (-120.5, 37.5, "Central Valley"),
        (-122.0, 38.5, "North Bay"),
        (-119.0, 36.5, "Sierra Nevada"),
        (-117.5, 34.0, "Southern California"),
    ];

    for fire_id in 0..num_fires {
        // Random start date (fire season: May-October)
        let year = rng.gen_range(start_year..=end_year);
        let month = rng.gen_range(5..=10);
        let day = rng.gen_range(1..28);
        
        let start_date = NaiveDate::from_ymd_opt(year as i32, month, day)
            .ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        let start_time = common::date_to_datetime(start_date);

        // Fire duration (3-30 days)
        let duration_days = rng.gen_range(3..30);

        // Pick a zone
        let zone = &fire_zones[rng.gen_range(0..fire_zones.len())];
        let base_lon = zone.0 + rng.gen_range(-0.5..0.5);
        let base_lat = zone.1 + rng.gen_range(-0.5..0.5);

        // Initial fire size
        let mut radius = rng.gen_range(0.01..0.05);

        // Generate daily perimeter updates
        for day in 0..duration_days {
            let timestamp = start_time + Duration::days(day);

            // Fire grows then gets contained
            if day < duration_days / 2 {
                radius *= rng.gen_range(1.1..1.3); // Growth
            } else {
                radius *= rng.gen_range(0.95..1.0); // Containment
            }

            let severity = if day < duration_days / 3 {
                "extreme"
            } else if day < duration_days * 2 / 3 {
                "high"
            } else {
                "moderate"
            };

            let mut properties = Map::new();
            properties.insert("fire_id".to_string(), json!(fire_id));
            properties.insert("name".to_string(), json!(format!("{} Fire", zone.2)));
            properties.insert("day".to_string(), json!(day));
            properties.insert("acres".to_string(), json!((radius * 100000.0) as u32));
            properties.insert("severity".to_string(), json!(severity));
            properties.insert("containment".to_string(), json!(if day > duration_days / 2 { 
                (day - duration_days / 2) * 100 / (duration_days / 2)
            } else { 
                0 
            }));

            // Create point (in production, would be polygon perimeter)
            let feature = common::create_point_feature(base_lon, base_lat, timestamp, properties);
            features.push(feature);
        }

        if (fire_id + 1) % 10 == 0 {
            println!("  Generated fire {}/{}...", fire_id + 1, num_fires);
        }
    }

    Ok(features)
}

