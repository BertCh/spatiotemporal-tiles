//! Generate earthquake data from USGS API
//!
//! Data source: https://earthquake.usgs.gov/fdsnws/event/1/

mod common;

use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-earthquake-data")]
#[command(about = "Generate earthquake data from USGS")]
struct Args {
    /// Output file (use .csv for streaming output, .geojson for JSON)
    #[arg(short, long, default_value = "earthquakes.csv")]
    output: PathBuf,

    /// Start date (YYYY-MM-DD)
    #[arg(long, default_value = "2023-01-01")]
    start_date: String,

    /// End date (YYYY-MM-DD)
    #[arg(long, default_value = "2023-12-31")]
    end_date: String,

    /// Minimum magnitude
    #[arg(long, default_value = "4.0")]
    min_magnitude: f64,
}

#[derive(Debug, Deserialize)]
struct UsgsResponse {
    features: Vec<UsgsFeature>,
}

#[derive(Debug, Deserialize)]
struct UsgsFeature {
    geometry: UsgsGeometry,
    properties: UsgsProperties,
}

#[derive(Debug, Deserialize)]
struct UsgsGeometry {
    coordinates: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct UsgsProperties {
    mag: f64,
    place: String,
    time: i64,
    #[serde(default)]
    depth: Option<f64>,
    #[serde(rename = "type")]
    event_type: String,
    title: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🌍 Earthquake Data Generator");
    println!("============================\n");

    println!("📡 Fetching earthquake data from USGS...");
    println!("  Date range: {} to {}", args.start_date, args.end_date);
    println!("  Min magnitude: {}", args.min_magnitude);

    let use_csv = common::is_csv_output(&args.output);
    if use_csv {
        println!("📄 Using streaming CSV output (memory-efficient)");
    }

    // USGS API has a limit, so we fetch in yearly chunks
    let start_year = args.start_date[..4].parse::<i32>()?;
    let end_year = args.end_date[..4].parse::<i32>()?;

    let property_columns = vec![
        "magnitude".to_string(),
        "place".to_string(),
        "depth".to_string(),
        "type".to_string(),
        "title".to_string(),
        "value".to_string(),
    ];

    if use_csv {
        // Streaming CSV mode
        let mut csv_writer = common::StreamingCsvWriter::new(&args.output, property_columns)?;

        for year in start_year..=end_year {
            let year_start = format!("{}-01-01", year);
            let year_end = format!("{}-12-31", year);

            println!("\n📅 Fetching data for {}...", year);

            let url = format!(
                "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={}&endtime={}&minmagnitude={}",
                year_start, year_end, args.min_magnitude
            );

            let response = reqwest::blocking::get(&url)?;
            let data: UsgsResponse = response.json()?;

            println!("  ✓ Fetched {} earthquakes", data.features.len());

            for usgs_feature in data.features {
                let (lon, lat, timestamp, properties) = extract_usgs_data(usgs_feature)?;
                csv_writer.write_point(lon, lat, timestamp, &properties)?;
            }
        }

        let row_count = csv_writer.finish()?;
        println!("\n📊 Total earthquakes: {}", row_count);
    } else {
        // GeoJSON mode
        let mut all_features = Vec::new();

        for year in start_year..=end_year {
            let year_start = format!("{}-01-01", year);
            let year_end = format!("{}-12-31", year);

            println!("\n📅 Fetching data for {}...", year);

            let url = format!(
                "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={}&endtime={}&minmagnitude={}",
                year_start, year_end, args.min_magnitude
            );

            let response = reqwest::blocking::get(&url)?;
            let data: UsgsResponse = response.json()?;

            println!("  ✓ Fetched {} earthquakes", data.features.len());

            for usgs_feature in data.features {
                let (lon, lat, timestamp, properties) = extract_usgs_data(usgs_feature)?;
                let feature = common::create_point_feature(lon, lat, timestamp, properties);
                all_features.push(feature);
            }
        }

        println!("\n📊 Total earthquakes: {}", all_features.len());
        println!("\n💾 Writing output...");
        common::write_geojson(all_features, &args.output)?;
    }

    println!("\n✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output earthquakes.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 0 \\");
    println!("             --max-zoom 8 \\");
    println!("             --compression gzip");

    Ok(())
}

/// Extract data from a USGS feature for writing
fn extract_usgs_data(usgs: UsgsFeature) -> Result<(f64, f64, DateTime<Utc>, Map<String, serde_json::Value>)> {
    let lon = usgs.geometry.coordinates[0];
    let lat = usgs.geometry.coordinates[1];
    let depth = usgs.geometry.coordinates.get(2).copied().unwrap_or(0.0);

    // Convert Unix milliseconds to DateTime
    let timestamp =
        DateTime::from_timestamp_millis(usgs.properties.time).unwrap_or_else(|| Utc::now());

    let mut properties = Map::new();
    properties.insert("magnitude".to_string(), json!(usgs.properties.mag));
    properties.insert("place".to_string(), json!(usgs.properties.place));
    properties.insert("depth".to_string(), json!(depth));
    properties.insert("type".to_string(), json!(usgs.properties.event_type));
    properties.insert("title".to_string(), json!(usgs.properties.title));
    properties.insert("value".to_string(), json!(usgs.properties.mag)); // For visualization

    Ok((lon, lat, timestamp, properties))
}
