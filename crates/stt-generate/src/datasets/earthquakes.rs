//! Generate earthquake data from USGS API
//!
//! Data source: https://earthquake.usgs.gov/fdsnws/event/1/

use crate::common::{self, PointRecord, StreamingParquetWriter};
use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Generate earthquake data from USGS")]
pub struct Args {
    /// Output file (.stt, .geojson, or .csv)
    #[arg(short, long, default_value = "earthquakes.stt")]
    pub output: PathBuf,

    /// Start date (YYYY-MM-DD)
    #[arg(long, default_value = "2020-01-01")]
    pub start_date: String,

    /// End date (YYYY-MM-DD)
    #[arg(long, default_value = "2024-12-31")]
    pub end_date: String,

    /// Minimum magnitude
    #[arg(long, default_value = "4.0")]
    pub min_magnitude: f64,

    /// Skip stt-build step (just output GeoJSON/CSV)
    #[arg(long)]
    pub skip_build: bool,
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

pub fn run(args: Args) -> Result<()> {
    println!("🌍 Earthquake Data Generator");
    println!("============================\n");

    println!("📡 Fetching earthquake data from USGS...");
    println!("   Date range: {} to {}", args.start_date, args.end_date);
    println!("   Min magnitude: {}", args.min_magnitude);

    // Determine intermediate output format (prefer Parquet for efficiency)
    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    let use_parquet = common::is_parquet_output(&intermediate_path);
    let use_csv = common::is_csv_output(&intermediate_path);
    
    if use_parquet {
        println!("📄 Using streaming GeoParquet output (efficient)");
    } else if use_csv {
        println!("📄 Using streaming CSV output");
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

    let mut all_features = Vec::new();
    let mut parquet_writer = if use_parquet {
        Some(StreamingParquetWriter::new(&intermediate_path, property_columns.clone())?)
    } else {
        None
    };
    let mut csv_writer = if use_csv {
        Some(common::StreamingCsvWriter::new(&intermediate_path, property_columns)?)
    } else {
        None
    };

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

        println!("   ✓ Fetched {} earthquakes", data.features.len());

        for usgs_feature in data.features {
            let (lon, lat, timestamp, properties) = extract_usgs_data(usgs_feature)?;

            if let Some(ref mut writer) = parquet_writer {
                let record = PointRecord::new(lon, lat, timestamp, properties);
                writer.write_point(&record)?;
            } else if let Some(ref mut writer) = csv_writer {
                writer.write_point(lon, lat, timestamp, &properties)?;
            } else {
                let feature = common::create_point_feature(lon, lat, timestamp, properties);
                all_features.push(feature);
            }
        }
    }

    let total_count = if let Some(writer) = parquet_writer.take() {
        writer.finish()?
    } else if let Some(writer) = csv_writer.take() {
        writer.finish()?
    } else {
        let count = all_features.len();
        println!("\n💾 Writing GeoJSON...");
        common::write_geojson(all_features, &intermediate_path)?;
        count
    };

    println!("\n📊 Total earthquakes: {}", total_count);

    // Build STT if output is .stt
    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        common::run_stt_build(
            &intermediate_path,
            &args.output,
            "timestamp",
            0,
            10,
            "gzip",
        )?;

        // Clean up intermediate file
        let _ = std::fs::remove_file(&intermediate_path);
    }

    println!("\n✅ Earthquake data generation complete!");

    Ok(())
}

fn extract_usgs_data(usgs: UsgsFeature) -> Result<(f64, f64, DateTime<Utc>, Map<String, serde_json::Value>)> {
    let lon = usgs.geometry.coordinates[0];
    let lat = usgs.geometry.coordinates[1];
    let depth = usgs.geometry.coordinates.get(2).copied().unwrap_or(0.0);

    let timestamp = DateTime::from_timestamp_millis(usgs.properties.time)
        .unwrap_or_else(|| Utc::now());

    let mut properties = Map::new();
    properties.insert("magnitude".to_string(), json!(usgs.properties.mag));
    properties.insert("place".to_string(), json!(usgs.properties.place));
    properties.insert("depth".to_string(), json!(depth));
    properties.insert("type".to_string(), json!(usgs.properties.event_type));
    properties.insert("title".to_string(), json!(usgs.properties.title));
    properties.insert("value".to_string(), json!(usgs.properties.mag));

    Ok((lon, lat, timestamp, properties))
}


