//! Generate earthquake data from USGS API
//!
//! Data source: https://earthquake.usgs.gov/fdsnws/event/1/
//!
//! Supports CSV, GeoJSON, and GeoParquet output formats.
//! GeoParquet is recommended for large datasets.

mod common;

use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::Parser;
use common::{PropertyColumn, StreamingGeoParquetWriter};
use serde::Deserialize;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-earthquake-data")]
#[command(about = "Generate earthquake data from USGS")]
struct Args {
    /// Output file format determined by extension:
    ///   .parquet / .geoparquet - GeoParquet (recommended)
    ///   .csv - Streaming CSV
    ///   .geojson / .json - GeoJSON
    #[arg(short, long, default_value = "earthquakes.parquet")]
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

    // Detect output format
    let use_geoparquet = common::is_geoparquet_output(&args.output);
    let use_csv = common::is_csv_output(&args.output);

    if use_geoparquet {
        println!("📦 Using GeoParquet output (columnar, efficient)");
    } else if use_csv {
        println!("📄 Using streaming CSV output (memory-efficient)");
    } else {
        println!("📄 Using GeoJSON output");
    }

    // USGS API has a limit, so we fetch in yearly chunks
    let start_year = args.start_date[..4].parse::<i32>()?;
    let end_year = args.end_date[..4].parse::<i32>()?;

    // Property columns for CSV output
    let csv_property_columns = vec![
        "magnitude".to_string(),
        "place".to_string(),
        "depth".to_string(),
        "type".to_string(),
        "title".to_string(),
        "value".to_string(),
    ];

    // Property columns for GeoParquet output (typed)
    let geoparquet_property_columns = vec![
        PropertyColumn::float64("magnitude"),
        PropertyColumn::string("place"),
        PropertyColumn::float64("depth"),
        PropertyColumn::string("type"),
        PropertyColumn::string("title"),
        PropertyColumn::float64("value"),
    ];

    if use_geoparquet {
        // GeoParquet mode
        let mut writer = StreamingGeoParquetWriter::new(&args.output, geoparquet_property_columns)?;

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
                writer.write_point(lon, lat, timestamp, &properties)?;
            }
        }

        let row_count = writer.finish()?;
        println!("\n📊 Total earthquakes: {}", row_count);
    } else if use_csv {
        // Streaming CSV mode
        let mut csv_writer = common::StreamingCsvWriter::new(&args.output, csv_property_columns)?;

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
    if use_geoparquet {
        println!(
            "   stt-build --input {} --output earthquakes.stt \\",
            args.output.display()
        );
        println!("             --time-field timestamp \\");
        println!("             --min-zoom 0 --max-zoom 8 \\");
        println!("             --compression gzip --features geoparquet");
    } else {
        println!(
            "   stt-build --input {} --output earthquakes.stt \\",
            args.output.display()
        );
        println!("             --time-field timestamp \\");
        println!("             --min-zoom 0 --max-zoom 8 \\");
        println!("             --compression gzip");
    }

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
