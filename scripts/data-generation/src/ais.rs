//! Process real AIS data from NOAA Marine Cadastre
//!
//! Downloads and converts USCG Marine Cadastre AIS data to CSV, GeoJSON, or GeoParquet.
//! Source: https://coast.noaa.gov/htdata/CMSP/AISDataHandler
//!
//! GeoParquet is recommended for large datasets - it's 10-50x smaller than GeoJSON
//! and loads much faster in tools like DuckDB, QGIS, and Python/geopandas.

mod common;

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use clap::Parser;
use common::{PropertyColumn, StreamingGeoParquetWriter};
use csv::ReaderBuilder;
use serde::Deserialize;
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-ais-data")]
#[command(about = "Process real AIS data from NOAA Marine Cadastre")]
struct Args {
    /// Input CSV file (unzipped AIS data)
    #[arg(short, long)]
    input: PathBuf,

    /// Output file format determined by extension:
    ///   .parquet / .geoparquet - GeoParquet (recommended, smallest & fastest)
    ///   .csv - Streaming CSV (memory-efficient)
    ///   .geojson / .json - GeoJSON (readable but large)
    #[arg(short, long, default_value = "ais-traffic.parquet")]
    output: PathBuf,

    /// Sampling interval in minutes (default: 10 = keep 1 position per vessel per 10 min)
    #[arg(long, default_value = "10")]
    sample_minutes: i64,

    /// Maximum number of vessels to include (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_vessels: usize,

    /// Geographic bounds: min_lat,min_lon,max_lat,max_lon
    /// Example: 25.0,-80.0,45.0,-65.0 (US East Coast)
    #[arg(long)]
    bounds: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AisRecord {
    #[serde(rename = "MMSI")]
    mmsi: String,

    #[serde(rename = "BaseDateTime")]
    base_date_time: String,

    #[serde(rename = "LAT")]
    lat: f64,

    #[serde(rename = "LON")]
    lon: f64,

    #[serde(rename = "SOG")]
    sog: Option<f64>, // Speed Over Ground (knots)

    #[serde(rename = "COG")]
    cog: Option<f64>, // Course Over Ground (degrees)

    #[serde(rename = "Heading")]
    heading: Option<f64>,

    #[serde(rename = "VesselName")]
    vessel_name: Option<String>,

    #[serde(rename = "VesselType")]
    vessel_type: Option<u8>,

    #[serde(rename = "Length")]
    length: Option<f64>,

    #[serde(rename = "Width")]
    width: Option<f64>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🚢 AIS Data Processor (NOAA Marine Cadastre)");
    println!("============================================\n");
    println!("📂 Input: {}", args.input.display());
    println!(
        "📊 Sampling: 1 position per vessel per {} minutes",
        args.sample_minutes
    );

    if let Some(ref bounds_str) = args.bounds {
        println!("📍 Geographic filter: {}", bounds_str);
    }

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

    println!();

    // Parse bounds if provided
    let bounds = args.bounds.as_ref().map(|s| parse_bounds(s)).transpose()?;

    // Read and process CSV
    println!("🔄 Reading CSV data...");
    let file = File::open(&args.input)
        .with_context(|| format!("Failed to open input file: {}", args.input.display()))?;

    let reader = BufReader::new(file);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    let mut vessel_last_time: HashMap<String, i64> = HashMap::new();
    let mut total_records = 0;
    let mut filtered_records = 0;
    let mut unique_vessels = std::collections::HashSet::new();

    // Property columns for CSV output (string names)
    let csv_property_columns = vec![
        "mmsi".to_string(),
        "vessel_type".to_string(),
        "speed".to_string(),
        "course".to_string(),
        "heading".to_string(),
        "vessel_name".to_string(),
        "length".to_string(),
        "width".to_string(),
    ];
    
    // Property columns for GeoParquet output (typed)
    let geoparquet_property_columns = vec![
        PropertyColumn::string("mmsi"),
        PropertyColumn::string("vessel_type"),
        PropertyColumn::float64("speed"),
        PropertyColumn::float64("course"),
        PropertyColumn::float64("heading"),
        PropertyColumn::string("vessel_name"),
        PropertyColumn::float64("length"),
        PropertyColumn::float64("width"),
    ];

    // Choose output mode
    let mut geoparquet_writer = if use_geoparquet {
        Some(StreamingGeoParquetWriter::new(&args.output, geoparquet_property_columns)?)
    } else {
        None
    };
    let mut csv_writer = if use_csv {
        Some(common::StreamingCsvWriter::new(&args.output, csv_property_columns)?)
    } else {
        None
    };
    let mut features = Vec::new();

    for result in csv_reader.deserialize() {
        total_records += 1;

        if total_records % 100000 == 0 {
            let count = if use_geoparquet {
                geoparquet_writer.as_ref().map(|w| w.row_count()).unwrap_or(0)
            } else if use_csv {
                csv_writer.as_ref().map(|w| w.row_count()).unwrap_or(0)
            } else {
                features.len()
            };
            println!(
                "  Processed {} records, kept {} features, {} unique vessels...",
                total_records,
                count,
                unique_vessels.len()
            );
        }

        let record: AisRecord = match result {
            Ok(r) => r,
            Err(e) => {
                if total_records < 10 {
                    eprintln!("  Warning: Failed to parse record {}: {}", total_records, e);
                }
                continue;
            }
        };

        // Filter invalid positions
        if record.lat == 0.0 && record.lon == 0.0 {
            continue;
        }
        if record.lat > 90.0 || record.lat < -90.0 {
            continue;
        }
        if record.lon > 180.0 || record.lon < -180.0 {
            continue;
        }
        if record.lat == 91.0 {
            // NOAA uses 91 for invalid
            continue;
        }

        // Apply geographic bounds filter
        if let Some((min_lat, min_lon, max_lat, max_lon)) = bounds {
            if record.lat < min_lat
                || record.lat > max_lat
                || record.lon < min_lon
                || record.lon > max_lon
            {
                filtered_records += 1;
                continue;
            }
        }

        // Apply max vessels limit
        if args.max_vessels > 0 && unique_vessels.len() >= args.max_vessels {
            if !unique_vessels.contains(&record.mmsi) {
                continue;
            }
        }

        // Parse timestamp
        let timestamp =
            match NaiveDateTime::parse_from_str(&record.base_date_time, "%Y-%m-%dT%H:%M:%S") {
                Ok(dt) => dt.and_utc(),
                Err(_) => continue,
            };
        let timestamp_ms = timestamp.timestamp_millis();

        // Apply temporal sampling
        let sample_interval_ms = args.sample_minutes * 60 * 1000;
        if let Some(&last_time) = vessel_last_time.get(&record.mmsi) {
            if timestamp_ms - last_time < sample_interval_ms {
                continue;
            }
        }
        vessel_last_time.insert(record.mmsi.clone(), timestamp_ms);
        unique_vessels.insert(record.mmsi.clone());

        // Convert vessel type code to category
        let vessel_category = vessel_type_to_category(record.vessel_type);

        // Build properties
        let mut properties = Map::new();
        properties.insert("mmsi".to_string(), json!(record.mmsi));
        properties.insert("vessel_type".to_string(), json!(vessel_category));

        if let Some(sog) = record.sog {
            if sog >= 0.0 && sog < 100.0 {
                properties.insert("speed".to_string(), json!(sog));
            }
        }

        if let Some(cog) = record.cog {
            if cog >= 0.0 && cog <= 360.0 {
                properties.insert("course".to_string(), json!(cog));
            }
        }

        if let Some(heading) = record.heading {
            if heading >= 0.0 && heading <= 360.0 {
                properties.insert("heading".to_string(), json!(heading));
            }
        }

        if let Some(ref name) = record.vessel_name {
            if !name.is_empty() && name != "Unknown" {
                properties.insert("vessel_name".to_string(), json!(name.clone()));
            }
        }

        if let Some(length) = record.length {
            if length > 0.0 && length < 500.0 {
                properties.insert("length".to_string(), json!(length));
            }
        }

        if let Some(width) = record.width {
            if width > 0.0 && width < 100.0 {
                properties.insert("width".to_string(), json!(width));
            }
        }

        if use_geoparquet {
            geoparquet_writer.as_mut().unwrap().write_point(
                record.lon,
                record.lat,
                timestamp,
                &properties,
            )?;
        } else if use_csv {
            csv_writer.as_mut().unwrap().write_point(
                record.lon,
                record.lat,
                timestamp,
                &properties,
            )?;
        } else {
            use geojson::{Feature, Geometry, Value as GeoValue};
            // Add timestamp to properties for GeoJSON output
            let mut props = properties.clone();
            props.insert("timestamp".to_string(), json!(timestamp.to_rfc3339()));
            let feature = Feature {
                geometry: Some(Geometry::new(GeoValue::Point(vec![record.lon, record.lat]))),
                properties: Some(props),
                ..Default::default()
            };
            features.push(feature);
        }
    }

    let final_count = if use_geoparquet {
        let writer = geoparquet_writer.take().unwrap();
        writer.finish()?
    } else if use_csv {
        let writer = csv_writer.take().unwrap();
        writer.finish()?
    } else {
        features.len()
    };

    println!("\n📊 Processing Summary:");
    println!("  Total records read: {}", total_records);
    println!("  Geographic filter: {} records removed", filtered_records);
    println!("  Final features: {}", final_count);
    println!("  Unique vessels: {}", unique_vessels.len());

    // Write GeoJSON if using that format
    if !use_geoparquet && !use_csv {
        println!("\n💾 Writing GeoJSON...");
        common::write_geojson(features, &args.output)?;
    }

    println!("\n✅ Success! Now run:");
    if use_geoparquet {
        println!(
            "   stt-build --input {} --output ais-traffic.stt \\",
            args.output.display()
        );
        println!("             --time-field timestamp \\");
        println!("             --min-zoom 0 --max-zoom 14 \\");
        println!("             --compression gzip --features geoparquet");
    } else {
        println!(
            "   stt-build --input {} --output ais-traffic.stt \\",
            args.output.display()
        );
        println!("             --time-field timestamp \\");
        println!("             --min-zoom 0 --max-zoom 14 \\");
        println!("             --compression gzip");
    }

    Ok(())
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

fn vessel_type_to_category(type_code: Option<u8>) -> &'static str {
    match type_code {
        Some(30) => "fishing",
        Some(60..=69) => "passenger",
        Some(70..=79) => "cargo",
        Some(80..=89) => "tanker",
        Some(31..=32) => "towing",
        Some(50..=59) => "special",
        _ => "other",
    }
}
