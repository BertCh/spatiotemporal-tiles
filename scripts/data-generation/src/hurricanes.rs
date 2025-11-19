//! Generate hurricane track data from NOAA IBTrACS
//!
//! Data source: https://www.ncei.noaa.gov/products/international-best-track-archive

mod common;

use anyhow::Result;
use chrono::{DateTime, NaiveDateTime, Utc, Datelike};
use clap::Parser;
use geojson::Feature;
use serde::Deserialize;
use serde_json::{json, Map};
use std::fs::File;
use std::io::BufRead;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-hurricane-data")]
#[command(about = "Generate hurricane track data from NOAA")]
struct Args {
    /// Output GeoJSON file
    #[arg(short, long, default_value = "hurricanes.geojson")]
    output: PathBuf,

    /// Start year
    #[arg(long, default_value = "2020")]
    start_year: u32,

    /// End year
    #[arg(long, default_value = "2024")]
    end_year: u32,

    /// Use cached data
    #[arg(long)]
    cached: bool,

    /// Create a synthetic year from multiple years of data
    #[arg(long)]
    synthetic: bool,

    /// Target year for synthetic data (default: start_year)
    #[arg(long)]
    synthetic_year: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct IbtracsRecord {
    #[serde(rename = "SID")]
    storm_id: String,
    #[serde(rename = "NAME")]
    name: String,
    #[serde(rename = "ISO_TIME")]
    iso_time: String,
    #[serde(rename = "LAT")]
    lat: f64,
    #[serde(rename = "LON")]
    lon: f64,
    #[serde(rename = "USA_WIND")]
    wind_speed: String,
    #[serde(rename = "USA_SSHS")]
    category: String,
    #[serde(rename = "BASIN")]
    basin: String,
    #[serde(rename = "NATURE")]
    nature: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🌀 Hurricane Data Generator");
    println!("============================\n");

    // Download IBTrACS data
    let ibtracs_csv = PathBuf::from("data/ibtracs.csv");
    if !args.cached {
        std::fs::create_dir_all("data")?;
        println!("📥 Downloading NOAA IBTrACS data (this may take a few minutes)...");
        common::download_file(
            "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NA.list.v04r00.csv",
            &ibtracs_csv,
        )?;
    }

    println!("\n📊 Processing hurricane tracks...");
    let features = process_hurricane_data(&ibtracs_csv, &args)?;
    println!("✓ Generated {} track segments", features.len());

    // Write GeoJSON
    println!("\n💾 Writing output...");
    common::write_geojson(features, &args.output)?;

    println!("\n✅ Success! Now run:");
    println!("   stt-build --input {} --output hurricanes.stt \\", args.output.display());
    println!("             --time-field timestamp \\");
    println!("             --temporal-bucket hour \\");
    println!("             --min-zoom 0 \\");
    println!("             --max-zoom 10 \\");
    println!("             --compression gzip");
    println!("\n💡 Temporal bucketing: hour (hurricane positions reported every 6 hours)");

    Ok(())
}

fn process_hurricane_data(path: &PathBuf, args: &Args) -> Result<Vec<Feature>> {
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut rdr = csv::ReaderBuilder::new().from_reader(reader);

    // Group records by storm ID
    let mut storm_records: std::collections::HashMap<String, Vec<IbtracsRecord>> = std::collections::HashMap::new();

    for result in rdr.deserialize() {
        let record: IbtracsRecord = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Filter by basin (Atlantic)
        // Note: If we want global cyclones, we'd need to download the ALL dataset
        if record.basin != "NA" {
            continue;
        }

        storm_records.entry(record.storm_id.clone())
            .or_insert_with(Vec::new)
            .push(record);
    }

    let mut features = Vec::new();
    let mut processed_storms = 0;

    // Determine target year for synthetic data
    let target_year = args.synthetic_year.unwrap_or(args.start_year as i32);

    for (_, mut records) in storm_records {
        // Sort by time
        records.sort_by(|a, b| a.iso_time.cmp(&b.iso_time));

        // Skip if no records
        if records.is_empty() {
            continue;
        }

        // Check if storm is within requested years
        // Use the first record's year
        let first_record_dt = match NaiveDateTime::parse_from_str(&records[0].iso_time, "%Y-%m-%d %H:%M:%S") {
            Ok(dt) => dt,
            Err(_) => continue,
        };

        let start_year = first_record_dt.year();
        
        // Filter by source years
        if (start_year as u32) < args.start_year || (start_year as u32) > args.end_year {
            continue;
        }

        // Calculate year offset for synthetic mode
        let year_offset = if args.synthetic {
            target_year - start_year
        } else {
            0
        };

        // Create line segments
        for i in 0..records.len().saturating_sub(1) {
            let start_record = &records[i];
            let end_record = &records[i+1];

            // Parse timestamps
            let mut start_dt = match NaiveDateTime::parse_from_str(&start_record.iso_time, "%Y-%m-%d %H:%M:%S") {
                Ok(dt) => dt,
                Err(_) => continue,
            };

            // Apply synthetic year offset
            if args.synthetic {
                start_dt = start_dt.with_year(start_dt.year() + year_offset).unwrap_or(start_dt);
            }
            
            let start_time = DateTime::from_naive_utc_and_offset(start_dt, Utc);
            
            // Parse wind speed and category
            let wind_speed: f64 = start_record.wind_speed.trim().parse().unwrap_or(0.0);
            let category = start_record.category.trim();
            let category_num = if category.is_empty() || category == " " {
                0
            } else {
                category.parse().unwrap_or(0)
            };
            
            let nature = start_record.nature.as_deref().unwrap_or("").trim();

            let mut properties = Map::new();
            properties.insert("storm_id".to_string(), json!(start_record.storm_id));
            properties.insert("name".to_string(), json!(start_record.name));
            properties.insert("wind_speed".to_string(), json!(wind_speed));
            properties.insert("category".to_string(), json!(category_num));
            properties.insert("status".to_string(), json!(get_status(category_num, nature)));
            properties.insert("nature".to_string(), json!(nature));
            properties.insert("value".to_string(), json!(wind_speed));

            // Create LineString segment
            let feature = common::create_linestring_feature(
                vec![
                    [start_record.lon, start_record.lat],
                    [end_record.lon, end_record.lat]
                ],
                start_time,
                properties
            );
            
            features.push(feature);
        }

        processed_storms += 1;
        if processed_storms % 100 == 0 {
             println!("  Processed {} storms...", processed_storms);
        }
    }

    Ok(features)
}

fn get_status(category: i32, nature: &str) -> String {
    // First check nature for specific types
    match nature {
        "SS" => return "subtropical_storm".to_string(),
        "ET" => return "extratropical".to_string(),
        "DS" => return "disturbance".to_string(),
        _ => {}
    }

    // Fallback to category based status
    match category {
        -1 => "tropical_depression".to_string(),
        0 => "tropical_storm".to_string(),
        1 => "category_1".to_string(),
        2 => "category_2".to_string(),
        3 => "category_3".to_string(),
        4 => "category_4".to_string(),
        5 => "category_5".to_string(),
        _ => "unknown".to_string(),
    }
}

