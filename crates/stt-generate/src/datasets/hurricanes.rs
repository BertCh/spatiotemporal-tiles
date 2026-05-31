//! Generate hurricane track data from NOAA IBTrACS
//!
//! Data source: https://www.ncei.noaa.gov/products/international-best-track-archive

use crate::common::{self, LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter};
use anyhow::Result;
use chrono::{DateTime, Datelike, NaiveDateTime, Utc};
use clap::Parser;
use geojson::Feature;
use serde::Deserialize;
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Generate hurricane track data from NOAA IBTrACS")]
pub struct Args {
    /// Output file (.stt, .geojson, or .csv)
    #[arg(short, long, default_value = "hurricanes.stt")]
    pub output: PathBuf,

    /// Start year
    #[arg(long, default_value = "2020")]
    pub start_year: u32,

    /// End year
    #[arg(long, default_value = "2024")]
    pub end_year: u32,

    /// Use cached data (skip download)
    #[arg(long)]
    pub cached: bool,

    /// Create a synthetic year from multiple years of data
    #[arg(long)]
    pub synthetic: bool,

    /// Target year for synthetic data
    #[arg(long)]
    pub synthetic_year: Option<i32>,

    /// Skip stt-build step
    #[arg(long)]
    pub skip_build: bool,
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

pub fn run(args: Args) -> Result<()> {
    println!("🌀 Hurricane Data Generator");
    println!("============================\n");

    // Download IBTrACS data
    let ibtracs_csv = PathBuf::from("data/ibtracs.csv");
    if !args.cached {
        std::fs::create_dir_all("data")?;
        println!("📥 Downloading NOAA IBTrACS data...");
        common::download_file(
            "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NA.list.v04r00.csv",
            &ibtracs_csv,
        )?;
    }

    // Determine intermediate output format (prefer Parquet for efficiency)
    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    let use_parquet = common::is_parquet_output(&intermediate_path);
    
    if use_parquet {
        println!("📄 Using streaming GeoParquet output (efficient)");
    }

    println!("\n📊 Processing hurricane tracks...");

    if use_parquet {
        let count = process_hurricane_data_parquet(&ibtracs_csv, &intermediate_path, &args)?;
        println!("✓ Generated {} track segments", count);
    } else {
        let features = process_hurricane_data(&ibtracs_csv, &args)?;
        println!("✓ Generated {} track segments", features.len());
        println!("\n💾 Writing output...");
        common::write_geojson(features, &intermediate_path)?;
    }

    // Build STT if output is .stt
    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        common::run_stt_build_with_end_time(
            &intermediate_path,
            &args.output,
            "timestamp",
            Some("end_timestamp"),
            0,
            10,
            "zstd",
        )?;

        // Clean up intermediate file
        let _ = std::fs::remove_file(&intermediate_path);
    }

    println!("\n✅ Hurricane data generation complete!");

    Ok(())
}

fn process_hurricane_data(path: &PathBuf, args: &Args) -> Result<Vec<Feature>> {
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut rdr = csv::ReaderBuilder::new().from_reader(reader);

    // Group records by storm ID
    let mut storm_records: HashMap<String, Vec<IbtracsRecord>> = HashMap::new();

    for result in rdr.deserialize() {
        let record: IbtracsRecord = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Filter by basin (Atlantic)
        if record.basin != "NA" {
            continue;
        }

        storm_records
            .entry(record.storm_id.clone())
            .or_insert_with(Vec::new)
            .push(record);
    }

    let mut features = Vec::new();
    let mut processed_storms = 0;

    let target_year = args.synthetic_year.unwrap_or(args.start_year as i32);

    for (_, mut records) in storm_records {
        records.sort_by(|a, b| a.iso_time.cmp(&b.iso_time));

        if records.is_empty() {
            continue;
        }

        let first_record_dt =
            match NaiveDateTime::parse_from_str(&records[0].iso_time, "%Y-%m-%d %H:%M:%S") {
                Ok(dt) => dt,
                Err(_) => continue,
            };

        let start_year = first_record_dt.year();

        if (start_year as u32) < args.start_year || (start_year as u32) > args.end_year {
            continue;
        }

        let year_offset = if args.synthetic {
            target_year - start_year
        } else {
            0
        };

        for i in 0..records.len().saturating_sub(1) {
            let start_record = &records[i];
            let end_record = &records[i + 1];

            let mut start_dt =
                match NaiveDateTime::parse_from_str(&start_record.iso_time, "%Y-%m-%d %H:%M:%S") {
                    Ok(dt) => dt,
                    Err(_) => continue,
                };

            if args.synthetic {
                start_dt = start_dt
                    .with_year(start_dt.year() + year_offset)
                    .unwrap_or(start_dt);
            }

            let start_time = DateTime::from_naive_utc_and_offset(start_dt, Utc);

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

            let feature = common::create_linestring_feature(
                vec![
                    [start_record.lon, start_record.lat],
                    [end_record.lon, end_record.lat],
                ],
                start_time,
                properties,
            );

            features.push(feature);
        }

        processed_storms += 1;
        if processed_storms % 100 == 0 {
            println!("   Processed {} storms...", processed_storms);
        }
    }

    Ok(features)
}

fn get_status(category: i32, nature: &str) -> String {
    match nature {
        "SS" => return "subtropical_storm".to_string(),
        "ET" => return "extratropical".to_string(),
        "DS" => return "disturbance".to_string(),
        _ => {}
    }

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

/// Process hurricane data directly to GeoParquet format (streaming)
fn process_hurricane_data_parquet(path: &PathBuf, output: &PathBuf, args: &Args) -> Result<usize> {
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut rdr = csv::ReaderBuilder::new().from_reader(reader);

    // Group records by storm ID
    let mut storm_records: HashMap<String, Vec<IbtracsRecord>> = HashMap::new();

    for result in rdr.deserialize() {
        let record: IbtracsRecord = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Filter by basin (Atlantic)
        if record.basin != "NA" {
            continue;
        }

        storm_records
            .entry(record.storm_id.clone())
            .or_insert_with(Vec::new)
            .push(record);
    }

    // Create streaming Parquet writer
    let property_columns = vec![
        PropertyColumn::string("storm_id"),
        PropertyColumn::string("name"),
        PropertyColumn::numeric("wind_speed"),
        PropertyColumn::numeric("category"),
        PropertyColumn::string("status"),
        PropertyColumn::string("nature"),
        PropertyColumn::numeric("value"),
    ];
    let mut writer = StreamingLineStringParquetWriter::with_columns(output, property_columns)?;

    let mut processed_storms = 0;

    let target_year = args.synthetic_year.unwrap_or(args.start_year as i32);

    for (_, mut records) in storm_records {
        records.sort_by(|a, b| a.iso_time.cmp(&b.iso_time));

        if records.is_empty() {
            continue;
        }

        let first_record_dt =
            match NaiveDateTime::parse_from_str(&records[0].iso_time, "%Y-%m-%d %H:%M:%S") {
                Ok(dt) => dt,
                Err(_) => continue,
            };

        let start_year = first_record_dt.year();

        if (start_year as u32) < args.start_year || (start_year as u32) > args.end_year {
            continue;
        }

        let year_offset = if args.synthetic {
            target_year - start_year
        } else {
            0
        };

        for i in 0..records.len().saturating_sub(1) {
            let start_record = &records[i];
            let end_record = &records[i + 1];

            let mut start_dt =
                match NaiveDateTime::parse_from_str(&start_record.iso_time, "%Y-%m-%d %H:%M:%S") {
                    Ok(dt) => dt,
                    Err(_) => continue,
                };

            let mut end_dt =
                match NaiveDateTime::parse_from_str(&end_record.iso_time, "%Y-%m-%d %H:%M:%S") {
                    Ok(dt) => dt,
                    Err(_) => continue,
                };

            if args.synthetic {
                start_dt = start_dt
                    .with_year(start_dt.year() + year_offset)
                    .unwrap_or(start_dt);
                end_dt = end_dt
                    .with_year(end_dt.year() + year_offset)
                    .unwrap_or(end_dt);
            }

            let start_time = DateTime::from_naive_utc_and_offset(start_dt, Utc);
            let end_time = DateTime::from_naive_utc_and_offset(end_dt, Utc);

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

            let coordinates = vec![
                [start_record.lon, start_record.lat],
                [end_record.lon, end_record.lat],
            ];

            let record = LineStringRecord::new(
                coordinates,
                start_time,
                Some(end_time),
                properties,
            );

            writer.write_linestring(&record)?;
        }

        processed_storms += 1;
        if processed_storms % 100 == 0 {
            println!("   Processed {} storms...", processed_storms);
        }
    }

    writer.finish()
}


