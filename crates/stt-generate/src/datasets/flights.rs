//! Download and process historical OpenSky state vector data
//!
//! Data source: https://s3.opensky-network.org/data-samples/

use crate::common;
use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, TimeZone, Utc};
use clap::Parser;
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use geojson::Feature;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use serde_json::{json, Map};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(about = "Download and process OpenSky historical flight data")]
pub struct Args {
    /// Output file (.stt, .geojson, or .csv)
    #[arg(short, long, default_value = "flights.stt")]
    pub output: PathBuf,

    /// Date to download (YYYY-MM-DD, must be a Monday from 2017-2020)
    #[arg(long, default_value = "2020-01-06")]
    pub date: String,

    /// Hours to download (e.g., "0-23" for full day)
    #[arg(long, default_value = "0-23")]
    pub hours: String,

    /// Geographic bounds: min_lat,min_lon,max_lat,max_lon
    #[arg(long, default_value = "25,-125,50,-65")]
    pub bounds: String,

    /// Sampling interval in seconds
    #[arg(long, default_value = "60")]
    pub sample_seconds: i64,

    /// Keep downloaded files after processing
    #[arg(long)]
    pub keep_downloads: bool,

    /// Download directory
    #[arg(long, default_value = "data/opensky-historical")]
    pub download_dir: PathBuf,

    /// Skip download, process existing files only
    #[arg(long)]
    pub skip_download: bool,

    /// Skip stt-build step
    #[arg(long)]
    pub skip_build: bool,
}

#[derive(Debug, Deserialize)]
struct StateRecord {
    time: i64,
    icao24: String,
    lat: Option<f64>,
    lon: Option<f64>,
    velocity: Option<f64>,
    heading: Option<f64>,
    vertrate: Option<f64>,
    callsign: Option<String>,
    onground: String,
    #[allow(dead_code)]
    alert: String,
    #[allow(dead_code)]
    spi: String,
    squawk: Option<String>,
    baroaltitude: Option<f64>,
    #[allow(dead_code)]
    geoaltitude: Option<f64>,
    #[allow(dead_code)]
    lastposupdate: Option<f64>,
    #[allow(dead_code)]
    lastcontact: Option<f64>,
}

pub fn run(args: Args) -> Result<()> {
    println!("✈️  OpenSky Historical Flight Data Generator");
    println!("============================================\n");

    // Parse date
    let parsed_date = NaiveDate::parse_from_str(&args.date, "%Y-%m-%d")
        .with_context(|| format!("Invalid date format: {}. Use YYYY-MM-DD", args.date))?;

    if parsed_date.weekday() != chrono::Weekday::Mon {
        println!("⚠️  Warning: {} is not a Monday. OpenSky data is only available for Mondays.", args.date);
    }

    let (start_hour, end_hour) = parse_hours_range(&args.hours)?;
    let total_hours = end_hour - start_hour + 1;

    println!("📅 Date: {}", args.date);
    println!("⏰ Hours: {:02}:00 - {:02}:59 UTC ({} hours)", start_hour, end_hour, total_hours);
    println!("📍 Bounds: {}", args.bounds);
    println!();

    // Determine intermediate output format
    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("geojson")
    } else {
        args.output.clone()
    };

    // Download and process
    if !args.skip_download {
        fs::create_dir_all(&args.download_dir)?;

        println!("⬇️  Downloading {} hours of data...\n", total_hours);

        let pb = ProgressBar::new(total_hours as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} hours ({eta})")?
                .progress_chars("#>-"),
        );

        let mut downloaded_files: Vec<PathBuf> = Vec::new();

        for hour in start_hour..=end_hour {
            let hour_str = format!("{:02}", hour);
            let filename = format!("states_{}-{}.csv", args.date, hour_str);
            let tar_filename = format!("states_{}-{}.csv.tar", args.date, hour_str);
            let gz_filename = format!("states_{}-{}.csv.gz", args.date, hour_str);

            let tar_path = args.download_dir.join(&tar_filename);
            let gz_path = args.download_dir.join(&gz_filename);
            let csv_path = args.download_dir.join(&filename);

            if csv_path.exists() {
                downloaded_files.push(csv_path);
                pb.inc(1);
                continue;
            }

            let url = format!(
                "https://s3.opensky-network.org/data-samples/states/.{}/{}/states_{}-{}.csv.tar",
                args.date, hour_str, args.date, hour_str
            );

            if !tar_path.exists() {
                if let Err(e) = download_file(&url, &tar_path) {
                    eprintln!("Warning: Failed to download hour {}: {}", hour, e);
                    pb.inc(1);
                    continue;
                }
            }

            // Extract tar
            if tar_path.exists() && !gz_path.exists() {
                let _ = Command::new("tar")
                    .args(["-xf", tar_path.to_str().unwrap()])
                    .current_dir(&args.download_dir)
                    .output();
            }

            // Decompress gzip
            if gz_path.exists() && !csv_path.exists() {
                let _ = Command::new("gunzip")
                    .args(["-f", gz_path.to_str().unwrap()])
                    .output();
            }

            if csv_path.exists() {
                downloaded_files.push(csv_path);
            }

            pb.inc(1);
        }

        pb.finish_with_message("Download complete");

        if downloaded_files.is_empty() {
            anyhow::bail!("No files were downloaded. Check the date and try again.");
        }

        // Combine CSV files
        println!("\n📦 Combining {} CSV files...", downloaded_files.len());
        let combined_path = args.download_dir.join(format!("combined-{}.csv", args.date));

        {
            let mut combined_file = File::create(&combined_path)?;

            // Write header from first file
            let first_file = File::open(&downloaded_files[0])?;
            let mut reader = BufReader::new(first_file);
            let mut header = String::new();
            std::io::BufRead::read_line(&mut reader, &mut header)?;
            combined_file.write_all(header.as_bytes())?;

            for csv_path in &downloaded_files {
                let file = File::open(csv_path)?;
                let reader = BufReader::new(file);
                let mut first_line = true;
                for line in std::io::BufRead::lines(reader) {
                    let line = line?;
                    if first_line {
                        first_line = false;
                        continue;
                    }
                    combined_file.write_all(line.as_bytes())?;
                    combined_file.write_all(b"\n")?;
                }
            }
        }

        // Process combined file
        process_csv(&combined_path, &intermediate_path, Some(&args.bounds), args.sample_seconds)?;

        // Cleanup if requested
        if !args.keep_downloads {
            println!("\n🧹 Cleaning up downloaded files...");
            for file in downloaded_files {
                let _ = fs::remove_file(&file);
            }
            let _ = fs::remove_file(&combined_path);
            for entry in fs::read_dir(&args.download_dir)? {
                let entry = entry?;
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "tar" || ext == "gz" {
                        let _ = fs::remove_file(path);
                    }
                }
            }
        }
    } else {
        // Process existing combined file
        let combined_path = args.download_dir.join(format!("combined-{}.csv", args.date));
        if !combined_path.exists() {
            anyhow::bail!("Combined CSV not found: {}. Run without --skip-download first.", combined_path.display());
        }
        process_csv(&combined_path, &intermediate_path, Some(&args.bounds), args.sample_seconds)?;
    }

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
        let _ = fs::remove_file(&intermediate_path);
    }

    println!("\n✅ Flight data generation complete!");

    Ok(())
}

fn download_file(url: &str, output_path: &PathBuf) -> Result<()> {
    let response = reqwest::blocking::get(url)
        .with_context(|| format!("Failed to download: {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    let bytes = response.bytes()?;
    let mut file = File::create(output_path)?;
    file.write_all(&bytes)?;

    Ok(())
}

fn parse_hours_range(hours: &str) -> Result<(u8, u8)> {
    let parts: Vec<&str> = hours.split('-').collect();
    if parts.len() != 2 {
        anyhow::bail!("Hours must be in format: START-END (e.g., 0-23)");
    }

    let start: u8 = parts[0].parse()?;
    let end: u8 = parts[1].parse()?;

    if start > 23 || end > 23 || start > end {
        anyhow::bail!("Hours must be 0-23 and start <= end");
    }

    Ok((start, end))
}

fn process_csv(
    input: &PathBuf,
    output: &PathBuf,
    bounds: Option<&str>,
    sample_seconds: i64,
) -> Result<()> {
    println!("\n🔄 Processing flight data...");

    let bounds_parsed = bounds.map(|s| common::parse_bounds(s)).transpose()?;

    let file = File::open(input)?;
    let is_gzipped = input.extension().map(|ext| ext == "gz").unwrap_or(false);

    let reader: Box<dyn Read> = if is_gzipped {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let buf_reader = BufReader::new(reader);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(buf_reader);

    let mut aircraft_last_time: HashMap<String, i64> = HashMap::new();
    let mut features: Vec<Feature> = Vec::new();
    let mut total_records = 0;
    let mut filtered_records = 0;
    let mut ground_filtered = 0;
    let mut unique_aircraft = HashSet::new();

    for result in csv_reader.deserialize() {
        total_records += 1;

        if total_records % 500000 == 0 {
            println!(
                "   Processed {} records, {} features, {} aircraft...",
                total_records, features.len(), unique_aircraft.len()
            );
        }

        let record: StateRecord = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        if record.onground.to_lowercase() == "true" {
            ground_filtered += 1;
            continue;
        }

        let (lat, lon) = match (record.lat, record.lon) {
            (Some(lat), Some(lon)) if lat.abs() <= 90.0 && lon.abs() <= 180.0 => (lat, lon),
            _ => continue,
        };

        if let Some((min_lat, min_lon, max_lat, max_lon)) = bounds_parsed {
            if lat < min_lat || lat > max_lat || lon < min_lon || lon > max_lon {
                filtered_records += 1;
                continue;
            }
        }

        if let Some(&last_time) = aircraft_last_time.get(&record.icao24) {
            if record.time - last_time < sample_seconds {
                continue;
            }
        }
        aircraft_last_time.insert(record.icao24.clone(), record.time);
        unique_aircraft.insert(record.icao24.clone());

        let timestamp = Utc.timestamp_opt(record.time, 0).single().unwrap_or(Utc::now());

        let mut properties = Map::new();
        properties.insert("icao24".to_string(), json!(record.icao24));

        if let Some(ref callsign) = record.callsign {
            let cs = callsign.trim();
            if !cs.is_empty() {
                properties.insert("callsign".to_string(), json!(cs));
            }
        }

        if let Some(altitude) = record.baroaltitude {
            properties.insert("altitude".to_string(), json!((altitude * 3.28084) as i32));
        }

        if let Some(velocity) = record.velocity {
            properties.insert("speed".to_string(), json!((velocity * 1.94384) as i32));
        }

        if let Some(heading) = record.heading {
            properties.insert("heading".to_string(), json!(heading as i32));
        }

        if let Some(vertrate) = record.vertrate {
            properties.insert("vertical_rate".to_string(), json!((vertrate * 196.85) as i32));
        }

        if let Some(ref squawk) = record.squawk {
            properties.insert("squawk".to_string(), json!(squawk));
        }

        let feature = common::create_point_feature(lon, lat, timestamp, properties);
        features.push(feature);
    }

    println!("\n📊 Processing Summary:");
    println!("   Total records: {}", total_records);
    println!("   Ground filtered: {}", ground_filtered);
    println!("   Geographic filtered: {}", filtered_records);
    println!("   Final features: {}", features.len());
    println!("   Unique aircraft: {}", unique_aircraft.len());

    if !features.is_empty() {
        common::write_geojson(features, output)?;
    }

    Ok(())
}


