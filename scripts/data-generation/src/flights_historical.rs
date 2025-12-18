//! Download and process historical OpenSky state vector data
//!
//! Downloads and processes historical flight data from OpenSky's free S3 bucket.
//! Data is available for Mondays from 2017-2020.
//!
//! Data source: https://s3.opensky-network.org/data-samples/
//! Format: CSV with state vectors at 10-second intervals

mod common;

use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Utc};
use clap::{Parser, Subcommand};
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use geojson::Feature;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(name = "process-opensky-historical")]
#[command(about = "Download and process OpenSky historical flight data")]
#[command(long_about = "
Download and process historical flight data from OpenSky Network's free S3 bucket.

DATA SOURCE:
  URL: https://s3.opensky-network.org/data-samples/states/
  Available: Weekly Monday snapshots from 2017 to May 2020
  Format: Gzipped CSV files (~150 MB/hour compressed, ~400 MB uncompressed)
  Update: 10-second intervals

AVAILABLE DATES (Mondays only):
  2017-06-05 through 2020-05-18

EXAMPLES:
  # Download and process a full day of US flight data
  cargo run --release --bin process-opensky-historical -- download \\
    --date 2020-01-06 \\
    --output flights.geojson \\
    --bounds 25,-125,50,-65

  # Process an already-downloaded CSV file
  cargo run --release --bin process-opensky-historical -- process \\
    --input combined.csv \\
    --output flights.geojson \\
    --bounds 25,-125,50,-65
")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download historical data from OpenSky and process it
    Download {
        /// Date to download (YYYY-MM-DD, must be a Monday from 2017-2020)
        #[arg(short, long)]
        date: String,

        /// Hours to download (e.g., "0-23" for full day, "12-18" for afternoon)
        #[arg(long, default_value = "0-23")]
        hours: String,

        /// Output GeoJSON file
        #[arg(short, long, default_value = "flights-historical.geojson")]
        output: PathBuf,

        /// Geographic bounds: min_lat,min_lon,max_lat,max_lon
        #[arg(long)]
        bounds: Option<String>,

        /// Sampling interval in seconds
        #[arg(long, default_value = "60")]
        sample_seconds: i64,

        /// Keep downloaded files after processing
        #[arg(long)]
        keep_downloads: bool,

        /// Download directory
        #[arg(long, default_value = "data/opensky-historical")]
        download_dir: PathBuf,
    },

    /// Process an already-downloaded CSV file
    Process {
        /// Input CSV file (can be .csv or .csv.gz)
        #[arg(short, long)]
        input: PathBuf,

        /// Output GeoJSON file
        #[arg(short, long, default_value = "flights-historical.geojson")]
        output: PathBuf,

        /// Geographic bounds: min_lat,min_lon,max_lat,max_lon
        #[arg(long)]
        bounds: Option<String>,

        /// Sampling interval in seconds
        #[arg(long, default_value = "60")]
        sample_seconds: i64,

        /// Maximum number of features to output (0 = unlimited)
        #[arg(long, default_value = "0")]
        max_features: usize,

        /// Filter to only include flights in the air
        #[arg(long, default_value = "true")]
        in_flight_only: bool,
    },
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

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Download {
            date,
            hours,
            output,
            bounds,
            sample_seconds,
            keep_downloads,
            download_dir,
        } => {
            download_and_process(
                &date,
                &hours,
                &output,
                bounds.as_deref(),
                sample_seconds,
                keep_downloads,
                &download_dir,
            )
        }
        Commands::Process {
            input,
            output,
            bounds,
            sample_seconds,
            max_features,
            in_flight_only,
        } => process_csv(&input, &output, bounds.as_deref(), sample_seconds, max_features, in_flight_only),
    }
}

fn download_and_process(
    date: &str,
    hours: &str,
    output: &PathBuf,
    bounds: Option<&str>,
    sample_seconds: i64,
    keep_downloads: bool,
    download_dir: &PathBuf,
) -> Result<()> {
    println!("✈️  OpenSky Historical Data Downloader");
    println!("======================================\n");

    // Parse date
    let parsed_date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .with_context(|| format!("Invalid date format: {}. Use YYYY-MM-DD", date))?;

    // Validate it's a Monday
    if parsed_date.weekday() != chrono::Weekday::Mon {
        println!("⚠️  Warning: {} is not a Monday. OpenSky data is only available for Mondays.", date);
        println!("   Proceeding anyway, but download may fail.\n");
    }

    // Parse hours range
    let (start_hour, end_hour) = parse_hours_range(hours)?;
    let total_hours = end_hour - start_hour + 1;

    println!("📅 Date: {}", date);
    println!("⏰ Hours: {:02}:00 - {:02}:59 UTC ({} hours)", start_hour, end_hour, total_hours);
    println!("📁 Download directory: {}", download_dir.display());
    println!();

    // Create download directory
    fs::create_dir_all(download_dir)?;

    // Download files
    println!("⬇️  Downloading {} hours of data...\n", total_hours);

    let pb = ProgressBar::new(total_hours as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} hours ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut downloaded_files: Vec<PathBuf> = Vec::new();

    for hour in start_hour..=end_hour {
        let hour_str = format!("{:02}", hour);
        let filename = format!("states_{}-{}.csv", date, hour_str);
        let tar_filename = format!("states_{}-{}.csv.tar", date, hour_str);
        let gz_filename = format!("states_{}-{}.csv.gz", date, hour_str);

        let tar_path = download_dir.join(&tar_filename);
        let gz_path = download_dir.join(&gz_filename);
        let csv_path = download_dir.join(&filename);

        // Check if CSV already exists
        if csv_path.exists() {
            downloaded_files.push(csv_path);
            pb.inc(1);
            continue;
        }

        // Download tar file
        let url = format!(
            "https://s3.opensky-network.org/data-samples/states/.{}/{}/states_{}-{}.csv.tar",
            date, hour_str, date, hour_str
        );

        if !tar_path.exists() {
            download_file(&url, &tar_path)?;
        }

        // Extract tar
        if tar_path.exists() && !gz_path.exists() {
            let output = Command::new("tar")
                .args(["-xf", tar_path.to_str().unwrap()])
                .current_dir(download_dir)
                .output()?;

            if !output.status.success() {
                eprintln!("Warning: Failed to extract {}", tar_filename);
            }
        }

        // Decompress gzip
        if gz_path.exists() && !csv_path.exists() {
            let output = Command::new("gunzip")
                .args(["-f", gz_path.to_str().unwrap()])
                .output()?;

            if !output.status.success() {
                eprintln!("Warning: Failed to decompress {}", gz_filename);
            }
        }

        if csv_path.exists() {
            downloaded_files.push(csv_path);
        }

        pb.inc(1);
    }

    pb.finish_with_message("Download complete");
    println!();

    if downloaded_files.is_empty() {
        anyhow::bail!("No files were downloaded. Check the date and try again.");
    }

    // Combine CSV files
    println!("📦 Combining {} CSV files...", downloaded_files.len());
    let combined_path = download_dir.join(format!("combined-{}.csv", date));

    {
        let mut combined_file = File::create(&combined_path)?;

        // Write header from first file
        let first_file = File::open(&downloaded_files[0])?;
        let mut reader = BufReader::new(first_file);
        let mut header = String::new();
        reader.read_line(&mut header)?;
        combined_file.write_all(header.as_bytes())?;

        // Append data from all files
        for (i, csv_path) in downloaded_files.iter().enumerate() {
            let file = File::open(csv_path)?;
            let reader = BufReader::new(file);
            let mut first_line = true;
            for line in std::io::BufRead::lines(reader) {
                let line = line?;
                if first_line {
                    first_line = false;
                    continue; // Skip header
                }
                combined_file.write_all(line.as_bytes())?;
                combined_file.write_all(b"\n")?;
            }
            print!("\r  Combined {} of {} files...", i + 1, downloaded_files.len());
            std::io::stdout().flush()?;
        }
        println!("\r  Combined {} files.              ", downloaded_files.len());
    }

    println!();

    // Process combined file
    process_csv(&combined_path, output, bounds, sample_seconds, 0, true)?;

    // Cleanup if requested
    if !keep_downloads {
        println!();
        println!("🧹 Cleaning up downloaded files...");
        for file in downloaded_files {
            let _ = fs::remove_file(&file);
        }
        let _ = fs::remove_file(&combined_path);
        // Remove tar and gz files
        for entry in fs::read_dir(download_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "tar" || ext == "gz" {
                    let _ = fs::remove_file(path);
                }
            }
        }
        println!("✅ Cleanup complete");
    }

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
        anyhow::bail!("Hours must be in format: START-END (e.g., 0-23 or 12-18)");
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
    max_features: usize,
    in_flight_only: bool,
) -> Result<()> {
    println!("✈️  OpenSky Historical Data Processor");
    println!("=====================================\n");
    println!("📂 Input: {}", input.display());
    println!("📊 Sampling: 1 position per aircraft per {} seconds", sample_seconds);

    let bounds_parsed = bounds.map(|s| parse_bounds(s)).transpose()?;
    if let Some((min_lat, min_lon, max_lat, max_lon)) = bounds_parsed {
        println!("📍 Geographic filter: [{}, {}] to [{}, {}]", min_lat, min_lon, max_lat, max_lon);
    } else {
        println!("📍 Geographic filter: None (global)");
    }

    println!();

    let file = File::open(input)
        .with_context(|| format!("Failed to open input file: {}", input.display()))?;

    let is_gzipped = input.extension().map(|ext| ext == "gz").unwrap_or(false);

    let reader: Box<dyn std::io::Read> = if is_gzipped {
        println!("📦 Decompressing gzipped input...");
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
    let mut unique_aircraft = std::collections::HashSet::new();

    println!("🔄 Processing CSV data...");

    for result in csv_reader.deserialize() {
        total_records += 1;

        if total_records % 500000 == 0 {
            println!(
                "  Processed {} records, kept {} features, {} unique aircraft...",
                total_records,
                features.len(),
                unique_aircraft.len()
            );
        }

        if max_features > 0 && features.len() >= max_features {
            println!("  Reached max features limit: {}", max_features);
            break;
        }

        let record: StateRecord = match result {
            Ok(r) => r,
            Err(e) => {
                if total_records < 10 {
                    eprintln!("  Warning: Failed to parse record {}: {}", total_records, e);
                }
                continue;
            }
        };

        if in_flight_only && record.onground.to_lowercase() == "true" {
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

        let sample_interval = sample_seconds;
        if let Some(&last_time) = aircraft_last_time.get(&record.icao24) {
            if record.time - last_time < sample_interval {
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

    println!();
    println!("📊 Processing Summary:");
    println!("  Total records read: {}", total_records);
    println!("  Ground traffic filtered: {}", ground_filtered);
    println!("  Geographic filter: {} records removed", filtered_records);
    println!("  Final features: {}", features.len());
    println!("  Unique aircraft: {}", unique_aircraft.len());

    if features.is_empty() {
        println!();
        println!("⚠️  No features to write! Check your filters.");
        return Ok(());
    }

    let first_time: Option<DateTime<Utc>> = features
        .first()
        .and_then(|f| f.properties.as_ref())
        .and_then(|p| p.get("timestamp"))
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));

    let last_time: Option<DateTime<Utc>> = features
        .last()
        .and_then(|f| f.properties.as_ref())
        .and_then(|p| p.get("timestamp"))
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));

    if let (Some(start), Some(end)) = (first_time, last_time) {
        println!(
            "  Time range: {} to {}",
            start.format("%Y-%m-%d %H:%M"),
            end.format("%Y-%m-%d %H:%M")
        );
    }

    println!();
    println!("💾 Writing GeoJSON...");
    common::write_geojson(features, output)?;

    println!();
    println!("✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output flights.stt \\",
        output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 0 \\");
    println!("             --max-zoom 10 \\");
    println!("             --compression gzip");

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

// Helper trait to read lines
trait ReadLine {
    fn read_line(&mut self, buf: &mut String) -> std::io::Result<usize>;
}

impl<R: Read> ReadLine for BufReader<R> {
    fn read_line(&mut self, buf: &mut String) -> std::io::Result<usize> {
        std::io::BufRead::read_line(self, buf)
    }
}
