//! Download and process historical OpenSky state vector data
//!
//! Data source: https://s3.opensky-network.org/data-samples/

use crate::common::{
    self, LineStringRecord, PointRecord, PropertyColumn, StreamingLineStringParquetWriter,
    StreamingParquetWriter,
};
use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, TimeZone, Utc};
use clap::Parser;
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use serde_json::{json, Map};
use std::collections::{BTreeMap, HashMap, HashSet};
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

    /// Explicit sampling interval in seconds (0 preserves every usable row)
    #[arg(long, default_value = "0", value_parser = parse_nonnegative_i64)]
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

    /// Output flight paths (LineStrings) instead of individual points
    #[arg(long)]
    pub paths: bool,

    /// Minimum points per flight for path output (filters short tracks)
    #[arg(long, default_value = "5")]
    pub min_points: usize,

    /// Maximum gap in seconds between points before starting a new segment
    #[arg(long, default_value = "300")]
    pub max_gap_seconds: i64,
}

fn parse_nonnegative_i64(value: &str) -> std::result::Result<i64, String> {
    let parsed = value
        .parse::<i64>()
        .map_err(|_| format!("expected a non-negative integer, got {value:?}"))?;
    if parsed < 0 {
        return Err(format!("expected a non-negative integer, got {parsed}"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod args_tests {
    use super::Args;
    use clap::Parser;

    #[test]
    fn sampling_is_lossless_by_default_and_negative_values_are_rejected() {
        let args = Args::try_parse_from(["flights"]).unwrap();
        assert_eq!(args.sample_seconds, 0);
        assert!(Args::try_parse_from(["flights", "--sample-seconds=-1"]).is_err());
    }
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
    squawk: Option<String>,
    baroaltitude: Option<f64>,
}

pub fn run(args: Args) -> Result<()> {
    println!("✈️  OpenSky Historical Flight Data Generator");
    println!("============================================\n");

    // Parse date
    let parsed_date = NaiveDate::parse_from_str(&args.date, "%Y-%m-%d")
        .with_context(|| format!("Invalid date format: {}. Use YYYY-MM-DD", args.date))?;

    if parsed_date.weekday() != chrono::Weekday::Mon {
        println!(
            "⚠️  Warning: {} is not a Monday. OpenSky data is only available for Mondays.",
            args.date
        );
    }

    let (start_hour, end_hour) = parse_hours_range(&args.hours)?;
    let total_hours = end_hour - start_hour + 1;

    println!("📅 Date: {}", args.date);
    println!(
        "⏰ Hours: {:02}:00 - {:02}:59 UTC ({} hours)",
        start_hour, end_hour, total_hours
    );
    println!("📍 Bounds: {}", args.bounds);
    if args.paths {
        println!("🛤️  Mode: Flight paths (LineStrings)");
    } else {
        println!("📍 Mode: Flight points");
    }
    println!();

    // Determine intermediate output format (use Parquet for efficiency)
    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("parquet")
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

            // Extract tar (produces .csv.gz file). `current_dir` is already the
            // download dir, so pass the BARE filename — passing the full
            // download-dir-prefixed path here made tar resolve
            // `<download_dir>/<download_dir>/states_….csv.tar`, which never
            // exists ("Failed to open archive"), silently breaking every build.
            if tar_path.exists() && !gz_path.exists() {
                let status = Command::new("tar")
                    .args(["-xf", &tar_filename])
                    .current_dir(&args.download_dir)
                    .status();
                if let Err(e) = status {
                    eprintln!("Warning: Failed to extract {}: {}", tar_path.display(), e);
                }
            }

            // Decompress gzip (produces .csv file)
            if gz_path.exists() && !csv_path.exists() {
                let status = Command::new("gunzip")
                    .arg("-k") // -k keeps the original .gz file
                    .arg(&gz_path)
                    .status();
                if let Err(e) = status {
                    eprintln!("Warning: Failed to decompress {}: {}", gz_path.display(), e);
                }
            }

            if csv_path.exists() {
                downloaded_files.push(csv_path);
            } else {
                eprintln!(
                    "Warning: CSV file not found after extraction: {}",
                    csv_path.display()
                );
            }

            pb.inc(1);
        }

        pb.finish_with_message("Download complete");

        if downloaded_files.is_empty() {
            anyhow::bail!("No files were downloaded. Check the date and try again.");
        }

        // Combine CSV files
        println!("\n📦 Combining {} CSV files...", downloaded_files.len());
        let combined_path = args
            .download_dir
            .join(format!("combined-{}.csv", args.date));
        combine_csv_files(&downloaded_files, &combined_path)?;

        // Process combined file
        if args.paths {
            process_csv_paths(
                &combined_path,
                &intermediate_path,
                Some(&args.bounds),
                args.sample_seconds,
                args.min_points,
                args.max_gap_seconds,
            )?;
        } else {
            process_csv(
                &combined_path,
                &intermediate_path,
                Some(&args.bounds),
                args.sample_seconds,
            )?;
        }

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
        // Process existing files
        let combined_path = args
            .download_dir
            .join(format!("combined-{}.csv", args.date));

        if !combined_path.exists() {
            // Try to find and combine individual CSV files
            let (start_hour, end_hour) = parse_hours_range(&args.hours)?;
            let mut csv_files: Vec<PathBuf> = Vec::new();

            for hour in start_hour..=end_hour {
                let hour_str = format!("{:02}", hour);
                let csv_path = args
                    .download_dir
                    .join(format!("states_{}-{}.csv", args.date, hour_str));
                let gz_path = args
                    .download_dir
                    .join(format!("states_{}-{}.csv.gz", args.date, hour_str));

                // If only .gz exists, decompress it
                if gz_path.exists() && !csv_path.exists() {
                    println!("   Decompressing {}...", gz_path.display());
                    let status = Command::new("gunzip").arg("-k").arg(&gz_path).status();
                    if let Err(e) = status {
                        eprintln!("Warning: Failed to decompress {}: {}", gz_path.display(), e);
                    }
                }

                if csv_path.exists() {
                    csv_files.push(csv_path);
                }
            }

            if csv_files.is_empty() {
                anyhow::bail!(
                    "No CSV files found in {}. Run without --skip-download first.",
                    args.download_dir.display()
                );
            }

            // Combine CSV files
            println!("📦 Combining {} CSV files...", csv_files.len());
            combine_csv_files(&csv_files, &combined_path)?;
        }

        if args.paths {
            process_csv_paths(
                &combined_path,
                &intermediate_path,
                Some(&args.bounds),
                args.sample_seconds,
                args.min_points,
                args.max_gap_seconds,
            )?;
        } else {
            process_csv(
                &combined_path,
                &intermediate_path,
                Some(&args.bounds),
                args.sample_seconds,
            )?;
        }
    }

    // Build STT if output is .stt
    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        if args.paths {
            common::run_stt_build_with_end_time(
                &intermediate_path,
                &args.output,
                "timestamp",
                Some("end_timestamp"),
                0,
                10,
                "zstd",
            )?;
        } else {
            common::run_stt_build(&intermediate_path, &args.output, "timestamp", 0, 10, "zstd")?;
        }

        // Clean up intermediate file
        let _ = fs::remove_file(&intermediate_path);
    }

    println!("\n✅ Flight data generation complete!");

    Ok(())
}

fn download_file(url: &str, output_path: &PathBuf) -> Result<()> {
    // Use curl for more reliable downloads of large files (100-200MB each)
    // reqwest tends to timeout on these large files
    let status = Command::new("curl")
        .args([
            "-L", // Follow redirects
            "-f", // Fail silently on server errors
            "-S", // Show error messages
            "--connect-timeout",
            "30",
            "--max-time",
            "600", // 10 minute max for large files
            "-o",
        ])
        .arg(output_path)
        .arg(url)
        .status()
        .with_context(|| "Failed to run curl command")?;

    if !status.success() {
        // Clean up partial download
        let _ = fs::remove_file(output_path);
        anyhow::bail!("Download failed for: {}", url);
    }

    Ok(())
}

fn combine_csv_files(files: &[PathBuf], output: &PathBuf) -> Result<()> {
    let mut combined_file = File::create(output)?;
    let mut header_written = false;

    for csv_path in files {
        let file = File::open(csv_path)?;
        let reader = BufReader::new(file);
        let mut first_line = true;

        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if first_line {
                first_line = false;
                if !header_written {
                    combined_file.write_all(line.as_bytes())?;
                    combined_file.write_all(b"\n")?;
                    header_written = true;
                }
                continue;
            }
            combined_file.write_all(line.as_bytes())?;
            combined_file.write_all(b"\n")?;
        }
    }

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
    let mut csv_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(buf_reader);

    let mut aircraft_last_time: HashMap<String, i64> = HashMap::new();
    let mut total_records = 0;
    let mut filtered_records = 0;
    let mut ground_filtered = 0;
    let mut skipped_bad_time = 0usize;
    let mut unique_aircraft = HashSet::new();

    // Typed columns: the kinematic quantities are numeric so they survive
    // stt-build's columnar inference and can drive numeric ramps — notably
    // `altitude`, which the showcase reads as `elevationProperty`. icao24 /
    // callsign / squawk are identifiers and stay categorical (squawk can have
    // leading zeros; a sniff would wrongly promote it).
    let property_columns = vec![
        PropertyColumn::string("icao24"),
        PropertyColumn::string("callsign"),
        PropertyColumn::numeric("altitude"),
        PropertyColumn::numeric("speed"),
        PropertyColumn::numeric("heading"),
        PropertyColumn::numeric("vertical_rate"),
        PropertyColumn::string("squawk"),
    ];
    let mut writer = StreamingParquetWriter::with_columns(output, property_columns)?;

    for result in csv_reader.deserialize() {
        total_records += 1;

        if total_records % 500000 == 0 {
            println!(
                "   Processed {} records, {} written, {} aircraft...",
                total_records,
                writer.row_count(),
                unique_aircraft.len()
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

        if sample_seconds > 0 {
            if let Some(&last_time) = aircraft_last_time.get(&record.icao24) {
                if record.time - last_time < sample_seconds {
                    continue;
                }
            }
            aircraft_last_time.insert(record.icao24.clone(), record.time);
        }
        unique_aircraft.insert(record.icao24.clone());

        // Skip (don't fabricate now()) on an out-of-range epoch — a corrupt
        // row placed at "today" would land outside the advertised window and
        // contaminate the temporal index.
        let Some(timestamp) = Utc.timestamp_opt(record.time, 0).single() else {
            skipped_bad_time += 1;
            continue;
        };

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
            properties.insert(
                "vertical_rate".to_string(),
                json!((vertrate * 196.85) as i32),
            );
        }

        if let Some(ref squawk) = record.squawk {
            properties.insert("squawk".to_string(), json!(squawk));
        }

        let point_record = PointRecord::new(lon, lat, timestamp, properties);
        writer.write_point(&point_record)?;
    }

    let final_count = writer.finish()?;

    println!("\n📊 Processing Summary:");
    println!("   Total records: {}", total_records);
    println!("   Ground filtered: {}", ground_filtered);
    println!("   Geographic filtered: {}", filtered_records);
    if skipped_bad_time > 0 {
        println!("   Skipped (bad timestamp): {}", skipped_bad_time);
    }
    println!("   Final features: {}", final_count);
    println!("   Unique aircraft: {}", unique_aircraft.len());

    Ok(())
}

/// A flight position with timestamp
#[derive(Debug, Clone)]
struct FlightPosition {
    time: i64,
    lon: f64,
    lat: f64,
    callsign: Option<String>,
}

/// Process CSV to generate flight path LineStrings (grouped by aircraft)
fn process_csv_paths(
    input: &PathBuf,
    output: &PathBuf,
    bounds: Option<&str>,
    sample_seconds: i64,
    min_points: usize,
    max_gap_seconds: i64,
) -> Result<()> {
    println!("\n🔄 Processing flight data into paths...");

    let bounds_parsed = bounds.map(|s| common::parse_bounds(s)).transpose()?;

    let file = File::open(input)?;
    let is_gzipped = input.extension().map(|ext| ext == "gz").unwrap_or(false);

    let reader: Box<dyn Read> = if is_gzipped {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let buf_reader = BufReader::new(reader);
    let mut csv_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(buf_reader);

    // Group positions by aircraft (icao24)
    // Use BTreeMap to maintain insertion order for reproducibility
    let mut aircraft_positions: HashMap<String, BTreeMap<i64, FlightPosition>> = HashMap::new();
    let mut total_records = 0;
    let mut filtered_records = 0;
    let mut ground_filtered = 0;

    println!("   Reading and grouping flight positions...");

    for result in csv_reader.deserialize() {
        total_records += 1;

        if total_records % 500000 == 0 {
            println!(
                "   Read {} records, {} aircraft...",
                total_records,
                aircraft_positions.len()
            );
        }

        let record: StateRecord = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Skip ground positions
        if record.onground.to_lowercase() == "true" {
            ground_filtered += 1;
            continue;
        }

        let (lat, lon) = match (record.lat, record.lon) {
            (Some(lat), Some(lon)) if lat.abs() <= 90.0 && lon.abs() <= 180.0 => (lat, lon),
            _ => continue,
        };

        // Apply bounds filter
        if let Some((min_lat, min_lon, max_lat, max_lon)) = bounds_parsed {
            if lat < min_lat || lat > max_lat || lon < min_lon || lon > max_lon {
                filtered_records += 1;
                continue;
            }
        }

        let positions = aircraft_positions
            .entry(record.icao24.clone())
            .or_insert_with(BTreeMap::new);

        // Use exact time when sampling is disabled; otherwise coalesce into
        // the explicitly requested interval. Duplicate timestamps still
        // collapse deterministically through the BTreeMap key.
        let sample_key = if sample_seconds > 0 {
            record.time / sample_seconds * sample_seconds
        } else {
            record.time
        };

        if !positions.contains_key(&sample_key) {
            positions.insert(
                sample_key,
                FlightPosition {
                    time: record.time,
                    lon,
                    lat,
                    callsign: record.callsign.clone(),
                },
            );
        }
    }

    println!("   Found {} unique aircraft", aircraft_positions.len());
    println!("   Creating path segments...");

    // Create streaming LineString writer
    let property_columns = vec![
        PropertyColumn::string("icao24"),
        PropertyColumn::string("callsign"),
        PropertyColumn::numeric("segment_id"),
    ];
    let mut writer = StreamingLineStringParquetWriter::with_columns(output, property_columns)?;

    let mut total_points = 0;

    let pb = ProgressBar::new(aircraft_positions.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} aircraft")?
            .progress_chars("=>-"),
    );

    for (icao24, positions) in aircraft_positions {
        if positions.len() < min_points {
            pb.inc(1);
            continue;
        }

        // Convert to sorted vec
        let sorted_positions: Vec<FlightPosition> = positions.into_values().collect();

        // Split into segments based on time gaps
        let segments = split_into_segments(&sorted_positions, max_gap_seconds, min_points);

        for (segment_id, segment) in segments.iter().enumerate() {
            if segment.len() < 2 {
                continue;
            }

            // Drop the segment (rather than fabricate now()) if either
            // endpoint has an out-of-range epoch.
            let (Some(start_time), Some(end_time)) = (
                Utc.timestamp_opt(segment.first().unwrap().time, 0).single(),
                Utc.timestamp_opt(segment.last().unwrap().time, 0).single(),
            ) else {
                continue;
            };

            // Use the most common callsign in this segment
            let callsign = segment
                .iter()
                .filter_map(|p| p.callsign.as_ref())
                .filter(|s| !s.trim().is_empty())
                .next()
                .cloned();

            let mut properties = Map::new();
            properties.insert("icao24".to_string(), json!(icao24));
            if let Some(cs) = callsign {
                properties.insert("callsign".to_string(), json!(cs.trim()));
            }
            properties.insert("segment_id".to_string(), json!(segment_id));

            // Create 2D coordinates (altitude is scaled for visualization)
            // Note: For proper 3D, we'd need to handle altitude separately
            let coordinates: Vec<[f64; 2]> = segment.iter().map(|p| [p.lon, p.lat]).collect();

            let record = LineStringRecord::new(coordinates, start_time, Some(end_time), properties);

            writer.write_linestring(&record)?;
            total_points += segment.len();
        }

        pb.inc(1);
    }

    pb.finish_and_clear();
    let final_count = writer.finish()?;

    println!("\n📊 Processing Summary:");
    println!("   Total records: {}", total_records);
    println!("   Ground filtered: {}", ground_filtered);
    println!("   Geographic filtered: {}", filtered_records);
    println!("   Final path segments: {}", final_count);
    println!("   Total points in paths: {}", total_points);

    Ok(())
}

/// Split a sorted list of positions into segments based on time gaps
fn split_into_segments(
    positions: &[FlightPosition],
    max_gap_seconds: i64,
    min_points: usize,
) -> Vec<Vec<FlightPosition>> {
    let mut segments: Vec<Vec<FlightPosition>> = Vec::new();
    let mut current_segment: Vec<FlightPosition> = Vec::new();

    for pos in positions {
        if let Some(last) = current_segment.last() {
            if pos.time - last.time > max_gap_seconds {
                // Gap too large - start new segment
                if current_segment.len() >= min_points {
                    segments.push(std::mem::take(&mut current_segment));
                } else {
                    current_segment.clear();
                }
            }
        }
        current_segment.push(pos.clone());
    }

    // Don't forget the last segment
    if current_segment.len() >= min_points {
        segments.push(current_segment);
    }

    segments
}
