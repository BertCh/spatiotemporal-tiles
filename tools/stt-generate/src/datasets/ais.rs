//! Process real AIS data from NOAA Marine Cadastre
//!
//! Downloads and converts USCG Marine Cadastre AIS data to STT format
//! Source: https://coast.noaa.gov/htdata/CMSP/AISDataHandler
//!
//! Usage:
//!   stt-generate ais --date 2024-01-01 --output ais.stt
//!   stt-generate ais --start-date 2024-01-01 --end-date 2024-01-07 --output ais-week.stt
//!   stt-generate ais --input existing.csv --output ais.stt

use crate::common::{self, PointRecord, PropertyColumn, StreamingParquetWriter};
use anyhow::{anyhow, Context, Result};
use chrono::{NaiveDate, NaiveDateTime};
use clap::Parser;
use csv::ReaderBuilder;
use geojson::{Feature, Geometry, Value as GeoValue};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use serde_json::{json, Map};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::PathBuf;

/// NOAA Marine Cadastre AIS data URL pattern
/// Files are organized by year: https://coast.noaa.gov/htdata/CMSP/AISDataHandler/YYYY/
const MARINE_CADASTRE_BASE_URL: &str = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler";

#[derive(Parser, Debug)]
#[command(about = "Download and process AIS maritime traffic data from NOAA Marine Cadastre")]
pub struct Args {
    /// Input CSV file (use this OR --date/--start-date)
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Download data for a specific date (format: YYYY-MM-DD)
    #[arg(long, conflicts_with = "input")]
    pub date: Option<String>,

    /// Download data starting from this date (format: YYYY-MM-DD)
    #[arg(long, conflicts_with = "input", conflicts_with = "date")]
    pub start_date: Option<String>,

    /// Download data ending on this date (format: YYYY-MM-DD, requires --start-date)
    #[arg(long, requires = "start_date")]
    pub end_date: Option<String>,

    /// Output file (.stt, .geojson, or .csv)
    #[arg(short, long, default_value = "ais-traffic.stt")]
    pub output: PathBuf,

    /// Explicit sampling interval in minutes (0 preserves every usable row)
    #[arg(long, default_value = "0", value_parser = parse_nonnegative_i64)]
    pub sample_minutes: i64,

    /// Maximum number of vessels to include (0 = unlimited)
    #[arg(long, default_value = "0")]
    pub max_vessels: usize,

    /// Geographic bounds: min_lat,min_lon,max_lat,max_lon
    #[arg(long)]
    pub bounds: Option<String>,

    /// Download directory for cached files
    #[arg(long, default_value = "data/ais")]
    pub download_dir: PathBuf,

    /// Keep downloaded files after processing
    #[arg(long)]
    pub keep_downloads: bool,

    /// Skip download, use existing files in download directory
    #[arg(long)]
    pub skip_download: bool,

    /// Skip stt-build step
    #[arg(long)]
    pub skip_build: bool,
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
        let args = Args::try_parse_from(["ais"]).unwrap();
        assert_eq!(args.sample_minutes, 0);
        assert!(Args::try_parse_from(["ais", "--sample-minutes=-1"]).is_err());
    }
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
    sog: Option<f64>,

    #[serde(rename = "COG")]
    cog: Option<f64>,

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

pub fn run(args: Args) -> Result<()> {
    println!("🚢 AIS Data Processor (NOAA Marine Cadastre)");
    println!("============================================\n");

    // Determine input source
    let input_files = if let Some(ref input) = args.input {
        // Use provided input file
        println!("📂 Input: {}", input.display());
        vec![input.clone()]
    } else if let Some(ref date_str) = args.date {
        // Download single date
        let date = parse_date(date_str)?;
        println!("📅 Date: {}", date);
        download_ais_data(&[date], &args)?
    } else if let Some(ref start_str) = args.start_date {
        // Download date range
        let start = parse_date(start_str)?;
        let end = args
            .end_date
            .as_ref()
            .map(|s| parse_date(s))
            .transpose()?
            .unwrap_or(start);

        if end < start {
            return Err(anyhow!("End date must be >= start date"));
        }

        let dates: Vec<NaiveDate> = (0..=(end - start).num_days())
            .map(|d| start + chrono::Duration::days(d))
            .collect();

        println!("📅 Date range: {} to {} ({} days)", start, end, dates.len());
        download_ais_data(&dates, &args)?
    } else {
        return Err(anyhow!(
            "Either --input, --date, or --start-date must be specified.\n\
             Examples:\n\
             --date 2024-01-01                         Download and process one day\n\
             --start-date 2024-01-01 --end-date 2024-01-07   Download and process a week\n\
             --input data/AIS_2024_01_01.csv           Use existing CSV file"
        ));
    };

    if input_files.is_empty() {
        return Err(anyhow!("No input files available"));
    }

    if args.sample_minutes > 0 {
        println!(
            "📊 Explicit sampling: 1 position per vessel per {} minutes",
            args.sample_minutes
        );
    } else {
        println!("📊 Sampling: disabled (preserving every usable AIS row)");
    }
    if let Some(ref bounds_str) = args.bounds {
        println!("📍 Geographic filter: {}", bounds_str);
    }

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

    println!();

    // Process all input files
    let (final_count, unique_vessels) = process_ais_files(
        &input_files,
        &intermediate_path,
        &args,
        use_parquet,
        use_csv,
    )?;

    println!("\n📊 Processing Summary:");
    println!("   Final features: {}", final_count);
    println!("   Unique vessels: {}", unique_vessels);

    // Build STT if output is .stt
    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        common::run_stt_build(&intermediate_path, &args.output, "timestamp", 0, 14, "zstd")?;

        // Clean up intermediate file
        let _ = fs::remove_file(&intermediate_path);
    }

    // Clean up downloaded files if not keeping
    if args.date.is_some() || args.start_date.is_some() {
        if !args.keep_downloads {
            println!("\n🧹 Cleaning up downloaded files...");
            for file in &input_files {
                let _ = fs::remove_file(file);
            }
        }
    }

    println!("\n✅ AIS data processing complete!");

    Ok(())
}

fn parse_date(s: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .with_context(|| format!("Invalid date format: {}. Use YYYY-MM-DD", s))
}

/// Download AIS data for specified dates from NOAA Marine Cadastre
fn download_ais_data(dates: &[NaiveDate], args: &Args) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(&args.download_dir)?;

    let mut csv_files = Vec::new();

    println!("\n📥 Downloading AIS data from NOAA Marine Cadastre...\n");

    let pb = ProgressBar::new(dates.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} days ({eta})")?
            .progress_chars("#>-"),
    );

    for date in dates {
        let year = date.format("%Y").to_string();
        let date_str = date.format("%Y_%m_%d").to_string();
        let filename = format!("AIS_{}.csv", date_str);
        let zip_filename = format!("AIS_{}.zip", date_str);

        let csv_path = args.download_dir.join(&filename);
        let zip_path = args.download_dir.join(&zip_filename);

        // Check if CSV already exists
        if csv_path.exists() {
            csv_files.push(csv_path);
            pb.inc(1);
            continue;
        }

        if args.skip_download {
            pb.println(format!(
                "⚠️  Skipping download for {} (file not found)",
                date
            ));
            pb.inc(1);
            continue;
        }

        // Download URL
        let url = format!("{}/{}/{}", MARINE_CADASTRE_BASE_URL, year, zip_filename);

        pb.println(format!("📥 Downloading {}...", zip_filename));

        // Download the zip file
        match download_file(&url, &zip_path) {
            Ok(_) => {
                // Unzip
                if let Err(e) = unzip_ais_file(&zip_path, &args.download_dir) {
                    pb.println(format!("⚠️  Failed to unzip {}: {}", zip_filename, e));
                    pb.inc(1);
                    continue;
                }

                // Clean up zip file
                let _ = fs::remove_file(&zip_path);

                if csv_path.exists() {
                    csv_files.push(csv_path);
                }
            }
            Err(e) => {
                pb.println(format!("⚠️  Failed to download {}: {}", date, e));
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("Download complete");

    println!("\n✓ Downloaded {} files", csv_files.len());

    Ok(csv_files)
}

/// Download a file with progress indication
fn download_file(url: &str, output_path: &PathBuf) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600)) // 10 min timeout for large files
        .build()?;

    let response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to download: {}", url))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Download failed with status: {} for {}",
            response.status(),
            url
        ));
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("   [{bar:40}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    let bytes = response.bytes()?;
    let mut file = File::create(output_path)?;
    file.write_all(&bytes)?;

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    Ok(())
}

/// Unzip AIS data file
fn unzip_ais_file(zip_path: &PathBuf, output_dir: &PathBuf) -> Result<()> {
    let file = File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = output_dir.join(file.name());

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }

    Ok(())
}

/// Process multiple AIS CSV files
fn process_ais_files(
    input_files: &[PathBuf],
    intermediate_path: &PathBuf,
    args: &Args,
    use_parquet: bool,
    use_csv: bool,
) -> Result<(usize, usize)> {
    // Parse bounds if provided
    let bounds = args
        .bounds
        .as_ref()
        .map(|s| common::parse_bounds(s))
        .transpose()?;

    let mut vessel_last_time: HashMap<String, i64> = HashMap::new();
    let mut total_records = 0;
    let mut unique_vessels = HashSet::new();

    let typed_columns = vec![
        PropertyColumn::string("mmsi"),
        PropertyColumn::string("vessel_type"),
        PropertyColumn::numeric("speed"),
        PropertyColumn::numeric("course"),
        PropertyColumn::numeric("heading"),
        PropertyColumn::string("vessel_name"),
        PropertyColumn::numeric("length"),
        PropertyColumn::numeric("width"),
    ];
    let property_column_names: Vec<String> = typed_columns.iter().map(|c| c.name.clone()).collect();

    let mut parquet_writer = if use_parquet {
        Some(StreamingParquetWriter::with_columns(
            intermediate_path,
            typed_columns.clone(),
        )?)
    } else {
        None
    };
    let mut csv_writer = if use_csv {
        Some(common::StreamingCsvWriter::new(
            intermediate_path,
            property_column_names,
        )?)
    } else {
        None
    };
    let mut features: Vec<Feature> = Vec::new();

    for (file_idx, input_file) in input_files.iter().enumerate() {
        println!(
            "🔄 Processing file {}/{}: {}",
            file_idx + 1,
            input_files.len(),
            input_file.file_name().unwrap_or_default().to_string_lossy()
        );

        let file = File::open(input_file)
            .with_context(|| format!("Failed to open input file: {}", input_file.display()))?;

        let reader = BufReader::new(file);
        let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

        let mut file_records = 0;

        for result in csv_reader.deserialize() {
            total_records += 1;
            file_records += 1;

            if file_records % 500000 == 0 {
                let count = if let Some(ref w) = parquet_writer {
                    w.row_count()
                } else if let Some(ref w) = csv_writer {
                    w.row_count()
                } else {
                    features.len()
                };
                println!(
                    "   Processed {} records, kept {} features, {} unique vessels...",
                    file_records,
                    count,
                    unique_vessels.len()
                );
            }

            let record: AisRecord = match result {
                Ok(r) => r,
                Err(e) => {
                    if total_records < 10 {
                        eprintln!(
                            "   Warning: Failed to parse record {}: {}",
                            total_records, e
                        );
                    }
                    continue;
                }
            };

            // Filter invalid positions
            if record.lat == 0.0 && record.lon == 0.0 {
                continue;
            }
            if record.lat > 90.0 || record.lat < -90.0 || record.lat == 91.0 {
                continue;
            }
            if record.lon > 180.0 || record.lon < -180.0 {
                continue;
            }

            // Apply geographic bounds filter
            if let Some((min_lat, min_lon, max_lat, max_lon)) = bounds {
                if record.lat < min_lat
                    || record.lat > max_lat
                    || record.lon < min_lon
                    || record.lon > max_lon
                {
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
            if args.sample_minutes > 0 {
                let sample_interval_ms = args.sample_minutes * 60 * 1000;
                if let Some(&last_time) = vessel_last_time.get(&record.mmsi) {
                    if timestamp_ms - last_time < sample_interval_ms {
                        continue;
                    }
                }
                vessel_last_time.insert(record.mmsi.clone(), timestamp_ms);
            }
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

            if let Some(ref mut writer) = parquet_writer {
                let point_record = PointRecord::new(record.lon, record.lat, timestamp, properties);
                writer.write_point(&point_record)?;
            } else if let Some(ref mut writer) = csv_writer {
                writer.write_point(record.lon, record.lat, timestamp, &properties)?;
            } else {
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
    }

    let final_count = if let Some(writer) = parquet_writer.take() {
        writer.finish()?
    } else if let Some(writer) = csv_writer.take() {
        writer.finish()?
    } else {
        let count = features.len();
        common::write_geojson(features, intermediate_path)?;
        count
    };

    Ok((final_count, unique_vessels.len()))
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
