//! Common utilities for data generation

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use csv::Writer as CsvWriter;
use geojson::{Feature, FeatureCollection, GeoJson, Geometry, Value};
use serde_json::{json, Map, Value as JsonValue};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Download a file from a URL with progress bar
pub fn download_file(url: &str, output_path: &Path) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};

    println!("⬇️  Downloading: {}", url);

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let response = reqwest::blocking::get(url)?;
    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );

    let mut file = File::create(output_path)?;
    let mut downloaded: u64 = 0;
    let mut content = response;

    loop {
        let mut buffer = vec![0; 8192];
        match std::io::Read::read(&mut content, &mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                file.write_all(&buffer[..n])?;
                downloaded += n as u64;
                pb.set_position(downloaded);
            }
            Err(e) => return Err(e.into()),
        }
    }

    pb.finish_with_message("Download complete");
    Ok(())
}

/// Create a GeoJSON Point feature
pub fn create_point_feature(
    lon: f64,
    lat: f64,
    timestamp: DateTime<Utc>,
    properties: Map<String, JsonValue>,
) -> Feature {
    let mut props = properties;
    props.insert("timestamp".to_string(), json!(timestamp.to_rfc3339()));

    Feature {
        bbox: None,
        geometry: Some(Geometry::new(Value::Point(vec![lon, lat]))),
        id: None,
        properties: Some(props),
        foreign_members: None,
    }
}

/// Create a GeoJSON LineString feature
pub fn create_linestring_feature(
    coordinates: Vec<[f64; 2]>,
    timestamp: DateTime<Utc>,
    properties: Map<String, JsonValue>,
) -> Feature {
    let mut props = properties;
    props.insert("timestamp".to_string(), json!(timestamp.to_rfc3339()));

    let coords: Vec<Vec<f64>> = coordinates
        .into_iter()
        .map(|[lon, lat]| vec![lon, lat])
        .collect();

    Feature {
        bbox: None,
        geometry: Some(Geometry::new(Value::LineString(coords))),
        id: None,
        properties: Some(props),
        foreign_members: None,
    }
}

/// Create a GeoJSON LineString feature with time range
pub fn create_linestring_feature_with_time_range(
    coordinates: Vec<[f64; 2]>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    properties: Map<String, JsonValue>,
) -> Feature {
    let mut props = properties;
    props.insert("timestamp".to_string(), json!(start_time.to_rfc3339()));
    props.insert("end_time".to_string(), json!(end_time.to_rfc3339()));

    let coords: Vec<Vec<f64>> = coordinates
        .into_iter()
        .map(|[lon, lat]| vec![lon, lat])
        .collect();

    Feature {
        bbox: None,
        geometry: Some(Geometry::new(Value::LineString(coords))),
        id: None,
        properties: Some(props),
        foreign_members: None,
    }
}

/// Write GeoJSON FeatureCollection to file
pub fn write_geojson(features: Vec<Feature>, output_path: &Path) -> Result<()> {
    println!("💾 Writing {} features to {:?}", features.len(), output_path);

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let collection = FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    };

    let geojson = GeoJson::FeatureCollection(collection);
    let json_string = serde_json::to_string(&geojson)?;

    let mut file = File::create(output_path)?;
    file.write_all(json_string.as_bytes())?;

    println!("✓ GeoJSON written successfully");
    Ok(())
}

/// Parse date string in YYYY-MM-DD format
pub fn parse_date(date_str: &str) -> Result<NaiveDate> {
    Ok(NaiveDate::parse_from_str(date_str, "%Y-%m-%d")?)
}

/// Convert NaiveDate to DateTime<Utc>
pub fn date_to_datetime(date: NaiveDate) -> DateTime<Utc> {
    DateTime::from_naive_utc_and_offset(date.and_hms_opt(0, 0, 0).unwrap(), Utc)
}

/// Unzip a file
pub fn unzip_file(zip_path: &Path, output_dir: &Path) -> Result<()> {
    use zip::ZipArchive;

    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = output_dir.join(file.name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }

    Ok(())
}

/// Streaming CSV writer for point data
pub struct StreamingCsvWriter {
    writer: CsvWriter<BufWriter<File>>,
    headers_written: bool,
    property_columns: Vec<String>,
    row_count: usize,
}

impl StreamingCsvWriter {
    /// Create a new streaming CSV writer
    pub fn new(output_path: &Path, property_columns: Vec<String>) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(output_path)?;
        let buf_writer = BufWriter::new(file);
        let writer = CsvWriter::from_writer(buf_writer);

        Ok(Self {
            writer,
            headers_written: false,
            property_columns,
            row_count: 0,
        })
    }

    fn write_headers(&mut self) -> Result<()> {
        if self.headers_written {
            return Ok(());
        }

        let mut headers = vec!["lon".to_string(), "lat".to_string(), "timestamp".to_string()];
        headers.extend(self.property_columns.clone());
        self.writer.write_record(&headers)?;
        self.headers_written = true;
        Ok(())
    }

    /// Write a single point record
    pub fn write_point(
        &mut self,
        lon: f64,
        lat: f64,
        timestamp: DateTime<Utc>,
        properties: &Map<String, JsonValue>,
    ) -> Result<()> {
        self.write_headers()?;

        let mut record = vec![
            format!("{:.6}", lon),
            format!("{:.6}", lat),
            timestamp.to_rfc3339(),
        ];

        for col in &self.property_columns {
            let value = properties
                .get(col)
                .map(|v| json_value_to_string(v))
                .unwrap_or_default();
            record.push(value);
        }

        self.writer.write_record(&record)?;
        self.row_count += 1;
        Ok(())
    }

    /// Finish writing and return the number of rows written
    pub fn finish(mut self) -> Result<usize> {
        self.writer.flush()?;
        println!("✓ CSV written successfully ({} rows)", self.row_count);
        Ok(self.row_count)
    }

    /// Get the current row count
    pub fn row_count(&self) -> usize {
        self.row_count
    }
}

fn json_value_to_string(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => String::new(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::String(s) => s.clone(),
        JsonValue::Array(arr) => serde_json::to_string(arr).unwrap_or_default(),
        JsonValue::Object(obj) => serde_json::to_string(obj).unwrap_or_default(),
    }
}

/// Determine output format based on file extension
pub fn is_csv_output(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("csv"))
        .unwrap_or(false)
}

/// Run stt-build to create an STT archive
pub fn run_stt_build(
    input: &Path,
    output: &Path,
    time_field: &str,
    min_zoom: u8,
    max_zoom: u8,
    compression: &str,
) -> Result<()> {
    use std::process::Command;

    println!("\n📦 Building STT archive...");

    let status = Command::new("stt-build")
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--time-field")
        .arg(time_field)
        .arg("--min-zoom")
        .arg(min_zoom.to_string())
        .arg("--max-zoom")
        .arg(max_zoom.to_string())
        .arg("--compression")
        .arg(compression)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("✅ STT archive created: {}", output.display());
            Ok(())
        }
        Ok(s) => {
            anyhow::bail!("stt-build exited with status: {}", s);
        }
        Err(e) => {
            anyhow::bail!("Failed to run stt-build: {}. Is it installed?", e);
        }
    }
}

/// Parse geographic bounds from string "min_lat,min_lon,max_lat,max_lon"
pub fn parse_bounds(s: &str) -> Result<(f64, f64, f64, f64)> {
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

/// Calculate haversine distance between two points in meters
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS: f64 = 6371000.0; // meters

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS * c
}


