//! Common utilities for data generation scripts

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use geojson::{Feature, FeatureCollection, GeoJson, Geometry, Value};
use serde_json::{json, Map, Value as JsonValue};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Download a file from a URL with progress bar
pub fn download_file(url: &str, output_path: &Path) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::io::copy;

    println!("Downloading: {}", url);
    
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

/// Write GeoJSON FeatureCollection to file
pub fn write_geojson(features: Vec<Feature>, output_path: &Path) -> Result<()> {
    println!("Writing {} features to {:?}", features.len(), output_path);

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
    let json_string = serde_json::to_string_pretty(&geojson)?;

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
    DateTime::from_naive_utc_and_offset(
        date.and_hms_opt(0, 0, 0).unwrap(),
        Utc,
    )
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

