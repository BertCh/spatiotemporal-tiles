//! Input file parsing and feature loading
//!
//! Supports GeoJSON and CSV formats using standard Rust geospatial libraries.

use anyhow::{Context, Result};
use geojson::{Feature, FeatureCollection, GeoJson};
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use stt_core::types::{BoundingBox, TimeRange};

/// Parsed feature with geometry and temporal information
#[derive(Debug, Clone)]
pub struct ParsedFeature {
    pub geojson: Feature,
    pub timestamp: u64,
    pub lon: f64,
    pub lat: f64,
}

/// Load features from an input file
pub fn load_features(
    path: &Path,
    time_field: &str,
    time_format: &str,
) -> Result<Vec<ParsedFeature>> {
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    match extension.to_lowercase().as_str() {
        "geojson" | "json" => load_geojson(path, time_field, time_format),
        "csv" => load_csv(path, time_field, time_format),
        _ => anyhow::bail!("Unsupported file format: {}", extension),
    }
}

/// Load GeoJSON file
fn load_geojson(
    path: &Path,
    time_field: &str,
    time_format: &str,
) -> Result<Vec<ParsedFeature>> {
    let file = File::open(path).context("Failed to open input file")?;
    let reader = BufReader::new(file);
    let geojson: GeoJson = serde_json::from_reader(reader)
        .context("Failed to parse GeoJSON")?;

    let features = match geojson {
        GeoJson::FeatureCollection(fc) => fc.features,
        GeoJson::Feature(f) => vec![f],
        _ => anyhow::bail!("Expected FeatureCollection or Feature"),
    };

    let mut parsed = Vec::new();

    for feature in features {
        if let Some(parsed_feature) = parse_feature(feature, time_field, time_format)? {
            parsed.push(parsed_feature);
        }
    }

    Ok(parsed)
}

/// Load CSV file
///
/// Now uses the standard `csv` crate for robust parsing.
/// Expects columns: lon, lat, timestamp, and optional property columns.
fn load_csv(
    path: &Path,
    time_field: &str,
    time_format: &str,
) -> Result<Vec<ParsedFeature>> {
    use csv::ReaderBuilder;
    use geojson::{Geometry, Value as GeomValue};
    
    let file = File::open(path).context("Failed to open CSV file")?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);
    
    let headers = reader.headers()
        .context("Failed to read CSV headers")?
        .clone();
    
    // Find required columns
    let lon_idx = find_column(&headers, &["lon", "longitude", "lng", "x"])
        .ok_or_else(|| anyhow::anyhow!("CSV must have a longitude column (lon, longitude, lng, or x)"))?;
    let lat_idx = find_column(&headers, &["lat", "latitude", "y"])
        .ok_or_else(|| anyhow::anyhow!("CSV must have a latitude column (lat, latitude, or y)"))?;
    let time_idx = find_column(&headers, &[time_field])
        .ok_or_else(|| anyhow::anyhow!("CSV must have a {} column", time_field))?;
    
    let mut parsed = Vec::new();
    
    for result in reader.records() {
        let record = result.context("Failed to read CSV record")?;
        
        // Parse coordinates
        let lon: f64 = record.get(lon_idx)
            .ok_or_else(|| anyhow::anyhow!("Missing longitude value"))?
            .parse()
            .context("Failed to parse longitude")?;
        let lat: f64 = record.get(lat_idx)
            .ok_or_else(|| anyhow::anyhow!("Missing latitude value"))?
            .parse()
            .context("Failed to parse latitude")?;
        
        // Parse timestamp
        let time_str = record.get(time_idx)
            .ok_or_else(|| anyhow::anyhow!("Missing timestamp value"))?;
        let timestamp = parse_timestamp_str(time_str, time_format)?;
        
        // Build properties from other columns
        let mut properties = serde_json::Map::new();
        for (idx, header) in headers.iter().enumerate() {
            if idx != lon_idx && idx != lat_idx && idx != time_idx {
                if let Some(value) = record.get(idx) {
                    // Try to parse as number, otherwise treat as string
                    let json_value = if let Ok(n) = value.parse::<f64>() {
                        serde_json::json!(n)
                    } else if let Ok(i) = value.parse::<i64>() {
                        serde_json::json!(i)
                    } else if let Ok(b) = value.parse::<bool>() {
                        serde_json::json!(b)
                    } else {
                        serde_json::json!(value)
                    };
                    properties.insert(header.to_string(), json_value);
                }
            }
        }
        
        // Create GeoJSON feature
        let feature = Feature {
            bbox: None,
            geometry: Some(Geometry::new(GeomValue::Point(vec![lon, lat]))),
            id: None,
            properties: Some(properties),
            foreign_members: None,
        };
        
        parsed.push(ParsedFeature {
            geojson: feature,
            timestamp,
            lon,
            lat,
        });
    }
    
    Ok(parsed)
}

/// Find a column index by trying multiple possible names
fn find_column(headers: &csv::StringRecord, names: &[&str]) -> Option<usize> {
    for name in names {
        for (idx, header) in headers.iter().enumerate() {
            if header.eq_ignore_ascii_case(name) {
                return Some(idx);
            }
        }
    }
    None
}

/// Parse timestamp from string
fn parse_timestamp_str(s: &str, format: &str) -> Result<u64> {
    match format {
        "unix-ms" => {
            let ms = s.parse::<u64>()
                .context("Expected numeric timestamp in milliseconds")?;
            Ok(ms)
        }
        "unix-sec" => {
            let sec = s.parse::<u64>()
                .context("Expected numeric timestamp in seconds")?;
            Ok(sec * 1000)
        }
        "iso8601" => {
            parse_iso8601(s)
        }
        _ => anyhow::bail!("Unknown time format: {}", format),
    }
}

/// Parse a single GeoJSON feature
fn parse_feature(
    feature: Feature,
    time_field: &str,
    time_format: &str,
) -> Result<Option<ParsedFeature>> {
    // Extract timestamp from properties
    let timestamp = match feature.properties.as_ref() {
        Some(props) => {
            let value = props.get(time_field).context(format!(
                "Time field '{}' not found in feature properties",
                time_field
            ))?;
            parse_timestamp(value, time_format)?
        }
        None => return Ok(None),
    };

    // Extract coordinates
    let (lon, lat) = match &feature.geometry {
        Some(geom) => extract_coordinates(geom)?,
        None => return Ok(None),
    };

    Ok(Some(ParsedFeature {
        geojson: feature,
        timestamp,
        lon,
        lat,
    }))
}

/// Parse timestamp from a JSON value
fn parse_timestamp(value: &Value, format: &str) -> Result<u64> {
    match format {
        "unix-ms" => {
            let ms = value.as_u64()
                .or_else(|| value.as_i64().map(|i| i as u64))
                .context("Expected numeric timestamp")?;
            Ok(ms)
        }
        "unix-sec" => {
            let sec = value.as_u64()
                .or_else(|| value.as_i64().map(|i| i as u64))
                .context("Expected numeric timestamp")?;
            Ok(sec * 1000)
        }
        "iso8601" => {
            let s = value.as_str().context("Expected string timestamp")?;
            parse_iso8601(s)
        }
        _ => anyhow::bail!("Unknown time format: {}", format),
    }
}

/// Parse ISO 8601 timestamp to Unix milliseconds
fn parse_iso8601(s: &str) -> Result<u64> {
    use chrono::{DateTime, NaiveDateTime};

    // Try parsing as DateTime with timezone
    if let Ok(dt) = s.parse::<DateTime<chrono::Utc>>() {
        return Ok(dt.timestamp_millis() as u64);
    }

    // Try parsing as NaiveDateTime
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt.and_utc().timestamp_millis() as u64);
    }

    // Try parsing as date only
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        return Ok(dt.timestamp_millis() as u64);
    }

    anyhow::bail!("Failed to parse timestamp: {}", s)
}

/// Extract coordinates from GeoJSON geometry
fn extract_coordinates(geom: &geojson::Geometry) -> Result<(f64, f64)> {
    use geojson::Value as GeomValue;

    match &geom.value {
        GeomValue::Point(coords) => {
            if coords.len() >= 2 {
                Ok((coords[0], coords[1]))
            } else {
                anyhow::bail!("Invalid Point coordinates")
            }
        }
        GeomValue::LineString(coords) | GeomValue::MultiPoint(coords) => {
            if let Some(first) = coords.first() {
                if first.len() >= 2 {
                    return Ok((first[0], first[1]));
                }
            }
            anyhow::bail!("Invalid LineString/MultiPoint coordinates")
        }
        GeomValue::Polygon(rings) => {
            if let Some(exterior) = rings.first() {
                if let Some(first) = exterior.first() {
                    if first.len() >= 2 {
                        return Ok((first[0], first[1]));
                    }
                }
            }
            anyhow::bail!("Invalid Polygon coordinates")
        }
        _ => anyhow::bail!("Unsupported geometry type for coordinate extraction"),
    }
}

/// Calculate bounding box and time range from features
pub fn calculate_bounds(features: &[ParsedFeature]) -> Result<(BoundingBox, TimeRange)> {
    if features.is_empty() {
        anyhow::bail!("Cannot calculate bounds from empty feature set");
    }

    let mut min_lon = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    let mut min_time = u64::MAX;
    let mut max_time = u64::MIN;

    for feature in features {
        min_lon = min_lon.min(feature.lon);
        max_lon = max_lon.max(feature.lon);
        min_lat = min_lat.min(feature.lat);
        max_lat = max_lat.max(feature.lat);
        min_time = min_time.min(feature.timestamp);
        max_time = max_time.max(feature.timestamp);
    }

    let bounds = BoundingBox::new(min_lon, min_lat, max_lon, max_lat);
    let time_range = TimeRange::new(min_time, max_time);

    Ok((bounds, time_range))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_iso8601() {
        // ISO 8601 with timezone
        let ts = parse_iso8601("2020-01-01T00:00:00Z").unwrap();
        assert_eq!(ts, 1577836800000);

        // Date only
        let ts = parse_iso8601("2020-01-01").unwrap();
        assert_eq!(ts, 1577836800000);
    }

    #[test]
    fn test_parse_timestamp() {
        // Unix milliseconds
        let value = serde_json::json!(1577836800000u64);
        let ts = parse_timestamp(&value, "unix-ms").unwrap();
        assert_eq!(ts, 1577836800000);

        // Unix seconds
        let value = serde_json::json!(1577836800u64);
        let ts = parse_timestamp(&value, "unix-sec").unwrap();
        assert_eq!(ts, 1577836800000);

        // ISO 8601
        let value = serde_json::json!("2020-01-01T00:00:00Z");
        let ts = parse_timestamp(&value, "iso8601").unwrap();
        assert_eq!(ts, 1577836800000);
    }
}

