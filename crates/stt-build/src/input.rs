//! Input file parsing and feature loading
//!
//! Supports GeoJSON, CSV, and GeoParquet formats using standard Rust geospatial libraries.
//!
//! GeoParquet support requires the `geoparquet` feature flag.

use anyhow::{Context, Result};
use geojson::{Feature, GeoJson};
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use stt_core::types::{BoundingBox, TimeRange};

#[cfg(feature = "geoparquet")]
use {
    arrow::array::{Array, Float64Array, Int64Array, StringArray, TimestampMillisecondArray},
    arrow::datatypes::DataType,
    geojson::{Geometry, Value as GeomValue},
    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
};

/// Parsed feature with geometry and temporal information
#[derive(Debug, Clone)]
pub struct ParsedFeature {
    pub geojson: Feature,
    pub timestamp: u64,
    /// End timestamp for features with time ranges (if provided)
    pub end_timestamp: Option<u64>,
    pub lon: f64,
    pub lat: f64,
}

/// Load features from an input file
///
/// Supports:
/// - GeoJSON (.geojson, .json)
/// - CSV (.csv) with lon/lat columns
/// - GeoParquet (.parquet, .geoparquet) - requires `geoparquet` feature
pub fn load_features(
    path: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: &str,
) -> Result<Vec<ParsedFeature>> {
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    match extension.to_lowercase().as_str() {
        "geojson" | "json" => load_geojson(path, time_field, end_time_field, time_format),
        "csv" => load_csv(path, time_field, end_time_field, time_format),
        #[cfg(feature = "geoparquet")]
        "parquet" | "geoparquet" => load_geoparquet(path, time_field, end_time_field, time_format),
        #[cfg(not(feature = "geoparquet"))]
        "parquet" | "geoparquet" => anyhow::bail!(
            "GeoParquet support requires the 'geoparquet' feature. \
             Rebuild with: cargo build --features geoparquet"
        ),
        _ => anyhow::bail!("Unsupported file format: {}", extension),
    }
}

/// Load GeoJSON file
fn load_geojson(path: &Path, time_field: &str, end_time_field: Option<&str>, time_format: &str) -> Result<Vec<ParsedFeature>> {
    let file = File::open(path).context("Failed to open input file")?;
    let reader = BufReader::new(file);
    let geojson: GeoJson = serde_json::from_reader(reader).context("Failed to parse GeoJSON")?;

    let features = match geojson {
        GeoJson::FeatureCollection(fc) => fc.features,
        GeoJson::Feature(f) => vec![f],
        _ => anyhow::bail!("Expected FeatureCollection or Feature"),
    };

    let mut parsed = Vec::new();

    for feature in features {
        if let Some(parsed_feature) = parse_feature(feature, time_field, end_time_field, time_format)? {
            parsed.push(parsed_feature);
        }
    }

    Ok(parsed)
}

/// Load CSV file
///
/// Now uses the standard `csv` crate for robust parsing.
/// Expects columns: lon, lat, timestamp, and optional property columns.
fn load_csv(path: &Path, time_field: &str, end_time_field: Option<&str>, time_format: &str) -> Result<Vec<ParsedFeature>> {
    use csv::ReaderBuilder;
    use geojson::{Geometry, Value as GeomValue};

    let file = File::open(path).context("Failed to open CSV file")?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let headers = reader
        .headers()
        .context("Failed to read CSV headers")?
        .clone();

    // Find required columns
    let lon_idx = find_column(&headers, &["lon", "longitude", "lng", "x"]).ok_or_else(|| {
        anyhow::anyhow!("CSV must have a longitude column (lon, longitude, lng, or x)")
    })?;
    let lat_idx = find_column(&headers, &["lat", "latitude", "y"])
        .ok_or_else(|| anyhow::anyhow!("CSV must have a latitude column (lat, latitude, or y)"))?;
    let time_idx = find_column(&headers, &[time_field])
        .ok_or_else(|| anyhow::anyhow!("CSV must have a {} column", time_field))?;
    
    // Find optional end time column
    let end_time_idx = end_time_field.and_then(|field| find_column(&headers, &[field]));

    let mut parsed = Vec::new();

    for result in reader.records() {
        let record = result.context("Failed to read CSV record")?;

        // Parse coordinates
        let lon: f64 = record
            .get(lon_idx)
            .ok_or_else(|| anyhow::anyhow!("Missing longitude value"))?
            .parse()
            .context("Failed to parse longitude")?;
        let lat: f64 = record
            .get(lat_idx)
            .ok_or_else(|| anyhow::anyhow!("Missing latitude value"))?
            .parse()
            .context("Failed to parse latitude")?;

        // Parse timestamp
        let time_str = record
            .get(time_idx)
            .ok_or_else(|| anyhow::anyhow!("Missing timestamp value"))?;
        let timestamp = parse_timestamp_str(time_str, time_format)?;
        
        // Parse optional end timestamp
        let end_timestamp = if let Some(end_idx) = end_time_idx {
            if let Some(end_time_str) = record.get(end_idx) {
                if !end_time_str.is_empty() {
                    Some(parse_timestamp_str(end_time_str, time_format)?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Build properties from other columns
        let mut properties = serde_json::Map::new();
        for (idx, header) in headers.iter().enumerate() {
            if idx != lon_idx && idx != lat_idx && idx != time_idx && Some(idx) != end_time_idx {
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
            end_timestamp,
            lon,
            lat,
        });
    }

    Ok(parsed)
}

/// Load GeoParquet file
///
/// Reads GeoParquet files using the geoarrow crate. Supports:
/// - Point geometries (directly from geometry column)
/// - LineString and Polygon geometries (extracts first coordinate)
/// - Temporal columns (timestamp, start_time, end_time, etc.)
///
/// The geometry column is auto-detected from GeoParquet metadata.
#[cfg(feature = "geoparquet")]
fn load_geoparquet(
    path: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: &str,
) -> Result<Vec<ParsedFeature>> {
    let file = File::open(path).context("Failed to open GeoParquet file")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader")?;
    
    let schema = builder.schema().clone();
    let reader = builder.build().context("Failed to build Parquet reader")?;

    // Find geometry column (look for common names or WKB column)
    let geom_col_name = find_geometry_column(&schema)?;
    
    // Find time columns
    let time_col_idx = schema.fields().iter().position(|f| f.name() == time_field)
        .ok_or_else(|| anyhow::anyhow!("Time field '{}' not found in Parquet schema", time_field))?;
    
    let end_time_col_idx = end_time_field
        .and_then(|field| schema.fields().iter().position(|f| f.name() == field));

    let mut parsed = Vec::new();

    for batch_result in reader {
        let batch = batch_result.context("Failed to read Parquet batch")?;
        
        // Extract coordinates from geometry column
        let coords = extract_coords_from_batch(&batch, &geom_col_name)?;
        
        // Extract timestamps
        let timestamps = extract_timestamps_from_batch(&batch, time_col_idx, time_format)?;
        let end_timestamps = end_time_col_idx
            .map(|idx| extract_timestamps_from_batch(&batch, idx, time_format))
            .transpose()?;

        // Build features
        for i in 0..batch.num_rows() {
            let (lon, lat) = coords.get(i).copied()
                .ok_or_else(|| anyhow::anyhow!("Missing coordinates at row {}", i))?;
            let timestamp = timestamps.get(i).copied()
                .ok_or_else(|| anyhow::anyhow!("Missing timestamp at row {}", i))?;
            let end_timestamp = end_timestamps.as_ref().and_then(|ts| ts.get(i).copied());

            // Build properties from other columns
            let mut properties = serde_json::Map::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col_name = field.name();
                // Skip geometry and time columns
                if col_name == &geom_col_name 
                    || col_name == time_field 
                    || end_time_field.map(|f| col_name == f).unwrap_or(false) 
                {
                    continue;
                }
                
                if let Some(value) = extract_property_value(&batch, col_idx, i) {
                    properties.insert(col_name.clone(), value);
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
                end_timestamp,
                lon,
                lat,
            });
        }
    }

    Ok(parsed)
}

/// Find the geometry column in a Parquet schema
#[cfg(feature = "geoparquet")]
fn find_geometry_column(schema: &arrow::datatypes::Schema) -> Result<String> {
    // Common geometry column names
    let common_names = ["geometry", "geom", "wkb_geometry", "the_geom", "shape"];
    
    for name in common_names {
        if schema.field_with_name(name).is_ok() {
            return Ok(name.to_string());
        }
    }
    
    // Look for binary columns that might contain WKB
    for field in schema.fields() {
        if matches!(field.data_type(), DataType::Binary | DataType::LargeBinary) {
            return Ok(field.name().clone());
        }
    }
    
    // Look for struct columns (GeoArrow native encoding)
    for field in schema.fields() {
        if matches!(field.data_type(), DataType::Struct(_)) {
            return Ok(field.name().clone());
        }
    }
    
    anyhow::bail!("Could not find geometry column in Parquet schema. Expected columns: {:?}", common_names)
}

/// Extract coordinates from a record batch's geometry column
#[cfg(feature = "geoparquet")]
fn extract_coords_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    geom_col_name: &str,
) -> Result<Vec<(f64, f64)>> {
    let geom_col = batch.column_by_name(geom_col_name)
        .ok_or_else(|| anyhow::anyhow!("Geometry column '{}' not found", geom_col_name))?;
    
    let mut coords = Vec::with_capacity(batch.num_rows());
    
    // Try to interpret as GeoArrow point array (struct with x, y fields)
    if let Some(struct_array) = geom_col.as_any().downcast_ref::<arrow::array::StructArray>() {
        // GeoArrow native point encoding: struct<x: f64, y: f64>
        let x_col = struct_array.column_by_name("x")
            .or_else(|| struct_array.column_by_name("longitude"))
            .or_else(|| struct_array.column_by_name("lon"));
        let y_col = struct_array.column_by_name("y")
            .or_else(|| struct_array.column_by_name("latitude"))
            .or_else(|| struct_array.column_by_name("lat"));
        
        if let (Some(x), Some(y)) = (x_col, y_col) {
            if let (Some(x_arr), Some(y_arr)) = (
                x.as_any().downcast_ref::<Float64Array>(),
                y.as_any().downcast_ref::<Float64Array>(),
            ) {
                for i in 0..batch.num_rows() {
                    if x_arr.is_valid(i) && y_arr.is_valid(i) {
                        coords.push((x_arr.value(i), y_arr.value(i)));
                    } else {
                        coords.push((0.0, 0.0));
                    }
                }
                return Ok(coords);
            }
        }
    }
    
    // Try WKB binary column
    if let Some(binary_array) = geom_col.as_any().downcast_ref::<arrow::array::BinaryArray>() {
        for i in 0..batch.num_rows() {
            if binary_array.is_valid(i) {
                let wkb = binary_array.value(i);
                let (lon, lat) = parse_wkb_point(wkb).unwrap_or((0.0, 0.0));
                coords.push((lon, lat));
            } else {
                coords.push((0.0, 0.0));
            }
        }
        return Ok(coords);
    }
    
    // Fallback: look for separate lon/lat columns
    let lon_col = batch.column_by_name("lon")
        .or_else(|| batch.column_by_name("longitude"))
        .or_else(|| batch.column_by_name("x"));
    let lat_col = batch.column_by_name("lat")
        .or_else(|| batch.column_by_name("latitude"))
        .or_else(|| batch.column_by_name("y"));
    
    if let (Some(lon), Some(lat)) = (lon_col, lat_col) {
        if let (Some(lon_arr), Some(lat_arr)) = (
            lon.as_any().downcast_ref::<Float64Array>(),
            lat.as_any().downcast_ref::<Float64Array>(),
        ) {
            for i in 0..batch.num_rows() {
                if lon_arr.is_valid(i) && lat_arr.is_valid(i) {
                    coords.push((lon_arr.value(i), lat_arr.value(i)));
                } else {
                    coords.push((0.0, 0.0));
                }
            }
            return Ok(coords);
        }
    }
    
    anyhow::bail!("Could not extract coordinates from geometry column '{}'", geom_col_name)
}

/// Parse WKB Point geometry (simplified - only handles Point type)
#[cfg(feature = "geoparquet")]
fn parse_wkb_point(wkb: &[u8]) -> Option<(f64, f64)> {
    if wkb.len() < 21 {
        return None;
    }
    
    // Byte order (1 = little endian, 0 = big endian)
    let little_endian = wkb[0] == 1;
    
    // Geometry type (1 = Point, handle 2D and 3D variants)
    let geom_type = if little_endian {
        u32::from_le_bytes([wkb[1], wkb[2], wkb[3], wkb[4]])
    } else {
        u32::from_be_bytes([wkb[1], wkb[2], wkb[3], wkb[4]])
    };
    
    // Point types: 1 (2D), 1001 (Z), 2001 (M), 3001 (ZM)
    if ![1, 1001, 2001, 3001].contains(&geom_type) {
        // Not a point - for lines/polygons, we'd need to extract first coord
        return None;
    }
    
    // Extract X and Y coordinates
    let x = if little_endian {
        f64::from_le_bytes([wkb[5], wkb[6], wkb[7], wkb[8], wkb[9], wkb[10], wkb[11], wkb[12]])
    } else {
        f64::from_be_bytes([wkb[5], wkb[6], wkb[7], wkb[8], wkb[9], wkb[10], wkb[11], wkb[12]])
    };
    
    let y = if little_endian {
        f64::from_le_bytes([wkb[13], wkb[14], wkb[15], wkb[16], wkb[17], wkb[18], wkb[19], wkb[20]])
    } else {
        f64::from_be_bytes([wkb[13], wkb[14], wkb[15], wkb[16], wkb[17], wkb[18], wkb[19], wkb[20]])
    };
    
    Some((x, y))
}

/// Extract timestamps from a column
#[cfg(feature = "geoparquet")]
fn extract_timestamps_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
    time_format: &str,
) -> Result<Vec<u64>> {
    let column = batch.column(col_idx);
    let mut timestamps = Vec::with_capacity(batch.num_rows());
    
    // Try as timestamp array
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampMillisecondArray>() {
        for i in 0..batch.num_rows() {
            if ts_array.is_valid(i) {
                timestamps.push(ts_array.value(i) as u64);
            } else {
                timestamps.push(0);
            }
        }
        return Ok(timestamps);
    }
    
    // Try as i64 array (unix timestamp)
    if let Some(int_array) = column.as_any().downcast_ref::<Int64Array>() {
        for i in 0..batch.num_rows() {
            if int_array.is_valid(i) {
                let value = int_array.value(i) as u64;
                let ts = match time_format {
                    "unix-sec" => value * 1000,
                    _ => value, // Assume unix-ms
                };
                timestamps.push(ts);
            } else {
                timestamps.push(0);
            }
        }
        return Ok(timestamps);
    }
    
    // Try as string array (ISO8601)
    if let Some(str_array) = column.as_any().downcast_ref::<StringArray>() {
        for i in 0..batch.num_rows() {
            if str_array.is_valid(i) {
                let s = str_array.value(i);
                let ts = parse_iso8601(s).unwrap_or(0);
                timestamps.push(ts);
            } else {
                timestamps.push(0);
            }
        }
        return Ok(timestamps);
    }
    
    anyhow::bail!("Unsupported timestamp column type")
}

/// Extract a property value from a column at a given row
#[cfg(feature = "geoparquet")]
fn extract_property_value(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
    row_idx: usize,
) -> Option<serde_json::Value> {
    let column = batch.column(col_idx);
    
    if !column.is_valid(row_idx) {
        return None;
    }
    
    // Try various types
    if let Some(arr) = column.as_any().downcast_ref::<Float64Array>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    if let Some(arr) = column.as_any().downcast_ref::<Int64Array>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    if let Some(arr) = column.as_any().downcast_ref::<StringArray>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::BooleanArray>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::Float32Array>() {
        return Some(serde_json::json!(arr.value(row_idx) as f64));
    }
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::Int32Array>() {
        return Some(serde_json::json!(arr.value(row_idx) as i64));
    }
    
    None
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
            let ms = s
                .parse::<u64>()
                .context("Expected numeric timestamp in milliseconds")?;
            Ok(ms)
        }
        "unix-sec" => {
            let sec = s
                .parse::<u64>()
                .context("Expected numeric timestamp in seconds")?;
            Ok(sec * 1000)
        }
        "iso8601" => parse_iso8601(s),
        _ => anyhow::bail!("Unknown time format: {}", format),
    }
}

/// Parse a single GeoJSON feature
fn parse_feature(
    feature: Feature,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: &str,
) -> Result<Option<ParsedFeature>> {
    // Extract timestamp from properties
    let (timestamp, end_timestamp) = match feature.properties.as_ref() {
        Some(props) => {
            let value = props.get(time_field).context(format!(
                "Time field '{}' not found in feature properties",
                time_field
            ))?;
            let start = parse_timestamp(value, time_format)?;
            
            // Extract optional end timestamp
            let end = if let Some(end_field) = end_time_field {
                props.get(end_field)
                    .map(|v| parse_timestamp(v, time_format))
                    .transpose()?
            } else {
                None
            };
            
            (start, end)
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
        end_timestamp,
        lon,
        lat,
    }))
}

/// Parse timestamp from a JSON value
fn parse_timestamp(value: &Value, format: &str) -> Result<u64> {
    match format {
        "unix-ms" => {
            let ms = value
                .as_u64()
                .or_else(|| value.as_i64().map(|i| i as u64))
                .context("Expected numeric timestamp")?;
            Ok(ms)
        }
        "unix-sec" => {
            let sec = value
                .as_u64()
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
