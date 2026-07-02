//! Data loading for GeoParquet files
//!
//! Provides unified data loading from different sources for analysis.

use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, Int64Array, StringArray, TimestampSecondArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::path::{Path, PathBuf};
use stt_core::timestamp::{normalize_timestamp_to_ms, TimestampUnit};
use stt_core::types::{BoundingBox, TimeRange};

/// Data source specification
#[derive(Debug, Clone)]
pub enum DataSource {
    GeoParquet {
        path: PathBuf,
        time_field: String,
        time_format: String,
    },
}

impl DataSource {
    pub fn display_name(&self) -> String {
        match self {
            DataSource::GeoParquet { path, .. } => {
                path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            }
        }
    }
}

/// A loaded feature for analysis
#[derive(Debug, Clone)]
pub struct AnalyzableFeature {
    /// Longitude (centroid for complex geometries)
    pub lon: f64,
    /// Latitude (centroid for complex geometries)
    pub lat: f64,
    /// Timestamp in milliseconds
    pub timestamp: u64,
    /// Geometry type
    pub geometry_type: GeometryType,
    /// Number of vertices in the geometry
    pub vertex_count: usize,
    /// Estimated serialized size in bytes
    pub estimated_size: usize,
    /// Number of properties
    pub property_count: usize,
}

/// Geometry type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryType {
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    Unknown,
}

impl std::fmt::Display for GeometryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryType::Point => write!(f, "Point"),
            GeometryType::LineString => write!(f, "LineString"),
            GeometryType::Polygon => write!(f, "Polygon"),
            GeometryType::MultiPoint => write!(f, "MultiPoint"),
            GeometryType::MultiLineString => write!(f, "MultiLineString"),
            GeometryType::MultiPolygon => write!(f, "MultiPolygon"),
            GeometryType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Loaded dataset for analysis
#[derive(Debug)]
pub struct LoadedData {
    pub features: Vec<AnalyzableFeature>,
    pub bounds: BoundingBox,
    pub time_range: TimeRange,
}

/// Load data from a data source
pub fn load_data(source: &DataSource) -> Result<LoadedData> {
    match source {
        DataSource::GeoParquet { path, time_field, time_format } => {
            load_geoparquet(path, time_field, time_format)
        }
    }
}

/// Load features from a GeoParquet file
fn load_geoparquet(path: &Path, time_field: &str, time_format: &str) -> Result<LoadedData> {
    use indicatif::{ProgressBar, ProgressStyle};

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message("Loading GeoParquet file...");

    let file = std::fs::File::open(path).context("Failed to open GeoParquet file")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    // Find geometry and time columns
    let geom_col_name = find_geometry_column(&schema)?;
    let time_col_idx = schema.fields().iter().position(|f| f.name() == time_field)
        .ok_or_else(|| anyhow::anyhow!("Time field '{}' not found", time_field))?;

    let reader = builder.build()?;

    let mut features = Vec::new();
    let mut min_lon = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let mut min_time = u64::MAX;
    let mut max_time = u64::MIN;

    for batch_result in reader {
        let batch = batch_result.context("Failed to read Parquet batch")?;

        let geometries = extract_geometries_from_batch(&batch, &geom_col_name)?;
        let timestamps = extract_timestamps_from_batch(&batch, time_col_idx, time_format)?;

        // Count property columns
        let property_count = schema.fields().len() - 2; // Exclude geometry and time

        for i in 0..batch.num_rows() {
            let (geom_type, vertex_count, lon, lat) = geometries.get(i)
                .cloned()
                .unwrap_or((GeometryType::Unknown, 0, 0.0, 0.0));
            let timestamp = timestamps.get(i).copied().unwrap_or(0);

            // Update bounds
            min_lon = min_lon.min(lon);
            max_lon = max_lon.max(lon);
            min_lat = min_lat.min(lat);
            max_lat = max_lat.max(lat);
            min_time = min_time.min(timestamp);
            max_time = max_time.max(timestamp);

            // Estimate size: base overhead + vertices + properties
            let estimated_size = 100 + (vertex_count * 16) + (property_count * 20);

            features.push(AnalyzableFeature {
                lon,
                lat,
                timestamp,
                geometry_type: geom_type,
                vertex_count,
                estimated_size,
                property_count,
            });
        }

        if features.len() % 100_000 == 0 {
            pb.set_message(format!("Loaded {} features...", features.len()));
        }
    }

    pb.finish_with_message(format!("Loaded {} features", features.len()));

    Ok(LoadedData {
        features,
        bounds: BoundingBox::new(min_lon, min_lat, max_lon, max_lat),
        time_range: TimeRange::new(min_time, max_time),
    })
}

/// Load features from an STT archive
// =============================================================================
// Helper Functions
// =============================================================================

/// Find the geometry column in a Parquet schema
fn find_geometry_column(schema: &arrow::datatypes::Schema) -> Result<String> {
    let common_names = ["geometry", "geom", "wkb_geometry", "the_geom", "shape"];

    for name in common_names {
        if schema.field_with_name(name).is_ok() {
            return Ok(name.to_string());
        }
    }

    // Look for binary columns (WKB)
    for field in schema.fields() {
        if matches!(field.data_type(), DataType::Binary | DataType::LargeBinary) {
            return Ok(field.name().clone());
        }
    }

    // Look for struct columns (GeoArrow)
    for field in schema.fields() {
        if matches!(field.data_type(), DataType::Struct(_)) {
            return Ok(field.name().clone());
        }
    }

    // Check for separate lon/lat columns
    let has_lon = schema.field_with_name("lon").is_ok()
        || schema.field_with_name("longitude").is_ok()
        || schema.field_with_name("x").is_ok();
    let has_lat = schema.field_with_name("lat").is_ok()
        || schema.field_with_name("latitude").is_ok()
        || schema.field_with_name("y").is_ok();

    if has_lon && has_lat {
        return Ok("__lon_lat__".to_string());
    }

    anyhow::bail!("Could not find geometry column in Parquet schema")
}

/// Extract geometries from a batch
fn extract_geometries_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    geom_col_name: &str,
) -> Result<Vec<(GeometryType, usize, f64, f64)>> {
    let mut results = Vec::with_capacity(batch.num_rows());

    // Handle separate lon/lat columns
    if geom_col_name == "__lon_lat__" {
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
                        results.push((GeometryType::Point, 1, lon_arr.value(i), lat_arr.value(i)));
                    } else {
                        results.push((GeometryType::Unknown, 0, 0.0, 0.0));
                    }
                }
                return Ok(results);
            }
        }
        anyhow::bail!("Expected lon/lat columns but could not read them");
    }

    let geom_col = batch.column_by_name(geom_col_name)
        .ok_or_else(|| anyhow::anyhow!("Geometry column '{}' not found", geom_col_name))?;

    // Try GeoArrow struct
    if let Some(struct_array) = geom_col.as_any().downcast_ref::<arrow::array::StructArray>() {
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
                        results.push((GeometryType::Point, 1, x_arr.value(i), y_arr.value(i)));
                    } else {
                        results.push((GeometryType::Unknown, 0, 0.0, 0.0));
                    }
                }
                return Ok(results);
            }
        }
    }

    // Try WKB binary column
    if let Some(binary_array) = geom_col.as_any().downcast_ref::<arrow::array::BinaryArray>() {
        for i in 0..batch.num_rows() {
            if binary_array.is_valid(i) {
                let wkb = binary_array.value(i);
                if let Some((geom_type, vertex_count, lon, lat)) = parse_wkb_info(wkb) {
                    results.push((geom_type, vertex_count, lon, lat));
                } else {
                    results.push((GeometryType::Unknown, 0, 0.0, 0.0));
                }
            } else {
                results.push((GeometryType::Unknown, 0, 0.0, 0.0));
            }
        }
        return Ok(results);
    }

    // Fallback: try separate lon/lat columns
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
                    results.push((GeometryType::Point, 1, lon_arr.value(i), lat_arr.value(i)));
                } else {
                    results.push((GeometryType::Unknown, 0, 0.0, 0.0));
                }
            }
            return Ok(results);
        }
    }

    anyhow::bail!("Could not extract geometries from column '{}'", geom_col_name)
}

/// Extract timestamps from a column
fn extract_timestamps_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
    time_format: &str,
) -> Result<Vec<u64>> {
    let column = batch.column(col_idx);
    let mut timestamps = Vec::with_capacity(batch.num_rows());

    // Timestamp-unit scaling routes through the shared `normalize_timestamp_to_ms`
    // (stt_core::timestamp) so this analysis loader agrees byte-for-byte with the
    // stt-build scalar/vertex readers and the DuckDB reader — the divergent local
    // `.max(0)`/hand-rolled ÷ arithmetic here was the audited bug. Null entries
    // still coerce to 0 (this loader's historical tolerance); a pre-1970 or
    // second→ms-overflowing value now hard-errors like the build path.
    macro_rules! push_ts_column {
        ($arr:expr, $unit:expr) => {{
            for i in 0..batch.num_rows() {
                if $arr.is_valid(i) {
                    timestamps.push(normalize_timestamp_to_ms(i, $arr.value(i), $unit)?);
                } else {
                    timestamps.push(0);
                }
            }
            return Ok(timestamps);
        }};
    }

    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampSecondArray>() {
        push_ts_column!(ts_array, TimestampUnit::Second);
    }
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampMillisecondArray>() {
        push_ts_column!(ts_array, TimestampUnit::Millisecond);
    }
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        push_ts_column!(ts_array, TimestampUnit::Microsecond);
    }
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampNanosecondArray>() {
        push_ts_column!(ts_array, TimestampUnit::Nanosecond);
    }

    // Try as i64 array (unix timestamp), interpreted per `--time-format`.
    if let Some(int_array) = column.as_any().downcast_ref::<Int64Array>() {
        let unit = match time_format {
            "unix-sec" => TimestampUnit::Second,
            _ => TimestampUnit::Millisecond,
        };
        for i in 0..batch.num_rows() {
            if int_array.is_valid(i) {
                timestamps.push(normalize_timestamp_to_ms(i, int_array.value(i), unit)?);
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

/// Parse ISO 8601 timestamp to Unix milliseconds
fn parse_iso8601(s: &str) -> Result<u64> {
    use chrono::{DateTime, NaiveDateTime};

    if let Ok(dt) = s.parse::<DateTime<chrono::Utc>>() {
        return Ok(dt.timestamp_millis() as u64);
    }

    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt.and_utc().timestamp_millis() as u64);
    }

    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        return Ok(dt.timestamp_millis() as u64);
    }

    anyhow::bail!("Failed to parse timestamp: {}", s)
}

// =============================================================================
// WKB Parsing
// =============================================================================

const WKB_POINT: u32 = 1;
const WKB_LINESTRING: u32 = 2;
const WKB_POLYGON: u32 = 3;
const WKB_MULTIPOINT: u32 = 4;
const WKB_MULTILINESTRING: u32 = 5;
const WKB_MULTIPOLYGON: u32 = 6;

fn parse_wkb_info(wkb: &[u8]) -> Option<(GeometryType, usize, f64, f64)> {
    if wkb.len() < 5 {
        return None;
    }

    let little_endian = wkb[0] == 1;
    let geom_type = if little_endian {
        u32::from_le_bytes([wkb[1], wkb[2], wkb[3], wkb[4]]) % 1000
    } else {
        u32::from_be_bytes([wkb[1], wkb[2], wkb[3], wkb[4]]) % 1000
    };

    match geom_type {
        WKB_POINT => parse_wkb_point_info(wkb, little_endian),
        WKB_LINESTRING => parse_wkb_linestring_info(wkb, little_endian),
        WKB_POLYGON => parse_wkb_polygon_info(wkb, little_endian),
        WKB_MULTIPOINT => parse_wkb_multipoint_info(wkb, little_endian),
        WKB_MULTILINESTRING => parse_wkb_multilinestring_info(wkb, little_endian),
        WKB_MULTIPOLYGON => parse_wkb_multipolygon_info(wkb, little_endian),
        _ => None,
    }
}

fn read_f64(wkb: &[u8], offset: usize, little_endian: bool) -> Option<f64> {
    if offset + 8 > wkb.len() {
        return None;
    }
    let bytes: [u8; 8] = wkb[offset..offset+8].try_into().ok()?;
    Some(if little_endian {
        f64::from_le_bytes(bytes)
    } else {
        f64::from_be_bytes(bytes)
    })
}

fn read_u32(wkb: &[u8], offset: usize, little_endian: bool) -> Option<u32> {
    if offset + 4 > wkb.len() {
        return None;
    }
    let bytes: [u8; 4] = wkb[offset..offset+4].try_into().ok()?;
    Some(if little_endian {
        u32::from_le_bytes(bytes)
    } else {
        u32::from_be_bytes(bytes)
    })
}

fn parse_wkb_point_info(wkb: &[u8], little_endian: bool) -> Option<(GeometryType, usize, f64, f64)> {
    let x = read_f64(wkb, 5, little_endian)?;
    let y = read_f64(wkb, 13, little_endian)?;
    Some((GeometryType::Point, 1, x, y))
}

fn parse_wkb_linestring_info(wkb: &[u8], little_endian: bool) -> Option<(GeometryType, usize, f64, f64)> {
    let num_points = read_u32(wkb, 5, little_endian)? as usize;
    if num_points == 0 {
        return None;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut offset = 9;

    for _ in 0..num_points {
        let x = read_f64(wkb, offset, little_endian)?;
        let y = read_f64(wkb, offset + 8, little_endian)?;
        sum_x += x;
        sum_y += y;
        offset += 16;
    }

    Some((GeometryType::LineString, num_points, sum_x / num_points as f64, sum_y / num_points as f64))
}

fn parse_wkb_polygon_info(wkb: &[u8], little_endian: bool) -> Option<(GeometryType, usize, f64, f64)> {
    let num_rings = read_u32(wkb, 5, little_endian)? as usize;
    if num_rings == 0 {
        return None;
    }

    let mut total_points = 0usize;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut offset = 9;

    for ring_idx in 0..num_rings {
        let num_points = read_u32(wkb, offset, little_endian)? as usize;
        offset += 4;

        for _ in 0..num_points {
            let x = read_f64(wkb, offset, little_endian)?;
            let y = read_f64(wkb, offset + 8, little_endian)?;
            
            if ring_idx == 0 {
                sum_x += x;
                sum_y += y;
                total_points += 1;
            }
            
            offset += 16;
        }
    }

    let vertex_count: usize = total_points;
    let centroid_x = if total_points > 0 { sum_x / total_points as f64 } else { 0.0 };
    let centroid_y = if total_points > 0 { sum_y / total_points as f64 } else { 0.0 };

    Some((GeometryType::Polygon, vertex_count, centroid_x, centroid_y))
}

fn parse_wkb_multipoint_info(wkb: &[u8], little_endian: bool) -> Option<(GeometryType, usize, f64, f64)> {
    let num_geoms = read_u32(wkb, 5, little_endian)? as usize;
    if num_geoms == 0 {
        return None;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut offset = 9;

    for _ in 0..num_geoms {
        offset += 5; // Skip WKB header
        let x = read_f64(wkb, offset, little_endian)?;
        let y = read_f64(wkb, offset + 8, little_endian)?;
        sum_x += x;
        sum_y += y;
        offset += 16;
    }

    Some((GeometryType::MultiPoint, num_geoms, sum_x / num_geoms as f64, sum_y / num_geoms as f64))
}

fn parse_wkb_multilinestring_info(wkb: &[u8], little_endian: bool) -> Option<(GeometryType, usize, f64, f64)> {
    let num_geoms = read_u32(wkb, 5, little_endian)? as usize;
    if num_geoms == 0 {
        return None;
    }

    let mut total_points = 0usize;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut offset = 9;

    for _ in 0..num_geoms {
        offset += 5; // Skip WKB header
        let num_points = read_u32(wkb, offset, little_endian)? as usize;
        offset += 4;

        for _ in 0..num_points {
            let x = read_f64(wkb, offset, little_endian)?;
            let y = read_f64(wkb, offset + 8, little_endian)?;
            sum_x += x;
            sum_y += y;
            total_points += 1;
            offset += 16;
        }
    }

    let centroid_x = if total_points > 0 { sum_x / total_points as f64 } else { 0.0 };
    let centroid_y = if total_points > 0 { sum_y / total_points as f64 } else { 0.0 };

    Some((GeometryType::MultiLineString, total_points, centroid_x, centroid_y))
}

fn parse_wkb_multipolygon_info(wkb: &[u8], little_endian: bool) -> Option<(GeometryType, usize, f64, f64)> {
    let num_geoms = read_u32(wkb, 5, little_endian)? as usize;
    if num_geoms == 0 {
        return None;
    }

    let mut total_points = 0usize;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut offset = 9;

    for poly_idx in 0..num_geoms {
        offset += 5; // Skip WKB header
        let num_rings = read_u32(wkb, offset, little_endian)? as usize;
        offset += 4;

        for ring_idx in 0..num_rings {
            let num_points = read_u32(wkb, offset, little_endian)? as usize;
            offset += 4;

            for _ in 0..num_points {
                let x = read_f64(wkb, offset, little_endian)?;
                let y = read_f64(wkb, offset + 8, little_endian)?;
                
                if poly_idx == 0 && ring_idx == 0 {
                    sum_x += x;
                    sum_y += y;
                    total_points += 1;
                }
                
                offset += 16;
            }
        }
    }

    let centroid_x = if total_points > 0 { sum_x / total_points as f64 } else { 0.0 };
    let centroid_y = if total_points > 0 { sum_y / total_points as f64 } else { 0.0 };

    Some((GeometryType::MultiPolygon, total_points, centroid_x, centroid_y))
}

