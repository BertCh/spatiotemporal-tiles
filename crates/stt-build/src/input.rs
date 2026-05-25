//! GeoParquet input parsing and feature loading

use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, Int64Array, ListArray, StringArray, TimestampMillisecondArray, TimestampMicrosecondArray};
use arrow::datatypes::DataType;
use geojson::{Feature, Geometry, Value as GeomValue};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use stt_core::types::{BoundingBox, TimeRange};

/// Shared properties map to avoid cloning per-segment
pub type SharedProperties = Arc<serde_json::Map<String, serde_json::Value>>;

/// Parsed feature with geometry and temporal information
#[derive(Debug, Clone)]
pub struct ParsedFeature {
    pub geojson: Feature,
    /// Shared properties reference (avoids cloning during clipping)
    pub shared_properties: Option<SharedProperties>,
    pub timestamp: u64,
    /// End timestamp for features with time ranges (if provided)
    pub end_timestamp: Option<u64>,
    /// Optional per-vertex absolute Unix-ms timestamps for LineString
    /// geometries. When present, the line-layer builder uses these directly
    /// instead of interpolating uniformly by distance — lets the producer
    /// pass real per-segment timing (e.g. OSRM `annotations=duration`)
    /// through to the GPU.
    ///
    /// MUST be the same length as the geometry's coord count when set;
    /// length-mismatched values are dropped at the reader (logged once).
    pub vertex_timestamps: Option<Vec<u64>>,
    pub lon: f64,
    pub lat: f64,
}

/// Load all features from a GeoParquet file into memory.
///
/// This is the simple in-memory approach that works well for datasets
/// that fit in RAM (up to several million features on a typical machine).
/// Strictness for input parsing. `Warn` keeps the legacy "coerce bad
/// timestamps to epoch 0 and log" behaviour; `Strict` aborts the build on
/// the first parse failure with a row-counted error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputStrictness {
    Warn,
    Strict,
}

pub fn load_features(
    path: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: &str,
    strictness: InputStrictness,
) -> Result<Vec<ParsedFeature>> {
    let mut features = Vec::new();
    let mut row_count = 0usize;
    stream_features(path, time_field, end_time_field, time_format, strictness, |batch| {
        row_count += batch.len();
        features.extend(batch);
        if row_count.is_multiple_of(100_000) {
            tracing::info!("Loaded {} features...", row_count);
        }
        Ok(())
    })?;
    tracing::info!("Loaded {} total features", features.len());
    Ok(features)
}

/// Stream a GeoParquet input one record batch at a time, invoking
/// `on_batch` with the materialised `ParsedFeature`s for that batch.
///
/// Peak memory is bounded by one Parquet batch (typically 1–8k rows) rather
/// than the entire input. This is the entry point for the streaming build
/// pipeline; downstream tilers must not retain references into prior
/// batches across calls.
///
/// `on_batch` may consume or discard the batch; the function returns when
/// the input is exhausted or when `on_batch` returns an error.
pub fn stream_features<F>(
    path: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: &str,
    strictness: InputStrictness,
    mut on_batch: F,
) -> Result<()>
where
    F: FnMut(Vec<ParsedFeature>) -> Result<()>,
{
    let file = File::open(path).context("Failed to open GeoParquet file")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    let geom_col_name = find_geometry_column(&schema)?;
    let time_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == time_field)
        .ok_or_else(|| anyhow::anyhow!("Time field '{}' not found", time_field))?;
    let end_time_col_idx = end_time_field
        .and_then(|field| schema.fields().iter().position(|f| f.name() == field));

    // Optional per-vertex timestamps column (List<Timestamp> or List<Int64>).
    // Producers that have real per-segment timing (e.g. nyc-rideshare with
    // OSRM annotations) populate this; absence falls back to legacy
    // uniform-by-distance interpolation in columnar.rs.
    let vertex_times_col_idx = schema.fields().iter().position(|f| f.name() == "vertex_timestamps");

    let reader = builder.build()?;

    // Property column indices computed once.
    let property_cols: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(idx, field)| {
            let name = field.name();
            let is_meta = name == &geom_col_name
                || name == time_field
                || end_time_field.map(|f| name == f).unwrap_or(false)
                || name == "vertex_timestamps"
                || matches!(name.as_str(), "lon" | "lat" | "longitude" | "latitude" | "x" | "y");
            if is_meta {
                None
            } else {
                Some(idx)
            }
        })
        .collect();

    let mut row_count = 0usize;
    for batch_result in reader {
        let batch = batch_result.context("Failed to read Parquet batch")?;
        let parsed = parse_batch(
            &batch,
            &schema,
            &geom_col_name,
            time_col_idx,
            end_time_col_idx,
            vertex_times_col_idx,
            &property_cols,
            time_format,
            strictness,
            row_count,
        )?;
        row_count += batch.num_rows();
        on_batch(parsed)?;
    }
    Ok(())
}

/// Materialise one record batch into a `Vec<ParsedFeature>` without holding
/// any other batches in memory. Pulled out of `stream_features` so the
/// streaming loop and the eager `load_features` share one definition.
#[allow(clippy::too_many_arguments)]
fn parse_batch(
    batch: &arrow::record_batch::RecordBatch,
    schema: &arrow::datatypes::Schema,
    geom_col_name: &str,
    time_col_idx: usize,
    end_time_col_idx: Option<usize>,
    vertex_times_col_idx: Option<usize>,
    property_cols: &[usize],
    time_format: &str,
    strictness: InputStrictness,
    row_offset: usize,
) -> Result<Vec<ParsedFeature>> {
    let geometries = extract_geometries_from_batch(batch, geom_col_name)?;
    let timestamps =
        extract_timestamps_from_batch(batch, time_col_idx, time_format, strictness)?;
    let end_timestamps = end_time_col_idx
        .map(|idx| extract_timestamps_from_batch(batch, idx, time_format, strictness))
        .transpose()?;
    let vertex_times = vertex_times_col_idx
        .map(|idx| extract_vertex_timestamps_from_batch(batch, idx))
        .transpose()?;

    let mut features = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let (geometry, lon, lat) = geometries
            .get(i)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing geometry at row {}", row_offset + i))?;
        let timestamp = timestamps
            .get(i)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Missing timestamp at row {}", row_offset + i))?;
        let end_timestamp = end_timestamps.as_ref().and_then(|ts| ts.get(i).copied());
        let row_vertex_times = vertex_times
            .as_ref()
            .and_then(|v| v.get(i).cloned().flatten());

        // Build properties from non-meta columns. We only materialise this
        // map when the row actually has property values; an empty map is
        // dropped so empty rows pay no allocation.
        let mut properties = serde_json::Map::new();
        for &col_idx in property_cols {
            if let Some(value) = extract_property_value(batch, col_idx, i) {
                let name = schema.field(col_idx).name();
                properties.insert(name.clone(), value);
            }
        }
        let shared_properties = if properties.is_empty() {
            None
        } else {
            Some(Arc::new(properties))
        };

        let feature = Feature {
            bbox: None,
            geometry: Some(geometry),
            id: None,
            properties: None,
            foreign_members: None,
        };
        features.push(ParsedFeature {
            geojson: feature,
            shared_properties,
            timestamp,
            end_timestamp,
            vertex_timestamps: row_vertex_times,
            lon,
            lat,
        });
    }
    Ok(features)
}

/// Extract optional per-row vertex-timestamp lists from a `vertex_timestamps`
/// column. Tolerates both `List<Timestamp(Millisecond)>` and `List<Int64>`
/// children (ms in either case). Null rows return `None` for that slot;
/// non-list columns return an error.
fn extract_vertex_timestamps_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
) -> Result<Vec<Option<Vec<u64>>>> {
    let column = batch.column(col_idx);
    let list = column
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow::anyhow!("vertex_timestamps column is not a List array"))?;

    let mut out: Vec<Option<Vec<u64>>> = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        if !list.is_valid(row) {
            out.push(None);
            continue;
        }
        let values = list.value(row);
        // The child can be either a TimestampMillisecondArray (preferred —
        // self-describing) or a plain Int64Array (fallback for callers that
        // wrote raw integers). Both store ms-since-epoch.
        let row_times: Vec<u64> = if let Some(ts) =
            values.as_any().downcast_ref::<TimestampMillisecondArray>()
        {
            (0..ts.len())
                .map(|i| if ts.is_valid(i) { ts.value(i) as u64 } else { 0 })
                .collect()
        } else if let Some(ints) = values.as_any().downcast_ref::<Int64Array>() {
            (0..ints.len())
                .map(|i| if ints.is_valid(i) { ints.value(i) as u64 } else { 0 })
                .collect()
        } else {
            anyhow::bail!(
                "vertex_timestamps child must be Timestamp(Millisecond) or Int64; got {:?}",
                values.data_type()
            );
        };
        out.push(Some(row_times));
    }
    Ok(out)
}

/// Calculate spatial and temporal bounds from features
pub fn calculate_bounds(features: &[ParsedFeature]) -> Result<(BoundingBox, TimeRange)> {
    if features.is_empty() {
        return Ok((
            BoundingBox::new(-180.0, -90.0, 180.0, 90.0),
            TimeRange::new(0, 0),
        ));
    }

    let mut min_lon = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let mut min_time = u64::MAX;
    let mut max_time = u64::MIN;

    for f in features {
        min_lon = min_lon.min(f.lon);
        max_lon = max_lon.max(f.lon);
        min_lat = min_lat.min(f.lat);
        max_lat = max_lat.max(f.lat);
        min_time = min_time.min(f.timestamp);
        max_time = max_time.max(f.end_timestamp.unwrap_or(f.timestamp));
    }

    Ok((
        BoundingBox::new(min_lon, min_lat, max_lon, max_lat),
        TimeRange::new(min_time, max_time),
    ))
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Find the geometry column in a Parquet schema
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
    
    anyhow::bail!("Could not find geometry column in Parquet schema. Expected columns: {:?}", common_names)
}

/// Extract geometries from a batch
fn extract_geometries_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    geom_col_name: &str,
) -> Result<Vec<(Geometry, f64, f64)>> {
    let mut geometries = Vec::with_capacity(batch.num_rows());
    
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
                        let x = lon_arr.value(i);
                        let y = lat_arr.value(i);
                        geometries.push((
                            Geometry::new(GeomValue::Point(vec![x, y])),
                            x, y
                        ));
                    } else {
                        geometries.push((
                            Geometry::new(GeomValue::Point(vec![0.0, 0.0])),
                            0.0, 0.0
                        ));
                    }
                }
                return Ok(geometries);
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
                        let x = x_arr.value(i);
                        let y = y_arr.value(i);
                        geometries.push((
                            Geometry::new(GeomValue::Point(vec![x, y])),
                            x, y
                        ));
                    } else {
                        geometries.push((
                            Geometry::new(GeomValue::Point(vec![0.0, 0.0])),
                            0.0, 0.0
                        ));
                    }
                }
                return Ok(geometries);
            }
        }
    }
    
    // Try WKB binary column
    if let Some(binary_array) = geom_col.as_any().downcast_ref::<arrow::array::BinaryArray>() {
        for i in 0..batch.num_rows() {
            if binary_array.is_valid(i) {
                let wkb = binary_array.value(i);
                if let Some((geom, lon, lat)) = parse_wkb_geometry(wkb) {
                    geometries.push((geom, lon, lat));
                } else {
                    geometries.push((
                        Geometry::new(GeomValue::Point(vec![0.0, 0.0])),
                        0.0, 0.0
                    ));
                }
            } else {
                geometries.push((
                    Geometry::new(GeomValue::Point(vec![0.0, 0.0])),
                    0.0, 0.0
                ));
            }
        }
        return Ok(geometries);
    }
    
    // Fallback: separate lon/lat columns
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
                    let x = lon_arr.value(i);
                    let y = lat_arr.value(i);
                    geometries.push((
                        Geometry::new(GeomValue::Point(vec![x, y])),
                        x, y
                    ));
                } else {
                    geometries.push((
                        Geometry::new(GeomValue::Point(vec![0.0, 0.0])),
                        0.0, 0.0
                    ));
                }
            }
            return Ok(geometries);
        }
    }
    
    anyhow::bail!("Could not extract geometries from column '{}'", geom_col_name)
}

/// Extract timestamps from a column
fn extract_timestamps_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
    time_format: &str,
    strictness: InputStrictness,
) -> Result<Vec<u64>> {
    let column = batch.column(col_idx);
    let mut timestamps = Vec::with_capacity(batch.num_rows());

    // Count rows whose timestamp could not be parsed (null or invalid).
    // In Warn mode they are coerced to Unix epoch 0 and we log; in Strict
    // mode we fail the build on the first bad row with row context.
    let mut parse_failures: usize = 0;
    let record_failure =
        |row: usize, parse_failures: &mut usize, reason: &str| -> Result<u64> {
            *parse_failures += 1;
            if strictness == InputStrictness::Strict {
                anyhow::bail!(
                    "row {row}: {reason} (rerun without --strict-times to coerce to epoch 0)"
                );
            }
            Ok(0)
        };

    // Try as timestamp array (milliseconds)
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampMillisecondArray>() {
        for i in 0..batch.num_rows() {
            if ts_array.is_valid(i) {
                timestamps.push(ts_array.value(i) as u64);
            } else {
                timestamps.push(record_failure(i, &mut parse_failures, "null timestamp")?);
            }
        }
        warn_timestamp_failures(parse_failures, batch.num_rows());
        return Ok(timestamps);
    }

    // Try as timestamp array (microseconds) - convert to milliseconds
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        for i in 0..batch.num_rows() {
            if ts_array.is_valid(i) {
                timestamps.push((ts_array.value(i) / 1000) as u64);
            } else {
                timestamps.push(record_failure(i, &mut parse_failures, "null timestamp")?);
            }
        }
        warn_timestamp_failures(parse_failures, batch.num_rows());
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
                timestamps.push(record_failure(i, &mut parse_failures, "null timestamp")?);
            }
        }
        warn_timestamp_failures(parse_failures, batch.num_rows());
        return Ok(timestamps);
    }

    // Try as string array (ISO8601)
    if let Some(str_array) = column.as_any().downcast_ref::<StringArray>() {
        for i in 0..batch.num_rows() {
            if str_array.is_valid(i) {
                let s = str_array.value(i);
                match parse_iso8601(s) {
                    Ok(ts) => timestamps.push(ts),
                    Err(_) => {
                        let reason = format!("unparseable ISO8601 timestamp {s:?}");
                        timestamps.push(record_failure(i, &mut parse_failures, &reason)?);
                    }
                }
            } else {
                timestamps.push(record_failure(i, &mut parse_failures, "null timestamp")?);
            }
        }
        warn_timestamp_failures(parse_failures, batch.num_rows());
        return Ok(timestamps);
    }

    anyhow::bail!("Unsupported timestamp column type")
}

/// Emit a warning summary if any rows had unparseable/null timestamps that
/// were silently coerced to Unix epoch 0.
fn warn_timestamp_failures(failures: usize, total: usize) {
    if failures > 0 {
        tracing::warn!(
            "{} of {} rows had null or unparseable timestamps; \
             these were coerced to Unix epoch 0 (1970-01-01)",
            failures,
            total
        );
    }
}

/// Extract a property value from a column
fn extract_property_value(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
    row_idx: usize,
) -> Option<serde_json::Value> {
    let column = batch.column(col_idx);
    
    if !column.is_valid(row_idx) {
        return None;
    }
    
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

// =============================================================================
// WKB Parsing
// =============================================================================

/// Parse a WKB/EWKB blob into a GeoJSON geometry and its centroid `(lon, lat)`.
///
/// Delegates to `geozero`, which correctly handles 2D, 3D (WKB Z/M), and EWKB
/// (SRID-prefixed) inputs. The previous hand-rolled parser assumed a fixed 2D
/// 16-byte coordinate stride and silently misread any geometry carrying Z/M.
fn parse_wkb_geometry(wkb: &[u8]) -> Option<(Geometry, f64, f64)> {
    use geo::algorithm::centroid::Centroid;
    use geozero::ToGeo;

    // `Ewkb` parses both plain ISO WKB and SRID-prefixed EWKB.
    let geo_geom = geozero::wkb::Ewkb(wkb.to_vec()).to_geo().ok()?;
    let centroid = geo_geom.centroid()?;
    Some((
        Geometry::new(GeomValue::from(&geo_geom)),
        centroid.x(),
        centroid.y(),
    ))
}
