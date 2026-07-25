//! Data loading for GeoParquet files
//!
//! Provides unified data loading from different sources for analysis.

use anyhow::{Context, Result};
use arrow::array::{
    Array, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
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
            DataSource::GeoParquet { path, .. } => path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
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

/// Sample cap for [`LoadedData::sample`]. Enough features for the measured
/// sample encoding (`crate::measure`) to be representative while keeping the
/// retained geometries/properties memory-bounded.
const MAX_SAMPLE_FEATURES: usize = 5000;

/// A sampled property value retained for sample-encode measurement.
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    /// A numeric Arrow column value, widened to f64.
    Number(f64),
    /// A utf8 / dictionary-encoded string value.
    Text(String),
}

/// One deterministically-sampled source feature, retaining the full geometry
/// and property values so `crate::measure` can push it through the real
/// stt-core encoder. Only sampled rows pay this retention cost; the bulk
/// [`AnalyzableFeature`] path keeps its compact summary shape.
#[derive(Debug, Clone)]
pub struct SampledFeature {
    /// Full parsed geometry (WGS84 lon/lat).
    pub geometry: geo_types::Geometry<f64>,
    /// Timestamp in Unix milliseconds.
    pub timestamp_ms: u64,
    /// Property `(name, value)` pairs; columns with unsupported Arrow types
    /// and null values are omitted.
    pub properties: Vec<(String, PropValue)>,
}

/// Loaded dataset for analysis
#[derive(Debug)]
pub struct LoadedData {
    pub features: Vec<AnalyzableFeature>,
    pub bounds: BoundingBox,
    pub time_range: TimeRange,
    /// Deterministic stride sample (never random — the same file always yields
    /// the same sample) of at most [`MAX_SAMPLE_FEATURES`] features, retained
    /// in full for measured sample encoding.
    pub sample: Vec<SampledFeature>,
}

/// Load data from a data source
pub fn load_data(source: &DataSource) -> Result<LoadedData> {
    match source {
        DataSource::GeoParquet {
            path,
            time_field,
            time_format,
        } => load_geoparquet(path, time_field, time_format),
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
    let time_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == time_field)
        .ok_or_else(|| time_field_not_found_error(&schema, time_field))?;

    // Deterministic stride sample: every Nth row (first row included), sized
    // so at most MAX_SAMPLE_FEATURES rows are retained across the whole file.
    let total_rows = builder.metadata().file_metadata().num_rows().max(0) as usize;
    let sample_stride = ((total_rows + MAX_SAMPLE_FEATURES - 1) / MAX_SAMPLE_FEATURES).max(1);
    let mut sample: Vec<SampledFeature> = Vec::new();
    let mut row_index: usize = 0;

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
            // `row_index` counts every physical row so the deterministic sample
            // stride stays aligned with the file-wide row count, even when a row
            // is skipped below.
            let current_row = row_index;
            row_index += 1;

            // Skip rows whose geometry is null or failed to parse: do not fold a
            // phantom (0, 0) into the spatial bounds or append it as a feature.
            // (Mirrors `sample_geometry_at` returning `None` for the same rows.)
            let Some((geom_type, vertex_count, lon, lat)) = geometries.get(i).copied().flatten()
            else {
                continue;
            };
            // A null/unparseable timestamp is kept out of the temporal min/max
            // (`timestamp_opt`) but the feature is still retained for spatial
            // stats with a 0 placeholder.
            let timestamp_opt = timestamps.get(i).copied().flatten();
            let timestamp = timestamp_opt.unwrap_or(0);

            // Update spatial bounds (valid geometry only).
            min_lon = min_lon.min(lon);
            max_lon = max_lon.max(lon);
            min_lat = min_lat.min(lat);
            max_lat = max_lat.max(lat);
            // Update temporal bounds only for rows with a valid timestamp.
            if let Some(ts) = timestamp_opt {
                min_time = min_time.min(ts);
                max_time = max_time.max(ts);
            }

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

            // Sample retention: rows on the stride keep their full geometry
            // and property values (rows whose geometry fails to parse are
            // already skipped above and never reach the sample).
            if current_row % sample_stride == 0 && sample.len() < MAX_SAMPLE_FEATURES {
                if let Some(geometry) = sample_geometry_at(&batch, &geom_col_name, i) {
                    sample.push(SampledFeature {
                        geometry,
                        timestamp_ms: timestamp,
                        properties: sample_properties_at(&batch, i, &geom_col_name, time_field),
                    });
                }
            }
        }

        if features.len() % 100_000 == 0 {
            pb.set_message(format!("Loaded {} features...", features.len()));
        }
    }

    pb.finish_with_message(format!("Loaded {} features", features.len()));

    // If no row carried a valid timestamp the min/max sentinels never moved;
    // collapse them to an empty [0, 0] range rather than leaking u64::MAX.
    let (min_time, max_time) = if min_time == u64::MAX {
        (0, 0)
    } else {
        (min_time, max_time)
    };

    Ok(LoadedData {
        features,
        bounds: BoundingBox::new(min_lon, min_lat, max_lon, max_lat),
        time_range: TimeRange::new(min_time, max_time),
        sample,
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Builds an actionable error for a missing `--time-field`: lists the columns
/// the file actually has, and flags the ones whose names look temporal, so the
/// caller can pick the right one without separately inspecting the schema.
pub(crate) fn time_field_not_found_error(
    schema: &arrow::datatypes::Schema,
    time_field: &str,
) -> anyhow::Error {
    let available: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let likely: Vec<&str> = available
        .iter()
        .copied()
        .filter(|n| looks_temporal(n))
        .collect();
    let hint = if likely.is_empty() {
        String::new()
    } else {
        format!(" Likely time column(s): {}.", likely.join(", "))
    };
    anyhow::anyhow!(
        "Time field '{}' not found. Available columns: {}.{} Pass --time-field <name> to select the timestamp column.",
        time_field,
        available.join(", "),
        hint
    )
}

/// Heuristic: does a column name look like it holds a timestamp/date?
/// (`contains("time")`/`contains("date")` already cover "timestamp" and
/// "datetime".)
pub(crate) fn looks_temporal(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l == "ts" || l.contains("time") || l.contains("date") || l.contains("epoch")
}

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

/// Extract geometries from a batch.
///
/// Each slot is `Some((type, vertex_count, lon, lat))` for a readable geometry,
/// or `None` for a null cell or one whose WKB failed to parse. `None` rows are
/// skipped entirely by the caller (no bounds fold, no feature) rather than
/// coerced into a phantom `(0, 0)` point.
fn extract_geometries_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    geom_col_name: &str,
) -> Result<Vec<Option<(GeometryType, usize, f64, f64)>>> {
    let mut results = Vec::with_capacity(batch.num_rows());

    // Handle separate lon/lat columns
    if geom_col_name == "__lon_lat__" {
        let lon_col = batch
            .column_by_name("lon")
            .or_else(|| batch.column_by_name("longitude"))
            .or_else(|| batch.column_by_name("x"));
        let lat_col = batch
            .column_by_name("lat")
            .or_else(|| batch.column_by_name("latitude"))
            .or_else(|| batch.column_by_name("y"));

        if let (Some(lon), Some(lat)) = (lon_col, lat_col) {
            if let (Some(lon_arr), Some(lat_arr)) = (
                lon.as_any().downcast_ref::<Float64Array>(),
                lat.as_any().downcast_ref::<Float64Array>(),
            ) {
                for i in 0..batch.num_rows() {
                    if lon_arr.is_valid(i) && lat_arr.is_valid(i) {
                        results.push(Some((
                            GeometryType::Point,
                            1,
                            lon_arr.value(i),
                            lat_arr.value(i),
                        )));
                    } else {
                        results.push(None);
                    }
                }
                return Ok(results);
            }
        }
        anyhow::bail!("Expected lon/lat columns but could not read them");
    }

    let geom_col = batch
        .column_by_name(geom_col_name)
        .ok_or_else(|| anyhow::anyhow!("Geometry column '{}' not found", geom_col_name))?;

    // Try GeoArrow struct
    if let Some(struct_array) = geom_col
        .as_any()
        .downcast_ref::<arrow::array::StructArray>()
    {
        let x_col = struct_array
            .column_by_name("x")
            .or_else(|| struct_array.column_by_name("longitude"))
            .or_else(|| struct_array.column_by_name("lon"));
        let y_col = struct_array
            .column_by_name("y")
            .or_else(|| struct_array.column_by_name("latitude"))
            .or_else(|| struct_array.column_by_name("lat"));

        if let (Some(x), Some(y)) = (x_col, y_col) {
            if let (Some(x_arr), Some(y_arr)) = (
                x.as_any().downcast_ref::<Float64Array>(),
                y.as_any().downcast_ref::<Float64Array>(),
            ) {
                for i in 0..batch.num_rows() {
                    if x_arr.is_valid(i) && y_arr.is_valid(i) {
                        results.push(Some((
                            GeometryType::Point,
                            1,
                            x_arr.value(i),
                            y_arr.value(i),
                        )));
                    } else {
                        results.push(None);
                    }
                }
                return Ok(results);
            }
        }
    }

    // Try WKB binary column
    if let Some(binary_array) = geom_col
        .as_any()
        .downcast_ref::<arrow::array::BinaryArray>()
    {
        for i in 0..batch.num_rows() {
            if binary_array.is_valid(i) {
                let wkb = binary_array.value(i);
                if let Some((geom_type, vertex_count, lon, lat)) = parse_wkb_info(wkb) {
                    results.push(Some((geom_type, vertex_count, lon, lat)));
                } else {
                    results.push(None);
                }
            } else {
                results.push(None);
            }
        }
        return Ok(results);
    }

    // Fallback: try separate lon/lat columns
    let lon_col = batch
        .column_by_name("lon")
        .or_else(|| batch.column_by_name("longitude"))
        .or_else(|| batch.column_by_name("x"));
    let lat_col = batch
        .column_by_name("lat")
        .or_else(|| batch.column_by_name("latitude"))
        .or_else(|| batch.column_by_name("y"));

    if let (Some(lon), Some(lat)) = (lon_col, lat_col) {
        if let (Some(lon_arr), Some(lat_arr)) = (
            lon.as_any().downcast_ref::<Float64Array>(),
            lat.as_any().downcast_ref::<Float64Array>(),
        ) {
            for i in 0..batch.num_rows() {
                if lon_arr.is_valid(i) && lat_arr.is_valid(i) {
                    results.push(Some((
                        GeometryType::Point,
                        1,
                        lon_arr.value(i),
                        lat_arr.value(i),
                    )));
                } else {
                    results.push(None);
                }
            }
            return Ok(results);
        }
    }

    anyhow::bail!(
        "Could not extract geometries from column '{}'",
        geom_col_name
    )
}

// =============================================================================
// Sample retention (full geometry + property values for measured encoding)
// =============================================================================

/// Column names that can serve as a bare lon/lat geometry pair (see
/// `find_geometry_column`); excluded from sampled properties for `__lon_lat__`
/// sources so the coordinates aren't double-counted as numeric columns.
const LONLAT_COLUMN_NAMES: &[&str] = &["lon", "longitude", "lat", "latitude", "x", "y"];

/// Full geometry for one sampled row, mirroring `extract_geometries_from_batch`'s
/// column resolution: WKB parses through [`parse_wkb_geometry`]; GeoArrow
/// structs and bare lon/lat column pairs become Points.
fn sample_geometry_at(
    batch: &arrow::record_batch::RecordBatch,
    geom_col_name: &str,
    row: usize,
) -> Option<geo_types::Geometry<f64>> {
    if geom_col_name != "__lon_lat__" {
        if let Some(col) = batch.column_by_name(geom_col_name) {
            if let Some(binary) = col.as_any().downcast_ref::<arrow::array::BinaryArray>() {
                if !binary.is_valid(row) {
                    return None;
                }
                return parse_wkb_geometry(binary.value(row));
            }
            if let Some(struct_array) = col.as_any().downcast_ref::<arrow::array::StructArray>() {
                if let Some(point) = struct_point_at(struct_array, row) {
                    return Some(point);
                }
                // Struct without readable x/y falls through to the lon/lat
                // fallback, like the summary extractor.
            }
        }
    }
    lonlat_point_at(batch, row)
}

/// Point geometry from a GeoArrow-style struct column's x/y children.
fn struct_point_at(
    struct_array: &arrow::array::StructArray,
    row: usize,
) -> Option<geo_types::Geometry<f64>> {
    let x = struct_array
        .column_by_name("x")
        .or_else(|| struct_array.column_by_name("longitude"))
        .or_else(|| struct_array.column_by_name("lon"))?;
    let y = struct_array
        .column_by_name("y")
        .or_else(|| struct_array.column_by_name("latitude"))
        .or_else(|| struct_array.column_by_name("lat"))?;
    let x_arr = x.as_any().downcast_ref::<Float64Array>()?;
    let y_arr = y.as_any().downcast_ref::<Float64Array>()?;
    (x_arr.is_valid(row) && y_arr.is_valid(row)).then(|| {
        geo_types::Geometry::Point(geo_types::Point::new(x_arr.value(row), y_arr.value(row)))
    })
}

/// Point geometry from bare lon/lat columns.
fn lonlat_point_at(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
) -> Option<geo_types::Geometry<f64>> {
    let lon = batch
        .column_by_name("lon")
        .or_else(|| batch.column_by_name("longitude"))
        .or_else(|| batch.column_by_name("x"))?;
    let lat = batch
        .column_by_name("lat")
        .or_else(|| batch.column_by_name("latitude"))
        .or_else(|| batch.column_by_name("y"))?;
    let lon_arr = lon.as_any().downcast_ref::<Float64Array>()?;
    let lat_arr = lat.as_any().downcast_ref::<Float64Array>()?;
    (lon_arr.is_valid(row) && lat_arr.is_valid(row)).then(|| {
        geo_types::Geometry::Point(geo_types::Point::new(
            lon_arr.value(row),
            lat_arr.value(row),
        ))
    })
}

/// Property values for one sampled row: every column except the geometry, the
/// time field, and (for `__lon_lat__` sources) the coordinate pair itself.
/// Columns with unsupported Arrow types are omitted.
fn sample_properties_at(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
    geom_col_name: &str,
    time_field: &str,
) -> Vec<(String, PropValue)> {
    let schema = batch.schema();
    let mut properties = Vec::new();
    for (idx, field) in schema.fields().iter().enumerate() {
        let name = field.name();
        if name == geom_col_name || name == time_field {
            continue;
        }
        if geom_col_name == "__lon_lat__" && LONLAT_COLUMN_NAMES.contains(&name.as_str()) {
            continue;
        }
        if let Some(value) = prop_value_at(batch.column(idx).as_ref(), row) {
            properties.push((name.clone(), value));
        }
    }
    properties
}

/// One property value as a [`PropValue`]: numeric Arrow types widen to f64,
/// utf8/dictionary strings copy out. Nulls and unsupported types yield `None`.
fn prop_value_at(col: &dyn Array, row: usize) -> Option<PropValue> {
    use arrow::array::{
        Float32Array, Int16Array, Int32Array, Int8Array, LargeStringArray, UInt16Array,
        UInt32Array, UInt64Array, UInt8Array,
    };

    if col.is_null(row) {
        return None;
    }
    macro_rules! num {
        ($t:ty) => {
            col.as_any()
                .downcast_ref::<$t>()
                .map(|a| PropValue::Number(a.value(row) as f64))
        };
    }
    match col.data_type() {
        DataType::Float64 => num!(Float64Array),
        DataType::Float32 => num!(Float32Array),
        DataType::Int64 => num!(Int64Array),
        DataType::Int32 => num!(Int32Array),
        DataType::Int16 => num!(Int16Array),
        DataType::Int8 => num!(Int8Array),
        DataType::UInt64 => num!(UInt64Array),
        DataType::UInt32 => num!(UInt32Array),
        DataType::UInt16 => num!(UInt16Array),
        DataType::UInt8 => num!(UInt8Array),
        DataType::Utf8 => col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| PropValue::Text(a.value(row).to_string())),
        DataType::LargeUtf8 => col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|a| PropValue::Text(a.value(row).to_string())),
        // Dictionary<*, Utf8>: cast the single sampled row rather than
        // matching every possible key width.
        DataType::Dictionary(_, values)
            if matches!(values.as_ref(), DataType::Utf8 | DataType::LargeUtf8) =>
        {
            let one = col.slice(row, 1);
            let casted = arrow::compute::cast(one.as_ref(), &DataType::Utf8).ok()?;
            let strings = casted.as_any().downcast_ref::<StringArray>()?;
            strings
                .is_valid(0)
                .then(|| PropValue::Text(strings.value(0).to_string()))
        }
        _ => None,
    }
}

/// Extract timestamps from a column.
///
/// Each slot is `Some(unix_ms)` for a valid timestamp, or `None` for a null
/// cell or (string columns) an unparseable value. `None` rows are excluded from
/// the temporal min/max rather than coerced to epoch 0, which used to pin
/// `min_time` to 1970 and inflate the reported duration by decades.
fn extract_timestamps_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
    time_format: &str,
) -> Result<Vec<Option<u64>>> {
    let column = batch.column(col_idx);
    let mut timestamps = Vec::with_capacity(batch.num_rows());

    // Timestamp-unit scaling routes through the shared `normalize_timestamp_to_ms`
    // (stt_core::timestamp) so this analysis loader agrees byte-for-byte with the
    // stt-build scalar/vertex readers and the DuckDB reader — the divergent local
    // `.max(0)`/hand-rolled ÷ arithmetic here was the audited bug. Null entries
    // yield `None` (skipped from temporal stats, not coerced to 0); a pre-1970 or
    // second→ms-overflowing value still hard-errors like the build path.
    macro_rules! push_ts_column {
        ($arr:expr, $unit:expr) => {{
            for i in 0..batch.num_rows() {
                if $arr.is_valid(i) {
                    timestamps.push(Some(normalize_timestamp_to_ms(i, $arr.value(i), $unit)?));
                } else {
                    timestamps.push(None);
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
                timestamps.push(Some(normalize_timestamp_to_ms(
                    i,
                    int_array.value(i),
                    unit,
                )?));
            } else {
                timestamps.push(None);
            }
        }
        return Ok(timestamps);
    }

    // Try as string array (ISO8601). An unparseable value is excluded from the
    // temporal stats (`None`) rather than coerced to epoch 0.
    if let Some(str_array) = column.as_any().downcast_ref::<StringArray>() {
        for i in 0..batch.num_rows() {
            if str_array.is_valid(i) {
                let s = str_array.value(i);
                timestamps.push(parse_iso8601(s).ok());
            } else {
                timestamps.push(None);
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

/// Parse a WKB/EWKB blob into a `geo_types` geometry.
///
/// Delegates to `geozero` (the same reader stt-build uses), which correctly
/// handles 2D, 3D (WKB Z/M), and SRID-prefixed EWKB inputs. Returns `None`
/// for malformed bytes.
pub fn parse_wkb_geometry(bytes: &[u8]) -> Option<geo_types::Geometry<f64>> {
    use geozero::ToGeo;
    // `Ewkb` parses both plain ISO WKB and SRID-prefixed EWKB.
    geozero::wkb::Ewkb(bytes.to_vec()).to_geo().ok()
}

/// Derive the analysis summary (type, total vertex count, centroid) from WKB.
fn parse_wkb_info(wkb: &[u8]) -> Option<(GeometryType, usize, f64, f64)> {
    let geom = parse_wkb_geometry(wkb)?;
    let (lon, lat) = geometry_centroid(&geom)?;
    Some((classify_geometry(&geom), count_vertices(&geom), lon, lat))
}

/// Map a parsed geometry onto the analysis `GeometryType` enum.
fn classify_geometry(geom: &geo_types::Geometry<f64>) -> GeometryType {
    use geo_types::Geometry as G;
    match geom {
        G::Point(_) => GeometryType::Point,
        G::Line(_) | G::LineString(_) => GeometryType::LineString,
        G::Polygon(_) | G::Rect(_) | G::Triangle(_) => GeometryType::Polygon,
        G::MultiPoint(_) => GeometryType::MultiPoint,
        G::MultiLineString(_) => GeometryType::MultiLineString,
        G::MultiPolygon(_) => GeometryType::MultiPolygon,
        G::GeometryCollection(_) => GeometryType::Unknown,
    }
}

fn polygon_vertex_count(polygon: &geo_types::Polygon<f64>) -> usize {
    polygon.exterior().0.len()
        + polygon
            .interiors()
            .iter()
            .map(|ring| ring.0.len())
            .sum::<usize>()
}

/// Total vertex count across ALL rings and parts (the hand-rolled parser this
/// replaced only counted the first ring / first member geometry).
fn count_vertices(geom: &geo_types::Geometry<f64>) -> usize {
    use geo_types::Geometry as G;
    match geom {
        G::Point(_) => 1,
        G::Line(_) => 2,
        G::LineString(ls) => ls.0.len(),
        G::Polygon(polygon) => polygon_vertex_count(polygon),
        G::MultiPoint(mp) => mp.0.len(),
        G::MultiLineString(mls) => mls.0.iter().map(|ls| ls.0.len()).sum(),
        G::MultiPolygon(mp) => mp.0.iter().map(polygon_vertex_count).sum(),
        G::GeometryCollection(gc) => gc.0.iter().map(count_vertices).sum(),
        G::Rect(_) => 4,
        G::Triangle(_) => 3,
    }
}

/// Geometric centroid as `(lon, lat)`, falling back to the bounding-rect
/// center when the centroid is undefined (e.g. empty geometries).
fn geometry_centroid(geom: &geo_types::Geometry<f64>) -> Option<(f64, f64)> {
    use geo::algorithm::bounding_rect::BoundingRect;
    use geo::algorithm::centroid::Centroid;

    if let Some(c) = geom.centroid() {
        return Some((c.x(), c.y()));
    }
    geom.bounding_rect().map(|rect| {
        let c = rect.center();
        (c.x, c.y)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry, LineString, MultiPolygon, Point, Polygon};
    use geozero::{CoordDimensions, ToWkb};

    #[test]
    fn missing_time_field_error_lists_columns_and_flags_likely_ones() {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Schema::new(vec![
            Field::new("iso_time", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("wmo_wind", DataType::Float64, false),
        ]);
        let err = time_field_not_found_error(&schema, "timestamp").to_string();
        assert!(err.contains("'timestamp' not found"), "{err}");
        // Lists every available column…
        assert!(err.contains("iso_time"), "{err}");
        assert!(err.contains("wmo_wind"), "{err}");
        // …and flags the temporal-looking one specifically.
        assert!(err.contains("Likely time column(s): iso_time"), "{err}");
    }

    #[test]
    fn missing_time_field_error_omits_hint_when_no_temporal_columns() {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("wind", DataType::Float64, false),
        ]);
        let err = time_field_not_found_error(&schema, "timestamp").to_string();
        assert!(!err.contains("Likely time column(s)"), "{err}");
    }

    #[test]
    fn looks_temporal_matches_common_names() {
        for name in [
            "iso_time",
            "timestamp",
            "start_date",
            "TS",
            "epoch_ms",
            "datetime",
        ] {
            assert!(looks_temporal(name), "{name} should look temporal");
        }
        for name in ["name", "wind", "magnitude", "id"] {
            assert!(!looks_temporal(name), "{name} should NOT look temporal");
        }
    }

    /// Encode a geo-types geometry as ISO WKB via geozero's writer.
    fn wkb(geom: &Geometry<f64>) -> Vec<u8> {
        geom.to_wkb(CoordDimensions::xy())
            .expect("encode WKB fixture")
    }

    fn closed_ring(coords: &[(f64, f64)]) -> LineString<f64> {
        LineString::from(coords.to_vec())
    }

    #[test]
    fn parses_point() {
        let bytes = wkb(&Geometry::Point(Point::new(1.5, -2.5)));
        let (geom_type, vertices, lon, lat) = parse_wkb_info(&bytes).unwrap();
        assert_eq!(geom_type, GeometryType::Point);
        assert_eq!(vertices, 1);
        assert_eq!((lon, lat), (1.5, -2.5));
    }

    #[test]
    fn parses_linestring() {
        // Evenly spaced straight line so geo's length-weighted centroid
        // lands on the middle vertex.
        let line = LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]);
        let bytes = wkb(&Geometry::LineString(line));
        let (geom_type, vertices, lon, lat) = parse_wkb_info(&bytes).unwrap();
        assert_eq!(geom_type, GeometryType::LineString);
        assert_eq!(vertices, 3);
        assert!((lon - 1.0).abs() < 1e-9);
        assert!(lat.abs() < 1e-9);
    }

    #[test]
    fn polygon_vertex_count_includes_interior_rings() {
        let exterior = closed_ring(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)]);
        let interior = closed_ring(&[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)]);
        let bytes = wkb(&Geometry::Polygon(Polygon::new(exterior, vec![interior])));
        let (geom_type, vertices, _, _) = parse_wkb_info(&bytes).unwrap();
        assert_eq!(geom_type, GeometryType::Polygon);
        // 5 exterior + 5 interior vertices (the old parser reported 5).
        assert_eq!(vertices, 10);
    }

    #[test]
    fn multipolygon_counts_all_parts() {
        let tri_a = Polygon::new(
            closed_ring(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]),
            vec![],
        );
        let tri_b = Polygon::new(
            closed_ring(&[(10.0, 10.0), (11.0, 10.0), (10.0, 11.0), (10.0, 10.0)]),
            vec![],
        );
        let bytes = wkb(&Geometry::MultiPolygon(MultiPolygon(vec![tri_a, tri_b])));
        let (geom_type, vertices, _, _) = parse_wkb_info(&bytes).unwrap();
        assert_eq!(geom_type, GeometryType::MultiPolygon);
        // 4 + 4 vertices (the old parser reported the first ring only = 4).
        assert_eq!(vertices, 8);
    }

    #[test]
    fn unit_square_centroid() {
        let square = Polygon::new(
            closed_ring(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]),
            vec![],
        );
        let bytes = wkb(&Geometry::Polygon(square));
        let (_, _, lon, lat) = parse_wkb_info(&bytes).unwrap();
        assert!((lon - 0.5).abs() < 1e-9);
        assert!((lat - 0.5).abs() < 1e-9);
    }

    #[test]
    fn malformed_bytes_return_none() {
        assert!(parse_wkb_geometry(&[]).is_none());
        assert!(parse_wkb_geometry(&[0x01, 0x02, 0x03]).is_none());
        // Valid header claiming a point, but truncated coordinates.
        assert!(parse_wkb_geometry(&[0x01, 0x01, 0x00, 0x00, 0x00, 0x00]).is_none());
        assert!(parse_wkb_info(&[0xff; 16]).is_none());
    }

    #[test]
    fn load_geoparquet_retains_deterministic_sample() {
        use arrow::array::{BinaryArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let n = 120usize;
        let wkbs: Vec<Vec<u8>> = (0..n)
            .map(|i| wkb(&Geometry::Point(Point::new(-73.0 + i as f64 * 0.01, 45.0))))
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(
                    wkbs.iter().map(|v| v.as_slice()),
                )),
                Arc::new(Int64Array::from(
                    (0..n as i64)
                        .map(|i| 1_600_000_000_000 + i * 1_000)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    (0..n).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    (0..n).map(|i| format!("cat-{}", i % 3)).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.parquet");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let data = load_data(&DataSource::GeoParquet {
            path,
            time_field: "timestamp".to_string(),
            time_format: "unix-ms".to_string(),
        })
        .unwrap();

        assert_eq!(data.features.len(), n);
        // Below the cap the stride is 1: every row is retained.
        assert_eq!(data.sample.len(), n);
        let s = &data.sample[3];
        assert!(matches!(s.geometry, Geometry::Point(_)));
        assert_eq!(s.timestamp_ms, 1_600_000_000_000 + 3 * 1_000);
        assert_eq!(
            s.properties,
            vec![
                ("value".to_string(), PropValue::Number(1.5)),
                ("name".to_string(), PropValue::Text("cat-0".to_string())),
            ]
        );

        // The retained sample is measurable end-to-end.
        let measured = crate::measure::measure_sample(
            &data.sample,
            &crate::measure::MeasureSettings::default(),
        )
        .unwrap()
        .expect("sample is large enough to measure");
        assert_eq!(measured.features, n);
        assert_eq!(measured.geometry_kind, "point");
    }

    #[test]
    fn bad_timestamp_and_geometry_rows_do_not_poison_bounds_or_time() {
        // Regression: a null/unparseable timestamp must not pin min_time to
        // epoch 0, and a parse-failure geometry must not fold a phantom (0, 0)
        // into the spatial bounds (nor survive as a feature).
        use arrow::array::{BinaryArray, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let good_a = wkb(&Geometry::Point(Point::new(-73.0, 45.0)));
        let good_b = wkb(&Geometry::Point(Point::new(-71.0, 46.0)));
        let bad_geom: Vec<u8> = vec![0xff, 0xff, 0xff, 0xff]; // unparseable WKB

        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("timestamp", DataType::Utf8, true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values([
                    good_a.as_slice(),
                    good_b.as_slice(),
                    bad_geom.as_slice(),
                ])),
                // Row 0: valid ISO (2020-09-13T12:26:40Z == 1_600_000_000_000 ms)
                // Row 1: unparseable "N/A" -> excluded from temporal stats
                // Row 2: valid time, but its geometry fails to parse so the whole
                //        row (including this timestamp) is skipped.
                Arc::new(StringArray::from(vec![
                    "2020-09-13T12:26:40Z",
                    "N/A",
                    "2020-09-13T12:28:20Z",
                ])),
            ],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_rows.parquet");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let data = load_data(&DataSource::GeoParquet {
            path,
            time_field: "timestamp".to_string(),
            time_format: "iso8601".to_string(),
        })
        .unwrap();

        // The parse-failure geometry row is dropped entirely.
        assert_eq!(data.features.len(), 2);

        // Only row 0 carried a valid timestamp, so the range is exactly that
        // instant — NOT collapsed to epoch 0 by the "N/A"/dropped rows.
        assert_eq!(data.time_range.start, 1_600_000_000_000);
        assert_eq!(data.time_range.end, 1_600_000_000_000);

        // Bounds span only the two valid points; (0, 0) never leaks in.
        assert_eq!(data.bounds.min_lon, -73.0);
        assert_eq!(data.bounds.max_lon, -71.0);
        assert_eq!(data.bounds.min_lat, 45.0);
        assert_eq!(data.bounds.max_lat, 46.0);
    }

    #[test]
    fn prop_value_at_widens_numerics_and_copies_strings() {
        use arrow::array::{DictionaryArray, Float64Array, Int32Array, StringArray};
        use arrow::datatypes::Int32Type;

        let floats = Float64Array::from(vec![Some(1.5), None]);
        assert_eq!(prop_value_at(&floats, 0), Some(PropValue::Number(1.5)));
        assert_eq!(prop_value_at(&floats, 1), None);

        let ints = Int32Array::from(vec![7]);
        assert_eq!(prop_value_at(&ints, 0), Some(PropValue::Number(7.0)));

        let strings = StringArray::from(vec!["storm"]);
        assert_eq!(
            prop_value_at(&strings, 0),
            Some(PropValue::Text("storm".to_string()))
        );

        let dict: DictionaryArray<Int32Type> = vec!["a", "b", "a"].into_iter().collect();
        assert_eq!(
            prop_value_at(&dict, 2),
            Some(PropValue::Text("a".to_string()))
        );
    }
}
