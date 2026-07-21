//! GeoParquet input parsing and feature loading

use crate::columnar::{PropertyKind, PropertyTypes};
use anyhow::{Context, Result};
use arrow::array::{
    Array, BinaryViewArray, Float32Array, Float64Array, Int64Array, LargeBinaryArray, ListArray,
    StringArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray,
};
use arrow::datatypes::DataType;
use geojson::{Feature, Geometry, Value as GeomValue};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::KeyValue;
use std::collections::HashMap;
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
    /// Optional per-vertex scalar values (producer-defined; e.g. sea-surface
    /// temperature for the ocean-drifter dataset). Same length contract as
    /// `vertex_timestamps`; flows through clipping and into the tile's
    /// `vertex_value` column so renderers can color the line by it.
    pub vertex_values: Option<Vec<f32>>,
    /// Optional per-vertex × per-bucket value matrix, flattened **vertex-major**
    /// (`matrix[v * num_buckets + b]`). Length is `coord_count * num_buckets`.
    /// Carried through clipping (each bucket channel resampled like
    /// `vertex_values`) into the tile's `vertex_value_matrix` column so a
    /// static-geometry overview can animate per-bucket without re-emitting
    /// geometry. Mutually exclusive with `vertex_values` in practice.
    pub vertex_value_matrix: Option<Vec<f32>>,
    pub lon: f64,
    pub lat: f64,
}

/// Load all features from a GeoParquet file into memory.
///
/// This is the simple in-memory approach that works well for datasets
/// that fit in RAM (up to several million features on a typical machine).
/// Strictness for input parsing, applied independently to timestamps
/// (`--strict-times`) and geometries (`--strict-geometry`). For timestamps,
/// `Warn` keeps the legacy "coerce bad timestamps to epoch 0 and log"
/// behaviour; for geometries, `Warn` skips the row entirely (a feature with
/// no parseable position cannot be tiled). `Strict` aborts the build on the
/// first parse failure with a row-counted error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputStrictness {
    Warn,
    Strict,
}

/// Wire format of the `--time-field` column. Only consulted for integer
/// (Int64) time columns — Arrow Timestamp columns are self-describing and
/// String columns are always parsed as ISO 8601 regardless of this flag.
///
/// The `serde` rename spellings match the clap `ValueEnum` value names exactly,
/// so a JSON config file (e.g. `stt-serve --config`) accepts the same
/// `"iso8601"`/`"unix-sec"`/`"unix-ms"` strings the CLI does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Deserialize)]
pub enum TimeFormat {
    /// ISO 8601 strings (e.g. `2024-06-21T12:00:00Z`).
    #[serde(rename = "iso8601")]
    Iso8601,
    /// Integer seconds since the Unix epoch.
    #[serde(rename = "unix-sec")]
    UnixSec,
    /// Integer milliseconds since the Unix epoch.
    #[serde(rename = "unix-ms")]
    UnixMs,
}

impl TimeFormat {
    /// Canonical CLI spelling — matches the clap `ValueEnum` value names.
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeFormat::Iso8601 => "iso8601",
            TimeFormat::UnixSec => "unix-sec",
            TimeFormat::UnixMs => "unix-ms",
        }
    }
}

pub fn load_features(
    path: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
) -> Result<Vec<ParsedFeature>> {
    let mut features = Vec::new();
    let mut row_count = 0usize;
    stream_features(
        path,
        time_field,
        end_time_field,
        time_format,
        time_strictness,
        geometry_strictness,
        |batch| {
            row_count += batch.len();
            features.extend(batch);
            if row_count.is_multiple_of(100_000) {
                tracing::info!("Loaded {} features...", row_count);
            }
            Ok(())
        },
    )?;
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
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
    mut on_batch: F,
) -> Result<()>
where
    F: FnMut(Vec<ParsedFeature>) -> Result<()>,
{
    let file = File::open(path).context("Failed to open GeoParquet file")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    // GeoParquet `geo` footer metadata (absent for plain Parquet inputs with
    // lon/lat columns — those keep working through the name heuristics).
    let geo_meta = parse_geo_metadata(
        builder.metadata().file_metadata().key_value_metadata(),
        &schema,
    );
    let geom_col_name = find_geometry_column(&schema, geo_meta.as_ref())?;
    if let Some(meta) = geo_meta.as_ref() {
        validate_geo_column(meta, &geom_col_name, geometry_strictness)?;
    }
    let time_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == time_field)
        .ok_or_else(|| time_field_not_found_error(&schema, time_field))?;
    // An Int64 time column can't hold ISO 8601 strings; the values fall back
    // to the unix-ms interpretation. Surface the mismatch instead of letting
    // the documented default silently mean unix-ms.
    if time_format == TimeFormat::Iso8601
        && matches!(schema.field(time_col_idx).data_type(), DataType::Int64)
    {
        tracing::warn!(
            "--time-format iso8601 but time column '{}' is Int64; integer \
             values are interpreted as unix-ms (pass --time-format unix-ms \
             or unix-sec to make this explicit)",
            time_field
        );
    }
    let end_time_col_idx =
        end_time_field.and_then(|field| schema.fields().iter().position(|f| f.name() == field));

    // Optional per-vertex timestamps column (List<Timestamp> or List<Int64>).
    // Producers that have real per-segment timing (e.g. nyc-rideshare with
    // OSRM annotations) populate this; absence falls back to legacy
    // uniform-by-distance interpolation in columnar.rs.
    let vertex_times_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "vertex_timestamps");

    // Optional per-vertex scalar column (List<Float32> or List<Float64>),
    // e.g. sea-surface temperature for the ocean-drifter dataset. Aligned with
    // the geometry vertices like `vertex_timestamps`.
    let vertex_values_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "vertex_values");

    // Optional per-vertex × per-bucket value matrix (List<Float32>), flattened
    // vertex-major. Present for static-geometry overviews (flow corridors);
    // animated by selecting the active bucket column at render time.
    let vertex_value_matrix_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "vertex_value_matrix");

    let reader = builder.build()?;

    // Coordinate columns actually consumed as geometry for this build — only
    // these are withheld from the property set (a real `x`/`lat` attribute on a
    // WKB-geometry input survives).
    let coordinate_cols = consumed_coordinate_columns(&schema, &geom_col_name);

    // Property column indices computed once.
    let property_cols: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(idx, field)| {
            if is_property_column(
                field.name(),
                &geom_col_name,
                time_field,
                end_time_field,
                &coordinate_cols,
            ) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    let mut row_count = 0usize;
    // Aligned with `property_cols`: flips true the first time the column
    // yields a value, so columns that never do can be reported at EOF (the
    // silent-drop accounting the PostGIS/DuckDB readers already have).
    let mut seen_props = vec![false; property_cols.len()];
    // Whole-input accounting for the main time column. Scattered failures are
    // per-row dirt (warn + coerce to epoch 0); EVERY row failing means the
    // column/format is wrong, and the all-1970 archive it would produce is
    // silent garbage — fail the build instead, even in Warn mode.
    let mut time_parse_failures = 0usize;
    for batch_result in reader {
        let batch = batch_result.context("Failed to read Parquet batch")?;
        let parsed = parse_batch(
            &batch,
            &schema,
            &geom_col_name,
            time_col_idx,
            end_time_col_idx,
            vertex_times_col_idx,
            vertex_values_col_idx,
            vertex_value_matrix_col_idx,
            &property_cols,
            &mut seen_props,
            time_format,
            time_strictness,
            geometry_strictness,
            row_count,
            &mut time_parse_failures,
        )?;
        row_count += batch.num_rows();
        on_batch(parsed)?;
    }
    if row_count > 0 && time_parse_failures == row_count {
        anyhow::bail!(
            "all {row_count} rows in time column '{time_field}' were null or unparseable — \
             refusing to write an archive whose every feature is coerced to epoch 0 \
             (1970-01-01). Check --time-field/--time-format: zone-less ISO 8601 strings \
             are read as UTC; integer columns are unix-ms unless --time-format unix-sec"
        );
    }
    warn_dropped_property_columns(&schema, &property_cols, &seen_props, row_count);
    Ok(())
}

/// Builds an actionable error for a missing `--time-field`: lists the columns
/// the source actually has, and flags the ones whose names look temporal, so
/// the caller can pick the right one without separately inspecting the schema.
/// Mirrors `stt_optimize`'s recommend loader so both surfaces fail the same way.
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

/// Warn once about property columns present in the source but carrying no
/// value in any row — the silent-drop cases (unmappable Arrow column type, or
/// entirely NULL) made visible so a missing tile column isn't a mystery.
/// Mirrors the PostGIS/DuckDB readers' `warn_dropped_columns` so all three
/// input adaptors degrade equally loudly.
fn warn_dropped_property_columns(
    schema: &arrow::datatypes::Schema,
    property_cols: &[usize],
    seen: &[bool],
    total_rows: usize,
) {
    if total_rows == 0 {
        return;
    }
    // Report `name (arrow_type)` so an operator can tell an unmappable-type
    // drop (e.g. `foo (Decimal128(38, 9))`) from an all-NULL column at a glance.
    let dropped: Vec<String> = property_cols
        .iter()
        .zip(seen)
        .filter(|&(_, &s)| !s)
        .map(|(&idx, _)| {
            let field = schema.field(idx);
            format!("{} ({})", field.name(), field.data_type())
        })
        .collect();
    if !dropped.is_empty() {
        tracing::warn!(
            "{} source column(s) carried no value in any of {total_rows} rows and were \
             dropped from tiles (unmappable Arrow column type — e.g. decimal/struct/binary — \
             or entirely NULL): {}",
            dropped.len(),
            dropped.join(", ")
        );
    }
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
    vertex_values_col_idx: Option<usize>,
    vertex_value_matrix_col_idx: Option<usize>,
    property_cols: &[usize],
    seen_props: &mut [bool],
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
    row_offset: usize,
    time_parse_failures: &mut usize,
) -> Result<Vec<ParsedFeature>> {
    let geometries = extract_geometries_from_batch(batch, geom_col_name)?;
    let (timestamps, batch_time_failures) = extract_timestamps_from_batch(
        batch,
        time_col_idx,
        time_format,
        time_strictness,
        row_offset,
    )?;
    *time_parse_failures += batch_time_failures;
    let end_timestamps = end_time_col_idx
        .map(|idx| {
            extract_timestamps_from_batch(batch, idx, time_format, time_strictness, row_offset)
        })
        .transpose()?
        .map(|(ts, _)| ts);
    let vertex_times = vertex_times_col_idx
        .map(|idx| extract_vertex_timestamps_from_batch(batch, idx, row_offset))
        .transpose()?;
    let vertex_values = vertex_values_col_idx
        .map(|idx| extract_vertex_values_from_batch(batch, idx))
        .transpose()?;
    // Matrix rows are flat List<Float32> just like vertex_values — reuse the
    // same extractor; the bucket reshape happens at the renderer.
    let vertex_value_matrices = vertex_value_matrix_col_idx
        .map(|idx| extract_vertex_values_from_batch(batch, idx))
        .transpose()?;

    let mut features = Vec::with_capacity(batch.num_rows());
    let mut geometry_failures = 0usize;
    for i in 0..batch.num_rows() {
        let slot = geometries
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("Missing geometry at row {}", row_offset + i))?;
        // A row whose geometry is null or unparseable has no position to tile
        // at. Strict mode fails the build; Warn mode skips the row (it must
        // NOT fall through as a (0,0) point — that tiled garbage at Null
        // Island and dragged it into every zoom level).
        let Some((geometry, lon, lat)) = slot.clone() else {
            geometry_failures += 1;
            if geometry_strictness == InputStrictness::Strict {
                anyhow::bail!(
                    "row {}: null or unparseable geometry (rerun without \
                     --strict-geometry to skip such rows)",
                    row_offset + i
                );
            }
            continue;
        };
        let timestamp = timestamps
            .get(i)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Missing timestamp at row {}", row_offset + i))?;
        let end_timestamp = end_timestamps.as_ref().and_then(|ts| ts.get(i).copied());
        let row_vertex_times = vertex_times
            .as_ref()
            .and_then(|v| v.get(i).cloned().flatten());
        let row_vertex_values = vertex_values
            .as_ref()
            .and_then(|v| v.get(i).cloned().flatten());
        let row_vertex_value_matrix = vertex_value_matrices
            .as_ref()
            .and_then(|v| v.get(i).cloned().flatten());

        // Build properties from non-meta columns. We only materialise this
        // map when the row actually has property values; an empty map is
        // dropped so empty rows pay no allocation.
        let mut properties = serde_json::Map::new();
        for (pc, &col_idx) in property_cols.iter().enumerate() {
            if let Some(value) = extract_property_value(batch, col_idx, i) {
                seen_props[pc] = true;
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
            vertex_values: row_vertex_values,
            vertex_value_matrix: row_vertex_value_matrix,
            lon,
            lat,
        });
    }
    warn_geometry_failures(geometry_failures, batch.num_rows());
    Ok(features)
}

/// Extract optional per-row vertex-timestamp lists from a `vertex_timestamps`
/// column. Tolerates both `List<Timestamp(Millisecond)>` and `List<Int64>`
/// children (ms in either case). Null rows return `None` for that slot;
/// non-list columns return an error.
/// Column names that *may* denote geometry-component coordinates rather than
/// user properties. Case-SENSITIVE lowercase.
///
/// NO built-in adaptor excludes columns by this blanket name test anymore:
/// the GeoParquet reader withholds only the precise columns from
/// [`consumed_coordinate_columns`], and the PostGIS/DuckDB readers keep every
/// SELECTed column (their geometry always comes from the geometry column, so
/// a column named `x`/`lat`/etc. is a genuine user attribute there). Kept as
/// the canonical name list for [`consumed_coordinate_columns`]'s candidates
/// and for downstream tooling.
pub fn is_coordinate_column_name(name: &str) -> bool {
    matches!(name, "lon" | "lat" | "longitude" | "latitude" | "x" | "y")
}

/// The per-vertex array column names recognised as geometry metadata (LineString
/// trajectory timing / per-vertex values / animated-overview matrix). These
/// populate the `ParsedFeature.vertex_*` fields, not the property set. Shared
/// across all input adaptors so a new reader can't silently disagree.
pub const VERTEX_METADATA_COLUMNS: [&str; 3] =
    ["vertex_timestamps", "vertex_values", "vertex_value_matrix"];

/// Whether `name` is one of [`VERTEX_METADATA_COLUMNS`].
pub fn is_vertex_metadata_column(name: &str) -> bool {
    VERTEX_METADATA_COLUMNS.contains(&name)
}

/// The top-level column name(s) actually consumed as geometry coordinates for
/// THIS build. Empty unless the geometry comes from separated lon/lat columns
/// (`geom_col_name == "__lon_lat__"`), in which case it is exactly the two
/// columns resolved as the lon and lat legs — the first present of
/// `lon`/`longitude`/`x` and of `lat`/`latitude`/`y`, matching the resolution
/// order in [`extract_geometries_from_batch`] and [`find_geometry_column`].
///
/// A WKB / GeoArrow-struct geometry consumes NO top-level coordinate columns
/// (the struct's x/y legs are nested inside the geometry column, already
/// excluded by name), so in that case a column that merely happens to be named
/// `x`/`lat`/etc. is a genuine user attribute and must survive into tiles.
fn consumed_coordinate_columns(
    schema: &arrow::datatypes::Schema,
    geom_col_name: &str,
) -> Vec<String> {
    if geom_col_name != "__lon_lat__" {
        return Vec::new();
    }
    let mut cols = Vec::new();
    for candidate in ["lon", "longitude", "x"] {
        if schema.field_with_name(candidate).is_ok() {
            cols.push(candidate.to_string());
            break;
        }
    }
    for candidate in ["lat", "latitude", "y"] {
        if schema.field_with_name(candidate).is_ok() {
            cols.push(candidate.to_string());
            break;
        }
    }
    cols
}

/// Whether a source column is a user property (vs geometry / time / vertex
/// metadata / coordinate component). The single predicate behind BOTH the
/// row-reading property selection in [`stream_features`] and the schema-level
/// [`property_kinds`] map, so the two can't disagree.
///
/// `coordinate_cols` is the precise set of coordinate columns actually consumed
/// as this build's geometry (from [`consumed_coordinate_columns`]) — NOT every
/// column that happens to be named `x`/`lat`/etc. A real `x`/`lat` attribute on
/// a WKB-geometry input therefore stays a property instead of being dropped.
fn is_property_column(
    name: &str,
    geom_col_name: &str,
    time_field: &str,
    end_time_field: Option<&str>,
    coordinate_cols: &[String],
) -> bool {
    !(name == geom_col_name
        || name == time_field
        || end_time_field.map(|f| name == f).unwrap_or(false)
        || is_vertex_metadata_column(name)
        || coordinate_cols.iter().any(|c| c == name))
}

/// Schema-level mirror of [`extract_property_value`]'s downcasts: the tile
/// kind an Arrow property column of this type will produce. `None` =
/// unmappable (decimal/struct/binary/…) — such columns yield no values and
/// are reported by the EOF drop accounting.
fn property_kind_for(dt: &DataType) -> Option<PropertyKind> {
    match dt {
        // Floats + every signed/unsigned integer width. Unsigned and narrow
        // integers used to fall through to `None`, silently dropping common
        // columns (e.g. a UInt32 id/count) from every tile.
        DataType::Float64
        | DataType::Float32
        | DataType::Int64
        | DataType::Int32
        | DataType::Int16
        | DataType::Int8
        | DataType::UInt64
        | DataType::UInt32
        | DataType::UInt16
        | DataType::UInt8
        // Dates/timestamps are emitted as epoch-ms numerics; times as their
        // raw sub-day integer; decimals as f64. See `extract_property_value`.
        | DataType::Date32
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => Some(PropertyKind::Numeric),
        // `LargeUtf8` is the 64-bit-offset string layout — carried categorical
        // exactly like `Utf8`.
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Boolean => {
            Some(PropertyKind::Categorical)
        }
        _ => None,
    }
}

/// Derive the authoritative property-type map from a GeoParquet input's
/// schema — the source of truth `TileConfig::property_types` wants so a
/// column that happens to be all-null within one tile still gets its column
/// there (per-tile value sniffing otherwise drops it and the layer schema
/// drifts across tiles; see `ColumnarOptions::property_types`).
///
/// Uses the same geometry-column detection and property-column selection as
/// [`stream_features`], and the same type mapping as its per-row value
/// extraction. Note the schema is authoritative by design: a Utf8 column
/// whose values all happen to look numeric stays Categorical (cast it in the
/// source if you want a numeric column).
pub fn property_kinds(
    path: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
) -> Result<PropertyTypes> {
    let file = File::open(path).context("Failed to open GeoParquet file")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let geo_meta = parse_geo_metadata(
        builder.metadata().file_metadata().key_value_metadata(),
        &schema,
    );
    let geom_col_name = find_geometry_column(&schema, geo_meta.as_ref())?;
    let coordinate_cols = consumed_coordinate_columns(&schema, &geom_col_name);

    let mut kinds = PropertyTypes::new();
    for field in schema.fields() {
        if !is_property_column(
            field.name(),
            &geom_col_name,
            time_field,
            end_time_field,
            &coordinate_cols,
        ) {
            continue;
        }
        if let Some(kind) = property_kind_for(field.data_type()) {
            kinds.insert(field.name().clone(), kind);
        }
    }
    Ok(kinds)
}

fn extract_vertex_timestamps_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
    row_offset: usize,
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
        // The child can be a Timestamp array of ANY precision (Second/
        // Millisecond/Microsecond/Nanosecond — all normalized to ms via the
        // shared `normalize_timestamp_to_ms`, so this path, the scalar
        // `--time-field` path, and the DuckDB reader agree) or a plain
        // Int64Array (raw integer ms). A null element pushes `0` to preserve
        // per-vertex alignment.
        let row_no = row_offset + row;
        // `normalize_timestamp_to_ms` now lives in stt-core and returns a
        // `stt_core::Error`; map it into anyhow here so the `.collect::<Result<…>>()`
        // (anyhow) call sites below infer the right error type.
        let scale = |value: i64, unit: TimestampUnit| -> Result<u64> {
            Ok(normalize_timestamp_to_ms(row_no, value, unit)?)
        };
        let row_times: Vec<u64> =
            if let Some(ts) = values.as_any().downcast_ref::<TimestampSecondArray>() {
                (0..ts.len())
                    .map(|i| {
                        if ts.is_valid(i) {
                            scale(ts.value(i), TimestampUnit::Second)
                        } else {
                            Ok(0)
                        }
                    })
                    .collect::<Result<Vec<u64>>>()?
            } else if let Some(ts) = values.as_any().downcast_ref::<TimestampMillisecondArray>() {
                (0..ts.len())
                    .map(|i| {
                        if ts.is_valid(i) {
                            scale(ts.value(i), TimestampUnit::Millisecond)
                        } else {
                            Ok(0)
                        }
                    })
                    .collect::<Result<Vec<u64>>>()?
            } else if let Some(ts) = values.as_any().downcast_ref::<TimestampMicrosecondArray>() {
                (0..ts.len())
                    .map(|i| {
                        if ts.is_valid(i) {
                            scale(ts.value(i), TimestampUnit::Microsecond)
                        } else {
                            Ok(0)
                        }
                    })
                    .collect::<Result<Vec<u64>>>()?
            } else if let Some(ts) = values.as_any().downcast_ref::<TimestampNanosecondArray>() {
                (0..ts.len())
                    .map(|i| {
                        if ts.is_valid(i) {
                            scale(ts.value(i), TimestampUnit::Nanosecond)
                        } else {
                            Ok(0)
                        }
                    })
                    .collect::<Result<Vec<u64>>>()?
            } else if let Some(ints) = values.as_any().downcast_ref::<Int64Array>() {
                (0..ints.len())
                    .map(|i| {
                        if ints.is_valid(i) {
                            scale(ints.value(i), TimestampUnit::Millisecond)
                        } else {
                            Ok(0)
                        }
                    })
                    .collect::<Result<Vec<u64>>>()?
            } else {
                anyhow::bail!(
                    "vertex_timestamps child must be a Timestamp (second/millisecond/microsecond/\
                 nanosecond) or Int64 (raw ms) array; got {:?}",
                    values.data_type()
                );
            };
        out.push(Some(row_times));
    }
    Ok(out)
}

/// Extract optional per-row vertex-scalar lists from a `vertex_values` column.
/// Tolerates `List<Float32>` and `List<Float64>` children. Null rows return
/// `None` for that slot; null entries within a list become `NaN` so the
/// per-vertex alignment is preserved. Non-list columns return an error.
fn extract_vertex_values_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
) -> Result<Vec<Option<Vec<f32>>>> {
    let column = batch.column(col_idx);
    let list = column
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow::anyhow!("vertex_values column is not a List array"))?;

    let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        if !list.is_valid(row) {
            out.push(None);
            continue;
        }
        let values = list.value(row);
        let row_vals: Vec<f32> = if let Some(f32s) = values.as_any().downcast_ref::<Float32Array>()
        {
            (0..f32s.len())
                .map(|i| {
                    if f32s.is_valid(i) {
                        f32s.value(i)
                    } else {
                        f32::NAN
                    }
                })
                .collect()
        } else if let Some(f64s) = values.as_any().downcast_ref::<Float64Array>() {
            (0..f64s.len())
                .map(|i| {
                    if f64s.is_valid(i) {
                        f64s.value(i) as f32
                    } else {
                        f32::NAN
                    }
                })
                .collect()
        } else {
            anyhow::bail!(
                "vertex_values child must be Float32 or Float64; got {:?}",
                values.data_type()
            );
        };
        out.push(Some(row_vals));
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

    // Time range spans every row — a row whose geometry failed to parse may
    // still carry a valid timestamp.
    let mut min_time = u64::MAX;
    let mut max_time = u64::MIN;
    for f in features {
        min_time = min_time.min(f.timestamp);
        max_time = max_time.max(f.end_timestamp.unwrap_or(f.timestamp));
    }

    // Spatial bounds exclude the null-island sentinel. The reader now skips
    // rows with unparseable/null geometry entirely, but features built
    // programmatically (tests, generators) can still carry the (0.0, 0.0)
    // sentinel; including them would let a single bad row widen the archive
    // bbox to the whole globe (and make the showcase open zoomed all the way
    // out). Real data at precisely (0,0) to full f64 precision is effectively
    // never legitimate.
    let mut min_lon = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let mut counted = 0usize;
    let mut skipped_null_island = 0usize;
    for f in features {
        if f.lon == 0.0 && f.lat == 0.0 {
            skipped_null_island += 1;
            continue;
        }
        min_lon = min_lon.min(f.lon);
        max_lon = max_lon.max(f.lon);
        min_lat = min_lat.min(f.lat);
        max_lat = max_lat.max(f.lat);
        counted += 1;
    }

    if counted == 0 {
        // Degenerate: every feature sits at the sentinel (e.g. a dataset that
        // genuinely straddles null island, or one that is entirely malformed).
        // Fall back to the raw extent rather than return an inside-out bbox.
        for f in features {
            min_lon = min_lon.min(f.lon);
            max_lon = max_lon.max(f.lon);
            min_lat = min_lat.min(f.lat);
            max_lat = max_lat.max(f.lat);
        }
    } else if skipped_null_island > 0 {
        tracing::warn!(
            "excluded {} null-island (0,0) feature(s) from archive bounds \
             (likely coerced bad/missing geometry)",
            skipped_null_island
        );
    }

    Ok((
        BoundingBox::new(min_lon, min_lat, max_lon, max_lat),
        TimeRange::new(min_time, max_time),
    ))
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parsed subset of the GeoParquet `geo` footer key-value metadata. Only the
/// fields the reader acts on (geometry-column selection, CRS gate, encoding
/// gate) are kept; everything else is ignored.
#[derive(Debug, Default, serde::Deserialize)]
struct GeoFileMeta {
    #[serde(default)]
    primary_column: Option<String>,
    #[serde(default)]
    columns: HashMap<String, GeoColumnMeta>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct GeoColumnMeta {
    #[serde(default)]
    encoding: Option<String>,
    /// PROJJSON object, an `"AUTH:CODE"` string, or absent/`null`
    /// (= OGC:CRS84 per the GeoParquet spec).
    #[serde(default)]
    crs: Option<serde_json::Value>,
}

/// Read the GeoParquet `geo` entry from the Parquet footer key-value
/// metadata (falling back to the Arrow schema metadata map). Returns `None`
/// for plain Parquet inputs without the entry; malformed JSON is logged and
/// treated as absent so non-GeoParquet producers can't brick a build.
fn parse_geo_metadata(
    kv: Option<&Vec<KeyValue>>,
    schema: &arrow::datatypes::Schema,
) -> Option<GeoFileMeta> {
    let raw = kv
        .and_then(|entries| {
            entries
                .iter()
                .find(|e| e.key == "geo")
                .and_then(|e| e.value.clone())
        })
        .or_else(|| schema.metadata().get("geo").cloned())?;
    match serde_json::from_str::<GeoFileMeta>(&raw) {
        Ok(meta) => Some(meta),
        Err(e) => {
            tracing::warn!(
                "ignoring malformed GeoParquet 'geo' footer metadata ({e}); \
                 falling back to geometry-column name heuristics"
            );
            None
        }
    }
}

/// Check that a GeoParquet column CRS is lon/lat WGS 84 (OGC:CRS84 /
/// EPSG:4326). Returns a human-readable name of the offending CRS otherwise.
/// Absent / `null` means CRS84 per the GeoParquet spec.
fn crs_is_lonlat_wgs84(crs: &serde_json::Value) -> std::result::Result<(), String> {
    let auth_code_ok = |auth: &str, code: &str| -> bool {
        (auth.eq_ignore_ascii_case("OGC") && code.eq_ignore_ascii_case("CRS84"))
            || (auth.eq_ignore_ascii_case("EPSG") && code == "4326")
    };
    match crs {
        serde_json::Value::Null => Ok(()),
        serde_json::Value::String(s) => {
            // "EPSG:4326" / "OGC:CRS84" / urn:ogc:def:crs:OGC:1.3:CRS84 forms,
            // plus the bare aliases producers commonly write ("CRS84", "4326",
            // "WGS 84", "WGS84"). Match leniently — these are all WGS 84 lon/lat.
            let norm = s.trim();
            let bare_ok = matches!(
                norm.to_ascii_uppercase().as_str(),
                "CRS84" | "4326" | "WGS 84" | "WGS84" | "WGS_1984" | "WGS84(DD)"
            );
            let ok = bare_ok
                || match norm.rsplit_once(':') {
                    Some((head, code)) => {
                        let auth = head.rsplit(':').find(|p| !p.is_empty()).unwrap_or(head);
                        auth_code_ok(auth, code)
                            || (norm.to_ascii_lowercase().contains("ogc")
                                && code.eq_ignore_ascii_case("CRS84"))
                    }
                    None => false,
                };
            if ok {
                Ok(())
            } else {
                Err(format!("'{norm}'"))
            }
        }
        serde_json::Value::Object(obj) => {
            // PROJJSON: prefer the authority id, fall back to the name.
            let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(id) = obj.get("id").and_then(|v| v.as_object()) {
                let auth = id.get("authority").and_then(|v| v.as_str()).unwrap_or("");
                let code = match id.get("code") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Number(n)) => n.to_string(),
                    _ => String::new(),
                };
                if auth_code_ok(auth, &code) {
                    return Ok(());
                }
                return Err(if name.is_empty() {
                    format!("{auth}:{code}")
                } else {
                    format!("{auth}:{code} ({name})")
                });
            }
            // No authority id — accept only the canonical CRS84 names.
            if matches!(name, "WGS 84 (CRS84)" | "WGS 84") {
                Ok(())
            } else if name.is_empty() {
                Err("an unrecognized PROJJSON CRS without an authority id".to_string())
            } else {
                Err(format!("'{name}'"))
            }
        }
        other => Err(format!("{other}")),
    }
}

/// Enforce the GeoParquet column constraints the rest of the reader assumes:
/// lon/lat WGS 84 coordinates, and an encoding the extractors can actually
/// ingest (WKB or native point — the native linestring/polygon/multi*
/// layouts have no extraction path here).
///
/// The CRS gate is advisory under `--strict-geometry == Warn`: a non-WGS84
/// declaration is a *correctness hazard*, not an ingestion blocker — the whole
/// pipeline assumes lon/lat degrees and will tile the raw coordinates as-is, so
/// non-WGS84 input yields silently-wrong tiles. We warn loudly by default (so a
/// mis-declared/already-lon-lat file still builds) and hard-fail under
/// `--strict-geometry`. The encoding gate stays a hard error in both modes: an
/// unsupported native encoding has no extraction path, so there is nothing to
/// tile at all.
fn validate_geo_column(
    meta: &GeoFileMeta,
    geom_col_name: &str,
    geometry_strictness: InputStrictness,
) -> Result<()> {
    let Some(col) = meta.columns.get(geom_col_name) else {
        return Ok(());
    };
    if let Some(crs) = &col.crs {
        if let Err(found) = crs_is_lonlat_wgs84(crs) {
            // Coordinates are consumed verbatim downstream; a non-WGS84 CRS
            // means every tiled position is wrong. Fail in strict mode, warn
            // (once, at load) otherwise.
            if geometry_strictness == InputStrictness::Strict {
                anyhow::bail!(
                    "GeoParquet geometry column '{geom_col_name}' declares CRS {found}, \
                     but stt-build requires lon/lat degrees (OGC:CRS84 / EPSG:4326). \
                     Reproject the input before export (e.g. geopandas: \
                     gdf.to_crs(4326).to_parquet(...))."
                );
            }
            tracing::warn!(
                "GeoParquet geometry column '{geom_col_name}' declares CRS {found}, but \
                 stt-build assumes WGS 84 lon/lat degrees (OGC:CRS84 / EPSG:4326) and \
                 tiles coordinates AS-IS — output tiles will be WRONG if the data is \
                 actually in another CRS. Reproject to EPSG:4326 first (e.g. geopandas: \
                 gdf.to_crs(4326).to_parquet(...)), or pass --strict-geometry to make \
                 this a hard error.",
            );
        }
    }
    if let Some(encoding) = col.encoding.as_deref() {
        let unsupported = matches!(
            encoding.to_ascii_lowercase().as_str(),
            "linestring" | "polygon" | "multipoint" | "multilinestring" | "multipolygon"
        );
        if unsupported {
            anyhow::bail!(
                "GeoParquet geometry column '{geom_col_name}' uses the native \
                 geoarrow '{encoding}' encoding, which this reader cannot ingest. \
                 Re-export with WKB geometry encoding (e.g. geopandas: \
                 gdf.to_parquet(..., geometry_encoding='WKB'))."
            );
        }
    }
    Ok(())
}

/// Find the geometry column in a Parquet schema. The GeoParquet
/// `primary_column` declaration wins when present and resolvable; the name /
/// type heuristics below cover plain Parquet inputs without `geo` metadata.
fn find_geometry_column(
    schema: &arrow::datatypes::Schema,
    geo_meta: Option<&GeoFileMeta>,
) -> Result<String> {
    if let Some(primary) = geo_meta.and_then(|m| m.primary_column.as_deref()) {
        if schema.field_with_name(primary).is_ok() {
            return Ok(primary.to_string());
        }
        tracing::warn!(
            "GeoParquet metadata names primary_column '{primary}' but the \
             schema has no such column; falling back to name heuristics"
        );
    }

    // Common geometry column names
    let common_names = ["geometry", "geom", "wkb_geometry", "the_geom", "shape"];

    for name in common_names {
        if schema.field_with_name(name).is_ok() {
            return Ok(name.to_string());
        }
    }

    // Look for binary columns that might contain WKB
    for field in schema.fields() {
        if matches!(
            field.data_type(),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        ) {
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

    anyhow::bail!(
        "Could not find geometry column in Parquet schema. Expected columns: {:?}",
        common_names
    )
}

/// Extract geometries from a batch. A `None` slot means the row's geometry
/// is null or unparseable — the caller decides whether to skip or bail
/// (`--strict-geometry`). Such rows must never be materialised as (0,0)
/// points: that tiles garbage at Null Island.
fn extract_geometries_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    geom_col_name: &str,
) -> Result<Vec<Option<(Geometry, f64, f64)>>> {
    // Point rows from a pair of x/y Float64 arrays (top-level lon/lat
    // columns or the legs of a separated GeoArrow point struct).
    fn points_from_xy(
        x_arr: &Float64Array,
        y_arr: &Float64Array,
        num_rows: usize,
    ) -> Vec<Option<(Geometry, f64, f64)>> {
        (0..num_rows)
            .map(|i| {
                if x_arr.is_valid(i) && y_arr.is_valid(i) {
                    let x = x_arr.value(i);
                    let y = y_arr.value(i);
                    Some((Geometry::new(GeomValue::Point(vec![x, y])), x, y))
                } else {
                    None
                }
            })
            .collect()
    }

    // WKB rows from any binary-flavoured array (Binary/LargeBinary/BinaryView).
    fn points_from_wkb<'a>(
        rows: impl Iterator<Item = Option<&'a [u8]>>,
    ) -> Vec<Option<(Geometry, f64, f64)>> {
        rows.map(|wkb| wkb.and_then(parse_wkb_geometry)).collect()
    }

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
                return Ok(points_from_xy(lon_arr, lat_arr, batch.num_rows()));
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
                return Ok(points_from_xy(x_arr, y_arr, batch.num_rows()));
            }
        }
    }

    // Try WKB binary column — all three binary layouts carry the same bytes.
    if let Some(arr) = geom_col
        .as_any()
        .downcast_ref::<arrow::array::BinaryArray>()
    {
        return Ok(points_from_wkb(
            (0..batch.num_rows()).map(|i| arr.is_valid(i).then(|| arr.value(i))),
        ));
    }
    if let Some(arr) = geom_col.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(points_from_wkb(
            (0..batch.num_rows()).map(|i| arr.is_valid(i).then(|| arr.value(i))),
        ));
    }
    if let Some(arr) = geom_col.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok(points_from_wkb(
            (0..batch.num_rows()).map(|i| arr.is_valid(i).then(|| arr.value(i))),
        ));
    }

    // Fallback: separate lon/lat columns
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
            return Ok(points_from_xy(lon_arr, lat_arr, batch.num_rows()));
        }
    }

    anyhow::bail!(
        "Could not extract geometries from column '{}' (Arrow type {:?}). \
         Supported encodings: WKB (Binary/LargeBinary/BinaryView), separated \
         x/y point structs, or top-level lon/lat columns",
        geom_col_name,
        geom_col.data_type()
    )
}

// Timestamp-unit normalization now lives in `stt_core::timestamp` so every
// input adaptor (this GeoParquet reader's scalar + per-vertex paths, the DuckDB
// reader, and the stt-optimize analysis loader) shares ONE implementation. The
// re-exports below keep this module's historical call sites and public surface
// (`input::normalize_timestamp_to_ms`, `input::TimestampUnit`, etc.) unchanged.
pub use stt_core::timestamp::{
    normalize_timestamp_to_ms, reject_negative_timestamp, scale_timestamp_to_ms, TimestampUnit,
};

/// Extract timestamps from a column. `row_offset` is the batch's absolute
/// row position in the file, used for error context. Returns the parsed
/// values plus the number of rows that failed (null or unparseable, coerced
/// to epoch 0 in Warn mode) so the caller can account across the whole input.
fn extract_timestamps_from_batch(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
    time_format: TimeFormat,
    strictness: InputStrictness,
    row_offset: usize,
) -> Result<(Vec<u64>, usize)> {
    let column = batch.column(col_idx);
    let mut timestamps = Vec::with_capacity(batch.num_rows());

    // Count rows whose timestamp could not be parsed (null or invalid).
    // In Warn mode they are coerced to Unix epoch 0 and we log; in Strict
    // mode we fail the build on the first bad row with row context.
    let mut parse_failures: usize = 0;
    let record_failure = |row: usize, parse_failures: &mut usize, reason: &str| -> Result<u64> {
        *parse_failures += 1;
        if strictness == InputStrictness::Strict {
            anyhow::bail!(
                "row {row}: {reason} (rerun without --strict-times to coerce to epoch 0)"
            );
        }
        Ok(0)
    };

    // Arrow Timestamp columns are self-describing: scale any precision to ms
    // through the shared `normalize_timestamp_to_ms` (agrees with the per-vertex
    // path and the DuckDB reader). A null pushes epoch 0 (Warn) or fails
    // (Strict) via `record_failure`.
    macro_rules! push_timestamp_column {
        ($arr:expr, $unit:expr) => {{
            for i in 0..batch.num_rows() {
                if $arr.is_valid(i) {
                    timestamps.push(normalize_timestamp_to_ms(
                        row_offset + i,
                        $arr.value(i),
                        $unit,
                    )?);
                } else {
                    timestamps.push(record_failure(
                        row_offset + i,
                        &mut parse_failures,
                        "null timestamp",
                    )?);
                }
            }
            warn_timestamp_failures(parse_failures, batch.num_rows());
            return Ok((timestamps, parse_failures));
        }};
    }

    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampSecondArray>() {
        push_timestamp_column!(ts_array, TimestampUnit::Second);
    }
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampMillisecondArray>() {
        push_timestamp_column!(ts_array, TimestampUnit::Millisecond);
    }
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        push_timestamp_column!(ts_array, TimestampUnit::Microsecond);
    }
    if let Some(ts_array) = column.as_any().downcast_ref::<TimestampNanosecondArray>() {
        push_timestamp_column!(ts_array, TimestampUnit::Nanosecond);
    }

    // Try as i64 array (unix timestamp). The integer is interpreted per
    // `--time-format`: unix-sec ⇒ Second (×1000, overflow-checked), unix-ms /
    // iso8601 ⇒ Millisecond passthrough (Int64 + iso8601 warned once at schema
    // time in stream_features).
    if let Some(int_array) = column.as_any().downcast_ref::<Int64Array>() {
        let unit = match time_format {
            TimeFormat::UnixSec => TimestampUnit::Second,
            TimeFormat::UnixMs | TimeFormat::Iso8601 => TimestampUnit::Millisecond,
        };
        for i in 0..batch.num_rows() {
            if int_array.is_valid(i) {
                timestamps.push(normalize_timestamp_to_ms(
                    row_offset + i,
                    int_array.value(i),
                    unit,
                )?);
            } else {
                timestamps.push(record_failure(
                    row_offset + i,
                    &mut parse_failures,
                    "null timestamp",
                )?);
            }
        }
        warn_timestamp_failures(parse_failures, batch.num_rows());
        return Ok((timestamps, parse_failures));
    }

    // Try as string array (ISO8601)
    if let Some(str_array) = column.as_any().downcast_ref::<StringArray>() {
        for i in 0..batch.num_rows() {
            if str_array.is_valid(i) {
                let s = str_array.value(i);
                match parse_iso8601(s) {
                    Ok(ms) => {
                        reject_negative_timestamp(row_offset + i, ms)?;
                        timestamps.push(ms as u64);
                    }
                    Err(_) => {
                        let reason = format!("unparseable ISO8601 timestamp {s:?}");
                        timestamps.push(record_failure(
                            row_offset + i,
                            &mut parse_failures,
                            &reason,
                        )?);
                    }
                }
            } else {
                timestamps.push(record_failure(
                    row_offset + i,
                    &mut parse_failures,
                    "null timestamp",
                )?);
            }
        }
        warn_timestamp_failures(parse_failures, batch.num_rows());
        return Ok((timestamps, parse_failures));
    }

    anyhow::bail!(
        "unsupported timestamp column type {:?}: expected a Timestamp \
         (second/millisecond/microsecond/nanosecond), an Int64 (unix seconds or \
         milliseconds per --time-format), or an ISO 8601 String column",
        column.data_type()
    )
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

/// Emit a warning summary if any rows had null/unparseable geometries and
/// were skipped (pass --strict-geometry to fail the build instead).
fn warn_geometry_failures(failures: usize, total: usize) {
    if failures > 0 {
        tracing::warn!(
            "{} of {} rows had null or unparseable geometries and were \
             skipped (pass --strict-geometry to fail the build instead)",
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
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::Int16Array>() {
        return Some(serde_json::json!(arr.value(row_idx) as i64));
    }
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::Int8Array>() {
        return Some(serde_json::json!(arr.value(row_idx) as i64));
    }
    // Unsigned integers: a `serde_json::Number` holds each width natively (u64
    // as u64), so these round-trip through `value_as_f64` without loss for the
    // common cases. Previously dropped entirely.
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::UInt8Array>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::UInt16Array>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::UInt32Array>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::UInt64Array>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    // Date columns → epoch ms (Date32 is whole days since epoch, Date64 ms).
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::Date32Array>() {
        return Some(serde_json::json!(arr.value(row_idx) as i64 * 86_400_000));
    }
    if let Some(arr) = column.as_any().downcast_ref::<arrow::array::Date64Array>() {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    // Timestamp columns of any precision → epoch ms via the shared normalizer
    // (agrees with the `--time-field` path and the DuckDB reader). Values that
    // overflow or predate the epoch drop to `None` rather than wrap silently.
    if let Some(arr) = column.as_any().downcast_ref::<TimestampSecondArray>() {
        return normalize_timestamp_to_ms(row_idx, arr.value(row_idx), TimestampUnit::Second)
            .ok()
            .map(|ms| serde_json::json!(ms));
    }
    if let Some(arr) = column.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return normalize_timestamp_to_ms(row_idx, arr.value(row_idx), TimestampUnit::Millisecond)
            .ok()
            .map(|ms| serde_json::json!(ms));
    }
    if let Some(arr) = column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return normalize_timestamp_to_ms(row_idx, arr.value(row_idx), TimestampUnit::Microsecond)
            .ok()
            .map(|ms| serde_json::json!(ms));
    }
    if let Some(arr) = column.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return normalize_timestamp_to_ms(row_idx, arr.value(row_idx), TimestampUnit::Nanosecond)
            .ok()
            .map(|ms| serde_json::json!(ms));
    }
    // Time-of-day columns → their raw sub-day integer (seconds/ms for Time32,
    // micros/nanos for Time64). Better preserved as numeric than dropped.
    if let Some(arr) = column
        .as_any()
        .downcast_ref::<arrow::array::Time32SecondArray>()
    {
        return Some(serde_json::json!(arr.value(row_idx) as i64));
    }
    if let Some(arr) = column
        .as_any()
        .downcast_ref::<arrow::array::Time32MillisecondArray>()
    {
        return Some(serde_json::json!(arr.value(row_idx) as i64));
    }
    if let Some(arr) = column
        .as_any()
        .downcast_ref::<arrow::array::Time64MicrosecondArray>()
    {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    if let Some(arr) = column
        .as_any()
        .downcast_ref::<arrow::array::Time64NanosecondArray>()
    {
        return Some(serde_json::json!(arr.value(row_idx)));
    }
    // Decimal → f64 (unscaled value / 10^scale). A Decimal256 magnitude that
    // exceeds i128 drops to `None` rather than producing a garbage f64.
    if let Some(arr) = column
        .as_any()
        .downcast_ref::<arrow::array::Decimal128Array>()
    {
        let scaled = arr.value(row_idx) as f64 / 10f64.powi(arr.scale() as i32);
        return Some(serde_json::json!(scaled));
    }
    if let Some(arr) = column
        .as_any()
        .downcast_ref::<arrow::array::Decimal256Array>()
    {
        let scale = arr.scale() as i32;
        return arr
            .value(row_idx)
            .to_i128()
            .map(|v| serde_json::json!(v as f64 / 10f64.powi(scale)));
    }
    // 64-bit-offset UTF-8 strings — categorical exactly like `StringArray`.
    if let Some(arr) = column
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
    {
        return Some(serde_json::json!(arr.value(row_idx)));
    }

    None
}

/// Parse ISO 8601 timestamp to Unix milliseconds. Returns the signed value
/// so the caller can reject pre-1970 (negative) instants explicitly rather
/// than letting them wrap through `as u64`.
pub(crate) fn parse_iso8601(s: &str) -> Result<i64> {
    use chrono::{DateTime, NaiveDateTime};

    // Try parsing as DateTime with timezone
    if let Ok(dt) = s.parse::<DateTime<chrono::Utc>>() {
        return Ok(dt.timestamp_millis());
    }

    // Zone-less (naive) datetimes are interpreted as UTC: both the T- and
    // space-separated forms, with optional fractional seconds (`%.f` also
    // matches no fraction). Real-world CSV/Parquet exports (e.g. NOAA Marine
    // Cadastre AIS BaseDateTime) are commonly `2024-09-28T12:00:00` with no
    // zone suffix, which the zoned parse above rejects.
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(dt.and_utc().timestamp_millis());
        }
    }

    // Try parsing as date only
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        return Ok(dt.timestamp_millis());
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
pub(crate) fn parse_wkb_geometry(wkb: &[u8]) -> Option<(Geometry, f64, f64)> {
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

#[cfg(test)]
mod tests {
    use super::{looks_temporal, parse_iso8601, time_field_not_found_error};

    const SEP_28_2024_NOON_MS: i64 = 1_727_524_800_000; // 2024-09-28T12:00:00Z

    // Guards the invariant asserted in time_field_not_found_error's doc comment:
    // stt-build fails the same, helpful way as stt-optimize's twin (loader.rs).
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
        assert!(err.contains("iso_time"), "{err}");
        assert!(err.contains("wmo_wind"), "{err}");
        assert!(err.contains("Likely time column(s): iso_time"), "{err}");
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

    #[test]
    fn parses_zoned_timestamps() {
        assert_eq!(
            parse_iso8601("2024-09-28T12:00:00Z").unwrap(),
            SEP_28_2024_NOON_MS
        );
        assert_eq!(
            parse_iso8601("2024-09-28T14:00:00+02:00").unwrap(),
            SEP_28_2024_NOON_MS
        );
    }

    #[test]
    fn parses_naive_timestamps_as_utc() {
        // The T-separated zone-less form (NOAA AIS BaseDateTime and most CSV
        // exports) and the space-separated form must agree, both = UTC.
        assert_eq!(
            parse_iso8601("2024-09-28T12:00:00").unwrap(),
            SEP_28_2024_NOON_MS
        );
        assert_eq!(
            parse_iso8601("2024-09-28 12:00:00").unwrap(),
            SEP_28_2024_NOON_MS
        );
    }

    #[test]
    fn parses_naive_fractional_seconds() {
        assert_eq!(
            parse_iso8601("2024-09-28T12:00:00.250").unwrap(),
            SEP_28_2024_NOON_MS + 250
        );
        assert_eq!(
            parse_iso8601("2024-09-28 12:00:00.250").unwrap(),
            SEP_28_2024_NOON_MS + 250
        );
    }

    #[test]
    fn parses_date_only_as_utc_midnight() {
        assert_eq!(
            parse_iso8601("2024-09-28").unwrap(),
            SEP_28_2024_NOON_MS - 12 * 3_600_000
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_iso8601("not a time").is_err());
        assert!(parse_iso8601("2024-13-40T99:00:00").is_err());
        assert!(parse_iso8601("").is_err());
    }

    // ---- property type coverage (finding: silent column drop) ----

    use crate::columnar::PropertyKind;
    use arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn property_kind_covers_unsigned_narrow_date_and_largeutf8() {
        // Types that previously fell through to `None` and silently dropped the
        // whole column from every tile.
        for dt in [
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Int8,
            DataType::Int16,
            DataType::Date32,
            DataType::Date64,
            DataType::Decimal128(38, 9),
        ] {
            assert_eq!(
                super::property_kind_for(&dt),
                Some(PropertyKind::Numeric),
                "{dt:?} should map to Numeric"
            );
        }
        assert_eq!(
            super::property_kind_for(&DataType::LargeUtf8),
            Some(PropertyKind::Categorical)
        );
        // Genuinely unmappable types still drop (and are reported at EOF).
        assert_eq!(super::property_kind_for(&DataType::Binary), None);
    }

    #[test]
    fn extract_value_handles_uint_largeutf8_and_date() {
        use arrow::array::{ArrayRef, Date32Array, LargeStringArray, UInt32Array};
        use std::sync::Arc;

        let u: ArrayRef = Arc::new(UInt32Array::from(vec![Some(7u32), None]));
        let s: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("hello"), None]));
        // Date32 counts whole days since the epoch → 1 day == 86_400_000 ms.
        let d: ArrayRef = Arc::new(Date32Array::from(vec![Some(1i32), None]));
        let schema = Schema::new(vec![
            Field::new("u", DataType::UInt32, true),
            Field::new("s", DataType::LargeUtf8, true),
            Field::new("d", DataType::Date32, true),
        ]);
        let batch =
            arrow::record_batch::RecordBatch::try_new(Arc::new(schema), vec![u, s, d]).unwrap();

        assert_eq!(
            super::extract_property_value(&batch, 0, 0),
            Some(serde_json::json!(7))
        );
        assert_eq!(
            super::extract_property_value(&batch, 1, 0),
            Some(serde_json::json!("hello"))
        );
        assert_eq!(
            super::extract_property_value(&batch, 2, 0),
            Some(serde_json::json!(86_400_000i64))
        );
        // Null slots still yield None for every new type.
        assert_eq!(super::extract_property_value(&batch, 0, 1), None);
        assert_eq!(super::extract_property_value(&batch, 1, 1), None);
        assert_eq!(super::extract_property_value(&batch, 2, 1), None);
    }

    // ---- coordinate-column narrowing (finding: real x/lat attrs dropped) ----

    #[test]
    fn wkb_geometry_keeps_named_coordinate_attributes() {
        // A WKB-geometry input consumes NO top-level coordinate columns, so a
        // real `x` / `lat` attribute must survive as a property.
        let schema = Schema::new(vec![
            Field::new("geometry", DataType::Binary, true),
            Field::new("x", DataType::Float64, true),
            Field::new("lat", DataType::Float64, true),
        ]);
        let coords = super::consumed_coordinate_columns(&schema, "geometry");
        assert!(
            coords.is_empty(),
            "WKB build consumes no coordinate columns"
        );
        assert!(super::is_property_column(
            "x", "geometry", "t", None, &coords
        ));
        assert!(super::is_property_column(
            "lat", "geometry", "t", None, &coords
        ));
    }

    #[test]
    fn lonlat_geometry_excludes_only_the_consumed_legs() {
        // A separated lon/lat build consumes exactly its lon + lat legs; those
        // are withheld, everything else (including a stray `x`) stays a property.
        let schema = Schema::new(vec![
            Field::new("lon", DataType::Float64, true),
            Field::new("lat", DataType::Float64, true),
            Field::new("x", DataType::Float64, true),
            Field::new("value", DataType::Float64, true),
        ]);
        let coords = super::consumed_coordinate_columns(&schema, "__lon_lat__");
        assert_eq!(coords, vec!["lon".to_string(), "lat".to_string()]);
        assert!(!super::is_property_column(
            "lon",
            "__lon_lat__",
            "t",
            None,
            &coords
        ));
        assert!(!super::is_property_column(
            "lat",
            "__lon_lat__",
            "t",
            None,
            &coords
        ));
        // `x` is present but NOT the resolved lon leg (lon won), so it survives.
        assert!(super::is_property_column(
            "x",
            "__lon_lat__",
            "t",
            None,
            &coords
        ));
        assert!(super::is_property_column(
            "value",
            "__lon_lat__",
            "t",
            None,
            &coords
        ));
    }
}
