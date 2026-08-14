//! GeoParquet input parsing and feature loading

use crate::columnar::{PropertyKind, PropertyTypes};
use crate::props::FeatureProperties;
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
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use stt_core::metadata::{
    coord_quant_step_deg, ContentFingerprint, CONTENT_FINGERPRINT_VERSION,
    FINGERPRINT_CARDINALITY_CAP,
};
use stt_core::types::{BoundingBox, TimeRange};

/// Shared properties map to avoid cloning per-segment. Still the payload of
/// [`FeatureProperties::Owned`] — the representation used by the DB readers and
/// by synthesised features, where there is no batch to build a column over.
pub type SharedProperties = Arc<serde_json::Map<String, serde_json::Value>>;

/// Parsed feature with geometry and temporal information
#[derive(Debug, Clone)]
pub struct ParsedFeature {
    /// DT-2: the ONE zoom this feature is stored at under additive
    /// decomposition, when `--additive-lod` assigned one. `None` = today's
    /// replicated placement (the feature appears at every zoom in its band).
    ///
    /// Read by the tiler's band mechanism ahead of `min_zoom_field`, so the
    /// placement authority itself needs no change.
    pub home_zoom: Option<u8>,
    pub geojson: Feature,
    /// This feature's properties — either a handle into its batch's columnar
    /// table (the GeoParquet path) or an owned map. Cloning is a refcount bump
    /// either way, so clipping still shares rather than copies.
    pub shared_properties: Option<FeatureProperties>,
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

/// Total rows a Parquet file declares in its footer, for pre-sizing the feature
/// vector. `None` when the file cannot be opened or read as Parquet — the
/// caller then falls back to a growing vector, and the real error surfaces from
/// [`stream_features`] a moment later with its own context.
fn parquet_row_count(path: &Path) -> Option<usize> {
    let file = File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    usize::try_from(builder.metadata().file_metadata().num_rows()).ok()
}

pub fn load_features(
    path: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
) -> Result<Vec<ParsedFeature>> {
    // Reserve from the footer's row count rather than growing by doubling.
    // `ParsedFeature` is 328 bytes, so at national point-cloud scale the final
    // doubling alone would transiently hold the old vector AND the new one —
    // tens of gigabytes of pure copy overhead at the moment of reallocation,
    // for a figure the file already knows. Rows that fail geometry parsing just
    // leave the reservation slightly over.
    let mut features = match parquet_row_count(path) {
        Some(rows) => Vec::with_capacity(rows),
        None => Vec::new(),
    };
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

    // Properties, columnar, once per batch — NOT one `serde_json::Map` per row.
    // A `BTreeMap` allocates a full leaf node (~630 B for `(String, Value)`)
    // even for a single key, and every row also cloned every key name and paid
    // an `Arc`; that alone put peak memory near a kilobyte per feature and is
    // what stopped national-scale point clouds from tiling at all. See
    // [`crate::props`].
    //
    // Indexed by BATCH row, so rows skipped below for unusable geometry simply
    // leave an unread entry — the alternative (indexing by surviving feature)
    // would shear the table against the batch on the first dropped row.
    let prop_names: Vec<String> = property_cols
        .iter()
        .map(|&idx| schema.field(idx).name().clone())
        .collect();
    let prop_table = if prop_names.is_empty() {
        None
    } else {
        let mut builder = crate::props::PropertyTableBuilder::new(prop_names, batch.num_rows());
        for i in 0..batch.num_rows() {
            for (pc, &col_idx) in property_cols.iter().enumerate() {
                let value = extract_property_value(batch, col_idx, i);
                if value.is_some() {
                    seen_props[pc] = true;
                }
                builder.push(pc, value.as_ref());
            }
            builder.end_row();
        }
        Some(Arc::new(builder.finish()))
    };

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

        // Properties come from the batch's columnar table (built above). A row
        // whose every property is null gets `None`, exactly as the old empty
        // map did — an all-null row must stay indistinguishable from a row with
        // no property columns at all.
        let shared_properties = prop_table.as_ref().and_then(|t| {
            t.row_has_values(i).then(|| FeatureProperties::Row {
                table: Arc::clone(t),
                row: i as u32,
            })
        });

        let feature = Feature {
            bbox: None,
            geometry: Some(geometry),
            id: None,
            properties: None,
            foreign_members: None,
        };
        features.push(ParsedFeature {
            home_zoom: None,
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
        // Floats + every signed/unsigned integer width. Every width must be
        // listed: a type that falls through to `None` here silently drops the
        // whole column (e.g. a UInt32 id/count) from every tile.
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

/// Calculate spatial and temporal bounds from features.
///
/// Thin wrapper over [`profile_features`] kept for every existing caller: it
/// returns the same pair it always did, now computed under
/// [`DEFAULT_BOUNDS_MODE`] — which is [`BoundsMode::Vertex`] since R1, so the
/// bbox is the honest, conservative superset rather than the centroid box.
/// Callers that need the vertical extent, the mode actually used, or both boxes
/// side by side want [`profile_features_with`] instead; this two-field view
/// exists only so pre-SH-2 call sites keep compiling.
pub fn calculate_bounds(features: &[ParsedFeature]) -> Result<(BoundingBox, TimeRange)> {
    let profile = profile_features(features)?;
    Ok((profile.bounds, profile.time_range))
}

// =============================================================================
// Feature profiling — honest bounds, temporal extent, vertical extent (SH-2)
// =============================================================================

/// Which geometric quantity an archive's declared `metadata.bounds` is taken
/// from.
///
/// **The defect this exists to close (backlog K11).** The historical bbox is
/// the min/max of [`ParsedFeature::lon`]/[`ParsedFeature::lat`] — the parsed
/// *anchor*, which for every non-point geometry is the **centroid** (see
/// `parse_wkb_geometry`). Tiles, however, are addressed by **vertex**
/// (`tiler::place_non_trajectory` walks `feature.geojson.geometry`). A
/// LineString or ring whose vertices reach past its centroid is therefore not
/// bounded by the number the manifest advertises: the declared bbox provably
/// *under-states* the real extent. Anything that pre-intersects a query box
/// against `metadata.bounds` (tile selection, frustum pre-culling, the
/// showcase's opening camera) is unsound on such an archive — it can discard
/// tiles that really do carry visible data.
///
/// [`BoundsMode::Vertex`] computes the honest quantity. It is a **conservative
/// superset**: every vertex the archive can decode to lies inside it, and it is
/// never tighter than [`BoundsMode::Centroid`] on the same features (a centroid
/// lies in the convex hull of its own vertices, hence inside their bbox).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundsMode {
    /// Legacy: min/max over feature anchors (centroids). Under-states the real
    /// extent for any geometry wider than a point. Kept as the documented
    /// rollback (`stt-build --bounds-mode centroid`) and as the shape every
    /// pre-R1 published manifest carries.
    Centroid,
    /// Honest: min/max over every geometry vertex, falling back to the anchor
    /// for a feature carrying no (usable) geometry. **The default since R1.**
    #[default]
    Vertex,
}

impl BoundsMode {
    /// The value stamped into `metadata.properties[`[`BOUNDS_MODE_PROPERTY`]`]`.
    ///
    /// Lower-case, stable, and compared case-insensitively by the validator —
    /// this string is a cross-crate contract, not a display label.
    pub const fn as_manifest_value(self) -> &'static str {
        match self {
            BoundsMode::Centroid => "centroid",
            BoundsMode::Vertex => "vertex",
        }
    }
}

/// The `metadata.properties` key by which a writer records **which** quantity
/// `metadata.bounds` was taken from.
///
/// ⚠️ **Cross-crate contract.** `stt-validate`'s check 13 reads this exact key
/// (its own `BOUNDS_MODE_PROPERTY`) and treats the value `"vertex"` as an
/// *attestation*: on such an archive a bbox that fails to contain the decoded
/// vertices is an **error**, whereas on a legacy archive (key absent — the whole
/// pre-R1 fleet) the same finding is a warning naming the rebuild. Renaming the
/// key, or emitting a different value spelling, silently demotes every new
/// archive back to warn-only.
///
/// It rides `properties` — the free-form `BTreeMap<String, String>` every
/// manifest already carries — rather than `manifest.capabilities`, because a
/// reader **rejects** an archive declaring a capability it does not implement,
/// so minting one here would make every honest archive unopenable by the
/// deployed fleet's readers. A property is inert to any reader that ignores it.
pub const BOUNDS_MODE_PROPERTY: &str = "bounds_mode";

/// The bounds mode a build uses when the caller does not choose one.
///
/// ⚠️ **This constant is byte-changing and it has now been flipped (R1).** A
/// rebuilt archive's `metadata.bounds` numbers move — they *widen* — which moves
/// the golden manifest pins under `crates/stt-core/tests/fixtures/v2-golden/`.
/// Golden pins are re-blessed exactly once in the 2026-08 optimization program,
/// by work item TB-14 inside rebuild window R1; if a golden test is red because
/// of this flip, that is the expected pin move, **not** a licence to re-bless it
/// anywhere else.
///
/// Why the flip is the fix and not a weakening of the check: the centroid box
/// provably *under-states* the extent of every non-point geometry (K11), and
/// anything that pre-intersects a query box against `metadata.bounds` — tile
/// selection, frustum pre-culling, the showcase's opening camera — then discards
/// tiles that really do carry visible data, with no error anywhere in the stack.
/// The vertex box is the conservative superset those consumers need.
///
/// The legacy quantity is not deleted: `stt-build --bounds-mode centroid` still
/// selects it, and [`FeatureProfile::centroid_bounds`] is computed on every
/// build whatever the mode.
pub const DEFAULT_BOUNDS_MODE: BoundsMode = BoundsMode::Vertex;

// ===========================================================================
// Feature-id SCOPE — the sibling attestation, for check 12's distinct count.
// ===========================================================================

/// Largest feature set this crate will attempt to PROVE pairwise-distinct ids
/// over.
///
/// The proof needs the whole id multiset in memory to look for a collision, so
/// it costs one `u64` per feature (8 B; 40 MB at the cap) plus one sort. That
/// is a rounding error next to the `ParsedFeature` vector it rides — hundreds
/// of bytes per row, held for the whole build — but it is not unbounded, and an
/// unbounded allocation inside a *diagnostic* is not a trade worth making.
///
/// Above the cap the attestation is DECLINED rather than guessed: the archive
/// falls back to the conservative `SourceFeatures` basis, which costs it only a
/// note. An operator who knows their ids are distinct can still assert it with
/// `stt-build --feature-id-scope global`.
pub const FEATURE_ID_ATTESTATION_CAP: usize = 5_000_000;

/// HOW the writer will construct the wire `id` for one source feature.
///
/// # Why this type exists (BLOCKER 1)
///
/// `--feature-id-scope auto` used to ask one question — "does every source
/// feature carry an id?" — and **no ingest path populates
/// `ParsedFeature.geojson.id`**: the GeoParquet reader
/// ([`load_features_from_geoparquet`]'s `Feature { id: None, .. }`), the
/// PostGIS reader and the DuckDB reader all leave it `None`. So the strict
/// basis had no live producer, `auto` resolved to `local` on every archive
/// anyone could build, and check 12's distinct-id comparison was structurally
/// disarmed. Worse, the resulting report *described* the fleet's point archives
/// — "the wire id is the PER-TILE ROW INDEX" — on line, polygon and trip
/// archives where that is simply not what the writer did.
///
/// The writer does not need an operator's assertion, because it already knows
/// which id-construction path it will take. This enum is that fact, and
/// [`feature_id_report`] derives the attestation from it.
///
/// # The four paths, and which are dataset-wide keys
///
/// | construction | where | one id per source feature? |
/// |---|---|---|
/// | [`Source`](FeatureIdConstruction::Source) | `columnar::determine_feature_id` maps `geojson.id` verbatim (integers) or via FNV-1a-64 (strings) | **yes** |
/// | [`AnchorHash`](FeatureIdConstruction::AnchorHash) | `determine_feature_id`'s fallback, `FNV(timestamp, lon, lat)` over the feature ANCHOR | **yes** — the tiler copies the parent anchor into every clipped piece precisely so "id-less pieces hash to the SAME synthetic feature id in every tile" (`tiler::place_polygon`) |
/// | [`RowIndex`](FeatureIdConstruction::RowIndex) | `columnar::build_point_layer` overwrites an id-less point's id with `i as u64` | no — unique only WITHIN a tile |
/// | [`SegmentHash`](FeatureIdConstruction::SegmentHash) | `columnar::segment_feature_id`'s fallback, `FNV(start_time, first coord of the CLIPPED segment)` | no — a fresh id per clipped segment |
///
/// Only the first two make `distinct_feature_count` and the decoded distinct-id
/// count the same quantity, so only those two arm the strict comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureIdConstruction {
    /// The source feature's own id, mapped by `columnar::determine_feature_id`.
    Source,
    /// `FNV(timestamp, anchor lon, anchor lat)` — ONE id per source feature,
    /// copied verbatim into every tile and every zoom the feature reaches.
    AnchorHash,
    /// The **per-tile row index** (`columnar::build_point_layer`). Unique inside
    /// a tile, repeated across tiles: not a key.
    RowIndex,
    /// A fresh hash per CLIPPED SEGMENT (`columnar::segment_feature_id`). One
    /// source trajectory becomes many ids, and the writer cannot enumerate them
    /// before tiling, so it cannot prove anything about them.
    SegmentHash,
}

impl FeatureIdConstruction {
    /// Is this construction a dataset-wide key — exactly one distinct wire id
    /// per source feature, stable across every tile and pyramid level?
    ///
    /// This is the whole gate: `true` here is what lets `stt-validate` compare
    /// `distinct_feature_count` (a count of SOURCE FEATURES) against the ids it
    /// decodes and call a shortfall FEATURE LOSS.
    pub const fn is_dataset_wide_key(self) -> bool {
        matches!(
            self,
            FeatureIdConstruction::Source | FeatureIdConstruction::AnchorHash
        )
    }

    /// The value stamped into
    /// `metadata.properties[`[`FEATURE_ID_CONSTRUCTION_PROPERTY`]`]`.
    ///
    /// ⚠️ Cross-crate contract, compared case-insensitively by
    /// `stt_core::metadata::distinct_id_basis`. These strings are wire values,
    /// not display labels.
    pub const fn as_manifest_value(self) -> &'static str {
        use stt_core::metadata as m;
        match self {
            FeatureIdConstruction::Source => m::FEATURE_ID_CONSTRUCTION_SOURCE,
            FeatureIdConstruction::AnchorHash => m::FEATURE_ID_CONSTRUCTION_ANCHOR_HASH,
            FeatureIdConstruction::RowIndex => m::FEATURE_ID_CONSTRUCTION_ROW_INDEX,
            FeatureIdConstruction::SegmentHash => m::FEATURE_ID_CONSTRUCTION_SEGMENT_HASH,
        }
    }
}

/// Build levers [`feature_id_report`] needs in order to predict the writer's
/// own id construction.
///
/// Exactly one today, and it is load-bearing: trajectory clipping is what turns
/// a duration LineString from one [`AnchorHash`](FeatureIdConstruction::AnchorHash)
/// id into many [`SegmentHash`](FeatureIdConstruction::SegmentHash) ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureIdOptions {
    /// `TileConfig::clip_trajectories` (i.e. `!stt-build --no-clip`). With it
    /// ON, `tiler::place_feature` cuts a duration (multi)LineString into
    /// per-tile/per-bucket segments, each of which mints its own id when the
    /// source carries none. With it OFF the same feature is placed whole and
    /// keeps a single anchor hash.
    pub clip_trajectories: bool,
}

impl Default for FeatureIdOptions {
    /// The builder's own default (`clip_trajectories: true`), which is also the
    /// CONSERVATIVE choice: assuming clipping can only move a trajectory into
    /// the non-key `SegmentHash` class, never out of it.
    fn default() -> Self {
        Self {
            clip_trajectories: true,
        }
    }
}

/// Why [`feature_ids_are_globally_distinct`] did (or did not) conclude that the
/// wire `id` column is a dataset-wide key.
///
/// Every non-[`Distinct`](FeatureIdAttestation::Distinct) variant carries the
/// evidence, so a build log can say which feature stopped the proof rather than
/// just "no".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureIdAttestation {
    /// PROVEN. Every feature's wire id is a dataset-wide key — either its own
    /// source id (mapped verbatim for integers, through the spec-fixed
    /// FNV-1a-64 for strings) or the whole-feature ANCHOR HASH the tiler copies
    /// into every clipped piece — and no two of them collide.
    Distinct,
    /// Declined at feature `index`: it is an id-less POINT, so
    /// `columnar::build_point_layer` overwrites its id with the **per-tile row
    /// index**, which is unique within a tile and repeats across tiles.
    ///
    /// This is the fleet's normal state on point archives and the whole reason
    /// the basis exists. (It is emitted only for the row-index construction —
    /// an id-less LINE or POLYGON keeps a stable anchor hash and is attestable.)
    NoSourceId { index: usize },
    /// Declined at feature `index`: it is an id-less duration (multi)LineString
    /// under trajectory clipping, so `columnar::segment_feature_id` mints a
    /// fresh id per CLIPPED SEGMENT. The writer cannot enumerate those ids
    /// before tiling, so it cannot prove they are a key.
    SegmentIds { index: usize },
    /// Declined at feature `index`: its `Id::Number` is neither a `u64` nor an
    /// `i64` (a JSON float), so `determine_feature_id` falls through to its
    /// positional hash for it exactly as if it carried no id — and the feature's
    /// geometry then puts it in a non-key construction class.
    NonIntegerId { index: usize },
    /// Declined: two source features map to the SAME wire id.
    Collision {
        value: u64,
        first: usize,
        second: usize,
    },
    /// Declined: the feature set is larger than [`FEATURE_ID_ATTESTATION_CAP`],
    /// so the proof was not attempted.
    AboveCap { features: usize },
}

impl FeatureIdAttestation {
    /// Did the proof succeed?
    pub fn is_distinct(&self) -> bool {
        matches!(self, FeatureIdAttestation::Distinct)
    }

    /// One line of evidence for the build log.
    pub fn reason(&self) -> String {
        match self {
            FeatureIdAttestation::Distinct => {
                "every source feature maps to one distinct, dataset-wide wire id".to_string()
            }
            FeatureIdAttestation::NoSourceId { index } => format!(
                "source feature #{index} is an id-less POINT, so the writer overwrites its id \
                 with the PER-TILE ROW INDEX, which repeats across tiles"
            ),
            FeatureIdAttestation::SegmentIds { index } => format!(
                "source feature #{index} is an id-less duration (multi)LineString under \
                 trajectory clipping, so the writer mints a fresh id per CLIPPED SEGMENT — ids \
                 the build cannot enumerate before tiling (build with --no-clip, or give the \
                 source an id column, to make them a key)"
            ),
            FeatureIdAttestation::NonIntegerId { index } => format!(
                "source feature #{index} carries a non-integer numeric id, which the writer \
                 cannot map verbatim — it falls through to the positional hash"
            ),
            FeatureIdAttestation::Collision {
                value,
                first,
                second,
            } => format!(
                "source features #{first} and #{second} both map to wire id {value}, so the id \
                 column is not a key"
            ),
            FeatureIdAttestation::AboveCap { features } => format!(
                "{features} features is above the {FEATURE_ID_ATTESTATION_CAP}-feature \
                 attestation cap, so distinctness was not proven (assert it with \
                 --feature-id-scope global if you know it holds)"
            ),
        }
    }
}

/// FNV-1a-64 over raw bytes — **a deliberate mirror** of `columnar::fnv1a_64`.
///
/// ⚠️ Why a second copy instead of a call. The original is module-private to
/// `columnar`, and this change does not own that file. The algorithm is fixed
/// by spec (offset basis `0xcbf29ce484222325`, prime `0x100000001b3`;
/// <http://www.isthe.com/chongo/tech/comp/fnv/>) precisely so that it can be
/// re-implemented without drift, and
/// `fnv1a_64_matches_the_spec_fixed_constants` pins this copy against
/// independently computed vectors. If the two ever disagree, the attestation
/// would claim distinctness over ids the encoder never wrote — hence the pin.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The wire id `columnar::determine_feature_id` will emit for `feature`, but
/// ONLY when that id is derived from the source — `None` whenever the writer
/// would fall through to its positional hash.
///
/// `None` is not by itself a decline: an id-less LINE or POLYGON falls through
/// to the ANCHOR HASH, which the tiler copies into every clipped piece and is
/// therefore still a dataset-wide key. [`feature_id_construction`] is what
/// decides; this function answers only "did the source supply it?".
fn attestable_feature_id(feature: &ParsedFeature) -> Option<u64> {
    use geojson::feature::Id;
    match feature.geojson.id.as_ref()? {
        Id::Number(num) => num
            .as_u64()
            // Same widening the writer does: a negative i64 reinterpreted as
            // u64 is injective, and a genuine clash with a large u64 id shows
            // up as a Collision below rather than being waved through.
            .or_else(|| num.as_i64().map(|v| v as u64)),
        Id::String(s) => Some(fnv1a_64(s.as_bytes())),
    }
}

/// The whole-feature ANCHOR HASH `columnar::determine_feature_id` falls through
/// to for an id-less feature: `FNV-1a-64(timestamp, lon.to_bits(), lat.to_bits())`
/// over the feature's representative point.
///
/// ⚠️ **A deliberate mirror of `columnar::determine_feature_id`'s fallback**,
/// for the same reason [`fnv1a_64`] is a mirror: the original is module-private
/// to `columnar`, which this change does not own. The field order (timestamp,
/// lon bits, lat bits) and the little-endian folding are what must not drift —
/// `anchor_hash_mirrors_the_writers_fallback_id` pins it against independently
/// computed vectors.
///
/// Why it is a key even though nothing "assigns" it: every per-tile piece the
/// tiler derives from a feature carries the PARENT's `timestamp`/`lon`/`lat`
/// (`tiler::place_polygon`: "Parent's representative point, so id-less pieces
/// hash to the SAME synthetic feature id in every tile"; `place_polyline`'s
/// timeless arm does the same), so one source feature yields exactly one wire
/// id however many tiles and zooms it is replicated into.
fn anchor_feature_id(feature: &ParsedFeature) -> u64 {
    let mut bytes = [0u8; 24];
    bytes[0..8].copy_from_slice(&feature.timestamp.to_le_bytes());
    bytes[8..16].copy_from_slice(&feature.lon.to_bits().to_le_bytes());
    bytes[16..24].copy_from_slice(&feature.lat.to_bits().to_le_bytes());
    fnv1a_64(&bytes)
}

/// Which of the writer's four id-construction paths `feature` will take.
///
/// Mirrors the dispatch in `tiler::place_feature` → `tiler::place_non_trajectory`
/// and `columnar::build_layers_from_features_with`, in that order:
///
/// * a **source id** wins outright. `segment_feature_id` reads the parent id off
///   every clipped segment and `build_point_layer` leaves an explicit id alone,
///   so the id is the same value everywhere regardless of geometry.
/// * otherwise **POINT / MultiPoint / GeometryCollection-of-points** →
///   [`RowIndex`](FeatureIdConstruction::RowIndex). (A MultiPoint's derived
///   pieces are Points, so they land in `build_point_layer` too.)
/// * otherwise a **duration (multi)LineString under trajectory clipping** →
///   [`SegmentHash`](FeatureIdConstruction::SegmentHash).
/// * otherwise → [`AnchorHash`](FeatureIdConstruction::AnchorHash): polygons,
///   timeless lines, and duration lines built `--no-clip`.
///
/// Every disagreement with the real dispatch is resolved toward the NON-key
/// class, so a misprediction can only ever cost enforcement — never manufacture
/// it. Two deliberate over-approximations: a 1-vertex duration LineString is
/// classed `SegmentHash` although `is_clippable_trajectory` would reject it and
/// place it whole, and `place_polyline`'s fully-inside-one-tile fast path can
/// hand a duration line an anchor hash at coarse zooms while clipping it at fine
/// ones. Both decline; neither attests.
pub fn feature_id_construction(
    feature: &ParsedFeature,
    options: FeatureIdOptions,
) -> FeatureIdConstruction {
    use geojson::Value as G;
    if attestable_feature_id(feature).is_some() {
        return FeatureIdConstruction::Source;
    }
    // No geometry at all: `determine_geometry_type` errors and the build fails
    // before any of this matters. Classify conservatively rather than guess.
    let Some(geometry) = feature.geojson.geometry.as_ref() else {
        return FeatureIdConstruction::RowIndex;
    };
    // A GeometryCollection is routed by its FIRST member (columnar's
    // `determine_geometry_type`), so unwrap one level before deciding.
    let value = match &geometry.value {
        G::GeometryCollection(members) => match members.first() {
            Some(first) => &first.value,
            // Empty collection: `determine_geometry_type` calls it a Polygon,
            // but nothing can be extracted from it and the columnar builder
            // drops it. Conservative.
            None => return FeatureIdConstruction::RowIndex,
        },
        other => other,
    };
    match value {
        G::Point(_) | G::MultiPoint(_) => FeatureIdConstruction::RowIndex,
        G::LineString(_) | G::MultiLineString(_) => {
            if options.clip_trajectories && feature.end_timestamp.is_some() {
                FeatureIdConstruction::SegmentHash
            } else {
                FeatureIdConstruction::AnchorHash
            }
        }
        G::Polygon(_) | G::MultiPolygon(_) => FeatureIdConstruction::AnchorHash,
        // A nested GeometryCollection: one unwrap is all `determine_geometry_type`
        // does, and it calls this shape a Polygon. Decline rather than model it.
        G::GeometryCollection(_) => FeatureIdConstruction::RowIndex,
    }
}

/// What one build's id column IS, and whether it can be attested — both derived
/// from the writer's own construction, in a single pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureIdReport {
    /// The construction the whole feature set resolves to. When the set mixes
    /// kinds this is the FIRST non-key kind in index order, and otherwise
    /// [`AnchorHash`](FeatureIdConstruction::AnchorHash) if any feature needed
    /// it, else [`Source`](FeatureIdConstruction::Source) — a total,
    /// index-order-deterministic join.
    pub construction: FeatureIdConstruction,
    /// Whether the ids that construction produces are provably a dataset-wide
    /// key. [`FeatureIdAttestation::Distinct`] iff `construction`
    /// [`is_dataset_wide_key`](FeatureIdConstruction::is_dataset_wide_key) AND
    /// no two features collide.
    pub attestation: FeatureIdAttestation,
}

/// Can this feature set's wire `id` column be attested as a **dataset-wide
/// key** — one distinct id per source feature, stable across tiles and pyramid
/// levels?
///
/// This is the builder half of `stt_core::metadata::DistinctIdBasis`. A `true`
/// here is what licenses `stt-build` to stamp
/// `metadata.properties.feature_id_scope = "global"`, which is what licenses
/// `stt-validate` to compare `distinct_feature_count` against the decoded id
/// count and call a shortfall FEATURE LOSS.
///
/// Uses [`FeatureIdOptions::default`] (clipping ON, the conservative reading);
/// a build that knows its `--no-clip` setting should call [`feature_id_report`].
pub fn feature_ids_are_globally_distinct(features: &[ParsedFeature]) -> bool {
    feature_id_attestation(features).is_distinct()
}

/// [`feature_ids_are_globally_distinct`], with the evidence kept.
pub fn feature_id_attestation(features: &[ParsedFeature]) -> FeatureIdAttestation {
    feature_id_report(features, FeatureIdOptions::default()).attestation
}

/// The construction-aware proof: what the writer's id column IS, and whether it
/// is provably a dataset-wide key.
///
/// # ⭐ Why the question is about CONSTRUCTION, not about a source id column
///
/// The predecessor asked "does every source feature carry an id?", and the
/// answer is structurally NO — no ingest path in this crate populates
/// `ParsedFeature.geojson.id`. The strict basis therefore had no live producer,
/// and a rebuild that silently dropped half a line/polygon dataset validated
/// clean: the per-zoom row floor is loose exactly where clipping replicates a
/// feature across tiles, because the surviving features' extra rows cover for
/// the missing ones.
///
/// [`feature_id_construction`] answers the question the writer can actually
/// answer. Two of its four paths — a source id, and the whole-feature ANCHOR
/// HASH the tiler copies into every clipped piece — put exactly one distinct
/// wire id on each source feature, and those are attestable with no operator
/// assertion anywhere.
///
/// # Shape of the proof
///
/// One pass, and it **bails on the first feature whose construction is not a
/// key** — an id-less point (row index) or an id-less clipped trajectory
/// (per-segment ids). That is the overwhelmingly common case on point archives
/// and the reason a 44 M-point no-id archive costs nothing here. Only if every
/// feature clears that gate does it pay for the pairwise-distinctness check, and
/// only up to [`FEATURE_ID_ATTESTATION_CAP`].
///
/// # What "distinct" buys, precisely
///
/// Every source feature is placed at least once at the deepest tier that
/// survives it, and carries the SAME id into every tile and zoom it reaches. So
/// on a complete decode `distinct decoded ids >= surviving source features`, and
/// — because the ids are proven pairwise distinct — a shortfall against
/// `distinct_feature_count` means features are genuinely absent. Clipping,
/// pyramid replication and LOD tiers cannot move that count in either direction;
/// they only add ROWS, which is why this is tight where the row floor is loose.
///
/// # Determinism
///
/// A pure function of the feature sequence: a sort with a total tiebreak, no
/// hashing container, no RNG, no clock. Two runs over the same input return the
/// same answer, and so does a run over the same input in reverse (the
/// conclusion is; the *evidence indices* name the earliest offending pair in
/// index order).
pub fn feature_id_report(features: &[ParsedFeature], options: FeatureIdOptions) -> FeatureIdReport {
    // An empty set has no colliding pair, and the fingerprint declares 0 —
    // which check 12 skips anyway. Vacuously distinct.
    if features.is_empty() {
        return FeatureIdReport {
            construction: FeatureIdConstruction::Source,
            attestation: FeatureIdAttestation::Distinct,
        };
    }

    let mut ids: Vec<(u64, usize)> = Vec::with_capacity(features.len().min(1024));
    // Did any feature need the anchor-hash fallback? Only used to LABEL the set
    // when every feature is attestable; it never changes the verdict.
    let mut any_anchor = false;
    for (index, feature) in features.iter().enumerate() {
        let construction = feature_id_construction(feature, options);
        // The early bail, on the first NON-KEY construction. Checked BEFORE the
        // cap so a common id-less point archive gets the informative reason
        // rather than "too big", and so the walk stops at feature #0 rather
        // than at the cap.
        if !construction.is_dataset_wide_key() {
            let attestation = match (construction, feature.geojson.id.is_some()) {
                // A non-integer numeric id: the writer could not map it, so the
                // feature fell through to a positional construction. Name the
                // id, which is the actionable half.
                (_, true) => FeatureIdAttestation::NonIntegerId { index },
                (FeatureIdConstruction::SegmentHash, false) => {
                    FeatureIdAttestation::SegmentIds { index }
                }
                (_, false) => FeatureIdAttestation::NoSourceId { index },
            };
            return FeatureIdReport {
                construction,
                attestation,
            };
        }
        let id = match construction {
            FeatureIdConstruction::Source => {
                attestable_feature_id(feature).expect("Source construction implies a mapped id")
            }
            _ => {
                any_anchor = true;
                anchor_feature_id(feature)
            }
        };
        if ids.len() == FEATURE_ID_ATTESTATION_CAP {
            return FeatureIdReport {
                construction: set_construction(any_anchor),
                attestation: FeatureIdAttestation::AboveCap {
                    features: features.len(),
                },
            };
        }
        ids.push((id, index));
    }

    let construction = set_construction(any_anchor);

    // Pairwise distinctness. Sorting by (id, index) makes any duplicate
    // adjacent, and the index tiebreak makes the reported pair the earliest
    // one — a stable total order, so the evidence is reproducible.
    ids.sort_unstable();
    for pair in ids.windows(2) {
        let [(a, first), (b, second)] = [pair[0], pair[1]];
        if a == b {
            return FeatureIdReport {
                construction,
                attestation: FeatureIdAttestation::Collision {
                    value: a,
                    first,
                    second,
                },
            };
        }
    }
    FeatureIdReport {
        construction,
        attestation: FeatureIdAttestation::Distinct,
    }
}

/// The set-level label for a feature set whose every member is attestable: the
/// weaker of the two key constructions wins, so a mixed source/anchor build is
/// reported as `anchor-hash`.
const fn set_construction(any_anchor: bool) -> FeatureIdConstruction {
    if any_anchor {
        FeatureIdConstruction::AnchorHash
    } else {
        FeatureIdConstruction::Source
    }
}

/// Knobs for [`profile_features_with`].
#[derive(Debug, Clone, Copy)]
pub struct FeatureProfileOptions<'a> {
    /// Which quantity [`FeatureProfile::bounds`] reports. Defaults to
    /// [`DEFAULT_BOUNDS_MODE`].
    pub bounds_mode: BoundsMode,
    /// Opt-in property column whose finite numeric values fold into
    /// [`FeatureProfile::z_range`] — for datasets whose altitude lives in a
    /// property rather than in a 3-element geometry position.
    ///
    /// **Metadata only.** It does not touch geometry, matching the standing
    /// preference that render-side depth comes from `elevationProperty` over a
    /// column rather than from rewritten coordinates.
    ///
    /// ⚠️ **A build using `--point-elevation-column` MUST pass that same column
    /// here.** That flag folds the column into POINT z at *encode* time (the
    /// `elevation-fold` capability, `arrow_tile::encode`), long after this pass
    /// runs — so the parsed geometry this profiler walks is still 2D, and the
    /// declared `z_range` would be absent while the tiles decode to 3D. Passing
    /// the column keeps the manifest's vertical claim a superset of what a
    /// reader actually gets.
    ///
    /// ⚠️ …and only for POINT features. The encoder's fold is gated on
    /// `GeometryColumn::Point`, so a line/polygon feature's value contributes
    /// nothing here — see [`elevation_column_is_folded`]. Folding it anyway
    /// would declare vertical extent for an archive whose tiles are flat.
    pub elevation_column: Option<&'a str>,
}

impl Default for FeatureProfileOptions<'_> {
    fn default() -> Self {
        Self {
            bounds_mode: DEFAULT_BOUNDS_MODE,
            elevation_column: None,
        }
    }
}

/// One pass of dataset-global statistics over the ingested source features,
/// computed **before tiling and encode**.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureProfile {
    /// The bbox to declare, per [`FeatureProfileOptions::bounds_mode`].
    pub bounds: BoundingBox,
    /// Temporal extent across every row (including rows whose geometry failed
    /// to parse — a bad geometry does not invalidate a good timestamp).
    pub time_range: TimeRange,
    /// Vertical extent `[min_z, max_z]` when any geometry position carried a
    /// third ordinate, or when [`FeatureProfileOptions::elevation_column`]
    /// named a column with finite numeric values. `None` for a purely 2D
    /// dataset — which is what keeps `metadata.z_range` absent (and manifests
    /// byte-identical) for everything that is not volumetric.
    pub z_range: Option<[f64; 2]>,
    /// The honest, vertex-derived bbox. Always computed, whatever the mode, so
    /// a validator or report can compare the declared bbox against the truth
    /// without a second pass.
    pub vertex_bounds: BoundingBox,
    /// The legacy, centroid-derived bbox. Always computed, so the two can be
    /// diffed (and so the mode switch is a selection, not a recomputation).
    pub centroid_bounds: BoundingBox,
    /// Which of the two [`Self::bounds`] was taken from.
    pub bounds_mode: BoundsMode,
}

/// Profile features under [`FeatureProfileOptions::default`].
///
/// This is the shape SH-1's fingerprint pass shares; [`calculate_bounds`] is
/// the two-field view of the same computation.
pub fn profile_features(features: &[ParsedFeature]) -> Result<FeatureProfile> {
    profile_features_with(features, &FeatureProfileOptions::default())
}

/// Profile features, choosing the bounds quantity and (optionally) folding a
/// property column into the vertical extent.
///
/// **Determinism.** Every statistic is an order-independent fold of `min`/`max`
/// over the input features: no sort, no RNG, no clock, no hash iteration. The
/// result is therefore a pure function of the feature *set*, identical across
/// re-runs and across any permutation of the input — pinned by
/// `bounds_are_order_independent_and_reproducible`.
///
/// **Antimeridian.** The longitude fold is a plain min/max, deliberately
/// unchanged from the legacy behaviour: a dataset straddling ±180° reports a
/// full-width `[-180, 180]`-ish bbox rather than the (narrower, correct)
/// wrapped interval. That is *loose but sound* — the declared box still
/// contains every vertex, which is the direction pre-intersection requires. A
/// wrapped bbox would be tighter and would need a wrap-aware reader on the
/// other side; the recorded antimeridian gotchas stay untouched here, and
/// widening a bbox can never make a superset claim false.
///
/// **Poles.** Latitudes fold unclamped. A vertex at ±90° widens the box to
/// ±90°; nothing is projected here, so the Mercator latitude clamp (and the
/// polar tile-row collapse it exists for) is a tiler concern, not a bounds one.
/// Clamping would *shrink* the declared box below the data, which is exactly
/// the unsoundness this item removes.
///
/// **Non-finite coordinates** (NaN/±inf) are skipped rather than folded — one
/// NaN would otherwise poison a comparison chain, and one `inf` would swallow
/// the globe. A feature that contributes no finite vertex falls back to its
/// anchor, so it is never silently dropped from the extent.
pub fn profile_features_with(
    features: &[ParsedFeature],
    options: &FeatureProfileOptions<'_>,
) -> Result<FeatureProfile> {
    if features.is_empty() {
        // Unchanged empty-input contract: the whole-world placeholder, NOT an
        // inside-out box, and a zero time range.
        let world = BoundingBox::new(-180.0, -90.0, 180.0, 90.0);
        return Ok(FeatureProfile {
            bounds: world,
            time_range: TimeRange::new(0, 0),
            z_range: None,
            vertex_bounds: world,
            centroid_bounds: world,
            bounds_mode: options.bounds_mode,
        });
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
    //
    // The sentinel test stays keyed on the parsed ANCHOR for both modes: a
    // feature whose anchor is the sentinel had its geometry coerced, so its
    // vertices are no more trustworthy than its centroid.
    let mut centroid = ExtentAccumulator::default();
    let mut vertex = ExtentAccumulator::default();
    let mut z = ZAccumulator::default();
    let mut counted = 0usize;
    let mut skipped_null_island = 0usize;
    for f in features {
        if f.lon == 0.0 && f.lat == 0.0 {
            skipped_null_island += 1;
            continue;
        }
        centroid.push_raw(f.lon, f.lat);
        accumulate_feature_extent(f, &mut vertex, &mut z);
        accumulate_elevation_column(f, options.elevation_column, &mut z);
        counted += 1;
    }

    if counted == 0 {
        // Degenerate: every feature sits at the sentinel (e.g. a dataset that
        // genuinely straddles null island, or one that is entirely malformed).
        // Fall back to the raw extent rather than return an inside-out bbox.
        for f in features {
            centroid.push_raw(f.lon, f.lat);
            accumulate_feature_extent(f, &mut vertex, &mut z);
            accumulate_elevation_column(f, options.elevation_column, &mut z);
        }
    } else if skipped_null_island > 0 {
        tracing::warn!(
            "excluded {} null-island (0,0) feature(s) from archive bounds \
             (likely coerced bad/missing geometry)",
            skipped_null_island
        );
    }

    // The centroid box is emitted verbatim — including the inside-out sentinel
    // values a pathological all-non-finite input would leave — so the legacy
    // mode is bit-for-bit what it always was. The vertex box, which is new
    // surface, degrades to the whole-world placeholder instead: a superset is
    // the only honest answer when nothing finite was seen.
    let centroid_bounds = centroid.into_raw_bbox();
    let vertex_bounds = vertex
        .into_bbox()
        .unwrap_or_else(|| BoundingBox::new(-180.0, -90.0, 180.0, 90.0));

    let bounds = match options.bounds_mode {
        BoundsMode::Centroid => centroid_bounds,
        BoundsMode::Vertex => vertex_bounds,
    };

    Ok(FeatureProfile {
        bounds,
        time_range: TimeRange::new(min_time, max_time),
        z_range: z.finish(),
        vertex_bounds,
        centroid_bounds,
        bounds_mode: options.bounds_mode,
    })
}

/// Fold one feature's vertices into the vertex extent and the vertical extent.
///
/// A feature with no geometry — or whose geometry holds no finite position —
/// contributes its anchor instead, so it still widens the box rather than
/// vanishing from it.
fn accumulate_feature_extent(
    f: &ParsedFeature,
    vertex: &mut ExtentAccumulator,
    z: &mut ZAccumulator,
) {
    let before = vertex.count;
    if let Some(geom) = f.geojson.geometry.as_ref() {
        for_each_position(geom, |p| {
            if p.len() >= 2 {
                vertex.push(p[0], p[1]);
            }
            if p.len() >= 3 {
                z.push(p[2]);
            }
        });
    }
    if vertex.count == before {
        vertex.push(f.lon, f.lat);
    }
}

/// Whether the encoder will actually CONSUME `--point-elevation-column` for
/// this feature — i.e. fold its value into the geometry's third ordinate and
/// drop the column from the tile's property set.
///
/// ⚠️ **The answer is "only for points."** `arrow_tile::encode` gates the fold
/// on `matches!(layer.geometry, GeometryColumn::Point(_))`, and the flag's own
/// help says so ("Only affects POINT layers"). Assuming the fold always happens
/// is not a conservative simplification, it is a double falsehood on any
/// line/polygon build: it invents a `z_range` for an archive with no 3D
/// geometry, AND it removes the column from `numeric_ranges` — which disables
/// the very attribute check that catches a corrupted rebuild. Observed on a
/// linestring build as `z_range = [5.0, 34.0]` with the `speed` column silently
/// gone.
///
/// Classification goes through [`crate::columnar::determine_geometry_type`],
/// the same function `build_layers_from_features_with` partitions on, so this
/// predicate cannot drift from the layer the tiler actually builds. Trajectory
/// clipping cannot change the answer either: `is_clippable_trajectory` only
/// ever fires on a `LineString`, so a feature that classifies as `Point` here
/// always lands in a `GeometryColumn::Point` layer.
///
/// A feature whose geometry cannot be classified answers **false** — the safe
/// direction. Guessing "point" would re-create the defect above; guessing
/// "not point" at worst declares a range for a column no tile carries, which
/// the validator reports as a warning.
fn elevation_column_is_folded(f: &ParsedFeature) -> bool {
    matches!(
        crate::columnar::determine_geometry_type(f),
        Ok(stt_core::types::GeometryType::Point)
    )
}

/// Fold the opt-in elevation property column into the vertical extent, for the
/// features whose layer the encoder will actually fold it into.
///
/// Reads through the same `shared_properties` handle the zoom-bound fields use
/// (`tiler::feature_zoom_bound`), so the columnar and owned-map property paths
/// behave identically. Non-numeric and missing values are simply absent — a
/// string that looks like a number is NOT coerced, matching `PropValue::as_f64`.
fn accumulate_elevation_column(f: &ParsedFeature, column: Option<&str>, z: &mut ZAccumulator) {
    let Some(column) = column else {
        return;
    };
    // Non-point features keep the column as a plain property; their tiles ship
    // 2D geometry, so folding their values into `z_range` would claim vertical
    // extent the archive does not have.
    if !elevation_column_is_folded(f) {
        return;
    }
    if let Some(v) = f
        .shared_properties
        .as_ref()
        .and_then(|p| p.get(column))
        .and_then(|v| v.as_f64())
    {
        z.push(v);
    }
}

/// Visit every coordinate position of a GeoJSON geometry, including nested
/// `GeometryCollection` members.
///
/// Iterative (explicit work stack) rather than recursive: a hostile input can
/// nest collections arbitrarily deep, and a bbox must not be a stack-overflow
/// vector. Visit order is unspecified and irrelevant — the only consumers are
/// order-independent min/max folds.
///
/// The stack stays EMPTY (and therefore unallocated) unless a
/// `GeometryCollection` is actually met: this walk runs once per feature, and a
/// 44 M-point archive cannot afford a heap allocation per row.
fn for_each_position(geom: &Geometry, mut visit: impl FnMut(&[f64])) {
    let mut stack: Vec<&Geometry> = Vec::new();
    let mut current = geom;
    loop {
        match &current.value {
            GeomValue::Point(p) => visit(p),
            GeomValue::MultiPoint(ps) | GeomValue::LineString(ps) => {
                for p in ps {
                    visit(p);
                }
            }
            GeomValue::MultiLineString(lines) | GeomValue::Polygon(lines) => {
                for line in lines {
                    for p in line {
                        visit(p);
                    }
                }
            }
            GeomValue::MultiPolygon(polys) => {
                for poly in polys {
                    for ring in poly {
                        for p in ring {
                            visit(p);
                        }
                    }
                }
            }
            GeomValue::GeometryCollection(members) => stack.extend(members.iter()),
        }
        match stack.pop() {
            Some(next) => current = next,
            None => break,
        }
    }
}

/// Order-independent min/max fold over lon/lat pairs.
#[derive(Debug, Default)]
struct ExtentAccumulator {
    min_lon: f64,
    max_lon: f64,
    min_lat: f64,
    max_lat: f64,
    count: usize,
    /// Seeded lazily so an empty accumulator is distinguishable from one that
    /// legitimately saw `(0, 0)`.
    seeded: bool,
}

impl ExtentAccumulator {
    /// Fold a position, ignoring non-finite ordinates.
    fn push(&mut self, lon: f64, lat: f64) {
        if !lon.is_finite() || !lat.is_finite() {
            return;
        }
        if !self.seeded {
            self.min_lon = lon;
            self.max_lon = lon;
            self.min_lat = lat;
            self.max_lat = lat;
            self.seeded = true;
        } else {
            self.min_lon = self.min_lon.min(lon);
            self.max_lon = self.max_lon.max(lon);
            self.min_lat = self.min_lat.min(lat);
            self.max_lat = self.max_lat.max(lat);
        }
        self.count += 1;
    }

    /// Fold a position the LEGACY way: `f64::min`/`f64::max` from the
    /// `MAX`/`MIN` sentinels, non-finite values included. Bit-for-bit the
    /// pre-SH-2 centroid fold, sentinel edge cases and all.
    fn push_raw(&mut self, lon: f64, lat: f64) {
        if !self.seeded {
            self.min_lon = f64::MAX;
            self.max_lon = f64::MIN;
            self.min_lat = f64::MAX;
            self.max_lat = f64::MIN;
            self.seeded = true;
        }
        self.min_lon = self.min_lon.min(lon);
        self.max_lon = self.max_lon.max(lon);
        self.min_lat = self.min_lat.min(lat);
        self.max_lat = self.max_lat.max(lat);
        self.count += 1;
    }

    fn into_bbox(self) -> Option<BoundingBox> {
        if self.count == 0 || !self.seeded {
            return None;
        }
        Some(BoundingBox::new(
            self.min_lon,
            self.min_lat,
            self.max_lon,
            self.max_lat,
        ))
    }

    /// The legacy view: emit whatever the fold holds, sentinels included.
    fn into_raw_bbox(self) -> BoundingBox {
        if !self.seeded {
            return BoundingBox::new(f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        }
        BoundingBox::new(self.min_lon, self.min_lat, self.max_lon, self.max_lat)
    }
}

/// Order-independent min/max fold over elevations.
#[derive(Debug, Default)]
struct ZAccumulator {
    min: f64,
    max: f64,
    seen: bool,
}

impl ZAccumulator {
    fn push(&mut self, z: f64) {
        if !z.is_finite() {
            return;
        }
        if !self.seen {
            self.min = z;
            self.max = z;
            self.seen = true;
        } else {
            self.min = self.min.min(z);
            self.max = self.max.max(z);
        }
    }

    fn finish(self) -> Option<[f64; 2]> {
        self.seen.then_some([self.min, self.max])
    }
}

// =============================================================================
// Semantic content fingerprint — build-time emission (SH-1)
// =============================================================================

/// Knobs [`content_fingerprint`] derives its declared TOLERANCES from.
///
/// Every entry corresponds to a build lever that moves a decoded value away
/// from its source value. Miss one and an honest quantized archive fails
/// validation; declare one the archive did not earn and the validator rejects
/// it (the tolerances are capability-gated on the reading side).
#[derive(Debug, Clone, Default)]
pub struct FingerprintOptions<'a> {
    /// `--quantize-coords METERS`, or `None`/`0` when coordinates ship as
    /// exact Float64.
    pub quantize_coords_m: Option<f64>,
    /// Explicit `--quantize-attr NAME=PREC` pairs. `--quantize-attrs-auto`
    /// needs no entry here: its step is range-adaptive per tile, so no writer
    /// could declare it up front — the validator reads that step off the
    /// column's own `stt:qa` affine instead.
    pub attr_precisions: BTreeMap<String, f64>,
    /// `--point-elevation-column NAME`.
    ///
    /// ⚠️ **Per feature, not per build.** The encoder folds the column into
    /// geometry `z` and removes it from the property set **only for POINT
    /// layers** (`arrow_tile::encode` gates on `GeometryColumn::Point`; the
    /// flag's help says "Only affects POINT layers"). So a point feature's
    /// value is recorded in `z_range` and omitted from `numeric_ranges`, while
    /// a line/polygon feature's value stays an ordinary property and keeps its
    /// declared range. Folding unconditionally invents a `z_range` for a
    /// flat archive and blinds the check on the named column — see
    /// [`elevation_column_is_folded`].
    pub elevation_column: Option<&'a str>,
}

/// Compute the archive's [`ContentFingerprint`] from the ingested source
/// features, before tiling and encode.
///
/// # Why this is a separate walk from [`profile_features_with`]
///
/// `metadata.bounds` is a **presentation** quantity: it deliberately excludes
/// null-island `(0, 0)` features so one coerced row cannot make the showcase
/// open zoomed all the way out. The fingerprint's bbox is a **containment
/// claim** about what the archive decodes to, and excluding data the archive
/// will later be asked to contain would make the claim false. So this pass
/// folds EVERY feature, sentinel rows included, and is otherwise the same
/// order-independent min/max fold.
///
/// # Determinism
///
/// `min`/`max` over finite values, `BTreeMap`/`BTreeSet` throughout, no sort,
/// no RNG, no clock, no sampling stride (the style-hints profiler samples;
/// a containment claim must not). Two builds of one input emit byte-identical
/// `content_fingerprint` JSON — pinned by
/// `content_fingerprint_is_order_independent_and_reproducible`.
pub fn content_fingerprint(
    features: &[ParsedFeature],
    options: &FingerprintOptions<'_>,
) -> Result<ContentFingerprint> {
    let mut vertex = ExtentAccumulator::default();
    let mut z = ZAccumulator::default();
    let mut numeric: BTreeMap<String, [f64; 2]> = BTreeMap::new();
    let mut categorical: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for f in features {
        // Sentinel rows included — see the doc comment.
        accumulate_feature_extent(f, &mut vertex, &mut z);
        accumulate_elevation_column(f, options.elevation_column, &mut z);
        // Whether THIS feature's value is consumed by the elevation fold. The
        // encoder folds only into POINT layers, so on a line/polygon build the
        // column survives as an ordinary property and MUST keep its declared
        // range — dropping it would disable the attribute check on exactly the
        // column an operator singled out.
        let elevation_folded = options.elevation_column.is_some() && elevation_column_is_folded(f);

        let Some(props) = f.shared_properties.as_ref() else {
            continue;
        };
        for (name, value) in props.iter() {
            if elevation_folded && options.elevation_column == Some(name) {
                // Folded into geometry z by the encoder and dropped from the
                // property set, so declaring a numeric range for it would
                // describe a column the archive does not carry.
                continue;
            }
            // Matches the profiler's kind rule exactly: `as_f64` is
            // numbers-only, so a numeric-looking STRING stays categorical.
            if let Some(v) = value.as_f64().filter(|v| v.is_finite()) {
                match numeric.get_mut(name) {
                    Some(slot) => {
                        slot[0] = slot[0].min(v);
                        slot[1] = slot[1].max(v);
                    }
                    None => {
                        numeric.insert(name.to_string(), [v, v]);
                    }
                }
            } else if let Some(s) = value.as_str() {
                let set = categorical.entry(name.to_string()).or_default();
                if set.len() < FINGERPRINT_CARDINALITY_CAP as usize {
                    set.insert(s.to_string());
                }
            }
        }
    }

    let bbox = vertex
        .into_bbox()
        // No finite vertex anywhere: claim the whole world rather than an
        // inside-out box, so the claim stays a (vacuous) superset.
        .unwrap_or_else(|| BoundingBox::new(-180.0, -90.0, 180.0, 90.0));

    let mut column_tolerance: BTreeMap<String, f64> = BTreeMap::new();
    for (column, precision) in &options.attr_precisions {
        if precision.is_finite() && *precision > 0.0 {
            column_tolerance.insert(column.clone(), *precision);
        }
    }

    Ok(ContentFingerprint {
        version: CONTENT_FINGERPRINT_VERSION,
        bbox: [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat],
        z_range: z.finish(),
        distinct_feature_count: features.len() as u64,
        numeric_ranges: numeric,
        categorical_cardinality: categorical
            .into_iter()
            .map(|(k, v)| (k, v.len() as u32))
            .collect(),
        coord_tolerance_deg: coord_quant_step_deg(options.quantize_coords_m.unwrap_or(0.0)),
        column_tolerance,
    })
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
    // common cases.
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

    // ---- property type coverage (no silent column drops) ----

    use crate::columnar::PropertyKind;
    use arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn property_kind_covers_unsigned_narrow_date_and_largeutf8() {
        // A type that maps to `None` here silently drops the whole column from
        // every tile, so each of these must map to a kind.
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

    // ---- coordinate-column narrowing (real x/lat attrs must survive) ----

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

    /// SH-2 / backlog K11 — honest (vertex) manifest bounds and the additive
    /// vertical extent.
    ///
    /// The whole point of these cases is DIRECTION: the declared bbox must be a
    /// conservative SUPERSET of the data's extent. A box that is too tight
    /// makes query-box pre-intersection unsound (it discards tiles that really
    /// do hold visible data); a box that is too loose only costs a wasted
    /// intersection test. Every assertion below therefore checks containment,
    /// not just equality.
    mod feature_profile {
        use crate::input::{
            calculate_bounds, profile_features, profile_features_with, BoundsMode, FeatureProfile,
            FeatureProfileOptions, ParsedFeature, DEFAULT_BOUNDS_MODE,
        };
        use crate::props::FeatureProperties;
        use geojson::{Geometry, Value as G};
        use stt_core::types::BoundingBox;

        /// A feature with an explicit anchor (`lon`/`lat` — what the parsers
        /// fill with the geometry's CENTROID) and an optional geometry.
        fn feature(geometry: Option<G>, lon: f64, lat: f64) -> ParsedFeature {
            ParsedFeature {
                home_zoom: None,
                geojson: geojson::Feature {
                    bbox: None,
                    geometry: geometry.map(Geometry::new),
                    id: None,
                    properties: None,
                    foreign_members: None,
                },
                shared_properties: None,
                timestamp: 1_000,
                end_timestamp: None,
                vertex_timestamps: None,
                vertex_values: None,
                vertex_value_matrix: None,
                lon,
                lat,
            }
        }

        fn with_props(mut f: ParsedFeature, pairs: &[(&str, serde_json::Value)]) -> ParsedFeature {
            let mut map = serde_json::Map::new();
            for (k, v) in pairs {
                map.insert((*k).to_string(), v.clone());
            }
            f.shared_properties = FeatureProperties::from_map(map);
            f
        }

        fn vertex_profile(features: &[ParsedFeature]) -> FeatureProfile {
            profile_features_with(
                features,
                &FeatureProfileOptions {
                    bounds_mode: BoundsMode::Vertex,
                    ..Default::default()
                },
            )
            .unwrap()
        }

        fn contains(b: &BoundingBox, lon: f64, lat: f64) -> bool {
            lon >= b.min_lon && lon <= b.max_lon && lat >= b.min_lat && lat <= b.max_lat
        }

        /// ⚠️ THE BYTE GATE, now flipped (R1). The builder's DEFAULT is the
        /// honest vertex box; a rebuilt archive's `metadata.bounds` widens, and
        /// the golden manifest pins under
        /// `crates/stt-core/tests/fixtures/v2-golden/**` move exactly once for
        /// it — at TB-14, inside rebuild window R1, and nowhere else.
        ///
        /// The direction is the whole point: the new default must be a
        /// **superset** of the old one on the same features, never tighter.
        /// A tighter box is what made query-box pre-intersection unsound.
        #[test]
        fn default_bounds_mode_is_vertex_since_the_r1_rebuild() {
            assert_eq!(DEFAULT_BOUNDS_MODE, BoundsMode::Vertex);
            // One source of truth: `FeatureProfileOptions::default()` reads the
            // constant, and `BoundsMode::default()` must not disagree with it.
            assert_eq!(BoundsMode::default(), DEFAULT_BOUNDS_MODE);
            assert_eq!(
                FeatureProfileOptions::default().bounds_mode,
                DEFAULT_BOUNDS_MODE
            );

            let feats = vec![feature(
                Some(G::LineString(vec![vec![-10.0, -4.0], vec![10.0, 6.0]])),
                0.0,
                1.0,
            )];
            let p = profile_features(&feats).unwrap();
            assert_eq!(p.bounds_mode, BoundsMode::Vertex);
            assert_eq!(p.bounds, p.vertex_bounds);
            // The legacy quantity is still computed, and the new default
            // contains it — a widening, not a different answer.
            assert_eq!(p.centroid_bounds, BoundingBox::new(0.0, 1.0, 0.0, 1.0));
            assert!(p.bounds.min_lon <= p.centroid_bounds.min_lon);
            assert!(p.bounds.min_lat <= p.centroid_bounds.min_lat);
            assert!(p.bounds.max_lon >= p.centroid_bounds.max_lon);
            assert!(p.bounds.max_lat >= p.centroid_bounds.max_lat);

            // The two-field entry point every legacy call site uses now answers
            // the honest box too — this is the byte-changing edge.
            let (bounds, _) = calculate_bounds(&feats).unwrap();
            assert_eq!(bounds, BoundingBox::new(-10.0, -4.0, 10.0, 6.0));

            // …and the documented rollback still yields the pre-R1 number
            // verbatim, so `--bounds-mode centroid` is a real escape hatch.
            let legacy = profile_features_with(
                &feats,
                &FeatureProfileOptions {
                    bounds_mode: BoundsMode::Centroid,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(legacy.bounds, BoundingBox::new(0.0, 1.0, 0.0, 1.0));
        }

        /// The manifest stamp is a cross-crate contract with `stt-validate`'s
        /// check 13: the key and the `vertex` spelling are what promote a
        /// containment failure from "legacy warning" to "error".
        #[test]
        fn bounds_mode_manifest_stamp_matches_the_validator_contract() {
            use crate::input::BOUNDS_MODE_PROPERTY;
            assert_eq!(BOUNDS_MODE_PROPERTY, "bounds_mode");
            assert_eq!(BoundsMode::Vertex.as_manifest_value(), "vertex");
            assert_eq!(BoundsMode::Centroid.as_manifest_value(), "centroid");
        }

        /// The defect itself: a LineString reaches well past its own centroid,
        /// so the centroid box does not bound the geometry the tiler addresses.
        #[test]
        fn linestring_vertices_widen_the_bbox_past_the_centroid_box() {
            let feats = vec![feature(
                Some(G::LineString(vec![vec![-10.0, -4.0], vec![10.0, 6.0]])),
                0.0,
                1.0,
            )];

            let p = vertex_profile(&feats);
            assert_eq!(p.centroid_bounds, BoundingBox::new(0.0, 1.0, 0.0, 1.0));
            assert_eq!(p.vertex_bounds, BoundingBox::new(-10.0, -4.0, 10.0, 6.0));
            assert_eq!(p.bounds, p.vertex_bounds);
            assert_eq!(p.bounds_mode, BoundsMode::Vertex);

            // Strictly wider on all four edges — the centroid box under-stated
            // the extent in every direction.
            assert!(p.vertex_bounds.min_lon < p.centroid_bounds.min_lon);
            assert!(p.vertex_bounds.max_lon > p.centroid_bounds.max_lon);
            assert!(p.vertex_bounds.min_lat < p.centroid_bounds.min_lat);
            assert!(p.vertex_bounds.max_lat > p.centroid_bounds.max_lat);

            // And it really does bound every vertex.
            for v in [(-10.0, -4.0), (10.0, 6.0)] {
                assert!(contains(&p.vertex_bounds, v.0, v.1), "vertex {v:?} escaped");
            }
        }

        #[test]
        fn polygon_ring_vertices_widen_the_bbox() {
            // A closed square ring centred on (5, 5); its centroid anchor sits
            // 5 degrees inside every edge.
            let ring = vec![
                vec![0.0, 0.0],
                vec![10.0, 0.0],
                vec![10.0, 10.0],
                vec![0.0, 10.0],
                vec![0.0, 0.0],
            ];
            let feats = vec![feature(Some(G::Polygon(vec![ring.clone()])), 5.0, 5.0)];

            let p = vertex_profile(&feats);
            assert_eq!(p.centroid_bounds, BoundingBox::new(5.0, 5.0, 5.0, 5.0));
            assert_eq!(p.vertex_bounds, BoundingBox::new(0.0, 0.0, 10.0, 10.0));
            for v in &ring {
                assert!(
                    contains(&p.vertex_bounds, v[0], v[1]),
                    "ring vertex escaped"
                );
            }
        }

        #[test]
        fn multipolygon_and_nested_geometry_collections_are_walked() {
            // A GeometryCollection nesting another collection: the walk is
            // iterative, so depth costs no stack.
            let inner = Geometry::new(G::GeometryCollection(vec![Geometry::new(G::Point(vec![
                -30.0, 40.0,
            ]))]));
            let geom = G::GeometryCollection(vec![
                Geometry::new(G::MultiPolygon(vec![vec![vec![
                    vec![1.0, 1.0],
                    vec![2.0, 1.0],
                    vec![2.0, 2.0],
                    vec![1.0, 1.0],
                ]]])),
                Geometry::new(G::MultiLineString(vec![vec![
                    vec![50.0, -20.0],
                    vec![51.0, -21.0],
                ]])),
                inner,
            ]);
            let feats = vec![feature(Some(geom), 10.0, 10.0)];

            let p = vertex_profile(&feats);
            assert_eq!(p.vertex_bounds, BoundingBox::new(-30.0, -21.0, 51.0, 40.0));
        }

        #[test]
        fn multipoint_vertices_are_walked() {
            let feats = vec![feature(
                Some(G::MultiPoint(vec![vec![-3.0, -3.0], vec![7.0, 9.0]])),
                2.0,
                3.0,
            )];
            assert_eq!(
                vertex_profile(&feats).vertex_bounds,
                BoundingBox::new(-3.0, -3.0, 7.0, 9.0)
            );
        }

        /// Soundness, stated as the property the whole item exists for: the
        /// declared box is a superset of the data, and never tighter than the
        /// legacy one it replaces.
        #[test]
        fn vertex_bbox_is_a_conservative_superset_of_every_vertex_and_of_the_centroid_box() {
            let feats = vec![
                feature(Some(G::Point(vec![12.5, -3.25])), 12.5, -3.25),
                feature(
                    Some(G::LineString(vec![
                        vec![-44.0, 8.0],
                        vec![-40.0, 12.0],
                        vec![-48.0, 3.0],
                    ])),
                    -44.0,
                    7.6,
                ),
                feature(
                    Some(G::Polygon(vec![vec![
                        vec![100.0, -60.0],
                        vec![101.0, -60.0],
                        vec![101.0, -59.0],
                        vec![100.0, -60.0],
                    ]])),
                    100.3,
                    -59.7,
                ),
                feature(None, 5.0, 5.0),
            ];

            let p = vertex_profile(&feats);
            let all_vertices = [
                (12.5, -3.25),
                (-44.0, 8.0),
                (-40.0, 12.0),
                (-48.0, 3.0),
                (100.0, -60.0),
                (101.0, -60.0),
                (101.0, -59.0),
                (5.0, 5.0), // the geometry-less feature contributes its anchor
            ];
            for (lon, lat) in all_vertices {
                assert!(
                    contains(&p.vertex_bounds, lon, lat),
                    "({lon}, {lat}) escaped the declared bbox — pre-intersection would be unsound"
                );
            }

            // Never tighter than the box it replaces, on any edge.
            let c = p.centroid_bounds;
            let v = p.vertex_bounds;
            assert!(v.min_lon <= c.min_lon && v.min_lat <= c.min_lat);
            assert!(v.max_lon >= c.max_lon && v.max_lat >= c.max_lat);
        }

        /// Antimeridian: the longitude fold stays a plain min/max, so a dataset
        /// straddling ±180° reports the LOOSE full-width box, exactly as today.
        /// Loose is sound (it still contains every vertex); tight-but-wrapped
        /// would need a wrap-aware consumer and is deliberately not attempted
        /// here.
        #[test]
        fn antimeridian_crossing_keeps_the_loose_full_width_bbox() {
            let feats = vec![
                feature(Some(G::Point(vec![179.9, 10.0])), 179.9, 10.0),
                feature(Some(G::Point(vec![-179.9, 11.0])), -179.9, 11.0),
            ];
            let p = vertex_profile(&feats);
            assert_eq!(p.vertex_bounds, BoundingBox::new(-179.9, 10.0, 179.9, 11.0));
            // The claim that matters: both vertices are inside it.
            assert!(contains(&p.vertex_bounds, 179.9, 10.0));
            assert!(contains(&p.vertex_bounds, -179.9, 11.0));
            // Same convention as the legacy centroid box — no regression, no
            // new wrapping semantics.
            assert_eq!(p.centroid_bounds, p.vertex_bounds);
        }

        /// Poles: latitudes are NOT clamped to the Mercator limit. Clamping
        /// would shrink the declared box below the data — the exact unsoundness
        /// this item removes. (The projection clamp is a tiler concern.)
        #[test]
        fn polar_latitudes_are_not_clamped_to_the_mercator_limit() {
            let feats = vec![feature(
                Some(G::LineString(vec![vec![0.0, -89.9], vec![1.0, 89.9]])),
                0.5,
                0.0,
            )];
            let p = vertex_profile(&feats);
            assert_eq!(p.vertex_bounds.min_lat, -89.9);
            assert_eq!(p.vertex_bounds.max_lat, 89.9);
        }

        #[test]
        fn z_range_comes_from_three_element_positions() {
            let feats = vec![
                feature(
                    Some(G::LineString(vec![
                        vec![0.0, 0.5, 120.0],
                        vec![1.0, 1.5, -7.5],
                    ])),
                    0.5,
                    1.0,
                ),
                feature(Some(G::Point(vec![2.0, 2.0, 4000.0])), 2.0, 2.0),
            ];
            let p = vertex_profile(&feats);
            assert_eq!(p.z_range, Some([-7.5, 4000.0]));
        }

        #[test]
        fn z_range_is_absent_for_a_purely_2d_dataset() {
            let feats = vec![feature(Some(G::Point(vec![1.0, 2.0])), 1.0, 2.0)];
            assert_eq!(vertex_profile(&feats).z_range, None);
            assert_eq!(profile_features(&feats).unwrap().z_range, None);
        }

        #[test]
        fn elevation_column_folds_into_the_z_range() {
            let feats = vec![
                with_props(
                    feature(Some(G::Point(vec![1.0, 2.0])), 1.0, 2.0),
                    &[("alt_m", serde_json::json!(35.5))],
                ),
                with_props(
                    feature(Some(G::Point(vec![1.5, 2.5])), 1.5, 2.5),
                    &[("alt_m", serde_json::json!(-12))],
                ),
                // No such property: contributes nothing rather than a zero.
                feature(Some(G::Point(vec![1.7, 2.7])), 1.7, 2.7),
                // Present but not a number: NOT coerced.
                with_props(
                    feature(Some(G::Point(vec![1.8, 2.8])), 1.8, 2.8),
                    &[("alt_m", serde_json::json!("900"))],
                ),
            ];

            let p = profile_features_with(
                &feats,
                &FeatureProfileOptions {
                    bounds_mode: BoundsMode::Vertex,
                    elevation_column: Some("alt_m"),
                },
            )
            .unwrap();
            assert_eq!(p.z_range, Some([-12.0, 35.5]));

            // Opt-in: without the flag the same features stay 2D, and the
            // geometry is untouched either way.
            assert_eq!(vertex_profile(&feats).z_range, None);
            assert_eq!(p.vertex_bounds, vertex_profile(&feats).vertex_bounds);
        }

        /// The same per-feature gate the fingerprint uses, on the profiler that
        /// feeds `metadata.z_range`: a line/polygon build declares no vertical
        /// extent from a property column the encoder never folds. Both claims
        /// come from one predicate, so they cannot drift apart.
        #[test]
        fn elevation_column_does_not_fold_on_non_point_geometry() {
            let feats = vec![
                with_props(
                    feature(
                        Some(G::LineString(vec![vec![1.0, 2.0], vec![1.5, 2.5]])),
                        1.25,
                        2.25,
                    ),
                    &[("alt_m", serde_json::json!(35.5))],
                ),
                with_props(
                    feature(
                        Some(G::Polygon(vec![vec![
                            vec![0.0, 0.0],
                            vec![1.0, 0.0],
                            vec![1.0, 1.0],
                            vec![0.0, 0.0],
                        ]])),
                        0.5,
                        0.5,
                    ),
                    &[("alt_m", serde_json::json!(-12.0))],
                ),
            ];
            let p = profile_features_with(
                &feats,
                &FeatureProfileOptions {
                    bounds_mode: BoundsMode::Vertex,
                    elevation_column: Some("alt_m"),
                },
            )
            .unwrap();
            assert_eq!(
                p.z_range, None,
                "the encoder folds only into POINT layers, so these tiles stay flat"
            );
        }

        #[test]
        fn elevation_column_and_geometry_z_combine() {
            let feats = vec![with_props(
                feature(Some(G::Point(vec![1.0, 2.0, 10.0])), 1.0, 2.0),
                &[("alt_m", serde_json::json!(-500.0))],
            )];
            let p = profile_features_with(
                &feats,
                &FeatureProfileOptions {
                    bounds_mode: BoundsMode::Vertex,
                    elevation_column: Some("alt_m"),
                },
            )
            .unwrap();
            assert_eq!(p.z_range, Some([-500.0, 10.0]));
        }

        /// Behavioural pin: the null-island sentinel policy is load-bearing —
        /// dropping it re-opens the whole-globe-bbox failure the current code
        /// documents. It skips the feature's WHOLE contribution, vertices
        /// included, because a sentinel anchor means the geometry was coerced.
        #[test]
        fn null_island_sentinel_features_are_excluded_from_both_bboxes() {
            let feats = vec![
                feature(Some(G::Point(vec![10.0, 20.0])), 10.0, 20.0),
                feature(Some(G::Point(vec![11.0, 21.0])), 11.0, 21.0),
                // Coerced row: anchor at the sentinel, geometry equally bogus.
                feature(Some(G::Point(vec![0.0, 0.0])), 0.0, 0.0),
            ];
            let p = vertex_profile(&feats);
            assert_eq!(p.vertex_bounds, BoundingBox::new(10.0, 20.0, 11.0, 21.0));
            assert_eq!(p.centroid_bounds, BoundingBox::new(10.0, 20.0, 11.0, 21.0));
        }

        /// The degenerate all-sentinel fallback: rather than return an
        /// inside-out bbox, fall back to the raw extent. The centroid answer is
        /// pinned unchanged; the vertex answer is the honest version of it.
        #[test]
        fn all_sentinel_input_falls_back_to_the_raw_extent() {
            let feats = vec![
                feature(
                    Some(G::LineString(vec![vec![-1.0, -1.0], vec![1.0, 1.0]])),
                    0.0,
                    0.0,
                ),
                feature(Some(G::Point(vec![0.0, 0.0])), 0.0, 0.0),
            ];
            let p = vertex_profile(&feats);
            assert_eq!(p.centroid_bounds, BoundingBox::new(0.0, 0.0, 0.0, 0.0));
            assert_eq!(p.vertex_bounds, BoundingBox::new(-1.0, -1.0, 1.0, 1.0));

            // The default (vertex since R1) takes the honest branch of the same
            // fallback; the legacy branch is still reachable and unchanged.
            let (default_bounds, _) = calculate_bounds(&feats).unwrap();
            assert_eq!(default_bounds, BoundingBox::new(-1.0, -1.0, 1.0, 1.0));
            assert_eq!(
                profile_features_with(
                    &feats,
                    &FeatureProfileOptions {
                        bounds_mode: BoundsMode::Centroid,
                        ..Default::default()
                    }
                )
                .unwrap()
                .bounds,
                BoundingBox::new(0.0, 0.0, 0.0, 0.0)
            );
        }

        #[test]
        fn empty_input_returns_the_world_placeholder_unchanged() {
            let (bounds, time_range) = calculate_bounds(&[]).unwrap();
            assert_eq!(bounds, BoundingBox::new(-180.0, -90.0, 180.0, 90.0));
            assert_eq!(time_range.start, 0);
            assert_eq!(time_range.end, 0);

            let p = vertex_profile(&[]);
            assert_eq!(p.bounds, BoundingBox::new(-180.0, -90.0, 180.0, 90.0));
            assert_eq!(p.vertex_bounds, p.bounds);
            assert_eq!(p.z_range, None);
        }

        #[test]
        fn feature_without_geometry_contributes_its_anchor() {
            let feats = vec![
                feature(None, -20.0, -30.0),
                feature(Some(G::Point(vec![1.0, 1.0])), 1.0, 1.0),
            ];
            let p = vertex_profile(&feats);
            assert_eq!(p.vertex_bounds, BoundingBox::new(-20.0, -30.0, 1.0, 1.0));
        }

        /// A NaN/inf ordinate must neither poison the fold nor erase the
        /// feature: it falls back to the anchor, which still widens the box.
        #[test]
        fn non_finite_vertices_fall_back_to_the_anchor() {
            let feats = vec![
                feature(
                    Some(G::LineString(vec![
                        vec![f64::NAN, 5.0],
                        vec![f64::INFINITY, f64::NAN],
                    ])),
                    -7.0,
                    -8.0,
                ),
                feature(Some(G::Point(vec![1.0, 1.0])), 1.0, 1.0),
            ];
            let p = vertex_profile(&feats);
            assert_eq!(p.vertex_bounds, BoundingBox::new(-7.0, -8.0, 1.0, 1.0));
            assert!(p.vertex_bounds.min_lon.is_finite());
            assert_eq!(p.z_range, None);
        }

        /// Determinism (§13.1): the profile is a pure, order-independent fold —
        /// identical across re-runs AND across any permutation of the input, so
        /// two builds of the same data declare byte-identical bounds.
        #[test]
        fn bounds_are_order_independent_and_reproducible() {
            let mut feats = vec![
                feature(
                    Some(G::LineString(vec![
                        vec![-44.0, 8.0, 15.0],
                        vec![-40.0, 12.0, 22.0],
                    ])),
                    -42.0,
                    10.0,
                ),
                feature(Some(G::Point(vec![12.5, -3.25, -4.0])), 12.5, -3.25),
                feature(
                    Some(G::Polygon(vec![vec![
                        vec![100.0, -60.0],
                        vec![101.0, -60.0],
                        vec![101.0, -59.0],
                        vec![100.0, -60.0],
                    ]])),
                    100.3,
                    -59.7,
                ),
                feature(None, 5.0, 5.0),
            ];

            let first = vertex_profile(&feats);
            // Re-run: same input, same answer.
            assert_eq!(vertex_profile(&feats), first);

            // Permutation: reverse, then rotate — a deterministic shuffle that
            // touches every position.
            feats.reverse();
            feats.rotate_left(1);
            assert_eq!(vertex_profile(&feats), first);
            feats.rotate_left(2);
            assert_eq!(vertex_profile(&feats), first);

            // The two-field view is order-independent too, and (since R1) it
            // answers the same honest box.
            assert_eq!(calculate_bounds(&feats).unwrap().0, first.vertex_bounds);
        }
    }

    /// BLOCKER A — the feature-id SCOPE attestation.
    ///
    /// `stt-validate` may only compare `distinct_feature_count` against decoded
    /// ids on an archive whose writer proved the id column is a dataset-wide
    /// key. These tests pin what that proof accepts, what it declines, and that
    /// declining is cheap.
    mod feature_id_scope {
        use crate::input::{
            attestable_feature_id, feature_id_attestation, feature_ids_are_globally_distinct,
            fnv1a_64, FeatureIdAttestation, ParsedFeature,
        };
        use geojson::feature::Id;
        use geojson::{Geometry, Value as G};

        fn feature(id: Option<Id>, lon: f64, lat: f64) -> ParsedFeature {
            ParsedFeature {
                home_zoom: None,
                geojson: geojson::Feature {
                    bbox: None,
                    geometry: Some(Geometry::new(G::Point(vec![lon, lat]))),
                    id,
                    properties: None,
                    foreign_members: None,
                },
                shared_properties: None,
                timestamp: 1_000,
                end_timestamp: None,
                vertex_timestamps: None,
                vertex_values: None,
                vertex_value_matrix: None,
                lon,
                lat,
            }
        }

        fn numeric(n: u64) -> Option<Id> {
            Some(Id::Number(serde_json::Number::from(n)))
        }

        fn string(s: &str) -> Option<Id> {
            Some(Id::String(s.to_string()))
        }

        /// ⚠️ The load-bearing pin. This crate carries its OWN copy of
        /// FNV-1a-64 (the original is module-private to `columnar`, which this
        /// change does not own). If the copy drifts, the attestation would
        /// prove distinctness over ids the encoder never wrote.
        ///
        /// The expectations are the FNV-1a-64 reference vectors from the
        /// algorithm's own specification, not values read back out of this
        /// implementation.
        #[test]
        fn fnv1a_64_matches_the_spec_fixed_constants() {
            assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
            assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
            assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
            // One more, computed off-tree from the same spec, so a
            // single mistyped digit above cannot pass unnoticed.
            assert_eq!(fnv1a_64(b"quake-1"), 0xb832_28a8_e922_7888);
        }

        /// The happy path: distinct explicit ids, numeric and string alike.
        #[test]
        fn distinct_explicit_ids_are_attested() {
            let numeric_ids = vec![
                feature(numeric(7), 0.0, 0.0),
                feature(numeric(9), 1.0, 1.0),
                feature(numeric(11), 2.0, 2.0),
            ];
            assert_eq!(
                feature_id_attestation(&numeric_ids),
                FeatureIdAttestation::Distinct
            );
            assert!(feature_ids_are_globally_distinct(&numeric_ids));

            let string_ids = vec![
                feature(string("quake-1"), 0.0, 0.0),
                feature(string("quake-2"), 1.0, 1.0),
            ];
            assert!(feature_ids_are_globally_distinct(&string_ids));
            // …and through the writer's mapping, not the raw string.
            assert_eq!(
                attestable_feature_id(&string_ids[0]),
                Some(fnv1a_64(b"quake-1"))
            );

            // Vacuous, not an error.
            assert!(feature_ids_are_globally_distinct(&[]));
        }

        /// ⭐ THE BUG'S SHAPE. One id-less feature is enough to decline —
        /// and it declines AT THAT FEATURE, which is what keeps a 44 M-point
        /// no-id archive free.
        #[test]
        fn one_id_less_feature_declines_the_whole_set_immediately() {
            let feats = vec![
                feature(None, 0.0, 0.0),
                feature(numeric(9), 1.0, 1.0),
                feature(numeric(11), 2.0, 2.0),
            ];
            assert_eq!(
                feature_id_attestation(&feats),
                FeatureIdAttestation::NoSourceId { index: 0 }
            );
            assert!(!feature_ids_are_globally_distinct(&feats));

            // Mixed the other way round: the bail names the offending index.
            let mut later = feats.clone();
            later.swap(0, 2);
            assert_eq!(
                feature_id_attestation(&later),
                FeatureIdAttestation::NoSourceId { index: 2 }
            );
            assert!(feature_id_attestation(&later).reason().contains("#2"));
        }

        /// A non-integer numeric id is NOT attestable: the writer's
        /// `determine_feature_id` falls through to the positional hash for it,
        /// exactly as if the feature carried no id.
        #[test]
        fn non_integer_numeric_ids_are_not_attestable() {
            let float_id = feature(
                Some(Id::Number(serde_json::Number::from_f64(1.5).unwrap())),
                0.0,
                0.0,
            );
            assert_eq!(attestable_feature_id(&float_id), None);
            assert_eq!(
                feature_id_attestation(std::slice::from_ref(&float_id)),
                FeatureIdAttestation::NonIntegerId { index: 0 }
            );

            // A negative INTEGER id is attestable — the writer widens it the
            // same way, and the widening is injective.
            let negative = feature(Some(Id::Number(serde_json::Number::from(-3i64))), 0.0, 0.0);
            assert_eq!(attestable_feature_id(&negative), Some(-3i64 as u64));
            assert!(feature_ids_are_globally_distinct(std::slice::from_ref(
                &negative
            )));
        }

        /// Duplicate ids mean the id column is not a key, so it cannot be
        /// compared against a source-feature COUNT. The evidence names the
        /// earliest pair, deterministically.
        #[test]
        fn colliding_ids_decline_and_name_the_pair() {
            let feats = vec![
                feature(numeric(7), 0.0, 0.0),
                feature(numeric(9), 1.0, 1.0),
                feature(numeric(7), 2.0, 2.0),
                feature(numeric(7), 3.0, 3.0),
            ];
            assert_eq!(
                feature_id_attestation(&feats),
                FeatureIdAttestation::Collision {
                    value: 7,
                    first: 0,
                    second: 2,
                }
            );
            assert!(!feature_ids_are_globally_distinct(&feats));

            // Repeat runs agree — a sort with a total tiebreak, no hashing
            // container, no iteration-order dependence.
            for _ in 0..4 {
                assert_eq!(
                    feature_id_attestation(&feats),
                    FeatureIdAttestation::Collision {
                        value: 7,
                        first: 0,
                        second: 2,
                    }
                );
            }
        }

        /// The trajectory shape that used to false-positive the old check:
        /// every point of one track shares the track's id. Distinct SOURCE
        /// FEATURES, one id — so the attestation declines and check 12 stays on
        /// the conservative basis.
        #[test]
        fn one_id_per_track_is_declined_not_attested() {
            let feats: Vec<ParsedFeature> = (0..8)
                .map(|i| feature(string("track-A"), i as f64, i as f64))
                .collect();
            assert!(!feature_ids_are_globally_distinct(&feats));
        }

        // -------------------------------------------------------------------
        // ⭐ BLOCKER 1 — the CONSTRUCTION classifier and the proof built on it.
        // -------------------------------------------------------------------

        use crate::input::{
            anchor_feature_id, feature_id_construction, feature_id_report, FeatureIdConstruction,
            FeatureIdOptions,
        };

        /// The same feature, with the geometry and duration swapped in — the two
        /// axes the classifier actually reads.
        fn shaped(
            id: Option<Id>,
            geometry: G,
            end_timestamp: Option<u64>,
            lon: f64,
        ) -> ParsedFeature {
            ParsedFeature {
                home_zoom: None,
                geojson: geojson::Feature {
                    bbox: None,
                    geometry: Some(Geometry::new(geometry)),
                    id,
                    properties: None,
                    foreign_members: None,
                },
                shared_properties: None,
                timestamp: 1_000,
                end_timestamp,
                vertex_timestamps: None,
                vertex_values: None,
                vertex_value_matrix: None,
                lon,
                lat: 0.0,
            }
        }

        fn line() -> G {
            G::LineString(vec![vec![0.0, 0.0], vec![1.0, 1.0]])
        }
        fn poly() -> G {
            G::Polygon(vec![vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
            ]])
        }

        /// ⚠️ THE MIRROR PIN, and the sibling of
        /// `fnv1a_64_matches_the_spec_fixed_constants`. [`anchor_feature_id`]
        /// re-implements `columnar::determine_feature_id`'s fallback, which is
        /// module-private to a file this change does not own. If the field order
        /// or the folding drifts, the proof would establish distinctness over
        /// ids the encoder never wrote — and the validator would then read a
        /// shortfall as feature loss on a healthy archive.
        ///
        /// The expectation is computed here from the spec constants
        /// independently of the implementation.
        #[test]
        fn anchor_hash_mirrors_the_writers_fallback_id() {
            let f = shaped(None, poly(), None, -122.394);
            // FNV-1a-64 over LE(timestamp) ++ LE(lon.to_bits()) ++ LE(lat.to_bits()).
            let mut expected = Vec::new();
            expected.extend_from_slice(&1_000u64.to_le_bytes());
            expected.extend_from_slice(&(-122.394f64).to_bits().to_le_bytes());
            expected.extend_from_slice(&0.0f64.to_bits().to_le_bytes());
            assert_eq!(anchor_feature_id(&f), fnv1a_64(&expected));

            // It is a function of the ANCHOR, so two features sharing a
            // timestamp and a representative point collide — which is exactly
            // what the pairwise check below has to catch.
            let twin = shaped(None, line(), None, -122.394);
            assert_eq!(anchor_feature_id(&f), anchor_feature_id(&twin));
        }

        /// The classifier's full table. Each row is a real writer path; the two
        /// key rows are the ones that were unreachable before.
        #[test]
        fn construction_classifier_covers_every_writer_path() {
            let clipping = FeatureIdOptions {
                clip_trajectories: true,
            };
            let no_clip = FeatureIdOptions {
                clip_trajectories: false,
            };
            let k = |f: &ParsedFeature, o| feature_id_construction(f, o);

            // A source id wins outright, whatever the geometry: `build_point_layer`
            // leaves it alone and `segment_feature_id` reads it off every segment.
            for geom in [G::Point(vec![0.0, 0.0]), line(), poly()] {
                let f = shaped(numeric(7), geom, Some(2_000), 0.0);
                assert_eq!(k(&f, clipping), FeatureIdConstruction::Source);
            }

            // Id-less points (and multipoints, whose derived pieces are points)
            // get the per-tile ROW INDEX.
            for geom in [
                G::Point(vec![0.0, 0.0]),
                G::MultiPoint(vec![vec![0.0, 0.0], vec![1.0, 1.0]]),
            ] {
                let f = shaped(None, geom, None, 0.0);
                assert_eq!(k(&f, clipping), FeatureIdConstruction::RowIndex);
            }

            // ⭐ Id-less polygons and TIMELESS lines keep the whole-feature
            // anchor hash — the class the row floor cannot police and the whole
            // reason this change exists.
            for geom in [
                line(),
                poly(),
                G::MultiLineString(vec![vec![vec![0.0, 0.0], vec![1.0, 1.0]]]),
            ] {
                let f = shaped(None, geom, None, 0.0);
                assert_eq!(k(&f, clipping), FeatureIdConstruction::AnchorHash);
            }

            // A DURATION line under clipping mints per-segment ids…
            let trip = shaped(None, line(), Some(2_000), 0.0);
            assert_eq!(k(&trip, clipping), FeatureIdConstruction::SegmentHash);
            // …and the very same feature placed whole does not. The decline is a
            // property of the CONSTRUCTION, not of the geometry kind.
            assert_eq!(k(&trip, no_clip), FeatureIdConstruction::AnchorHash);

            // Degenerate shapes resolve toward the NON-key class: a
            // misprediction may cost enforcement, never manufacture it.
            let no_geometry = ParsedFeature {
                home_zoom: None,
                geojson: geojson::Feature {
                    bbox: None,
                    geometry: None,
                    id: None,
                    properties: None,
                    foreign_members: None,
                },
                ..shaped(None, poly(), None, 0.0)
            };
            assert_eq!(k(&no_geometry, clipping), FeatureIdConstruction::RowIndex);
            let empty_collection = shaped(None, G::GeometryCollection(vec![]), None, 0.0);
            assert_eq!(
                k(&empty_collection, clipping),
                FeatureIdConstruction::RowIndex
            );
            // A GeometryCollection routes by its FIRST member, like columnar's
            // `determine_geometry_type`.
            let gc_poly = shaped(
                None,
                G::GeometryCollection(vec![Geometry::new(poly())]),
                None,
                0.0,
            );
            assert_eq!(k(&gc_poly, clipping), FeatureIdConstruction::AnchorHash);
        }

        /// ⭐ THE REGRESSION ITSELF. No ingest path populates `geojson.id`, so
        /// an id-less line/polygon set is what a real build sees — and it must
        /// now ATTEST, where the predecessor declined every archive anyone
        /// could build.
        #[test]
        fn id_less_lines_and_polygons_attest_where_points_decline() {
            let opts = FeatureIdOptions::default();

            // Distinct anchors (distinct longitudes) ⇒ distinct ids ⇒ proven.
            for geom in [line as fn() -> G, poly as fn() -> G] {
                let feats: Vec<ParsedFeature> = (0..16)
                    .map(|i| shaped(None, geom(), None, i as f64))
                    .collect();
                let report = feature_id_report(&feats, opts);
                assert_eq!(report.construction, FeatureIdConstruction::AnchorHash);
                assert_eq!(report.attestation, FeatureIdAttestation::Distinct);
            }

            // The point control, unchanged: still the row index, still declined,
            // and still declined AT feature #0 so a 44 M-point archive pays
            // nothing for the walk.
            let points: Vec<ParsedFeature> =
                (0..16).map(|i| feature(None, i as f64, 0.0)).collect();
            let report = feature_id_report(&points, opts);
            assert_eq!(report.construction, FeatureIdConstruction::RowIndex);
            assert_eq!(
                report.attestation,
                FeatureIdAttestation::NoSourceId { index: 0 }
            );

            // A clipped trip declines with its OWN evidence, naming the segment
            // mechanism rather than the point one.
            let trips: Vec<ParsedFeature> = (0..4)
                .map(|i| shaped(None, line(), Some(2_000), i as f64))
                .collect();
            let report = feature_id_report(&trips, opts);
            assert_eq!(report.construction, FeatureIdConstruction::SegmentHash);
            assert_eq!(
                report.attestation,
                FeatureIdAttestation::SegmentIds { index: 0 }
            );
            assert!(report.attestation.reason().contains("CLIPPED SEGMENT"));
            // …and `--no-clip` moves the identical features into the key class.
            let report = feature_id_report(
                &trips,
                FeatureIdOptions {
                    clip_trajectories: false,
                },
            );
            assert_eq!(report.construction, FeatureIdConstruction::AnchorHash);
            assert_eq!(report.attestation, FeatureIdAttestation::Distinct);
        }

        /// ⚠️ The soundness guard on the new arming path. The anchor hash is a
        /// function of `(timestamp, lon, lat)`, so two features sharing all
        /// three map to ONE wire id — and the archive would then decode fewer
        /// distinct ids than it has features with nothing missing. The pairwise
        /// check must catch that BEFORE the manifest claims a key.
        #[test]
        fn colliding_anchors_decline_so_the_new_path_cannot_false_positive() {
            let feats = vec![
                shaped(None, poly(), None, 10.0),
                shaped(None, poly(), None, 11.0),
                // Same timestamp, same anchor as #0: the two polygons are
                // different, their wire ids are not.
                shaped(None, line(), None, 10.0),
            ];
            let report = feature_id_report(&feats, FeatureIdOptions::default());
            assert_eq!(report.construction, FeatureIdConstruction::AnchorHash);
            assert!(
                matches!(
                    report.attestation,
                    FeatureIdAttestation::Collision {
                        first: 0,
                        second: 2,
                        ..
                    }
                ),
                "a colliding anchor pair must decline, and name the earliest pair: {:?}",
                report.attestation
            );

            // Deterministic: the sort carries a total tiebreak, so repeat runs
            // and a reversed input agree on the conclusion.
            for _ in 0..4 {
                assert_eq!(
                    feature_id_report(&feats, FeatureIdOptions::default()).attestation,
                    report.attestation
                );
            }
        }

        /// A MIXED set is judged by its weakest member, and the label follows.
        #[test]
        fn a_mixed_set_is_judged_by_its_weakest_construction() {
            let opts = FeatureIdOptions::default();

            // One id-less point among polygons ⇒ the whole set declines, at
            // that feature.
            let mixed = vec![
                shaped(None, poly(), None, 1.0),
                shaped(None, poly(), None, 2.0),
                feature(None, 3.0, 0.0),
                shaped(None, poly(), None, 4.0),
            ];
            let report = feature_id_report(&mixed, opts);
            assert_eq!(report.construction, FeatureIdConstruction::RowIndex);
            assert_eq!(
                report.attestation,
                FeatureIdAttestation::NoSourceId { index: 2 }
            );

            // Source ids everywhere ⇒ labelled `source`; one anchor among them
            // ⇒ labelled `anchor-hash` (the weaker of the two key kinds), and
            // both still attest.
            let sourced = vec![
                feature(numeric(1), 0.0, 0.0),
                shaped(numeric(2), poly(), None, 1.0),
            ];
            let report = feature_id_report(&sourced, opts);
            assert_eq!(report.construction, FeatureIdConstruction::Source);
            assert_eq!(report.attestation, FeatureIdAttestation::Distinct);

            let blended = vec![
                feature(numeric(1), 0.0, 0.0),
                shaped(None, poly(), None, 1.0),
            ];
            let report = feature_id_report(&blended, opts);
            assert_eq!(report.construction, FeatureIdConstruction::AnchorHash);
            assert_eq!(report.attestation, FeatureIdAttestation::Distinct);
        }
    }

    /// SH-1 — the build-time semantic content fingerprint.
    ///
    /// Its bbox is a CONTAINMENT CLAIM, not a camera hint, which is the one
    /// place it deliberately diverges from `metadata.bounds`: it folds every
    /// feature, null-island rows included, because excluding data the archive
    /// will later be asked to contain would make the claim false.
    mod content_fingerprint {
        use crate::input::{content_fingerprint, FingerprintOptions, ParsedFeature};
        use crate::props::FeatureProperties;
        use geojson::{Geometry, Value as G};
        use std::collections::BTreeMap;

        fn feature(geometry: Option<G>, lon: f64, lat: f64) -> ParsedFeature {
            ParsedFeature {
                home_zoom: None,
                geojson: geojson::Feature {
                    bbox: None,
                    geometry: geometry.map(Geometry::new),
                    id: None,
                    properties: None,
                    foreign_members: None,
                },
                shared_properties: None,
                timestamp: 1_000,
                end_timestamp: None,
                vertex_timestamps: None,
                vertex_values: None,
                vertex_value_matrix: None,
                lon,
                lat,
            }
        }

        fn with_props(mut f: ParsedFeature, pairs: &[(&str, serde_json::Value)]) -> ParsedFeature {
            let mut map = serde_json::Map::new();
            for (k, v) in pairs {
                map.insert((*k).to_string(), v.clone());
            }
            f.shared_properties = FeatureProperties::from_map(map);
            f
        }

        fn fingerprint(features: &[ParsedFeature]) -> stt_core::metadata::ContentFingerprint {
            content_fingerprint(features, &FingerprintOptions::default()).unwrap()
        }

        /// The bbox is taken from VERTICES (the quantity tiles are addressed
        /// by), not from the centroid anchor — otherwise every line and polygon
        /// archive would fail its own containment check on the first decode.
        #[test]
        fn bbox_is_taken_from_vertices_not_the_centroid() {
            let feats = vec![feature(
                Some(G::LineString(vec![vec![-10.0, -4.0], vec![10.0, 6.0]])),
                0.0,
                1.0,
            )];
            let fp = fingerprint(&feats);
            assert_eq!(fp.bbox, [-10.0, -4.0, 10.0, 6.0]);
            assert_eq!(fp.version, stt_core::metadata::CONTENT_FINGERPRINT_VERSION);
            assert_eq!(fp.distinct_feature_count, 1);
        }

        /// ⚠️ THE DIVERGENCE FROM `metadata.bounds`. A null-island feature is
        /// EXCLUDED from the declared camera bbox (one coerced row must not
        /// zoom the showcase out to the whole globe) but INCLUDED here: if the
        /// archive decodes a vertex at (0, 0), the containment claim has to
        /// cover it or the validator reports a defect that is not there.
        #[test]
        fn null_island_features_are_included_in_the_containment_claim() {
            let feats = vec![
                feature(Some(G::Point(vec![10.0, 10.0])), 10.0, 10.0),
                feature(Some(G::Point(vec![0.0, 0.0])), 0.0, 0.0),
            ];
            let fp = fingerprint(&feats);
            assert_eq!(
                fp.bbox,
                [0.0, 0.0, 10.0, 10.0],
                "the sentinel must be covered"
            );

            // ...whereas the camera bbox still excludes it, unchanged.
            let profile = super::super::profile_features_with(
                &feats,
                &crate::input::FeatureProfileOptions {
                    bounds_mode: crate::input::BoundsMode::Vertex,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(profile.vertex_bounds.min_lon, 10.0);
        }

        /// A 3-element position contributes altitude; a 2-element one leaves
        /// `z_range` absent, which is what keeps 2D fingerprints small.
        #[test]
        fn z_comes_from_three_element_positions() {
            let flat = vec![feature(Some(G::Point(vec![1.0, 2.0])), 1.0, 2.0)];
            assert_eq!(fingerprint(&flat).z_range, None);

            let volumetric = vec![
                feature(Some(G::Point(vec![1.0, 2.0, 5.0])), 1.0, 2.0),
                feature(Some(G::Point(vec![1.5, 2.5, -3.0])), 1.5, 2.5),
            ];
            assert_eq!(fingerprint(&volumetric).z_range, Some([-3.0, 5.0]));
        }

        /// Property columns split by KIND exactly the way the style-hints
        /// profiler splits them: `as_f64` is numbers-only, so a numeric-looking
        /// string stays categorical.
        #[test]
        fn property_ranges_and_cardinality() {
            let feats = vec![
                with_props(
                    feature(Some(G::Point(vec![0.0, 0.0])), 0.0, 0.0),
                    &[
                        ("speed", serde_json::json!(2.0)),
                        ("kind", serde_json::json!("car")),
                        ("code", serde_json::json!("42")),
                    ],
                ),
                with_props(
                    feature(Some(G::Point(vec![1.0, 1.0])), 1.0, 1.0),
                    &[
                        ("speed", serde_json::json!(30.0)),
                        ("kind", serde_json::json!("bus")),
                        ("code", serde_json::json!("42")),
                    ],
                ),
            ];
            let fp = fingerprint(&feats);
            assert_eq!(fp.numeric_ranges.get("speed"), Some(&[2.0, 30.0]));
            assert_eq!(fp.categorical_cardinality.get("kind"), Some(&2));
            assert_eq!(
                fp.categorical_cardinality.get("code"),
                Some(&1),
                "a numeric-LOOKING string stays categorical, like the profiler"
            );
            assert!(!fp.numeric_ranges.contains_key("code"));
        }

        /// Tolerances come from the levers that actually move decoded values,
        /// and only from those: no quantization ⇒ an exact claim.
        #[test]
        fn tolerances_track_the_quantization_levers() {
            let feats = vec![with_props(
                feature(Some(G::Point(vec![1.0, 2.0])), 1.0, 2.0),
                &[("z", serde_json::json!(7.5))],
            )];

            let exact = fingerprint(&feats);
            assert_eq!(exact.coord_tolerance_deg, 0.0);
            assert!(exact.column_tolerance.is_empty());

            let quantized = content_fingerprint(
                &feats,
                &FingerprintOptions {
                    quantize_coords_m: Some(1.0),
                    attr_precisions: BTreeMap::from([("z".to_string(), 0.05)]),
                    elevation_column: None,
                },
            )
            .unwrap();
            assert!(
                (quantized.coord_tolerance_deg - 1.0 / 111_320.0).abs() < 1e-15,
                "got {}",
                quantized.coord_tolerance_deg
            );
            assert_eq!(quantized.column_tolerance.get("z"), Some(&0.05));
        }

        /// `--point-elevation-column` moves a column OUT of the property set and
        /// into geometry z, so the fingerprint must follow it — declaring a
        /// numeric range for a column the archive no longer carries would fire a
        /// finding on every honest volumetric build.
        #[test]
        fn elevation_column_folds_into_z_and_leaves_the_property_set() {
            let feats = vec![
                with_props(
                    feature(Some(G::Point(vec![1.0, 2.0])), 1.0, 2.0),
                    &[("alt", serde_json::json!(3.0))],
                ),
                with_props(
                    feature(Some(G::Point(vec![1.5, 2.5])), 1.5, 2.5),
                    &[("alt", serde_json::json!(9.0))],
                ),
            ];
            let fp = content_fingerprint(
                &feats,
                &FingerprintOptions {
                    elevation_column: Some("alt"),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(fp.z_range, Some([3.0, 9.0]));
            assert!(
                !fp.numeric_ranges.contains_key("alt"),
                "the folded column must not be declared twice"
            );
        }

        /// ⭐ …and the SAME flag on a NON-point layer must do neither.
        ///
        /// `--point-elevation-column` is a shipped flag combination on a
        /// linestring build, and the fold never happens there — `encode` gates
        /// on `GeometryColumn::Point` and the flag's help says "Only affects
        /// POINT layers". Assuming it happened baked a fingerprint that was
        /// wrong twice over: a `z_range` invented for an archive with no 3D
        /// geometry, and the named column dropped from `numeric_ranges`, which
        /// switched OFF the attribute check that catches a corrupted rebuild.
        /// A wrong fingerprint is worse than none, because it certifies.
        #[test]
        fn elevation_column_is_not_folded_on_a_non_point_layer() {
            let line = |lo: f64, hi: f64, alt: f64| {
                with_props(
                    feature(
                        Some(G::LineString(vec![vec![lo, lo], vec![hi, hi]])),
                        (lo + hi) / 2.0,
                        (lo + hi) / 2.0,
                    ),
                    &[("alt", serde_json::json!(alt))],
                )
            };
            let feats = vec![line(1.0, 2.0, 5.0), line(2.0, 3.0, 34.0)];
            let fp = content_fingerprint(
                &feats,
                &FingerprintOptions {
                    elevation_column: Some("alt"),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                fp.z_range, None,
                "a linestring build has no 3D geometry, so it may claim no vertical extent"
            );
            assert_eq!(
                fp.numeric_ranges.get("alt"),
                Some(&[5.0, 34.0]),
                "the column is still a property on a non-point layer and MUST stay checkable"
            );

            // Polygons behave the same way — the gate is "is it a point", not
            // "is it not a line".
            let poly = with_props(
                feature(
                    Some(G::Polygon(vec![vec![
                        vec![0.0, 0.0],
                        vec![1.0, 0.0],
                        vec![1.0, 1.0],
                        vec![0.0, 0.0],
                    ]])),
                    0.5,
                    0.5,
                ),
                &[("alt", serde_json::json!(12.0))],
            );
            let fp = content_fingerprint(
                std::slice::from_ref(&poly),
                &FingerprintOptions {
                    elevation_column: Some("alt"),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(fp.z_range, None);
            assert_eq!(fp.numeric_ranges.get("alt"), Some(&[12.0, 12.0]));
        }

        /// A MIXED build splits the claim exactly where the encoder splits it:
        /// the point layer's values move into `z_range`, the line layer's stay
        /// in `numeric_ranges`. Neither statistic may absorb the other's
        /// values — over-declaring either one is a finding on an honest
        /// archive under a full decode.
        #[test]
        fn mixed_geometry_splits_the_elevation_claim_by_layer() {
            let feats = vec![
                with_props(
                    feature(Some(G::Point(vec![1.0, 2.0])), 1.0, 2.0),
                    &[("alt", serde_json::json!(3.0))],
                ),
                with_props(
                    feature(
                        Some(G::LineString(vec![vec![1.0, 1.0], vec![2.0, 2.0]])),
                        1.5,
                        1.5,
                    ),
                    &[("alt", serde_json::json!(900.0))],
                ),
            ];
            let fp = content_fingerprint(
                &feats,
                &FingerprintOptions {
                    elevation_column: Some("alt"),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                fp.z_range,
                Some([3.0, 3.0]),
                "only the POINT feature's altitude reaches geometry z"
            );
            assert_eq!(
                fp.numeric_ranges.get("alt"),
                Some(&[900.0, 900.0]),
                "only the LINE feature's altitude survives as a property"
            );
        }

        /// DETERMINISM. Two runs over the same feature set — in any order —
        /// serialise to BYTE-IDENTICAL `content_fingerprint` JSON. This is the
        /// build-reproducibility pin: min/max folds, BTreeMaps, no sampling
        /// stride, no clock, no hash iteration order.
        #[test]
        fn content_fingerprint_is_order_independent_and_reproducible() {
            let mut feats = vec![
                with_props(
                    feature(Some(G::Point(vec![-3.0, 4.0])), -3.0, 4.0),
                    &[
                        ("speed", serde_json::json!(1.5)),
                        ("kind", serde_json::json!("a")),
                    ],
                ),
                with_props(
                    feature(
                        Some(G::LineString(vec![vec![10.0, -8.0], vec![12.0, -6.0]])),
                        11.0,
                        -7.0,
                    ),
                    &[
                        ("speed", serde_json::json!(9.5)),
                        ("kind", serde_json::json!("b")),
                    ],
                ),
                with_props(
                    feature(Some(G::Point(vec![0.5, 0.5, 2.0])), 0.5, 0.5),
                    &[
                        ("speed", serde_json::json!(4.0)),
                        ("kind", serde_json::json!("a")),
                    ],
                ),
            ];

            let bytes = |f: &[ParsedFeature]| serde_json::to_vec(&fingerprint(f)).unwrap();
            let first = bytes(&feats);
            assert_eq!(bytes(&feats), first, "a re-run must be byte-identical");

            feats.reverse();
            assert_eq!(bytes(&feats), first, "input order must not leak");
            feats.rotate_left(1);
            assert_eq!(bytes(&feats), first, "nor must a rotation");

            // ...and the bytes really do describe the data.
            let fp = fingerprint(&feats);
            assert_eq!(fp.bbox, [-3.0, -8.0, 12.0, 4.0]);
            assert_eq!(fp.z_range, Some([2.0, 2.0]));
            assert_eq!(fp.numeric_ranges.get("speed"), Some(&[1.5, 9.5]));
            assert_eq!(fp.categorical_cardinality.get("kind"), Some(&2));
        }

        /// Non-finite ordinates are skipped rather than folded (one NaN would
        /// poison the chain, one inf would swallow the globe), and an entirely
        /// unusable input degrades to a vacuous whole-world superset instead of
        /// an inside-out box.
        #[test]
        fn non_finite_and_empty_inputs_degrade_to_a_superset() {
            let feats = vec![feature(
                Some(G::Point(vec![f64::NAN, f64::NAN])),
                f64::NAN,
                f64::NAN,
            )];
            assert_eq!(fingerprint(&feats).bbox, [-180.0, -90.0, 180.0, 90.0]);
            assert_eq!(fingerprint(&[]).bbox, [-180.0, -90.0, 180.0, 90.0]);
            assert_eq!(fingerprint(&[]).distinct_feature_count, 0);
        }
    }
}
