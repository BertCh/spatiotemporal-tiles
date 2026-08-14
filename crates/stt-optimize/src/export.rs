//! GeoParquet export OUT of a built packed archive (`stt-optimize export`).
//!
//! Every other path in this repo moves data *into* the packed format. This one
//! moves it back out, so an archive is a render tier over a lakehouse rather
//! than a terminus: the tiles are already Arrow with interleaved GeoArrow
//! coordinates, so the export is a re-encode of the geometry column to WKB plus
//! the GeoParquet file metadata that makes the result self-describing to
//! DuckDB / GeoPandas / Sedona / Iceberg.
//!
//! ## What one row is
//!
//! A row is a **tile-local feature record**, not a source feature. Two facts
//! from the build side drive every design decision below:
//!
//! * `stt-build` **clips** features that span tile boundaries (see
//!   `TileConfig::clip_trajectories` / `clip_non_trajectory`) and gives every
//!   piece the parent's feature id. Deduplicating on `id` would therefore
//!   silently delete geometry, so this exporter never dedupes — it emits the
//!   pieces and records their provenance in `stt_zoom`/`stt_x`/`stt_y`, so a
//!   consumer can reassemble with `GROUP BY id` + a union aggregate.
//! * The same feature is re-tiled at every zoom, at a different simplification
//!   tolerance, and a temporal-LOD archive additionally carries coarser
//!   aggregate tiles at those same zooms. Exporting the whole directory would
//!   emit each feature once per zoom *and* once per LOD tier. So an export is
//!   always **one zoom** (default: the deepest one present) and **base-bucket
//!   tiles only** — one coherent generalization level, no double counting.
//!
//! ## One file, many row groups
//!
//! Output is one Parquet file per layer, never one per tile. The `geo`
//! metadata that makes a file GeoParquet is *file*-level (version, primary
//! column, geometry types, bbox), so a file-per-tile archive would repeat that
//! header tens of thousands of times and hand the user an object-store
//! anti-pattern instead of an artifact. The pruning that file-per-tile would
//! buy is recovered inside the single file: directory entries arrive in
//! `(zoom, hilbert, time_start)` order, so rows land in spatially and
//! temporally coherent row groups, and the emitted GeoParquet 1.1 `covering`
//! bbox column plus the `start_time` statistics let a reader skip row groups on
//! both axes. Separate *layers* do get separate files — a point layer and a
//! linestring layer have different schemas and cannot share one.
//!
//! ## Two passes
//!
//! `stt-build` seals a layer's property set per tile from the values actually
//! observed in that tile (`columnar.rs::PropertyAccumulator::seal`); only keys
//! present in a declared input schema are guaranteed in *every* tile. So the
//! property column set can legitimately drift between tiles of one layer, while
//! `ArrowWriter` needs its schema up front. Pass 1 therefore decodes the
//! selected tiles to union their output schemas; pass 2 decodes again and
//! writes. The double decode is the price of never silently dropping a column
//! that appears late in the directory.

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{
    Array, ArrayRef, BinaryBuilder, Float32Array, Float64Array, Int32Array, Int64Array, ListArray,
    RecordBatch, StructArray, UInt32Array, UInt8Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, TimeUnit};
use parquet::arrow::arrow_writer::ArrowWriterOptions;
use parquet::arrow::{ArrowSchemaConverter, ArrowWriter};
use parquet::basic::{Compression, LogicalType, Type as PhysicalType, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::packed::PackedTileset;
use stt_core::arrow_tile::{AttrQuant, QuantAffine, STT_QUANT_ATTR_META_KEY, STT_QUANT_META_KEY};
use stt_core::projection::tile_geo_bounds;
use stt_core::TileEntry;

/// GeoParquet spec version written into the `geo` file metadata. 1.1 is the
/// version that standardises the `covering` bbox column this exporter emits.
const GEOPARQUET_VERSION: &str = "1.1.0";

/// The Parquet file-metadata key GeoParquet claims.
const GEO_METADATA_KEY: &str = "geo";

/// Output geometry column name (also the `geo` metadata's `primary_column`).
const GEOMETRY_COLUMN: &str = "geometry";

/// Output covering-bbox struct column name, referenced by `covering.bbox` in
/// the `geo` metadata.
const BBOX_COLUMN: &str = "bbox";

/// Provenance columns naming the tile each row was read from. Present because
/// clipped features are split across tiles under one id (see the module docs) —
/// without them a consumer cannot tell a clipped piece from a whole feature.
const PROVENANCE_COLUMNS: [&str; 3] = ["stt_zoom", "stt_x", "stt_y"];

/// Reserved tile columns that are pre-baked *renderer* state rather than data:
/// `triangles` is the earcut tessellation of the polygon in the same row, so
/// exporting it would ship a derived index buffer no GeoParquet reader can use
/// and that any tessellator can regenerate. `part_offsets` is the same kind of
/// thing for multi-part polygons — ring indices into the row's own geometry,
/// which the WKB below flattens, so the numbers would name nothing a Parquet
/// consumer can address. (The better use of that column is to emit a real WKB
/// `MultiPolygon` instead of one `Polygon` with the extra parts masquerading as
/// holes; that is a separate change to the geometry writer, tracked as a
/// follow-up — the flattening predates the column.)
const DERIVED_COLUMNS: [&str; 2] = ["triangles", "part_offsets"];

/// Schema-metadata keys of the delta `vertex_time` encoding. Private in
/// `stt_core::arrow_tile`, mirrored here because the exporter must reconstruct
/// absolute per-vertex times (`origin + delta * step`) — shipping raw deltas
/// would export a column whose numbers mean nothing outside this format.
const VERTEX_TIME_ORIGIN_KEY: &str = "stt:vertex_time_origin_ms";
/// TB-11 extension 2: per-vertex time deltas measured from each feature's own
/// `start_time`. Present INSTEAD of the origin/step pair above.
const VERTEX_TIME_FEATURE_STEP_KEY: &str = "stt:vertex_time_feature_step_ms";
/// The tile's absolute time anchor, added back to a compact `UInt32`
/// `start_time` before it can serve as a vertex-time anchor.
const TIME_OFFSET_MS_KEY: &str = "stt:time_offset_ms";
const VERTEX_TIME_STEP_KEY: &str = "stt:vertex_time_step_ms";

/// How the geometry column is typed in the written Parquet file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeometryEncoding {
    /// `BYTE_ARRAY` of WKB described by the GeoParquet 1.1 `geo` metadata —
    /// what every deployed reader understands today.
    #[default]
    Wkb,
    /// The same WKB bytes, additionally carrying Parquet's native `GEOMETRY`
    /// logical type (Parquet format 2.11 / Iceberg v3 / GeoParquet 2.0). The
    /// `geo` metadata is still written, so 1.1 readers are unaffected.
    Native,
}

impl GeometryEncoding {
    /// Parse the CLI value.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "wkb" => Ok(Self::Wkb),
            "native" => Ok(Self::Native),
            other => bail!("unknown geometry encoding '{other}' (expected 'wkb' or 'native')"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Wkb => "wkb",
            Self::Native => "native",
        }
    }
}

/// What to pull out of the archive.
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Zoom level to export. `None` ⇒ the deepest zoom present.
    pub zoom: Option<u8>,
    /// Layer to export. `None` ⇒ every layer (one file each).
    pub layer: Option<String>,
    /// `[min_lon, min_lat, max_lon, max_lat]` row filter (WGS84).
    pub bbox: Option<[f64; 4]>,
    /// Inclusive Unix-ms lower bound on a feature's `[start_time, end_time]`.
    pub start: Option<i64>,
    /// Inclusive Unix-ms upper bound on a feature's `[start_time, end_time]`.
    pub end: Option<i64>,
    /// Geometry column typing.
    pub geometry_encoding: GeometryEncoding,
}

/// Parse a `min_lon,min_lat,max_lon,max_lat` CLI bbox.
pub fn parse_bbox(s: &str) -> Result<[f64; 4]> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        bail!(
            "--bbox needs 4 comma-separated numbers (min_lon,min_lat,max_lon,max_lat), got {}",
            parts.len()
        );
    }
    let mut out = [0.0f64; 4];
    for (i, p) in parts.iter().enumerate() {
        out[i] = p
            .parse::<f64>()
            .with_context(|| format!("--bbox component {} ('{p}') is not a number", i + 1))?;
    }
    Ok(out)
}

/// Parse a time bound: an ISO-8601 instant (`2024-03-01T12:00:00Z`), a bare
/// date (`2024-03-01`, midnight UTC), or Unix milliseconds.
///
/// ISO is tried first, so a 4-digit year-like value only ever reads as epoch
/// milliseconds when it is not a valid date — `2024` is 2024 ms, not the year.
pub fn parse_time_bound(s: &str) -> Result<i64> {
    let s = s.trim();
    if let Ok(dt) = s.parse::<chrono::DateTime<chrono::Utc>>() {
        return Ok(dt.timestamp_millis());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt.and_utc().timestamp_millis());
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(d
            .and_hms_opt(0, 0, 0)
            .expect("midnight is a valid time")
            .and_utc()
            .timestamp_millis());
    }
    s.parse::<i64>()
        .with_context(|| format!("'{s}' is neither an ISO-8601 timestamp nor Unix milliseconds"))
}

/// One written Parquet file.
#[derive(Debug, Clone, Serialize)]
pub struct ExportedFile {
    /// Path written.
    pub path: String,
    /// Source layer name.
    pub layer: String,
    /// Rows written (tile-local feature records — see the module docs).
    pub rows: u64,
    /// Tiles that contributed at least one row.
    pub tiles: usize,
    /// `[min_lon, min_lat, max_lon, max_lat]` of the written rows, `None` when
    /// nothing was written or every geometry was empty.
    pub bbox: Option<[f64; 4]>,
    /// Distinct GeoParquet geometry-type names observed.
    pub geometry_types: Vec<String>,
    /// Tile columns deliberately not exported (derived renderer state).
    pub dropped_columns: Vec<String>,
}

/// Result of one `export` run.
#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    /// Manifest the rows came from.
    pub archive: String,
    /// Zoom exported.
    pub zoom: u8,
    /// Directory entries at that zoom in the base temporal tier.
    pub tiles_in_zoom: usize,
    /// Of those, the entries that survived the bbox/time prune.
    pub tiles_selected: usize,
    /// GeoParquet spec version written.
    pub geoparquet_version: String,
    /// `wkb` or `native`.
    pub geometry_encoding: String,
    /// Files written.
    pub files: Vec<ExportedFile>,
}

/// Render an [`ExportReport`] for the terminal.
pub fn format_text(report: &ExportReport) -> String {
    let mut s = String::new();
    s.push_str("GeoParquet export\n");
    s.push_str(&format!("  archive:  {}\n", report.archive));
    s.push_str(&format!(
        "  zoom:     {} ({} of {} base tiles selected)\n",
        report.zoom, report.tiles_selected, report.tiles_in_zoom
    ));
    s.push_str(&format!(
        "  encoding: GeoParquet {} / {}\n",
        report.geoparquet_version, report.geometry_encoding
    ));
    if report.files.is_empty() {
        s.push_str("\n  (no rows matched — nothing written)\n");
        return s;
    }
    s.push('\n');
    for f in &report.files {
        s.push_str(&format!("  {} \u{2192} {}\n", f.layer, f.path));
        s.push_str(&format!(
            "    {} row(s) from {} tile(s); types [{}]\n",
            f.rows,
            f.tiles,
            f.geometry_types.join(", ")
        ));
        if let Some(b) = f.bbox {
            s.push_str(&format!(
                "    bbox [{:.5}, {:.5}, {:.5}, {:.5}]\n",
                b[0], b[1], b[2], b[3]
            ));
        }
        if !f.dropped_columns.is_empty() {
            s.push_str(&format!(
                "    dropped (derived): {}\n",
                f.dropped_columns.join(", ")
            ));
        }
    }
    s
}

// ----------------------------------------------------------------------------
// Entry selection
// ----------------------------------------------------------------------------

/// Entries in the **base** temporal tier only.
///
/// A temporal-LOD archive carries extra aggregate tiles at the same `(z, x, y)`
/// with a coarser bucket; including them would emit every feature twice (once
/// per tier). `Metadata::temporal_bucket_ms` is the base, and LOD levels are
/// validated to be strictly larger multiples of it (`tiler.rs`), so an equality
/// test is exact. Entries predating the per-entry bucket column carry `None`
/// and are, by definition, base tiles.
fn is_base_tier(entry: &TileEntry, base_bucket_ms: u64) -> bool {
    match entry.temporal_bucket_ms {
        Some(b) if base_bucket_ms > 0 => b == base_bucket_ms,
        _ => true,
    }
}

/// Does the tile's geographic footprint touch the requested bbox?
fn tile_overlaps_bbox(entry: &TileEntry, bbox: &[f64; 4]) -> bool {
    let (min_lon, min_lat, max_lon, max_lat) = tile_geo_bounds(entry.zoom, entry.x, entry.y);
    min_lon <= bbox[2] && max_lon >= bbox[0] && min_lat <= bbox[3] && max_lat >= bbox[1]
}

/// Does the tile's covered time span touch `[start, end]`?
///
/// Uses `cover_t_min` (the tight lower bound — the earliest feature actually
/// in the tile) when the archive recorded it; `time_start` is only the
/// addressable bucket boundary and is deliberately loose, so pruning on it
/// would keep tiles whose data all lies outside the window.
fn tile_overlaps_time(entry: &TileEntry, start: Option<i64>, end: Option<i64>) -> bool {
    let lo = entry.cover_t_min.unwrap_or(entry.time_start);
    let hi = entry.time_end;
    start.map_or(true, |s| hi >= s) && end.map_or(true, |e| lo <= e)
}

/// Deepest zoom present in the base tier.
fn deepest_base_zoom(tileset: &PackedTileset) -> Option<u8> {
    let base = tileset.temporal_bucket_ms();
    tileset
        .entries()
        .iter()
        .filter(|e| is_base_tier(e, base))
        .map(|e| e.zoom)
        .max()
}

// ----------------------------------------------------------------------------
// Per-column plan
// ----------------------------------------------------------------------------

/// What the exporter does with one decoded tile column.
#[derive(Debug, Clone)]
enum ColumnPlan {
    /// Copy the array through unchanged.
    Pass,
    /// `Int64` Unix ms → `Timestamp(Millisecond, "UTC")`, so the times land in
    /// a lakehouse as timestamps rather than anonymous integers. The failure
    /// this prevents is a round-trip whose time column arrives as a bare i64
    /// nobody downstream reads as time.
    EpochMillis,
    /// Numeric property stored as fixed-point indices — reconstruct `Float64`
    /// via `value = o + q * s`. Exporting the raw indices would ship numbers
    /// that are off by orders of magnitude with no way to notice.
    DequantizeAttr(AttrQuant),
    /// `List<UInt16>` per-vertex time deltas → `List<Int64>` absolute Unix ms
    /// via `origin + delta * step`.
    VertexTimeDeltas { origin: i64, step: i64 },
    /// TB-11 extension 2. `anchors[row] + delta * step`, so unlike every other
    /// plan this one depends on a SECOND column (`start_time`) and must be
    /// built with the batch in hand. Anchors are already row-selected to match
    /// the array being inflated.
    VertexTimeFeatureDeltas { step: i64, anchors: Arc<Vec<i64>> },
    /// Derived renderer state — not exported.
    Drop,
}

/// Decide the plan for one decoded tile column.
fn plan_column(
    field: &Field,
    schema_meta: &std::collections::HashMap<String, String>,
) -> ColumnPlan {
    let name = field.name().as_str();
    if DERIVED_COLUMNS.contains(&name) {
        return ColumnPlan::Drop;
    }
    if name == "start_time" || name == "end_time" {
        return ColumnPlan::EpochMillis;
    }
    if name == "vertex_time" {
        // Only the delta encodings need reconstruction; a `List<Int64>`
        // vertex_time is already absolute. The encoder picks the narrowest
        // delta width that fits the tile's span (`UInt16`, then `UInt32`), so
        // BOTH must be reconstructed — matching only `UInt16` here silently
        // exported a wider tile's raw deltas as if they were epoch millis.
        let is_delta = matches!(
            field.data_type(),
            DataType::List(child)
                if matches!(child.data_type(), DataType::UInt16 | DataType::UInt32)
        );
        if is_delta {
            let origin = schema_meta
                .get(VERTEX_TIME_ORIGIN_KEY)
                .and_then(|v| v.parse::<i64>().ok());
            let step = schema_meta
                .get(VERTEX_TIME_STEP_KEY)
                .and_then(|v| v.parse::<i64>().ok());
            if let (Some(origin), Some(step)) = (origin, step) {
                return ColumnPlan::VertexTimeDeltas { origin, step };
            }
            // TB-11 extension 2. The step is in the schema; the ANCHORS are in
            // the batch's `start_time` column, which this function does not
            // have. Return the variant with empty anchors — enough to type the
            // output column as Int64 — and let the apply loop fill them in.
            // Falling through to a plain `Pass` instead would export RAW DELTAS
            // as if they were epoch millis, the failure the comment above warns
            // about wearing a different hat.
            if let Some(step) = schema_meta
                .get(VERTEX_TIME_FEATURE_STEP_KEY)
                .and_then(|v| v.parse::<i64>().ok())
            {
                return ColumnPlan::VertexTimeFeatureDeltas {
                    step,
                    anchors: Arc::new(Vec::new()),
                };
            }
        }
        return ColumnPlan::Pass;
    }
    if let Some(json) = field.metadata().get(STT_QUANT_ATTR_META_KEY) {
        if let Some(q) = AttrQuant::from_json(json) {
            return ColumnPlan::DequantizeAttr(q);
        }
    }
    ColumnPlan::Pass
}

/// The output field a plan produces, or `None` when the column is dropped.
/// Field metadata is stripped: the `stt:` keys describe an encoding the export
/// has already undone, and leaving them would tell a reader the values are
/// still quantized.
fn output_field(field: &Field, plan: &ColumnPlan) -> Option<Field> {
    let name = field.name().clone();
    let nullable = field.is_nullable();
    Some(match plan {
        ColumnPlan::Drop => return None,
        ColumnPlan::Pass => Field::new(name, field.data_type().clone(), nullable),
        ColumnPlan::EpochMillis => Field::new(
            name,
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            nullable,
        ),
        ColumnPlan::DequantizeAttr(_) => Field::new(name, DataType::Float64, true),
        ColumnPlan::VertexTimeFeatureDeltas { .. } | ColumnPlan::VertexTimeDeltas { .. } => {
            Field::new(
                name,
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                nullable,
            )
        }
    })
}

/// Apply a plan to one array.
fn apply_plan(array: &ArrayRef, plan: &ColumnPlan, out_type: &DataType) -> Result<ArrayRef> {
    Ok(match plan {
        ColumnPlan::Drop => unreachable!("dropped columns never reach apply_plan"),
        ColumnPlan::Pass => Arc::clone(array),
        ColumnPlan::EpochMillis => arrow::compute::cast(array, out_type)
            .context("casting a Unix-ms column to Timestamp(ms, UTC)")?,
        ColumnPlan::DequantizeAttr(q) => dequantize_attr(array, q)?,
        ColumnPlan::VertexTimeDeltas { origin, step } => {
            vertex_times_absolute(array, *origin, *step)?
        }
        ColumnPlan::VertexTimeFeatureDeltas { step, anchors } => {
            vertex_times_feature_absolute(array, *step, anchors)?
        }
    })
}

/// `value = o + q * s` over a `UInt16`/`Int32` (or any integer) index column.
fn dequantize_attr(array: &ArrayRef, q: &AttrQuant) -> Result<ArrayRef> {
    // Cast to i64 first so one branch covers every integer width the encoder
    // picks (it chooses the smallest of UInt16/Int32 per column).
    let idx = arrow::compute::cast(array, &DataType::Int64)
        .context("casting a quantized property column to Int64 for dequantization")?;
    let idx = idx
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow!("quantized property column did not cast to Int64"))?;
    let values: Float64Array = idx.iter().map(|v| v.map(|q_i| q.value(q_i))).collect();
    Ok(Arc::new(values))
}

/// The absolute `start_time` of each selected row — the anchor TB-11's
/// feature-anchored vertex times are measured from.
///
/// `start_time` ships either as absolute `Int64` or, under compact times, as a
/// `UInt32` offset from the tile's `t0`. Both are reconstructed here, because
/// reading the compact form as absolute would place every vertex in 1970 — and
/// the encoder anchored against the ABSOLUTE value.
fn feature_anchors(
    batch: &RecordBatch,
    schema_meta: &std::collections::HashMap<String, String>,
    indices: &UInt32Array,
) -> Result<Vec<i64>> {
    let col = batch
        .column_by_name("start_time")
        .ok_or_else(|| anyhow!("feature-anchored vertex times need a start_time column"))?;
    let taken = arrow::compute::take(col, indices, None)
        .context("selecting rows of 'start_time' for the vertex-time anchor")?;
    let as_i64 = arrow::compute::cast(&taken, &DataType::Int64)
        .context("casting start_time to Int64 for the vertex-time anchor")?;
    let as_i64 = as_i64
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow!("start_time did not cast to Int64"))?;
    // Compact times store an offset from t0; absolute times store no t0 to add.
    let t0 = if matches!(taken.data_type(), DataType::UInt32) {
        schema_meta
            .get(TIME_OFFSET_MS_KEY)
            .and_then(|v| v.parse::<i64>().ok())
            .ok_or_else(|| anyhow!("compact start_time without a {} anchor", TIME_OFFSET_MS_KEY))?
    } else {
        0
    };
    Ok(as_i64.iter().map(|v| t0 + v.unwrap_or(0)).collect())
}

/// `List<UInt16>` deltas → `List<Int64>` absolute Unix ms, anchored PER FEATURE.
fn vertex_times_feature_absolute(array: &ArrayRef, step: i64, anchors: &[i64]) -> Result<ArrayRef> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("vertex_time column is not a List"))?;
    let child = arrow::compute::cast(list.values(), &DataType::Int64)
        .context("casting vertex_time deltas to Int64")?;
    let child = child
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow!("vertex_time child did not cast to Int64"))?;
    let offsets = list.offsets();
    let mut out: Vec<Option<i64>> = Vec::with_capacity(child.len());
    for row in 0..list.len() {
        let anchor = *anchors
            .get(row)
            .ok_or_else(|| anyhow!("vertex-time anchor missing for row {row}"))?;
        let (lo, hi) = (offsets[row] as usize, offsets[row + 1] as usize);
        for j in lo..hi {
            out.push(if child.is_null(j) {
                None
            } else {
                Some(anchor + child.value(j) * step)
            });
        }
    }
    Ok(Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Int64, true)),
        list.offsets().clone(),
        Arc::new(Int64Array::from(out)),
        list.nulls().cloned(),
    )))
}

/// `List<UInt16>` deltas → `List<Int64>` absolute Unix ms.
fn vertex_times_absolute(array: &ArrayRef, origin: i64, step: i64) -> Result<ArrayRef> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("vertex_time column is not a List"))?;
    let child = arrow::compute::cast(list.values(), &DataType::Int64)
        .context("casting vertex_time deltas to Int64")?;
    let child = child
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow!("vertex_time child did not cast to Int64"))?;
    let absolute: Int64Array = child.iter().map(|v| v.map(|d| origin + d * step)).collect();
    Ok(Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Int64, true)),
        list.offsets().clone(),
        Arc::new(absolute),
        list.nulls().cloned(),
    )))
}

// ----------------------------------------------------------------------------
// Geometry → WKB
// ----------------------------------------------------------------------------

/// WKB little-endian byte-order marker.
const WKB_LE: u8 = 1;
/// ISO WKB adds 1000 to the type code for a Z (3D) geometry.
const WKB_Z: u32 = 1000;

/// A decoded tile geometry column, ready to be walked row by row.
///
/// Coordinates are either GeoArrow `Float64` or, when the archive was built
/// with `--quantize-coords`, `Int32` grid indices plus the world-anchored
/// affine from the geometry field's `stt:quant` metadata. Exporting the raw
/// indices would put every feature at a nonsense location, so dequantization is
/// not optional here.
struct GeometryColumn<'a> {
    kind: GeometryKind,
    /// Leaf coordinate accessor.
    coords: CoordSource<'a>,
    /// Components per coordinate (2 or 3).
    width: usize,
    /// `List` offsets, outermost first (empty for points).
    outer: Option<&'a OffsetBuffer<i32>>,
    inner: Option<&'a OffsetBuffer<i32>>,
    rows: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GeometryKind {
    Point,
    LineString,
    Polygon,
}

enum CoordSource<'a> {
    F64(&'a Float64Array),
    Quantized(&'a Int32Array, QuantAffine),
}

impl CoordSource<'_> {
    /// The `(x, y, z)` of the coordinate whose first component sits at leaf
    /// index `base`.
    fn coord(&self, base: usize, width: usize) -> (f64, f64, Option<f64>) {
        match self {
            CoordSource::F64(a) => (
                a.value(base),
                a.value(base + 1),
                (width == 3).then(|| a.value(base + 2)),
            ),
            CoordSource::Quantized(a, q) => (
                q.lon(a.value(base)),
                q.lat(a.value(base + 1)),
                (width == 3).then(|| {
                    let z0 = q.z0.unwrap_or(0.0);
                    let sz = q.sz.unwrap_or(1.0);
                    z0 + a.value(base + 2) as f64 * sz
                }),
            ),
        }
    }
}

/// Peel the GeoArrow nesting off a decoded `geometry` column.
fn open_geometry<'a>(field: &Field, array: &'a ArrayRef) -> Result<GeometryColumn<'a>> {
    let quant = field
        .metadata()
        .get(STT_QUANT_META_KEY)
        .and_then(|j| QuantAffine::from_json(j));

    // Walk down at most two List levels: Polygon is List<List<FixedSizeList>>,
    // LineString is List<FixedSizeList>, Point is FixedSizeList.
    let (kind, outer, inner, leaf): (GeometryKind, _, _, &dyn Array) = match array.data_type() {
        DataType::FixedSizeList(_, _) => (GeometryKind::Point, None, None, array.as_ref()),
        DataType::List(_) => {
            let l1 = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| anyhow!("geometry column claims List but is not a ListArray"))?;
            match l1.values().data_type() {
                DataType::FixedSizeList(_, _) => (
                    GeometryKind::LineString,
                    Some(l1.offsets()),
                    None,
                    l1.values().as_ref(),
                ),
                DataType::List(_) => {
                    let l2 = l1
                        .values()
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .ok_or_else(|| anyhow!("polygon ring level is not a ListArray"))?;
                    (
                        GeometryKind::Polygon,
                        Some(l1.offsets()),
                        Some(l2.offsets()),
                        l2.values().as_ref(),
                    )
                }
                other => bail!("unsupported geometry nesting: List<{other}>"),
            }
        }
        other => bail!("unsupported geometry column type {other}"),
    };

    let fsl = leaf
        .as_any()
        .downcast_ref::<arrow::array::FixedSizeListArray>()
        .ok_or_else(|| anyhow!("geometry leaf is not a FixedSizeList of coordinates"))?;
    let width = fsl.value_length() as usize;
    if width != 2 && width != 3 {
        bail!("geometry leaf has {width} components per coordinate (expected 2 or 3)");
    }
    let values = fsl.values();
    let coords = match (values.data_type(), quant) {
        (DataType::Float64, _) => CoordSource::F64(
            values
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("Float64 coordinate leaf did not downcast"))?,
        ),
        (DataType::Int32, Some(q)) => CoordSource::Quantized(
            values
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| anyhow!("Int32 coordinate leaf did not downcast"))?,
            q,
        ),
        // An Int32 leaf without the affine is unrecoverable: the indices are
        // meaningless without the grid, so refuse rather than write garbage
        // coordinates that look plausible.
        (DataType::Int32, None) => bail!(
            "geometry ships quantized Int32 grid indices but the `{STT_QUANT_META_KEY}` \
             affine is missing from the field metadata — the archive cannot be \
             dequantized and exporting it would place every feature at a wrong location"
        ),
        (other, _) => bail!("unsupported coordinate leaf type {other}"),
    };

    // A FixedSizeList point column has one coordinate per row; the nested
    // shapes carry their row count on the outermost list.
    let rows = array.len();
    Ok(GeometryColumn {
        kind,
        coords,
        width,
        outer,
        inner,
        rows,
    })
}

/// Bounding box accumulated while a row's WKB is written.
#[derive(Debug, Clone, Copy)]
struct RowBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl RowBox {
    fn empty() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }
    fn add(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }
    fn is_empty(&self) -> bool {
        self.min_x > self.max_x
    }
    fn intersects(&self, b: &[f64; 4]) -> bool {
        !self.is_empty()
            && self.min_x <= b[2]
            && self.max_x >= b[0]
            && self.min_y <= b[3]
            && self.max_y >= b[1]
    }
}

fn push_coord(out: &mut Vec<u8>, x: f64, y: f64, z: Option<f64>) {
    out.extend_from_slice(&x.to_le_bytes());
    out.extend_from_slice(&y.to_le_bytes());
    if let Some(z) = z {
        out.extend_from_slice(&z.to_le_bytes());
    }
}

impl GeometryColumn<'_> {
    /// The GeoParquet `geometry_types` name for this column.
    fn type_name(&self) -> String {
        let base = match self.kind {
            GeometryKind::Point => "Point",
            GeometryKind::LineString => "LineString",
            GeometryKind::Polygon => "Polygon",
        };
        if self.width == 3 {
            format!("{base} Z")
        } else {
            base.to_string()
        }
    }

    /// Serialise row `row` as little-endian ISO WKB into `out`, returning its
    /// bounding box.
    fn write_wkb(&self, row: usize, out: &mut Vec<u8>) -> RowBox {
        let mut bbox = RowBox::empty();
        let z3 = self.width == 3;
        let type_code = |base: u32| if z3 { base + WKB_Z } else { base };
        match self.kind {
            GeometryKind::Point => {
                out.push(WKB_LE);
                out.extend_from_slice(&type_code(1).to_le_bytes());
                let (x, y, z) = self.coords.coord(row * self.width, self.width);
                bbox.add(x, y);
                push_coord(out, x, y, z);
            }
            GeometryKind::LineString => {
                let offsets = self.outer.expect("linestring carries vertex offsets");
                let (lo, hi) = (offsets[row] as usize, offsets[row + 1] as usize);
                out.push(WKB_LE);
                out.extend_from_slice(&type_code(2).to_le_bytes());
                out.extend_from_slice(&((hi - lo) as u32).to_le_bytes());
                for v in lo..hi {
                    let (x, y, z) = self.coords.coord(v * self.width, self.width);
                    bbox.add(x, y);
                    push_coord(out, x, y, z);
                }
            }
            GeometryKind::Polygon => {
                let rings = self.outer.expect("polygon carries ring offsets");
                let verts = self.inner.expect("polygon carries vertex offsets");
                let (r_lo, r_hi) = (rings[row] as usize, rings[row + 1] as usize);
                out.push(WKB_LE);
                out.extend_from_slice(&type_code(3).to_le_bytes());
                out.extend_from_slice(&((r_hi - r_lo) as u32).to_le_bytes());
                for r in r_lo..r_hi {
                    let (lo, hi) = (verts[r] as usize, verts[r + 1] as usize);
                    out.extend_from_slice(&((hi - lo) as u32).to_le_bytes());
                    for v in lo..hi {
                        let (x, y, z) = self.coords.coord(v * self.width, self.width);
                        bbox.add(x, y);
                        push_coord(out, x, y, z);
                    }
                }
            }
        }
        bbox
    }
}

// ----------------------------------------------------------------------------
// Covering bbox (GeoParquet 1.1)
// ----------------------------------------------------------------------------

/// Nearest `f32` at or below `v`. GeoParquet recommends a `float` covering with
/// outward rounding so the box a reader prunes against never excludes a row it
/// actually contains — plain `as f32` rounds to nearest and can shrink the box.
fn f32_floor(v: f64) -> f32 {
    let r = v as f32;
    if (r as f64) > v {
        next_down_f32(r)
    } else {
        r
    }
}

/// Nearest `f32` at or above `v` (see [`f32_floor`]).
fn f32_ceil(v: f64) -> f32 {
    let r = v as f32;
    if (r as f64) < v {
        next_up_f32(r)
    } else {
        r
    }
}

fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let bits = x.to_bits();
    f32::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = x.to_bits();
    f32::from_bits(if x > 0.0 { bits - 1 } else { bits + 1 })
}

/// Arrow type of the `covering.bbox` struct column.
fn bbox_fields() -> Fields {
    Fields::from(vec![
        Field::new("xmin", DataType::Float32, false),
        Field::new("ymin", DataType::Float32, false),
        Field::new("xmax", DataType::Float32, false),
        Field::new("ymax", DataType::Float32, false),
    ])
}

// ----------------------------------------------------------------------------
// Pass 1 — output schema union
// ----------------------------------------------------------------------------

/// The output schema of one layer, accumulated across tiles.
#[derive(Default)]
struct LayerSchema {
    /// Output fields in first-seen order (the encoder's column order).
    fields: Vec<Field>,
    /// Names dropped as derived renderer state.
    dropped: BTreeSet<String>,
}

impl LayerSchema {
    /// Fold one tile-layer's fields in, erroring on a type that disagrees with
    /// what an earlier tile declared for the same name.
    fn merge(&mut self, batch: &RecordBatch, layer: &str) -> Result<()> {
        let schema = batch.schema();
        let meta = schema.metadata();
        for field in schema.fields() {
            if field.name() == GEOMETRY_COLUMN {
                continue; // rebuilt as WKB, not carried through
            }
            let plan = plan_column(field, meta);
            let Some(out) = output_field(field, &plan) else {
                self.dropped.insert(field.name().clone());
                continue;
            };
            match self.fields.iter().find(|f| f.name() == out.name()) {
                Some(existing) if existing.data_type() != out.data_type() => bail!(
                    "layer '{layer}' column '{}' has type {} in one tile and {} in another; \
                     rebuild the archive with a declared input schema (stt-build reads property \
                     types from the source schema, which pins one type per column across every \
                     tile) or export the layers separately",
                    out.name(),
                    existing.data_type(),
                    out.data_type()
                ),
                Some(_) => {}
                None => self.fields.push(out),
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Export
// ----------------------------------------------------------------------------

/// Export a built packed archive to GeoParquet.
///
/// `out` is the output file when a single layer is written, and the *stem* for
/// `<stem>.<layer>.parquet` when the archive has several layers and none was
/// selected.
pub fn export(tileset: &PackedTileset, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    if let Some(b) = opts.bbox {
        if !(b[0] <= b[2] && b[1] <= b[3]) {
            bail!(
                "--bbox must be min_lon,min_lat,max_lon,max_lat with min <= max (got \
                 [{}, {}, {}, {}])",
                b[0],
                b[1],
                b[2],
                b[3]
            );
        }
    }
    if let (Some(s), Some(e)) = (opts.start, opts.end) {
        if s > e {
            bail!("--start ({s}) is after --end ({e})");
        }
    }

    let base_bucket = tileset.temporal_bucket_ms();
    let zoom = match opts.zoom {
        Some(z) => z,
        None => {
            deepest_base_zoom(tileset).ok_or_else(|| anyhow!("archive has no tiles to export"))?
        }
    };

    let in_zoom: Vec<&TileEntry> = tileset
        .entries()
        .iter()
        .filter(|e| e.zoom == zoom && is_base_tier(e, base_bucket))
        .collect();
    if in_zoom.is_empty() {
        bail!(
            "no base-tier tiles at zoom {zoom} (present: {})",
            present_zooms(tileset, base_bucket)
        );
    }
    let selected: Vec<&TileEntry> = in_zoom
        .iter()
        .copied()
        .filter(|e| opts.bbox.map_or(true, |b| tile_overlaps_bbox(e, &b)))
        .filter(|e| tile_overlaps_time(e, opts.start, opts.end))
        .collect();

    // Pass 1: union each layer's output schema across the selected tiles.
    let mut layers: BTreeMap<String, LayerSchema> = BTreeMap::new();
    for entry in &selected {
        for decoded in tileset
            .read_layers(entry)
            .with_context(|| format!("decoding tile z{}/{}/{}", entry.zoom, entry.x, entry.y))?
        {
            if opts.layer.as_deref().is_some_and(|l| l != decoded.name) {
                continue;
            }
            layers
                .entry(decoded.name.clone())
                .or_default()
                .merge(&decoded.batch, &decoded.name)?;
        }
    }
    if let Some(want) = opts.layer.as_deref() {
        if !layers.contains_key(want) {
            bail!("no layer named '{want}' in the selected tiles");
        }
    }

    // Pass 2: one file per layer.
    let multi = layers.len() > 1;
    let mut files = Vec::new();
    for (name, schema) in &layers {
        let path = layer_output_path(out, name, multi);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
        }
        files.push(write_layer(tileset, &selected, name, schema, &path, opts)?);
    }

    Ok(ExportReport {
        archive: tileset.manifest_path().display().to_string(),
        zoom,
        tiles_in_zoom: in_zoom.len(),
        tiles_selected: selected.len(),
        geoparquet_version: GEOPARQUET_VERSION.to_string(),
        geometry_encoding: opts.geometry_encoding.label().to_string(),
        files,
    })
}

fn present_zooms(tileset: &PackedTileset, base_bucket: u64) -> String {
    let zs: BTreeSet<u8> = tileset
        .entries()
        .iter()
        .filter(|e| is_base_tier(e, base_bucket))
        .map(|e| e.zoom)
        .collect();
    zs.iter()
        .map(|z| z.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// `out` verbatim for a single layer; `<stem>.<layer>.parquet` when several
/// layers share one `out`.
fn layer_output_path(out: &Path, layer: &str, multi: bool) -> PathBuf {
    if !multi {
        return out.to_path_buf();
    }
    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("parquet");
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("export");
    out.with_file_name(format!("{stem}.{layer}.{ext}"))
}

/// Full output schema of a layer: provenance, core columns, then the WKB
/// geometry and its covering bbox.
fn build_output_schema(schema: &LayerSchema) -> Result<Schema> {
    let mut fields: Vec<FieldRef> = Vec::with_capacity(schema.fields.len() + 5);
    for name in PROVENANCE_COLUMNS {
        if schema.fields.iter().any(|f| f.name() == name) {
            bail!(
                "the archive already carries a property column named '{name}', which collides \
                 with the tile-provenance column this export adds; rebuild with \
                 `stt-build --exclude {name}` or rename the property"
            );
        }
    }
    fields.push(Arc::new(Field::new(
        PROVENANCE_COLUMNS[0],
        DataType::UInt8,
        false,
    )));
    fields.push(Arc::new(Field::new(
        PROVENANCE_COLUMNS[1],
        DataType::UInt32,
        false,
    )));
    fields.push(Arc::new(Field::new(
        PROVENANCE_COLUMNS[2],
        DataType::UInt32,
        false,
    )));
    for f in &schema.fields {
        fields.push(Arc::new(f.clone()));
    }
    fields.push(Arc::new(Field::new(
        GEOMETRY_COLUMN,
        DataType::Binary,
        false,
    )));
    fields.push(Arc::new(Field::new(
        BBOX_COLUMN,
        DataType::Struct(bbox_fields()),
        true,
    )));
    Ok(Schema::new(fields))
}

/// Write one layer's Parquet file.
fn write_layer(
    tileset: &PackedTileset,
    entries: &[&TileEntry],
    layer: &str,
    layer_schema: &LayerSchema,
    path: &Path,
    opts: &ExportOptions,
) -> Result<ExportedFile> {
    let schema = Arc::new(build_output_schema(layer_schema)?);
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .set_created_by(format!("stt-optimize {}", env!("CARGO_PKG_VERSION")))
        .build();
    // The overridden schema has to be derived under the SAME conversion
    // settings the writer would have used, or the two disagree on some column
    // the override never meant to touch.
    let native = (opts.geometry_encoding == GeometryEncoding::Native)
        .then(|| native_geometry_schema(&schema, props.coerce_types()))
        .transpose()?;
    let mut writer_opts = ArrowWriterOptions::new().with_properties(props);
    if let Some(descr) = native {
        writer_opts = writer_opts.with_parquet_schema(descr);
    }
    let mut writer = ArrowWriter::try_new_with_options(file, Arc::clone(&schema), writer_opts)
        .with_context(|| format!("opening the Parquet writer for {}", path.display()))?;

    let mut rows: u64 = 0;
    let mut tiles = 0usize;
    let mut file_box = RowBox::empty();
    let mut geometry_types: BTreeSet<String> = BTreeSet::new();

    for entry in entries {
        let decoded = tileset
            .read_layers(entry)
            .with_context(|| format!("decoding tile z{}/{}/{}", entry.zoom, entry.x, entry.y))?;
        for d in decoded.iter().filter(|d| d.name == layer) {
            let Some(batch) = tile_batch(
                &d.batch,
                entry,
                &schema,
                opts,
                &mut file_box,
                &mut geometry_types,
            )?
            else {
                continue;
            };
            rows += batch.num_rows() as u64;
            tiles += 1;
            writer
                .write(&batch)
                .with_context(|| format!("writing a row batch to {}", path.display()))?;
        }
    }

    // The `geo` metadata carries the observed bbox and geometry types, so it
    // can only be built once every row has been seen; Parquet keeps its
    // key/value metadata in the footer, so appending it here still lands ahead
    // of `close`.
    let bbox = (!file_box.is_empty()).then(|| {
        [
            file_box.min_x,
            file_box.min_y,
            file_box.max_x,
            file_box.max_y,
        ]
    });
    let geo = geo_metadata(&geometry_types, bbox);
    writer.append_key_value_metadata(KeyValue::new(
        GEO_METADATA_KEY.to_string(),
        serde_json::to_string(&geo)?,
    ));
    writer
        .close()
        .with_context(|| format!("finalizing {}", path.display()))?;

    Ok(ExportedFile {
        path: path.display().to_string(),
        layer: layer.to_string(),
        rows,
        tiles,
        bbox,
        geometry_types: geometry_types.into_iter().collect(),
        dropped_columns: layer_schema.dropped.iter().cloned().collect(),
    })
}

/// Convert one decoded tile layer into an output batch, applying the row-level
/// bbox / time filters. `None` when no row survived.
fn tile_batch(
    batch: &RecordBatch,
    entry: &TileEntry,
    out_schema: &Arc<Schema>,
    opts: &ExportOptions,
    file_box: &mut RowBox,
    geometry_types: &mut BTreeSet<String>,
) -> Result<Option<RecordBatch>> {
    let schema = batch.schema();
    let geom_idx = schema
        .index_of(GEOMETRY_COLUMN)
        .map_err(|_| anyhow!("tile layer has no `geometry` column"))?;
    let geom = open_geometry(schema.field(geom_idx), batch.column(geom_idx))?;
    if geom.rows != batch.num_rows() {
        bail!(
            "geometry column has {} rows but the batch has {}",
            geom.rows,
            batch.num_rows()
        );
    }

    // Row-level time filter needs the absolute (pre-cast) columns.
    let starts = int64_column(batch, "start_time");
    let ends = int64_column(batch, "end_time");

    let mut keep: Vec<u32> = Vec::new();
    let mut wkb = BinaryBuilder::new();
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (
        Vec::<f32>::new(),
        Vec::<f32>::new(),
        Vec::<f32>::new(),
        Vec::<f32>::new(),
    );
    let mut bbox_valid: Vec<bool> = Vec::new();
    let mut scratch: Vec<u8> = Vec::with_capacity(64);

    for row in 0..batch.num_rows() {
        if let (Some(s), Some(e)) = (starts.map(|a| a.value(row)), ends.map(|a| a.value(row))) {
            // A feature is kept when its own [start, end] span overlaps the
            // requested window — pruning on start alone would drop a trip that
            // began before the window and is still running inside it.
            if opts.start.is_some_and(|q| e < q) || opts.end.is_some_and(|q| s > q) {
                continue;
            }
        }
        scratch.clear();
        let rb = geom.write_wkb(row, &mut scratch);
        if let Some(b) = opts.bbox {
            if !rb.intersects(&b) {
                continue;
            }
        }
        keep.push(row as u32);
        wkb.append_value(&scratch);
        if rb.is_empty() {
            xmin.push(0.0);
            ymin.push(0.0);
            xmax.push(0.0);
            ymax.push(0.0);
            bbox_valid.push(false);
        } else {
            xmin.push(f32_floor(rb.min_x));
            ymin.push(f32_floor(rb.min_y));
            xmax.push(f32_ceil(rb.max_x));
            ymax.push(f32_ceil(rb.max_y));
            bbox_valid.push(true);
            file_box.add(rb.min_x, rb.min_y);
            file_box.add(rb.max_x, rb.max_y);
        }
    }

    if keep.is_empty() {
        return Ok(None);
    }
    geometry_types.insert(geom.type_name());

    let n = keep.len();
    let indices = UInt32Array::from(keep);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(out_schema.fields().len());
    columns.push(Arc::new(UInt8Array::from(vec![entry.zoom; n])));
    columns.push(Arc::new(UInt32Array::from(vec![entry.x; n])));
    columns.push(Arc::new(UInt32Array::from(vec![entry.y; n])));

    let meta = schema.metadata();
    for field in out_schema.fields().iter().skip(PROVENANCE_COLUMNS.len()) {
        let name = field.name().as_str();
        if name == GEOMETRY_COLUMN || name == BBOX_COLUMN {
            continue;
        }
        match schema.index_of(name) {
            Ok(idx) => {
                let mut plan = plan_column(schema.field(idx), meta);
                // TB-11 extension 2: bind the per-feature anchors, row-selected
                // the same way the column about to be inflated is.
                if let ColumnPlan::VertexTimeFeatureDeltas { step, .. } = plan {
                    plan = ColumnPlan::VertexTimeFeatureDeltas {
                        step,
                        anchors: Arc::new(feature_anchors(batch, meta, &indices)?),
                    };
                }
                let taken = arrow::compute::take(batch.column(idx), &indices, None)
                    .with_context(|| format!("selecting rows of column '{name}'"))?;
                columns.push(apply_plan(&taken, &plan, field.data_type())?);
            }
            // A column another tile of this layer carries but this one does
            // not (per-tile property sealing — see the module docs). Null-fill
            // rather than fail: the value genuinely is absent here.
            Err(_) => columns.push(arrow::array::new_null_array(field.data_type(), n)),
        }
    }

    columns.push(Arc::new(wkb.finish()));
    columns.push(Arc::new(StructArray::new(
        bbox_fields(),
        vec![
            Arc::new(Float32Array::from(xmin)) as ArrayRef,
            Arc::new(Float32Array::from(ymin)),
            Arc::new(Float32Array::from(xmax)),
            Arc::new(Float32Array::from(ymax)),
        ],
        Some(NullBuffer::from(bbox_valid)),
    )));

    Ok(Some(
        RecordBatch::try_new(Arc::clone(out_schema), columns)
            .context("assembling the exported row batch")?,
    ))
}

fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a Int64Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
}

/// The GeoParquet `geo` file metadata.
///
/// `crs` is deliberately **omitted**: GeoParquet defines an absent `crs` as
/// OGC:CRS84 (longitude/latitude on WGS84), which is exactly what tile
/// coordinates are, while an explicit `null` would instead assert "CRS
/// unknown" and strand the data.
fn geo_metadata(geometry_types: &BTreeSet<String>, bbox: Option<[f64; 4]>) -> serde_json::Value {
    let mut column = serde_json::json!({
        "encoding": "WKB",
        "geometry_types": geometry_types.iter().cloned().collect::<Vec<_>>(),
        "covering": {
            "bbox": {
                "xmin": [BBOX_COLUMN, "xmin"],
                "ymin": [BBOX_COLUMN, "ymin"],
                "xmax": [BBOX_COLUMN, "xmax"],
                "ymax": [BBOX_COLUMN, "ymax"],
            }
        },
    });
    if let Some(b) = bbox {
        column["bbox"] = serde_json::json!([b[0], b[1], b[2], b[3]]);
    }
    serde_json::json!({
        "version": GEOPARQUET_VERSION,
        "primary_column": GEOMETRY_COLUMN,
        "columns": { GEOMETRY_COLUMN: column },
    })
}

/// The Parquet schema for `--geometry-encoding native`: exactly what the Arrow
/// converter would derive, with the `geometry` leaf re-declared as the native
/// `GEOMETRY` logical type.
///
/// Written this way, and not by hand, because reimplementing the Arrow→Parquet
/// mapping for every property type (dictionaries, lists, structs, timestamps)
/// to change one leaf is how a writer starts emitting subtly wrong types for
/// columns nobody was thinking about.
///
/// The logical type is reachable at this pin even though `parquet`'s
/// `geospatial` feature is off: the `LogicalType::Geometry` variant itself is
/// unconditional, only the Arrow *extension-type* bridge that would infer it
/// from field metadata is feature-gated.
fn native_geometry_schema(schema: &Schema, coerce_types: bool) -> Result<SchemaDescriptor> {
    let derived = ArrowSchemaConverter::new()
        .with_coerce_types(coerce_types)
        .convert(schema)
        .context("deriving the Parquet schema for the native GEOMETRY encoding")?;
    let root = derived.root_schema();
    let mut fields = Vec::with_capacity(root.get_fields().len());
    for f in root.get_fields() {
        if f.name() == GEOMETRY_COLUMN {
            fields.push(Arc::new(
                ParquetType::primitive_type_builder(GEOMETRY_COLUMN, PhysicalType::BYTE_ARRAY)
                    // `crs = None` is the Parquet default, OGC:CRS84 — the same
                    // claim the omitted GeoParquet `crs` key makes.
                    .with_logical_type(Some(LogicalType::geometry(None)))
                    // Carried over from the derived type rather than hardcoded:
                    // a repetition that disagrees with the Arrow schema makes
                    // the writer encode definition levels the footer denies.
                    .with_repetition(f.get_basic_info().repetition())
                    .build()
                    .context("building the native GEOMETRY column type")?,
            ));
        } else {
            fields.push(Arc::clone(f));
        }
    }
    let root = ParquetType::group_type_builder(root.name())
        .with_fields(fields)
        .build()
        .context("rebuilding the Parquet root schema")?;
    Ok(SchemaDescriptor::new(Arc::new(root)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_covering_rounds_outward() {
        // 0.1 is not representable in f32; the covering must widen, never
        // narrow, or a reader prunes away a row the box really contains.
        let v = 0.1f64;
        assert!((f32_floor(v) as f64) <= v);
        assert!((f32_ceil(v) as f64) >= v);
        let n = -0.1f64;
        assert!((f32_floor(n) as f64) <= n);
        assert!((f32_ceil(n) as f64) >= n);
        // An exactly representable value is left alone.
        assert_eq!(f32_floor(0.5), 0.5f32);
        assert_eq!(f32_ceil(0.5), 0.5f32);
    }

    #[test]
    fn geometry_encoding_parses() {
        assert_eq!(
            GeometryEncoding::parse("wkb").unwrap(),
            GeometryEncoding::Wkb
        );
        assert_eq!(
            GeometryEncoding::parse("native").unwrap(),
            GeometryEncoding::Native
        );
        assert!(GeometryEncoding::parse("geoarrow").is_err());
    }

    #[test]
    fn multi_layer_paths_get_a_layer_infix() {
        let out = Path::new("/tmp/city.parquet");
        assert_eq!(layer_output_path(out, "points", false), out.to_path_buf());
        assert_eq!(
            layer_output_path(out, "points", true),
            PathBuf::from("/tmp/city.points.parquet")
        );
    }

    /// A temporal-LOD archive's coarser tiles must not join a base export, or
    /// every feature ships twice.
    #[test]
    fn lod_entries_are_not_base_tier() {
        let mut e = TileEntry {
            zoom: 8,
            x: 1,
            y: 2,
            time_start: 0,
            time_end: 3_599_999,
            variant_id: stt_core::tile::RAW_VARIANT_ID,
            pack_id: 0,
            offset: 0,
            length: 1,
            uncompressed_size: 1,
            feature_count: 1,
            hilbert: 0,
            crc32c: 0,
            temporal_bucket_ms: Some(3_600_000),
            cover_t_min: None,
        };
        assert!(is_base_tier(&e, 3_600_000));
        e.temporal_bucket_ms = Some(86_400_000);
        assert!(!is_base_tier(&e, 3_600_000));
        // Archives without the per-entry column are base tiles by definition.
        e.temporal_bucket_ms = None;
        assert!(is_base_tier(&e, 3_600_000));
    }

    /// `cover_t_min` is the tight bound; `time_start` is only the bucket
    /// boundary, so pruning on it would keep tiles with no data in the window.
    #[test]
    fn time_prune_uses_the_tight_lower_bound() {
        let e = TileEntry {
            zoom: 5,
            x: 0,
            y: 0,
            time_start: 0,
            time_end: 3_599_999,
            variant_id: stt_core::tile::RAW_VARIANT_ID,
            pack_id: 0,
            offset: 0,
            length: 1,
            uncompressed_size: 1,
            feature_count: 1,
            hilbert: 0,
            crc32c: 0,
            temporal_bucket_ms: None,
            cover_t_min: Some(3_000_000),
        };
        assert!(!tile_overlaps_time(&e, None, Some(1_000_000)));
        assert!(tile_overlaps_time(&e, None, Some(3_100_000)));
        assert!(tile_overlaps_time(&e, Some(3_500_000), None));
        assert!(!tile_overlaps_time(&e, Some(3_600_000), None));
    }
}
