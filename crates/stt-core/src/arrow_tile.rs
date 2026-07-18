//! Arrow-based tile payload format.
//!
//! A tile's payload is one or more *layers*. Each layer is a single Arrow
//! [`RecordBatch`] serialised as an Arrow IPC **stream**. Geometry is encoded
//! using the GeoArrow interleaved-coordinate convention so the payload can be
//! consumed directly by GeoArrow-aware renderers (e.g. `@geoarrow/deck.gl`).
//!
//! ## Per-layer schema
//!
//! | column        | type                                    | notes                         |
//! |---------------|-----------------------------------------|-------------------------------|
//! | `id`          | `UInt64`                                | feature id                    |
//! | `start_time`  | `Int64`                                 | Unix ms, absolute             |
//! | `end_time`    | `Int64`                                 | Unix ms, absolute             |
//! | `geometry`    | GeoArrow point / linestring / polygon   | interleaved f64 lon/lat       |
//! | `vertex_time` | `List<UInt16>` deltas or `List<Int64>` (nullable) | per-vertex Unix ms (optional; see [`build_vertex_time_array`]) |
//! | `vertex_value`| `List<Float32>` (nullable)              | per-vertex scalar, e.g. SST (optional) |
//! | `triangles`   | `List<UInt16>` or `List<UInt32>`        | pre-baked earcut indices, feature-local (optional; polygon only) |
//! | `<property>`  | `Float64` or `Dictionary<UInt16,Utf8>`   | one column per property       |
//!
//! All layers in one tile are concatenated with a tiny frame so a tile can
//! carry, say, a linestring layer and a point layer side by side. Two frame
//! shapes exist, selected by [`EncoderConfig::format_version`] (the payload
//! side of the packed format's `manifest.formatVersion` — the two are bumped
//! in lockstep, see `docs/spec/stt-packed-format.md` §9):
//!
//! **v1** (`format_version: 1`, the 0.3.x wire shape — frozen, byte-identical):
//!
//! ```text
//! [u16 layer_count | ALIGNED_FRAME_FLAG]
//!   repeated: [u16 name_len][name utf8][u32 ipc_len][pad to 8][ipc stream bytes]
//! ```
//!
//! The leading u16's top bit ([`ALIGNED_FRAME_FLAG`]) marks the *aligned*
//! frame: zero padding after each `ipc_len` places every Arrow IPC stream at
//! an 8-byte boundary relative to the payload start, so readers can hand the
//! stream to an Arrow implementation zero-copy (Arrow buffers are 8-byte
//! aligned *within* a stream; the stream itself must start aligned for that
//! to survive). The pad length is not stored — readers derive it as
//! `(8 - pos % 8) % 8` from the position after `ipc_len`. Frames without the
//! flag (every archive written before the flag existed) carry no padding and
//! decode exactly as before.
//!
//! **v2** (`format_version: 2`, default for new `stt-build` output): a
//! sectioned frame that hoists each layer's Arrow IPC *schema message* into a
//! per-dataset **template** (referenced by blake3-128 hash, resolved through
//! the manifest's `schemas` table) so the per-tile schema tax disappears, and
//! carries the per-tile-varying metadata in a compact [`TILE_META`
//! section](TileMeta) instead of the schema:
//!
//! ```text
//! u16  0xFFFF                    # v2 escape (see FRAME_V2_ESCAPE)
//! u8   frame_version = 2
//! u8   flags = 0                 # reserved, MUST be 0
//! u16  layer_count
//! per layer:
//!   u16  name_len, [name utf8]
//!   u8   ref_kind_core           # 0 = INLINE_SCHEMA_CORE section present;
//!                                # 1 = the next 16 bytes are the template hash
//!   [16] core template hash     # iff ref_kind_core == 1
//!   u8   ref_kind_props          # 0/1 as above; 2 = NO props sections at all
//!   [16] props template hash    # iff ref_kind_props == 1
//!   u8   section_count
//!   per section (TOC): u8 tag, u32 length     # at-rest bytes, pad excluded
//!   [pad to 8, derived]
//!   per section: [section bytes][pad to 8, derived]
//! ```
//!
//! Reserved columns (`id`/times/geometry/vertex_*/`triangles`) form the CORE
//! batch; property columns form the PROPS batch with its own schema/template
//! (emitted only when properties exist). Each `*_BATCH` section is the IPC
//! stream's **tail** — dictionary batch(es) + record batch + end-of-stream —
//! and a reader materialises `concat(template, tail)` for a stock Arrow
//! reader. Unknown section tags are skippable via the TOC. Rows are
//! stable-sorted by `start_time` at encode (after id assignment), declared by
//! `TILE_META.sorted`. The v2 escape is unreachable from the v1 writer: the
//! v1 path caps `layer_count` below `0x7fff`, so an aligned v1 frame can
//! never start with `0xFFFF`.

use crate::error::{Error, Result};
use crate::types::GeometryType;
use arrow::array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Float32Builder,
    Float64Array, Int32Array, Int32Builder, Int64Array, Int64Builder, ListArray, ListBuilder,
    RecordBatch, StringArray, UInt16Array, UInt16Builder, UInt32Builder, UInt64Array, UInt8Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema, UInt16Type};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::ipc::{root_as_message, MessageHeader};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// Top bit of the layer frame's leading u16: marks an *aligned* frame whose
/// IPC streams are padded to 8-byte boundaries (see the module docs). The
/// remaining 15 bits carry the layer count, so a tile may have at most
/// `0x7fff` layers. Old frames never set this bit (layer counts are tiny).
pub const ALIGNED_FRAME_FLAG: u16 = 0x8000;

/// Alignment (bytes) of each layer's Arrow IPC stream within an aligned frame.
const FRAME_ALIGN: usize = 8;

// ----------------------------------------------------------------------------
// Layer frame v2 (packed formatVersion 2) — see the module docs.
// ----------------------------------------------------------------------------

/// Leading u16 of a **v2** layer frame. Deliberately unreachable from the v1
/// writer: the v1 encoder rejects tiles whose `count | ALIGNED_FRAME_FLAG`
/// would collide with this escape, so the two frame shapes are disjoint on
/// their first two bytes. Manifest `formatVersion` remains the authoritative
/// discriminator (design ★F6); this escape is defense-in-depth.
pub const FRAME_V2_ESCAPE: u16 = 0xFFFF;

/// `frame_version` byte of the v2 frame.
const FRAME_V2_VERSION: u8 = 2;

/// v2 section tag: full IPC schema prefix for the CORE batch (self-contained
/// mode, `ref_kind == REF_KIND_INLINE`).
pub const SECTION_INLINE_SCHEMA_CORE: u8 = 0x01;
/// v2 section tag: the canonical [`TileMeta`] JSON.
pub const SECTION_TILE_META: u8 = 0x02;
/// v2 section tag: CORE batch IPC tail (dict batches + record batch + EOS).
pub const SECTION_CORE_BATCH: u8 = 0x03;
/// v2 section tag: full IPC schema prefix for the PROPS batch (as 0x01).
pub const SECTION_INLINE_SCHEMA_PROPS: u8 = 0x04;
/// v2 section tag: PROPS batch IPC tail (as 0x03, props schema).
pub const SECTION_PROPS_BATCH: u8 = 0x05;

/// v2 `ref_kind`: the schema rides inline in an `INLINE_SCHEMA_*` section
/// (self-contained blob — no template registry needed to decode).
const REF_KIND_INLINE: u8 = 0;
/// v2 `ref_kind`: the next 16 bytes are the blake3-128 template hash,
/// resolved against the dataset's [`TemplateRegistry`].
const REF_KIND_TEMPLATE_HASH: u8 = 1;
/// v2 `ref_kind_props`: the layer has no property columns — no props
/// schema/template and no `PROPS_BATCH` section.
const REF_KIND_NO_PROPS: u8 = 2;

/// The frame format this encoder emits when nothing opts in explicitly —
/// v1, the frozen 0.3.x wire shape. `stt-build` opts into v2 explicitly
/// (its `--format-version` default is 2); `stt-serve` and every other
/// existing caller stays v1 without changes (design §7).
pub const FORMAT_VERSION_V1: u32 = 1;
/// The sectioned template-referencing frame (packed formatVersion 2).
pub const FORMAT_VERSION_V2: u32 = 2;

/// blake3 content hash truncated to 128 bits — the v2 template reference
/// (16 raw bytes in the frame; lowercase hex in `manifest.schemas`).
pub(crate) fn blake3_128(bytes: &[u8]) -> [u8; 16] {
    let hash = blake3::hash(bytes);
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    out
}

/// Thread-safe encode-side sink for the schema templates a v2 build produces.
///
/// The encoder [`record`](Self::record)s each layer's stripped schema prefix
/// and embeds the returned hash in the frame; `PackWriter::finalize` snapshots
/// the collector into the manifest's `schemas` table. Content-addressed keys
/// make the result independent of encode parallelism/order (design ★F1/E1):
/// two tiles sharing a schema record the same entry, and the snapshot is
/// sorted by hash.
#[derive(Debug, Default)]
pub struct TemplateCollector {
    templates: Mutex<BTreeMap<[u8; 16], Vec<u8>>>,
}

impl TemplateCollector {
    /// New, empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a template's bytes (idempotent), returning its blake3-128 hash —
    /// the 16-byte reference the v2 frame carries.
    pub fn record(&self, template: &[u8]) -> [u8; 16] {
        let hash = blake3_128(template);
        self.templates
            .lock()
            .unwrap()
            .entry(hash)
            .or_insert_with(|| template.to_vec());
        hash
    }

    /// Snapshot of every recorded `(hash, template bytes)`, sorted by hash
    /// (deduped by construction) — the byte-reproducible manifest order.
    pub fn snapshot(&self) -> Vec<([u8; 16], Vec<u8>)> {
        self.templates
            .lock()
            .unwrap()
            .iter()
            .map(|(h, b)| (*h, b.clone()))
            .collect()
    }

    /// Number of distinct templates recorded so far.
    pub fn len(&self) -> usize {
        self.templates.lock().unwrap().len()
    }

    /// Whether no template has been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Decode-side template lookup: blake3-128 hash → raw template bytes.
///
/// Built once per dataset from `manifest.schemas` (each entry hash-validated
/// at open) and threaded into [`decode_tile_with_templates`].
#[derive(Debug, Default, Clone)]
pub struct TemplateRegistry {
    templates: HashMap<[u8; 16], Arc<Vec<u8>>>,
}

impl TemplateRegistry {
    /// New, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a template, returning its blake3-128 hash.
    pub fn insert(&mut self, template: Vec<u8>) -> [u8; 16] {
        let hash = blake3_128(&template);
        self.templates.insert(hash, Arc::new(template));
        hash
    }

    /// Look a template up by its 16-byte hash.
    pub fn get(&self, hash: &[u8; 16]) -> Option<&[u8]> {
        self.templates.get(hash).map(|t| t.as_slice())
    }

    /// Iterate every `(hash, template bytes)` pair (arbitrary order). Lets a
    /// verbatim-repack tool seed a [`TemplateCollector`] from a source
    /// dataset's registry ([`crate::pack::PackWriter::with_seeded_templates`]).
    pub fn iter(&self) -> impl Iterator<Item = (&[u8; 16], &[u8])> {
        self.templates.iter().map(|(h, t)| (h, t.as_slice()))
    }

    /// Number of templates registered.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

/// The per-tile-varying metadata a v2 frame carries in its `TILE_META`
/// section instead of the (now dataset-constant, template-resident) Arrow
/// schema metadata. Canonical serialization (design §4.3): JSON, keys sorted
/// — field order below IS alphabetical and the `qa` map is a `BTreeMap` — no
/// whitespace. Readers MUST ignore unknown keys (serde's default here), so
/// the section can evolve additively. Presence rules: a key is present iff
/// the corresponding feature is (`qa` omits non-quantized columns entirely;
/// `t0` iff a start-time column exists; `vt` iff delta-encoded vertex_time;
/// `vb` iff a value matrix).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TileMeta {
    /// Per-property attribute-quantization affines, `column → [o, s]`
    /// (`value = o + q*s`) — the v1 `stt:qa` field metadata, hoisted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qa: Option<BTreeMap<String, (f64, f64)>>,
    /// Rows are stable-sorted by `start_time` (always `true` from this
    /// writer; a flag so Stage III can demote the sort without a frame bump).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sorted: Option<bool>,
    /// The v1 `stt:time_offset_ms` schema key: minimum feature start-time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t0: Option<i64>,
    /// The v1 `stt:vertex_value_buckets` schema key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vb: Option<u32>,
    /// The v1 `stt:vertex_time_origin_ms` / `stt:vertex_time_step_ms` pair,
    /// as `[origin_ms, step_ms]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vt: Option<(i64, u32)>,
}

/// GeoArrow extension-name metadata key.
const GEOARROW_EXT_KEY: &str = "ARROW:extension:name";

/// GeoArrow extension-*metadata* key — the sibling of [`GEOARROW_EXT_KEY`] that
/// carries the per-geometry-type JSON metadata (CRS, edge interpretation, ...).
const GEOARROW_EXT_META_KEY: &str = "ARROW:extension:metadata";

/// CRS advertised on every geometry field.
///
/// STT stores interleaved `[lon, lat]` in WGS84, i.e. **OGC:CRS84** (the GeoJSON
/// longitude-first axis order) — *not* `EPSG:4326`, which strict readers treat as
/// lat/lon. Emitting it as a GeoArrow `authority_code` makes every tile
/// self-describing to GDAL / GeoPandas / lonboard / QGIS; without it those
/// readers fall back to "unknown CRS" even though the geometry is plain lon/lat.
const GEOARROW_CRS_METADATA: &str = r#"{"crs":"OGC:CRS84","crs_type":"authority_code"}"#;

/// A single coordinate pair (lon, lat) in WGS84 degrees.
pub type Coord = [f64; 2];

/// Geometry for one layer, grouped by kind. Every feature in a layer shares
/// one kind — the tiler emits a separate layer per geometry type.
#[derive(Debug, Clone)]
pub enum GeometryColumn {
    /// One coordinate per feature.
    Point(Vec<Coord>),
    /// A vertex list per feature.
    LineString(Vec<Vec<Coord>>),
    /// A list of rings per feature (ring 0 is the exterior).
    Polygon(Vec<Vec<Vec<Coord>>>),
}

impl GeometryColumn {
    /// Number of features represented.
    pub fn len(&self) -> usize {
        match self {
            GeometryColumn::Point(v) => v.len(),
            GeometryColumn::LineString(v) => v.len(),
            GeometryColumn::Polygon(v) => v.len(),
        }
    }

    /// Whether the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The matching [`GeometryType`].
    pub fn kind(&self) -> GeometryType {
        match self {
            GeometryColumn::Point(_) => GeometryType::Point,
            GeometryColumn::LineString(_) => GeometryType::LineString,
            GeometryColumn::Polygon(_) => GeometryType::Polygon,
        }
    }

    /// The GeoArrow extension name for this geometry kind.
    fn geoarrow_name(&self) -> &'static str {
        match self {
            GeometryColumn::Point(_) => "geoarrow.point",
            GeometryColumn::LineString(_) => "geoarrow.linestring",
            GeometryColumn::Polygon(_) => "geoarrow.polygon",
        }
    }
}

/// Leaf element type of a [`PropertyColumn::Vector`] — the GPU upload type the
/// renderer binds the decoded child buffer as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorElem {
    /// `Float32` leaf — quaternion / scale / generic vec attributes.
    F32,
    /// `UInt8` leaf — packed RGBA colour (0–255 per channel).
    U8,
}

/// A property column. Values are per-feature and may be missing.
#[derive(Debug, Clone)]
pub enum PropertyColumn {
    /// Numeric values (f64).
    Numeric(Vec<Option<f64>>),
    /// Categorical / string values.
    Categorical(Vec<Option<String>>),
    /// A fixed-width interleaved vector per feature — a GPU-ready instance
    /// attribute baked at build time (e.g. a `[qx,qy,qz,qw]` surfel quaternion,
    /// a `[s_major,s_minor]` scale, an `[r,g,b,a]` colour). Encoded as
    /// `FixedSizeList<Float32|UInt8, width>` so the TS decoder hands the
    /// contiguous child buffer straight to deck.gl with **zero per-point work**
    /// (no main-thread re-interleave). `values` is row-major and flattened:
    /// feature `i` occupies `[i*width, (i+1)*width)`. No per-element nulls (a
    /// missing feature is encoded as a zero/identity vector by the producer).
    Vector {
        /// Components per feature (the FixedSizeList list size).
        width: usize,
        /// Leaf upload type (`F32` or `U8`).
        elem: VectorElem,
        /// Flattened row-major values; length must be `width * feature_count`.
        /// `U8` values are rounded+clamped to `[0,255]` at encode.
        values: Vec<f32>,
    },
}

/// One decoded/encodable tile layer.
#[derive(Debug, Clone)]
pub struct ColumnarLayer {
    /// Layer name (e.g. `"default"`, `"default_originals"`).
    pub name: String,
    /// Per-feature id.
    pub feature_ids: Vec<u64>,
    /// Per-feature start time (Unix ms, absolute).
    pub start_times: Vec<i64>,
    /// Per-feature end time (Unix ms, absolute).
    pub end_times: Vec<i64>,
    /// Geometry, one entry per feature.
    pub geometry: GeometryColumn,
    /// Optional per-vertex timestamps (Unix ms). When present, length equals
    /// the feature count and each inner vec matches that feature's vertex count.
    pub vertex_times: Option<Vec<Vec<i64>>>,
    /// Optional per-vertex scalar values (producer-defined; e.g. sea-surface
    /// temperature for the ocean-drifter dataset). When present, length equals
    /// the feature count and each inner vec matches that feature's vertex count.
    /// `NaN` marks a vertex with no value; renderers map it to a fallback color.
    pub vertex_values: Option<Vec<Vec<f32>>>,
    /// Optional per-vertex × per-time-bucket value matrix, flattened
    /// **vertex-major** per feature: `matrix[v * num_buckets + b]`. When
    /// present, length equals the feature count and each inner vec is
    /// `feature_vertex_count * num_buckets` long. Lets a static-geometry
    /// overview carry a per-vertex time series (e.g. flow-corridor counts per
    /// bin) so the renderer animates resident data instead of re-fetching
    /// geometry per bucket. Encoded as the tile's `vertex_value_matrix` column;
    /// `num_buckets` is recorded in schema metadata under
    /// `stt:vertex_value_buckets`.
    pub vertex_value_matrix: Option<Vec<Vec<f32>>>,
    /// Optional pre-baked triangle indices for polygon features (MLT-style).
    ///
    /// When present, `triangles.len() == feature_count` and each inner vec is
    /// the flat triangle-index list (groups of 3 vertex indices) produced by
    /// earcut at build time. Indices are LOCAL to the feature: they reference
    /// positions within that feature's own ring coordinates, so the renderer
    /// only needs to add the feature's `startIndex` to each one.
    ///
    /// Only meaningful for `GeometryColumn::Polygon`. Encoders that set this
    /// on a non-polygon layer will have it dropped at encode time.
    pub triangles: Option<Vec<Vec<u32>>>,
    /// Property columns, keyed by name. Each column has one value per feature.
    pub properties: Vec<(String, PropertyColumn)>,
}

impl ColumnarLayer {
    /// Feature count for this layer.
    pub fn feature_count(&self) -> usize {
        self.feature_ids.len()
    }

    /// Validate that every column has a consistent length.
    fn validate(&self) -> Result<()> {
        let n = self.feature_ids.len();
        let check = |label: &str, len: usize| -> Result<()> {
            if len != n {
                return Err(Error::Other(format!(
                    "tile layer '{}': {} has {} entries, expected {}",
                    self.name, label, len, n
                )));
            }
            Ok(())
        };
        check("start_times", self.start_times.len())?;
        check("end_times", self.end_times.len())?;
        check("geometry", self.geometry.len())?;
        if let Some(vt) = &self.vertex_times {
            check("vertex_times", vt.len())?;
        }
        if let Some(vv) = &self.vertex_values {
            check("vertex_values", vv.len())?;
        }
        if let Some(vm) = &self.vertex_value_matrix {
            check("vertex_value_matrix", vm.len())?;
        }
        if let Some(tri) = &self.triangles {
            check("triangles", tri.len())?;
        }
        for (name, col) in &self.properties {
            match col {
                PropertyColumn::Numeric(v) => check(&format!("property '{}'", name), v.len())?,
                PropertyColumn::Categorical(v) => check(&format!("property '{}'", name), v.len())?,
                PropertyColumn::Vector { width, values, .. } => {
                    if *width == 0 {
                        return Err(Error::Other(format!(
                            "tile layer '{}': vector property '{}' has width 0",
                            self.name, name
                        )));
                    }
                    if values.len() != width * n {
                        return Err(Error::Other(format!(
                            "tile layer '{}': vector property '{}' has {} values, expected {} ({} × {})",
                            self.name,
                            name,
                            values.len(),
                            width * n,
                            width,
                            n
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Schema-metadata key set on layers that carry pre-baked triangle indices.
pub const TRIANGLES_METADATA_KEY: &str = "stt:has_triangles";

/// Tessellate one polygon feature (a list of rings) using earcut. Returns the
/// flat triangle index list — each triple of indices is one triangle, indices
/// are LOCAL (relative to the start of the feature's coordinate run, where
/// the exterior ring sits first followed by every hole).
///
/// Returns an empty vec for degenerate inputs (no exterior ring, <3 vertices).
pub fn tessellate_polygon(rings: &[Vec<Coord>]) -> Vec<u32> {
    if rings.is_empty() {
        return Vec::new();
    }
    // Flatten coords into the [x0, y0, x1, y1, ...] format earcutr expects.
    let mut flat: Vec<f64> = Vec::with_capacity(rings.iter().map(|r| r.len()).sum::<usize>() * 2);
    // Hole offsets are vertex indices (not coord-pair indices) where each
    // hole begins. The first hole starts after the exterior ring.
    let mut hole_indices: Vec<usize> = Vec::with_capacity(rings.len().saturating_sub(1));
    let mut running = 0usize;
    for (i, ring) in rings.iter().enumerate() {
        if i > 0 {
            hole_indices.push(running);
        }
        for [x, y] in ring {
            flat.push(*x);
            flat.push(*y);
        }
        running += ring.len();
    }
    if running < 3 {
        return Vec::new();
    }
    match earcutr::earcut(&flat, &hole_indices, 2) {
        Ok(tris) => tris.into_iter().map(|i| i as u32).collect(),
        Err(_) => Vec::new(),
    }
}

// ----------------------------------------------------------------------------
// Geometry array construction (GeoArrow interleaved)
// ----------------------------------------------------------------------------

/// Build an `i32` offset buffer from per-element counts.
///
/// Errors when the accumulated count exceeds `i32::MAX`: Arrow list offsets
/// are 32-bit, so a larger layer would wrap into corrupt offsets in release
/// builds. No budget cap guarantees safety here (budgets are optional), so the
/// overflow must be a hard, descriptive failure.
fn offsets_from_counts(counts: impl Iterator<Item = usize>) -> Result<OffsetBuffer<i32>> {
    let mut acc = 0i32;
    let mut offsets = vec![0i32];
    for c in counts {
        acc = i32::try_from(c)
            .ok()
            .and_then(|c| acc.checked_add(c))
            .ok_or_else(|| {
                Error::Other(format!(
                    "layer geometry exceeds {} total vertices/rings, which Arrow's \
                     32-bit list offsets cannot address; split the layer",
                    i32::MAX
                ))
            })?;
        offsets.push(acc);
    }
    Ok(OffsetBuffer::new(offsets.into()))
}

/// Meters per degree of latitude (WGS84 mean) — the constant the coordinate
/// quantizer sizes its grid from. Longitude scales by `cos(lat)`.
const M_PER_DEG_LAT: f64 = 111_320.0;

/// Schema-metadata key (on the `geometry` field) that flags a tile whose
/// coordinates are fixed-point `i32` grid indices rather than GeoArrow Float64
/// lon/lat. Its value is the [`QuantAffine`] JSON; absent ⇒ standard Float64.
pub const STT_QUANT_META_KEY: &str = "stt:quant";

/// Per-layer coordinate-quantization affine. Coordinates ship as `i32` grid
/// indices; the decoder reconstructs `lon = x0 + qx*sx`, `lat = y0 + qy*sy`.
/// `sx`/`sy` (degrees per quantum) are sized from a target ground precision in
/// meters at the layer's mid-latitude, so the worst-case error is ≤ half a
/// quantum (~`meters/2`). This trades GeoArrow self-describing Float64 for size
/// (coords are the dominant, near-incompressible column) and is opt-in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantAffine {
    pub x0: f64,
    pub y0: f64,
    pub sx: f64,
    pub sy: f64,
    /// Z-axis origin/step (metres), present ONLY for 3D point geometry
    /// (`FixedSizeList<i32,3>`). `None` ⇒ plain 2D coords, byte-identical to the
    /// historical affine (the `z0`/`sz` keys are simply omitted from the JSON).
    pub z0: Option<f64>,
    pub sz: Option<f64>,
}

impl QuantAffine {
    fn to_json(&self) -> String {
        // Full f64 round-trip precision (17 sig digits) so decode is exact. The
        // z keys are emitted only for 3D affines, so a 2D affine is byte-identical.
        match (self.z0, self.sz) {
            (Some(z0), Some(sz)) => format!(
                r#"{{"x0":{:.17e},"y0":{:.17e},"sx":{:.17e},"sy":{:.17e},"z0":{:.17e},"sz":{:.17e}}}"#,
                self.x0, self.y0, self.sx, self.sy, z0, sz
            ),
            _ => format!(
                r#"{{"x0":{:.17e},"y0":{:.17e},"sx":{:.17e},"sy":{:.17e}}}"#,
                self.x0, self.y0, self.sx, self.sy
            ),
        }
    }

    /// Parse the affine from its [`STT_QUANT_META_KEY`] JSON value. The TS
    /// reader applies the identical reconstruction (`tile.ts`).
    pub fn from_json(s: &str) -> Option<QuantAffine> {
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64());
        Some(QuantAffine {
            x0: f("x0")?,
            y0: f("y0")?,
            sx: f("sx")?,
            sy: f("sy")?,
            z0: f("z0"),
            sz: f("sz"),
        })
    }

    /// Reconstruct longitude from a quantized x grid index.
    #[inline]
    pub fn lon(&self, qx: i32) -> f64 {
        self.x0 + qx as f64 * self.sx
    }
    /// Reconstruct latitude from a quantized y grid index.
    #[inline]
    pub fn lat(&self, qy: i32) -> f64 {
        self.y0 + qy as f64 * self.sy
    }

    #[inline]
    fn qx(&self, lon: f64) -> i32 {
        (((lon - self.x0) / self.sx).round() as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }
    #[inline]
    fn qy(&self, lat: f64) -> i32 {
        (((lat - self.y0) / self.sy).round() as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }
    /// Quantize a metre altitude to a grid index (3D affines only; `z0`/`sz` set).
    ///
    /// Altitude is unbounded input (unlike lon/lat), so an index outside i32 is
    /// a hard error identifying the offending value — the old silent clamp
    /// snapped such points to ±i32::MAX quanta without a trace.
    #[inline]
    fn qz(&self, z: f64) -> Result<i32> {
        let z0 = self.z0.unwrap_or(0.0);
        let sz = self.sz.unwrap_or(1.0);
        let q = ((z - z0) / sz).round();
        if !(q >= i32::MIN as f64 && q <= i32::MAX as f64) {
            return Err(Error::Other(format!(
                "altitude {z} does not fit the quantization grid (origin {z0}, step {sz}): \
                 index {q} is outside i32; use a coarser --quantize-coords precision or drop \
                 the point-elevation fold for this dataset"
            )));
        }
        Ok(q as i32)
    }
}

/// Finest coordinate-quantization precision (meters) the world-anchored grid
/// supports: the grid is anchored at `(-180, -90)` with a uniform step of
/// `meters / M_PER_DEG_LAT` degrees, so the largest longitude index is
/// `360 * M_PER_DEG_LAT / meters`. Below this floor (≈ 0.0187 m, ~19 mm) that
/// index exceeds `i32::MAX` and quantization would silently snap far-east
/// coordinates to wrong locations — so finer precisions are rejected.
pub const MIN_QUANTIZE_COORDS_M: f64 = 360.0 * M_PER_DEG_LAT / (i32::MAX as f64);

/// Validate a coordinate-quantization precision against the world grid's
/// [`MIN_QUANTIZE_COORDS_M`] floor. `meters <= 0` (quantization off) passes.
/// Public so config-building paths ([`EncoderConfig`] consumers like
/// `stt-serve`) can fail fast at startup with the same error the global
/// setter and the encode-time guard produce.
pub fn validate_quantize_coords_m(meters: f64) -> Result<()> {
    if meters > 0.0 && meters < MIN_QUANTIZE_COORDS_M {
        return Err(Error::Other(format!(
            "coordinate quantization precision {meters} m is finer than the minimum \
             {MIN_QUANTIZE_COORDS_M} m (~19 mm): the world-anchored grid's ±180° longitude \
             index would overflow i32 and snap points to wrong locations"
        )));
    }
    Ok(())
}

/// The fixed, dataset-independent quantization grid for a target ground
/// precision in meters, or `None` when off (`meters <= 0`).
///
/// The origin is the world corner `(-180, -90)` and the step is uniform in
/// degrees (`meters / M_PER_DEG_LAT`) — deliberately **not** a per-tile
/// bbox-relative grid. A per-tile grid gives the same coordinate different
/// indices in different tiles, which destroys the packed format's
/// content-addressed blob dedup (measured: +61% on a dedup-heavy dataset). A
/// single world grid keeps identical geometry byte-identical across tiles.
///
/// Latitude precision is exactly `meters`; longitude is `meters * cos(lat)` —
/// i.e. always ≤ `meters` (finer toward the poles), never coarser than asked.
/// At 1 m the largest index is `360 * M_PER_DEG_LAT ≈ 4.0e7`, well within i32.
fn world_grid_affine(meters: f64) -> Option<QuantAffine> {
    if !(meters > 0.0) {
        return None;
    }
    let step = meters / M_PER_DEG_LAT;
    Some(QuantAffine {
        x0: -180.0,
        y0: -90.0,
        sx: step,
        sy: step,
        z0: None,
        sz: None,
    })
}

/// The 3D variant of [`world_grid_affine`]: same world-grid xy plus a Z axis
/// quantized to the SAME ground precision in metres, origin pinned to a fixed
/// global datum (`z0 = 0`) so identical surfels stay byte-identical across tiles
/// (dedup-preserving, like the xy world grid). For point clouds whose altitude
/// rides the geometry's 3rd coordinate (`--point-elevation-column`).
fn world_grid_affine_3d(meters: f64) -> Option<QuantAffine> {
    let a = world_grid_affine(meters)?;
    Some(QuantAffine {
        z0: Some(0.0),
        sz: Some(meters),
        ..a
    })
}

/// Optionally quantizing coordinates to an `i32` grid
/// (when `quant` is `Some`). The List/FixedSizeList nesting and offset buffers
/// are identical either way — only the leaf changes from `Float64` to `Int32`,
/// so `quant = None` is byte-identical to the historical encoder.
///
/// When `point_elev` is `Some` (POINT geometry only), each point's altitude is
/// folded in as a 3rd coordinate: the leaf becomes `FixedSizeList<_,3>`
/// (`[x,y,z]`), quantized with the affine's `z0`/`sz` when quantizing. The
/// renderer then binds 3D positions zero-copy (no pad). `None` ⇒ 2D, unchanged.
fn build_geometry_array_q(
    geom: &GeometryColumn,
    quant: Option<&QuantAffine>,
    point_elev: Option<&[f64]>,
) -> Result<ArrayRef> {
    // 3D POINT path: interleave [x,y,z] into a FixedSizeList<_,3> leaf.
    if let (GeometryColumn::Point(points), Some(elev)) = (geom, point_elev) {
        let list_size = 3;
        let dt = if quant.is_some() {
            DataType::Int32
        } else {
            DataType::Float64
        };
        let field = Arc::new(Field::new("xyz", dt, false));
        let child: ArrayRef = match quant {
            Some(q) => {
                let mut iv = Vec::with_capacity(points.len() * 3);
                for (i, [x, y]) in points.iter().enumerate() {
                    iv.push(q.qx(*x));
                    iv.push(q.qy(*y));
                    iv.push(q.qz(elev.get(i).copied().unwrap_or(0.0))?);
                }
                Arc::new(Int32Array::from(iv))
            }
            None => {
                let mut flat = Vec::with_capacity(points.len() * 3);
                for (i, [x, y]) in points.iter().enumerate() {
                    flat.push(*x);
                    flat.push(*y);
                    flat.push(elev.get(i).copied().unwrap_or(0.0));
                }
                Arc::new(Float64Array::from(flat))
            }
        };
        return Ok(Arc::new(FixedSizeListArray::new(
            field, list_size, child, None,
        )));
    }

    let coord_field = || {
        let dt = if quant.is_some() {
            DataType::Int32
        } else {
            DataType::Float64
        };
        Arc::new(Field::new("xy", dt, false))
    };
    // Turn a flat `[x,y,x,y,...]` f64 run into the FixedSizeList<_,2> leaf,
    // quantizing to i32 grid indices when an affine is supplied.
    let make_leaf = |flat: Vec<f64>| -> ArrayRef {
        match quant {
            Some(q) => {
                let mut iv = Vec::with_capacity(flat.len());
                let mut i = 0;
                while i + 1 < flat.len() {
                    iv.push(q.qx(flat[i]));
                    iv.push(q.qy(flat[i + 1]));
                    i += 2;
                }
                Arc::new(FixedSizeListArray::new(
                    coord_field(),
                    2,
                    Arc::new(Int32Array::from(iv)),
                    None,
                ))
            }
            None => Arc::new(FixedSizeListArray::new(
                coord_field(),
                2,
                Arc::new(Float64Array::from(flat)),
                None,
            )),
        }
    };

    Ok(match geom {
        GeometryColumn::Point(points) => {
            let mut flat = Vec::with_capacity(points.len() * 2);
            for [x, y] in points {
                flat.push(*x);
                flat.push(*y);
            }
            make_leaf(flat)
        }
        GeometryColumn::LineString(lines) => {
            let mut flat: Vec<f64> = Vec::new();
            for line in lines {
                for [x, y] in line {
                    flat.push(*x);
                    flat.push(*y);
                }
            }
            let coords = make_leaf(flat);
            let offsets = offsets_from_counts(lines.iter().map(|l| l.len()))?;
            let vertex_field = Arc::new(Field::new("vertices", coords.data_type().clone(), false));
            Arc::new(ListArray::new(vertex_field, offsets, coords, None))
        }
        GeometryColumn::Polygon(polys) => {
            // Flatten all coordinates, ring sizes, and ring counts per feature.
            let mut flat: Vec<f64> = Vec::new();
            let mut ring_sizes: Vec<usize> = Vec::new();
            let mut rings_per_feature: Vec<usize> = Vec::new();
            for feature in polys {
                rings_per_feature.push(feature.len());
                for ring in feature {
                    ring_sizes.push(ring.len());
                    for [x, y] in ring {
                        flat.push(*x);
                        flat.push(*y);
                    }
                }
            }
            let coords = make_leaf(flat);
            // Ring level: List<FixedSizeList>.
            let ring_offsets = offsets_from_counts(ring_sizes.into_iter())?;
            let vertex_field = Arc::new(Field::new("vertices", coords.data_type().clone(), false));
            let rings: ArrayRef =
                Arc::new(ListArray::new(vertex_field, ring_offsets, coords, None));
            // Feature level: List<List<FixedSizeList>>.
            let feature_offsets = offsets_from_counts(rings_per_feature.into_iter())?;
            let ring_field = Arc::new(Field::new("rings", rings.data_type().clone(), false));
            Arc::new(ListArray::new(ring_field, feature_offsets, rings, None))
        }
    })
}

/// Schema metadata keys for the v3 per-vertex time encoding.
const VERTEX_TIME_ORIGIN_KEY: &str = "stt:vertex_time_origin_ms";
const VERTEX_TIME_STEP_KEY: &str = "stt:vertex_time_step_ms";
/// Number of time buckets packed into each row of the `vertex_value_matrix`
/// column. The renderer reshapes the flat vertex-major list back into a
/// `[vertex][bucket]` grid using this count.
const VERTEX_VALUE_BUCKETS_KEY: &str = "stt:vertex_value_buckets";
/// Baked minimum feature start-time (integer Unix ms) for the layer. The TS
/// decoder relativizes every start/end time against this value so the times
/// fit an f32; baking it here lets the decoder skip its client-side min-scan
/// over the whole start-time column. Mirrors exactly what the decoder computes
/// (the min of the `start_time` column) — see `packages/core/src/tile.ts`.
const TIME_OFFSET_MS_KEY: &str = "stt:time_offset_ms";

/// Default ceiling (ms) on the u16-delta `vertex_time` quantization step.
///
/// The u16 encoding trades precision for a 4x payload shrink, but the step is
/// derived from the layer's temporal span — without a ceiling, a wide layer
/// (e.g. a 30-day temporal-LOD bucket) silently quantizes vertex times to
/// tens of seconds. A 1 s ceiling keeps the worst-case error below anything
/// playback can show; layers needing a coarser step fall back to the exact
/// `List<Int64>` shape instead (no thinning).
pub const DEFAULT_VERTEX_TIME_MAX_STEP_MS: u32 = 1000;

/// Process-wide step ceiling, settable once at startup (e.g. from
/// `stt-build --vertex-time-precision`). Reads/writes are monotonic-free
/// config, so `Relaxed` is sufficient.
static VERTEX_TIME_MAX_STEP_MS: AtomicU32 = AtomicU32::new(DEFAULT_VERTEX_TIME_MAX_STEP_MS);

/// Latch so the "u16-delta ceiling exceeded" warning fires at most once per
/// process instead of once per tile.
static VERTEX_TIME_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

/// Override the u16-delta `vertex_time` step ceiling for every subsequent
/// [`encode_layer`] call. Values below 1 ms clamp to 1 (every layer with a
/// span beyond u16 milliseconds then takes the exact `List<Int64>` path).
pub fn set_vertex_time_max_step_ms(ms: u32) {
    VERTEX_TIME_MAX_STEP_MS.store(ms.max(1), Ordering::Relaxed);
}

/// The currently configured u16-delta `vertex_time` step ceiling (ms).
pub fn vertex_time_max_step_ms() -> u32 {
    VERTEX_TIME_MAX_STEP_MS.load(Ordering::Relaxed)
}

/// Build-global coordinate quantization precision, in **micrometers** (lets the
/// `AtomicU32` carry sub-mm..km without a float). `0` = off (Float64 coords).
/// Set once per build (e.g. `stt-build --quantize-coords`); read by the default
/// [`encode_tile`] / [`encode_layer`] path so it covers both the streaming and
/// in-memory builders without threading through every tile.
static QUANTIZE_COORDS_UM: AtomicU32 = AtomicU32::new(0);

/// Set the build-global coordinate quantization precision in meters for every
/// subsequent default [`encode_tile`] call. `<= 0` (the default) turns it off —
/// coordinates stay Float64 GeoArrow. See [`encode_layer_quantized`] for the
/// size/precision trade-off. Errors (storing nothing) for a positive precision
/// below [`MIN_QUANTIZE_COORDS_M`], which would overflow the world grid's
/// longitude index.
pub fn set_quantize_coords_m(meters: f64) -> Result<()> {
    validate_quantize_coords_m(meters)?;
    let um = if meters > 0.0 {
        (meters * 1.0e6).round().clamp(1.0, u32::MAX as f64) as u32
    } else {
        0
    };
    QUANTIZE_COORDS_UM.store(um, Ordering::Relaxed);
    Ok(())
}

/// The build-global quantization precision in meters, or `None` when off.
pub fn quantize_coords_m() -> Option<f64> {
    let um = QUANTIZE_COORDS_UM.load(Ordering::Relaxed);
    (um > 0).then(|| um as f64 / 1.0e6)
}

/// Field-metadata key flagging a *numeric property* column that ships as
/// fixed-point integer indices (smallest of `UInt16`/`Int32`) instead of
/// `Float64`. Its value is the [`AttrQuant`] JSON (`value = o + q*s`). Sibling
/// of [`STT_QUANT_META_KEY`] for geometry; lives on the property field, so a
/// reader reconstructs Float64 the same way it does coordinates.
pub const STT_QUANT_ATTR_META_KEY: &str = "stt:qa";

/// Per-numeric-property quantization affine: `value = o + q * s`, where `o` is
/// the column minimum (the dequantization offset) and `s` the requested ground
/// precision (the step). Reconstruction is lossy to ≤ `s/2`; the reader applies
/// the identical math (`tile.ts`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttrQuant {
    pub o: f64,
    pub s: f64,
}

impl AttrQuant {
    fn to_json(&self) -> String {
        // Full f64 round-trip precision so the offset/step decode exactly.
        format!(r#"{{"o":{:.17e},"s":{:.17e}}}"#, self.o, self.s)
    }

    /// Parse the affine from its [`STT_QUANT_ATTR_META_KEY`] JSON value.
    pub fn from_json(s: &str) -> Option<AttrQuant> {
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(AttrQuant {
            o: v.get("o")?.as_f64()?,
            s: v.get("s")?.as_f64()?,
        })
    }

    /// Reconstruct the original value from a quantized index.
    #[inline]
    pub fn value(&self, q: i64) -> f64 {
        self.o + q as f64 * self.s
    }
}

/// Build-global map of `property-name → ground precision (units)`. A numeric
/// property named here is stored quantized (see [`build_quantized_numeric`]);
/// every other numeric property stays `Float64`. Set once per build from
/// `stt-build --quantize-attr name=prec`. Empty (the default) ⇒ all numeric
/// properties stay `Float64`, byte-identical to the historical encoder.
fn quant_attrs_cell() -> &'static RwLock<HashMap<String, f64>> {
    static A: OnceLock<RwLock<HashMap<String, f64>>> = OnceLock::new();
    A.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Replace the build-global numeric-property quantization map.
pub fn set_quantize_attrs(map: HashMap<String, f64>) {
    *quant_attrs_cell().write().unwrap() = map;
}

/// The current numeric-property quantization map (a clone).
pub fn quantize_attrs() -> HashMap<String, f64> {
    quant_attrs_cell().read().unwrap().clone()
}

/// When set, EVERY `Float64` numeric property not given an explicit precision in
/// the [`set_quantize_attrs`] map is quantized automatically: its step is sized
/// so the column's full `[min, max]` range spans the 16-bit index space
/// (`step = (max-min)/65535`), i.e. a range-adaptive `UInt16` with ~65k levels.
/// This is the "born-optimized" generation default — no per-column precision to
/// pick, and >=16 bits of dynamic range is visually lossless for the scalar
/// fields STT carries (magnitude, depth, altitude, speed, SST, dBZ, ...). The
/// default-off keeps `stt-build` byte-identical unless a caller opts in.
static QUANTIZE_ATTRS_AUTO: AtomicBool = AtomicBool::new(false);

/// Enable/disable automatic range-adaptive quantization of every otherwise-raw
/// `Float64` numeric property (see [`QUANTIZE_ATTRS_AUTO`]). Explicit precisions
/// in [`set_quantize_attrs`] always win.
pub fn set_quantize_attrs_auto(on: bool) {
    QUANTIZE_ATTRS_AUTO.store(on, Ordering::Relaxed);
}

/// Whether automatic numeric-property quantization is enabled.
pub fn quantize_attrs_auto() -> bool {
    QUANTIZE_ATTRS_AUTO.load(Ordering::Relaxed)
}

/// A build-time directive to fuse several scalar numeric properties into one
/// GPU-ready interleaved [`PropertyColumn::Vector`]. The component order is the
/// vector's component order (e.g. `["qx","qy","qz","qw"]` → `instanceQuaternions`).
/// Applied at encode time (like the quantization maps), so the producer keeps
/// emitting plain scalar columns and the renderer still binds the result
/// zero-copy. See [`set_vector_groups`].
#[derive(Debug, Clone)]
pub struct VectorGroup {
    /// Output column name (the FixedSizeList field name the decoder keys on).
    pub name: String,
    /// Source scalar-property names, in component order.
    pub components: Vec<String>,
    /// Leaf upload type (`F32` for quat/scale, `U8` for 0–255 RGBA).
    pub elem: VectorElem,
}

/// Build-global list of vector groups (see [`VectorGroup`]). Set once per build
/// from `stt-build --vector-group`. Empty (the default) ⇒ every property stays a
/// scalar column, byte-identical to the historical encoder.
fn vector_groups_cell() -> &'static RwLock<Vec<VectorGroup>> {
    static A: OnceLock<RwLock<Vec<VectorGroup>>> = OnceLock::new();
    A.get_or_init(|| RwLock::new(Vec::new()))
}

/// Replace the build-global vector-group list.
pub fn set_vector_groups(groups: Vec<VectorGroup>) {
    *vector_groups_cell().write().unwrap() = groups;
}

/// The current vector-group list (a clone).
pub fn vector_groups() -> Vec<VectorGroup> {
    vector_groups_cell().read().unwrap().clone()
}

/// Build-global name of the numeric property to fold into POINT geometry as the
/// 3rd (altitude) coordinate, so the tile ships true 3D points
/// (`FixedSizeList<_,3>`) the renderer binds zero-copy — no per-point pad to 3D
/// on the main thread. The named column is REMOVED from the property set (it
/// lives in the geometry instead). Empty (default) ⇒ plain 2D points.
fn point_elevation_column_cell() -> &'static RwLock<String> {
    static A: OnceLock<RwLock<String>> = OnceLock::new();
    A.get_or_init(|| RwLock::new(String::new()))
}

/// Set the build-global point-elevation column name (see above). Empty disables.
pub fn set_point_elevation_column(name: &str) {
    *point_elevation_column_cell().write().unwrap() = name.to_string();
}

/// The current point-elevation column name (empty ⇒ disabled).
pub fn point_elevation_column() -> String {
    point_elevation_column_cell().read().unwrap().clone()
}

/// Build-global layer-frame format version ([`FORMAT_VERSION_V1`] |
/// [`FORMAT_VERSION_V2`]). Defaults to v1 so every existing globals-path
/// caller (including `stt-serve`, which must stay v1 per the serve protocol)
/// is byte-identical to the 0.3.x writer; `stt-build` opts the offline build
/// into v2 explicitly.
static FORMAT_VERSION: AtomicU32 = AtomicU32::new(FORMAT_VERSION_V1);

/// Set the build-global frame format version for every subsequent default
/// [`encode_tile`] call. Errors (storing nothing) on anything but 1 or 2.
pub fn set_format_version(v: u32) -> Result<()> {
    if v != FORMAT_VERSION_V1 && v != FORMAT_VERSION_V2 {
        return Err(Error::Other(format!(
            "unsupported format version {v} (this writer emits 1 or 2)"
        )));
    }
    FORMAT_VERSION.store(v, Ordering::Relaxed);
    Ok(())
}

/// The build-global frame format version.
pub fn format_version() -> u32 {
    FORMAT_VERSION.load(Ordering::Relaxed)
}

/// Build-global template collector for the v2 hash-referencing mode. `None`
/// (the default) makes a v2 encode self-contained (inline schema sections);
/// `stt-build` installs the `PackWriter`'s collector here so the offline
/// globals-path encodes reference templates the manifest will carry.
fn template_collector_cell() -> &'static RwLock<Option<Arc<TemplateCollector>>> {
    static A: OnceLock<RwLock<Option<Arc<TemplateCollector>>>> = OnceLock::new();
    A.get_or_init(|| RwLock::new(None))
}

/// Install (or clear, with `None`) the build-global [`TemplateCollector`].
pub fn set_template_collector(collector: Option<Arc<TemplateCollector>>) {
    *template_collector_cell().write().unwrap() = collector;
}

/// The current build-global template collector, if any.
pub fn template_collector() -> Option<Arc<TemplateCollector>> {
    template_collector_cell().read().unwrap().clone()
}

/// Resolved, explicit encoder settings — the values the tile encoder reads at
/// encode time (coordinate + attribute quantization, vector grouping, the
/// point-elevation fold, vertex-time precision).
///
/// Historically these lived only in process-wide mutable statics set via the
/// `set_*` functions above; that made a new encode caller silently inherit
/// whatever the last `set_*` left behind (the origin of the `stt-serve` parity
/// bug) and made it impossible to encode two different configurations in one
/// process (blocking multi-dataset serve + parallel per-config tests). Passing an
/// `EncoderConfig` explicitly to [`encode_tile_with`] / [`encode_layer_with`]
/// removes both problems. The globals + the no-arg [`encode_tile`] /
/// [`encode_layer`] wrappers remain for the one-shot CLI and existing callers;
/// [`EncoderConfig::from_globals`] snapshots them.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Fixed-point coordinate quantization ground precision in meters
    /// (`None` = Float64 GeoArrow coordinates, the default).
    pub quantize_coords_m: Option<f64>,
    /// Per-property explicit fixed-point precisions (`name → precision`).
    pub quantize_attrs: HashMap<String, f64>,
    /// Range-adaptive `UInt16` quantization for every un-listed Float64 property.
    pub quantize_attrs_auto: bool,
    /// Scalar columns to fuse into interleaved `FixedSizeList` vector columns.
    pub vector_groups: Vec<VectorGroup>,
    /// Property folded into POINT geometry as the z coordinate (empty = none).
    pub point_elevation_column: String,
    /// Ceiling (ms) on the per-vertex time u16-delta quantization step.
    pub vertex_time_max_step_ms: u32,
    /// Layer-frame format version ([`FORMAT_VERSION_V1`] |
    /// [`FORMAT_VERSION_V2`]). The SINGLE branch point between the frozen
    /// 0.3.x v1 frame and the sectioned v2 frame (design §1 ★F3): everything
    /// upstream of frame assembly + metadata placement is shared. Defaults to
    /// v1 so every existing caller stays byte-identical; `stt-build` passes 2.
    pub format_version: u32,
    /// v2 only: when set, layer schemas are hoisted into this collector and
    /// frames carry 16-byte template-hash references (the packed-dataset
    /// mode — `PackWriter::finalize` publishes the collected templates in
    /// `manifest.schemas`). `None` under v2 emits self-contained frames with
    /// inline schema sections. Ignored under v1.
    pub template_collector: Option<Arc<TemplateCollector>>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            quantize_coords_m: None,
            quantize_attrs: HashMap::new(),
            quantize_attrs_auto: false,
            vector_groups: Vec::new(),
            point_elevation_column: String::new(),
            vertex_time_max_step_ms: DEFAULT_VERTEX_TIME_MAX_STEP_MS,
            format_version: FORMAT_VERSION_V1,
            template_collector: None,
        }
    }
}

impl EncoderConfig {
    /// Snapshot the current process-wide encoder globals into an explicit config.
    /// Used by the no-arg [`encode_tile`] / [`encode_layer`] back-compat wrappers.
    pub fn from_globals() -> Self {
        Self {
            quantize_coords_m: quantize_coords_m(),
            quantize_attrs: quantize_attrs(),
            quantize_attrs_auto: quantize_attrs_auto(),
            vector_groups: vector_groups(),
            point_elevation_column: point_elevation_column(),
            vertex_time_max_step_ms: vertex_time_max_step_ms(),
            format_version: format_version(),
            template_collector: template_collector(),
        }
    }
}

/// Fuse the scalar columns named by each configured [`VectorGroup`] into a single
/// [`PropertyColumn::Vector`], leaving every other column untouched and in order.
///
/// A group whose components are not ALL present as `Numeric` columns is skipped
/// (its scalars stay as-is) — so a tile that happens not to carry one of the
/// inputs degrades to scalars rather than dropping data. Missing per-feature
/// values (`None`) encode as `0.0`. Returns `None` when no group applied, letting
/// the caller iterate `layer.properties` directly with no clone.
fn group_vector_properties(
    props: &[(String, PropertyColumn)],
    n: usize,
    groups: &[VectorGroup],
) -> Option<Vec<(String, PropertyColumn)>> {
    if groups.is_empty() {
        return None;
    }
    // name → numeric values slice, for component lookup.
    let numeric: HashMap<&str, &[Option<f64>]> = props
        .iter()
        .filter_map(|(k, c)| match c {
            PropertyColumn::Numeric(v) => Some((k.as_str(), v.as_slice())),
            _ => None,
        })
        .collect();

    let mut out: Vec<(String, PropertyColumn)> = Vec::new();
    let mut consumed: HashSet<String> = HashSet::new();
    let mut any = false;
    for g in groups {
        if g.components.is_empty()
            || !g
                .components
                .iter()
                .all(|c| numeric.contains_key(c.as_str()))
        {
            continue;
        }
        let width = g.components.len();
        let mut values = vec![0f32; width * n];
        for (ci, cname) in g.components.iter().enumerate() {
            let col = numeric[cname.as_str()];
            for i in 0..n {
                values[i * width + ci] = col[i].map(|x| x as f32).unwrap_or(0.0);
            }
        }
        for c in &g.components {
            consumed.insert(c.clone());
        }
        out.push((
            g.name.clone(),
            PropertyColumn::Vector {
                width,
                elem: g.elem,
                values,
            },
        ));
        any = true;
    }
    if !any {
        return None;
    }
    // Keep every column not pulled into a group, in original order.
    for (k, c) in props {
        if !consumed.contains(k) {
            out.push((k.clone(), c.clone()));
        }
    }
    Some(out)
}

/// Range-adaptive auto quantization: size the step from the column's own span so
/// `[min, max]` maps onto `[0, 65535]`. ALWAYS returns a `UInt16` column (never
/// `None`, never `Int32`) so a column is quantized to the *same type in every
/// tile* — a per-tile range-adaptive choice would otherwise leave constant /
/// all-null tiles as `Float64` and drift the layer schema across tiles. A
/// constant column quantizes to all-zeros and an all-null column to all-nulls;
/// both compress to nothing, so the uniform `UInt16` costs nothing and keeps the
/// schema consistent.
fn build_quantized_numeric_auto(values: &[Option<f64>]) -> Option<(ArrayRef, String)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in values.iter().flatten() {
        if v.is_finite() {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    // o = offset (column min, or 0 when no finite value); s = step. A zero span
    // (constant / no-data) uses step 1 → every present value maps to index 0,
    // which reconstructs to `o` exactly.
    let (o, s) = if min.is_finite() {
        if max > min {
            (min, (max - min) / u16::MAX as f64)
        } else {
            (min, 1.0)
        }
    } else {
        (0.0, 1.0)
    };
    let affine = AttrQuant { o, s };
    let mut b = UInt16Builder::with_capacity(values.len());
    for v in values {
        match v {
            Some(x) if x.is_finite() => {
                let q = (((*x - o) / s).round()).clamp(0.0, u16::MAX as f64) as u16;
                b.append_value(q);
            }
            _ => b.append_null(),
        }
    }
    Some((Arc::new(b.finish()), affine.to_json()))
}

/// Quantize a numeric property column to the smallest integer leaf at `prec`
/// units, with the offset pinned to the column minimum. Returns
/// `(array, affine_json)` — a `UInt16` leaf when the quantized range fits 16
/// bits, else `Int32` — or `None` when no finite value exists (caller keeps the
/// `Float64` column). Nulls and non-finite values become Arrow nulls. The
/// offset is the per-column minimum: identical columns quantize identically, so
/// the packed format's content-addressed dedup is preserved.
///
/// Errors when a value's quantized index exceeds `i32::MAX` — the widest leaf
/// this encoding ships. Erroring is deliberate (mirrors the dictionary-overflow
/// error in [`build_dictionary_indices`]): the previous behaviour silently
/// clamped every overflowing value to `i32::MAX`, mislabeling features without
/// a trace. A producer whose column genuinely spans more than `i32::MAX`
/// quanta should pick a coarser precision or leave the column `Float64`.
fn build_quantized_numeric(
    values: &[Option<f64>],
    prec: f64,
) -> Result<Option<(ArrayRef, String)>> {
    if !(prec > 0.0) {
        return Ok(None);
    }
    let mut min = f64::INFINITY;
    for v in values.iter().flatten() {
        if v.is_finite() && *v < min {
            min = *v;
        }
    }
    if !min.is_finite() {
        return Ok(None); // no finite values — keep Float64
    }
    let affine = AttrQuant { o: min, s: prec };
    let mut q: Vec<Option<i64>> = Vec::with_capacity(values.len());
    let mut max_q: i64 = 0;
    for v in values {
        match v {
            Some(x) if x.is_finite() => {
                let qi = (((*x - affine.o) / affine.s).round() as i64).max(0);
                if qi > i32::MAX as i64 {
                    return Err(Error::Other(format!(
                        "numeric property quantization overflows: value {x} at precision \
                         {prec} quantizes to index {qi} (offset {min}), beyond the Int32 \
                         leaf's {} ceiling; use a coarser --quantize-attr precision or \
                         leave the column Float64",
                        i32::MAX
                    )));
                }
                if qi > max_q {
                    max_q = qi;
                }
                q.push(Some(qi));
            }
            _ => q.push(None),
        }
    }
    let array: ArrayRef = if max_q <= u16::MAX as i64 {
        let mut b = UInt16Builder::with_capacity(q.len());
        for qi in &q {
            match qi {
                Some(v) => b.append_value(*v as u16),
                None => b.append_null(),
            }
        }
        Arc::new(b.finish())
    } else {
        let mut b = Int32Builder::with_capacity(q.len());
        for qi in &q {
            match qi {
                Some(v) => b.append_value(*v as i32),
                None => b.append_null(),
            }
        }
        Arc::new(b.finish())
    };
    Ok(Some((array, affine.to_json())))
}

/// Build a (key_array, value_array) pair for a Dictionary<UInt16, Utf8>
/// column. Null inputs become null keys (the corresponding string is not
/// inserted into the dictionary); strings are deduplicated in first-seen
/// order so the on-disk Arrow dictionary is stable across runs.
///
/// Errors if a column has more than `u16::MAX` distinct values, which the
/// `UInt16` key space cannot address. In practice STT categorical columns top
/// out in the low hundreds, so this only fires on pathological input (e.g. a
/// per-feature unique id mistaken for a category). Erroring is deliberate: the
/// previous behaviour silently collapsed every overflowing value to a single
/// index, mislabeling features without a trace. A producer that genuinely needs
/// >65k distinct strings should split the column or widen the key type.
fn build_dictionary_indices(values: &[Option<String>]) -> Result<(Vec<Option<u16>>, Vec<String>)> {
    let mut categories: Vec<String> = Vec::new();
    let mut lookup: HashMap<String, u16> = HashMap::new();
    let mut indices: Vec<Option<u16>> = Vec::with_capacity(values.len());
    for v in values {
        match v {
            Some(s) => {
                if let Some(&idx) = lookup.get(s) {
                    indices.push(Some(idx));
                } else if categories.len() < u16::MAX as usize {
                    let idx = categories.len() as u16;
                    categories.push(s.clone());
                    lookup.insert(s.clone(), idx);
                    indices.push(Some(idx));
                } else {
                    return Err(Error::Other(format!(
                        "categorical column has more than {} distinct values, which a \
                         Dictionary<UInt16, Utf8> key cannot address; split the column \
                         into multiple categorical fields or widen the key type",
                        u16::MAX
                    )));
                }
            }
            None => indices.push(None),
        }
    }
    Ok((indices, categories))
}

/// Built per-vertex time column, alongside the per-layer schema metadata
/// that lets the reader reconstruct absolute timestamps.
struct VertexTimeColumn {
    array: ArrayRef,
    /// `(origin_ms, step_ms)` when the column is u16-delta-encoded. `None`
    /// when the column kept its absolute `List<Int64>` shape (the exact
    /// fallback path, used for layers whose temporal span would need a step
    /// beyond the configured ceiling — see [`DEFAULT_VERTEX_TIME_MAX_STEP_MS`]).
    encoding: Option<(i64, u32)>,
}

/// Build the optional per-vertex time column.
///
/// v3 attempts to encode timestamps as `List<UInt16>` deltas relative to
/// a per-layer origin and step (`absolute = origin + delta * step`), with
/// the step bounded by `max_step_ms` (see
/// [`DEFAULT_VERTEX_TIME_MAX_STEP_MS`]). When the layer's temporal span
/// would need a coarser step than that, the column keeps the exact v2
/// `List<Int64>` shape instead — bounded quantization or no quantization,
/// never a silent precision cliff.
fn build_vertex_time_array(
    vertex_times: &Option<Vec<Vec<i64>>>,
    feature_count: usize,
    max_step_ms: u32,
) -> Option<VertexTimeColumn> {
    let vt = vertex_times.as_ref()?;

    // Discover the layer's temporal span across every (feature, vertex) pair.
    // Empty / null lists are skipped — a list-of-nulls is fine, it just won't
    // shrink the span.
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut any = false;
    for times in vt.iter().take(feature_count) {
        for &t in times {
            if t < min {
                min = t;
            }
            if t > max {
                max = t;
            }
            any = true;
        }
    }

    if any && max >= min {
        // Pick the smallest step (in ms) that keeps every (t - min) inside
        // u16::MAX. step=1 means "exact ms granularity"; larger steps trade
        // precision (bounded by step ms) for a 4x payload shrink vs i64.
        let span = (max - min) as u64;
        // Computed in u64 and compared BEFORE narrowing: a pathological span
        // whose step overflows u32 must hit the i64 path, not wrap small.
        let step = if span <= u16::MAX as u64 {
            1u64
        } else {
            ((span + u16::MAX as u64 - 1) / u16::MAX as u64).max(1)
        };
        // Step ceiling: beyond it the quantization error would exceed the
        // configured precision, so take the exact i64 path below instead.
        if step <= max_step_ms as u64 {
            let step = step as u32;
            let mut builder = ListBuilder::new(UInt16Builder::new());
            for i in 0..feature_count {
                match vt.get(i) {
                    Some(times) if !times.is_empty() => {
                        for &t in times {
                            // Saturate at u16::MAX — for a sensibly chosen
                            // step this branch can only fire on inputs that
                            // disagree with the (min,max) scan above (e.g.
                            // a `vertex_times` longer than feature_count).
                            let delta =
                                ((t - min) as u64 / step as u64).min(u16::MAX as u64) as u16;
                            builder.values().append_value(delta);
                        }
                        builder.append(true);
                    }
                    _ => builder.append(false),
                }
            }
            return Some(VertexTimeColumn {
                array: Arc::new(builder.finish()),
                encoding: Some((min, step)),
            });
        }
        // Reached the exact-Int64 fallback because the span needs a coarser
        // step than the ceiling. Warn once per process (not per tile).
        if !VERTEX_TIME_FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "vertex-time span {}ms exceeds u16-delta ceiling (step {}ms > max {}ms); \
                 falling back to exact Int64 — payload keeps full precision but is ~4x larger",
                span,
                step,
                max_step_ms
            );
        }
    }

    // Fallback: legacy absolute List<Int64> for empty-ish columns or
    // pathological steps. Identical wire shape to v2.
    let mut builder = ListBuilder::new(Int64Builder::new());
    for i in 0..feature_count {
        match vt.get(i) {
            Some(times) if !times.is_empty() => {
                for &t in times {
                    builder.values().append_value(t);
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Some(VertexTimeColumn {
        array: Arc::new(builder.finish()),
        encoding: None,
    })
}

/// Build the optional per-vertex scalar column as a nullable `List<Float32>`.
///
/// Unlike `vertex_time`, there's no delta/origin/step encoding — the values are
/// producer-defined scalars (e.g. sea-surface temperature) with a small range
/// where f32 precision is ample. A feature with no per-vertex values appends a
/// null list; `NaN` entries within a list mark individual vertices with no value.
/// Returns `None` when no feature carries any value, so the column is omitted.
fn build_vertex_value_array(
    vertex_values: &Option<Vec<Vec<f32>>>,
    feature_count: usize,
) -> Option<ArrayRef> {
    let vv = vertex_values.as_ref()?;
    let any = vv.iter().take(feature_count).any(|v| !v.is_empty());
    if !any {
        return None;
    }
    let mut builder = ListBuilder::new(Float32Builder::new());
    for i in 0..feature_count {
        match vv.get(i) {
            Some(values) if !values.is_empty() => {
                for &v in values {
                    builder.values().append_value(v);
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Some(Arc::new(builder.finish()))
}

/// Recover the per-vertex value-matrix bucket count: for the first LineString
/// feature carrying both vertices and matrix data, `num_buckets = matrix_len /
/// vertex_count` (the matrix is vertex-major, `num_vertices * num_buckets`
/// long). Returns `None` for non-line geometry or when no feature carries a
/// clean multiple — the matrix column is then present but non-animatable.
fn infer_vertex_value_buckets(matrix: &[Vec<f32>], geometry: &GeometryColumn) -> Option<u32> {
    let lines = match geometry {
        GeometryColumn::LineString(lines) => lines,
        _ => return None,
    };
    for (i, m) in matrix.iter().enumerate() {
        let nv = lines.get(i)?.len();
        if !m.is_empty() && nv > 0 && m.len() % nv == 0 {
            return Some((m.len() / nv) as u32);
        }
    }
    None
}

// ----------------------------------------------------------------------------
// Encoding
// ----------------------------------------------------------------------------

/// Encode a single layer to an Arrow IPC stream.
pub fn encode_layer(layer: &ColumnarLayer) -> Result<Vec<u8>> {
    encode_layer_cfg(layer, &EncoderConfig::from_globals())
}

/// [`encode_layer`] with optional fixed-point coordinate quantization.
///
/// `quantize_m = Some(meters)` stores coordinates as `i32` grid indices at that
/// ground precision (default-off `None` is byte-identical to [`encode_layer`]).
/// Coordinates are the dominant, near-incompressible tile column, so quantizing
/// them is the single largest size lever — at the cost of GeoArrow Float64
/// self-description, hence opt-in. The per-layer affine rides in the geometry
/// field metadata under [`STT_QUANT_META_KEY`]; the reader reconstructs Float64.
///
/// The non-coordinate settings (attribute quantization, vector grouping,
/// point-elevation fold, vertex-time precision) come from the process-wide
/// globals; use [`encode_layer_with`] to pass every setting explicitly.
pub fn encode_layer_quantized(layer: &ColumnarLayer, quantize_m: Option<f64>) -> Result<Vec<u8>> {
    encode_layer_cfg(
        layer,
        &EncoderConfig {
            quantize_coords_m: quantize_m,
            ..EncoderConfig::from_globals()
        },
    )
}

/// [`encode_layer`] with a fully-explicit [`EncoderConfig`] — no process-wide
/// globals are read. This is the concurrency- and multi-config-safe entry point
/// (e.g. a dynamic server fronting several datasets with different settings).
pub fn encode_layer_with(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<Vec<u8>> {
    encode_layer_cfg(layer, cfg)
}

/// The single-layer encode implementation, driven entirely by an explicit
/// [`EncoderConfig`]. Every public `encode_layer*` entry point funnels here.
///
/// Always emits the self-describing v1 shape — one Arrow IPC stream with ALL
/// metadata inline. The v2 tile frame reuses the identical column/field build
/// ([`build_layer_parts`]) and changes only metadata PLACEMENT at assembly
/// ([`encode_layer_v2_parts`]) — the design's single v1/v2 branch point.
fn encode_layer_cfg(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<Vec<u8>> {
    let parts = build_layer_parts(layer, cfg)?;
    assemble_layer_ipc_v1(parts)
}

/// Everything a frame assembler needs about one built layer: the Arrow fields
/// + columns in canonical order (reserved columns first, then properties) and
/// the per-tile schema-metadata values. Both frame versions build this
/// identically; only where the metadata LANDS differs (v1: schema metadata;
/// v2: `TILE_META` section + stripped, dataset-constant templates).
struct LayerParts {
    fields: Vec<Arc<Field>>,
    columns: Vec<ArrayRef>,
    /// `fields[..reserved_len]` are the reserved (CORE) columns; the rest are
    /// property (PROPS) columns.
    reserved_len: usize,
    layer_name: String,
    geometry_name: &'static str,
    /// Minimum feature start-time (v1 `stt:time_offset_ms` / v2 `t0`).
    min_start_time: Option<i64>,
    /// `(origin_ms, step_ms)` of a u16-delta `vertex_time` column.
    vertex_time_encoding: Option<(i64, u32)>,
    /// v1 `stt:vertex_value_buckets` / v2 `vb`.
    vertex_value_buckets: Option<u32>,
    has_triangles: bool,
}

/// Build the Arrow fields + columns for one layer — the version-independent
/// front of the encoder (validation, quantization, vector grouping, the
/// point-elevation fold, vertex-time encoding).
fn build_layer_parts(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<LayerParts> {
    layer.validate()?;
    let n = layer.feature_count();

    let mut fields: Vec<Arc<Field>> = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    fields.push(Arc::new(Field::new("id", DataType::UInt64, false)));
    columns.push(Arc::new(UInt64Array::from(layer.feature_ids.clone())));

    fields.push(Arc::new(Field::new("start_time", DataType::Int64, false)));
    columns.push(Arc::new(Int64Array::from(layer.start_times.clone())));

    fields.push(Arc::new(Field::new("end_time", DataType::Int64, false)));
    columns.push(Arc::new(Int64Array::from(layer.end_times.clone())));

    // 3D POINT geometry: fold the configured numeric column into the geometry's
    // 3rd coordinate, so the tile ships true 3D points the renderer binds
    // zero-copy (no per-point pad). The column is then dropped from properties.
    let elev_col = cfg.point_elevation_column.clone();
    let point_elev: Option<Vec<f64>> =
        if !elev_col.is_empty() && matches!(layer.geometry, GeometryColumn::Point(_)) {
            layer.properties.iter().find_map(|(name, col)| match col {
                PropertyColumn::Numeric(v) if name == &elev_col => {
                    let n = layer.feature_count();
                    let mut out = vec![0.0f64; n];
                    for (i, x) in v.iter().enumerate() {
                        if let Some(val) = x {
                            out[i] = *val;
                        }
                    }
                    Some(out)
                }
                _ => None,
            })
        } else {
            None
        };
    let elev_consumed = point_elev.is_some();

    // Geometry column carries the GeoArrow extension name in field metadata.
    // The precision floor is enforced HERE (where the meters value is
    // consumed), so every entry point — globals, explicit EncoderConfig, a
    // server's per-request config — hits the same guard.
    if let Some(m) = cfg.quantize_coords_m {
        validate_quantize_coords_m(m)?;
    }
    let quant = cfg.quantize_coords_m.and_then(|m| {
        if point_elev.is_some() {
            world_grid_affine_3d(m)
        } else {
            world_grid_affine(m)
        }
    });
    let geom_array =
        build_geometry_array_q(&layer.geometry, quant.as_ref(), point_elev.as_deref())?;
    // Assemble field metadata in a BTreeMap so the key set is emitted in a
    // deterministic (lexicographic) order regardless of insertion order. Arrow
    // ≥59 serializes IPC schema metadata in sorted order, so building from a
    // sorted source makes the raw metadata-region bytes byte-reproducible across
    // runs (guarded by `same_tile_encodes_byte_identically` in
    // reproducible_build.rs) — closing the old arrow-54 HashMap-iteration gap.
    let mut geom_meta = BTreeMap::new();
    geom_meta.insert(
        GEOARROW_EXT_KEY.to_string(),
        layer.geometry.geoarrow_name().to_string(),
    );
    match &quant {
        // A quantized tile's `xy` leaf is i32 grid indices, not Float64 lon/lat,
        // so the GeoArrow CRS doesn't apply — swap it for the reconstruction
        // affine (whose presence is the reader's quantization signal). Field
        // metadata is assembled in a BTreeMap (deterministic key order); Arrow
        // ≥59 serializes IPC schema metadata in sorted order, so the raw
        // metadata-region bytes are byte-reproducible across runs (guarded by
        // the `same_tile_encodes_byte_identically` test in reproducible_build.rs).
        Some(q) => {
            geom_meta.insert(STT_QUANT_META_KEY.to_string(), q.to_json());
        }
        // Advertise the CRS so the tile is self-describing to GeoArrow consumers.
        None => {
            geom_meta.insert(
                GEOARROW_EXT_META_KEY.to_string(),
                GEOARROW_CRS_METADATA.to_string(),
            );
        }
    }
    fields.push(Arc::new(
        Field::new("geometry", geom_array.data_type().clone(), false)
            .with_metadata(geom_meta.into_iter().collect()),
    ));
    columns.push(geom_array);

    // Track per-layer vertex-time encoding so the schema metadata (set
    // below) records the origin/step needed for the u16-delta reader path.
    let mut vertex_time_encoding: Option<(i64, u32)> = None;
    if let Some(vt_col) =
        build_vertex_time_array(&layer.vertex_times, n, cfg.vertex_time_max_step_ms)
    {
        fields.push(Arc::new(Field::new(
            "vertex_time",
            vt_col.array.data_type().clone(),
            true,
        )));
        columns.push(vt_col.array);
        vertex_time_encoding = vt_col.encoding;
    }

    // Optional per-vertex scalar column (e.g. sea-surface temperature),
    // aligned 1:1 with the geometry vertices like `vertex_time`.
    if let Some(vv_array) = build_vertex_value_array(&layer.vertex_values, n) {
        fields.push(Arc::new(Field::new(
            "vertex_value",
            vv_array.data_type().clone(),
            true,
        )));
        columns.push(vv_array);
    }

    // Optional per-vertex × per-bucket value matrix (static-geometry overview
    // animation). Reuses the `vertex_value` List<Float32> encoding — each row
    // is just longer (vertex_count * num_buckets, vertex-major). num_buckets is
    // recovered from the per-feature vertex count and recorded in schema meta.
    let mut vertex_value_buckets: Option<u32> = None;
    if let Some(vm_array) = build_vertex_value_array(&layer.vertex_value_matrix, n) {
        fields.push(Arc::new(Field::new(
            "vertex_value_matrix",
            vm_array.data_type().clone(),
            true,
        )));
        columns.push(vm_array);
        vertex_value_buckets = layer
            .vertex_value_matrix
            .as_ref()
            .and_then(|vm| infer_vertex_value_buckets(vm, &layer.geometry));
    }

    // Pre-baked triangle indices (MLT-style). Only emitted for polygon
    // layers; for any other geometry kind the column is silently dropped so
    // an over-eager builder can't poison a point/line layer with stale data.
    let has_triangles = matches!(layer.geometry, GeometryColumn::Polygon(_))
        && layer
            .triangles
            .as_ref()
            .map(|t| t.iter().any(|f| !f.is_empty()))
            .unwrap_or(false);
    if has_triangles {
        let tri = layer.triangles.as_ref().unwrap();
        // Indices are feature-LOCAL (see the field doc above), so they're
        // almost always well under 65,536 even for large layers. Mirrors
        // build_vertex_time_array's width-selection: scan once, use the
        // narrower UInt16 (half the bytes) when every index fits, UInt32
        // otherwise. The Arrow field type is derived from the array below,
        // so this is fully self-describing — the TS decoder branches on the
        // runtime child-array type exactly like it already does for
        // vertex_time's UInt16-delta vs Int64-absolute split.
        let max_index = tri.iter().flatten().copied().max().unwrap_or(0);
        let array: ArrayRef = if max_index <= u16::MAX as u32 {
            let mut builder = ListBuilder::new(UInt16Builder::new());
            for feature in tri {
                for &idx in feature {
                    builder.values().append_value(idx as u16);
                }
                // Always append a (possibly empty) list — readers expect one
                // entry per feature.
                builder.append(true);
            }
            Arc::new(builder.finish())
        } else {
            let mut builder = ListBuilder::new(UInt32Builder::new());
            for feature in tri {
                for &idx in feature {
                    builder.values().append_value(idx);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        };
        fields.push(Arc::new(Field::new(
            "triangles",
            array.data_type().clone(),
            false,
        )));
        columns.push(array);
    }

    // Every field pushed so far is a reserved column — the v2 CORE batch.
    // Property columns (the v2 PROPS batch) follow.
    let reserved_len = fields.len();

    // Fuse configured scalar columns into GPU-ready interleaved Vector columns
    // (e.g. qx/qy/qz/qw → one FixedSizeList<f32,4>). Runs BEFORE the quantize
    // loop so grouped components are written as the raw vector, not individually
    // quantized. No groups configured ⇒ iterate `layer.properties` with no clone.
    let grouped =
        group_vector_properties(&layer.properties, layer.feature_count(), &cfg.vector_groups);
    let props_iter: &[(String, PropertyColumn)] = grouped.as_deref().unwrap_or(&layer.properties);
    for (name, col) in props_iter {
        // The point-elevation column now lives in the geometry's 3rd coordinate;
        // don't also emit it as a scalar property.
        if elev_consumed && name == &elev_col {
            continue;
        }
        match col {
            PropertyColumn::Numeric(values) => {
                // Opt-in: a numeric property named in the build-global
                // quantization map ships as fixed-point ints + a per-column
                // affine in field metadata (the reader reconstructs Float64).
                // For a LiDAR `z` column this is the single largest size lever
                // after `id` — a raw Float64 elevation barely compresses, while
                // the i16 grid is both smaller and far more compressible.
                let quantized = match cfg.quantize_attrs.get(name).copied().filter(|p| *p > 0.0) {
                    Some(p) => build_quantized_numeric(values, p)?,
                    None => None,
                }
                .or_else(|| {
                    // No explicit precision — fall back to the configured
                    // automatic range-adaptive quantization when enabled.
                    cfg.quantize_attrs_auto
                        .then(|| build_quantized_numeric_auto(values))
                        .flatten()
                });
                match quantized {
                    Some((array, affine_json)) => {
                        let mut m = HashMap::new();
                        m.insert(STT_QUANT_ATTR_META_KEY.to_string(), affine_json);
                        fields.push(Arc::new(
                            Field::new(name, array.data_type().clone(), true).with_metadata(m),
                        ));
                        columns.push(array);
                    }
                    None => {
                        fields.push(Arc::new(Field::new(name, DataType::Float64, true)));
                        columns.push(Arc::new(Float64Array::from(values.clone())));
                    }
                }
            }
            PropertyColumn::Categorical(values) => {
                // Build a Dictionary<UInt16, Utf8>: deduplicate strings once
                // here so the TS reader can lift the dictionary table out of
                // the Arrow batch directly instead of rebuilding it per tile.
                let (indices, categories) = build_dictionary_indices(values)?;
                let key_type = DataType::UInt16;
                let value_type = DataType::Utf8;
                let dict_type = DataType::Dictionary(Box::new(key_type), Box::new(value_type));
                fields.push(Arc::new(Field::new(name, dict_type, true)));

                let value_array: ArrayRef = Arc::new(StringArray::from(
                    categories
                        .iter()
                        .map(|s| Some(s.as_str()))
                        .collect::<Vec<_>>(),
                ));
                let key_array = UInt16Array::from(indices);
                let dict = DictionaryArray::<UInt16Type>::try_new(key_array, value_array)
                    .map_err(|e| Error::Other(format!("dictionary build failed: {e}")))?;
                columns.push(Arc::new(dict));
            }
            PropertyColumn::Vector {
                width,
                elem,
                values,
            } => {
                // Interleaved GPU-ready vector → FixedSizeList<leaf, width>. The
                // child buffer is the flattened row-major run, so the TS decoder
                // hands `child.values.subarray(...)` straight to deck.gl with no
                // per-point re-interleave. Non-null leaf (producer encodes a
                // missing feature as a zero/identity vector).
                let (child, child_dt): (ArrayRef, DataType) = match elem {
                    VectorElem::F32 => (
                        Arc::new(Float32Array::from(values.clone())),
                        DataType::Float32,
                    ),
                    VectorElem::U8 => {
                        let bytes: Vec<u8> = values
                            .iter()
                            .map(|v| v.round().clamp(0.0, 255.0) as u8)
                            .collect();
                        (Arc::new(UInt8Array::from(bytes)), DataType::UInt8)
                    }
                };
                let item_field = Arc::new(Field::new("item", child_dt, false));
                let width_i32 = i32::try_from(*width).map_err(|_| {
                    Error::Other(format!(
                        "vector property '{name}' has width {width}, which exceeds the \
                         FixedSizeList i32 size limit"
                    ))
                })?;
                let fsl = FixedSizeListArray::new(item_field, width_i32, child, None);
                fields.push(Arc::new(Field::new(name, fsl.data_type().clone(), true)));
                columns.push(Arc::new(fsl));
            }
        }
    }

    Ok(LayerParts {
        fields,
        columns,
        reserved_len,
        layer_name: layer.name.clone(),
        geometry_name: layer.geometry.geoarrow_name(),
        // The layer's minimum feature start-time (integer Unix ms), so the TS
        // decoder can skip its client-side min-scan over the whole start-time
        // column and relativize times against this value directly. Mirrors
        // exactly what the decoder computes (the min of the `start_time`
        // column); only emitted when a start-time column is present. See
        // packages/core/src/tile.ts.
        min_start_time: layer.start_times.iter().copied().min(),
        vertex_time_encoding,
        vertex_value_buckets,
        has_triangles,
    })
}

/// Serialize one `(schema, columns)` batch as an Arrow IPC stream.
fn write_ipc_stream(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<Vec<u8>> {
    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Error::Other(format!("failed to build tile RecordBatch: {e}")))?;
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)
            .map_err(|e| Error::Other(format!("Arrow IPC writer init failed: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| Error::Other(format!("Arrow IPC write failed: {e}")))?;
        writer
            .finish()
            .map_err(|e| Error::Other(format!("Arrow IPC finish failed: {e}")))?;
    }
    Ok(buf)
}

/// v1 assembly: one schema carrying EVERY metadata key (dataset-constant AND
/// per-tile-varying), serialized as a single IPC stream — the frozen 0.3.x
/// bytes, guarded by the `tests/v1_golden.rs` fixture.
fn assemble_layer_ipc_v1(parts: LayerParts) -> Result<Vec<u8>> {
    // Schema-level metadata records the layer name and geometry kind so a
    // reader does not have to inspect the geometry column. When the
    // vertex_time column is u16-delta encoded we add `origin_ms` and
    // `step_ms` so the reader can reconstruct absolute timestamps as
    // `origin + delta * step`.
    //
    // Built in a BTreeMap so the key set is assembled in deterministic
    // (lexicographic) order — this encoder contributes no ordering
    // non-determinism. Arrow ≥59 serializes IPC schema metadata in sorted order,
    // so the raw metadata-region bytes of two identical tiles are now identical
    // across runs (unblocking content-addressed pack dedup) — closing the old
    // arrow-54 HashMap-iteration gap.
    let mut schema_meta: BTreeMap<String, String> = BTreeMap::new();
    schema_meta.insert("stt:layer".to_string(), parts.layer_name.clone());
    schema_meta.insert("stt:geometry".to_string(), parts.geometry_name.to_string());
    if let Some(min_start) = parts.min_start_time {
        schema_meta.insert(TIME_OFFSET_MS_KEY.to_string(), min_start.to_string());
    }
    if let Some((origin, step)) = parts.vertex_time_encoding {
        schema_meta.insert(VERTEX_TIME_ORIGIN_KEY.to_string(), origin.to_string());
        schema_meta.insert(VERTEX_TIME_STEP_KEY.to_string(), step.to_string());
    }
    if let Some(buckets) = parts.vertex_value_buckets {
        schema_meta.insert(VERTEX_VALUE_BUCKETS_KEY.to_string(), buckets.to_string());
    }
    if parts.has_triangles {
        schema_meta.insert(TRIANGLES_METADATA_KEY.to_string(), "true".to_string());
    }
    let schema =
        Arc::new(Schema::new(parts.fields).with_metadata(schema_meta.into_iter().collect()));
    write_ipc_stream(schema, parts.columns)
}

// ----------------------------------------------------------------------------
// v2 layer encoding (template extraction + TILE_META + core/props split)
// ----------------------------------------------------------------------------

/// One layer, encoded for the v2 frame: templates split off, tails verbatim,
/// per-tile metadata canonicalized into the TILE_META JSON.
struct EncodedLayerV2 {
    core_template: Vec<u8>,
    core_tail: Vec<u8>,
    /// `(template, tail)` of the PROPS batch; `None` when the layer has no
    /// property columns (`ref_kind_props = 2`).
    props: Option<(Vec<u8>, Vec<u8>)>,
    tile_meta_json: String,
}

/// Locate the end of the leading schema message by walking the Arrow IPC
/// encapsulated framing: `[0xFFFFFFFF][i32 metadata_len][flatbuffer (padded)]`
/// with the schema's `bodyLength == 0` (spike-proven boundary — deterministic,
/// no re-serialization; `metadata_len` already includes the flatbuffer's
/// padding). Everything before the boundary is the template, everything after
/// is the tail (dictionary batches + record batch + EOS).
fn split_ipc_at_schema(ipc: &[u8]) -> Result<usize> {
    if ipc.len() < 8 || ipc[0..4] != [0xFF, 0xFF, 0xFF, 0xFF] {
        return Err(Error::Other(
            "layer IPC stream does not start with an encapsulated message".into(),
        ));
    }
    let meta_len = i32::from_le_bytes(ipc[4..8].try_into().expect("4 bytes"));
    if meta_len <= 0 {
        return Err(Error::Other(
            "layer IPC stream starts with an end-of-stream marker".into(),
        ));
    }
    let boundary = 8usize
        .checked_add(meta_len as usize)
        .filter(|b| *b <= ipc.len())
        .ok_or_else(|| Error::Other("layer IPC schema message overruns the stream".into()))?;
    let msg = root_as_message(&ipc[8..boundary])
        .map_err(|e| Error::Other(format!("layer IPC schema flatbuffer parse failed: {e}")))?;
    if msg.header_type() != MessageHeader::Schema {
        return Err(Error::Other(format!(
            "layer IPC stream must start with a Schema message, got {:?}",
            msg.header_type()
        )));
    }
    if msg.bodyLength() != 0 {
        return Err(Error::Other(
            "layer IPC schema message unexpectedly carries a body".into(),
        ));
    }
    Ok(boundary)
}

/// v2 row order (design §4.2, ★F10): stable-sort rows by `start_time` at
/// ENCODE time — after the tiler assigned feature ids — so ids stay
/// order-independent. Returns the layer unchanged (borrowed) when its rows
/// are already non-decreasing, which also makes the sort a no-op for
/// pre-sorted producers.
fn sort_rows_by_start_time(layer: &ColumnarLayer) -> Cow<'_, ColumnarLayer> {
    if layer.start_times.windows(2).all(|w| w[0] <= w[1]) {
        return Cow::Borrowed(layer);
    }
    let mut idx: Vec<usize> = (0..layer.feature_count()).collect();
    idx.sort_by_key(|&i| layer.start_times[i]); // stable

    // Per-feature Option<Vec<Vec<_>>> columns tolerate inner vecs shorter
    // than the feature count at encode (`vt.get(i)` ⇒ null list); the
    // permutation preserves that semantic via `.get(..).unwrap_or_default()`.
    fn permute_nested<T: Clone>(v: &Option<Vec<Vec<T>>>, idx: &[usize]) -> Option<Vec<Vec<T>>> {
        v.as_ref().map(|v| {
            idx.iter()
                .map(|&i| v.get(i).cloned().unwrap_or_default())
                .collect()
        })
    }

    let geometry = match &layer.geometry {
        GeometryColumn::Point(v) => GeometryColumn::Point(idx.iter().map(|&i| v[i]).collect()),
        GeometryColumn::LineString(v) => {
            GeometryColumn::LineString(idx.iter().map(|&i| v[i].clone()).collect())
        }
        GeometryColumn::Polygon(v) => {
            GeometryColumn::Polygon(idx.iter().map(|&i| v[i].clone()).collect())
        }
    };
    let properties = layer
        .properties
        .iter()
        .map(|(name, col)| {
            let col = match col {
                PropertyColumn::Numeric(v) => {
                    PropertyColumn::Numeric(idx.iter().map(|&i| v[i]).collect())
                }
                PropertyColumn::Categorical(v) => {
                    PropertyColumn::Categorical(idx.iter().map(|&i| v[i].clone()).collect())
                }
                PropertyColumn::Vector {
                    width,
                    elem,
                    values,
                } => PropertyColumn::Vector {
                    width: *width,
                    elem: *elem,
                    values: idx
                        .iter()
                        .flat_map(|&i| values[i * width..(i + 1) * width].iter().copied())
                        .collect(),
                },
            };
            (name.clone(), col)
        })
        .collect();

    Cow::Owned(ColumnarLayer {
        name: layer.name.clone(),
        feature_ids: idx.iter().map(|&i| layer.feature_ids[i]).collect(),
        start_times: idx.iter().map(|&i| layer.start_times[i]).collect(),
        end_times: idx.iter().map(|&i| layer.end_times[i]).collect(),
        geometry,
        vertex_times: permute_nested(&layer.vertex_times, &idx),
        vertex_values: permute_nested(&layer.vertex_values, &idx),
        vertex_value_matrix: permute_nested(&layer.vertex_value_matrix, &idx),
        triangles: permute_nested(&layer.triangles, &idx),
        properties,
    })
}

/// Encode one layer for the v2 frame: sort rows, build the shared parts, move
/// per-tile-varying metadata into TILE_META, split the CORE and PROPS IPC
/// streams at their schema boundaries.
///
/// Metadata placement (design §3.1 audit): per-tile-varying and hoisted into
/// TILE_META are exactly `stt:qa` (property fields) and the schema-level
/// `stt:time_offset_ms` / `stt:vertex_time_origin_ms` / `stt:vertex_time_step_ms`
/// / `stt:vertex_value_buckets`. Dataset-constant and template-resident:
/// `ARROW:extension:name` / `ARROW:extension:metadata` (CRS), `stt:quant`
/// (world-anchored), `stt:layer`, `stt:geometry`, `stt:has_triangles`.
fn encode_layer_v2_parts(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<EncodedLayerV2> {
    // Validate BEFORE the row sort: `sort_rows_by_start_time` indexes every
    // column by the feature count, so a length-inconsistent layer would panic
    // there instead of returning the v1 path's descriptive error.
    // (`build_layer_parts` re-validates the sorted layer; the check is pure.)
    layer.validate()?;
    let sorted = sort_rows_by_start_time(layer);
    let parts = build_layer_parts(&sorted, cfg)?;

    // Strip `stt:qa` off the property fields into the TILE_META `qa` map —
    // spike-proven to leave the batch tail bytes byte-identical (only the
    // schema message changes). The affine JSON round-trips exactly
    // (`AttrQuant::to_json` is a pure function of `(o, s)`), so decode
    // re-injects byte-identical field metadata.
    let mut qa: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    let mut props_fields: Vec<Arc<Field>> =
        Vec::with_capacity(parts.fields.len() - parts.reserved_len);
    for field in &parts.fields[parts.reserved_len..] {
        let mut meta = field.metadata().clone();
        if let Some(json) = meta.remove(STT_QUANT_ATTR_META_KEY) {
            let affine = AttrQuant::from_json(&json).ok_or_else(|| {
                Error::Other(format!(
                    "property '{}' carries an unparseable {STT_QUANT_ATTR_META_KEY} affine",
                    field.name()
                ))
            })?;
            qa.insert(field.name().clone(), (affine.o, affine.s));
            props_fields.push(Arc::new(field.as_ref().clone().with_metadata(meta)));
        } else {
            props_fields.push(field.clone());
        }
    }

    let tile_meta = TileMeta {
        qa: (!qa.is_empty()).then_some(qa),
        sorted: Some(true),
        t0: parts.min_start_time,
        vb: parts.vertex_value_buckets,
        vt: parts.vertex_time_encoding,
    };
    let tile_meta_json = serde_json::to_string(&tile_meta)
        .map_err(|e| Error::Other(format!("TILE_META encode failed: {e}")))?;

    // CORE schema: dataset-constant metadata ONLY (the per-tile keys live in
    // TILE_META), so the template bytes are identical across every tile of
    // the layer's shape — the whole point of the hoist.
    let mut core_meta: BTreeMap<String, String> = BTreeMap::new();
    core_meta.insert("stt:layer".to_string(), parts.layer_name.clone());
    core_meta.insert("stt:geometry".to_string(), parts.geometry_name.to_string());
    if parts.has_triangles {
        core_meta.insert(TRIANGLES_METADATA_KEY.to_string(), "true".to_string());
    }
    let core_fields: Vec<Arc<Field>> = parts.fields[..parts.reserved_len].to_vec();
    let core_columns: Vec<ArrayRef> = parts.columns[..parts.reserved_len].to_vec();
    let core_schema =
        Arc::new(Schema::new(core_fields).with_metadata(core_meta.into_iter().collect()));
    let core_ipc = write_ipc_stream(core_schema, core_columns)?;
    let core_boundary = split_ipc_at_schema(&core_ipc)?;
    let core_tail = core_ipc[core_boundary..].to_vec();
    let mut core_template = core_ipc;
    core_template.truncate(core_boundary);

    // PROPS schema: property fields only, no schema-level metadata (every
    // dataset-constant key lives on the CORE template).
    let props = if props_fields.is_empty() {
        None
    } else {
        let props_columns: Vec<ArrayRef> = parts.columns[parts.reserved_len..].to_vec();
        let props_schema = Arc::new(Schema::new(props_fields));
        let props_ipc = write_ipc_stream(props_schema, props_columns)?;
        let boundary = split_ipc_at_schema(&props_ipc)?;
        let tail = props_ipc[boundary..].to_vec();
        let mut template = props_ipc;
        template.truncate(boundary);
        Some((template, tail))
    };

    Ok(EncodedLayerV2 {
        core_template,
        core_tail,
        props,
        tile_meta_json,
    })
}

/// Encode a full tile payload (one or more layers) with the layer frame.
///
/// Always emits the *aligned* frame ([`ALIGNED_FRAME_FLAG`] set): each
/// layer's IPC stream is preceded by zero padding to an 8-byte boundary
/// relative to the payload start, so readers can wrap the stream zero-copy.
/// `ipc_len` records the exact IPC byte length (padding excluded); readers
/// derive the pad from alignment math alone.
pub fn encode_tile(layers: &[ColumnarLayer]) -> Result<Vec<u8>> {
    encode_tile_cfg(layers, &EncoderConfig::from_globals())
}

/// [`encode_tile`] with optional fixed-point coordinate quantization applied to
/// every layer (see [`encode_layer_quantized`]). `quantize_m = None` is
/// byte-identical to [`encode_tile`]; the other encoder settings come from the
/// process-wide globals.
pub fn encode_tile_quantized(layers: &[ColumnarLayer], quantize_m: Option<f64>) -> Result<Vec<u8>> {
    encode_tile_cfg(
        layers,
        &EncoderConfig {
            quantize_coords_m: quantize_m,
            ..EncoderConfig::from_globals()
        },
    )
}

/// [`encode_tile`] with a fully-explicit [`EncoderConfig`] — no process-wide
/// globals are read. The concurrency- and multi-config-safe entry point a
/// dynamic per-request tile server uses so each dataset/request encodes with its
/// own settings without touching shared state.
pub fn encode_tile_with(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    encode_tile_cfg(layers, cfg)
}

/// The single tile-encode implementation, driven entirely by an explicit
/// [`EncoderConfig`]. Every public `encode_tile*` entry point funnels here,
/// then branches ONCE on [`EncoderConfig::format_version`] (design §1 ★F3):
/// everything upstream (column/field build) is shared.
fn encode_tile_cfg(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    match cfg.format_version {
        FORMAT_VERSION_V1 => encode_tile_frame_v1(layers, cfg),
        FORMAT_VERSION_V2 => encode_tile_frame_v2(layers, cfg),
        other => Err(Error::Other(format!(
            "unsupported format version {other} (this writer emits 1 or 2)"
        ))),
    }
}

/// v1 frame layer-count guard: the aligned frame's leading u16 spends bit 15
/// on [`ALIGNED_FRAME_FLAG`], leaving 15 bits for the count — so 0x7FFE is the
/// maximum representable count. 0x7FFF is additionally reserved (its OR with
/// the flag collides with [`FRAME_V2_ESCAPE`]), and any count with bit 15 set
/// (0x8000..) would OR into the flag as a silently-mangled smaller count.
/// `>= 0x7FFF` rejects the collision AND the whole unrepresentable range.
fn v1_layer_count_ok(count: usize) -> bool {
    count < 0x7FFF
}

/// v1 frame assembly — the frozen 0.3.x bytes (`tests/v1_golden.rs`).
fn encode_tile_frame_v1(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    // Cap the count one below the historical 0x7fff limit (see
    // [`v1_layer_count_ok`]). No real tile comes within orders of magnitude
    // of the limit, so only the error bound moves.
    if !v1_layer_count_ok(layers.len()) {
        return Err(Error::Other(format!(
            "tile has {} layers, exceeds the {} frame limit",
            layers.len(),
            ALIGNED_FRAME_FLAG - 2
        )));
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(layers.len() as u16 | ALIGNED_FRAME_FLAG).to_le_bytes());
    for layer in layers {
        let name = layer.name.as_bytes();
        if name.len() > u16::MAX as usize {
            return Err(Error::Other("layer name too long".into()));
        }
        let ipc = encode_layer_cfg(layer, cfg)?;
        let ipc_len = u32::try_from(ipc.len()).map_err(|_| {
            Error::Other(format!(
                "layer '{}' IPC stream is {} bytes, exceeding the frame's u32 length field",
                layer.name,
                ipc.len()
            ))
        })?;
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&ipc_len.to_le_bytes());
        let pad = (FRAME_ALIGN - out.len() % FRAME_ALIGN) % FRAME_ALIGN;
        out.extend_from_slice(&[0u8; FRAME_ALIGN][..pad]);
        out.extend_from_slice(&ipc);
    }
    Ok(out)
}

/// Zero-pad `out` to the next [`FRAME_ALIGN`] boundary (v2 derived padding —
/// like v1, the pad length is never stored).
fn pad_to_frame_align(out: &mut Vec<u8>) {
    let pad = (FRAME_ALIGN - out.len() % FRAME_ALIGN) % FRAME_ALIGN;
    out.extend_from_slice(&[0u8; FRAME_ALIGN][..pad]);
}

/// v2 frame assembly (module docs / design §4). With a
/// [`TemplateCollector`] configured, schemas are recorded there and frames
/// carry 16-byte hash references (the packed-dataset mode); without one the
/// frame is self-contained via `INLINE_SCHEMA_*` sections.
fn encode_tile_frame_v2(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    if layers.len() > u16::MAX as usize {
        return Err(Error::Other(format!(
            "tile has {} layers, exceeds the {} frame limit",
            layers.len(),
            u16::MAX
        )));
    }
    let collector = cfg.template_collector.as_deref();
    let mut out = Vec::new();
    out.extend_from_slice(&FRAME_V2_ESCAPE.to_le_bytes());
    out.push(FRAME_V2_VERSION);
    out.push(0); // flags — reserved, MUST be 0
    out.extend_from_slice(&(layers.len() as u16).to_le_bytes());
    for layer in layers {
        let name = layer.name.as_bytes();
        if name.len() > u16::MAX as usize {
            return Err(Error::Other("layer name too long".into()));
        }
        let enc = encode_layer_v2_parts(layer, cfg)?;
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);

        // Schema references. Hash mode records the template with the
        // collector (content-addressed, so parallel encode order is
        // irrelevant); inline mode ships it as a section instead.
        match collector {
            Some(c) => {
                out.push(REF_KIND_TEMPLATE_HASH);
                out.extend_from_slice(&c.record(&enc.core_template));
            }
            None => out.push(REF_KIND_INLINE),
        }
        match (&enc.props, collector) {
            (None, _) => out.push(REF_KIND_NO_PROPS),
            (Some((template, _)), Some(c)) => {
                out.push(REF_KIND_TEMPLATE_HASH);
                out.extend_from_slice(&c.record(template));
            }
            (Some(_), None) => out.push(REF_KIND_INLINE),
        }

        // Sections in ascending tag order; TOC lengths are the exact at-rest
        // byte counts (padding derived, never stored).
        let mut sections: Vec<(u8, &[u8])> = Vec::new();
        if collector.is_none() {
            sections.push((SECTION_INLINE_SCHEMA_CORE, &enc.core_template));
        }
        sections.push((SECTION_TILE_META, enc.tile_meta_json.as_bytes()));
        sections.push((SECTION_CORE_BATCH, &enc.core_tail));
        if let Some((template, tail)) = &enc.props {
            if collector.is_none() {
                sections.push((SECTION_INLINE_SCHEMA_PROPS, template));
            }
            sections.push((SECTION_PROPS_BATCH, tail));
        }
        out.push(sections.len() as u8);
        for (tag, bytes) in &sections {
            out.push(*tag);
            let len = u32::try_from(bytes.len()).map_err(|_| {
                Error::Other(format!(
                    "layer '{}' section 0x{tag:02x} is {} bytes, exceeding the TOC's u32 \
                     length field",
                    layer.name,
                    bytes.len()
                ))
            })?;
            out.extend_from_slice(&len.to_le_bytes());
        }
        pad_to_frame_align(&mut out);
        for (_, bytes) in &sections {
            out.extend_from_slice(bytes);
            pad_to_frame_align(&mut out);
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Decoding
// ----------------------------------------------------------------------------

/// A decoded tile layer: its name and the raw Arrow [`RecordBatch`].
#[derive(Debug, Clone)]
pub struct DecodedLayer {
    /// Layer name from the layer frame.
    pub name: String,
    /// The decoded Arrow record batch.
    pub batch: RecordBatch,
}

/// Decode a single-layer Arrow IPC stream into a [`RecordBatch`].
pub fn decode_layer(ipc: &[u8]) -> Result<RecordBatch> {
    let reader = StreamReader::try_new(ipc, None)
        .map_err(|e| Error::Other(format!("Arrow IPC reader init failed: {e}")))?;
    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|e| Error::Other(format!("Arrow IPC read failed: {e}")))?);
    }
    match batches.len() {
        0 => Err(Error::Other(
            "tile layer IPC contained no record batch".into(),
        )),
        1 => Ok(batches.into_iter().next().unwrap()),
        // A layer is written as exactly one batch; concatenating is the safe
        // fallback if a producer ever splits it.
        _ => arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .map_err(|e| Error::Other(format!("failed to concat tile batches: {e}"))),
    }
}

/// Whether a tile payload carries the v2 sectioned frame (leading u16 is the
/// [`FRAME_V2_ESCAPE`]). The manifest's `formatVersion` remains authoritative
/// (★F6) — this sniff is defense-in-depth for the payload level.
pub fn is_frame_v2(payload: &[u8]) -> bool {
    payload.len() >= 2 && u16::from_le_bytes([payload[0], payload[1]]) == FRAME_V2_ESCAPE
}

/// Decode a full tile payload (the layer frame) into its layers.
///
/// Accepts the v1 frame in both shapes — aligned ([`ALIGNED_FRAME_FLAG`] set,
/// derived padding before each IPC stream) and the legacy unpadded frame
/// written before the flag existed — plus **self-contained** v2 frames
/// (inline schema sections). A v2 frame that references templates by hash
/// needs the dataset's registry: use [`decode_tile_with_templates`] (a hash
/// reference here is a descriptive error, not a panic).
pub fn decode_tile(payload: &[u8]) -> Result<Vec<DecodedLayer>> {
    if is_frame_v2(payload) {
        return decode_tile_v2(payload, None);
    }
    decode_tile_v1(payload)
}

/// [`decode_tile`] with the dataset's [`TemplateRegistry`], so v2 frames can
/// resolve their 16-byte template-hash references. v1 frames decode
/// unchanged (the registry is simply unused).
pub fn decode_tile_with_templates(
    payload: &[u8],
    templates: &TemplateRegistry,
) -> Result<Vec<DecodedLayer>> {
    if is_frame_v2(payload) {
        return decode_tile_v2(payload, Some(templates));
    }
    decode_tile_v1(payload)
}

/// The v1 frame walk — UNCHANGED from the 0.3.x reader.
fn decode_tile_v1(payload: &[u8]) -> Result<Vec<DecodedLayer>> {
    if payload.len() < 2 {
        return Err(Error::Other(
            "tile payload too short for layer frame".into(),
        ));
    }
    let raw_count = u16::from_le_bytes([payload[0], payload[1]]);
    let aligned = raw_count & ALIGNED_FRAME_FLAG != 0;
    let count = (raw_count & !ALIGNED_FRAME_FLAG) as usize;
    let mut pos = 2usize;
    // Cap the pre-allocation: `count` is attacker-controlled (up to 0x7FFE) and
    // each real layer costs many wire bytes, so a doctored count must not force
    // a giant up-front allocation. Mirrors the v2 walker.
    let mut layers = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let name_len = read_u16(payload, &mut pos)? as usize;
        let name = read_slice(payload, &mut pos, name_len)?;
        let name = String::from_utf8(name.to_vec())
            .map_err(|e| Error::Other(format!("layer name not utf8: {e}")))?;
        let ipc_len = read_u32(payload, &mut pos)? as usize;
        if aligned {
            let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
            read_slice(payload, &mut pos, pad)?;
        }
        let ipc = read_slice(payload, &mut pos, ipc_len)?;
        let batch = decode_layer(ipc)?;
        layers.push(DecodedLayer { name, batch });
    }
    Ok(layers)
}

/// Splice a template onto a section tail and decode the resulting stream.
///
/// Normative guards (design §3.4): the tail is EXACTLY the TOC-declared bytes
/// (the caller sliced it that way) and MUST begin with the `0xFFFFFFFF`
/// continuation marker — stray zero bytes make arrow-rs silently return an
/// EMPTY tile (they parse as a legacy 4-byte end-of-stream), so a malformed
/// section must error loudly instead. The template gets the same check
/// (a corrupt manifest entry hashes consistently but still must not splice).
fn splice_decode(template: &[u8], tail: &[u8], what: &str) -> Result<RecordBatch> {
    if template.len() < 4 || template[0..4] != [0xFF, 0xFF, 0xFF, 0xFF] {
        return Err(Error::Other(format!(
            "{what}: schema template does not start with an encapsulated Arrow message"
        )));
    }
    if tail.len() < 4 || tail[0..4] != [0xFF, 0xFF, 0xFF, 0xFF] {
        return Err(Error::Other(format!(
            "{what}: batch section does not start with the 0xFFFFFFFF continuation marker \
             (corrupt or misaligned section)"
        )));
    }
    let mut buf = Vec::with_capacity(template.len() + tail.len());
    buf.extend_from_slice(template);
    buf.extend_from_slice(tail);
    decode_layer(&buf)
}

/// Resolve a v2 layer's schema template: inline section or registry lookup.
fn resolve_template<'a>(
    ref_kind: u8,
    hash: Option<[u8; 16]>,
    inline: Option<&'a [u8]>,
    registry: Option<&'a TemplateRegistry>,
    what: &str,
) -> Result<&'a [u8]> {
    match ref_kind {
        REF_KIND_INLINE => inline.ok_or_else(|| {
            Error::Other(format!(
                "{what}: inline schema section missing from the frame"
            ))
        }),
        REF_KIND_TEMPLATE_HASH => {
            let hash = hash.expect("hash read for ref_kind 1");
            let registry = registry.ok_or_else(|| {
                Error::Other(format!(
                    "{what}: frame references schema template {} but no template registry \
                     was provided — open the dataset through its manifest (or use \
                     decode_tile_with_templates)",
                    hex_16(&hash)
                ))
            })?;
            registry.get(&hash).ok_or_else(|| {
                Error::Other(format!(
                    "{what}: schema template {} is not in the dataset's registry \
                     (manifest.schemas is incomplete or the frame is corrupt)",
                    hex_16(&hash)
                ))
            })
        }
        other => Err(Error::Other(format!(
            "{what}: unknown schema ref_kind {other} (this reader knows 0..=2)"
        ))),
    }
}

fn hex_16(hash: &[u8; 16]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// Merge a v2 layer's decoded CORE + PROPS batches back into the v1-shaped
/// single batch, RE-INJECTING the TILE_META values into schema/field metadata
/// with the exact v1 formatting — so every downstream consumer (validator,
/// stt-optimize, tests, renderers) sees v1-equivalent decoded layers with
/// zero changes.
fn merge_v2_layer(
    core: RecordBatch,
    props: Option<RecordBatch>,
    meta: &TileMeta,
) -> Result<RecordBatch> {
    let mut fields: Vec<Arc<Field>> = core.schema().fields().iter().cloned().collect();
    let mut columns: Vec<ArrayRef> = core.columns().to_vec();
    let mut schema_meta: HashMap<String, String> = core.schema().metadata().clone();

    if let Some(props) = props {
        if props.num_rows() != core.num_rows() {
            return Err(Error::Other(format!(
                "tile layer CORE/PROPS row counts disagree: {} vs {}",
                core.num_rows(),
                props.num_rows()
            )));
        }
        for (field, column) in props.schema().fields().iter().zip(props.columns()) {
            // Re-inject the hoisted attribute-quantization affine
            // (byte-identical to the v1 field metadata: `to_json` is a pure
            // function of the `[o, s]` pair TILE_META carries).
            let field = match meta.qa.as_ref().and_then(|qa| qa.get(field.name())) {
                Some(&(o, s)) => {
                    let mut m = field.metadata().clone();
                    m.insert(
                        STT_QUANT_ATTR_META_KEY.to_string(),
                        AttrQuant { o, s }.to_json(),
                    );
                    Arc::new(field.as_ref().clone().with_metadata(m))
                }
                None => field.clone(),
            };
            fields.push(field);
            columns.push(column.clone());
        }
    }

    // Schema-level re-injection, mirroring the v1 assembler's formatting.
    if let Some(t0) = meta.t0 {
        schema_meta.insert(TIME_OFFSET_MS_KEY.to_string(), t0.to_string());
    }
    if let Some((origin, step)) = meta.vt {
        schema_meta.insert(VERTEX_TIME_ORIGIN_KEY.to_string(), origin.to_string());
        schema_meta.insert(VERTEX_TIME_STEP_KEY.to_string(), step.to_string());
    }
    if let Some(buckets) = meta.vb {
        schema_meta.insert(VERTEX_VALUE_BUCKETS_KEY.to_string(), buckets.to_string());
    }

    let schema = Arc::new(Schema::new_with_metadata(fields, schema_meta));
    RecordBatch::try_new(schema, columns)
        .map_err(|e| Error::Other(format!("failed to merge v2 CORE/PROPS batches: {e}")))
}

/// The v2 frame walk. Bounds-checked byte reads throughout (`read_slice`), so
/// arbitrary/truncated input errors instead of panicking; unknown section
/// tags are skipped via their TOC length (additive evolution).
fn decode_tile_v2(
    payload: &[u8],
    registry: Option<&TemplateRegistry>,
) -> Result<Vec<DecodedLayer>> {
    let mut pos = 0usize;
    let escape = read_u16(payload, &mut pos)?;
    debug_assert_eq!(escape, FRAME_V2_ESCAPE, "caller dispatched on the escape");
    let frame_version = read_slice(payload, &mut pos, 1)?[0];
    if frame_version != FRAME_V2_VERSION {
        return Err(Error::Other(format!(
            "unsupported layer-frame version {frame_version} (this reader knows v2)"
        )));
    }
    let flags = read_slice(payload, &mut pos, 1)?[0];
    if flags != 0 {
        return Err(Error::Other(format!(
            "reserved v2 layer-frame flags must be 0, got {flags:#04x}"
        )));
    }
    let count = read_u16(payload, &mut pos)? as usize;
    let mut layers = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let name_len = read_u16(payload, &mut pos)? as usize;
        let name = read_slice(payload, &mut pos, name_len)?;
        let name = String::from_utf8(name.to_vec())
            .map_err(|e| Error::Other(format!("layer name not utf8: {e}")))?;

        let mut read_ref = |what: &str| -> Result<(u8, Option<[u8; 16]>)> {
            let kind = read_slice(payload, &mut pos, 1)?[0];
            let hash = if kind == REF_KIND_TEMPLATE_HASH {
                let mut h = [0u8; 16];
                h.copy_from_slice(read_slice(payload, &mut pos, 16)?);
                Some(h)
            } else if kind == REF_KIND_INLINE || kind == REF_KIND_NO_PROPS {
                None
            } else {
                return Err(Error::Other(format!(
                    "layer '{name}' {what}: unknown schema ref_kind {kind} \
                     (this reader knows 0..=2)"
                )));
            };
            Ok((kind, hash))
        };
        let (ref_core, core_hash) = read_ref("core")?;
        if ref_core == REF_KIND_NO_PROPS {
            return Err(Error::Other(format!(
                "layer '{name}': ref_kind_core 2 is invalid (every layer has a CORE batch)"
            )));
        }
        let (ref_props, props_hash) = read_ref("props")?;

        let section_count = read_slice(payload, &mut pos, 1)?[0] as usize;
        let mut toc: Vec<(u8, usize)> = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            let tag = read_slice(payload, &mut pos, 1)?[0];
            let len = read_u32(payload, &mut pos)? as usize;
            toc.push((tag, len));
        }
        let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
        read_slice(payload, &mut pos, pad)?;

        let mut sections: BTreeMap<u8, &[u8]> = BTreeMap::new();
        for (tag, len) in toc {
            let bytes = read_slice(payload, &mut pos, len)?;
            let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
            read_slice(payload, &mut pos, pad)?;
            if sections.insert(tag, bytes).is_some() {
                return Err(Error::Other(format!(
                    "layer '{name}': duplicate section tag 0x{tag:02x} in the TOC"
                )));
            }
        }

        // TILE_META: canonical JSON, unknown keys ignored (additive contract).
        let tile_meta: TileMeta = match sections.get(&SECTION_TILE_META) {
            Some(bytes) => serde_json::from_slice(bytes).map_err(|e| {
                Error::Other(format!("layer '{name}': TILE_META JSON decode failed: {e}"))
            })?,
            None => TileMeta::default(),
        };

        let core_template = resolve_template(
            ref_core,
            core_hash,
            sections.get(&SECTION_INLINE_SCHEMA_CORE).copied(),
            registry,
            &format!("layer '{name}' core"),
        )?;
        let core_tail = sections
            .get(&SECTION_CORE_BATCH)
            .ok_or_else(|| Error::Other(format!("layer '{name}': CORE_BATCH section missing")))?;
        let core = splice_decode(core_template, core_tail, &format!("layer '{name}' core"))?;

        let props = if ref_props == REF_KIND_NO_PROPS {
            if sections.contains_key(&SECTION_PROPS_BATCH) {
                return Err(Error::Other(format!(
                    "layer '{name}': PROPS_BATCH section present but ref_kind_props \
                     declares no props"
                )));
            }
            None
        } else {
            let template = resolve_template(
                ref_props,
                props_hash,
                sections.get(&SECTION_INLINE_SCHEMA_PROPS).copied(),
                registry,
                &format!("layer '{name}' props"),
            )?;
            let tail = sections.get(&SECTION_PROPS_BATCH).ok_or_else(|| {
                Error::Other(format!("layer '{name}': PROPS_BATCH section missing"))
            })?;
            Some(splice_decode(
                template,
                tail,
                &format!("layer '{name}' props"),
            )?)
        };

        let batch = merge_v2_layer(core, props, &tile_meta)?;
        layers.push(DecodedLayer { name, batch });
    }
    Ok(layers)
}

/// Walk ONLY a v2 frame's header structure — escape/version/flags/count,
/// per-layer name + schema ref_kinds (+ their 16-byte hashes), TOC-driven
/// section skips; **no Arrow decode, no section parse** — and return every
/// template hash the frame references. The packed-format validators use this
/// to prove each referenced hash resolves in `manifest.schemas` without
/// decoding tiles. Errors (never panics) on truncated/malformed headers.
pub fn frame_v2_template_refs(payload: &[u8]) -> Result<Vec<[u8; 16]>> {
    if !is_frame_v2(payload) {
        return Err(Error::Other("not a v2 layer frame (missing escape)".into()));
    }
    let mut pos = 2usize; // past the escape
    let frame_version = read_slice(payload, &mut pos, 1)?[0];
    if frame_version != FRAME_V2_VERSION {
        return Err(Error::Other(format!(
            "unsupported layer-frame version {frame_version} (this reader knows v2)"
        )));
    }
    let flags = read_slice(payload, &mut pos, 1)?[0];
    if flags != 0 {
        return Err(Error::Other(format!(
            "reserved v2 layer-frame flags must be 0, got {flags:#04x}"
        )));
    }
    let count = read_u16(payload, &mut pos)? as usize;
    let mut refs: Vec<[u8; 16]> = Vec::new();
    for _ in 0..count {
        let name_len = read_u16(payload, &mut pos)? as usize;
        read_slice(payload, &mut pos, name_len)?;
        for what in ["core", "props"] {
            let kind = read_slice(payload, &mut pos, 1)?[0];
            if kind == REF_KIND_TEMPLATE_HASH {
                let mut h = [0u8; 16];
                h.copy_from_slice(read_slice(payload, &mut pos, 16)?);
                refs.push(h);
            } else if kind != REF_KIND_INLINE && kind != REF_KIND_NO_PROPS {
                return Err(Error::Other(format!(
                    "{what}: unknown schema ref_kind {kind} (this reader knows 0..=2)"
                )));
            }
        }
        let section_count = read_slice(payload, &mut pos, 1)?[0] as usize;
        let mut toc: Vec<usize> = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            read_slice(payload, &mut pos, 1)?; // tag
            toc.push(read_u32(payload, &mut pos)? as usize);
        }
        let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
        read_slice(payload, &mut pos, pad)?;
        for len in toc {
            read_slice(payload, &mut pos, len)?;
            let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
            read_slice(payload, &mut pos, pad)?;
        }
    }
    Ok(refs)
}

fn read_u16(buf: &[u8], pos: &mut usize) -> Result<u16> {
    let s = read_slice(buf, pos, 2)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let s = read_slice(buf, pos, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_slice<'a>(buf: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::Other("tile frame length overflow".into()))?;
    if end > buf.len() {
        return Err(Error::Other("tile frame truncated".into()));
    }
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point_layer() -> ColumnarLayer {
        ColumnarLayer {
            name: "points".to_string(),
            feature_ids: vec![1, 2, 3],
            start_times: vec![1000, 2000, 3000],
            end_times: vec![1500, 2500, 3500],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8], [-122.6, 37.9]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "speed".to_string(),
                    PropertyColumn::Numeric(vec![Some(10.0), None, Some(30.0)]),
                ),
                (
                    "kind".to_string(),
                    PropertyColumn::Categorical(vec![
                        Some("car".to_string()),
                        Some("bus".to_string()),
                        None,
                    ]),
                ),
            ],
        }
    }

    /// Two DIFFERENT [`EncoderConfig`]s encode the SAME layer to DIFFERENT tiles
    /// in ONE process, and the output is driven purely by the passed config — not
    /// by the process-wide globals. This is the property that unblocks a dynamic
    /// server hosting several datasets/configs concurrently: if `encode_tile_with`
    /// read the (unset) globals instead of the config, the quantized and plain
    /// encodes would be identical and this would fail.
    #[test]
    fn encode_tile_with_is_config_driven_not_global() {
        let layer = sample_point_layer();
        let layers = std::slice::from_ref(&layer);

        let plain_cfg = EncoderConfig::default();
        let quant_cfg = EncoderConfig {
            quantize_coords_m: Some(1.0),
            ..EncoderConfig::default()
        };
        let attr_cfg = EncoderConfig {
            quantize_attrs_auto: true,
            ..EncoderConfig::default()
        };

        let plain = encode_tile_with(layers, &plain_cfg).unwrap();
        let quant = encode_tile_with(layers, &quant_cfg).unwrap();
        let attr = encode_tile_with(layers, &attr_cfg).unwrap();

        // Each explicit config yields a distinct tile — the config, not shared
        // state, decides the encoding. (If `encode_tile_with` read the unset
        // globals instead of the config, all three would be identical.) These
        // differences are config-driven at the WIRE COLUMN level — coord
        // quantization changes the geometry column (i32 grid vs Float64) and
        // attribute quantization changes the `speed` column (u16 vs Float64) — so
        // the inequality is not attributable to the encoder's (separately
        // tracked) non-deterministic Arrow-metadata ordering, which we therefore
        // deliberately do NOT byte-assert here.
        assert_ne!(plain, quant, "coord quantization must change the tile");
        assert_ne!(plain, attr, "attribute quantization must change the tile");
        assert_ne!(quant, attr, "the two quantizations differ from each other");

        // All three still decode to the SAME feature set — encoding differs, data
        // does not.
        for tile in [&plain, &quant, &attr] {
            let rows: usize = decode_tile(tile)
                .unwrap()
                .iter()
                .map(|l| l.batch.num_rows())
                .sum();
            assert_eq!(rows, 3);
        }
    }

    fn sample_line_layer() -> ColumnarLayer {
        ColumnarLayer {
            name: "tracks".to_string(),
            feature_ids: vec![10, 11],
            start_times: vec![0, 100],
            end_times: vec![50, 200],
            geometry: GeometryColumn::LineString(vec![
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                vec![[5.0, 5.0], [6.0, 6.0]],
            ]),
            vertex_times: Some(vec![vec![0, 25, 50], vec![100, 200]]),
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        }
    }

    fn sample_polygon_layer() -> ColumnarLayer {
        ColumnarLayer {
            name: "zones".to_string(),
            feature_ids: vec![42],
            start_times: vec![0],
            end_times: vec![1000],
            geometry: GeometryColumn::Polygon(vec![vec![
                // exterior ring
                vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
                // hole
                vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]],
            ]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        }
    }

    #[test]
    fn categorical_columns_use_dictionary_encoding() {
        let layer = ColumnarLayer {
            name: "cars".into(),
            feature_ids: vec![1, 2, 3, 4, 5],
            start_times: vec![0; 5],
            end_times: vec![1; 5],
            geometry: GeometryColumn::Point(vec![[0.0, 0.0]; 5]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![(
                "kind".into(),
                PropertyColumn::Categorical(vec![
                    Some("car".into()),
                    Some("bus".into()),
                    Some("car".into()),
                    None,
                    Some("car".into()),
                ]),
            )],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        let field = batch.schema().field_with_name("kind").unwrap().clone();
        match field.data_type() {
            DataType::Dictionary(k, v) => {
                assert_eq!(k.as_ref(), &DataType::UInt16);
                assert_eq!(v.as_ref(), &DataType::Utf8);
            }
            other => panic!("expected Dictionary<UInt16, Utf8>, got {other:?}"),
        }

        let col = batch
            .column_by_name("kind")
            .unwrap()
            .as_any()
            .downcast_ref::<DictionaryArray<UInt16Type>>()
            .unwrap();
        let values = col.values().as_any().downcast_ref::<StringArray>().unwrap();
        // First-seen order: "car" then "bus".
        let mut categories: Vec<&str> = (0..values.len()).map(|i| values.value(i)).collect();
        categories.sort();
        assert_eq!(categories, vec!["bus", "car"]);

        // The 4th row is null; others reference one of the two slots.
        assert!(col.is_null(3));
        let keys = col.keys();
        for i in [0usize, 1, 2, 4] {
            assert!(keys.value(i) < values.len() as u16);
        }
    }

    #[test]
    fn categorical_overflow_errors_instead_of_corrupting() {
        // A column whose distinct-value count exceeds the UInt16 dictionary
        // key space must be rejected, not silently collapsed onto one index.
        let n = u16::MAX as usize + 1; // 65_536 distinct strings
        let kinds: Vec<Option<String>> = (0..n).map(|i| Some(format!("c{i}"))).collect();
        let layer = ColumnarLayer {
            name: "huge".into(),
            feature_ids: (0..n as u64).collect(),
            start_times: vec![0; n],
            end_times: vec![1; n],
            geometry: GeometryColumn::Point(vec![[0.0, 0.0]; n]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![("kind".into(), PropertyColumn::Categorical(kinds))],
        };
        let err = encode_layer(&layer).expect_err("overflowing dictionary must error");
        assert!(
            err.to_string().contains("distinct values"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn geometry_field_advertises_crs_metadata() {
        // Every geometry field carries the GeoArrow extension *name* and the
        // CRS in extension *metadata*, so external GeoArrow readers see WGS84
        // lon/lat (OGC:CRS84) rather than an unknown CRS.
        for layer in [
            sample_point_layer(),
            sample_line_layer(),
            sample_polygon_layer(),
        ] {
            let ipc = encode_layer(&layer).unwrap();
            let batch = decode_layer(&ipc).unwrap();
            let field = batch.schema().field_with_name("geometry").unwrap().clone();
            let meta = field.metadata();
            assert_eq!(
                meta.get(GEOARROW_EXT_KEY).map(String::as_str),
                Some(layer.geometry.geoarrow_name())
            );
            let crs = meta
                .get(GEOARROW_EXT_META_KEY)
                .expect("geometry field must carry ARROW:extension:metadata");
            assert!(crs.contains("OGC:CRS84"), "crs metadata was: {crs}");
            assert!(crs.contains("crs_type"), "crs metadata was: {crs}");
        }
    }

    #[test]
    fn point_layer_roundtrips() {
        let layer = sample_point_layer();
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        assert_eq!(batch.num_rows(), 3);
        // id / start / end / geometry / speed / kind
        assert_eq!(batch.num_columns(), 6);

        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2, 3]);

        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(geom.len(), 3);
        assert_eq!(geom.value_length(), 2);

        // Geometry field carries the GeoArrow extension name.
        let geom_field = batch.schema().field_with_name("geometry").unwrap().clone();
        assert_eq!(
            geom_field
                .metadata()
                .get(GEOARROW_EXT_KEY)
                .map(String::as_str),
            Some("geoarrow.point")
        );

        // Nullable numeric property preserves the null.
        let speed = batch
            .column_by_name("speed")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(speed.is_null(1));
        assert_eq!(speed.value(0), 10.0);
    }

    #[test]
    fn vector_property_roundtrips_as_fixed_size_list() {
        // A Vector property encodes as FixedSizeList<leaf, width>: the f32 quat
        // as <Float32,4>, the u8 colour as <UInt8,4>, with the child buffer the
        // flattened row-major run the TS decoder hands to the GPU zero-copy.
        use arrow::array::{Float32Array, UInt8Array};
        let layer = ColumnarLayer {
            name: "surfels".to_string(),
            feature_ids: vec![1, 2],
            start_times: vec![0, 10],
            end_times: vec![0, 10],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "surfel_quat".to_string(),
                    PropertyColumn::Vector {
                        width: 4,
                        elem: VectorElem::F32,
                        values: vec![0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
                    },
                ),
                (
                    "surfel_rgba".to_string(),
                    PropertyColumn::Vector {
                        width: 4,
                        elem: VectorElem::U8,
                        values: vec![255.0, 0.0, 0.0, 128.0, 0.0, 255.0, 0.0, 255.0],
                    },
                ),
            ],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        let quat = batch
            .column_by_name("surfel_quat")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(quat.len(), 2);
        assert_eq!(quat.value_length(), 4);
        let qchild = quat
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(qchild.values(), &[0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]);

        let rgba = batch
            .column_by_name("surfel_rgba")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(rgba.value_length(), 4);
        let cchild = rgba.values().as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(cchild.values(), &[255, 0, 0, 128, 0, 255, 0, 255]);
    }

    #[test]
    fn vector_groups_fuse_scalar_columns_at_encode() {
        // `--vector-group` fuses named scalar columns into one interleaved
        // FixedSizeList and drops the scalars; ungrouped columns are untouched.
        use arrow::array::Float32Array;
        let layer = ColumnarLayer {
            name: "surfels".to_string(),
            feature_ids: vec![1, 2],
            start_times: vec![0, 10],
            end_times: vec![0, 10],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "qx".into(),
                    PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)]),
                ),
                (
                    "qy".into(),
                    PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)]),
                ),
                (
                    "qz".into(),
                    PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)]),
                ),
                (
                    "qw".into(),
                    PropertyColumn::Numeric(vec![Some(1.0), Some(0.5)]),
                ),
                (
                    "z".into(),
                    PropertyColumn::Numeric(vec![Some(3.0), Some(4.0)]),
                ),
            ],
        };
        // Explicit config (not the process-global setter) so this test can't
        // leak a non-default vector-group into a concurrently-running test that
        // reads the encoder globals via bare `encode_layer`.
        let cfg = EncoderConfig {
            vector_groups: vec![VectorGroup {
                name: "surfel_quat".to_string(),
                components: vec!["qx".into(), "qy".into(), "qz".into(), "qw".into()],
                elem: VectorElem::F32,
            }],
            ..EncoderConfig::default()
        };
        let ipc = encode_layer_with(&layer, &cfg).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        // Scalars fused away; the grouped vector + the ungrouped `z` remain.
        assert!(batch.column_by_name("qx").is_none());
        assert!(batch.column_by_name("z").is_some());
        let quat = batch
            .column_by_name("surfel_quat")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(quat.value_length(), 4);
        let qchild = quat
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(qchild.values(), &[0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn point_elevation_folds_into_3d_geometry_unquantized() {
        use arrow::array::Float64Array;
        let layer = ColumnarLayer {
            name: "cloud".into(),
            feature_ids: vec![1, 2],
            start_times: vec![0, 0],
            end_times: vec![0, 0],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "z".into(),
                    PropertyColumn::Numeric(vec![Some(3.5), Some(9.0)]),
                ),
                (
                    "speed".into(),
                    PropertyColumn::Numeric(vec![Some(1.0), Some(2.0)]),
                ),
            ],
        };
        let cfg = EncoderConfig {
            point_elevation_column: "z".to_string(),
            ..EncoderConfig::default()
        };
        let ipc = encode_layer_with(&layer, &cfg).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        // Geometry is now a 3-wide list with z folded in; `z` is gone as a property.
        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(geom.value_length(), 3);
        let coords = geom
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(coords.value(2), 3.5); // feature 0 z
        assert_eq!(coords.value(5), 9.0); // feature 1 z
        assert!(
            batch.column_by_name("z").is_none(),
            "z folded into geometry"
        );
        assert!(
            batch.column_by_name("speed").is_some(),
            "other props untouched"
        );
    }

    #[test]
    fn point_elevation_3d_geometry_quantizes_with_z_affine() {
        use arrow::array::Int32Array;
        let layer = ColumnarLayer {
            name: "cloud".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![0],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![("z".into(), PropertyColumn::Numeric(vec![Some(5.0)]))],
        };
        let cfg = EncoderConfig {
            quantize_coords_m: Some(0.05),
            point_elevation_column: "z".to_string(),
            ..EncoderConfig::default()
        };
        let ipc = encode_layer_with(&layer, &cfg).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        let field = batch.schema().field_with_name("geometry").unwrap().clone();
        let affine =
            QuantAffine::from_json(field.metadata().get(STT_QUANT_META_KEY).unwrap()).unwrap();
        assert_eq!(affine.z0, Some(0.0));
        assert_eq!(affine.sz, Some(0.05));
        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(geom.value_length(), 3);
        let coords = geom.values().as_any().downcast_ref::<Int32Array>().unwrap();
        // z = 5.0 / 0.05 = 100; reconstructs to z0 + 100*sz = 5.0.
        assert_eq!(coords.value(2), 100);
        assert_eq!(
            affine.z0.unwrap() + coords.value(2) as f64 * affine.sz.unwrap(),
            5.0
        );
    }

    #[test]
    fn quantized_point_layer_roundtrips_within_precision() {
        let layer = sample_point_layer();
        let ipc = encode_layer_quantized(&layer, Some(1.0)).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        // Geometry leaf is now i32 grid indices, and the affine rides in metadata.
        let geom_field = batch.schema().field_with_name("geometry").unwrap().clone();
        let affine = QuantAffine::from_json(
            geom_field
                .metadata()
                .get(STT_QUANT_META_KEY)
                .expect("quantized tile must carry the affine"),
        )
        .unwrap();

        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(geom.value_type(), DataType::Int32);
        let coords = geom.values().as_any().downcast_ref::<Int32Array>().unwrap();

        let original = [[-122.4, 37.7], [-122.5, 37.8], [-122.6, 37.9]];
        for (i, [lon, lat]) in original.iter().enumerate() {
            let rlon = affine.lon(coords.value(i * 2));
            let rlat = affine.lat(coords.value(i * 2 + 1));
            // Worst-case reconstruction error ≤ ~half a quantum (~0.5 m).
            let dlon_m = (rlon - lon).abs() * M_PER_DEG_LAT * lat.to_radians().cos();
            let dlat_m = (rlat - lat).abs() * M_PER_DEG_LAT;
            assert!(dlon_m < 1.0, "lon err {dlon_m} m at point {i}");
            assert!(dlat_m < 1.0, "lat err {dlat_m} m at point {i}");
        }
    }

    #[test]
    fn quantized_numeric_attr_roundtrips_within_precision_and_is_opt_in() {
        // A LiDAR-style `z` elevation column: high-entropy Float64 by default,
        // but fixed-point UInt16 when the build opts the column in. The reader
        // reconstructs `value = o + q*s`, lossy to <= s/2.
        let zvals: Vec<Option<f64>> = vec![Some(1.07), Some(-2.4), Some(15.9), None, Some(40.02)];
        let make = || ColumnarLayer {
            name: "lidar".into(),
            feature_ids: vec![1, 2, 3, 4, 5],
            start_times: vec![0; 5],
            end_times: vec![1; 5],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7]; 5]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![("z".into(), PropertyColumn::Numeric(zvals.clone()))],
        };

        // Default (no attr-quant configured): `z` stays Float64, byte-identical
        // to the historical encoder. Explicit config (not the global setter) so
        // this test stays hermetic under the parallel test runner.
        let plain =
            decode_layer(&encode_layer_with(&make(), &EncoderConfig::default()).unwrap()).unwrap();
        let zf = plain.schema().field_with_name("z").unwrap().clone();
        assert_eq!(zf.data_type(), &DataType::Float64);
        assert!(zf.metadata().get(STT_QUANT_ATTR_META_KEY).is_none());

        // Opt the `z` column in at 0.05-unit precision.
        let q = encode_layer_with(
            &make(),
            &EncoderConfig {
                quantize_attrs: HashMap::from([("z".to_string(), 0.05f64)]),
                ..EncoderConfig::default()
            },
        )
        .unwrap();

        let batch = decode_layer(&q).unwrap();
        let field = batch.schema().field_with_name("z").unwrap().clone();
        // Range (-2.4..40.02)/0.05 ~= 848 fits 16 bits → UInt16 leaf.
        assert_eq!(field.data_type(), &DataType::UInt16);
        let affine = AttrQuant::from_json(
            field
                .metadata()
                .get(STT_QUANT_ATTR_META_KEY)
                .expect("quantized attr must carry the affine"),
        )
        .unwrap();

        let col = batch
            .column_by_name("z")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap();
        for (i, want) in zvals.iter().enumerate() {
            match want {
                Some(v) => {
                    assert!(!col.is_null(i), "row {i} should be present");
                    let got = affine.value(col.value(i) as i64);
                    assert!((got - v).abs() <= 0.05 / 2.0 + 1e-9, "z[{i}] {got} vs {v}");
                }
                None => assert!(col.is_null(i), "row {i} should be null"),
            }
        }
    }

    #[test]
    fn auto_numeric_quantization_is_range_adaptive_and_opt_in() {
        // With auto-quant enabled, a raw Float64 property is quantized to a
        // UInt16 sized from its own [min,max] span (no precision configured),
        // and reconstructs to <= span/65535. Default-off keeps it Float64.
        let depth: Vec<Option<f64>> = vec![Some(0.0), Some(10.0), Some(123.4), Some(700.0)];
        let make = || ColumnarLayer {
            name: "q".into(),
            feature_ids: vec![1, 2, 3, 4],
            start_times: vec![0; 4],
            end_times: vec![1; 4],
            geometry: GeometryColumn::Point(vec![[0.0, 0.0]; 4]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![("depth".into(), PropertyColumn::Numeric(depth.clone()))],
        };

        // Default: auto off → Float64. Explicit config (not the global setter)
        // so this test can't flip `quantize_attrs_auto` under a concurrently
        // running test that reads the encoder globals via bare `encode_layer`.
        let plain =
            decode_layer(&encode_layer_with(&make(), &EncoderConfig::default()).unwrap()).unwrap();
        assert_eq!(
            plain.schema().field_with_name("depth").unwrap().data_type(),
            &DataType::Float64
        );

        // Auto on → range-adaptive UInt16 + affine.
        let batch = decode_layer(
            &encode_layer_with(
                &make(),
                &EncoderConfig {
                    quantize_attrs_auto: true,
                    ..EncoderConfig::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let field = batch.schema().field_with_name("depth").unwrap().clone();
        assert_eq!(field.data_type(), &DataType::UInt16);
        let aff =
            AttrQuant::from_json(field.metadata().get(STT_QUANT_ATTR_META_KEY).unwrap()).unwrap();
        let col = batch
            .column_by_name("depth")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap();
        let tol = (700.0 - 0.0) / u16::MAX as f64 / 2.0 + 1e-9;
        for (i, want) in depth.iter().enumerate() {
            let got = aff.value(col.value(i) as i64);
            assert!(
                (got - want.unwrap()).abs() <= tol,
                "depth[{i}] {got} vs {want:?}"
            );
        }
        // Min and max land on the index endpoints (full 16-bit span used).
        assert_eq!(col.value(0), 0);
        assert_eq!(col.value(3), u16::MAX);
    }

    #[test]
    fn quantization_shrinks_geometry_and_is_opt_in() {
        // A many-vertex line is coordinate-dominated; quantization should shrink
        // the IPC, and the default (None) path must stay byte-identical.
        let line: Vec<[f64; 2]> = (0..400)
            .map(|k| [-73.95 + k as f64 * 1e-4, 40.75 + k as f64 * 7e-5])
            .collect();
        let layer = ColumnarLayer {
            name: "q".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![1],
            geometry: GeometryColumn::LineString(vec![line]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let plain = encode_layer_quantized(&layer, None).unwrap();
        let quant = encode_layer_quantized(&layer, Some(1.0)).unwrap();

        // None keeps the Float64 GeoArrow leaf and carries no affine.
        let pb = decode_layer(&plain).unwrap();
        let pf = pb.schema().field_with_name("geometry").unwrap().clone();
        assert!(pf.metadata().get(STT_QUANT_META_KEY).is_none());

        // Some(_) switches the leaf to Int32 and emits the affine.
        let qb = decode_layer(&quant).unwrap();
        let qf = qb.schema().field_with_name("geometry").unwrap().clone();
        assert!(qf.metadata().get(STT_QUANT_META_KEY).is_some());

        // i32 coords (4 B) replace f64 (8 B) for a coordinate-dominated layer.
        assert!(
            quant.len() < plain.len(),
            "quantized {} should be smaller than f64 {}",
            quant.len(),
            plain.len()
        );
    }

    #[test]
    fn line_layer_roundtrips_with_vertex_times() {
        let layer = sample_line_layer();
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        assert_eq!(batch.num_rows(), 2);
        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        // Feature 0 has 3 vertices, feature 1 has 2.
        assert_eq!(geom.value(0).len(), 3);
        assert_eq!(geom.value(1).len(), 2);

        // v3 layers with a tight temporal span carry u16-delta vertex times
        // and the origin/step metadata needed to reconstruct absolutes.
        let meta = batch.schema().metadata().clone();
        let origin: i64 = meta
            .get("stt:vertex_time_origin_ms")
            .expect("u16 vertex-time layers carry an origin")
            .parse()
            .unwrap();
        let step: u32 = meta
            .get("stt:vertex_time_step_ms")
            .expect("u16 vertex-time layers carry a step")
            .parse()
            .unwrap();
        assert_eq!(origin, 0);
        assert_eq!(step, 1);

        let vt = batch
            .column_by_name("vertex_time")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(vt.len(), 2);
        let first = vt.value(0);
        let deltas = first
            .as_any()
            .downcast_ref::<arrow::array::UInt16Array>()
            .unwrap();
        let absolutes: Vec<i64> = deltas
            .values()
            .iter()
            .map(|d| origin + (*d as i64) * step as i64)
            .collect();
        assert_eq!(absolutes, vec![0, 25, 50]);
    }

    #[test]
    fn line_layer_roundtrips_with_vertex_values() {
        // Per-vertex scalars (e.g. SST) ride a nullable List<Float32> aligned
        // with the geometry vertices. A NaN entry marks a vertex with no value.
        let layer = ColumnarLayer {
            name: "drift".into(),
            feature_ids: vec![1, 2],
            start_times: vec![0, 0],
            end_times: vec![100, 100],
            geometry: GeometryColumn::LineString(vec![
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                vec![[3.0, 3.0], [4.0, 4.0]],
            ]),
            vertex_times: None,
            vertex_values: Some(vec![vec![5.0, f32::NAN, 27.5], vec![12.0, 13.0]]),
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        let vv = batch
            .column_by_name("vertex_value")
            .expect("layers with per-vertex values carry a vertex_value column")
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(vv.len(), 2);
        let first = vv.value(0);
        let vals = first
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap();
        assert_eq!(vals.value(0), 5.0);
        assert!(vals.value(1).is_nan());
        assert_eq!(vals.value(2), 27.5);
        let second = vv.value(1);
        let vals2 = second
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap();
        assert_eq!(vals2.values(), &[12.0, 13.0]);
    }

    #[test]
    fn line_layer_roundtrips_with_vertex_value_matrix() {
        // Static-geometry overview: per-vertex × per-bucket value matrix rides a
        // nullable List<Float32>, flattened vertex-major (vertex 0's buckets,
        // then vertex 1's, ...). num_buckets is recorded in schema metadata.
        // Feature 0: 3 vertices × 2 buckets; feature 1: 2 vertices × 2 buckets.
        let layer = ColumnarLayer {
            name: "flows".into(),
            feature_ids: vec![1, 2],
            start_times: vec![0, 0],
            end_times: vec![1800, 1800],
            geometry: GeometryColumn::LineString(vec![
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                vec![[3.0, 3.0], [4.0, 4.0]],
            ]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            // vertex-major: [v0b0, v0b1, v1b0, v1b1, v2b0, v2b1]
            vertex_value_matrix: Some(vec![
                vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
                vec![40.0, 41.0, 50.0, 51.0],
            ]),
            properties: vec![],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        let vm = batch
            .column_by_name("vertex_value_matrix")
            .expect("matrix layers carry a vertex_value_matrix column")
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(vm.len(), 2);
        let f0 = vm.value(0);
        let f0v = f0
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap();
        assert_eq!(f0v.values(), &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
        let f1 = vm.value(1);
        let f1v = f1
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap();
        assert_eq!(f1v.values(), &[40.0, 41.0, 50.0, 51.0]);

        // num_buckets = matrix_len / vertex_count = 6 / 3 = 2, in schema meta.
        assert_eq!(
            batch.schema().metadata().get("stt:vertex_value_buckets"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn vertex_time_falls_back_to_int64_for_wide_spans() {
        // span = 100 billion ms; the u16 encoding would need step ≈ 1.5e6 ms,
        // far beyond the DEFAULT_VERTEX_TIME_MAX_STEP_MS ceiling — so the
        // encoder must take the exact List<Int64> path, byte-for-byte
        // absolute timestamps, with no origin/step metadata.
        let layer = ColumnarLayer {
            name: "edge".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![100],
            geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0]]]),
            vertex_times: Some(vec![vec![0, 100_000_000_000]]),
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        let schema = batch.schema();
        let meta = schema.metadata();
        assert!(meta.get("stt:vertex_time_origin_ms").is_none());
        assert!(meta.get("stt:vertex_time_step_ms").is_none());
        let vt = batch
            .column_by_name("vertex_time")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let first = vt.value(0);
        let absolutes = first
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("wide spans must keep the exact Int64 shape");
        assert_eq!(absolutes.values(), &[0, 100_000_000_000]);
    }

    #[test]
    fn vertex_time_step_ceiling_is_the_u16_vs_int64_threshold() {
        // span = 65_535_000 ms quantizes at exactly the 1000 ms default
        // ceiling → u16 deltas; one ms more pushes the step to 1001 → i64.
        let make = |span: i64| ColumnarLayer {
            name: "edge".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![100],
            geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0]]]),
            vertex_times: Some(vec![vec![0, span]]),
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };

        let at_ceiling = decode_layer(&encode_layer(&make(65_535_000)).unwrap()).unwrap();
        let schema = at_ceiling.schema();
        let step: u32 = schema
            .metadata()
            .get("stt:vertex_time_step_ms")
            .expect("span at the ceiling stays u16-delta encoded")
            .parse()
            .unwrap();
        assert_eq!(step, DEFAULT_VERTEX_TIME_MAX_STEP_MS);

        let past_ceiling = decode_layer(&encode_layer(&make(65_536_000)).unwrap()).unwrap();
        assert!(past_ceiling
            .schema()
            .metadata()
            .get("stt:vertex_time_step_ms")
            .is_none());
        let vt = past_ceiling
            .column_by_name("vertex_time")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let first = vt.value(0);
        let absolutes = first.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(absolutes.values(), &[0, 65_536_000]);
    }

    #[test]
    fn polygon_layer_roundtrips_with_rings() {
        let layer = sample_polygon_layer();
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(geom.len(), 1);
        // One feature with two rings (exterior + hole).
        let rings = geom.value(0);
        let rings = rings.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(rings.len(), 2);
        assert_eq!(rings.value(0).len(), 5); // exterior ring vertices
        assert_eq!(rings.value(1).len(), 5); // hole vertices
    }

    #[test]
    fn multi_layer_tile_frame_roundtrips() {
        let layers = vec![sample_line_layer(), sample_point_layer()];
        let payload = encode_tile(&layers).unwrap();
        let decoded = decode_tile(&payload).unwrap();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].name, "tracks");
        assert_eq!(decoded[1].name, "points");
        assert_eq!(decoded[0].batch.num_rows(), 2);
        assert_eq!(decoded[1].batch.num_rows(), 3);
        // Schema metadata records the layer name on the batch too.
        assert_eq!(
            decoded[1]
                .batch
                .schema()
                .metadata()
                .get("stt:layer")
                .map(String::as_str),
            Some("points")
        );
    }

    #[test]
    fn tessellate_polygon_emits_two_triangles_for_a_square() {
        // A simple closed square (5 verts, last duplicates first) earcuts into
        // exactly 2 triangles, 6 indices in [0, 3].
        let ring: Vec<Coord> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
        let tris = tessellate_polygon(&[ring]);
        assert_eq!(tris.len(), 6);
        for &i in &tris {
            assert!(i < 5);
        }
    }

    #[test]
    fn tessellate_polygon_handles_a_hole() {
        // 4x4 square with a 1x1 hole — earcut should still produce a valid
        // tessellation. Index count is implementation-dependent but must be a
        // multiple of 3 and reference valid vertex indices.
        let exterior: Vec<Coord> = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]];
        let hole: Vec<Coord> = vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]];
        let tris = tessellate_polygon(&[exterior, hole]);
        assert!(tris.len() >= 6);
        assert_eq!(tris.len() % 3, 0);
        for &i in &tris {
            assert!(i < 10);
        }
    }

    #[test]
    fn tessellate_polygon_handles_degenerate_input() {
        // No rings → empty result, not a panic.
        assert!(tessellate_polygon(&[]).is_empty());
        // Single 2-vert ring is below the 3-vertex minimum.
        let degenerate: Vec<Coord> = vec![[0.0, 0.0], [1.0, 1.0]];
        assert!(tessellate_polygon(&[degenerate]).is_empty());
    }

    #[test]
    fn polygon_layer_with_triangles_roundtrips() {
        let exterior: Vec<Coord> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
        let tris = tessellate_polygon(&[exterior.clone()]);
        assert_eq!(tris.len(), 6);
        let layer = ColumnarLayer {
            name: "zones".into(),
            feature_ids: vec![42],
            start_times: vec![0],
            end_times: vec![1000],
            geometry: GeometryColumn::Polygon(vec![vec![exterior]]),
            vertex_times: None,
            vertex_values: None,
            triangles: Some(vec![tris.clone()]),
            vertex_value_matrix: None,
            properties: vec![],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        // Schema metadata advertises the sidecar.
        assert_eq!(
            batch
                .schema()
                .metadata()
                .get(TRIANGLES_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
        // Column exists with the expected shape. Indices here are tiny
        // (well under u16::MAX), so the narrower UInt16 encoding applies.
        let col = batch
            .column_by_name("triangles")
            .expect("triangles column present")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("triangles is a List");
        assert_eq!(col.len(), 1);
        let first = col.value(0);
        let values: &arrow::array::UInt16Array = first
            .as_any()
            .downcast_ref::<arrow::array::UInt16Array>()
            .expect("triangle values are UInt16 for small feature-local indices");
        assert_eq!(
            values
                .values()
                .iter()
                .map(|&v| v as u32)
                .collect::<Vec<_>>(),
            tris
        );
    }

    #[test]
    fn polygon_layer_with_oversized_triangle_index_falls_back_to_uint32() {
        // A feature-local triangle index beyond u16::MAX (pathological, but
        // possible for a single giant polygon) must fall back to UInt32
        // rather than silently truncating.
        let exterior: Vec<Coord> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
        let big_tris = vec![0u32, 1, 70_000];
        let layer = ColumnarLayer {
            name: "zones".into(),
            feature_ids: vec![42],
            start_times: vec![0],
            end_times: vec![1000],
            geometry: GeometryColumn::Polygon(vec![vec![exterior]]),
            vertex_times: None,
            vertex_values: None,
            triangles: Some(vec![big_tris.clone()]),
            vertex_value_matrix: None,
            properties: vec![],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        let col = batch
            .column_by_name("triangles")
            .expect("triangles column present")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("triangles is a List");
        let first = col.value(0);
        let values: &arrow::array::UInt32Array = first
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .expect("triangle values fall back to UInt32 when an index exceeds u16::MAX");
        assert_eq!(values.values().to_vec(), big_tris);
    }

    #[test]
    fn polygon_layer_without_triangles_skips_the_metadata_key() {
        // Backwards-compat guarantee: a v3 polygon layer that was NOT built
        // with pre-tessellation must not carry the metadata flag — otherwise
        // a reader would expect a column that isn't there.
        let layer = sample_polygon_layer();
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        assert!(!batch
            .schema()
            .metadata()
            .contains_key(TRIANGLES_METADATA_KEY));
        assert!(batch.column_by_name("triangles").is_none());
    }

    #[test]
    fn non_polygon_layer_drops_stray_triangles() {
        // A producer that mistakenly attaches `triangles` to a point or line
        // layer must not poison the wire format. The encoder silently drops
        // the column so the metadata key never appears.
        let mut layer = sample_point_layer();
        // Add a bogus per-feature triangle list. The encoder must ignore it.
        layer.triangles = Some(vec![vec![0, 1, 2]; layer.feature_ids.len()]);
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        assert!(!batch
            .schema()
            .metadata()
            .contains_key(TRIANGLES_METADATA_KEY));
        assert!(batch.column_by_name("triangles").is_none());
    }

    /// Walk a frame and return each layer's IPC start offset + length,
    /// honouring the aligned-frame padding rule.
    fn ipc_offsets(payload: &[u8]) -> Vec<(usize, usize)> {
        let raw = u16::from_le_bytes([payload[0], payload[1]]);
        let aligned = raw & ALIGNED_FRAME_FLAG != 0;
        let count = (raw & !ALIGNED_FRAME_FLAG) as usize;
        let mut pos = 2usize;
        let mut out = Vec::new();
        for _ in 0..count {
            let name_len = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
            pos += 2 + name_len;
            let ipc_len = u32::from_le_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]) as usize;
            pos += 4;
            if aligned {
                pos += (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
            }
            out.push((pos, ipc_len));
            pos += ipc_len;
        }
        out
    }

    #[test]
    fn encoded_frames_align_every_ipc_stream_to_8_bytes() {
        // Layer names of varying lengths so the unpadded offsets would land
        // all over the place; the aligned frame must place every IPC stream
        // at an 8-byte boundary regardless.
        let mut a = sample_line_layer();
        a.name = "x".into();
        let mut b = sample_point_layer();
        b.name = "a-longer-layer-name".into();
        let payload = encode_tile(&[a, b]).unwrap();

        let raw = u16::from_le_bytes([payload[0], payload[1]]);
        assert_ne!(
            raw & ALIGNED_FRAME_FLAG,
            0,
            "writer must set the aligned flag"
        );

        let offsets = ipc_offsets(&payload);
        assert_eq!(offsets.len(), 2);
        for (off, _) in &offsets {
            assert_eq!(off % 8, 0, "IPC stream at offset {off} is misaligned");
        }

        // And the padded frame still round-trips.
        let decoded = decode_tile(&payload).unwrap();
        assert_eq!(decoded[0].name, "x");
        assert_eq!(decoded[1].name, "a-longer-layer-name");
        assert_eq!(decoded[0].batch.num_rows(), 2);
        assert_eq!(decoded[1].batch.num_rows(), 3);
    }

    #[test]
    fn legacy_unpadded_frames_still_decode() {
        // Rebuild an old-style frame (no flag, no padding) from the layers'
        // IPC bytes — the shape every pre-alignment archive carries — and
        // assert the decoder reproduces the aligned frame's batches.
        let layers = vec![sample_line_layer(), sample_point_layer()];
        let aligned_payload = encode_tile(&layers).unwrap();
        let aligned = decode_tile(&aligned_payload).unwrap();

        let mut legacy: Vec<u8> = Vec::new();
        legacy.extend_from_slice(&(layers.len() as u16).to_le_bytes());
        for ((off, len), layer) in ipc_offsets(&aligned_payload).iter().zip(&layers) {
            let name = layer.name.as_bytes();
            legacy.extend_from_slice(&(name.len() as u16).to_le_bytes());
            legacy.extend_from_slice(name);
            legacy.extend_from_slice(&(*len as u32).to_le_bytes());
            legacy.extend_from_slice(&aligned_payload[*off..*off + *len]);
        }

        let decoded = decode_tile(&legacy).unwrap();
        assert_eq!(decoded.len(), aligned.len());
        for (l, a) in decoded.iter().zip(&aligned) {
            assert_eq!(l.name, a.name);
            assert_eq!(l.batch, a.batch);
        }
    }

    #[test]
    fn truncated_tile_frame_errors_cleanly() {
        let payload = encode_tile(&[sample_point_layer()]).unwrap();
        // Chop the payload mid-stream; decode must error, not panic.
        let truncated = &payload[..payload.len() / 2];
        assert!(decode_tile(truncated).is_err());
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let mut layer = sample_point_layer();
        layer.start_times.pop(); // now 2 entries vs 3 features
        assert!(encode_layer(&layer).is_err());
    }

    /// The v1 test's scenario through the V2 path: a length-inconsistent
    /// layer must be the same descriptive Err, not an index-out-of-bounds
    /// panic inside `sort_rows_by_start_time`'s column permutation. The
    /// start times are UNSORTED so the pre-sort actually permutes (a sorted
    /// layer short-circuits before touching the truncated column).
    #[test]
    fn length_mismatch_is_rejected_by_v2_frame_too() {
        let mut layer = sample_point_layer();
        layer.start_times = vec![3000, 1000, 2000];
        layer.end_times.pop(); // now 2 entries vs 3 features
        let err = encode_tile_with(
            &[layer],
            &EncoderConfig {
                format_version: FORMAT_VERSION_V2,
                ..EncoderConfig::default()
            },
        )
        .expect_err("length-inconsistent layer must Err through the v2 path");
        assert!(err.to_string().contains("end_times"), "got: {err}");
    }

    /// The v1 frame's leading u16 has only 15 bits for the layer count (bit
    /// 15 is the aligned flag), so counts in [0x8000, 0xFFFE] would OR into
    /// the flag as silently-mangled smaller counts, and 0x7FFF collides with
    /// the v2 escape. The guard must reject the ENTIRE range, not just the
    /// escape collision.
    #[test]
    fn v1_frame_rejects_unrepresentable_layer_counts() {
        // Boundary on the extracted predicate (cheap — no encoding).
        assert!(v1_layer_count_ok(0));
        assert!(v1_layer_count_ok(0x7FFE), "max representable count");
        assert!(!v1_layer_count_ok(0x7FFF), "v2-escape collision");
        assert!(!v1_layer_count_ok(0x8000), "bit-15 OR is a no-op → count 0");
        assert!(!v1_layer_count_ok(0xFFFE), "top of the mangled range");
        assert!(!v1_layer_count_ok(0x10000));

        // End-to-end: a 0x8000-layer tile previously ENCODED with a mangled
        // count of 0; now a descriptive error. The guard runs before any
        // per-layer encode, so 32Ki empty layers are cheap.
        let empty = ColumnarLayer {
            name: "l".to_string(),
            feature_ids: vec![],
            start_times: vec![],
            end_times: vec![],
            geometry: GeometryColumn::Point(vec![]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let layers = vec![empty; 0x8000];
        let err = encode_tile_with(&layers, &EncoderConfig::default())
            .expect_err("0x8000 layers must be rejected");
        assert!(err.to_string().contains("frame limit"), "got: {err}");
    }

    #[test]
    fn offsets_from_counts_errors_on_i32_overflow() {
        // Accumulating past i32::MAX must be a hard error, not a silent wrap
        // (release builds would otherwise emit corrupt Arrow list offsets).
        let ok = offsets_from_counts([3usize, 2, 0].into_iter()).unwrap();
        assert_eq!(ok.len(), 4); // N+1 offsets

        let at_limit = offsets_from_counts([i32::MAX as usize].into_iter());
        assert!(at_limit.is_ok(), "exactly i32::MAX vertices still fits");

        let over = offsets_from_counts([i32::MAX as usize, 1].into_iter())
            .expect_err("i32::MAX + 1 total vertices must error");
        assert!(
            over.to_string().contains("32-bit list offsets"),
            "got: {over}"
        );

        // A single count beyond i32::MAX errors too (the per-count try_from).
        assert!(offsets_from_counts([usize::MAX].into_iter()).is_err());
    }

    #[test]
    fn quantize_precision_below_floor_is_rejected() {
        // 1 mm would put the ±180° longitude index past i32::MAX. Both the
        // explicit-config path and the global setter must reject it with the
        // minimum in the message; a precision at/above the floor encodes fine.
        let layer = sample_point_layer();
        let err = encode_layer_with(
            &layer,
            &EncoderConfig {
                quantize_coords_m: Some(0.001),
                ..EncoderConfig::default()
            },
        )
        .expect_err("1 mm precision must be rejected");
        assert!(
            err.to_string().contains("minimum") && err.to_string().contains("overflow"),
            "error must state the minimum: {err}"
        );

        assert!(0.0187 > MIN_QUANTIZE_COORDS_M);
        assert!(encode_layer_with(
            &layer,
            &EncoderConfig {
                quantize_coords_m: Some(0.0187),
                ..EncoderConfig::default()
            },
        )
        .is_ok());

        // Global setter: rejects the same value WITHOUT storing it; <= 0
        // (off) still passes. (No positive value is stored here so the test
        // can't leak quantization into concurrently running global-path tests.)
        assert!(set_quantize_coords_m(0.001).is_err());
        assert!(set_quantize_coords_m(0.0).is_ok());
    }

    #[test]
    fn quantized_altitude_outside_i32_errors_instead_of_clamping() {
        // The z axis is unbounded input: a value whose grid index leaves i32
        // must error (identifying the value), not clamp to ±i32::MAX quanta.
        let make = |z: f64| ColumnarLayer {
            name: "cloud".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![0],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![("z".into(), PropertyColumn::Numeric(vec![Some(z)]))],
        };
        let cfg = EncoderConfig {
            quantize_coords_m: Some(0.05),
            point_elevation_column: "z".to_string(),
            ..EncoderConfig::default()
        };
        // Sane altitude still encodes.
        assert!(encode_layer_with(&make(5.0), &cfg).is_ok());
        // 1e18 m at a 0.05 m step → index 2e19, far outside i32.
        let err =
            encode_layer_with(&make(1.0e18), &cfg).expect_err("overflowing altitude must error");
        let msg = err.to_string();
        // f64 Display renders 1.0e18 as its full integer form.
        assert!(
            msg.contains("altitude") && msg.contains("1000000000000000000"),
            "got: {msg}"
        );
    }

    #[test]
    fn quantized_attr_index_beyond_i32_errors_instead_of_clamping() {
        // An attribute whose quantized index exceeds i32::MAX must error
        // (mirroring the dictionary-overflow error), not clamp to i32::MAX.
        let layer = ColumnarLayer {
            name: "wide".into(),
            feature_ids: vec![1, 2],
            start_times: vec![0; 2],
            end_times: vec![1; 2],
            geometry: GeometryColumn::Point(vec![[0.0, 0.0]; 2]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![(
                "v".into(),
                PropertyColumn::Numeric(vec![Some(0.0), Some(3.0e9)]),
            )],
        };
        let err = encode_layer_with(
            &layer,
            &EncoderConfig {
                quantize_attrs: HashMap::from([("v".to_string(), 1.0f64)]),
                ..EncoderConfig::default()
            },
        )
        .expect_err("index 3e9 > i32::MAX must error");
        let msg = err.to_string();
        assert!(
            msg.contains("overflows") && msg.contains("3000000000"),
            "got: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // Layer frame v2 (packed formatVersion 2)
    // ------------------------------------------------------------------

    /// v2 config in SELF-CONTAINED mode (inline schema sections, no registry
    /// needed to decode), layered over `base`.
    fn v2_inline(base: &EncoderConfig) -> EncoderConfig {
        EncoderConfig {
            format_version: FORMAT_VERSION_V2,
            template_collector: None,
            ..base.clone()
        }
    }

    /// v2 config in HASH-REFERENCING mode (templates recorded with
    /// `collector`; frames carry 16-byte hashes), layered over `base`.
    fn v2_hashed(base: &EncoderConfig, collector: &Arc<TemplateCollector>) -> EncoderConfig {
        EncoderConfig {
            format_version: FORMAT_VERSION_V2,
            template_collector: Some(Arc::clone(collector)),
            ..base.clone()
        }
    }

    /// Decode-side registry over everything `collector` recorded.
    fn registry_from(collector: &TemplateCollector) -> TemplateRegistry {
        let mut registry = TemplateRegistry::new();
        for (_, template) in collector.snapshot() {
            registry.insert(template);
        }
        registry
    }

    /// The v1-equivalence contract (design §4.3 / merge_v2_layer): the SAME
    /// layers encoded v1 and v2 (both v2 modes) decode to EQUAL
    /// `DecodedLayer`s — batch equality covers columns AND schema/field
    /// metadata, i.e. the TILE_META re-injection must reproduce the v1
    /// metadata byte-for-byte.
    fn assert_v2_decodes_like_v1(layers: &[ColumnarLayer], base: &EncoderConfig, what: &str) {
        let v1 = decode_tile(&encode_tile_with(layers, base).unwrap()).unwrap();

        let inline = decode_tile(&encode_tile_with(layers, &v2_inline(base)).unwrap()).unwrap();
        assert_eq!(inline.len(), v1.len(), "{what}: inline layer count");
        for (a, b) in inline.iter().zip(&v1) {
            assert_eq!(a.name, b.name, "{what}: inline layer name");
            assert_eq!(a.batch, b.batch, "{what}: inline v2 decode != v1 decode");
        }

        let collector = Arc::new(TemplateCollector::new());
        let payload = encode_tile_with(layers, &v2_hashed(base, &collector)).unwrap();
        let registry = registry_from(&collector);
        let hashed = decode_tile_with_templates(&payload, &registry).unwrap();
        assert_eq!(hashed.len(), v1.len(), "{what}: hashed layer count");
        for (a, b) in hashed.iter().zip(&v1) {
            assert_eq!(a.name, b.name, "{what}: hashed layer name");
            assert_eq!(a.batch, b.batch, "{what}: hashed v2 decode != v1 decode");
        }
    }

    /// Every payload shape the v2 break touches round-trips to EXACTLY the v1
    /// decode: plain + quantized points, dictionary props (incl. TWO
    /// dictionary columns), u16-delta AND exact-Int64 vertex_time, the
    /// vertex-value matrix, pre-tessellated triangles, an empty-bucket tile
    /// (zero rows, dictionary column intact), and a multi-layer tile.
    #[test]
    fn v2_roundtrip_equals_v1_decode_across_payload_shapes() {
        let plain = EncoderConfig::default();
        let quant = EncoderConfig {
            quantize_coords_m: Some(1.0),
            quantize_attrs_auto: true,
            ..EncoderConfig::default()
        };

        let two_dicts = ColumnarLayer {
            properties: vec![
                (
                    "kind".to_string(),
                    PropertyColumn::Categorical(vec![Some("car".into()), Some("bus".into()), None]),
                ),
                (
                    "color".to_string(),
                    PropertyColumn::Categorical(vec![
                        Some("red".into()),
                        None,
                        Some("blue".into()),
                    ]),
                ),
            ],
            ..sample_point_layer()
        };

        let wide_span_vt = ColumnarLayer {
            vertex_times: Some(vec![vec![0, 100_000_000_000], vec![0, 1]]),
            ..sample_line_layer()
        };

        let matrix = ColumnarLayer {
            name: "flows".into(),
            feature_ids: vec![1, 2],
            start_times: vec![0, 0],
            end_times: vec![1800, 1800],
            geometry: GeometryColumn::LineString(vec![
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                vec![[3.0, 3.0], [4.0, 4.0]],
            ]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: Some(vec![
                vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
                vec![40.0, 41.0, 50.0, 51.0],
            ]),
            properties: vec![],
        };

        let exterior: Vec<Coord> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
        let triangles = ColumnarLayer {
            triangles: Some(vec![tessellate_polygon(&[exterior.clone()])]),
            geometry: GeometryColumn::Polygon(vec![vec![exterior]]),
            ..sample_polygon_layer()
        };

        let empty = ColumnarLayer {
            name: "points".into(),
            feature_ids: vec![],
            start_times: vec![],
            end_times: vec![],
            geometry: GeometryColumn::Point(vec![]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                ("speed".into(), PropertyColumn::Numeric(vec![])),
                ("kind".into(), PropertyColumn::Categorical(vec![])),
            ],
        };

        assert_v2_decodes_like_v1(&[sample_point_layer()], &plain, "points");
        assert_v2_decodes_like_v1(&[sample_point_layer()], &quant, "quantized points");
        assert_v2_decodes_like_v1(&[two_dicts], &plain, "two dictionary columns");
        assert_v2_decodes_like_v1(&[sample_line_layer()], &plain, "u16-delta vertex_time");
        assert_v2_decodes_like_v1(&[wide_span_vt], &plain, "exact Int64 vertex_time");
        assert_v2_decodes_like_v1(&[matrix], &plain, "vertex-value matrix");
        assert_v2_decodes_like_v1(&[triangles], &plain, "pre-tessellated triangles");
        assert_v2_decodes_like_v1(&[empty], &quant, "empty-bucket tile");
        assert_v2_decodes_like_v1(
            &[sample_line_layer(), sample_point_layer()],
            &plain,
            "multi-layer tile",
        );
    }

    /// v2 row order (design §4.2 ★F10): rows come out stable-sorted by
    /// `start_time`, ids travel WITH their rows (the sort runs after id
    /// assignment), and the result equals the v1 decode of the pre-sorted
    /// layer. v1 encoding of the same input stays in input order.
    #[test]
    fn v2_rows_stable_sorted_by_start_time_after_id_assignment() {
        let unsorted = ColumnarLayer {
            name: "points".into(),
            feature_ids: vec![1, 2, 3, 4],
            start_times: vec![3000, 1000, 2000, 1000],
            end_times: vec![3500, 1500, 2500, 1600],
            geometry: GeometryColumn::Point(vec![
                [-122.4, 37.7],
                [-122.5, 37.8],
                [-122.6, 37.9],
                [-122.7, 38.0],
            ]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![(
                "kind".into(),
                PropertyColumn::Categorical(vec![
                    Some("a".into()),
                    Some("b".into()),
                    Some("c".into()),
                    Some("d".into()),
                ]),
            )],
        };

        let decoded = decode_tile(
            &encode_tile_with(&[unsorted.clone()], &v2_inline(&EncoderConfig::default())).unwrap(),
        )
        .unwrap();
        let batch = &decoded[0].batch;
        let starts = batch
            .column_by_name("start_time")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        // Stable: the two 1000s keep input order (id 2 before id 4).
        assert_eq!(starts, vec![1000, 1000, 2000, 3000]);
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(ids, vec![2, 4, 3, 1], "ids must travel with their rows");

        // Equivalent to v1-encoding the manually pre-sorted layer.
        let presorted = sort_rows_by_start_time(&unsorted).into_owned();
        let v1 = decode_tile(&encode_tile_with(&[presorted], &EncoderConfig::default()).unwrap())
            .unwrap();
        assert_eq!(batch, &v1[0].batch);
    }

    /// Template constancy (design §3.1d): tiles differing ONLY per-tile —
    /// qa affines, t0, dictionary categories, row counts — share ONE
    /// CORE + ONE PROPS template. Type variants (u16-delta vs exact-Int64
    /// vertex_time) legitimately mint DISTINCT templates (§2.2 cardinality),
    /// and every recorded template resolves through the registry.
    #[test]
    fn v2_template_constancy_and_type_variant_cardinality() {
        let quant = EncoderConfig {
            quantize_coords_m: Some(1.0),
            quantize_attrs_auto: true,
            ..EncoderConfig::default()
        };
        let collector = Arc::new(TemplateCollector::new());
        let cfg = v2_hashed(&quant, &collector);

        let tile = |seed: i64, cats: [&str; 2], n: usize| ColumnarLayer {
            name: "points".into(),
            feature_ids: (0..n as u64).collect(),
            start_times: (0..n as i64).map(|i| seed + i * 250).collect(),
            end_times: (0..n as i64).map(|i| seed + i * 250 + 100).collect(),
            geometry: GeometryColumn::Point(
                (0..n).map(|i| [-122.0 + i as f64 * 1e-3, 37.0]).collect(),
            ),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "speed".into(),
                    PropertyColumn::Numeric(
                        (0..n)
                            .map(|i| Some(seed as f64 * 0.01 + i as f64))
                            .collect(),
                    ),
                ),
                (
                    "kind".into(),
                    PropertyColumn::Categorical(
                        (0..n).map(|i| Some(cats[i % 2].to_string())).collect(),
                    ),
                ),
            ],
        };

        let a = encode_tile_with(&[tile(1_000_000, ["car", "bus"], 3)], &cfg).unwrap();
        let b = encode_tile_with(&[tile(9_000_000, ["tram", "ferry"], 5)], &cfg).unwrap();
        assert_ne!(a, b, "per-tile content must still differ");
        assert_eq!(
            collector.len(),
            2,
            "qa/t0/category/row-count variance must NOT mint new templates (core+props)"
        );

        // Type variants: narrow-span (u16-delta) vs wide-span (Int64)
        // vertex_time change the CORE schema → one new template each.
        let narrow = sample_line_layer();
        let wide = ColumnarLayer {
            vertex_times: Some(vec![vec![0, 100_000_000_000], vec![0, 1]]),
            ..sample_line_layer()
        };
        let c = encode_tile_with(&[narrow], &cfg).unwrap();
        let d = encode_tile_with(&[wide], &cfg).unwrap();
        assert_eq!(
            collector.len(),
            4,
            "u16 vs Int64 vertex_time are distinct templates"
        );

        // Every frame resolves through a registry built from the collector.
        let registry = registry_from(&collector);
        for payload in [&a, &b, &c, &d] {
            decode_tile_with_templates(payload, &registry).unwrap();
        }
    }

    /// TILE_META canonical serialization (design §4.3): alphabetical keys, no
    /// whitespace, and unknown keys are ignored on decode (additive contract).
    #[test]
    fn v2_tile_meta_is_canonical_json_and_ignores_unknown_keys() {
        let meta = TileMeta {
            qa: Some(BTreeMap::from([("speed".to_string(), (0.0, 0.15))])),
            sorted: Some(true),
            t0: Some(1_577_836_800_000),
            vb: Some(24),
            vt: Some((1_577_836_800_000, 1000)),
        };
        assert_eq!(
            serde_json::to_string(&meta).unwrap(),
            r#"{"qa":{"speed":[0.0,0.15]},"sorted":true,"t0":1577836800000,"vb":24,"vt":[1577836800000,1000]}"#
        );
        // Presence rules: absent features serialize NO key at all.
        assert_eq!(serde_json::to_string(&TileMeta::default()).unwrap(), "{}");
        // Unknown keys from a future writer must be ignored, not rejected.
        let parsed: TileMeta =
            serde_json::from_str(r#"{"sorted":true,"zz_future":{"x":1}}"#).unwrap();
        assert_eq!(parsed.sorted, Some(true));
    }

    /// Walk a SINGLE-layer v2 frame's header and return each section's
    /// `(tag, payload_offset, len)` — test-side mirror of the decoder's TOC
    /// walk, so guard tests can doctor exact section bytes.
    fn v2_section_spans(payload: &[u8]) -> Vec<(u8, usize, usize)> {
        assert!(is_frame_v2(payload));
        let mut pos = 6usize; // escape + frame_version + flags + layer_count
        let name_len = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
        pos += 2 + name_len;
        for _ in 0..2 {
            let kind = payload[pos];
            pos += 1;
            if kind == REF_KIND_TEMPLATE_HASH {
                pos += 16;
            }
        }
        let section_count = payload[pos] as usize;
        pos += 1;
        let mut toc = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            let tag = payload[pos];
            let len = u32::from_le_bytes(payload[pos + 1..pos + 5].try_into().unwrap()) as usize;
            toc.push((tag, len));
            pos += 5;
        }
        pos += (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
        let mut spans = Vec::with_capacity(section_count);
        for (tag, len) in toc {
            spans.push((tag, pos, len));
            pos += len;
            pos += (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
        }
        spans
    }

    /// Splice guards (design §3.4): stray zeros at a batch section's head
    /// must ERROR LOUDLY — under arrow-rs they'd otherwise parse as a legacy
    /// 4-byte end-of-stream and silently EMPTY the tile.
    #[test]
    fn v2_stray_zeros_in_batch_section_error_instead_of_empty_tile() {
        let payload = encode_tile_with(
            &[sample_point_layer()],
            &v2_inline(&EncoderConfig::default()),
        )
        .unwrap();
        let (_, off, len) = *v2_section_spans(&payload)
            .iter()
            .find(|(tag, _, _)| *tag == SECTION_CORE_BATCH)
            .expect("CORE_BATCH present");
        assert!(len > 4);

        let mut doctored = payload.clone();
        doctored[off..off + 4].fill(0);
        let err = decode_tile(&doctored).expect_err("zeroed continuation must error");
        assert!(
            err.to_string().contains("0xFFFFFFFF"),
            "error must name the continuation guard: {err}"
        );

        // Direct splice guard: both halves are checked.
        let template = &payload[..4]; // any bytes NOT starting with FFFFFFFF
        assert!(splice_decode(&[0u8; 8], template, "guard").is_err());
    }

    /// Frame guards: truncations anywhere in the header error cleanly, and a
    /// TOC length overrunning the payload is rejected before Arrow sees it.
    #[test]
    fn v2_truncated_header_and_lying_toc_length_error() {
        let payload = encode_tile_with(
            &[sample_point_layer()],
            &v2_inline(&EncoderConfig::default()),
        )
        .unwrap();
        let first_section_off = v2_section_spans(&payload)[0].1;
        for cut in 0..first_section_off {
            assert!(
                decode_tile(&payload[..cut]).is_err(),
                "cut at {cut} must error"
            );
        }

        // Inflate the first TOC length (u32 after the 1-byte tag): the
        // decoder's bounds-checked read must reject it, not over-read.
        let mut doctored = payload.clone();
        let toc0 = first_toc_offset(&payload);
        doctored[toc0 + 1..toc0 + 5].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = decode_tile(&doctored).expect_err("overrunning TOC length must error");
        assert!(err.to_string().contains("truncated"), "got: {err}");
    }

    /// Byte offset of a single-layer v2 frame's FIRST TOC entry.
    fn first_toc_offset(payload: &[u8]) -> usize {
        let name_len = u16::from_le_bytes([payload[6], payload[7]]) as usize;
        let mut pos = 8 + name_len;
        for _ in 0..2 {
            let kind = payload[pos];
            pos += 1;
            if kind == REF_KIND_TEMPLATE_HASH {
                pos += 16;
            }
        }
        pos + 1 // past section_count
    }

    /// Unknown section tags are SKIPPED via their TOC length (additive
    /// evolution): re-tagging TILE_META to an unknown tag still decodes the
    /// batch — only the re-injected per-tile metadata disappears.
    #[test]
    fn v2_unknown_section_tag_is_skipped() {
        let payload = encode_tile_with(
            &[sample_point_layer()],
            &v2_inline(&EncoderConfig::default()),
        )
        .unwrap();
        let toc0 = first_toc_offset(&payload);
        let mut doctored = payload.clone();
        // TOC entries are (u8 tag, u32 len); find TILE_META's entry.
        let section_count = doctored[toc0 - 1] as usize;
        let mut retagged = false;
        for i in 0..section_count {
            let at = toc0 + i * 5;
            if doctored[at] == SECTION_TILE_META {
                doctored[at] = 0x6f; // unknown tag
                retagged = true;
            }
        }
        assert!(retagged);
        let decoded = decode_tile(&doctored).unwrap();
        assert_eq!(decoded[0].batch.num_rows(), 3);
        assert!(
            decoded[0]
                .batch
                .schema()
                .metadata()
                .get(TIME_OFFSET_MS_KEY)
                .is_none(),
            "skipped TILE_META means no t0 re-injection"
        );
    }

    /// A hash-referencing v2 frame decoded WITHOUT (or with an incomplete)
    /// registry is a descriptive error naming the fix — never a panic.
    #[test]
    fn v2_hash_frame_without_registry_errors_descriptively() {
        let collector = Arc::new(TemplateCollector::new());
        let payload = encode_tile_with(
            &[sample_point_layer()],
            &v2_hashed(&EncoderConfig::default(), &collector),
        )
        .unwrap();

        let err = decode_tile(&payload).expect_err("no registry must error");
        assert!(
            err.to_string().contains("decode_tile_with_templates"),
            "error must point at the registry entry point: {err}"
        );

        let empty = TemplateRegistry::new();
        let err = decode_tile_with_templates(&payload, &empty)
            .expect_err("incomplete registry must error");
        assert!(
            err.to_string().contains("not in the dataset's registry"),
            "got: {err}"
        );
    }

    /// The strip is tail-invariant (design §3.6, spike-proven): hoisting the
    /// per-tile metadata out of the schema changes ONLY the schema message —
    /// two tiles differing in per-tile metadata alone share byte-identical
    /// templates while their tails differ, which is exactly what makes
    /// cross-tile template dedup sound.
    #[test]
    fn v2_metadata_strip_leaves_template_constant_and_tails_differing() {
        let quant = EncoderConfig {
            quantize_attrs_auto: true,
            ..EncoderConfig::default()
        };
        let mut early = sample_point_layer();
        // Different t0 + different qa affine (value range) per tile.
        for t in early.start_times.iter_mut() {
            *t += 7_000;
        }
        let late = sample_point_layer();

        let a = encode_layer_v2_parts(&early, &quant).unwrap();
        let b = encode_layer_v2_parts(&late, &quant).unwrap();
        assert_eq!(
            a.core_template, b.core_template,
            "CORE template must be constant"
        );
        let (a_props_template, a_props_tail) = a.props.as_ref().unwrap();
        let (b_props_template, b_props_tail) = b.props.as_ref().unwrap();
        assert_eq!(
            a_props_template, b_props_template,
            "PROPS template must be constant"
        );
        assert_ne!(a.core_tail, b.core_tail, "t0 shift must land in the tail");
        assert_ne!(
            a.tile_meta_json, b.tile_meta_json,
            "TILE_META varies per tile"
        );
        // Same qa affine + same categories here → identical props tails is
        // fine; what matters is templates never absorb per-tile variance.
        let _ = (a_props_tail, b_props_tail);
    }
}
