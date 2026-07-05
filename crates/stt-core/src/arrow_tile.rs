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
//! carry, say, a linestring layer and a point layer side by side:
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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

/// Top bit of the layer frame's leading u16: marks an *aligned* frame whose
/// IPC streams are padded to 8-byte boundaries (see the module docs). The
/// remaining 15 bits carry the layer count, so a tile may have at most
/// `0x7fff` layers. Old frames never set this bit (layer counts are tiny).
pub const ALIGNED_FRAME_FLAG: u16 = 0x8000;

/// Alignment (bytes) of each layer's Arrow IPC stream within an aligned frame.
const FRAME_ALIGN: usize = 8;

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
fn offsets_from_counts(counts: impl Iterator<Item = usize>) -> OffsetBuffer<i32> {
    let mut acc = 0i32;
    let mut offsets = vec![0i32];
    for c in counts {
        acc += c as i32;
        offsets.push(acc);
    }
    OffsetBuffer::new(offsets.into())
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
    #[inline]
    fn qz(&self, z: f64) -> i32 {
        let z0 = self.z0.unwrap_or(0.0);
        let sz = self.sz.unwrap_or(1.0);
        (((z - z0) / sz).round() as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }
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
) -> ArrayRef {
    // 3D POINT path: interleave [x,y,z] into a FixedSizeList<_,3> leaf.
    if let (GeometryColumn::Point(points), Some(elev)) = (geom, point_elev) {
        let list_size = 3;
        let dt = if quant.is_some() { DataType::Int32 } else { DataType::Float64 };
        let field = Arc::new(Field::new("xyz", dt, false));
        let child: ArrayRef = match quant {
            Some(q) => {
                let mut iv = Vec::with_capacity(points.len() * 3);
                for (i, [x, y]) in points.iter().enumerate() {
                    iv.push(q.qx(*x));
                    iv.push(q.qy(*y));
                    iv.push(q.qz(elev.get(i).copied().unwrap_or(0.0)));
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
        return Arc::new(FixedSizeListArray::new(field, list_size, child, None));
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

    match geom {
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
            let offsets = offsets_from_counts(lines.iter().map(|l| l.len()));
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
            let ring_offsets = offsets_from_counts(ring_sizes.into_iter());
            let vertex_field = Arc::new(Field::new("vertices", coords.data_type().clone(), false));
            let rings: ArrayRef = Arc::new(ListArray::new(
                vertex_field,
                ring_offsets,
                coords,
                None,
            ));
            // Feature level: List<List<FixedSizeList>>.
            let feature_offsets = offsets_from_counts(rings_per_feature.into_iter());
            let ring_field = Arc::new(Field::new("rings", rings.data_type().clone(), false));
            Arc::new(ListArray::new(ring_field, feature_offsets, rings, None))
        }
    }
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
/// size/precision trade-off.
pub fn set_quantize_coords_m(meters: f64) {
    let um = if meters > 0.0 {
        (meters * 1.0e6).round().clamp(1.0, u32::MAX as f64) as u32
    } else {
        0
    };
    QUANTIZE_COORDS_UM.store(um, Ordering::Relaxed);
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
            || !g.components.iter().all(|c| numeric.contains_key(c.as_str()))
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
fn build_quantized_numeric(values: &[Option<f64>], prec: f64) -> Option<(ArrayRef, String)> {
    if !(prec > 0.0) {
        return None;
    }
    let mut min = f64::INFINITY;
    for v in values.iter().flatten() {
        if v.is_finite() && *v < min {
            min = *v;
        }
    }
    if !min.is_finite() {
        return None; // no finite values — keep Float64
    }
    let affine = AttrQuant { o: min, s: prec };
    let mut q: Vec<Option<i64>> = Vec::with_capacity(values.len());
    let mut max_q: i64 = 0;
    for v in values {
        match v {
            Some(x) if x.is_finite() => {
                let qi = (((*x - affine.o) / affine.s).round() as i64).max(0);
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
                Some(v) => b.append_value((*v).clamp(0, i32::MAX as i64) as i32),
                None => b.append_null(),
            }
        }
        Arc::new(b.finish())
    };
    Some((array, affine.to_json()))
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
fn build_dictionary_indices(
    values: &[Option<String>],
) -> Result<(Vec<Option<u16>>, Vec<String>)> {
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
                            let delta = ((t - min) as u64 / step as u64).min(u16::MAX as u64) as u16;
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

/// The single encode implementation, driven entirely by an explicit
/// [`EncoderConfig`]. Every public `encode_layer*` entry point funnels here.
fn encode_layer_cfg(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<Vec<u8>> {
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
    let point_elev: Option<Vec<f64>> = if !elev_col.is_empty()
        && matches!(layer.geometry, GeometryColumn::Point(_))
    {
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
    let quant = cfg.quantize_coords_m.and_then(|m| {
        if point_elev.is_some() {
            world_grid_affine_3d(m)
        } else {
            world_grid_affine(m)
        }
    });
    let geom_array =
        build_geometry_array_q(&layer.geometry, quant.as_ref(), point_elev.as_deref());
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
    if let Some(vt_col) = build_vertex_time_array(&layer.vertex_times, n, cfg.vertex_time_max_step_ms) {
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

    // Fuse configured scalar columns into GPU-ready interleaved Vector columns
    // (e.g. qx/qy/qz/qw → one FixedSizeList<f32,4>). Runs BEFORE the quantize
    // loop so grouped components are written as the raw vector, not individually
    // quantized. No groups configured ⇒ iterate `layer.properties` with no clone.
    let grouped =
        group_vector_properties(&layer.properties, layer.feature_count(), &cfg.vector_groups);
    let props_iter: &[(String, PropertyColumn)] =
        grouped.as_deref().unwrap_or(&layer.properties);
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
                let quantized = cfg
                    .quantize_attrs
                    .get(name)
                    .copied()
                    .filter(|p| *p > 0.0)
                    .and_then(|p| build_quantized_numeric(values, p))
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
                    categories.iter().map(|s| Some(s.as_str())).collect::<Vec<_>>(),
                ));
                let key_array = UInt16Array::from(indices);
                let dict = DictionaryArray::<UInt16Type>::try_new(key_array, value_array)
                    .map_err(|e| Error::Other(format!("dictionary build failed: {e}")))?;
                columns.push(Arc::new(dict));
            }
            PropertyColumn::Vector { width, elem, values } => {
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
                let fsl = FixedSizeListArray::new(item_field, *width as i32, child, None);
                fields.push(Arc::new(Field::new(
                    name,
                    fsl.data_type().clone(),
                    true,
                )));
                columns.push(Arc::new(fsl));
            }
        }
    }

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
    schema_meta.insert("stt:layer".to_string(), layer.name.clone());
    schema_meta.insert(
        "stt:geometry".to_string(),
        layer.geometry.geoarrow_name().to_string(),
    );
    // Bake the layer's minimum feature start-time (integer Unix ms) so the TS
    // decoder can skip its client-side min-scan over the whole start-time column
    // and relativize times against this value directly. Mirrors exactly what the
    // decoder computes (the min of the `start_time` column); only emitted when a
    // start-time column is present. See packages/core/src/tile.ts.
    if let Some(min_start) = layer.start_times.iter().copied().min() {
        schema_meta.insert(TIME_OFFSET_MS_KEY.to_string(), min_start.to_string());
    }
    if let Some((origin, step)) = vertex_time_encoding {
        schema_meta.insert(VERTEX_TIME_ORIGIN_KEY.to_string(), origin.to_string());
        schema_meta.insert(VERTEX_TIME_STEP_KEY.to_string(), step.to_string());
    }
    if let Some(buckets) = vertex_value_buckets {
        schema_meta.insert(VERTEX_VALUE_BUCKETS_KEY.to_string(), buckets.to_string());
    }
    if has_triangles {
        schema_meta.insert(TRIANGLES_METADATA_KEY.to_string(), "true".to_string());
    }
    let schema = Arc::new(Schema::new(fields).with_metadata(schema_meta.into_iter().collect()));

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
/// [`EncoderConfig`]. Every public `encode_tile*` entry point funnels here.
fn encode_tile_cfg(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    if layers.len() >= ALIGNED_FRAME_FLAG as usize {
        return Err(Error::Other(format!(
            "tile has {} layers, exceeds the {} frame limit",
            layers.len(),
            ALIGNED_FRAME_FLAG - 1
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
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&(ipc.len() as u32).to_le_bytes());
        let pad = (FRAME_ALIGN - out.len() % FRAME_ALIGN) % FRAME_ALIGN;
        out.extend_from_slice(&[0u8; FRAME_ALIGN][..pad]);
        out.extend_from_slice(&ipc);
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
        0 => Err(Error::Other("tile layer IPC contained no record batch".into())),
        1 => Ok(batches.into_iter().next().unwrap()),
        // A layer is written as exactly one batch; concatenating is the safe
        // fallback if a producer ever splits it.
        _ => arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .map_err(|e| Error::Other(format!("failed to concat tile batches: {e}"))),
    }
}

/// Decode a full tile payload (the layer frame) into its layers.
///
/// Accepts both frame shapes: the aligned frame ([`ALIGNED_FRAME_FLAG`] set,
/// with derived padding before each IPC stream) and the legacy unpadded
/// frame written by every archive that predates the flag.
pub fn decode_tile(payload: &[u8]) -> Result<Vec<DecodedLayer>> {
    if payload.len() < 2 {
        return Err(Error::Other("tile payload too short for layer frame".into()));
    }
    let raw_count = u16::from_le_bytes([payload[0], payload[1]]);
    let aligned = raw_count & ALIGNED_FRAME_FLAG != 0;
    let count = (raw_count & !ALIGNED_FRAME_FLAG) as usize;
    let mut pos = 2usize;
    let mut layers = Vec::with_capacity(count);
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
            geometry: GeometryColumn::Point(vec![
                [-122.4, 37.7],
                [-122.5, 37.8],
                [-122.6, 37.9],
            ]),
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
            let rows: usize = decode_tile(tile).unwrap().iter().map(|l| l.batch.num_rows()).sum();
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
        let values = col
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
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
        for layer in [sample_point_layer(), sample_line_layer(), sample_polygon_layer()] {
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
            geom_field.metadata().get(GEOARROW_EXT_KEY).map(String::as_str),
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
        assert_eq!(
            qchild.values(),
            &[0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]
        );

        let rgba = batch
            .column_by_name("surfel_rgba")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(rgba.value_length(), 4);
        let cchild = rgba
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();
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
                ("qx".into(), PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)])),
                ("qy".into(), PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)])),
                ("qz".into(), PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)])),
                ("qw".into(), PropertyColumn::Numeric(vec![Some(1.0), Some(0.5)])),
                ("z".into(), PropertyColumn::Numeric(vec![Some(3.0), Some(4.0)])),
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
                ("z".into(), PropertyColumn::Numeric(vec![Some(3.5), Some(9.0)])),
                ("speed".into(), PropertyColumn::Numeric(vec![Some(1.0), Some(2.0)])),
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
        let coords = geom.values().as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(coords.value(2), 3.5); // feature 0 z
        assert_eq!(coords.value(5), 9.0); // feature 1 z
        assert!(batch.column_by_name("z").is_none(), "z folded into geometry");
        assert!(batch.column_by_name("speed").is_some(), "other props untouched");
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
        let affine = QuantAffine::from_json(field.metadata().get(STT_QUANT_META_KEY).unwrap()).unwrap();
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
        assert_eq!(affine.z0.unwrap() + coords.value(2) as f64 * affine.sz.unwrap(), 5.0);
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
        let coords = geom
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

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
        let zvals: Vec<Option<f64>> =
            vec![Some(1.07), Some(-2.4), Some(15.9), None, Some(40.02)];
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
        let aff = AttrQuant::from_json(field.metadata().get(STT_QUANT_ATTR_META_KEY).unwrap()).unwrap();
        let col = batch.column_by_name("depth").unwrap().as_any().downcast_ref::<UInt16Array>().unwrap();
        let tol = (700.0 - 0.0) / u16::MAX as f64 / 2.0 + 1e-9;
        for (i, want) in depth.iter().enumerate() {
            let got = aff.value(col.value(i) as i64);
            assert!((got - want.unwrap()).abs() <= tol, "depth[{i}] {got} vs {want:?}");
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
        let deltas = first.as_any().downcast_ref::<arrow::array::UInt16Array>().unwrap();
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
        let vals = first.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
        assert_eq!(vals.value(0), 5.0);
        assert!(vals.value(1).is_nan());
        assert_eq!(vals.value(2), 27.5);
        let second = vv.value(1);
        let vals2 = second.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
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
        let f0v = f0.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
        assert_eq!(f0v.values(), &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
        let f1 = vm.value(1);
        let f1v = f1.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
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
        let ring: Vec<Coord> = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ];
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
        let exterior: Vec<Coord> =
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]];
        let hole: Vec<Coord> =
            vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]];
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
        let exterior: Vec<Coord> =
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
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
            values.values().iter().map(|&v| v as u32).collect::<Vec<_>>(),
            tris
        );
    }

    #[test]
    fn polygon_layer_with_oversized_triangle_index_falls_back_to_uint32() {
        // A feature-local triangle index beyond u16::MAX (pathological, but
        // possible for a single giant polygon) must fall back to UInt32
        // rather than silently truncating.
        let exterior: Vec<Coord> =
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
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
        assert!(!batch.schema().metadata().contains_key(TRIANGLES_METADATA_KEY));
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
        assert!(!batch.schema().metadata().contains_key(TRIANGLES_METADATA_KEY));
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
            let name_len =
                u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
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
        assert_ne!(raw & ALIGNED_FRAME_FLAG, 0, "writer must set the aligned flag");

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
}
