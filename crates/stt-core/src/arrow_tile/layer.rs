//! The in-memory tile model — what a producer hands the encoder.
//!
//! [`ColumnarLayer`] and its column types ([`GeometryColumn`],
//! [`PropertyColumn`]), the GeoArrow field-metadata key names that describe
//! them on the wire, and the earcut tessellation helper that bakes polygon
//! triangle indices.

use crate::error::{Error, Result};
use crate::types::GeometryType;

/// GeoArrow extension-name metadata key.
pub(crate) const GEOARROW_EXT_KEY: &str = "ARROW:extension:name";

/// GeoArrow extension-*metadata* key — the sibling of [`GEOARROW_EXT_KEY`] that
/// carries the per-geometry-type JSON metadata (CRS, edge interpretation, ...).
pub(crate) const GEOARROW_EXT_META_KEY: &str = "ARROW:extension:metadata";

/// CRS advertised on every geometry field.
///
/// STT stores interleaved `[lon, lat]` in WGS84, i.e. **OGC:CRS84** (the GeoJSON
/// longitude-first axis order) — *not* `EPSG:4326`, which strict readers treat as
/// lat/lon. Emitting it as a GeoArrow `authority_code` makes every tile
/// self-describing to GDAL / GeoPandas / lonboard / QGIS; without it those
/// readers fall back to "unknown CRS" even though the geometry is plain lon/lat.
pub(crate) const GEOARROW_CRS_METADATA: &str = r#"{"crs":"OGC:CRS84","crs_type":"authority_code"}"#;
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
    pub(crate) fn geoarrow_name(&self) -> &'static str {
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
    pub(crate) fn validate(&self) -> Result<()> {
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
