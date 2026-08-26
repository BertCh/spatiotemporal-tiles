//! The in-memory tile model — what a producer hands the encoder.
//!
//! [`ColumnarLayer`] and its column types ([`GeometryColumn`],
//! [`PropertyColumn`]), the GeoArrow field-metadata key names that describe
//! them on the wire, the earcut tessellation helper that bakes polygon
//! triangle indices, and the feature-id identity model ([`FeatureIdOrigin`],
//! [`IdRenumberMap`]) that lets the builder make the `id` column
//! delta-codable without ever forfeiting cross-tile identity.

use crate::error::{Error, Result};
use crate::types::GeometryType;
use arrow::datatypes::{DataType, Field};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

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

/// vis.gl temporal field-metadata keys — the `visgl:temporal-*` vocabulary
/// luma.gl's `@luma.gl/arrow` writes onto a prepared temporal field
/// (`modules/arrow/src/arrow/vectors/arrow-temporal-gpu-vector.ts`).
///
/// Why borrow someone else's namespace instead of minting `stt:*` twins: this
/// is the same job [`GEOARROW_EXT_KEY`] does for geometry. A time column that
/// says only `Int64` is not self-describing — nothing in the type distinguishes
/// epoch milliseconds from an opaque counter. Emitting the vocabulary a generic
/// Arrow consumer already looks for is what makes a tile's time columns legible
/// to something that has never heard of STT.
///
/// **These keys are never written to the wire.** Every value is derivable from
/// the decoded column, so paying for them per tile would be paying for a
/// restatement: a self-contained frame (what `stt-serve` emits, with an inline
/// schema section per tile) would carry ~450 B of them on EVERY tile, measured
/// at +65 to +181 B compressed per tile on `stt-optimize`'s sample layouts.
/// They are instead stamped on at DECODE, by [`decorate_temporal_fields`] here
/// and by `toGeoArrowTable()` in `poopdeck:packages/core/src/tile.ts` — which also means
/// every archive ever written gets them, not just ones built after this landed,
/// and no content address churns.
pub(crate) const TEMPORAL_KIND_KEY: &str = "visgl:temporal-kind";
/// See [`TEMPORAL_KIND_KEY`]. One of `day`/`second`/`millisecond`/
/// `microsecond`/`nanosecond`; STT is always `millisecond`.
pub(crate) const TEMPORAL_UNIT_KEY: &str = "visgl:temporal-unit";
/// See [`TEMPORAL_KIND_KEY`]. The absolute value the column's values are
/// measured FROM, in the unit named by [`TEMPORAL_UNIT_KEY`].
pub(crate) const TEMPORAL_ORIGIN_KEY: &str = "visgl:temporal-origin";
/// See [`TEMPORAL_KIND_KEY`]. How [`TEMPORAL_ORIGIN_KEY`] was chosen —
/// `zero` (absolute values) or `first-valid` (relativized against the column's
/// own first valid value).
pub(crate) const TEMPORAL_ORIGIN_POLICY_KEY: &str = "visgl:temporal-origin-policy";
/// See [`TEMPORAL_KIND_KEY`]. STT timestamps are always UTC.
pub(crate) const TEMPORAL_TIMEZONE_KEY: &str = "visgl:temporal-timezone";

/// The STT time columns, in the order they appear in a decoded batch.
pub(crate) const TEMPORAL_COLUMNS: [&str; 3] = ["start_time", "end_time", "vertex_time"];

/// Whether `field` holds ABSOLUTE Unix milliseconds rather than one of the
/// compact relative forms.
///
/// Signed and 64-bit. Every compact form is UNSIGNED — `UInt32` start offsets
/// and durations, `List<UInt16|UInt32>` vertex-time deltas — so dropping the
/// signedness clause would declare all of them absolute, which is exactly the
/// wrong answer.
fn is_absolute_ms(field: &Field) -> bool {
    match field.data_type() {
        DataType::Int64 => true,
        DataType::List(item) => matches!(item.data_type(), DataType::Int64),
        _ => false,
    }
}

/// Stamp the `visgl:temporal-*` descriptor onto a decoded layer's time columns.
///
/// Runs at the END of decode, after every compact form has been re-inflated, so
/// it describes what the caller is actually handed rather than what was on the
/// wire. `start_time` / `end_time` are always absolute by then; `vertex_time`
/// is the one column that is NOT re-inflated, so its delta forms report the
/// domain and stay silent about the origin — their anchor is the per-tile
/// `stt:vertex_time_origin_ms`, which is not a property of the column at all.
/// Advertising one would place every vertex of every other tile at the wrong
/// instant.
///
/// The TypeScript counterpart is `toGeoArrowTable()` in
/// `poopdeck:packages/core/src/tile.ts`; the two must agree key for key.
pub(crate) fn decorate_temporal_fields(fields: &mut [Arc<Field>]) {
    for field in fields.iter_mut() {
        if !TEMPORAL_COLUMNS.contains(&field.name().as_str()) {
            continue;
        }
        let mut m = field.metadata().clone();
        m.insert(TEMPORAL_KIND_KEY.to_string(), "timestamp".to_string());
        m.insert(TEMPORAL_UNIT_KEY.to_string(), "millisecond".to_string());
        m.insert(TEMPORAL_TIMEZONE_KEY.to_string(), "UTC".to_string());
        if is_absolute_ms(field) {
            m.insert(TEMPORAL_ORIGIN_KEY.to_string(), "0".to_string());
            m.insert(TEMPORAL_ORIGIN_POLICY_KEY.to_string(), "zero".to_string());
        }
        *field = Arc::new(field.as_ref().clone().with_metadata(m));
    }
}

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
    /// Optional per-feature PART boundaries for polygon features: the RING
    /// INDEX (relative to that feature's own first ring) at which each part of
    /// a MultiPolygon begins.
    ///
    /// Part 0 always starts at ring 0, so a single-part `Polygon`'s entry is
    /// `[0]`; a two-part MultiPolygon whose first part is a square with one
    /// hole is `[0, 2]`. When present, `polygon_parts.len() == feature_count`.
    ///
    /// This is the one thing `geoarrow.polygon` (`List<List<FixedSizeList>>` —
    /// one flat ring list per feature) cannot express: the builder flattens a
    /// MultiPolygon's parts into that ring list, after which part-vs-hole is
    /// unrecoverable and every conformant GeoArrow consumer reads parts 2..n
    /// as holes of part 1.
    ///
    /// Encoded as the optional `part_offsets` `List<UInt32>` column, and ONLY
    /// when at least one feature in the layer has more than one part — the
    /// column's ABSENCE means every feature is single-part. Purely additive
    /// (an older reader ignores an unknown column), so it needs no manifest
    /// capability. Like [`Self::triangles`] it is dropped at encode time for
    /// non-polygon layers.
    pub polygon_parts: Option<Vec<Vec<u32>>>,
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
        if let Some(parts) = &self.polygon_parts {
            check("polygon_parts", parts.len())?;
            // Ring counts to bound the part starts against — only a polygon
            // layer has them, and only a polygon layer keeps the column.
            let rings_per_feature: Option<Vec<usize>> = match &self.geometry {
                GeometryColumn::Polygon(polys) => Some(polys.iter().map(|f| f.len()).collect()),
                _ => None,
            };
            for (i, starts) in parts.iter().enumerate() {
                // An EMPTY entry is tolerated (it encodes as the single-part
                // `[0]`); a malformed non-empty one is not, because it would
                // silently mis-split the feature on every reader.
                if starts.is_empty() {
                    continue;
                }
                if starts[0] != 0 {
                    return Err(Error::Other(format!(
                        "tile layer '{}': polygon_parts[{}] starts at ring {} — part 0 must \
                         start at ring 0",
                        self.name, i, starts[0]
                    )));
                }
                if !starts.windows(2).all(|w| w[0] < w[1]) {
                    return Err(Error::Other(format!(
                        "tile layer '{}': polygon_parts[{}] = {:?} is not strictly increasing",
                        self.name, i, starts
                    )));
                }
                if let Some(rings) = &rings_per_feature {
                    let last = *starts.last().expect("checked non-empty") as usize;
                    if rings[i] > 0 && last >= rings[i] {
                        return Err(Error::Other(format!(
                            "tile layer '{}': polygon_parts[{}] starts a part at ring {} but \
                             the feature only has {} ring(s)",
                            self.name, i, last, rings[i]
                        )));
                    }
                }
            }
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
// Feature-id identity
// ----------------------------------------------------------------------------
//
// The `id` column is `UInt64` on the wire and, for anything the builder had to
// synthesise, it is a HASH — 8 near-incompressible bytes per row. Measured on
// Waymo LiDAR that was ~40 % of a point's compressed bytes, which is why
// `build_point_layer` already substitutes per-tile row indices for unkeyed
// points. Clipped line/polygon layers get no such relief today: their ids must
// survive a tile cut, so they stay hashes.
//
// Both problems are the same problem — an id shape chosen per tile where a
// dataset-global one was wanted — and both are fixed the same way: keep the
// builder's spec-fixed FNV-1a-64 id as the *intermediate source key* and map
// the whole dataset's key set, once, onto a dense `0..k` range in
// first-appearance-in-time order ([`IdRenumberMap`]). Dense + the encoder's
// start-time row sort ⇒ near-monotone ⇒ delta-codable, on every layer kind.
//
// TWO PROPERTIES ARE NON-NEGOTIABLE and every item below exists to hold them:
//
//   * **Per-tile uniqueness.** Picking joins a hit back to a feature through
//     the id, so two rows in one tile must not share one id unless they are
//     genuinely the same source feature.
//   * **Cross-tile identity.** A feature clipped across a tile boundary is ONE
//     feature. Its id must be the same in every tile it touches, or picking and
//     every cross-tile join corrupt silently.
//
// A bijection preserves both by construction: equal source keys map to equal
// dense ids (identity), distinct source keys map to distinct dense ids
// (uniqueness). Nothing here re-derives an id from tile-local data.
//
// WIRING — the state of it, precisely, because "the API exists" and "the API
// runs" are different claims and only the second one moves a byte:
//
//   1. ✅ **LIVE.** `build_point_layer` (stt-build `columnar.rs`) ends with
//      `layer.apply_synthetic_row_ids_for(origin)`, where `origin` comes from
//      [`FeatureIdOrigin::from_synthetic_count`] — `SyntheticRowIndex` only when
//      EVERY feature in the layer lacked a source id, so a MIXED layer stays
//      `Keyed`. That guard is the load-bearing half of the wiring, not a
//      formality: a mixed point layer is the one place today's ids can already
//      collide inside a tile (explicit id `5` vs. row index `5`), and
//      re-densifying across such a layer would overwrite the explicit ids and
//      corrupt identity. No encoder change is required: the layer leaves
//      already sorted, so the encoder's own row sort borrows it unchanged.
//      Kill switch: `ColumnarOptions::synthetic_point_row_ids = false`.
//   2. ⛔ **NOT WIRED YET.** Pass 1 (stt-build `dataset_stats.rs`) folds an
//      [`IdRenumberBuilder`] over the feature stream — `observe(source_key,
//      start_time)`, with `merge` for the parallel chunks — and seals it once;
//      pass 2 resolves each id through [`IdRenumberMap::dense`] at
//      `determine_feature_id` / `segment_feature_id`, or renumbers whole layers
//      with [`IdRenumberMap::renumber_layer_ids`]. The writer stores
//      [`IdRenumberMap::to_sidecar_bytes`] as one content-addressed object so
//      consumers can join a picked dense id back to its source id. Until that
//      lands, clipped line/polygon layers keep paying 8 near-incompressible
//      bytes per row and the types below are exercised only by their tests.
//
// Both are opt-in, so `--single-pass` is simply "do not call them".

/// What a layer's [`ColumnarLayer::feature_ids`] column *means* — the contract
/// a renumbering or re-densifying pass has to honor.
///
/// This is a description of ids the producer already minted, not a request:
/// the producer knows which of the two it built, and only the producer can say.
/// Defaults to the conservative [`Keyed`](Self::Keyed) reading, so a caller
/// that says nothing gets today's behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FeatureIdOrigin {
    /// **Identity-bearing.** The id derives from the SOURCE feature — an
    /// explicit source id taken verbatim, or the builder's spec-fixed
    /// FNV-1a-64 hash of the source's stable fields. A feature clipped across
    /// a tile boundary carries the same id in every tile it touches, and
    /// picking and cross-tile joins depend on that.
    ///
    /// These ids may be replaced only through a dataset-global BIJECTION
    /// ([`IdRenumberMap`]) — never by a per-tile rewrite, which would decode
    /// one source feature as several distinct features.
    #[default]
    Keyed,
    /// **Per-tile synthetic row indices.** *Every* id in the layer was minted
    /// by the producer as the feature's position within this tile, because the
    /// geometry kind cannot be clipped (points) and the source carried no id
    /// worth preserving. There is no cross-tile contract to break, so the
    /// column may be re-densified freely — see
    /// [`ColumnarLayer::apply_synthetic_row_ids`].
    ///
    /// ⚠️ "Every" is load-bearing. `build_point_layer` substitutes row indices
    /// only for the features whose source id is absent, so a point layer that
    /// mixes explicit source ids with synthetic ones is **`Keyed`** — a
    /// re-densify would overwrite the explicit ids. A mixed layer is also the
    /// one place today's ids can collide inside a tile (explicit id `5` vs.
    /// row index `5`); [`ColumnarLayer::feature_ids_are_unique`] detects it.
    SyntheticRowIndex,
}

impl FeatureIdOrigin {
    /// Classify a layer's id column from how many of its ids the producer
    /// actually synthesised — **the guard**, in one place, so no producer has
    /// to re-derive the rule and get the "every" wrong.
    ///
    /// [`SyntheticRowIndex`](Self::SyntheticRowIndex) iff `synthetic == total`
    /// and the layer is non-empty; anything else — one explicit source id among
    /// a thousand row indices — is [`Keyed`](Self::Keyed), because the guarded
    /// entry points must not touch that one explicit id.
    ///
    /// The empty layer answers `Keyed`: both readings are vacuous (there is no
    /// id to rewrite), and the conservative one is the right default to hand a
    /// caller that then adds a row.
    ///
    /// This is a total function of two counts, so it cannot depend on iteration
    /// order, and the miscount that would be dangerous — claiming
    /// `SyntheticRowIndex` for a mixed layer — is unreachable by construction:
    /// `synthetic` is only ever the count of ids the producer overwrote.
    pub fn from_synthetic_count(synthetic: usize, total: usize) -> Self {
        if total > 0 && synthetic == total {
            FeatureIdOrigin::SyntheticRowIndex
        } else {
            FeatureIdOrigin::Keyed
        }
    }
}

impl ColumnarLayer {
    /// Whether every feature in this layer has a distinct id — the picking
    /// invariant, checked rather than assumed.
    ///
    /// Note this is a *per-tile* property. Two rows sharing an id is legal
    /// only when they are genuinely the same clipped source feature re-entering
    /// the tile; a producer that cannot rule that out should treat `false` as a
    /// defect.
    pub fn feature_ids_are_unique(&self) -> bool {
        self.duplicate_feature_ids().is_empty()
    }

    /// The ids that appear more than once, ascending and deduplicated — the
    /// diagnostic half of [`Self::feature_ids_are_unique`]. Sorted output keeps
    /// any message built from it deterministic.
    pub fn duplicate_feature_ids(&self) -> Vec<u64> {
        let mut sorted = self.feature_ids.clone();
        sorted.sort_unstable();
        let mut dupes: Vec<u64> = Vec::new();
        for w in sorted.windows(2) {
            if w[0] == w[1] && dupes.last() != Some(&w[0]) {
                dupes.push(w[0]);
            }
        }
        dupes
    }

    /// Whether `feature_ids` is exactly the dense stored-order sequence
    /// `0, 1, .., n-1` — the shape [`Self::densify_row_ids`] produces and the
    /// cheapest possible id column.
    pub fn feature_ids_are_dense_row_order(&self) -> bool {
        self.feature_ids
            .iter()
            .enumerate()
            .all(|(i, &id)| id == i as u64)
    }

    /// Whether `feature_ids` is *some* permutation of `0..n` — the shape a
    /// synthetic-row-index layer has after the encoder's start-time sort has
    /// shuffled it.
    ///
    /// ⚠️ **Not a substitute for [`FeatureIdOrigin`].** A genuinely keyed layer
    /// whose ids happen to be `0..n` in this tile also answers `true`, and
    /// re-densifying it would silently renumber identity-bearing ids. Use this
    /// for assertions and diagnostics; use the producer's declared origin to
    /// decide.
    pub fn feature_ids_are_row_index_permutation(&self) -> bool {
        let mut seen = vec![false; self.feature_ids.len()];
        for &id in &self.feature_ids {
            let Ok(i) = usize::try_from(id) else {
                return false;
            };
            match seen.get_mut(i) {
                Some(slot) if !*slot => *slot = true,
                _ => return false,
            }
        }
        true
    }

    /// Permute rows into the encoder's canonical STORED order: a stable sort by
    /// `start_time`, exactly the one the layer-frame encoder
    /// ([`encode_tile_with`](crate::arrow_tile::encode_tile_with)) applies —
    /// this calls the encoder's own routine, so the two can never drift. Ties
    /// keep the caller's order, which makes the permutation total and
    /// deterministic.
    ///
    /// A no-op when the rows are already non-decreasing, hence idempotent — and
    /// hence *free* for a producer that is already time-ordered.
    ///
    /// Why a producer would want it: the encoder sorts AFTER the producer has
    /// assigned ids, so any ordering property the producer built into the id
    /// column is destroyed in transit. Sorting first and numbering second keeps
    /// it. See [`Self::apply_synthetic_row_ids`].
    ///
    /// [`encode_layer_v2_parts`]: crate::arrow_tile::encode_tile_with
    pub fn sort_rows_by_start_time_in_place(&mut self) {
        if let Cow::Owned(sorted) = super::encode::sort_rows_by_start_time(self) {
            *self = sorted;
        }
    }

    /// Rewrite `feature_ids` to the dense stored-order sequence `0..n`.
    ///
    /// ⚠️ **Destroys identity.** Only legal on a layer whose every id is a
    /// [`FeatureIdOrigin::SyntheticRowIndex`]; prefer the guarded
    /// [`Self::densify_row_ids_for`], which will not fire on a keyed layer.
    pub fn densify_row_ids(&mut self) {
        for (i, id) in self.feature_ids.iter_mut().enumerate() {
            *id = i as u64;
        }
    }

    /// [`Self::densify_row_ids`], gated on the producer's declared origin.
    /// Returns whether it fired. A [`FeatureIdOrigin::Keyed`] layer is left
    /// exactly as it was — that is the whole point of the gate.
    pub fn densify_row_ids_for(&mut self, origin: FeatureIdOrigin) -> bool {
        if origin != FeatureIdOrigin::SyntheticRowIndex {
            return false;
        }
        self.densify_row_ids();
        true
    }

    /// The complete *ids-after-sort* fix, for a producer that mints per-tile
    /// synthetic row ids: sort rows into the encoder's stored order FIRST, then
    /// number them `0..n`.
    ///
    /// Today the producer numbers first and the encoder sorts second, so the
    /// stored id column is a *shuffled* `0..n` — dense, but not monotone, and a
    /// shuffled permutation delta-codes no better than noise. Numbering after
    /// the sort makes the stored column strictly increasing by 1, which is the
    /// cheapest column an integer codec can be handed.
    ///
    /// Equivalent, byte for byte, to numbering first and re-densifying inside
    /// the encoder after its sort (the two orders produce the same row
    /// permutation and the same ids), and it needs no encoder change: because
    /// the layer leaves here already non-decreasing in `start_time`, the
    /// encoder's own sort is a no-op that borrows the layer unchanged.
    ///
    /// ⚠️ Same precondition as [`Self::densify_row_ids`] — every id in the
    /// layer must be synthetic. See [`FeatureIdOrigin::SyntheticRowIndex`].
    pub fn apply_synthetic_row_ids(&mut self) {
        self.sort_rows_by_start_time_in_place();
        self.densify_row_ids();
    }

    /// [`Self::apply_synthetic_row_ids`], gated on the producer's declared
    /// origin. Returns whether it fired.
    pub fn apply_synthetic_row_ids_for(&mut self, origin: FeatureIdOrigin) -> bool {
        if origin != FeatureIdOrigin::SyntheticRowIndex {
            return false;
        }
        self.apply_synthetic_row_ids();
        true
    }

    /// Observe every `(feature_id, start_time)` pair in this layer into an
    /// [`IdRenumberBuilder`] — the tiler-side alternative to observing the
    /// parsed feature stream, for a pass that already has layers in hand.
    ///
    /// Only meaningful while `feature_ids` still holds SOURCE KEYS (i.e. before
    /// any renumbering, and never on a synthetic-row-index layer).
    pub fn observe_feature_ids(&self, builder: &mut IdRenumberBuilder) {
        for (i, &key) in self.feature_ids.iter().enumerate() {
            let t = self.start_times.get(i).copied().unwrap_or(i64::MAX);
            builder.observe(key, t);
        }
    }
}

/// Accumulator for [`IdRenumberMap`] — the pass-1 side of the global dense
/// renumbering.
///
/// Feed it every source feature's `(source key, start time)`. The source key is
/// exactly the u64 the builder resolves today: an explicit source id verbatim,
/// or the spec-fixed FNV-1a-64 hash of the source's stable fields. **This type
/// never hashes anything** — FNV stays where it is, upstream, unchanged, and
/// keys travel through here opaquely. That is deliberate: the hash is a pinned
/// part of the format's identity story, so it is kept as the intermediate key
/// rather than replaced.
///
/// Determinism: [`observe`](Self::observe) keeps the MINIMUM start time seen
/// per key and [`merge`](Self::merge) takes the pointwise minimum, so the
/// accumulator is associative *and* commutative — a parallel fold over feature
/// chunks lands on the same map as a serial scan, whatever the worker count or
/// chunk boundaries. The internal map is a [`BTreeMap`], so nothing in the
/// output path ever depends on hash iteration order.
#[derive(Debug, Clone, Default)]
pub struct IdRenumberBuilder {
    /// source key → earliest start time observed for it.
    first_seen: BTreeMap<u64, i64>,
}

impl IdRenumberBuilder {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one appearance of `source_key` at `start_time`. Idempotent for a
    /// repeat of the same pair, and order-independent: only the minimum time
    /// per key survives.
    pub fn observe(&mut self, source_key: u64, start_time: i64) {
        self.first_seen
            .entry(source_key)
            .and_modify(|t| {
                if start_time < *t {
                    *t = start_time;
                }
            })
            .or_insert(start_time);
    }

    /// Fold another accumulator in. Associative and commutative — the merge a
    /// parallel pass-1 scan needs.
    pub fn merge(&mut self, other: &Self) {
        for (&key, &t) in &other.first_seen {
            self.observe(key, t);
        }
    }

    /// Distinct source keys observed so far.
    pub fn len(&self) -> usize {
        self.first_seen.len()
    }

    /// Whether nothing has been observed.
    pub fn is_empty(&self) -> bool {
        self.first_seen.is_empty()
    }

    /// Freeze the accumulator into the bijection.
    ///
    /// Dense ids are assigned in **first-appearance-in-time order**, ties broken
    /// by the source key — a total order over a set, so the result is a pure
    /// function of the observed `(key, min time)` set and nothing else.
    ///
    /// First-appearance-in-time (rather than, say, key order) is what buys the
    /// win: the encoder stores rows sorted by `start_time`, so ids assigned in
    /// time order come out of the sort near-monotone within every tile.
    pub fn seal(self) -> IdRenumberMap {
        // BTreeMap iteration is key-ascending, so this vec is already in the
        // tiebreak order; the sort below only needs to be total, not stable.
        let mut ranked: Vec<(i64, u64)> =
            self.first_seen.into_iter().map(|(k, t)| (t, k)).collect();
        ranked.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let source_by_dense: Vec<u64> = ranked.into_iter().map(|(_, k)| k).collect();
        IdRenumberMap::from_source_order(source_by_dense)
    }
}

/// The dataset-global dense id bijection: source key ↔ dense id in `0..k`.
///
/// Built once in pass 1 ([`IdRenumberBuilder`]) and applied wherever the
/// builder resolves a feature id, so:
///
/// * every id column becomes dense and — after the encoder's start-time row
///   sort — near-monotone, i.e. delta-codable, **including** the clipped
///   line/polygon layers that today always pay 8 near-incompressible bytes per
///   row;
/// * **cross-tile identity holds**: the clipped pieces of one source feature
///   share one source key, so they share one dense id;
/// * **per-tile uniqueness holds**: distinct source keys get distinct dense
///   ids, so no tile gains a collision it did not already have.
///
/// Source fidelity is not lost: [`source_ids`](Self::source_ids) /
/// [`to_sidecar_bytes`](Self::to_sidecar_bytes) is the reverse map a consumer
/// joins a picked dense id back through to the original source id. The wire
/// `id` column stays `UInt64`, so no reader decode path changes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IdRenumberMap {
    /// Reverse map: `source_by_dense[dense] = source key`. Also the sidecar
    /// payload, verbatim.
    source_by_dense: Vec<u64>,
    /// Dense ids ordered by their source key — the binary-search index behind
    /// [`Self::dense`]. Held as a permutation rather than a second key copy so
    /// the structure costs 16 B per distinct id, not 24.
    order: Vec<u64>,
}

impl IdRenumberMap {
    /// Build from an explicit dense-order key list (`source_by_dense[d]` is the
    /// source key of dense id `d`). Duplicate keys are silently collapsed to
    /// their FIRST dense id in the forward direction, which cannot happen via
    /// [`IdRenumberBuilder::seal`] (a set) — [`Self::from_sidecar_bytes`]
    /// rejects them outright rather than accept a non-bijection off the wire.
    fn from_source_order(source_by_dense: Vec<u64>) -> Self {
        let mut order: Vec<u64> = (0..source_by_dense.len() as u64).collect();
        order.sort_unstable_by(|&a, &b| {
            source_by_dense[a as usize]
                .cmp(&source_by_dense[b as usize])
                .then_with(|| a.cmp(&b))
        });
        Self {
            source_by_dense,
            order,
        }
    }

    /// Distinct ids in the map.
    pub fn len(&self) -> usize {
        self.source_by_dense.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.source_by_dense.is_empty()
    }

    /// Forward lookup: source key → dense id. `None` means the key was never
    /// observed in pass 1 — data drift between the two passes, which callers
    /// must treat as an error rather than invent an id for.
    pub fn dense(&self, source_key: u64) -> Option<u64> {
        self.order
            .binary_search_by(|&d| self.source_by_dense[d as usize].cmp(&source_key))
            .ok()
            .map(|i| self.order[i])
    }

    /// Whether `source_key` was observed.
    pub fn contains_source(&self, source_key: u64) -> bool {
        self.dense(source_key).is_some()
    }

    /// Reverse lookup: dense id → the source key it replaced. The join a
    /// consumer performs after picking.
    pub fn source_id(&self, dense: u64) -> Option<u64> {
        usize::try_from(dense)
            .ok()
            .and_then(|i| self.source_by_dense.get(i))
            .copied()
    }

    /// The whole reverse map, indexed by dense id.
    pub fn source_ids(&self) -> &[u64] {
        &self.source_by_dense
    }

    /// Whether every dense id fits in 32 bits.
    ///
    /// Worth asserting because the TS reader exposes `featureIds` as the masked
    /// low 32 bits of the `id` column (`id & 0xffffffff`), which is lossy for
    /// the wide ids in use today (H3 cell indices, every Quadbin id). Dense ids
    /// are `0..k`, so any dataset under 2³² features makes that contract exact
    /// — a strict improvement, not a new hazard.
    pub fn fits_u32(&self) -> bool {
        self.source_by_dense.len() <= u32::MAX as usize
    }

    /// Replace a layer's `feature_ids` (still holding SOURCE KEYS) with their
    /// dense ids, in place.
    ///
    /// An unmapped key is a hard error naming the layer and the key: it means
    /// pass 1 and pass 2 disagreed about the feature set, and a silent fallback
    /// there would mint an id that collides with a real one.
    pub fn renumber_layer_ids(&self, layer: &mut ColumnarLayer) -> Result<()> {
        for id in layer.feature_ids.iter_mut() {
            let Some(dense) = self.dense(*id) else {
                return Err(Error::Other(format!(
                    "tile layer '{}': source feature id {} is not in the dataset id map \
                     (pass-1/pass-2 feature-set drift)",
                    layer.name, id
                )));
            };
            *id = dense;
        }
        Ok(())
    }

    /// The reverse map as the content-addressable sidecar payload: one
    /// little-endian `u64` per dense id, in dense-id order. A pure function of
    /// the map, so the same dataset always hashes to the same object.
    pub fn to_sidecar_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.source_by_dense.len() * 8);
        for key in &self.source_by_dense {
            out.extend_from_slice(&key.to_le_bytes());
        }
        out
    }

    /// Rebuild the map from [`Self::to_sidecar_bytes`].
    ///
    /// Rejects a ragged payload and a payload whose keys are not distinct — the
    /// latter would not be a bijection, and a consumer joining through it would
    /// silently resolve two features to one source.
    pub fn from_sidecar_bytes(bytes: &[u8]) -> Result<Self> {
        if !bytes.len().is_multiple_of(8) {
            return Err(Error::Other(format!(
                "id map sidecar is {} bytes, not a whole number of u64 entries",
                bytes.len()
            )));
        }
        let source_by_dense: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
            .collect();
        let map = Self::from_source_order(source_by_dense);
        // `order` is sorted by source key, so a duplicate is an adjacent pair.
        if map
            .order
            .windows(2)
            .any(|w| map.source_by_dense[w[0] as usize] == map.source_by_dense[w[1] as usize])
        {
            return Err(Error::Other(
                "id map sidecar repeats a source key — it is not a bijection".into(),
            ));
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    //! Feature-id identity tests.
    //!
    //! The two properties under test everywhere below are the ones a
    //! renumbering can silently destroy: **per-tile uniqueness** (picking joins
    //! a hit back through the id) and **cross-tile identity** (a clipped
    //! feature is one feature in every tile it touches).
    //!
    //! These live here rather than in `arrow_tile::tests` because they are
    //! about the in-memory model this file owns; they reach across to the
    //! encoder only to prove the stored bytes agree.

    use super::*;
    use crate::arrow_tile::{decode_tile, encode_tile_with, EncoderConfig};
    use arrow::array::{Array, Float64Array, UInt64Array};
    use proptest::prelude::*;

    // ------------------------------------------------------------------
    // Fixtures
    // ------------------------------------------------------------------

    /// A point layer whose `speed` property mirrors the row's id, so any
    /// permutation that moved a column without moving its neighbours shows up
    /// as a mismatched pair rather than as a silent reordering.
    fn point_layer(name: &str, ids: &[u64], starts: &[i64]) -> ColumnarLayer {
        assert_eq!(ids.len(), starts.len());
        ColumnarLayer {
            name: name.to_string(),
            feature_ids: ids.to_vec(),
            start_times: starts.to_vec(),
            end_times: starts.iter().map(|t| t + 10).collect(),
            geometry: GeometryColumn::Point(
                ids.iter().map(|&id| [id as f64 * 0.001, 1.0]).collect(),
            ),
            vertex_times: None,
            vertex_values: None,
            vertex_value_matrix: None,
            triangles: None,
            polygon_parts: None,
            properties: vec![(
                "speed".into(),
                PropertyColumn::Numeric(ids.iter().map(|&id| Some(id as f64)).collect()),
            )],
        }
    }

    /// One tile's worth of a clipped LINE layer: every entry is a piece of the
    /// source feature carrying that source key.
    fn clipped_line_layer(name: &str, source_keys: &[u64], starts: &[i64]) -> ColumnarLayer {
        assert_eq!(source_keys.len(), starts.len());
        ColumnarLayer {
            name: name.to_string(),
            feature_ids: source_keys.to_vec(),
            start_times: starts.to_vec(),
            end_times: starts.iter().map(|t| t + 100).collect(),
            geometry: GeometryColumn::LineString(
                source_keys
                    .iter()
                    .map(|_| vec![[0.0, 0.0], [0.1, 0.1]])
                    .collect(),
            ),
            vertex_times: None,
            vertex_values: None,
            vertex_value_matrix: None,
            triangles: None,
            polygon_parts: None,
            properties: vec![],
        }
    }

    fn stored_ids(layer: &ColumnarLayer) -> Vec<u64> {
        let payload = encode_tile_with(std::slice::from_ref(layer), &EncoderConfig::default())
            .expect("layer encodes");
        let decoded = decode_tile(&payload).expect("frame decodes");
        assert_eq!(decoded.len(), 1);
        decoded[0]
            .batch
            .column_by_name("id")
            .expect("every layer carries an id column")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("id is UInt64 on the wire")
            .values()
            .to_vec()
    }

    fn stored_speeds(layer: &ColumnarLayer) -> Vec<f64> {
        let payload = encode_tile_with(std::slice::from_ref(layer), &EncoderConfig::default())
            .expect("layer encodes");
        let decoded = decode_tile(&payload).expect("frame decodes");
        decoded[0]
            .batch
            .column_by_name("speed")
            .expect("speed column")
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("speed is Float64")
            .values()
            .to_vec()
    }

    // ------------------------------------------------------------------
    // Ids-after-sort
    // ------------------------------------------------------------------

    /// The in-place sort IS the encoder's sort: after it, the encoder's own
    /// routine finds nothing to do and hands the layer straight back
    /// (`Cow::Borrowed`). That borrow is what makes numbering-after-sorting a
    /// producer-side change with no encoder change behind it.
    #[test]
    fn sort_rows_by_start_time_in_place_is_the_encoders_own_sort() {
        let mut layer = point_layer("p", &[7, 3, 9, 1], &[400, 100, 300, 200]);
        layer.sort_rows_by_start_time_in_place();

        assert_eq!(layer.start_times, vec![100, 200, 300, 400]);
        assert!(matches!(
            crate::arrow_tile::encode::sort_rows_by_start_time(&layer),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn sort_rows_by_start_time_in_place_is_idempotent() {
        let mut once = point_layer("p", &[7, 3, 9, 1], &[400, 100, 300, 200]);
        once.sort_rows_by_start_time_in_place();
        let mut twice = once.clone();
        twice.sort_rows_by_start_time_in_place();
        assert_eq!(once.feature_ids, twice.feature_ids);
        assert_eq!(once.start_times, twice.start_times);
    }

    /// A tie in `start_time` must resolve the same way every run. The incumbent
    /// rule is "keep the caller's order", which is what makes the permutation
    /// total; nothing here is free to invent a secondary key.
    #[test]
    fn sort_rows_by_start_time_in_place_keeps_caller_order_on_ties() {
        let mut layer = point_layer("p", &[40, 10, 30, 20], &[5, 5, 5, 5]);
        layer.sort_rows_by_start_time_in_place();
        assert_eq!(layer.feature_ids, vec![40, 10, 30, 20]);
    }

    /// Every column travels with its row.
    #[test]
    fn sort_rows_by_start_time_in_place_permutes_every_column_together() {
        let mut layer = point_layer("p", &[7, 3, 9, 1], &[400, 100, 300, 200]);
        layer.sort_rows_by_start_time_in_place();
        assert_eq!(layer.feature_ids, vec![3, 1, 9, 7]);
        let PropertyColumn::Numeric(speed) = &layer.properties[0].1 else {
            panic!("speed is numeric");
        };
        assert_eq!(speed, &vec![Some(3.0), Some(1.0), Some(9.0), Some(7.0)]);
        let GeometryColumn::Point(pts) = &layer.geometry else {
            panic!("point layer");
        };
        assert_eq!(pts[0][0], 3.0 * 0.001);
    }

    /// The headline of half 1: the STORED id column is `0..n`, strictly
    /// increasing by one, not a shuffled permutation of it.
    #[test]
    fn apply_synthetic_row_ids_stores_ids_dense_in_stored_order() {
        // What `build_point_layer` produces today: row indices in the
        // producer's (unsorted) order.
        let mut layer = point_layer("p", &[0, 1, 2, 3, 4], &[500, 100, 400, 200, 300]);

        // Incumbent: number first, let the encoder sort → dense but SHUFFLED.
        let incumbent = stored_ids(&layer);
        assert_eq!(incumbent, vec![1, 3, 4, 2, 0]);
        assert!(
            !incumbent.windows(2).all(|w| w[0] < w[1]),
            "incumbent stored ids are not monotone — that is the defect"
        );

        // Fixed: sort first, number second.
        layer.apply_synthetic_row_ids();
        assert!(layer.feature_ids_are_dense_row_order());
        assert_eq!(stored_ids(&layer), vec![0, 1, 2, 3, 4]);
        // Rows themselves did not move relative to the incumbent encode.
        assert_eq!(layer.start_times, vec![100, 200, 300, 400, 500]);
        assert_eq!(stored_speeds(&layer), vec![1.0, 3.0, 4.0, 2.0, 0.0]);
    }

    /// Equivalence with the alternative wiring (number first, then re-densify
    /// INSIDE the encoder after its sort). Same row order, same ids — so the
    /// producer-side spelling used here is a byte-for-byte stand-in for the
    /// encoder-side one, and neither is free to drift from the other.
    #[test]
    fn apply_synthetic_row_ids_equals_densifying_after_the_encoder_sort() {
        let starts = [500i64, 100, 400, 200, 300];
        let ids: Vec<u64> = (0..starts.len() as u64).collect();

        let mut producer_side = point_layer("p", &ids, &starts);
        producer_side.apply_synthetic_row_ids();

        let numbered_first = point_layer("p", &ids, &starts);
        let mut encoder_side =
            crate::arrow_tile::encode::sort_rows_by_start_time(&numbered_first).into_owned();
        encoder_side.densify_row_ids();

        assert_eq!(producer_side.feature_ids, encoder_side.feature_ids);
        assert_eq!(producer_side.start_times, encoder_side.start_times);
        assert_eq!(
            encode_tile_with(&[producer_side], &EncoderConfig::default()).unwrap(),
            encode_tile_with(&[encoder_side], &EncoderConfig::default()).unwrap(),
        );
    }

    /// The safety gate. A keyed layer's ids ARE identity; the guarded entry
    /// points must refuse to touch them even though the mechanical rewrite
    /// would "work".
    #[test]
    fn densify_is_a_no_op_on_a_keyed_layer() {
        let keyed = point_layer("p", &[900, 100, 700], &[3, 1, 2]);

        let mut a = keyed.clone();
        assert!(!a.densify_row_ids_for(FeatureIdOrigin::Keyed));
        assert_eq!(a.feature_ids, keyed.feature_ids);

        let mut b = keyed.clone();
        assert!(!b.apply_synthetic_row_ids_for(FeatureIdOrigin::Keyed));
        assert_eq!(b.feature_ids, keyed.feature_ids);
        assert_eq!(
            b.start_times, keyed.start_times,
            "rows must not move either"
        );

        // Default origin is the conservative one, so a caller that says nothing
        // gets today's behavior.
        assert_eq!(FeatureIdOrigin::default(), FeatureIdOrigin::Keyed);

        let mut c = keyed.clone();
        assert!(c.apply_synthetic_row_ids_for(FeatureIdOrigin::SyntheticRowIndex));
        assert_eq!(c.feature_ids, vec![0, 1, 2]);
    }

    /// The classifier the producer's guard is made of. "Every" is the whole
    /// contract: one explicit id among any number of synthetic ones drops the
    /// layer back to `Keyed`, because re-densifying would overwrite that one.
    #[test]
    fn from_synthetic_count_only_says_synthetic_when_every_id_is() {
        assert_eq!(
            FeatureIdOrigin::from_synthetic_count(3, 3),
            FeatureIdOrigin::SyntheticRowIndex
        );
        // MIXED, in both directions and at both extremes.
        assert_eq!(
            FeatureIdOrigin::from_synthetic_count(2, 3),
            FeatureIdOrigin::Keyed
        );
        assert_eq!(
            FeatureIdOrigin::from_synthetic_count(1, 1_000),
            FeatureIdOrigin::Keyed
        );
        assert_eq!(
            FeatureIdOrigin::from_synthetic_count(999, 1_000),
            FeatureIdOrigin::Keyed
        );
        // Fully keyed.
        assert_eq!(
            FeatureIdOrigin::from_synthetic_count(0, 3),
            FeatureIdOrigin::Keyed
        );
        // The empty layer takes the conservative reading.
        assert_eq!(
            FeatureIdOrigin::from_synthetic_count(0, 0),
            FeatureIdOrigin::Keyed
        );
    }

    /// The classifier composed with the gate: a mixed layer's ids survive
    /// untouched, and — the reason that matters — an explicit id that already
    /// collides with a synthetic row index is left exactly as it was rather
    /// than being renumbered across.
    #[test]
    fn a_mixed_layer_is_classified_keyed_and_survives_the_gate_intact() {
        // Row 0 carries explicit source id 5; rows 1 and 2 got row indices
        // 1 and 2. Row 2's synthetic id has no collision; a fourth row with
        // explicit id 5 would — see the collision test below.
        let mixed = point_layer("p", &[5, 1, 2], &[30, 10, 20]);
        let origin = FeatureIdOrigin::from_synthetic_count(2, 3);
        assert_eq!(origin, FeatureIdOrigin::Keyed);

        let mut guarded = mixed.clone();
        assert!(!guarded.apply_synthetic_row_ids_for(origin));
        assert_eq!(guarded.feature_ids, mixed.feature_ids);
        assert_eq!(guarded.start_times, mixed.start_times);

        // And the collision case the detector already documents: renumbering
        // across it would silently move BOTH rows onto new ids.
        let colliding = point_layer("p", &[5, 1, 5], &[1, 2, 3]);
        assert_eq!(colliding.duplicate_feature_ids(), vec![5]);
        let mut still = colliding.clone();
        assert!(!still.apply_synthetic_row_ids_for(FeatureIdOrigin::from_synthetic_count(2, 3)));
        assert_eq!(still.feature_ids, colliding.feature_ids);
    }

    /// Determinism: the same layer encodes to the same bytes twice, with the
    /// id lever on.
    #[test]
    fn dense_row_ids_encode_byte_identically_across_runs() {
        let mut layer = point_layer("p", &[0, 1, 2, 3, 4], &[500, 100, 400, 200, 300]);
        layer.apply_synthetic_row_ids();
        let a = encode_tile_with(&[layer.clone()], &EncoderConfig::default()).unwrap();
        let b = encode_tile_with(&[layer], &EncoderConfig::default()).unwrap();
        assert_eq!(a, b);
    }

    /// The reason any of this is worth doing, measured rather than asserted.
    /// Three id columns over the same 4 096 features, zstd-19 as the archive
    /// stores them: the builder's hash ids (what a clipped layer pays today),
    /// the same ids dense-but-shuffled (what half 1's incumbent ordering
    /// leaves), and dense-and-monotone (what numbering after the sort gives).
    #[test]
    fn monotone_ids_compress_where_hash_ids_do_not() {
        fn zstd_len(ids: &[u64]) -> usize {
            let mut raw = Vec::with_capacity(ids.len() * 8);
            for id in ids {
                raw.extend_from_slice(&id.to_le_bytes());
            }
            zstd::stream::encode_all(raw.as_slice(), 19).unwrap().len()
        }
        let n = 4_096u64;
        // Stand-in for the builder's FNV ids: high-entropy and unordered.
        let hashed: Vec<u64> = (0..n)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();
        // Dense but shuffled — a fixed, seed-free permutation (multiply by an
        // odd number mod 2^k is a bijection on 0..n).
        let shuffled: Vec<u64> = (0..n).map(|i| i.wrapping_mul(2_654_435_761) % n).collect();
        let monotone: Vec<u64> = (0..n).collect();

        let (h, s, m) = (zstd_len(&hashed), zstd_len(&shuffled), zstd_len(&monotone));
        assert!(
            m * 4 < h,
            "monotone {m} B vs hashed {h} B — expected a large multiple"
        );
        assert!(
            m < s,
            "monotone {m} B must beat dense-but-shuffled {s} B (the incumbent)"
        );
        assert!(
            s < h,
            "dense-but-shuffled {s} B must still beat hashed {h} B"
        );
    }

    // ------------------------------------------------------------------
    // Per-tile uniqueness detectors
    // ------------------------------------------------------------------

    /// A point layer that MIXES explicit source ids with synthetic row indices
    /// — what `build_point_layer` produces when only some features carry an id
    /// — can already collide inside one tile. The detector names it; the
    /// origin enum documents that such a layer is `Keyed`, never
    /// `SyntheticRowIndex`.
    #[test]
    fn feature_ids_are_unique_flags_the_mixed_point_layer_collision() {
        let mixed = point_layer("p", &[5, 1, 5], &[1, 2, 3]);
        assert!(!mixed.feature_ids_are_unique());
        assert_eq!(mixed.duplicate_feature_ids(), vec![5]);

        let clean = point_layer("p", &[5, 1, 9], &[1, 2, 3]);
        assert!(clean.feature_ids_are_unique());
        assert!(clean.duplicate_feature_ids().is_empty());
    }

    #[test]
    fn duplicate_feature_ids_is_sorted_and_deduplicated() {
        let layer = point_layer("p", &[9, 3, 9, 3, 9, 1], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(layer.duplicate_feature_ids(), vec![3, 9]);
    }

    #[test]
    fn row_index_permutation_detector_is_exact() {
        assert!(point_layer("p", &[2, 0, 1], &[1, 2, 3]).feature_ids_are_row_index_permutation());
        assert!(!point_layer("p", &[1, 2, 3], &[1, 2, 3]).feature_ids_are_row_index_permutation());
        assert!(!point_layer("p", &[0, 0, 1], &[1, 2, 3]).feature_ids_are_row_index_permutation());
        assert!(
            point_layer("p", &[], &[]).feature_ids_are_row_index_permutation(),
            "the empty layer is vacuously a permutation"
        );
        assert!(point_layer("p", &[0, 1, 2], &[1, 2, 3]).feature_ids_are_dense_row_order());
        assert!(!point_layer("p", &[2, 0, 1], &[1, 2, 3]).feature_ids_are_dense_row_order());
    }

    // ------------------------------------------------------------------
    // The global renumber map
    // ------------------------------------------------------------------

    fn seal(pairs: &[(u64, i64)]) -> IdRenumberMap {
        let mut b = IdRenumberBuilder::new();
        for &(key, t) in pairs {
            b.observe(key, t);
        }
        b.seal()
    }

    #[test]
    fn dense_ids_follow_first_appearance_in_time() {
        // Keys deliberately anti-correlated with time so key order cannot be
        // mistaken for time order.
        let map = seal(&[(900, 300), (100, 100), (500, 200)]);
        assert_eq!(map.dense(100), Some(0));
        assert_eq!(map.dense(500), Some(1));
        assert_eq!(map.dense(900), Some(2));
        assert_eq!(map.source_ids(), &[100, 500, 900]);
    }

    /// A key's rank uses its EARLIEST appearance, not its last or its first
    /// observed — otherwise the map would depend on scan order.
    #[test]
    fn first_appearance_is_the_minimum_time_not_the_first_observed() {
        let map = seal(&[(7, 900), (9, 100), (7, 50)]);
        assert_eq!(map.dense(7), Some(0), "7 was first seen at t=50");
        assert_eq!(map.dense(9), Some(1));
    }

    #[test]
    fn time_ties_break_on_the_source_key() {
        let map = seal(&[(30, 5), (10, 5), (20, 5)]);
        assert_eq!(map.source_ids(), &[10, 20, 30]);
    }

    /// Determinism at the map level: observation order cannot reach the output.
    #[test]
    fn renumber_map_is_independent_of_observation_order() {
        let forward = seal(&[(900, 300), (100, 100), (500, 200), (100, 700)]);
        let backward = seal(&[(100, 700), (500, 200), (100, 100), (900, 300)]);
        assert_eq!(forward, backward);
        assert_eq!(forward.to_sidecar_bytes(), backward.to_sidecar_bytes());
    }

    /// Merge is commutative and associative, so a parallel pass-1 fold over
    /// feature chunks lands where a serial scan does — whatever the worker
    /// count or the chunk boundaries.
    #[test]
    fn builder_merge_is_commutative_and_associative() {
        let chunk = |pairs: &[(u64, i64)]| {
            let mut b = IdRenumberBuilder::new();
            for &(k, t) in pairs {
                b.observe(k, t);
            }
            b
        };
        let a = chunk(&[(900, 300), (100, 400)]);
        let b = chunk(&[(100, 100), (500, 200)]);
        let c = chunk(&[(500, 900), (700, 50)]);

        let mut serial = IdRenumberBuilder::new();
        for &(k, t) in &[
            (900, 300),
            (100, 400),
            (100, 100),
            (500, 200),
            (500, 900),
            (700, 50),
        ] {
            serial.observe(k, t);
        }

        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);

        let mut right = c.clone();
        right.merge(&b);
        right.merge(&a);

        let mut nested = b.clone();
        nested.merge(&c);
        let mut outer = a.clone();
        outer.merge(&nested);

        assert_eq!(left.clone().seal(), serial.seal());
        assert_eq!(left.clone().seal(), right.seal());
        assert_eq!(left.seal(), outer.seal());
    }

    /// **Cross-tile identity.** A source line clipped into three tiles keeps
    /// ONE dense id — the property picking and every cross-tile join stand on.
    #[test]
    fn clipped_pieces_of_one_source_feature_share_one_dense_id() {
        const A: u64 = 0xdead_beef_dead_beef; // source key of the clipped line
        const B: u64 = 0x0123_4567_89ab_cdef; // a second, unrelated feature

        let map = seal(&[(A, 1_000), (A, 1_000), (A, 1_000), (B, 2_000)]);

        let mut west = clipped_line_layer("tracks", &[A, B], &[1_000, 2_000]);
        let mut middle = clipped_line_layer("tracks", &[A], &[1_000]);
        let mut east = clipped_line_layer("tracks", &[A], &[1_000]);
        for tile in [&mut west, &mut middle, &mut east] {
            map.renumber_layer_ids(tile).unwrap();
        }

        let dense_a = map.dense(A).unwrap();
        assert_eq!(west.feature_ids[0], dense_a);
        assert_eq!(middle.feature_ids[0], dense_a);
        assert_eq!(east.feature_ids[0], dense_a);
        assert_ne!(west.feature_ids[1], dense_a, "B must stay a distinct id");

        // And it survives the encoder, which is where a reordering could still
        // have separated them.
        assert_eq!(stored_ids(&middle), vec![dense_a]);
    }

    /// **Per-tile uniqueness.** Distinct source keys stay distinct, and — the
    /// other half of the same coin — a tile that already had a repeated source
    /// key keeps exactly that repeat, neither gaining nor losing one.
    #[test]
    fn renumbering_preserves_per_tile_uniqueness_and_collisions_alike() {
        let map = seal(&[(0xaa, 1), (0xbb, 2), (0xcc, 3)]);

        let mut distinct = clipped_line_layer("t", &[0xcc, 0xaa, 0xbb], &[3, 1, 2]);
        map.renumber_layer_ids(&mut distinct).unwrap();
        assert!(distinct.feature_ids_are_unique());
        assert_eq!(distinct.feature_ids, vec![2, 0, 1]);

        // One source feature re-entering the same tile: two rows, one identity.
        // That is what the format already means, and renumbering must not
        // "repair" it into two features.
        let mut reentrant = clipped_line_layer("t", &[0xaa, 0xbb, 0xaa], &[1, 2, 1]);
        map.renumber_layer_ids(&mut reentrant).unwrap();
        assert_eq!(reentrant.feature_ids, vec![0, 1, 0]);
        assert_eq!(reentrant.duplicate_feature_ids(), vec![0]);
    }

    #[test]
    fn renumber_map_is_a_bijection() {
        let map = seal(&[(900, 300), (100, 100), (500, 200), (700, 250)]);
        assert_eq!(map.len(), 4);
        for dense in 0..map.len() as u64 {
            let source = map.source_id(dense).expect("every dense id resolves");
            assert_eq!(map.dense(source), Some(dense));
        }
        let mut dense: Vec<u64> = map
            .source_ids()
            .iter()
            .map(|&k| map.dense(k).unwrap())
            .collect();
        dense.sort_unstable();
        assert_eq!(dense, vec![0, 1, 2, 3], "dense ids are exactly 0..k");
        assert_eq!(map.source_id(4), None);
        assert_eq!(map.dense(12_345), None);
        assert!(!map.contains_source(12_345));
        assert!(map.contains_source(700));
    }

    /// Pass-1/pass-2 drift is an error, never a silently invented id — an
    /// invented id would collide with a real one and corrupt picking.
    #[test]
    fn renumber_layer_ids_rejects_an_unmapped_source_key() {
        let map = seal(&[(1, 10), (2, 20)]);
        let mut layer = clipped_line_layer("tracks", &[1, 999], &[10, 30]);
        let err = map.renumber_layer_ids(&mut layer).unwrap_err().to_string();
        assert!(err.contains("999"), "error names the offending key: {err}");
        assert!(err.contains("tracks"), "error names the layer: {err}");
    }

    /// **FNV guard.** This module hashes nothing: source keys — including the
    /// builder's spec-fixed FNV-1a-64 values — travel through opaquely and come
    /// back out of the reverse map bit for bit. FNV stays the intermediate key
    /// it has always been; the dense id is a second, separate layer on top.
    #[test]
    fn source_keys_travel_through_opaquely() {
        // A spec-fixed FNV-1a-64 value, computed by the builder, pasted here as
        // an opaque constant: nothing in stt-core may re-derive or alter it.
        const FNV_OF_SOMETHING: u64 = 0xcbf2_9ce4_8422_2325;
        const NEIGHBOUR: u64 = FNV_OF_SOMETHING ^ 1;
        let map = seal(&[(FNV_OF_SOMETHING, 5), (NEIGHBOUR, 6), (u64::MAX, 7), (0, 8)]);

        assert_eq!(
            map.source_ids(),
            &[FNV_OF_SOMETHING, NEIGHBOUR, u64::MAX, 0]
        );
        assert_eq!(map.source_id(map.dense(u64::MAX).unwrap()), Some(u64::MAX));
        assert_eq!(map.source_id(map.dense(0).unwrap()), Some(0));
    }

    /// Dense ids are `0..k`, so the TS reader's masked-low-32-bits
    /// `featureIds` view (`id & 0xffffffff`) becomes exact for any dataset
    /// under 2³² features — a strict improvement on the wide ids in use today.
    #[test]
    fn dense_ids_fit_the_masked_low_32_bit_contract() {
        let map = seal(&[(u64::MAX, 1), (u64::MAX - 1, 2)]);
        assert!(map.fits_u32());
        for &key in map.source_ids() {
            let dense = map.dense(key).unwrap();
            assert_eq!(dense & 0xffff_ffff, dense, "dense id survives the mask");
        }
    }

    // ------------------------------------------------------------------
    // The reverse-map sidecar
    // ------------------------------------------------------------------

    /// The end-to-end join a consumer performs: pick a feature out of a decoded
    /// tile, take its dense id, and recover the source id through the sidecar.
    #[test]
    fn a_picked_dense_id_joins_back_to_its_source_id_through_the_sidecar() {
        const SOURCE: u64 = 0x0f0f_0f0f_dead_0001;
        let map = seal(&[(SOURCE, 100), (SOURCE ^ 0xff, 200)]);

        let mut tile = clipped_line_layer("tracks", &[SOURCE], &[100]);
        map.renumber_layer_ids(&mut tile).unwrap();

        let picked = stored_ids(&tile)[0];
        let sidecar = IdRenumberMap::from_sidecar_bytes(&map.to_sidecar_bytes()).unwrap();
        assert_eq!(sidecar.source_id(picked), Some(SOURCE));
        assert_eq!(sidecar, map, "the sidecar rebuilds the map exactly");
    }

    #[test]
    fn sidecar_bytes_are_deterministic_and_dense_ordered() {
        let map = seal(&[(900, 300), (100, 100), (500, 200)]);
        let bytes = map.to_sidecar_bytes();
        assert_eq!(bytes, map.to_sidecar_bytes());
        assert_eq!(bytes.len(), 3 * 8);
        assert_eq!(&bytes[0..8], &100u64.to_le_bytes());
        assert_eq!(&bytes[8..16], &500u64.to_le_bytes());
        assert_eq!(&bytes[16..24], &900u64.to_le_bytes());
        assert_eq!(
            IdRenumberMap::from_sidecar_bytes(&[]).unwrap(),
            IdRenumberMap::default()
        );
    }

    #[test]
    fn sidecar_rejects_a_ragged_payload() {
        let err = IdRenumberMap::from_sidecar_bytes(&[0u8; 9])
            .unwrap_err()
            .to_string();
        assert!(err.contains("9 bytes"), "{err}");
    }

    /// A repeated source key is not a bijection: joining through it would
    /// resolve two distinct features to one source.
    #[test]
    fn sidecar_rejects_a_repeated_source_key() {
        let mut bytes = 42u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice(&42u64.to_le_bytes());
        let err = IdRenumberMap::from_sidecar_bytes(&bytes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("bijection"), "{err}");
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    proptest! {
        /// The map is a bijection onto `0..k` and is a pure function of the
        /// observed `(key, min time)` SET — shuffling the observations, or
        /// splitting them across chunks that merge, cannot move a single id.
        #[test]
        fn prop_renumber_map_is_order_independent_and_bijective(
            mut pairs in prop::collection::vec((any::<u64>(), -1_000i64..1_000), 1..64),
            cut in 0usize..64,
        ) {
            let mut serial = IdRenumberBuilder::new();
            for &(k, t) in &pairs {
                serial.observe(k, t);
            }
            let map = serial.seal();

            // Bijection onto exactly 0..k.
            let mut distinct: Vec<u64> = pairs.iter().map(|p| p.0).collect();
            distinct.sort_unstable();
            distinct.dedup();
            prop_assert_eq!(map.len(), distinct.len());
            let mut dense: Vec<u64> = distinct.iter().map(|&k| map.dense(k).unwrap()).collect();
            dense.sort_unstable();
            prop_assert_eq!(dense, (0..distinct.len() as u64).collect::<Vec<_>>());
            for d in 0..map.len() as u64 {
                prop_assert_eq!(map.dense(map.source_id(d).unwrap()), Some(d));
            }

            // Order independence: reverse, then chunk-and-merge.
            pairs.reverse();
            let mut reversed = IdRenumberBuilder::new();
            for &(k, t) in &pairs {
                reversed.observe(k, t);
            }
            prop_assert_eq!(&reversed.seal(), &map);

            let cut = cut.min(pairs.len());
            let mut left = IdRenumberBuilder::new();
            for &(k, t) in &pairs[..cut] {
                left.observe(k, t);
            }
            let mut right = IdRenumberBuilder::new();
            for &(k, t) in &pairs[cut..] {
                right.observe(k, t);
            }
            left.merge(&right);
            prop_assert_eq!(&left.seal(), &map);

            // Sidecar round-trip.
            let rebuilt = IdRenumberMap::from_sidecar_bytes(&map.to_sidecar_bytes()).unwrap();
            prop_assert_eq!(&rebuilt, &map);
        }

        /// Identity survives arbitrary clipping: however a source feature is cut
        /// up and scattered across tiles, every piece carries the same dense id,
        /// and no two source features ever collide onto one.
        #[test]
        fn prop_identity_survives_random_clipping_splits(
            features in prop::collection::vec((any::<u64>(), 0i64..500, 1usize..5), 1..24),
            tiles in 1usize..6,
        ) {
            // Distinct source features, keyed as the builder would key them.
            let mut by_key: BTreeMap<u64, (i64, usize)> = BTreeMap::new();
            for &(key, t, pieces) in &features {
                by_key.entry(key).or_insert((t, pieces));
            }

            // Pass 1: observe every piece (a clipped piece is still one
            // appearance of its source feature).
            let mut builder = IdRenumberBuilder::new();
            for (&key, &(t, pieces)) in &by_key {
                for p in 0..pieces {
                    builder.observe(key, t + p as i64);
                }
            }
            let map = builder.seal();

            // Pass 2: scatter the pieces round-robin across tiles and renumber.
            let mut tile_keys: Vec<Vec<u64>> = vec![Vec::new(); tiles];
            let mut tile_times: Vec<Vec<i64>> = vec![Vec::new(); tiles];
            let mut n = 0usize;
            for (&key, &(t, pieces)) in &by_key {
                for p in 0..pieces {
                    let tile = n % tiles;
                    tile_keys[tile].push(key);
                    tile_times[tile].push(t + p as i64);
                    n += 1;
                }
            }
            let mut seen: BTreeMap<u64, u64> = BTreeMap::new(); // dense -> source
            for (keys, times) in tile_keys.iter().zip(tile_times.iter()) {
                let mut layer = clipped_line_layer("t", keys, times);
                let sources = layer.feature_ids.clone();
                map.renumber_layer_ids(&mut layer).unwrap();
                for (source, &dense) in sources.iter().zip(layer.feature_ids.iter()) {
                    // Cross-tile identity: same source ⇒ same dense id, and
                    // no two sources ever share one.
                    if let Some(previous) = seen.insert(dense, *source) {
                        prop_assert_eq!(previous, *source);
                    }
                }
            }
            // Every distinct source feature got its own dense id.
            prop_assert_eq!(seen.len(), by_key.len());
        }
    }
}
