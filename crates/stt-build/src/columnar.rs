//! Build Arrow [`ColumnarLayer`]s from parsed features and clipped segments.
//!
//! This stage emits geometry as real WGS84 lon/lat (`f64`), unquantized and
//! undelta'd; coordinate quantization, if enabled, happens later at encode
//! time. Nothing here compresses — the encoder writes Arrow IPC and the pack
//! writer applies per-blob zstd (the only tile codec; see
//! `stt_core::compression`). The payload stays consumable directly by
//! GeoArrow-aware renderers.

use crate::clip::ClippedSegment;
use crate::input::ParsedFeature;
use crate::props::{FeatureProperties, PropValue};
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Arc;
use stt_core::arrow_tile::{
    tessellate_polygon, ColumnarLayer, Coord, FeatureIdOrigin, GeometryColumn, PropertyColumn,
};
use stt_core::projection::haversine_distance;
use stt_core::types::GeometryType;

/// Opt-in user-property selection (tippecanoe `--exclude`/`--include`/
/// `--exclude-all` mental model). Applied at the point property columns are
/// materialised, so it governs BOTH point/line/polygon layers and clipped
/// trajectory-segment layers.
///
/// SYSTEM columns (feature id, start/end time, geometry, vertex_time /
/// vertex_value / vertex_value_matrix, triangles) are NOT user properties and
/// are therefore never reachable here — they always survive. This filter only
/// ever touches the `properties` map.
#[derive(Debug, Clone, Default)]
pub enum AttributeFilter {
    /// No filtering — every user property is kept (the default; byte-for-byte
    /// identical to a build with none of the attribute flags set).
    #[default]
    KeepAll,
    /// Drop exactly these property names (`--exclude`).
    Exclude(std::collections::HashSet<String>),
    /// Keep ONLY these property names (`--include`).
    Include(std::collections::HashSet<String>),
    /// Drop every user property — geometry + times only (`--exclude-all`).
    ExcludeAll,
}

impl AttributeFilter {
    /// Should the named user property be emitted?
    pub fn keeps(&self, name: &str) -> bool {
        match self {
            AttributeFilter::KeepAll => true,
            AttributeFilter::Exclude(set) => !set.contains(name),
            AttributeFilter::Include(set) => set.contains(name),
            AttributeFilter::ExcludeAll => false,
        }
    }

    /// True when the filter is the inert default (no property is ever dropped).
    pub fn is_keep_all(&self) -> bool {
        matches!(self, AttributeFilter::KeepAll)
    }
}

/// Authoritative kind of one user property, derived from the input source's
/// schema (Arrow/GeoParquet column type, DB column type) rather than sniffed
/// from values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyKind {
    /// Emitted as a Float64 (or quantized) tile column.
    Numeric,
    /// Emitted as a Dictionary<UInt16, Utf8> tile column.
    Categorical,
}

/// Map property name → authoritative kind. Empty = no schema known
/// (GeoJSON-shaped producers) — per-tile value sniffing applies.
pub type PropertyTypes = BTreeMap<String, PropertyKind>;

/// Per-tile build options that influence the columnar layout (independent of
/// the tile-level partitioning logic the tiler owns).
///
/// ⚠️ [`Default`] is hand-written rather than derived because
/// [`Self::synthetic_point_row_ids`] defaults to **true** — see its docs.
#[derive(Debug, Clone)]
pub struct ColumnarOptions {
    /// When true, polygon layers will carry pre-baked earcut triangle indices
    /// in a `triangles` sidecar column — letting the renderer skip its own
    /// CPU-side tessellation on tile arrival (MLT-style).
    pub pre_tessellate: bool,
    /// Opt-in user-property selection. Default [`AttributeFilter::KeepAll`] —
    /// inert unless `--exclude`/`--include`/`--exclude-all` is passed.
    pub attribute_filter: AttributeFilter,
    /// Authoritative per-property kinds from the input source's schema. A
    /// listed key produces a column of the declared kind in EVERY tile — even
    /// a tile where all its values happen to be null. Without this, per-tile
    /// value sniffing silently DROPS an all-null-in-tile column (and can
    /// reclassify a column tile-by-tile), drifting the layer schema across
    /// tiles. Keys not listed keep the sniffing behaviour, which schema-less
    /// producers rely on.
    pub property_types: Arc<PropertyTypes>,
    /// **Ids-after-sort** (TB-5 half 1). When true — the default — a point
    /// layer whose every id `build_point_layer` had to synthesise is sorted
    /// into the encoder's stored order FIRST and numbered `0..n` SECOND, so the
    /// stored `id` column is strictly increasing by one instead of a shuffled
    /// permutation of `0..n`.
    ///
    /// The incumbent numbers first and lets the encoder's start-time sort
    /// shuffle the result; a shuffled dense column delta-codes no better than
    /// noise, which throws away most of the win the row-index substitution was
    /// introduced for. Numbering after the sort is byte-equivalent to
    /// re-densifying inside the encoder (proved in
    /// `stt_core::arrow_tile::layer`'s
    /// `apply_synthetic_row_ids_equals_densifying_after_the_encoder_sort`) and
    /// needs no encoder change.
    ///
    /// **Only ever fires on a layer whose ids are ALL synthetic.** A point
    /// layer mixing explicit source ids with synthetic row indices classifies
    /// [`FeatureIdOrigin::Keyed`] and is left untouched — see
    /// [`FeatureIdOrigin::from_synthetic_count`].
    ///
    /// BYTE-CHANGING when true. Set false for the documented `--single-pass`
    /// rollback: the id column reverts to the incumbent shuffled-dense shape,
    /// byte-for-byte.
    pub synthetic_point_row_ids: bool,
    /// **TB-12 per-feature triangle emission.** When true — the default — a
    /// polygon layer bakes triangle indices only for the features a renderer's
    /// own single-boundary earcut cannot reproduce (holes, multi-part) and
    /// leaves every other list EMPTY for the decoder to backfill.
    ///
    /// BYTE-CHANGING when true. Set false for the rollback: the triangle column
    /// reverts to the incumbent all-or-nothing shape, byte-for-byte, and no
    /// capability is owed.
    ///
    /// `stt-serve` sets it false by DEFAULT and gates it behind an explicit
    /// flag, for the same reason it keeps `--compact-times` opt-in: a packed
    /// archive lets an old reader refuse at open, but a served tile gives it no
    /// such chance, so a re-typing default there would silently drop geometry
    /// instead of failing loudly.
    pub partial_triangles: bool,
    /// TB-12 OBSERVATION CHANNEL (out-parameter, not a setting). Set to `true`
    /// by [`build_polygon_layer`] the first time a layer actually MIXES empty
    /// and baked triangle lists, which is the condition that obliges the
    /// manifest to declare [`stt_core::pack::CAPABILITY_TRIANGLES_PARTIAL`].
    ///
    /// Shared across the tiler's rayon workers, so it is an atomic. The value
    /// is a monotone OR over every layer built — order-independent, hence
    /// deterministic regardless of worker scheduling. `Relaxed` is sufficient:
    /// nothing is published through this flag, and it is read only after the
    /// tiling pool has joined.
    pub partial_triangles_observed: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for ColumnarOptions {
    fn default() -> Self {
        Self {
            pre_tessellate: false,
            attribute_filter: AttributeFilter::default(),
            property_types: Arc::default(),
            // The R1 default: on. See the field's docs for the rollback.
            synthetic_point_row_ids: true,
            // The R1 default: on. See the field's docs for the rollback.
            partial_triangles: true,
            partial_triangles_observed: Arc::default(),
        }
    }
}

/// Build layers from a set of features sharing a tile. Features are grouped by
/// geometry type — a single layer holds exactly one geometry kind, so a tile
/// with mixed points and polygons yields one layer per kind.
///
/// Convenience wrapper for callers that don't care about extra build knobs.
pub fn build_layers_from_features(
    features: &[&ParsedFeature],
    layer_name: &str,
) -> Result<Vec<ColumnarLayer>> {
    build_layers_from_features_with(features, layer_name, ColumnarOptions::default())
}

/// Build layers from features with explicit build options.
pub fn build_layers_from_features_with(
    features: &[&ParsedFeature],
    layer_name: &str,
    opts: ColumnarOptions,
) -> Result<Vec<ColumnarLayer>> {
    if features.is_empty() {
        return Ok(vec![]);
    }

    // Partition by geometry type, preserving input order within each group.
    let mut points: Vec<&ParsedFeature> = Vec::new();
    let mut lines: Vec<&ParsedFeature> = Vec::new();
    let mut polygons: Vec<&ParsedFeature> = Vec::new();
    for f in features {
        match determine_geometry_type(f) {
            Ok(GeometryType::Point) => points.push(f),
            Ok(GeometryType::LineString) => lines.push(f),
            Ok(GeometryType::Polygon) => polygons.push(f),
            Err(error) => {
                anyhow::bail!("layer {layer_name:?}: cannot classify feature geometry: {error}")
            }
        }
    }

    let mut layers = Vec::new();
    // When a tile has multiple kinds, suffix the layer name so a reader can
    // tell them apart; the dominant kind keeps the bare name.
    let kinds_present = [!points.is_empty(), !lines.is_empty(), !polygons.is_empty()]
        .iter()
        .filter(|p| **p)
        .count();
    let name_for = |kind: &str| -> String {
        if kinds_present <= 1 {
            layer_name.to_string()
        } else {
            format!("{layer_name}_{kind}")
        }
    };

    if !points.is_empty() {
        layers.push(build_point_layer(&points, name_for("points"), &opts)?);
    }
    if !lines.is_empty() {
        layers.push(build_line_layer(&lines, name_for("lines"), &opts)?);
    }
    if !polygons.is_empty() {
        layers.push(build_polygon_layer(&polygons, name_for("polygons"), &opts)?);
    }
    Ok(layers)
}

/// Build a single linestring layer from clipped trajectory segments. Segments
/// carry real per-vertex timestamps produced by the clipper.
pub fn build_layer_from_segments(
    segments: &[&ClippedSegment],
    layer_name: &str,
    opts: &ColumnarOptions,
) -> Result<ColumnarLayer> {
    let n = segments.len();
    let mut feature_ids = Vec::with_capacity(n);
    let mut start_times = Vec::with_capacity(n);
    let mut end_times = Vec::with_capacity(n);
    let mut geometry: Vec<Vec<Coord>> = Vec::with_capacity(n);
    let mut vertex_times: Vec<Vec<i64>> = Vec::with_capacity(n);
    let mut vertex_values: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut vertex_value_matrix: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut any_values = false;
    let mut any_matrix = false;

    let mut props = PropertyAccumulator::new(
        opts.attribute_filter.clone(),
        Arc::clone(&opts.property_types),
    );

    for seg in segments {
        feature_ids.push(segment_feature_id(seg));
        start_times.push(seg.start_time as i64);
        end_times.push(seg.end_time as i64);

        let coords: Vec<Coord> = seg
            .coordinates
            .iter()
            .map(|(x, y, _alt)| [*x, *y])
            .collect();

        // Per-vertex timestamps: use the segment's real timestamps where
        // present, padding with the start time if the clipper produced fewer.
        let mut times: Vec<i64> = Vec::with_capacity(coords.len());
        for i in 0..coords.len() {
            let t = seg.timestamps.get(i).copied().unwrap_or(seg.start_time);
            times.push(t as i64);
        }

        // Per-vertex scalar values (e.g. SST), aligned with coords. Missing
        // entries become NaN so the column always has one value per vertex.
        if !seg.vertex_values.is_empty() {
            any_values = true;
        }
        let mut vals: Vec<f32> = Vec::with_capacity(coords.len());
        for i in 0..coords.len() {
            vals.push(seg.vertex_values.get(i).copied().unwrap_or(f32::NAN));
        }

        // Per-vertex × per-bucket matrix, flattened vertex-major. Each segment
        // row is `[vertex][bucket]`, aligned 1:1 with `coordinates` by the
        // clipper, so concatenating rows yields the tile's vertex-major layout.
        if !seg.vertex_value_matrix.is_empty() {
            any_matrix = true;
            let nb = seg.vertex_value_matrix[0].len();
            let mut flat = Vec::with_capacity(coords.len() * nb);
            for row in &seg.vertex_value_matrix {
                flat.extend_from_slice(row);
            }
            vertex_value_matrix.push(flat);
        } else {
            vertex_value_matrix.push(Vec::new());
        }

        geometry.push(coords);
        vertex_times.push(times);
        vertex_values.push(vals);

        props.observe(seg.properties.as_ref());
    }
    // Second pass fills one value per feature for every discovered property.
    for seg in segments {
        props.push_row(seg.properties.as_ref());
    }

    Ok(ColumnarLayer {
        polygon_parts: None,
        name: layer_name.to_string(),
        feature_ids,
        start_times,
        end_times,
        geometry: GeometryColumn::LineString(geometry),
        // Matrix corridors are timeless (animated by the matrix, not per-vertex
        // times) — drop the dead per-vertex time column for them; keep it for
        // ordinary trajectory segments that drive the trail animation.
        vertex_times: (!any_matrix).then_some(vertex_times),
        // Only attach per-vertex values if at least one segment carried them.
        vertex_values: any_values.then_some(vertex_values),
        triangles: None,
        vertex_value_matrix: any_matrix.then_some(vertex_value_matrix),
        properties: props.finish(),
    })
}

// ----------------------------------------------------------------------------
// Per-geometry-kind builders
// ----------------------------------------------------------------------------

fn build_point_layer(
    features: &[&ParsedFeature],
    name: String,
    opts: &ColumnarOptions,
) -> Result<ColumnarLayer> {
    let (mut ids, start, end, props) = common_columns(features, opts);
    // Points are never split across tile boundaries, so a point feature needs no
    // globally stable id (unlike a clipped line/polygon, which must keep one id
    // across the tiles it spans). When the source carries no explicit id, the
    // fallback in `determine_feature_id` is a hash of (time, lon, lat): a
    // high-entropy u64 that zstd cannot compress and that — measured on Waymo
    // LiDAR — is the single largest column in the tile (~40% of a point's
    // compressed bytes). Replace those synthetic ids with the per-tile row
    // index: still unique within the tile (picking stays correct) but monotonic,
    // so the column compresses to a few bits per point. Explicit source ids
    // (e.g. earthquake/storm-cell ids a consumer may surface) are preserved.
    let mut synthetic = 0usize;
    for (i, f) in features.iter().enumerate() {
        if f.geojson.id.is_none() {
            ids[i] = i as u64;
            synthetic += 1;
        }
    }
    // THE GUARD. `SyntheticRowIndex` is a claim about EVERY id in the layer,
    // and it is the claim that licenses re-densifying the whole column. This
    // loop only overwrote the ids of features that carried none, so a layer
    // where `synthetic < features.len()` MIXES minted row indices with explicit
    // source ids and must stay `Keyed` — re-densifying it would renumber the
    // explicit ids, which are identity and which a consumer may be joining on.
    let origin = FeatureIdOrigin::from_synthetic_count(synthetic, features.len());
    // A mixed layer is also the one place today's point ids can collide inside
    // a single tile — explicit source id `5` against minted row index `5` — and
    // that collision predates any of this. It is not silently repaired here
    // (the repair would be to renumber, which is exactly what the guard above
    // forbids), but it is no longer silent: two rows sharing an id break the
    // picking join, so a build that produces one should say so.
    if origin == FeatureIdOrigin::Keyed && synthetic > 0 {
        let collisions = mixed_point_id_collisions(features, &ids);
        if !collisions.is_empty() {
            warn_mixed_point_id_collision(&name, &collisions);
        }
    }
    let geometry: Vec<Coord> = features.iter().map(|f| [f.lon, f.lat]).collect();
    let mut layer = ColumnarLayer {
        polygon_parts: None,
        name,
        feature_ids: ids,
        start_times: start,
        end_times: end,
        geometry: GeometryColumn::Point(geometry),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: props,
    };
    // Ids-after-sort (TB-5 half 1). Sorting into the encoder's stored order
    // here and numbering second makes the STORED id column `0, 1, .., n-1`
    // rather than a shuffled permutation of it. Gated on `origin`, so this is a
    // no-op on every layer the guard above classified `Keyed`, and gated again
    // on the option, which is the `--single-pass` rollback. The encoder's own
    // start-time sort then finds nothing to do and borrows the layer unchanged.
    if opts.synthetic_point_row_ids {
        layer.apply_synthetic_row_ids_for(origin);
    }
    Ok(layer)
}

/// The ids a MIXED point layer minted as row indices that collide with an
/// explicit source id in the same layer — the picking hazard
/// [`ColumnarLayer::feature_ids_are_unique`] detects, localized to its one real
/// cause so a build can name it.
///
/// `features` and `ids` are row-aligned, and `ids[i]` is the *post-substitution*
/// id: a row whose feature carried no source id holds `i`. Ascending and
/// deduplicated, so any message built from it is deterministic.
///
/// Costs one `Vec<bool>` of the layer's length and a linear scan, and the caller
/// only runs it on a mixed layer — the rare shape — so the common all-synthetic
/// and all-keyed layers pay nothing.
fn mixed_point_id_collisions(features: &[&ParsedFeature], ids: &[u64]) -> Vec<u64> {
    let n = ids.len();
    let mut minted = vec![false; n];
    for (i, f) in features.iter().enumerate() {
        if f.geojson.id.is_none() {
            minted[i] = true;
        }
    }
    let mut hits: Vec<u64> = Vec::new();
    for (i, f) in features.iter().enumerate() {
        if f.geojson.id.is_some() {
            let id = ids[i];
            // An explicit id only collides if it lands inside `0..n` AND that
            // row index was actually minted for some other row.
            if let Ok(slot) = usize::try_from(id) {
                if minted.get(slot).copied().unwrap_or(false) {
                    hits.push(id);
                }
            }
        }
    }
    hits.sort_unstable();
    hits.dedup();
    hits
}

/// Warn once per process about the mixed-point-layer id collision. Once,
/// because the shape is a property of the SOURCE, so every tile of an affected
/// dataset would otherwise repeat it — and a warning nobody can read is a
/// warning nobody acts on. Log volume only; nothing here reaches a tile byte.
fn warn_mixed_point_id_collision(layer: &str, collisions: &[u64]) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "point layer {layer:?}: {} feature id(s) minted as row indices collide with an \
             explicit source id in the same tile (first: {}). Two rows sharing one id break the \
             picking join, and the collision is NOT repaired here — renumbering across it would \
             overwrite the explicit ids. Give every feature an explicit id, or none. (Further \
             collisions in this build are silent.)",
            collisions.len(),
            collisions[0],
        );
    });
}

fn build_line_layer(
    features: &[&ParsedFeature],
    name: String,
    opts: &ColumnarOptions,
) -> Result<ColumnarLayer> {
    // Geometry FIRST, and only then the id/time/property columns: a feature
    // whose geometry cannot be read is DROPPED here (never fabricated), and
    // every other column has to be built over the survivors so the rows stay
    // aligned.
    let (features, coords_per_feature) =
        extract_or_drop(features, "line", &name, extract_line_coords)?;
    let features = &features[..];
    let (ids, start, end, props) = common_columns(features, opts);

    let mut geometry: Vec<Vec<Coord>> = Vec::with_capacity(features.len());
    let mut vertex_times: Vec<Vec<i64>> = Vec::with_capacity(features.len());
    let mut vertex_values: Vec<Vec<f32>> = Vec::with_capacity(features.len());
    let mut vertex_value_matrix: Vec<Vec<f32>> = Vec::with_capacity(features.len());
    let mut any_duration = false;
    let mut any_values = false;
    let mut any_matrix = false;
    let mut length_mismatch_warned = false;

    for (f, coords) in features.iter().zip(coords_per_feature) {
        // Priority for per-vertex times, in order:
        //   1. Producer-supplied `vertex_timestamps` (e.g. OSRM annotations) —
        //      real per-segment timing reflecting street class.
        //   2. Distance-interpolated from start..end when a duration exists.
        //   3. Flat: every vertex shares the feature start time.
        // The supplied path is rejected if its length doesn't match the
        // geometry's vertex count (logged once per build to surface bad
        // producers rather than silently corrupting the timing).
        let times = if let Some(supplied) = f.vertex_timestamps.as_ref() {
            if supplied.len() == coords.len() {
                any_duration = true;
                supplied.iter().map(|&t| t as i64).collect()
            } else {
                if !length_mismatch_warned {
                    tracing::warn!(
                        "vertex_timestamps length {} != coord count {} for a line \
                         feature; falling back to distance interpolation (further \
                         mismatches in this build will be silent)",
                        supplied.len(),
                        coords.len()
                    );
                    length_mismatch_warned = true;
                }
                if let Some(end_ts) = f.end_timestamp {
                    any_duration = true;
                    interpolate_vertex_times(&coords, f.timestamp, end_ts)
                } else {
                    vec![f.timestamp as i64; coords.len()]
                }
            }
        } else if let Some(end_ts) = f.end_timestamp {
            any_duration = true;
            interpolate_vertex_times(&coords, f.timestamp, end_ts)
        } else {
            vec![f.timestamp as i64; coords.len()]
        };
        // Per-vertex scalar values (e.g. SST). Accepted only when the supplied
        // length matches the geometry; otherwise NaN-filled (gray at render).
        let vals: Vec<f32> = match f.vertex_values.as_ref() {
            Some(supplied) if supplied.len() == coords.len() => {
                any_values = true;
                supplied.clone()
            }
            _ => vec![f32::NAN; coords.len()],
        };
        // Per-vertex × per-bucket matrix (flat vertex-major). Accepted only when
        // the length is a clean multiple of the vertex count.
        let matrix: Vec<f32> = match f.vertex_value_matrix.as_ref() {
            Some(m) if !m.is_empty() && m.len() % coords.len() == 0 => {
                any_matrix = true;
                m.clone()
            }
            _ => Vec::new(),
        };

        geometry.push(coords);
        vertex_times.push(times);
        vertex_values.push(vals);
        vertex_value_matrix.push(matrix);
    }

    Ok(ColumnarLayer {
        polygon_parts: None,
        name,
        feature_ids: ids,
        start_times: start,
        end_times: end,
        geometry: GeometryColumn::LineString(geometry),
        // Attach per-vertex times only when a feature has a real duration AND the
        // layer carries no value matrix. A matrix corridor is TIMELESS — its
        // geometry is static and the animation comes from the matrix, not from
        // per-vertex times — so the interpolated times are dead weight no
        // consumer reads (the decoder + trips layers synthesize them when absent).
        // Dropping them keeps flow-corridor / baked-bundle tiles small.
        vertex_times: (any_duration && !any_matrix).then_some(vertex_times),
        // Likewise only attach per-vertex values if a feature supplied them.
        vertex_values: any_values.then_some(vertex_values),
        triangles: None,
        vertex_value_matrix: any_matrix.then_some(vertex_value_matrix),
        properties: props,
    })
}

fn build_polygon_layer(
    features: &[&ParsedFeature],
    name: String,
    opts: &ColumnarOptions,
) -> Result<ColumnarLayer> {
    // Geometry FIRST — see `build_line_layer`. `parts[i].part_starts` is the
    // per-feature PART boundary list (the ring index each part of a
    // MultiPolygon begins at); it drives the tessellator below AND is what the
    // `part_offsets` wire column is built from.
    let (features, parts) = extract_or_drop(features, "polygon", &name, extract_polygon_rings)?;
    let features = &features[..];
    let (ids, start, end, props) = common_columns(features, opts);

    // Build the triangle index sidecar by running earcut over each feature's
    // rings, PART BY PART. The same coords feed both the geometry column and
    // the tessellator — indices are local to the feature.
    //
    // We bake triangles when explicitly requested (`--pre-tessellate`) OR
    // whenever ANY feature has more than one ring (a polygon with holes, a
    // perimeter carrying interior rings, or a multi-part MultiPolygon — which
    // is also what the tiler emits when clipping cuts one polygon into several
    // pieces inside a tile). Such features CANNOT render correctly through
    // deck.gl's binary earcut path: with `_normalize:false` and no index buffer
    // it triangulates the feature's concatenated ring run as a SINGLE boundary,
    // bridging disjoint rings with spanning triangles — the storm-radar isoband
    // streaks and wildfire-perimeter spikes. Supplying the baked, hole-aware
    // `indices` buffer makes the renderer skip earcut, so the sidecar is
    // MANDATORY for these layers, not just a perf win. Layers whose every
    // feature is a single ring stay lean (no sidecar).
    //
    // PER-FEATURE since TB-12. The triangle column is 40-45% of the wire bytes
    // of a polygon-heavy tile (measured on storm-field and storm4d-cloudtop),
    // and most of that is spent on single-ring features every renderer can
    // earcut itself — so we emit an EMPTY list for those and keep baked indices
    // only for the features that genuinely need them.
    //
    // This was UNSAFE before the reader change that now precedes it. All three
    // readers used to branch on whether the LAYER had a triangle column and then
    // trust each feature's slice verbatim, so an empty list meant "draw
    // nothing" and every single-ring polygon vanished. The decoder in
    // poopdeck:packages/core/src/tile.ts now BACKFILLS an empty run by earcutting the
    // feature's single ring, which completes the buffer before any of the three
    // sees it — deck's whole-layer `indices` handoff and three's layer-global
    // `hasPreBaked` switch keep working unchanged.
    //
    // Readers that predate that backfill must not open these archives at all,
    // hence CAPABILITY_TRIANGLES_PARTIAL below: a loud refusal at open instead
    // of silently missing geometry.
    //
    // `--pre-tessellate` keeps the old bake-everything behaviour, for clients
    // that would rather spend the bytes than the decode-time CPU.
    let needs_triangles = opts.pre_tessellate || parts.iter().any(PolygonParts::needs_triangles);
    // Only the features a renderer's own earcut cannot reproduce get baked. A
    // layer where EVERY feature needs baking mixes nothing and so stays
    // byte-identical to the incumbent — and declares no capability.
    let partial = needs_triangles
        && opts.partial_triangles
        && !opts.pre_tessellate
        && parts.iter().any(|p| !p.needs_triangles());
    if partial {
        opts.partial_triangles_observed
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let triangles = needs_triangles.then(|| {
        parts
            .iter()
            .map(|p| {
                if partial && !p.needs_triangles() {
                    Vec::new()
                } else {
                    tessellate_parts(p)
                }
            })
            .collect::<Vec<_>>()
    });

    // The PART boundaries the geometry column is about to erase. `geoarrow.
    // polygon` is one flat ring list per feature, so once the parts are
    // flattened nothing on the wire distinguishes "part 2's exterior" from
    // "part 1's hole" — and every conformant GeoArrow consumer (GeoPandas,
    // lonboard, geoarrow-rs, @geoarrow/deck.gl-layers) reads the former as the
    // latter. `part_offsets` is the sidecar that keeps them separable; the
    // encoder emits the column only when some feature really is multi-part, so
    // a plain-Polygon layer carries nothing extra.
    let any_multipart = parts.iter().any(|p| p.part_starts.len() > 1);
    let polygon_parts = any_multipart.then(|| {
        parts
            .iter()
            .map(|p| {
                p.part_starts
                    .iter()
                    .map(|&s| s as u32)
                    .collect::<Vec<u32>>()
            })
            .collect::<Vec<_>>()
    });

    let geometry: Vec<Vec<Vec<Coord>>> = parts.into_iter().map(|p| p.rings).collect();
    Ok(ColumnarLayer {
        name,
        feature_ids: ids,
        start_times: start,
        end_times: end,
        geometry: GeometryColumn::Polygon(geometry),
        vertex_times: None,
        vertex_values: None,
        triangles,
        polygon_parts,
        vertex_value_matrix: None,
        properties: props,
    })
}

/// Build the id / start / end / property columns shared by every layer kind.
fn common_columns(
    features: &[&ParsedFeature],
    opts: &ColumnarOptions,
) -> (Vec<u64>, Vec<i64>, Vec<i64>, Vec<(String, PropertyColumn)>) {
    let mut ids = Vec::with_capacity(features.len());
    let mut start = Vec::with_capacity(features.len());
    let mut end = Vec::with_capacity(features.len());
    let mut props = PropertyAccumulator::new(
        opts.attribute_filter.clone(),
        Arc::clone(&opts.property_types),
    );

    for f in features {
        ids.push(determine_feature_id(f));
        start.push(f.timestamp as i64);
        end.push(f.end_timestamp.unwrap_or(f.timestamp) as i64);
        props.observe(f.shared_properties.as_ref());
    }
    for f in features {
        props.push_row(f.shared_properties.as_ref());
    }
    (ids, start, end, props.finish())
}

// ----------------------------------------------------------------------------
// Property accumulation
// ----------------------------------------------------------------------------

/// Discovers the property schema across a group of features (the union of all
/// keys, classifying each as numeric or categorical) and then materialises one
/// value per feature, inserting `None` for missing entries.
struct PropertyAccumulator {
    /// Per-key type evidence gathered during the first (`observe`) pass.
    seen: BTreeMap<String, KeyKind>,
    /// Numeric columns, materialised at seal time, in stable (sorted) order.
    numeric: BTreeMap<String, Vec<Option<f64>>>,
    /// Categorical columns, materialised at seal time.
    categorical: BTreeMap<String, Vec<Option<String>>>,
    /// True once the schema is frozen (first `push_row`); `observe` is then a
    /// no-op and the numeric/categorical split is fixed.
    sealed: bool,
    /// Opt-in user-property selection. A key the filter rejects is never
    /// recorded in `seen`, so it produces no column at all. Default `KeepAll`
    /// (every key kept) keeps output byte-for-byte identical to the no-flag
    /// build.
    filter: AttributeFilter,
    /// Authoritative kinds from the input schema — see
    /// [`ColumnarOptions::property_types`]. Declared keys bypass the sniffed
    /// evidence entirely at seal time.
    declared: Arc<PropertyTypes>,
}

/// Type evidence for one property key across a feature group.
#[derive(Default)]
struct KeyKind {
    /// Saw a real JSON number.
    has_number: bool,
    /// Saw a string that parses cleanly as a finite f64 (e.g. "1000.0").
    has_numeric_string: bool,
    /// Saw a value that can't be numeric (non-numeric string, boolean, …).
    has_other: bool,
}

/// Coerce a JSON value to f64, accepting both real numbers and strings that
/// hold a number — so a producer that encoded e.g. `altitude` as the string
/// "1000.0" (a known line/polygon writer bug) still yields a numeric column
/// that can drive colour ramps and elevation.
fn value_as_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(_) => v.as_f64(),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok().filter(|f| f.is_finite()),
        _ => None,
    }
}

/// The numeric coercion a NUMERIC property column applies to one cell — on a
/// BORROWED [`PropValue`], allocating nothing.
///
/// Exactly `value_as_f64(&value.to_json())`, with the transient
/// `serde_json::Value` elided. It has to be exactly that: this is the same
/// function [`PropertyAccumulator::push_row`] materialises tile columns with
/// AND the one the dataset-global statistics pass
/// ([`crate::dataset_stats`]) accumulates over. A pin derived from a domain
/// the encoder disagrees with is worse than no pin at all, so the two share one
/// definition rather than two that agree today.
///
/// `to_json` is not free — for a string cell it clones the string, and for a
/// `Json` cell it deep-clones — and pass 1 runs it once per (row × column) over
/// the whole resident feature vector. That is the per-row allocation the
/// columnar-property rewrite exists to have removed.
///
/// The non-obvious branches, all inherited rather than invented:
/// * a non-finite float has no JSON number spelling, so `to_json` yields
///   `Value::Null` and the cell is absent — hence the `is_finite` filter;
/// * a numeric-looking STRING coerces (that is `value_as_f64`'s whole point),
///   while a boolean does not.
pub(crate) fn prop_value_as_f64(value: PropValue<'_>) -> Option<f64> {
    match value {
        // `json!(f64)` maps NaN/±inf to `Value::Null`, which `value_as_f64`
        // then rejects.
        PropValue::F64(x) => x.is_finite().then_some(x),
        PropValue::I64(x) => Some(x as f64),
        PropValue::U64(x) => Some(x as f64),
        PropValue::Bool(_) => None,
        PropValue::Str(s) => s.trim().parse::<f64>().ok().filter(|f| f.is_finite()),
        PropValue::Json(v) => value_as_f64(v),
    }
}

/// The stringification a CATEGORICAL property column applies to one cell,
/// borrowing where it can and formatting into a caller-owned `scratch` buffer
/// otherwise. `None` for a cell that produces no category (a null, a non-finite
/// float, a nested array/object).
///
/// Mirrors [`PropertyAccumulator::push_row`]'s categorical arm — and, like
/// [`prop_value_as_f64`], is shared with it so the dataset-global scan and the
/// tile encoder can never disagree about what a category IS.
///
/// The callback shape (rather than returning `Cow<'_, str>`) is what keeps the
/// scan allocation-free: a string cell hands back a borrow of the feature's own
/// bytes, and a number/bool cell reuses one buffer across every row instead of
/// minting a `String` per row. The caller only allocates when a category is
/// genuinely new — which is the capped, dataset-sized cost, not a per-row one.
pub(crate) fn with_category<R>(
    value: PropValue<'_>,
    scratch: &mut String,
    f: impl FnOnce(&str) -> R,
) -> Option<R> {
    use std::fmt::Write as _;
    match value {
        PropValue::Str(s) => Some(f(s)),
        PropValue::Bool(b) => Some(f(if b { "true" } else { "false" })),
        PropValue::I64(x) => {
            scratch.clear();
            let _ = write!(scratch, "{x}");
            Some(f(scratch))
        }
        PropValue::U64(x) => {
            scratch.clear();
            let _ = write!(scratch, "{x}");
            Some(f(scratch))
        }
        // A non-finite float has no JSON number spelling: `to_json` yields
        // `Value::Null`, so the categorical arm sees no value at all.
        PropValue::F64(x) => {
            let n = serde_json::Number::from_f64(x)?;
            scratch.clear();
            let _ = write!(scratch, "{n}");
            Some(f(scratch))
        }
        PropValue::Json(v) => match v {
            serde_json::Value::String(s) => Some(f(s)),
            serde_json::Value::Bool(b) => Some(f(if *b { "true" } else { "false" })),
            serde_json::Value::Number(n) => {
                scratch.clear();
                let _ = write!(scratch, "{n}");
                Some(f(scratch))
            }
            _ => None,
        },
    }
}

/// Infer the property kinds for a WHOLE DATASET, so every tile emits the same
/// columns with the same types.
///
/// [`PropertyAccumulator`] sniffs types from the features of ONE tile, which is
/// only safe when every tile sees the same evidence. It doesn't: a column whose
/// values are all null in one tile vanishes from that tile's schema, and a
/// column that happens to hold only numeric-looking strings in one tile but a
/// real string in another flips between `Float64`/quantized and
/// `Dictionary<UInt16, Utf8>`. Both drift the layer schema across tiles — the
/// registry then carries one template per variant, and a consumer that styles
/// on the column gets a different type depending on which tile it reads.
///
/// Feeding the result in as [`ColumnarOptions::property_types`] pins the kinds
/// up front (declared keys bypass per-tile evidence at seal time, and are
/// emitted in EVERY tile even where all their values are null). Database inputs
/// already supply this from the source schema; file inputs derive it here.
///
/// The rule is deliberately IDENTICAL to the per-tile one — numeric iff every
/// observed value across the dataset was a number or a numeric-looking string
/// and nothing forced it categorical — so this only widens the evidence, never
/// changes how a given body of evidence is classified.
pub fn infer_property_types<'a>(
    features: impl IntoIterator<Item = Option<&'a FeatureProperties>>,
    filter: &AttributeFilter,
) -> PropertyTypes {
    let mut acc = PropertyAccumulator::new(filter.clone(), Arc::new(PropertyTypes::new()));
    for props in features {
        acc.observe(props);
    }
    acc.seen
        .iter()
        .map(|(key, kind)| {
            let numeric = (kind.has_number || kind.has_numeric_string) && !kind.has_other;
            (
                key.clone(),
                if numeric {
                    PropertyKind::Numeric
                } else {
                    PropertyKind::Categorical
                },
            )
        })
        .collect()
}

/// Fill the property kinds an input schema did not type, from a whole-dataset
/// pass over the features.
///
/// `declared` is authoritative and is never overridden — a `Utf8` column whose
/// values all happen to look numeric stays [`PropertyKind::Categorical`]. But a
/// schema does not answer for every column: the GeoParquet mapping types the
/// numeric/string/bool columns and returns nothing for the rest (a
/// `Dictionary`-encoded, `List`, or all-`Null` column), and the map is empty
/// outright when the schema could not be read.
///
/// Any column left untyped falls through to per-tile sniffing, which drifts the
/// layer schema across tiles — see [`infer_property_types`]. This closes those
/// gaps. Returns the merged map plus the names that were filled in, so the
/// caller can report them.
pub fn fill_property_type_gaps<'a>(
    declared: &PropertyTypes,
    features: impl IntoIterator<Item = Option<&'a FeatureProperties>>,
    filter: &AttributeFilter,
) -> (PropertyTypes, Vec<String>) {
    let inferred = infer_property_types(features, filter);
    let mut merged = declared.clone();
    let mut filled = Vec::new();
    for (name, kind) in inferred {
        if !merged.contains_key(&name) {
            merged.insert(name.clone(), kind);
            filled.push(name);
        }
    }
    filled.sort();
    (merged, filled)
}

impl PropertyAccumulator {
    /// Construct with an opt-in user-property selection and the (possibly
    /// empty) authoritative schema kinds. Pass [`AttributeFilter::KeepAll`] +
    /// an empty map for the default "keep everything, sniff types" behaviour.
    fn new(filter: AttributeFilter, declared: Arc<PropertyTypes>) -> Self {
        Self {
            seen: BTreeMap::new(),
            numeric: BTreeMap::new(),
            categorical: BTreeMap::new(),
            sealed: false,
            filter,
            declared,
        }
    }

    /// First pass: record type evidence for every key present on a feature.
    ///
    /// `iter()` yields only non-null values, which is why the old
    /// `value.is_null()` guard is gone rather than lost — a null cell is absent
    /// in both the columnar and the owned representation.
    fn observe(&mut self, props: Option<&FeatureProperties>) {
        if self.sealed {
            return;
        }
        let Some(props) = props else { return };
        for (key, value) in props.iter() {
            // Opt-in attribute control: a rejected key never enters `seen`, so
            // it yields no column. System columns aren't user properties and
            // never pass through here, so they always survive.
            if !self.filter.keeps(key) {
                continue;
            }
            let kind = self.seen.entry(key.to_string()).or_default();
            if value.is_number() {
                kind.has_number = true;
            } else if let Some(s) = value.as_str() {
                if s.trim()
                    .parse::<f64>()
                    .map(|f| f.is_finite())
                    .unwrap_or(false)
                {
                    kind.has_numeric_string = true;
                } else {
                    kind.has_other = true;
                }
            } else {
                // Booleans (and anything else non-numeric) → categorical. A flag
                // a producer wants to *sum* should be emitted as numeric 0/1.
                kind.has_other = true;
            }
        }
    }

    /// Freeze the schema. Declared keys (input-schema authority) come first:
    /// each produces a column of its declared kind in EVERY tile, even one
    /// where all its values are null — per-tile evidence would otherwise
    /// silently drop such a column (or reclassify it), drifting the layer
    /// schema across tiles. Undeclared keys keep the evidence rule: numeric
    /// iff every observed value was a number (or a numeric-looking string)
    /// and nothing forced it categorical.
    fn seal(&mut self) {
        if self.sealed {
            return;
        }
        self.sealed = true;
        let declared = Arc::clone(&self.declared);
        for (key, kind) in declared.iter() {
            if !self.filter.keeps(key) {
                continue;
            }
            match kind {
                PropertyKind::Numeric => {
                    self.numeric.insert(key.clone(), Vec::new());
                }
                PropertyKind::Categorical => {
                    self.categorical.insert(key.clone(), Vec::new());
                }
            }
        }
        for (key, kind) in &self.seen {
            if self.numeric.contains_key(key) || self.categorical.contains_key(key) {
                continue; // declared — schema wins over sniffed evidence
            }
            let is_numeric = (kind.has_number || kind.has_numeric_string) && !kind.has_other;
            if is_numeric {
                self.numeric.insert(key.clone(), Vec::new());
            } else {
                self.categorical.insert(key.clone(), Vec::new());
            }
        }
    }

    /// Second pass: append this feature's value for every discovered column.
    fn push_row(&mut self, props: Option<&FeatureProperties>) {
        if !self.sealed {
            self.seal();
        }
        // One reusable formatting buffer for the categorical arm's number/bool
        // cells, so a numeric-backed categorical column costs one allocation
        // per row (the retained `String`) rather than three.
        let mut scratch = String::new();
        for (key, col) in self.numeric.iter_mut() {
            // Routed through the shared `prop_value_as_f64` — the SAME coercion
            // the dataset-global statistics pass accumulates over, so a pinned
            // affine can never be derived from a domain the encoder disagrees
            // with. Numeric-looking strings still coerce and the number widths
            // still round-trip.
            let v = props.and_then(|p| p.get(key)).and_then(prop_value_as_f64);
            col.push(v);
        }
        for (key, col) in self.categorical.iter_mut() {
            let v = props
                .and_then(|p| p.get(key))
                .and_then(|v| with_category(v, &mut scratch, |s| s.to_string()));
            col.push(v);
        }
    }

    fn finish(self) -> Vec<(String, PropertyColumn)> {
        let mut out = Vec::new();
        for (name, values) in self.numeric {
            out.push((name, PropertyColumn::Numeric(values)));
        }
        for (name, values) in self.categorical {
            out.push((name, PropertyColumn::Categorical(values)));
        }
        out
    }
}

// ----------------------------------------------------------------------------
// Geometry extraction
// ----------------------------------------------------------------------------

/// Determine a feature's geometry type.
pub fn determine_geometry_type(feature: &ParsedFeature) -> Result<GeometryType> {
    use geojson::Value as GeomValue;
    let geom = feature
        .geojson
        .geometry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("feature has no geometry"))?;
    Ok(match &geom.value {
        GeomValue::Point(_) | GeomValue::MultiPoint(_) => GeometryType::Point,
        GeomValue::LineString(_) | GeomValue::MultiLineString(_) => GeometryType::LineString,
        GeomValue::Polygon(_) | GeomValue::MultiPolygon(_) => GeometryType::Polygon,
        GeomValue::GeometryCollection(c) => match c.first().map(|g| &g.value) {
            Some(GeomValue::Point(_)) | Some(GeomValue::MultiPoint(_)) => GeometryType::Point,
            Some(GeomValue::LineString(_)) | Some(GeomValue::MultiLineString(_)) => {
                GeometryType::LineString
            }
            _ => GeometryType::Polygon,
        },
    })
}

/// A short, human-readable identity for one feature, for diagnostics only —
/// so a dropped feature can be found in the source rather than just counted.
fn feature_label(feature: &ParsedFeature) -> String {
    use geojson::feature::Id;
    match &feature.geojson.id {
        Some(Id::String(s)) => format!("id {s:?} at ({:.6}, {:.6})", feature.lon, feature.lat),
        Some(Id::Number(n)) => format!("id {n} at ({:.6}, {:.6})", feature.lon, feature.lat),
        None => format!("at ({:.6}, {:.6})", feature.lon, feature.lat),
    }
}

/// Run a per-feature geometry extractor over a group, DROPPING the features it
/// rejects instead of writing something made up in their place.
///
/// An extractor must never paper over unusable geometry with a
/// plausible-looking placeholder — a single-point "line", a one-vertex "ring"
/// at the feature's centroid. Both produce a row that every downstream consumer
/// (validators, feature counts, pickers) treats as real data while rendering as
/// nothing, so a broken source looks like a healthy build. Dropping is the
/// honest answer: the feature is gone, it is counted, and the reason for the
/// first one is logged.
///
/// Returns the surviving features alongside their extracted geometry, in the
/// same order, so the caller can build every other column over exactly those
/// rows. Errors only when NOTHING survived — the caller decides whether that is
/// fatal (it isn't: `build_layers_from_features_with` logs and skips the layer).
fn extract_or_drop<'a, T>(
    features: &[&'a ParsedFeature],
    kind: &str,
    layer: &str,
    extract: impl Fn(&ParsedFeature) -> Result<T>,
) -> Result<(Vec<&'a ParsedFeature>, Vec<T>)> {
    let mut kept: Vec<&'a ParsedFeature> = Vec::with_capacity(features.len());
    let mut extracted: Vec<T> = Vec::with_capacity(features.len());
    let mut first_error: Option<String> = None;
    for f in features {
        match extract(f) {
            Ok(value) => {
                kept.push(*f);
                extracted.push(value);
            }
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(e.to_string());
                }
            }
        }
    }
    let dropped = features.len() - kept.len();
    let reason = first_error.as_deref().unwrap_or("unknown");
    if kept.is_empty() {
        // Every feature was unusable — the caller turns this into a skipped
        // layer, so the rest of the tile still builds.
        anyhow::bail!("all {dropped} {kind} feature(s) had unusable geometry; first: {reason}");
    }
    if dropped > 0 {
        // Aggregated per layer build (one line per tile at worst), matching the
        // `vertex_timestamps` mismatch warning above. NOTE: there is no
        // build-wide total — that counter lives in the tiler's
        // `PlacementCounters`, which this module cannot reach.
        tracing::warn!(
            "layer {layer:?}: dropped {dropped} of {} {kind} feature(s) with unusable \
             geometry — NOT written to this tile (first: {reason})",
            features.len()
        );
    }
    Ok((kept, extracted))
}

/// Extract a flat vertex list for a (multi)linestring feature.
///
/// Errors — rather than inventing a one-vertex line at the feature's
/// representative point — when the geometry is not a (multi)linestring or
/// carries no usable position. See [`extract_or_drop`].
fn extract_line_coords(feature: &ParsedFeature) -> Result<Vec<Coord>> {
    use geojson::Value as GeomValue;
    let geom = feature.geojson.geometry.as_ref().ok_or_else(|| {
        anyhow::anyhow!("line feature {} has no geometry", feature_label(feature))
    })?;
    let coords: Vec<Coord> = match &geom.value {
        GeomValue::LineString(pts) => pts
            .iter()
            .filter(|c| c.len() >= 2)
            .map(|c| [c[0], c[1]])
            .collect(),
        GeomValue::MultiLineString(lines) => lines
            .iter()
            .flatten()
            .filter(|c| c.len() >= 2)
            .map(|c| [c[0], c[1]])
            .collect(),
        // Routed here as a line by `determine_geometry_type` (a
        // GeometryCollection whose first member is a line) but not readable as
        // one. There is no line to extract, and a synthesised single-point one
        // is not a line either.
        other => anyhow::bail!(
            "line feature {} carries {} geometry, which holds no line vertices",
            feature_label(feature),
            other.type_name()
        ),
    };
    if coords.is_empty() {
        anyhow::bail!(
            "line feature {} has no position with both x and y",
            feature_label(feature)
        );
    }
    Ok(coords)
}

/// A polygon feature's rings, flattened PART-MAJOR, plus the part boundaries
/// that the flattening would otherwise erase.
///
/// GeoArrow `ring_offsets` keep the individual RINGS separable on the wire, but
/// they cannot say where one PART of a MultiPolygon ends and the next begins —
/// and that distinction is load-bearing, because ring 0 of a part is an
/// EXTERIOR ring while every later ring of the SAME part is a hole. Keeping
/// `part_starts` alongside the rings lets the tessellator treat each part as
/// its own polygon (see [`tessellate_parts`]) instead of folding parts 2..n
/// into part 1's hole list.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PolygonParts {
    /// Every usable ring of the feature, in part-major order — exactly the
    /// order (and the flattened vertex run) the geometry column is written in.
    pub(crate) rings: Vec<Vec<Coord>>,
    /// Ring index at which each part begins. Always starts at 0, strictly
    /// increasing, one entry per part: `[0]` for a plain `Polygon`, and
    /// `part_starts.len()` is the part count. Part `p` owns
    /// `rings[part_starts[p] .. part_starts[p + 1]]` (the last runs to
    /// `rings.len()`).
    pub(crate) part_starts: Vec<usize>,
}

impl PolygonParts {
    /// Ring index ranges, one per part.
    fn part_ranges(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let starts = &self.part_starts;
        let total = self.rings.len();
        starts.iter().enumerate().map(move |(i, &start)| {
            let end = starts.get(i + 1).copied().unwrap_or(total);
            start..end
        })
    }

    /// True when this feature cannot survive a renderer's own single-boundary
    /// earcut and therefore needs baked triangle indices: it has a hole, or it
    /// has more than one part.
    fn needs_triangles(&self) -> bool {
        self.rings.len() > 1 || self.part_starts.len() > 1
    }
}

/// Append one polygon PART (its exterior ring first, then its holes) to `out`,
/// keeping only rings with enough positions to bound an area.
///
/// A part whose EXTERIOR ring is unusable contributes nothing at all: promoting
/// one of its holes to exterior — which is what a flat ring filter did — turns
/// a hole into a solid slab.
fn push_polygon_part(out: &mut PolygonParts, part: &[Vec<Vec<f64>>]) {
    let mut rings: Vec<Vec<Coord>> = Vec::with_capacity(part.len());
    for (i, ring) in part.iter().enumerate() {
        let coords: Vec<Coord> = ring
            .iter()
            .filter(|c| c.len() >= 2)
            .map(|c| [c[0], c[1]])
            .collect();
        // A closed ring needs 4 positions (3 distinct corners + the repeat of
        // the first); anything shorter bounds no area.
        if coords.len() < 4 {
            if i == 0 {
                return; // no exterior ring ⇒ the whole part is unusable
            }
            continue; // drop just this hole
        }
        rings.push(coords);
    }
    if rings.is_empty() {
        return;
    }
    out.part_starts.push(out.rings.len());
    out.rings.extend(rings);
}

/// Extract a polygon feature's rings, part-major, with the part boundaries.
///
/// Errors — rather than inventing a zero-area ring at the feature's centroid —
/// when no usable ring survives. See [`extract_or_drop`].
fn extract_polygon_rings(feature: &ParsedFeature) -> Result<PolygonParts> {
    use geojson::Value as GeomValue;
    let geom = feature.geojson.geometry.as_ref().ok_or_else(|| {
        anyhow::anyhow!("polygon feature {} has no geometry", feature_label(feature))
    })?;
    let mut parts = PolygonParts::default();
    match &geom.value {
        GeomValue::Polygon(rings) => push_polygon_part(&mut parts, rings),
        GeomValue::MultiPolygon(polys) => {
            for poly in polys {
                push_polygon_part(&mut parts, poly);
            }
        }
        // Routed here as a polygon by `determine_geometry_type` (its catch-all
        // arm for GeometryCollection) but not readable as one.
        other => anyhow::bail!(
            "polygon feature {} carries {} geometry, which holds no rings",
            feature_label(feature),
            other.type_name()
        ),
    }
    if parts.rings.is_empty() {
        anyhow::bail!(
            "polygon feature {} has no ring with at least 4 positions",
            feature_label(feature)
        );
    }
    Ok(parts)
}

/// Tessellate one polygon feature into feature-LOCAL triangle indices, one PART
/// at a time.
///
/// Each part is earcut on its own and its indices are shifted by the running
/// vertex count, so the result indexes the feature's flattened, part-major
/// coordinate run — precisely the run the geometry column writes and both
/// reference readers rebuild.
///
/// Handing the whole flattened ring list to [`tessellate_polygon`] instead
/// passes rings 1..n as earcut HOLE indices, which is right for a `Polygon` and
/// wrong for a `MultiPolygon`: two disjoint unit squares came back as
/// `[0,1,2,2,3,0]` — the 2 triangles of the FIRST square, with the second part
/// missing from the index buffer entirely. Since the renderers bind `triangles`
/// as the GPU index buffer when it is present, those parts rendered invisible.
/// This is not an exotic input: the tiler emits a `MultiPolygon` whenever
/// clipping cuts one source polygon into several pieces inside a tile.
fn tessellate_parts(parts: &PolygonParts) -> Vec<u32> {
    // Single part — every plain `Polygon`, holes included. Byte-identical to
    // the pre-fix path.
    if parts.part_starts.len() <= 1 {
        return tessellate_polygon(&parts.rings);
    }
    let mut indices: Vec<u32> = Vec::new();
    let mut vertex_base: u32 = 0;
    for range in parts.part_ranges() {
        let part = &parts.rings[range];
        let vertices: u32 = part.iter().map(|r| r.len() as u32).sum();
        indices.extend(
            tessellate_polygon(part)
                .into_iter()
                .map(|i| i + vertex_base),
        );
        vertex_base += vertices;
    }
    indices
}

/// Synthesise per-vertex timestamps by cumulative distance along a path.
fn interpolate_vertex_times(coords: &[Coord], start: u64, end: u64) -> Vec<i64> {
    let n = coords.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start as i64];
    }
    let mut cumulative = vec![0.0f64; n];
    for i in 1..n {
        let [lon1, lat1] = coords[i - 1];
        let [lon2, lat2] = coords[i];
        cumulative[i] = cumulative[i - 1] + haversine_distance(lat1, lon1, lat2, lon2);
    }
    let total = cumulative[n - 1];
    let duration = end as f64 - start as f64;
    if total <= 0.0 {
        return vec![start as i64; n];
    }
    cumulative
        .iter()
        .map(|d| start as i64 + (d / total * duration) as i64)
        .collect()
}

// ----------------------------------------------------------------------------
// Feature ids
// ----------------------------------------------------------------------------

/// 64-bit FNV-1a over a byte slice — the synthetic-feature-id hash.
///
/// Deliberately NOT `std::collections::hash_map::DefaultHasher`: its algorithm
/// is explicitly unspecified across Rust releases, so a toolchain bump could
/// change every synthetic id, every tile byte and every content address,
/// silently breaking the incremental-deploy economics. FNV-1a is fixed by
/// spec: offset basis `0xcbf29ce484222325`, prime `0x100000001b3`
/// (http://www.isthe.com/chongo/tech/comp/fnv/). Multi-field inputs are fed
/// as the concatenation of each field's little-endian bytes.
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

/// FNV-1a over a sequence of u64 fields (each folded in as little-endian).
fn fnv1a_64_fields(fields: &[u64]) -> u64 {
    let mut bytes = Vec::with_capacity(fields.len() * 8);
    for f in fields {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    fnv1a_64(&bytes)
}

/// Resolve a stable u64 feature id (from the GeoJSON id, else a hash).
fn determine_feature_id(feature: &ParsedFeature) -> u64 {
    use geojson::feature::Id;

    if let Some(id) = &feature.geojson.id {
        match id {
            Id::Number(num) => {
                if let Some(v) = num.as_u64() {
                    return v;
                }
                if let Some(v) = num.as_i64() {
                    return v as u64;
                }
            }
            Id::String(s) => {
                return fnv1a_64(s.as_bytes());
            }
        }
    }
    fnv1a_64_fields(&[
        feature.timestamp,
        feature.lon.to_bits(),
        feature.lat.to_bits(),
    ])
}

/// Resolve a stable u64 id for a clipped segment.
fn segment_feature_id(segment: &ClippedSegment) -> u64 {
    use geojson::feature::Id;

    if let Some(id) = &segment.feature_id {
        match id {
            Id::Number(num) => {
                if let Some(v) = num.as_u64() {
                    return v;
                }
                if let Some(v) = num.as_i64() {
                    return v as u64;
                }
            }
            Id::String(s) => {
                return fnv1a_64(s.as_bytes());
            }
        }
    }
    match segment.coordinates.first() {
        Some((lon, lat, _)) => fnv1a_64_fields(&[segment.start_time, lon.to_bits(), lat.to_bits()]),
        None => fnv1a_64_fields(&[segment.start_time]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geojson::{Feature, Geometry, Value as GeomValue};
    use serde_json::json;

    /// Every `PropValue` shape a property cell can take, as the borrowed value
    /// AND as the owning JSON the pre-M2 code path rebuilt per row. The two
    /// spellings have to agree cell for cell — see the two tests below.
    fn coercion_corpus() -> Vec<serde_json::Value> {
        vec![
            json!(1.5),
            json!(-0.0),
            json!(0),
            json!(-7),
            json!(9_007_199_254_740_993i64),
            json!(18_446_744_073_709_551_615u64),
            json!(true),
            json!(false),
            json!("alpha"),
            json!("1000.0"),
            json!("  42  "),
            json!("1e309"), // parses to +inf → NOT numeric
            json!("NaN"),   // parses to NaN → NOT numeric
            json!(""),
            json!([1, 2]),
            json!({ "nested": 1 }),
            json!(null),
        ]
    }

    /// `prop_value_as_f64` is EXACTLY `value_as_f64(&value.to_json())`.
    ///
    /// The borrowed coercion exists so the dataset-global statistics pass does
    /// not allocate a `serde_json::Value` per (row × column) — the cost the
    /// columnar-property rewrite was written to remove. But a statistics pass
    /// that coerces differently from the encoder derives a pinned affine over
    /// the wrong domain, which is a silent wrong-values bug rather than a
    /// crash. So the elision is pinned against the literal expression it
    /// replaced, over every value shape a cell can hold.
    #[test]
    fn borrowed_numeric_coercion_matches_the_json_round_trip() {
        for value in coercion_corpus() {
            let Some(borrowed) = PropValue::from_json(&value) else {
                assert!(value.is_null(), "only null yields no PropValue: {value}");
                continue;
            };
            let via_json = value_as_f64(&borrowed.to_json());
            let direct = prop_value_as_f64(borrowed);
            assert_eq!(
                direct.map(f64::to_bits),
                via_json.map(f64::to_bits),
                "numeric coercion diverged for {value}"
            );
        }
    }

    /// `with_category` is EXACTLY the `to_json`-based categorical arm.
    ///
    /// Same argument as above, plus one of its own: the category STRING is the
    /// dictionary key, so a divergence here would put a value in the global
    /// category list that no tile ever emits (or vice versa) — and under the
    /// global-dictionary hoist a category missing from the pin is a hard build
    /// error, not a degradation.
    #[test]
    fn borrowed_category_stringification_matches_the_json_round_trip() {
        let mut scratch = String::new();
        for value in coercion_corpus() {
            let Some(borrowed) = PropValue::from_json(&value) else {
                continue;
            };
            let via_json = match borrowed.to_json() {
                serde_json::Value::String(s) => Some(s),
                serde_json::Value::Bool(b) => Some(b.to_string()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            };
            let direct = with_category(borrowed, &mut scratch, |s| s.to_string());
            assert_eq!(
                direct, via_json,
                "categorical spelling diverged for {value}"
            );
        }
        // A non-finite float has no JSON number spelling, so it is no category.
        assert_eq!(
            with_category(PropValue::F64(f64::NAN), &mut scratch, |s| s.to_string()),
            None
        );
        assert_eq!(
            with_category(PropValue::F64(f64::INFINITY), &mut scratch, |s| s
                .to_string()),
            None
        );
        // ...and it is no numeric value either.
        assert_eq!(prop_value_as_f64(PropValue::F64(f64::NAN)), None);
    }

    /// The schema stays authoritative for the columns it types; only the gaps
    /// are filled. A `Utf8` column whose values all look numeric must NOT be
    /// re-typed numeric by the inference pass.
    #[test]
    fn gap_fill_never_overrides_a_declared_kind() {
        let feats = vec![
            point_feature(0.0, 0.0, json!({ "code": "123", "untyped": "abc" })),
            point_feature(1.0, 1.0, json!({ "code": "456", "untyped": "def" })),
        ];
        let declared: PropertyTypes = [("code".to_string(), PropertyKind::Categorical)]
            .into_iter()
            .collect();
        let (merged, filled) = fill_property_type_gaps(
            &declared,
            feats.iter().map(|f| f.shared_properties.as_ref()),
            &AttributeFilter::KeepAll,
        );
        assert_eq!(
            merged.get("code"),
            Some(&PropertyKind::Categorical),
            "declared kind survives even though the values look numeric"
        );
        assert_eq!(merged.get("untyped"), Some(&PropertyKind::Categorical));
        assert_eq!(
            filled,
            vec!["untyped".to_string()],
            "only the gap is filled"
        );
    }

    /// A column the schema DID type but which is absent from the sampled
    /// features must not be dropped from the merged map.
    #[test]
    fn gap_fill_keeps_declared_columns_absent_from_the_features() {
        let feats = vec![point_feature(0.0, 0.0, json!({ "present": 1 }))];
        let declared: PropertyTypes = [("only_in_schema".to_string(), PropertyKind::Numeric)]
            .into_iter()
            .collect();
        let (merged, filled) = fill_property_type_gaps(
            &declared,
            feats.iter().map(|f| f.shared_properties.as_ref()),
            &AttributeFilter::KeepAll,
        );
        assert_eq!(merged.get("only_in_schema"), Some(&PropertyKind::Numeric));
        assert_eq!(merged.get("present"), Some(&PropertyKind::Numeric));
        assert_eq!(filled, vec!["present".to_string()]);
    }

    /// Dataset-wide inference pins a column's kind so every tile agrees.
    ///
    /// Two tiles' worth of features: in tile A `code` holds only
    /// numeric-looking strings, in tile B it holds a real word. Sniffing each
    /// tile separately makes `code` NUMERIC in A and CATEGORICAL in B — one
    /// layer with two schemas, which is what `stt-validate` reports as schema
    /// drift. The dataset-wide pass sees both and pins Categorical.
    #[test]
    fn dataset_wide_inference_pins_a_column_that_would_flip_per_tile() {
        let tile_a = vec![
            point_feature(0.0, 0.0, json!({ "code": "123" })),
            point_feature(0.1, 0.1, json!({ "code": "456" })),
        ];
        let tile_b = vec![point_feature(9.0, 9.0, json!({ "code": "unknown" }))];

        // Per-tile sniffing disagrees across the two tiles.
        let kind_of = |feats: &[ParsedFeature], declared: Arc<PropertyTypes>| {
            let refs: Vec<&ParsedFeature> = feats.iter().collect();
            let layers = build_layers_from_features_with(
                &refs,
                "default",
                ColumnarOptions {
                    property_types: declared,
                    ..Default::default()
                },
            )
            .unwrap();
            let (_, col) = layers[0]
                .properties
                .iter()
                .find(|(n, _)| n == "code")
                .expect("code column present");
            matches!(col, PropertyColumn::Numeric(_))
        };
        assert!(
            kind_of(&tile_a, Arc::default()),
            "tile A alone sniffs numeric"
        );
        assert!(
            !kind_of(&tile_b, Arc::default()),
            "tile B alone sniffs categorical — the drift"
        );

        // Inferred over BOTH tiles, the column is categorical everywhere.
        let all: Vec<ParsedFeature> = tile_a.iter().chain(tile_b.iter()).cloned().collect();
        let declared = Arc::new(infer_property_types(
            all.iter().map(|f| f.shared_properties.as_ref()),
            &AttributeFilter::KeepAll,
        ));
        assert_eq!(declared.get("code"), Some(&PropertyKind::Categorical));
        assert!(!kind_of(&tile_a, Arc::clone(&declared)));
        assert!(!kind_of(&tile_b, Arc::clone(&declared)));
    }

    /// A column that is all-null in one tile must still appear there, with the
    /// dataset's kind — otherwise that tile's schema is missing a column the
    /// others have (the second drift class).
    #[test]
    fn dataset_wide_inference_keeps_an_all_null_column_present() {
        let with_value = vec![point_feature(0.0, 0.0, json!({ "sst": 12.5 }))];
        let all_null = vec![point_feature(9.0, 9.0, json!({ "other": 1 }))];
        let all: Vec<ParsedFeature> = with_value.iter().chain(all_null.iter()).cloned().collect();
        let declared = Arc::new(infer_property_types(
            all.iter().map(|f| f.shared_properties.as_ref()),
            &AttributeFilter::KeepAll,
        ));

        let refs: Vec<&ParsedFeature> = all_null.iter().collect();
        let layers = build_layers_from_features_with(
            &refs,
            "default",
            ColumnarOptions {
                property_types: Arc::clone(&declared),
                ..Default::default()
            },
        )
        .unwrap();
        let names: Vec<&str> = layers[0]
            .properties
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(
            names.contains(&"sst"),
            "an all-null column must still be emitted: {names:?}"
        );
    }

    fn point_feature(lon: f64, lat: f64, props: serde_json::Value) -> ParsedFeature {
        ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::Point(vec![lon, lat]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            // Properties live in shared_properties (see input.rs).
            shared_properties: props
                .as_object()
                .cloned()
                .and_then(crate::props::FeatureProperties::from_map),
            timestamp: 1000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon,
            lat,
        }
    }

    fn line_feature(coords: Vec<[f64; 2]>, start: u64, end: Option<u64>) -> ParsedFeature {
        let pts: Vec<Vec<f64>> = coords.iter().map(|c| vec![c[0], c[1]]).collect();
        ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::LineString(pts))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: start,
            end_timestamp: end,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: coords[0][0],
            lat: coords[0][1],
        }
    }

    #[test]
    fn point_features_become_one_layer() {
        let f1 = point_feature(-122.4, 37.7, json!({ "speed": 10.0, "kind": "car" }));
        let f2 = point_feature(-122.5, 37.8, json!({ "speed": 20.0 }));
        let refs = vec![&f1, &f2];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].feature_count(), 2);
        // "kind" present only on f1 must still be discovered, with None for f2.
        let kind = layers[0]
            .properties
            .iter()
            .find(|(n, _)| n == "kind")
            .expect("kind column");
        match &kind.1 {
            PropertyColumn::Categorical(v) => {
                assert_eq!(v[0].as_deref(), Some("car"));
                assert_eq!(v[1], None);
            }
            _ => panic!("kind should be categorical"),
        }
    }

    #[test]
    fn numeric_string_and_boolean_properties_are_classified() {
        // Guards the columnar inference contract the typed writers rely on:
        // numbers -> Numeric, strings -> Categorical, and booleans carried as
        // Categorical "true"/"false" rather than silently dropped (the pre-fix
        // behaviour matched neither arm in `observe`).
        let f1 = point_feature(
            -122.4,
            37.7,
            json!({ "altitude": 1000.0, "label": "alpha", "active": true }),
        );
        let f2 = point_feature(
            -122.5,
            37.8,
            json!({ "altitude": 2000.0, "label": "beta", "active": false }),
        );
        let refs = vec![&f1, &f2];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        let col = |name: &str| {
            layers[0]
                .properties
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, c)| c)
        };

        match col("altitude").expect("altitude column") {
            PropertyColumn::Numeric(v) => {
                assert_eq!(v[0], Some(1000.0));
                assert_eq!(v[1], Some(2000.0));
            }
            _ => panic!("altitude should be numeric"),
        }
        match col("label").expect("label column") {
            PropertyColumn::Categorical(v) => {
                assert_eq!(v[0].as_deref(), Some("alpha"));
                assert_eq!(v[1].as_deref(), Some("beta"));
            }
            _ => panic!("label should be categorical"),
        }
        // The boolean column must be present (not dropped) and carried as a
        // "true"/"false" categorical.
        match col("active").expect("boolean column must be present, not dropped") {
            PropertyColumn::Categorical(v) => {
                assert_eq!(v[0].as_deref(), Some("true"));
                assert_eq!(v[1].as_deref(), Some("false"));
            }
            _ => panic!("boolean should be carried as categorical"),
        }
    }

    /// Producers do encode numbers as strings, so a property whose every value
    /// parses as a number is classified NUMERIC and can drive ramps and
    /// elevation — while a genuinely non-numeric string column stays categorical.
    #[test]
    fn numeric_strings_are_promoted_to_numeric() {
        let f1 = point_feature(
            -122.4,
            37.7,
            json!({ "altitude": "1000.0", "code": "A12", "mixed": "5" }),
        );
        let f2 = point_feature(
            -122.5,
            37.8,
            json!({ "altitude": "2000", "code": "B7", "mixed": "n/a" }),
        );
        let layers = build_layers_from_features(&[&f1, &f2], "default").unwrap();
        let col = |name: &str| {
            layers[0]
                .properties
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, c)| c)
        };

        // All-numeric strings → promoted to a Numeric column.
        match col("altitude").expect("altitude column") {
            PropertyColumn::Numeric(v) => {
                assert_eq!(v[0], Some(1000.0));
                assert_eq!(v[1], Some(2000.0));
            }
            _ => panic!("string-encoded numbers should promote to numeric"),
        }
        // Non-numeric strings → stays categorical.
        match col("code").expect("code column") {
            PropertyColumn::Categorical(v) => {
                assert_eq!(v[0].as_deref(), Some("A12"));
                assert_eq!(v[1].as_deref(), Some("B7"));
            }
            _ => panic!("non-numeric strings should stay categorical"),
        }
        // A column with *any* non-numeric value stays categorical (no partial
        // promotion that would silently null-out the "n/a" row).
        match col("mixed").expect("mixed column") {
            PropertyColumn::Categorical(v) => {
                assert_eq!(v[0].as_deref(), Some("5"));
                assert_eq!(v[1].as_deref(), Some("n/a"));
            }
            _ => panic!("mixed numeric/non-numeric column should stay categorical"),
        }
    }

    /// `--exclude` drops the named property while leaving every other user
    /// property AND all system columns (id/start/end/geometry) intact.
    #[test]
    fn exclude_drops_only_named_property() {
        let f1 = point_feature(
            -122.4,
            37.7,
            json!({ "speed": 10.0, "kind": "car", "name": "a" }),
        );
        let f2 = point_feature(
            -122.5,
            37.8,
            json!({ "speed": 20.0, "kind": "bus", "name": "b" }),
        );
        let opts = ColumnarOptions {
            attribute_filter: AttributeFilter::Exclude(["kind".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let layers = build_layers_from_features_with(&[&f1, &f2], "default", opts).unwrap();
        let names: Vec<&str> = layers[0]
            .properties
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(!names.contains(&"kind"), "excluded property must be gone");
        assert!(names.contains(&"speed"));
        assert!(names.contains(&"name"));
        // System columns untouched.
        assert_eq!(layers[0].feature_count(), 2);
        assert_eq!(layers[0].start_times.len(), 2);
        assert_eq!(layers[0].geometry.len(), 2);
    }

    /// `--include` keeps ONLY the named properties (plus system columns).
    #[test]
    fn include_keeps_only_named_properties() {
        let f1 = point_feature(
            -122.4,
            37.7,
            json!({ "speed": 10.0, "kind": "car", "name": "a" }),
        );
        let f2 = point_feature(
            -122.5,
            37.8,
            json!({ "speed": 20.0, "kind": "bus", "name": "b" }),
        );
        let opts = ColumnarOptions {
            attribute_filter: AttributeFilter::Include(["speed".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let layers = build_layers_from_features_with(&[&f1, &f2], "default", opts).unwrap();
        let names: Vec<&str> = layers[0]
            .properties
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(names, vec!["speed"], "only the included property survives");
        // System columns untouched.
        assert_eq!(layers[0].feature_count(), 2);
        assert!(!layers[0].start_times.is_empty());
        assert!(!layers[0].end_times.is_empty());
    }

    /// `--exclude-all` drops every user property but keeps system columns.
    #[test]
    fn exclude_all_drops_every_user_property() {
        let f1 = point_feature(-122.4, 37.7, json!({ "speed": 10.0, "kind": "car" }));
        let opts = ColumnarOptions {
            attribute_filter: AttributeFilter::ExcludeAll,
            ..Default::default()
        };
        let layers = build_layers_from_features_with(&[&f1], "default", opts).unwrap();
        assert!(layers[0].properties.is_empty(), "no user property survives");
        // System columns remain.
        assert_eq!(layers[0].feature_count(), 1);
        assert_eq!(layers[0].geometry.len(), 1);
    }

    /// Declared property kinds pin the tile schema: a column that is all-null
    /// within one tile still yields a (all-null) column of the declared kind
    /// there, instead of vanishing. Without the pin, a GeoParquet input with a
    /// sparsely-populated column drifts schema from tile to tile (e.g. AIS
    /// `sog` null for every row that lands in one tile).
    #[test]
    fn declared_property_kinds_pin_schema_for_all_null_tiles() {
        // Tile A's features: `sog` present. Tile B's: `sog` all-null (absent
        // from the JSON map — how the parquet reader materialises NULL).
        let with_val = point_feature(-122.4, 37.7, json!({ "sog": 3.5, "class": "cargo" }));
        let all_null = point_feature(10.0, 50.0, json!({ "class": "tanker" }));

        let declared: PropertyTypes = [
            ("sog".to_string(), PropertyKind::Numeric),
            ("class".to_string(), PropertyKind::Categorical),
        ]
        .into_iter()
        .collect();
        let opts = ColumnarOptions {
            property_types: Arc::new(declared),
            ..Default::default()
        };

        // Without the declared map, this tile would have NO `sog` column.
        let tile_b =
            build_layers_from_features_with(&[&all_null], "default", opts.clone()).unwrap();
        let names_b: Vec<&str> = tile_b[0]
            .properties
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        // finish() emits numeric columns first, then categorical.
        assert_eq!(
            names_b,
            vec!["sog", "class"],
            "declared columns always present"
        );
        match &tile_b[0]
            .properties
            .iter()
            .find(|(n, _)| n == "sog")
            .unwrap()
            .1
        {
            PropertyColumn::Numeric(v) => assert_eq!(v, &vec![None]),
            other => panic!("declared-numeric sog must stay Numeric, got {other:?}"),
        }

        // The populated tile has the identical property schema.
        let tile_a = build_layers_from_features_with(&[&with_val], "default", opts).unwrap();
        let names_a: Vec<&str> = tile_a[0]
            .properties
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(names_a, names_b, "schema identical across tiles");
        match &tile_a[0]
            .properties
            .iter()
            .find(|(n, _)| n == "sog")
            .unwrap()
            .1
        {
            PropertyColumn::Numeric(v) => assert_eq!(v, &vec![Some(3.5)]),
            other => panic!("expected Numeric sog, got {other:?}"),
        }

        // Declared kind beats sniffed evidence: a numeric-looking string in a
        // declared-Categorical column stays categorical (schema authority).
        let numeric_string = point_feature(0.0, 0.0, json!({ "class": "42" }));
        let opts2 = ColumnarOptions {
            property_types: Arc::new(
                [("class".to_string(), PropertyKind::Categorical)]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        };
        let tile_c = build_layers_from_features_with(&[&numeric_string], "default", opts2).unwrap();
        match &tile_c[0]
            .properties
            .iter()
            .find(|(n, _)| n == "class")
            .unwrap()
            .1
        {
            PropertyColumn::Categorical(v) => assert_eq!(v, &vec![Some("42".to_string())]),
            other => panic!("declared-categorical must stay Categorical, got {other:?}"),
        }
    }

    /// Clipped-segment layers honour the same attribute filter.
    #[test]
    fn segment_layer_honours_attribute_filter() {
        use crate::clip::ClippedSegment;
        let props = json!({ "road": "main", "lanes": 4 })
            .as_object()
            .cloned()
            .and_then(crate::props::FeatureProperties::from_map);
        let seg = ClippedSegment {
            tile_x: 0,
            tile_y: 0,
            zoom: 10,
            coordinates: vec![(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
            timestamps: vec![1000, 2000],
            vertex_values: vec![],
            vertex_value_matrix: vec![],
            start_time: 1000,
            end_time: 2000,
            properties: props,
            feature_id: None,
        };
        let layer = build_layer_from_segments(
            &[&seg],
            "tracks",
            &ColumnarOptions {
                attribute_filter: AttributeFilter::Include(
                    ["road".to_string()].into_iter().collect(),
                ),
                ..Default::default()
            },
        )
        .unwrap();
        let names: Vec<&str> = layer.properties.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["road"]);
        // vertex_time is a SYSTEM column and must survive the filter.
        assert!(layer.vertex_times.is_some());
    }

    #[test]
    fn mixed_geometry_types_split_into_separate_layers() {
        let pt = point_feature(0.0, 0.0, json!({}));
        let line = line_feature(vec![[0.0, 0.0], [1.0, 1.0]], 1000, None);
        let refs = vec![&pt, &line];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert_eq!(layers.len(), 2);
        // Distinct, kind-suffixed names so a reader can tell them apart.
        let names: Vec<&str> = layers.iter().map(|l| l.name.as_str()).collect();
        assert!(names.contains(&"default_points"));
        assert!(names.contains(&"default_lines"));
    }

    #[test]
    fn line_with_duration_gets_interpolated_vertex_times() {
        let line = line_feature(vec![[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], 1000, Some(3000));
        let refs = vec![&line];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        let vt = layers[0]
            .vertex_times
            .as_ref()
            .expect("vertex times present");
        assert_eq!(vt[0].len(), 3);
        // Evenly spaced vertices -> first 1000, last 3000, middle ~2000.
        assert_eq!(vt[0][0], 1000);
        assert_eq!(vt[0][2], 3000);
        assert!((vt[0][1] - 2000).abs() <= 1);
    }

    #[test]
    fn line_without_duration_has_no_vertex_times() {
        let line = line_feature(vec![[0.0, 0.0], [1.0, 1.0]], 1000, None);
        let refs = vec![&line];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert!(layers[0].vertex_times.is_none());
    }

    #[test]
    fn matrix_corridor_drops_dead_vertex_times() {
        // A flow corridor carries a per-vertex×bucket matrix and spans the whole
        // range (start..end), so build_line_layer WOULD interpolate per-vertex
        // times — but a matrix corridor is timeless (animated by the matrix), so
        // those times are dead weight and must be suppressed.
        let mut line = line_feature(vec![[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], 1000, Some(3000));
        // 3 vertices × 2 buckets, vertex-major (matrix.len() % verts == 0).
        line.vertex_value_matrix = Some(vec![5.0, 7.0, 5.0, 7.0, 5.0, 7.0]);
        let refs = vec![&line];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert!(
            layers[0].vertex_times.is_none(),
            "matrix corridor must not carry a per-vertex time column"
        );
        // The matrix itself is still attached (it's the time signal).
        assert!(layers[0].vertex_value_matrix.is_some());
    }

    /// Build a square polygon feature for the pre-tessellation tests.
    fn polygon_feature(corner: [f64; 2], size: f64) -> ParsedFeature {
        let [x, y] = corner;
        let ring: Vec<Vec<f64>> = vec![
            vec![x, y],
            vec![x + size, y],
            vec![x + size, y + size],
            vec![x, y + size],
            vec![x, y], // closing vertex
        ];
        ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::Polygon(vec![ring]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 1000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: x,
            lat: y,
        }
    }

    /// A square polygon with a square hole (two rings) — exercises the
    /// multi-ring auto-tessellation path.
    fn polygon_feature_with_hole() -> ParsedFeature {
        let exterior: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![4.0, 0.0],
            vec![4.0, 4.0],
            vec![0.0, 4.0],
            vec![0.0, 0.0],
        ];
        let hole: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![2.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 1.0],
        ];
        ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::Polygon(vec![exterior, hole]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 1000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: 2.0,
            lat: 2.0,
        }
    }

    #[test]
    fn polygon_layer_omits_triangles_by_default() {
        let p = polygon_feature([0.0, 0.0], 1.0);
        let refs = vec![&p];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert_eq!(layers.len(), 1);
        assert!(layers[0].triangles.is_none());
    }

    #[test]
    fn multi_ring_polygon_auto_bakes_triangles_without_flag() {
        // A hole-bearing polygon CANNOT render through deck.gl's binary earcut
        // path (it bridges the exterior and hole rings with spanning
        // triangles). The builder must bake the hole-aware index sidecar even
        // when --pre-tessellate is OFF.
        //
        // TB-12 CHANGED THE OTHER HALF of this test: the single-ring feature
        // sharing the layer used to be tessellated too, for a uniform per-tile
        // index buffer. It now gets an EMPTY list, because the decoder earcuts
        // it on arrival. The holed feature's assertions below are unchanged —
        // that is the half that must never regress.
        let holed = polygon_feature_with_hole();
        let simple = polygon_feature([10.0, 10.0], 1.0);
        let refs = vec![&holed, &simple];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert_eq!(layers.len(), 1);
        let tri = layers[0]
            .triangles
            .as_ref()
            .expect("multi-ring layer must auto-bake triangles even without the flag");
        assert_eq!(tri.len(), 2);
        // The holed square (8 distinct verts across two rings) tessellates into
        // a ring of quads = 8 triangles = 24 indices; every index stays within
        // the feature's 10 vertices (5 per closed ring), never bridging out.
        assert!(!tri[0].is_empty() && tri[0].len() % 3 == 0);
        for &i in &tri[0] {
            assert!((i as usize) < 10, "triangle index escapes the feature");
        }
        // The simple square is left to the decoder's backfill.
        assert!(
            tri[1].is_empty(),
            "TB-12: a single-ring feature must not be baked"
        );
    }

    /// TB-12 — `--pre-tessellate` is the documented way back to bake-everything,
    /// for clients that would rather spend bytes than decode CPU. It must bake
    /// the single-ring feature the default now skips, and declare NO capability
    /// (nothing is mixed, so every reader can still open the archive).
    #[test]
    fn pre_tessellate_still_bakes_every_feature_and_declares_nothing() {
        let holed = polygon_feature_with_hole();
        let simple = polygon_feature([10.0, 10.0], 1.0);
        let refs = vec![&holed, &simple];
        let opts = ColumnarOptions {
            pre_tessellate: true,
            ..ColumnarOptions::default()
        };
        let layers = build_layers_from_features_with(&refs, "default", opts.clone()).unwrap();
        let tri = layers[0].triangles.as_ref().expect("pre-tessellate bakes");
        assert!(!tri[0].is_empty());
        assert_eq!(
            tri[1].len(),
            6,
            "--pre-tessellate must bake the simple ring"
        );
        assert!(
            !opts
                .partial_triangles_observed
                .load(std::sync::atomic::Ordering::Relaxed),
            "nothing was mixed, so no capability is owed"
        );
    }

    /// TB-12 — the capability is owed only when a layer actually MIXES. A layer
    /// whose every feature needs baking emits the incumbent bytes, so locking
    /// older readers out of it would be gratuitous.
    #[test]
    fn the_capability_is_observed_only_when_a_layer_actually_mixes() {
        // Two holed features: everything is baked, nothing is empty.
        let a = polygon_feature_with_hole();
        let b = polygon_feature_with_hole();
        let refs = vec![&a, &b];
        let all_baked = ColumnarOptions::default();
        let layers = build_layers_from_features_with(&refs, "default", all_baked.clone()).unwrap();
        let tri = layers[0].triangles.as_ref().unwrap();
        assert!(tri.iter().all(|t| !t.is_empty()));
        assert!(
            !all_baked
                .partial_triangles_observed
                .load(std::sync::atomic::Ordering::Relaxed),
            "a uniformly-baked layer mixes nothing and owes no capability"
        );

        // Add a single-ring feature and the layer mixes.
        let simple = polygon_feature([10.0, 10.0], 1.0);
        let refs = vec![&a, &simple];
        let mixed = ColumnarOptions::default();
        let layers = build_layers_from_features_with(&refs, "default", mixed.clone()).unwrap();
        let tri = layers[0].triangles.as_ref().unwrap();
        assert!(!tri[0].is_empty() && tri[1].is_empty());
        assert!(
            mixed
                .partial_triangles_observed
                .load(std::sync::atomic::Ordering::Relaxed),
            "a mixed layer MUST be observed — an undeclared one vanishes geometry"
        );

        // And a layer with no polygons at all observes nothing.
        let untouched = ColumnarOptions::default();
        let p = point_feature(0.0, 0.0, json!({}));
        build_layers_from_features_with(&[&p], "default", untouched.clone()).unwrap();
        assert!(!untouched
            .partial_triangles_observed
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    /// Synthetic ids MUST come from the documented FNV-1a 64 (never
    /// `DefaultHasher`, whose algorithm may change between Rust releases and
    /// would churn every content address). Pinned against the published FNV
    /// test vectors plus the exact composition rules for both id paths, so
    /// any accidental change to the hash breaks loudly here.
    #[test]
    fn synthetic_ids_use_stable_fnv1a() {
        // Published FNV-1a 64 vectors: "" → offset basis, "a", "foobar".
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x85944171f73967e8);

        // String-id path: FNV-1a over the raw UTF-8 bytes.
        let mut f = point_feature(-122.4, 37.7, json!({}));
        f.geojson.id = Some(geojson::feature::Id::String("quake-42".to_string()));
        assert_eq!(determine_feature_id(&f), fnv1a_64(b"quake-42"));

        // Synthetic fallback: little-endian (timestamp, lon bits, lat bits).
        let f2 = point_feature(-122.4, 37.7, json!({}));
        assert_eq!(
            determine_feature_id(&f2),
            fnv1a_64_fields(&[1000, (-122.4f64).to_bits(), 37.7f64.to_bits()])
        );

        // Segment fallback: little-endian (start_time, first lon bits, first lat bits).
        let seg = crate::clip::ClippedSegment {
            tile_x: 0,
            tile_y: 0,
            zoom: 10,
            coordinates: vec![(1.5, 2.5, 0.0)],
            timestamps: vec![7],
            vertex_values: vec![],
            vertex_value_matrix: vec![],
            start_time: 7,
            end_time: 7,
            properties: None,
            feature_id: None,
        };
        assert_eq!(
            segment_feature_id(&seg),
            fnv1a_64_fields(&[7, 1.5f64.to_bits(), 2.5f64.to_bits()])
        );
    }

    /// Build a MultiPolygon feature from explicit parts (each part a ring list).
    fn multipolygon_feature(parts: Vec<Vec<Vec<[f64; 2]>>>) -> ParsedFeature {
        let value: Vec<Vec<Vec<Vec<f64>>>> = parts
            .iter()
            .map(|part| {
                part.iter()
                    .map(|ring| ring.iter().map(|c| vec![c[0], c[1]]).collect())
                    .collect()
            })
            .collect();
        let first = parts[0][0][0];
        ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::MultiPolygon(value))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 1000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: first[0],
            lat: first[1],
        }
    }

    /// A closed square ring, counter-clockwise, at `corner` with side `size`.
    fn square_ring(corner: [f64; 2], size: f64) -> Vec<[f64; 2]> {
        let [x, y] = corner;
        vec![
            [x, y],
            [x + size, y],
            [x + size, y + size],
            [x, y + size],
            [x, y],
        ]
    }

    /// Total area covered by a feature's baked triangles, computed from the
    /// feature's OWN flattened coordinate run (the same run the readers rebuild
    /// from the geometry column). This is the assertion that actually pins
    /// tessellation correctness: bridged parts, dropped parts and un-subtracted
    /// holes all move the number.
    fn triangulated_area(rings: &[Vec<Coord>], triangles: &[u32]) -> f64 {
        let flat: Vec<Coord> = rings.iter().flatten().copied().collect();
        assert_eq!(
            triangles.len() % 3,
            0,
            "index count must be a multiple of 3"
        );
        let mut area = 0.0;
        for t in triangles.chunks(3) {
            let a = flat[t[0] as usize];
            let b = flat[t[1] as usize];
            let c = flat[t[2] as usize];
            area += ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() / 2.0;
        }
        area
    }

    /// Every part of a MultiPolygon is tessellated on its own. Flattening the
    /// parts into ONE ring list and handing that to earcut treats rings 1..n as
    /// HOLES, so part 2 of two disjoint squares is subtracted from part 1
    /// rather than triangulated and vanishes from the index buffer (invisible
    /// on screen, because the renderers bind `triangles` as the GPU index
    /// buffer when present). The tiler emits exactly this shape whenever
    /// clipping cuts a polygon into several pieces inside one tile.
    #[test]
    fn multipolygon_parts_are_each_tessellated() {
        let f = multipolygon_feature(vec![
            vec![square_ring([0.0, 0.0], 1.0)],
            vec![square_ring([10.0, 10.0], 1.0)],
        ]);
        let layers = build_layers_from_features(&[&f], "default").unwrap();
        let rings = match &layers[0].geometry {
            GeometryColumn::Polygon(polys) => &polys[0],
            other => panic!("expected a polygon layer, got {other:?}"),
        };
        assert_eq!(rings.len(), 2, "one exterior ring per part");
        let tri = layers[0]
            .triangles
            .as_ref()
            .expect("a multi-part feature must carry baked triangles")[0]
            .clone();

        // Both unit squares triangulated: 2 triangles each.
        assert_eq!(
            tri.len(),
            12,
            "both parts must be triangulated, got {tri:?}"
        );
        // Pre-fix this was 2.0 → 1.0 (only the first square survived).
        assert!(
            (triangulated_area(rings, &tri) - 2.0).abs() < 1e-9,
            "both unit squares must be covered exactly once: {tri:?}"
        );
        // Every index stays inside the feature (5 vertices per closed ring).
        assert!(tri.iter().all(|&i| (i as usize) < 10));
        // …and the second part is genuinely referenced.
        assert!(
            tri.iter().any(|&i| i >= 5),
            "no index reaches part 2: {tri:?}"
        );
    }

    /// The counterpart guard: rings 1..n of a SINGLE part are still holes, and
    /// must still be subtracted. A 4×4 square with a 1×1 hole covers 15, not 16.
    #[test]
    fn polygon_hole_is_still_subtracted() {
        let f = polygon_feature_with_hole();
        let layers = build_layers_from_features(&[&f], "default").unwrap();
        let rings = match &layers[0].geometry {
            GeometryColumn::Polygon(polys) => &polys[0],
            other => panic!("expected a polygon layer, got {other:?}"),
        };
        assert_eq!(rings.len(), 2, "exterior + hole");
        let tri = &layers[0]
            .triangles
            .as_ref()
            .expect("a holed feature must carry baked triangles")[0];
        assert!(
            (triangulated_area(rings, tri) - 15.0).abs() < 1e-9,
            "the hole must be subtracted (16 - 1), got {}",
            triangulated_area(rings, tri)
        );
    }

    /// A MultiPolygon part keeps its OWN holes: two squares, the first holed.
    /// Cross-part hole assignment would show up as a different covered area.
    #[test]
    fn multipolygon_holes_stay_with_their_part() {
        let f = multipolygon_feature(vec![
            vec![square_ring([0.0, 0.0], 4.0), square_ring([1.0, 1.0], 1.0)],
            vec![square_ring([10.0, 10.0], 2.0)],
        ]);
        let layers = build_layers_from_features(&[&f], "default").unwrap();
        let rings = match &layers[0].geometry {
            GeometryColumn::Polygon(polys) => &polys[0],
            other => panic!("expected a polygon layer, got {other:?}"),
        };
        let tri = &layers[0].triangles.as_ref().unwrap()[0];
        // (16 - 1) + 4 = 19.
        assert!(
            (triangulated_area(rings, tri) - 19.0).abs() < 1e-9,
            "got {}",
            triangulated_area(rings, tri)
        );
    }

    /// Contract pin for the triangle sidecar, TB-12 edition: emission is
    /// PER-FEATURE, and the boundary is exactly "can a renderer's own
    /// single-boundary earcut reproduce this?".
    ///
    /// The pin that matters is the negative one. A hole or an extra part left
    /// unbaked would be earcut as one flat loop by the decoder's backfill,
    /// bridging disjoint rings and filling holes — the storm-radar isoband
    /// streaks, back as a SILENT wrong render. So multi-part and holed features
    /// must always carry real indices; only genuinely single-ring features may
    /// be left to the reader.
    #[test]
    fn only_features_needing_baked_indices_get_them() {
        let multipart = multipolygon_feature(vec![
            vec![square_ring([0.0, 0.0], 1.0)],
            vec![square_ring([5.0, 5.0], 1.0)],
        ]);
        let holed = polygon_feature_with_hole();
        let simple = polygon_feature([20.0, 20.0], 1.0);
        let opts = ColumnarOptions::default();
        let layers = build_layers_from_features_with(
            &[&multipart, &holed, &simple],
            "default",
            opts.clone(),
        )
        .unwrap();
        let tri = layers[0]
            .triangles
            .as_ref()
            .expect("a layer with a multi-part / holed feature bakes triangles");
        assert_eq!(tri.len(), 3, "one list per feature");
        // Multi-part and holed: baked, and a real triangulation.
        for i in [0usize, 1] {
            assert!(
                !tri[i].is_empty() && tri[i].len() % 3 == 0,
                "feature {i} is not reproducible by a flat earcut and MUST be baked, got {:?}",
                tri[i]
            );
        }
        // Single ring: left to the decoder's backfill.
        assert!(tri[2].is_empty(), "a single ring must not be baked");
        // ...and because this layer mixes, the capability is owed.
        assert!(opts
            .partial_triangles_observed
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    /// An unreadable polygon is DROPPED, never replaced by a one-vertex ring at
    /// the feature's representative point — such a row counts as a feature,
    /// passes validation and renders nothing. The surviving rows stay aligned
    /// across every column.
    #[test]
    fn degenerate_polygon_is_dropped_not_fabricated() {
        // A "polygon" with a 2-position ring: no area, nothing to extract.
        let mut broken = polygon_feature([0.0, 0.0], 1.0);
        broken.geojson.geometry = Some(Geometry::new(GeomValue::Polygon(vec![vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ]])));
        broken.shared_properties = json!({ "name": "broken" })
            .as_object()
            .cloned()
            .and_then(crate::props::FeatureProperties::from_map);
        let mut good = polygon_feature([10.0, 10.0], 1.0);
        good.shared_properties = json!({ "name": "good" })
            .as_object()
            .cloned()
            .and_then(crate::props::FeatureProperties::from_map);

        let layers = build_layers_from_features(&[&broken, &good], "default").unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(
            layers[0].feature_count(),
            1,
            "the broken feature is dropped"
        );
        // Column alignment: the surviving row's property is the GOOD one.
        match &layers[0]
            .properties
            .iter()
            .find(|(n, _)| n == "name")
            .expect("name column")
            .1
        {
            PropertyColumn::Categorical(v) => assert_eq!(v, &vec![Some("good".to_string())]),
            other => panic!("expected a categorical name column, got {other:?}"),
        }
        // Geometry is the real square, never a 1-vertex placeholder.
        match &layers[0].geometry {
            GeometryColumn::Polygon(polys) => {
                assert_eq!(polys.len(), 1);
                assert_eq!(polys[0][0].len(), 5);
            }
            other => panic!("expected a polygon layer, got {other:?}"),
        }

        // When NOTHING survives, fail closed instead of making the source loss
        // look like an ordinary empty tile.
        let error = build_layers_from_features(&[&broken], "default")
            .err()
            .expect("fully unusable polygon layer must fail");
        assert!(error.to_string().contains("all 1 polygon feature"));
    }

    /// Same for lines: an empty LineString is dropped, never turned into a
    /// single-point "line" at the representative point.
    #[test]
    fn degenerate_line_is_dropped_not_fabricated() {
        let mut broken = line_feature(vec![[0.0, 0.0], [1.0, 1.0]], 1000, None);
        broken.geojson.geometry = Some(Geometry::new(GeomValue::LineString(vec![])));
        let good = line_feature(vec![[5.0, 5.0], [6.0, 6.0]], 1000, None);

        let layers = build_layers_from_features(&[&broken, &good], "default").unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].feature_count(), 1);
        match &layers[0].geometry {
            GeometryColumn::LineString(lines) => {
                assert_eq!(lines[0], vec![[5.0, 5.0], [6.0, 6.0]]);
            }
            other => panic!("expected a line layer, got {other:?}"),
        }

        let error = build_layers_from_features(&[&broken], "default")
            .err()
            .expect("fully unusable line layer must fail");
        assert!(error.to_string().contains("all 1 line feature"));
    }

    /// One kind's total failure aborts the combined layer build: otherwise the
    /// archive would silently omit that kind while appearing successful.
    #[test]
    fn a_fully_unusable_kind_fails_the_mixed_layer_build() {
        let pt = point_feature(0.0, 0.0, json!({ "ok": 1 }));
        let mut broken = line_feature(vec![[0.0, 0.0], [1.0, 1.0]], 1000, None);
        broken.geojson.geometry = Some(Geometry::new(GeomValue::LineString(vec![])));
        let error = build_layers_from_features(&[&pt, &broken], "default")
            .err()
            .expect("mixed build must not hide a failed geometry kind");
        assert!(error.to_string().contains("all 1 line feature"));
    }

    /// A part whose EXTERIOR ring is unusable is dropped whole — its holes must
    /// not be promoted to exteriors (which would render a hole as a solid).
    #[test]
    fn a_part_without_a_usable_exterior_is_dropped_whole() {
        let f = multipolygon_feature(vec![
            // Unusable exterior (2 positions) + an otherwise valid inner ring.
            vec![vec![[0.0, 0.0], [1.0, 1.0]], square_ring([0.2, 0.2], 0.5)],
            vec![square_ring([10.0, 10.0], 1.0)],
        ]);
        let layers = build_layers_from_features(&[&f], "default").unwrap();
        let rings = match &layers[0].geometry {
            GeometryColumn::Polygon(polys) => &polys[0],
            other => panic!("expected a polygon layer, got {other:?}"),
        };
        assert_eq!(rings.len(), 1, "only the intact part survives: {rings:?}");
        assert_eq!(rings[0][0], [10.0, 10.0]);
        // Single surviving part, single ring ⇒ no triangle sidecar needed.
        assert!(layers[0].triangles.is_none());
    }

    #[test]
    fn pre_tessellate_option_bakes_triangle_indices_per_feature() {
        let p1 = polygon_feature([0.0, 0.0], 1.0);
        let p2 = polygon_feature([5.0, 5.0], 2.0);
        let refs = vec![&p1, &p2];
        let layers = build_layers_from_features_with(
            &refs,
            "default",
            ColumnarOptions {
                pre_tessellate: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(layers.len(), 1);
        let tri = layers[0]
            .triangles
            .as_ref()
            .expect("triangles populated when pre_tessellate is on");
        assert_eq!(tri.len(), 2);
        // Each square produces exactly two triangles → 6 indices.
        assert_eq!(tri[0].len(), 6);
        assert_eq!(tri[1].len(), 6);
        // Indices reference the 5 coords of that feature's exterior ring.
        for &i in &tri[0] {
            assert!(i < 5);
        }
    }

    /// The PART boundaries the geometry column erases must survive into
    /// `ColumnarLayer::polygon_parts` — the value the encoder writes as the
    /// `part_offsets` column. Without them nothing on the wire distinguishes
    /// "part 2's exterior" from "part 1's hole", and every conformant GeoArrow
    /// consumer reads the former as the latter.
    #[test]
    fn multipolygon_layer_carries_its_part_boundaries() {
        // Feature 0: two parts, the first with a hole → ring layout
        // [ext0, hole0, ext1] ⇒ part starts [0, 2].
        let multi = multipolygon_feature(vec![
            vec![square_ring([0.0, 0.0], 4.0), square_ring([1.0, 1.0], 1.0)],
            vec![square_ring([10.0, 10.0], 1.0)],
        ]);
        // Feature 1: a plain single-part Polygon in the SAME layer ⇒ [0].
        let single = polygon_feature([20.0, 20.0], 1.0);
        let layers = build_layers_from_features(&[&multi, &single], "default").unwrap();

        assert_eq!(
            layers[0].polygon_parts.as_deref(),
            Some(&[vec![0u32, 2], vec![0]][..]),
            "ring indices are relative to each feature's own first ring"
        );
        // The part starts must index the geometry column's actual ring runs.
        let rings = match &layers[0].geometry {
            GeometryColumn::Polygon(polys) => polys,
            other => panic!("expected a polygon layer, got {other:?}"),
        };
        assert_eq!(rings[0].len(), 3);
        assert_eq!(rings[0][2][0], [10.0, 10.0], "ring 2 is part 1's exterior");
        assert_eq!(rings[1].len(), 1);
        // …and the layer still validates as an encodable tile layer.
        stt_core::arrow_tile::encode_tile(&layers).expect("multi-part layer must encode");
    }

    /// A layer whose every feature is single-part records nothing: the encoder
    /// then omits the column entirely, and its ABSENCE is what tells a reader
    /// every feature is single-part. Holes do NOT make a feature multi-part.
    #[test]
    fn single_part_polygon_layer_records_no_part_boundaries() {
        let plain = polygon_feature([0.0, 0.0], 1.0);
        let holed = polygon_feature_with_hole();
        let layers = build_layers_from_features(&[&plain, &holed], "default").unwrap();
        assert!(
            layers[0].polygon_parts.is_none(),
            "a single-part layer must not pay for the column"
        );
    }

    // ------------------------------------------------------------------
    // Ids-after-sort and its guard (TB-5 half 1)
    // ------------------------------------------------------------------
    //
    // The two claims under test are the ones a naive wiring breaks:
    //   * an ALL-synthetic point layer gets a dense, MONOTONE stored id column;
    //   * a MIXED point layer — one explicit source id among the minted row
    //     indices — is not touched at all, because renumbering it would
    //     overwrite identity a consumer may be joining on.

    /// A point feature at `lon`, timestamped, with an optional explicit id.
    fn timed_point(lon: f64, timestamp: u64, id: Option<u64>) -> ParsedFeature {
        let mut f = point_feature(lon, 37.0, json!({ "speed": lon }));
        f.timestamp = timestamp;
        f.geojson.id = id.map(|v| geojson::feature::Id::Number(v.into()));
        f
    }

    /// The headline: rows leave sorted by `start_time` and the id column is
    /// `0, 1, .., n-1` IN THAT ORDER — strictly increasing by one, which is the
    /// cheapest column an integer codec can be handed. The incumbent produced
    /// a dense but SHUFFLED permutation, which delta-codes like noise.
    #[test]
    fn all_synthetic_point_layer_gets_dense_monotone_ids_in_stored_order() {
        let feats: Vec<ParsedFeature> = [500u64, 100, 400, 200, 300]
            .into_iter()
            .enumerate()
            .map(|(i, t)| timed_point(i as f64, t, None))
            .collect();
        let refs: Vec<&ParsedFeature> = feats.iter().collect();

        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert_eq!(layers.len(), 1);
        let layer = &layers[0];
        assert_eq!(layer.feature_ids, vec![0, 1, 2, 3, 4]);
        assert!(layer.feature_ids_are_dense_row_order());
        assert_eq!(layer.start_times, vec![100, 200, 300, 400, 500]);
        // Every column travelled with its row: `speed` mirrors the source's
        // lon, and the source order was 500,100,400,200,300.
        match &layer
            .properties
            .iter()
            .find(|(n, _)| n == "speed")
            .unwrap()
            .1
        {
            PropertyColumn::Numeric(v) => assert_eq!(
                v,
                &vec![Some(1.0), Some(3.0), Some(4.0), Some(2.0), Some(0.0)]
            ),
            other => panic!("speed should be numeric, got {other:?}"),
        }
        // And it survives the encoder, which is where the incumbent's sort used
        // to shuffle the ids back apart.
        let encoded = stt_core::arrow_tile::encode_tile(&layers).unwrap();
        let decoded = stt_core::arrow_tile::decode_tile(&encoded).unwrap();
        let ids = decoded[0].batch.column_by_name("id").unwrap();
        let ids = ids
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        assert_eq!(ids.values().to_vec(), vec![0, 1, 2, 3, 4]);
    }

    /// The incumbent, for contrast — and the rollback. With the lever off the
    /// column is the dense-but-shuffled shape, byte-for-byte as before.
    #[test]
    fn synthetic_point_row_ids_off_restores_the_incumbent_shuffled_column() {
        let feats: Vec<ParsedFeature> = [500u64, 100, 400, 200, 300]
            .into_iter()
            .enumerate()
            .map(|(i, t)| timed_point(i as f64, t, None))
            .collect();
        let refs: Vec<&ParsedFeature> = feats.iter().collect();

        let layers = build_layers_from_features_with(
            &refs,
            "default",
            ColumnarOptions {
                synthetic_point_row_ids: false,
                ..Default::default()
            },
        )
        .unwrap();
        // Numbered in SOURCE order, un-sorted — what the encoder then shuffles.
        assert_eq!(layers[0].feature_ids, vec![0, 1, 2, 3, 4]);
        assert_eq!(layers[0].start_times, vec![500, 100, 400, 200, 300]);

        let encoded = stt_core::arrow_tile::encode_tile(&layers).unwrap();
        let decoded = stt_core::arrow_tile::decode_tile(&encoded).unwrap();
        let ids = decoded[0].batch.column_by_name("id").unwrap();
        let ids = ids
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        assert_eq!(
            ids.values().to_vec(),
            vec![1, 3, 4, 2, 0],
            "the incumbent stores a SHUFFLED dense column — the defect, kept \
             reachable as the documented rollback"
        );
    }

    /// **The guard.** One explicit source id among the minted row indices makes
    /// the layer `Keyed`, and a `Keyed` layer is not renumbered and not even
    /// re-sorted — the ids stay exactly what they were, explicit ones included.
    #[test]
    fn mixed_point_layer_is_keyed_and_keeps_every_explicit_id() {
        // Row 0 carries explicit id 900; rows 1..3 carry none.
        let feats = vec![
            timed_point(0.0, 400, Some(900)),
            timed_point(1.0, 100, None),
            timed_point(2.0, 300, None),
            timed_point(3.0, 200, None),
        ];
        let refs: Vec<&ParsedFeature> = feats.iter().collect();
        let layers = build_layers_from_features(&refs, "default").unwrap();
        let layer = &layers[0];

        assert_eq!(
            layer.feature_ids,
            vec![900, 1, 2, 3],
            "the explicit id must survive and the minted ones must stay put"
        );
        assert_eq!(
            layer.start_times,
            vec![400, 100, 300, 200],
            "a Keyed layer's rows must not be re-ordered either"
        );

        // All-keyed is the same answer, arrived at the other way.
        let all_keyed: Vec<ParsedFeature> = [(0.0, 400u64, 7u64), (1.0, 100, 8), (2.0, 300, 9)]
            .into_iter()
            .map(|(lon, t, id)| timed_point(lon, t, Some(id)))
            .collect();
        let refs: Vec<&ParsedFeature> = all_keyed.iter().collect();
        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert_eq!(layers[0].feature_ids, vec![7, 8, 9]);
    }

    /// The pre-existing collision the guard is protecting: explicit source id
    /// `1` against the row index `1` minted for its neighbour. Detected and
    /// named — and NOT renumbered across, because doing so would move both
    /// rows onto new ids and break the picking join for good.
    #[test]
    fn mixed_point_layer_collision_is_detected_and_never_renumbered_across() {
        let feats = vec![
            timed_point(0.0, 300, Some(1)), // explicit 1
            timed_point(1.0, 100, None),    // minted row index 1  ← collision
            timed_point(2.0, 200, None),    // minted row index 2
        ];
        let refs: Vec<&ParsedFeature> = feats.iter().collect();
        let layers = build_layers_from_features(&refs, "default").unwrap();
        let layer = &layers[0];

        assert_eq!(layer.feature_ids, vec![1, 1, 2]);
        assert!(!layer.feature_ids_are_unique());
        assert_eq!(layer.duplicate_feature_ids(), vec![1]);
        // Unchanged rows: the guard held.
        assert_eq!(layer.start_times, vec![300, 100, 200]);

        // The detector, directly.
        assert_eq!(
            mixed_point_id_collisions(&refs, &layer.feature_ids),
            vec![1]
        );
    }

    /// The detector is exact in both directions: an explicit id outside `0..n`,
    /// or one that lands on a row that was NOT minted, is not a collision.
    #[test]
    fn mixed_point_id_collisions_is_exact() {
        // Explicit ids 0 and 2; minted row index 1 only. Neither explicit id
        // collides: 0 and 2 are not minted slots.
        let feats = vec![
            timed_point(0.0, 100, Some(0)),
            timed_point(1.0, 200, None),
            timed_point(2.0, 300, Some(2)),
        ];
        let refs: Vec<&ParsedFeature> = feats.iter().collect();
        assert!(mixed_point_id_collisions(&refs, &[0, 1, 2]).is_empty());

        // A huge explicit id cannot collide with any row index.
        let feats = vec![
            timed_point(0.0, 100, Some(u64::MAX)),
            timed_point(1.0, 200, None),
        ];
        let refs: Vec<&ParsedFeature> = feats.iter().collect();
        assert!(mixed_point_id_collisions(&refs, &[u64::MAX, 1]).is_empty());

        // Explicit id 1 lands on row index 1, which is NOT a minted slot here
        // (row 1 carries an explicit id), so it is not a collision — the
        // detector tests membership of the MINTED set, not merely `id < n`.
        // Explicit id 2 does land on a minted slot, twice, and dedups to one.
        let feats = vec![
            timed_point(0.0, 100, Some(2)),
            timed_point(1.0, 200, Some(1)),
            timed_point(2.0, 300, None), // mints row index 2
            timed_point(3.0, 400, None), // mints row index 3
            timed_point(4.0, 500, Some(2)),
        ];
        let refs: Vec<&ParsedFeature> = feats.iter().collect();
        assert_eq!(mixed_point_id_collisions(&refs, &[2, 1, 2, 3, 2]), vec![2]);

        // Now make slot 1 minted too, and BOTH explicit ids collide: sorted,
        // deduplicated.
        let feats = vec![
            timed_point(0.0, 100, Some(2)),
            timed_point(1.0, 200, None), // mints row index 1
            timed_point(2.0, 300, None), // mints row index 2
            timed_point(3.0, 400, Some(1)),
        ];
        let refs: Vec<&ParsedFeature> = feats.iter().collect();
        assert_eq!(mixed_point_id_collisions(&refs, &[2, 1, 2, 1]), vec![1, 2]);
    }

    /// Lines and polygons are untouched by any of this: their ids must survive
    /// a tile cut, so they are never synthetic and the lever must not reach
    /// them. Two layers of one tile, only the point one is renumbered.
    #[test]
    fn the_lever_reaches_points_only() {
        let p = timed_point(0.0, 500, None);
        let l = line_feature(vec![[0.0, 0.0], [1.0, 1.0]], 100, Some(200));
        let layers = build_layers_from_features(&[&p, &l], "default").unwrap();
        assert_eq!(layers.len(), 2);
        let points = layers
            .iter()
            .find(|layer| layer.name.ends_with("_points"))
            .expect("a point layer");
        let lines = layers
            .iter()
            .find(|layer| layer.name.ends_with("_lines"))
            .expect("a line layer");
        assert_eq!(points.feature_ids, vec![0]);
        // The line's id is still the builder's spec-fixed FNV hash, untouched.
        assert_eq!(lines.feature_ids, vec![determine_feature_id(&l)]);
        assert_ne!(lines.feature_ids[0], 0);
    }

    /// Determinism: the same features build the same layer twice, and the
    /// input order the features arrive in is the only order that reaches the
    /// output (ties in `start_time` keep the caller's order).
    #[test]
    fn ids_after_sort_is_deterministic_and_stable_on_ties() {
        let feats: Vec<ParsedFeature> = [5u64, 5, 5, 5]
            .into_iter()
            .enumerate()
            .map(|(i, t)| timed_point(i as f64, t, None))
            .collect();
        let refs: Vec<&ParsedFeature> = feats.iter().collect();
        let a = build_layers_from_features(&refs, "default").unwrap();
        let b = build_layers_from_features(&refs, "default").unwrap();
        assert_eq!(a[0].feature_ids, b[0].feature_ids);
        assert_eq!(a[0].feature_ids, vec![0, 1, 2, 3]);
        // Ties kept the caller's order, so `speed` (which mirrors lon) is
        // still in source order.
        match &a[0]
            .properties
            .iter()
            .find(|(n, _)| n == "speed")
            .unwrap()
            .1
        {
            PropertyColumn::Numeric(v) => {
                assert_eq!(v, &vec![Some(0.0), Some(1.0), Some(2.0), Some(3.0)])
            }
            other => panic!("speed should be numeric, got {other:?}"),
        }
        assert_eq!(
            stt_core::arrow_tile::encode_tile(&a).unwrap(),
            stt_core::arrow_tile::encode_tile(&b).unwrap()
        );
    }
}
