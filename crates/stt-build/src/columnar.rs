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
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Arc;
use stt_core::arrow_tile::{
    tessellate_polygon, ColumnarLayer, Coord, GeometryColumn, PropertyColumn,
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
#[derive(Debug, Clone, Default)]
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
            Err(e) => tracing::warn!("skipping feature with no geometry: {e}"),
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
    // A geometry kind whose features are ALL unusable must not take the other
    // kinds down with it: the line/polygon builders fail only when every one of
    // their features had geometry that could not be read (see
    // `extract_or_drop`), which says nothing about the points sharing the tile.
    // Log the loss and keep going — the tiler applies the same rule one level
    // up, and an empty layer list simply collapses the tile.
    if !lines.is_empty() {
        match build_line_layer(&lines, name_for("lines"), &opts) {
            Ok(layer) => layers.push(layer),
            Err(e) => tracing::warn!(
                "layer {layer_name:?}: dropping {} line feature(s); {e}",
                lines.len()
            ),
        }
    }
    if !polygons.is_empty() {
        match build_polygon_layer(&polygons, name_for("polygons"), &opts) {
            Ok(layer) => layers.push(layer),
            Err(e) => tracing::warn!(
                "layer {layer_name:?}: dropping {} polygon feature(s); {e}",
                polygons.len()
            ),
        }
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

        props.observe(seg.properties.as_deref());
    }
    // Second pass fills one value per feature for every discovered property.
    for seg in segments {
        props.push_row(seg.properties.as_deref());
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
    for (i, f) in features.iter().enumerate() {
        if f.geojson.id.is_none() {
            ids[i] = i as u64;
        }
    }
    let geometry: Vec<Coord> = features.iter().map(|f| [f.lon, f.lat]).collect();
    Ok(ColumnarLayer {
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
    })
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
    // ALL-OR-NOTHING PER LAYER, deliberately. The triangle column is 40-45% of
    // the wire bytes of a polygon-heavy tile (measured on storm-field and
    // storm4d-cloudtop), and most of that is spent on single-ring features that
    // every renderer could earcut itself — so it is tempting to emit an EMPTY
    // list for those and keep the baked indices only for the features that need
    // them. That is NOT safe against the current readers, all three of which
    // branch on whether the LAYER has a triangle column and then trust each
    // feature's slice verbatim, with no per-feature fallback:
    //   - packages/layers/.../animated-polygon-layer.ts binds `binary.triangles`
    //     as ONE whole-layer index buffer, so a feature missing from it is
    //     simply never drawn;
    //   - packages/core/src/render/geometry.ts `tessellateFeature` returns the
    //     empty slice (maplibre draws nothing for that feature);
    //   - packages/three/src/layers/polygon-buffers.ts takes its `hasPreBaked`
    //     branch per LAYER and pushes no cap triangles for an empty range.
    // Mixing empty and non-empty lists therefore makes single-ring polygons
    // vanish. Unlocking the saving needs a reader-side change first (the
    // cheapest is a backfill in the decoder: earcut a feature's single ring when
    // its baked list is empty, in packages/core/src/tile.ts) — until then every
    // feature in a triangle-bearing layer gets a real triangulation.
    let needs_triangles = opts.pre_tessellate || parts.iter().any(PolygonParts::needs_triangles);
    let triangles = needs_triangles.then(|| parts.iter().map(tessellate_parts).collect::<Vec<_>>());

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
        props.observe(f.shared_properties.as_deref());
    }
    for f in features {
        props.push_row(f.shared_properties.as_deref());
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
    features: impl IntoIterator<Item = Option<&'a serde_json::Map<String, serde_json::Value>>>,
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
    features: impl IntoIterator<Item = Option<&'a serde_json::Map<String, serde_json::Value>>>,
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
    fn observe(&mut self, props: Option<&serde_json::Map<String, serde_json::Value>>) {
        if self.sealed {
            return;
        }
        let Some(props) = props else { return };
        for (key, value) in props {
            if value.is_null() {
                continue;
            }
            // Opt-in attribute control: a rejected key never enters `seen`, so
            // it yields no column. System columns aren't user properties and
            // never pass through here, so they always survive.
            if !self.filter.keeps(key) {
                continue;
            }
            let kind = self.seen.entry(key.clone()).or_default();
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
    fn push_row(&mut self, props: Option<&serde_json::Map<String, serde_json::Value>>) {
        if !self.sealed {
            self.seal();
        }
        for (key, col) in self.numeric.iter_mut() {
            let v = props.and_then(|p| p.get(key)).and_then(value_as_f64);
            col.push(v);
        }
        for (key, col) in self.categorical.iter_mut() {
            let v = props.and_then(|p| p.get(key)).and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Bool(b) => Some(b.to_string()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            });
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
            feats.iter().map(|f| f.shared_properties.as_deref()),
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
            feats.iter().map(|f| f.shared_properties.as_deref()),
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
            all.iter().map(|f| f.shared_properties.as_deref()),
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
            all.iter().map(|f| f.shared_properties.as_deref()),
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
                .filter(|m| !m.is_empty())
                .map(|m| std::sync::Arc::new(m.clone())),
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
            .map(std::sync::Arc::new);
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
        // when --pre-tessellate is OFF. A single-ring feature sharing the layer
        // is tessellated too (consistent per-tile index buffer).
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
        // The simple square still earcuts to two triangles.
        assert_eq!(tri[1].len(), 6);
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

    /// Contract pin for the triangle sidecar: it is ALL-OR-NOTHING per layer.
    /// Every reader branches on whether the LAYER carries triangles and then
    /// trusts each feature's slice verbatim — deck binds one whole-layer index
    /// buffer, `tessellateFeature` returns the empty slice, three's polygon
    /// buffers push no cap triangles — so a per-feature EMPTY list makes that
    /// polygon disappear. A layer that bakes triangles must bake them for every
    /// feature, single-ring ones included.
    #[test]
    fn every_feature_in_a_triangle_bearing_layer_gets_indices() {
        let multipart = multipolygon_feature(vec![
            vec![square_ring([0.0, 0.0], 1.0)],
            vec![square_ring([5.0, 5.0], 1.0)],
        ]);
        let holed = polygon_feature_with_hole();
        let simple = polygon_feature([20.0, 20.0], 1.0);
        let layers = build_layers_from_features(&[&multipart, &holed, &simple], "default").unwrap();
        let tri = layers[0]
            .triangles
            .as_ref()
            .expect("a layer with a multi-part / holed feature bakes triangles");
        assert_eq!(tri.len(), 3, "one list per feature");
        for (i, t) in tri.iter().enumerate() {
            assert!(
                !t.is_empty() && t.len() % 3 == 0,
                "feature {i} must carry a real triangulation, got {t:?}"
            );
        }
        // The single-ring feature is tessellated too, not left empty.
        assert_eq!(tri[2].len(), 6);
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
            .map(|m| std::sync::Arc::new(m.clone()));
        let mut good = polygon_feature([10.0, 10.0], 1.0);
        good.shared_properties = json!({ "name": "good" })
            .as_object()
            .map(|m| std::sync::Arc::new(m.clone()));

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

        // When NOTHING survives the layer is skipped, not filled with phantoms.
        let empty = build_layers_from_features(&[&broken], "default").unwrap();
        assert!(empty.is_empty(), "no layer is emitted for {empty:?}");
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

        let empty = build_layers_from_features(&[&broken], "default").unwrap();
        assert!(empty.is_empty());
    }

    /// One kind's total failure must not take the other kinds down with it —
    /// the points sharing the tile are perfectly good data.
    #[test]
    fn a_fully_unusable_kind_does_not_drop_the_other_layers() {
        let pt = point_feature(0.0, 0.0, json!({ "ok": 1 }));
        let mut broken = line_feature(vec![[0.0, 0.0], [1.0, 1.0]], 1000, None);
        broken.geojson.geometry = Some(Geometry::new(GeomValue::LineString(vec![])));
        let layers = build_layers_from_features(&[&pt, &broken], "default").unwrap();
        let names: Vec<&str> = layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["default_points"], "the point layer survives");
        assert_eq!(layers[0].feature_count(), 1);
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
}
