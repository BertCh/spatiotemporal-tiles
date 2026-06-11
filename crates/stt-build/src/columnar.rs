//! Build Arrow [`ColumnarLayer`]s from parsed features and clipped segments.
//!
//! Geometry is stored as real WGS84 lon/lat (`f64`) — no quantization, no
//! delta encoding. Arrow IPC + gzip handle compression, and the payload is
//! consumable directly by GeoArrow-aware renderers.

use crate::clip::ClippedSegment;
use crate::input::ParsedFeature;
use anyhow::Result;
use std::collections::BTreeMap;
use stt_core::arrow_tile::{
    tessellate_polygon, ColumnarLayer, Coord, GeometryColumn, PropertyColumn,
};
use stt_core::types::GeometryType;

/// Per-tile build options that influence the columnar layout (independent of
/// the tile-level partitioning logic the tiler owns).
#[derive(Debug, Clone, Copy, Default)]
pub struct ColumnarOptions {
    /// When true, polygon layers will carry pre-baked earcut triangle indices
    /// in a `triangles` sidecar column — letting the renderer skip its own
    /// CPU-side tessellation on tile arrival (MLT-style).
    pub pre_tessellate: bool,
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
    let kinds_present =
        [!points.is_empty(), !lines.is_empty(), !polygons.is_empty()]
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
        layers.push(build_point_layer(&points, name_for("points"))?);
    }
    if !lines.is_empty() {
        layers.push(build_line_layer(&lines, name_for("lines"))?);
    }
    if !polygons.is_empty() {
        layers.push(build_polygon_layer(&polygons, name_for("polygons"), opts)?);
    }
    Ok(layers)
}

/// Build a single linestring layer from clipped trajectory segments. Segments
/// carry real per-vertex timestamps produced by the clipper.
pub fn build_layer_from_segments(
    segments: &[&ClippedSegment],
    layer_name: &str,
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

    let mut props = PropertyAccumulator::new();

    for seg in segments {
        feature_ids.push(segment_feature_id(seg));
        start_times.push(seg.start_time as i64);
        end_times.push(seg.end_time as i64);

        let coords: Vec<Coord> = seg.coordinates.iter().map(|(x, y, _alt)| [*x, *y]).collect();

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
        name: layer_name.to_string(),
        feature_ids,
        start_times,
        end_times,
        geometry: GeometryColumn::LineString(geometry),
        vertex_times: Some(vertex_times),
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

fn build_point_layer(features: &[&ParsedFeature], name: String) -> Result<ColumnarLayer> {
    let (ids, start, end, props) = common_columns(features);
    let geometry: Vec<Coord> = features.iter().map(|f| [f.lon, f.lat]).collect();
    Ok(ColumnarLayer {
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

fn build_line_layer(features: &[&ParsedFeature], name: String) -> Result<ColumnarLayer> {
    let (ids, start, end, props) = common_columns(features);

    let mut geometry: Vec<Vec<Coord>> = Vec::with_capacity(features.len());
    let mut vertex_times: Vec<Vec<i64>> = Vec::with_capacity(features.len());
    let mut vertex_values: Vec<Vec<f32>> = Vec::with_capacity(features.len());
    let mut vertex_value_matrix: Vec<Vec<f32>> = Vec::with_capacity(features.len());
    let mut any_duration = false;
    let mut any_values = false;
    let mut any_matrix = false;
    let mut length_mismatch_warned = false;

    for f in features {
        let coords = extract_line_coords(f)?;
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
        name,
        feature_ids: ids,
        start_times: start,
        end_times: end,
        geometry: GeometryColumn::LineString(geometry),
        // Only attach per-vertex times if at least one feature has a real
        // duration — otherwise they carry no information.
        vertex_times: any_duration.then_some(vertex_times),
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
    opts: ColumnarOptions,
) -> Result<ColumnarLayer> {
    let (ids, start, end, props) = common_columns(features);
    let mut geometry: Vec<Vec<Vec<Coord>>> = Vec::with_capacity(features.len());
    for f in features {
        geometry.push(extract_polygon_rings(f)?);
    }
    // Build the optional triangle index sidecar by running earcut over each
    // feature's rings. The same coords feed both the geometry column and the
    // tessellator — indices are local to the feature.
    let triangles = if opts.pre_tessellate {
        let mut tris: Vec<Vec<u32>> = Vec::with_capacity(geometry.len());
        for rings in &geometry {
            tris.push(tessellate_polygon(rings));
        }
        Some(tris)
    } else {
        None
    };
    Ok(ColumnarLayer {
        name,
        feature_ids: ids,
        start_times: start,
        end_times: end,
        geometry: GeometryColumn::Polygon(geometry),
        vertex_times: None,
        vertex_values: None,
        triangles,
        vertex_value_matrix: None,
        properties: props,
    })
}

/// Build the id / start / end / property columns shared by every layer kind.
fn common_columns(
    features: &[&ParsedFeature],
) -> (Vec<u64>, Vec<i64>, Vec<i64>, Vec<(String, PropertyColumn)>) {
    let mut ids = Vec::with_capacity(features.len());
    let mut start = Vec::with_capacity(features.len());
    let mut end = Vec::with_capacity(features.len());
    let mut props = PropertyAccumulator::new();

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

impl PropertyAccumulator {
    fn new() -> Self {
        Self {
            seen: BTreeMap::new(),
            numeric: BTreeMap::new(),
            categorical: BTreeMap::new(),
            sealed: false,
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
            let kind = self.seen.entry(key.clone()).or_default();
            if value.is_number() {
                kind.has_number = true;
            } else if let Some(s) = value.as_str() {
                if s.trim().parse::<f64>().map(|f| f.is_finite()).unwrap_or(false) {
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

    /// Freeze the schema: a key is numeric iff every observed value was a
    /// number (or a numeric-looking string) and nothing forced it categorical.
    fn seal(&mut self) {
        if self.sealed {
            return;
        }
        self.sealed = true;
        for (key, kind) in &self.seen {
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

/// Extract a flat vertex list for a (multi)linestring feature.
fn extract_line_coords(feature: &ParsedFeature) -> Result<Vec<Coord>> {
    use geojson::Value as GeomValue;
    let geom = feature
        .geojson
        .geometry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("feature has no geometry"))?;
    let coords: Vec<Coord> = match &geom.value {
        GeomValue::LineString(pts) => pts.iter().filter(|c| c.len() >= 2).map(|c| [c[0], c[1]]).collect(),
        GeomValue::MultiLineString(lines) => lines
            .iter()
            .flatten()
            .filter(|c| c.len() >= 2)
            .map(|c| [c[0], c[1]])
            .collect(),
        _ => vec![[feature.lon, feature.lat]],
    };
    if coords.is_empty() {
        Ok(vec![[feature.lon, feature.lat]])
    } else {
        Ok(coords)
    }
}

/// Extract polygon rings (ring 0 is the exterior). MultiPolygon rings are
/// flattened — `ring_offsets` semantics in GeoArrow keep them separable.
fn extract_polygon_rings(feature: &ParsedFeature) -> Result<Vec<Vec<Coord>>> {
    use geojson::Value as GeomValue;
    let geom = feature
        .geojson
        .geometry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("feature has no geometry"))?;
    let to_ring = |ring: &Vec<Vec<f64>>| -> Vec<Coord> {
        ring.iter().filter(|c| c.len() >= 2).map(|c| [c[0], c[1]]).collect()
    };
    let rings: Vec<Vec<Coord>> = match &geom.value {
        GeomValue::Polygon(rings) => rings
            .iter()
            .map(to_ring)
            .filter(|r| r.len() >= 4)
            .collect(),
        GeomValue::MultiPolygon(polys) => polys
            .iter()
            .flat_map(|p| p.iter().map(to_ring))
            .filter(|r| r.len() >= 4)
            .collect(),
        _ => vec![],
    };
    if rings.is_empty() {
        // Degenerate fallback: a zero-area ring at the centroid.
        Ok(vec![vec![[feature.lon, feature.lat]]])
    } else {
        Ok(rings)
    }
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

/// Haversine distance in metres.
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS: f64 = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    EARTH_RADIUS * 2.0 * a.sqrt().asin()
}

// ----------------------------------------------------------------------------
// Feature ids
// ----------------------------------------------------------------------------

/// Resolve a stable u64 feature id (from the GeoJSON id, else a hash).
fn determine_feature_id(feature: &ParsedFeature) -> u64 {
    use geojson::feature::Id;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

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
                let mut h = DefaultHasher::new();
                s.hash(&mut h);
                return h.finish();
            }
        }
    }
    let mut h = DefaultHasher::new();
    h.write_u64(feature.timestamp);
    h.write_u64(feature.lon.to_bits());
    h.write_u64(feature.lat.to_bits());
    h.finish()
}

/// Resolve a stable u64 id for a clipped segment.
fn segment_feature_id(segment: &ClippedSegment) -> u64 {
    use geojson::feature::Id;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

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
                let mut h = DefaultHasher::new();
                s.hash(&mut h);
                return h.finish();
            }
        }
    }
    let mut h = DefaultHasher::new();
    h.write_u64(segment.start_time);
    if let Some((lon, lat, _)) = segment.coordinates.first() {
        h.write_u64(lon.to_bits());
        h.write_u64(lat.to_bits());
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geojson::{Feature, Geometry, Value as GeomValue};
    use serde_json::json;

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
        // Regression guard: the boolean column must be present (not dropped)
        // and carried as a "true"/"false" categorical.
        match col("active").expect("boolean column must be present, not dropped") {
            PropertyColumn::Categorical(v) => {
                assert_eq!(v[0].as_deref(), Some("true"));
                assert_eq!(v[1].as_deref(), Some("false"));
            }
            _ => panic!("boolean should be carried as categorical"),
        }
    }

    /// The keystone producer-drift fix: a property a generator encoded as
    /// numeric *strings* (the line/polygon writer bug that flattened flights'
    /// altitude) must still be classified numeric so it can drive ramps and
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
        let line = line_feature(
            vec![[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            1000,
            Some(3000),
        );
        let refs = vec![&line];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        let vt = layers[0].vertex_times.as_ref().expect("vertex times present");
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

    #[test]
    fn polygon_layer_omits_triangles_by_default() {
        let p = polygon_feature([0.0, 0.0], 1.0);
        let refs = vec![&p];
        let layers = build_layers_from_features(&refs, "default").unwrap();
        assert_eq!(layers.len(), 1);
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
            ColumnarOptions { pre_tessellate: true },
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
}
