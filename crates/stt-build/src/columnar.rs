//! Build Arrow [`ColumnarLayer`]s from parsed features and clipped segments.
//!
//! Geometry is stored as real WGS84 lon/lat (`f64`) — no quantization, no
//! delta encoding. Arrow IPC + gzip handle compression, and the payload is
//! consumable directly by GeoArrow-aware renderers.

use crate::clip::ClippedSegment;
use crate::input::ParsedFeature;
use anyhow::Result;
use std::collections::BTreeMap;
use stt_core::arrow_tile::{ColumnarLayer, Coord, GeometryColumn, PropertyColumn};
use stt_core::types::GeometryType;

/// Build layers from a set of features sharing a tile. Features are grouped by
/// geometry type — a single layer holds exactly one geometry kind, so a tile
/// with mixed points and polygons yields one layer per kind.
pub fn build_layers_from_features(
    features: &[&ParsedFeature],
    layer_name: &str,
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
        layers.push(build_polygon_layer(&polygons, name_for("polygons"))?);
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
        geometry.push(coords);
        vertex_times.push(times);

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
        properties: props,
    })
}

fn build_line_layer(features: &[&ParsedFeature], name: String) -> Result<ColumnarLayer> {
    let (ids, start, end, props) = common_columns(features);

    let mut geometry: Vec<Vec<Coord>> = Vec::with_capacity(features.len());
    let mut vertex_times: Vec<Vec<i64>> = Vec::with_capacity(features.len());
    let mut any_duration = false;

    for f in features {
        let coords = extract_line_coords(f)?;
        // Synthesise per-vertex times by distance when the feature has a
        // duration; otherwise every vertex shares the feature start time.
        let times = if let Some(end_ts) = f.end_timestamp {
            any_duration = true;
            interpolate_vertex_times(&coords, f.timestamp, end_ts)
        } else {
            vec![f.timestamp as i64; coords.len()]
        };
        geometry.push(coords);
        vertex_times.push(times);
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
        properties: props,
    })
}

fn build_polygon_layer(features: &[&ParsedFeature], name: String) -> Result<ColumnarLayer> {
    let (ids, start, end, props) = common_columns(features);
    let mut geometry: Vec<Vec<Vec<Coord>>> = Vec::with_capacity(features.len());
    for f in features {
        geometry.push(extract_polygon_rings(f)?);
    }
    Ok(ColumnarLayer {
        name,
        feature_ids: ids,
        start_times: start,
        end_times: end,
        geometry: GeometryColumn::Polygon(geometry),
        vertex_times: None,
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
    /// Discovered columns, in stable (sorted) order.
    numeric: BTreeMap<String, Vec<Option<f64>>>,
    categorical: BTreeMap<String, Vec<Option<String>>>,
    /// True once `push_row` has started; `observe` is then a no-op.
    sealed: bool,
}

impl PropertyAccumulator {
    fn new() -> Self {
        Self {
            numeric: BTreeMap::new(),
            categorical: BTreeMap::new(),
            sealed: false,
        }
    }

    /// First pass: register the keys present on a feature.
    fn observe(&mut self, props: Option<&serde_json::Map<String, serde_json::Value>>) {
        if self.sealed {
            return;
        }
        let Some(props) = props else { return };
        for (key, value) in props {
            if value.is_null() {
                continue;
            }
            if value.is_number() {
                // Numeric wins over categorical for a given key.
                if !self.categorical.contains_key(key) {
                    self.numeric.entry(key.clone()).or_default();
                }
            } else if value.is_string() && !self.numeric.contains_key(key) {
                self.categorical.entry(key.clone()).or_default();
            }
        }
    }

    /// Second pass: append this feature's value for every discovered column.
    fn push_row(&mut self, props: Option<&serde_json::Map<String, serde_json::Value>>) {
        self.sealed = true;
        for (key, col) in self.numeric.iter_mut() {
            let v = props
                .and_then(|p| p.get(key))
                .and_then(|v| v.as_f64());
            col.push(v);
        }
        for (key, col) in self.categorical.iter_mut() {
            let v = props
                .and_then(|p| p.get(key))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
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
}
