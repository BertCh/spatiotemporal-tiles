//! Columnar tile generation for GPU-optimized rendering
//!
//! This module generates tiles with:
//! - Quantized tile-relative coordinates (MVT-style)
//! - Columnar property layout for efficient GPU upload
//! - Delta-encoded timestamps

use crate::input::ParsedFeature;
use anyhow::Result;
use std::collections::HashMap;
use stt_core::projection;
use stt_core::tile::TileId;
use stt_core::types::GeometryType;

/// Configuration for columnar tile generation
#[derive(Debug, Clone)]
pub struct ColumnarConfig {
    pub extent: u32,
    pub layer_name: String,
}

/// Build columnar features from parsed features
pub fn build_columnar_features(
    features: &[&ParsedFeature],
    tile_id: &TileId,
    config: &ColumnarConfig,
    time_start: u64,
) -> Result<stt_core::proto::ColumnarFeatures> {
    let feature_count = features.len() as u32;
    
    if feature_count == 0 {
        return Ok(stt_core::proto::ColumnarFeatures::default());
    }
    
    // Determine geometry type from first feature
    let geometry_type = determine_geometry_type(&features[0])?;
    let is_point = geometry_type == GeometryType::Point;
    let is_polygon = geometry_type == GeometryType::Polygon;
    
    // Check if we have 3D data
    let is_3d = has_3d_coordinates(features);
    
    // Pre-allocate arrays
    let mut feature_ids: Vec<u64> = Vec::with_capacity(features.len());
    let mut geometry: Vec<i32> = Vec::new();
    let mut geometry_offsets: Vec<u32> = if is_point { vec![] } else { Vec::with_capacity(features.len() + 1) };
    let mut start_times: Vec<i64> = Vec::with_capacity(features.len());
    let mut end_times: Vec<i64> = Vec::with_capacity(features.len());
    let mut altitudes: Vec<f32> = if is_3d { Vec::new() } else { vec![] };
    
    // Per-vertex timestamps for smooth trail animation (LineStrings with duration)
    let mut vertex_timestamps: Vec<i64> = Vec::new();
    let is_linestring = geometry_type == GeometryType::LineString;
    
    // Ring offsets for polygon geometries
    let mut ring_offsets: Vec<u32> = Vec::new();
    let mut ring_offsets_offsets: Vec<u32> = if is_polygon { Vec::with_capacity(features.len() + 1) } else { vec![] };
    
    // Collect property names and types from first feature
    let mut numeric_props: HashMap<String, Vec<f32>> = HashMap::new();
    let mut categorical_props: HashMap<String, CategoricalBuilder> = HashMap::new();
    
    if let Some(props) = &features[0].geojson.properties {
        for (key, value) in props {
            if value.is_null() {
                continue;
            }
            if value.is_number() || value.is_f64() {
                numeric_props.insert(key.clone(), Vec::with_capacity(features.len()));
            } else if value.is_string() {
                categorical_props.insert(key.clone(), CategoricalBuilder::new(features.len()));
            }
        }
    }
    
    // Process each feature
    let mut prev_x: i32 = 0;
    let mut prev_y: i32 = 0;
    
    for feature in features {
        // Feature ID
        feature_ids.push(determine_feature_id(feature));
        
        // Geometry offset for non-point
        if !is_point {
            geometry_offsets.push(geometry.len() as u32 / 2);
        }
        
        // Ring offsets offset for polygons
        if is_polygon {
            ring_offsets_offsets.push(ring_offsets.len() as u32);
        }
        
        // Extract and encode coordinates with ring structure for polygons
        if is_polygon {
            let rings = extract_polygon_rings(feature)?;
            for ring in rings {
                // Record the start of this ring
                ring_offsets.push(geometry.len() as u32 / 2);
                
                for (lon, lat) in ring {
                    let (qx, qy) = quantize_position(lon, lat, tile_id, config.extent);
                    
                    // Delta encode
                    let dx = qx - prev_x;
                    let dy = qy - prev_y;
                    
                    geometry.push(dx);
                    geometry.push(dy);
                    
                    prev_x = qx;
                    prev_y = qy;
                }
            }
        } else {
            // Non-polygon: flatten coordinates with optional 3D
            let coords = extract_coordinates_3d(feature)?;
            
            // For LineStrings with duration, compute per-vertex timestamps
            // This enables smooth GPU-based trail animation without path segmentation
            let has_duration = is_linestring && feature.end_timestamp.is_some();
            let duration = if has_duration {
                feature.end_timestamp.unwrap() as i64 - feature.timestamp as i64
            } else {
                0
            };
            
            // Calculate cumulative distances for interpolation
            let cumulative_distances: Vec<f64> = if has_duration && coords.len() > 1 {
                let mut distances = vec![0.0];
                let mut total = 0.0;
                for i in 1..coords.len() {
                    let (lon1, lat1, _) = coords[i - 1];
                    let (lon2, lat2, _) = coords[i];
                    let dist = haversine_distance(lat1, lon1, lat2, lon2);
                    total += dist;
                    distances.push(total);
                }
                distances
            } else {
                vec![]
            };
            
            let total_distance = cumulative_distances.last().copied().unwrap_or(0.0);
            
            for (i, (lon, lat, alt)) in coords.iter().enumerate() {
                let (qx, qy) = quantize_position(*lon, *lat, tile_id, config.extent);
                
                // Delta encode geometry
                let dx = qx - prev_x;
                let dy = qy - prev_y;
                
                geometry.push(dx);
                geometry.push(dy);
                
                if is_3d {
                    altitudes.push(*alt as f32);
                }
                
                // Compute per-vertex timestamp for LineStrings with duration
                if has_duration && total_distance > 0.0 {
                    let fraction = cumulative_distances[i] / total_distance;
                    let vertex_time = feature.timestamp as i64 + (fraction * duration as f64) as i64;
                    let vertex_time_delta = vertex_time - time_start as i64;
                    vertex_timestamps.push(vertex_time_delta);
                } else if is_linestring {
                    // No duration - use feature start time for all vertices
                    let vertex_time_delta = feature.timestamp as i64 - time_start as i64;
                    vertex_timestamps.push(vertex_time_delta);
                }
                
                prev_x = qx;
                prev_y = qy;
            }
        }
        
        // Timestamps (delta encoded relative to tile start)
        let start_delta = feature.timestamp as i64 - time_start as i64;
        start_times.push(start_delta);
        
        // Use end_timestamp if available, otherwise duration is 0
        let end_delta = feature.end_timestamp
            .map(|t| t as i64 - time_start as i64)
            .unwrap_or(start_delta);
        let duration = end_delta - start_delta;
        end_times.push(duration); // Store duration rather than absolute end time
        
        // Properties
        if let Some(props) = &feature.geojson.properties {
            for (key, value) in props {
                if let Some(col) = numeric_props.get_mut(key) {
                    col.push(value.as_f64().unwrap_or(0.0) as f32);
                }
                if let Some(col) = categorical_props.get_mut(key) {
                    col.add(value.as_str().unwrap_or(""));
                }
            }
        }
        
        // Fill missing properties with defaults
        for col in numeric_props.values_mut() {
            if col.len() < feature_ids.len() {
                col.push(0.0);
            }
        }
        for col in categorical_props.values_mut() {
            if col.indices.len() < feature_ids.len() {
                col.add("");
            }
        }
    }
    
    // Final geometry offset
    if !is_point {
        geometry_offsets.push(geometry.len() as u32 / 2);
    }
    
    // Final ring offsets offset for polygons
    if is_polygon {
        ring_offsets_offsets.push(ring_offsets.len() as u32);
    }
    
    // Build proto message
    Ok(stt_core::proto::ColumnarFeatures {
        feature_count,
        geometry_type: geometry_type.to_proto(),
        feature_ids,
        geometry,
        geometry_offsets,
        start_times,
        end_times,
        ring_offsets,
        ring_offsets_offsets,
        numeric_properties: numeric_props
            .into_iter()
            .map(|(name, values)| stt_core::proto::NumericColumn {
                name,
                values,
                values_f64: vec![],
            })
            .collect(),
        categorical_properties: categorical_props
            .into_iter()
            .map(|(name, builder)| stt_core::proto::CategoricalColumn {
                name,
                categories: builder.categories,
                indices: builder.indices,
            })
            .collect(),
        altitudes,
        // Per-vertex timestamps for smooth trail animation (LineStrings with duration)
        // Each vertex has its timestamp as delta from tile time_start
        // This enables GPU-based trail clipping without path segmentation
        vertex_timestamps,
    })
}

/// Quantize WGS84 coordinates to tile-relative integers
/// Note: Returns i32 because coordinates can be outside tile bounds [0, extent]
/// for paths that cross tile boundaries
fn quantize_position(lon: f64, lat: f64, tile: &TileId, extent: u32) -> (i32, i32) {
    projection::lonlat_to_tile_coords(
        lon, lat,
        tile.z,
        tile.x,
        tile.y,
        extent,
    )
}

/// Extract 2D coordinates from a feature (flattened - for non-polygon geometries)
fn extract_coordinates(feature: &ParsedFeature) -> Result<Vec<(f64, f64)>> {
    let coords_3d = extract_coordinates_3d(feature)?;
    Ok(coords_3d.into_iter().map(|(lon, lat, _)| (lon, lat)).collect())
}

/// Extract 3D coordinates from a feature (lon, lat, altitude)
/// Returns (lon, lat, altitude) tuples. Altitude is 0.0 if not present.
fn extract_coordinates_3d(feature: &ParsedFeature) -> Result<Vec<(f64, f64, f64)>> {
    let geom = feature
        .geojson
        .geometry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Feature has no geometry"))?;
    
    use geojson::Value as GeomValue;
    
    // Helper to extract coord with optional altitude
    fn coord_3d(c: &Vec<f64>) -> (f64, f64, f64) {
        let alt = if c.len() >= 3 { c[2] } else { 0.0 };
        (c[0], c[1], alt)
    }
    
    let coords = match &geom.value {
        GeomValue::Point(c) if c.len() >= 2 => vec![coord_3d(c)],
        GeomValue::MultiPoint(points) => points
            .iter()
            .filter(|c| c.len() >= 2)
            .map(|c| coord_3d(c))
            .collect(),
        GeomValue::LineString(points) => points
            .iter()
            .filter(|c| c.len() >= 2)
            .map(|c| coord_3d(c))
            .collect(),
        GeomValue::MultiLineString(lines) => lines
            .iter()
            .flatten()
            .filter(|c| c.len() >= 2)
            .map(|c| coord_3d(c))
            .collect(),
        // For polygons called from non-polygon path (shouldn't happen but handle gracefully)
        GeomValue::Polygon(rings) => rings
            .iter()
            .flatten()
            .filter(|c| c.len() >= 2)
            .map(|c| coord_3d(c))
            .collect(),
        GeomValue::MultiPolygon(polygons) => polygons
            .iter()
            .flatten()
            .flatten()
            .filter(|c| c.len() >= 2)
            .map(|c| coord_3d(c))
            .collect(),
        _ => vec![(feature.lon, feature.lat, 0.0)],
    };
    
    if coords.is_empty() {
        Ok(vec![(feature.lon, feature.lat, 0.0)])
    } else {
        Ok(coords)
    }
}

/// Check if features have 3D coordinates (altitude)
fn has_3d_coordinates(features: &[&ParsedFeature]) -> bool {
    features.iter().any(|f| {
        f.geojson.geometry.as_ref().map_or(false, |geom| {
            use geojson::Value as GeomValue;
            match &geom.value {
                GeomValue::Point(c) => c.len() >= 3 && c[2] != 0.0,
                GeomValue::LineString(points) => points.iter().any(|c| c.len() >= 3 && c[2] != 0.0),
                GeomValue::MultiLineString(lines) => lines.iter().flatten().any(|c| c.len() >= 3 && c[2] != 0.0),
                _ => false,
            }
        })
    })
}

/// Extract polygon rings preserving the ring structure
/// Returns a vector of rings, where each ring is a vector of (lon, lat) coordinates
/// For Polygon: returns all rings (first is exterior, rest are holes)
/// For MultiPolygon: returns all rings from all polygons flattened into one list
fn extract_polygon_rings(feature: &ParsedFeature) -> Result<Vec<Vec<(f64, f64)>>> {
    let geom = feature
        .geojson
        .geometry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Feature has no geometry"))?;
    
    use geojson::Value as GeomValue;
    
    let rings = match &geom.value {
        GeomValue::Polygon(polygon_rings) => {
            polygon_rings
                .iter()
                .map(|ring| {
                    ring.iter()
                        .filter(|c| c.len() >= 2)
                        .map(|c| (c[0], c[1]))
                        .collect()
                })
                .filter(|ring: &Vec<(f64, f64)>| ring.len() >= 4) // Valid ring needs at least 4 points
                .collect()
        }
        GeomValue::MultiPolygon(polygons) => {
            // Flatten all rings from all polygons into a single list
            // deck.gl will interpret this correctly when we provide ring offsets
            polygons
                .iter()
                .flat_map(|polygon| {
                    polygon.iter().map(|ring| {
                        ring.iter()
                            .filter(|c| c.len() >= 2)
                            .map(|c| (c[0], c[1]))
                            .collect()
                    })
                })
                .filter(|ring: &Vec<(f64, f64)>| ring.len() >= 4) // Valid ring needs at least 4 points
                .collect()
        }
        // Fallback for other geometry types that shouldn't reach here
        _ => {
            let coords = extract_coordinates(feature)?;
            if coords.len() >= 4 {
                vec![coords]
            } else {
                vec![]
            }
        }
    };
    
    // If no valid rings found, create a single ring from the centroid point
    if rings.is_empty() {
        // Return a degenerate "ring" with just the feature centroid
        Ok(vec![vec![(feature.lon, feature.lat)]])
    } else {
        Ok(rings)
    }
}

/// Determine geometry type from a feature
fn determine_geometry_type(feature: &ParsedFeature) -> Result<GeometryType> {
    let geom = feature
        .geojson
        .geometry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Feature has no geometry"))?;
    
    use geojson::Value as GeomValue;
    
    Ok(match &geom.value {
        GeomValue::Point(_) | GeomValue::MultiPoint(_) => GeometryType::Point,
        GeomValue::LineString(_) | GeomValue::MultiLineString(_) => GeometryType::LineString,
        GeomValue::Polygon(_) | GeomValue::MultiPolygon(_) => GeometryType::Polygon,
        GeomValue::GeometryCollection(collection) => {
            if let Some(first) = collection.first() {
                match &first.value {
                    GeomValue::Point(_) | GeomValue::MultiPoint(_) => GeometryType::Point,
                    GeomValue::LineString(_) | GeomValue::MultiLineString(_) => GeometryType::LineString,
                    _ => GeometryType::Polygon,
                }
            } else {
                GeometryType::Point
            }
        }
    })
}

/// Determine feature ID (same as original tiler)
fn determine_feature_id(feature: &ParsedFeature) -> u64 {
    use geojson::feature::Id;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    if let Some(id) = &feature.geojson.id {
        match id {
            Id::Number(num) => {
                if let Some(value) = num.as_u64() {
                    return value;
                }
                if let Some(value) = num.as_i64() {
                    return value as u64;
                }
            }
            Id::String(s) => {
                let mut hasher = DefaultHasher::new();
                s.hash(&mut hasher);
                return hasher.finish();
            }
        }
    }

    let mut hasher = DefaultHasher::new();
    hasher.write_u64(feature.timestamp);
    hasher.write_u64(feature.lon.to_bits());
    hasher.write_u64(feature.lat.to_bits());
    hasher.finish()
}

/// Haversine distance between two points in meters
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS: f64 = 6_371_000.0; // meters
    
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    
    let a = (dlat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    
    EARTH_RADIUS * c
}

/// Builder for categorical (string) properties with dictionary encoding
struct CategoricalBuilder {
    categories: Vec<String>,
    category_map: HashMap<String, u8>,
    indices: Vec<u8>,
}

impl CategoricalBuilder {
    fn new(capacity: usize) -> Self {
        Self {
            categories: Vec::new(),
            category_map: HashMap::new(),
            indices: Vec::with_capacity(capacity),
        }
    }
    
    fn add(&mut self, value: &str) {
        let index = if let Some(&idx) = self.category_map.get(value) {
            idx
        } else {
            let idx = self.categories.len() as u8;
            if idx < 255 {
                self.categories.push(value.to_string());
                self.category_map.insert(value.to_string(), idx);
                idx
            } else {
                // Overflow - use last category
                254
            }
        };
        self.indices.push(index);
    }
}

