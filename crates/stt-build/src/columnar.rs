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
    
    // Pre-allocate arrays
    let mut feature_ids: Vec<u64> = Vec::with_capacity(features.len());
    let mut geometry: Vec<i32> = Vec::new();
    let mut geometry_offsets: Vec<u32> = if is_point { vec![] } else { Vec::with_capacity(features.len() + 1) };
    let mut start_times: Vec<i64> = Vec::with_capacity(features.len());
    let mut end_times: Vec<i64> = Vec::with_capacity(features.len());
    
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
            // Non-polygon: flatten coordinates as before
            let coords = extract_coordinates(feature)?;
            for (lon, lat) in coords {
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
    })
}

/// Quantize WGS84 coordinates to tile-relative integers
fn quantize_position(lon: f64, lat: f64, tile: &TileId, extent: u32) -> (i32, i32) {
    let (x, y) = projection::lonlat_to_tile_coords(
        lon, lat,
        tile.z,
        tile.x,
        tile.y,
        extent,
    );
    (x as i32, y as i32)
}

/// Extract coordinates from a feature (flattened - for non-polygon geometries)
fn extract_coordinates(feature: &ParsedFeature) -> Result<Vec<(f64, f64)>> {
    let geom = feature
        .geojson
        .geometry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Feature has no geometry"))?;
    
    use geojson::Value as GeomValue;
    
    let coords = match &geom.value {
        GeomValue::Point(c) if c.len() >= 2 => vec![(c[0], c[1])],
        GeomValue::MultiPoint(points) => points
            .iter()
            .filter(|c| c.len() >= 2)
            .map(|c| (c[0], c[1]))
            .collect(),
        GeomValue::LineString(points) => points
            .iter()
            .filter(|c| c.len() >= 2)
            .map(|c| (c[0], c[1]))
            .collect(),
        GeomValue::MultiLineString(lines) => lines
            .iter()
            .flatten()
            .filter(|c| c.len() >= 2)
            .map(|c| (c[0], c[1]))
            .collect(),
        // For polygons called from non-polygon path (shouldn't happen but handle gracefully)
        GeomValue::Polygon(rings) => rings
            .iter()
            .flatten()
            .filter(|c| c.len() >= 2)
            .map(|c| (c[0], c[1]))
            .collect(),
        GeomValue::MultiPolygon(polygons) => polygons
            .iter()
            .flatten()
            .flatten()
            .filter(|c| c.len() >= 2)
            .map(|c| (c[0], c[1]))
            .collect(),
        _ => vec![(feature.lon, feature.lat)],
    };
    
    if coords.is_empty() {
        Ok(vec![(feature.lon, feature.lat)])
    } else {
        Ok(coords)
    }
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

