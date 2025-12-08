//! Tile generation logic

use crate::columnar::{build_columnar_features, ColumnarConfig};
use crate::input::ParsedFeature;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use stt_core::geometry::simplify_linestring_coords;
use stt_core::projection;
use stt_core::tile::TileId;
use stt_core::types::GeometryType;

/// Configuration for tile generation
#[derive(Debug, Clone)]
pub struct TileConfig {
    pub min_zoom: u8,
    pub max_zoom: u8,
    pub extent: u32,
    pub simplification: f64,
    pub layer_name: String,
    /// Target chunk size in bytes (default: 512KB)
    pub target_chunk_size: usize,
    /// Use Version 2 format (quantized coords, columnar properties)
    pub use_v2_format: bool,
}

/// Generated tile with Proto representation
#[derive(Debug)]
pub struct GeneratedTile {
    pub id: TileId,
    pub proto: stt_core::proto::Tile,
}

/// Generate tiles from features
pub fn generate_tiles(
    features: &[ParsedFeature],
    config: &TileConfig,
    workers: usize,
) -> Result<Vec<GeneratedTile>> {
    // Set up parallel processing
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global()
        .ok();

    // Group features by spatial tile (Z, X, Y)
    let mut spatial_map: HashMap<(u8, u32, u32), Vec<&ParsedFeature>> = HashMap::new();

    for feature in features {
        // For each zoom level
        for zoom in config.min_zoom..=config.max_zoom {
            // Calculate spatial tile coordinates using the standardized projection function
            let (x, y) =
                projection::lonlat_to_tile(feature.lon, feature.lat, zoom).unwrap_or((0, 0));

            spatial_map
                .entry((zoom, x, y))
                .or_insert_with(Vec::new)
                .push(feature);
        }
    }

    // Process spatial groups
    let tiles: Vec<GeneratedTile> = spatial_map
        .into_par_iter()
        .flat_map(|((z, x, y), mut features)| {
            // Sort by timestamp
            features.sort_by_key(|f| f.timestamp);

            // Chunk features dynamically based on size
            let chunks = chunk_features_dynamically(&features, config.target_chunk_size);

            let mut generated_tiles = Vec::new();

            for chunk in chunks {
                if chunk.is_empty() {
                    continue;
                }

                let min_time = chunk.first().map(|f| f.timestamp).unwrap_or(0);
                let max_time = chunk.last().map(|f| f.timestamp).unwrap_or(0);

                let tile_id = TileId::new(z, x, y, min_time);

                match create_tile(tile_id, &chunk, config, min_time, max_time) {
                    Ok(tile) => generated_tiles.push(tile),
                    Err(e) => {
                        tracing::warn!("Failed to create tile {:?}: {}", tile_id, e);
                    }
                }
            }

            generated_tiles
        })
        .collect();

    Ok(tiles)
}

/// Chunk features into groups that fit within target size
fn chunk_features_dynamically<'a>(
    features: &[&'a ParsedFeature],
    target_size: usize,
) -> Vec<Vec<&'a ParsedFeature>> {
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_size = 0;

    for feature in features {
        let size = estimate_feature_size(feature);

        // If adding this feature would exceed target size (significantly),
        // and we have enough data, start a new chunk.
        // We check current_size > 0 to ensure we don't create empty chunks or fail on single large features
        if current_size > 0 && current_size + size > target_size {
            chunks.push(current_chunk);
            current_chunk = Vec::new();
            current_size = 0;
        }

        current_chunk.push(*feature);
        current_size += size;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

/// Estimate binary size of a feature (approximate)
fn estimate_feature_size(feature: &ParsedFeature) -> usize {
    // Base overhead (timestamp + IDs + headers)
    let mut size = 100;

    // Geometry estimation (assuming compressed varints)
    if let Some(geom) = &feature.geojson.geometry {
        match &geom.value {
            geojson::Value::Point(_) => size += 32, // 2 coords * 8 bytes + overhead
            geojson::Value::LineString(coords) => size += coords.len() * 16,
            geojson::Value::Polygon(rings) => {
                for ring in rings {
                    size += ring.len() * 16;
                }
            }
            geojson::Value::MultiPoint(coords) => size += coords.len() * 32,
            _ => size += 100,
        }
    }

    // Properties estimation
    if let Some(props) = &feature.geojson.properties {
        // Rough estimate of JSON serialization size
        // In proto they are dictionary encoded so actual size might be smaller,
        // but we want a safe upper bound for chunking.
        size += props.len() * 20;
    }

    size
}

/// Create a single tile from features
fn create_tile(
    tile_id: TileId,
    features: &[&ParsedFeature],
    config: &TileConfig,
    time_start: u64,
    time_end: u64,
) -> Result<GeneratedTile> {
    if config.use_v2_format {
        // Version 2: Columnar format with quantized coordinates
        create_tile_v2(tile_id, features, config, time_start, time_end)
    } else {
        // Version 1: Original format
        create_tile_v1(tile_id, features, config, time_start, time_end)
    }
}

/// Create a tile using Version 1 format (original)
fn create_tile_v1(
    tile_id: TileId,
    features: &[&ParsedFeature],
    config: &TileConfig,
    time_start: u64,
    time_end: u64,
) -> Result<GeneratedTile> {
    let mut proto_features = Vec::with_capacity(features.len());

    for feature in features {
        let proto_feature = build_feature(feature, config)?;
        proto_features.push(proto_feature);
    }

    let layer = stt_core::proto::Layer {
        name: config.layer_name.clone(),
        extent: config.extent,
        features: proto_features,
        columnar: None,
    };

    let proto_tile = stt_core::proto::Tile {
        version: 1,
        time_start,
        time_end,
        layers: vec![layer],
    };

    Ok(GeneratedTile {
        id: tile_id,
        proto: proto_tile,
    })
}

/// Create a tile using Version 2 format (columnar, quantized)
fn create_tile_v2(
    tile_id: TileId,
    features: &[&ParsedFeature],
    config: &TileConfig,
    time_start: u64,
    time_end: u64,
) -> Result<GeneratedTile> {
    let columnar_config = ColumnarConfig {
        extent: config.extent,
        layer_name: config.layer_name.clone(),
    };
    
    let columnar = build_columnar_features(features, &tile_id, &columnar_config, time_start)?;
    
    let layer = stt_core::proto::Layer {
        name: config.layer_name.clone(),
        extent: config.extent,
        features: vec![], // Empty for V2 - data is in columnar
        columnar: Some(columnar),
    };

    let proto_tile = stt_core::proto::Tile {
        version: 2,
        time_start,
        time_end,
        layers: vec![layer],
    };

    Ok(GeneratedTile {
        id: tile_id,
        proto: proto_tile,
    })
}

fn build_feature(feature: &ParsedFeature, config: &TileConfig) -> Result<stt_core::proto::Feature> {
    let (geom_type, positions) = geometry_to_positions(feature, config)?;

    let mut properties = HashMap::new();
    if let Some(props) = &feature.geojson.properties {
        for (key, value) in props {
            if value.is_null() {
                continue;
            }
            properties.insert(key.clone(), json_to_proto_value(value));
        }
    }

    Ok(stt_core::proto::Feature {
        id: determine_feature_id(feature),
        r#type: geom_type_to_proto(geom_type),
        positions,
        properties,
        valid_from: feature.timestamp,
        valid_to: feature.timestamp,
        geometry: vec![], // V2 field - empty for V1 format
    })
}

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

fn geometry_to_positions(
    feature: &ParsedFeature,
    config: &TileConfig,
) -> Result<(GeometryType, Vec<stt_core::proto::Position>)> {
    let geom = feature
        .geojson
        .geometry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Feature has no geometry"))?;

    let (geom_type, coords) = collect_coordinates(geom, config.simplification)?;
    let coords = if coords.is_empty() {
        vec![(feature.lon, feature.lat)]
    } else {
        coords
    };

    let positions = coords
        .into_iter()
        .map(|(lon, lat)| stt_core::proto::Position { lon, lat })
        .collect();

    Ok((geom_type, positions))
}

fn collect_coordinates(
    geom: &geojson::Geometry,
    simplification: f64,
) -> Result<(GeometryType, Vec<(f64, f64)>)> {
    use geojson::Value as GeomValue;

    match &geom.value {
        GeomValue::Point(coords) => {
            if coords.len() < 2 {
                anyhow::bail!("Point missing coordinates");
            }
            Ok((GeometryType::Point, vec![(coords[0], coords[1])]))
        }
        GeomValue::MultiPoint(points) => {
            let coords = points
                .iter()
                .filter(|c| c.len() >= 2)
                .map(|c| (c[0], c[1]))
                .collect();
            Ok((GeometryType::Point, coords))
        }
        GeomValue::LineString(points) => {
            let coords = linestring_to_pairs(points, simplification);
            Ok((GeometryType::LineString, coords))
        }
        GeomValue::MultiLineString(lines) => {
            let mut coords = Vec::new();
            for line in lines {
                coords.extend(linestring_to_pairs(line, simplification));
            }
            Ok((GeometryType::LineString, coords))
        }
        GeomValue::Polygon(rings) => {
            let mut coords = Vec::new();
            for ring in rings {
                coords.extend(linestring_to_pairs(ring, simplification));
            }
            Ok((GeometryType::Polygon, coords))
        }
        GeomValue::MultiPolygon(polygons) => {
            let mut coords = Vec::new();
            for polygon in polygons {
                for ring in polygon {
                    coords.extend(linestring_to_pairs(ring, simplification));
                }
            }
            Ok((GeometryType::Polygon, coords))
        }
        GeomValue::GeometryCollection(collection) => {
            for geom in collection {
                if let Ok((geom_type, coords)) = collect_coordinates(geom, simplification) {
                    if !coords.is_empty() {
                        return Ok((geom_type, coords));
                    }
                }
            }
            Ok((GeometryType::Point, Vec::new()))
        }
    }
}

fn linestring_to_pairs(points: &[Vec<f64>], simplification: f64) -> Vec<(f64, f64)> {
    let mut coords: Vec<(f64, f64)> = points
        .iter()
        .filter(|p| p.len() >= 2)
        .map(|p| (p[0], p[1]))
        .collect();

    if simplification > 0.0 && coords.len() > 2 {
        coords = simplify_linestring_coords(&coords, simplification);
    }

    coords
}

fn geom_type_to_proto(geom_type: GeometryType) -> i32 {
    match geom_type {
        GeometryType::Point => 0,
        GeometryType::LineString => 1,
        GeometryType::Polygon => 2,
    }
}

/// Convert JSON value to proto Value
fn json_to_proto_value(value: &serde_json::Value) -> stt_core::proto::Value {
    use stt_core::proto::value::ValueType;

    let value_type = match value {
        serde_json::Value::String(s) => Some(ValueType::StringValue(s.clone())),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(ValueType::IntValue(i))
            } else if let Some(u) = n.as_u64() {
                Some(ValueType::UintValue(u))
            } else if let Some(f) = n.as_f64() {
                Some(ValueType::DoubleValue(f))
            } else {
                Some(ValueType::StringValue(n.to_string()))
            }
        }
        serde_json::Value::Bool(b) => Some(ValueType::BoolValue(*b)),
        serde_json::Value::Null => Some(ValueType::StringValue(String::new())),
        _ => Some(ValueType::StringValue(value.to_string())),
    };

    stt_core::proto::Value { value_type }
}
