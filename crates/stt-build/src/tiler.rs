//! Tile generation logic

use crate::columnar::{build_columnar_features, ColumnarConfig};
use crate::input::ParsedFeature;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use stt_core::projection;
use stt_core::tile::TileId;

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

/// Create a single tile from features (Version 2 format - columnar, quantized)
fn create_tile(
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
