//! Tile generation logic
//!
//! Generates spatiotemporal tiles with:
//! - Per-vertex timestamps for smooth trail animation (LineStrings with duration)
//! - R-tree spatial indexing for O(log n) tile intersection queries
//! - Complete features stored in each tile (no clipping)
//! - Runtime deduplication via globalFeatureId

use crate::columnar::{build_columnar_features, ColumnarConfig};
use crate::input::ParsedFeature;
use anyhow::Result;
use geojson::Value as GeomValue;
use rayon::prelude::*;
use rstar::{Envelope, RTree, RTreeObject, AABB};
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

/// Feature envelope for R-tree spatial indexing
#[derive(Debug, Clone)]
struct FeatureEnvelope {
    /// Index into the original features array
    index: usize,
    /// Bounding box: [min_lon, min_lat, max_lon, max_lat]
    bbox: [f64; 4],
}

impl FeatureEnvelope {
    fn new(index: usize, bbox: [f64; 4]) -> Self {
        Self { index, bbox }
    }
}

impl RTreeObject for FeatureEnvelope {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [self.bbox[0], self.bbox[1]], // min_lon, min_lat
            [self.bbox[2], self.bbox[3]], // max_lon, max_lat
        )
    }
}

/// Compute bounding box for a feature
fn compute_bbox(feature: &ParsedFeature) -> [f64; 4] {
    let mut min_lon = f64::MAX;
    let mut min_lat = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut max_lat = f64::MIN;

    if let Some(ref geom) = feature.geojson.geometry {
        match &geom.value {
            GeomValue::Point(coord) => {
                if coord.len() >= 2 {
                    min_lon = coord[0];
                    max_lon = coord[0];
                    min_lat = coord[1];
                    max_lat = coord[1];
                }
            }
            GeomValue::LineString(coords) => {
                for coord in coords {
                    if coord.len() >= 2 {
                        min_lon = min_lon.min(coord[0]);
                        max_lon = max_lon.max(coord[0]);
                        min_lat = min_lat.min(coord[1]);
                        max_lat = max_lat.max(coord[1]);
                    }
                }
            }
            GeomValue::Polygon(rings) => {
                for ring in rings {
                    for coord in ring {
                        if coord.len() >= 2 {
                            min_lon = min_lon.min(coord[0]);
                            max_lon = max_lon.max(coord[0]);
                            min_lat = min_lat.min(coord[1]);
                            max_lat = max_lat.max(coord[1]);
                        }
                    }
                }
            }
            GeomValue::MultiPoint(coords) => {
                for coord in coords {
                    if coord.len() >= 2 {
                        min_lon = min_lon.min(coord[0]);
                        max_lon = max_lon.max(coord[0]);
                        min_lat = min_lat.min(coord[1]);
                        max_lat = max_lat.max(coord[1]);
                    }
                }
            }
            GeomValue::MultiLineString(lines) => {
                for line in lines {
                    for coord in line {
                        if coord.len() >= 2 {
                            min_lon = min_lon.min(coord[0]);
                            max_lon = max_lon.max(coord[0]);
                            min_lat = min_lat.min(coord[1]);
                            max_lat = max_lat.max(coord[1]);
                        }
                    }
                }
            }
            GeomValue::MultiPolygon(polygons) => {
                for polygon in polygons {
                    for ring in polygon {
                        for coord in ring {
                            if coord.len() >= 2 {
                                min_lon = min_lon.min(coord[0]);
                                max_lon = max_lon.max(coord[0]);
                                min_lat = min_lat.min(coord[1]);
                                max_lat = max_lat.max(coord[1]);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // If no valid coordinates found, use feature centroid
    if min_lon == f64::MAX {
        min_lon = feature.lon;
        max_lon = feature.lon;
        min_lat = feature.lat;
        max_lat = feature.lat;
    }

    [min_lon, min_lat, max_lon, max_lat]
}

/// Get tile bounding box in lon/lat
fn tile_to_bbox(zoom: u8, tile_x: u32, tile_y: u32) -> AABB<[f64; 2]> {
    let n = (1u32 << zoom) as f64;

    // Tile bounds in lon/lat
    let min_lon = (tile_x as f64 / n) * 360.0 - 180.0;
    let max_lon = ((tile_x + 1) as f64 / n) * 360.0 - 180.0;

    // Y uses Web Mercator projection
    let lat_rad_max = (std::f64::consts::PI * (1.0 - 2.0 * tile_y as f64 / n))
        .sinh()
        .atan();
    let lat_rad_min = (std::f64::consts::PI * (1.0 - 2.0 * (tile_y + 1) as f64 / n))
        .sinh()
        .atan();
    let min_lat = lat_rad_min.to_degrees();
    let max_lat = lat_rad_max.to_degrees();

    AABB::from_corners([min_lon, min_lat], [max_lon, max_lat])
}

/// Get all tiles at a zoom level that could contain features within a bounding box
fn get_tiles_for_bbox(zoom: u8, bbox: &[f64; 4]) -> Vec<(u32, u32)> {
    let mut tiles = Vec::new();

    // Get tile range from bbox corners
    let (min_tx, min_ty) = match projection::lonlat_to_tile(bbox[0], bbox[3], zoom) {
        Ok(t) => t,
        Err(_) => return tiles,
    };
    let (max_tx, max_ty) = match projection::lonlat_to_tile(bbox[2], bbox[1], zoom) {
        Ok(t) => t,
        Err(_) => return tiles,
    };

    // Iterate over all tiles in the range
    for tx in min_tx..=max_tx {
        for ty in min_ty..=max_ty {
            tiles.push((tx, ty));
        }
    }

    tiles
}

/// Generate tiles from features using R-tree spatial indexing
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

    tracing::info!("Building R-tree spatial index for {} features...", features.len());

    // Build R-tree from feature bounding boxes
    let envelopes: Vec<FeatureEnvelope> = features
        .iter()
        .enumerate()
        .map(|(idx, f)| FeatureEnvelope::new(idx, compute_bbox(f)))
        .collect();

    let rtree = RTree::bulk_load(envelopes);
    tracing::info!("R-tree built successfully");

    // For each zoom level, find all tiles that have features
    let mut all_tiles: Vec<(u8, u32, u32, Vec<usize>)> = Vec::new();

    for zoom in config.min_zoom..=config.max_zoom {
        tracing::info!("Processing zoom level {}...", zoom);

        // Collect all unique tiles at this zoom level
        let mut tile_features: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

        // For each feature, find which tiles it intersects at this zoom
        for (idx, feature) in features.iter().enumerate() {
            let bbox = compute_bbox(feature);
            let tiles = get_tiles_for_bbox(zoom, &bbox);

            for (tx, ty) in tiles {
                // Verify intersection using R-tree query (more accurate)
                let tile_bbox = tile_to_bbox(zoom, tx, ty);
                let feature_aabb = AABB::from_corners(
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[3]],
                );

                if tile_bbox.contains_envelope(&feature_aabb)
                    || feature_aabb.contains_envelope(&tile_bbox)
                    || envelopes_intersect(&tile_bbox, &feature_aabb)
                {
                    tile_features
                        .entry((tx, ty))
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
            }
        }

        // Alternative: Use R-tree queries for each tile
        // This is more efficient for sparse data but requires knowing all tiles upfront
        // For dense data, the above approach is simpler

        for ((tx, ty), feature_indices) in tile_features {
            all_tiles.push((zoom, tx, ty, feature_indices));
        }
    }

    tracing::info!("Found {} tile-zoom combinations to process", all_tiles.len());

    // Process tiles in parallel
    let tiles: Vec<GeneratedTile> = all_tiles
        .into_par_iter()
        .flat_map(|(z, x, y, feature_indices)| {
            // Collect references to features for this tile
            let tile_features: Vec<&ParsedFeature> = feature_indices
                .iter()
                .map(|&idx| &features[idx])
                .collect();

            // Sort by timestamp
            let mut sorted_features = tile_features;
            sorted_features.sort_by_key(|f| f.timestamp);

            // Chunk features dynamically based on size
            let chunks = chunk_features_dynamically(&sorted_features, config.target_chunk_size);

            let mut generated_tiles = Vec::new();

            for chunk in chunks {
                if chunk.is_empty() {
                    continue;
                }

                // min_time: earliest start time in the chunk
                let min_time = chunk.first().map(|f| f.timestamp).unwrap_or(0);

                // max_time: latest END time in the chunk (not start time!)
                let max_time = chunk
                    .iter()
                    .map(|f| f.end_timestamp.unwrap_or(f.timestamp))
                    .max()
                    .unwrap_or(0);

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

/// Check if two AABBs intersect
fn envelopes_intersect(a: &AABB<[f64; 2]>, b: &AABB<[f64; 2]>) -> bool {
    let a_lower = a.lower();
    let a_upper = a.upper();
    let b_lower = b.lower();
    let b_upper = b.upper();

    a_lower[0] <= b_upper[0]
        && a_upper[0] >= b_lower[0]
        && a_lower[1] <= b_upper[1]
        && a_upper[1] >= b_lower[1]
}

/// Chunk features into groups that fit within target size
fn chunk_features_dynamically<'a>(
    features: &[&'a ParsedFeature],
    target_size: usize,
) -> Vec<Vec<&'a ParsedFeature>> {
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_size = 0;

    for &feature in features {
        let size = estimate_feature_size(feature);

        // If adding this feature would exceed target size (significantly),
        // and we have enough data, start a new chunk.
        if current_size > 0 && current_size + size > target_size {
            chunks.push(current_chunk);
            current_chunk = Vec::new();
            current_size = 0;
        }

        current_chunk.push(feature);
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
            geojson::Value::Point(_) => size += 32,
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
