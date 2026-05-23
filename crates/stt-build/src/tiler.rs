//! Tile generation: clip trajectories, bucket features spatially and
//! temporally, and emit Arrow [`ColumnarLayer`]s per tile.

use crate::clip::{clip_trajectory, is_clippable_trajectory, ClipConfig, ClippedSegment};
use crate::columnar::{build_layer_from_segments, build_layers_from_features};
use crate::input::ParsedFeature;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use stt_core::arrow_tile::{encode_tile, ColumnarLayer};
use stt_core::projection;
use stt_core::tile::TileId;

/// A generated tile: its identity, temporal span, and Arrow layers.
#[derive(Debug)]
pub struct GeneratedTile {
    /// Tile identity.
    pub id: TileId,
    /// Inclusive temporal start (Unix ms).
    pub time_start: i64,
    /// Inclusive temporal end (Unix ms).
    pub time_end: i64,
    /// One or more Arrow layers (grouped by geometry kind / clip status).
    pub layers: Vec<ColumnarLayer>,
}

impl GeneratedTile {
    /// Total feature count across the tile's layers.
    pub fn feature_count(&self) -> u32 {
        self.layers.iter().map(|l| l.feature_count() as u32).sum()
    }
}

/// Sink for generated tiles (lets the tiler stream into an archive).
pub trait TileWriter {
    /// Persist one tile.
    fn write_tile(&mut self, tile: &GeneratedTile) -> Result<()>;
}

/// Statistics from a tile-generation run.
#[derive(Debug, Default)]
pub struct TileStats {
    /// Total tiles produced.
    pub total_tiles: usize,
    /// Clipped trajectory segments emitted.
    pub clipped_segments: usize,
    /// Un-clipped original features emitted.
    pub original_features: usize,
}

/// Configuration for tile generation.
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Minimum zoom level.
    pub min_zoom: u8,
    /// Maximum zoom level.
    pub max_zoom: u8,
    /// Base layer name.
    pub layer_name: String,
    /// Temporal bucket size (ms) for chunking tiles into aligned intervals.
    pub temporal_bucket_ms: u64,
    /// Whether to clip LineString trajectories at tile boundaries.
    pub clip_trajectories: bool,
    /// Minimum vertices required before a trajectory is clipped.
    pub clip_min_vertices: usize,
    /// Whether to simplify geometry at lower zoom levels.
    pub simplify: bool,
    /// Highest zoom that still receives simplification.
    pub simplify_max_zoom: u8,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            min_zoom: 0,
            max_zoom: 14,
            layer_name: "default".to_string(),
            temporal_bucket_ms: 3600 * 1000,
            clip_trajectories: true,
            clip_min_vertices: 2,
            simplify: false,
            simplify_max_zoom: 14,
        }
    }
}

/// A feature assigned to a tile — either an original feature or a clipped
/// trajectory segment.
#[derive(Debug, Clone)]
enum TileFeature<'a> {
    Original(&'a ParsedFeature),
    Clipped(ClippedSegment),
}

impl<'a> TileFeature<'a> {
    fn timestamp(&self) -> u64 {
        match self {
            TileFeature::Original(f) => f.timestamp,
            TileFeature::Clipped(s) => s.start_time,
        }
    }

    fn end_timestamp(&self) -> u64 {
        match self {
            TileFeature::Original(f) => f.end_timestamp.unwrap_or(f.timestamp),
            TileFeature::Clipped(s) => s.end_time,
        }
    }
}

/// Generate every tile into memory (one zoom level processed at a time).
pub fn generate_tiles(
    features: &[ParsedFeature],
    config: &TileConfig,
    workers: usize,
) -> Result<Vec<GeneratedTile>> {
    init_thread_pool(workers);
    let clip_config = clip_config_from(config);
    let total_clipped = AtomicUsize::new(0);
    let total_original = AtomicUsize::new(0);
    let mut all = Vec::new();

    for zoom in config.min_zoom..=config.max_zoom {
        let start = std::time::Instant::now();
        let tiles = process_zoom_level(
            features,
            zoom,
            config,
            &clip_config,
            &total_clipped,
            &total_original,
        )?;
        tracing::info!(
            "zoom {}: {} tiles in {:.1}s",
            zoom,
            tiles.len(),
            start.elapsed().as_secs_f64()
        );
        all.extend(tiles);
    }
    Ok(all)
}

/// Generate tiles and stream them straight into a [`TileWriter`], bounding
/// memory to a single zoom level at a time.
pub fn generate_tiles_streaming<W: TileWriter + Send>(
    features: &[ParsedFeature],
    config: &TileConfig,
    writer: &mut W,
    workers: usize,
) -> Result<TileStats> {
    init_thread_pool(workers);
    let clip_config = clip_config_from(config);
    let total_clipped = AtomicUsize::new(0);
    let total_original = AtomicUsize::new(0);
    let mut total_tiles = 0;

    for zoom in config.min_zoom..=config.max_zoom {
        let start = std::time::Instant::now();
        let tiles = process_zoom_level(
            features,
            zoom,
            config,
            &clip_config,
            &total_clipped,
            &total_original,
        )?;
        for tile in &tiles {
            writer.write_tile(tile)?;
        }
        total_tiles += tiles.len();
        tracing::info!(
            "zoom {}: {} tiles written in {:.1}s",
            zoom,
            tiles.len(),
            start.elapsed().as_secs_f64()
        );
    }

    Ok(TileStats {
        total_tiles,
        clipped_segments: total_clipped.load(Ordering::Relaxed),
        original_features: total_original.load(Ordering::Relaxed),
    })
}

fn init_thread_pool(workers: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global()
        .ok();
}

fn clip_config_from(config: &TileConfig) -> ClipConfig {
    ClipConfig {
        min_vertices: config.clip_min_vertices,
        buffer_degrees: 0.001,
        temporal_granularity_ms: Some(config.temporal_bucket_ms),
        simplify: config.simplify,
        simplify_max_zoom: config.simplify_max_zoom,
    }
}

/// Process a single zoom level: clip in parallel, bucket spatially then
/// temporally, and build each tile's layers.
fn process_zoom_level(
    features: &[ParsedFeature],
    zoom: u8,
    config: &TileConfig,
    clip_config: &ClipConfig,
    total_clipped: &AtomicUsize,
    total_original: &AtomicUsize,
) -> Result<Vec<GeneratedTile>> {
    // Parallel clip: each feature yields one or more (tile_x, tile_y, feature).
    let placed: Vec<(u32, u32, TileFeature)> = features
        .par_iter()
        .flat_map(|feature| {
            let should_clip = config.clip_trajectories
                && is_clippable_trajectory(&feature.geojson, feature.end_timestamp);
            if should_clip {
                let segments = clip_trajectory(
                    &feature.geojson,
                    feature.shared_properties.clone(),
                    feature.timestamp,
                    feature.end_timestamp.unwrap_or(feature.timestamp),
                    zoom,
                    clip_config,
                );
                if segments.is_empty() {
                    total_original.fetch_add(1, Ordering::Relaxed);
                    let (x, y) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom)
                        .unwrap_or((0, 0));
                    vec![(x, y, TileFeature::Original(feature))]
                } else {
                    total_clipped.fetch_add(segments.len(), Ordering::Relaxed);
                    segments
                        .into_iter()
                        .map(|s| (s.tile_x, s.tile_y, TileFeature::Clipped(s)))
                        .collect()
                }
            } else {
                total_original.fetch_add(1, Ordering::Relaxed);
                let (x, y) =
                    projection::lonlat_to_tile(feature.lon, feature.lat, zoom).unwrap_or((0, 0));
                vec![(x, y, TileFeature::Original(feature))]
            }
        })
        .collect();

    // Group by spatial tile.
    let mut spatial: HashMap<(u32, u32), Vec<TileFeature>> = HashMap::new();
    for (x, y, f) in placed {
        spatial.entry((x, y)).or_default().push(f);
    }

    // Build tiles in parallel: each spatial cell is chunked into temporal
    // buckets, and every (cell, bucket) pair becomes one tile.
    let tiles: Vec<GeneratedTile> = spatial
        .into_par_iter()
        .flat_map(|((x, y), feats)| {
            let buckets = chunk_by_temporal_bucket(feats, config.temporal_bucket_ms);
            let mut out = Vec::new();
            for (bucket_start, chunk) in buckets {
                if chunk.is_empty() {
                    continue;
                }
                let time_end = chunk
                    .iter()
                    .map(|f| f.end_timestamp())
                    .max()
                    .unwrap_or(bucket_start + config.temporal_bucket_ms);
                let id = TileId::new(zoom, x, y, bucket_start);
                match build_tile(id, &chunk, config, bucket_start as i64, time_end as i64) {
                    Ok(Some(tile)) => out.push(tile),
                    Ok(None) => {}
                    Err(e) => tracing::warn!("failed to build tile {id:?}: {e}"),
                }
            }
            out
        })
        .collect();

    Ok(tiles)
}

/// Chunk a spatial cell's features into fixed temporal buckets.
fn chunk_by_temporal_bucket(
    features: Vec<TileFeature>,
    bucket_ms: u64,
) -> Vec<(u64, Vec<TileFeature>)> {
    let bucket_ms = bucket_ms.max(1);
    let mut buckets: BTreeMap<u64, Vec<TileFeature>> = BTreeMap::new();
    for f in features {
        let bucket = (f.timestamp() / bucket_ms) * bucket_ms;
        buckets.entry(bucket).or_default().push(f);
    }
    buckets.into_iter().collect()
}

/// Build one tile's layers from a chunk of features.
fn build_tile(
    id: TileId,
    features: &[TileFeature],
    config: &TileConfig,
    time_start: i64,
    time_end: i64,
) -> Result<Option<GeneratedTile>> {
    let mut originals: Vec<&ParsedFeature> = Vec::new();
    let mut segments: Vec<&ClippedSegment> = Vec::new();
    for f in features {
        match f {
            TileFeature::Original(o) => originals.push(o),
            TileFeature::Clipped(s) => segments.push(s),
        }
    }

    let mut layers: Vec<ColumnarLayer> = Vec::new();

    if !segments.is_empty() {
        layers.push(build_layer_from_segments(&segments, &config.layer_name)?);
    }
    if !originals.is_empty() {
        // Suffix the originals layer name when clipped segments are also
        // present so layer names stay unique within the tile.
        let base = if segments.is_empty() {
            config.layer_name.clone()
        } else {
            format!("{}_originals", config.layer_name)
        };
        layers.extend(build_layers_from_features(&originals, &base)?);
    }

    if layers.is_empty() {
        return Ok(None);
    }
    Ok(Some(GeneratedTile {
        id,
        time_start,
        time_end,
        layers,
    }))
}

/// Stream generated tiles straight into an [`stt_core::archive::ArchiveWriter`].
impl TileWriter for stt_core::archive::ArchiveWriter {
    fn write_tile(&mut self, tile: &GeneratedTile) -> Result<()> {
        let payload = encode_tile(&tile.layers)?;
        self.add_tile(
            &tile.id,
            tile.time_start,
            tile.time_end,
            tile.feature_count(),
            &payload,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geojson::{Feature, Geometry, Value as GeomValue};
    use stt_core::archive::{Archive, ArchiveReader};
    use stt_core::metadata::Metadata;
    use stt_core::types::Compression;

    fn point(lon: f64, lat: f64, ts: u64) -> ParsedFeature {
        let props = serde_json::json!({ "v": ts as f64 })
            .as_object()
            .cloned()
            .map(std::sync::Arc::new);
        ParsedFeature {
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::Point(vec![lon, lat]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: props,
            timestamp: ts,
            end_timestamp: None,
            lon,
            lat,
        }
    }

    fn trajectory(start: u64, end: u64) -> ParsedFeature {
        // A path crossing several tiles near San Francisco.
        let coords: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![-122.5 + i as f64 * 0.02, 37.7 + i as f64 * 0.01])
            .collect();
        let first = coords[0].clone();
        ParsedFeature {
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::LineString(coords))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: start,
            end_timestamp: Some(end),
            lon: first[0],
            lat: first[1],
        }
    }

    /// Full pipeline: features -> tiles -> archive -> read back.
    #[test]
    fn end_to_end_points_archive_roundtrip() {
        let hour = 3_600_000u64;
        // 40 points across two temporal buckets near SF.
        let mut features = Vec::new();
        for i in 0..40u64 {
            let lon = -122.45 + (i % 8) as f64 * 0.01;
            let lat = 37.75 + (i / 8) as f64 * 0.01;
            let ts = 1_600_000_000_000 + (i % 2) * hour + i * 1000;
            features.push(point(lon, lat, ts));
        }

        let config = TileConfig {
            min_zoom: 8,
            max_zoom: 11,
            layer_name: "default".to_string(),
            temporal_bucket_ms: hour,
            clip_trajectories: false,
            ..TileConfig::default()
        };

        let tiles = generate_tiles(&features, &config, 2).unwrap();
        assert!(!tiles.is_empty(), "expected tiles to be generated");

        let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = Archive::create(&path, Compression::Gzip).unwrap();
        for tile in &tiles {
            writer.write_tile(tile).unwrap();
        }
        let total_features: usize =
            tiles.iter().map(|t| t.feature_count() as usize).sum();
        writer.finalize(&Metadata::new("e2e-points")).unwrap();

        let mut reader = ArchiveReader::open(&path).unwrap();
        assert_eq!(reader.entries().len(), tiles.len());

        // Every feature is represented somewhere (summed over all tiles).
        let archived: usize =
            reader.entries().iter().map(|e| e.feature_count as usize).sum();
        assert_eq!(archived, total_features);

        // Decode one tile and confirm its Arrow layer is intact.
        let entry = reader.entries()[0].clone();
        let layers = reader.read_layers(&entry).unwrap();
        assert!(!layers.is_empty());
        assert!(layers[0].batch.num_rows() > 0);
        assert!(layers[0].batch.column_by_name("geometry").is_some());
        assert!(layers[0].batch.column_by_name("v").is_some());
    }

    /// Trajectory clipping produces clipped linestring segments with
    /// per-vertex timestamps that survive the archive roundtrip.
    #[test]
    fn end_to_end_trajectory_clipping() {
        let features = vec![trajectory(1_000_000, 1_000_000 + 3_600_000)];
        let config = TileConfig {
            min_zoom: 9,
            max_zoom: 10,
            layer_name: "tracks".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: true,
            clip_min_vertices: 2,
            ..TileConfig::default()
        };

        let tiles = generate_tiles(&features, &config, 2).unwrap();
        assert!(
            tiles.len() > 1,
            "a multi-tile trajectory should clip into several tiles, got {}",
            tiles.len()
        );

        let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = Archive::create(&path, Compression::Gzip).unwrap();
        for tile in &tiles {
            writer.write_tile(tile).unwrap();
        }
        writer.finalize(&Metadata::new("e2e-tracks")).unwrap();

        let mut reader = ArchiveReader::open(&path).unwrap();
        let entry = reader.entries()[0].clone();
        let layers = reader.read_layers(&entry).unwrap();
        // Clipped segments are linestrings carrying a vertex_time column.
        assert!(layers[0].batch.column_by_name("vertex_time").is_some());
        assert!(layers[0].batch.column_by_name("geometry").is_some());
    }
}
