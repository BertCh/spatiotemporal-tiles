//! Tile generation: clip trajectories, bucket features spatially and
//! temporally, and emit Arrow [`ColumnarLayer`]s per tile.

use crate::clip::{clip_trajectory, is_clippable_trajectory, ClipConfig, ClippedSegment};
use crate::columnar::{
    build_layer_from_segments, build_layers_from_features_with, AttributeFilter, ColumnarOptions,
};
use crate::input::ParsedFeature;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use stt_core::arrow_tile::{encode_tile, ColumnarLayer};
use stt_core::budget::TileBudget;
use stt_core::projection;
use stt_core::tile::TileId;

/// A generated tile: its identity, temporal span, and Arrow layers.
#[derive(Debug)]
pub struct GeneratedTile {
    /// Tile identity.
    pub id: TileId,
    /// Inclusive temporal start (Unix ms) — the addressable bucket boundary.
    pub time_start: i64,
    /// Inclusive temporal end (Unix ms) — the latest feature end in the tile.
    pub time_end: i64,
    /// Tight lower covering bound: the earliest feature *start* time actually
    /// present in the tile (≤ `time_end`, and may be ≥ or < `time_start`).
    /// Stored in the directory so a client can prune a tile whose data lies
    /// entirely after a query window. See [`stt_core::archive::TileEntry::cover_t_min`].
    pub cover_t_min: i64,
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
    /// When true, polygon layers carry pre-baked earcut triangle indices in a
    /// `triangles` sidecar column — letting the renderer skip its own CPU
    /// tessellation at tile-arrival time.
    pub pre_tessellate: bool,
    /// Optional temporal LOD pyramid. When non-empty, the build emits an
    /// extra aggregate tile per (zoom, spatial cell, lod-bucket) using the
    /// LOD level's `bucket_ms` instead of the base `temporal_bucket_ms`.
    /// Each level applies up to (and including) `max_zoom_level`. Levels
    /// MUST be sorted by ascending bucket size and every bucket MUST be a
    /// multiple of the base bucket.
    pub temporal_lod: Vec<stt_core::metadata::TemporalLodLevel>,
    /// Drop tiles whose feature_count is below this threshold. Default 1
    /// (write every non-empty tile). For globally sparse point datasets, a
    /// threshold like 2 skips the long tail of single-feature deep-zoom
    /// tiles where per-tile Arrow IPC + zstd-frame overhead dominates the
    /// payload. The renderer relies on the tileset's parent-fallback
    /// strategy to surface those features at shallower zooms.
    pub min_features_per_tile: u32,
    /// Use time-aware TD-TR (Synchronized Euclidean Distance) simplification
    /// instead of plain spatial Visvalingam. Preserves per-vertex timing —
    /// important for temporal LOD so zoomed-out playback keeps moving objects
    /// in the right place at the right time.
    pub time_aware_simplify: bool,
    /// When set, replaces fixed `temporal_bucket_ms` chunking with adaptive
    /// windows of ~this many features each: dense periods get fine time windows,
    /// sparse periods coarse ones (the tippecanoe `--maximum-tile-features`
    /// idea applied to the time axis). Each window becomes one tile with its own
    /// `[time_start, time_end]`. In-memory path only (the streaming path keeps
    /// fixed buckets).
    pub adaptive_target_features: Option<u32>,
    /// When set, the named per-feature numeric property is a road-class-style
    /// LOD floor: a feature is SKIPPED at any zoom below its value (vector-tile
    /// "show major roads when zoomed out"). Whole-feature inclusion only — the
    /// feature's geometry/attributes (incl. the value matrix) are untouched.
    /// `None` = no filter (every feature at every zoom in range).
    pub min_zoom_field: Option<String>,
    /// Opt-in per-tile size/feature budget (tippecanoe
    /// `--maximum-tile-bytes`/`--maximum-tile-features`). `None` (the default)
    /// means NO budget — every feature gathered for a tile is emitted, byte-for-
    /// byte identical to a build without the flags. When `Some`, a tile whose
    /// gathered features exceed the cap has its lowest-importance features
    /// dropped to fit (importance-scored, never random), and the exact dropped
    /// count is logged per affected tile. Honours the project's "no thinning by
    /// default" principle: inert unless explicitly opted in.
    pub tile_budget: Option<TileBudget>,
    /// Opt-in user-property selection (`--exclude`/`--include`/`--exclude-all`).
    /// Default [`AttributeFilter::KeepAll`] — every user property kept. System
    /// columns always survive regardless.
    pub attribute_filter: AttributeFilter,
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
            pre_tessellate: false,
            temporal_lod: Vec::new(),
            min_features_per_tile: 1,
            time_aware_simplify: false,
            adaptive_target_features: None,
            min_zoom_field: None,
            tile_budget: None,
            attribute_filter: AttributeFilter::KeepAll,
        }
    }
}

impl TileConfig {
    /// Project to the lower-level `ColumnarOptions` consumed by the columnar
    /// builders. Keeps `tiler` from leaking columnar-level concerns.
    fn columnar_options(&self) -> ColumnarOptions {
        ColumnarOptions {
            pre_tessellate: self.pre_tessellate,
            attribute_filter: self.attribute_filter.clone(),
        }
    }
}

impl TileConfig {
    /// Return the LOD level that applies at `zoom`, if any. Mirrors
    /// `stt_core::metadata::Metadata::temporal_lod_for_zoom` — the coarsest
    /// applicable level wins.
    pub fn lod_for_zoom(&self, zoom: u8) -> Option<&stt_core::metadata::TemporalLodLevel> {
        self.temporal_lod
            .iter()
            .filter(|l| zoom <= l.max_zoom_level)
            .max_by_key(|l| l.bucket_ms)
    }
}

/// A generated tile tagged with the temporal-LOD bucket it represents.
///
/// `bucket_ms == None` means "base tile" (use the archive's
/// `temporal_bucket_ms`). `Some(b)` means this is an aggregate tile produced
/// for an LOD level; the writer records `b` in the directory so the reader
/// can dispatch on bucket size at lookup time.
#[derive(Debug)]
pub struct LodTaggedTile {
    pub tile: GeneratedTile,
    pub temporal_bucket_ms: Option<u64>,
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

/// Generate every tile into memory across all configured zoom levels,
/// returning base tiles + LOD aggregate tiles tagged with their bucket size.
///
/// The base tiles are tagged `Some(config.temporal_bucket_ms)` so the
/// writer can record the bucket size on every directory entry — that's
/// what makes the reader's LOD dispatch possible without ambiguity at the
/// `(z, x, y, t)` lookup level.
///
/// LOD tiles are emitted alongside base tiles at the same spatial cell;
/// each LOD level produces one tile per (zoom, cell, bucket-of-that-level).
pub fn generate_tiles_with_lod(
    features: &[ParsedFeature],
    config: &TileConfig,
    workers: usize,
) -> Result<Vec<LodTaggedTile>> {
    validate_lod(config)?;
    let base = generate_tiles(features, config, workers)?;
    let mut out: Vec<LodTaggedTile> = base
        .into_iter()
        .map(|tile| LodTaggedTile {
            tile,
            temporal_bucket_ms: Some(config.temporal_bucket_ms),
        })
        .collect();
    if !config.temporal_lod.is_empty() {
        let pool = build_pool(workers)?;
        pool.install(|| -> Result<()> {
            for level in &config.temporal_lod {
                let lod_tiles = generate_lod_level(features, level, config)?;
                tracing::info!(
                    "temporal LOD level bucket={}ms max_zoom={}: {} tiles",
                    level.bucket_ms,
                    level.max_zoom_level,
                    lod_tiles.len()
                );
                for tile in lod_tiles {
                    out.push(LodTaggedTile {
                        tile,
                        temporal_bucket_ms: Some(level.bucket_ms),
                    });
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// Emit aggregate tiles for one temporal LOD level.
///
/// The aggregator follows the spatial-summary pattern: features are placed
/// onto the spatial tile grid, then *re-bucketed by the LOD's bucket size*
/// rather than the base. Each (zoom, spatial cell, lod_bucket) becomes one
/// aggregate tile. Within a tile the existing layer builder handles the
/// per-cell aggregation (sum/mean/count fall out of the regrouping for
/// numeric properties).
///
/// The scaffold *re-bucketes only* — feature-level simplification (collapse
/// 1000 points per cell into 50 means) is left as a follow-up; the format
/// already supports it because the per-tile aggregator is plugged in here.
fn generate_lod_level(
    features: &[ParsedFeature],
    level: &stt_core::metadata::TemporalLodLevel,
    base_config: &TileConfig,
) -> Result<Vec<GeneratedTile>> {
    // Reuse the base tile config but override the temporal bucket size for
    // this level, and clamp the zoom range to the level's reach. Clipping +
    // simplification stays on so trajectories that span the LOD bucket are
    // still decomposed cell-by-cell.
    let lod_config = TileConfig {
        temporal_bucket_ms: level.bucket_ms,
        max_zoom: base_config.max_zoom.min(level.max_zoom_level),
        temporal_lod: Vec::new(), // do NOT recurse
        ..base_config.clone()
    };
    if lod_config.max_zoom < lod_config.min_zoom {
        // The LOD level's max_zoom_level falls below the archive's min_zoom;
        // nothing to emit (no spatial zoom in range).
        return Ok(Vec::new());
    }
    let clip_config = clip_config_from(&lod_config);
    let total_clipped = AtomicUsize::new(0);
    let total_original = AtomicUsize::new(0);
    let mut all = Vec::new();
    for zoom in lod_config.min_zoom..=lod_config.max_zoom {
        let tiles = process_zoom_level(
            features,
            zoom,
            &lod_config,
            &clip_config,
            &total_clipped,
            &total_original,
        )?;
        all.extend(tiles);
    }
    Ok(all)
}

/// Validate every level against the archive's base bucket. Mirrors the
/// invariants enforced by `Metadata::with_temporal_lod` so a TileConfig
/// built independently can't slip a bad pyramid past the type checker.
fn validate_lod(config: &TileConfig) -> Result<()> {
    if config.temporal_lod.is_empty() {
        return Ok(());
    }
    let base = config.temporal_bucket_ms;
    anyhow::ensure!(base > 0, "temporal_bucket_ms must be > 0 when using LOD");
    let mut prev: Option<u64> = None;
    for (i, level) in config.temporal_lod.iter().enumerate() {
        anyhow::ensure!(
            level.bucket_ms > base,
            "temporal_lod[{i}].bucket_ms ({}) must be > base bucket ({})",
            level.bucket_ms,
            base
        );
        anyhow::ensure!(
            level.bucket_ms % base == 0,
            "temporal_lod[{i}].bucket_ms ({}) must be a multiple of base ({})",
            level.bucket_ms,
            base
        );
        if let Some(p) = prev {
            anyhow::ensure!(
                level.bucket_ms > p,
                "temporal_lod must be sorted by ascending bucket_ms"
            );
        }
        prev = Some(level.bucket_ms);
    }
    Ok(())
}

/// Generate every tile into memory (one zoom level processed at a time).
pub fn generate_tiles(
    features: &[ParsedFeature],
    config: &TileConfig,
    workers: usize,
) -> Result<Vec<GeneratedTile>> {
    let pool = build_pool(workers)?;
    let clip_config = clip_config_from(config);
    let total_clipped = AtomicUsize::new(0);
    let total_original = AtomicUsize::new(0);
    let mut all = Vec::new();

    // Install the scoped pool for the duration of the build. Anything inside
    // `pool.install(...)` that hits a rayon parallel-iterator runs there
    // rather than the (possibly already-initialised) global pool.
    pool.install(|| -> Result<()> {
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
        Ok(())
    })?;
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
    let pool = build_pool(workers)?;
    let clip_config = clip_config_from(config);
    let total_clipped = AtomicUsize::new(0);
    let total_original = AtomicUsize::new(0);
    let mut total_tiles = 0;

    pool.install(|| -> Result<()> {
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
            let min_features = config.min_features_per_tile.max(1);
            let mut written = 0usize;
            for tile in &tiles {
                if tile.feature_count() < min_features {
                    continue;
                }
                writer.write_tile(tile)?;
                written += 1;
            }
            total_tiles += written;
            tracing::info!(
                "zoom {}: {} tiles written (of {} generated) in {:.1}s",
                zoom,
                written,
                tiles.len(),
                start.elapsed().as_secs_f64()
            );
        }
        Ok(())
    })?;

    Ok(TileStats {
        total_tiles,
        clipped_segments: total_clipped.load(Ordering::Relaxed),
        original_features: total_original.load(Ordering::Relaxed),
    })
}

/// Build a rayon thread pool scoped to a single build run.
///
/// The previous implementation called `build_global()` and silently swallowed
/// the error if some other caller (or a previous build in the same process)
/// had already initialised the global pool, so `--workers N` was effectively
/// ignored after the first run. This builds a fresh local pool so the worker
/// count is always honoured.
fn build_pool(workers: usize) -> Result<rayon::ThreadPool> {
    let threads = workers.max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("stt-build-{i}"))
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build rayon pool: {e}"))
}

fn clip_config_from(config: &TileConfig) -> ClipConfig {
    ClipConfig {
        min_vertices: config.clip_min_vertices,
        buffer_degrees: 0.001,
        // With adaptive temporal windows there's no fixed grid to slice
        // trajectories against, so disable fixed-bucket temporal slicing in that
        // mode; segments are assigned to a window by their start time instead.
        temporal_granularity_ms: if config.adaptive_target_features.is_some() {
            None
        } else {
            Some(config.temporal_bucket_ms)
        },
        simplify: config.simplify,
        simplify_max_zoom: config.simplify_max_zoom,
        time_aware_simplify: config.time_aware_simplify,
    }
}

/// Process a single zoom level: clip in parallel, bucket spatially then
/// temporally, and build each tile's layers.
/// Read a feature's LOD floor from the configured `min_zoom_field` property:
/// the shallowest zoom the feature appears at. `None` = always shown.
fn feature_min_zoom(feature: &ParsedFeature, field: &Option<String>) -> Option<u8> {
    let field = field.as_deref()?;
    feature
        .shared_properties
        .as_ref()?
        .get(field)
        .and_then(|v| v.as_f64())
        .map(|z| z.round() as u8)
}

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
            // Road-class LOD: hide a feature at zooms below its min_zoom floor.
            // Whole-feature skip BEFORE clip — the value matrix is never touched.
            if let Some(mz) = feature_min_zoom(feature, &config.min_zoom_field) {
                if zoom < mz {
                    return Vec::new();
                }
            }
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
                    feature.vertex_timestamps.as_deref(),
                    feature.vertex_values.as_deref(),
                    feature.vertex_value_matrix.as_deref(),
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
            let buckets = match config.adaptive_target_features {
                Some(target) => chunk_adaptive_by_count(feats, target),
                None => chunk_by_temporal_bucket(feats, config.temporal_bucket_ms),
            };
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

/// Adaptive temporal chunking: partition a spatial cell's features into windows
/// of ~`target` features each, ordered by time. Dense periods produce many fine
/// windows, sparse periods few coarse ones. Each window's key is its first
/// feature's timestamp; a window is never closed in the middle of a run of
/// identical timestamps, so the per-window `(zoom, x, y, t)` keys stay distinct.
/// Features sharing one exact timestamp in a cell are inseparable (they map to
/// the same `(z, x, y, t)` key) and stay in a single window even past `target`.
fn chunk_adaptive_by_count(
    mut features: Vec<TileFeature>,
    target: u32,
) -> Vec<(u64, Vec<TileFeature>)> {
    let target = target.max(1) as usize;
    features.sort_by_key(|f| f.timestamp());
    let mut out: Vec<(u64, Vec<TileFeature>)> = Vec::new();
    let mut current: Vec<TileFeature> = Vec::new();
    let mut current_start = 0u64;
    for f in features {
        if current.is_empty() {
            current_start = f.timestamp();
        } else if current.len() >= target
            && f.timestamp() != current.last().unwrap().timestamp()
        {
            // Window is full and the next feature opens a new timestamp — close
            // here so two windows can't share a start time (TileId collision).
            out.push((current_start, std::mem::take(&mut current)));
            current_start = f.timestamp();
        }
        current.push(f);
    }
    if !current.is_empty() {
        out.push((current_start, current));
    }
    out
}

/// Build one tile's layers from a chunk of features.
fn build_tile(
    id: TileId,
    features: &[TileFeature],
    config: &TileConfig,
    time_start: i64,
    time_end: i64,
) -> Result<Option<GeneratedTile>> {
    // Opt-in per-tile budget. Default (`tile_budget: None`) skips this entirely,
    // so a build without `--maximum-tile-bytes`/`--maximum-tile-features` is
    // byte-for-byte identical to before. When a budget is set and the tile
    // exceeds it, the lowest-importance features are dropped to fit and the
    // exact dropped count is logged for THIS tile (no silent truncation).
    let kept_indices = config
        .tile_budget
        .as_ref()
        .map(|budget| apply_tile_budget(budget, id, features));
    // Materialise the surviving feature list only when the budget actually
    // dropped something; otherwise reference the originals in place.
    let kept_features: Vec<&TileFeature> = match &kept_indices {
        Some(keep) if keep.len() < features.len() => {
            keep.iter().map(|&i| &features[i]).collect()
        }
        _ => features.iter().collect(),
    };

    let mut originals: Vec<&ParsedFeature> = Vec::new();
    let mut segments: Vec<&ClippedSegment> = Vec::new();
    for f in &kept_features {
        match f {
            TileFeature::Original(o) => originals.push(o),
            TileFeature::Clipped(s) => segments.push(s),
        }
    }

    let mut layers: Vec<ColumnarLayer> = Vec::new();

    if !segments.is_empty() {
        layers.push(build_layer_from_segments(
            &segments,
            &config.layer_name,
            &config.attribute_filter,
        )?);
    }
    if !originals.is_empty() {
        // Suffix the originals layer name when clipped segments are also
        // present so layer names stay unique within the tile.
        let base = if segments.is_empty() {
            config.layer_name.clone()
        } else {
            format!("{}_originals", config.layer_name)
        };
        layers.extend(build_layers_from_features_with(
            &originals,
            &base,
            config.columnar_options(),
        )?);
    }

    if layers.is_empty() {
        return Ok(None);
    }
    // Tight lower covering bound: earliest feature start actually in the tile
    // (vs `time_start`, the addressable bucket edge). Computed over the KEPT
    // features so a dropped early feature can't widen the bound. Falls back to
    // `time_start` for an (unexpected) empty feature set.
    let cover_t_min = kept_features
        .iter()
        .map(|f| f.timestamp() as i64)
        .min()
        .unwrap_or(time_start);
    Ok(Some(GeneratedTile {
        id,
        time_start,
        time_end,
        cover_t_min,
        layers,
    }))
}

/// Estimated uncompressed payload bytes for one tile feature (geometry + props).
/// Mirrors `budget.rs`'s 16-bytes-per-coordinate-pair + per-property estimate so
/// the byte cap is comparable to `TileBudget::estimate_size`.
fn tile_feature_size(f: &TileFeature) -> usize {
    let (verts, props) = tile_feature_signals(f);
    verts * 16 + props * 16 + 32
}

/// `(vertex_count, property_count)` signals the budget's importance scorer
/// needs, extracted without building an `stt_core::tile::Feature`.
fn tile_feature_signals(f: &TileFeature) -> (usize, usize) {
    match f {
        TileFeature::Original(o) => {
            let verts = geojson_vertex_count(&o.geojson);
            let props = o.shared_properties.as_ref().map(|p| p.len()).unwrap_or(0);
            (verts, props)
        }
        TileFeature::Clipped(s) => {
            let props = s.properties.as_ref().map(|p| p.len()).unwrap_or(0);
            (s.coordinates.len(), props)
        }
    }
}

/// Count the vertices in a GeoJSON feature's geometry (0 when absent).
fn geojson_vertex_count(f: &geojson::Feature) -> usize {
    use geojson::Value as G;
    let Some(geom) = f.geometry.as_ref() else {
        return 1;
    };
    match &geom.value {
        G::Point(_) => 1,
        G::MultiPoint(pts) => pts.len(),
        G::LineString(c) => c.len(),
        G::MultiLineString(lines) => lines.iter().map(|l| l.len()).sum(),
        G::Polygon(rings) => rings.iter().map(|r| r.len()).sum(),
        G::MultiPolygon(polys) => polys.iter().flatten().map(|r| r.len()).sum(),
        G::GeometryCollection(_) => 1,
    }
}

/// Run a tile's gathered features through the budget, returning the indices to
/// KEEP (ascending). Logs the per-tile dropped count whenever anything is
/// dropped — the "no silent caps" guarantee.
fn apply_tile_budget(
    budget: &TileBudget,
    id: TileId,
    features: &[TileFeature],
) -> Vec<usize> {
    let keep = budget.enforce_indexed(
        features.len(),
        |i| {
            let (v, p) = tile_feature_signals(&features[i]);
            budget.score_signals(v, p)
        },
        |i| tile_feature_size(&features[i]),
    );
    let dropped = features.len() - keep.len();
    if dropped > 0 {
        tracing::warn!(
            "tile z{} x{} y{} t{}: dropped {} of {} features to fit budget \
             (max_features={}, max_bytes={})",
            id.z,
            id.x,
            id.y,
            id.t,
            dropped,
            features.len(),
            budget.max_feature_count,
            budget.max_uncompressed_size,
        );
    }
    keep
}

/// Stream generated tiles straight into an [`stt_core::archive::ArchiveWriter`].
impl TileWriter for stt_core::archive::ArchiveWriter {
    fn write_tile(&mut self, tile: &GeneratedTile) -> Result<()> {
        let payload = encode_tile(&tile.layers)?;
        self.add_tile_full(
            &tile.id,
            tile.time_start,
            tile.time_end,
            Some(tile.cover_t_min),
            tile.feature_count(),
            None,
            &payload,
        )?;
        Ok(())
    }
}

/// Stream generated tiles straight into a packed-format [`stt_core::PackWriter`].
///
/// Identical mapping to the [`stt_core::archive::ArchiveWriter`] impl above —
/// `PackWriter` shares the same `add_tile_full` contract; it just buffers the
/// tiles and cuts them into content-addressed packs at finalize.
impl TileWriter for stt_core::PackWriter {
    fn write_tile(&mut self, tile: &GeneratedTile) -> Result<()> {
        let payload = encode_tile(&tile.layers)?;
        self.add_tile_full(
            &tile.id,
            tile.time_start,
            tile.time_end,
            Some(tile.cover_t_min),
            tile.feature_count(),
            None,
            &payload,
        )?;
        Ok(())
    }
}

/// Sink that also forwards the per-tile temporal bucket size.
pub trait LodTileWriter {
    /// Persist one tile, tagging the directory entry with `temporal_bucket_ms`.
    fn write_lod_tile(
        &mut self,
        tile: &GeneratedTile,
        temporal_bucket_ms: Option<u64>,
    ) -> Result<()>;
}

impl LodTileWriter for stt_core::archive::ArchiveWriter {
    fn write_lod_tile(
        &mut self,
        tile: &GeneratedTile,
        temporal_bucket_ms: Option<u64>,
    ) -> Result<()> {
        let payload = encode_tile(&tile.layers)?;
        self.add_tile_full(
            &tile.id,
            tile.time_start,
            tile.time_end,
            Some(tile.cover_t_min),
            tile.feature_count(),
            temporal_bucket_ms,
            &payload,
        )?;
        Ok(())
    }
}

impl LodTileWriter for stt_core::PackWriter {
    fn write_lod_tile(
        &mut self,
        tile: &GeneratedTile,
        temporal_bucket_ms: Option<u64>,
    ) -> Result<()> {
        let payload = encode_tile(&tile.layers)?;
        self.add_tile_full(
            &tile.id,
            tile.time_start,
            tile.time_end,
            Some(tile.cover_t_min),
            tile.feature_count(),
            temporal_bucket_ms,
            &payload,
        )?;
        Ok(())
    }
}

// =============================================================================
// True-streaming pipeline: input -> partitioned tile accumulators -> writer.
// =============================================================================

/// Owned counterpart of `TileFeature` — needed by the streaming pipeline
/// because the source `ParsedFeature` batches are dropped as we move on to
/// the next batch, so we can't borrow into them.
#[derive(Debug)]
enum OwnedTileFeature {
    Original(ParsedFeature),
    Clipped(ClippedSegment),
}

impl OwnedTileFeature {
    fn start_timestamp(&self) -> u64 {
        match self {
            OwnedTileFeature::Original(f) => f.timestamp,
            OwnedTileFeature::Clipped(s) => s.start_time,
        }
    }
    fn end_timestamp(&self) -> u64 {
        match self {
            OwnedTileFeature::Original(f) => f.end_timestamp.unwrap_or(f.timestamp),
            OwnedTileFeature::Clipped(s) => s.end_time,
        }
    }
    /// Rough byte estimate for the per-tile spill budget. Numbers are
    /// deliberately conservative (over-estimating) so the streaming
    /// pipeline flushes earlier rather than later — under-sizing a tile
    /// only costs an extra writer call.
    fn estimated_size(&self) -> usize {
        match self {
            OwnedTileFeature::Original(f) => {
                // geojson::Feature (~120 B carrier) + ParsedFeature fields
                // (~40 B) + per-property entry (~32 B amortised). For a
                // Point we're typically at ~180 B; for a LineString add
                // the coord vector cost.
                let mut s = 200usize;
                if let Some(geom) = f.geojson.geometry.as_ref() {
                    use geojson::Value as G;
                    match &geom.value {
                        G::LineString(c) => s += c.len() * 32,
                        G::MultiLineString(lines) => {
                            s += lines.iter().map(|l| l.len() * 32).sum::<usize>()
                        }
                        G::Polygon(rings) => s += rings.iter().map(|r| r.len() * 32).sum::<usize>(),
                        G::MultiPolygon(polys) => {
                            s += polys
                                .iter()
                                .flat_map(|p| p.iter())
                                .map(|r| r.len() * 32)
                                .sum::<usize>()
                        }
                        _ => {}
                    }
                }
                if let Some(props) = &f.shared_properties {
                    s += props.len() * 48;
                }
                s
            }
            OwnedTileFeature::Clipped(s) => 96 + s.coordinates.len() * 32 + s.timestamps.len() * 8,
        }
    }
}

/// Per-tile accumulator: holds owned features until size budget or input EOF
/// forces a flush.
#[derive(Debug, Default)]
struct TileBucket {
    features: Vec<OwnedTileFeature>,
    bytes: usize,
}

/// Build features straight off a streaming Parquet input. Tile accumulators
/// are keyed by `(zoom, tile_x, tile_y, time_bucket)`; when an accumulator
/// reaches `spill_bytes` it's flushed through the writer. When the input
/// ends, every remaining accumulator is flushed.
///
/// Memory bound = (one Parquet batch) + (total bytes across all live
/// accumulators), capped per-accumulator at `spill_bytes`. There is no
/// `Vec<GeneratedTile>` held in RAM — tiles go straight to the writer.
///
/// `external_sort` controls whether the writer's directory is left sorted by
/// (zoom, hilbert) at finalize time (the default `ArchiveWriter` behaviour)
/// or whether this function pre-flushes per-zoom so the writer's stream is
/// already sorted and no global tile-set sort happens.
pub fn build_streaming_from_batches<W, I>(
    batches: I,
    config: &TileConfig,
    writer: &mut W,
    _workers: usize,
    spill_bytes: usize,
) -> Result<TileStats>
where
    W: TileWriter,
    I: IntoIterator<Item = Result<Vec<ParsedFeature>>>,
{
    let clip_config = clip_config_from(config);
    let total_clipped = AtomicUsize::new(0);
    let total_original = AtomicUsize::new(0);
    let mut total_tiles = 0usize;

    // One accumulator map per (zoom, tile, bucket). Per-zoom accumulators
    // are kept in a single BTreeMap so the streaming writer can emit them
    // pre-sorted by (zoom, tile_y, tile_x, bucket) — the writer's
    // finalize() will then re-sort by Hilbert with no extra global pass.
    type ZoomBuckets = BTreeMap<(u32, u32, u64), TileBucket>;
    let mut by_zoom: Vec<ZoomBuckets> =
        (0..=(config.max_zoom.saturating_sub(config.min_zoom)) as usize)
            .map(|_| BTreeMap::new())
            .collect();

    let bucket_ms = config.temporal_bucket_ms.max(1);
    let zooms: Vec<u8> = (config.min_zoom..=config.max_zoom).collect();
    let spill = spill_bytes.max(1024);

    let mut process = |features: Vec<ParsedFeature>,
                       by_zoom: &mut Vec<ZoomBuckets>|
     -> Result<()> {
        // Parallel clip+assign per zoom. We don't go batch-into-rayon
        // (batches are usually small enough that overhead dominates) — we
        // parallelise the zoom loop inner with par_iter at the per-feature
        // level via the existing pool. For simplicity and determinism the
        // streaming path is serial here; the heavy work (encode + compress)
        // still happens on the writer side.
        for (zi, &zoom) in zooms.iter().enumerate() {
            for feature in &features {
                // Road-class LOD: hide a feature below its min_zoom floor.
                if let Some(mz) = feature_min_zoom(feature, &config.min_zoom_field) {
                    if zoom < mz {
                        continue;
                    }
                }
                let should_clip = config.clip_trajectories
                    && is_clippable_trajectory(&feature.geojson, feature.end_timestamp);
                if should_clip {
                    let segments = clip_trajectory(
                        &feature.geojson,
                        feature.shared_properties.clone(),
                        feature.timestamp,
                        feature.end_timestamp.unwrap_or(feature.timestamp),
                        zoom,
                        &clip_config,
                        feature.vertex_timestamps.as_deref(),
                        feature.vertex_values.as_deref(),
                        feature.vertex_value_matrix.as_deref(),
                    );
                    if segments.is_empty() {
                        let (x, y) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom)
                            .unwrap_or((0, 0));
                        let bucket = (feature.timestamp / bucket_ms) * bucket_ms;
                        push_feature(
                            &mut by_zoom[zi],
                            zoom,
                            (x, y, bucket),
                            OwnedTileFeature::Original(feature.clone()),
                            spill,
                            config,
                            writer,
                            &mut total_tiles,
                        )?;
                        total_original.fetch_add(1, Ordering::Relaxed);
                    } else {
                        total_clipped.fetch_add(segments.len(), Ordering::Relaxed);
                        for seg in segments {
                            let bucket = (seg.start_time / bucket_ms) * bucket_ms;
                            let key = (seg.tile_x, seg.tile_y, bucket);
                            push_feature(
                                &mut by_zoom[zi],
                                zoom,
                                key,
                                OwnedTileFeature::Clipped(seg),
                                spill,
                                config,
                                writer,
                                &mut total_tiles,
                            )?;
                        }
                    }
                } else {
                    let (x, y) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom)
                        .unwrap_or((0, 0));
                    let bucket = (feature.timestamp / bucket_ms) * bucket_ms;
                    push_feature(
                        &mut by_zoom[zi],
                        zoom,
                        (x, y, bucket),
                        OwnedTileFeature::Original(feature.clone()),
                        spill,
                        config,
                        writer,
                        &mut total_tiles,
                    )?;
                    total_original.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    };

    for batch in batches {
        let batch = batch?;
        process(batch, &mut by_zoom)?;
    }

    // Flush all remaining accumulators, zoom-by-zoom, in
    // (tile_y, tile_x, bucket) order. The writer's finalize() will
    // re-sort the directory by (zoom, hilbert) for spatial locality —
    // since each zoom is contiguous in our stream and tiles within a
    // zoom hit the writer pre-sorted, this is effectively an
    // external-partition step.
    for (zi, &zoom) in zooms.iter().enumerate() {
        let map = std::mem::take(&mut by_zoom[zi]);
        for ((x, y, bucket), bucket_data) in map {
            flush_bucket(zoom, x, y, bucket, bucket_data, config, writer, &mut total_tiles)?;
        }
    }

    Ok(TileStats {
        total_tiles,
        clipped_segments: total_clipped.load(Ordering::Relaxed),
        original_features: total_original.load(Ordering::Relaxed),
    })
}

/// Push a feature into its accumulator and flush if the byte budget is hit.
#[allow(clippy::too_many_arguments)]
fn push_feature<W: TileWriter>(
    buckets: &mut BTreeMap<(u32, u32, u64), TileBucket>,
    zoom: u8,
    key: (u32, u32, u64),
    feat: OwnedTileFeature,
    spill: usize,
    config: &TileConfig,
    writer: &mut W,
    total_tiles: &mut usize,
) -> Result<()> {
    let size = feat.estimated_size();
    let bucket = buckets.entry(key).or_default();
    bucket.features.push(feat);
    bucket.bytes += size;
    if bucket.bytes >= spill {
        let drained = std::mem::take(bucket);
        buckets.remove(&key);
        flush_bucket(zoom, key.0, key.1, key.2, drained, config, writer, total_tiles)?;
    }
    Ok(())
}

/// Build one tile from a drained bucket and write it.
fn flush_bucket<W: TileWriter>(
    zoom: u8,
    x: u32,
    y: u32,
    bucket_start: u64,
    bucket: TileBucket,
    config: &TileConfig,
    writer: &mut W,
    total_tiles: &mut usize,
) -> Result<()> {
    if bucket.features.is_empty() {
        return Ok(());
    }
    let time_end = bucket
        .features
        .iter()
        .map(|f| f.end_timestamp())
        .max()
        .unwrap_or(bucket_start + config.temporal_bucket_ms);
    let cover_t_min = bucket
        .features
        .iter()
        .map(|f| f.start_timestamp() as i64)
        .min()
        .unwrap_or(bucket_start as i64);
    let id = TileId::new(zoom, x, y, bucket_start);

    // Split into originals + clipped segments and reuse the existing
    // layer builders so the streaming path emits exactly the same
    // ColumnarLayer shape as the in-memory path.
    let mut originals: Vec<&ParsedFeature> = Vec::new();
    let mut segments: Vec<&ClippedSegment> = Vec::new();
    for f in &bucket.features {
        match f {
            OwnedTileFeature::Original(o) => originals.push(o),
            OwnedTileFeature::Clipped(s) => segments.push(s),
        }
    }

    // NOTE: the streaming-arrow path does not honour per-tile budgets or
    // attribute control (main.rs rejects the combination up front). The
    // `attribute_filter` here is therefore always `KeepAll` — passed only so
    // the same layer builders compile/behave identically.
    let mut layers: Vec<ColumnarLayer> = Vec::new();
    if !segments.is_empty() {
        layers.push(crate::columnar::build_layer_from_segments(
            &segments,
            &config.layer_name,
            &config.attribute_filter,
        )?);
    }
    if !originals.is_empty() {
        let base = if segments.is_empty() {
            config.layer_name.clone()
        } else {
            format!("{}_originals", config.layer_name)
        };
        layers.extend(crate::columnar::build_layers_from_features_with(
            &originals,
            &base,
            config.columnar_options(),
        )?);
    }
    if layers.is_empty() {
        return Ok(());
    }
    let tile = GeneratedTile {
        id,
        time_start: bucket_start as i64,
        time_end: time_end as i64,
        cover_t_min,
        layers,
    };
    if tile.feature_count() < config.min_features_per_tile.max(1) {
        return Ok(());
    }
    writer.write_tile(&tile)?;
    *total_tiles += 1;
    Ok(())
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
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
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
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: first[0],
            lat: first[1],
        }
    }

    /// A static-geometry corridor carrying a per-vertex × per-bucket value
    /// matrix must build into ONE tile per spatial cell spanning the WHOLE
    /// range — never fragmented across temporal buckets by its interpolated
    /// vertex times — so the client loads its geometry once and animates the
    /// resident matrix. (The build bucket here is small enough that, without
    /// the matrix time-pin, the corridor would fragment into several tiles.)
    #[test]
    fn matrix_corridor_builds_one_tile_spanning_range() {
        let num_buckets = 4usize;
        let bucket_ms = 900_000u64; // 15 min
        let start = 1_420_070_400_000u64;
        let end = start + num_buckets as u64 * bucket_ms;
        // 3 vertices kept inside a single zoom-10 tile.
        let coords: Vec<Vec<f64>> = vec![
            vec![-73.980, 40.750],
            vec![-73.979, 40.751],
            vec![-73.978, 40.752],
        ];
        let nverts = coords.len();
        let first = coords[0].clone();
        // Flat vertex-major matrix: nverts * num_buckets.
        let matrix: Vec<f32> = (0..nverts * num_buckets).map(|i| i as f32).collect();
        let feature = ParsedFeature {
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
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: Some(matrix),
            lon: first[0],
            lat: first[1],
        };
        let config = TileConfig {
            min_zoom: 10,
            max_zoom: 10,
            layer_name: "flows".to_string(),
            temporal_bucket_ms: bucket_ms,
            clip_trajectories: true,
            clip_min_vertices: 2,
            ..TileConfig::default()
        };

        let tiles = generate_tiles(&[feature], &config, 1).unwrap();
        assert_eq!(
            tiles.len(),
            1,
            "matrix corridor must build exactly one tile, got {}",
            tiles.len()
        );
        let tile = &tiles[0];
        // Spans the whole range so its time window matches every playback frame.
        assert_eq!(tile.time_start, start as i64);
        assert_eq!(tile.time_end, end as i64);
        // The matrix survived clipping into the tile's columnar layer.
        let layer = &tile.layers[0];
        let vm = layer
            .vertex_value_matrix
            .as_ref()
            .expect("tile layer must carry the per-vertex value matrix");
        assert_eq!(vm.len(), 1);
        assert_eq!(vm[0].len(), nverts * num_buckets);
    }

    /// The tight covering lower bound `cover_t_min` is the earliest feature
    /// START in a tile — strictly after the bucket-aligned `time_start` when the
    /// data sits late in the bucket — and survives build → write → read.
    #[test]
    fn cover_t_min_tracks_earliest_feature_through_build_and_read() {
        let hour = 3_600_000u64;
        let base = 1_600_000_000_000u64;
        // All points land in the SECOND half of their hour bucket, so the tight
        // lower bound is well after the bucket edge.
        let mut features = Vec::new();
        for i in 0..12u64 {
            let lon = -122.45 + i as f64 * 0.02; // spread across tiles
            let ts = base + hour / 2 + i * 1000;
            features.push(point(lon, 37.75, ts));
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
        assert!(!tiles.is_empty());
        // Every tile's covering bound is ≥ its bucket edge, and at least one is
        // strictly tighter (the whole point of the lever).
        assert!(tiles.iter().all(|t| t.cover_t_min >= t.time_start));
        assert!(
            tiles.iter().any(|t| t.cover_t_min > t.time_start),
            "expected a tile whose earliest feature is after the bucket edge"
        );

        let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = Archive::create(&path, Compression::Zstd).unwrap();
        for tile in &tiles {
            writer.write_tile(tile).unwrap();
        }
        writer.finalize(&Metadata::new("cover")).unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        // The covering section round-trips: every entry carries a bound and at
        // least one is tighter than its bucket edge.
        assert!(reader.entries().iter().all(|e| e.cover_t_min.is_some()));
        assert!(reader
            .entries()
            .iter()
            .any(|e| e.cover_t_min.unwrap() > e.time_start));
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
        let mut writer = Archive::create(&path, Compression::Zstd).unwrap();
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
        let mut writer = Archive::create(&path, Compression::Zstd).unwrap();
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

    /// Streaming pipeline emits the same tiles as the in-memory pipeline
    /// when fed identical features in chunks. This is the unit-level
    /// correctness check for `build_streaming_from_batches`; a larger
    /// peak-RSS test lives at `tests/streaming_peak_rss.rs` (uses
    /// platform `getrusage`).
    #[test]
    fn streaming_matches_in_memory_for_points() {
        let hour = 3_600_000u64;
        let mut features = Vec::new();
        for i in 0..200u64 {
            let lon = -122.45 + (i % 16) as f64 * 0.01;
            let lat = 37.75 + (i / 16) as f64 * 0.005;
            let ts = 1_600_000_000_000 + (i % 3) * hour + i * 1000;
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

        // Reference: in-memory pipeline.
        let in_mem_tiles = generate_tiles(&features, &config, 2).unwrap();
        let in_mem_total: usize = in_mem_tiles.iter().map(|t| t.feature_count() as usize).sum();

        // Streaming: feed the same features as four batches.
        let mut batches: Vec<Result<Vec<ParsedFeature>>> = Vec::new();
        for chunk in features.chunks(50) {
            batches.push(Ok(chunk.to_vec()));
        }
        let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = Archive::create(&path, Compression::Zstd).unwrap();
        let stats =
            build_streaming_from_batches(batches.into_iter(), &config, &mut writer, 2, 1024 * 1024)
                .unwrap();
        writer.finalize(&Metadata::new("streaming")).unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        let archived: usize = reader.entries().iter().map(|e| e.feature_count as usize).sum();

        // Stream and in-mem totals must match exactly (no features dropped,
        // no double-counting).
        assert_eq!(archived, in_mem_total, "feature count mismatch");
        // Streaming may emit slightly more tiles than the in-mem pipeline
        // when buckets spill mid-flight (a flush-then-refill sequence
        // becomes two tiles for the same (zoom, x, y) but different
        // time-spans). With a 1 MB spill and 200 features that should not
        // happen, but allow up to a few extra to keep this robust.
        assert!(
            stats.total_tiles >= in_mem_tiles.len()
                && stats.total_tiles <= in_mem_tiles.len() + 4,
            "tile count diverged: stream={} in_mem={}",
            stats.total_tiles,
            in_mem_tiles.len()
        );
    }

    /// Streaming handles clipped trajectories correctly: a single line
    /// fed in chunks should produce roughly the same tiles a single-pass
    /// in-mem build does.
    #[test]
    fn streaming_handles_trajectory_clipping() {
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

        let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = Archive::create(&path, Compression::Zstd).unwrap();
        let stats = build_streaming_from_batches(
            std::iter::once(Ok(features)),
            &config,
            &mut writer,
            2,
            16 * 1024,
        )
        .unwrap();
        writer.finalize(&Metadata::new("stream-tracks")).unwrap();
        assert!(stats.total_tiles > 1, "trajectory should cover multiple tiles");

        let reader = ArchiveReader::open(&path).unwrap();
        let entry = reader.entries()[0].clone();
        let layers = reader.read_layers(&entry).unwrap();
        assert!(layers[0].batch.column_by_name("vertex_time").is_some());
    }

    // ------------------------------------------------------------------
    // Temporal LOD aggregator
    // ------------------------------------------------------------------

    use stt_core::metadata::TemporalLodLevel;

    #[test]
    fn lod_aggregator_emits_base_plus_per_level_tiles() {
        // 72 hourly points starting at a midnight boundary so they fall
        // into exactly 3 contiguous daily buckets. With base bucket = 1h
        // and an LOD level at 1d:
        //   - the base path produces 72 tiles per zoom (one per hour),
        //   - the LOD path collapses each day's hourly tiles into a single
        //     daily tile per zoom.
        let hour = 3_600_000u64;
        let day = 24 * hour;
        // 1700006400000 = 2023-11-14 00:00:00 UTC — exact day boundary.
        let day_aligned = 1_700_006_400_000u64;
        assert_eq!(day_aligned % day, 0);
        let mut features = Vec::new();
        for hour_idx in 0..72u64 {
            features.push(point(-122.45, 37.75, day_aligned + hour_idx * hour));
        }
        let config = TileConfig {
            min_zoom: 8,
            max_zoom: 9,
            layer_name: "default".to_string(),
            temporal_bucket_ms: hour,
            clip_trajectories: false,
            temporal_lod: vec![TemporalLodLevel {
                bucket_ms: day,
                max_zoom_level: 9,
            }],
            ..TileConfig::default()
        };
        let tagged = generate_tiles_with_lod(&features, &config, 1).unwrap();
        let base: Vec<&LodTaggedTile> = tagged
            .iter()
            .filter(|t| t.temporal_bucket_ms == Some(hour))
            .collect();
        let lod: Vec<&LodTaggedTile> = tagged
            .iter()
            .filter(|t| t.temporal_bucket_ms == Some(day))
            .collect();
        // 72 hourly buckets × 2 zooms.
        assert_eq!(base.len(), 72 * 2);
        // 3 daily buckets × 2 zooms.
        assert_eq!(lod.len(), 3 * 2);
        // Every emitted tile carries some bucket tag (None is never produced
        // by the LOD writer path).
        assert!(tagged.iter().all(|t| t.temporal_bucket_ms.is_some()));
    }

    #[test]
    fn lod_aggregator_skips_zooms_above_max_zoom_level() {
        let hour = 3_600_000u64;
        let day = 24 * hour;
        let features = vec![point(0.0, 0.0, 1_000_000_000)];
        let config = TileConfig {
            min_zoom: 0,
            max_zoom: 10,
            layer_name: "default".to_string(),
            temporal_bucket_ms: hour,
            clip_trajectories: false,
            temporal_lod: vec![TemporalLodLevel {
                bucket_ms: day,
                max_zoom_level: 4,
            }],
            ..TileConfig::default()
        };
        let tagged = generate_tiles_with_lod(&features, &config, 1).unwrap();
        let lod: Vec<u8> = tagged
            .iter()
            .filter(|t| t.temporal_bucket_ms == Some(day))
            .map(|t| t.tile.id.z)
            .collect();
        // Every LOD tile sits at z<=4 (the level's max_zoom_level).
        assert!(!lod.is_empty());
        assert!(lod.iter().all(|&z| z <= 4));
        // Base tiles still cover the full 0..=10 zoom range.
        let base_zooms: std::collections::BTreeSet<u8> = tagged
            .iter()
            .filter(|t| t.temporal_bucket_ms == Some(hour))
            .map(|t| t.tile.id.z)
            .collect();
        assert_eq!(base_zooms, (0..=10).collect());
    }

    #[test]
    fn lod_aggregator_rejects_non_multiple_bucket() {
        let config = TileConfig {
            temporal_bucket_ms: 3_600_000,
            temporal_lod: vec![TemporalLodLevel {
                bucket_ms: 3_600_000 + 1, // not a multiple
                max_zoom_level: 6,
            }],
            ..TileConfig::default()
        };
        let err = generate_tiles_with_lod(&[], &config, 1).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("multiple"), "got: {msg}");
    }

    #[test]
    fn lod_aggregator_rejects_unsorted_levels() {
        let config = TileConfig {
            temporal_bucket_ms: 3_600_000,
            temporal_lod: vec![
                TemporalLodLevel { bucket_ms: 24 * 3_600_000, max_zoom_level: 6 },
                TemporalLodLevel { bucket_ms: 2 * 3_600_000, max_zoom_level: 6 },
            ],
            ..TileConfig::default()
        };
        assert!(generate_tiles_with_lod(&[], &config, 1).is_err());
    }

    #[test]
    fn lod_tiles_carry_bucket_size_through_archive_round_trip() {
        // Full pipeline: build LOD-tagged tiles, write them with
        // LodTileWriter, read back, and confirm every directory entry's
        // temporal_bucket_ms matches the level that produced it.
        let hour = 3_600_000u64;
        let day = 24 * hour;
        let features: Vec<ParsedFeature> = (0..48u64)
            .map(|h| point(-122.45, 37.75, 1_700_000_000_000 + h * hour))
            .collect();
        let config = TileConfig {
            min_zoom: 8,
            max_zoom: 9,
            layer_name: "default".to_string(),
            temporal_bucket_ms: hour,
            clip_trajectories: false,
            temporal_lod: vec![TemporalLodLevel { bucket_ms: day, max_zoom_level: 9 }],
            ..TileConfig::default()
        };
        let tagged = generate_tiles_with_lod(&features, &config, 1).unwrap();

        let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = Archive::create(&path, Compression::Zstd).unwrap();
        for t in &tagged {
            writer.write_lod_tile(&t.tile, t.temporal_bucket_ms).unwrap();
        }
        let metadata = stt_core::metadata::Metadata::new("lod")
            .with_temporal_bucket_ms(hour)
            .with_temporal_lod(vec![TemporalLodLevel { bucket_ms: day, max_zoom_level: 9 }])
            .unwrap();
        writer.finalize(&metadata).unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        let buckets: std::collections::BTreeSet<Option<u64>> =
            reader.entries().iter().map(|e| e.temporal_bucket_ms).collect();
        // The on-disk index distinguishes base + LOD bucket sizes.
        assert!(buckets.contains(&Some(hour)));
        assert!(buckets.contains(&Some(day)));
        assert!(!buckets.contains(&None));
    }

    // ------------------------------------------------------------------
    // Temporal clipping into the base path (WS-4)
    // ------------------------------------------------------------------

    #[test]
    fn base_path_temporally_clips_trajectory_into_buckets() {
        // A trajectory whose timing spans two hourly buckets must split so each
        // (tile, bucket) is self-contained: tiles appear in >=2 distinct
        // temporal buckets. Without temporal clipping the whole trajectory would
        // land in its start bucket only.
        let hour = 3_600_000u64;
        let coords: Vec<Vec<f64>> = (0..=20)
            .map(|i| vec![-122.45 + i as f64 * 0.001, 37.75 + i as f64 * 0.0005])
            .collect();
        let first = coords[0].clone();
        let feat = ParsedFeature {
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::LineString(coords))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 0,
            end_timestamp: Some(2 * hour),
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: first[0],
            lat: first[1],
        };
        let config = TileConfig {
            min_zoom: 12,
            max_zoom: 12,
            layer_name: "tracks".to_string(),
            temporal_bucket_ms: hour,
            clip_trajectories: true,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        let h = hour as i64;
        let buckets: std::collections::BTreeSet<i64> = tiles
            .iter()
            .map(|t| t.time_start - t.time_start.rem_euclid(h))
            .collect();
        assert!(
            buckets.len() >= 2,
            "trajectory should temporally clip into >=2 buckets, got {buckets:?}"
        );
    }

    // ------------------------------------------------------------------
    // Adaptive temporal chunking (WS-5)
    // ------------------------------------------------------------------

    #[test]
    fn adaptive_temporal_chunking_sizes_windows_by_count() {
        // 100 distinct-time points in one spatial cell. With target=10 the
        // adaptive chunker yields ~10 windows (vs ~1 with a 1h bucket), each a
        // self-contained tile with a distinct (z, x, y, t) key.
        let features: Vec<ParsedFeature> = (0..100u64)
            .map(|i| point(-122.45, 37.75, 1_700_000_000_000 + i * 60_000))
            .collect();
        let config = TileConfig {
            min_zoom: 8,
            max_zoom: 8,
            layer_name: "default".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            adaptive_target_features: Some(10),
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&features, &config, 1).unwrap();

        let total: usize = tiles.iter().map(|t| t.feature_count() as usize).sum();
        assert_eq!(total, 100, "every feature must appear exactly once");
        assert_eq!(tiles.len(), 10, "expected 10 windows of 10, got {}", tiles.len());

        let keys: std::collections::BTreeSet<(u8, u32, u32, i64)> = tiles
            .iter()
            .map(|t| (t.id.z, t.id.x, t.id.y, t.time_start))
            .collect();
        assert_eq!(keys.len(), tiles.len(), "window keys must be distinct");
        for t in &tiles {
            assert!(t.feature_count() <= 11, "window over budget: {}", t.feature_count());
        }
    }

    #[test]
    fn adaptive_chunking_keeps_identical_timestamps_together() {
        // Features sharing one exact timestamp in a cell map to the same
        // (z, x, y, t) key, so they cannot be split into separate tiles — they
        // stay in one window even past `target` (a documented constraint).
        let features: Vec<ParsedFeature> = (0..50)
            .map(|_| point(-122.45, 37.75, 1_700_000_000_000))
            .collect();
        let config = TileConfig {
            min_zoom: 8,
            max_zoom: 8,
            layer_name: "default".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            adaptive_target_features: Some(10),
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&features, &config, 1).unwrap();
        assert_eq!(tiles.len(), 1, "identical-timestamp features can't be split into tiles");
        assert_eq!(tiles[0].feature_count(), 50);
    }

    // ------------------------------------------------------------------
    // Time-aware (SED) simplification (WS-8)
    // ------------------------------------------------------------------

    #[test]
    fn time_aware_simplify_builds_tiles() {
        let feat = trajectory(0, 3_600_000);
        let config = TileConfig {
            min_zoom: 5,
            max_zoom: 5,
            layer_name: "tracks".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: true,
            simplify: true,
            time_aware_simplify: true,
            simplify_max_zoom: 14,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        assert!(!tiles.is_empty(), "time-aware simplify should still produce tiles");
        // Clipped trajectory layers carry per-vertex times (TD-TR preserved them).
        for t in &tiles {
            for l in &t.layers {
                assert!(l.vertex_times.is_some(), "trajectory layer should carry vertex_times");
            }
        }
    }

    // ------------------------------------------------------------------
    // Opt-in per-tile budgets (Wave-1)
    // ------------------------------------------------------------------

    /// Build 30 points that all land in one (zoom, x, y, bucket) tile so we can
    /// exercise the budget on a single dense tile.
    fn dense_single_tile_features(n: u64) -> (Vec<ParsedFeature>, TileConfig) {
        let base = 1_700_000_000_000u64;
        let features: Vec<ParsedFeature> = (0..n)
            // Tiny lon jitter keeps them in one zoom-6 tile while giving each a
            // distinct id; same timestamp -> one temporal bucket.
            .map(|i| point(-122.40 + i as f64 * 1e-6, 37.75, base))
            .collect();
        let config = TileConfig {
            min_zoom: 6,
            max_zoom: 6,
            layer_name: "default".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            ..TileConfig::default()
        };
        (features, config)
    }

    /// Default (no budget) leaves every feature in the tile — the inert path.
    #[test]
    fn budget_off_by_default_keeps_all_features() {
        let (features, config) = dense_single_tile_features(30);
        assert!(config.tile_budget.is_none());
        let tiles = generate_tiles(&features, &config, 1).unwrap();
        let total: u32 = tiles.iter().map(|t| t.feature_count()).sum();
        assert_eq!(total, 30, "no budget => no features dropped");
    }

    /// `--maximum-tile-features` caps the per-tile count and drops the surplus.
    #[test]
    fn maximum_tile_features_caps_feature_count() {
        let (features, mut config) = dense_single_tile_features(30);
        config.tile_budget = Some(
            TileBudget::new(usize::MAX, usize::MAX, 10)
                .with_scorer(stt_core::budget::ImportanceScorer::Combined),
        );
        let tiles = generate_tiles(&features, &config, 1).unwrap();
        // All 30 collapsed into one dense tile, capped to 10.
        assert_eq!(tiles.len(), 1, "all points share one tile");
        assert_eq!(
            tiles[0].feature_count(),
            10,
            "feature cap of 10 must be enforced"
        );
    }

    /// A tile already under the cap is left completely untouched.
    #[test]
    fn budget_under_cap_is_noop() {
        let (features, mut config) = dense_single_tile_features(5);
        config.tile_budget = Some(TileBudget::new(usize::MAX, usize::MAX, 10));
        let tiles = generate_tiles(&features, &config, 1).unwrap();
        let total: u32 = tiles.iter().map(|t| t.feature_count()).sum();
        assert_eq!(total, 5, "under-cap tile keeps every feature");
    }

    /// The byte-cap axis also drops features (here a very small cap forces a
    /// drop even though the feature count is modest).
    #[test]
    fn maximum_tile_bytes_drops_to_fit() {
        let (features, mut config) = dense_single_tile_features(30);
        // ~48 bytes per point estimate; a 200-byte cap keeps only a few.
        config.tile_budget = Some(TileBudget::new(200, 200, usize::MAX));
        let tiles = generate_tiles(&features, &config, 1).unwrap();
        let total: u32 = tiles.iter().map(|t| t.feature_count()).sum();
        assert!(total < 30, "byte cap must drop some features, kept {total}");
        assert!(total >= 1, "byte cap should still keep at least one feature");
    }
}
