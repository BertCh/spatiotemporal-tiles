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
use std::sync::Arc;
#[cfg(test)]
use stt_core::arrow_tile::encode_tile;
use stt_core::arrow_tile::ColumnarLayer;
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
    /// entirely after a query window. See `TileEntry::cover_t_min`.
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

    /// Persist a batch of tiles. Implementations MAY encode payloads in
    /// parallel but MUST hand tiles to storage in exactly the given order —
    /// callers rely on a deterministic write sequence for byte-reproducible
    /// output. The default forwards to [`write_tile`](Self::write_tile) one
    /// by one.
    fn write_tiles(&mut self, tiles: &[&GeneratedTile]) -> Result<()> {
        for tile in tiles {
            self.write_tile(tile)?;
        }
        Ok(())
    }
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
    /// Feature placements dropped because the position could not be projected
    /// (lon ∉ [-180, 180], |lat| beyond the Web-Mercator clamp, or non-finite).
    /// Counted per (feature, zoom) placement attempt and reported once at the
    /// end of the build — never silently, never into tile (0, 0).
    pub dropped_invalid_coords: usize,
    /// Features placed whole into a single tile because their
    /// antimeridian-crossing rings could not be split — pathological
    /// unsplittable / pole-enclosing rings only (an unwrapped exterior with no
    /// single ±180 seam strictly inside its span). Real dateline data (Fiji,
    /// Chukotka, storm cells) splits cleanly across both hemispheres, so this is
    /// EXPECTED to be 0; a non-zero count flags a degenerate source ring, never
    /// a routine crossing. Counted per (feature, zoom).
    pub antimeridian_fallbacks: usize,
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
    /// Use a latitude-corrected **metric** simplification tolerance instead of
    /// the legacy fixed per-zoom degree table (see [`ClipConfig::simplify_metric`]).
    /// Opt-in (default `false`) so existing builds stay byte-identical; a fixed
    /// degree tolerance over-keeps/over-drops detail by up to the cos(latitude)
    /// factor across a global dataset, which this corrects.
    pub simplify_metric: bool,
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
    /// When set, the named per-feature numeric property is a LOD *ceiling*: a
    /// feature is SKIPPED at any zoom ABOVE its value. Paired with
    /// [`Self::min_zoom_field`] it confines a feature to a zoom BAND
    /// `[min_zoom, max_zoom]` — e.g. coarse-zoom clustered/aggregated features
    /// that must NOT bleed into the full-resolution deep zooms. Whole-feature
    /// inclusion only — geometry/attributes (incl. the value matrix) untouched.
    /// `None` = no ceiling (a feature appears at every zoom ≥ its `min_zoom`).
    pub max_zoom_field: Option<String>,
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
    /// Authoritative per-property kinds from the input source's schema (see
    /// [`crate::columnar::ColumnarOptions::property_types`]). GeoParquet/DB
    /// inputs populate this so a column that is all-null within one tile still
    /// gets its column there — per-tile value sniffing otherwise drops it and
    /// the layer schema drifts across tiles. Default empty (schema-less
    /// producers keep sniffing).
    pub property_types: Arc<crate::columnar::PropertyTypes>,
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
            simplify_metric: false,
            pre_tessellate: false,
            temporal_lod: Vec::new(),
            min_features_per_tile: 1,
            time_aware_simplify: false,
            adaptive_target_features: None,
            min_zoom_field: None,
            max_zoom_field: None,
            tile_budget: None,
            attribute_filter: AttributeFilter::KeepAll,
            property_types: Arc::default(),
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
            property_types: Arc::clone(&self.property_types),
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

/// A feature assigned to a tile — an original feature, a per-tile piece of a
/// clipped non-trajectory feature, or a clipped trajectory segment.
#[derive(Debug, Clone)]
enum TileFeature<'a> {
    Original(&'a ParsedFeature),
    /// Owned per-tile piece of a clipped NON-trajectory feature (polygon,
    /// timeless line, MultiPoint member). Carries the parent's properties,
    /// times, id and representative point (so id-less pieces hash to the SAME
    /// synthetic feature id in every tile they span) with per-tile clipped
    /// geometry. Routed through the same layer builders as `Original`.
    ///
    /// EXCEPTION — MultiPoint members carry their OWN coordinates as
    /// `lon`/`lat`: for point layers that pair IS the emitted geometry
    /// (`build_point_layer` reads `f.lon`/`f.lat`), so the parent point
    /// would collapse every member onto one location. Identity is unharmed:
    /// pieces inherit the parent's explicit id when it has one, and id-less
    /// point ids are rewritten to per-tile row indices in
    /// `build_point_layer` before the synthetic hash ever reaches bytes.
    Derived(ParsedFeature),
    Clipped(ClippedSegment),
}

impl<'a> TileFeature<'a> {
    fn timestamp(&self) -> u64 {
        match self {
            TileFeature::Original(f) => f.timestamp,
            TileFeature::Derived(f) => f.timestamp,
            TileFeature::Clipped(s) => s.start_time,
        }
    }

    fn end_timestamp(&self) -> u64 {
        match self {
            TileFeature::Original(f) => f.end_timestamp.unwrap_or(f.timestamp),
            TileFeature::Derived(f) => f.end_timestamp.unwrap_or(f.timestamp),
            TileFeature::Clipped(s) => s.end_time,
        }
    }
}

/// Per-build counters for placements that could not be performed normally.
/// Surfaced ONCE at the end of a build via [`Self::report`] (never per-feature
/// spam) — the "no silent drops" guarantee for the placement stage.
#[derive(Debug, Default)]
struct PlacementCounters {
    /// Placements dropped because the position could not be projected
    /// (previously `unwrap_or((0, 0))` filed these into the top-left world
    /// tile as phantom features). Counted per (feature, zoom).
    ///
    /// A POLAR latitude is not one of these: it is clamped to
    /// `stt_core::projection::MERCATOR_MAX_LAT` and kept, the same as in the
    /// summary tier — only non-finite coordinates and out-of-range longitude
    /// land here.
    invalid_coords: AtomicUsize,
    /// Features that fell back to legacy whole-feature single-tile placement
    /// because their antimeridian-crossing rings were unsplittable
    /// (pole-enclosing / degenerate — no unique ±180 seam). The should-be-0
    /// safety net for the split-then-reuse polygon clipper; expected 0 on real
    /// data.
    antimeridian_fallback: AtomicUsize,
}

impl PlacementCounters {
    fn report(&self) {
        let invalid = self.invalid_coords.load(Ordering::Relaxed);
        if invalid > 0 {
            tracing::warn!(
                "dropped {invalid} feature placement(s) (per feature × zoom) with \
                 coordinates outside the Web-Mercator domain (lon∉[-180,180] or \
                 non-finite) — NOT written to any tile. Polar latitudes are NOT \
                 counted here: they clamp to the Mercator limit and are kept"
            );
        }
        let anti = self.antimeridian_fallback.load(Ordering::Relaxed);
        if anti > 0 {
            tracing::warn!(
                "{anti} feature placement(s) (per feature × zoom) used whole-feature \
                 single-tile fallback: their antimeridian-crossing rings were \
                 unsplittable (pole-enclosing or degenerate — no unique ±180 seam). \
                 Expected 0 on real data; inspect the source geometry"
            );
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
    let counters = PlacementCounters::default();
    let mut all = Vec::new();
    for zoom in lod_config.min_zoom..=lod_config.max_zoom {
        let tiles = process_zoom_level(
            features,
            zoom,
            &lod_config,
            &clip_config,
            &total_clipped,
            &total_original,
            &counters,
        )?;
        all.extend(tiles);
    }
    counters.report();
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
    let counters = PlacementCounters::default();
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
                &counters,
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
    counters.report();
    Ok(all)
}

/// Encode exactly one tile `(z, x, y, t)` from a candidate feature set, without
/// running the whole-dataset build (no rayon pool, no pack/directory writer, no
/// cross-tile state). The returned bytes are an **uncompressed** STT layer-frame
/// tile payload (Arrow IPC + GeoArrow geometry) — byte-identical to the frame
/// the offline build feeds INTO the pack writer *before* per-blob zstd, so a
/// dynamic server (`stt-serve`) can hand it out directly (it does its own
/// transport compression). `Ok(None)` means the tile is empty.
///
/// This is the reusable core a dynamic per-request tile server (`stt-serve`)
/// calls. `features` is the caller's candidate set — typically already narrowed
/// by a PostGIS bbox + time-window query; this function performs the
/// authoritative per-tile placement, clipping, temporal bucketing and encoding,
/// so it stays byte-identical to the offline `process_zoom_level` path.
///
/// `t` selects the temporal bucket: the tile covers
/// `[floor(t/bucket)*bucket, …)`, matching [`chunk_by_temporal_bucket`].
///
/// Also returns the number of features placed in the tile (after
/// clipping/placement/budget), so a dynamic server can apply a
/// `min_features_per_tile` gate identically to the offline build's writer loop.
/// `Ok(None)` means the tile is empty.
pub fn encode_single_tile_counted(
    features: &[ParsedFeature],
    z: u8,
    x: u32,
    y: u32,
    t: i64,
    config: &TileConfig,
    encoder: &stt_core::arrow_tile::EncoderConfig,
) -> Result<Option<(Vec<u8>, u32)>> {
    let clip_config = clip_config_from(config);
    let bucket_ms = config.temporal_bucket_ms.max(1);
    let bucket_start = (t.max(0) as u64 / bucket_ms) * bucket_ms;

    // Place each feature for this zoom, keeping only what lands in (x, y) — the
    // same clip-or-coverage placement `process_zoom_level` performs.
    let counters = PlacementCounters::default();
    let mut chunk: Vec<TileFeature> = Vec::new();
    for feature in features {
        if feature_out_of_band(feature, z, config) {
            continue;
        }
        for (fx, fy, tf) in place_feature(feature, z, config, &clip_config, &counters, Some((x, y)))
        {
            if fx == x && fy == y {
                chunk.push(tf);
            }
        }
    }
    counters.report();

    // Keep only the requested temporal bucket (matches chunk_by_temporal_bucket).
    chunk.retain(|f| (f.timestamp() / bucket_ms) * bucket_ms == bucket_start);
    if chunk.is_empty() {
        return Ok(None);
    }

    let time_end = chunk
        .iter()
        .map(|f| f.end_timestamp())
        .max()
        .unwrap_or(bucket_start + bucket_ms);
    let id = TileId::new(z, x, y, bucket_start);
    match build_tile(id, &chunk, config, bucket_start as i64, time_end as i64)? {
        Some(tile) => {
            let feature_count = tile.feature_count();
            // Encode with the caller's explicit encoder config (no globals), so a
            // dynamic server can serve several datasets/requests with different
            // settings concurrently without touching shared state.
            Ok(Some((
                stt_core::arrow_tile::encode_tile_with(&tile.layers, encoder)?,
                feature_count,
            )))
        }
        None => Ok(None),
    }
}

/// Encode exactly one tile `(z, x, y, t)`, discarding the placed-feature count.
/// The convenience form of [`encode_single_tile_counted`] for callers that don't
/// apply a `min_features_per_tile` gate. `Ok(None)` means the tile is empty.
pub fn encode_single_tile(
    features: &[ParsedFeature],
    z: u8,
    x: u32,
    y: u32,
    t: i64,
    config: &TileConfig,
    encoder: &stt_core::arrow_tile::EncoderConfig,
) -> Result<Option<Vec<u8>>> {
    Ok(encode_single_tile_counted(features, z, x, y, t, config, encoder)?.map(|(bytes, _)| bytes))
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
    let counters = PlacementCounters::default();
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
                &counters,
            )?;
            let min_features = config.min_features_per_tile.max(1);
            let keep: Vec<&GeneratedTile> = tiles
                .iter()
                .filter(|t| t.feature_count() >= min_features)
                .collect();
            // Batch write: a PackWriter sink encodes the payloads in parallel
            // (we're inside the --workers pool here) while handing tiles to
            // storage in this exact deterministic order.
            writer.write_tiles(&keep)?;
            let written = keep.len();
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
    counters.report();

    Ok(TileStats {
        total_tiles,
        clipped_segments: total_clipped.load(Ordering::Relaxed),
        original_features: total_original.load(Ordering::Relaxed),
        dropped_invalid_coords: counters.invalid_coords.load(Ordering::Relaxed),
        antimeridian_fallbacks: counters.antimeridian_fallback.load(Ordering::Relaxed),
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
        polygon_buffer_degrees: 0.0,
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
        simplify_metric: config.simplify_metric,
    }
}

/// Process a single zoom level: clip in parallel, bucket spatially then
/// temporally, and build each tile's layers.
/// Read a feature's LOD floor from the configured `min_zoom_field` property:
/// the shallowest zoom the feature appears at. `None` = always shown.
fn feature_min_zoom(feature: &ParsedFeature, field: &Option<String>) -> Option<u8> {
    feature_zoom_bound(feature, field)
}

/// Read a feature's LOD ceiling from the configured `max_zoom_field` property:
/// the deepest zoom the feature appears at. `None` = no ceiling.
fn feature_max_zoom(feature: &ParsedFeature, field: &Option<String>) -> Option<u8> {
    feature_zoom_bound(feature, field)
}

/// Shared reader for the per-feature numeric zoom-bound properties
/// (`min_zoom_field` / `max_zoom_field`).
fn feature_zoom_bound(feature: &ParsedFeature, field: &Option<String>) -> Option<u8> {
    let field = field.as_deref()?;
    feature
        .shared_properties
        .as_ref()?
        .get(field)
        .and_then(|v| v.as_f64())
        .map(|z| z.round() as u8)
}

/// `true` when `zoom` falls outside a feature's configured `[min_zoom,
/// max_zoom]` band (either bound absent = open on that side). Whole-feature
/// skip — callers return before any clip so the value matrix is never touched.
fn feature_out_of_band(feature: &ParsedFeature, zoom: u8, config: &TileConfig) -> bool {
    if let Some(mz) = feature_min_zoom(feature, &config.min_zoom_field) {
        if zoom < mz {
            return true;
        }
    }
    if let Some(mx) = feature_max_zoom(feature, &config.max_zoom_field) {
        if zoom > mx {
            return true;
        }
    }
    false
}

/// Legacy whole-feature placement: the single tile containing the feature's
/// representative point. Projection failures are dropped + counted — never
/// filed into tile (0, 0) as phantom features.
fn place_whole_feature<'a>(
    feature: &'a ParsedFeature,
    zoom: u8,
    counters: &PlacementCounters,
) -> Vec<(u32, u32, TileFeature<'a>)> {
    match projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
        Ok((x, y)) => vec![(x, y, TileFeature::Original(feature))],
        Err(_) => {
            counters.invalid_coords.fetch_add(1, Ordering::Relaxed);
            Vec::new()
        }
    }
}

/// Split a MultiPoint per containing tile: each member becomes its own
/// per-tile Point piece (the legacy path placed — and rendered — only the
/// whole feature's representative point). Members with unprojectable
/// coordinates are dropped + counted individually.
fn place_multipoint<'a>(
    feature: &'a ParsedFeature,
    points: &[Vec<f64>],
    zoom: u8,
    counters: &PlacementCounters,
) -> Vec<(u32, u32, TileFeature<'a>)> {
    if points.is_empty() {
        return place_whole_feature(feature, zoom, counters);
    }
    let mut out = Vec::with_capacity(points.len());
    for p in points {
        if p.len() < 2 {
            counters.invalid_coords.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        match projection::lonlat_to_tile(p[0], p[1], zoom) {
            Ok((x, y)) => {
                out.push((
                    x,
                    y,
                    TileFeature::Derived(ParsedFeature {
                        geojson: geojson::Feature {
                            bbox: None,
                            geometry: Some(geojson::Geometry::new(geojson::Value::Point(
                                p.clone(),
                            ))),
                            id: feature.geojson.id.clone(),
                            properties: None,
                            foreign_members: None,
                        },
                        shared_properties: feature.shared_properties.clone(),
                        timestamp: feature.timestamp,
                        end_timestamp: feature.end_timestamp,
                        vertex_timestamps: None,
                        vertex_values: None,
                        vertex_value_matrix: None,
                        lon: p[0],
                        lat: p[1],
                    }),
                ));
            }
            Err(_) => {
                counters.invalid_coords.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    out
}

/// Coverage placement for (Multi)Polygons: clip the rings against every tile
/// the geometry's bbox covers (Sutherland–Hodgman against the EXACT tile
/// rect — `polygon_buffer_degrees`, 0 by default: adjacent tiles emit
/// bit-identical seam vertices so fills rasterize watertight, where a
/// buffered strip would double-blend under `opacity < 1`), emitting one
/// per-tile piece wherever anything survives. Holes are preserved (rings
/// clip independently). Fast path: a polygon fully inside its
/// representative tile's rect takes the legacy single placement unchanged
/// (byte-identical output). Antimeridian-crossing rings (any edge with
/// `|Δlon| > 180°`) are first SPLIT into per-hemisphere pieces
/// (`clip::split_polygon_at_antimeridian`) and then clipped like any other
/// polygon, so a dateline feature lands — clipped — on BOTH sides of ±180.
/// Only a pathological unsplittable ring (pole-enclosing / no unique ±180
/// seam) falls back to legacy whole-feature placement and is counted.
/// `target` restricts the sweep to one tile (the stt-serve per-request
/// path); the piece emitted for that tile is byte-identical to the full
/// sweep's.
fn place_polygon<'a>(
    feature: &'a ParsedFeature,
    polygons: &[crate::clip::PolygonRings],
    multi: bool,
    zoom: u8,
    clip_config: &ClipConfig,
    counters: &PlacementCounters,
    target: Option<(u32, u32)>,
) -> Vec<(u32, u32, TileFeature<'a>)> {
    let mut min_lon = f64::MAX;
    let mut min_lat = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut max_lat = f64::MIN;
    for c in polygons.iter().flatten().flatten() {
        if c.len() >= 2 {
            min_lon = min_lon.min(c[0]);
            min_lat = min_lat.min(c[1]);
            max_lon = max_lon.max(c[0]);
            max_lat = max_lat.max(c[1]);
        }
    }
    if !(min_lon.is_finite() && min_lat.is_finite() && max_lon.is_finite() && max_lat.is_finite())
        || min_lon > max_lon
    {
        // Degenerate/garbage rings: let the representative point decide
        // (an unprojectable point is dropped + counted there).
        return place_whole_feature(feature, zoom, counters);
    }
    // Antimeridian split-then-reuse: a ring with any edge whose consecutive
    // vertices differ by more than 180° of longitude straddles the dateline.
    // Rings are closed, so `windows(2)` covers the closing edge too. This is a
    // per-EDGE test (the same signal the polyline splitter uses), not the old
    // whole-feature `bbox > 180°` test: a polygon that merely spans a wide (but
    // < 360°) longitude range WITHOUT any wrapping edge genuinely occupies those
    // columns and clips normally.
    let crosses = polygons.iter().flatten().any(|ring| {
        ring.windows(2)
            .any(|w| w[0].len() >= 2 && w[1].len() >= 2 && (w[1][0] - w[0][0]).abs() > 180.0)
    });

    let rep_tile = projection::lonlat_to_tile(feature.lon, feature.lat, zoom).ok();

    // Per-tile polygon simplification (A2), the fill parallel of the line path.
    // Gated strictly BELOW `simplify_max_zoom` so the max-tiled-zoom tier stays
    // lossless / byte-identical (watertight A1 seams depend on it). Applied per
    // tile AFTER the antimeridian split (`work` already split) AND AFTER the
    // Sutherland–Hodgman clip inside `emit` — never before.
    let simplify_here = clip_config.simplify && zoom < clip_config.simplify_max_zoom;

    // Clip a working polygon set to the tile(s) and map surviving pieces to
    // per-tile `Derived` features — the shared tail for BOTH the non-crossing
    // path (`work = polygons`) and the crossing path (`work = split pieces`).
    // Passing `work` (not `polygons`) through the empty-handling target re-run
    // keeps the single-tile serve path byte-identical to the full sweep.
    let emit = |work: &[crate::clip::PolygonRings]| -> Vec<(u32, u32, TileFeature<'a>)> {
        let pieces = crate::clip::clip_polygons_to_tiles(
            work,
            zoom,
            clip_config.polygon_buffer_degrees,
            target,
        );
        if pieces.is_empty() {
            // Nothing survived clipping (sliver thinner than the clipper keeps):
            // keep the legacy placement rather than dropping the feature. Under a
            // target restriction "empty" only means empty-at-target: the legacy
            // fallback lands at the representative tile, so it applies only when
            // the target IS that tile AND the UNRESTRICTED sweep is also empty
            // (else the full build placed pieces elsewhere and nothing at all at
            // the representative tile).
            return match target {
                None => place_whole_feature(feature, zoom, counters),
                Some(t) if rep_tile == Some(t) => {
                    if crate::clip::clip_polygons_to_tiles(
                        work,
                        zoom,
                        clip_config.polygon_buffer_degrees,
                        None,
                    )
                    .is_empty()
                    {
                        place_whole_feature(feature, zoom, counters)
                    } else {
                        Vec::new()
                    }
                }
                Some(_) => Vec::new(),
            };
        }
        pieces
            .into_iter()
            .map(|((x, y), mut polys)| {
                // Simplify each surviving per-tile polygon (topology-preserving),
                // AFTER the clip. No-op above the gate (max tier stays lossless).
                if simplify_here {
                    for rings in polys.iter_mut() {
                        *rings = crate::simplify::simplify_polygon_rings_for_zoom_with(
                            rings,
                            zoom,
                            clip_config.simplify_max_zoom,
                            clip_config.simplify_metric,
                        );
                    }
                }
                let geometry = if !multi && polys.len() == 1 {
                    geojson::Value::Polygon(polys.pop().unwrap())
                } else {
                    geojson::Value::MultiPolygon(polys)
                };
                (
                    x,
                    y,
                    TileFeature::Derived(ParsedFeature {
                        geojson: geojson::Feature {
                            bbox: None,
                            geometry: Some(geojson::Geometry::new(geometry)),
                            id: feature.geojson.id.clone(),
                            properties: None,
                            foreign_members: None,
                        },
                        shared_properties: feature.shared_properties.clone(),
                        timestamp: feature.timestamp,
                        end_timestamp: feature.end_timestamp,
                        vertex_timestamps: None,
                        vertex_values: None,
                        vertex_value_matrix: None,
                        // Parent's representative point, so id-less pieces hash
                        // to the SAME synthetic feature id in every tile.
                        lon: feature.lon,
                        lat: feature.lat,
                    }),
                )
            })
            .collect()
    };

    if crosses {
        // Split only the parts that ACTUALLY cross; pass non-crossing parts
        // through unchanged, then clip the working set like any other polygon
        // (NO fast-path — a crossing feature can't be tile-contained). `crosses`
        // is feature-wide, but a MultiPolygon can mix a crossing part (e.g. the
        // Aleutians straddling ±180) with ordinary parts (CONUS) — flat-mapping
        // the splitter over EVERY part would DROP the non-crossers, because
        // `split_polygon_at_antimeridian` returns empty for a part with no
        // dateline seam. A crossing part that is itself unsplittable
        // (pole-enclosing / degenerate → empty split) dead-letters the WHOLE
        // feature to the legacy whole-feature placement, which keeps every part
        // and counts the fallback — never a silent drop.
        let mut work: Vec<crate::clip::PolygonRings> = Vec::with_capacity(polygons.len());
        for part in polygons {
            let part_crosses = part.iter().any(|ring| {
                ring.windows(2).any(|w| {
                    w[0].len() >= 2 && w[1].len() >= 2 && (w[1][0] - w[0][0]).abs() > 180.0
                })
            });
            if part_crosses {
                let pieces = crate::clip::split_polygon_at_antimeridian(part);
                if pieces.is_empty() {
                    counters
                        .antimeridian_fallback
                        .fetch_add(1, Ordering::Relaxed);
                    return place_whole_feature(feature, zoom, counters);
                }
                work.extend(pieces);
            } else {
                work.push(part.clone());
            }
        }
        if work.is_empty() {
            counters
                .antimeridian_fallback
                .fetch_add(1, Ordering::Relaxed);
            return place_whole_feature(feature, zoom, counters);
        }
        return emit(&work);
    }

    // Non-crossing fast path: a polygon fully inside its representative tile's
    // rect takes the legacy single placement unchanged (byte-identical output).
    // Skipped when simplifying at THIS zoom: a contained polygon must still be
    // simplified, which only happens on the clip path (`emit`) — clipping a
    // fully-inside ring is an identity (Sutherland–Hodgman leaves it unchanged),
    // so it then simplifies exactly like any spanning piece. The fast path stays
    // in force for the lossless max tier, keeping that byte-identical.
    if !simplify_here {
        if let Some((fx, fy)) = rep_tile {
            // Same rect as the sweep below (polygon_buffer_degrees), so the two
            // paths agree on containment.
            let (bl, bb, br, bt) =
                crate::clip::buffered_tile_bounds(fx, fy, zoom, clip_config.polygon_buffer_degrees);
            if min_lon >= bl && max_lon <= br && min_lat >= bb && max_lat <= bt {
                return vec![(fx, fy, TileFeature::Original(feature))];
            }
        }
    }
    emit(polygons)
}

/// Clip a (Multi)LineString across the tiles it traverses.
///
/// Timeless lines are clipped spatially only — no temporal slicing, no
/// simplification (the legacy timeless path never simplified) — and re-emitted
/// as per-tile [`TileFeature::Derived`] pieces so the layer schema stays what
/// the originals path produces for timeless lines (no `vertex_time` column
/// unless the producer supplied per-vertex times). A MultiLineString WITH
/// duration routes each part through the existing trajectory clipper as its
/// own segment run (temporal slicing, matrix pinning and per-vertex-array
/// interpolation apply exactly as for LineString trajectories); per-vertex
/// arrays sized to the whole geometry are sliced per part.
fn place_polyline<'a>(
    feature: &'a ParsedFeature,
    parts: &[Vec<Vec<f64>>],
    zoom: u8,
    clip_config: &ClipConfig,
    counters: &PlacementCounters,
) -> Vec<(u32, u32, TileFeature<'a>)> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    if total < 2 || parts.iter().flatten().any(|c| c.len() < 2) {
        return place_whole_feature(feature, zoom, counters);
    }

    let mut min_lon = f64::MAX;
    let mut min_lat = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut max_lat = f64::MIN;
    for c in parts.iter().flatten() {
        min_lon = min_lon.min(c[0]);
        min_lat = min_lat.min(c[1]);
        max_lon = max_lon.max(c[0]);
        max_lat = max_lat.max(c[1]);
    }
    if !(min_lon.is_finite() && min_lat.is_finite() && max_lon.is_finite() && max_lat.is_finite()) {
        return place_whole_feature(feature, zoom, counters);
    }
    // Fast path: fully inside the representative tile's buffered rect —
    // legacy single placement, byte-identical to the old behaviour. (No
    // antimeridian bbox heuristic for lines: the trajectory clipper already
    // splits runs at >180° longitude jumps.)
    if let Ok((fx, fy)) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
        let (bl, bb, br, bt) =
            crate::clip::buffered_tile_bounds(fx, fy, zoom, clip_config.buffer_degrees);
        if min_lon >= bl && max_lon <= br && min_lat >= bb && max_lat <= bt {
            return vec![(fx, fy, TileFeature::Original(feature))];
        }
    }

    // Per-vertex arrays are accepted only when sized to the WHOLE geometry
    // (mirrors `build_line_layer`'s length contract) and sliced per part.
    let supplied_times = match feature.vertex_timestamps.as_deref() {
        Some(s) if s.len() == total => Some(s),
        Some(s) => {
            tracing::warn!(
                "vertex_timestamps length {} != vertex count {} on a clipped \
                 (multi)line; dropping the supplied times for this feature",
                s.len(),
                total
            );
            None
        }
        None => None,
    };
    let supplied_values = match feature.vertex_values.as_deref() {
        Some(s) if s.len() == total => Some(s),
        Some(s) => {
            tracing::warn!(
                "vertex_values length {} != vertex count {} on a clipped \
                 (multi)line; dropping the supplied values for this feature",
                s.len(),
                total
            );
            None
        }
        None => None,
    };
    let matrix = match feature.vertex_value_matrix.as_deref() {
        Some(m) if !m.is_empty() && m.len() % total == 0 => Some((m, m.len() / total)),
        Some(m) => {
            tracing::warn!(
                "vertex_value_matrix length {} is not a multiple of vertex count {} \
                 on a clipped (multi)line; dropping the matrix for this feature",
                m.len(),
                total
            );
            None
        }
        None => None,
    };

    if feature.end_timestamp.is_some() {
        // Duration MultiLineString: each part through the trajectory clipper
        // as its own segment run, sharing the parent's feature id. Flat
        // per-vertex times over the concatenated parts reproduce the legacy
        // flattened `build_line_layer` timing exactly.
        let end_time = feature.end_timestamp.unwrap_or(feature.timestamp);
        let flat_times: Vec<u64> = match supplied_times {
            Some(s) => s.to_vec(),
            None => {
                let flat: Vec<(f64, f64, f64)> = parts
                    .iter()
                    .flatten()
                    .map(|c| (c[0], c[1], if c.len() >= 3 { c[2] } else { 0.0 }))
                    .collect();
                crate::clip::compute_vertex_timestamps(&flat, feature.timestamp, end_time)
            }
        };
        let mut placements: Vec<(u32, u32, TileFeature<'a>)> = Vec::new();
        let mut offset = 0usize;
        for part in parts {
            let len = part.len();
            let part_times = &flat_times[offset..offset + len];
            let part_values = supplied_values.map(|v| &v[offset..offset + len]);
            let part_matrix = matrix.map(|(m, nb)| &m[offset * nb..(offset + len) * nb]);
            offset += len;
            if len < 2 {
                continue;
            }
            let synthetic = geojson::Feature {
                bbox: None,
                geometry: Some(geojson::Geometry::new(geojson::Value::LineString(
                    part.clone(),
                ))),
                id: feature.geojson.id.clone(),
                properties: None,
                foreign_members: None,
            };
            let segments = clip_trajectory(
                &synthetic,
                feature.shared_properties.clone(),
                feature.timestamp,
                end_time,
                zoom,
                clip_config,
                Some(part_times),
                part_values,
                part_matrix,
            );
            placements.extend(
                segments
                    .into_iter()
                    .map(|s| (s.tile_x, s.tile_y, TileFeature::Clipped(s))),
            );
        }
        if placements.is_empty() {
            return place_whole_feature(feature, zoom, counters);
        }
        return placements;
    }

    // Timeless: spatial-only clip.
    let timeless_cfg = ClipConfig {
        temporal_granularity_ms: None,
        simplify: false,
        ..clip_config.clone()
    };
    let mut per_tile: BTreeMap<(u32, u32), Vec<ClippedSegment>> = BTreeMap::new();
    let mut offset = 0usize;
    for part in parts {
        let len = part.len();
        let part_times = supplied_times.map(|s| &s[offset..offset + len]);
        let part_values = supplied_values.map(|v| &v[offset..offset + len]);
        let part_matrix = matrix.map(|(m, nb)| &m[offset * nb..(offset + len) * nb]);
        offset += len;
        if len < 2 {
            continue;
        }
        let synthetic = geojson::Feature {
            bbox: None,
            geometry: Some(geojson::Geometry::new(geojson::Value::LineString(
                part.clone(),
            ))),
            id: feature.geojson.id.clone(),
            properties: None,
            foreign_members: None,
        };
        // Properties ride the Derived feature, not the throwaway segments.
        let segments = clip_trajectory(
            &synthetic,
            None,
            feature.timestamp,
            feature.timestamp,
            zoom,
            &timeless_cfg,
            part_times,
            part_values,
            part_matrix,
        );
        for s in segments {
            per_tile.entry((s.tile_x, s.tile_y)).or_default().push(s);
        }
    }
    if per_tile.is_empty() {
        return place_whole_feature(feature, zoom, counters);
    }
    per_tile
        .into_iter()
        .map(|((x, y), segs)| {
            let mut geom_parts: Vec<Vec<Vec<f64>>> = segs
                .iter()
                .map(|s| {
                    s.coordinates
                        .iter()
                        .map(|(lx, ly, la)| vec![*lx, *ly, *la])
                        .collect()
                })
                .collect();
            let geometry = if geom_parts.len() == 1 {
                geojson::Value::LineString(geom_parts.pop().unwrap())
            } else {
                geojson::Value::MultiLineString(geom_parts)
            };
            let vertex_timestamps = supplied_times.map(|_| {
                segs.iter()
                    .flat_map(|s| s.timestamps.iter().copied())
                    .collect()
            });
            let vertex_values = supplied_values.map(|_| {
                segs.iter()
                    .flat_map(|s| s.vertex_values.iter().copied())
                    .collect()
            });
            let vertex_value_matrix = matrix.map(|_| {
                segs.iter()
                    .flat_map(|s| s.vertex_value_matrix.iter().flatten().copied())
                    .collect()
            });
            (
                x,
                y,
                TileFeature::Derived(ParsedFeature {
                    geojson: geojson::Feature {
                        bbox: None,
                        geometry: Some(geojson::Geometry::new(geometry)),
                        id: feature.geojson.id.clone(),
                        properties: None,
                        foreign_members: None,
                    },
                    shared_properties: feature.shared_properties.clone(),
                    timestamp: feature.timestamp,
                    end_timestamp: feature.end_timestamp,
                    vertex_timestamps,
                    vertex_values,
                    vertex_value_matrix,
                    lon: feature.lon,
                    lat: feature.lat,
                }),
            )
        })
        .collect()
}

/// Place a non-trajectory feature: coverage placement + clipping for
/// polygons, timeless (multi)lines and multipoints. Points and
/// GeometryCollections are always whole-feature (they cannot span tiles, or
/// carry no single clippable geometry).
fn place_non_trajectory<'a>(
    feature: &'a ParsedFeature,
    zoom: u8,
    config: &TileConfig,
    clip_config: &ClipConfig,
    counters: &PlacementCounters,
    target: Option<(u32, u32)>,
) -> Vec<(u32, u32, TileFeature<'a>)> {
    use geojson::Value as G;
    let Some(geom) = feature.geojson.geometry.as_ref() else {
        return place_whole_feature(feature, zoom, counters);
    };
    match &geom.value {
        G::Point(_) | G::GeometryCollection(_) => place_whole_feature(feature, zoom, counters),
        G::MultiPoint(points) => place_multipoint(feature, points, zoom, counters),
        G::Polygon(rings) => place_polygon(
            feature,
            std::slice::from_ref(rings),
            false,
            zoom,
            clip_config,
            counters,
            target,
        ),
        G::MultiPolygon(polys) => {
            place_polygon(feature, polys, true, zoom, clip_config, counters, target)
        }
        G::LineString(coords) => {
            if feature.end_timestamp.is_some() {
                // A duration LineString only reaches here when trajectory
                // clipping is off (--no-clip) or the geometry is degenerate:
                // honour the user's explicit whole-trajectory placement.
                place_whole_feature(feature, zoom, counters)
            } else {
                place_polyline(
                    feature,
                    std::slice::from_ref(coords),
                    zoom,
                    clip_config,
                    counters,
                )
            }
        }
        G::MultiLineString(parts) => {
            if feature.end_timestamp.is_some() && !config.clip_trajectories {
                place_whole_feature(feature, zoom, counters)
            } else {
                place_polyline(feature, parts, zoom, clip_config, counters)
            }
        }
    }
}

/// Place one feature at `zoom` — trajectory clip, non-trajectory coverage
/// clip, or whole-feature fallback. THE single placement authority: the
/// in-memory build, the streaming build and the dynamic single-tile encoder
/// (`stt-serve`) all route through here so their outputs stay identical.
/// `target` (the single-tile serve path) restricts the polygon coverage
/// sweep to one tile — the pieces emitted for that tile are byte-identical
/// to the full sweep's; other geometry kinds cost O(geometry), not
/// O(bbox tiles), and are simply filtered by the caller.
fn place_feature<'a>(
    feature: &'a ParsedFeature,
    zoom: u8,
    config: &TileConfig,
    clip_config: &ClipConfig,
    counters: &PlacementCounters,
    target: Option<(u32, u32)>,
) -> Vec<(u32, u32, TileFeature<'a>)> {
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
            place_whole_feature(feature, zoom, counters)
        } else {
            segments
                .into_iter()
                .map(|s| (s.tile_x, s.tile_y, TileFeature::Clipped(s)))
                .collect()
        }
    } else {
        place_non_trajectory(feature, zoom, config, clip_config, counters, target)
    }
}

fn process_zoom_level(
    features: &[ParsedFeature],
    zoom: u8,
    config: &TileConfig,
    clip_config: &ClipConfig,
    total_clipped: &AtomicUsize,
    total_original: &AtomicUsize,
    counters: &PlacementCounters,
) -> Result<Vec<GeneratedTile>> {
    // Parallel clip: each feature yields zero or more (tile_x, tile_y, feature).
    let placed: Vec<(u32, u32, TileFeature)> = features
        .par_iter()
        .flat_map(|feature| {
            // Road-class LOD: hide a feature outside its [min_zoom, max_zoom]
            // band. Whole-feature skip BEFORE clip — the value matrix is never
            // touched.
            if feature_out_of_band(feature, zoom, config) {
                return Vec::new();
            }
            let placements = place_feature(feature, zoom, config, clip_config, counters, None);
            let clipped = placements
                .iter()
                .filter(|(_, _, f)| matches!(f, TileFeature::Clipped(_)))
                .count();
            if clipped > 0 {
                total_clipped.fetch_add(clipped, Ordering::Relaxed);
            }
            if placements.len() > clipped {
                total_original.fetch_add(placements.len() - clipped, Ordering::Relaxed);
            }
            placements
        })
        .collect();

    // Group by spatial tile.
    let mut spatial: HashMap<(u32, u32), Vec<TileFeature>> = HashMap::new();
    for (x, y, f) in placed {
        spatial.entry((x, y)).or_default().push(f);
    }

    // Build tiles in parallel: each spatial cell is chunked into temporal
    // buckets, and every (cell, bucket) pair becomes one tile.
    // Per-layer build errors are already logged-and-skipped inside `build_tile`
    // (a bad layer never discards the tile's other features). Any error that
    // still escapes `build_tile` is genuinely tile-fatal, so propagate it rather
    // than silently swallowing a whole tile.
    let tiles: Vec<GeneratedTile> = spatial
        .into_par_iter()
        .map(|((x, y), feats)| -> Result<Vec<GeneratedTile>> {
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
                if let Some(tile) =
                    build_tile(id, &chunk, config, bucket_start as i64, time_end as i64)?
                {
                    out.push(tile);
                }
            }
            Ok(out)
        })
        .collect::<Result<Vec<Vec<GeneratedTile>>>>()?
        .into_iter()
        .flatten()
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
        } else if current.len() >= target && f.timestamp() != current.last().unwrap().timestamp() {
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
        Some(keep) if keep.len() < features.len() => keep.iter().map(|&i| &features[i]).collect(),
        _ => features.iter().collect(),
    };

    let mut originals: Vec<&ParsedFeature> = Vec::new();
    let mut segments: Vec<&ClippedSegment> = Vec::new();
    for f in &kept_features {
        match f {
            TileFeature::Original(o) => originals.push(o),
            TileFeature::Derived(d) => originals.push(d),
            TileFeature::Clipped(s) => segments.push(s),
        }
    }

    let mut layers: Vec<ColumnarLayer> = Vec::new();

    // A failure building ONE layer must not discard the whole tile's other
    // features (the clipped-segment layer and the whole-feature layer are
    // independent). Log the failing layer's feature count and skip just that
    // layer; if every layer fails the tile collapses to `Ok(None)` below,
    // exactly as an empty tile would.
    if !segments.is_empty() {
        match build_layer_from_segments(&segments, &config.layer_name, &config.columnar_options()) {
            Ok(layer) => layers.push(layer),
            Err(e) => tracing::warn!(
                "tile {id:?}: dropping {} clipped-segment feature(s); layer build failed: {e}",
                segments.len()
            ),
        }
    }
    if !originals.is_empty() {
        // Suffix the originals layer name when clipped segments are also
        // present so layer names stay unique within the tile.
        let base = if segments.is_empty() {
            config.layer_name.clone()
        } else {
            format!("{}_originals", config.layer_name)
        };
        match build_layers_from_features_with(&originals, &base, config.columnar_options()) {
            Ok(built) => layers.extend(built),
            Err(e) => tracing::warn!(
                "tile {id:?}: dropping {} whole-feature(s); layer build failed: {e}",
                originals.len()
            ),
        }
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
/// 16 bytes per coordinate pair + 16 per property + 32 of metadata overhead —
/// the same arithmetic the byte cap in `stt_core::budget` is calibrated against,
/// so the two stay comparable.
fn tile_feature_size(f: &TileFeature) -> usize {
    let (verts, props) = tile_feature_signals(f);
    verts * 16 + props * 16 + 32
}

/// `(vertex_count, property_count)` signals `TileBudget::score_signals` needs,
/// read straight off the columnar `TileFeature` — no owned feature struct is
/// ever materialised on the build path.
fn tile_feature_signals(f: &TileFeature) -> (usize, usize) {
    match f {
        TileFeature::Original(o) => {
            let verts = geojson_vertex_count(&o.geojson);
            let props = o.shared_properties.as_ref().map(|p| p.len()).unwrap_or(0);
            (verts, props)
        }
        TileFeature::Derived(d) => {
            let verts = geojson_vertex_count(&d.geojson);
            let props = d.shared_properties.as_ref().map(|p| p.len()).unwrap_or(0);
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
fn apply_tile_budget(budget: &TileBudget, id: TileId, features: &[TileFeature]) -> Vec<usize> {
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

/// Tiles per parallel-encode batch: enough to keep every worker busy, small
/// enough that the batch's encoded (uncompressed) payloads are a bounded,
/// transient allocation.
const ENCODE_CHUNK: usize = 1024;

/// Encode tiles to Arrow IPC payloads IN PARALLEL (on the current rayon
/// pool), then hand them to the pack writer strictly in the given order.
///
/// This is the shared engine behind every `PackWriter` write loop (plain,
/// LOD-tagged, streaming). Parallelism cannot change output bytes:
/// `encode_tile_with` is deterministic per tile, the v2 schema-template
/// collector snapshot is sorted + deduped regardless of insertion order
/// (packed-v2 design ★F1), and `add_tile_full` is called in exactly the
/// sequential order of `tiles` (and the writer's finalize re-sorts by the
/// total space-time key anyway). `on_written` fires once per tile after its
/// ordered hand-off (progress reporting).
///
/// The writer's `--pack-memory-budget` covers the payloads it has BUFFERED,
/// but a full [`ENCODE_CHUNK`] would additionally hold up to 1024 encoded
/// payloads invisible to that budget. With a budget set, sub-batches are cut
/// on a running encoded-byte cap (~budget/4, growing one worker-wave of tiles
/// at a time so parallelism never collapses) and each sub-batch is flushed to
/// `add_tile_full` before the next is encoded. Hand-off order is unchanged,
/// so output bytes are identical either way (pinned by
/// `parallel_encode_writes_byte_identical_dataset`).
fn encode_and_add_tiles(
    writer: &mut stt_core::PackWriter,
    tiles: &[(&GeneratedTile, Option<u64>)],
    mut on_written: impl FnMut(),
) -> Result<()> {
    let encoder = writer.encoder_config();
    let budget = writer.memory_budget();
    let byte_cap: u64 = if budget > 0 {
        (budget / 4).max(1)
    } else {
        u64::MAX
    };
    // Unlimited budget: one wave = the whole chunk (the legacy single
    // par_iter). Budgeted: waves of `workers` tiles, so at most one wave's
    // encoded bytes overshoot the cap.
    let wave = if budget > 0 {
        rayon::current_num_threads().max(1)
    } else {
        ENCODE_CHUNK
    };
    for chunk in tiles.chunks(ENCODE_CHUNK) {
        let mut start = 0usize;
        while start < chunk.len() {
            // Grow the sub-batch wave by wave until the running encoded-byte
            // total reaches the cap (always at least one wave).
            let mut payloads: Vec<Vec<u8>> = Vec::new();
            let mut bytes = 0u64;
            let mut end = start;
            while end < chunk.len() && (end == start || bytes < byte_cap) {
                let wave_end = (end + wave).min(chunk.len());
                let encoded: Vec<Vec<u8>> = chunk[end..wave_end]
                    .par_iter()
                    .map(|(tile, _)| stt_core::arrow_tile::encode_tile_with(&tile.layers, &encoder))
                    .collect::<std::result::Result<Vec<_>, stt_core::Error>>()?;
                bytes += encoded.iter().map(|p| p.len() as u64).sum::<u64>();
                payloads.extend(encoded);
                end = wave_end;
            }
            for ((tile, bucket), payload) in chunk[start..end].iter().zip(payloads) {
                writer.add_tile_full(
                    &tile.id,
                    tile.time_start,
                    tile.time_end,
                    Some(tile.cover_t_min),
                    tile.feature_count(),
                    *bucket,
                    &payload,
                )?;
                on_written();
            }
            start = end;
        }
    }
    Ok(())
}

/// Parallel-encode `tiles` (bounded by a `workers`-sized rayon pool — the
/// `--workers` convention) and write them to `writer` in the given order.
/// The CLI's non-streaming write loops call this; `bucket` tags each tile's
/// directory entry with its temporal bucket (`None` = base tile). See
/// [`encode_and_add_tiles`] for why parallelism can't change output bytes.
pub fn write_tiles_parallel(
    writer: &mut stt_core::PackWriter,
    tiles: &[(&GeneratedTile, Option<u64>)],
    workers: usize,
    on_written: impl FnMut() + Send,
) -> Result<()> {
    let pool = build_pool(workers)?;
    pool.install(|| encode_and_add_tiles(writer, tiles, on_written))
}

/// Stream generated tiles straight into a packed-format [`stt_core::PackWriter`].
///
/// Identical mapping to the `ArchiveWriter` impl above —
/// `PackWriter` shares the same `add_tile_full` contract; it just buffers the
/// tiles and cuts them into content-addressed packs at finalize. Payloads are
/// encoded with [`stt_core::PackWriter::encoder_config`] — the writer carries
/// the resolved `--quantize-* / --vector-group / --point-elevation-column`
/// settings plus its own frame version and template sink, so no encode path
/// here reads process-wide state.
impl TileWriter for stt_core::PackWriter {
    fn write_tile(&mut self, tile: &GeneratedTile) -> Result<()> {
        let payload = stt_core::arrow_tile::encode_tile_with(&tile.layers, &self.encoder_config())?;
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

    /// Batch write with parallel encode (on the caller's current rayon pool)
    /// and strictly ordered hand-off — see [`encode_and_add_tiles`].
    fn write_tiles(&mut self, tiles: &[&GeneratedTile]) -> Result<()> {
        let tagged: Vec<(&GeneratedTile, Option<u64>)> = tiles.iter().map(|t| (*t, None)).collect();
        encode_and_add_tiles(self, &tagged, || {})
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

impl LodTileWriter for stt_core::PackWriter {
    fn write_lod_tile(
        &mut self,
        tile: &GeneratedTile,
        temporal_bucket_ms: Option<u64>,
    ) -> Result<()> {
        let payload = stt_core::arrow_tile::encode_tile_with(&tile.layers, &self.encoder_config())?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use geojson::{Feature, Geometry, Value as GeomValue};
    use stt_core::metadata::Metadata;
    use stt_core::{BlobOrdering, PackWriter, PackedReader};

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

    /// `encode_single_tile` must produce a decodable STT blob containing exactly
    /// the features that fall in the requested `(z, x, y, bucket)` — the same
    /// selection the full build's `process_zoom_level` makes — and `None` for an
    /// empty tile. This is the core a dynamic per-request server (stt-serve)
    /// relies on, verified here with no database.
    #[test]
    fn encode_single_tile_selects_tile_and_bucket() {
        use stt_core::arrow_tile::decode_tile;
        let z = 12u8;
        let bucket_ms = 3_600_000u64; // 1h
        let lon = -122.42;
        let lat = 37.77;
        let (x, y) = projection::lonlat_to_tile(lon, lat, z).unwrap();
        let base = 1_700_000_000_000u64;
        let bucket_start = (base / bucket_ms) * bucket_ms;

        let feats = vec![
            point(lon, lat, bucket_start + 10),
            point(lon + 0.0003, lat + 0.0003, bucket_start + 20),
            // A third point one bucket later (same tile, different time bucket).
            point(lon, lat, bucket_start + bucket_ms + 5),
        ];
        let config = TileConfig {
            min_zoom: z,
            max_zoom: z,
            layer_name: "obs".to_string(),
            temporal_bucket_ms: bucket_ms,
            clip_trajectories: false,
            ..TileConfig::default()
        };

        let enc = stt_core::arrow_tile::EncoderConfig::default();

        // The requested tile + bucket has exactly the two in-bucket points.
        let bytes = encode_single_tile(&feats, z, x, y, bucket_start as i64, &config, &enc)
            .unwrap()
            .expect("tile should be non-empty");
        let rows: usize = decode_tile(&bytes)
            .unwrap()
            .iter()
            .map(|l| l.batch.num_rows())
            .sum();
        assert_eq!(rows, 2, "only the two points in this (tile, bucket)");

        // A different spatial cell is empty.
        assert!(
            encode_single_tile(&feats, z, x + 9, y, bucket_start as i64, &config, &enc)
                .unwrap()
                .is_none()
        );

        // The next bucket carries the single later point.
        let next = encode_single_tile(
            &feats,
            z,
            x,
            y,
            (bucket_start + bucket_ms) as i64,
            &config,
            &enc,
        )
        .unwrap()
        .expect("next bucket tile");
        let n: usize = decode_tile(&next)
            .unwrap()
            .iter()
            .map(|l| l.batch.num_rows())
            .sum();
        assert_eq!(n, 1);
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

    /// A `max_zoom_field` ceiling (paired with `min_zoom_field`) confines a
    /// feature to a single-zoom band: present at zoom == its band, absent above
    /// AND below. This is what keeps coarse-zoom clustered corridors out of the
    /// full-resolution deep zooms.
    #[test]
    fn max_zoom_field_confines_feature_to_band() {
        let mut p = point(-73.98, 40.75, 1_600_000_000_000);
        {
            let props = std::sync::Arc::make_mut(p.shared_properties.as_mut().unwrap());
            props.insert("min_zoom".to_string(), serde_json::json!(11));
            props.insert("max_zoom".to_string(), serde_json::json!(11));
        }
        let config = TileConfig {
            min_zoom: 10,
            max_zoom: 12,
            layer_name: "flows".to_string(),
            min_zoom_field: Some("min_zoom".to_string()),
            max_zoom_field: Some("max_zoom".to_string()),
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[p], &config, 1).unwrap();
        let zooms: Vec<u8> = tiles.iter().map(|t| t.id.z).collect();
        assert_eq!(
            zooms,
            vec![11],
            "feature must appear only at its single-zoom band, got {zooms:?}"
        );
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

        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            PackWriter::create(dir.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
        for tile in &tiles {
            writer.write_tile(tile).unwrap();
        }
        writer.finalize(&Metadata::new("cover")).unwrap();

        let reader = PackedReader::open(dir.path().join("manifest.json")).unwrap();
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

        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            PackWriter::create(dir.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
        for tile in &tiles {
            writer.write_tile(tile).unwrap();
        }
        let total_features: usize = tiles.iter().map(|t| t.feature_count() as usize).sum();
        writer.finalize(&Metadata::new("e2e-points")).unwrap();

        let reader = PackedReader::open(dir.path().join("manifest.json")).unwrap();
        assert_eq!(reader.entries().len(), tiles.len());

        // Every feature is represented somewhere (summed over all tiles).
        let archived: usize = reader
            .entries()
            .iter()
            .map(|e| e.feature_count as usize)
            .sum();
        assert_eq!(archived, total_features);

        // Decode one tile and confirm its Arrow layer is intact.
        let entry = reader.entries()[0].clone();
        let layers = reader.read_layers(&entry).unwrap();
        assert!(!layers.is_empty());
        assert!(layers[0].batch.num_rows() > 0);
        assert!(layers[0].batch.column_by_name("geometry").is_some());
        assert!(layers[0].batch.column_by_name("v").is_some());
    }

    /// The parallel-encode batch path (`write_tiles_parallel`) must produce a
    /// byte-identical dataset to the sequential per-tile `write_tile` path —
    /// same directory hash, same pack hashes, same manifest. This is the
    /// unit-level pin for the E1 "parallel encode, deterministic write order"
    /// contract.
    #[test]
    fn parallel_encode_writes_byte_identical_dataset() {
        let hour = 3_600_000u64;
        let mut features = Vec::new();
        for i in 0..120u64 {
            let lon = -122.45 + (i % 12) as f64 * 0.01;
            let lat = 37.75 + (i / 12) as f64 * 0.01;
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
        let tiles = generate_tiles(&features, &config, 2).unwrap();
        assert!(tiles.len() > 4, "want several tiles, got {}", tiles.len());

        // Sequential reference.
        let dir_seq = tempfile::tempdir().unwrap();
        let mut w_seq =
            PackWriter::create(dir_seq.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
        for tile in &tiles {
            w_seq.write_tile(tile).unwrap();
        }
        let m_seq = w_seq.finalize(&Metadata::new("par-enc")).unwrap();

        // Parallel encode, ordered hand-off (4 workers).
        let dir_par = tempfile::tempdir().unwrap();
        let mut w_par =
            PackWriter::create(dir_par.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
        let tagged: Vec<(&GeneratedTile, Option<u64>)> = tiles.iter().map(|t| (t, None)).collect();
        let mut written = 0usize;
        write_tiles_parallel(&mut w_par, &tagged, 4, || written += 1).unwrap();
        assert_eq!(written, tiles.len());
        let m_par = w_par.finalize(&Metadata::new("par-enc")).unwrap();

        assert_eq!(
            m_seq.directory.key, m_par.directory.key,
            "directory hash differs"
        );
        assert_eq!(
            m_seq.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            m_par.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            "pack hashes differ"
        );
        assert_eq!(
            m_seq.to_json_bytes().unwrap(),
            m_par.to_json_bytes().unwrap()
        );

        // Parallel encode under a TINY writer memory budget: the encode loop
        // now cuts sub-batches on the ~budget/4 encoded-byte cap (and the
        // writer itself spills), but the hand-off order — and therefore every
        // output byte — must not change.
        let dir_bud = tempfile::tempdir().unwrap();
        let mut w_bud = PackWriter::create(dir_bud.path(), BlobOrdering::Auto, 64 * 1024 * 1024)
            .unwrap()
            .with_memory_budget(1024);
        let mut written_bud = 0usize;
        write_tiles_parallel(&mut w_bud, &tagged, 4, || written_bud += 1).unwrap();
        assert_eq!(written_bud, tiles.len());
        let m_bud = w_bud.finalize(&Metadata::new("par-enc")).unwrap();
        assert_eq!(
            m_seq.directory.key, m_bud.directory.key,
            "directory hash must not depend on the encode-batch budget"
        );
        assert_eq!(
            m_seq.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            m_bud.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            "pack hashes must not depend on the encode-batch budget"
        );
        assert_eq!(
            m_seq.to_json_bytes().unwrap(),
            m_bud.to_json_bytes().unwrap()
        );
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

        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            PackWriter::create(dir.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
        for tile in &tiles {
            writer.write_tile(tile).unwrap();
        }
        writer.finalize(&Metadata::new("e2e-tracks")).unwrap();

        let reader = PackedReader::open(dir.path().join("manifest.json")).unwrap();
        let entry = reader.entries()[0].clone();
        let layers = reader.read_layers(&entry).unwrap();
        // Clipped segments are linestrings carrying a vertex_time column.
        assert!(layers[0].batch.column_by_name("vertex_time").is_some());
        assert!(layers[0].batch.column_by_name("geometry").is_some());
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
                TemporalLodLevel {
                    bucket_ms: 24 * 3_600_000,
                    max_zoom_level: 6,
                },
                TemporalLodLevel {
                    bucket_ms: 2 * 3_600_000,
                    max_zoom_level: 6,
                },
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
            temporal_lod: vec![TemporalLodLevel {
                bucket_ms: day,
                max_zoom_level: 9,
            }],
            ..TileConfig::default()
        };
        let tagged = generate_tiles_with_lod(&features, &config, 1).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            PackWriter::create(dir.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
        for t in &tagged {
            writer
                .write_lod_tile(&t.tile, t.temporal_bucket_ms)
                .unwrap();
        }
        let metadata = stt_core::metadata::Metadata::new("lod")
            .with_temporal_bucket_ms(hour)
            .with_temporal_lod(vec![TemporalLodLevel {
                bucket_ms: day,
                max_zoom_level: 9,
            }])
            .unwrap();
        writer.finalize(&metadata).unwrap();

        let reader = PackedReader::open(dir.path().join("manifest.json")).unwrap();
        let buckets: std::collections::BTreeSet<Option<u64>> = reader
            .entries()
            .iter()
            .map(|e| e.temporal_bucket_ms)
            .collect();
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
        assert_eq!(
            tiles.len(),
            10,
            "expected 10 windows of 10, got {}",
            tiles.len()
        );

        let keys: std::collections::BTreeSet<(u8, u32, u32, i64)> = tiles
            .iter()
            .map(|t| (t.id.z, t.id.x, t.id.y, t.time_start))
            .collect();
        assert_eq!(keys.len(), tiles.len(), "window keys must be distinct");
        for t in &tiles {
            assert!(
                t.feature_count() <= 11,
                "window over budget: {}",
                t.feature_count()
            );
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
        assert_eq!(
            tiles.len(),
            1,
            "identical-timestamp features can't be split into tiles"
        );
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
            simplify_metric: false,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        assert!(
            !tiles.is_empty(),
            "time-aware simplify should still produce tiles"
        );
        // Clipped trajectory layers carry per-vertex times (TD-TR preserved them).
        for t in &tiles {
            for l in &t.layers {
                assert!(
                    l.vertex_times.is_some(),
                    "trajectory layer should carry vertex_times"
                );
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
        assert!(
            total >= 1,
            "byte cap should still keep at least one feature"
        );
    }

    // ------------------------------------------------------------------
    // Non-trajectory clipping (T1.1) + projection-failure drops (T1.2)
    // ------------------------------------------------------------------

    fn square_ring(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Vec<Vec<f64>> {
        vec![
            vec![min_lon, min_lat],
            vec![max_lon, min_lat],
            vec![max_lon, max_lat],
            vec![min_lon, max_lat],
            vec![min_lon, min_lat],
        ]
    }

    fn polygon_feature(rings: Vec<Vec<Vec<f64>>>, ts: u64) -> ParsedFeature {
        let first = rings[0][0].clone();
        ParsedFeature {
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::Polygon(rings))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: ts,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: first[0],
            lat: first[1],
        }
    }

    /// Every ring of every polygon feature in the tile must be closed, non-
    /// degenerate, and stay inside the tile's buffered rect.
    fn assert_valid_polygon_rings(tile: &GeneratedTile) {
        use stt_core::arrow_tile::GeometryColumn;
        let (min_lon, min_lat, max_lon, max_lat) =
            stt_core::projection::tile_geo_bounds(tile.id.z, tile.id.x, tile.id.y);
        let (bl, bb, br, bt) = (
            min_lon - 0.001 - 1e-9,
            min_lat - 0.001 - 1e-9,
            max_lon + 0.001 + 1e-9,
            max_lat + 0.001 + 1e-9,
        );
        for layer in &tile.layers {
            let GeometryColumn::Polygon(features) = &layer.geometry else {
                panic!("expected polygon geometry in {:?}", tile.id);
            };
            for rings in features {
                assert!(!rings.is_empty(), "feature with no rings in {:?}", tile.id);
                for ring in rings {
                    assert!(
                        ring.len() >= 4,
                        "degenerate ring in {:?}: {ring:?}",
                        tile.id
                    );
                    assert_eq!(ring.first(), ring.last(), "unclosed ring in {:?}", tile.id);
                    for [lon, lat] in ring {
                        assert!(
                            *lon >= bl && *lon <= br && *lat >= bb && *lat <= bt,
                            "vertex ({lon}, {lat}) escapes buffered bounds of {:?}",
                            tile.id
                        );
                    }
                }
            }
        }
    }

    /// T1.1 headline case: a polygon straddling the 4-tile corner at (0°, 0°)
    /// must appear — clipped, with valid rings — in ALL 4 tiles, not just the
    /// one holding its representative point.
    #[test]
    fn polygon_spanning_four_tiles_is_clipped_into_each() {
        let feat = polygon_feature(vec![square_ring(-0.1, -0.1, 0.1, 0.1)], 1_700_000_000_000);
        let config = TileConfig {
            min_zoom: 10,
            max_zoom: 10,
            layer_name: "areas".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        let mut cells: Vec<(u32, u32, u32)> = tiles
            .iter()
            .map(|t| (t.id.x, t.id.y, t.feature_count()))
            .collect();
        cells.sort();
        assert_eq!(
            cells,
            vec![(511, 511, 1), (511, 512, 1), (512, 511, 1), (512, 512, 1)],
            "polygon around (0°,0°) must be present in all 4 corner tiles"
        );
        for tile in &tiles {
            assert_valid_polygon_rings(tile);
        }
    }

    /// A polygon with a hole keeps the hole in every tile it spans (rings are
    /// clipped independently) and the multi-ring pieces auto-bake the
    /// hole-aware triangle sidecar.
    #[test]
    fn polygon_with_hole_keeps_hole_in_every_tile() {
        use stt_core::arrow_tile::GeometryColumn;
        let feat = polygon_feature(
            vec![
                square_ring(-0.1, -0.1, 0.1, 0.1),
                square_ring(-0.05, -0.05, 0.05, 0.05),
            ],
            1_700_000_000_000,
        );
        let config = TileConfig {
            min_zoom: 10,
            max_zoom: 10,
            layer_name: "areas".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        assert_eq!(
            tiles.len(),
            4,
            "holed polygon still covers the 4 corner tiles"
        );
        for tile in &tiles {
            assert_valid_polygon_rings(tile);
            let layer = &tile.layers[0];
            let GeometryColumn::Polygon(features) = &layer.geometry else {
                panic!("expected polygon geometry");
            };
            assert_eq!(features.len(), 1);
            assert_eq!(
                features[0].len(),
                2,
                "tile {:?} lost the hole ring: {} ring(s)",
                tile.id,
                features[0].len()
            );
            assert!(
                layer.triangles.is_some(),
                "multi-ring piece must auto-bake hole-aware triangles"
            );
        }
    }

    /// FAST PATH: a polygon fully inside one buffered tile is placed whole in
    /// that single tile — the clipper must not split it or rewrite its ring.
    #[test]
    fn fully_inside_polygon_is_placed_whole_in_one_tile() {
        let ts = 1_700_000_000_000u64;
        let feat = polygon_feature(vec![square_ring(-122.5, 37.8, -122.45, 37.85)], ts);
        let ring = match feat.geojson.geometry.as_ref().map(|g| &g.value) {
            Some(GeomValue::Polygon(rings)) => rings[0].clone(),
            other => panic!("expected a polygon fixture, got {other:?}"),
        };
        let config = TileConfig {
            min_zoom: 10,
            max_zoom: 10,
            layer_name: "areas".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        assert_eq!(tiles.len(), 1, "a fully-inside polygon must not be split");
        assert_eq!(tiles[0].feature_count(), 1);

        // The ring survives untouched: the fast path skips the clipper, so no
        // tile-cut vertices are synthesized and no coordinate is moved.
        let bytes = encode_tile(&tiles[0].layers).unwrap();
        let decoded = stt_core::arrow_tile::decode_tile(&bytes).unwrap();
        let geom = &decoded[0].batch;
        assert_eq!(geom.num_rows(), 1);
        assert_eq!(
            ring.len(),
            match &tiles[0].layers[0].geometry {
                stt_core::arrow_tile::GeometryColumn::Polygon(polys) => polys[0][0].len(),
                other => panic!("expected a polygon column, got {other:?}"),
            },
            "the placed ring must keep its original vertex count"
        );
    }

    /// A timeless LineString (no duration) spanning several tiles must be
    /// present — spatially clipped, with NO vertex_time column (timeless
    /// semantics preserved) — in each of them.
    #[test]
    fn timeless_line_spanning_tiles_present_in_each() {
        use stt_core::arrow_tile::GeometryColumn;
        let coords = vec![vec![-0.1, 0.05], vec![0.1, 0.05]];
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
            timestamp: 1_700_000_000_000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: first[0],
            lat: first[1],
        };
        let config = TileConfig {
            min_zoom: 10,
            max_zoom: 10,
            layer_name: "roads".to_string(),
            temporal_bucket_ms: 3_600_000,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        let mut cells: Vec<(u32, u32)> = tiles.iter().map(|t| (t.id.x, t.id.y)).collect();
        cells.sort();
        assert_eq!(
            cells,
            vec![(511, 511), (512, 511)],
            "timeless line crossing lon 0 must be present in both tiles"
        );
        for tile in &tiles {
            assert_eq!(tile.feature_count(), 1);
            let layer = &tile.layers[0];
            assert!(
                layer.vertex_times.is_none(),
                "timeless line must not grow a vertex_time column"
            );
            let GeometryColumn::LineString(lines) = &layer.geometry else {
                panic!("expected linestring geometry");
            };
            assert!(lines[0].len() >= 2, "clipped line must keep >=2 vertices");
        }
    }

    /// MultiPoint members are split per containing tile — every member is
    /// rendered at its own position (the legacy path placed, and rendered,
    /// only the whole feature's representative point).
    #[test]
    fn multipoint_members_split_per_containing_tile() {
        use stt_core::arrow_tile::GeometryColumn;
        let members = vec![vec![-0.1, 0.05], vec![0.1, 0.05]];
        let feat = ParsedFeature {
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::MultiPoint(members.clone()))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 1_700_000_000_000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: members[0][0],
            lat: members[0][1],
        };
        let config = TileConfig {
            min_zoom: 10,
            max_zoom: 10,
            layer_name: "stations".to_string(),
            temporal_bucket_ms: 3_600_000,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        let mut cells: Vec<(u32, u32)> = tiles.iter().map(|t| (t.id.x, t.id.y)).collect();
        cells.sort();
        assert_eq!(cells, vec![(511, 511), (512, 511)], "one tile per member");
        let mut seen: Vec<[f64; 2]> = Vec::new();
        for tile in &tiles {
            assert_eq!(tile.feature_count(), 1);
            let GeometryColumn::Point(pts) = &tile.layers[0].geometry else {
                panic!("expected point geometry");
            };
            seen.push(pts[0]);
        }
        seen.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert_eq!(
            seen,
            vec![[-0.1, 0.05], [0.1, 0.05]],
            "each tile's point must sit at its member's own position"
        );
    }

    /// A duration MultiLineString routes each part through the trajectory
    /// clipper: pieces land in every crossed tile, carry vertex times, and
    /// share the parent's stable feature id.
    #[test]
    fn duration_multilinestring_clips_parts_as_segment_runs() {
        let parts = vec![
            vec![vec![-0.1, 0.05], vec![0.1, 0.05]],
            vec![vec![-0.1, 0.02], vec![0.1, 0.02]],
        ];
        let feat = ParsedFeature {
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::MultiLineString(parts))),
                id: Some(geojson::feature::Id::String("mls-1".to_string())),
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 0,
            end_timestamp: Some(3_600_000),
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: -0.1,
            lat: 0.05,
        };
        let config = TileConfig {
            min_zoom: 10,
            max_zoom: 10,
            layer_name: "tracks".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: true,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        let xs: std::collections::BTreeSet<u32> = tiles.iter().map(|t| t.id.x).collect();
        assert!(
            xs.len() >= 2,
            "duration MultiLineString must span multiple tile columns, got {xs:?}"
        );
        let mut ids: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for tile in &tiles {
            for layer in &tile.layers {
                assert!(
                    layer.vertex_times.is_some(),
                    "trajectory pieces must carry vertex times"
                );
                ids.extend(layer.feature_ids.iter().copied());
            }
        }
        assert_eq!(
            ids.len(),
            1,
            "all pieces must share the parent's stable feature id, got {ids:?}"
        );
    }

    /// Capture writer for streaming-stats assertions.
    struct CaptureWriter(Vec<(u8, u32, u32, u32)>);
    impl TileWriter for CaptureWriter {
        fn write_tile(&mut self, tile: &GeneratedTile) -> Result<()> {
            self.0
                .push((tile.id.z, tile.id.x, tile.id.y, tile.feature_count()));
            Ok(())
        }
    }

    /// A latitude beyond the Web-Mercator limit is CLAMPED into the edge tile
    /// row and KEPT — never dropped.
    ///
    /// The summary tier has always clamped-and-kept such a feature while the
    /// native tier dropped it, so one dataset answered "does this feature
    /// exist?" two different ways depending on which tier you asked. Both tiers
    /// now clamp to the single `stt_core::projection::MERCATOR_MAX_LAT`.
    #[test]
    fn polar_latitudes_are_clamped_into_the_edge_row_not_dropped() {
        let sf = point(-122.4194, 37.7749, 1_700_000_000_000);
        // Far beyond the limit, and beyond the ~3 m band between the old
        // rounded native limit and the old Quadbin one.
        let polar = point(0.0, 89.9, 1_700_000_000_000);
        let config = TileConfig {
            min_zoom: 2,
            max_zoom: 5,
            layer_name: "default".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            ..TileConfig::default()
        };
        let mut writer = CaptureWriter(Vec::new());
        let stats = generate_tiles_streaming(&[sf, polar], &config, &mut writer, 1).unwrap();
        assert_eq!(
            stats.dropped_invalid_coords, 0,
            "a polar latitude is projectable (clamped), not invalid"
        );
        assert_eq!(
            writer.0.len(),
            8,
            "two features × zooms 2..=5, each in its own tile"
        );
        for z in 2u8..=5 {
            let n = 1u32 << z;
            let edge: Vec<_> = writer
                .0
                .iter()
                .filter(|(tz, _, y, _)| *tz == z && *y == 0)
                .collect();
            assert_eq!(
                edge.len(),
                1,
                "z{z}: exactly one tile in the polar edge row"
            );
            assert_eq!(
                (edge[0].1, edge[0].3),
                (n / 2, 1),
                "z{z}: the clamped feature owns the lon-0 column of row 0"
            );
        }
    }

    /// T1.2: a feature whose coordinates cannot be projected AT ALL (here an
    /// out-of-range longitude) is DROPPED and COUNTED — it never lands in tile
    /// (0, 0) as a phantom.
    #[test]
    fn invalid_coordinates_are_dropped_and_counted() {
        let good = point(-122.4194, 37.7749, 1_700_000_000_000);
        let bad = point(200.0, 10.0, 1_700_000_000_000); // lon outside [-180, 180]
        let config = TileConfig {
            min_zoom: 0,
            max_zoom: 5,
            layer_name: "default".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            ..TileConfig::default()
        };
        let mut writer = CaptureWriter(Vec::new());
        let stats = generate_tiles_streaming(&[good, bad], &config, &mut writer, 1).unwrap();
        // Counted once per (feature, zoom): zooms 0..=5.
        assert_eq!(
            stats.dropped_invalid_coords, 6,
            "drop must be counted per zoom"
        );
        assert_eq!(
            writer.0.len(),
            6,
            "one tile per zoom for the valid point only"
        );
        for (z, x, y, count) in &writer.0 {
            assert_eq!(*count, 1, "phantom feature leaked into z{z}/{x}/{y}");
            let (ex, ey) = projection::lonlat_to_tile(-122.4194, 37.7749, *z).unwrap();
            assert_eq!((*x, *y), (ex, ey), "tile must be the valid point's tile");
        }
    }

    /// A dateline-crossing polygon SPLITS (split-then-reuse) and lands —
    /// clipped, with valid rings — on BOTH sides of ±180 across the zoom range,
    /// with ZERO antimeridian fallbacks and NO single-tile smear / world span.
    /// This replaces the retired single-tile-fallback pin (`place_polygon` now
    /// splits crossing rings instead of dead-lettering them).
    #[test]
    fn antimeridian_polygon_splits_across_dateline_no_fallback() {
        // Fiji-like square straddling +180: raw lon 177 → -177, lat 10 → 20.
        let crossing_ring = || {
            vec![
                vec![177.0, 10.0],
                vec![-177.0, 10.0],
                vec![-177.0, 20.0],
                vec![177.0, 20.0],
                vec![177.0, 10.0],
            ]
        };
        let (min_zoom, max_zoom) = (2u8, 6u8);
        let config = TileConfig {
            min_zoom,
            max_zoom,
            layer_name: "areas".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            ..TileConfig::default()
        };

        // Stats path: zero fallbacks anywhere in the zoom range (the campaign
        // gate — a healthy dateline corpus never dead-letters).
        let mut writer = CaptureWriter(Vec::new());
        let stats = generate_tiles_streaming(
            &[polygon_feature(vec![crossing_ring()], 1_700_000_000_000)],
            &config,
            &mut writer,
            1,
        )
        .unwrap();
        assert_eq!(
            stats.antimeridian_fallbacks, 0,
            "dateline polygon must split, never fall back"
        );

        // Geometry path: every emitted ring is valid (closed, in-buffered-bounds).
        let tiles = generate_tiles(
            &[polygon_feature(vec![crossing_ring()], 1_700_000_000_000)],
            &config,
            1,
        )
        .unwrap();
        assert!(!tiles.is_empty());
        for tile in &tiles {
            assert_valid_polygon_rings(tile);
        }

        // At the deepest zoom the two pieces land in OPPOSITE-edge columns
        // (west near +180, east near -180) and NOT the interior around lon 0 —
        // proof of a real split, not a single-tile smear or a world span.
        let z = max_zoom;
        let west_col = projection::lonlat_to_tile(178.0, 15.0, z).unwrap().0;
        let east_col = projection::lonlat_to_tile(-178.0, 15.0, z).unwrap().0;
        let mid_col = projection::lonlat_to_tile(0.0, 15.0, z).unwrap().0;
        let cols: std::collections::BTreeSet<u32> = tiles
            .iter()
            .filter(|t| t.id.z == z)
            .map(|t| t.id.x)
            .collect();
        assert!(
            west_col != east_col,
            "test setup: hemispheres must differ in column"
        );
        assert!(
            cols.contains(&west_col),
            "expected the west (+180) column {west_col} in {cols:?}"
        );
        assert!(
            cols.contains(&east_col),
            "expected the east (-180) column {east_col} in {cols:?}"
        );
        assert!(
            !cols.contains(&mid_col),
            "polygon smeared into the interior column {mid_col} (world span)"
        );

        // CCW exterior / CW holes on the split pieces survive per-tile clipping
        // (Sutherland–Hodgman preserves winding).
        for tile in &tiles {
            for layer in &tile.layers {
                let stt_core::arrow_tile::GeometryColumn::Polygon(features) = &layer.geometry
                else {
                    panic!("expected polygon geometry");
                };
                for rings in features {
                    for (i, ring) in rings.iter().enumerate() {
                        let area = ring_signed_area(ring);
                        // Skip seam-degenerate slivers (a fragment reduced to a
                        // near-zero-area strip against a tile edge).
                        if area.abs() <= 1e-12 {
                            continue;
                        }
                        if i == 0 {
                            assert!(area > 0.0, "exterior must be CCW in {:?}", tile.id);
                        } else {
                            assert!(area < 0.0, "hole must be CW in {:?}", tile.id);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn antimeridian_multipolygon_keeps_noncrossing_part() {
        // Regression (Wave-2 A1 review): a MultiPolygon with ONE dateline-crossing
        // part (Fiji-like, 177 → -177) plus one ordinary NON-crossing part near
        // lon 0 must keep BOTH — the crossing part splits across the dateline
        // while the non-crossing part still tiles at its own interior column.
        // The feature-wide `crosses` flag used to route the whole feature through
        // the splitter, which flat-mapped over every part and silently dropped the
        // non-crosser (a USA MultiPolygon would lose CONUS when the Aleutians cross
        // ±180). Now each part is routed independently.
        let crossing_part = vec![vec![
            vec![177.0, 10.0],
            vec![-177.0, 10.0],
            vec![-177.0, 20.0],
            vec![177.0, 20.0],
            vec![177.0, 10.0],
        ]];
        let noncrossing_part = vec![vec![
            vec![-2.0, 10.0],
            vec![2.0, 10.0],
            vec![2.0, 20.0],
            vec![-2.0, 20.0],
            vec![-2.0, 10.0],
        ]];
        let feature = ParsedFeature {
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::MultiPolygon(vec![
                    crossing_part,
                    noncrossing_part,
                ]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 1_700_000_000_000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: 0.0,
            lat: 15.0,
        };

        let (min_zoom, max_zoom) = (2u8, 6u8);
        let config = TileConfig {
            min_zoom,
            max_zoom,
            layer_name: "areas".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            ..TileConfig::default()
        };

        let mut writer = CaptureWriter(Vec::new());
        let stats =
            generate_tiles_streaming(std::slice::from_ref(&feature), &config, &mut writer, 1)
                .unwrap();
        assert_eq!(
            stats.antimeridian_fallbacks, 0,
            "mixed MultiPolygon must split the crossing part, not dead-letter"
        );

        let tiles = generate_tiles(std::slice::from_ref(&feature), &config, 1).unwrap();
        let z = max_zoom;
        let west_col = projection::lonlat_to_tile(178.0, 15.0, z).unwrap().0;
        let east_col = projection::lonlat_to_tile(-178.0, 15.0, z).unwrap().0;
        let mid_col = projection::lonlat_to_tile(0.0, 15.0, z).unwrap().0;
        let cols: std::collections::BTreeSet<u32> = tiles
            .iter()
            .filter(|t| t.id.z == z)
            .map(|t| t.id.x)
            .collect();
        // The crossing part still splits to BOTH dateline edges …
        assert!(
            cols.contains(&west_col) && cols.contains(&east_col),
            "crossing part must reach both dateline columns ({west_col},{east_col}) in {cols:?}"
        );
        // … AND the non-crossing part is preserved at its own interior column.
        assert!(
            cols.contains(&mid_col),
            "non-crossing MultiPolygon part near lon 0 (col {mid_col}) was dropped; got {cols:?}"
        );
        for tile in &tiles {
            assert_valid_polygon_rings(tile);
        }
    }

    /// Signed shoelace area of a closed `[[lon, lat], …]` ring (positive = CCW).
    fn ring_signed_area(ring: &[[f64; 2]]) -> f64 {
        let mut sum = 0.0;
        for w in ring.windows(2) {
            sum += w[0][0] * w[1][1] - w[1][0] * w[0][1];
        }
        sum / 2.0
    }

    // ------------------------------------------------------------------
    // Per-zoom polygon simplification (A2)
    // ------------------------------------------------------------------

    /// A dense CCW circle ring, closed, sampled `n` times around `(cx, cy)`.
    fn dense_circle_ring(cx: f64, cy: f64, r: f64, n: usize) -> Vec<Vec<f64>> {
        use std::f64::consts::PI;
        let mut ring: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        for i in 0..n {
            let theta = 2.0 * PI * (i as f64 / n as f64);
            ring.push(vec![cx + r * theta.cos(), cy + r * theta.sin()]);
        }
        ring.push(ring[0].clone());
        ring
    }

    /// Total polygon vertices across every polygon feature/ring in every tile.
    fn total_polygon_vertices(tiles: &[GeneratedTile]) -> usize {
        use stt_core::arrow_tile::GeometryColumn;
        let mut n = 0;
        for tile in tiles {
            for layer in &tile.layers {
                if let GeometryColumn::Polygon(features) = &layer.geometry {
                    for rings in features {
                        for ring in rings {
                            n += ring.len();
                        }
                    }
                }
            }
        }
        n
    }

    fn poly_config(
        min_zoom: u8,
        max_zoom: u8,
        simplify: bool,
        simplify_max_zoom: u8,
    ) -> TileConfig {
        TileConfig {
            min_zoom,
            max_zoom,
            layer_name: "areas".to_string(),
            temporal_bucket_ms: 3_600_000,
            clip_trajectories: false,
            simplify,
            simplify_max_zoom,
            ..TileConfig::default()
        }
    }

    /// A dense polygon that fits inside a SINGLE z8 tile — so clipping is an
    /// identity and only simplification can change the vertex count. Centre
    /// (2.0, 2.0) with r=0.02 lands wholly inside z8 tile (129, 126).
    fn dense_single_tile_polygon() -> ParsedFeature {
        polygon_feature(
            vec![dense_circle_ring(2.0, 2.0, 0.02, 240)],
            1_700_000_000_000,
        )
    }

    #[test]
    fn polygon_max_tier_lossless_with_simplify_on() {
        // At the max-tiled-zoom tier (zoom == simplify_max_zoom), turning
        // simplification ON must NOT change a single byte of geometry vs OFF.
        let feat = dense_single_tile_polygon();
        let z = 8u8;

        let on = generate_tiles(&[feat.clone()], &poly_config(z, z, true, z), 1).unwrap();
        let off = generate_tiles(&[feat], &poly_config(z, z, false, z), 1).unwrap();

        assert_eq!(on.len(), 1, "polygon should sit in exactly one z8 tile");
        assert_eq!(off.len(), 1);
        let geom = |tiles: &[GeneratedTile]| {
            use stt_core::arrow_tile::GeometryColumn;
            let GeometryColumn::Polygon(f) = &tiles[0].layers[0].geometry else {
                panic!("expected polygon geometry");
            };
            f.clone()
        };
        assert_eq!(
            geom(&on),
            geom(&off),
            "max-tier geometry must be byte-identical with simplify on vs off"
        );
    }

    #[test]
    fn polygon_simplifies_below_max_tier() {
        // Below simplify_max_zoom the same single-tile polygon is simplified:
        // fewer vertices, feature still present, rings still valid. clip is an
        // identity here, so the drop is purely simplification.
        let feat = dense_single_tile_polygon();
        let z = 8u8;

        let simp = generate_tiles(&[feat.clone()], &poly_config(z, z, true, 14), 1).unwrap();
        let full = generate_tiles(&[feat], &poly_config(z, z, false, 14), 1).unwrap();

        assert_eq!(simp.len(), 1, "feature must not be dropped");
        assert_eq!(simp[0].feature_count(), 1, "the polygon feature survives");
        let simp_v = total_polygon_vertices(&simp);
        let full_v = total_polygon_vertices(&full);
        assert!(
            simp_v < full_v,
            "expected vertex reduction at z8: {simp_v} !< {full_v}"
        );
        for tile in &simp {
            assert_valid_polygon_rings(tile);
        }
    }

    #[test]
    fn polygon_simplify_keeps_hole_and_winding_through_tiler() {
        // A dense holed polygon inside one z8 tile keeps its hole and winding
        // after simplification (exterior CCW, hole CW), rings valid, and the
        // vertex count drops.
        use stt_core::arrow_tile::GeometryColumn;
        let mut exterior = dense_circle_ring(2.0, 2.0, 0.03, 240);
        // Inner hole must wind CW (opposite the exterior) — reverse a CCW ring.
        let mut hole = dense_circle_ring(2.0, 2.0, 0.012, 200);
        hole.reverse();
        exterior.shrink_to_fit();
        hole.shrink_to_fit();
        let feat = polygon_feature(vec![exterior, hole], 1_700_000_000_000);
        let z = 8u8;

        let simp = generate_tiles(&[feat.clone()], &poly_config(z, z, true, 14), 1).unwrap();
        let full = generate_tiles(&[feat], &poly_config(z, z, false, 14), 1).unwrap();
        assert_eq!(simp.len(), 1);
        assert!(total_polygon_vertices(&simp) < total_polygon_vertices(&full));

        for tile in &simp {
            assert_valid_polygon_rings(tile);
            let GeometryColumn::Polygon(features) = &tile.layers[0].geometry else {
                panic!("expected polygon geometry");
            };
            for rings in features {
                assert_eq!(rings.len(), 2, "hole retained through simplification");
                assert!(ring_signed_area(&rings[0]) > 0.0, "exterior must stay CCW");
                assert!(ring_signed_area(&rings[1]) < 0.0, "hole must stay CW");
            }
        }
    }

    /// Regression: the pre-existing antimeridian split tests must stay green
    /// even with simplification enabled — the split runs BEFORE simplify, so a
    /// dateline polygon still lands, valid, on both sides at a simplified zoom.
    #[test]
    fn antimeridian_split_holds_with_simplify_enabled() {
        let crossing_ring = vec![
            vec![177.0, 10.0],
            vec![-177.0, 10.0],
            vec![-177.0, 20.0],
            vec![177.0, 20.0],
            vec![177.0, 10.0],
        ];
        let config = poly_config(2, 6, true, 14);
        let feat = polygon_feature(vec![crossing_ring], 1_700_000_000_000);

        let mut writer = CaptureWriter(Vec::new());
        let stats =
            generate_tiles_streaming(std::slice::from_ref(&feat), &config, &mut writer, 1).unwrap();
        assert_eq!(
            stats.antimeridian_fallbacks, 0,
            "dateline polygon must still split (not dead-letter) with simplify on"
        );

        let tiles = generate_tiles(&[feat], &config, 1).unwrap();
        assert!(!tiles.is_empty());
        let z = 6u8;
        let west_col = projection::lonlat_to_tile(178.0, 15.0, z).unwrap().0;
        let east_col = projection::lonlat_to_tile(-178.0, 15.0, z).unwrap().0;
        let cols: std::collections::BTreeSet<u32> = tiles
            .iter()
            .filter(|t| t.id.z == z)
            .map(|t| t.id.x)
            .collect();
        assert!(
            cols.contains(&west_col) && cols.contains(&east_col),
            "split must reach both dateline columns even with simplify on"
        );
        for tile in &tiles {
            assert_valid_polygon_rings(tile);
        }
    }
}
