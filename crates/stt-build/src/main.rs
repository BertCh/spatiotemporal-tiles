//! stt-build - CLI tool for building spatiotemporal tile archives
//!
//! This tool converts GeoParquet files into optimized STT archives for web visualization.

use stt_build::{input, summary, tiler};
use stt_build::tiler::TileWriter;

use anyhow::{Context, Result};
use clap::{parser::ValueSource, ArgMatches, CommandFactory, FromArgMatches, Parser};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "stt-build")]
#[command(about = "Build spatiotemporal tile archives from GeoParquet data", long_about = None)]
#[command(version)]
struct Args {
    /// Input GeoParquet file path (.parquet or .geoparquet)
    #[arg(short, long)]
    input: PathBuf,

    /// Output packed-dataset DIRECTORY (manifest.json + index/ + packs/).
    ///
    /// stt-build now emits the multi-object **packed format** instead of a
    /// single-file `.stt`. The output is a directory tree; for convenience a
    /// path ending in `.stt` has that extension stripped, so `-o foo.stt`
    /// produces `foo/{manifest.json,index/,packs/}`. A directory-like path is
    /// used as-is.
    #[arg(short, long)]
    output: PathBuf,

    /// Field name containing timestamps (Unix ms or ISO 8601)
    #[arg(short, long, default_value = "timestamp")]
    time_field: String,

    /// Field name containing end timestamps for features with time ranges (optional)
    /// If provided, features will have valid_from (time_field) and valid_to (end_time_field)
    #[arg(long)]
    end_time_field: Option<String>,

    /// Time format of the `--time-field` column. Only consulted for integer
    /// (Int64) columns — Arrow Timestamp columns are self-describing and
    /// String columns are always parsed as ISO 8601.
    #[arg(long, value_enum, default_value = "iso8601")]
    time_format: input::TimeFormat,

    /// Minimum zoom level
    #[arg(long, default_value = "0")]
    min_zoom: u8,

    /// Maximum zoom level
    #[arg(long, default_value = "14")]
    max_zoom: u8,

    /// Compression for tile payloads. The packed format is zstd-only (every
    /// blob is compressed per-blob with zstd so the TS reader can decode it);
    /// the legacy `gzip`/`none` choices were removed and now error.
    #[arg(long, default_value = "zstd")]
    compression: String,

    /// Pack ordering — how tile blobs are laid out before being cut into packs:
    /// `auto` (default; pick from the dataset's space-vs-time cardinality:
    /// wide-time → spatial-major, else the 3D-Hilbert generalist), or an explicit
    /// `spatial`, `time-major`, `hilbert3`, or `morton3`. Locality means fewer
    /// packs touched per viewport (fewer client range requests). The in-memory
    /// pipeline holds all tile payloads in RAM until finalize. With
    /// --streaming-arrow this controls the ordering of the transcode pass.
    #[arg(long, default_value = "auto")]
    blob_ordering: String,

    /// Target pack object size in MiB (default 64). Tile blobs are cut into
    /// packs of at most this size (a single blob larger than the target gets
    /// its own pack rather than being split). Smaller → finer cache
    /// granularity + more parallel range reads but more objects; larger →
    /// fewer, coarser objects. Stay well under the CDN per-object cap (512 MB).
    #[arg(long, default_value = "64")]
    pack_size: u64,

    /// Temporal bucket size for chunking tiles (e.g., "1h", "6h", "1d", "30m")
    /// Features are grouped into fixed temporal intervals, creating predictable tile boundaries
    /// that align with natural time units for efficient animation and prefetching.
    #[arg(long, default_value = "1h")]
    temporal_bucket: String,

    /// Optional temporal LOD pyramid (e.g. "1d,30d"). Each entry is a coarser
    /// bucket size. The archive will carry one extra aggregate-tile tier per
    /// level, in addition to the base `--temporal-bucket` tiles, so a client
    /// animating decades of data at "year scale" can pick the coarser tier.
    ///
    /// Each entry MUST be a strict multiple of `--temporal-bucket` and the
    /// list MUST be sorted by ascending duration. Each level applies up to
    /// (and including) `max-zoom-level`, configurable as `1d@8,30d@4`
    /// (default: every level applies at every zoom).
    #[arg(long)]
    temporal_lod: Option<String>,

    /// Number of parallel workers
    #[arg(short, long, default_value = "4")]
    workers: usize,

    /// Archive name (metadata)
    #[arg(long)]
    name: Option<String>,

    /// Archive description (metadata)
    #[arg(long)]
    description: Option<String>,

    /// Attribution text (metadata)
    #[arg(long)]
    attribution: Option<String>,

    /// Layer name
    #[arg(long, default_value = "default")]
    layer: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Output metadata JSON file (for generating datasets config)
    #[arg(long)]
    metadata_output: Option<PathBuf>,

    // --- Trajectory Clipping Options ---
    /// Disable trajectory clipping (store entire trajectories in centroid tile)
    /// By default, LineStrings with duration are clipped at tile boundaries
    #[arg(long)]
    no_clip: bool,

    /// Minimum vertices required to clip a trajectory (skip short paths)
    #[arg(long, default_value = "2")]
    clip_min_vertices: usize,

    // --- Memory/Performance Options ---
    /// Enable streaming mode (write tiles as each zoom level completes)
    /// Reduces peak memory usage for large datasets at cost of some parallelism
    #[arg(long)]
    streaming: bool,

    /// Enable the new Arrow-native streaming pipeline. Reads Parquet record
    /// batches lazily, partitions into per-tile accumulators, and emits
    /// tiles directly to the archive — peak RSS bounded by one batch plus
    /// the active tile-spill budget. Required for >10 GB inputs.
    #[arg(long)]
    streaming_arrow: bool,

    /// Enable line simplification for lower zoom levels (reduces memory and improves performance)
    #[arg(long)]
    simplify: bool,

    /// Maximum zoom level to apply simplification (higher zooms keep full detail)
    #[arg(long, default_value = "14")]
    simplify_max_zoom: u8,

    /// Use time-aware TD-TR (Synchronized Euclidean Distance) simplification
    /// instead of plain spatial Visvalingam. Preserves per-vertex timing so
    /// zoomed-out trajectory playback keeps moving objects in the right place at
    /// the right time. Takes effect together with `--simplify`.
    #[arg(long)]
    time_aware_simplify: bool,

    /// Adaptive temporal chunking: instead of fixed `--temporal-bucket` windows,
    /// partition each tile's features into windows of ~N features (dense periods
    /// get fine time windows, sparse periods coarse ones) — the density-adaptive
    /// analogue of tippecanoe's `--maximum-tile-features`. In-memory
    /// (non-streaming) builds only.
    #[arg(long)]
    adaptive_temporal: Option<u32>,

    /// Fail the build on any row with a null or unparseable timestamp,
    /// rather than coercing it to Unix epoch 0 with a warning.
    /// Negative (pre-1970) timestamps always fail the build — the temporal
    /// index stores unsigned ms-since-epoch and cannot represent them.
    #[arg(long)]
    strict_times: bool,

    /// Fail the build on any row with a null or unparseable geometry,
    /// rather than skipping the row with a warning. (Skipped rows are never
    /// tiled — there is no position to place them at.)
    #[arg(long)]
    strict_geometry: bool,

    /// Auto-tune zoom range, temporal bucket size, and compression by
    /// running stt-optimize's analyzer over the input before building.
    /// Any flag the user passes explicitly still wins; only unset/default
    /// values are filled in from the recommendation.
    #[arg(long)]
    auto: bool,

    // --- Summary-tier options (server-aggregated low-zoom tier) ---
    /// Emit a pre-aggregated summary tier alongside the raw tier.
    /// Aggregation scheme: `h3` (Uber H3 hexes) or `quadbin` (CARTO quadbin).
    /// Currently only `h3` is implemented.
    ///
    /// Summary tiles live in the same archive directory as raw tiles but use
    /// a distinct layer name (`summary` by default). Readers dispatch
    /// between the two tiers automatically.
    #[arg(long, value_enum)]
    summary_tier: Option<SummaryTierScheme>,

    /// Lowest zoom at which summary tiles are emitted. Defaults to the
    /// archive's `--min-zoom`.
    #[arg(long)]
    summary_min_zoom: Option<u8>,

    /// Highest zoom at which summary tiles are emitted. Defaults to
    /// `--min-zoom + 4` — past this point raw tiles take over.
    #[arg(long)]
    summary_max_zoom: Option<u8>,

    /// Aggregated columns for the summary tier. Comma-separated list of
    /// `name:agg` entries, e.g. `magnitude:mean,magnitude:max,depth:sum`.
    /// The implicit `count` aggregate is always emitted; pass `count`
    /// explicitly only if you want it positioned among the other columns
    /// (the implicit one is otherwise emitted first).
    #[arg(long, default_value = "")]
    summary_columns: String,

    /// Layer name carried in summary tile frames. Defaults to `summary`.
    /// You only need to change this if your raw layer name happens to be
    /// `summary` already.
    #[arg(long, default_value = "summary")]
    summary_layer: String,

    /// Number of fine-grained sub-buckets PER tile-temporal-bucket. Default
    /// 1 = legacy single-count behaviour. When > 1, each summary cell row
    /// carries N extra `bucket_<i>` numeric columns — counts of features
    /// observed within each `(bucket_ms / N)`-wide sub-window. The
    /// renderer can animate through these via a `currentSubBucket`
    /// uniform with zero data re-upload between frames.
    ///
    /// Recommended: 12-30 for hour-bucketed archives (one column per 2-5
    /// minutes). Tile size grows ~`N * 6 bytes per cell`; cap at 32 to
    /// keep deep-zoom tiles tractable.
    #[arg(long, default_value = "1")]
    summary_sub_buckets: u32,

    /// Pre-tessellate polygon features at build time and store the resulting
    /// earcut triangle indices in a sidecar column. Lets renderers skip CPU
    /// tessellation on tile arrival — wins scale with polygon vertex count.
    /// Adds ~4 bytes per triangle index to the tile payload.
    #[arg(long)]
    pre_tessellate: bool,

    /// Drop tiles whose feature count is below this threshold. Default 1
    /// (write every non-empty tile). For globally sparse point datasets,
    /// raising this to 2-5 skips the long tail of single-feature deep-zoom
    /// tiles where the Arrow IPC + compression overhead dwarfs the payload.
    /// The TS reader's parent-fallback (`refinementStrategy: 'best-available'`)
    /// surfaces the skipped features from their parent tile.
    #[arg(long, default_value = "1")]
    min_features_per_tile: u32,

    /// Feature property used to drive the HeatmapLayer's per-splat weight.
    /// When set, the build computes the property's [min, 95th percentile]
    /// across all features and bakes it into the archive metadata as the
    /// default heatmap-domain. The renderer reads it on archive open and
    /// pins `colorDomain` — no runtime GPU readback, ramp stays stable
    /// across tile churn.
    ///
    /// 95p (not absolute max) protects against single-outlier dimming —
    /// one M9.5 quake shouldn't make the rest of the dataset invisible.
    #[arg(long)]
    heatmap_weight: Option<String>,

    /// Categorical property whose values become per-class heatmap entries.
    /// When combined with --heatmap-weight, the build emits one domain
    /// entry per unique categorical value (up to 8). The renderer's
    /// `channels` spec is keyed on these ids so a stacked heatmap can
    /// pull per-class domains by id.
    ///
    /// Without --heatmap-weight, the per-class entries report constant
    /// [0, 1] (sufficient for the un-weighted gaussian-peak case).
    #[arg(long)]
    heatmap_class: Option<String>,

    /// Ceiling (ms) on the per-vertex time quantization step. Vertex
    /// timestamps ride a compact u16-delta encoding whose step is derived
    /// from each tile layer's temporal span; a layer that would need a step
    /// coarser than this ceiling is stored as exact i64 timestamps instead
    /// (larger payload, zero precision loss). Default 1000 ms — below
    /// anything playback can show. Raise it only to trade precision for
    /// payload size on very wide temporal-LOD buckets.
    #[arg(long, default_value_t = stt_core::arrow_tile::DEFAULT_VERTEX_TIME_MAX_STEP_MS, value_name = "MS")]
    vertex_time_precision: u32,
}

/// Aggregation scheme for `--summary-tier`. `quadbin` is reserved (declared
/// here so the CLI vocabulary is stable) but not implemented yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum SummaryTierScheme {
    H3,
    Quadbin,
}

fn main() -> Result<()> {
    let matches = Args::command().get_matches();
    let mut args = Args::from_arg_matches(&matches)
        .context("failed to parse stt-build arguments")?;

    // Initialize logging
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(if args.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set tracing subscriber")?;

    // Resolve the output into a packed-dataset directory. A path ending in
    // `.stt` has the extension stripped (so `-o foo.stt` -> `foo/`); anything
    // else is used as the directory as-is.
    let out_dir = packed_output_dir(&args.output);

    info!("Starting stt-build");
    info!("Input: {}", args.input.display());
    info!("Output (packed dataset dir): {}", out_dir.display());

    // Validate input file is GeoParquet
    let extension = args.input.extension().and_then(|s| s.to_str()).unwrap_or("");
    if !matches!(extension.to_lowercase().as_str(), "parquet" | "geoparquet") {
        anyhow::bail!(
            "Input must be a GeoParquet file (.parquet or .geoparquet), got: .{}",
            extension
        );
    }

    if args.auto {
        apply_auto_recommendations(&matches, &mut args)?;
    }

    // Per-vertex time precision is a process-wide encoder setting (the
    // encode sites sit behind generic tile-writer traits with no per-call
    // options channel). Set once, before any tile is encoded.
    stt_core::arrow_tile::set_vertex_time_max_step_ms(args.vertex_time_precision);

    // Parse compression
    let compression = parse_compression(&args.compression)?;

    // Parse the pack ordering (the packed format always buffers + reorders
    // before cutting packs). `eager` is accepted for backward-compat and maps
    // to `auto`.
    let pack_ordering: stt_core::BlobOrdering =
        if args.blob_ordering.trim().eq_ignore_ascii_case("eager") {
            stt_core::BlobOrdering::Auto
        } else {
            args.blob_ordering
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?
        };

    // Pack target size (MiB -> bytes). Never 0.
    let pack_target_bytes = args
        .pack_size
        .saturating_mul(1024 * 1024)
        .max(1);

    let time_strictness = if args.strict_times {
        input::InputStrictness::Strict
    } else {
        input::InputStrictness::Warn
    };
    let geometry_strictness = if args.strict_geometry {
        input::InputStrictness::Strict
    } else {
        input::InputStrictness::Warn
    };

    // --streaming-arrow: the new Arrow-native streaming pipeline. Reads
    // record batches lazily, builds per-tile accumulators, and writes
    // tiles directly without ever holding the full feature set in RAM.
    // The legacy path below keeps the small-dataset behaviour identical
    // to v2.
    if args.streaming_arrow {
        // The streaming pipeline finalizes the archive before the in-memory
        // path's summary-tier / heatmap-domain / raster / metadata-output
        // passes run, so silently honouring those flags here would drop them.
        // Refuse the combination with a clear error rather than producing an
        // archive that's quietly missing the requested features. (Wiring these
        // passes into the streaming finalize is tracked as a follow-up; until
        // then use the in-memory pipeline for these features.)
        let mut unsupported: Vec<&str> = Vec::new();
        if args.summary_tier.is_some() {
            unsupported.push("--summary-tier");
        }
        if args.heatmap_weight.is_some() {
            unsupported.push("--heatmap-weight");
        }
        if args.heatmap_class.is_some() {
            unsupported.push("--heatmap-class");
        }
        if args.metadata_output.is_some() {
            unsupported.push("--metadata-output");
        }
        if !unsupported.is_empty() {
            anyhow::bail!(
                "--streaming-arrow does not yet support {} (the streaming pipeline \
                 finalizes before those passes run). Re-run without --streaming-arrow \
                 to use {}, or drop the flag(s) to stream.",
                unsupported.join(", "),
                unsupported.join(", "),
            );
        }
        info!("Using streaming-Arrow pipeline (bounded RAM)...");
        let temporal_bucket_ms = parse_duration(&args.temporal_bucket)?;
        let temporal_lod = match args.temporal_lod.as_deref() {
            Some(s) => parse_temporal_lod(s, args.max_zoom)?,
            None => Vec::new(),
        };
        if !temporal_lod.is_empty() {
            warn!(
                "--temporal-lod ignored in --streaming-arrow mode: LOD aggregation \
                 requires the in-memory pipeline (will be lifted to streaming in a \
                 follow-up)."
            );
        }
        let tile_config = tiler::TileConfig {
            min_zoom: args.min_zoom,
            max_zoom: args.max_zoom,
            layer_name: args.layer.clone(),
            temporal_bucket_ms,
            clip_trajectories: !args.no_clip,
            clip_min_vertices: args.clip_min_vertices,
            simplify: args.simplify,
            simplify_max_zoom: args.simplify_max_zoom,
            pre_tessellate: args.pre_tessellate,
            // Temporal LOD only wired in the non-streaming branch below.
            temporal_lod: Vec::new(),
            min_features_per_tile: args.min_features_per_tile,
            time_aware_simplify: args.time_aware_simplify,
            // Adaptive temporal chunking is an in-memory-only feature; the
            // streaming path keeps fixed buckets, so this is ignored here.
            adaptive_target_features: None,
        };
        // --streaming-arrow stays bounded-RAM: it streams tiles into a TEMP
        // single-file archive (the only writer that doesn't buffer all
        // payloads), then transcodes that temp file into the packed dir. The
        // pack ordering is applied during the transcode pass.
        info!(
            "Streaming to a temp single-file archive, then transcoding to packs \
             (ordering {pack_ordering}, pack target {} MiB)",
            args.pack_size
        );
        let temp_archive = tempfile::Builder::new()
            .prefix("stt-build-streaming-")
            .suffix(".stt")
            .tempfile()
            .context("failed to create temp archive for --streaming-arrow")?;
        let temp_archive_path = temp_archive.path().to_path_buf();
        let mut writer = stt_core::Archive::create(&temp_archive_path, compression)?;
        let mut bounds_lon = (f64::MAX, f64::MIN);
        let mut bounds_lat = (f64::MAX, f64::MIN);
        let mut time_range = (u64::MAX, 0u64);
        let mut feature_count: u64 = 0;

        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<input::ParsedFeature>>>(2);
        // Producer: stream parquet -> channel.
        let input_path = args.input.clone();
        let time_field = args.time_field.clone();
        let end_time_field = args.end_time_field.clone();
        let time_format = args.time_format;
        let handle = std::thread::spawn(move || {
            let tx_clone = tx.clone();
            let res = input::stream_features(
                &input_path,
                &time_field,
                end_time_field.as_deref(),
                time_format,
                time_strictness,
                geometry_strictness,
                |batch| {
                    tx_clone
                        .send(Ok(batch))
                        .map_err(|e| anyhow::anyhow!("channel closed: {e}"))?;
                    Ok(())
                },
            );
            if let Err(e) = res {
                let _ = tx.send(Err(e));
            }
        });

        // Consumer: drive the streaming tiler and update bounds inline.
        let iter = std::iter::from_fn(|| rx.recv().ok()).map(|res| {
            res.map(|batch| {
                feature_count += batch.len() as u64;
                for f in &batch {
                    bounds_lon.0 = bounds_lon.0.min(f.lon);
                    bounds_lon.1 = bounds_lon.1.max(f.lon);
                    bounds_lat.0 = bounds_lat.0.min(f.lat);
                    bounds_lat.1 = bounds_lat.1.max(f.lat);
                    time_range.0 = time_range.0.min(f.timestamp);
                    time_range.1 = time_range
                        .1
                        .max(f.end_timestamp.unwrap_or(f.timestamp));
                }
                batch
            })
        });

        let stats = tiler::build_streaming_from_batches(
            iter,
            &tile_config,
            &mut writer,
            args.workers,
            // 64 MB per-tile spill budget by default. Bigger = fewer flushes
            // (better compression ratio) but more RAM in flight.
            64 * 1024 * 1024,
        )?;
        handle.join().ok();

        info!(
            "streaming: {} tiles ({} clipped, {} originals) from {} features",
            stats.total_tiles, stats.clipped_segments, stats.original_features, feature_count
        );

        let bounds = if feature_count == 0 {
            stt_core::types::BoundingBox::new(-180.0, -90.0, 180.0, 90.0)
        } else {
            stt_core::types::BoundingBox::new(
                bounds_lon.0,
                bounds_lat.0,
                bounds_lon.1,
                bounds_lat.1,
            )
        };
        let trange = if feature_count == 0 {
            stt_core::types::TimeRange::new(0, 0)
        } else {
            aligned_time_range(
                stt_core::types::TimeRange::new(time_range.0, time_range.1),
                temporal_bucket_ms,
                &[],
                false,
            )
        };
        let metadata = stt_core::metadata::Metadata::new(args.name.unwrap_or_else(|| {
            args.input
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string()
        }))
        .with_description(args.description.unwrap_or_default())
        .with_attribution(args.attribution.unwrap_or_default())
        .with_bounds(bounds)
        .with_time_range(trange)
        .with_zoom_levels(args.min_zoom, args.max_zoom)
        .with_temporal_bucket_ms(temporal_bucket_ms);
        writer.finalize(&metadata)?;

        // Transcode the bounded-RAM temp archive into the packed dataset dir,
        // then drop the temp file. Shared with the pack-transcode example.
        info!("Transcoding temp archive -> packed dir {}", out_dir.display());
        let manifest =
            stt_core::transcode_archive_to_packs(&temp_archive_path, &out_dir, pack_ordering, pack_target_bytes)?;
        drop(temp_archive); // delete the temp .stt
        let total_pack_bytes: u64 = manifest.packs.iter().map(|p| p.length).sum();
        info!(
            "Packed dataset written to {} ({} tiles, {} features, {} packs, {} pack bytes)",
            out_dir.display(),
            stats.total_tiles,
            feature_count,
            manifest.packs.len(),
            total_pack_bytes,
        );
        return Ok(());
    }

    // Step 1: Load all features into memory
    info!("Loading input data...");
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message("Reading input file...");

    let features = input::load_features(
        &args.input,
        &args.time_field,
        args.end_time_field.as_deref(),
        args.time_format,
        time_strictness,
        geometry_strictness,
    )?;

    pb.finish_with_message(format!("Loaded {} features", features.len()));
    info!("Loaded {} features", features.len());

    if features.is_empty() {
        warn!("No features found in input file");
        return Ok(());
    }

    // Step 2: Analyze data bounds
    info!("Analyzing data bounds...");
    let (bounds, time_range) = input::calculate_bounds(&features)?;
    info!("Spatial bounds: {:?}", bounds);
    info!("Time range: {} to {}", time_range.start, time_range.end);

    // Step 3: Generate tiles
    info!("Generating tiles...");

    // Parse temporal bucket size
    let temporal_bucket_ms = parse_duration(&args.temporal_bucket)?;
    info!("Temporal bucket size: {} ms ({}))", temporal_bucket_ms, args.temporal_bucket);

    if !args.no_clip {
        info!("Trajectory clipping enabled (min {} vertices)", args.clip_min_vertices);
    } else {
        info!("Trajectory clipping disabled (--no-clip)");
    }

    if args.simplify {
        info!(
            "Line simplification enabled (max zoom {})",
            args.simplify_max_zoom
        );
    }

    let temporal_lod = match args.temporal_lod.as_deref() {
        Some(s) => parse_temporal_lod(s, args.max_zoom)?,
        None => Vec::new(),
    };
    if !temporal_lod.is_empty() {
        info!(
            "Temporal LOD: {} levels — {}",
            temporal_lod.len(),
            temporal_lod
                .iter()
                .map(|l| format!("{}ms@z<={}", l.bucket_ms, l.max_zoom_level))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let tile_config = tiler::TileConfig {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        layer_name: args.layer.clone(),
        temporal_bucket_ms,
        clip_trajectories: !args.no_clip,
        clip_min_vertices: args.clip_min_vertices,
        simplify: args.simplify,
        simplify_max_zoom: args.simplify_max_zoom,
        pre_tessellate: args.pre_tessellate,
        temporal_lod: temporal_lod.clone(),
        min_features_per_tile: args.min_features_per_tile,
        time_aware_simplify: args.time_aware_simplify,
        adaptive_target_features: args.adaptive_temporal,
    };

    if args.pre_tessellate {
        info!("Pre-tessellation enabled (triangle indices written alongside polygon geometry)");
    }

    info!(
        "Pack ordering: {pack_ordering} (buffered — space-time blob layout + \
         byte-identical dedup, then cut into packs of ≤{} MiB; holds payloads \
         in RAM until finalize)",
        args.pack_size
    );
    let mut writer = stt_core::PackWriter::create(&out_dir, pack_ordering, pack_target_bytes)?;

    let tile_count = if !temporal_lod.is_empty() {
        // --temporal-lod path: emit base tiles + per-level aggregate tiles.
        // Streams through the LodTileWriter so each directory entry carries
        // its temporal_bucket_ms. The streaming pipeline doesn't (yet) do
        // LOD aggregation — that's a follow-up — so this path goes through
        // the in-memory builder regardless of --streaming.
        if args.streaming {
            warn!("--streaming ignored when --temporal-lod is set (in-memory pipeline used)");
        }
        use tiler::LodTileWriter;
        let tiles = tiler::generate_tiles_with_lod(&features, &tile_config, args.workers)?;
        info!(
            "Generated {} tiles (base + LOD aggregate tiers)",
            tiles.len()
        );
        let pb = ProgressBar::new(tiles.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );
        let min_features = args.min_features_per_tile.max(1);
        let mut written = 0usize;
        for tagged in &tiles {
            if tagged.tile.feature_count() < min_features {
                pb.inc(1);
                continue;
            }
            writer.write_lod_tile(&tagged.tile, tagged.temporal_bucket_ms)?;
            written += 1;
            pb.inc(1);
        }
        pb.finish_with_message("Tiles written");
        if written != tiles.len() {
            info!(
                "Skipped {} tiles below --min-features-per-tile={} ({} written)",
                tiles.len() - written,
                min_features,
                written
            );
        }
        written
    } else if args.streaming {
        // Streaming mode: write tiles as each zoom level completes
        info!("Using streaming mode (lower memory usage)...");
        let stats = tiler::generate_tiles_streaming(
            &features,
            &tile_config,
            &mut writer,
            args.workers,
        )?;
        info!(
            "Generated {} tiles ({} clipped segments, {} original features)",
            stats.total_tiles, stats.clipped_segments, stats.original_features
        );
        stats.total_tiles
    } else {
        // Standard mode: generate all tiles then write
        let tiles = tiler::generate_tiles(&features, &tile_config, args.workers)?;
        info!("Generated {} tiles", tiles.len());

        // Write archive
        info!("Writing archive...");
        let pb = ProgressBar::new(tiles.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );

        let min_features = args.min_features_per_tile.max(1);
        let mut written = 0usize;
        for tile in &tiles {
            if tile.feature_count() < min_features {
                pb.inc(1);
                continue;
            }
            writer.write_tile(tile)?;
            written += 1;
            pb.inc(1);
        }

        pb.finish_with_message("Tiles written");
        if written != tiles.len() {
            info!(
                "Skipped {} tiles below --min-features-per-tile={} ({} written)",
                tiles.len() - written,
                min_features,
                written
            );
        }
        written
    };

    // Step 4b: Optional summary tier (server-aggregated cells).
    //
    // The summary tier is written into the SAME archive as the raw tiles.
    // Raw tiles at low zoom levels still exist but the TS reader dispatches
    // to summary tiles when the metadata declares the tier covers that zoom.
    // This is intentional — keeping the raw tier untouched means a v3-aware
    // reader that doesn't understand `summary_tier` falls back cleanly.
    let summary_tier_descriptor = if let Some(scheme) = args.summary_tier {
        let scheme = match scheme {
            SummaryTierScheme::H3 => stt_core::metadata::SummaryScheme::H3,
            SummaryTierScheme::Quadbin => {
                anyhow::bail!("--summary-tier quadbin is not implemented yet (h3 only)");
            }
        };
        let sm_min = args.summary_min_zoom.unwrap_or(args.min_zoom);
        let sm_max = args
            .summary_max_zoom
            .unwrap_or_else(|| (args.min_zoom + 4).min(args.max_zoom));
        if sm_min > sm_max {
            anyhow::bail!(
                "--summary-min-zoom ({sm_min}) > --summary-max-zoom ({sm_max})"
            );
        }
        let mut cols = summary::parse_summary_columns(&args.summary_columns)?;
        // Guarantee a count aggregate is recorded in the metadata even if
        // the user did not list it. The build step always emits one in
        // the `count` column; recording it in the descriptor lets the
        // reader know it can be used as a heatmap weight.
        if !cols
            .iter()
            .any(|c| matches!(c.agg, stt_core::metadata::SummaryAggregation::Count))
        {
            cols.insert(
                0,
                stt_core::metadata::SummaryColumn {
                    name: "_count".to_string(),
                    agg: stt_core::metadata::SummaryAggregation::Count,
                },
            );
        }

        let summary_config = summary::SummaryConfig {
            scheme,
            min_zoom: sm_min,
            max_zoom: sm_max,
            temporal_bucket_ms,
            columns: cols,
            layer_name: args.summary_layer.clone(),
            sub_buckets: args.summary_sub_buckets.max(1),
        };

        info!(
            "Building summary tier ({scheme:?}, zooms {sm_min}..={sm_max}, columns {})",
            summary_config.columns.len()
        );
        let n_summary = summary::build_summary_tier(&features, &summary_config, &mut writer)?;
        info!("Summary tier: {n_summary} aggregate tiles emitted");
        Some(summary_config.to_tier())
    } else {
        None
    };

    // Step 5: Build metadata (combine summary-tier + temporal-LOD builders).
    let mut metadata = stt_core::metadata::Metadata::new(args.name.unwrap_or_else(|| {
        args.input
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string()
    }))
    .with_description(args.description.unwrap_or_default())
    .with_attribution(args.attribution.unwrap_or_default())
    .with_bounds(bounds)
    .with_time_range(aligned_time_range(
        time_range,
        temporal_bucket_ms,
        &temporal_lod,
        args.adaptive_temporal.is_some(),
    ))
    .with_zoom_levels(args.min_zoom, args.max_zoom)
    .with_temporal_bucket_ms(temporal_bucket_ms);
    if let Some(tier) = summary_tier_descriptor {
        metadata = metadata.with_summary_tier(tier);
    }
    if !temporal_lod.is_empty() {
        metadata = metadata
            .with_temporal_lod(temporal_lod.clone())
            .with_context(|| "temporal LOD validation failed")?;
    }
    if let Some(domain) = compute_heatmap_domain(
        &features,
        args.heatmap_weight.as_deref(),
        args.heatmap_class.as_deref(),
    ) {
        info!(
            "Heatmap domain: {} class entries — first: {} → [{:.3}, {:.3}]",
            domain.classes.len(),
            domain.classes.first().map(|c| c.id.as_str()).unwrap_or("?"),
            domain.classes.first().map(|c| c.min).unwrap_or(0.0),
            domain.classes.first().map(|c| c.max).unwrap_or(1.0),
        );
        metadata = metadata.with_heatmap_domain(domain);
    }

    let manifest = writer.finalize(&metadata)?;

    // Step 6: Write metadata JSON if requested
    if let Some(metadata_path) = args.metadata_output {
        info!("Writing metadata JSON to {}...", metadata_path.display());
        // The packed dataset is addressed by its manifest, so the dataset
        // "filename" is now `<dir>/manifest.json`.
        let manifest_rel = out_dir.join("manifest.json");
        let metadata_json = serde_json::json!({
            "filename": format!(
                "{}/manifest.json",
                out_dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
            ),
            "path": manifest_rel.to_string_lossy(),
            "name": metadata.name,
            "description": metadata.description,
            "attribution": metadata.attribution,
            "timeRange": {
                "start": time_range.start,
                "end": time_range.end,
            },
            "zoomLevels": {
                "min": args.min_zoom,
                "max": args.max_zoom,
            },
            "bounds": {
                "minLon": bounds.min_lon,
                "minLat": bounds.min_lat,
                "maxLon": bounds.max_lon,
                "maxLat": bounds.max_lat,
            },
            "tileCount": tile_count,
            "compression": args.compression,
            "temporalBucketMs": temporal_bucket_ms,
        });
        std::fs::write(metadata_path, serde_json::to_string_pretty(&metadata_json)?)?;
    }

    let total_pack_bytes: u64 = manifest.packs.iter().map(|p| p.length).sum();
    info!(
        "Packed dataset written successfully to {} ({} packs, {} pack bytes)",
        out_dir.display(),
        manifest.packs.len(),
        total_pack_bytes,
    );
    info!("Total tiles: {}", tile_count);
    info!("Total features: {}", features.len());

    Ok(())
}

/// Resolve the `-o/--output` value into a packed-dataset directory.
///
/// The packed format is a directory tree (`manifest.json` + `index/` +
/// `packs/`). For convenience a path ending in `.stt` (the old single-file
/// extension) has that extension stripped, so `-o foo.stt` -> `foo/`. Any other
/// path is used verbatim as the dataset directory.
fn packed_output_dir(output: &std::path::Path) -> PathBuf {
    let is_stt = output
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.eq_ignore_ascii_case("stt"))
        .unwrap_or(false);
    if is_stt {
        output.with_extension("")
    } else {
        output.to_path_buf()
    }
}

/// Compute the build-time HeatmapLayer intensity domain for the archive.
///
/// Returns `None` when there's nothing useful to bake (no `--heatmap-weight`
/// AND no `--heatmap-class`). The default un-weighted gaussian-peak case is
/// just `[0, 1]` per channel — the renderer hard-codes that fallback so we
/// don't bother emitting it.
///
/// When `weight_prop` is set, the property's min and 95th-percentile across
/// all features form the class domain. 95p (not absolute max) protects the
/// ramp from single-outlier dimming.
///
/// When `class_prop` is set, the build emits ONE class entry per unique
/// categorical value (capped at 8 to bound metadata size). Each entry's
/// min/max is computed over features that carry that class value.
fn compute_heatmap_domain(
    features: &[input::ParsedFeature],
    weight_prop: Option<&str>,
    class_prop: Option<&str>,
) -> Option<stt_core::metadata::HeatmapDomain> {
    if weight_prop.is_none() && class_prop.is_none() {
        return None;
    }
    use stt_core::metadata::{HeatmapClassDomain, HeatmapDomain};

    fn extract_f64(f: &input::ParsedFeature, name: &str) -> Option<f64> {
        f.shared_properties
            .as_deref()?
            .get(name)
            .and_then(|v| v.as_f64())
    }
    fn extract_str(f: &input::ParsedFeature, name: &str) -> Option<String> {
        f.shared_properties
            .as_deref()?
            .get(name)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
    fn p95(mut values: Vec<f64>) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 1.0);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = values[0];
        let idx = ((values.len() as f64 * 0.95).floor() as usize)
            .min(values.len() - 1);
        let max = values[idx];
        (min, max)
    }

    let classes = if let Some(cls) = class_prop {
        // Per-class domain. Group features by the categorical value.
        let mut groups: std::collections::BTreeMap<String, Vec<f64>> =
            std::collections::BTreeMap::new();
        for f in features {
            let Some(class_val) = extract_str(f, cls) else {
                continue;
            };
            // Cap on group count to keep metadata bounded.
            if groups.len() >= 8 && !groups.contains_key(&class_val) {
                continue;
            }
            let bucket = groups.entry(class_val).or_default();
            if let Some(w) = weight_prop.and_then(|p| extract_f64(f, p)) {
                bucket.push(w);
            } else {
                // Un-weighted: count of "present" so min/p95 trivially map to
                // [0, 1] post-aggregation. Push the gaussian-peak value.
                bucket.push(1.0);
            }
        }
        groups
            .into_iter()
            .map(|(id, values)| {
                let (min, max) = p95(values);
                HeatmapClassDomain {
                    id,
                    min,
                    max,
                    property: weight_prop.map(str::to_string),
                }
            })
            .collect::<Vec<_>>()
    } else {
        // No class split — one "default" entry.
        let values: Vec<f64> = match weight_prop {
            Some(p) => features.iter().filter_map(|f| extract_f64(f, p)).collect(),
            None => vec![1.0],
        };
        let (min, max) = p95(values);
        vec![HeatmapClassDomain {
            id: "default".to_string(),
            min,
            max,
            property: weight_prop.map(str::to_string),
        }]
    };

    if classes.is_empty() {
        return None;
    }
    Some(HeatmapDomain { classes })
}

/// Run stt-optimize over the input and fold its recommendations into `args`
/// for any flag the user did NOT pass explicitly.
fn apply_auto_recommendations(matches: &ArgMatches, args: &mut Args) -> Result<()> {
    info!("--auto: analyzing input for build recommendations...");
    let source = stt_optimize::DataSource::GeoParquet {
        path: args.input.clone(),
        time_field: args.time_field.clone(),
        time_format: args.time_format.as_str().to_string(),
    };
    let rec = stt_optimize::recommend_for(&source)
        .context("stt-optimize analyzer failed")?;

    let user_set = |name: &str| {
        matches!(matches.value_source(name), Some(ValueSource::CommandLine))
    };

    if !user_set("min_zoom") {
        info!("  min-zoom: {} (auto)", rec.min_zoom);
        args.min_zoom = rec.min_zoom;
    }
    if !user_set("max_zoom") {
        info!("  max-zoom: {} (auto)", rec.max_zoom);
        args.max_zoom = rec.max_zoom;
    }
    // rec.compression is NOT folded in: the packed format is zstd-only, so
    // an analyzer recommendation of gzip/none would just fail validation.
    if !user_set("temporal_bucket") && rec.temporal_bucket_ms > 0 {
        let human = rec.temporal_bucket_human.clone();
        info!("  temporal-bucket: {} (auto)", human);
        args.temporal_bucket = human;
    }

    info!(
        "  confidence: {}% — {} reasons",
        rec.confidence,
        rec.explanations.len()
    );
    for line in &rec.explanations {
        info!("    - {}", line);
    }
    Ok(())
}

/// Bucket-align the archive's start time down to the coarsest temporal bucket,
/// so the metadata range actually bounds the (bucket-aligned) tile starts. In
/// fixed-bucket mode a tile's `time_start` is `floor(t / bucket) * bucket`,
/// which can sit up to one bucket before the first raw event; without this the
/// validator (correctly) flags every first-bucket tile as out-of-range. In
/// adaptive mode tiles start at real event times, so no alignment is applied.
fn aligned_time_range(
    tr: stt_core::types::TimeRange,
    temporal_bucket_ms: u64,
    temporal_lod: &[stt_core::metadata::TemporalLodLevel],
    adaptive: bool,
) -> stt_core::types::TimeRange {
    if adaptive || temporal_bucket_ms == 0 {
        return tr;
    }
    let coarsest = temporal_lod
        .iter()
        .map(|l| l.bucket_ms)
        .max()
        .unwrap_or(0)
        .max(temporal_bucket_ms);
    let start = (tr.start / coarsest) * coarsest;
    stt_core::types::TimeRange::new(start, tr.end)
}

/// The packed format compresses per-blob with zstd (no shared dict, so the
/// TS reader can decode it); the vestigial `gzip`/`none` choices were
/// removed — they were already ignored-with-warning.
fn parse_compression(s: &str) -> Result<stt_core::types::Compression> {
    match s.to_lowercase().as_str() {
        "zstd" | "zstandard" => Ok(stt_core::types::Compression::Zstd),
        "gzip" | "none" => anyhow::bail!(
            "--compression {} has been removed: the packed format always \
             compresses tile payloads per-blob with zstd. Drop the flag \
             (zstd is the default).",
            s
        ),
        _ => anyhow::bail!(
            "Invalid compression method: {}. Only 'zstd' is supported",
            s
        ),
    }
}

/// Parse a `--temporal-lod` spec like `"1d,30d"` or `"1d@8,30d@4"`. Each
/// entry is `<duration>` (applies at every zoom) or `<duration>@<zoom>`
/// (applies at zoom <= the given level). Entries are returned in input
/// order so the build can re-validate sorting against the base bucket.
fn parse_temporal_lod(
    s: &str,
    fallback_max_zoom: u8,
) -> Result<Vec<stt_core::metadata::TemporalLodLevel>> {
    let mut levels = Vec::new();
    for piece in s.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let (dur, zoom) = match piece.split_once('@') {
            Some((d, z)) => {
                let z: u8 = z
                    .trim()
                    .parse()
                    .with_context(|| format!("invalid zoom in temporal-lod entry '{piece}'"))?;
                (d.trim(), z)
            }
            None => (piece, fallback_max_zoom),
        };
        let bucket_ms = parse_duration(dur)
            .with_context(|| format!("invalid duration in temporal-lod entry '{piece}'"))?;
        levels.push(stt_core::metadata::TemporalLodLevel {
            bucket_ms,
            max_zoom_level: zoom,
        });
    }
    Ok(levels)
}

/// Parse a duration string like "1h", "6h", "1d", "30m" into milliseconds
fn parse_duration(s: &str) -> Result<u64> {
    let s = s.trim().to_lowercase();

    // Try to extract number and unit
    let mut num_str = String::new();
    let mut unit = String::new();

    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            num_str.push(c);
        } else {
            unit.push(c);
        }
    }

    let value: f64 = num_str
        .parse()
        .context(format!("Invalid duration value: {}", s))?;

    let multiplier: u64 = match unit.as_str() {
        "ms" | "" => 1,
        "s" | "sec" => 1000,
        "m" | "min" => 60 * 1000,
        "h" | "hr" | "hour" => 60 * 60 * 1000,
        "d" | "day" => 24 * 60 * 60 * 1000,
        _ => anyhow::bail!(
            "Invalid duration unit '{}'. Use ms, s, m, h, or d (e.g., '1h', '30m', '6h')",
            unit
        ),
    };

    Ok((value * multiplier as f64) as u64)
}
