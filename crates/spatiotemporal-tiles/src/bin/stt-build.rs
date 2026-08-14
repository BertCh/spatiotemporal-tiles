//! stt-build - CLI tool for building spatiotemporal tile archives
//!
//! This tool converts GeoParquet files into optimized STT archives for web visualization.

use stt_build::build_options::{self, parse_duration, parse_temporal_lod, EncoderSettings};
use stt_build::{input, summary, tiler};

use anyhow::{Context, Result};
use clap::{parser::ValueSource, ArgMatches, CommandFactory, FromArgMatches, Parser};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "stt-build")]
#[command(about = "Build spatiotemporal tile archives from GeoParquet data", long_about = None)]
#[command(version)]
struct Args {
    /// Input GeoParquet file path (.parquet or .geoparquet).
    ///
    /// Optional: omit it to read from a database instead via
    /// `--postgres`/`--table`/`--sql` (requires `--features postgres`) or
    /// `--duckdb`/`--table`/`--sql` (requires `--features duckdb`).
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// PostgreSQL/PostGIS connection string (libpq URI or key=value). When set,
    /// stt-build reads features from a live PostGIS query INSTEAD of `--input`.
    /// Requires a build with `--features postgres`. Env fallback:
    /// `STT_POSTGRES_URL`, then `DATABASE_URL`.
    #[arg(long)]
    postgres: Option<String>,

    /// DuckDB database file to read (or `:memory:` to scan files via `--sql`,
    /// e.g. `read_parquet(...)`). When set, stt-build reads features from a
    /// DuckDB query INSTEAD of `--input`. Requires a `--features duckdb` build.
    /// The source geometry column must be a DuckDB spatial `GEOMETRY`.
    #[arg(long)]
    duckdb: Option<String>,

    /// Source table to read (with `--postgres`/`--duckdb`; optionally
    /// schema-qualified, e.g. `public.trips`). Mutually exclusive with `--sql`.
    #[arg(long)]
    table: Option<String>,

    /// Arbitrary SQL `SELECT` to read (with `--postgres`/`--duckdb`). Mutually
    /// exclusive with `--table`. Must expose the geometry column named by
    /// `--geom-column`.
    #[arg(long)]
    sql: Option<String>,

    /// Geometry column name in the PostGIS / DuckDB source.
    #[arg(long, default_value = "geom")]
    geom_column: String,

    /// Optional SQL `WHERE` filter applied to the PostGIS / DuckDB source.
    #[arg(long = "where")]
    where_clause: Option<String>,

    /// Reproject source geometry from this SRID to EPSG:4326 (`ST_Transform`).
    /// Omit when the geometry is already lon/lat (4326).
    #[arg(long)]
    source_srid: Option<i32>,

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
    /// `measured` (DEFAULT; simulate the three canonical access patterns — scrub
    /// a viewport across time, pan one instant, play a sliding time window — over
    /// this dataset's own tiles and lay down the cheapest order), `auto` (the
    /// cheap cardinality heuristic: shallow/wide-time → spatial-major, else the
    /// 3D-Hilbert generalist; also what `measured` falls back to on inputs too
    /// small to simulate), or an explicit `spatial`, `time-major`, `hilbert3`, or
    /// `morton3` (morton3 is research-only — empirically never the measured
    /// winner). Locality means fewer packs touched per viewport (fewer client
    /// range requests). Tile payloads buffer until finalize, in RAM up to
    /// `--pack-memory-budget` and spilled to disk beyond it.
    #[arg(long, default_value = "measured")]
    blob_ordering: String,

    /// Query weighting the `measured` picker ranks orderings under:
    /// `derived` (default) reads this dataset's dominant layer kind and its
    /// distinct time-bucket count and weights the playback query accordingly
    /// (a trips/points dataset over many buckets is playback-dominant); `legacy`
    /// forces the pre-2026-08 two-query weighting (scrub + pan only, playback
    /// unpriced) — the escape hatch for reproducing an older layout. Inert
    /// unless `--blob-ordering measured`. The resolved weighting is recorded in
    /// the archive so `stt-optimize order-audit` can flag drift.
    #[arg(long, default_value = "derived")]
    ordering_workload: String,

    /// Target pack object size in MiB (default 64). Tile blobs are cut into
    /// packs of at most this size (a single blob larger than the target gets
    /// its own pack rather than being split). Smaller → finer cache
    /// granularity + more parallel range reads but more objects; larger →
    /// fewer, coarser objects. Stay well under the CDN per-object cap (512 MB).
    #[arg(long, default_value = "64")]
    pack_size: u64,

    /// In-memory budget (MiB) for tile payloads buffered by the pack writer
    /// between encode and finalize. Beyond the budget, payloads spill to a
    /// temp file inside the output directory (removed on success and failure
    /// alike) and are read back during finalize — per-tile directory metadata
    /// stays in RAM. Purely a memory-behaviour lever: output bytes are
    /// IDENTICAL at any budget. `0` = unlimited (hold every payload in RAM).
    #[arg(long, default_value = "512", value_name = "MIB")]
    pack_memory_budget: u64,

    /// Opt OUT of the paged directory and emit a single whole-load `.sttd`
    /// instead. The directory is **paged by default**: a tiny root page + leaf
    /// pages so a cold reader fetches only the leaves its viewport / time-window
    /// touches. For small datasets paging is ~free (one leaf, whole-loaded by
    /// the reader); the single shape only saves a few hundred bytes of root.
    #[arg(long, default_value = "false")]
    single_directory: bool,

    /// Entries per leaf page (default 4096 — the sim-validated 1024–4096 sweet
    /// spot). Ignored with `--single-directory`.
    #[arg(long, default_value = "4096")]
    page_entries: usize,

    /// Minimum tile-directory entries before the default adaptive writer uses
    /// paging. Smaller archives use one compressed frame, avoiding root/page
    /// overhead that cannot save a meaningful fetch. Set to 1 to force paging.
    #[arg(long, default_value = "8192")]
    paged_directory_min_entries: usize,

    /// zstd level for tile blobs + directory (1..=22). Default 3 is zstd's
    /// "fast" tier; a **publish** build should pass 19 — the format is
    /// write-once / serve-many, so the higher (one-time, offline) build CPU buys
    /// −10..19% on every client fetch, and decode is level-independent (free on
    /// the client). 19 ≈ 22 on STT tiles, so there's no reason to go past 19.
    #[arg(long, default_value = "3")]
    zstd_level: i32,

    /// Deploy-ready build: raise the zstd level to 19 (−10..19% on the wire,
    /// decode-free) for serve-as-is output. The directory is already paged by
    /// default, so this only bumps the level; `--zstd-level` overrides it. This
    /// is what the dataset-generation workflow (`stt-generate`) uses, so a
    /// from-source build is publish-quality without a separate re-transcode.
    /// (Coordinate quantization stays a per-dataset opt-in via `--quantize-coords`.)
    #[arg(long, default_value = "false")]
    publish: bool,

    /// Temporal bucket size for chunking tiles (e.g., "1h", "6h", "1d", "30m")
    /// Features are grouped into fixed temporal intervals, creating predictable tile boundaries
    /// that align with natural time units for efficient animation and prefetching.
    #[arg(long, default_value = "1h")]
    temporal_bucket: String,

    /// Optional temporal LOD pyramid (e.g. "1d,30d"). Each entry is a coarser
    /// bucket size. The archive will carry one extra lossless coarse-bucket
    /// tier per level, in addition to the base `--temporal-bucket` tiles, so a client
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

    /// Skip pass 1 (the dataset-global statistics scan) and let every tile
    /// decide its own encoding.
    ///
    /// The two-pass build is the default: one scan over the already-loaded
    /// features resolves each property column's dataset domain, and the
    /// encoder pins its verdicts (the numeric affine, the integer leaf width,
    /// the dictionary-vs-Utf8 choice, the category list) to that domain instead
    /// of to whichever rows happened to land in a tile. Without it the same
    /// value can decode differently in different tiles and one column can fork
    /// a schema template per tile shape.
    ///
    /// This is the escape hatch: it restores the exact per-tile behaviour, for
    /// byte-compatibility debugging against an archive built before the pins
    /// existed. It does not make the build faster in any meaningful way — pass
    /// 1 is a single in-memory iteration.
    #[arg(long)]
    single_pass: bool,

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

    /// DT-2: assign each feature ONE home zoom (additive decomposition) instead
    /// of replicating it into every zoom of its band.
    ///
    /// The reader unions across `[minZoom..z]` (`lodMode: 'additive'`). Takes an
    /// optional voxel pitch in screen pixels (default 0.4). Mutually exclusive
    /// with a user-supplied `--min-zoom-field`, which it would otherwise
    /// overwrite. Declares `metadata.partition = "home-zoom"` and the
    /// must-understand capability `additive-partition`, so a reader that does
    /// not understand the partition refuses at open rather than rendering a
    /// sparse slice as if complete.
    #[arg(long, value_name = "S_PX", num_args = 0..=1, default_missing_value = "0.4")]
    additive_lod: Option<f64>,

    /// LINE clip buffer, in pixels of a 256-px tile, resolved per zoom.
    ///
    /// A fixed degree buffer is wrong at both ends of the pyramid; a
    /// tile-relative one is a constant fraction of tile width at every zoom.
    /// Polygons are unaffected — their buffer is pinned at 0 so adjacent tiles
    /// emit bit-identical seam vertices.
    #[arg(long, default_value = "8")]
    clip_buffer_px: f64,

    /// Pin the LINE clip buffer to a fixed degree value (legacy rollback).
    /// Overrides `--clip-buffer-px`; `0.001` reproduces the pre-TB-7 behaviour.
    #[arg(long)]
    clip_buffer_degrees: Option<f64>,

    // --- Memory/Performance Options ---
    /// Enable streaming mode (write tiles as each zoom level completes)
    /// Reduces peak memory usage for large datasets at cost of some parallelism
    #[arg(long)]
    streaming: bool,

    /// Enable line simplification for lower zoom levels (reduces memory and improves performance)
    #[arg(long)]
    simplify: bool,

    /// Maximum zoom level to apply simplification (higher zooms keep full detail)
    #[arg(long, default_value = "14")]
    simplify_max_zoom: u8,

    /// Accepted no-op alias — metric simplification is now the DEFAULT (TB-8).
    ///
    /// Kept for one release so existing scripts keep working. Use
    /// `--simplify-degree-table` to get the legacy behaviour back.
    #[arg(long)]
    simplify_metric: bool,

    /// Simplify with the legacy fixed per-zoom DEGREE table instead of the
    /// latitude-corrected metric tolerance (TB-8 rollback).
    ///
    /// The degree table is up to ~2× coarser in E–W terms at 60° latitude than
    /// at the equator, so a given zoom's tolerance means different GROUND
    /// distances at different latitudes. Metric mode scales the longitude axis
    /// by cos(latitude) before simplifying, which is why it is now the default.
    /// Only meaningful with `--simplify`.
    #[arg(long)]
    simplify_degree_table: bool,

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

    /// TB-10 ROLLBACK (debug-only, one release): with `--adaptive-temporal`,
    /// use the incumbent first-fit greedy instead of the exact balanced
    /// partition, and do not snap window keys onto shared boundaries. Inert
    /// unless `--adaptive-temporal` is set.
    #[arg(long)]
    adaptive_greedy: bool,

    /// TB-10: how many shared candidate boundaries to derive from the dataset's
    /// timestamp distribution for `--adaptive-temporal` window keys. Windows
    /// snap DOWN onto this set so adjacent spatial cells land on the same fetch
    /// instants (an enumerable key set, which is what a multi-cell prefetcher
    /// needs). 0 disables snapping. Inert unless `--adaptive-temporal` is set.
    #[arg(long, default_value = "256")]
    adaptive_boundary_count: usize,

    /// Compatibility spelling for the default behavior: fail the build on any
    /// row with a null or unparseable timestamp.
    /// Negative (pre-1970) timestamps always fail the build — the temporal
    /// index stores unsigned ms-since-epoch and cannot represent them.
    #[arg(long)]
    strict_times: bool,

    /// OPT-IN salvage mode: coerce null or unparseable timestamps to Unix
    /// epoch 0 with a warning. Published builds are strict by default.
    #[arg(long, conflicts_with = "strict_times")]
    salvage_invalid_times: bool,

    /// Compatibility spelling for the default behavior: fail the build on any
    /// row with a null or unparseable geometry.
    #[arg(long)]
    strict_geometry: bool,

    /// OPT-IN salvage mode: skip rows with null or unparseable geometry and
    /// warn. Skipped rows are never tiled; published builds are strict by
    /// default.
    #[arg(long, conflicts_with = "strict_geometry")]
    salvage_invalid_geometry: bool,

    /// Auto-tune build flags by running stt-optimize's analyzer over the
    /// input before building. Bare `--auto` (= `--auto basic`) fills in the
    /// zoom range and temporal bucket only; `--auto encode` additionally
    /// applies the advisors' NON-LOSSY byte-level levers (zstd level via the
    /// `--publish` advice, `--blob-ordering`, `--pack-size`). Any flag the
    /// user passes explicitly still wins, and LOSSY advice (quantization,
    /// per-tile budgets) is never auto-applied in either mode — it is logged
    /// loudly as "suggested, not applied".
    #[arg(
        long,
        value_enum,
        value_name = "MODE",
        num_args(0..=1),
        require_equals = false,
        default_missing_value = "basic"
    )]
    auto: Option<AutoMode>,

    /// Build to a TARGET ARCHIVE SIZE. Implies `--auto encode`: the analyzer
    /// solves for a recipe that fits the budget using only REVERSIBLE levers
    /// (zoom clamp, temporal bucket width, temporal-LOD tiers, zstd level,
    /// blob ordering, pack size) and this build applies it.
    ///
    /// Bytes, or a `K`/`M`/`G` (binary) or `KB`/`MB`/`GB` (decimal) suffix;
    /// `KiB`/`MiB`/`GiB` are accepted spellings of the binary forms
    /// (`--target-size 250MiB`, `1.5G`, `262144000`).
    ///
    /// NOTHING IS EVER DROPPED to hit the number: no feature cap, no sampling,
    /// no aggregation, and no lossy flag. Quantization is priced and reported
    /// as a "shadow price" for you to opt into by hand. If the budget is
    /// unreachable with reversible levers alone, the build proceeds at the
    /// smallest reversible recipe (the floor) and says so.
    ///
    /// Requires a GeoParquet `--input` (the analyzer reads the file), same as
    /// `--auto`.
    #[arg(long = "target-size", value_name = "SIZE")]
    target_size: Option<String>,

    // --- Summary-tier options (server-aggregated low-zoom tier) ---
    /// Emit a pre-aggregated summary tier alongside the raw tier.
    /// Aggregation scheme: `h3` (Uber H3 hexes) or `quadbin` (CARTO quadbin).
    /// Both `h3` and `quadbin` are implemented.
    ///
    /// Summary tiles live in the same archive directory as raw tiles under
    /// manifest variant 1 (raw is variant 0), and use a distinct layer name
    /// (`summary` by default). Readers dispatch between the tiers automatically
    /// even when their `(z,x,y,t)` coordinates are identical.
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

    /// TB-12 ROLLBACK: bake triangle indices for EVERY feature of a
    /// triangle-bearing polygon layer, the pre-TB-12 all-or-nothing shape.
    ///
    /// By default a layer bakes indices only for features a renderer's own
    /// single-boundary earcut cannot reproduce (holes, multi-part) and leaves
    /// the rest empty for the decoder to backfill, which declares the
    /// `triangles-partial` capability. Pass this to emit the incumbent bytes
    /// and declare nothing — the escape hatch if a consumer turns out not to
    /// implement the backfill. Distinct from `--pre-tessellate`, which ADDS a
    /// triangle column to layers that would carry none.
    #[arg(long)]
    no_partial_triangles: bool,

    /// Drop tiles whose feature count is below this threshold. Default 1
    /// (write every non-empty tile). For globally sparse point datasets,
    /// raising this to 2-5 skips the long tail of single-feature deep-zoom
    /// tiles where the Arrow IPC + compression overhead dwarfs the payload.
    /// The TS reader's parent-fallback (`refinementStrategy: 'best-available'`)
    /// surfaces the skipped features from their parent tile.
    #[arg(long, default_value = "1")]
    min_features_per_tile: u32,

    /// Per-feature numeric property naming the shallowest zoom a feature
    /// appears at (road-class-style LOD). A feature is skipped at any zoom
    /// below its value — major roads when zoomed out, all streets up close.
    /// Whole-feature filtering only; geometry/attributes are untouched.
    #[arg(long)]
    min_zoom_field: Option<String>,

    /// Per-feature numeric property naming the DEEPEST zoom a feature appears
    /// at (LOD ceiling). A feature is skipped at any zoom above its value.
    /// Paired with `--min-zoom-field` it confines a feature to a zoom band
    /// `[min_zoom, max_zoom]` — e.g. coarse-zoom clustered/aggregated overviews
    /// that must not bleed into full-resolution deep zooms. Whole-feature
    /// filtering only; geometry/attributes are untouched.
    #[arg(long)]
    max_zoom_field: Option<String>,

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

    /// Bake the FULL per-property "style hints" profile into the archive
    /// metadata (`style_hints`): numeric percentiles (p50/p90/p95/p97/p99) plus
    /// a suggested color domain [min, p97] (each endpoint rounded outward to 2
    /// significant figures) and categorical distinct-value counts. Hints are
    /// DEFAULTS — the renderer/user can always override them. Values are sampled
    /// at a deterministic stride capped at ~250k values per property (memory
    /// guard). NOTE: the cheap signals — the layer-kind hint and suggested
    /// playback duration — are already baked on every non-streaming build
    /// WITHOUT this flag; this flag adds the expensive per-property profile.
    /// In-memory pipeline only: skipped with a warning under --streaming.
    #[arg(long)]
    style_hints: bool,

    /// Bake the DERIVED playback parameters into `style_hints`: refit
    /// `suggested_playback_seconds` to a target data frame rate
    /// (`clamp(K/20, K/30, K/12)` clamped to [5, 300] s, K = bucket count)
    /// instead of the legacy `clamp(round(sqrt(K)), 20, 90)`, and add
    /// `suggested_time_window_ms` — the widest playback window whose resident
    /// payload still fits a 256 MiB reference client budget, capped at 24
    /// buckets. Both are DEFAULTS the renderer/user always overrides.
    /// OFF by default: turning it on changes the emitted hint VALUES, so it
    /// rides a deliberate rebuild window rather than drifting one archive at a
    /// time. Needs the in-memory pipeline (like --style-hints): warned and
    /// ignored under --streaming, which totals no payload bytes.
    #[arg(long)]
    derived_playback_params: bool,

    /// Bake the SEMANTIC CONTENT FINGERPRINT (`metadata.content_fingerprint`,
    /// version 1) into the archive: the vertex bbox, vertical extent, distinct
    /// feature count, per-property numeric ranges and categorical
    /// cardinalities of the SOURCE features, computed before tiling and
    /// encode. `stt-validate` then recomputes them from the decoded tiles and
    /// compares — containment under `--sample`, equality within the declared
    /// tolerances under a full decode.
    ///
    /// This is the check that catches coordinates which are structurally
    /// valid but semantically wrong: a stride-2 read of a 3D `xyz` leaf once
    /// flattened and scrambled 106 archives, every one of which passed
    /// structural validation.
    ///
    /// OFF by default: it adds a manifest key, so it belongs to a deliberate
    /// rebuild window rather than drifting one archive at a time. Needs the
    /// in-memory pipeline (like --style-hints): warned and ignored under
    /// --streaming, which never holds the feature set.
    #[arg(long)]
    content_fingerprint: bool,

    /// Which geometric quantity `metadata.bounds` is taken from.
    ///
    /// `vertex` (the DEFAULT) declares the bbox of every geometry VERTEX — the
    /// quantity the tiler actually addresses tiles by, and a conservative
    /// SUPERSET of the data's extent. `centroid` declares the bbox of feature
    /// ANCHORS, which is what every archive built before this flag existed
    /// carries; for anything wider than a point that box provably UNDER-STATES
    /// the extent, and every consumer that pre-intersects a query box against
    /// `metadata.bounds` (tile selection, frustum pre-culling, the showcase's
    /// opening camera) then discards tiles that really do hold visible data —
    /// with no error anywhere in the stack.
    ///
    /// The chosen mode is recorded in the manifest as the
    /// `metadata.properties.bounds_mode` entry, so a reader — and `stt-validate`
    /// check 13 — can tell an attested vertex bbox from a legacy centroid one.
    /// `vertex` promotes a bbox that fails to contain its own decoded vertices
    /// from a warning to an error.
    ///
    /// `centroid` is the documented rollback for reproducing a pre-R1 archive's
    /// manifest values byte for byte. It is not a size or speed lever: both
    /// modes cost one pass over the same features and neither moves a pack byte.
    #[arg(long = "bounds-mode", value_enum, default_value_t = BoundsModeArg::Vertex, value_name = "MODE")]
    bounds_mode: BoundsModeArg,

    /// Whether the archive ATTESTS that its wire `id` column is a dataset-wide
    /// key — one distinct id per source feature, stable across every tile and
    /// pyramid level.
    ///
    /// This is the seam `stt-validate` check 12 reads to decide whether
    /// `distinct_feature_count` (a count of SOURCE FEATURES) may be compared
    /// against the distinct ids it decodes. On most archives it may NOT: a
    /// point whose source carries no id is written with the PER-TILE ROW INDEX
    /// (a deliberate, measured saving — the synthetic hash id was ~40 % of a
    /// point's compressed bytes), so archive-wide the distinct-id count is
    /// roughly the row count of the largest single tile. Comparing the two is a
    /// category error; a real 600-feature CONUS build decoded 5 distinct ids
    /// and was reported as "99.2 % of the declared features are MISSING" with
    /// nothing whatsoever missing.
    ///
    /// `auto` (the DEFAULT) proves it or declines it from the writer's OWN id
    /// construction, which it records in
    /// `metadata.properties.feature_id_construction`:
    ///
    /// * `source` — the source feature's id. A key.
    /// * `anchor-hash` — `FNV(timestamp, lon, lat)` over the feature anchor,
    ///   which the tiler copies into every clipped piece, so one source feature
    ///   keeps one id across every tile and zoom. A key. This is what an id-less
    ///   POLYGON, timeless LINE or `--no-clip` trip build gets, and it is why
    ///   those archives now get the strict comparison with no operator in the
    ///   loop.
    /// * `row-index` — an id-less POINT, whose id is the per-tile row index.
    ///   Not a key.
    /// * `segment-hash` — an id-less clipped trajectory, which mints a fresh id
    ///   per segment. Not a key.
    ///
    /// `auto` stamps `feature_id_scope = global` when every feature's
    /// construction is a key AND no two ids collide, `local` otherwise. The
    /// proof bails at the first non-key feature, so a no-id POINT archive pays
    /// nothing for it, and it is declined above
    /// `stt_build::input::FEATURE_ID_ATTESTATION_CAP` features rather than
    /// paying an unbounded allocation.
    ///
    /// `global` asserts the attestation WITHOUT the proof — for a dataset whose
    /// ids are known distinct but too numerous to prove. ⚠️ A false assertion
    /// here makes `stt-validate` report FEATURE LOSS on a healthy archive.
    ///
    /// `local` is the documented rollback: never attest, whatever the proof
    /// says. Check 12 then reports a distinct-id deviation as a note.
    ///
    /// Feature LOSS stays detectable in every mode — the validator's
    /// decoded-row floor compares rows at the fullest zoom against the same
    /// declared count, which is a comparison that holds regardless of id scope.
    #[arg(long = "feature-id-scope", value_enum, default_value_t = FeatureIdScopeArg::Auto, value_name = "SCOPE")]
    feature_id_scope: FeatureIdScopeArg,

    /// Also write a STAC Item (`stac.json`) beside the manifest, so the dataset
    /// is discoverable by any STAC catalog, browser or `pystac` reader without
    /// them understanding the STT format at all (STAC catalogs assets; it does
    /// not constrain their format — packed-format spec §10.3).
    ///
    /// Everything in the Item is derived from the finished manifest — bbox and
    /// geometry from the dataset bounds, `start_datetime`/`end_datetime` from
    /// the time range, tile/feature counts and zoom range as `stt:` properties
    /// — and the single asset href is RELATIVE (`./manifest.json`), so the pair
    /// stays valid wherever the directory is published.
    #[arg(long)]
    stac: bool,

    /// Ceiling (ms) on the per-vertex time quantization step. Vertex
    /// timestamps ride a compact u16-delta encoding whose step is derived
    /// from each tile layer's temporal span; a layer that would need a step
    /// coarser than this ceiling is stored as exact i64 timestamps instead
    /// (larger payload, zero precision loss). Default 1000 ms — below
    /// anything playback can show. Raise it only to trade precision for
    /// payload size on very wide temporal-LOD buckets.
    #[arg(long, default_value_t = stt_core::arrow_tile::DEFAULT_VERTEX_TIME_MAX_STEP_MS, value_name = "MS")]
    vertex_time_precision: u32,

    /// Opt-in coordinate quantization: store geometry as fixed-point integers at
    /// this ground precision in **meters** instead of Float64 lon/lat. `0` (the
    /// default) keeps Float64 GeoArrow coords. Coordinates are the dominant,
    /// near-incompressible tile column, so e.g. `--quantize-coords 1` (sub-meter
    /// error) is the largest size lever — measured −25..47% on trip/path
    /// datasets. Trade-off: a quantized tile is no longer self-describing
    /// Float64 GeoArrow (the per-tile affine rides in geometry field metadata;
    /// the STT reader reconstructs Float64).
    #[arg(long, default_value_t = 0.0, value_name = "METERS")]
    quantize_coords: f64,

    /// Opt-in numeric-attribute quantization: store the named Float64 property as
    /// fixed-point integers at the given precision (in the property's own units)
    /// instead of raw Float64, with a per-column affine in field metadata (the
    /// reader reconstructs Float64). Repeatable: `--quantize-attr z=0.05
    /// --quantize-attr speed=0.1`. A raw Float64 attribute is near-incompressible;
    /// for a LiDAR `z` elevation this is the largest size lever after the geometry
    /// — measured ~−80% on the `z` column. Default: none (all Float64).
    #[arg(long = "quantize-attr", value_name = "NAME=PREC")]
    quantize_attr: Vec<String>,

    /// Automatically quantize EVERY Float64 numeric property (that has no
    /// explicit `--quantize-attr` precision) to a range-adaptive `UInt16`: the
    /// column's `[min,max]` span is mapped onto 16 bits (~65k levels), the reader
    /// reconstructs Float64. A raw Float64 scalar column is near-incompressible;
    /// 16 bits of dynamic range is visually lossless for STT's scalar fields, so
    /// this is the "born-optimized" default for generated datasets. Off by
    /// default (keeps `stt-build` output byte-identical unless opted in).
    #[arg(long = "quantize-attrs-auto", default_value_t = false)]
    quantize_attrs_auto: bool,

    /// Fuse several scalar numeric properties into ONE interleaved GPU-ready
    /// column (`FixedSizeList<f32|u8, width>`) so the renderer binds it zero-copy
    /// with no per-point re-interleave on the main thread. Format:
    /// `NAME=col1,col2,...[:f32|:u8]` (default leaf `f32`; use `u8` for 0–255
    /// RGBA). The component order is the vector's component order. Repeatable:
    /// `--vector-group surfel_quat=qx,qy,qz,qw --vector-group surfel_rgba=r,g,b,a:u8`.
    /// The source scalar columns are removed from the tile. Default: none.
    #[arg(long = "vector-group", value_name = "NAME=COLS[:f32|u8]")]
    vector_group: Vec<String>,

    /// Fold a numeric property into POINT geometry as the 3rd (altitude)
    /// coordinate, so the tile ships true 3D points (`FixedSizeList<_,3>`) the
    /// renderer binds zero-copy — no per-point pad-to-3D on the main thread. The
    /// column is removed from the property set (it lives in the geometry). Only
    /// affects POINT layers. Pairs with `--quantize-coords` (the z axis is
    /// quantized to the same ground precision). Default: none (plain 2D points).
    #[arg(long = "point-elevation-column", value_name = "NAME")]
    point_elevation_column: Option<String>,

    /// Kill switch for the compact feature-time columns (ON by default, the
    /// `time-delta` capability). Enabled, each tile layer stores `start_time`
    /// as a `UInt32` millisecond offset from the layer's own minimum
    /// (`TILE_META.t0`) and `end_time` as a `UInt32` duration — or omits
    /// `end_time` entirely when every feature is instantaneous, which is 100%
    /// of features on most event datasets. Measured: the two columns are 33%
    /// of `nyc-taxi-points` and 17% of `earthquakes-v2` per-column cost. Both
    /// reference readers re-inflate absolute `Int64` columns, so nothing
    /// downstream can tell. Pass this to emit the historical absolute `Int64`
    /// pair instead (and suppress the capability) for byte-compat with a
    /// reader that predates it.
    #[arg(long = "no-compact-times", default_value_t = false)]
    no_compact_times: bool,

    /// Store the per-vertex value columns (`vertex_value`,
    /// `vertex_value_matrix`) as `UInt16` indices under a per-column
    /// range-adaptive affine instead of raw `Float32` — half the bytes, and
    /// the `vertex-value-quant` capability. Those two are the only
    /// `List<Float32>` columns the format carries and they had NO size lever
    /// at all (`--quantize-attr` / `--quantize-attrs-auto` cover per-feature
    /// scalar properties only), while measuring 64.2% of `nyc-taxi-flows` and
    /// 93.7% of `bixi-corridors` tile bytes. The `NaN` "no value at this
    /// vertex" marker survives via a reserved index. Off by default because
    /// it is genuinely lossy (16 bits across the column's own range) on data
    /// a map colours by — unlike the exact `--quantize-coords`-free defaults.
    #[arg(long = "quantize-vertex-values", default_value_t = false)]
    quantize_vertex_values: bool,

    /// Deprecated compatibility flag. It is accepted only when the encoder
    /// requires no capabilities, where it has no effect; a conforming v2 build
    /// refuses to suppress any required-to-understand declaration.
    #[arg(long = "no-manifest-capabilities", default_value_t = false)]
    no_manifest_capabilities: bool,

    // --- Per-tile budgets (OPT-IN; default OFF) ----------------------------
    // The project follows a documented "no thinning / comprehensive data by
    // default" principle. These caps are inert unless explicitly set, and when
    // they DO drop features they log exactly how many per affected tile.
    /// OPT-IN soft cap on a tile's estimated UNCOMPRESSED payload in bytes.
    /// When a tile exceeds this, its lowest-importance features are dropped to
    /// fit (importance-scored — never random; see `--drop-densest-as-needed`).
    /// Unset (the default) = no byte cap, every feature is kept. Each affected
    /// tile logs its dropped count. tippecanoe analogue: `--maximum-tile-bytes`.
    #[arg(long, value_name = "BYTES")]
    maximum_tile_bytes: Option<usize>,

    /// OPT-IN hard cap on the number of features per tile. When a tile exceeds
    /// this, its lowest-importance features are dropped to fit. Unset (the
    /// default) = no feature cap. Each affected tile logs its dropped count.
    /// tippecanoe analogue: `--maximum-tile-features`.
    #[arg(long, value_name = "N")]
    maximum_tile_features: Option<usize>,

    /// OPT-IN: when a per-tile budget drops features, prefer to drop from the
    /// DENSEST features first (importance-per-byte via geometry size). Only
    /// meaningful together with `--maximum-tile-bytes`/`--maximum-tile-features`.
    /// Without this flag a budget still drops the LEAST-important features
    /// first (a combined geometry+property score) — never randomly; this flag
    /// switches to a pure geometry-size density strategy (tippecanoe's
    /// `--drop-densest-as-needed`).
    #[arg(long)]
    drop_densest_as_needed: bool,

    // --- Attribute control (OPT-IN; default = keep every property) ---------
    /// OPT-IN: drop these property columns from output tiles (repeatable).
    /// System columns (id/time/geometry/vertex_*/triangles) always survive.
    /// Mutually exclusive with `--include`. tippecanoe analogue: `--exclude`.
    #[arg(long, value_name = "PROP")]
    exclude: Vec<String>,

    /// OPT-IN: keep ONLY these property columns (repeatable). System columns
    /// always survive regardless. Mutually exclusive with `--exclude`.
    /// tippecanoe analogue: `--include`.
    #[arg(long, value_name = "PROP")]
    include: Vec<String>,

    /// OPT-IN: drop EVERY user property — geometry + times only. Mutually
    /// exclusive with `--exclude`/`--include`. tippecanoe analogue:
    /// `--exclude-all`.
    #[arg(long)]
    exclude_all: bool,
}

/// Aggregation scheme for `--summary-tier`: Uber `h3` hexes or CARTO `quadbin`
/// cells. Both are implemented (see the `summary` module and the `SummaryScheme`
/// mapping below).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum SummaryTierScheme {
    H3,
    Quadbin,
}

/// Which geometric quantity `--bounds-mode` declares in `metadata.bounds`.
///
/// A thin clap mirror of [`stt_build::input::BoundsMode`] (the builder crate
/// carries no clap dependency); [`BoundsModeArg::resolve`] is the only mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum BoundsModeArg {
    /// Honest: the bbox of every geometry VERTEX. The default.
    Vertex,
    /// Legacy: the bbox of feature ANCHORS (centroids) — the documented
    /// rollback, and what every pre-R1 published archive declares.
    Centroid,
}

impl BoundsModeArg {
    fn resolve(self) -> stt_build::input::BoundsMode {
        match self {
            BoundsModeArg::Vertex => stt_build::input::BoundsMode::Vertex,
            BoundsModeArg::Centroid => stt_build::input::BoundsMode::Centroid,
        }
    }
}

/// What `--feature-id-scope` stamps into
/// `metadata.properties.feature_id_scope`.
///
/// The sibling of [`BoundsModeArg`]: both are writer ATTESTATIONS that promote
/// a `stt-validate` finding from "not comparable" to "enforced", and both ride
/// `metadata.properties` rather than `manifest.capabilities` (a reader rejects
/// an archive declaring a capability it does not implement, so a capability
/// here would make every attested archive unopenable by the deployed fleet).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum FeatureIdScopeArg {
    /// Prove it or decline it. The default.
    Auto,
    /// Assert globally distinct ids without proving them. ⚠️ A false assertion
    /// makes `stt-validate` report FEATURE LOSS on a healthy archive.
    Global,
    /// Never attest — the documented rollback.
    Local,
}

/// Resolve `--feature-id-scope` against the evidence, returning the value to
/// stamp and the line to log.
///
/// Split out of `main` so the decision table is unit-testable
/// (`feature_id_scope_resolution_is_a_total_table`): the `auto` row is the only
/// one that reads `attestation`, and `global` deliberately ignores it — that is
/// what "assert without proof" means — while still surfacing the disagreement.
fn resolve_feature_id_scope(
    arg: FeatureIdScopeArg,
    attestation: &stt_build::input::FeatureIdAttestation,
) -> (&'static str, String) {
    use stt_core::metadata::{FEATURE_ID_SCOPE_GLOBAL, FEATURE_ID_SCOPE_LOCAL};
    match arg {
        FeatureIdScopeArg::Auto => {
            if attestation.is_distinct() {
                (
                    FEATURE_ID_SCOPE_GLOBAL,
                    format!(
                        "Feature ids attested GLOBAL ({}) — stt-validate will compare \
                         distinct_feature_count against the decoded id count and treat a \
                         shortfall as feature loss",
                        attestation.reason()
                    ),
                )
            } else {
                (
                    FEATURE_ID_SCOPE_LOCAL,
                    format!(
                        "Feature ids recorded LOCAL: {}. distinct_feature_count and the decoded \
                         id count are then different quantities, so stt-validate reports any \
                         deviation as a NOTE and detects real loss through its decoded-row floor \
                         instead — which is LOOSE on geometries that clipping replicates across \
                         tiles",
                        attestation.reason()
                    ),
                )
            }
        }
        FeatureIdScopeArg::Global => (
            FEATURE_ID_SCOPE_GLOBAL,
            if attestation.is_distinct() {
                format!(
                    "Feature ids attested GLOBAL by --feature-id-scope global; the builder \
                     agrees ({})",
                    attestation.reason()
                )
            } else {
                format!(
                    "--feature-id-scope global ASSERTS globally distinct feature ids, but the \
                     builder could not prove it: {}. stt-validate will now treat a distinct-id \
                     shortfall on this archive as FEATURE LOSS — re-run with --feature-id-scope \
                     auto if that assertion is wrong",
                    attestation.reason()
                )
            },
        ),
        FeatureIdScopeArg::Local => (
            FEATURE_ID_SCOPE_LOCAL,
            "Feature ids recorded LOCAL by --feature-id-scope local (the documented rollback); \
             the distinct-id comparison stays disarmed"
                .to_string(),
        ),
    }
}

/// `--auto` tiers. `basic` (the bare-`--auto` default) fills in the zoom
/// range + temporal bucket only — exactly the pre-tier behaviour. `encode`
/// additionally applies the advisors' non-lossy byte-level encoding levers
/// for flags the user did not set explicitly. Lossy and semantic advice is
/// never auto-applied in either mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum AutoMode {
    Basic,
    Encode,
}

/// Where features come from: a GeoParquet file, a live PostGIS query, or a
/// DuckDB query. Every arm yields the same [`input::ParsedFeature`] stream, so
/// the rest of the build pipeline is source-agnostic.
#[derive(Clone)]
enum InputSource {
    File(PathBuf),
    #[cfg(feature = "postgres")]
    Postgres {
        conn: String,
        spec: stt_build::postgres_input::QuerySpec,
    },
    #[cfg(feature = "duckdb")]
    DuckDb {
        db_path: String,
        spec: stt_build::duckdb_input::QuerySpec,
    },
}

impl InputSource {
    /// Human-readable source description for logs.
    fn describe(&self) -> String {
        match self {
            InputSource::File(p) => p.display().to_string(),
            #[cfg(feature = "postgres")]
            InputSource::Postgres { spec, .. } => match &spec.source {
                stt_build::postgres_input::QuerySource::Table(t) => format!("PostGIS table {t}"),
                stt_build::postgres_input::QuerySource::Sql(_) => "PostGIS query".to_string(),
            },
            #[cfg(feature = "duckdb")]
            InputSource::DuckDb { spec, .. } => match &spec.source {
                stt_build::duckdb_input::QuerySource::Table(t) => format!("DuckDB table {t}"),
                stt_build::duckdb_input::QuerySource::Sql(_) => "DuckDB query".to_string(),
            },
        }
    }

    /// Default archive name when `--name` is not given.
    fn default_name(&self) -> String {
        match self {
            InputSource::File(p) => p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "dataset".to_string()),
            #[cfg(feature = "postgres")]
            InputSource::Postgres { spec, .. } => match &spec.source {
                stt_build::postgres_input::QuerySource::Table(t) => {
                    t.rsplit('.').next().unwrap_or(t).to_string()
                }
                stt_build::postgres_input::QuerySource::Sql(_) => "postgis".to_string(),
            },
            #[cfg(feature = "duckdb")]
            InputSource::DuckDb { spec, .. } => match &spec.source {
                stt_build::duckdb_input::QuerySource::Table(t) => {
                    t.rsplit('.').next().unwrap_or(t).to_string()
                }
                stt_build::duckdb_input::QuerySource::Sql(_) => "duckdb".to_string(),
            },
        }
    }

    /// Authoritative property kinds from the source's schema, for
    /// `TileConfig::property_types` — pins every schema column's tile kind so
    /// a column that is all-null within one tile still gets its (all-null)
    /// column there instead of drifting the layer schema. GeoParquet derives
    /// this from the Arrow schema; the DB adaptors probe their result schema
    /// (DuckDB: `LIMIT 0` execution; PostGIS: statement prepare) so all three
    /// sources pin the same way. Any probe failure falls back to per-tile
    /// sniffing — the real load will surface the underlying error with full
    /// context, so the build is never failed from here.
    fn property_kinds(
        &self,
        time_field: &str,
        end_time_field: Option<&str>,
    ) -> Arc<stt_build::columnar::PropertyTypes> {
        let fall_back = |source: &str, e: anyhow::Error| {
            warn!(
                "could not derive property kinds from the {source} schema ({e}); \
                 falling back to per-tile type sniffing"
            );
            Arc::default()
        };
        match self {
            InputSource::File(path) => {
                match input::property_kinds(path, time_field, end_time_field) {
                    Ok(kinds) => Arc::new(kinds),
                    Err(e) => fall_back("input", e),
                }
            }
            #[cfg(feature = "postgres")]
            InputSource::Postgres { conn, spec } => {
                match stt_build::postgres_input::property_kinds(
                    conn,
                    spec,
                    time_field,
                    end_time_field,
                ) {
                    Ok(kinds) => Arc::new(kinds),
                    Err(e) => fall_back("PostGIS result", e),
                }
            }
            #[cfg(feature = "duckdb")]
            InputSource::DuckDb { db_path, spec } => {
                match stt_build::duckdb_input::property_kinds(
                    db_path,
                    spec,
                    time_field,
                    end_time_field,
                ) {
                    Ok(kinds) => Arc::new(kinds),
                    Err(e) => fall_back("DuckDB result", e),
                }
            }
        }
    }

    /// Eager load — collect every feature into memory.
    fn load(
        &self,
        time_field: &str,
        end_time_field: Option<&str>,
        time_format: input::TimeFormat,
        time_strictness: input::InputStrictness,
        geometry_strictness: input::InputStrictness,
    ) -> Result<Vec<input::ParsedFeature>> {
        match self {
            InputSource::File(path) => input::load_features(
                path,
                time_field,
                end_time_field,
                time_format,
                time_strictness,
                geometry_strictness,
            ),
            #[cfg(feature = "postgres")]
            InputSource::Postgres { conn, spec } => {
                stt_build::postgres_input::load_features_postgres(
                    conn,
                    spec,
                    time_field,
                    end_time_field,
                    time_format,
                    time_strictness,
                    geometry_strictness,
                )
            }
            #[cfg(feature = "duckdb")]
            InputSource::DuckDb { db_path, spec } => stt_build::duckdb_input::load_features_duckdb(
                db_path,
                spec,
                time_field,
                end_time_field,
                time_format,
                time_strictness,
                geometry_strictness,
            ),
        }
    }
}

/// Resolve the `Args` into a concrete [`InputSource`], validating that exactly
/// one of the file / PostGIS / DuckDB sources is requested.
fn resolve_source(args: &Args) -> Result<InputSource> {
    let wants_postgres = args.postgres.is_some();
    let wants_duckdb = args.duckdb.is_some();
    if wants_postgres && wants_duckdb {
        anyhow::bail!("--postgres and --duckdb are mutually exclusive");
    }
    // `--table`/`--sql` with no explicit backend default to PostGIS (which can
    // take its connection from STT_POSTGRES_URL / DATABASE_URL), preserving the
    // prior behaviour; DuckDB always needs an explicit `--duckdb <PATH|:memory:>`.
    let wants_db = wants_postgres || wants_duckdb || args.table.is_some() || args.sql.is_some();
    if wants_db {
        if args.input.is_some() {
            anyhow::bail!(
                "Provide either --input or a database source (--postgres/--duckdb with \
                 --table/--sql), not both"
            );
        }
        if wants_duckdb {
            #[cfg(feature = "duckdb")]
            {
                return resolve_duckdb_source(args);
            }
            #[cfg(not(feature = "duckdb"))]
            {
                let _ = (&args.geom_column, &args.where_clause, &args.source_srid);
                anyhow::bail!(
                    "stt-build was built without DuckDB support; rebuild with `--features duckdb`"
                );
            }
        }
        // Explicit `--postgres`, or bare `--table`/`--sql` (env-var connection).
        #[cfg(feature = "postgres")]
        {
            return resolve_postgres_source(args);
        }
        #[cfg(not(feature = "postgres"))]
        {
            // Keep the DB-only flags "read" in this build so they don't warn,
            // then explain why they can't be honoured.
            let _ = (&args.geom_column, &args.where_clause, &args.source_srid);
            anyhow::bail!(
                "stt-build was built without PostGIS support; rebuild with `--features postgres` \
                 (or pass --duckdb for the DuckDB source)"
            );
        }
    }

    let path = args.input.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "no input: pass --input <GeoParquet>, --postgres <CONN> with --table/--sql, or \
             --duckdb <PATH> with --table/--sql"
        )
    })?;
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if !matches!(ext.to_lowercase().as_str(), "parquet" | "geoparquet") {
        anyhow::bail!("Input must be a GeoParquet file (.parquet or .geoparquet), got: .{ext}");
    }
    Ok(InputSource::File(path))
}

/// Build a DuckDB [`InputSource`] from the relevant `Args`. Only compiled with
/// the `duckdb` feature.
#[cfg(feature = "duckdb")]
fn resolve_duckdb_source(args: &Args) -> Result<InputSource> {
    use stt_build::duckdb_input::{QuerySource, QuerySpec};
    // `wants_duckdb` gates this call, so `--duckdb` is Some; empty / ":memory:"
    // both mean an in-memory database (for scanning files via `--sql`).
    let db_path = args
        .duckdb
        .clone()
        .unwrap_or_else(|| ":memory:".to_string());
    let source = match (&args.table, &args.sql) {
        (Some(_), Some(_)) => anyhow::bail!("--table and --sql are mutually exclusive"),
        (Some(t), None) => QuerySource::Table(t.clone()),
        (None, Some(s)) => QuerySource::Sql(s.clone()),
        (None, None) => anyhow::bail!("--duckdb requires --table <NAME> or --sql <SELECT>"),
    };
    Ok(InputSource::DuckDb {
        db_path,
        spec: QuerySpec {
            source,
            geom_column: args.geom_column.clone(),
            where_clause: args.where_clause.clone(),
            reproject_from_srid: args.source_srid,
        },
    })
}

/// Build a PostGIS [`InputSource`] from the relevant `Args`. Only compiled with
/// the `postgres` feature.
#[cfg(feature = "postgres")]
fn resolve_postgres_source(args: &Args) -> Result<InputSource> {
    use stt_build::postgres_input::{QuerySource, QuerySpec};
    let conn = args
        .postgres
        .clone()
        .or_else(|| std::env::var("STT_POSTGRES_URL").ok())
        .or_else(|| std::env::var("DATABASE_URL").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--table/--sql require a connection: pass --postgres <CONN> or set \
                 STT_POSTGRES_URL / DATABASE_URL"
            )
        })?;
    let source = match (&args.table, &args.sql) {
        (Some(_), Some(_)) => anyhow::bail!("--table and --sql are mutually exclusive"),
        (Some(t), None) => QuerySource::Table(t.clone()),
        (None, Some(s)) => QuerySource::Sql(s.clone()),
        (None, None) => anyhow::bail!("--postgres requires --table <NAME> or --sql <SELECT>"),
    };
    Ok(InputSource::Postgres {
        conn,
        spec: QuerySpec {
            source,
            geom_column: args.geom_column.clone(),
            where_clause: args.where_clause.clone(),
            reproject_from_srid: args.source_srid,
        },
    })
}

/// Resolve the dataset-wide property kinds fed to every tile builder.
///
/// The input SCHEMA is authoritative wherever it has an answer: GeoParquet
/// column types, Postgres/DuckDB column types. But it does not answer for every
/// column — `property_kind_for` maps the numeric/string/bool types and returns
/// nothing for the rest (a `Dictionary`-encoded, `List`, or all-`Null` column,
/// say), and the whole map is empty if the schema could not be read at all.
///
/// A column the schema doesn't cover falls through to the tile builder, which
/// sniffs its kind from the features of ONE tile. That drifts the layer schema
/// across tiles two ways: a column whose values are all null in some tile
/// disappears from that tile's schema, and a column holding only
/// numeric-looking strings in one tile but a real string in another is numeric
/// in the first and categorical in the second. `stt-validate` reports both as
/// schema drift, and a consumer styling on the column gets a different type
/// depending on which tile it reads.
///
/// So: take the schema's answers, then fill the GAPS from a single pass over
/// every loaded feature. Schema entries are never overridden — a `Utf8` column
/// whose values all look numeric stays categorical, as documented. The
/// gap-filling rule is the same one the tile builder would have used; only the
/// evidence widens from one tile to the whole dataset.
fn resolve_property_types(
    source: &InputSource,
    args: &Args,
    features: &[stt_build::input::ParsedFeature],
    attribute_filter: &stt_build::columnar::AttributeFilter,
) -> Arc<stt_build::columnar::PropertyTypes> {
    let declared = source.property_kinds(&args.time_field, args.end_time_field.as_deref());
    let (merged, filled) = stt_build::columnar::fill_property_type_gaps(
        &declared,
        features.iter().map(|f| f.shared_properties.as_ref()),
        attribute_filter,
    );
    if filled.is_empty() {
        return declared;
    }
    info!(
        "Pinned {} property kind(s) the input schema did not type ({}) from a \
         whole-dataset pass, so every tile emits the same columns",
        filled.len(),
        filled.join(", ")
    );
    Arc::new(merged)
}

fn main() -> Result<()> {
    let matches = Args::command().get_matches();
    let mut args =
        Args::from_arg_matches(&matches).context("failed to parse stt-build arguments")?;

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
    // Resolve + validate the input source (GeoParquet file or PostGIS query).
    let source = resolve_source(&args)?;
    info!("Input: {}", source.describe());
    info!("Output (packed dataset dir): {}", out_dir.display());

    // Validate attribute-filter flags up front — before loading input — so a
    // misuse like `--exclude X --include Y` (or excluding a column that
    // `--heatmap-weight`/`--summary-columns` depends on) fails fast with a clear
    // message instead of after the GeoParquet file is opened. The filter is
    // rebuilt at the tiling stage; this call is purely for early validation.
    build_attribute_filter(&args)?;

    // `--target-size` IMPLIES `--auto encode`: the budget's answer is a set of
    // build flags, and only `encode` mode applies byte-level levers. Parsed
    // BEFORE the analysis so a typo'd size fails in milliseconds rather than
    // after a multi-second measurement pass.
    let target_bytes = match args.target_size.as_deref() {
        Some(text) => Some(stt_optimize::parse_size(text)?),
        None => None,
    };
    let auto_mode = match (args.auto, target_bytes) {
        (Some(AutoMode::Encode), _) | (None, None) => args.auto,
        (None, Some(_)) => {
            info!("--target-size implies --auto encode");
            Some(AutoMode::Encode)
        }
        (Some(AutoMode::Basic), Some(_)) => {
            // Not an error: the user asked for both, and `encode` is a superset
            // of `basic`. Loud, because it overrides something they typed.
            warn!(
                "--target-size implies --auto encode; upgrading the explicit --auto basic \
                 (a budget's answer IS byte-level levers, which basic mode does not apply)"
            );
            Some(AutoMode::Encode)
        }
        (Some(AutoMode::Basic), None) => args.auto,
    };
    if let Some(mode) = auto_mode {
        apply_auto_recommendations(&matches, &mut args, mode, target_bytes)?;
    }

    // --publish bundles the lossless deploy wins into the core build path so a
    // from-source build is deploy-ready (no separate re-transcode). Each bundled
    // setting yields to an explicit override on the command line.
    if args.publish {
        let explicit =
            |name: &str| matches!(matches.value_source(name), Some(ValueSource::CommandLine));
        if !explicit("zstd_level") {
            args.zstd_level = 19;
        }
        // (The directory is paged by default, so --publish only bumps the level.)
        info!("Publish build: zstd-{}", args.zstd_level);
    }

    // Encoder settings (vertex-time precision, coordinate + numeric-attribute
    // quantization, vector grouping, point-elevation fold). Parsed by the shared
    // `build_options` module — the same one `stt-serve` uses, so a served tile
    // is byte-identical to this build's — into an EXPLICIT `EncoderConfig` that
    // is handed to the pack writer below and reaches every encode as an
    // argument. Nothing process-wide is mutated: two datasets built in one
    // process (or served concurrently) cannot inherit each other's settings.
    let encoder_settings = EncoderSettings {
        vertex_time_precision: Some(args.vertex_time_precision),
        quantize_coords_m: args.quantize_coords,
        quantize_attr: args.quantize_attr.clone(),
        quantize_attrs_auto: args.quantize_attrs_auto,
        vector_group: args.vector_group.clone(),
        point_elevation_column: args.point_elevation_column.clone(),
        no_compact_times: args.no_compact_times,
        quantize_vertex_values: args.quantize_vertex_values,
    };
    // `mut` because pass 1 (below, after the features are loaded and their
    // property kinds resolved) fills in `global_pins`.
    let mut encoder_config = encoder_settings.resolve()?;
    // Snapshot the value-moving levers for SH-1's content fingerprint before the
    // config is handed to the writer (it moves). Taken from the RESOLVED config,
    // not from the raw flags: `--auto` can rewrite quantization behind the
    // user's back, and a declared tolerance that does not match the encoding
    // actually applied is what turns an honest quantized archive red.
    let fingerprint_levers = (
        encoder_config.quantize_coords_m,
        encoder_config
            .quantize_attrs
            .iter()
            .map(|(name, precision)| (name.clone(), *precision))
            .collect::<std::collections::BTreeMap<String, f64>>(),
        encoder_config.point_elevation_column.clone(),
    );
    let enabled = encoder_settings.enabled_summary()?;
    if !enabled.is_empty() {
        info!("Encoder settings ENABLED: {}", enabled.join(", "));
    }

    // Validate the --compression flag (packed output is zstd-only; the legacy
    // gzip/none choices now error). The parsed value is otherwise unused —
    // PackWriter is always per-blob zstd.
    parse_compression(&args.compression)?;

    // Parse the pack ordering (the packed format always buffers + reorders
    // before cutting packs). `measured` — the DEFAULT since the M4 workload
    // model closed the weighting question — is the per-dataset picker resolved
    // by simulating range-read cost at finalize (it uses the Auto slot as its
    // placeholder ordering, and degrades to Auto's heuristic on inputs below the
    // simulation floor).
    let blob_arg = args.blob_ordering.trim();
    let measured_ordering = blob_arg.eq_ignore_ascii_case("measured");
    let pack_ordering: stt_core::BlobOrdering = if measured_ordering {
        stt_core::BlobOrdering::Auto
    } else {
        blob_arg.parse().map_err(|e: String| anyhow::anyhow!(e))?
    };
    let ordering_workload = parse_ordering_workload(&args.ordering_workload)?;

    // Pack target size (MiB -> bytes). Never 0.
    let pack_target_bytes = args.pack_size.saturating_mul(1024 * 1024).max(1);

    let (time_strictness, geometry_strictness) = input_strictness(&args);

    // Step 1: Load all features into memory
    info!("Loading input data...");
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message("Reading input file...");

    let mut features = source.load(
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

    // `--point-elevation-column` is a POINT-layer lever (`arrow_tile::encode`
    // gates the fold on `GeometryColumn::Point`). On a line/polygon build the
    // encoder silently ignores it — the column stays an ordinary property and
    // the tiles stay flat. Say so, because the two things an operator expects
    // from the flag (3D geometry, and the column gone from properties) both
    // fail to happen, and a silent no-op is how a build ships believing it is
    // volumetric.
    if !encoder_config.point_elevation_column.is_empty() {
        let folded = features
            .iter()
            .filter(|f| {
                matches!(
                    stt_build::columnar::determine_geometry_type(f),
                    Ok(stt_core::types::GeometryType::Point)
                )
            })
            .count();
        if folded == 0 {
            warn!(
                "--point-elevation-column {} matched NO point features: the fold only applies \
                 to POINT layers, so the column stays an ordinary property, the tiles stay 2D, \
                 and no z_range is claimed. Use the renderer's `elevationProperty` over the \
                 column instead.",
                encoder_config.point_elevation_column
            );
        } else if folded < features.len() {
            warn!(
                "--point-elevation-column {}: {} of {} features are points and get the fold; \
                 the remaining {} keep the column as a property (the fold is POINT-only), so \
                 the archive is mixed",
                encoder_config.point_elevation_column,
                folded,
                features.len(),
                features.len() - folded
            );
        }
    }

    // Step 2: Analyze data bounds (SH-2).
    //
    // ONE pass yields the declared bbox, the temporal extent and the vertical
    // extent. The declared bbox defaults to the honest VERTEX box: the tiler
    // addresses tiles by vertex (`tiler::place_non_trajectory` walks
    // `feature.geojson.geometry`), so the legacy centroid box provably
    // under-states every non-point geometry — and an under-stated
    // `metadata.bounds` is silent data loss at QUERY time for everything that
    // pre-intersects against it. Both boxes are computed whatever the mode, so
    // the widening can be reported rather than merely happening.
    //
    // ⚠️ `--point-elevation-column` must be forwarded here: the encoder folds
    // that column into POINT z long after this pass, so without it the manifest
    // would claim no vertical extent for an archive whose tiles decode to 3D.
    // The profiler applies the same POINT-only gate the encoder does.
    info!("Analyzing data bounds...");
    let bounds_mode = args.bounds_mode.resolve();
    let profile = input::profile_features_with(
        &features,
        &input::FeatureProfileOptions {
            bounds_mode,
            elevation_column: (!encoder_config.point_elevation_column.is_empty())
                .then_some(encoder_config.point_elevation_column.as_str()),
        },
    )?;
    let bounds = profile.bounds;
    let time_range = profile.time_range;
    info!(
        "Spatial bounds ({}): {:?}",
        bounds_mode.as_manifest_value(),
        bounds
    );
    if profile.vertex_bounds != profile.centroid_bounds {
        match bounds_mode {
            input::BoundsMode::Vertex => info!(
                "Declared bounds widened from the legacy centroid box {:?} to the vertex box \
                 {:?} — the honest superset the tiler addresses tiles by",
                profile.centroid_bounds, profile.vertex_bounds
            ),
            input::BoundsMode::Centroid => warn!(
                "--bounds-mode centroid declares {:?}, which does NOT contain this dataset's \
                 vertex bbox {:?}: readers that pre-intersect a query box against \
                 metadata.bounds (tile selection, frustum pre-culling, the opening camera) will \
                 discard tiles that hold visible data. This is the legacy rollback; prefer \
                 --bounds-mode vertex",
                profile.centroid_bounds, profile.vertex_bounds
            ),
        }
    }
    if let Some([lo, hi]) = profile.z_range {
        info!("Vertical extent: [{lo}, {hi}]");
    }
    info!("Time range: {} to {}", time_range.start, time_range.end);

    // Step 3: Generate tiles
    info!("Generating tiles...");

    // Parse temporal bucket size
    let temporal_bucket_ms = parse_duration(&args.temporal_bucket)?;
    info!(
        "Temporal bucket size: {} ms ({}))",
        temporal_bucket_ms, args.temporal_bucket
    );

    // SH-4: a baked LOD-floor column (`--min-zoom-field`) was computed against a
    // BUCKET-KEYED thinning grid. If that grid's bucket and this build's bucket
    // disagree, the archive builds, validates and renders — at the wrong density
    // (median 13 %, worst 0 % of the visible bucket at z8 in the recorded
    // incident). The comparison is deliberately placed AFTER `--auto` may have
    // overwritten `--temporal-bucket` above: an auto-overridden bucket is the
    // silent-mismatch path a hand-passed flag never hits. Byte-neutral — a
    // passing build is bit-identical to one without the check.
    {
        let (stamped, malformed) = match &source {
            InputSource::File(path) => stt_build::lod_bucket::lod_grid_bucket_ms(path),
            // Non-Parquet sources have no LOD-column pipeline; nothing declares
            // a grid, so the check degrades to its warn path.
            #[allow(unreachable_patterns)]
            _ => (None, None),
        };
        if let Some(note) = malformed {
            warn!("{note}");
        }
        match stt_build::lod_bucket::check_lod_grid_bucket(
            stamped,
            args.min_zoom_field.as_deref(),
            temporal_bucket_ms,
        ) {
            stt_build::lod_bucket::LodBucketCheck::Ok => {}
            stt_build::lod_bucket::LodBucketCheck::Warn(msg) => warn!("{msg}"),
            stt_build::lod_bucket::LodBucketCheck::Mismatch(msg) => anyhow::bail!(msg),
        }
    }

    if !args.no_clip {
        info!(
            "Trajectory clipping enabled (min {} vertices)",
            args.clip_min_vertices
        );
    } else {
        info!("Trajectory clipping disabled (--no-clip)");
    }

    if args.simplify {
        info!(
            "Line simplification enabled (max zoom {})",
            args.simplify_max_zoom
        );
    }

    // TB-9: pass-1's per-zoom byte mass, used to default coarse-tier cutoffs.
    let mut lod_zoom_mass: Option<Vec<u64>> = None;
    let mut adaptive_boundaries: Vec<u64> = Vec::new();
    // TB-9: parsed with the legacy per-zoom default here; re-resolved against
    // pass-1's per-zoom byte mass once that exists (an explicit `@z` always wins).
    let mut temporal_lod = match args.temporal_lod.as_deref() {
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

    // Opt-in per-tile budget + attribute control (default OFF — the build is
    // byte-for-byte identical to before when none of these flags are set).
    let tile_budget = build_tile_budget(&args);
    if let Some(b) = &tile_budget {
        warn!(
            "Per-tile budget ENABLED (opt-in): max_features={}, max_bytes={}, \
             drop strategy={} — tiles over the cap will have their lowest-value \
             features dropped (each affected tile logs its dropped count)",
            if b.max_feature_count == usize::MAX {
                "∞".to_string()
            } else {
                b.max_feature_count.to_string()
            },
            if b.max_uncompressed_size == usize::MAX {
                "∞".to_string()
            } else {
                b.max_uncompressed_size.to_string()
            },
            if args.drop_densest_as_needed {
                "densest-first (geometry size)"
            } else {
                "least-important-first (combined)"
            },
        );
    }
    let attribute_filter = build_attribute_filter(&args)?;
    if !attribute_filter.is_keep_all() {
        info!("Attribute control ENABLED (opt-in): {:?}", attribute_filter);
    }

    let property_types = resolve_property_types(&source, &args, &features, &attribute_filter);

    // ---- Pass 1: the dataset-global statistics scan -------------------------
    //
    // Sits exactly here, between load and `generate_tiles*`, for a reason: it
    // needs the RESOLVED property kinds (above) and the attribute filter, and
    // it must run before the first tile is encoded. It re-reads nothing from
    // the source — one iteration over the already-resident feature vector,
    // which is what every backend returns (GeoParquet, PostGIS, DuckDB all
    // funnel through `InputSource::load`, so no backend can miss this).
    // DT-2: additive home-zoom assignment. Runs on the resident feature vector
    // (same funnel as pass 1) and synthesizes a per-feature band, then routes
    // through the EXISTING min_zoom_field/max_zoom_field mechanism — the
    // placement authority needs no change.
    let mut home_zoom_declared = false;
    if let Some(s_px) = args.additive_lod {
        if args.min_zoom_field.is_some() {
            anyhow::bail!(
                "--additive-lod computes a per-feature home zoom and would overwrite the band \
                 --min-zoom-field reads; pass one or the other"
            );
        }
        if !(s_px > 0.0) {
            anyhow::bail!("--additive-lod voxel pitch must be > 0 (got {s_px})");
        }
        let started = std::time::Instant::now();
        let candidates: Vec<stt_build::home_zoom::HomeZoomCandidate> = features
            .iter()
            .enumerate()
            .map(|(i, f)| stt_build::home_zoom::HomeZoomCandidate {
                id: i as u64,
                lon: f.lon,
                lat: f.lat,
                timestamp: f.timestamp,
                importance: 0.0,
            })
            .collect();
        let home = stt_build::home_zoom::assign_home_zooms(
            &candidates,
            args.min_zoom,
            args.max_zoom,
            s_px,
            temporal_bucket_ms,
        );
        let mut per_zoom: std::collections::BTreeMap<u8, usize> = Default::default();
        for z in home.values() {
            *per_zoom.entry(*z).or_default() += 1;
        }
        info!(
            "DT-2 additive home-zoom: {} feature(s) assigned across z{}..z{} in {:.2?} \
             (replication build would store ~{}x more index rows); per-zoom {:?}",
            home.len(),
            args.min_zoom,
            args.max_zoom,
            started.elapsed(),
            (args.max_zoom - args.min_zoom) as usize + 1,
            per_zoom
        );
        // Stamp the assignment onto the features; the tiler's band mechanism
        // reads `home_zoom` ahead of any configured field.
        for (i, f) in features.iter_mut().enumerate() {
            f.home_zoom = home.get(&(i as u64)).copied();
        }
        home_zoom_declared = true;
    }

    let encoder_pins = if args.single_pass {
        warn!(
            "--single-pass: skipping the dataset-global statistics scan. Every tile decides its \
             own numeric affine, leaf width and dictionary-vs-Utf8 verdict from its own rows, so \
             the same value can decode differently in different tiles and one column can fork a \
             schema template per tile shape. This is a byte-compatibility escape hatch, not a \
             performance one."
        );
        None
    } else {
        let started = std::time::Instant::now();
        let stats = stt_build::dataset_stats::collect_dataset_stats_with(
            &features,
            &attribute_filter,
            &property_types,
            &stt_build::dataset_stats::StatsOptions {
                zoom_range: (args.min_zoom, args.max_zoom),
                min_zoom_field: args.min_zoom_field.clone(),
                max_zoom_field: args.max_zoom_field.clone(),
                ..Default::default()
            },
        );
        lod_zoom_mass = Some(stats.byte_mass.per_zoom.clone());
        // TB-10: the dataset-wide candidate boundary set adaptive window keys
        // snap onto. Derived here because it must be the SAME set for every
        // spatial cell — that is the entire mechanism — and pass 1 is the only
        // place that has seen the whole timestamp distribution.
        if args.adaptive_temporal.is_some() && !args.adaptive_greedy {
            let mut b = stats.timestamps.quantiles(args.adaptive_boundary_count);
            b.sort_unstable();
            b.dedup();
            info!(
                "TB-10: {} shared adaptive boundaries from {} requested quantiles",
                b.len(),
                args.adaptive_boundary_count
            );
            adaptive_boundaries = b;
        }
        let pins = stats.to_pins();
        // The per-stage timing the acceptance metric is measured against
        // (target: pass 1 under 2% of total build time).
        info!(
            "Pass 1 (dataset statistics): {} feature(s), {} numeric + {} categorical column(s) \
             pinned in {:.2?}",
            stats.features,
            pins.attr.len(),
            pins.dict.len(),
            started.elapsed()
        );
        // The two diagnostics a per-tile encoder structurally cannot emit,
        // because each needs the whole domain to be sure of. The magnitude
        // refusal only means anything on the auto-quantization path — with the
        // lever off, every numeric column is Float64 anyway and the warning
        // would be noise.
        if encoder_config.quantize_attrs_auto {
            for (name, max_abs) in stats.refused_numeric_columns() {
                warn!(
                    "property {name:?} reaches |{max_abs:.3e}| across the dataset, at or past \
                     the auto-quantization magnitude ceiling: the column stays Float64 in EVERY \
                     tile. If it is an identifier this is correct; if it is a measurement with a \
                     rare outlier, the outlier is what costs you the size lever."
                );
            }
        }
        let overflowed = stats.overflowed_categorical_columns();
        if !overflowed.is_empty() {
            warn!(
                "categorical column(s) {} exceeded the pass-1 distinct-value cap ({} values / \
                 {} MiB): no global dictionary can be pinned for them, so they ship as plain \
                 Utf8 in every tile",
                overflowed.join(", "),
                stats.category_cap,
                stats.category_byte_cap / (1024 * 1024),
            );
        }
        tracing::debug!("encoder pins: {}", pins.to_canonical_json());
        Some(std::sync::Arc::new(pins))
    };
    // The pins ride the writer's `EncoderConfig`, so every tile sink (streaming,
    // batched, LOD, summary) sees them without process-wide state — the same
    // channel `stt-serve` fills per dataset.
    encoder_config.global_pins = encoder_pins;

    // TB-9: re-resolve un-annotated temporal-LOD cutoffs against the per-zoom
    // byte mass pass 1 measured. A coarse tier serves LOW-zoom queries, so
    // emitting it at every zoom is maximal duplication; the duplication term is
    // dominated by the high-zoom byte share. An explicit `@z` always wins, and
    // with no mass (single-pass builds) this is a no-op.
    if let (Some(spec), Some(mass)) = (args.temporal_lod.as_deref(), lod_zoom_mass.as_deref()) {
        let resolved = build_options::parse_temporal_lod_with_mass(
            spec,
            args.max_zoom,
            Some(mass),
            (args.min_zoom, args.max_zoom),
        )?;
        for (before, after) in temporal_lod.iter().zip(resolved.iter()) {
            if before.max_zoom_level != after.max_zoom_level {
                info!(
                    "temporal-LOD tier {}ms: default cutoff z{} (was z{}) from the per-zoom byte mass",
                    after.bucket_ms, after.max_zoom_level, before.max_zoom_level
                );
            }
        }
        temporal_lod = resolved;
    }

    let tile_config = tiler::TileConfig {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        layer_name: args.layer.clone(),
        temporal_bucket_ms,
        clip_trajectories: !args.no_clip,
        clip_min_vertices: args.clip_min_vertices,
        clip_buffer_px: args.clip_buffer_px,
        clip_buffer_degrees: args.clip_buffer_degrees,
        simplify: args.simplify,
        simplify_max_zoom: args.simplify_max_zoom,
        // TB-8: metric is the default; the degree table is the rollback.
        // `--simplify-metric` remains an accepted no-op alias for one release.
        simplify_metric: !args.simplify_degree_table,
        pre_tessellate: args.pre_tessellate,
        partial_triangles: !args.no_partial_triangles,
        temporal_lod: temporal_lod.clone(),
        min_features_per_tile: args.min_features_per_tile,
        time_aware_simplify: args.time_aware_simplify,
        adaptive_target_features: args.adaptive_temporal,
        adaptive_greedy: args.adaptive_greedy,
        adaptive_boundaries: adaptive_boundaries.clone(),
        min_zoom_field: args.min_zoom_field.clone(),
        max_zoom_field: args.max_zoom_field.clone(),
        tile_budget,
        attribute_filter,
        property_types,
        partial_triangles_observed: Default::default(),
    };

    if args.pre_tessellate {
        info!("Pre-tessellation enabled (triangle indices written alongside polygon geometry)");
    }

    info!(
        "Pack ordering: {pack_ordering} (buffered — space-time blob layout + \
         byte-identical dedup, then cut into packs of ≤{} MiB; payloads buffer \
         until finalize, spilling to disk beyond --pack-memory-budget)",
        args.pack_size
    );
    // Required-to-understand capability declarations (manifest.capabilities,
    // packed spec §3.1): derived from the encoder settings actually enabled,
    // so an older reader that lacks e.g. coord-quant refuses this dataset at
    // open instead of silently misdecoding its re-typed columns. Additive
    // features (vector groups, triangles) are never declared.
    let mut manifest_capabilities = encoder_settings.required_capabilities();
    // DT-2 + DT-1: home-zoom changes what the base tier at ONE zoom contains,
    // so it is a MUST-UNDERSTAND capability. Without it an older parent-fallback
    // reader would render a sparse per-zoom slice as if it were complete — the
    // silent-misdecode class capabilities exist to turn into a loud refusal.
    if home_zoom_declared {
        manifest_capabilities.push(stt_core::metadata::CAPABILITY_ADDITIVE_PARTITION.to_string());
        manifest_capabilities.sort();
        manifest_capabilities.dedup();
    }
    if args.no_manifest_capabilities {
        if !manifest_capabilities.is_empty() {
            anyhow::bail!(
                "--no-manifest-capabilities cannot suppress required capabilities in a \
                 conforming formatVersion-3 build (required by these encoder settings: {})",
                manifest_capabilities.join(", ")
            );
        }
        warn!(
            "--no-manifest-capabilities is deprecated and has no effect; this build uses \
             no required-to-understand encoder features"
        );
    }
    if !manifest_capabilities.is_empty() {
        info!(
            "Manifest capabilities (required-to-understand): {}",
            manifest_capabilities.join(", ")
        );
    }
    let mut writer = stt_core::PackWriter::create(&out_dir, pack_ordering, pack_target_bytes)?
        // TB-10: publish the shared candidate instants so a client can enumerate
        // adaptive window keys instead of discovering them from the directory.
        // Empty (and therefore absent from the manifest) on every other build.
        .with_adaptive_boundaries(adaptive_boundaries.iter().map(|&b| b as i64).collect())
        .with_zstd_level(args.zstd_level)
        .with_capabilities(manifest_capabilities)
        .with_measured_ordering(measured_ordering)
        .with_ordering_workload(ordering_workload)
        .with_memory_budget(args.pack_memory_budget.saturating_mul(1024 * 1024))
        // Every tile-writing path below (streaming, batched, LOD, summary) pulls
        // its `EncoderConfig` back off the writer, so the resolved flags travel
        // with the sink instead of through process-wide state.
        .with_encoder_config(encoder_config);
    writer = if args.single_directory {
        writer.with_paging(None)
    } else {
        writer.with_adaptive_paging(args.page_entries, args.paged_directory_min_entries)
    };
    if args.pack_memory_budget > 0 {
        info!(
            "Pack-writer memory budget: {} MiB (payloads beyond it spill to a temp \
             file in the output dir; output bytes are identical at any budget)",
            args.pack_memory_budget
        );
    } else {
        info!("Pack-writer memory budget: unlimited (--pack-memory-budget 0; all payloads in RAM)");
    }
    // The independent layer-frame version rides the writer config: the tile sinks
    // (`TileWriter`/`LodTileWriter`/`write_tiles_parallel` in stt_build::tiler)
    // encode every payload with the pinned frame version + template collector.
    if args.zstd_level != stt_core::compression::ZSTD_LEVEL {
        info!(
            "zstd level {} (publish tuning; default {})",
            args.zstd_level,
            stt_core::compression::ZSTD_LEVEL
        );
    }
    if args.single_directory {
        info!("Single whole-load directory (paging opted out via --single-directory)");
    } else {
        info!(
            "Adaptive directory: single frame below {} entries; otherwise paged \
             into leaves of ≤{} entries",
            args.paged_directory_min_entries, args.page_entries
        );
    }

    let tile_count = if !temporal_lod.is_empty() {
        // --temporal-lod path: emit base tiles + lossless coarse-bucket tiers.
        // Written via write_tiles_parallel with a per-tile bucket tag so each
        // directory entry carries its temporal_bucket_ms. The streaming
        // pipeline doesn't (yet) do LOD aggregation — that's a follow-up — so
        // this path goes through the in-memory builder regardless of
        // --streaming.
        if args.streaming {
            warn!("--streaming ignored when --temporal-lod is set (in-memory pipeline used)");
        }
        let tiles = tiler::generate_tiles_with_lod(&features, &tile_config, args.workers)?;
        info!(
            "Generated {} tiles (base + lossless coarse-bucket LOD tiers)",
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
        // Parallel encode (bounded by --workers), deterministic ordered write.
        let keep: Vec<(&tiler::GeneratedTile, Option<u64>)> = tiles
            .iter()
            .filter(|tagged| tagged.tile.feature_count() >= min_features)
            .map(|tagged| (&tagged.tile, tagged.temporal_bucket_ms))
            .collect();
        pb.inc((tiles.len() - keep.len()) as u64);
        tiler::write_tiles_parallel_with_base_bucket(
            &mut writer,
            &keep,
            args.workers,
            temporal_bucket_ms,
            || pb.inc(1),
        )?;
        let written = keep.len();
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
        let stats =
            tiler::generate_tiles_streaming(&features, &tile_config, &mut writer, args.workers)?;
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
        // Parallel encode (bounded by --workers), deterministic ordered write.
        let keep: Vec<(&tiler::GeneratedTile, Option<u64>)> = tiles
            .iter()
            .filter(|t| t.feature_count() >= min_features)
            .map(|t| (t, None))
            .collect();
        pb.inc((tiles.len() - keep.len()) as u64);
        tiler::write_tiles_parallel_with_base_bucket(
            &mut writer,
            &keep,
            args.workers,
            temporal_bucket_ms,
            || pb.inc(1),
        )?;
        let written = keep.len();

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
            SummaryTierScheme::Quadbin => stt_core::metadata::SummaryScheme::Quadbin,
        };
        let sm_min = args.summary_min_zoom.unwrap_or(args.min_zoom);
        let sm_max = args
            .summary_max_zoom
            .unwrap_or_else(|| (args.min_zoom + 4).min(args.max_zoom));
        if sm_min > sm_max {
            anyhow::bail!("--summary-min-zoom ({sm_min}) > --summary-max-zoom ({sm_max})");
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
    let mut metadata = stt_core::metadata::Metadata::new(
        args.name.clone().unwrap_or_else(|| source.default_name()),
    )
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
    // SH-2: additive vertical extent. Omitted (and byte-invisible) for every
    // purely 2D dataset, which is every archive that has no 3-element geometry
    // position and no folded elevation column.
    .with_z_range(profile.z_range)
    .with_temporal_bucket_ms(temporal_bucket_ms);
    // SH-2: record WHICH quantity `bounds` was taken from. This is what lets a
    // reader — and `stt-validate` check 13 — distinguish an attested vertex bbox
    // from a legacy centroid one: on `vertex`, a bbox that does not contain its
    // own decoded vertices is an ERROR; on an archive with no such key (the
    // whole pre-R1 fleet) the identical finding is a warning naming the rebuild.
    // Stamped in BOTH modes on purpose — a `--bounds-mode centroid` rollback
    // build is then self-describing rather than indistinguishable from an
    // archive that predates the flag.
    metadata.properties.insert(
        input::BOUNDS_MODE_PROPERTY.to_string(),
        bounds_mode.as_manifest_value().to_string(),
    );
    // BLOCKER A: the sibling attestation. Records whether the wire `id` column
    // is a dataset-wide key, which is what licenses `stt-validate` check 12 to
    // compare `distinct_feature_count` against the ids it decodes. Stamped in
    // BOTH directions on purpose, exactly like `bounds_mode`: a build that
    // considered the question and answered "local" is then distinguishable from
    // one that predates the question.
    //
    // Byte note: this adds one `metadata.properties` entry to every build, so
    // manifest bytes move. It rides the SAME rebuild window (R1) as the
    // `bounds_mode` stamp above and the in-flight v2 → v3 break — one window,
    // one republish (design principle P6). No pack payload byte moves.
    {
        // Derived from the writer's own id CONSTRUCTION, not from an operator
        // assertion: `--no-clip` is what decides whether a duration line keeps
        // one anchor hash or is cut into per-segment ids, so the flag has to
        // reach the proof.
        let report = input::feature_id_report(
            &features,
            input::FeatureIdOptions {
                clip_trajectories: !args.no_clip,
            },
        );
        let (scope, message) = resolve_feature_id_scope(args.feature_id_scope, &report.attestation);
        if args.feature_id_scope == FeatureIdScopeArg::Global && !report.attestation.is_distinct() {
            warn!("{message}");
        } else {
            info!("{message}");
        }
        metadata.properties.insert(
            stt_core::metadata::FEATURE_ID_SCOPE_PROPERTY.to_string(),
            scope.to_string(),
        );
        // The FACT beside the assertion. `stt-validate` keys the basis on this
        // when no explicit scope is asserted, and its findings quote it so the
        // report describes the mechanism this archive actually used rather than
        // the point-archive one.
        metadata.properties.insert(
            stt_core::metadata::FEATURE_ID_CONSTRUCTION_PROPERTY.to_string(),
            report.construction.as_manifest_value().to_string(),
        );
    }
    if let Some(tier) = summary_tier_descriptor {
        metadata = metadata.with_summary_tier(tier);
    }
    if !temporal_lod.is_empty() {
        metadata = metadata
            .with_temporal_lod(temporal_lod.clone())
            .with_context(|| "temporal LOD validation failed")?;
    }
    // DT-1 declaration for DT-2's assignment. The must-understand capability is
    // added alongside it; `validate_partition_capability` refuses the pair if
    // they ever drift apart.
    if home_zoom_declared {
        metadata.partition = Some(stt_core::metadata::Partition::HomeZoom);
        // Re-derive the declared set here rather than borrowing the moved vec:
        // the check is about what the MANIFEST will carry.
        let declared = vec![stt_core::metadata::CAPABILITY_ADDITIVE_PARTITION.to_string()];
        stt_core::metadata::validate_partition_capability(metadata.partition, &declared)
            .with_context(|| "DT-2 home-zoom partition declaration")?;
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
    // Style hints need the loaded feature slice (like --temporal-lod), so only
    // the in-memory pipeline computes them. The cheap signals — layer_hint +
    // suggested playback — are baked on EVERY non-streaming build so view-time
    // layer inference works without opting in; `--style-hints` additionally
    // bakes the expensive per-property percentile/cardinality profile.
    if args.streaming {
        if args.style_hints {
            warn!(
                "--style-hints ignored with --streaming: the style-hints profiler \
                 needs the in-memory pipeline. Re-run without --streaming to bake hints."
            );
        }
        if args.derived_playback_params {
            warn!(
                "--derived-playback-params ignored with --streaming: the derived window \
                 needs the in-memory pipeline's feature slice and payload total. Re-run \
                 without --streaming to bake them."
            );
        }
        // A streaming build carries no in-memory feature slice, so it emits
        // neither the layer hint nor the property profile (documented limitation).
    } else if let Some(hints) = stt_build::style_hints::compute_style_hints_with(
        &features,
        &time_range,
        temporal_bucket_ms,
        args.style_hints, // full percentile/cardinality profile only when requested
        // BH-10, flag-gated: OFF leaves this at PlaybackDerivation::default(),
        // which reproduces the legacy hints byte for byte — so no manifest in
        // the fleet moves until a rebuild deliberately asks for the new values.
        // The payload total is the writer's running sum: every tile (base,
        // pyramid, LOD tier and summary tier) has already been handed to it by
        // this point, so it is the whole archive's decoded byte weight.
        if args.derived_playback_params {
            stt_build::style_hints::DerivedPlaybackParams {
                total_payload_bytes: Some(writer.payload_bytes()),
                refit: true,
            }
        } else {
            stt_build::style_hints::DerivedPlaybackParams::default()
        },
    ) {
        info!(
            "Style hints: {} properties profiled, layer hint {}, suggested playback {}, \
             suggested window {}",
            hints.properties.len(),
            hints.layer_hint.as_deref().unwrap_or("(none)"),
            hints
                .suggested_playback_seconds
                .map(|s| format!("{s}s"))
                .unwrap_or_else(|| "(none)".to_string()),
            hints
                .suggested_time_window_ms
                .map(|w| format!("{w}ms"))
                .unwrap_or_else(|| "(none)".to_string()),
        );
        metadata = metadata.with_style_hints(hints);
    }

    // Record the DISTINCT source-feature count (pre-placement) so downstream
    // "N features" badges don't have to sum the index-weighted per-tile total,
    // which double-counts every feature that spans tiles / pyramid levels.
    metadata.distinct_feature_count = Some(features.len() as u64);

    // SH-1, flag-gated OFF: the semantic content fingerprint. Adding a manifest
    // key moves manifest bytes, so nothing in the fleet changes until a build
    // asks for it — which is what keeps the golden pins still (they move
    // exactly once, at TB-14, inside rebuild window R1).
    //
    // The tolerances come from `fingerprint_levers`, snapshotted off the
    // RESOLVED encoder config above.
    if args.content_fingerprint {
        if args.streaming {
            warn!(
                "--content-fingerprint ignored with --streaming: the fingerprint is computed \
                 from the in-memory feature set before tiling. Re-run without --streaming."
            );
        } else {
            let (quantize_coords_m, attr_precisions, elevation_column) = fingerprint_levers;
            let fingerprint = input::content_fingerprint(
                &features,
                &input::FingerprintOptions {
                    quantize_coords_m,
                    attr_precisions,
                    elevation_column: (!elevation_column.is_empty())
                        .then_some(elevation_column.as_str()),
                },
            )?;
            info!(
                "Content fingerprint v{}: bbox [{:.5}, {:.5}, {:.5}, {:.5}], z {}, {} distinct \
                 features, {} numeric + {} categorical columns, coord tolerance {} deg",
                fingerprint.version,
                fingerprint.bbox[0],
                fingerprint.bbox[1],
                fingerprint.bbox[2],
                fingerprint.bbox[3],
                // Printed because `--point-elevation-column` on a NON-point
                // layer must show "(flat)" here: the fold never happens, so a
                // vertical claim would be invented for an archive with no 3D
                // geometry — and the named column stays in `numeric_ranges`,
                // where it remains checkable.
                fingerprint
                    .z_range
                    .map(|[lo, hi]| format!("[{lo}, {hi}]"))
                    .unwrap_or_else(|| "(flat)".to_string()),
                fingerprint.distinct_feature_count,
                fingerprint.numeric_ranges.len(),
                fingerprint.categorical_cardinality.len(),
                fingerprint.coord_tolerance_deg,
            );
            metadata = metadata.with_content_fingerprint(fingerprint);
        }
    }

    // TB-12: the one capability that cannot be derived from settings. Per-feature
    // triangle emission only owes a declaration when a layer actually MIXED empty
    // and baked lists — a polygon layer whose every feature needs baking (and any
    // dataset with no polygons at all) emits the incumbent bytes and must not
    // gratuitously lock out readers. So it is declared from what the encoder
    // OBSERVED, here, after tiling has joined and before finalize writes the
    // manifest.
    if tile_config
        .partial_triangles_observed
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        writer.declare_capability(stt_core::pack::CAPABILITY_TRIANGLES_PARTIAL);
        info!(
            "Manifest capability (observed): {} — polygon layers bake triangles only for \
             hole/multi-part features; readers earcut the rest at decode. \
             Use --pre-tessellate to bake every feature instead.",
            stt_core::pack::CAPABILITY_TRIANGLES_PARTIAL
        );
    }

    let manifest = writer.finalize(&metadata)?;

    // Step 5b: the STAC Item sidecar. Written from the FINALIZED manifest, not
    // from `metadata`: finalize derives `tile_count`/`feature_count` from the
    // directory it just wrote, so an Item built from the pre-finalize metadata
    // would publish counts the dataset does not have.
    if args.stac {
        let stac_path = out_dir.join("stac.json");
        std::fs::write(
            &stac_path,
            stt_build::stac::stac_item_bytes(&manifest, &out_dir),
        )?;
        info!(
            "STAC Item written to {} (asset href is relative: ./manifest.json)",
            stac_path.display()
        );
    }

    // Step 6: Write metadata JSON if requested
    if let Some(metadata_path) = args.metadata_output {
        info!("Writing metadata JSON to {}...", metadata_path.display());
        // The packed dataset is addressed by its manifest, so the dataset
        // "filename" is `<dir>/manifest.json`.
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

    // `--target-size`: close the loop the publisher used to close by eye. The
    // whole DIRECTORY is measured, not just the packs — manifest, directory
    // pages and packs are all bytes a publisher pays to host.
    if let Some(target) = target_bytes {
        report_target_size(&out_dir, target);
    }

    Ok(())
}

/// Log the built archive's real byte size against the `--target-size` budget.
///
/// The comparison is the honest one: the whole output directory (manifest +
/// directory pages + packs), which is what a publisher uploads. It is a REPORT,
/// never a gate — the build has already written correct output, and a build
/// that deleted its own result for missing a projection would be the thinning
/// failure in a different costume. `stt-optimize recommend --fail-if-over-target`
/// is the CI gate for this, and it runs BEFORE a build rather than after.
fn report_target_size(out_dir: &std::path::Path, target_bytes: u64) {
    fn dir_bytes(dir: &std::path::Path) -> std::io::Result<u64> {
        let mut total = 0u64;
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                total += dir_bytes(&path)?;
            } else {
                total += std::fs::metadata(&path)?.len();
            }
        }
        Ok(total)
    }
    let built = match dir_bytes(out_dir) {
        Ok(bytes) => bytes,
        Err(e) => {
            warn!(
                "--target-size: could not measure {}: {e}",
                out_dir.display()
            );
            return;
        }
    };
    let pct = if target_bytes > 0 {
        built as f64 / target_bytes as f64 * 100.0
    } else {
        f64::INFINITY
    };
    if built <= target_bytes {
        info!(
            "--target-size: BUILT {built} B vs target {target_bytes} B ({pct:.1}% of budget, \
             {} B under)",
            target_bytes - built
        );
    } else {
        warn!(
            "--target-size: BUILT {built} B vs target {target_bytes} B ({pct:.1}% of budget, \
             {} B OVER). The projection extrapolates from a sample, so a miss is an \
             extrapolation error, not a dropped feature — every feature you gave this build is \
             in the archive. Re-run `stt-optimize recommend --target-size` on the built dataset's \
             source to see the shadow prices, or widen the budget.",
            built - target_bytes
        );
    }
}

fn input_strictness(args: &Args) -> (input::InputStrictness, input::InputStrictness) {
    let time = if args.salvage_invalid_times {
        input::InputStrictness::Warn
    } else {
        input::InputStrictness::Strict
    };
    let geometry = if args.salvage_invalid_geometry {
        input::InputStrictness::Warn
    } else {
        input::InputStrictness::Strict
    };
    (time, geometry)
}

/// Resolve the `-o/--output` value into a packed-dataset directory.
///
/// The packed format is a directory tree (`manifest.json` + `index/` +
/// `packs/`), never a single file. For convenience a path ending in `.stt` has
/// that extension stripped, so `-o foo.stt` -> `foo/`. Any other path is used
/// verbatim as the dataset directory.
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
            .as_ref()?
            .get(name)
            .and_then(|v| v.as_f64())
    }
    fn extract_str(f: &input::ParsedFeature, name: &str) -> Option<String> {
        f.shared_properties
            .as_ref()?
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
        let idx = ((values.len() as f64 * 0.95).floor() as usize).min(values.len() - 1);
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
///
/// `basic` (bare `--auto`) fills in the zoom range + temporal bucket only.
/// `encode` additionally applies the advisors' NON-LOSSY byte-level levers
/// (`--publish`-equivalent zstd level, `--blob-ordering`, `--pack-size`).
/// LOSSY advice (quantization, budgets) is never applied in either mode —
/// it is warn-logged as "suggested, not applied" — and semantic levers
/// (`--temporal-lod`, `--adaptive-temporal`, `--summary-tier`,
/// `--min-zoom-field`, …) stay suggestion-only too.
///
/// # Budget mode (`target_bytes = Some(..)`, i.e. `--target-size`)
///
/// The analyzer additionally solves for a recipe that fits the budget over the
/// REVERSIBLE lever set Θ₀ and returns it as `rec.budget`. Its verdict is
/// authoritative over the advisors on the flags it owns
/// ([`BUDGET_GOVERNED_FLAGS`](stt_optimize::budget_solver::BUDGET_GOVERNED_FLAGS)),
/// because a command that carried the advisor's `--publish` while the budget
/// was projected at level 3 would silently miss the number the user was shown.
///
/// ⚠️ **The one policy change, and its bounds.** Budget mode is the ONLY mode
/// in which `--temporal-lod` is auto-applied. Outside it the flag keeps hitting
/// the blanket "semantic lever, never auto-applied" arm below. The override is
/// justified because in budget mode a tier is not an unpriced semantic guess:
/// it is LOSSLESS (every feature is replicated into the coarse bucket, none is
/// dropped), ADDITIVE (its byte cost was computed against the budget's
/// remainder before it was proposed — the §12.4 gap this closes), M3 places it
/// in the auto-searched feasible set, it is logged with its measured byte
/// price, and an explicit `--temporal-lod` still wins.
///
/// What budget mode does NOT change: no lossy lever is ever applied (shadow
/// prices take the same loud warn-path lossy advice takes), and a
/// playback-caveated `spatial` blob ordering stays suggestion-only however
/// tight the budget gets.
fn apply_auto_recommendations(
    matches: &ArgMatches,
    args: &mut Args,
    mode: AutoMode,
    target_bytes: Option<u64>,
) -> Result<()> {
    info!("--auto ({mode:?}): analyzing input for build recommendations...");
    let path = args.input.clone().ok_or_else(|| {
        if target_bytes.is_some() {
            anyhow::anyhow!(
                "--target-size is not supported with --postgres/--duckdb: it implies --auto, and \
                 the analyzer reads a GeoParquet FILE (it samples and trial-encodes the source to \
                 measure the budget). Export the query to GeoParquet first, then build from that \
                 with --target-size."
            )
        } else {
            anyhow::anyhow!("--auto is not supported with --postgres/--duckdb (the analyzer reads a GeoParquet file)")
        }
    })?;
    let source = stt_optimize::DataSource::GeoParquet {
        path,
        time_field: args.time_field.clone(),
        time_format: args.time_format.as_str().to_string(),
    };
    if let Some(bytes) = target_bytes {
        info!("--target-size {bytes} B: solving the recipe against the budget (reversible levers only)");
    }
    let rec = stt_optimize::recommend_with(
        &source,
        &stt_optimize::RecommendOptions {
            target_size: target_bytes,
        },
    )
    .context("stt-optimize analyzer failed")?;

    let user_set =
        |name: &str| matches!(matches.value_source(name), Some(ValueSource::CommandLine));

    // The scalar recipe: the analysis recommendation, overridden by the budget
    // solver's own choices where it made any. Mirrors `recommend::to_command`'s
    // `command_scalars` so the pasteable command and the in-process build can
    // never disagree about what was chosen.
    let (min_zoom, max_zoom, temporal_bucket_ms) = budget_scalars(&rec);

    if !user_set("min_zoom") {
        info!("  min-zoom: {} (auto)", min_zoom);
        args.min_zoom = min_zoom;
    }
    if !user_set("max_zoom") {
        info!("  max-zoom: {} (auto)", max_zoom);
        args.max_zoom = max_zoom;
    }
    // rec.compression is NOT folded in: the packed format is zstd-only, so
    // an analyzer recommendation of gzip/none would just fail validation.
    if !user_set("temporal_bucket") && temporal_bucket_ms > 0 {
        if temporal_bucket_ms == rec.temporal_bucket_ms {
            info!("  temporal-bucket: {} (auto)", rec.temporal_bucket_human);
        } else {
            info!(
                "  temporal-bucket: {} ms (budget: widened from the recommended {} to fit the \
                 target — lossless, the same features cut into fewer tiles)",
                temporal_bucket_ms, rec.temporal_bucket_human
            );
        }
        // Fold in the ms form: the human string ("30 days") is for logs only
        // and is not always `parse_duration`-compatible.
        args.temporal_bucket = format!("{}ms", temporal_bucket_ms);
    } else if user_set("temporal_bucket") && temporal_bucket_ms != rec.temporal_bucket_ms {
        warn!(
            "  --temporal-bucket {} was passed explicitly, so the budget's {} ms width was NOT \
             applied — the projected size assumed the budget's width",
            args.temporal_bucket, temporal_bucket_ms
        );
    }

    info!(
        "  confidence: {}% — {} reasons",
        rec.confidence,
        rec.explanations.len()
    );
    for line in &rec.explanations {
        info!("    - {}", line);
    }

    // Flags the budget owns once a target size is set: the advisor's verdict on
    // these is SUPERSEDED, not merged. Empty without a budget, so an
    // unbudgeted run is byte-for-byte what it always was.
    let superseded: &[&str] = if rec.budget.is_some() {
        stt_optimize::budget_solver::BUDGET_GOVERNED_FLAGS
    } else {
        &[]
    };

    // Advisor suggestions. Only `--auto encode` applies anything, and ONLY
    // the non-lossy byte-level levers, and only for flags the user did not
    // set explicitly. Everything else is surfaced as a suggestion.
    let mut applied: Vec<String> = Vec::new();
    for advice in &rec.advice {
        let suggestion = match &advice.value {
            Some(v) => format!("{} {}", advice.flag, v),
            None => advice.flag.clone(),
        };
        if advice.lossy {
            // Lossy levers (quantization, budgets) discard or degrade data —
            // NEVER auto-applied (no-thinning principle). Loud by design.
            warn!("suggested, not applied: {suggestion} — {}", advice.why);
            continue;
        }
        if superseded.contains(&advice.flag.as_str()) {
            info!(
                "  {suggestion}: superseded by the --target-size solver's own choice for this flag"
            );
            continue;
        }
        if mode != AutoMode::Encode {
            info!(
                "  suggested, not applied (--auto encode applies byte-level levers): {suggestion}"
            );
            continue;
        }
        match advice.flag.as_str() {
            // --publish bundles zstd 19 over the already-paged-by-default
            // directory; apply the equivalent directly (set the level, leave
            // the paging default) rather than flipping the publish bool.
            "--publish" => {
                if user_set("zstd_level") || user_set("publish") {
                    info!("  --publish advice skipped (explicit --zstd-level/--publish wins)");
                } else {
                    args.zstd_level = 19;
                    applied.push("zstd-level 19 (--publish advice)".to_string());
                }
            }
            "--zstd-level" => match advice.value.as_deref().map(str::parse::<i32>) {
                Some(Ok(level)) if !user_set("zstd_level") => {
                    args.zstd_level = level;
                    applied.push(format!("zstd-level {level}"));
                }
                Some(Ok(_)) => info!("  --zstd-level advice skipped (explicit flag wins)"),
                _ => warn!("  --zstd-level advice has no usable value; skipped"),
            },
            "--blob-ordering" => match &advice.value {
                Some(ordering) if !user_set("blob_ordering") => {
                    args.blob_ordering = ordering.clone();
                    applied.push(format!("blob-ordering {ordering}"));
                }
                Some(_) => info!("  --blob-ordering advice skipped (explicit flag wins)"),
                None => warn!("  --blob-ordering advice has no value; skipped"),
            },
            "--pack-size" => match advice.value.as_deref().map(str::parse::<u64>) {
                Some(Ok(mib)) if !user_set("pack_size") => {
                    args.pack_size = mib;
                    applied.push(format!("pack-size {mib}"));
                }
                Some(Ok(_)) => info!("  --pack-size advice skipped (explicit flag wins)"),
                _ => warn!("  --pack-size advice has no usable value; skipped"),
            },
            // Semantic levers (--temporal-lod, --adaptive-temporal,
            // --summary-tier, --min-zoom-field, …) change what the archive
            // MEANS, not just its bytes — never auto-applied.
            //
            // ⚠️ `--temporal-lod` reaches this arm in every mode EXCEPT budget
            // mode, where it arrives instead as a priced `ChosenLever` through
            // `apply_budget_levers` (and is filtered out above as superseded).
            _ => info!(
                "  suggested, not applied (semantic lever): {suggestion} — {}",
                advice.why
            ),
        }
    }

    // The budget's own recipe, applied last so its levers are the final word on
    // the flags it owns.
    if let Some(budget) = &rec.budget {
        apply_budget_levers(args, budget, &user_set, &mut applied)?;
    }

    if mode == AutoMode::Encode {
        if applied.is_empty() {
            info!("--auto encode: no byte-level levers auto-applied (explicit flags win, or no applicable advice)");
        } else {
            info!("--auto encode applied: {}", applied.join(", "));
        }
    }
    Ok(())
}

/// The scalar recipe `--auto` should fold in: the analysis recommendation,
/// overridden by the budget solver's own choices where it made any.
///
/// The solver only emits `--max-zoom` / `--temporal-bucket` when it actually
/// MOVED them, so an unbudgeted run and a budget that needed no distortion
/// produce identical args. `suggestion_only` levers are skipped here exactly as
/// `recommend::to_command` skips them.
fn budget_scalars(rec: &stt_optimize::Recommendations) -> (u8, u8, u64) {
    let mut min_zoom = rec.min_zoom;
    let mut max_zoom = rec.max_zoom;
    let mut bucket_ms = rec.temporal_bucket_ms;
    let Some(budget) = &rec.budget else {
        return (min_zoom, max_zoom, bucket_ms);
    };
    for lever in budget.chosen.iter().filter(|l| !l.suggestion_only) {
        let Some(value) = lever.value.as_deref() else {
            continue;
        };
        match lever.flag.as_str() {
            "--min-zoom" => min_zoom = value.parse().unwrap_or(min_zoom),
            "--max-zoom" => max_zoom = value.parse().unwrap_or(max_zoom),
            "--temporal-bucket" => bucket_ms = value.parse().unwrap_or(bucket_ms),
            _ => {}
        }
    }
    (min_zoom, max_zoom, bucket_ms)
}

/// Fold the budget solver's NON-SCALAR chosen levers into `args`, each logged
/// with the measured byte price that bought it.
///
/// Every lever here is reversible and lossless by construction — a
/// [`ChosenLever`](stt_optimize::ChosenLever) has no `lossy` field, so the type
/// system, not a filter, is what keeps quantization out of this function. The
/// two filters that remain are the ones a type cannot express:
///
/// * `suggestion_only` — non-lossy but carrying a tradeoff a human must decide.
///   ⚠️ The recorded case is a playback-caveated `spatial` blob ordering:
///   `blobOrdering: spatial` silently breaks time-playback buffering (empty
///   buffered ranges → stalls), and BUDGET PRESSURE IS NOT A REASON TO SHIP A
///   BROKEN DEMO. It stays suggestion-only in every mode.
/// * explicit user flags, which always win.
fn apply_budget_levers(
    args: &mut Args,
    budget: &stt_optimize::BudgetReport,
    user_set: &dyn Fn(&str) -> bool,
    applied: &mut Vec<String>,
) -> Result<()> {
    info!("--target-size: {}", budget.headline());
    if !budget.feasible {
        warn!(
            "--target-size {} B is NOT reachable with reversible levers: the floor is {} B. \
             Building at the floor recipe — NOTHING has been dropped, sampled or aggregated to \
             close the gap.",
            budget.target_bytes, budget.floor_bytes
        );
    }
    for note in &budget.notes {
        info!("  note: {note}");
    }

    for lever in &budget.chosen {
        let shown = match &lever.value {
            Some(v) => format!("{} {}", lever.flag, v),
            None => lever.flag.clone(),
        };
        let price = match lever.delta_bytes {
            Some(d) => format!(" [{d:+} B measured]"),
            None => String::new(),
        };
        if lever.suggestion_only {
            // Non-lossy, but the tradeoff in its `why` needs a human. Budget
            // pressure does not promote it.
            warn!(
                "  budget lever NOT applied (needs a decision): {shown} — {}",
                lever.why
            );
            continue;
        }
        match lever.flag.as_str() {
            // Scalars, already folded by `budget_scalars` above.
            "--min-zoom" | "--max-zoom" | "--temporal-bucket" => {}
            "--publish" => {
                if user_set("zstd_level") || user_set("publish") {
                    info!("  budget --publish skipped (explicit --zstd-level/--publish wins)");
                } else {
                    args.zstd_level = 19;
                    applied.push(format!("zstd-level 19 (budget){price}"));
                }
            }
            "--zstd-level" => match lever.value.as_deref().map(str::parse::<i32>) {
                Some(Ok(level)) if !user_set("zstd_level") => {
                    args.zstd_level = level;
                    applied.push(format!("zstd-level {level} (budget){price}"));
                }
                Some(Ok(_)) => info!("  budget --zstd-level skipped (explicit flag wins)"),
                _ => warn!("  budget --zstd-level has no usable value; skipped"),
            },
            "--blob-ordering" => match &lever.value {
                Some(ordering) if !user_set("blob_ordering") => {
                    args.blob_ordering = ordering.clone();
                    applied.push(format!("blob-ordering {ordering} (budget)"));
                }
                Some(_) => info!("  budget --blob-ordering skipped (explicit flag wins)"),
                None => warn!("  budget --blob-ordering has no value; skipped"),
            },
            "--pack-size" => match lever.value.as_deref().map(str::parse::<u64>) {
                Some(Ok(mib)) if !user_set("pack_size") => {
                    args.pack_size = mib;
                    applied.push(format!("pack-size {mib} (budget)"));
                }
                Some(Ok(_)) => info!("  budget --pack-size skipped (explicit flag wins)"),
                _ => warn!("  budget --pack-size has no usable value; skipped"),
            },
            // ⚠️ THE ONE POLICY CHANGE — see this function's caller. Auto-applied
            // ONLY here, only with a `--target-size`, only when the tiers are
            // legal against the bucket this build will actually use, and always
            // yielding to an explicit `--temporal-lod`.
            "--temporal-lod" => {
                let Some(value) = lever.value.as_deref() else {
                    warn!("  budget --temporal-lod has no value; skipped");
                    continue;
                };
                if user_set("temporal_lod") {
                    info!("  budget --temporal-lod {value} skipped (explicit flag wins)");
                    continue;
                }
                if let Some(reason) = temporal_lod_is_illegal_here(args, value) {
                    warn!(
                        "  budget --temporal-lod {value} NOT applied: {reason}. The projected \
                         size assumed the tier, so this build will land under it."
                    );
                    continue;
                }
                args.temporal_lod = Some(value.to_string());
                applied.push(format!(
                    "temporal-lod {value} (budget, LOSSLESS + ADDITIVE){price}"
                ));
                info!(
                    "  budget --temporal-lod {value}{price}: a coarse-bucket REPLICA of every \
                     feature, priced against the budget's remainder before it was proposed. \
                     Nothing is dropped — {}",
                    lever.why
                );
            }
            other => info!("  budget lever {other} has no build flag to apply; reported only"),
        }
    }

    // Shadow prices: what a human could choose to spend quality on. They take
    // the SAME loud warn-path lossy advice takes, and nothing reads them.
    for price in &budget.shadow_prices {
        let shown = match &price.value {
            Some(v) => format!("{} {}", price.flag, v),
            None => price.flag.clone(),
        };
        warn!(
            "suggested, not applied (LOSSY shadow price, +{} B if you add it by hand): {shown} — {}",
            price.marginal_bytes, price.why
        );
    }
    Ok(())
}

/// Why the budget's `--temporal-lod` spec cannot be applied to THIS build, or
/// `None` when it can.
///
/// The solver prices tiers as multiples of the bucket IT chose. If the user
/// pinned a different `--temporal-bucket`, or asked for adaptive windows, the
/// spec is no longer legal (`validate_lod` requires every tier to be a strict
/// multiple of the base bucket) — and a build that aborts on a flag the user
/// never typed is the worst possible outcome of an auto-tuner.
fn temporal_lod_is_illegal_here(args: &Args, spec: &str) -> Option<String> {
    if args.adaptive_temporal.is_some() {
        return Some(
            "--adaptive-temporal replaces fixed buckets, so a bucket pyramid has no base"
                .to_string(),
        );
    }
    let base = parse_duration(&args.temporal_bucket).ok()?;
    if base == 0 {
        return Some("the effective --temporal-bucket is 0".to_string());
    }
    let levels = parse_temporal_lod(spec, args.max_zoom).ok()?;
    let illegal = levels
        .iter()
        .find(|l| l.bucket_ms == 0 || l.bucket_ms <= base || !l.bucket_ms.is_multiple_of(base));
    illegal.map(|l| {
        format!(
            "tier {} ms is not a strict multiple of the effective base bucket {} ms (an explicit \
             --temporal-bucket overrode the width the tiers were priced against)",
            l.bucket_ms, base
        )
    })
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

/// The packed format compresses per-blob with zstd (no shared dict, so the TS
/// reader can decode any blob in isolation). `zstd` is the only accepted value.
///
/// `gzip` and `none` are rejected loudly rather than ignored: `gzip` names a
/// codec no packed archive can contain, and silently accepting either would
/// produce a zstd archive that misreports what the caller asked for.
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

/// Parse `--ordering-workload`. `derived` (the default) weights the canonical
/// queries per dataset from its layer hint + bucket count; `legacy` pins the
/// pre-M4 scrub+pan weighting so an older layout can be reproduced exactly.
fn parse_ordering_workload(s: &str) -> Result<stt_core::ordering_sim::OrderingWorkloadMode> {
    use stt_core::ordering_sim::OrderingWorkloadMode;
    match s.trim().to_lowercase().as_str() {
        "derived" | "auto" => Ok(OrderingWorkloadMode::Derived),
        "legacy" => Ok(OrderingWorkloadMode::Legacy),
        other => anyhow::bail!(
            "Invalid --ordering-workload '{other}'. Expected 'derived' (default) or 'legacy'."
        ),
    }
}

/// Build the opt-in per-tile budget from the CLI flags, or `None` when neither
/// `--maximum-tile-bytes` nor `--maximum-tile-features` was passed (the default
/// "no thinning" behaviour — the budget is inert and the in-memory path is
/// byte-for-byte identical to before).
///
/// The scorer choice encodes feature #2: with `--drop-densest-as-needed` the
/// budget drops the densest features first (pure `GeometrySize`); without it the
/// budget still drops the LEAST-important features (a `Combined`
/// geometry+property score) rather than randomly.
fn build_tile_budget(args: &Args) -> Option<stt_core::budget::TileBudget> {
    build_options::build_tile_budget(
        args.maximum_tile_bytes,
        args.maximum_tile_features,
        args.drop_densest_as_needed,
    )
}

/// Resolve `--exclude` / `--include` / `--exclude-all` into an
/// [`stt_build::columnar::AttributeFilter`], validating mutual exclusivity and
/// guarding columns that other features still need.
///
/// Errors when:
/// * more than one of exclude/include/exclude-all is given, or
/// * a property the build still needs (`--heatmap-weight`, `--heatmap-class`,
///   or any `--summary-columns` source column, or `--min-zoom-field`) would be
///   removed by the filter — refusing rather than silently breaking those
///   features.
fn build_attribute_filter(args: &Args) -> Result<stt_build::columnar::AttributeFilter> {
    // Guard columns other features depend on. A filter that would drop a
    // property the heatmap/summary/min-zoom passes read is almost certainly a
    // mistake — the shared builder errors rather than emit a quietly-broken
    // archive. Summary aggregation source columns are resolved here (stt-build
    // specific) and threaded into the shared filter builder as `required`.
    let mut required: Vec<String> = Vec::new();
    if let Some(w) = &args.heatmap_weight {
        required.push(w.clone());
    }
    if let Some(c) = &args.heatmap_class {
        required.push(c.clone());
    }
    if let Some(z) = &args.min_zoom_field {
        required.push(z.clone());
    }
    if let Some(z) = &args.max_zoom_field {
        required.push(z.clone());
    }
    for col in summary::parse_summary_columns(&args.summary_columns)? {
        if !col.name.is_empty() && col.name != "_count" {
            required.push(col.name.clone());
        }
    }

    build_options::build_attribute_filter(&args.exclude, &args.include, args.exclude_all, &required)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_build::columnar::AttributeFilter;

    /// Parse `Args` from a flag list, always supplying the required
    /// `--input`/`--output`. Lets the CLI-validation helpers be unit-tested
    /// without running a full build.
    fn args_from(extra: &[&str]) -> Args {
        let mut argv: Vec<&str> = vec!["stt-build", "-i", "in.parquet", "-o", "out"];
        argv.extend_from_slice(extra);
        Args::parse_from(argv)
    }

    #[test]
    fn no_budget_flags_yields_no_budget() {
        let args = args_from(&[]);
        assert!(build_tile_budget(&args).is_none());
    }

    #[test]
    fn input_parsing_is_strict_by_default_and_salvage_is_explicit() {
        let defaults = args_from(&[]);
        assert_eq!(
            input_strictness(&defaults),
            (
                input::InputStrictness::Strict,
                input::InputStrictness::Strict
            )
        );

        let salvage = args_from(&["--salvage-invalid-times", "--salvage-invalid-geometry"]);
        assert_eq!(
            input_strictness(&salvage),
            (input::InputStrictness::Warn, input::InputStrictness::Warn)
        );
    }

    #[test]
    fn strict_and_salvage_spellings_conflict() {
        let argv = [
            "stt-build",
            "-i",
            "in.parquet",
            "-o",
            "out",
            "--strict-times",
            "--salvage-invalid-times",
        ];
        assert!(Args::try_parse_from(argv).is_err());
    }

    #[test]
    fn maximum_tile_features_builds_budget() {
        let args = args_from(&["--maximum-tile-features", "100"]);
        let budget = build_tile_budget(&args).expect("budget present");
        assert_eq!(budget.max_feature_count, 100);
        // Unset byte cap => unbounded on that axis.
        assert_eq!(budget.max_uncompressed_size, usize::MAX);
    }

    #[test]
    fn maximum_tile_bytes_builds_budget() {
        let args = args_from(&["--maximum-tile-bytes", "50000"]);
        let budget = build_tile_budget(&args).expect("budget present");
        assert_eq!(budget.max_uncompressed_size, 50_000);
        assert_eq!(budget.max_feature_count, usize::MAX);
    }

    #[test]
    fn no_attribute_flags_keeps_all() {
        let args = args_from(&[]);
        let filter = build_attribute_filter(&args).unwrap();
        assert!(matches!(filter, AttributeFilter::KeepAll));
    }

    #[test]
    fn exclude_builds_exclude_filter() {
        let args = args_from(&["--exclude", "secret", "--exclude", "debug"]);
        let filter = build_attribute_filter(&args).unwrap();
        assert!(!filter.keeps("secret"));
        assert!(!filter.keeps("debug"));
        assert!(filter.keeps("speed"));
    }

    #[test]
    fn include_builds_include_filter() {
        let args = args_from(&["--include", "speed"]);
        let filter = build_attribute_filter(&args).unwrap();
        assert!(filter.keeps("speed"));
        assert!(!filter.keeps("anything_else"));
    }

    #[test]
    fn exclude_and_include_together_is_an_error() {
        let args = args_from(&["--exclude", "a", "--include", "b"]);
        let err = build_attribute_filter(&args).unwrap_err().to_string();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    #[test]
    fn exclude_all_with_include_is_an_error() {
        let args = args_from(&["--exclude-all", "--include", "b"]);
        assert!(build_attribute_filter(&args).is_err());
    }

    #[test]
    fn filter_guards_heatmap_weight_column() {
        // Excluding the heatmap-weight column must error rather than silently
        // break the heatmap-domain pass.
        let args = args_from(&["--heatmap-weight", "magnitude", "--exclude", "magnitude"]);
        let err = build_attribute_filter(&args).unwrap_err().to_string();
        assert!(err.contains("magnitude"), "got: {err}");
    }

    #[test]
    fn filter_guards_summary_source_column() {
        // An --include that omits a summary source column must error.
        let args = args_from(&["--summary-columns", "depth:mean", "--include", "speed"]);
        let err = build_attribute_filter(&args).unwrap_err().to_string();
        assert!(err.contains("depth"), "got: {err}");
    }

    /// Build `Args` from a flag list WITHOUT an implicit `--input` (only the
    /// required `-o` is supplied), for exercising `resolve_source`'s backend
    /// selection. These assertions hold regardless of the postgres/duckdb
    /// cargo features (the mutual-exclusion + file/no-source paths run before
    /// any feature gate).
    fn args_src(extra: &[&str]) -> Args {
        let mut argv: Vec<&str> = vec!["stt-build", "-o", "out"];
        argv.extend_from_slice(extra);
        Args::parse_from(argv)
    }

    #[test]
    fn resolve_source_rejects_input_plus_database() {
        let err = resolve_source(&args_src(&[
            "--input",
            "x.parquet",
            "--duckdb",
            "d",
            "--table",
            "t",
        ]))
        .err()
        .expect("input + duckdb should conflict")
        .to_string();
        assert!(err.contains("not both"), "got: {err}");
    }

    #[test]
    fn resolve_source_rejects_postgres_plus_duckdb() {
        let err = resolve_source(&args_src(&[
            "--postgres",
            "postgresql://x",
            "--duckdb",
            "d",
            "--table",
            "t",
        ]))
        .err()
        .expect("postgres + duckdb should conflict")
        .to_string();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    #[test]
    fn resolve_source_file_input_ok() {
        let src =
            resolve_source(&args_src(&["--input", "x.parquet"])).expect("file input resolves");
        assert!(matches!(src, InputSource::File(_)));
    }

    #[test]
    fn resolve_source_requires_a_source() {
        let err = resolve_source(&args_src(&[]))
            .err()
            .expect("no source should error")
            .to_string();
        assert!(err.contains("no input"), "got: {err}");
    }

    #[test]
    fn auto_absent_parses_to_none() {
        assert_eq!(args_from(&[]).auto, None);
    }

    #[test]
    fn bare_auto_parses_to_basic() {
        // Plain `--auto` must keep its pre-tier meaning: basic (zoom + bucket).
        assert_eq!(args_from(&["--auto"]).auto, Some(AutoMode::Basic));
        assert_eq!(args_from(&["--auto", "basic"]).auto, Some(AutoMode::Basic));
    }

    #[test]
    fn auto_encode_parses_to_encode() {
        assert_eq!(
            args_from(&["--auto", "encode"]).auto,
            Some(AutoMode::Encode)
        );
        // `--auto=encode` (require_equals is false, but `=` still works).
        assert_eq!(args_from(&["--auto=encode"]).auto, Some(AutoMode::Encode));
    }

    #[test]
    fn bare_auto_does_not_swallow_the_next_flag() {
        let args = args_from(&["--auto", "--simplify"]);
        assert_eq!(args.auto, Some(AutoMode::Basic));
        assert!(args.simplify);
    }

    #[test]
    fn style_hints_flag_defaults_off_and_parses() {
        assert!(!args_from(&[]).style_hints);
        assert!(args_from(&["--style-hints"]).style_hints);
    }

    /// WM-3: `measured` is the build DEFAULT. No flag → the writer runs the
    /// simulator; every explicit choice still wins, including `auto` (which
    /// remains selectable and remains what `measured` degrades to on tiny
    /// inputs).
    #[test]
    fn blob_ordering_defaults_to_measured_and_explicit_flags_still_win() {
        let resolve = |argv: &[&str]| {
            let args = args_from(argv);
            let blob_arg = args.blob_ordering.trim().to_string();
            let measured = blob_arg.eq_ignore_ascii_case("measured");
            let concrete: Option<stt_core::BlobOrdering> = if measured {
                None
            } else {
                Some(blob_arg.parse().expect("explicit ordering parses"))
            };
            (measured, concrete)
        };
        assert_eq!(args_from(&[]).blob_ordering, "measured");
        assert_eq!(resolve(&[]), (true, None));
        assert_eq!(
            resolve(&["--blob-ordering", "auto"]),
            (false, Some(stt_core::BlobOrdering::Auto))
        );
        assert_eq!(
            resolve(&["--blob-ordering", "spatial"]),
            (false, Some(stt_core::BlobOrdering::SpatialMajor))
        );
        assert_eq!(
            resolve(&["--blob-ordering", "time-major"]),
            (false, Some(stt_core::BlobOrdering::TimeMajor))
        );
        assert_eq!(
            resolve(&["--blob-ordering", "hilbert3"]),
            (false, Some(stt_core::BlobOrdering::Hilbert3))
        );
        assert_eq!(
            resolve(&["--blob-ordering", "morton3"]),
            (false, Some(stt_core::BlobOrdering::Morton3))
        );
        // Case-insensitive, like the pre-flip parse.
        assert_eq!(resolve(&["--blob-ordering", "MEASURED"]), (true, None));
    }

    #[test]
    fn ordering_workload_defaults_to_derived_and_legacy_is_the_escape_hatch() {
        use stt_core::ordering_sim::OrderingWorkloadMode;
        assert_eq!(args_from(&[]).ordering_workload, "derived");
        assert_eq!(
            parse_ordering_workload(&args_from(&[]).ordering_workload).unwrap(),
            OrderingWorkloadMode::Derived
        );
        assert_eq!(
            parse_ordering_workload("legacy").unwrap(),
            OrderingWorkloadMode::Legacy
        );
        assert_eq!(
            parse_ordering_workload(" LEGACY ").unwrap(),
            OrderingWorkloadMode::Legacy
        );
        let err = parse_ordering_workload("playback-only")
            .unwrap_err()
            .to_string();
        assert!(err.contains("--ordering-workload"), "{err}");
    }

    #[test]
    fn auto_rejects_unknown_mode() {
        let argv = vec![
            "stt-build",
            "-i",
            "in.parquet",
            "-o",
            "out",
            "--auto",
            "bogus",
        ];
        assert!(Args::try_parse_from(argv).is_err());
    }

    // ------------------------------------------------------------------
    // MO-9: the `--target-size` handshake
    // ------------------------------------------------------------------

    use stt_optimize::{BudgetReport, ChosenLever, ShadowPrice};

    fn lever(flag: &str, value: Option<&str>, delta: Option<i64>) -> ChosenLever {
        ChosenLever {
            flag: flag.to_string(),
            value: value.map(str::to_string),
            why: format!("{flag}: synthetic rationale"),
            delta_bytes: delta,
            suggestion_only: false,
        }
    }

    fn shadow(flag: &str, value: Option<&str>, bytes: i64) -> ShadowPrice {
        ShadowPrice {
            flag: flag.to_string(),
            value: value.map(str::to_string),
            marginal_bytes: bytes,
            delta_frac: -0.2,
            stderr: 0.001,
            lossy: true,
            why: format!("{flag}: would shed {bytes} B at a quality cost"),
        }
    }

    fn budget_with(chosen: Vec<ChosenLever>, shadow_prices: Vec<ShadowPrice>) -> BudgetReport {
        BudgetReport {
            target_bytes: 1_000_000,
            projected_bytes: 900_000,
            projected_stderr: 1_000.0,
            feasible: true,
            within_noise: false,
            distortion: stt_optimize::DistortionClass::none(),
            chosen,
            floor_bytes: 700_000,
            floor_distortion: stt_optimize::DistortionClass::none(),
            shadow_prices,
            zstd_sweep: vec![3, 9, 19],
            classes_evaluated: 4,
            basis: stt_optimize::EstimateBasis::Measured,
            notes: Vec::new(),
        }
    }

    /// Apply a synthetic budget to freshly parsed args. `explicit` is the set of
    /// clap arg ids the caller is pretending the user typed.
    fn fold(args: &mut Args, budget: &BudgetReport, explicit: &[&str]) -> Vec<String> {
        let owned: Vec<String> = explicit.iter().map(|s| s.to_string()).collect();
        let user_set = move |name: &str| owned.iter().any(|n| n == name);
        let mut applied = Vec::new();
        apply_budget_levers(args, budget, &user_set, &mut applied).unwrap();
        applied
    }

    #[test]
    fn a_budget_applies_only_reversible_levers() {
        let mut args = args_from(&[]);
        let budget = budget_with(
            vec![
                lever("--publish", None, Some(-120_000)),
                lever("--blob-ordering", Some("time-major"), None),
                lever("--pack-size", Some("32"), None),
            ],
            Vec::new(),
        );
        let applied = fold(&mut args, &budget, &[]);

        assert_eq!(args.zstd_level, 19);
        assert_eq!(args.blob_ordering, "time-major");
        assert_eq!(args.pack_size, 32);
        // Each application names its measured byte price where the solver
        // published one.
        assert!(
            applied.iter().any(|a| a.contains("-120000 B measured")),
            "{applied:?}"
        );
    }

    #[test]
    fn no_shadow_price_can_reach_the_effective_build_config() {
        // ⚠️ THE NO-THINNING GUARD at the arg-folding boundary. Every lossy
        // lever the solver priced is present as a shadow price; NONE of them
        // may move a single field of the build config.
        let mut args = args_from(&[]);
        let budget = budget_with(
            vec![lever("--publish", None, Some(-1))],
            vec![
                shadow("--quantize-coords", Some("1"), 4_000_000),
                shadow("--quantize-attrs-auto", None, 1_500_000),
                shadow("--quantize-attr", Some("mag=0.01"), 900_000),
            ],
        );
        let applied = fold(&mut args, &budget, &[]);

        assert_eq!(args.quantize_coords, 0.0, "coordinates must stay Float64");
        assert!(!args.quantize_attrs_auto);
        assert!(args.quantize_attr.is_empty());
        assert!(!args.quantize_vertex_values);
        // …and the per-tile thinning budgets are equally untouched.
        assert!(args.maximum_tile_bytes.is_none());
        assert!(args.maximum_tile_features.is_none());
        assert!(args.min_features_per_tile <= 1);
        assert!(
            !applied.iter().any(|a| a.contains("quantize")),
            "a shadow price was recorded as APPLIED: {applied:?}"
        );
    }

    #[test]
    fn a_caveated_ordering_stays_suggestion_only_under_budget_pressure() {
        // The recorded gotcha: `blobOrdering: spatial` silently breaks
        // time-playback buffering (empty buffered ranges → stalls). Budget
        // pressure is not a reason to ship a broken demo, so a
        // `suggestion_only` lever is never folded however tight the budget is.
        let mut args = args_from(&[]);
        let before = args.blob_ordering.clone();
        let mut caveated = lever("--blob-ordering", Some("spatial"), None);
        caveated.suggestion_only = true;
        let budget = budget_with(vec![caveated], Vec::new());
        let applied = fold(&mut args, &budget, &[]);

        assert_eq!(args.blob_ordering, before);
        assert!(applied.is_empty(), "{applied:?}");
    }

    #[test]
    fn explicit_flags_beat_every_budget_lever() {
        let mut args = args_from(&[
            "--zstd-level",
            "7",
            "--blob-ordering",
            "hilbert3",
            "--pack-size",
            "64",
            "--temporal-lod",
            "1d",
        ]);
        let budget = budget_with(
            vec![
                lever("--publish", None, Some(-1)),
                lever("--blob-ordering", Some("time-major"), None),
                lever("--pack-size", Some("8"), None),
                lever("--temporal-lod", Some("6h"), Some(4_000)),
            ],
            Vec::new(),
        );
        let applied = fold(
            &mut args,
            &budget,
            &["zstd_level", "blob_ordering", "pack_size", "temporal_lod"],
        );

        assert_eq!(args.zstd_level, 7);
        assert_eq!(args.blob_ordering, "hilbert3");
        assert_eq!(args.pack_size, 64);
        assert_eq!(args.temporal_lod.as_deref(), Some("1d"));
        assert!(applied.is_empty(), "{applied:?}");
    }

    #[test]
    fn budget_mode_is_the_only_place_temporal_lod_is_auto_applied() {
        // ⚠️ THE ONE POLICY CHANGE, pinned. A priced tier IS folded under a
        // budget…
        let mut args = args_from(&["--temporal-bucket", "1h"]);
        let budget = budget_with(
            vec![lever("--temporal-lod", Some("4h,1d@6"), Some(45_000))],
            Vec::new(),
        );
        let applied = fold(&mut args, &budget, &[]);
        assert_eq!(args.temporal_lod.as_deref(), Some("4h,1d@6"));
        assert!(
            applied
                .iter()
                .any(|a| a.contains("temporal-lod") && a.contains("+45000 B measured")),
            "the tier must be logged with the price it cost: {applied:?}"
        );

        // …and the ADVISOR's identical suggestion, outside budget mode, is not:
        // it hits the blanket semantic-lever arm in `apply_auto_recommendations`,
        // which never writes the field. The structural guarantee is that the
        // whole binary assigns it in exactly ONE place — the budget path above.
        // (The needle is assembled at runtime so this assertion cannot match
        // itself.)
        let source =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bin/stt-build.rs"))
                .expect("read this binary's own source");
        let needle = format!("args.{} = Some(", "temporal_lod");
        assert_eq!(
            source.matches(needle.as_str()).count(),
            1,
            "`--temporal-lod` must be auto-applied from exactly ONE place (the budget path)"
        );
    }

    #[test]
    fn an_illegal_tier_is_skipped_rather_than_failing_the_build() {
        // The solver prices tiers as multiples of the bucket IT chose. A user
        // who pinned an incompatible `--temporal-bucket` must not get a build
        // that aborts inside `validate_lod` on a flag they never typed.
        let mut args = args_from(&["--temporal-bucket", "7m"]);
        let budget = budget_with(
            vec![lever("--temporal-lod", Some("4h"), Some(1_000))],
            Vec::new(),
        );
        let applied = fold(&mut args, &budget, &[]);
        assert!(args.temporal_lod.is_none(), "{:?}", args.temporal_lod);
        assert!(applied.is_empty(), "{applied:?}");

        // Adaptive windows have no fixed base bucket at all.
        let mut adaptive = args_from(&["--adaptive-temporal", "500"]);
        fold(&mut adaptive, &budget, &[]);
        assert!(adaptive.temporal_lod.is_none());

        // A legal tier against the same base still applies, so the guard is not
        // simply refusing everything.
        let mut legal = args_from(&["--temporal-bucket", "1h"]);
        fold(&mut legal, &budget, &[]);
        assert_eq!(legal.temporal_lod.as_deref(), Some("4h"));
    }

    #[test]
    fn budget_scalars_override_the_advisor_but_only_where_the_solver_moved_them() {
        let base = stt_optimize::Recommendations {
            min_zoom: 0,
            max_zoom: 12,
            temporal_bucket_ms: 3_600_000,
            temporal_bucket_human: "1 hour".to_string(),
            confidence: 90,
            dominant_type: "Point".to_string(),
            explanations: Vec::new(),
            advice: Vec::new(),
            composed_projected: None,
            composed_projected_with_lossy: None,
            budget: None,
        };
        // No budget: the advisor's scalars, untouched.
        assert_eq!(budget_scalars(&base), (0, 12, 3_600_000));

        // A budget that clamped zoom and widened the bucket.
        let mut clamped = base.clone();
        clamped.budget = Some(budget_with(
            vec![
                lever("--max-zoom", Some("9"), Some(-9_000_000)),
                lever("--temporal-bucket", Some("14400000"), Some(-2_000_000)),
            ],
            Vec::new(),
        ));
        assert_eq!(budget_scalars(&clamped), (0, 9, 14_400_000));

        // A `suggestion_only` scalar is never folded.
        let mut suggested = base.clone();
        let mut lever = lever("--max-zoom", Some("4"), None);
        lever.suggestion_only = true;
        suggested.budget = Some(budget_with(vec![lever], Vec::new()));
        assert_eq!(budget_scalars(&suggested), (0, 12, 3_600_000));
    }

    #[test]
    fn target_size_parses_the_documented_suffixes() {
        // The flag's value is parsed by the solver's own parser, so the CLI and
        // the report can never disagree about what `250M` means.
        assert_eq!(stt_optimize::parse_size("1024").unwrap(), 1024);
        assert_eq!(stt_optimize::parse_size("250MiB").unwrap(), 262_144_000);
        assert_eq!(stt_optimize::parse_size("1.5G").unwrap(), 1_610_612_736);
        assert_eq!(stt_optimize::parse_size("2MB").unwrap(), 2_000_000);
        assert!(stt_optimize::parse_size("nonsense").is_err());
        assert!(stt_optimize::parse_size("0").is_err());
    }

    #[test]
    fn target_size_is_optional_and_absent_by_default() {
        assert_eq!(args_from(&[]).target_size, None);
        assert_eq!(
            args_from(&["--target-size", "250MiB"])
                .target_size
                .as_deref(),
            Some("250MiB")
        );
    }

    /// BLOCKER A — the `--feature-id-scope` decision table, in full.
    ///
    /// `auto` is the only row that reads the evidence; `global` deliberately
    /// overrides it (that is what "assert without proof" means) while still
    /// surfacing the disagreement in its message, and `local` ignores it
    /// entirely.
    #[test]
    fn feature_id_scope_resolution_is_a_total_table() {
        use stt_build::input::FeatureIdAttestation;
        use stt_core::metadata::{FEATURE_ID_SCOPE_GLOBAL, FEATURE_ID_SCOPE_LOCAL};

        let proven = FeatureIdAttestation::Distinct;
        let declined = FeatureIdAttestation::NoSourceId { index: 0 };

        assert_eq!(args_from(&[]).feature_id_scope, FeatureIdScopeArg::Auto);

        // auto: follows the evidence, both ways.
        assert_eq!(
            resolve_feature_id_scope(FeatureIdScopeArg::Auto, &proven).0,
            FEATURE_ID_SCOPE_GLOBAL
        );
        let (scope, message) = resolve_feature_id_scope(FeatureIdScopeArg::Auto, &declined);
        assert_eq!(scope, FEATURE_ID_SCOPE_LOCAL);
        assert!(
            message.contains("id-less POINT") && message.contains("PER-TILE ROW INDEX"),
            "the log must name the evidence — and name the CONSTRUCTION, not \
             'carries no id', which is true of every archive this builder writes \
             including the line/polygon ones that ARE attestable: {message}"
        );

        // …and the OTHER decline the construction split introduced, so the
        // decision table stays total over the new evidence set.
        let clipped = FeatureIdAttestation::SegmentIds { index: 3 };
        let (scope, message) = resolve_feature_id_scope(FeatureIdScopeArg::Auto, &clipped);
        assert_eq!(scope, FEATURE_ID_SCOPE_LOCAL);
        assert!(
            message.contains("CLIPPED SEGMENT") && message.contains("#3"),
            "a trajectory decline must name the segment mechanism and the \
             feature: {message}"
        );

        // global: asserts regardless, and says so when it could not corroborate.
        assert_eq!(
            resolve_feature_id_scope(FeatureIdScopeArg::Global, &proven).0,
            FEATURE_ID_SCOPE_GLOBAL
        );
        let (scope, message) = resolve_feature_id_scope(FeatureIdScopeArg::Global, &declined);
        assert_eq!(scope, FEATURE_ID_SCOPE_GLOBAL);
        assert!(
            message.contains("ASSERTS") && message.contains("could not prove"),
            "an unproven assertion must announce itself: {message}"
        );

        // local: the rollback, whatever the evidence.
        for evidence in [&proven, &declined] {
            assert_eq!(
                resolve_feature_id_scope(FeatureIdScopeArg::Local, evidence).0,
                FEATURE_ID_SCOPE_LOCAL
            );
        }

        // The stamped spellings are a cross-crate contract with stt-validate.
        assert_eq!(FEATURE_ID_SCOPE_GLOBAL, "global");
        assert_eq!(FEATURE_ID_SCOPE_LOCAL, "local");
        assert_eq!(
            stt_core::metadata::FEATURE_ID_SCOPE_PROPERTY,
            "feature_id_scope"
        );
    }

    /// ⭐ BLOCKER 1 — the id-CONSTRUCTION vocabulary is a cross-crate contract
    /// between the builder that stamps it and `distinct_id_basis`, which arms
    /// the strict comparison off it. A spelling drift on either side silently
    /// demotes every new line/polygon archive back to un-checked.
    #[test]
    fn feature_id_construction_spellings_are_a_cross_crate_contract() {
        use stt_build::input::FeatureIdConstruction as C;
        use stt_core::metadata as m;

        assert_eq!(
            m::FEATURE_ID_CONSTRUCTION_PROPERTY,
            "feature_id_construction"
        );
        for (kind, wire, is_key) in [
            (C::Source, "source", true),
            (C::AnchorHash, "anchor-hash", true),
            (C::RowIndex, "row-index", false),
            (C::SegmentHash, "segment-hash", false),
        ] {
            assert_eq!(kind.as_manifest_value(), wire, "{kind:?}");
            assert_eq!(kind.is_dataset_wide_key(), is_key, "{kind:?}");
            // The reader half must agree with the writer half, value by value.
            assert_eq!(
                m::construction_is_a_dataset_wide_key(wire),
                is_key,
                "stt-core and stt-build disagree about {wire:?}"
            );
        }
        // A construction this build does not know never arms anything.
        assert!(!m::construction_is_a_dataset_wide_key("dense-renumbered"));
        assert!(!m::construction_is_a_dataset_wide_key(""));
    }

    /// Doc gate: every visible long flag must
    /// appear in THIS binary's section of `docs/api/cli-reference.md`, so a new
    /// flag fails the build until it is documented. Scoped to the section so a
    /// flag documented under another binary can't satisfy the gate.
    #[test]
    fn cli_flags_are_documented_in_cli_reference() {
        let doc = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/api/cli-reference.md"
        ))
        .expect("read docs/api/cli-reference.md");
        let start = doc
            .find("## `stt-build`")
            .expect("stt-build section heading");
        let body = &doc[start + 1..];
        let end = body
            .find("\n## `")
            .map(|i| start + 1 + i)
            .unwrap_or(doc.len());
        let section = &doc[start..end];
        let missing: Vec<String> = Args::command()
            .get_arguments()
            .filter(|a| !a.is_hide_set())
            .filter_map(|a| a.get_long())
            .filter(|l| !matches!(*l, "help" | "version"))
            .map(|l| format!("--{l}"))
            .filter(|f| !section.contains(f.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "flags missing from the `stt-build` section of docs/api/cli-reference.md \
             (document them before shipping): {missing:?}"
        );
    }
}
