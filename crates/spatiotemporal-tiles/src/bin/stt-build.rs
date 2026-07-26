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
    /// `auto` (default; pick from the dataset's OCCUPIED space-vs-time extent:
    /// shallow/wide-time → spatial-major, else the 3D-Hilbert generalist),
    /// `measured` (opt-in; pick the order that simulates the fewest range reads
    /// for this dataset), or an explicit `spatial`, `time-major`, `hilbert3`, or
    /// `morton3` (morton3 is research-only — empirically never the measured
    /// winner). Locality means fewer packs touched per viewport (fewer client
    /// range requests). Tile payloads buffer until finalize, in RAM up to
    /// `--pack-memory-budget` and spilled to disk beyond it.
    #[arg(long, default_value = "auto")]
    blob_ordering: String,

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
    /// IDENTICAL at any budget. `0` = unlimited (hold every payload in RAM,
    /// the legacy behaviour).
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

    /// Enable line simplification for lower zoom levels (reduces memory and improves performance)
    #[arg(long)]
    simplify: bool,

    /// Maximum zoom level to apply simplification (higher zooms keep full detail)
    #[arg(long, default_value = "14")]
    simplify_max_zoom: u8,

    /// Simplify with a latitude-corrected METRIC tolerance instead of the fixed
    /// per-zoom degree table. The longitude axis is scaled by cos(latitude)
    /// before simplifying, so a given zoom's tolerance means the same GROUND
    /// distance at every latitude (a fixed degree tolerance is up to ~2× coarser
    /// in E–W terms at 60° than at the equator). Opt-in — without it, builds are
    /// byte-identical to before. Requires `--simplify`.
    #[arg(long)]
    simplify_metric: bool,

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

    // --- Summary-tier options (server-aggregated low-zoom tier) ---
    /// Emit a pre-aggregated summary tier alongside the raw tier.
    /// Aggregation scheme: `h3` (Uber H3 hexes) or `quadbin` (CARTO quadbin).
    /// Both `h3` and `quadbin` are implemented.
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

    /// Escape hatch: do NOT declare required-to-understand features in
    /// `manifest.capabilities` (restores the pre-capabilities manifest bytes
    /// for a quantized / elevation-fold build). Only for byte-compat with
    /// tooling that predates the capability check — a reader that lacks a
    /// declared feature would then silently misdecode the re-typed columns
    /// instead of refusing at open. Builds using none of those features are
    /// unaffected (the key is omitted either way).
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
/// The input SCHEMA is authoritative where there is one (GeoParquet columns,
/// Postgres/DuckDB column types). GeoJSON has no schema, so that path returns
/// nothing and the tile builder would fall back to sniffing types from each
/// tile's own features — which drifts the layer schema across tiles: a column
/// whose values are all null in one tile disappears from it, and a column
/// holding only numeric-looking strings in one tile but a real string in
/// another flips between numeric and dictionary. Both leave the archive with
/// several schemas for one layer (`stt-validate` reports "schema drift").
///
/// So when the schema yields nothing, infer the kinds ONCE across every loaded
/// feature and pin them. The classification rule is the same either way; only
/// the evidence widens from one tile to the whole dataset.
fn resolve_property_types(
    source: &InputSource,
    args: &Args,
    features: &[stt_build::input::ParsedFeature],
    attribute_filter: &stt_build::columnar::AttributeFilter,
) -> Arc<stt_build::columnar::PropertyTypes> {
    let declared = source.property_kinds(&args.time_field, args.end_time_field.as_deref());
    if !declared.is_empty() {
        return declared;
    }
    let inferred = stt_build::columnar::infer_property_types(
        features.iter().map(|f| f.shared_properties.as_deref()),
        attribute_filter,
    );
    if inferred.is_empty() {
        return declared;
    }
    info!(
        "Inferred {} property kind(s) across the whole dataset (input carries no \
         schema); tile schemas stay identical",
        inferred.len()
    );
    Arc::new(inferred)
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

    if let Some(mode) = args.auto {
        apply_auto_recommendations(&matches, &mut args, mode)?;
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
    };
    let encoder_config = encoder_settings.resolve()?;
    let enabled = encoder_settings.enabled_summary()?;
    if !enabled.is_empty() {
        info!("Encoder settings ENABLED: {}", enabled.join(", "));
    }

    // Validate the --compression flag (packed output is zstd-only; the legacy
    // gzip/none choices now error). The parsed value is otherwise unused —
    // PackWriter is always per-blob zstd.
    parse_compression(&args.compression)?;

    // Parse the pack ordering (the packed format always buffers + reorders
    // before cutting packs). `measured` is the opt-in per-dataset picker
    // resolved by simulating range-read cost at finalize (it uses the Auto slot
    // as its placeholder ordering).
    let blob_arg = args.blob_ordering.trim();
    let measured_ordering = blob_arg.eq_ignore_ascii_case("measured");
    let pack_ordering: stt_core::BlobOrdering = if measured_ordering {
        stt_core::BlobOrdering::Auto
    } else {
        blob_arg.parse().map_err(|e: String| anyhow::anyhow!(e))?
    };

    // Pack target size (MiB -> bytes). Never 0.
    let pack_target_bytes = args.pack_size.saturating_mul(1024 * 1024).max(1);

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

    // Step 1: Load all features into memory
    info!("Loading input data...");
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message("Reading input file...");

    let features = source.load(
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
    info!(
        "Temporal bucket size: {} ms ({}))",
        temporal_bucket_ms, args.temporal_bucket
    );

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

    let tile_config = tiler::TileConfig {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        layer_name: args.layer.clone(),
        temporal_bucket_ms,
        clip_trajectories: !args.no_clip,
        clip_min_vertices: args.clip_min_vertices,
        simplify: args.simplify,
        simplify_max_zoom: args.simplify_max_zoom,
        simplify_metric: args.simplify_metric,
        pre_tessellate: args.pre_tessellate,
        temporal_lod: temporal_lod.clone(),
        min_features_per_tile: args.min_features_per_tile,
        time_aware_simplify: args.time_aware_simplify,
        adaptive_target_features: args.adaptive_temporal,
        min_zoom_field: args.min_zoom_field.clone(),
        max_zoom_field: args.max_zoom_field.clone(),
        tile_budget,
        attribute_filter,
        property_types,
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
    let manifest_capabilities = if args.no_manifest_capabilities {
        warn!(
            "manifest.capabilities SUPPRESSED (--no-manifest-capabilities): a reader \
             that does not implement the encoder features used here will silently \
             misdecode instead of refusing at open"
        );
        Vec::new()
    } else {
        let caps = encoder_settings.required_capabilities();
        if !caps.is_empty() {
            info!(
                "Manifest capabilities (required-to-understand): {}",
                caps.join(", ")
            );
        }
        caps
    };
    let mut writer = stt_core::PackWriter::create(&out_dir, pack_ordering, pack_target_bytes)?
        .with_paging((!args.single_directory).then_some(args.page_entries))
        .with_zstd_level(args.zstd_level)
        .with_capabilities(manifest_capabilities)
        .with_measured_ordering(measured_ordering)
        .with_memory_budget(args.pack_memory_budget.saturating_mul(1024 * 1024))
        // Every tile-writing path below (streaming, batched, LOD, summary) pulls
        // its `EncoderConfig` back off the writer, so the resolved flags travel
        // with the sink instead of through process-wide state.
        .with_encoder_config(encoder_config);
    if args.pack_memory_budget > 0 {
        info!(
            "Pack-writer memory budget: {} MiB (payloads beyond it spill to a temp \
             file in the output dir; output bytes are identical at any budget)",
            args.pack_memory_budget
        );
    } else {
        info!("Pack-writer memory budget: unlimited (--pack-memory-budget 0; all payloads in RAM)");
    }
    // The frame version rides the writer: the PackWriter tile sinks
    // (`TileWriter`/`LodTileWriter`/`write_tiles_parallel` in stt_build::tiler)
    // encode every payload with the writer's format_version + template
    // collector, so frames and `manifest.formatVersion` can never diverge.
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
            "Paged directory: root page + leaf pages of ≤{} entries (cold reader fetches \
             only visited leaves)",
            args.page_entries
        );
    }

    let tile_count = if !temporal_lod.is_empty() {
        // --temporal-lod path: emit base tiles + per-level aggregate tiles.
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
        // Parallel encode (bounded by --workers), deterministic ordered write.
        let keep: Vec<(&tiler::GeneratedTile, Option<u64>)> = tiles
            .iter()
            .filter(|tagged| tagged.tile.feature_count() >= min_features)
            .map(|tagged| (&tagged.tile, tagged.temporal_bucket_ms))
            .collect();
        pb.inc((tiles.len() - keep.len()) as u64);
        tiler::write_tiles_parallel(&mut writer, &keep, args.workers, || pb.inc(1))?;
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
        tiler::write_tiles_parallel(&mut writer, &keep, args.workers, || pb.inc(1))?;
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
        // A streaming build carries no in-memory feature slice, so it emits
        // neither the layer hint nor the property profile (documented limitation).
    } else if let Some(hints) = stt_build::style_hints::compute_style_hints(
        &features,
        &time_range,
        temporal_bucket_ms,
        args.style_hints, // full percentile/cardinality profile only when requested
    ) {
        info!(
            "Style hints: {} properties profiled, layer hint {}, suggested playback {}",
            hints.properties.len(),
            hints.layer_hint.as_deref().unwrap_or("(none)"),
            hints
                .suggested_playback_seconds
                .map(|s| format!("{s}s"))
                .unwrap_or_else(|| "(none)".to_string()),
        );
        metadata = metadata.with_style_hints(hints);
    }

    // Record the DISTINCT source-feature count (pre-placement) so downstream
    // "N features" badges don't have to sum the index-weighted per-tile total,
    // which double-counts every feature that spans tiles / pyramid levels.
    metadata.distinct_feature_count = Some(features.len() as u64);

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
fn apply_auto_recommendations(matches: &ArgMatches, args: &mut Args, mode: AutoMode) -> Result<()> {
    info!("--auto ({mode:?}): analyzing input for build recommendations...");
    let path = args.input.clone().ok_or_else(|| {
        anyhow::anyhow!("--auto is not supported with --postgres/--duckdb (the analyzer reads a GeoParquet file)")
    })?;
    let source = stt_optimize::DataSource::GeoParquet {
        path,
        time_field: args.time_field.clone(),
        time_format: args.time_format.as_str().to_string(),
    };
    let rec = stt_optimize::recommend_for(&source).context("stt-optimize analyzer failed")?;

    let user_set =
        |name: &str| matches!(matches.value_source(name), Some(ValueSource::CommandLine));

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
        info!("  temporal-bucket: {} (auto)", rec.temporal_bucket_human);
        // Fold in the ms form: the human string ("30 days") is for logs only
        // and is not always `parse_duration`-compatible.
        args.temporal_bucket = format!("{}ms", rec.temporal_bucket_ms);
    }

    info!(
        "  confidence: {}% — {} reasons",
        rec.confidence,
        rec.explanations.len()
    );
    for line in &rec.explanations {
        info!("    - {}", line);
    }

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
            _ => info!(
                "  suggested, not applied (semantic lever): {suggestion} — {}",
                advice.why
            ),
        }
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

    /// Doc gate (naming-types-consistency F9): every visible long flag must
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
