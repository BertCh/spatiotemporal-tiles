//! stt-serve — a dynamic STT tile server with full `stt-build` generation parity.
//!
//! Generates spatiotemporal tiles on the fly from a live **PostGIS** table or a
//! **DuckDB** database, one `(z, x, y, t)` per request — the `ST_AsMVT` analog
//! for the STT format. Each request:
//!   1. maps `(z, x, y)` to a WGS84 bounding box and `t` to a temporal bucket,
//!   2. runs a source query filtered by the bbox (spatial index) + time window,
//!   3. decodes the rows to features and encodes ONE STT tile blob
//!      (`stt_build::encode_single_tile_counted`) — Arrow IPC + GeoArrow,
//!      **byte-identical to what the offline build would emit for that tile**.
//!
//! Parity with `stt-build`: the encoder reads the same per-tile [`TileConfig`]
//! fields (clip / simplify / pre-tessellate / min-/max-zoom-field / per-tile
//! budget / attribute filter) and the same process-wide encoder globals
//! (coordinate + attribute quantization, vector grouping, point-elevation fold,
//! vertex-time precision). Both are configured from the SAME flags the CLI uses,
//! parsed by `stt_build::build_options`, so a served tile matches the offline
//! tile for the same `(z, x, y, t)` and the same source rows.
//!
//! What is NOT servable per single-tile request (each cleanly rejected at
//! startup): `--summary-tier` (cross-tile H3/quadbin aggregation) and
//! `--adaptive-temporal` (windows sized across a cell's whole time range).
//! `--temporal-lod` IS served: when an LOD level applies at the requested zoom,
//! the server widens the temporal bucket to that level's `bucket_ms` (the
//! coarsest applicable), matching the offline LOD tier. A whole-dataset heatmap
//! domain (`--heatmap-weight`/`--heatmap-class`) is computed once at startup via
//! a SQL aggregate and advertised in `/metadata.json`.
//!
//! The backend is chosen by `--postgres <CONN>` or `--duckdb <PATH>`; each is
//! compiled in by its cargo feature (`serve-postgres` / `serve-duckdb`), and at
//! least one must be enabled. PostGIS uses an async client + pool; DuckDB is
//! embedded and blocking, so its pool checkout / query / decode / encode all
//! run on `spawn_blocking`.
//!
//! Tiles are regenerated per request (no app cache), so unlike the pre-baked
//! packed format this is NOT edge-cacheable — that is the live-source trade-off.
//! The source geometry must be EPSG:4326 lon/lat (index the table in 4326;
//! reproject at ingest with `stt-build --source-srid` if needed — reprojecting
//! in a per-tile filter would defeat the spatial index).

// `serve-core` alone is the backend-less plumbing feature — building the bin
// from it directly is a misconfiguration (the public `serve` feature enables
// both backends; `serve-postgres` / `serve-duckdb` pick one).
#[cfg(not(any(feature = "serve-postgres", feature = "serve-duckdb")))]
compile_error!(
    "stt-serve needs at least one backend: enable `serve` (both backends), `serve-postgres`, or `serve-duckdb`"
);

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use stt_build::build_options::{self, EncoderSettings};
use stt_build::input::{InputStrictness, ParsedFeature, TimeFormat};
use stt_build::tiler::{encode_single_tile_counted, TileConfig};
use stt_core::metadata::{HeatmapClassDomain, HeatmapDomain};

// Postgres backend (feature `serve-postgres`).
#[cfg(feature = "serve-postgres")]
use deadpool_postgres::{Manager, ManagerConfig, Pool as PgPool, RecyclingMethod};
#[cfg(feature = "serve-postgres")]
use stt_build::postgres_input;
#[cfg(feature = "serve-postgres")]
use tokio_postgres::NoTls;

// DuckDB backend (feature `serve-duckdb`).
#[cfg(feature = "serve-duckdb")]
use duckdb::{AccessMode, Config as DuckConfig, DuckdbConnectionManager};
#[cfg(feature = "serve-duckdb")]
use r2d2::Pool as R2d2Pool;
#[cfg(feature = "serve-duckdb")]
use stt_build::duckdb_input;

/// CLI args AND the per-dataset schema of a `--config` JSON file: every field is
/// a per-dataset knob, so one `Args` describes one dataset either way. `serde`
/// fills any field a config entry omits from [`Args::default`] (= the clap
/// defaults, the single source of truth), so config-file datasets and the CLI
/// share identical defaults. Unknown JSON keys are rejected to catch typos.
#[derive(Parser, Clone, Deserialize)]
#[command(name = "stt-serve", version)]
#[command(
    about = "Serve STT tiles dynamically from a live PostGIS table or a DuckDB database, with full stt-build generation parity",
    long_about = None
)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct Args {
    /// Dataset name (the `/{name}/…` URL segment + the metadata `name`) in
    /// multi-dataset `--config` mode. Single-dataset mode derives it from the
    /// table/query and serves at the root.
    #[arg(long)]
    name: Option<String>,

    /// Multi-dataset mode: path to a JSON file `{ "datasets": [ {…}, … ] }` where
    /// each entry has the same fields as these flags. Each dataset is served
    /// under `/{name}/…`, with `/datasets` listing the catalog. CLI-only.
    #[arg(long)]
    #[serde(skip)]
    config: Option<PathBuf>,

    // --- Source ------------------------------------------------------------
    /// PostgreSQL/PostGIS connection string (libpq URI or key=value). Env
    /// fallback: STT_POSTGRES_URL, then DATABASE_URL. Mutually exclusive with
    /// --duckdb.
    #[cfg(feature = "serve-postgres")]
    #[arg(long)]
    postgres: Option<String>,

    /// DuckDB database file path (or `:memory:` with a `--sql` that scans
    /// external files, e.g. `read_parquet(...)`). Env fallback: STT_DUCKDB_PATH.
    /// Mutually exclusive with --postgres.
    #[cfg(feature = "serve-duckdb")]
    #[arg(long)]
    duckdb: Option<String>,

    /// Source table (optionally schema-qualified, e.g. public.hurricane_obs).
    /// Mutually exclusive with --sql; provide exactly one.
    #[arg(long)]
    table: Option<String>,

    /// Arbitrary SQL `SELECT` to serve from (wrapped as a subquery). Mutually
    /// exclusive with --table. Must expose `--geom-column` and `--time-field`.
    #[arg(long)]
    sql: Option<String>,

    /// Geometry column (must be EPSG:4326 lon/lat).
    #[arg(long, default_value = "geom")]
    geom_column: String,

    /// Timestamp column (a timestamp / timestamptz type, or an integer column
    /// interpreted per --time-format). Default matches `stt-build` /
    /// `stt-optimize` (`timestamp`) so the same source builds identically
    /// offline and live.
    #[arg(long, default_value = "timestamp")]
    time_field: String,

    /// Optional end-timestamp column for features with a time range.
    #[arg(long)]
    end_time_field: Option<String>,

    /// Wire format of an INTEGER time column (ignored for timestamp columns).
    #[arg(long, value_enum, default_value = "iso8601")]
    time_format: TimeFormat,

    // --- Tiling (matches stt-build) ----------------------------------------
    /// Minimum / maximum zoom advertised in /metadata.json (and the zoom range
    /// LOD levels apply within).
    #[arg(long, default_value = "0")]
    min_zoom: u8,
    #[arg(long, default_value = "14")]
    max_zoom: u8,

    /// Temporal bucket size (e.g. 1h, 30m, 6h, 1d) — must match how clients
    /// address tiles in time. Parsed identically to `stt-build`.
    #[arg(long, default_value = "1h")]
    temporal_bucket: String,

    /// Optional temporal LOD pyramid (e.g. "1d,30d" or "1d@8,30d@4"), matching
    /// `stt-build --temporal-lod`. When an LOD level applies at the requested
    /// zoom the server widens the bucket to that level's size (coarsest wins).
    #[arg(long)]
    temporal_lod: Option<String>,

    /// Layer name embedded in each tile.
    #[arg(long, default_value = "default")]
    layer: String,

    /// Disable trajectory clipping (store whole LineStrings in their centroid
    /// tile). No effect on point/polygon data.
    #[arg(long)]
    no_clip: bool,

    /// Minimum vertices required before a trajectory is clipped.
    #[arg(long, default_value = "2")]
    clip_min_vertices: usize,

    /// Enable line simplification at/below --simplify-max-zoom.
    #[arg(long)]
    simplify: bool,

    /// Highest zoom that still receives simplification.
    #[arg(long, default_value = "14")]
    simplify_max_zoom: u8,

    /// Use time-aware TD-TR (SED) simplification (preserves per-vertex timing).
    #[arg(long)]
    time_aware_simplify: bool,

    /// Pre-tessellate polygons (bake earcut triangle indices into a sidecar).
    #[arg(long)]
    pre_tessellate: bool,

    /// Per-feature numeric property naming the shallowest zoom a feature appears
    /// at (road-class-style LOD floor).
    #[arg(long)]
    min_zoom_field: Option<String>,

    /// Per-feature numeric property naming the deepest zoom a feature appears at
    /// (LOD ceiling).
    #[arg(long)]
    max_zoom_field: Option<String>,

    /// Drop a tile whose placed-feature count is below this threshold (returns
    /// 204 No Content), matching the offline writer's `--min-features-per-tile`.
    #[arg(long, default_value = "1")]
    min_features_per_tile: u32,

    // --- Per-tile budgets (opt-in) -----------------------------------------
    /// OPT-IN soft cap on a tile's estimated uncompressed payload in bytes.
    #[arg(long, value_name = "BYTES")]
    maximum_tile_bytes: Option<usize>,

    /// OPT-IN hard cap on the number of features per tile.
    #[arg(long, value_name = "N")]
    maximum_tile_features: Option<usize>,

    /// With a budget, drop the densest features first (pure geometry size).
    #[arg(long)]
    drop_densest_as_needed: bool,

    // --- Attribute control (opt-in) ----------------------------------------
    /// OPT-IN: drop these property columns from output tiles (repeatable).
    #[arg(long, value_name = "PROP")]
    exclude: Vec<String>,

    /// OPT-IN: keep ONLY these property columns (repeatable).
    #[arg(long, value_name = "PROP")]
    include: Vec<String>,

    /// OPT-IN: drop EVERY user property — geometry + times only.
    #[arg(long)]
    exclude_all: bool,

    // --- Process-wide encoder settings (matches stt-build) -----------------
    /// Ceiling (ms) on the per-vertex time quantization step.
    #[arg(long, default_value_t = stt_core::arrow_tile::DEFAULT_VERTEX_TIME_MAX_STEP_MS, value_name = "MS")]
    vertex_time_precision: u32,

    /// Opt-in coordinate quantization: ground precision in meters (`0` = off).
    #[arg(long, default_value_t = 0.0, value_name = "METERS")]
    quantize_coords: f64,

    /// Opt-in numeric-attribute quantization `NAME=PREC` (repeatable).
    #[arg(long = "quantize-attr", value_name = "NAME=PREC")]
    quantize_attr: Vec<String>,

    /// Auto-quantize every Float64 numeric property to a range-adaptive UInt16.
    #[arg(long = "quantize-attrs-auto", default_value_t = false)]
    quantize_attrs_auto: bool,

    /// Fuse scalar columns into an interleaved GPU-ready FixedSizeList:
    /// `NAME=col1,col2,...[:f32|u8]` (repeatable).
    #[arg(long = "vector-group", value_name = "NAME=COLS[:f32|u8]")]
    vector_group: Vec<String>,

    /// Fold a numeric property into POINT geometry as the z coordinate.
    #[arg(long = "point-elevation-column", value_name = "NAME")]
    point_elevation_column: Option<String>,

    // --- Whole-dataset metadata --------------------------------------------
    /// Feature property driving the HeatmapLayer weight. Its [min, 95th
    /// percentile] (per --heatmap-class if set) is computed once at startup via
    /// a SQL aggregate and advertised as the default heatmap domain.
    #[arg(long)]
    heatmap_weight: Option<String>,

    /// Categorical property whose distinct values become per-class heatmap
    /// entries (capped at 8).
    #[arg(long)]
    heatmap_class: Option<String>,

    // --- Rejected: not servable per single-tile request --------------------
    /// (Rejected.) Summary tiers are cross-tile aggregation — pre-bake them with
    /// `stt-build --summary-tier` and serve the static archive.
    #[arg(long)]
    summary_tier: Option<String>,

    /// (Rejected.) Adaptive temporal windows are sized across a cell's whole
    /// time range — pre-bake with `stt-build --adaptive-temporal`.
    #[arg(long)]
    adaptive_temporal: Option<u32>,

    // --- Server ------------------------------------------------------------
    /// Connection-pool size.
    #[arg(long, default_value = "8")]
    pool_size: usize,

    /// Listen address.
    #[arg(long, default_value = "127.0.0.1:8088")]
    bind: SocketAddr,
}

impl Default for Args {
    /// The clap defaults, reused verbatim as the serde defaults — so a config
    /// file dataset that omits a field gets exactly the CLI default. Parsing an
    /// empty argv can't fail: no `Args` field is a required clap argument.
    fn default() -> Self {
        Args::parse_from(["stt-serve"])
    }
}

/// A `--config` file: one or more datasets, each described by the same fields as
/// the CLI flags.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ServeConfig {
    datasets: Vec<Args>,
}

/// r2d2 manager that loads the DuckDB `spatial` extension and pins UTC on every
/// *new physical connection* (once, not per checkout), so `ST_AsWKB` /
/// `ST_Intersects` / `epoch_ms` are available and epoch math is tz-independent.
#[cfg(feature = "serve-duckdb")]
struct SpatialDuckManager {
    inner: DuckdbConnectionManager,
}

#[cfg(feature = "serve-duckdb")]
impl r2d2::ManageConnection for SpatialDuckManager {
    type Connection = duckdb::Connection;
    type Error = duckdb::Error;

    fn connect(&self) -> std::result::Result<Self::Connection, Self::Error> {
        let conn = self.inner.connect()?;
        conn.execute_batch("INSTALL spatial; LOAD spatial; SET TimeZone='UTC';")?;
        Ok(conn)
    }
    fn is_valid(&self, conn: &mut Self::Connection) -> std::result::Result<(), Self::Error> {
        self.inner.is_valid(conn)
    }
    fn has_broken(&self, conn: &mut Self::Connection) -> bool {
        self.inner.has_broken(conn)
    }
}

#[cfg(feature = "serve-duckdb")]
type DuckPool = R2d2Pool<SpatialDuckManager>;

/// The selected data source (only compiled-in backends have a variant).
enum Backend {
    #[cfg(feature = "serve-postgres")]
    Postgres(PgPool),
    #[cfg(feature = "serve-duckdb")]
    DuckDb(DuckPool),
}

/// Shared, immutable server state.
struct ServerState {
    backend: Backend,
    /// The FROM source — a table/view name, or `( <sql> ) AS __stt_src` for a
    /// `--sql` source. Interpolated into the per-tile + metadata queries.
    source: String,
    geom_column: String,
    time_field: String,
    end_time_field: Option<String>,
    time_format: TimeFormat,
    config: TileConfig,
    /// Resolved encoder settings passed EXPLICITLY to the encoder per request
    /// (`encode_single_tile_counted`) — the server never mutates the process-wide
    /// encoder globals, so several datasets/configs can be served concurrently.
    encoder: stt_core::arrow_tile::EncoderConfig,
    metadata: serde_json::Value,
}

/// Tile `(z, x, y)` → WGS84 `[min_lon, min_lat, max_lon, max_lat]` (Web
/// Mercator slippy formula) with a small buffer so a feature near a tile edge
/// is never missed by the SQL pre-filter. `encode_single_tile_counted` performs
/// the authoritative, exact per-tile placement afterward.
fn tile_bbox(z: u8, x: u32, y: u32) -> [f64; 4] {
    let n = (1u64 << z) as f64;
    let min_lon = x as f64 / n * 360.0 - 180.0;
    let max_lon = (x + 1) as f64 / n * 360.0 - 180.0;
    let lat = |yy: f64| (std::f64::consts::PI * (1.0 - 2.0 * yy / n)).sinh().atan().to_degrees();
    let max_lat = lat(y as f64);
    let min_lat = lat((y + 1) as f64);
    // The clipper pads every tile by a fixed 0.001° band
    // (`clip_config_from`, tiler.rs `buffer_degrees`); at high zoom 5% of a
    // tile is narrower than that band, so floor the prefilter buffer at 0.001°
    // or edge features the encoder would keep never reach it.
    let bx = ((max_lon - min_lon).abs() * 0.05).max(0.001);
    let by = ((max_lat - min_lat).abs() * 0.05).max(0.001);
    [min_lon - bx, min_lat - by, max_lon + bx, max_lat + by]
}

/// Single-dataset tile route (`/tiles/…`).
async fn tile_handler(
    State(st): State<Arc<ServerState>>,
    Path((z, x, y, t)): Path<(u8, u32, u32, String)>,
) -> Response {
    serve_tile(&st, z, x, y, t).await
}

/// Multi-dataset tile route (`/{dataset}/tiles/…`) — dispatch to the named
/// dataset's state, or 404.
async fn tile_handler_multi(
    State(reg): State<Arc<Registry>>,
    Path((dataset, z, x, y, t)): Path<(String, u8, u32, u32, String)>,
) -> Response {
    match reg.datasets.get(&dataset) {
        Some(st) => serve_tile(st, z, x, y, t).await,
        None => (StatusCode::NOT_FOUND, format!("unknown dataset '{dataset}'")).into_response(),
    }
}

/// The shared tile-serving core: parse `t`, generate the tile, and build the
/// HTTP response. Reused by the single- and multi-dataset routes.
async fn serve_tile(st: &Arc<ServerState>, z: u8, x: u32, y: u32, t: String) -> Response {
    let t_ms = match t.trim_end_matches(".stt").parse::<i64>() {
        Ok(v) => v,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "t must be an integer (ms since epoch)").into_response()
        }
    };
    match build_tile(st, z, x, y, t_ms).await {
        Ok(Some((bytes, micros))) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-stt-tile")
            .header(header::CACHE_CONTROL, "no-store")
            .header("x-stt-gen-micros", micros.to_string())
            .body(Body::from(bytes))
            .unwrap(),
        // Empty / below-threshold tile.
        Ok(None) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            // Full error chain (SQL, connection strings, …) goes to the log
            // only — clients get an opaque 500.
            tracing::error!("tile {z}/{x}/{y}/{t_ms} failed: {e:#}");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error generating tile").into_response()
        }
    }
}

/// Query the source for one tile's candidate features and encode the tile.
/// Returns `(bytes, generation_micros)`; `None` when the tile is empty or below
/// `--min-features-per-tile`.
async fn build_tile(
    st: &Arc<ServerState>,
    z: u8,
    x: u32,
    y: u32,
    t_ms: i64,
) -> Result<Option<(Vec<u8>, u128)>> {
    let started = Instant::now();
    // Pick the effective temporal bucket: an LOD level that applies at this zoom
    // widens it (coarsest applicable wins), matching the offline LOD tier.
    let base_bucket = st.config.temporal_bucket_ms.max(1);
    let lod_bucket = st.config.lod_for_zoom(z).map(|l| l.bucket_ms.max(1));
    let effective_bucket = lod_bucket.unwrap_or(base_bucket);
    let bucket_start = (t_ms.max(0) as u64 / effective_bucket) * effective_bucket;
    let bbox = tile_bbox(z, x, y);
    let t_start = bucket_start as i64;
    let t_end = (bucket_start + effective_bucket) as i64;

    let bytes: Option<Vec<u8>> = match &st.backend {
        #[cfg(feature = "serve-postgres")]
        Backend::Postgres(pool) => {
            let sql = postgres_input::build_tile_query(
                &st.source,
                &st.geom_column,
                &st.time_field,
                st.time_format,
                bbox,
                t_start,
                t_end,
            );
            let client = pool.get().await.context("get pooled PG connection")?;
            let rows = client.query(&sql, &[]).await.context("tile query")?;
            if rows.is_empty() {
                return Ok(None);
            }
            // Row decode + tile encode are CPU-bound — run off the async reactor.
            let st = st.clone();
            tokio::task::spawn_blocking(move || -> Result<Option<Vec<u8>>> {
                let features = postgres_input::decode_rows(
                    &rows,
                    &st.time_field,
                    st.end_time_field.as_deref(),
                    &st.geom_column,
                    st.time_format,
                    InputStrictness::Warn,
                    InputStrictness::Warn,
                )?;
                encode_gated(&st, &features, z, x, y, t_ms, lod_bucket)
            })
            .await
            .context("encode task join")??
        }
        #[cfg(feature = "serve-duckdb")]
        Backend::DuckDb(pool) => {
            let sql = duckdb_input::build_tile_query(
                &st.source,
                &st.geom_column,
                &st.time_field,
                st.time_format,
                bbox,
                t_start,
                t_end,
            );
            // DuckDB is fully blocking: pool checkout, query, decode, encode all
            // run on one blocking worker.
            let pool = pool.clone();
            let st = st.clone();
            tokio::task::spawn_blocking(move || -> Result<Option<Vec<u8>>> {
                let conn = pool.get().context("get pooled DuckDB connection")?;
                let features = duckdb_input::decode_query(
                    &conn,
                    &sql,
                    &st.time_field,
                    st.end_time_field.as_deref(),
                    &st.geom_column,
                    st.time_format,
                    InputStrictness::Warn,
                    InputStrictness::Warn,
                )?;
                encode_gated(&st, &features, z, x, y, t_ms, lod_bucket)
            })
            .await
            .context("encode task join")??
        }
    };

    Ok(bytes.map(|b| (b, started.elapsed().as_micros())))
}

/// Encode one tile and apply the `--min-features-per-tile` gate, returning the
/// raw STT frame bytes or `None` (empty or below threshold). When an LOD level
/// applies (`lod_bucket`), the per-request config widens the temporal bucket to
/// that level's size so `encode_single_tile_counted` buckets identically to the
/// offline LOD tier.
fn encode_gated(
    st: &Arc<ServerState>,
    features: &[ParsedFeature],
    z: u8,
    x: u32,
    y: u32,
    t_ms: i64,
    lod_bucket: Option<u64>,
) -> Result<Option<Vec<u8>>> {
    if features.is_empty() {
        return Ok(None);
    }
    let min_features = st.config.min_features_per_tile.max(1);
    let encoded = match lod_bucket {
        Some(b) => {
            let mut cfg = st.config.clone();
            cfg.temporal_bucket_ms = b;
            cfg.temporal_lod = Vec::new();
            encode_single_tile_counted(features, z, x, y, t_ms, &cfg, &st.encoder)?
        }
        None => encode_single_tile_counted(features, z, x, y, t_ms, &st.config, &st.encoder)?,
    };
    Ok(encoded.and_then(|(bytes, count)| (count >= min_features).then_some(bytes)))
}

async fn metadata_handler(State(st): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    Json(st.metadata.clone())
}

/// The multi-dataset registry served under `/{dataset}/…`. Each dataset carries
/// its own backend pool, `TileConfig`, and explicit `EncoderConfig`, so several
/// datasets with DIFFERENT quantization / vector groups are served concurrently
/// from one process — the payoff of threading encoder settings instead of using
/// process-wide globals.
struct Registry {
    datasets: HashMap<String, Arc<ServerState>>,
}

/// Multi-dataset metadata route (`/{dataset}/metadata.json`).
async fn metadata_handler_multi(
    State(reg): State<Arc<Registry>>,
    Path(dataset): Path<String>,
) -> Response {
    match reg.datasets.get(&dataset) {
        Some(st) => Json(st.metadata.clone()).into_response(),
        None => (StatusCode::NOT_FOUND, format!("unknown dataset '{dataset}'")).into_response(),
    }
}

/// Catalog route (`/datasets`) — the dataset names + their advertised metadata.
async fn datasets_handler(State(reg): State<Arc<Registry>>) -> Json<serde_json::Value> {
    let mut names: Vec<&String> = reg.datasets.keys().collect();
    names.sort();
    let list: Vec<serde_json::Value> = names
        .into_iter()
        .map(|name| {
            serde_json::json!({ "name": name, "metadata": reg.datasets[name].metadata })
        })
        .collect();
    Json(serde_json::json!({ "datasets": list }))
}

/// Assemble the metadata JSON for /metadata.json from resolved source extent
/// values, optionally carrying a whole-dataset heatmap domain. This runtime
/// descriptor is uniformly camelCase (loaders.gl `TileSource` style), so the
/// advisory `heatmapDomain` key is camelCase too — distinct from the packed
/// manifest's snake_case `heatmap_domain` field, which no client reads off the
/// serve response.
#[allow(clippy::too_many_arguments)]
fn metadata_json(
    name: &str,
    args: &Args,
    bucket_ms: u64,
    format: &str,
    min_lon: Option<f64>,
    min_lat: Option<f64>,
    max_lon: Option<f64>,
    max_lat: Option<f64>,
    t_start: Option<i64>,
    t_end: Option<i64>,
    count: i64,
    heatmap_domain: Option<HeatmapDomain>,
) -> Result<serde_json::Value> {
    let mut meta = serde_json::json!({
        "format": format,
        "name": name,
        "boundingBox": [
            [min_lon.unwrap_or(-180.0), min_lat.unwrap_or(-90.0)],
            [max_lon.unwrap_or(180.0), max_lat.unwrap_or(90.0)],
        ],
        "timeRange": {
            "start": t_start.unwrap_or(0),
            "end": t_end.unwrap_or(0),
        },
        "minZoom": args.min_zoom,
        "maxZoom": args.max_zoom,
        "temporalBucketMs": bucket_ms,
        "featureCount": count,
        "tileUrlTemplate": "/tiles/{z}/{x}/{y}/{t}.stt",
    });
    if let Some(domain) = heatmap_domain {
        meta["heatmapDomain"] = serde_json::to_value(domain).context("serialize heatmap domain")?;
    }
    Ok(meta)
}

/// Double-quote a SQL identifier (escaping embedded quotes).
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// PostGIS startup metadata aggregate (extent + time range + count) + optional
/// heatmap domain.
#[cfg(feature = "serve-postgres")]
async fn load_metadata_pg(
    pool: &PgPool,
    args: &Args,
    source: &str,
    name: &str,
    bucket_ms: u64,
) -> Result<serde_json::Value> {
    let client = pool.get().await.context("get connection for metadata")?;
    let geom = quote_ident(&args.geom_column);
    let time = quote_ident(&args.time_field);
    // Convert min/max time to epoch-ms per `--time-format` so an integer-epoch
    // column advertises the same extent it would when ingested.
    let (tmin_ms, tmax_ms) = match args.time_format {
        TimeFormat::Iso8601 => (
            format!("(EXTRACT(EPOCH FROM MIN({time})) * 1000.0)::float8"),
            format!("(EXTRACT(EPOCH FROM MAX({time})) * 1000.0)::float8"),
        ),
        TimeFormat::UnixMs => (
            format!("MIN({time})::float8"),
            format!("MAX({time})::float8"),
        ),
        TimeFormat::UnixSec => (
            format!("(MIN({time}) * 1000)::float8"),
            format!("(MAX({time}) * 1000)::float8"),
        ),
    };
    let sql = format!(
        "SELECT ST_XMin(ext), ST_YMin(ext), ST_XMax(ext), ST_YMax(ext), tmin, tmax, cnt \
         FROM ( \
            SELECT ST_Extent({geom}) AS ext, \
                   {tmin_ms} AS tmin, \
                   {tmax_ms} AS tmax, \
                   COUNT(*) AS cnt \
            FROM {source} \
         ) q",
    );
    let row = client.query_one(&sql, &[]).await.context("metadata aggregate query")?;
    let t_start: Option<f64> = row.get(4);
    let t_end: Option<f64> = row.get(5);
    let heatmap = load_heatmap_pg(
        pool,
        source,
        args.heatmap_weight.as_deref(),
        args.heatmap_class.as_deref(),
    )
    .await
    .context("compute heatmap domain")?;
    metadata_json(
        name,
        args,
        bucket_ms,
        "stt-postgis-dynamic",
        row.get(0),
        row.get(1),
        row.get(2),
        row.get(3),
        t_start.map(|v| v as i64),
        t_end.map(|v| v as i64),
        row.get(6),
        heatmap,
    )
}

/// PostGIS heatmap-domain aggregate. `min` + 95th-percentile of the weight
/// column (per class when `--heatmap-class` is set, capped at 8). Computed via
/// the DB's continuous `percentile_cont`, which may differ marginally from the
/// offline floor-index percentile (this is a style hint, not tile bytes).
#[cfg(feature = "serve-postgres")]
async fn load_heatmap_pg(
    pool: &PgPool,
    source: &str,
    weight: Option<&str>,
    class: Option<&str>,
) -> Result<Option<HeatmapDomain>> {
    if weight.is_none() && class.is_none() {
        return Ok(None);
    }
    let client = pool.get().await.context("get connection for heatmap")?;
    let classes: Vec<HeatmapClassDomain> = match (weight, class) {
        (Some(w), None) => {
            let wq = quote_ident(w);
            let sql = format!(
                "SELECT min({wq})::float8, \
                        percentile_cont(0.95) WITHIN GROUP (ORDER BY {wq})::float8 \
                 FROM {source} WHERE {wq} IS NOT NULL"
            );
            let row = client.query_one(&sql, &[]).await.context("heatmap aggregate")?;
            let mn: Option<f64> = row.get(0);
            let mx: Option<f64> = row.get(1);
            match mn {
                Some(min) => vec![HeatmapClassDomain {
                    id: "default".to_string(),
                    min,
                    max: mx.unwrap_or(1.0),
                    property: Some(w.to_string()),
                }],
                None => Vec::new(),
            }
        }
        (Some(w), Some(c)) => {
            let wq = quote_ident(w);
            let cq = quote_ident(c);
            let sql = format!(
                "SELECT {cq}::text AS cls, min({wq})::float8, \
                        percentile_cont(0.95) WITHIN GROUP (ORDER BY {wq})::float8 \
                 FROM {source} WHERE {wq} IS NOT NULL AND {cq} IS NOT NULL \
                 GROUP BY {cq} ORDER BY {cq} LIMIT 8"
            );
            let rows = client.query(&sql, &[]).await.context("heatmap class aggregate")?;
            rows.iter()
                .map(|r| {
                    let id: String = r.get(0);
                    let mn: Option<f64> = r.get(1);
                    let mx: Option<f64> = r.get(2);
                    HeatmapClassDomain {
                        id,
                        min: mn.unwrap_or(0.0),
                        max: mx.unwrap_or(1.0),
                        property: Some(w.to_string()),
                    }
                })
                .collect()
        }
        (None, Some(c)) => {
            let cq = quote_ident(c);
            let sql = format!(
                "SELECT {cq}::text AS cls FROM {source} WHERE {cq} IS NOT NULL \
                 GROUP BY {cq} ORDER BY {cq} LIMIT 8"
            );
            let rows = client.query(&sql, &[]).await.context("heatmap class enum")?;
            rows.iter()
                .map(|r| HeatmapClassDomain {
                    id: r.get(0),
                    min: 1.0,
                    max: 1.0,
                    property: None,
                })
                .collect()
        }
        (None, None) => unreachable!(),
    };
    Ok((!classes.is_empty()).then_some(HeatmapDomain { classes }))
}

/// DuckDB startup metadata aggregate (extent + time range + count) + optional
/// heatmap domain.
#[cfg(feature = "serve-duckdb")]
fn load_metadata_duckdb(
    pool: &DuckPool,
    args: &Args,
    source: &str,
    name: &str,
    bucket_ms: u64,
) -> Result<serde_json::Value> {
    let conn = pool.get().context("get DuckDB connection for metadata")?;
    let sql =
        duckdb_input::build_metadata_query(source, &args.geom_column, &args.time_field, args.time_format);
    let mut stmt = conn.prepare(&sql).context("prepare DuckDB metadata query")?;
    let mut rows = stmt.query([]).context("DuckDB metadata query")?;
    let row = rows
        .next()
        .context("DuckDB metadata query")?
        .context("metadata query returned no rows")?;
    let (min_lon, min_lat, max_lon, max_lat): (Option<f64>, Option<f64>, Option<f64>, Option<f64>) =
        (row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?);
    let (t_start, t_end, count): (Option<i64>, Option<i64>, i64) =
        (row.get(4)?, row.get(5)?, row.get(6)?);
    drop(rows);
    drop(stmt);
    let heatmap = load_heatmap_duckdb(
        &conn,
        source,
        args.heatmap_weight.as_deref(),
        args.heatmap_class.as_deref(),
    )
    .context("compute heatmap domain")?;
    metadata_json(
        name,
        args,
        bucket_ms,
        "stt-duckdb-dynamic",
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        t_start,
        t_end,
        count,
        heatmap,
    )
}

/// DuckDB heatmap-domain aggregate (mirrors the PostGIS `load_heatmap_pg` via
/// `quantile_cont`).
#[cfg(feature = "serve-duckdb")]
fn load_heatmap_duckdb(
    conn: &duckdb::Connection,
    source: &str,
    weight: Option<&str>,
    class: Option<&str>,
) -> Result<Option<HeatmapDomain>> {
    if weight.is_none() && class.is_none() {
        return Ok(None);
    }
    let classes: Vec<HeatmapClassDomain> = match (weight, class) {
        (Some(w), None) => {
            let wq = quote_ident(w);
            let sql = format!(
                "SELECT min({wq})::DOUBLE, quantile_cont({wq}, 0.95)::DOUBLE \
                 FROM {source} WHERE {wq} IS NOT NULL"
            );
            let mut stmt = conn.prepare(&sql).context("prepare heatmap aggregate")?;
            let mut rows = stmt.query([]).context("heatmap aggregate")?;
            match rows.next().context("heatmap aggregate")? {
                Some(row) => {
                    let mn: Option<f64> = row.get(0)?;
                    let mx: Option<f64> = row.get(1)?;
                    match mn {
                        Some(min) => vec![HeatmapClassDomain {
                            id: "default".to_string(),
                            min,
                            max: mx.unwrap_or(1.0),
                            property: Some(w.to_string()),
                        }],
                        None => Vec::new(),
                    }
                }
                None => Vec::new(),
            }
        }
        (Some(w), Some(c)) => {
            let wq = quote_ident(w);
            let cq = quote_ident(c);
            let sql = format!(
                "SELECT {cq}::VARCHAR AS cls, min({wq})::DOUBLE, quantile_cont({wq}, 0.95)::DOUBLE \
                 FROM {source} WHERE {wq} IS NOT NULL AND {cq} IS NOT NULL \
                 GROUP BY {cq} ORDER BY {cq} LIMIT 8"
            );
            let mut stmt = conn.prepare(&sql).context("prepare heatmap class aggregate")?;
            let mut rows = stmt.query([]).context("heatmap class aggregate")?;
            let mut out = Vec::new();
            while let Some(row) = rows.next().context("heatmap class row")? {
                let id: String = row.get(0)?;
                let mn: Option<f64> = row.get(1)?;
                let mx: Option<f64> = row.get(2)?;
                out.push(HeatmapClassDomain {
                    id,
                    min: mn.unwrap_or(0.0),
                    max: mx.unwrap_or(1.0),
                    property: Some(w.to_string()),
                });
            }
            out
        }
        (None, Some(c)) => {
            let cq = quote_ident(c);
            let sql = format!(
                "SELECT {cq}::VARCHAR AS cls FROM {source} WHERE {cq} IS NOT NULL \
                 GROUP BY {cq} ORDER BY {cq} LIMIT 8"
            );
            let mut stmt = conn.prepare(&sql).context("prepare heatmap class enum")?;
            let mut rows = stmt.query([]).context("heatmap class enum")?;
            let mut out = Vec::new();
            while let Some(row) = rows.next().context("heatmap class row")? {
                out.push(HeatmapClassDomain {
                    id: row.get(0)?,
                    min: 1.0,
                    max: 1.0,
                    property: None,
                });
            }
            out
        }
        (None, None) => unreachable!(),
    };
    Ok((!classes.is_empty()).then_some(HeatmapDomain { classes }))
}

/// Build the DuckDB r2d2 pool (read-only for a real file; in-memory otherwise).
#[cfg(feature = "serve-duckdb")]
fn build_duckdb_pool(path: &str, pool_size: usize) -> Result<DuckPool> {
    let inner = if path.is_empty() || path == ":memory:" {
        tracing::warn!(
            "DuckDB :memory: — pooled connections share ONE in-memory database (try_clone) that \
             starts empty, so only a --sql scanning external files (e.g. read_parquet) works; a \
             pre-existing table name will not be found"
        );
        DuckdbConnectionManager::memory().context("create in-memory DuckDB manager")?
    } else {
        let cfg = DuckConfig::default()
            .access_mode(AccessMode::ReadOnly)
            .context("configure DuckDB read-only access")?;
        DuckdbConnectionManager::file_with_flags(path, cfg)
            .with_context(|| format!("open DuckDB database '{path}' (read-only)"))?
    };
    R2d2Pool::builder()
        .max_size(pool_size.max(1) as u32)
        .build(SpatialDuckManager { inner })
        .context("build DuckDB connection pool (loads the spatial extension)")
}

/// Resolve `--table` / `--sql` into a single FROM source string and a name. A
/// `--sql` source is wrapped as a parenthesised, aliased subquery so it slots
/// into `SELECT … FROM {source}` and `FROM ( SELECT * FROM {source} ) q`.
fn resolve_source(args: &Args) -> Result<(String, String)> {
    match (&args.table, &args.sql) {
        (Some(_), Some(_)) => anyhow::bail!("--table and --sql are mutually exclusive"),
        (Some(t), None) => Ok((t.clone(), t.rsplit('.').next().unwrap_or(t).to_string())),
        (None, Some(s)) => Ok((format!("( {s} ) AS __stt_src"), "query".to_string())),
        (None, None) => anyhow::bail!("provide --table <NAME> or --sql <SELECT>"),
    }
}

/// Build the full per-tile [`TileConfig`] from the CLI flags, sharing the exact
/// builders the offline CLI uses, plus a resolved [`EncoderConfig`] passed
/// EXPLICITLY to the encoder per request (no process-wide globals — so the
/// server never mutates shared state and can host several datasets/configs).
/// Rejects the features that can't be served per single-tile request.
fn build_config(
    args: &Args,
    bucket_ms: u64,
) -> Result<(TileConfig, stt_core::arrow_tile::EncoderConfig)> {
    if let Some(scheme) = &args.summary_tier {
        anyhow::bail!(
            "--summary-tier ({scheme}) is not servable per request (it aggregates across many \
             spatial tiles). Pre-bake it with `stt-build --summary-tier` and serve the static \
             archive."
        );
    }
    if args.adaptive_temporal.is_some() {
        anyhow::bail!(
            "--adaptive-temporal is not servable per request (windows are sized across a cell's \
             whole time range). Pre-bake it with `stt-build --adaptive-temporal`."
        );
    }

    // Resolve encoder settings into an EXPLICIT config (parses + validates the
    // specs) — no process-wide globals are touched.
    let encoder = EncoderSettings {
        vertex_time_precision: Some(args.vertex_time_precision),
        quantize_coords_m: args.quantize_coords,
        quantize_attr: args.quantize_attr.clone(),
        quantize_attrs_auto: args.quantize_attrs_auto,
        vector_group: args.vector_group.clone(),
        point_elevation_column: args.point_elevation_column.clone(),
    }
    .resolve()?;
    let mut enc_enabled: Vec<String> = Vec::new();
    if let Some(m) = encoder.quantize_coords_m {
        enc_enabled.push(format!("quantize-coords={m}m"));
    }
    if !encoder.quantize_attrs.is_empty() {
        enc_enabled.push(format!("quantize-attr={}", encoder.quantize_attrs.len()));
    }
    if encoder.quantize_attrs_auto {
        enc_enabled.push("quantize-attrs-auto".into());
    }
    if !encoder.vector_groups.is_empty() {
        enc_enabled.push(format!("vector-groups={}", encoder.vector_groups.len()));
    }
    if !encoder.point_elevation_column.is_empty() {
        enc_enabled.push(format!("point-elevation-column={}", encoder.point_elevation_column));
    }
    if encoder.vertex_time_max_step_ms != stt_core::arrow_tile::DEFAULT_VERTEX_TIME_MAX_STEP_MS {
        enc_enabled.push(format!("vertex-time-precision={}ms", encoder.vertex_time_max_step_ms));
    }
    if !enc_enabled.is_empty() {
        tracing::info!("Encoder settings ENABLED: {}", enc_enabled.join(", "));
    }

    // Temporal LOD: parse + validate against the base bucket (same invariants
    // the offline build enforces).
    let temporal_lod = match args.temporal_lod.as_deref() {
        Some(s) => build_options::parse_temporal_lod(s, args.max_zoom)?,
        None => Vec::new(),
    };
    let mut prev: Option<u64> = None;
    for (i, l) in temporal_lod.iter().enumerate() {
        anyhow::ensure!(
            l.bucket_ms > bucket_ms,
            "--temporal-lod[{i}] bucket ({}) must be > base bucket ({bucket_ms})",
            l.bucket_ms
        );
        anyhow::ensure!(
            l.bucket_ms % bucket_ms == 0,
            "--temporal-lod[{i}] bucket ({}) must be a multiple of the base bucket ({bucket_ms})",
            l.bucket_ms
        );
        anyhow::ensure!(
            prev.map(|p| l.bucket_ms > p).unwrap_or(true),
            "--temporal-lod must be sorted by ascending bucket size"
        );
        prev = Some(l.bucket_ms);
    }

    // Attribute filter (with the same required-column guard the CLI applies).
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
    let attribute_filter = build_options::build_attribute_filter(
        &args.exclude,
        &args.include,
        args.exclude_all,
        &required,
    )?;

    let tile_budget = build_options::build_tile_budget(
        args.maximum_tile_bytes,
        args.maximum_tile_features,
        args.drop_densest_as_needed,
    );

    // Log the active parity-affecting tile config so an operator can confirm the
    // server is configured to match the offline build that produced the static
    // archive (silent config drift was the original serve-parity hazard).
    let mut active: Vec<String> = Vec::new();
    if args.no_clip {
        active.push("no-clip".into());
    } else if args.clip_min_vertices != 2 {
        active.push(format!("clip-min-vertices={}", args.clip_min_vertices));
    }
    if args.simplify {
        active.push(format!("simplify(max-zoom={})", args.simplify_max_zoom));
    }
    if args.time_aware_simplify {
        active.push("time-aware-simplify".into());
    }
    if args.pre_tessellate {
        active.push("pre-tessellate".into());
    }
    if let Some(z) = &args.min_zoom_field {
        active.push(format!("min-zoom-field={z}"));
    }
    if let Some(z) = &args.max_zoom_field {
        active.push(format!("max-zoom-field={z}"));
    }
    if args.min_features_per_tile != 1 {
        active.push(format!("min-features-per-tile={}", args.min_features_per_tile));
    }
    if tile_budget.is_some() {
        active.push("tile-budget".into());
    }
    if !attribute_filter.is_keep_all() {
        active.push("attribute-filter".into());
    }
    if !temporal_lod.is_empty() {
        active.push(format!("temporal-lod({} levels)", temporal_lod.len()));
    }
    tracing::info!(
        "Serve tile config: base-bucket={bucket_ms}ms zoom={}..={}{}",
        args.min_zoom,
        args.max_zoom,
        if active.is_empty() {
            String::new()
        } else {
            format!(", active: {}", active.join(", "))
        },
    );

    let config = TileConfig {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        layer_name: args.layer.clone(),
        temporal_bucket_ms: bucket_ms,
        clip_trajectories: !args.no_clip,
        clip_min_vertices: args.clip_min_vertices,
        simplify: args.simplify,
        simplify_max_zoom: args.simplify_max_zoom,
        pre_tessellate: args.pre_tessellate,
        temporal_lod,
        min_features_per_tile: args.min_features_per_tile,
        time_aware_simplify: args.time_aware_simplify,
        // Rejected above, so always inert here (single-tile serving can't
        // reproduce adaptive temporal windowing).
        adaptive_target_features: None,
        min_zoom_field: args.min_zoom_field.clone(),
        max_zoom_field: args.max_zoom_field.clone(),
        tile_budget,
        attribute_filter,
        // DB-backed serving keeps per-tile type sniffing for now — the row
        // decode knows the column types but doesn't thread them here yet
        // (see ColumnarOptions::property_types).
        property_types: Default::default(),
    };
    Ok((config, encoder))
}

/// Build one dataset's [`ServerState`] from its (per-dataset) `Args`: resolve the
/// source, build the `TileConfig` + explicit `EncoderConfig`, open the backend
/// pool, and load the startup metadata. Returns the resolved dataset name (the
/// `--name` override or the table/query-derived name) alongside the state.
async fn build_dataset_state(args: &Args) -> Result<(String, ServerState)> {
    #[cfg(all(feature = "serve-postgres", feature = "serve-duckdb"))]
    if args.postgres.is_some() && args.duckdb.is_some() {
        anyhow::bail!("--postgres and --duckdb are mutually exclusive");
    }
    let bucket_ms = build_options::parse_duration(&args.temporal_bucket)?;
    let (source, derived_name) = resolve_source(args)?;
    let name = args.name.clone().unwrap_or(derived_name);
    let (config, encoder) = build_config(args, bucket_ms)?;

    // Resolve the backend: explicit per-dataset flags first, then env
    // fallbacks. DuckDB is tried first (but an explicit --postgres outranks the
    // STT_DUCKDB_PATH env fallback), matching the pre-gating precedence.
    let mut resolved: Option<(Backend, serde_json::Value)> = None;

    #[cfg(feature = "serve-duckdb")]
    if resolved.is_none() {
        let duck_path = args
            .duckdb
            .clone()
            .or_else(|| std::env::var("STT_DUCKDB_PATH").ok());
        #[cfg(feature = "serve-postgres")]
        let duck_path = duck_path.filter(|_| args.postgres.is_none());
        if let Some(path) = duck_path {
            let pool = build_duckdb_pool(&path, args.pool_size)?;
            let metadata = load_metadata_duckdb(&pool, args, &source, &name, bucket_ms)
                .context("load DuckDB source metadata")?;
            resolved = Some((Backend::DuckDb(pool), metadata));
        }
    }

    #[cfg(feature = "serve-postgres")]
    if resolved.is_none() {
        let pg_conn = args
            .postgres
            .clone()
            .or_else(|| std::env::var("STT_POSTGRES_URL").ok())
            .or_else(|| std::env::var("DATABASE_URL").ok());
        if let Some(conn) = pg_conn {
            let pg_config: tokio_postgres::Config = conn.parse().context("parse connection string")?;
            let mgr = Manager::from_config(
                pg_config,
                NoTls,
                ManagerConfig {
                    recycling_method: RecyclingMethod::Fast,
                },
            );
            let pool = PgPool::builder(mgr)
                .max_size(args.pool_size)
                .build()
                .context("build connection pool")?;
            let metadata = load_metadata_pg(&pool, args, &source, &name, bucket_ms)
                .await
                .context("load PostGIS source metadata")?;
            resolved = Some((Backend::Postgres(pool), metadata));
        }
    }

    // The "no source" hint only names the flags/envs of compiled-in backends.
    #[cfg(all(feature = "serve-postgres", feature = "serve-duckdb"))]
    const NO_SOURCE: &str = "no source: pass --postgres <CONN> or --duckdb <PATH> (or set \
                             STT_POSTGRES_URL / DATABASE_URL / STT_DUCKDB_PATH)";
    #[cfg(all(feature = "serve-postgres", not(feature = "serve-duckdb")))]
    const NO_SOURCE: &str =
        "no source: pass --postgres <CONN> (or set STT_POSTGRES_URL / DATABASE_URL)";
    #[cfg(all(feature = "serve-duckdb", not(feature = "serve-postgres")))]
    const NO_SOURCE: &str = "no source: pass --duckdb <PATH> (or set STT_DUCKDB_PATH)";

    let (backend, mut metadata) = match resolved {
        Some(v) => v,
        None => anyhow::bail!(NO_SOURCE),
    };

    // Advertise the temporal-LOD pyramid so a client can discover the coarser
    // tiers the server serves at low zooms. Keyed `temporalLod` to keep the
    // /metadata.json object uniformly camelCase (loaders.gl TileSource style);
    // this is the dynamic serve descriptor, NOT the packed manifest (whose
    // canonical key is snake_case `temporal_lod`).
    if !config.temporal_lod.is_empty() {
        metadata["temporalLod"] =
            serde_json::to_value(&config.temporal_lod).context("serialize temporal LOD levels")?;
    }

    Ok((
        name,
        ServerState {
            backend,
            source,
            geom_column: args.geom_column.clone(),
            time_field: args.time_field.clone(),
            end_time_field: args.end_time_field.clone(),
            time_format: args.time_format,
            config,
            encoder,
            metadata,
        },
    ))
}

/// Bind + serve `app` with graceful ctrl-c shutdown.
async fn run_server(app: Router, bind: std::net::SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("server error")?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "stt_serve=info".into()),
        )
        .init();

    let args = Args::parse();

    // Multi-dataset mode: a `--config` JSON file describing N datasets, each
    // served under `/{name}/…`. Each dataset carries its own explicit
    // `EncoderConfig` (never the process-wide globals), so datasets with
    // different quantization / vector groups coexist in one process.
    if let Some(config_path) = &args.config {
        let text = std::fs::read_to_string(config_path)
            .with_context(|| format!("read --config {}", config_path.display()))?;
        let cfg: ServeConfig = serde_json::from_str(&text)
            .with_context(|| format!("parse --config {}", config_path.display()))?;
        anyhow::ensure!(!cfg.datasets.is_empty(), "--config has no datasets");

        let mut datasets: HashMap<String, Arc<ServerState>> = HashMap::new();
        for ds in &cfg.datasets {
            let (name, mut state) = build_dataset_state(ds)
                .await
                .with_context(|| format!("dataset '{}'", ds.name.as_deref().unwrap_or("?")))?;
            anyhow::ensure!(
                !datasets.contains_key(&name),
                "duplicate dataset name '{name}' — set a distinct \"name\" per dataset"
            );
            // Multi-dataset tiles are routed under `/{dataset}/tiles/…`, so the
            // advertised template must carry the dataset prefix (the root
            // `/tiles/…` template only exists in single-dataset mode).
            state.metadata["tileUrlTemplate"] =
                serde_json::Value::String(format!("/{name}/tiles/{{z}}/{{x}}/{{y}}/{{t}}.stt"));
            tracing::info!("dataset '{name}' ready ({} zoom {}..={})", ds.temporal_bucket, ds.min_zoom, ds.max_zoom);
            datasets.insert(name, Arc::new(state));
        }
        let n = datasets.len();
        let registry = Arc::new(Registry { datasets });
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/datasets", get(datasets_handler))
            .route("/:dataset/metadata.json", get(metadata_handler_multi))
            .route("/:dataset/tiles/:z/:x/:y/:t", get(tile_handler_multi))
            .with_state(registry);
        tracing::info!(
            "stt-serve listening on http://{} — {n} dataset(s): GET /{{dataset}}/tiles/{{z}}/{{x}}/{{y}}/{{t}}.stt, GET /datasets",
            args.bind
        );
        return run_server(app, args.bind).await;
    }

    // Single-dataset mode: root routes, backward-compatible.
    let (_name, state) = build_dataset_state(&args).await?;
    let state = Arc::new(state);
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/metadata.json", get(metadata_handler))
        .route("/tiles/:z/:x/:y/:t", get(tile_handler))
        .with_state(state);
    tracing::info!(
        "stt-serve listening on http://{} — GET /tiles/{{z}}/{{x}}/{{y}}/{{t}}.stt",
        args.bind
    );
    run_server(app, args.bind).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_bbox_covers_expected_lonlat() {
        let b = tile_bbox(0, 0, 0);
        assert!(b[0] <= -180.0 && b[2] >= 180.0);
        let (x, y) = (163u32, 395u32);
        let b = tile_bbox(10, x, y);
        assert!(b[0] < b[2] && b[1] < b[3]);
        assert!(b[0] > -130.0 && b[2] < -110.0);
    }

    #[test]
    fn resolve_source_table_and_sql() {
        let mut a = Args::parse_from(["stt-serve", "--table", "public.obs"]);
        let (src, name) = resolve_source(&a).unwrap();
        assert_eq!(src, "public.obs");
        assert_eq!(name, "obs");

        a = Args::parse_from(["stt-serve", "--sql", "SELECT * FROM v WHERE ok"]);
        let (src, name) = resolve_source(&a).unwrap();
        assert!(src.contains("SELECT * FROM v WHERE ok"));
        assert!(src.contains("__stt_src"));
        assert_eq!(name, "query");

        a = Args::parse_from(["stt-serve", "--table", "t", "--sql", "SELECT 1"]);
        assert!(resolve_source(&a).is_err());
    }

    #[test]
    fn build_config_rejects_unservable() {
        let a = Args::parse_from(["stt-serve", "--table", "t", "--summary-tier", "h3"]);
        assert!(build_config(&a, 3_600_000).is_err());
        let a = Args::parse_from(["stt-serve", "--table", "t", "--adaptive-temporal", "1000"]);
        assert!(build_config(&a, 3_600_000).is_err());
    }

    // References both backends' `--config` keys/fields, so it needs both
    // compiled in (a compiled-out backend's key is an unknown field).
    #[cfg(all(feature = "serve-postgres", feature = "serve-duckdb"))]
    #[test]
    fn config_json_deserializes_datasets_with_kebab_keys_and_clap_defaults() {
        // Config-file keys mirror the CLI flag names (kebab-case); any omitted
        // field falls back to the clap default (single source of truth).
        let json = r#"{
            "datasets": [
                { "name": "quakes", "postgres": "postgresql://x", "table": "obs",
                  "time-field": "iso_time", "temporal-bucket": "1d", "quantize-coords": 1.0 },
                { "name": "trips", "duckdb": "t.duckdb", "sql": "SELECT * FROM t",
                  "geom-column": "the_geom", "time-format": "unix-ms", "min-zoom": 2, "max-zoom": 10 }
            ]
        }"#;
        let cfg: ServeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.datasets.len(), 2);

        let d0 = &cfg.datasets[0];
        assert_eq!(d0.name.as_deref(), Some("quakes"));
        assert_eq!(d0.table.as_deref(), Some("obs"));
        assert_eq!(d0.time_field, "iso_time");
        assert_eq!(d0.temporal_bucket, "1d");
        assert_eq!(d0.quantize_coords, 1.0);
        // Omitted → clap defaults.
        assert_eq!(d0.geom_column, "geom");
        assert_eq!(d0.max_zoom, 14);
        assert!(matches!(d0.time_format, TimeFormat::Iso8601));

        let d1 = &cfg.datasets[1];
        assert_eq!(d1.duckdb.as_deref(), Some("t.duckdb"));
        assert_eq!(d1.geom_column, "the_geom");
        assert!(matches!(d1.time_format, TimeFormat::UnixMs));
        assert_eq!((d1.min_zoom, d1.max_zoom), (2, 10));
    }

    #[test]
    fn config_json_rejects_unknown_field() {
        // deny_unknown_fields catches a typo'd key rather than silently ignoring it.
        let json = r#"{ "datasets": [ { "table": "obs", "tmeporal-bucket": "1d" } ] }"#;
        assert!(serde_json::from_str::<ServeConfig>(json).is_err());
    }

    #[test]
    fn metadata_json_is_uniformly_camel_case() {
        // /metadata.json is a loaders.gl TileSource-style runtime descriptor:
        // every top-level key (core AND advisory) must be camelCase so the
        // object has one casing convention. This locks the heatmap_domain →
        // heatmapDomain / temporal_lod → temporalLod fix against regression.
        let args = Args::parse_from(["stt-serve", "--table", "t"]);
        let domain = HeatmapDomain {
            classes: vec![HeatmapClassDomain {
                id: "pickup".into(),
                min: 0.0,
                max: 1.0,
                property: None,
            }],
        };
        let meta = metadata_json(
            "obs",
            &args,
            3_600_000,
            "duckdb",
            Some(-1.0),
            Some(-2.0),
            Some(1.0),
            Some(2.0),
            Some(0),
            Some(1000),
            42,
            Some(domain),
        )
        .expect("metadata_json");
        let obj = meta.as_object().expect("metadata is a JSON object");
        for key in obj.keys() {
            assert!(
                !key.contains('_'),
                "metadata.json key '{key}' is not camelCase (the whole descriptor must be)"
            );
        }
        // The advisory heatmap key specifically must be the camelCase spelling.
        assert!(obj.contains_key("heatmapDomain"), "keys: {:?}", obj.keys().collect::<Vec<_>>());
        assert!(!obj.contains_key("heatmap_domain"));
    }

    #[test]
    fn build_config_validates_temporal_lod() {
        // 30m is NOT a multiple of a 1h base → error.
        let a = Args::parse_from(["stt-serve", "--table", "t", "--temporal-lod", "30m"]);
        assert!(build_config(&a, 3_600_000).is_err());
        // 1d,7d on a 1h base is valid.
        let a = Args::parse_from(["stt-serve", "--table", "t", "--temporal-lod", "1d,7d"]);
        let (cfg, _enc) = build_config(&a, 3_600_000).expect("valid lod");
        assert_eq!(cfg.temporal_lod.len(), 2);
    }

    /// Doc gate (naming-types-consistency F9): every visible long flag must
    /// appear in the `stt-serve` section of `docs/api/cli-reference.md`, so a
    /// new flag fails the build until it is documented.
    #[test]
    fn cli_flags_are_documented_in_cli_reference() {
        use clap::CommandFactory;
        let doc = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/api/cli-reference.md"
        ))
        .expect("read docs/api/cli-reference.md");
        let start = doc.find("## `stt-serve`").expect("stt-serve section heading");
        let body = &doc[start + 1..];
        let end = body.find("\n## `").map(|i| start + 1 + i).unwrap_or(doc.len());
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
            "flags missing from the `stt-serve` section of docs/api/cli-reference.md \
             (document them before shipping): {missing:?}"
        );
    }
}
