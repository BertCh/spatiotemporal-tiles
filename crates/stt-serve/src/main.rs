//! stt-serve — a dynamic STT tile server.
//!
//! Generates spatiotemporal tiles on the fly from a live **PostGIS** table or a
//! **DuckDB** database, one `(z, x, y, t)` per request — the `ST_AsMVT` analog
//! for the STT format. Each request:
//!   1. maps `(z, x, y)` to a WGS84 bounding box and `t` to a temporal bucket,
//!   2. runs a source query filtered by the bbox (spatial index) + time window,
//!   3. decodes the rows to features and encodes ONE STT tile blob
//!      (`stt_build::encode_single_tile`) — Arrow IPC + GeoArrow, per-blob zstd,
//!      byte-identical to what the offline build would emit for that tile.
//!
//! The backend is chosen by `--postgres <CONN>` or `--duckdb <PATH>`. PostGIS
//! uses an async client + pool; DuckDB is embedded and blocking, so its pool
//! checkout / query / decode / encode all run on `spawn_blocking`.
//!
//! This is a benchmark / reference server: tiles are regenerated per request
//! (no app cache), so it measures raw source-to-tile latency. Unlike the
//! pre-baked packed format it is NOT edge-cacheable — that is the live-source
//! trade-off.

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
use stt_build::input::{InputStrictness, ParsedFeature, TimeFormat};
use stt_build::tiler::TileConfig;

// Postgres backend.
use deadpool_postgres::{Manager, ManagerConfig, Pool as PgPool, RecyclingMethod};
use stt_build::postgres_input;
use tokio_postgres::NoTls;

// DuckDB backend.
use duckdb::{AccessMode, Config as DuckConfig, DuckdbConnectionManager};
use r2d2::Pool as R2d2Pool;
use stt_build::duckdb_input;

#[derive(Parser, Clone)]
#[command(name = "stt-serve", version)]
#[command(
    about = "Serve STT tiles dynamically from a live PostGIS table or a DuckDB database",
    long_about = None
)]
struct Args {
    /// PostgreSQL/PostGIS connection string (libpq URI or key=value). Env
    /// fallback: STT_POSTGRES_URL, then DATABASE_URL. Mutually exclusive with
    /// --duckdb.
    #[arg(long)]
    postgres: Option<String>,

    /// DuckDB database file path (or `:memory:` with a `--table` that scans
    /// external files, e.g. a parenthesised `read_parquet(...)` subquery). Env
    /// fallback: STT_DUCKDB_PATH. Mutually exclusive with --postgres.
    #[arg(long)]
    duckdb: Option<String>,

    /// Source table (optionally schema-qualified, e.g. public.hurricane_obs).
    /// For DuckDB this may also be a parenthesised subquery / view.
    #[arg(long)]
    table: String,

    /// Geometry column (must be EPSG:4326 lon/lat).
    #[arg(long, default_value = "geom")]
    geom_column: String,

    /// Timestamp column (a timestamp / timestamptz type).
    #[arg(long, default_value = "ts")]
    time_field: String,

    /// Optional end-timestamp column for features with a time range.
    #[arg(long)]
    end_time_field: Option<String>,

    /// Wire format of an INTEGER time column (ignored for timestamp columns).
    #[arg(long, value_enum, default_value = "iso8601")]
    time_format: TimeFormat,

    /// Minimum / maximum zoom advertised in /metadata.json.
    #[arg(long, default_value = "0")]
    min_zoom: u8,
    #[arg(long, default_value = "14")]
    max_zoom: u8,

    /// Temporal bucket size (e.g. 1h, 30m, 6h, 1d) — must match how clients
    /// address tiles in time.
    #[arg(long, default_value = "1h")]
    temporal_bucket: String,

    /// Layer name embedded in each tile.
    #[arg(long, default_value = "default")]
    layer: String,

    /// Disable trajectory clipping (store whole LineStrings in their centroid
    /// tile). No effect on point/polygon data.
    #[arg(long)]
    no_clip: bool,

    /// Connection-pool size.
    #[arg(long, default_value = "8")]
    pool_size: usize,

    /// Listen address.
    #[arg(long, default_value = "127.0.0.1:8088")]
    bind: SocketAddr,
}

/// r2d2 manager that loads the DuckDB `spatial` extension and pins UTC on every
/// *new physical connection* (once, not per checkout), so `ST_AsWKB` /
/// `ST_Intersects` / `epoch_ms` are available and epoch math is tz-independent.
struct SpatialDuckManager {
    inner: DuckdbConnectionManager,
}

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

type DuckPool = R2d2Pool<SpatialDuckManager>;

/// The selected data source.
enum Backend {
    Postgres(PgPool),
    DuckDb(DuckPool),
}

/// Shared, immutable server state.
struct ServerState {
    backend: Backend,
    table: String,
    geom_column: String,
    time_field: String,
    end_time_field: Option<String>,
    time_format: TimeFormat,
    config: TileConfig,
    metadata: serde_json::Value,
}

/// Parse a duration string like `1h`, `30m`, `6h`, `1d`, `90s` to milliseconds.
fn parse_bucket_ms(s: &str) -> Result<u64> {
    let s = s.trim();
    let (num, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
    let n: u64 = num.parse().with_context(|| format!("invalid duration '{s}'"))?;
    let mult = match unit {
        "ms" => 1,
        "s" | "" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        other => anyhow::bail!("unknown duration unit '{other}' (use ms/s/m/h/d)"),
    };
    Ok(n.saturating_mul(mult).max(1))
}

/// Tile `(z, x, y)` → WGS84 `[min_lon, min_lat, max_lon, max_lat]` (Web
/// Mercator slippy formula) with a small buffer so a feature near a tile edge
/// is never missed by the SQL pre-filter. `encode_single_tile` performs the
/// authoritative, exact per-tile placement afterward.
fn tile_bbox(z: u8, x: u32, y: u32) -> [f64; 4] {
    let n = (1u64 << z) as f64;
    let min_lon = x as f64 / n * 360.0 - 180.0;
    let max_lon = (x + 1) as f64 / n * 360.0 - 180.0;
    let lat = |yy: f64| (std::f64::consts::PI * (1.0 - 2.0 * yy / n)).sinh().atan().to_degrees();
    let max_lat = lat(y as f64);
    let min_lat = lat((y + 1) as f64);
    let bx = (max_lon - min_lon).abs() * 0.05;
    let by = (max_lat - min_lat).abs() * 0.05;
    [min_lon - bx, min_lat - by, max_lon + bx, max_lat + by]
}

async fn tile_handler(
    State(st): State<Arc<ServerState>>,
    Path((z, x, y, t)): Path<(u8, u32, u32, String)>,
) -> Response {
    let t_ms = match t.trim_end_matches(".stt").parse::<i64>() {
        Ok(v) => v,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "t must be an integer (ms since epoch)").into_response()
        }
    };
    match build_tile(&st, z, x, y, t_ms).await {
        Ok(Some((bytes, micros))) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-stt-tile")
            .header(header::CACHE_CONTROL, "no-store")
            .header("x-stt-gen-micros", micros.to_string())
            .body(Body::from(bytes))
            .unwrap(),
        // Empty tile — no features in this (z, x, y, bucket).
        Ok(None) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            tracing::warn!("tile {z}/{x}/{y}/{t_ms} failed: {e:#}");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response()
        }
    }
}

/// Query the source for one tile's candidate features and encode the tile.
/// Returns `(bytes, generation_micros)`; `None` when the tile is empty.
async fn build_tile(
    st: &Arc<ServerState>,
    z: u8,
    x: u32,
    y: u32,
    t_ms: i64,
) -> Result<Option<(Vec<u8>, u128)>> {
    let started = Instant::now();
    let bucket_ms = st.config.temporal_bucket_ms.max(1);
    let bucket_start = (t_ms.max(0) as u64 / bucket_ms) * bucket_ms;
    let bbox = tile_bbox(z, x, y);
    let t_start = bucket_start as i64;
    let t_end = (bucket_start + bucket_ms) as i64;

    let bytes: Option<Vec<u8>> = match &st.backend {
        Backend::Postgres(pool) => {
            let sql = postgres_input::build_tile_query(
                &st.table,
                &st.geom_column,
                &st.time_field,
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
                encode(&features, z, x, y, t_ms, &st.config)
            })
            .await
            .context("encode task join")??
        }
        Backend::DuckDb(pool) => {
            let sql = duckdb_input::build_tile_query(
                &st.table,
                &st.geom_column,
                &st.time_field,
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
                encode(&features, z, x, y, t_ms, &st.config)
            })
            .await
            .context("encode task join")??
        }
    };

    Ok(bytes.map(|b| (b, started.elapsed().as_micros())))
}

/// Encode features into one STT tile blob, or `None` if there are none.
fn encode(
    features: &[ParsedFeature],
    z: u8,
    x: u32,
    y: u32,
    t_ms: i64,
    config: &TileConfig,
) -> Result<Option<Vec<u8>>> {
    if features.is_empty() {
        return Ok(None);
    }
    stt_build::encode_single_tile(features, z, x, y, t_ms, config)
}

async fn metadata_handler(State(st): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    Json(st.metadata.clone())
}

/// Assemble the `ArchiveMetadata`-shaped JSON for /metadata.json from resolved
/// source extent values.
#[allow(clippy::too_many_arguments)]
fn metadata_json(
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
) -> serde_json::Value {
    serde_json::json!({
        "format": format,
        "name": args.table,
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
    })
}

/// PostGIS startup metadata aggregate (extent + time range + count).
async fn load_metadata_pg(pool: &PgPool, args: &Args, bucket_ms: u64) -> Result<serde_json::Value> {
    let client = pool.get().await.context("get connection for metadata")?;
    let geom = format!("\"{}\"", args.geom_column.replace('"', "\"\""));
    let time = format!("\"{}\"", args.time_field.replace('"', "\"\""));
    let sql = format!(
        "SELECT ST_XMin(ext), ST_YMin(ext), ST_XMax(ext), ST_YMax(ext), tmin, tmax, cnt \
         FROM ( \
            SELECT ST_Extent({geom}) AS ext, \
                   (EXTRACT(EPOCH FROM MIN({time})) * 1000.0)::float8 AS tmin, \
                   (EXTRACT(EPOCH FROM MAX({time})) * 1000.0)::float8 AS tmax, \
                   COUNT(*) AS cnt \
            FROM {table} \
         ) q",
        table = args.table,
    );
    let row = client.query_one(&sql, &[]).await.context("metadata aggregate query")?;
    let t_start: Option<f64> = row.get(4);
    let t_end: Option<f64> = row.get(5);
    Ok(metadata_json(
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
    ))
}

/// DuckDB startup metadata aggregate (extent + time range + count).
fn load_metadata_duckdb(pool: &DuckPool, args: &Args, bucket_ms: u64) -> Result<serde_json::Value> {
    let conn = pool.get().context("get DuckDB connection for metadata")?;
    let sql = duckdb_input::build_metadata_query(&args.table, &args.geom_column, &args.time_field);
    let mut stmt = conn.prepare(&sql).context("prepare DuckDB metadata query")?;
    let mut rows = stmt.query([]).context("DuckDB metadata query")?;
    let row = rows
        .next()
        .context("DuckDB metadata query")?
        .context("metadata query returned no rows")?;
    Ok(metadata_json(
        args,
        bucket_ms,
        "stt-duckdb-dynamic",
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
    ))
}

/// Build the DuckDB r2d2 pool (read-only for a real file; in-memory otherwise).
fn build_duckdb_pool(path: &str, pool_size: usize) -> Result<DuckPool> {
    let inner = if path.is_empty() || path == ":memory:" {
        tracing::warn!(
            "DuckDB :memory: — pooled connections share ONE in-memory database (try_clone) that \
             starts empty, so only a --table/--sql scanning external files (e.g. read_parquet) \
             works; a pre-existing table name will not be found"
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "stt_serve=info".into()),
        )
        .init();

    let args = Args::parse();
    if args.postgres.is_some() && args.duckdb.is_some() {
        anyhow::bail!("--postgres and --duckdb are mutually exclusive");
    }

    let bucket_ms = parse_bucket_ms(&args.temporal_bucket)?;

    // Resolve the backend: explicit flags first, then env fallbacks.
    let duck_path = args
        .duckdb
        .clone()
        .or_else(|| std::env::var("STT_DUCKDB_PATH").ok());
    let pg_conn = args
        .postgres
        .clone()
        .or_else(|| std::env::var("STT_POSTGRES_URL").ok())
        .or_else(|| std::env::var("DATABASE_URL").ok());

    let (backend, metadata) = if let Some(path) = duck_path.filter(|_| args.postgres.is_none()) {
        let pool = build_duckdb_pool(&path, args.pool_size)?;
        let metadata = load_metadata_duckdb(&pool, &args, bucket_ms)
            .context("load DuckDB source metadata")?;
        tracing::info!("DuckDB source '{path}' metadata: {metadata}");
        (Backend::DuckDb(pool), metadata)
    } else if let Some(conn) = pg_conn {
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
        let metadata = load_metadata_pg(&pool, &args, bucket_ms)
            .await
            .context("load PostGIS source metadata")?;
        tracing::info!("PostGIS source metadata: {metadata}");
        (Backend::Postgres(pool), metadata)
    } else {
        anyhow::bail!(
            "no source: pass --postgres <CONN> or --duckdb <PATH> (or set STT_POSTGRES_URL / \
             DATABASE_URL / STT_DUCKDB_PATH)"
        );
    };

    let config = TileConfig {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        layer_name: args.layer.clone(),
        temporal_bucket_ms: bucket_ms,
        clip_trajectories: !args.no_clip,
        ..TileConfig::default()
    };

    let state = Arc::new(ServerState {
        backend,
        table: args.table.clone(),
        geom_column: args.geom_column.clone(),
        time_field: args.time_field.clone(),
        end_time_field: args.end_time_field.clone(),
        time_format: args.time_format,
        config,
        metadata,
    });

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/metadata.json", get(metadata_handler))
        .route("/tiles/:z/:x/:y/:t", get(tile_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("bind {}", args.bind))?;
    tracing::info!(
        "stt-serve listening on http://{} — GET /tiles/{{z}}/{{x}}/{{y}}/{{t}}.stt  (bucket {} ms)",
        args.bind,
        bucket_ms
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("server error")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_durations() {
        assert_eq!(parse_bucket_ms("1h").unwrap(), 3_600_000);
        assert_eq!(parse_bucket_ms("30m").unwrap(), 1_800_000);
        assert_eq!(parse_bucket_ms("1d").unwrap(), 86_400_000);
        assert_eq!(parse_bucket_ms("500ms").unwrap(), 500);
        assert!(parse_bucket_ms("5w").is_err());
    }

    #[test]
    fn tile_bbox_covers_expected_lonlat() {
        let b = tile_bbox(0, 0, 0);
        assert!(b[0] <= -180.0 && b[2] >= 180.0);
        let (x, y) = (163u32, 395u32);
        let b = tile_bbox(10, x, y);
        assert!(b[0] < b[2] && b[1] < b[3]);
        assert!(b[0] > -130.0 && b[2] < -110.0);
    }
}
