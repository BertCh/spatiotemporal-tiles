//! DuckDB input source (cargo feature `duckdb`).
//!
//! Reads features directly from a DuckDB query — a `.duckdb` database file, or
//! `:memory:` against anything DuckDB can scan (Parquet/CSV/GeoJSON/… via
//! `read_parquet`/`read_csv_auto`/…) — instead of a GeoParquet file, producing
//! the exact same [`ParsedFeature`] stream the GeoParquet reader emits
//! ([`crate::input`]) — so the whole tiler/writer pipeline is reused unchanged.
//! This is the DuckDB sibling of [`crate::postgres_input`].
//!
//! The bridge is WKB: DuckDB's **spatial** extension `ST_AsWKB(geom)` returns
//! standard OGC WKB, which [`crate::input::parse_wkb_geometry`] already decodes
//! (via `geozero`'s `Ewkb`, which reads both plain WKB and SRID-prefixed EWKB).
//! We wrap the user's table / SQL as a subquery, project the geometry to a
//! `BLOB` WKB column, and read every other returned column as a feature
//! property. Column values are decoded from DuckDB's self-describing
//! [`duckdb::types::ValueRef`], so — unlike the PostGIS path — no per-type
//! schema introspection is needed.
//!
//! The spatial extension is a *separate downloadable* extension (not part of
//! the statically-bundled `libduckdb`); we `INSTALL spatial; LOAD spatial;` on
//! connect (a one-time network fetch, cached under `~/.duckdb` thereafter), and
//! pin the session to UTC so epoch math is timezone-independent.

use anyhow::{Context, Result};
use duckdb::types::{TimeUnit, ValueRef};
use duckdb::{AccessMode, Config, Connection, Row};
use geojson::Feature;
use std::sync::Arc;

use crate::db_input_common::{
    apply_int_time_format, decimal_string_to_json, json_number_or_null, warn_dropped_columns,
    RowOutcome, VertexCoercions,
};
use crate::input::{
    parse_iso8601, parse_wkb_geometry, reject_negative_timestamp, InputStrictness, ParsedFeature,
    TimeFormat,
};

/// Alias the wrapped query projects the geometry into (as a `BLOB` of WKB).
/// Public so a dynamic tile server building its own per-tile query keeps the
/// geometry column name that [`decode_query`] expects.
pub const WKB_ALIAS: &str = "__stt_wkb";
/// Default rows accumulated per `on_batch` flush in the streaming reader.
pub const DEFAULT_BATCH_SIZE: usize = 8192;

/// What to read from DuckDB — either a table (we `SELECT *`) or an arbitrary
/// `SELECT` the caller supplies (e.g. one that scans a Parquet file).
#[derive(Debug, Clone)]
pub enum QuerySource {
    /// A table or view (optionally schema-qualified, e.g. `main.trips`).
    Table(String),
    /// An arbitrary `SELECT …` statement.
    Sql(String),
}

/// Fully describes the DuckDB read: the source, which column holds geometry,
/// an optional `WHERE` filter, and optional reprojection to EPSG:4326.
#[derive(Debug, Clone)]
pub struct QuerySpec {
    pub source: QuerySource,
    pub geom_column: String,
    pub where_clause: Option<String>,
    /// When set, geometry is `ST_Transform`ed from this source SRID to 4326.
    /// Omit when the source geometry is already lon/lat (EPSG:4326).
    pub reproject_from_srid: Option<i32>,
}

impl QuerySpec {
    /// Quote a SQL identifier (double-quote, escaping embedded quotes).
    fn quote_ident(name: &str) -> String {
        format!("\"{}\"", name.replace('"', "\"\""))
    }

    /// Build the wrapped query: project the geometry to a WKB `BLOB` column
    /// aliased [`WKB_ALIAS`], and pass every other column through as `q.*`.
    fn wrapped_query(&self) -> String {
        let inner = match &self.source {
            // A bare table name may be schema-qualified; trust the operator.
            QuerySource::Table(t) => format!("SELECT * FROM {t}"),
            QuerySource::Sql(s) => s.clone(),
        };
        let geom = format!("q.{}", Self::quote_ident(&self.geom_column));
        let wkb_expr = match self.reproject_from_srid {
            // DuckDB GEOMETRY carries no per-row SRID (PostGIS divergence) and
            // there is no `ST_SetSRID`; the source CRS is passed to
            // `ST_Transform` as an explicit string. `always_xy => true` keeps
            // EPSG:4326 output as lon/lat (x,y) — without it PROJ honours the
            // authority's lat/lon axis order and swaps the coordinates.
            Some(srid) => {
                format!("ST_AsWKB(ST_Transform({geom}, 'EPSG:{srid}', 'EPSG:4326', true))")
            }
            None => format!("ST_AsWKB({geom})"),
        };
        let where_sql = self
            .where_clause
            .as_ref()
            .map(|w| format!(" WHERE {w}"))
            .unwrap_or_default();
        format!("SELECT {wkb_expr} AS {WKB_ALIAS}, q.* FROM ( {inner} ) AS q{where_sql}")
    }
}

/// Build a per-tile query for a dynamic server: project geometry to a WKB
/// `BLOB` aliased [`WKB_ALIAS`], pass every other column through, and filter by
/// a WGS84 bounding box (`ST_Intersects_Extent` — bounding-box overlap, the
/// PostGIS `&&` analog: a pure MBR test with no per-row GEOS work, and a
/// SUPERSET prefilter, so a feature the tiler places by centroid is never
/// dropped by an exact-intersection test; the authoritative per-tile placement
/// happens in `encode_single_tile`) and a half-open time window
/// `[t_start_ms, t_end_ms)`. The geometry / time literals are server-formatted
/// numbers (request `z/x/y/t` are parsed to integers upstream), so there is no
/// request-controlled SQL. The time column must be a `TIMESTAMP`/`TIMESTAMP
/// WITH TIME ZONE` (compared via `epoch_ms`, which yields a UTC `TIMESTAMP`;
/// pin the session to UTC).
///
/// `source_srid`: when the stored geometry is not EPSG:4326, it is
/// `ST_Transform`ed to 4326 *before* both the WKB projection and the bbox
/// test — the filter runs in tile space (still a strict superset), and the
/// output matches an offline `stt-build --source-srid` ingest exactly. The
/// per-row transform makes this a full-scan predicate; store 4326 if you need
/// the cheap path.
///
/// Feed the result to [`decode_query`].
#[allow(clippy::too_many_arguments)]
pub fn build_tile_query(
    table: &str,
    geom_column: &str,
    time_field: &str,
    time_format: TimeFormat,
    bbox: [f64; 4],
    t_start_ms: i64,
    t_end_ms: i64,
    source_srid: Option<i32>,
) -> String {
    let geom_raw = format!("q.{}", QuerySpec::quote_ident(geom_column));
    let geom = match source_srid {
        Some(srid) => format!("ST_Transform({geom_raw}, 'EPSG:{srid}', 'EPSG:4326', true)"),
        None => geom_raw,
    };
    let time = format!("q.{}", QuerySpec::quote_ident(time_field));
    let time_where = duckdb_time_window(&time, time_format, t_start_ms, t_end_ms);
    format!(
        "SELECT ST_AsWKB({geom}) AS {WKB_ALIAS}, q.* \
         FROM ( SELECT * FROM {table} ) AS q \
         WHERE ST_Intersects_Extent({geom}, ST_MakeEnvelope({}, {}, {}, {})) \
           AND {time_where}",
        bbox[0], bbox[1], bbox[2], bbox[3]
    )
}

/// The half-open `[t_start_ms, t_end_ms)` predicate for the time column, chosen
/// by `--time-format` (mirrors the PostGIS reader): `Iso8601` (default) assumes a
/// `TIMESTAMP` column and compares via `epoch_ms`; `UnixMs`/`UnixSec` assume an
/// integer column and compare numerically. Seconds compare the RAW column
/// against ceil-divided second bounds (`t*1000 >= a ⟺ t >= ⌈a/1000⌉` and
/// `t*1000 < b ⟺ t < ⌈b/1000⌉` for integer `t`) — exactly equivalent to
/// scaling the column, but sargable, so zone-map / index pruning on the column
/// still applies. Literals are server-formatted numbers — no
/// request-controlled SQL.
fn duckdb_time_window(
    time: &str,
    time_format: TimeFormat,
    t_start_ms: i64,
    t_end_ms: i64,
) -> String {
    match time_format {
        TimeFormat::Iso8601 => format!(
            "{time} >= epoch_ms({t_start_ms}::BIGINT) AND {time} <  epoch_ms({t_end_ms}::BIGINT)"
        ),
        TimeFormat::UnixMs => {
            format!("{time} >= {t_start_ms} AND {time} < {t_end_ms}")
        }
        TimeFormat::UnixSec => {
            let start_s = crate::db_input_common::ceil_ms_to_seconds(t_start_ms);
            let end_s = crate::db_input_common::ceil_ms_to_seconds(t_end_ms);
            format!("{time} >= {start_s} AND {time} < {end_s}")
        }
    }
}

/// Build the one-shot startup aggregate a dynamic server runs to advertise its
/// extent: spatial bounds, `[t_start, t_end]` (epoch-ms), and the row count.
/// `ST_Extent_Agg` (aggregate) yields a bbox GEOMETRY; the scalar `ST_Extent`
/// turns it into a `BOX_2D` for `ST_XMin`/… `epoch_ms(TIMESTAMP)` yields BIGINT
/// epoch-ms. The bounds + time values are NULL-safe (read as `Option`) for an
/// empty table; `cnt` (`COUNT(*)`) is never NULL. `source_srid` reprojects the
/// geometry to EPSG:4326 before aggregating, so the advertised bounds are
/// always lon/lat (matches [`build_tile_query`]).
pub fn build_metadata_query(
    table: &str,
    geom_column: &str,
    time_field: &str,
    time_format: TimeFormat,
    source_srid: Option<i32>,
) -> String {
    let geom_raw = QuerySpec::quote_ident(geom_column);
    let geom = match source_srid {
        Some(srid) => format!("ST_Transform({geom_raw}, 'EPSG:{srid}', 'EPSG:4326', true)"),
        None => geom_raw,
    };
    let time = QuerySpec::quote_ident(time_field);
    // Convert the aggregated min/max time to epoch-ms per `--time-format`, so an
    // integer-epoch column advertises the same extent it would when ingested.
    let (tmin_ms, tmax_ms) = match time_format {
        TimeFormat::Iso8601 => ("epoch_ms(tmin)::BIGINT", "epoch_ms(tmax)::BIGINT"),
        TimeFormat::UnixMs => ("tmin::BIGINT", "tmax::BIGINT"),
        TimeFormat::UnixSec => ("(tmin * 1000)::BIGINT", "(tmax * 1000)::BIGINT"),
    };
    format!(
        "SELECT ST_XMin(ext) AS min_lon, ST_YMin(ext) AS min_lat, \
                ST_XMax(ext) AS max_lon, ST_YMax(ext) AS max_lat, \
                {tmin_ms} AS t_start, {tmax_ms} AS t_end, cnt \
         FROM ( \
            SELECT ST_Extent(ST_Extent_Agg({geom})) AS ext, \
                   MIN({time}) AS tmin, MAX({time}) AS tmax, COUNT(*) AS cnt \
            FROM {table} \
         ) q"
    )
}

/// Open a DuckDB connection and prepare it for spatial reads: load the
/// `spatial` extension (`ST_AsWKB`/`ST_Transform`/`ST_Intersects`) and pin the
/// session to UTC. A real file path is opened **read-only** (so a build can run
/// against a database another process holds, and never mutates the user's
/// data); `:memory:` / empty opens a fresh in-memory database (for scanning
/// Parquet/CSV via `--sql`).
pub fn open_connection(db_path: &str) -> Result<Connection> {
    let conn = if db_path.is_empty() || db_path == ":memory:" {
        Connection::open_in_memory().context("failed to open in-memory DuckDB")?
    } else {
        let config = Config::default()
            .access_mode(AccessMode::ReadOnly)
            .context("configure DuckDB read-only access")?;
        Connection::open_with_flags(db_path, config)
            .with_context(|| format!("failed to open DuckDB database '{db_path}' (read-only)"))?
    };
    // INSTALL needs network the first time, then loads from the `~/.duckdb`
    // cache offline. Spatial is not autoloadable, so the explicit LOAD is
    // required even with autoload settings on.
    conn.execute_batch("INSTALL spatial; LOAD spatial; SET TimeZone='UTC';")
        .context(
            "failed to load the DuckDB spatial extension — it is a separate download that needs \
             a one-time `INSTALL spatial` with network access (cached under ~/.duckdb afterward)",
        )?;
    Ok(conn)
}

/// Eager variant: collect the whole DuckDB query into memory. Mirrors
/// [`crate::input::load_features`].
#[allow(clippy::too_many_arguments)]
pub fn load_features_duckdb(
    db_path: &str,
    spec: &QuerySpec,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
) -> Result<Vec<ParsedFeature>> {
    let mut features = Vec::new();
    stream_features_duckdb(
        db_path,
        spec,
        time_field,
        end_time_field,
        time_format,
        time_strictness,
        geometry_strictness,
        DEFAULT_BATCH_SIZE,
        |batch| {
            features.extend(batch);
            Ok(())
        },
    )?;
    tracing::info!("Loaded {} total features from DuckDB", features.len());
    Ok(features)
}

/// Stream a DuckDB query, invoking `on_batch` with materialised
/// [`ParsedFeature`]s every `batch_size` rows. Our handed-off buffer is bounded
/// by `batch_size`, mirroring [`crate::input::stream_features`]. (DuckDB itself
/// computes the result set into its compact in-memory columnar format; for very
/// large tables this is the analog of the table already living in the DB —
/// downstream tiling stays bounded by the flush size.)
#[allow(clippy::too_many_arguments)]
pub fn stream_features_duckdb<F>(
    db_path: &str,
    spec: &QuerySpec,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
    batch_size: usize,
    mut on_batch: F,
) -> Result<()>
where
    F: FnMut(Vec<ParsedFeature>) -> Result<()>,
{
    let wrapped = spec.wrapped_query();
    tracing::debug!("DuckDB source query: {wrapped}");

    let conn = open_connection(db_path)?;
    let mut stmt = conn.prepare(&wrapped).with_context(|| {
        "failed to prepare DuckDB query — check --table/--sql, --geom-column and that the \
         spatial extension loaded"
    })?;
    let mut rows = stmt.query([]).context("DuckDB query failed")?;

    // Result column names are needed to map time/end-time/geom/props by name.
    // Safe after `query()` (the statement has been executed). Per-cell values
    // are decoded from their self-describing `ValueRef`, so no type metadata is
    // needed here.
    let column_names: Vec<String> = rows.as_ref().map(|s| s.column_names()).unwrap_or_default();
    // Capture the result column types once so the end-of-read dropped-column
    // warning can print `name (TYPE)` (helps distinguish an unmappable-type drop
    // from an all-NULL column). Ingest path only.
    let column_types: Vec<String> = rows
        .as_ref()
        .map(|s| {
            (0..column_names.len())
                .map(|i| format!("{:?}", s.column_type(i)))
                .collect()
        })
        .unwrap_or_default();
    let schema = RowSchema::resolve_with_types(
        &column_names,
        &column_types,
        time_field,
        end_time_field,
        &spec.geom_column,
        time_format,
    )?;

    let cap = batch_size.clamp(1, 8192);
    let mut batch = Vec::with_capacity(cap);
    let mut total_rows = 0usize;
    let mut geom_failures = 0usize;
    // Property columns that carried a value in ANY row — to warn once about
    // source columns that produced nothing (unmappable type, or all-NULL).
    let mut seen_props: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Per-vertex array elements silently defaulted (NULL/unmappable element → 0
    // for timestamps, NaN for values), tallied once for an end-of-read summary.
    let mut vertex_coercions = VertexCoercions::default();

    while let Some(row) = rows.next().context("DuckDB row fetch failed")? {
        match schema.parse_row(
            row,
            time_strictness,
            geometry_strictness,
            total_rows,
            &mut vertex_coercions,
        )? {
            RowOutcome::Feature(f) => {
                // Track which property columns actually produced a value, before
                // the feature moves into the batch (stop once all have).
                if seen_props.len() < schema.props.len() {
                    if let Some(props) = &f.shared_properties {
                        for k in props.keys() {
                            seen_props.insert(k.clone());
                        }
                    }
                }
                batch.push(*f);
            }
            RowOutcome::GeomSkip => geom_failures += 1,
        }
        total_rows += 1;
        if batch.len() >= batch_size.max(1) {
            on_batch(std::mem::take(&mut batch))?;
            batch.reserve(cap);
        }
        if total_rows % 100_000 == 0 {
            tracing::info!("Loaded {total_rows} rows from DuckDB...");
        }
    }
    if !batch.is_empty() {
        on_batch(batch)?;
    }

    if geom_failures > 0 {
        tracing::warn!(
            "{geom_failures}/{total_rows} DuckDB rows had null/unparseable geometry and were skipped"
        );
    }
    warn_dropped_columns(
        schema.props.iter().map(|p| (p.name.as_str(), p.ty.clone())),
        &seen_props,
        total_rows,
        "DuckDB",
        "GEOMETRY/LIST/STRUCT",
    );
    vertex_coercions.warn("DuckDB");
    Ok(())
}

/// Decode a query whose geometry is projected to a WKB column aliased
/// [`WKB_ALIAS`] (e.g. [`build_tile_query`]) into [`ParsedFeature`]s, using an
/// already-open connection. This is the single-tile entry point a dynamic tile
/// server (`stt-serve`) calls per request. Unlike the PostGIS
/// `decode_rows(&[Row])`, DuckDB rows borrow their statement and cannot be
/// collected into an owned slice, so the prepare + iterate + decode happens
/// here behind one call.
#[allow(clippy::too_many_arguments)]
pub fn decode_query(
    conn: &Connection,
    sql: &str,
    time_field: &str,
    end_time_field: Option<&str>,
    geom_column: &str,
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
) -> Result<Vec<ParsedFeature>> {
    let mut stmt = conn.prepare(sql).context("prepare DuckDB tile query")?;
    let mut rows = stmt.query([]).context("DuckDB tile query failed")?;
    let column_names: Vec<String> = rows.as_ref().map(|s| s.column_names()).unwrap_or_default();
    if column_names.is_empty() {
        return Ok(Vec::new());
    }
    let schema = RowSchema::resolve(
        &column_names,
        time_field,
        end_time_field,
        geom_column,
        time_format,
    )?;

    let mut out = Vec::new();
    let mut i = 0usize;
    // Per-tile serve path: don't tally coercions (no end-of-read summary, no spam).
    let mut sink = VertexCoercions::default();
    while let Some(row) = rows.next().context("DuckDB tile row fetch failed")? {
        if let RowOutcome::Feature(f) =
            schema.parse_row(row, time_strictness, geometry_strictness, i, &mut sink)?
        {
            out.push(*f);
        }
        i += 1;
    }
    Ok(out)
}

/// Schema-level property-kind map for a DuckDB source — the DB analog of
/// [`crate::input::property_kinds`] (GeoParquet): probe the wrapped query's
/// result schema (a `LIMIT 0` execution — plans and types every column, scans
/// no rows) and pin each property column's tile kind from its result type, so
/// `TileConfig::property_types` keeps the layer schema stable when a column is
/// all-NULL within one tile (per-tile value sniffing drops it there and the
/// layer schema drifts across tiles; the file reader has pinned kinds from the
/// Parquet schema since 0.1.1 — this brings the DB path to parity).
pub fn property_kinds(
    db_path: &str,
    spec: &QuerySpec,
    time_field: &str,
    end_time_field: Option<&str>,
) -> Result<crate::columnar::PropertyTypes> {
    let conn = open_connection(db_path)?;
    property_kinds_on(&conn, spec, time_field, end_time_field)
}

/// As [`property_kinds`], on an already-open connection — the serve path
/// probes through its pool at startup.
pub fn property_kinds_on(
    conn: &Connection,
    spec: &QuerySpec,
    time_field: &str,
    end_time_field: Option<&str>,
) -> Result<crate::columnar::PropertyTypes> {
    property_kinds_for_query(
        conn,
        &spec.wrapped_query(),
        time_field,
        end_time_field,
        &spec.geom_column,
    )
}

/// As [`property_kinds`], for an arbitrary query that projects the geometry to
/// [`WKB_ALIAS`] — the same contract as [`decode_query`]. Excludes the system
/// columns with the same predicates [`RowSchema`] uses, so the pinned map and
/// the row decode can't disagree about which columns are properties.
pub fn property_kinds_for_query(
    conn: &Connection,
    sql: &str,
    time_field: &str,
    end_time_field: Option<&str>,
    geom_column: &str,
) -> Result<crate::columnar::PropertyTypes> {
    let probe = format!("SELECT * FROM ( {sql} ) AS __stt_schema_probe LIMIT 0");
    let mut stmt = conn
        .prepare(&probe)
        .context("prepare DuckDB schema probe")?;
    let rows = stmt.query([]).context("DuckDB schema probe failed")?;
    let mut kinds = crate::columnar::PropertyTypes::new();
    let Some(executed) = rows.as_ref() else {
        return Ok(kinds);
    };
    let names = executed.column_names();
    for (idx, name) in names.iter().enumerate() {
        if name == WKB_ALIAS
            || name == time_field
            || end_time_field == Some(name.as_str())
            || name == geom_column
            || crate::input::is_vertex_metadata_column(name)
        {
            continue;
        }
        if let Some(kind) = property_kind_for_duck(&executed.column_type(idx)) {
            kinds.insert(name.clone(), kind);
        }
    }
    Ok(kinds)
}

/// Schema-level mirror of [`decode_property_value`]: the tile kind a DuckDB
/// result column of this (vendored-Arrow) type decodes to. `None` = not
/// pinned — unmappable at read time (blobs, intervals, nested types), or a
/// `DECIMAL(p>18, 0)`/HUGEINT-shaped column whose out-of-i64-range values fall
/// back to strings, where a pinned Numeric could lie; those stay on per-tile
/// sniffing.
fn property_kind_for_duck(
    dt: &duckdb::arrow::datatypes::DataType,
) -> Option<crate::columnar::PropertyKind> {
    use crate::columnar::PropertyKind;
    use duckdb::arrow::datatypes::DataType as T;
    match dt {
        T::Int8
        | T::Int16
        | T::Int32
        | T::Int64
        | T::UInt8
        | T::UInt16
        | T::UInt32
        | T::UInt64
        | T::Float32
        | T::Float64
        // Timestamps/dates decode to epoch-ms numbers.
        | T::Timestamp(..)
        | T::Date32 => Some(PropertyKind::Numeric),
        // Scale-0 decimals surface as HugeInt (i128): DECIMAL(p ≤ 18, 0)
        // always fits i64 → always a number; wider stays unpinned (see above).
        T::Decimal128(p, 0) if *p <= 18 => Some(PropertyKind::Numeric),
        // Scaled decimals surface as `ValueRef::Decimal` → nearest-f64 number.
        T::Decimal128(_, s) if *s > 0 => Some(PropertyKind::Numeric),
        T::Boolean | T::Utf8 | T::LargeUtf8 => Some(PropertyKind::Categorical),
        _ => None,
    }
}

/// A property column: index in the result, its output name, and (when the
/// caller captured it) the DuckDB result type — used only to enrich the
/// end-of-read dropped-column warning; `None` on the serve path (which never
/// warns).
struct PropCol {
    idx: usize,
    name: String,
    ty: Option<String>,
}

/// Result-set schema resolved once from the result's column names. Indices only
/// — values are decoded from each cell's self-describing `ValueRef`.
struct RowSchema {
    wkb_idx: usize,
    time_idx: usize,
    end_time_idx: Option<usize>,
    /// Optional per-vertex array columns, detected by the exact column names the
    /// GeoParquet reader hardcodes (`vertex_timestamps` / `vertex_values` /
    /// `vertex_value_matrix`). `None` when the query exposes no such column.
    vertex_timestamps_idx: Option<usize>,
    vertex_values_idx: Option<usize>,
    vertex_value_matrix_idx: Option<usize>,
    props: Vec<PropCol>,
    time_format: TimeFormat,
}

impl RowSchema {
    fn resolve(
        names: &[String],
        time_field: &str,
        end_time_field: Option<&str>,
        geom_column: &str,
        time_format: TimeFormat,
    ) -> Result<Self> {
        Self::resolve_with_types(
            names,
            &[],
            time_field,
            end_time_field,
            geom_column,
            time_format,
        )
    }

    /// As [`resolve`], but records the per-column DuckDB result type string (from
    /// `Statement::column_type`, formatted by the caller) on each [`PropCol`] so
    /// the ingest-path dropped-column warning can print `name (TYPE)`. Pass an
    /// empty `types` slice to skip the type capture (the serve path, which never
    /// warns).
    fn resolve_with_types(
        names: &[String],
        types: &[String],
        time_field: &str,
        end_time_field: Option<&str>,
        geom_column: &str,
        time_format: TimeFormat,
    ) -> Result<Self> {
        let find = |name: &str| names.iter().position(|c| c == name);

        let wkb_idx = find(WKB_ALIAS).ok_or_else(|| {
            anyhow::anyhow!("internal: wrapped query did not project the {WKB_ALIAS} column")
        })?;
        let time_idx = find(time_field).ok_or_else(|| {
            anyhow::anyhow!("--time-field '{time_field}' not found in the DuckDB result columns")
        })?;
        let end_time_idx = match end_time_field {
            Some(f) => Some(find(f).ok_or_else(|| {
                anyhow::anyhow!("--end-time-field '{f}' not found in the DuckDB result columns")
            })?),
            None => None,
        };

        // Optional per-vertex array columns, detected by the exact column names
        // the GeoParquet reader hardcodes.
        let vertex_timestamps_idx = find("vertex_timestamps");
        let vertex_values_idx = find("vertex_values");
        let vertex_value_matrix_idx = find("vertex_value_matrix");

        // Every remaining column becomes a property, except the system columns
        // (wkb, time, end-time), the original geometry column, and the
        // per-vertex array columns. Coordinate-NAMED columns (lon/lat/x/y) are
        // deliberately NOT excluded: geometry always comes from the geometry
        // column here, so a SELECTed `x`/`lat` is a genuine user attribute —
        // dropping it by name was silent data loss (same fix as the
        // GeoParquet reader's consumed-columns narrowing; the SELECT list is
        // the escape hatch for redundant convenience copies). Columns whose
        // type we can't map (raw GEOMETRY, decimals, unmapped arrays, …)
        // decode to None at read time and are silently dropped per-row.
        let mut props = Vec::new();
        for (idx, name) in names.iter().enumerate() {
            if idx == wkb_idx
                || idx == time_idx
                || end_time_idx == Some(idx)
                || name == geom_column
                || crate::input::is_vertex_metadata_column(name)
            {
                continue;
            }
            props.push(PropCol {
                idx,
                name: name.clone(),
                ty: types.get(idx).cloned(),
            });
        }

        Ok(RowSchema {
            wkb_idx,
            time_idx,
            end_time_idx,
            vertex_timestamps_idx,
            vertex_values_idx,
            vertex_value_matrix_idx,
            props,
            time_format,
        })
    }

    fn parse_row(
        &self,
        row: &Row,
        time_strictness: InputStrictness,
        geometry_strictness: InputStrictness,
        row_no: usize,
        coercions: &mut VertexCoercions,
    ) -> Result<RowOutcome> {
        // Geometry (WKB blob -> GeoJSON + centroid lon/lat).
        let parsed = match row.get_ref(self.wkb_idx) {
            Ok(ValueRef::Blob(bytes)) => parse_wkb_geometry(bytes),
            _ => None,
        };
        let Some((geometry, lon, lat)) = parsed else {
            if geometry_strictness == InputStrictness::Strict {
                anyhow::bail!(
                    "row {row_no}: null or unparseable geometry (rerun without --strict-geometry to skip)"
                );
            }
            return Ok(RowOutcome::GeomSkip);
        };

        // Timestamp.
        let timestamp = match decode_time(row, self.time_idx, self.time_format, row_no)? {
            Some(ts) => ts,
            None => {
                if time_strictness == InputStrictness::Strict {
                    anyhow::bail!(
                        "row {row_no}: null/unparseable timestamp (rerun without --strict-times to coerce to epoch 0)"
                    );
                }
                // Warn mode mirrors the GeoParquet path: coerce to epoch 0.
                0
            }
        };
        let end_timestamp = match self.end_time_idx {
            Some(idx) => decode_time(row, idx, self.time_format, row_no)?,
            None => None,
        };

        // Per-vertex arrays (None when the column is absent or SQL NULL),
        // mirroring the GeoParquet reader's `vertex_*` columns.
        let vertex_timestamps = match self.vertex_timestamps_idx {
            Some(idx) => decode_u64_ms_list(row, idx, row_no, &mut coercions.timestamps)?,
            None => None,
        };
        let vertex_values = match self.vertex_values_idx {
            Some(idx) => decode_f32_list(row, idx, &mut coercions.values),
            None => None,
        };
        let vertex_value_matrix = match self.vertex_value_matrix_idx {
            Some(idx) => decode_f32_list(row, idx, &mut coercions.values),
            None => None,
        };

        // Properties.
        let mut properties = serde_json::Map::new();
        for p in &self.props {
            if let Ok(v) = row.get_ref(p.idx) {
                if let Some(value) = decode_property_value(v) {
                    properties.insert(p.name.clone(), value);
                }
            }
        }
        let shared_properties = if properties.is_empty() {
            None
        } else {
            Some(Arc::new(properties))
        };

        let feature = Feature {
            bbox: None,
            geometry: Some(geometry),
            id: None,
            properties: None,
            foreign_members: None,
        };
        Ok(RowOutcome::Feature(Box::new(ParsedFeature {
            geojson: feature,
            shared_properties,
            timestamp,
            end_timestamp,
            vertex_timestamps,
            vertex_values,
            vertex_value_matrix,
            lon,
            lat,
        })))
    }
}

/// Decode a time column value to Unix milliseconds, mirroring the GeoParquet
/// reader's per-type rules. Returns `Ok(None)` for SQL NULL / unmappable type.
fn decode_time(
    row: &Row,
    idx: usize,
    time_format: TimeFormat,
    row_no: usize,
) -> Result<Option<u64>> {
    let ms: Option<i64> = match row.get_ref(idx) {
        Ok(v) => decode_time_value(v, time_format, row_no)?,
        Err(_) => None,
    };
    match ms {
        Some(v) => {
            reject_negative_timestamp(row_no, v)?;
            Ok(Some(v as u64))
        }
        None => Ok(None),
    }
}

/// Map a `ValueRef` to Unix-ms for the time column, mirroring the GeoParquet
/// reader: TIMESTAMP/TIMESTAMPTZ directly, DATE at midnight UTC, integer
/// columns via `--time-format` (second→ms overflow-checked, like the file
/// path), text parsed as ISO 8601.
fn decode_time_value(v: ValueRef, time_format: TimeFormat, row_no: usize) -> Result<Option<i64>> {
    Ok(match v {
        ValueRef::Timestamp(unit, value) => Some(timestamp_unit_to_ms(unit, value)?),
        ValueRef::Date32(days) => Some((days as i64) * 86_400_000),
        ValueRef::BigInt(n) => Some(apply_int_time_format(n, time_format, row_no)?),
        ValueRef::Int(n) => Some(apply_int_time_format(n as i64, time_format, row_no)?),
        ValueRef::UInt(n) => Some(apply_int_time_format(n as i64, time_format, row_no)?),
        ValueRef::UBigInt(n) if n <= i64::MAX as u64 => {
            Some(apply_int_time_format(n as i64, time_format, row_no)?)
        }
        ValueRef::Text(bytes) => std::str::from_utf8(bytes)
            .ok()
            .and_then(|s| parse_iso8601(s).ok()),
        _ => None,
    })
}

/// DuckDB `TIMESTAMP`/`TIMESTAMPTZ` value (in `unit`) → Unix milliseconds.
/// Delegates to the shared [`crate::input::scale_timestamp_to_ms`] so the DuckDB
/// reader scales precision identically to the GeoParquet file reader, including
/// the overflow-CHECKED second→ms multiply (the non-negative guard is applied by
/// the wire-timestamp callers, [`decode_time`] / [`decode_u64_ms_list`]; a
/// `Timestamp` emitted as a plain *property* keeps signed ms so a pre-1970 value
/// survives).
fn timestamp_unit_to_ms(unit: TimeUnit, value: i64) -> Result<i64> {
    use crate::input::TimestampUnit;
    let unit = match unit {
        TimeUnit::Second => TimestampUnit::Second,
        TimeUnit::Millisecond => TimestampUnit::Millisecond,
        TimeUnit::Microsecond => TimestampUnit::Microsecond,
        TimeUnit::Nanosecond => TimestampUnit::Nanosecond,
    };
    Ok(crate::input::scale_timestamp_to_ms(value, unit)?)
}

/// Decode a per-vertex timestamp LIST/ARRAY cell to `Vec<u64>` ms-since-epoch,
/// mirroring the GeoParquet reader's `vertex_timestamps` rules: integer elements
/// are RAW ms (never reinterpreted via `--time-format`), timestamp/date elements
/// convert to UTC ms, a whole-cell SQL NULL yields `None`, a NULL element
/// becomes `0` (preserving per-vertex alignment), and any negative (pre-1970)
/// value is a hard error in both strictness modes. DuckDB picks `Value::List`
/// for `BIGINT[]`/`TIMESTAMP[]` and `Value::Array` for fixed-size declarations.
fn decode_u64_ms_list(
    row: &Row,
    idx: usize,
    row_no: usize,
    coerced: &mut usize,
) -> Result<Option<Vec<u64>>> {
    use duckdb::types::Value;
    let items = match row.get::<usize, Value>(idx) {
        Ok(Value::List(items)) | Ok(Value::Array(items)) => items,
        // Whole-cell NULL, a scalar, or an unconvertible value → no per-vertex data.
        _ => return Ok(None),
    };
    let mut out = Vec::with_capacity(items.len());
    for v in items {
        let ms: i64 = match v {
            Value::BigInt(n) => n,
            Value::Int(n) => n as i64,
            Value::UInt(n) => n as i64,
            Value::UBigInt(n) if n <= i64::MAX as u64 => n as i64,
            Value::Timestamp(unit, val) => timestamp_unit_to_ms(unit, val)?,
            Value::Date32(days) => (days as i64) * 86_400_000,
            // NULL element (and any other type) → 0, matching the file reader.
            _ => {
                *coerced += 1;
                0
            }
        };
        if ms < 0 {
            anyhow::bail!(
                "row {row_no}: negative vertex timestamp {ms} (pre-1970). The STT temporal index \
                 stores unsigned ms-since-epoch; filter or re-epoch these rows."
            );
        }
        out.push(ms as u64);
    }
    Ok(Some(out))
}

/// Decode a per-vertex float LIST/ARRAY cell to `Vec<f32>`, mirroring the
/// GeoParquet reader's `vertex_values` rules: `DOUBLE`/`FLOAT` elements map to
/// f32, a whole-cell SQL NULL yields `None`, and a NULL element becomes
/// `f32::NAN` (preserving per-vertex alignment).
fn decode_f32_list(row: &Row, idx: usize, coerced: &mut usize) -> Option<Vec<f32>> {
    use duckdb::types::Value;
    let items = match row.get::<usize, Value>(idx) {
        Ok(Value::List(items)) | Ok(Value::Array(items)) => items,
        _ => return None,
    };
    Some(
        items
            .into_iter()
            .map(|v| match v {
                Value::Double(d) => d as f32,
                Value::Float(f) => f,
                // NULL element (and any other type) → NaN, matching the file reader.
                _ => {
                    *coerced += 1;
                    f32::NAN
                }
            })
            .collect(),
    )
}

/// Map one property cell to a JSON value. SQL NULL and unmappable types (raw
/// GEOMETRY/other BLOBs, intervals, nested types, …) return `None` and are
/// dropped.
///
/// This is a documented **superset** of `crate::input::extract_property_value`,
/// not an exact mirror: it shares the six core types (bool / Int32 / Int64 /
/// Float32 / Float64 / text) and the NaN→JSON-null rule, but additionally maps
/// `TinyInt`/`SmallInt`, all unsigned ints, `HugeInt`, `Decimal` (→ nearest-f64
/// number via [`decimal_string_to_json`] — the file reader drops Parquet
/// decimals), and `Timestamp`/`Date32` (→ epoch-ms numbers) — types a
/// GeoParquet file can't express the same way, so they only ever add properties
/// a file build couldn't carry, never diverge on the common type set. The
/// DuckDB and PostGIS readers stay in lockstep here (NUMERIC/DECIMAL uses the
/// one shared conversion).
fn decode_property_value(v: ValueRef) -> Option<serde_json::Value> {
    use serde_json::Value;
    match v {
        ValueRef::Boolean(b) => Some(Value::Bool(b)),
        ValueRef::TinyInt(n) => Some(Value::from(n as i64)),
        ValueRef::SmallInt(n) => Some(Value::from(n as i64)),
        ValueRef::Int(n) => Some(Value::from(n as i64)),
        ValueRef::BigInt(n) => Some(Value::from(n)),
        ValueRef::HugeInt(n) => i64::try_from(n)
            .map(Value::from)
            .ok()
            .or_else(|| Some(Value::String(n.to_string()))),
        ValueRef::UTinyInt(n) => Some(Value::from(n as u64)),
        ValueRef::USmallInt(n) => Some(Value::from(n as u64)),
        ValueRef::UInt(n) => Some(Value::from(n as u64)),
        ValueRef::UBigInt(n) => Some(Value::from(n)),
        ValueRef::Float(f) => Some(json_number_or_null(f as f64)),
        ValueRef::Double(f) => Some(json_number_or_null(f)),
        // Fixed-point DECIMAL → nearest f64, via the conversion shared with the
        // PostGIS reader's NUMERIC arm (identical value → identical number).
        ValueRef::Decimal(d) => Some(decimal_string_to_json(&d.to_string())),
        ValueRef::Text(bytes) => std::str::from_utf8(bytes)
            .ok()
            .map(|s| Value::String(s.to_string())),
        // Overflowing timestamps are dropped like any other unmappable cell.
        ValueRef::Timestamp(unit, value) => timestamp_unit_to_ms(unit, value).ok().map(Value::from),
        ValueRef::Date32(days) => Some(Value::from((days as i64) * 86_400_000)),
        // NULL, Blob (incl. raw GEOMETRY/WKB), Time64, Interval, and the
        // nested List/Struct/Map/Enum/Array/Union types are dropped.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_table_query_with_wkb_projection() {
        let spec = QuerySpec {
            source: QuerySource::Table("main.trips".into()),
            geom_column: "geom".into(),
            where_clause: None,
            reproject_from_srid: None,
        };
        let q = spec.wrapped_query();
        assert!(q.contains("ST_AsWKB(q.\"geom\")"), "{q}");
        assert!(q.contains(WKB_ALIAS), "{q}");
        assert!(q.contains("FROM ( SELECT * FROM main.trips ) AS q"), "{q}");
    }

    #[test]
    fn wraps_sql_with_reprojection_and_where() {
        let spec = QuerySpec {
            source: QuerySource::Sql("SELECT * FROM read_parquet('o.parquet')".into()),
            geom_column: "the_geom".into(),
            where_clause: Some("valid AND ts > '2020-01-01'".into()),
            reproject_from_srid: Some(3857),
        };
        let q = spec.wrapped_query();
        assert!(
            q.contains("ST_AsWKB(ST_Transform(q.\"the_geom\", 'EPSG:3857', 'EPSG:4326', true))"),
            "{q}"
        );
        assert!(q.contains("WHERE valid AND ts > '2020-01-01'"), "{q}");
    }

    #[test]
    fn tile_query_filters_bbox_and_time() {
        // Default (Iso8601 → timestamp column): epoch_ms comparison; the bbox
        // prefilter is the pure-MBR ST_Intersects_Extent (the `&&` analog).
        let q = build_tile_query(
            "obs",
            "geom",
            "ts",
            TimeFormat::Iso8601,
            [-10.0, -5.0, 10.0, 5.0],
            1000,
            2000,
            None,
        );
        assert!(q.contains("ST_AsWKB(q.\"geom\")"), "{q}");
        assert!(
            q.contains("ST_Intersects_Extent(q.\"geom\", ST_MakeEnvelope(-10, -5, 10, 5))"),
            "{q}"
        );
        assert!(q.contains("q.\"ts\" >= epoch_ms(1000::BIGINT)"), "{q}");
        assert!(q.contains("q.\"ts\" <  epoch_ms(2000::BIGINT)"), "{q}");

        // Integer-epoch column: numeric comparison, no epoch_ms.
        let ms = build_tile_query(
            "obs",
            "geom",
            "ts",
            TimeFormat::UnixMs,
            [-10.0, -5.0, 10.0, 5.0],
            1000,
            2000,
            None,
        );
        assert!(ms.contains("q.\"ts\" >= 1000 AND q.\"ts\" < 2000"), "{ms}");
        assert!(!ms.contains("epoch_ms"), "{ms}");

        // Seconds: sargable — the RAW column against ceil-divided second
        // bounds (t*1000 >= 1000 ⟺ t >= 1; t*1000 < 2000 ⟺ t < 2), never a
        // scaled-column expression that would defeat zone-map pruning.
        let sec = build_tile_query(
            "obs",
            "geom",
            "ts",
            TimeFormat::UnixSec,
            [-10.0, -5.0, 10.0, 5.0],
            1000,
            2000,
            None,
        );
        assert!(sec.contains("q.\"ts\" >= 1 AND q.\"ts\" < 2"), "{sec}");
        assert!(!sec.contains("* 1000"), "{sec}");

        // Non-multiple-of-1000 ms bounds round up on both ends (half-open).
        let odd = duckdb_time_window("t", TimeFormat::UnixSec, 1500, 2001);
        assert_eq!(odd, "t >= 2 AND t < 3");
    }

    #[test]
    fn tile_query_reprojects_when_source_srid_given() {
        let q = build_tile_query(
            "obs",
            "geom",
            "ts",
            TimeFormat::Iso8601,
            [-10.0, -5.0, 10.0, 5.0],
            1000,
            2000,
            Some(3857),
        );
        // The transformed geometry feeds BOTH the WKB projection and the bbox
        // test, so the filter runs in tile (4326) space — a strict superset.
        assert!(
            q.contains("ST_AsWKB(ST_Transform(q.\"geom\", 'EPSG:3857', 'EPSG:4326', true))"),
            "{q}"
        );
        assert!(
            q.contains(
                "ST_Intersects_Extent(ST_Transform(q.\"geom\", 'EPSG:3857', 'EPSG:4326', true), \
                 ST_MakeEnvelope(-10, -5, 10, 5))"
            ),
            "{q}"
        );

        let mq = build_metadata_query("obs", "geom", "ts", TimeFormat::Iso8601, Some(3857));
        assert!(
            mq.contains("ST_Extent_Agg(ST_Transform(\"geom\", 'EPSG:3857', 'EPSG:4326', true))"),
            "{mq}"
        );
    }

    #[test]
    fn timestamp_units_to_ms() {
        assert_eq!(timestamp_unit_to_ms(TimeUnit::Second, 5).unwrap(), 5000);
        assert_eq!(timestamp_unit_to_ms(TimeUnit::Millisecond, 5).unwrap(), 5);
        assert_eq!(
            timestamp_unit_to_ms(TimeUnit::Microsecond, 5_000).unwrap(),
            5
        );
        assert_eq!(
            timestamp_unit_to_ms(TimeUnit::Nanosecond, 5_000_000).unwrap(),
            5
        );
    }

    #[test]
    fn quotes_identifiers_safely() {
        assert_eq!(QuerySpec::quote_ident("geom"), "\"geom\"");
        assert_eq!(QuerySpec::quote_ident("we\"ird"), "\"we\"\"ird\"");
    }

    /// WKB for POINT(1 2), little-endian — shared by the no-spatial decode tests
    /// (the geometry rides as a pre-made `BLOB` aliased to [`WKB_ALIAS`]).
    const POINT_WKB: &[u8] = &[
        0x01, // little-endian
        0x01, 0x00, 0x00, 0x00, // geometry type 1 = Point
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // x = 1.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y = 2.0
    ];

    /// End-to-end decode against a real (bundled) in-memory DuckDB — no spatial
    /// extension / network needed: the geometry is supplied as a pre-made WKB
    /// `BLOB` aliased to [`WKB_ALIAS`], exercising the `ValueRef` decode path,
    /// column-name resolution, timestamp→ms, and property mapping.
    #[test]
    fn decodes_blob_geometry_time_and_props() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (g BLOB, ts TIMESTAMP, n INTEGER, name VARCHAR);")
            .unwrap();
        conn.execute(
            "INSERT INTO t VALUES (?, TIMESTAMP '2024-06-21 12:00:00', 7, 'hi')",
            duckdb::params![POINT_WKB],
        )
        .unwrap();

        let mut stmt = conn
            .prepare("SELECT g AS __stt_wkb, ts AS \"timestamp\", n AS val, name FROM t")
            .unwrap();
        let mut rows = stmt.query([]).unwrap();
        let names: Vec<String> = rows.as_ref().unwrap().column_names();
        let schema =
            RowSchema::resolve(&names, "timestamp", None, "g", TimeFormat::Iso8601).unwrap();

        let row = rows.next().unwrap().unwrap();
        let mut coercions = VertexCoercions::default();
        let out = schema
            .parse_row(
                row,
                InputStrictness::Warn,
                InputStrictness::Warn,
                0,
                &mut coercions,
            )
            .unwrap();
        let RowOutcome::Feature(f) = out else {
            panic!("expected a decoded feature");
        };

        assert_eq!(f.lon, 1.0);
        assert_eq!(f.lat, 2.0);
        let expected = chrono::NaiveDate::from_ymd_opt(2024, 6, 21)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis() as u64;
        assert_eq!(f.timestamp, expected);

        let props = f.shared_properties.expect("properties present");
        assert_eq!(props.get("val"), Some(&serde_json::Value::from(7i64)));
        assert_eq!(
            props.get("name"),
            Some(&serde_json::Value::String("hi".into()))
        );
    }

    /// `DECIMAL` property cells decode to nearest-f64 JSON numbers (they were
    /// silently dropped before); a NULL decimal keeps dropping the key.
    #[test]
    fn decodes_decimal_properties() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE t (g BLOB, ts TIMESTAMP, price DECIMAL(10,2), fee DECIMAL(18,6));
             INSERT INTO t VALUES (NULL, TIMESTAMP '2024-06-21 12:00:00', 12.34, NULL);",
        )
        .unwrap();
        conn.execute("UPDATE t SET g = ?", duckdb::params![POINT_WKB])
            .unwrap();

        let mut stmt = conn
            .prepare("SELECT g AS __stt_wkb, ts AS \"timestamp\", price, fee FROM t")
            .unwrap();
        let mut rows = stmt.query([]).unwrap();
        let names: Vec<String> = rows.as_ref().unwrap().column_names();
        let schema =
            RowSchema::resolve(&names, "timestamp", None, "g", TimeFormat::Iso8601).unwrap();
        let row = rows.next().unwrap().unwrap();
        let mut coercions = VertexCoercions::default();
        let RowOutcome::Feature(f) = schema
            .parse_row(
                row,
                InputStrictness::Warn,
                InputStrictness::Warn,
                0,
                &mut coercions,
            )
            .unwrap()
        else {
            panic!("expected a decoded feature");
        };
        let props = f.shared_properties.expect("properties present");
        assert_eq!(props.get("price"), Some(&serde_json::Value::from(12.34)));
        assert_eq!(props.get("fee"), None, "NULL decimal drops the key");
    }

    /// Live smoke test of the actual spatial SQL the server emits — exercises
    /// `ST_Point`/`ST_AsWKB`/`ST_MakeEnvelope`/`ST_Intersects_Extent`/`epoch_ms` (tile
    /// query) and `ST_Transform` (reprojection). Ignored by default because the
    /// first `INSTALL spatial` needs network (cached under `~/.duckdb` after).
    /// Run with: `cargo test -p stt-build --features duckdb -- --ignored`.
    #[test]
    #[ignore = "requires a one-time network INSTALL of the DuckDB spatial extension"]
    fn spatial_roundtrip_smoke() {
        let conn = open_connection(":memory:").expect("open + load spatial");
        conn.execute_batch(
            "CREATE TABLE obs (geom GEOMETRY, ts TIMESTAMP, mag DOUBLE);
             INSERT INTO obs VALUES
               (ST_Point(-73.9, 40.7), TIMESTAMP '2024-06-21 12:00:00', 3.5),
               (ST_Point(2.35, 48.85),  TIMESTAMP '2024-06-21 12:00:00', 1.0);",
        )
        .unwrap();

        let t0 = chrono::NaiveDate::from_ymd_opt(2024, 6, 21)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let t1 = t0 + 86_400_000;
        // bbox around NYC — excludes the Paris point.
        let sql = build_tile_query(
            "obs",
            "geom",
            "ts",
            TimeFormat::Iso8601,
            [-74.5, 40.0, -73.0, 41.0],
            t0,
            t1,
            None,
        );
        let feats = decode_query(
            &conn,
            &sql,
            "ts",
            None,
            "geom",
            TimeFormat::Iso8601,
            InputStrictness::Warn,
            InputStrictness::Warn,
        )
        .unwrap();
        assert_eq!(feats.len(), 1, "only the NYC point is in the bbox + window");
        assert!(
            (feats[0].lon - (-73.9)).abs() < 1e-6,
            "lon {}",
            feats[0].lon
        );
        assert!((feats[0].lat - 40.7).abs() < 1e-6, "lat {}", feats[0].lat);
        assert_eq!(
            feats[0].shared_properties.as_ref().unwrap().get("mag"),
            Some(&serde_json::Value::from(3.5))
        );

        // Reproject path: store a Web-Mercator (3857) geometry, transform back.
        conn.execute_batch(
            "CREATE TABLE merc (geom GEOMETRY, ts TIMESTAMP);
             INSERT INTO merc VALUES
               (ST_Transform(ST_Point(-73.9, 40.7), 'EPSG:4326', 'EPSG:3857', true),
                TIMESTAMP '2024-06-21 12:00:00');",
        )
        .unwrap();
        let spec = QuerySpec {
            source: QuerySource::Table("merc".into()),
            geom_column: "geom".into(),
            where_clause: None,
            reproject_from_srid: Some(3857),
        };
        let f2 = decode_query(
            &conn,
            &spec.wrapped_query(),
            "ts",
            None,
            "geom",
            TimeFormat::Iso8601,
            InputStrictness::Warn,
            InputStrictness::Warn,
        )
        .unwrap();
        assert_eq!(f2.len(), 1);
        assert!(
            (f2[0].lon - (-73.9)).abs() < 1e-4,
            "reprojected lon {}",
            f2[0].lon
        );
        assert!(
            (f2[0].lat - 40.7).abs() < 1e-4,
            "reprojected lat {}",
            f2[0].lat
        );

        // Serve-style reprojected tile query (the --source-srid serve path):
        // the same NYC bbox against the Web-Mercator table must reproject,
        // bbox-filter in 4326 space, and decode to lon/lat.
        let srid_sql = build_tile_query(
            "merc",
            "geom",
            "ts",
            TimeFormat::Iso8601,
            [-74.5, 40.0, -73.0, 41.0],
            t0,
            t1,
            Some(3857),
        );
        let f3 = decode_query(
            &conn,
            &srid_sql,
            "ts",
            None,
            "geom",
            TimeFormat::Iso8601,
            InputStrictness::Warn,
            InputStrictness::Warn,
        )
        .unwrap();
        assert_eq!(
            f3.len(),
            1,
            "reprojected tile query finds the mercator point"
        );
        assert!(
            (f3[0].lon - (-73.9)).abs() < 1e-4,
            "srid tile lon {}",
            f3[0].lon
        );
        assert!(
            (f3[0].lat - 40.7).abs() < 1e-4,
            "srid tile lat {}",
            f3[0].lat
        );

        // Reprojected metadata bounds come back in lon/lat.
        let mq_srid = build_metadata_query("merc", "geom", "ts", TimeFormat::Iso8601, Some(3857));
        let mut stmt = conn.prepare(&mq_srid).unwrap();
        let mut rows = stmt.query([]).unwrap();
        let row = rows.next().unwrap().unwrap();
        let min_lon: Option<f64> = row.get(0).unwrap();
        assert!(
            (min_lon.unwrap() - (-73.9)).abs() < 1e-4,
            "srid metadata min_lon {min_lon:?}"
        );
        drop(rows);
        drop(stmt);

        // Metadata aggregate: bounds, [t_start, t_end] epoch-ms, count.
        let mq = build_metadata_query("obs", "geom", "ts", TimeFormat::Iso8601, None);
        let mut stmt = conn.prepare(&mq).unwrap();
        let mut rows = stmt.query([]).unwrap();
        let row = rows.next().unwrap().unwrap();
        let min_lon: Option<f64> = row.get(0).unwrap();
        let max_lon: Option<f64> = row.get(2).unwrap();
        let t_start: Option<i64> = row.get(4).unwrap();
        let cnt: i64 = row.get(6).unwrap();
        assert_eq!(cnt, 2);
        assert!(
            min_lon.unwrap() <= -73.9 && max_lon.unwrap() >= 2.35,
            "bounds span both points"
        );
        let noon = chrono::NaiveDate::from_ymd_opt(2024, 6, 21)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        assert_eq!(t_start.unwrap(), noon);
    }
}
