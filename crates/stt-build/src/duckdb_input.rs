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

use crate::input::{parse_iso8601, parse_wkb_geometry, InputStrictness, ParsedFeature, TimeFormat};

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
/// a WGS84 bounding box (`ST_Intersects` against an envelope — RTREE-index
/// accelerable) and a half-open time window `[t_start_ms, t_end_ms)`. The
/// geometry / time literals are server-formatted numbers (request `z/x/y/t` are
/// parsed to integers upstream), so there is no request-controlled SQL. The
/// time column must be a `TIMESTAMP`/`TIMESTAMP WITH TIME ZONE` (compared via
/// `epoch_ms`, which yields a UTC `TIMESTAMP`; pin the session to UTC).
///
/// Feed the result to [`decode_query`].
pub fn build_tile_query(
    table: &str,
    geom_column: &str,
    time_field: &str,
    bbox: [f64; 4],
    t_start_ms: i64,
    t_end_ms: i64,
) -> String {
    let geom = format!("q.{}", QuerySpec::quote_ident(geom_column));
    let time = format!("q.{}", QuerySpec::quote_ident(time_field));
    format!(
        "SELECT ST_AsWKB({geom}) AS {WKB_ALIAS}, q.* \
         FROM ( SELECT * FROM {table} ) AS q \
         WHERE ST_Intersects({geom}, ST_MakeEnvelope({}, {}, {}, {})) \
           AND {time} >= epoch_ms({}::BIGINT) \
           AND {time} <  epoch_ms({}::BIGINT)",
        bbox[0], bbox[1], bbox[2], bbox[3], t_start_ms, t_end_ms
    )
}

/// Build the one-shot startup aggregate a dynamic server runs to advertise its
/// extent: spatial bounds, `[t_start, t_end]` (epoch-ms), and the row count.
/// `ST_Extent_Agg` (aggregate) yields a bbox GEOMETRY; the scalar `ST_Extent`
/// turns it into a `BOX_2D` for `ST_XMin`/… `epoch_ms(TIMESTAMP)` yields BIGINT
/// epoch-ms. The bounds + time values are NULL-safe (read as `Option`) for an
/// empty table; `cnt` (`COUNT(*)`) is never NULL.
pub fn build_metadata_query(table: &str, geom_column: &str, time_field: &str) -> String {
    let geom = QuerySpec::quote_ident(geom_column);
    let time = QuerySpec::quote_ident(time_field);
    format!(
        "SELECT ST_XMin(ext) AS min_lon, ST_YMin(ext) AS min_lat, \
                ST_XMax(ext) AS max_lon, ST_YMax(ext) AS max_lat, \
                epoch_ms(tmin)::BIGINT AS t_start, epoch_ms(tmax)::BIGINT AS t_end, cnt \
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
    let column_names: Vec<String> = rows
        .as_ref()
        .map(|s| s.column_names())
        .unwrap_or_default();
    let schema = RowSchema::resolve(
        &column_names,
        time_field,
        end_time_field,
        &spec.geom_column,
        time_format,
    )?;

    let cap = batch_size.clamp(1, 8192);
    let mut batch = Vec::with_capacity(cap);
    let mut total_rows = 0usize;
    let mut geom_failures = 0usize;

    while let Some(row) = rows.next().context("DuckDB row fetch failed")? {
        match schema.parse_row(row, time_strictness, geometry_strictness, total_rows)? {
            RowOutcome::Feature(f) => batch.push(*f),
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
    let column_names: Vec<String> = rows
        .as_ref()
        .map(|s| s.column_names())
        .unwrap_or_default();
    if column_names.is_empty() {
        return Ok(Vec::new());
    }
    let schema =
        RowSchema::resolve(&column_names, time_field, end_time_field, geom_column, time_format)?;

    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(row) = rows.next().context("DuckDB tile row fetch failed")? {
        if let RowOutcome::Feature(f) =
            schema.parse_row(row, time_strictness, geometry_strictness, i)?
        {
            out.push(*f);
        }
        i += 1;
    }
    Ok(out)
}

/// A property column: index in the result and its output name.
struct PropCol {
    idx: usize,
    name: String,
}

/// Result-set schema resolved once from the result's column names. Indices only
/// — values are decoded from each cell's self-describing `ValueRef`.
struct RowSchema {
    wkb_idx: usize,
    time_idx: usize,
    end_time_idx: Option<usize>,
    props: Vec<PropCol>,
    time_format: TimeFormat,
}

enum RowOutcome {
    Feature(Box<ParsedFeature>),
    GeomSkip,
}

impl RowSchema {
    fn resolve(
        names: &[String],
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

        // Every remaining column becomes a property, except the system columns
        // (wkb, time, end-time), the original geometry column, and the
        // geometry-component coordinate names (`lon`/`lat`/`x`/`y`/…) — which
        // the GeoParquet reader also treats as geometry metadata, not
        // properties (`crate::input` `property_cols`), so the ingest paths stay
        // consistent and tiles don't carry coordinates twice. Columns whose
        // type we can't map (raw GEOMETRY, decimals, arrays, …) decode to None
        // at read time and are silently dropped per-row.
        let is_coord_name = |n: &str| {
            matches!(
                n.to_ascii_lowercase().as_str(),
                "lon" | "lat" | "longitude" | "latitude" | "x" | "y"
            )
        };
        let mut props = Vec::new();
        for (idx, name) in names.iter().enumerate() {
            if idx == wkb_idx
                || idx == time_idx
                || end_time_idx == Some(idx)
                || name == geom_column
                || is_coord_name(name)
            {
                continue;
            }
            props.push(PropCol {
                idx,
                name: name.clone(),
            });
        }

        Ok(RowSchema {
            wkb_idx,
            time_idx,
            end_time_idx,
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
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
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
        Ok(v) => decode_time_value(v, time_format),
        Err(_) => None,
    };
    match ms {
        Some(v) if v < 0 => anyhow::bail!(
            "row {row_no}: negative timestamp {v} (pre-1970). The STT temporal index stores \
             unsigned ms-since-epoch; filter or re-epoch these rows before building."
        ),
        Some(v) => Ok(Some(v as u64)),
        None => Ok(None),
    }
}

/// Map a `ValueRef` to Unix-ms for the time column, mirroring the GeoParquet
/// reader: TIMESTAMP/TIMESTAMPTZ directly, DATE at midnight UTC, integer
/// columns via `--time-format`, text parsed as ISO 8601.
fn decode_time_value(v: ValueRef, time_format: TimeFormat) -> Option<i64> {
    match v {
        ValueRef::Timestamp(unit, value) => Some(timestamp_unit_to_ms(unit, value)),
        ValueRef::Date32(days) => Some((days as i64) * 86_400_000),
        ValueRef::BigInt(n) => Some(apply_int_time_format(n, time_format)),
        ValueRef::Int(n) => Some(apply_int_time_format(n as i64, time_format)),
        ValueRef::UInt(n) => Some(apply_int_time_format(n as i64, time_format)),
        ValueRef::UBigInt(n) if n <= i64::MAX as u64 => {
            Some(apply_int_time_format(n as i64, time_format))
        }
        ValueRef::Text(bytes) => std::str::from_utf8(bytes).ok().and_then(|s| parse_iso8601(s).ok()),
        _ => None,
    }
}

/// DuckDB `TIMESTAMP`/`TIMESTAMPTZ` value (in `unit`) → Unix milliseconds.
fn timestamp_unit_to_ms(unit: TimeUnit, value: i64) -> i64 {
    match unit {
        TimeUnit::Second => value.saturating_mul(1000),
        TimeUnit::Millisecond => value,
        TimeUnit::Microsecond => value / 1_000,
        TimeUnit::Nanosecond => value / 1_000_000,
    }
}

/// Integer time column → ms, per `--time-format` (matches the GeoParquet path).
fn apply_int_time_format(v: i64, time_format: TimeFormat) -> i64 {
    match time_format {
        TimeFormat::UnixSec => v.saturating_mul(1000),
        // Int + iso8601 falls back to unix-ms, same as the GeoParquet reader.
        TimeFormat::UnixMs | TimeFormat::Iso8601 => v,
    }
}

/// Map one property cell to a JSON value, mirroring
/// `crate::input::extract_property_value`. SQL NULL and unmappable types
/// (raw GEOMETRY/other BLOBs, decimals, intervals, nested types, …) return
/// `None` and are dropped.
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
        ValueRef::Float(f) => serde_json::Number::from_f64(f as f64).map(Value::Number),
        ValueRef::Double(f) => serde_json::Number::from_f64(f).map(Value::Number),
        ValueRef::Text(bytes) => {
            std::str::from_utf8(bytes).ok().map(|s| Value::String(s.to_string()))
        }
        ValueRef::Timestamp(unit, value) => Some(Value::from(timestamp_unit_to_ms(unit, value))),
        ValueRef::Date32(days) => Some(Value::from((days as i64) * 86_400_000)),
        // NULL, Blob (incl. raw GEOMETRY/WKB), Decimal, Time64, Interval, and
        // the nested List/Struct/Map/Enum/Array/Union types are dropped.
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
        let q = build_tile_query("obs", "geom", "ts", [-10.0, -5.0, 10.0, 5.0], 1000, 2000);
        assert!(q.contains("ST_AsWKB(q.\"geom\")"), "{q}");
        assert!(q.contains("ST_Intersects(q.\"geom\", ST_MakeEnvelope(-10, -5, 10, 5))"), "{q}");
        assert!(q.contains("q.\"ts\" >= epoch_ms(1000::BIGINT)"), "{q}");
        assert!(q.contains("q.\"ts\" <  epoch_ms(2000::BIGINT)"), "{q}");
    }

    #[test]
    fn int_time_format_mapping() {
        assert_eq!(apply_int_time_format(5, TimeFormat::UnixSec), 5000);
        assert_eq!(apply_int_time_format(5, TimeFormat::UnixMs), 5);
        assert_eq!(apply_int_time_format(5, TimeFormat::Iso8601), 5);
    }

    #[test]
    fn timestamp_units_to_ms() {
        assert_eq!(timestamp_unit_to_ms(TimeUnit::Second, 5), 5000);
        assert_eq!(timestamp_unit_to_ms(TimeUnit::Millisecond, 5), 5);
        assert_eq!(timestamp_unit_to_ms(TimeUnit::Microsecond, 5_000), 5);
        assert_eq!(timestamp_unit_to_ms(TimeUnit::Nanosecond, 5_000_000), 5);
    }

    #[test]
    fn quotes_identifiers_safely() {
        assert_eq!(QuerySpec::quote_ident("geom"), "\"geom\"");
        assert_eq!(QuerySpec::quote_ident("we\"ird"), "\"we\"\"ird\"");
    }

    /// End-to-end decode against a real (bundled) in-memory DuckDB — no spatial
    /// extension / network needed: the geometry is supplied as a pre-made WKB
    /// `BLOB` aliased to [`WKB_ALIAS`], exercising the `ValueRef` decode path,
    /// column-name resolution, timestamp→ms, and property mapping.
    #[test]
    fn decodes_blob_geometry_time_and_props() {
        // WKB for POINT(1 2), little-endian.
        const POINT_WKB: &[u8] = &[
            0x01, // little-endian
            0x01, 0x00, 0x00, 0x00, // geometry type 1 = Point
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // x = 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y = 2.0
        ];
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
        let out = schema
            .parse_row(row, InputStrictness::Warn, InputStrictness::Warn, 0)
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

    /// Live smoke test of the actual spatial SQL the server emits — exercises
    /// `ST_Point`/`ST_AsWKB`/`ST_MakeEnvelope`/`ST_Intersects`/`epoch_ms` (tile
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
        let sql = build_tile_query("obs", "geom", "ts", [-74.5, 40.0, -73.0, 41.0], t0, t1);
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
        assert!((feats[0].lon - (-73.9)).abs() < 1e-6, "lon {}", feats[0].lon);
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
        assert!((f2[0].lon - (-73.9)).abs() < 1e-4, "reprojected lon {}", f2[0].lon);
        assert!((f2[0].lat - 40.7).abs() < 1e-4, "reprojected lat {}", f2[0].lat);

        // Metadata aggregate: bounds, [t_start, t_end] epoch-ms, count.
        let mq = build_metadata_query("obs", "geom", "ts");
        let mut stmt = conn.prepare(&mq).unwrap();
        let mut rows = stmt.query([]).unwrap();
        let row = rows.next().unwrap().unwrap();
        let min_lon: Option<f64> = row.get(0).unwrap();
        let max_lon: Option<f64> = row.get(2).unwrap();
        let t_start: Option<i64> = row.get(4).unwrap();
        let cnt: i64 = row.get(6).unwrap();
        assert_eq!(cnt, 2);
        assert!(min_lon.unwrap() <= -73.9 && max_lon.unwrap() >= 2.35, "bounds span both points");
        let noon = chrono::NaiveDate::from_ymd_opt(2024, 6, 21)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        assert_eq!(t_start.unwrap(), noon);
    }
}
