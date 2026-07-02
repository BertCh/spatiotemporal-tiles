//! PostGIS input source (cargo feature `postgres`).
//!
//! Reads features directly from a live PostgreSQL/PostGIS query instead of a
//! GeoParquet file, producing the exact same [`ParsedFeature`] stream the
//! GeoParquet reader emits ([`crate::input`]) — so the whole tiler/writer
//! pipeline is reused unchanged.
//!
//! The bridge is WKB: PostGIS `ST_AsEWKB(geom)` returns the same SRID-prefixed
//! EWKB blob that [`crate::input::parse_wkb_geometry`] already decodes (via
//! `geozero`). We wrap the user's table / SQL as a subquery, project the
//! geometry to a `bytea` WKB column, and read every other returned column as a
//! feature property.
//!
//! Streaming is bounded-memory via a server-side `DECLARE … CURSOR` +
//! `FETCH FORWARD <batch>` loop, mirroring [`crate::input::stream_features`].

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use geojson::Feature;
use postgres::types::Type;
use postgres::{Client, NoTls, Row};
use std::sync::Arc;

use crate::db_input_common::{
    apply_int_time_format, json_number_or_null, warn_dropped_columns, RowOutcome, VertexCoercions,
};
use crate::input::{
    parse_iso8601, parse_wkb_geometry, reject_negative_timestamp, InputStrictness, ParsedFeature,
    TimeFormat,
};

/// Alias the wrapped query projects the geometry into (as `bytea` EWKB). Public
/// so a dynamic tile server building its own per-tile query keeps the geometry
/// column name that [`decode_rows`] expects.
pub const WKB_ALIAS: &str = "__stt_wkb";
/// Default rows fetched per cursor round-trip.
pub const DEFAULT_BATCH_SIZE: usize = 8192;

/// What to read from PostGIS — either a table (we `SELECT *`) or an arbitrary
/// `SELECT` the caller supplies.
#[derive(Debug, Clone)]
pub enum QuerySource {
    /// A table (optionally schema-qualified, e.g. `public.trips`).
    Table(String),
    /// An arbitrary `SELECT …` statement.
    Sql(String),
}

/// Fully describes the PostGIS read: the source, which column holds geometry,
/// an optional `WHERE` filter, and optional reprojection to EPSG:4326.
#[derive(Debug, Clone)]
pub struct QuerySpec {
    pub source: QuerySource,
    pub geom_column: String,
    pub where_clause: Option<String>,
    /// When set, geometry is `ST_Transform`ed from this SRID to 4326. Omit when
    /// the source geometry is already lon/lat (EPSG:4326).
    pub reproject_from_srid: Option<i32>,
}

impl QuerySpec {
    /// Quote a SQL identifier (double-quote, escaping embedded quotes).
    fn quote_ident(name: &str) -> String {
        format!("\"{}\"", name.replace('"', "\"\""))
    }

    /// Build the wrapped query: project the geometry to an EWKB `bytea` column
    /// aliased [`WKB_ALIAS`], and pass every other column through as `q.*`.
    fn wrapped_query(&self) -> String {
        let inner = match &self.source {
            // A bare table name may be schema-qualified; trust the operator.
            QuerySource::Table(t) => format!("SELECT * FROM {t}"),
            QuerySource::Sql(s) => s.clone(),
        };
        let geom = format!("q.{}", Self::quote_ident(&self.geom_column));
        let wkb_expr = match self.reproject_from_srid {
            Some(srid) => {
                format!("ST_AsEWKB(ST_Transform(ST_SetSRID({geom}, {srid}), 4326))")
            }
            None => format!("ST_AsEWKB({geom})"),
        };
        let where_sql = self
            .where_clause
            .as_ref()
            .map(|w| format!(" WHERE {w}"))
            .unwrap_or_default();
        format!("SELECT {wkb_expr} AS {WKB_ALIAS}, q.* FROM ( {inner} ) AS q{where_sql}")
    }
}

/// Build a per-tile query for a dynamic server: project geometry to an EWKB
/// `bytea` aliased [`WKB_ALIAS`], pass every other column through, and filter by
/// a WGS84 bounding box (`&&` against the GiST index) and a half-open time
/// window `[t_start_ms, t_end_ms)`. The geometry / time literals are
/// server-formatted numbers (request `z/x/y/t` are parsed to integers upstream),
/// so there is no request-controlled SQL. The time column must be a
/// `timestamp`/`timestamptz` (compared via `to_timestamp`).
///
/// Feed the result rows straight to [`decode_rows`].
#[allow(clippy::too_many_arguments)]
pub fn build_tile_query(
    table: &str,
    geom_column: &str,
    time_field: &str,
    time_format: TimeFormat,
    bbox: [f64; 4],
    t_start_ms: i64,
    t_end_ms: i64,
) -> String {
    let geom = format!("q.{}", QuerySpec::quote_ident(geom_column));
    let time = format!("q.{}", QuerySpec::quote_ident(time_field));
    let time_where = pg_time_window(&time, time_format, t_start_ms, t_end_ms);
    format!(
        "SELECT ST_AsEWKB({geom}) AS {WKB_ALIAS}, q.* \
         FROM ( SELECT * FROM {table} ) AS q \
         WHERE {geom} && ST_MakeEnvelope({}, {}, {}, {}, 4326) \
           AND {time_where}",
        bbox[0], bbox[1], bbox[2], bbox[3]
    )
}

/// The half-open `[t_start_ms, t_end_ms)` predicate for the time column, chosen
/// by `--time-format` so an INTEGER epoch column serves as well as it ingests
/// (the ingest path already honours `--time-format` for integer columns).
/// `Iso8601` (the default) assumes a `timestamp`/`timestamptz` column and
/// compares via `to_timestamp`; `UnixMs`/`UnixSec` assume an integer column and
/// compare numerically (scaling seconds to ms so the ms-precision bucket bounds
/// stay exact). Literals are server-formatted numbers — no request-controlled SQL.
fn pg_time_window(time: &str, time_format: TimeFormat, t_start_ms: i64, t_end_ms: i64) -> String {
    match time_format {
        TimeFormat::Iso8601 => format!(
            "{time} >= to_timestamp({t_start_ms}::double precision / 1000.0) \
             AND {time} <  to_timestamp({t_end_ms}::double precision / 1000.0)"
        ),
        TimeFormat::UnixMs => {
            format!("{time} >= {t_start_ms} AND {time} < {t_end_ms}")
        }
        TimeFormat::UnixSec => {
            format!("{time} * 1000 >= {t_start_ms} AND {time} * 1000 < {t_end_ms}")
        }
    }
}

/// Decode a slice of PostGIS rows (from a query that projects geometry to an
/// EWKB column aliased [`WKB_ALIAS`], e.g. [`build_tile_query`]) into
/// [`ParsedFeature`]s. The per-row schema is resolved once from `rows[0]`.
#[allow(clippy::too_many_arguments)]
pub fn decode_rows(
    rows: &[Row],
    time_field: &str,
    end_time_field: Option<&str>,
    geom_column: &str,
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
) -> Result<Vec<ParsedFeature>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let schema = RowSchema::from_columns(
        rows[0].columns(),
        time_field,
        end_time_field,
        geom_column,
        time_format,
    )?;
    let mut out = Vec::with_capacity(rows.len());
    // Per-tile serve path: don't tally coercions (no end-of-read summary, no spam).
    let mut sink = VertexCoercions::default();
    for (i, row) in rows.iter().enumerate() {
        if let RowOutcome::Feature(f) =
            schema.parse_row(row, time_strictness, geometry_strictness, i, &mut sink)?
        {
            out.push(*f);
        }
    }
    Ok(out)
}

/// Eager variant: collect the whole PostGIS query into memory. Mirrors
/// [`crate::input::load_features`].
#[allow(clippy::too_many_arguments)]
pub fn load_features_postgres(
    conn: &str,
    spec: &QuerySpec,
    time_field: &str,
    end_time_field: Option<&str>,
    time_format: TimeFormat,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
) -> Result<Vec<ParsedFeature>> {
    let mut features = Vec::new();
    stream_features_postgres(
        conn,
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
    tracing::info!("Loaded {} total features from PostGIS", features.len());
    Ok(features)
}

/// Stream a PostGIS query one cursor batch at a time, invoking `on_batch` with
/// the materialised [`ParsedFeature`]s. Peak memory is bounded by `batch_size`
/// rows, mirroring [`crate::input::stream_features`].
#[allow(clippy::too_many_arguments)]
pub fn stream_features_postgres<F>(
    conn: &str,
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
    tracing::debug!("PostGIS source query: {wrapped}");

    let mut client = Client::connect(conn, NoTls)
        .with_context(|| "failed to connect to PostgreSQL (NoTls; localhost / non-TLS only)")?;
    let mut txn = client.transaction()?;

    // Server-side cursor: bounded memory regardless of result size.
    let declare = format!("DECLARE stt_cur NO SCROLL CURSOR FOR {wrapped}");
    txn.batch_execute(&declare).with_context(|| {
        "failed to open PostGIS cursor — check --table/--sql, --geom-column and the connection"
    })?;
    let fetch = format!("FETCH FORWARD {} FROM stt_cur", batch_size.max(1));

    let mut schema: Option<RowSchema> = None;
    let mut total_rows = 0usize;
    let mut geom_failures = 0usize;
    // Per-vertex array elements silently defaulted (NULL/unmappable element →
    // 0 for timestamps, NaN for values), tallied once for an end-of-read summary.
    let mut vertex_coercions = VertexCoercions::default();
    // Property columns that carried a value in ANY row — so we can warn once
    // about source columns that produced nothing (unmappable type, or all-NULL).
    let mut seen_props: std::collections::HashSet<String> = std::collections::HashSet::new();

    loop {
        let rows = txn.query(&fetch, &[]).context("PostGIS cursor FETCH failed")?;
        if rows.is_empty() {
            break;
        }
        if schema.is_none() {
            schema = Some(RowSchema::from_columns(
                rows[0].columns(),
                time_field,
                end_time_field,
                &spec.geom_column,
                time_format,
            )?);
        }
        let sch = schema.as_ref().unwrap();

        let mut batch = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            match sch.parse_row(
                row,
                time_strictness,
                geometry_strictness,
                total_rows + i,
                &mut vertex_coercions,
            )? {
                RowOutcome::Feature(f) => batch.push(*f),
                RowOutcome::GeomSkip => geom_failures += 1,
            }
        }
        total_rows += rows.len();
        if total_rows % 100_000 < rows.len() {
            tracing::info!("Loaded {total_rows} rows from PostGIS...");
        }
        // Record which property columns actually produced a value (stop once all
        // have, so there's no per-row overhead in the common all-present case).
        if seen_props.len() < sch.props.len() {
            for f in &batch {
                if let Some(props) = &f.shared_properties {
                    for k in props.keys() {
                        seen_props.insert(k.clone());
                    }
                }
            }
        }
        on_batch(batch)?;
    }
    let _ = txn.batch_execute("CLOSE stt_cur");

    if geom_failures > 0 {
        tracing::warn!(
            "{geom_failures}/{total_rows} PostGIS rows had null/unparseable geometry and were skipped"
        );
    }
    if let Some(sch) = schema.as_ref() {
        warn_dropped_columns(
            sch.props
                .iter()
                .map(|p| (p.name.as_str(), Some(p.ty.name().to_string()))),
            &seen_props,
            total_rows,
            "PostGIS",
            "geometry/uuid/numeric/array",
        );
    }
    vertex_coercions.warn("PostGIS");
    Ok(())
}

/// A time column resolved against the result set: its index and PG type.
struct TimeCol {
    idx: usize,
    ty: Type,
}

/// A property column: index, output name, and PG type.
struct PropCol {
    idx: usize,
    name: String,
    ty: Type,
}

/// Result-set schema resolved once from the first batch's column metadata.
struct RowSchema {
    wkb_idx: usize,
    time: TimeCol,
    end_time: Option<TimeCol>,
    /// Optional per-vertex array columns, matching the GeoParquet reader's
    /// hardcoded `vertex_timestamps` / `vertex_values` / `vertex_value_matrix`
    /// columns (LineString trajectory per-vertex timing/values + animated
    /// static-geometry overviews). `TimeCol` is reused purely as an
    /// `{ idx, ty }` pair. `None` when the source query exposes no such column.
    vertex_timestamps: Option<TimeCol>,
    vertex_values: Option<TimeCol>,
    vertex_value_matrix: Option<TimeCol>,
    props: Vec<PropCol>,
    time_format: TimeFormat,
}

impl RowSchema {
    fn from_columns(
        columns: &[postgres::Column],
        time_field: &str,
        end_time_field: Option<&str>,
        geom_column: &str,
        time_format: TimeFormat,
    ) -> Result<Self> {
        let find = |name: &str| columns.iter().position(|c| c.name() == name);

        let wkb_idx = find(WKB_ALIAS).ok_or_else(|| {
            anyhow::anyhow!("internal: wrapped query did not project the {WKB_ALIAS} column")
        })?;
        let time_idx = find(time_field).ok_or_else(|| {
            anyhow::anyhow!("--time-field '{time_field}' not found in the PostGIS result columns")
        })?;
        let time = TimeCol {
            idx: time_idx,
            ty: columns[time_idx].type_().clone(),
        };
        let end_time = match end_time_field {
            Some(f) => {
                let idx = find(f).ok_or_else(|| {
                    anyhow::anyhow!("--end-time-field '{f}' not found in the PostGIS result columns")
                })?;
                Some(TimeCol {
                    idx,
                    ty: columns[idx].type_().clone(),
                })
            }
            None => None,
        };

        // Optional per-vertex array columns, detected by the exact column names
        // the GeoParquet reader hardcodes. `TimeCol` is reused as `{ idx, ty }`.
        let array_col = |name: &str| {
            find(name).map(|idx| TimeCol {
                idx,
                ty: columns[idx].type_().clone(),
            })
        };
        let vertex_timestamps = array_col("vertex_timestamps");
        let vertex_values = array_col("vertex_values");
        let vertex_value_matrix = array_col("vertex_value_matrix");

        // Every remaining column becomes a property, except the system columns
        // (wkb, time, end-time), the original geometry column, the per-vertex
        // array columns, and the geometry-component coordinate names — all
        // excluded via the SHARED `crate::input` predicates so every input
        // adaptor (file/PostGIS/DuckDB) excludes the identical set and tiles
        // don't carry coordinates twice. Columns whose type we can't map
        // (PostGIS `geometry`, unmapped arrays, …) decode to None at read time
        // and are silently dropped per-row.
        let mut props = Vec::new();
        for (idx, col) in columns.iter().enumerate() {
            let name = col.name();
            if idx == wkb_idx
                || idx == time.idx
                || end_time.as_ref().map(|e| e.idx) == Some(idx)
                || name == geom_column
                || crate::input::is_coordinate_column_name(name)
                || crate::input::is_vertex_metadata_column(name)
            {
                continue;
            }
            props.push(PropCol {
                idx,
                name: name.to_string(),
                ty: col.type_().clone(),
            });
        }

        Ok(RowSchema {
            wkb_idx,
            time,
            end_time,
            vertex_timestamps,
            vertex_values,
            vertex_value_matrix,
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
        // Geometry (EWKB bytea -> GeoJSON + centroid lon/lat).
        let wkb: Option<Vec<u8>> = row.try_get(self.wkb_idx).ok();
        let parsed = wkb.as_deref().and_then(parse_wkb_geometry);
        let Some((geometry, lon, lat)) = parsed else {
            if geometry_strictness == InputStrictness::Strict {
                anyhow::bail!(
                    "row {row_no}: null or unparseable geometry (rerun without --strict-geometry to skip)"
                );
            }
            return Ok(RowOutcome::GeomSkip);
        };

        // Timestamp.
        let timestamp = match decode_time(row, &self.time, self.time_format, row_no)? {
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
        let end_timestamp = match &self.end_time {
            Some(tc) => decode_time(row, tc, self.time_format, row_no)?,
            None => None,
        };

        // Per-vertex arrays (None when the column is absent or SQL NULL),
        // mirroring the GeoParquet reader's `vertex_*` columns.
        let vertex_timestamps = match &self.vertex_timestamps {
            Some(c) => decode_u64_ms_array(row, c, row_no, &mut coercions.timestamps)?,
            None => None,
        };
        let vertex_values = match &self.vertex_values {
            Some(c) => decode_f32_array(row, c, &mut coercions.values),
            None => None,
        };
        let vertex_value_matrix = match &self.vertex_value_matrix {
            Some(c) => decode_f32_array(row, c, &mut coercions.values),
            None => None,
        };

        // Properties.
        let mut properties = serde_json::Map::new();
        for p in &self.props {
            if let Some(value) = decode_property(row, p) {
                properties.insert(p.name.clone(), value);
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
fn decode_time(row: &Row, col: &TimeCol, time_format: TimeFormat, row_no: usize) -> Result<Option<u64>> {
    let ms: Option<i64> = match col.ty {
        Type::TIMESTAMPTZ => row
            .try_get::<_, Option<DateTime<Utc>>>(col.idx)
            .ok()
            .flatten()
            .map(|dt| dt.timestamp_millis()),
        Type::TIMESTAMP => row
            .try_get::<_, Option<NaiveDateTime>>(col.idx)
            .ok()
            .flatten()
            .map(|dt| dt.and_utc().timestamp_millis()),
        Type::DATE => row
            .try_get::<_, Option<NaiveDate>>(col.idx)
            .ok()
            .flatten()
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|dt| dt.and_utc().timestamp_millis()),
        Type::INT8 => row
            .try_get::<_, Option<i64>>(col.idx)
            .ok()
            .flatten()
            .map(|v| apply_int_time_format(v, time_format, row_no))
            .transpose()?,
        Type::INT4 => row
            .try_get::<_, Option<i32>>(col.idx)
            .ok()
            .flatten()
            .map(|v| apply_int_time_format(v as i64, time_format, row_no))
            .transpose()?,
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => row
            .try_get::<_, Option<String>>(col.idx)
            .ok()
            .flatten()
            .and_then(|s| parse_iso8601(&s).ok()),
        _ => None,
    };
    match ms {
        Some(v) => {
            reject_negative_timestamp(row_no, v)?;
            Ok(Some(v as u64))
        }
        None => Ok(None),
    }
}

/// Decode a per-vertex timestamp array column to `Vec<u64>` ms-since-epoch,
/// mirroring the GeoParquet reader's `vertex_timestamps` rules: integer arrays
/// are RAW ms (the per-vertex column is never reinterpreted via `--time-format`,
/// matching `extract_vertex_timestamps_from_batch`), timestamp/date arrays
/// convert to UTC ms, a whole-cell SQL NULL yields `None`, a NULL element
/// becomes `0` (preserving per-vertex alignment), and any negative (pre-1970)
/// value is a hard error in both strictness modes (the shared stt-core guard).
fn decode_u64_ms_array(
    row: &Row,
    col: &TimeCol,
    row_no: usize,
    coerced: &mut usize,
) -> Result<Option<Vec<u64>>> {
    // Count each element that is NULL (or, for DATE, an unrepresentable value) —
    // it is silently defaulted to 0 below, so the summary can flag the gaps.
    let mut count_nulls = |nulls: usize| *coerced += nulls;
    let ms: Option<Vec<i64>> = match col.ty {
        Type::INT8_ARRAY => row
            .try_get::<_, Option<Vec<Option<i64>>>>(col.idx)
            .ok()
            .flatten()
            .map(|v| {
                count_nulls(v.iter().filter(|x| x.is_none()).count());
                v.into_iter().map(|x| x.unwrap_or(0)).collect()
            }),
        Type::INT4_ARRAY => row
            .try_get::<_, Option<Vec<Option<i32>>>>(col.idx)
            .ok()
            .flatten()
            .map(|v| {
                count_nulls(v.iter().filter(|x| x.is_none()).count());
                v.into_iter().map(|x| x.map(i64::from).unwrap_or(0)).collect()
            }),
        Type::TIMESTAMP_ARRAY => row
            .try_get::<_, Option<Vec<Option<NaiveDateTime>>>>(col.idx)
            .ok()
            .flatten()
            .map(|v| {
                count_nulls(v.iter().filter(|x| x.is_none()).count());
                v.into_iter()
                    .map(|x| x.map(|t| t.and_utc().timestamp_millis()).unwrap_or(0))
                    .collect()
            }),
        Type::TIMESTAMPTZ_ARRAY => row
            .try_get::<_, Option<Vec<Option<DateTime<Utc>>>>>(col.idx)
            .ok()
            .flatten()
            .map(|v| {
                count_nulls(v.iter().filter(|x| x.is_none()).count());
                v.into_iter()
                    .map(|x| x.map(|t| t.timestamp_millis()).unwrap_or(0))
                    .collect()
            }),
        Type::DATE_ARRAY => row
            .try_get::<_, Option<Vec<Option<NaiveDate>>>>(col.idx)
            .ok()
            .flatten()
            .map(|v| {
                count_nulls(
                    v.iter()
                        .filter(|x| {
                            x.and_then(|d| d.and_hms_opt(0, 0, 0)).is_none()
                        })
                        .count(),
                );
                v.into_iter()
                    .map(|x| {
                        x.and_then(|d| d.and_hms_opt(0, 0, 0))
                            .map(|t| t.and_utc().timestamp_millis())
                            .unwrap_or(0)
                    })
                    .collect()
            }),
        _ => None,
    };
    match ms {
        Some(v) => {
            for &x in &v {
                reject_negative_timestamp(row_no, x)?;
            }
            Ok(Some(v.into_iter().map(|x| x as u64).collect()))
        }
        None => Ok(None),
    }
}

/// Decode a per-vertex float array column to `Vec<f32>`, mirroring the
/// GeoParquet reader's `vertex_values` rules: `float8[]`/`float4[]` map to f32,
/// a whole-cell SQL NULL yields `None`, and a NULL element becomes `f32::NAN`
/// (preserving per-vertex alignment).
fn decode_f32_array(row: &Row, col: &TimeCol, coerced: &mut usize) -> Option<Vec<f32>> {
    match col.ty {
        Type::FLOAT8_ARRAY => row
            .try_get::<_, Option<Vec<Option<f64>>>>(col.idx)
            .ok()
            .flatten()
            .map(|v| {
                *coerced += v.iter().filter(|x| x.is_none()).count();
                v.into_iter()
                    .map(|x| x.map(|n| n as f32).unwrap_or(f32::NAN))
                    .collect()
            }),
        Type::FLOAT4_ARRAY => row
            .try_get::<_, Option<Vec<Option<f32>>>>(col.idx)
            .ok()
            .flatten()
            .map(|v| {
                *coerced += v.iter().filter(|x| x.is_none()).count();
                v.into_iter().map(|x| x.unwrap_or(f32::NAN)).collect()
            }),
        _ => None,
    }
}

/// Map one property column value to a JSON value. SQL NULL and unmappable PG
/// types (geometry, numeric, uuid, arrays, …) return `None` and are dropped.
///
/// This is a documented **superset** of `crate::input::extract_property_value`,
/// not an exact mirror: it shares the six core types (bool / Int32 / Int64 /
/// Float32 / Float64 / text) and the NaN→JSON-null rule, but additionally maps
/// `INT2`, `TIMESTAMP`/`TIMESTAMPTZ`/`DATE` (→ epoch-ms numbers) and
/// `JSON`/`JSONB` — types a GeoParquet file can't express the same way, so they
/// only ever add properties a file build couldn't carry, never diverge on the
/// common type set. The PostGIS and DuckDB readers stay in lockstep on that
/// common set; `JSON`/`JSONB` parsed to *nested* JSON is PostGIS-only, because
/// duckdb-rs surfaces DuckDB's JSON logical type as plain `ValueRef::Text`
/// (arrives as a string property there — see db-input-adaptors.md).
fn decode_property(row: &Row, col: &PropCol) -> Option<serde_json::Value> {
    use serde_json::Value;
    match col.ty {
        Type::BOOL => row
            .try_get::<_, Option<bool>>(col.idx)
            .ok()
            .flatten()
            .map(Value::Bool),
        Type::INT2 => row
            .try_get::<_, Option<i16>>(col.idx)
            .ok()
            .flatten()
            .map(|v| Value::from(v as i64)),
        Type::INT4 => row
            .try_get::<_, Option<i32>>(col.idx)
            .ok()
            .flatten()
            .map(|v| Value::from(v as i64)),
        Type::INT8 => row
            .try_get::<_, Option<i64>>(col.idx)
            .ok()
            .flatten()
            .map(Value::from),
        Type::FLOAT4 => row
            .try_get::<_, Option<f32>>(col.idx)
            .ok()
            .flatten()
            .map(|v| json_number_or_null(v as f64)),
        Type::FLOAT8 => row
            .try_get::<_, Option<f64>>(col.idx)
            .ok()
            .flatten()
            .map(json_number_or_null),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => row
            .try_get::<_, Option<String>>(col.idx)
            .ok()
            .flatten()
            .map(Value::String),
        Type::TIMESTAMPTZ => row
            .try_get::<_, Option<DateTime<Utc>>>(col.idx)
            .ok()
            .flatten()
            .map(|dt| Value::from(dt.timestamp_millis())),
        Type::TIMESTAMP => row
            .try_get::<_, Option<NaiveDateTime>>(col.idx)
            .ok()
            .flatten()
            .map(|dt| Value::from(dt.and_utc().timestamp_millis())),
        Type::DATE => row
            .try_get::<_, Option<NaiveDate>>(col.idx)
            .ok()
            .flatten()
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|dt| Value::from(dt.and_utc().timestamp_millis())),
        Type::JSON | Type::JSONB => row.try_get::<_, Option<Value>>(col.idx).ok().flatten(),
        // Unmappable (geometry, numeric, uuid, arrays, …) — drop silently.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_table_query_with_ewkb_projection() {
        let spec = QuerySpec {
            source: QuerySource::Table("public.trips".into()),
            geom_column: "geom".into(),
            where_clause: None,
            reproject_from_srid: None,
        };
        let q = spec.wrapped_query();
        assert!(q.contains("ST_AsEWKB(q.\"geom\")"), "{q}");
        assert!(q.contains(WKB_ALIAS), "{q}");
        assert!(q.contains("FROM ( SELECT * FROM public.trips ) AS q"), "{q}");
    }

    #[test]
    fn wraps_sql_with_reprojection_and_where() {
        let spec = QuerySpec {
            source: QuerySource::Sql("SELECT * FROM obs WHERE valid".into()),
            geom_column: "the_geom".into(),
            where_clause: Some("ts > now() - interval '1 day'".into()),
            reproject_from_srid: Some(3857),
        };
        let q = spec.wrapped_query();
        assert!(
            q.contains("ST_AsEWKB(ST_Transform(ST_SetSRID(q.\"the_geom\", 3857), 4326))"),
            "{q}"
        );
        assert!(q.contains("WHERE ts > now()"), "{q}");
    }

    #[test]
    fn quotes_identifiers_safely() {
        assert_eq!(QuerySpec::quote_ident("geom"), "\"geom\"");
        assert_eq!(QuerySpec::quote_ident("we\"ird"), "\"we\"\"ird\"");
    }

    #[test]
    fn tile_query_time_filter_honors_time_format() {
        let bbox = [-10.0, -5.0, 10.0, 5.0];
        // Default (Iso8601 → timestamp column): to_timestamp comparison.
        let iso = build_tile_query("obs", "geom", "ts", TimeFormat::Iso8601, bbox, 1000, 2000);
        assert!(iso.contains("to_timestamp(1000::double precision / 1000.0)"), "{iso}");
        assert!(iso.contains("to_timestamp(2000::double precision / 1000.0)"), "{iso}");

        // Integer-epoch-ms column: numeric comparison, no to_timestamp.
        let ms = build_tile_query("obs", "geom", "ts", TimeFormat::UnixMs, bbox, 1000, 2000);
        assert!(ms.contains("q.\"ts\" >= 1000 AND q.\"ts\" < 2000"), "{ms}");
        assert!(!ms.contains("to_timestamp"), "{ms}");

        // Seconds: scaled ×1000 so the ms bucket bounds stay exact.
        let sec = build_tile_query("obs", "geom", "ts", TimeFormat::UnixSec, bbox, 1000, 2000);
        assert!(sec.contains("q.\"ts\" * 1000 >= 1000 AND q.\"ts\" * 1000 < 2000"), "{sec}");
    }
}
