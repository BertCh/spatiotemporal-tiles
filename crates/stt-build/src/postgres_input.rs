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

use crate::input::{parse_iso8601, parse_wkb_geometry, InputStrictness, ParsedFeature, TimeFormat};

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
        "SELECT ST_AsEWKB({geom}) AS {WKB_ALIAS}, q.* \
         FROM ( SELECT * FROM {table} ) AS q \
         WHERE {geom} && ST_MakeEnvelope({}, {}, {}, {}, 4326) \
           AND {time} >= to_timestamp({}::double precision / 1000.0) \
           AND {time} <  to_timestamp({}::double precision / 1000.0)",
        bbox[0], bbox[1], bbox[2], bbox[3], t_start_ms, t_end_ms
    )
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
    for (i, row) in rows.iter().enumerate() {
        if let RowOutcome::Feature(f) =
            schema.parse_row(row, time_strictness, geometry_strictness, i)?
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
            match sch.parse_row(row, time_strictness, geometry_strictness, total_rows + i)? {
                RowOutcome::Feature(f) => batch.push(*f),
                RowOutcome::GeomSkip => geom_failures += 1,
            }
        }
        total_rows += rows.len();
        if total_rows % 100_000 < rows.len() {
            tracing::info!("Loaded {total_rows} rows from PostGIS...");
        }
        on_batch(batch)?;
    }
    let _ = txn.batch_execute("CLOSE stt_cur");

    if geom_failures > 0 {
        tracing::warn!(
            "{geom_failures}/{total_rows} PostGIS rows had null/unparseable geometry and were skipped"
        );
    }
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
    props: Vec<PropCol>,
    time_format: TimeFormat,
}

enum RowOutcome {
    Feature(Box<ParsedFeature>),
    GeomSkip,
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

        // Every remaining column becomes a property, except the system columns
        // (wkb, time, end-time), the original geometry column, and the
        // geometry-component coordinate names (`lon`/`lat`/`x`/`y`/…) — which
        // the GeoParquet reader also treats as geometry metadata, not
        // properties (`crate::input` `property_cols`), so the two ingest paths
        // stay consistent and tiles don't carry coordinates twice. Columns
        // whose type we can't map (PostGIS `geometry`, arrays, …) decode to
        // None at read time and are silently dropped per-row.
        let is_coord_name = |n: &str| {
            matches!(
                n.to_ascii_lowercase().as_str(),
                "lon" | "lat" | "longitude" | "latitude" | "x" | "y"
            )
        };
        let mut props = Vec::new();
        for (idx, col) in columns.iter().enumerate() {
            let name = col.name();
            if idx == wkb_idx
                || idx == time.idx
                || end_time.as_ref().map(|e| e.idx) == Some(idx)
                || name == geom_column
                || is_coord_name(name)
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
            .map(|v| apply_int_time_format(v, time_format)),
        Type::INT4 => row
            .try_get::<_, Option<i32>>(col.idx)
            .ok()
            .flatten()
            .map(|v| apply_int_time_format(v as i64, time_format)),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => row
            .try_get::<_, Option<String>>(col.idx)
            .ok()
            .flatten()
            .and_then(|s| parse_iso8601(&s).ok()),
        _ => None,
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

/// Integer time column → ms, per `--time-format` (matches the GeoParquet path).
fn apply_int_time_format(v: i64, time_format: TimeFormat) -> i64 {
    match time_format {
        TimeFormat::UnixSec => v.saturating_mul(1000),
        // Int + iso8601 falls back to unix-ms, same as the GeoParquet reader.
        TimeFormat::UnixMs | TimeFormat::Iso8601 => v,
    }
}

/// Map one property column value to a JSON value, mirroring
/// `crate::input::extract_property_value`. SQL NULL and unmappable PG types
/// (geometry, numeric, uuid, arrays, …) return `None` and are dropped.
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
            .and_then(|v| serde_json::Number::from_f64(v as f64).map(Value::Number)),
        Type::FLOAT8 => row
            .try_get::<_, Option<f64>>(col.idx)
            .ok()
            .flatten()
            .and_then(|v| serde_json::Number::from_f64(v).map(Value::Number)),
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
    fn int_time_format_mapping() {
        assert_eq!(apply_int_time_format(5, TimeFormat::UnixSec), 5000);
        assert_eq!(apply_int_time_format(5, TimeFormat::UnixMs), 5);
        assert_eq!(apply_int_time_format(5, TimeFormat::Iso8601), 5);
    }

    #[test]
    fn quotes_identifiers_safely() {
        assert_eq!(QuerySpec::quote_ident("geom"), "\"geom\"");
        assert_eq!(QuerySpec::quote_ident("we\"ird"), "\"we\"\"ird\"");
    }
}
