//! Shared fixture + comparator for the cross-source ingest-parity suite
//! (`source_parity.rs`). One logical dataset is expressed once as Rust values
//! ([`fixture_rows`]) and materialised into:
//!   * a GeoParquet file (the canonical `input.rs` reader), and
//!   * an in-memory DuckDB table (via core `read_parquet`, no spatial extension), and
//!   * (gated) a live PostgreSQL table.
//!
//! All three must decode to the SAME [`ParsedFeature`] stream and build the SAME
//! archive. The fixture deliberately spans every parity-relevant shape: points
//! AND linestrings, the per-vertex `vertex_timestamps` / `vertex_values` columns
//! (the data the DB readers previously dropped, including a NULL element), every
//! common property type, a NaN float property, and a null geometry (skip parity).
//!
//! Comparison granularity is order-independent and robust (per the project's
//! non-deterministic-encoding caveats): exact `ParsedFeature` equality after a
//! stable sort, plus the per-tile `(zoom,x,y,time_start,time_end,feature_count)`
//! key-set of the built archive. Raw bytes / compressed size are NOT compared.
//!
//! The fixture uses only the property/time types BOTH the file reader and the DB
//! readers map identically (Int64 ms time, Float64 / Int64 / Utf8 / Bool props).
//! The DB readers additionally accept a broader type set (Int16, unsigned,
//! HugeInt, JSON, Date/Timestamp-as-property) — an intentional, documented
//! superset that a GeoParquet file can't express the same way, so it is out of
//! scope for a "same logical data" parity check.

#![allow(dead_code)] // each integration test uses a subset of these helpers.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Float64Array, Float64Builder, Int64Array, Int64Builder,
    ListBuilder,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use stt_build::input::{self, InputStrictness, ParsedFeature, TimeFormat};
use stt_build::tiler::{generate_tiles_streaming, TileConfig};
use stt_core::metadata::Metadata;
use stt_core::{BlobOrdering, PackWriter, PackedReader};

/// One fixture row. `wkb == None` is a null geometry (skip-parity case). `vts` /
/// `vvs` are the per-vertex arrays (None = no per-vertex column for this row);
/// `vvs` elements are `Option` so a NULL element (→ NaN) can be expressed.
pub struct Row {
    pub wkb: Option<Vec<u8>>,
    pub ts: i64,
    pub end: i64,
    pub mag: f64,
    pub cnt: i64,
    pub name: String,
    pub flag: bool,
    pub vts: Option<Vec<i64>>,
    pub vvs: Option<Vec<Option<f64>>>,
}

/// Little-endian ISO WKB for a 2D point.
pub fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = vec![0x01, 0x01, 0x00, 0x00, 0x00];
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

/// Little-endian ISO WKB for a 2D LineString.
pub fn wkb_linestring(coords: &[(f64, f64)]) -> Vec<u8> {
    let mut b = vec![0x01];
    b.extend_from_slice(&2u32.to_le_bytes()); // geometry type 2 = LineString
    b.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for (x, y) in coords {
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
    }
    b
}

const T0: i64 = 1_700_000_000_000;

/// The canonical parity fixture (6 rows; one has a null geometry → 5 features).
pub fn fixture_rows() -> Vec<Row> {
    vec![
        Row {
            wkb: Some(wkb_point(-122.4, 37.7)),
            ts: T0,
            end: T0,
            mag: 1.5,
            cnt: 10,
            name: "alpha".into(),
            flag: true,
            vts: None,
            vvs: None,
        },
        // NaN float property → JSON null (key present) in every reader.
        Row {
            wkb: Some(wkb_point(2.35, 48.85)),
            ts: T0 + 3_600_000,
            end: T0 + 3_600_000,
            mag: f64::NAN,
            cnt: 20,
            name: "bravo".into(),
            flag: false,
            vts: None,
            vvs: None,
        },
        // LineString WITH per-vertex timestamps + values (3 vertices).
        Row {
            wkb: Some(wkb_linestring(&[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])),
            ts: T0 + 7_200_000,
            end: T0 + 7_200_000 + 2000,
            mag: 3.0,
            cnt: 30,
            name: "charlie".into(),
            flag: true,
            vts: Some(vec![
                T0 + 7_200_000,
                T0 + 7_200_000 + 1000,
                T0 + 7_200_000 + 2000,
            ]),
            vvs: Some(vec![Some(0.1), Some(0.2), Some(0.3)]),
        },
        // LineString WITH a NULL per-vertex value element (→ NaN) (2 vertices).
        Row {
            wkb: Some(wkb_linestring(&[(10.0, 10.0), (11.0, 11.0)])),
            ts: T0 + 10_800_000,
            end: T0 + 10_800_000 + 5000,
            mag: 4.0,
            cnt: 40,
            name: "delta".into(),
            flag: false,
            vts: Some(vec![T0 + 10_800_000, T0 + 10_800_000 + 5000]),
            vvs: Some(vec![None, Some(0.5)]),
        },
        // Null geometry → skipped by every reader (never tiled at 0,0).
        Row {
            wkb: None,
            ts: T0 + 14_400_000,
            end: T0 + 14_400_000,
            mag: 9.0,
            cnt: 50,
            name: "echo".into(),
            flag: true,
            vts: None,
            vvs: None,
        },
        Row {
            wkb: Some(wkb_point(-73.9, 40.7)),
            ts: T0 + 18_000_000,
            end: T0 + 18_000_000,
            mag: 2.0,
            cnt: 60,
            name: "foxtrot".into(),
            flag: false,
            vts: None,
            vvs: None,
        },
    ]
}

/// Materialise the fixture as a single-batch GeoParquet file. The time columns
/// are Int64 unix-ms; the per-vertex columns are `List<Int64>` / `List<Float64>`
/// (decoded raw-ms / f32, matching both readers).
pub fn write_fixture_parquet(rows: &[Row], path: &std::path::Path) {
    let geom: Vec<Option<&[u8]>> = rows.iter().map(|r| r.wkb.as_deref()).collect();
    let geom_arr = Arc::new(BinaryArray::from_opt_vec(geom)) as ArrayRef;
    let ts_arr = Arc::new(Int64Array::from(
        rows.iter().map(|r| r.ts).collect::<Vec<_>>(),
    )) as ArrayRef;
    let end_arr = Arc::new(Int64Array::from(
        rows.iter().map(|r| r.end).collect::<Vec<_>>(),
    )) as ArrayRef;
    let mag_arr = Arc::new(Float64Array::from(
        rows.iter().map(|r| r.mag).collect::<Vec<_>>(),
    )) as ArrayRef;
    let cnt_arr = Arc::new(Int64Array::from(
        rows.iter().map(|r| r.cnt).collect::<Vec<_>>(),
    )) as ArrayRef;
    let name_arr = Arc::new(arrow::array::StringArray::from(
        rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
    )) as ArrayRef;
    let flag_arr = Arc::new(BooleanArray::from(
        rows.iter().map(|r| r.flag).collect::<Vec<_>>(),
    )) as ArrayRef;

    let mut vts_b = ListBuilder::new(Int64Builder::new());
    for r in rows {
        match &r.vts {
            Some(v) => {
                for &x in v {
                    vts_b.values().append_value(x);
                }
                vts_b.append(true);
            }
            None => vts_b.append(false),
        }
    }
    let vts_arr = Arc::new(vts_b.finish()) as ArrayRef;

    let mut vvs_b = ListBuilder::new(Float64Builder::new());
    for r in rows {
        match &r.vvs {
            Some(v) => {
                for x in v {
                    match x {
                        Some(f) => vvs_b.values().append_value(*f),
                        None => vvs_b.values().append_null(),
                    }
                }
                vvs_b.append(true);
            }
            None => vvs_b.append(false),
        }
    }
    let vvs_arr = Arc::new(vvs_b.finish()) as ArrayRef;

    let cols: Vec<(&str, ArrayRef)> = vec![
        ("geometry", geom_arr),
        ("timestamp", ts_arr),
        ("end_time", end_arr),
        ("mag", mag_arr),
        ("cnt", cnt_arr),
        ("name", name_arr),
        ("flag", flag_arr),
        ("vertex_timestamps", vts_arr),
        ("vertex_values", vvs_arr),
    ];
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = cols.iter().map(|(_, a)| a.clone()).collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();

    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Decode the fixture via the canonical GeoParquet file reader.
pub fn load_file(path: &std::path::Path) -> Vec<ParsedFeature> {
    input::load_features(
        path,
        "timestamp",
        Some("end_time"),
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .expect("file reader")
}

/// Decode the fixture via DuckDB — read the parquet back through core
/// `read_parquet` (no spatial extension) and decode the WKB BLOB geometry with
/// the same `parse_wkb_geometry` the file path uses.
#[cfg(feature = "duckdb")]
pub fn load_duckdb(path: &std::path::Path) -> Vec<ParsedFeature> {
    use stt_build::duckdb_input;
    let conn = duckdb::Connection::open_in_memory().expect("open in-memory duckdb");
    conn.execute_batch(&format!(
        "CREATE TABLE t AS SELECT * FROM read_parquet('{}')",
        path.display()
    ))
    .expect("read_parquet into table");
    // Alias the WKB BLOB column to the geometry sentinel the decoder expects; the
    // original `geometry` column is excluded as the geom_column.
    let sql = "SELECT *, \"geometry\" AS __stt_wkb FROM t";
    duckdb_input::decode_query(
        &conn,
        sql,
        "timestamp",
        Some("end_time"),
        "geometry",
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .expect("duckdb decode")
}

/// Decode the fixture via a live PostgreSQL table (raw `bytea` WKB + native
/// arrays — no PostGIS functions needed). Loads into a temp table, decodes via
/// `decode_rows`, and drops the table.
#[cfg(feature = "postgres")]
pub fn load_postgres(dsn: &str, table: &str, rows: &[Row]) -> anyhow::Result<Vec<ParsedFeature>> {
    use postgres::{Client, NoTls};
    use stt_build::postgres_input;

    // `table` is a test-controlled literal (not user input); quote it defensively
    // and give each test its own table so the gated tests can run in parallel.
    let t = format!("\"{}\"", table.replace('"', "\"\""));
    let mut client = Client::connect(dsn, NoTls)?;
    client.batch_execute(&format!(
        "DROP TABLE IF EXISTS {t}; \
         CREATE TABLE {t} ( \
            geom bytea, ts int8, end_time int8, mag float8, cnt int8, \
            name text, flag bool, vertex_timestamps int8[], vertex_values float8[] );",
    ))?;
    let stmt = client.prepare(&format!(
        "INSERT INTO {t} \
         (geom, ts, end_time, mag, cnt, name, flag, vertex_timestamps, vertex_values) \
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
    ))?;
    for r in rows {
        client.execute(
            &stmt,
            &[
                &r.wkb, &r.ts, &r.end, &r.mag, &r.cnt, &r.name, &r.flag, &r.vts, &r.vvs,
            ],
        )?;
    }
    let pg_rows = client.query(
        &format!(
            "SELECT geom AS __stt_wkb, ts, end_time, mag, cnt, name, flag, \
                    vertex_timestamps, vertex_values \
             FROM {t}"
        ),
        &[],
    )?;
    let feats = postgres_input::decode_rows(
        &pg_rows,
        "ts",
        Some("end_time"),
        "geom",
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )?;
    client.batch_execute(&format!("DROP TABLE {t}"))?;
    Ok(feats)
}

/// NaN-aware `Vec<f32>` equality (NaN == NaN, for per-vertex value parity).
fn opt_f32_vec_eq(a: &Option<Vec<f32>>, b: &Option<Vec<f32>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|(p, q)| (p.is_nan() && q.is_nan()) || p == q)
        }
        _ => false,
    }
}

/// Assert two reader outputs are the SAME feature set, field by field, after a
/// stable sort (order-independent). `label` names the non-canonical source.
pub fn assert_features_equal(file: &[ParsedFeature], other: &[ParsedFeature], label: &str) {
    assert_eq!(
        file.len(),
        other.len(),
        "{label}: feature count differs (file={}, {label}={})",
        file.len(),
        other.len()
    );
    let key = |f: &ParsedFeature| (f.timestamp, f.lon.to_bits(), f.lat.to_bits());
    let mut a: Vec<&ParsedFeature> = file.iter().collect();
    let mut b: Vec<&ParsedFeature> = other.iter().collect();
    a.sort_by_key(|f| key(f));
    b.sort_by_key(|f| key(f));

    for (i, (fa, fb)) in a.iter().zip(&b).enumerate() {
        assert_eq!(fa.timestamp, fb.timestamp, "{label} row {i}: timestamp");
        assert_eq!(
            fa.end_timestamp, fb.end_timestamp,
            "{label} row {i}: end_timestamp"
        );
        assert!(
            (fa.lon - fb.lon).abs() < 1e-9 && (fa.lat - fb.lat).abs() < 1e-9,
            "{label} row {i}: centroid (file={:?}, {label}={:?})",
            (fa.lon, fa.lat),
            (fb.lon, fb.lat)
        );
        assert_eq!(
            fa.geojson.geometry, fb.geojson.geometry,
            "{label} row {i}: geometry"
        );
        assert_eq!(
            fa.vertex_timestamps, fb.vertex_timestamps,
            "{label} row {i}: vertex_timestamps"
        );
        assert!(
            opt_f32_vec_eq(&fa.vertex_values, &fb.vertex_values),
            "{label} row {i}: vertex_values (file={:?}, {label}={:?})",
            fa.vertex_values,
            fb.vertex_values
        );
        assert!(
            opt_f32_vec_eq(&fa.vertex_value_matrix, &fb.vertex_value_matrix),
            "{label} row {i}: vertex_value_matrix"
        );
        assert_eq!(
            fa.shared_properties.as_ref().map(|p| p.to_map()),
            fb.shared_properties.as_ref().map(|p| p.to_map()),
            "{label} row {i}: properties"
        );
    }
}

/// Build a packed dataset from a feature set with the given tile config,
/// returning the temp dir. Bounds/time-range derive from the features exactly
/// as the CLI does.
pub fn build_archive(features: &[ParsedFeature], config: &TileConfig) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = PackWriter::create(dir.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
    generate_tiles_streaming(features, config, &mut writer, 2).unwrap();
    let (bounds, time_range) = input::calculate_bounds(features).unwrap();
    let metadata = Metadata::new("parity")
        .with_bounds(bounds)
        .with_time_range(time_range)
        .with_zoom_levels(config.min_zoom, config.max_zoom)
        .with_temporal_bucket_ms(config.temporal_bucket_ms);
    writer.finalize(&metadata).unwrap();
    dir
}

/// The per-tile identity key-set of an archive: `(zoom, x, y, time_start,
/// time_end, feature_count)` for every tile — order-independent and robust to
/// within-tile feature ordering (which legitimately differs by source).
pub fn archive_tile_keys(
    dir: &std::path::Path,
) -> std::collections::BTreeSet<(u8, u32, u32, i64, i64, u32)> {
    let reader = PackedReader::open(dir.join("manifest.json")).unwrap();
    reader
        .entries()
        .iter()
        .map(|e| (e.zoom, e.x, e.y, e.time_start, e.time_end, e.feature_count))
        .collect()
}

/// Assert two archives carry the identical set of tiles (keys + per-tile feature
/// counts). Raw bytes / compressed sizes are intentionally NOT compared.
pub fn assert_archives_equal(a: &std::path::Path, b: &std::path::Path, label: &str) {
    let ka = archive_tile_keys(a);
    let kb = archive_tile_keys(b);
    assert_eq!(
        ka,
        kb,
        "{label}: archive tile key-sets differ ({} vs {} tiles)",
        ka.len(),
        kb.len()
    );
}
