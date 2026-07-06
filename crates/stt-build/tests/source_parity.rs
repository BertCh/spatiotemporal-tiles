//! Cross-source ingest parity: the GeoParquet file reader, the DuckDB reader,
//! and (gated) the PostgreSQL reader must produce the SAME `ParsedFeature`
//! stream and the SAME archive from the same logical data.
//!
//! This is the comprehensive proof for the DB-parity work — especially the
//! per-vertex `vertex_timestamps` / `vertex_values` columns the DB readers
//! previously dropped, plus NaN-float properties, null-element handling, and
//! null-geometry skip parity. See `tests/common/mod.rs` for the fixture +
//! comparator.
//!
//! DuckDB is statically bundled, so `duckdb_matches_file_*` run in CI with no
//! external service and no spatial extension (the fixture parquet is read back
//! through core `read_parquet`; WKB rides as a BLOB column). PostgreSQL parity
//! needs a live server, so it is `#[ignore]`d and gated on `STT_TEST_PG_DSN`.
//!
//! Run: `cargo test -p stt-build --features duckdb` (CI)
//!      `STT_TEST_PG_DSN=postgresql://… cargo test -p stt-build --features duckdb,postgres -- --ignored`

#![cfg(any(feature = "duckdb", feature = "postgres"))]

mod common;

use stt_build::tiler::TileConfig;

/// The fixture has 6 rows; one carries a null geometry → 5 decoded features.
const EXPECTED_FEATURES: usize = 5;

#[cfg(feature = "duckdb")]
#[test]
fn duckdb_matches_file_parsed_features() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.parquet");
    let rows = common::fixture_rows();
    common::write_fixture_parquet(&rows, &path);

    let file = common::load_file(&path);
    let duck = common::load_duckdb(&path);

    assert_eq!(file.len(), EXPECTED_FEATURES, "file feature count");
    assert_eq!(duck.len(), EXPECTED_FEATURES, "duckdb feature count");
    common::assert_features_equal(&file, &duck, "duckdb");

    // The core fix: per-vertex columns now survive the DB path. (Both linestring
    // rows carry vertex_timestamps + vertex_values; before the fix the DB reader
    // hardcoded all three to None.)
    let with_vts = duck.iter().filter(|f| f.vertex_timestamps.is_some()).count();
    let with_vvs = duck.iter().filter(|f| f.vertex_values.is_some()).count();
    assert_eq!(with_vts, 2, "duckdb must carry per-vertex timestamps for the 2 linestrings");
    assert_eq!(with_vvs, 2, "duckdb must carry per-vertex values for the 2 linestrings");
}

#[cfg(feature = "duckdb")]
#[test]
fn duckdb_matches_file_archive() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.parquet");
    let rows = common::fixture_rows();
    common::write_fixture_parquet(&rows, &path);
    let file = common::load_file(&path);
    let duck = common::load_duckdb(&path);

    // Vary per-config TileConfig fields across the parity-defining axes (zoom
    // range, temporal bucket, simplification). Encoder GLOBALS (quantize /
    // vector-group) are process-wide and operate purely downstream of
    // ParsedFeature, so their source parity is already covered by the feature-
    // equality test; varying them here would race other parallel tests.
    let configs = [
        TileConfig {
            min_zoom: 0,
            max_zoom: 6,
            temporal_bucket_ms: 3_600_000,
            ..TileConfig::default()
        },
        TileConfig {
            min_zoom: 0,
            max_zoom: 8,
            temporal_bucket_ms: 86_400_000,
            ..TileConfig::default()
        },
        TileConfig {
            min_zoom: 0,
            max_zoom: 8,
            temporal_bucket_ms: 3_600_000,
            simplify: true,
            ..TileConfig::default()
        },
    ];
    for (i, cfg) in configs.iter().enumerate() {
        let a = common::build_archive(&file, cfg);
        let b = common::build_archive(&duck, cfg);
        common::assert_archives_equal(a.path(), b.path(), &format!("duckdb cfg#{i}"));
    }
}

/// Schema-pinned property kinds: the DuckDB result-schema probe must pin the
/// exact map the GeoParquet schema pins — otherwise a column that is all-null
/// within one tile gets a column in a file build but drifts out of the layer
/// schema in a DB build.
#[cfg(feature = "duckdb")]
#[test]
fn duckdb_property_kinds_match_file() {
    use stt_build::columnar::PropertyKind;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.parquet");
    let rows = common::fixture_rows();
    common::write_fixture_parquet(&rows, &path);

    let file_kinds = stt_build::input::property_kinds(&path, "timestamp", Some("end_time"))
        .expect("file property kinds");

    let conn = duckdb::Connection::open_in_memory().expect("open in-memory duckdb");
    conn.execute_batch(&format!(
        "CREATE TABLE t AS SELECT * FROM read_parquet('{}')",
        path.display()
    ))
    .expect("read_parquet into table");
    let duck_kinds = stt_build::duckdb_input::property_kinds_for_query(
        &conn,
        "SELECT *, \"geometry\" AS __stt_wkb FROM t",
        "timestamp",
        Some("end_time"),
        "geometry",
    )
    .expect("duckdb property kinds");

    assert_eq!(file_kinds, duck_kinds, "file and DuckDB must pin identical kinds");
    // Spot-check the expected shape (and that system columns stayed out).
    assert_eq!(duck_kinds.get("mag"), Some(&PropertyKind::Numeric));
    assert_eq!(duck_kinds.get("cnt"), Some(&PropertyKind::Numeric));
    assert_eq!(duck_kinds.get("name"), Some(&PropertyKind::Categorical));
    assert_eq!(duck_kinds.get("flag"), Some(&PropertyKind::Categorical));
    assert!(!duck_kinds.contains_key("vertex_timestamps"));
    assert!(!duck_kinds.contains_key("__stt_wkb"));
    assert!(!duck_kinds.contains_key("geometry"));
}

#[cfg(feature = "postgres")]
#[test]
#[ignore = "needs a Postgres server; set STT_TEST_PG_DSN and run with -- --ignored"]
fn postgres_matches_file_parsed_features() {
    let Ok(dsn) = std::env::var("STT_TEST_PG_DSN") else {
        eprintln!("STT_TEST_PG_DSN not set — skipping postgres parity");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.parquet");
    let rows = common::fixture_rows();
    common::write_fixture_parquet(&rows, &path);

    let file = common::load_file(&path);
    let pg = common::load_postgres(&dsn, "stt_parity_features", &rows).expect("postgres load");

    assert_eq!(pg.len(), EXPECTED_FEATURES, "postgres feature count");
    common::assert_features_equal(&file, &pg, "postgres");

    let with_vts = pg.iter().filter(|f| f.vertex_timestamps.is_some()).count();
    assert_eq!(with_vts, 2, "postgres must carry per-vertex timestamps");
}

#[cfg(feature = "postgres")]
#[test]
#[ignore = "needs a Postgres server; set STT_TEST_PG_DSN and run with -- --ignored"]
fn postgres_matches_file_archive() {
    let Ok(dsn) = std::env::var("STT_TEST_PG_DSN") else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.parquet");
    let rows = common::fixture_rows();
    common::write_fixture_parquet(&rows, &path);

    let file = common::load_file(&path);
    let pg = common::load_postgres(&dsn, "stt_parity_archive", &rows).expect("postgres load");

    let cfg = TileConfig {
        min_zoom: 0,
        max_zoom: 8,
        temporal_bucket_ms: 3_600_000,
        ..TileConfig::default()
    };
    let a = common::build_archive(&file, &cfg);
    let b = common::build_archive(&pg, &cfg);
    common::assert_archives_equal(a.path(), b.path(), "postgres");
}

/// Live-PG value-decode coverage for the PostGIS reader's DB-only *superset*
/// property types — NUMERIC, DATE, INT2 — which no cross-source parity fixture
/// can carry: the GeoParquet file reader drops Parquet decimals and can't
/// express a DATE-as-property, so putting them in `fixture_rows` would break the
/// file≡DB equality the other parity tests assert. This drives `decode_property`
/// end to end for those arms against a real server — NUMERIC → nearest-f64 (via
/// the `decimal_string_to_json` conversion shared with the DuckDB DECIMAL arm),
/// DATE → UTC-midnight epoch-ms, INT2 → JSON integer — so the classification
/// unit tests in `postgres_input` are matched by a real value round-trip. Runs
/// in the `rust-postgres-parity` CI job (which executes all `--ignored` tests).
#[cfg(feature = "postgres")]
#[test]
#[ignore = "needs a Postgres server; set STT_TEST_PG_DSN and run with -- --ignored"]
fn postgres_decodes_numeric_date_and_int2_properties() {
    use postgres::{Client, NoTls};
    use stt_build::input::{InputStrictness, TimeFormat};

    let Ok(dsn) = std::env::var("STT_TEST_PG_DSN") else {
        eprintln!("STT_TEST_PG_DSN not set — skipping postgres superset-props decode");
        return;
    };

    // Own table so this can run alongside the other gated tests.
    let table = "\"stt_parity_super_props\"";
    let mut client = Client::connect(&dsn, NoTls).expect("connect");
    client
        .batch_execute(&format!(
            "DROP TABLE IF EXISTS {table}; \
             CREATE TABLE {table} ( \
                geom bytea, ts int8, price numeric, small int2, day date );"
        ))
        .expect("create table");

    // One point row. The property columns are inserted as SQL literals (all
    // test-controlled) so the test needs no rust_decimal / NaiveDate param
    // plumbing; only the WKB and time columns are bound.
    let wkb = common::wkb_point(4.9, 52.37);
    let ts: i64 = 1_600_000_000_000;
    client
        .execute(
            &format!(
                "INSERT INTO {table} (geom, ts, price, small, day) \
                 VALUES ($1, $2, 12.34, 7, DATE '2021-06-15')"
            ),
            &[&wkb, &ts],
        )
        .expect("insert");

    // Mirror the reader's expected shape: WKB aliased to the sentinel column,
    // geom_column excluded from properties (exactly as `common::load_postgres`).
    let rows = client
        .query(
            &format!("SELECT geom AS __stt_wkb, ts, price, small, day FROM {table}"),
            &[],
        )
        .expect("query");

    let feats = stt_build::postgres_input::decode_rows(
        &rows,
        "ts",
        None,
        "geom",
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .expect("decode_rows");

    client
        .batch_execute(&format!("DROP TABLE {table}"))
        .expect("drop table");

    assert_eq!(feats.len(), 1, "one valid point row → one feature");
    let props = feats[0]
        .shared_properties
        .as_ref()
        .expect("row carries decoded properties");

    // NUMERIC 12.34 → nearest f64 (shared decimal→JSON conversion).
    assert_eq!(
        props.get("price"),
        Some(&serde_json::json!(12.34)),
        "NUMERIC property decodes to a JSON number"
    );
    // INT2 7 → JSON integer.
    assert_eq!(
        props.get("small"),
        Some(&serde_json::json!(7)),
        "INT2 property decodes to a JSON integer"
    );
    // DATE 2021-06-15 → UTC-midnight epoch-ms (1_623_715_200_000).
    assert_eq!(
        props.get("day"),
        Some(&serde_json::json!(1_623_715_200_000_i64)),
        "DATE property decodes to UTC-midnight epoch-ms"
    );
}
