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
