//! Regression lock for the per-vertex timestamp unit parity fix.
//!
//! A `vertex_timestamps` column whose child is a microsecond/nanosecond
//! `Timestamp` (the natural shape of a DuckDB `TIMESTAMP[]` exported to Parquet)
//! must decode to the SAME ms values through the GeoParquet file reader and the
//! DuckDB reader. Before the fix the file reader hard-errored on a non-ms
//! Timestamp child while DuckDB silently rescaled it — an asymmetry where one
//! path built and the other failed. This pins both to "accept + scale to ms".

#![cfg(feature = "duckdb")]

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, Int64Array, ListBuilder, TimestampMicrosecondBuilder,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use stt_build::input::{load_features, InputStrictness, TimeFormat};

fn wkb_linestring(coords: &[(f64, f64)]) -> Vec<u8> {
    let mut b = vec![0x01];
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for (x, y) in coords {
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
    }
    b
}

fn wkb_point(x: f64, y: f64) -> Vec<u8> {
    let mut b = vec![0x01];
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&x.to_le_bytes());
    b.extend_from_slice(&y.to_le_bytes());
    b
}

/// Write a single-row GeoParquet with a WKB point `geometry` column and one
/// scalar `start_time` column of the given Arrow type; return the file path.
fn write_scalar_time_parquet(dir: &std::path::Path, label: &str, start_time: ArrayRef) -> std::path::PathBuf {
    let path = dir.join(format!("scalar_{label}.parquet"));
    let geom = Arc::new(BinaryArray::from_opt_vec(vec![Some(wkb_point(1.0, 2.0).as_slice())])) as ArrayRef;
    let cols: Vec<(&str, ArrayRef)> = vec![("geometry", geom), ("start_time", start_time)];
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch =
        RecordBatch::try_new(schema.clone(), cols.iter().map(|(_, a)| a.clone()).collect()).unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

#[test]
fn microsecond_vertex_timestamps_parity_file_vs_duckdb() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("us_vertex.parquet");

    // One LineString with 3 vertices; vertex_timestamps as Timestamp(µs).
    let t_ms: [i64; 3] = [1_700_000_000_000, 1_700_000_001_000, 1_700_000_002_000];
    let t_us: Vec<i64> = t_ms.iter().map(|m| m * 1000).collect();

    let geom = Arc::new(BinaryArray::from_opt_vec(vec![Some(
        wkb_linestring(&[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]).as_slice(),
    )])) as ArrayRef;
    let ts = Arc::new(Int64Array::from(vec![t_ms[0]])) as ArrayRef;
    let mut vb = ListBuilder::new(TimestampMicrosecondBuilder::new());
    for v in &t_us {
        vb.values().append_value(*v);
    }
    vb.append(true);
    let vts = Arc::new(vb.finish()) as ArrayRef;

    let cols: Vec<(&str, ArrayRef)> = vec![
        ("geometry", geom),
        ("timestamp", ts),
        ("vertex_timestamps", vts),
    ];
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch =
        RecordBatch::try_new(schema.clone(), cols.iter().map(|(_, a)| a.clone()).collect()).unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    // File reader now ACCEPTS the µs Timestamp child and scales to ms.
    let file_feats = load_features(
        &path,
        "timestamp",
        None,
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .expect("file reader must accept a Timestamp(µs) vertex_timestamps child");
    assert_eq!(file_feats.len(), 1);
    assert_eq!(
        file_feats[0].vertex_timestamps.as_deref(),
        Some(&[t_ms[0] as u64, t_ms[1] as u64, t_ms[2] as u64][..]),
        "file reader must scale µs vertex timestamps to ms"
    );

    // DuckDB reads the same parquet (LIST(TIMESTAMP)) and must produce identical ms.
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "CREATE TABLE t AS SELECT * FROM read_parquet('{}')",
        path.display()
    ))
    .unwrap();
    let duck_feats = stt_build::duckdb_input::decode_query(
        &conn,
        "SELECT *, \"geometry\" AS __stt_wkb FROM t",
        "timestamp",
        None,
        "geometry",
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .unwrap();
    assert_eq!(duck_feats.len(), 1);
    assert_eq!(
        duck_feats[0].vertex_timestamps, file_feats[0].vertex_timestamps,
        "DuckDB and file reader must agree on µs→ms vertex timestamps"
    );
}

/// Read a single-row scalar-time parquet through both readers, returning
/// `(file_ms, duckdb_ms)` for the one feature's `start_time`.
fn read_both(path: &std::path::Path, time_format: TimeFormat) -> (u64, u64) {
    let file_feats = load_features(
        path,
        "start_time",
        None,
        time_format,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .expect("file reader must accept the scalar Timestamp/Int64 time column");
    assert_eq!(file_feats.len(), 1);

    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "CREATE TABLE t AS SELECT * FROM read_parquet('{}')",
        path.display()
    ))
    .unwrap();
    let duck_feats = stt_build::duckdb_input::decode_query(
        &conn,
        "SELECT *, \"geometry\" AS __stt_wkb FROM t",
        "start_time",
        None,
        "geometry",
        time_format,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .unwrap();
    assert_eq!(duck_feats.len(), 1);
    (file_feats[0].timestamp, duck_feats[0].timestamp)
}

#[test]
fn scalar_nanosecond_and_second_timestamps_parity_file_vs_duckdb() {
    let dir = tempfile::tempdir().unwrap();
    let base_ms: i64 = 1_700_000_000_000; // an ms-aligned instant

    // A pandas `datetime64[ns]` GeoParquet: a Timestamp(Nanosecond) scalar time
    // column. Before the fix the file reader hard-failed this ("Unsupported
    // timestamp column type") while DuckDB rescaled it — the asymmetric bug this
    // pins. `base_ms` is ms-aligned so the result is identical even if DuckDB
    // reads the column at microsecond precision.
    let ns_path = write_scalar_time_parquet(
        dir.path(),
        "ns",
        Arc::new(TimestampNanosecondArray::from(vec![base_ms * 1_000_000])) as ArrayRef,
    );
    let (file_ns, duck_ns) = read_both(&ns_path, TimeFormat::UnixMs);
    assert_eq!(file_ns, base_ms as u64, "file reader scales ns → ms");
    assert_eq!(file_ns, duck_ns, "file and DuckDB must agree on ns → ms");

    // Seconds travel as an integer column + `--time-format unix-sec` (parquet
    // has no second-precision TIMESTAMP logical type); both readers must ×1000
    // through the shared Second scaling and agree.
    let sec_path = write_scalar_time_parquet(
        dir.path(),
        "sec",
        Arc::new(Int64Array::from(vec![base_ms / 1000])) as ArrayRef,
    );
    let (file_sec, duck_sec) = read_both(&sec_path, TimeFormat::UnixSec);
    assert_eq!(file_sec, base_ms as u64, "file reader scales unix-sec → ms");
    assert_eq!(file_sec, duck_sec, "file and DuckDB must agree on unix-sec → ms");
}

#[test]
fn scalar_second_precision_timestamp_accepted_by_file_reader() {
    // The Arrow `Timestamp(Second)` scalar branch the fix ADDED: this column
    // type previously hit "Unsupported timestamp column type"; it now scales
    // ×1000 like every other precision.
    let dir = tempfile::tempdir().unwrap();
    let secs: i64 = 1_700_000_000;
    let path = write_scalar_time_parquet(
        dir.path(),
        "ts_sec",
        Arc::new(TimestampSecondArray::from(vec![secs])) as ArrayRef,
    );
    let feats = load_features(
        &path,
        "start_time",
        None,
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .expect("file reader must accept a Timestamp(Second) time column");
    assert_eq!(feats.len(), 1);
    assert_eq!(feats[0].timestamp, (secs * 1000) as u64);
}
