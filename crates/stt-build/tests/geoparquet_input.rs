//! Input-side contract tests for the GeoParquet reader:
//! `geo` footer metadata (CRS gate, primary_column, encoding gate),
//! pre-1970 timestamp rejection, invalid-geometry skip/strict handling,
//! and the LargeBinary WKB arm.

use arrow::array::{ArrayRef, BinaryArray, Int64Array, LargeBinaryArray, TimestampMillisecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use std::sync::Arc;
use stt_build::input::{load_features, InputStrictness, TimeFormat};

/// Little-endian ISO WKB for a 2D point.
fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = vec![0x01, 0x01, 0x00, 0x00, 0x00];
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

/// Write a single-batch Parquet file, optionally with a GeoParquet `geo`
/// footer key-value entry.
fn write_parquet(
    path: &std::path::Path,
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    geo: Option<String>,
) {
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let mut props = WriterProperties::builder();
    if let Some(geo_json) = geo {
        props = props
            .set_key_value_metadata(Some(vec![KeyValue::new("geo".to_string(), geo_json)]));
    }
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props.build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn binary_geom_schema(geom_name: &str) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(geom_name, DataType::Binary, true),
        Field::new("timestamp", DataType::Int64, true),
    ]))
}

fn load(
    path: &std::path::Path,
    time_strictness: InputStrictness,
    geometry_strictness: InputStrictness,
) -> anyhow::Result<Vec<stt_build::input::ParsedFeature>> {
    load_features(
        path,
        "timestamp",
        None,
        TimeFormat::UnixMs,
        time_strictness,
        geometry_strictness,
    )
}

/// Non-WGS84 CRS is a correctness hazard, not an ingestion blocker: by default
/// the build proceeds with a loud warning (so files with a mis-declared or
/// already-lon-lat CRS still build), and only `--strict-geometry` makes it a
/// hard error. This test pins both halves of that contract.
#[test]
fn non_wgs84_crs_warns_by_default_and_fails_under_strict() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mercator.parquet");
    let geo = serde_json::json!({
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": {
                    "type": "ProjectedCRS",
                    "name": "WGS 84 / Pseudo-Mercator",
                    "id": { "authority": "EPSG", "code": 3857 }
                }
            }
        }
    })
    .to_string();
    let columns = || {
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![Some(
                wkb_point(1.0, 2.0).as_slice(),
            )])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_700_000_000_000i64])) as ArrayRef,
        ]
    };
    write_parquet(&path, binary_geom_schema("geometry"), columns(), Some(geo));

    // Default (Warn): the build proceeds — the CRS is advisory only.
    let features = load(&path, InputStrictness::Warn, InputStrictness::Warn)
        .expect("non-WGS84 CRS must not block the build by default");
    assert_eq!(features.len(), 1, "row tiled as-is under the warn-only guard");

    // --strict-geometry (Strict): the same input hard-errors, naming both the
    // found CRS and the required one.
    let err = load(&path, InputStrictness::Warn, InputStrictness::Strict)
        .expect_err("EPSG:3857 input must hard-error under --strict-geometry");
    let msg = format!("{err:#}");
    assert!(msg.contains("EPSG:3857"), "must name the found CRS: {msg}");
    assert!(msg.contains("CRS84"), "must name the required CRS: {msg}");
}

#[test]
fn crs84_projjson_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crs84.parquet");
    let geo = serde_json::json!({
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": {
                    "type": "GeographicCRS",
                    "name": "WGS 84 (CRS84)",
                    "id": { "authority": "OGC", "code": "CRS84" }
                }
            }
        }
    })
    .to_string();
    write_parquet(
        &path,
        binary_geom_schema("geometry"),
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![Some(
                wkb_point(-122.4, 37.7).as_slice(),
            )])),
            Arc::new(Int64Array::from(vec![1_700_000_000_000i64])),
        ],
        Some(geo),
    );

    let features = load(&path, InputStrictness::Warn, InputStrictness::Warn).unwrap();
    assert_eq!(features.len(), 1);
    assert!((features[0].lon - -122.4).abs() < 1e-9);
    assert!((features[0].lat - 37.7).abs() < 1e-9);
}

#[test]
fn primary_column_overrides_name_heuristics() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("primary.parquet");
    // The decoy binary column comes first, so the heuristics would pick it
    // (no common geometry names in the schema). primary_column must win.
    let schema = Arc::new(Schema::new(vec![
        Field::new("blob", DataType::Binary, true),
        Field::new("geom_custom", DataType::Binary, true),
        Field::new("timestamp", DataType::Int64, true),
    ]));
    let geo = serde_json::json!({
        "version": "1.1.0",
        "primary_column": "geom_custom",
        "columns": {
            "geom_custom": { "encoding": "WKB", "geometry_types": ["Point"], "crs": null }
        }
    })
    .to_string();
    write_parquet(
        &path,
        schema,
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(b"not-wkb".as_slice()),
                Some(b"not-wkb".as_slice()),
            ])),
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(wkb_point(10.0, 20.0).as_slice()),
                Some(wkb_point(11.0, 21.0).as_slice()),
            ])),
            Arc::new(Int64Array::from(vec![
                1_700_000_000_000i64,
                1_700_000_001_000i64,
            ])),
        ],
        Some(geo),
    );

    let features = load(&path, InputStrictness::Warn, InputStrictness::Warn).unwrap();
    assert_eq!(features.len(), 2, "primary_column geometries must be used");
    assert!((features[0].lon - 10.0).abs() < 1e-9);
    assert!((features[1].lat - 21.0).abs() < 1e-9);
}

#[test]
fn native_geoarrow_encoding_gets_targeted_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("native.parquet");
    let geo = serde_json::json!({
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": { "encoding": "linestring", "geometry_types": ["LineString"], "crs": null }
        }
    })
    .to_string();
    write_parquet(
        &path,
        binary_geom_schema("geometry"),
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![Some(b"".as_slice())])),
            Arc::new(Int64Array::from(vec![1_700_000_000_000i64])),
        ],
        Some(geo),
    );

    let err = load(&path, InputStrictness::Warn, InputStrictness::Warn)
        .expect_err("native geoarrow linestring must hard-error");
    let msg = format!("{err:#}");
    assert!(msg.contains("linestring"), "must name the encoding: {msg}");
    assert!(msg.contains("WKB"), "must suggest WKB re-export: {msg}");
}

#[test]
fn negative_int64_timestamp_errors_even_in_warn_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("neg-int64.parquet");
    write_parquet(
        &path,
        binary_geom_schema("geometry"),
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(wkb_point(1.0, 2.0).as_slice()),
                Some(wkb_point(3.0, 4.0).as_slice()),
            ])),
            Arc::new(Int64Array::from(vec![1_700_000_000_000i64, -86_400_000])),
        ],
        None,
    );

    let err = load(&path, InputStrictness::Warn, InputStrictness::Warn)
        .expect_err("pre-1970 timestamp must hard-error in Warn mode");
    let msg = format!("{err:#}");
    assert!(msg.contains("pre-1970"), "{msg}");
    assert!(msg.contains("row 1"), "must carry row context: {msg}");
}

#[test]
fn negative_timestamp_ms_array_errors_even_in_warn_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("neg-tsms.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
    ]));
    write_parquet(
        &path,
        schema,
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![Some(
                wkb_point(1.0, 2.0).as_slice(),
            )])),
            Arc::new(TimestampMillisecondArray::from(vec![-1i64])),
        ],
        None,
    );

    let err = load(&path, InputStrictness::Warn, InputStrictness::Warn)
        .expect_err("pre-1970 Timestamp(ms) must hard-error in Warn mode");
    assert!(format!("{err:#}").contains("pre-1970"));
}

#[test]
fn invalid_geometry_is_skipped_in_warn_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("badgeom-warn.parquet");
    write_parquet(
        &path,
        binary_geom_schema("geometry"),
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(wkb_point(5.0, 6.0).as_slice()),
                Some(b"garbage-not-wkb".as_slice()),
                None,
            ])),
            Arc::new(Int64Array::from(vec![
                1_700_000_000_000i64,
                1_700_000_001_000i64,
                1_700_000_002_000i64,
            ])),
        ],
        None,
    );

    let features = load(&path, InputStrictness::Warn, InputStrictness::Warn).unwrap();
    // The unparseable + null geometry rows are skipped, NOT tiled at (0,0).
    assert_eq!(features.len(), 1);
    assert!((features[0].lon - 5.0).abs() < 1e-9);
    assert!(
        features.iter().all(|f| !(f.lon == 0.0 && f.lat == 0.0)),
        "no Null Island sentinels may survive parsing"
    );
}

#[test]
fn invalid_geometry_bails_in_strict_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("badgeom-strict.parquet");
    write_parquet(
        &path,
        binary_geom_schema("geometry"),
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(wkb_point(5.0, 6.0).as_slice()),
                Some(b"garbage-not-wkb".as_slice()),
            ])),
            Arc::new(Int64Array::from(vec![
                1_700_000_000_000i64,
                1_700_000_001_000i64,
            ])),
        ],
        None,
    );

    let err = load(&path, InputStrictness::Warn, InputStrictness::Strict)
        .expect_err("--strict-geometry must fail on unparseable geometry");
    let msg = format!("{err:#}");
    assert!(msg.contains("row 1"), "must carry row context: {msg}");
    assert!(msg.contains("geometry"), "{msg}");
}

#[test]
fn large_binary_wkb_geometry_builds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("largebin.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::LargeBinary, true),
        Field::new("timestamp", DataType::Int64, true),
    ]));
    write_parquet(
        &path,
        schema,
        vec![
            Arc::new(LargeBinaryArray::from_opt_vec(vec![
                Some(wkb_point(100.5, -45.25).as_slice()),
                Some(wkb_point(101.5, -46.25).as_slice()),
            ])),
            Arc::new(Int64Array::from(vec![
                1_700_000_000_000i64,
                1_700_000_001_000i64,
            ])),
        ],
        None,
    );

    let features = load(&path, InputStrictness::Warn, InputStrictness::Warn).unwrap();
    assert_eq!(features.len(), 2);
    assert!((features[0].lon - 100.5).abs() < 1e-9);
    assert!((features[0].lat - -45.25).abs() < 1e-9);
    assert!((features[1].lon - 101.5).abs() < 1e-9);
}

#[test]
fn unix_sec_format_scales_int64_to_ms() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unixsec.parquet");
    write_parquet(
        &path,
        binary_geom_schema("geometry"),
        vec![
            Arc::new(BinaryArray::from_opt_vec(vec![Some(
                wkb_point(1.0, 2.0).as_slice(),
            )])),
            Arc::new(Int64Array::from(vec![1_700_000_000i64])),
        ],
        None,
    );

    let features = load_features(
        &path,
        "timestamp",
        None,
        TimeFormat::UnixSec,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .unwrap();
    assert_eq!(features[0].timestamp, 1_700_000_000_000);
}
