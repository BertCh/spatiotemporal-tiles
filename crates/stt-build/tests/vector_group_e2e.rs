//! End-to-end contract for `--vector-group`: a GeoParquet carrying scalar
//! surfel columns (`qx,qy,qz,qw`) flows through the full pipeline — ingest →
//! columnar layer → encode with the build-global vector-group config → tile —
//! and the result is a single interleaved `FixedSizeList<Float32,4>` column the
//! TS decoder hands to the GPU zero-copy (with the scalar inputs removed).
//!
//! The encode-side grouping is unit-tested in `arrow_tile.rs`; this proves the
//! CLI path (parquet ingestion producing the scalar `Numeric` columns that the
//! grouping then fuses) composes correctly.

use arrow::array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::sync::Arc;
use stt_build::columnar::build_layers_from_features;
use stt_build::input::{load_features, InputStrictness, TimeFormat};
use stt_core::arrow_tile::{
    decode_layer, encode_layer, encode_layer_quantized, set_point_elevation_column,
    set_vector_groups, QuantAffine, VectorElem, VectorGroup, STT_QUANT_META_KEY,
};

fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = vec![0x01, 0x01, 0x00, 0x00, 0x00];
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

#[test]
fn vector_group_fuses_scalar_surfel_columns_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("surfels.parquet");

    let n = 3usize;
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new("timestamp", DataType::Int64, true),
        Field::new("qx", DataType::Float64, true),
        Field::new("qy", DataType::Float64, true),
        Field::new("qz", DataType::Float64, true),
        Field::new("qw", DataType::Float64, true),
    ]));
    let geom_bytes: Vec<Vec<u8>> = (0..n)
        .map(|i| wkb_point(-122.4 + i as f64 * 0.001, 37.7))
        .collect();
    let geom_refs: Vec<&[u8]> = geom_bytes.iter().map(|v| v.as_slice()).collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BinaryArray::from(geom_refs)),
        Arc::new(Int64Array::from(vec![1000i64, 2000, 3000])),
        Arc::new(Float64Array::from(vec![0.0, 0.5, 0.0])),
        Arc::new(Float64Array::from(vec![0.0, 0.5, 0.0])),
        Arc::new(Float64Array::from(vec![0.0, 0.5, 0.0])),
        Arc::new(Float64Array::from(vec![1.0, 0.5, 1.0])),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Ingest → features → one point ColumnarLayer (scalar qx..qw Numeric columns).
    let features = load_features(
        &path,
        "timestamp",
        None,
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .unwrap();
    let refs: Vec<&_> = features.iter().collect();
    let layers = build_layers_from_features(&refs, "surfels").unwrap();
    assert_eq!(layers.len(), 1, "expected one point layer");

    // Configure the build-global grouping (what the CLI's --vector-group does),
    // encode, then reset so the global doesn't leak to other tests.
    set_vector_groups(vec![VectorGroup {
        name: "surfel_quat".to_string(),
        components: vec!["qx".into(), "qy".into(), "qz".into(), "qw".into()],
        elem: VectorElem::F32,
    }]);
    let ipc = encode_layer(&layers[0]).unwrap();
    set_vector_groups(Vec::new());

    let batch = decode_layer(&ipc).unwrap();
    // Scalars fused away; one interleaved FixedSizeList<Float32,4> remains.
    assert!(
        batch.column_by_name("qx").is_none(),
        "scalar qx must be consumed"
    );
    let quat = batch
        .column_by_name("surfel_quat")
        .expect("grouped surfel_quat column present")
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(quat.len(), n);
    assert_eq!(quat.value_length(), 4);
    let child = quat
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    // Row-major interleave: feature i occupies [i*4, i*4+4).
    assert_eq!(child.value(3), 1.0); // feature 0, qw
    assert_eq!(child.value(4), 0.5); // feature 1, qx
}

#[test]
fn point_elevation_folds_into_3d_geometry_end_to_end() {
    // A GeoParquet of 2D WKB points + a `z` column flows through ingest → columnar
    // → encode-with-`--point-elevation-column z`, and the tile ships true 3D
    // points (FixedSizeList<i32,3>) with a z affine — the renderer binds 3D
    // positions zero-copy (no pad). `z` is removed from the scalar properties.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cloud.parquet");
    let n = 3usize;
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new("timestamp", DataType::Int64, true),
        Field::new("z", DataType::Float64, true),
    ]));
    let geom_bytes: Vec<Vec<u8>> = (0..n)
        .map(|i| wkb_point(-122.4 + i as f64 * 0.001, 37.7))
        .collect();
    let geom_refs: Vec<&[u8]> = geom_bytes.iter().map(|v| v.as_slice()).collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BinaryArray::from(geom_refs)),
        Arc::new(Int64Array::from(vec![1000i64, 2000, 3000])),
        Arc::new(Float64Array::from(vec![3.5, 9.0, 0.5])),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let features = load_features(
        &path,
        "timestamp",
        None,
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .unwrap();
    let refs: Vec<&_> = features.iter().collect();
    let layers = build_layers_from_features(&refs, "cloud").unwrap();

    set_point_elevation_column("z");
    let ipc = encode_layer_quantized(&layers[0], Some(0.05)).unwrap();
    set_point_elevation_column("");
    let batch = decode_layer(&ipc).unwrap();

    assert!(
        batch.column_by_name("z").is_none(),
        "z folded into geometry"
    );
    let geom = batch
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(geom.value_length(), 3, "3D point geometry");
    let field = batch.schema().field_with_name("geometry").unwrap().clone();
    let affine = QuantAffine::from_json(field.metadata().get(STT_QUANT_META_KEY).unwrap()).unwrap();
    assert_eq!(affine.sz, Some(0.05));
}
