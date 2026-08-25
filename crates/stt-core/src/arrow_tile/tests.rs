//! Unit tests for the tile payload format.
//!
//! Kept as one module (the pre-split `arrow_tile::tests`) because the cases
//! deliberately cross the encode/decode boundary.

use super::*;
// The pre-split `mod tests` inherited these from the enclosing file's import
// block via `use super::*`; the submodules own their own imports now, so the
// test module carries the set it actually uses.
use arrow::array::{
    Array, DictionaryArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    ListArray, RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, UInt16Type};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

fn sample_point_layer() -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "points".to_string(),
        feature_ids: vec![1, 2, 3],
        start_times: vec![1000, 2000, 3000],
        end_times: vec![1500, 2500, 3500],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8], [-122.6, 37.9]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "speed".to_string(),
                PropertyColumn::Numeric(vec![Some(10.0), None, Some(30.0)]),
            ),
            (
                "kind".to_string(),
                PropertyColumn::Categorical(vec![
                    Some("car".to_string()),
                    Some("bus".to_string()),
                    None,
                ]),
            ),
        ],
    }
}

/// Two DIFFERENT [`EncoderConfig`]s encode the SAME layer to DIFFERENT tiles
/// in ONE process, and the output is driven purely by the passed config — not
/// by the process-wide globals. This is the property that unblocks a dynamic
/// server hosting several datasets/configs concurrently: if `encode_tile_with`
/// read the (unset) globals instead of the config, the quantized and plain
/// encodes would be identical and this would fail.
#[test]
fn encode_tile_with_is_config_driven_not_global() {
    let layer = sample_point_layer();
    let layers = std::slice::from_ref(&layer);

    let plain_cfg = EncoderConfig::default();
    let quant_cfg = EncoderConfig {
        quantize_coords_m: Some(1.0),
        ..EncoderConfig::default()
    };
    let attr_cfg = EncoderConfig {
        quantize_attrs_auto: true,
        ..EncoderConfig::default()
    };

    let plain = encode_tile_with(layers, &plain_cfg).unwrap();
    let quant = encode_tile_with(layers, &quant_cfg).unwrap();
    let attr = encode_tile_with(layers, &attr_cfg).unwrap();

    // Each explicit config yields a distinct tile — the config, not shared
    // state, decides the encoding. (If `encode_tile_with` read the unset
    // globals instead of the config, all three would be identical.) These
    // differences are config-driven at the WIRE COLUMN level — coord
    // quantization changes the geometry column (i32 grid vs Float64) and
    // attribute quantization changes the `speed` column (u16 vs Float64) — so
    // the inequality is not attributable to the encoder's (separately
    // tracked) non-deterministic Arrow-metadata ordering, which we therefore
    // deliberately do NOT byte-assert here.
    assert_ne!(plain, quant, "coord quantization must change the tile");
    assert_ne!(plain, attr, "attribute quantization must change the tile");
    assert_ne!(quant, attr, "the two quantizations differ from each other");

    // All three still decode to the SAME feature set — encoding differs, data
    // does not.
    for tile in [&plain, &quant, &attr] {
        let rows: usize = decode_tile(tile)
            .unwrap()
            .iter()
            .map(|l| l.batch.num_rows())
            .sum();
        assert_eq!(rows, 3);
    }
}

fn sample_line_layer() -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "tracks".to_string(),
        feature_ids: vec![10, 11],
        start_times: vec![0, 100],
        end_times: vec![50, 200],
        geometry: GeometryColumn::LineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            vec![[5.0, 5.0], [6.0, 6.0]],
        ]),
        vertex_times: Some(vec![vec![0, 25, 50], vec![100, 200]]),
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

fn sample_polygon_layer() -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "zones".to_string(),
        feature_ids: vec![42],
        start_times: vec![0],
        end_times: vec![1000],
        geometry: GeometryColumn::Polygon(vec![vec![
            // exterior ring
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
            // hole
            vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]],
        ]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

#[test]
fn small_categorical_columns_use_plain_utf8() {
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "cars".into(),
        feature_ids: vec![1, 2, 3, 4, 5],
        start_times: vec![0; 5],
        end_times: vec![1; 5],
        geometry: GeometryColumn::Point(vec![[0.0, 0.0]; 5]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "kind".into(),
            PropertyColumn::Categorical(vec![
                Some("car".into()),
                Some("bus".into()),
                Some("car".into()),
                None,
                Some("car".into()),
            ]),
        )],
    };
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();
    let field = batch.schema().field_with_name("kind").unwrap().clone();
    assert_eq!(field.data_type(), &DataType::Utf8);
    let col = batch
        .column_by_name("kind")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(col.value(0), "car");
    assert_eq!(col.value(1), "bus");
    assert_eq!(col.value(2), "car");
    assert!(col.is_null(3));
    assert_eq!(col.value(4), "car");
}

#[test]
fn repeated_categorical_columns_use_dictionary_encoding() {
    let n = 1_000;
    let kinds = (0..n)
        .map(|i| Some(if i % 2 == 0 { "car" } else { "bus" }.to_string()))
        .collect();
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "vehicles".into(),
        feature_ids: (0..n as u64).collect(),
        start_times: vec![0; n],
        end_times: vec![1; n],
        geometry: GeometryColumn::Point(vec![[0.0, 0.0]; n]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![("kind".into(), PropertyColumn::Categorical(kinds))],
    };
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();
    let schema = batch.schema();
    let field = schema.field_with_name("kind").unwrap();
    match field.data_type() {
        DataType::Dictionary(key, value) => {
            assert_eq!(key.as_ref(), &DataType::UInt16);
            assert_eq!(value.as_ref(), &DataType::Utf8);
        }
        other => panic!("expected Dictionary<UInt16, Utf8>, got {other:?}"),
    }
}

#[test]
fn categorical_overflow_falls_back_to_exact_utf8() {
    // A column whose distinct-value count exceeds the UInt16 key space remains
    // lossless by using plain Utf8 rather than failing or dropping values.
    let n = u16::MAX as usize + 1; // 65_536 distinct strings
    let kinds: Vec<Option<String>> = (0..n).map(|i| Some(format!("c{i}"))).collect();
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "huge".into(),
        feature_ids: (0..n as u64).collect(),
        start_times: vec![0; n],
        end_times: vec![1; n],
        geometry: GeometryColumn::Point(vec![[0.0, 0.0]; n]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![("kind".into(), PropertyColumn::Categorical(kinds))],
    };
    let ipc = encode_layer(&layer).expect("high-cardinality strings remain encodable");
    let batch = decode_layer(&ipc).unwrap();
    let schema = batch.schema();
    let field = schema.field_with_name("kind").unwrap();
    assert_eq!(field.data_type(), &DataType::Utf8);
    let col = batch
        .column_by_name("kind")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(col.value(0), "c0");
    assert_eq!(col.value(n - 1), format!("c{}", n - 1));
}

#[test]
fn geometry_field_advertises_crs_metadata() {
    // Every geometry field carries the GeoArrow extension *name* and the
    // CRS in extension *metadata*, so external GeoArrow readers see WGS84
    // lon/lat (OGC:CRS84) rather than an unknown CRS.
    for layer in [
        sample_point_layer(),
        sample_line_layer(),
        sample_polygon_layer(),
    ] {
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        let field = batch.schema().field_with_name("geometry").unwrap().clone();
        let meta = field.metadata();
        assert_eq!(
            meta.get(GEOARROW_EXT_KEY).map(String::as_str),
            Some(layer.geometry.geoarrow_name())
        );
        let crs = meta
            .get(GEOARROW_EXT_META_KEY)
            .expect("geometry field must carry ARROW:extension:metadata");
        assert!(crs.contains("OGC:CRS84"), "crs metadata was: {crs}");
        assert!(crs.contains("crs_type"), "crs metadata was: {crs}");
    }
}

#[test]
fn point_layer_roundtrips() {
    let layer = sample_point_layer();
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    assert_eq!(batch.num_rows(), 3);
    // id / start / end / geometry / speed / kind
    assert_eq!(batch.num_columns(), 6);

    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2, 3]);

    let geom = batch
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(geom.len(), 3);
    assert_eq!(geom.value_length(), 2);

    // Geometry field carries the GeoArrow extension name.
    let geom_field = batch.schema().field_with_name("geometry").unwrap().clone();
    assert_eq!(
        geom_field
            .metadata()
            .get(GEOARROW_EXT_KEY)
            .map(String::as_str),
        Some("geoarrow.point")
    );

    // Nullable numeric property preserves the null.
    let speed = batch
        .column_by_name("speed")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(speed.is_null(1));
    assert_eq!(speed.value(0), 10.0);
}

#[test]
fn vector_property_roundtrips_as_fixed_size_list() {
    // A Vector property encodes as FixedSizeList<leaf, width>: the f32 quat
    // as <Float32,4>, the u8 colour as <UInt8,4>, with the child buffer the
    // flattened row-major run the TS decoder hands to the GPU zero-copy.
    use arrow::array::{Float32Array, UInt8Array};
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "surfels".to_string(),
        feature_ids: vec![1, 2],
        start_times: vec![0, 10],
        end_times: vec![0, 10],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "surfel_quat".to_string(),
                PropertyColumn::Vector {
                    width: 4,
                    elem: VectorElem::F32,
                    values: vec![0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
                },
            ),
            (
                "surfel_rgba".to_string(),
                PropertyColumn::Vector {
                    width: 4,
                    elem: VectorElem::U8,
                    values: vec![255.0, 0.0, 0.0, 128.0, 0.0, 255.0, 0.0, 255.0],
                },
            ),
        ],
    };
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    let quat = batch
        .column_by_name("surfel_quat")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(quat.len(), 2);
    assert_eq!(quat.value_length(), 4);
    let qchild = quat
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(qchild.values(), &[0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]);

    let rgba = batch
        .column_by_name("surfel_rgba")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(rgba.value_length(), 4);
    let cchild = rgba.values().as_any().downcast_ref::<UInt8Array>().unwrap();
    assert_eq!(cchild.values(), &[255, 0, 0, 128, 0, 255, 0, 255]);
}

#[test]
fn vector_groups_fuse_scalar_columns_at_encode() {
    // `--vector-group` fuses named scalar columns into one interleaved
    // FixedSizeList and drops the scalars; ungrouped columns are untouched.
    use arrow::array::Float32Array;
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "surfels".to_string(),
        feature_ids: vec![1, 2],
        start_times: vec![0, 10],
        end_times: vec![0, 10],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "qx".into(),
                PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)]),
            ),
            (
                "qy".into(),
                PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)]),
            ),
            (
                "qz".into(),
                PropertyColumn::Numeric(vec![Some(0.0), Some(0.5)]),
            ),
            (
                "qw".into(),
                PropertyColumn::Numeric(vec![Some(1.0), Some(0.5)]),
            ),
            (
                "z".into(),
                PropertyColumn::Numeric(vec![Some(3.0), Some(4.0)]),
            ),
        ],
    };
    // Explicit config (not the process-global setter) so this test can't
    // leak a non-default vector-group into a concurrently-running test that
    // reads the encoder globals via bare `encode_layer`.
    let cfg = EncoderConfig {
        vector_groups: vec![VectorGroup {
            name: "surfel_quat".to_string(),
            components: vec!["qx".into(), "qy".into(), "qz".into(), "qw".into()],
            elem: VectorElem::F32,
        }],
        ..EncoderConfig::default()
    };
    let ipc = encode_layer_cfg(&layer, &cfg).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    // Scalars fused away; the grouped vector + the ungrouped `z` remain.
    assert!(batch.column_by_name("qx").is_none());
    assert!(batch.column_by_name("z").is_some());
    let quat = batch
        .column_by_name("surfel_quat")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(quat.value_length(), 4);
    let qchild = quat
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(qchild.values(), &[0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]);
}

#[test]
fn point_elevation_folds_into_3d_geometry_unquantized() {
    use arrow::array::Float64Array;
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "cloud".into(),
        feature_ids: vec![1, 2],
        start_times: vec![0, 0],
        end_times: vec![0, 0],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "z".into(),
                PropertyColumn::Numeric(vec![Some(3.5), Some(9.0)]),
            ),
            (
                "speed".into(),
                PropertyColumn::Numeric(vec![Some(1.0), Some(2.0)]),
            ),
        ],
    };
    let cfg = EncoderConfig {
        point_elevation_column: "z".to_string(),
        ..EncoderConfig::default()
    };
    let ipc = encode_layer_cfg(&layer, &cfg).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    // Geometry is a 3-wide list with z folded in; `z` is not a property column.
    let geom = batch
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(geom.value_length(), 3);
    let coords = geom
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(coords.value(2), 3.5); // feature 0 z
    assert_eq!(coords.value(5), 9.0); // feature 1 z
    assert!(
        batch.column_by_name("z").is_none(),
        "z folded into geometry"
    );
    assert!(
        batch.column_by_name("speed").is_some(),
        "other props untouched"
    );
}

#[test]
fn point_elevation_3d_geometry_quantizes_with_z_affine() {
    use arrow::array::Int32Array;
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "cloud".into(),
        feature_ids: vec![1],
        start_times: vec![0],
        end_times: vec![0],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![("z".into(), PropertyColumn::Numeric(vec![Some(5.0)]))],
    };
    let cfg = EncoderConfig {
        quantize_coords_m: Some(0.05),
        point_elevation_column: "z".to_string(),
        ..EncoderConfig::default()
    };
    let ipc = encode_layer_cfg(&layer, &cfg).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    let field = batch.schema().field_with_name("geometry").unwrap().clone();
    let affine = QuantAffine::from_json(field.metadata().get(STT_QUANT_META_KEY).unwrap()).unwrap();
    assert_eq!(affine.z0, Some(0.0));
    assert_eq!(affine.sz, Some(0.05));
    let geom = batch
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(geom.value_length(), 3);
    let coords = geom.values().as_any().downcast_ref::<Int32Array>().unwrap();
    // z = 5.0 / 0.05 = 100; reconstructs to z0 + 100*sz = 5.0.
    assert_eq!(coords.value(2), 100);
    assert_eq!(
        affine.z0.unwrap() + coords.value(2) as f64 * affine.sz.unwrap(),
        5.0
    );
}

#[test]
fn quantized_point_layer_roundtrips_within_precision() {
    let layer = sample_point_layer();
    let ipc = encode_layer_quantized(&layer, Some(1.0)).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    // Geometry leaf is i32 grid indices, and the affine rides in metadata.
    let geom_field = batch.schema().field_with_name("geometry").unwrap().clone();
    let affine = QuantAffine::from_json(
        geom_field
            .metadata()
            .get(STT_QUANT_META_KEY)
            .expect("quantized tile must carry the affine"),
    )
    .unwrap();

    let geom = batch
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(geom.value_type(), DataType::Int32);
    let coords = geom.values().as_any().downcast_ref::<Int32Array>().unwrap();

    let original = [[-122.4, 37.7], [-122.5, 37.8], [-122.6, 37.9]];
    for (i, [lon, lat]) in original.iter().enumerate() {
        let rlon = affine.lon(coords.value(i * 2));
        let rlat = affine.lat(coords.value(i * 2 + 1));
        // Worst-case reconstruction error ≤ ~half a quantum (~0.5 m).
        let dlon_m = (rlon - lon).abs() * M_PER_DEG_LAT * lat.to_radians().cos();
        let dlat_m = (rlat - lat).abs() * M_PER_DEG_LAT;
        assert!(dlon_m < 1.0, "lon err {dlon_m} m at point {i}");
        assert!(dlat_m < 1.0, "lat err {dlat_m} m at point {i}");
    }
}

#[test]
fn quantized_numeric_attr_roundtrips_within_precision_and_is_opt_in() {
    // A LiDAR-style `z` elevation column: high-entropy Float64 by default,
    // but fixed-point UInt16 when the build opts the column in. The reader
    // reconstructs `value = o + q*s`, lossy to <= s/2.
    let zvals: Vec<Option<f64>> = vec![Some(1.07), Some(-2.4), Some(15.9), None, Some(40.02)];
    let make = || ColumnarLayer {
        polygon_parts: None,
        name: "lidar".into(),
        feature_ids: vec![1, 2, 3, 4, 5],
        start_times: vec![0; 5],
        end_times: vec![1; 5],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7]; 5]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![("z".into(), PropertyColumn::Numeric(zvals.clone()))],
    };

    // Default (no attr-quant configured): `z` stays Float64, byte-identical
    // to the historical encoder. Explicit config (not the global setter) so
    // this test stays hermetic under the parallel test runner.
    let plain =
        decode_layer(&encode_layer_cfg(&make(), &EncoderConfig::default()).unwrap()).unwrap();
    let zf = plain.schema().field_with_name("z").unwrap().clone();
    assert_eq!(zf.data_type(), &DataType::Float64);
    assert!(zf.metadata().get(STT_QUANT_ATTR_META_KEY).is_none());

    // Opt the `z` column in at 0.05-unit precision.
    let q = encode_layer_cfg(
        &make(),
        &EncoderConfig {
            quantize_attrs: HashMap::from([("z".to_string(), 0.05f64)]),
            ..EncoderConfig::default()
        },
    )
    .unwrap();

    let batch = decode_layer(&q).unwrap();
    let field = batch.schema().field_with_name("z").unwrap().clone();
    // Range (-2.4..40.02)/0.05 ~= 848 fits 16 bits → UInt16 leaf.
    assert_eq!(field.data_type(), &DataType::UInt16);
    let affine = AttrQuant::from_json(
        field
            .metadata()
            .get(STT_QUANT_ATTR_META_KEY)
            .expect("quantized attr must carry the affine"),
    )
    .unwrap();

    let col = batch
        .column_by_name("z")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .unwrap();
    for (i, want) in zvals.iter().enumerate() {
        match want {
            Some(v) => {
                assert!(!col.is_null(i), "row {i} should be present");
                let got = affine.value(col.value(i) as i64);
                assert!((got - v).abs() <= 0.05 / 2.0 + 1e-9, "z[{i}] {got} vs {v}");
            }
            None => assert!(col.is_null(i), "row {i} should be null"),
        }
    }
}

#[test]
fn auto_numeric_quantization_is_range_adaptive_and_opt_in() {
    // With auto-quant enabled, a raw Float64 property is quantized to a
    // UInt16 sized from its own [min,max] span (no precision configured),
    // and reconstructs to <= span/65535. Default-off keeps it Float64.
    let depth: Vec<Option<f64>> = vec![Some(0.0), Some(10.0), Some(123.4), Some(700.0)];
    let make = || ColumnarLayer {
        polygon_parts: None,
        name: "q".into(),
        feature_ids: vec![1, 2, 3, 4],
        start_times: vec![0; 4],
        end_times: vec![1; 4],
        geometry: GeometryColumn::Point(vec![[0.0, 0.0]; 4]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![("depth".into(), PropertyColumn::Numeric(depth.clone()))],
    };

    // Default: auto off → Float64. Explicit config (not the global setter)
    // so this test can't flip `quantize_attrs_auto` under a concurrently
    // running test that reads the encoder globals via bare `encode_layer`.
    let plain =
        decode_layer(&encode_layer_cfg(&make(), &EncoderConfig::default()).unwrap()).unwrap();
    assert_eq!(
        plain.schema().field_with_name("depth").unwrap().data_type(),
        &DataType::Float64
    );

    // Auto on → range-adaptive UInt16 + affine.
    let batch = decode_layer(
        &encode_layer_cfg(
            &make(),
            &EncoderConfig {
                quantize_attrs_auto: true,
                ..EncoderConfig::default()
            },
        )
        .unwrap(),
    )
    .unwrap();

    let field = batch.schema().field_with_name("depth").unwrap().clone();
    assert_eq!(field.data_type(), &DataType::UInt16);
    let aff = AttrQuant::from_json(field.metadata().get(STT_QUANT_ATTR_META_KEY).unwrap()).unwrap();
    let col = batch
        .column_by_name("depth")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .unwrap();
    let tol = (700.0 - 0.0) / u16::MAX as f64 / 2.0 + 1e-9;
    for (i, want) in depth.iter().enumerate() {
        let got = aff.value(col.value(i) as i64);
        assert!(
            (got - want.unwrap()).abs() <= tol,
            "depth[{i}] {got} vs {want:?}"
        );
    }
    // Min and max land on the index endpoints (full 16-bit span used).
    assert_eq!(col.value(0), 0);
    assert_eq!(col.value(3), u16::MAX);
}

// ------------------------------------------------------------------
// Auto numeric quantization: exactness, the ONE refusal, type stability
// across tiles, and the two corruption classes that matter in practice
// (integer columns; hash identifier columns).
// ------------------------------------------------------------------

/// Run `build_quantized_numeric_auto` and reconstruct every cell through the
/// affine it shipped — i.e. do exactly what both reference readers do. Returns
/// `(leaf type, affine, reconstructed values)`, or `None` when the encoder
/// refused to quantize (the caller keeps the column `Float64`).
fn auto_quant_roundtrip(values: &[Option<f64>]) -> Option<(DataType, AttrQuant, Vec<Option<f64>>)> {
    let (array, json) = build_quantized_numeric_auto(values)?;
    let affine = AttrQuant::from_json(&json).expect("auto quantization must ship a parseable qa");
    let data_type = array.data_type().clone();
    let back: Vec<Option<f64>> = match &data_type {
        DataType::UInt16 => {
            let c = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            (0..c.len())
                .map(|i| (!c.is_null(i)).then(|| affine.value(c.value(i) as i64)))
                .collect()
        }
        DataType::Int32 => {
            let c = array.as_any().downcast_ref::<Int32Array>().unwrap();
            (0..c.len())
                .map(|i| (!c.is_null(i)).then(|| affine.value(c.value(i) as i64)))
                .collect()
        }
        other => panic!("quantized property leaf must be UInt16/Int32, got {other:?}"),
    };
    Some((data_type, affine, back))
}

#[test]
fn auto_quant_integer_column_is_exact_at_uint16_width() {
    // Summary-tier `count` / `bucket_i` / `sum_*` (measured on
    // examples/showcase/public/data/nyc-taxi-od-summary) are integers carried
    // as f64. Mapping [min,max] onto [0,65535] gives a FRACTIONAL step, so a
    // count of 7 comes back as ~7.03. An integer column must quantize at
    // step 1.0 and round-trip bit-for-bit.
    let count: Vec<Option<f64>> = vec![
        Some(1.0),
        Some(7.0),
        None,
        Some(12_345.0),
        Some(2.0),
        Some(0.0),
    ];
    let (dt, affine, back) = auto_quant_roundtrip(&count).expect("integer column must quantize");
    assert_eq!(dt, DataType::UInt16, "span 12345 fits the 16-bit leaf");
    assert_eq!(affine.s, 1.0, "an integer column quantizes at step 1");
    assert_eq!(affine.o, 0.0, "offset is the column minimum");
    assert_eq!(
        back, count,
        "integer round-trip must be EXACT, not within s/2"
    );

    // The pre-fix step this replaces, spelled out: 7 really did decode wrong.
    let lossy = AttrQuant {
        o: 0.0,
        s: 12_345.0 / u16::MAX as f64,
    };
    let q = ((7.0 - lossy.o) / lossy.s).round() as i64;
    assert_ne!(lossy.value(q), 7.0, "the old range-adaptive step was lossy");

    // Full 16-bit span still fits exactly (span == u16::MAX is the boundary).
    let wide: Vec<Option<f64>> = vec![Some(-5.0), Some(65_530.0), Some(0.0)];
    let (dt, affine, back) = auto_quant_roundtrip(&wide).expect("span 65535 fits UInt16");
    assert_eq!(dt, DataType::UInt16);
    assert_eq!((affine.o, affine.s), (-5.0, 1.0));
    assert_eq!(back, wide);
}

#[test]
fn auto_quant_integer_column_widens_to_int32_and_stays_exact() {
    // One quantum past the 16-bit span the leaf widens rather than going lossy.
    let just_over: Vec<Option<f64>> = vec![Some(10.0), Some(10.0 + 65_536.0), Some(11.0)];
    let (dt, affine, back) = auto_quant_roundtrip(&just_over).expect("must still quantize");
    assert_eq!(dt, DataType::Int32, "span 65536 no longer fits UInt16");
    assert_eq!((affine.o, affine.s), (10.0, 1.0));
    assert_eq!(back, just_over);

    // Near the Int32 leaf's ceiling — still exact, still smaller than Float64.
    let huge: Vec<Option<f64>> = vec![Some(0.0), Some(2_000_000_000.0), Some(7.0), None];
    let (dt, _, back) = auto_quant_roundtrip(&huge).expect("span 2e9 fits Int32");
    assert_eq!(dt, DataType::Int32);
    assert_eq!(back, huge);

    // Span wider than the Int32 leaf while every MAGNITUDE stays under the
    // refusal threshold (only reachable by straddling zero): step 1 no longer
    // indexes the span, so the column drops to the historical LOSSY
    // range-adaptive UInt16 rather than refusing. Refusing on span would have
    // made the Arrow type depend on which rows this tile caught (Float64 here,
    // UInt16/Int32 in a narrower tile), which stt-validate rates as structural
    // schema drift and hard-fails on. Magnitude, unlike span, does not vary that
    // way — which is why THAT is the refusal.
    let too_wide: Vec<Option<f64>> = vec![Some(-2_000_000_000.0), Some(7.0), Some(2_000_000_000.0)];
    let (dt, affine, back) = auto_quant_roundtrip(&too_wide)
        .expect("a span past Int32 range-adapts, it does not refuse");
    assert_eq!(
        dt,
        DataType::UInt16,
        "type stability outranks exactness here"
    );
    assert_eq!(affine.o, -2_000_000_000.0);
    assert!(
        (affine.s - 4_000_000_000.0 / u16::MAX as f64).abs() < 1e-6,
        "step is the span over the 16-bit index space, got {}",
        affine.s
    );
    for (got, want) in back.iter().zip(&too_wide) {
        assert!(
            (got.unwrap() - want.unwrap()).abs() <= affine.s / 2.0 + 1e-6,
            "range-adaptive error must stay within half a step"
        );
    }

    // Past the magnitude threshold the column is left alone entirely: at this
    // scale the step-1 path cannot index into Int32 and the range-adaptive step
    // would be ~46k, so neither encoding is defensible.
    let past_threshold: Vec<Option<f64>> = vec![Some(0.0), Some(7.0), Some(3_000_000_000.0)];
    assert!(
        build_quantized_numeric_auto(&past_threshold).is_none(),
        "a column reaching i32::MAX stays Float64"
    );
}

#[test]
fn auto_quant_leaves_hash_identifier_columns_float64() {
    // `trip_id` on examples/showcase/public/data/nyc-taxi-points holds 64-bit
    // hashes. Quantizing them lands on o=2.35e18, s=2.3e14, so every
    // reconstructed value is wrong by up to ±1.15e14 AND the same trip decodes
    // differently in different tiles. Refusing keeps them Float64, which is
    // exact for integers up to 2^53 and for these ids is the only honest
    // encoding this format has.
    let ids: Vec<Option<f64>> = vec![
        Some(2_350_000_000_000_000_000.0),
        Some(9_100_000_000_000_000_000.0),
        Some(4_004_004_004_004_004_000.0),
        None,
    ];
    assert!(
        build_quantized_numeric_auto(&ids).is_none(),
        "hash ids must NOT be quantized"
    );

    // The type-stability trap: a tile holding ONE distinct id has span 0, so a
    // span-only rule would quantize it (exactly!) while every other tile of the
    // same dataset stayed Float64 — drifting the column's Arrow type and the
    // PROPS schema template. The magnitude test keys off the value domain, not
    // the tile's sample, so this tile refuses too.
    let single = vec![Some(2_350_000_000_000_000_000.0); 4];
    assert!(
        build_quantized_numeric_auto(&single).is_none(),
        "a single-id tile must refuse for the same reason every other tile does"
    );

    // MID-MAGNITUDE identifiers — the class a 2^53-only threshold lets through.
    // OSM node ids sit around 1.2e10: comfortably inside f64's exact-integer
    // range, but past the Int32 leaf, so a range-adaptive rule fits them into
    // UInt16 at a step of ~168k and every id decodes off by up to ±84k,
    // silently, in every tile. They must refuse for the same reason the 64-bit
    // hashes do.
    let osm_node_ids: Vec<Option<f64>> = vec![
        Some(12_345_678_901.0),
        Some(12_345_679_733.0),
        Some(11_982_004_117.0),
        None,
    ];
    assert!(
        build_quantized_numeric_auto(&osm_node_ids).is_none(),
        "mid-magnitude ids must NOT be quantized either"
    );

    // Both sides of the threshold itself.
    let under: Vec<Option<f64>> = vec![
        Some(AUTO_QUANT_MAX_ABS - 100.0),
        Some(AUTO_QUANT_MAX_ABS - 1.0),
    ];
    let (dt, _, back) =
        auto_quant_roundtrip(&under).expect("inside the threshold is exactly encodable");
    assert_eq!(dt, DataType::UInt16);
    assert_eq!(back, under);
    let at: Vec<Option<f64>> = vec![Some(0.0), Some(AUTO_QUANT_MAX_ABS)];
    assert!(
        build_quantized_numeric_auto(&at).is_none(),
        "the threshold is inclusive"
    );
    // A negative magnitude counts the same — the test is on |v|.
    let negative: Vec<Option<f64>> = vec![Some(-AUTO_QUANT_MAX_ABS - 1.0), Some(-1.0)];
    assert!(
        build_quantized_numeric_auto(&negative).is_none(),
        "magnitude is absolute"
    );
    // f64 still holds every one of these integers exactly, which is what makes
    // "leave it Float64" a lossless answer rather than a punt.
    assert!(AUTO_QUANT_MAX_ABS < F64_EXACT_INT_LIMIT);

    // The magnitude test is applied BEFORE the integer/fraction split, so one
    // fractional row in a tile cannot route the huge values around it: were the
    // test reachable only on the all-integer path, this tile would quantize
    // while its all-integer siblings stayed Float64 — the same structural drift
    // by another door.
    let mut with_a_fraction = ids.clone();
    with_a_fraction.push(Some(0.5));
    assert!(
        build_quantized_numeric_auto(&with_a_fraction).is_none(),
        "a fractional row must not smuggle >2^53 values into the quantizer"
    );
}

#[test]
fn auto_quant_constant_and_all_null_columns_stay_uint16() {
    // A constant column maps every present value to index 0, which reconstructs
    // to the offset EXACTLY — for integers and fractions alike. Keeping it
    // UInt16 costs nothing (it compresses to nothing) and keeps the column's
    // type identical to the tiles that do carry a range.
    let constant: Vec<Option<f64>> = vec![Some(3.5), Some(3.5), None, Some(3.5)];
    let (dt, affine, back) = auto_quant_roundtrip(&constant).expect("constant column quantizes");
    assert_eq!(dt, DataType::UInt16);
    assert_eq!((affine.o, affine.s), (3.5, 1.0));
    assert_eq!(back, constant);

    let constant_int: Vec<Option<f64>> = vec![Some(42.0); 3];
    let (dt, affine, back) = auto_quant_roundtrip(&constant_int).unwrap();
    assert_eq!(dt, DataType::UInt16);
    assert_eq!((affine.o, affine.s), (42.0, 1.0));
    assert_eq!(back, constant_int);

    // No finite value at all: all-null UInt16 at the neutral {0, 1} affine.
    // There is no data to reason about, so this keeps the historical choice.
    let empty: Vec<Option<f64>> = vec![None, None, Some(f64::NAN), Some(f64::INFINITY)];
    let (dt, affine, back) = auto_quant_roundtrip(&empty).expect("all-null column still quantizes");
    assert_eq!(dt, DataType::UInt16);
    assert_eq!((affine.o, affine.s), (0.0, 1.0));
    assert_eq!(
        back,
        vec![None, None, None, None],
        "non-finite cells are null"
    );
}

#[test]
fn auto_quant_continuous_column_range_adapts_and_never_refuses_on_the_sample() {
    // A genuinely continuous column is untouched by the integer work: still the
    // historical range-adaptive UInt16 with s = span/65535 and error <= s/2.
    let sst: Vec<Option<f64>> = (0..64)
        .map(|i| Some(-1.8 + i as f64 * 0.4732))
        .chain([None])
        .collect();
    let (dt, affine, back) = auto_quant_roundtrip(&sst).expect("continuous column must quantize");
    assert_eq!(dt, DataType::UInt16);
    let span = 63.0 * 0.4732;
    assert!(
        (affine.s - span / u16::MAX as f64).abs() < 1e-12,
        "step is the column span over the 16-bit index space, got {}",
        affine.s
    );
    for (got, want) in back.iter().zip(&sst) {
        match (got, want) {
            (Some(g), Some(w)) => assert!((g - w).abs() <= affine.s / 2.0 + 1e-12),
            (None, None) => {}
            _ => panic!("null pattern must survive"),
        }
    }

    // Outlier-inflated span: 20 readings around 1.5 plus one value at 1e6 — all
    // well inside the magnitude threshold, so the refusal does not apply. The
    // step becomes ~15, so the body of the column collapses onto index 0 and
    // decodes as 1.5 — genuinely coarse, and it STILL quantizes. A distribution
    // guard is NOT the answer here: it keys off the tile's own sample, so the
    // identical column drops to Float64 in whichever tiles catch the outlier
    // and stays UInt16 everywhere else, which is the drift stt-validate
    // hard-fails on. Coarseness is the advertised cost of an opt-in lossy lever;
    // a column whose Arrow type depends on its rows is a broken archive. The
    // real fix is a dataset-wide range pre-pass — see the deferred register in
    // docs/roadmap/stt-packed-format-decisions.md.
    let mut outlier: Vec<Option<f64>> = (0..20).map(|i| Some(1.5 + i as f64 * 0.1)).collect();
    outlier.push(Some(1.0e6 + 0.5));
    let (dt, affine, _) =
        auto_quant_roundtrip(&outlier).expect("an outlier does not veto the lever");
    assert_eq!(dt, DataType::UInt16);
    assert!(affine.s > 10.0, "the outlier really does inflate the step");

    // The same body WITHOUT the outlier is well conditioned and quantizes finely
    // — the two tiles differ in PRECISION, which is the lever's business, and
    // agree on TYPE, which is the format's.
    let body: Vec<Option<f64>> = (0..21).map(|i| Some(1.5 + i as f64 * 0.1)).collect();
    let (dt, _, back) = auto_quant_roundtrip(&body).expect("well-conditioned column quantizes");
    assert_eq!(dt, DataType::UInt16);
    for (got, want) in back.iter().zip(&body) {
        assert!((got.unwrap() - want.unwrap()).abs() < 1e-4);
    }
}

#[test]
fn auto_quant_gives_one_column_the_same_arrow_type_in_every_tile() {
    // One property must land on ONE Arrow type across every tile of a layer.
    // Take layer `pts`, property `v`: tile 0 holds 40 well-conditioned values
    // and quantizes to UInt16; tile 1 holds the SAME values plus one 1.0e9
    // outlier. If that outlier drops the tile to Float64, stt-validate's
    // `classify_column_drift` calls it `ColumnDrift::Structural` (Float64
    // shares no integer-width prefix with UInt16) and `stt-validate --json`
    // exits non-zero on the whole dataset. The encoder owes the validator a
    // stable type, not the validator a looser rule.
    let body: Vec<Option<f64>> = (0..40).map(|i| Some(12.5 + i as f64 * 0.25)).collect();
    let mut with_outlier = body.clone();
    with_outlier.push(Some(1.0e9 + 0.5));

    let tile = |values: &[Option<f64>]| -> DataType {
        let layer = ColumnarLayer {
            polygon_parts: None,
            name: "pts".into(),
            feature_ids: (0..values.len() as u64).collect(),
            start_times: (0..values.len() as i64).collect(),
            end_times: (0..values.len() as i64).map(|t| t + 1).collect(),
            geometry: GeometryColumn::Point(vec![[-73.98, 40.75]; values.len()]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![("v".into(), PropertyColumn::Numeric(values.to_vec()))],
        };
        let batch = decode_layer(
            &encode_layer_cfg(
                &layer,
                &EncoderConfig {
                    quantize_attrs_auto: true,
                    ..EncoderConfig::default()
                },
            )
            .unwrap(),
        )
        .unwrap();
        batch
            .schema()
            .field_with_name("v")
            .unwrap()
            .data_type()
            .clone()
    };

    assert_eq!(
        tile(&body),
        tile(&with_outlier),
        "one outlier must not change the column's Arrow type"
    );
    assert_eq!(tile(&body), DataType::UInt16);

    // Same claim for the integer regime, where the residual per-tile variation
    // lives: a wide tile widens the LEAF (UInt16 -> Int32), which
    // `classify_column_drift` rates AdaptiveWidth, and a span past the Int32
    // leaf but still under the magnitude threshold comes back to UInt16 rather
    // than falling out to Float64. Below the threshold, no tile of an integer
    // column is ever Float64.
    let narrow: Vec<Option<f64>> = vec![Some(1.0), Some(7.0), Some(1_000.0)];
    let wide: Vec<Option<f64>> = vec![Some(1.0), Some(7.0), Some(1_000_000.0)];
    let vast: Vec<Option<f64>> = vec![Some(-2.0e9), Some(7.0), Some(2.0e9)];
    assert_eq!(tile(&narrow), DataType::UInt16);
    assert_eq!(tile(&wide), DataType::Int32);
    assert_eq!(tile(&vast), DataType::UInt16);
    for t in [&narrow, &wide, &vast] {
        assert_ne!(tile(t), DataType::Float64, "no integer tile stays Float64");
    }

    // And the refusal itself is stable in the same way: an identifier column is
    // Float64 in EVERY tile, whether that tile caught one id or a wide spread of
    // them. This is the property the whole magnitude-over-span design exists to
    // buy — a per-tile refusal is exactly what stt-validate hard-fails on.
    let one_id: Vec<Option<f64>> = vec![Some(12_345_678_901.0); 3];
    let many_ids: Vec<Option<f64>> = vec![
        Some(12_345_678_901.0),
        Some(11_982_004_117.0),
        Some(13_500_000_000.0),
    ];
    assert_eq!(tile(&one_id), DataType::Float64);
    assert_eq!(tile(&many_ids), DataType::Float64);
}

#[test]
fn auto_quant_summary_and_id_columns_survive_a_full_encode_decode() {
    // Both corruption classes, end to end through the real encoder: a
    // summary-tier layer whose `count` / `bucket_i` / `sum_fare` are integers
    // and whose `trip_id` is a 64-bit hash. Counts must come back exact (via
    // the qa affine both readers apply), ids must stay Float64.
    let count: Vec<Option<f64>> = vec![Some(1.0), Some(7.0), Some(12_345.0), Some(2.0)];
    let bucket_i: Vec<Option<f64>> = vec![Some(0.0), Some(3.0), Some(11.0), Some(3.0)];
    let sum_fare: Vec<Option<f64>> = vec![Some(9.0), Some(1_000_003.0), Some(17.0), Some(4.0)];
    let trip_id: Vec<Option<f64>> = vec![
        Some(2_350_000_000_000_000_000.0),
        Some(9_100_000_000_000_000_000.0),
        Some(4_004_004_004_004_004_000.0),
        Some(7_777_777_777_777_777_000.0),
    ];
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "cells".into(),
        feature_ids: vec![1, 2, 3, 4],
        // Ascending, so the encoder's stable start_time sort is a no-op and the
        // decoded rows line up with the inputs.
        start_times: vec![0, 1, 2, 3],
        end_times: vec![10, 11, 12, 13],
        geometry: GeometryColumn::Point(vec![[-73.98, 40.75]; 4]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            ("count".into(), PropertyColumn::Numeric(count.clone())),
            ("bucket_i".into(), PropertyColumn::Numeric(bucket_i.clone())),
            ("sum_fare".into(), PropertyColumn::Numeric(sum_fare.clone())),
            ("trip_id".into(), PropertyColumn::Numeric(trip_id.clone())),
        ],
    };
    let batch = decode_layer(
        &encode_layer_cfg(
            &layer,
            &EncoderConfig {
                quantize_attrs_auto: true,
                ..EncoderConfig::default()
            },
        )
        .unwrap(),
    )
    .unwrap();

    // Every integer column reconstructs bit-for-bit through its shipped affine.
    for (name, want, leaf) in [
        ("count", &count, DataType::UInt16),
        ("bucket_i", &bucket_i, DataType::UInt16),
        // span 999_999 → widened leaf, still step 1.
        ("sum_fare", &sum_fare, DataType::Int32),
    ] {
        let field = batch.schema().field_with_name(name).unwrap().clone();
        assert_eq!(field.data_type(), &leaf, "{name} leaf");
        let affine =
            AttrQuant::from_json(field.metadata().get(STT_QUANT_ATTR_META_KEY).unwrap()).unwrap();
        assert_eq!(affine.s, 1.0, "{name} must quantize at step 1");
        let column = batch.column_by_name(name).unwrap();
        let got: Vec<f64> = match leaf {
            DataType::UInt16 => {
                let c = column.as_any().downcast_ref::<UInt16Array>().unwrap();
                (0..c.len())
                    .map(|i| affine.value(c.value(i) as i64))
                    .collect()
            }
            _ => {
                let c = column.as_any().downcast_ref::<Int32Array>().unwrap();
                (0..c.len())
                    .map(|i| affine.value(c.value(i) as i64))
                    .collect()
            }
        };
        let want: Vec<f64> = want.iter().map(|v| v.unwrap()).collect();
        assert_eq!(got, want, "{name} must round-trip exactly");
    }

    // The id column is refused outright: Float64, no affine, values untouched.
    let field = batch.schema().field_with_name("trip_id").unwrap().clone();
    assert_eq!(field.data_type(), &DataType::Float64);
    assert!(
        field.metadata().get(STT_QUANT_ATTR_META_KEY).is_none(),
        "an unquantized column must not carry a qa affine"
    );
    let ids = batch
        .column_by_name("trip_id")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for (i, want) in trip_id.iter().enumerate() {
        assert_eq!(
            ids.value(i),
            want.unwrap(),
            "trip_id[{i}] must be untouched"
        );
    }
}

#[test]
fn explicit_precision_snaps_integer_columns_to_an_exact_step() {
    // The explicit path must not be WORSE than the auto path: a requested
    // precision >= 1 on an integer column snaps down to step 1 (lossless, and
    // never coarser than asked), while everything else keeps its semantics.
    let count: Vec<Option<f64>> = vec![Some(1.0), Some(7.0), Some(12_345.0), None];
    let (array, json) = build_quantized_numeric(&count, 10.0).unwrap().unwrap();
    let affine = AttrQuant::from_json(&json).unwrap();
    assert_eq!(
        (affine.o, affine.s),
        (1.0, 1.0),
        "step 10 on an integer column snaps down to the exact step 1"
    );
    let c = array.as_any().downcast_ref::<UInt16Array>().unwrap();
    assert_eq!(affine.value(c.value(1) as i64), 7.0);
    assert!(c.is_null(3));

    // Sub-unit precision is untouched — the requested step still governs.
    let (_, json) = build_quantized_numeric(&count, 0.5).unwrap().unwrap();
    assert_eq!(AttrQuant::from_json(&json).unwrap().s, 0.5);

    // A fractional column at precision 10 keeps precision 10: the snap-down is
    // only ever applied where it is lossless.
    let mixed: Vec<Option<f64>> = vec![Some(1.5), Some(7.0), Some(300.0)];
    let (_, json) = build_quantized_numeric(&mixed, 10.0).unwrap().unwrap();
    assert_eq!(AttrQuant::from_json(&json).unwrap().s, 10.0);

    // Nor where the exact step would not fit the Int32 leaf: the requested
    // precision stands and the existing overflow guard keeps its behaviour.
    let vast: Vec<Option<f64>> = vec![Some(0.0), Some(3.0e9)];
    let (_, json) = build_quantized_numeric(&vast, 100.0).unwrap().unwrap();
    assert_eq!(AttrQuant::from_json(&json).unwrap().s, 100.0);
    assert!(build_quantized_numeric(&vast, 1.0).is_err());

    // And — the point of the gate — never where snapping would make the column
    // BIGGER. An integer column spanning 0..1_000_000 at precision 100 wants
    // 10_001 indices (UInt16, 2 B/row); step 1 would want 1_000_001 (Int32,
    // 4 B/row), doubling a column the user explicitly asked to shrink. The
    // request stands, at exactly the coarseness that was requested.
    let wide: Vec<Option<f64>> = vec![Some(0.0), Some(700.0), Some(1_000_000.0)];
    let (array, json) = build_quantized_numeric(&wide, 100.0).unwrap().unwrap();
    let affine = AttrQuant::from_json(&json).unwrap();
    assert_eq!(
        affine.s, 100.0,
        "snapping to step 1 here would widen the leaf to Int32"
    );
    assert_eq!(array.data_type(), &DataType::UInt16);
    let c = array.as_any().downcast_ref::<UInt16Array>().unwrap();
    assert_eq!(c.value(2), 10_000, "max index stays inside the 16-bit leaf");

    // The boundary: a span of exactly u16::MAX still snaps (step 1 indexes it
    // in UInt16), one past it does not.
    let at_edge: Vec<Option<f64>> = vec![Some(0.0), Some(u16::MAX as f64)];
    let (edge_array, json) = build_quantized_numeric(&at_edge, 4.0).unwrap().unwrap();
    assert_eq!(AttrQuant::from_json(&json).unwrap().s, 1.0);
    assert_eq!(edge_array.data_type(), &DataType::UInt16);
    let past_edge: Vec<Option<f64>> = vec![Some(0.0), Some(u16::MAX as f64 + 1.0)];
    let (past_array, json) = build_quantized_numeric(&past_edge, 4.0).unwrap().unwrap();
    assert_eq!(AttrQuant::from_json(&json).unwrap().s, 4.0);
    assert_eq!(past_array.data_type(), &DataType::UInt16);
}

#[test]
fn quantization_shrinks_geometry_and_is_opt_in() {
    // A many-vertex line is coordinate-dominated; quantization should shrink
    // the IPC, and the default (None) path must stay byte-identical.
    let line: Vec<[f64; 2]> = (0..400)
        .map(|k| [-73.95 + k as f64 * 1e-4, 40.75 + k as f64 * 7e-5])
        .collect();
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "q".into(),
        feature_ids: vec![1],
        start_times: vec![0],
        end_times: vec![1],
        geometry: GeometryColumn::LineString(vec![line]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    };
    let plain = encode_layer_quantized(&layer, None).unwrap();
    let quant = encode_layer_quantized(&layer, Some(1.0)).unwrap();

    // None keeps the Float64 GeoArrow leaf and carries no affine.
    let pb = decode_layer(&plain).unwrap();
    let pf = pb.schema().field_with_name("geometry").unwrap().clone();
    assert!(pf.metadata().get(STT_QUANT_META_KEY).is_none());

    // Some(_) switches the leaf to Int32 and emits the affine.
    let qb = decode_layer(&quant).unwrap();
    let qf = qb.schema().field_with_name("geometry").unwrap().clone();
    assert!(qf.metadata().get(STT_QUANT_META_KEY).is_some());

    // i32 coords (4 B) replace f64 (8 B) for a coordinate-dominated layer.
    assert!(
        quant.len() < plain.len(),
        "quantized {} should be smaller than f64 {}",
        quant.len(),
        plain.len()
    );
}

#[test]
fn line_layer_roundtrips_with_vertex_times() {
    let layer = sample_line_layer();
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    assert_eq!(batch.num_rows(), 2);
    let geom = batch
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    // Feature 0 has 3 vertices, feature 1 has 2.
    assert_eq!(geom.value(0).len(), 3);
    assert_eq!(geom.value(1).len(), 2);

    // v3 layers with a tight temporal span carry u16-delta vertex times
    // and the origin/step metadata needed to reconstruct absolutes.
    let meta = batch.schema().metadata().clone();
    let origin: i64 = meta
        .get("stt:vertex_time_origin_ms")
        .expect("u16 vertex-time layers carry an origin")
        .parse()
        .unwrap();
    let step: u32 = meta
        .get("stt:vertex_time_step_ms")
        .expect("u16 vertex-time layers carry a step")
        .parse()
        .unwrap();
    assert_eq!(origin, 0);
    assert_eq!(step, 1);

    let vt = batch
        .column_by_name("vertex_time")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(vt.len(), 2);
    let first = vt.value(0);
    let deltas = first
        .as_any()
        .downcast_ref::<arrow::array::UInt16Array>()
        .unwrap();
    let absolutes: Vec<i64> = deltas
        .values()
        .iter()
        .map(|d| origin + (*d as i64) * step as i64)
        .collect();
    assert_eq!(absolutes, vec![0, 25, 50]);
}

#[test]
fn line_layer_roundtrips_with_vertex_values() {
    // Per-vertex scalars (e.g. SST) ride a nullable List<Float32> aligned
    // with the geometry vertices. A NaN entry marks a vertex with no value.
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "drift".into(),
        feature_ids: vec![1, 2],
        start_times: vec![0, 0],
        end_times: vec![100, 100],
        geometry: GeometryColumn::LineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            vec![[3.0, 3.0], [4.0, 4.0]],
        ]),
        vertex_times: None,
        vertex_values: Some(vec![vec![5.0, f32::NAN, 27.5], vec![12.0, 13.0]]),
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    };
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    let vv = batch
        .column_by_name("vertex_value")
        .expect("layers with per-vertex values carry a vertex_value column")
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(vv.len(), 2);
    let first = vv.value(0);
    let vals = first
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap();
    assert_eq!(vals.value(0), 5.0);
    assert!(vals.value(1).is_nan());
    assert_eq!(vals.value(2), 27.5);
    let second = vv.value(1);
    let vals2 = second
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap();
    assert_eq!(vals2.values(), &[12.0, 13.0]);
}

#[test]
fn line_layer_roundtrips_with_vertex_value_matrix() {
    // Static-geometry overview: per-vertex × per-bucket value matrix rides a
    // nullable List<Float32>, flattened vertex-major (vertex 0's buckets,
    // then vertex 1's, ...). num_buckets is recorded in schema metadata.
    // Feature 0: 3 vertices × 2 buckets; feature 1: 2 vertices × 2 buckets.
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "flows".into(),
        feature_ids: vec![1, 2],
        start_times: vec![0, 0],
        end_times: vec![1800, 1800],
        geometry: GeometryColumn::LineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            vec![[3.0, 3.0], [4.0, 4.0]],
        ]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        // vertex-major: [v0b0, v0b1, v1b0, v1b1, v2b0, v2b1]
        vertex_value_matrix: Some(vec![
            vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            vec![40.0, 41.0, 50.0, 51.0],
        ]),
        properties: vec![],
    };
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    let vm = batch
        .column_by_name("vertex_value_matrix")
        .expect("matrix layers carry a vertex_value_matrix column")
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(vm.len(), 2);
    let f0 = vm.value(0);
    let f0v = f0
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap();
    assert_eq!(f0v.values(), &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    let f1 = vm.value(1);
    let f1v = f1
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap();
    assert_eq!(f1v.values(), &[40.0, 41.0, 50.0, 51.0]);

    // num_buckets = matrix_len / vertex_count = 6 / 3 = 2, in schema meta.
    assert_eq!(
        batch.schema().metadata().get("stt:vertex_value_buckets"),
        Some(&"2".to_string())
    );
}

/// One single-feature LineString layer whose two vertices are `span` ms apart
/// — the vertex-time tier ladder's only input.
fn vertex_time_span_layer(span: i64) -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "edge".into(),
        feature_ids: vec![1],
        start_times: vec![0],
        end_times: vec![100],
        geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0]]]),
        vertex_times: Some(vec![vec![0, span]]),
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

/// `(origin, step)` from the decoded schema metadata, `None` when the column
/// kept the absolute `List<Int64>` shape.
fn vertex_time_encoding_of(batch: &RecordBatch) -> Option<(i64, u32)> {
    let schema = batch.schema();
    let meta = schema.metadata();
    let origin = meta.get("stt:vertex_time_origin_ms")?;
    let step = meta
        .get("stt:vertex_time_step_ms")
        .expect("step rides origin");
    Some((origin.parse().unwrap(), step.parse().unwrap()))
}

#[test]
fn vertex_time_falls_back_to_int64_only_beyond_the_u32_tier() {
    // The i64 fallback is reached only when EVEN a u32 delta would need a
    // step past the ceiling — span > u32::MAX * 1000 ms ≈ 136 years. Below
    // that the u32 tier is both smaller and no less precise, so nothing
    // should ever take this path in practice.
    let layer = vertex_time_span_layer(5_000_000_000_000);
    let batch = decode_layer(&encode_layer(&layer).unwrap()).unwrap();
    assert_eq!(vertex_time_encoding_of(&batch), None);
    let vt = batch
        .column_by_name("vertex_time")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let first = vt.value(0);
    let absolutes = first
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("spans past every delta tier must keep the exact Int64 shape");
    assert_eq!(absolutes.values(), &[0, 5_000_000_000_000]);
}

#[test]
fn vertex_time_step_ceiling_is_the_u16_vs_u32_threshold() {
    // span = 65_535_000 ms quantizes at exactly the 1000 ms default ceiling →
    // u16 deltas; one ms more pushes the u16 step to 1001, so the encoder
    // drops to the u32 tier — which at this span fits at step 1, i.e. EXACT
    // millisecond precision at half the bytes the absolute i64 fallback costs.
    let at_ceiling = decode_layer(&encode_layer(&vertex_time_span_layer(65_535_000)).unwrap())
        .expect("decode at ceiling");
    assert_eq!(
        vertex_time_encoding_of(&at_ceiling),
        Some((0, DEFAULT_VERTEX_TIME_MAX_STEP_MS))
    );
    let vt = at_ceiling
        .column_by_name("vertex_time")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let first = vt.value(0);
    assert!(
        first.as_any().downcast_ref::<UInt16Array>().is_some(),
        "a step at the ceiling must stay on the u16 tier"
    );

    let past_ceiling = decode_layer(&encode_layer(&vertex_time_span_layer(65_536_000)).unwrap())
        .expect("decode past ceiling");
    assert_eq!(vertex_time_encoding_of(&past_ceiling), Some((0, 1)));
    let vt = past_ceiling
        .column_by_name("vertex_time")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let first = vt.value(0);
    let deltas = first
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("past the u16 ceiling the encoder takes the u32 delta tier");
    assert_eq!(deltas.values(), &[0, 65_536_000]);
}

/// The u32 tier's real-world case (`nyc-taxi-flows`): a day-wide layer that
/// the u16 tier cannot hold at 1 s precision but u32 holds EXACTLY, at half
/// the bytes the absolute `List<Int64>` fallback would cost.
#[test]
fn vertex_time_u32_tier_is_exact_for_multi_hour_spans() {
    for span in [86_400_000i64, 4_294_967_295] {
        let batch = decode_layer(&encode_layer(&vertex_time_span_layer(span)).unwrap()).unwrap();
        assert_eq!(
            vertex_time_encoding_of(&batch),
            Some((0, 1)),
            "span {span}ms must land on an exact-ms u32 delta"
        );
        let vt = batch
            .column_by_name("vertex_time")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let first = vt.value(0);
        let deltas = first.as_any().downcast_ref::<UInt32Array>().unwrap();
        // origin + delta * step reconstructs the input byte-for-byte.
        assert_eq!(deltas.values(), &[0, span as u32]);
    }

    // One ms past u32::MAX no longer fits at step 1; the tier stays u32 and
    // widens the step instead of falling to i64.
    let batch = decode_layer(&encode_layer(&vertex_time_span_layer(4_294_967_296)).unwrap())
        .expect("decode");
    assert_eq!(vertex_time_encoding_of(&batch), Some((0, 2)));
}

#[test]
fn polygon_layer_roundtrips_with_rings() {
    let layer = sample_polygon_layer();
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    let geom = batch
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(geom.len(), 1);
    // One feature with two rings (exterior + hole).
    let rings = geom.value(0);
    let rings = rings.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(rings.len(), 2);
    assert_eq!(rings.value(0).len(), 5); // exterior ring vertices
    assert_eq!(rings.value(1).len(), 5); // hole vertices
}

#[test]
fn multi_layer_tile_frame_roundtrips() {
    let layers = vec![sample_line_layer(), sample_point_layer()];
    let payload = encode_tile(&layers).unwrap();
    let decoded = decode_tile(&payload).unwrap();

    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].name, "tracks");
    assert_eq!(decoded[1].name, "points");
    assert_eq!(decoded[0].batch.num_rows(), 2);
    assert_eq!(decoded[1].batch.num_rows(), 3);
    // Schema metadata records the layer name on the batch too.
    assert_eq!(
        decoded[1]
            .batch
            .schema()
            .metadata()
            .get("stt:layer")
            .map(String::as_str),
        Some("points")
    );
}

#[test]
fn tessellate_polygon_emits_two_triangles_for_a_square() {
    // A simple closed square (5 verts, last duplicates first) earcuts into
    // exactly 2 triangles, 6 indices in [0, 3].
    let ring: Vec<Coord> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
    let tris = tessellate_polygon(&[ring]);
    assert_eq!(tris.len(), 6);
    for &i in &tris {
        assert!(i < 5);
    }
}

#[test]
fn tessellate_polygon_handles_a_hole() {
    // 4x4 square with a 1x1 hole — earcut should still produce a valid
    // tessellation. Index count is implementation-dependent but must be a
    // multiple of 3 and reference valid vertex indices.
    let exterior: Vec<Coord> = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]];
    let hole: Vec<Coord> = vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]];
    let tris = tessellate_polygon(&[exterior, hole]);
    assert!(tris.len() >= 6);
    assert_eq!(tris.len() % 3, 0);
    for &i in &tris {
        assert!(i < 10);
    }
}

#[test]
fn tessellate_polygon_handles_degenerate_input() {
    // No rings → empty result, not a panic.
    assert!(tessellate_polygon(&[]).is_empty());
    // Single 2-vert ring is below the 3-vertex minimum.
    let degenerate: Vec<Coord> = vec![[0.0, 0.0], [1.0, 1.0]];
    assert!(tessellate_polygon(&[degenerate]).is_empty());
}

#[test]
fn polygon_layer_with_triangles_roundtrips() {
    let exterior: Vec<Coord> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
    let tris = tessellate_polygon(&[exterior.clone()]);
    assert_eq!(tris.len(), 6);
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "zones".into(),
        feature_ids: vec![42],
        start_times: vec![0],
        end_times: vec![1000],
        geometry: GeometryColumn::Polygon(vec![vec![exterior]]),
        vertex_times: None,
        vertex_values: None,
        triangles: Some(vec![tris.clone()]),
        vertex_value_matrix: None,
        properties: vec![],
    };
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    // Schema metadata advertises the sidecar.
    assert_eq!(
        batch
            .schema()
            .metadata()
            .get(TRIANGLES_METADATA_KEY)
            .map(String::as_str),
        Some("true")
    );
    // Column exists with the expected shape. Indices here are tiny
    // (well under u16::MAX), so the narrower UInt16 encoding applies.
    let col = batch
        .column_by_name("triangles")
        .expect("triangles column present")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("triangles is a List");
    assert_eq!(col.len(), 1);
    let first = col.value(0);
    let values: &arrow::array::UInt16Array = first
        .as_any()
        .downcast_ref::<arrow::array::UInt16Array>()
        .expect("triangle values are UInt16 for small feature-local indices");
    assert_eq!(
        values
            .values()
            .iter()
            .map(|&v| v as u32)
            .collect::<Vec<_>>(),
        tris
    );
}

#[test]
fn polygon_layer_with_oversized_triangle_index_falls_back_to_uint32() {
    // A feature-local triangle index beyond u16::MAX (pathological, but
    // possible for a single giant polygon) must fall back to UInt32
    // rather than silently truncating.
    let exterior: Vec<Coord> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
    let big_tris = vec![0u32, 1, 70_000];
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "zones".into(),
        feature_ids: vec![42],
        start_times: vec![0],
        end_times: vec![1000],
        geometry: GeometryColumn::Polygon(vec![vec![exterior]]),
        vertex_times: None,
        vertex_values: None,
        triangles: Some(vec![big_tris.clone()]),
        vertex_value_matrix: None,
        properties: vec![],
    };
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    let col = batch
        .column_by_name("triangles")
        .expect("triangles column present")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("triangles is a List");
    let first = col.value(0);
    let values: &arrow::array::UInt32Array = first
        .as_any()
        .downcast_ref::<arrow::array::UInt32Array>()
        .expect("triangle values fall back to UInt32 when an index exceeds u16::MAX");
    assert_eq!(values.values().to_vec(), big_tris);
}

#[test]
fn polygon_layer_without_triangles_skips_the_metadata_key() {
    // Backwards-compat guarantee: a v3 polygon layer that was NOT built
    // with pre-tessellation must not carry the metadata flag — otherwise
    // a reader would expect a column that isn't there.
    let layer = sample_polygon_layer();
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();
    assert!(!batch
        .schema()
        .metadata()
        .contains_key(TRIANGLES_METADATA_KEY));
    assert!(batch.column_by_name("triangles").is_none());
}

#[test]
fn non_polygon_layer_drops_stray_triangles() {
    // A producer that mistakenly attaches `triangles` to a point or line
    // layer must not poison the wire format. The encoder silently drops
    // the column so the metadata key never appears.
    let mut layer = sample_point_layer();
    // Add a bogus per-feature triangle list. The encoder must ignore it.
    layer.triangles = Some(vec![vec![0, 1, 2]; layer.feature_ids.len()]);
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();
    assert!(!batch
        .schema()
        .metadata()
        .contains_key(TRIANGLES_METADATA_KEY));
    assert!(batch.column_by_name("triangles").is_none());
}

#[test]
fn truncated_tile_frame_errors_cleanly() {
    let payload = encode_tile(&[sample_point_layer()]).unwrap();
    // Chop the payload mid-stream; decode must error, not panic.
    let truncated = &payload[..payload.len() / 2];
    assert!(decode_tile(truncated).is_err());
}

#[test]
fn length_mismatch_is_rejected() {
    let mut layer = sample_point_layer();
    layer.start_times.pop(); // now 2 entries vs 3 features
    assert!(encode_layer(&layer).is_err());
}

/// `length_mismatch_is_rejected`'s scenario through the layer-frame path: a
/// length-inconsistent layer must be the same descriptive Err, not an index-out-of-bounds
/// panic inside `sort_rows_by_start_time`'s column permutation. The
/// start times are UNSORTED so the pre-sort actually permutes (a sorted
/// layer short-circuits before touching the truncated column).
#[test]
fn length_mismatch_is_rejected_by_v2_frame_too() {
    let mut layer = sample_point_layer();
    layer.start_times = vec![3000, 1000, 2000];
    layer.end_times.pop(); // now 2 entries vs 3 features
    let err = encode_tile_with(
        &[layer],
        &EncoderConfig {
            format_version: LAYER_FRAME_VERSION,
            ..EncoderConfig::default()
        },
    )
    .expect_err("length-inconsistent layer must Err through the layer-frame path");
    assert!(err.to_string().contains("end_times"), "got: {err}");
}

#[test]
fn offsets_from_counts_errors_on_i32_overflow() {
    // Accumulating past i32::MAX must be a hard error, not a silent wrap
    // (release builds would otherwise emit corrupt Arrow list offsets).
    let ok = offsets_from_counts([3usize, 2, 0].into_iter()).unwrap();
    assert_eq!(ok.len(), 4); // N+1 offsets

    let at_limit = offsets_from_counts([i32::MAX as usize].into_iter());
    assert!(at_limit.is_ok(), "exactly i32::MAX vertices still fits");

    let over = offsets_from_counts([i32::MAX as usize, 1].into_iter())
        .expect_err("i32::MAX + 1 total vertices must error");
    assert!(
        over.to_string().contains("32-bit list offsets"),
        "got: {over}"
    );

    // A single count beyond i32::MAX errors too (the per-count try_from).
    assert!(offsets_from_counts([usize::MAX].into_iter()).is_err());
}

#[test]
fn quantize_precision_below_floor_is_rejected() {
    // 1 mm would put the ±180° longitude index past i32::MAX. Both the
    // explicit-config path and the global setter must reject it with the
    // minimum in the message; a precision at/above the floor encodes fine.
    let layer = sample_point_layer();
    let err = encode_layer_cfg(
        &layer,
        &EncoderConfig {
            quantize_coords_m: Some(0.001),
            ..EncoderConfig::default()
        },
    )
    .expect_err("1 mm precision must be rejected");
    assert!(
        err.to_string().contains("minimum") && err.to_string().contains("overflow"),
        "error must state the minimum: {err}"
    );

    assert!(0.0187 > MIN_QUANTIZE_COORDS_M);
    assert!(encode_layer_cfg(
        &layer,
        &EncoderConfig {
            quantize_coords_m: Some(0.0187),
            ..EncoderConfig::default()
        },
    )
    .is_ok());

    // Global setter: rejects the same value WITHOUT storing it; <= 0
    // (off) still passes. (No positive value is stored here so the test
    // can't leak quantization into concurrently running global-path tests.)
    assert!(set_quantize_coords_m(0.001).is_err());
    assert!(set_quantize_coords_m(0.0).is_ok());
}

#[test]
fn quantized_altitude_outside_i32_errors_instead_of_clamping() {
    // The z axis is unbounded input: a value whose grid index leaves i32
    // must error (identifying the value), not clamp to ±i32::MAX quanta.
    let make = |z: f64| ColumnarLayer {
        polygon_parts: None,
        name: "cloud".into(),
        feature_ids: vec![1],
        start_times: vec![0],
        end_times: vec![0],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![("z".into(), PropertyColumn::Numeric(vec![Some(z)]))],
    };
    let cfg = EncoderConfig {
        quantize_coords_m: Some(0.05),
        point_elevation_column: "z".to_string(),
        ..EncoderConfig::default()
    };
    // Sane altitude still encodes.
    assert!(encode_layer_cfg(&make(5.0), &cfg).is_ok());
    // 1e18 m at a 0.05 m step → index 2e19, far outside i32.
    let err = encode_layer_cfg(&make(1.0e18), &cfg).expect_err("overflowing altitude must error");
    let msg = err.to_string();
    // f64 Display renders 1.0e18 as its full integer form.
    assert!(
        msg.contains("altitude") && msg.contains("1000000000000000000"),
        "got: {msg}"
    );
}

#[test]
fn quantized_attr_index_beyond_i32_errors_instead_of_clamping() {
    // An attribute whose quantized index exceeds i32::MAX must error
    // (mirroring the dictionary-overflow error), not clamp to i32::MAX.
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "wide".into(),
        feature_ids: vec![1, 2],
        start_times: vec![0; 2],
        end_times: vec![1; 2],
        geometry: GeometryColumn::Point(vec![[0.0, 0.0]; 2]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "v".into(),
            PropertyColumn::Numeric(vec![Some(0.0), Some(3.0e9)]),
        )],
    };
    let err = encode_layer_cfg(
        &layer,
        &EncoderConfig {
            quantize_attrs: HashMap::from([("v".to_string(), 1.0f64)]),
            ..EncoderConfig::default()
        },
    )
    .expect_err("index 3e9 > i32::MAX must error");
    let msg = err.to_string();
    assert!(
        msg.contains("overflows") && msg.contains("3000000000"),
        "got: {msg}"
    );
}

// ------------------------------------------------------------------
// Layer frame v2 (packed formatVersion 3)
// ------------------------------------------------------------------

/// Layer-frame config in SELF-CONTAINED mode (inline schema sections, no registry
/// needed to decode), layered over `base`.
fn v2_inline(base: &EncoderConfig) -> EncoderConfig {
    EncoderConfig {
        format_version: LAYER_FRAME_VERSION,
        template_collector: None,
        ..base.clone()
    }
}

/// Layer-frame config in HASH-REFERENCING mode (templates recorded with
/// `collector`; frames carry 16-byte hashes), layered over `base`.
fn v2_hashed(base: &EncoderConfig, collector: &Arc<TemplateCollector>) -> EncoderConfig {
    EncoderConfig {
        format_version: LAYER_FRAME_VERSION,
        template_collector: Some(Arc::clone(collector)),
        ..base.clone()
    }
}

/// Decode-side registry over everything `collector` recorded.
fn registry_from(collector: &TemplateCollector) -> TemplateRegistry {
    let mut registry = TemplateRegistry::new();
    for (_, template) in collector.snapshot() {
        registry.insert(template);
    }
    registry
}

/// The decode-equivalence contract `merge_v2_layer` owes downstream: the SAME
/// layers decode to EQUAL `DecodedLayer`s through the caller's own config and
/// through BOTH template modes (inline schema sections, hash references).
/// Batch equality covers columns AND schema/field metadata, so the TILE_META
/// re-injection must reproduce the Arrow metadata byte-for-byte.
fn assert_v2_decodes_like_v1(layers: &[ColumnarLayer], base: &EncoderConfig, what: &str) {
    let baseline = decode_tile(&encode_tile_with(layers, base).unwrap()).unwrap();

    let inline = decode_tile(&encode_tile_with(layers, &v2_inline(base)).unwrap()).unwrap();
    assert_eq!(inline.len(), baseline.len(), "{what}: inline layer count");
    for (a, b) in inline.iter().zip(&baseline) {
        assert_eq!(a.name, b.name, "{what}: inline layer name");
        assert_eq!(
            a.batch, b.batch,
            "{what}: inline-schema decode != baseline decode"
        );
    }

    let collector = Arc::new(TemplateCollector::new());
    let payload = encode_tile_with(layers, &v2_hashed(base, &collector)).unwrap();
    let registry = registry_from(&collector);
    let hashed = decode_tile_with_templates(&payload, &registry).unwrap();
    assert_eq!(hashed.len(), baseline.len(), "{what}: hashed layer count");
    for (a, b) in hashed.iter().zip(&baseline) {
        assert_eq!(a.name, b.name, "{what}: hashed layer name");
        assert_eq!(
            a.batch, b.batch,
            "{what}: hash-referencing decode != baseline decode"
        );
    }
}

/// Every payload shape the layer frame touches decodes to the SAME batch in
/// every template mode: plain + quantized points, dictionary props (incl. TWO
/// dictionary columns), u16-delta AND exact-Int64 vertex_time, the
/// vertex-value matrix, pre-tessellated triangles, an empty-bucket tile
/// (zero rows, dictionary column intact), and a multi-layer tile.
#[test]
fn v2_roundtrip_equals_v1_decode_across_payload_shapes() {
    let plain = EncoderConfig::default();
    let quant = EncoderConfig {
        quantize_coords_m: Some(1.0),
        quantize_attrs_auto: true,
        ..EncoderConfig::default()
    };

    let two_dicts = ColumnarLayer {
        properties: vec![
            (
                "kind".to_string(),
                PropertyColumn::Categorical(vec![Some("car".into()), Some("bus".into()), None]),
            ),
            (
                "color".to_string(),
                PropertyColumn::Categorical(vec![Some("red".into()), None, Some("blue".into())]),
            ),
        ],
        ..sample_point_layer()
    };

    let wide_span_vt = ColumnarLayer {
        vertex_times: Some(vec![vec![0, 100_000_000_000], vec![0, 1]]),
        ..sample_line_layer()
    };

    let matrix = ColumnarLayer {
        polygon_parts: None,
        name: "flows".into(),
        feature_ids: vec![1, 2],
        start_times: vec![0, 0],
        end_times: vec![1800, 1800],
        geometry: GeometryColumn::LineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            vec![[3.0, 3.0], [4.0, 4.0]],
        ]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: Some(vec![
            vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            vec![40.0, 41.0, 50.0, 51.0],
        ]),
        properties: vec![],
    };

    let exterior: Vec<Coord> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
    let triangles = ColumnarLayer {
        triangles: Some(vec![tessellate_polygon(&[exterior.clone()])]),
        geometry: GeometryColumn::Polygon(vec![vec![exterior]]),
        ..sample_polygon_layer()
    };

    let empty = ColumnarLayer {
        polygon_parts: None,
        name: "points".into(),
        feature_ids: vec![],
        start_times: vec![],
        end_times: vec![],
        geometry: GeometryColumn::Point(vec![]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            ("speed".into(), PropertyColumn::Numeric(vec![])),
            ("kind".into(), PropertyColumn::Categorical(vec![])),
        ],
    };

    assert_v2_decodes_like_v1(&[sample_point_layer()], &plain, "points");
    assert_v2_decodes_like_v1(&[sample_point_layer()], &quant, "quantized points");
    assert_v2_decodes_like_v1(&[two_dicts], &plain, "two dictionary columns");
    assert_v2_decodes_like_v1(&[sample_line_layer()], &plain, "u16-delta vertex_time");
    assert_v2_decodes_like_v1(&[wide_span_vt], &plain, "exact Int64 vertex_time");
    assert_v2_decodes_like_v1(&[matrix], &plain, "vertex-value matrix");
    assert_v2_decodes_like_v1(&[triangles], &plain, "pre-tessellated triangles");
    assert_v2_decodes_like_v1(&[empty], &quant, "empty-bucket tile");
    assert_v2_decodes_like_v1(
        &[sample_line_layer(), sample_point_layer()],
        &plain,
        "multi-layer tile",
    );
}

/// Layer-frame row order: rows come out stable-sorted by `start_time`, ids
/// travel WITH their rows (the sort runs after id assignment), and the result
/// equals the decode of the same layer pre-sorted by hand.
#[test]
fn v2_rows_stable_sorted_by_start_time_after_id_assignment() {
    let unsorted = ColumnarLayer {
        polygon_parts: None,
        name: "points".into(),
        feature_ids: vec![1, 2, 3, 4],
        start_times: vec![3000, 1000, 2000, 1000],
        end_times: vec![3500, 1500, 2500, 1600],
        geometry: GeometryColumn::Point(vec![
            [-122.4, 37.7],
            [-122.5, 37.8],
            [-122.6, 37.9],
            [-122.7, 38.0],
        ]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "kind".into(),
            PropertyColumn::Categorical(vec![
                Some("a".into()),
                Some("b".into()),
                Some("c".into()),
                Some("d".into()),
            ]),
        )],
    };

    let decoded = decode_tile(
        &encode_tile_with(&[unsorted.clone()], &v2_inline(&EncoderConfig::default())).unwrap(),
    )
    .unwrap();
    let batch = &decoded[0].batch;
    let starts = batch
        .column_by_name("start_time")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    // Stable: the two 1000s keep input order (id 2 before id 4).
    assert_eq!(starts, vec![1000, 1000, 2000, 3000]);
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(ids, vec![2, 4, 3, 1], "ids must travel with their rows");

    // Equivalent to encoding the manually pre-sorted layer.
    let presorted = sort_rows_by_start_time(&unsorted).into_owned();
    let presorted_decoded =
        decode_tile(&encode_tile_with(&[presorted], &EncoderConfig::default()).unwrap()).unwrap();
    assert_eq!(batch, &presorted_decoded[0].batch);
}

/// Template constancy: tiles differing ONLY per-tile — qa affines, t0,
/// dictionary categories, row counts — share ONE CORE + ONE PROPS template,
/// which is what makes the template table small. Type variants (u16-delta vs
/// exact-Int64 vertex_time) change the schema itself and so legitimately mint
/// DISTINCT templates, and every recorded template resolves through the
/// registry.
#[test]
fn v2_template_constancy_and_type_variant_cardinality() {
    let quant = EncoderConfig {
        quantize_coords_m: Some(1.0),
        quantize_attrs_auto: true,
        ..EncoderConfig::default()
    };
    let collector = Arc::new(TemplateCollector::new());
    let cfg = v2_hashed(&quant, &collector);

    let tile = |seed: i64, cats: [&str; 2], n: usize| ColumnarLayer {
        polygon_parts: None,
        name: "points".into(),
        feature_ids: (0..n as u64).collect(),
        start_times: (0..n as i64).map(|i| seed + i * 250).collect(),
        end_times: (0..n as i64).map(|i| seed + i * 250 + 100).collect(),
        geometry: GeometryColumn::Point((0..n).map(|i| [-122.0 + i as f64 * 1e-3, 37.0]).collect()),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "speed".into(),
                PropertyColumn::Numeric(
                    (0..n)
                        .map(|i| Some(seed as f64 * 0.01 + i as f64))
                        .collect(),
                ),
            ),
            (
                "kind".into(),
                PropertyColumn::Categorical(
                    (0..n).map(|i| Some(cats[i % 2].to_string())).collect(),
                ),
            ),
        ],
    };

    let a = encode_tile_with(&[tile(1_000_000, ["car", "bus"], 3)], &cfg).unwrap();
    let b = encode_tile_with(&[tile(9_000_000, ["tram", "ferry"], 5)], &cfg).unwrap();
    assert_ne!(a, b, "per-tile content must still differ");
    assert_eq!(
        collector.len(),
        2,
        "qa/t0/category/row-count variance must NOT mint new templates (core+props)"
    );

    // Type variants: narrow-span (u16-delta) vs wide-span (Int64)
    // vertex_time change the CORE schema → one new template each.
    let narrow = sample_line_layer();
    let wide = ColumnarLayer {
        vertex_times: Some(vec![vec![0, 100_000_000_000], vec![0, 1]]),
        ..sample_line_layer()
    };
    let c = encode_tile_with(&[narrow], &cfg).unwrap();
    let d = encode_tile_with(&[wide], &cfg).unwrap();
    assert_eq!(
        collector.len(),
        4,
        "u16 vs Int64 vertex_time are distinct templates"
    );

    // Every frame resolves through a registry built from the collector.
    let registry = registry_from(&collector);
    for payload in [&a, &b, &c, &d] {
        decode_tile_with_templates(payload, &registry).unwrap();
    }
}

/// TILE_META canonical serialization: alphabetical keys, no
/// whitespace, and unknown keys are ignored on decode (additive contract).
#[test]
fn v2_tile_meta_is_canonical_json_and_ignores_unknown_keys() {
    let meta = TileMeta {
        et: Some(EndTimeForm::Dur32),
        qa: Some(BTreeMap::from([("speed".to_string(), (0.0, 0.15))])),
        sorted: Some(true),
        st: Some(StartTimeForm::U32Offset),
        t0: Some(1_577_836_800_000),
        vb: Some(24),
        vq: Some(BTreeMap::from([(
            "vertex_value".to_string(),
            (-2.5, 0.001),
        )])),
        vt: Some((1_577_836_800_000, 1000)),
        // `vtf` is the OTHER vertex-time form and never ships beside `vt`; the
        // canonical bytes below therefore stay exactly as they were before it
        // existed, which is the point of an additive key.
        vtf: None,
    };
    assert_eq!(
        serde_json::to_string(&meta).unwrap(),
        r#"{"et":"dur32","qa":{"speed":[0.0,0.15]},"sorted":true,"st":"u32","t0":1577836800000,"vb":24,"vq":{"vertex_value":[-2.5,0.001]},"vt":[1577836800000,1000]}"#
    );
    // The compact-time keys are OPTIONAL and slot into the alphabetical order
    // without disturbing the pre-existing keys' relative positions — a tile
    // that takes neither compact form serializes exactly what it always did.
    assert_eq!(
        serde_json::to_string(&TileMeta {
            sorted: Some(true),
            t0: Some(7),
            ..TileMeta::default()
        })
        .unwrap(),
        r#"{"sorted":true,"t0":7}"#
    );
    // An unknown VALUE of a known compact-time key is a hard error, not a
    // silent misread: the `time-delta` capability is the version gate.
    assert!(serde_json::from_str::<TileMeta>(r#"{"st":"u64"}"#).is_err());
    assert!(serde_json::from_str::<TileMeta>(r#"{"et":"i32"}"#).is_err());
    // Presence rules: absent features serialize NO key at all.
    assert_eq!(serde_json::to_string(&TileMeta::default()).unwrap(), "{}");
    // Unknown keys from a future writer must be ignored, not rejected.
    let parsed: TileMeta = serde_json::from_str(r#"{"sorted":true,"zz_future":{"x":1}}"#).unwrap();
    assert_eq!(parsed.sorted, Some(true));
}

/// Walk a SINGLE-layer layer frame's header and return each section's
/// `(tag, payload_offset, len)` — test-side mirror of the decoder's TOC
/// walk, so guard tests can doctor exact section bytes.
fn v2_section_spans(payload: &[u8]) -> Vec<(u8, usize, usize)> {
    assert!(is_frame_v2(payload));
    let mut pos = 6usize; // escape + frame_version + flags + layer_count
    let name_len = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
    pos += 2 + name_len;
    for _ in 0..2 {
        let kind = payload[pos];
        pos += 1;
        if kind == REF_KIND_TEMPLATE_HASH {
            pos += 16;
        }
    }
    let section_count = payload[pos] as usize;
    pos += 1;
    let mut toc = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        let tag = payload[pos];
        let len = u32::from_le_bytes(payload[pos + 1..pos + 5].try_into().unwrap()) as usize;
        toc.push((tag, len));
        pos += 5;
    }
    pos += (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
    let mut spans = Vec::with_capacity(section_count);
    for (tag, len) in toc {
        spans.push((tag, pos, len));
        pos += len;
        pos += (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
    }
    spans
}

/// Splice guards: stray zeros at a batch section's head
/// must ERROR LOUDLY — under arrow-rs they'd otherwise parse as a legacy
/// 4-byte end-of-stream and silently EMPTY the tile.
#[test]
fn v2_stray_zeros_in_batch_section_error_instead_of_empty_tile() {
    let payload = encode_tile_with(
        &[sample_point_layer()],
        &v2_inline(&EncoderConfig::default()),
    )
    .unwrap();
    let (_, off, len) = *v2_section_spans(&payload)
        .iter()
        .find(|(tag, _, _)| *tag == SECTION_CORE_BATCH)
        .expect("CORE_BATCH present");
    assert!(len > 4);

    let mut doctored = payload.clone();
    doctored[off..off + 4].fill(0);
    let err = decode_tile(&doctored).expect_err("zeroed continuation must error");
    assert!(
        err.to_string().contains("0xFFFFFFFF"),
        "error must name the continuation guard: {err}"
    );

    // Direct splice guard: both halves are checked.
    let template = &payload[..4]; // any bytes NOT starting with FFFFFFFF
    assert!(splice_decode(&[0u8; 8], template, "guard").is_err());
}

/// Frame guards: truncations anywhere in the header error cleanly, and a
/// TOC length overrunning the payload is rejected before Arrow sees it.
#[test]
fn v2_truncated_header_and_lying_toc_length_error() {
    let payload = encode_tile_with(
        &[sample_point_layer()],
        &v2_inline(&EncoderConfig::default()),
    )
    .unwrap();
    let first_section_off = v2_section_spans(&payload)[0].1;
    for cut in 0..first_section_off {
        assert!(
            decode_tile(&payload[..cut]).is_err(),
            "cut at {cut} must error"
        );
    }

    // Inflate the first TOC length (u32 after the 1-byte tag): the
    // decoder's bounds-checked read must reject it, not over-read.
    let mut doctored = payload.clone();
    let toc0 = first_toc_offset(&payload);
    doctored[toc0 + 1..toc0 + 5].copy_from_slice(&u32::MAX.to_le_bytes());
    let err = decode_tile(&doctored).expect_err("overrunning TOC length must error");
    assert!(err.to_string().contains("truncated"), "got: {err}");
}

/// Byte offset of a single-layer layer frame's FIRST TOC entry.
fn first_toc_offset(payload: &[u8]) -> usize {
    let name_len = u16::from_le_bytes([payload[6], payload[7]]) as usize;
    let mut pos = 8 + name_len;
    for _ in 0..2 {
        let kind = payload[pos];
        pos += 1;
        if kind == REF_KIND_TEMPLATE_HASH {
            pos += 16;
        }
    }
    pos + 1 // past section_count
}

/// Unknown section tags are SKIPPED via their TOC length (additive
/// evolution): re-tagging TILE_META to an unknown tag still decodes the
/// batch — only the re-injected per-tile metadata disappears.
#[test]
fn v2_unknown_section_tag_is_skipped() {
    let payload = encode_tile_with(
        &[sample_point_layer()],
        &v2_inline(&EncoderConfig::default()),
    )
    .unwrap();
    let toc0 = first_toc_offset(&payload);
    let mut doctored = payload.clone();
    // TOC entries are (u8 tag, u32 len); find TILE_META's entry.
    let section_count = doctored[toc0 - 1] as usize;
    let mut retagged = false;
    for i in 0..section_count {
        let at = toc0 + i * 5;
        if doctored[at] == SECTION_TILE_META {
            doctored[at] = 0x6f; // unknown tag
            retagged = true;
        }
    }
    assert!(retagged);
    let decoded = decode_tile(&doctored).unwrap();
    assert_eq!(decoded[0].batch.num_rows(), 3);
    assert!(
        decoded[0]
            .batch
            .schema()
            .metadata()
            .get(TIME_OFFSET_MS_KEY)
            .is_none(),
        "skipped TILE_META means no t0 re-injection"
    );
}

/// A hash-referencing layer frame decoded WITHOUT (or with an incomplete)
/// registry is a descriptive error naming the fix — never a panic.
#[test]
fn v2_hash_frame_without_registry_errors_descriptively() {
    let collector = Arc::new(TemplateCollector::new());
    let payload = encode_tile_with(
        &[sample_point_layer()],
        &v2_hashed(&EncoderConfig::default(), &collector),
    )
    .unwrap();

    let err = decode_tile(&payload).expect_err("no registry must error");
    assert!(
        err.to_string().contains("decode_tile_with_templates"),
        "error must point at the registry entry point: {err}"
    );

    let empty = TemplateRegistry::new();
    let err =
        decode_tile_with_templates(&payload, &empty).expect_err("incomplete registry must error");
    assert!(
        err.to_string().contains("not in the dataset's registry"),
        "got: {err}"
    );
}

/// The strip is tail-invariant: hoisting the
/// per-tile metadata out of the schema changes ONLY the schema message —
/// two tiles differing in per-tile metadata alone share byte-identical
/// templates while their tails differ, which is exactly what makes
/// cross-tile template dedup sound.
#[test]
fn v2_metadata_strip_leaves_template_constant_and_tails_differing() {
    let quant = EncoderConfig {
        quantize_attrs_auto: true,
        ..EncoderConfig::default()
    };
    let mut early = sample_point_layer();
    // Different t0 + different qa affine (value range) per tile. BOTH time
    // columns shift together: the compact-time forms are derived per layer
    // from `end - start`, so shifting only the starts would give the two
    // tiles different `end_time` types (negative durations fall back to
    // absolute Int64) and fork the template for a reason this test isn't
    // about.
    for (s, e) in early.start_times.iter_mut().zip(early.end_times.iter_mut()) {
        *s += 7_000;
        *e += 7_000;
    }
    let late = sample_point_layer();

    let a = encode_layer_v2_parts(&early, &quant).unwrap();
    let b = encode_layer_v2_parts(&late, &quant).unwrap();
    assert_eq!(
        a.core_template, b.core_template,
        "CORE template must be constant"
    );
    let (a_props_template, a_props_tail) = a.props.as_ref().unwrap();
    let (b_props_template, b_props_tail) = b.props.as_ref().unwrap();
    assert_eq!(
        a_props_template, b_props_template,
        "PROPS template must be constant"
    );
    assert_ne!(
        a.tile_meta_json, b.tile_meta_json,
        "TILE_META varies per tile"
    );
    // A pure t0 shift does not reach the CORE tail at all: with compact
    // times `start_time` is an offset FROM t0 and `end_time` a duration, so
    // translating a whole layer through time changes only TILE_META's `t0`.
    // (The tail still absorbs everything else per-tile — geometry, ids,
    // durations — which is what keeps the template constant.)
    assert_eq!(
        a.core_tail, b.core_tail,
        "a whole-layer time shift is absorbed by TILE_META.t0, not the tail"
    );
    let mut shifted_duration = sample_point_layer();
    shifted_duration.end_times[0] += 1;
    let c = encode_layer_v2_parts(&shifted_duration, &quant).unwrap();
    assert_eq!(
        c.core_template, b.core_template,
        "CORE template must be constant"
    );
    assert_ne!(
        c.core_tail, b.core_tail,
        "a per-feature duration change must still land in the tail"
    );
    // Same qa affine + same categories here → identical props tails is
    // fine; what matters is templates never absorb per-tile variance.
    let _ = (a_props_tail, b_props_tail);
}

// ----------------------------------------------------------------------------
// Arrow IPC buffer alignment (see `IPC_BUFFER_ALIGNMENT`)
// ----------------------------------------------------------------------------

/// Walk an Arrow IPC stream and collect, for every RecordBatch / DictionaryBatch
/// message, the absolute stream offset of the body plus of each buffer the
/// message declares. Those are exactly the addresses a zero-copy reader takes a
/// typed view at.
fn ipc_body_and_buffer_offsets(ipc: &[u8]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= ipc.len() {
        assert_eq!(
            &ipc[pos..pos + 4],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            "every message starts with the continuation marker"
        );
        let meta_len = i32::from_le_bytes(ipc[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if meta_len == 0 {
            break; // end-of-stream marker
        }
        let msg = arrow::ipc::root_as_message(&ipc[pos + 8..pos + 8 + meta_len]).unwrap();
        let body_start = pos + 8 + meta_len;
        let buffers = match msg.header_type() {
            arrow::ipc::MessageHeader::RecordBatch => {
                msg.header_as_record_batch().and_then(|rb| rb.buffers())
            }
            arrow::ipc::MessageHeader::DictionaryBatch => msg
                .header_as_dictionary_batch()
                .and_then(|db| db.data())
                .and_then(|rb| rb.buffers()),
            _ => None,
        };
        if let Some(buffers) = buffers {
            out.push(body_start);
            for b in buffers.iter() {
                out.push(body_start + b.offset() as usize);
            }
        }
        pos = body_start + msg.bodyLength() as usize;
    }
    assert!(!out.is_empty(), "stream must carry at least one batch");
    out
}

/// IPC streams are written at the Arrow spec's 8-byte buffer alignment.
///
/// arrow-rs' `IpcWriteOptions::default()` pads every buffer to 64 bytes (a SIMD
/// recommendation; the Arrow spec requires 8). Across the shipped fleet that
/// padding is 19–39% of RAW IPC bytes — and the UNCOMPRESSED size is what
/// drives reader allocation and the client memory budget. Falling back to 64
/// (e.g. via `StreamWriter::try_new`) makes the size assertion below fail.
#[test]
fn ipc_streams_are_written_at_8_byte_buffer_alignment() {
    assert_eq!(
        IPC_BUFFER_ALIGNMENT, 8,
        "8 is the Arrow IPC spec's requirement AND the floor the TS reader's \
         zero-copy Float64/Float32 subarray paths need"
    );

    let layer = sample_point_layer();
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();

    // Re-serialize the same batch at arrow-rs' 64-byte default: the encoder's
    // stream must be strictly smaller, i.e. the padding it declines to write.
    let mut wide = Vec::new();
    {
        let opts = arrow::ipc::writer::IpcWriteOptions::try_new(
            64,
            false,
            arrow::ipc::MetadataVersion::V5,
        )
        .unwrap();
        let mut w = arrow::ipc::writer::StreamWriter::try_new_with_options(
            &mut wide,
            &batch.schema(),
            opts,
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
    }
    assert!(
        ipc.len() < wide.len(),
        "alignment 8 must emit fewer bytes than alignment 64 ({} vs {})",
        ipc.len(),
        wide.len()
    );

    // Every body and every declared buffer still starts on an 8-byte boundary
    // — the zero-copy floor. (The frame pads each section to FRAME_ALIGN = 8
    // and arrow-rs pads the schema message to the same alignment, so these
    // stream-relative offsets survive the template/tail splice.)
    for offset in ipc_body_and_buffer_offsets(&ipc) {
        assert_eq!(
            offset % IPC_BUFFER_ALIGNMENT,
            0,
            "buffer at stream offset {offset} is not 8-byte aligned"
        );
    }
}

// ----------------------------------------------------------------------------
// Compact feature times (TILE_META `st` / `et`)
// ----------------------------------------------------------------------------

/// A minimal props-less POINT layer with the given feature times.
fn point_layer_with_times(starts: &[i64], ends: &[i64]) -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "t".into(),
        feature_ids: (1..=starts.len() as u64).collect(),
        start_times: starts.to_vec(),
        end_times: ends.to_vec(),
        geometry: GeometryColumn::Point((0..starts.len()).map(|i| [i as f64 * 0.1, 1.0]).collect()),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

fn int64_column(batch: &RecordBatch, name: &str) -> Vec<i64> {
    let field = batch.schema().field_with_name(name).unwrap().clone();
    assert_eq!(
        field.data_type(),
        &DataType::Int64,
        "{name} must decode as absolute Int64"
    );
    assert!(!field.is_nullable(), "{name} is non-null per spec");
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

/// Every reachable (start form × end form) combination round-trips to the
/// EXACT absolute times, in a batch indistinguishable — schema, metadata,
/// column order and all — from the same layer encoded with the feature off.
///
/// That indistinguishability is the whole contract: it is why no downstream
/// Rust consumer (stt-validate's schema gate, stt-optimize's exporter, the
/// examples) needed a single line changed.
#[test]
fn compact_times_roundtrip_every_reachable_form() {
    const U32_MAX: i64 = u32::MAX as i64;
    let compact = EncoderConfig::default();
    let absolute = EncoderConfig {
        compact_times: false,
        ..EncoderConfig::default()
    };

    let cases: [(&str, Vec<i64>, Vec<i64>, Option<&str>, Option<&str>); 6] = [
        // start offsets fit u32 …
        (
            "u32 start / omitted end",
            vec![1000, 2000, 3000],
            vec![1000, 2000, 3000],
            Some("u32"),
            Some("zero"),
        ),
        (
            "u32 start / dur32 end",
            vec![1000, 2000, 3000],
            vec![1500, 2000, 9000],
            Some("u32"),
            Some("dur32"),
        ),
        (
            "u32 start / absolute end",
            vec![1000, 2000, 3000],
            vec![1000, 2000, 3000 + U32_MAX + 1],
            Some("u32"),
            None,
        ),
        // … and start offsets that do NOT (a >49.7-day tile span).
        (
            "absolute start / omitted end",
            vec![0, 10, U32_MAX + 1],
            vec![0, 10, U32_MAX + 1],
            None,
            Some("zero"),
        ),
        (
            "absolute start / dur32 end",
            vec![0, 10, U32_MAX + 1],
            vec![5, 10, U32_MAX + 7],
            None,
            Some("dur32"),
        ),
        (
            "absolute start / absolute end",
            vec![0, 10, U32_MAX + 1],
            vec![0, 10, 2 * U32_MAX + 3],
            None,
            None,
        ),
    ];

    for (label, starts, ends, want_st, want_et) in cases {
        let layer = point_layer_with_times(&starts, &ends);

        // 1. TILE_META records the choice (readers MUST branch on these keys,
        //    never on the Arrow DataType alone).
        let parts = encode_layer_v2_parts(&layer, &compact).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&parts.tile_meta_json).unwrap();
        assert_eq!(
            meta.get("st").and_then(|v| v.as_str()),
            want_st,
            "{label}: TILE_META.st"
        );
        assert_eq!(
            meta.get("et").and_then(|v| v.as_str()),
            want_et,
            "{label}: TILE_META.et"
        );
        assert_eq!(
            meta.get("t0").and_then(|v| v.as_i64()),
            Some(starts.iter().copied().min().unwrap()),
            "{label}: t0 anchors the u32 offsets and is therefore mandatory"
        );

        // 2. The decoded batch recovers the absolute times exactly …
        let payload = encode_tile_with(std::slice::from_ref(&layer), &compact).unwrap();
        let decoded = decode_tile(&payload).unwrap();
        let batch = &decoded[0].batch;
        assert_eq!(int64_column(batch, "start_time"), starts, "{label}: starts");
        assert_eq!(int64_column(batch, "end_time"), ends, "{label}: ends");

        // 3. … in the canonical column order, with `end_time` back at index 2
        //    even when the CORE batch never carried it.
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec!["id", "start_time", "end_time", "geometry"],
            "{label}: column order"
        );

        // 4. … and is byte-for-byte the batch the feature-off encoder yields.
        let plain =
            decode_tile(&encode_tile_with(std::slice::from_ref(&layer), &absolute).unwrap())
                .unwrap();
        assert_eq!(
            batch.schema(),
            plain[0].batch.schema(),
            "{label}: schema must be indistinguishable"
        );
        assert_eq!(
            batch, &plain[0].batch,
            "{label}: decoded batch must be indistinguishable"
        );
    }
}

/// Degenerate layers must all still encode and decode: an empty one keeps the
/// absolute pair outright (no `t0` to anchor against), and a single-feature
/// one compacts like any other.
#[test]
fn compact_times_handle_empty_and_single_feature_layers() {
    let cfg = EncoderConfig::default();

    let empty = point_layer_with_times(&[], &[]);
    let parts = encode_layer_v2_parts(&empty, &cfg).unwrap();
    assert_eq!(
        parts.tile_meta_json, r#"{"sorted":true}"#,
        "an empty layer declares neither compact form (nor a t0)"
    );
    let batch = &decode_tile(&encode_tile_with(std::slice::from_ref(&empty), &cfg).unwrap())
        .unwrap()[0]
        .batch;
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(int64_column(batch, "start_time"), Vec::<i64>::new());
    assert_eq!(int64_column(batch, "end_time"), Vec::<i64>::new());

    let one = point_layer_with_times(&[1_700_000_000_000], &[1_700_000_000_000]);
    let parts = encode_layer_v2_parts(&one, &cfg).unwrap();
    assert_eq!(
        parts.tile_meta_json,
        r#"{"et":"zero","sorted":true,"st":"u32","t0":1700000000000}"#
    );
    let batch = &decode_tile(&encode_tile_with(std::slice::from_ref(&one), &cfg).unwrap()).unwrap()
        [0]
    .batch;
    assert_eq!(int64_column(batch, "start_time"), vec![1_700_000_000_000]);
    assert_eq!(int64_column(batch, "end_time"), vec![1_700_000_000_000]);
}

/// The measured win: on the shape that dominates the fleet (instantaneous
/// events, `end == start` for 100% of features) the two time columns collapse
/// from 16 bytes/feature to 4.
#[test]
fn compact_times_shrink_the_instantaneous_event_shape() {
    let starts: Vec<i64> = (0..1000).map(|i| 1_700_000_000_000 + i * 37).collect();
    let layer = point_layer_with_times(&starts, &starts);

    let compact =
        encode_tile_with(std::slice::from_ref(&layer), &EncoderConfig::default()).unwrap();
    let absolute = encode_tile_with(
        std::slice::from_ref(&layer),
        &EncoderConfig {
            compact_times: false,
            ..EncoderConfig::default()
        },
    )
    .unwrap();

    // 1000 features: 8000 B start + 8000 B end → 4000 B start + 0 B end.
    let saved = absolute.len() - compact.len();
    assert!(
        saved >= 12_000,
        "expected ≥12 KiB saved on 1000 instantaneous features, got {saved} \
         ({} → {})",
        absolute.len(),
        compact.len()
    );
}

/// `--no-compact-times` is a true kill switch: the emitted frame carries
/// neither TILE_META key and the columns are the historical absolute pair.
#[test]
fn compact_times_kill_switch_emits_the_absolute_pair() {
    let layer = point_layer_with_times(&[1000, 2000], &[1500, 2500]);
    let parts = encode_layer_v2_parts(
        &layer,
        &EncoderConfig {
            compact_times: false,
            ..EncoderConfig::default()
        },
    )
    .unwrap();
    assert_eq!(parts.tile_meta_json, r#"{"sorted":true,"t0":1000}"#);
}

/// The STANDALONE layer shape never compacts: it has no TILE_META section to
/// carry `st`/`et` and `decode_layer` performs no re-inflation, so compacting
/// there would re-type a column with no way to read it back.
#[test]
fn standalone_layer_shape_never_compacts_times() {
    let starts = vec![1000i64, 2000, 3000];
    let layer = point_layer_with_times(&starts, &starts);
    let batch =
        decode_layer(&encode_layer_with(&layer, &EncoderConfig::default()).unwrap()).unwrap();
    assert_eq!(int64_column(&batch, "start_time"), starts);
    assert_eq!(int64_column(&batch, "end_time"), starts);
}

/// A TILE_META that declares a compact form the CORE batch does not carry is a
/// loud decode error, never a silent misread.
#[test]
fn compact_times_reject_inconsistent_tile_meta() {
    let starts = vec![1000i64, 2000];
    let layer = point_layer_with_times(&starts, &starts);
    // Encoded with the feature OFF → absolute Int64 columns, no st/et keys.
    let absolute = EncoderConfig {
        compact_times: false,
        ..EncoderConfig::default()
    };
    let ipc = encode_layer_with(&layer, &absolute).unwrap();
    let core = decode_layer(&ipc).unwrap();

    for (meta, want) in [
        (
            TileMeta {
                st: Some(StartTimeForm::U32Offset),
                t0: Some(1000),
                ..TileMeta::default()
            },
            "but 'start_time' is Int64",
        ),
        (
            TileMeta {
                st: Some(StartTimeForm::U32Offset),
                ..TileMeta::default()
            },
            "carries no 't0' anchor",
        ),
        (
            TileMeta {
                et: Some(EndTimeForm::Zero),
                ..TileMeta::default()
            },
            "carries an 'end_time' column",
        ),
        (
            TileMeta {
                et: Some(EndTimeForm::Dur32),
                ..TileMeta::default()
            },
            "but 'end_time' is Int64",
        ),
    ] {
        let err = merge_v2_layer(core.clone(), None, &meta)
            .expect_err("inconsistent TILE_META must error");
        assert!(
            err.to_string().contains(want),
            "expected {want:?}, got: {err}"
        );
    }
}

// ----------------------------------------------------------------------------
// Per-vertex value quantization (TILE_META `vq`)
// ----------------------------------------------------------------------------

/// A LINE layer whose features carry the given per-vertex values, and
/// optionally the same numbers again as a per-vertex value MATRIX. Geometry is
/// synthesized to match each feature's value count 1:1 (the alignment the
/// renderer relies on).
fn line_layer_with_vertex_values(
    values: Vec<Vec<f32>>,
    matrix: Option<Vec<Vec<f32>>>,
) -> ColumnarLayer {
    let n = values.len();
    let geometry: Vec<Vec<Coord>> = values
        .iter()
        .enumerate()
        .map(|(f, vs)| (0..vs.len()).map(|i| [f as f64, i as f64]).collect())
        .collect();
    ColumnarLayer {
        name: "vv".into(),
        feature_ids: (1..=n as u64).collect(),
        start_times: vec![0; n],
        end_times: vec![0; n],
        geometry: GeometryColumn::LineString(geometry),
        vertex_times: None,
        vertex_values: Some(values),
        triangles: None,
        polygon_parts: None,
        vertex_value_matrix: matrix,
        properties: vec![],
    }
}

/// Read a decoded `List<Float32>` column back as one vec per feature, asserting
/// the leaf really is `Float32` (i.e. the decoder re-inflated it).
fn float_list_column(batch: &RecordBatch, name: &str) -> Vec<Vec<f32>> {
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column '{name}' is missing"));
    let list = col
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap_or_else(|| panic!("column '{name}' is {}, expected a List", col.data_type()));
    assert_eq!(
        list.values().data_type(),
        &DataType::Float32,
        "'{name}' must decode as List<Float32> whatever it shipped as"
    );
    (0..list.len())
        .map(|i| {
            if list.is_null(i) {
                return Vec::new();
            }
            list.value(i)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

/// Decode ONLY a framed layer's CORE batch (no TILE_META re-inflation), so a test
/// can assert what actually went on the wire rather than what comes back out.
fn core_batch_of(parts: &EncodedLayerV2) -> RecordBatch {
    splice_decode(&parts.core_template, &parts.core_tail, "core").unwrap()
}

/// The contract in one test: the wire column is `UInt16`, `TILE_META.vq`
/// carries the affine, `NaN` survives via the reserved sentinel, every finite
/// value comes back inside half a step, and the DECODED batch is
/// indistinguishable in shape from the unquantized encode's.
#[test]
fn vertex_value_quant_roundtrips_within_half_a_step_and_keeps_nan() {
    let values = vec![
        vec![-2.5f32, 0.0, 12.25, f32::NAN],
        vec![31.75, 7.5],
        vec![], // a feature with no values at all → null list, both ways
    ];
    let layer = line_layer_with_vertex_values(values.clone(), None);
    let quant = EncoderConfig {
        quantize_vertex_values: true,
        ..EncoderConfig::default()
    };
    let plain = EncoderConfig::default();

    // 1. TILE_META records the affine, and only for the column that shipped
    //    quantized.
    let parts = encode_layer_v2_parts(&layer, &quant).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&parts.tile_meta_json).unwrap();
    let vq = meta.get("vq").expect("vq must be present");
    assert!(
        vq.get("vertex_value_matrix").is_none(),
        "no matrix on this layer"
    );
    let affine = vq.get("vertex_value").expect("vq.vertex_value");
    let (o, s) = (affine[0].as_f64().unwrap(), affine[1].as_f64().unwrap());
    assert_eq!(o, -2.5, "the offset is the column's finite minimum");
    assert!(
        (s - (31.75f64 - -2.5f64) / 65534.0).abs() < 1e-18,
        "the step spreads the finite range over 65535 levels, got {s}"
    );

    // 2. The WIRE column is a UInt16 list; the NaN vertex is the sentinel.
    let core = core_batch_of(&parts);
    let wire = core.column_by_name("vertex_value").unwrap();
    let wire_list = wire.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(wire_list.values().data_type(), &DataType::UInt16);
    let wire_leaf = wire_list
        .values()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .unwrap();
    assert_eq!(
        wire_leaf.value(3),
        VERTEX_VALUE_QUANT_SENTINEL,
        "the NaN vertex takes the reserved index"
    );
    assert!(
        wire_leaf
            .values()
            .iter()
            .filter(|&&q| q != VERTEX_VALUE_QUANT_SENTINEL)
            .all(|&q| q <= VERTEX_VALUE_QUANT_MAX),
        "no finite value may collide with the sentinel"
    );

    // 3. Decoding re-inflates Float32, within half a step, NaN preserved.
    let payload = encode_tile_with(std::slice::from_ref(&layer), &quant).unwrap();
    let batch = &decode_tile(&payload).unwrap()[0].batch;
    let got = float_list_column(batch, "vertex_value");
    assert_eq!(got.len(), 3);
    assert!(got[2].is_empty(), "the valueless feature stays a null list");
    for (feature, (want_row, got_row)) in values.iter().zip(&got).enumerate() {
        assert_eq!(want_row.len(), got_row.len(), "feature {feature} length");
        for (i, (&want, &got)) in want_row.iter().zip(got_row).enumerate() {
            if want.is_nan() {
                assert!(got.is_nan(), "feature {feature} vertex {i} must stay NaN");
            } else {
                assert!(
                    (f64::from(got) - f64::from(want)).abs() <= s / 2.0 + 1e-6,
                    "feature {feature} vertex {i}: {got} is more than half a step from {want}"
                );
            }
        }
    }

    // 4. Shape indistinguishability: same schema, same column order, same
    //    nulls as the unquantized encode — which is why nothing downstream of
    //    the decoder needed a line changed.
    let plain_batch =
        &decode_tile(&encode_tile_with(std::slice::from_ref(&layer), &plain).unwrap()).unwrap()[0]
            .batch;
    assert_eq!(
        batch.schema(),
        plain_batch.schema(),
        "the quantized tile must decode to the very same schema"
    );
    assert_eq!(
        batch.column_by_name("vertex_value").unwrap().nulls(),
        plain_batch.column_by_name("vertex_value").unwrap().nulls(),
    );
}

/// The two per-vertex value columns are quantized INDEPENDENTLY: each gets its
/// own affine sized from its own range, both under `vq`.
#[test]
fn vertex_value_matrix_gets_its_own_affine() {
    let values = vec![vec![0.0f32, 1.0], vec![2.0, 3.0]];
    // The matrix lives on a completely different scale (vehicle counts).
    let matrix = vec![vec![0.0f32, 4000.0], vec![100.0, 250.0]];
    let layer = line_layer_with_vertex_values(values, Some(matrix.clone()));
    let cfg = EncoderConfig {
        quantize_vertex_values: true,
        ..EncoderConfig::default()
    };

    let parts = encode_layer_v2_parts(&layer, &cfg).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&parts.tile_meta_json).unwrap();
    let vq = meta.get("vq").unwrap();
    assert_eq!(vq["vertex_value"][0].as_f64().unwrap(), 0.0);
    assert_eq!(vq["vertex_value"][1].as_f64().unwrap(), 3.0 / 65534.0);
    assert_eq!(vq["vertex_value_matrix"][0].as_f64().unwrap(), 0.0);
    assert_eq!(
        vq["vertex_value_matrix"][1].as_f64().unwrap(),
        4000.0 / 65534.0
    );
    // `vb` still describes the matrix's bucket count — the two are orthogonal.
    assert_eq!(meta.get("vb").and_then(|v| v.as_u64()), Some(1));

    let batch = &decode_tile(&encode_tile_with(std::slice::from_ref(&layer), &cfg).unwrap())
        .unwrap()[0]
        .batch;
    let got = float_list_column(batch, "vertex_value_matrix");
    for (f, (want_row, got_row)) in matrix.iter().zip(&got).enumerate() {
        for (i, (&want, &got)) in want_row.iter().zip(got_row).enumerate() {
            assert!(
                (f64::from(got) - f64::from(want)).abs() <= 4000.0 / 65534.0 / 2.0 + 1e-3,
                "matrix feature {f} cell {i}: {got} vs {want}"
            );
        }
    }
}

/// OFF by default: the flag is opt-in because it is lossy, so an unflagged
/// build must declare no `vq` affine and emit the bytes it would without the
/// lever at all.
#[test]
fn vertex_value_quant_is_off_by_default() {
    let layer = line_layer_with_vertex_values(vec![vec![1.0f32, 2.0, 3.0]], None);
    let parts = encode_layer_v2_parts(&layer, &EncoderConfig::default()).unwrap();
    assert!(
        !parts.tile_meta_json.contains("\"vq\""),
        "the default encode must not declare a vq affine: {}",
        parts.tile_meta_json
    );
    let core = core_batch_of(&parts);
    let list = core
        .column_by_name("vertex_value")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(
        list.values().data_type(),
        &DataType::Float32,
        "the default wire shape stays raw Float32"
    );
    assert!(!EncoderConfig::default().quantize_vertex_values);
}

/// The degenerate columns `build_quantized_numeric_auto` also has to survive:
/// one with a single repeated value (span 0) and one with nothing finite at all.
#[test]
fn vertex_value_quant_handles_constant_and_all_nan_columns() {
    let cfg = EncoderConfig {
        quantize_vertex_values: true,
        ..EncoderConfig::default()
    };

    // Constant column: step 1, every value at index 0 → reconstructs EXACTLY.
    let constant = line_layer_with_vertex_values(vec![vec![7.5f32, 7.5, 7.5]], None);
    let parts = encode_layer_v2_parts(&constant, &cfg).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&parts.tile_meta_json).unwrap();
    assert_eq!(meta["vq"]["vertex_value"][0].as_f64().unwrap(), 7.5);
    assert_eq!(meta["vq"]["vertex_value"][1].as_f64().unwrap(), 1.0);
    let batch = &decode_tile(&encode_tile_with(std::slice::from_ref(&constant), &cfg).unwrap())
        .unwrap()[0]
        .batch;
    assert_eq!(
        float_list_column(batch, "vertex_value")[0],
        vec![7.5f32, 7.5, 7.5],
        "a constant column round-trips exactly, not approximately"
    );

    // Nothing finite: the affine is pinned to (0, 1) so the TILE_META bytes
    // stay reproducible, and every vertex comes back NaN.
    let all_nan = line_layer_with_vertex_values(vec![vec![f32::NAN, f32::NAN]], None);
    let parts = encode_layer_v2_parts(&all_nan, &cfg).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&parts.tile_meta_json).unwrap();
    assert_eq!(meta["vq"]["vertex_value"][0].as_f64().unwrap(), 0.0);
    assert_eq!(meta["vq"]["vertex_value"][1].as_f64().unwrap(), 1.0);
    let batch = &decode_tile(&encode_tile_with(std::slice::from_ref(&all_nan), &cfg).unwrap())
        .unwrap()[0]
        .batch;
    assert!(float_list_column(batch, "vertex_value")[0]
        .iter()
        .all(|v| v.is_nan()));
}

/// The standalone (`encode_layer*`) shape has no TILE_META to carry `vq`, so
/// it MUST keep the raw Float32 leaf however the config is set — the same fork
/// point `compact_times` closes.
#[test]
fn standalone_layer_shape_never_quantizes_vertex_values() {
    let layer = line_layer_with_vertex_values(vec![vec![0.0f32, 100.0]], None);
    let cfg = EncoderConfig {
        quantize_vertex_values: true,
        ..EncoderConfig::default()
    };
    let batch = decode_layer(&encode_layer_with(&layer, &cfg).unwrap()).unwrap();
    assert_eq!(
        float_list_column(&batch, "vertex_value")[0],
        vec![0.0f32, 100.0],
        "the standalone shape must stay self-describing Float32"
    );
}

/// The measured point of the feature: half the bytes on the columns that
/// dominate a flow/corridor tile.
#[test]
fn vertex_value_quant_halves_the_column() {
    let values: Vec<Vec<f32>> = (0..40)
        .map(|f| (0..64).map(|i| (f * 64 + i) as f32 * 0.25).collect())
        .collect();
    let layer = line_layer_with_vertex_values(values.clone(), Some(values));
    let plain = encode_tile_with(std::slice::from_ref(&layer), &EncoderConfig::default()).unwrap();
    let quantized = encode_tile_with(
        std::slice::from_ref(&layer),
        &EncoderConfig {
            quantize_vertex_values: true,
            ..EncoderConfig::default()
        },
    )
    .unwrap();
    // 2 columns * 2560 values * 2 saved bytes = 10240; allow for the vq JSON
    // and the frame's 8-byte section padding.
    assert!(
        plain.len() > quantized.len() + 10_000,
        "expected ~10 KiB saved, got {} -> {}",
        plain.len(),
        quantized.len()
    );
}

/// Two encodes of the same layer produce the same bytes — the affine is a pure
/// function of the column, and `vq` is a BTreeMap, so nothing here reintroduces
/// the ordering non-determinism the content-addressed pack dedup depends on.
#[test]
fn vertex_value_quant_is_byte_reproducible() {
    let layer = line_layer_with_vertex_values(
        vec![vec![3.0f32, f32::NAN, -1.0], vec![9.5, 0.25]],
        Some(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0]]),
    );
    let cfg = EncoderConfig {
        quantize_vertex_values: true,
        ..EncoderConfig::default()
    };
    let a = encode_tile_with(std::slice::from_ref(&layer), &cfg).unwrap();
    let b = encode_tile_with(std::slice::from_ref(&layer), &cfg).unwrap();
    assert_eq!(a, b, "quantized encodes must be byte-reproducible");
}

/// A crafted or corrupt `vq` must be a loud decode error, never a silent
/// misread of raw indices as physical values.
#[test]
fn vertex_value_quant_rejects_inconsistent_tile_meta() {
    let layer = line_layer_with_vertex_values(vec![vec![1.0f32, 2.0]], None);
    let parts = encode_layer_v2_parts(&layer, &EncoderConfig::default()).unwrap();
    let core = core_batch_of(&parts);

    let vq = |name: &str| Some(BTreeMap::from([(name.to_string(), (0.0f64, 1.0f64))]));
    for (meta, want) in [
        (
            TileMeta {
                vq: vq("speed"),
                ..TileMeta::default()
            },
            "not a per-vertex value column",
        ),
        (
            TileMeta {
                vq: vq("vertex_value_matrix"),
                ..TileMeta::default()
            },
            "the layer has no such column",
        ),
        (
            TileMeta {
                vq: vq("vertex_value"),
                ..TileMeta::default()
            },
            "list leaf is Float32",
        ),
    ] {
        let err =
            merge_v2_layer(core.clone(), None, &meta).expect_err("an inconsistent vq must error");
        assert!(
            err.to_string().contains(want),
            "expected {want:?}, got: {err}"
        );
    }
}

// ----------------------------------------------------------------------------
// Multi-part polygon boundaries (the `part_offsets` column)
// ----------------------------------------------------------------------------

/// An axis-aligned closed square ring at `(x, y)` with side 1.
fn unit_ring(x: f64, y: f64) -> Vec<Coord> {
    vec![
        [x, y],
        [x + 1.0, y],
        [x + 1.0, y + 1.0],
        [x, y + 1.0],
        [x, y],
    ]
}

/// A polygon layer built straight from per-feature (rings, part_starts) pairs.
fn polygon_layer_with_parts(features: Vec<(Vec<Vec<Coord>>, Vec<u32>)>) -> ColumnarLayer {
    let n = features.len();
    let mut geometry = Vec::with_capacity(n);
    let mut parts = Vec::with_capacity(n);
    for (rings, starts) in features {
        geometry.push(rings);
        parts.push(starts);
    }
    ColumnarLayer {
        name: "polys".into(),
        feature_ids: (1..=n as u64).collect(),
        start_times: vec![0; n],
        end_times: vec![0; n],
        geometry: GeometryColumn::Polygon(geometry),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        polygon_parts: Some(parts),
        vertex_value_matrix: None,
        properties: vec![],
    }
}

/// Read a decoded `List<UInt32>` column back as one vec per feature.
fn u32_list_column(batch: &RecordBatch, name: &str) -> Vec<Vec<u32>> {
    let list = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column '{name}' is missing"))
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("List");
    assert_eq!(list.values().data_type(), &DataType::UInt32);
    (0..list.len())
        .map(|i| {
            list.value(i)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

/// The column is emitted ONLY when it carries information — some feature has
/// more than one part — and its absence is what tells a reader every feature is
/// single-part.
#[test]
fn part_offsets_appear_only_for_a_multi_part_layer() {
    let cfg = EncoderConfig::default();

    // A plain Polygon (one part, one hole) and a Polygon with two rings: no
    // column at all, so no already-published layer shape changes.
    let single = polygon_layer_with_parts(vec![
        (vec![unit_ring(0.0, 0.0)], vec![0]),
        (vec![unit_ring(0.0, 0.0), unit_ring(0.25, 0.25)], vec![0]),
    ]);
    let batch = &decode_tile(&encode_tile_with(std::slice::from_ref(&single), &cfg).unwrap())
        .unwrap()[0]
        .batch;
    assert!(
        batch.column_by_name("part_offsets").is_none(),
        "a single-part layer must not pay for the column"
    );

    // One two-part feature is enough to turn it on — and every feature then
    // gets an entry, including the single-part ones (`[0]`).
    let multi = polygon_layer_with_parts(vec![
        (vec![unit_ring(0.0, 0.0)], vec![0]),
        (
            vec![
                unit_ring(0.0, 0.0),
                unit_ring(0.25, 0.25), // hole of part 0
                unit_ring(5.0, 5.0),   // exterior of part 1
            ],
            vec![0, 2],
        ),
    ]);
    let batch = &decode_tile(&encode_tile_with(std::slice::from_ref(&multi), &cfg).unwrap())
        .unwrap()[0]
        .batch;
    assert_eq!(
        u32_list_column(batch, "part_offsets"),
        vec![vec![0], vec![0, 2]],
        "ring indices are relative to each feature's own first ring"
    );
    // Purely additive: it lands in the CORE batch, AFTER every column that
    // existed before it, so no positional decoder shifts.
    let schema = batch.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec!["id", "start_time", "end_time", "geometry", "part_offsets"],
        "part_offsets is appended last among the reserved columns"
    );
    assert!(
        !schema
            .field_with_name("part_offsets")
            .unwrap()
            .is_nullable(),
        "every feature gets an entry, so the column is non-null"
    );
}

/// The rows are stable-sorted by `start_time` at encode; the per-feature part
/// boundaries must travel with their feature (they are feature-local ring
/// indices, so they move verbatim — no rebasing).
#[test]
fn part_offsets_follow_the_encode_time_row_sort() {
    let mut layer = polygon_layer_with_parts(vec![
        (
            vec![unit_ring(0.0, 0.0), unit_ring(5.0, 5.0)],
            vec![0, 1], // two single-ring parts
        ),
        (vec![unit_ring(9.0, 9.0)], vec![0]),
    ]);
    // Feature 0 sorts AFTER feature 1.
    layer.start_times = vec![5_000, 1_000];
    layer.end_times = vec![5_000, 1_000];

    let batch = &decode_tile(
        &encode_tile_with(std::slice::from_ref(&layer), &EncoderConfig::default()).unwrap(),
    )
    .unwrap()[0]
        .batch;
    assert_eq!(
        u32_list_column(batch, "part_offsets"),
        vec![vec![0], vec![0, 1]],
        "the permutation must carry part boundaries alongside their geometry"
    );
    assert_eq!(int64_column(batch, "start_time"), vec![1_000, 5_000]);
}

/// Like `triangles`, the column is silently dropped for a non-polygon layer so
/// an over-eager builder cannot poison a point/line tile with it.
#[test]
fn part_offsets_are_dropped_for_non_polygon_layers() {
    let mut layer = point_layer_with_times(&[0, 1], &[0, 1]);
    layer.polygon_parts = Some(vec![vec![0, 1], vec![0]]);
    let batch = &decode_tile(
        &encode_tile_with(std::slice::from_ref(&layer), &EncoderConfig::default()).unwrap(),
    )
    .unwrap()[0]
        .batch;
    assert!(batch.column_by_name("part_offsets").is_none());
}

/// A malformed part list would silently mis-split a feature on every reader,
/// so `ColumnarLayer::validate` refuses it at the encode boundary.
#[test]
fn malformed_polygon_parts_are_rejected_at_encode() {
    let base = || {
        polygon_layer_with_parts(vec![(
            vec![unit_ring(0.0, 0.0), unit_ring(5.0, 5.0)],
            vec![0, 1],
        )])
    };
    let cfg = EncoderConfig::default();

    let mut wrong_len = base();
    wrong_len.polygon_parts = Some(vec![vec![0], vec![0]]);
    assert!(encode_tile_with(&[wrong_len], &cfg)
        .unwrap_err()
        .to_string()
        .contains("polygon_parts has 2 entries"));

    let mut not_zero = base();
    not_zero.polygon_parts = Some(vec![vec![1]]);
    assert!(encode_tile_with(&[not_zero], &cfg)
        .unwrap_err()
        .to_string()
        .contains("part 0 must start at ring 0"));

    let mut not_increasing = base();
    not_increasing.polygon_parts = Some(vec![vec![0, 0]]);
    assert!(encode_tile_with(&[not_increasing], &cfg)
        .unwrap_err()
        .to_string()
        .contains("not strictly increasing"));

    let mut past_the_end = base();
    past_the_end.polygon_parts = Some(vec![vec![0, 2]]);
    assert!(encode_tile_with(&[past_the_end], &cfg)
        .unwrap_err()
        .to_string()
        .contains("only has 2 ring(s)"));
}

/// An EMPTY polygon layer encodes and decodes with no part column and no
/// template fork — the same "degenerate layers must not cost a schema" rule the
/// compact-time forms follow.
#[test]
fn part_offsets_handle_an_empty_polygon_layer() {
    let empty = polygon_layer_with_parts(vec![]);
    let cfg = EncoderConfig::default();
    let batch = &decode_tile(&encode_tile_with(std::slice::from_ref(&empty), &cfg).unwrap())
        .unwrap()[0]
        .batch;
    assert_eq!(batch.num_rows(), 0);
    assert!(batch.column_by_name("part_offsets").is_none());
}

// ----------------------------------------------------------------------------
// TB-3 / TB-4 — the dataset-global categorical verdict and the dictionary hoist
// ----------------------------------------------------------------------------
//
// The defect both items repair is per-tile greed where a dataset-global pin is
// wanted: the incumbent encoder decides `Dictionary<UInt16, Utf8>` vs plain
// `Utf8` from the rows that happened to land in ONE tile, so a column is a
// dictionary in the dense tiles and a `Utf8` in the sparse ones. That forks a
// PROPS schema template per variant (§13.2's conformance-invariance rule broken
// by construction) and re-ships the category list in every single tile.
//
// TB-3 makes the verdict a function of `EncoderConfig::global_pins` — the
// dataset DOMAIN — so one column is one Arrow type everywhere. TB-4 then moves
// the now-provably-constant `DictionaryBatch` message into the schema TEMPLATE,
// where the category list is stored once per realized layer shape.

/// A `Dictionary` verdict over `categories`, in first-seen order.
fn dict_verdict(categories: &[&str]) -> GlobalDictVerdict {
    GlobalDictVerdict::Dictionary(Arc::new(
        categories.iter().map(|s| (*s).to_string()).collect(),
    ))
}

/// An [`EncoderConfig`] carrying exactly these categorical pins.
fn cfg_with_dict_pins(
    base: &EncoderConfig,
    entries: &[(&str, GlobalDictVerdict)],
) -> EncoderConfig {
    let mut pins = GlobalColumnPins::default();
    for (name, verdict) in entries {
        pins.dict.insert((*name).to_string(), verdict.clone());
    }
    EncoderConfig {
        global_pins: Some(Arc::new(pins)),
        ..base.clone()
    }
}

/// A point layer with one categorical column, sized by `values`.
///
/// Every feature shares `start_time == end_time == 0` so the CORE schema (and
/// therefore the CORE template) is constant across every tile these tests build
/// — which is what lets the PROPS template count be read directly off the
/// collector.
fn kind_layer(values: &[Option<&str>]) -> ColumnarLayer {
    kind_layer_named("kind", values)
}

fn kind_layer_named(column: &str, values: &[Option<&str>]) -> ColumnarLayer {
    let n = values.len();
    ColumnarLayer {
        polygon_parts: None,
        name: "kinds".into(),
        feature_ids: (0..n as u64).collect(),
        start_times: vec![0; n],
        end_times: vec![0; n],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7]; n]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            column.to_string(),
            PropertyColumn::Categorical(values.iter().map(|v| v.map(str::to_string)).collect()),
        )],
    }
}

/// A tile so sparse the per-tile surrogate always picks `Utf8` (3 rows: the
/// dictionary's fixed IPC allowance dwarfs 6 bytes of values).
fn sparse_kind_layer() -> ColumnarLayer {
    kind_layer(&[Some("car"), Some("bus"), None])
}

/// A tile so dense the per-tile surrogate always picks `Dictionary` (1 000 rows
/// over 2 categories).
fn dense_kind_layer() -> ColumnarLayer {
    let values: Vec<Option<&str>> = (0..1_000)
        .map(|i| Some(if i % 2 == 0 { "car" } else { "bus" }))
        .collect();
    kind_layer(&values)
}

/// Arrow IPC message headers of a byte run, in order. Tolerates a bare TAIL
/// (a message sequence that does not begin with a Schema), which is exactly what
/// the hoist has to be inspected on.
fn ipc_message_kinds(bytes: &[u8]) -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= bytes.len() {
        assert_eq!(
            &bytes[pos..pos + 4],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            "message at offset {pos} must start with the continuation marker"
        );
        let meta_len = i32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
        if meta_len <= 0 {
            out.push("EOS");
            break;
        }
        let header_end = pos + 8 + meta_len as usize;
        let msg = arrow::ipc::root_as_message(&bytes[pos + 8..header_end]).unwrap();
        out.push(match msg.header_type() {
            arrow::ipc::MessageHeader::Schema => "Schema",
            arrow::ipc::MessageHeader::DictionaryBatch => "DictionaryBatch",
            arrow::ipc::MessageHeader::RecordBatch => "RecordBatch",
            _ => "Other",
        });
        pos = header_end + msg.bodyLength() as usize;
    }
    out
}

/// The decoded categorical column's `(categories, per-row values)`.
fn decoded_categorical(batch: &RecordBatch, column: &str) -> (Vec<String>, Vec<Option<String>>) {
    let col = batch.column_by_name(column).expect("column present");
    match col.data_type() {
        DataType::Dictionary(..) => {
            let dict = col
                .as_any()
                .downcast_ref::<DictionaryArray<UInt16Type>>()
                .expect("Dictionary<UInt16, _>");
            let values = dict
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 dictionary values");
            let categories: Vec<String> =
                (0..values.len()).map(|i| values.value(i).into()).collect();
            let rows = (0..dict.len())
                .map(|i| {
                    dict.is_valid(i)
                        .then(|| categories[dict.keys().value(i) as usize].clone())
                })
                .collect();
            (categories, rows)
        }
        DataType::Utf8 => {
            let s = col
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 column");
            let rows: Vec<Option<String>> = (0..s.len())
                .map(|i| s.is_valid(i).then(|| s.value(i).to_string()))
                .collect();
            (Vec::new(), rows)
        }
        other => panic!("unexpected categorical storage type {other:?}"),
    }
}

// --- TB-3: one column, one Arrow type, everywhere --------------------------

/// The pinned verdict wins over the per-tile surrogate in BOTH directions: a
/// sparse tile whose own arithmetic says `Utf8` still ships a dictionary under a
/// `Dictionary` pin, and a dense tile whose own arithmetic says `Dictionary`
/// still ships plain strings under a `Utf8` pin.
///
/// This is the §13.2 invariance rule made executable — the Arrow type is a
/// function of the pin and of nothing the tile carries.
#[test]
fn pinned_dict_verdict_overrides_the_per_tile_surrogate_in_both_directions() {
    let base = EncoderConfig::default();
    let sparse = sparse_kind_layer();
    let dense = dense_kind_layer();

    // Baseline: the incumbent per-tile decision genuinely disagrees across the
    // two tiles. (If it did not, this test would prove nothing.)
    let field_type = |layer: &ColumnarLayer, cfg: &EncoderConfig| {
        let batch = decode_layer(&encode_layer_with(layer, cfg).unwrap()).unwrap();
        batch
            .schema()
            .field_with_name("kind")
            .unwrap()
            .data_type()
            .clone()
    };
    assert_eq!(field_type(&sparse, &base), DataType::Utf8);
    assert!(matches!(
        field_type(&dense, &base),
        DataType::Dictionary(..)
    ));

    // Pin Dictionary: the SPARSE tile flips to a dictionary.
    let pinned_dict = cfg_with_dict_pins(&base, &[("kind", dict_verdict(&["car", "bus"]))]);
    assert!(matches!(
        field_type(&sparse, &pinned_dict),
        DataType::Dictionary(..)
    ));
    assert!(matches!(
        field_type(&dense, &pinned_dict),
        DataType::Dictionary(..)
    ));

    // Pin Utf8: the DENSE tile flips to plain strings.
    let pinned_utf8 = cfg_with_dict_pins(&base, &[("kind", GlobalDictVerdict::Utf8)]);
    assert_eq!(field_type(&dense, &pinned_utf8), DataType::Utf8);
    assert_eq!(field_type(&sparse, &pinned_utf8), DataType::Utf8);
}

/// GUARD (the partial-hoist boundary from the other side): a column pinned
/// `Utf8` emits NO dictionary anywhere — not in a dense tile, not in the
/// template, not in the tail. A dictionary that exists in only some tiles is
/// precisely the fork TB-3 removes.
#[test]
fn pinned_utf8_never_emits_a_dictionary_anywhere() {
    let cfg = cfg_with_dict_pins(
        &EncoderConfig::default(),
        &[("kind", GlobalDictVerdict::Utf8)],
    );
    for layer in [
        sparse_kind_layer(),
        dense_kind_layer(),
        kind_layer(&[None; 4]),
    ] {
        // Standalone-layer shape.
        let ipc = encode_layer_with(&layer, &cfg).unwrap();
        assert!(
            !ipc_message_kinds(&ipc).contains(&"DictionaryBatch"),
            "a Utf8-pinned column must not emit a dictionary message"
        );
        // Frame shape: neither template nor tail may carry one.
        let enc = encode_layer_v2_parts(&layer, &cfg).unwrap();
        let (template, tail) = enc.props.expect("the layer has a property column");
        assert_eq!(ipc_message_kinds(&template), vec!["Schema"]);
        assert_eq!(ipc_message_kinds(&tail), vec!["RecordBatch", "EOS"]);
    }
}

/// Exact strings and NULLs survive identically on both pinned paths — the
/// verdict is a storage choice, never a data change.
#[test]
fn pinned_verdicts_preserve_exact_strings_and_nulls_on_both_paths() {
    let base = EncoderConfig::default();
    let values = [Some("car"), None, Some("bus"), Some("car"), None];
    let layer = kind_layer(&values);
    let expected: Vec<Option<String>> = values.iter().map(|v| v.map(str::to_string)).collect();

    for cfg in [
        cfg_with_dict_pins(&base, &[("kind", dict_verdict(&["car", "bus"]))]),
        cfg_with_dict_pins(&base, &[("kind", GlobalDictVerdict::Utf8)]),
        base.clone(),
    ] {
        let batch = decode_layer(&encode_layer_with(&layer, &cfg).unwrap()).unwrap();
        let (_, rows) = decoded_categorical(&batch, "kind");
        assert_eq!(rows, expected, "exact strings and nulls must round-trip");
    }
}

/// The template-fork this item exists to kill: WITHOUT a pin a dense tile and a
/// sparse tile of the same layer shape mint TWO PROPS templates; WITH one they
/// share a single template, so `manifest.schemas` stops growing an entry per
/// (dense-tile, sparse-tile) split.
#[test]
fn pinned_dict_verdict_collapses_the_props_template_fork() {
    let layers = [sparse_kind_layer(), dense_kind_layer()];

    let unpinned_collector = Arc::new(TemplateCollector::new());
    let unpinned = v2_hashed(&EncoderConfig::default(), &unpinned_collector);
    for layer in &layers {
        encode_tile_with(std::slice::from_ref(layer), &unpinned).unwrap();
    }
    assert_eq!(
        unpinned_collector.len(),
        3,
        "per-tile verdicts fork the PROPS template: 1 core + Utf8 props + Dictionary props"
    );

    let pinned_collector = Arc::new(TemplateCollector::new());
    let pinned = v2_hashed(
        &cfg_with_dict_pins(
            &EncoderConfig::default(),
            &[("kind", dict_verdict(&["car", "bus"]))],
        ),
        &pinned_collector,
    );
    let payloads: Vec<Vec<u8>> = layers
        .iter()
        .map(|layer| encode_tile_with(std::slice::from_ref(layer), &pinned).unwrap())
        .collect();
    assert_eq!(
        pinned_collector.len(),
        2,
        "one global verdict ⇒ exactly one CORE + one PROPS template"
    );

    // ...and both tiles still decode, through the registry those templates build.
    let registry = registry_from(&pinned_collector);
    for payload in &payloads {
        decode_tile_with_templates(payload, &registry).unwrap();
    }
}

// --- TB-4: the dictionary hoist -------------------------------------------

/// `split_ipc_after_dictionaries` walks 0, 1 and n `DictionaryBatch` messages
/// and lands exactly on the RecordBatch. With zero dictionaries it agrees with
/// `split_ipc_at_schema` (reached here through the frame path, which uses that
/// splitter whenever nothing is hoistable).
///
/// The exact counts are also the DELTA-DICTIONARY assertion the hoist depends
/// on: k dictionary columns must produce exactly k messages. arrow-rs writes
/// delta dictionaries only when asked and this writer never asks, but if that
/// default ever moved, a second (delta) message for a column would land on the
/// template side and make it tile-dependent — so the count is pinned here.
#[test]
fn split_ipc_after_dictionaries_walks_zero_one_and_many_dictionaries() {
    let base = EncoderConfig::default();

    // 0 dictionaries — a Utf8-pinned column.
    let none = encode_layer_with(
        &sparse_kind_layer(),
        &cfg_with_dict_pins(&base, &[("kind", GlobalDictVerdict::Utf8)]),
    )
    .unwrap();
    // 1 dictionary — one pinned categorical column.
    let one = encode_layer_with(
        &sparse_kind_layer(),
        &cfg_with_dict_pins(&base, &[("kind", dict_verdict(&["car", "bus"]))]),
    )
    .unwrap();
    // 2 dictionaries — two pinned categorical columns on one layer.
    let mut two_col = sparse_kind_layer();
    two_col.properties.push((
        "colour".into(),
        PropertyColumn::Categorical(vec![
            Some("red".into()),
            Some("blue".into()),
            Some("red".into()),
        ]),
    ));
    let two = encode_layer_with(
        &two_col,
        &cfg_with_dict_pins(
            &base,
            &[
                ("kind", dict_verdict(&["car", "bus"])),
                ("colour", dict_verdict(&["red", "blue"])),
            ],
        ),
    )
    .unwrap();

    for (ipc, dictionaries) in [(&none, 0usize), (&one, 1), (&two, 2)] {
        let boundary = split_ipc_after_dictionaries(ipc).unwrap();
        let mut expected = vec!["Schema"];
        expected.extend(std::iter::repeat_n("DictionaryBatch", dictionaries));
        assert_eq!(
            ipc_message_kinds(&ipc[..boundary]),
            expected,
            "template side for {dictionaries} dictionary message(s)"
        );
        assert_eq!(
            ipc_message_kinds(&ipc[boundary..]),
            vec!["RecordBatch", "EOS"],
            "tail side for {dictionaries} dictionary message(s)"
        );
    }
}

/// Malformed framing errors LOUDLY rather than cutting in the wrong place — a
/// mis-cut template splices into a stream arrow-rs decodes as EMPTY, which is
/// the one failure mode that must never be quiet. Mirrors (and reaches through
/// to) `split_ipc_at_schema`'s own error cases.
#[test]
fn split_ipc_after_dictionaries_rejects_malformed_framing() {
    let msg = |bytes: &[u8]| split_ipc_after_dictionaries(bytes).unwrap_err().to_string();

    // Shared with split_ipc_at_schema: no encapsulated message, and an
    // end-of-stream marker where a schema belongs.
    assert!(msg(&[]).contains("does not start with an encapsulated message"));
    assert!(msg(&[0u8; 16]).contains("does not start with an encapsulated message"));
    assert!(msg(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]).contains("end-of-stream marker"));

    let ipc = encode_layer_with(
        &sparse_kind_layer(),
        &cfg_with_dict_pins(
            &EncoderConfig::default(),
            &[("kind", dict_verdict(&["car", "bus"]))],
        ),
    )
    .unwrap();
    let schema_end = {
        let meta_len = i32::from_le_bytes(ipc[4..8].try_into().unwrap()) as usize;
        8 + meta_len
    };

    // Truncated right after the schema: nothing left to walk.
    assert!(msg(&ipc[..schema_end]).contains("no record batch"));
    assert!(msg(&ipc[..schema_end + 4]).contains("no record batch"));

    // A corrupted continuation marker at the dictionary boundary.
    let mut corrupt = ipc.clone();
    corrupt[schema_end] = 0x00;
    assert!(msg(&corrupt).contains("continuation marker"));

    // A dictionary header whose declared body runs off the end of the stream.
    let dict_meta_len =
        i32::from_le_bytes(ipc[schema_end + 4..schema_end + 8].try_into().unwrap()) as usize;
    assert!(msg(&ipc[..schema_end + 8 + dict_meta_len]).contains("body overruns the stream"));

    // A dictionary header whose declared metadata runs off the end.
    assert!(
        msg(&ipc[..schema_end + 8 + dict_meta_len - 1]).contains("metadata overruns the stream")
    );
}

/// The headline round-trip: with the hoist on, tiles carrying a STRICT SUBSET of
/// the pinned categories — and tiles carrying none at all — decode to their
/// exact strings through `decode_tile_with_templates`, and they all share ONE
/// PROPS template.
#[test]
fn hoisted_dictionary_round_trips_subsets_and_all_null_tiles() {
    let categories = ["car", "bus", "tram", "ferry", "bike"];
    let collector = Arc::new(TemplateCollector::new());
    let cfg = v2_hashed(
        &cfg_with_dict_pins(
            &EncoderConfig::default(),
            &[("kind", dict_verdict(&categories))],
        ),
        &collector,
    );

    let cases: Vec<Vec<Option<&str>>> = vec![
        vec![
            Some("car"),
            Some("bus"),
            Some("tram"),
            Some("ferry"),
            Some("bike"),
        ], // all
        vec![Some("ferry")],                    // one only
        vec![Some("bike"), None, Some("bike")], // subset + null
        vec![None, None],                       // all null
        vec![],                                 // empty tile
    ];
    let payloads: Vec<Vec<u8>> = cases
        .iter()
        .map(|values| encode_tile_with(&[kind_layer(values)], &cfg).unwrap())
        .collect();

    let registry = registry_from(&collector);
    for (values, payload) in cases.iter().zip(&payloads) {
        let decoded = decode_tile_with_templates(payload, &registry).unwrap();
        let (cats, rows) = decoded_categorical(&decoded[0].batch, "kind");
        assert_eq!(
            cats,
            categories.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "every tile ships the FULL pinned list, in first-seen order"
        );
        assert_eq!(
            rows,
            values
                .iter()
                .map(|v| v.map(str::to_string))
                .collect::<Vec<_>>(),
            "exact strings (and nulls) round-trip through the hoisted dictionary"
        );
    }

    // An EMPTY layer's compact-time forms differ from a populated one's, so it
    // legitimately mints its own CORE template; the PROPS template is shared by
    // all five.
    let props_templates: std::collections::BTreeSet<Vec<u8>> = cases
        .iter()
        .map(|values| {
            encode_layer_v2_parts(&kind_layer(values), &cfg)
                .unwrap()
                .props
                .expect("props present")
                .0
        })
        .collect();
    assert_eq!(
        props_templates.len(),
        1,
        "the hoisted PROPS template must be byte-identical across every tile"
    );
}

/// Where the bytes actually move: the `DictionaryBatch` message leaves the
/// per-tile TAIL and lands in the TEMPLATE, and the CORE stream — which has no
/// dictionary columns by construction — is untouched.
#[test]
fn hoist_moves_the_dictionary_message_from_tail_to_template() {
    let layer = dense_kind_layer();
    let base = EncoderConfig::default();

    // Incumbent: the dictionary rides the tail, once per tile.
    let unpinned = encode_layer_v2_parts(&layer, &base).unwrap();
    let (unpinned_template, unpinned_tail) = unpinned.props.expect("props present");
    assert_eq!(ipc_message_kinds(&unpinned_template), vec!["Schema"]);
    assert_eq!(
        ipc_message_kinds(&unpinned_tail),
        vec!["DictionaryBatch", "RecordBatch", "EOS"]
    );

    // Hoisted: the same message is template-resident, and the tail is keys-only.
    let pinned_cfg = cfg_with_dict_pins(&base, &[("kind", dict_verdict(&["car", "bus"]))]);
    let pinned = encode_layer_v2_parts(&layer, &pinned_cfg).unwrap();
    let (template, tail) = pinned.props.expect("props present");
    assert_eq!(
        ipc_message_kinds(&template),
        vec!["Schema", "DictionaryBatch"]
    );
    assert_eq!(ipc_message_kinds(&tail), vec!["RecordBatch", "EOS"]);
    assert!(
        template.len() > unpinned_template.len(),
        "the template absorbs the category list"
    );
    assert!(
        tail.len() < unpinned_tail.len(),
        "the per-tile tail sheds it ({} vs {} bytes)",
        tail.len(),
        unpinned_tail.len()
    );

    // CORE carries no dictionary either way — the core split is untouched.
    assert_eq!(ipc_message_kinds(&pinned.core_template), vec!["Schema"]);
    assert_eq!(pinned.core_template, unpinned.core_template);
}

/// ALL-OR-NOTHING, observed: a tile holding ONE of five categories still ships
/// all five. A tile-local subset would fork a template per subset, and the
/// encoder has no way to spell one (`PinnedCategories` has a single constructor,
/// whose only input is the pin).
#[test]
fn hoist_ships_the_full_pinned_list_from_a_single_category_tile() {
    let categories = ["car", "bus", "tram", "ferry", "bike"];
    let cfg = cfg_with_dict_pins(
        &EncoderConfig::default(),
        &[("kind", dict_verdict(&categories))],
    );
    let batch =
        decode_layer(&encode_layer_with(&kind_layer(&[Some("tram"), Some("tram")]), &cfg).unwrap())
            .unwrap();
    let (cats, rows) = decoded_categorical(&batch, "kind");
    assert_eq!(cats.len(), 5);
    assert_eq!(cats[0], "car", "first-seen order is the pin's order");
    assert_eq!(cats[4], "bike");
    assert_eq!(rows, vec![Some("tram".into()), Some("tram".into())]);
}

/// A tile-local dictionary must never block the hoist by *degrading the whole
/// stream to `Tail`* — because `Tail` puts EVERY dictionary in the tail, the
/// pinned one included, and the pinned one is the dataset's entire list.
///
/// This test used to pin the opposite contract, and the bytes it asserted were
/// the defect: `tail = [DictionaryBatch, DictionaryBatch, RecordBatch, EOS]`
/// with a 2 002-category pin measured **55 128 B per tile** against a 240-B
/// template. The stream is now made hoistable instead, by demoting the UNPINNED
/// column to `Utf8` (`unpinned_categoricals_take_utf8`): same layer, same pin,
/// **50 672 B of template stored once + 8 480 B per tile** (−84.6 % per tile).
/// The invariant the old test protected — no per-tile template — still holds,
/// and now holds without the per-tile list.
#[test]
fn an_unpinned_dictionary_column_does_not_drag_the_pinned_list_into_every_tile() {
    let mut layer = dense_kind_layer();
    // A second categorical column, dense enough that the per-tile surrogate
    // picks a dictionary for it, and deliberately left out of the pins.
    layer.properties.push((
        "colour".into(),
        PropertyColumn::Categorical(
            (0..1_000)
                .map(|i| Some(if i % 3 == 0 { "red" } else { "blue" }.to_string()))
                .collect(),
        ),
    ));
    let cfg = cfg_with_dict_pins(
        &EncoderConfig::default(),
        &[("kind", dict_verdict(&["car", "bus"]))],
    );

    let enc = encode_layer_v2_parts(&layer, &cfg).unwrap();
    let (template, tail) = enc.props.expect("props present");
    assert_eq!(
        ipc_message_kinds(&template),
        vec!["Schema", "DictionaryBatch"],
        "the pinned dictionary is dataset-constant and belongs in the template"
    );
    assert_eq!(
        ipc_message_kinds(&tail),
        vec!["RecordBatch", "EOS"],
        "no dictionary may ride the per-tile tail once a pin is in play"
    );

    // The pinned column still gets its GLOBAL type and list (TB-3 holds
    // independently of TB-4), and the tile decodes.
    let batch = decode_layer(&encode_layer_with(&layer, &cfg).unwrap()).unwrap();
    let (cats, rows) = decoded_categorical(&batch, "kind");
    assert_eq!(cats, vec!["car".to_string(), "bus".to_string()]);
    assert_eq!(rows[0], Some("car".to_string()));
    // The demoted sibling keeps its exact strings.
    let (_, colours) = decoded_categorical(&batch, "colour");
    assert_eq!(colours[0], Some("red".to_string()));
    assert_eq!(colours[1], Some("blue".to_string()));
}

/// Data outside the pin is a HARD ERROR, never a silent degrade: it means pass 1
/// and the encoded feature stream disagree about the column's domain, and every
/// guarantee downstream (one type everywhere, a constant dictionary message, a
/// hoistable template) rests on them agreeing.
#[test]
fn pinned_dictionary_rejects_a_value_absent_from_the_pin() {
    let cfg = cfg_with_dict_pins(
        &EncoderConfig::default(),
        &[("kind", dict_verdict(&["car"]))],
    );
    let err = encode_layer_with(&kind_layer(&[Some("car"), Some("bus")]), &cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("absent"), "{err}");
    assert!(err.contains("\"bus\""), "{err}");
    assert!(
        err.contains("--single-pass"),
        "the escape hatch is named: {err}"
    );
}

/// READER CONTRACT: a pinned list that could mint the key `0xffff` degrades to
/// `Utf8` for the WHOLE dataset.
///
/// The TS reader spells "this row has no category" as the in-band sentinel
/// `0xffff` in its `Uint16Array` of indices, so a live key of 65 535 would be
/// indistinguishable from a null. The per-tile builder can never produce one
/// (`build_dictionary_indices` stops at `u16::MAX` categories) but pass 1's cap
/// is `MAX_CATEGORIES = 65 536`, one wider — so the pin can, and must not.
#[test]
fn an_oversize_pin_degrades_to_utf8_for_the_whole_dataset() {
    let base = EncoderConfig::default();
    let at_limit: Vec<String> = (0..u16::MAX as usize).map(|i| format!("c{i}")).collect();
    let over_limit: Vec<String> = (0..u16::MAX as usize + 1)
        .map(|i| format!("c{i}"))
        .collect();
    let layer = kind_layer(&[Some("c0"), Some("c1")]);

    let cfg_at = EncoderConfig {
        global_pins: Some(Arc::new(GlobalColumnPins {
            attr: Default::default(),
            dict: [(
                "kind".to_string(),
                GlobalDictVerdict::Dictionary(Arc::new(at_limit)),
            )]
            .into_iter()
            .collect(),
        })),
        ..base.clone()
    };
    let cfg_over = EncoderConfig {
        global_pins: Some(Arc::new(GlobalColumnPins {
            attr: Default::default(),
            dict: [(
                "kind".to_string(),
                GlobalDictVerdict::Dictionary(Arc::new(over_limit)),
            )]
            .into_iter()
            .collect(),
        })),
        ..base.clone()
    };

    // Exactly `u16::MAX` categories ⇒ keys 0..=65_534, still below the sentinel.
    let ipc_at = encode_layer_with(&layer, &cfg_at).unwrap();
    assert!(matches!(
        decode_layer(&ipc_at)
            .unwrap()
            .schema()
            .field_with_name("kind")
            .unwrap()
            .data_type(),
        DataType::Dictionary(..)
    ));

    // One more ⇒ Utf8 everywhere, and no dictionary message anywhere.
    let ipc_over = encode_layer_with(&layer, &cfg_over).unwrap();
    let batch = decode_layer(&ipc_over).unwrap();
    assert_eq!(
        batch.schema().field_with_name("kind").unwrap().data_type(),
        &DataType::Utf8
    );
    assert!(!ipc_message_kinds(&ipc_over).contains(&"DictionaryBatch"));
    let (_, rows) = decoded_categorical(&batch, "kind");
    assert_eq!(rows, vec![Some("c0".to_string()), Some("c1".to_string())]);
}

/// DETERMINISM (the mandatory byte-identical re-run): the pinned + hoisted
/// encode is a pure function of `(layer, pins)`. Two runs in one process, and
/// two independently-constructed but equal pin structs, produce identical bytes
/// — the property content-addressed pack dedup rests on.
#[test]
fn pinned_encode_is_byte_identical_across_runs() {
    let layers = [dense_kind_layer(), sparse_kind_layer()];
    let make_cfg = || {
        cfg_with_dict_pins(
            &EncoderConfig {
                quantize_attrs_auto: true,
                ..EncoderConfig::default()
            },
            &[("kind", dict_verdict(&["car", "bus"]))],
        )
    };
    for layer in &layers {
        let a = encode_tile_with(std::slice::from_ref(layer), &make_cfg()).unwrap();
        let b = encode_tile_with(std::slice::from_ref(layer), &make_cfg()).unwrap();
        let c = encode_tile_with(std::slice::from_ref(layer), &make_cfg()).unwrap();
        assert_eq!(a, b, "two encodes under equal pins must be byte-identical");
        assert_eq!(b, c);

        // ...and so is the template/tail split the frame is assembled from.
        let x = encode_layer_v2_parts(layer, &make_cfg()).unwrap();
        let y = encode_layer_v2_parts(layer, &make_cfg()).unwrap();
        assert_eq!(x.props, y.props);
        assert_eq!(x.core_template, y.core_template);
        assert_eq!(x.core_tail, y.core_tail);
        assert_eq!(x.tile_meta_json, y.tile_meta_json);
    }
}

/// THE FALLBACK IS BYTE-NEUTRAL: an encode with no pins — and an encode whose
/// pins name only OTHER columns — takes the incumbent per-tile path and emits
/// exactly the pre-M2 bytes. This is what `--single-pass` restores and what
/// keeps every un-pinned caller (`stt-serve` without a sidecar, a one-shot
/// external caller) unaffected.
#[test]
fn an_encode_with_no_applicable_pins_is_byte_identical_to_the_incumbent() {
    let base = EncoderConfig::default();
    let unrelated = cfg_with_dict_pins(&base, &[("some_other_column", dict_verdict(&["a", "b"]))]);
    let empty_pins = EncoderConfig {
        global_pins: Some(Arc::new(GlobalColumnPins::default())),
        ..base.clone()
    };
    for layer in [sparse_kind_layer(), dense_kind_layer()] {
        let incumbent = encode_tile_with(std::slice::from_ref(&layer), &base).unwrap();
        assert_eq!(
            encode_tile_with(std::slice::from_ref(&layer), &unrelated).unwrap(),
            incumbent,
            "a pin for a different column must not move a byte"
        );
        assert_eq!(
            encode_tile_with(std::slice::from_ref(&layer), &empty_pins).unwrap(),
            incumbent,
            "an empty pin set must not move a byte"
        );
    }
}

/// The hoisted form decodes identically through BOTH template modes (inline
/// schema sections and hash references) and against the caller's own baseline —
/// the same decode-equivalence contract every other encoding owes.
#[test]
fn hoisted_dictionary_decodes_equivalently_in_every_template_mode() {
    let cfg = cfg_with_dict_pins(
        &EncoderConfig::default(),
        &[
            ("kind", dict_verdict(&["car", "bus", "tram"])),
            ("colour", dict_verdict(&["red", "blue"])),
        ],
    );
    let mut two_dicts = kind_layer(&[Some("tram"), None, Some("car")]);
    two_dicts.properties.push((
        "colour".into(),
        PropertyColumn::Categorical(vec![Some("blue".into()), Some("red".into()), None]),
    ));
    assert_v2_decodes_like_v1(&[two_dicts], &cfg, "hoisted dictionaries");
    assert_v2_decodes_like_v1(
        &[kind_layer(&[None, None])],
        &cfg,
        "hoisted dictionary, all-null tile",
    );
}

// ---------------------------------------------------------------------------
// The MIXED stream: one pinned column beside an unpinned one
// ---------------------------------------------------------------------------
//
// The hoist is a whole-stream verdict, so before `unpinned_categoricals_take_utf8`
// a layer holding one PINNED categorical and one UNPINNED categorical whose
// per-tile surrogate chose a dictionary fell to `DictHoist::Tail` — which puts
// EVERY dictionary in the tail, the pinned one included. The dataset-global
// list then shipped in every tile, un-deduplicated: worse than hoisting AND
// worse than not pinning. The likeliest home for the shape is a summary/LOD
// tier, where derived columns are unpinned by design.

/// A dense layer with a pinned `kind` column and an unpinned `derived` column
/// whose 1 000 rows over 2 categories make the per-tile surrogate choose a
/// dictionary — the exact combination that used to fall to `Tail`.
fn mixed_pinned_and_unpinned_layer(pinned_categories: usize) -> ColumnarLayer {
    let mut layer = dense_kind_layer();
    let n = layer.feature_ids.len();
    layer.properties.push((
        "derived".into(),
        PropertyColumn::Categorical(
            (0..n)
                .map(|i| {
                    Some(if i % 2 == 0 {
                        "lo".to_string()
                    } else {
                        "hi".to_string()
                    })
                })
                .collect(),
        ),
    ));
    // Widen `kind`'s PIN (not its values) so the hoisted list is big enough for
    // the byte difference between the two splits to be unmissable.
    let _ = pinned_categories;
    layer
}

/// A wide pin for `kind`: the two categories the rows actually use plus a long
/// tail the tile never references — a dataset-global list, which is exactly
/// what a pin is.
fn wide_kind_pin(extra: usize) -> GlobalDictVerdict {
    let mut categories = vec!["car".to_string(), "bus".to_string()];
    categories.extend((0..extra).map(|i| format!("filler-category-{i:05}")));
    GlobalDictVerdict::Dictionary(Arc::new(categories))
}

/// THE REGRESSION THIS FIXES, stated in bytes: with a sibling unpinned
/// categorical present, the pinned column's whole global list must NOT end up
/// in the per-tile tail.
#[test]
fn a_mixed_pinned_unpinned_stream_never_ships_the_global_list_in_the_tail() {
    let layer = mixed_pinned_and_unpinned_layer(0);
    let cfg = cfg_with_dict_pins(&EncoderConfig::default(), &[("kind", wide_kind_pin(2_000))]);
    let parts = encode_layer_v2_parts(&layer, &cfg).unwrap();
    let (template, tail) = parts.props.as_ref().expect("props present");

    // The hoist survived the mixed stream: dictionaries are template-resident.
    assert_eq!(
        ipc_message_kinds(template),
        vec!["Schema", "DictionaryBatch"],
        "the pinned dictionary must live in the TEMPLATE, stored once per layer shape"
    );
    assert_eq!(
        ipc_message_kinds(tail),
        vec!["RecordBatch", "EOS"],
        "the per-tile tail must carry no dictionary at all"
    );

    // The 2 002-category pin is ~30 KB. If it were in the tail it would be in
    // every tile; the point of the fix is that the tail is small and constant.
    assert!(
        template.len() > 20_000,
        "sanity: the wide pin should dominate the template ({} B)",
        template.len()
    );
    assert!(
        tail.len() < template.len() / 4,
        "the per-tile tail ({} B) must not carry the global list ({} B template)",
        tail.len(),
        template.len()
    );
}

/// ...and the sibling column is the one that gives way: an UNPINNED categorical
/// in a stream that has a pin takes `Utf8`, which also removes its per-tile
/// dictionary-vs-`Utf8` flip — the very template fork TB-3 exists to kill.
#[test]
fn an_unpinned_categorical_beside_a_pinned_one_takes_utf8_in_every_tile() {
    let cfg = cfg_with_dict_pins(
        &EncoderConfig::default(),
        &[("kind", dict_verdict(&["car", "bus"]))],
    );

    // Dense: the per-tile surrogate would have chosen Dictionary for `derived`.
    let dense = mixed_pinned_and_unpinned_layer(0);
    let dense_batch = decode_layer(&encode_layer_with(&dense, &cfg).unwrap()).unwrap();
    assert_eq!(
        dense_batch
            .schema()
            .field_with_name("derived")
            .unwrap()
            .data_type(),
        &DataType::Utf8,
        "an unpinned sibling of a pinned column must not mint a tile-local dictionary"
    );
    assert!(matches!(
        dense_batch
            .schema()
            .field_with_name("kind")
            .unwrap()
            .data_type(),
        DataType::Dictionary(..)
    ));

    // Sparse: the surrogate would have chosen Utf8 anyway. Same type ⇒ ONE
    // PROPS schema across both densities, which is the fork collapse.
    let mut sparse = kind_layer(&[Some("car"), Some("bus"), None]);
    sparse.properties.push((
        "derived".into(),
        PropertyColumn::Categorical(vec![Some("lo".into()), Some("hi".into()), None]),
    ));
    let sparse_batch = decode_layer(&encode_layer_with(&sparse, &cfg).unwrap()).unwrap();
    assert_eq!(
        dense_batch.schema().fields(),
        sparse_batch.schema().fields(),
        "dense and sparse tiles of a mixed layer must share one PROPS schema"
    );

    // Exact strings survive on both sides.
    let (_, rows) = decoded_categorical(&sparse_batch, "derived");
    assert_eq!(
        rows,
        vec![Some("lo".to_string()), Some("hi".to_string()), None]
    );
}

/// The demotion is scoped to streams that actually carry a pin. With NO pins
/// (the `--single-pass` fallback, or any external one-shot caller) the
/// incumbent per-tile path is untouched, byte for byte.
#[test]
fn the_unpinned_demotion_does_not_fire_without_a_pin() {
    let layer = mixed_pinned_and_unpinned_layer(0);
    let base = EncoderConfig::default();
    let batch = decode_layer(&encode_layer_with(&layer, &base).unwrap()).unwrap();
    for column in ["kind", "derived"] {
        assert!(
            matches!(
                batch.schema().field_with_name(column).unwrap().data_type(),
                DataType::Dictionary(..)
            ),
            "{column}: with no pins the dense per-tile surrogate must still choose Dictionary"
        );
    }
    // A pin naming only an UNRELATED column is also not a pin on this stream.
    let unrelated = cfg_with_dict_pins(&base, &[("elsewhere", dict_verdict(&["a", "b"]))]);
    assert_eq!(
        encode_layer_with(&layer, &unrelated).unwrap(),
        encode_layer_with(&layer, &base).unwrap(),
        "a pin for a column this layer does not have must not move a byte"
    );
}

/// The dataset-scale surrogate is the DATASET sibling of the per-tile one, and
/// its whole point is that the per-tile IPC allowance is charged ONCE (the
/// hoisted message serves the entire dataset) instead of per tile. A sparse
/// per-tile sample and the dataset it belongs to can therefore disagree — which
/// is exactly the D5 defect being repaired, not a bug.
#[test]
fn the_dataset_surrogate_amortizes_what_the_per_tile_one_repeats() {
    // One 3-row tile of a 2-category column: the per-tile comparison says Utf8.
    let values = [Some("car".to_string()), Some("bus".to_string()), None];
    let categories = ["car".to_string(), "bus".to_string()];
    assert!(!categorical_dictionary_is_smaller(&values, &categories));

    // The same column across a million rows: the dictionary wins comfortably.
    assert!(dataset_dictionary_is_smaller(
        3_000_000, // total value bytes
        1_000_000, // rows
        6,         // category bytes
        2,         // distinct categories
    ));

    // A near-unique column is not helped by a dictionary at either scale.
    assert!(!dataset_dictionary_is_smaller(
        1_000_000, 100_000, 1_000_000, 100_000
    ));

    // Saturating arithmetic: absurd inputs must not panic.
    assert!(!dataset_dictionary_is_smaller(
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX
    ));
}

// ------------------------------------------------------------------
// TB-11 — bucket-proportional vertex-time ceiling (§2.5)
// ------------------------------------------------------------------

/// A coarse tier scales the ceiling proportionally; the base tier does not.
#[test]
fn vertex_time_ceiling_scales_with_the_tier_bucket() {
    let cfg = EncoderConfig {
        vertex_time_max_step_ms: 1_000,
        ..EncoderConfig::default()
    };
    let hour = 3_600_000u64;

    // Base tier — unchanged, so the default path stays byte-identical.
    assert_eq!(cfg.vertex_time_step_for_bucket(None, hour), 1_000);
    assert_eq!(cfg.vertex_time_step_for_bucket(Some(hour), hour), 1_000);

    // A 24x coarser tier tolerates a 24x coarser step at equal perceptual error.
    assert_eq!(
        cfg.vertex_time_step_for_bucket(Some(24 * hour), hour),
        24_000
    );
    // 30-day tier over an hourly base.
    assert_eq!(
        cfg.vertex_time_step_for_bucket(Some(720 * hour), hour),
        720_000
    );
}

/// Degenerate inputs never scale and never panic — an unknown base bucket must
/// leave the incumbent ceiling exactly in place.
#[test]
fn an_unknown_or_finer_bucket_leaves_the_ceiling_alone() {
    let cfg = EncoderConfig {
        vertex_time_max_step_ms: 1_000,
        ..EncoderConfig::default()
    };
    assert_eq!(cfg.vertex_time_step_for_bucket(Some(3_600_000), 0), 1_000);
    assert_eq!(cfg.vertex_time_step_for_bucket(None, 0), 1_000);
    // A tier FINER than base is not a tier; leave it alone.
    assert_eq!(
        cfg.vertex_time_step_for_bucket(Some(1_000), 3_600_000),
        1_000
    );
    // Saturating rather than overflowing.
    let big = EncoderConfig {
        vertex_time_max_step_ms: u32::MAX,
        ..EncoderConfig::default()
    };
    assert_eq!(big.vertex_time_step_for_bucket(Some(u64::MAX), 1), u32::MAX);
}

/// The base tier BORROWS the config — no clone, no divergence — so the common
/// path allocates nothing and encodes identically.
#[test]
fn the_base_tier_borrows_the_config_unchanged() {
    let cfg = EncoderConfig {
        vertex_time_max_step_ms: 1_000,
        ..EncoderConfig::default()
    };
    let hour = 3_600_000u64;
    let base = cfg.for_temporal_tier(None, hour);
    assert!(matches!(base, std::borrow::Cow::Borrowed(_)));
    assert_eq!(base.vertex_time_max_step_ms, 1_000);

    let tier = cfg.for_temporal_tier(Some(24 * hour), hour);
    assert!(matches!(tier, std::borrow::Cow::Owned(_)));
    assert_eq!(tier.vertex_time_max_step_ms, 24_000);
    // And nothing else about the config moved.
    assert_eq!(tier.quantize_attrs_auto, cfg.quantize_attrs_auto);
    assert_eq!(tier.compact_times, cfg.compact_times);
}

// ── TB-11 extension 2 — the feature-anchored vertex-time tier ───────────────

/// A trip-shaped layer: a WIDE layer span (trips spread over `layer_span_ms`)
/// with a NARROW per-feature span (each trip lasts `trip_ms`). This is the shape
/// the layer-anchored tiers price badly and the feature anchor rescues.
fn trip_shaped_layer(features: usize, layer_span_ms: i64, trip_ms: i64) -> ColumnarLayer {
    let starts: Vec<i64> = (0..features as i64)
        .map(|i| i * (layer_span_ms / features.max(1) as i64))
        .collect();
    let vertex_times: Vec<Vec<i64>> = starts
        .iter()
        .map(|&s| vec![s, s + trip_ms / 2, s + trip_ms])
        .collect();
    ColumnarLayer {
        polygon_parts: None,
        name: "trips".into(),
        feature_ids: (1..=features as u64).collect(),
        start_times: starts.clone(),
        end_times: starts.iter().map(|s| s + trip_ms).collect(),
        geometry: GeometryColumn::LineString(
            (0..features)
                .map(|i| {
                    let x = i as f64 * 0.001;
                    vec![[x, 0.0], [x + 0.0005, 0.5], [x + 0.001, 1.0]]
                })
                .collect(),
        ),
        vertex_times: Some(vertex_times),
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

/// The feature-anchored step from the decoded schema metadata.
fn vertex_time_feature_step_of(batch: &RecordBatch) -> Option<u32> {
    batch
        .schema()
        .metadata()
        .get("stt:vertex_time_feature_step_ms")
        .map(|v| v.parse().unwrap())
}

#[test]
fn a_trip_shaped_layer_takes_the_feature_anchored_tier_at_exact_precision() {
    // Layer span 30 days — u16 at step 1 covers only 65.5 s, so the
    // layer-anchored u16 tier needs step 39_582, far past the 1000 ms ceiling.
    // Per-trip span 60 s, which fits u16 at step 1 when anchored per feature.
    let layer = trip_shaped_layer(64, 30 * 86_400_000, 60_000);
    let batch = decode_layer(&encode_layer(&layer).unwrap()).unwrap();

    assert_eq!(
        vertex_time_encoding_of(&batch),
        None,
        "the layer-anchored form must not be declared alongside the feature-anchored one"
    );
    assert_eq!(
        vertex_time_feature_step_of(&batch),
        Some(1),
        "a 60-second per-trip span fits u16 at EXACT millisecond precision"
    );

    // Round-trip: every vertex time is recovered exactly.
    let vt = batch
        .column_by_name("vertex_time")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let expected = layer.vertex_times.as_ref().unwrap();
    for (i, want) in expected.iter().enumerate() {
        let row = vt.value(i);
        let deltas = row
            .as_any()
            .downcast_ref::<UInt16Array>()
            .expect("the feature-anchored tier is UInt16");
        let anchor = layer.start_times[i];
        let got: Vec<i64> = deltas.values().iter().map(|&d| anchor + d as i64).collect();
        assert_eq!(&got, want, "feature {i} did not round-trip");
    }
}

#[test]
fn the_feature_anchor_is_not_used_when_the_layer_anchor_already_fits() {
    // A narrow layer span: the layer-anchored u16 tier wins outright, and
    // claiming the feature anchor would owe a capability for nothing.
    let layer = trip_shaped_layer(8, 30_000, 5_000);
    let batch = decode_layer(&encode_layer(&layer).unwrap()).unwrap();
    assert!(vertex_time_encoding_of(&batch).is_some());
    assert_eq!(vertex_time_feature_step_of(&batch), None);
}

#[test]
fn the_feature_anchor_declines_rather_than_wrapping_a_pre_start_vertex() {
    // A vertex BEFORE its feature's own start_time cannot be an UNSIGNED delta
    // from it. The tier must decline — never wrap into a wildly wrong instant —
    // and the scan then continues to the next tier as if the form did not exist.
    let mut layer = trip_shaped_layer(4, 30 * 86_400_000, 60_000);
    // Same layer, unmutated, DOES take the feature anchor: that is what makes
    // this test about the pre-start vertex and not about the layer's shape.
    let clean = decode_layer(&encode_layer(&layer).unwrap()).unwrap();
    assert_eq!(vertex_time_feature_step_of(&clean), Some(1));

    layer.vertex_times.as_mut().unwrap()[2][0] = layer.start_times[2] - 1;
    let batch = decode_layer(&encode_layer(&layer).unwrap()).unwrap();
    assert_eq!(
        vertex_time_feature_step_of(&batch),
        None,
        "one un-anchorable vertex must disqualify the form for the whole layer"
    );
    // The scan falls through to the layer-anchored u32 tier, which at a 30-day
    // span is exact (step 1) — wider bytes, no precision lost.
    // Origin 0: feature 0 still starts the layer, and the mutated vertex sits
    // 15 days in, so the layer minimum is unmoved.
    assert_eq!(vertex_time_encoding_of(&batch), Some((0, 1)));
    let vt = batch
        .column_by_name("vertex_time")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let row = vt.value(2);
    let deltas = row.as_any().downcast_ref::<UInt32Array>().unwrap();
    assert_eq!(
        deltas.value(0) as i64,
        layer.start_times[2] - 1,
        "the pre-start vertex must still round-trip exactly"
    );
}

#[test]
fn the_feature_anchored_form_is_observed_for_the_capability() {
    let layer = trip_shaped_layer(64, 30 * 86_400_000, 60_000);
    let cfg = EncoderConfig::default();
    let (_, observed) = crate::arrow_tile::encode_tile_observed(&[layer], &cfg).unwrap();
    assert!(
        observed.feature_anchored_vertex_times,
        "the writer learns it owes vertex-time-feature-anchor only by encoding"
    );

    // ...and a layer that does not use the form observes nothing, so no
    // capability is owed and no reader is locked out gratuitously.
    let narrow = trip_shaped_layer(8, 30_000, 5_000);
    let (_, observed) = crate::arrow_tile::encode_tile_observed(&[narrow], &cfg).unwrap();
    assert!(!observed.feature_anchored_vertex_times);
}

// ─── vis.gl temporal field metadata ─────────────────────────────────────────

/// The `visgl:temporal-*` descriptor on one field of a decoded batch, as
/// `(kind, unit, timezone, origin, policy)`.
fn temporal_meta(
    batch: &RecordBatch,
    name: &str,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    let schema = batch.schema();
    let f = schema
        .field_with_name(name)
        .unwrap_or_else(|_| panic!("no `{name}` column in the decoded batch"));
    let g = |k: &str| f.metadata().get(k).cloned();
    (
        g(TEMPORAL_KIND_KEY),
        g(TEMPORAL_UNIT_KEY),
        g(TEMPORAL_TIMEZONE_KEY),
        g(TEMPORAL_ORIGIN_KEY),
        g(TEMPORAL_ORIGIN_POLICY_KEY),
    )
}

/// THE CONTRACT the descriptor buys: whichever compact form a tile's time
/// columns took on the wire, the batch a reader is handed self-describes as
/// ABSOLUTE UTC epoch milliseconds. Nothing in `Int64` says that on its own,
/// and a consumer that has never heard of STT (luma.gl's `@luma.gl/arrow`,
/// lonboard, geoarrow-rs) has no other way to learn it.
#[test]
fn decoded_time_columns_self_describe_as_absolute_utc_ms() {
    let layer = sample_line_layer();
    let cfg = EncoderConfig::default(); // compact_times: true
    let payload = encode_tile_with(std::slice::from_ref(&layer), &cfg).unwrap();
    let decoded = decode_tile(&payload).unwrap();
    let batch = &decoded[0].batch;

    // The columns really are compact on the wire — otherwise this test would
    // pass without ever exercising the re-inflation completion.
    assert_eq!(
        batch
            .schema()
            .field_with_name("start_time")
            .unwrap()
            .data_type(),
        &DataType::Int64
    );

    for column in ["start_time", "end_time"] {
        let (kind, unit, tz, origin, policy) = temporal_meta(batch, column);
        assert_eq!(kind.as_deref(), Some("timestamp"), "{column}: kind");
        assert_eq!(unit.as_deref(), Some("millisecond"), "{column}: unit");
        assert_eq!(tz.as_deref(), Some("UTC"), "{column}: timezone");
        assert_eq!(origin.as_deref(), Some("0"), "{column}: origin");
        assert_eq!(policy.as_deref(), Some("zero"), "{column}: policy");
    }

    // …and the values are untouched by any of it.
    assert_eq!(int64_column(batch, "start_time"), layer.start_times);
    assert_eq!(int64_column(batch, "end_time"), layer.end_times);
}

/// The `et="zero"` form omits `end_time` entirely and the decoder SYNTHESIZES
/// it. That column is built from nothing, so it is the one place a descriptor
/// can go missing without any encoder change looking wrong.
#[test]
fn a_synthesized_end_time_column_carries_the_descriptor_too() {
    let mut layer = sample_line_layer();
    layer.end_times = layer.start_times.clone(); // end == start → EndTimeForm::Zero
    let payload =
        encode_tile_with(std::slice::from_ref(&layer), &EncoderConfig::default()).unwrap();
    let decoded = decode_tile(&payload).unwrap();
    let batch = &decoded[0].batch;

    let (kind, unit, tz, origin, policy) = temporal_meta(batch, "end_time");
    assert_eq!(kind.as_deref(), Some("timestamp"));
    assert_eq!(unit.as_deref(), Some("millisecond"));
    assert_eq!(tz.as_deref(), Some("UTC"));
    assert_eq!(origin.as_deref(), Some("0"));
    assert_eq!(policy.as_deref(), Some("zero"));
    assert_eq!(int64_column(batch, "end_time"), layer.start_times);
}

/// `vertex_time` is the one time column decode does NOT re-inflate — the
/// `List<UInt16>` deltas stay deltas, anchored per-TILE by
/// `stt:vertex_time_origin_ms`. So it advertises the logical domain and stays
/// silent about the origin: claiming `origin = 0` there would place every
/// vertex of every tile at the wrong instant.
#[test]
fn delta_encoded_vertex_times_advertise_the_domain_but_not_an_origin() {
    let layer = sample_line_layer();
    let payload =
        encode_tile_with(std::slice::from_ref(&layer), &EncoderConfig::default()).unwrap();
    let decoded = decode_tile(&payload).unwrap();
    let batch = &decoded[0].batch;

    let schema = batch.schema();
    let vt = schema.field_with_name("vertex_time").unwrap();
    assert!(
        matches!(vt.data_type(), DataType::List(_)),
        "expected the delta list form, got {:?}",
        vt.data_type()
    );

    let (kind, unit, tz, origin, policy) = temporal_meta(batch, "vertex_time");
    assert_eq!(kind.as_deref(), Some("timestamp"));
    assert_eq!(unit.as_deref(), Some("millisecond"));
    assert_eq!(tz.as_deref(), Some("UTC"));
    assert_eq!(
        origin, None,
        "a per-tile anchor must never be advertised as a column origin"
    );
    assert_eq!(policy, None, "policy without an origin is noise");

    // The anchor a reader MUST use instead is present and unambiguous.
    assert!(schema.metadata().contains_key("stt:vertex_time_origin_ms"));
}

/// The descriptor must never reach the WIRE. It is derivable from the decoded
/// column, so paying ~450 B of it per tile — which a self-contained frame
/// (`stt-serve`'s inline-schema shape) would do — buys nothing and costs 2-6%
/// of a small tile. This is also what keeps every already-published archive
/// address stable: the encoder is untouched.
#[test]
fn the_temporal_descriptor_is_never_encoded() {
    let layer = sample_line_layer();
    let payload =
        encode_tile_with(std::slice::from_ref(&layer), &EncoderConfig::default()).unwrap();
    let haystack = String::from_utf8_lossy(&payload);
    for key in [
        TEMPORAL_KIND_KEY,
        TEMPORAL_UNIT_KEY,
        TEMPORAL_ORIGIN_KEY,
        TEMPORAL_ORIGIN_POLICY_KEY,
        TEMPORAL_TIMEZONE_KEY,
    ] {
        assert!(
            !haystack.contains(key),
            "`{key}` reached the encoded tile; it is a decode-side descriptor"
        );
    }
}
