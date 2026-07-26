//! Unit tests for the tile payload format.
//!
//! Kept as one module (the pre-split `arrow_tile::tests`) because the cases
//! deliberately cross the encode/decode boundary.

use super::*;
// The pre-split `mod tests` inherited these from the enclosing file's import
// block via `use super::*`; the submodules own their own imports now, so the
// test module carries the set it actually uses.
use arrow::array::{
    Array, DictionaryArray, FixedSizeListArray, Float64Array, Int32Array, Int64Array, ListArray,
    StringArray, UInt16Array, UInt64Array,
};
use arrow::datatypes::{DataType, UInt16Type};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

fn sample_point_layer() -> ColumnarLayer {
    ColumnarLayer {
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
fn categorical_columns_use_dictionary_encoding() {
    let layer = ColumnarLayer {
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
    match field.data_type() {
        DataType::Dictionary(k, v) => {
            assert_eq!(k.as_ref(), &DataType::UInt16);
            assert_eq!(v.as_ref(), &DataType::Utf8);
        }
        other => panic!("expected Dictionary<UInt16, Utf8>, got {other:?}"),
    }

    let col = batch
        .column_by_name("kind")
        .unwrap()
        .as_any()
        .downcast_ref::<DictionaryArray<UInt16Type>>()
        .unwrap();
    let values = col.values().as_any().downcast_ref::<StringArray>().unwrap();
    // First-seen order: "car" then "bus".
    let mut categories: Vec<&str> = (0..values.len()).map(|i| values.value(i)).collect();
    categories.sort();
    assert_eq!(categories, vec!["bus", "car"]);

    // The 4th row is null; others reference one of the two slots.
    assert!(col.is_null(3));
    let keys = col.keys();
    for i in [0usize, 1, 2, 4] {
        assert!(keys.value(i) < values.len() as u16);
    }
}

#[test]
fn categorical_overflow_errors_instead_of_corrupting() {
    // A column whose distinct-value count exceeds the UInt16 dictionary
    // key space must be rejected, not silently collapsed onto one index.
    let n = u16::MAX as usize + 1; // 65_536 distinct strings
    let kinds: Vec<Option<String>> = (0..n).map(|i| Some(format!("c{i}"))).collect();
    let layer = ColumnarLayer {
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
    let err = encode_layer(&layer).expect_err("overflowing dictionary must error");
    assert!(
        err.to_string().contains("distinct values"),
        "unexpected error: {err}"
    );
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

    // Geometry is now a 3-wide list with z folded in; `z` is gone as a property.
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

    // Geometry leaf is now i32 grid indices, and the affine rides in metadata.
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

#[test]
fn quantization_shrinks_geometry_and_is_opt_in() {
    // A many-vertex line is coordinate-dominated; quantization should shrink
    // the IPC, and the default (None) path must stay byte-identical.
    let line: Vec<[f64; 2]> = (0..400)
        .map(|k| [-73.95 + k as f64 * 1e-4, 40.75 + k as f64 * 7e-5])
        .collect();
    let layer = ColumnarLayer {
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

#[test]
fn vertex_time_falls_back_to_int64_for_wide_spans() {
    // span = 100 billion ms; the u16 encoding would need step ≈ 1.5e6 ms,
    // far beyond the DEFAULT_VERTEX_TIME_MAX_STEP_MS ceiling — so the
    // encoder must take the exact List<Int64> path, byte-for-byte
    // absolute timestamps, with no origin/step metadata.
    let layer = ColumnarLayer {
        name: "edge".into(),
        feature_ids: vec![1],
        start_times: vec![0],
        end_times: vec![100],
        geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0]]]),
        vertex_times: Some(vec![vec![0, 100_000_000_000]]),
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    };
    let ipc = encode_layer(&layer).unwrap();
    let batch = decode_layer(&ipc).unwrap();
    let schema = batch.schema();
    let meta = schema.metadata();
    assert!(meta.get("stt:vertex_time_origin_ms").is_none());
    assert!(meta.get("stt:vertex_time_step_ms").is_none());
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
        .expect("wide spans must keep the exact Int64 shape");
    assert_eq!(absolutes.values(), &[0, 100_000_000_000]);
}

#[test]
fn vertex_time_step_ceiling_is_the_u16_vs_int64_threshold() {
    // span = 65_535_000 ms quantizes at exactly the 1000 ms default
    // ceiling → u16 deltas; one ms more pushes the step to 1001 → i64.
    let make = |span: i64| ColumnarLayer {
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
    };

    let at_ceiling = decode_layer(&encode_layer(&make(65_535_000)).unwrap()).unwrap();
    let schema = at_ceiling.schema();
    let step: u32 = schema
        .metadata()
        .get("stt:vertex_time_step_ms")
        .expect("span at the ceiling stays u16-delta encoded")
        .parse()
        .unwrap();
    assert_eq!(step, DEFAULT_VERTEX_TIME_MAX_STEP_MS);

    let past_ceiling = decode_layer(&encode_layer(&make(65_536_000)).unwrap()).unwrap();
    assert!(past_ceiling
        .schema()
        .metadata()
        .get("stt:vertex_time_step_ms")
        .is_none());
    let vt = past_ceiling
        .column_by_name("vertex_time")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let first = vt.value(0);
    let absolutes = first.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(absolutes.values(), &[0, 65_536_000]);
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

/// The v1 test's scenario through the V2 path: a length-inconsistent
/// layer must be the same descriptive Err, not an index-out-of-bounds
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
            format_version: FORMAT_VERSION,
            ..EncoderConfig::default()
        },
    )
    .expect_err("length-inconsistent layer must Err through the v2 path");
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
// Layer frame v2 (packed formatVersion 2)
// ------------------------------------------------------------------

/// v2 config in SELF-CONTAINED mode (inline schema sections, no registry
/// needed to decode), layered over `base`.
fn v2_inline(base: &EncoderConfig) -> EncoderConfig {
    EncoderConfig {
        format_version: FORMAT_VERSION,
        template_collector: None,
        ..base.clone()
    }
}

/// v2 config in HASH-REFERENCING mode (templates recorded with
/// `collector`; frames carry 16-byte hashes), layered over `base`.
fn v2_hashed(base: &EncoderConfig, collector: &Arc<TemplateCollector>) -> EncoderConfig {
    EncoderConfig {
        format_version: FORMAT_VERSION,
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

/// The v1-equivalence contract (design §4.3 / merge_v2_layer): the SAME
/// layers encoded v1 and v2 (both v2 modes) decode to EQUAL
/// `DecodedLayer`s — batch equality covers columns AND schema/field
/// metadata, i.e. the TILE_META re-injection must reproduce the v1
/// metadata byte-for-byte.
fn assert_v2_decodes_like_v1(layers: &[ColumnarLayer], base: &EncoderConfig, what: &str) {
    let v1 = decode_tile(&encode_tile_with(layers, base).unwrap()).unwrap();

    let inline = decode_tile(&encode_tile_with(layers, &v2_inline(base)).unwrap()).unwrap();
    assert_eq!(inline.len(), v1.len(), "{what}: inline layer count");
    for (a, b) in inline.iter().zip(&v1) {
        assert_eq!(a.name, b.name, "{what}: inline layer name");
        assert_eq!(a.batch, b.batch, "{what}: inline v2 decode != v1 decode");
    }

    let collector = Arc::new(TemplateCollector::new());
    let payload = encode_tile_with(layers, &v2_hashed(base, &collector)).unwrap();
    let registry = registry_from(&collector);
    let hashed = decode_tile_with_templates(&payload, &registry).unwrap();
    assert_eq!(hashed.len(), v1.len(), "{what}: hashed layer count");
    for (a, b) in hashed.iter().zip(&v1) {
        assert_eq!(a.name, b.name, "{what}: hashed layer name");
        assert_eq!(a.batch, b.batch, "{what}: hashed v2 decode != v1 decode");
    }
}

/// Every payload shape the v2 break touches round-trips to EXACTLY the v1
/// decode: plain + quantized points, dictionary props (incl. TWO
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

/// v2 row order (design §4.2 ★F10): rows come out stable-sorted by
/// `start_time`, ids travel WITH their rows (the sort runs after id
/// assignment), and the result equals the v1 decode of the pre-sorted
/// layer. v1 encoding of the same input stays in input order.
#[test]
fn v2_rows_stable_sorted_by_start_time_after_id_assignment() {
    let unsorted = ColumnarLayer {
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

    // Equivalent to v1-encoding the manually pre-sorted layer.
    let presorted = sort_rows_by_start_time(&unsorted).into_owned();
    let v1 =
        decode_tile(&encode_tile_with(&[presorted], &EncoderConfig::default()).unwrap()).unwrap();
    assert_eq!(batch, &v1[0].batch);
}

/// Template constancy (design §3.1d): tiles differing ONLY per-tile —
/// qa affines, t0, dictionary categories, row counts — share ONE
/// CORE + ONE PROPS template. Type variants (u16-delta vs exact-Int64
/// vertex_time) legitimately mint DISTINCT templates (§2.2 cardinality),
/// and every recorded template resolves through the registry.
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

/// TILE_META canonical serialization (design §4.3): alphabetical keys, no
/// whitespace, and unknown keys are ignored on decode (additive contract).
#[test]
fn v2_tile_meta_is_canonical_json_and_ignores_unknown_keys() {
    let meta = TileMeta {
        qa: Some(BTreeMap::from([("speed".to_string(), (0.0, 0.15))])),
        sorted: Some(true),
        t0: Some(1_577_836_800_000),
        vb: Some(24),
        vt: Some((1_577_836_800_000, 1000)),
    };
    assert_eq!(
        serde_json::to_string(&meta).unwrap(),
        r#"{"qa":{"speed":[0.0,0.15]},"sorted":true,"t0":1577836800000,"vb":24,"vt":[1577836800000,1000]}"#
    );
    // Presence rules: absent features serialize NO key at all.
    assert_eq!(serde_json::to_string(&TileMeta::default()).unwrap(), "{}");
    // Unknown keys from a future writer must be ignored, not rejected.
    let parsed: TileMeta = serde_json::from_str(r#"{"sorted":true,"zz_future":{"x":1}}"#).unwrap();
    assert_eq!(parsed.sorted, Some(true));
}

/// Walk a SINGLE-layer v2 frame's header and return each section's
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

/// Splice guards (design §3.4): stray zeros at a batch section's head
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

/// Byte offset of a single-layer v2 frame's FIRST TOC entry.
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

/// A hash-referencing v2 frame decoded WITHOUT (or with an incomplete)
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

/// The strip is tail-invariant (design §3.6, spike-proven): hoisting the
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
    // Different t0 + different qa affine (value range) per tile.
    for t in early.start_times.iter_mut() {
        *t += 7_000;
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
    assert_ne!(a.core_tail, b.core_tail, "t0 shift must land in the tail");
    assert_ne!(
        a.tile_meta_json, b.tile_meta_json,
        "TILE_META varies per tile"
    );
    // Same qa affine + same categories here → identical props tails is
    // fine; what matters is templates never absorb per-tile variance.
    let _ = (a_props_tail, b_props_tail);
}
