//! Spec-conformance test: locks the per-layer Arrow schema documented in
//! `docs/architecture/data-format.md` to the actual encoder output so the
//! spec and the code cannot silently drift.
//!
//! For each geometry kind (point, linestring, polygon) it builds a tiny tile
//! through the PUBLIC `stt-core` API, decodes the per-layer Arrow schema, and
//! asserts that the documented columns are present with the documented types:
//!
//! | column        | type                                  |
//! |---------------|---------------------------------------|
//! | `id`          | `UInt64`                              |
//! | `start_time`  | `Int64`                               |
//! | `end_time`    | `Int64`                               |
//! | `geometry`    | present + GeoArrow extension name     |
//! | `triangles`   | `List<UInt32>` (polygon + pre-tess)   |
//!
//! It also pins the opt-in re-typings and containers the spec documents:
//! attribute quantization (`stt:qa`), coordinate quantization (`stt:quant`),
//! the `vertex_value_matrix` + `stt:vertex_value_buckets` space-time cube
//! payload, and the paged-directory encode→decode round-trip.
//!
//! See data-format.md § "Per-layer Arrow schema".
//!
//! The final section locks the *manifest* side of the same contract: the three
//! M7 honesty blocks (`metadata.content_fingerprint`, `metadata.z_range`,
//! `orderingWorkload` / `metadata.ordering_workload`) against their
//! declarations in `docs/spec/manifest.schema.json`, and the validator's check
//! numbering against the two documents that restate it. Rules in
//! `docs/spec/conformance.md` §3 that nothing executes are prose that rots;
//! these are the executions.

use arrow::datatypes::DataType;
use stt_core::arrow_tile::{
    decode_tile, encode_tile, encode_tile_quantized, encode_tile_with, tessellate_polygon,
    AttrQuant, ColumnarLayer, Coord, DecodedLayer, EncoderConfig, GeometryColumn, PropertyColumn,
    QuantAffine, VectorElem, STT_QUANT_ATTR_META_KEY, STT_QUANT_META_KEY, TRIANGLES_METADATA_KEY,
};

mod common;
// The points/timed-linestring fixtures are shared with `reproducible_build.rs`;
// their exact time values don't matter to these schema-lock assertions, only
// their column *shapes* (numeric+categorical props, tight-span vertex times).
use common::{canonical_order, line_fixture as line_layer, permute, point_fixture as point_layer};

/// The standard GeoArrow extension-name metadata key the spec mandates on the
/// `geometry` field.
const GEOARROW_EXT_KEY: &str = "ARROW:extension:name";

fn polygon_layer(triangles: Option<Vec<Vec<u32>>>) -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "zones".to_string(),
        feature_ids: vec![42],
        start_times: vec![0],
        end_times: vec![1000],
        geometry: GeometryColumn::Polygon(vec![vec![vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ]]]),
        vertex_times: None,
        vertex_values: None,
        triangles,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

/// Encode a single-layer tile, decode it, and return the one decoded layer.
fn roundtrip(layer: ColumnarLayer) -> DecodedLayer {
    let payload = encode_tile(std::slice::from_ref(&layer)).expect("encode_tile");
    let mut decoded = decode_tile(&payload).expect("decode_tile");
    assert_eq!(decoded.len(), 1, "expected exactly one decoded layer");
    decoded.pop().unwrap()
}

/// Assert the spec-mandated core columns: id/start_time/end_time/geometry, plus
/// the GeoArrow extension name on the geometry field.
fn assert_core_columns(decoded: &DecodedLayer, expected_geoarrow_name: &str) {
    let schema = decoded.batch.schema();

    let id = schema.field_with_name("id").expect("`id` column present");
    assert_eq!(id.data_type(), &DataType::UInt64, "id must be UInt64");
    assert!(!id.is_nullable(), "id is non-null per spec");

    let start = schema
        .field_with_name("start_time")
        .expect("`start_time` column present");
    assert_eq!(
        start.data_type(),
        &DataType::Int64,
        "start_time must be Int64"
    );
    assert!(!start.is_nullable(), "start_time is non-null per spec");

    let end = schema
        .field_with_name("end_time")
        .expect("`end_time` column present");
    assert_eq!(end.data_type(), &DataType::Int64, "end_time must be Int64");
    assert!(!end.is_nullable(), "end_time is non-null per spec");

    let geom = schema
        .field_with_name("geometry")
        .expect("`geometry` column present");
    assert_eq!(
        geom.metadata().get(GEOARROW_EXT_KEY).map(String::as_str),
        Some(expected_geoarrow_name),
        "geometry field must carry the GeoArrow extension name '{expected_geoarrow_name}'",
    );
}

#[test]
fn point_layer_matches_spec_schema() {
    let decoded = roundtrip(point_layer());
    assert_core_columns(&decoded, "geoarrow.point");

    let schema = decoded.batch.schema();

    // Numeric property: Float64, nullable (per spec table).
    let speed = schema
        .field_with_name("speed")
        .expect("numeric property column present");
    assert_eq!(speed.data_type(), &DataType::Float64);
    assert!(speed.is_nullable());

    // Small categorical property: plain Utf8 is cheaper than an IPC
    // dictionary batch (the representation is chosen per tile).
    let kind = schema
        .field_with_name("kind")
        .expect("categorical property column present");
    assert_eq!(kind.data_type(), &DataType::Utf8);

    // A point layer carries no triangle column.
    assert!(decoded.batch.column_by_name("triangles").is_none());
}

#[test]
fn linestring_layer_matches_spec_schema() {
    let decoded = roundtrip(line_layer());
    assert_core_columns(&decoded, "geoarrow.linestring");

    let schema = decoded.batch.schema();

    // vertex_time is present and nullable; in the tight-span case the spec
    // says it is List<UInt16> deltas keyed on origin/step schema metadata.
    let vt = schema
        .field_with_name("vertex_time")
        .expect("`vertex_time` column present for a timed linestring");
    assert!(vt.is_nullable(), "vertex_time is nullable per spec");
    match vt.data_type() {
        DataType::List(child) => assert_eq!(
            child.data_type(),
            &DataType::UInt16,
            "tight-span vertex_time must be List<UInt16> deltas",
        ),
        other => panic!("vertex_time must be a List, got {other:?}"),
    }

    // The delta-encoding metadata keys documented in the spec must be present.
    let meta = schema.metadata();
    assert!(
        meta.contains_key("stt:vertex_time_origin_ms"),
        "u16-delta vertex_time layers must carry stt:vertex_time_origin_ms",
    );
    assert!(
        meta.contains_key("stt:vertex_time_step_ms"),
        "u16-delta vertex_time layers must carry stt:vertex_time_step_ms",
    );
}

#[test]
fn polygon_layer_matches_spec_schema() {
    // Without pre-tessellation: no triangles column, no metadata flag.
    let plain = roundtrip(polygon_layer(None));
    assert_core_columns(&plain, "geoarrow.polygon");
    assert!(
        plain.batch.column_by_name("triangles").is_none(),
        "polygon without --pre-tessellate must not carry a triangles column",
    );
    assert!(!plain
        .batch
        .schema()
        .metadata()
        .contains_key(TRIANGLES_METADATA_KEY));
}

#[test]
fn pre_tessellated_polygon_carries_triangles_column() {
    // With pre-tessellation on (the `--pre-tessellate` build path).
    let exterior: Vec<Coord> = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]];
    let tris = tessellate_polygon(&[exterior]);
    assert!(
        !tris.is_empty(),
        "earcut should produce indices for a square"
    );

    let decoded = roundtrip(polygon_layer(Some(vec![tris])));
    assert_core_columns(&decoded, "geoarrow.polygon");

    let schema = decoded.batch.schema();
    let tri = schema
        .field_with_name("triangles")
        .expect("`triangles` column present when pre-tessellation is on");
    match tri.data_type() {
        // Feature-local indices narrow to UInt16 when they fit (the common
        // case — this fixture's indices are all < 4) and fall back to
        // UInt32 only when a feature's index exceeds u16::MAX.
        DataType::List(child) => assert!(
            matches!(child.data_type(), DataType::UInt16 | DataType::UInt32),
            "triangles must be List<UInt16> or List<UInt32>, got List<{:?}>",
            child.data_type(),
        ),
        other => panic!("triangles must be a List, got {other:?}"),
    }

    // The spec's schema-metadata advertisement flag must be set.
    assert_eq!(
        schema
            .metadata()
            .get(TRIANGLES_METADATA_KEY)
            .map(String::as_str),
        Some("true"),
        "pre-tessellated layers must advertise stt:has_triangles=true",
    );
}

#[test]
fn vector_group_column_matches_spec_schema() {
    // A `--vector-group` column bakes an interleaved GPU-ready instance
    // attribute as `FixedSizeList<Float32|UInt8, width>` (data-format.md), so
    // the TS decoder can hand the contiguous child buffer to deck.gl zero-copy.
    use arrow::array::{Array, FixedSizeListArray};
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
                "point_rgba".to_string(),
                PropertyColumn::Vector {
                    width: 4,
                    elem: VectorElem::U8,
                    values: vec![255.0, 0.0, 0.0, 128.0, 0.0, 255.0, 0.0, 255.0],
                },
            ),
        ],
    };
    let decoded = roundtrip(layer);
    assert_core_columns(&decoded, "geoarrow.point");
    let schema = decoded.batch.schema();

    // f32 vector → FixedSizeList<Float32, width>; the column field is nullable.
    let quat = schema
        .field_with_name("surfel_quat")
        .expect("f32 vector-group column present");
    assert!(
        quat.is_nullable(),
        "vector-group column is nullable per spec"
    );
    match quat.data_type() {
        DataType::FixedSizeList(child, len) => {
            assert_eq!(*len, 4, "vector width is the FixedSizeList size");
            assert_eq!(child.data_type(), &DataType::Float32, "f32 vector leaf");
        }
        other => panic!("f32 vector must be FixedSizeList<Float32,4>, got {other:?}"),
    }

    // u8 vector → FixedSizeList<UInt8, width> (packed RGBA).
    let rgba = schema
        .field_with_name("point_rgba")
        .expect("u8 vector-group column present");
    match rgba.data_type() {
        DataType::FixedSizeList(child, len) => {
            assert_eq!(*len, 4);
            assert_eq!(child.data_type(), &DataType::UInt8, "u8 vector leaf");
        }
        other => panic!("u8 vector must be FixedSizeList<UInt8,4>, got {other:?}"),
    }

    // The interleaved child buffer round-trips with the documented row-major
    // layout (feature `i` occupies `[i*width, (i+1)*width)`).
    let arr = decoded
        .batch
        .column_by_name("surfel_quat")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.value_length(), 4);
}

#[test]
fn attribute_quantization_roundtrips_via_stt_qa() {
    // A numeric property built with an explicit `--quantize-attr` precision
    // ships as fixed-point ints + the `stt:qa` affine in FIELD-level metadata
    // (data-format.md § "Numeric attribute quantization"): `value = o + q * s`,
    // `o` = the column's finite minimum, `s` = the requested precision, nulls
    // stay null, reconstruction error ≤ s/2.
    use arrow::array::{Array, UInt16Array};
    use std::collections::HashMap;

    let precision = 0.5f64;
    let cfg = EncoderConfig {
        quantize_attrs: HashMap::from([("speed".to_string(), precision)]),
        ..EncoderConfig::default()
    };
    let layer = point_layer(); // speed = [Some(10.0), None, Some(30.0)]
    let payload = encode_tile_with(std::slice::from_ref(&layer), &cfg).expect("encode_tile_with");
    let mut decoded = decode_tile(&payload).expect("decode_tile");
    let decoded = decoded.pop().unwrap();

    let schema = decoded.batch.schema();
    let speed = schema
        .field_with_name("speed")
        .expect("quantized property present");
    assert!(
        matches!(speed.data_type(), DataType::UInt16 | DataType::Int32),
        "quantized property must be a UInt16/Int32 fixed-point leaf, got {:?}",
        speed.data_type()
    );
    let affine = speed
        .metadata()
        .get(STT_QUANT_ATTR_META_KEY)
        .expect("quantized property must carry stt:qa field metadata");
    let qa = AttrQuant::from_json(affine).expect("stt:qa must parse as the {o,s} affine");
    assert_eq!(qa.o, 10.0, "o is the column's finite minimum");
    assert_eq!(qa.s, precision, "s is the requested precision");

    // Reconstruct through the documented affine and compare to the originals.
    let col = decoded
        .batch
        .column_by_name("speed")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .expect("this fixture's range fits the UInt16 leaf");
    // The fixture's start times are deliberately out of order, so decoded rows
    // arrive in canonical (start_time-sorted) order — permute the expectation.
    let order = canonical_order(&layer.start_times);
    let PropertyColumn::Numeric(speed_want) = &layer.properties[0].1 else {
        unreachable!()
    };
    let speed_want = permute(speed_want, &order);
    for (i, want) in speed_want.iter().enumerate() {
        assert_eq!(
            col.is_null(i),
            want.is_none(),
            "row {i}: nulls survive quantization"
        );
    }
    for (i, want) in speed_want
        .iter()
        .enumerate()
        .filter_map(|(i, w)| w.map(|v| (i, v)))
    {
        let got = qa.value(col.value(i) as i64);
        assert!(
            (got - want).abs() <= precision / 2.0,
            "row {i}: reconstructed {got}, want {want} ± {}",
            precision / 2.0
        );
    }
}

#[test]
fn coordinate_quantization_roundtrips_via_stt_quant() {
    // A coordinate-quantized layer keeps the GeoArrow nesting but swaps the
    // leaf to Int32 grid indices; the reconstruction affine rides the geometry
    // field's `stt:quant` metadata and REPLACES the CRS84 extension metadata
    // (data-format.md § "Coordinate quantization"). Worst-case error is half a
    // quantum of the requested ground precision.
    use arrow::array::{FixedSizeListArray, Int32Array};

    let meters = 1.0f64;
    let layer = point_layer();
    let payload =
        encode_tile_quantized(std::slice::from_ref(&layer), Some(meters)).expect("encode");
    let mut decoded = decode_tile(&payload).expect("decode_tile");
    let decoded = decoded.pop().unwrap();

    let schema = decoded.batch.schema();
    let geom = schema
        .field_with_name("geometry")
        .expect("geometry present");
    assert_eq!(
        geom.metadata().get(GEOARROW_EXT_KEY).map(String::as_str),
        Some("geoarrow.point"),
        "quantized geometry keeps its GeoArrow extension name"
    );
    assert!(
        geom.metadata().get("ARROW:extension:metadata").is_none(),
        "quantized geometry must NOT carry the CRS84 metadata (the affine replaces it)"
    );
    let affine = geom
        .metadata()
        .get(STT_QUANT_META_KEY)
        .expect("quantized geometry must carry stt:quant field metadata");
    let q = QuantAffine::from_json(affine).expect("stt:quant must parse as the affine");
    // World-anchored grid constants (dataset-independent, dedup-preserving).
    assert_eq!((q.x0, q.y0), (-180.0, -90.0), "world-grid origin");
    assert!(q.sx > 0.0 && q.sy > 0.0);

    let arr = decoded
        .batch
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("point geometry is FixedSizeList");
    let leaf = arr
        .values()
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("quantized leaf must be Int32");
    let GeometryColumn::Point(points) = &layer.geometry else {
        unreachable!()
    };
    let points = permute(points, &canonical_order(&layer.start_times));
    for (i, [lon, lat]) in points.iter().enumerate() {
        let got_lon = q.lon(leaf.value(i * 2));
        let got_lat = q.lat(leaf.value(i * 2 + 1));
        assert!(
            (got_lon - lon).abs() <= q.sx / 2.0 + f64::EPSILON,
            "point {i}: lon {got_lon} vs {lon} (quantum {})",
            q.sx
        );
        assert!(
            (got_lat - lat).abs() <= q.sy / 2.0 + f64::EPSILON,
            "point {i}: lat {got_lat} vs {lat} (quantum {})",
            q.sy
        );
    }

    // The unquantized encode of the same layer DOES carry the CRS84 metadata —
    // the writer-MUST in docs/spec/conformance.md §3.
    let plain = roundtrip(point_layer());
    let plain_geom = plain
        .batch
        .schema()
        .field_with_name("geometry")
        .unwrap()
        .clone();
    let crs = plain_geom
        .metadata()
        .get("ARROW:extension:metadata")
        .expect("unquantized geometry must carry the GeoArrow CRS metadata");
    assert!(
        crs.contains("OGC:CRS84"),
        "CRS must pin OGC:CRS84, got {crs}"
    );
}

#[test]
fn vertex_value_matrix_carries_consistent_bucket_count() {
    // The space-time cube payload (data-format.md § "Space-time cube payload"):
    // static LineString geometry + a vertex-major `vertex_count × num_buckets`
    // matrix per feature, with `num_buckets` in schema metadata under
    // `stt:vertex_value_buckets` so the reader can reshape the flat rows.
    use arrow::array::{Array, Float32Array, ListArray};

    let buckets = 2usize;
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "corridors".to_string(),
        feature_ids: vec![1, 2],
        start_times: vec![0, 0],
        end_times: vec![7_200_000, 7_200_000],
        geometry: GeometryColumn::LineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], // 3 vertices → 6 values
            vec![[5.0, 5.0], [6.0, 6.0]],             // 2 vertices → 4 values
        ]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: Some(vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0],
        ]),
        properties: vec![],
    };
    let decoded = roundtrip(layer.clone());
    assert_core_columns(&decoded, "geoarrow.linestring");

    let schema = decoded.batch.schema();
    let vm = schema
        .field_with_name("vertex_value_matrix")
        .expect("`vertex_value_matrix` column present");
    assert!(vm.is_nullable());
    match vm.data_type() {
        DataType::List(child) => assert_eq!(
            child.data_type(),
            &DataType::Float32,
            "vertex_value_matrix must be List<Float32>",
        ),
        other => panic!("vertex_value_matrix must be a List, got {other:?}"),
    }
    assert_eq!(
        schema
            .metadata()
            .get("stt:vertex_value_buckets")
            .map(String::as_str),
        Some("2"),
        "bucket count must be recorded in schema metadata",
    );

    // Row widths are vertex_count × buckets and the vertex-major values
    // round-trip exactly (`matrix[v * num_buckets + b]`).
    let col = decoded
        .batch
        .column_by_name("vertex_value_matrix")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let want = layer.vertex_value_matrix.as_ref().unwrap();
    for (i, (row, n_vertices)) in want.iter().zip([3usize, 2]).enumerate() {
        assert_eq!(
            col.value_length(i) as usize,
            n_vertices * buckets,
            "row {i} width"
        );
        let vals = col.value(i);
        let vals = vals.as_any().downcast_ref::<Float32Array>().unwrap();
        let got: Vec<f32> = (0..vals.len()).map(|j| vals.value(j)).collect();
        assert_eq!(&got, row, "row {i} values");
    }
}

// ===========================================================================
// Manifest-honesty contract pins (M7 — SH-1 fingerprint, SH-2 z_range,
// SH-3 ordering workload). Normative text: docs/spec/conformance.md §3.
// ===========================================================================
//
// These three manifest blocks are unlike every other key in the envelope: they
// make a CLAIM about content rather than describing structure. A malformed
// claim is worse than an absent one, because `stt-validate` compares decoded
// content against it — so the wire shape is pinned on BOTH sides of the
// cross-language boundary. `packages/core/test/manifest-schema.test.ts` pins
// the TS reader against the same declarations; this is the writer half, and it
// asserts one thing a plain schema validation cannot: that the Rust type
// introduces no key the published schema has not declared.

use serde_json::Value;
use std::collections::BTreeMap;
use stt_core::metadata::{
    ContentFingerprint, Metadata, OrderingWorkload, CONTENT_FINGERPRINT_VERSION,
};

/// The published manifest schema, or `None` on a published-crate checkout
/// (where `docs/` does not travel with the tarball) — skipping loudly, exactly
/// like `capability_registry.rs`.
fn published_schema() -> Option<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/spec/manifest.schema.json"
    );
    let text = std::fs::read_to_string(path).ok().or_else(|| {
        eprintln!(
            "spec_conformance: manifest.schema.json absent (published-crate build) — skipping"
        );
        None
    })?;
    Some(serde_json::from_str(&text).expect("manifest.schema.json parses"))
}

/// Read a repo file, or `None` (loudly) when it is not part of this checkout.
fn repo_file(relative: &str) -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(_) => {
            eprintln!("spec_conformance: {relative} absent — skipping");
            None
        }
    }
}

/// Assert `value` satisfies the subset of JSON Schema this schema uses, AND —
/// the direction a reader-side validator can never check — that it carries no
/// property the schema has not declared. On the WRITER side an undeclared key
/// means the published schema is behind the Rust type, which is precisely the
/// drift a three-way pin exists to catch.
fn assert_conforms(value: &Value, schema: &Value, path: &str) {
    if let Some(ty) = schema.get("type").and_then(Value::as_str) {
        let ok = match ty {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "string" => value.is_string(),
            "boolean" => value.is_boolean(),
            _ => true,
        };
        assert!(ok, "{path}: expected JSON {ty}, got {value}");
    }
    if let Some(obj) = value.as_object() {
        for req in schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let key = req.as_str().expect("`required` entries are strings");
            assert!(
                obj.contains_key(key),
                "{path}: missing required property `{key}`"
            );
        }
        let props = schema.get("properties").and_then(Value::as_object);
        let extra = schema.get("additionalProperties").filter(|v| v.is_object());
        for (key, sub_value) in obj {
            if let Some(sub) = props.and_then(|p| p.get(key)) {
                assert_conforms(sub_value, sub, &format!("{path}.{key}"));
            } else if let Some(extra) = extra {
                assert_conforms(sub_value, extra, &format!("{path}.{key}"));
            } else {
                panic!(
                    "{path}: `{key}` is not declared in docs/spec/manifest.schema.json — \
                     the published schema is behind the Rust type. Add it there (and to the \
                     TS pin) in the same change that added the field."
                );
            }
        }
    }
    if let Some(items) = value.as_array() {
        if let Some(min) = schema.get("minItems").and_then(Value::as_u64) {
            assert!(
                items.len() as u64 >= min,
                "{path}: {} item(s) < minItems {min}",
                items.len()
            );
        }
        if let Some(max) = schema.get("maxItems").and_then(Value::as_u64) {
            assert!(
                items.len() as u64 <= max,
                "{path}: {} item(s) > maxItems {max}",
                items.len()
            );
        }
        if let Some(item_schema) = schema.get("items") {
            for (i, item) in items.iter().enumerate() {
                assert_conforms(item, item_schema, &format!("{path}[{i}]"));
            }
        }
    }
    if let (Some(min), Some(n)) = (
        schema.get("minimum").and_then(Value::as_f64),
        value.as_f64(),
    ) {
        assert!(n >= min, "{path}: {n} < minimum {min}");
    }
}

/// Metadata carrying every M7 honesty block, with every optional sub-map
/// populated so the pin actually inspects their declared shapes.
fn honest_metadata() -> Metadata {
    Metadata::new("honest")
        .with_z_range(Some([-430.0, 18_500.5]))
        .with_ordering_workload(OrderingWorkload {
            scrub: 3,
            pan: 2,
            playback: 1,
            playback_window_buckets: 8,
            runway_multiplier: 4,
            coalesce_gap_bytes: 2 * 1024 * 1024,
        })
        .with_content_fingerprint(ContentFingerprint {
            version: CONTENT_FINGERPRINT_VERSION,
            bbox: [-122.52, 37.70, -122.35, 37.83],
            z_range: Some([0.0, 1250.5]),
            distinct_feature_count: 4096,
            numeric_ranges: BTreeMap::from([("speed".to_string(), [0.0, 31.25])]),
            categorical_cardinality: BTreeMap::from([("kind".to_string(), 3)]),
            coord_tolerance_deg: 0.0,
            column_tolerance: BTreeMap::from([("speed".to_string(), 0.25)]),
        })
}

#[test]
fn metadata_honesty_blocks_match_the_published_schema() {
    let Some(schema) = published_schema() else {
        return;
    };
    let json = serde_json::to_value(honest_metadata()).expect("metadata serializes");
    let metadata_props = &schema["properties"]["metadata"]["properties"];

    for key in ["z_range", "content_fingerprint", "ordering_workload"] {
        let declared = &metadata_props[key];
        assert!(
            !declared.is_null(),
            "docs/spec/manifest.schema.json does not pin `metadata.{key}` — a content CLAIM \
             must be pinned, or an external validator cannot reject a malformed one"
        );
        let value = json
            .get(key)
            .unwrap_or_else(|| panic!("the writer emits `metadata.{key}` when it is set"));
        assert_conforms(value, declared, &format!("metadata.{key}"));
    }

    // The top-level `orderingWorkload` is documented as a byte-identical mirror
    // of the metadata copy, so one value must satisfy both declarations.
    assert_conforms(
        &json["ordering_workload"],
        &schema["properties"]["orderingWorkload"],
        "orderingWorkload",
    );

    // The version this build emits must be one the published block admits.
    let version_min = schema["properties"]["metadata"]["properties"]["content_fingerprint"]
        ["properties"]["version"]["minimum"]
        .as_u64()
        .expect("the fingerprint version carries a declared minimum");
    assert!(
        u64::from(CONTENT_FINGERPRINT_VERSION) >= version_min,
        "CONTENT_FINGERPRINT_VERSION {CONTENT_FINGERPRINT_VERSION} is below the schema's \
         declared minimum {version_min}"
    );
}

#[test]
fn honesty_blocks_are_absent_from_a_legacy_metadata() {
    // The additive criterion, asserted at the spec boundary rather than only in
    // the type's own unit tests: a build that sets none of the three emits none
    // of the three, so every manifest written before they existed keeps its
    // bytes and stays conformant.
    let json = serde_json::to_string(&Metadata::new("legacy")).expect("serializes");
    for key in ["z_range", "content_fingerprint", "ordering_workload"] {
        assert!(
            !json.contains(key),
            "an unset `{key}` must be omitted entirely, not null-filled: {json}"
        );
    }
    // …and the schema agrees: none of them is required.
    let Some(schema) = published_schema() else {
        return;
    };
    assert!(
        schema["properties"]["metadata"]["required"].is_null(),
        "no `metadata` sub-block may be required — pre-field manifests must stay valid"
    );
    assert!(
        !schema["required"]
            .as_array()
            .expect("top-level required list")
            .iter()
            .any(|v| v == "orderingWorkload"),
        "`orderingWorkload` must stay optional"
    );
}

#[test]
fn validator_check_numbering_agrees_across_the_documents() {
    // `stt-validate`'s numbered check list is quoted in three places: the
    // binary's module doc (the source of truth), the conformance spec's §2.3
    // prose, and the CLI reference. A finding is cited as "check N" in reports,
    // reviews and this plan's rule text, so the three lists must number the
    // same checks the same way or "check 12" means different things depending
    // on which document a reader reached for.
    let Some(module_doc) = repo_file("crates/spatiotemporal-tiles/src/bin/stt-validate/main.rs")
    else {
        return;
    };
    let highest_check = module_doc
        .lines()
        .take_while(|l| l.starts_with("//!"))
        .filter_map(|l| {
            let rest = l.trim_start_matches("//!").trim_start();
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if digits.is_empty() || !rest[digits.len()..].starts_with('.') {
                return None;
            }
            digits.parse::<u32>().ok()
        })
        .max()
        .expect("the stt-validate module doc carries a numbered check list");
    assert!(
        highest_check >= 12,
        "the module doc lost its numbered checks (highest = {highest_check})"
    );

    if let Some(cli_reference) = repo_file("docs/api/cli-reference.md") {
        let section = cli_reference
            .split("\n## ")
            .find(|s| s.starts_with("`stt-validate`"))
            .expect("cli-reference.md documents stt-validate");
        let documented = section
            .lines()
            .filter_map(|l| {
                let digits: String = l.chars().take_while(char::is_ascii_digit).collect();
                if digits.is_empty() || !l[digits.len()..].starts_with(". ") {
                    return None;
                }
                digits.parse::<u32>().ok()
            })
            .max()
            .expect("the stt-validate section carries a numbered check list");
        assert_eq!(
            documented, highest_check,
            "docs/api/cli-reference.md numbers stt-validate's checks 1..{documented} but the \
             binary's module doc numbers them 1..{highest_check} — renumber the doc in the same \
             change that added the check, or 'check N' names two different checks"
        );
    }

    if let Some(conformance) = repo_file("docs/spec/conformance.md") {
        assert!(
            conformance.contains(&format!("check {highest_check}")),
            "docs/spec/conformance.md §2.3 does not mention `check {highest_check}` — every \
             validator check is part of the conformance contract, not just the ones that \
             predate the spec's last edit"
        );
    }
}
