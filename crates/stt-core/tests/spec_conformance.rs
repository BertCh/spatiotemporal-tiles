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
//! See data-format.md § "Per-layer Arrow schema".

use arrow::datatypes::DataType;
use stt_core::arrow_tile::{
    decode_tile, encode_tile, tessellate_polygon, ColumnarLayer, Coord, GeometryColumn,
    PropertyColumn, DecodedLayer, TRIANGLES_METADATA_KEY,
};

/// The standard GeoArrow extension-name metadata key the spec mandates on the
/// `geometry` field.
const GEOARROW_EXT_KEY: &str = "ARROW:extension:name";

fn point_layer() -> ColumnarLayer {
    ColumnarLayer {
        name: "points".to_string(),
        feature_ids: vec![1, 2, 3],
        start_times: vec![1000, 2000, 3000],
        end_times: vec![1500, 2500, 3500],
        geometry: GeometryColumn::Point(vec![
            [-122.4, 37.7],
            [-122.5, 37.8],
            [-122.6, 37.9],
        ]),
        vertex_times: None,
        triangles: None,
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

fn line_layer() -> ColumnarLayer {
    ColumnarLayer {
        name: "tracks".to_string(),
        feature_ids: vec![10, 11],
        start_times: vec![0, 100],
        end_times: vec![50, 200],
        geometry: GeometryColumn::LineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            vec![[5.0, 5.0], [6.0, 6.0]],
        ]),
        // A tight temporal span → u16-delta vertex-time encoding (per spec).
        vertex_times: Some(vec![vec![0, 25, 50], vec![100, 200]]),
        triangles: None,
        properties: vec![],
    }
}

fn polygon_layer(triangles: Option<Vec<Vec<u32>>>) -> ColumnarLayer {
    ColumnarLayer {
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
        triangles,
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
    assert_eq!(start.data_type(), &DataType::Int64, "start_time must be Int64");
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

    // Categorical property: Dictionary<UInt16, Utf8> (per spec table).
    let kind = schema
        .field_with_name("kind")
        .expect("categorical property column present");
    match kind.data_type() {
        DataType::Dictionary(k, v) => {
            assert_eq!(k.as_ref(), &DataType::UInt16);
            assert_eq!(v.as_ref(), &DataType::Utf8);
        }
        other => panic!("categorical property must be Dictionary<UInt16, Utf8>, got {other:?}"),
    }

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
    assert!(!plain.batch.schema().metadata().contains_key(TRIANGLES_METADATA_KEY));
}

#[test]
fn pre_tessellated_polygon_carries_triangles_column() {
    // With pre-tessellation on (the `--pre-tessellate` build path).
    let exterior: Vec<Coord> = vec![
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 4.0],
        [0.0, 4.0],
        [0.0, 0.0],
    ];
    let tris = tessellate_polygon(&[exterior]);
    assert!(!tris.is_empty(), "earcut should produce indices for a square");

    let decoded = roundtrip(polygon_layer(Some(vec![tris])));
    assert_core_columns(&decoded, "geoarrow.polygon");

    let schema = decoded.batch.schema();
    let tri = schema
        .field_with_name("triangles")
        .expect("`triangles` column present when pre-tessellation is on");
    match tri.data_type() {
        DataType::List(child) => assert_eq!(
            child.data_type(),
            &DataType::UInt32,
            "triangles must be List<UInt32>",
        ),
        other => panic!("triangles must be a List, got {other:?}"),
    }

    // The spec's schema-metadata advertisement flag must be set.
    assert_eq!(
        schema.metadata().get(TRIANGLES_METADATA_KEY).map(String::as_str),
        Some("true"),
        "pre-tessellated layers must advertise stt:has_triangles=true",
    );
}
