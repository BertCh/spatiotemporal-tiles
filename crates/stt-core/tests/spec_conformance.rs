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

use arrow::datatypes::DataType;
use stt_core::arrow_tile::{
    decode_tile, encode_tile, encode_tile_quantized, encode_tile_with, tessellate_polygon,
    AttrQuant, ColumnarLayer, Coord, EncoderConfig, GeometryColumn, PropertyColumn, QuantAffine,
    DecodedLayer, VectorElem, STT_QUANT_ATTR_META_KEY, STT_QUANT_META_KEY, TRIANGLES_METADATA_KEY,
};
use stt_core::metadata::{
    Metadata, SummaryAggregation, SummaryColumn, SummaryScheme, SummaryTier,
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
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
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
        schema.metadata().get(TRIANGLES_METADATA_KEY).map(String::as_str),
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
    assert!(quat.is_nullable(), "vector-group column is nullable per spec");
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
    let speed = schema.field_with_name("speed").expect("quantized property present");
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
    assert!(!col.is_null(0) && col.is_null(1) && !col.is_null(2), "nulls survive quantization");
    for (i, want) in [(0usize, 10.0f64), (2, 30.0)] {
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
    let geom = schema.field_with_name("geometry").expect("geometry present");
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
    let GeometryColumn::Point(points) = &layer.geometry else { unreachable!() };
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
    let plain_geom = plain.batch.schema().field_with_name("geometry").unwrap().clone();
    let crs = plain_geom
        .metadata()
        .get("ARROW:extension:metadata")
        .expect("unquantized geometry must carry the GeoArrow CRS metadata");
    assert!(crs.contains("OGC:CRS84"), "CRS must pin OGC:CRS84, got {crs}");
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
        schema.metadata().get("stt:vertex_value_buckets").map(String::as_str),
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
        assert_eq!(col.value_length(i) as usize, n_vertices * buckets, "row {i} width");
        let vals = col.value(i);
        let vals = vals.as_any().downcast_ref::<Float32Array>().unwrap();
        let got: Vec<f32> = (0..vals.len()).map(|j| vals.value(j)).collect();
        assert_eq!(&got, row, "row {i} values");
    }
}

#[test]
fn paged_directory_encode_decode_roundtrips() {
    // The paged `.sttd` container (packed-format §4.1): encode → decode must
    // reproduce the entry list in directory order `(zoom, hilbert, time_start,
    // temporal_bucket_ms)` for any page size and both framings, and the
    // structural validator must find a clean encode clean.
    use stt_core::archive::TileEntry;
    use stt_core::tile::TileId;
    use stt_core::{decode_paged_directory, encode_paged_directory, verify_paged_structure};

    let entry = |z: u8, x: u32, y: u32, ts: i64| TileEntry {
        zoom: z,
        x,
        y,
        time_start: ts,
        time_end: ts + 3_599_000,
        pack_id: 0,
        offset: (x as u64) * 64,
        length: 50 + x,
        uncompressed_size: 100 + x,
        feature_count: 1 + x,
        hilbert: TileId::new(z, x, y, 0).hilbert_index(),
        crc32c: 0xC0FFEE ^ x,
        temporal_bucket_ms: Some(3_600_000),
        cover_t_min: Some(ts + 250),
    };
    let mut entries = Vec::new();
    for z in [4u8, 8] {
        for i in 0..9u32 {
            for b in 0..2i64 {
                entries.push(entry(z, (i * 7) % (1 << z), (i * 13) % (1 << z), b * 3_600_000));
            }
        }
    }

    let mut want = entries.clone();
    want.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start, e.temporal_bucket_ms));

    for page_entries in [1usize, 5, 4096] {
        for zstd in [true, false] {
            let enc = encode_paged_directory(&entries, page_entries, zstd)
                .expect("encode_paged_directory");
            assert_eq!(enc.page_count, entries.len().div_ceil(page_entries));
            let got = decode_paged_directory(&enc.bytes, enc.root_length, zstd)
                .expect("decode_paged_directory");
            assert_eq!(
                got, want,
                "paged decode must equal the directory-ordered input \
                 (page_entries={page_entries}, zstd={zstd})"
            );
            let issues = verify_paged_structure(&enc.bytes, enc.root_length, zstd)
                .expect("verify_paged_structure");
            assert!(issues.is_empty(), "clean encode reported issues: {issues:?}");
        }
    }
}

#[test]
fn summary_tier_quadbin_and_sub_buckets_roundtrip() {
    // Both the CARTO `quadbin` scheme and multi-`sub_buckets` summary tiers are
    // shipped + documented; they must survive the manifest JSON round-trip and
    // the scheme must serialize with its documented lowercase spelling.
    let tier = SummaryTier {
        scheme: SummaryScheme::Quadbin,
        min_zoom: 0,
        max_zoom: 6,
        cell_resolution_per_zoom: vec![0, 1, 2, 3, 4, 5, 6],
        columns: vec![SummaryColumn {
            name: "magnitude".to_string(),
            agg: SummaryAggregation::Mean,
        }],
        layer_name: "summary".to_string(),
        sub_buckets: 24,
    };
    let bytes = Metadata::new("q")
        .with_summary_tier(tier)
        .to_json_bytes()
        .expect("serialize metadata");
    let decoded = Metadata::from_json_bytes(&bytes).expect("deserialize metadata");
    let dt = decoded.summary_tier.expect("summary tier round-trips");
    assert_eq!(dt.scheme, SummaryScheme::Quadbin, "quadbin scheme survives");
    assert_eq!(dt.sub_buckets, 24, "sub_buckets survives the round-trip");
    assert_eq!(dt.max_zoom, 6);

    let json = String::from_utf8(bytes).unwrap();
    assert!(json.contains("quadbin"), "scheme serializes lowercase: {json}");
    assert!(json.contains("sub_buckets"), "sub_buckets is a manifest field: {json}");
}

#[test]
fn summary_tier_h3_defaults_to_single_sub_bucket() {
    // A pre-`sub_buckets` manifest (the field absent) must decode to the legacy
    // single-count behaviour via serde's documented default.
    let json = br#"{
        "name": "old-summary",
        "description": "",
        "attribution": "",
        "bounds": {"min_lon":-180.0,"min_lat":-85.0,"max_lon":180.0,"max_lat":85.0},
        "time_range": {"start":0,"end":1000},
        "min_zoom": 0,
        "max_zoom": 8,
        "tile_count": 0,
        "feature_count": 0,
        "layers": ["default"],
        "properties": {},
        "temporal_bucket_ms": 3600000,
        "summary_tier": {
            "scheme": "h3",
            "min_zoom": 0,
            "max_zoom": 4,
            "cell_resolution_per_zoom": [0,1,2,3,4],
            "columns": [{"name":"magnitude","agg":"mean"}]
        }
    }"#;
    let m = Metadata::from_json_bytes(json).expect("legacy summary tier decodes");
    let dt = m.summary_tier.expect("summary tier present");
    assert_eq!(dt.scheme, SummaryScheme::H3);
    assert_eq!(dt.sub_buckets, 1, "absent sub_buckets defaults to 1");
    assert_eq!(dt.layer_name, "summary", "absent layer_name defaults to 'summary'");
}
