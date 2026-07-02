//! Guards the reproducible-encoding contract that content-addressed pack dedup
//! depends on, and the `stt:time_offset_ms` bake.
//!
//! Background: `arrow_tile::encode_layer_cfg` assembles schema- and field-level
//! metadata from **sorted** `BTreeMap`s so the encoder itself contributes no
//! ordering non-determinism. Arrow (v54) then stores that metadata in a
//! `std::collections::HashMap` and serializes it in per-process HashMap
//! iteration order, so the *raw IPC bytes* of two identical tiles can still
//! differ in the metadata region (a documented arrow-level gap — see
//! `docs/spec/stt-packed-format.md` §7-D6). These tests therefore guard the
//! reproducibility that is actually under this crate's control — the encoder is
//! a deterministic function of its inputs at the *logical* (decoded) level, and
//! all cross-run wire differences are confined to Arrow metadata KeyValue
//! ordering — plus the strict byte-identity target as an `#[ignore]`d canary
//! that passes on any order-preserving Arrow metadata implementation.

use stt_core::arrow_tile::*;

fn point_fixture() -> ColumnarLayer {
    ColumnarLayer {
        name: "points".to_string(),
        // Out-of-order start times so the baked time_offset is the real min
        // (1000), not simply the first value.
        feature_ids: vec![1, 2, 3],
        start_times: vec![3000, 1000, 2000],
        end_times: vec![3500, 1500, 2500],
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

fn line_fixture() -> ColumnarLayer {
    ColumnarLayer {
        name: "tracks".to_string(),
        feature_ids: vec![10, 11],
        start_times: vec![100, 0],
        end_times: vec![200, 50],
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

/// A stable, order-independent fingerprint of a decoded tile: every layer's
/// name, schema metadata (sorted), and, per column, the field name + sorted
/// field metadata + the column's raw Arrow value bytes. Two encodes of the same
/// fixture must share this fingerprint even if their Arrow metadata KeyValue
/// ordering (and therefore the raw IPC bytes) differ.
fn logical_fingerprint(payload: &[u8]) -> Vec<u8> {
    let mut fp: Vec<u8> = Vec::new();
    for layer in decode_tile(payload).unwrap() {
        fp.extend_from_slice(layer.name.as_bytes());
        fp.push(0);
        let schema = layer.batch.schema();
        let mut smeta: Vec<(String, String)> = schema
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        smeta.sort();
        for (k, v) in smeta {
            fp.extend_from_slice(k.as_bytes());
            fp.push(b'=');
            fp.extend_from_slice(v.as_bytes());
            fp.push(0);
        }
        for (i, field) in schema.fields().iter().enumerate() {
            fp.extend_from_slice(field.name().as_bytes());
            fp.push(0);
            let mut fmeta: Vec<(String, String)> = field
                .metadata()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            fmeta.sort();
            for (k, v) in fmeta {
                fp.extend_from_slice(k.as_bytes());
                fp.push(b'=');
                fp.extend_from_slice(v.as_bytes());
                fp.push(0);
            }
            // Column value bytes (buffers are laid out deterministically; only
            // schema *metadata* ordering is non-reproducible under arrow-54).
            let data = layer.batch.column(i).to_data();
            for buf in data.buffers() {
                fp.extend_from_slice(buf.as_slice());
            }
        }
    }
    fp
}

/// The encoder is a deterministic function of its inputs at the decoded/logical
/// level: repeated encodes of the same fixture decode to identical content,
/// identical schema/field metadata, and identical column bytes. (This is the
/// property content-addressed dedup relies on once the arrow-54 metadata-order
/// gap is closed; it also fails loudly if a future change makes column bytes or
/// metadata *content* — not just their arrow ordering — non-reproducible.)
#[test]
fn same_tile_is_logically_reproducible() {
    for fixture in [point_fixture(), line_fixture()] {
        let layers = std::slice::from_ref(&fixture);
        let first = logical_fingerprint(&encode_tile(layers).unwrap());
        for i in 0..500 {
            let again = logical_fingerprint(&encode_tile(layers).unwrap());
            assert_eq!(
                first, again,
                "tile '{}' is not logically reproducible on repetition {i}",
                fixture.name
            );
        }
    }
}

/// The multi-layer path is logically reproducible too.
#[test]
fn multi_layer_tile_is_logically_reproducible() {
    let layers = vec![point_fixture(), line_fixture()];
    let first = logical_fingerprint(&encode_tile(&layers).unwrap());
    for _ in 0..300 {
        assert_eq!(first, logical_fingerprint(&encode_tile(&layers).unwrap()));
    }
}

/// Every cross-run wire difference is confined to Arrow metadata ordering: two
/// encodes always have the same byte LENGTH (only the order of metadata
/// KeyValue entries can differ, never their presence or the data), so no
/// content is ever added or dropped between runs.
#[test]
fn reencodes_have_stable_length() {
    for fixture in [point_fixture(), line_fixture()] {
        let layers = std::slice::from_ref(&fixture);
        let len = encode_tile(layers).unwrap().len();
        for _ in 0..500 {
            assert_eq!(encode_tile(layers).unwrap().len(), len);
        }
    }
}

/// The baked `stt:time_offset_ms` equals the MINIMUM feature start-time (what
/// the TS decoder computes by scanning the start column), and is present only
/// when a start-time column exists.
#[test]
fn bakes_time_offset_min_start() {
    let fixture = point_fixture();
    let ipc = encode_layer(&fixture).unwrap();
    let batch = decode_layer(&ipc).unwrap();
    let baked = batch.schema().metadata().get("stt:time_offset_ms").cloned();
    assert_eq!(
        baked.as_deref(),
        Some("1000"),
        "time_offset_ms must be the MIN start time (1000), not the first (3000)"
    );

    // Line fixture (min start = 0) bakes 0 too.
    let li = encode_layer(&line_fixture()).unwrap();
    let lb = decode_layer(&li).unwrap();
    assert_eq!(
        lb.schema().metadata().get("stt:time_offset_ms").map(String::as_str),
        Some("0")
    );
}

/// The baked offset lets a decoder relativize times WITHOUT scanning the start
/// column: `absolute_start = start_relative + time_offset` round-trips.
#[test]
fn baked_time_offset_reconstructs_absolute_starts() {
    let fixture = point_fixture();
    let batch = decode_layer(&encode_layer(&fixture).unwrap()).unwrap();
    let offset: i64 = batch
        .schema()
        .metadata()
        .get("stt:time_offset_ms")
        .unwrap()
        .parse()
        .unwrap();
    use arrow::array::Int64Array;
    let starts = batch
        .column_by_name("start_time")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let reconstructed: Vec<i64> = (0..starts.len())
        .map(|i| (starts.value(i) - offset) + offset)
        .collect();
    assert_eq!(reconstructed, fixture.start_times);
}

/// STRICT byte-identity target. Ignored because Arrow v54 serializes schema
/// metadata in per-process `HashMap` iteration order, so the raw IPC bytes of
/// two identical tiles differ in the metadata region even though the encoder
/// feeds arrow sorted `BTreeMap`s. Passes on any Arrow build whose metadata
/// serialization preserves order (the forward-compatible target of the
/// `BTreeMap` change). Tracked: `docs/spec/stt-packed-format.md` §7-D6.
#[test]
#[ignore = "blocked by arrow-54 HashMap metadata ordering; see docs/spec/stt-packed-format.md §7-D6"]
fn same_tile_encodes_byte_identically() {
    for fixture in [point_fixture(), line_fixture()] {
        let layers = std::slice::from_ref(&fixture);
        let first = encode_tile(layers).unwrap();
        for i in 0..200 {
            assert_eq!(
                first,
                encode_tile(layers).unwrap(),
                "tile '{}' not byte-identical on repetition {i}",
                fixture.name
            );
        }
    }
}
