//! End-to-end cover for the wasm-facing API, run on the HOST.
//!
//! A wasm-only test needs a browser (or node) runner and `wasm-bindgen-test`;
//! that buys coverage of the wasm-bindgen glue, not of the decode. The decode
//! is what can be wrong, so the crate is built `crate-type = ["cdylib",
//! "rlib"]` and these tests drive the exact exported entry points
//! (`SttArchive::open` / `loadDirectory` / `tile` / `decodeTile`) natively.
//! Only the error paths go through [`Archive`] instead: `JsError` construction
//! calls a wasm-bindgen import that is not linked on a host build.
//!
//! Every fixture is written by the real `PackWriter`, so a format change lands
//! here rather than in a broken published artifact — and `parity_*` compares
//! this crate's byte-slice reader against `PackedReader` batch-for-batch,
//! which is what pins the open-time checks the wasm reader has to re-state
//! (stt-core keeps them private to `pack.rs`).

use std::fs;
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::ipc::reader::StreamReader;
use stt_core::arrow_tile::{encode_tile_with, ColumnarLayer, GeometryColumn, PropertyColumn};
use stt_core::curve::BlobOrdering;
use stt_core::metadata::Metadata;
use stt_core::pack::PACKED_FORMAT_VERSION;
use stt_core::types::TimeRange;
use stt_core::{PackWriter, PackedReader, TileId};
use stt_wasm::{Archive, SttArchive};
use tempfile::TempDir;

const T0: i64 = 1_700_000_000_000;
const BUCKET: i64 = 3_600_000;

/// A three-point layer with a numeric and a categorical property, one of each
/// null — the shapes that exercise the property-column decode paths.
fn points_layer() -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "points".to_string(),
        feature_ids: vec![1, 2, 3],
        start_times: vec![T0 + 3000, T0 + 1000, T0 + 2000],
        end_times: vec![T0 + 3500, T0 + 1500, T0 + 2500],
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

/// A timed-linestring layer — a second layer in the same tile, so the tests
/// see a multi-layer frame rather than the trivial one-layer case.
fn tracks_layer() -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "tracks".to_string(),
        feature_ids: vec![10, 11],
        start_times: vec![T0 + 100, T0],
        end_times: vec![T0 + 200, T0 + 50],
        geometry: GeometryColumn::LineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            vec![[5.0, 5.0], [6.0, 6.0]],
        ]),
        vertex_times: Some(vec![vec![T0, T0 + 25, T0 + 50], vec![T0 + 100, T0 + 200]]),
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

/// Write a two-tile dataset with the real writer. `paging` opts into the paged
/// directory container; `format_version` selects the manifest/frame version.
fn build_dataset(format_version: u32, paging: Option<usize>) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = PackWriter::create(dir.path(), BlobOrdering::SpatialMajor, 64 * 1024)
        .unwrap()
        .with_format_version(format_version)
        .with_paging(paging);

    // The encoder settings MUST come from the writer: frame version and
    // template sink are the coupling that keeps a dataset's frames and its
    // manifest declaration in lockstep (mixed versions are a reader hard
    // error, and unrecorded v2 templates fail resolution at read time).
    let cfg = writer.encoder_config();
    let tiles = [
        (
            TileId::new(5, 5, 12, T0 as u64),
            T0,
            T0 + BUCKET - 1,
            5u32,
            encode_tile_with(&[points_layer(), tracks_layer()], &cfg).unwrap(),
        ),
        (
            TileId::new(5, 6, 12, (T0 + BUCKET) as u64),
            T0 + BUCKET,
            T0 + 2 * BUCKET - 1,
            3,
            encode_tile_with(&[points_layer()], &cfg).unwrap(),
        ),
    ];
    for (id, ts, te, n, payload) in &tiles {
        writer
            .add_tile_full(id, *ts, *te, Some(*ts), *n, Some(BUCKET as u64), payload)
            .unwrap();
    }
    let meta = Metadata::new("stt-wasm-fixture")
        .with_zoom_levels(5, 5)
        .with_temporal_bucket_ms(BUCKET as u64)
        .with_time_range(TimeRange::new(T0 as u64, (T0 + 2 * BUCKET - 1) as u64));
    writer.finalize(&meta).unwrap();
    dir
}

/// Manifest bytes + directory-object bytes for a dataset on disk — everything
/// the reader needs to get to a tile list. Standing in for the host's fetch.
fn open_from_bytes(root: &Path) -> Archive {
    let manifest_bytes = fs::read(root.join("manifest.json")).unwrap();
    let mut archive = Archive::open(&manifest_bytes).unwrap();
    let dir_bytes = fs::read(root.join(archive.directory_key())).unwrap();
    archive.load_directory(&dir_bytes).unwrap();
    archive
}

/// The blob range for tile `index`, read out of its pack object — the bytes an
/// HTTP host would get back from one `Range:` request.
fn blob_for(archive: &Archive, root: &Path, index: usize) -> Vec<u8> {
    let tile = archive.tile(index).unwrap();
    let pack = fs::read(root.join(archive.pack_key(tile.pack_id() as usize).unwrap())).unwrap();
    let start = tile.offset() as usize;
    pack[start..start + tile.length() as usize].to_vec()
}

/// The message of a call that must fail. (`unwrap_err` would need `Debug` on
/// the success type, and `Archive` deliberately has none — a `Debug` on it
/// would print the whole tile directory.)
fn refusal<T>(result: stt_core::error::Result<T>) -> String {
    match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected a refusal, got a successful read"),
    }
}

/// Parse one Arrow IPC stream back into its single record batch.
fn batch_from_ipc(ipc: &[u8]) -> RecordBatch {
    let mut reader = StreamReader::try_new(ipc, None).unwrap();
    let batch = reader.next().expect("IPC stream carries a batch").unwrap();
    assert!(reader.next().is_none(), "a layer is exactly one batch");
    batch
}

/// The headline case: drive the EXPORTED wasm API end to end and get Arrow out.
#[test]
fn wasm_facade_decodes_a_real_tile() {
    let dataset = build_dataset(PACKED_FORMAT_VERSION, None);
    let root = dataset.path();

    let manifest_bytes = fs::read(root.join("manifest.json")).unwrap();
    let mut archive = SttArchive::open(&manifest_bytes).unwrap();
    assert_eq!(archive.format_version(), 2);
    assert!(archive
        .metadata_json()
        .unwrap()
        .contains("stt-wasm-fixture"));

    let dir_bytes = fs::read(root.join(archive.directory_key())).unwrap();
    assert_eq!(archive.directory_length() as usize, dir_bytes.len());
    let n = archive.load_directory(&dir_bytes).unwrap();
    assert_eq!(n, 2);
    assert_eq!(archive.tile_count(), 2);

    // The tile list a host would index on.
    let listed: serde_json::Value = serde_json::from_str(&archive.tiles_json().unwrap()).unwrap();
    assert_eq!(listed.as_array().unwrap().len(), 2);

    // Locate the two-layer tile by its address, then fetch exactly its blob.
    let index = (0..archive.tile_count())
        .find(|i| {
            let t = archive.tile(*i).unwrap();
            (t.zoom(), t.x(), t.y()) == (5, 5, 12)
        })
        .expect("the z5/5/12 tile is in the directory");
    let info = archive.tile(index).unwrap();
    assert_eq!(info.zoom(), 5);
    assert_eq!(info.time_start(), T0 as f64);
    assert_eq!(info.feature_count(), 5);
    assert!(archive
        .pack_key(info.pack_id() as usize)
        .unwrap()
        .starts_with("packs/"));

    let pack = fs::read(root.join(archive.pack_key(info.pack_id() as usize).unwrap())).unwrap();
    let start = info.offset() as usize;
    let blob = &pack[start..start + info.length() as usize];

    let tile = archive.decode_tile(index, blob).unwrap();
    assert_eq!(tile.layer_count(), 2);
    assert_eq!(tile.layer_name(0).unwrap(), "points");
    assert_eq!(tile.layer_name(1).unwrap(), "tracks");

    let points = batch_from_ipc(&tile.layer_ipc(0).unwrap());
    assert_eq!(points.num_rows(), 3);
    let tracks = batch_from_ipc(&tile.layer_ipc(1).unwrap());
    assert_eq!(tracks.num_rows(), 2);

    // Whole-pack entry point must agree with the ranged one, byte for byte.
    let in_pack = archive.decode_tile_in_pack(index, &pack).unwrap();
    assert_eq!(in_pack.layer_ipc(0).unwrap(), tile.layer_ipc(0).unwrap());
}

/// Batch-for-batch equality with `PackedReader` over every tile and layer.
///
/// This is the test that earns the duplicated open-time logic: the manifest
/// refusals, the template registry and the frame-version dispatch are private
/// to `pack.rs`, so the wasm reader re-states them. Any drift between the two
/// shows up here as a decode difference instead of as a wrong map.
fn assert_parity(format_version: u32, paging: Option<usize>) {
    let dataset = build_dataset(format_version, paging);
    let root = dataset.path();

    let reference = PackedReader::open(root.join("manifest.json")).unwrap();
    let archive = open_from_bytes(root);
    assert_eq!(archive.tile_count(), reference.entries().len());
    assert!(archive.tile_count() > 0);

    for (index, entry) in reference.entries().iter().enumerate() {
        assert_eq!(archive.tile(index).unwrap().entry(), entry);

        let expected = reference.read_layers(entry).unwrap();
        let blob = blob_for(&archive, root, index);
        let actual = archive.decode_tile(index, &blob).unwrap();
        assert_eq!(actual.len(), expected.len(), "layer count, tile {index}");
        for (got, want) in actual.iter().zip(&expected) {
            assert_eq!(got.name, want.name);
            assert_eq!(
                batch_from_ipc(&got.ipc),
                want.batch,
                "layer {:?} of tile {index}",
                want.name
            );
        }
    }
}

#[test]
fn parity_v2_single_directory() {
    assert_parity(PACKED_FORMAT_VERSION, None);
}

#[test]
fn parity_v2_paged_directory() {
    // One entry per page, so the fixture really exercises the root + leaf walk.
    assert_parity(PACKED_FORMAT_VERSION, Some(1));
}

/// A short read is the commonest host bug (truncated range response, or a CDN
/// error page served with 200). It must name the fetch, not the archive —
/// without the length guards it reaches the CRC check and reports "corrupt
/// pack", sending the reader after the wrong thing entirely.
#[test]
fn short_reads_blame_the_fetch_not_the_archive() {
    let dataset = build_dataset(PACKED_FORMAT_VERSION, None);
    let root = dataset.path();

    let manifest_bytes = fs::read(root.join("manifest.json")).unwrap();
    let mut archive = Archive::open(&manifest_bytes).unwrap();
    let dir_bytes = fs::read(root.join(archive.directory_key())).unwrap();

    let err = refusal(archive.load_directory(&dir_bytes[..dir_bytes.len() - 4]));
    assert!(err.contains("manifest declared"), "got: {err}");

    archive.load_directory(&dir_bytes).unwrap();
    let blob = blob_for(&archive, root, 0);
    let err = refusal(archive.decode_tile(0, &blob[..blob.len() - 1]));
    assert!(err.contains("directory declared"), "got: {err}");
}

/// The open-time refusals are the reason `Archive::open` is not a `serde_json`
/// call: a capability this reader does not implement RE-TYPES existing columns,
/// so accepting it would silently misdecode every tile rather than fail.
#[test]
fn open_refuses_manifests_it_cannot_read() {
    let dataset = build_dataset(PACKED_FORMAT_VERSION, None);
    let manifest = fs::read(dataset.path().join("manifest.json")).unwrap();
    let text = String::from_utf8(manifest).unwrap();

    let unknown_capability = text.replacen(
        "\"formatVersion\"",
        "\"capabilities\": [\"stt:from-the-future\"],\n  \"formatVersion\"",
        1,
    );
    let err = refusal(Archive::open(unknown_capability.as_bytes()));
    assert!(
        err.contains("capabilities this reader does not implement"),
        "got: {err}"
    );

    let future_version = text.replacen("\"formatVersion\": 2", "\"formatVersion\": 99", 1);
    let err = refusal(Archive::open(future_version.as_bytes()));
    assert!(
        err.contains("unsupported packed formatVersion 99"),
        "got: {err}"
    );

    let err = refusal(Archive::open(b"{\"format\":\"mvt\"}"));
    assert!(err.contains("manifest JSON decode failed"), "got: {err}");
}

/// A pack object handed over whole is checked for its self-identifying magic,
/// so passing the wrong object is a named error and not a CRC failure that
/// reads like archive corruption.
#[test]
fn wrong_object_in_the_whole_pack_path_is_named() {
    let dataset = build_dataset(PACKED_FORMAT_VERSION, None);
    let root = dataset.path();
    let archive = open_from_bytes(root);

    let tile = archive.tile(0).unwrap();
    let pack_id = tile.pack_id() as usize;
    let declared = archive.pack_length(pack_id).unwrap() as usize;
    let mut not_a_pack = vec![0u8; declared];
    not_a_pack[..4].copy_from_slice(b"STTD");
    let err = refusal(archive.decode_tile_in_pack(0, &not_a_pack));
    assert!(err.contains("STTP"), "got: {err}");
}
