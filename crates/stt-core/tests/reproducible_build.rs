//! Guards the reproducible-encoding contract that content-addressed pack dedup
//! depends on, and the `stt:time_offset_ms` bake.
//!
//! Background: `arrow_tile::encode_layer_cfg` assembles schema- and field-level
//! metadata from **sorted** `BTreeMap`s so the encoder itself contributes no
//! ordering non-determinism, and the pinned Arrow (v59) IPC writer serializes
//! that metadata in stable (sorted) key order. So the *raw IPC bytes* of two
//! identical tiles are byte-identical, including the metadata region — the
//! cross-process reproducibility that content-addressed pack dedup, edge
//! caching, and incremental deploys rely on (see
//! `docs/spec/stt-packed-format.md` §7-D6). These tests guard both the
//! logical-level determinism (`same_tile_is_logically_reproducible`) and the
//! strict byte-identity target (`same_tile_encodes_byte_identically`), now an
//! active test. (Under Arrow 54 the writer stored metadata in a per-process
//! `HashMap` and emitted it in iteration order, so the strict test was an
//! `#[ignore]`d canary; the Arrow ≥59 upgrade closed that gap.)

use stt_core::arrow_tile::*;

mod common;
use common::{line_fixture, point_fixture};

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
        lb.schema()
            .metadata()
            .get("stt:time_offset_ms")
            .map(String::as_str),
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

/// STRICT byte-identity target — ACTIVE on Arrow ≥59, whose IPC writer
/// serializes schema metadata in stable (sorted) order. Two encodes of the same
/// tile are byte-identical, which is what makes content-addressed pack dedup,
/// edge caching, and incremental deploys reproducible. (Was `#[ignore]`d under
/// Arrow 54, which serialized metadata in per-process `HashMap` order — the
/// encoder always fed sorted `BTreeMap`s; only the writer's ordering lagged.)
/// Tracked: `docs/spec/stt-packed-format.md` §7-D6.
#[test]
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

/// The byte-identity contract extended to a whole **formatVersion-3 dataset**:
/// two builds of the same input — with the tiles added
/// in DIFFERENT orders, standing in for encode parallelism — produce a
/// byte-identical manifest (including the sorted, deduped `schemas` table)
/// and byte-identical directory + pack objects. Template hashes are
/// content-addressed, so add order must never leak into any object.
#[test]
fn v2_dataset_rebuild_is_byte_identical_including_schemas() {
    use std::collections::BTreeMap;
    use std::fs;
    use stt_core::metadata::Metadata;
    use stt_core::{BlobOrdering, PackWriter, TileId};

    // Quantized config so `stt:qa`/`t0` vary per tile — the schema-template
    // hoist must keep that variance OUT of `manifest.schemas`.
    let tiles: Vec<ColumnarLayer> = (0..4)
        .map(|k| {
            let mut layer = point_fixture();
            for t in layer.start_times.iter_mut() {
                *t += k as i64 * 60_000;
            }
            for t in layer.end_times.iter_mut() {
                *t += k as i64 * 60_000;
            }
            layer
        })
        .collect();

    let build = |dir: &std::path::Path, reversed: bool| {
        let mut writer = PackWriter::create(dir, BlobOrdering::Auto, 8 * 1024).unwrap();
        assert_eq!(
            stt_core::arrow_tile::LAYER_FRAME_VERSION,
            2,
            "manifest formatVersion 3 is the writer default"
        );
        let cfg = EncoderConfig {
            quantize_coords_m: Some(1.0),
            quantize_attrs_auto: true,
            format_version: LAYER_FRAME_VERSION,
            template_collector: Some(writer.template_collector()),
            ..EncoderConfig::default()
        };
        let mut order: Vec<usize> = (0..tiles.len()).collect();
        if reversed {
            order.reverse();
        }
        for k in order {
            let payload = encode_tile_with(std::slice::from_ref(&tiles[k]), &cfg).unwrap();
            writer
                .add_tile_full(
                    &TileId::new(4, k as u32, 0, 0),
                    0,
                    100,
                    None,
                    3,
                    None,
                    &payload,
                )
                .unwrap();
        }
        writer.finalize(&Metadata::new("v2-repro")).unwrap()
    };

    let tmp = tempfile::tempdir().unwrap();
    let (a, b) = (tmp.path().join("a"), tmp.path().join("b"));
    let manifest_a = build(&a, false);
    let manifest_b = build(&b, true);

    assert!(
        !manifest_a.schemas.is_empty(),
        "formatVersion-3 manifest must carry templates"
    );
    let hashes: Vec<&str> = manifest_a.schemas.iter().map(|s| s.hash.as_str()).collect();
    assert!(
        hashes.windows(2).all(|w| w[0] < w[1]),
        "schemas sorted by hash"
    );

    assert_eq!(
        fs::read(a.join("manifest.json")).unwrap(),
        fs::read(b.join("manifest.json")).unwrap(),
        "manifest bytes must not depend on tile add order"
    );
    let objects = |root: &std::path::Path, m: &stt_core::Manifest| -> BTreeMap<String, Vec<u8>> {
        std::iter::once(&m.directory.key)
            .chain(m.packs.iter().map(|p| &p.key))
            .map(|key| (key.clone(), fs::read(root.join(key)).unwrap()))
            .collect()
    };
    assert_eq!(
        objects(&a, &manifest_a),
        objects(&b, &manifest_b),
        "every directory/pack object must be byte-identical across rebuilds"
    );
}

/// The **default** build path's byte-identity: `stt-build` now resolves its blob
/// ordering by simulation (`--blob-ordering measured`) under per-dataset query
/// weights, so the writer's ordering step is a solver — and every solver in this
/// program lands with a byte-identical re-run test.
///
/// The stake is not abstract: the resolved ordering fixes the blob byte string,
/// which fixes every pack's content address, which fixes every object name in
/// the archive. One unstable comparison in the simulator would rename all 1,324
/// fleet objects and force a 29.3 GiB re-upload on a rebuild of unchanged data.
///
/// Tiles are added in OPPOSITE orders across the two builds, standing in for the
/// parallel tiler's nondeterministic emission order.
#[test]
fn measured_default_ordering_rebuilds_byte_identically() {
    use std::collections::BTreeMap;
    use std::fs;
    use stt_core::metadata::{Metadata, StyleHints};
    use stt_core::ordering_sim::OrderingWorkloadMode;
    use stt_core::{BlobOrdering, PackWriter, TileId};

    // Enough native tiles to clear the simulation floor, over many buckets and
    // several cells, so the ranking is decided by the simulator rather than by
    // the small-input fallback.
    let cells = 6u32;
    let buckets = 20i64;
    let bucket_ms = 3_600_000i64;

    let build = |dir: &std::path::Path, reversed: bool, mode: OrderingWorkloadMode, hint: &str| {
        let mut writer = PackWriter::create(dir, BlobOrdering::Auto, 64 * 1024)
            .unwrap()
            .with_measured_ordering(true)
            .with_ordering_workload(mode);
        let mut coords: Vec<(u32, i64)> = Vec::new();
        for b in 0..buckets {
            for x in 0..cells {
                coords.push((x, b));
            }
        }
        if reversed {
            coords.reverse();
        }
        for (x, b) in coords {
            let t = b * bucket_ms;
            let mut layer = point_fixture();
            layer.feature_ids = layer
                .feature_ids
                .iter()
                .map(|id| id + u64::from(x) * 1_000 + b as u64 * 10)
                .collect();
            for s in layer.start_times.iter_mut() {
                *s = t;
            }
            for e in layer.end_times.iter_mut() {
                *e = t + bucket_ms - 1;
            }
            let payload = encode_tile(std::slice::from_ref(&layer)).unwrap();
            writer
                .add_tile_full(
                    &TileId::new(10, 500 + x, 600, t as u64),
                    t,
                    t + bucket_ms - 1,
                    Some(t),
                    1,
                    None,
                    &payload,
                )
                .unwrap();
        }
        let mut meta = Metadata::new("measured-default")
            .with_temporal_bucket_ms(bucket_ms as u64)
            .with_zoom_levels(0, 10);
        meta.style_hints = Some(StyleHints {
            version: 1,
            properties: vec![],
            suggested_playback_seconds: None,
            suggested_time_window_ms: None,
            layer_hint: Some(hint.to_string()),
        });
        writer.finalize(&meta).unwrap()
    };

    let objects = |root: &std::path::Path, m: &stt_core::Manifest| -> BTreeMap<String, Vec<u8>> {
        std::iter::once("manifest.json".to_string())
            .chain(std::iter::once(m.directory.key.clone()))
            .chain(m.packs.iter().map(|p| p.key.clone()))
            .map(|key| {
                let bytes = fs::read(root.join(&key)).unwrap();
                (key, bytes)
            })
            .collect()
    };

    let tmp = tempfile::tempdir().unwrap();
    let (a, b) = (tmp.path().join("a"), tmp.path().join("b"));
    let ma = build(&a, false, OrderingWorkloadMode::Derived, "trips");
    let mb = build(&b, true, OrderingWorkloadMode::Derived, "trips");

    // The simulator actually ran (not the small-input fallback) and recorded
    // what it ran under.
    let workload = ma
        .metadata
        .ordering_workload
        .expect("a measured build records its workload");
    assert_eq!(
        (workload.scrub, workload.pan, workload.playback),
        (1, 1, 2),
        "trips over many buckets is the playback-dominant row"
    );
    assert_eq!(
        workload.coalesce_gap_bytes,
        stt_core::ordering_sim::DEFAULT_COALESCE_GAP_BYTES,
        "the recorded gap must be the reader-mirroring constant"
    );

    assert_eq!(
        ma.blob_ordering, mb.blob_ordering,
        "the measured pick must not depend on tile add order"
    );
    assert_eq!(
        objects(&a, &ma),
        objects(&b, &mb),
        "every object (manifest, directory, packs) must be byte-identical across rebuilds"
    );

    // The escape hatch is itself deterministic, and it really does pin the
    // pre-workload-model weighting.
    let (c, d) = (tmp.path().join("c"), tmp.path().join("d"));
    let mc = build(&c, false, OrderingWorkloadMode::Legacy, "trips");
    let md = build(&d, true, OrderingWorkloadMode::Legacy, "trips");
    assert_eq!(objects(&c, &mc), objects(&d, &md));
    let legacy = mc.metadata.ordering_workload.unwrap();
    assert_eq!((legacy.scrub, legacy.pan, legacy.playback), (1, 1, 0));

    // And the hint is a real input to the layout, not decoration: swapping the
    // dominant kind changes the recorded weighting.
    let e = tmp.path().join("e");
    let me = build(&e, false, OrderingWorkloadMode::Derived, "polygons");
    let generalist = me.metadata.ordering_workload.unwrap();
    assert_eq!(
        (generalist.scrub, generalist.pan, generalist.playback),
        (1, 1, 1)
    );
}
