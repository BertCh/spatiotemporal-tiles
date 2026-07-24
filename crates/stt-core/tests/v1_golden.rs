//! Golden byte-stability pin for the **formatVersion 1** writer (packed v2
//! design doc `docs/roadmap/stt-packed-format-decisions.md`, ★F3).
//!
//! THE CONTRACT (reworded 2026-07): v1 mode is **0.3.x-READER-compatible v1
//! emission** — v1 frames, no object magic, `formatVersion: 1`, no `schemas`
//! — pinned **byte-stable against the CURRENT writer** by the committed
//! fixture under `tests/fixtures/v1-golden/`: `--format-version 1` (i.e.
//! `PackWriter` v1 mode + the encoder's v1 frame branch) MUST keep
//! reproducing these exact bytes — every object AND the manifest — so the
//! kill switch can never drift. Two deliberate deviations from a literal
//! "byte-identical to the 0.3.0 writer" reading:
//!   * Additive manifest fields the current writer emits (`capabilities`,
//!     `blobOrdering`) are IGNORED by 0.3.x readers under the manifest's
//!     open-envelope rule (spec §9.1) — reader compatibility, not bit-parity
//!     with a historical binary, is what F3 guarantees.
//!   * Bit-parity with the 0.3.0 writer was already broken by the deliberate
//!     FNV-1a synthetic-id migration (spec §9.3 "v1, builder-behavior") and
//!     the `Auto`-ordering occupied-extent fix; this fixture contains
//!     post-0.3.0 bytes and pins the writer as it stands.
//!
//! Two datasets cover both directory container shapes the writer emits:
//!   * `single/` — whole-load v5 directory.
//!   * `paged/`  — root + leaf pages (`with_paging`).
//!
//! Tile content exercises the v1 surfaces the v2 break touches: a quantized
//! point layer (coord quant + range-adaptive attr quant → per-tile `stt:qa` /
//! `stt:quant` metadata), a categorical property (dictionary column), a
//! trajectory layer with u16-delta `vertex_time` (per-tile origin/step schema
//! metadata), and an empty-bucket tile (zero rows, dictionary column intact).
//!
//! `expected-hashes.json` records the blake3-128 of every object so drift is
//! reported by name even when a byte-diff would be opaque.
//!
//! Regenerate (ONLY for an intentional, spec-versioned change to v1 emission
//! — which by contract must not happen; v1 output is frozen going forward):
//!   cargo test -p stt-core --test v1_golden -- --ignored regenerate_v1_golden

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use stt_core::arrow_tile::{
    encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn, PropertyColumn,
};
use stt_core::metadata::Metadata;
use stt_core::pack::{CAPABILITY_ATTR_QUANT, CAPABILITY_COORD_QUANT};
use stt_core::{BlobOrdering, PackWriter, TileId};

/// blake3 content address, 128-bit → 32 lowercase hex chars (the packed
/// format's object-address convention).
fn blake3_128_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).as_bytes()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("v1-golden")
}

/// The production-like encoder settings the fixture pins: world-grid coordinate
/// quantization at 1 m + range-adaptive UInt16 attribute quantization — the
/// config that makes `stt:qa` / `stt:time_offset_ms` per-tile-varying (the v1
/// schema-tax shape the v2 break exists to fix, and exactly what the
/// `--format-version 1` kill switch must keep byte-identical).
fn golden_encoder_config() -> EncoderConfig {
    EncoderConfig {
        quantize_coords_m: Some(1.0),
        quantize_attrs_auto: true,
        // Explicit (even though it's the default): the frames must stay v1.
        format_version: stt_core::arrow_tile::FORMAT_VERSION_V1,
        ..EncoderConfig::default()
    }
}

/// Quantized point layer: numeric `speed` (with a null) + categorical `kind`
/// (with a null). `seed` shifts ids/times/values so distinct tiles get
/// distinct `stt:qa` affines and `stt:time_offset_ms` origins.
fn point_layer(seed: u64, n: usize) -> ColumnarLayer {
    let base = 1_700_000_000_000i64 + seed as i64 * 3_600_000;
    let cats = ["car", "bus", "tram"];
    ColumnarLayer {
        name: "default".to_string(),
        feature_ids: (0..n as u64).map(|i| seed * 1000 + i).collect(),
        start_times: (0..n as i64).map(|i| base + i * 1000).collect(),
        end_times: (0..n as i64).map(|i| base + i * 1000 + 60_000).collect(),
        geometry: GeometryColumn::Point(
            (0..n)
                .map(|i| [-122.4 + seed as f64 * 0.01 + i as f64 * 1e-4, 37.7 + i as f64 * 5e-5])
                .collect(),
        ),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "speed".to_string(),
                PropertyColumn::Numeric(
                    (0..n)
                        .map(|i| {
                            if i == 1 {
                                None
                            } else {
                                Some(seed as f64 * 7.5 + i as f64 * 3.25)
                            }
                        })
                        .collect(),
                ),
            ),
            (
                "kind".to_string(),
                PropertyColumn::Categorical(
                    (0..n)
                        .map(|i| {
                            if i == 2 {
                                None
                            } else {
                                Some(cats[i % cats.len()].to_string())
                            }
                        })
                        .collect(),
                ),
            ),
        ],
    }
}

/// Trajectory layer with a tight per-vertex time span → the u16-delta
/// `vertex_time` encoding (per-layer origin/step in schema metadata).
fn trajectory_layer() -> ColumnarLayer {
    let base = 1_700_003_600_000i64;
    ColumnarLayer {
        name: "tracks".to_string(),
        feature_ids: vec![9001, 9002],
        start_times: vec![base, base + 5_000],
        end_times: vec![base + 30_000, base + 42_000],
        geometry: GeometryColumn::LineString(vec![
            vec![[-122.40, 37.70], [-122.41, 37.71], [-122.42, 37.72]],
            vec![[-122.50, 37.75], [-122.51, 37.76]],
        ]),
        vertex_times: Some(vec![
            vec![base, base + 15_000, base + 30_000],
            vec![base + 5_000, base + 42_000],
        ]),
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "speed".to_string(),
            PropertyColumn::Numeric(vec![Some(12.5), Some(30.0)]),
        )],
    }
}

/// Empty-bucket tile: zero rows, but the full column set (numeric +
/// categorical) — a dictionary column at n=0 still carries a DictionaryBatch.
fn empty_layer() -> ColumnarLayer {
    ColumnarLayer {
        name: "default".to_string(),
        feature_ids: vec![],
        start_times: vec![],
        end_times: vec![],
        geometry: GeometryColumn::Point(vec![]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            ("speed".to_string(), PropertyColumn::Numeric(vec![])),
            ("kind".to_string(), PropertyColumn::Categorical(vec![])),
        ],
    }
}

/// Build one golden dataset (single or paged directory) into `out`.
///
/// Everything is deterministic: fixed payload content, `SpatialMajor` ordering
/// (independent of the Auto heuristic), fixed pack target, default zstd level.
fn build_v1_dataset(out: &Path, paging: Option<usize>) -> stt_core::Manifest {
    let cfg = golden_encoder_config();
    let bucket = 3_600_000i64;
    let t0 = 1_700_000_000_000i64 - 1_700_000_000_000i64 % bucket;

    // (id, time_start, time_end, feature_count, payload)
    let mut tiles: Vec<(TileId, i64, i64, u32, Vec<u8>)> = Vec::new();
    // Quantized point tiles across a few cells/buckets (distinct qa/t0 per tile).
    for k in 0..4u64 {
        let layer = point_layer(k, 5);
        let ts = t0 + (k as i64 % 2) * bucket;
        let payload = encode_tile_with(std::slice::from_ref(&layer), &cfg).unwrap();
        tiles.push((
            TileId::new(5, 5 + k as u32, 11, ts.max(0) as u64),
            ts,
            ts + bucket - 1,
            5,
            payload,
        ));
    }
    // Trajectory tile (vertex_time u16-delta path).
    let traj = encode_tile_with(&[trajectory_layer()], &cfg).unwrap();
    tiles.push((
        TileId::new(5, 9, 11, t0.max(0) as u64),
        t0,
        t0 + bucket - 1,
        2,
        traj,
    ));
    // Empty-bucket tile.
    let empty = encode_tile_with(&[empty_layer()], &cfg).unwrap();
    tiles.push((
        TileId::new(5, 10, 11, (t0 + bucket).max(0) as u64),
        t0 + bucket,
        t0 + 2 * bucket - 1,
        0,
        empty,
    ));

    let mut writer = PackWriter::create(out, BlobOrdering::SpatialMajor, 4 * 1024)
        .unwrap()
        // THE kill switch under test: v1 mode must keep reproducing the
        // committed fixture's bytes exactly (0.3.x-READER-compatible
        // emission, pinned byte-stable against the current writer — see the
        // header comment for why this is NOT 0.3.0-writer bit-parity).
        .with_format_version(stt_core::pack::PACKED_FORMAT_VERSION_V1)
        .with_paging(paging)
        .with_capabilities(vec![
            CAPABILITY_COORD_QUANT.to_string(),
            CAPABILITY_ATTR_QUANT.to_string(),
        ]);
    for (id, ts, te, n, payload) in &tiles {
        writer
            .add_tile_full(id, *ts, *te, Some(*ts), *n, Some(bucket as u64), payload)
            .unwrap();
    }
    let meta = Metadata::new("v1-golden")
        .with_description("Frozen formatVersion-1 byte-identity fixture")
        .with_zoom_levels(5, 5)
        .with_temporal_bucket_ms(bucket as u64)
        .with_time_range(stt_core::types::TimeRange::new(
            t0 as u64,
            (t0 + 2 * bucket - 1) as u64,
        ));
    writer.finalize(&meta).unwrap()
}

/// Recursively collect `relative path → file bytes` for a dataset directory.
fn collect_files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                out.insert(rel, fs::read(&path).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// Both dataset shapes, keyed by fixture subdirectory.
fn dataset_shapes() -> [(&'static str, Option<usize>); 2] {
    [("single", None), ("paged", Some(3))]
}

/// The v1 writer reproduces the committed golden corpus byte-for-byte: every
/// object file AND the manifest, plus the recorded blake3 of each object.
#[test]
fn v1_build_is_byte_identical_to_golden() {
    let fixture_root = fixture_dir();
    assert!(
        fixture_root.exists(),
        "missing fixture {} — the committed golden corpus pins the current \
         writer's v1 emission; regenerate ONLY for an intentional, \
         spec-versioned v1 change (see the header comment)",
        fixture_root.display()
    );
    let expected_hashes: BTreeMap<String, String> = serde_json::from_slice(
        &fs::read(fixture_root.join("expected-hashes.json")).unwrap(),
    )
    .unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let mut rebuilt_hashes: BTreeMap<String, String> = BTreeMap::new();
    for (name, paging) in dataset_shapes() {
        let out = tmp.path().join(name);
        build_v1_dataset(&out, paging);

        let golden = collect_files(&fixture_root.join(name));
        let rebuilt = collect_files(&out);
        assert_eq!(
            golden.keys().collect::<Vec<_>>(),
            rebuilt.keys().collect::<Vec<_>>(),
            "[{name}] object set drifted from the v1 golden fixture"
        );
        for (rel, bytes) in &rebuilt {
            assert_eq!(
                golden.get(rel).unwrap(),
                bytes,
                "[{name}] {rel}: bytes differ from the frozen v1 fixture \
                 (--format-version 1 is pinned byte-stable against the current \
                 writer — 0.3.x-READER-compatible emission must never drift)"
            );
            rebuilt_hashes.insert(format!("{name}/{rel}"), blake3_128_hex(bytes));
        }
    }
    assert_eq!(
        expected_hashes, rebuilt_hashes,
        "recorded blake3 pins drifted from the rebuilt v1 datasets"
    );
}

/// Fixture generator — regenerating REPLACES the pin, so run it ONLY for an
/// intentional, spec-versioned change to v1 emission (see header comment).
#[test]
#[ignore = "fixture regeneration replaces the v1 byte-stability pin"]
fn regenerate_v1_golden() {
    let fixture_root = fixture_dir();
    if fixture_root.exists() {
        fs::remove_dir_all(&fixture_root).unwrap();
    }
    fs::create_dir_all(&fixture_root).unwrap();

    let mut hashes: BTreeMap<String, String> = BTreeMap::new();
    for (name, paging) in dataset_shapes() {
        let out = fixture_root.join(name);
        build_v1_dataset(&out, paging);
        for (rel, bytes) in collect_files(&out) {
            hashes.insert(format!("{name}/{rel}"), blake3_128_hex(&bytes));
        }
    }
    fs::write(
        fixture_root.join("expected-hashes.json"),
        serde_json::to_vec_pretty(&hashes).unwrap(),
    )
    .unwrap();
    println!("wrote v1 golden corpus to {}", fixture_root.display());
}
