//! End-to-end tests for the `stt-validate` binary.
//!
//! Each test builds a tiny packed dataset in a temp dir (via the `stt_core`
//! writers + the single-file → packed transcode, the same pipeline
//! `stt-build --streaming-arrow` uses), then runs the compiled binary
//! (`CARGO_BIN_EXE_*`) over it with `--json` and asserts on the parsed report.
//! This exercises the real decode path, the schema checks, and the `--sample`
//! accounting end to end.

// The stt-validate binary (and CARGO_BIN_EXE_stt-validate) only exists when
// the `validate-cli` feature is on; compile the suite out otherwise.
#![cfg(feature = "validate-cli")]

use std::path::Path;
use std::process::Command;

use stt_core::arrow_tile::{encode_tile, ColumnarLayer, GeometryColumn, PropertyColumn};
use stt_core::metadata::Metadata;
use stt_core::types::{Compression, TimeRange};
use stt_core::{transcode_archive_to_packs, Archive, BlobOrdering, TileId};

/// Build a point layer of `n` features whose times fall inside [start, end].
fn point_layer(name: &str, base_id: u64, n: usize, start: i64, end: i64) -> ColumnarLayer {
    ColumnarLayer {
        name: name.into(),
        feature_ids: (0..n as u64).map(|i| base_id + i).collect(),
        start_times: vec![start; n],
        end_times: vec![end; n],
        geometry: GeometryColumn::Point((0..n).map(|i| [i as f64 * 0.01, i as f64 * 0.01]).collect()),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "speed".into(),
            PropertyColumn::Numeric((0..n).map(|i| Some(i as f64)).collect()),
        )],
    }
}

/// Write a valid packed dataset of `tile_count` tiles, each holding `per_tile`
/// point features, into the directory `out_dir`. Builds a temp single-file
/// archive and transcodes it to packs — the same staging pipeline
/// `stt-build --streaming-arrow` uses. Returns the total feature count so
/// callers can assert the metadata grand total matches.
fn write_dataset(out_dir: &Path, tile_count: u32, per_tile: usize) -> u64 {
    let staging = tempfile::Builder::new().suffix(".stt").tempfile().unwrap();
    let mut writer = Archive::create(staging.path(), Compression::Zstd).unwrap();
    let t_start = 1_000i64;
    let t_end = 2_000i64;
    let mut features = 0u64;
    for x in 0..tile_count {
        let id = TileId::new(8, x, 0, t_start as u64);
        let layer = point_layer("default", x as u64 * 1000, per_tile, t_start, t_end);
        let payload = encode_tile(&[layer]).unwrap();
        writer
            .add_tile(&id, t_start, t_end, per_tile as u32, &payload)
            .unwrap();
        features += per_tile as u64;
    }
    let meta = Metadata::new("test")
        .with_time_range(TimeRange::new(t_start as u64, t_end as u64))
        .with_zoom_levels(8, 8);
    let mut meta = meta;
    meta.feature_count = features;
    meta.tile_count = tile_count as u64;
    writer.finalize(&meta).unwrap();

    transcode_archive_to_packs(staging.path(), out_dir, BlobOrdering::Auto, 64 * 1024 * 1024)
        .unwrap();
    features
}

/// Run the compiled binary with `--json` plus `extra` args; return (success,
/// parsed report json).
fn run_json(archive: &Path, extra: &[&str]) -> (bool, serde_json::Value) {
    let bin = env!("CARGO_BIN_EXE_stt-validate");
    let mut cmd = Command::new(bin);
    cmd.arg(archive).arg("--json");
    for a in extra {
        cmd.arg(a);
    }
    let out = cmd.output().expect("failed to run stt-validate");
    let report: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("binary must emit valid JSON");
    (out.status.success(), report)
}

#[test]
fn valid_archive_passes_full_validation() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("ok");
    let features = write_dataset(&archive, 10, 4);

    let (ok, report) = run_json(&archive, &[]);
    assert!(ok, "valid archive should exit 0; report: {report}");
    assert_eq!(report["errors"].as_array().unwrap().len(), 0);
    assert_eq!(report["tile_count"], 10);
    assert_eq!(report["tiles_decoded"], 10);
    assert_eq!(report["sampled"], false);
    assert_eq!(report["feature_count_decoded_complete"], true);
    assert_eq!(report["feature_count_decoded"].as_u64().unwrap(), features);
    // All tiles share one schema → exactly one distinct signature.
    assert_eq!(report["distinct_schemas"], 1);
}

#[test]
fn sample_limits_decoded_tile_count_and_skips_grand_total() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("big");
    write_dataset(&archive, 20, 4);

    // --sample 5 over 20 tiles → stride = ceil(20/5) = 4 → indices 0,4,8,12,16
    // → exactly 5 decoded tiles.
    let (ok, report) = run_json(&archive, &["--sample", "5"]);
    assert!(ok, "sampled run of a valid archive should still pass: {report}");
    assert_eq!(report["tile_count"], 20);
    assert_eq!(report["tiles_decoded"], 5, "sample must cap decoded count");
    assert_eq!(report["sampled"], true);
    // Grand-total feature check must be skipped (not spuriously failing).
    assert_eq!(report["feature_count_decoded_complete"], false);
    assert_eq!(report["errors"].as_array().unwrap().len(), 0);
}

#[test]
fn sample_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("det");
    write_dataset(&archive, 20, 4);

    let (_, a) = run_json(&archive, &["--sample", "5"]);
    let (_, b) = run_json(&archive, &["--sample", "5"]);
    assert_eq!(a["tiles_decoded"], b["tiles_decoded"]);
    assert_eq!(a["feature_count_decoded"], b["feature_count_decoded"]);
}

#[test]
fn sample_larger_than_archive_decodes_everything() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("small");
    write_dataset(&archive, 3, 4);

    let (ok, report) = run_json(&archive, &["--sample", "100"]);
    assert!(ok, "report: {report}");
    assert_eq!(report["tiles_decoded"], 3, "N >= total decodes all tiles");
    assert_eq!(report["sampled"], true);
}

#[test]
fn skip_decode_reports_zero_decoded_and_incomplete() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("skip");
    write_dataset(&archive, 5, 4);

    let (ok, report) = run_json(&archive, &["--skip-decode"]);
    assert!(ok, "report: {report}");
    assert_eq!(report["tiles_decoded"], 0);
    assert_eq!(report["feature_count_decoded_complete"], false);
    assert_eq!(report["sampled"], false);
}
