//! SH-4 CLI gate: `stt-build` must REFUSE a build whose `--temporal-bucket`
//! disagrees with the thinning-grid bucket the input's baked LOD-floor column
//! was computed against.
//!
//! `crates/stt-build/tests/lod_bucket_assert.rs` pins the library seam (footer
//! read, verdict table, byte-neutrality). This file pins the part only the
//! binary can prove: that a mismatch exits NON-ZERO with both values in the
//! message, and that the two benign shapes still build.
//!
//! The check's placement is load-bearing — it runs after `--auto` may have
//! rewritten `--temporal-bucket`, so the value compared is the bucket the
//! archive is actually tiled at, not the one the user typed. That ordering is
//! unit-pinned in `stt_build::lod_bucket`; here we exercise the hand-passed
//! path, which is the one a human hits.

// The stt-build binary (and CARGO_BIN_EXE_stt-build) only exists when the
// build-cli feature is on. It is on by default; skip the whole file otherwise
// so a `--no-default-features` check still compiles.
#![cfg(feature = "build-cli")]

use arrow::array::{ArrayRef, BinaryArray, Int64Array, UInt8Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

/// Must match `stt_build::lod_bucket::LOD_GRID_BUCKET_KEY`. Spelled out rather
/// than imported so the wire contract is asserted from the outside: a rename in
/// the library breaks this test, which is the point.
const LOD_GRID_BUCKET_KEY: &str = "stt:lod_grid_bucket_ms";

fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = vec![0x01, 0x01, 0x00, 0x00, 0x00];
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

fn write_input(path: &Path, stamp: Option<&str>) {
    let n = 40usize;
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new("timestamp", DataType::Int64, true),
        Field::new("lod_min_zoom", DataType::UInt8, true),
    ]));
    let wkbs: Vec<Vec<u8>> = (0..n)
        .map(|i| wkb_point(-100.0 + i as f64 * 0.05, 40.0 + i as f64 * 0.05))
        .collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BinaryArray::from_iter_values(
            wkbs.iter().map(|b| b.as_slice()),
        )),
        Arc::new(Int64Array::from_iter_values(
            (0..n).map(|i| 1_700_000_000_000i64 + i as i64 * 300_000),
        )),
        Arc::new(UInt8Array::from_iter_values((0..n).map(|i| (i % 3) as u8))),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let mut props = WriterProperties::builder();
    if let Some(v) = stamp {
        props = props.set_key_value_metadata(Some(vec![KeyValue::new(
            LOD_GRID_BUCKET_KEY.to_string(),
            v.to_string(),
        )]));
    }
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props.build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Run the real binary; return `(success, combined stdout+stderr)`.
fn run_build(input: &Path, out: &Path, bucket: &str, lod_field: Option<&str>) -> (bool, String) {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_stt-build"));
    let mut cmd = Command::new(bin);
    cmd.arg("--input")
        .arg(input)
        .arg("--output")
        .arg(out)
        .args([
            "--time-field",
            "timestamp",
            "--time-format",
            "unix-ms",
            "--min-zoom",
            "2",
            "--max-zoom",
            "5",
            "--temporal-bucket",
            bucket,
        ]);
    if let Some(f) = lod_field {
        cmd.args(["--min-zoom-field", f]);
    }
    let output = cmd.output().expect("run stt-build");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), text)
}

#[test]
fn a_bucket_that_disagrees_with_the_baked_lod_grid_fails_the_build() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("stamped.parquet");
    write_input(&input, Some("300000"));

    let (ok, log) = run_build(
        &input,
        &dir.path().join("mismatch"),
        "1h",
        Some("lod_min_zoom"),
    );
    assert!(
        !ok,
        "a 1h build over a 5m LOD grid must exit non-zero; log:\n{log}"
    );
    assert!(
        log.contains("300000"),
        "log must name the grid bucket:\n{log}"
    );
    assert!(
        log.contains("3600000"),
        "log must name the build bucket:\n{log}"
    );
    assert!(
        log.contains(LOD_GRID_BUCKET_KEY),
        "log must name the footer key:\n{log}"
    );
    // ...and nothing was written.
    assert!(
        !dir.path().join("mismatch").join("manifest.json").exists(),
        "a rejected build must not leave an archive behind"
    );
}

#[test]
fn an_agreeing_bucket_builds_and_the_two_benign_shapes_only_warn() {
    let dir = tempfile::tempdir().unwrap();
    let stamped = dir.path().join("stamped.parquet");
    let bare = dir.path().join("bare.parquet");
    write_input(&stamped, Some("300000"));
    write_input(&bare, None);

    // Agreement: silent pass.
    let out = dir.path().join("match");
    let (ok, log) = run_build(&stamped, &out, "5m", Some("lod_min_zoom"));
    assert!(ok, "matching buckets must build; log:\n{log}");
    assert!(out.join("manifest.json").exists());

    // Legacy input (no stamp), flag set: warns, builds.
    let out = dir.path().join("legacy");
    let (ok, log) = run_build(&bare, &out, "5m", Some("lod_min_zoom"));
    assert!(ok, "an unstamped input must still build; log:\n{log}");
    assert!(
        log.contains(LOD_GRID_BUCKET_KEY),
        "the un-asserted case must warn:\n{log}"
    );
    assert!(out.join("manifest.json").exists());

    // Stamped input, no flag: warns, builds.
    let out = dir.path().join("ignored");
    let (ok, log) = run_build(&stamped, &out, "1h", None);
    assert!(ok, "an ignored stamp must not fail a build; log:\n{log}");
    assert!(
        log.contains(LOD_GRID_BUCKET_KEY),
        "the ignored-stamp case must warn:\n{log}"
    );
    assert!(out.join("manifest.json").exists());
}
