//! TB-10 end-to-end — exact adaptive partition + shared boundaries.
//!
//! The unit tests in `stt_build::tiler` prove the partition is optimal and the
//! keys are enumerable. This one measures the thing the item is actually FOR,
//! through the real binary:
//!
//!  * Metric 1 (the plan's acceptance): per-tile byte variance strictly down
//!    versus the greedy on a bursty dataset.
//!  * The manifest publishes `adaptiveBoundaries`, and only for adaptive builds.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

const T0: i64 = 1_700_000_000_000;

fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = Vec::with_capacity(21);
    b.push(1);
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

/// A deliberately BURSTY dataset: long quiet stretches punctuated by dense
/// clusters, spread over several spatial cells with different densities. This is
/// the shape the greedy handles worst — each burst overfills one window and
/// leaves a runt behind.
fn write_bursty(path: &Path) {
    let mut wkb: Vec<Vec<u8>> = Vec::new();
    let mut ts: Vec<i64> = Vec::new();
    // Deterministic LCG; no wall-clock, no rand dependency.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u64
    };

    for cell in 0..6u64 {
        let lon = -122.6 + cell as f64 * 0.05;
        let lat = 37.7;
        let mut t = T0;
        for _ in 0..30 {
            // A burst of 1..=40 events inside a few seconds, then a long gap.
            let burst = 1 + next() % 40;
            for j in 0..burst {
                wkb.push(wkb_point(lon + (j as f64) * 1e-5, lat));
                ts.push(t + (j as i64) * 250);
            }
            t += 60_000 * (1 + next() % 30) as i64;
        }
    }

    let geom: Vec<Option<&[u8]>> = wkb.iter().map(|w| Some(w.as_slice())).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new("timestamp", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(BinaryArray::from_opt_vec(geom)) as ArrayRef,
            Arc::new(Int64Array::from(ts)) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn stt_build_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stt-build"))
}

fn run_build(input: &Path, out: &Path, extra: &[&str]) {
    let output = Command::new(stt_build_bin())
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(out)
        .args([
            "--time-field",
            "timestamp",
            "--time-format",
            "unix-ms",
            "--min-zoom",
            "9",
            "--max-zoom",
            "9",
            "--workers",
            "2",
            "--name",
            "tb10",
        ])
        .args(extra)
        .output()
        .expect("failed to spawn stt-build");
    assert!(
        output.status.success(),
        "stt-build failed ({}):\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

fn manifest(out: &Path) -> serde_json::Value {
    serde_json::from_str(&fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap()
}

/// Per-tile compressed lengths, read from the directory the build just wrote.
fn tile_lengths(out: &Path) -> Vec<u64> {
    let reader =
        stt_core::PackedReader::open(out.join("manifest.json")).expect("open packed archive");
    reader.entries().iter().map(|e| e.length as u64).collect()
}

fn variance(xs: &[u64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n
}

#[test]
fn the_exact_partition_lowers_per_tile_byte_variance() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("bursty.parquet");
    write_bursty(&input);

    let exact = dir.path().join("exact");
    let greedy = dir.path().join("greedy");
    run_build(&input, &exact, &["--adaptive-temporal", "40"]);
    run_build(
        &input,
        &greedy,
        &["--adaptive-temporal", "40", "--adaptive-greedy"],
    );

    let (le, lg) = (tile_lengths(&exact), tile_lengths(&greedy));
    // Losslessness first: the rebalance must not change how many features ship.
    let fe: u64 = manifest(&exact)["metadata"]["feature_count"]
        .as_u64()
        .unwrap_or_default();
    let fg: u64 = manifest(&greedy)["metadata"]["feature_count"]
        .as_u64()
        .unwrap_or_default();
    assert_eq!(fe, fg, "the partition must not add or drop features");

    let (ve, vg) = (variance(&le), variance(&lg));
    println!(
        "TB-10 per-tile bytes: exact n={} var={:.0} max={:?} | greedy n={} var={:.0} max={:?}",
        le.len(),
        ve,
        le.iter().max(),
        lg.len(),
        vg,
        lg.iter().max()
    );
    assert!(
        ve < vg,
        "exact partition must reduce per-tile byte variance (exact {ve:.0} vs greedy {vg:.0})"
    );
    // And the largest tile can never grow — the dominance property.
    assert!(
        le.iter().max() <= lg.iter().max(),
        "the largest tile must not grow"
    );
}

#[test]
fn adaptive_boundaries_are_published_only_for_adaptive_builds() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("bursty.parquet");
    write_bursty(&input);

    let adaptive = dir.path().join("adaptive");
    run_build(&input, &adaptive, &["--adaptive-temporal", "40"]);
    let b = manifest(&adaptive)["adaptiveBoundaries"]
        .as_array()
        .expect("an adaptive build publishes adaptiveBoundaries")
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect::<Vec<_>>();
    assert!(!b.is_empty());
    assert!(b.windows(2).all(|w| w[0] < w[1]), "ascending and deduped");

    // A fixed-bucket build must be byte-unchanged: no key at all.
    let fixed = dir.path().join("fixed");
    run_build(&input, &fixed, &["--temporal-bucket", "1h"]);
    assert!(
        manifest(&fixed).get("adaptiveBoundaries").is_none(),
        "a non-adaptive manifest must not carry the key"
    );

    // ...and so must the rollback, which does no snapping.
    let rollback = dir.path().join("rollback");
    run_build(
        &input,
        &rollback,
        &["--adaptive-temporal", "40", "--adaptive-greedy"],
    );
    assert!(
        manifest(&rollback).get("adaptiveBoundaries").is_none(),
        "--adaptive-greedy snaps nothing, so it publishes nothing"
    );
}

/// Determinism: byte-identical rebuild, the constraint every item is held to.
#[test]
fn an_adaptive_build_is_byte_reproducible() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("bursty.parquet");
    write_bursty(&input);

    let a = dir.path().join("a");
    let b = dir.path().join("b");
    run_build(&input, &a, &["--adaptive-temporal", "40"]);
    run_build(&input, &b, &["--adaptive-temporal", "40"]);

    assert_eq!(
        fs::read(a.join("manifest.json")).unwrap(),
        fs::read(b.join("manifest.json")).unwrap(),
        "adaptive manifests must be byte-identical across runs"
    );
    let pack_bytes = |d: &Path| -> Vec<Vec<u8>> {
        let mut names: Vec<_> = fs::read_dir(d.join("packs"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        names.sort();
        names.iter().map(|p| fs::read(p).unwrap()).collect()
    };
    assert_eq!(pack_bytes(&a), pack_bytes(&b), "pack bytes must match");
}

/// Diagnostic (printed with `--nocapture`): where the per-tile bytes actually
/// go, so the variance comparison is interpretable rather than a bare number.
#[test]
fn report_per_tile_distributions() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("bursty.parquet");
    write_bursty(&input);
    let exact = dir.path().join("exact");
    let greedy = dir.path().join("greedy");
    run_build(&input, &exact, &["--adaptive-temporal", "40"]);
    run_build(
        &input,
        &greedy,
        &["--adaptive-temporal", "40", "--adaptive-greedy"],
    );

    for (label, out) in [("exact", &exact), ("greedy", &greedy)] {
        let reader =
            stt_core::PackedReader::open(out.join("manifest.json")).expect("open packed archive");
        let mut feats: Vec<u64> = Vec::new();
        let mut bytes: Vec<u64> = Vec::new();
        let mut spans: Vec<i64> = Vec::new();
        for e in reader.entries() {
            feats.push(e.feature_count as u64);
            bytes.push(e.length as u64);
            spans.push(e.time_end - e.time_start);
        }
        let mean = |v: &[u64]| v.iter().sum::<u64>() as f64 / v.len() as f64;
        println!(
            "{label:6}: tiles={} feats(mean {:.1}, var {:.0}, min {:?}, max {:?}) \
             bytes(mean {:.0}, var {:.0}) span_ms(mean {:.0}, max {:?})",
            feats.len(),
            mean(&feats),
            variance(&feats),
            feats.iter().min(),
            feats.iter().max(),
            mean(&bytes),
            variance(&bytes),
            spans.iter().map(|&s| s as f64).sum::<f64>() / spans.len() as f64,
            spans.iter().max()
        );
    }
}
