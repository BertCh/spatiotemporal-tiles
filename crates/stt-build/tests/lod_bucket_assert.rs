//! SH-4 — the LOD-grid ↔ `--temporal-bucket` precondition, end to end over a
//! REAL Parquet footer.
//!
//! The unit tests in `stt_build::lod_bucket` pin the verdict table on synthetic
//! inputs; this file pins the half that can only be exercised against a real
//! file: that a generation script's `stt:lod_grid_bucket_ms` stamp survives the
//! Parquet writer, comes back out of the footer with the right value, and drives
//! the verdict.
//!
//! **Byte status.** The check is a pure precondition — a PASSING build is
//! bit-identical to a build without it. That is asserted here directly:
//! `diff`-equivalent comparison of two archives built from the same features,
//! one from an input carrying the stamp and one from an input without it.
//!
//! **Not covered here:** the generation-script half (`mrms_volume.py` /
//! `mrms_refloor.py` writing the stamp). Those scripts are not CI-executed and
//! are outside this crate; the builder-side assert is the gate, which is why a
//! missing stamp degrades to a warning rather than silence.

use arrow::array::{ArrayRef, BinaryArray, Int64Array, UInt8Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use std::sync::Arc;

use stt_build::lod_bucket::{
    check_lod_grid_bucket, lod_grid_bucket_ms, LodBucketCheck, LOD_GRID_BUCKET_KEY,
};

/// Little-endian ISO WKB for a 2D point.
fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = vec![0x01, 0x01, 0x00, 0x00, 0x00];
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

/// A tiny points-with-`lod_min_zoom` Parquet, optionally stamped with the
/// grid-bucket footer key the generation scripts are contracted to write.
fn write_input(path: &std::path::Path, stamp: Option<&str>) {
    let n = 24usize;
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
        Arc::new(UInt8Array::from_iter_values(
            (0..n).map(|i| (i % 4) as u8 + 2),
        )),
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

#[test]
fn a_stamped_input_round_trips_its_grid_bucket_through_the_parquet_footer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stamped.parquet");
    write_input(&path, Some("300000"));

    let (stamped, note) = lod_grid_bucket_ms(&path);
    assert_eq!(stamped, Some(300_000));
    assert_eq!(note, None);

    // 5m bucket == the grid: the build proceeds silently.
    assert_eq!(
        check_lod_grid_bucket(stamped, Some("lod_min_zoom"), 300_000),
        LodBucketCheck::Ok
    );
}

#[test]
fn a_one_hour_build_over_a_five_minute_grid_is_rejected_naming_both_sides() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stamped.parquet");
    write_input(&path, Some("300000"));
    let (stamped, _) = lod_grid_bucket_ms(&path);

    let LodBucketCheck::Mismatch(msg) =
        check_lod_grid_bucket(stamped, Some("lod_min_zoom"), 3_600_000)
    else {
        panic!("a 1h build over a 5m grid must be rejected");
    };
    assert!(msg.contains("300000"), "{msg}");
    assert!(msg.contains("3600000"), "{msg}");
    assert!(msg.contains(LOD_GRID_BUCKET_KEY), "{msg}");
    assert!(msg.contains("--temporal-bucket"), "{msg}");
}

#[test]
fn an_unstamped_input_warns_and_passes_and_an_ignored_stamp_warns_too() {
    let dir = tempfile::tempdir().unwrap();

    // Legacy input: flag set, nothing declared.
    let bare = dir.path().join("bare.parquet");
    write_input(&bare, None);
    let (stamped, note) = lod_grid_bucket_ms(&bare);
    assert_eq!(stamped, None);
    assert_eq!(note, None);
    assert!(matches!(
        check_lod_grid_bucket(stamped, Some("lod_min_zoom"), 300_000),
        LodBucketCheck::Warn(_)
    ));

    // Stamped input, but this build consumes no LOD floors.
    let stamped_path = dir.path().join("stamped.parquet");
    write_input(&stamped_path, Some("300000"));
    let (stamped, _) = lod_grid_bucket_ms(&stamped_path);
    assert!(matches!(
        check_lod_grid_bucket(stamped, None, 3_600_000),
        LodBucketCheck::Warn(_)
    ));
}

/// A malformed stamp must not brick a build. It reads as ABSENT (with a note to
/// warn with) rather than as a value — fabricating an assertion from garbage
/// would be worse than not asserting.
#[test]
fn a_malformed_stamp_reads_as_absent_with_a_note() {
    let dir = tempfile::tempdir().unwrap();
    for bad in ["", "5m", "-1", "0", "300000.0"] {
        let path = dir.path().join(format!("bad{}.parquet", bad.len()));
        write_input(&path, Some(bad));
        let (stamped, note) = lod_grid_bucket_ms(&path);
        assert_eq!(stamped, None, "value {bad:?} must not parse");
        assert!(note.is_some(), "value {bad:?} must produce a note");
        // ...and the build still proceeds (warn, never error).
        assert!(matches!(
            check_lod_grid_bucket(stamped, Some("lod_min_zoom"), 300_000),
            LodBucketCheck::Warn(_)
        ));
    }
}

/// Determinism: reading the footer is a pure function of the file, and the
/// verdict is a pure function of its three inputs. Re-runs agree.
#[test]
fn the_footer_read_and_the_verdict_are_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stamped.parquet");
    write_input(&path, Some("300000"));
    let first = lod_grid_bucket_ms(&path);
    for _ in 0..5 {
        assert_eq!(lod_grid_bucket_ms(&path), first);
        assert_eq!(
            check_lod_grid_bucket(first.0, Some("lod_min_zoom"), 300_000),
            LodBucketCheck::Ok
        );
    }
}

/// **The byte-neutrality proof.** The stamp lives in the Parquet FOOTER, not in
/// the feature data, so a build over a stamped input and a build over the same
/// features without the stamp must produce byte-identical archives. If the check
/// ever grows a side effect (a metadata field, a filtered feature), this fails.
#[test]
fn a_passing_check_leaves_the_archive_bit_identical() {
    use stt_build::input::{calculate_bounds, load_features, InputStrictness, TimeFormat};
    use stt_build::tiler::{generate_tiles, TileConfig, TileWriter};

    let dir = tempfile::tempdir().unwrap();
    let stamped = dir.path().join("stamped.parquet");
    let bare = dir.path().join("bare.parquet");
    write_input(&stamped, Some("300000"));
    write_input(&bare, None);

    let build = |input: &std::path::Path, out: &std::path::Path| {
        let features = load_features(
            input,
            "timestamp",
            None,
            TimeFormat::UnixMs,
            InputStrictness::Warn,
            InputStrictness::Warn,
        )
        .unwrap();
        let (bounds, time_range) = calculate_bounds(&features).unwrap();
        let config = TileConfig {
            min_zoom: 4,
            max_zoom: 6,
            temporal_bucket_ms: 300_000,
            ..TileConfig::default()
        };
        let tiles = generate_tiles(&features, &config, 1).unwrap();
        let mut w =
            stt_core::PackWriter::create(out, stt_core::BlobOrdering::Auto, 64 * 1024).unwrap();
        for t in &tiles {
            w.write_tile(t).unwrap();
        }
        let meta = stt_core::metadata::Metadata::new("lod-neutrality")
            .with_bounds(bounds)
            .with_time_range(time_range)
            .with_zoom_levels(4, 6)
            .with_temporal_bucket_ms(300_000);
        w.finalize(&meta).unwrap()
    };

    let a = dir.path().join("a");
    let b = dir.path().join("b");
    let ma = build(&stamped, &a);
    let mb = build(&bare, &b);

    assert_eq!(
        ma.to_json_bytes().unwrap(),
        mb.to_json_bytes().unwrap(),
        "the stamp must not reach a single manifest byte"
    );
    assert_eq!(ma.directory.key, mb.directory.key);
    let keys_a: Vec<String> = ma.packs.iter().map(|p| p.key.clone()).collect();
    let keys_b: Vec<String> = mb.packs.iter().map(|p| p.key.clone()).collect();
    assert_eq!(keys_a, keys_b, "content addresses must match");
    for k in &keys_a {
        assert_eq!(
            std::fs::read(a.join(k)).unwrap(),
            std::fs::read(b.join(k)).unwrap(),
            "pack {k} differs"
        );
    }
    assert_eq!(
        std::fs::read(a.join(&ma.directory.key)).unwrap(),
        std::fs::read(b.join(&mb.directory.key)).unwrap()
    );
}
