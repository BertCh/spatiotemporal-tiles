//! MO-1 integration: stratified decode sampling against a REAL packed archive.
//!
//! The unit tests in `analysis::inspect` prove the allocator; this suite proves
//! the thing the allocator exists for — that a sampled inspect of a real
//! archive agrees with an exhaustive one on the numbers downstream consumers
//! (the doctor, `diff`, the quantize advisor) actually key off: the per-column
//! shares.
//!
//! The fixture is deliberately shaped like the aliasing trap: the directory
//! interleaves temporal buckets, and each bucket carries a DIFFERENT column
//! mix, so a sample that lands on one bucket phase reports shares that are
//! badly wrong. Archive bytes are never touched by any of this — sampling is
//! read-side only.

use std::path::Path;

use stt_core::arrow_tile::{
    encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn, PropertyColumn,
};
use stt_core::curve::BlobOrdering;
use stt_core::metadata::Metadata;
use stt_core::pack::PackWriter;
use stt_core::tile::TileId;
use stt_optimize::analysis::inspect::{
    inspect, inspect_with_design, stratified_sample_indices, stride_sample_indices, SampleDesign,
};
use stt_optimize::PackedTileset;

/// Base temporal bucket of the fixture (1 hour).
const BUCKET_MS: i64 = 3_600_000;
/// Temporal buckets the fixture writes.
const BUCKETS: i64 = 4;
/// Tiles per bucket.
const TILES_PER_BUCKET: u32 = 8;
/// Points per tile.
const POINTS: usize = 120;

/// One point layer. `bucket` drives the property mix: bucket 0 ships a
/// high-entropy `payload` string column that the other buckets do not, so a
/// bucket-aliased sample over-attributes wildly.
fn point_layer(bucket: i64, tile: u32) -> ColumnarLayer {
    let n = POINTS;
    let geometry: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let jitter = |salt: u64| {
                ((i as u64).wrapping_add(salt).wrapping_mul(2_654_435_761) % 100_000) as f64 * 1e-7
            };
            [
                -73.6 + tile as f64 * 0.01 + i as f64 * 1e-5 + jitter(3),
                45.5 + bucket as f64 * 0.01 + jitter(11),
            ]
        })
        .collect();
    let t0 = bucket * BUCKET_MS;
    let mut properties: Vec<(String, PropertyColumn)> = vec![
        (
            "speed".to_string(),
            PropertyColumn::Numeric(
                (0..n)
                    .map(|i| Some((i as f64) * 0.137 + bucket as f64))
                    .collect(),
            ),
        ),
        (
            "kind".to_string(),
            PropertyColumn::Categorical(
                (0..n)
                    .map(|i| Some(["bike", "ferry", "bus"][i % 3].to_string()))
                    .collect(),
            ),
        ),
    ];
    if bucket == 0 {
        // Near-incompressible per-row text: only present in bucket 0.
        properties.push((
            "payload".to_string(),
            PropertyColumn::Categorical(
                (0..n)
                    .map(|i| {
                        Some(format!(
                            "{:016x}{:016x}",
                            (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                            (i as u64 + tile as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
                        ))
                    })
                    .collect(),
            ),
        ));
    }
    ColumnarLayer {
        polygon_parts: None,
        name: "default".to_string(),
        feature_ids: (0..n as u64).map(|i| bucket as u64 * 10_000 + i).collect(),
        start_times: vec![t0; n],
        end_times: vec![t0 + BUCKET_MS - 1; n],
        geometry: GeometryColumn::Point(geometry),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties,
    }
}

/// Write a packed archive of `TILES_PER_BUCKET` spatial cells, each carrying
/// all `BUCKETS` temporal buckets.
///
/// The directory is space–time-curve ordered, so a cell's buckets land
/// consecutively and the entry sequence has period `BUCKETS` — which is
/// exactly what a flat stride aliases against. That periodicity is not
/// contrived: it is what every time-major archive's directory looks like.
fn build_fixture(out: &Path) {
    let mut w = PackWriter::create(out, BlobOrdering::Auto, 256 * 1024).unwrap();
    let cfg = EncoderConfig {
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(w.template_collector()),
        ..EncoderConfig::default()
    };
    for tile in 0..TILES_PER_BUCKET {
        for bucket in 0..BUCKETS {
            let payload = encode_tile_with(&[point_layer(bucket, tile)], &cfg).unwrap();
            let t0 = bucket * BUCKET_MS;
            w.add_tile_full(
                &TileId::new(10, 300 + tile, 380, t0 as u64),
                t0,
                t0 + BUCKET_MS - 1,
                Some(t0),
                POINTS as u32,
                Some(BUCKET_MS as u64),
                &payload,
            )
            .unwrap();
        }
    }
    let meta = Metadata::new("inspect-sampling-fixture")
        .with_temporal_bucket_ms(BUCKET_MS as u64)
        .with_zoom_levels(10, 10);
    w.finalize(&meta).unwrap();
}

fn share_of(report: &stt_optimize::InspectReport, name: &str) -> f64 {
    report
        .per_column
        .iter()
        .find(|c| c.name == name)
        .map(|c| c.share)
        .unwrap_or(0.0)
}

/// Largest per-column share error against the exhaustive decode, in share
/// points (0.01 = one percentage point).
fn max_share_error(
    exhaustive: &stt_optimize::InspectReport,
    sampled: &stt_optimize::InspectReport,
) -> f64 {
    exhaustive
        .per_column
        .iter()
        .map(|c| (c.share - share_of(sampled, &c.name)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn stratified_sample_tracks_exhaustive_shares_and_beats_the_stride() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dataset");
    build_fixture(&out);
    let ts = PackedTileset::open(&out).unwrap();

    let entries = ts.entries();
    assert_eq!(entries.len(), (TILES_PER_BUCKET as i64 * BUCKETS) as usize);

    let exhaustive = inspect(&ts, None).unwrap();
    assert!(!exhaustive.decode.sampled);
    assert_eq!(exhaustive.decode.tiles_decoded, entries.len() as u64);

    // n = 8 over 32 entries: stride 4, and the directory's bucket period is 4.
    let n = 8;
    let strided_ix = stride_sample_indices(entries, n);
    let strided_buckets: std::collections::BTreeSet<i64> =
        strided_ix.iter().map(|&i| entries[i].time_start).collect();
    let stratified_ix = stratified_sample_indices(entries, n);
    let stratified_buckets: std::collections::BTreeSet<i64> = stratified_ix
        .iter()
        .map(|&i| entries[i].time_start)
        .collect();
    assert_eq!(
        strided_buckets.len(),
        1,
        "guard for the REJECTED design: stride-v1 must alias to one bucket on this directory"
    );
    assert_eq!(
        stratified_buckets.len(),
        BUCKETS as usize,
        "stratified must reach every temporal bucket"
    );

    let stratified = inspect(&ts, Some(n)).unwrap();
    let strided = inspect_with_design(&ts, Some(n), SampleDesign::StrideV1).unwrap();
    assert_eq!(stratified.decode.tiles_decoded, n as u64);
    assert_eq!(strided.decode.tiles_decoded, n as u64);

    // Directory-derived totals are exhaustive under BOTH designs — the diff
    // gate's exact metric must never route through the sampled path.
    for r in [&stratified, &strided] {
        assert_eq!(r.tile_count, exhaustive.tile_count);
        assert_eq!(r.compressed_bytes, exhaustive.compressed_bytes);
        assert_eq!(r.uncompressed_bytes, exhaustive.uncompressed_bytes);
        assert_eq!(r.feature_count, exhaustive.feature_count);
        assert_eq!(r.dedup.entries, exhaustive.dedup.entries);
        assert_eq!(r.dedup.distinct_blobs, exhaustive.dedup.distinct_blobs);
    }

    // Acceptance (O2): stratified error <= systematic error at equal n.
    let strat_err = max_share_error(&exhaustive, &stratified);
    let stride_err = max_share_error(&exhaustive, &strided);
    // Measured on this fixture: stratified 0.0007, stride 0.2767 (share points
    // 0.07 vs 27.7). The bounds below are loose enough to be stable.
    assert!(
        strat_err <= stride_err,
        "stratified max share error {strat_err:.4} must not exceed stride {stride_err:.4}"
    );
    assert!(
        strat_err < 0.05,
        "stratified max share error {strat_err:.4} should stay within 5 share points"
    );
    // The aliased sample sees bucket 0's `payload` column in EVERY tile and
    // charges it the whole archive's share; the exhaustive truth is ~1/4 of
    // that. This is the misattribution stratification exists to remove.
    assert!(
        share_of(&strided, "payload") > share_of(&exhaustive, "payload") * 1.5,
        "stride over-attributes payload: {} vs exhaustive {}",
        share_of(&strided, "payload"),
        share_of(&exhaustive, "payload")
    );
    assert!(
        (share_of(&stratified, "payload") - share_of(&exhaustive, "payload")).abs()
            < (share_of(&strided, "payload") - share_of(&exhaustive, "payload")).abs()
    );
}

#[test]
fn sampled_inspect_is_reproducible_and_sample_zero_reads_no_payloads() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dataset");
    build_fixture(&out);
    let ts = PackedTileset::open(&out).unwrap();

    let a = serde_json::to_string(&inspect(&ts, Some(8)).unwrap()).unwrap();
    let b = serde_json::to_string(&inspect(&ts, Some(8)).unwrap()).unwrap();
    assert_eq!(a, b, "two inspect runs must serialize byte-identical JSON");

    // `--sample 0` keeps its directory-only fast-mode semantics.
    let none = inspect(&ts, Some(0)).unwrap();
    assert_eq!(none.decode.tiles_decoded, 0);
    assert_eq!(none.decode.features_decoded, 0);
    assert!(none.per_column.is_empty());
    assert!(none.decode.sampled);
    assert_eq!(none.tile_count, (TILES_PER_BUCKET as i64 * BUCKETS) as u64);
    assert!(none.compressed_bytes > 0);
    assert_eq!(none.per_zoom.len(), 1);
}

#[test]
fn doctor_sees_every_temporal_bucket_on_the_aliasing_fixture() {
    // The doctor decodes only 8 tiles, so it was the most exposed consumer of
    // the flat stride: a bucket-aliased sample makes `dead-columns` reason
    // about one bucket's schema. With stratified selection its 8 tiles span
    // all four buckets, so bucket 0's `payload` column is seen as varying
    // rather than absent.
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dataset");
    build_fixture(&out);
    let ts = PackedTileset::open(&out).unwrap();
    let report = inspect(&ts, None).unwrap();

    let doc = stt_optimize::doctor::doctor(&ts, &report).unwrap();
    // Nothing in this fixture is constant or all-null, so no dead-column
    // finding may be raised off the sampled decode.
    assert!(
        !doc.findings.iter().any(|f| f.code == "dead-columns"),
        "unexpected dead-columns finding: {:?}",
        doc.findings
    );
}
