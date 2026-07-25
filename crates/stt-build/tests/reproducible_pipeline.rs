//! Full-pipeline byte-reproducibility COMPOSITION test.
//!
//! The two halves of the reproducible-build contract are each covered on their
//! own elsewhere:
//!   * per-tile encode determinism — `stt-core`'s `reproducible_build.rs`
//!     (`same_tile_encodes_byte_identically`), and
//!   * pack/directory ASSEMBLY order-independence — `stt-core`'s
//!     `pack.rs::rebuilds_are_byte_reproducible` (forward vs reverse add order).
//!
//! Neither exercises their COMPOSITION through the real `stt-build` tiler
//! pipeline: parse → clip → spatial/temporal bucketing → per-tile encode →
//! content-addressed pack + directory. This test runs `generate_tiles_streaming`
//! twice over the SAME parsed feature stream and asserts the two archives are
//! byte-identical — a strictly STRONGER check than `common::assert_archives_equal`
//! (which compares only the per-tile `(z,x,y,t,count)` key-set): equal directory
//! hash + equal directory-object bytes + equal pack key list + equal pack bytes +
//! equal `manifest.to_json_bytes()`. This is the immutable-pack CDN contract end
//! to end — a rebuild of unchanged data must not invalidate the edge cache.
//!
//! The tiler runs on a rayon pool, so tile EMISSION order legitimately varies
//! run-to-run (the spatial grouping is a `HashMap`); the point of the test is
//! that the writer normalises that away and the on-disk bytes do not move.

use std::path::Path;

use stt_build::input::{self, ParsedFeature};
use stt_build::tiler::{generate_tiles_streaming, TileConfig};
use stt_core::metadata::Metadata;
use stt_core::{BlobOrdering, Manifest, PackWriter};

mod common;
use common::{fixture_rows, load_file, write_fixture_parquet};

/// Run the real streaming pipeline into `out_dir` and return its manifest.
/// A deliberately small max-pack-size forces the tiles to roll across MULTIPLE
/// packs, so pack-assembly determinism (not merely single-pack output) is
/// exercised.
fn build_pipeline(features: &[ParsedFeature], config: &TileConfig, out_dir: &Path) -> Manifest {
    let mut writer = PackWriter::create(out_dir, BlobOrdering::Auto, 512).unwrap();
    generate_tiles_streaming(features, config, &mut writer, 2).unwrap();
    let (bounds, time_range) = input::calculate_bounds(features).unwrap();
    let metadata = Metadata::new("repro-pipeline")
        .with_bounds(bounds)
        .with_time_range(time_range)
        .with_zoom_levels(config.min_zoom, config.max_zoom)
        .with_temporal_bucket_ms(config.temporal_bucket_ms);
    writer.finalize(&metadata).unwrap()
}

#[test]
fn full_pipeline_is_byte_reproducible() {
    // Parse the canonical mixed fixture (points + timed linestrings with
    // per-vertex arrays + a null-geometry skip) ONCE through the real GeoParquet
    // reader, so both builds see exactly the same input feature stream — the
    // composition under test is the tiler + pack writer, not the reader.
    let src = tempfile::tempdir().unwrap();
    let parquet = src.path().join("input.parquet");
    write_fixture_parquet(&fixture_rows(), &parquet);
    let features = load_file(&parquet);
    assert!(!features.is_empty(), "fixture must yield tiled features");

    let config = TileConfig {
        min_zoom: 0,
        max_zoom: 8,
        temporal_bucket_ms: 3_600_000,
        ..TileConfig::default()
    };

    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let out_a = dir_a.path();
    let out_b = dir_b.path();
    let man_a = build_pipeline(&features, &config, out_a);
    let man_b = build_pipeline(&features, &config, out_b);

    // Sanity: the pipeline actually produced content across several packs, so
    // this is not an accidental empty==empty pass.
    assert!(
        man_a.packs.len() >= 2,
        "expected the fixture + 8-zoom build to roll across multiple packs, got {}",
        man_a.packs.len()
    );

    // (1) Directory content-address (hash) AND the directory-object bytes at
    // rest are identical.
    assert_eq!(
        man_a.directory.key, man_b.directory.key,
        "directory content-address (hash) must be stable across rebuilds"
    );
    let dir_bytes_a = std::fs::read(out_a.join(&man_a.directory.key)).unwrap();
    let dir_bytes_b = std::fs::read(out_b.join(&man_b.directory.key)).unwrap();
    assert_eq!(
        dir_bytes_a, dir_bytes_b,
        "directory object bytes differ across rebuilds"
    );

    // (2) Pack key list identical (order + content-address), and every pack's
    // on-disk bytes identical.
    let keys_a: Vec<&String> = man_a.packs.iter().map(|p| &p.key).collect();
    let keys_b: Vec<&String> = man_b.packs.iter().map(|p| &p.key).collect();
    assert_eq!(
        keys_a, keys_b,
        "pack content-addresses must be stable across rebuilds"
    );
    for pack in &man_a.packs {
        let bytes_a = std::fs::read(out_a.join(&pack.key)).unwrap();
        let bytes_b = std::fs::read(out_b.join(&pack.key)).unwrap();
        assert_eq!(
            bytes_a, bytes_b,
            "pack {} bytes differ across rebuilds",
            pack.key
        );
    }

    // (3) The whole manifest serialises byte-identically (directory ref + pack
    // refs + metadata + tile count).
    assert_eq!(
        man_a.to_json_bytes().unwrap(),
        man_b.to_json_bytes().unwrap(),
        "manifest JSON must be byte-identical across rebuilds"
    );

    // The shared key-set comparator is strictly weaker than the byte checks
    // above; assert it too so the "stronger ⇒ weaker" relationship is explicit
    // and any regression that trips one is understood against the other.
    common::assert_archives_equal(out_a, out_b, "pipeline-repro");
}
