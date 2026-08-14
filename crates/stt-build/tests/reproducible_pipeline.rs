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
//!
//! A second test, `byte_comparison_reports_a_single_flipped_pack_byte`, is the
//! NEGATIVE guard on the comparison itself: it corrupts one byte of one pack
//! and asserts the equality helper reports it, so the helper cannot rot into a
//! tautology that keeps the positive test green while checking nothing.
//!
//! The seam ABOVE this one — the `stt-build` binary, i.e. arg parsing, the
//! GeoParquet reader and the parallel parse, none of which this file's
//! "parse once, build twice" shape can reach — is covered by
//! `crates/spatiotemporal-tiles/tests/build_cli_reproducible.rs`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use stt_build::input::{self, ParsedFeature};
use stt_build::tiler::{generate_tiles_streaming, TileConfig};
use stt_core::arrow_tile::GlobalColumnPins;
use stt_core::metadata::Metadata;
use stt_core::{BlobOrdering, Manifest, PackWriter};

mod common;
use common::{fixture_rows, load_file, write_fixture_parquet};

/// Run pass 1 over a feature stream and resolve the dataset-global encoder
/// pins, exactly as `stt-build` does between load and tiling.
fn pins_for(features: &[ParsedFeature]) -> GlobalColumnPins {
    use stt_build::columnar::{infer_property_types, AttributeFilter};
    use stt_build::dataset_stats::collect_dataset_stats;

    let filter = AttributeFilter::KeepAll;
    let types = infer_property_types(
        features.iter().map(|f| f.shared_properties.as_ref()),
        &filter,
    );
    collect_dataset_stats(features, &filter, &types).to_pins()
}

/// Run the real streaming pipeline into `out_dir` and return its manifest.
/// A deliberately small max-pack-size forces the tiles to roll across MULTIPLE
/// packs, so pack-assembly determinism (not merely single-pack output) is
/// exercised.
fn build_pipeline(features: &[ParsedFeature], config: &TileConfig, out_dir: &Path) -> Manifest {
    build_pipeline_with(features, config, out_dir, false)
}

/// As [`build_pipeline`], but able to select the **default** ordering path:
/// `measured` resolves the concrete blob order by simulation at finalize, under
/// per-dataset query weights derived from the metadata's layer hint.
fn build_pipeline_with(
    features: &[ParsedFeature],
    config: &TileConfig,
    out_dir: &Path,
    measured: bool,
) -> Manifest {
    let mut writer = PackWriter::create(out_dir, BlobOrdering::Auto, 512)
        .unwrap()
        .with_measured_ordering(measured);
    generate_tiles_streaming(features, config, &mut writer, 2).unwrap();
    let (bounds, time_range) = input::calculate_bounds(features).unwrap();
    let mut metadata = Metadata::new("repro-pipeline")
        .with_bounds(bounds)
        .with_time_range(time_range)
        .with_zoom_levels(config.min_zoom, config.max_zoom)
        .with_temporal_bucket_ms(config.temporal_bucket_ms);
    // The real `stt-build` bakes the cheap style hints (layer kind + suggested
    // playback) on every non-streaming build, and the measured picker's weights
    // read the layer hint — so a test of the DEFAULT path has to carry one.
    metadata.style_hints = Some(stt_core::metadata::StyleHints {
        version: 1,
        properties: Vec::new(),
        suggested_playback_seconds: None,
        suggested_time_window_ms: None,
        layer_hint: Some("points".to_string()),
    });
    writer.finalize(&metadata).unwrap()
}

/// Every at-rest object a manifest declares — `manifest.json`, the directory
/// object, and every pack — keyed by its manifest key.
///
/// Keying off the MANIFEST rather than a directory walk is deliberate: it means
/// an object the manifest forgot to declare, or one it declares but never
/// wrote, is a hard read error here instead of a silently skipped comparison.
fn archive_objects(root: &Path, manifest: &Manifest) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let read = |key: &str| {
        std::fs::read(root.join(key))
            .unwrap_or_else(|e| panic!("declared object {key} unreadable under {root:?}: {e}"))
    };
    out.insert("manifest.json".to_string(), read("manifest.json"));
    out.insert(
        manifest.directory.key.clone(),
        read(&manifest.directory.key),
    );
    for pack in &manifest.packs {
        out.insert(pack.key.clone(), read(&pack.key));
    }
    out
}

/// The byte comparison the reproducibility assertion rests on, factored out so
/// the negative guard below can drive the SAME code the positive test drives.
///
/// Returns one line per difference and an EMPTY vec when the two object maps
/// are byte-identical — it returns rather than asserts precisely so the guard
/// can feed it a corrupted archive and check that it speaks up.
fn object_diffs(a: &BTreeMap<String, Vec<u8>>, b: &BTreeMap<String, Vec<u8>>) -> Vec<String> {
    let mut diffs = Vec::new();
    for key in a.keys() {
        if !b.contains_key(key) {
            diffs.push(format!("missing from second archive: {key}"));
        }
    }
    for key in b.keys() {
        if !a.contains_key(key) {
            diffs.push(format!("extra in second archive: {key}"));
        }
    }
    for (key, bytes_a) in a {
        if let Some(bytes_b) = b.get(key) {
            if bytes_a != bytes_b {
                let at = bytes_a
                    .iter()
                    .zip(bytes_b.iter())
                    .position(|(x, y)| x != y)
                    .map(|i| format!("first differing byte at offset {i}"))
                    .unwrap_or_else(|| "identical prefix, differing length".to_string());
                diffs.push(format!(
                    "bytes differ: {key} ({} B vs {} B; {at})",
                    bytes_a.len(),
                    bytes_b.len()
                ));
            }
        }
    }
    diffs.sort();
    diffs
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

    // (4) The same comparison the negative guard below exercises, run over the
    // MANIFEST-declared object set: catches an object one build wrote and the
    // other did not, which the per-pack loop above (driven by `man_a` alone)
    // would walk straight past.
    let diffs = object_diffs(
        &archive_objects(out_a, &man_a),
        &archive_objects(out_b, &man_b),
    );
    assert!(
        diffs.is_empty(),
        "two pipeline runs over one feature stream produced different objects \
         ({} difference(s)):\n  {}",
        diffs.len(),
        diffs.join("\n  ")
    );

    // The shared key-set comparator is strictly weaker than the byte checks
    // above; assert it too so the "stronger ⇒ weaker" relationship is explicit
    // and any regression that trips one is understood against the other.
    common::assert_archives_equal(out_a, out_b, "pipeline-repro");
}

/// **Negative guard for the equality helper** (P0-7).
///
/// A reproducibility test whose comparator has quietly rotted into a tautology
/// — comparing a tree against itself, skipping the payloads, swallowing a
/// missing object — stays green forever while asserting nothing, and the two
/// historical catches this lane exists for (the Arrow-54 metadata order, the
/// FNV id migration) each cost a full 29.3 GiB / 1,324-object fleet re-upload.
/// So: build one archive, copy it, flip ONE bit of ONE pack byte, and assert
/// `object_diffs` — the same function `full_pipeline_is_byte_reproducible`
/// depends on — reports it, names the pack, and locates the offset. Then do the
/// same for a deleted object and an unexpected extra one, because "byte
/// equality" has to mean the object SET as well as the payloads.
///
/// Note what is NOT being guarded here: `common::assert_archives_equal`
/// compares per-tile `(z,x,y,t,count)` keys only and by design would NOT see a
/// flipped payload byte. The byte comparison is the strong one, so the byte
/// comparison is the one that needs the guard.
#[test]
fn byte_comparison_reports_a_single_flipped_pack_byte() {
    let src = tempfile::tempdir().unwrap();
    let parquet = src.path().join("input.parquet");
    write_fixture_parquet(&fixture_rows(), &parquet);
    let features = load_file(&parquet);

    let config = TileConfig {
        min_zoom: 0,
        max_zoom: 8,
        temporal_bucket_ms: 3_600_000,
        ..TileConfig::default()
    };

    // One build, then a faithful copy — the flip is then the ONLY difference
    // between the two trees, so anything the comparator reports is the flip.
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("good");
    let manifest = build_pipeline(&features, &config, &good);
    let corrupt = dir.path().join("corrupt");
    copy_tree(&good, &corrupt);

    let objects_good = archive_objects(&good, &manifest);
    assert!(
        object_diffs(&objects_good, &archive_objects(&corrupt, &manifest)).is_empty(),
        "a byte-for-byte copy must compare clean before the flip — otherwise \
         the guard below is measuring the copy, not the corruption"
    );

    // Flip one bit in the middle of the first pack object.
    let pack_key = manifest
        .packs
        .first()
        .expect("archive has a pack")
        .key
        .clone();
    let victim = corrupt.join(&pack_key);
    let mut bytes = std::fs::read(&victim).unwrap();
    assert!(!bytes.is_empty(), "pack {pack_key} is empty");
    let offset = bytes.len() / 2;
    bytes[offset] ^= 0x01;
    std::fs::write(&victim, &bytes).unwrap();

    let diffs = object_diffs(&objects_good, &archive_objects(&corrupt, &manifest));
    assert_eq!(
        diffs.len(),
        1,
        "one flipped byte must produce exactly one finding, got: {diffs:?}"
    );
    assert!(
        diffs[0].contains(&pack_key) && diffs[0].contains("bytes differ"),
        "the finding must name the corrupted pack, got: {}",
        diffs[0]
    );
    assert!(
        diffs[0].contains(&format!("offset {offset}")),
        "the finding must locate the flip at offset {offset}, got: {}",
        diffs[0]
    );

    // A vanished object and an unexpected one are the other two ways the trees
    // can diverge. `archive_objects` panics on a missing DECLARED object (that
    // is its contract), so drive these through plain maps.
    let mut without_pack = objects_good.clone();
    without_pack.remove(&pack_key);
    let missing = object_diffs(&objects_good, &without_pack);
    assert!(
        missing
            .iter()
            .any(|d| d.contains("missing from second archive") && d.contains(&pack_key)),
        "a vanished object must be reported as missing, got: {missing:?}"
    );

    let mut with_intruder = objects_good.clone();
    with_intruder.insert("packs/intruder.sttp".to_string(), b"not a pack".to_vec());
    let extra = object_diffs(&objects_good, &with_intruder);
    assert!(
        extra
            .iter()
            .any(|d| d.contains("extra in second archive") && d.contains("intruder.sttp")),
        "an unexpected object must be reported as extra, got: {extra:?}"
    );
}

/// Recursive directory copy (no external dev-dependency for three lines).
fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let path = entry.unwrap().path();
        let target = dst.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_tree(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

/// The same composition, driven through the **default** ordering path.
///
/// `stt-build` no longer defaults to `--blob-ordering auto`: it defaults to
/// `measured`, which resolves the concrete blob order by running the
/// range-read simulator over the dataset's own native tiles under per-dataset
/// query weights. That makes the ordering step a solver on the default path,
/// and every solver in this program lands with a byte-identical re-run test —
/// because the resolved order fixes the blob byte string, which fixes every
/// content-addressed pack name, which is the immutable-CDN contract.
///
/// The tiler runs on a rayon pool, so the simulator sees its samples in a
/// legitimately varying order run to run; the point is that the pick, and every
/// byte downstream of it, does not move.
#[test]
fn default_measured_ordering_pipeline_is_byte_reproducible() {
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
    let man_a = build_pipeline_with(&features, &config, out_a, true);
    let man_b = build_pipeline_with(&features, &config, out_b, true);

    assert!(
        man_a.packs.len() >= 2,
        "expected the fixture + 8-zoom build to roll across multiple packs, got {}",
        man_a.packs.len()
    );

    // The resolved ordering itself is stable, and it is a concrete order —
    // never the `auto`/`measured` markers.
    assert_eq!(
        man_a.blob_ordering, man_b.blob_ordering,
        "the measured pick must not depend on tile emission order"
    );
    let resolved = man_a
        .blob_ordering
        .clone()
        .expect("a measured build records the concrete order it laid down");
    assert!(
        matches!(
            resolved.as_str(),
            "spatial" | "time-major" | "hilbert3" | "morton3"
        ),
        "unexpected recorded ordering {resolved}"
    );

    // Every declared object, byte for byte, through the same comparator the
    // `auto`-path test uses (and whose negative guard lives below it).
    let diffs = object_diffs(
        &archive_objects(out_a, &man_a),
        &archive_objects(out_b, &man_b),
    );
    assert!(
        diffs.is_empty(),
        "two DEFAULT-path runs over one feature stream produced different objects \
         ({} difference(s)):\n  {}",
        diffs.len(),
        diffs.join("\n  ")
    );
    assert_eq!(
        man_a.to_json_bytes().unwrap(),
        man_b.to_json_bytes().unwrap(),
        "manifest JSON must be byte-identical across default-path rebuilds"
    );

    // The workload the layout was chosen under is co-versioned into the archive
    // — including the reader-mirroring coalescing gap — and is itself stable.
    let workload = man_a
        .ordering_workload
        .expect("a measured build records its workload");
    assert_eq!(man_b.ordering_workload, Some(workload));
    // ...at BOTH pinned keys: the canonical top-level one and the reader-compat
    // mirror inside `metadata` the shipped TS reader resolves the build-assumed
    // coalescing gap through.
    assert_eq!(man_a.metadata.ordering_workload, Some(workload));
    assert_eq!(man_b.metadata.ordering_workload, Some(workload));
    assert_eq!(
        (workload.scrub, workload.pan),
        (1, 1),
        "no family is starved"
    );
    assert_eq!(
        workload.coalesce_gap_bytes,
        stt_core::ordering_sim::DEFAULT_COALESCE_GAP_BYTES
    );

    common::assert_archives_equal(out_a, out_b, "pipeline-repro-measured");
}

/// BH-10's mandatory determinism test: with the DERIVED playback parameters
/// emitted, two runs over one feature stream are still byte-identical.
///
/// The derived window is a function of the pack writer's running payload total,
/// and that total is accumulated as tiles arrive from a rayon pool — so if the
/// counter were residency-based (or double-counted a spilled payload) the hint
/// would wobble run to run and every manifest byte after it with it. Nothing
/// else in the archive may move either: the derivation reads bytes, it does not
/// produce them.
#[test]
fn derived_playback_params_do_not_break_pipeline_byte_reproducibility() {
    use stt_build::style_hints::{compute_style_hints_with, DerivedPlaybackParams};

    let src = tempfile::tempdir().unwrap();
    let parquet = src.path().join("input.parquet");
    write_fixture_parquet(&fixture_rows(), &parquet);
    let features = load_file(&parquet);
    assert!(!features.is_empty());

    let config = TileConfig {
        min_zoom: 0,
        max_zoom: 8,
        temporal_bucket_ms: 3_600_000,
        ..TileConfig::default()
    };

    let build = |out: &Path| -> Manifest {
        let mut writer = PackWriter::create(out, BlobOrdering::Auto, 512).unwrap();
        generate_tiles_streaming(&features, &config, &mut writer, 2).unwrap();
        let (bounds, time_range) = input::calculate_bounds(&features).unwrap();
        let hints = compute_style_hints_with(
            &features,
            &time_range,
            config.temporal_bucket_ms,
            true,
            DerivedPlaybackParams {
                total_payload_bytes: Some(writer.payload_bytes()),
                refit: true,
            },
        )
        .expect("non-empty features profile");
        let metadata = Metadata::new("repro-derived")
            .with_bounds(bounds)
            .with_time_range(time_range)
            .with_zoom_levels(config.min_zoom, config.max_zoom)
            .with_temporal_bucket_ms(config.temporal_bucket_ms)
            .with_style_hints(hints);
        writer.finalize(&metadata).unwrap()
    };

    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let man_a = build(dir_a.path());
    let man_b = build(dir_b.path());

    let hints_a = man_a.metadata.style_hints.as_ref().expect("hints emitted");
    assert!(
        hints_a.suggested_time_window_ms.is_some(),
        "the derived window must actually be present, or this test is vacuous"
    );
    assert_eq!(
        hints_a,
        man_b.metadata.style_hints.as_ref().unwrap(),
        "the derived hints moved between two identical runs"
    );

    let diffs = object_diffs(
        &archive_objects(dir_a.path(), &man_a),
        &archive_objects(dir_b.path(), &man_b),
    );
    assert!(
        diffs.is_empty(),
        "derived playback params broke byte reproducibility ({} difference(s)):\n  {}",
        diffs.len(),
        diffs.join("\n  ")
    );
    assert_eq!(
        man_a.to_json_bytes().unwrap(),
        man_b.to_json_bytes().unwrap()
    );
}

/// MO-9's mandatory determinism test: the lever set `stt-build --target-size`
/// can now apply AUTOMATICALLY is byte-reproducible through the real pipeline.
///
/// # Why this shape, and why here
///
/// Two identical `--target-size` builds must produce byte-identical archives.
/// That decomposes into two claims:
///
/// 1. **same input ⇒ same recipe** — a property of the solver, pinned in
///    `stt-optimize` (`recommend_with` and `budget_solver::solve` each have a
///    byte-identical re-run test), and
/// 2. **same recipe ⇒ same bytes** — a property of THIS pipeline, which is what
///    this test adds.
///
/// The end-to-end composition is pinned at the binary seam in
/// `spatiotemporal-tiles/tests/build_auto_target.rs`
/// (`two_identical_target_size_builds_are_byte_identical`).
///
/// The configuration under test is deliberately not the default one: it is a
/// budget-shaped recipe — a CLAMPED max zoom, a WIDENED temporal bucket, and a
/// `--temporal-lod` tier pyramid. The tier matters most. Budget mode is the
/// first and only path that applies `--temporal-lod` without a human typing it,
/// and tiers run through `generate_tiles_with_lod` (the in-memory builder, which
/// `--temporal-lod` forces even under `--streaming`) — a code path this file's
/// other three tests, all `generate_tiles_streaming`, never touch. An
/// auto-applied lever whose pipeline was never under a byte-identity assertion
/// is exactly the kind of gap that costs a full fleet re-upload.
#[test]
fn the_budget_mode_lever_set_is_byte_reproducible() {
    use stt_build::tiler::{generate_tiles_with_lod, write_tiles_parallel};
    use stt_core::metadata::TemporalLodLevel;

    let src = tempfile::tempdir().unwrap();
    let parquet = src.path().join("input.parquet");
    write_fixture_parquet(&fixture_rows(), &parquet);
    let features = load_file(&parquet);
    assert!(!features.is_empty(), "fixture must yield tiled features");

    // A recipe of the shape the budget solver emits: zoom clamped from the
    // fixture's natural 8 down to 6, the bucket widened 4x off 1h, and two
    // additive lossless tiers as strict multiples of that widened base.
    let base_bucket_ms = 4 * 3_600_000;
    let config = TileConfig {
        min_zoom: 0,
        max_zoom: 6,
        temporal_bucket_ms: base_bucket_ms,
        temporal_lod: vec![
            TemporalLodLevel {
                bucket_ms: base_bucket_ms * 4,
                max_zoom_level: 6,
                contract: None,
                method: None,
            },
            TemporalLodLevel {
                bucket_ms: base_bucket_ms * 16,
                max_zoom_level: 4,
                contract: None,
                method: None,
            },
        ],
        ..TileConfig::default()
    };

    let build = |out: &Path| -> Manifest {
        // The publish-grade level the zstd sweep picks under budget pressure,
        // so the compressor is exercised where a budgeted build runs it.
        let mut writer = PackWriter::create(out, BlobOrdering::Auto, 512)
            .unwrap()
            .with_zstd_level(19);
        let tiles = generate_tiles_with_lod(&features, &config, 2).unwrap();
        let keep: Vec<(&stt_build::tiler::GeneratedTile, Option<u64>)> = tiles
            .iter()
            .map(|tagged| (&tagged.tile, tagged.temporal_bucket_ms))
            .collect();
        write_tiles_parallel(&mut writer, &keep, 2, || {}).unwrap();
        let (bounds, time_range) = input::calculate_bounds(&features).unwrap();
        let metadata = Metadata::new("repro-budget")
            .with_bounds(bounds)
            .with_time_range(time_range)
            .with_zoom_levels(config.min_zoom, config.max_zoom)
            .with_temporal_bucket_ms(config.temporal_bucket_ms)
            .with_temporal_lod(config.temporal_lod.clone())
            .expect("the tiers are strict multiples of the widened base bucket");
        writer.finalize(&metadata).unwrap()
    };

    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let man_a = build(dir_a.path());
    let man_b = build(dir_b.path());

    // Fixture guard: the tiers must actually have produced tiles, or the tier
    // half of this test is vacuous. LOD tiles are tagged with their own bucket
    // width, so the directory's distinct bucket widths are the evidence.
    assert!(
        man_a
            .metadata
            .temporal_lod
            .as_ref()
            .is_some_and(|l| l.len() == 2),
        "the recipe under test must carry its two tiers: {:?}",
        man_a.metadata.temporal_lod
    );
    assert!(
        man_a.metadata.tile_count > 0 && !man_a.packs.is_empty(),
        "no tiles were written, so nothing was compared"
    );

    let diffs = object_diffs(
        &archive_objects(dir_a.path(), &man_a),
        &archive_objects(dir_b.path(), &man_b),
    );
    assert!(
        diffs.is_empty(),
        "a budget-shaped recipe (zoom clamp + widened bucket + temporal-LOD tiers) \
         is not byte-reproducible ({} difference(s)):\n  {}",
        diffs.len(),
        diffs.join("\n  ")
    );
    assert_eq!(
        man_a.to_json_bytes().unwrap(),
        man_b.to_json_bytes().unwrap(),
        "the manifest moved between two identical budget-recipe builds"
    );
}

/// Build the pipeline with an EXPLICIT encoder config, so a pinned and an
/// unpinned build differ in exactly one field.
///
/// Routing both sides through `with_encoder_config` matters: that call is also
/// what merges the required-capability declarations into the manifest, so a
/// helper that only called it on the pinned side would compare two archives
/// that differ in `manifest.capabilities` and blame the pins for it.
fn build_pipeline_pinned(
    features: &[ParsedFeature],
    config: &TileConfig,
    out_dir: &Path,
    pins: Option<Arc<GlobalColumnPins>>,
) -> Manifest {
    let mut writer = PackWriter::create(out_dir, BlobOrdering::Auto, 512).unwrap();
    let cfg = stt_core::arrow_tile::EncoderConfig {
        global_pins: pins,
        ..writer.encoder_config()
    };
    writer = writer.with_encoder_config(cfg);
    generate_tiles_streaming(features, config, &mut writer, 2).unwrap();
    let (bounds, time_range) = input::calculate_bounds(features).unwrap();
    let metadata = Metadata::new("repro-pinned")
        .with_bounds(bounds)
        .with_time_range(time_range)
        .with_zoom_levels(config.min_zoom, config.max_zoom)
        .with_temporal_bucket_ms(config.temporal_bucket_ms);
    writer.finalize(&metadata).unwrap()
}

/// TB-1's mandatory determinism test: the composition is still byte-reproducible
/// with the dataset-global pins ATTACHED.
///
/// The pins are the artifact that will fix every downstream encoding verdict, so
/// if they wobble between two runs of one build the archive wobbles with them —
/// and because pack names are content addresses, a wobbling archive invalidates
/// the whole edge cache on a rebuild of unchanged data. The scan behind them is
/// a parallel fold over a rayon pool whose split points vary run to run, which
/// is exactly the shape that hides an order dependence, so pinning it here (at
/// the archive-bytes level, not only at the struct level) is the guard that
/// matters.
#[test]
fn pinned_pipeline_is_byte_reproducible() {
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

    // Pass 1 runs twice too — the pins must not move either, or nothing
    // downstream of them can be stable.
    let pins_a = pins_for(&features);
    let pins_b = pins_for(&features);
    assert_eq!(
        pins_a.to_canonical_json(),
        pins_b.to_canonical_json(),
        "two pass-1 scans over one feature stream resolved different pins"
    );
    // Fixture guard: the fixture really does have pinnable columns (a numeric
    // `mag`/`cnt` and a categorical `name`/`flag`), so this is not an
    // empty==empty pass.
    assert!(
        !pins_a.attr.is_empty() && !pins_a.dict.is_empty(),
        "the fixture must pin both a numeric and a categorical column: {pins_a:?}"
    );

    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let pins = Arc::new(pins_a);
    let man_a = build_pipeline_pinned(&features, &config, dir_a.path(), Some(pins.clone()));
    let man_b = build_pipeline_pinned(&features, &config, dir_b.path(), Some(pins.clone()));

    assert!(
        man_a.metadata.tile_count > 0 && !man_a.packs.is_empty(),
        "no tiles were written, so nothing was compared"
    );
    let diffs = object_diffs(
        &archive_objects(dir_a.path(), &man_a),
        &archive_objects(dir_b.path(), &man_b),
    );
    assert!(
        diffs.is_empty(),
        "two PINNED pipeline runs over one feature stream produced different objects \
         ({} difference(s)):\n  {}",
        diffs.len(),
        diffs.join("\n  ")
    );
    assert_eq!(
        man_a.to_json_bytes().unwrap(),
        man_b.to_json_bytes().unwrap(),
        "manifest JSON must be byte-identical across pinned rebuilds"
    );
    common::assert_archives_equal(dir_a.path(), dir_b.path(), "pipeline-repro-pinned");
}

/// **TB-1's byte-neutrality claim, made executable.**
///
/// Pass 1 only COMPUTES: it resolves the dataset's domain and attaches the
/// resulting pins to the encoder config. No encode path reads them yet, so an
/// archive built with pins attached must be byte-for-byte the archive built
/// without them. That is what lets the statistics pass land ahead of the
/// encoder changes that flip bytes, instead of dragging the whole R1 rebuild
/// window forward with it.
///
/// # When this test goes red
///
/// It is SUPPOSED to, exactly once, at the hand-off: the first item that makes
/// the encoder consult `EncoderConfig::global_pins` (the global attribute-range
/// pin, and then the global dict-vs-`Utf8` verdict) is the item that turns pins
/// into bytes. That item owns inverting this assertion — pinned and unpinned
/// archives then legitimately differ, and the byte-neutral claim moves to
/// "`--single-pass` reproduces a pre-M2 archive". Do NOT delete the test to make
/// it pass: a silent flip is exactly how a byte break ships without a rebuild
/// window, and this program has one window (R1) for all of them.
#[test]
fn attaching_pins_moves_no_archive_byte() {
    let src = tempfile::tempdir().unwrap();
    let parquet = src.path().join("input.parquet");
    write_fixture_parquet(&fixture_rows(), &parquet);
    let features = load_file(&parquet);

    let config = TileConfig {
        min_zoom: 0,
        max_zoom: 8,
        temporal_bucket_ms: 3_600_000,
        ..TileConfig::default()
    };

    let pins = Arc::new(pins_for(&features));
    assert!(!pins.is_empty(), "a vacuous pin set would prove nothing");

    let dir_pinned = tempfile::tempdir().unwrap();
    let dir_plain = tempfile::tempdir().unwrap();
    let man_pinned = build_pipeline_pinned(&features, &config, dir_pinned.path(), Some(pins));
    let man_plain = build_pipeline_pinned(&features, &config, dir_plain.path(), None);

    let diffs = object_diffs(
        &archive_objects(dir_pinned.path(), &man_pinned),
        &archive_objects(dir_plain.path(), &man_plain),
    );
    assert!(
        diffs.is_empty(),
        "pass 1 is byte-neutral by contract, but attaching its pins moved {} object(s). \
         If this is the item that starts CONSUMING pins, invert this assertion (see the \
         doc comment) rather than deleting it:\n  {}",
        diffs.len(),
        diffs.join("\n  ")
    );
    assert_eq!(
        man_pinned.to_json_bytes().unwrap(),
        man_plain.to_json_bytes().unwrap(),
        "attaching pins moved the manifest"
    );
}
