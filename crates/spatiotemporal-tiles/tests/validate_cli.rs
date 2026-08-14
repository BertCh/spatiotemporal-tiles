//! End-to-end tests for the `stt-validate` binary.
//!
//! Each test builds a tiny packed dataset in a temp dir via `stt_core`'s
//! `PackWriter` (the same writer the offline `stt-build` uses), then runs the
//! compiled binary (`CARGO_BIN_EXE_*`) over it with `--json` and asserts on the
//! parsed report. This exercises the real decode path, the schema checks, and
//! the `--sample` accounting end to end.
//!
//! Two halves:
//!
//! * **Positive** — a well-formed archive passes, and `--sample`/`--skip-decode`
//!   account for what they did and did not verify.
//! * **Negative** (`corrupt_*` / `*_fails`) — one deliberately broken archive per
//!   named check in `docs/spec/conformance.md`, each asserting a **non-zero
//!   exit** AND the specific error text. `stt-validate` is the executable
//!   specification: a validator whose only tests are green inputs would still
//!   pass if every check were deleted, so each of these is the proof that the
//!   corresponding check is wired to the exit code.
//!
//! Corrupt fixtures are always synthesised at test time — either written
//! deliberately wrong through `PackWriter` (which takes the feature count and
//! the cell ids from the caller, so a lying builder needs no byte surgery) or
//! produced by mutating a good archive's bytes in the temp dir. Nothing broken
//! is committed to the repo.

// The stt-validate binary (and CARGO_BIN_EXE_stt-validate) only exists when
// the `validate-cli` feature is on; compile the suite out otherwise.
#![cfg(feature = "validate-cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

use stt_core::arrow_tile::{
    encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn, PropertyColumn,
};
use stt_core::metadata::{Metadata, SummaryAggregation, SummaryColumn, SummaryScheme, SummaryTier};
use stt_core::types::TimeRange;
use stt_core::{BlobOrdering, Manifest, PackWriter, TileId};

/// Build a point layer of `n` features whose times fall inside [start, end].
fn point_layer(name: &str, base_id: u64, n: usize, start: i64, end: i64) -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: name.into(),
        feature_ids: (0..n as u64).map(|i| base_id + i).collect(),
        start_times: vec![start; n],
        end_times: vec![end; n],
        geometry: GeometryColumn::Point(
            (0..n).map(|i| [i as f64 * 0.01, i as f64 * 0.01]).collect(),
        ),
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
/// point features, into the directory `out_dir` — straight through `PackWriter`,
/// the same writer the offline `stt-build` uses. Returns the total feature count
/// so callers can assert the metadata grand total matches.
fn write_dataset(out_dir: &Path, tile_count: u32, per_tile: usize) -> u64 {
    write_dataset_declaring(out_dir, tile_count, per_tile, per_tile as u32)
}

/// The dataset writer, with the per-tile feature count the directory *declares*
/// split out from the number of rows actually encoded. They are equal for every
/// honest archive — `declared != per_tile` is how the negative test for check 6
/// (directory count vs decoded rows) builds a lying archive without touching a
/// byte: `PackWriter` takes the count from the caller, exactly the trust an
/// upstream builder bug would abuse.
fn write_dataset_declaring(
    out_dir: &Path,
    tile_count: u32,
    per_tile: usize,
    declared_per_tile: u32,
) -> u64 {
    let mut writer = PackWriter::create(out_dir, BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
    // Frames follow the writer's (default v2) formatVersion + template
    // collector, exactly like stt-build's encoder wiring — `add_tile_full`
    // enforces frame/manifest version coherence.
    let cfg = EncoderConfig {
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(writer.template_collector()),
        ..EncoderConfig::default()
    };
    let t_start = 1_000i64;
    let t_end = 2_000i64;
    let mut features = 0u64;
    for x in 0..tile_count {
        let id = TileId::new(8, x, 0, t_start as u64);
        let layer = point_layer("default", x as u64 * 1000, per_tile, t_start, t_end);
        let payload = encode_tile_with(&[layer], &cfg).unwrap();
        writer
            .add_tile_full(&id, t_start, t_end, None, declared_per_tile, None, &payload)
            .unwrap();
        features += per_tile as u64;
    }
    let mut meta = Metadata::new("test")
        .with_time_range(TimeRange::new(t_start as u64, t_end as u64))
        .with_zoom_levels(8, 8);
    meta.feature_count = features;
    meta.tile_count = tile_count as u64;
    writer.finalize(&meta).unwrap();
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
    assert!(
        ok,
        "sampled run of a valid archive should still pass: {report}"
    );
    assert_eq!(report["tile_count"], 20);
    assert_eq!(report["tiles_decoded"], 5, "sample must cap decoded count");
    assert_eq!(report["sampled"], true);
    // Grand-total feature check must be skipped (not spuriously failing).
    assert_eq!(report["feature_count_decoded_complete"], false);
    assert_eq!(report["errors"].as_array().unwrap().len(), 0);
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

/// Every payload shape the encoder can emit must be *accepted* by the
/// validator, not merely decodable by it. `stt-validate` is a closed allow-list
/// (column names, leaf types, required metadata), so an encoder that grows a
/// column or a leaf width WITHOUT the matching entry here turns every rebuilt
/// archive red — one error per layer per tile — while the archive itself is
/// perfectly fine. That failure mode is invisible to the encoder's own unit
/// tests, which never run the validator, so it belongs here.
///
/// This archive deliberately carries, in one dataset:
///   * `part_offsets` — a two-part polygon (`List<UInt32>`, reserved column);
///   * a `List<UInt32>` `vertex_time` — a track spanning 3 days, i.e. past the
///     u16 delta tier's ~18.2 h reach at the 1 ms step ceiling;
///   * quantized per-vertex values (`--quantize-vertex-values`), which the
///     merge re-inflate must make invisible (the validator should see
///     `List<Float32>`, never the `UInt16` wire leaf).
#[test]
fn every_encoder_payload_shape_passes_the_validator() {
    use arrow::datatypes::DataType;

    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("shapes");

    let day = 24 * 60 * 60 * 1000i64;
    let t_start = 1_700_000_000_000i64;
    let t_end = t_start + 3 * day;

    let mut writer = PackWriter::create(&archive, BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
    let cfg = EncoderConfig {
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(writer.template_collector()),
        // The one lossy, opt-in encoding — on here precisely because the
        // validator must NOT be able to tell.
        quantize_vertex_values: true,
        ..EncoderConfig::default()
    };

    // Tile 1: a multi-part polygon (two disjoint squares in one feature, the
    // second square holed) → `part_offsets` + `triangles`.
    let square =
        |x: f64, y: f64, s: f64| vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]];
    let polygons = ColumnarLayer {
        // Part 0 = ring 0; part 1 = rings 1..2 (exterior + hole).
        polygon_parts: Some(vec![vec![0, 1]]),
        name: "areas".into(),
        feature_ids: vec![1],
        start_times: vec![t_start],
        end_times: vec![t_end],
        geometry: GeometryColumn::Polygon(vec![vec![
            square(0.0, 0.0, 1.0),
            square(10.0, 0.0, 1.0),
            square(10.25, 0.25, 0.5),
        ]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    };

    // Tile 2: a 3-day track → u32 vertex-time deltas, plus per-vertex values
    // (quantized on the wire) and a 2-bucket value matrix.
    let tracks = ColumnarLayer {
        polygon_parts: None,
        name: "tracks".into(),
        feature_ids: vec![2],
        start_times: vec![t_start],
        end_times: vec![t_end],
        geometry: GeometryColumn::LineString(vec![vec![
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
        ]]),
        vertex_times: Some(vec![vec![t_start, t_start + day, t_start + 2 * day, t_end]]),
        vertex_values: Some(vec![vec![1.0, 2.5, f32::NAN, 9.75]]),
        triangles: None,
        vertex_value_matrix: Some(vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]),
        properties: vec![("speed".into(), PropertyColumn::Numeric(vec![Some(12.5)]))],
    };

    for (i, layer) in [polygons, tracks].into_iter().enumerate() {
        let id = TileId::new(8, i as u32, 0, t_start as u64);
        let payload = encode_tile_with(&[layer], &cfg).unwrap();
        writer
            .add_tile_full(&id, t_start, t_end, None, 1, None, &payload)
            .unwrap();
    }
    let mut meta = Metadata::new("shapes")
        .with_time_range(TimeRange::new(t_start as u64, t_end as u64))
        .with_zoom_levels(8, 8);
    meta.feature_count = 2;
    meta.tile_count = 2;
    writer.finalize(&meta).unwrap();

    let (ok, report) = run_json(&archive, &[]);
    assert!(
        ok,
        "an archive using every encoder payload shape must validate clean; report: {report}"
    );
    assert_eq!(
        report["errors"].as_array().unwrap().len(),
        0,
        "report: {report}"
    );
    assert_eq!(report["tiles_decoded"], 2);
    assert_eq!(report["feature_count_decoded"].as_u64().unwrap(), 2);

    // Guard the guard. A clean report proves nothing if the corpus quietly
    // stopped producing the shapes it is supposed to cover — so re-decode the
    // archive and assert each one is really on the wire.
    let reader = stt_core::PackedReader::open(archive.join("manifest.json")).unwrap();
    let registry = reader.templates().expect("v2 archive exposes templates");
    let mut saw_part_offsets = false;
    let mut saw_u32_vertex_time = false;
    let mut saw_float32_vertex_value = false;
    for entry in reader.entries() {
        let payload = reader.read_payload(entry).unwrap();
        for layer in stt_core::arrow_tile::decode_tile_with_templates(&payload, registry).unwrap() {
            let schema = layer.batch.schema();
            let leaf = |col: &str| match schema.field_with_name(col).map(|f| f.data_type().clone())
            {
                Ok(DataType::List(inner)) => Some(inner.data_type().clone()),
                _ => None,
            };
            if leaf("part_offsets") == Some(DataType::UInt32) {
                saw_part_offsets = true;
            }
            if leaf("vertex_time") == Some(DataType::UInt32) {
                saw_u32_vertex_time = true;
            }
            // Quantized on the wire, Float32 after the merge re-inflate.
            if leaf("vertex_value") == Some(DataType::Float32) {
                saw_float32_vertex_value = true;
            }
        }
    }
    assert!(saw_part_offsets, "corpus lost its multi-part polygon");
    assert!(saw_u32_vertex_time, "corpus lost its u32 vertex-time tier");
    assert!(
        saw_float32_vertex_value,
        "quantized vertex values must re-inflate to List<Float32> before any \
         consumer (including the validator) sees them"
    );
}

/// Write one tile per supplied layer (z8, x = index, one shared time bucket).
///
/// Unlike [`write_dataset`] the caller controls each tile's layer wholesale,
/// which is what the schema-drift tests need: two tiles carrying the SAME layer
/// NAME whose column sets (or column types) differ. Drift detection is keyed on
/// the layer name, so tiles holding differently-named layers are never compared
/// against each other.
fn write_tiles_of(archive: &Path, layers: Vec<ColumnarLayer>) {
    let t_start = 1_700_000_000_000i64;
    let t_end = t_start + 60_000;
    let mut writer = PackWriter::create(archive, BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
    let cfg = EncoderConfig {
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(writer.template_collector()),
        ..EncoderConfig::default()
    };
    let tile_count = layers.len() as u64;
    let mut features = 0u64;
    for (i, layer) in layers.into_iter().enumerate() {
        let n = layer.feature_ids.len() as u32;
        features += u64::from(n);
        let payload = encode_tile_with(&[layer], &cfg).unwrap();
        writer
            .add_tile_full(
                &TileId::new(8, i as u32, 0, t_start as u64),
                t_start,
                t_end,
                None,
                n,
                None,
                &payload,
            )
            .unwrap();
    }
    let mut meta = Metadata::new("drift")
        .with_time_range(TimeRange::new(t_start as u64, t_end as u64))
        .with_zoom_levels(8, 8);
    meta.feature_count = features;
    meta.tile_count = tile_count;
    writer.finalize(&meta).unwrap();
}

/// A closed square ring, so the polygon fixtures below read as shapes.
fn square(x: f64, y: f64, s: f64) -> Vec<[f64; 2]> {
    vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]
}

/// One layer, two tiles, `part_offsets` in only ONE of them → must PASS.
///
/// Every OPTIONAL reserved column is emitted per TILE, only when that tile's
/// features need it: `part_offsets` only when some feature is multi-part,
/// `triangles` only when some feature needs hole-aware tessellation, the vertex
/// columns only when per-vertex arrays exist. A polygon dataset therefore ships
/// tiles of ONE layer that disagree about which of those columns are present —
/// the common case, since the tiler emits a MultiPolygon exactly where clipping
/// cuts a source polygon into pieces, i.e. in some tiles and not others.
///
/// The drift check classified either side being absent as `Structural` and hard
/// -failed the whole archive (`schema drift: ... — part_offsets: <absent> vs
/// List(...UInt32...)`), contradicting the schema module's own documentation.
/// The presence variation must instead be reported as an informational
/// WARNING — visible, never fatal — so this asserts both halves.
#[test]
fn optional_reserved_column_present_in_only_one_tile_passes() {
    use arrow::datatypes::DataType;

    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("mixed-parts");

    let t_start = 1_700_000_000_000i64;
    let t_end = t_start + 60_000;
    // Same layer NAME, same geometry kind, same properties: `part_offsets` is
    // the only thing that can differ between these two tiles.
    let areas = |id: u64, rings: Vec<Vec<[f64; 2]>>, parts: Vec<u32>| ColumnarLayer {
        polygon_parts: Some(vec![parts]),
        name: "areas".into(),
        feature_ids: vec![id],
        start_times: vec![t_start],
        end_times: vec![t_end],
        geometry: GeometryColumn::Polygon(vec![rings]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "speed".into(),
            PropertyColumn::Numeric(vec![Some(id as f64)]),
        )],
    };
    write_tiles_of(
        &archive,
        vec![
            // Tile 0: two disjoint squares in one feature → parts at rings 0, 1.
            areas(
                1,
                vec![square(0.0, 0.0, 1.0), square(10.0, 0.0, 1.0)],
                vec![0, 1],
            ),
            // Tile 1: a single square → single-part, so no `part_offsets`.
            areas(2, vec![square(20.0, 0.0, 1.0)], vec![0]),
        ],
    );

    let (ok, report) = run_json(&archive, &[]);
    assert!(
        ok,
        "a layer whose tiles differ only in an OPTIONAL reserved column must \
         validate clean; report: {report}"
    );
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    assert_eq!(report["tiles_decoded"], 2);

    // The variation must stay VISIBLE: informational, not silently swallowed.
    let warnings: Vec<String> = report["warnings"]
        .as_array()
        .expect("report carries a warnings array")
        .iter()
        .map(|w| w.as_str().unwrap().to_string())
        .collect();
    assert!(
        warnings.iter().any(|w| w.contains("part_offsets")),
        "the presence variation must still be reported (as a warning); \
         warnings were: {warnings:#?}"
    );

    // Guard the guard: a green report proves nothing if the fixture stopped
    // producing the mixed-presence corpus it exists to cover.
    let reader = stt_core::PackedReader::open(archive.join("manifest.json")).unwrap();
    let registry = reader.templates().expect("v2 archive exposes templates");
    let mut with_parts = 0;
    let mut without_parts = 0;
    for entry in reader.entries() {
        let payload = reader.read_payload(entry).unwrap();
        for layer in stt_core::arrow_tile::decode_tile_with_templates(&payload, registry).unwrap() {
            assert_eq!(layer.name, "areas", "both tiles must share one layer name");
            match layer.batch.schema().field_with_name("part_offsets") {
                Ok(f) => {
                    assert!(matches!(f.data_type(), DataType::List(_)));
                    with_parts += 1;
                }
                Err(_) => without_parts += 1,
            }
        }
    }
    assert_eq!(
        (with_parts, without_parts),
        (1, 1),
        "fixture must carry part_offsets in exactly one of its two tiles"
    );
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

// ---------------------------------------------------------------------------
// Negative end-to-end tests — one corrupt archive per named check.
//
// `docs/spec/conformance.md` calls stt-validate "the executable specification",
// which only means anything if a broken input actually turns it red. Every test
// below asserts BOTH halves of that: a non-zero process exit AND the specific
// error the check is supposed to name. Asserting only "exit != 0" would pass if
// the binary failed for an unrelated reason; asserting only on the message would
// pass if the check reported but never affected the exit code (the difference
// between an error and a warning here).
// ---------------------------------------------------------------------------

/// The `errors` array of a report, as owned strings.
fn errors_of(report: &serde_json::Value) -> Vec<String> {
    report["errors"]
        .as_array()
        .expect("report must carry an errors array")
        .iter()
        .map(|e| e.as_str().expect("errors are strings").to_string())
        .collect()
}

/// Assert the run failed (non-zero exit) and at least one error contains
/// `needle`. Both halves matter — see the section comment.
fn assert_failed_with(ok: bool, report: &serde_json::Value, needle: &str) -> Vec<String> {
    let errors = errors_of(report);
    assert!(
        !ok,
        "corrupt archive must exit NON-ZERO (a reported-but-tolerated finding is \
         a warning, not a validation failure); report: {report}"
    );
    assert!(
        errors.iter().any(|e| e.contains(needle)),
        "no error named {needle:?}; errors were: {errors:#?}"
    );
    errors
}

/// Read a packed archive's `manifest.json`, hand the parsed manifest to
/// `mutate`, and write it back.
///
/// The manifest is the one file in a packed archive that is NOT
/// content-addressed (it is the root pointer), so editing it corrupts the
/// dataset while leaving every object name valid — the only way to test a
/// manifest-vs-directory disagreement in isolation.
fn mutate_manifest(archive: &Path, mutate: impl FnOnce(&mut Manifest)) {
    let path = archive.join("manifest.json");
    let mut manifest = Manifest::from_json_bytes(&std::fs::read(&path).unwrap()).unwrap();
    mutate(&mut manifest);
    std::fs::write(&path, manifest.to_json_bytes().unwrap()).unwrap();
}

/// Absolute path of the pack object holding the archive's first tile blob,
/// plus that blob's `[offset, length)` window inside it. v2 blob offsets are
/// object-absolute (the 8-byte magic prelude occupies `[0, 8)`), so the window
/// indexes the file directly.
fn first_blob_window(archive: &Path) -> (PathBuf, u64, u32) {
    let manifest_path = archive.join("manifest.json");
    let manifest = Manifest::from_json_bytes(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let reader = stt_core::PackedReader::open(&manifest_path).unwrap();
    let entry = reader.entries().first().expect("archive has tiles").clone();
    let pack = archive.join(&manifest.packs[entry.pack_id as usize].key);
    // Drop the reader (and its mmap) before the caller rewrites the file.
    drop(reader);
    (pack, entry.offset, entry.length)
}

/// (a) A pack whose blake3 filename no longer describes its content.
///
/// Appending a byte leaves every tile blob — and therefore every CRC — intact,
/// so this isolates the content-addressing tier: the object still *reads*, it
/// just is not the object the manifest named. That distinction is the whole
/// point of content addressing (a CDN or mirror that served a stale/rewritten
/// pack under the right name would look fine at every other layer).
#[test]
fn pack_that_does_not_match_its_content_address_fails() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("addr");
    write_dataset(&archive, 4, 3);

    let (pack, _, _) = first_blob_window(&archive);
    let mut bytes = std::fs::read(&pack).unwrap();
    bytes.push(0x00);
    std::fs::write(&pack, &bytes).unwrap();

    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "content-address mismatch");
    assert!(
        errors.iter().any(|e| e.contains("on-disk length")),
        "the declared-length check must fire too; errors: {errors:#?}"
    );
    assert!(
        !errors.iter().any(|e| e.contains("failed integrity check")),
        "appending past the last blob must NOT trip a per-tile CRC — this test \
         pins the content-address tier on its own; errors: {errors:#?}"
    );
}

/// (b) A CRC32C failure inside a tile blob.
///
/// Flipping a byte *inside* the first blob's window is the corruption a bad
/// disk or a truncated transfer produces: the read path must refuse the tile
/// rather than hand torn bytes to the Arrow decoder.
///
/// The content-address error rides along unavoidably (a blob byte is part of
/// the pack the pack name hashes, and re-naming the pack would need the
/// blake3-128 helper, which `stt-core` keeps private) — so this asserts the
/// per-tile CRC message specifically rather than counting errors.
#[test]
fn corrupt_tile_blob_fails_the_crc_check() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("crc");
    write_dataset(&archive, 4, 3);

    let (pack, offset, length) = first_blob_window(&archive);
    let mut bytes = std::fs::read(&pack).unwrap();
    let victim = offset as usize + (length as usize / 2);
    bytes[victim] ^= 0xff;
    std::fs::write(&pack, &bytes).unwrap();

    let errors = {
        let (ok, report) = run_json(&archive, &[]);
        assert_failed_with(ok, &report, "failed integrity check (corrupt pack)")
    };
    assert!(
        errors.iter().any(|e| e.contains("payload read failed")),
        "the CRC failure must surface as a per-tile payload read failure; \
         errors: {errors:#?}"
    );
}

/// (c) A tile whose temporal extent falls outside the manifest's declared
/// `time_range`.
///
/// A reader prunes by the metadata range before it ever looks at the
/// directory, so a tile outside it is unreachable data: the archive renders
/// short with nothing logged anywhere. Narrowing the declared range (a
/// manifest-only edit — no object is re-addressed) reproduces exactly what an
/// extent computed over a pre-filter subset would ship.
#[test]
fn tile_outside_declared_time_range_fails() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("temporal");
    write_dataset(&archive, 3, 3);

    // Tiles span [1000, 2000]; declare a range strictly inside that.
    mutate_manifest(&archive, |m| {
        m.metadata.time_range = TimeRange::new(1_500, 1_600);
    });

    let (ok, report) = run_json(&archive, &[]);
    assert_failed_with(
        ok,
        &report,
        "temporal extent [1000, 2000] outside archive range [1500, 1600]",
    );
}

/// Write a summary-tier archive whose `summary` layer carries `ids` as its `id`
/// column, and whose metadata declares an H3 tier at resolution 5 for zoom 8.
///
/// The ids are the ONLY variable: the caller passes real H3 cells for the
/// control and a sequential counter for the regression.
fn write_summary_dataset(out_dir: &Path, ids: &[u64]) {
    let mut writer = PackWriter::create(out_dir, BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
    let cfg = EncoderConfig {
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(writer.template_collector()),
        ..EncoderConfig::default()
    };
    let (t_start, t_end) = (1_000i64, 2_000i64);
    let n = ids.len();
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "summary".into(),
        feature_ids: ids.to_vec(),
        start_times: vec![t_start; n],
        end_times: vec![t_end; n],
        geometry: GeometryColumn::Point((0..n).map(|i| [i as f64 * 0.01, 40.0]).collect()),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "count".into(),
            PropertyColumn::Numeric((0..n).map(|i| Some(i as f64 + 1.0)).collect()),
        )],
    };
    let payload = encode_tile_with(&[layer], &cfg).unwrap();
    writer
        .add_tile_full(
            &TileId::new(8, 0, 0, t_start as u64),
            t_start,
            t_end,
            None,
            n as u32,
            None,
            &payload,
        )
        .unwrap();
    let meta = Metadata::new("summary-test")
        .with_time_range(TimeRange::new(t_start as u64, t_end as u64))
        .with_zoom_levels(8, 8)
        .with_summary_tier(SummaryTier {
            variant_id: stt_core::tile::SUMMARY_VARIANT_ID,
            scheme: SummaryScheme::H3,
            min_zoom: 8,
            max_zoom: 8,
            cell_resolution_per_zoom: vec![5],
            columns: vec![SummaryColumn {
                name: "count".into(),
                agg: SummaryAggregation::Count,
            }],
            layer_name: "summary".into(),
            sub_buckets: 1,
        });
    writer.finalize(&meta).unwrap();
}

/// Three H3 cells at resolution 5 — the resolution the tier above maps zoom 8
/// to — minted with the same `h3o` crate the builder mints cells with.
fn h3_res5_cells() -> Vec<u64> {
    [(40.0, -105.0), (40.1, -105.1), (40.2, -105.2)]
        .iter()
        .map(|&(lat, lng)| {
            u64::from(
                h3o::LatLng::new(lat, lng)
                    .unwrap()
                    .to_cell(h3o::Resolution::Five),
            )
        })
        .collect()
}

/// (d) THE BLANK-DEMO BUG. A summary-tier layer whose `id` column holds a
/// sequential counter instead of the aggregation cell index.
///
/// The client reconstructs each cell's polygon from the id alone, so sequential
/// ids draw nothing: no decode error, no schema error, no HTTP error — three
/// shipped demo datasets rendered blank this way and the archives "validated"
/// clean. This is the highest-value case in the file, so it is paired with the
/// control below: the SAME archive shape with real H3 cells must pass, which is
/// what proves the check discriminates rather than rejecting all summary tiers.
#[test]
fn summary_tier_with_sequential_ids_fails() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("blank-demo");
    write_summary_dataset(&archive, &[0, 1, 2]);

    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "are not valid H3 cells at resolution 5");
    assert!(
        errors.iter().any(|e| e.contains("renders blank")),
        "the error must name the symptom an operator would otherwise chase in \
         the browser; errors: {errors:#?}"
    );
}

/// The control for the case above: identical archive, real H3 cells, passes.
/// Without this, `summary_tier_with_sequential_ids_fails` would also pass if
/// the check rejected every summary archive outright.
#[test]
fn summary_tier_with_real_h3_cells_passes() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("good-summary");
    write_summary_dataset(&archive, &h3_res5_cells());

    let (ok, report) = run_json(&archive, &[]);
    assert!(ok, "valid summary archive must pass; report: {report}");
    assert_eq!(errors_of(&report).len(), 0);
}

/// (e) A directory `feature_count` that disagrees with the rows actually in the
/// tile.
///
/// The count is what a reader budgets buffers and progress from without
/// decoding, so a lying directory is a silent over-allocation upstream and a
/// wrong "N features" everywhere it is displayed. `PackWriter` takes the count
/// from its caller, so this archive is built dishonestly rather than mutated.
#[test]
fn feature_count_disagreeing_with_decoded_rows_fails() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("miscount");
    // 3 tiles × 4 real rows, each declaring 5.
    write_dataset_declaring(&archive, 3, 4, 5);

    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "directory says 5 features, decoded 4");
    assert_eq!(
        errors
            .iter()
            .filter(|e| e.contains("directory says 5 features, decoded 4"))
            .count(),
        3,
        "every miscounted tile must be named, not just the first; errors: {errors:#?}"
    );
    // The manifest total is derived from the directory at finalize, so it
    // inherits the lie — the grand-total check catches the same defect from the
    // other side.
    assert!(
        errors
            .iter()
            .any(|e| e.contains("disagrees with decoded sum 12")),
        "errors: {errors:#?}"
    );
    assert_eq!(report["feature_count_index"], 15);
    assert_eq!(report["feature_count_decoded"], 12);
}

/// (f) A PROPERTY column that changes encoding class between two tiles of one
/// layer.
///
/// The counterpart to `optional_reserved_column_present_in_only_one_tile_passes`
/// above, and the reason that relaxation is scoped to reserved columns: a
/// property arriving as `Dictionary<UInt16,Utf8>` in one tile and `Float64` in
/// the next hands every consumer a different thing per tile (a string category
/// vs a number), with no documented default to fall back on. Widening the drift
/// check must not launder THIS.
#[test]
fn property_column_type_drift_across_tiles_fails() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("prop-drift");

    let t_start = 1_700_000_000_000i64;
    let t_end = t_start + 60_000;
    // One layer name, one geometry kind; only `speed`'s encoding differs.
    let layer = |id: u64, speed: PropertyColumn| ColumnarLayer {
        polygon_parts: None,
        name: "default".into(),
        feature_ids: vec![id],
        start_times: vec![t_start],
        end_times: vec![t_end],
        geometry: GeometryColumn::Point(vec![[id as f64, 1.0]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![("speed".into(), speed)],
    };
    write_tiles_of(
        &archive,
        vec![
            layer(1, PropertyColumn::Numeric(vec![Some(12.5)])),
            layer(2, PropertyColumn::Categorical(vec![Some("fast".into())])),
        ],
    );

    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "schema drift");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("schema drift") && e.contains("speed")),
        "the drift error must name the offending property column; errors: {errors:#?}"
    );
    assert_eq!(report["distinct_schemas"], 2, "report: {report}");
}

/// `--skip-decode` must not silently launder a corrupt archive: the cheap tier
/// (content addressing, CRC, temporal bounds) still runs over every tile, so a
/// corrupt blob is still a non-zero exit even when nothing is decoded. Without
/// this, the fast path would be a way to make any of the above go green.
#[test]
fn skip_decode_still_fails_a_corrupt_blob() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("crc-skip");
    write_dataset(&archive, 4, 3);

    let (pack, offset, length) = first_blob_window(&archive);
    let mut bytes = std::fs::read(&pack).unwrap();
    bytes[offset as usize + (length as usize / 2)] ^= 0xff;
    std::fs::write(&pack, &bytes).unwrap();

    let (ok, report) = run_json(&archive, &["--skip-decode"]);
    assert_failed_with(ok, &report, "failed integrity check (corrupt pack)");
    assert_eq!(report["tiles_decoded"], 0, "report: {report}");
}

/// An archive built through the **default** ordering path — `stt-build`'s
/// `--blob-ordering measured` with derived query weights — validates clean, and
/// carries both halves of the layout record: the concrete order it laid down
/// (`blobOrdering`) and the workload model that order was chosen under
/// (`orderingWorkload`, mirrored at `metadata.ordering_workload`).
///
/// The pairing is the point. `blobOrdering` alone says WHAT the layout is;
/// without the workload beside it there is no way to tell a layout that is
/// still optimal from one chosen under a weighting — or a reader coalescing gap
/// — that has since moved.
#[test]
fn default_measured_build_validates_and_records_its_ordering_and_workload() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("measured");

    let mut writer = PackWriter::create(&archive, BlobOrdering::Auto, 64 * 1024 * 1024)
        .unwrap()
        .with_measured_ordering(true);
    let cfg = EncoderConfig {
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(writer.template_collector()),
        ..EncoderConfig::default()
    };
    let bucket = 3_600_000i64;
    let cells = 5u32;
    let buckets = 8i64;
    let mut features = 0u64;
    for b in 0..buckets {
        for x in 0..cells {
            let t = b * bucket;
            let id = TileId::new(8, 40 + x, 60, t as u64);
            let layer = point_layer(
                "default",
                u64::from(x) * 10_000 + b as u64 * 100,
                3,
                t,
                t + bucket - 1,
            );
            let payload = encode_tile_with(&[layer], &cfg).unwrap();
            writer
                .add_tile_full(&id, t, t + bucket - 1, Some(t), 3, None, &payload)
                .unwrap();
            features += 3;
        }
    }
    let mut meta = Metadata::new("measured-default")
        .with_time_range(TimeRange::new(0, (buckets * bucket - 1) as u64))
        .with_zoom_levels(8, 8)
        .with_temporal_bucket_ms(bucket as u64);
    meta.feature_count = features;
    meta.style_hints = Some(stt_core::metadata::StyleHints {
        version: 1,
        properties: Vec::new(),
        suggested_playback_seconds: None,
        suggested_time_window_ms: None,
        layer_hint: Some("points".to_string()),
    });
    let manifest = writer.finalize(&meta).unwrap();

    let (ok, report) = run_json(&archive, &[]);
    assert!(
        ok,
        "a default-path (measured) archive must validate clean; report: {report}"
    );
    assert_eq!(report["errors"].as_array().unwrap().len(), 0);

    // The concrete order is recorded, never the `auto`/`measured` markers.
    let ordering = manifest
        .blob_ordering
        .clone()
        .expect("a measured build records blobOrdering");
    assert!(
        matches!(
            ordering.as_str(),
            "spatial" | "time-major" | "hilbert3" | "morton3"
        ),
        "unexpected blobOrdering {ordering}"
    );

    // ...and the workload it was chosen under, including the reader-mirroring
    // coalescing gap the simulation priced at. Recorded at BOTH pinned keys:
    // the canonical top-level `orderingWorkload` (beside `blobOrdering`) and the
    // reader-compat mirror `metadata.ordering_workload` the shipped TS reader
    // resolves the build-assumed gap through.
    let workload = manifest
        .ordering_workload
        .expect("a measured build records its ordering workload");
    assert_eq!(
        manifest.metadata.ordering_workload,
        Some(workload),
        "the metadata mirror must carry the identical object"
    );
    assert_eq!(
        (workload.scrub, workload.pan, workload.playback),
        (1, 1, 2),
        "points over {buckets} buckets is the playback-dominant row"
    );
    assert_eq!(
        workload.coalesce_gap_bytes,
        stt_core::ordering_sim::DEFAULT_COALESCE_GAP_BYTES
    );
    assert!(workload.runway_multiplier >= 1);
    assert!(workload.playback_window_buckets >= 1);

    // Both facts survive a manifest round-trip through the on-disk JSON — this
    // is what `stt-optimize order-audit` reads back to detect drift.
    let bytes = std::fs::read(archive.join("manifest.json")).unwrap();
    let reparsed = Manifest::from_json_bytes(&bytes).unwrap();
    assert_eq!(reparsed.blob_ordering, manifest.blob_ordering);
    assert_eq!(reparsed.ordering_workload, Some(workload));
    assert_eq!(reparsed.metadata.ordering_workload, Some(workload));

    // Both keys are pinned by docs/spec/manifest.schema.json; assert the wire
    // shape a third-party reader would see, not just the Rust round-trip.
    let raw: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        raw["orderingWorkload"], raw["metadata"]["ordering_workload"],
        "the two co-versioning keys must serialize identically"
    );
    assert!(raw["orderingWorkload"]["coalesce_gap_bytes"].is_u64());
}

/// BH-10 schema case: an archive carrying the DERIVED playback hints validates
/// clean, and both hint values sit inside the bounds
/// `docs/spec/manifest.schema.json` declares for them.
///
/// The validator does not (today) enforce `style_hints` field ranges, so this is
/// the pin that keeps the writer and the published schema from drifting: the
/// refit widened `suggested_playback_seconds` from `[20, 90]` to `[5, 300]`, and
/// a writer that emits outside the declared range would otherwise ship a
/// manifest no third-party schema check accepts, with nothing red.
#[test]
fn derived_playback_hints_validate_and_stay_inside_the_published_schema_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("derived-hints");

    let mut writer =
        PackWriter::create(&archive, BlobOrdering::SpatialMajor, 64 * 1024 * 1024).unwrap();
    let cfg = EncoderConfig {
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(writer.template_collector()),
        ..EncoderConfig::default()
    };
    let bucket = 3_600_000i64;
    let buckets = 6i64;
    for b in 0..buckets {
        let t = b * bucket;
        let id = TileId::new(6, 18, 24, t as u64);
        let layer = point_layer("default", b as u64 * 100, 3, t, t + bucket - 1);
        let payload = encode_tile_with(&[layer], &cfg).unwrap();
        writer
            .add_tile_full(&id, t, t + bucket - 1, Some(t), 3, None, &payload)
            .unwrap();
    }
    // The two derived values, at the extremes the schema has to admit: the
    // refit's floor (5 s) and a window pinned to one bucket.
    let mut meta = Metadata::new("derived-hints")
        .with_time_range(TimeRange::new(0, (buckets * bucket - 1) as u64))
        .with_zoom_levels(6, 6)
        .with_temporal_bucket_ms(bucket as u64);
    meta.style_hints = Some(stt_core::metadata::StyleHints {
        version: 1,
        properties: Vec::new(),
        suggested_playback_seconds: Some(5),
        suggested_time_window_ms: Some(bucket as u64),
        layer_hint: Some("points".to_string()),
    });
    writer.finalize(&meta).unwrap();

    let (ok, report) = run_json(&archive, &[]);
    assert!(ok, "derived hints must validate clean; report: {report}");
    assert_eq!(report["errors"].as_array().unwrap().len(), 0);

    // The published schema's declared bounds, read from the file the TS contract
    // test also reads — so this cannot drift into asserting a private copy.
    let schema: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/spec/manifest.schema.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let hints_schema = &schema["properties"]["metadata"]["properties"]["style_hints"]["properties"];
    let seconds = &hints_schema["suggested_playback_seconds"];
    assert_eq!(
        (seconds["minimum"].as_u64(), seconds["maximum"].as_u64()),
        (Some(5), Some(300)),
        "the schema must admit the refit band [5, 300] as well as the legacy [20, 90]"
    );
    let window = &hints_schema["suggested_time_window_ms"];
    assert_eq!(window["type"], "integer");
    assert_eq!(window["minimum"].as_u64(), Some(1));

    // ...and the emitted manifest sits inside them.
    let raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(archive.join("manifest.json")).unwrap()).unwrap();
    let sh = &raw["metadata"]["style_hints"];
    let emitted_seconds = sh["suggested_playback_seconds"].as_u64().unwrap();
    assert!((5..=300).contains(&emitted_seconds));
    let emitted_window = sh["suggested_time_window_ms"].as_u64().unwrap();
    assert!(emitted_window >= 1);
}

// ---------------------------------------------------------------------------
// Check 12 — the SEMANTIC CONTENT FINGERPRINT (SH-1).
//
// This block is the reason M7 exists. A `reoptimize` sweep once re-read 3D
// `xyz` coordinate leaves with a stride of 2 and flattened + scrambled 106 AV
// archives. Every one of them passed `stt-validate`: the packs hashed, the
// CRCs held, the Arrow schemas were perfect, the counts agreed. Nothing was
// malformed — the numbers were simply wrong. A human caught it by comparing
// `stt-optimize export` bboxes.
//
// `fingerprint_catches_scrambled_coordinates` is that regression class, encoded
// as a test. It is the acceptance case for the whole item: if the fingerprint
// cannot turn a scrambled archive red, the fingerprint is decoration.
// ---------------------------------------------------------------------------

/// The AV scene the scramble is reproduced over: real lon/lat near San
/// Francisco, altitude in metres, flattened into one interleaved `xyz` run
/// exactly as the format stores a 3D point leaf.
fn av_xyz_stream() -> Vec<f64> {
    vec![
        -122.4050, 37.7900, 0.0, //
        -122.4046, 37.7908, 6.25, //
        -122.4042, 37.7916, 12.5, //
        -122.4038, 37.7924, 18.75, //
        -122.4034, 37.7932, 25.0, //
        -122.4030, 37.7940, 30.0, //
    ]
}

/// The honest 2D reading of [`av_xyz_stream`]: stride 3, keep `(x, y)`.
fn honest_points() -> Vec<[f64; 2]> {
    av_xyz_stream()
        .chunks_exact(3)
        .map(|c| [c[0], c[1]])
        .collect()
}

/// THE DEFECT: the same buffer re-read with a stride of **2**. Altitudes land
/// in longitude and latitude slots, the trailing ordinate is dropped, and the
/// result is a structurally perfect archive of nonsense.
fn scrambled_points() -> Vec<[f64; 2]> {
    av_xyz_stream()
        .chunks_exact(2)
        .map(|c| [c[0], c[1]])
        .collect()
}

/// The fingerprint an honest writer declares for [`honest_points`] with the
/// `speed` column running `[0, 30]`.
fn honest_declaration(distinct: u64) -> stt_core::metadata::ContentFingerprint {
    stt_core::metadata::ContentFingerprint {
        version: stt_core::metadata::CONTENT_FINGERPRINT_VERSION,
        bbox: [-122.4050, 37.7900, -122.4030, 37.7940],
        z_range: None,
        distinct_feature_count: distinct,
        numeric_ranges: std::collections::BTreeMap::from([("speed".to_string(), [0.0, 30.0])]),
        categorical_cardinality: std::collections::BTreeMap::new(),
        coord_tolerance_deg: 0.0,
        column_tolerance: std::collections::BTreeMap::new(),
    }
}

/// Write one tile per entry of `tiles`, each a point layer with a `speed`
/// column, and declare `fingerprint` in the metadata.
///
/// Like `write_dataset_declaring`, the dishonesty (when there is any) lives in
/// what the caller ENCODES versus what it DECLARES — `PackWriter` takes the
/// metadata from its caller, the same trust seam a buggy upstream transform
/// would abuse. No bytes are surgically edited.
fn write_fingerprinted_dataset(
    out_dir: &Path,
    tiles: &[(Vec<[f64; 2]>, Vec<f64>)],
    fingerprint: Option<stt_core::metadata::ContentFingerprint>,
) {
    write_declared_dataset(out_dir, tiles, fingerprint, &Declaration::default());
}

/// What an archive DECLARES about its own spatial extent, independent of what
/// it encodes — check 13's whole surface.
///
/// Defaults reproduce `Metadata::new`: the whole-world placeholder bbox, no
/// vertical claim, and no vertex-bounds attestation.
#[derive(Default)]
struct Declaration {
    bounds: Option<stt_core::types::BoundingBox>,
    z_range: Option<[f64; 2]>,
    /// Stamp `bounds_mode = vertex` into `metadata.properties` — the
    /// attestation that turns check 13's under-statement finding from a
    /// legacy warning into an error without carrying a fingerprint.
    attest_vertex_bounds: bool,
    /// `manifest.capabilities` entries to declare. Needed to reach check 12's
    /// tolerance MAGNITUDE gate at all: a `coord_tolerance_deg` without
    /// `coord-quant` is rejected by the capability gate first, so a fixture
    /// probing the magnitude bound has to hold the capability.
    capabilities: Vec<String>,
}

fn write_declared_dataset(
    out_dir: &Path,
    tiles: &[(Vec<[f64; 2]>, Vec<f64>)],
    fingerprint: Option<stt_core::metadata::ContentFingerprint>,
    declaration: &Declaration,
) {
    let mut writer = PackWriter::create(out_dir, BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
    if !declaration.capabilities.is_empty() {
        writer = writer.with_capabilities(declaration.capabilities.clone());
    }
    let cfg = EncoderConfig {
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(writer.template_collector()),
        ..EncoderConfig::default()
    };
    let (t_start, t_end) = (1_000i64, 2_000i64);
    let mut features = 0u64;
    let mut next_id = 1u64;
    for (index, (points, speeds)) in tiles.iter().enumerate() {
        let n = points.len();
        let layer = ColumnarLayer {
            polygon_parts: None,
            name: "default".into(),
            feature_ids: (0..n as u64).map(|i| next_id + i).collect(),
            start_times: vec![t_start; n],
            end_times: vec![t_end; n],
            geometry: GeometryColumn::Point(points.clone()),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![(
                "speed".into(),
                PropertyColumn::Numeric(speeds.iter().map(|s| Some(*s)).collect()),
            )],
        };
        next_id += n as u64;
        let payload = encode_tile_with(&[layer], &cfg).unwrap();
        writer
            .add_tile_full(
                &TileId::new(8, index as u32, 0, t_start as u64),
                t_start,
                t_end,
                None,
                n as u32,
                None,
                &payload,
            )
            .unwrap();
        features += n as u64;
    }
    let mut meta = Metadata::new("fingerprint")
        .with_time_range(TimeRange::new(t_start as u64, t_end as u64))
        .with_zoom_levels(8, 8)
        .with_z_range(declaration.z_range);
    if let Some(bounds) = declaration.bounds {
        meta = meta.with_bounds(bounds);
    }
    if declaration.attest_vertex_bounds {
        meta.properties
            .insert("bounds_mode".to_string(), "vertex".to_string());
    }
    meta.feature_count = features;
    meta.tile_count = tiles.len() as u64;
    meta.distinct_feature_count = Some(features);
    meta.content_fingerprint = fingerprint;
    writer.finalize(&meta).unwrap();
}

/// Split a point run into `chunks` tiles, pairing each point with a `speed`
/// value spread over `[0, 30]`.
fn tiles_of(points: &[[f64; 2]], chunks: usize) -> Vec<(Vec<[f64; 2]>, Vec<f64>)> {
    let per = points.len().div_ceil(chunks.max(1));
    let step = if points.len() > 1 {
        30.0 / (points.len() - 1) as f64
    } else {
        0.0
    };
    points
        .chunks(per.max(1))
        .enumerate()
        .map(|(c, chunk)| {
            let speeds = (0..chunk.len())
                .map(|i| (c * per + i) as f64 * step)
                .collect();
            (chunk.to_vec(), speeds)
        })
        .collect()
}

/// The control. Without it every negative below would also pass if the check
/// simply rejected any archive carrying a fingerprint.
#[test]
fn honest_fingerprint_passes_full_and_sampled() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("honest");
    let tiles = tiles_of(&honest_points(), 3);
    write_fingerprinted_dataset(&archive, &tiles, Some(honest_declaration(6)));

    let (ok, report) = run_json(&archive, &[]);
    assert!(ok, "an honest fingerprint must validate clean: {report}");
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    assert_eq!(
        report["fingerprint_checked"], true,
        "the report must say the CONTENT was verified, not just the structure: {report}"
    );

    // The same archive under a sampled decode: containment only, still clean.
    let (ok, report) = run_json(&archive, &["--sample", "2"]);
    assert!(ok, "a sampled honest run must stay clean: {report}");
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    assert_eq!(report["fingerprint_checked"], true);
    assert_eq!(report["sampled"], true);
}

/// ⭐ THE ACCEPTANCE CASE. The 2-strided 3D `xyz` fold — 106 archives, all
/// green under structural validation — must now exit NON-ZERO, naming the bbox
/// violation.
#[test]
fn fingerprint_catches_scrambled_coordinates() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("scrambled");
    // Identical declaration to the honest control above; only the ENCODED
    // coordinates differ, which is exactly the shape of the recorded defect.
    let scrambled = scrambled_points();
    let tiles = tiles_of(&scrambled, 3);
    write_fingerprinted_dataset(&archive, &tiles, Some(honest_declaration(6)));

    // Structural validation alone still sees nothing wrong — that is the whole
    // problem, and it is asserted here so the test cannot silently start
    // passing for an unrelated reason.
    let (ok, structural) = run_json(&archive, &["--skip-decode"]);
    assert!(
        ok,
        "the scrambled archive is STRUCTURALLY perfect — if this fails, the \
         fixture is broken rather than the check being good: {structural}"
    );

    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "content fingerprint");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("escapes the declared bbox")),
        "the error must name the bbox violation; errors: {errors:#?}"
    );
    assert_eq!(report["fingerprint_checked"], true);
}

/// The same defect must still be caught when only a fraction of the tiles are
/// decoded — the containment semantics are what make `--sample` a real check.
///
/// Four tiles, `--sample 2` → stride `ceil(4/2) = 2` → tiles 0 and 2 decode.
/// The scrambled points live ONLY in tile 2, so a pass here would mean the
/// sampled path never reached them.
#[test]
fn sampled_fingerprint_still_catches_out_of_bbox() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("sampled-scramble");

    let honest = honest_points();
    let good = |i: usize| (vec![honest[i]], vec![i as f64]);
    let tiles = vec![
        good(0),
        good(1),
        // Tile 2 — inside the sampled stride — carries a scrambled coordinate
        // whose longitude is really an altitude read out of the wrong slot.
        (vec![[12.5, 37.7916]], vec![2.0]),
        good(3),
    ];
    write_fingerprinted_dataset(&archive, &tiles, Some(honest_declaration(4)));

    let (ok, report) = run_json(&archive, &["--sample", "2"]);
    assert_eq!(report["tiles_decoded"], 2, "report: {report}");
    let errors = assert_failed_with(ok, &report, "escapes the declared bbox");
    assert!(
        errors.iter().any(|e| e.contains("content fingerprint")),
        "errors: {errors:#?}"
    );
}

/// A property column whose values were rescaled (unit confusion, a botched
/// re-encode) decodes outside its declared range.
#[test]
fn fingerprint_catches_scaled_property_column() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("scaled-column");

    // Honest geometry, `speed` multiplied by 10.
    let mut tiles = tiles_of(&honest_points(), 3);
    for (_, speeds) in &mut tiles {
        for s in speeds.iter_mut() {
            *s *= 10.0;
        }
    }
    write_fingerprinted_dataset(&archive, &tiles, Some(honest_declaration(6)));

    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "outside its declared range");
    assert!(
        errors.iter().any(|e| e.contains("column 'speed'")),
        "the error must name the offending column; errors: {errors:#?}"
    );
    // The geometry is honest, so the bbox must NOT be implicated — the check
    // has to discriminate rather than blanket-fail.
    assert!(
        !errors
            .iter()
            .any(|e| e.contains("escapes the declared bbox")),
        "errors: {errors:#?}"
    );
}

// ---------------------------------------------------------------------------
// ⭐ BLOCKER 4 — the capability gate was PRESENCE-ONLY, so the tolerance made
// every bbox comparison vacuous.
//
// `coord_tolerance_deg` is a request not to be checked. It was gated on the
// archive declaring `coord-quant` and on nothing else, so an archive quantized
// at 1 m could declare `coord_tolerance_deg: 90.0`, exit 0 with zero errors,
// and have the report state that its CONTENT was verified — while the
// comparison that catches the recorded stride-2 scramble had been widened past
// the point of meaning anything. A wrong fingerprint is worse than no
// fingerprint, because it certifies.
//
// The bound is the archive's OWN on-wire `stt:quant` step, which
// `check_fingerprint` already admits as slack regardless of the declaration —
// so bounding by it costs an honest archive exactly nothing.
// ---------------------------------------------------------------------------

/// A tolerance wide enough to swallow the scrambled fixture's displacement.
/// The stride-2 fold puts a latitude of `-122.4` where a longitude belongs, so
/// laundering it needs ~160°; 200 clears that with room to spare.
const LAUNDERING_TOLERANCE_DEG: f64 = 200.0;

#[test]
fn inflated_coord_tolerance_is_rejected_instead_of_silencing_the_bbox_check() {
    let dir = tempfile::tempdir().unwrap();
    let coord_quant = vec![stt_core::pack::CAPABILITY_COORD_QUANT.to_string()];

    // (a) The reviewer's reproduction: honest content, an inflated tolerance,
    //     and the capability that used to be the ONLY gate. This exited 0 with
    //     zero errors.
    let inflated = dir.path().join("inflated");
    let mut declaration = honest_declaration(6);
    declaration.coord_tolerance_deg = 90.0;
    write_declared_dataset(
        &inflated,
        &tiles_of(&honest_points(), 3),
        Some(declaration),
        &Declaration {
            capabilities: coord_quant.clone(),
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&inflated, &[]);
    let errors = assert_failed_with(ok, &report, "on-wire 'stt:quant' step");
    assert!(
        errors.iter().any(|e| e.contains("content fingerprint")),
        "the finding belongs to check 12: {errors:#?}"
    );

    // (b) …and the reason it matters. Same inflation, this time over the
    //     ACTUAL recorded defect: the stride-2 xyz fold. With the tolerance
    //     admitted, the bbox comparison produces nothing at all — so if the
    //     magnitude bound did not fail this run, a scrambled fleet would ship
    //     with a green validator and a "content verified" report.
    let laundered = dir.path().join("laundered-scramble");
    let mut wide = honest_declaration(6);
    wide.coord_tolerance_deg = LAUNDERING_TOLERANCE_DEG;
    write_declared_dataset(
        &laundered,
        &tiles_of(&scrambled_points(), 3),
        Some(wide),
        &Declaration {
            capabilities: coord_quant.clone(),
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&laundered, &[]);
    let errors = assert_failed_with(ok, &report, "on-wire 'stt:quant' step");
    assert!(
        !errors
            .iter()
            .any(|e| e.contains("escapes the declared bbox")),
        "fixture check: a {LAUNDERING_TOLERANCE_DEG} deg tolerance is supposed to \
         swallow this scramble's bbox violation — if it no longer does, this test \
         has stopped demonstrating the vacuity it was written for: {errors:#?}"
    );

    // (c) The legitimate case stays legitimate: NO declared tolerance at all on
    //     the same honest archive validates clean. The rejection above must be
    //     about the inflation, not about the archive.
    let plain = dir.path().join("no-tolerance");
    write_declared_dataset(
        &plain,
        &tiles_of(&honest_points(), 3),
        Some(honest_declaration(6)),
        &Declaration {
            capabilities: coord_quant,
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&plain, &[]);
    assert!(ok, "an archive declaring no slack must pass: {report}");
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
}

/// The inflation must not survive `--emit-fingerprint` either. Rejecting it in
/// the manifest is only half the fix: the capture used to copy the DECLARED
/// tolerance into the baseline file, so the vacuity escaped into a document
/// that every later `--expect-fingerprint` run would trust as measured truth.
#[test]
fn emitted_baseline_reports_the_wire_step_not_the_inflated_declaration() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("inflated-source");
    let truth = dir.path().join("truth.json");

    let mut declaration = honest_declaration(6);
    declaration.coord_tolerance_deg = LAUNDERING_TOLERANCE_DEG;
    write_declared_dataset(
        &archive,
        &tiles_of(&honest_points(), 3),
        Some(declaration),
        &Declaration {
            capabilities: vec![stt_core::pack::CAPABILITY_COORD_QUANT.to_string()],
            ..Declaration::default()
        },
    );

    // The run itself fails (the declaration is rejected), but the capture is
    // still written — and what it captured is what matters here.
    let (_ok, _report) = run_json(&archive, &["--emit-fingerprint", truth.to_str().unwrap()]);
    let captured: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&truth).unwrap()).unwrap();
    assert_eq!(
        captured["coord_tolerance_deg"].as_f64(),
        Some(0.0),
        "the capture must report the archive's own on-wire step (this fixture is \
         unquantized, so 0.0), never the {LAUNDERING_TOLERANCE_DEG} deg claim it \
         was handed: {captured}"
    );
}

// ---------------------------------------------------------------------------
// ⭐ BLOCKER 1 — a rebuild that silently DROPS features.
//
// R1 rebuilds the entire published fleet. If a builder change loses features,
// EVERY structural check still passes: each tile's decoded row count matches its
// own directory entry, the metadata totals match the directory the same build
// wrote, the CRCs are of the bytes that exist. The count the archive cannot
// forge is the one the fingerprint carries from the SOURCE, and until now
// disagreeing with it was a warning — indistinguishable from the benign warning
// an honest archive emits under `--sample`.
//
// The fixture keeps the FIRST and LAST feature, so the lossy archive decodes a
// bit-identical bbox and a bit-identical `speed` range: nothing but the distinct
// count can catch it, which is exactly the point.
// ---------------------------------------------------------------------------

/// `n` features marching along one segment, each carrying its index as `speed`.
///
/// Every ordinate is generated by the same expression, so a SUBSET that keeps
/// the first and last entry decodes extremes that are bit-identical to the full
/// set's — no float slop can leak into the bbox/range comparisons and
/// masquerade as (or mask) the distinct-count finding.
fn marching_features(n: usize) -> Vec<([f64; 2], f64)> {
    (0..n)
        .map(|i| {
            (
                [-122.4050 + i as f64 * 0.00002, 37.7900 + i as f64 * 0.00004],
                i as f64,
            )
        })
        .collect()
}

/// The fingerprint an honest writer declares for exactly `features`.
fn declaration_over(
    features: &[([f64; 2], f64)],
    distinct: u64,
) -> stt_core::metadata::ContentFingerprint {
    let fold = |pick: fn(&([f64; 2], f64)) -> f64, hi: bool| {
        features.iter().map(pick).fold(
            if hi { f64::MIN } else { f64::MAX },
            if hi { f64::max } else { f64::min },
        )
    };
    stt_core::metadata::ContentFingerprint {
        version: stt_core::metadata::CONTENT_FINGERPRINT_VERSION,
        bbox: [
            fold(|(p, _)| p[0], false),
            fold(|(p, _)| p[1], false),
            fold(|(p, _)| p[0], true),
            fold(|(p, _)| p[1], true),
        ],
        z_range: None,
        distinct_feature_count: distinct,
        numeric_ranges: std::collections::BTreeMap::from([(
            "speed".to_string(),
            [fold(|(_, s)| *s, false), fold(|(_, s)| *s, true)],
        )]),
        categorical_cardinality: std::collections::BTreeMap::new(),
        coord_tolerance_deg: 0.0,
        column_tolerance: std::collections::BTreeMap::new(),
    }
}

/// Split marching features into `chunks` tiles.
fn tiles_from(features: &[([f64; 2], f64)], chunks: usize) -> Vec<(Vec<[f64; 2]>, Vec<f64>)> {
    let per = features.len().div_ceil(chunks.max(1)).max(1);
    features
        .chunks(per)
        .map(|c| {
            (
                c.iter().map(|(p, _)| *p).collect(),
                c.iter().map(|(_, s)| *s).collect(),
            )
        })
        .collect()
}

/// ⭐ THE BLOCKER-1 ACCEPTANCE CASE. An archive that holds 26 of the 100
/// features its fingerprint declares — a **74 % loss** — must EXIT NON-ZERO
/// naming the loss, and the SAME archive under `--sample` must not.
///
/// Both halves are the requirement. Erroring on a sampled decode would make the
/// gate a false-positive generator (a subset of tiles carries a subset of the
/// ids by construction), and that is precisely why the finding was a warning
/// before: one warning served both cases, so a real loss was unreadable.
#[test]
fn dropped_features_fail_a_full_decode_and_only_warn_under_sampling() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("lossy-rebuild");

    let full = marching_features(100);
    // The rebuild kept 26 of 100 — every 4th feature, plus the last — so the
    // extremes (and therefore the bbox and the `speed` range) are untouched and
    // 74 features are simply gone.
    let kept: Vec<([f64; 2], f64)> = full
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 == 0 || *i == full.len() - 1)
        .map(|(_, f)| *f)
        .collect();
    assert_eq!(kept.len(), 26, "the fixture must drop exactly 74 of 100");

    // The manifest describes the SOURCE (100 features); the archive holds the
    // rebuild's output. Nothing is byte-edited — this is the same writer seam a
    // real lossy rebuild would come through.
    let declared = declaration_over(&full, full.len() as u64);
    write_fingerprinted_dataset(&archive, &tiles_from(&kept, 3), Some(declared.clone()));

    // Control: the loss is invisible to every structural check.
    let (ok, structural) = run_json(&archive, &["--skip-decode"]);
    assert!(
        ok,
        "a 74% feature loss is STRUCTURALLY perfect — if this fails, the fixture \
         is broken rather than the check being good: {structural}"
    );

    // Full decode: ERROR, naming the loss and its size.
    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "FEATURE LOSS");
    // The exit CODE is the contract a CI gate reads, so pin it on both sides
    // rather than trusting `success()`.
    let exit = |extra: &[&str]| -> Option<i32> {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_stt-validate"));
        cmd.arg(&archive).arg("--json");
        for a in extra {
            cmd.arg(a);
        }
        cmd.output().expect("run stt-validate").status.code()
    };
    assert_eq!(exit(&[]), Some(1), "a full decode must exit 1");
    assert_eq!(
        exit(&["--sample", "2"]),
        Some(0),
        "a sampled decode must exit 0"
    );
    assert!(
        errors
            .iter()
            .any(|e| e.contains("74.0%") && e.contains("MISSING")),
        "the error must name the loss FRACTION, not just disagree; errors: {errors:#?}"
    );
    // The bbox and the column range are honest, so nothing else may be
    // implicated: the distinct count is what caught this.
    assert!(
        !errors.iter().any(|e| e.contains("bbox")
            || e.contains("outside its declared range")
            || e.contains("WIDER")),
        "the loss must be caught by the distinct count alone; errors: {errors:#?}"
    );

    // The SAME archive under --sample: no error, and a warning that cannot be
    // read as the one above.
    let (ok, sampled) = run_json(&archive, &["--sample", "2"]);
    assert!(
        ok,
        "a sampled decode legitimately sees fewer ids — erroring here would make \
         the gate a false-positive generator: {sampled}"
    );
    assert_eq!(errors_of(&sampled).len(), 0, "report: {sampled}");
    let warnings: Vec<String> = sampled["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap().to_string())
        .collect();
    let noise: Vec<&String> = warnings
        .iter()
        .filter(|w| w.contains("SAMPLING NOISE"))
        .collect();
    assert_eq!(noise.len(), 1, "warnings: {warnings:#?}");
    assert!(
        !noise[0].contains("FEATURE LOSS"),
        "sampling noise and feature loss must never read the same: {}",
        noise[0]
    );

    // The escape hatch is explicit, scoped, and still reports the finding.
    let (ok, allowed) = run_json(&archive, &["--allow-distinct-shortfall"]);
    assert!(ok, "the kill switch must clear the run: {allowed}");
    assert_eq!(errors_of(&allowed).len(), 0, "report: {allowed}");
    let downgraded: Vec<String> = allowed["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap().to_string())
        .filter(|w| w.contains("DOWNGRADED by --allow-distinct-shortfall"))
        .collect();
    assert_eq!(downgraded.len(), 1, "report: {allowed}");
    assert!(downgraded[0].contains("FEATURE LOSS"), "{}", downgraded[0]);

    // The control that keeps the negative honest: the SAME writer, the SAME
    // declaration, with nothing dropped, validates clean under a full decode.
    let intact = dir.path().join("intact");
    write_fingerprinted_dataset(&intact, &tiles_from(&full, 3), Some(declared));
    let (ok, report) = run_json(&intact, &[]);
    assert!(ok, "the intact rebuild must pass: {report}");
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    assert_eq!(report["fingerprint_checked"], true);
}

/// An archive with NO fingerprint — every archive published before SH-1 — must
/// WARN and pass. Turning the fleet red would teach operators to ignore the
/// validator, which is how the CRS84 case is handled too.
#[test]
fn absent_fingerprint_warns_but_never_fails() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("legacy");
    write_fingerprinted_dataset(&archive, &tiles_of(&honest_points(), 2), None);

    let (ok, report) = run_json(&archive, &[]);
    assert!(ok, "a pre-fingerprint archive must still pass: {report}");
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    assert_eq!(
        report["fingerprint_checked"], false,
        "the report must NOT claim the content was verified: {report}"
    );
    let warnings: Vec<String> = report["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap().to_string())
        .collect();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("content_fingerprint is absent")),
        "the gap must stay visible; warnings: {warnings:#?}"
    );
}

/// Transform acceptance, both directions.
///
/// The relapse-proof rule in executable form: a lossless transform is accepted
/// against a fingerprint captured from the archive as it was BEFORE the
/// transform, never one the transform recomputed from its own output. An
/// identical archive is accepted; a scrambled one is rejected — and the
/// capture itself refuses to run off a sampled decode, because an understated
/// expectation is an expectation that cannot fail.
#[test]
fn expect_fingerprint_accepts_identical_and_rejects_mutated() {
    let dir = tempfile::tempdir().unwrap();
    let before = dir.path().join("before");
    let after_ok = dir.path().join("after-ok");
    let after_bad = dir.path().join("after-bad");
    let truth = dir.path().join("truth.json");

    let honest = tiles_of(&honest_points(), 3);
    write_fingerprinted_dataset(&before, &honest, None);
    // A lossless transform: same features, re-cut into a different number of
    // tiles. Pack boundaries and blob layout move; content does not.
    write_fingerprinted_dataset(&after_ok, &tiles_of(&honest_points(), 2), None);
    write_fingerprinted_dataset(&after_bad, &tiles_of(&scrambled_points(), 2), None);

    // Capture from the trusted source.
    let (ok, report) = run_json(&before, &["--emit-fingerprint", truth.to_str().unwrap()]);
    assert!(ok, "the capture run must succeed: {report}");
    assert!(truth.is_file(), "--emit-fingerprint must write the file");

    // The transformed-but-equal archive is accepted...
    let (ok, report) = run_json(
        &after_ok,
        &["--expect-fingerprint", truth.to_str().unwrap()],
    );
    assert!(
        ok,
        "a re-tiled but content-identical archive must be accepted — the \
         fingerprint is invariant to pack boundaries and blob ordering: {report}"
    );
    assert_eq!(report["fingerprint_checked"], true);

    // ...and the scrambled one is not.
    let (ok, report) = run_json(
        &after_bad,
        &["--expect-fingerprint", truth.to_str().unwrap()],
    );
    let errors = assert_failed_with(ok, &report, "expected fingerprint");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("escapes the declared bbox")),
        "errors: {errors:#?}"
    );

    // A capture off a sampled decode would understate the content, so it is
    // refused outright rather than written and silently trusted later.
    let bin = env!("CARGO_BIN_EXE_stt-validate");
    let out = Command::new(bin)
        .arg(&before)
        .arg("--sample")
        .arg("1")
        .arg("--emit-fingerprint")
        .arg(dir.path().join("partial.json"))
        .output()
        .expect("run stt-validate");
    assert!(!out.status.success(), "a sampled capture must be refused");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("FULL decode"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !dir.path().join("partial.json").exists(),
        "a refused capture must not leave a file behind"
    );
}

// ---------------------------------------------------------------------------
// Check 13 — DECLARED BOUNDS CONTAINMENT (SH-2).
//
// The gap R1 opens for itself. R1's headline byte change flips
// `metadata.bounds` from a centroid bbox to a vertex-derived one, and until
// this block existed NOTHING compared that number against the archive's own
// coordinates: an archive whose declared bounds sat in the WRONG HEMISPHERE
// validated clean, because every check above it is either structural or keyed
// on `content_fingerprint.bbox` — a different declaration.
//
// The direction is the point and it is asymmetric. Too SMALL is silent data
// loss at query time (tile selection, frustum pre-culling and the opening
// camera all pre-intersect against `metadata.bounds`, so they discard tiles
// that hold visible data); too LARGE is conservative and warns at most.
// ---------------------------------------------------------------------------

/// The binary's raw process exit code — what CI actually gates on. `run_json`
/// reduces it to a bool, which is enough for most assertions but hides whether
/// a failure was a validation failure (1) or a crash.
fn exit_code(archive: &Path, extra: &[&str]) -> Option<i32> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_stt-validate"));
    cmd.arg(archive).arg("--json");
    for a in extra {
        cmd.arg(a);
    }
    cmd.output()
        .expect("failed to run stt-validate")
        .status
        .code()
}

/// The exact vertex bbox of [`honest_points`] — what an honest vertex-bounds
/// writer declares for this archive.
fn honest_bounds() -> stt_core::types::BoundingBox {
    stt_core::types::BoundingBox::new(-122.4050, 37.7900, -122.4030, 37.7940)
}

/// The reviewer's archive: same data, bbox declared in the eastern hemisphere.
fn wrong_hemisphere_bounds() -> stt_core::types::BoundingBox {
    stt_core::types::BoundingBox::new(100.0, 37.7900, 110.0, 37.7940)
}

/// ⭐ THE ACCEPTANCE CASE FOR CHECK 13. Declared bounds in the wrong
/// hemisphere, on an archive that ATTESTS vertex-derived bounds, must exit
/// NON-ZERO — and it must be check 13 that says so, not check 12: the
/// fingerprint declared here is perfectly honest, so if the fingerprint block
/// were doing the work this test would prove nothing about `metadata.bounds`.
#[test]
fn wrong_hemisphere_declared_bounds_fails() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("wrong-hemisphere");
    let tiles = tiles_of(&honest_points(), 3);
    write_declared_dataset(
        &archive,
        &tiles,
        Some(honest_declaration(6)),
        &Declaration {
            bounds: Some(wrong_hemisphere_bounds()),
            ..Declaration::default()
        },
    );

    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "declared bounds");
    assert!(
        errors.iter().any(|e| e.contains("does NOT contain")),
        "the error must name the containment violation: {errors:#?}"
    );
    // `content fingerprint: ` is check 12's error prefix; check 13's own text
    // names the block too (it is the attestation), so match on the prefix.
    assert!(
        errors
            .iter()
            .all(|e| !e.starts_with("content fingerprint:")),
        "check 12 must be clean here — otherwise this proves nothing about \
         metadata.bounds: {errors:#?}"
    );
    assert_eq!(report["bounds_checked"], true, "report: {report}");
    assert_eq!(report["bounds_enforced"], true, "report: {report}");
    // The literal process exit code, not just "not success" — this is the
    // number CI gates on.
    assert_eq!(
        exit_code(&archive, &[]),
        Some(1),
        "the R1 gate reads the exit code"
    );

    // Containment holds on any subset, so a sampled decode catches it too —
    // the same property that makes check 12 sound under `--sample`.
    let (ok, report) = run_json(&archive, &["--sample", "2"]);
    assert_failed_with(ok, &report, "declared bounds");
    assert_eq!(exit_code(&archive, &["--sample", "2"]), Some(1));
}

/// The control. Without it the case above would also pass if check 13 simply
/// rejected every archive that declares a bbox.
#[test]
fn honest_declared_bounds_pass_full_and_sampled() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("honest-bounds");
    let tiles = tiles_of(&honest_points(), 3);
    write_declared_dataset(
        &archive,
        &tiles,
        Some(honest_declaration(6)),
        &Declaration {
            bounds: Some(honest_bounds()),
            ..Declaration::default()
        },
    );

    let (ok, report) = run_json(&archive, &[]);
    assert!(ok, "an honest vertex bbox must validate clean: {report}");
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    assert_eq!(report["bounds_checked"], true, "report: {report}");
    assert_eq!(exit_code(&archive, &[]), Some(0), "report: {report}");
    // No check-13 warning either: an exactly-tight box is neither under- nor
    // over-stated.
    let warnings: Vec<String> = report["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap().to_string())
        .collect();
    assert!(
        warnings.iter().all(|w| !w.contains("declared bounds")),
        "an exact bbox must produce no bounds finding at all: {warnings:#?}"
    );

    let (ok, report) = run_json(&archive, &["--sample", "2"]);
    assert!(ok, "a sampled honest run must stay clean: {report}");
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
}

/// The attestation gate, both halves. The SAME understated bbox is a WARNING
/// on a legacy archive — every published archive carries the centroid bbox,
/// which under-states non-point geometry by construction, and turning the
/// whole fleet red is how operators learn to ignore a validator — and an ERROR
/// as soon as the archive attests vertex-derived bounds.
#[test]
fn understated_bounds_warn_on_legacy_and_fail_once_attested() {
    let dir = tempfile::tempdir().unwrap();
    let tiles = tiles_of(&honest_points(), 3);

    // Legacy: no fingerprint, no attestation → warning only, exit 0.
    let legacy = dir.path().join("legacy");
    write_declared_dataset(
        &legacy,
        &tiles,
        None,
        &Declaration {
            bounds: Some(wrong_hemisphere_bounds()),
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&legacy, &[]);
    assert!(
        ok,
        "a pre-R1 archive must not be turned red by check 13: {report}"
    );
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    assert_eq!(report["bounds_enforced"], false, "report: {report}");
    let warnings: Vec<String> = report["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap().to_string())
        .collect();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("declared bounds") && w.contains("BoundsMode::Vertex")),
        "the legacy warning must name the rebuild as the fix, exactly like the \
         absent-fingerprint path: {warnings:#?}"
    );

    // Attested by the `bounds_mode = vertex` property alone — no fingerprint —
    // so the gate is not merely "does check 12 run".
    let attested = dir.path().join("attested");
    write_declared_dataset(
        &attested,
        &tiles,
        None,
        &Declaration {
            bounds: Some(wrong_hemisphere_bounds()),
            attest_vertex_bounds: true,
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&attested, &[]);
    assert_failed_with(ok, &report, "declared bounds");
    assert_eq!(report["bounds_enforced"], true, "report: {report}");
}

/// ⚠️ Antimeridian is a recorded hazard in this repo, and a spurious failure
/// on a ±180°-straddling dataset would be worse than no check at all. Two
/// shapes, neither of which may fail:
///
/// * the loose full-width longitude interval `profile_features_with` actually
///   emits for a straddling dataset — it contains every vertex, so it passes;
/// * a WRAPPED interval (`min_lon > max_lon`) from some other writer — not
///   decidable against an unwrapped decoded bbox, so the longitude axis is
///   skipped with a warning rather than guessed at.
#[test]
fn antimeridian_datasets_do_not_false_positive() {
    let dir = tempfile::tempdir().unwrap();

    // Four vertices straddling the date line; the min/max fold a writer
    // performs gives [-179.95, 9.5, 179.95, 11.0].
    let points = vec![
        [179.90, 10.00],
        [-179.90, 10.50],
        [179.95, 9.50],
        [-179.95, 11.00],
    ];
    let tiles = vec![(points, vec![0.0, 10.0, 20.0, 30.0])];
    let fingerprint = stt_core::metadata::ContentFingerprint {
        version: stt_core::metadata::CONTENT_FINGERPRINT_VERSION,
        bbox: [-179.95, 9.50, 179.95, 11.00],
        z_range: None,
        distinct_feature_count: 4,
        numeric_ranges: std::collections::BTreeMap::from([("speed".to_string(), [0.0, 30.0])]),
        categorical_cardinality: std::collections::BTreeMap::new(),
        coord_tolerance_deg: 0.0,
        column_tolerance: std::collections::BTreeMap::new(),
    };

    let loose = dir.path().join("straddling");
    write_declared_dataset(
        &loose,
        &tiles,
        Some(fingerprint.clone()),
        &Declaration {
            bounds: Some(stt_core::types::BoundingBox::new(
                -179.95, 9.50, 179.95, 11.00,
            )),
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&loose, &[]);
    assert!(
        ok,
        "a ±180°-straddling dataset must not fail check 13: {report}"
    );
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");

    // A wrapped declaration: warn, never fail.
    let wrapped = dir.path().join("wrapped");
    write_declared_dataset(
        &wrapped,
        &tiles,
        Some(fingerprint),
        &Declaration {
            bounds: Some(stt_core::types::BoundingBox::new(
                170.0, 9.50, -170.0, 11.00,
            )),
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&wrapped, &[]);
    assert!(
        ok,
        "a wrapped longitude interval must not manufacture a failure: {report}"
    );
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    let warnings: Vec<String> = report["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap().to_string())
        .collect();
    assert!(
        warnings.iter().any(|w| w.contains("WRAPPED")),
        "the skipped axis must be stated, not silent: {warnings:#?}"
    );
}

// ---------------------------------------------------------------------------
// Check 12 × the antimeridian seam.
//
// The seam relaxation originally landed in check 13 (`metadata.bounds`) only,
// and check 12 (`ContentFingerprint::bbox`) kept an unconditional containment
// test. But BOTH declared boxes are min/max folds over the SOURCE vertices
// taken before tiling (`input::content_fingerprint` and
// `input::profile_features_with`), and the ring splitter synthesises vertices
// at exactly ±180 that no source vertex carried — so both under-state a
// seam-crossing dataset in exactly the same way.
//
// Measured on 40 honest quads spanning 178E→178W built `--content-fingerprint`:
// exit 1 from check 12 accusing the archive of "the recorded stride-2 xyz fold
// that flattened 106 archives", sitting directly beside check 13's calm warning
// explaining that this is the ring splitter and not corruption. A red build and
// a warning that contradicts it, on Pacific storm polygons and global coverage
// polygons this repo actually ships.
//
// The pair below is the fix and its guard rail. `the_check_12_seam_exemption_
// cannot_hide_a_scramble` is the one that matters: the exemption must not
// become a hiding place, so the recorded regression class has to stay caught
// *on a seam-crossing fixture*, at full decode and under `--sample`.
// ---------------------------------------------------------------------------

/// The fingerprint an honest seam-crossing polygon build declares: the plain
/// unwrapped min/max fold over source vertices that sit at 178°E and 178°W.
fn seam_crossing_declaration(distinct: u64) -> stt_core::metadata::ContentFingerprint {
    stt_core::metadata::ContentFingerprint {
        version: stt_core::metadata::CONTENT_FINGERPRINT_VERSION,
        bbox: [-178.0, -20.0, 178.0, 19.8],
        z_range: None,
        distinct_feature_count: distinct,
        numeric_ranges: std::collections::BTreeMap::from([("speed".to_string(), [0.0, 30.0])]),
        categorical_cardinality: std::collections::BTreeMap::new(),
        coord_tolerance_deg: 0.0,
        column_tolerance: std::collections::BTreeMap::new(),
    }
}

/// The `warnings` array of a report, as owned strings.
fn report_warnings(report: &serde_json::Value) -> Vec<String> {
    report["warnings"]
        .as_array()
        .expect("report must carry a warnings array")
        .iter()
        .map(|w| w.as_str().expect("warnings are strings").to_string())
        .collect()
}

/// ⭐ An honest dateline-crossing archive must not be accused of corruption by
/// check 12 — and must not go silent either.
///
/// The decode carries the splitter's synthesised ±180 seam vertices; both
/// declared boxes stop at the extreme SOURCE vertices (±178). Before the fix
/// this exited 1 with the stride-2-scramble message. Now both checks agree:
/// two warnings, both naming the seam, exit 0.
#[test]
fn check_12_does_not_accuse_an_honest_seam_crossing_archive() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("dateline-honest");

    // What a split ring decodes to: the synthesised seam vertices at exactly
    // ±180, plus interior vertices on both sides of the seam. Latitudes stay
    // inside the declared [-20, 19.8] — a meridian cannot move latitude.
    let tiles = vec![
        (
            vec![[-180.0, -20.0], [-179.0, -10.0], [-178.0, 0.0]],
            vec![0.0, 5.0, 10.0],
        ),
        (
            vec![[178.0, 5.0], [179.0, 12.0], [180.0, 19.8]],
            vec![15.0, 22.0, 30.0],
        ),
    ];
    write_declared_dataset(
        &archive,
        &tiles,
        Some(seam_crossing_declaration(6)),
        &Declaration {
            // The same source-vertex fold check 13 reads, so BOTH checks face
            // the identical escape — which is the whole point.
            bounds: Some(stt_core::types::BoundingBox::new(
                -178.0, -20.0, 178.0, 19.8,
            )),
            ..Declaration::default()
        },
    );

    let (ok, report) = run_json(&archive, &[]);
    assert!(
        ok,
        "an honest seam-crossing archive must not fail check 12: {report}"
    );
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    assert_eq!(
        report["fingerprint_checked"], true,
        "the content still has to be CHECKED — the seam is an exemption on one \
         comparison, not a skip of the whole fingerprint: {report}"
    );

    let warnings = report_warnings(&report);
    // Check 12's warning: it names the declared BBOX (the fingerprint's), the
    // seam, and the writer-side fix.
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("escapes the declared bbox")
                && w.contains("ANTIMERIDIAN SEAM")
                && w.contains("widening the declared longitude interval")),
        "check 12 must EXPLAIN the seam rather than either failing or going \
         silent: {warnings:#?}"
    );
    // Check 13's warning, unchanged, still beside it — and no longer
    // contradicted by an error.
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("metadata.bounds") && w.contains("ANTIMERIDIAN SEAM")),
        "check 13's seam warning must survive: {warnings:#?}"
    );

    // Containment is what makes sampling sound, so the sampled run must reach
    // the same verdict rather than trading one false positive for another.
    let (ok, report) = run_json(&archive, &["--sample", "1"]);
    assert!(ok, "a sampled run must agree: {report}");
    assert_eq!(errors_of(&report).len(), 0, "report: {report}");
}

/// ⭐⭐ THE ADVERSARIAL CASE. The seam exemption must not become a hiding place:
/// a SEAM-CROSSING declaration (declared width 356° — condition 3 satisfied,
/// which is the only condition an attacker gets for free) must still turn red
/// on the recorded stride-2 `xyz` fold, at full decode AND under `--sample`.
///
/// Four probes, one per way the exemption could have been made unsound:
///
/// 1. the real scramble on a seam-crossing archive → latitude escapes → ERROR;
/// 2. an escape that stops SHORT of the seam (±179) → ERROR;
/// 3. an escape PAST the seam (±181) → ERROR;
/// 4. a COMPACT declared box smeared to the world edges → ERROR (condition 3).
#[test]
fn the_check_12_seam_exemption_cannot_hide_a_scramble() {
    let dir = tempfile::tempdir().unwrap();

    // (1) The 2-strided read of a seam-crossing 3D `xyz` run. Reading
    // [lon, lat, alt, …] two-at-a-time puts source LONGITUDES (±178) into the
    // LATITUDE slot, so latitude escapes [-20, 19.8] by ~158° — and latitude is
    // never excused, because a meridian cannot move it.
    let seam_xyz: Vec<f64> = vec![
        178.0, 10.0, 0.0, //
        -178.0, 12.0, 6.25, //
        179.5, -15.0, 12.5, //
        -179.5, 19.8, 18.75, //
        177.0, -20.0, 25.0, //
        -177.0, 3.0, 30.0, //
    ];
    let scrambled: Vec<[f64; 2]> = seam_xyz.chunks_exact(2).map(|c| [c[0], c[1]]).collect();

    let scramble_dir = dir.path().join("dateline-scrambled");
    write_declared_dataset(
        &scramble_dir,
        &tiles_of(&scrambled, 3),
        Some(seam_crossing_declaration(9)),
        &Declaration {
            bounds: Some(stt_core::types::BoundingBox::new(
                -178.0, -20.0, 178.0, 19.8,
            )),
            ..Declaration::default()
        },
    );

    // Structurally perfect — that is the whole shape of the recorded defect.
    let (ok, structural) = run_json(&scramble_dir, &["--skip-decode"]);
    assert!(
        ok,
        "the fixture must be structurally clean, or this proves nothing: {structural}"
    );

    let (ok, report) = run_json(&scramble_dir, &[]);
    let errors = assert_failed_with(ok, &report, "content fingerprint");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("escapes the declared bbox")),
        "a scramble on a SEAM-CROSSING archive must still name the bbox \
         violation as an ERROR: {errors:#?}"
    );
    assert!(
        !report_warnings(&report)
            .iter()
            .any(|w| w.contains("ANTIMERIDIAN SEAM") && w.contains("escapes the declared bbox")),
        "check 12 must not ALSO excuse what it just failed: {report}"
    );

    // …and under `--sample`, which is what makes the check affordable on the
    // 44 M-point archives.
    let (ok, sampled) = run_json(&scramble_dir, &["--sample", "1"]);
    assert!(sampled["sampled"] == true, "report: {sampled}");
    assert_failed_with(ok, &sampled, "escapes the declared bbox");

    // (2) An escape that stops SHORT of ±180 is not the splitter's signature.
    let short_dir = dir.path().join("short-of-seam");
    write_declared_dataset(
        &short_dir,
        &[(vec![[-179.0, 0.0], [179.0, 1.0]], vec![0.0, 30.0])],
        Some(seam_crossing_declaration(2)),
        &Declaration {
            bounds: Some(stt_core::types::BoundingBox::new(
                -178.0, -20.0, 178.0, 19.8,
            )),
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&short_dir, &[]);
    assert_failed_with(ok, &report, "escapes the declared bbox");

    // (3) An escape PAST ±180 is out-of-world data, never a seam artifact.
    let past_dir = dir.path().join("past-seam");
    write_declared_dataset(
        &past_dir,
        &[(vec![[-181.0, 0.0], [181.0, 1.0]], vec![0.0, 30.0])],
        Some(seam_crossing_declaration(2)),
        &Declaration {
            bounds: Some(stt_core::types::BoundingBox::new(
                -178.0, -20.0, 178.0, 19.8,
            )),
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&past_dir, &[]);
    assert_failed_with(ok, &report, "escapes the declared bbox");

    // (4) ⭐ Condition 3: a COMPACT declared box whose decode smeared to the
    // world edges is the 106-archive shape, and stays an error. Without this
    // the exemption would excuse every scramble that happened to land on ±180.
    let compact_dir = dir.path().join("compact-smeared");
    write_declared_dataset(
        &compact_dir,
        &[(vec![[-180.0, 37.7900], [180.0, 37.7940]], vec![0.0, 30.0])],
        Some(honest_declaration(2)),
        &Declaration {
            bounds: Some(stt_core::types::BoundingBox::new(
                -122.4050, 37.7900, -122.4030, 37.7940,
            )),
            ..Declaration::default()
        },
    );
    let (ok, report) = run_json(&compact_dir, &[]);
    assert_failed_with(ok, &report, "escapes the declared bbox");
}

/// The two tests above synthesise the DECODE and the DECLARATION separately,
/// through `PackWriter`, which is what lets them state an adversarial case the
/// real builder would never emit. This module closes the other half: that the
/// honest case is real — that `stt-build` itself, over ordinary
/// dateline-crossing polygons, produces exactly the declaration/decode
/// disagreement the exemption is scoped to, and that the pair validates clean
/// end to end.
///
/// Gated on `build-cli` because that is the exact precondition for
/// `CARGO_BIN_EXE_stt-build` to exist (it is on by default, and implied by
/// `cli`); the rest of the file only needs `validate-cli`.
#[cfg(feature = "build-cli")]
mod seam_crossing_build_e2e {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    const T0: i64 = 1_700_000_000_000;

    /// Little-endian WKB polygon. Rings are emitted verbatim, including the
    /// `|Δlon| > 180` edge that tells the builder this ring crosses ±180.
    fn wkb_polygon(rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
        let mut b = Vec::new();
        b.push(1); // little-endian
        b.extend_from_slice(&3u32.to_le_bytes()); // Polygon
        b.extend_from_slice(&(rings.len() as u32).to_le_bytes());
        for ring in rings {
            b.extend_from_slice(&(ring.len() as u32).to_le_bytes());
            for (lon, lat) in ring {
                b.extend_from_slice(&lon.to_le_bytes());
                b.extend_from_slice(&lat.to_le_bytes());
            }
        }
        b
    }

    /// 40 honest quads spanning 178°E→178°W, stacked in latitude across
    /// `[-20, 19.8]` — the shape of a Pacific storm cell or a global coverage
    /// polygon, and the exact fixture the seam defect was measured on.
    ///
    /// Every source vertex sits at |lon| = 178, so the writer's min/max fold
    /// declares `[-178, 178]` while the splitter's output decodes to
    /// `[-180, 180]`.
    fn dateline_quads() -> Vec<(Vec<u8>, i64, f64)> {
        (0..40i64)
            .map(|i| {
                let y0 = -20.0 + i as f64;
                let y1 = y0 + 0.8;
                let ring = vec![
                    (178.0, y0),
                    (-178.0, y0),
                    (-178.0, y1),
                    (178.0, y1),
                    (178.0, y0),
                ];
                (
                    wkb_polygon(&[ring]),
                    T0 + i * 60_000,
                    i as f64 * 0.5, // a `mag` column, so check 12 has a range to compare
                )
            })
            .collect()
    }

    /// Materialise the quads as a single-row-group GeoParquet file. Geometry is
    /// a plain WKB `Binary` column named `geometry`, which the reader's name
    /// heuristic picks up without any `geo` file metadata (same shape as
    /// `build_cli_reproducible.rs`'s fixture).
    fn write_input_parquet(rows: &[(Vec<u8>, i64, f64)], path: &Path) {
        let geom: Vec<&[u8]> = rows.iter().map(|(w, _, _)| w.as_slice()).collect();
        let cols: Vec<(&str, ArrayRef)> = vec![
            (
                "geometry",
                Arc::new(BinaryArray::from_vec(geom)) as ArrayRef,
            ),
            (
                "timestamp",
                Arc::new(Int64Array::from(
                    rows.iter().map(|(_, t, _)| *t).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "mag",
                Arc::new(Float64Array::from(
                    rows.iter().map(|(_, _, m)| *m).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ];
        let fields: Vec<Field> = cols
            .iter()
            .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = cols.iter().map(|(_, a)| a.clone()).collect();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();

        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// ⭐ THE MEASURED CASE. Before check 12 shared check 13's seam predicate
    /// this build exited 1, accusing an archive the builder had just written
    /// correctly of "the recorded stride-2 xyz fold that flattened 106
    /// archives" — while check 13, one line below, explained that it was the
    /// ring splitter.
    #[test]
    fn honest_dateline_polygons_built_by_stt_build_validate_clean() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("dateline-quads.parquet");
        let out = dir.path().join("archive");
        write_input_parquet(&dateline_quads(), &input);

        let build = Command::new(env!("CARGO_BIN_EXE_stt-build"))
            .arg("--input")
            .arg(&input)
            .arg("--output")
            .arg(&out)
            .args([
                "--time-field",
                "timestamp",
                "--time-format",
                "unix-ms",
                "--min-zoom",
                "0",
                "--max-zoom",
                "4",
                "--name",
                "dateline-quads",
                "--content-fingerprint",
            ])
            .output()
            .expect("failed to spawn stt-build");
        assert!(
            build.status.success(),
            "stt-build failed ({}):\nstdout:\n{}\nstderr:\n{}",
            build.status,
            String::from_utf8_lossy(&build.stdout),
            String::from_utf8_lossy(&build.stderr),
        );

        // The premise, asserted rather than assumed: the writer really did
        // declare a box its own decode escapes. If a future writer widens the
        // declared interval to the full [-180, 180] when the source straddles
        // — the fix this warning names — this assertion is what tells the next
        // reader the exemption has stopped being load-bearing here.
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("manifest.json")).unwrap()).unwrap();
        let declared = &manifest["metadata"]["content_fingerprint"]["bbox"];
        assert!(
            declared[0].as_f64().unwrap() > -180.0 && declared[2].as_f64().unwrap() < 180.0,
            "the fixture must reproduce the source-vertex fold that stops short \
             of the seam, or this test proves nothing: {declared}"
        );

        let (ok, report) = run_json(&out, &[]);
        assert!(
            ok,
            "an honest dateline-crossing polygon build must validate clean: {report}"
        );
        assert_eq!(errors_of(&report).len(), 0, "report: {report}");
        assert_eq!(report["fingerprint_checked"], true, "report: {report}");

        let warnings = report_warnings(&report);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("ANTIMERIDIAN SEAM") && w.contains("declared bbox")),
            "check 12 must say out loud WHY it did not fail: {warnings:#?}"
        );
    }
}

/// The vertical axis, same direction: a declared `metadata.z_range` that does
/// not contain the decoded altitudes fails an attesting archive. Altitude-aware
/// selection pre-intersects against it exactly as tile selection does against
/// bounds.
#[test]
fn understated_z_range_fails_an_attesting_archive() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("shallow-claim");

    // A 3D point layer — the `elevation-fold` encoder path, which is how an AV
    // archive ships altitude. Altitudes run 0..30; the manifest claims 0..10.
    let mut writer = PackWriter::create(&archive, BlobOrdering::Auto, 64 * 1024 * 1024)
        .unwrap()
        .with_encoder_config(EncoderConfig {
            point_elevation_column: "alt".into(),
            ..EncoderConfig::default()
        });
    let cfg = writer.encoder_config();
    let (t_start, t_end) = (1_000i64, 2_000i64);
    let points = vec![
        [-122.4050, 37.7900],
        [-122.4040, 37.7920],
        [-122.4030, 37.7940],
    ];
    let n = points.len();
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "default".into(),
        feature_ids: (1..=n as u64).collect(),
        start_times: vec![t_start; n],
        end_times: vec![t_end; n],
        geometry: GeometryColumn::Point(points),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "alt".into(),
            PropertyColumn::Numeric(vec![Some(0.0), Some(15.0), Some(30.0)]),
        )],
    };
    let payload = encode_tile_with(&[layer], &cfg).unwrap();
    writer
        .add_tile_full(
            &TileId::new(8, 0, 0, t_start as u64),
            t_start,
            t_end,
            None,
            n as u32,
            None,
            &payload,
        )
        .unwrap();
    let mut meta = Metadata::new("z-claim")
        .with_time_range(TimeRange::new(t_start as u64, t_end as u64))
        .with_zoom_levels(8, 8)
        .with_bounds(honest_bounds())
        .with_z_range(Some([0.0, 10.0]));
    meta.properties
        .insert("bounds_mode".to_string(), "vertex".to_string());
    writer.finalize(&meta).unwrap();

    let (ok, report) = run_json(&archive, &[]);
    let errors = assert_failed_with(ok, &report, "declared bounds");
    assert!(
        errors.iter().any(|e| e.contains("z_range")),
        "the vertical claim must be named: {errors:#?}"
    );
}

// ---------------------------------------------------------------------------
// End to end: `stt-build --content-fingerprint` → `stt-validate`.
//
// Every test above synthesises its archive through `PackWriter` directly, which
// proves the CHECK works but says nothing about the WRITER. These two drive the
// real `stt-build` binary over a real GeoParquet, so the emission side, the
// tolerance derivation and the recompute are all exercised through the shipping
// pipeline — the loop the R1 gate will actually run.
//
// Gated on `build-cli` (this file already requires `validate-cli`); with only
// the validator feature on, the block compiles out rather than failing to find
// `CARGO_BIN_EXE_stt-build`.
// ---------------------------------------------------------------------------
#[cfg(feature = "build-cli")]
mod end_to_end {
    use super::*;
    use std::sync::Arc;

    /// Little-endian WKB point — the geometry encoding `stt-build`'s reader
    /// picks up from a `geometry` Binary column with no `geo` file metadata.
    fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
        let mut b = Vec::with_capacity(21);
        b.push(1);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&lon.to_le_bytes());
        b.extend_from_slice(&lat.to_le_bytes());
        b
    }

    /// A 24-row point dataset over a small AV-scale extent, with one numeric
    /// and one categorical property.
    fn write_input_parquet(path: &Path) {
        use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let n = 24usize;
        let t0 = 1_700_000_000_000i64;
        let geom: Vec<Vec<u8>> = (0..n)
            .map(|i| wkb_point(-122.4050 + i as f64 * 0.0002, 37.7900 + i as f64 * 0.0002))
            .collect();
        let cols: Vec<(&str, ArrayRef)> = vec![
            (
                "geometry",
                Arc::new(BinaryArray::from_opt_vec(
                    geom.iter().map(|g| Some(g.as_slice())).collect(),
                )) as ArrayRef,
            ),
            (
                "timestamp",
                Arc::new(Int64Array::from(
                    (0..n).map(|i| t0 + i as i64 * 60_000).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "speed",
                Arc::new(Float64Array::from(
                    (0..n).map(|i| i as f64 * 1.25).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "kind",
                Arc::new(StringArray::from(
                    (0..n)
                        .map(|i| if i % 2 == 0 { "car" } else { "bus" })
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ];
        let schema = Arc::new(Schema::new(
            cols.iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            cols.iter().map(|(_, a)| a.clone()).collect(),
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn build(input: &Path, out: &Path, extra: &[&str]) {
        build_zoomed(input, out, "8", "10", extra);
    }

    /// The same builder invocation with the zoom range spelled out, so a
    /// fixture can be forced to span MORE THAN ONE TILE at its coarsest zoom —
    /// the only configuration in which per-tile row indices and feature ids
    /// stop coinciding (see `multi_tile_*` below).
    fn build_zoomed(input: &Path, out: &Path, min_zoom: &str, max_zoom: &str, extra: &[&str]) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_stt-build"));
        cmd.arg("--input")
            .arg(input)
            .arg("--output")
            .arg(out)
            .args(["--time-field", "timestamp", "--time-format", "unix-ms"])
            .args(["--min-zoom", min_zoom, "--max-zoom", max_zoom])
            .args(["--temporal-bucket", "1h"])
            .args(extra);
        let output = cmd.output().expect("run stt-build");
        assert!(
            output.status.success(),
            "stt-build failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// ⭐ THE R1 GATE LOOP. A real `stt-build --content-fingerprint` archive
    /// validates clean under a FULL decode — which is the strict mode
    /// (containment *and* equality on every bbox edge and column range). If the
    /// builder's declared statistics did not match what the tiler, clipper,
    /// pyramid and encoder actually emit, this is where it shows.
    #[test]
    fn built_archive_declares_a_fingerprint_that_validates() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.parquet");
        write_input_parquet(&input);

        // Control: no flag ⇒ no key, and the manifest is the shape every
        // published archive has.
        let plain = dir.path().join("plain");
        build(&input, &plain, &[]);
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(plain.join("manifest.json")).unwrap()).unwrap();
        assert!(
            raw["metadata"].get("content_fingerprint").is_none(),
            "the flag is OFF by default — no manifest may gain a key without it"
        );
        let (ok, report) = run_json(&plain, &[]);
        assert!(ok, "report: {report}");
        assert_eq!(report["fingerprint_checked"], false);

        // With the flag: the key is emitted and the full-decode comparison is
        // clean.
        let stamped = dir.path().join("stamped");
        build(&input, &stamped, &["--content-fingerprint"]);
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(stamped.join("manifest.json")).unwrap()).unwrap();
        let fp = &raw["metadata"]["content_fingerprint"];
        assert_eq!(fp["version"], 1, "manifest: {raw}");
        assert_eq!(fp["distinct_feature_count"], 24);
        assert!(fp["numeric_ranges"]["speed"].is_array(), "manifest: {raw}");
        assert_eq!(fp["categorical_cardinality"]["kind"], 2);
        assert_eq!(
            fp["coord_tolerance_deg"], 0.0,
            "an unquantized build claims EXACT coordinates"
        );

        let (ok, report) = run_json(&stamped, &[]);
        assert!(
            ok,
            "a freshly built archive must satisfy its own fingerprint under a \
             FULL decode (containment AND equality): {report}"
        );
        assert_eq!(errors_of(&report).len(), 0, "report: {report}");
        assert_eq!(report["fingerprint_checked"], true);

        // ...and under a sampled decode, where only containment applies.
        let (ok, report) = run_json(&stamped, &["--sample", "3"]);
        assert!(ok, "report: {report}");
        assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    }

    /// DETERMINISM. Two builds of one input emit BYTE-IDENTICAL
    /// `content_fingerprint` JSON — the fingerprint is a pure function of the
    /// decoded content, with no clock, no RNG and no iteration-order leak.
    ///
    /// And a quantized build declares the tolerance its own grid implies, so an
    /// honest lossy archive still validates clean rather than tripping the
    /// check it exists to pass.
    #[test]
    fn fingerprint_is_byte_identical_across_builds_and_survives_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.parquet");
        write_input_parquet(&input);

        let read_fp = |archive: &Path| -> serde_json::Value {
            let raw: serde_json::Value =
                serde_json::from_slice(&std::fs::read(archive.join("manifest.json")).unwrap())
                    .unwrap();
            raw["metadata"]["content_fingerprint"].clone()
        };

        let first = dir.path().join("build-1");
        let second = dir.path().join("build-2");
        build(&input, &first, &["--content-fingerprint"]);
        build(&input, &second, &["--content-fingerprint"]);
        assert_eq!(
            serde_json::to_string(&read_fp(&first)).unwrap(),
            serde_json::to_string(&read_fp(&second)).unwrap(),
            "two builds of one input must emit byte-identical fingerprint JSON"
        );

        // Quantized: the declared tolerance is the world grid's step, and the
        // archive declares the `coord-quant` capability that entitles it to
        // one. Both halves matter — the validator rejects a tolerance whose
        // capability is missing.
        let quantized = dir.path().join("quantized");
        build(
            &input,
            &quantized,
            &["--content-fingerprint", "--quantize-coords", "1"],
        );
        let fp = read_fp(&quantized);
        let tolerance = fp["coord_tolerance_deg"].as_f64().unwrap();
        assert!(
            (tolerance - 1.0 / 111_320.0).abs() < 1e-12,
            "a 1 m grid implies a 1/111320 deg step, got {tolerance}"
        );
        let manifest =
            Manifest::from_json_bytes(&std::fs::read(quantized.join("manifest.json")).unwrap())
                .unwrap();
        assert!(
            manifest
                .capabilities
                .iter()
                .any(|c| c == stt_core::pack::CAPABILITY_COORD_QUANT),
            "capabilities were {:?}",
            manifest.capabilities
        );

        let (ok, report) = run_json(&quantized, &[]);
        assert!(
            ok,
            "an honestly quantized archive must validate clean — the tolerance \
             exists precisely so it does: {report}"
        );
        assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    }

    // -----------------------------------------------------------------------
    // ⭐ BLOCKER 3 — `--content-fingerprint --point-elevation-column <col>` on
    // a NON-POINT layer.
    //
    // The builder folded the named column into `z_range` and dropped it from
    // `numeric_ranges` on the assumption that the encoder had consumed it. The
    // encoder does no such thing off a point layer — it gates on
    // `GeometryColumn::Point`, and the flag's own help says "Only affects POINT
    // layers". So on a linestring build the fingerprint was wrong twice:
    //
    //   * a `z_range` invented for an archive with NO 3D geometry, and
    //   * the named column missing from `numeric_ranges`, which switched OFF
    //     the attribute check that catches a corrupted rebuild — on precisely
    //     the column an operator cared enough about to name.
    //
    // Two warnings, EXIT=0. A fingerprint that is wrong is worse than none,
    // because it CERTIFIES.
    // -----------------------------------------------------------------------

    /// Little-endian WKB LineString.
    fn wkb_linestring(points: &[(f64, f64)]) -> Vec<u8> {
        let mut b = Vec::with_capacity(9 + points.len() * 16);
        b.push(1);
        b.extend_from_slice(&2u32.to_le_bytes()); // LineString
        b.extend_from_slice(&(points.len() as u32).to_le_bytes());
        for (lon, lat) in points {
            b.extend_from_slice(&lon.to_le_bytes());
            b.extend_from_slice(&lat.to_le_bytes());
        }
        b
    }

    /// 24 two-vertex linestrings, laid out as 6 axis-aligned BOX OUTLINES (four
    /// edges each) over the same AV-scale extent the point fixture uses.
    ///
    /// The box-outline shape is load-bearing, not decoration. A feature's
    /// anchor is its geometric CENTROID, so for a general line the legacy
    /// centroid bbox is strictly inside the vertex bbox — and check 13 would
    /// then fail this archive for an under-stated `metadata.bounds`, which is a
    /// DIFFERENT defect (the R1 bounds flip owns it, `DEFAULT_BOUNDS_MODE` is
    /// still `Centroid`). A box outline's four edge midpoints hit every corner
    /// coordinate of the box, so centroid bbox == vertex bbox and this test
    /// isolates the fingerprint defect it is actually about.
    fn write_linestring_parquet(path: &Path) {
        use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let boxes = 6usize;
        let n = boxes * 4;
        let t0 = 1_700_000_000_000i64;
        let mut geom: Vec<Vec<u8>> = Vec::with_capacity(n);
        for b in 0..boxes {
            let x0 = -122.4050 + b as f64 * 0.0020;
            let x1 = x0 + 0.0010;
            let y0 = 37.7900 + b as f64 * 0.0020;
            let y1 = y0 + 0.0010;
            geom.push(wkb_linestring(&[(x0, y0), (x0, y1)])); // left edge
            geom.push(wkb_linestring(&[(x1, y0), (x1, y1)])); // right edge
            geom.push(wkb_linestring(&[(x0, y0), (x1, y0)])); // bottom edge
            geom.push(wkb_linestring(&[(x0, y1), (x1, y1)])); // top edge
        }
        let cols: Vec<(&str, ArrayRef)> = vec![
            (
                "geometry",
                Arc::new(BinaryArray::from_opt_vec(
                    geom.iter().map(|g| Some(g.as_slice())).collect(),
                )) as ArrayRef,
            ),
            (
                "timestamp",
                Arc::new(Int64Array::from(
                    (0..n).map(|i| t0 + i as i64 * 60_000).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "speed",
                Arc::new(Float64Array::from(
                    (0..n).map(|i| i as f64 * 1.25).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "kind",
                Arc::new(StringArray::from(
                    (0..n)
                        .map(|i| if i % 2 == 0 { "car" } else { "bus" })
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ];
        let schema = Arc::new(Schema::new(
            cols.iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            cols.iter().map(|(_, a)| a.clone()).collect(),
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn read_fingerprint(archive: &Path) -> serde_json::Value {
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(archive.join("manifest.json")).unwrap()).unwrap();
        raw["metadata"]["content_fingerprint"].clone()
    }

    /// ⭐ THE PIN. On a LINESTRING build the elevation fold never happens, so
    /// the fingerprint must invent no `z_range` and must KEEP the named column
    /// checkable — and the archive must still validate clean under a full
    /// decode (containment AND equality).
    #[test]
    fn point_elevation_column_invents_no_z_range_on_a_linestring_build() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("lines.parquet");
        write_linestring_parquet(&input);

        let folded = dir.path().join("lines-folded");
        build(
            &input,
            &folded,
            &["--content-fingerprint", "--point-elevation-column", "speed"],
        );
        let fp = read_fingerprint(&folded);

        assert!(
            fp.get("z_range").is_none(),
            "a linestring archive has NO 3D geometry, so the fingerprint may claim no \
             vertical extent — the encoder folds only into POINT layers: {fp}"
        );
        let speed = fp["numeric_ranges"]["speed"]
            .as_array()
            .unwrap_or_else(|| panic!("`speed` must stay in numeric_ranges: {fp}"));
        assert_eq!(speed[0].as_f64(), Some(0.0), "fingerprint: {fp}");
        assert_eq!(speed[1].as_f64(), Some(28.75), "fingerprint: {fp}");

        // The strongest statement of the contract: on a non-point layer the
        // flag is a NO-OP, so it may not move a single byte of the fingerprint.
        let plain = dir.path().join("lines-plain");
        build(&input, &plain, &["--content-fingerprint"]);
        assert_eq!(
            serde_json::to_string(&read_fingerprint(&plain)).unwrap(),
            serde_json::to_string(&fp).unwrap(),
            "--point-elevation-column does nothing to a linestring build, so the two \
             fingerprints must be byte-identical"
        );

        // And the declared block is true: a full decode agrees with it.
        let (ok, report) = run_json(&folded, &[]);
        assert!(ok, "report: {report}");
        assert_eq!(errors_of(&report).len(), 0, "report: {report}");
        assert_eq!(report["fingerprint_checked"], true);
    }

    /// The control that keeps the fix a DISCRIMINATION rather than a blanket
    /// disable: on a POINT build the same flag really does fold, so `z_range`
    /// appears and the column leaves `numeric_ranges` — and that archive
    /// validates clean too.
    #[test]
    fn point_elevation_column_still_folds_on_a_point_build() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("points.parquet");
        write_input_parquet(&input);

        let folded = dir.path().join("points-folded");
        build(
            &input,
            &folded,
            &["--content-fingerprint", "--point-elevation-column", "speed"],
        );
        let fp = read_fingerprint(&folded);

        assert_eq!(
            fp["z_range"][0].as_f64(),
            Some(0.0),
            "the fold DID happen here, so the vertical claim is real: {fp}"
        );
        assert_eq!(fp["z_range"][1].as_f64(), Some(28.75), "fingerprint: {fp}");
        assert!(
            fp["numeric_ranges"].get("speed").is_none(),
            "the folded column lives in geometry z now, so declaring a property range \
             for it would describe a column the archive does not carry: {fp}"
        );

        let (ok, report) = run_json(&folded, &[]);
        assert!(
            ok,
            "the 3D point archive must satisfy its own fingerprint: {report}"
        );
        assert_eq!(errors_of(&report).len(), 0, "report: {report}");
    }

    // -----------------------------------------------------------------------
    // BLOCKER A — the distinct-id BASIS and the decoded-ROW floor, end to end.
    //
    // ⚠️ EVERY fixture here is MULTI-TILE, and each test PROVES it is before it
    // asserts anything else. That is not ceremony: the bug these tests pin
    // survived ~30 single-tile end-to-end tests precisely because a per-tile row
    // index and a source-feature id COINCIDE when the whole dataset is one tile.
    // A fixture that silently degenerated to one tile would go green while
    // testing nothing, so the precondition is asserted, not assumed.
    // -----------------------------------------------------------------------

    /// A 600-point CONUS grid — wide enough that `--min-zoom 2` really does
    /// span more than one tile — with one numeric and one categorical property.
    ///
    /// `keep` selects the subset of the 600 source rows to write, which is how
    /// the LOSSY half of the row-floor pin is built: the same generator, fewer
    /// rows, so the two archives differ in exactly one way.
    fn write_conus_parquet(path: &Path, keep: &dyn Fn(usize) -> bool) {
        use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let n = 600usize;
        let cols_wide = 25usize;
        let t0 = 1_700_000_000_000i64;
        let idx: Vec<usize> = (0..n).filter(|i| keep(*i)).collect();
        let geom: Vec<Vec<u8>> = idx
            .iter()
            .map(|i| {
                wkb_point(
                    -120.0 + (i % cols_wide) as f64 * 1.8,
                    26.0 + ((i / cols_wide) % 24) as f64 * 0.9,
                )
            })
            .collect();
        let cols: Vec<(&str, ArrayRef)> = vec![
            (
                "geometry",
                Arc::new(BinaryArray::from_opt_vec(
                    geom.iter().map(|g| Some(g.as_slice())).collect(),
                )) as ArrayRef,
            ),
            (
                "timestamp",
                Arc::new(Int64Array::from(
                    idx.iter()
                        .map(|i| t0 + *i as i64 * 720_000)
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "speed",
                Arc::new(Float64Array::from(
                    idx.iter()
                        .map(|i| (i % cols_wide) as f64 * 1.25)
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "kind",
                Arc::new(StringArray::from(
                    idx.iter()
                        .map(|i| if i % 2 == 0 { "car" } else { "bus" })
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ];
        let schema = Arc::new(Schema::new(
            cols.iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            cols.iter().map(|(_, a)| a.clone()).collect(),
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Open the archive and return `(distinct (x,y) tiles at min_zoom, rows per
    /// zoom)`, ASSERTING that the coarsest zoom really is covered by more than
    /// one tile.
    ///
    /// This is the guard that stops the fixture degenerating. When a dataset
    /// fits in a single tile, `build_point_layer`'s per-tile row index and a
    /// source feature's identity are the same number, the archive-wide distinct
    /// id count equals the feature count by accident, and every assertion below
    /// passes while proving nothing.
    fn assert_multi_tile(archive: &Path) -> (usize, std::collections::BTreeMap<u8, u64>) {
        let reader = stt_core::PackedReader::open(archive.join("manifest.json")).unwrap();
        let min_zoom = reader.metadata().min_zoom;
        let mut coarse: std::collections::BTreeSet<(u32, u32)> = Default::default();
        let mut rows_by_zoom: std::collections::BTreeMap<u8, u64> = Default::default();
        for e in reader.entries() {
            *rows_by_zoom.entry(e.zoom).or_default() += e.feature_count as u64;
            if e.zoom == min_zoom {
                coarse.insert((e.x, e.y));
            }
        }
        assert!(
            coarse.len() > 1,
            "FIXTURE DEGENERATED to a single tile at z{min_zoom} ({} distinct (x,y)). The bug \
             under test is invisible on a single-tile archive — widen the fixture or lower \
             --min-zoom. rows per zoom: {rows_by_zoom:?}",
            coarse.len()
        );
        (coarse.len(), rows_by_zoom)
    }

    fn notes_of(report: &serde_json::Value) -> Vec<String> {
        report["notes"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|n| n.as_str().expect("notes are strings").to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn warnings_of(report: &serde_json::Value) -> Vec<String> {
        report["warnings"]
            .as_array()
            .expect("report must carry a warnings array")
            .iter()
            .map(|w| w.as_str().expect("warnings are strings").to_string())
            .collect()
    }

    /// ⭐ THE BLOCKER-A REGRESSION. A healthy MULTI-TILE archive of id-less
    /// points validates CLEAN, and says why the two counts differ in a NOTE.
    ///
    /// Before the basis gate this exact build — 600 WKB points over CONUS,
    /// `--min-zoom 2 --max-zoom 8 --temporal-bucket 1h --content-fingerprint` —
    /// exited 1 with "FEATURE LOSS: the archive decodes 5 distinct feature ids
    /// but the fingerprint declares 600 — 99.2 % of the declared features are
    /// MISSING", over a complete decode of 4 200 rows, with nothing missing.
    #[test]
    fn healthy_multi_tile_archive_notes_the_id_basis_and_exits_zero() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("conus.parquet");
        write_conus_parquet(&input, &|_| true);
        let archive = dir.path().join("conus");
        build_zoomed(&input, &archive, "2", "8", &["--content-fingerprint"]);

        // PRECONDITION, asserted: more than one tile at the coarsest zoom, and
        // the pyramid really does replicate (so a whole-archive row total would
        // NOT have caught a loss here — see the row-floor test below).
        let (coarse_tiles, rows_by_zoom) = assert_multi_tile(&archive);
        assert!(coarse_tiles > 1);
        let total_rows: u64 = rows_by_zoom.values().sum();
        assert!(
            total_rows > 600 * 4,
            "the pyramid must replicate for this fixture to be the one that hid the bug: \
             {rows_by_zoom:?}"
        );
        assert_eq!(
            rows_by_zoom.values().copied().max(), // the fullest single zoom
            Some(600),
            "the row floor's comparand must be the source-feature count: {rows_by_zoom:?}"
        );

        let fp = read_fingerprint(&archive);
        assert_eq!(fp["distinct_feature_count"].as_u64(), Some(600));

        let (ok, report) = run_json(&archive, &[]);
        assert!(
            ok,
            "a healthy multi-tile archive must exit 0; errors: {:#?}",
            errors_of(&report)
        );
        assert_eq!(errors_of(&report).len(), 0, "report: {report}");

        // The basis is the conservative one, resolved from the archive's own
        // attestation (`feature_id_scope = local`), not from the numbers.
        assert_eq!(report["distinct_id_basis"], "source-features");
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(archive.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(
            raw["metadata"]["properties"]["feature_id_scope"], "local",
            "the builder must record what it could (not) prove: {}",
            raw["metadata"]["properties"]
        );

        // The deviation is STATED — as a note.
        let notes = notes_of(&report);
        assert_eq!(notes.len(), 1, "notes: {notes:#?}");
        assert!(
            notes[0].contains("DIFFERENT QUANTITIES")
                && notes[0].contains("PER-TILE ROW INDEX")
                && notes[0].contains("feature_id_scope"),
            "the note must state the category error and name what would arm the \
             real check: {}",
            notes[0]
        );

        // …and NOT as a warning. Warnings are the budget O4 drives to zero; a
        // finding expected on every archive in the fleet may not spend it.
        let warnings = warnings_of(&report);
        assert!(
            !warnings.iter().any(|w| w.contains("distinct-id")),
            "the note must not be filed as a warning: {warnings:#?}"
        );
    }

    /// ⭐ RETAINED LOSS DETECTION, on a MULTI-TILE archive. Real loss still
    /// fails the run — through the decoded-ROW floor, not the distinct-id
    /// comparison the basis gate disarmed.
    ///
    /// The workflow is the documented one for accepting a rebuild: capture the
    /// truth from the good archive with `--emit-fingerprint`, then hold the
    /// rebuild to it with `--expect-fingerprint`.
    #[test]
    fn real_loss_on_a_multi_tile_archive_still_fails_through_the_row_floor() {
        let dir = tempfile::tempdir().unwrap();

        // The good archive, and its captured truth.
        let full_input = dir.path().join("full.parquet");
        write_conus_parquet(&full_input, &|_| true);
        let full = dir.path().join("full");
        build_zoomed(&full_input, &full, "2", "8", &["--content-fingerprint"]);
        assert_multi_tile(&full);
        let truth = dir.path().join("truth.json");
        let (ok, _) = run_json(&full, &["--emit-fingerprint", truth.to_str().unwrap()]);
        assert!(ok, "the good archive must validate before it can be truth");

        // The lossy rebuild: the same generator with a CONTIGUOUS INTERIOR
        // block dropped, chosen so every extreme the fingerprint records —
        // bbox, time range, `speed` range, `kind` cardinality — is carried by a
        // surviving row. The ONLY thing that changed is how many features there
        // are, which is exactly the defect class under test.
        let lossy_input = dir.path().join("lossy.parquet");
        write_conus_parquet(&lossy_input, &|i| !(200..400).contains(&i));
        let lossy = dir.path().join("lossy");
        build_zoomed(&lossy_input, &lossy, "2", "8", &["--content-fingerprint"]);

        let (_, rows_by_zoom) = assert_multi_tile(&lossy);
        assert_eq!(
            rows_by_zoom.values().copied().max(),
            Some(400),
            "the lossy fixture must really be short: {rows_by_zoom:?}"
        );
        // ⚠️ WHY THE FLOOR IS PER-ZOOM. The lossy archive still decodes 2 800
        // rows in TOTAL against a declared 600 — a whole-archive row floor sees
        // 2 800 >= 600 and says nothing at all. Pyramid replication masks the
        // loss; the fullest single zoom does not.
        let total: u64 = rows_by_zoom.values().sum();
        assert!(
            total > 600,
            "a total-row floor would be vacuous here: {rows_by_zoom:?}"
        );

        // On its OWN (rebuilt, now-400) fingerprint the archive is internally
        // consistent — which is precisely why the captured truth is needed.
        let (ok, report) = run_json(&lossy, &[]);
        assert!(ok, "self-consistent rebuild: {:#?}", errors_of(&report));

        // Held to the truth, it fails, and the failure is the ROW FLOOR.
        let (ok, report) = run_json(&lossy, &["--expect-fingerprint", truth.to_str().unwrap()]);
        let errors = assert_failed_with(ok, &report, "FEATURE LOSS:");
        let loss: Vec<&String> = errors
            .iter()
            .filter(|e| e.contains("FEATURE LOSS:"))
            .collect();
        assert!(
            loss.iter().any(|e| e.contains("fullest decoded zoom")
                && e.contains("400")
                && e.contains("600")
                && e.contains("33.3%")),
            "the error must be the decoded-row floor and must name the numbers: {loss:#?}"
        );
        assert!(
            !loss.iter().any(|e| e.contains("distinct feature ids")),
            "the distinct-id comparison is disarmed on this archive; only the row \
             floor may fire: {loss:#?}"
        );

        // The documented escape hatch reaches it, and says that it did.
        let (ok, report) = run_json(
            &lossy,
            &[
                "--expect-fingerprint",
                truth.to_str().unwrap(),
                "--allow-distinct-shortfall",
            ],
        );
        assert!(ok, "downgraded: {:#?}", errors_of(&report));
        assert!(
            warnings_of(&report)
                .iter()
                .any(|w| w.contains("DOWNGRADED") && w.contains("FEATURE LOSS:")),
            "a suppressed finding must not read as a clean run: {:#?}",
            warnings_of(&report)
        );

        // A SAMPLED run over the same lossy archive reports the shortfall as
        // sampling noise, never as the error — the two wordings must not be
        // confusable.
        let (ok, report) = run_json(
            &lossy,
            &[
                "--expect-fingerprint",
                truth.to_str().unwrap(),
                "--sample",
                "4",
            ],
        );
        assert!(ok, "sampled: {:#?}", errors_of(&report));
        assert!(
            warnings_of(&report)
                .iter()
                .any(|w| w.contains("SAMPLING NOISE") && !w.contains("FEATURE LOSS:")),
            "warnings: {:#?}",
            warnings_of(&report)
        );
    }

    /// The attestation seam is WIRED, not decoration: an archive that declares
    /// `feature_id_scope = global` really does get the strict comparison back.
    ///
    /// `--feature-id-scope global` is the operator asserting what the builder
    /// could not prove, and this fixture is the case the flag's ⚠️ warns about
    /// — id-less points, so the assertion is FALSE and the strict check then
    /// reports loss on a healthy archive. That is the point: it proves the seam
    /// changes behaviour, and it proves the DEFAULT is the safe one.
    #[test]
    fn the_global_attestation_really_arms_the_distinct_id_check() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("conus.parquet");
        write_conus_parquet(&input, &|_| true);
        let archive = dir.path().join("conus-attested");
        build_zoomed(
            &input,
            &archive,
            "2",
            "8",
            &["--content-fingerprint", "--feature-id-scope", "global"],
        );
        assert_multi_tile(&archive);

        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(archive.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(raw["metadata"]["properties"]["feature_id_scope"], "global");

        // The FACT is recorded beside the assertion, and it disagrees with it:
        // these are id-less points, so the writer really did use the row index.
        assert_eq!(
            raw["metadata"]["properties"]["feature_id_construction"], "row-index",
            "--feature-id-scope global asserts; it must not falsify the record \
             of what the writer actually did: {}",
            raw["metadata"]["properties"]
        );

        let (ok, report) = run_json(&archive, &[]);
        assert_eq!(report["distinct_id_basis"], "decoded-ids");
        assert_eq!(
            report["feature_id_construction"], "row-index",
            "the report must surface the construction the basis disagrees with: {report}"
        );
        let errors = assert_failed_with(ok, &report, "FEATURE LOSS:");
        assert!(
            errors.iter().any(|e| e.contains("distinct feature ids")
                && e.contains("dataset-wide key")
                && e.contains("feature_id_scope=global")),
            "the strict path must fire and must name what armed it: {errors:#?}"
        );

        // The row floor is silent here — 600 rows at the fullest zoom against a
        // declared 600 — so the two checks are independent, not duplicates.
        assert!(
            !errors.iter().any(|e| e.contains("fullest decoded zoom")),
            "the row floor must stay quiet on a healthy archive: {errors:#?}"
        );

        // Rolling back to `local` restores the clean run on the same data —
        // and it must do so THROUGH the construction key, which on a different
        // fixture would have armed the check on its own.
        let rolled_back = dir.path().join("conus-local");
        build_zoomed(
            &input,
            &rolled_back,
            "2",
            "8",
            &["--content-fingerprint", "--feature-id-scope", "local"],
        );
        let (ok, report) = run_json(&rolled_back, &[]);
        assert!(ok, "rollback: {:#?}", errors_of(&report));
        assert_eq!(report["distinct_id_basis"], "source-features");
    }

    // -----------------------------------------------------------------------
    // ⭐⭐ BLOCKER 1 — the id BASIS keyed on the writer's id CONSTRUCTION.
    //
    // `--feature-id-scope auto` used to ask "does every source feature carry an
    // id?", and NO ingest path in stt-build populates `ParsedFeature.geojson.id`
    // (GeoParquet, PostGIS and DuckDB all build `Feature { id: None, .. }`). So
    // the strict basis had no live producer, and the per-zoom row floor — the
    // only loss detection left — is LOOSE exactly where clipping replicates a
    // feature across tiles. On a line or polygon dataset the surviving features'
    // extra rows cover for the dropped ones, and a 40 % rebuild loss validated
    // clean with exit 0.
    //
    // The writer already knows which id-construction path it took. An id-less
    // LINE or POLYGON gets the whole-feature ANCHOR HASH, which the tiler copies
    // into every clipped piece, so one source feature keeps one id everywhere —
    // a dataset-wide key, no operator assertion required.
    //
    // ⚠️ EVERY fixture below is MULTI-TILE and CLIPPING, and each test PROVES
    // both before it asserts anything else — `assert_multi_tile` for the first,
    // a rows-vs-features ratio for the second. A fixture that degenerated to one
    // tile per feature would make the row floor tight again and the test would
    // go green while testing nothing.
    // -----------------------------------------------------------------------

    /// Little-endian WKB Polygon with a single exterior ring.
    fn wkb_polygon(ring: &[(f64, f64)]) -> Vec<u8> {
        let mut b = Vec::with_capacity(13 + ring.len() * 16);
        b.push(1);
        b.extend_from_slice(&3u32.to_le_bytes()); // Polygon
        b.extend_from_slice(&1u32.to_le_bytes()); // one ring
        b.extend_from_slice(&(ring.len() as u32).to_le_bytes());
        for (lon, lat) in ring {
            b.extend_from_slice(&lon.to_le_bytes());
            b.extend_from_slice(&lat.to_le_bytes());
        }
        b
    }

    /// 600 CONUS features that each span SEVERAL tiles at the fullest zoom —
    /// the shape the row floor cannot police.
    ///
    /// `polygons` picks the geometry kind (a 1.6° × 0.7° box outline vs the
    /// filled box); `keep` selects the subset of the 600 source rows, which is
    /// how the lossy half is built from the same generator.
    ///
    /// The extent is deliberately the same CONUS grid `write_conus_parquet`
    /// uses, so the only difference between the point fixture (row-index ids,
    /// loss caught by the row floor) and this one (anchor-hash ids, loss caught
    /// by the distinct-id comparison) is the geometry kind.
    fn write_conus_spanning_parquet(path: &Path, polygons: bool, keep: &dyn Fn(usize) -> bool) {
        use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let n = 600usize;
        let cols_wide = 25usize;
        let t0 = 1_700_000_000_000i64;
        let idx: Vec<usize> = (0..n).filter(|i| keep(*i)).collect();
        let geom: Vec<Vec<u8>> = idx
            .iter()
            .map(|i| {
                let x0 = -120.0 + (i % cols_wide) as f64 * 1.8;
                let y0 = 26.0 + ((i / cols_wide) % 24) as f64 * 0.9;
                let (x1, y1) = (x0 + 1.6, y0 + 0.7);
                if polygons {
                    wkb_polygon(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
                } else {
                    // A closed box OUTLINE, not a diagonal: its four edge
                    // midpoints touch every corner, so the feature's centroid
                    // anchor sits at the box centre and the legacy centroid
                    // bbox question stays out of this test entirely.
                    wkb_linestring(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
                }
            })
            .collect();
        let cols: Vec<(&str, ArrayRef)> = vec![
            (
                "geometry",
                Arc::new(BinaryArray::from_opt_vec(
                    geom.iter().map(|g| Some(g.as_slice())).collect(),
                )) as ArrayRef,
            ),
            (
                "timestamp",
                Arc::new(Int64Array::from(
                    idx.iter()
                        .map(|i| t0 + *i as i64 * 720_000)
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "speed",
                Arc::new(Float64Array::from(
                    idx.iter()
                        .map(|i| (i % cols_wide) as f64 * 1.25)
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "kind",
                Arc::new(StringArray::from(
                    idx.iter()
                        .map(|i| if i % 2 == 0 { "car" } else { "bus" })
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ];
        let schema = Arc::new(Schema::new(
            cols.iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            cols.iter().map(|(_, a)| a.clone()).collect(),
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn manifest_properties(archive: &Path) -> serde_json::Value {
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(archive.join("manifest.json")).unwrap()).unwrap();
        raw["metadata"]["properties"].clone()
    }

    /// ⭐ THE BLOCKER-1 PIN, run over LINES and POLYGONS alike.
    ///
    /// Four claims per geometry kind, in order, because dropping any one of
    /// them would let the fix look done and not be:
    ///
    /// 1. **The fixture really replicates.** Rows at the fullest zoom are
    ///    materially MORE than the source-feature count, which is precisely the
    ///    condition under which the row floor stops being a floor.
    /// 2. **The writer records the fact and it arms the check with no operator
    ///    in the loop** — `feature_id_construction = anchor-hash`,
    ///    `feature_id_scope = global`, `distinct_id_basis = decoded-ids` — and
    ///    the healthy archive still exits 0 with no error and no distinct-id
    ///    note.
    /// 3. **A 40 % rebuild loss FAILS**, through the distinct-id comparison.
    /// 4. **The row floor stays SILENT on that same loss** — the measurement
    ///    that makes claim 3 load-bearing rather than duplicated work.
    #[test]
    fn clipped_line_and_polygon_loss_is_caught_by_the_anchor_hash_id_basis() {
        for (kind, polygons) in [("lines", false), ("polys", true)] {
            let dir = tempfile::tempdir().unwrap();

            // --- the healthy archive, and its captured truth -----------------
            let full_input = dir.path().join(format!("{kind}-full.parquet"));
            write_conus_spanning_parquet(&full_input, polygons, &|_| true);
            let full = dir.path().join(format!("{kind}-full"));
            build_zoomed(&full_input, &full, "2", "8", &["--content-fingerprint"]);

            let (coarse_tiles, rows_by_zoom) = assert_multi_tile(&full);
            assert!(coarse_tiles > 1, "{kind}: {rows_by_zoom:?}");
            let fullest = rows_by_zoom
                .values()
                .copied()
                .max()
                .expect("some zoom decoded rows");
            // (1) THE PRECONDITION THAT MAKES THIS TEST MEAN ANYTHING. Clipping
            // must replicate: if one feature produced one row the row floor
            // would be tight and the distinct-id check redundant.
            assert!(
                fullest >= 600 * 3 / 2,
                "{kind}: FIXTURE DEGENERATED — the fullest zoom carries {fullest} rows for 600 \
                 features, so clipping is not replicating and the row floor would still be \
                 tight. Widen the geometry or raise --max-zoom. rows per zoom: {rows_by_zoom:?}"
            );

            // (2) The writer's FACT, and the basis derived from it.
            let props = manifest_properties(&full);
            assert_eq!(
                props["feature_id_construction"], "anchor-hash",
                "{kind}: an id-less line/polygon keeps ONE anchor hash across every tile — the \
                 writer must record that: {props}"
            );
            assert_eq!(
                props["feature_id_scope"], "global",
                "{kind}: `auto` must PROVE the key from the construction, with no \
                 --feature-id-scope anywhere: {props}"
            );

            let (ok, report) = run_json(&full, &[]);
            assert!(
                ok,
                "{kind}: a healthy archive must exit 0; errors: {:#?}",
                errors_of(&report)
            );
            assert_eq!(errors_of(&report).len(), 0, "{kind}: {report}");
            assert_eq!(report["distinct_id_basis"], "decoded-ids", "{kind}");
            assert_eq!(report["feature_id_construction"], "anchor-hash", "{kind}");
            // Armed AND quiet: no deviation to report at all on healthy data.
            assert!(
                !notes_of(&report).iter().any(|n| n.contains("distinct-id")),
                "{kind}: the counts agree, so nothing should be said: {:#?}",
                notes_of(&report)
            );

            let truth = dir.path().join("truth.json");
            let (ok, _) = run_json(&full, &["--emit-fingerprint", truth.to_str().unwrap()]);
            assert!(ok, "{kind}: the good archive must validate to be truth");

            // --- the lossy rebuild: 40 % of the features gone ---------------
            // A contiguous INTERIOR block, so every extreme the fingerprint
            // records — bbox, `speed` range, `kind` cardinality — is still
            // carried by a surviving row. The only thing that changed is how
            // many features there are.
            let lossy_input = dir.path().join(format!("{kind}-lossy.parquet"));
            write_conus_spanning_parquet(&lossy_input, polygons, &|i| !(120..360).contains(&i));
            let lossy = dir.path().join(format!("{kind}-lossy"));
            build_zoomed(&lossy_input, &lossy, "2", "8", &["--content-fingerprint"]);
            let (_, lossy_rows) = assert_multi_tile(&lossy);

            let (ok, report) = run_json(&lossy, &["--expect-fingerprint", truth.to_str().unwrap()]);
            let errors = assert_failed_with(ok, &report, "FEATURE LOSS:");

            // (3) The distinct-id comparison is what fires, and it names the
            // numbers: 360 surviving features against a declared 600.
            assert!(
                errors.iter().any(|e| e.contains("FEATURE LOSS:")
                    && e.contains("distinct feature ids")
                    && e.contains("360")
                    && e.contains("600")
                    && e.contains("40.0%")),
                "{kind}: the distinct-id comparison must catch the loss and name it: {errors:#?}"
            );

            // (4) ⭐ AND THE ROW FLOOR DID NOT. This is the measurement the whole
            // change exists for: the surviving 360 features still replicate into
            // more rows at the fullest zoom than there were source features, so
            // the floor sees no shortfall at all.
            let lossy_fullest = lossy_rows.values().copied().max().unwrap_or(0);
            assert!(
                lossy_fullest >= 600,
                "{kind}: the row floor must be VACUOUS here for this test to be about \
                 anything — 40% of the features are gone and the fullest zoom still carries \
                 {lossy_fullest} rows against a declared 600: {lossy_rows:?}"
            );
            assert!(
                !errors.iter().any(|e| e.contains("fullest decoded zoom")),
                "{kind}: the row floor must be silent — if it fires, this fixture is not the \
                 loose case the basis fix is for: {errors:#?}"
            );

            // …and QUANTIFY the blind spot, so "the floor is loose here" is a
            // measurement rather than a claim. Each surviving feature still
            // contributes `fullest / 600` rows, so the floor cannot fire until
            // fewer than `600 / that` features remain. MEASURED on this fixture:
            // 2 106 rows at z8 for 600 features (3.51 rows each), so the floor
            // stays silent until ~71 % of the dataset is gone — which is the
            // 50–78 % band the reviewer measured, made executable.
            let rows_per_feature = fullest as f64 / 600.0;
            let floor_blind_spot = 100.0 * (1.0 - 1.0 / rows_per_feature);
            assert!(
                floor_blind_spot > 50.0,
                "{kind}: the row floor's blind spot is only {floor_blind_spot:.1}% \
                 ({rows_per_feature:.2} rows per feature at the fullest zoom) — this fixture no \
                 longer demonstrates the loose case, so the test would pass without proving \
                 the distinct-id check is load-bearing"
            );

            // The documented escape hatch still reaches the new error, and says
            // out loud that it did.
            let (ok, report) = run_json(
                &lossy,
                &[
                    "--expect-fingerprint",
                    truth.to_str().unwrap(),
                    "--allow-distinct-shortfall",
                ],
            );
            assert!(ok, "{kind}: downgraded: {:#?}", errors_of(&report));
            assert!(
                warnings_of(&report)
                    .iter()
                    .any(|w| w.contains("DOWNGRADED") && w.contains("FEATURE LOSS:")),
                "{kind}: a suppressed finding must not read as a clean run: {:#?}",
                warnings_of(&report)
            );

            // A SAMPLED run reports the same shortfall as sampling noise, never
            // as the error — the two wordings must stay unconfusable.
            let (ok, report) = run_json(
                &lossy,
                &[
                    "--expect-fingerprint",
                    truth.to_str().unwrap(),
                    "--sample",
                    "4",
                ],
            );
            assert!(ok, "{kind}: sampled: {:#?}", errors_of(&report));
            assert!(
                warnings_of(&report)
                    .iter()
                    .any(|w| w.contains("SAMPLING NOISE") && !w.contains("FEATURE LOSS:")),
                "{kind}: warnings: {:#?}",
                warnings_of(&report)
            );

            // …and `--feature-id-scope local` is still the working rollback: the
            // same lossy archive goes back to exit 0 with a note.
            let rolled_back = dir.path().join(format!("{kind}-lossy-local"));
            build_zoomed(
                &lossy_input,
                &rolled_back,
                "2",
                "8",
                &["--content-fingerprint", "--feature-id-scope", "local"],
            );
            let (ok, report) = run_json(
                &rolled_back,
                &["--expect-fingerprint", truth.to_str().unwrap()],
            );
            assert!(
                ok,
                "{kind}: --feature-id-scope local must still disarm the comparison it now \
                 arms by default: {:#?}",
                errors_of(&report)
            );
            assert_eq!(report["distinct_id_basis"], "source-features", "{kind}");
            assert_eq!(report["feature_id_construction"], "anchor-hash", "{kind}");
        }
    }

    /// The DISCRIMINATION, and the honest statement of the residual.
    ///
    /// A clipped TRIP (a duration LineString under trajectory clipping) is the
    /// one replicating shape the fix does NOT close: `segment_feature_id` mints
    /// a fresh id per clipped segment, and the writer cannot enumerate those ids
    /// before tiling, so it can prove nothing about them. It must therefore
    /// record `segment-hash`, decline the attestation, and fall back to the
    /// (loose) row floor — rather than quietly arming a comparison whose
    /// shortfall direction it has not proven.
    ///
    /// `--no-clip` on the same input moves the same features into the
    /// whole-feature anchor hash and the check arms itself, which is what makes
    /// this a property of the CONSTRUCTION rather than of the geometry kind.
    #[test]
    fn clipped_trips_record_segment_hash_and_decline_rather_than_overclaim() {
        use arrow::array::{ArrayRef, BinaryArray, Int64Array};
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("trips.parquet");
        let n = 200usize;
        let t0 = 1_700_000_000_000i64;
        let geom: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let x0 = -120.0 + (i % 20) as f64 * 2.2;
                let y0 = 26.0 + ((i / 20) % 10) as f64 * 1.7;
                wkb_linestring(&[(x0, y0), (x0 + 1.1, y0 + 0.8), (x0 + 2.0, y0 + 1.5)])
            })
            .collect();
        let cols: Vec<(&str, ArrayRef)> = vec![
            (
                "geometry",
                Arc::new(BinaryArray::from_opt_vec(
                    geom.iter().map(|g| Some(g.as_slice())).collect(),
                )) as ArrayRef,
            ),
            (
                "timestamp",
                Arc::new(Int64Array::from(
                    (0..n).map(|i| t0 + i as i64 * 720_000).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "end_timestamp",
                Arc::new(Int64Array::from(
                    (0..n)
                        .map(|i| t0 + i as i64 * 720_000 + 600_000)
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ];
        let schema = Arc::new(Schema::new(
            cols.iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            cols.iter().map(|(_, a)| a.clone()).collect(),
        )
        .unwrap();
        let file = std::fs::File::create(&input).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let trips = |out: &Path, extra: &[&str]| {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_stt-build"));
            cmd.arg("--input")
                .arg(&input)
                .arg("--output")
                .arg(out)
                .args(["--time-field", "timestamp", "--time-format", "unix-ms"])
                .args(["--end-time-field", "end_timestamp"])
                .args(["--min-zoom", "2", "--max-zoom", "8"])
                .args(["--temporal-bucket", "1h", "--content-fingerprint"])
                .args(extra);
            let output = cmd.output().expect("run stt-build");
            assert!(
                output.status.success(),
                "stt-build failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        };

        // Clipped (the default): per-segment ids ⇒ declined, and SAID so.
        let clipped = dir.path().join("trips-clipped");
        trips(&clipped, &[]);
        assert_multi_tile(&clipped);
        let props = manifest_properties(&clipped);
        assert_eq!(
            props["feature_id_construction"], "segment-hash",
            "a clipped trajectory mints one id per SEGMENT — the writer must record that \
             rather than claim a key it cannot prove: {props}"
        );
        assert_eq!(props["feature_id_scope"], "local", "{props}");

        let (ok, report) = run_json(&clipped, &[]);
        assert!(ok, "healthy trips: {:#?}", errors_of(&report));
        assert_eq!(report["distinct_id_basis"], "source-features");
        assert_eq!(report["feature_id_construction"], "segment-hash");
        // If anything IS said about the deviation, it must describe the
        // mechanism this archive used — the misdescription is the defect.
        for note in notes_of(&report)
            .iter()
            .filter(|n| n.contains("distinct-id"))
        {
            assert!(
                note.contains("segment-hash") && note.contains("feature_id_construction"),
                "the note must point at the recorded construction rather than tell the \
                 point-archive story on a trip archive: {note}"
            );
        }

        // --no-clip: the SAME features, placed whole, get the anchor hash — so
        // the decline above is a property of the construction, not of lines.
        let whole = dir.path().join("trips-noclip");
        trips(&whole, &["--no-clip"]);
        let props = manifest_properties(&whole);
        assert_eq!(props["feature_id_construction"], "anchor-hash", "{props}");
        assert_eq!(props["feature_id_scope"], "global", "{props}");
        let (ok, report) = run_json(&whole, &[]);
        assert!(ok, "--no-clip trips: {:#?}", errors_of(&report));
        assert_eq!(report["distinct_id_basis"], "decoded-ids");
    }

    /// ⚠️ THE NO-FALSE-POSITIVE SWEEP for the newly-armed path.
    ///
    /// Arming a comparison on line/polygon archives puts a new ERROR path in
    /// front of build shapes that were previously unreachable by it. The two
    /// that add rows or tiers to a healthy archive — a **summary tier** (whose
    /// derived cells the accumulator must keep skipping) and a **temporal LOD**
    /// pyramid (extra coarse-bucket tiers over the same features) — must stay
    /// clean, because in both the union of decoded ids is still exactly the
    /// source-feature set.
    ///
    /// Both dimensions are already part of the gate's proven no-false-positive
    /// matrix; this keeps them there now that the archive under them is armed.
    #[test]
    fn arming_the_check_does_not_false_positive_on_summary_or_lod_tiers() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("spanning.parquet");
        write_conus_spanning_parquet(&input, true, &|_| true);

        for (label, extra) in [
            ("summary", vec!["--summary-tier", "h3"]),
            ("lod", vec!["--temporal-lod", "4h"]),
            (
                "both",
                vec!["--summary-tier", "quadbin", "--temporal-lod", "4h,12h"],
            ),
        ] {
            let archive = dir.path().join(format!("armed-{label}"));
            let mut args = vec!["--content-fingerprint"];
            args.extend(extra);
            build_zoomed(&input, &archive, "2", "8", &args);
            assert_multi_tile(&archive);

            let (ok, report) = run_json(&archive, &[]);
            assert!(
                ok,
                "{label}: an armed but HEALTHY archive must stay clean — a derived summary \
                 cell or a coarse LOD tier is not feature loss: {:#?}",
                errors_of(&report)
            );
            assert_eq!(errors_of(&report).len(), 0, "{label}: {report}");
            assert_eq!(
                report["distinct_id_basis"], "decoded-ids",
                "{label}: the tiers must not disarm the check either — that would trade a \
                 false positive for a silent gap: {report}"
            );
            assert_eq!(report["feature_id_construction"], "anchor-hash", "{label}");
        }
    }

    /// ⚠️ THE SOUNDNESS GUARD on the newly-armed path, end to end.
    ///
    /// The anchor hash is `FNV(timestamp, lon, lat)`, so two polygons sharing a
    /// timestamp and a representative point map to ONE wire id. Such an archive
    /// decodes fewer distinct ids than it has features with **nothing missing** —
    /// which is precisely the false positive the whole basis machinery exists to
    /// prevent, now reachable through the path this change opened.
    ///
    /// The builder must prove distinctness before claiming the key: it records
    /// the construction honestly (`anchor-hash`, because that IS what it did) and
    /// stamps `feature_id_scope = local` beside it, which the validator resolves
    /// first. Exit 0, and a note rather than a FEATURE LOSS error.
    #[test]
    fn colliding_anchors_keep_the_new_arming_path_from_false_positiving() {
        use arrow::array::{ArrayRef, BinaryArray, Int64Array};
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("concentric.parquet");
        // 200 CONCENTRIC box pairs: each pair shares a centroid and a timestamp,
        // so the writer's anchor hash cannot tell the two apart.
        let pairs = 100usize;
        let t0 = 1_700_000_000_000i64;
        let mut geom: Vec<Vec<u8>> = Vec::new();
        let mut times: Vec<i64> = Vec::new();
        for p in 0..pairs {
            let cx = -120.0 + (p % 10) as f64 * 5.0;
            let cy = 28.0 + (p / 10) as f64 * 3.0;
            for half in [1.4f64, 0.7f64] {
                geom.push(wkb_polygon(&[
                    (cx - half, cy - half),
                    (cx + half, cy - half),
                    (cx + half, cy + half),
                    (cx - half, cy + half),
                    (cx - half, cy - half),
                ]));
                times.push(t0 + p as i64 * 720_000);
            }
        }
        let cols: Vec<(&str, ArrayRef)> = vec![
            (
                "geometry",
                Arc::new(BinaryArray::from_opt_vec(
                    geom.iter().map(|g| Some(g.as_slice())).collect(),
                )) as ArrayRef,
            ),
            ("timestamp", Arc::new(Int64Array::from(times)) as ArrayRef),
        ];
        let schema = Arc::new(Schema::new(
            cols.iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            cols.iter().map(|(_, a)| a.clone()).collect(),
        )
        .unwrap();
        let file = std::fs::File::create(&input).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let archive = dir.path().join("concentric");
        build_zoomed(&input, &archive, "2", "8", &["--content-fingerprint"]);
        assert_multi_tile(&archive);

        let props = manifest_properties(&archive);
        assert_eq!(
            props["feature_id_construction"], "anchor-hash",
            "the construction is recorded honestly — it IS what the writer did: {props}"
        );
        assert_eq!(
            props["feature_id_scope"], "local",
            "…but the PROOF declined on the colliding pair, so the comparison must \
             stay disarmed: {props}"
        );

        let (ok, report) = run_json(&archive, &[]);
        assert!(
            ok,
            "a healthy archive whose anchors collide must NOT be reported as feature \
             loss: {:#?}",
            errors_of(&report)
        );
        assert_eq!(errors_of(&report).len(), 0, "report: {report}");
        assert_eq!(report["distinct_id_basis"], "source-features");
        assert_eq!(report["feature_id_construction"], "anchor-hash");
    }
}
