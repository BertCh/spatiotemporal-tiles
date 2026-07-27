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
        format_version: writer.format_version(),
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
        format_version: writer.format_version(),
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
        format_version: writer.format_version(),
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
        format_version: writer.format_version(),
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
