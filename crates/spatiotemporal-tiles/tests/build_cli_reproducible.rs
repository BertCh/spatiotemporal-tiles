//! Byte-reproducibility at the **CLI seam**: two runs of the `stt-build`
//! BINARY over the same input file produce byte-identical output directories.
//!
//! # Why this exists, given the lane already has three members
//!
//! The determinism lane covers the layers *below* the binary:
//!
//! * per-tile encode — `stt-core`'s `reproducible_build.rs`
//!   (`same_tile_encodes_byte_identically`, 200 repetitions),
//! * whole-dataset assembly, order-independent —
//!   `v2_dataset_rebuild_is_byte_identical_including_schemas`,
//! * the library pipeline — `stt-build`'s `reproducible_pipeline.rs`
//!   (`generate_tiles_streaming` + `PackWriter` run twice over one parsed
//!   feature stream), and
//! * the byte pin itself — `stt-core`'s `v2_golden.rs`.
//!
//! Every one of them starts from an already-parsed feature list or an
//! already-built layer. **Nothing double-builds through the binary**, so the
//! whole front half of a real build — arg parsing and defaulting, the
//! GeoParquet reader, the parallel parse, bounds/metadata derivation, the
//! `--blob-ordering auto` decision, output-path handling — has never been under
//! a byte-identity assertion. That front half is exactly where the 2026-07-28
//! columnar-ingest refactor worked, and its stated acceptance standard was
//! literally "byte-identical archives via `diff -r`". This file is that
//! standard, encoded.
//!
//! # What is asserted
//!
//! Recursive byte equality of the two output trees: the same relative-path SET
//! (no extra, no missing object), and identical bytes for every file —
//! `manifest.json`, every `index/*.sttd` page, every `packs/*.sttp`. That is
//! `diff -r` with a better failure message.
//!
//! # The negative guard
//!
//! `comparison_reports_a_single_flipped_byte` flips one byte of one pack in a
//! copy of a real archive and asserts the comparator REPORTS it. Without it a
//! refactor could quietly turn `compare_trees` into a tautology (compare a tree
//! against itself, skip the payloads, swallow a missing file) and the positive
//! test would stay green forever. The comparator therefore returns its
//! findings instead of asserting internally, and both tests read the same
//! return value.
//!
//! # Fixture honesty
//!
//! The implementation plan (P0-7) says to run over "a committed fixture parquet
//! from `crates/stt-build/tests/fixtures/`". **No such file exists** — that
//! directory holds three hand-authored GeoJSON antimeridian fixtures and
//! nothing else, and `git ls-files '*.parquet'` is empty repo-wide. So the
//! input is synthesised here, deterministically, from a hand-written row table
//! and written once into the temp dir; BOTH builds then read that one file.
//! Nothing about the property under test depends on the input being committed:
//! the assertion is that two reads of one identical file produce identical
//! output.
//!
//! # The rule this file binds forward
//!
//! Determinism is not a property of one encoder — it is the format's economic
//! contract. Content-addressed packs mean a rebuild that moves bytes for no
//! semantic reason re-uploads and re-invalidates the whole fleet: the lane's
//! two historical catches (Arrow 54 serialising schema metadata in per-process
//! `HashMap` order; the FNV id migration) each cost a 29.3 GiB / 1,324-object
//! re-upload. So the standing rule, of which this file is the CLI-seam
//! instance:
//!
//! > **Every solver or encoder change lands with a byte-identical re-run
//! > test.** No RNG, no wall clock, no arrival-order or `HashMap`-iteration
//! > dependence in any output-affecting path; ties broken by a total key.
//!
//! Concretely that binds the sampler (deterministic sampling is contractual),
//! the workload simulator (otherwise pack names churn), any budget sweep, and
//! any offline policy replayer — each owes a test in this lane's shape:
//! run twice, compare bytes, and keep a negative guard on the comparator.
//!
//! The lane's members, for anyone adding to it:
//! `stt-core` `reproducible_build.rs` (per-tile + whole-dataset),
//! `stt-core` `v2_golden.rs` (the byte pin),
//! `stt-build` `reproducible_pipeline.rs` (library pipeline + comparator
//! guard), and this file (the binary).

// `CARGO_BIN_EXE_stt-build` only exists when the binary is built, and the
// binary's `required-features` is `build-cli` — so that, not `cli`, is the
// exact precondition for this file to compile. `cli` implies `build-cli`, so
// this still rides the `rust-all-features` lane as P0-7 asks; gating on the
// narrower `cli` would have EXCLUDED it from the default `rust` lane, where
// the rest of the determinism lane already runs.
#![cfg(feature = "build-cli")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Float64Array, Int64Array, Int64Builder, ListBuilder,
    StringArray,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use stt_core::Manifest;

// ---------------------------------------------------------------------------
// Fixture synthesis
// ---------------------------------------------------------------------------

/// Little-endian WKB point.
fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = Vec::with_capacity(21);
    b.push(1); // little-endian
    b.extend_from_slice(&1u32.to_le_bytes()); // Point
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

/// Little-endian WKB linestring.
fn wkb_linestring(coords: &[(f64, f64)]) -> Vec<u8> {
    let mut b = Vec::with_capacity(9 + coords.len() * 16);
    b.push(1);
    b.extend_from_slice(&2u32.to_le_bytes()); // LineString
    b.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for (lon, lat) in coords {
        b.extend_from_slice(&lon.to_le_bytes());
        b.extend_from_slice(&lat.to_le_bytes());
    }
    b
}

/// One input row. Deliberately mixed so the CLI's front half has real work:
/// points AND timed linestrings (which carry the per-vertex `List<Int64>` /
/// `List<Float64>` columns the reader auto-detects by name), a null geometry
/// (the skip path), and four property columns of four different types (the
/// columnar ingest path, including a low-cardinality string that should
/// dictionary-encode).
struct Row {
    wkb: Option<Vec<u8>>,
    ts: i64,
    end: i64,
    mag: f64,
    cnt: i64,
    name: String,
    flag: bool,
    vts: Option<Vec<i64>>,
    vvs: Option<Vec<Option<f64>>>,
}

const T0: i64 = 1_700_000_000_000; // fixed epoch-ms; no wall clock anywhere
const HOUR_MS: i64 = 3_600_000;

/// A deterministic, spatially spread, multi-hour row table.
///
/// Spread matters: the tiler buckets by (zoom, tile, time-bucket) through a
/// `HashMap` on a rayon pool, so a fixture confined to one tile and one bucket
/// would have no emission order to normalise away and the test would pass
/// vacuously. 600 points across a 30x20 lon/lat lattice over 24 hours plus 24
/// tracks gives thousands of distinct tiles across zooms 0..=8.
fn fixture_rows() -> Vec<Row> {
    let mut rows = Vec::with_capacity(640);
    let names = ["alpha", "bravo", "charlie", "delta"];

    for i in 0..600i64 {
        let gx = i % 30;
        let gy = i / 30;
        let lon = -170.0 + (gx as f64) * 11.3;
        let lat = -60.0 + (gy as f64) * 6.1;
        let ts = T0 + (i % 24) * HOUR_MS + (i % 7) * 60_000;
        rows.push(Row {
            wkb: Some(wkb_point(lon, lat)),
            ts,
            end: ts + 90_000,
            mag: 0.5 + (i % 47) as f64 * 0.11,
            cnt: i % 13,
            name: names[(i % 4) as usize].to_string(),
            flag: i % 3 == 0,
            vts: None,
            vvs: None,
        });
    }

    for k in 0..24i64 {
        let base_lon = -150.0 + (k as f64) * 12.0;
        let base_lat = -20.0 + (k as f64) * 2.5;
        let coords: Vec<(f64, f64)> = (0..5)
            .map(|v| (base_lon + v as f64 * 1.7, base_lat + v as f64 * 0.9))
            .collect();
        let ts = T0 + k * HOUR_MS;
        rows.push(Row {
            wkb: Some(wkb_linestring(&coords)),
            ts,
            end: ts + 4 * 60_000,
            mag: 3.0 + k as f64 * 0.25,
            cnt: 100 + k,
            name: names[(k % 4) as usize].to_string(),
            flag: k % 2 == 0,
            vts: Some((0..5).map(|v| ts + v * 60_000).collect()),
            vvs: Some(
                (0..5)
                    .map(|v| {
                        if v == 3 {
                            None
                        } else {
                            Some(1.0 + v as f64 * 0.5)
                        }
                    })
                    .collect(),
            ),
        });
    }

    // The null-geometry skip path.
    rows.push(Row {
        wkb: None,
        ts: T0,
        end: T0 + 1000,
        mag: 0.0,
        cnt: 0,
        name: "null-geom".to_string(),
        flag: false,
        vts: None,
        vvs: None,
    });

    rows
}

/// Materialise the rows as a single-row-group GeoParquet file. Geometry is a
/// plain WKB `Binary` column named `geometry`, which the reader's name
/// heuristic picks up without any `geo` file metadata.
fn write_input_parquet(rows: &[Row], path: &Path) {
    let geom: Vec<Option<&[u8]>> = rows.iter().map(|r| r.wkb.as_deref()).collect();
    let geom_arr = Arc::new(BinaryArray::from_opt_vec(geom)) as ArrayRef;
    let ts_arr = Arc::new(Int64Array::from(
        rows.iter().map(|r| r.ts).collect::<Vec<_>>(),
    )) as ArrayRef;
    let end_arr = Arc::new(Int64Array::from(
        rows.iter().map(|r| r.end).collect::<Vec<_>>(),
    )) as ArrayRef;
    let mag_arr = Arc::new(Float64Array::from(
        rows.iter().map(|r| r.mag).collect::<Vec<_>>(),
    )) as ArrayRef;
    let cnt_arr = Arc::new(Int64Array::from(
        rows.iter().map(|r| r.cnt).collect::<Vec<_>>(),
    )) as ArrayRef;
    let name_arr = Arc::new(StringArray::from(
        rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
    )) as ArrayRef;
    let flag_arr = Arc::new(BooleanArray::from(
        rows.iter().map(|r| r.flag).collect::<Vec<_>>(),
    )) as ArrayRef;

    let mut vts_b = ListBuilder::new(Int64Builder::new());
    for r in rows {
        match &r.vts {
            Some(v) => {
                for &x in v {
                    vts_b.values().append_value(x);
                }
                vts_b.append(true);
            }
            None => vts_b.append(false),
        }
    }
    let vts_arr = Arc::new(vts_b.finish()) as ArrayRef;

    let mut vvs_b = ListBuilder::new(arrow::array::Float64Builder::new());
    for r in rows {
        match &r.vvs {
            Some(v) => {
                for x in v {
                    match x {
                        Some(f) => vvs_b.values().append_value(*f),
                        None => vvs_b.values().append_null(),
                    }
                }
                vvs_b.append(true);
            }
            None => vvs_b.append(false),
        }
    }
    let vvs_arr = Arc::new(vvs_b.finish()) as ArrayRef;

    let cols: Vec<(&str, ArrayRef)> = vec![
        ("geometry", geom_arr),
        ("timestamp", ts_arr),
        ("end_time", end_arr),
        ("mag", mag_arr),
        ("cnt", cnt_arr),
        ("name", name_arr),
        ("flag", flag_arr),
        ("vertex_timestamps", vts_arr),
        ("vertex_values", vvs_arr),
    ];
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = cols.iter().map(|(_, a)| a.clone()).collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();

    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// Driving the binary
// ---------------------------------------------------------------------------

fn stt_build_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stt-build"))
}

/// Run the real `stt-build` binary into `out`.
///
/// Flags are pinned rather than left to defaults where a default would make the
/// run vacuous or slow: `--time-format unix-ms` (the column is Int64, and the
/// `iso8601` default is only consulted for integer columns), `--max-zoom 8`
/// (14 would be gratuitous for a 625-row fixture), and
/// `--paged-directory-min-entries 1` to force the PAGED directory shape — the
/// adaptive writer would otherwise emit one whole-load frame at this size, and
/// the paged shape is what real archives ship, so it is the shape whose
/// multi-object determinism is worth pinning. `--workers 4` keeps the rayon
/// pool genuinely parallel, which is the emission-order noise being normalised.
/// `--salvage-invalid-geometry` is the one non-default: input parsing is STRICT
/// by default, so the fixture's deliberate null-geometry row would abort the
/// run instead of exercising the skip path (a skip shifts the parsed-feature
/// indices relative to the row indices — exactly the kind of off-by-one a
/// reader refactor can make order-dependent).
fn run_build(input: &Path, out: &Path) {
    let output = Command::new(stt_build_bin())
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(out)
        .args([
            "--time-field",
            "timestamp",
            "--end-time-field",
            "end_time",
            "--time-format",
            "unix-ms",
            "--min-zoom",
            "0",
            "--max-zoom",
            "8",
            "--temporal-bucket",
            "1h",
            "--workers",
            "4",
            "--paged-directory-min-entries",
            "1",
            "--page-entries",
            "64",
            "--name",
            "cli-repro",
            "--salvage-invalid-geometry",
        ])
        .output()
        .expect("failed to spawn stt-build");

    assert!(
        output.status.success(),
        "stt-build failed ({}):\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

// ---------------------------------------------------------------------------
// Recursive tree comparison
// ---------------------------------------------------------------------------

/// Every file under `root`, keyed by its `/`-joined relative path.
fn collect_files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        let mut entries: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for path in entries {
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

/// `diff -r` over two archive trees. Returns one human-readable line per
/// difference and an EMPTY vec when the trees are byte-identical.
///
/// This RETURNS rather than asserts on purpose: the negative guard below feeds
/// it a deliberately corrupted tree and asserts the findings are non-empty, so
/// the comparator cannot silently decay into "always equal".
fn compare_trees(a: &Path, b: &Path) -> Vec<String> {
    let files_a = collect_files(a);
    let files_b = collect_files(b);
    let mut diffs = Vec::new();

    for key in files_a.keys() {
        if !files_b.contains_key(key) {
            diffs.push(format!("missing from second build: {key}"));
        }
    }
    for key in files_b.keys() {
        if !files_a.contains_key(key) {
            diffs.push(format!("extra in second build: {key}"));
        }
    }
    for (key, bytes_a) in &files_a {
        if let Some(bytes_b) = files_b.get(key) {
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

fn copy_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let path = entry.unwrap().path();
        let target = dst.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_tree(&path, &target);
        } else {
            fs::copy(&path, &target).unwrap();
        }
    }
}

/// Sanity floor: the archive is a real multi-object packed dataset, so an
/// `{} == {}` pass is impossible.
fn assert_archive_is_substantial(root: &Path, files: &BTreeMap<String, Vec<u8>>) {
    assert!(
        root.join("manifest.json").is_file(),
        "no manifest.json in {}",
        root.display()
    );
    let inventory = || files.keys().cloned().collect::<Vec<_>>().join(", ");
    let packs = files.keys().filter(|k| k.starts_with("packs/")).count();
    let index = files.keys().filter(|k| k.starts_with("index/")).count();
    assert!(
        packs >= 1,
        "expected at least one pack object, got {packs} — tree: {}",
        inventory()
    );
    assert!(
        index >= 1,
        "expected a directory object, got {index} — tree: {}",
        inventory()
    );
    assert!(
        files.len() >= 3,
        "expected a multi-object archive, got {} files — tree: {}",
        files.len(),
        inventory()
    );

    // The paged directory carries its root + leaf pages INSIDE one `.sttd`
    // object, so object COUNT cannot prove paging; the manifest's declared
    // shape can. Assert it so a future default flip that quietly drops this
    // run back to the whole-load frame is loud rather than silent — and use
    // the leaf-page count as the "the fixture really did produce thousands of
    // tiles" floor, since `--page-entries 64` makes pages a direct proxy.
    let manifest = Manifest::from_json_bytes(&files["manifest.json"]).expect("manifest parses");
    assert!(
        manifest.directory.is_paged(),
        "expected the paged directory shape, got layout {:?}",
        manifest.directory.layout
    );
    let pages = manifest.directory.page_count.unwrap_or(0);
    assert!(
        pages >= 8,
        "expected a fixture big enough to fill many 64-entry leaf pages \
         (>=8 pages, i.e. >~450 tiles), got {pages}"
    );
    assert!(
        manifest.directory.page_hashes.as_ref().map_or(0, Vec::len) as u64 == pages,
        "manifest must carry one leaf hash per page"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// **The CLI seam.** Two runs of the binary over one input file produce
/// byte-identical output trees — arg parsing, the GeoParquet reader, the
/// parallel parse, bounds/metadata derivation, `--blob-ordering auto`, encode,
/// pack cutting and directory paging included.
#[test]
fn stt_build_cli_double_build_is_byte_identical() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("input.parquet");
    write_input_parquet(&fixture_rows(), &input);

    let out_a = tmp.path().join("a");
    let out_b = tmp.path().join("b");
    run_build(&input, &out_a);
    run_build(&input, &out_b);

    let files_a = collect_files(&out_a);
    assert_archive_is_substantial(&out_a, &files_a);

    let diffs = compare_trees(&out_a, &out_b);
    assert!(
        diffs.is_empty(),
        "two stt-build runs over the same input produced different archives \
         ({} difference(s)):\n  {}",
        diffs.len(),
        diffs.join("\n  ")
    );

    // Spell the headline object out separately: a manifest drift is the one
    // that invalidates every client, so name it rather than leaving it inside
    // the generic list.
    assert_eq!(
        fs::read(out_a.join("manifest.json")).unwrap(),
        fs::read(out_b.join("manifest.json")).unwrap(),
        "manifest.json bytes must not depend on which run produced them"
    );
}

/// **Negative guard for the comparator.** One flipped byte in one pack must be
/// REPORTED. Without this, `compare_trees` could rot into a tautology and the
/// positive test above would stay green while asserting nothing.
///
/// Runs the binary once and corrupts a copy, so the guard costs one build, not
/// two.
#[test]
fn comparison_reports_a_single_flipped_byte() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("input.parquet");
    write_input_parquet(&fixture_rows(), &input);

    let good = tmp.path().join("good");
    run_build(&input, &good);
    let corrupt = tmp.path().join("corrupt");
    copy_tree(&good, &corrupt);

    // Baseline: a faithful copy compares clean. (If this fires, the guard
    // below would be measuring the copy, not the flip.)
    assert!(
        compare_trees(&good, &corrupt).is_empty(),
        "a byte-for-byte copy must compare clean before the flip"
    );

    // Flip one bit of one byte of the first pack object.
    let files = collect_files(&good);
    let pack_rel = files
        .keys()
        .find(|k| k.starts_with("packs/"))
        .expect("archive must contain a pack object")
        .clone();
    let victim = corrupt.join(&pack_rel);
    let mut bytes = fs::read(&victim).unwrap();
    assert!(!bytes.is_empty(), "pack {pack_rel} is empty");
    let offset = bytes.len() / 2;
    bytes[offset] ^= 0x01;
    fs::write(&victim, &bytes).unwrap();

    let diffs = compare_trees(&good, &corrupt);
    assert_eq!(
        diffs.len(),
        1,
        "one flipped byte must produce exactly one finding, got: {diffs:?}"
    );
    assert!(
        diffs[0].contains(&pack_rel) && diffs[0].contains("bytes differ"),
        "the finding must name the corrupted pack, got: {}",
        diffs[0]
    );
    assert!(
        diffs[0].contains(&format!("offset {offset}")),
        "the finding must locate the flip at offset {offset}, got: {}",
        diffs[0]
    );

    // A missing object and an extra object are the other two ways a tree can
    // differ; assert the comparator sees those too, so "recursive byte
    // equality" means the file SET as well as the payloads.
    fs::remove_file(&victim).unwrap();
    let after_delete = compare_trees(&good, &corrupt);
    assert!(
        after_delete
            .iter()
            .any(|d| d.contains("missing from second build") && d.contains(&pack_rel)),
        "deleting a pack must be reported as missing, got: {after_delete:?}"
    );

    fs::write(corrupt.join("packs").join("intruder.sttp"), b"not a pack").unwrap();
    let after_extra = compare_trees(&good, &corrupt);
    assert!(
        after_extra
            .iter()
            .any(|d| d.contains("extra in second build") && d.contains("intruder.sttp")),
        "an unexpected object must be reported as extra, got: {after_extra:?}"
    );
}
