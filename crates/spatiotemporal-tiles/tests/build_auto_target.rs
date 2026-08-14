//! `stt-build --target-size`: the budget handshake, at the BINARY boundary.
//!
//! Until this flag existed there was no target archive size anywhere in the
//! CLI. A publisher who needed a dataset to fit a budget hand-tuned ~10 coupled
//! knobs across repeated builds and closed the loop by eye. `--target-size`
//! implies `--auto encode`, runs the `stt-optimize` budget solver, and applies
//! the recipe it returns.
//!
//! # The rule that outranks the feature
//!
//! **Nothing automated may drop, sample, or aggregate a feature to hit a
//! number.** The library-side guarantees are structural (a `ChosenLever` has no
//! `lossy` field; the solver's feasible set excludes the quantization family;
//! `to_command`'s filter is inherited verbatim) and are tested there. This file
//! is the LAST LINE OF DEFENSE: it drives the real compiled binary and asserts
//! that whatever the library said, the effective build config carries no lossy
//! lever and the archive still contains every feature it was given.
//!
//! That is why the assertions here are deliberately end-of-pipe — process exit
//! status, the log the operator actually reads, the manifest that was actually
//! written — rather than a re-derivation of what the solver should have chosen.
//!
//! # What each test costs
//!
//! Every `--target-size` invocation runs the full analysis: the loader sample,
//! the advisor iteration, and the solver's re-measured frontier (~12 distortion
//! classes × up to three zstd levels of real encodes). That is seconds, not
//! milliseconds, so the fixture is deliberately small and the builds are
//! shared between assertions wherever one run can carry two.

// `CARGO_BIN_EXE_stt-build` only exists when the binary is built, and the
// binary's `required-features` is `build-cli`.
#![cfg(feature = "build-cli")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

const T0: i64 = 1_700_000_000_000;
const HOUR_MS: i64 = 3_600_000;
/// Rows in the fixture. Enough that the analyzer's sample clears its
/// measurement floor with room for four replicate blocks, small enough that a
/// full solve is seconds.
const ROWS: usize = 420;

fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = Vec::with_capacity(21);
    b.push(1);
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

/// splitmix64 — deterministic high-entropy values with no RNG, so the fixture
/// (and therefore every projection measured from it) is identical on every run
/// and every machine.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A spatially spread, multi-hour point fixture carrying a high-precision
/// `Float64` property.
///
/// The float column matters: it is what gives the solver a `--quantize-attrs-auto`
/// shadow price to publish, which is the lossy surface these tests check stays
/// a suggestion.
fn write_fixture(path: &Path) {
    let mut wkbs: Vec<Vec<u8>> = Vec::with_capacity(ROWS);
    let mut ts: Vec<i64> = Vec::with_capacity(ROWS);
    let mut mag: Vec<f64> = Vec::with_capacity(ROWS);
    let mut name: Vec<String> = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let jitter = (mix(i as u64) % 1_000_000) as f64 / 1e9;
        let lon = -122.6 + (i % 21) as f64 * 0.031 + jitter;
        let lat = 37.2 + (i / 21) as f64 * 0.027 + jitter;
        wkbs.push(wkb_point(lon, lat));
        ts.push(T0 + (i as i64 % 48) * HOUR_MS + (i as i64 % 11) * 60_000);
        mag.push(1.0 + (mix(i as u64 + 7) % 1_000_000) as f64 / 1e5);
        name.push(format!("station-{}", i % 6));
    }

    let cols: Vec<(&str, ArrayRef)> = vec![
        (
            "geometry",
            Arc::new(BinaryArray::from_iter_values(
                wkbs.iter().map(|v| v.as_slice()),
            )) as ArrayRef,
        ),
        ("timestamp", Arc::new(Int64Array::from(ts)) as ArrayRef),
        ("mag", Arc::new(Float64Array::from(mag)) as ArrayRef),
        (
            "name",
            Arc::new(StringArray::from(
                name.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ];
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(
        schema.clone(),
        cols.iter().map(|(_, a)| a.clone()).collect(),
    )
    .unwrap();
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

/// One `stt-build` run. Returns the raw output so tests can read the log the
/// operator reads (tracing writes to stderr).
fn run(input: &Path, out: &Path, extra: &[&str]) -> Output {
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
            "--workers",
            "2",
            "--name",
            "budget-fixture",
        ])
        .args(extra)
        .output()
        .expect("failed to spawn stt-build");
    output
}

fn log_of(output: &Output) -> String {
    let mut text = String::from_utf8_lossy(&output.stderr).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text
}

fn assert_ok(output: &Output, what: &str) -> String {
    let log = log_of(output);
    assert!(
        output.status.success(),
        "{what} failed ({}):\n{log}",
        output.status
    );
    log
}

/// Total bytes of the written dataset directory — manifest + directory pages +
/// packs, i.e. what a publisher actually uploads, and what `--target-size`
/// budgets.
fn dir_bytes(root: &Path) -> u64 {
    let mut total = 0;
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        total += if path.is_dir() {
            dir_bytes(&path)
        } else {
            fs::metadata(&path).unwrap().len()
        };
    }
    total
}

fn collect_files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        let mut entries: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap()
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

fn manifest_json(root: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap()
}

/// The one line every operator looks for. Extracted so a test that asserts on
/// the comparison cannot pass on a build that never printed one.
fn target_size_line(log: &str) -> &str {
    log.lines()
        .find(|l| l.contains("--target-size: BUILT"))
        .unwrap_or_else(|| panic!("no built-vs-target line in the log:\n{log}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn a_reachable_target_is_hit_and_the_comparison_is_logged() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.parquet");
    write_fixture(&input);

    // The unconstrained build, for scale.
    let plain_dir = dir.path().join("plain");
    let plain_log = assert_ok(&run(&input, &plain_dir, &[]), "unconstrained build");
    let plain_bytes = dir_bytes(&plain_dir);
    assert!(plain_bytes > 0);
    assert!(
        !plain_log.contains("--target-size"),
        "an unbudgeted build must not mention the budget at all"
    );

    // A budget the reversible levers can comfortably reach.
    let target = plain_bytes * 4;
    let budget_dir = dir.path().join("budgeted");
    let log = assert_ok(
        &run(&input, &budget_dir, &["--target-size", &target.to_string()]),
        "budgeted build",
    );

    // The handshake announced itself…
    assert!(
        log.contains("--target-size implies --auto encode"),
        "the implied mode must be logged:\n{log}"
    );
    // …the solver ran and said the budget fits…
    assert!(log.contains("FITS"), "expected a feasible verdict:\n{log}");
    // …and the build closed the loop the publisher used to close by eye.
    let line = target_size_line(&log);
    let built = dir_bytes(&budget_dir);
    assert!(
        line.contains(&built.to_string()) && line.contains(&target.to_string()),
        "the comparison must carry BOTH real numbers: {line}"
    );
    assert!(
        built <= target,
        "built {built} B over the {target} B budget: {line}"
    );

    // NOTHING WAS DROPPED to get there: the same feature count reached the
    // archive as in the unconstrained build.
    let features = |log: &str| {
        log.lines()
            .find_map(|l| l.split("Total features: ").nth(1))
            .map(|s| s.trim().to_string())
            .expect("every build logs its feature count")
    };
    assert_eq!(features(&plain_log), features(&log));
    assert_eq!(features(&log), ROWS.to_string());
}

#[test]
fn an_unreachable_target_reports_the_floor_and_still_builds_everything() {
    // THE NO-THINNING GUARD at the binary boundary. A budget no reversible
    // lever can reach must NOT start shedding features: the build proceeds at
    // the floor recipe, says so, and contains every feature it was given.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.parquet");
    write_fixture(&input);

    let out = dir.path().join("floor");
    // 1 KiB is smaller than the manifest alone.
    let log = assert_ok(
        &run(&input, &out, &["--target-size", "1KiB"]),
        "infeasible-budget build",
    );

    assert!(
        log.contains("DOES NOT FIT") || log.contains("is NOT reachable with reversible levers"),
        "an unreachable budget must say so:\n{log}"
    );
    assert!(
        log.contains("NOTHING has been dropped") || log.contains("NOTHING HAS BEEN DROPPED"),
        "…and must say what it did NOT do:\n{log}"
    );
    // Every feature is still there.
    assert!(
        log.contains(&format!("Total features: {ROWS}")),
        "features were lost under budget pressure:\n{log}"
    );
    // No thinning knob was engaged behind the user's back.
    assert_no_lossy_lever(&log, &out);
    // And it really did build: a readable archive, not an empty directory.
    let manifest = manifest_json(&out);
    assert!(
        manifest["packs"].as_array().map(|p| p.len()).unwrap_or(0) > 0,
        "no packs were written: {manifest}"
    );
    // The overshoot is reported, not hidden.
    assert!(target_size_line(&log).contains("OVER"));
}

#[test]
fn no_lossy_flag_reaches_the_build_and_shadow_prices_stay_suggestions() {
    // Budget pressure is exactly the condition under which a solver would be
    // tempted to reach for quantization. It may PRICE it — that is the shadow
    // price, and it is information — but it may never APPLY it.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.parquet");
    write_fixture(&input);

    let out = dir.path().join("pressured");
    let log = assert_ok(
        &run(&input, &out, &["--target-size", "8KiB"]),
        "pressured build",
    );

    assert_no_lossy_lever(&log, &out);

    // The lossy levers were PRICED and OFFERED, on the same loud warn-path
    // lossy advice has always taken — or the report said plainly that none
    // could be priced. Silence about the lossy family is not an option.
    let offered: Vec<&str> = log
        .lines()
        .filter(|l| l.contains("LOSSY shadow price"))
        .collect();
    assert!(
        !offered.is_empty() || log.contains("No lossy lever could be priced"),
        "the lossy family must be either offered or explicitly not priced:\n{log}"
    );
    for line in &offered {
        assert!(
            line.contains("suggested, not applied"),
            "a shadow price must be labelled as not applied: {line}"
        );
    }
}

/// No lossy lever reached the EFFECTIVE build config — checked from the log the
/// build itself prints about its encoder settings, and from the archive it
/// wrote.
fn assert_no_lossy_lever(log: &str, out: &Path) {
    for line in log
        .lines()
        .filter(|l| l.contains("Encoder settings ENABLED"))
    {
        assert!(
            !line.contains("quantize"),
            "a quantization lever reached the encoder: {line}"
        );
    }
    // The applied-levers summary is the auto-tuner's own account of what it did.
    for line in log.lines().filter(|l| l.contains("--auto encode applied")) {
        assert!(
            !line.contains("quantize") && !line.contains("max-features"),
            "the auto-tuner applied a lossy lever: {line}"
        );
    }
    // …and the built archive agrees: no capability that re-types a column into
    // a quantized form was declared.
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let capabilities = manifest["capabilities"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for cap in &capabilities {
        let name = cap.as_str().unwrap_or_default();
        assert!(
            !name.contains("quantiz"),
            "the archive declares the `{name}` capability, so a lossy lever was applied"
        );
    }
}

#[test]
fn explicit_flags_survive_budget_folding() {
    // The oldest `--auto` contract, under the newest pressure: a flag the user
    // typed always wins, however much the budget would like to move it.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.parquet");
    write_fixture(&input);

    let out = dir.path().join("explicit");
    let log = assert_ok(
        &run(
            &input,
            &out,
            &[
                "--target-size",
                "8KiB",
                // Pinned as a PAIR: the analyzer's recommended min-zoom for this
                // fixture sits above 6, and an explicit max-zoom below an auto
                // min-zoom would tile nothing at all.
                "--min-zoom",
                "0",
                "--max-zoom",
                "6",
                "--temporal-bucket",
                "1h",
                "--temporal-lod",
                "6h",
                "--zstd-level",
                "7",
            ],
        ),
        "explicit-flag build",
    );

    let manifest = manifest_json(&out);
    let meta = &manifest["metadata"];
    assert_eq!(
        meta["max_zoom"].as_u64(),
        Some(6),
        "the budget overrode an explicit --max-zoom: {meta}"
    );
    assert_eq!(
        meta["temporal_bucket_ms"].as_u64(),
        Some(HOUR_MS as u64),
        "the budget overrode an explicit --temporal-bucket: {meta}"
    );
    // The user's tier survived, and the solver's (if it wanted one) did not
    // replace it.
    let lod = meta["temporal_lod"].as_array().cloned().unwrap_or_default();
    assert_eq!(lod.len(), 1, "expected exactly the user's one tier: {meta}");
    assert_eq!(lod[0]["bucket_ms"].as_u64(), Some(6 * HOUR_MS as u64));
    // The log says WHY, so an operator can see the budget yielded rather than
    // wonder whether it was ignored.
    assert!(
        log.contains("explicit flag wins") || log.contains("explicit --zstd-level/--publish wins"),
        "the yield must be explained:\n{log}"
    );
    assert_no_lossy_lever(&log, &out);
}

#[test]
fn two_identical_target_size_builds_are_byte_identical() {
    // THE determinism rule for a solver change. Same input ⇒ same
    // recommendation ⇒ byte-identical build. Packs are content-addressed, so a
    // wobble anywhere in the solver re-uploads and re-invalidates a whole
    // published fleet — this is the CLI-seam instance of the standing rule.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.parquet");
    write_fixture(&input);

    let first = dir.path().join("a");
    let second = dir.path().join("b");
    let flags = ["--target-size", "48KiB"];
    assert_ok(&run(&input, &first, &flags), "first budgeted build");
    assert_ok(&run(&input, &second, &flags), "second budgeted build");

    let a = collect_files(&first);
    let b = collect_files(&second);
    assert!(
        a.keys().any(|k| k.starts_with("packs/")) && a.contains_key("manifest.json"),
        "the comparison must run over a real archive: {:?}",
        a.keys().collect::<Vec<_>>()
    );

    let mut diffs: Vec<String> = Vec::new();
    for key in a.keys() {
        if !b.contains_key(key) {
            diffs.push(format!("missing from the second build: {key}"));
        }
    }
    for key in b.keys() {
        if !a.contains_key(key) {
            diffs.push(format!("extra in the second build: {key}"));
        }
    }
    for (key, bytes_a) in &a {
        if let Some(bytes_b) = b.get(key) {
            if bytes_a != bytes_b {
                diffs.push(format!(
                    "bytes differ: {key} ({} B vs {} B)",
                    bytes_a.len(),
                    bytes_b.len()
                ));
            }
        }
    }
    assert!(
        diffs.is_empty(),
        "two identical --target-size builds diverged:\n{}",
        diffs.join("\n")
    );
}

#[test]
fn target_size_needs_a_geoparquet_input_and_says_so() {
    // `--auto` reads a FILE — it samples and trial-encodes the source to
    // measure anything at all — and `--target-size` inherits the limitation.
    // Users will hit this, so the message has to name the flag they typed and
    // the way out, not just the flag it implies.
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("nope");
    let output = Command::new(stt_build_bin())
        .args([
            "--postgres",
            "postgresql://localhost/does-not-exist",
            "--table",
            "public.events",
        ])
        .arg("--output")
        .arg(&out)
        .args(["--time-field", "timestamp", "--target-size", "10MiB"])
        .output()
        .expect("failed to spawn stt-build");

    assert!(
        !output.status.success(),
        "a database source with --target-size must fail loudly"
    );
    let log = log_of(&output);
    assert!(
        log.contains("--target-size") && log.contains("GeoParquet"),
        "the error must name the flag and the requirement:\n{log}"
    );
}
