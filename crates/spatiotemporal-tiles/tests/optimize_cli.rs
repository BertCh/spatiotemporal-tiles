//! End-to-end tests for the `stt-optimize` binary.
//!
//! The pattern is `validate_cli.rs`'s: synthesise a tiny packed dataset in a
//! temp dir through `stt_core`'s `PackWriter` (the same writer the offline
//! `stt-build` uses), then run the COMPILED binary (`CARGO_BIN_EXE_*`) over it
//! and assert on the parsed JSON and the process exit code. Nothing is committed
//! to the repo; every fixture is built at test time.
//!
//! What this file exists to pin (MO-7):
//!
//! * **`doctor --strict` keys off SEVERITY ALONE.** MO-7 re-ranks findings
//!   within a severity tier by projected bytes. That re-rank must be invisible
//!   to CI, so these tests derive the expected exit code from the report's
//!   severities and assert the binary agrees — on a dirty archive AND on a clean
//!   one. If a future ranking change could flip `--strict`, this fails.
//! * **The byte-ranked order and the additive fields** survive the CLI's JSON
//!   serialization: `projected_bytes` on findings, `joint` on the report.
//! * **Determinism**: two identical `doctor` invocations emit byte-identical
//!   stdout. The measured-shrink pass re-encodes columns, so this is a real
//!   determinism claim, not a formality.
//!
//! What it exists to pin (MO-6) — the second half of the file:
//!
//! * **The binary runs the ITERATED advisor pass.** MO-6 built
//!   `run_iterative` / `recommend_for`, and the binary went on calling the
//!   round-0 `run_all` — so the mechanism existed in the library and reached
//!   nobody. These tests assert the *composed measurement* comes out of the
//!   process, on both `recommend` and `analyze`. That figure is producible only
//!   by the iterated path, so it is a true witness for the wiring rather than a
//!   restatement of it.
//! * **CLI-seam determinism.** `crates/stt-optimize/tests/recommend_determinism.rs`
//!   pins the LIBRARY call; this file pins the PROCESS, which is the thing
//!   `stt-build --auto` scripts and the MCP `recommend_build` tool actually
//!   invoke. It has to live here and not there: `CARGO_BIN_EXE_stt-optimize`
//!   only exists for test targets in the package that declares the binary.
//! * **The no-thinning guard, at the binary boundary.** `to_command`'s unit
//!   tests (`lossy_advice_never_joins_to_command`,
//!   `suggestion_only_advice_never_joins_to_command`) pin the filter; these pin
//!   that the filter is still the thing standing between the emitted command and
//!   a lossy flag AFTER a whole extra measurement pass has had the chance to
//!   change its mind. Iteration refines numbers, never admissibility.

// The stt-optimize binary (and CARGO_BIN_EXE_stt-optimize) only exists when the
// `optimize-cli` feature is on; compile the suite out otherwise.
#![cfg(feature = "optimize-cli")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{BinaryArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use stt_core::arrow_tile::{
    encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn, PropertyColumn,
};
use stt_core::metadata::Metadata;
use stt_core::{BlobOrdering, PackWriter, TileId};

/// splitmix64 — deterministic full-entropy values, so a "random" column is
/// genuinely incompressible without any RNG (and without any run-to-run drift).
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic uniform f64 in `[0, 1)`.
fn rand01(x: u64) -> f64 {
    (mix(x) >> 11) as f64 / (1u64 << 53) as f64
}

/// A point layer with a full-entropy `magnitude` Float64 property — the column
/// the `raw-f64-column` rule is built to find.
fn points_layer(seed: u64, rows: usize) -> ColumnarLayer {
    ColumnarLayer {
        polygon_parts: None,
        name: "default".to_string(),
        feature_ids: (0..rows as u64).map(|i| seed * 1_000_000 + i).collect(),
        start_times: vec![0; rows],
        end_times: vec![100; rows],
        geometry: GeometryColumn::Point(
            (0..rows)
                .map(|i| {
                    [
                        -73.9 + (i % 50) as f64 * 0.001,
                        45.4 + (i / 50) as f64 * 0.001,
                    ]
                })
                .collect(),
        ),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "magnitude".to_string(),
                PropertyColumn::Numeric(
                    (0..rows)
                        .map(|i| Some(rand01(seed * 31 + i as u64) * 10.0))
                        .collect(),
                ),
            ),
            (
                "kind".to_string(),
                PropertyColumn::Categorical(
                    (0..rows)
                        .map(|i| Some(["bike", "ferry"][i % 2].to_string()))
                        .collect(),
                ),
            ),
        ],
    }
}

/// Write a packed dataset of `tiles` tiles into `out`.
///
/// `quantize_magnitude` is the whole difference between the two fixtures these
/// tests use: unquantized, the doctor raises a Warning-or-worse finding and
/// `--strict` must fail; quantized, it does not and `--strict` must pass.
fn build_dataset(out: &Path, tiles: usize, rows: usize, quantize_magnitude: bool) {
    let mut w = PackWriter::create(out, BlobOrdering::Auto, 64 * 1024).unwrap();
    let cfg = EncoderConfig {
        quantize_attrs: if quantize_magnitude {
            [("magnitude".to_string(), 0.01)].into_iter().collect()
        } else {
            Default::default()
        },
        format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
        template_collector: Some(w.template_collector()),
        ..Default::default()
    };
    let bucket = 3_600_000i64;
    for k in 0..tiles {
        let payload = encode_tile_with(&[points_layer(k as u64, rows)], &cfg).unwrap();
        let t0 = (k as i64) * bucket;
        w.add_tile_full(
            &TileId::new(10, k as u32, 0, t0 as u64),
            t0,
            t0 + bucket - 1,
            Some(t0),
            rows as u32,
            Some(bucket as u64),
            &payload,
        )
        .unwrap();
    }
    let meta = Metadata::new("optimize-cli-fixture")
        .with_temporal_bucket_ms(bucket as u64)
        .with_zoom_levels(5, 10);
    w.finalize(&meta).unwrap();
}

/// Run the compiled binary with `args`; return (exit status success, stdout,
/// stderr).
fn run(args: &[&str]) -> (bool, String, String) {
    let bin = env!("CARGO_BIN_EXE_stt-optimize");
    let out = Command::new(bin)
        .args(args)
        .output()
        .expect("failed to run stt-optimize");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// `doctor --format json` over an archive, parsed.
fn doctor_json(archive: &Path, extra: &[&str]) -> (bool, serde_json::Value, String) {
    let archive = archive.to_string_lossy().into_owned();
    let mut args = vec!["doctor", "--archive", archive.as_str(), "--format", "json"];
    args.extend_from_slice(extra);
    let (ok, stdout, stderr) = run(&args);
    let report: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("doctor must emit valid JSON ({e}); stdout was:\n{stdout}"));
    (ok, report, stderr)
}

/// Severity ranks, most severe first — the order the report sorts by and the
/// only thing `--strict` looks at.
fn severity_rank(s: &str) -> u8 {
    match s {
        "critical" => 0,
        "warning" => 1,
        "info" => 2,
        other => panic!("unknown severity {other}"),
    }
}

/// Warning-or-worse findings — the `--strict` gate's own predicate, recomputed
/// here from the JSON so the test derives the expected exit code rather than
/// hard-coding it.
fn gate_count(report: &serde_json::Value) -> usize {
    report["findings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| severity_rank(f["severity"].as_str().unwrap()) <= severity_rank("warning"))
        .count()
}

fn fixture(dir: &tempfile::TempDir, name: &str, quantized: bool) -> PathBuf {
    let path = dir.path().join(name);
    build_dataset(&path, 3, 2_000, quantized);
    path
}

/// The `--strict` CI gate is a pure function of the findings' SEVERITIES.
///
/// MO-7 re-ranks findings inside a severity tier by projected bytes; this is the
/// test that says the re-rank cannot reach CI. Both directions are asserted: a
/// dirty archive fails, a clean one passes, and in each case the exit code is
/// derived from the report rather than assumed.
#[test]
fn strict_gate_keys_off_severity_alone() {
    let dir = tempfile::tempdir().unwrap();

    // Dirty: raw Float64 `magnitude` ⇒ at least one Warning-or-worse finding.
    let dirty = fixture(&dir, "dirty", false);
    let (plain_ok, report, _) = doctor_json(&dirty, &[]);
    assert!(plain_ok, "doctor without --strict always exits 0");
    let bad = gate_count(&report);
    assert!(
        bad > 0,
        "the unquantized fixture must raise a Warning-or-worse finding: {report}"
    );

    let (strict_ok, strict_report, stderr) = doctor_json(&dirty, &["--strict"]);
    assert!(
        !strict_ok,
        "--strict must exit non-zero on {bad} finding(s)"
    );
    assert!(
        stderr.contains("FAIL: --strict gate"),
        "stderr must name the gate: {stderr}"
    );
    // Same report either way — `--strict` only changes the exit code.
    assert_eq!(gate_count(&strict_report), bad);

    // Clean: the same data, quantized ⇒ no Warning-or-worse finding.
    let clean = fixture(&dir, "clean", true);
    let (clean_ok, clean_report, clean_stderr) = doctor_json(&clean, &["--strict"]);
    assert_eq!(
        gate_count(&clean_report),
        0,
        "quantized fixture still has Warning+ findings: {clean_report}"
    );
    assert!(
        clean_ok,
        "--strict must exit 0 with no Warning-or-worse findings: {clean_stderr}"
    );
    assert!(clean_stderr.contains("--strict gate OK"), "{clean_stderr}");
}

/// Findings come out of the binary in the MO-7 total order — severity first,
/// then the largest projected byte win — and carry the additive fields.
#[test]
fn doctor_json_is_byte_ranked_and_carries_the_additive_fields() {
    let dir = tempfile::tempdir().unwrap();
    let archive = fixture(&dir, "ranked", false);
    let (_ok, report, _) = doctor_json(&archive, &[]);

    let findings = report["findings"].as_array().unwrap();
    assert!(!findings.is_empty());

    // (severity, projected_bytes desc, code, message) — ascending.
    let key = |f: &serde_json::Value| {
        (
            severity_rank(f["severity"].as_str().unwrap()),
            std::cmp::Reverse(f["projected_bytes"].as_u64().unwrap_or(0)),
            f["code"].as_str().unwrap().to_string(),
            f["message"].as_str().unwrap().to_string(),
        )
    };
    let observed: Vec<_> = findings.iter().map(key).collect();
    let mut expected = observed.clone();
    expected.sort();
    assert_eq!(observed, expected, "findings not byte-ranked: {report}");

    // The raw-f64 finding carries a measured projection in bytes.
    let raw = findings
        .iter()
        .find(|f| f["code"] == "raw-f64-column")
        .expect("raw-f64-column fires on the unquantized fixture");
    assert!(
        raw["projected_bytes"].as_u64().unwrap_or(0) > 0,
        "projected_bytes must be present and non-zero: {raw}"
    );
    assert!(
        raw["projected"]
            .as_str()
            .unwrap()
            .contains("(measured on sampled tiles)"),
        "the shrink is measured, not assumed: {raw}"
    );

    // Additive fields: `projected_bytes` is omitted where there is no honest
    // number, never serialized as null.
    let text = serde_json::to_string(&report).unwrap();
    assert!(!text.contains("\"projected_bytes\":null"));
}

/// Two identical `doctor` runs emit byte-identical stdout.
///
/// This is the P0-7 rule applied to MO-7: the measured-shrink pass is new
/// output-affecting work (it re-encodes columns through zstd), so it has to come
/// with a byte-identical re-run test.
#[test]
fn doctor_output_is_byte_identical_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let archive = fixture(&dir, "deterministic", false);
    let path = archive.to_string_lossy().into_owned();

    let json = |_: ()| run(&["doctor", "--archive", path.as_str(), "--format", "json"]).1;
    assert_eq!(json(()), json(()), "doctor JSON must be reproducible");

    let text = |_: ()| run(&["doctor", "--archive", path.as_str()]).1;
    assert_eq!(text(()), text(()), "doctor text must be reproducible");
}

/// `--sample 0` is the directory-only fast mode: no decode, so no column
/// findings and no measured projections — the documented fallback path.
#[test]
fn sample_zero_skips_the_decode_and_the_measured_projections() {
    let dir = tempfile::tempdir().unwrap();
    let archive = fixture(&dir, "fast-mode", false);
    let (ok, report, _) = doctor_json(&archive, &["--sample", "0"]);
    assert!(ok);

    for f in report["findings"].as_array().unwrap() {
        if let Some(p) = f["projected"].as_str() {
            assert!(
                !p.contains("(measured on sampled tiles)"),
                "nothing may claim a measurement when the decode was skipped: {f}"
            );
        }
    }
    // The column rules read `per_column`, which `--sample 0` leaves empty.
    assert!(
        !report["findings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["code"] == "raw-f64-column"),
        "column rules cannot fire without a decode: {report}"
    );
}

// ===========================================================================
// MO-6 — the binary runs the ITERATED advisor pass
// ===========================================================================

/// Rows in the GeoParquet fixture. Enough that the loader's sample clears the
/// measurement floor with room for the trial oracle's replicate blocks; few
/// enough that the ~two dozen sample encodes an iterated run performs stay quick
/// in the debug build these tests run against.
const RECOMMEND_ROWS: usize = 600;

/// A GeoParquet of `RECOMMEND_ROWS` points with one high-entropy numeric column
/// and one heavy-repeat categorical one — enough advisor surface that the
/// composed recipe has something to compose.
///
/// Written at test time, deterministically: there is no committed `.parquet`
/// anywhere in this repo (see `build_cli_reproducible.rs`'s header), and the
/// property under test is about two reads of ONE identical file, so a
/// synthesised fixture is not a compromise.
fn point_parquet(dir: &Path) -> PathBuf {
    let wkbs: Vec<Vec<u8>> = (0..RECOMMEND_ROWS)
        .map(|i| {
            // Little-endian WKB point, on a grid with sub-metre jitter so the
            // coordinate column is neither perfectly repetitive nor pure noise.
            let mut v = vec![0x01, 0x01, 0x00, 0x00, 0x00];
            let lon = -73.7 + (i % 30) as f64 * 0.004 + (mix(i as u64) % 1_000_000) as f64 * 1e-9;
            let lat =
                45.4 + (i / 30) as f64 * 0.004 + (mix(i as u64 + 17) % 1_000_000) as f64 * 1e-9;
            v.extend_from_slice(&lon.to_le_bytes());
            v.extend_from_slice(&lat.to_le_bytes());
            v
        })
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("magnitude", DataType::Float64, false),
        Field::new("region", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(BinaryArray::from_iter_values(
                wkbs.iter().map(|v| v.as_slice()),
            )),
            Arc::new(Int64Array::from(
                (0..RECOMMEND_ROWS as i64)
                    .map(|i| 1_700_000_000_000 + i * 30_000)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..RECOMMEND_ROWS)
                    .map(|i| (mix(i as u64 + 991) % 10_000_000) as f64 / 1e5)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..RECOMMEND_ROWS)
                    .map(|i| {
                        ["ville-marie", "verdun", "rosemont", "lachine", "sud-ouest"][i % 5]
                            .to_string()
                    })
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    let path = dir.join("points.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

/// Run a subcommand over the GeoParquet fixture. Returns (ok, stdout, stderr).
fn run_over_input(sub: &str, input: &Path, extra: &[&str]) -> (bool, String, String) {
    let input = input.to_string_lossy().into_owned();
    let mut args = vec![
        sub,
        "--input",
        input.as_str(),
        "--time-field",
        "timestamp",
        "--time-format",
        "unix-ms",
    ];
    args.extend_from_slice(extra);
    run(&args)
}

/// `recommend`'s stdout JSON, parsed.
fn recommend_json(input: &Path, extra: &[&str]) -> (serde_json::Value, String, String) {
    let (ok, stdout, stderr) = run_over_input("recommend", input, extra);
    assert!(ok, "recommend must exit 0; stderr:\n{stderr}");
    // `--explain`/`--show-command` append prose after the config, so parse the
    // leading JSON object rather than the whole stream.
    let mut de = serde_json::Deserializer::from_str(&stdout).into_iter::<serde_json::Value>();
    let value = de
        .next()
        .unwrap_or_else(|| panic!("recommend must emit a JSON config; stdout was:\n{stdout}"))
        .unwrap_or_else(|e| panic!("recommend config must be valid JSON ({e}):\n{stdout}"));
    (value, stdout, stderr)
}

/// **The wiring witness.** `composed_projected` is producible only by the
/// iterated pass (`run_iterative` → `generate_recommendations_composed`); the
/// round-0 `run_all` path leaves it null by construction. So finding it on the
/// process's stdout proves the binary is on the iterated path — the exact defect
/// this change fixes, where the mechanism existed in the library and the CLI
/// called past it.
#[test]
fn recommend_emits_the_composed_recipe_measurement() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let (config, _, _) = recommend_json(&input, &[]);

    let projected = config["composed_projected"]
        .as_str()
        .unwrap_or_else(|| panic!("the binary must run the ITERATED pass: {config}"));
    // It names its reference and how many rounds refined it…
    assert!(projected.contains("at build defaults"), "{projected}");
    assert!(projected.contains("refinement round"), "{projected}");
    // …and what the extra measurement cost, which is the plan's documented
    // requirement for this mechanism's headline risk.
    assert!(projected.contains("sample encodes"), "{projected}");
    assert!(projected.contains("cache hits"), "{projected}");

    // The same line reaches the human-facing explanation list.
    let explanations = config["explanations"].as_array().unwrap();
    assert!(
        explanations.iter().any(|e| e
            .as_str()
            .is_some_and(|s| s.starts_with("Composed recipe: "))),
        "{config}"
    );

    // MO-8's rollback, asserted on the same (expensive) process run: without
    // `--target-size` the config gains no `budget` key at all — not even a null
    // one — so every existing consumer of this JSON sees exactly what it always
    // saw.
    assert!(
        config.get("budget").is_none(),
        "an unbudgeted recommend must not emit a budget key: {config}"
    );
}

/// `analyze` is wired too — and deliberately so.
///
/// It is the surface where a HUMAN reads a lossy lever's projected win and
/// decides whether to accept the quality tradeoff. A single-pass number there
/// can be measured at the build-default zstd level and evaporate on the
/// level-19 recipe the same run recommends, so leaving `analyze` on `run_all`
/// would have put the misleading figure in front of the reader least able to
/// check it while `recommend` quietly got the right one.
#[test]
fn analyze_report_carries_the_composed_recipe_measurement() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let (ok, stdout, stderr) = run_over_input("analyze", &input, &["--format", "json"]);
    assert!(ok, "analyze must exit 0; stderr:\n{stderr}");
    let report: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("analyze --format json must be valid JSON ({e}):\n{stdout}"));

    let explanations = report["recommendations"]["explanations"]
        .as_array()
        .unwrap_or_else(|| panic!("analyze report must carry explanations: {report}"));
    assert!(
        explanations.iter().any(|e| e
            .as_str()
            .is_some_and(|s| s.starts_with("Composed recipe: "))),
        "analyze must run the ITERATED pass: {report}"
    );

    // stdout stays a pure report — the progress banner and the composed figure
    // ride stderr, so `--format json` still pipes into jq.
    assert!(
        stderr.contains("Composed recipe: "),
        "the composed figure belongs on the banner channel: {stderr}"
    );
}

/// Two identical `recommend` runs emit byte-identical stdout.
///
/// The P0-7 determinism rule at the seam that matters. A recommendation becomes
/// build flags, build flags become archive bytes, and pack names in this format
/// are content-addressed — a recipe that wobbles between runs churns pack names
/// across the fleet on the next republish. Iteration added a whole coordinate
/// descent with a measurement cache behind it, so this is a real claim.
///
/// `analyze` is checked in the same test because it now runs the same descent.
#[test]
fn recommend_and_analyze_output_is_byte_identical_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let before = std::fs::read(&input).unwrap();

    let rec = |_: ()| run_over_input("recommend", &input, &["--explain", "--show-command"]).1;
    assert_eq!(rec(()), rec(()), "recommend stdout must be reproducible");

    let analyze = |_: ()| run_over_input("analyze", &input, &["--format", "json"]).1;
    assert_eq!(
        analyze(()),
        analyze(()),
        "analyze stdout must be reproducible"
    );

    // And the analysis is measurement-side: nothing writes to the input.
    assert_eq!(
        before,
        std::fs::read(&input).unwrap(),
        "analysis must not touch the input"
    );
}

/// `--explain` surfaces the composed recipe AND what measuring it cost.
///
/// The evidence table above the block prices levers one at a time; this block is
/// the only place the reader sees the recipe as a whole. The cost line is not
/// decoration — the plan names "analysis wall-time grows ~5-10x in oracle calls"
/// as this mechanism's headline risk and requires it be documented here rather
/// than experienced as an unexplained slowdown.
#[test]
fn explain_surfaces_the_composed_recipe_and_its_measured_cost() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let (_, stdout, _) = recommend_json(&input, &["--explain"]);

    assert!(
        stdout.contains("Composed recipe (measured):"),
        "--explain must show the composed recipe: {stdout}"
    );
    assert!(
        stdout.contains("Cost: refinement re-encodes the sample"),
        "--explain must state what iteration cost: {stdout}"
    );
    assert!(
        stdout.contains("does NOT grow with the dataset"),
        "the cost statement must say how it scales: {stdout}"
    );
    // The per-run counters, not just prose.
    assert!(stdout.contains("sample encodes"), "{stdout}");
}

/// **The no-thinning guard at the binary boundary.**
///
/// Iteration is allowed to change a lever's NUMBERS and even to withdraw a
/// lossy advisory whose win evaporated on the composed recipe. It is never
/// allowed to change a lever's ADMISSIBILITY — so whatever the descent decided,
/// no `lossy` and no `suggestion_only` flag may appear in the command the binary
/// hands a user or an agent.
///
/// This is derived from the emitted advice rather than hard-coded, so it keeps
/// biting as advisors are added. It does not replace
/// `lossy_advice_never_joins_to_command` /
/// `suggestion_only_advice_never_joins_to_command` — those pin the filter, this
/// pins that the filter is still what stands between the two after an extra
/// measurement pass.
#[test]
fn no_lossy_or_suggestion_only_flag_survives_iteration_into_the_command() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let (config, stdout, _) = recommend_json(&input, &["--show-command"]);

    let command = config["command"].as_str().expect("command string");
    let advice = config["advice"].as_array().expect("advice array");
    assert!(
        !advice.is_empty(),
        "fixture guard: this input must produce advice, or the test proves nothing: {config}"
    );

    let mut gated = 0usize;
    for a in advice {
        let flag = a["flag"].as_str().unwrap();
        let lossy = a["lossy"].as_bool().unwrap_or(false);
        let suggestion_only = a["suggestion_only"].as_bool().unwrap_or(false);
        if lossy || suggestion_only {
            gated += 1;
            assert!(
                !command.contains(flag),
                "{flag} (lossy={lossy}, suggestion_only={suggestion_only}) leaked into the \
                 command after iteration: {command}"
            );
        }
    }
    assert!(
        gated > 0,
        "fixture guard: this input must produce at least one gated lever: {config}"
    );

    // `--show-command` prints the same filtered command, so the human-facing
    // paste target is covered by the same assertion.
    let printed = stdout
        .split_once("Suggested stt-build command:")
        .map(|(_, tail)| tail.to_string())
        .expect("--show-command must print the command");
    for a in advice {
        let flag = a["flag"].as_str().unwrap();
        if a["lossy"].as_bool().unwrap_or(false) || a["suggestion_only"].as_bool().unwrap_or(false)
        {
            assert!(!printed.contains(flag), "{flag} leaked: {printed}");
        }
    }
}

// ===========================================================================
// MO-8 — `--target-size B`, the budget solver, at the binary boundary
// ===========================================================================
//
// What these pin, and why it has to be the PROCESS rather than the library:
//
// * **The flag exists and reaches the solver.** `--target-size` is the
//   program's most user-facing move; a solver nobody can invoke is not one.
// * **THE NO-THINNING GUARD, at the boundary a user actually crosses.** An
//   unreachable budget must report the lexicographic floor and drop nothing —
//   no feature cap, no quantization, no lossy flag in the command the binary
//   hands a human or an agent, however much budget pressure it is under.
// * **`--fail-if-over-target` is the CI shape of that**: print the whole
//   report, then exit non-zero — `diff --fail-on-growth`'s pattern, so a red
//   gate still leaves the operator everything they need.
// * **Determinism at the seam that becomes bytes.** A budget recipe becomes
//   build flags and pack names are content-addressed.

/// `recommend --target-size …`, parsed. Returns (ok, config JSON, stdout,
/// stderr) — the gate can legitimately exit non-zero, so this does NOT assert
/// success the way `recommend_json` does.
fn budget_run(input: &Path, extra: &[&str]) -> (bool, serde_json::Value, String, String) {
    let (ok, stdout, stderr) = run_over_input("recommend", input, extra);
    let mut de = serde_json::Deserializer::from_str(&stdout).into_iter::<serde_json::Value>();
    let config = de
        .next()
        .unwrap_or_else(|| panic!("recommend must emit a JSON config; stdout was:\n{stdout}"))
        .unwrap_or_else(|e| panic!("recommend config must be valid JSON ({e}):\n{stdout}"));
    (ok, config, stdout, stderr)
}

/// A reachable budget: the solver fits it, prints the table, and the emitted
/// command describes the recipe it priced.
#[test]
fn a_reachable_target_size_solves_prints_the_table_and_passes_the_gate() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let (ok, config, stdout, stderr) = budget_run(
        &input,
        &[
            "--target-size",
            "1G",
            "--fail-if-over-target",
            "--show-command",
        ],
    );
    assert!(ok, "a reachable budget must exit 0; stderr:\n{stderr}");

    // The report reaches the machine-readable config…
    let budget = &config["budget"];
    assert!(!budget.is_null(), "{config}");
    // …with the size suffix resolved (1G = 1 GiB, binary).
    assert_eq!(budget["target_bytes"].as_u64(), Some(1024 * 1024 * 1024));
    assert_eq!(budget["feasible"].as_bool(), Some(true));
    assert!(budget["projected_bytes"].as_u64().unwrap() <= 1024 * 1024 * 1024);
    assert!(budget["floor_bytes"].as_u64().unwrap() > 0);
    assert!(!budget["zstd_sweep"].as_array().unwrap().is_empty());
    // The register's standing rejection, checked on real output: 19 is the cap.
    for level in budget["zstd_sweep"].as_array().unwrap() {
        assert!(
            level.as_i64().unwrap() <= 19,
            "zstd sweep past 19: {budget}"
        );
    }
    // Every shadow price is lossy, by construction.
    for price in budget["shadow_prices"].as_array().unwrap() {
        assert_eq!(price["lossy"].as_bool(), Some(true), "{price}");
    }
    // No chosen lever may be a thinning lever.
    for lever in budget["chosen"].as_array().unwrap() {
        let flag = lever["flag"].as_str().unwrap();
        assert!(!flag.starts_with("--quantize"), "{lever}");
        assert!(!flag.starts_with("--maximum-tile"), "{lever}");
        assert!(flag != "--drop-densest-as-needed", "{lever}");
    }

    // …and to the human, as a table.
    assert!(
        stdout.contains("Budget solver (--target-size):"),
        "{stdout}"
    );
    assert!(stdout.contains("no feature is dropped"), "{stdout}");
    assert!(
        stdout.contains("LOSSY levers, opt-in only"),
        "the shadow-price table must announce itself: {stdout}"
    );
    assert!(stdout.contains("FITS"), "{stdout}");

    // The gate reports its verdict on the banner channel.
    assert!(stderr.contains("--fail-if-over-target gate OK"), "{stderr}");

    // The pasteable command carries the solved recipe and NO lossy flag.
    let command = config["command"].as_str().expect("command string");
    for lossy in [
        "--quantize-coords",
        "--quantize-attrs-auto",
        "--quantize-attr",
        "--maximum-tile-features",
        "--maximum-tile-bytes",
        "--drop-densest-as-needed",
    ] {
        assert!(!command.contains(lossy), "lossy flag leaked: {command}");
    }
    for lever in budget["chosen"].as_array().unwrap() {
        if lever["suggestion_only"].as_bool().unwrap_or(false) {
            let flag = lever["flag"].as_str().unwrap();
            assert!(
                !command.contains(flag),
                "suggestion-only {flag} leaked into the command: {command}"
            );
        }
    }
}

/// **The no-thinning guard, at the binary.** A budget nothing reversible can
/// reach reports the floor, drops nothing, and fails the CI gate.
#[test]
fn an_unreachable_target_reports_the_floor_drops_nothing_and_fails_the_gate() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let (ok, config, stdout, stderr) = budget_run(
        &input,
        &[
            "--target-size",
            "512",
            "--fail-if-over-target",
            "--show-command",
        ],
    );
    assert!(
        !ok,
        "an unreachable budget must fail the gate; stdout:\n{stdout}"
    );

    let budget = &config["budget"];
    assert_eq!(budget["feasible"].as_bool(), Some(false), "{budget}");
    // The floor is REPORTED, not chased.
    assert_eq!(budget["projected_bytes"], budget["floor_bytes"]);
    assert!(budget["floor_bytes"].as_u64().unwrap() > 512);
    let notes = budget["notes"]
        .as_array()
        .unwrap()
        .iter()
        .fold(String::new(), |mut acc, n| {
            acc.push_str(n.as_str().unwrap_or_default());
            acc.push(' ');
            acc
        });
    assert!(notes.contains("INFEASIBLE"), "{notes}");
    assert!(notes.contains("NOTHING HAS BEEN DROPPED"), "{notes}");

    // …and the shadow prices are what it offers instead, all lossy.
    let prices = budget["shadow_prices"].as_array().unwrap();
    assert!(!prices.is_empty(), "{budget}");
    assert!(prices.iter().all(|p| p["lossy"].as_bool() == Some(true)));

    // NOT ONE lossy flag reaches the emitted command, under maximum pressure.
    let command = config["command"].as_str().expect("command string");
    let printed = stdout
        .split_once("Suggested stt-build command:")
        .map(|(_, tail)| tail.to_string())
        .expect("--show-command must print the command");
    for lossy in [
        "--quantize-coords",
        "--quantize-attrs-auto",
        "--quantize-attr",
        "--maximum-tile-features",
        "--maximum-tile-bytes",
        "--drop-densest-as-needed",
        "--min-zoom-field",
        "--summary-tier",
    ] {
        assert!(!command.contains(lossy), "lossy flag leaked: {command}");
        assert!(!printed.contains(lossy), "lossy flag leaked: {printed}");
    }
    // The failure explains itself and says what was NOT done.
    assert!(
        stderr.contains("FAIL: --fail-if-over-target gate"),
        "{stderr}"
    );
    assert!(stderr.contains("Nothing was dropped"), "{stderr}");
    assert!(stdout.contains("DOES NOT FIT"), "{stdout}");
}

/// The gate needs something to gate against, and says so — instantly, without
/// paying for an analysis pass first.
#[test]
fn fail_if_over_target_without_a_target_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let (ok, _stdout, stderr) = run_over_input("recommend", &input, &["--fail-if-over-target"]);
    assert!(!ok, "the flag must not silently pass");
    assert!(
        stderr.contains("--fail-if-over-target needs a --target-size"),
        "{stderr}"
    );

    // A malformed size fails the same way, before any measurement happens.
    let (ok, _stdout, stderr) = run_over_input("recommend", &input, &["--target-size", "1QB"]);
    assert!(!ok);
    assert!(stderr.contains("unknown unit"), "{stderr}");
}

/// Two identical budget runs emit byte-identical stdout.
///
/// The P0-7 determinism rule at the seam where a recommendation becomes build
/// flags. The solver added a whole enumeration over re-measured candidate
/// points on top of the coordinate descent, so this is a real claim, not a
/// formality.
#[test]
fn budget_mode_output_is_byte_identical_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let input = point_parquet(dir.path());
    let args = ["--target-size", "1G", "--show-command"];
    let run_once = |_: ()| run_over_input("recommend", &input, &args).1;
    assert_eq!(
        run_once(()),
        run_once(()),
        "a budget recipe must be reproducible byte for byte"
    );
}
