//! MO-6 integration: `recommend_for` is a deterministic solver.
//!
//! Iteration turned `recommend` from one pass into a small coordinate descent:
//! it composes the advisors' recipe, measures it, re-prices each lever against
//! it, and does that twice. Every one of those steps is a decision, and a
//! decision that is not reproducible is not a recommendation — it is a coin
//! flip with a rationale attached.
//!
//! The specific hazard is not abstract. Recommendations become build flags,
//! build flags become archive bytes, and pack names in this format are
//! CONTENT-ADDRESSED: a recipe that wobbles between two runs churns pack names
//! across the whole fleet on the next republish. So the contract this file pins
//! is the strict one — two full runs over the same file must serialise BYTE for
//! BYTE, not merely agree in their conclusions.
//!
//! It also pins the rollback (`rounds = 0` reproduces the historical single
//! pass through the public API) and the two standing guard rails that iteration
//! is forbidden to touch: lossy advice never joins the emitted command, and
//! nothing here writes a byte to the input.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{BinaryArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use stt_optimize::advisors::{self, RECOMMEND_ROUNDS};
use stt_optimize::recommend::{self, Recommendations};
use stt_optimize::{analyze_source, recommend_for, recommend_for_with_rounds, DataSource};

/// Rows in the fixture. Enough that the loader's sample clears the measurement
/// floor with room for the oracle's four replicate blocks, few enough that the
/// ~two dozen sample encodes an iterated run performs stay quick in a debug
/// build.
const ROWS: usize = 600;

/// splitmix64 — full-entropy values with no RNG dependency, so the fixture file
/// is byte-identical on every run and every machine.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A GeoParquet of `ROWS` points carrying one high-entropy numeric property and
/// one heavy-repeat categorical one — enough advisor surface that the composed
/// recipe has something to compose.
fn point_parquet(dir: &Path) -> PathBuf {
    let wkbs: Vec<Vec<u8>> = (0..ROWS)
        .map(|i| {
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
                (0..ROWS as i64)
                    .map(|i| 1_700_000_000_000 + i * 30_000)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..ROWS)
                    .map(|i| (mix(i as u64 + 991) % 10_000_000) as f64 / 1e5)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..ROWS)
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

fn source_at(path: PathBuf) -> DataSource {
    DataSource::GeoParquet {
        path,
        time_field: "timestamp".to_string(),
        time_format: "unix-ms".to_string(),
    }
}

fn json(rec: &Recommendations) -> String {
    serde_json::to_string(rec).unwrap()
}

#[test]
fn two_recommend_runs_over_one_file_serialise_byte_identically() {
    let dir = tempfile::tempdir().unwrap();
    let path = point_parquet(dir.path());
    let before = std::fs::read(&path).unwrap();
    let source = source_at(path.clone());

    let first = recommend_for(&source).unwrap();
    let second = recommend_for(&source).unwrap();

    // THE determinism contract. Not "the same advice" — the same BYTES, because
    // this recipe becomes build flags and pack names are content-addressed.
    assert_eq!(
        json(&first),
        json(&second),
        "two recommend runs must serialise byte-identically"
    );
    // The pasteable command is derived, so it must be identical too.
    let cmd = |rec: &Recommendations| recommend::to_command(rec, &path, "timestamp");
    assert_eq!(cmd(&first), cmd(&second));
    assert_eq!(
        serde_json::to_string(&recommend::to_build_config(&first, &path, "timestamp")).unwrap(),
        serde_json::to_string(&recommend::to_build_config(&second, &path, "timestamp")).unwrap(),
    );

    // Analysis is measurement-side: the input is not written to.
    assert_eq!(
        before,
        std::fs::read(&path).unwrap(),
        "recommend must not touch the input"
    );
}

#[test]
fn the_composed_recipe_is_measured_and_reported() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_at(point_parquet(dir.path()));
    let rec = recommend_for(&source).unwrap();

    let projected = rec
        .composed_projected
        .as_deref()
        .expect("a measurable sample must publish the composed figure");
    assert!(
        projected.contains("at build defaults"),
        "the composed figure must name its reference: {projected}"
    );
    assert!(
        projected.contains(&format!("{RECOMMEND_ROUNDS} refinement rounds")),
        "{projected}"
    );
    // The cost of iteration is documented in the output rather than hidden:
    // this is the item's headline risk (analysis wall-time grows ~5-10x in
    // encode work), and the reader gets to see how much was spent and how much
    // the cache saved.
    assert!(projected.contains("sample encodes"), "{projected}");
    assert!(projected.contains("cache hits"), "{projected}");

    // The same line reaches the human-facing surfaces.
    assert!(
        rec.explanations
            .iter()
            .any(|e| e.starts_with("Composed recipe: ")),
        "{:?}",
        rec.explanations
    );
    let config = recommend::to_build_config(&rec, Path::new("points.parquet"), "timestamp");
    assert_eq!(config["composed_projected"], serde_json::json!(projected));

    // Iteration refines numbers, never admissibility: whatever it decided, no
    // lossy lever may reach the pasteable command.
    let command = recommend::to_command(&rec, Path::new("points.parquet"), "timestamp");
    for lossy in [
        "--quantize-coords",
        "--quantize-attrs-auto",
        "--maximum-tile-features",
    ] {
        assert!(!command.contains(lossy), "lossy flag leaked: {command}");
    }
    for advice in &rec.advice {
        if advice.lossy || advice.suggestion_only {
            assert!(
                !command.contains(&advice.flag),
                "{} leaked into the command: {command}",
                advice.flag
            );
        }
    }
}

#[test]
fn zero_rounds_reproduces_the_single_pass_through_the_public_api() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_at(point_parquet(dir.path()));

    let rolled_back = recommend_for_with_rounds(&source, 0).unwrap();
    // The historical pass, assembled by hand from the same public pieces.
    let (result, data) = analyze_source(&source).unwrap();
    let single_pass =
        recommend::generate_recommendations(&result, advisors::run_all(&result, &data).unwrap());

    assert_eq!(
        json(&rolled_back),
        json(&single_pass),
        "rounds = 0 is the rollback: it must reproduce the single pass exactly"
    );
    assert!(rolled_back.composed_projected.is_none());

    // …and the rollback is a strictly cheaper, strictly less informed answer:
    // the iterated run measures the recipe, the single pass never did.
    let iterated = recommend_for(&source).unwrap();
    assert!(iterated.composed_projected.is_some());
    assert_eq!(
        iterated.min_zoom, rolled_back.min_zoom,
        "iteration must not move the zoom recipe"
    );
    assert_eq!(iterated.max_zoom, rolled_back.max_zoom);
    assert_eq!(
        iterated.temporal_bucket_ms, rolled_back.temporal_bucket_ms,
        "iteration must not move the temporal recipe"
    );
}
