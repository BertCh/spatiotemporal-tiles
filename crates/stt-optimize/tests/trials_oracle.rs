//! MO-5 integration: the trial oracle against a real loaded source.
//!
//! The unit tests in `oracle` prove each candidate's arithmetic on a
//! hand-built sample. This suite proves the thing that arithmetic is only
//! useful *inside*: that trials run on the sample a real GeoParquet analysis
//! produced, under the layout that analysis actually measured itself with, and
//! that the numbers therefore compose with `AnalysisResult::measured` instead of
//! merely resembling it.
//!
//! The failure this exists to catch is the quiet one — a trial measured under a
//! DIFFERENT tile cut from the baseline. Its delta then prices the cut, not the
//! lever, and every consumer downstream (the advisors, the doctor's what-ifs,
//! the `--target-size` solver) inherits a number that is confidently wrong. The
//! first test pins the identity that rules it out: a no-op candidate must
//! measure the run's own baseline byte for byte.
//!
//! Nothing here writes an archive. The oracle is measurement-side and
//! BYTE-NEUTRAL; the input file's bytes are checked to be untouched.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{BinaryArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use stt_optimize::measure::{measure_sample_layout, MeasureSettings, SyntheticLayout};
use stt_optimize::oracle::{run_trials, Candidate, TrialScope, MAX_ZSTD_LEVEL};
use stt_optimize::{analyze_source_with, DataSource, MeasurementMode};

/// Rows in the fixture. Enough that the loader's sample clears the measurement
/// floor with room for four replicate blocks, and that the density scan sees
/// many occupied cells.
const ROWS: usize = 3_000;

/// Distinct values in the fixture's categorical column — a heavy-repeat Utf8
/// column, which is the shape the encoder's pre-compression dictionary model is
/// most confident about and most wrong about.
const CATEGORIES: usize = 6;

/// Write a GeoParquet of `ROWS` points over a small region, carrying one
/// high-entropy numeric property and one heavy-repeat categorical one.
fn point_parquet(dir: &Path) -> PathBuf {
    // splitmix64: full-entropy coordinates with no RNG dependency, so the file
    // is byte-identical on every run and every machine.
    let mix = |x: u64| -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let wkbs: Vec<Vec<u8>> = (0..ROWS)
        .map(|i| {
            let mut v = vec![0x01, 0x01, 0x00, 0x00, 0x00];
            let jitter = |salt: u64| (mix(i as u64 + salt) % 1_000_000) as f64 * 1e-8;
            let lon = -73.7 + (i % 60) as f64 * 0.004 + jitter(0);
            let lat = 45.4 + (i / 60) as f64 * 0.004 + jitter(17);
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
            // 0..100 with five decimals of entropy: high-entropy enough that
            // quantization has real bytes to win, narrow enough that the
            // `--quantize-attr` rung below stays inside the Int32 leaf.
            Arc::new(Float64Array::from(
                (0..ROWS)
                    .map(|i| (mix(i as u64 + 991) % 10_000_000) as f64 / 1e5)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..ROWS)
                    .map(|i| {
                        [
                            "ville-marie",
                            "le-plateau-mont-royal",
                            "rosemont-la-petite-patrie",
                            "verdun",
                            "cote-des-neiges",
                            "mercier-hochelaga-maisonneuve",
                        ][i % CATEGORIES]
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

/// The candidate set the suite prices — one of every variant, so a change to
/// any arm shows up somewhere.
fn candidates() -> Vec<Candidate> {
    vec![
        Candidate::ZstdLevel(MAX_ZSTD_LEVEL),
        Candidate::CompactTimes(false),
        Candidate::QuantizeCoords(Some(0.5)),
        Candidate::QuantizeAttr {
            column: "magnitude".to_string(),
            step: 0.001,
        },
        Candidate::QuantizeAttrsAuto(true),
        Candidate::CategoricalDict {
            column: "region".to_string(),
            dict: true,
        },
        Candidate::FeatureBudgetBytes { max_features: 96 },
    ]
}

#[test]
fn trials_measure_under_the_run_layout_the_analysis_reconstructs() {
    // THE composability identity. `analyze_source_with(DensityLayout)` measures
    // the sample under a layout derived from the density occupancy scan; a
    // consumer holding only the finished `AnalysisResult` rebuilds that layout
    // with `SyntheticLayout::from_density`. If the oracle is handed that
    // reconstruction, a no-op candidate must land exactly on the run's own
    // measured bytes — anything else means the trials are pricing the cut.
    let dir = tempfile::tempdir().unwrap();
    let source = source_at(point_parquet(dir.path()));
    let (result, data) = analyze_source_with(&source, MeasurementMode::DensityLayout).unwrap();

    let layout = SyntheticLayout::from_density(&result.density);
    let baseline = MeasureSettings::default();
    let measured = result.measured.as_ref().expect("the sample is measurable");
    assert!(measured.tiles > 1, "the run layout must be multi-tile");

    // `CompactTimes(true)` and `QuantizeCoords(None)` are both the build
    // default, so each is a no-op that must reproduce the baseline exactly.
    let trials = run_trials(
        &data.sample,
        &layout,
        &baseline,
        &[
            Candidate::CompactTimes(true),
            Candidate::QuantizeCoords(None),
        ],
    )
    .unwrap();
    for trial in &trials {
        assert_eq!(
            trial.bytes_total, measured.bytes_total,
            "a no-op candidate must reproduce the run's measured bytes: {trial:?}"
        );
        assert_eq!(trial.delta_bytes, 0, "{trial:?}");
        assert_eq!(trial.delta_frac, 0.0, "{trial:?}");
    }
}

#[test]
fn every_candidate_prices_on_a_real_loaded_source() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_at(point_parquet(dir.path()));
    let (result, data) = analyze_source_with(&source, MeasurementMode::DensityLayout).unwrap();
    let layout = SyntheticLayout::from_density(&result.density);
    let baseline = MeasureSettings::default();

    let candidates = candidates();
    let trials = run_trials(&data.sample, &layout, &baseline, &candidates).unwrap();
    assert_eq!(trials.len(), candidates.len());

    for (trial, candidate) in trials.iter().zip(candidates.iter()) {
        assert_eq!(&trial.candidate, candidate, "results keep input order");
        assert!(trial.bytes_total > 0, "{trial:?}");
        assert!(trial.delta_frac.is_finite(), "{trial:?}");
        assert!(
            trial.stderr.is_finite() && (0.0..0.5).contains(&trial.stderr),
            "a published spread must be finite and must not swamp the delta: {trial:?}"
        );
        // Whole-tile results are on the sample's scale; the column result is on
        // its own column's scale and must NEVER be ranked against them.
        let baseline_bytes = result.measured.as_ref().unwrap().bytes_total as f64;
        let ratio = trial.bytes_total as f64 / baseline_bytes;
        match candidate.scope() {
            TrialScope::WholeTile => assert!(
                (0.25..4.0).contains(&ratio),
                "a whole-tile trial is on the sample's scale and must stay near the baseline \
                 ({ratio:.3}x): {trial:?}"
            ),
            TrialScope::Column => assert!(
                ratio < 1.0,
                "a column trial is on ONE column's scale and cannot exceed the whole sample \
                 ({ratio:.3}x): {trial:?}"
            ),
        }
    }

    // The levers that must pay on this fixture, by construction: publish-grade
    // zstd on compressible tiles, and coordinate quantization on jittered
    // Float64 coordinates.
    let by = |c: &Candidate| {
        trials
            .iter()
            .find(|t| &t.candidate == c)
            .unwrap_or_else(|| panic!("missing {c:?}"))
    };
    assert!(
        by(&Candidate::ZstdLevel(MAX_ZSTD_LEVEL)).delta_bytes < 0,
        "zstd {MAX_ZSTD_LEVEL} must shrink the wire"
    );
    assert!(
        by(&Candidate::QuantizeCoords(Some(0.5))).delta_bytes < 0,
        "coordinate quantization must shrink the wire"
    );
    // §4.3: capping occupancy re-cuts into more tiles, which costs framing.
    assert!(
        by(&Candidate::FeatureBudgetBytes { max_features: 96 }).delta_bytes > 0,
        "a tighter occupancy cap means more per-tile framing on the wire"
    );

    // Lossy levers are priced, never hidden — and they announce themselves, so
    // the no-thinning filter downstream can drop them without re-deriving why.
    for candidate in &candidates {
        assert_eq!(
            candidate.lossy(),
            matches!(
                candidate,
                Candidate::QuantizeCoords(Some(_))
                    | Candidate::QuantizeAttr { .. }
                    | Candidate::QuantizeAttrsAuto(true)
            ),
            "{candidate:?}"
        );
    }
}

#[test]
fn trials_are_byte_identical_across_independent_analyses() {
    // The determinism contract at the seam that matters: two INDEPENDENT
    // analyses of the same file (fresh load, fresh sample, fresh density scan,
    // fresh layout) must produce the same trials, byte for byte. This vector is
    // solver input, and pack names are content-addressed.
    let dir = tempfile::tempdir().unwrap();
    let source = source_at(point_parquet(dir.path()));
    let candidates = candidates();

    let json = |_: ()| {
        let (result, data) = analyze_source_with(&source, MeasurementMode::DensityLayout).unwrap();
        let layout = SyntheticLayout::from_density(&result.density);
        let trials = run_trials(
            &data.sample,
            &layout,
            &MeasureSettings::default(),
            &candidates,
        )
        .unwrap();
        serde_json::to_string(&trials).unwrap()
    };
    assert_eq!(json(()), json(()));
}

#[test]
fn the_zstd_cap_holds_at_the_library_boundary() {
    // The register guard, exercised through the public entry point rather than
    // through `Candidate::validate` alone: an over-cap level must be refused
    // for the WHOLE list, before anything is encoded.
    let dir = tempfile::tempdir().unwrap();
    let source = source_at(point_parquet(dir.path()));
    let (result, data) = analyze_source_with(&source, MeasurementMode::DensityLayout).unwrap();
    let layout = SyntheticLayout::from_density(&result.density);

    let err = run_trials(
        &data.sample,
        &layout,
        &MeasureSettings::default(),
        &[
            Candidate::ZstdLevel(MAX_ZSTD_LEVEL),
            Candidate::ZstdLevel(22),
        ],
    )
    .unwrap_err();
    assert!(err.to_string().contains("standing rejection"), "{err}");
    assert_eq!(MAX_ZSTD_LEVEL, 19);
}

#[test]
fn the_oracle_reads_and_never_writes() {
    // BYTE-NEUTRAL, at the only bytes this suite has: the input file. Running
    // the full candidate set must not touch it, and the analysis it feeds must
    // still measure the same after.
    let dir = tempfile::tempdir().unwrap();
    let path = point_parquet(dir.path());
    let before = std::fs::read(&path).unwrap();
    let source = source_at(path.clone());

    let (result, data) = analyze_source_with(&source, MeasurementMode::DensityLayout).unwrap();
    let layout = SyntheticLayout::from_density(&result.density);
    run_trials(
        &data.sample,
        &layout,
        &MeasureSettings::default(),
        &candidates(),
    )
    .unwrap();

    assert_eq!(
        before,
        std::fs::read(&path).unwrap(),
        "the input file moved"
    );
    let after = measure_sample_layout(&data.sample, &MeasureSettings::default(), &layout)
        .unwrap()
        .unwrap();
    assert_eq!(
        after.bytes_total,
        result.measured.as_ref().unwrap().bytes_total,
        "re-measuring the baseline after a trial run must give the same bytes"
    );
    // No process-global encoder state was set behind anyone's back: a config
    // built from the globals is still the untouched default.
    assert_eq!(stt_core::arrow_tile::quantize_coords_m(), None);
    assert!(!stt_core::arrow_tile::quantize_attrs_auto());
    assert!(stt_core::arrow_tile::quantize_attrs().is_empty());
}
