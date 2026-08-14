//! stt-optimize as a library.
//!
//! `stt-build --auto` calls into [`recommend_for`] to pick zoom levels and
//! a temporal bucket from an input file before building.
//! The `stt-optimize` CLI (a binary in the `spatiotemporal-tiles` facade
//! crate) is a thin wrapper around the same functions.

pub mod advisors;
pub mod analysis;
pub mod attribution;
pub mod budget_solver;
pub mod diff;
pub mod doctor;
pub mod export;
pub mod loader;
pub mod measure;
pub mod oracle;
pub mod order_audit;
pub mod packed;
pub mod read_amp;
pub mod recommend;
pub mod report;

#[cfg(test)]
mod test_support;

use anyhow::Result;

pub use advisors::{
    run_iterative, Advice, ComposedMeasurement, IterativeAdvice, OracleUsage, RECOMMEND_ROUNDS,
};
pub use analysis::inspect::InspectReport;
pub use attribution::AttributionDesign;
pub use budget_solver::{
    parse_size, BudgetReport, ChosenLever, DistortionClass, EstimateBasis, ShadowPrice,
};
pub use diff::DiffReport;
pub use doctor::DoctorReport;
pub use export::{ExportOptions, ExportReport, GeometryEncoding};
pub use loader::DataSource;
pub use measure::{MeasureSettings, MeasuredEncoding, SyntheticLayout};
pub use oracle::{run_trials, Candidate, TrialResult, TrialScope, MAX_ZSTD_LEVEL};
pub use order_audit::OrderAuditReport;
pub use packed::PackedTileset;
pub use recommend::Recommendations;

/// How an analysis run cuts its sample before encoding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeasurementMode {
    /// One synthetic tile for the whole sample — the pre-MO-4 behaviour, kept
    /// as the documented ROLLBACK.
    ///
    /// It amortises per-tile encoder + zstd framing over the entire sample, so
    /// its bytes/feature is biased LOW (a measured 1.47× on an 800-feature
    /// point sample). Use it only to reproduce a pre-layout number.
    SingleTile,
    /// Cut the sample into the density-derived [`measure::SyntheticLayout`], so
    /// per-tile encoder + zstd framing is charged at real tile occupancy. The
    /// default.
    #[default]
    DensityLayout,
}

/// Run the full analysis pipeline over an input with the default
/// ([`MeasurementMode::DensityLayout`]) measurement mode, returning the loaded
/// data alongside the result so callers can hand both to the advisors.
pub fn analyze_source(
    source: &DataSource,
) -> Result<(analysis::AnalysisResult, loader::LoadedData)> {
    analyze_source_with(source, MeasurementMode::default())
}

/// Run the full analysis pipeline, choosing the measurement mode.
///
/// **Pass ordering is load-bearing under [`MeasurementMode::DensityLayout`].**
/// The measured sample encoding must be cut into synthetic tiles that match
/// REAL tile occupancy, so the density occupancy scan runs FIRST (uncalibrated:
/// it is passed no measurement), the [`measure::SyntheticLayout`] comes off it,
/// the measurement follows, and the final density pass is re-run with that
/// measurement for its size estimates. The occupancy statistics the layout
/// reads (median / p95 / max features per tile) do NOT depend on `measured`, so
/// they are identical in both passes — which is what makes
/// `SyntheticLayout::from_density(&result.density)` reconstruct exactly the
/// layout `result.measured` was produced under. Any later trial encode MUST be
/// measured under that same layout, via
/// [`measure::measure_sample_layout`], or its delta confounds the lever with
/// the layout.
///
/// ⚠️ **The gate this default used to sit behind, and why it is closed.**
/// `DensityLayout` was held back because the advisors in
/// `advisors/quantize.rs` and `advisors/layout.rs` took their BASELINE from
/// `result.measured` but measured their TRIALS single-tile. On a measured
/// 800-feature point sample the layout measurement is **1.47×** the single-tile
/// one, so a layout baseline against a single-tile trial reads as a ~32%
/// shrink — enough to fire every `--quantize-coords` / `--quantize-attrs-auto`
/// advisory, both LOSSY, at High confidence on every dataset. Spuriously
/// recommending lossy levers on data that does not need them is precisely what
/// the no-thinning rule forbids, so the gate was right while the skew existed.
///
/// It no longer exists. Both advisors now measure baseline AND trials through
/// [`measure::measure_sample_layout`] under
/// `SyntheticLayout::from_density(&result.density)`, and a mismatched baseline
/// is no longer merely discouraged — [`measure::SyntheticLayout::produced`]
/// detects it from the measurement's own `(features, tiles)` pair and
/// [`measure::baseline_under`] re-measures rather than compare across cuts.
/// A test pins the outcome empirically: the quantization advisories fire on
/// exactly the same datasets, with the same flags, as they did under the
/// all-single-tile path
/// (`advisors::quantize::tests::the_layout_migration_did_not_move_which_datasets_fire`).
///
/// [`MeasurementMode::SingleTile`] stays reachable as the rollback.
pub fn analyze_source_with(
    source: &DataSource,
    mode: MeasurementMode,
) -> Result<(analysis::AnalysisResult, loader::LoadedData)> {
    let data = loader::load_data(source)?;
    let spatial = analysis::spatial::analyze(&data)?;
    let temporal = analysis::temporal::analyze(&data)?;
    let geometry = analysis::geometry::analyze(&data)?;
    // Measured sample encoding at build defaults; None (formula fallback) when
    // the sample is too small.
    let measured = match mode {
        // No layout needed, so the occupancy pre-pass is skipped entirely and
        // this mode costs exactly what it costs today.
        MeasurementMode::SingleTile => {
            measure::measure_sample(&data.sample, &measure::MeasureSettings::default())?
        }
        MeasurementMode::DensityLayout => {
            // Occupancy first: nothing here depends on the measurement that
            // depends on it.
            let occupancy = analysis::density::analyze(&data, &spatial, &temporal, None)?;
            let layout = measure::SyntheticLayout::from_density(&occupancy);
            measure::measure_sample_layout(
                &data.sample,
                &measure::MeasureSettings::default(),
                &layout,
            )?
        }
    };
    let density = analysis::density::analyze(&data, &spatial, &temporal, measured.as_ref())?;
    let result = analysis::AnalysisResult {
        source: source.display_name(),
        feature_count: data.features.len(),
        bounds: data.bounds,
        spatial,
        temporal,
        geometry,
        density,
        measured,
    };
    Ok((result, data))
}

/// What a recommendation run should solve for, beyond the defaults.
///
/// `Default` is the historical request — analyse, advise, iterate — so
/// [`recommend_for`] and [`recommend_with`] at the default are the same call.
/// Every field is additive and optional by construction: a consumer that does
/// not know about a lever gets exactly today's answer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecommendOptions {
    /// Solve the recipe against a target archive size in bytes
    /// (`--target-size`), attaching [`Recommendations::budget`].
    ///
    /// ⚠️ A budget changes WHICH REVERSIBLE LEVERS are chosen. It can never
    /// make the answer lossy: the solver's feasible set Θ₀ excludes the
    /// quantization family structurally (a [`ChosenLever`](budget_solver::ChosenLever)
    /// has no `lossy` field), and the lossy levers surface only as
    /// [`ShadowPrice`](budget_solver::ShadowPrice)s, which no automated path
    /// reads. An unreachable budget reports the lexicographic floor and drops
    /// nothing.
    pub target_size: Option<u64>,
}

/// Analyze a GeoParquet or STT input and produce build recommendations.
///
/// Runs the advisors [`advisors::RECOMMEND_ROUNDS`]-times-refined: the round-0
/// pass, then coordinate descent that measures the recipe those advisories
/// COMPOSE into and re-prices each measured lever against it. The refinement
/// changes numbers, never admissibility — lossy and `suggestion_only` advice is
/// classified exactly as before, and `stt-build --auto` consumes the same
/// `Recommendations` shape it always did.
pub fn recommend_for(source: &DataSource) -> Result<Recommendations> {
    recommend_with(source, &RecommendOptions::default())
}

/// [`recommend_for`] with an explicit refinement-round count.
///
/// `rounds = 0` is the documented ROLLBACK: it reproduces the historical single
/// pass exactly — the same advice, field for field, at the same cost — and
/// publishes no composed measurement. Use it if iteration's extra encode work
/// (roughly 5–10× the single pass, reported in the composed figure) is not
/// affordable on a given input.
pub fn recommend_for_with_rounds(source: &DataSource, rounds: usize) -> Result<Recommendations> {
    recommend_with_rounds(source, &RecommendOptions::default(), rounds)
}

/// THE recipe entry point, with options — what `stt-build --target-size` calls.
///
/// Identical to [`recommend_for`] when `opts` is the default. With
/// `opts.target_size` set it runs the same analysis and the same advisor
/// iteration and then solves [`budget_solver::solve`] on top, attaching the
/// answer as [`Recommendations::budget`].
///
/// Having ONE library function do this is the point: `stt-build --auto`, the
/// `stt-optimize recommend` CLI and the MCP surface must never be able to drift
/// into asking for different recipes.
pub fn recommend_with(source: &DataSource, opts: &RecommendOptions) -> Result<Recommendations> {
    recommend_with_rounds(source, opts, advisors::RECOMMEND_ROUNDS)
}

/// [`recommend_with`] with an explicit refinement-round count.
pub fn recommend_with_rounds(
    source: &DataSource,
    opts: &RecommendOptions,
    rounds: usize,
) -> Result<Recommendations> {
    let (result, data) = analyze_source(source)?;
    // The run's ONE layout, reconstructed from the density occupancy scan —
    // the MO-4 invariant that makes every trial delta attributable to its
    // lever rather than to the tile cut. Under the default
    // `MeasurementMode::DensityLayout`, `result.measured` is itself this
    // layout's measurement and the round-0 advisors reconstruct the same
    // layout from the same `result.density`, so every arm of every comparison
    // — round 0's and the iteration's alike — shares one cut.
    let layout = measure::SyntheticLayout::from_density(&result.density);
    // Evidence-based flag advisors (quantize, temporal, layout, budget) —
    // attached to the recommendations; lossy entries are surface-only.
    let iterative = advisors::run_iterative(&result, &data, &layout, rounds)?;
    // The budget rides ON TOP of the finished advisor pass, never instead of
    // it: the solver re-measures the frontier through the same oracle and the
    // same layout, and `generate_recommendations_budgeted` is what makes its
    // verdict authoritative over the advisors on the flags it owns.
    let budget = match opts.target_size {
        Some(target_bytes) => Some(budget_solver::solve(&result, &data, &layout, target_bytes)?),
        None => None,
    };
    Ok(recommend::generate_recommendations_budgeted(
        &result,
        iterative.advice,
        iterative.composed.as_ref(),
        budget,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    /// A GeoParquet of `n` points spread over a small region, enough rows that
    /// the loader's sample clears `MIN_MEASURE_FEATURES` and the density scan
    /// sees many occupied cells.
    fn point_parquet(dir: &std::path::Path, n: usize) -> std::path::PathBuf {
        let wkbs: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                // Little-endian WKB point.
                let mut v = vec![0x01, 0x01, 0x00, 0x00, 0x00];
                let lon = -73.0 + (i % 40) as f64 * 0.01;
                let lat = 45.0 + (i / 40) as f64 * 0.01;
                v.extend_from_slice(&lon.to_le_bytes());
                v.extend_from_slice(&lat.to_le_bytes());
                v
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(
                    wkbs.iter().map(|v| v.as_slice()),
                )),
                Arc::new(Int64Array::from(
                    (0..n as i64)
                        .map(|i| 1_600_000_000_000 + i * 60_000)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    (0..n).map(|i| i as f64 * 0.137).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    (0..n).map(|i| format!("cat-{}", i % 5)).collect::<Vec<_>>(),
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

    fn source_at(path: std::path::PathBuf) -> DataSource {
        DataSource::GeoParquet {
            path,
            time_field: "timestamp".to_string(),
            time_format: "unix-ms".to_string(),
        }
    }

    #[test]
    fn measurement_modes_agree_on_everything_except_the_tile_cut() {
        let dir = tempfile::tempdir().unwrap();
        let source = source_at(point_parquet(dir.path(), 2_000));

        let (single, _) = analyze_source_with(&source, MeasurementMode::SingleTile).unwrap();
        let (layout, _) = analyze_source_with(&source, MeasurementMode::DensityLayout).unwrap();

        let single_m = single.measured.as_ref().expect("sample is measurable");
        let layout_m = layout.measured.as_ref().expect("sample is measurable");

        // The default is DensityLayout: the gate that held it at SingleTile
        // (advisors baselining on a layout while trialling single-tile) is
        // closed, so a default analysis measures at real tile occupancy.
        let (default_mode, _) = analyze_source(&source).unwrap();
        let default_m = default_mode.measured.as_ref().unwrap();
        assert_eq!(default_m.tiles, layout_m.tiles);
        assert_eq!(default_m.bytes_total, layout_m.bytes_total);
        assert_eq!(single_m.tiles, 1, "SingleTile stays the rollback");
        assert!(layout_m.tiles > 1, "layout mode must split: {layout_m:?}");

        // Same features encoded either way — the layout changes the CUT, never
        // what is measured (no thinning, no sampling change).
        assert_eq!(single_m.features, layout_m.features);
        assert_eq!(single_m.geometry_kind, layout_m.geometry_kind);
        // Per-tile framing is real, so the honest cut costs more bytes.
        assert!(
            layout_m.bytes_total > single_m.bytes_total,
            "layout {} vs single {}",
            layout_m.bytes_total,
            single_m.bytes_total
        );

        // Occupancy statistics are identical across modes: the layout-mode
        // pre-pass and the final calibrated pass see the same cells, so nothing
        // downstream can tell which pass produced them.
        assert_eq!(single.density.per_zoom.len(), layout.density.per_zoom.len());
        for (a, b) in single
            .density
            .per_zoom
            .iter()
            .zip(layout.density.per_zoom.iter())
        {
            assert_eq!(a.zoom, b.zoom);
            assert_eq!(a.tile_count, b.tile_count);
            assert_eq!(a.median_features_per_tile, b.median_features_per_tile);
            assert_eq!(a.p95_features_per_tile, b.p95_features_per_tile);
            assert_eq!(a.max_features_per_tile, b.max_features_per_tile);
        }
    }

    #[test]
    fn the_run_layout_is_reconstructible_from_the_finished_result() {
        // THE MO-4 INVARIANT. A consumer holding only the finished
        // AnalysisResult must be able to rebuild the exact layout the
        // measurement was taken under — otherwise its trial encodes are not
        // comparable to the baseline.
        let dir = tempfile::tempdir().unwrap();
        let source = source_at(point_parquet(dir.path(), 2_000));
        let (result, data) = analyze_source_with(&source, MeasurementMode::DensityLayout).unwrap();

        let reconstructed = measure::SyntheticLayout::from_density(&result.density);
        let rerun = measure::measure_sample_layout(
            &data.sample,
            &measure::MeasureSettings::default(),
            &reconstructed,
        )
        .unwrap()
        .unwrap();

        let baseline = result.measured.as_ref().unwrap();
        assert_eq!(rerun.tiles, baseline.tiles);
        assert_eq!(rerun.bytes_total, baseline.bytes_total);
        assert_eq!(
            serde_json::to_string(&rerun).unwrap(),
            serde_json::to_string(baseline).unwrap(),
            "the reconstructed layout must reproduce the run's measurement byte for byte"
        );
    }

    // ------------------------------------------------------------------
    // MO-9: the `recommend_with` handshake `stt-build --target-size` uses
    // ------------------------------------------------------------------

    #[test]
    fn the_default_options_are_the_historical_call_exactly() {
        // `recommend_for` IS `recommend_with(.., &default())`. If that ever
        // stops being true, every consumer that did not opt into a budget has
        // silently changed behaviour.
        let dir = tempfile::tempdir().unwrap();
        let source = source_at(point_parquet(dir.path(), 800));

        let plain = recommend_for(&source).unwrap();
        let with_defaults = recommend_with(&source, &RecommendOptions::default()).unwrap();
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            serde_json::to_string(&with_defaults).unwrap(),
        );
        assert!(
            plain.budget.is_none(),
            "no target size was asked for, so no budget may be attached"
        );
    }

    #[test]
    fn a_target_size_attaches_a_budget_and_never_a_lossy_flag() {
        // The no-thinning guard at the LIBRARY boundary (the binary boundary is
        // `spatiotemporal-tiles/tests/build_auto_target.rs`). A budget may pick
        // any reversible lever it likes; it may not put a quantization flag in
        // front of an automated consumer.
        let dir = tempfile::tempdir().unwrap();
        let path = point_parquet(dir.path(), 2_000);
        let source = source_at(path.clone());

        let unconstrained = recommend_for(&source).unwrap();
        // A budget tight enough to force real work out of the solver.
        let opts = RecommendOptions {
            target_size: Some(4_096),
        };
        let budgeted = recommend_with(&source, &opts).unwrap();

        let budget = budgeted.budget.as_ref().expect("a target size was given");
        assert_eq!(budget.target_bytes, 4_096);
        // Every chosen lever is a Θ₀ lever, by name.
        for lever in &budget.chosen {
            assert!(
                matches!(
                    lever.flag.as_str(),
                    "--min-zoom"
                        | "--max-zoom"
                        | "--temporal-bucket"
                        | "--temporal-lod"
                        | "--publish"
                        | "--zstd-level"
                        | "--blob-ordering"
                        | "--pack-size"
                ),
                "a lever outside Θ₀ reached the recipe: {lever:?}"
            );
        }
        // The lossy family appears ONLY as shadow prices, and every entry says
        // so in its own serialized form.
        for price in &budget.shadow_prices {
            assert!(price.lossy, "a shadow price that is not lossy: {price:?}");
        }
        // …and none of them reach the emitted command.
        let command = recommend::to_command(&budgeted, &path, "timestamp");
        for lossy in [
            "--quantize-coords",
            "--quantize-attrs-auto",
            "--quantize-attr",
            "--max-features-per-tile",
            "--max-tile-bytes",
        ] {
            assert!(
                !command.contains(lossy),
                "budget pressure put `{lossy}` in the command:\n{command}"
            );
        }
        // The unbudgeted answer is unchanged by the existence of budget mode.
        assert!(unconstrained.budget.is_none());
    }

    #[test]
    fn two_budgeted_recommendations_over_one_input_are_byte_identical() {
        // THE determinism rule for a solver change: same input ⇒ same
        // recommendation. `stt-build --target-size` folds this answer into its
        // args, and pack names are content-addressed, so a wobble here is a
        // fleet-wide re-upload.
        let dir = tempfile::tempdir().unwrap();
        let source = source_at(point_parquet(dir.path(), 1_200));
        let opts = RecommendOptions {
            target_size: Some(64_000),
        };
        let a = recommend_with(&source, &opts).unwrap();
        let b = recommend_with(&source, &opts).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
            "two budgeted runs over one file must serialise identically"
        );
    }

    #[test]
    fn analysis_is_deterministic_across_runs_in_both_modes() {
        let dir = tempfile::tempdir().unwrap();
        let source = source_at(point_parquet(dir.path(), 1_200));
        for mode in [MeasurementMode::SingleTile, MeasurementMode::DensityLayout] {
            let a = analyze_source_with(&source, mode).unwrap().0;
            let b = analyze_source_with(&source, mode).unwrap().0;
            assert_eq!(
                serde_json::to_string(&a.measured).unwrap(),
                serde_json::to_string(&b.measured).unwrap(),
                "mode {mode:?}"
            );
        }
    }
}
