//! Parameter recommendation engine
//!
//! Combines analysis results to generate optimal stt-build parameters.

use crate::advisors::{Advice, ComposedMeasurement};
use crate::analysis::AnalysisResult;
use crate::budget_solver::{BudgetReport, ChosenLever, BUDGET_GOVERNED_FLAGS};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Recommended parameters for stt-build.
///
/// `stt-build --auto` folds `min_zoom`, `max_zoom`, and `temporal_bucket_ms`
/// into its own args and logs `confidence`/`explanations` (`--auto encode`
/// additionally applies the non-lossy byte-level entries of `advice`); all
/// fields also appear in the standalone `analyze`/`recommend` reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendations {
    /// Minimum zoom level
    pub min_zoom: u8,
    /// Maximum zoom level
    pub max_zoom: u8,
    /// Suggested temporal bucket size in milliseconds (0 = no bucketing)
    pub temporal_bucket_ms: u64,
    /// Human-readable temporal bucket description
    pub temporal_bucket_human: String,
    /// Confidence level in recommendations (0-100)
    pub confidence: u8,
    /// Dominant source geometry type (from [`crate::analysis::geometry`],
    /// e.g. `"Point"`/`"LineString"`/`"Polygon"`). Carried through so the
    /// AI surface can pick a matching render layer instead of guessing.
    #[serde(default)]
    pub dominant_type: String,
    /// Explanation of key decisions
    pub explanations: Vec<String>,
    /// Evidence-based advisor suggestions ([`crate::advisors::run_all`]),
    /// in advisor order (quantize, temporal, layout, budget). Entries with
    /// `lossy: true` are surfaced only — they never join [`to_command`] and
    /// are never auto-applied by `stt-build --auto`.
    #[serde(default)]
    pub advice: Vec<Advice>,
    /// What the composed NON-LOSSY recipe measures on the sample, versus build
    /// defaults, when [`crate::advisors::run_iterative`] measured it
    /// ([`ComposedMeasurement::projected`]).
    ///
    /// `None` when nothing was composed: the single-pass rollback (`rounds = 0`)
    /// or a sample below the measurement floor. This is the first number in the
    /// pipeline that describes the RECIPE rather than one lever at a time — the
    /// single pass never measured the combination at all.
    #[serde(default)]
    pub composed_projected: Option<String>,
    /// The parallel with-lossy figure
    /// ([`ComposedMeasurement::projected_with_lossy`]): what opting into the
    /// lossy advisories on top of the recipe would measure.
    ///
    /// ⚠️ REPORT ONLY. Its presence is not a recommendation and nothing
    /// downstream may read it as one — lossy levers stay out of [`to_command`]
    /// and out of `stt-build --auto` regardless of what this says.
    #[serde(default)]
    pub composed_projected_with_lossy: Option<String>,
    /// The answer to `--target-size B`, when a budget was requested.
    ///
    /// `None` — and absent from the serialized form — whenever no target size
    /// was given, so the whole pipeline is byte-for-byte what it was before the
    /// budget solver existed.
    ///
    /// When present it is authoritative over the flags it names: the solver
    /// measured the recipe against a byte budget, so [`to_command`] takes its
    /// zoom/bucket scalars from [`BudgetReport::chosen`] and drops the advisor's
    /// verdict on the flags in
    /// [`BUDGET_GOVERNED_FLAGS`](crate::budget_solver::BUDGET_GOVERNED_FLAGS) in
    /// favour of the solver's.
    ///
    /// ⚠️ It cannot make the command lossy. [`ChosenLever`] has no `lossy` field
    /// — a lossy lever is not representable as a chosen one — and
    /// [`BudgetReport::shadow_prices`], which is where the lossy levers live, is
    /// never consulted by [`to_command`] at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetReport>,
}

/// Generate recommendations from analysis results, attaching the advisor
/// suggestions (pass an empty `Vec` when the advisors were not run).
///
/// The single-pass form: no composed measurement, so `composed_projected` stays
/// `None`. Callers that ran [`crate::advisors::run_iterative`] should use
/// [`generate_recommendations_composed`] instead.
pub fn generate_recommendations(result: &AnalysisResult, advice: Vec<Advice>) -> Recommendations {
    generate_recommendations_composed(result, advice, None)
}

/// Generate recommendations, attaching the advisor suggestions AND the composed
/// recipe measurement.
///
/// The composed figures ride two places: the dedicated
/// `composed_projected` fields, and an `explanations` line — which is what
/// surfaces them in `stt-optimize recommend --explain` and in `stt-build
/// --auto`'s log, where the cost of having measured them is also worth seeing.
pub fn generate_recommendations_composed(
    result: &AnalysisResult,
    advice: Vec<Advice>,
    composed: Option<&ComposedMeasurement>,
) -> Recommendations {
    generate_recommendations_budgeted(result, advice, composed, None)
}

/// [`generate_recommendations_composed`] with the `--target-size` solver's
/// answer attached.
///
/// `budget: None` reproduces `generate_recommendations_composed` exactly — the
/// documented rollback for the whole budget mechanism is simply not passing a
/// target size.
pub fn generate_recommendations_budgeted(
    result: &AnalysisResult,
    advice: Vec<Advice>,
    composed: Option<&ComposedMeasurement>,
    budget: Option<BudgetReport>,
) -> Recommendations {
    let mut explanations = Vec::new();

    // Zoom levels from spatial analysis
    let min_zoom = result.spatial.recommended_min_zoom;
    let max_zoom = result.spatial.recommended_max_zoom;
    explanations.push(format!(
        "Zoom range {}-{} based on spatial coverage ({})",
        min_zoom, max_zoom, result.spatial.distribution
    ));

    // Temporal bucketing
    let temporal_bucket_ms = result.temporal.recommended_bucket_ms;
    let temporal_bucket_human = result.temporal.recommended_bucket_human.clone();
    if temporal_bucket_ms > 0 {
        explanations.push(format!(
            "Temporal bucket {} for {} distribution",
            temporal_bucket_human, result.temporal.distribution
        ));
    }

    // Calculate confidence based on data quality
    let confidence = calculate_confidence(result);

    // The composed recipe, when it was measured. The explanation line carries
    // the measurement AND its cost: iteration buys the interaction between
    // levers by spending real encode time, and a slow `recommend` should say
    // why it was slow.
    let composed_projected = composed.map(ComposedMeasurement::projected);
    let composed_projected_with_lossy =
        composed.and_then(ComposedMeasurement::projected_with_lossy);
    if let Some(line) = &composed_projected {
        explanations.push(format!("Composed recipe: {line}"));
    }
    if let Some(line) = &composed_projected_with_lossy {
        explanations.push(format!("Lossy what-if (not applied): {line}"));
    }
    // The budget verdict rides the human-facing explanation list too, so a
    // reader of `--auto`'s log or the analyze report sees whether the target was
    // met without having to parse the report struct.
    if let Some(report) = &budget {
        explanations.push(format!("Budget (--target-size): {}", report.headline()));
    }

    Recommendations {
        min_zoom,
        max_zoom,
        temporal_bucket_ms,
        temporal_bucket_human,
        confidence,
        dominant_type: result.geometry.dominant_type.clone(),
        explanations,
        advice,
        composed_projected,
        composed_projected_with_lossy,
        budget,
    }
}

/// Calculate confidence score (0-100)
fn calculate_confidence(result: &AnalysisResult) -> u8 {
    let mut score = 100u8;

    // Lower confidence with fewer features
    if result.feature_count < 1000 {
        score = score.saturating_sub(20);
    } else if result.feature_count < 10000 {
        score = score.saturating_sub(10);
    }

    // Lower confidence with many issues
    if result.density.issues.len() > 3 {
        score = score.saturating_sub(15);
    } else if !result.density.issues.is_empty() {
        score = score.saturating_sub(5);
    }

    // Lower confidence with sparse data
    if matches!(
        result.spatial.distribution,
        crate::analysis::spatial::SpatialDistribution::Sparse
    ) {
        score = score.saturating_sub(10);
    }

    // Lower confidence with complex geometry
    if matches!(
        result.geometry.complexity,
        crate::analysis::geometry::GeometryComplexity::VeryComplex
    ) {
        score = score.saturating_sub(10);
    }

    score
}

/// Convert recommendations to a build config JSON structure.
///
/// Carries the scalar recipe (`min_zoom`/`max_zoom`/`temporal_bucket_ms`/
/// `confidence`/`explanations`) alongside the richer signals a downstream
/// consumer (the MCP `recommend_build` tool) would otherwise recompute or do
/// without: `temporal_bucket_human`, `dominant_type`, the full evidence-based
/// `advice` array (with its `lossy`/`confidence` markers), and the ready-to-run
/// `command` assembled by [`to_command`] (non-lossy advisor flags already
/// appended). The shape is append-only: consumers pin the scalar keys, so new
/// signals go alongside them and never reshape an existing key.
pub fn to_build_config(
    recommendations: &Recommendations,
    input: &Path,
    time_field: &str,
) -> serde_json::Value {
    let mut config = serde_json::json!({
        "input": input.to_string_lossy(),
        "time_field": time_field,
        "min_zoom": recommendations.min_zoom,
        "max_zoom": recommendations.max_zoom,
        "temporal_bucket_ms": recommendations.temporal_bucket_ms,
        "temporal_bucket_human": recommendations.temporal_bucket_human,
        "dominant_type": recommendations.dominant_type,
        "confidence": recommendations.confidence,
        "explanations": recommendations.explanations,
        "advice": recommendations.advice,
        // Recipe-level measurement (null on the single-pass rollback). The
        // with-lossy figure is a what-if for a human, never an instruction.
        "composed_projected": recommendations.composed_projected,
        "composed_projected_with_lossy": recommendations.composed_projected_with_lossy,
        "command": to_command(recommendations, input, time_field),
    });
    // The budget report is INSERTED rather than declared in the literal so a run
    // without `--target-size` emits the exact same JSON it always did — no new
    // null key, no shape change for existing consumers.
    if let Some(report) = &recommendations.budget {
        if let (Some(object), Ok(value)) = (config.as_object_mut(), serde_json::to_value(report)) {
            object.insert("budget".to_string(), value);
        }
    }
    config
}

/// The scalar recipe the command should carry: the analysis recommendation,
/// overridden by the budget solver's own choices where it made any.
///
/// The solver only emits `--max-zoom` / `--temporal-bucket` when it actually
/// MOVED them, so an unbudgeted run and a budget that needed no distortion
/// produce the same command.
fn command_scalars(recommendations: &Recommendations) -> (u8, u8, u64) {
    let mut min_zoom = recommendations.min_zoom;
    let mut max_zoom = recommendations.max_zoom;
    let mut bucket_ms = recommendations.temporal_bucket_ms;
    let Some(budget) = &recommendations.budget else {
        return (min_zoom, max_zoom, bucket_ms);
    };
    for lever in budget.chosen.iter().filter(|l| !l.suggestion_only) {
        let Some(value) = lever.value.as_deref() else {
            continue;
        };
        match lever.flag.as_str() {
            "--min-zoom" => min_zoom = value.parse().unwrap_or(min_zoom),
            "--max-zoom" => max_zoom = value.parse().unwrap_or(max_zoom),
            "--temporal-bucket" => bucket_ms = value.parse().unwrap_or(bucket_ms),
            _ => {}
        }
    }
    (min_zoom, max_zoom, bucket_ms)
}

/// The budget's non-scalar levers, in solver order, filtered by the SAME
/// admissibility predicate the advisor list is filtered by.
///
/// A [`ChosenLever`] cannot be lossy (the type has no such field), so the
/// predicate reduces to the `suggestion_only` half — which is exactly what keeps
/// a playback-caveated `spatial` blob ordering out of the command no matter how
/// tight the budget got.
fn command_budget_levers(recommendations: &Recommendations) -> Vec<&ChosenLever> {
    let Some(budget) = &recommendations.budget else {
        return Vec::new();
    };
    budget
        .chosen
        .iter()
        .filter(|lever| !lever.suggestion_only)
        .filter(|lever| {
            !matches!(
                lever.flag.as_str(),
                "--min-zoom" | "--max-zoom" | "--temporal-bucket"
            )
        })
        .collect()
}

/// Convert recommendations to an stt-build command line.
///
/// NON-LOSSY advisor advice is appended (flag + value, in advisor order) so
/// the pasteable command carries the reversible byte-level/semantic levers.
/// LOSSY advice (quantization, budgets) NEVER joins the command — it degrades
/// data and stays a per-dataset opt-in the user must add by hand. The same
/// goes for `suggestion_only` advice: non-lossy, but the tradeoff spelled out
/// in its `why` needs a human decision before the flag is safe to run.
///
/// # With a `--target-size` budget attached
///
/// When [`Recommendations::budget`] is present the solver's recipe is
/// authoritative over the flags it owns: the zoom/bucket scalars come from
/// [`crate::budget_solver::BudgetReport::chosen`], and the advisor's opinion on
/// the flags in [`BUDGET_GOVERNED_FLAGS`] is dropped in favour of the solver's —
/// otherwise the command could carry `--publish` (zstd 19) while the budget was
/// projected at level 3, and a user would be shown a number the build then
/// silently undershoots.
///
/// **The lossy filter below is unchanged and unweakened.** It is the same
/// expression it always was, and the budget path adds nothing to it that could
/// let a lossy lever through: lossy levers are not representable as chosen ones,
/// and the budget's shadow-price table is never read here.
pub fn to_command(recommendations: &Recommendations, input: &Path, time_field: &str) -> String {
    // The input exactly as the caller addressed it (pasteable from the same
    // cwd); a basename would break the command from anywhere else. Suggest the
    // packed dataset DIRECTORY (the input's stem): stt-build's output is a
    // directory tree, not a single file.
    let output = input.with_extension("");
    let output_str = if output.as_os_str().is_empty() {
        "output".to_string()
    } else {
        output.to_string_lossy().to_string()
    };
    let input_str = input.to_string_lossy();
    let (min_zoom, max_zoom, temporal_bucket_ms) = command_scalars(recommendations);

    let mut cmd = format!(
        "stt-build --input {} --output {} \\\n  --time-field {} --min-zoom {} --max-zoom {}",
        input_str, output_str, time_field, min_zoom, max_zoom,
    );
    // The recommended bucket is the recipe's core scalar — the command must
    // carry it or a paste-and-run build silently falls back to the 1h default.
    // Bare milliseconds: always valid `parse_duration` input.
    if temporal_bucket_ms > 0 {
        cmd.push_str(&format!(" \\\n  --temporal-bucket {}", temporal_bucket_ms));
    }
    // Flags the budget solver owns once a target size is set — the advisor's
    // verdict on these is superseded, not merged.
    let superseded: &[&str] = if recommendations.budget.is_some() {
        BUDGET_GOVERNED_FLAGS
    } else {
        &[]
    };
    // Stable order: exactly the advisor emit order (quantize, temporal,
    // layout, budget), filtered to the auto-applicable entries.
    for advice in recommendations
        .advice
        .iter()
        .filter(|a| !a.lossy && !a.suggestion_only)
        .filter(|a| !superseded.contains(&a.flag.as_str()))
    {
        cmd.push_str(" \\\n  ");
        cmd.push_str(&advice.flag);
        if let Some(value) = &advice.value {
            cmd.push(' ');
            cmd.push_str(value);
        }
    }
    // …then the budget's own levers, in solver order. Empty without a budget,
    // so an unbudgeted command is byte-for-byte what it always was.
    for lever in command_budget_levers(recommendations) {
        cmd.push_str(" \\\n  ");
        cmd.push_str(&lever.flag);
        if let Some(value) = &lever.value {
            cmd.push(' ');
            cmd.push_str(value);
        }
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisors::AdviceConfidence;
    use crate::budget_solver::{BudgetReport, ChosenLever};

    /// Minimal synthetic analysis result — just enough populated fields for
    /// `generate_recommendations`/`calculate_confidence` to run. The shared
    /// default fixture is exactly what these tests need.
    fn synthetic_result() -> AnalysisResult {
        crate::test_support::sample_analysis()
    }

    fn advice(flag: &str, value: Option<&str>, lossy: bool) -> Advice {
        Advice {
            flag: flag.to_string(),
            value: value.map(str::to_string),
            why: format!("{flag}: synthetic rationale"),
            projected: None,
            lossy,
            suggestion_only: false,
            confidence: AdviceConfidence::Medium,
        }
    }

    fn rec_with_advice(advice: Vec<Advice>) -> Recommendations {
        Recommendations {
            min_zoom: 0,
            max_zoom: 10,
            temporal_bucket_ms: 3_600_000,
            temporal_bucket_human: "1 hour".to_string(),
            confidence: 85,
            dominant_type: "Point".to_string(),
            explanations: vec![],
            advice,
            composed_projected: None,
            composed_projected_with_lossy: None,
            budget: None,
        }
    }

    /// A chosen lever, as the budget solver would emit one.
    fn chosen(flag: &str, value: Option<&str>) -> ChosenLever {
        ChosenLever {
            flag: flag.to_string(),
            value: value.map(str::to_string),
            why: format!("{flag}: synthetic budget rationale"),
            delta_bytes: Some(-1_000),
            suggestion_only: false,
        }
    }

    /// A budget report carrying `chosen`, feasible at an arbitrary size.
    fn budget_with(chosen: Vec<ChosenLever>, feasible: bool) -> BudgetReport {
        BudgetReport {
            target_bytes: 1_000_000,
            projected_bytes: if feasible { 900_000 } else { 1_500_000 },
            projected_stderr: 0.0,
            feasible,
            within_noise: false,
            distortion: crate::budget_solver::DistortionClass::none(),
            chosen,
            floor_bytes: 800_000,
            floor_distortion: crate::budget_solver::DistortionClass::none(),
            shadow_prices: vec![crate::budget_solver::ShadowPrice {
                flag: "--quantize-coords".to_string(),
                value: Some("1".to_string()),
                marginal_bytes: 400_000,
                delta_frac: -0.4,
                stderr: 0.01,
                lossy: true,
                why: "synthetic shadow price".to_string(),
            }],
            zstd_sweep: vec![3, 9, 19],
            classes_evaluated: 2,
            basis: crate::budget_solver::EstimateBasis::Measured,
            notes: vec!["synthetic".to_string()],
        }
    }

    /// A composed measurement over a recipe that shrinks the sample.
    fn composed(with_lossy: Option<usize>) -> ComposedMeasurement {
        ComposedMeasurement {
            rounds: 2,
            settings: crate::measure::MeasureSettings {
                zstd_level: 19,
                ..crate::measure::MeasureSettings::default()
            },
            features: 800,
            composed_bytes: 9_000,
            default_bytes: 10_000,
            with_lossy_bytes: with_lossy,
            usage: crate::advisors::OracleUsage {
                measurements: 3,
                oracle_calls: 2,
                trials_priced: 3,
                cache_hits: 4,
            },
        }
    }

    #[test]
    fn test_to_command() {
        let rec = rec_with_advice(vec![]);
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        assert!(cmd.contains("--min-zoom 0"));
        assert!(cmd.contains("--max-zoom 10"));
        // The recipe's core scalar must ride the pasteable command, or a
        // paste-and-run build silently buckets at the 1h default.
        assert!(cmd.contains("--temporal-bucket 3600000"), "command: {cmd}");
    }

    #[test]
    fn to_command_keeps_caller_path_and_derives_output_stem() {
        let rec = rec_with_advice(vec![]);
        let cmd = to_command(&rec, Path::new("/data/in/quakes.parquet"), "timestamp");
        // Full path as addressed by the caller — a basename would break the
        // command pasted from any other cwd.
        assert!(
            cmd.contains("--input /data/in/quakes.parquet"),
            "command: {cmd}"
        );
        assert!(cmd.contains("--output /data/in/quakes"), "command: {cmd}");
    }

    #[test]
    fn suggestion_only_advice_never_joins_to_command() {
        let mut caveated = advice("--blob-ordering", Some("spatial"), false);
        caveated.suggestion_only = true;
        let mut semantic = advice("--min-zoom-field", Some("category"), false);
        semantic.suggestion_only = true;
        let rec = rec_with_advice(vec![caveated, semantic, advice("--publish", None, false)]);
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        // Suggestion-only levers stay out of the auto path even though they
        // are non-lossy — their `why` carries a decision the user must make.
        assert!(!cmd.contains("--blob-ordering"), "command: {cmd}");
        assert!(!cmd.contains("--min-zoom-field"), "command: {cmd}");
        assert!(cmd.contains("--publish"), "command: {cmd}");
    }

    #[test]
    fn generate_recommendations_attaches_advice() {
        let result = synthetic_result();
        let rec = generate_recommendations(
            &result,
            vec![
                advice("--publish", None, false),
                advice("--quantize-coords", Some("1"), true),
            ],
        );
        assert_eq!(rec.advice.len(), 2);
        assert_eq!(rec.advice[0].flag, "--publish");
        assert_eq!(rec.advice[1].flag, "--quantize-coords");
        // The advisor layer must not disturb the basic recommendations.
        assert_eq!(rec.min_zoom, 0);
        assert_eq!(rec.max_zoom, 10);
        assert_eq!(rec.temporal_bucket_ms, 3_600_000);
    }

    #[test]
    fn to_command_appends_only_non_lossy_advice_in_stable_order() {
        let rec = rec_with_advice(vec![
            advice("--quantize-coords", Some("1"), true), // lossy: excluded
            advice("--temporal-lod", Some("1d,30d"), false),
            advice("--publish", None, false),
            advice("--blob-ordering", Some("spatial"), false),
            advice("--maximum-tile-features", Some("10000"), true), // lossy: excluded
        ]);
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");

        // Non-lossy advice present, with values, in advisor (input) order.
        let lod = cmd
            .find("--temporal-lod 1d,30d")
            .expect("temporal-lod in command");
        let publish = cmd.find("--publish").expect("publish in command");
        let ordering = cmd
            .find("--blob-ordering spatial")
            .expect("blob-ordering in command");
        assert!(
            lod < publish && publish < ordering,
            "advisor order preserved: {cmd}"
        );
    }

    #[test]
    fn to_build_config_carries_advice_command_and_geometry() {
        let rec = rec_with_advice(vec![
            advice("--blob-ordering", Some("spatial"), false),
            advice("--quantize-coords", Some("1"), true),
        ]);
        let config = to_build_config(&rec, Path::new("data.parquet"), "timestamp");

        // Original scalar recipe still present and unchanged.
        assert_eq!(config["min_zoom"], 0);
        assert_eq!(config["max_zoom"], 10);
        assert_eq!(config["temporal_bucket_ms"], 3_600_000);
        assert_eq!(config["confidence"], 85);

        // Enriched signals now reach the JSON.
        assert_eq!(config["temporal_bucket_human"], "1 hour");
        assert_eq!(config["dominant_type"], "Point");
        assert_eq!(config["advice"].as_array().expect("advice array").len(), 2);
        assert_eq!(config["advice"][0]["flag"], "--blob-ordering");
        assert_eq!(config["advice"][1]["lossy"], true);
        let command = config["command"].as_str().expect("command string");
        assert!(command.contains("--min-zoom 0"), "command: {command}");
        // Non-lossy advice rides the command; lossy quantize does not.
        assert!(
            command.contains("--blob-ordering spatial"),
            "command: {command}"
        );
        assert!(!command.contains("--quantize-coords"), "command: {command}");
    }

    #[test]
    fn composed_measurement_rides_the_recommendations_and_the_config() {
        let result = synthetic_result();
        let rec = generate_recommendations_composed(
            &result,
            vec![advice("--publish", None, false)],
            Some(&composed(Some(6_000))),
        );

        let projected = rec.composed_projected.as_deref().expect("composed figure");
        // The recipe's own bytes, the reference it beats, and the cost of
        // having measured it.
        assert!(projected.contains("9000 B"), "{projected}");
        assert!(projected.contains("10000 B"), "{projected}");
        assert!(projected.contains("800-feature sample"), "{projected}");
        assert!(projected.contains("-10.0%"), "{projected}");
        assert!(projected.contains("2 refinement rounds"), "{projected}");
        assert!(projected.contains("cache hits"), "{projected}");

        // The lossy what-if is present, and says loudly that it is not applied.
        let lossy = rec
            .composed_projected_with_lossy
            .as_deref()
            .expect("lossy what-if");
        assert!(lossy.contains("6000 B"), "{lossy}");
        assert!(lossy.contains("never auto-applied"), "{lossy}");

        // Both surface in the explanations (what `--explain` and `--auto` log).
        assert!(rec
            .explanations
            .iter()
            .any(|e| e.starts_with("Composed recipe: ")));
        assert!(rec
            .explanations
            .iter()
            .any(|e| e.starts_with("Lossy what-if (not applied): ")));

        // …and in the build config JSON.
        let config = to_build_config(&rec, Path::new("data.parquet"), "timestamp");
        assert_eq!(config["composed_projected"], serde_json::json!(projected));
        assert_eq!(
            config["composed_projected_with_lossy"],
            serde_json::json!(lossy)
        );
    }

    #[test]
    fn single_pass_recommendations_carry_no_composed_figure() {
        // The rollback shape: `generate_recommendations` is the one-pass form,
        // so the composed keys are present-but-null rather than fabricated.
        let rec = generate_recommendations(&synthetic_result(), vec![]);
        assert!(rec.composed_projected.is_none());
        assert!(rec.composed_projected_with_lossy.is_none());
        assert!(
            !rec.explanations
                .iter()
                .any(|e| e.contains("Composed recipe")),
            "{:?}",
            rec.explanations
        );
        let config = to_build_config(&rec, Path::new("data.parquet"), "timestamp");
        assert!(config["composed_projected"].is_null());

        // A composed measurement with no lossy advice publishes no what-if.
        let rec =
            generate_recommendations_composed(&synthetic_result(), vec![], Some(&composed(None)));
        assert!(rec.composed_projected.is_some());
        assert!(rec.composed_projected_with_lossy.is_none());
    }

    #[test]
    fn recommendations_deserialize_without_dominant_type() {
        // An older serialized Recommendations (pre-dominant_type) must still
        // deserialize thanks to #[serde(default)].
        let legacy = serde_json::json!({
            "min_zoom": 2,
            "max_zoom": 12,
            "temporal_bucket_ms": 60_000,
            "temporal_bucket_human": "1 minute",
            "confidence": 70,
            "explanations": ["legacy"],
            "advice": [],
        });
        let rec: Recommendations = serde_json::from_value(legacy).expect("legacy deserializes");
        assert_eq!(rec.dominant_type, "");
        assert_eq!(rec.max_zoom, 12);
    }

    #[test]
    fn without_a_budget_nothing_about_the_command_or_the_config_moves() {
        // The rollback for the whole `--target-size` mechanism is simply not
        // passing one, and it has to be byte-for-byte invisible.
        let rec = rec_with_advice(vec![
            advice("--publish", None, false),
            advice("--blob-ordering", Some("time-major"), false),
        ]);
        assert!(rec.budget.is_none());
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        assert!(cmd.contains("--min-zoom 0 --max-zoom 10"), "{cmd}");
        assert!(cmd.contains("--temporal-bucket 3600000"), "{cmd}");
        assert!(cmd.contains("--publish"), "{cmd}");
        assert!(cmd.contains("--blob-ordering time-major"), "{cmd}");

        // …and the config JSON gains no key at all (not even a null one).
        let config = to_build_config(&rec, Path::new("data.parquet"), "timestamp");
        assert!(
            config.get("budget").is_none(),
            "an unbudgeted config must not grow a `budget` key: {config}"
        );
        let serialized = serde_json::to_string(&rec).unwrap();
        assert!(
            !serialized.contains("\"budget\""),
            "Recommendations must not serialize an absent budget: {serialized}"
        );
    }

    #[test]
    fn a_budget_supersedes_the_scalars_and_the_flags_it_owns() {
        // The advisor wants publish-grade zstd; the budget solved at level 3 and
        // clamped the pyramid. The command must describe the BUDGET's recipe —
        // otherwise the user is shown a projection the build then undershoots.
        let mut rec = rec_with_advice(vec![
            advice("--publish", None, false),
            advice("--adaptive-temporal", Some("5000"), false),
        ]);
        rec.budget = Some(budget_with(
            vec![
                chosen("--max-zoom", Some("8")),
                chosen("--temporal-bucket", Some("14400000")),
                chosen("--temporal-lod", Some("1d@6")),
            ],
            true,
        ));
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");

        assert!(cmd.contains("--max-zoom 8"), "{cmd}");
        assert!(!cmd.contains("--max-zoom 10"), "{cmd}");
        assert!(cmd.contains("--temporal-bucket 14400000"), "{cmd}");
        assert!(!cmd.contains("--temporal-bucket 3600000"), "{cmd}");
        assert!(cmd.contains("--temporal-lod 1d@6"), "{cmd}");
        // The advisor's zstd verdict is superseded, not merged.
        assert!(!cmd.contains("--publish"), "{cmd}");
        // An advisory the budget does NOT own rides through untouched.
        assert!(cmd.contains("--adaptive-temporal 5000"), "{cmd}");

        // The report reaches the config JSON for `stt-build --auto` to fold.
        let config = to_build_config(&rec, Path::new("data.parquet"), "timestamp");
        assert_eq!(config["budget"]["target_bytes"], 1_000_000);
        assert_eq!(config["budget"]["feasible"], true);
        assert_eq!(config["budget"]["chosen"][0]["flag"], "--max-zoom");
    }

    #[test]
    fn a_budgets_shadow_prices_never_reach_the_command() {
        // THE no-thinning guard at this seam. A `ChosenLever` cannot be lossy
        // (no such field), and the lossy levers that DO exist live in
        // `shadow_prices`, which `to_command` never reads — under budget
        // pressure exactly as much as without it.
        let mut rec = rec_with_advice(vec![]);
        rec.budget = Some(budget_with(vec![chosen("--max-zoom", Some("6"))], false));
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        assert!(cmd.contains("--max-zoom 6"), "{cmd}");
        for lossy in [
            "--quantize-coords",
            "--quantize-attrs-auto",
            "--quantize-attr",
            "--maximum-tile-features",
        ] {
            assert!(!cmd.contains(lossy), "lossy flag leaked: {cmd}");
        }
        assert!(
            rec.budget
                .as_ref()
                .unwrap()
                .shadow_prices
                .iter()
                .all(|p| p.lossy),
            "every shadow price is lossy"
        );
    }

    #[test]
    fn a_suggestion_only_budget_lever_stays_out_of_the_command() {
        // A playback-caveated `spatial` ordering arrives from the layout advisor
        // with `suggestion_only: true` and keeps it under budget pressure. It
        // must not reach the command, and it must not override a scalar either.
        let mut ordering = chosen("--blob-ordering", Some("spatial"));
        ordering.suggestion_only = true;
        let mut clamp = chosen("--max-zoom", Some("7"));
        clamp.suggestion_only = true;
        let mut rec = rec_with_advice(vec![]);
        rec.budget = Some(budget_with(vec![clamp, ordering], true));

        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        assert!(!cmd.contains("--blob-ordering"), "{cmd}");
        // …and a suggestion-only scalar does not silently rewrite the recipe.
        assert!(cmd.contains("--max-zoom 10"), "{cmd}");
    }

    #[test]
    fn the_budget_verdict_reaches_the_explanations() {
        let rec = generate_recommendations_budgeted(
            &synthetic_result(),
            vec![],
            None,
            Some(budget_with(vec![], false)),
        );
        let line = rec
            .explanations
            .iter()
            .find(|e| e.starts_with("Budget (--target-size): "))
            .unwrap_or_else(|| panic!("{:?}", rec.explanations));
        assert!(line.contains("DOES NOT FIT"), "{line}");
        assert!(line.contains("nothing dropped"), "{line}");
    }

    #[test]
    fn recommendations_deserialize_without_a_budget() {
        // Append-only schema: a report written before the budget field existed
        // must still deserialize.
        let legacy = serde_json::json!({
            "min_zoom": 0,
            "max_zoom": 10,
            "temporal_bucket_ms": 3_600_000,
            "temporal_bucket_human": "1 hour",
            "confidence": 85,
            "explanations": [],
            "advice": [],
        });
        let rec: Recommendations = serde_json::from_value(legacy).expect("legacy deserializes");
        assert!(rec.budget.is_none());
    }

    #[test]
    fn lossy_advice_never_joins_to_command() {
        let rec = rec_with_advice(vec![
            advice("--quantize-coords", Some("1"), true),
            advice("--quantize-attrs-auto", None, true),
            advice("--maximum-tile-features", Some("10000"), true),
        ]);
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        assert!(
            !cmd.contains("--quantize-coords"),
            "lossy flag leaked: {cmd}"
        );
        assert!(
            !cmd.contains("--quantize-attrs-auto"),
            "lossy flag leaked: {cmd}"
        );
        assert!(
            !cmd.contains("--maximum-tile-features"),
            "lossy flag leaked: {cmd}"
        );
    }
}
