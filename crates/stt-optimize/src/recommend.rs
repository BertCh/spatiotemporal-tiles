//! Parameter recommendation engine
//!
//! Combines analysis results to generate optimal stt-build parameters.

use crate::advisors::Advice;
use crate::analysis::AnalysisResult;
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
    /// Explanation of key decisions
    pub explanations: Vec<String>,
    /// Evidence-based advisor suggestions ([`crate::advisors::run_all`]),
    /// in advisor order (quantize, temporal, layout, budget). Entries with
    /// `lossy: true` are surfaced only — they never join [`to_command`] and
    /// are never auto-applied by `stt-build --auto`.
    #[serde(default)]
    pub advice: Vec<Advice>,
}

/// Generate recommendations from analysis results, attaching the advisor
/// suggestions (pass an empty `Vec` when the advisors were not run).
pub fn generate_recommendations(result: &AnalysisResult, advice: Vec<Advice>) -> Recommendations {
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

    Recommendations {
        min_zoom,
        max_zoom,
        temporal_bucket_ms,
        temporal_bucket_human,
        confidence,
        explanations,
        advice,
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
    if matches!(result.spatial.distribution, crate::analysis::spatial::SpatialDistribution::Sparse) {
        score = score.saturating_sub(10);
    }

    // Lower confidence with complex geometry
    if matches!(result.geometry.complexity, crate::analysis::geometry::GeometryComplexity::VeryComplex) {
        score = score.saturating_sub(10);
    }

    score
}

/// Convert recommendations to a build config JSON structure
pub fn to_build_config(
    recommendations: &Recommendations,
    input: &Path,
    time_field: &str,
) -> serde_json::Value {
    serde_json::json!({
        "input": input.to_string_lossy(),
        "time_field": time_field,
        "min_zoom": recommendations.min_zoom,
        "max_zoom": recommendations.max_zoom,
        "temporal_bucket_ms": recommendations.temporal_bucket_ms,
        "confidence": recommendations.confidence,
        "explanations": recommendations.explanations,
    })
}

/// Convert recommendations to an stt-build command line.
///
/// NON-LOSSY advisor advice is appended (flag + value, in advisor order) so
/// the pasteable command carries the reversible byte-level/semantic levers.
/// LOSSY advice (quantization, budgets) NEVER joins the command — it degrades
/// data and stays a per-dataset opt-in the user must add by hand.
pub fn to_command(
    recommendations: &Recommendations,
    input: &Path,
    time_field: &str,
) -> String {
    // Suggest the packed dataset DIRECTORY (the input's stem): stt-build's
    // output is a directory tree, not a single file.
    let output = input.with_extension("");
    let output_str = output.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());

    let input_str = input.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "input.parquet".to_string());

    let mut cmd = format!(
        "stt-build --input {} --output {} \\\n  --time-field {} --min-zoom {} --max-zoom {}",
        input_str,
        output_str,
        time_field,
        recommendations.min_zoom,
        recommendations.max_zoom,
    );
    // Stable order: exactly the advisor emit order (quantize, temporal,
    // layout, budget), filtered to the non-lossy entries.
    for advice in recommendations.advice.iter().filter(|a| !a.lossy) {
        cmd.push_str(" \\\n  ");
        cmd.push_str(&advice.flag);
        if let Some(value) = &advice.value {
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
            explanations: vec![],
            advice,
        }
    }

    #[test]
    fn test_to_command() {
        let rec = rec_with_advice(vec![]);
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        assert!(cmd.contains("--min-zoom 0"));
        assert!(cmd.contains("--max-zoom 10"));
    }

    #[test]
    fn generate_recommendations_attaches_advice() {
        let result = synthetic_result();
        let rec = generate_recommendations(
            &result,
            vec![advice("--publish", None, false), advice("--quantize-coords", Some("1"), true)],
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
        let lod = cmd.find("--temporal-lod 1d,30d").expect("temporal-lod in command");
        let publish = cmd.find("--publish").expect("publish in command");
        let ordering = cmd.find("--blob-ordering spatial").expect("blob-ordering in command");
        assert!(lod < publish && publish < ordering, "advisor order preserved: {cmd}");
    }

    #[test]
    fn lossy_advice_never_joins_to_command() {
        let rec = rec_with_advice(vec![
            advice("--quantize-coords", Some("1"), true),
            advice("--quantize-attrs-auto", None, true),
            advice("--maximum-tile-features", Some("10000"), true),
        ]);
        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        assert!(!cmd.contains("--quantize-coords"), "lossy flag leaked: {cmd}");
        assert!(!cmd.contains("--quantize-attrs-auto"), "lossy flag leaked: {cmd}");
        assert!(!cmd.contains("--maximum-tile-features"), "lossy flag leaked: {cmd}");
    }
}


