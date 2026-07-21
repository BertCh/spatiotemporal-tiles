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
        dominant_type: result.geometry.dominant_type.clone(),
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

/// Convert recommendations to a build config JSON structure.
///
/// The original scalar recipe (`min_zoom`/`max_zoom`/`temporal_bucket_ms`/
/// `confidence`/`explanations`) is preserved unchanged, plus the richer signals
/// downstream consumers (the MCP `recommend_build` tool) previously had to
/// recompute or do without: `temporal_bucket_human`, `dominant_type`, the full
/// evidence-based `advice` array (with its `lossy`/`confidence` markers), and
/// the ready-to-run `command` assembled by [`to_command`] (non-lossy advisor
/// flags already appended). Purely additive — no existing key changes shape.
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
        "temporal_bucket_human": recommendations.temporal_bucket_human,
        "dominant_type": recommendations.dominant_type,
        "confidence": recommendations.confidence,
        "explanations": recommendations.explanations,
        "advice": recommendations.advice,
        "command": to_command(recommendations, input, time_field),
    })
}

/// Convert recommendations to an stt-build command line.
///
/// NON-LOSSY advisor advice is appended (flag + value, in advisor order) so
/// the pasteable command carries the reversible byte-level/semantic levers.
/// LOSSY advice (quantization, budgets) NEVER joins the command — it degrades
/// data and stays a per-dataset opt-in the user must add by hand. The same
/// goes for `suggestion_only` advice: non-lossy, but the tradeoff spelled out
/// in its `why` needs a human decision before the flag is safe to run.
pub fn to_command(
    recommendations: &Recommendations,
    input: &Path,
    time_field: &str,
) -> String {
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

    let mut cmd = format!(
        "stt-build --input {} --output {} \\\n  --time-field {} --min-zoom {} --max-zoom {}",
        input_str,
        output_str,
        time_field,
        recommendations.min_zoom,
        recommendations.max_zoom,
    );
    // The recommended bucket is the recipe's core scalar — the command must
    // carry it or a paste-and-run build silently falls back to the 1h default.
    // Bare milliseconds: always valid `parse_duration` input.
    if recommendations.temporal_bucket_ms > 0 {
        cmd.push_str(&format!(
            " \\\n  --temporal-bucket {}",
            recommendations.temporal_bucket_ms
        ));
    }
    // Stable order: exactly the advisor emit order (quantize, temporal,
    // layout, budget), filtered to the auto-applicable entries.
    for advice in recommendations
        .advice
        .iter()
        .filter(|a| !a.lossy && !a.suggestion_only)
    {
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
        assert!(cmd.contains("--input /data/in/quakes.parquet"), "command: {cmd}");
        assert!(cmd.contains("--output /data/in/quakes"), "command: {cmd}");
    }

    #[test]
    fn suggestion_only_advice_never_joins_to_command() {
        let mut caveated = advice("--blob-ordering", Some("spatial"), false);
        caveated.suggestion_only = true;
        let mut semantic = advice("--min-zoom-field", Some("category"), false);
        semantic.suggestion_only = true;
        let rec = rec_with_advice(vec![
            caveated,
            semantic,
            advice("--publish", None, false),
        ]);
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
        assert!(command.contains("--blob-ordering spatial"), "command: {command}");
        assert!(!command.contains("--quantize-coords"), "command: {command}");
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


