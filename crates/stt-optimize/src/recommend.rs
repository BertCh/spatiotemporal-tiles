//! Parameter recommendation engine
//!
//! Combines analysis results to generate optimal stt-build parameters.

use crate::analysis::AnalysisResult;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Recommended parameters for stt-build.
///
/// `stt-build --auto` folds `min_zoom`, `max_zoom`, and `temporal_bucket_ms`
/// into its own args and logs `confidence`/`explanations`; all fields also
/// appear in the standalone `analyze`/`recommend` reports.
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
}

/// Generate recommendations from analysis results
pub fn generate_recommendations(result: &AnalysisResult) -> Recommendations {
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

/// Convert recommendations to an stt-build command line
pub fn to_command(
    recommendations: &Recommendations,
    input: &Path,
    time_field: &str,
) -> String {
    let output = input.with_extension("stt");
    let output_str = output.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "output.stt".to_string());

    let input_str = input.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "input.parquet".to_string());

    format!(
        "stt-build --input {} --output {} \\\n  --time-field {} --min-zoom {} --max-zoom {}",
        input_str,
        output_str,
        time_field,
        recommendations.min_zoom,
        recommendations.max_zoom,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_command() {
        let rec = Recommendations {
            min_zoom: 0,
            max_zoom: 10,
            temporal_bucket_ms: 3_600_000,
            temporal_bucket_human: "1 hour".to_string(),
            confidence: 85,
            explanations: vec![],
        };

        let cmd = to_command(&rec, Path::new("data.parquet"), "timestamp");
        assert!(cmd.contains("--min-zoom 0"));
        assert!(cmd.contains("--max-zoom 10"));
    }
}


