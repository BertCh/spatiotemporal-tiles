//! Report generation (text and JSON formats)
//!
//! Generates human-readable and machine-readable reports from analysis results.

use crate::analysis::density::IssueSeverity;
use crate::analysis::AnalysisResult;
use crate::recommend::Recommendations;
use anyhow::Result;

/// Generate a text report
pub fn generate_text(result: &AnalysisResult, recommendations: &Recommendations) -> String {
    let mut output = String::new();

    // Header
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    output.push_str(&format!("         STT Optimization Report - {}\n", result.source));
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    // Dataset Summary
    output.push_str("📊 Dataset Summary\n");
    output.push_str(&format!("  Features:        {:>12}\n", format_number(result.feature_count)));
    output.push_str(&format!("  Time Range:      {}\n", result.temporal.time_range_description()));
    output.push_str(&format!(
        "  Spatial Bounds:  [{:.2}, {:.2}] to [{:.2}, {:.2}]\n",
        result.spatial.zoom_coverage.first().map(|_| -180.0).unwrap_or(0.0),
        -90.0,
        180.0,
        90.0
    ));
    output.push_str(&format!("  Geometry Type:   {} ({})\n", 
        result.geometry.dominant_type,
        format_type_distribution(&result.geometry.type_distribution)
    ));
    output.push('\n');

    // Spatial Analysis
    output.push_str("🗺️  Spatial Analysis\n");
    output.push_str(&format!("  Distribution:    {}\n", result.spatial.distribution));
    
    // Show coverage at key zoom levels
    for z in [4, 6, 8, 10, 12].iter() {
        if let Some(cov) = result.spatial.zoom_coverage.iter().find(|c| c.zoom == *z) {
            output.push_str(&format!(
                "  Coverage at z{}:  {:.2}% ({} tiles)\n",
                z, cov.coverage_percent, cov.occupied_tiles
            ));
        }
    }
    
    output.push_str(&format!(
        "  Recommended Zoom: {}-{}\n",
        result.spatial.recommended_min_zoom,
        result.spatial.recommended_max_zoom
    ));

    if !result.spatial.hotspots.is_empty() {
        output.push_str("  Hotspots:\n");
        for (i, hotspot) in result.spatial.hotspots.iter().take(3).enumerate() {
            let name = hotspot.name.as_deref().unwrap_or("Unknown region");
            output.push_str(&format!(
                "    {}. {} ({} features)\n",
                i + 1, name, format_number(hotspot.feature_count)
            ));
        }
    }
    output.push('\n');

    // Temporal Analysis
    output.push_str("⏰ Temporal Analysis\n");
    output.push_str(&format!("  Duration:        {}\n", result.temporal.duration_human));
    output.push_str(&format!("  Distribution:    {}\n", result.temporal.distribution));
    output.push_str(&format!(
        "  Events/day avg:  {:.1}\n",
        result.temporal.events_per_day.avg
    ));
    output.push_str(&format!(
        "  Unique times:    {}\n",
        format_number(result.temporal.unique_timestamps)
    ));
    output.push_str(&format!(
        "  Suggested bucket: {}\n",
        result.temporal.recommended_bucket_human
    ));
    output.push('\n');

    // Geometry Analysis
    output.push_str("📐 Geometry Analysis\n");
    output.push_str(&format!("  Complexity:      {}\n", result.geometry.complexity));
    output.push_str(&format!(
        "  Vertices (avg):  {:.1}\n",
        result.geometry.vertex_stats.avg
    ));
    output.push_str(&format!(
        "  Vertices (p95):  {}\n",
        result.geometry.vertex_stats.p95
    ));
    output.push_str(&format!(
        "  Avg size/feat:   {} bytes\n",
        result.geometry.size_stats.avg as usize
    ));
    output.push_str(&format!(
        "  Total size:      {}\n",
        format_bytes(result.geometry.size_stats.total)
    ));
    output.push('\n');

    // Size Estimation
    output.push_str("💾 Size Estimation\n");
    output.push_str(&format!(
        "  Est. tiles:      {} (at recommended settings)\n",
        format_number(result.density.estimated_tile_count)
    ));
    output.push_str(&format!(
        "  Est. archive:    {} compressed\n",
        format_bytes(result.density.estimated_archive_size)
    ));
    output.push('\n');

    // Issues
    if !result.density.issues.is_empty() {
        output.push_str("⚠️  Issues\n");
        for issue in &result.density.issues {
            let icon = match issue.severity {
                IssueSeverity::Error => "❌",
                IssueSeverity::Warning => "⚠️",
                IssueSeverity::Info => "ℹ️",
            };
            output.push_str(&format!("  {} {}\n", icon, issue.description));
            output.push_str(&format!("     → {}\n", issue.suggestion));
        }
        output.push('\n');
    }

    // Recommendations
    output.push_str("💡 Recommendations\n");
    output.push_str(&format!(
        "  --min-zoom {}\n",
        recommendations.min_zoom
    ));
    output.push_str(&format!(
        "  --max-zoom {}\n",
        recommendations.max_zoom
    ));
    output.push_str(&format!(
        "  --chunk-size {}\n",
        recommendations.chunk_size
    ));
    // Advisory only: the packed STT format is zstd-only, so compression is not a
    // build knob. stt-build --auto consumes only the zoom/temporal-bucket values.
    output.push_str(&format!(
        "  compression: {} (fixed; packed format is zstd-only)\n",
        recommendations.compression
    ));
    output.push('\n');

    // Confidence
    output.push_str(&format!(
        "  Confidence: {}%\n",
        recommendations.confidence
    ));
    output.push('\n');

    // Suggested Command
    output.push_str("📋 Suggested Command:\n");
    output.push_str(&format!(
        "  stt-build --input {} --output {}.stt \\\n",
        result.source,
        result.source.trim_end_matches(".parquet").trim_end_matches(".geoparquet")
    ));
    output.push_str(&format!(
        "    --time-field timestamp --min-zoom {} --max-zoom {} \\\n",
        recommendations.min_zoom, recommendations.max_zoom
    ));
    output.push_str(&format!(
        "    --chunk-size {} --compression {}\n",
        recommendations.chunk_size, recommendations.compression
    ));
    output.push('\n');

    // Footer
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    output
}

/// Generate a JSON report
pub fn generate_json(result: &AnalysisResult, recommendations: &Recommendations) -> Result<String> {
    let report = serde_json::json!({
        "source": result.source,
        "feature_count": result.feature_count,
        "spatial": {
            "distribution": format!("{}", result.spatial.distribution),
            "recommended_min_zoom": result.spatial.recommended_min_zoom,
            "recommended_max_zoom": result.spatial.recommended_max_zoom,
            "hotspots": result.spatial.hotspots.iter().map(|h| {
                serde_json::json!({
                    "lon": h.lon,
                    "lat": h.lat,
                    "feature_count": h.feature_count,
                    "name": h.name,
                })
            }).collect::<Vec<_>>(),
            "zoom_coverage": result.spatial.zoom_coverage.iter().map(|c| {
                serde_json::json!({
                    "zoom": c.zoom,
                    "coverage_percent": c.coverage_percent,
                    "occupied_tiles": c.occupied_tiles,
                    "avg_features_per_tile": c.avg_features_per_tile,
                })
            }).collect::<Vec<_>>(),
        },
        "temporal": {
            "time_start": result.temporal.time_start,
            "time_end": result.temporal.time_end,
            "duration_ms": result.temporal.duration_ms,
            "duration_human": result.temporal.duration_human,
            "distribution": format!("{}", result.temporal.distribution),
            "unique_timestamps": result.temporal.unique_timestamps,
            "recommended_bucket_ms": result.temporal.recommended_bucket_ms,
            "recommended_bucket_human": result.temporal.recommended_bucket_human,
            "events_per_day": {
                "avg": result.temporal.events_per_day.avg,
                "min": result.temporal.events_per_day.min,
                "max": result.temporal.events_per_day.max,
            },
        },
        "geometry": {
            "dominant_type": result.geometry.dominant_type,
            "type_distribution": result.geometry.type_distribution,
            "complexity": format!("{}", result.geometry.complexity),
            "vertex_stats": {
                "min": result.geometry.vertex_stats.min,
                "max": result.geometry.vertex_stats.max,
                "avg": result.geometry.vertex_stats.avg,
                "median": result.geometry.vertex_stats.median,
                "p95": result.geometry.vertex_stats.p95,
            },
            "size_stats": {
                "min": result.geometry.size_stats.min,
                "max": result.geometry.size_stats.max,
                "avg": result.geometry.size_stats.avg,
                "total": result.geometry.size_stats.total,
            },
        },
        "density": {
            "recommended_chunk_size": result.density.recommended_chunk_size,
            "estimated_tile_count": result.density.estimated_tile_count,
            "estimated_archive_size": result.density.estimated_archive_size,
            "issues": result.density.issues.iter().map(|i| {
                serde_json::json!({
                    "severity": format!("{}", i.severity),
                    "description": i.description,
                    "suggestion": i.suggestion,
                })
            }).collect::<Vec<_>>(),
        },
        "recommendations": {
            "min_zoom": recommendations.min_zoom,
            "max_zoom": recommendations.max_zoom,
            "chunk_size": recommendations.chunk_size,
            "compression": recommendations.compression,
            "temporal_bucket_ms": recommendations.temporal_bucket_ms,
            "confidence": recommendations.confidence,
            "explanations": recommendations.explanations,
        },
    });

    Ok(serde_json::to_string_pretty(&report)?)
}

/// Format a number with thousands separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

/// Format bytes as human-readable string
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Format type distribution as percentage string
fn format_type_distribution(dist: &std::collections::HashMap<String, usize>) -> String {
    let total: usize = dist.values().sum();
    if total == 0 {
        return "N/A".to_string();
    }

    let mut parts: Vec<String> = dist
        .iter()
        .map(|(t, c)| {
            let pct = *c as f64 / total as f64 * 100.0;
            if pct > 99.0 {
                format!("100% {}", t)
            } else if pct > 1.0 {
                format!("{:.0}% {}", pct, t)
            } else {
                String::new()
            }
        })
        .filter(|s| !s.is_empty())
        .collect();

    parts.sort();
    parts.reverse();
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(123), "123");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}


