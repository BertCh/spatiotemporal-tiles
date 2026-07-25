//! Geometry complexity and size analysis
//!
//! Analyzes geometry types, vertex counts, and estimated sizes
//! to inform complexity classification.

use crate::loader::LoadedData;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Geometry analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryAnalysis {
    /// Distribution of geometry types
    pub type_distribution: HashMap<String, usize>,
    /// Dominant geometry type (most common)
    pub dominant_type: String,
    /// Vertex count statistics
    pub vertex_stats: VertexStats,
    /// Estimated size statistics (bytes per feature)
    pub size_stats: SizeStats,
    /// Property count statistics
    pub property_stats: PropertyStats,
    /// Geometry complexity classification
    pub complexity: GeometryComplexity,
}

/// Vertex count statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexStats {
    pub min: usize,
    pub max: usize,
    pub avg: f64,
    pub median: usize,
    pub p95: usize,
    pub p99: usize,
    pub total: usize,
}

/// Size statistics (estimated bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeStats {
    pub min: usize,
    pub max: usize,
    pub avg: f64,
    pub median: usize,
    pub p95: usize,
    pub p99: usize,
    pub total: usize,
}

/// Property count statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyStats {
    pub min: usize,
    pub max: usize,
    pub avg: f64,
}

/// Classification of geometry complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometryComplexity {
    /// Simple point data
    Simple,
    /// Moderate complexity (short linestrings, simple polygons)
    Moderate,
    /// High complexity (long linestrings, complex polygons)
    Complex,
    /// Very high complexity (multigeometries, many vertices)
    VeryComplex,
}

impl std::fmt::Display for GeometryComplexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryComplexity::Simple => write!(f, "Simple (points)"),
            GeometryComplexity::Moderate => write!(f, "Moderate"),
            GeometryComplexity::Complex => write!(f, "Complex"),
            GeometryComplexity::VeryComplex => write!(f, "Very Complex"),
        }
    }
}

/// Analyze geometry characteristics of the dataset
pub fn analyze(data: &LoadedData) -> Result<GeometryAnalysis> {
    if data.features.is_empty() {
        return Ok(empty_analysis());
    }

    // Count geometry types
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for feature in &data.features {
        *type_counts
            .entry(feature.geometry_type.to_string())
            .or_insert(0) += 1;
    }

    // Find dominant type
    let dominant_type = type_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(t, _)| t.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    // Calculate vertex statistics
    let mut vertex_counts: Vec<usize> = data.features.iter().map(|f| f.vertex_count).collect();
    vertex_counts.sort();
    let vertex_stats = calculate_stats(&vertex_counts);

    // Calculate size statistics
    let mut sizes: Vec<usize> = data.features.iter().map(|f| f.estimated_size).collect();
    sizes.sort();
    let size_stats = calculate_size_stats(&sizes);

    // Calculate property statistics
    let property_counts: Vec<usize> = data.features.iter().map(|f| f.property_count).collect();
    let property_stats = PropertyStats {
        min: property_counts.iter().copied().min().unwrap_or(0),
        max: property_counts.iter().copied().max().unwrap_or(0),
        avg: if property_counts.is_empty() {
            0.0
        } else {
            property_counts.iter().sum::<usize>() as f64 / property_counts.len() as f64
        },
    };

    // Classify complexity
    let complexity = classify_complexity(&vertex_stats, &dominant_type, &type_counts);

    Ok(GeometryAnalysis {
        type_distribution: type_counts,
        dominant_type,
        vertex_stats,
        size_stats,
        property_stats,
        complexity,
    })
}

fn empty_analysis() -> GeometryAnalysis {
    GeometryAnalysis {
        type_distribution: HashMap::new(),
        dominant_type: "None".to_string(),
        vertex_stats: VertexStats {
            min: 0,
            max: 0,
            avg: 0.0,
            median: 0,
            p95: 0,
            p99: 0,
            total: 0,
        },
        size_stats: SizeStats {
            min: 0,
            max: 0,
            avg: 0.0,
            median: 0,
            p95: 0,
            p99: 0,
            total: 0,
        },
        property_stats: PropertyStats {
            min: 0,
            max: 0,
            avg: 0.0,
        },
        complexity: GeometryComplexity::Simple,
    }
}

/// Calculate statistics from a sorted list of counts
fn calculate_stats(sorted: &[usize]) -> VertexStats {
    if sorted.is_empty() {
        return VertexStats {
            min: 0,
            max: 0,
            avg: 0.0,
            median: 0,
            p95: 0,
            p99: 0,
            total: 0,
        };
    }

    let n = sorted.len();
    let total: usize = sorted.iter().sum();

    VertexStats {
        min: sorted[0],
        max: sorted[n - 1],
        avg: total as f64 / n as f64,
        median: sorted[n / 2],
        p95: sorted[((n as f64 * 0.95) as usize).min(n - 1)],
        p99: sorted[((n as f64 * 0.99) as usize).min(n - 1)],
        total,
    }
}

/// Calculate size statistics from a sorted list of sizes
fn calculate_size_stats(sorted: &[usize]) -> SizeStats {
    if sorted.is_empty() {
        return SizeStats {
            min: 0,
            max: 0,
            avg: 0.0,
            median: 0,
            p95: 0,
            p99: 0,
            total: 0,
        };
    }

    let n = sorted.len();
    let total: usize = sorted.iter().sum();

    SizeStats {
        min: sorted[0],
        max: sorted[n - 1],
        avg: total as f64 / n as f64,
        median: sorted[n / 2],
        p95: sorted[((n as f64 * 0.95) as usize).min(n - 1)],
        p99: sorted[((n as f64 * 0.99) as usize).min(n - 1)],
        total,
    }
}

/// Classify geometry complexity
fn classify_complexity(
    vertex_stats: &VertexStats,
    dominant_type: &str,
    type_counts: &HashMap<String, usize>,
) -> GeometryComplexity {
    // Check for multi-geometries
    let has_multi = type_counts.keys().any(|t| t.starts_with("Multi"));

    // Simple points
    if dominant_type == "Point" && vertex_stats.avg < 1.5 && !has_multi {
        return GeometryComplexity::Simple;
    }

    // Very complex: high vertex counts or multi-geometries
    if vertex_stats.avg > 1000.0 || vertex_stats.p95 > 5000 {
        return GeometryComplexity::VeryComplex;
    }

    // Complex: moderate vertex counts
    if vertex_stats.avg > 100.0 || vertex_stats.p95 > 500 {
        return GeometryComplexity::Complex;
    }

    // Polygons tend to be more complex
    if dominant_type == "Polygon" || dominant_type == "MultiPolygon" {
        if vertex_stats.avg > 20.0 {
            return GeometryComplexity::Complex;
        }
        return GeometryComplexity::Moderate;
    }

    // LineStrings
    if dominant_type == "LineString" || dominant_type == "MultiLineString" {
        if vertex_stats.avg > 50.0 {
            return GeometryComplexity::Complex;
        }
        return GeometryComplexity::Moderate;
    }

    GeometryComplexity::Moderate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_stats() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let stats = calculate_stats(&data);
        assert_eq!(stats.min, 1);
        assert_eq!(stats.max, 10);
        // Median at index 5 (0-indexed) is 6
        assert_eq!(stats.median, 6);
        assert!((stats.avg - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_classify_complexity_simple() {
        let stats = VertexStats {
            min: 1,
            max: 1,
            avg: 1.0,
            median: 1,
            p95: 1,
            p99: 1,
            total: 100,
        };
        let types: HashMap<String, usize> = [("Point".to_string(), 100)].into_iter().collect();
        let complexity = classify_complexity(&stats, "Point", &types);
        assert!(matches!(complexity, GeometryComplexity::Simple));
    }
}
