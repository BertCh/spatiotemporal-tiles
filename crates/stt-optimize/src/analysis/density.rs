//! Tile density simulation and chunk size analysis
//!
//! Simulates tile generation at various chunk sizes to predict
//! optimal parameters and identify potential issues.

use crate::analysis::spatial::SpatialAnalysis;
use crate::loader::LoadedData;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use stt_core::projection;

/// Density analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityAnalysis {
    /// Simulated results at different chunk sizes
    pub chunk_simulations: Vec<ChunkSimulation>,
    /// Recommended chunk size in bytes
    pub recommended_chunk_size: usize,
    /// Estimated total tile count at recommended settings
    pub estimated_tile_count: usize,
    /// Estimated archive size in bytes (compressed)
    pub estimated_archive_size: usize,
    /// Potential issues identified
    pub issues: Vec<DensityIssue>,
}

/// Simulation results for a specific chunk size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSimulation {
    /// Target chunk size in bytes
    pub chunk_size: usize,
    /// Estimated number of tiles
    pub tile_count: usize,
    /// Average features per tile
    pub avg_features_per_tile: f64,
    /// Median features per tile
    pub median_features_per_tile: usize,
    /// Maximum features in any tile
    pub max_features_per_tile: usize,
    /// Number of oversized tiles (> max recommended features)
    pub oversized_tiles: usize,
    /// Number of undersized tiles (< 10 features)
    pub undersized_tiles: usize,
    /// Estimated total size (uncompressed)
    pub estimated_size_uncompressed: usize,
    /// Estimated total size (compressed, assuming 3x ratio)
    pub estimated_size_compressed: usize,
}

/// A potential density issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Suggested fix
    pub suggestion: String,
}

/// Issue severity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Info => write!(f, "INFO"),
            IssueSeverity::Warning => write!(f, "WARNING"),
            IssueSeverity::Error => write!(f, "ERROR"),
        }
    }
}

/// Analyze density characteristics and simulate chunk sizing
pub fn analyze(data: &LoadedData, spatial: &SpatialAnalysis) -> Result<DensityAnalysis> {
    // Chunk sizes to simulate (in bytes)
    let chunk_sizes = [
        64_000,   // 64 KB
        128_000,  // 128 KB
        256_000,  // 256 KB
        500_000,  // 500 KB (default)
        1_000_000, // 1 MB
        2_000_000, // 2 MB
    ];

    let mut simulations = Vec::new();
    let recommended_zoom = spatial.recommended_max_zoom;

    for &chunk_size in &chunk_sizes {
        let sim = simulate_chunk_size(data, chunk_size, recommended_zoom);
        simulations.push(sim);
    }

    // Find optimal chunk size (minimize oversized tiles while keeping tile count reasonable)
    let (recommended_chunk_size, best_sim_idx) = find_optimal_chunk_size(&simulations);

    let estimated_tile_count = simulations[best_sim_idx].tile_count;
    let estimated_archive_size = simulations[best_sim_idx].estimated_size_compressed;

    // Identify issues
    let issues = identify_issues(data, spatial, &simulations[best_sim_idx]);

    Ok(DensityAnalysis {
        chunk_simulations: simulations,
        recommended_chunk_size,
        estimated_tile_count,
        estimated_archive_size,
        issues,
    })
}

/// Simulate tile generation with a specific chunk size
fn simulate_chunk_size(data: &LoadedData, chunk_size: usize, zoom: u8) -> ChunkSimulation {
    // Group features by spatial tile at the target zoom
    let mut tile_features: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for (idx, feature) in data.features.iter().enumerate() {
        if let Ok((x, y)) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
            tile_features.entry((x, y)).or_insert_with(Vec::new).push(idx);
        }
    }

    // Simulate chunking based on size
    let mut tile_count = 0;
    let mut feature_counts = Vec::new();
    let mut total_uncompressed = 0;

    for (_, feature_indices) in &tile_features {
        // Sort by timestamp
        let mut indices = feature_indices.clone();
        indices.sort_by_key(|&i| data.features[i].timestamp);

        // Chunk by size
        let mut current_chunk_size = 0;
        let mut current_chunk_features = 0;

        for &idx in &indices {
            let feature_size = data.features[idx].estimated_size;

            if current_chunk_size > 0 && current_chunk_size + feature_size > chunk_size {
                // Start new chunk
                tile_count += 1;
                feature_counts.push(current_chunk_features);
                total_uncompressed += current_chunk_size;
                current_chunk_size = 0;
                current_chunk_features = 0;
            }

            current_chunk_size += feature_size;
            current_chunk_features += 1;
        }

        // Last chunk
        if current_chunk_features > 0 {
            tile_count += 1;
            feature_counts.push(current_chunk_features);
            total_uncompressed += current_chunk_size;
        }
    }

    // Calculate statistics
    feature_counts.sort();
    let avg_features = if tile_count > 0 {
        feature_counts.iter().sum::<usize>() as f64 / tile_count as f64
    } else {
        0.0
    };

    let median_features = if !feature_counts.is_empty() {
        feature_counts[feature_counts.len() / 2]
    } else {
        0
    };

    let max_features = feature_counts.iter().copied().max().unwrap_or(0);
    let oversized = feature_counts.iter().filter(|&&c| c > 10_000).count();
    let undersized = feature_counts.iter().filter(|&&c| c < 10).count();

    // Estimate compressed size (assuming 3x compression ratio for gzip)
    let estimated_compressed = total_uncompressed / 3;

    ChunkSimulation {
        chunk_size,
        tile_count,
        avg_features_per_tile: avg_features,
        median_features_per_tile: median_features,
        max_features_per_tile: max_features,
        oversized_tiles: oversized,
        undersized_tiles: undersized,
        estimated_size_uncompressed: total_uncompressed,
        estimated_size_compressed: estimated_compressed,
    }
}

/// Find the optimal chunk size from simulations
fn find_optimal_chunk_size(simulations: &[ChunkSimulation]) -> (usize, usize) {
    // Score each simulation
    // Prefer: low oversized, reasonable tile count, good compression potential
    let scores: Vec<(usize, f64)> = simulations
        .iter()
        .enumerate()
        .map(|(idx, sim)| {
            let mut score = 100.0;

            // Penalize oversized tiles heavily
            score -= sim.oversized_tiles as f64 * 10.0;

            // Penalize too many undersized tiles
            score -= (sim.undersized_tiles as f64 / 10.0).min(20.0);

            // Prefer moderate tile counts (1000-5000)
            if sim.tile_count < 100 {
                score -= 20.0;
            } else if sim.tile_count > 10000 {
                score -= (sim.tile_count as f64 - 10000.0) / 1000.0;
            }

            // Prefer chunk sizes around 256-500 KB
            let chunk_kb = sim.chunk_size / 1000;
            if chunk_kb < 100 {
                score -= 5.0;
            } else if chunk_kb > 1000 {
                score -= 5.0;
            }

            (idx, score)
        })
        .collect();

    // Find best score
    let (best_idx, _) = scores
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .cloned()
        .unwrap_or((2, 0.0)); // Default to 256 KB

    (simulations[best_idx].chunk_size, best_idx)
}

/// Identify potential issues
fn identify_issues(
    data: &LoadedData,
    spatial: &SpatialAnalysis,
    sim: &ChunkSimulation,
) -> Vec<DensityIssue> {
    let mut issues = Vec::new();

    // Check for oversized tiles
    if sim.oversized_tiles > 0 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!(
                "{} tiles exceed 10,000 features (max: {})",
                sim.oversized_tiles, sim.max_features_per_tile
            ),
            suggestion: "Consider enabling tile budgets or reducing max zoom level".to_string(),
        });
    }

    // Check for many undersized tiles
    let undersized_pct = if sim.tile_count > 0 {
        sim.undersized_tiles as f64 / sim.tile_count as f64 * 100.0
    } else {
        0.0
    };
    if undersized_pct > 20.0 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Info,
            description: format!(
                "{:.1}% of tiles have fewer than 10 features",
                undersized_pct
            ),
            suggestion: "Consider reducing max zoom or increasing temporal bucketing".to_string(),
        });
    }

    // Check for very high tile count
    if sim.tile_count > 50_000 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!(
                "High tile count ({}) may impact loading performance",
                sim.tile_count
            ),
            suggestion: "Consider reducing zoom range or increasing chunk size".to_string(),
        });
    }

    // Check for sparse data at high zooms
    if let Some(z_max) = spatial.zoom_coverage.iter().find(|z| z.zoom == spatial.recommended_max_zoom) {
        if z_max.coverage_percent < 0.1 {
            issues.push(DensityIssue {
                severity: IssueSeverity::Info,
                description: format!(
                    "Only {:.2}% coverage at zoom {}",
                    z_max.coverage_percent, spatial.recommended_max_zoom
                ),
                suggestion: "Data is sparse at this zoom level, consider reducing max zoom".to_string(),
            });
        }
    }

    // Check estimated archive size
    let size_mb = sim.estimated_size_compressed as f64 / 1_048_576.0;
    if size_mb > 500.0 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!("Large estimated archive size ({:.1} MB)", size_mb),
            suggestion: "Consider splitting into multiple archives or reducing data scope".to_string(),
        });
    }

    // Check for hotspot concentration
    if !spatial.hotspots.is_empty() {
        let top_hotspot = &spatial.hotspots[0];
        let hotspot_pct = top_hotspot.feature_count as f64 / data.features.len() as f64 * 100.0;
        if hotspot_pct > 50.0 {
            issues.push(DensityIssue {
                severity: IssueSeverity::Info,
                description: format!(
                    "{:.1}% of features concentrated in {}",
                    hotspot_pct,
                    top_hotspot.name.as_deref().unwrap_or("one region")
                ),
                suggestion: "Hotspot areas may have larger tiles; budgets recommended".to_string(),
            });
        }
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_optimal_chunk_size() {
        let simulations = vec![
            ChunkSimulation {
                chunk_size: 128_000,
                tile_count: 5000,
                avg_features_per_tile: 50.0,
                median_features_per_tile: 45,
                max_features_per_tile: 500,
                oversized_tiles: 0,
                undersized_tiles: 100,
                estimated_size_uncompressed: 10_000_000,
                estimated_size_compressed: 3_000_000,
            },
            ChunkSimulation {
                chunk_size: 256_000,
                tile_count: 3000,
                avg_features_per_tile: 80.0,
                median_features_per_tile: 75,
                max_features_per_tile: 800,
                oversized_tiles: 0,
                undersized_tiles: 50,
                estimated_size_uncompressed: 10_000_000,
                estimated_size_compressed: 3_000_000,
            },
        ];

        let (chunk_size, _) = find_optimal_chunk_size(&simulations);
        assert_eq!(chunk_size, 256_000);
    }
}

