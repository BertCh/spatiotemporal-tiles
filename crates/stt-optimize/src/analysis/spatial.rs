//! Spatial density and coverage analysis
//!
//! Analyzes the spatial distribution of features to recommend zoom levels
//! and identify hotspots.

use crate::loader::LoadedData;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use stt_core::projection;

/// Spatial analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAnalysis {
    /// Coverage statistics per zoom level
    pub zoom_coverage: Vec<ZoomCoverage>,
    /// Identified hotspots (regions with high density)
    pub hotspots: Vec<Hotspot>,
    /// Recommended minimum zoom level
    pub recommended_min_zoom: u8,
    /// Recommended maximum zoom level
    pub recommended_max_zoom: u8,
    /// Spatial distribution classification
    pub distribution: SpatialDistribution,
}

/// Coverage statistics for a zoom level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomCoverage {
    /// Zoom level
    pub zoom: u8,
    /// Total possible tiles at this zoom
    pub total_tiles: u64,
    /// Tiles containing at least one feature
    pub occupied_tiles: u64,
    /// Coverage percentage (0-100)
    pub coverage_percent: f64,
    /// Average features per occupied tile
    pub avg_features_per_tile: f64,
    /// Maximum features in any single tile
    pub max_features_in_tile: usize,
    /// Median features per tile
    pub median_features_per_tile: usize,
}

/// A spatial hotspot (region with high feature density)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hotspot {
    /// Center longitude
    pub lon: f64,
    /// Center latitude
    pub lat: f64,
    /// Approximate radius in degrees
    pub radius: f64,
    /// Feature count in this hotspot
    pub feature_count: usize,
    /// Descriptive name (if determinable)
    pub name: Option<String>,
}

/// Classification of spatial distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialDistribution {
    /// Features are spread evenly across the globe
    Global,
    /// Features clustered in specific regions
    Regional,
    /// Features concentrated in one or few small areas
    Localized,
    /// Very sparse data
    Sparse,
}

impl std::fmt::Display for SpatialDistribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpatialDistribution::Global => write!(f, "Global (spread worldwide)"),
            SpatialDistribution::Regional => write!(f, "Regional (clustered in regions)"),
            SpatialDistribution::Localized => write!(f, "Localized (concentrated areas)"),
            SpatialDistribution::Sparse => write!(f, "Sparse (very low density)"),
        }
    }
}

/// Analyze spatial characteristics of the dataset
pub fn analyze(data: &LoadedData) -> Result<SpatialAnalysis> {
    use indicatif::{ProgressBar, ProgressStyle};

    let pb = ProgressBar::new(15); // We analyze z0-z14
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:30.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("##-"),
    );
    pb.set_message("Analyzing spatial coverage");

    let mut zoom_coverage = Vec::new();

    // Analyze each zoom level
    for zoom in 0..=14u8 {
        let coverage = analyze_zoom_level(data, zoom);
        zoom_coverage.push(coverage);
        pb.inc(1);
    }

    pb.finish_with_message("Spatial analysis complete");

    // Determine recommended zoom levels
    let (min_zoom, max_zoom) = recommend_zoom_levels(&zoom_coverage, data.features.len());

    // Detect hotspots using a grid-based approach
    let hotspots = detect_hotspots(data);

    // Classify spatial distribution
    let distribution = classify_distribution(&zoom_coverage, &hotspots, &data.bounds);

    Ok(SpatialAnalysis {
        zoom_coverage,
        hotspots,
        recommended_min_zoom: min_zoom,
        recommended_max_zoom: max_zoom,
        distribution,
    })
}

/// Analyze coverage at a specific zoom level
fn analyze_zoom_level(data: &LoadedData, zoom: u8) -> ZoomCoverage {
    let mut tile_counts: HashMap<(u32, u32), usize> = HashMap::new();

    for feature in &data.features {
        if let Ok((x, y)) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
            *tile_counts.entry((x, y)).or_insert(0) += 1;
        }
    }

    let total_tiles = 1u64 << (2 * zoom as u64);
    let occupied_tiles = tile_counts.len() as u64;
    let coverage_percent = if total_tiles > 0 {
        (occupied_tiles as f64 / total_tiles as f64) * 100.0
    } else {
        0.0
    };

    let counts: Vec<usize> = tile_counts.values().copied().collect();
    let avg_features_per_tile = if !counts.is_empty() {
        counts.iter().sum::<usize>() as f64 / counts.len() as f64
    } else {
        0.0
    };

    let max_features_in_tile = counts.iter().copied().max().unwrap_or(0);

    let median_features_per_tile = if !counts.is_empty() {
        let mut sorted = counts.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    } else {
        0
    };

    ZoomCoverage {
        zoom,
        total_tiles,
        occupied_tiles,
        coverage_percent,
        avg_features_per_tile,
        max_features_in_tile,
        median_features_per_tile,
    }
}

/// Recommend min and max zoom levels based on coverage
fn recommend_zoom_levels(coverage: &[ZoomCoverage], _total_features: usize) -> (u8, u8) {
    // Find min zoom: where we have meaningful aggregation
    // Start from zoom 0 and go up until we have reasonable tile counts
    let mut min_zoom = 0u8;
    for cov in coverage.iter() {
        // If average features per tile drops below 5, we might be too zoomed in at this min level
        if cov.avg_features_per_tile < 2.0 && cov.zoom > 0 {
            min_zoom = cov.zoom.saturating_sub(1);
            break;
        }
        min_zoom = cov.zoom;
    }

    // Find max zoom: where tiles are not too sparse
    let mut max_zoom = 14u8;
    for cov in coverage.iter().rev() {
        // If we have reasonable coverage (at least some tiles with data) at this zoom
        if cov.occupied_tiles > 0 && cov.avg_features_per_tile >= 1.0 {
            max_zoom = cov.zoom;
            break;
        }
    }

    // Additional heuristics
    // If data is very sparse at high zooms, reduce max_zoom
    if let Some(cov) = coverage.iter().find(|c| c.zoom == max_zoom) {
        // If less than 0.1% of tiles have data at max zoom, reduce
        if cov.coverage_percent < 0.1 && max_zoom > 6 {
            max_zoom = max_zoom.saturating_sub(2);
        }
    }

    // Ensure min <= max
    if min_zoom > max_zoom {
        min_zoom = 0;
    }

    // Cap at reasonable values
    max_zoom = max_zoom.min(14);

    (min_zoom, max_zoom)
}

/// Detect hotspots using a grid-based density approach
fn detect_hotspots(data: &LoadedData) -> Vec<Hotspot> {
    // Use a coarse grid (about 10 degrees cells)
    let grid_size = 10.0;
    let mut grid_counts: HashMap<(i32, i32), (f64, f64, usize)> = HashMap::new();

    for feature in &data.features {
        let grid_x = (feature.lon / grid_size).floor() as i32;
        let grid_y = (feature.lat / grid_size).floor() as i32;

        let entry = grid_counts.entry((grid_x, grid_y)).or_insert((0.0, 0.0, 0));
        entry.0 += feature.lon;
        entry.1 += feature.lat;
        entry.2 += 1;
    }

    // Find cells with above-average density
    let total_features = data.features.len();
    let avg_per_cell = if !grid_counts.is_empty() {
        total_features as f64 / grid_counts.len() as f64
    } else {
        0.0
    };

    let threshold = avg_per_cell * 2.0; // Hotspot if 2x average

    let mut hotspots: Vec<Hotspot> = grid_counts
        .iter()
        .filter(|(_, (_, _, count))| *count as f64 > threshold && *count > 100)
        .map(|((_gx, _gy), (sum_lon, sum_lat, count))| {
            let center_lon = sum_lon / *count as f64;
            let center_lat = sum_lat / *count as f64;
            Hotspot {
                lon: center_lon,
                lat: center_lat,
                radius: grid_size / 2.0,
                feature_count: *count,
                name: get_region_name(center_lon, center_lat),
            }
        })
        .collect();

    // Sort by feature count (descending)
    hotspots.sort_by(|a, b| b.feature_count.cmp(&a.feature_count));

    // Keep top 10 hotspots
    hotspots.truncate(10);

    hotspots
}

/// Get a rough region name based on coordinates
fn get_region_name(lon: f64, lat: f64) -> Option<String> {
    // Simple heuristic based on major regions
    let name = if lon >= -180.0 && lon <= -100.0 {
        if lat >= 25.0 && lat <= 50.0 {
            Some("Western North America")
        } else if lat >= -60.0 && lat <= 15.0 {
            Some("South America (West)")
        } else {
            None
        }
    } else if lon > -100.0 && lon <= -30.0 {
        if lat >= 25.0 && lat <= 50.0 {
            Some("Eastern North America")
        } else if lat >= -60.0 && lat <= 15.0 {
            Some("South America (East)")
        } else {
            None
        }
    } else if lon > -30.0 && lon <= 60.0 {
        if lat >= 35.0 && lat <= 70.0 {
            Some("Europe")
        } else if lat >= -35.0 && lat <= 35.0 {
            Some("Africa")
        } else {
            None
        }
    } else if lon > 60.0 && lon <= 150.0 {
        if lat >= 20.0 && lat <= 55.0 {
            Some("Asia (Central/East)")
        } else if lat >= -10.0 && lat <= 30.0 {
            Some("South/Southeast Asia")
        } else {
            None
        }
    } else if lon > 100.0 || lon <= -150.0 {
        if lat >= -50.0 && lat <= 0.0 {
            Some("Oceania/Australia")
        } else if lat >= 30.0 && lat <= 45.0 {
            Some("Pacific Ring (Japan/Korea)")
        } else {
            None
        }
    } else {
        None
    };

    name.map(|s| s.to_string())
}

/// Classify the overall spatial distribution
fn classify_distribution(
    coverage: &[ZoomCoverage],
    hotspots: &[Hotspot],
    bounds: &stt_core::types::BoundingBox,
) -> SpatialDistribution {
    // Check bounds extent
    let lon_extent = bounds.max_lon - bounds.min_lon;
    let lat_extent = bounds.max_lat - bounds.min_lat;

    // Very localized if bounds are small
    if lon_extent < 10.0 && lat_extent < 10.0 {
        return SpatialDistribution::Localized;
    }

    // Check coverage at mid-zoom (z6)
    let z6_coverage = coverage.iter().find(|c| c.zoom == 6);

    if let Some(cov) = z6_coverage {
        if cov.coverage_percent < 0.5 {
            return SpatialDistribution::Sparse;
        }

        // If many hotspots with high concentration
        if hotspots.len() >= 3 {
            let hotspot_features: usize = hotspots.iter().take(5).map(|h| h.feature_count).sum();
            let z6_features: usize = coverage
                .iter()
                .find(|c| c.zoom == 6)
                .map(|c| (c.avg_features_per_tile * c.occupied_tiles as f64) as usize)
                .unwrap_or(0);

            if hotspot_features > z6_features / 2 {
                return SpatialDistribution::Regional;
            }
        }

        // Wide coverage
        if cov.coverage_percent > 5.0 && lon_extent > 100.0 {
            return SpatialDistribution::Global;
        }
    }

    SpatialDistribution::Regional
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_name() {
        assert_eq!(get_region_name(-122.0, 37.0), Some("Western North America".to_string()));
        assert_eq!(get_region_name(2.0, 48.0), Some("Europe".to_string()));
        // 139.0 lon, 35.0 lat is in Asia (Central/East) range
        assert_eq!(get_region_name(139.0, 35.0), Some("Asia (Central/East)".to_string()));
    }
}

