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

    let pb = ProgressBar::new(MAX_SUPPORTED_ZOOM as u64 + 1); // We analyze z0-z18
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:30.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("##-"),
    );
    pb.set_message("Analyzing spatial coverage");

    let mut zoom_coverage = Vec::new();

    // Analyze each zoom level
    for zoom in 0..=MAX_SUPPORTED_ZOOM {
        let coverage = analyze_zoom_level(data, zoom);
        zoom_coverage.push(coverage);
        pb.inc(1);
    }

    pb.finish_with_message("Spatial analysis complete");

    // Determine recommended zoom levels. The max-zoom is derived from the
    // typical inter-feature spacing (Tippecanoe `-zg` style), then clamped by
    // the coarse occupancy heuristic as a sanity guard.
    let (min_zoom, max_zoom) = recommend_zoom_levels(&zoom_coverage, data, data.features.len());

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

/// Hard cap on the recommended max zoom. The shipped fleet builds tiles up to
/// z18 (e.g. AV LiDAR), so the analysis scans and the recommendation is
/// bounded by the same z0-z18 window.
const MAX_SUPPORTED_ZOOM: u8 = 18;

/// Web Mercator world circumference in meters at the equator (EPSG:3857 extent
/// width = 2·π·a where a = 6_378_137 m). One full map at zoom 0 spans this many
/// meters across, so the ground size of a single tile at zoom `z` is
/// `WORLD_CIRCUMFERENCE_M / 2^z`.
const WORLD_CIRCUMFERENCE_M: f64 = 40_075_016.686;

/// Recommend min and max zoom levels based on coverage and feature density.
fn recommend_zoom_levels(
    coverage: &[ZoomCoverage],
    data: &LoadedData,
    total_features: usize,
) -> (u8, u8) {
    // ---- Min zoom -------------------------------------------------------
    // Find min zoom: the coarsest level that still aggregates meaningfully.
    // Start from zoom 0 and go up until average features per tile drops too low.
    let mut min_zoom = 0u8;
    for cov in coverage.iter() {
        // If average features per tile drops below 2, we are too zoomed in to
        // be a useful *minimum* (overview) level; back off one.
        if cov.avg_features_per_tile < 2.0 && cov.zoom > 0 {
            min_zoom = cov.zoom.saturating_sub(1);
            break;
        }
        min_zoom = cov.zoom;
    }

    // ---- Max zoom (density-derived, Tippecanoe `-zg` style) -------------
    // Pick the zoom at which features sit roughly one tile apart, so detail is
    // preserved without generating wastefully deep, near-empty tiles.
    let density_max_zoom = density_based_max_zoom(data, total_features);

    // ---- Occupancy sanity clamp ----------------------------------------
    // The density estimate is a global average; guard it with the empirical
    // per-zoom occupancy so we never recommend a zoom where the data has
    // already become uselessly sparse. We walk down from the density estimate
    // to the first zoom that still has occupied tiles averaging >= 1 feature.
    let mut max_zoom = density_max_zoom;
    for cov in coverage.iter().rev() {
        if cov.zoom <= density_max_zoom
            && cov.occupied_tiles > 0
            && cov.avg_features_per_tile >= 1.0
        {
            max_zoom = cov.zoom;
            break;
        }
    }

    // Ensure min <= max
    if min_zoom > max_zoom {
        min_zoom = 0;
    }

    // Cap at the analyzed window.
    max_zoom = max_zoom.min(MAX_SUPPORTED_ZOOM);

    (min_zoom, max_zoom)
}

/// Estimate the max zoom from the typical inter-feature spacing.
///
/// # Derivation (Tippecanoe `-zg` analogue)
///
/// We want the deepest zoom `z` at which neighbouring features are about one
/// tile apart. Two quantities drive this:
///
/// * **Typical inter-feature spacing** `s` (meters). As a cheap, allocation-free
///   proxy for the median nearest-neighbour distance we use
///   `s = sqrt(area / n)`, the side length of the square each feature would
///   occupy if `n` features were spread uniformly over the dataset's projected
///   area `area`. (Uniform-packing spacing; for clustered data this slightly
///   under-estimates local spacing, which the occupancy clamp then corrects.)
///
/// * **Tile ground size at zoom `z`** `t(z) = WORLD_CIRCUMFERENCE_M / 2^z`
///   meters (Web Mercator, measured at the data's mean latitude so the
///   longitudinal shrink `cos(lat)` is accounted for).
///
/// Setting `t(z) == s` and solving for `z`:
///
/// ```text
///   WORLD_CIRCUMFERENCE_M * cos(lat) / 2^z  ==  s
///   2^z  ==  WORLD_CIRCUMFERENCE_M * cos(lat) / s
///   z    ==  log2( WORLD_CIRCUMFERENCE_M * cos(lat) / s )
/// ```
///
/// We round to the nearest integer zoom and clamp to `[0, MAX_SUPPORTED_ZOOM]`.
fn density_based_max_zoom(data: &LoadedData, total_features: usize) -> u8 {
    // Degenerate inputs: nothing to derive from.
    if total_features < 2 {
        return 0;
    }

    let b = &data.bounds;
    let lon_extent = (b.max_lon - b.min_lon).abs();
    let lat_extent = (b.max_lat - b.min_lat).abs();

    // Mean latitude drives the east-west meter-per-degree shrink.
    let mean_lat = ((b.min_lat + b.max_lat) / 2.0).clamp(-85.0511, 85.0511);
    let cos_lat = mean_lat.to_radians().cos().max(1e-6);

    // Meters per degree of latitude (constant) and longitude (shrinks with lat).
    let m_per_deg_lat = WORLD_CIRCUMFERENCE_M / 360.0;
    let m_per_deg_lon = m_per_deg_lat * cos_lat;

    // Projected area covered by the data, in square meters. A zero-extent axis
    // (all features share a lon or lat) would zero the area, so floor each axis
    // at a single-tile-at-MAX_SUPPORTED_ZOOM span to keep the estimate finite
    // and push such degenerate-but-real datasets to the deepest zoom.
    let min_span_m = WORLD_CIRCUMFERENCE_M / (1u64 << MAX_SUPPORTED_ZOOM) as f64;
    let width_m = (lon_extent * m_per_deg_lon).max(min_span_m);
    let height_m = (lat_extent * m_per_deg_lat).max(min_span_m);
    let area_m2 = width_m * height_m;

    // Uniform-packing inter-feature spacing.
    let spacing_m = (area_m2 / total_features as f64).sqrt();
    if spacing_m <= 0.0 {
        return MAX_SUPPORTED_ZOOM;
    }

    // z = log2( tile-meters-at-z0 / spacing ), using the latitude-adjusted
    // world width so the tile size matches the spacing's projection.
    let world_width_m = WORLD_CIRCUMFERENCE_M * cos_lat;
    let z = (world_width_m / spacing_m).log2();

    // Round to nearest integer zoom, clamp to the supported window.
    let z = z.round();
    if z.is_nan() || z < 0.0 {
        0
    } else if z > MAX_SUPPORTED_ZOOM as f64 {
        MAX_SUPPORTED_ZOOM
    } else {
        z as u8
    }
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
    use crate::loader::{AnalyzableFeature, GeometryType, LoadedData};
    use stt_core::types::{BoundingBox, TimeRange};

    /// Build a synthetic `LoadedData` from (lon, lat) points. Bounds are derived
    /// from the points so the density estimate sees a realistic extent.
    fn make_data(points: &[(f64, f64)]) -> LoadedData {
        let features: Vec<AnalyzableFeature> = points
            .iter()
            .map(|&(lon, lat)| AnalyzableFeature {
                lon,
                lat,
                timestamp: 0,
                geometry_type: GeometryType::Point,
                vertex_count: 1,
                estimated_size: 120,
                property_count: 1,
            })
            .collect();

        let mut min_lon = f64::MAX;
        let mut max_lon = f64::MIN;
        let mut min_lat = f64::MAX;
        let mut max_lat = f64::MIN;
        for &(lon, lat) in points {
            min_lon = min_lon.min(lon);
            max_lon = max_lon.max(lon);
            min_lat = min_lat.min(lat);
            max_lat = max_lat.max(lat);
        }

        LoadedData {
            features,
            bounds: BoundingBox::new(min_lon, min_lat, max_lon, max_lat),
            time_range: TimeRange::new(0, 0),
            sample: Vec::new(),
        }
    }

    #[test]
    fn test_region_name() {
        assert_eq!(
            get_region_name(-122.0, 37.0),
            Some("Western North America".to_string())
        );
        assert_eq!(get_region_name(2.0, 48.0), Some("Europe".to_string()));
        // 139.0 lon, 35.0 lat is in Asia (Central/East) range
        assert_eq!(
            get_region_name(139.0, 35.0),
            Some("Asia (Central/East)".to_string())
        );
    }

    #[test]
    fn test_density_max_zoom_dense_cluster_is_high() {
        // 2500 points packed into a ~0.01° x 0.01° box near San Francisco
        // (~1.1 km across). Inter-feature spacing is a few tens of meters, so the
        // formula should land near the deepest supported zoom — past the old z14
        // cap now that the window extends to z18.
        let mut pts = Vec::new();
        let n = 50;
        for i in 0..n {
            for j in 0..n {
                let lon = -122.4 + (i as f64) * 0.0002;
                let lat = 37.77 + (j as f64) * 0.0002;
                pts.push((lon, lat));
            }
        }
        let data = make_data(&pts);
        let z = density_based_max_zoom(&data, data.features.len());
        assert!(z >= 15, "dense cluster should yield a high zoom, got {}", z);
        assert!(z <= MAX_SUPPORTED_ZOOM, "zoom must be clamped, got {}", z);
    }

    #[test]
    fn test_density_max_zoom_sparse_global_is_low() {
        // 200 points scattered across the whole globe. Spacing is hundreds of km,
        // so the formula should pick a low (overview) zoom.
        let mut pts = Vec::new();
        let n = 200;
        for i in 0..n {
            // Spread roughly uniformly over lon/lat.
            let lon = -180.0 + (i as f64) * (360.0 / n as f64);
            let lat = -80.0 + ((i * 7) % 160) as f64; // pseudo-spread in lat
            pts.push((lon, lat));
        }
        let data = make_data(&pts);
        let z = density_based_max_zoom(&data, data.features.len());
        assert!(
            z <= 6,
            "sparse global scatter should yield a low zoom, got {}",
            z
        );
    }

    #[test]
    fn test_density_max_zoom_degenerate_inputs() {
        // Fewer than 2 features cannot define spacing -> zoom 0.
        let data = make_data(&[(0.0, 0.0)]);
        assert_eq!(density_based_max_zoom(&data, 1), 0);
        let empty = make_data(&[]);
        assert_eq!(density_based_max_zoom(&empty, 0), 0);
    }

    #[test]
    fn test_recommend_zoom_dense_cluster_high_max() {
        // End-to-end through analyze(): a dense cluster should recommend a high
        // max zoom that survives the occupancy clamp.
        let mut pts = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                pts.push((-122.4 + (i as f64) * 0.0003, 37.77 + (j as f64) * 0.0003));
            }
        }
        let data = make_data(&pts);
        let analysis = analyze(&data).unwrap();
        assert!(
            analysis.recommended_max_zoom >= 11,
            "dense cluster recommended_max_zoom too low: {}",
            analysis.recommended_max_zoom
        );
        assert!(analysis.recommended_min_zoom <= analysis.recommended_max_zoom);
    }
}
