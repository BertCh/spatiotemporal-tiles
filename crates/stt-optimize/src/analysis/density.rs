//! Tile density analysis
//!
//! Buckets features into (zoom, x, y, time-bucket) tiles — the same cut a
//! real `stt-build` run makes with `--temporal-bucket` — to predict tile
//! counts, per-tile feature loads, and archive size, and to flag issues.

use crate::analysis::spatial::SpatialAnalysis;
use crate::analysis::temporal::TemporalAnalysis;
use crate::loader::LoadedData;
use crate::measure::MeasuredEncoding;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use stt_core::projection;

/// Last-resort zstd compression ratio, used ONLY when the sample is too small to
/// measure a real ratio (see [`crate::measure::measure_sample`]). A deliberately
/// conservative rough guess — real STT tiles routinely compress 3–6× — kept as a
/// named constant so this is clearly an unmeasured estimate, not a fact. When a
/// measurement exists the real [`MeasuredEncoding::zstd_ratio`] is used instead.
const FALLBACK_ZSTD_RATIO: f64 = 3.0;

/// Density analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityAnalysis {
    /// Per-zoom tile statistics across the recommended zoom range
    pub per_zoom: Vec<ZoomDensity>,
    /// Estimated total tile count at recommended settings (summed over zooms)
    pub estimated_tile_count: usize,
    /// Estimated archive size in bytes (compressed)
    pub estimated_archive_size: usize,
    /// Potential issues identified
    pub issues: Vec<DensityIssue>,
}

/// Tile statistics for one zoom level, with features split into
/// (x, y, time-bucket) tiles by the recommended temporal bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomDensity {
    /// Zoom level
    pub zoom: u8,
    /// Number of (x, y, time-bucket) tiles at this zoom
    pub tile_count: usize,
    /// Average features per tile
    pub avg_features_per_tile: f64,
    /// Median features per tile
    pub median_features_per_tile: usize,
    /// Maximum features in any tile
    pub max_features_per_tile: usize,
    /// Number of oversized tiles (> 10,000 features)
    pub oversized_tiles: usize,
    /// Number of undersized tiles (< 10 features)
    pub undersized_tiles: usize,
    /// Estimated total size at this zoom, uncompressed (measured-sample
    /// calibrated when a measurement is available, else summed per-feature
    /// formula estimates)
    pub estimated_size_uncompressed: usize,
    /// Estimated total size at this zoom, compressed (measured bytes/feature
    /// and zstd ratio when available, else an assumed 3x ratio)
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

/// Analyze tile density across the recommended zoom range.
///
/// Models the real build: at every zoom in the recommended
/// `[min_zoom, max_zoom]` range each feature lands in its containing (x, y)
/// tile, split by fixed `--temporal-bucket`-sized time buckets (the
/// recommended bucket from the temporal analysis). This is the cut `stt-build`
/// actually makes, so predicted tile counts and sizes track a real build.
///
/// When a `measured` sample encoding is present, size estimates use its real
/// encoder bytes/feature and zstd ratio; otherwise the per-feature formula
/// estimate with an assumed 3x compression ratio is the fallback.
pub fn analyze(
    data: &LoadedData,
    spatial: &SpatialAnalysis,
    temporal: &TemporalAnalysis,
    measured: Option<&MeasuredEncoding>,
) -> Result<DensityAnalysis> {
    let bucket_ms = temporal.recommended_bucket_ms;
    let zooms: Vec<u8> = (spatial.recommended_min_zoom..=spatial.recommended_max_zoom).collect();
    tracing::debug!(
        "density: bucketing {} features into (x, y, t/{}ms) tiles at zooms {:?}",
        data.features.len(),
        bucket_ms,
        zooms
    );

    let mut per_zoom = Vec::with_capacity(zooms.len());
    for &zoom in &zooms {
        per_zoom.push(bucket_zoom(data, zoom, bucket_ms, measured));
    }

    let estimated_tile_count = per_zoom.iter().map(|z| z.tile_count).sum();
    let estimated_archive_size = per_zoom.iter().map(|z| z.estimated_size_compressed).sum();

    let issues = identify_issues(
        data,
        spatial,
        &per_zoom,
        estimated_tile_count,
        estimated_archive_size,
    );

    Ok(DensityAnalysis {
        per_zoom,
        estimated_tile_count,
        estimated_archive_size,
        issues,
    })
}

/// Bucket every feature into its (x, y, time-bucket) tile at one zoom and
/// compute per-tile statistics. `bucket_ms == 0` (no temporal bucketing, e.g.
/// an instantaneous dataset) collapses to a single time bucket per tile.
fn bucket_zoom(
    data: &LoadedData,
    zoom: u8,
    bucket_ms: u64,
    measured: Option<&MeasuredEncoding>,
) -> ZoomDensity {
    // (feature count, estimated bytes) per (x, y, t_bucket) tile.
    let mut tiles: HashMap<(u32, u32, u64), (usize, usize)> = HashMap::new();

    // NOTE: this model assigns each feature to the single tile containing its
    // centroid. It does NOT replicate the builder's trajectory/polygon clipping,
    // which can place one non-point feature into several tiles (and split its
    // bytes across them). So for LineString/Polygon/Multi* inputs the per-tile
    // counts and sizes here are approximate; point datasets are exact.
    for feature in &data.features {
        if let Ok((x, y)) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
            let t_bucket = if bucket_ms > 0 {
                feature.timestamp / bucket_ms
            } else {
                0
            };
            let entry = tiles.entry((x, y, t_bucket)).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += feature.estimated_size;
        }
    }

    let mut feature_counts: Vec<usize> = tiles.values().map(|&(count, _)| count).collect();
    feature_counts.sort_unstable();
    let total_uncompressed: usize = tiles.values().map(|&(_, bytes)| bytes).sum();

    let tile_count = feature_counts.len();
    let avg_features_per_tile = if tile_count > 0 {
        feature_counts.iter().sum::<usize>() as f64 / tile_count as f64
    } else {
        0.0
    };
    let median_features_per_tile = feature_counts.get(tile_count / 2).copied().unwrap_or(0);
    let max_features_per_tile = feature_counts.last().copied().unwrap_or(0);
    // 10,000-feature "oversized" threshold is a rough rule of thumb for a tile
    // that will be slow to decode/render; it is not a hard format limit.
    let oversized_tiles = feature_counts.iter().filter(|&&c| c > 10_000).count();
    let undersized_tiles = feature_counts.iter().filter(|&&c| c < 10).count();

    // Size estimates: calibrated by the measured sample encoding when present
    // (real encoder + real zstd bytes/feature and ratio — the preferred path,
    // now reached for more datasets since MIN_MEASURE_FEATURES was lowered).
    // Only when even the reduced sample can't be measured do we fall back to the
    // summed per-feature formula and a rough assumed compression ratio.
    let (estimated_size_uncompressed, estimated_size_compressed) = match measured {
        Some(m) => {
            let bucketed_features: usize = feature_counts.iter().sum();
            let compressed = (bucketed_features as f64 * m.bytes_per_feature).round() as usize;
            let uncompressed = (compressed as f64 * m.zstd_ratio).round() as usize;
            (uncompressed, compressed)
        }
        None => (
            total_uncompressed,
            (total_uncompressed as f64 / FALLBACK_ZSTD_RATIO).round() as usize,
        ),
    };

    ZoomDensity {
        zoom,
        tile_count,
        avg_features_per_tile,
        median_features_per_tile,
        max_features_per_tile,
        oversized_tiles,
        undersized_tiles,
        estimated_size_uncompressed,
        estimated_size_compressed,
    }
}

/// Identify potential issues from the per-zoom breakdown. Every suggestion
/// names real `stt-build` flags; per-tile budgets are always described as
/// opt-in with a data-loss tradeoff.
fn identify_issues(
    data: &LoadedData,
    spatial: &SpatialAnalysis,
    per_zoom: &[ZoomDensity],
    total_tile_count: usize,
    estimated_archive_size: usize,
) -> Vec<DensityIssue> {
    let mut issues = Vec::new();

    let oversized: usize = per_zoom.iter().map(|z| z.oversized_tiles).sum();
    let undersized: usize = per_zoom.iter().map(|z| z.undersized_tiles).sum();
    let max_features = per_zoom
        .iter()
        .map(|z| z.max_features_per_tile)
        .max()
        .unwrap_or(0);

    // Check for oversized tiles
    if oversized > 0 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!(
                "{} tiles exceed 10,000 features (max: {})",
                oversized, max_features
            ),
            suggestion: "Use a finer --temporal-bucket to spread features over more time \
                         buckets, or opt into per-tile budgets (--maximum-tile-bytes / \
                         --maximum-tile-features, optionally --drop-densest-as-needed) — \
                         budgets drop features to fit, trading data loss for tile size. For \
                         very dense point sets, --summary-tier bakes aggregate overview tiles"
                .to_string(),
        });
    }

    // Check for many undersized tiles
    let undersized_pct = if total_tile_count > 0 {
        undersized as f64 / total_tile_count as f64 * 100.0
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
            suggestion: "Lower --max-zoom or use a coarser --temporal-bucket so tiles \
                         aggregate more features"
                .to_string(),
        });
    }

    // Check for very high tile count
    if total_tile_count > 50_000 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!(
                "High tile count ({}) may impact loading performance",
                total_tile_count
            ),
            suggestion: "Narrow the zoom range (--min-zoom / --max-zoom) or use a coarser \
                         --temporal-bucket"
                .to_string(),
        });
    }

    // Check for sparse data at high zooms
    if let Some(z_max) = spatial
        .zoom_coverage
        .iter()
        .find(|z| z.zoom == spatial.recommended_max_zoom)
    {
        if z_max.coverage_percent < 0.1 {
            issues.push(DensityIssue {
                severity: IssueSeverity::Info,
                description: format!(
                    "Only {:.2}% coverage at zoom {}",
                    z_max.coverage_percent, spatial.recommended_max_zoom
                ),
                suggestion: "Data is sparse at this zoom level; lower --max-zoom".to_string(),
            });
        }
    }

    // Check estimated archive size
    let size_mb = estimated_archive_size as f64 / 1_048_576.0;
    if size_mb > 500.0 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!("Large estimated archive size ({:.1} MB)", size_mb),
            suggestion: "Lower --max-zoom, or opt into per-tile budgets \
                         (--maximum-tile-bytes / --maximum-tile-features, optionally \
                         --drop-densest-as-needed) which drop features to fit (data loss). \
                         For very dense point sets, --summary-tier bakes aggregate overview \
                         tiles instead of full-resolution features"
                .to_string(),
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
                suggestion: "Hotspot tiles will be large; opt-in per-tile budgets \
                             (--maximum-tile-bytes / --maximum-tile-features, which drop \
                             features to fit — data loss) cap them, or a per-feature \
                             --min-zoom-field keeps coarse zooms light by holding minor \
                             features back to deeper zooms"
                    .to_string(),
            });
        }
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{AnalyzableFeature, GeometryType};
    use stt_core::types::{BoundingBox, TimeRange};

    fn feature(lon: f64, lat: f64, timestamp: u64) -> AnalyzableFeature {
        AnalyzableFeature {
            lon,
            lat,
            timestamp,
            geometry_type: GeometryType::Point,
            vertex_count: 1,
            estimated_size: 150,
            property_count: 2,
        }
    }

    /// Build synthetic data spread over a small region with timestamps spread
    /// over ~n² seconds.
    fn make_grid_data(n_side: usize) -> LoadedData {
        let mut features = Vec::new();
        let mut min_lon = f64::MAX;
        let mut max_lon = f64::MIN;
        let mut min_lat = f64::MAX;
        let mut max_lat = f64::MIN;
        for i in 0..n_side {
            for j in 0..n_side {
                let lon = -100.0 + (i as f64) * 0.05;
                let lat = 40.0 + (j as f64) * 0.05;
                min_lon = min_lon.min(lon);
                max_lon = max_lon.max(lon);
                min_lat = min_lat.min(lat);
                max_lat = max_lat.max(lat);
                features.push(feature(lon, lat, (i * n_side + j) as u64 * 1000));
            }
        }
        LoadedData {
            features,
            bounds: BoundingBox::new(min_lon, min_lat, max_lon, max_lat),
            time_range: TimeRange::new(0, 1_000_000),
            sample: Vec::new(),
        }
    }

    #[test]
    fn test_bucket_zoom_splits_by_temporal_bucket() {
        // 100 features at ONE location spread over 100s: with a 10s bucket the
        // single spatial tile must split into 10 (x, y, t) tiles; with no
        // bucketing (bucket_ms = 0) it stays a single tile.
        let features: Vec<_> = (0..100u64)
            .map(|i| feature(-100.0, 40.0, i * 1_000))
            .collect();
        let data = LoadedData {
            features,
            bounds: BoundingBox::new(-100.0, 40.0, -100.0, 40.0),
            time_range: TimeRange::new(0, 100_000),
            sample: Vec::new(),
        };

        let bucketed = bucket_zoom(&data, 10, 10_000, None);
        assert_eq!(bucketed.tile_count, 10);
        assert_eq!(bucketed.max_features_per_tile, 10);
        assert_eq!(bucketed.estimated_size_uncompressed, 100 * 150);

        let unbucketed = bucket_zoom(&data, 10, 0, None);
        assert_eq!(unbucketed.tile_count, 1);
        assert_eq!(unbucketed.max_features_per_tile, 100);
    }

    #[test]
    fn test_measured_calibration_replaces_formula() {
        // A measured sample encoding must drive both size estimates (real
        // bytes/feature and zstd ratio), replacing the formula + /3 fallback.
        let features: Vec<_> = (0..100u64)
            .map(|i| feature(-100.0, 40.0, i * 1_000))
            .collect();
        let data = LoadedData {
            features,
            bounds: BoundingBox::new(-100.0, 40.0, -100.0, 40.0),
            time_range: TimeRange::new(0, 100_000),
            sample: Vec::new(),
        };
        let measured = MeasuredEncoding {
            features: 100,
            geometry_kind: "point".to_string(),
            bytes_total: 4_200,
            bytes_per_feature: 42.0,
            zstd_ratio: 2.0,
            per_column: Vec::new(),
        };

        let calibrated = bucket_zoom(&data, 10, 0, Some(&measured));
        assert_eq!(calibrated.estimated_size_compressed, 100 * 42);
        assert_eq!(calibrated.estimated_size_uncompressed, 100 * 42 * 2);

        // The no-measurement fallback keeps the formula estimates + the named
        // rough compression ratio (no bare magic constant).
        let fallback = bucket_zoom(&data, 10, 0, None);
        assert_eq!(fallback.estimated_size_uncompressed, 100 * 150);
        assert_eq!(
            fallback.estimated_size_compressed,
            (100.0 * 150.0 / FALLBACK_ZSTD_RATIO).round() as usize,
        );
    }

    #[test]
    fn test_analyze_aggregates_across_zoom_range() {
        // End-to-end: analyze() must produce one ZoomDensity per zoom in the
        // recommended range, and the aggregates must sum the per-zoom stats.
        let data = make_grid_data(20); // 400 points
        let spatial = crate::analysis::spatial::analyze(&data).unwrap();
        let temporal = crate::analysis::temporal::analyze(&data).unwrap();
        let density = analyze(&data, &spatial, &temporal, None).unwrap();

        let expected_zooms = (spatial.recommended_min_zoom..=spatial.recommended_max_zoom).count();
        assert_eq!(density.per_zoom.len(), expected_zooms);
        assert_eq!(
            density.estimated_tile_count,
            density.per_zoom.iter().map(|z| z.tile_count).sum::<usize>()
        );
        assert!(density.estimated_archive_size > 0);
        assert!(density.per_zoom.iter().all(|z| z.tile_count > 0));
        // Point data: a deeper zoom can only split (x, y, t) tiles, never merge
        // them, so tile counts are non-decreasing across the range.
        for pair in density.per_zoom.windows(2) {
            assert!(
                pair[1].tile_count >= pair[0].tile_count,
                "z{} tile_count {} < z{} tile_count {}",
                pair[1].zoom,
                pair[1].tile_count,
                pair[0].zoom,
                pair[0].tile_count
            );
        }
    }

    #[test]
    fn test_oversized_issue_names_real_build_flags() {
        // >10k features in one (x, y, t) tile must yield an oversized warning
        // whose suggestion names real stt-build flags.
        let features: Vec<_> = (0..10_001).map(|_| feature(-100.0, 40.0, 0)).collect();
        let data = LoadedData {
            features,
            bounds: BoundingBox::new(-100.0, 40.0, -100.0, 40.0),
            time_range: TimeRange::new(0, 0),
            sample: Vec::new(),
        };
        let spatial = crate::analysis::spatial::analyze(&data).unwrap();
        let temporal = crate::analysis::temporal::analyze(&data).unwrap();
        let density = analyze(&data, &spatial, &temporal, None).unwrap();

        let oversized: usize = density.per_zoom.iter().map(|z| z.oversized_tiles).sum();
        assert!(oversized > 0, "expected oversized tiles");
        let issue = density
            .issues
            .iter()
            .find(|i| i.description.contains("10,000"))
            .expect("oversized issue present");
        assert!(issue.suggestion.contains("--maximum-tile-bytes"));
        assert!(issue.suggestion.contains("--maximum-tile-features"));
        assert!(issue.suggestion.contains("--temporal-bucket"));
    }
}
