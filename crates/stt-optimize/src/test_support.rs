//! Shared fixtures for the advisor + recommendation unit tests.
//!
//! The advisor tests (`advisors::{budget,temporal,layout}`) and
//! `recommend` each need a fully-populated [`AnalysisResult`] to feed the
//! advice logic. Rather than repeat the ~60-line struct literal (spatial +
//! temporal + geometry + density sub-analyses) in every module, they build on
//! [`sample_analysis`] and override only the fields their assertions exercise,
//! composing the recurring sub-analyses from the helpers here.
//!
//! `quantize`'s fixture is intentionally NOT built from these: it runs the real
//! `analysis::{spatial,temporal,geometry,density}::analyze` over sampled data
//! (its precision-ladder assertions depend on real analyzer output), so there
//! is no synthetic literal to share.

use std::collections::HashMap;

use stt_core::types::BoundingBox;

use crate::analysis::density::DensityAnalysis;
use crate::analysis::geometry::{
    GeometryAnalysis, GeometryComplexity, PropertyStats, SizeStats, VertexStats,
};
use crate::analysis::spatial::{SpatialAnalysis, SpatialDistribution};
use crate::analysis::temporal::{EventsPerDayStats, TemporalAnalysis, TemporalDistribution};
use crate::analysis::AnalysisResult;

/// A spatial analysis over an empty coverage/hotspot set at the given zoom
/// window and distribution — the shape every synthetic advisor fixture uses.
pub(crate) fn sample_spatial(
    min_zoom: u8,
    max_zoom: u8,
    distribution: SpatialDistribution,
) -> SpatialAnalysis {
    SpatialAnalysis {
        zoom_coverage: Vec::new(),
        hotspots: Vec::new(),
        recommended_min_zoom: min_zoom,
        recommended_max_zoom: max_zoom,
        distribution,
    }
}

/// All-zero events-per-day stats.
pub(crate) fn no_events() -> EventsPerDayStats {
    EventsPerDayStats {
        min: 0.0,
        max: 0.0,
        avg: 0.0,
        median: 0.0,
        std_dev: 0.0,
    }
}

/// A flat, single-uniform-day temporal analysis (1-hour bucket, no per-day
/// variation) — the default when a test doesn't exercise the temporal advisor.
pub(crate) fn sample_temporal() -> TemporalAnalysis {
    TemporalAnalysis {
        time_start: 0,
        time_end: 86_400_000,
        duration_ms: 86_400_000,
        duration_human: "1 day".to_string(),
        unique_timestamps: 1_000,
        distribution: TemporalDistribution::Uniform,
        recommended_bucket_ms: 3_600_000,
        recommended_bucket_human: "1 hour".to_string(),
        hourly_distribution: vec![0; 24],
        daily_distribution: vec![0; 7],
        monthly_distribution: vec![0; 12],
        events_per_day: no_events(),
    }
}

/// A simple point-geometry analysis: one vertex per feature, a flat 100-byte
/// footprint per feature, two properties each. The `total`s scale with
/// `feature_count`; `type_distribution` is left empty (tests that read the
/// dominant type set their own).
pub(crate) fn point_geometry(feature_count: usize) -> GeometryAnalysis {
    GeometryAnalysis {
        type_distribution: HashMap::new(),
        dominant_type: "Point".to_string(),
        vertex_stats: VertexStats {
            min: 1,
            max: 1,
            avg: 1.0,
            median: 1,
            p95: 1,
            p99: 1,
            total: feature_count,
        },
        size_stats: SizeStats {
            min: 100,
            max: 100,
            avg: 100.0,
            median: 100,
            p95: 100,
            p99: 100,
            total: feature_count * 100,
        },
        property_stats: PropertyStats {
            min: 2,
            max: 2,
            avg: 2.0,
        },
        complexity: GeometryComplexity::Simple,
    }
}

/// A fully-populated default [`AnalysisResult`] — 10 000 point features over a
/// small box, one uniform day, empty density. Tests take this and override the
/// specific fields their assertions exercise.
pub(crate) fn sample_analysis() -> AnalysisResult {
    AnalysisResult {
        source: "synthetic.parquet".to_string(),
        feature_count: 10_000,
        bounds: BoundingBox::new(-74.0, 45.0, -73.0, 46.0),
        spatial: sample_spatial(0, 10, SpatialDistribution::Regional),
        temporal: sample_temporal(),
        geometry: point_geometry(10_000),
        density: DensityAnalysis {
            per_zoom: Vec::new(),
            estimated_tile_count: 100,
            estimated_archive_size: 1 << 20,
            issues: Vec::new(),
        },
        measured: None,
    }
}
