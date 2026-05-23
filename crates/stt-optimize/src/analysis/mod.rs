//! Analysis modules for spatiotemporal data
//!
//! This module provides various analyzers for understanding dataset characteristics
//! and deriving optimization recommendations.

pub mod spatial;
pub mod temporal;
pub mod geometry;
pub mod density;

use serde::{Deserialize, Serialize};

/// Combined analysis result from all analyzers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Source file name
    pub source: String,
    /// Total feature count
    pub feature_count: usize,
    /// Spatial analysis results
    pub spatial: spatial::SpatialAnalysis,
    /// Temporal analysis results
    pub temporal: temporal::TemporalAnalysis,
    /// Geometry analysis results
    pub geometry: geometry::GeometryAnalysis,
    /// Density analysis results
    pub density: density::DensityAnalysis,
}

impl AnalysisResult {
    /// Get a summary string
    #[allow(dead_code)]
    pub fn summary(&self) -> String {
        format!(
            "{} features, {} geometry, {}",
            self.feature_count,
            self.geometry.dominant_type,
            self.temporal.time_range_description()
        )
    }
}

