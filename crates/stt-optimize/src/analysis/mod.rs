//! Analysis modules for spatiotemporal data
//!
//! This module provides various analyzers for understanding dataset characteristics
//! and deriving optimization recommendations.

pub mod spatial;
pub mod temporal;
pub mod geometry;
pub mod density;
pub mod inspect;
pub mod properties;

use serde::{Deserialize, Serialize};

/// Combined analysis result from all analyzers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Source file name
    pub source: String,
    /// Total feature count
    pub feature_count: usize,
    /// Dataset spatial bounds (WGS84 lon/lat)
    pub bounds: stt_core::types::BoundingBox,
    /// Spatial analysis results
    pub spatial: spatial::SpatialAnalysis,
    /// Temporal analysis results
    pub temporal: temporal::TemporalAnalysis,
    /// Geometry analysis results
    pub geometry: geometry::GeometryAnalysis,
    /// Density analysis results
    pub density: density::DensityAnalysis,
    /// Measured sample encoding (real encoder + zstd on the loader's stride
    /// sample); `None` when the sample was too small to measure.
    pub measured: Option<crate::measure::MeasuredEncoding>,
}


