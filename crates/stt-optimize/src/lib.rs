//! stt-optimize as a library.
//!
//! `stt-build --auto` calls into [`recommend_for`] to pick zoom levels,
//! temporal bucket, and compression from an input file before building.
//! The CLI in `main.rs` is a thin wrapper around the same functions.

pub mod analysis;
pub mod loader;
pub mod recommend;
pub mod report;

use anyhow::Result;

pub use loader::DataSource;
pub use recommend::Recommendations;

/// Analyze a GeoParquet or STT input and produce build recommendations.
pub fn recommend_for(source: &DataSource) -> Result<Recommendations> {
    let data = loader::load_data(source)?;
    let spatial = analysis::spatial::analyze(&data)?;
    let temporal = analysis::temporal::analyze(&data)?;
    let geometry = analysis::geometry::analyze(&data)?;
    let density = analysis::density::analyze(&data, &spatial)?;
    let result = analysis::AnalysisResult {
        source: source.display_name(),
        feature_count: data.features.len(),
        spatial,
        temporal,
        geometry,
        density,
    };
    Ok(recommend::generate_recommendations(&result))
}
