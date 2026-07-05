//! stt-optimize as a library.
//!
//! `stt-build --auto` calls into [`recommend_for`] to pick zoom levels and
//! a temporal bucket from an input file before building.
//! The `stt-optimize` CLI (a binary in the `spatiotemporal-tiles` facade
//! crate) is a thin wrapper around the same functions.

pub mod advisors;
pub mod analysis;
pub mod diff;
pub mod doctor;
pub mod loader;
pub mod measure;
pub mod packed;
pub mod recommend;
pub mod report;

use anyhow::Result;

pub use analysis::inspect::InspectReport;
pub use diff::DiffReport;
pub use doctor::DoctorReport;
pub use loader::DataSource;
pub use packed::PackedTileset;
pub use recommend::Recommendations;

/// Analyze a GeoParquet or STT input and produce build recommendations.
pub fn recommend_for(source: &DataSource) -> Result<Recommendations> {
    let data = loader::load_data(source)?;
    let spatial = analysis::spatial::analyze(&data)?;
    let temporal = analysis::temporal::analyze(&data)?;
    let geometry = analysis::geometry::analyze(&data)?;
    // Measured sample encoding at build defaults; None (formula fallback)
    // when the sample is too small.
    let measured = measure::measure_sample(&data.sample, &measure::MeasureSettings::default())?;
    let density = analysis::density::analyze(&data, &spatial, &temporal, measured.as_ref())?;
    let result = analysis::AnalysisResult {
        source: source.display_name(),
        feature_count: data.features.len(),
        bounds: data.bounds,
        spatial,
        temporal,
        geometry,
        density,
        measured,
    };
    // Evidence-based flag advisors (quantize, temporal, layout, budget) —
    // attached to the recommendations; lossy entries are surface-only.
    let advice = advisors::run_all(&result, &data)?;
    Ok(recommend::generate_recommendations(&result, advice))
}
