//! Coordinate / attribute quantization advisor (`--quantize-coords`,
//! `--quantize-attrs-auto`): derives candidate precision from the max-zoom
//! ground resolution and verifies the win by trial-encoding the sample.
//!
//! Quantization is the repo's #1 measured size lever (coords −25..47% on
//! coord-heavy datasets, attrs up to −80% per Float64 column), but it is
//! LOSSY, so this advisor never speaks without evidence: every recommendation
//! is backed by a real [`crate::measure::measure_sample`] trial encode
//! through the production stt-core encoder, compared against ONE shared
//! baseline measurement at build defaults. Too-small samples produce no
//! advice at all.

use anyhow::Result;

use super::{Advice, AdviceConfidence};
use crate::analysis::AnalysisResult;
use crate::loader::{LoadedData, PropValue, SampledFeature};
use crate::measure::{measure_sample, MeasureSettings, MeasuredEncoding};

/// WGS84 equatorial circumference in meters — the `2πR` in the standard
/// tile-pyramid ground-resolution formula.
const EARTH_CIRCUMFERENCE_M: f64 = 40_075_016.686;

/// Candidate `--quantize-coords` precisions in meters, descending. A derived
/// precision snaps DOWN to the first entry it covers and never goes below the
/// last (0.01 m is already sub-centimeter — finer buys nothing).
const PRECISION_LADDER_M: [f64; 6] = [5.0, 1.0, 0.5, 0.1, 0.05, 0.01];

/// Minimum measured shrink to recommend `--quantize-coords`.
const MIN_COORD_SHRINK: f64 = 0.05;

/// Minimum measured shrink to recommend `--quantize-attrs-auto`.
const MIN_ATTR_SHRINK: f64 = 0.03;

/// Measured shrink at or above this is `High` confidence, below is `Medium`.
const HIGH_CONFIDENCE_SHRINK: f64 = 0.15;

/// Recommend coordinate / attribute quantization when a trial encode of the
/// loader sample measures a real shrink versus the build-default baseline.
pub fn advise(result: &AnalysisResult, data: &LoadedData) -> Result<Vec<Advice>> {
    // ONE baseline at build defaults, shared by both trials. The analysis
    // pipeline usually measured it already; re-measure only when absent.
    let baseline = match result.measured.clone() {
        Some(measured) => Some(measured),
        None => measure_sample(&data.sample, &MeasureSettings::default())?,
    };
    let Some(baseline) = baseline else {
        // Sample too small to trial-encode. Quantization is lossy, so no
        // unmeasured claims — emit nothing.
        return Ok(Vec::new());
    };

    let mut advice = Vec::new();
    if let Some(coords) = coords_advice(result, data, &baseline)? {
        advice.push(coords);
    }
    if let Some(attrs) = attrs_advice(data, &baseline)? {
        advice.push(attrs);
    }
    Ok(advice)
}

/// `--quantize-coords`: candidate precision = a quarter-pixel of ground
/// resolution at the recommended max zoom (rendering cannot show the
/// discarded bits), snapped down the [`PRECISION_LADDER_M`]; emitted only
/// when the trial encode shrinks the sample by [`MIN_COORD_SHRINK`] or more.
fn coords_advice(
    result: &AnalysisResult,
    data: &LoadedData,
    baseline: &MeasuredEncoding,
) -> Result<Option<Advice>> {
    let max_zoom = result.spatial.recommended_max_zoom;
    let lat_mid = (result.bounds.min_lat + result.bounds.max_lat) / 2.0;
    let m_per_px = meters_per_pixel(max_zoom, lat_mid);
    let candidate = snap_down_precision(m_per_px / 4.0);

    let trial = measure_sample(
        &data.sample,
        &MeasureSettings {
            quantize_coords_m: Some(candidate),
            ..MeasureSettings::default()
        },
    )?;
    let Some(trial) = trial else {
        return Ok(None);
    };
    let shrink = shrink_vs(baseline, &trial);
    if shrink < MIN_COORD_SHRINK {
        return Ok(None);
    }

    let resolution = format!(
        "at max zoom {max_zoom} one pixel covers ~{m_per_px:.2} m at lat {lat_mid:.1}°, \
         so {candidate} m fixed-point coords stay below a quarter-pixel of error"
    );
    let why = match column_share(baseline, "geometry") {
        Some(share) => format!(
            "geometry is {:.0}% of measured column bytes; {resolution}",
            share * 100.0
        ),
        None => resolution,
    };
    Ok(Some(Advice {
        flag: "--quantize-coords".to_string(),
        value: Some(candidate.to_string()),
        why,
        projected: Some(projected_shrink(shrink)),
        lossy: true,
        confidence: confidence_for(shrink),
    }))
}

/// `--quantize-attrs-auto`: when the sample carries fractional Float64
/// properties, trial-encode with range-adaptive UInt16 attribute quantization
/// (coords untouched so the effect is attributable); emitted only when the
/// measured shrink reaches [`MIN_ATTR_SHRINK`].
fn attrs_advice(data: &LoadedData, baseline: &MeasuredEncoding) -> Result<Option<Advice>> {
    let float_cols = fractional_property_names(&data.sample);
    if float_cols.is_empty() {
        return Ok(None);
    }

    let trial = measure_sample(
        &data.sample,
        &MeasureSettings {
            quantize_attrs_auto: true,
            ..MeasureSettings::default()
        },
    )?;
    let Some(trial) = trial else {
        return Ok(None);
    };
    let shrink = shrink_vs(baseline, &trial);
    if shrink < MIN_ATTR_SHRINK {
        return Ok(None);
    }

    // Name the top float column(s) by measured baseline share (per_column is
    // already sorted descending by bytes).
    let mut cited: Vec<String> = baseline
        .per_column
        .iter()
        .filter(|c| float_cols.iter().any(|name| name == &c.name))
        .take(2)
        .map(|c| {
            format!(
                "`{}` ({:.0}% of measured column bytes)",
                c.name,
                c.share * 100.0
            )
        })
        .collect();
    if cited.is_empty() {
        cited = float_cols
            .iter()
            .take(2)
            .map(|name| format!("`{name}`"))
            .collect();
    }
    let why = format!(
        "near-incompressible Float64 propert{} {} shrink to range-adaptive UInt16 (~65k levels)",
        if cited.len() == 1 { "y" } else { "ies" },
        cited.join(" and ")
    );
    Ok(Some(Advice {
        flag: "--quantize-attrs-auto".to_string(),
        value: None,
        why,
        projected: Some(projected_shrink(shrink)),
        lossy: true,
        confidence: confidence_for(shrink),
    }))
}

/// Ground resolution of one 256px-tile pixel at `zoom`, at latitude
/// `lat_deg`: `circumference * cos(lat) / (256 * 2^zoom)`.
fn meters_per_pixel(zoom: u8, lat_deg: f64) -> f64 {
    EARTH_CIRCUMFERENCE_M * lat_deg.to_radians().cos() / (256.0 * f64::powi(2.0, zoom as i32))
}

/// Snap a derived precision DOWN to the [`PRECISION_LADDER_M`] (the first
/// rung it covers); precisions below the ladder clamp to its finest rung.
fn snap_down_precision(precision_m: f64) -> f64 {
    for &rung in &PRECISION_LADDER_M {
        if precision_m >= rung {
            return rung;
        }
    }
    PRECISION_LADDER_M[PRECISION_LADDER_M.len() - 1]
}

/// Measured baseline share of the named column, if it was attributed.
fn column_share(baseline: &MeasuredEncoding, name: &str) -> Option<f64> {
    baseline
        .per_column
        .iter()
        .find(|c| c.name == name)
        .map(|c| c.share)
}

/// Fractional shrink of a trial encode versus the baseline (positive =
/// smaller, negative = the trial got BIGGER).
fn shrink_vs(baseline: &MeasuredEncoding, trial: &MeasuredEncoding) -> f64 {
    1.0 - trial.bytes_total as f64 / baseline.bytes_total.max(1) as f64
}

/// The `projected` string for a measured shrink, e.g. `-36% sample encode
/// (measured)`.
fn projected_shrink(shrink: f64) -> String {
    format!("-{:.0}% sample encode (measured)", shrink * 100.0)
}

fn confidence_for(shrink: f64) -> AdviceConfidence {
    if shrink >= HIGH_CONFIDENCE_SHRINK {
        AdviceConfidence::High
    } else {
        AdviceConfidence::Medium
    }
}

/// Sampled property names carrying at least one finite fractional numeric
/// value, in first-seen order. The loader widens every numeric Arrow type to
/// f64 (dropping the source type), so a fractional value is the evidence that
/// a column is genuinely Float64 — integer-valued columns lose nothing to
/// `--quantize-attrs-auto` being skipped and never trigger it here.
fn fractional_property_names(sample: &[SampledFeature]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for feature in sample {
        for (name, value) in &feature.properties {
            if let PropValue::Number(x) = value {
                if x.is_finite() && x.fract() != 0.0 && !names.iter().any(|n| n == name) {
                    names.push(name.clone());
                }
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis;
    use crate::loader::{AnalyzableFeature, GeometryType};
    use geo_types::{Geometry, Point};
    use stt_core::types::{BoundingBox, TimeRange};

    /// Deterministic pseudo-noise in [0, 1) (Knuth multiplicative hash) —
    /// keeps f64 mantissas high-entropy so quantization has real bytes to win.
    fn noise(i: usize, salt: u64) -> f64 {
        ((i as u64).wrapping_add(salt).wrapping_mul(2_654_435_761) % 1_000_000) as f64 / 1e6
    }

    /// n Montréal-area points. `float_prop` picks a high-precision fractional
    /// f64 property (`magnitude`) versus an integer-valued one (`count`); a
    /// categorical `region` string rides along either way.
    fn point_sample(n: usize, float_prop: bool) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| {
                let numeric = if float_prop {
                    (
                        "magnitude".to_string(),
                        PropValue::Number(1.0 + noise(i, 7) * 9.0),
                    )
                } else {
                    ("count".to_string(), PropValue::Number((i % 10) as f64))
                };
                SampledFeature {
                    geometry: Geometry::Point(Point::new(
                        -73.8 + (i as f64 * 0.003) % 0.6 + noise(i, 0) * 1e-3,
                        45.2 + (i % 7) as f64 * 0.08 + noise(i, 17) * 1e-3,
                    )),
                    timestamp_ms: 1_600_000_000_000 + i as u64 * 60_000,
                    properties: vec![
                        numeric,
                        (
                            "region".to_string(),
                            PropValue::Text(format!("region-{}", i % 5)),
                        ),
                    ],
                }
            })
            .collect()
    }

    fn loaded(sample: Vec<SampledFeature>) -> LoadedData {
        let features = sample
            .iter()
            .map(|f| {
                let (lon, lat) = match &f.geometry {
                    Geometry::Point(p) => (p.x(), p.y()),
                    _ => (0.0, 0.0),
                };
                AnalyzableFeature {
                    lon,
                    lat,
                    timestamp: f.timestamp_ms,
                    geometry_type: GeometryType::Point,
                    vertex_count: 1,
                    estimated_size: 25,
                    property_count: f.properties.len(),
                }
            })
            .collect();
        LoadedData {
            features,
            bounds: BoundingBox {
                min_lon: -73.8,
                min_lat: 45.2,
                max_lon: -73.2,
                max_lat: 45.8,
            },
            time_range: TimeRange {
                start: 1_600_000_000_000,
                end: 1_600_012_000_000,
            },
            sample,
        }
    }

    /// Real-analyzer AnalysisResult over the synthetic data, with the
    /// recommended max zoom pinned so the precision-ladder derivation is
    /// deterministic.
    fn analysis_result(data: &LoadedData, max_zoom: u8) -> AnalysisResult {
        let mut spatial = analysis::spatial::analyze(data).unwrap();
        spatial.recommended_max_zoom = max_zoom;
        spatial.recommended_min_zoom = spatial.recommended_min_zoom.min(max_zoom);
        let temporal = analysis::temporal::analyze(data).unwrap();
        let geometry = analysis::geometry::analyze(data).unwrap();
        let measured = measure_sample(&data.sample, &MeasureSettings::default()).unwrap();
        let density =
            analysis::density::analyze(data, &spatial, &temporal, measured.as_ref()).unwrap();
        AnalysisResult {
            source: "synthetic.parquet".to_string(),
            feature_count: data.features.len(),
            bounds: data.bounds,
            spatial,
            temporal,
            geometry,
            density,
            measured,
        }
    }

    #[test]
    fn high_precision_points_get_measured_coords_advice() {
        let data = loaded(point_sample(200, true));
        let result = analysis_result(&data, 14);
        let advice = advise(&result, &data).unwrap();
        let coords = advice
            .iter()
            .find(|a| a.flag == "--quantize-coords")
            .expect("high-entropy f64 coords must produce measured coords advice");
        // z14 at lat 45.5 → ~6.7 m/px, /4 = 1.67 m → snaps down to 1 m.
        assert_eq!(coords.value.as_deref(), Some("1"));
        assert!(coords.lossy);
        let projected = coords.projected.as_deref().unwrap();
        assert!(
            projected.starts_with('-') && projected.contains("(measured)"),
            "projected = {projected}"
        );
        assert!(
            coords.why.contains("zoom 14") && coords.why.contains("1 m"),
            "why = {}",
            coords.why
        );
    }

    #[test]
    fn fractional_float_property_gets_attrs_advice() {
        let data = loaded(point_sample(200, true));
        let result = analysis_result(&data, 14);
        let advice = advise(&result, &data).unwrap();
        let attrs = advice
            .iter()
            .find(|a| a.flag == "--quantize-attrs-auto")
            .expect("high-entropy Float64 property must produce attrs advice");
        assert!(attrs.lossy);
        assert!(attrs.value.is_none());
        assert!(attrs.why.contains("magnitude"), "why = {}", attrs.why);
        assert!(attrs.projected.as_deref().unwrap().contains("(measured)"));
    }

    #[test]
    fn tiny_sample_produces_no_advice() {
        let data = loaded(point_sample(30, true));
        let result = analysis_result(&data, 14);
        assert!(
            result.measured.is_none(),
            "under 50 features must not measure"
        );
        let advice = advise(&result, &data).unwrap();
        assert!(
            advice.is_empty(),
            "lossy advice without measurement: {advice:?}"
        );
    }

    #[test]
    fn integer_and_string_properties_get_no_attrs_advice() {
        let data = loaded(point_sample(200, false));
        let result = analysis_result(&data, 14);
        let advice = advise(&result, &data).unwrap();
        assert!(
            !advice.iter().any(|a| a.flag == "--quantize-attrs-auto"),
            "integer-valued numeric props must not trigger attr quantization"
        );
    }

    #[test]
    fn precision_snaps_down_the_ladder_and_clamps() {
        assert_eq!(snap_down_precision(23.0), 5.0);
        assert_eq!(snap_down_precision(5.0), 5.0);
        assert_eq!(snap_down_precision(1.67), 1.0);
        assert_eq!(snap_down_precision(0.7), 0.5);
        assert_eq!(snap_down_precision(0.09), 0.05);
        assert_eq!(snap_down_precision(0.002), 0.01);
    }
}
