//! Wire/layout advisor (`--publish` (zstd 19), `--blob-ordering`,
//! `--pack-size`): compression measured on the sample, ordering from the
//! spatial/temporal access shape.
//!
//! This is the weakest-evidence advice tier: only the zstd trial re-encodes
//! the real sample (and can reach High confidence); blob ordering is an
//! access-shape heuristic and pack sizing a formula estimate, so both are
//! capped at Low confidence. Everything here is byte-level and reversible —
//! `lossy: false` throughout.

use anyhow::Result;

use super::{Advice, AdviceConfidence};
use crate::analysis::spatial::SpatialDistribution;
use crate::analysis::AnalysisResult;
use crate::loader::LoadedData;
use crate::measure::{measure_sample, MeasureSettings, MeasuredEncoding};

/// Measured sample shrink at zstd 19 (vs level 3) at or above this many
/// percent earns High confidence; a smaller measured win stays Medium.
const ZSTD_HIGH_CONFIDENCE_SHRINK_PCT: f64 = 5.0;

/// "Short" playback window for the spatial-major heuristic (~7 days in ms).
const SHORT_DURATION_MS: u64 = 7 * 86_400_000;

/// "Long" playback window for the time-major heuristic (~90 days in ms).
const LONG_DURATION_MS: u64 = 90 * 86_400_000;

/// Only archives estimated above this size (~5 GiB) get pack-size advice —
/// below it the default 64 MiB packs keep the object count small anyway.
const PACK_ADVICE_MIN_ARCHIVE_BYTES: usize = 5 * (1 << 30);

/// The stt-build default pack target (MiB), per docs/api/cli-reference.md.
const DEFAULT_PACK_MIB: usize = 64;

/// Suggested pack target (MiB) for large archives: halves the object count
/// while staying well under the 512 MB CDN per-object cap.
const SUGGESTED_PACK_MIB: usize = 128;

/// Advise on wire/layout levers: publish-grade zstd (measured on the sample
/// when possible), blob ordering from the access shape, and pack sizing for
/// large archives.
pub fn advise(result: &AnalysisResult, data: &LoadedData) -> Result<Vec<Advice>> {
    let mut advice = Vec::new();
    advice.push(publish_advice(result, data)?);
    advice.extend(blob_ordering_advice(result));
    advice.extend(pack_size_advice(result));
    Ok(advice)
}

/// `--publish` (bundles zstd 19; the directory is already paged by default):
/// trial-encode the sample at level 19 against the level-3 baseline. Falls
/// back to typical-range guidance at Low confidence when the sample can't be
/// measured.
fn publish_advice(result: &AnalysisResult, data: &LoadedData) -> Result<Advice> {
    let publish_settings = MeasureSettings {
        zstd_level: 19,
        ..MeasureSettings::default()
    };
    let at_19 = measure_sample(&data.sample, &publish_settings)?;
    // Baseline at the build default (level 3): reuse the analysis measurement
    // when present, otherwise measure it here with identical settings.
    let at_3: Option<MeasuredEncoding> = match &result.measured {
        Some(m) => Some(m.clone()),
        None => measure_sample(&data.sample, &MeasureSettings::default())?,
    };

    if let (Some(base), Some(hi)) = (&at_3, &at_19) {
        if base.bytes_total > 0 {
            let shrink_pct =
                (base.bytes_total as f64 - hi.bytes_total as f64) / base.bytes_total as f64 * 100.0;
            let confidence = if shrink_pct >= ZSTD_HIGH_CONFIDENCE_SHRINK_PCT {
                AdviceConfidence::High
            } else {
                AdviceConfidence::Medium
            };
            return Ok(Advice {
                flag: "--publish".to_string(),
                value: None,
                why: format!(
                    "zstd 19 encodes the {}-feature sample to {} B vs {} B at the \
                     default level 3; decode-free wire savings for deployment \
                     builds; dev builds can stay at 3 for speed",
                    hi.features, hi.bytes_total, base.bytes_total
                ),
                projected: Some(format!(
                    "{:+.1}% sample encode (measured, zstd 3 vs 19)",
                    -shrink_pct
                )),
                lossy: false,
                confidence,
            });
        }
    }

    // No measurable sample: safe generic guidance, clearly sourced as typical
    // (not this dataset's numbers) and downgraded to Low.
    Ok(Advice {
        flag: "--publish".to_string(),
        value: None,
        why: format!(
            "sample too small to trial-encode ({} usable sampled features); \
             zstd 19 typically saves 10..19% wire bytes on STT tiles (typical, \
             not measured on this dataset); decode-free for clients; dev \
             builds can stay at 3 for speed",
            data.sample.len()
        ),
        projected: None,
        lossy: false,
        confidence: AdviceConfidence::Low,
    })
}

/// `--blob-ordering`: only when the access shape clearly matches one of the
/// two strong patterns (Localized + short window -> `spatial`, Global/Regional
/// + long window -> `time-major`); anything else keeps the `auto` default.
fn blob_ordering_advice(result: &AnalysisResult) -> Option<Advice> {
    let duration_ms = result.temporal.duration_ms;
    let duration = &result.temporal.duration_human;
    let (value, why) = match result.spatial.distribution {
        SpatialDistribution::Localized if duration_ms < SHORT_DURATION_MS => (
            "spatial",
            format!(
                "{} features are spatially Localized over only {}: viewport-local \
                 reads dominate, so spatial-major blob order keeps a viewport's \
                 tiles in fewer packs (access-shape heuristic, not simulated)",
                result.feature_count, duration
            ),
        ),
        SpatialDistribution::Global | SpatialDistribution::Regional
            if duration_ms > LONG_DURATION_MS =>
        {
            (
                "time-major",
                format!(
                    "{} features spread {} across {}: playback sweeps time, so \
                 time-major blob order keeps consecutive time buckets in the \
                 same packs (access-shape heuristic, not simulated)",
                    result.feature_count, result.spatial.distribution, duration
                ),
            )
        }
        // Ambiguous access shape: the `auto` default is fine — emitting it
        // would just restate the default.
        _ => return None,
    };
    Some(Advice {
        flag: "--blob-ordering".to_string(),
        value: Some(value.to_string()),
        why,
        projected: None,
        lossy: false,
        confidence: AdviceConfidence::Low,
    })
}

/// `--pack-size 128`: only for archives estimated above ~5 GiB, where the
/// default 64 MiB target starts producing a large pack-object count.
fn pack_size_advice(result: &AnalysisResult) -> Option<Advice> {
    let archive = result.density.estimated_archive_size;
    if archive <= PACK_ADVICE_MIN_ARCHIVE_BYTES {
        return None;
    }
    let gib = archive as f64 / (1u64 << 30) as f64;
    let packs_default = archive.div_ceil(DEFAULT_PACK_MIB << 20);
    let packs_suggested = archive.div_ceil(SUGGESTED_PACK_MIB << 20);
    Some(Advice {
        flag: "--pack-size".to_string(),
        value: Some(SUGGESTED_PACK_MIB.to_string()),
        why: format!(
            "estimated archive ~{:.1} GiB is ~{} pack objects at the default \
             {} MiB; {} MiB caps the object count (CDN/R2 list+range \
             friendliness) while staying well under the 512 MB per-object cap",
            gib, packs_default, DEFAULT_PACK_MIB, SUGGESTED_PACK_MIB
        ),
        projected: Some(format!(
            "~{} pack objects instead of ~{} (estimate)",
            packs_suggested, packs_default
        )),
        lossy: false,
        confidence: AdviceConfidence::Low,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{PropValue, SampledFeature};
    use geo_types::{Geometry, Point};
    use stt_core::types::{BoundingBox, TimeRange};

    const DAY_MS: u64 = 86_400_000;

    /// n spread-out points with a repetitive string property (compressible, so
    /// zstd 19 has real bytes to win over level 3) and a jittered numeric one.
    fn point_sample(n: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| {
                let jitter = |salt: u64| {
                    ((i as u64).wrapping_add(salt).wrapping_mul(2_654_435_761) % 100_000) as f64
                        * 1e-7
                };
                SampledFeature {
                    geometry: Geometry::Point(Point::new(
                        -73.5 + i as f64 * 0.0013 + jitter(0),
                        45.5 + (i % 7) as f64 * 0.0021 + jitter(17),
                    )),
                    timestamp_ms: 1_600_000_000_000 + i as u64 * 1_000,
                    properties: vec![
                        (
                            "magnitude".to_string(),
                            PropValue::Number(1.0 + (i % 90) as f64 * 0.137),
                        ),
                        (
                            "region".to_string(),
                            PropValue::Text(format!("region-{}", i % 5)),
                        ),
                    ],
                }
            })
            .collect()
    }

    fn synthetic_data(sample: Vec<SampledFeature>) -> LoadedData {
        LoadedData {
            features: Vec::new(),
            bounds: BoundingBox::new(-74.0, 45.0, -73.0, 46.0),
            time_range: TimeRange::new(0, DAY_MS),
            sample,
        }
    }

    fn synthetic_result(
        distribution: SpatialDistribution,
        duration_ms: u64,
        estimated_archive_size: usize,
        measured: Option<MeasuredEncoding>,
    ) -> AnalysisResult {
        // The shared default (10 000 points, one uniform day, empty density),
        // with only the fields the layout advisor reads overridden.
        let mut r = crate::test_support::sample_analysis();
        r.spatial.distribution = distribution;
        r.temporal.time_end = duration_ms;
        r.temporal.duration_ms = duration_ms;
        r.temporal.duration_human = format!("{:.1} days", duration_ms as f64 / DAY_MS as f64);
        r.density.estimated_archive_size = estimated_archive_size;
        r.measured = measured;
        r
    }

    fn find<'a>(advice: &'a [Advice], flag: &str) -> Option<&'a Advice> {
        advice.iter().find(|a| a.flag == flag)
    }

    #[test]
    fn measured_publish_advice_projects_negative_shrink() {
        let data = synthetic_data(point_sample(400));
        let measured = measure_sample(&data.sample, &MeasureSettings::default())
            .unwrap()
            .expect("400 features is enough to measure");
        let result = synthetic_result(
            SpatialDistribution::Regional,
            30 * DAY_MS,
            100 << 20,
            Some(measured),
        );

        let advice = advise(&result, &data).unwrap();
        let publish = find(&advice, "--publish").expect("publish advice");
        assert!(publish.value.is_none());
        assert!(!publish.lossy);
        let projected = publish.projected.as_deref().expect("measured projection");
        assert!(
            projected.starts_with('-'),
            "zstd 19 should shrink the sample, got {projected}"
        );
        assert!(projected.contains("measured"));
        // Measured advice never sits at the unmeasured Low tier.
        assert!(!matches!(publish.confidence, AdviceConfidence::Low));
        // Rule 2: the why cites this dataset's sample numbers.
        assert!(publish.why.contains("400-feature sample"));
    }

    #[test]
    fn unmeasurable_sample_downgrades_publish_to_low() {
        let data = synthetic_data(Vec::new());
        let result = synthetic_result(SpatialDistribution::Regional, 30 * DAY_MS, 100 << 20, None);

        let advice = advise(&result, &data).unwrap();
        let publish = find(&advice, "--publish").expect("publish advice");
        assert!(publish.projected.is_none());
        assert!(matches!(publish.confidence, AdviceConfidence::Low));
        assert!(publish.why.contains("typical"));
        assert!(!publish.lossy);
    }

    #[test]
    fn localized_short_dataset_orders_spatial() {
        let data = synthetic_data(Vec::new());
        let result = synthetic_result(SpatialDistribution::Localized, 3 * DAY_MS, 100 << 20, None);

        let advice = advise(&result, &data).unwrap();
        let ordering = find(&advice, "--blob-ordering").expect("blob-ordering advice");
        assert_eq!(ordering.value.as_deref(), Some("spatial"));
        assert!(!ordering.lossy);
        assert!(matches!(ordering.confidence, AdviceConfidence::Low));
        assert!(ordering.why.contains("not simulated"));
    }

    #[test]
    fn global_yearlong_dataset_orders_time_major() {
        let data = synthetic_data(Vec::new());
        let result = synthetic_result(SpatialDistribution::Global, 365 * DAY_MS, 100 << 20, None);

        let advice = advise(&result, &data).unwrap();
        let ordering = find(&advice, "--blob-ordering").expect("blob-ordering advice");
        assert_eq!(ordering.value.as_deref(), Some("time-major"));
        assert!(matches!(ordering.confidence, AdviceConfidence::Low));
    }

    #[test]
    fn ambiguous_access_shape_keeps_auto_ordering() {
        let data = synthetic_data(Vec::new());
        // Regional + 30 days matches neither strong pattern; nor does a
        // Localized dataset with a long window.
        for (dist, days) in [
            (SpatialDistribution::Regional, 30),
            (SpatialDistribution::Localized, 365),
        ] {
            let result = synthetic_result(dist, days * DAY_MS, 100 << 20, None);
            let advice = advise(&result, &data).unwrap();
            assert!(
                find(&advice, "--blob-ordering").is_none(),
                "auto default should not be restated"
            );
        }
    }

    #[test]
    fn small_archive_yields_no_pack_size_advice() {
        let data = synthetic_data(Vec::new());
        let result = synthetic_result(SpatialDistribution::Regional, 30 * DAY_MS, 1 << 30, None);

        let advice = advise(&result, &data).unwrap();
        assert!(find(&advice, "--pack-size").is_none());
    }

    #[test]
    fn large_archive_suggests_128_mib_packs() {
        let data = synthetic_data(Vec::new());
        let result = synthetic_result(SpatialDistribution::Regional, 30 * DAY_MS, 20 << 30, None);

        let advice = advise(&result, &data).unwrap();
        let pack = find(&advice, "--pack-size").expect("pack-size advice");
        assert_eq!(pack.value.as_deref(), Some("128"));
        assert!(matches!(pack.confidence, AdviceConfidence::Low));
        assert!(!pack.lossy);
        assert!(pack.why.contains("20.0 GiB"));
        assert!(pack.why.contains("512 MB"));
        // 20 GiB: 320 packs at 64 MiB, 160 at 128 MiB.
        assert_eq!(
            pack.projected.as_deref(),
            Some("~160 pack objects instead of ~320 (estimate)")
        );
    }
}
