//! Temporal shaping advisor (`--temporal-lod`, `--adaptive-temporal`): tier
//! and window suggestions from the temporal distribution and bucket count.
//!
//! `--temporal-lod` is additive (lossless coarse-bucket replicas on top of the base
//! buckets) and `--adaptive-temporal` rebuckets windows without dropping
//! features, so both are `lossy: false` — the LOD tradeoff (extra tiers grow
//! the archive) is called out in `why` instead. `--vertex-time-precision` is
//! deliberately NOT advised here: source features carry no per-vertex times,
//! so that lever belongs to the packed-side doctor.
//!
//! Neither lever is a sample-encode measurement — both read the temporal
//! distribution and the density model — and neither maps onto an encoder knob a
//! [`MeasureSettings`](crate::measure::MeasureSettings) point can carry. So
//! [`crate::advisors::run_iterative`] re-prices nothing here: these advisories
//! pass through every round verbatim, which is why this module has no composed
//! entry point. (What a coarser bucket costs in BYTES is a rebuild-shaped
//! question — it changes the tiling, not the encoding of a fixed sample — and
//! belongs to the two-pass build, not to a sample oracle.)

use anyhow::Result;

use super::{Advice, AdviceConfidence};
use crate::analysis::density::ZoomDensity;
use crate::analysis::temporal::TemporalDistribution;
use crate::analysis::AnalysisResult;
use crate::loader::LoadedData;

/// Frames one watchable playback loop shows: `TARGET_PLAYBACK_SECONDS` (45) ×
/// `PLAYBACK_FPS` (30) in [`crate::analysis::temporal`], where the recommended
/// bucket width is derived from exactly this model. Mirrored here because those
/// two constants are private to the analyzer; if they move, move this.
const PLAYBACK_LOOP_FRAMES: u64 = 45 * 30;

/// Base-bucket count above which a coarse `--temporal-lod` pyramid pays for its
/// bytes: past this, a client scanning the whole timeline at one zoom is
/// fetching more buckets than half a watchable playback loop can even display
/// as distinct frames, so zoomed-out and scrubbing views over-fetch by
/// construction.
///
/// # ⚠️ Calibration — this constant was unreachable (F7)
///
/// It was `5_000`, and **nothing could ever clear it.**
/// `analysis::temporal::recommend_bucket_size` caps its own target bucket count
/// at `45 s × 30 fps × 1.5 = 2025` (the 1.5 is the Bursty frame factor; Uniform
/// and Periodic get 1350, Sparse 675) and then returns the FINEST standard
/// bucket whose count is `<= target`. So the recommended base bucket count is
/// `<= 2025` on every dataset except one that falls through to the 30-day
/// fallback, which needs a span of over 411 years to reach 5000 buckets. The
/// trigger — and, through the mirror this constant now feeds, the byte-budget
/// solver's whole temporal-LOD lever — was dead code.
///
/// The value is therefore calibrated against the distribution of the quantity
/// it actually tests, not against a guess about request counts: any threshold
/// at or above 1350 is unreachable for every non-Bursty dataset, so the largest
/// figure in the recommender's own playback model that its output can clear is
/// `PLAYBACK_LOOP_FRAMES / 2` — which is also the bucket-count target the
/// recommender itself uses for Sparse data. Measured over the repo's local
/// corpus it fires on the long-timeline datasets (osm-nyc-changesets 1068
/// buckets, goes-glm-lightning 864, cmorph-2h 729) and stays quiet on the
/// short-timeline ones (wpc-fronts 18, storm4d-isolines 56, hrrr 285,
/// bixi-points 287, cpc-rainfall 364).
///
/// Rollback is a one-line re-pin to `5_000`, which restores "never fires".
pub const LOD_BUCKET_THRESHOLD: u64 = PLAYBACK_LOOP_FRAMES / 2;

/// Minimum factor between the base bucket and the first tier (and between
/// successive tiers): a tier that only doubles the bucket size doesn't buy
/// enough aggregation to justify the extra tiles.
const TIER_STEP_FACTOR: u64 = 4;

/// Max/median features-per-tile skew (at the max recommended zoom) above
/// which adaptive windows are advised for bursty data.
const ADAPTIVE_SKEW_THRESHOLD: f64 = 10.0;

/// Skew below this is only borderline over the threshold → Low confidence.
const ADAPTIVE_SKEW_CONFIDENT: f64 = 20.0;

/// `--adaptive-temporal` window-size clamp (features per window).
const ADAPTIVE_N_MIN: u64 = 1_000;
const ADAPTIVE_N_MAX: u64 = 50_000;

/// Clean calendar-unit tier durations, ascending, expressed in stt-build's
/// duration syntax (`parse_temporal_lod` accepts ms/s/m/h/d). Tiers are only
/// picked from entries that are STRICT multiples of the recommended base
/// bucket, so the emitted spec always passes stt-build's multiple-of check
/// (e.g. base 1h → `1d,30d`; base 1m → `1h,1d`).
const CLEAN_TIERS: &[(u64, &str)] = &[
    (1_000, "1s"),
    (60_000, "1m"),
    (3_600_000, "1h"),
    (86_400_000, "1d"),
    (2_592_000_000, "30d"),
];

/// Advise temporal shaping flags from the temporal distribution, base bucket
/// count, and per-zoom density skew. When both a coarse-tier pyramid and
/// adaptive windows apply (bursty long timeline), both are kept but each
/// `why` notes the interaction.
pub fn advise(result: &AnalysisResult, _data: &LoadedData) -> Result<Vec<Advice>> {
    let mut lod = advise_temporal_lod(result);
    let mut adaptive = advise_adaptive_temporal(result);
    if let (Some(lod), Some(adaptive)) = (lod.as_mut(), adaptive.as_mut()) {
        const INTERACTION: &str = " Both a coarse-tier pyramid and adaptive windows apply here \
             (bursty long timeline) — consider one or the other first.";
        lod.why.push_str(INTERACTION);
        adaptive.why.push_str(INTERACTION);
    }
    Ok(lod.into_iter().chain(adaptive).collect())
}

/// `--temporal-lod`: emit when the base bucket count (duration / recommended
/// bucket) exceeds [`LOD_BUCKET_THRESHOLD`], with 1-2 clean-duration tiers
/// that are strict multiples of the base bucket.
fn advise_temporal_lod(result: &AnalysisResult) -> Option<Advice> {
    let t = &result.temporal;
    if t.recommended_bucket_ms == 0 || t.duration_ms == 0 {
        return None;
    }
    let bucket_count = t.duration_ms / t.recommended_bucket_ms;
    if bucket_count <= LOD_BUCKET_THRESHOLD {
        return None;
    }

    let tiers = pick_tiers(t.recommended_bucket_ms, t.duration_ms);
    let (coarsest_ms, coarsest_label) = *tiers.last()?;
    let value = tiers
        .iter()
        .map(|&(_, label)| label)
        .collect::<Vec<_>>()
        .join(",");
    let coarsest_buckets = (t.duration_ms / coarsest_ms).max(1);

    Some(Advice {
        flag: "--temporal-lod".to_string(),
        value: Some(value.clone()),
        why: format!(
            "{} span at the recommended {} bucket = {} time buckets; zoomed-out full-range \
             playback would touch thousands of buckets, so lossless coarse-bucket tier(s) {} cap \
             request count. Every feature is preserved. Tradeoff: tiers are ADDITIVE — extra \
             coarse tiles grow the archive.",
            t.duration_human, t.recommended_bucket_human, bucket_count, value
        ),
        projected: Some(format!(
            "full-range scan touches ~{} buckets at the coarsest {} tier vs {} at base \
             (estimated; archive grows by the added tier tiles)",
            coarsest_buckets, coarsest_label, bucket_count
        )),
        lossy: false,
        suggestion_only: false,
        confidence: AdviceConfidence::Medium,
    })
}

/// Pick 1-2 tiers from [`CLEAN_TIERS`]: each must be a strict multiple of the
/// base bucket, at least [`TIER_STEP_FACTOR`]× coarser than the previous
/// level, and still leave >= 2 buckets over the dataset's span.
fn pick_tiers(base_ms: u64, duration_ms: u64) -> Vec<(u64, &'static str)> {
    let mut tiers: Vec<(u64, &'static str)> = Vec::new();
    let mut floor = base_ms;
    for &(tier_ms, label) in CLEAN_TIERS {
        if tiers.len() == 2 {
            break;
        }
        if tier_ms % base_ms != 0 || tier_ms < floor.saturating_mul(TIER_STEP_FACTOR) {
            continue;
        }
        if duration_ms / tier_ms < 2 {
            break;
        }
        tiers.push((tier_ms, label));
        floor = tier_ms;
    }
    tiers
}

/// `--adaptive-temporal <N>`: emit when the temporal distribution is Bursty
/// (events/day CoV > 1.5, classified upstream) AND the density model shows
/// high max/median features-per-tile skew at the max recommended zoom.
fn advise_adaptive_temporal(result: &AnalysisResult) -> Option<Advice> {
    if !matches!(result.temporal.distribution, TemporalDistribution::Bursty) {
        return None;
    }
    let zd = density_at_max_zoom(result)?;
    if zd.max_features_per_tile == 0 {
        return None;
    }
    let median = zd.median_features_per_tile.max(1);
    let skew = zd.max_features_per_tile as f64 / median as f64;
    if skew < ADAPTIVE_SKEW_THRESHOLD {
        return None;
    }

    let n = round_nice(median as f64 * 4.0).clamp(ADAPTIVE_N_MIN, ADAPTIVE_N_MAX);
    let epd = &result.temporal.events_per_day;
    let cov = if epd.avg > 0.0 {
        epd.std_dev / epd.avg
    } else {
        0.0
    };
    let confidence = if skew >= ADAPTIVE_SKEW_CONFIDENT {
        AdviceConfidence::Medium
    } else {
        AdviceConfidence::Low
    };

    Some(Advice {
        flag: "--adaptive-temporal".to_string(),
        value: Some(n.to_string()),
        why: format!(
            "bursty timeline (events/day CoV {:.1}: std dev {:.0} vs avg {:.0}) with skewed \
             z{} tiles — max {} vs median {} features/tile ({:.0}x): ~{}-feature adaptive \
             windows bucket dense bursts finer and sparse gaps coarser, no data dropped.",
            cov, epd.std_dev, epd.avg, zd.zoom, zd.max_features_per_tile, median, skew, n
        ),
        projected: None,
        lossy: false,
        suggestion_only: false,
        confidence,
    })
}

/// Density row at the max recommended zoom, falling back to the deepest
/// modeled zoom when the exact level is missing.
fn density_at_max_zoom(result: &AnalysisResult) -> Option<&ZoomDensity> {
    result
        .density
        .per_zoom
        .iter()
        .find(|z| z.zoom == result.spatial.recommended_max_zoom)
        .or_else(|| result.density.per_zoom.last())
}

/// Snap to the log-nearest "nice" number (1/2/5 × 10^k), e.g. 3_200 → 5_000,
/// 480 → 500.
fn round_nice(x: f64) -> u64 {
    if x <= 0.0 {
        return 1;
    }
    let pow = 10f64.powf(x.log10().floor());
    let mantissa = x / pow; // [1, 10)
    let nice = [1.0f64, 2.0, 5.0, 10.0]
        .into_iter()
        .min_by(|a, b| {
            let ra = (mantissa / a).max(a / mantissa);
            let rb = (mantissa / b).max(b / mantissa);
            ra.partial_cmp(&rb).expect("finite ratios")
        })
        .expect("non-empty candidates");
    (nice * pow).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::density::{DensityAnalysis, ZoomDensity};
    use crate::analysis::spatial::SpatialDistribution;
    use crate::analysis::temporal::{EventsPerDayStats, TemporalAnalysis, TemporalDistribution};
    use crate::analysis::AnalysisResult;
    use crate::loader::LoadedData;
    use stt_core::types::{BoundingBox, TimeRange};

    const HOUR_MS: u64 = 3_600_000;
    const DAY_MS: u64 = 86_400_000;

    fn empty_data() -> LoadedData {
        LoadedData {
            features: Vec::new(),
            bounds: BoundingBox::new(-74.0, 40.0, -73.0, 41.0),
            time_range: TimeRange::new(0, 1),
            sample: Vec::new(),
        }
    }

    fn zoom_density(zoom: u8, median: usize, max: usize) -> ZoomDensity {
        ZoomDensity {
            zoom,
            tile_count: 200,
            avg_features_per_tile: median as f64 * 1.5,
            median_features_per_tile: median,
            max_features_per_tile: max,
            p95_features_per_tile: max,
            top1pct_feature_share: 0.0,
            oversized_tiles: 0,
            undersized_tiles: 0,
            estimated_size_uncompressed: 0,
            estimated_size_compressed: 0,
        }
    }

    fn uniform_events() -> EventsPerDayStats {
        EventsPerDayStats {
            min: 90.0,
            max: 110.0,
            avg: 100.0,
            median: 100.0,
            std_dev: 5.0,
        }
    }

    fn bursty_events() -> EventsPerDayStats {
        EventsPerDayStats {
            min: 0.0,
            max: 5_000.0,
            avg: 100.0,
            median: 20.0,
            std_dev: 300.0,
        }
    }

    fn synthetic_result(
        duration_ms: u64,
        bucket_ms: u64,
        distribution: TemporalDistribution,
        events_per_day: EventsPerDayStats,
        max_zoom: u8,
        per_zoom: Vec<ZoomDensity>,
    ) -> AnalysisResult {
        AnalysisResult {
            source: "synthetic.parquet".to_string(),
            feature_count: 100_000,
            bounds: BoundingBox::new(-74.0, 40.0, -73.0, 41.0),
            spatial: crate::test_support::sample_spatial(
                0,
                max_zoom,
                SpatialDistribution::Regional,
            ),
            temporal: TemporalAnalysis {
                time_start: 0,
                time_end: duration_ms,
                duration_ms,
                duration_human: format!("{} days", duration_ms / DAY_MS),
                unique_timestamps: 10_000,
                distribution,
                recommended_bucket_ms: bucket_ms,
                recommended_bucket_human: format!("{} ms", bucket_ms),
                hourly_distribution: vec![0; 24],
                daily_distribution: vec![0; 7],
                monthly_distribution: vec![0; 12],
                events_per_day,
            },
            geometry: crate::test_support::point_geometry(100_000),
            density: DensityAnalysis {
                estimated_tile_count: per_zoom.iter().map(|z| z.tile_count).sum(),
                estimated_archive_size: 0,
                issues: Vec::new(),
                per_zoom,
            },
            measured: None,
        }
    }

    /// Replicates stt-build `parse_temporal_lod`'s per-entry duration parsing
    /// (a dev-dep on stt-build would be a cycle): `<num><unit>` with unit
    /// ms/s/m/h/d, optional `@zoom` suffix stripped.
    fn parse_tier_ms(entry: &str) -> u64 {
        let dur = entry.split('@').next().unwrap().trim();
        let split = dur
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(dur.len());
        let value: f64 = dur[..split].parse().expect("numeric duration value");
        let multiplier: u64 = match &dur[split..] {
            "ms" | "" => 1,
            "s" | "sec" => 1_000,
            "m" | "min" => 60_000,
            "h" | "hr" | "hour" => HOUR_MS,
            "d" | "day" => DAY_MS,
            unit => panic!("unit {unit:?} not accepted by stt-build parse_duration"),
        };
        (value * multiplier as f64) as u64
    }

    #[test]
    fn two_year_hourly_span_gets_strict_multiple_lod_tiers() {
        let base = HOUR_MS;
        let duration = 730 * DAY_MS; // 2 years => 17,520 hourly buckets
        let result = synthetic_result(
            duration,
            base,
            TemporalDistribution::Uniform,
            uniform_events(),
            10,
            vec![zoom_density(10, 100, 300)], // no skew: adaptive must not fire
        );

        let advice = advise(&result, &empty_data()).unwrap();
        assert_eq!(advice.len(), 1, "expected only --temporal-lod: {advice:?}");
        let lod = &advice[0];
        assert_eq!(lod.flag, "--temporal-lod");
        assert!(!lod.lossy, "temporal LOD is additive, never lossy");
        assert_eq!(lod.value.as_deref(), Some("1d,30d"));
        assert!(
            lod.why.contains("17520"),
            "why must cite the bucket count: {}",
            lod.why
        );

        // stt-build acceptance, replicated: every tier a STRICT multiple of
        // the base bucket, ascending.
        let tiers: Vec<u64> = lod
            .value
            .as_deref()
            .unwrap()
            .split(',')
            .map(parse_tier_ms)
            .collect();
        assert!(!tiers.is_empty() && tiers.len() <= 2);
        let mut prev = base;
        for &tier in &tiers {
            assert_eq!(tier % base, 0, "tier {tier} not a multiple of base {base}");
            assert!(tier > prev, "tiers must be strictly ascending above base");
            prev = tier;
        }
    }

    #[test]
    fn uniform_short_dataset_emits_nothing() {
        // 30 days at 2h = 360 buckets: under the LOD threshold, not bursty.
        // (This fixture used to be 30 days at 30m = 1,440 buckets, which was
        // "under the threshold" only because the threshold could never fire at
        // all — see LOD_BUCKET_THRESHOLD's calibration note.)
        let result = synthetic_result(
            30 * DAY_MS,
            2 * HOUR_MS,
            TemporalDistribution::Uniform,
            uniform_events(),
            10,
            vec![zoom_density(10, 100, 300)],
        );
        let buckets = result.temporal.duration_ms / result.temporal.recommended_bucket_ms;
        assert!(
            buckets < LOD_BUCKET_THRESHOLD,
            "fixture guard: {buckets} buckets must sit under the trigger"
        );
        let advice = advise(&result, &empty_data()).unwrap();
        assert!(advice.is_empty(), "expected no advice: {advice:?}");
    }

    // ------------------------------------------------------------------
    // F7: the trigger has to be REACHABLE from the bucket recommender
    // ------------------------------------------------------------------

    #[test]
    fn the_lod_threshold_sits_under_the_bucket_recommenders_own_ceiling() {
        // `recommend_bucket_size` returns the finest standard bucket whose
        // count is <= its target, and that target is at most
        // PLAYBACK_LOOP_FRAMES × 1.5 (Bursty) and exactly PLAYBACK_LOOP_FRAMES
        // for Uniform/Periodic. A threshold at or above the latter is therefore
        // unreachable for every non-Bursty dataset, and the old 5_000 was
        // unreachable for everything short of a 411-year span.
        let (trigger, uniform_target) = (LOD_BUCKET_THRESHOLD, PLAYBACK_LOOP_FRAMES);
        assert!(
            trigger < uniform_target,
            "a `--temporal-lod` trigger above the recommender's own bucket-count \
             target ({uniform_target}) can only ever fire on Bursty data: {trigger}"
        );
        assert_eq!(uniform_target, 1_350, "45 s x 30 fps");
    }

    #[test]
    fn a_real_long_timeline_dataset_clears_the_threshold_end_to_end() {
        use crate::loader::{AnalyzableFeature, GeometryType};
        use stt_core::types::BoundingBox;

        // Three years of evenly-spaced events, 12k features — the shape of the
        // repo's own osm-nyc-changesets (20 years, 380k features, 1068 weekly
        // buckets). Driven through the REAL temporal analyzer, not a synthetic
        // `TemporalAnalysis`, because the defect being pinned is precisely that
        // the analyzer's output could never reach the advisor's trigger.
        const N: u64 = 12_000;
        const SPAN_MS: u64 = 3 * 365 * DAY_MS;
        let step = SPAN_MS / N;
        let features: Vec<AnalyzableFeature> = (0..N)
            .map(|i| AnalyzableFeature {
                lon: -74.0 + (i % 97) as f64 * 0.01,
                lat: 40.0 + (i % 89) as f64 * 0.01,
                timestamp: 1_400_000_000_000 + i * step,
                geometry_type: GeometryType::Point,
                vertex_count: 1,
                estimated_size: 100,
                property_count: 1,
            })
            .collect();
        let data = LoadedData {
            features,
            bounds: BoundingBox::new(-74.0, 40.0, -73.0, 41.0),
            time_range: TimeRange::new(1_400_000_000_000, 1_400_000_000_000 + SPAN_MS),
            sample: Vec::new(),
        };

        let temporal = crate::analysis::temporal::analyze(&data).unwrap();
        let buckets = temporal.duration_ms / temporal.recommended_bucket_ms.max(1);
        assert!(
            buckets > LOD_BUCKET_THRESHOLD,
            "the recommender produced {buckets} buckets at {} ms, which does not \
             clear the {LOD_BUCKET_THRESHOLD}-bucket trigger — the lever is dead again",
            temporal.recommended_bucket_ms
        );

        let mut result = synthetic_result(
            temporal.duration_ms,
            temporal.recommended_bucket_ms,
            TemporalDistribution::Uniform,
            uniform_events(),
            10,
            vec![zoom_density(10, 100, 300)],
        );
        result.temporal = temporal;
        let advice = advise(&result, &data).unwrap();
        let lod = advice
            .iter()
            .find(|a| a.flag == "--temporal-lod")
            .unwrap_or_else(|| panic!("the tier advisor must fire here: {advice:?}"));
        assert!(!lod.lossy, "a coarse tier is additive, never lossy");
        assert!(lod.value.as_deref().is_some_and(|v| !v.is_empty()));
    }

    #[test]
    fn bursty_skewed_dataset_gets_adaptive_temporal_in_range() {
        // Short timeline (480 buckets, no LOD) but bursty and heavily skewed
        // at max zoom: median 500 vs max 25,000 = 50x.
        let result = synthetic_result(
            10 * DAY_MS,
            30 * 60_000,
            TemporalDistribution::Bursty,
            bursty_events(),
            10,
            vec![zoom_density(8, 50, 200), zoom_density(10, 500, 25_000)],
        );

        let advice = advise(&result, &empty_data()).unwrap();
        assert_eq!(
            advice.len(),
            1,
            "expected only --adaptive-temporal: {advice:?}"
        );
        let adaptive = &advice[0];
        assert_eq!(adaptive.flag, "--adaptive-temporal");
        assert!(!adaptive.lossy, "adaptive rebucketing drops no data");
        assert!(matches!(adaptive.confidence, AdviceConfidence::Medium));
        let n: u64 = adaptive.value.as_deref().unwrap().parse().unwrap();
        assert!((1_000..=50_000).contains(&n), "N {n} out of clamp range");
        // median 500 * 4 = 2,000 is already a nice number.
        assert_eq!(n, 2_000);
        assert!(
            adaptive.why.contains("25000") && adaptive.why.contains("500"),
            "why must cite the skew numbers: {}",
            adaptive.why
        );
    }

    #[test]
    fn borderline_skew_downgrades_to_low_confidence() {
        // Skew 12x: over the 10x threshold but under the 20x confident bar.
        let result = synthetic_result(
            10 * DAY_MS,
            30 * 60_000,
            TemporalDistribution::Bursty,
            bursty_events(),
            10,
            vec![zoom_density(10, 500, 6_000)],
        );
        let advice = advise(&result, &empty_data()).unwrap();
        assert_eq!(advice.len(), 1);
        assert_eq!(advice[0].flag, "--adaptive-temporal");
        assert!(matches!(advice[0].confidence, AdviceConfidence::Low));
    }

    #[test]
    fn bursty_long_timeline_keeps_both_and_notes_interaction() {
        let result = synthetic_result(
            730 * DAY_MS,
            HOUR_MS,
            TemporalDistribution::Bursty,
            bursty_events(),
            10,
            vec![zoom_density(10, 500, 25_000)],
        );
        let advice = advise(&result, &empty_data()).unwrap();
        assert_eq!(advice.len(), 2, "both levers apply: {advice:?}");
        for a in &advice {
            assert!(
                a.why.contains("consider one or the other first"),
                "{} why must note the interaction: {}",
                a.flag,
                a.why
            );
        }
    }

    #[test]
    fn minute_base_bucket_snaps_to_hour_day_tiers() {
        // Base 1m over ~35 days = ~50,400 buckets; tiers snap to 1h,1d.
        let result = synthetic_result(
            35 * DAY_MS,
            60_000,
            TemporalDistribution::Uniform,
            uniform_events(),
            10,
            vec![zoom_density(10, 100, 300)],
        );
        let advice = advise(&result, &empty_data()).unwrap();
        assert_eq!(advice.len(), 1);
        assert_eq!(advice[0].value.as_deref(), Some("1h,1d"));
    }

    #[test]
    fn round_nice_snaps_to_1_2_5_decades() {
        assert_eq!(round_nice(3_200.0), 5_000);
        assert_eq!(round_nice(2_000.0), 2_000);
        assert_eq!(round_nice(480.0), 500);
        assert_eq!(round_nice(1_400.0), 1_000);
        assert_eq!(round_nice(70_000.0), 50_000);
    }
}
