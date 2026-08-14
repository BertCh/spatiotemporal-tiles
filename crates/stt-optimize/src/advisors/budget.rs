//! Tile-budget / aggregation advisor (`--maximum-tile-features`,
//! `--summary-tier`, `--min-zoom-field`): flags deep-zoom oversized tiles from
//! the density model, suggests an additive summary tier for very large dense
//! point sets, and points hotspot-skewed data at per-feature LOD floors.
//!
//! Budget advice (anything that drops features) is always `lossy: true` —
//! opt-in only, per the no-thinning principle — and its `why` says so
//! explicitly. `--drop-densest-as-needed` is NEVER emitted as its own advice;
//! it is only mentioned inside the budget `why` as the escalation path.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;

use super::{Advice, AdviceConfidence};
use crate::analysis::density::ZoomDensity;
use crate::analysis::spatial::SpatialDistribution;
use crate::analysis::AnalysisResult;
use crate::loader::{LoadedData, PropValue, SampledFeature};

/// The 10,000-features-per-tile rule of thumb the density model's
/// `oversized_tiles` counter uses — also the suggested cap value.
const OVERSIZED_TILE_FEATURES: usize = 10_000;

/// Feature-count floor for the summary-tier recommendation: below this an
/// overview zoom can usually ship raw points.
const SUMMARY_TIER_MIN_FEATURES: usize = 1_000_000;

/// Average features/tile at the recommended min zoom above which the overview
/// tier counts as "heavy" for the summary-tier recommendation.
const SUMMARY_TIER_HEAVY_AVG: f64 = 5_000.0;

/// Share of features in the top 1% of overview cells above which the data is
/// "hotspot-skewed" enough to warrant a summary tier even when the AVERAGE
/// overview tile is light — a few dense cells the raw tier can't serve. This is
/// the Sedona "cluster the partitioner can't balance" signal applied to tiles;
/// the volume+average rule alone misses it because empty cells drag the average
/// down.
const SUMMARY_TIER_SKEW_SHARE: f64 = 0.60;

/// Lower feature-count floor for the skew-triggered summary tier: concentration,
/// not sheer volume, is the problem there, so it fires well below the
/// dense-uniform [`SUMMARY_TIER_MIN_FEATURES`] floor.
const SUMMARY_TIER_SKEW_MIN_FEATURES: usize = 100_000;

/// Share of all features in the single top hotspot above which a per-feature
/// LOD floor is worth suggesting (the existing density-issue rule).
const HOTSPOT_SHARE: f64 = 0.5;

/// Minimum sampled rows carrying a property before its cardinality is trusted
/// as evidence for a LOD-floor candidate.
const MIN_PROPERTY_SAMPLES: usize = 20;

/// Maximum distinct values for a "low-cardinality" LOD-floor candidate column
/// (a class/rank column, not an id or a measurement).
const MAX_LOD_CANDIDATE_CARDINALITY: usize = 16;

/// Integer magnitude ceiling for a "small-int" rank column; beyond this the
/// values look like ids or measurements, not class codes.
const SMALL_INT_MAX_ABS: f64 = 32.0;

/// Advise on tile budgets and aggregation levers from the density model: an
/// opt-in `--maximum-tile-features` cap when deep zooms have oversized tiles,
/// an additive `--summary-tier` for very large dense point sets, and a
/// `--min-zoom-field` LOD floor when one hotspot dominates and the sample
/// carries a plausible rank column.
pub fn advise(result: &AnalysisResult, data: &LoadedData) -> Result<Vec<Advice>> {
    let mut advice = Vec::new();
    advice.extend(oversized_tile_advice(result));
    advice.extend(summary_tier_advice(result));
    advice.extend(hotspot_lod_advice(result, data));
    Ok(advice)
}

/// Opt-in `--maximum-tile-features` cap when tiles near the recommended max
/// zoom blow past the 10k-features rule. Lossy by definition — the `why`
/// leads with the non-destructive alternatives.
fn oversized_tile_advice(result: &AnalysisResult) -> Option<Advice> {
    let deep_floor = result.spatial.recommended_max_zoom.saturating_sub(1);
    let deep_oversized: Vec<&ZoomDensity> = result
        .density
        .per_zoom
        .iter()
        .filter(|z| z.zoom >= deep_floor && z.oversized_tiles > 0)
        .collect();
    let worst = deep_oversized
        .iter()
        .copied()
        .max_by_key(|z| z.max_features_per_tile)?;
    let oversized: usize = deep_oversized.iter().map(|z| z.oversized_tiles).sum();
    let worst_features = worst.max_features_per_tile;

    // Worst-tile size: measured bytes/feature (real encoder + zstd on the
    // loader sample) when available, else the zoom row's formula estimate
    // scaled down to one feature.
    let (bytes_per_feature, basis) = match &result.measured {
        Some(m) => (m.bytes_per_feature, "measured sample encode"),
        None => {
            let bucketed = (worst.avg_features_per_tile * worst.tile_count as f64).max(1.0);
            (
                worst.estimated_size_compressed as f64 / bucketed,
                "formula estimate",
            )
        }
    };
    let worst_bytes = (worst_features as f64 * bytes_per_feature).round() as usize;

    Some(Advice {
        flag: "--maximum-tile-features".to_string(),
        value: Some(OVERSIZED_TILE_FEATURES.to_string()),
        why: format!(
            "{} tiles at z{}+ exceed {} features (worst {} features ≈ {} compressed, {}); \
             --maximum-tile-features DROPS each oversized tile's lowest-importance features \
             to fit — data loss, OFF by default, strictly opt-in. Prefer raising --min-zoom \
             or --summary-tier first; --drop-densest-as-needed is the further escalation if \
             capped tiles still read poorly",
            oversized,
            deep_floor,
            OVERSIZED_TILE_FEATURES,
            worst_features,
            fmt_bytes(worst_bytes),
            basis,
        ),
        projected: Some(format!(
            "caps worst tile {} → ≤{} features",
            worst_features, OVERSIZED_TILE_FEATURES
        )),
        lossy: true,
        suggestion_only: false,
        confidence: AdviceConfidence::Medium,
    })
}

/// Additive `--summary-tier` for point datasets whose overview zoom can't ship
/// raw points — either because it is DENSE-UNIFORM (many features, heavy average
/// tile) or HOTSPOT-SKEWED (a few overview cells hold most of the data, so the
/// average is light but the peak is unservable; the density model's
/// `top1pct_feature_share` is the skew signal). Picks hexagonal H3 for tightly
/// concentrated (Localized) data — equal-area cells aggregate a dense blob more
/// evenly and avoid Quadbin's axis-aligned blockiness — and the tile-aligned
/// Quadbin grid otherwise. Not lossy: summary tiles are carried IN ADDITION to
/// the raw tier and readers dispatch via `metadata.summaryTier`.
fn summary_tier_advice(result: &AnalysisResult) -> Option<Advice> {
    if !matches!(
        result.geometry.dominant_type.as_str(),
        "Point" | "MultiPoint"
    ) {
        return None;
    }
    let overview_zoom = result.spatial.recommended_min_zoom;
    let overview = result
        .density
        .per_zoom
        .iter()
        .find(|z| z.zoom == overview_zoom)?;

    // (1) Dense-uniform: sheer volume with a heavy average overview tile.
    let dense_uniform = result.feature_count > SUMMARY_TIER_MIN_FEATURES
        && overview.avg_features_per_tile > SUMMARY_TIER_HEAVY_AVG;
    // (2) Hotspot-skewed: the densest 1% of overview cells dominate and their
    //     peak is oversized, even if the average tile is light — the volume/avg
    //     rule misses this (empty cells drag the average down).
    let hotspot_skewed = result.feature_count > SUMMARY_TIER_SKEW_MIN_FEATURES
        && overview.top1pct_feature_share >= SUMMARY_TIER_SKEW_SHARE
        && overview.max_features_per_tile > OVERSIZED_TILE_FEATURES;
    if !dense_uniform && !hotspot_skewed {
        return None;
    }

    let scheme = if matches!(result.spatial.distribution, SpatialDistribution::Localized) {
        "h3"
    } else {
        "quadbin"
    };

    // Skew-only triggers explain themselves with the concentration numbers; the
    // dense-uniform message keeps the original average-based phrasing.
    let (why, projected) = if hotspot_skewed && !dense_uniform {
        (
            format!(
                "{} point features are hotspot-skewed at overview zoom z{}: the densest 1% \
                 of cells hold {:.0}% of all features (peak {} features/tile) while the \
                 average tile is light — a summary tier serves aggregate {} cells for the \
                 hot cells the raw tier can't. Additive, not lossy; readers dispatch via \
                 metadata.summaryTier",
                result.feature_count,
                overview_zoom,
                overview.top1pct_feature_share * 100.0,
                overview.max_features_per_tile,
                scheme,
            ),
            format!(
                "z{} hotspot cells (peak {} pts) ship as {} aggregates instead of raw points",
                overview_zoom, overview.max_features_per_tile, scheme,
            ),
        )
    } else {
        (
            format!(
                "{} point features average {:.0} features/tile at overview zoom z{} — \
                 server-aggregated low-zoom cells replace shipping every point at overview \
                 zooms. Additive, not lossy: raw tiles are unchanged and readers dispatch \
                 via metadata.summaryTier",
                result.feature_count, overview.avg_features_per_tile, overview_zoom,
            ),
            format!(
                "z{} tiles carry aggregate cells instead of ~{:.0} raw points each",
                overview_zoom, overview.avg_features_per_tile
            ),
        )
    };

    Some(Advice {
        flag: "--summary-tier".to_string(),
        value: Some(scheme.to_string()),
        why,
        projected: Some(projected),
        lossy: false,
        suggestion_only: false,
        confidence: AdviceConfidence::Medium,
    })
}

/// `--min-zoom-field` LOD floor when one hotspot holds most of the data AND
/// the sample carries a plausible class/rank column to key the floor on.
/// Changes zoom placement, not data retention — but a feature is invisible at
/// zooms below its floor, so the `why` says so. Always suggestion-only: it is
/// a semantic lever (same policy as stt-build `--auto`), and on a categorical
/// candidate the flag as emitted is a silent no-op until the user bakes a
/// numeric rank column.
fn hotspot_lod_advice(result: &AnalysisResult, data: &LoadedData) -> Option<Advice> {
    let top = result.spatial.hotspots.first()?;
    let total = result.feature_count.max(1);
    let share = top.feature_count as f64 / total as f64;
    if share <= HOTSPOT_SHARE {
        return None;
    }
    let candidate = lod_candidate_column(&data.sample)?;

    let desc = if candidate.numeric {
        format!("small-int, {} distinct values", candidate.cardinality)
    } else {
        format!(
            "categorical, {} distinct values; --min-zoom-field takes a NUMERIC property, \
             so bake its ranks into a numeric column first",
            candidate.cardinality
        )
    };
    let why = format!(
        "{:.0}% of features ({} of {}) concentrate in {}; a per-feature zoom floor keyed \
         on `{}` ({}) holds minor features back to deeper zooms so hotspot overview tiles \
         stay light. Zoom placement only — no data is dropped, but a feature is invisible \
         at zooms below its floor",
        share * 100.0,
        top.feature_count,
        total,
        top.name.as_deref().unwrap_or("one hotspot"),
        candidate.name,
        desc,
    );

    Some(Advice {
        flag: "--min-zoom-field".to_string(),
        value: Some(candidate.name),
        why,
        projected: None,
        lossy: false,
        suggestion_only: true,
        confidence: AdviceConfidence::Low,
    })
}

/// A sampled property column plausibly usable as a per-feature LOD floor.
struct LodCandidate {
    /// Column name (the suggested `--min-zoom-field` value).
    name: String,
    /// Distinct values seen in the sample.
    cardinality: usize,
    /// All-numeric small-int values (directly usable — the flag takes a
    /// numeric property) vs an all-text categorical column (needs a numeric
    /// mapping first).
    numeric: bool,
}

/// Scan the loader sample for a low-cardinality class/rank column: either a
/// small-int numeric column or a low-cardinality string column, seen on at
/// least [`MIN_PROPERTY_SAMPLES`] sampled rows with 2..=16 distinct values.
/// Preference is deterministic: numeric first, then lowest cardinality, then
/// name.
fn lod_candidate_column(sample: &[SampledFeature]) -> Option<LodCandidate> {
    #[derive(Default)]
    struct ColStats {
        numeric_rows: usize,
        text_rows: usize,
        /// A numeric value was fractional, non-finite, or too large to be a
        /// class code — disqualifies the numeric-candidate path.
        big_or_fractional: bool,
        numeric_distinct: BTreeSet<u64>,
        text_distinct: BTreeSet<String>,
    }

    let mut cols: BTreeMap<&str, ColStats> = BTreeMap::new();
    for feature in sample {
        for (name, value) in &feature.properties {
            let stats = cols.entry(name.as_str()).or_default();
            match value {
                PropValue::Number(x) => {
                    stats.numeric_rows += 1;
                    if !(x.is_finite() && x.fract() == 0.0 && x.abs() <= SMALL_INT_MAX_ABS) {
                        stats.big_or_fractional = true;
                    }
                    stats.numeric_distinct.insert(x.to_bits());
                }
                PropValue::Text(s) => {
                    stats.text_rows += 1;
                    if !stats.text_distinct.contains(s) {
                        stats.text_distinct.insert(s.clone());
                    }
                }
            }
        }
    }

    let mut candidates: Vec<LodCandidate> = cols
        .into_iter()
        .filter_map(|(name, stats)| {
            let card_range = 2..=MAX_LOD_CANDIDATE_CARDINALITY;
            let numeric_ok = stats.text_rows == 0
                && stats.numeric_rows >= MIN_PROPERTY_SAMPLES
                && !stats.big_or_fractional
                && card_range.contains(&stats.numeric_distinct.len());
            let text_ok = stats.numeric_rows == 0
                && stats.text_rows >= MIN_PROPERTY_SAMPLES
                && card_range.contains(&stats.text_distinct.len());
            if numeric_ok {
                Some(LodCandidate {
                    name: name.to_string(),
                    cardinality: stats.numeric_distinct.len(),
                    numeric: true,
                })
            } else if text_ok {
                Some(LodCandidate {
                    name: name.to_string(),
                    cardinality: stats.text_distinct.len(),
                    numeric: false,
                })
            } else {
                None
            }
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.numeric
            .cmp(&a.numeric)
            .then(a.cardinality.cmp(&b.cardinality))
            .then_with(|| a.name.cmp(&b.name))
    });
    candidates.into_iter().next()
}

/// Human-readable byte count for advice strings.
fn fmt_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::density::{DensityAnalysis, ZoomDensity};
    use crate::analysis::geometry::{
        GeometryAnalysis, GeometryComplexity, PropertyStats, SizeStats, VertexStats,
    };
    use crate::analysis::spatial::{Hotspot, SpatialDistribution};
    use crate::measure::MeasuredEncoding;
    use std::collections::HashMap;
    use stt_core::types::{BoundingBox, TimeRange};

    fn geometry(dominant: &str) -> GeometryAnalysis {
        GeometryAnalysis {
            type_distribution: HashMap::from([(dominant.to_string(), 1)]),
            dominant_type: dominant.to_string(),
            vertex_stats: VertexStats {
                min: 1,
                max: 1,
                avg: 1.0,
                median: 1,
                p95: 1,
                p99: 1,
                total: 1,
            },
            size_stats: SizeStats {
                min: 150,
                max: 150,
                avg: 150.0,
                median: 150,
                p95: 150,
                p99: 150,
                total: 150,
            },
            property_stats: PropertyStats {
                min: 0,
                max: 2,
                avg: 1.0,
            },
            complexity: GeometryComplexity::Simple,
        }
    }

    /// One synthetic zoom row: sizes follow the 150-B formula estimate with
    /// the /3 compression fallback the density model uses without a
    /// measurement... except simplified to a flat 50 B/feature compressed.
    fn zoom_row(
        zoom: u8,
        tile_count: usize,
        avg: f64,
        max: usize,
        oversized: usize,
    ) -> ZoomDensity {
        let features = (avg * tile_count as f64) as usize;
        ZoomDensity {
            zoom,
            tile_count,
            avg_features_per_tile: avg,
            median_features_per_tile: avg as usize,
            max_features_per_tile: max,
            p95_features_per_tile: max,
            top1pct_feature_share: 0.0,
            oversized_tiles: oversized,
            undersized_tiles: 0,
            estimated_size_uncompressed: features * 150,
            estimated_size_compressed: features * 50,
        }
    }

    fn result_with(
        feature_count: usize,
        dominant: &str,
        min_zoom: u8,
        max_zoom: u8,
        per_zoom: Vec<ZoomDensity>,
    ) -> AnalysisResult {
        let estimated_tile_count = per_zoom.iter().map(|z| z.tile_count).sum();
        let estimated_archive_size = per_zoom.iter().map(|z| z.estimated_size_compressed).sum();
        AnalysisResult {
            source: "synthetic.parquet".to_string(),
            feature_count,
            bounds: BoundingBox::new(-74.0, 40.0, -73.0, 41.0),
            spatial: crate::test_support::sample_spatial(
                min_zoom,
                max_zoom,
                SpatialDistribution::Regional,
            ),
            temporal: crate::test_support::sample_temporal(),
            geometry: geometry(dominant),
            density: DensityAnalysis {
                per_zoom,
                estimated_tile_count,
                estimated_archive_size,
                issues: Vec::new(),
            },
            measured: None,
        }
    }

    fn with_hotspot(mut result: AnalysisResult, feature_count: usize) -> AnalysisResult {
        result.spatial.hotspots.push(Hotspot {
            lon: -73.97,
            lat: 40.77,
            radius: 0.1,
            feature_count,
            name: Some("Manhattan".to_string()),
        });
        result
    }

    fn empty_data() -> LoadedData {
        data_with_sample(Vec::new())
    }

    fn data_with_sample(sample: Vec<SampledFeature>) -> LoadedData {
        LoadedData {
            features: Vec::new(),
            bounds: BoundingBox::new(-74.0, 40.0, -73.0, 41.0),
            time_range: TimeRange::new(0, 86_400_000),
            sample,
        }
    }

    fn point_sample(
        n: usize,
        props: impl Fn(usize) -> Vec<(String, PropValue)>,
    ) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| SampledFeature {
                geometry: geo_types::Geometry::Point(geo_types::Point::new(-73.97, 40.77)),
                timestamp_ms: i as u64 * 1_000,
                properties: props(i),
            })
            .collect()
    }

    #[test]
    fn skewed_max_zoom_density_yields_optin_budget_advice() {
        let result = result_with(
            500_000,
            "Point",
            4,
            10,
            vec![
                zoom_row(4, 10, 4_000.0, 9_000, 0),
                zoom_row(10, 50, 4_000.0, 25_000, 3),
            ],
        );
        let advice = advise(&result, &empty_data()).unwrap();
        let budget = advice
            .iter()
            .find(|a| a.flag == "--maximum-tile-features")
            .expect("budget advice for oversized deep-zoom tiles");
        assert_eq!(budget.value.as_deref(), Some("10000"));
        assert!(budget.lossy, "budget advice must always be lossy");
        assert_eq!(budget.confidence, AdviceConfidence::Medium);

        let why = budget.why.to_lowercase();
        assert!(
            why.contains("opt-in"),
            "why must say opt-in: {}",
            budget.why
        );
        assert!(
            why.contains("off by default"),
            "why must say off by default: {}",
            budget.why
        );
        assert!(
            why.contains("drop"),
            "why must say it drops data: {}",
            budget.why
        );
        // Cites this dataset's numbers: oversized count and worst-tile load.
        assert!(budget.why.contains("3 tiles"));
        assert!(budget.why.contains("25000"));
        // Non-destructive alternatives paired in the same why, and the
        // drop-densest escalation is mentioned there — never its own entry.
        assert!(budget.why.contains("--min-zoom"));
        assert!(budget.why.contains("--summary-tier"));
        assert!(budget.why.contains("--drop-densest-as-needed"));
        assert!(advice.iter().all(|a| a.flag != "--drop-densest-as-needed"));
    }

    #[test]
    fn measured_bytes_per_feature_drives_worst_tile_estimate() {
        let mut result = result_with(
            500_000,
            "Point",
            4,
            10,
            vec![zoom_row(10, 50, 4_000.0, 20_000, 2)],
        );
        result.measured = Some(MeasuredEncoding {
            features: 5_000,
            geometry_kind: "point".to_string(),
            bytes_total: 500_000,
            bytes_per_feature: 100.0,
            zstd_ratio: 3.0,
            tiles: 1,
            per_column: Vec::new(),
        });
        let advice = advise(&result, &empty_data()).unwrap();
        let budget = advice
            .iter()
            .find(|a| a.flag == "--maximum-tile-features")
            .expect("budget advice");
        // 20,000 features × 100 B/feature (measured) = 2,000,000 B ≈ 1.91 MB.
        assert!(budget.why.contains("1.91 MB"), "why: {}", budget.why);
        assert!(budget.why.contains("measured"), "why: {}", budget.why);
    }

    #[test]
    fn oversized_only_at_shallow_zoom_is_not_budget_flagged() {
        // Oversized tiles at z2 with max zoom 12: --maximum-tile-features
        // targets deep zooms only (z >= max_zoom - 1); shallow-zoom weight is
        // the summary tier's / min-zoom's problem.
        let result = result_with(
            500_000,
            "Point",
            2,
            12,
            vec![
                zoom_row(2, 4, 4_900.0, 30_000, 2),
                zoom_row(12, 40_000, 12.0, 500, 0),
            ],
        );
        let advice = advise(&result, &empty_data()).unwrap();
        assert!(advice.iter().all(|a| a.flag != "--maximum-tile-features"));
    }

    #[test]
    fn two_million_heavy_overview_points_yield_summary_tier() {
        let result = result_with(
            2_000_000,
            "Point",
            3,
            12,
            vec![
                zoom_row(3, 40, 8_000.0, 9_500, 0),
                zoom_row(12, 30_000, 60.0, 800, 0),
            ],
        );
        let advice = advise(&result, &empty_data()).unwrap();
        let tier = advice
            .iter()
            .find(|a| a.flag == "--summary-tier")
            .expect("summary-tier advice for a large dense point set");
        assert_eq!(tier.value.as_deref(), Some("quadbin"));
        assert!(!tier.lossy, "summary tier is additive, not lossy");
        assert_eq!(tier.confidence, AdviceConfidence::Medium);
        // Cites this dataset's numbers.
        assert!(tier.why.contains("2000000"), "why: {}", tier.why);
        assert!(tier.why.contains("8000"), "why: {}", tier.why);
        assert!(tier.why.contains("z3"), "why: {}", tier.why);
        // No oversized deep-zoom tiles → no budget cap alongside it.
        assert!(advice.iter().all(|a| a.flag != "--maximum-tile-features"));
    }

    #[test]
    fn hotspot_skewed_points_yield_h3_summary_tier_below_the_volume_floor() {
        // 300k points — below the 1M dense-uniform floor — but the densest 1% of
        // overview cells hold 80% of them with an unservable 50k-feature peak,
        // and the average overview tile is light. The skew trigger must still
        // fire, and a Localized distribution selects hexagonal H3.
        let mut result = result_with(
            300_000,
            "Point",
            4,
            12,
            vec![
                zoom_row(4, 500, 600.0, 50_000, 0),
                zoom_row(12, 30_000, 10.0, 300, 0),
            ],
        );
        result.spatial.distribution = SpatialDistribution::Localized;
        // Concentration lives on the overview row (zoom_row defaults it to 0.0).
        result.density.per_zoom[0].top1pct_feature_share = 0.80;

        let advice = advise(&result, &empty_data()).unwrap();
        let tier = advice
            .iter()
            .find(|a| a.flag == "--summary-tier")
            .expect("skew-triggered summary tier below the volume floor");
        assert_eq!(tier.value.as_deref(), Some("h3"), "Localized ⇒ H3");
        assert!(!tier.lossy, "summary tier is additive");
        assert_eq!(tier.confidence, AdviceConfidence::Medium);
        assert!(
            tier.why.contains("80%"),
            "why must cite the concentration: {}",
            tier.why
        );
        assert!(
            tier.why.contains("50000"),
            "why must cite the peak tile: {}",
            tier.why
        );
    }

    #[test]
    fn light_average_without_skew_yields_no_summary_tier() {
        // Same modest total and light average, but NO concentration (top1pct
        // stays at the uniform default) — neither trigger fires.
        let result = result_with(
            300_000,
            "Point",
            4,
            12,
            vec![zoom_row(4, 500, 600.0, 5_000, 0)],
        );
        let advice = advise(&result, &empty_data()).unwrap();
        assert!(advice.iter().all(|a| a.flag != "--summary-tier"));
    }

    #[test]
    fn regional_dense_uniform_still_picks_quadbin() {
        // The dense-uniform path on a Regional distribution keeps Quadbin (H3 is
        // reserved for tightly Localized clusters).
        let result = result_with(
            2_000_000,
            "Point",
            3,
            12,
            vec![zoom_row(3, 40, 8_000.0, 9_500, 0)],
        );
        let tier = advise(&result, &empty_data())
            .unwrap()
            .into_iter()
            .find(|a| a.flag == "--summary-tier")
            .expect("dense-uniform summary tier");
        assert_eq!(tier.value.as_deref(), Some("quadbin"));
    }

    #[test]
    fn line_dominant_data_never_gets_summary_tier() {
        let result = result_with(
            2_000_000,
            "LineString",
            3,
            12,
            vec![zoom_row(3, 40, 8_000.0, 9_500, 0)],
        );
        let advice = advise(&result, &empty_data()).unwrap();
        assert!(advice.iter().all(|a| a.flag != "--summary-tier"));
    }

    #[test]
    fn small_uniform_dataset_yields_no_advice() {
        let result = result_with(
            20_000,
            "Point",
            4,
            10,
            vec![
                zoom_row(4, 100, 200.0, 900, 0),
                zoom_row(10, 2_000, 10.0, 80, 0),
            ],
        );
        assert!(advise(&result, &empty_data()).unwrap().is_empty());
    }

    #[test]
    fn hotspot_with_small_int_rank_column_yields_min_zoom_field() {
        let result = with_hotspot(
            result_with(
                100_000,
                "Point",
                4,
                10,
                vec![zoom_row(4, 100, 1_000.0, 5_000, 0)],
            ),
            60_000,
        );
        let data = data_with_sample(point_sample(40, |i| {
            vec![
                ("road_class".to_string(), PropValue::Number((i % 4) as f64)),
                // High-magnitude ints look like ids, not ranks — must lose.
                ("osm_id".to_string(), PropValue::Number(1.0e9 + i as f64)),
            ]
        }));
        let advice = advise(&result, &data).unwrap();
        let lod = advice
            .iter()
            .find(|a| a.flag == "--min-zoom-field")
            .expect("LOD-floor advice for hotspot concentration");
        assert_eq!(lod.value.as_deref(), Some("road_class"));
        assert!(!lod.lossy);
        // Semantic lever: surfaced, never auto-applied (same policy as
        // stt-build --auto).
        assert!(lod.suggestion_only);
        assert_eq!(lod.confidence, AdviceConfidence::Low);
        // Cites the hotspot share and calls out the visibility tradeoff.
        assert!(lod.why.contains("60%"), "why: {}", lod.why);
        assert!(lod.why.contains("60000"), "why: {}", lod.why);
        assert!(
            lod.why.to_lowercase().contains("invisible"),
            "why: {}",
            lod.why
        );
    }

    #[test]
    fn hotspot_categorical_string_candidate_notes_numeric_mapping() {
        let result = with_hotspot(
            result_with(
                100_000,
                "Point",
                4,
                10,
                vec![zoom_row(4, 100, 1_000.0, 5_000, 0)],
            ),
            60_000,
        );
        let data = data_with_sample(point_sample(40, |i| {
            vec![(
                "category".to_string(),
                PropValue::Text(format!("class-{}", i % 3)),
            )]
        }));
        let advice = advise(&result, &data).unwrap();
        let lod = advice
            .iter()
            .find(|a| a.flag == "--min-zoom-field")
            .expect("LOD-floor advice");
        assert_eq!(lod.value.as_deref(), Some("category"));
        // The flag takes a numeric property; a string column must say so —
        // and the advice must be suggestion-only, because as emitted the flag
        // is a silent no-op until the user bakes a numeric rank column.
        assert!(
            lod.why.to_lowercase().contains("numeric"),
            "why: {}",
            lod.why
        );
        assert!(
            lod.suggestion_only,
            "categorical --min-zoom-field must not auto-apply"
        );
    }

    #[test]
    fn hotspot_without_candidate_column_yields_nothing() {
        let result = with_hotspot(
            result_with(
                100_000,
                "Point",
                4,
                10,
                vec![zoom_row(4, 100, 1_000.0, 5_000, 0)],
            ),
            60_000,
        );
        // High-cardinality continuous values only — no plausible rank column.
        let data = data_with_sample(point_sample(40, |i| {
            vec![("magnitude".to_string(), PropValue::Number(i as f64 * 1.37))]
        }));
        assert!(advise(&result, &data).unwrap().is_empty());
    }

    #[test]
    fn minor_hotspot_yields_nothing_even_with_candidate() {
        // 40% in the top hotspot is below the >50% concentration rule.
        let result = with_hotspot(
            result_with(
                100_000,
                "Point",
                4,
                10,
                vec![zoom_row(4, 100, 1_000.0, 5_000, 0)],
            ),
            40_000,
        );
        let data = data_with_sample(point_sample(40, |i| {
            vec![("road_class".to_string(), PropValue::Number((i % 4) as f64))]
        }));
        assert!(advise(&result, &data).unwrap().is_empty());
    }
}
