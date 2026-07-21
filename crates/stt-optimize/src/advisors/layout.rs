//! Wire/layout advisor (`--publish` (zstd 19), `--blob-ordering`,
//! `--pack-size`): compression measured on the sample, ordering simulated over
//! the synthesized native tiles, pack sizing from the estimated archive size.
//!
//! The zstd trial and the blob-ordering pick are both evidence-based: `--publish`
//! re-encodes the real sample, and `--blob-ordering` runs the SAME range-read
//! simulator the post-build `order-audit` uses ([`stt_core::ordering_sim`]) over
//! tiles synthesized from the loaded features — so its pre-build recommendation
//! agrees with what `order-audit measured` finds on the built archive. Only pack
//! sizing stays a formula estimate at Low confidence. Everything here is
//! byte-level and reversible — `lossy: false` throughout.

use anyhow::Result;

use stt_core::curve::{self, BlobOrdering};
use stt_core::ordering_sim::{self, SimOptions, TileSample};
use stt_core::tile::TileId;

use super::{Advice, AdviceConfidence};
use crate::analysis::spatial::SpatialDistribution;
use crate::analysis::AnalysisResult;
use crate::loader::LoadedData;
use crate::measure::{measure_sample, MeasureSettings, MeasuredEncoding};
use crate::order_audit::playback_caveat_for;
use stt_core::projection;

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
    advice.extend(blob_ordering_advice(result, data));
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
                suggestion_only: false,
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
        suggestion_only: false,
        confidence: AdviceConfidence::Low,
    })
}

/// Fewest synthesized native tiles worth trusting the range-read simulation on.
/// Below this the occupied grid is too small to discriminate orderings, so we
/// fall back to the access-shape heuristic.
const MIN_TILES_TO_SIMULATE: usize = 8;

/// `--blob-ordering`: synthesize the native-tier tiles the build would produce
/// (one per occupied `(x, y, time-bucket)` cell at the recommended max zoom) and
/// run the SAME range-read simulator as `order-audit` to pick the measured-best
/// ordering. Emitted only when that differs from what `--blob-ordering auto`
/// would pick (else `auto` already gets it right), or when the cost-best pick is
/// `spatial` on a multi-bucket dataset (the playback-buffering tension worth
/// surfacing). Falls back to the access-shape heuristic when too few tiles are
/// synthesized to trust the simulation.
fn blob_ordering_advice(result: &AnalysisResult, data: &LoadedData) -> Option<Advice> {
    let samples = synthesize_tiles(result, data);
    if samples.len() < MIN_TILES_TO_SIMULATE {
        return blob_ordering_advice_heuristic(result);
    }

    let opts = SimOptions::default();
    let recommended = ordering_sim::measured_ordering(&samples, opts);
    let auto = auto_choice(&samples);

    let distinct_buckets = {
        let mut tbs: Vec<i64> = samples.iter().map(|s| s.tb).collect();
        tbs.sort_unstable();
        tbs.dedup();
        tbs.len()
    };
    let value = recommended.as_str();
    let caveat = playback_caveat_for(value, distinct_buckets);

    // Nothing to say when the simulated best is exactly what `auto` already
    // picks AND there is no playback tension to flag.
    if recommended == auto && caveat.is_none() {
        return None;
    }

    let mut why = format!(
        "range-read simulation over {} synthesized native tiles ({} time bucket{}) \
         ranks `{}` cheapest; `--blob-ordering auto` would pick `{}`",
        samples.len(),
        distinct_buckets,
        if distinct_buckets == 1 { "" } else { "s" },
        value,
        auto.as_str(),
    );
    // A spatial win on a multi-bucket dataset trades playback buffering for
    // range-read cost; surface that tension (drops confidence to Medium) and
    // make the advice suggestion-only — auto-applying `spatial` here silently
    // stalls time-playback (empty buffered ranges), so the tradeoff is the
    // user's call, never the auto path's.
    let confidence = if let Some(c) = &caveat {
        why.push_str(". ");
        why.push_str(c);
        AdviceConfidence::Medium
    } else {
        AdviceConfidence::High
    };

    Some(Advice {
        flag: "--blob-ordering".to_string(),
        value: Some(value.to_string()),
        why,
        projected: Some(format!("measured-best over {} tiles (simulated)", samples.len())),
        lossy: false,
        suggestion_only: caveat.is_some(),
        confidence,
    })
}

/// Synthesize one [`TileSample`] per occupied `(x, y, time-bucket)` cell at the
/// recommended max zoom — the same tiling the density model uses, projected into
/// the ordering simulator's view. `len` is summed per-feature bytes (measured
/// bytes/feature when available, else the size-formula estimate); `hilbert` is
/// the tile's 2D spatial Hilbert index (reusing [`TileId::hilbert_index`]) so the
/// simulated layout matches the writer's. Non-point features are assigned by
/// centroid (same approximation the density model documents).
fn synthesize_tiles(result: &AnalysisResult, data: &LoadedData) -> Vec<TileSample> {
    let zoom = result.spatial.recommended_max_zoom;
    let bucket_ms = result.temporal.recommended_bucket_ms.max(1);
    let per_feature_bytes = |count: u64, summed_est: u64| -> u64 {
        match &result.measured {
            Some(m) => (count as f64 * m.bytes_per_feature).round() as u64,
            None => summed_est,
        }
    };

    // (feature count, summed estimated bytes) per (x, y, tb).
    use std::collections::HashMap;
    let mut cells: HashMap<(u32, u32, i64), (u64, u64)> = HashMap::new();
    for feature in &data.features {
        if let Ok((x, y)) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
            let tb = (feature.timestamp / bucket_ms) as i64;
            let entry = cells.entry((x, y, tb)).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += feature.estimated_size as u64;
        }
    }

    cells
        .into_iter()
        .map(|((x, y, tb), (count, summed_est))| TileSample {
            z: zoom,
            x,
            y,
            hilbert: TileId::new(zoom, x, y, 0).hilbert_index(),
            time_start: tb.saturating_mul(bucket_ms as i64),
            tb,
            len: per_feature_bytes(count, summed_est).max(1),
        })
        .collect()
}

/// What `--blob-ordering auto` (the cardinality heuristic) would pick over these
/// tiles — the same `bits_for`/`BlobOrdering::choose` logic the writer's `auto`
/// mode and `order-audit` use, so the advisor's "auto would pick X" is exact.
fn auto_choice(samples: &[TileSample]) -> BlobOrdering {
    if samples.is_empty() {
        return BlobOrdering::SpatialMajor;
    }
    let (mut x_min, mut x_max, mut y_min, mut y_max) = (u32::MAX, 0u32, u32::MAX, 0u32);
    let (mut tb_min, mut tb_max) = (i64::MAX, i64::MIN);
    for s in samples {
        x_min = x_min.min(s.x);
        x_max = x_max.max(s.x);
        y_min = y_min.min(s.y);
        y_max = y_max.max(s.y);
        tb_min = tb_min.min(s.tb);
        tb_max = tb_max.max(s.tb);
    }
    let space_bits = curve::bits_for((x_max - x_min).max(y_max - y_min) as u64 + 1);
    let time_bits = curve::bits_for((tb_max - tb_min).max(0) as u64 + 1);
    BlobOrdering::choose(space_bits, time_bits)
}

/// The pre-simulation fallback: the old access-shape heuristic, used only when
/// too few native tiles are synthesized to trust the range-read simulation
/// (`Localized` + short window -> `spatial`, `Global`/`Regional` + long window ->
/// `time-major`; anything else keeps `auto`). Low confidence — it is a guess.
fn blob_ordering_advice_heuristic(result: &AnalysisResult) -> Option<Advice> {
    let duration_ms = result.temporal.duration_ms;
    let duration = &result.temporal.duration_human;
    let (value, why) = match result.spatial.distribution {
        SpatialDistribution::Localized if duration_ms < SHORT_DURATION_MS => (
            "spatial",
            format!(
                "{} features are spatially Localized over only {}: viewport-local \
                 reads dominate, so spatial-major blob order keeps a viewport's \
                 tiles in fewer packs (access-shape heuristic — too few tiles to simulate)",
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
                 same packs (access-shape heuristic — too few tiles to simulate)",
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
        suggestion_only: false,
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
        suggestion_only: false,
        confidence: AdviceConfidence::Low,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{AnalyzableFeature, GeometryType, PropValue, SampledFeature};
    use geo_types::{Geometry, Point};
    use stt_core::types::{BoundingBox, TimeRange};

    const DAY_MS: u64 = 86_400_000;
    const HOUR_MS: u64 = 3_600_000;

    /// A point feature at (lon, lat, ts) with the size-formula estimate the
    /// loader would assign a bare point (no vertices beyond one, no properties).
    fn analyzable(lon: f64, lat: f64, ts: u64) -> AnalyzableFeature {
        AnalyzableFeature {
            lon,
            lat,
            timestamp: ts,
            geometry_type: GeometryType::Point,
            vertex_count: 1,
            estimated_size: 116, // 100 + 1*16 + 0*20
            property_count: 0,
        }
    }

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

    fn synthetic_data_with_features(features: Vec<AnalyzableFeature>) -> LoadedData {
        LoadedData {
            features,
            bounds: BoundingBox::new(-74.0, 45.0, -73.0, 46.0),
            time_range: TimeRange::new(0, 24 * HOUR_MS),
            sample: Vec::new(),
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
    fn too_few_tiles_falls_back_to_localized_short_heuristic() {
        // No loaded features → nothing to simulate → the access-shape fallback.
        let data = synthetic_data(Vec::new());
        let result = synthetic_result(SpatialDistribution::Localized, 3 * DAY_MS, 100 << 20, None);

        let advice = advise(&result, &data).unwrap();
        let ordering = find(&advice, "--blob-ordering").expect("blob-ordering advice");
        assert_eq!(ordering.value.as_deref(), Some("spatial"));
        assert!(!ordering.lossy);
        assert!(matches!(ordering.confidence, AdviceConfidence::Low));
        assert!(ordering.why.contains("too few tiles to simulate"));
    }

    #[test]
    fn too_few_tiles_falls_back_to_global_yearlong_heuristic() {
        let data = synthetic_data(Vec::new());
        let result = synthetic_result(SpatialDistribution::Global, 365 * DAY_MS, 100 << 20, None);

        let advice = advise(&result, &data).unwrap();
        let ordering = find(&advice, "--blob-ordering").expect("blob-ordering advice");
        assert_eq!(ordering.value.as_deref(), Some("time-major"));
        assert!(matches!(ordering.confidence, AdviceConfidence::Low));
        assert!(ordering.why.contains("too few tiles to simulate"));
    }

    #[test]
    fn ambiguous_access_shape_with_no_features_keeps_auto_ordering() {
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
    fn simulated_deep_time_recommends_spatial_with_playback_caveat() {
        // Two spatial cells (distinct tiles at zoom 10) sampled across 24 hourly
        // buckets: the range-read simulator prefers `spatial` (each cell's whole
        // timeline byte-contiguous), which `auto` would NOT pick over a
        // time-dominant extent — and because it spans many buckets, the playback
        // caveat must fire and drop confidence to Medium.
        let mut features = Vec::new();
        for b in 0..24u64 {
            let ts = b * HOUR_MS;
            features.push(analyzable(-73.0, 45.0, ts));
            features.push(analyzable(-72.0, 45.0, ts));
        }
        let data = synthetic_data_with_features(features);
        // Localized/short so the OLD heuristic would have said "spatial" too, but
        // this path must reach the SIMULATION (48 tiles ≥ the floor), not the fallback.
        let result = synthetic_result(SpatialDistribution::Regional, 24 * HOUR_MS, 100 << 20, None);

        let advice = advise(&result, &data).unwrap();
        let ordering = find(&advice, "--blob-ordering").expect("blob-ordering advice");
        assert_eq!(ordering.value.as_deref(), Some("spatial"));
        assert!(!ordering.lossy);
        // Simulated, not the heuristic fallback.
        assert!(ordering.why.contains("range-read simulation"), "{}", ordering.why);
        assert!(!ordering.why.contains("too few tiles"));
        // Playback tension surfaced → Medium confidence + the time-major hint,
        // AND the advice is suggestion-only: auto-applying `spatial` on a
        // multi-bucket dataset silently stalls time-playback, so it must stay
        // out of `to_command` and the MCP auto-args.
        assert!(matches!(ordering.confidence, AdviceConfidence::Medium));
        assert!(ordering.suggestion_only, "playback-caveated spatial must not auto-apply");
        assert!(ordering.why.contains("time-major"), "{}", ordering.why);
        assert!(ordering.projected.as_deref().unwrap().contains("simulated"));
    }

    #[test]
    fn synthesize_tiles_buckets_by_cell_and_time() {
        // 2 cells × 3 buckets = 6 occupied tiles; each carries the summed bytes.
        let mut features = Vec::new();
        for b in 0..3u64 {
            features.push(analyzable(-73.0, 45.0, b * HOUR_MS));
            features.push(analyzable(-72.0, 45.0, b * HOUR_MS));
        }
        let data = synthetic_data_with_features(features);
        let result = synthetic_result(SpatialDistribution::Regional, 3 * HOUR_MS, 100 << 20, None);
        let tiles = synthesize_tiles(&result, &data);
        assert_eq!(tiles.len(), 6);
        assert!(tiles.iter().all(|t| t.z == 10 && t.len >= 116));
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
