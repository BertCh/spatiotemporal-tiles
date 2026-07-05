//! Per-property statistics profiler: numeric percentiles (p50..p99) and
//! categorical cardinality, feeding manifest `style_hints` (suggested color
//! domains, playback duration, layer hint) so a fresh build renders sensibly
//! without hand-tuning.
//!
//! The input seam ([`PropertyValues`]) is caller-agnostic: `stt-build` feeds
//! it from the loaded feature set, and any future caller (a packed-archive
//! re-profiler, a DB adaptor) can feed pre-collected values the same way.
//! Hints are DEFAULTS — the renderer/user can always override them.

use stt_core::metadata::{PropertyStyleHint, StyleHints};

/// Version of the emitted `style_hints` block.
pub const STYLE_HINTS_VERSION: u32 = 1;

/// The layer-hint vocabulary the block may carry. Anything outside it is
/// dropped rather than emitted (the manifest schema pins this enum).
const LAYER_HINTS: [&str; 4] = ["points", "paths", "trips", "polygons"];

/// Collected values for one property, as gathered by the caller.
///
/// * `Numeric` — samples widened to `f64` (any numeric source type).
///   Non-finite entries are filtered by the profiler; callers need not
///   pre-clean.
/// * `Categorical` — string(-like) properties only need their distinct-value
///   count; the values themselves never cross this seam.
#[derive(Debug, Clone)]
pub enum PropertyValues {
    /// Numeric samples (any numeric source type widened to f64).
    Numeric(Vec<f64>),
    /// Categorical (string) property: distinct-value count.
    Categorical { distinct: usize },
}

/// Profile a set of per-property value collections into a [`StyleHints`]
/// block.
///
/// * Numeric properties yield min/p50/p90/p95/p97/p99/max plus a
///   `suggested_domain` of `[min, p97]` with each endpoint rounded OUTWARD to
///   2 significant figures (the "domain clamps at ~p97" convention). A
///   property whose values are all non-finite (or empty) is omitted.
/// * Categorical properties yield ONLY `name` + `cardinality`.
/// * `suggested_playback_seconds = clamp(round(sqrt(bucket_count)), 20, 90)`
///   where `bucket_count = time_range_ms / temporal_bucket_ms`; absent when
///   `temporal_bucket_ms` is 0.
/// * `layer_kind_hint` passes through only when it is one of the pinned
///   `"points" | "paths" | "trips" | "polygons"` vocabulary.
pub fn profile_properties(
    props: &[(String, PropertyValues)],
    time_range_ms: u64,
    temporal_bucket_ms: u64,
    layer_kind_hint: Option<&str>,
) -> StyleHints {
    let properties = props
        .iter()
        .filter_map(|(name, values)| match values {
            PropertyValues::Numeric(raw) => numeric_hint(name, raw),
            PropertyValues::Categorical { distinct } => Some(PropertyStyleHint {
                name: name.clone(),
                cardinality: Some((*distinct).min(u32::MAX as usize) as u32),
                ..Default::default()
            }),
        })
        .collect();

    StyleHints {
        version: STYLE_HINTS_VERSION,
        properties,
        suggested_playback_seconds: suggested_playback_seconds(
            time_range_ms,
            temporal_bucket_ms,
        ),
        layer_hint: layer_kind_hint
            .filter(|h| LAYER_HINTS.contains(h))
            .map(str::to_string),
    }
}

/// Numeric percentile profile for one property. `None` when no finite value
/// survives filtering (the property is then omitted from the block).
fn numeric_hint(name: &str, raw: &[f64]) -> Option<PropertyStyleHint> {
    let mut vals: Vec<f64> = raw.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = vals[0];
    let max = *vals.last().expect("non-empty per check");
    let p97 = percentile(&vals, 0.97);
    Some(PropertyStyleHint {
        name: name.to_string(),
        min: Some(min),
        p50: Some(percentile(&vals, 0.50)),
        p90: Some(percentile(&vals, 0.90)),
        p95: Some(percentile(&vals, 0.95)),
        p97: Some(p97),
        p99: Some(percentile(&vals, 0.99)),
        max: Some(max),
        suggested_domain: Some([
            round_2sf_outward(min, false),
            round_2sf_outward(p97, true),
        ]),
        cardinality: None,
    })
}

/// Sort-based percentile matching stt-build's heatmap-domain convention:
/// `sorted[min(floor(n * q), n - 1)]`.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() as f64 * q).floor() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Round `v` to 2 significant figures, OUTWARD from the suggested domain:
/// `up = false` for the lower endpoint (toward −∞), `up = true` for the upper
/// (toward +∞). Outward rounding guarantees the rounded domain still covers
/// the raw `[min, p97]` interval. A small epsilon tolerates float noise so an
/// already-2-sig-fig endpoint round-trips to itself (5.3 stays 5.3).
fn round_2sf_outward(v: f64, up: bool) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    // Exponent of the 2nd significant digit: 5.27 -> -1, 971 -> 1, 0.123 -> -2.
    let exp = v.abs().log10().floor() as i32 - 1;
    let pow = 10f64.powi(exp.abs());
    let scaled = if exp >= 0 { v / pow } else { v * pow };
    let eps = 1e-9;
    let r = if up {
        (scaled - eps).ceil()
    } else {
        (scaled + eps).floor()
    };
    if exp >= 0 {
        r * pow
    } else {
        // Divide (not multiply by 10^exp) so the result is the correctly
        // rounded double, which prints clean: 53 / 10 -> 5.3, not 5.300…01.
        r / pow
    }
}

/// `clamp(round(sqrt(bucket_count)), 20, 90)` — short archives still get a
/// watchable loop, huge ones don't crawl. `None` when the bucket size is
/// unknown (0): no defensible default exists without a bucket count.
fn suggested_playback_seconds(time_range_ms: u64, temporal_bucket_ms: u64) -> Option<u32> {
    if temporal_bucket_ms == 0 {
        return None;
    }
    let buckets = time_range_ms as f64 / temporal_bucket_ms as f64;
    Some((buckets.sqrt().round() as u64).clamp(20, 90) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric(name: &str, vals: Vec<f64>) -> (String, PropertyValues) {
        (name.to_string(), PropertyValues::Numeric(vals))
    }

    #[test]
    fn percentiles_on_known_1000_value_vec() {
        // 1.0..=1000.0 shuffled deterministically: percentile indexing must
        // match the heatmap-domain convention sorted[floor(n*q)].
        let mut vals: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        vals.reverse(); // profiler sorts internally
        let hints = profile_properties(&[numeric("v", vals)], 0, 0, None);
        assert_eq!(hints.version, STYLE_HINTS_VERSION);
        let p = &hints.properties[0];
        assert_eq!(p.name, "v");
        assert_eq!(p.min, Some(1.0));
        assert_eq!(p.p50, Some(501.0)); // sorted[floor(1000*0.50)] = sorted[500]
        assert_eq!(p.p90, Some(901.0));
        assert_eq!(p.p95, Some(951.0));
        assert_eq!(p.p97, Some(971.0));
        assert_eq!(p.p99, Some(991.0));
        assert_eq!(p.max, Some(1000.0));
        // Domain: [min, p97] rounded outward to 2 sig figs -> [1, 980].
        assert_eq!(p.suggested_domain, Some([1.0, 980.0]));
        assert_eq!(p.cardinality, None);
    }

    #[test]
    fn domain_rounds_outward_to_two_sig_figs() {
        // 100 values: min = 0.123, sorted[97] (= p97) = 5.27.
        let mut vals = vec![0.123];
        vals.extend((1..=96).map(|i| 0.2 + i as f64 * 0.05)); // 0.25..5.0
        vals.extend([5.27, 6.0, 9.1]);
        assert_eq!(vals.len(), 100);
        let hints = profile_properties(&[numeric("mag", vals)], 0, 0, None);
        let p = &hints.properties[0];
        assert_eq!(p.p97, Some(5.27));
        assert_eq!(p.suggested_domain, Some([0.12, 5.3]));
    }

    #[test]
    fn outward_rounding_handles_signs_and_exact_endpoints() {
        // Lower endpoint rounds toward -inf, upper toward +inf.
        assert_eq!(round_2sf_outward(0.123, false), 0.12);
        assert_eq!(round_2sf_outward(5.27, true), 5.3);
        assert_eq!(round_2sf_outward(-0.123, false), -0.13);
        assert_eq!(round_2sf_outward(-5.27, true), -5.2);
        assert_eq!(round_2sf_outward(971.0, true), 980.0);
        // Already 2 sig figs: stays put in both directions.
        assert_eq!(round_2sf_outward(5.3, true), 5.3);
        assert_eq!(round_2sf_outward(5.3, false), 5.3);
        assert_eq!(round_2sf_outward(0.0, true), 0.0);
    }

    #[test]
    fn nan_and_infinite_values_are_filtered() {
        let vals = vec![f64::NAN, 1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY, 3.0];
        let hints = profile_properties(&[numeric("v", vals)], 0, 0, None);
        let p = &hints.properties[0];
        assert_eq!(p.min, Some(1.0));
        assert_eq!(p.max, Some(3.0));
    }

    #[test]
    fn empty_or_all_nan_numeric_property_is_omitted() {
        let hints = profile_properties(
            &[
                numeric("empty", vec![]),
                numeric("nans", vec![f64::NAN, f64::NAN]),
                numeric("ok", vec![1.0]),
            ],
            0,
            0,
            None,
        );
        assert_eq!(hints.properties.len(), 1);
        assert_eq!(hints.properties[0].name, "ok");
    }

    #[test]
    fn categorical_carries_cardinality_only() {
        let hints = profile_properties(
            &[("category".to_string(), PropertyValues::Categorical { distinct: 7 })],
            0,
            0,
            None,
        );
        let p = &hints.properties[0];
        assert_eq!(p.name, "category");
        assert_eq!(p.cardinality, Some(7));
        assert_eq!(p.min, None);
        assert_eq!(p.p50, None);
        assert_eq!(p.p97, None);
        assert_eq!(p.max, None);
        assert_eq!(p.suggested_domain, None);
    }

    #[test]
    fn playback_seconds_clamps_and_guards_zero_bucket() {
        let hour = 3_600_000u64;
        // 4 buckets -> sqrt = 2 -> clamps up to 20.
        assert_eq!(
            profile_properties(&[], 4 * hour, hour, None).suggested_playback_seconds,
            Some(20)
        );
        // 2025 buckets -> sqrt = 45 -> passes through.
        assert_eq!(
            profile_properties(&[], 2025 * hour, hour, None).suggested_playback_seconds,
            Some(45)
        );
        // 100_000 buckets -> sqrt ~316 -> clamps down to 90.
        assert_eq!(
            profile_properties(&[], 100_000 * hour, hour, None).suggested_playback_seconds,
            Some(90)
        );
        // Zero bucket size: no defensible default -> absent.
        assert_eq!(
            profile_properties(&[], hour, 0, None).suggested_playback_seconds,
            None
        );
    }

    #[test]
    fn layer_hint_passes_known_vocabulary_only() {
        assert_eq!(
            profile_properties(&[], 0, 0, Some("trips")).layer_hint.as_deref(),
            Some("trips")
        );
        assert_eq!(profile_properties(&[], 0, 0, Some("hexagons")).layer_hint, None);
        assert_eq!(profile_properties(&[], 0, 0, None).layer_hint, None);
    }
}
