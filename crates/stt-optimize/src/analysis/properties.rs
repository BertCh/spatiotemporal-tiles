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

/// Target implied DATA frame rate (buckets rendered per second of playback) the
/// refit playback duration aims for.
///
/// The legacy `sqrt(bucket_count)` duration has no frame-rate meaning: a
/// 10-bucket archive clamped up to 20 s plays one frame per 2 s (a slideshow),
/// while a 100 000-bucket archive clamped down to 90 s implies ~1 100 data-fps
/// (a blur). Fitting the duration to a frame rate puts both in the range a human
/// reads as motion.
pub const F_TARGET_FPS: f64 = 20.0;

/// Hard floor / ceiling on the refit playback duration, in seconds. Below the
/// floor a loop is too short to read; above the ceiling it stops feeling like a
/// loop at all. These bound the frame-rate fit, so very small and very large
/// archives leave the `[K/30, K/12]` band by design.
pub const PLAYBACK_SECONDS_FLOOR: f64 = 5.0;
/// See [`PLAYBACK_SECONDS_FLOOR`].
pub const PLAYBACK_SECONDS_CEILING: f64 = 300.0;

/// Reference client memory budget for RESIDENT tile payload, in bytes, used to
/// derive [`StyleHints::suggested_time_window_ms`].
///
/// ⚠️ **Knowingly a constant where a function could be.** The honest form is a
/// per-session budget the client measures (`navigator.deviceMemory`, the
/// observed cache high-water mark, or a governor-reported ceiling) and feeds
/// back; the build has no access to any of that, so the archive publishes a
/// window fitted to ONE reference device and the reader overrides it whenever it
/// knows better. 256 MiB is derived once, from the device budget the eviction
/// tiering is sized against — the low end of the phones the showcase targets,
/// leaving headroom for the renderer's own buffers.
///
/// **Revival trigger:** when the client reports a measured resident-byte budget
/// through the telemetry surface, this constant becomes that reading's default
/// and the derivation moves reader-side. Until then a re-fit means changing this
/// number and rebuilding — which is exactly why the value it produces is
/// recorded in the manifest rather than recomputed at read time.
pub const M_REF: u64 = 256 * 1024 * 1024;

/// Cap on the suggested window, in temporal buckets. A window wider than ~a day
/// of buckets stops being a window and starts being "the whole archive", which
/// is not a playback default under any memory budget.
pub const MAX_WINDOW_BUCKETS: u64 = 24;

/// Inputs for the derived playback parameters (BH-10), separated from the
/// profiler's own inputs because they come from a different stage of the build:
/// the payload total is only known after tiling.
///
/// [`Default`] is the LEGACY behaviour — `refit` off, no byte total — so
/// `profile_properties` and every existing caller keep emitting exactly the
/// bytes they emit today.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlaybackDerivation {
    /// Total UNCOMPRESSED tile-payload bytes the build produced, or `None` when
    /// the pipeline could not total them (streaming builds, re-profilers). Drives
    /// `suggested_time_window_ms`; `None` omits that field.
    pub total_payload_bytes: Option<u64>,
    /// Use the frame-rate refit for `suggested_playback_seconds` instead of the
    /// legacy `sqrt(bucket_count)` formula.
    ///
    /// OFF by default and gated at the CLI (`--derived-playback-params`): turning
    /// it on CHANGES the emitted hint value for every rebuilt archive, which is
    /// a fleet-wide change to how demos pace. It rides the next rebuild window
    /// rather than drifting in one archive at a time.
    pub refit: bool,
}

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
    profile_properties_with(
        props,
        time_range_ms,
        temporal_bucket_ms,
        layer_kind_hint,
        PlaybackDerivation::default(),
    )
}

/// [`profile_properties`] with the derived playback parameters (BH-10).
///
/// `derivation` carries what the profiler cannot know on its own: the build's
/// total payload bytes, and whether to use the frame-rate refit for the playback
/// duration. [`PlaybackDerivation::default()`] reproduces `profile_properties`
/// byte-for-byte, so this is purely additive — the incumbent formula stays the
/// documented fallback rather than being deleted.
pub fn profile_properties_with(
    props: &[(String, PropertyValues)],
    time_range_ms: u64,
    temporal_bucket_ms: u64,
    layer_kind_hint: Option<&str>,
    derivation: PlaybackDerivation,
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
        suggested_playback_seconds: if derivation.refit {
            refit_playback_seconds(time_range_ms, temporal_bucket_ms)
        } else {
            suggested_playback_seconds(time_range_ms, temporal_bucket_ms)
        },
        suggested_time_window_ms: derivation
            .total_payload_bytes
            .and_then(|bytes| suggested_time_window_ms(time_range_ms, temporal_bucket_ms, bytes)),
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
        suggested_domain: Some([round_2sf_outward(min, false), round_2sf_outward(p97, true)]),
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

/// The frame-rate refit (BH-10): `P = clamp(K / F_TARGET_FPS, K/30, K/12)`
/// further clamped to `[5, 300]` s, with `K` = bucket count.
///
/// The inner band is what makes this a FIT rather than a division: whatever
/// [`F_TARGET_FPS`] is re-fitted to, the implied data frame rate stays inside
/// `[12, 30]` — the range that reads as motion rather than a slideshow or a
/// blur. The outer clamp is the human bound on loop length and deliberately wins
/// over the band at both extremes (a 10-bucket archive gets 5 s / 2 data-fps
/// because a 0.5 s loop is unusable, not because 2 fps is good).
///
/// `None` when the bucket size is unknown (0) — same guard as the legacy
/// formula, same reason: no bucket count, no defensible duration.
fn refit_playback_seconds(time_range_ms: u64, temporal_bucket_ms: u64) -> Option<u32> {
    if temporal_bucket_ms == 0 {
        return None;
    }
    let k = time_range_ms as f64 / temporal_bucket_ms as f64;
    // k >= 0 always, so k/30 <= k/12 and the clamp cannot invert.
    let banded = (k / F_TARGET_FPS).clamp(k / 30.0, k / 12.0);
    let seconds = banded.clamp(PLAYBACK_SECONDS_FLOOR, PLAYBACK_SECONDS_CEILING);
    Some(seconds.round() as u32)
}

/// The byte-feasible playback window (BH-10):
/// `min(argmax{W : β̄·W ≤ M_REF}, MAX_WINDOW_BUCKETS·Δ)`, floored at one bucket.
///
/// `β̄` = `total_payload_bytes / time_range_ms` is the archive's mean payload
/// bytes per millisecond of span, so `argmax{W : β̄·W ≤ M_REF}` is
/// `M_REF · span / bytes` — computed in `u128` integer arithmetic so the answer
/// is exactly reproducible and cannot overflow on a 300 GiB archive.
///
/// The floor at one bucket is a correctness bound, not a taste one: a window
/// narrower than `Δ` can address no complete temporal bucket, so it would
/// display nothing. A dense archive therefore gets `Δ` and pays the memory —
/// the alternative would be a hint that renders an empty map.
///
/// `None` when any input makes the ratio meaningless (no bucket, no span, no
/// bytes); the reader then keeps its own bucket-multiple default.
fn suggested_time_window_ms(
    time_range_ms: u64,
    temporal_bucket_ms: u64,
    total_payload_bytes: u64,
) -> Option<u64> {
    if temporal_bucket_ms == 0 || time_range_ms == 0 || total_payload_bytes == 0 {
        return None;
    }
    let feasible =
        (u128::from(M_REF) * u128::from(time_range_ms)) / u128::from(total_payload_bytes);
    let capped = feasible.min(u128::from(
        temporal_bucket_ms.saturating_mul(MAX_WINDOW_BUCKETS),
    ));
    Some((capped as u64).max(temporal_bucket_ms))
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
            &[(
                "category".to_string(),
                PropertyValues::Categorical { distinct: 7 },
            )],
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

    /// BH-10 band math. The refit holds the implied data frame rate inside
    /// `[12, 30]` wherever the `[5, 300]` s human bound leaves room, and hands
    /// the extremes to that bound rather than to the band.
    #[test]
    fn refit_playback_seconds_holds_the_frame_rate_band_between_its_clamps() {
        let hour = 3_600_000u64;
        let refit = PlaybackDerivation {
            refit: true,
            ..Default::default()
        };
        let p = |buckets: u64| {
            profile_properties_with(&[], buckets * hour, hour, None, refit)
                .suggested_playback_seconds
                .expect("a known bucket size always yields a duration")
        };

        // Mid-range: the fit binds and the data-fps lands on the target.
        for buckets in [200u64, 1_000, 2_025, 4_000] {
            let seconds = p(buckets);
            let fps = buckets as f64 / f64::from(seconds);
            assert!(
                (12.0..=30.0).contains(&fps),
                "{buckets} buckets → {seconds}s → {fps} data-fps, outside [12, 30]"
            );
            assert!((5..=300).contains(&seconds));
        }

        // K = 10: K/20 = 0.5 s, so the 5 s FLOOR wins. Outside the band by
        // design — a half-second loop is unusable — but a huge improvement on
        // the legacy 20 s (which played one frame per 2 s).
        assert_eq!(p(10), 5);
        assert!(
            profile_properties(&[], 10 * hour, hour, None).suggested_playback_seconds == Some(20),
            "legacy formula must be untouched by the refit"
        );

        // K = 100_000: K/20 = 5_000 s, so the 300 s CEILING wins (legacy: 90 s).
        assert_eq!(p(100_000), 300);

        // Unknown bucket size: absent under both formulas.
        assert_eq!(
            profile_properties_with(&[], hour, 0, None, refit).suggested_playback_seconds,
            None
        );
    }

    /// The refit is OFF unless asked for: the default derivation must reproduce
    /// the legacy value exactly, or every rebuilt archive's pacing moves by
    /// accident.
    #[test]
    fn the_default_derivation_is_byte_for_byte_the_legacy_hint() {
        let hour = 3_600_000u64;
        for buckets in [0u64, 1, 4, 400, 2_025, 100_000] {
            let legacy = profile_properties(&[], buckets * hour, hour, Some("points"));
            let defaulted = profile_properties_with(
                &[],
                buckets * hour,
                hour,
                Some("points"),
                PlaybackDerivation::default(),
            );
            assert_eq!(legacy, defaulted, "{buckets} buckets");
            assert_eq!(
                defaulted.suggested_time_window_ms, None,
                "no byte total ⇒ no window hint"
            );
        }
    }

    /// BH-10 window derivation: the byte-feasible width, capped at 24 buckets
    /// and floored at one.
    #[test]
    fn suggested_time_window_is_the_byte_feasible_width_capped_and_floored() {
        let hour = 3_600_000u64;
        let span = 1_000 * hour;
        let window = |bytes: u64| {
            profile_properties_with(
                &[],
                span,
                hour,
                None,
                PlaybackDerivation {
                    total_payload_bytes: Some(bytes),
                    refit: true,
                },
            )
            .suggested_time_window_ms
        };

        // Sparse archive: the 24-bucket CAP binds, not the memory budget.
        assert_eq!(window(1024), Some(24 * hour));

        // Mid-density: β̄ = bytes/span, so W = M_REF·span/bytes. Pick bytes so
        // the answer is exactly 8 buckets.
        let bytes =
            u64::try_from(u128::from(M_REF) * u128::from(span) / u128::from(8 * hour)).unwrap();
        assert_eq!(window(bytes), Some(8 * hour));

        // Dense archive: the feasible width collapses below one bucket, and the
        // floor takes over — a narrower window could address no whole bucket.
        assert_eq!(window(u64::from(u32::MAX) * 1024), Some(hour));

        // Missing inputs ⇒ absent, never a guess.
        assert_eq!(window(0), None);
        assert_eq!(
            profile_properties_with(
                &[],
                0,
                hour,
                None,
                PlaybackDerivation {
                    total_payload_bytes: Some(1_000_000),
                    refit: true
                }
            )
            .suggested_time_window_ms,
            None
        );
        assert_eq!(
            profile_properties_with(
                &[],
                span,
                0,
                None,
                PlaybackDerivation {
                    total_payload_bytes: Some(1_000_000),
                    refit: true
                }
            )
            .suggested_time_window_ms,
            None
        );
    }

    /// Determinism: both derived parameters are pure integer/IEEE functions of
    /// their inputs, so repeated calls agree exactly.
    #[test]
    fn derived_playback_params_are_deterministic() {
        let hour = 3_600_000u64;
        let d = PlaybackDerivation {
            total_payload_bytes: Some(987_654_321),
            refit: true,
        };
        let first = profile_properties_with(&[], 4_321 * hour, hour, Some("trips"), d);
        for _ in 0..16 {
            assert_eq!(
                profile_properties_with(&[], 4_321 * hour, hour, Some("trips"), d),
                first
            );
        }
    }

    #[test]
    fn layer_hint_passes_known_vocabulary_only() {
        assert_eq!(
            profile_properties(&[], 0, 0, Some("trips"))
                .layer_hint
                .as_deref(),
            Some("trips")
        );
        assert_eq!(
            profile_properties(&[], 0, 0, Some("hexagons")).layer_hint,
            None
        );
        assert_eq!(profile_properties(&[], 0, 0, None).layer_hint, None);
    }
}
