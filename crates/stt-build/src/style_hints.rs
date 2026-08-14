//! Build-time `style_hints` collection.
//!
//! Bridges the loaded feature set to the generic stt-optimize profiler
//! ([`stt_optimize::analysis::properties`]): a layer-kind hint derived from the
//! kinds the build produces plus a suggested playback duration (both cheap,
//! emitted on EVERY non-streaming build) and — only under `--style-hints` —
//! bounded per-property value sampling with categorical distinct counting. The
//! resulting [`stt_core::metadata::StyleHints`] is attached to archive metadata
//! (and so flows into the packed `manifest.json` verbatim). In-memory pipeline
//! only — the profiler needs the whole feature slice.

use crate::input::ParsedFeature;
use std::collections::{BTreeMap, HashSet};
use stt_core::metadata::StyleHints;
use stt_core::types::TimeRange;
use stt_optimize::analysis::properties::{
    profile_properties_with, PlaybackDerivation, PropertyValues,
};

pub use stt_optimize::analysis::properties::PlaybackDerivation as DerivedPlaybackParams;

/// Memory guard: at most this many sampled values per property. Sampling is a
/// DETERMINISTIC stride over the feature slice (every `ceil(n / cap)`-th
/// feature), so re-builds of the same input emit byte-identical hints.
pub const MAX_VALUES_PER_PROPERTY: usize = 250_000;

/// Cap on the distinct-string set tracked per categorical property. When the
/// cap is hit the reported cardinality is exactly this value: "at least 10k
/// distinct values" is already actionable for palette sizing (every real
/// categorical ramp saturates far below it), so an exact count isn't worth
/// unbounded memory. Note the count is over the stride SAMPLE, so it is a
/// lower bound on the true cardinality either way.
pub const MAX_DISTINCT_VALUES: usize = 10_000;

/// Per-property accumulator over the sampled features.
#[derive(Default)]
struct Collector {
    numeric: Vec<f64>,
    numeric_seen: u64,
    strings: HashSet<String>,
    string_seen: u64,
}

/// Profile the loaded features into a [`StyleHints`] block.
///
/// Returns `None` for an empty feature set.
///
/// `full` controls how much is computed:
/// * `false` (the DEFAULT on every non-streaming build) — the cheap signals
///   only: the geometry-derived [`layer_hint`] and the
///   `suggested_playback_seconds` duration. `properties` is empty. This is what
///   makes view-time layer inference and a sensible playback loop work without
///   the user opting into the expensive profiler.
/// * `true` (`stt-build --style-hints`) — additionally scans every feature's
///   properties at a deterministic stride (see [`MAX_VALUES_PER_PROPERTY`]) for
///   numeric percentiles / categorical cardinality: JSON numbers widen to
///   `f64`, strings feed a capped distinct set ([`MAX_DISTINCT_VALUES`]);
///   bool/null/array/object values carry no styling signal and are skipped. A
///   column with both numeric and string values resolves to the majority kind.
///
/// `time_range` + `temporal_bucket_ms` drive the suggested playback duration in
/// both modes.
pub fn compute_style_hints(
    features: &[ParsedFeature],
    time_range: &TimeRange,
    temporal_bucket_ms: u64,
    full: bool,
) -> Option<StyleHints> {
    compute_style_hints_with(
        features,
        time_range,
        temporal_bucket_ms,
        full,
        PlaybackDerivation::default(),
    )
}

/// [`compute_style_hints`] with the derived playback parameters (BH-10).
///
/// `derivation` carries the build's total tile-payload bytes (only known after
/// tiling — the caller reads it off the pack writer) and the refit gate for
/// `suggested_playback_seconds`. [`PlaybackDerivation::default()`] reproduces
/// [`compute_style_hints`] exactly, which is what keeps the emission opt-in:
/// nothing in a manifest moves until a build asks for it.
pub fn compute_style_hints_with(
    features: &[ParsedFeature],
    time_range: &TimeRange,
    temporal_bucket_ms: u64,
    full: bool,
    derivation: PlaybackDerivation,
) -> Option<StyleHints> {
    if features.is_empty() {
        return None;
    }
    let props = if full {
        collect_property_values(features)
    } else {
        Vec::new()
    };

    Some(profile_properties_with(
        &props,
        time_range.end.saturating_sub(time_range.start),
        temporal_bucket_ms,
        layer_hint(features),
        derivation,
    ))
}

/// The expensive per-property value scan behind `--style-hints`: a deterministic
/// stride over the feature slice collecting numeric samples and capped
/// categorical distinct-sets, resolved to the majority kind per column.
fn collect_property_values(features: &[ParsedFeature]) -> Vec<(String, PropertyValues)> {
    let stride = features.len().div_ceil(MAX_VALUES_PER_PROPERTY).max(1);
    // BTreeMap: deterministic property order in the emitted block.
    let mut collectors: BTreeMap<String, Collector> = BTreeMap::new();
    for feature in features.iter().step_by(stride) {
        let Some(props) = feature.shared_properties.as_ref() else {
            continue;
        };
        // `as_f64` is numbers-only (a numeric-looking STRING stays a string
        // here, as it did when this matched `Value::Number` directly), and
        // `iter` never yields nulls — so bool/null/array/object still
        // contribute no styling signal.
        for (name, value) in props.iter() {
            if let Some(v) = value.as_f64().filter(|v| v.is_finite()) {
                let c = collectors.entry(name.to_string()).or_default();
                c.numeric_seen += 1;
                c.numeric.push(v);
            } else if let Some(s) = value.as_str() {
                let c = collectors.entry(name.to_string()).or_default();
                c.string_seen += 1;
                if c.strings.len() < MAX_DISTINCT_VALUES {
                    c.strings.insert(s.to_string());
                }
            }
        }
    }

    collectors
        .into_iter()
        .filter_map(|(name, c)| {
            if !c.numeric.is_empty() && c.numeric_seen >= c.string_seen {
                Some((name, PropertyValues::Numeric(c.numeric)))
            } else if c.string_seen > 0 {
                Some((
                    name,
                    PropertyValues::Categorical {
                        distinct: c.strings.len(),
                    },
                ))
            } else {
                None
            }
        })
        .collect()
}

/// Layer hint from the kinds the build will produce: points → `"points"`,
/// lines WITH per-vertex times (explicit `vertex_timestamps` or an
/// interpolatable `[start, end]` range) → `"trips"`, timeless lines →
/// `"paths"`, polygons → `"polygons"`. Mixed inputs resolve to the kind with
/// the most features; ties break in that fixed order for determinism.
fn layer_hint(features: &[ParsedFeature]) -> Option<&'static str> {
    let (mut points, mut paths, mut trips, mut polygons) = (0u64, 0u64, 0u64, 0u64);
    for f in features {
        use geojson::Value as G;
        match f.geojson.geometry.as_ref().map(|g| &g.value) {
            Some(G::Point(_)) | Some(G::MultiPoint(_)) => points += 1,
            Some(G::LineString(_)) | Some(G::MultiLineString(_)) => {
                if f.vertex_timestamps.is_some() || f.end_timestamp.is_some_and(|e| e > f.timestamp)
                {
                    trips += 1;
                } else {
                    paths += 1;
                }
            }
            Some(G::Polygon(_)) | Some(G::MultiPolygon(_)) => polygons += 1,
            _ => {}
        }
    }
    // `max_by_key` keeps the LAST maximum, so iterate the ranking reversed to
    // make the FIRST entry win ties.
    [
        (points, "points"),
        (trips, "trips"),
        (paths, "paths"),
        (polygons, "polygons"),
    ]
    .iter()
    .rev()
    .filter(|(n, _)| *n > 0)
    .max_by_key(|(n, _)| *n)
    .map(|&(_, kind)| kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geojson::{Feature, Geometry, Value};

    fn feature(
        geometry: Value,
        props: serde_json::Value,
        timestamp: u64,
        end_timestamp: Option<u64>,
        vertex_timestamps: Option<Vec<u64>>,
    ) -> ParsedFeature {
        ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(geometry)),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: props
                .as_object()
                .cloned()
                .and_then(crate::props::FeatureProperties::from_map),
            timestamp,
            end_timestamp,
            vertex_timestamps,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: 0.0,
            lat: 0.0,
        }
    }

    fn point(props: serde_json::Value) -> ParsedFeature {
        feature(Value::Point(vec![0.0, 0.0]), props, 1_000, None, None)
    }

    fn line(end: Option<u64>, vertex_ts: Option<Vec<u64>>) -> ParsedFeature {
        feature(
            Value::LineString(vec![vec![0.0, 0.0], vec![1.0, 1.0]]),
            serde_json::json!({}),
            1_000,
            end,
            vertex_ts,
        )
    }

    #[test]
    fn empty_features_yield_no_hints() {
        assert!(compute_style_hints(&[], &TimeRange::new(0, 1), 1, false).is_none());
        assert!(compute_style_hints(&[], &TimeRange::new(0, 1), 1, true).is_none());
    }

    #[test]
    fn minimal_mode_emits_layer_hint_and_playback_but_no_properties() {
        // full=false: the cheap default on every build. Even though the
        // features carry a numeric property, the expensive profile is skipped —
        // only layer_hint + playback survive.
        let features: Vec<ParsedFeature> = (0..100)
            .map(|i| point(serde_json::json!({ "magnitude": i as f64 / 10.0 })))
            .collect();
        let hints = compute_style_hints(
            &features,
            &TimeRange::new(0, 4 * 3_600_000),
            3_600_000,
            false,
        )
        .unwrap();
        assert_eq!(hints.layer_hint.as_deref(), Some("points"));
        assert_eq!(hints.suggested_playback_seconds, Some(20));
        assert!(hints.properties.is_empty(), "{:?}", hints.properties);
    }

    #[test]
    fn numeric_and_categorical_properties_profile() {
        let features: Vec<ParsedFeature> = (0..100)
            .map(|i| {
                point(serde_json::json!({
                    "magnitude": i as f64 / 10.0,
                    "category": format!("class-{}", i % 7),
                    "flag": true, // bool: no styling signal, must be skipped
                }))
            })
            .collect();
        let hints = compute_style_hints(
            &features,
            &TimeRange::new(0, 4 * 3_600_000),
            3_600_000,
            true,
        )
        .unwrap();
        assert_eq!(hints.layer_hint.as_deref(), Some("points"));
        assert_eq!(hints.suggested_playback_seconds, Some(20)); // 4 buckets clamps up
        assert_eq!(hints.properties.len(), 2, "{:?}", hints.properties);
        // BTreeMap order: category < magnitude.
        assert_eq!(hints.properties[0].name, "category");
        assert_eq!(hints.properties[0].cardinality, Some(7));
        assert_eq!(hints.properties[0].min, None);
        assert_eq!(hints.properties[1].name, "magnitude");
        assert_eq!(hints.properties[1].min, Some(0.0));
        assert_eq!(hints.properties[1].max, Some(9.9));
        assert!(hints.properties[1].suggested_domain.is_some());
        assert_eq!(hints.properties[1].cardinality, None);
    }

    #[test]
    fn lines_with_vertex_times_hint_trips_without_hint_paths() {
        let trips: Vec<ParsedFeature> = vec![
            line(Some(2_000), None),
            line(None, Some(vec![1_000, 2_000])),
        ];
        assert_eq!(
            compute_style_hints(&trips, &TimeRange::new(0, 1), 1, false)
                .unwrap()
                .layer_hint
                .as_deref(),
            Some("trips")
        );
        let paths: Vec<ParsedFeature> = vec![line(None, None), line(None, None)];
        assert_eq!(
            compute_style_hints(&paths, &TimeRange::new(0, 1), 1, false)
                .unwrap()
                .layer_hint
                .as_deref(),
            Some("paths")
        );
    }

    #[test]
    fn mixed_kinds_resolve_to_the_majority() {
        let mut features: Vec<ParsedFeature> =
            (0..3).map(|_| point(serde_json::json!({}))).collect();
        features.push(line(None, None));
        assert_eq!(
            compute_style_hints(&features, &TimeRange::new(0, 1), 1, false)
                .unwrap()
                .layer_hint
                .as_deref(),
            Some("points")
        );
    }

    #[test]
    fn stride_sampling_is_deterministic_and_bounded() {
        // With the cap at 250k, 2 features/cap would need a synthetic giant
        // vec — instead pin the stride math itself.
        assert_eq!(600_000usize.div_ceil(MAX_VALUES_PER_PROPERTY), 3);
        assert_eq!(250_000usize.div_ceil(MAX_VALUES_PER_PROPERTY), 1);
        assert_eq!(250_001usize.div_ceil(MAX_VALUES_PER_PROPERTY), 2);
        // And that repeated runs over the same input agree.
        let features: Vec<ParsedFeature> = (0..50)
            .map(|i| point(serde_json::json!({ "v": i })))
            .collect();
        let tr = TimeRange::new(0, 3_600_000);
        let a = compute_style_hints(&features, &tr, 3_600_000, true).unwrap();
        let b = compute_style_hints(&features, &tr, 3_600_000, true).unwrap();
        assert_eq!(a, b);
    }

    /// BH-10 at the build seam. The gate is OFF by default, so the emitted
    /// block must be byte-for-byte what it was before the field existed; ON, the
    /// duration is refit and the window appears.
    #[test]
    fn derived_playback_params_are_opt_in_at_this_seam() {
        let hour = 3_600_000u64;
        let features: Vec<ParsedFeature> = (0..40)
            .map(|i| point(serde_json::json!({ "v": i })))
            .collect();
        let tr = TimeRange::new(0, 4_000 * hour);

        // Default: legacy duration (sqrt(4000) ≈ 63), no window.
        let legacy = compute_style_hints(&features, &tr, hour, false).unwrap();
        assert_eq!(legacy.suggested_playback_seconds, Some(63));
        assert_eq!(legacy.suggested_time_window_ms, None);
        assert_eq!(
            compute_style_hints_with(
                &features,
                &tr,
                hour,
                false,
                DerivedPlaybackParams::default()
            )
            .unwrap(),
            legacy,
            "the default derivation must be indistinguishable from the legacy call"
        );

        // Gated on: 4000/20 = 200 s, and a window derived from the byte total.
        let derived = compute_style_hints_with(
            &features,
            &tr,
            hour,
            false,
            DerivedPlaybackParams {
                total_payload_bytes: Some(64 * 1024 * 1024),
                refit: true,
            },
        )
        .unwrap();
        assert_eq!(derived.suggested_playback_seconds, Some(200));
        let window = derived.suggested_time_window_ms.expect("window derived");
        assert!(window >= hour, "never narrower than one bucket");
        assert!(window <= 24 * hour, "never wider than the 24-bucket cap");
        // Everything else about the block is untouched by the gate.
        assert_eq!(derived.layer_hint, legacy.layer_hint);
        assert_eq!(derived.properties, legacy.properties);
        assert_eq!(derived.version, legacy.version);
    }

    /// Determinism with the gate ON: the mandatory byte-identical re-run.
    #[test]
    fn derived_hints_are_identical_across_reruns() {
        let hour = 3_600_000u64;
        let features: Vec<ParsedFeature> = (0..50)
            .map(|i| point(serde_json::json!({ "v": i, "c": format!("k{}", i % 5) })))
            .collect();
        let tr = TimeRange::new(0, 777 * hour);
        let d = DerivedPlaybackParams {
            total_payload_bytes: Some(123_456_789),
            refit: true,
        };
        let a = compute_style_hints_with(&features, &tr, hour, true, d).unwrap();
        for _ in 0..8 {
            let b = compute_style_hints_with(&features, &tr, hour, true, d).unwrap();
            assert_eq!(a, b);
            // The serialized bytes, not just the struct — the manifest is what ships.
            assert_eq!(
                serde_json::to_vec(&a).unwrap(),
                serde_json::to_vec(&b).unwrap()
            );
        }
    }

    #[test]
    fn distinct_count_caps_at_limit() {
        let features: Vec<ParsedFeature> = (0..(MAX_DISTINCT_VALUES + 500))
            .map(|i| point(serde_json::json!({ "id": format!("unique-{i}") })))
            .collect();
        let hints = compute_style_hints(&features, &TimeRange::new(0, 1), 1, true).unwrap();
        assert_eq!(
            hints.properties[0].cardinality,
            Some(MAX_DISTINCT_VALUES as u32)
        );
    }
}
