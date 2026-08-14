//! DT-2 — additive home-zoom assignment (§1.1, §4.5).
//!
//! Today `place_feature` REPLICATES every feature into every zoom of its band:
//! the O(|Z|·N) corner of the assignment polytope. Measured on a storm-cell
//! rebuild, that is a duplication factor of **36×** (6,694 stored index rows
//! for 186 source features), and geometry is the dominant byte share (31.8 %
//! alone; 56.1 % counting the per-vertex columns that replicate with it).
//!
//! Additive decomposition is the O(N) corner: each feature is claimed by
//! exactly ONE home zoom, and the reader unions across `[minZoom..z]`
//! (`lodMode: 'additive'`, which already ships). This module computes the
//! assignment; it does not touch the placement authority — it synthesizes the
//! per-feature band and routes through the EXISTING `min_zoom_field` /
//! `max_zoom_field` mechanism.
//!
//! ## Determinism
//!
//! Coarse-to-fine voxel claiming with a total order on candidates —
//! `(importance desc, feature id asc)` — no RNG, no arrival-order dependence,
//! and a pinned voxel origin. Byte-reproducible across thread counts.
//!
//! ## The bucket-keyed constraint (register: hard)
//!
//! For animated datasets the voxel key MUST include `⌊t / temporal_bucket_ms⌋`.
//! A space-only grid over time-varying data showed a median 13 % of the visible
//! bucket at z8 (worst case 0 %); keying the bucket in took it to 65 %. Never
//! revert to space-only.

use std::collections::BTreeMap;

/// Metres per pixel at zoom 0 on the equator — the Web Mercator constant.
const M_PER_PX_Z0: f64 = 156_543.033_928_04;

/// Default voxel pitch in screen pixels.
///
/// ⚠️ P1 debt, recorded rather than hidden: 0.4 comes from ONE Miami sweep and
/// has never been re-fit. It is the knob most likely to want fitting before any
/// fleet dataset graduates.
pub const DEFAULT_HOME_ZOOM_PX: f64 = 0.4;

/// One feature's inputs to the assignment.
#[derive(Debug, Clone, Copy)]
pub struct HomeZoomCandidate {
    /// Stable feature id — the §3.6 FNV-1a-64 identity. The total tiebreak.
    pub id: u64,
    pub lon: f64,
    pub lat: f64,
    /// Feature timestamp, for the bucket-keyed voxel.
    pub timestamp: u64,
    /// Higher wins a contested voxel. Absent importance ⇒ 0.0, which changes
    /// WHICH feature claims a voxel, never WHETHER coverage holds.
    pub importance: f64,
}

/// Voxel pitch in degrees of longitude at `zoom`, for a voxel of `s_px` pixels.
///
/// `s_px · 156543·cos(lat) / 2^z` metres, expressed in degrees so the claim
/// grid is directly comparable with lon/lat inputs.
pub fn voxel_pitch_deg(s_px: f64, zoom: u8, lat: f64) -> f64 {
    let cos_lat = lat.to_radians().cos().abs().max(1e-6);
    let metres = s_px * M_PER_PX_Z0 * cos_lat / f64::from(1u32 << zoom.min(31));
    // Metres → degrees of longitude at this latitude.
    metres / (111_320.0 * cos_lat)
}

/// Assign each candidate exactly one home zoom in `[min_zoom, max_zoom]`.
///
/// Coarse-to-fine: at each zoom, a candidate claims its voxel if that voxel is
/// still unclaimed. Every candidate is claimed by `max_zoom` at the latest, so
/// the partition is total — no feature is ever dropped (this is an assignment,
/// not a filter, and the no-thinning rule is preserved by construction).
///
/// Returns `id → home zoom`, deterministic for a given input set.
pub fn assign_home_zooms(
    candidates: &[HomeZoomCandidate],
    min_zoom: u8,
    max_zoom: u8,
    s_px: f64,
    temporal_bucket_ms: u64,
) -> BTreeMap<u64, u8> {
    let mut home: BTreeMap<u64, u8> = BTreeMap::new();
    if candidates.is_empty() || max_zoom < min_zoom {
        return home;
    }

    // Total order: importance desc, then id asc. `total_cmp` keeps NaN from
    // making the sort non-deterministic.
    let mut order: Vec<&HomeZoomCandidate> = candidates.iter().collect();
    order.sort_by(|a, b| {
        b.importance
            .total_cmp(&a.importance)
            .then_with(|| a.id.cmp(&b.id))
    });

    let bucket = temporal_bucket_ms.max(1);
    for zoom in min_zoom..=max_zoom {
        // Voxel keys claimed at THIS zoom. A claim is per-zoom: a voxel
        // occupied at z6 does not block the same cell at z7.
        let mut claimed: BTreeMap<(i64, i64, u64), ()> = BTreeMap::new();
        for c in &order {
            if home.contains_key(&c.id) {
                continue;
            }
            let pitch = voxel_pitch_deg(s_px, zoom, c.lat);
            if !(pitch > 0.0) || !c.lon.is_finite() || !c.lat.is_finite() {
                // Degenerate geometry claims at the deepest zoom rather than
                // vanishing — fail-open, never fail-drop.
                if zoom == max_zoom {
                    home.insert(c.id, max_zoom);
                }
                continue;
            }
            // Origin pinned at (-180, -90) so the grid nests across zooms
            // rather than drifting — the `mrms_refloor.py` lesson.
            let vx = ((c.lon + 180.0) / pitch).floor() as i64;
            let vy = ((c.lat + 90.0) / pitch).floor() as i64;
            // ⚠️ REGISTER (hard): the bucket is part of the key. Space-only
            // grids over animated data showed 13 % visible-bucket coverage.
            let vt = c.timestamp / bucket;
            if claimed.insert((vx, vy, vt), ()).is_none() {
                home.insert(c.id, zoom);
            }
        }
    }

    // Totality: anything still unclaimed (only reachable via degenerate
    // coordinates) lands at max_zoom. No feature is ever dropped.
    for c in candidates {
        home.entry(c.id).or_insert(max_zoom);
    }
    home
}

/// Property name the assignment writes, and `min_zoom_field` then reads.
///
/// Synthesized rather than user-supplied, which is why `--additive-lod` refuses
/// to run alongside an explicit `--min-zoom-field`.
pub const HOME_ZOOM_PROPERTY: &str = "__stt_home_zoom";

/// Write each feature's assigned home zoom onto it as [`HOME_ZOOM_PROPERTY`],
/// so the existing band mechanism picks it up unchanged.
///
/// Indices must match the `id`s handed to [`assign_home_zooms`].
pub fn assignment_as_bands(home: &BTreeMap<u64, u8>, count: usize) -> Vec<u8> {
    (0..count)
        .map(|i| home.get(&(i as u64)).copied().unwrap_or(0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(id: u64, lon: f64, lat: f64, t: u64, imp: f64) -> HomeZoomCandidate {
        HomeZoomCandidate {
            id,
            lon,
            lat,
            timestamp: t,
            importance: imp,
        }
    }

    /// The partition is TOTAL: every feature gets exactly one home zoom, so an
    /// additive build stores each feature once rather than |Z| times.
    #[test]
    fn every_feature_is_claimed_exactly_once() {
        let cands: Vec<_> = (0..500)
            .map(|i| {
                c(
                    i,
                    -122.0 + (i as f64) * 0.001,
                    37.0 + (i as f64) * 0.001,
                    (i as u64) * 1_000,
                    0.0,
                )
            })
            .collect();
        let home = assign_home_zooms(&cands, 0, 10, DEFAULT_HOME_ZOOM_PX, 3_600_000);
        assert_eq!(home.len(), cands.len(), "assignment must be total");
        for c in &cands {
            let z = home[&c.id];
            assert!((0..=10).contains(&z), "id {} got z{z}", c.id);
        }
    }

    /// Deterministic: identical inputs in any order produce an identical
    /// assignment. Pack names are content-addressed, so this is existential.
    #[test]
    fn the_assignment_is_order_independent() {
        let mut cands: Vec<_> = (0..200)
            .map(|i| {
                c(
                    i,
                    -100.0 + (i % 13) as f64 * 0.01,
                    40.0 + (i % 7) as f64 * 0.01,
                    (i as u64 % 5) * 3_600_000,
                    (i % 3) as f64,
                )
            })
            .collect();
        let a = assign_home_zooms(&cands, 2, 9, DEFAULT_HOME_ZOOM_PX, 3_600_000);
        cands.reverse();
        let b = assign_home_zooms(&cands, 2, 9, DEFAULT_HOME_ZOOM_PX, 3_600_000);
        assert_eq!(a, b);
    }

    /// Importance decides WHO claims a contested voxel; the id breaks ties.
    #[test]
    fn importance_wins_a_contested_voxel_and_the_id_breaks_ties() {
        // Two features in the same voxel at every zoom (identical position).
        let cands = vec![c(7, -100.0, 40.0, 0, 1.0), c(3, -100.0, 40.0, 0, 9.0)];
        let home = assign_home_zooms(&cands, 0, 4, DEFAULT_HOME_ZOOM_PX, 3_600_000);
        // The important one claims the coarsest zoom.
        assert_eq!(home[&3], 0);
        assert!(home[&7] > 0, "the loser is pushed finer, never dropped");

        // Equal importance ⇒ lower id wins.
        let tie = vec![c(7, -100.0, 40.0, 0, 1.0), c(3, -100.0, 40.0, 0, 1.0)];
        let home = assign_home_zooms(&tie, 0, 4, DEFAULT_HOME_ZOOM_PX, 3_600_000);
        assert_eq!(home[&3], 0);
    }

    /// ⚠️ REGISTER (hard): the voxel key includes the temporal bucket. Two
    /// features at the same place in DIFFERENT buckets must not contend — a
    /// space-only grid is what took visible-bucket coverage to 13 %.
    #[test]
    fn the_voxel_key_includes_the_temporal_bucket() {
        let bucket = 3_600_000;
        let same_place_different_buckets = vec![
            c(1, -100.0, 40.0, 0, 0.0),
            c(2, -100.0, 40.0, bucket * 5, 0.0),
        ];
        let home = assign_home_zooms(
            &same_place_different_buckets,
            0,
            6,
            DEFAULT_HOME_ZOOM_PX,
            bucket,
        );
        assert_eq!(home[&1], 0, "{home:?}");
        assert_eq!(
            home[&2], 0,
            "different buckets must NOT contend for one voxel: {home:?}"
        );

        // Same place AND same bucket DO contend.
        let same_bucket = vec![c(1, -100.0, 40.0, 0, 0.0), c(2, -100.0, 40.0, 10, 0.0)];
        let home = assign_home_zooms(&same_bucket, 0, 6, DEFAULT_HOME_ZOOM_PX, bucket);
        assert_ne!(
            home[&1], home[&2],
            "same voxel must push one finer: {home:?}"
        );
    }

    /// Voxel pitch shrinks with zoom and is latitude-corrected.
    #[test]
    fn voxel_pitch_shrinks_with_zoom() {
        for z in 0u8..14 {
            assert!(
                voxel_pitch_deg(DEFAULT_HOME_ZOOM_PX, z, 40.0)
                    > voxel_pitch_deg(DEFAULT_HOME_ZOOM_PX, z + 1, 40.0)
            );
        }
        assert!(voxel_pitch_deg(DEFAULT_HOME_ZOOM_PX, 8, 0.0).is_finite());
        assert!(voxel_pitch_deg(DEFAULT_HOME_ZOOM_PX, 8, 89.9).is_finite());
    }

    /// Degenerate coordinates land at max_zoom rather than vanishing —
    /// fail-open, never fail-drop.
    #[test]
    fn degenerate_geometry_is_never_dropped() {
        let cands = vec![
            c(1, f64::NAN, 40.0, 0, 0.0),
            c(2, -100.0, f64::INFINITY, 0, 0.0),
        ];
        let home = assign_home_zooms(&cands, 0, 5, DEFAULT_HOME_ZOOM_PX, 3_600_000);
        assert_eq!(home.len(), 2);
        assert_eq!(home[&1], 5);
        assert_eq!(home[&2], 5);
    }
}
