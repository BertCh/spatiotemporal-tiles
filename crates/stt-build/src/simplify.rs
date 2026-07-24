//! Line simplification for lower zoom levels
//!
//! Applies Douglas-Peucker or Visvalingam-Whyatt simplification to reduce
//! vertex count at lower zoom levels, improving both memory usage and
//! rendering performance.

use geo::{Coord, LineString, Polygon, SimplifyVw, SimplifyVwPreserve};

// ---------------------------------------------------------------------------
// Latitude-corrected METRIC simplification tolerance (opt-in)
// ---------------------------------------------------------------------------
//
// The legacy path ([`calculate_epsilon`]) uses a fixed per-zoom tolerance in
// DEGREES. A degree of longitude shrinks with latitude (× cos φ), so a fixed
// degree tolerance means a different GROUND distance at every latitude — high-
// latitude data is generalised inconsistently vs the equator (~2× different in
// E–W terms at 60°). The metric path fixes this the correct (anisotropic) way:
// it scales the longitude axis by cos(latitude) so the working space is locally
// isometric in metres on BOTH axes, runs the SAME Visvalingam / TD-TR selector
// there against a metre-derived tolerance, then unscales longitude exactly (all
// three simplifiers only ever SELECT a subset of the input vertices, so the
// scale round-trips losslessly). At the equator cos φ = 1 and the transform is
// the identity, but the metric tolerance table differs from the degree table,
// so the two modes are NOT byte-identical — hence metric mode is opt-in.

/// Metres per degree of latitude (WGS84 mean) — matches the world-grid affine
/// used by coordinate quantisation (`arrow_tile::M_PER_DEG_LAT`). In the
/// cos(latitude)-scaled working space one unit on EITHER axis is this many
/// metres, at every latitude.
const M_PER_DEG_LAT: f64 = 111_320.0;

/// Web-Mercator ground resolution at zoom 0 for 256-px tiles: `2·π·R / 256`
/// ≈ 156 543 m/px (equator). Ground metres per pixel at zoom `z` is this / 2^z.
const GROUND_RES_Z0_M: f64 = 156_543.033_928_041_1;

/// Simplification tolerance expressed in pixels of the tile's own zoom. 1.0 px
/// drops only sub-pixel-invisible detail — the principled default for a render
/// format (and it lands near the resolutions the legacy table's comments
/// *intended*, which the degree-as-area values never actually achieved).
const TOLERANCE_PIXELS: f64 = 1.0;

/// Floor on cos(latitude) so the longitude scale can't blow up toward the poles
/// (cos 87° ≈ 0.05). Beyond this the anisotropy correction saturates rather
/// than dividing by ~0.
const MIN_COS_LAT: f64 = 0.05;

/// cos of the representative latitude, floored — the factor the longitude axis
/// is multiplied by to make it metric-isometric with the latitude axis.
fn lat_scale(mean_lat_deg: f64) -> f64 {
    mean_lat_deg.to_radians().cos().abs().max(MIN_COS_LAT)
}

/// Ground tolerance in metres at a zoom (latitude-independent — the cos φ
/// scaling of the coordinates is what carries the latitude correction).
fn metric_tol_m(zoom: u8) -> f64 {
    TOLERANCE_PIXELS * GROUND_RES_Z0_M / 2f64.powi(i32::from(zoom))
}

/// Linear tolerance in cos-latitude-scaled degree units (for TD-TR's SED).
fn metric_linear_epsilon(zoom: u8) -> f64 {
    metric_tol_m(zoom) / M_PER_DEG_LAT
}

/// Visvalingam AREA threshold in scaled degree² units — the square of the
/// linear tolerance, so a triangle whose characteristic size is below the
/// ground tolerance is removed.
fn metric_area_epsilon(zoom: u8) -> f64 {
    let e = metric_linear_epsilon(zoom);
    e * e
}

/// Mean latitude of a 3D coordinate list (representative point for the cos φ
/// correction). Empty → equator (scale 1.0).
fn mean_lat_coords(coords: &[(f64, f64, f64)]) -> f64 {
    if coords.is_empty() {
        return 0.0;
    }
    coords.iter().map(|(_, y, _)| *y).sum::<f64>() / coords.len() as f64
}

/// Mean latitude of a GeoJSON ring (`[[lon, lat], …]`). Empty → equator.
fn mean_lat_ring(ring: &[Vec<f64>]) -> f64 {
    let ys: Vec<f64> = ring.iter().filter(|c| c.len() >= 2).map(|c| c[1]).collect();
    if ys.is_empty() {
        return 0.0;
    }
    ys.iter().sum::<f64>() / ys.len() as f64
}

/// Simplify coordinates for a given zoom level
///
/// Returns simplified coordinates with altitudes preserved through interpolation.
/// At high zoom levels (15+), no simplification is applied.
pub fn simplify_for_zoom(
    coords: &[(f64, f64, f64)],
    zoom: u8,
    simplify_max_zoom: u8,
) -> Vec<(f64, f64, f64)> {
    simplify_for_zoom_with(coords, zoom, simplify_max_zoom, false)
}

/// [`simplify_for_zoom`] with an explicit tolerance mode. `metric = false` is
/// byte-identical to the legacy degree path; `metric = true` uses the
/// latitude-corrected metric tolerance (see the module header).
pub fn simplify_for_zoom_with(
    coords: &[(f64, f64, f64)],
    zoom: u8,
    simplify_max_zoom: u8,
    metric: bool,
) -> Vec<(f64, f64, f64)> {
    // Don't simplify if zoom is above threshold or too few points
    if zoom > simplify_max_zoom || coords.len() < 3 {
        return coords.to_vec();
    }

    // epsilon is a Visvalingam AREA threshold. `xs` scales the longitude axis so
    // the working space is metric-isometric (metric mode); `xs == 1.0` in degree
    // mode is an exact f64 no-op, so that path stays byte-identical.
    let (epsilon, xs) = if metric {
        (
            metric_area_epsilon(zoom),
            lat_scale(mean_lat_coords(coords)),
        )
    } else {
        (calculate_epsilon(zoom), 1.0)
    };

    if epsilon <= 0.0 {
        return coords.to_vec();
    }

    // Convert to geo LineString (2D), longitude scaled into the working space.
    let line: LineString<f64> = coords
        .iter()
        .map(|(x, y, _)| geo::Coord { x: *x * xs, y: *y })
        .collect();

    // Apply Visvalingam-Whyatt simplification (better for preserving shape)
    let simplified = line.simplify_vw(&epsilon);

    // If simplification didn't help or made it too short, return original
    if simplified.0.len() < 2 || simplified.0.len() >= coords.len() {
        return coords.to_vec();
    }

    // Map simplified coords back (unscaling longitude) with interpolated altitudes
    let inv = 1.0 / xs;
    let mut result = Vec::with_capacity(simplified.0.len());

    for coord in simplified.0.iter() {
        let lon = coord.x * inv;
        let lat = coord.y;
        // Find the closest original point to interpolate altitude
        let alt = interpolate_altitude(coords, lon, lat);
        result.push((lon, lat, alt));
    }

    result
}

/// Signed shoelace area of a closed GeoJSON ring (`[[lon, lat], …]`);
/// positive = CCW. Used only to guard polygon simplification (winding sign +
/// non-degeneracy).
fn ring_signed_area(ring: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for w in ring.windows(2) {
        if w[0].len() >= 2 && w[1].len() >= 2 {
            sum += w[0][0] * w[1][1] - w[1][0] * w[0][1];
        }
    }
    sum / 2.0
}

/// Topology-preserving per-tile polygon simplification for lower zoom levels —
/// the fill-geometry parallel of [`simplify_for_zoom`] for lines.
///
/// `rings` is ONE polygon in GeoJSON shape (`[ring][vertex][lon, lat]`, exterior
/// first, holes after, each ring closed). Runs the topology-PRESERVING
/// Visvalingam–Whyatt variant ([`geo::SimplifyVwPreserve`], which drops a vertex
/// only when doing so does NOT create a ring self-intersection — it shares one
/// R*-tree across the exterior + all holes, so shell/hole intersections are
/// detected too) at the SAME per-zoom epsilon the line simplifier uses, then
/// hardens the result:
///
/// * GATE — simplify ONLY for `zoom < simplify_max_zoom`; the max-tiled-zoom
///   tier is returned verbatim (lossless / byte-identical). This is the strict
///   `<` (lines use `<=`) on purpose: A1's watertight polygon seams require
///   adjacent tiles to emit BIT-IDENTICAL edge vertices, so the fill max tier
///   must stay bit-exact and un-thinned.
/// * Never drops a ring; never collapses one below a valid closed ring (>= 4
///   coords ⇒ >= 3 distinct vertices). `SimplifyVwPreserve` itself will not take
///   a polygon ring below 4 coords, and we re-check. A ring whose simplification
///   would be invalid (too few coords, unclosed, ~zero area, or flipped winding)
///   keeps its un-simplified form; if the EXTERIOR would be invalid the WHOLE
///   polygon is returned un-simplified (a hole without a valid shell is
///   meaningless).
/// * Rings stay closed and winding is preserved (exterior CCW, holes CW), so the
///   output is a drop-in replacement for the input rings.
///
/// MUST run AFTER the antimeridian split AND AFTER the per-tile Sutherland–
/// Hodgman clip (see the note on `clip::split_polygon_at_antimeridian`):
/// simplifying before either could move a seam vertex off ±180 or reintroduce a
/// > 180° edge and break the watertight cut.
pub fn simplify_polygon_rings_for_zoom(
    rings: &[Vec<Vec<f64>>],
    zoom: u8,
    simplify_max_zoom: u8,
) -> Vec<Vec<Vec<f64>>> {
    simplify_polygon_rings_for_zoom_with(rings, zoom, simplify_max_zoom, false)
}

/// [`simplify_polygon_rings_for_zoom`] with an explicit tolerance mode.
/// `metric = false` is byte-identical to the legacy degree path; `metric = true`
/// applies the latitude-corrected metric tolerance (longitude scaled by cos φ
/// before the topology-preserving simplify, unscaled after — a positive uniform
/// scale preserves ring winding and non-self-intersection, so the topology
/// guarantee carries back).
pub fn simplify_polygon_rings_for_zoom_with(
    rings: &[Vec<Vec<f64>>],
    zoom: u8,
    simplify_max_zoom: u8,
    metric: bool,
) -> Vec<Vec<Vec<f64>>> {
    // Max-tiled-zoom tier stays lossless (see GATE above).
    if zoom >= simplify_max_zoom || rings.is_empty() {
        return rings.to_vec();
    }
    let (epsilon, xs) = if metric {
        (
            metric_area_epsilon(zoom),
            lat_scale(mean_lat_ring(&rings[0])),
        )
    } else {
        (calculate_epsilon(zoom), 1.0)
    };
    if epsilon <= 0.0 {
        return rings.to_vec();
    }
    let inv = 1.0 / xs;

    // GeoJSON rings → 2D geo Polygon, longitude scaled into the working space.
    // Polygon geometry columns are 2D by the time clipping has run (Sutherland–
    // Hodgman drops altitude), so altitude handling is unnecessary here.
    let to_line = |ring: &[Vec<f64>]| -> LineString<f64> {
        ring.iter()
            .filter(|c| c.len() >= 2)
            .map(|c| Coord {
                x: c[0] * xs,
                y: c[1],
            })
            .collect()
    };
    let poly = Polygon::new(
        to_line(&rings[0]),
        rings[1..].iter().map(|r| to_line(r)).collect(),
    );
    let simplified = poly.simplify_vw_preserve(&epsilon);

    // Unscale longitude on the way back out.
    let ls_to_ring = |ls: &LineString<f64>| -> Vec<Vec<f64>> {
        ls.coords().map(|c| vec![c.x * inv, c.y]).collect()
    };
    // A candidate ring is acceptable iff it is a valid closed ring (>= 4 coords),
    // keeps a non-degenerate area, and preserves winding sign vs its original.
    let accept = |orig: &[Vec<f64>], cand: &[Vec<f64>]| -> bool {
        if cand.len() < 4 || cand.first() != cand.last() {
            return false;
        }
        let a = ring_signed_area(cand);
        a.abs() > 0.0 && (a > 0.0) == (ring_signed_area(orig) > 0.0)
    };

    // Exterior: an invalid simplification keeps the ENTIRE polygon un-simplified.
    let ext_cand = ls_to_ring(simplified.exterior());
    if !accept(&rings[0], &ext_cand) {
        return rings.to_vec();
    }
    let mut out: Vec<Vec<Vec<f64>>> = Vec::with_capacity(rings.len());
    out.push(ext_cand);

    // Holes: keep each simplified hole when valid, else fall back to its
    // original. `simplify_vw_preserve` emits one interior per input interior in
    // order, so they align 1:1; a defensive `.get` guards the count regardless.
    let interiors = simplified.interiors();
    for (i, orig_hole) in rings[1..].iter().enumerate() {
        match interiors.get(i).map(&ls_to_ring) {
            Some(c) if accept(orig_hole, &c) => out.push(c),
            _ => out.push(orig_hole.clone()),
        }
    }
    out
}

/// Time-aware trajectory simplification (Top-Down Time-Ratio, TD-TR) using
/// **Synchronized Euclidean Distance**.
///
/// Unlike Douglas–Peucker / Visvalingam (which preserve only *spatial* shape
/// and so can place a moving object in the right place at the wrong time when
/// zoomed out), TD-TR measures each candidate vertex against the point on the
/// anchor chord at that vertex's *own* time ratio. It therefore keeps the
/// vertices that carry the motion's timing — pauses, speed changes — and drops
/// only those redundant in both space and time. Returns a subset of the input
/// vertices with their original times/values preserved (no re-interpolation).
pub fn simplify_td_tr_for_zoom(
    coords: &[(f64, f64, f64)],
    times: &[u64],
    values: &[f32],
    zoom: u8,
    simplify_max_zoom: u8,
) -> (Vec<(f64, f64, f64)>, Vec<u64>, Vec<f32>) {
    simplify_td_tr_for_zoom_with(coords, times, values, zoom, simplify_max_zoom, false)
}

/// [`simplify_td_tr_for_zoom`] with an explicit tolerance mode. `metric = false`
/// is byte-identical to the legacy degree path; `metric = true` scales the
/// longitude component of the Synchronized-Euclidean-Distance by cos φ and uses
/// a metre-derived linear tolerance, so the kept-vertex decision is latitude-
/// uniform in ground terms.
pub fn simplify_td_tr_for_zoom_with(
    coords: &[(f64, f64, f64)],
    times: &[u64],
    values: &[f32],
    zoom: u8,
    simplify_max_zoom: u8,
    metric: bool,
) -> (Vec<(f64, f64, f64)>, Vec<u64>, Vec<f32>) {
    if zoom > simplify_max_zoom
        || coords.len() < 3
        || times.len() != coords.len()
        || values.len() != coords.len()
    {
        return (coords.to_vec(), times.to_vec(), values.to_vec());
    }
    let (epsilon, x_scale) = if metric {
        (
            metric_linear_epsilon(zoom),
            lat_scale(mean_lat_coords(coords)),
        )
    } else {
        (calculate_epsilon(zoom), 1.0)
    };
    if epsilon <= 0.0 {
        return (coords.to_vec(), times.to_vec(), values.to_vec());
    }
    simplify_td_tr_scaled(coords, times, values, epsilon, x_scale)
}

/// TD-TR core: keep the endpoints, then recursively keep the max-SED vertex
/// while it exceeds `epsilon` (in degrees, matching the spatial simplifier).
#[cfg(test)]
fn simplify_td_tr(
    coords: &[(f64, f64, f64)],
    times: &[u64],
    values: &[f32],
    epsilon: f64,
) -> (Vec<(f64, f64, f64)>, Vec<u64>, Vec<f32>) {
    simplify_td_tr_scaled(coords, times, values, epsilon, 1.0)
}

/// TD-TR core with an explicit longitude scale for the metric mode. `x_scale`
/// multiplies the longitude component of the SED so distances are measured in
/// the cos-latitude-scaled (metric-isometric) space; `x_scale == 1.0` is the
/// legacy degree behaviour.
fn simplify_td_tr_scaled(
    coords: &[(f64, f64, f64)],
    times: &[u64],
    values: &[f32],
    epsilon: f64,
    x_scale: f64,
) -> (Vec<(f64, f64, f64)>, Vec<u64>, Vec<f32>) {
    let n = coords.len();
    if n <= 2 || epsilon <= 0.0 || times.len() != n || values.len() != n {
        return (coords.to_vec(), times.to_vec(), values.to_vec());
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    td_tr_recurse(coords, times, 0, n - 1, epsilon, x_scale, &mut keep);

    let mut sc = Vec::new();
    let mut st = Vec::new();
    let mut sv = Vec::new();
    for i in 0..n {
        if keep[i] {
            sc.push(coords[i]);
            st.push(times[i]);
            sv.push(values[i]);
        }
    }
    (sc, st, sv)
}

fn td_tr_recurse(
    coords: &[(f64, f64, f64)],
    times: &[u64],
    start: usize,
    end: usize,
    epsilon: f64,
    x_scale: f64,
    keep: &mut [bool],
) {
    if end <= start + 1 {
        return;
    }
    let t_start = times[start] as f64;
    let t_end = times[end] as f64;
    let dt = t_end - t_start;
    let (xs, ys, _) = coords[start];
    let (xe, ye, _) = coords[end];

    let mut max_sed = 0.0f64;
    let mut max_idx = start;
    for i in (start + 1)..end {
        // Synchronized reference: where a constant-velocity mover would be at
        // this vertex's own time.
        let ratio = if dt > 0.0 {
            (times[i] as f64 - t_start) / dt
        } else {
            0.0
        };
        let sync_x = xs + ratio * (xe - xs);
        let sync_y = ys + ratio * (ye - ys);
        let (xi, yi, _) = coords[i];
        // Longitude component scaled into the metric-isometric space
        // (`x_scale == 1.0` = legacy degree behaviour).
        let dx = (xi - sync_x) * x_scale;
        let sed = (dx.powi(2) + (yi - sync_y).powi(2)).sqrt();
        if sed > max_sed {
            max_sed = sed;
            max_idx = i;
        }
    }

    if max_sed > epsilon {
        keep[max_idx] = true;
        td_tr_recurse(coords, times, start, max_idx, epsilon, x_scale, keep);
        td_tr_recurse(coords, times, max_idx, end, epsilon, x_scale, keep);
    }
}

/// Calculate simplification epsilon for a zoom level
fn calculate_epsilon(zoom: u8) -> f64 {
    // Epsilon in degrees - corresponds roughly to pixel resolution
    // tile_size = 256 pixels, world = 360 degrees
    // At zoom z: degrees_per_pixel = 360 / (256 * 2^z)
    match zoom {
        0..=6 => 0.01,     // ~1km resolution
        7..=9 => 0.002,    // ~200m resolution
        10..=11 => 0.0008, // ~80m resolution
        12..=13 => 0.0003, // ~30m resolution
        14 => 0.0001,      // ~10m resolution
        _ => 0.0,          // No simplification
    }
}

/// Interpolate altitude at a given lon/lat from original coordinates
fn interpolate_altitude(coords: &[(f64, f64, f64)], lon: f64, lat: f64) -> f64 {
    // Find the segment containing this point
    let mut best_alt = 0.0;
    let mut best_dist = f64::MAX;

    for window in coords.windows(2) {
        let (x1, y1, alt1) = window[0];
        let (x2, y2, alt2) = window[1];

        // Project point onto segment
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;

        let t = if len_sq > 0.0 {
            ((lon - x1) * dx + (lat - y1) * dy) / len_sq
        } else {
            0.0
        };

        let t = t.clamp(0.0, 1.0);

        // Point on segment
        let px = x1 + t * dx;
        let py = y1 + t * dy;

        // Distance from projected point
        let dist = (lon - px).powi(2) + (lat - py).powi(2);

        if dist < best_dist {
            best_dist = dist;
            best_alt = alt1 + t * (alt2 - alt1);
        }
    }

    // Also check individual points (for exact matches)
    for (x, y, alt) in coords {
        let dist = (lon - x).powi(2) + (lat - y).powi(2);
        if dist < best_dist {
            best_dist = dist;
            best_alt = *alt;
        }
    }

    best_alt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_simplification_high_zoom() {
        let coords = vec![
            (-122.4, 37.7, 0.0),
            (-122.41, 37.71, 10.0),
            (-122.42, 37.72, 20.0),
        ];

        // At zoom 16, should return original
        let result = simplify_for_zoom(&coords, 16, 14);
        assert_eq!(result.len(), coords.len());
    }

    #[test]
    fn test_simplification_low_zoom() {
        // Create a line with many points that can be simplified
        let mut coords = Vec::new();
        for i in 0..100 {
            let t = i as f64 / 100.0;
            // Slightly noisy line from SF to Oakland
            let noise = (i as f64 * 0.1).sin() * 0.0001;
            coords.push((-122.4 + t * 0.15 + noise, 37.7 + t * 0.1 + noise, t * 100.0));
        }

        // At zoom 8, should simplify significantly
        let result = simplify_for_zoom(&coords, 8, 14);
        assert!(result.len() < coords.len(), "Should have fewer points");
        assert!(result.len() >= 2, "Should have at least 2 points");
    }

    #[test]
    fn test_preserves_altitude() {
        let coords = vec![(0.0, 0.0, 100.0), (0.5, 0.5, 200.0), (1.0, 1.0, 300.0)];

        let result = simplify_for_zoom(&coords, 5, 14);

        // Check that altitudes are reasonable
        for (_, _, alt) in &result {
            assert!(
                *alt >= 100.0 && *alt <= 300.0,
                "Altitude should be in range"
            );
        }
    }

    #[test]
    fn td_tr_keeps_temporally_displaced_collinear_vertex() {
        // Straight line in space, but the object PAUSES near the midpoint
        // (reaches x=1 only at t=9 of 10). Plain spatial DP/VW drops the
        // collinear midpoint; TD-TR keeps it because the time-synced reference
        // (at ratio 0.9 → x=1.8) is far from where the object actually is (x=1).
        let coords = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)];
        let times = vec![0u64, 9000, 10000];
        let values = vec![f32::NAN; 3];
        let (sc, st, _) = simplify_td_tr(&coords, &times, &values, 0.1);
        assert_eq!(
            sc.len(),
            3,
            "TD-TR should keep the temporally-displaced midpoint"
        );
        assert_eq!(st, times, "kept vertices preserve their real times");
    }

    #[test]
    fn td_tr_drops_collinear_uniform_time_vertex() {
        // Same straight line, UNIFORM time → the midpoint lies on the sync line
        // (SED 0) and is correctly dropped.
        let coords = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)];
        let times = vec![0u64, 5000, 10000];
        let values = vec![f32::NAN; 3];
        let (sc, _, _) = simplify_td_tr(&coords, &times, &values, 0.1);
        assert_eq!(
            sc.len(),
            2,
            "uniform-time collinear midpoint should be dropped"
        );
    }

    #[test]
    fn td_tr_for_zoom_is_noop_above_max_zoom() {
        let coords = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)];
        let times = vec![0u64, 9000, 10000];
        let values = vec![f32::NAN; 3];
        let (sc, _, _) = simplify_td_tr_for_zoom(&coords, &times, &values, 16, 14);
        assert_eq!(sc.len(), 3);
    }

    #[test]
    fn test_epsilon_decreases_with_zoom() {
        let eps10 = calculate_epsilon(10);
        let eps12 = calculate_epsilon(12);
        let eps14 = calculate_epsilon(14);

        assert!(eps10 > eps12, "Lower zoom should have larger epsilon");
        assert!(eps12 > eps14, "Lower zoom should have larger epsilon");
    }

    // ------------------------------------------------------------------
    // Polygon simplification (A2)
    // ------------------------------------------------------------------

    /// A closed circle ring sampled `n` times, CCW (`ccw = true`, positive area)
    /// or CW (holes). First vertex repeated as the GeoJSON closing duplicate.
    fn circle_ring(cx: f64, cy: f64, r: f64, n: usize, ccw: bool) -> Vec<Vec<f64>> {
        use std::f64::consts::PI;
        let mut ring: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        for i in 0..n {
            let frac = i as f64 / n as f64;
            let theta = if ccw {
                2.0 * PI * frac
            } else {
                -2.0 * PI * frac
            };
            ring.push(vec![cx + r * theta.cos(), cy + r * theta.sin()]);
        }
        ring.push(ring[0].clone());
        ring
    }

    #[test]
    fn polygon_max_zoom_tier_is_byte_identical() {
        // At/above simplify_max_zoom the rings come back verbatim — the
        // max-tiled-zoom tier stays lossless (NO THINNING), watertight-seam safe.
        let rings = vec![
            circle_ring(0.0, 0.0, 0.5, 128, true),
            circle_ring(0.0, 0.0, 0.2, 96, false),
        ];
        // zoom == simplify_max_zoom (the strict `<` gate excludes it).
        assert_eq!(simplify_polygon_rings_for_zoom(&rings, 14, 14), rings);
        // zoom above the tier is also untouched.
        assert_eq!(simplify_polygon_rings_for_zoom(&rings, 18, 14), rings);
    }

    #[test]
    fn polygon_simplify_reduces_vertices_at_low_zoom() {
        // A dense 400-vertex ring collapses hard at zoom 8 (epsilon 2e-3) while
        // staying a valid closed ring.
        let rings = vec![circle_ring(0.0, 0.0, 0.5, 400, true)];
        let orig: usize = rings.iter().map(|r| r.len()).sum();
        let out = simplify_polygon_rings_for_zoom(&rings, 8, 14);
        let after: usize = out.iter().map(|r| r.len()).sum();
        assert!(
            after < orig,
            "expected vertex reduction at z8: {after} !< {orig}"
        );
        assert!(
            out[0].len() >= 4,
            "ring collapsed below a triangle: {:?}",
            out[0]
        );
        assert_eq!(out[0].first(), out[0].last(), "ring must stay closed");
    }

    #[test]
    fn polygon_simplify_preserves_topology_and_holes() {
        // Dense CCW exterior + dense CW hole. After simplification: same ring
        // count (hole retained), each ring closed with >= 4 coords, exterior CCW
        // (positive area), hole CW (negative area), and overall vertex reduction.
        let rings = vec![
            circle_ring(0.0, 0.0, 0.6, 300, true),   // exterior, CCW
            circle_ring(0.0, 0.0, 0.25, 240, false), // hole, CW
        ];
        assert!(
            ring_signed_area(&rings[0]) > 0.0,
            "test setup: exterior CCW"
        );
        assert!(ring_signed_area(&rings[1]) < 0.0, "test setup: hole CW");
        let orig: usize = rings.iter().map(|r| r.len()).sum();

        let out = simplify_polygon_rings_for_zoom(&rings, 8, 14);
        assert_eq!(out.len(), 2, "hole must be retained, never dropped");
        let after: usize = out.iter().map(|r| r.len()).sum();
        assert!(after < orig, "expected vertex reduction: {after} !< {orig}");

        for (i, ring) in out.iter().enumerate() {
            assert!(
                ring.len() >= 4,
                "ring {i} collapsed below a triangle: {ring:?}"
            );
            assert_eq!(ring.first(), ring.last(), "ring {i} not closed");
            let area = ring_signed_area(ring);
            if i == 0 {
                assert!(area > 0.0, "exterior winding flipped (area {area})");
            } else {
                assert!(area < 0.0, "hole winding flipped (area {area})");
            }
        }
    }

    // ------------------------------------------------------------------
    // Latitude-corrected metric tolerance (T1-①)
    // ------------------------------------------------------------------

    #[test]
    fn lat_scale_tracks_cos_latitude_with_pole_floor() {
        assert!((lat_scale(0.0) - 1.0).abs() < 1e-12, "equator ⇒ 1.0");
        assert!((lat_scale(60.0) - 0.5).abs() < 1e-3, "60° ⇒ cos 60° = 0.5");
        assert!((lat_scale(-60.0) - 0.5).abs() < 1e-3, "sign-independent");
        assert!(lat_scale(89.9) >= MIN_COS_LAT, "floored near the pole");
    }

    /// A trajectory that moves due north at constant speed with a fixed
    /// GROUND-metre east–west wiggle, placed at latitude `lat0`. The lon offset
    /// is divided by cos(lat0) so the wiggle is the SAME ground size at every
    /// latitude — the exact case a degree tolerance mishandles.
    fn ew_wiggle_ground(
        lat0: f64,
        n: usize,
        amp_m: f64,
        north_step_m: f64,
    ) -> (Vec<(f64, f64, f64)>, Vec<u64>, Vec<f32>) {
        let cphi = lat0.to_radians().cos();
        let mut coords = Vec::with_capacity(n);
        let mut times = Vec::with_capacity(n);
        for i in 0..n {
            let lat = lat0 + (i as f64 * north_step_m) / M_PER_DEG_LAT;
            let east_m = if i % 2 == 1 { amp_m } else { 0.0 };
            let lon = east_m / (M_PER_DEG_LAT * cphi);
            coords.push((lon, lat, 0.0));
            times.push(i as u64 * 1000);
        }
        let values = vec![f32::NAN; n];
        (coords, times, values)
    }

    #[test]
    fn metric_td_tr_is_latitude_uniform_but_degrees_is_not() {
        // Same ground shape (30 m E–W wiggle) at the equator and at 60°N, at a
        // zoom whose metric tolerance (~38 m at z12) straddles the wiggle.
        let n = 41;
        let step = 200.0; // north metres between vertices
        let amp = 30.0; // east–west ground amplitude
        let z = 12u8;

        let (c0, t0, v0) = ew_wiggle_ground(0.0, n, amp, step);
        let (c60, t60, v60) = ew_wiggle_ground(60.0, n, amp, step);

        // METRIC: the cos φ scaling normalises the wiggle to its true ground
        // size, so the kept-vertex decision is identical at both latitudes.
        let m0 = simplify_td_tr_for_zoom_with(&c0, &t0, &v0, z, 14, true)
            .0
            .len();
        let m60 = simplify_td_tr_for_zoom_with(&c60, &t60, &v60, z, 14, true)
            .0
            .len();
        assert_eq!(
            m0, m60,
            "metric tolerance must treat the same ground shape identically at \
             every latitude (equator {m0} vs 60°N {m60})"
        );

        // DEGREES: a fixed degree tolerance sees the 60°N longitude wiggle as
        // ~2× larger, so it keeps strictly more vertices there — the veracity
        // inconsistency the metric mode fixes.
        let d0 = simplify_td_tr_for_zoom_with(&c0, &t0, &v0, z, 14, false)
            .0
            .len();
        let d60 = simplify_td_tr_for_zoom_with(&c60, &t60, &v60, z, 14, false)
            .0
            .len();
        assert!(
            d60 > d0,
            "degree tolerance is latitude-sensitive: expected more kept at \
             60°N than equator, got {d60} vs {d0}"
        );
    }

    #[test]
    fn metric_line_and_polygon_still_reduce_and_stay_valid() {
        // Metric mode must still simplify (a dense high-latitude ring) and keep
        // a valid closed, correctly-wound ring.
        let ring = circle_ring(0.0, 70.0, 0.5, 400, true);
        let out = simplify_polygon_rings_for_zoom_with(&[ring.clone()], 8, 14, true);
        assert!(out[0].len() < ring.len(), "metric mode should still reduce");
        assert!(out[0].len() >= 4, "ring must stay a valid closed ring");
        assert_eq!(out[0].first(), out[0].last(), "ring must stay closed");
        assert!(ring_signed_area(&out[0]) > 0.0, "winding preserved (CCW)");

        // Max-tier stays lossless in metric mode too.
        assert_eq!(
            simplify_polygon_rings_for_zoom_with(&[ring.clone()], 14, 14, true),
            vec![ring],
        );
    }

    #[test]
    fn polygon_simplify_keeps_sparse_ring_untouched() {
        // A ring already at the minimum (a single closed triangle) cannot be
        // reduced further and comes back unchanged — never collapsed.
        let rings = vec![vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ]];
        let out = simplify_polygon_rings_for_zoom(&rings, 4, 14);
        assert_eq!(out, rings, "a minimal triangle must survive verbatim");
    }
}
