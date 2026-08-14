//! Deterministic CPU **kernel-density edge bundling** (KDEEB; Hurter, Ersoy &
//! Telea 2012) for *baking* bundled flow geometry into the tile pyramid at build
//! time — the producer-side counterpart to the GPU `EdgeBundler` in
//! `@poopdeck.gl/layers`.
//!
//! The GPU layer bundles the union of *visible* tiles every time the view
//! changes; this module bundles the **whole edge set once per zoom band** at
//! build time and stores the resulting polylines, so the client renders a static
//! curve (no float-blend GPU capability needed, no per-frame relaxation, works on
//! mobile). Per the edge-bundling literature, bundling is a *global* operation —
//! bundling tile-local subsets independently would seam — so this is only ever
//! called on a complete per-zoom edge set (see [`super::datasets::bixi`]).
//!
//! ## Determinism
//! Baked tiles are content-addressed, so the output must be byte-reproducible.
//! KDEEB (unlike CUBu's cuRAND-perturbed sampling) uses a *uniform* step and no
//! randomness, and we pin the density-map resolution ([`DENSITY_RES`]) — the two
//! things that would otherwise make the geometry machine- or run-dependent.
//!
//! ## Pipeline (mirrors the GPU `EdgeBundler`, on the CPU)
//! Each edge is resampled to `P` control points, normalized into a fixed
//! [`WORK_SIZE`] isotropic box (cosLat-corrected so the kernel behaves the same
//! at any latitude/extent), then for ≈15 annealed iterations:
//!  1. **splat** an Epanechnikov density kernel of bandwidth `h` at every control
//!     point into a [`DENSITY_RES`]² grid,
//!  2. **advect** each interior point a step `h` up the normalized density
//!     gradient,
//!  3. **resample** each edge to uniform arc-length spacing,
//!  4. **smooth** one 1-D Laplacian pass along each edge,
//!  5. **anneal** `h *= λ` and repeat.
//! Endpoints are pinned every pass. The result is mapped back to lon/lat.

/// Side length of the normalized isotropic simulation box. KDEEB's kernel/step
/// constants are scale-relative, so every edge set is mapped into this box.
const WORK_SIZE: f64 = 1000.0;

/// Density-map resolution. **Pinned** (not a parameter) because the density grid
/// determines the bundled geometry — varying it would break reproducibility of
/// the content-addressed packs. 512² matches the CUBu/GPU default.
const DENSITY_RES: usize = 512;

const EPS: f64 = 1e-9;

/// Tuning for [`bundle_edges`]. Field meanings mirror the GPU `EdgeBundler`'s
/// props so a baked build and a live build look the same.
#[derive(Clone, Copy, Debug)]
pub struct BundleParams {
    /// Control points per edge, `P` (≥ 3; endpoints pinned). Fewer points = a
    /// smaller per-corridor matrix (each baked vertex carries a time-series row),
    /// more = a smoother curve.
    pub points: usize,
    /// KDEEB density-advection iterations (≈15 converges).
    pub iterations: usize,
    /// Initial kernel bandwidth as a FRACTION of the drawing size (≈0.05). The
    /// headline knob: larger bundles flows together more aggressively.
    pub kernel_fraction: f64,
    /// Per-iteration Laplacian smoothing strength `f ∈ [0,1]` (≈0.5).
    pub smoothing: f64,
    /// Kernel-bandwidth anneal factor per iteration, `λ ∈ [0.3,0.95]` (≈0.8).
    pub anneal: f64,
}

impl Default for BundleParams {
    fn default() -> Self {
        Self {
            points: 24,
            iterations: 15,
            kernel_fraction: 0.05,
            smoothing: 0.5,
            anneal: 0.8,
        }
    }
}

type P2 = [f64; 2];

#[inline]
fn dist(a: P2, b: P2) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Radial **Epanechnikov** kernel weight: `1 − (d/h)²` inside the bandwidth `h`,
/// `0` outside (KDEEB's density kernel; smooth, finite-support, cheap).
#[inline]
pub fn epanechnikov_weight(d: f64, radius: f64) -> f64 {
    if radius <= EPS {
        return 0.0;
    }
    let x = d / radius;
    if x >= 1.0 {
        0.0
    } else {
        1.0 - x * x
    }
}

/// One 1-D Laplacian smoothing step toward the midpoint of a point's same-edge
/// neighbours, by strength `f`: `cur + f·(½(prev+next) − cur)`. Evenly-spaced
/// collinear points are unchanged; kinks relax.
#[inline]
pub fn laplacian_step(prev: P2, cur: P2, next: P2, f: f64) -> P2 {
    let mx = 0.5 * (prev[0] + next[0]) - cur[0];
    let my = 0.5 * (prev[1] + next[1]) - cur[1];
    [cur[0] + f * mx, cur[1] + f * my]
}

/// Resample a polyline to `n` evenly arc-length-spaced points, preserving the
/// endpoints. Seeds each edge's control points (a 2-vertex OD pair stays
/// straight; an N-vertex trajectory keeps its curve) and mirrors the
/// per-iteration resample pass.
pub fn subdivide(points: &[P2], n: usize) -> Vec<P2> {
    if n < 2 || points.len() < 2 {
        return points.to_vec();
    }
    let mut cum = vec![0.0f64; points.len()];
    for i in 1..points.len() {
        cum[i] = cum[i - 1] + dist(points[i], points[i - 1]);
    }
    let total = *cum.last().unwrap();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let target = total * k as f64 / (n - 1) as f64;
        let mut i = 1;
        while i < cum.len() - 1 && cum[i] < target {
            i += 1;
        }
        let seg = cum[i] - cum[i - 1];
        let f = if seg < EPS {
            0.0
        } else {
            (target - cum[i - 1]) / seg
        };
        out.push([
            points[i - 1][0] + f * (points[i][0] - points[i - 1][0]),
            points[i - 1][1] + f * (points[i][1] - points[i - 1][1]),
        ]);
    }
    out
}

/// Bundle `edges` (each a polyline in `[lon, lat]`) into smooth rivers, returning
/// each edge resampled to `params.points` bundled control points in `[lon, lat]`.
///
/// Deterministic and order-independent up to the input order: the same edges in
/// the same order always produce byte-identical output. Edges with fewer than 2
/// vertices, and degenerate sets (< 2 edges or zero extent), are returned simply
/// resampled (no bundling) so the caller never has to special-case them.
pub fn bundle_edges(edges: &[Vec<P2>], params: &BundleParams) -> Vec<Vec<P2>> {
    let p = params.points.max(3);
    let e = edges.len();

    // Resample every edge to P points up front (the working representation).
    let resampled: Vec<Vec<P2>> = edges
        .iter()
        .map(|edge| {
            if edge.len() >= 2 {
                subdivide(edge, p)
            } else {
                let pt = edge.first().copied().unwrap_or([0.0, 0.0]);
                vec![pt; p]
            }
        })
        .collect();

    if e < 2 {
        return resampled;
    }

    // cosLat correction makes the longitude axis isotropic in the work box.
    let mut lat_sum = 0.0f64;
    let mut count = 0.0f64;
    for edge in &resampled {
        for v in edge {
            lat_sum += v[1];
            count += 1.0;
        }
    }
    let cos_lat0 = (lat_sum / count.max(1.0) * std::f64::consts::PI / 180.0)
        .cos()
        .max(0.1);

    // Normalize cosLat-corrected control points into the fixed WORK box.
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for edge in &resampled {
        for v in edge {
            let cx = v[0] * cos_lat0;
            min_x = min_x.min(cx);
            max_x = max_x.max(cx);
            min_y = min_y.min(v[1]);
            max_y = max_y.max(v[1]);
        }
    }
    let extent = (max_x - min_x).max(max_y - min_y).max(EPS);
    let scale = WORK_SIZE / extent;
    if extent <= EPS {
        return resampled;
    }

    // Flat work-space buffer, edge-major then point-major: work[(ei*P + i)].
    let mut work: Vec<P2> = Vec::with_capacity(e * p);
    for edge in &resampled {
        for v in edge {
            work.push([(v[0] * cos_lat0 - min_x) * scale, (v[1] - min_y) * scale]);
        }
    }

    let kernel0 = (params.kernel_fraction * WORK_SIZE).max(2.0);
    let anneal = params.anneal.clamp(0.3, 0.95);
    let smoothing = params.smoothing.clamp(0.0, 1.0);
    let iterations = params.iterations.max(1);

    let mut density = vec![0.0f64; DENSITY_RES * DENSITY_RES];
    let mut next = work.clone();

    for it in 0..iterations {
        let h = kernel0 * anneal.powi(it as i32);

        // 1. SPLAT every control point into the density grid.
        density.iter_mut().for_each(|c| *c = 0.0);
        splat_density(&work, h, &mut density);

        // 2. ADVECT interior points up the normalized density gradient by `h`.
        for ei in 0..e {
            for i in 0..p {
                let idx = ei * p + i;
                if i == 0 || i == p - 1 {
                    next[idx] = work[idx];
                    continue;
                }
                next[idx] = advect_point(work[idx], &density, h);
            }
        }
        std::mem::swap(&mut work, &mut next);

        // 3. RESAMPLE each edge to uniform arc-length spacing.
        for ei in 0..e {
            let slice = &work[ei * p..ei * p + p];
            let r = subdivide(slice, p);
            next[ei * p..ei * p + p].copy_from_slice(&r);
        }
        std::mem::swap(&mut work, &mut next);

        // 4. SMOOTH one Laplacian pass (endpoints pinned).
        for ei in 0..e {
            for i in 0..p {
                let idx = ei * p + i;
                if i == 0 || i == p - 1 {
                    next[idx] = work[idx];
                } else {
                    next[idx] = laplacian_step(work[idx - 1], work[idx], work[idx + 1], smoothing);
                }
            }
        }
        std::mem::swap(&mut work, &mut next);
    }

    // Map work space back to lon/lat.
    let mut out = Vec::with_capacity(e);
    for ei in 0..e {
        let mut edge = Vec::with_capacity(p);
        for i in 0..p {
            let w = work[ei * p + i];
            let cx = w[0] / scale + min_x;
            let cy = w[1] / scale + min_y;
            edge.push([cx / cos_lat0, cy]);
        }
        out.push(edge);
    }
    out
}

/// Additively rasterize an Epanechnikov kernel of work-space bandwidth `h` at
/// every control point into the [`DENSITY_RES`]² grid (the mean-shift density
/// estimate). The splat is bounded to the kernel's grid bounding box.
fn splat_density(work: &[P2], h: f64, density: &mut [f64]) {
    let to_grid = DENSITY_RES as f64 / WORK_SIZE;
    let h_grid = (h * to_grid).max(1.0);
    let hr = h_grid.ceil() as i64;
    let last = DENSITY_RES as i64 - 1;
    for w in work {
        let gx = w[0] * to_grid;
        let gy = w[1] * to_grid;
        let cx = gx.floor() as i64;
        let cy = gy.floor() as i64;
        for dy in -hr..=hr {
            let y = cy + dy;
            if y < 0 || y > last {
                continue;
            }
            for dx in -hr..=hr {
                let x = cx + dx;
                if x < 0 || x > last {
                    continue;
                }
                let d = (((x as f64 + 0.5) - gx).powi(2) + ((y as f64 + 0.5) - gy).powi(2)).sqrt();
                let wgt = epanechnikov_weight(d, h_grid);
                if wgt > 0.0 {
                    density[y as usize * DENSITY_RES + x as usize] += wgt;
                }
            }
        }
    }
}

/// Move one work-space point a step `h` up the normalized density gradient
/// (central differences on the grid; the grid→work scale cancels under
/// normalization). Returns the point unchanged where the gradient vanishes.
fn advect_point(p: P2, density: &[f64], h: f64) -> P2 {
    let to_grid = DENSITY_RES as f64 / WORK_SIZE;
    let last = DENSITY_RES as i64 - 1;
    let cl = |v: i64| v.clamp(0, last) as usize;
    let gx = (p[0] * to_grid).floor() as i64;
    let gy = (p[1] * to_grid).floor() as i64;
    let x = cl(gx);
    let y = cl(gy);
    let at = |xx: usize, yy: usize| density[yy * DENSITY_RES + xx];
    let dl = at(cl(gx - 1), y);
    let dr = at(cl(gx + 1), y);
    let db = at(x, cl(gy - 1));
    let dt = at(x, cl(gy + 1));
    let grad = [dr - dl, dt - db];
    let g = (grad[0] * grad[0] + grad[1] * grad[1]).sqrt();
    if g > 1e-8 {
        [p[0] + h * grad[0] / g, p[1] + h * grad[1] / g]
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epanechnikov_is_finite_support() {
        assert!((epanechnikov_weight(0.0, 10.0) - 1.0).abs() < 1e-12);
        assert!(epanechnikov_weight(10.0, 10.0).abs() < 1e-12);
        assert!(epanechnikov_weight(11.0, 10.0).abs() < 1e-12);
        assert_eq!(epanechnikov_weight(5.0, 0.0), 0.0);
    }

    #[test]
    fn laplacian_leaves_collinear_evenly_spaced_unchanged() {
        let p = laplacian_step([0.0, 0.0], [1.0, 0.0], [2.0, 0.0], 0.5);
        assert!((p[0] - 1.0).abs() < 1e-12 && p[1].abs() < 1e-12);
    }

    #[test]
    fn laplacian_relaxes_a_kink() {
        // A point pushed off the line moves back toward the midpoint.
        let p = laplacian_step([0.0, 0.0], [1.0, 1.0], [2.0, 0.0], 0.5);
        assert!(p[1] < 1.0 && p[1] > 0.0);
    }

    #[test]
    fn subdivide_preserves_endpoints_and_count() {
        let line = vec![[0.0, 0.0], [10.0, 0.0]];
        let r = subdivide(&line, 5);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0], [0.0, 0.0]);
        assert_eq!(r[4], [10.0, 0.0]);
        // Evenly spaced along a straight segment.
        assert!((r[2][0] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn bundle_pins_endpoints() {
        // Two near-parallel edges; endpoints must not move.
        let edges = vec![
            vec![[-73.60, 45.50], [-73.50, 45.55]],
            vec![[-73.601, 45.501], [-73.501, 45.551]],
        ];
        let out = bundle_edges(&edges, &BundleParams::default());
        assert_eq!(out.len(), 2);
        for (i, e) in out.iter().enumerate() {
            assert_eq!(e.len(), BundleParams::default().points);
            assert!((e[0][0] - edges[i][0][0]).abs() < 1e-9, "start lon pinned");
            assert!((e[0][1] - edges[i][0][1]).abs() < 1e-9, "start lat pinned");
            let n = e.len() - 1;
            assert!((e[n][0] - edges[i][1][0]).abs() < 1e-9, "end lon pinned");
            assert!((e[n][1] - edges[i][1][1]).abs() < 1e-9, "end lat pinned");
        }
    }

    #[test]
    fn bundle_pulls_parallel_edges_together() {
        // Two parallel edges a fixed gap apart should be closer at their
        // midpoints after bundling than their original straight midpoints.
        let a: Vec<[f64; 2]> = vec![[-73.60, 45.50], [-73.40, 45.50]];
        let b: Vec<[f64; 2]> = vec![[-73.60, 45.52], [-73.40, 45.52]];
        let gap0 = (a[0][1] - b[0][1]).abs(); // 0.02 deg at every x
        let out = bundle_edges(
            &[a, b],
            &BundleParams {
                iterations: 15,
                ..Default::default()
            },
        );
        let mid = out[0].len() / 2;
        let gap_mid = (out[0][mid][1] - out[1][mid][1]).abs();
        assert!(
            gap_mid < gap0,
            "midpoint gap {gap_mid} should shrink below {gap0} after bundling"
        );
    }

    #[test]
    fn bundle_is_deterministic() {
        let edges = vec![
            vec![[-73.60, 45.50], [-73.50, 45.56]],
            vec![[-73.59, 45.49], [-73.51, 45.55]],
            vec![[-73.61, 45.51], [-73.52, 45.54]],
        ];
        let a = bundle_edges(&edges, &BundleParams::default());
        let b = bundle_edges(&edges, &BundleParams::default());
        assert_eq!(a, b, "same input must produce byte-identical output");
    }

    #[test]
    fn single_edge_is_returned_resampled_not_bundled() {
        let edges = vec![vec![[-73.60, 45.50], [-73.50, 45.55]]];
        let out = bundle_edges(
            &edges,
            &BundleParams {
                points: 8,
                ..Default::default()
            },
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 8);
        // Straight line: still collinear (no neighbours to bundle toward).
        let n = out[0].len() - 1;
        let dx = out[0][n][0] - out[0][0][0];
        let dy = out[0][n][1] - out[0][0][1];
        for pt in &out[0] {
            // cross product of (pt-start) with (end-start) ≈ 0 → on the line.
            let cross = (pt[0] - out[0][0][0]) * dy - (pt[1] - out[0][0][1]) * dx;
            assert!(cross.abs() < 1e-6, "single edge should stay straight");
        }
    }
}
