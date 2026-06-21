#!/usr/bin/env python
"""Empirical bake-off: how to SUMMARIZE a LiDAR sweep with fewer points while
keeping detail. Runs several decimation strategies at matched point budgets on a
real full-density Waymo sweep and scores each on geometric fidelity, detail
preservation (curvature-weighted), and coverage uniformity.

This is the measurement-before-optimizing tool (sibling of point_column_stats.rs):
it picks the winner on fidelity/perf so the in-browser look only has to confirm
aesthetics, not discover them.

Run:  venv-waymo/bin/python lidar_summarize_eval.py
"""
import sys
import numpy as np
from scipy.spatial import cKDTree

REF = "../../examples/showcase/public/data/waymo-sf-day-splat/lidar-high.parquet"
SURFEL_K = 12          # matches compute_surfels in waymo_extract.py
RNG = np.random.default_rng(0)


# ── load one coherent sweep, in local metres ─────────────────────────────────
def load_sweep(path, which="mid"):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["geometry", "timestamp", "z"])
    ts = t.column("timestamp").to_numpy()
    z = t.column("z").to_numpy().astype(np.float64)
    uniq = np.unique(ts)
    sweep = {"start": uniq[5], "mid": uniq[len(uniq) // 2], "end": uniq[-5]}[which]
    m = ts == sweep
    geom = t.column("geometry").to_pylist()
    raw = b"".join(g for g, keep in zip(geom, m) if keep)
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 21)   # WKB point = 21 bytes
    assert b[0, 0] == 1, "expected little-endian WKB"
    lon = b[:, 5:13].copy().view(np.float64).reshape(-1)
    lat = b[:, 13:21].copy().view(np.float64).reshape(-1)
    zz = z[m]
    lon0, lat0 = lon.mean(), lat.mean()
    mx = (lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
    my = (lat - lat0) * 111320.0
    return np.column_stack([mx, my, zz])


# ── surface variation σ = λ0/(λ0+λ1+λ2) — the same eigenframe compute_surfels fits ─
def surface_variation(P, k=SURFEL_K):
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k + 1)
    neigh = P[idx]
    c = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / (k + 1)
    evals = np.linalg.eigvalsh(cov)            # ascending
    return evals[:, 0] / (evals.sum(axis=1) + 1e-12)


def voxel_centroids(P, v):
    keys = np.floor(P / v).astype(np.int64)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, P)
    return sums / counts[:, None]


def voxel_real_indices(P, v):
    """Index of the REAL point nearest each occupied voxel's centroid (no synthetic pts)."""
    keys = np.floor(P / v).astype(np.int64)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, P)
    cent = sums / counts[:, None]
    d2 = np.einsum("ni,ni->n", P - cent[inv], P - cent[inv])
    best = np.full(len(uniq), np.inf)
    np.minimum.at(best, inv, d2)
    return np.flatnonzero(d2 == best[inv])      # ties keep >1 but rare; trimmed by caller


def voxel_size_for_count(P, target):
    """Bisect voxel edge so #occupied voxels ≈ target."""
    lo, hi = 0.02, 8.0
    for _ in range(28):
        v = np.sqrt(lo * hi)
        n = len(np.unique(np.floor(P / v).astype(np.int64), axis=0))
        if n > target:
            lo = v
        else:
            hi = v
    return np.sqrt(lo * hi)


# ── strategies: each returns kept points D (n×3), some real, some centroids ───
def s_uniform(P, K, sv):
    s = max(1, round(len(P) / K))
    return P[::s]


def s_random(P, K, sv):
    return P[RNG.choice(len(P), K, replace=False)]


def s_voxel(P, K, sv):
    return voxel_centroids(P, voxel_size_for_count(P, K))


def s_fps(P, K, sv):
    """Farthest-point (blue-noise) sampling — even, clump-free coverage."""
    n = len(P)
    sel = [RNG.integers(n)]
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    for _ in range(K - 1):
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return P[sel]


def s_curv_hybrid(P, K, sv, alpha=0.6):
    """alpha*K guaranteed-detail real points (top curvature) + rest random coverage."""
    order = np.argsort(sv)[::-1]
    ne = int(alpha * K)
    edge = order[:ne]
    cov = RNG.choice(order[ne:], K - ne, replace=False)
    return P[np.concatenate([edge, cov])]


def s_adaptive_voxel(P, K, sv, beta=0.5):
    """Keep ALL top-curvature points as real returns; voxel-summarize the flat rest."""
    order = np.argsort(sv)[::-1]
    ne = int(beta * K)
    edge_idx = order[:ne]
    edge = P[edge_idx]
    flat_mask = np.ones(len(P), bool)
    flat_mask[edge_idx] = False
    flat = P[flat_mask]
    cent = voxel_centroids(flat, voxel_size_for_count(flat, K - ne))
    return np.vstack([edge, cent])


def s_adaptive_voxel_real(P, K, sv, beta=0.5):
    """Index-only twin of adaptive-voxel: edges + voxel-REPRESENTATIVE real points.
    This is the implementable version — a smarter `sel` index, no synthetic points,
    so attributes (rgb/seg_class) ride along untouched."""
    order = np.argsort(sv)[::-1]
    ne = int(beta * K)
    edge_idx = order[:ne]
    flat_mask = np.ones(len(P), bool)
    flat_mask[edge_idx] = False
    flat_pos = np.flatnonzero(flat_mask)
    flat = P[flat_pos]
    rep_local = voxel_real_indices(flat, voxel_size_for_count(flat, K - ne))
    keep = np.concatenate([edge_idx, flat_pos[rep_local]])
    return P[keep]


STRATS = {
    "uniform(baseline)": s_uniform,
    "random": s_random,
    "voxel": s_voxel,
    "fps/blue-noise": s_fps,
    "curv-hybrid": s_curv_hybrid,
    "adaptive-voxel": s_adaptive_voxel,
    "adaptive-vox-real": s_adaptive_voxel_real,
}


# ── metrics (all in cm) ──────────────────────────────────────────────────────
def point_to_plane_rms(P, D, mask=None, k=6):
    """For each ref point, distance to the local plane fit of its k nearest D points."""
    treeD = cKDTree(D)
    _, idx = treeD.query(P, k=min(k, len(D)))
    nb = D[idx]                                   # (n,k,3)
    cen = nb.mean(axis=1)
    c = nb - cen[:, None, :]
    cov = np.einsum("nki,nkj->nij", c, c) / nb.shape[1]
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, :, 0]                        # smallest eigenvector
    dist = np.abs(np.einsum("ni,ni->n", P - cen, normal))
    if mask is not None:
        dist = dist[mask]
    return np.sqrt(np.mean(dist ** 2)) * 100.0


def coverage(P, D):
    d, _ = cKDTree(D).query(P, k=1)
    return d.mean() * 100.0, np.percentile(d, 95) * 100.0


def run(which="mid"):
    P = load_sweep(REF, which)
    N = len(P)
    sv = surface_variation(P)
    detail_mask = sv >= np.percentile(sv, 90)      # top-decile curvature = edges/poles
    print(f"\n### sweep '{which}': N={N} pts, "
          f"detail(top-10% curvature)={detail_mask.sum()} pts\n")
    for f in (1 / 2, 1 / 4, 1 / 8, 1 / 16):
        K = round(f * N)
        print(f"--- budget {f:.4f}  (K≈{K} pts, ~{K*4.4/1e3:.0f} KB @4.4B/pt) ---")
        print(f"  {'strategy':<20}{'kept':>7}{'surfRMS':>9}{'detailRMS':>11}"
              f"{'cover̄':>8}{'cover95':>9}")
        rows = []
        for name, fn in STRATS.items():
            D = fn(P, K, sv)
            surf = point_to_plane_rms(P, D)
            det = point_to_plane_rms(P, D, mask=detail_mask)
            cm, c95 = coverage(P, D)
            rows.append((name, len(D), surf, det, cm, c95))
        best_det = min(r[3] for r in rows[1:])     # exclude baseline from "best"
        for name, n, surf, det, cm, c95 in rows:
            star = " *" if det == best_det else ""
            print(f"  {name:<20}{n:>7}{surf:>9.1f}{det:>11.1f}{cm:>8.1f}{c95:>9.1f}{star}")
        print()


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["mid"]):
        run(w)
