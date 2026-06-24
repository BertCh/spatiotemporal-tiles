#!/usr/bin/env python3
"""Unit tests for the additive-octree LOD home-zoom assignment in ``av_common``:

* ``lod_home_zoom`` — per-sweep hierarchical voxel subsample assigning every
  return a SINGLE home zoom (coarse = sparse overview, fine = residual detail).
* round-trip of the ``home_zoom`` column through ``write_lidar_points``.

The whole point of additive-octree LOD: each point is materialized at exactly
one zoom and the UNION over levels reconstructs the full cloud (lossless), so the
client loads minZoom..cameraZoom and zooming in adds only the residual.

Runs standalone (``venv-*/bin/python test_lod_home_zoom.py``) — also
pytest-discoverable. Pure numpy / pyarrow, no raw AV data (synthetic clouds).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import av_common as avc


def _ground_grid(n_side=80, spacing_m=0.25, lat0=25.77, lon0=-80.19):
    """A dense flat ground patch: ``n_side``×``n_side`` returns at ``spacing_m``.

    Returned as lon/lat/z so it feeds ``lod_home_zoom`` directly. The patch is
    metres-square via the same deg→m scaling the function uses internally.
    """
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    gx, gy = np.meshgrid(
        np.arange(n_side) * spacing_m, np.arange(n_side) * spacing_m
    )
    x_m = gx.ravel()
    y_m = gy.ravel()
    lon = lon0 + x_m / m_per_deg_lon
    lat = lat0 + y_m / 110540.0
    z = np.zeros(x_m.size)
    return lon, lat, z


def test_every_point_assigned_one_zoom_in_band():
    lon, lat, z = _ground_grid()
    t = np.full(lon.size, 1000, dtype="int64")
    home = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=19)
    assert home.shape == (lon.size,)
    assert home.dtype == np.int64
    # Every return lands in exactly one level within [min, max].
    assert home.min() >= 14 and home.max() <= 19
    # Union of all levels == the whole cloud (lossless): trivially true since each
    # point has a home zoom, but assert the count reconciles.
    assert sum(int((home == zz).sum()) for zz in range(14, 20)) == lon.size


def test_coarse_levels_are_sparser_than_residual():
    """A dense ground patch: coarse zooms hold a FEW reps, the residual (max) holds
    the bulk — the additive pyramid shape."""
    lon, lat, z = _ground_grid(n_side=100, spacing_m=0.2)
    t = np.zeros(lon.size, dtype="int64")
    # Uniform-grid invariant — pin curvature off (the geometry-aware path
    # intentionally grades per-region density, so it isn't monotonic by level) and
    # pin px_per_point so the 0.2 m grid isn't fully resolved before the residual
    # (otherwise the deepest level, not the residual, holds the bulk).
    home = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=19,
                             curvature=False, px_per_point=1.0)
    counts = {zz: int((home == zz).sum()) for zz in range(14, 20)}
    # Coarsest level is a sparse overview …
    assert counts[14] < counts[19], counts
    # … and counts grow ~monotonically toward the residual (each finer voxel grid
    # claims more reps). Allow equality (some adjacent levels can tie on a grid).
    ordered = [counts[zz] for zz in range(14, 20)]
    assert all(b >= a for a, b in zip(ordered, ordered[1:])), counts
    # Coarsest level is genuinely sparse vs the full cloud.
    assert counts[14] < lon.size // 10, counts


def test_deterministic():
    lon, lat, z = _ground_grid()
    t = np.full(lon.size, 7, dtype="int64")
    a = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=19)
    b = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=19)
    assert np.array_equal(a, b)


def test_per_sweep_independence():
    """Two sweeps with IDENTICAL geometry but different timestamps must get the
    same per-level distribution — home zoom is assigned per sweep, not globally
    (else sweep 2's points would all be 'already claimed' by sweep 1)."""
    lon, lat, z = _ground_grid(n_side=60, spacing_m=0.3)
    t1 = np.zeros(lon.size, dtype="int64")
    t2 = np.full(lon.size, 100, dtype="int64")
    lon2 = np.concatenate([lon, lon])
    lat2 = np.concatenate([lat, lat])
    z2 = np.concatenate([z, z])
    tt = np.concatenate([t1, t2])
    home = avc.lod_home_zoom(lon2, lat2, z2, tt, min_zoom=14, max_zoom=19)
    h1 = home[: lon.size]
    h2 = home[lon.size:]
    # Same geometry sampled in two sweeps → identical home zoom per matching point.
    assert np.array_equal(h1, h2)


def test_single_point_lands_at_coarsest():
    """One return is claimed by the very first (coarsest) voxel grid."""
    home = avc.lod_home_zoom(
        np.array([-80.19]), np.array([25.77]), np.array([1.5]),
        np.array([0], dtype="int64"), min_zoom=14, max_zoom=19)
    assert home.tolist() == [14]


def test_empty_cloud():
    home = avc.lod_home_zoom(
        np.array([]), np.array([]), np.array([]), np.array([], dtype="int64"),
        min_zoom=14, max_zoom=19)
    assert home.shape == (0,)


def test_vertical_structure_preserved():
    """3D voxels: a pole (stacked returns at one x,y over a range of z) must not
    collapse to a single coarse point — its upper returns survive to finer levels.
    Guards against a 2D-ground voxelisation that would merge the whole column."""
    lat0, lon0 = 25.77, -80.19
    zcol = np.linspace(0.0, 6.0, 40)  # 6 m pole, 15 cm spacing
    lon = np.full(zcol.size, lon0)
    lat = np.full(zcol.size, lat0)
    t = np.zeros(zcol.size, dtype="int64")
    # Property of the uniform 3D voxel grid (a clean line is low-σ, so the
    # geometry-aware path treats it as flat — see the foliage test for curvature).
    home = avc.lod_home_zoom(lon, lat, zcol, t, min_zoom=14, max_zoom=19,
                             curvature=False)
    # The column is NOT all-coarse: finer levels claim the vertically-separated
    # returns (a 2D voxeliser would put them all at z14).
    assert home.max() > 14, home.tolist()


def test_home_zoom_roundtrips_through_write_lidar_points():
    import pyarrow.parquet as pq

    lon, lat, z = _ground_grid(n_side=20, spacing_m=0.5)
    t = np.zeros(lon.size, dtype="int64")
    home = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=19)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "lidar-lod.parquet"
        n = avc.write_lidar_points(
            out, lon=lon, lat=lat, timestamp=t, z=z, home_zoom=home)
        assert n == lon.size
        tbl = pq.read_table(out)
        assert "home_zoom" in tbl.column_names
        got = tbl.column("home_zoom").to_numpy()
        assert np.array_equal(got, home)


def _plane_and_foliage():
    """A flat ground plane (low σ) next to a 'foliage' slab — the same grid
    footprint but with strong z-noise, so it is locally NON-planar (high σ).
    Returns lon/lat/z + a boolean mask marking the high-curvature foliage."""
    lat0, lon0 = 25.77, -80.19
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    gx, gy = np.meshgrid(np.arange(80) * 0.2, np.arange(80) * 0.2)
    fx, fy = gx.ravel(), gy.ravel()
    fz = np.zeros(fx.size)
    rng = np.random.default_rng(1)
    hx, hy = np.meshgrid(20.0 + np.arange(80) * 0.2, np.arange(80) * 0.2)
    hx, hy = hx.ravel(), hy.ravel()
    hz = rng.normal(0.0, 0.4, hx.size)  # 3D scatter → high surface variation
    x = np.concatenate([fx, hx])
    y = np.concatenate([fy, hy])
    z = np.concatenate([fz, hz])
    lon = lon0 + x / m_per_deg_lon
    lat = lat0 + y / 110540.0
    is_foliage = np.concatenate(
        [np.zeros(fx.size, bool), np.ones(hx.size, bool)]
    )
    return lon, lat, z, is_foliage


def test_curvature_concentrates_coarse_budget_on_structure():
    """The whole point of the geometry-aware path: at the COARSE (overview) zooms,
    the kept points skew toward the high-curvature region vs the uniform grid —
    'preserve the geometry with the fewest points'."""
    lon, lat, z, is_fol = _plane_and_foliage()
    t = np.zeros(lon.size, dtype="int64")
    hc = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=20, curvature=True)
    hu = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=20, curvature=False)
    # Still a complete, single-home-zoom partition.
    assert hc.min() >= 14 and hc.max() <= 20
    assert sum(int((hc == zz).sum()) for zz in range(14, 21)) == lon.size
    # At the coarse levels (z14-16), curvature's kept points are MORE concentrated
    # on the high-σ foliage than the uniform grid's.
    coarse_c = hc <= 16
    coarse_u = hu <= 16
    assert is_fol[coarse_c].mean() > is_fol[coarse_u].mean(), (
        is_fol[coarse_c].mean(), is_fol[coarse_u].mean())


def test_curvature_defers_flat_to_finer_zoom():
    """A flat point's mean home zoom is FINER (higher) under curvature than the
    foliage's — flats are deferred, structure is brought forward."""
    lon, lat, z, is_fol = _plane_and_foliage()
    t = np.zeros(lon.size, dtype="int64")
    hc = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=20, curvature=True)
    assert hc[~is_fol].mean() > hc[is_fol].mean(), (
        hc[~is_fol].mean(), hc[is_fol].mean())


def test_curvature_deterministic_and_lossless():
    lon, lat, z, _ = _plane_and_foliage()
    t = np.zeros(lon.size, dtype="int64")
    a = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=20, curvature=True)
    b = avc.lod_home_zoom(lon, lat, z, t, min_zoom=14, max_zoom=20, curvature=True)
    assert np.array_equal(a, b)
    assert sum(int((a == zz).sum()) for zz in range(14, 21)) == lon.size


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"All {len(fns)} LOD home-zoom tests passed.")


if __name__ == "__main__":
    sys.exit(_run_all())
