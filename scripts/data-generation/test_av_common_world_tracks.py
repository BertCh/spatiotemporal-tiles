#!/usr/bin/env python3
"""Unit tests for the new AV render-mode helpers in ``av_common``:

* Feature 1 — Worldbuild: ``WorldVoxelAccumulator`` (streaming voxel dedup),
  ``point_in_moving_boxes`` (box-local membership), ``worldbuild_merge``
  (static dedup + dynamic pass-through + columns).
* Feature 2 — Tracks: ``build_tracks`` (group / sort / ≥2-vertex guard) and a
  round-trip through ``write_track_lines``.

Runs standalone (``venv-*/bin/python test_av_common_world_tracks.py``) — also
pytest-discoverable if pytest is installed. Pure numpy / pyarrow / shapely, no
raw AV data needed (synthetic clouds only).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import av_common as avc


# ── Feature 1: WorldVoxelAccumulator + worldbuild_merge ──────────────────────
def _three_sweep_static_box_cloud():
    """A synthetic STATIC wall sampled by 3 sweeps at decreasing times.

    Same 5×5 grid of points on a vertical plane (a "wall") is seen in all three
    sweeps with tiny jitter, plus a camera color that grows brighter each sweep.
    The 3 sweeps share voxels (jitter << voxel size) so the accumulator must
    dedup 3N raw returns → ~N voxels, first_seen = the earliest (smallest) time.
    """
    rng = np.random.default_rng(0)
    # Grid points sit at voxel-CELL CENTRES (0.075 + k·0.15) so tiny jitter never
    # crosses a voxel boundary → the 3 sweeps share exactly the same 25 voxels.
    coords = 0.075 + np.arange(5) * 0.30  # 5 cells, every other 0.15 voxel
    gx, gz = np.meshgrid(coords, coords)
    base_x = gx.ravel()
    base_z = gz.ravel()
    base_y = np.full_like(base_x, 0.075)  # wall at the y-cell centre
    sweeps = []
    for k, t in enumerate((3000, 1000, 2000)):  # deliberately out of order
        jitter = rng.normal(scale=1e-4, size=(len(base_x), 3))  # << voxel 0.15
        xy = np.column_stack([base_x + jitter[:, 0], base_y + jitter[:, 1]])
        z = base_z + jitter[:, 2]
        rgb = np.full((len(base_x), 3), [60 + 30 * k, 0, 0], dtype="float64")
        hit = np.ones(len(base_x), dtype=bool)
        sweeps.append((xy, z, t, rgb, hit))
    return sweeps, len(base_x)


def test_voxel_dedup_and_first_seen():
    sweeps, n_unique = _three_sweep_static_box_cloud()
    acc = avc.WorldVoxelAccumulator(voxel_m=0.15)
    for xy, z, t, rgb, hit in sweeps:
        acc.add(xy, z, t, rgb=rgb, rgb_hit=hit)
    xy, z, rgb, first_seen = acc.finalize()
    # 3 sweeps × N returns share voxels → ~N occupied voxels (jitter tiny).
    assert len(z) == n_unique, f"expected {n_unique} voxels, got {len(z)}"
    # first_seen is the EARLIEST sweep time per voxel (min of 1000/2000/3000).
    assert int(first_seen.min()) == 1000 and int(first_seen.max()) == 1000, \
        f"first_seen should all == 1000 (the min), got {np.unique(first_seen)}"
    # RGB is the mean over the 3 sweeps' reds: (60+90+120)/3 = 90.
    assert abs(float(rgb[:, 0].mean()) - 90.0) <= 1.0, \
        f"mean red should be ~90, got {rgb[:, 0].mean()}"
    print("PASS test_voxel_dedup_and_first_seen "
          f"({len(z)} voxels, first_seen=1000, mean red {rgb[:, 0].mean():.1f})")


def test_point_in_moving_box():
    # A box centred at (10,0), heading 0, 4 m long × 2 m wide, observed at t=1000.
    boxes = [dict(cx=10.0, cy=0.0, heading=0.0, length=4.0, width=2.0, t=1000)]
    # Inside (origin-ish), on the +x edge, and well outside.
    pts = np.array([[10.0, 0.0],   # centre → inside
                    [11.9, 0.0],   # within length/2=2 → inside
                    [13.0, 0.0],   # x'=3 > 2 → outside
                    [10.0, 5.0]])  # y'=5 > 1 → outside
    z = np.zeros(len(pts))
    mask = avc.point_in_moving_boxes(pts, z, 1000, boxes)
    assert mask.tolist() == [True, True, False, False], mask.tolist()
    # A return at a DIFFERENT time than the box observation is NOT tested.
    mask2 = avc.point_in_moving_boxes(pts[:1], z[:1], 9999, boxes)
    assert mask2.tolist() == [False], mask2.tolist()
    # Rotated box (heading 90°): a point offset along world +y is now within length/2.
    rbox = [dict(cx=0.0, cy=0.0, heading=np.pi / 2, length=4.0, width=2.0, t=1000)]
    rpts = np.array([[0.0, 1.9],   # along the box's long axis (rotated) → inside
                     [1.9, 0.0]])  # along the box's short axis → outside (1.9>1)
    rmask = avc.point_in_moving_boxes(rpts, np.zeros(2), 1000, rbox)
    assert rmask.tolist() == [True, False], rmask.tolist()
    print("PASS test_point_in_moving_box")


def test_worldbuild_merge_columns_and_dynamic_passthrough():
    sweeps, n_unique = _three_sweep_static_box_cloud()
    acc = avc.WorldVoxelAccumulator(voxel_m=0.15)
    for xy, z, t, rgb, hit in sweeps:
        acc.add(xy, z, t, rgb=rgb, rgb_hit=hit)

    # Dynamic returns: a moving actor seen at 2 sweep times, 4 returns each, all
    # at the SAME world spot would still pass through UN-merged (no voxel dedup).
    d_xy = np.array([[10.0, 0.0]] * 4 + [[10.1, 0.0]] * 4)
    d_z = np.zeros(8)
    d_rgb = np.full((8, 3), [200, 50, 50], dtype="float64")
    d_t = np.array([1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000], dtype="int64")
    dynamic = dict(xy=d_xy, z=d_z, rgb=d_rgb, t=d_t)

    res = avc.worldbuild_merge(acc, dynamic)
    n_static = n_unique
    n_dyn = 8
    m = n_static + n_dyn
    assert len(res["z"]) == m, f"merged count {len(res['z'])} != {m}"
    # Dynamic returns are NOT voxel-merged: all 8 survive.
    assert int(res["is_dynamic"].sum()) == n_dyn, \
        f"is_dynamic sum {res['is_dynamic'].sum()} != {n_dyn}"
    # is_dynamic is numeric 0/1; world_class is the matching STRING label.
    assert set(np.unique(res["is_dynamic"]).tolist()) <= {0, 1}
    wc = set(map(str, res["world_class"].tolist()))
    assert wc == {"static", "dynamic"}, wc
    # world_class values are NON-NUMERIC strings (the categorical promotion guard).
    for v in wc:
        assert not v.lstrip("-").isdigit(), f"world_class {v!r} is numeric-looking"
    # Static points carry first_seen (=1000) as their timestamp; dynamic keep theirs.
    static_ts = res["timestamp"][res["is_dynamic"] == 0]
    dyn_ts = res["timestamp"][res["is_dynamic"] == 1]
    assert set(static_ts.tolist()) == {1000}, set(static_ts.tolist())
    assert set(dyn_ts.tolist()) == {1000, 2000}, set(dyn_ts.tolist())
    # surfels are (M,7).
    assert res["surfels"].shape == (m, 7), res["surfels"].shape
    # rgb is uint8 (M,3).
    assert res["rgb"].shape == (m, 3) and res["rgb"].dtype == np.uint8
    print(f"PASS test_worldbuild_merge_columns_and_dynamic_passthrough "
          f"(static {n_static}, dynamic {n_dyn}, merged {m})")


def test_worldbuild_fallback_rgb_when_no_camera():
    """Voxels no camera saw fall back to the slate FALLBACK_RGB."""
    acc = avc.WorldVoxelAccumulator(voxel_m=0.15)
    xy = np.array([[0.0, 0.0], [5.0, 5.0]])
    z = np.zeros(2)
    acc.add(xy, z, 1000, rgb=None, rgb_hit=None)  # no camera color at all
    _xy, _z, rgb, _fs = acc.finalize()
    fb = np.asarray(avc.FALLBACK_RGB, dtype="uint8")
    assert np.all(rgb == fb), f"uncolored voxels should be FALLBACK_RGB, got {rgb}"
    print("PASS test_worldbuild_fallback_rgb_when_no_camera")


# ── Feature 2: build_tracks + write_track_lines ──────────────────────────────
def test_build_tracks_grouping_sort_and_guard():
    # Two real tracks (A: 3 obs out of order, B: 2 obs) + a singleton C (dropped).
    lon = [0.0, 0.2, 0.1,   1.0, 1.1,   5.0]
    lat = [0.0, 0.0, 0.0,   2.0, 2.0,   9.0]
    ts = [3000, 1000, 2000, 1000, 2000, 1000]
    cat = ["car", "car", "car", "pedestrian", "pedestrian", "bus"]
    tid = ["A", "A", "A", "B", "B", "C"]
    spd = [3.0, 1.0, 2.0, 0.5, 0.6, 9.9]
    tracks = avc.build_tracks(lon=lon, lat=lat, timestamp=ts, category=cat,
                              track_id=tid, speed=spd)
    by_cat = {t["category"]: t for t in tracks}
    # Singleton C is dropped (< 2 vertices).
    assert "bus" not in by_cat, "singleton track C should be dropped"
    assert set(by_cat) == {"car", "pedestrian"}, set(by_cat)
    # Track A is time-SORTED (1000,2000,3000) → vts ascending, vvals follow.
    a = by_cat["car"]
    assert a["vts"] == [1000, 2000, 3000], a["vts"]
    assert a["vvals"] == [1.0, 2.0, 3.0], a["vvals"]
    # lon followed the sort too (the t=1000 obs had lon 0.2).
    assert abs(a["lon"][0] - 0.2) < 1e-9, a["lon"]
    print(f"PASS test_build_tracks_grouping_sort_and_guard "
          f"({len(tracks)} tracks, A sorted)")


def test_write_track_lines_roundtrip_with_ego():
    import pyarrow.parquet as pq

    lon = [0.0, 0.2, 0.1, 1.0, 1.1]
    lat = [0.0, 0.0, 0.0, 2.0, 2.0]
    ts = [3000, 1000, 2000, 1000, 2000]
    cat = ["car", "car", "car", "pedestrian", "pedestrian"]
    tid = ["A", "A", "A", "B", "B"]
    spd = [3.0, 1.0, 2.0, 0.5, 0.6]
    tracks = avc.build_tracks(lon=lon, lat=lat, timestamp=ts, category=cat,
                              track_id=tid, speed=spd)
    # Fold the ego in as one more track, category exactly "ego".
    tracks.append(dict(lon=np.array([0.0, 0.5, 1.0]), lat=np.array([0.0, 0.5, 1.0]),
                       vts=[1000, 2000, 3000], vvals=[10.0, 11.0, 12.0],
                       category="ego"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "tracks.parquet"
        n = avc.write_track_lines(out, tracks=tracks)
        assert n == 3, n  # car + pedestrian + ego
        tbl = pq.read_table(out)
        cols = set(tbl.column_names)
        assert {"geometry", "timestamp", "end_timestamp", "vertex_timestamps",
                "vertex_values", "category"} <= cols, cols
        cats = set(tbl.column("category").to_pylist())
        assert "ego" in cats, cats
        # vertex_timestamps is a List<Int64>, aligned to the coords (load-bearing).
        vts = tbl.column("vertex_timestamps").to_pylist()
        cat_col = tbl.column("category").to_pylist()
        ego_row = cat_col.index("ego")
        assert vts[ego_row] == [1000, 2000, 3000], vts[ego_row]
        # timestamp/end_timestamp = first/last vertex time.
        t0 = tbl.column("timestamp").to_pylist()[ego_row]
        t1 = tbl.column("end_timestamp").to_pylist()[ego_row]
        assert (t0, t1) == (1000, 3000), (t0, t1)
    print("PASS test_write_track_lines_roundtrip_with_ego")


def test_write_lidar_points_world_columns_roundtrip():
    """write_lidar_points writes world_class (categorical) + is_dynamic (numeric)."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    n = 6
    lon = np.linspace(0, 0.001, n)
    lat = np.linspace(0, 0.001, n)
    ts = np.arange(n, dtype="int64") + 1000
    z = np.linspace(-1, 3, n)
    wc = np.array(["static", "static", "static", "dynamic", "dynamic", "static"],
                  dtype=object)
    idn = np.array([0, 0, 0, 1, 1, 0], dtype="int64")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "world.parquet"
        avc.write_lidar_points(out, lon=lon, lat=lat, timestamp=ts, z=z,
                               world_class=wc, is_dynamic=idn)
        tbl = pq.read_table(out)
        assert "world_class" in tbl.column_names and "is_dynamic" in tbl.column_names
        # world_class is a STRING column (not promoted to numeric).
        assert pa.types.is_string(tbl.schema.field("world_class").type)
        assert pa.types.is_integer(tbl.schema.field("is_dynamic").type)
        assert tbl.column("world_class").to_pylist() == list(wc)
        assert tbl.column("is_dynamic").to_pylist() == idn.tolist()
    print("PASS test_write_lidar_points_world_columns_roundtrip")


def test_write_lidar_points_scan_phase_roundtrip():
    import pyarrow.parquet as pq
    import pyarrow as pa

    n = 5
    lon = np.linspace(0, 0.001, n)
    lat = np.linspace(0, 0.001, n)
    ts = np.arange(n, dtype="int64") + 1000
    z = np.zeros(n)
    phase = np.linspace(0, 1, n)
    rgb = np.column_stack([phase * 255, np.zeros(n), (1 - phase) * 255])
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "scan.parquet"
        avc.write_lidar_points(out, lon=lon, lat=lat, timestamp=ts, z=z,
                               rgb=rgb, scan_phase=phase)
        tbl = pq.read_table(out)
        assert "scan_phase" in tbl.column_names
        assert pa.types.is_floating(tbl.schema.field("scan_phase").type)
        got = np.asarray(tbl.column("scan_phase").to_pylist())
        assert np.allclose(got, phase), got
        # rgb path still wrote r/g/b.
        assert {"r", "g", "b"} <= set(tbl.column_names)
    print("PASS test_write_lidar_points_scan_phase_roundtrip")


def test_phase_to_rgb_ramp():
    """Feature 5: scan_phase → an HSV rainbow (red → ... → red), uint8 (N,3)."""
    phase = np.array([0.0, 1.0 / 6, 1.0 / 3, 0.5, 2.0 / 3, 5.0 / 6, 1.0])
    rgb = avc.phase_to_rgb(phase)
    assert rgb.shape == (7, 3) and rgb.dtype == np.uint8
    # phase 0 → red, phase 1 → red (loop closes).
    assert tuple(rgb[0]) == (255, 0, 0), rgb[0]
    assert tuple(rgb[-1]) == (255, 0, 0), rgb[-1]
    # phase 1/3 → green-ish (g dominant), 2/3 → blue-ish (b dominant).
    assert rgb[2][1] >= 200 and rgb[2][0] < 60, rgb[2]
    assert rgb[4][2] >= 200 and rgb[4][0] < 60, rgb[4]
    # monotone-ish hue: not all the same colour.
    assert len({tuple(c) for c in rgb}) >= 5
    print("PASS test_phase_to_rgb_ramp")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    try:
        _run_all()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
