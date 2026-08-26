#!/usr/bin/env python3
"""Extract one Waymo Open Dataset (Perception v2.0.1) segment into an AV scene bundle.

Waymo's Perception dataset v2.0.1 is the *modular* format: each "component"
(lidar, lidar_box, vehicle_pose, camera_image, …) is an Apache Parquet file, one
per ~20 s driving *segment*, joined on ``key.segment_context_name`` +
``key.frame_timestamp_micros``. Crucially this means **no TensorFlow / no
waymo-open-dataset library** is needed — we read the Parquet directly with
pyarrow and decode the LIDAR range images in pure numpy (the official spherical
convention from ``range_image_utils``). That's the whole reason we picked v2.0.1
over the v1.4.x TFRecord release (whose reader ships Linux-only wheels).

This adapter emits the cockpit stream set MINUS the HD map:

* ``lidar``   — all 5 lasers' range images → point cloud (height_band colored,
  like Argoverse 2; Waymo's 3D-semseg labels cover only ~20/199 frames so
  per-point semantic coloring isn't viable for smooth playback). Built at several
  DENSITY TIERS (``lidar/`` default + ``lidar-med/`` / ``lidar-high/`` /
  ``lidar-ultra/``) for the cockpit's runtime density selector — see
  ``LIDAR_DENSITY_TIERS``,
* ``ego``     — the vehicle trajectory (``vehicle_pose.world_from_vehicle``),
* ``objects`` — 3D box tracks (``lidar_box``; vehicle-frame center/size/heading +
  Waymo's REAL per-box velocity, so the cockpit's velocity arrows are exact, not
  finite-differenced), carrying ``num_interior_pts`` for ghost-box dropping,
* ``camera``  — the FRONT camera, decimated to ~20 keyframes,
* ``telemetry`` — DERIVED from the ego pose (speed/accel/yaw-rate/heading); Waymo
  Perception ships no CAN bus, same situation as AV2.

────────────────────────────────────────────────────────────────────────────
DELIBERATE DEVIATIONS FROM THE av-cockpit.md CONTRACT (Waymo affordances differ)
────────────────────────────────────────────────────────────────────────────
* **No georeferencing.** Unlike nuScenes (documented SW map origins) and AV2
  (city UTM), Waymo Perception poses live in an *arbitrary* local "world" frame
  with NO disclosed lat/lon, offset ~48 km from its own origin. So we (a) subtract
  the first frame's world translation to recover a local ENU metric frame, then
  (b) anchor that frame at a *plausible but approximate* lat/lon by ``location``
  (SF / PHX / other from the ``stats`` component). The lidar therefore will NOT
  line up with real basemap streets — so Waymo scenes are meant to render on a
  DARK / neutral basemap (configured frontend-side in ``datasets.ts``), where
  there are no streets to contradict. ``scene.json.location`` is labelled
  honestly (e.g. ``"San Francisco (Waymo world frame, anchored)"``).
* **No HD map** (the v2.0.1 "modular without maps" trade-off) → no
  ``map_poly``/``map_line`` streams. The cockpit already treats those as optional.
* **Vehicle taxonomy collapses to ``car``.** Waymo labels only VEHICLE /
  PEDESTRIAN / SIGN / CYCLIST (no truck/bus subdivision); we map VEHICLE→car,
  CYCLIST→bicycle, SIGN→barrier (roadside furniture, as AV2 maps signs), and the
  box SIZES still distinguish a bus from a sedan even though both read "car".

────────────────────────────────────────────────────────────────────────────
DOWNLOAD (license-gated; Waymo Dataset License Agreement — non-commercial, NO
redistribution; accept at waymo.com/open, then ``gcloud auth login``)
────────────────────────────────────────────────────────────────────────────
  gcloud storage cp \\
    "gs://waymo_open_dataset_v_2_0_1/validation/{lidar,lidar_calibration,lidar_box,\\
vehicle_pose,camera_image,stats}/<segment>.parquet" \\
    waymo-raw/validation/<component>/

Then (from scripts/data-generation, with venv-waymo active):

  venv-waymo/bin/python waymo_extract.py --seg <segment_context_name> \\
    --out ../../data-fleet/waymo-<short>

Use ``--skip-build`` to stop at GeoParquet + JSON.

────────────────────────────────────────────────────────────────────────────
RANGE-IMAGE → POINT-CLOUD (verified against waymo_open_dataset range_image_utils)
────────────────────────────────────────────────────────────────────────────
* Range image is ``[H, W, C]`` flat float; channels ``[range, intensity,
  elongation, is_in_no_label_zone]``. We keep ``range > 0`` returns and drop NLZ.
* Per-row INCLINATION: TOP laser ships 64 explicit ``beam_inclination.values``;
  the 4 side lasers ship only ``[min, max]`` → linear ``min+(max-min)*(0.5+i)/H``.
  Both are REVERSED so range-image row 0 = the highest beam (matches the devkit).
* Per-column AZIMUTH: ``az = (((W-0.5-col)/W)*2 - 1)*π - az_correction`` where
  ``az_correction = atan2(extrinsic[1,0], extrinsic[0,0])`` (the sensor's yaw).
* Cartesian (sensor frame): ``x=r·cosα·cosθ, y=r·sinα·cosθ, z=r·sinθ`` then the
  4×4 ``extrinsic`` rotates+translates sensor→vehicle. (We skip the per-pixel
  ``lidar_pose`` rolling-shutter compensation — a sub-metre refinement at urban
  speeds, invisible once points are bucketed at 100 ms.)
* x,y are then lifted vehicle→world (per-frame ``world_from_vehicle``) and
  origin-subtracted → local metres → lon/lat. z keeps the VEHICLE-frame height
  (height-above-ego-ground), so the ``height_band`` ramp reads consistently even
  as the ego drives over terrain — identical to argoverse_extract's z handling.

Pipeline:  Waymo v2.0.1 Parquet  →  GeoParquet + JSON  →  stt-build  →  bundle.
"""

from __future__ import annotations

import argparse
import collections
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

import av_common as avc

# ── Waymo enums ──────────────────────────────────────────────────────────────
# Object type (LiDARBoxComponent.type) → canonical 10-class taxonomy. Waymo only
# labels these four classes; vehicles aren't subdivided (truck/bus read "car" but
# keep their real box size). SIGN folds into barrier (roadside furniture), exactly
# as argoverse_extract maps its sign/bollard classes.
WAYMO_TYPE_TO_CATEGORY = {
    0: avc.OTHER_CATEGORY,  # TYPE_UNKNOWN
    1: "car",               # TYPE_VEHICLE
    2: "pedestrian",        # TYPE_PEDESTRIAN
    3: "barrier",           # TYPE_SIGN
    4: "bicycle",           # TYPE_CYCLIST
}

# camera_name enum — 1 = FRONT (the cockpit inset).
CAMERA_FRONT = 1

# All five Waymo cameras (name enum → label), used by the optional
# image-projection colorizer (--colorize). FRONT + the two front-corners + the
# two sides cover ~252° around the car; the ~108° rear wedge has NO camera, so
# those returns can't be colored and fall back to FALLBACK_RGB.
CAMERA_NAMES: dict[int, str] = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}
# Cool slate-grey for returns no camera can see (the rear wedge) — the shared
# av_common constant, dim enough that the camera-colored front/sides read as
# the "real" part of the cloud.
FALLBACK_RGB = avc.FALLBACK_RGB
# Returns closer than this (m) to the camera are skipped — they're on the ego
# vehicle / its shadow and would smear a single pixel's color across the hood.
MIN_CAM_DEPTH_M = 1.0

# Plausible (NOT exact — Waymo discloses no geo) per-location anchor lat/lon. Only
# the metro is right; the streets are not. Used purely as the local-ENU anchor.
LOCATION_ANCHORS: dict[str, tuple[float, float]] = {
    "location_sf": (37.7749, -122.4194),    # San Francisco
    "location_phx": (33.4484, -112.0740),   # Phoenix
    "location_other": (37.3861, -122.0839),  # Mountain View (catch-all)
}
DEFAULT_ANCHOR = LOCATION_ANCHORS["location_other"]

# LIDAR DENSITY TIERS. The cockpit ships a runtime density selector, so each scene
# builds several lidar archives at increasing point counts; the user A/B's them at
# /drive/<id>. We decode the full sweep ONCE at the finest tier (`FINEST_DECIMATE`
# = 1 ⇒ every valid first-return, ~29M pts/scene) then stride that cloud down to the
# coarser tiers. `small` (1/24 ≈ 1.2M) is the DEFAULT `lidar/` archive; `full` (1/1,
# the complete first-return ~29M) is the ceiling — deck renders even ~29M billboards
# fine, so all five tiers ship for every scene. (The 2nd lidar return isn't read.)
FINEST_DECIMATE = 1
# (id, label, decimate, archive_dir, radius_px, radius_min_px) — ASCENDING density;
# `small` → the default `lidar/`. DENSER tiers use SMALLER points so the cloud reads
# as 3D structure instead of a solid mass (radius ≈ 1.0/√(density / small)).
LIDAR_DENSITY_TIERS = [
    ("small", "Small", 24, "lidar", 1.0, 0.8),          # ~1.2M  (default)
    ("medium", "Medium", 12, "lidar-med", 0.8, 0.65),   # ~2.4M
    ("large", "Large", 6, "lidar-high", 0.65, 0.55),    # ~4.8M
    ("ultra", "Ultra", 3, "lidar-ultra", 0.5, 0.45),    # ~9.5M
    ("full", "Full", FINEST_DECIMATE, "lidar-full", 0.4, 0.35),  # ~28.5M  (full 1st return)
]
# Build-size levers (the cockpit is a fixed street-level view, so the format can be
# tuned tight): clamp the bottom of the zoom pyramid — the cockpit opens at z18 and
# never zooms below a city block, so the z<13 tiers are pure wasted bytes (the "z0
# bomb") — and quantize coordinates to a few cm (viz-exact for a point cloud). Both
# are decode-free on the client; together they ~halve every tier vs a naive build.
LIDAR_MIN_ZOOM = 14  # ~2.4 km tile; the scene (~160 m) sits in one tile below this
LIDAR_QUANTIZE_M = 0.05  # ~3.75 cm grid (av_common warns to keep AV quantize <= 0.1)

# FRONT camera runs 10 Hz → ~199 frames; keep ~20 keyframes for the inset.
CAMERA_DECIMATE = 10


# ── component paths ──────────────────────────────────────────────────────────
def _component_path(raw_dir: Path, split: str, component: str, seg: str) -> Path:
    return raw_dir / split / component / f"{seg}.parquet"


# ── range-image decode (pure numpy; see module docstring) ────────────────────
def decode_range_image(values: np.ndarray, shape, ext: np.ndarray,
                       incl_values, incl_min: float, incl_max: float):
    """One laser's flat range-image ``values`` → (vehicle-frame Nx3, intensity N).

    ``values`` is the flat ``[H*W*C]`` float array, ``shape`` is ``[H, W, C]``,
    ``ext`` the 4×4 sensor→vehicle extrinsic. Returns only valid (range>0,
    non-NLZ) returns.
    """
    H, W, C = int(shape[0]), int(shape[1]), int(shape[2])
    ri = np.asarray(values, dtype=np.float64).reshape(H, W, C)
    rng = ri[..., 0]
    intensity = ri[..., 1]

    if incl_values is not None and len(incl_values) == H:
        incl = np.asarray(incl_values, dtype=np.float64)
    else:  # side lasers: linear between [min, max]
        incl = incl_min + (incl_max - incl_min) * (0.5 + np.arange(H)) / H
    incl = incl[::-1]  # range-image row 0 = highest beam (devkit convention)

    az_correction = math.atan2(ext[1, 0], ext[0, 0])
    ratios = (W - 0.5 - np.arange(W)) / W
    azimuth = (ratios * 2.0 - 1.0) * math.pi - az_correction  # [W]

    cos_i = np.cos(incl)[:, None]
    sin_i = np.sin(incl)[:, None]
    x = rng * np.cos(azimuth)[None, :] * cos_i
    y = rng * np.sin(azimuth)[None, :] * cos_i
    z = rng * sin_i
    pts = np.stack([x, y, z], axis=-1) @ ext[:3, :3].T + ext[:3, 3]  # sensor→vehicle

    mask = rng > 0
    if C >= 4:
        mask &= ri[..., 3] <= 0  # drop no-label-zone returns
    return pts[mask], intensity[mask]


def load_calibration(path: Path) -> dict:
    """laser_name → {ext (4×4), incl_values|None, incl_min, incl_max}."""
    cal = {}
    for r in pq.read_table(path).to_pylist():
        ln = r["key.laser_name"]
        ext = np.asarray(r["[LiDARCalibrationComponent].extrinsic.transform"],
                         dtype=np.float64).reshape(4, 4)
        vals = r["[LiDARCalibrationComponent].beam_inclination.values"]
        cal[ln] = dict(
            ext=ext,
            incl_values=(np.asarray(vals, dtype=np.float64) if vals else None),
            incl_min=r["[LiDARCalibrationComponent].beam_inclination.min"],
            incl_max=r["[LiDARCalibrationComponent].beam_inclination.max"],
        )
    return cal


# ── camera projection colorizer (optional, --colorize) ───────────────────────
def load_camera_calibrations(path: Path) -> dict:
    """camera_name → calibration for projecting vehicle-frame points into pixels.

    Waymo's camera model (``CameraCalibrationComponent``): a pinhole with
    radial-tangential distortion ``[k1,k2,p1,p2,k3]`` and the SENSOR convention
    that the optical axis is the camera's **+x** (so a point's depth is its x in
    camera frame, NOT z). ``extrinsic.transform`` is the 4×4 camera→vehicle pose;
    we cache its rotation ``R`` (3×3) and translation ``t`` so vehicle→camera is
    ``pc = (pv - t) @ R`` (i.e. ``Rᵀ(pv - t)``).
    """
    cal: dict[int, dict] = {}
    K = "[CameraCalibrationComponent]"
    for r in pq.read_table(path).to_pylist():
        ext = np.asarray(r[f"{K}.extrinsic.transform"], dtype=np.float64).reshape(4, 4)
        cal[int(r["key.camera_name"])] = dict(
            f_u=float(r[f"{K}.intrinsic.f_u"]), f_v=float(r[f"{K}.intrinsic.f_v"]),
            c_u=float(r[f"{K}.intrinsic.c_u"]), c_v=float(r[f"{K}.intrinsic.c_v"]),
            k1=float(r[f"{K}.intrinsic.k1"]), k2=float(r[f"{K}.intrinsic.k2"]),
            p1=float(r[f"{K}.intrinsic.p1"]), p2=float(r[f"{K}.intrinsic.p2"]),
            k3=float(r[f"{K}.intrinsic.k3"]),
            R=ext[:3, :3], t=ext[:3, 3],
            width=int(r[f"{K}.width"]), height=int(r[f"{K}.height"]),
        )
    return cal


def load_camera_images(path: Path) -> dict:
    """``(camera_name, ts_micros)`` → JPEG bytes, for ALL five cameras.

    Holds the compressed bytes (~200 MB for a 20 s segment) and decodes to pixels
    lazily + LRU-cached in ``CameraColorizer`` — far cheaper than holding ~990
    decoded 1920×1280×3 frames (~37 GB) at once.
    """
    store: dict[tuple[int, int], bytes] = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=16,
                                 columns=["key.frame_timestamp_micros",
                                          "key.camera_name",
                                          "[CameraImageComponent].image"]):
        ts_col = batch.column(0).to_numpy()
        cam_col = batch.column(1).to_numpy()
        img_col = batch.column(2)
        for k in range(batch.num_rows):
            store[(int(cam_col[k]), int(ts_col[k]))] = img_col[k].as_py()
    return store


def project_to_pixels(pts_vehicle: np.ndarray, c: dict, width: int, height: int):
    """Vehicle-frame pts (N×3) → (ui, vi, r2, valid) for one camera calibration.

    Waymo's pinhole-with-distortion model, optical axis = camera **+x**. Shared by
    ``CameraColorizer.colorize`` (production) and ``debug_overlay`` (the
    projection-correctness check) so both exercise the SAME math.
    """
    # vehicle → camera frame: pc = Rᵀ(pv - t)  ⇔  (pv - t) @ R
    pc = (pts_vehicle - c["t"]) @ c["R"]
    x = pc[:, 0]  # optical axis = camera +x ⇒ depth
    in_front = x > MIN_CAM_DEPTH_M
    xs = np.where(in_front, x, 1.0)  # guard the division
    u_n = -pc[:, 1] / xs  # image +u (right) = camera -y (left is +y)
    v_n = -pc[:, 2] / xs  # image +v (down)  = camera -z (up is +z)
    r2 = u_n * u_n + v_n * v_n
    radial = 1.0 + c["k1"] * r2 + c["k2"] * r2 * r2 + c["k3"] * r2 * r2 * r2
    u_d = u_n * radial + 2.0 * c["p1"] * u_n * v_n + c["p2"] * (r2 + 2.0 * u_n * u_n)
    v_d = v_n * radial + c["p1"] * (r2 + 2.0 * v_n * v_n) + 2.0 * c["p2"] * u_n * v_n
    ui = np.round(c["f_u"] * u_d + c["c_u"]).astype(np.int64)
    vi = np.round(c["f_v"] * v_d + c["c_v"]).astype(np.int64)
    valid = in_front & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return ui, vi, r2, valid


class CameraColorizer:
    """Projects vehicle-frame LIDAR returns into the camera images to sample RGB.

    Per frame ``ts``, each return is projected into every camera that captured
    that frame; a return visible to several cameras takes the color of the one
    that sees it most CENTRALLY (smallest normalized radius r² ⇒ least lens
    distortion, most reliable color). Returns no camera sees keep ``got=False``.
    """

    def __init__(self, cal: dict, store: dict):
        self.cal = cal
        self.store = store
        self._cache: "collections.OrderedDict[tuple[int, int], np.ndarray]" = \
            collections.OrderedDict()

    def image(self, cam: int, ts: int):
        """Decode (LRU-cached, ≤12 frames) the JPEG for one camera+frame → HxWx3."""
        from io import BytesIO
        from PIL import Image

        key = (cam, ts)
        arr = self._cache.get(key)
        if arr is not None:
            self._cache.move_to_end(key)
            return arr
        raw = self.store.get(key)
        if raw is None:
            return None
        arr = np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        self._cache[key] = arr
        if len(self._cache) > 12:
            self._cache.popitem(last=False)
        return arr

    def colorize(self, pts_vehicle: np.ndarray, ts: int):
        """(N×3 vehicle-frame pts, frame ts) → (rgb uint8 N×3, got bool N)."""
        n = len(pts_vehicle)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        best_r2 = np.full(n, np.inf)
        got = np.zeros(n, dtype=bool)
        for cam, c in self.cal.items():
            img = self.image(cam, ts)
            if img is None:
                continue
            ui, vi, r2, valid = project_to_pixels(pts_vehicle, c, img.shape[1], img.shape[0])
            better = valid & (r2 < best_r2)
            if better.any():
                bi = np.where(better)[0]
                rgb[bi] = img[vi[bi], ui[bi]]
                best_r2[bi] = r2[bi]
                got[bi] = True
        return rgb, got


def decode_frame_lidar(path: Path, cal: dict, ts: int) -> np.ndarray:
    """All lasers of ONE frame → vehicle-frame pts (N×3), no decimation.

    Used by the debug overlay to project a single dense sweep into the cameras.
    """
    VAL = "[LiDARComponent].range_image_return1.values"
    SH = "[LiDARComponent].range_image_return1.shape"
    chunks = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=8,
                                 columns=["key.frame_timestamp_micros",
                                          "key.laser_name", VAL, SH]):
        ts_col = batch.column(0).to_numpy()
        laser_col = batch.column(1).to_numpy()
        val_list = batch.column(2)
        flat = val_list.values.to_numpy(zero_copy_only=False)
        offs = val_list.offsets.to_numpy()
        shapes = batch.column(3).to_pylist()
        for k in range(batch.num_rows):
            if int(ts_col[k]) != ts:
                continue
            c = cal.get(int(laser_col[k]))
            if c is None:
                continue
            pts, _ = decode_range_image(flat[offs[k]:offs[k + 1]], shapes[k], c["ext"],
                                        c["incl_values"], c["incl_min"], c["incl_max"])
            if len(pts):
                chunks.append(pts)
    return np.concatenate(chunks) if chunks else np.empty((0, 3))


def debug_overlay(out: Path, lidar_path: Path, lidar_cal: dict,
                  colorizer: "CameraColorizer", frame_ts: list[int], frame_index: int):
    """Project one frame's full sweep into every camera, dots colored by DEPTH.

    Saves ``<out>/_debug_overlay_<CAM>.png`` per camera — a projection-correctness
    check: if the depth-tinted dots land ON the cars / road / buildings in the
    photo, the extrinsic + intrinsic + sign conventions are right.
    """
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    idx = max(0, min(frame_index, len(frame_ts) - 1))
    ts = frame_ts[idx]
    pts = decode_frame_lidar(lidar_path, lidar_cal, ts)
    print(f"  debug overlay: frame {idx} (ts {ts}), {len(pts)} returns")
    for cam, c in sorted(colorizer.cal.items()):
        img = colorizer.image(cam, ts)
        if img is None:
            continue
        ui, vi, _r2, valid = project_to_pixels(pts, c, img.shape[1], img.shape[0])
        depth = ((pts - c["t"]) @ c["R"])[:, 0]
        canvas = img.copy()
        bi = np.where(valid)[0]
        if len(bi):
            colors = avc.depth_ramp(depth[bi], depth[bi].min(), np.percentile(depth[bi], 95))
            h, w = canvas.shape[0], canvas.shape[1]
            for dy in (-1, 0, 1):  # 3×3 splat so the dots read at full res
                for dx in (-1, 0, 1):
                    yy = np.clip(vi[bi] + dy, 0, h - 1)
                    xx = np.clip(ui[bi] + dx, 0, w - 1)
                    canvas[yy, xx] = colors
        name = CAMERA_NAMES.get(cam, str(cam))
        path = out / f"_debug_overlay_{name}.png"
        Image.fromarray(canvas).save(path)
        print(f"    wrote {path}  ({len(bi)} projected returns)")


def _yaw_of(R: np.ndarray) -> float:
    """Yaw (rad, CCW about +z) of a 3×3 rotation."""
    # Per-dataset ON PURPOSE (not an av_common hoist): Waymo poses are 4×4
    # matrices; argoverse_extract._yaw_of takes vectorised quaternion columns,
    # nuscenes_extract._yaw_of a devkit quaternion record.
    return math.atan2(R[1, 0], R[0, 0])


# ── ego / pose ───────────────────────────────────────────────────────────────
def load_poses(path: Path):
    """Sorted ego frames + a per-frame pose lookup.

    Returns ``(frame_ts[list], pose_by_ts{ts: (4×4 world_from_vehicle, yaw)},
    origin[3])`` where ``origin`` is frame-0's world translation (subtracted from
    every world position to recover a local metric frame — Waymo's world frame is
    offset ~48 km from its origin).
    """
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda r: r["key.frame_timestamp_micros"])
    pose_by_ts: dict[int, tuple[np.ndarray, float]] = {}
    frame_ts: list[int] = []
    for r in rows:
        ts = r["key.frame_timestamp_micros"]
        T = np.asarray(r["[VehiclePoseComponent].world_from_vehicle.transform"],
                       dtype=np.float64).reshape(4, 4)
        pose_by_ts[ts] = (T, _yaw_of(T[:3, :3]))
        frame_ts.append(ts)
    origin = pose_by_ts[frame_ts[0]][0][:3, 3].copy()
    return frame_ts, pose_by_ts, origin


def extract_ego(frame_ts, pose_by_ts, origin, to_lonlat):
    """Ego trajectory → (t_ms, lon, lat, speed, yaw) arrays (world-local→lon/lat)."""
    t_ms = np.asarray([ts // 1000 for ts in frame_ts], dtype="int64")
    wx = np.asarray([pose_by_ts[ts][0][0, 3] for ts in frame_ts]) - origin[0]
    wy = np.asarray([pose_by_ts[ts][0][1, 3] for ts in frame_ts]) - origin[1]
    yaw = np.asarray([pose_by_ts[ts][1] for ts in frame_ts])
    lon, lat = to_lonlat(wx, wy)
    dt = np.gradient(t_ms / 1000.0)
    speed = np.hypot(np.gradient(wx), np.gradient(wy)) / np.maximum(np.abs(dt), 1e-3)
    return t_ms, np.asarray(lon), np.asarray(lat), speed, yaw


# ── objects ──────────────────────────────────────────────────────────────────
def extract_objects(path: Path, pose_by_ts, origin, to_lonlat, drop_empty=True):
    """lidar_box → the ``write_objects_points`` kwargs dict (vehicle→world→lon/lat).

    Per box: center lifted vehicle→world via its frame's pose, heading += ego yaw,
    speed = |Waymo per-box velocity| (real, not finite-differenced). Zero-LIDAR-
    point (occluded/ghost) boxes dropped by default.
    """
    rows = pq.read_table(path).to_pylist()
    o = collections.defaultdict(list)
    wx, wy = [], []
    n_dropped = 0
    for r in rows:
        ts = r["key.frame_timestamp_micros"]
        if ts not in pose_by_ts:
            continue
        nip = int(r["[LiDARBoxComponent].num_lidar_points_in_box"])
        if drop_empty and nip == 0:
            n_dropped += 1
            continue
        T, ego_yaw = pose_by_ts[ts]
        cx = r["[LiDARBoxComponent].box.center.x"]
        cy = r["[LiDARBoxComponent].box.center.y"]
        cz = r["[LiDARBoxComponent].box.center.z"]
        pw = T[:3, :3] @ np.array([cx, cy, cz]) + T[:3, 3]
        wx.append(pw[0] - origin[0])
        wy.append(pw[1] - origin[1])
        o["timestamp"].append(ts // 1000)
        o["category"].append(WAYMO_TYPE_TO_CATEGORY.get(r["[LiDARBoxComponent].type"],
                                                         avc.OTHER_CATEGORY))
        o["heading"].append(float(r["[LiDARBoxComponent].box.heading"] + ego_yaw))
        o["length"].append(float(r["[LiDARBoxComponent].box.size.x"]))  # x = length
        o["width"].append(float(r["[LiDARBoxComponent].box.size.y"]))   # y = width
        o["height"].append(float(r["[LiDARBoxComponent].box.size.z"]))  # z = height
        o["track_id"].append(str(r["key.laser_object_id"]))
        o["speed"].append(float(math.hypot(r["[LiDARBoxComponent].speed.x"],
                                            r["[LiDARBoxComponent].speed.y"])))
        o["num_interior_pts"].append(nip)
    if drop_empty:
        print(f"  objects: dropped {n_dropped} zero-point (occluded/ghost) box(es)")
    lon, lat = to_lonlat(np.asarray(wx), np.asarray(wy))
    return dict(
        lon=[float(v) for v in np.atleast_1d(lon)],
        lat=[float(v) for v in np.atleast_1d(lat)],
        timestamp=o["timestamp"], category=o["category"], heading=o["heading"],
        length=o["length"], width=o["width"], height=o["height"],
        track_id=o["track_id"], speed=o["speed"],
        num_interior_pts=o["num_interior_pts"],
    )


# ── lidar (streamed range-image decode, memory-bounded) ──────────────────────
def extract_lidar(path: Path, cal: dict, pose_by_ts, origin, to_lonlat,
                  decimate: int = FINEST_DECIMATE, colorizer: "CameraColorizer | None" = None):
    """All lasers' range images → decimated (lon, lat, t_ms, z, intensity[, rgb]).

    Streamed in small Parquet batches (one row ≈ one laser-frame range image, a
    few MB) so peak memory stays ~tens of MB rather than loading the whole ~1.5 GB
    decompressed range-image column. x,y go vehicle→world→local→lon/lat; z keeps
    the vehicle-frame height (height-above-ground) for a consistent height_band.

    When ``colorizer`` is given, each decimated return is projected (in its still
    VEHICLE-frame coordinates — same frame the cameras are calibrated to, BEFORE
    the world lift) into the cameras and assigned an RGB color; returns no camera
    sees take ``FALLBACK_RGB``. Returns a 6th array ``rgb`` (N×3 uint8) then, else
    ``None``.
    """
    VAL = "[LiDARComponent].range_image_return1.values"
    SH = "[LiDARComponent].range_image_return1.shape"
    out_lon, out_lat, out_t, out_z, out_i = [], [], [], [], []
    out_rgb = [] if colorizer is not None else None
    fallback = np.asarray(FALLBACK_RGB, dtype=np.uint8)
    pf = pq.ParquetFile(path)
    carry = 0  # running offset so the decimation stride is global, not per-frame
    for batch in pf.iter_batches(batch_size=8,
                                 columns=["key.frame_timestamp_micros",
                                          "key.laser_name", VAL, SH]):
        ts_col = batch.column(0).to_numpy()
        laser_col = batch.column(1).to_numpy()
        val_list = batch.column(2)               # ListArray
        flat = val_list.values.to_numpy(zero_copy_only=False)
        offs = val_list.offsets.to_numpy()
        shapes = batch.column(3).to_pylist()
        for k in range(batch.num_rows):
            ts = int(ts_col[k])
            if ts not in pose_by_ts:
                continue
            ln = int(laser_col[k])
            c = cal.get(ln)
            if c is None:
                continue
            vals = flat[offs[k]:offs[k + 1]]
            pts, inten = decode_range_image(vals, shapes[k], c["ext"],
                                            c["incl_values"], c["incl_min"], c["incl_max"])
            if len(pts) == 0:
                continue
            # global decimation stride across all lasers/frames
            sel = np.arange((-carry) % decimate, len(pts), decimate)
            carry = (carry + len(pts)) % decimate
            if len(sel) == 0:
                continue
            pts = pts[sel]
            inten = inten[sel]
            if colorizer is not None:
                # Project in the VEHICLE frame (cameras are calibrated to it) —
                # must happen before the world lift below.
                rgb, hit = colorizer.colorize(pts, ts)
                rgb[~hit] = fallback
                out_rgb.append(rgb)
            T = pose_by_ts[ts][0]
            pw = pts @ T[:3, :3].T + T[:3, 3]
            lon, lat = to_lonlat(pw[:, 0] - origin[0], pw[:, 1] - origin[1])
            out_lon.append(np.atleast_1d(lon))
            out_lat.append(np.atleast_1d(lat))
            out_z.append(pts[:, 2])              # vehicle-frame height
            out_i.append(inten)
            out_t.append(np.full(len(pts), ts // 1000, dtype="int64"))
    rgb_all = np.concatenate(out_rgb) if out_rgb else None
    return (np.concatenate(out_lon), np.concatenate(out_lat),
            np.concatenate(out_t), np.concatenate(out_z), np.concatenate(out_i),
            rgb_all)


# ── surfel covariance (build-time, per-sweep k-NN) ───────────────────────────
# A "formal" splat needs ORIENTED anisotropic primitives, not round dots. For a
# LIDAR scan the principled way to get them WITHOUT training a 3DGS model is
# surface splatting (Pfister/Zwicker surfels): estimate each return's local
# surface frame from its k nearest neighbours' covariance — the smallest
# eigenvector is the surface normal, the two larger span the tangent plane, and
# the eigenvalues give the patch extents. The client's `SplatLayer` then renders
# each return as an oriented Gaussian disk lying on the surface.
#
# Computed PER SWEEP (one frame_timestamp), in the VEHICLE frame — a single
# sweep is a coherent range-image sampling of the surface, so its neighbourhoods
# are real adjacency; mixing frames as the ego moves would smear a wall into a
# slab. The resulting basis is then rotated by the ego pose into the render ENU
# frame (where the disk offsets live). Orientation comes from the FULL-density
# neighbourhood (clean normals); the disk SIZE is scaled to the DECIMATED point
# spacing so the sparser rendered set still tiles the surface without gaps.
SURFEL_K = avc.SURFEL_K  # neighbours for the covariance estimate (shared tuning)
# Default render decimation for the surfel tier. Denser than the round-dot point
# tiers: at low density each SWEEP is too sparse to tile, so the per-point disk
# is sized large and reads as blobs — a denser sample makes each disk small and
# the surface crisp. The soft TEMPORAL Gaussian caps per-frame cost to the ~±3σ
# window around the playhead (not the whole cloud), so density is nearly free at
# render time. ~1/6 of the first-return cloud ≈ 4.8M oriented disks for a 20 s
# segment (~few hundred k drawn at any instant).
SURFEL_DECIMATE = 6


def compute_surfels(pts: np.ndarray, sel: np.ndarray, R_ego: np.ndarray):
    """Per-sweep surfels for the selected returns → (quat N×4, scale2 N×2, opacity N).

    ``pts`` is the FULL-density vehicle-frame sweep (N×3); ``sel`` indexes the
    rendered (decimated) subset. Orientation is estimated from each selected
    point's full-density neighbourhood, then rotated by the ego YAW into the
    render frame. (Yaw only, NOT the full ``R_ego``: the cloud is rendered with
    world x,y but VEHICLE-frame z — a deliberate stable-height simplification —
    so a full-pose rotation would tilt ground disks by the ego pitch/roll off the
    flattened rendered ground, which reads as shimmer at high density on hilly
    scenes like SF.) Disk extents are scaled to the DECIMATED spacing so the
    sparser rendered cloud still tiles the surface. Confidence (alpha) tracks
    local planarity. Degenerate neighbourhoods fall back to a flat, up-facing,
    low-confidence disk.

    Delegates the eigen→quaternion + extents + opacity CORE to the shared
    ``av_common._surfel_frame`` (so the per-sweep and the merged Worldbuild paths
    fit surfels identically — the dual-copy hazard this repo keeps hitting). The
    view direction is toward the sensor at the vehicle origin (``-sel_pts``) and
    the ego-YAW matrix is passed as ``R_post``, so the output is bit-for-bit what
    this function produced before the refactor.
    """
    yaw = math.atan2(R_ego[1, 0], R_ego[0, 0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return avc._surfel_frame(pts, sel, -pts[sel], R_post=R_yaw)


def extract_lidar_surfels(path: Path, cal: dict, pose_by_ts, origin, to_lonlat,
                          colorizer: "CameraColorizer", decimate: int = SURFEL_DECIMATE,
                          adaptive: bool = False):
    """Per-sweep surfel cloud → (lon, lat, t_ms, z, intensity, rgb, surfels N×7).

    Streams the range images grouped by ``frame_timestamp`` (the Waymo lidar
    Parquet is written per frame, so a sweep's five lasers arrive together); per
    sweep it decodes all lasers at FULL density, fits surfels for the decimated
    subset (full-density neighbourhoods, see ``compute_surfels``), camera-colours
    that subset, and lifts it vehicle→world→lon/lat. ``surfels`` is the
    ``[qx,qy,qz,qw,s_major,s_minor,opacity]`` block for ``write_lidar_points``.
    """
    VAL = "[LiDARComponent].range_image_return1.values"
    SH = "[LiDARComponent].range_image_return1.shape"
    out_lon, out_lat, out_t, out_z, out_i, out_rgb, out_sf = [], [], [], [], [], [], []
    fallback = np.asarray(FALLBACK_RGB, dtype=np.uint8)

    def flush(ts: int, parts: list, iparts: list):
        if ts not in pose_by_ts or not parts:
            return
        full = np.concatenate(parts)
        inten = np.concatenate(iparts)
        if len(full) < SURFEL_K + 2:
            return
        sel = (avc.adaptive_lidar_select(full, decimate) if adaptive
               else np.arange(0, len(full), decimate))
        if len(sel) == 0:
            return
        T = pose_by_ts[ts][0]
        R_ego = T[:3, :3]
        sel_pts = full[sel]
        rgb, hit = colorizer.colorize(sel_pts, ts)
        rgb[~hit] = fallback
        quat, scale2, op = compute_surfels(full, sel, R_ego)
        pw = sel_pts @ R_ego.T + T[:3, 3]
        lon, lat = to_lonlat(pw[:, 0] - origin[0], pw[:, 1] - origin[1])
        out_lon.append(np.atleast_1d(lon))
        out_lat.append(np.atleast_1d(lat))
        out_t.append(np.full(len(sel), ts // 1000, dtype="int64"))
        out_z.append(sel_pts[:, 2])
        out_i.append(inten[sel])
        out_rgb.append(rgb)
        out_sf.append(np.concatenate([quat, scale2, op[:, None]], axis=1))

    cur_ts = None
    parts: list = []
    iparts: list = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=8,
                                 columns=["key.frame_timestamp_micros",
                                          "key.laser_name", VAL, SH]):
        ts_col = batch.column(0).to_numpy()
        laser_col = batch.column(1).to_numpy()
        val_list = batch.column(2)
        flat = val_list.values.to_numpy(zero_copy_only=False)
        offs = val_list.offsets.to_numpy()
        shapes = batch.column(3).to_pylist()
        for k in range(batch.num_rows):
            ts = int(ts_col[k])
            if ts != cur_ts:
                flush(cur_ts, parts, iparts)
                cur_ts, parts, iparts = ts, [], []
            c = cal.get(int(laser_col[k]))
            if c is None:
                continue
            pts, inten = decode_range_image(flat[offs[k]:offs[k + 1]], shapes[k], c["ext"],
                                            c["incl_values"], c["incl_min"], c["incl_max"])
            if len(pts):
                parts.append(pts)
                iparts.append(inten)
    flush(cur_ts, parts, iparts)  # final sweep

    if not out_lon:
        raise SystemExit("surfel extraction produced no points")
    return (np.concatenate(out_lon), np.concatenate(out_lat),
            np.concatenate(out_t), np.concatenate(out_z), np.concatenate(out_i),
            np.concatenate(out_rgb), np.concatenate(out_sf))


# ── camera (FRONT inset, decimated) ──────────────────────────────────────────
def extract_camera(path: Path, out: Path, decimate: int = CAMERA_DECIMATE):
    """Decimate the FRONT camera's JPEGs into ``<out>/cam/`` → [{t, url}, …] or None."""
    frames = []  # (ts_micros, jpeg_bytes)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=16,
                                 columns=["key.frame_timestamp_micros",
                                          "key.camera_name",
                                          "[CameraImageComponent].image"]):
        ts_col = batch.column(0).to_numpy()
        cam_col = batch.column(1).to_numpy()
        img_col = batch.column(2)
        for k in range(batch.num_rows):
            if int(cam_col[k]) == CAMERA_FRONT:
                frames.append((int(ts_col[k]), img_col[k].as_py()))
    if not frames:
        print("  no FRONT camera frames; omitting camera stream")
        return None
    frames.sort(key=lambda f: f[0])
    idx = list(range(0, len(frames), max(1, decimate)))
    if idx and idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    cam_dir = out / "cam"
    cam_dir.mkdir(parents=True, exist_ok=True)
    out_frames = []
    for j, i in enumerate(idx):
        ts, jpg = frames[i]
        name = f"{j:04d}.jpg"
        (cam_dir / name).write_bytes(jpg)
        out_frames.append({"t": ts // 1000, "url": f"cam/{name}"})
    print(f"  copied {len(out_frames)} FRONT keyframe(s) → {cam_dir}")
    return out_frames


# ── curation helper ──────────────────────────────────────────────────────────
def read_scene_meta(stats_path: Path):
    """(time_of_day, location, weather) modal values from the stats component."""
    rows = pq.read_table(stats_path, columns=[
        "[StatsComponent].time_of_day", "[StatsComponent].location",
        "[StatsComponent].weather"]).to_pylist()

    def modal(col):
        return collections.Counter(r[col] for r in rows).most_common(1)[0][0]

    return (modal("[StatsComponent].time_of_day"),
            modal("[StatsComponent].location"),
            modal("[StatsComponent].weather"))


# ── Worldbuild (--worldbuild): cross-sweep-merged consolidated cloud ─────────
# Build ONE consolidated render-world cloud over the whole drive: STATIC returns
# are voxel-deduped across all sweeps (a wall sampled 100× → one crisp surface)
# while DYNAMIC returns (inside a MOVING actor box at their sweep time) pass
# through UN-merged so a moving car doesn't smear into a solid streak. Streamed
# into av_common.WorldVoxelAccumulator so peak memory ∝ occupied voxels, not the
# ~180M raw returns. See av_common.worldbuild_merge for the column contract.
WORLDBUILD_VOXEL_M = 0.15        # default static-merge voxel edge (m)
WORLDBUILD_STATIC_SPEED = 0.5    # m/s — a track at/above this is "moving"
ERASOR_SCAN_STRIDE = avc.ERASOR_SCAN_STRIDE  # decimate per-sweep ERASOR scans


def extract_moving_boxes_world(path: Path, pose_by_ts, origin, static_speed: float):
    """lidar_box → world-metric moving-box dicts (the dynamic-return test set).

    Mirrors ``extract_objects`` but keeps the box centre in the SAME render-world
    metric frame (``pw - origin``) the worldbuild cloud is accumulated in, and
    keeps ONLY moving boxes (Waymo's real per-box speed ≥ ``static_speed``). Each
    dict is ``{cx, cy, heading, length, width, t}`` (ms) for
    ``av_common.point_in_moving_boxes``.
    """
    rows = pq.read_table(path).to_pylist()
    boxes: list[dict] = []
    for r in rows:
        ts = r["key.frame_timestamp_micros"]
        if ts not in pose_by_ts:
            continue
        spd = float(math.hypot(r["[LiDARBoxComponent].speed.x"],
                               r["[LiDARBoxComponent].speed.y"]))
        if spd < static_speed:
            continue
        T, ego_yaw = pose_by_ts[ts]
        cx = r["[LiDARBoxComponent].box.center.x"]
        cy = r["[LiDARBoxComponent].box.center.y"]
        cz = r["[LiDARBoxComponent].box.center.z"]
        pw = T[:3, :3] @ np.array([cx, cy, cz]) + T[:3, 3]
        boxes.append(dict(
            cx=float(pw[0] - origin[0]), cy=float(pw[1] - origin[1]),
            heading=float(r["[LiDARBoxComponent].box.heading"] + ego_yaw),
            length=float(r["[LiDARBoxComponent].box.size.x"]),
            width=float(r["[LiDARBoxComponent].box.size.y"]),
            t=int(ts // 1000)))
    print(f"  worldbuild: {len(boxes)} moving-box observation(s) "
          f"(speed ≥ {static_speed} m/s)")
    return boxes


def extract_worldbuild(path: Path, cal: dict, pose_by_ts, origin,
                       colorizer: "CameraColorizer", moving_boxes,
                       voxel_m: float = WORLDBUILD_VOXEL_M,
                       collect_scans: bool = False,
                       scan_stride: int = ERASOR_SCAN_STRIDE,
                       color_mode: str = "mean"):
    """Stream-accumulate the worldbuild cloud → (WorldVoxelAccumulator, dynamic).

    Decodes each sweep's five lasers at FULL density (in the vehicle frame), lifts
    x,y vehicle→world→local-metric (z stays vehicle-frame height), camera-colours
    the returns, tags each return dynamic iff it falls inside a moving box at its
    sweep time, folds STATIC returns into the streaming voxel accumulator, and
    collects the DYNAMIC returns un-merged. Peak memory ∝ occupied voxels + the
    (small) dynamic-return set. ``dynamic`` is the ``{xy,z,rgb,t}`` dict for
    ``av_common.worldbuild_merge``.

    Shared by Worldbuild and the scene-split "stage + actors" mode: ``color_mode``
    (default "mean") → "linear_weighted" fuses cross-sweep colour in LINEAR light
    (no washout); ``collect_scans`` also returns each sweep's STATIC returns
    (decimated by ``scan_stride``) + ego xy as ``(xy,z,ego_xy)`` for
    ``av_common.erasor_scrub``. Returns ``(acc, dynamic)`` — or
    ``(acc, dynamic, scans)`` when ``collect_scans``.
    """
    VAL = "[LiDARComponent].range_image_return1.values"
    SH = "[LiDARComponent].range_image_return1.shape"
    acc = avc.WorldVoxelAccumulator(voxel_m, color_mode=color_mode)
    d_xy, d_z, d_rgb, d_t = [], [], [], []
    scans: list = []
    fallback = np.asarray(FALLBACK_RGB, dtype=np.uint8)

    def flush(ts: int, parts: list):
        if ts not in pose_by_ts or not parts:
            return
        full = np.concatenate(parts)
        T = pose_by_ts[ts][0]
        R_ego = T[:3, :3]
        t_ms = int(ts // 1000)
        # camera colour in the VEHICLE frame (cameras calibrated to it).
        if colorizer is not None:
            rgb, hit = colorizer.colorize(full, ts)
            rgb = rgb.astype("float64")
            rgb[~hit] = fallback
        else:
            rgb = np.tile(fallback.astype("float64"), (len(full), 1))
            hit = np.zeros(len(full), dtype=bool)
        # lift x,y vehicle→world→local-metric (z keeps vehicle-frame height).
        pw = full @ R_ego.T + T[:3, 3]
        xw = pw[:, 0] - origin[0]
        yw = pw[:, 1] - origin[1]
        z = full[:, 2]
        xy = np.column_stack([xw, yw])
        dyn = avc.point_in_moving_boxes(xy, z, t_ms, moving_boxes)
        stat = ~dyn
        if stat.any():
            acc.add(xy[stat], z[stat], t_ms, rgb=rgb[stat], rgb_hit=hit[stat])
            if collect_scans:
                ego_xy = np.array([T[0, 3] - origin[0], T[1, 3] - origin[1]],
                                  dtype="float64")
                scans.append((xy[stat][::scan_stride], z[stat][::scan_stride], ego_xy))
        if dyn.any():
            d_xy.append(xy[dyn])
            d_z.append(z[dyn])
            d_rgb.append(rgb[dyn])
            d_t.append(np.full(int(dyn.sum()), t_ms, dtype="int64"))

    cur_ts = None
    parts: list = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=8,
                                 columns=["key.frame_timestamp_micros",
                                          "key.laser_name", VAL, SH]):
        ts_col = batch.column(0).to_numpy()
        laser_col = batch.column(1).to_numpy()
        val_list = batch.column(2)
        flat = val_list.values.to_numpy(zero_copy_only=False)
        offs = val_list.offsets.to_numpy()
        shapes = batch.column(3).to_pylist()
        for k in range(batch.num_rows):
            ts = int(ts_col[k])
            if ts != cur_ts:
                flush(cur_ts, parts)
                cur_ts, parts = ts, []
            c = cal.get(int(laser_col[k]))
            if c is None:
                continue
            pts, _inten = decode_range_image(flat[offs[k]:offs[k + 1]], shapes[k],
                                             c["ext"], c["incl_values"],
                                             c["incl_min"], c["incl_max"])
            if len(pts):
                parts.append(pts)
    flush(cur_ts, parts)  # final sweep

    dynamic = dict(
        xy=np.concatenate(d_xy) if d_xy else np.empty((0, 2)),
        z=np.concatenate(d_z) if d_z else np.empty(0),
        rgb=np.concatenate(d_rgb) if d_rgb else np.empty((0, 3)),
        t=np.concatenate(d_t) if d_t else np.empty(0, dtype="int64"),
    )
    if collect_scans:
        return acc, dynamic, scans
    return acc, dynamic


# ── driver ───────────────────────────────────────────────────────────────────
def generate(args):
    seg = args.seg
    raw, split = args.raw_dir, args.split

    def cpath(component):
        p = _component_path(raw, split, component, seg)
        if not p.exists():
            raise SystemExit(f"missing component file: {p}\n  download it first "
                             f"(see the module docstring).")
        return p

    tod, location, weather = read_scene_meta(cpath("stats"))
    anchor = LOCATION_ANCHORS.get(args.location or location, DEFAULT_ANCHOR)
    if args.origin_lat is not None and args.origin_lon is not None:
        anchor = (args.origin_lat, args.origin_lon)
    anchor_lat, anchor_lon = anchor
    loc_label = {"location_sf": "San Francisco", "location_phx": "Phoenix",
                 "location_other": "USA"}.get(location, location)
    print(f"Waymo segment {seg[:16]}…  {tod} · {loc_label} · {weather}  "
          f"(anchor {anchor_lat:.4f},{anchor_lon:.4f})")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    def to_lonlat(x_m, y_m):  # local ENU metres → lon/lat about the anchor
        return avc.local_to_lonlat(x_m, y_m, anchor_lat, anchor_lon)

    # ego / poses
    frame_ts, pose_by_ts, origin = load_poses(cpath("vehicle_pose"))
    ego_t, ego_lon, ego_lat, ego_speed, ego_yaw = extract_ego(
        frame_ts, pose_by_ts, origin, to_lonlat)
    t_start, t_end = int(ego_t[0]), int(ego_t[-1])

    cal = load_calibration(cpath("lidar_calibration"))

    # --worldbuild / --scene-split are mutually exclusive with --surfel / --contours
    # (and each other) and (like --surfel) force camera colour on.
    scene_split_mode = bool(getattr(args, "scene_split", False))
    if sum((bool(args.worldbuild), scene_split_mode, bool(args.surfel),
            bool(args.contours), bool(args.lod))) > 1:
        raise SystemExit("--worldbuild / --scene-split / --surfel / --contours / --lod "
                         "are mutually exclusive")

    # Optional camera-projection colorize (and its correctness overlay): load the
    # 5 cameras' calibrations + images and build the projector. Both need the
    # camera_calibration component (download it alongside camera_image).
    # --worldbuild / --scene-split imply camera colour (like --surfel).
    colorizer = None
    if (args.colorize or args.surfel or args.worldbuild or scene_split_mode
            or args.debug_overlay is not None):
        print("  loading camera calibrations + images (5 cams) for projection…")
        cam_cal = load_camera_calibrations(cpath("camera_calibration"))
        cam_store = load_camera_images(cpath("camera_image"))
        colorizer = CameraColorizer(cam_cal, cam_store)

    if args.debug_overlay is not None:
        debug_overlay(out, cpath("lidar"), cal, colorizer, frame_ts, args.debug_overlay)
        print("  (--debug-overlay) wrote projection overlays; stopping.")
        return

    # archives
    ego_pq = out / "ego.parquet"
    avc.write_ego_trips(ego_pq, lon=ego_lon, lat=ego_lat,
                        vertex_timestamps=ego_t.tolist(),
                        vertex_values=ego_speed.tolist(), vehicle="ego")

    objects = extract_objects(cpath("lidar_box"), pose_by_ts, origin, to_lonlat,
                              drop_empty=not args.keep_empty_boxes)
    obj_pq = out / "objects.parquet"
    n_objects = avc.write_objects_points(obj_pq, **objects)
    obj_categories = sorted(set(objects["category"]))

    # --- tracks (Feature 2): one LineString per tracked object, ego folded in
    # as category "ego". kind="line" (NOT trips/point — no --simplify /
    # --quantize-coords, which would desync vertex_timestamps from coords). ---
    tracks = avc.build_tracks(
        lon=objects["lon"], lat=objects["lat"], timestamp=objects["timestamp"],
        category=objects["category"], track_id=objects["track_id"],
        speed=objects["speed"])
    tracks.append(dict(lon=ego_lon, lat=ego_lat,
                       vts=[int(t) for t in ego_t],
                       vvals=[float(v) for v in ego_speed], category="ego"))
    track_pq = out / "tracks.parquet"
    n_tracks = avc.write_track_lines(track_pq, tracks=tracks)
    track_categories = sorted({str(tr["category"]) for tr in tracks})

    # LIDAR. The SURFEL path (--surfel) bakes a single oriented-Gaussian-surfel
    # tier (per-sweep k-NN covariance → quaternion + extents + confidence) for
    # the client's SplatLayer; the default path builds the height/colour point
    # tiers. Both end with `lidar_densities` + `n_lidar` + `surfel_mode`.
    lidar_densities = []
    surfel_mode = bool(args.surfel)
    contour_mode = bool(args.contours)
    worldbuild_mode = bool(args.worldbuild)
    lod_mode = bool(args.lod)
    fl_rgb = None
    if contour_mode:
        # DENSITY ISO-LINES: contour where the cloud clusters (height-independent),
        # per playhead window, so the contour map morphs live as the car drives.
        # Replaces the point/surfel tiers with ONE windowed-LineString archive in
        # the `lidar/` slot (no density selector — there's a single archive).
        dec = args.contour_decimate
        mode3d = "3D z-layered" if args.contour_z_step > 0 else "2D flat"
        print(f"  decoding LIDAR for density iso-lines ({mode3d}, uniform 1/{dec})…")
        cl_lon, cl_lat, cl_t, cl_z, _ci, _ = extract_lidar(
            cpath("lidar"), cal, pose_by_ts, origin, to_lonlat,
            decimate=dec, colorizer=None)
        contours = avc.build_density_contours(
            cl_lon, cl_lat, cl_z, cl_t, anchor_lat=anchor_lat,
            **avc.contour_kwargs(args))
        iso_pq = out / "lidar.parquet"
        n_iso = avc.write_contour_lines(iso_pq, contours)
        if not args.skip_build and n_iso > 0:
            tier_out = out / "lidar"
            shutil.rmtree(tier_out, ignore_errors=True)
            # kind="line": windowed LineStrings, NO --simplify / NO quantize-coords
            # (the line-quantize GL bug). Bucket = the window step so each window's
            # contours land in their own temporal bucket.
            avc.run_stt_build(iso_pq, tier_out, "line", stt_build=args.stt_build,
                              temporal_bucket=f"{int(args.contour_step)}ms",
                              min_zoom=LIDAR_MIN_ZOOM)
        elif n_iso == 0:
            print("  (--contours) NO contours produced — try a smaller --contour-cell "
                  "/ --contour-min-len or a larger --contour-width.")
        n_lidar = int(n_iso)  # contour-line count (iso has no point-density tiers)
    elif scene_split_mode:
        # SCENE-SPLIT ("stage + actors"): TWO archives. STATIC stage (every sweep
        # accumulated, moving returns removed via in-box + ERASOR, curvature-adaptive
        # compaction + photo grade, baked as one timeless full-range cloud) + DYNAMIC
        # actors (moving returns, kept per-sweep + animated). Cross-sweep colour is
        # fused in LINEAR light (no washout). Shared av_common pipeline w/ AV2.
        ss = avc.scene_split_config(args)
        print(f"  scene-split: stage voxel {ss['voxel_m']} m, static-speed "
              f"{ss['static_speed']} m/s, erasor={'on' if ss['erasor'] else 'off'}…")
        moving_boxes = extract_moving_boxes_world(
            cpath("lidar_box"), pose_by_ts, origin, ss["static_speed"])
        acc, dynamic, scans = extract_worldbuild(
            cpath("lidar"), cal, pose_by_ts, origin, colorizer, moving_boxes,
            voxel_m=ss["voxel_m"], collect_scans=True, color_mode="linear_weighted")
        static_final = acc.finalize()
        drop = (avc.erasor_scrub(static_final, scans, **ss["erasor_params"])
                if ss["erasor"] else None)
        if drop is not None:
            print(f"  scene-split: ERASOR scrubbed {int(drop.sum())} / {len(drop)} "
                  f"stage voxel(s) as residual dynamic")
        stage = avc.finalize_stage(static_final, drop_mask=drop,
                                   adaptive=ss.get("adaptive"), grade=ss.get("grade"))
        actors = avc.finalize_actors(dynamic, grade=ss.get("actor_grade"),
                                     denoise=ss.get("denoise"))
        for d in (stage, actors):
            s_lon, s_lat = to_lonlat(d["xy"][:, 0], d["xy"][:, 1])
            d["lon"], d["lat"] = np.atleast_1d(s_lon), np.atleast_1d(s_lat)
        n_stage, n_actors = len(stage["z"]), len(actors["z"])
        fl_rgb = stage["rgb"]  # mark the bundle camera-colored in scene.json
        print(f"  scene-split: {n_stage} stage + {n_actors} actor point(s)")
        stage_pq = out / "static.parquet"
        avc.write_lidar_points(
            stage_pq, lon=stage["lon"], lat=stage["lat"], timestamp=stage["timestamp"],
            end_timestamp=t_end, z=stage["z"], rgb=stage["rgb"], surfels=stage["surfels"],
            world_class=stage["world_class"], is_dynamic=stage["is_dynamic"],
            pack_quat=args.pack_quat)
        actors_pq = out / "dynamic.parquet"
        avc.write_lidar_points(
            actors_pq, lon=actors["lon"], lat=actors["lat"], timestamp=actors["timestamp"],
            z=actors["z"], rgb=actors["rgb"], surfels=actors["surfels"],
            world_class=actors["world_class"], is_dynamic=actors["is_dynamic"],
            pack_quat=args.pack_quat)
        if not args.skip_build and n_stage > 0:
            shutil.rmtree(out / "static", ignore_errors=True)  # prune orphan packs
            avc.run_stt_build(
                stage_pq, out / "static", "point", stt_build=args.stt_build,
                min_zoom=args.stage_min_zoom, quantize_coords=LIDAR_QUANTIZE_M,
                quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                vector_groups=avc.lidar_vector_groups(surfel=True, colored=True),
                static_full_range=True)
        if not args.skip_build and n_actors > 0:
            shutil.rmtree(out / "dynamic", ignore_errors=True)  # prune orphan packs
            avc.run_stt_build(
                actors_pq, out / "dynamic", "point", stt_build=args.stt_build,
                temporal_bucket=args.temporal_bucket,
                min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                vector_groups=avc.lidar_vector_groups(surfel=True, colored=True))
        elif n_actors == 0:
            print("  scene-split: NO dynamic (moving-actor) returns this scene.")
        lidar_densities.append({"id": "stage", "label": "Stage", "points": n_stage,
                                "url": "static/manifest.json"})
        lidar_densities.append({"id": "actors", "label": "Actors", "points": n_actors,
                                "url": "dynamic/manifest.json"})
        n_lidar = n_actors
    elif worldbuild_mode:
        # WORLDBUILD: one consolidated cross-sweep-merged cloud. STATIC returns
        # voxel-deduped over the whole drive; DYNAMIC (moving-actor) returns pass
        # through un-merged keeping their own per-sweep time. Surfels fit on the
        # merged cloud → SplatLayer. Camera-coloured (forced above).
        print(f"  worldbuild: merging all sweeps (voxel {args.worldbuild_voxel} m, "
              f"static-speed {args.worldbuild_static_speed} m/s)…")
        moving_boxes = extract_moving_boxes_world(
            cpath("lidar_box"), pose_by_ts, origin, args.worldbuild_static_speed)
        acc, dynamic = extract_worldbuild(
            cpath("lidar"), cal, pose_by_ts, origin, colorizer, moving_boxes,
            voxel_m=args.worldbuild_voxel)
        wb = avc.worldbuild_merge(acc, dynamic)
        n_dyn = int(wb["is_dynamic"].sum())
        wb_lon, wb_lat = to_lonlat(wb["xy"][:, 0], wb["xy"][:, 1])
        fl_rgb = wb["rgb"]  # marks the bundle as camera-colored in scene.json
        n_seen = int((wb["rgb"] != np.asarray(FALLBACK_RGB, np.uint8)).any(axis=1).sum())
        print(f"  worldbuild: {len(wb['z'])} merged points "
              f"({len(wb['z']) - n_dyn} static + {n_dyn} dynamic); camera color "
              f"{n_seen}/{len(wb['rgb'])} ({100 * n_seen / max(len(wb['rgb']), 1):.0f}%)")
        tier_pq = out / "lidar.parquet"
        n = avc.write_lidar_points(
            tier_pq, lon=np.atleast_1d(wb_lon), lat=np.atleast_1d(wb_lat),
            timestamp=wb["timestamp"], z=wb["z"], rgb=wb["rgb"], surfels=wb["surfels"],
            world_class=wb["world_class"], is_dynamic=wb["is_dynamic"],
            pack_quat=args.pack_quat)
        if not args.skip_build:
            tier_out = out / "lidar"
            shutil.rmtree(tier_out, ignore_errors=True)
            avc.run_stt_build(tier_pq, tier_out, "point",
                              stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                              min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                              quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                              vector_groups=avc.lidar_vector_groups(surfel=True, colored=True))
        lidar_densities.append({"id": "world", "label": "Worldbuild", "points": int(n),
                                "url": "lidar/manifest.json"})
        n_lidar = int(n)
    elif surfel_mode:
        dec = args.surfel_decimate
        mode = "adaptive (curvature+voxel)" if args.adaptive_decimate else f"uniform 1/{dec}"
        print(f"  decoding LIDAR + fitting surfels per sweep ({mode}, k={SURFEL_K})…")
        sf_lon, sf_lat, sf_t, sf_z, sf_i, sf_rgb, sf_surf = extract_lidar_surfels(
            cpath("lidar"), cal, pose_by_ts, origin, to_lonlat, colorizer, decimate=dec,
            adaptive=args.adaptive_decimate)
        fl_rgb = sf_rgb  # marks the bundle as camera-colored in scene.json below
        n_seen = int((sf_rgb != np.asarray(FALLBACK_RGB, np.uint8)).any(axis=1).sum())
        print(f"  surfels: {len(sf_surf)} oriented disks; camera color "
              f"{n_seen}/{len(sf_rgb)} ({100 * n_seen / max(len(sf_rgb), 1):.0f}%)")
        tier_pq = out / "lidar.parquet"
        n = avc.write_lidar_points(tier_pq, lon=sf_lon, lat=sf_lat, timestamp=sf_t,
                                   z=sf_z, intensity=sf_i, rgb=sf_rgb, surfels=sf_surf,
                                   pack_quat=args.pack_quat)
        if not args.skip_build:
            tier_out = out / "lidar"
            shutil.rmtree(tier_out, ignore_errors=True)
            avc.run_stt_build(tier_pq, tier_out, "point",
                              stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                              min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                              quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                              vector_groups=avc.lidar_vector_groups(
                                  surfel=True, colored=True))
        lidar_densities.append({"id": "surfel", "label": "Surfel", "points": int(n),
                                "url": "lidar/manifest.json"})
        n_lidar = int(n)
    elif lod_mode:
        # ADDITIVE-OCTREE LOD: ONE 'lidar/' archive replacing the 5 density tiers.
        # Each return is materialized at a single per-sweep, geometry-aware home
        # zoom (avc.lod_home_zoom); stt-build places it in exactly that zoom's tiles
        # via --min/max-zoom-field=home_zoom, so the client loads the union.
        print("  decoding LIDAR range images (all 5 lasers) at full density for LOD…"
              + ("  (+camera colorize)" if colorizer is not None else ""))
        fl_lon, fl_lat, fl_t, fl_z, fl_i, fl_rgb = extract_lidar(
            cpath("lidar"), cal, pose_by_ts, origin, to_lonlat, decimate=1,
            colorizer=colorizer)
        if fl_rgb is not None:
            n_seen = int((fl_rgb != np.asarray(FALLBACK_RGB, np.uint8)).any(axis=1).sum())
            print(f"  camera colorize: {n_seen}/{len(fl_rgb)} returns "
                  f"({100 * n_seen / max(len(fl_rgb), 1):.0f}%) got a camera color")
        home = avc.lod_home_zoom(
            fl_lon, fl_lat, fl_z, fl_t,
            min_zoom=args.lod_min_zoom, max_zoom=args.lod_max_zoom,
            px_per_point=args.lod_px_per_point, curvature=not args.lod_uniform)
        levels, counts = np.unique(home, return_counts=True)
        pyramid = " ".join(f"z{int(zl)}:{int(c) // 1000}k" for zl, c in zip(levels, counts))
        print(f"  LOD octree: {len(home)} returns over home-zooms "
              f"[{args.lod_min_zoom},{args.lod_max_zoom}] → {pyramid}")
        tier_pq = out / "lidar.parquet"
        n = avc.write_lidar_points(
            tier_pq, lon=fl_lon, lat=fl_lat, timestamp=fl_t, z=fl_z, intensity=fl_i,
            rgb=(fl_rgb if fl_rgb is not None else None), home_zoom=home)
        if not args.skip_build:
            tier_out = out / "lidar"
            shutil.rmtree(tier_out, ignore_errors=True)  # prune orphan packs on rebuild
            avc.run_stt_build(
                tier_pq, tier_out, "point",
                stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                min_zoom=args.lod_min_zoom, max_zoom=args.lod_max_zoom,
                quantize_coords=LIDAR_QUANTIZE_M,
                quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                vector_groups=avc.lidar_vector_groups(surfel=False, colored=True),
                min_zoom_field="home_zoom", max_zoom_field="home_zoom")
            tier_pq.unlink(missing_ok=True)  # drop the big full-density intermediate
        lidar_densities.append({
            "id": "lod", "label": "Zoom LOD", "points": int(n),
            "url": "lidar/manifest.json", "lod": True,
            "minZoom": args.lod_min_zoom, "maxZoom": args.lod_max_zoom})
        n_lidar = int(n)
    else:
        tiers = list(LIDAR_DENSITY_TIERS)
        finest = min(t[2] for t in tiers)  # = 1 (full); coarser tiers stride down from it
        print(f"  decoding LIDAR range images (all 5 lasers) at finest 1/{finest}…"
              + ("  (+camera colorize)" if colorizer is not None else ""))
        fl_lon, fl_lat, fl_t, fl_z, fl_i, fl_rgb = extract_lidar(
            cpath("lidar"), cal, pose_by_ts, origin, to_lonlat, decimate=finest,
            colorizer=colorizer)
        if fl_rgb is not None:
            n_seen = int((fl_rgb != np.asarray(FALLBACK_RGB, np.uint8)).any(axis=1).sum())
            print(f"  camera colorize: {n_seen}/{len(fl_rgb)} returns "
                  f"({100 * n_seen / max(len(fl_rgb), 1):.0f}%) got a camera color")
        # Build each density tier by nested-striding the finest cloud, then stt-build it.
        for tier_id, label, dec, dname, radius_px, radius_min_px in tiers:
            stride = dec // finest  # exact subset (all tiers are multiples of finest)
            s = slice(None, None, stride)
            tier_pq = out / f"{dname}.parquet"
            n = avc.write_lidar_points(tier_pq, lon=fl_lon[s], lat=fl_lat[s],
                                       timestamp=fl_t[s], z=fl_z[s], intensity=fl_i[s],
                                       rgb=(fl_rgb[s] if fl_rgb is not None else None))
            if not args.skip_build:
                tier_out = out / dname
                # Content-addressed packs aren't pruned on rebuild, so an existing dir
                # would accumulate ORPHAN packs (silent size bloat). Clear it first.
                shutil.rmtree(tier_out, ignore_errors=True)
                avc.run_stt_build(tier_pq, tier_out, "point",
                                  stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                                  min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                                  quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                                  vector_groups=avc.lidar_vector_groups(
                                      surfel=False, colored=True))
            lidar_densities.append({"id": tier_id, "label": label, "points": int(n),
                                    "url": f"{dname}/manifest.json",
                                    "radius": radius_px, "radiusMinPixels": radius_min_px})
        n_lidar = lidar_densities[0]["points"]  # default = low tier

    telemetry_fields = avc.derive_telemetry(ego_t, ego_speed, ego_yaw)
    tel_hz = round(len(ego_t) / max((t_end - t_start) / 1000.0, 1e-3), 1)
    avc.write_telemetry_json(out / "telemetry.json", t0=t_start, hz=tel_hz,
                             fields=telemetry_fields)

    cam_frames = None
    if not args.no_camera:
        cam_frames = extract_camera(cpath("camera_image"), out)
        if cam_frames:
            avc.write_cameras_json(out / "cameras.json", camera="FRONT",
                                   frames=cam_frames)

    streams = {
        # `url` = the default (low) archive; `densities` drives the cockpit's
        # runtime LIDAR-density selector (ascending point count).
        "lidar": {"url": "lidar/manifest.json", "points": int(n_lidar),
                  "densities": lidar_densities,
                  # Per-point RGB sampled from the cameras → the frontend paints
                  # each return its own color (r/g/b columns) instead of the
                  # height ramp. See AnimatedPointLayer.rgbColorColumns.
                  **({"colored": True} if fl_rgb is not None else {}),
                  # Surfel / worldbuild modes bake oriented-Gaussian columns
                  # (quaternion + extents + confidence); the frontend renders the
                  # tier with SplatLayer instead of AnimatedPointLayer.
                  **({"splat": "surfel"} if (surfel_mode or worldbuild_mode) else {}),
                  # Contour mode: the `lidar/` archive is windowed density iso-line
                  # LineStrings (not points); the frontend renders it with
                  # AnimatedPathLayer. A marker for the cockpit / debugging — the
                  # render path is driven by the dataset's `lidarIso` flag.
                  **({"style": "iso"} if contour_mode else {}),
                  # Worldbuild mode: the consolidated cross-sweep-merged cloud
                  # (world_class static/dynamic + is_dynamic columns). The
                  # frontend keys the static-vs-dynamic styling on it.
                  **({"style": "world"} if worldbuild_mode else {})},
        "ego": {"url": "ego/manifest.json",
                "path": avc.downsample_ego_path(ego_t, ego_lon, ego_lat)},
        "objects": {"url": "objects/manifest.json", "categories": obj_categories},
        # Feature 2: per-object motion-trail LineStrings (ego folded in as "ego").
        "tracks": {"url": "tracks/manifest.json", "count": int(n_tracks),
                   "categories": track_categories},
        "telemetry": {"url": "telemetry.json"},
    }
    if cam_frames:
        streams["camera"] = {"url": "cameras.json"}
    if scene_split_mode:
        # Scene-split overrides the single `lidar` stream with two surfel archives:
        # `lidar` aliases the animated DYNAMIC actors (single-stream code resolves),
        # plus a `stage` stream for the timeless STATIC backdrop.
        streams["lidar"] = {"url": "dynamic/manifest.json", "points": int(n_actors),
                            "colored": True, "splat": "surfel", "style": "actors",
                            "densities": lidar_densities}
        streams["stage"] = {"url": "static/manifest.json", "points": int(n_stage),
                            "colored": True, "splat": "surfel", "style": "stage",
                            "static": True}

    if not args.skip_build:
        # (lidar tiers were built above, per density.) Clear stale packs first —
        # content-addressed packs aren't pruned on rebuild, so reusing a dir would
        # accumulate orphans.
        shutil.rmtree(out / "objects", ignore_errors=True)
        shutil.rmtree(out / "ego", ignore_errors=True)
        shutil.rmtree(out / "tracks", ignore_errors=True)
        avc.run_stt_build(obj_pq, out / "objects", "point",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                          min_zoom=LIDAR_MIN_ZOOM)  # boxes only matter at street zoom too
        avc.run_stt_build(ego_pq, out / "ego", "trips",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        # Feature 2 tracks — kind="line" (NO --simplify / quantize-coords so
        # vertex_timestamps stay aligned to the coords).
        avc.run_stt_build(track_pq, out / "tracks", "line",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)

    avc.write_scene_json(
        out / "scene.json",
        id=out.name,
        name=f"Waymo · {loc_label} · {tod}",
        dataset="Waymo Open Dataset (Perception v2.0.1)",
        dataset_url="https://waymo.com/open/",
        license="Waymo Dataset License Agreement (non-commercial)",
        location=f"{loc_label} (Waymo world frame, anchored)",
        description=(
            f"Waymo Open Dataset perception segment ({tod.lower()}, {weather}): "
            "5-laser LIDAR point cloud, 3D object box tracks with real per-box "
            "velocity, ego trajectory, a FRONT-camera inset, and telemetry derived "
            "from the ego pose (Waymo Perception ships no CAN bus). Waymo discloses "
            "no georeferencing, so the scene is anchored to an approximate local "
            "frame and shown on a neutral basemap — the lidar is the map."),
        origin_lat=anchor_lat,
        origin_lon=anchor_lon,
        time_range=(t_start, t_end),
        initial_view={"longitude": float(ego_lon[len(ego_lon) // 2]),
                      "latitude": float(ego_lat[len(ego_lat) // 2]),
                      "zoom": 18, "pitch": 55, "bearing": 20},
        object_colors=avc.OBJECT_COLORS,
        streams=streams,
    )
    lidar_summary = (f"iso {n_lidar} lines" if contour_mode
                     else ", ".join(f"{d['id']}={d['points']}" for d in lidar_densities))
    print(f"Done: {out} ({len(ego_t)} ego, {n_objects} objects "
          f"[{', '.join(obj_categories)}], {n_tracks} tracks, lidar[{lidar_summary}]"
          + (f", {len(cam_frames)} cam" if cam_frames else "") + ")")
    if args.skip_build:
        print("  (--skip-build) re-run without it to build the packed archives.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg", required=True, help="Waymo segment context name")
    p.add_argument("--out", type=Path, required=True, help="output scene-bundle dir")
    p.add_argument("--raw-dir", type=Path, default=Path("waymo-raw"),
                   help="root of downloaded components (default ./waymo-raw)")
    p.add_argument("--split", default="validation", help="training|validation (default validation)")
    p.add_argument("--location", default=None,
                   help="override stats location for the anchor (location_sf|location_phx|location_other)")
    p.add_argument("--origin-lat", type=float, default=None, help="explicit anchor lat (escape hatch)")
    p.add_argument("--origin-lon", type=float, default=None, help="explicit anchor lon (escape hatch)")
    p.add_argument("--no-camera", action="store_true", help="skip the camera inset")
    p.add_argument("--colorize", action="store_true",
                   help="bake per-point RGB by projecting each LIDAR return into the "
                        "5 cameras (adds r/g/b columns; needs the camera_calibration "
                        "component). Returns no camera sees get a slate fallback.")
    p.add_argument("--surfel", action="store_true",
                   help="bake ORIENTED GAUSSIAN SURFELS: per-sweep k-NN covariance → a "
                        "quaternion + in-plane extents + confidence per return, so the "
                        "client's SplatLayer renders oriented disks instead of dots. "
                        "Implies --colorize (surfels are camera-colored); builds a single "
                        "surfel tier.")
    # --contours + all --contour-* knobs (shared with argoverse_extract.py).
    avc.add_contour_args(p)
    p.add_argument("--worldbuild", action="store_true",
                   help="WORLDBUILD view: ONE consolidated cross-sweep-merged cloud. "
                        "STATIC returns are voxel-deduped over the whole drive (a wall "
                        "seen 100× → one crisp surface, keyed first_seen); DYNAMIC "
                        "(moving-actor) returns pass through UN-merged, each keeping its "
                        "own per-sweep time (a moving car doesn't smear into a streak). "
                        "Adds world_class (static/dynamic categorical) + is_dynamic "
                        "(0/1 numeric) columns; surfels fit on the merged cloud → "
                        "SplatLayer. Implies camera colour; mutually exclusive with "
                        "--surfel / --contours.")
    p.add_argument("--worldbuild-voxel", type=float, default=WORLDBUILD_VOXEL_M,
                   dest="worldbuild_voxel",
                   help=f"static-merge voxel edge in metres (default {WORLDBUILD_VOXEL_M}).")
    p.add_argument("--worldbuild-static-speed", type=float,
                   default=WORLDBUILD_STATIC_SPEED, dest="worldbuild_static_speed",
                   help="a track at/above this speed (m/s) is MOVING → its returns are "
                        f"tagged dynamic + passed through un-merged (default "
                        f"{WORLDBUILD_STATIC_SPEED}).")
    # Scene-split ("stage + actors") flags, shared with argoverse_extract.py.
    avc.add_scene_split_args(p)
    p.add_argument("--surfel-decimate", type=int, default=SURFEL_DECIMATE,
                   help=f"render decimation for the surfel tier (default {SURFEL_DECIMATE}; "
                        "neighbourhoods are always full-density).")
    p.add_argument("--adaptive-decimate", action="store_true",
                   help="geometry-aware decimation instead of a uniform stride: keep every "
                        "high-curvature return (edges/poles/objects) + one voxel "
                        "representative per flat region. Same point budget, ~2-4x more edge "
                        "detail; lets the surfel tier go sparser for the same look. "
                        "(See lidar_summarize_eval.py for the measured bake-off.)")
    p.add_argument("--pack-quat", action="store_true",
                   help="smallest-three surfel-quaternion packing: store 3 of the 4 "
                        "quat components + a 2-bit largest-index instead of all 4 "
                        "(the 4th is reconstructed in the shader from the unit "
                        "constraint). Drops one near-incompressible orientation "
                        "column ≈12%% off the tile, LOSSLESS to ~0.16°. The client's "
                        "SplatLayer auto-detects the `q_imax` column and unpacks.")
    p.add_argument("--debug-overlay", type=int, default=None, metavar="FRAME",
                   help="project frame FRAME's full sweep into every camera, dots "
                        "colored by depth, save _debug_overlay_<CAM>.png, and STOP "
                        "(projection-correctness check; implies camera load).")
    p.add_argument("--keep-empty-boxes", action="store_true",
                   help="keep boxes with num_lidar_points_in_box==0 (default drops them)")
    p.add_argument("--lod", action="store_true",
                   help="ADDITIVE-OCTREE LOD: bake ONE 'lidar/' archive where each "
                        "return is materialized at a single per-sweep 'home zoom' "
                        "(geometry-aware: keep high-curvature edges/poles at coarse "
                        "zoom, defer flat surfaces). The client loads the union "
                        "minZoom..cameraZoom, so zooming in streams detail without "
                        "re-fetching the coarse cloud. Replaces the 5 density tiers. "
                        "Mutually exclusive with --surfel/--worldbuild/--contours/"
                        "--scene-split.")
    p.add_argument("--lod-min-zoom", type=int, default=avc.LOD_MIN_ZOOM,
                   dest="lod_min_zoom", help=f"coarsest LOD level (default {avc.LOD_MIN_ZOOM}).")
    p.add_argument("--lod-max-zoom", type=int, default=avc.LOD_MAX_ZOOM,
                   dest="lod_max_zoom",
                   help=f"finest LOD level — full-density residual (default {avc.LOD_MAX_ZOOM}).")
    p.add_argument("--lod-px-per-point", type=float, default=avc.LOD_PX_PER_POINT,
                   dest="lod_px_per_point",
                   help=f"return spacing (px) at a point's home zoom (default "
                        f"{avc.LOD_PX_PER_POINT}; smaller = denser per level).")
    p.add_argument("--lod-uniform", action="store_true", dest="lod_uniform",
                   help="plain uniform voxel grid for the LOD home-zoom (default is "
                        "geometry-aware). For A/B comparison.")
    p.add_argument("--temporal-bucket", default="100ms",
                   help="stt-build temporal bucket (Waymo lidar 10 Hz → 100ms)")
    p.add_argument("--stt-build", default="stt-build", help="stt-build binary path")
    p.add_argument("--skip-build", action="store_true", help="stop at GeoParquet + JSON")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
