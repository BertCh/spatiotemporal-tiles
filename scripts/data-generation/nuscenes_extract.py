#!/usr/bin/env python3
"""Extract one nuScenes scene into an AV cockpit scene bundle.

Reads a nuScenes ``v1.0-mini`` (or full) download via the official
``nuscenes-devkit`` and emits the same bundle layout as ``av_synthetic.py``
(see ``av_common`` / av-cockpit.md §2): ego trajectory, tracked-object boxes,
decimated LIDAR returns, CAN-bus telemetry, and CAM_FRONT keyframes — all
georeferenced onto the documented map origin (§1).

────────────────────────────────────────────────────────────────────────────
DOWNLOAD (login-gated, like ECCO's Earthdata flow)
────────────────────────────────────────────────────────────────────────────
nuScenes is CC BY-NC-SA 4.0 (non-commercial; attribute the source — the cockpit
renders the attribution from scene.json).

  1. Register a free account at https://www.nuscenes.org/nuscenes#download
  2. Download the **Mini** split (v1.0-mini, ~4 GB: metadata + 10 scenes of
     sensor data) and the **CAN bus expansion** (can_bus.zip).
  3. Unpack so the dataroot looks like::

         nuscenes/
           v1.0-mini/            # the metadata json tables
           samples/  sweeps/     # sensor data
           maps/
           can_bus/              # CAN expansion (vehicle_monitor, pose, ...)

  4. pip install nuscenes-devkit  (pulls pyquaternion, etc.)

Then:

  python nuscenes_extract.py --dataroot ./nuscenes --version v1.0-mini \\
    --scene 0 --out ../../examples/showcase/public/data/nuscenes-0061

``--scene`` accepts a scene name (e.g. ``scene-0061``) or an integer index into
``nusc.scene``. Use ``--skip-build`` to stop at GeoParquet + JSON.

────────────────────────────────────────────────────────────────────────────
COORDINATE NOTES (verified against the devkit schema)
────────────────────────────────────────────────────────────────────────────
* ``sample['timestamp']``, ``ego_pose['timestamp']`` are **microseconds** since
  the Unix epoch → divide by 1000 for the Unix-ms the contract wants.
* ``ego_pose['translation']`` / ``sample_annotation['translation']`` are in the
  **global map frame** — a LOCAL metric frame (true ground metres) whose origin is
  the map's documented SW corner (NUSCENES_MAP_ORIGINS), exactly as the official
  devkit's ``export_poses.derive_latlon`` treats it. So
  ``av_common.local_to_lonlat(x, y, origin_lat, origin_lon)`` (ground metres) with
  that SW-corner origin places the scene on the real basemap. EVERY nuScenes
  conversion (ego, objects, lidar, HD map) uses this same transform so map +
  sensors stay aligned. (Do NOT treat these as EPSG:3857 web-mercator metres — the
  nuScenes frame is local, not the global mercator projection; the old
  ``mercator=True`` deflation shifted Boston scenes ~450 m off the basemap and
  shrank them ~26 %.)
* The HD-map vectors (``NuScenesMap.extract_polygon`` / ``extract_line``) live in
  the SAME global map frame (ground metres), so they share the exact same
  ``local_to_lonlat`` transform — map + sensors stay aligned.
* ``sample_annotation['size']`` is ``[width, length, height]`` — a well-known
  nuScenes ordering GOTCHA (NOT l,w,h).
* yaw = ``Quaternion(rotation).yaw_pitch_roll[0]`` (radians, CCW about +z,
  0 = box x-axis along global +x/east) — exactly the contract's heading.
* ``vehicle_monitor['vehicle_speed']`` is **km/h** → ÷3.6 for m/s.
* ``vehicle_monitor['throttle']`` is an integer pedal position in **[0, 1000]**
  → ÷1000 for the contract's ``"frac"`` (0–1).
* ``vehicle_monitor['brake']`` is braking pressure in **bar, [0, 126]** (NOT a
  0–1000 pedal value — the CAN README is explicit) → ÷126 for the ``"frac"``.
  ÷1000 (the old bug) left the brake gauge pinned near zero.
* Steering: prefer ``steeranglefeedback['value']`` (**100 Hz, already radians**,
  0 = straight, + = left) for a smooth wheel gauge; fall back to
  ``vehicle_monitor['steering']`` (**degrees**, 2 Hz → radians) when absent.
* ``pose['accel']`` (50 Hz, ego frame m/s²) → longitudinal accel gauge (accel[0]).
* CAN bus is absent for a few scenes; this is handled gracefully (telemetry
  stream omitted).

Pipeline:  nuScenes  →  GeoParquet + JSON  →  stt-build  →  packed bundle.
"""

from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

import numpy as np

import av_common as avc

# Decimate LIDAR to keep tiles light (every Nth return). ~30k pts/sweep × 40
# keyframes ≈ 1.2M raw → /8 ≈ 150k, in the contract's "few hundred k" band.
LIDAR_DECIMATE = 8

# Size levers for the LiDAR build, matching waymo/argoverse (av_common docstrings):
# clamp the zoom pyramid (a nuScenes scene sits in ~one tile below z14, so z0-13
# is the "z0 bomb") and store geometry + numeric attrs as fixed-point ints. These
# were previously omitted here, so nuScenes shipped a full pyramid of raw-Float64
# tiles — the heaviest LiDAR build in the showcase despite being the default demo.
LIDAR_MIN_ZOOM = 14
LIDAR_QUANTIZE_M = 0.05  # ~3.75 cm grid (av_common warns to keep AV quantize <= 0.1)

# Decimate CAM_FRONT keyframes to ~20 frames in the cockpit's camera inset
# (every Nth keyframe). 40 keyframes / 2 ≈ 20, the contract's camera band.
CAMERA_DECIMATE = 2

# HD-map substrate (av-cockpit.md §2 R2.2). The polygon vs line layers of the
# nuScenes map-expansion we surface, in the cockpit's painter order. ``MAP_PAD_M``
# expands the ego bounding box by this many metres so the visible map extends a
# little past the drive (clip patch is in map-frame ground metres, same as
# the ego/object coords).
MAP_POLY_LAYERS = (
    "drivable_area",
    "road_segment",
    "lane",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
)
MAP_LINE_LAYERS = ("road_divider", "lane_divider")
MAP_PAD_M = 80.0

# ── camera-projection colorize (optional, --colorize) ─────────────────────────
# nuScenes ships SIX cameras covering the full 360° around the car (unlike Waymo's
# ~252° front+sides), so a colorized cloud is nearly complete. Each camera's
# `calibrated_sensor` carries a 3×3 `camera_intrinsic` (rectified pinhole, NO
# distortion) + a sensor→ego rotation (quaternion w,x,y,z) + translation. The
# optical axis is camera +z (standard CV convention).
NUSCENES_CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
# Cool slate for returns no camera sees (above the FOV / occluded) — the shared
# av_common constant, so colored scenes read consistently across sources.
FALLBACK_RGB = avc.FALLBACK_RGB
# Returns nearer than this (m) to a camera are skipped (on the ego shell).
MIN_CAM_DEPTH_M = 1.0


class NuScenesColorizer:
    """Projects GLOBAL-frame LIDAR returns into the 6 cameras to sample RGB.

    Per keyframe, each return is projected into every camera; a return seen by
    several cameras takes the color of the one viewing it most CENTRALLY (smallest
    normalized angular radius from the optical axis ⇒ least oblique, most reliable
    color). The transform chain is the canonical devkit one
    (``map_pointcloud_to_image``): global → ego(camera ts) → camera → pixel via
    ``camera_intrinsic``. Returns no camera sees keep ``got=False``.
    """

    def __init__(self, dataroot: Path):
        self.dataroot = dataroot
        self._cache: "collections.OrderedDict[str, np.ndarray]" = \
            collections.OrderedDict()
        self.missing: set[str] = set()

    def _image(self, filename: str):
        """Decode (LRU-cached, ≤12) one camera JPG under the dataroot → HxWx3."""
        from PIL import Image

        arr = self._cache.get(filename)
        if arr is not None:
            self._cache.move_to_end(filename)
            return arr
        path = self.dataroot / filename
        if not path.exists():
            if not self.missing:
                print(f"  warning: camera JPG missing under {self.dataroot} "
                      f"(first: {filename}) — affected returns fall back to "
                      f"FALLBACK_RGB; total reported at end")
            self.missing.add(filename)
            return None
        arr = np.asarray(Image.open(path).convert("RGB"))
        self._cache[filename] = arr
        if len(self._cache) > 12:
            self._cache.popitem(last=False)
        return arr

    def colorize(self, nusc, sample, pts_global: np.ndarray):
        """(global-frame N×3 pts, the keyframe sample) → (rgb uint8 N×3, got N)."""
        from pyquaternion import Quaternion

        n = len(pts_global)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        best_r2 = np.full(n, np.inf)
        got = np.zeros(n, dtype=bool)
        for cam in NUSCENES_CAMERAS:
            tok = sample["data"].get(cam)
            if tok is None:
                continue
            sd = nusc.get("sample_data", tok)
            img = self._image(sd["filename"])
            if img is None:
                continue
            h, w = img.shape[0], img.shape[1]
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            K = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
            R_ep = Quaternion(ep["rotation"]).rotation_matrix
            t_ep = np.asarray(ep["translation"], dtype=np.float64)
            R_cs = Quaternion(cs["rotation"]).rotation_matrix
            t_cs = np.asarray(cs["translation"], dtype=np.float64)
            # global → ego(cam ts) → camera frame (Rᵀ(p−t) == (p−t)@R).
            pe = (pts_global - t_ep) @ R_ep
            pc = (pe - t_cs) @ R_cs
            depth = pc[:, 2]  # optical axis = +z
            in_front = depth > MIN_CAM_DEPTH_M
            ds = np.where(in_front, depth, 1.0)
            xn = pc[:, 0] / ds
            yn = pc[:, 1] / ds
            ui = np.round(K[0, 0] * xn + K[0, 2]).astype(np.int64)
            vi = np.round(K[1, 1] * yn + K[1, 2]).astype(np.int64)
            valid = in_front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            r2 = xn * xn + yn * yn  # angular distance² from the optical axis
            better = valid & (r2 < best_r2)
            if better.any():
                bi = np.where(better)[0]
                rgb[bi] = img[vi[bi], ui[bi]]
                best_r2[bi] = r2[bi]
                got[bi] = True
        return rgb, got


def debug_overlay_nuscenes(nusc, scene, dataroot: Path, out: Path,
                           colorizer: "NuScenesColorizer", sample_index: int):
    """Project one keyframe's full sweep into every camera, dots colored by DEPTH.

    Saves ``<out>/_debug_overlay_<CAM>.png`` per camera — the projection-
    correctness check (dots must land ON cars/road/buildings).
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    samples = list(_iter_samples(nusc, scene))
    idx = max(0, min(sample_index, len(samples) - 1))
    sample = samples[idx]
    sd_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", sd_lidar["ego_pose_token"])
    cs_l = nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])
    pc = LidarPointCloud.from_file(str(dataroot / sd_lidar["filename"]))
    pc.rotate(Quaternion(cs_l["rotation"]).rotation_matrix)
    pc.translate(np.asarray(cs_l["translation"]))
    pc.rotate(Quaternion(pose["rotation"]).rotation_matrix)
    pc.translate(np.asarray(pose["translation"]))
    pts_g = pc.points[:3].T  # (N,3) global, full density
    print(f"  debug overlay: keyframe {idx}, {len(pts_g)} returns")
    for cam in NUSCENES_CAMERAS:
        tok = sample["data"].get(cam)
        if tok is None:
            continue
        sd = nusc.get("sample_data", tok)
        img = colorizer._image(sd["filename"])
        if img is None:
            continue
        h, w = img.shape[0], img.shape[1]
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        K = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
        pe = (pts_g - np.asarray(ep["translation"])) @ Quaternion(ep["rotation"]).rotation_matrix
        pcam = (pe - np.asarray(cs["translation"])) @ Quaternion(cs["rotation"]).rotation_matrix
        depth = pcam[:, 2]
        in_front = depth > MIN_CAM_DEPTH_M
        ds = np.where(in_front, depth, 1.0)
        ui = np.round(K[0, 0] * pcam[:, 0] / ds + K[0, 2]).astype(np.int64)
        vi = np.round(K[1, 1] * pcam[:, 1] / ds + K[1, 2]).astype(np.int64)
        valid = in_front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        canvas = img.copy()
        bi = np.where(valid)[0]
        if len(bi):
            colors = avc.depth_ramp(depth[bi], depth[bi].min(), np.percentile(depth[bi], 95))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    canvas[np.clip(vi[bi] + dy, 0, h - 1), np.clip(ui[bi] + dx, 0, w - 1)] = colors
        path = out / f"_debug_overlay_{cam}.png"
        Image.fromarray(canvas).save(path)
        print(f"    wrote {path}  ({len(bi)} projected returns)")


def copy_camera_frames(cam_frames, dataroot: Path, out: Path, decimate: int):
    """Resolve CAM_FRONT keyframe records against the dataroot and bundle them
    into ``<out>/cam/``.

    ``cam_frames`` entries are ``{"t": t_ms, "src": relative_jpg_path}``; the
    shared ``avc.copy_camera_frames`` owns the decimate / copy / url-rewrite step.
    """
    keyframes = [(f["t"], dataroot / f["src"]) for f in cam_frames]
    return avc.copy_camera_frames(keyframes, out, decimate, "CAM_FRONT")


def _yaw_of(rotation) -> float:
    """Global-frame yaw (rad, CCW about +z) from a nuScenes quaternion [w,x,y,z]."""
    # Per-dataset ON PURPOSE (not an av_common hoist): nuScenes annotations carry
    # devkit quaternion records; argoverse_extract._yaw_of takes vectorised
    # quaternion columns, waymo_extract._yaw_of a 3×3 rotation.
    from pyquaternion import Quaternion

    return float(Quaternion(rotation).yaw_pitch_roll[0])


def _resolve_scene(nusc, scene_arg: str):
    """Resolve --scene (name or integer index) to a scene record."""
    try:
        idx = int(scene_arg)
        return nusc.scene[idx]
    except (ValueError, IndexError):
        for s in nusc.scene:
            if s["name"] == scene_arg:
                return s
    raise SystemExit(f"scene {scene_arg!r} not found among {[s['name'] for s in nusc.scene]}")


def _iter_samples(nusc, scene):
    """Yield each sample record in a scene, in time order."""
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        yield sample
        token = sample["next"]


def extract(nusc, scene, dataroot: Path, origin_lat: float, origin_lon: float,
            seg_lut=None, colorizer=None):
    """Pull ego / objects / lidar / camera arrays out of one scene (global frame).

    When ``seg_lut`` is given (a numpy array mapping each lidarseg class index to
    a coarse ``av_common.LIDARSEG_CLASSES`` label), the per-point semantic class
    is read from the scene's lidarseg ``.bin`` files and decimated in lockstep
    with the points, so the returned ``lidar`` carries a ``seg`` array; else
    ``seg`` is ``None`` and the cloud falls back to height-band coloring.

    When ``colorizer`` (a ``NuScenesColorizer``) is given, each decimated return is
    projected into the 6 cameras and assigned an RGB color (returns no camera sees
    take ``FALLBACK_RGB``), so the cloud can be painted per-point from the imagery;
    the returned ``lidar`` then carries an ``rgb`` array (else ``None``).
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion

    ego_t, ego_x, ego_y, ego_speed = [], [], [], []
    o_lon, o_lat, o_t = [], [], []
    o_cat, o_head, o_len, o_wid, o_hgt, o_track, o_speed = [], [], [], [], [], [], []
    l_x, l_y, l_z, l_i, l_t = [], [], [], [], []
    l_seg = [] if seg_lut is not None else None
    l_rgb = [] if colorizer is not None else None
    fallback = np.asarray(FALLBACK_RGB, dtype=np.uint8)
    cam_frames = []

    prev_xy = None
    prev_t = None
    for sample in _iter_samples(nusc, scene):
        t_ms = sample["timestamp"] // 1000  # µs → ms

        # --- ego pose (via the LIDAR_TOP sample_data's ego_pose) ---
        sd_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd_lidar["ego_pose_token"])
        ex, ey = float(pose["translation"][0]), float(pose["translation"][1])
        ego_t.append(t_ms)
        ego_x.append(ex)
        ego_y.append(ey)
        # ego speed from finite difference of successive poses (m/s).
        if prev_xy is not None and prev_t is not None and t_ms > prev_t:
            d = math.hypot(ex - prev_xy[0], ey - prev_xy[1])
            ego_speed.append(d / ((t_ms - prev_t) / 1000.0))
        else:
            ego_speed.append(0.0)
        prev_xy, prev_t = (ex, ey), t_ms

        # --- tracked objects (sample_annotation, global frame) ---
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            ox, oy = float(ann["translation"][0]), float(ann["translation"][1])
            lon, lat = avc.local_to_lonlat(ox, oy, origin_lat, origin_lon)
            w, length, h = ann["size"]  # GOTCHA: nuScenes size is [w, l, h]
            vel = nusc.box_velocity(ann_token)  # [vx,vy,vz] m/s global, may be nan
            spd = float(np.hypot(vel[0], vel[1])) if np.all(np.isfinite(vel[:2])) else 0.0
            o_lon.append(float(lon))
            o_lat.append(float(lat))
            o_t.append(t_ms)
            o_cat.append(avc.map_category(ann["category_name"]))
            o_head.append(_yaw_of(ann["rotation"]))
            o_len.append(float(length))
            o_wid.append(float(w))
            o_hgt.append(float(h))
            o_track.append(ann["instance_token"])
            o_speed.append(spd)

        # --- LIDAR_TOP sweep → global frame, decimated ---
        pc = LidarPointCloud.from_file(str(dataroot / sd_lidar["filename"]))
        cs = nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])
        # sensor → ego → global (the canonical devkit transform sequence).
        pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
        pc.translate(np.array(cs["translation"]))
        pc.rotate(Quaternion(pose["rotation"]).rotation_matrix)
        pc.translate(np.array(pose["translation"]))
        # Decimate the SAME every-Nth slice for points AND lidarseg labels so
        # they stay index-aligned (the label file is in raw sweep order, the
        # same order LidarPointCloud.from_file reads).
        dec = slice(None, None, LIDAR_DECIMATE)
        pts = pc.points[:, dec]  # (4, N/dec): x,y,z,intensity (GLOBAL frame)
        lon, lat = avc.local_to_lonlat(pts[0], pts[1], origin_lat, origin_lon)
        # z relative to the ego ground (global z minus sensor-height baseline).
        z = pts[2] - float(pose["translation"][2])
        l_x.append(np.asarray(lon))
        l_y.append(np.asarray(lat))
        l_z.append(z)
        l_i.append(pts[3])
        l_t.append(np.full(pts.shape[1], t_ms, dtype="int64"))
        if l_rgb is not None:
            # Project the SAME decimated global-frame returns into the cameras.
            rgb_chunk, got = colorizer.colorize(nusc, sample, pts[:3].T)
            rgb_chunk[~got] = fallback
            l_rgb.append(rgb_chunk)
        if l_seg is not None:
            # lidarseg label file = uint8 / point, same length + order as the
            # raw sweep; index the coarse-class LUT then decimate identically.
            lseg_rec = nusc.get("lidarseg", sd_lidar["token"])
            labels = np.fromfile(str(dataroot / lseg_rec["filename"]), dtype=np.uint8)
            l_seg.append(seg_lut[labels[dec]])

        # --- CAM_FRONT keyframe (carry the source path; copied + renamed later) ---
        sd_cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cam_frames.append({"t": sd_cam["timestamp"] // 1000, "src": sd_cam["filename"]})

    ego = (
        np.array(ego_t, dtype="int64"),
        np.array(ego_x), np.array(ego_y), np.array(ego_speed),
    )
    objects = dict(
        lon=o_lon, lat=o_lat, timestamp=o_t, category=o_cat, heading=o_head,
        length=o_len, width=o_wid, height=o_hgt, track_id=o_track, speed=o_speed,
    )
    lidar = (
        np.concatenate(l_x), np.concatenate(l_y),
        np.concatenate(l_t), np.concatenate(l_z), np.concatenate(l_i),
        np.concatenate(l_seg) if l_seg is not None else None,
        np.concatenate(l_rgb) if l_rgb is not None else None,
    )
    return ego, objects, lidar, cam_frames


def extract_can(nusc_can, scene_name: str):
    """Pull CAN-bus telemetry for a scene, or None if absent.

    Gauges the contract wants (av-cockpit.md §2d), each from its best-fidelity
    CAN stream — every field carries its own sample series, so mixed rates are
    fine (the cockpit binary-searches each at the playhead):

    * ``vehicle_monitor`` (2 Hz): speed (km/h→m/s), throttle (0–1000→frac),
      brake (**bar [0,126]**→frac — NOT 0–1000), plus the richer channels
      (av-cockpit.md §2 R2.4) ``yaw_rate`` (rad/s, + = left), ``left_signal`` /
      ``right_signal`` (0/1 turn-indicators).
    * ``steeranglefeedback`` (100 Hz, native radians) for a smooth steering gauge,
      else vehicle_monitor's 2 Hz degrees.
    * ``pose`` (50 Hz, ego frame): longitudinal accel (accel[0], m/s²).

    Returns a telemetry dict or None.
    """
    def _msgs(name):
        """get_messages for one stream, [] if the stream is absent for this scene."""
        try:
            return nusc_can.get_messages(scene_name, name)
        except Exception:  # noqa: BLE001 — devkit raises bare Exception when absent
            return []

    vm = _msgs("vehicle_monitor")
    if not vm:
        print(f"  no vehicle_monitor messages for {scene_name}; omitting telemetry.")
        return None

    speed = [[m["utime"] // 1000, m["vehicle_speed"] / 3.6] for m in vm]  # km/h → m/s
    throttle = [[m["utime"] // 1000, m["throttle"] / 1000.0] for m in vm]  # 0–1000 → frac
    brake = [[m["utime"] // 1000, m["brake"] / 126.0] for m in vm]  # bar [0,126] → frac
    # Richer CAN channels (av-cockpit.md §2 R2.4): yaw-rate (rad/s, + = left turn)
    # and the turn-signal indicators (0/1 booleans) — all straight off
    # vehicle_monitor, so R1.4's strip-charts auto-render them.
    yaw_rate = [[m["utime"] // 1000, float(m["yaw_rate"])] for m in vm]  # rad/s
    left_signal = [[m["utime"] // 1000, float(m["left_signal"])] for m in vm]  # 0/1
    right_signal = [[m["utime"] // 1000, float(m["right_signal"])] for m in vm]  # 0/1
    fields = {
        "speed": {"unit": "m/s", "label": "Speed", "samples": speed},
        "throttle": {"unit": "frac", "label": "Throttle", "samples": throttle},
        "brake": {"unit": "frac", "label": "Brake", "samples": brake},
        "yaw_rate": {"unit": "rad/s", "label": "Yaw rate", "samples": yaw_rate},
        "left_signal": {"unit": "bool", "label": "Left signal", "samples": left_signal},
        "right_signal": {"unit": "bool", "label": "Right signal", "samples": right_signal},
    }

    # Steering: 100 Hz steeranglefeedback (already radians) when present, else the
    # coarse 2 Hz vehicle_monitor degrees. Both land as the contract's "rad".
    saf = _msgs("steeranglefeedback")
    if saf:
        steer = [[m["utime"] // 1000, float(m["value"])] for m in saf]
    else:
        steer = [[m["utime"] // 1000, math.radians(m["steering"])] for m in vm]
    fields["steer"] = {"unit": "rad", "label": "Steering", "samples": steer}

    # Longitudinal accel from the 50 Hz ego-frame pose stream (accel[0] = forward).
    pose = _msgs("pose")
    if pose:
        accel = [[m["utime"] // 1000, float(m["accel"][0])] for m in pose]
        fields["accel"] = {"unit": "m/s²", "label": "Accel", "samples": accel}

    t0 = min(s[0] for s in speed)
    return {"t0": t0, "hz": 2.0, "fields": fields}


def _poly_to_lonlat(poly, origin_lat: float, origin_lon: float):
    """Map-frame (ground metres) shapely Polygon → lon/lat shapely Polygon.

    Transforms the exterior ring AND every interior ring (hole) through
    ``local_to_lonlat (ground metres)`` so the HD-map shares the exact georef
    of the ego/object/lidar streams. Returns ``None`` if the result is degenerate
    (empty / invalid after the transform).
    """
    from shapely.geometry import Polygon

    ex = np.asarray(poly.exterior.coords)  # (N, 2+) map-frame metres
    elon, elat = avc.local_to_lonlat(ex[:, 0], ex[:, 1], origin_lat, origin_lon)
    shell = list(zip(elon.tolist(), elat.tolist()))
    holes = []
    for ring in poly.interiors:
        h = np.asarray(ring.coords)
        hlon, hlat = avc.local_to_lonlat(h[:, 0], h[:, 1], origin_lat, origin_lon)
        holes.append(list(zip(hlon.tolist(), hlat.tolist())))
    out = Polygon(shell, holes)
    if out.is_empty or not out.is_valid:
        out = out.buffer(0)  # heal self-touching rings
    return out if (not out.is_empty and out.geom_type == "Polygon") else None


def _line_to_lonlat(line, origin_lat: float, origin_lon: float):
    """Map-frame (ground metres) shapely LineString → lon/lat LineString."""
    from shapely.geometry import LineString

    c = np.asarray(line.coords)
    lon, lat = avc.local_to_lonlat(c[:, 0], c[:, 1], origin_lat, origin_lon)
    if len(lon) < 2:
        return None
    return LineString(list(zip(lon.tolist(), lat.tolist())))


def extract_map(dataroot: Path, location: str, ego_x, ego_y, origin_lat, origin_lon):
    """Extract the HD-map substrate clipped to the scene (av-cockpit.md §2 R2.2).

    Loads ``NuScenesMap`` for the scene's ``location`` (requires the map-expansion
    unpacked under ``<dataroot>/maps/expansion/``), clips to the ego bounding box
    padded by ``MAP_PAD_M`` via ``get_records_in_patch``, and turns each record
    into a lon/lat geometry tagged with its layer name:

    * POLYGON layers (``MAP_POLY_LAYERS``) → ``extract_polygon`` per geometric
      primitive (``drivable_area`` references many via ``polygon_tokens``; the
      rest a single ``polygon_token``) → ``_poly_to_lonlat``.
    * LINE layers (``MAP_LINE_LAYERS``) → ``extract_line`` (via ``line_token``)
      → ``_line_to_lonlat``.

    Returns ``(polys, lines)`` where each is a list of ``(shapely geom in lon/lat,
    map_layer:str)`` — exactly the iterable ``av_common.write_map_polygons`` /
    ``write_map_lines`` expect. Coordinates use the SAME ground-metre georef
    as the sensor streams, so the map registers under the lidar/objects.
    """
    from nuscenes.map_expansion.map_api import NuScenesMap

    nmap = NuScenesMap(dataroot=str(dataroot), map_name=location)

    x_min, x_max = float(np.min(ego_x)), float(np.max(ego_x))
    y_min, y_max = float(np.min(ego_y)), float(np.max(ego_y))
    patch = (x_min - MAP_PAD_M, y_min - MAP_PAD_M, x_max + MAP_PAD_M, y_max + MAP_PAD_M)
    layer_names = list(MAP_POLY_LAYERS) + list(MAP_LINE_LAYERS)
    recs = nmap.get_records_in_patch(patch, layer_names, mode="intersect")
    print(f"  HD-map patch {tuple(round(c, 1) for c in patch)} (map-frame m): "
          + ", ".join(f"{k}={len(recs.get(k, []))}" for k in layer_names))

    polys: list = []
    for layer in MAP_POLY_LAYERS:
        for rec_token in recs.get(layer, []):
            rec = nmap.get(layer, rec_token)
            # drivable_area groups MANY primitives; everything else has one.
            poly_tokens = rec.get("polygon_tokens") or [rec["polygon_token"]]
            for pt in poly_tokens:
                geom = _poly_to_lonlat(nmap.extract_polygon(pt), origin_lat, origin_lon)
                if geom is not None:
                    polys.append((geom, layer))

    lines: list = []
    for layer in MAP_LINE_LAYERS:
        for rec_token in recs.get(layer, []):
            rec = nmap.get(layer, rec_token)
            geom = _line_to_lonlat(nmap.extract_line(rec["line_token"]), origin_lat, origin_lon)
            if geom is not None:
                lines.append((geom, layer))

    print(f"  HD-map extracted: {len(polys)} polygon(s) + {len(lines)} line(s)")
    return polys, lines


def generate(args):
    from nuscenes.nuscenes import NuScenes

    # Feature 1 (Worldbuild) is NOT implemented for nuScenes: the 32-beam sweep is
    # far too sparse to cross-sweep-merge into a coherent consolidated cloud
    # (Waymo's 5×64-beam and AV2's 2×64-beam are dense enough; nuScenes is not).
    # Tracks (Feature 2) are still emitted normally. The Sweep mode (Feature 5) is
    # AV2-only (needs the per-return offset_ns column nuScenes lacks).
    if getattr(args, "worldbuild", False):
        raise SystemExit(
            "--worldbuild is NOT supported for nuScenes (the 32-beam sweep is too "
            "sparse to cross-sweep-merge). Use waymo_extract.py / argoverse_extract.py "
            "--worldbuild instead; nuScenes still emits the tracks/ archive normally.")

    dataroot = args.dataroot
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)
    scene = _resolve_scene(nusc, args.scene)
    location = nusc.get("log", scene["log_token"])["location"]
    if location not in avc.NUSCENES_MAP_ORIGINS:
        raise SystemExit(f"no documented origin for map {location!r}")
    origin_lat, origin_lon = avc.NUSCENES_MAP_ORIGINS[location]

    # Build the lidarseg index→coarse-class LUT if the dataroot carries the
    # nuScenes-lidarseg labels (semantic LIDAR coloring; av_common.LIDARSEG_*).
    # Probe the scene's first keyframe so a partially-labelled dataroot falls
    # back to height-band coloring instead of crashing mid-scene.
    seg_lut = None
    if not args.no_lidarseg and getattr(nusc, "lidarseg", None):
        first = nusc.get("sample", scene["first_sample_token"])
        try:
            nusc.get("lidarseg", first["data"]["LIDAR_TOP"])
            idx2name = nusc.lidarseg_idx2name_mapping  # {idx: fine class name}
            n_idx = max(idx2name) + 1
            seg_lut = np.array(
                [avc.lidarseg_class(idx2name.get(i, "other")) for i in range(n_idx)],
                dtype=object,
            )
            print(f"  lidarseg available → semantic LIDAR coloring "
                  f"({len(idx2name)} fine classes → {len(set(seg_lut))} coarse)")
        except Exception:  # noqa: BLE001 — no label for this scene's keyframes
            seg_lut = None
    if seg_lut is None:
        print("  no lidarseg labels → height-band LIDAR coloring")

    # Optional camera-projection colorize (+ its correctness overlay).
    colorizer = None
    if args.colorize or args.debug_overlay is not None:
        colorizer = NuScenesColorizer(dataroot)
    if args.debug_overlay is not None:
        debug_overlay_nuscenes(nusc, scene, dataroot, args.out, colorizer,
                               args.debug_overlay)
        print("  (--debug-overlay) wrote projection overlays; stopping.")
        return

    print(f"Extracting {scene['name']} @ {location} → {args.out}"
          + ("  (+camera colorize)" if colorizer else ""))
    ego, objects, lidar, cam_frames = extract(
        nusc, scene, dataroot, origin_lat, origin_lon, seg_lut=seg_lut,
        colorizer=colorizer)
    ego_t, ego_x, ego_y, ego_speed = ego
    t_start, t_end = int(ego_t[0]), int(ego_t[-1])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # --- ego (trips) ---
    ego_lon, ego_lat = avc.local_to_lonlat(ego_x, ego_y, origin_lat, origin_lon)
    ego_pq = out / "ego.parquet"
    avc.write_ego_trips(ego_pq, lon=ego_lon, lat=ego_lat,
                        vertex_timestamps=ego_t.tolist(),
                        vertex_values=ego_speed.tolist(), vehicle="ego")

    # --- objects (points) ---
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

    # --- lidar (points) ---
    l_lon, l_lat, l_t, l_z, l_i, l_seg, l_rgb = lidar
    if l_rgb is not None:
        n_seen = int((l_rgb != np.asarray(FALLBACK_RGB, np.uint8)).any(axis=1).sum())
        print(f"  camera colorize: {n_seen}/{len(l_rgb)} returns "
              f"({100 * n_seen / max(len(l_rgb), 1):.0f}%) got a camera color")
        if colorizer is not None and colorizer.missing:
            print(f"  warning: {len(colorizer.missing)} camera JPG(s) missing under "
                  f"{dataroot} — those views contributed no color (FALLBACK_RGB)")
    lid_pq = out / "lidar.parquet"
    n_lidar = avc.write_lidar_points(lid_pq, lon=l_lon, lat=l_lat,
                                     timestamp=l_t, z=l_z, intensity=l_i,
                                     seg_class=l_seg, rgb=l_rgb)

    # --- HD-map substrate (static, full-range — av-cockpit.md §2 R2.2) ---
    # Clip the nuScenes map-expansion to the ego bbox+pad. ego_x/ego_y are raw
    # map-frame (ground) metres, the patch units get_records_in_patch wants;
    # extract_map projects every geom through the same ground-metre georef.
    map_polys, map_lines = extract_map(
        dataroot, location, ego_x, ego_y, origin_lat, origin_lon)
    map_poly_pq = out / "map_poly.parquet"
    map_line_pq = out / "map_line.parquet"
    n_map_poly = avc.write_map_polygons(map_poly_pq, map_polys, t_start, t_end)
    n_map_line = avc.write_map_lines(map_line_pq, map_lines, t_start, t_end)
    # Present layer names, deduped + ordered (poly layers first, then line layers).
    map_layers: list[str] = []
    for _, layer in map_polys + map_lines:
        if layer not in map_layers:
            map_layers.append(layer)

    # --- lightweight ego polyline for the ego-follow camera (av-cockpit.md §3d).
    # Same lon/lat/t samples as the ego trips archive → the follow path tracks the
    # rendered ego trail exactly. ~40–80 pts; pure JSON, no rebuild.
    ego_path = avc.downsample_ego_path(ego_t, ego_lon, ego_lat)

    # --- telemetry (CAN bus, optional) ---
    streams = {
        "lidar": {"url": "lidar/manifest.json", "points": int(n_lidar),
                  # Per-point RGB sampled from the 6 cameras → the frontend paints
                  # each return its own color (see AnimatedPointLayer.rgbColorColumns).
                  **({"colored": True} if l_rgb is not None else {})},
        "ego": {"url": "ego/manifest.json", "path": ego_path},
        "objects": {"url": "objects/manifest.json", "categories": obj_categories},
        # Feature 2: per-object motion-trail LineStrings (ego folded in as "ego").
        "tracks": {"url": "tracks/manifest.json", "count": int(n_tracks),
                   "categories": track_categories},
        "map": {
            "polyUrl": "map_poly/manifest.json",
            "lineUrl": "map_line/manifest.json",
            "layers": map_layers,
        },
    }
    tel = None
    try:
        from nuscenes.can_bus.can_bus_api import NuScenesCanBus

        tel = extract_can(NuScenesCanBus(dataroot=str(dataroot)), scene["name"])
    except Exception as e:  # noqa: BLE001
        print(f"  CAN expansion not installed ({e}); omitting telemetry.")
    if tel is not None:
        avc.write_telemetry_json(out / "telemetry.json", t0=tel["t0"], hz=tel["hz"],
                                 fields=tel["fields"])
        streams["telemetry"] = {"url": "telemetry.json"}

    # --- cameras (decimate keyframes, copy JPGs into cam/, rewrite urls) ---
    if cam_frames:
        frames = copy_camera_frames(cam_frames, dataroot, out, args.camera_decimate)
        avc.write_cameras_json(out / "cameras.json", camera="CAM_FRONT", frames=frames)
        streams["camera"] = {"url": "cameras.json"}

    # --- build packed archives ---
    if not args.skip_build:
        avc.run_stt_build(lid_pq, out / "lidar", "point",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                          min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                          quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M))
        avc.run_stt_build(obj_pq, out / "objects", "point",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        avc.run_stt_build(ego_pq, out / "ego", "trips",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        # Feature 2 tracks — kind="line" (NO --simplify / quantize-coords so
        # vertex_timestamps stay aligned to the coords).
        avc.run_stt_build(track_pq, out / "tracks", "line",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        # Static HD-map archives — one bucket spanning the whole replay (the
        # map_temporal_bucket default >> the ~20s scene), so they load once.
        avc.run_stt_build(map_poly_pq, out / "map_poly", "map_poly",
                          stt_build=args.stt_build)
        avc.run_stt_build(map_line_pq, out / "map_line", "map_line",
                          stt_build=args.stt_build)

    # --- scene.json ---
    cx_lon, cx_lat = float(ego_lon[len(ego_lon) // 2]), float(ego_lat[len(ego_lat) // 2])
    avc.write_scene_json(
        out / "scene.json",
        id=out.name,
        name=f"nuScenes · {scene['name']}",
        dataset="nuScenes",
        dataset_url="https://www.nuscenes.org/nuscenes",
        license="CC BY-NC-SA 4.0",
        location=location,
        description=scene.get("description", "nuScenes urban drive: 32-beam LIDAR, "
                                             "CAN telemetry, 3D object boxes."),
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        time_range=(t_start, t_end),
        initial_view={"longitude": cx_lon, "latitude": cx_lat,
                      "zoom": 17, "pitch": 55, "bearing": 20},
        object_colors=avc.OBJECT_COLORS,
        streams=streams,
        lidar_colors=avc.LIDARSEG_COLORS if l_seg is not None else None,
    )
    print(f"Done: {scene['name']} → {out} "
          f"({len(ego_t)} ego vertices, {len(ego_path)} ego.path pts, "
          f"{n_objects} objects, {n_tracks} tracks, {n_lidar} lidar, "
          f"{n_map_poly} map polygon(s) + {n_map_line} line(s) "
          f"[{', '.join(map_layers)}])")
    if args.skip_build:
        print("  (--skip-build) re-run without it to build the packed archives.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataroot", type=Path, default=Path("./nuscenes"),
                   help="nuScenes dataroot (contains v1.0-mini/, samples/, can_bus/)")
    p.add_argument("--version", default="v1.0-mini", help="dataset version")
    p.add_argument("--scene", default="0", help="scene name (scene-0061) or index")
    p.add_argument("--out", type=Path, required=True, help="output scene-bundle dir")
    p.add_argument("--temporal-bucket", default="500ms",
                   help="stt-build temporal bucket (nuScenes annotates at 2 Hz → 500ms)")
    p.add_argument("--camera-decimate", type=int, default=CAMERA_DECIMATE,
                   help="keep every Nth CAM_FRONT keyframe (40 keyframes /2 ≈ 20 frames)")
    p.add_argument("--stt-build", default="stt-build", help="stt-build binary path")
    p.add_argument("--worldbuild", action="store_true",
                   help="(NOT SUPPORTED for nuScenes — the 32-beam sweep is too "
                        "sparse to cross-sweep-merge; errors out with guidance. Use "
                        "waymo_extract.py / argoverse_extract.py --worldbuild.)")
    p.add_argument("--no-lidarseg", action="store_true",
                   help="ignore nuScenes-lidarseg labels; color LIDAR by height band")
    p.add_argument("--colorize", action="store_true",
                   help="bake per-point RGB by projecting each LIDAR return into the "
                        "6 cameras (adds r/g/b columns). Returns no camera sees get a "
                        "slate fallback. Pairs with the cockpit's camera-colored splat.")
    p.add_argument("--debug-overlay", type=int, default=None, metavar="KEYFRAME",
                   help="project keyframe KEYFRAME's sweep into every camera, dots "
                        "colored by depth, save _debug_overlay_<CAM>.png, and STOP "
                        "(projection-correctness check).")
    p.add_argument("--skip-build", action="store_true",
                   help="stop at GeoParquet + JSON (don't run stt-build)")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
