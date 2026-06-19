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

* ``lidar``   — all 5 lasers' range images → a single point cloud (height_band
  colored, like Argoverse 2; Waymo's 3D-semseg labels cover only ~20/199 frames
  so per-point semantic coloring isn't viable for smooth playback),
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
    --out ../../examples/showcase/public/data/waymo-<short>

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

# Plausible (NOT exact — Waymo discloses no geo) per-location anchor lat/lon. Only
# the metro is right; the streets are not. Used purely as the local-ENU anchor.
LOCATION_ANCHORS: dict[str, tuple[float, float]] = {
    "location_sf": (37.7749, -122.4194),    # San Francisco
    "location_phx": (33.4484, -112.0740),   # Phoenix
    "location_other": (37.3861, -122.0839),  # Mountain View (catch-all)
}
DEFAULT_ANCHOR = LOCATION_ANCHORS["location_other"]

# Keep every Nth valid LIDAR return (~147k/frame raw × 199 frames ≈ 29M → a few
# hundred-k total), matching argoverse_extract's light-tiles / crisp-sweep target.
LIDAR_DECIMATE = 70
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


def _yaw_of(R: np.ndarray) -> float:
    """Yaw (rad, CCW about +z) of a 3×3 rotation."""
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


def derive_telemetry(t_ms, speed, yaw) -> dict:
    """Ego-pose-derived telemetry fields (Waymo Perception ships no CAN bus).

    Mirrors argoverse_extract.derive_telemetry exactly so the cockpit gauge panel
    lights up, honestly labelled "derived from ego pose".
    """
    t_s = np.asarray(t_ms, dtype="float64") / 1000.0
    dt = np.gradient(t_s)
    dt = np.where(np.abs(dt) < 1e-3, 1e-3, dt)
    speed = np.asarray(speed, dtype="float64")
    accel = np.gradient(speed) / dt
    yaw_rate = np.gradient(np.unwrap(np.asarray(yaw, dtype="float64"))) / dt

    def series(vals):
        return [[int(t), float(v)] for t, v in zip(t_ms, vals)]

    return {
        "speed": {"unit": "m/s", "label": "Speed", "samples": series(speed)},
        "accel": {"unit": "m/s²", "label": "Accel", "samples": series(accel)},
        "yaw_rate": {"unit": "rad/s", "label": "Yaw rate", "samples": series(yaw_rate)},
        "heading": {"unit": "rad", "label": "Heading", "samples": series(yaw)},
    }


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
                  decimate: int = LIDAR_DECIMATE):
    """All lasers' range images → decimated (lon, lat, t_ms, z, intensity) arrays.

    Streamed in small Parquet batches (one row ≈ one laser-frame range image, a
    few MB) so peak memory stays ~tens of MB rather than loading the whole ~1.5 GB
    decompressed range-image column. x,y go vehicle→world→local→lon/lat; z keeps
    the vehicle-frame height (height-above-ground) for a consistent height_band.
    """
    VAL = "[LiDARComponent].range_image_return1.values"
    SH = "[LiDARComponent].range_image_return1.shape"
    out_lon, out_lat, out_t, out_z, out_i = [], [], [], [], []
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
            T = pose_by_ts[ts][0]
            pw = pts @ T[:3, :3].T + T[:3, 3]
            lon, lat = to_lonlat(pw[:, 0] - origin[0], pw[:, 1] - origin[1])
            out_lon.append(np.atleast_1d(lon))
            out_lat.append(np.atleast_1d(lat))
            out_z.append(pts[:, 2])              # vehicle-frame height
            out_i.append(inten)
            out_t.append(np.full(len(pts), ts // 1000, dtype="int64"))
    return (np.concatenate(out_lon), np.concatenate(out_lat),
            np.concatenate(out_t), np.concatenate(out_z), np.concatenate(out_i))


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


def downsample_ego_path(ego_t, ego_lon, ego_lat, target: int = 60):
    """Tiny [{t,lon,lat}] polyline for the cockpit follow-camera (av-cockpit.md §3d)."""
    n = len(ego_t)
    if n <= target:
        idx = list(range(n))
    else:
        step = max(1, n // target)
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
    return [{"t": int(ego_t[i]), "lon": round(float(ego_lon[i]), 7),
             "lat": round(float(ego_lat[i]), 7)} for i in idx]


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
        return avc.local_to_lonlat(x_m, y_m, anchor_lat, anchor_lon, mercator=False)

    # ego / poses
    frame_ts, pose_by_ts, origin = load_poses(cpath("vehicle_pose"))
    ego_t, ego_lon, ego_lat, ego_speed, ego_yaw = extract_ego(
        frame_ts, pose_by_ts, origin, to_lonlat)
    t_start, t_end = int(ego_t[0]), int(ego_t[-1])

    cal = load_calibration(cpath("lidar_calibration"))

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

    print("  decoding LIDAR range images (all 5 lasers)…")
    l_lon, l_lat, l_t, l_z, l_i = extract_lidar(
        cpath("lidar"), cal, pose_by_ts, origin, to_lonlat, decimate=args.lidar_decimate)
    lid_pq = out / "lidar.parquet"
    n_lidar = avc.write_lidar_points(lid_pq, lon=l_lon, lat=l_lat,
                                     timestamp=l_t, z=l_z, intensity=l_i)

    telemetry_fields = derive_telemetry(ego_t, ego_speed, ego_yaw)
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
        "lidar": {"url": "lidar/manifest.json", "points": int(n_lidar)},
        "ego": {"url": "ego/manifest.json",
                "path": downsample_ego_path(ego_t, ego_lon, ego_lat)},
        "objects": {"url": "objects/manifest.json", "categories": obj_categories},
        "telemetry": {"url": "telemetry.json"},
    }
    if cam_frames:
        streams["camera"] = {"url": "cameras.json"}

    if not args.skip_build:
        avc.run_stt_build(lid_pq, out / "lidar", "point",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        avc.run_stt_build(obj_pq, out / "objects", "point",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        avc.run_stt_build(ego_pq, out / "ego", "trips",
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
    print(f"Done: {out} ({len(ego_t)} ego, {n_objects} objects "
          f"[{', '.join(obj_categories)}], {n_lidar} lidar pts"
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
    p.add_argument("--lidar-decimate", type=int, default=LIDAR_DECIMATE,
                   help=f"keep every Nth LIDAR return (default {LIDAR_DECIMATE})")
    p.add_argument("--no-camera", action="store_true", help="skip the camera inset")
    p.add_argument("--keep-empty-boxes", action="store_true",
                   help="keep boxes with num_lidar_points_in_box==0 (default drops them)")
    p.add_argument("--temporal-bucket", default="100ms",
                   help="stt-build temporal bucket (Waymo lidar 10 Hz → 100ms)")
    p.add_argument("--stt-build", default="stt-build", help="stt-build binary path")
    p.add_argument("--skip-build", action="store_true", help="stop at GeoParquet + JSON")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
