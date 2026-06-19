#!/usr/bin/env python3
"""Extract one Argoverse 2 (AV2) Sensor log into an AV cockpit scene bundle.

Argoverse 2's Sensor dataset is multi-city autonomous-vehicle logs with ego
poses, 3D object tracks, LIDAR sweeps, a per-log HD map, and 9 cameras — but
**no CAN-bus telemetry**. This adapter emits the full cockpit stream set:

* ``lidar`` / ``ego`` / ``objects`` (objects carry ``num_interior_pts``; the
  fully-occluded zero-point GT boxes are dropped by default),
* ``map`` (lane boundaries + **lane centerlines** + drivable areas + crosswalks),
* ``camera`` (one ring camera, ``ring_front_center`` by default), and
* ``telemetry`` **derived from the ego pose** (speed / accel / yaw-rate / heading),
  since AV2 has no CAN bus — so the cockpit's gauge panel still lights up.

Coordinates use the av2 devkit's exact per-city CRS, so all six cities
(ATX/DTW/MIA/PAO/PIT/WDC) register on the basemap to within metres. The cockpit
renders only the streams present in ``scene.json`` (av-cockpit.md §2).

────────────────────────────────────────────────────────────────────────────
DOWNLOAD (login-gated; CC BY-NC-SA 4.0 — non-commercial, attribute the source)
────────────────────────────────────────────────────────────────────────────
  1. Get the AV2 Sensor dataset per https://www.argoverse.org/av2.html#download
     (uses an s5cmd one-liner from the docs). A single *log* is a few GB.
  2. A log directory looks like::

         <split>/<log_id>/
           city_SE3_egovehicle.feather     # ego poses (city frame)
           annotations.feather             # 3D object tracks (ego frame)
           calibration/egovehicle_SE3_sensor.feather
           sensors/lidar/<timestamp_ns>.feather
           sensors/cameras/<cam>/<timestamp_ns>.jpg

  3. pip install av2 pyarrow shapely numpy   (av2 brings the readers; we read the
     feathers directly with pyarrow here so the heavy devkit isn't required).

Then:

  python argoverse_extract.py --log-dir ./av2/val/<log_id> --city PIT \\
    --out ../../examples/showcase/public/data/argoverse-<log_id>

Use ``--skip-build`` to stop at GeoParquet + JSON.

────────────────────────────────────────────────────────────────────────────
COORDINATE / FRAME NOTES (verified against the av2-api)
────────────────────────────────────────────────────────────────────────────
* All timestamps are **nanoseconds** (``timestamp_ns``) → ÷1e6 for Unix-ms.
* ``city_SE3_egovehicle.feather`` columns: ``timestamp_ns, qw,qx,qy,qz,
  tx_m,ty_m,tz_m`` — translation is in the **city frame** (metres).
* ``annotations.feather`` poses are in the **EGO-VEHICLE frame** (``length_m,
  width_m, height_m, qw..qz, tx_m..tz_m``), so each object is transformed
  ego → city using the nearest ego pose, then city → lon/lat.
* LIDAR sweep feathers (``x,y,z,intensity,laser_number,offset_ns``) are in the
  **ego frame** → ego → city → lon/lat with the matching pose.
* AV2 uses a city-specific UTM frame. Cities with a documented UTM origin
  (``AV2_CITY_UTM``) are georeferenced with a REAL CRS transform
  (``av_common.utm_to_lonlat(E0 + x, N0 + y, epsg)``, av-refinement.md §R2.1) —
  e.g. PIT ``E0=583710.0070 N0=4477259.9999 EPSG:32617``. The old flat-earth
  equirectangular transform drifts ~75 m at a scene ~6.9 km from the origin, so
  the UTM path removes that error. Cities without a published origin fall back
  to the equirectangular approximation about an approximate city-center (the
  relative geometry within the scene stays correct; only the absolute basemap
  registration is approximate). Pass ``--origin-lat/--origin-lon`` to override.
* The HD-map vector layers (``map/log_map_archive_*.json``) are in the same
  city frame → lon/lat via the same transform, emitted as the static ``map``
  stream (lane boundaries as lines, drivable areas + crosswalks as polygons).
* ``annotations.feather`` carries NO velocity, so per-object ``speed`` is
  finite-differenced from the city-frame center grouped by ``track_uuid``
  (av-refinement.md §R2.3) — the cockpit's velocity arrows read it.

Pipeline:  Argoverse 2  →  GeoParquet + JSON  →  stt-build  →  packed bundle.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np

import av_common as avc

# AV2 ring_front_center runs ~20 Hz → ~320 frames / 15 s log. Keep every Nth so
# the cockpit camera inset is ~20 keyframes (matches nuscenes_extract's band).
CAMERA_DECIMATE = 16

# AV2 lane-marking colour → ``MAP_COLORS`` key (av-refinement.md §R2.2). The AV2
# ``LaneMarkType`` enum encodes both colour and pattern (e.g.
# ``DOUBLE_SOLID_YELLOW``); we collapse to the painted colour — white / yellow /
# blue / red — and fall back to the generic ``lane_boundary`` for unpainted
# (``NONE``) / ``UNKNOWN`` boundaries. Keys must exist in ``avc.MAP_COLORS``.
def _lane_mark_layer(mark_type) -> str:
    name = str(mark_type).rsplit(".", 1)[-1].upper()  # "LaneMarkType.SOLID_WHITE" → "SOLID_WHITE"
    if "YELLOW" in name:
        return "lane_yellow"
    if "WHITE" in name:
        return "lane_white"
    if "BLUE" in name:
        return "lane_blue"
    if "RED" in name:
        return "lane_red"
    return "lane_boundary"  # NONE / UNKNOWN / anything unpainted

# AV2 city frames are city-specific UTM metres offset from a documented origin.
# Rather than hardcode per-city UTM origins (the old PIT/MIA-only table that fell
# back to an approximate city-center for the other four cities), we use the av2
# devkit's own exact transform: ``av2.geometry.utm.convert_city_coords_to_wgs84``
# covers ALL six cities (ATX/DTW/MIA/PAO/PIT/WDC) with no approximation — every
# city now registers on the basemap to within metres. See ``_make_city_to_lonlat``.

# AV2 LIDAR is 64-beam @ ~10 Hz → ~90-95 k returns/sweep, ~150 sweeps/log ≈ 14 M
# raw points per log. Keep every Nth return so the whole scene lands at a few
# hundred-k points (the synthetic bundle is ~160 k) — light tiles, crisp sweeps.
LIDAR_DECIMATE = 75


def downsample_ego_path(ego_t, ego_lon, ego_lat, target: int = 60):
    """Downsample the ego trajectory to a tiny ``[{t, lon, lat}, …]`` polyline.

    Reuses the SAME lon/lat/t samples the ego trips archive is built from so the
    cockpit follow-camera path tracks the rendered ego trail exactly (never a
    recomputed path). Picks ~``target`` evenly-spaced vertices, always keeping the
    first and last so the polyline spans the full timeRange. ~40–80 pts keeps
    ego-follow smooth while staying a few KB in scene.json. Identical contract to
    ``av_synthetic.downsample_ego_path`` (av-cockpit.md §3d).
    """
    n = len(ego_t)
    if n <= target:
        idx = list(range(n))
    else:
        step = max(1, n // target)
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)  # always pin the final vertex
    return [
        {
            "t": int(ego_t[i]),
            "lon": round(float(ego_lon[i]), 7),
            "lat": round(float(ego_lat[i]), 7),
        }
        for i in idx
    ]


def _city_name(city: str):
    """Resolve a city code string to the av2 ``CityName`` enum (or exit clearly)."""
    from av2.geometry.utm import CityName

    try:
        return CityName[city.upper()]
    except KeyError:
        valid = ", ".join(c.name for c in CityName)
        raise SystemExit(f"unknown AV2 city {city!r}; expected one of: {valid}")


def _resolve_origin(args):
    """The (origin_lat, origin_lon) display anchor written into scene.json.georef.

    Defaults to the devkit's documented per-city origin (the city-frame (0,0)
    point); ``--origin-lat/--origin-lon`` overrides it as an escape hatch. This is
    only the scene's display anchor — the actual coordinate transform is the exact
    devkit CRS in ``_make_city_to_lonlat``, not this origin.
    """
    if args.origin_lat is not None and args.origin_lon is not None:
        return args.origin_lat, args.origin_lon, "explicit --origin-lat/-lon"
    from av2.geometry.utm import CITY_ORIGIN_LATLONG_DICT

    lat, lon = CITY_ORIGIN_LATLONG_DICT[_city_name(args.city)]
    return float(lat), float(lon), f"av2 devkit origin for {args.city.upper()}"


def _read_feather(path: Path):
    import pyarrow.feather as feather

    return feather.read_table(path).to_pandas()


def _yaw_of(qw, qx, qy, qz) -> np.ndarray:
    """Yaw (rad, CCW about +z) from quaternion columns (vectorised)."""
    # Standard quaternion → yaw about z.
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny, cosy)


def _quat_rotate_z(x, y, qw, qx, qy, qz):
    """Rotate planar (x,y) by the yaw of a quaternion (z-rotation only)."""
    yaw = _yaw_of(qw, qx, qy, qz)
    c, s = np.cos(yaw), np.sin(yaw)
    return x * c - y * s, x * s + y * c


def _make_city_to_lonlat(city: str):
    """Build a ``(city_x_m, city_y_m) → (lon, lat)`` transform via the av2 devkit.

    AV2 city frames are city-specific UTM metres offset from a documented origin.
    ``av2.geometry.utm.convert_city_coords_to_wgs84`` applies the exact per-city
    CRS transform (returns ``(lat, lon)`` per point) for ALL six cities, so there
    is no approximate equirectangular fallback any more — every city registers on
    the basemap to within metres (the old flat-earth path drifted ~75 m on scenes
    several km from the origin, and was only correct for PIT/MIA).

    Accepts scalars or numpy arrays; returns ``(lon, lat)`` matching the input
    shape, so every existing call site (ego/objects/lidar/map) is unchanged.
    """
    from av2.geometry.utm import convert_city_coords_to_wgs84

    cn = _city_name(city)

    def to_lonlat(x_m, y_m):
        x = np.atleast_1d(np.asarray(x_m, dtype="float64")).ravel()
        y = np.atleast_1d(np.asarray(y_m, dtype="float64")).ravel()
        out = convert_city_coords_to_wgs84(np.column_stack([x, y]), cn)  # (N,2)=(lat,lon)
        lon, lat = out[:, 1], out[:, 0]
        if np.ndim(x_m) == 0 and np.ndim(y_m) == 0:
            return float(lon[0]), float(lat[0])
        return lon, lat

    return to_lonlat, f"av2 devkit CRS for {cn.name}"


def derive_telemetry(ego_t_ms, ego_speed, ego_yaw) -> dict:
    """Build a ``telemetry.json`` fields dict from the ego trajectory.

    Argoverse 2 ships **no CAN bus**, so the cockpit's gauge panel would stay
    empty. Instead we DERIVE a plausible telemetry set from the 6-DoF ego pose so
    the panel lights up (honestly labelled "derived from ego pose", not CAN):

    * ``speed`` — m/s, finite-difference of the city-frame position (already
      computed by ``extract``); the cockpit formats it as km/h.
    * ``accel`` — m/s², d(speed)/dt (longitudinal accel).
    * ``yaw_rate`` — rad/s, d(yaw)/dt on the UNWRAPPED yaw (so the ±π seam
      doesn't spike the derivative).
    * ``heading`` — rad, the ego yaw (0 = east, CCW+); the cockpit shows it in deg.

    Returns the ``fields`` mapping for ``avc.write_telemetry_json``.
    """
    t_s = np.asarray(ego_t_ms, dtype="float64") / 1000.0
    dt = np.gradient(t_s)
    dt = np.where(np.abs(dt) < 1e-3, 1e-3, dt)
    speed = np.asarray(ego_speed, dtype="float64")
    accel = np.gradient(speed) / dt
    yaw_rate = np.gradient(np.unwrap(np.asarray(ego_yaw, dtype="float64"))) / dt
    heading = np.asarray(ego_yaw, dtype="float64")

    def series(vals):
        return [[int(t), float(v)] for t, v in zip(ego_t_ms, vals)]

    return {
        "speed": {"unit": "m/s", "label": "Speed", "samples": series(speed)},
        "accel": {"unit": "m/s²", "label": "Accel", "samples": series(accel)},
        "yaw_rate": {"unit": "rad/s", "label": "Yaw rate", "samples": series(yaw_rate)},
        "heading": {"unit": "rad", "label": "Heading", "samples": series(heading)},
    }


def copy_camera_frames(log_dir: Path, camera: str, out: Path, decimate: int):
    """Decimate one ring camera's JPGs, copy them into ``<out>/cam/``, and return
    the rewritten ``[{t, url}, …]`` frame list (url relative to the scene dir).

    AV2 frames live at ``sensors/cameras/<camera>/<timestamp_ns>.jpg``; we take
    every ``decimate``-th frame (always pinning the last) so the inset is ~20
    frames, and rename them ``0000.jpg…`` for a deterministic, source-path-
    independent bundle. Mirrors ``nuscenes_extract.copy_camera_frames``. Returns
    ``None`` (and omits the camera stream) if the camera dir is absent/empty.
    """
    cam_src_dir = log_dir / "sensors" / "cameras" / camera
    jpgs = sorted(cam_src_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not jpgs:
        print(f"  no {camera} frames found under {cam_src_dir}; omitting camera stream")
        return None
    n = len(jpgs)
    idx = list(range(0, n, max(1, decimate)))
    if idx and idx[-1] != n - 1:
        idx.append(n - 1)  # pin the final keyframe so the inset spans the scene
    cam_dir = out / "cam"
    cam_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for j, i in enumerate(idx):
        src = jpgs[i]
        dst_name = f"{j:04d}.jpg"
        shutil.copyfile(src, cam_dir / dst_name)
        frames.append({"t": int(src.stem) // 1_000_000, "url": f"cam/{dst_name}"})
    print(f"  copied {len(frames)} {camera} keyframe(s) → {cam_dir}")
    return frames


def extract(log_dir: Path, city: str, drop_empty_boxes: bool = True):
    """Pull ego / objects / lidar / telemetry / HD-map arrays out of one AV2 log.

    ``drop_empty_boxes`` (default ``True``) drops annotation cuboids with
    ``num_interior_pts == 0`` (fully-occluded / ghost ground-truth boxes that
    have no LIDAR returns inside them) so the rendered scene isn't littered with
    boxes around nothing. Returns
    ``(ego, objects, lidar, telemetry_fields, map_polys, map_lines)``.
    """
    to_lonlat, frame_note = _make_city_to_lonlat(city)
    print(f"  city-frame → lon/lat via {frame_note}")
    ego_df = _read_feather(log_dir / "city_SE3_egovehicle.feather")
    ego_df = ego_df.sort_values("timestamp_ns").reset_index(drop=True)
    ego_t = (ego_df["timestamp_ns"].to_numpy() // 1_000_000).astype("int64")
    ego_cx = ego_df["tx_m"].to_numpy()  # city-frame metres
    ego_cy = ego_df["ty_m"].to_numpy()
    ego_yaw = _yaw_of(ego_df["qw"].to_numpy(), ego_df["qx"].to_numpy(),
                      ego_df["qy"].to_numpy(), ego_df["qz"].to_numpy())
    ego_lon, ego_lat = to_lonlat(ego_cx, ego_cy)
    # ego speed: finite difference of city-frame position.
    dt = np.gradient(ego_t / 1000.0)
    ego_speed = np.hypot(np.gradient(ego_cx), np.gradient(ego_cy)) / np.maximum(dt, 1e-3)
    ego = (ego_t, ego_lon, ego_lat, ego_speed)

    # Derived telemetry (AV2 ships no CAN bus) — finite-difference the ego pose so
    # the cockpit's gauge panel still lights up (honestly labelled "derived").
    telemetry_fields = derive_telemetry(ego_t, ego_speed, ego_yaw)

    # Per-ego-timestamp pose lookup for transforming ego-frame data → city.
    def ego_pose_at(t_ns_ms):
        i = int(np.searchsorted(ego_t, t_ns_ms))
        i = max(0, min(i, len(ego_t) - 1))
        return ego_cx[i], ego_cy[i], ego_yaw[i]

    # --- objects (annotations are in the EGO frame) ---
    # Pass 1: lift each annotation ego-frame center → city frame (keep city-frame
    # cx/cy around so pass 2 can finite-diff a real per-object velocity — AV2
    # annotations carry no velocity column).
    ann = _read_feather(log_dir / "annotations.feather")
    has_nip = "num_interior_pts" in ann.columns  # AV2 LIDAR-point count / cuboid
    o_t = []
    o_cat, o_head, o_len, o_wid, o_hgt, o_track, o_nip = [], [], [], [], [], [], []
    o_cx, o_cy = [], []  # city-frame metres (lon/lat batch-converted after the loop)
    n_dropped = 0
    for _, r in ann.iterrows():
        nip = int(r["num_interior_pts"]) if has_nip else -1
        if drop_empty_boxes and has_nip and nip == 0:
            n_dropped += 1  # fully-occluded / ghost GT box — no LIDAR returns inside
            continue
        t_ms = int(r["timestamp_ns"]) // 1_000_000
        ecx, ecy, eyaw = ego_pose_at(t_ms)
        # ego-frame center → city frame (rotate by ego yaw, translate by ego pos).
        c, s = math.cos(eyaw), math.sin(eyaw)
        cx = ecx + r["tx_m"] * c - r["ty_m"] * s
        cy = ecy + r["tx_m"] * s + r["ty_m"] * c
        obj_yaw = _yaw_of(r["qw"], r["qx"], r["qy"], r["qz"]) + eyaw  # ego yaw + ego→city
        o_t.append(t_ms)
        o_cat.append(avc.map_category(r["category"]))
        o_head.append(float(obj_yaw))
        o_len.append(float(r["length_m"]))
        o_wid.append(float(r["width_m"]))
        o_hgt.append(float(r["height_m"]))
        o_track.append(str(r["track_uuid"]))
        o_nip.append(nip)
        o_cx.append(float(cx))
        o_cy.append(float(cy))
    if has_nip and drop_empty_boxes:
        print(f"  objects: dropped {n_dropped} zero-point (occluded/ghost) box(es)")
    # Batch the city→lon/lat transform (one CRS call vs one per annotation row).
    if o_cx:
        o_lon_arr, o_lat_arr = to_lonlat(np.asarray(o_cx), np.asarray(o_cy))
        o_lon = [float(v) for v in np.atleast_1d(o_lon_arr)]
        o_lat = [float(v) for v in np.atleast_1d(o_lat_arr)]
    else:
        o_lon, o_lat = [], []

    # Pass 2: real per-object velocity (av-refinement.md §R2.3). Group rows by
    # track_uuid, sort by timestamp, finite-diff the city-frame center → speed
    # (m/s) + vel_heading (atan2(dy,dx)). The cockpit's velocity arrows derive
    # (vx,vy) from speed + heading, so writing a real speed (was hardcoded 0.0)
    # makes them finally appear on AV2. We OVERWRITE the box heading with the
    # motion heading for moving tracks (the velocity arrow then points the way
    # the object is actually travelling); near-stationary objects keep their
    # annotated box yaw and read speed≈0 (arrows hidden by the < 0.3 m/s gate).
    o_speed = [0.0] * len(o_t)
    o_t_arr = np.asarray(o_t, dtype="float64")
    o_cx_arr = np.asarray(o_cx, dtype="float64")
    o_cy_arr = np.asarray(o_cy, dtype="float64")
    by_track: dict[str, list[int]] = {}
    for i, tid in enumerate(o_track):
        by_track.setdefault(tid, []).append(i)
    for idx in by_track.values():
        idx.sort(key=lambda i: o_t_arr[i])
        if len(idx) < 2:
            continue  # single observation → no velocity, leave speed 0
        cx_t = o_cx_arr[idx]
        cy_t = o_cy_arr[idx]
        t_s = o_t_arr[idx] / 1000.0  # ms → s
        dt = np.gradient(t_s)
        vx = np.gradient(cx_t) / np.maximum(dt, 1e-3)
        vy = np.gradient(cy_t) / np.maximum(dt, 1e-3)
        spd = np.hypot(vx, vy)
        for k, i in enumerate(idx):
            o_speed[i] = float(spd[k])
            if spd[k] > 0.3:  # moving → heading from motion (matches the arrow)
                o_head[i] = float(math.atan2(vy[k], vx[k]))
    objects = dict(
        lon=o_lon, lat=o_lat, timestamp=o_t, category=o_cat, heading=o_head,
        length=o_len, width=o_wid, height=o_hgt, track_id=o_track, speed=o_speed,
    )
    if has_nip:
        objects["num_interior_pts"] = o_nip  # → optional Int64 column on the archive

    # --- lidar sweeps (ego frame → city → lon/lat), decimated ---
    l_lon, l_lat, l_t, l_z, l_i = [], [], [], [], []
    sweep_dir = log_dir / "sensors" / "lidar"
    for sweep in sorted(sweep_dir.glob("*.feather")):
        t_ms = int(sweep.stem) // 1_000_000
        ecx, ecy, eyaw = ego_pose_at(t_ms)
        df = _read_feather(sweep)
        x = df["x"].to_numpy()[::LIDAR_DECIMATE]
        y = df["y"].to_numpy()[::LIDAR_DECIMATE]
        z = df["z"].to_numpy()[::LIDAR_DECIMATE]
        intensity = (df["intensity"].to_numpy()[::LIDAR_DECIMATE]
                     if "intensity" in df else np.full(len(x), 128.0))
        c, s = math.cos(eyaw), math.sin(eyaw)
        cx = ecx + x * c - y * s
        cy = ecy + x * s + y * c
        lon, lat = to_lonlat(cx, cy)
        l_lon.append(np.asarray(lon))
        l_lat.append(np.asarray(lat))
        l_z.append(z)
        l_i.append(intensity.astype("float64"))
        l_t.append(np.full(len(x), t_ms, dtype="int64"))
    lidar = (np.concatenate(l_lon), np.concatenate(l_lat),
             np.concatenate(l_t), np.concatenate(l_z), np.concatenate(l_i))

    # --- HD map (static substrate, av-refinement.md §R2.2) ---
    map_polys, map_lines = extract_map(log_dir, to_lonlat)
    return ego, objects, lidar, telemetry_fields, map_polys, map_lines


def extract_map(log_dir: Path, to_lonlat):
    """Extract the AV2 HD-map vector layers → (poly, layer) + (line, layer) lists.

    Loads the per-log ``ArgoverseStaticMap`` (vector only, no raster) and emits:

    * **lane boundaries → lines** — for each lane segment, its left + right
      boundary polylines, tagged by ``LaneMarkType`` colour
      (``lane_white`` / ``lane_yellow`` / ``lane_blue`` / ``lane_red`` /
      ``lane_boundary``).
    * **lane centerlines → lines** — AV2's signature map feature: each lane
      segment's centerline, tagged ``lane_centerline`` (or
      ``lane_centerline_intersection`` when the segment ``is_intersection``).
    * **drivable areas → polygons** tagged ``drivable``.
    * **pedestrian crossings → polygons** tagged ``crosswalk``.

    All coords are AV2 *city-frame metres* → lon/lat via the same ``to_lonlat``
    used for ego/objects/lidar (so the whole scene registers consistently).
    """
    from av2.map.map_api import ArgoverseStaticMap
    from shapely.geometry import LineString, Polygon

    smap = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=False)

    def _line(xy):  # (N,2) city metres → lon/lat LineString (drop degenerate)
        lon, lat = to_lonlat(xy[:, 0], xy[:, 1])
        coords = list(zip(np.atleast_1d(lon).tolist(), np.atleast_1d(lat).tolist()))
        return LineString(coords) if len(coords) >= 2 else None

    def _poly(xyz):  # (N,3+) city metres ring → lon/lat Polygon
        lon, lat = to_lonlat(xyz[:, 0], xyz[:, 1])
        coords = list(zip(np.atleast_1d(lon).tolist(), np.atleast_1d(lat).tolist()))
        if len(coords) < 4:  # need >=3 distinct + closure for a valid ring
            return None
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)  # repair self-touching rings
        return poly if (not poly.is_empty and poly.geom_type == "Polygon") else None

    map_lines: list[tuple] = []
    n_center = 0
    for lane_id, seg in smap.vector_lane_segments.items():
        for bound, mark in (
            (seg.left_lane_boundary, seg.left_mark_type),
            (seg.right_lane_boundary, seg.right_mark_type),
        ):
            ls = _line(bound.xyz[:, :2])
            if ls is not None:
                map_lines.append((ls, _lane_mark_layer(mark)))
        # Lane CENTERLINE — interpolated midline of the two boundaries; the
        # signature AV2 map feature. Intersection lanes get a distinct layer.
        try:
            centerline = smap.get_lane_segment_centerline(lane_id)  # (N,3) city m
        except Exception:
            centerline = None
        cls = _line(centerline[:, :2]) if centerline is not None else None
        if cls is not None:
            layer = ("lane_centerline_intersection" if seg.is_intersection
                     else "lane_centerline")
            map_lines.append((cls, layer))
            n_center += 1

    map_polys: list[tuple] = []
    for da in smap.vector_drivable_areas.values():
        p = _poly(da.xyz)
        if p is not None:
            map_polys.append((p, "drivable"))
    for pc in smap.vector_pedestrian_crossings.values():
        p = _poly(pc.polygon)
        if p is not None:
            map_polys.append((p, "crosswalk"))

    print(f"  HD map: {len(smap.vector_lane_segments)} lane segs → "
          f"{len(map_lines)} line(s) ({n_center} centerline), {len(map_polys)} polygon(s) "
          f"({len(smap.vector_drivable_areas)} drivable + "
          f"{len(smap.vector_pedestrian_crossings)} crosswalk)")
    return map_polys, map_lines


def generate(args):
    origin_lat, origin_lon, origin_note = _resolve_origin(args)
    print(f"Argoverse 2 log {args.log_dir.name} @ {args.city}  (origin: {origin_note})")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    ego, objects, lidar, telemetry_fields, map_polys, map_lines = extract(
        args.log_dir, args.city, drop_empty_boxes=not args.keep_empty_boxes)
    ego_t, ego_lon, ego_lat, ego_speed = ego
    t_start, t_end = int(ego_t[0]), int(ego_t[-1])

    # --- ego (trips) ---
    ego_pq = out / "ego.parquet"
    avc.write_ego_trips(ego_pq, lon=ego_lon, lat=ego_lat,
                        vertex_timestamps=ego_t.tolist(),
                        vertex_values=ego_speed.tolist(), vehicle="ego")

    # --- objects (points) ---
    obj_pq = out / "objects.parquet"
    n_objects = avc.write_objects_points(obj_pq, **objects)
    obj_categories = sorted(set(objects["category"]))

    # --- lidar (points) ---
    l_lon, l_lat, l_t, l_z, l_i = lidar
    lid_pq = out / "lidar.parquet"
    n_lidar = avc.write_lidar_points(lid_pq, lon=l_lon, lat=l_lat,
                                     timestamp=l_t, z=l_z, intensity=l_i)

    # --- HD map (static, full-range) parquet (av-refinement.md §R2.2) ---
    map_poly_pq = out / "map_poly.parquet"
    map_line_pq = out / "map_line.parquet"
    n_map_poly = avc.write_map_polygons(map_poly_pq, map_polys, t_start, t_end)
    n_map_line = avc.write_map_lines(map_line_pq, map_lines, t_start, t_end)
    # Present layer names, deduped + ordered (poly layers first, then line layers).
    map_layers: list[str] = []
    for _, layer in map_polys + map_lines:
        if layer not in map_layers:
            map_layers.append(layer)

    # lightweight ego polyline for the cockpit ego-follow camera (av-cockpit.md
    # §3d): same lon/lat/t samples as the ego trips archive → tracks it exactly.
    ego_path = downsample_ego_path(ego_t, ego_lon, ego_lat)

    # --- derived telemetry sidecar (AV2 ships no CAN bus; this is ego-derived) ---
    tel_hz = round(len(ego_t) / max((t_end - t_start) / 1000.0, 1e-3), 1)
    avc.write_telemetry_json(out / "telemetry.json", t0=t_start, hz=tel_hz,
                             fields=telemetry_fields)

    # --- camera sidecar (one ring camera; optional, skipped with --no-camera) ---
    cam_frames = None
    if not args.no_camera:
        cam_frames = copy_camera_frames(args.log_dir, args.camera, out, CAMERA_DECIMATE)
        if cam_frames:
            avc.write_cameras_json(out / "cameras.json", camera=args.camera,
                                   frames=cam_frames)

    streams = {
        "lidar": {"url": "lidar/manifest.json", "points": int(n_lidar)},
        "ego": {"url": "ego/manifest.json", "path": ego_path},
        "objects": {"url": "objects/manifest.json", "categories": obj_categories},
        "map": {
            "polyUrl": "map_poly/manifest.json",
            "lineUrl": "map_line/manifest.json",
            "layers": map_layers,
        },
        "telemetry": {"url": "telemetry.json"},
    }
    if cam_frames:
        streams["camera"] = {"url": "cameras.json"}

    # --- build packed archives (telemetry + camera are sidecars, written above) ---
    if not args.skip_build:
        avc.run_stt_build(lid_pq, out / "lidar", "point",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        avc.run_stt_build(obj_pq, out / "objects", "point",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        avc.run_stt_build(ego_pq, out / "ego", "trips",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        # Static HD-map archives — one bucket spanning the whole replay (the
        # map_temporal_bucket default >> the ~16s scene), so they load once.
        avc.run_stt_build(map_poly_pq, out / "map_poly", "map_poly",
                          stt_build=args.stt_build)
        avc.run_stt_build(map_line_pq, out / "map_line", "map_line",
                          stt_build=args.stt_build)

    # --- scene.json ---
    avc.write_scene_json(
        out / "scene.json",
        id=out.name,
        name=f"Argoverse 2 · {args.city} · {args.log_dir.name[:8]}",
        dataset="Argoverse 2",
        dataset_url="https://www.argoverse.org/av2.html",
        license="CC BY-NC-SA 4.0",
        location=args.city,
        description="Argoverse 2 sensor log: LIDAR sweeps, 3D object tracks, ego "
                    "trajectory, HD-map substrate (lane boundaries + centerlines, "
                    "drivable areas, crosswalks), a ring-camera inset, and "
                    "telemetry derived from the ego pose (AV2 ships no CAN bus).",
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        time_range=(t_start, t_end),
        initial_view={"longitude": float(ego_lon[len(ego_lon) // 2]),
                      "latitude": float(ego_lat[len(ego_lat) // 2]),
                      "zoom": 18, "pitch": 55, "bearing": 20},
        object_colors=avc.OBJECT_COLORS,
        streams=streams,
    )
    print(f"Done: {out} ({len(ego_t)} ego [{len(ego_path)} path pts], "
          f"{n_objects} objects, {n_lidar} lidar, "
          f"{n_map_poly} map poly + {n_map_line} map line "
          f"[{', '.join(map_layers)}])")
    if args.skip_build:
        print("  (--skip-build) re-run without it to build the packed archives.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--log-dir", type=Path, required=True,
                   help="path to one AV2 Sensor log dir (contains "
                        "city_SE3_egovehicle.feather, annotations.feather, sensors/)")
    p.add_argument("--city", required=True,
                   help="AV2 city code: PIT MIA ATX DTW PAO WDC "
                        "(all georeferenced exactly via the av2 devkit)")
    p.add_argument("--out", type=Path, required=True, help="output scene-bundle dir")
    p.add_argument("--origin-lat", type=float, default=None,
                   help="override the scene.json display origin latitude (escape hatch)")
    p.add_argument("--origin-lon", type=float, default=None,
                   help="override the scene.json display origin longitude (escape hatch)")
    p.add_argument("--camera", default="ring_front_center",
                   help="ring camera channel for the cockpit inset "
                        "(default ring_front_center)")
    p.add_argument("--no-camera", action="store_true",
                   help="skip the camera inset (don't copy any JPGs)")
    p.add_argument("--keep-empty-boxes", action="store_true",
                   help="keep annotation cuboids with num_interior_pts==0 "
                        "(default drops these occluded/ghost GT boxes)")
    p.add_argument("--temporal-bucket", default="200ms",
                   help="stt-build temporal bucket (AV2 lidar ~10 Hz → 100-200ms)")
    p.add_argument("--stt-build", default="stt-build", help="stt-build binary path")
    p.add_argument("--skip-build", action="store_true",
                   help="stop at GeoParquet + JSON (don't run stt-build)")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
