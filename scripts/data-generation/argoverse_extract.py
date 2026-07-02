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
  (``av_common.utm_to_lonlat(E0 + x, N0 + y, epsg)``, av-cockpit.md §2 R2.1) —
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
  (av-cockpit.md §2 R2.3) — the cockpit's velocity arrows read it.

Pipeline:  Argoverse 2  →  GeoParquet + JSON  →  stt-build  →  packed bundle.
"""

from __future__ import annotations

import argparse
import collections
import math
import shutil
from pathlib import Path

import numpy as np

import av_common as avc

# AV2 ring_front_center runs ~20 Hz → ~320 frames / 15 s log. Keep every Nth so
# the cockpit camera inset is ~20 keyframes (matches nuscenes_extract's band).
CAMERA_DECIMATE = 16

# ── camera-projection colorize (optional, --colorize) ─────────────────────────
# AV2 ships SEVEN ring cameras covering ~360° around the car. We project each
# EGO-frame LIDAR return into them via the devkit's own `PinholeCamera`
# (intrinsics.feather + egovehicle_SE3_sensor.feather → project_ego_to_img), so
# the camera model + extrinsics are exactly the av2-api's. A return seen by
# several cameras takes the color of the one viewing it most centrally.
AV2_RING_CAMERAS = (
    "ring_front_center", "ring_front_left", "ring_front_right",
    "ring_side_left", "ring_side_right", "ring_rear_left", "ring_rear_right",
)
# Cool slate for returns no camera sees — the shared av_common constant, so all
# colorizers (Waymo/nuScenes/AV2/Worldbuild) read consistently across sources.
FALLBACK_RGB = avc.FALLBACK_RGB


def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``img`` (H×W×3) at float pixel coords ``u,v`` → (N,3) float.

    Smoother than nearest-pixel (less aliasing on the projected LiDAR colour).
    Callers must pre-clamp so the 2×2 neighbourhood stays in-bounds.
    """
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    im = img.astype(np.float64)
    c00, c10 = im[y0, x0], im[y0, x1]
    c01, c11 = im[y1, x0], im[y1, x1]
    return (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
            + c01 * (1 - fx) * fy + c11 * fx * fy)


class Av2Colorizer:
    """Projects EGO-frame LIDAR returns into the 7 ring cameras to sample RGB.

    Per sweep, each return is projected into every camera (via the devkit
    ``PinholeCamera.project_ego_to_img``); the camera whose image is NEAREST in
    time to the sweep is used, and a return visible to several cameras takes the
    color of the one viewing it most CENTRALLY (smallest normalized radius from
    the optical axis). Returns no camera sees keep ``got=False``.
    """

    def __init__(self, log_dir: Path):
        from av2.geometry.camera.pinhole_camera import PinholeCamera

        self.cams: dict = {}        # cam_name → PinholeCamera
        self.frames: dict = {}      # cam_name → (ts_ns int64 array, [jpg Path])
        for cam in AV2_RING_CAMERAS:
            jpgs = sorted((log_dir / "sensors" / "cameras" / cam).glob("*.jpg"),
                          key=lambda p: int(p.stem))
            if not jpgs:
                continue
            try:
                self.cams[cam] = PinholeCamera.from_feather(log_dir, cam)
            except Exception as e:  # noqa: BLE001 — missing calib row for a cam
                print(f"  ({cam} calibration unavailable: {e})")
                continue
            self.frames[cam] = (np.asarray([int(p.stem) for p in jpgs], dtype=np.int64),
                                jpgs)
        self._cache: "collections.OrderedDict[str, np.ndarray]" = \
            collections.OrderedDict()

    def _nearest_image(self, cam: str, t_ns: int):
        """Decode (LRU ≤14) the cam frame nearest ``t_ns`` → HxWx3, or None."""
        from PIL import Image

        ts, jpgs = self.frames[cam]
        j = int(np.searchsorted(ts, t_ns))
        j = max(0, min(j, len(ts) - 1))
        if j > 0 and abs(ts[j - 1] - t_ns) < abs(ts[j] - t_ns):
            j -= 1
        path = jpgs[j]
        key = str(path)
        arr = self._cache.get(key)
        if arr is not None:
            self._cache.move_to_end(key)
            return arr
        arr = np.asarray(Image.open(path).convert("RGB"))
        self._cache[key] = arr
        if len(self._cache) > 14:
            self._cache.popitem(last=False)
        return arr

    def colorize(self, pts_ego: np.ndarray, sweep_t_ns: int, return_weight: bool = False):
        """(ego-frame N×3 pts, sweep ts ns) → (rgb uint8 N×3, got N[, weight N]).

        Each return takes the colour of the camera viewing it most CENTRALLY
        (smallest normalized radius), BILINEARLY sampled (smoother than nearest).
        With ``return_weight`` also returns a per-return view-quality weight
        (≈ 1/depth — closer passes are sharper) for ``WorldVoxelAccumulator``'s
        linear-weighted cross-sweep fusion. Returns no camera sees keep ``got=False``.
        """
        pts_ego = np.ascontiguousarray(pts_ego, dtype=np.float64)
        n = len(pts_ego)
        rgb = np.zeros((n, 3), dtype=np.float64)
        best_r2 = np.full(n, np.inf)
        depth = np.full(n, np.inf)
        got = np.zeros(n, dtype=bool)
        for cam, pc in self.cams.items():
            img = self._nearest_image(cam, sweep_t_ns)
            if img is None:
                continue
            uv, pts_cam, valid = pc.project_ego_to_img(pts_ego)
            if not valid.any():
                continue
            h, w = img.shape[0], img.shape[1]
            u, v = uv[:, 0], uv[:, 1]
            # Bilinear needs the 2×2 neighbourhood → keep one px inside each edge.
            valid = valid & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            zc = np.where(pts_cam[:, 2] > 0.1, pts_cam[:, 2], 0.1)
            r2 = (pts_cam[:, 0] / zc) ** 2 + (pts_cam[:, 1] / zc) ** 2
            better = valid & (r2 < best_r2)
            if better.any():
                bi = np.where(better)[0]
                rgb[bi] = _bilinear_sample(img, u[bi], v[bi])
                best_r2[bi] = r2[bi]
                depth[bi] = pts_cam[bi, 2]
                got[bi] = True
        rgb_u8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        if not return_weight:
            return rgb_u8, got
        weight = np.where(got, 1.0 / np.maximum(depth, 1.0), 0.0)
        return rgb_u8, got, weight


def debug_overlay_av2(log_dir: Path, out: Path, colorizer: "Av2Colorizer",
                      sweep_index: int):
    """Project one sweep into every ring camera, dots colored by DEPTH → PNGs.

    Saves ``<out>/_debug_overlay_<CAM>.png`` per camera (projection-correctness
    check: dots must land ON cars / road / buildings).
    """
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    sweeps = sorted((log_dir / "sensors" / "lidar").glob("*.feather"),
                    key=lambda p: int(p.stem))
    idx = max(0, min(sweep_index, len(sweeps) - 1))
    sweep = sweeps[idx]
    t_ns = int(sweep.stem)
    df = _read_feather(sweep)
    pts_ego = np.stack([df["x"].to_numpy(), df["y"].to_numpy(),
                        df["z"].to_numpy()], axis=1).astype(np.float64)
    print(f"  debug overlay: sweep {idx} (ts {t_ns}), {len(pts_ego)} returns")
    for cam, pc in colorizer.cams.items():
        img = colorizer._nearest_image(cam, t_ns)
        if img is None:
            continue
        uv, pts_cam, valid = pc.project_ego_to_img(pts_ego)
        h, w = img.shape[0], img.shape[1]
        ui = np.round(uv[:, 0]).astype(np.int64)
        vi = np.round(uv[:, 1]).astype(np.int64)
        valid = valid & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        canvas = img.copy()
        bi = np.where(valid)[0]
        if len(bi):
            depth = pts_cam[bi, 2]
            colors = avc.depth_ramp(depth, depth.min(), np.percentile(depth, 95))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    canvas[np.clip(vi[bi] + dy, 0, h - 1),
                           np.clip(ui[bi] + dx, 0, w - 1)] = colors
        path = out / f"_debug_overlay_{cam}.png"
        Image.fromarray(canvas).save(path)
        print(f"    wrote {path}  ({len(bi)} projected returns)")

# AV2 lane-marking colour → ``map_layer`` name (av-cockpit.md §2 R2.2). The AV2
# ``LaneMarkType`` enum encodes both colour and pattern (e.g.
# ``DOUBLE_SOLID_YELLOW``); we collapse to the painted colour — white / yellow /
# blue / red — and fall back to the generic ``lane_boundary`` for unpainted
# (``NONE``) / ``UNKNOWN`` boundaries. Every layer name returned must be a member
# of ``avc.MAP_LAYERS`` (the valid ``map_layer`` name set; colors live in TS).
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

# AV2 LIDAR is 64-beam dual @ ~10 Hz → ~90-95 k returns/sweep, ~150 sweeps/log ≈ 14 M
# raw points per log. DENSITY TIERS (mirrors waymo_extract): decode every return
# ONCE (FINEST_DECIMATE = 1) then stride down for the cockpit's runtime density
# selector (Small→Full). AV2 is georeferenced (real UTM) + HD-mapped, so even the
# dense tiers overlay actual streets.
FINEST_DECIMATE = 1
# (id, label, decimate, archive_dir, radius_px, radius_min_px) — ASCENDING density;
# `small` → the default `lidar/`. Denser tiers ship SMALLER dots so the cloud reads
# as 3D structure, not a solid mass.
LIDAR_DENSITY_TIERS = [
    ("small", "Small", 16, "lidar", 1.0, 0.8),          # ~0.9M  (default)
    ("medium", "Medium", 8, "lidar-med", 0.8, 0.65),    # ~1.8M
    ("large", "Large", 4, "lidar-high", 0.65, 0.55),    # ~3.5M
    ("ultra", "Ultra", 2, "lidar-ultra", 0.5, 0.45),    # ~7M
    ("full", "Full", FINEST_DECIMATE, "lidar-full", 0.4, 0.35),  # ~14M  (full sweep)
]
# Street-level cockpit: clamp the zoom-pyramid floor (z<14 tiers are wasted bytes,
# the "z0 bomb") + quantize coords to a few cm (viz-exact). Both decode-free client-side.
LIDAR_MIN_ZOOM = 14
LIDAR_QUANTIZE_M = 0.05


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
    # Per-dataset ON PURPOSE (not an av_common hoist): AV2 poses arrive as
    # vectorised quaternion COLUMNS; waymo_extract._yaw_of takes a 3×3 rotation,
    # nuscenes_extract._yaw_of a devkit quaternion record.
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


def copy_camera_frames(log_dir: Path, camera: str, out: Path, decimate: int):
    """Discover one ring camera's JPGs and bundle them into ``<out>/cam/``.

    AV2 frames live at ``sensors/cameras/<camera>/<timestamp_ns>.jpg`` (t_ms =
    stem ns ÷ 1e6); the shared ``avc.copy_camera_frames`` owns the decimate /
    copy / url-rewrite step. Returns ``None`` (and omits the camera stream) if
    the camera dir is absent/empty.
    """
    cam_src_dir = log_dir / "sensors" / "cameras" / camera
    jpgs = sorted(cam_src_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not jpgs:
        print(f"  no {camera} frames found under {cam_src_dir}; omitting camera stream")
        return None
    keyframes = [(int(p.stem) // 1_000_000, p) for p in jpgs]
    return avc.copy_camera_frames(keyframes, out, decimate, camera)


# ── surfel covariance (build-time, per-sweep k-NN) ───────────────────────────
# A "formal" splat needs ORIENTED anisotropic primitives, not round dots. For a
# LIDAR scan the principled way to get them WITHOUT training a 3DGS model is
# surface splatting (Pfister/Zwicker surfels): estimate each return's local
# surface frame from its k nearest neighbours' covariance — the smallest
# eigenvector is the surface normal, the two larger span the tangent plane, and
# the eigenvalues give the patch extents. The client's `SplatLayer` then renders
# each return as an oriented Gaussian disk lying on the surface. This mirrors
# `waymo_extract.compute_surfels`, adapted to AV2's EGO-frame per-sweep feathers
# (one file per sweep, so neighbourhoods are already a coherent range-image
# sampling) and AV2's yaw-only city lift (the render keeps vehicle-frame z, so
# disks are yawed — not full-pose-rotated — into the render frame).
SURFEL_K = avc.SURFEL_K  # neighbours for the covariance estimate (shared tuning)
# Default render decimation for the surfel tier. Denser than the round-dot point
# tiers (the default `small` is 1/16): at low density each SWEEP is too sparse to
# tile, so the per-point disk is sized large and reads as blobs — a denser sample
# makes each disk small and the surface crisp. The soft TEMPORAL Gaussian caps
# per-frame cost to the ~±3σ window around the playhead (not the whole cloud), so
# density is nearly free at render time. ~1/6 of the ~14M-return log ≈ 2.3M
# oriented disks (~few hundred k drawn at any instant).
SURFEL_DECIMATE = 6


def compute_surfels(pts: np.ndarray, sel: np.ndarray, yaw: float):
    """Per-sweep surfels for the selected returns → (quat N×4, scale2 N×2, opacity N).

    ``pts`` is the FULL-density EGO-frame sweep (N×3); ``sel`` indexes the
    rendered (decimated) subset. Orientation is estimated from each selected
    point's full-density neighbourhood, then rotated by the ego ``yaw`` into the
    render frame. (Yaw only: the cloud is lifted to the city frame with world x,y
    but VEHICLE-frame z — the same stable-height simplification the AV2 lidar lift
    uses — so a full-pose rotation would tilt ground disks off the flattened
    rendered ground.) Disk extents are scaled to the DECIMATED spacing so the
    sparser rendered cloud still tiles the surface. Confidence (alpha) tracks
    local planarity. Degenerate neighbourhoods fall back to a flat, up-facing,
    low-confidence disk.

    Delegates the eigen→quaternion + extents + opacity CORE to the shared
    ``av_common._surfel_frame`` (so the per-sweep and the merged Worldbuild paths
    fit surfels identically — the dual-copy hazard this repo keeps hitting). The
    view direction is toward the sensor at the ego origin (``-sel_pts``) and the
    ego-YAW matrix is passed as ``R_post``, so the output is bit-for-bit what this
    function produced before the refactor.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return avc._surfel_frame(pts, sel, -pts[sel], R_post=R_yaw)


def extract_lidar_surfels(sweep_dir: Path, ego_pose_at, to_lonlat, colorizer,
                          decimate: int = SURFEL_DECIMATE):
    """Per-sweep oriented-surfel cloud → (lon, lat, t_ms, z, intensity, rgb, surfels N×7).

    Each AV2 lidar sweep is one ``<ts_ns>.feather`` in the EGO frame; per sweep we
    read all returns at FULL density, fit surfels for the decimated subset (with
    full-density neighbourhoods, see ``compute_surfels``), camera-colour that
    subset via the 7 ring cameras, and lift it ego→city→lon/lat (yaw rotate +
    translate; z kept vehicle-frame). ``surfels`` is the
    ``[qx,qy,qz,qw,s_major,s_minor,opacity]`` block for ``avc.write_lidar_points``.
    A ``colorizer`` is required (surfels are camera-coloured); ``generate`` forces
    one on in surfel mode.
    """
    fallback = np.asarray(FALLBACK_RGB, dtype=np.uint8)
    out_lon, out_lat, out_t, out_z, out_i, out_rgb, out_sf = [], [], [], [], [], [], []
    for sweep in sorted(sweep_dir.glob("*.feather")):
        t_ns = int(sweep.stem)
        t_ms = t_ns // 1_000_000
        ecx, ecy, eyaw = ego_pose_at(t_ms)
        df = _read_feather(sweep)
        full = np.stack([df["x"].to_numpy(), df["y"].to_numpy(),
                         df["z"].to_numpy()], axis=1).astype(np.float64)
        if len(full) < SURFEL_K + 2:
            continue
        sel = np.arange(0, len(full), decimate)
        if len(sel) == 0:
            continue
        sel_pts = full[sel]
        intensity = (df["intensity"].to_numpy()[sel].astype("float64")
                     if "intensity" in df else np.full(len(sel), 128.0))
        # Project the EGO-frame returns into the ring cameras (the cameras are
        # calibrated to the ego frame, so this is BEFORE the city lift).
        rgb, hit = colorizer.colorize(sel_pts, t_ns)
        rgb[~hit] = fallback
        quat, scale2, op = compute_surfels(full, sel, eyaw)
        # ego-frame → city (rotate by ego yaw, translate by ego pos); z stays.
        c, s = math.cos(eyaw), math.sin(eyaw)
        cx = ecx + sel_pts[:, 0] * c - sel_pts[:, 1] * s
        cy = ecy + sel_pts[:, 0] * s + sel_pts[:, 1] * c
        lon, lat = to_lonlat(cx, cy)
        out_lon.append(np.atleast_1d(lon))
        out_lat.append(np.atleast_1d(lat))
        out_t.append(np.full(len(sel), t_ms, dtype="int64"))
        out_z.append(sel_pts[:, 2])
        out_i.append(intensity)
        out_rgb.append(rgb)
        out_sf.append(np.concatenate([quat, scale2, op[:, None]], axis=1))
    if not out_lon:
        raise SystemExit("surfel extraction produced no points")
    return (np.concatenate(out_lon), np.concatenate(out_lat),
            np.concatenate(out_t), np.concatenate(out_z), np.concatenate(out_i),
            np.concatenate(out_rgb), np.concatenate(out_sf))


# ── Worldbuild (--worldbuild): cross-sweep-merged consolidated cloud ─────────
# Build ONE consolidated render-world (CITY-metric) cloud over the whole log:
# STATIC returns are voxel-deduped across all sweeps (a wall sampled 100× → one
# crisp surface, keyed by first_seen) while DYNAMIC returns (inside a MOVING
# actor box at their sweep time) pass through UN-merged so a moving car doesn't
# smear into a streak. Streamed into av_common.WorldVoxelAccumulator so peak
# memory ∝ occupied voxels. See av_common.worldbuild_merge for the columns.
WORLDBUILD_VOXEL_M = 0.15        # default static-merge voxel edge (m)
WORLDBUILD_STATIC_SPEED = 0.5    # m/s — a track at/above this is "moving"


def _av2_moving_boxes(log_dir: Path, ego_pose_at, static_speed: float):
    """annotations.feather → CITY-metric moving-box dicts (the dynamic test set).

    AV2 annotations carry no velocity, so (like ``extract``'s objects pass-2) we
    lift each cuboid ego→city, group by ``track_uuid``, finite-difference the
    city-frame centre → per-observation speed, and keep ONLY observations at/above
    ``static_speed``. Each dict is ``{cx, cy, heading, length, width, t}`` (ms) in
    the SAME city-metric frame the worldbuild cloud accumulates in.
    """
    ann = _read_feather(log_dir / "annotations.feather")
    rows = []
    for _, r in ann.iterrows():
        t_ms = int(r["timestamp_ns"]) // 1_000_000
        ecx, ecy, eyaw = ego_pose_at(t_ms)
        c, s = math.cos(eyaw), math.sin(eyaw)
        cx = ecx + r["tx_m"] * c - r["ty_m"] * s
        cy = ecy + r["tx_m"] * s + r["ty_m"] * c
        obj_yaw = _yaw_of(r["qw"], r["qx"], r["qy"], r["qz"]) + eyaw
        rows.append(dict(cx=float(cx), cy=float(cy), heading=float(obj_yaw),
                         length=float(r["length_m"]), width=float(r["width_m"]),
                         t=t_ms, track=str(r["track_uuid"])))
    # per-track finite-difference speed (== the objects pass-2 idiom).
    by_track: dict[str, list[int]] = {}
    for i, b in enumerate(rows):
        by_track.setdefault(b["track"], []).append(i)
    moving: list[dict] = []
    for idx in by_track.values():
        idx.sort(key=lambda i: rows[i]["t"])
        if len(idx) < 2:
            continue
        cx_t = np.asarray([rows[i]["cx"] for i in idx])
        cy_t = np.asarray([rows[i]["cy"] for i in idx])
        t_s = np.asarray([rows[i]["t"] for i in idx], dtype="float64") / 1000.0
        dt = np.gradient(t_s)
        spd = np.hypot(np.gradient(cx_t) / np.maximum(dt, 1e-3),
                       np.gradient(cy_t) / np.maximum(dt, 1e-3))
        for k, i in enumerate(idx):
            if spd[k] >= static_speed:
                b = rows[i]
                moving.append(dict(cx=b["cx"], cy=b["cy"], heading=b["heading"],
                                   length=b["length"], width=b["width"], t=b["t"]))
    print(f"  worldbuild: {len(moving)} moving-box observation(s) "
          f"(speed ≥ {static_speed} m/s)")
    return moving


ERASOR_SCAN_STRIDE = avc.ERASOR_SCAN_STRIDE  # decimate per-sweep ERASOR scans


def extract_worldbuild_av2(sweep_dir: Path, ego_pose_at, colorizer, moving_boxes,
                           voxel_m: float = WORLDBUILD_VOXEL_M,
                           collect_scans: bool = False,
                           scan_stride: int = ERASOR_SCAN_STRIDE,
                           color_mode: str = "mean"):
    """Stream-accumulate the AV2 static cloud + collect dynamic returns.

    Each AV2 lidar sweep is one ``<ts_ns>.feather`` in the EGO frame; per sweep we
    read all returns at FULL density, camera-colour them (7 ring cameras), lift
    x,y ego→city (yaw rotate + translate; z stays vehicle-frame), tag each return
    dynamic iff it falls inside a moving box at its sweep time, fold STATIC returns
    into the streaming voxel accumulator and collect the DYNAMIC returns un-merged.

    Shared by Worldbuild (one merged archive via ``av_common.worldbuild_merge``)
    and the scene-split "stage + actors" mode (two archives via
    ``av_common.finalize_stage`` / ``finalize_actors``). When ``collect_scans`` is
    set (scene-split + ERASOR), each sweep's STATIC returns are also kept
    (decimated by ``scan_stride`` — ERASOR bins are coarse, so a strided scan still
    fills them, and this bounds memory) as ``(xy (N,2), z (N), ego_xy (2,))`` for
    ``av_common.erasor_scrub``.

    Returns ``(acc, {xy,z,rgb,t})`` — or ``(acc, dynamic, scans)`` when
    ``collect_scans``.
    """
    acc = avc.WorldVoxelAccumulator(voxel_m, color_mode=color_mode)
    weighted = color_mode == "linear_weighted"
    d_xy, d_z, d_rgb, d_t = [], [], [], []
    scans: list[tuple] = []
    fallback = np.asarray(FALLBACK_RGB, dtype=np.uint8)
    for sweep in sorted(sweep_dir.glob("*.feather")):
        t_ns = int(sweep.stem)
        t_ms = t_ns // 1_000_000
        ecx, ecy, eyaw = ego_pose_at(t_ms)
        df = _read_feather(sweep)
        full = np.stack([df["x"].to_numpy(), df["y"].to_numpy(),
                         df["z"].to_numpy()], axis=1).astype(np.float64)
        if len(full) == 0:
            continue
        weight = None
        if colorizer is not None:
            if weighted:
                rgb, hit, weight = colorizer.colorize(full, t_ns, return_weight=True)
            else:
                rgb, hit = colorizer.colorize(full, t_ns)
            rgb = rgb.astype("float64")
            rgb[~hit] = fallback
        else:
            rgb = np.tile(fallback.astype("float64"), (len(full), 1))
            hit = np.zeros(len(full), dtype=bool)
        c, s = math.cos(eyaw), math.sin(eyaw)
        cx = ecx + full[:, 0] * c - full[:, 1] * s
        cy = ecy + full[:, 0] * s + full[:, 1] * c
        z = full[:, 2]
        xy = np.column_stack([cx, cy])
        dyn = avc.point_in_moving_boxes(xy, z, t_ms, moving_boxes)
        stat = ~dyn
        if stat.any():
            acc.add(xy[stat], z[stat], t_ms, rgb=rgb[stat], rgb_hit=hit[stat],
                    rgb_weight=(weight[stat] if weight is not None else None))
            if collect_scans:
                xy_s = xy[stat][::scan_stride]
                z_s = z[stat][::scan_stride]
                scans.append((xy_s, z_s, np.array([ecx, ecy], dtype="float64")))
        if dyn.any():
            d_xy.append(xy[dyn])
            d_z.append(z[dyn])
            d_rgb.append(rgb[dyn])
            d_t.append(np.full(int(dyn.sum()), t_ms, dtype="int64"))
    dynamic = dict(
        xy=np.concatenate(d_xy) if d_xy else np.empty((0, 2)),
        z=np.concatenate(d_z) if d_z else np.empty(0),
        rgb=np.concatenate(d_rgb) if d_rgb else np.empty((0, 3)),
        t=np.concatenate(d_t) if d_t else np.empty(0, dtype="int64"),
    )
    if collect_scans:
        return acc, dynamic, scans
    return acc, dynamic


# ── Sweep (--scan, AV2 only): per-return scan-time + phase→hue beam ──────────
# A SEPARATE `<id>-scan` bundle whose lidar carries each return's TRUE per-return
# scan time (sweep_ts + offset_ns→ms) so the playhead reveals the beam sweeping
# the scene, plus a scan_phase (offset normalized [0,1] within the sweep) coloured
# as a rotating rainbow (av_common.phase_to_rgb → the existing r/g/b path). AV2's
# lidar feathers ship `offset_ns` next to x/y/z; Waymo has no such column → no
# --scan there. Guard: if offset_ns is absent, fall back to the sweep time + warn.
SCAN_DECIMATE = 4  # single mid-density tier (no pyramid; protects the base archive)


def extract_scan_av2(sweep_dir: Path, ego_pose_at, to_lonlat, decimate: int = SCAN_DECIMATE):
    """Per-return scan-time + phase cloud → dict(lon, lat, timestamp, z, scan_phase, rgb).

    Each AV2 sweep is one ``<ts_ns>.feather`` (ego frame). Per return we read the
    (currently-dropped) ``offset_ns`` column, compute the TRUE per-return time
    ``sweep_ts_ms + round(offset_ns / 1e6)`` and ``scan_phase`` = offset normalized
    to ``[0,1]`` within the sweep, colour by ``av_common.phase_to_rgb``, and lift
    x,y ego→city→lon/lat (z stays vehicle-frame). Single mid-density tier (stride
    ``decimate``). If ``offset_ns`` is absent, falls back to the broadcast sweep
    time (phase 0) + a one-time warning.
    """
    out_lon, out_lat, out_t, out_z, out_phase = [], [], [], [], []
    warned = [False]
    for sweep in sorted(sweep_dir.glob("*.feather")):
        t_ns = int(sweep.stem)
        sweep_t_ms = t_ns // 1_000_000
        ecx, ecy, eyaw = ego_pose_at(sweep_t_ms)
        df = _read_feather(sweep)
        s = slice(None, None, decimate)
        x = df["x"].to_numpy()[s]
        y = df["y"].to_numpy()[s]
        z = df["z"].to_numpy()[s]
        if "offset_ns" in df:
            off = df["offset_ns"].to_numpy()[s].astype("float64")
            span = float(off.max() - off.min()) if len(off) else 0.0
            phase = (off - off.min()) / span if span > 0 else np.zeros(len(off))
            t_ms = sweep_t_ms + np.round(off / 1e6).astype("int64")
        else:
            if not warned[0]:
                print("  (--scan) WARNING: no offset_ns column — falling back to "
                      "the broadcast sweep time (no per-return scan reveal).")
                warned[0] = True
            phase = np.zeros(len(x))
            t_ms = np.full(len(x), sweep_t_ms, dtype="int64")
        c, s2 = math.cos(eyaw), math.sin(eyaw)
        cx = ecx + x * c - y * s2
        cy = ecy + x * s2 + y * c
        lon, lat = to_lonlat(cx, cy)
        out_lon.append(np.atleast_1d(lon))
        out_lat.append(np.atleast_1d(lat))
        out_t.append(np.asarray(t_ms, dtype="int64"))
        out_z.append(z)
        out_phase.append(phase)
    if not out_lon:
        raise SystemExit("scan extraction produced no points")
    phase = np.concatenate(out_phase)
    return dict(
        lon=np.concatenate(out_lon), lat=np.concatenate(out_lat),
        timestamp=np.concatenate(out_t), z=np.concatenate(out_z),
        scan_phase=phase, rgb=avc.phase_to_rgb(phase),
    )


def extract(log_dir: Path, city: str, drop_empty_boxes: bool = True,
            colorizer=None, surfel_decimate=None, worldbuild=None, scan=None,
            scene_split=None):
    """Pull ego / objects / lidar / telemetry / HD-map arrays out of one AV2 log.

    ``drop_empty_boxes`` (default ``True``) drops annotation cuboids with
    ``num_interior_pts == 0`` (fully-occluded / ghost ground-truth boxes that
    have no LIDAR returns inside them) so the rendered scene isn't littered with
    boxes around nothing.

    When ``colorizer`` (an ``Av2Colorizer``) is given, each decimated EGO-frame
    return is projected into the 7 ring cameras and assigned an RGB color
    (returns no camera sees take ``FALLBACK_RGB``); the returned ``lidar`` then
    carries an ``rgb`` array (else ``None``).

    When ``surfel_decimate`` is set, the LIDAR is extracted as an ORIENTED-SURFEL
    cloud instead (per-sweep k-NN covariance → ``compute_surfels`` /
    ``extract_lidar_surfels``); the ``lidar`` tuple's 7th element then carries the
    ``(N,7)`` surfel block and a ``colorizer`` is required (surfels are
    camera-coloured). Otherwise the 7th element is ``None``.

    When ``worldbuild`` (a ``{voxel_m, static_speed}`` dict) is set, the LIDAR is
    the consolidated CROSS-SWEEP-MERGED Worldbuild cloud (Feature 1): the ``lidar``
    slot is then a ``("worldbuild", merged)`` tuple where ``merged`` is the
    ``av_common.worldbuild_merge`` result dict (xy / z / rgb / timestamp /
    is_dynamic / world_class / surfels). A ``colorizer`` is required.

    When ``scan`` (a ``{decimate}`` dict) is set, the LIDAR is the per-return
    scan-time + phase→hue Sweep cloud (Feature 5, AV2 only): the ``lidar`` slot is
    a ``("scan", sc)`` tuple where ``sc`` is the ``extract_scan_av2`` dict
    (lon / lat / timestamp / z / scan_phase / rgb). NO colorizer needed.

    Returns ``(ego, objects, lidar, telemetry_fields, map_polys, map_lines)``
    where ``lidar = (lon, lat, t_ms, z, intensity, rgb, surfels)`` (or the
    worldbuild / scan 2-tuple above).
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
    telemetry_fields = avc.derive_telemetry(ego_t, ego_speed, ego_yaw)

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

    # Pass 2: real per-object velocity (av-cockpit.md §2 R2.3). Group rows by
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

    # --- lidar sweeps (ego frame → city → lon/lat) ---
    sweep_dir = log_dir / "sensors" / "lidar"
    if scan is not None:
        # SWEEP: per-return scan-time + phase→hue beam (a separate `<id>-scan`
        # bundle). NOT camera-coloured — the rainbow comes from scan_phase.
        print(f"  scan: per-return scan time + phase→hue (1/{scan['decimate']})…")
        sc = extract_scan_av2(sweep_dir, ego_pose_at, to_lonlat,
                              decimate=scan["decimate"])
        lidar = ("scan", sc)
    elif worldbuild is not None:
        # WORLDBUILD: one consolidated cross-sweep-merged cloud (city-metric),
        # static voxel-deduped + dynamic un-merged, surfels on the merged cloud.
        print(f"  worldbuild: merging all sweeps (voxel {worldbuild['voxel_m']} m, "
              f"static-speed {worldbuild['static_speed']} m/s)…")
        moving_boxes = _av2_moving_boxes(log_dir, ego_pose_at, worldbuild["static_speed"])
        acc, dynamic = extract_worldbuild_av2(
            sweep_dir, ego_pose_at, colorizer, moving_boxes,
            voxel_m=worldbuild["voxel_m"])
        merged = avc.worldbuild_merge(acc, dynamic)
        wb_lon, wb_lat = to_lonlat(merged["xy"][:, 0], merged["xy"][:, 1])
        merged["lon"] = np.atleast_1d(wb_lon)
        merged["lat"] = np.atleast_1d(wb_lat)
        lidar = ("worldbuild", merged)
    elif scene_split is not None:
        # SCENE-SPLIT ("stage + actors", Feature 6): decompose into TWO archives —
        # the STATIC stage (fixed environment, accumulated + voxel-deduped across
        # every sweep) and the DYNAMIC actors (returns inside MOVING boxes, kept
        # per-sweep). Optional ERASOR second pass scrubs residual ghost trails off
        # the stage. Each half is finalized + surfel-fit independently.
        print(f"  scene-split: stage voxel {scene_split['voxel_m']} m, "
              f"static-speed {scene_split['static_speed']} m/s, "
              f"erasor={'on' if scene_split['erasor'] else 'off'}…")
        moving_boxes = _av2_moving_boxes(log_dir, ego_pose_at, scene_split["static_speed"])
        acc, dynamic, scans = extract_worldbuild_av2(
            sweep_dir, ego_pose_at, colorizer, moving_boxes,
            voxel_m=scene_split["voxel_m"], collect_scans=True,
            color_mode="linear_weighted")  # photoreal cross-sweep colour fusion
        # Finalize the accumulator ONCE and share the arrays with ERASOR so the
        # drop-mask stays index-aligned with finalize_stage's voxel order.
        static_final = acc.finalize()
        drop = (avc.erasor_scrub(static_final, scans, **scene_split["erasor_params"])
                if scene_split["erasor"] else None)
        if drop is not None:
            print(f"  scene-split: ERASOR scrubbed {int(drop.sum())} / {len(drop)} "
                  f"stage voxel(s) as residual dynamic")
        stage = avc.finalize_stage(static_final, drop_mask=drop,
                                   adaptive=scene_split.get("adaptive"),
                                   grade=scene_split.get("grade"))
        actors = avc.finalize_actors(dynamic, grade=scene_split.get("actor_grade"),
                                     denoise=scene_split.get("denoise"))
        for d in (stage, actors):
            s_lon, s_lat = to_lonlat(d["xy"][:, 0], d["xy"][:, 1])
            d["lon"] = np.atleast_1d(s_lon)
            d["lat"] = np.atleast_1d(s_lat)
        print(f"  scene-split: {len(stage['z'])} stage + {len(actors['z'])} "
              f"actor point(s)")
        lidar = ("scene_split", {"stage": stage, "actors": actors})
    elif surfel_decimate is not None:
        # ORIENTED-SURFEL tier: per-sweep k-NN covariance → quaternion + extents +
        # confidence for the client's SplatLayer (camera-coloured; a colorizer is
        # required — generate() forces one on in surfel mode).
        lidar = extract_lidar_surfels(sweep_dir, ego_pose_at, to_lonlat, colorizer,
                                      decimate=surfel_decimate)
    else:
        l_lon, l_lat, l_t, l_z, l_i = [], [], [], [], []
        l_rgb = [] if colorizer is not None else None
        fallback = np.asarray(FALLBACK_RGB, dtype=np.uint8)
        for sweep in sorted(sweep_dir.glob("*.feather")):
            t_ns = int(sweep.stem)
            t_ms = t_ns // 1_000_000
            ecx, ecy, eyaw = ego_pose_at(t_ms)
            df = _read_feather(sweep)
            x = df["x"].to_numpy()[::FINEST_DECIMATE]
            y = df["y"].to_numpy()[::FINEST_DECIMATE]
            z = df["z"].to_numpy()[::FINEST_DECIMATE]
            intensity = (df["intensity"].to_numpy()[::FINEST_DECIMATE]
                         if "intensity" in df else np.full(len(x), 128.0))
            if l_rgb is not None:
                # Project the SAME decimated EGO-frame returns into the ring cameras
                # (before the city lift — the cameras are calibrated to the ego frame).
                rgb_chunk, got = colorizer.colorize(np.stack([x, y, z], axis=1), t_ns)
                rgb_chunk[~got] = fallback
                l_rgb.append(rgb_chunk)
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
                 np.concatenate(l_t), np.concatenate(l_z), np.concatenate(l_i),
                 np.concatenate(l_rgb) if l_rgb is not None else None,
                 None)  # no surfels in the default (round-dot) path

    # --- HD map (static substrate, av-cockpit.md §2 R2.2) ---
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
    n_center_err = 0
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
        except (KeyError, IndexError, ValueError, ZeroDivisionError):
            # Degenerate boundaries the devkit can't interpolate — counted and
            # reported below rather than silently thinning the map.
            n_center_err += 1
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
    if n_center_err:
        print(f"  warning: {n_center_err} lane centerline(s) failed devkit "
              f"interpolation and were skipped")
    return map_polys, map_lines


def generate(args):
    origin_lat, origin_lon, origin_note = _resolve_origin(args)
    print(f"Argoverse 2 log {args.log_dir.name} @ {args.city}  (origin: {origin_note})")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # --surfel / --worldbuild / --scan / --contours / --scene-split are mutually
    # exclusive LIDAR modes.
    surfel_mode = bool(args.surfel)
    worldbuild_mode = bool(args.worldbuild)
    scan_mode = bool(args.scan)
    contour_mode = bool(args.contours)
    scene_split_mode = bool(args.scene_split)
    lod_mode = bool(args.lod)
    if sum((surfel_mode, worldbuild_mode, scan_mode, contour_mode,
            scene_split_mode, lod_mode)) > 1:
        raise SystemExit(
            "--surfel / --worldbuild / --scan / --contours / --scene-split / --lod "
            "are mutually exclusive")

    # Optional camera-projection colorize (+ its correctness overlay). The SURFEL,
    # WORLDBUILD and SCENE-SPLIT paths imply camera colour (the disks are painted
    # from the ring cameras), so they force a colorizer on regardless of
    # --colorize. --scan does NOT need one (its rainbow comes from scan_phase).
    colorizer = None
    if (args.colorize or surfel_mode or worldbuild_mode or scene_split_mode
            or args.debug_overlay is not None):
        colorizer = Av2Colorizer(args.log_dir)
    if args.debug_overlay is not None:
        debug_overlay_av2(args.log_dir, out, colorizer, args.debug_overlay)
        print("  (--debug-overlay) wrote projection overlays; stopping.")
        return

    if surfel_mode:
        print(f"  fitting oriented surfels per sweep (1/{args.surfel_decimate}, "
              f"k={SURFEL_K})…")
    ego, objects, lidar, telemetry_fields, map_polys, map_lines = extract(
        args.log_dir, args.city, drop_empty_boxes=not args.keep_empty_boxes,
        colorizer=colorizer,
        surfel_decimate=(args.surfel_decimate if surfel_mode else None),
        worldbuild=(dict(voxel_m=args.worldbuild_voxel,
                         static_speed=args.worldbuild_static_speed)
                    if worldbuild_mode else None),
        scan=(dict(decimate=args.scan_decimate) if scan_mode else None),
        scene_split=(avc.scene_split_config(args) if scene_split_mode else None))
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

    # --- lidar ---
    # SURFEL mode bakes a SINGLE oriented-Gaussian-surfel tier (per-sweep k-NN
    # covariance → quaternion + extents + confidence) for the client's SplatLayer.
    # WORLDBUILD bakes the consolidated cross-sweep-merged cloud (one 2-tuple).
    # The default path builds the round-dot density tiers (stride the finest cloud).
    lidar_densities = []
    if contour_mode:
        # DENSITY ISO-LINES: contour where the cloud clusters, per playhead window.
        # With --contour-z-step it goes TRUE-3D (density contoured per height layer,
        # each contour tagged with its real `z_layer` altitude). Builds ONE
        # windowed-LineString `lidar/` archive (rendered by AnimatedPathLayer); the
        # returns come from the default (round-dot) extract path, then strided.
        fl_rgb = None  # iso-lines are density-colored, not camera-colored
        cl_lon, cl_lat, cl_t, cl_z, _ci, _rgb, _sf = lidar
        dec = max(1, args.contour_decimate)
        cl_lon, cl_lat = np.asarray(cl_lon)[::dec], np.asarray(cl_lat)[::dec]
        cl_t, cl_z = np.asarray(cl_t)[::dec], np.asarray(cl_z)[::dec]
        mode3d = "3D z-layered" if args.contour_z_step > 0 else "2D flat"
        print(f"  density iso-lines ({mode3d}) from {len(cl_lon)} returns (1/{dec})…")
        contours = avc.build_density_contours(
            cl_lon, cl_lat, cl_z, cl_t,
            anchor_lat=float(np.median(cl_lat)), **avc.contour_kwargs(args))
        iso_pq = out / "lidar.parquet"
        n = avc.write_contour_lines(iso_pq, contours)
        if not args.skip_build and n > 0:
            tier_out = out / "lidar"
            shutil.rmtree(tier_out, ignore_errors=True)  # prune orphan packs on rebuild
            # kind="line": windowed LineStrings, NO --simplify / NO quantize-coords
            # (the line-quantize GL bug). Bucket = the window step.
            avc.run_stt_build(iso_pq, tier_out, "line", stt_build=args.stt_build,
                              temporal_bucket=f"{int(args.contour_step)}ms",
                              min_zoom=LIDAR_MIN_ZOOM)
            iso_pq.unlink(missing_ok=True)
        elif n == 0:
            print("  (--contours) NO contours produced — tune --contour-cell / "
                  "--contour-min-len / --contour-width.")
        lidar_densities.append({"id": "iso", "label": "Iso-lines", "points": int(n),
                                "url": "lidar/manifest.json"})
        n_lidar = int(n)
    elif scan_mode:
        _tag, sc = lidar
        fl_rgb = sc["rgb"]  # phase→hue rainbow (NOT camera) — mark colored
        print(f"  scan: {len(sc['z'])} returns over "
              f"[{int(sc['timestamp'].min())}, {int(sc['timestamp'].max())}] ms "
              f"(per-return scan time + phase→hue)")
        tier_pq = out / "lidar.parquet"
        n = avc.write_lidar_points(
            tier_pq, lon=sc["lon"], lat=sc["lat"], timestamp=sc["timestamp"],
            z=sc["z"], rgb=sc["rgb"], scan_phase=sc["scan_phase"])
        if not args.skip_build:
            tier_out = out / "lidar"
            shutil.rmtree(tier_out, ignore_errors=True)  # prune orphan packs on rebuild
            # single mid-density tier (no pyramid beyond the zoom-floor clamp);
            # scan_phase quantization rides in the canonical attr map.
            avc.run_stt_build(tier_pq, tier_out, "point",
                              stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                              min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                              quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                              vector_groups=avc.lidar_vector_groups(surfel=False, colored=True),
                              point_elevation_column="z")
            tier_pq.unlink(missing_ok=True)
        lidar_densities.append({"id": "scan", "label": "Sweep", "points": int(n),
                                "url": "lidar/manifest.json"})
        n_lidar = int(n)
    elif worldbuild_mode:
        _tag, wb = lidar
        n_dyn = int(wb["is_dynamic"].sum())
        fl_rgb = wb["rgb"]  # marks the bundle as camera-colored in scene.json
        n_seen = int((wb["rgb"] != np.asarray(FALLBACK_RGB, np.uint8)).any(axis=1).sum())
        print(f"  worldbuild: {len(wb['z'])} merged points "
              f"({len(wb['z']) - n_dyn} static + {n_dyn} dynamic); camera color "
              f"{n_seen}/{len(wb['rgb'])} ({100 * n_seen / max(len(wb['rgb']), 1):.0f}%)")
        tier_pq = out / "lidar.parquet"
        n = avc.write_lidar_points(
            tier_pq, lon=wb["lon"], lat=wb["lat"], timestamp=wb["timestamp"],
            z=wb["z"], rgb=wb["rgb"], surfels=wb["surfels"],
            world_class=wb["world_class"], is_dynamic=wb["is_dynamic"])
        if not args.skip_build:
            tier_out = out / "lidar"
            shutil.rmtree(tier_out, ignore_errors=True)  # prune orphan packs on rebuild
            avc.run_stt_build(tier_pq, tier_out, "point",
                              stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                              min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                              quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                              vector_groups=avc.lidar_vector_groups(surfel=True, colored=True))
            tier_pq.unlink(missing_ok=True)
        lidar_densities.append({"id": "world", "label": "Worldbuild", "points": int(n),
                                "url": "lidar/manifest.json"})
        n_lidar = int(n)
    elif scene_split_mode:
        # SCENE-SPLIT ("stage + actors"): TWO archives from one source. The STATIC
        # stage is a timeless full-range cloud (loads once, persists the whole
        # replay); the DYNAMIC actors are time-bucketed + animated. Identical
        # quantize / vector-group knobs on both so their stt:qa affines + interleaved
        # colour/quat columns decode the same way (content-addressed dedup).
        _tag, ss = lidar
        stage, actors = ss["stage"], ss["actors"]
        fl_rgb = stage["rgb"]  # mark the bundle camera-colored in scene.json
        n_stage = len(stage["z"])
        n_actors = len(actors["z"])
        # STATIC "stage" → static.parquet (one whole-scene bucket via
        # static_full_range + a per-point end_timestamp = scene end).
        stage_pq = out / "static.parquet"
        avc.write_lidar_points(
            stage_pq, lon=stage["lon"], lat=stage["lat"],
            timestamp=stage["timestamp"], end_timestamp=t_end, z=stage["z"],
            rgb=stage["rgb"], surfels=stage["surfels"],
            world_class=stage["world_class"], is_dynamic=stage["is_dynamic"])
        # DYNAMIC "actors" → dynamic.parquet (per-sweep animated).
        actors_pq = out / "dynamic.parquet"
        avc.write_lidar_points(
            actors_pq, lon=actors["lon"], lat=actors["lat"],
            timestamp=actors["timestamp"], z=actors["z"],
            rgb=actors["rgb"], surfels=actors["surfels"],
            world_class=actors["world_class"], is_dynamic=actors["is_dynamic"])
        if not args.skip_build and n_stage > 0:
            stage_out = out / "static"
            shutil.rmtree(stage_out, ignore_errors=True)  # prune orphan packs
            avc.run_stt_build(
                stage_pq, stage_out, "point", stt_build=args.stt_build,
                min_zoom=args.stage_min_zoom, quantize_coords=LIDAR_QUANTIZE_M,
                quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                vector_groups=avc.lidar_vector_groups(surfel=True, colored=True),
                static_full_range=True)
            stage_pq.unlink(missing_ok=True)
        if not args.skip_build and n_actors > 0:
            actors_out = out / "dynamic"
            shutil.rmtree(actors_out, ignore_errors=True)  # prune orphan packs
            avc.run_stt_build(
                actors_pq, actors_out, "point", stt_build=args.stt_build,
                temporal_bucket=args.temporal_bucket,
                min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                vector_groups=avc.lidar_vector_groups(surfel=True, colored=True))
            actors_pq.unlink(missing_ok=True)
        elif n_actors == 0:
            print("  scene-split: NO dynamic (moving-actor) returns — the scene has "
                  "no tracked movers above --scene-static-speed.")
        lidar_densities.append({"id": "stage", "label": "Stage", "points": n_stage,
                                "url": "static/manifest.json"})
        lidar_densities.append({"id": "actors", "label": "Actors", "points": n_actors,
                                "url": "dynamic/manifest.json"})
        n_lidar = n_actors
        print(f"  scene-split: built static/ ({n_stage} stage) + dynamic/ "
              f"({n_actors} actor) point(s)")
    elif surfel_mode:
        fl_lon, fl_lat, fl_t, fl_z, fl_i, fl_rgb, fl_surf = lidar
        if fl_rgb is not None:
            n_seen = int((fl_rgb != np.asarray(FALLBACK_RGB, np.uint8)).any(axis=1).sum())
            print(f"  camera colorize: {n_seen}/{len(fl_rgb)} returns "
                  f"({100 * n_seen / max(len(fl_rgb), 1):.0f}%) got a camera color")
        print(f"  surfels: {len(fl_surf)} oriented disks → single 'surfel' tier")
        tier_pq = out / "lidar.parquet"
        n = avc.write_lidar_points(tier_pq, lon=fl_lon, lat=fl_lat, timestamp=fl_t,
                                   z=fl_z, intensity=fl_i, rgb=fl_rgb, surfels=fl_surf)
        if not args.skip_build:
            tier_out = out / "lidar"
            shutil.rmtree(tier_out, ignore_errors=True)  # prune orphan packs on rebuild
            avc.run_stt_build(tier_pq, tier_out, "point",
                              stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                              min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                              quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                              vector_groups=avc.lidar_vector_groups(
                                  surfel=True, colored=True))
            tier_pq.unlink(missing_ok=True)
        lidar_densities.append({"id": "surfel", "label": "Surfel", "points": int(n),
                                "url": "lidar/manifest.json"})
        n_lidar = int(n)
    elif lod_mode:
        # ADDITIVE-OCTREE LOD: ONE 'lidar-lod/' archive replacing the 5 fixed
        # density tiers. Each return is materialized at a single per-sweep home
        # zoom (avc.lod_home_zoom) and stt-build places it in exactly that zoom's
        # tiles via --min-zoom-field=--max-zoom-field=home_zoom — so each zoom
        # level holds only its residual and the client loads the union. Same
        # source as the default (round-dot) path: camera RGB when --colorize, else
        # the height-band ramp (write_lidar_points auto-fills it).
        fl_lon, fl_lat, fl_t, fl_z, fl_i, fl_rgb, _fl_surf = lidar
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
        # Self-contained bundle: the single LOD archive lands in the standard
        # `lidar/` dir (same idiom as --surfel / --scan), so a `<id>-lod` scene
        # bundle's avLidarUrl resolves to lidar/manifest.json. The home_zoom
        # column + --min/max-zoom-field make each zoom level hold only its
        # residual; the client loads the union (lodMode:'additive').
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
                point_elevation_column="z",
                min_zoom_field="home_zoom", max_zoom_field="home_zoom")
            tier_pq.unlink(missing_ok=True)
        lidar_densities.append({
            "id": "lod", "label": "Zoom LOD", "points": int(n),
            "url": "lidar/manifest.json", "lod": True,
            "minZoom": args.lod_min_zoom, "maxZoom": args.lod_max_zoom})
        n_lidar = int(n)
    else:
        fl_lon, fl_lat, fl_t, fl_z, fl_i, fl_rgb, fl_surf = lidar
        if fl_rgb is not None:
            n_seen = int((fl_rgb != np.asarray(FALLBACK_RGB, np.uint8)).any(axis=1).sum())
            print(f"  camera colorize: {n_seen}/{len(fl_rgb)} returns "
                  f"({100 * n_seen / max(len(fl_rgb), 1):.0f}%) got a camera color")
        skip_tiers = {t.strip() for t in args.skip_tiers.split(",") if t.strip()}
        for tier_id, label, dec, dname, radius_px, radius_min_px in LIDAR_DENSITY_TIERS:
            if tier_id in skip_tiers:
                continue  # caller dropped this tier (e.g. the heavy 14M `full` on tight disk)
            stride = dec // FINEST_DECIMATE  # exact every-Nth sample of the finest cloud
            s = slice(None, None, stride)
            tier_pq = out / f"{dname}.parquet"
            n = avc.write_lidar_points(tier_pq, lon=fl_lon[s], lat=fl_lat[s],
                                       timestamp=fl_t[s], z=fl_z[s], intensity=fl_i[s],
                                       rgb=(fl_rgb[s] if fl_rgb is not None else None))
            if not args.skip_build:
                tier_out = out / dname
                shutil.rmtree(tier_out, ignore_errors=True)  # prune orphan packs on rebuild
                avc.run_stt_build(tier_pq, tier_out, "point",
                                  stt_build=args.stt_build, temporal_bucket=args.temporal_bucket,
                                  min_zoom=LIDAR_MIN_ZOOM, quantize_coords=LIDAR_QUANTIZE_M,
                                  quantize_attrs=avc.lidar_quantize_attrs(LIDAR_QUANTIZE_M),
                                  vector_groups=avc.lidar_vector_groups(
                                      surfel=False, colored=True),
                                  point_elevation_column="z")
                tier_pq.unlink(missing_ok=True)  # drop the big intermediate (raw log is deleted anyway)
            lidar_densities.append({"id": tier_id, "label": label, "points": int(n),
                                    "url": f"{dname}/manifest.json",
                                    "radius": radius_px, "radiusMinPixels": radius_min_px})
        n_lidar = lidar_densities[0]["points"]  # default = small tier

    # --- HD map (static, full-range) parquet (av-cockpit.md §2 R2.2) ---
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
    ego_path = avc.downsample_ego_path(ego_t, ego_lon, ego_lat)

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
        "lidar": {"url": "lidar/manifest.json", "points": int(n_lidar),
                  "densities": lidar_densities,
                  # Per-point RGB sampled from the 7 ring cameras → the frontend
                  # paints each return (see AnimatedPointLayer.rgbColorColumns).
                  **({"colored": True} if fl_rgb is not None else {}),
                  # Surfel / worldbuild modes bake oriented-Gaussian columns
                  # (quaternion + extents + confidence); the cockpit renders the
                  # tier with SplatLayer instead of AnimatedPointLayer.
                  **({"splat": "surfel"} if (surfel_mode or worldbuild_mode) else {}),
                  # Worldbuild mode: the consolidated cross-sweep-merged cloud
                  # (world_class static/dynamic + is_dynamic columns); the frontend
                  # keys the static-vs-dynamic styling on it.
                  **({"style": "world"} if worldbuild_mode else {}),
                  # Scan mode: the per-return scan-time + phase→hue Sweep cloud
                  # (scan_phase column + a rotating-rainbow r/g/b). A separate
                  # `<id>-scan` bundle; the frontend reveals the beam over time.
                  **({"scan": True, "style": "scan"} if scan_mode else {}),
                  # Contour mode: the `lidar/` archive is windowed density iso-line
                  # LineStrings (density_band + z_layer columns); the frontend
                  # renders it with AnimatedPathLayer, lifting by z_layer in 3D.
                  **({"style": "iso"} if contour_mode else {})},
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
        "telemetry": {"url": "telemetry.json"},
    }
    if cam_frames:
        streams["camera"] = {"url": "cameras.json"}
    if scene_split_mode:
        # Scene-split overrides the single `lidar` stream with two surfel archives:
        # `lidar` aliases the animated DYNAMIC actors (so single-stream cockpit code
        # still resolves), plus a `stage` stream for the timeless STATIC backdrop
        # (style="stage", static=true → rendered with no playhead filter).
        streams["lidar"] = {"url": "dynamic/manifest.json", "points": int(n_actors),
                            "colored": True, "splat": "surfel", "style": "actors",
                            "densities": lidar_densities}
        streams["stage"] = {"url": "static/manifest.json", "points": int(n_stage),
                            "colored": True, "splat": "surfel", "style": "stage",
                            "static": True}

    # --- build packed archives (telemetry + camera are sidecars, written above) ---
    if not args.skip_build:
        # (lidar density tiers were built above, per density.)
        avc.run_stt_build(obj_pq, out / "objects", "point",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        avc.run_stt_build(ego_pq, out / "ego", "trips",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)
        # Feature 2 tracks — kind="line" (windowed LineStrings, NO --simplify /
        # quantize-coords so vertex_timestamps stay aligned to the coords).
        shutil.rmtree(out / "tracks", ignore_errors=True)
        avc.run_stt_build(track_pq, out / "tracks", "line",
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
          f"{n_objects} objects, {n_tracks} tracks, {n_lidar} lidar, "
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
    p.add_argument("--colorize", action="store_true",
                   help="bake per-point RGB by projecting each LIDAR return into the "
                        "7 ring cameras (adds r/g/b columns; needs the calibration/ + "
                        "all ring-camera JPGs). Returns no camera sees get a slate "
                        "fallback. Pairs with the cockpit's camera-colored splat.")
    # --contours + all --contour-* knobs (shared with waymo_extract.py).
    avc.add_contour_args(p)
    p.add_argument("--surfel", action="store_true",
                   help="bake ORIENTED GAUSSIAN SURFELS instead of the round-dot "
                        "density tiers: per-sweep k-NN covariance → orientation "
                        "quaternion + in-plane extents + confidence per return, in a "
                        "single 'surfel' tier the cockpit renders with SplatLayer "
                        "(scene.json lidar.splat='surfel'). Forces camera colorize "
                        "(surfels are camera-colored — needs calibration/ + all 7 "
                        "ring-camera JPGs). Output a sibling '<id>-surfel' bundle.")
    p.add_argument("--surfel-decimate", type=int, default=SURFEL_DECIMATE,
                   help=f"render decimation for the surfel tier (default {SURFEL_DECIMATE}; "
                        "orientation still uses the FULL-density neighbourhood)")
    p.add_argument("--worldbuild", action="store_true",
                   help="WORLDBUILD view: ONE consolidated cross-sweep-merged cloud. "
                        "STATIC returns voxel-deduped over the whole log (a wall seen "
                        "100× → one crisp surface, keyed first_seen); DYNAMIC "
                        "(moving-actor) returns pass through UN-merged, each keeping "
                        "its own per-sweep time. Adds world_class (static/dynamic "
                        "categorical) + is_dynamic (0/1 numeric); surfels fit on the "
                        "merged cloud → SplatLayer. Implies camera colour; mutually "
                        "exclusive with --surfel / --scan. Output a sibling "
                        "'<id>-world' bundle.")
    p.add_argument("--worldbuild-voxel", type=float, default=WORLDBUILD_VOXEL_M,
                   dest="worldbuild_voxel",
                   help=f"static-merge voxel edge in metres (default {WORLDBUILD_VOXEL_M}).")
    p.add_argument("--worldbuild-static-speed", type=float,
                   default=WORLDBUILD_STATIC_SPEED, dest="worldbuild_static_speed",
                   help="a track at/above this speed (m/s) is MOVING → its returns are "
                        "tagged dynamic + passed through un-merged (default "
                        f"{WORLDBUILD_STATIC_SPEED}).")
    # Shared scene-split CLI (avc.add_scene_split_args) — same knobs as Waymo.
    avc.add_scene_split_args(p)
    p.add_argument("--scan", action="store_true",
                   help="SWEEP view (AV2 only): a separate '<id>-scan' bundle whose "
                        "lidar carries each return's TRUE per-return scan time "
                        "(sweep_ts + offset_ns→ms) so the playhead reveals the beam "
                        "sweeping the scene, coloured by scan_phase (offset normalized "
                        "[0,1]) as a rotating rainbow. Single mid-density tier, NOT a "
                        "base-archive change. Does NOT need the camera colorizer.")
    p.add_argument("--scan-decimate", type=int, default=SCAN_DECIMATE,
                   dest="scan_decimate",
                   help=f"render decimation for the scan tier (default {SCAN_DECIMATE}).")
    p.add_argument("--lod", action="store_true",
                   help="ADDITIVE-OCTREE LOD: bake ONE 'lidar-lod/' archive where "
                        "each return is materialized at a single per-sweep 'home "
                        "zoom' (hierarchical voxel subsample). Coarse zooms are "
                        "sparse overviews; finer zooms add ONLY the residual "
                        "detail, so the client loads the union minZoom..cameraZoom "
                        "and zooming in streams detail without re-fetching the "
                        "coarse cloud. Lossless (union = full cloud) and ~½ the "
                        "bytes of the 5 fixed density tiers it replaces. Mutually "
                        "exclusive with --surfel/--worldbuild/--scan/--contours/"
                        "--scene-split.")
    p.add_argument("--lod-min-zoom", type=int, default=avc.LOD_MIN_ZOOM,
                   dest="lod_min_zoom",
                   help=f"coarsest LOD level (default {avc.LOD_MIN_ZOOM}).")
    p.add_argument("--lod-max-zoom", type=int, default=avc.LOD_MAX_ZOOM,
                   dest="lod_max_zoom",
                   help=f"finest LOD level — the full-density residual lands here "
                        f"(default {avc.LOD_MAX_ZOOM}).")
    p.add_argument("--lod-px-per-point", type=float, default=avc.LOD_PX_PER_POINT,
                   dest="lod_px_per_point",
                   help=f"target return spacing (screen px) at a point's home zoom "
                        f"(default {avc.LOD_PX_PER_POINT}; smaller = denser per level).")
    p.add_argument("--lod-uniform", action="store_true", dest="lod_uniform",
                   help="use a PLAIN uniform voxel grid for the LOD home-zoom "
                        "(default is geometry-aware: keep high-curvature edges/poles "
                        "at coarse zoom, defer flat surfaces). For A/B comparison.")
    p.add_argument("--debug-overlay", type=int, default=None, metavar="SWEEP",
                   help="project sweep SWEEP into every ring camera, dots colored by "
                        "depth, save _debug_overlay_<CAM>.png, and STOP "
                        "(projection-correctness check).")
    p.add_argument("--keep-empty-boxes", action="store_true",
                   help="keep annotation cuboids with num_interior_pts==0 "
                        "(default drops these occluded/ghost GT boxes)")
    p.add_argument("--skip-tiers", default="",
                   help="comma list of LIDAR density tier ids to NOT build "
                        "(e.g. 'full' to drop the heavy ~14M / ~1.1GB tier on tight disk)")
    p.add_argument("--temporal-bucket", default="200ms",
                   help="stt-build temporal bucket (AV2 lidar ~10 Hz → 100-200ms)")
    p.add_argument("--stt-build", default="stt-build", help="stt-build binary path")
    p.add_argument("--skip-build", action="store_true",
                   help="stop at GeoParquet + JSON (don't run stt-build)")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
