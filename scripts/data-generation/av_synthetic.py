#!/usr/bin/env python3
"""Procedurally generate a complete, valid AV scene bundle — NO download.

This is the cockpit's **bootstrap scene** (``sceneId: av-synthetic``). It needs
nothing but ``numpy`` / ``pyarrow`` / ``shapely`` (via ``av_common``) and emits a
byte-identical bundle to the real-data adapters, so the entire AV cockpit +
catalog is runnable/verifiable today, before any login-gated download.

It synthesises a realistic ~20 s urban drive at the documented **boston-seaport**
origin (av-cockpit.md §1):

* **ego** — gently curving trajectory through an intersection, sampled at 10 Hz,
  with per-vertex speed (m/s).
* **objects** — ~12 tracked agents (cars, a couple pedestrians, a cyclist) on
  plausible straight / curved paths with realistic box dims + headings, sampled
  at 10 Hz; one point per agent per sample.
* **lidar** — a rotating-ring sweep per ~0.1 s sampled onto the ground plane plus
  a few facade walls, decimated to ~150–250k points total, each carrying a
  ``height_band`` + ``intensity``.
* **telemetry** — sinusoidal-but-plausible CAN gauges (speed / steer / accel /
  throttle / brake) at 50 Hz.
* **cameras** — ~20 placeholder camera-frame entries (valid JSON; images
  omitted — the cockpit hides the inset when a frame url is missing).

Everything is deterministic (fixed RNG seed) so rebuilds are reproducible.

Usage:

    python av_synthetic.py                         # full build (needs stt-build)
    python av_synthetic.py --skip-build            # parquet + JSON only
    python av_synthetic.py --stt-build ../../target/release/stt-build
    python av_synthetic.py --out /path/to/av-synthetic

Pipeline:  (this script)  →  GeoParquet + sidecar JSON  →  stt-build  →  packed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import av_common as avc

# ── Scene constants ──────────────────────────────────────────────────────────
SCENE_ID = "av-synthetic"
LOCATION = "boston-seaport"
ORIGIN_LAT, ORIGIN_LON = avc.NUSCENES_MAP_ORIGINS[LOCATION]

# Place the scene a little inside the map so the basemap has context around it.
SCENE_CX, SCENE_CY = 600.0, 500.0  # local-frame centre (m from SW origin)

T0_MS = 1_700_000_000_000  # arbitrary fixed wall-clock start (Unix-ms), deterministic
DURATION_S = 20.0
SCENE_HZ = 10.0  # ego / object / lidar-sweep cadence
TELEMETRY_HZ = 50.0
CAMERA_HZ = 1.0

# Repo-root-anchored default so the bundle lands in the showcase data dir
# regardless of the cwd the script is invoked from (this file lives at
# <repo>/scripts/data-generation/av_synthetic.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "examples/showcase/public/data/av-synthetic"


# ── Ego trajectory ───────────────────────────────────────────────────────────
def make_ego(rng: np.random.Generator):
    """A gentle S-through-an-intersection at ~8 m/s, sampled at SCENE_HZ.

    Returns ``(t_ms, x, y, heading, speed)`` arrays in the local map frame.
    """
    n = int(DURATION_S * SCENE_HZ) + 1
    t_s = np.linspace(0.0, DURATION_S, n)
    t_ms = (T0_MS + t_s * 1000.0).astype("int64")

    # Drive roughly +x (east) with a smooth lateral weave + a slowdown at the
    # intersection mid-scene (so speed/accel telemetry has structure).
    base_speed = 8.0  # m/s ≈ 29 km/h
    slow = 3.0 * np.exp(-((t_s - DURATION_S / 2) ** 2) / (2 * 2.2**2))  # dip mid-scene
    speed = np.clip(base_speed - slow, 1.5, base_speed)

    # Integrate heading: a gentle curve that bends left then right.
    heading = math.radians(8.0) * np.sin(2 * math.pi * t_s / DURATION_S)  # ±8° weave
    dt = np.gradient(t_s)
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = SCENE_CX - (speed.mean() * DURATION_S) / 2 + np.cumsum(vx * dt)
    y = SCENE_CY + np.cumsum(vy * dt) - np.cumsum(vy * dt)[0]
    return t_ms, x, y, heading, speed


# ── Tracked agents ───────────────────────────────────────────────────────────
def make_objects(rng: np.random.Generator, ego):
    """~12 plausible agents on straight / curved paths, sampled at SCENE_HZ.

    Returns flat per-(agent, sample) arrays ready for the objects writer.
    """
    ego_t, ego_x, ego_y, _, _ = ego
    n = len(ego_t)
    t_s = (ego_t - ego_t[0]) / 1000.0

    # (category, length, width, height, base_speed) templates.
    fleet = (
        [("car", 4.6, 1.9, 1.5)] * 6
        + [("truck", 7.5, 2.5, 3.2)]
        + [("bus", 11.0, 2.9, 3.4)]
        + [("bicycle", 1.8, 0.6, 1.4)]
        + [("pedestrian", 0.6, 0.6, 1.75)] * 2
        + [("motorcycle", 2.1, 0.8, 1.4)]
    )

    out_lon, out_lat, out_t = [], [], []
    out_cat, out_head, out_len, out_wid, out_hgt = [], [], [], [], []
    out_track, out_speed = [], []

    for i, (cat, length, width, height) in enumerate(fleet):
        # Seed each agent near the ego corridor on a plausible lane.
        lane = rng.uniform(-18.0, 18.0)
        along0 = rng.uniform(-40.0, 80.0)
        oncoming = rng.random() < 0.45 and cat in ("car", "truck", "bus", "motorcycle")
        if cat == "pedestrian":
            spd = rng.uniform(0.8, 1.6)
            cross = rng.random() < 0.5  # crossing vs walking along
            dir_ang = (math.pi / 2 if cross else 0.0) + rng.uniform(-0.2, 0.2)
        elif cat == "bicycle":
            spd = rng.uniform(3.0, 5.0)
            dir_ang = rng.uniform(-0.15, 0.15)
        else:
            spd = rng.uniform(6.0, 11.0)
            dir_ang = math.pi if oncoming else 0.0
            dir_ang += rng.uniform(-0.05, 0.05)

        # Slight curvature so paths aren't all dead-straight.
        curve = rng.uniform(-0.004, 0.004)
        ang = dir_ang + curve * (t_s * spd)
        x0 = SCENE_CX + along0 * math.cos(dir_ang) - lane * math.sin(dir_ang)
        y0 = SCENE_CY + along0 * math.sin(dir_ang) + lane * math.cos(dir_ang)
        x = x0 + np.cumsum(spd * np.cos(ang) * np.gradient(t_s))
        y = y0 + np.cumsum(spd * np.sin(ang) * np.gradient(t_s))

        lon, lat = avc.local_to_lonlat(x, y, ORIGIN_LAT, ORIGIN_LON)
        # heading in WORLD frame, 0 = +x (east), CCW+ (contract §2c).
        head = ang
        tid = f"obj-{i:02d}-{cat}"

        out_lon.extend(lon.tolist())
        out_lat.extend(lat.tolist())
        out_t.extend(ego_t.tolist())
        out_cat.extend([cat] * n)
        out_head.extend(head.tolist())
        out_len.extend([length] * n)
        out_wid.extend([width] * n)
        out_hgt.extend([height] * n)
        out_track.extend([tid] * n)
        out_speed.extend([spd] * n)

    return dict(
        lon=out_lon, lat=out_lat, timestamp=out_t, category=out_cat,
        heading=out_head, length=out_len, width=out_wid, height=out_hgt,
        track_id=out_track, speed=out_speed,
    )


# ── LIDAR sweeps ─────────────────────────────────────────────────────────────
def make_lidar(rng: np.random.Generator, ego, target_points=160_000):
    """A rotating-ring LIDAR sweep per SCENE_HZ tick, ground plane + facades.

    Each sweep is sampled from the ego's current pose: rings of azimuth×range
    intersected with a flat ground plane, plus a few vertical facade walls
    bordering the road. Decimated to ~target_points total. Returns flat
    ``(lon, lat, timestamp, z, intensity)`` arrays.
    """
    ego_t, ego_x, ego_y, ego_h, _ = ego
    n_sweeps = len(ego_t)
    pts_per_sweep = max(1, target_points // n_sweeps)

    # 32-beam-ish ring elevations (deg) → ground-plane range from sensor height.
    sensor_h = 1.8  # LIDAR_TOP height above ground (m)
    elevations = np.radians(np.linspace(-24.0, 2.0, 32))
    # Facade lines: two parallel walls ±road_half offset along the lane, 3 m tall.
    road_half = 14.0
    facade_h = 3.0

    out_x, out_y, out_z, out_t, out_i = [], [], [], [], []

    for s in range(n_sweeps):
        ex, ey, eh = float(ego_x[s]), float(ego_y[s]), float(ego_h[s])
        # Split the budget: ~70% ground returns, ~30% facade returns.
        n_ground = int(pts_per_sweep * 0.7)
        n_facade = pts_per_sweep - n_ground

        # --- Ground returns: ring × azimuth, downward beams hit the plane. ---
        az = rng.uniform(0, 2 * math.pi, size=n_ground)
        el = elevations[rng.integers(0, len(elevations), size=n_ground)]
        el = np.minimum(el, math.radians(-0.5))  # only downward beams reach ground
        rng_dist = sensor_h / np.maximum(-np.sin(el), 1e-3)  # range to ground hit
        rng_dist = np.clip(rng_dist, 1.5, 60.0)  # max LIDAR range
        gx = ex + rng_dist * np.cos(az + eh)
        gy = ey + rng_dist * np.sin(az + eh)
        gz = rng.normal(0.0, 0.04, size=n_ground)  # ground ≈ z 0 ± noise
        gi = np.clip(rng.normal(90, 30, size=n_ground), 0, 255)  # asphalt intensity

        # --- Facade returns: points up the two road-side walls. ---
        side = rng.choice([-1.0, 1.0], size=n_facade)
        along = rng.uniform(-30.0, 50.0, size=n_facade)
        zf = rng.uniform(0.0, facade_h, size=n_facade)
        # Walls run parallel to ego heading, offset ±road_half laterally.
        lat_off = side * road_half
        fx = ex + along * math.cos(eh) - lat_off * math.sin(eh)
        fy = ey + along * math.sin(eh) + lat_off * math.cos(eh)
        fi = np.clip(rng.normal(160, 40, size=n_facade), 0, 255)  # brighter facade

        sx = np.concatenate([gx, fx])
        sy = np.concatenate([gy, fy])
        sz = np.concatenate([gz, zf])
        si = np.concatenate([gi, fi])

        out_x.append(sx)
        out_y.append(sy)
        out_z.append(sz)
        out_i.append(si)
        out_t.append(np.full(len(sx), int(ego_t[s]), dtype="int64"))

    X = np.concatenate(out_x)
    Y = np.concatenate(out_y)
    Z = np.concatenate(out_z)
    I = np.concatenate(out_i)
    T = np.concatenate(out_t)
    lon, lat = avc.local_to_lonlat(X, Y, ORIGIN_LAT, ORIGIN_LON)
    return lon, lat, T, Z, I


# ── Telemetry ────────────────────────────────────────────────────────────────
def make_telemetry(ego):
    """Sinusoidal-but-plausible CAN gauges at TELEMETRY_HZ derived from the ego."""
    ego_t, _, _, ego_h, ego_speed = ego
    n = int(DURATION_S * TELEMETRY_HZ) + 1
    t_s = np.linspace(0.0, DURATION_S, n)
    t_ms = (T0_MS + t_s * 1000.0).astype("int64")

    # Resample ego speed/heading-rate onto the 50 Hz grid.
    ego_ts = (ego_t - ego_t[0]) / 1000.0
    speed = np.interp(t_s, ego_ts, ego_speed)
    head = np.interp(t_s, ego_ts, ego_h)
    accel = np.gradient(speed, t_s)
    steer = np.gradient(head, t_s)  # rad/s proxy → steering signal
    # Throttle/brake split out of the (signed) accel.
    throttle = np.clip(accel / 2.5, 0.0, 1.0)
    brake = np.clip(-accel / 2.5, 0.0, 1.0)

    def pack(arr):
        return [[int(t), float(v)] for t, v in zip(t_ms, arr)]

    return dict(
        t0=int(t_ms[0]),
        hz=TELEMETRY_HZ,
        fields={
            "speed": {"unit": "m/s", "label": "Speed", "samples": pack(speed)},
            "steer": {"unit": "rad", "label": "Steering", "samples": pack(steer)},
            "accel": {"unit": "m/s²", "label": "Accel", "samples": pack(accel)},
            "throttle": {"unit": "frac", "label": "Throttle", "samples": pack(throttle)},
            "brake": {"unit": "frac", "label": "Brake", "samples": pack(brake)},
        },
    )


# ── Fabricated HD map ────────────────────────────────────────────────────────
def make_map():
    """A small FABRICATED HD map in the boston-seaport local frame (no download).

    Demonstrates the static map substrate (av-cockpit.md §2 R2.2) with NO
    external map data: a ``drivable_area`` polygon covering the road corridor the
    ego drives, ``lane_divider`` + ``road_divider`` lines along the lanes, and a
    ``ped_crossing`` polygon at the intersection. All geometry is built in local
    metres about (SCENE_CX, SCENE_CY) and projected with ``local_to_lonlat``
    (mercator=False — same true-ground-metre frame as ego/objects/lidar, so the
    map registers exactly under them).

    Returns ``(polys, lines)`` where ``polys`` / ``lines`` are lists of
    ``(shapely geometry in lon/lat, map_layer:str)`` ready for the map writers.
    """
    from shapely.geometry import LineString, Polygon

    def to_lonlat_ring(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lon, lat = avc.local_to_lonlat(xs, ys, ORIGIN_LAT, ORIGIN_LON)
        return list(zip(lon.tolist(), lat.tolist()))

    # The ego drives roughly +x (east) through SCENE_CX,SCENE_CY. Lay an E–W road
    # corridor (the main drivable area) and a short N–S cross street so there's a
    # real intersection to put a crosswalk at.
    road_half = 14.0  # matches the lidar facade walls (road is ~28 m wide)
    ew_lo, ew_hi = SCENE_CX - 120.0, SCENE_CX + 120.0  # E–W extent
    ns_lo, ns_hi = SCENE_CY - 90.0, SCENE_CY + 90.0  # N–S extent

    polys: list = []
    lines: list = []

    # --- drivable_area: the E–W corridor + N–S corridor unioned into one poly. ---
    from shapely.ops import unary_union

    ew_road = Polygon(
        [
            (ew_lo, SCENE_CY - road_half),
            (ew_hi, SCENE_CY - road_half),
            (ew_hi, SCENE_CY + road_half),
            (ew_lo, SCENE_CY + road_half),
        ]
    )
    ns_road = Polygon(
        [
            (SCENE_CX - road_half, ns_lo),
            (SCENE_CX + road_half, ns_lo),
            (SCENE_CX + road_half, ns_hi),
            (SCENE_CX - road_half, ns_hi),
        ]
    )
    drivable = unary_union([ew_road, ns_road])
    drivable_lonlat = Polygon(to_lonlat_ring(list(drivable.exterior.coords)))
    polys.append((drivable_lonlat, "drivable_area"))

    # --- ped_crossing: a striped box across the E–W road at the intersection. ---
    cw_half_w = road_half  # spans the road width
    cw_x0 = SCENE_CX + road_half + 2.0  # just east of the cross street
    cw_x1 = cw_x0 + 4.0  # 4 m deep crossing
    crossing = Polygon(
        [
            (cw_x0, SCENE_CY - cw_half_w),
            (cw_x1, SCENE_CY - cw_half_w),
            (cw_x1, SCENE_CY + cw_half_w),
            (cw_x0, SCENE_CY + cw_half_w),
        ]
    )
    polys.append((Polygon(to_lonlat_ring(list(crossing.exterior.coords))), "ped_crossing"))

    # --- road_divider: the double-yellow centre line down the E–W road. ---
    centre = LineString([(ew_lo, SCENE_CY), (ew_hi, SCENE_CY)])
    lines.append((LineString(to_lonlat_ring(list(centre.coords))), "road_divider"))

    # --- lane_divider: dashed lane lines either side of centre (±7 m → 2 lanes
    #     per direction on a 28 m road). Kept as straight lines along the road. ---
    for off in (-7.0, 7.0):
        lane = LineString([(ew_lo, SCENE_CY + off), (ew_hi, SCENE_CY + off)])
        lines.append((LineString(to_lonlat_ring(list(lane.coords))), "lane_divider"))

    return polys, lines


def make_camera_frames():
    """~20 placeholder camera-frame entries (no images committed)."""
    n = int(DURATION_S * CAMERA_HZ) + 1
    t_ms = (T0_MS + np.linspace(0.0, DURATION_S, n) * 1000.0).astype("int64")
    # url left as a conventional path; no image is committed, so the cockpit's
    # CameraInset hides itself when the <img> fails to load. Emitting valid
    # entries keeps the sidecar shape exercised end-to-end.
    return [{"t": int(t), "url": f"cam/{i:04d}.jpg"} for i, t in enumerate(t_ms)]


# ── Orchestration ────────────────────────────────────────────────────────────
def generate(
    out_dir: Path,
    stt_build: str,
    skip_build: bool,
    seed: int = 7,
    point_min_zoom: int = 0,
    quantize_coords: float | None = None,
):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesising {SCENE_ID} ({DURATION_S:.0f}s @ {LOCATION}) …")
    ego = make_ego(rng)
    ego_t, ego_x, ego_y, ego_h, ego_speed = ego
    t_start, t_end = int(ego_t[0]), int(ego_t[-1])

    # --- ego (trips) parquet ---
    ego_lon, ego_lat = avc.local_to_lonlat(ego_x, ego_y, ORIGIN_LAT, ORIGIN_LON)
    ego_pq = out_dir / "ego.parquet"
    avc.write_ego_trips(
        ego_pq, lon=ego_lon, lat=ego_lat,
        vertex_timestamps=ego_t.tolist(), vertex_values=ego_speed.tolist(),
        vehicle="ego",
    )

    # --- objects (points) parquet ---
    objs = make_objects(rng, ego)
    obj_pq = out_dir / "objects.parquet"
    n_objects = avc.write_objects_points(obj_pq, **objs)
    obj_categories = sorted(set(objs["category"]))

    # --- lidar (points) parquet ---
    lid_lon, lid_lat, lid_t, lid_z, lid_i = make_lidar(rng, ego)
    lid_pq = out_dir / "lidar.parquet"
    n_lidar = avc.write_lidar_points(
        lid_pq, lon=lid_lon, lat=lid_lat, timestamp=lid_t, z=lid_z, intensity=lid_i,
    )

    # --- HD map (fabricated, static full-range) parquet ---
    map_polys, map_lines = make_map()
    map_poly_pq = out_dir / "map_poly.parquet"
    map_line_pq = out_dir / "map_line.parquet"
    n_map_poly = avc.write_map_polygons(map_poly_pq, map_polys, t_start, t_end)
    n_map_line = avc.write_map_lines(map_line_pq, map_lines, t_start, t_end)
    # Present layer names, deduped + ordered (poly layers first, then line layers).
    map_layers: list[str] = []
    for _, layer in map_polys + map_lines:
        if layer not in map_layers:
            map_layers.append(layer)

    # --- telemetry + cameras sidecars (live next to scene.json) ---
    tel = make_telemetry(ego)
    avc.write_telemetry_json(
        out_dir / "telemetry.json", t0=tel["t0"], hz=tel["hz"], fields=tel["fields"],
    )
    avc.write_cameras_json(
        out_dir / "cameras.json", camera="CAM_FRONT", frames=make_camera_frames(),
    )

    # --- initial camera view centred on the scene ---
    cx_lon, cx_lat = avc.local_to_lonlat(SCENE_CX, SCENE_CY, ORIGIN_LAT, ORIGIN_LON)
    initial_view = {
        "longitude": float(cx_lon), "latitude": float(cx_lat),
        "zoom": 18, "pitch": 55, "bearing": 20,
    }

    # --- build the three packed archives (unless --skip-build) ---
    if not skip_build:
        # LIDAR sweeps step at 10 Hz → 100ms buckets; objects likewise. The two
        # dense POINT clouds carry the opt-in size levers (min_zoom clamp +
        # coordinate quantization); ego (one sparse trip) keeps the defaults so
        # its path precision is untouched.
        avc.run_stt_build(lid_pq, out_dir / "lidar", "point",
                          stt_build=stt_build, min_zoom=point_min_zoom, max_zoom=18,
                          temporal_bucket="100ms", quantize_coords=quantize_coords)
        avc.run_stt_build(obj_pq, out_dir / "objects", "point",
                          stt_build=stt_build, min_zoom=point_min_zoom, max_zoom=18,
                          temporal_bucket="100ms", quantize_coords=quantize_coords)
        avc.run_stt_build(ego_pq, out_dir / "ego", "trips",
                          stt_build=stt_build, max_zoom=18, temporal_bucket="100ms")
        # Static HD-map archives — one bucket spanning the whole replay (the
        # map_temporal_bucket default >> the 20s scene), so they load once.
        avc.run_stt_build(map_poly_pq, out_dir / "map_poly", "map_poly",
                          stt_build=stt_build, max_zoom=18)
        avc.run_stt_build(map_line_pq, out_dir / "map_line", "map_line",
                          stt_build=stt_build, max_zoom=18)

    # --- lightweight ego polyline for the ego-follow camera (spec/sidecar-assets.md §3.1).
    # Same lon/lat/t samples as the ego trips archive above → the follow path
    # tracks the rendered ego trail exactly. ~40–80 pts; pure JSON, no rebuild.
    ego_path = avc.downsample_ego_path(ego_t, ego_lon, ego_lat)

    # --- scene.json manifest (streams reference the packed manifests) ---
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
        "camera": {"url": "cameras.json"},
    }
    avc.write_scene_json(
        out_dir / "scene.json",
        id=SCENE_ID,
        name="Synthetic · Boston Seaport Drive",
        dataset="Synthetic (poopdeck.gl)",
        dataset_url="https://github.com/",
        license="CC0-1.0",
        location=LOCATION,
        description="Procedural 20s urban drive: rotating LIDAR ring, CAN telemetry, "
                    "12 tracked agents, ego trajectory. No external data.",
        origin_lat=ORIGIN_LAT,
        origin_lon=ORIGIN_LON,
        time_range=(t_start, t_end),
        initial_view=initial_view,
        object_colors=avc.OBJECT_COLORS,
        streams=streams,
    )

    print(
        f"\nDone: {SCENE_ID} → {out_dir}\n"
        f"  ego vertices : {len(ego_t)} ({len(ego_path)} pts in scene.json ego.path)\n"
        f"  objects      : {n_objects} points ({len(obj_categories)} categories)\n"
        f"  lidar        : {n_lidar} points\n"
        f"  map          : {n_map_poly} polygon(s) + {n_map_line} line(s) "
        f"[{', '.join(map_layers)}]\n"
        f"  telemetry    : {len(tel['fields'])} fields @ {tel['hz']:.0f} Hz\n"
    )
    if skip_build:
        # Mirror the opt-in point-cloud levers in the printed hint so a manual
        # build matches what `generate` would have run.
        pt_flags = ""
        if point_min_zoom:
            pt_flags += f"--min-zoom {point_min_zoom} "
        if quantize_coords:
            pt_flags += f"--quantize-coords {float(quantize_coords)!r} "
        print(
            "  (--skip-build) GeoParquet + JSON only. Build the archives with:\n"
            f"    {stt_build} --input {lid_pq} --output {out_dir/'lidar'} "
            "--time-field timestamp --time-format unix-ms --temporal-bucket 100ms "
            f"{pt_flags}--compression zstd --publish\n"
            f"    {stt_build} --input {obj_pq} --output {out_dir/'objects'} "
            "--time-field timestamp --time-format unix-ms --temporal-bucket 100ms "
            f"{pt_flags}--compression zstd --publish\n"
            f"    {stt_build} --input {ego_pq} --output {out_dir/'ego'} "
            "--time-field timestamp --end-time-field end_timestamp --time-format unix-ms "
            "--temporal-bucket 100ms --compression zstd --simplify --publish\n"
            f"    {stt_build} --input {map_poly_pq} --output {out_dir/'map_poly'} "
            "--time-field timestamp --end-time-field end_timestamp --time-format unix-ms "
            "--temporal-bucket 1d --compression zstd --publish\n"
            f"    {stt_build} --input {map_line_pq} --output {out_dir/'map_line'} "
            "--time-field timestamp --end-time-field end_timestamp --time-format unix-ms "
            "--temporal-bucket 1d --no-clip --compression zstd --publish\n"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output scene-bundle directory (default {DEFAULT_OUT})")
    p.add_argument("--stt-build", default="stt-build",
                   help="path to the stt-build binary (default: stt-build on PATH)")
    p.add_argument("--skip-build", action="store_true",
                   help="stop at GeoParquet + sidecar JSON (don't run stt-build)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (deterministic)")
    p.add_argument("--point-min-zoom", type=int, default=0,
                   help="OPT-IN: clamp the bottom zoom for the LIDAR/objects point "
                        "clouds (default 0). ~14 avoids the useless ultra-dense "
                        "z0 tile + pack bloat; measure before committing.")
    p.add_argument("--quantize-coords", type=float, default=None, metavar="METERS",
                   help="OPT-IN: quantize LIDAR/objects geometry to this ground "
                        "precision in meters (the largest size lever). For "
                        "meter-scale AV scenes use <= 0.1 (~7.5 cm); off by default.")
    args = p.parse_args()
    generate(args.out, args.stt_build, args.skip_build, seed=args.seed,
             point_min_zoom=args.point_min_zoom, quantize_coords=args.quantize_coords)


if __name__ == "__main__":
    main()
