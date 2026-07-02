#!/usr/bin/env python3
"""Extract one comma2k19 driving segment into an AV cockpit scene bundle.

comma.ai's comma2k19 dataset is real highway driving from a Honda Civic / Toyota
RAV4 running openpilot. Unlike nuScenes/Argoverse it carries **GPS lat/lon
directly** (well, ECEF positions we convert to lat/lon) plus a rich CAN bus, but
**no LIDAR and no tracked objects** — so this adapter emits only the ``ego`` +
``telemetry`` (+ optional ``camera``) streams. The cockpit renders only the
streams present in scene.json (av-cockpit.md §2), so that is a valid bundle.

────────────────────────────────────────────────────────────────────────────
DOWNLOAD (free; comma.ai data is permissively licensed) — TWO SOURCES
────────────────────────────────────────────────────────────────────────────
The full comma2k19 is only published as monolithic ~10 GB ``Chunk_*.zip`` files
(github / huggingface ``raw_data/``). DO NOT pull a 10 GB chunk for one segment.

(A) PREFERRED — HuggingFace ``demo`` parquet (file-level, ~73-81 MB each):
    The HF mirror ``commaai/comma2k19`` exposes ``data/demo-*-of-00003.parquet``
    — three small parquet files, EACH carrying ~21 fully-decoded 1-min segments
    (ECEF poses + frame_times + CAN speed/steering + IMU accel + a JPEG
    ``preview`` image), with NO openpilot-tools and NO 10 GB download::

        pip install huggingface_hub
        huggingface-cli download commaai/comma2k19 \\
            data/demo-00001-of-00003.parquet \\
            --repo-type dataset --local-dir comma-raw

    Then point this script at the parquet + a row index (0-based)::

        python comma_extract.py \\
            --demo-parquet comma-raw/data/demo-00001-of-00003.parquet \\
            --row 6 --out ../../examples/showcase/public/data/comma-<short>

    The demo-parquet ``log`` struct flattens the on-disk tree with ``__`` joins
    (``global_pose__frame_positions``, ``processed_log__CAN__speed__value`` …).
    The ``preview`` column is a single road-camera JPEG used for the inset (so
    ``--extract-frames`` is unnecessary in this mode).

(B) On-disk segment (full chunk download). A *segment* directory looks like::

         <dongle_id>|<route>/<segment_idx>/
           global_pose/
             frame_positions      # (N,3) ECEF metres
             frame_velocities     # (N,3) ECEF m/s
             frame_times          # (N,)  boot time (seconds)
             frame_gps_times      # (N,2) [gps_week, time_of_week_seconds]
           processed_log/
             CAN/speed/{value,t}            # m/s, boot-time t
             CAN/steering_angle/{value,t}   # degrees (vehicle-dependent), boot-time t
             IMU/accelerometer/{value,t}    # (M,3) m/s², boot-time t
           video.hevc                       # road camera (needs ffmpeg to decode)

    pip install numpy pyarrow shapely pyproj   (pyproj for ECEF→geodetic)

         python comma_extract.py --segment '<dongle>|<route>/0' \\
             --out ../../examples/showcase/public/data/comma-<route>

Use ``--skip-build`` to stop at GeoParquet + JSON. In on-disk mode camera frames
are emitted as ENTRIES only (no images committed); pass ``--extract-frames`` to
also dump JPEGs via ffmpeg (documented below). In demo-parquet mode the single
``preview`` JPEG is written automatically.

────────────────────────────────────────────────────────────────────────────
COORDINATE / UNIT NOTES (verified against the comma2k19 README + paper)
────────────────────────────────────────────────────────────────────────────
* ``global_pose/frame_positions`` is **ECEF metres** — convert to geodetic
  lat/lon with pyproj (``EPSG:4978`` ECEF → ``EPSG:4326`` lon/lat). comma ships
  no direct lat/lon array.
* Times in ``global_pose`` and ``processed_log`` are **boot-relative seconds**.
  We anchor t=0 of the first GPS frame to a fixed Unix epoch so the cockpit's
  scrubber has wall-clock-shaped Unix-ms (the absolute date is synthetic; the
  RELATIVE timing is real). Use ``--epoch-ms`` to override the anchor.
* CAN ``speed`` is **m/s** already; ``steering_angle`` is **degrees** (converted
  to radians); accelerometer is **m/s²** (longitudinal component used).
* No georef origin is needed — GPS gives true lon/lat; ``georef`` in scene.json
  is set to the first GPS fix for reference.

Pipeline:  comma2k19  →  GeoParquet + JSON  →  stt-build  →  packed bundle.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

import av_common as avc

# Decimate the ~20 Hz pose stream a touch so the ego LineString isn't enormous
# on a long segment (every Nth frame). 1 = keep all.
EGO_DECIMATE = 1


def _ecef_to_lonlat(ecef: np.ndarray):
    """ECEF (N,3) metres → (lon, lat) degrees via pyproj WGS84."""
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon, lat, _alt = tf.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return np.asarray(lon), np.asarray(lat)


def _bearing_from_velocity(lon, lat, vel, t_ms):
    """Per-frame compass bearing (deg, 0=N, CW+) from ECEF velocity → ENU.

    ``vel`` is the ``global_pose/frame_velocities`` (N,3) ECEF m/s array (same
    decimation + length as ``lon``/``lat``/``t_ms``); rotate each ECEF velocity
    into the local East-North-Up frame at its lon/lat and take ``atan2(E, N)``.
    Returns ``[[t_ms, bearing_deg], …]``, or ``None`` when no velocities exist.
    Samples whose speed is below ~0.5 m/s are dropped (bearing is meaningless at
    a standstill). This is the same heading the cockpit derives client-side, so
    it's optional polish — but it gives the strip charts a real heading channel.
    """
    if vel is None or not len(vel):
        return None
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    vel = np.asarray(vel, dtype="float64")
    n = min(len(lon), len(lat), len(vel), len(t_ms))
    if n < 1:
        return None
    lon, lat, vel, t_ms = lon[:n], lat[:n], vel[:n], np.asarray(t_ms)[:n]
    lam = np.radians(lon)
    phi = np.radians(lat)
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
    east = -np.sin(lam) * vx + np.cos(lam) * vy
    north = (-np.sin(phi) * np.cos(lam) * vx
             - np.sin(phi) * np.sin(lam) * vy
             + np.cos(phi) * vz)
    bearing = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    moving = np.hypot(east, north) >= 0.5  # drop near-standstill noise
    return [[int(t), float(b)] for t, b, m in zip(t_ms, bearing, moving) if m]


def _load(seg: Path, *rel: str) -> np.ndarray:
    """Load a comma2k19 numpy array (files are bare, no .npy suffix)."""
    path = seg.joinpath(*rel)
    if not path.exists() and path.with_suffix(".npy").exists():
        path = path.with_suffix(".npy")
    return np.load(path, allow_pickle=False)


def _to_unix_ms(boot_t: np.ndarray, t0_boot: float, epoch_ms: int) -> np.ndarray:
    """Boot-relative seconds → absolute Unix-ms anchored at ``epoch_ms``."""
    return (epoch_ms + (np.asarray(boot_t, dtype="float64") - t0_boot) * 1000.0).astype("int64")


def extract(seg: Path, epoch_ms: int):
    """Return (ego dict, telemetry dict, t0_boot, first_lonlat)."""
    pos = _load(seg, "global_pose", "frame_positions")  # (N,3) ECEF metres
    ftimes = _load(seg, "global_pose", "frame_times")  # (N,) boot seconds
    vel = None
    try:
        vel = _load(seg, "global_pose", "frame_velocities")  # (N,3) ECEF m/s
    except FileNotFoundError:
        print("  warning: global_pose/frame_velocities missing — ego speed falls "
              "back to finite difference and the heading channel is omitted")

    pos = pos[::EGO_DECIMATE]
    ftimes = ftimes[::EGO_DECIMATE]
    lon, lat = _ecef_to_lonlat(pos)
    t0_boot = float(ftimes[0])
    ego_t = _to_unix_ms(ftimes, t0_boot, epoch_ms)

    # ego speed: magnitude of ECEF velocity if present, else finite difference.
    if vel is not None:
        vel = vel[::EGO_DECIMATE]
        ego_speed = np.linalg.norm(vel, axis=1)
    else:
        # Finite difference on the great-ish-circle ground distance.
        dt = np.gradient(ego_t / 1000.0)
        dx = np.gradient(lon) * avc.M_PER_DEG_LAT * np.cos(np.radians(lat))
        dy = np.gradient(lat) * avc.M_PER_DEG_LAT
        ego_speed = np.hypot(dx, dy) / np.maximum(dt, 1e-3)

    ego = dict(lon=lon, lat=lat, t=ego_t, speed=ego_speed)

    # --- CAN telemetry ---
    fields = {}

    # See ``extract_demo_row`` for the full channel rationale: each *column* of a
    # vector signal is its own physical channel, so we emit one field per column
    # (the old ``v[:, 0]`` slice silently dropped wheels 1.. and accel axes 1..).
    # brake / gas / steering-torque are raw-CAN/DBC only and are NOT emitted.
    def _raw(category, *rel):
        """Return ``(t_ms, value_array)`` keeping the native (N,) / (N,k) shape."""
        try:
            v = _load(seg, *rel, f"{category}", "value").astype("float64")
            t = _load(seg, *rel, f"{category}", "t").astype("float64")
        except FileNotFoundError:
            return None
        n = min(len(v), len(t))
        return _to_unix_ms(t[:n], t0_boot, epoch_ms), v[:n]

    def _mk(t_ms, col, *, scale=1.0, transform=None):
        vals = transform(col) if transform else col * scale
        return [[int(a), float(b)] for a, b in zip(t_ms, vals)]

    def _add_scalar(category, rel, fid, *, unit, label, scale=1.0, transform=None):
        raw = _raw(category, *rel)
        if raw is None:
            return
        t_ms, v = raw
        if v.ndim > 1:
            v = v[:, 0]
        s = _mk(t_ms, v, scale=scale, transform=transform)
        if s:
            fields[fid] = {"unit": unit, "label": label, "samples": s}

    def _add_columns(category, rel, specs, *, unit):
        raw = _raw(category, *rel)
        if raw is None:
            return
        t_ms, v = raw
        if v.ndim == 1:
            v = v[:, None]
        ncols = v.shape[1]
        for col_idx, fid, label in specs:
            if col_idx >= ncols:
                continue
            s = _mk(t_ms, v[:, col_idx])
            if s:
                fields[fid] = {"unit": unit, "label": label, "samples": s}

    _add_scalar("speed", ("processed_log", "CAN"), "speed", unit="m/s", label="Speed")
    _add_scalar("steering_angle", ("processed_log", "CAN"), "steer",
                unit="rad", label="Steering", transform=lambda v: np.radians(v))
    # On-disk comma2k19 names the per-wheel speeds ``wheel_speeds`` (plural).
    _add_columns("wheel_speeds", ("processed_log", "CAN"),
                 [(0, "wheel_fl", "Wheel FL"), (1, "wheel_fr", "Wheel FR"),
                  (2, "wheel_rl", "Wheel RL"), (3, "wheel_rr", "Wheel RR")],
                 unit="m/s")
    _add_columns("accelerometer", ("processed_log", "IMU"),
                 [(0, "accel_long", "Accel long"), (1, "accel_lat", "Accel lat"),
                  (2, "accel_vert", "Accel vert")],
                 unit="m/s²")
    _add_columns("gyro", ("processed_log", "IMU"),
                 [(2, "yaw_rate", "Yaw rate")], unit="rad/s")

    bearing_samples = _bearing_from_velocity(lon, lat, vel, ego_t)
    if bearing_samples:
        fields["bearing"] = {"unit": "deg", "label": "Bearing", "samples": bearing_samples}

    telemetry = None
    if fields:
        t0 = min(s[0] for f in fields.values() for s in f["samples"])
        telemetry = {"t0": t0, "hz": 100.0, "fields": fields}

    return ego, telemetry, t0_boot, (float(lon[0]), float(lat[0]))


def _flatten_can_value(value) -> np.ndarray:
    """Flatten a comma2k19 demo-parquet CAN/IMU ``value`` column to (N,) or (N,k).

    The HF demo parquet stores some signals nested one level deeper than the
    on-disk ``.npy`` form: ``speed.value`` is ``[[v],[v],…]`` (N,1),
    ``accelerometer.value`` is ``[[x,y,z],…]`` (N,3), while ``steering_angle``
    is a flat ``[v,…]`` (N,). We coerce to a numpy array and, for a trailing
    singleton dimension, squeeze it so scalar signals come back 1-D.
    """
    arr = np.asarray(value, dtype="float64")
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def extract_demo_row(parquet_path: Path, row: int, epoch_ms: int):
    """Extract one segment from a HuggingFace comma2k19 ``demo`` parquet row.

    Each row of ``data/demo-*-of-00003.parquet`` is a fully-decoded 1-min
    segment whose ``log`` struct flattens the on-disk tree with ``__`` joins.
    Returns ``(ego, telemetry, t0_boot, (lon0, lat0), segment_id, preview_bytes)``
    where ``preview_bytes`` is a road-camera JPEG (or ``None``).
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    n_rows = table.num_rows
    if not (0 <= row < n_rows):
        raise IndexError(
            f"--row {row} out of range (parquet has {n_rows} segments: 0..{n_rows - 1})"
        )
    log = table.column("log")[row].as_py()
    seg_ids = table.column("segment_id").to_pylist()
    segment_id = seg_ids[row]

    pos = np.asarray(log["global_pose__frame_positions"], dtype="float64")  # (N,3) ECEF
    ftimes = np.asarray(log["global_pose__frame_times"], dtype="float64")  # (N,) boot s
    vel = log.get("global_pose__frame_velocities")
    vel = np.asarray(vel, dtype="float64") if vel is not None and len(vel) else None

    pos = pos[::EGO_DECIMATE]
    ftimes = ftimes[::EGO_DECIMATE]
    lon, lat = _ecef_to_lonlat(pos)
    t0_boot = float(ftimes[0])
    ego_t = _to_unix_ms(ftimes, t0_boot, epoch_ms)

    if vel is not None:
        vel = vel[::EGO_DECIMATE]
        ego_speed = np.linalg.norm(vel, axis=1)
    else:
        dt = np.gradient(ego_t / 1000.0)
        dx = np.gradient(lon) * avc.M_PER_DEG_LAT * np.cos(np.radians(lat))
        dy = np.gradient(lat) * avc.M_PER_DEG_LAT
        ego_speed = np.hypot(dx, dy) / np.maximum(dt, 1e-3)

    ego = dict(lon=lon, lat=lat, t=ego_t, speed=ego_speed)

    # --- CAN / IMU telemetry from the flattened struct ---
    #
    # comma2k19 ``processed_log`` exposes scalar AND vector signals: ``speed`` is
    # (N,1)→scalar, ``steering_angle`` (N,), ``wheel_speed`` (N,4) per-wheel,
    # ``accelerometer``/``gyro`` (N,3) per-axis. Each *column* of a vector signal
    # is its own physical channel, so we emit ONE telemetry.json field per column
    # rather than dropping columns 1.. (the old ``v[:, 0]`` slice silently kept
    # only the first wheel / the longitudinal accel axis and threw the rest away).
    #
    # IMU axis convention (verified on this parquet — gravity ≈ −9.6 m/s² sits on
    # axis 2, so axis 2 is vertical/up and the yaw-rate is the axis-2 gyro):
    #   accelerometer = [longitudinal, lateral, vertical]  (m/s²)
    #   gyro          = [roll, pitch, yaw] rates           (rad/s)
    # wheel_speed columns are the comma2k19 order [FL, FR, RL, RR] (m/s).
    #
    # NOTE: brake / gas / steering-torque are NOT in ``processed_log`` — they live
    # only in the raw CAN frames (``raw_can``) and need the vehicle DBC to decode,
    # so we do not (and must not) synthesise them here.
    fields = {}

    def _demo_raw(key):
        """Return ``(t_ms, value_array)`` for a flattened signal, or ``None``.

        ``value_array`` keeps its native shape — (N,) for scalar signals, (N,k)
        for vector signals — so the caller can emit one field per column. A
        trailing singleton dimension (e.g. ``speed`` is (N,1)) is squeezed to
        (N,) by ``_flatten_can_value``.
        """
        vkey, tkey = f"{key}__value", f"{key}__t"
        if vkey not in log or tkey not in log or not len(log[vkey]):
            return None
        v = _flatten_can_value(log[vkey])
        t = np.asarray(log[tkey], dtype="float64")
        n = min(len(v), len(t))
        v, t = v[:n], t[:n]
        return _to_unix_ms(t, t0_boot, epoch_ms), v

    def _samples(t_ms, col, *, scale=1.0, transform=None):
        """Zip a (N,) value column with its (N,) times into ``[[t,v],…]``."""
        vals = transform(col) if transform else col * scale
        return [[int(a), float(b)] for a, b in zip(t_ms, vals)]

    def _add_scalar(key, fid, *, unit, label, scale=1.0, transform=None):
        """Register a scalar signal (or column-0 of a 1-wide value) as one field."""
        raw = _demo_raw(key)
        if raw is None:
            return
        t_ms, v = raw
        if v.ndim > 1:
            v = v[:, 0]
        s = _samples(t_ms, v, scale=scale, transform=transform)
        if s:
            fields[fid] = {"unit": unit, "label": label, "samples": s}

    def _add_columns(key, specs, *, unit):
        """Register one field per column of a vector signal.

        ``specs`` is a list of ``(col_index, field_id, label)``; columns absent
        from the actual value width are skipped (so a (N,3) accel won't try to
        read a 4th axis).
        """
        raw = _demo_raw(key)
        if raw is None:
            return
        t_ms, v = raw
        if v.ndim == 1:  # a single-column vector signal collapsed to (N,)
            v = v[:, None]
        ncols = v.shape[1]
        for col_idx, fid, label in specs:
            if col_idx >= ncols:
                continue
            s = _samples(t_ms, v[:, col_idx])
            if s:
                fields[fid] = {"unit": unit, "label": label, "samples": s}

    # Speed (CAN, m/s) — (N,1) scalar.
    _add_scalar("processed_log__CAN__speed", "speed", unit="m/s", label="Speed")
    # Steering wheel angle (CAN, deg → rad).
    _add_scalar("processed_log__CAN__steering_angle", "steer",
                unit="rad", label="Steering", transform=lambda v: np.radians(v))
    # Per-wheel speeds (CAN, m/s) — (N,4) [FL, FR, RL, RR].
    _add_columns(
        "processed_log__CAN__wheel_speed",
        [(0, "wheel_fl", "Wheel FL"), (1, "wheel_fr", "Wheel FR"),
         (2, "wheel_rl", "Wheel RL"), (3, "wheel_rr", "Wheel RR")],
        unit="m/s",
    )
    # IMU accelerometer (m/s²) — (N,3) [longitudinal, lateral, vertical].
    _add_columns(
        "processed_log__IMU__accelerometer",
        [(0, "accel_long", "Accel long"), (1, "accel_lat", "Accel lat"),
         (2, "accel_vert", "Accel vert")],
        unit="m/s²",
    )
    # IMU gyro yaw-rate (rad/s) — axis 2 is the vertical/yaw axis (see above).
    _add_columns(
        "processed_log__IMU__gyro",
        [(2, "yaw_rate", "Yaw rate")],
        unit="rad/s",
    )

    # GNSS bearing (deg). Prefer the dense per-frame velocity heading (ECEF→ENU
    # bearing) over the sparse GNSS fix bearing — it's per-pose, clean, and tied
    # directly to motion. Falls back to nothing if velocities are absent.
    bearing_samples = _bearing_from_velocity(lon, lat, vel, ego_t)
    if bearing_samples:
        fields["bearing"] = {"unit": "deg", "label": "Bearing", "samples": bearing_samples}

    telemetry = None
    if fields:
        t0 = min(s[0] for f in fields.values() for s in f["samples"])
        telemetry = {"t0": t0, "hz": 100.0, "fields": fields}

    preview = table.column("preview")[row].as_py() if "preview" in table.column_names else None
    preview_bytes = preview.get("bytes") if isinstance(preview, dict) else None

    return ego, telemetry, t0_boot, (float(lon[0]), float(lat[0])), segment_id, preview_bytes


def write_preview_camera(out: Path, preview_bytes: bytes | None, ego_t) -> list:
    """Write the demo-parquet road-camera JPEG into ``cam/`` and return entries.

    The HF demo parquet ships ONE preview JPEG per segment (not a video), so we
    emit a single camera entry anchored at the segment start. Returns ``[]`` when
    there's no preview image (the cockpit then hides the inset).
    """
    if not preview_bytes:
        return []
    cam_dir = out / "cam"
    cam_dir.mkdir(parents=True, exist_ok=True)
    (cam_dir / "0000.jpg").write_bytes(preview_bytes)
    print(f"  wrote road-camera preview → {cam_dir / '0000.jpg'} ({len(preview_bytes)} bytes)")
    return [{"t": int(ego_t[0]), "url": "cam/0000.jpg"}]


def maybe_extract_frames(seg: Path, out: Path, ego_t, t0_boot, epoch_ms, decode=False):
    """Build road-camera frame entries, optionally decoding JPEGs via ffmpeg.

    Always returns a list of ~1 fps camera-frame entries aligned to ego times.
    When ``decode`` is set (and ffmpeg + video.hevc are present) it also dumps
    the matching JPEGs into ``<out>/cam/``; otherwise entries point at
    conventional paths and the cockpit hides the inset if the image is missing.
    """
    video = seg / "video.hevc"
    frames = []
    # ~1 fps entries spread across the segment.
    n = max(2, int((ego_t[-1] - ego_t[0]) / 1000) + 1)
    idx = np.linspace(0, len(ego_t) - 1, n).astype(int)
    for k, i in enumerate(idx):
        frames.append({"t": int(ego_t[i]), "url": f"cam/{k:04d}.jpg"})

    if decode and video.exists() and shutil.which("ffmpeg"):
        cam_dir = out / "cam"
        cam_dir.mkdir(parents=True, exist_ok=True)
        # 1 fps JPEGs; the cockpit matches by nearest timestamp.
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-vf", "fps=1",
             str(cam_dir / "%04d.jpg")],
            check=False,
        )
        print(f"  extracted road-camera frames → {cam_dir}")
    else:
        print("  emitting camera entries without images "
              "(pass --extract-frames + have ffmpeg/video.hevc to decode JPEGs).")
    return frames


def generate(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    preview_bytes = None
    if args.demo_parquet is not None:
        print(f"Extracting comma2k19 demo-parquet row {args.row} "
              f"({args.demo_parquet.name}) → {out}")
        (ego, telemetry, t0_boot, (lon0, lat0),
         segment_id, preview_bytes) = extract_demo_row(
            args.demo_parquet, args.row, args.epoch_ms)
        # segment_id is "<dongle>|<route>/<idx>"; route name for the scene title.
        route_name = segment_id.split("|", 1)[-1] if "|" in segment_id else segment_id
        seg = None
    else:
        seg = args.segment
        print(f"Extracting comma2k19 segment {seg} → {out}")
        ego, telemetry, t0_boot, (lon0, lat0) = extract(seg, args.epoch_ms)
        route_name = seg.parent.name

    ego_t = ego["t"]
    t_start, t_end = int(ego_t[0]), int(ego_t[-1])

    # --- ego (trips) ---
    ego_pq = out / "ego.parquet"
    avc.write_ego_trips(ego_pq, lon=ego["lon"], lat=ego["lat"],
                        vertex_timestamps=ego_t.tolist(),
                        vertex_values=ego["speed"].tolist(), vehicle="ego")

    streams = {"ego": {"url": "ego/manifest.json"}}

    # --- telemetry ---
    if telemetry is not None:
        avc.write_telemetry_json(out / "telemetry.json", t0=telemetry["t0"],
                                 hz=telemetry["hz"], fields=telemetry["fields"])
        streams["telemetry"] = {"url": "telemetry.json"}

    # --- cameras ---
    if args.demo_parquet is not None:
        # The demo parquet ships a single road-camera preview JPEG per segment.
        frames = write_preview_camera(out, preview_bytes, ego_t)
    else:
        # On-disk mode: emit entries (and decode JPEGs with --extract-frames).
        frames = maybe_extract_frames(seg, out, ego_t, t0_boot, args.epoch_ms,
                                      decode=args.extract_frames)
    if frames:
        avc.write_cameras_json(out / "cameras.json", camera="ROAD", frames=frames)
        streams["camera"] = {"url": "cameras.json"}

    # --- build packed ego archive (only stream needing tiles) ---
    if not args.skip_build:
        avc.run_stt_build(ego_pq, out / "ego", "trips",
                          stt_build=args.stt_build, temporal_bucket=args.temporal_bucket)

    # --- scene.json (no lidar/objects streams) ---
    avc.write_scene_json(
        out / "scene.json",
        id=out.name,
        name=f"comma.ai · {route_name}",
        dataset="comma2k19",
        dataset_url="https://github.com/commaai/comma2k19",
        license="MIT",  # comma2k19 is permissively licensed
        location="highway",
        description="comma.ai openpilot drive: GPS ego trajectory + CAN telemetry "
                    "(speed/steering/accel). No LIDAR or object boxes.",
        origin_lat=lat0,
        origin_lon=lon0,
        time_range=(t_start, t_end),
        initial_view={"longitude": float(ego["lon"][len(ego["lon"]) // 2]),
                      "latitude": float(ego["lat"][len(ego["lat"]) // 2]),
                      "zoom": 15, "pitch": 50, "bearing": 0},
        object_colors=avc.OBJECT_COLORS,
        streams=streams,
    )
    print(f"Done: {out} ({len(ego_t)} ego vertices)")
    if args.skip_build:
        print("  (--skip-build) re-run without it to build the ego archive.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--segment", type=Path,
                     help="ON-DISK MODE: path to a comma2k19 segment dir (contains "
                          "global_pose/, processed_log/, video.hevc)")
    src.add_argument("--demo-parquet", type=Path,
                     help="DEMO-PARQUET MODE: path to a HuggingFace "
                          "data/demo-*-of-00003.parquet (commaai/comma2k19); pick a "
                          "segment with --row")
    p.add_argument("--row", type=int, default=0,
                   help="segment row index within --demo-parquet (0-based)")
    p.add_argument("--out", type=Path, required=True, help="output scene-bundle dir")
    p.add_argument("--epoch-ms", type=int, default=1_700_000_000_000,
                   help="Unix-ms to anchor the (boot-relative) segment start to")
    p.add_argument("--temporal-bucket", default="1s",
                   help="stt-build temporal bucket for the ego trip")
    p.add_argument("--extract-frames", action="store_true",
                   help="also decode road-camera JPEGs via ffmpeg (default: entries only)")
    p.add_argument("--stt-build", default="stt-build", help="stt-build binary path")
    p.add_argument("--skip-build", action="store_true",
                   help="stop at GeoParquet + JSON (don't run stt-build)")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
