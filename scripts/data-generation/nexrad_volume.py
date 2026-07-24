#!/usr/bin/env python3
"""NEXRAD Level II volumetric radar gates → STT 3D point volume + couplet markers.

The volumetric core of the Storm-4D Greenfield campaign
(docs/roadmap/storm-4d-greenfield-2026-07.md §9.1): every radar gate of every
elevation cut of every KDMX volume scan becomes one 3D point (`alt_m` property
column, LiDAR pattern — NOT geometry-Z), so a supercell renders as a true 4D
object. A second, tiny archive carries auto-detected velocity-couplet
(mesocyclone) markers from azimuthal shear on the lowest velocity sweep.

Source: NEXRAD Level II on the anonymous Unidata S3 bucket (the legacy
`noaa-nexrad-level2` bucket is dead/403 since 2025-09):

    https://unidata-nexrad-level2.s3.amazonaws.com/<YYYY>/<MM>/<DD>/<SITE>/
        <SITE><YYYYMMDD>_<HHMMSS>_V06        (skip *_MDM sidecars)

Per volume file (Py-ART):
  read_nexrad_archive → dealias_region_based (nyquist from instrument
  parameters — mandatory, raw NEXRAD velocity folds at ~28 m/s) →
  per-gate lon/lat/alt via radar.get_gate_lat_lon_alt (alt = meters MSL,
  beam height + site elevation) → knobs (below) → GeoParquet → stt-build.

ALL elevation cuts are kept, including SAILS repeats — they ARE the temporal
resolution; each gate's `timestamp` is its own sweep's start time. Split-cut
tilts (CS reflectivity sweep + CD velocity sweep at the same angle) emit
reflectivity-driven gates ONCE (from the CS sweep) with `vel_ms` attached by
nearest-azimuth pairing from the co-located CD sweep; gates without velocity
coverage carry NULL vel columns (omitted per-feature downstream).

Knobs (Waves-A0-tunable; see §9.1 — dBZ floor + crop are *semantic filters*
declared in demoMeta, gate decimation is a declared reduced tier under the
2026-07-10 Waymo-class amendment; raw Level II stays the citable base):
  --dbz-floor       10   keep gates ≥ this dBZ ("no meteorological echo" below)
  --crop-km        150   keep gates within this range of Greenfield (41.305,-94.461)
  --range-decim-km 0.5   decimate range to this spacing (super-res is 0.25 km);
                         dBZ averaged in linear Z units per output cell
  --az-decim-deg   1.0   decimate azimuth to this spacing (super-res is 0.5°);
                         velocity is NOT averaged (circular) — center gate wins

Couplet detection (storm4d-couplet, §9.1 C1): on every lowest-tilt dealiased
velocity sweep (SAILS repeats included), gate-to-gate azimuthal Δv maxima in
2 km range bins; peaks with Δv ≥ 30 m/s are greedily clustered within 5 km and
each cluster emits one marker: timestamp, strength_ms (peak Δv), alt_m.
Measured refinements on top of the §9.1 letter (the bare Δv ≥ 30 criterion
fired 305×/volume on dealias noise up to 368 km out — see campaign notes):
a peak must be spatially coherent (a neighboring ray-pair/range-gate also
carries ≥ 25 m/s Δv — mesocyclones span multiple gates, dealias spikes don't)
and inside the same --crop-km scene as the volume archive. An opposite-sign
(inbound|outbound) requirement was tried and REJECTED: it deleted the actual
Greenfield EF4 couplet, whose pair at 20:26:00Z was (−62.9, −9.5) m/s —
violent rotation embedded in strong one-signed inflow.

Archive size gate: full-window storm4d-volume ≤ 1.2 GB — tighten (in order)
--az-decim-deg 1.5, then --dbz-floor 15 if the projection busts it.

Pipeline:  Level II (S3, cached)  →  Py-ART decode+dealias  →  GeoParquet  →  stt-build

Use --skip-build to stop at the GeoParquet, or --stt-build /path to the binary.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import subprocess
import sys
import time as time_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BUCKET = "unidata-nexrad-level2"
BASE = f"https://{BUCKET}.s3.amazonaws.com"

# Storm reference point (crop center): Greenfield, Iowa (§9.0).
REF_LAT, REF_LON = 41.305, -94.461

# Smoke window for pipeline validation (§9.0).
SMOKE_START, SMOKE_END = "2024-05-21T20:00:00Z", "2024-05-21T20:30:00Z"

EARTH_R = 6_371_000.0

# ── contract band labels (§9.1 — byte-for-byte, FE colorMapping keys match) ──
DBZ_EDGES = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], dtype=np.float64)
DBZ_LABELS = [
    "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45",
    "45-50", "50-55", "55-60", "60-65", "65-70", "70+",
]
VEL_LABELS = [
    "in-extreme", "in-strong", "in-mod", "calm", "out-mod", "out-strong", "out-extreme",
]

# Couplet-detection constants (§9.1 storm4d-couplet + measured refinements).
COUPLET_RANGE_BIN_M = 2_000.0
COUPLET_CLUSTER_KM = 5.0
# A neighboring ray-pair / range-gate must also carry this much Δv
# (single-gate spikes are dealias noise, mesocyclones span multiple gates).
# NOTE deliberately NO opposite-sign requirement — the Greenfield EF4 couplet
# itself was one-signed at 0.5° (−62.9 | −9.5 m/s), rotation riding inflow.
COUPLET_SUPPORT_MS = 25.0

# Minimum unmasked fraction for a sweep to count as "has this moment".
COVERAGE_MIN = 0.005
# Split-cut pairing: CS and CD sweeps of one tilt agree within this angle.
PAIR_ANGLE_TOL_DEG = 0.3

_TS_TOKEN = re.compile(r"(\d{8})_(\d{6})")


def parse_when(spec: str) -> datetime:
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def key_time(name: str) -> datetime | None:
    """Volume start time encoded in a Level II key basename."""
    m = _TS_TOKEN.search(name)
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    return datetime(
        int(d[0:4]), int(d[4:6]), int(d[6:8]),
        int(t[0:2]), int(t[2:4]), int(t[4:6]), tzinfo=timezone.utc,
    )


# ── S3 listing (anonymous ListObjectsV2 REST → XML; glm_lightning pattern) ────
def list_prefix(prefix: str, retries: int = 4) -> list[str]:
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    keys: list[str] = []
    token: str | None = None
    while True:
        url = f"{BASE}/?list-type=2&prefix={prefix}"
        if token:
            url += f"&continuation-token={quote(token, safe='')}"
        body = None
        for attempt in range(retries):
            try:
                with urlopen(url, timeout=60) as r:
                    body = r.read()
                break
            except Exception:
                if attempt == retries - 1:
                    return keys
        root = ET.fromstring(body)
        token = None
        for el in root:
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "Contents":
                for child in el:
                    if child.tag.rsplit("}", 1)[-1] == "Key" and child.text:
                        keys.append(child.text)
            elif tag == "NextContinuationToken" and el.text:
                token = el.text
        if not token:
            return keys


def gather_keys(site: str, start: datetime, end: datetime) -> list[str]:
    """Every V06 volume key (no _MDM) whose start time falls in [start, end]."""
    keys: list[str] = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        keys.extend(list_prefix(f"{day:%Y/%m/%d}/{site}/"))
        day += timedelta(days=1)
    picked = []
    for k in keys:
        name = Path(k).name
        if not name.endswith("_V06"):
            continue  # *_MDM sidecars and any non-V06 relics
        t = key_time(name)
        if t is not None and start <= t <= end:
            picked.append(k)
    picked.sort()
    return picked


def fetch(key: str, cache: Path, retries: int = 3) -> Path | None:
    path = cache / Path(key).name
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urlopen(f"{BASE}/{key}", timeout=300) as r:
                data = r.read()
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)
            return path
        except Exception:
            if attempt == retries - 1:
                return None
    return None


def download_all(keys: list[str], cache: Path, workers: int) -> list[Path]:
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, k, cache): k for k in keys}
        for i, f in enumerate(as_completed(futs)):
            p = f.result()
            if p is not None:
                paths.append(p)
            if (i + 1) % 10 == 0 or i + 1 == len(keys):
                print(f"  fetched {i + 1}/{len(keys)} volume(s)")
    paths.sort()
    return paths


# ── geometry helpers ──────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km); vectorized over the first pair."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R / 1000.0 * np.arcsin(np.sqrt(a))


def nearest_az_rows(src_az: np.ndarray, dst_az: np.ndarray) -> np.ndarray:
    """For each dst azimuth, the row index of the circularly nearest src azimuth."""
    order = np.argsort(src_az)
    s = src_az[order]
    idx = np.searchsorted(s, dst_az)
    i0 = (idx - 1) % len(s)
    i1 = idx % len(s)
    d0 = np.abs((dst_az - s[i0] + 180.0) % 360.0 - 180.0)
    d1 = np.abs((s[i1] - dst_az + 180.0) % 360.0 - 180.0)
    return order[np.where(d0 <= d1, i0, i1)]


# ── per-volume worker ─────────────────────────────────────────────────────────
def _sweep_field(radar, sweep: int, name: str):
    arr = radar.get_field(sweep, name)
    return arr.filled(np.nan).astype(np.float64), ~np.ma.getmaskarray(arr)


def _coverage(radar, sweep: int, name: str) -> float:
    return float((~np.ma.getmaskarray(radar.get_field(sweep, name))).mean())


def _detect_couplets(vel: np.ndarray, valid: np.ndarray, az: np.ndarray,
                     rng: np.ndarray, geom, t_ms: int, min_dv: float,
                     crop_km: float) -> list[tuple]:
    """Azimuthal-shear couplet candidates on one dealiased velocity sweep.

    Returns rows (lon, lat, t_ms, strength, alt) — one per ≥min_dv peak cluster
    that is spatially coherent and inside the demo scene.
    """
    order = np.argsort(az)
    v = vel[order]
    ok = valid[order]
    # Gate-to-gate Δv between azimuthally adjacent radials (wraparound pair incl.)
    v2 = np.roll(v, -1, axis=0)
    ok2 = ok & np.roll(ok, -1, axis=0)
    dv = np.abs(v2 - v)
    dv[~ok2] = 0.0
    dv[~np.isfinite(dv)] = 0.0
    # Spatial coherence: a neighboring ray-pair or range gate carries support Δv.
    support = np.maximum.reduce([
        np.roll(dv, 1, axis=0), np.roll(dv, -1, axis=0),
        np.roll(dv, 1, axis=1), np.roll(dv, -1, axis=1),
    ])
    dv[support < COUPLET_SUPPORT_MS] = 0.0

    lat2d, lon2d, alt2d = geom
    lat_s, lon_s, alt_s = lat2d[order], lon2d[order], alt2d[order]
    # Same scene as the volume archive: markers outside the crop are other
    # storms' rotation and would drag the demo's eye off the narrative.
    dv[haversine_km(lat_s.astype(np.float64), lon_s.astype(np.float64),
                    REF_LAT, REF_LON) > crop_km] = 0.0

    rbin = (rng // COUPLET_RANGE_BIN_M).astype(np.int64)
    starts = np.unique(rbin, return_index=True)[1]
    cand: list[tuple[float, int, int]] = []  # (strength, sorted_ray, gate)
    for b, s0 in enumerate(starts):
        s1 = starts[b + 1] if b + 1 < len(starts) else dv.shape[1]
        block = dv[:, s0:s1]
        g_rel = np.argmax(block, axis=1)
        strength = block[np.arange(block.shape[0]), g_rel]
        for ray in np.nonzero(strength >= min_dv)[0]:
            cand.append((float(strength[ray]), int(ray), int(s0 + g_rel[ray])))
    if not cand:
        return []

    cand.sort(reverse=True)  # strongest first → greedy 5 km suppression
    out: list[tuple] = []
    kept: list[tuple[float, float]] = []
    for strength, ray, gate in cand:
        la, lo = float(lat_s[ray, gate]), float(lon_s[ray, gate])
        if any(haversine_km(la, lo, kla, klo) < COUPLET_CLUSTER_KM for kla, klo in kept):
            continue
        kept.append((la, lo))
        out.append((lo, la, t_ms, strength, float(alt_s[ray, gate])))
    return out


def process_volume(task: tuple) -> dict:
    """Decode + dealias one Level II volume → decimated gate arrays + couplets."""
    path_str, dbz_floor, crop_km, range_decim_km, az_decim_deg, min_dv = task
    import warnings

    warnings.filterwarnings("ignore")
    import pyart  # deferred: heavy import happens in the worker process

    t_start = time_mod.perf_counter()
    name = Path(path_str).name
    stats = dict(name=name, raw=0, floor=0, crop=0, decim=0, couplets=0, error=None)
    try:
        radar = pyart.io.read_nexrad_archive(path_str)
        corr = pyart.correct.dealias_region_based(radar, vel_field="velocity")
        radar.add_field("corrected_velocity", corr, replace_existing=True)
    except Exception as e:  # truncated/corrupt volume: report, don't kill the run
        stats["error"] = f"{type(e).__name__}: {e}"
        stats["seconds"] = time_mod.perf_counter() - t_start
        return {"stats": stats, "gates": None, "couplets": []}

    base_time = parse_when(radar.time["units"].split("since", 1)[1])
    rng = radar.range["data"].astype(np.float64)
    n_sweeps = radar.nsweeps
    angles = radar.fixed_angle["data"]
    ref_cov = [_coverage(radar, s, "reflectivity") for s in range(n_sweeps)]
    vel_cov = [_coverage(radar, s, "velocity") for s in range(n_sweeps)]

    # Split-cut pairing: a CS (refl-only) sweep takes the adjacent CD sweep at
    # the same tilt as its velocity donor; the donor's own reflectivity is then
    # NOT emitted (it would double the tilt).
    donor_of: dict[int, int | None] = {}
    consumed: set[int] = set()
    for s in range(n_sweeps):
        if ref_cov[s] < COVERAGE_MIN:
            continue
        if vel_cov[s] >= COVERAGE_MIN:
            continue  # has its own velocity — not a CS cut
        donor = None
        for j in (s + 1, s - 1):
            if 0 <= j < n_sweeps and j not in consumed \
                    and abs(float(angles[j]) - float(angles[s])) <= PAIR_ANGLE_TOL_DEG \
                    and vel_cov[j] >= COVERAGE_MIN:
                donor = j
                break
        donor_of[s] = donor
        if donor is not None:
            consumed.add(donor)

    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("lon", "lat", "t", "alt", "dbz", "vel", "sweep")
    }
    for s in range(n_sweeps):
        if ref_cov[s] < COVERAGE_MIN or (s in consumed):
            continue
        ref, ref_ok = _sweep_field(radar, s, "reflectivity")
        i0 = radar.sweep_start_ray_index["data"][s]
        t_ms = int((base_time + timedelta(seconds=float(radar.time["data"][i0]))
                    ).timestamp() * 1000)
        lat2d, lon2d, alt2d = radar.get_gate_lat_lon_alt(s)
        stats["raw"] += int(ref_ok.sum())

        keep = ref_ok & (ref >= dbz_floor)
        stats["floor"] += int(keep.sum())
        dist = haversine_km(lat2d.astype(np.float64), lon2d.astype(np.float64),
                            REF_LAT, REF_LON)
        keep &= dist <= crop_km
        n_kept = int(keep.sum())
        stats["crop"] += n_kept
        if n_kept == 0:
            continue

        # ── decimate to (az_decim_deg × range_decim_km) cells ────────────────
        az = radar.azimuth["data"][i0: radar.sweep_end_ray_index["data"][s] + 1]
        ray_idx, gate_idx = np.nonzero(keep)
        g_az = az[ray_idx].astype(np.float64) % 360.0
        g_r = rng[gate_idx]
        r_step = range_decim_km * 1000.0
        az_bin = np.minimum((g_az // az_decim_deg).astype(np.int64),
                            int(np.ceil(360.0 / az_decim_deg)) - 1)
        r_bin = (g_r // r_step).astype(np.int64)
        bin_id = az_bin * (int(rng[-1] // r_step) + 2) + r_bin
        uniq, inv = np.unique(bin_id, return_inverse=True)

        # dBZ: cell average in LINEAR Z units (10^(dBZ/10)), back to dBZ.
        z_lin = np.power(10.0, ref[ray_idx, gate_idx] / 10.0)
        cell_dbz = 10.0 * np.log10(
            np.bincount(inv, weights=z_lin) / np.bincount(inv))

        # Representative (center) gate per cell: nearest to the cell center in
        # normalized (az, range) — velocity is circular, so NO averaging.
        d_az = ((g_az - (az_bin + 0.5) * az_decim_deg + 180.0) % 360.0 - 180.0) / az_decim_deg
        d_r = (g_r - (r_bin + 0.5) * r_step) / r_step
        penalty = d_az * d_az + d_r * d_r
        order = np.lexsort((penalty, inv))
        first = np.unique(inv[order], return_index=True)[1]
        rep = order[first]  # cell order matches `uniq` (ascending bin id)
        rep_ray, rep_gate = ray_idx[rep], gate_idx[rep]

        # Velocity at the representative gate: own sweep, or the paired CD
        # donor at the circularly nearest azimuth (same range gate — the range
        # table is shared across sweeps).
        vel_vals = np.full(len(rep), np.nan)
        donor = donor_of.get(s) if s in donor_of else (s if vel_cov[s] >= COVERAGE_MIN else None)
        if donor is not None:
            vel, vel_ok = _sweep_field(radar, donor, "corrected_velocity")
            if donor == s:
                v_ray = rep_ray
            else:
                d0 = radar.sweep_start_ray_index["data"][donor]
                d1 = radar.sweep_end_ray_index["data"][donor]
                v_ray = nearest_az_rows(radar.azimuth["data"][d0: d1 + 1] % 360.0,
                                        az[rep_ray] % 360.0)
            ok = vel_ok[v_ray, rep_gate]
            vel_vals[ok] = vel[v_ray, rep_gate][ok]

        n = len(rep)
        stats["decim"] += n
        cols["lon"].append(lon2d[rep_ray, rep_gate].astype(np.float64))
        cols["lat"].append(lat2d[rep_ray, rep_gate].astype(np.float64))
        cols["t"].append(np.full(n, t_ms, dtype=np.int64))
        cols["alt"].append(alt2d[rep_ray, rep_gate].astype(np.float32))
        cols["dbz"].append(cell_dbz.astype(np.float32))
        cols["vel"].append(vel_vals.astype(np.float32))
        cols["sweep"].append(np.full(n, float(angles[s]), dtype=np.float32))

    # ── couplet detection: every lowest-tilt dealiased velocity sweep ────────
    couplets: list[tuple] = []
    vel_sweeps = [s for s in range(n_sweeps) if vel_cov[s] >= COVERAGE_MIN]
    if vel_sweeps:
        low = min(float(angles[s]) for s in vel_sweeps)
        for s in vel_sweeps:
            if abs(float(angles[s]) - low) > 0.05:
                continue
            vel, vel_ok = _sweep_field(radar, s, "corrected_velocity")
            i0 = radar.sweep_start_ray_index["data"][s]
            i1 = radar.sweep_end_ray_index["data"][s]
            t_ms = int((base_time + timedelta(seconds=float(radar.time["data"][i0]))
                        ).timestamp() * 1000)
            if not (vel_ok.any()):
                continue
            couplets.extend(_detect_couplets(
                vel, vel_ok, radar.azimuth["data"][i0: i1 + 1] % 360.0, rng,
                radar.get_gate_lat_lon_alt(s), t_ms, min_dv, crop_km))
    stats["couplets"] = len(couplets)

    gates = {k: np.concatenate(v) if v else np.empty(0) for k, v in cols.items()}
    stats["seconds"] = time_mod.perf_counter() - t_start
    return {"stats": stats, "gates": gates, "couplets": couplets}


# ── band columns (contract labels; §9.1) ──────────────────────────────────────
def dbz_band_column(dbz: np.ndarray) -> pa.Array:
    idx = np.digitize(dbz, DBZ_EDGES).astype(np.int32)
    return pc.take(pa.array(DBZ_LABELS), pa.array(idx))


def vel_band_column(vel: np.ndarray) -> pa.Array:
    # Negative = toward the radar. Bin edges per §9.1 (inclusive as written).
    idx = np.select(
        [vel <= -40, vel <= -25, vel <= -10, vel < 10, vel < 25, vel < 40],
        [0, 1, 2, 3, 4, 5],
        default=6,
    ).astype(np.int32)
    return pc.take(pa.array(VEL_LABELS),
                   pa.array(idx, mask=~np.isfinite(vel)))


# ── stratified LOD min-zoom assignment ────────────────────────────────────────
def lod_min_zoom(lon: np.ndarray, lat: np.ndarray, alt: np.ndarray, dbz: np.ndarray,
                 min_zoom: int, max_zoom: int,
                 cell_deg_top: float, cell_alt_top: float) -> np.ndarray:
    """Assign each gate the coarsest zoom at which it survives a nested 3D-grid
    thinning, so `--min-zoom-field` yields a real spatial LOD pyramid for a
    homogeneous point cloud (where `--maximum-tile-features --drop-densest` is a
    no-op — every single-vertex gate scores equally, so the cap just truncates
    scan order and drops whole regions).

    At each zoom below `max_zoom` we lay a 3D grid (horizontal `cell_deg`, vertical
    `cell_alt`) sized `cell_*_top` at `max_zoom-1` and DOUBLING per zoom out, and
    keep one representative per cell — the strongest-echo gate (so the coarse
    skeleton is the meaningful storm core, not noise). A gate's `min_zoom` is the
    coarsest zoom where it wins its cell; gates that never win default to
    `max_zoom` → the deepest tier stays lossless (every gate present). Deterministic
    (stable argsort on dBZ). Returns an int16 column.
    """
    n = len(lon)
    alt = np.nan_to_num(alt.astype(np.float64), nan=0.0)
    dbz = np.nan_to_num(dbz.astype(np.float64), nan=-99.0)
    lon0, lat0, alt0 = lon.min(), lat.min(), alt.min()
    order = np.argsort(-dbz, kind="stable")          # strongest echoes win cells
    min_z = np.full(n, max_zoom, dtype=np.int16)
    for z in range(min_zoom, max_zoom):
        scale = 2 ** (max_zoom - 1 - z)              # top tier = 1×, coarser doubles
        cd, ca = cell_deg_top * scale, cell_alt_top * scale
        cx = ((lon - lon0) / cd).astype(np.int64)
        cy = ((lat - lat0) / cd).astype(np.int64)
        cz = ((alt - alt0) / ca).astype(np.int64)
        code = (cx << 40) ^ (cy << 20) ^ cz
        _, first = np.unique(code[order], return_index=True)
        np.minimum.at(min_z, order[first], z)        # rep of this z-cell
    return min_z


# ── stt-build (glm_lightning / storms flag style) ─────────────────────────────
def run_stt_build(parquet: Path, out_dir: Path, stt_build: str,
                  min_zoom: int, max_zoom: int, quantize: bool, publish: bool,
                  min_zoom_field: str | None = None) -> None:
    cmd = [
        stt_build,
        "--input", str(parquet),
        "--output", str(out_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", str(min_zoom),
        "--max-zoom", str(max_zoom),
        "--temporal-bucket", "5m",
        "--compression", "zstd",
        # Multi-cell time PLAYBACK requires time-major ordering — `auto` can
        # pick spatial and silently stall the playhead (blob-ordering gotcha).
        "--blob-ordering", "time-major",
    ]
    if min_zoom_field:
        # Stratified LOD pyramid (§5, §9.1 reduced-tier amendment): each gate's
        # `min_zoom` column names the coarsest zoom it appears at. Coarse zooms
        # carry a spatially-uniform, strongest-echo-first skeleton; the deepest
        # zoom stays LOSSLESS (every gate present). Cuts per-bucket resident
        # gates ~7× at the framing zoom AND shrinks the archive ~2.5× — with
        # zero data loss, since full detail is one zoom-in away.
        cmd += ["--min-zoom-field", min_zoom_field]
    if quantize:
        cmd += ["--quantize-coords", "100", "--quantize-attrs-auto"]
    if publish:
        cmd += ["--publish"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", help="window start, ISO-8601 UTC (e.g. 2024-05-21T17:30Z)")
    ap.add_argument("--end", help="window end, ISO-8601 UTC")
    ap.add_argument("--smoke", action="store_true",
                    help=f"pipeline-validation window {SMOKE_START} → {SMOKE_END}")
    ap.add_argument("--site", default="KDMX", help="radar site (default KDMX)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="directory receiving storm4d-volume/ + storm4d-couplet/ "
                         "archives (and their .parquet intermediates)")
    ap.add_argument("--cache", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "storm4d" / "nexrad")
    # Contract knobs (§9.1) — defaults are the ratified values.
    ap.add_argument("--dbz-floor", type=float, default=10.0)
    ap.add_argument("--crop-km", type=float, default=150.0)
    ap.add_argument("--range-decim-km", type=float, default=0.5)
    ap.add_argument("--az-decim-deg", type=float, default=1.0)
    ap.add_argument("--couplet-min-dv", type=float, default=30.0,
                    help="min gate-to-gate azimuthal Δv (m/s) for a couplet peak")
    # Stratified LOD pyramid (lossless — deepest zoom keeps every gate). The two
    # cell knobs set the grid at the top LOD tier (z8); shrink them to densify
    # the coarse-zoom skeleton (higher fidelity at the framing zoom, less perf).
    ap.add_argument("--no-lod", action="store_true",
                    help="disable the min-zoom LOD pyramid (legacy full-duplication build)")
    ap.add_argument("--lod-cell-deg", type=float, default=0.0015,
                    help="horizontal LOD cell (deg) at the z8 tier; doubles per zoom out")
    ap.add_argument("--lod-cell-alt", type=float, default=250.0,
                    help="vertical LOD cell (m) at the z8 tier; doubles per zoom out")
    ap.add_argument("--workers", type=int, default=6,
                    help="volume-decode processes (each ~1.5 GB peak)")
    ap.add_argument("--skip-fetch", action="store_true", help="use cached volumes only")
    ap.add_argument("--skip-build", action="store_true", help="stop at parquet")
    ap.add_argument("--publish", action="store_true", help="zstd-19 deploy build")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    if args.smoke:
        start, end = parse_when(SMOKE_START), parse_when(SMOKE_END)
    elif args.start and args.end:
        start, end = parse_when(args.start), parse_when(args.end)
    else:
        sys.exit("Pass --start and --end (or --smoke).")
    if end <= start:
        sys.exit("--end must be after --start")

    print(f"Listing {BUCKET} {args.site} volumes {start:%Y-%m-%dT%H:%M}Z → {end:%Y-%m-%dT%H:%M}Z …")
    keys = gather_keys(args.site, start, end)
    if not keys:
        sys.exit("No volumes listed for that window — check --start/--end/--site.")
    print(f"  {len(keys)} volume(s) in window")

    if args.skip_fetch:
        paths = sorted(p for p in (args.cache / Path(k).name for k in keys) if p.exists())
        print(f"Using {len(paths)} cached volume(s).")
    else:
        print(f"Downloading into {args.cache} …")
        paths = download_all(keys, args.cache, workers=8)
    if not paths:
        sys.exit("No volumes on disk — nothing to decode.")

    tasks = [(str(p), args.dbz_floor, args.crop_km, args.range_decim_km,
              args.az_decim_deg, args.couplet_min_dv) for p in paths]
    print(f"Decoding {len(tasks)} volume(s) on {args.workers} worker(s) …")
    t_all = time_mod.perf_counter()
    results = []
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for res in pool.imap_unordered(process_volume, tasks):
            st = res["stats"]
            if st["error"]:
                print(f"  [{st['name']}] FAILED: {st['error']}")
            else:
                print(f"  [{st['name']}] {st['seconds']:6.1f}s  "
                      f"raw={st['raw']:>9,} floor={st['floor']:>9,} "
                      f"crop={st['crop']:>9,} decim={st['decim']:>8,} "
                      f"couplets={st['couplets']}")
            results.append(res)
    wall = time_mod.perf_counter() - t_all

    good = [r for r in results if r["stats"]["error"] is None]
    if not good:
        sys.exit("Every volume failed to decode.")
    totals = {k: sum(r["stats"][k] for r in good) for k in ("raw", "floor", "crop", "decim")}
    secs = [r["stats"]["seconds"] for r in good]
    print(f"\nGate funnel over {len(good)} volume(s) "
          f"(floor ≥{args.dbz_floor:g} dBZ, crop ≤{args.crop_km:g} km, "
          f"decim {args.range_decim_km:g} km × {args.az_decim_deg:g}°):")
    print(f"  raw (valid refl on emitting sweeps): {totals['raw']:>12,}")
    print(f"  after dBZ floor:                     {totals['floor']:>12,}")
    print(f"  after crop:                          {totals['crop']:>12,}")
    print(f"  after decimation (emitted):          {totals['decim']:>12,}")
    print(f"  per-volume decode: mean {np.mean(secs):.1f}s  max {np.max(secs):.1f}s  "
          f"(pool wall {wall:.1f}s)")

    # ── volume parquet ────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gates = {k: np.concatenate([r["gates"][k] for r in good if r["gates"] is not None])
             for k in ("lon", "lat", "t", "alt", "dbz", "vel", "sweep")}
    order = np.argsort(gates["t"], kind="stable")
    gates = {k: v[order] for k, v in gates.items()}
    nan_vel = ~np.isfinite(gates["vel"])
    cols = {
        "lon": pa.array(gates["lon"], type=pa.float64()),
        "lat": pa.array(gates["lat"], type=pa.float64()),
        "timestamp": pa.array(gates["t"], type=pa.int64()),
        "alt_m": pa.array(gates["alt"], type=pa.float32()),
        "dbz": pa.array(gates["dbz"], type=pa.float32()),
        "vel_ms": pa.array(gates["vel"], type=pa.float32(), mask=nan_vel),
        "sweep_deg": pa.array(gates["sweep"], type=pa.float32()),
        "dbz_band": dbz_band_column(gates["dbz"]),
        "vel_band": vel_band_column(gates["vel"]),
    }
    if not args.no_lod:
        mz = lod_min_zoom(gates["lon"], gates["lat"], gates["alt"], gates["dbz"],
                          min_zoom=4, max_zoom=9,
                          cell_deg_top=args.lod_cell_deg, cell_alt_top=args.lod_cell_alt)
        cols["min_zoom"] = pa.array(mz, type=pa.int16())
        cum = np.cumsum(np.bincount(mz - 4, minlength=6))
        print("  LOD min-zoom pyramid (gates loaded at each zoom / % of full):")
        for i, z in enumerate(range(4, 10)):
            print(f"    z{z}: {cum[i]:>10,}  ({100.0 * cum[i] / max(len(mz), 1):5.1f}%)")
    table = pa.table(cols)
    vol_pq = args.out_dir / "storm4d-volume.parquet"
    pq.write_table(table, vol_pq, compression="snappy")
    with_vel = int((~nan_vel).sum())
    print(f"\nWrote {table.num_rows:,} gates → {vol_pq}")
    print(f"  with velocity: {with_vel:,} ({100.0 * with_vel / max(table.num_rows, 1):.1f}%)")

    # ── couplet parquet ───────────────────────────────────────────────────────
    rows = sorted((c for r in good for c in r["couplets"]), key=lambda c: c[2])
    coup_pq = args.out_dir / "storm4d-couplet.parquet"
    if rows:
        lon, lat, t, strength, alt = (np.array(x) for x in zip(*rows))
        pq.write_table(pa.table({
            "lon": pa.array(lon, type=pa.float64()),
            "lat": pa.array(lat, type=pa.float64()),
            "timestamp": pa.array(t.astype(np.int64), type=pa.int64()),
            "strength_ms": pa.array(strength.astype(np.float32), type=pa.float32()),
            "alt_m": pa.array(alt.astype(np.float32), type=pa.float32()),
        }), coup_pq, compression="snappy")
        print(f"Wrote {len(rows):,} couplet marker(s) → {coup_pq}")
    else:
        print("No couplet markers detected in window — skipping couplet parquet.")

    if args.skip_build:
        print("\nGeoParquet only (--skip-build).")
        return 0

    # storm4d-volume: POINTS z4-9, 5m buckets (§9.1). alt_m stays a PROPERTY
    # column (§9.0 LiDAR pattern — elevationProperty + one shared
    # STORM4D_ELEVATION_SCALE in the FE), NOT --point-elevation-column.
    run_stt_build(vol_pq, args.out_dir / "storm4d-volume", args.stt_build,
                  min_zoom=4, max_zoom=9, quantize=True, publish=args.publish,
                  min_zoom_field=None if args.no_lod else "min_zoom")
    if rows:
        # storm4d-couplet: a handful of markers — z3-9, no quantization needed.
        run_stt_build(coup_pq, args.out_dir / "storm4d-couplet", args.stt_build,
                      min_zoom=3, max_zoom=9, quantize=False, publish=args.publish)
    return 0


if __name__ == "__main__":
    sys.exit(main())
