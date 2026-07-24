#!/usr/bin/env python3
"""Advect virtual particles through NOAA HRRR wind → STT streamline ribbons.

The wind member of the weather suite — the atmospheric analog of ecco_advect.py
(which does the same for ocean currents). It reads HRRR U/V wind at a chosen
level for every hour of a window, seeds a cloud of massless particles over
CONUS, integrates each through the *time-varying* wind with 4th-order
Runge–Kutta, and emits every particle path as one LineString carrying
per-vertex timestamps and a per-vertex wind-speed value — the exact `trips`
shape the showcase renders, so the trips renderer, time scrubber, and speed
gradient all work unchanged (earth.nullschool look via ribbons; no GPU
particle tier).

`--level` picks the atmospheric layer (default **500mb**, the storm STEERING
level): convective systems move with the mid-tropospheric flow, not the 10 m
surface wind, so 500 mb streamlines visually track MRMS storm motion in the
weather-suite composite while `--level 10m` runs slower and rotated relative
to the cells. Pressure levels read the wrfprsf00 pressure file; `10m` reads
the wrfsfcf00 surface file.

Source: HRRR (High-Resolution Rapid Refresh) on the anonymous
`noaa-hrrr-bdp-pds` S3 bucket, GRIB2, 3 km CONUS, hourly:

    hrrr.<YYYYMMDD>/conus/hrrr.t<HH>z.wrfsfcf00.grib2   (10 m)
    hrrr.<YYYYMMDD>/conus/hrrr.t<HH>z.wrfprsf00.grib2   (pressure levels)

HRRR is on a Lambert-Conformal grid, so for each hour we byte-range-subset ONLY
the two `UGRD`/`VGRD` messages for the level (~1-5 MB, via the `.idx` sidecar —
no full-file download), read them with cfgrib, and regrid the native LCC field
onto one common regular lat/lon grid (scipy) so the ECCO bilinear sampler +
advection core apply unchanged.

Pipeline:  HRRR GRIB2 (.idx subset)  →  regrid  →  RK4 advection  →  GeoParquet  →  stt-build

Use --skip-build to stop at the GeoParquet, or --stt-build /path to point at the binary.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

EARTH_R = 6_371_000.0
M_PER_DEG_LAT = math.pi * EARTH_R / 180.0
MS_PER_HOUR = 3_600_000

BUCKET = "noaa-hrrr-bdp-pds"
BASE = f"https://{BUCKET}.s3.amazonaws.com"
CONUS = (24.0, -122.0, 50.0, -67.0)  # min_lat, min_lon, max_lat, max_lon


def parse_bounds(spec: str):
    p = [float(x) for x in spec.split(",")]
    if len(p) != 4:
        sys.exit("--bounds must be min_lat,min_lon,max_lat,max_lon")
    return tuple(p)


def parse_when(spec: str) -> datetime:
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── HRRR download: byte-range subset of U/V at one level via the .idx sidecar ──
def level_spec(level: str) -> tuple[str, str, str]:
    """(file suffix, .idx level tag, cache tag) for a `--level` value.
    `10m` lives in the surface file; pressure levels (e.g. `500mb`) in the
    pressure file, whose .idx tags them like "500 mb"."""
    if level == "10m":
        return "wrfsfcf00", "10 m above ground", "10m"
    if level.endswith("mb") and level[:-2].isdigit():
        return "wrfprsf00", f"{level[:-2]} mb", level
    sys.exit(f"--level must be '10m' or '<N>mb' (e.g. 500mb); got {level!r}")


def hrrr_key(hour: datetime, level: str) -> str:
    suffix, _, _ = level_spec(level)
    return f"hrrr.{hour:%Y%m%d}/conus/hrrr.t{hour:%H}z.{suffix}.grib2"


def cache_path(cache: Path, cache_tag: str, hour: datetime, want_temp: bool) -> Path:
    """Cache filename for one hour's subset. The `windtmp` variant is kept
    SEPARATE from `wind` so a temperature build never reuses a wind-only cache
    (and vice-versa)."""
    suffix = "windtmp" if want_temp else "wind"
    return cache / f"hrrr_{cache_tag}{suffix}_{hour:%Y%m%d_%H}z.grib2"


def fetch_wind_subset(hour: datetime, cache: Path, level: str,
                      want_temp: bool = False, retries: int = 3) -> Path | None:
    """Range-GET the UGRD+VGRD (and, when `want_temp`, TMP) messages at `level`
    for one hour, concatenated into one GRIB2 (~1-5 MB). Temperature rides in
    the SAME pressure/surface file as the wind, so it adds one more message to
    the byte-range subset — never a second file download."""
    key = hrrr_key(hour, level)
    _, idx_tag, cache_tag = level_spec(level)
    out = cache_path(cache, cache_tag, hour, want_temp)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    wanted = ("UGRD", "VGRD", "TMP") if want_temp else ("UGRD", "VGRD")
    for attempt in range(retries):
        try:
            idx = urlopen(f"{BASE}/{key}.idx", timeout=60).read().decode()
            lines = [ln.split(":") for ln in idx.splitlines() if ln]
            offs = {int(f[0]): int(f[1]) for f in lines if len(f) > 1}
            # Message number for each wanted variable at the level's .idx tag.
            msg: dict[str, int] = {}
            for f in lines:
                if len(f) > 4 and f[4] == idx_tag and f[3] in wanted:
                    msg[f[3]] = int(f[0])
            if any(v not in msg for v in wanted):
                # A dropped hour is invisible downstream (the decode loop just
                # `continue`s past the absent cache file), so name the miss.
                print(f"    ⚠️  {key}: .idx has no {idx_tag} "
                      f"{sorted(set(wanted) - set(msg))} — hour skipped")
                return None
            # One range-GET per message (messages need not be adjacent in the
            # file). GRIB2 is self-delimiting, so concatenating the messages in
            # any order yields a valid multi-message file cfgrib can read.
            chunks = []
            for v in wanted:
                m = msg[v]
                lo = offs[m]
                hi = offs.get(m + 1)  # None → to EOF (last message in the .idx)
                rng = f"bytes={lo}-" + ("" if hi is None else str(hi - 1))
                chunks.append(urlopen(
                    Request(f"{BASE}/{key}", headers={"Range": rng}),
                    timeout=120).read())
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(b"".join(chunks))
            tmp.rename(out)
            return out
        except Exception as e:
            if attempt == retries - 1:
                print(f"    ⚠️  {key}: fetch failed after {retries} tries "
                      f"({e}) — hour skipped")
                return None
    return None


def load_hrrr_wind(start, end, bounds, grid_step, cache, workers, skip_fetch,
                   native_stride, level, want_temp=False):
    """Return (times_ms, lats, lons, U, V, T): U/V are (T, ny, nx) on a regular
    lat/lon grid over `bounds` at ~grid_step degrees; NaN outside the HRRR
    domain. `T` is the matching air-temperature field in °C when `want_temp`,
    else None — sampled from the SAME GRIB file/level as the wind, so it shares
    the wind's exact 3 km / hourly spatial-temporal resolution by construction."""
    import xarray as xr
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    _, _, cache_tag = level_spec(level)
    hours = []
    h = start.replace(minute=0, second=0, microsecond=0)
    while h <= end:
        hours.append(h)
        h += timedelta(hours=1)

    if not skip_fetch:
        kind = "wind+temp" if want_temp else "wind"
        print(f"Fetching {len(hours)} hourly HRRR {level} {kind} subsets → {cache} …")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            done = 0
            for _ in ex.map(lambda hh: fetch_wind_subset(hh, cache, level, want_temp), hours):
                done += 1
                if done % 12 == 0 or done == len(hours):
                    print(f"  fetched {done}/{len(hours)}")

    min_lat, min_lon, max_lat, max_lon = bounds
    lons = np.arange(min_lon, max_lon + 1e-9, grid_step)
    lats = np.arange(min_lat, max_lat + 1e-9, grid_step)
    LON, LAT = np.meshgrid(lons, lats)

    times, U, V, T = [], [], [], []
    tri = None
    for hh in hours:
        p = cache_path(cache, cache_tag, hh, want_temp)
        if not p.exists():
            continue
        try:
            ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
            # Surface subsets decode as u10/v10; pressure-level subsets as u/v.
            u_name = "u10" if "u10" in ds else "u"
            v_name = "v10" if "v10" in ds else "v"
            u = ds[u_name].values.astype("float64")
            v = ds[v_name].values.astype("float64")
            # Temperature: pressure-level TMP decodes as `t`, 2 m surface as `t2m`.
            tK = None
            if want_temp:
                t_name = "t" if "t" in ds else ("t2m" if "t2m" in ds else None)
                if t_name is not None:
                    tK = ds[t_name].values.astype("float64")
            glat = ds["latitude"].values.astype("float64")
            glon = ((ds["longitude"].values.astype("float64") + 180.0) % 360.0) - 180.0
            ds.close()
        except Exception as e:
            print(f"    ⚠️  {p.name}: {e}")
            continue

        sub = slice(None, None, native_stride)
        pts = np.column_stack([glon.ravel()[sub], glat.ravel()[sub]])
        if tri is None:
            tri = Delaunay(pts)  # native grid is fixed → triangulate once, reuse
        iu = LinearNDInterpolator(tri, u.ravel()[sub])
        iv = LinearNDInterpolator(tri, v.ravel()[sub])
        U.append(iu(LON, LAT).astype("float32"))
        V.append(iv(LON, LAT).astype("float32"))
        if want_temp:
            if tK is not None:
                it = LinearNDInterpolator(tri, tK.ravel()[sub] - 273.15)  # K → °C
                T.append(it(LON, LAT).astype("float32"))
            else:
                print(f"    ⚠️  {p.name}: no TMP message — filling NaN")
                T.append(np.full(LON.shape, np.nan, dtype="float32"))
        times.append(int(hh.timestamp() * 1000))

    if len(times) < 2:
        sys.exit("Need ≥2 hourly HRRR frames to advect; check the window/cache.")
    print(f"  loaded {len(times)} frames on a {len(lats)}×{len(lons)} regular grid"
          f"{' (with air temperature)' if want_temp else ''}")
    return (np.array(times, dtype="int64"), lats, lons,
            np.stack(U), np.stack(V), np.stack(T) if want_temp else None)


# ── Bilinear sampler over a REGIONAL regular grid (NaN outside; no wrap) ───────
class RegionalField:
    def __init__(self, lats, lons):
        self.lat0, self.lon0 = lats[0], lons[0]
        self.dlat = lats[1] - lats[0]
        self.dlon = lons[1] - lons[0]
        self.nlat, self.nlon = len(lats), len(lons)

    def sample(self, fe, fn, plon, plat):
        fy = (plat - self.lat0) / self.dlat
        fx = (plon - self.lon0) / self.dlon
        inside = (fx >= 0) & (fx <= self.nlon - 1) & (fy >= 0) & (fy <= self.nlat - 1)
        y0 = np.clip(np.floor(fy).astype(np.int64), 0, self.nlat - 2)
        x0 = np.clip(np.floor(fx).astype(np.int64), 0, self.nlon - 2)
        wy, wx = fy - y0, fx - x0
        y1, x1 = y0 + 1, x0 + 1

        def bilin(fld):
            v00, v01 = fld[y0, x0], fld[y0, x1]
            v10, v11 = fld[y1, x0], fld[y1, x1]
            w00 = (1 - wy) * (1 - wx); w01 = (1 - wy) * wx
            w10 = wy * (1 - wx); w11 = wy * wx
            sv = np.stack([v00, v01, v10, v11]); sw = np.stack([w00, w01, w10, w11])
            mask = ~np.isnan(sv); sw = np.where(mask, sw, 0.0)
            wsum = sw.sum(axis=0)
            vsum = np.where(mask, np.nan_to_num(sv) * sw, 0.0).sum(axis=0)
            out = np.full_like(wsum, np.nan, dtype="float64")
            ok = wsum > 1e-9
            out[ok] = vsum[ok] / wsum[ok]
            return out

        e = np.where(inside, bilin(fe), np.nan)
        n = np.where(inside, bilin(fn), np.nan)
        return e, n


def time_slabs(times_ms, t_ms, U, V):
    if t_ms <= times_ms[0]:
        return U[0], V[0]
    if t_ms >= times_ms[-1]:
        return U[-1], V[-1]
    j = int(np.searchsorted(times_ms, t_ms))
    t0, t1 = times_ms[j - 1], times_ms[j]
    f = (t_ms - t0) / (t1 - t0)
    return (1 - f) * U[j - 1] + f * U[j], (1 - f) * V[j - 1] + f * V[j]


def seed_particles(field: RegionalField, u0, v0, n, rng, bounds):
    min_lat, min_lon, max_lat, max_lon = bounds
    lon_s, lat_s = [], []
    while len(lon_s) < n:
        batch = max(n * 3, 3000)
        plon = rng.uniform(min_lon, max_lon, batch)
        plat = rng.uniform(min_lat, max_lat, batch)
        e, nn = field.sample(u0, v0, plon, plat)
        ok = ~np.isnan(e) & ~np.isnan(nn)
        lon_s.extend(plon[ok].tolist()); lat_s.extend(plat[ok].tolist())
    return np.array(lon_s[:n]), np.array(lat_s[:n])


def region_of(lon):
    return "West" if lon < -104 else ("Central" if lon < -90 else "East")


# ── RK4 advection (adapted from ecco_advect.py; hours instead of days) ─────────
def advect(times_ms, lats, lons, U, V, args, rng, bounds, T=None):
    field = RegionalField(lats, lons)
    t_start, t_end = int(times_ms[0]), int(times_ms[-1])
    step_ms = int(args.step_hours * MS_PER_HOUR)
    substeps = max(1, args.substeps)
    dt_ms = step_ms / substeps
    dt_s = dt_ms / 1000.0

    print(f"Seeding {args.particles} particles over CONUS …")
    plon, plat = seed_particles(field, U[0], V[0], args.particles, rng, bounds)
    n = len(plon)
    tl = [[plon[i]] for i in range(n)]
    ta = [[plat[i]] for i in range(n)]
    tt = [[t_start] for _ in range(n)]
    tv = [[np.nan] for _ in range(n)]
    # Per-vertex air temperature (°C), sampled alongside the wind. Seeded with
    # the ACTUAL temperature at the spawn point (not NaN) so a freshly-born
    # dot is already colored — no gray flash at particle birth.
    if T is not None:
        T0, _ = time_slabs(times_ms, t_start, T, T)
        tc0, _ = field.sample(T0, T0, plon, plat)
        tc = [[float(tc0[i])] for i in range(n)]
    else:
        tc = None
    finished: list[dict] = []

    def deriv(lon, lat, t_ms):
        e_fld, n_fld = time_slabs(times_ms, t_ms, U, V)
        e, nn = field.sample(e_fld, n_fld, lon, lat)
        dlat = nn / M_PER_DEG_LAT
        coslat = np.cos(np.radians(np.clip(lat, -89.9, 89.9)))
        dlon = e / (M_PER_DEG_LAT * np.maximum(coslat, 1e-3))
        return dlat, dlon, np.hypot(e, nn)

    n_steps = (t_end - t_start) // step_ms
    print(f"Advecting {n_steps} steps × {substeps} substeps (dt={dt_s/60:.0f} min) "
          f"over {(t_end - t_start)/MS_PER_HOUR:.0f} h …")

    lifetime_ms = float(args.lifetime_hours * MS_PER_HOUR)
    death = t_start + rng.uniform(0.2, 1.0, size=n) * lifetime_ms
    u0, v0 = U[0], V[0]

    def emit(i):
        if len(tl[i]) >= args.min_points:
            rec = {"lon": tl[i], "lat": ta[i], "t": tt[i], "v": tv[i]}
            if tc is not None:
                rec["temp"] = tc[i]
            finished.append(rec)

    t_ms = float(t_start)
    n_emit = 0
    for s in range(int(n_steps)):
        stuck = np.zeros(n, dtype=bool)
        for _ in range(substeps):
            la, lo = plat, plon
            k1a, k1o, _ = deriv(lo, la, t_ms)
            k2a, k2o, _ = deriv(lo + 0.5 * dt_s * k1o, la + 0.5 * dt_s * k1a, t_ms + dt_ms / 2)
            k3a, k3o, _ = deriv(lo + 0.5 * dt_s * k2o, la + 0.5 * dt_s * k2a, t_ms + dt_ms / 2)
            k4a, k4o, sp = deriv(lo + dt_s * k3o, la + dt_s * k3a, t_ms + dt_ms)
            nlat = la + (dt_s / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
            nlon = lo + (dt_s / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
            beached = np.isnan(k4a) | np.isnan(k4o) | np.isnan(sp)
            plat = np.where(beached, la, nlat)
            plon = np.where(beached, lo, nlon)
            stuck |= beached
            t_ms += dt_ms

        out_t = int(round(t_ms))
        e_fld, n_fld = time_slabs(times_ms, out_t, U, V)
        e_all, n_all = field.sample(e_fld, n_fld, plon, plat)
        sp_all = np.hypot(e_all, n_all)
        if tc is not None:
            T_slab, _ = time_slabs(times_ms, out_t, T, T)
            tc_all, _ = field.sample(T_slab, T_slab, plon, plat)
        retire = stuck | (t_ms >= death) | np.isnan(sp_all)

        for i in np.where(~retire)[0]:
            tl[i].append(float(plon[i])); ta[i].append(float(plat[i]))
            tt[i].append(out_t); tv[i].append(float(sp_all[i]))
            if tc is not None:
                tc[i].append(float(tc_all[i]))

        ret = np.where(retire)[0]
        for i in ret:
            emit(i)
        n_emit += len(ret)
        if len(ret):
            nlon, nlat = seed_particles(field, u0, v0, len(ret), rng, bounds)
            plon[ret], plat[ret] = nlon, nlat
            death[ret] = t_ms + rng.uniform(0.5, 1.0, size=len(ret)) * lifetime_ms
            if tc is not None:
                Tr, _ = time_slabs(times_ms, out_t, T, T)
                tcr, _ = field.sample(Tr, Tr, nlon, nlat)
            for j, i in enumerate(ret):
                tl[i] = [float(nlon[j])]; ta[i] = [float(nlat[j])]
                tt[i] = [out_t]; tv[i] = [np.nan]
                if tc is not None:
                    tc[i] = [float(tcr[j])]

        if n_steps and (s + 1) % max(1, n_steps // 10) == 0:
            print(f"  step {s+1}/{n_steps}  emitted={n_emit}")

    for i in range(n):
        emit(i)
    print(f"Produced {len(finished)} streamline tracks (≥{args.min_points} points).")
    return finished


# ── GeoParquet (trips) + stt-build ────────────────────────────────────────────
def write_geoparquet(tracks, out_path: Path, with_values: bool = True,
                     color_by: str = "speed"):
    """Write the advected tracks as a `trips` GeoParquet.

    `with_values=False` (the `--no-values` build) omits the per-vertex
    `vertex_values` column: a constant-color DOT field
    (`AnimatedTripHeadsLayer`, which reads only positions + per-vertex times)
    never touches it, and it is the second-largest per-vertex column — dropping
    it trims the archive for a background particle field with no visible loss.

    `color_by` selects what that per-vertex `vertex_values` column CARRIES:
    `'speed'` (wind speed, m/s — the streamline gradient) or `'temp'` (500 mb
    air temperature, °C — colors the drift-dot field by the air each particle
    rides). The per-track `speed`/`region` scalars are written regardless.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import wkb
    from shapely.geometry import LineString

    geom, ts, ets, vts, vvals, speed, region = [], [], [], [], [], [], []
    for tr in tracks:
        coords = list(zip(tr["lon"], tr["lat"]))
        if len(coords) < 2:
            continue
        geom.append(wkb.dumps(LineString(coords)))
        t = [int(x) for x in tr["t"]]
        ts.append(t[0]); ets.append(t[-1]); vts.append(t)
        sp = [float(x) if x == x else float("nan") for x in tr["v"]]
        if with_values:
            if color_by == "temp":
                # Temperature varies SMOOTHLY along a path, so quantizing to
                # 0.5 °C cuts entropy and shrinks the packed vertex_value column
                # (better zstd) with no visible banding at the dot scale — a
                # byte lever, not feature thinning.
                src = tr.get("temp", sp)
                vvals.append([
                    np.float32(round(x * 2.0) / 2.0) if x == x else np.float32("nan")
                    for x in src
                ])
            else:
                vvals.append([np.float32(x) for x in sp])
        finite = [x for x in sp if x == x]
        speed.append(float(np.mean(finite)) if finite else float("nan"))
        mid = coords[len(coords) // 2]
        region.append(region_of(mid[0]))
    cols = {
        "geometry": pa.array(geom, type=pa.binary()),
        "timestamp": pa.array(ts, type=pa.int64()),
        "end_timestamp": pa.array(ets, type=pa.int64()),
        "vertex_timestamps": pa.array(vts, type=pa.list_(pa.int64())),
        "speed": pa.array(speed, type=pa.float64()),
        "region": pa.array(region, type=pa.string()),
    }
    if with_values:
        cols["vertex_values"] = pa.array(vvals, type=pa.list_(pa.float32()))
    tbl = pa.table(cols)
    pq.write_table(tbl, out_path, compression="snappy")
    print(f"Wrote {tbl.num_rows} streamlines → {out_path}"
          f"{'' if with_values else ' (no per-vertex values)'}")
    return tbl.num_rows


def run_stt_build(parquet: Path, stt_dir: Path, stt_build: str, max_zoom: int, bucket: str):
    cmd = [
        stt_build, "--input", str(parquet), "--output", str(stt_dir),
        "--time-field", "timestamp", "--end-time-field", "end_timestamp",
        "--time-format", "unix-ms", "--min-zoom", "0", "--max-zoom", str(max_zoom),
        "--temporal-bucket", bucket, "--blob-ordering", "time-major",
        "--compression", "zstd", "--simplify",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="ISO-8601 UTC")
    ap.add_argument("--end", required=True, help="ISO-8601 UTC")
    ap.add_argument("--bounds", default=",".join(str(x) for x in CONUS))
    ap.add_argument("--grid-step", type=float, default=0.08, help="regrid target step (deg)")
    ap.add_argument("--native-stride", type=int, default=3, help="subsample native HRRR points")
    # 500 mb = the storm STEERING flow — streamlines visually track MRMS cell
    # motion in the composite; the 10 m surface wind runs slower and rotated
    # relative to the storms (it reads as "out of sync").
    ap.add_argument("--level", default="500mb",
                    help="wind level: '10m' (surface file) or '<N>mb' (pressure file)")
    # Defaults balance render perf against ribbon smoothness on the
    # continental demo: resident GPU load scales with
    # particles × (window / step-hours), and the original 24k-particle /
    # 1h-step 10 m build left the solo demo GPU-bound (~33fps). The vertex
    # cadence is the SMOOTHNESS knob: 500 mb flow moves 100-180 km/h, so
    # 30-min segments still read faceted at continental zoom — 15-min vertices
    # (~30-45 km, ~10 px) curve properly, paid for by trimming the particle
    # count.
    ap.add_argument("--particles", type=int, default=10000)
    ap.add_argument("--step-hours", type=float, default=0.25, help="output vertex cadence (h)")
    ap.add_argument("--substeps", type=int, default=4, help="RK4 substeps per output step")
    ap.add_argument("--lifetime-hours", type=float, default=8.0,
                    help="particle lifetime before retire+respawn")
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--no-values", action="store_true",
                    help="omit the per-vertex value column — leaner archive for "
                         "a constant-color head-dot particle field")
    ap.add_argument("--color-by", choices=["speed", "temp"], default="speed",
                    help="per-vertex scalar written to vertex_values: 'speed' "
                         "(wind speed m/s, the streamline gradient) or 'temp' "
                         "(air temperature °C at --level, sampled from the SAME "
                         "HRRR file — colors the drift-dot field by temperature)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path, default=Path("data/hrrr"))
    ap.add_argument("--out", type=Path, required=True, help="output .stt (or .parquet)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-zoom", type=int, default=6)
    # 2h, NOT the 1h field cadence: streamlines render with a 90-min trail, so
    # the loader keeps a ~3h window resident either way — 2h buckets halve the
    # tile/sublayer count (and per-bucket fragment splits) for the same
    # resident geometry.
    ap.add_argument("--temporal-bucket", default="2h")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = parse_when(args.start), parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    if args.color_by == "temp" and args.no_values:
        sys.exit("--color-by temp needs the per-vertex column; drop --no-values.")
    bounds = parse_bounds(args.bounds)
    rng = np.random.default_rng(args.seed)

    want_temp = args.color_by == "temp"
    times_ms, lats, lons, U, V, T = load_hrrr_wind(
        start, end, bounds, args.grid_step, args.cache, args.workers,
        args.skip_fetch, args.native_stride, args.level, want_temp=want_temp)
    tracks = advect(times_ms, lats, lons, U, V, args, rng, bounds, T=T)

    # Parquet intermediate lives in the cache dir, not next to the served archive.
    if args.out.suffix == ".parquet":
        pq_path = args.out
    else:
        args.cache.mkdir(parents=True, exist_ok=True)
        pq_path = args.cache / (args.out.name + ".parquet")
    write_geoparquet(tracks, pq_path, with_values=not args.no_values,
                     color_by=args.color_by)
    if args.skip_build or args.out.suffix == ".parquet":
        print(f"\nGeoParquet only. Build with stt-build --simplify --temporal-bucket "
              f"{args.temporal_bucket} --blob-ordering time-major.")
        return 0
    run_stt_build(pq_path, args.out.with_suffix(""), args.stt_build,
                  args.max_zoom, args.temporal_bucket)
    return 0


if __name__ == "__main__":
    main()
