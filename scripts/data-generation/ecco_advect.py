#!/usr/bin/env python3
"""Advect virtual particles through ECCO surface currents → STT trajectories.

This is the "Perpetual Ocean" demo generator. It reads NASA ECCO V4r4 ocean
velocity granules (the 0.5° interpolated product, variables ``EVEL`` /
``NVEL``), seeds a cloud of massless particles on the ocean surface, and
integrates each one through the *time-varying* velocity field with a 4th-order
Runge–Kutta scheme. Every particle's path becomes one LineString carrying
per-vertex timestamps and a per-vertex speed value — exactly the shape the
``drifters`` generator emits, so the showcase's trips renderer, time scrubber,
and globe view all work unchanged.

Pipeline:  ECCO NetCDF  →  (this script)  →  GeoParquet  →  stt-build  →  .stt

The GeoParquet schema is the stt-build LineString contract:
  geometry           Binary    WKB LineString (lon, lat)
  timestamp          Int64     track start, Unix-ms        (--time-format unix-ms)
  end_timestamp      Int64     track end, Unix-ms          (--end-time-field)
  vertex_timestamps  List<Int64>   one absolute Unix-ms per vertex
  vertex_values      List<Float32> one current speed (m/s) per vertex
  speed              Float64   mean speed along track (numeric property)
  seed_lat           Float64   seeding latitude            (numeric property)
  basin              Utf8      coarse ocean basin label    (categorical property)

Download the input first (free Earthdata login required), e.g.:

    pip install podaac-data-downloader
    podaac-data-downloader \
      -c ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4 \
      -d ./ecco-vel -sd 2017-01-01T00:00:00Z -ed 2017-12-31T23:59:59Z

then:

    pip install xarray netcdf4 numpy pyarrow shapely
    python ecco_advect.py --input ./ecco-vel \
      --output ../../examples/showcase/public/data/ecco-currents.stt

Use ``--skip-build`` to stop at the GeoParquet (handy when stt-build isn't on
PATH yet), or ``--stt-build /path/to/stt-build`` to point at the binary.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
EARTH_R = 6_371_000.0          # mean Earth radius, metres
M_PER_DEG_LAT = math.pi * EARTH_R / 180.0   # ≈ 111_195 m per degree latitude
MS_PER_DAY = 86_400_000

# Antimeridian: split a polyline when consecutive vertices jump more than this
# in longitude (a wrap, not real motion).
DATELINE_JUMP_DEG = 180.0


# ── ECCO loading ─────────────────────────────────────────────────────────────
def load_ecco_surface(input_path: Path):
    """Open the ECCO velocity granules and return surface EVEL/NVEL stacks.

    Returns ``(times_ms, lats, lons, evel, nvel)`` where ``evel``/``nvel`` are
    ``(T, nlat, nlon)`` float32 arrays in m/s with land/ice as NaN, ``times_ms``
    is an int64 array of Unix-ms, and ``lats``/``lons`` are the ascending grid
    axes (degrees).
    """
    import xarray as xr

    files = sorted(input_path.glob("*.nc")) if input_path.is_dir() else [input_path]
    if not files:
        sys.exit(f"No .nc files found under {input_path}")
    print(f"Opening {len(files)} ECCO granule(s) from {input_path} …")

    # Open each granule eagerly (no dask) and concat along time. The surface
    # velocity for a year of monthly 0.5° fields is small enough to hold in RAM.
    parts = [xr.open_dataset(f) for f in files]
    ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]

    # Variable names: the 0.5° interpolated product carries EVEL (eastward) and
    # NVEL (northward), already rotated to geographic axes. Fall back to the
    # native UVEL/VVEL only if that's all that's present (not rotation-correct,
    # but lets the script run on the llc90 product for a quick look).
    evar = "EVEL" if "EVEL" in ds else "UVEL"
    nvar = "NVEL" if "NVEL" in ds else "VVEL"
    if evar not in ds or nvar not in ds:
        sys.exit(f"Could not find EVEL/NVEL (or UVEL/VVEL) in {list(ds.data_vars)}")
    if evar in ("UVEL", "VVEL"):
        print("  WARNING: using native UVEL/VVEL — directions are on the model "
              "grid, not true east/north. Prefer the *_05DEG_* product.")

    da_e, da_n = ds[evar], ds[nvar]

    # Pick the surface level: the first index of whatever vertical dim exists
    # (Z / k / depth), which for ECCO velocity is ≈ -5 m.
    horiz = {_lat_dim(da_e), _lon_dim(da_e), "time"}
    for d in list(da_e.dims):
        if d not in horiz:
            da_e = da_e.isel({d: 0})
            da_n = da_n.isel({d: 0})

    da_e = da_e.sortby("time")
    da_n = da_n.sortby("time")

    lat_name, lon_name = _lat_dim(da_e), _lon_dim(da_e)
    da_e = da_e.sortby(lat_name).sortby(lon_name)
    da_n = da_n.sortby(lat_name).sortby(lon_name)

    lats = da_e[lat_name].values.astype("float64")
    lons = da_e[lon_name].values.astype("float64")
    times_ms = da_e["time"].values.astype("datetime64[ms]").astype("int64")

    print("  Loading velocity into memory "
          f"(T={len(times_ms)}, grid={len(lats)}×{len(lons)}) …")
    evel = np.ascontiguousarray(da_e.values, dtype="float32")
    nvel = np.ascontiguousarray(da_n.values, dtype="float32")
    ds.close()
    return times_ms, lats, lons, evel, nvel


def _lat_dim(da):
    for c in ("latitude", "lat", "YC", "j"):
        if c in da.dims:
            return c
    sys.exit(f"No latitude dim among {da.dims}")


def _lon_dim(da):
    for c in ("longitude", "lon", "XC", "i"):
        if c in da.dims:
            return c
    sys.exit(f"No longitude dim among {da.dims}")


# ── Vectorised bilinear sampling ─────────────────────────────────────────────
class Field:
    """Bilinear sampler over a regular lat/lon grid with NaN land masking.

    Longitude wraps (the 0.5° grid is global). A sample whose four surrounding
    cells are all NaN (deep inland / fully masked) returns NaN so the caller can
    treat the particle as beached.
    """

    def __init__(self, lats, lons):
        self.lats = lats
        self.lons = lons
        self.lat0 = lats[0]
        self.lon0 = lons[0]
        self.dlat = lats[1] - lats[0]
        self.dlon = lons[1] - lons[0]
        self.nlat = len(lats)
        self.nlon = len(lons)

    def sample(self, fld_e, fld_n, plon, plat):
        """Sample (E, N) at particle lon/lat arrays. NaN where masked."""
        # Fractional grid coordinates.
        fy = (plat - self.lat0) / self.dlat
        fx = (((plon - self.lon0) % (self.nlon * self.dlon)) / self.dlon)

        y0 = np.floor(fy).astype(np.int64)
        x0 = np.floor(fx).astype(np.int64)
        wy = fy - y0
        wx = fx - x0

        y0 = np.clip(y0, 0, self.nlat - 2)
        y1 = y0 + 1
        x0m = x0 % self.nlon
        x1m = (x0 + 1) % self.nlon

        def bilin(fld):
            v00 = fld[y0, x0m]
            v01 = fld[y0, x1m]
            v10 = fld[y1, x0m]
            v11 = fld[y1, x1m]
            # NaN-aware blend: ignore masked corners, renormalise weights. If
            # all four are NaN the result is NaN (beached).
            w00 = (1 - wy) * (1 - wx)
            w01 = (1 - wy) * wx
            w10 = wy * (1 - wx)
            w11 = wy * wx
            stack_v = np.stack([v00, v01, v10, v11])
            stack_w = np.stack([w00, w01, w10, w11])
            mask = ~np.isnan(stack_v)
            stack_w = np.where(mask, stack_w, 0.0)
            wsum = stack_w.sum(axis=0)
            vsum = np.where(mask, np.nan_to_num(stack_v) * stack_w, 0.0).sum(axis=0)
            out = np.full_like(wsum, np.nan, dtype="float64")
            ok = wsum > 1e-9
            out[ok] = vsum[ok] / wsum[ok]
            return out

        return bilin(fld_e), bilin(fld_n)


def time_slabs(times_ms, t_ms, evel, nvel):
    """Return velocity fields linearly interpolated in time at ``t_ms``."""
    if t_ms <= times_ms[0]:
        return evel[0], nvel[0]
    if t_ms >= times_ms[-1]:
        return evel[-1], nvel[-1]
    j = int(np.searchsorted(times_ms, t_ms))   # times_ms[j-1] <= t < times_ms[j]
    t0, t1 = times_ms[j - 1], times_ms[j]
    f = (t_ms - t0) / (t1 - t0)
    e = (1 - f) * evel[j - 1] + f * evel[j]
    n = (1 - f) * nvel[j - 1] + f * nvel[j]
    return e, n


# ── Seeding ──────────────────────────────────────────────────────────────────
def seed_particles(field: Field, evel0, nvel0, n_particles, rng, lat_limit):
    """Sample ocean-only seed positions, area-weighted by cos(lat)."""
    seeds_lon, seeds_lat = [], []
    target = n_particles
    batch = max(target * 4, 4000)
    while len(seeds_lon) < target:
        plat = np.degrees(np.arcsin(rng.uniform(
            math.sin(math.radians(-lat_limit)),
            math.sin(math.radians(lat_limit)),
            size=batch)))
        plon = rng.uniform(field.lons[0], field.lons[0] + field.nlon * field.dlon, size=batch)
        plon = ((plon + 180) % 360) - 180
        e, n = field.sample(evel0, nvel0, plon, plat)
        ocean = ~np.isnan(e) & ~np.isnan(n)
        seeds_lon.extend(plon[ocean].tolist())
        seeds_lat.extend(plat[ocean].tolist())
    return (np.array(seeds_lon[:target], dtype="float64"),
            np.array(seeds_lat[:target], dtype="float64"))


def basin_of(lon, lat):
    """Very coarse ocean-basin label, just for a categorical legend facet."""
    if lat > 60:
        return "Arctic"
    if lat < -45:
        return "Southern"
    if -70 <= lon <= 20:
        return "Atlantic"
    if 20 < lon <= 147:
        return "Indian"
    return "Pacific"


# ── RK4 advection ────────────────────────────────────────────────────────────
def advect(times_ms, lats, lons, evel, nvel, args, rng):
    """Integrate particles and return a list of track dicts."""
    field = Field(lats, lons)

    t_start = int(times_ms[0])
    t_end = int(times_ms[-1])
    step_ms = int(args.step_days * MS_PER_DAY)            # output cadence
    substeps = max(1, int(args.substeps))
    dt_ms = step_ms / substeps                            # integration dt
    dt_s = dt_ms / 1000.0

    print(f"Seeding {args.particles} particles …")
    plon, plat = seed_particles(field, evel[0], nvel[0], args.particles, rng, args.lat_limit)
    n = len(plon)

    # Per-particle accumulating tracks (lon, lat, t_ms, speed).
    track_lon = [[plon[i]] for i in range(n)]
    track_lat = [[plat[i]] for i in range(n)]
    track_t = [[t_start] for i in range(n)]
    track_v = [[np.nan] for i in range(n)]
    seed_lat = plat.copy()
    finished: list[dict] = []

    def deriv(lon, lat, t_ms):
        """d(lon,lat)/dt in deg/s from the (E,N) field at time t_ms."""
        e_fld, n_fld = time_slabs(times_ms, t_ms, evel, nvel)
        e, nn = field.sample(e_fld, n_fld, lon, lat)
        dlat = nn / M_PER_DEG_LAT                                   # deg/s
        coslat = np.cos(np.radians(np.clip(lat, -89.9, 89.9)))
        dlon = e / (M_PER_DEG_LAT * np.maximum(coslat, 1e-3))       # deg/s
        speed = np.sqrt(e * e + nn * nn)
        return dlat, dlon, speed

    n_steps = (t_end - t_start) // step_ms
    print(f"Advecting: {n_steps} output steps × {substeps} substeps "
          f"(dt={dt_s/3600:.1f} h) over {(t_end - t_start)/MS_PER_DAY:.0f} days …")

    # Finite lifetimes + continuous respawning keep a CONSTANT live population so
    # the ocean stays evenly covered. Without this, passive particles physically
    # converge into the subtropical gyres (the "garbage patches") and drain the
    # divergence zones, leaving big empty areas. Each particle lives ~lifetime
    # then retires (emitting its track) and is respawned at a fresh random ocean
    # cell; initial deaths are staggered so retirements spread across time rather
    # than happening in lockstep. Beached particles retire + respawn immediately.
    lifetime_ms = float(args.lifetime_days * MS_PER_DAY)
    death = t_start + rng.uniform(0.15, 1.0, size=n) * lifetime_ms
    ocean_e0, ocean_n0 = evel[0], nvel[0]   # ~constant ocean mask for respawns

    def emit(i):
        if len(track_lon[i]) >= args.min_points:
            finished.append({
                "lon": track_lon[i], "lat": track_lat[i],
                "t": track_t[i], "v": track_v[i],
                "seed_lat": float(seed_lat[i]),
            })

    print(f"Lifetime {args.lifetime_days:.0f} d, respawning to hold {n} live "
          f"particles …")
    t_ms = float(t_start)
    n_emitted = 0
    for s in range(int(n_steps)):
        stuck = np.zeros(n, dtype=bool)   # hit land at any substep this step
        for _ in range(substeps):
            la, lo = plat, plon
            # RK4 in (lon, lat), full population.
            k1la, k1lo, _ = deriv(lo, la, t_ms)
            k2la, k2lo, _ = deriv(lo + 0.5 * dt_s * k1lo, la + 0.5 * dt_s * k1la, t_ms + dt_ms / 2)
            k3la, k3lo, _ = deriv(lo + 0.5 * dt_s * k2lo, la + 0.5 * dt_s * k2la, t_ms + dt_ms / 2)
            k4la, k4lo, sp = deriv(lo + dt_s * k3lo, la + dt_s * k3la, t_ms + dt_ms)

            new_lat = la + (dt_s / 6.0) * (k1la + 2 * k2la + 2 * k3la + k4la)
            new_lon = lo + (dt_s / 6.0) * (k1lo + 2 * k2lo + 2 * k3lo + k4lo)
            new_lon = ((new_lon + 180) % 360) - 180

            beached = np.isnan(k4la) | np.isnan(k4lo) | np.isnan(sp)
            plat = np.where(beached, la, new_lat)
            plon = np.where(beached, lo, new_lon)
            stuck |= beached
            t_ms += dt_ms

        out_t = int(round(t_ms))
        e_fld, n_fld = time_slabs(times_ms, out_t, evel, nvel)
        e_all, n_all = field.sample(e_fld, n_fld, plon, plat)
        sp_all = np.hypot(e_all, n_all)

        retire = stuck | (t_ms >= death) | np.isnan(sp_all)

        # Extend the still-living tracks.
        for i in np.where(~retire)[0]:
            track_lon[i].append(float(plon[i]))
            track_lat[i].append(float(plat[i]))
            track_t[i].append(out_t)
            track_v[i].append(float(sp_all[i]))

        # Retire + respawn the rest (batch the ocean-seed sampling).
        ret = np.where(retire)[0]
        for i in ret:
            emit(i)
        n_emitted += len(ret)
        if len(ret) > 0:
            nlon, nlat = seed_particles(field, ocean_e0, ocean_n0, len(ret), rng, args.lat_limit)
            plon[ret], plat[ret] = nlon, nlat
            death[ret] = t_ms + rng.uniform(0.5, 1.0, size=len(ret)) * lifetime_ms
            seed_lat[ret] = nlat
            for j, i in enumerate(ret):
                track_lon[i] = [float(nlon[j])]
                track_lat[i] = [float(nlat[j])]
                track_t[i] = [out_t]
                track_v[i] = [np.nan]

        if (s + 1) % max(1, n_steps // 10) == 0:
            print(f"  step {s+1}/{n_steps}  live={n}  emitted={n_emitted}")

    # Emit the final in-flight tracks.
    for i in range(n):
        emit(i)
    print(f"Produced {len(finished)} tracks (≥{args.min_points} points).")
    return finished


# ── Antimeridian split + GeoParquet ──────────────────────────────────────────
def split_dateline(track):
    """Split a track into segments wherever it wraps the antimeridian."""
    lon, lat, t, v = track["lon"], track["lat"], track["t"], track["v"]
    segs, cur = [], [0]
    for k in range(1, len(lon)):
        if abs(lon[k] - lon[k - 1]) > DATELINE_JUMP_DEG:
            segs.append(cur)
            cur = [k]
        else:
            cur.append(k)
    segs.append(cur)
    out = []
    for idx in segs:
        if len(idx) < 2:
            continue
        out.append({
            "lon": [lon[k] for k in idx],
            "lat": [lat[k] for k in idx],
            "t": [t[k] for k in idx],
            "v": [v[k] for k in idx],
            "seed_lat": track["seed_lat"],
        })
    return out


def write_geoparquet(tracks, out_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely.geometry import LineString
    from shapely import wkb

    geom, ts, ets, vts, vvals = [], [], [], [], []
    speed_prop, seedlat_prop, basin_prop = [], [], []

    for tr in tracks:
        for seg in split_dateline(tr):
            coords = list(zip(seg["lon"], seg["lat"]))
            geom.append(wkb.dumps(LineString(coords)))
            tv = [int(x) for x in seg["t"]]
            ts.append(tv[0])
            ets.append(tv[-1])
            vts.append(tv)
            speeds = [float(x) if x == x else float("nan") for x in seg["v"]]
            vvals.append([np.float32(x) for x in speeds])
            finite = [x for x in speeds if x == x]
            speed_prop.append(float(np.mean(finite)) if finite else float("nan"))
            seedlat_prop.append(seg["seed_lat"])
            mid = len(coords) // 2
            basin_prop.append(basin_of(coords[mid][0], coords[mid][1]))

    table = pa.table({
        "geometry": pa.array(geom, type=pa.binary()),
        "timestamp": pa.array(ts, type=pa.int64()),
        "end_timestamp": pa.array(ets, type=pa.int64()),
        "vertex_timestamps": pa.array(vts, type=pa.list_(pa.int64())),
        "vertex_values": pa.array(vvals, type=pa.list_(pa.float32())),
        "speed": pa.array(speed_prop, type=pa.float64()),
        "seed_lat": pa.array(seedlat_prop, type=pa.float64()),
        "basin": pa.array(basin_prop, type=pa.string()),
    })
    pq.write_table(table, out_path, compression="snappy")
    print(f"Wrote {table.num_rows} LineStrings → {out_path}")
    return table.num_rows


# ── stt-build ────────────────────────────────────────────────────────────────
def run_stt_build(parquet_path: Path, stt_path: Path, stt_build: str,
                  max_zoom: int, temporal_bucket: str):
    cmd = [
        stt_build,
        "--input", str(parquet_path),
        "--output", str(stt_path),
        "--time-field", "timestamp",
        "--end-time-field", "end_timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", "0",
        "--max-zoom", str(max_zoom),
        "--temporal-bucket", temporal_bucket,
        "--compression", "zstd",
        "--simplify",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {stt_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="ECCO velocity .nc file or a directory of granules")
    p.add_argument("--output", type=Path, default=Path("ecco-currents.stt"),
                   help="output .stt (or .parquet with --skip-build)")
    p.add_argument("--particles", type=int, default=20000,
                   help="number of seed particles (default 20000)")
    p.add_argument("--step-days", type=float, default=5.0,
                   help="output vertex cadence in days (default 5)")
    p.add_argument("--substeps", type=int, default=5,
                   help="RK4 integration substeps per output step (default 5)")
    p.add_argument("--lifetime-days", type=float, default=75.0,
                   help="particle lifetime before retire+respawn (default 75); "
                        "shorter → more even coverage, longer → longer ribbons")
    p.add_argument("--min-points", type=int, default=4,
                   help="drop tracks shorter than this many vertices")
    p.add_argument("--lat-limit", type=float, default=80.0,
                   help="don't seed poleward of ±this latitude (default 80)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible particle clouds")
    p.add_argument("--max-zoom", type=int, default=5)
    p.add_argument("--temporal-bucket", default="5d")
    p.add_argument("--stt-build", default="stt-build",
                   help="path to the stt-build binary")
    p.add_argument("--skip-build", action="store_true",
                   help="stop at GeoParquet (writes <output>.parquet)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    times_ms, lats, lons, evel, nvel = load_ecco_surface(args.input)
    if len(times_ms) < 2:
        sys.exit("Need at least 2 time steps to advect; download more granules.")

    tracks = advect(times_ms, lats, lons, evel, nvel, args, rng)

    if args.skip_build or args.output.suffix == ".parquet":
        pq_path = args.output if args.output.suffix == ".parquet" \
            else args.output.with_suffix(".parquet")
        write_geoparquet(tracks, pq_path)
        if args.skip_build:
            print(f"\nDone (GeoParquet only). Build with:\n"
                  f"  {args.stt_build} --input {pq_path} --output "
                  f"{args.output.with_suffix('.stt')} --time-field timestamp "
                  f"--end-time-field end_timestamp --time-format unix-ms "
                  f"--temporal-bucket {args.temporal_bucket} --simplify")
            return

    pq_path = args.output.with_suffix(".parquet")
    write_geoparquet(tracks, pq_path)
    run_stt_build(pq_path, args.output.with_suffix(".stt"), args.stt_build,
                  args.max_zoom, args.temporal_bucket)


if __name__ == "__main__":
    main()
