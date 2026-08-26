#!/usr/bin/env python3
"""Radiosonde balloon-ascent trail for the Storm-4D Greenfield demo → STT points.

`storm4d-sounding` (§9.1 of docs/roadmap/storm-4d-greenfield-2026-07.md): the
OAX (Valley, NE — WMO 72558) 2024-05-21 **18Z special launch**, fetched from the
University of Wyoming NEW wsgi endpoint and re-emitted as a tiny 3D ascent
trail — one STT point per reported level:

- `timestamp`  = 18:00Z + alt_m / (5 m s⁻¹)  (nominal constant ascent rate)
- position     = drift-integrated from the sounding's own wind profile,
                 starting at OAX (41.32, −96.37) and stepping level-to-level
                 with the layer-mean wind × layer ascent time
- `alt_m`      = reported geopotential height (MSL) — the FE lifts the trail
                 with `elevationProperty:'alt_m'` under the shared
                 STORM4D_ELEVATION_SCALE
- `tmpc` / `dwpc` / `wspd_kt` — temperature, dewpoint (°C), wind speed (kt)

The wsgi endpoint wraps the TEXT:LIST table in HTML; the table itself is
fixed-width (11 × 7-char fields). The 2024+ BUFR-derived listings report wind
speed in m/s under `SPED`; older listings use knots under `SKNT` — the parser
reads the header row and converts either to `wspd_kt`.

Pipeline:  Wyoming wsgi TEXT:LIST  →  parse + drift-integrate  →  GeoParquet
           →  stt-build (points, --temporal-bucket 1m, zoom 3-9)

Full build (exactly the §9.1 archive):

    venv-storm4d/bin/python storm4d_sounding.py \
        --out ../../data-fleet/storm4d-sounding

Use --skip-build to stop at the GeoParquet.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

EARTH_R = 6_371_000.0
M_PER_DEG_LAT = math.pi * EARTH_R / 180.0
MS_TO_KT = 1.0 / 0.514444

STATION_ID = "72558"  # OAX — Valley, NE (NWS Omaha)
LAUNCH_UTC = datetime(2024, 5, 21, 18, 0, tzinfo=timezone.utc)
OAX_LAT, OAX_LON = 41.32, -96.37  # §9.1 drift start point
ASCENT_M_S = 5.0  # nominal balloon ascent rate

WYOMING_URL = (
    "https://weather.uwyo.edu/wsgi/sounding"
    "?datetime={dt}&id={sid}&type=TEXT:LIST"
)


def fetch_sounding(cache: Path, when: datetime, station: str) -> str:
    """Fetch (or reuse cached) Wyoming TEXT:LIST HTML for one launch."""
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"oax_{when:%Y%m%d_%H}z.txt"
    if out.exists() and out.stat().st_size > 0:
        print(f"Using cached {out}")
        return out.read_text()
    url = WYOMING_URL.format(dt=f"{when:%Y-%m-%d}%20{when:%H:%M:%S}", sid=station)
    print(f"Fetching {url}")
    body = urlopen(Request(url), timeout=120).read().decode("utf-8", "replace")
    out.write_text(body)
    return body


def parse_levels(html: str) -> dict[str, np.ndarray]:
    """Parse the first <PRE> fixed-width level table into column arrays.

    Returns arrays keyed by UPPER-CASE header name (PRES, HGHT, TEMP, DWPT,
    DRCT, SPED/SKNT, …); blank 7-char fields become NaN."""
    try:
        pre = html.split("<PRE>", 1)[1].split("</PRE>", 1)[0]
    except IndexError:
        sys.exit("No <PRE> level table in the Wyoming response — launch missing?")
    lines = pre.splitlines()
    header: list[str] = []
    rows: list[list[float]] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("-"):
            continue
        if not header:
            if "PRES" in s and "HGHT" in s:
                header = s.split()
            continue
        if any(c.isalpha() for c in s):
            continue  # units row ("hPa  m  C …") or stray text
        # Fixed-width: 7 chars per field, one field per header column.
        vals: list[float] = []
        for i in range(len(header)):
            cell = ln[i * 7:(i + 1) * 7].strip()
            try:
                vals.append(float(cell))
            except ValueError:
                vals.append(float("nan"))
        if not math.isnan(vals[0]) and not math.isnan(vals[1]):
            rows.append(vals)
    if not rows:
        sys.exit("Parsed 0 levels from the Wyoming table.")
    arr = np.array(rows, dtype="float64")
    return {name: arr[:, i] for i, name in enumerate(header)}


def drift_positions(hght: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Integrate balloon drift from the wind profile: start at OAX, step
    level-to-level with the layer-mean wind × layer ascent time (Δh / 5 m/s).
    Levels missing wind are filled by linear interpolation in height."""
    n = len(hght)
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.any():
        u = np.interp(hght, hght[ok], u[ok])
        v = np.interp(hght, hght[ok], v[ok])
    else:
        u = np.zeros(n)
        v = np.zeros(n)
    lat = np.empty(n)
    lon = np.empty(n)
    lat[0], lon[0] = OAX_LAT, OAX_LON
    for i in range(1, n):
        dt_s = max(0.0, (hght[i] - hght[i - 1]) / ASCENT_M_S)
        um = 0.5 * (u[i] + u[i - 1])
        vm = 0.5 * (v[i] + v[i - 1])
        lat[i] = lat[i - 1] + vm * dt_s / M_PER_DEG_LAT
        coslat = math.cos(math.radians(lat[i - 1]))
        lon[i] = lon[i - 1] + um * dt_s / (M_PER_DEG_LAT * max(coslat, 1e-3))
    return lon, lat


def build_table(cols: dict[str, np.ndarray]):
    import pyarrow as pa

    hght = cols["HGHT"]
    order = np.argsort(hght, kind="stable")  # ascend: bottom → top
    hght = hght[order]

    # Wind: DRCT (met "from" direction) + SPED (m/s, BUFR era) or SKNT (kt).
    drct = cols.get("DRCT", np.full(len(hght), np.nan))[order]
    if "SPED" in cols:
        spd_ms = cols["SPED"][order]
    elif "SKNT" in cols:
        spd_ms = cols["SKNT"][order] * 0.514444
    else:
        spd_ms = np.full(len(hght), np.nan)
    rad = np.radians(drct)
    u = -spd_ms * np.sin(rad)
    v = -spd_ms * np.cos(rad)

    lon, lat = drift_positions(hght, u, v)
    # §9.1: timestamp = 18:00Z + alt / (5 m s⁻¹)
    t0 = int(LAUNCH_UTC.timestamp() * 1000)
    t_ms = t0 + np.round(hght / ASCENT_M_S * 1000.0).astype("int64")

    tmpc = cols.get("TEMP", np.full(len(hght), np.nan))[order]
    dwpc = cols.get("DWPT", np.full(len(hght), np.nan))[order]
    wspd_kt = spd_ms * MS_TO_KT

    table = pa.table(
        {
            "lon": pa.array(lon, type=pa.float64()),
            "lat": pa.array(lat, type=pa.float64()),
            "timestamp": pa.array(t_ms, type=pa.int64()),
            "alt_m": pa.array(hght.astype("float32"), type=pa.float32()),
            "tmpc": pa.array(tmpc.astype("float32"), type=pa.float32()),
            "dwpc": pa.array(dwpc.astype("float32"), type=pa.float32()),
            "wspd_kt": pa.array(wspd_kt.astype("float32"), type=pa.float32()),
        }
    )
    top = hght[-1]
    end = LAUNCH_UTC + timedelta(seconds=float(top) / ASCENT_M_S)
    print(
        f"{table.num_rows} levels: {hght[0]:.0f}→{top:.0f} m MSL, "
        f"ascent {LAUNCH_UTC:%H:%M}Z→{end:%H:%M}Z, "
        f"drift end ({lat[-1]:.2f}, {lon[-1]:.2f})"
    )
    return table


def run_stt_build(parquet: Path, stt_dir: Path, stt_build: str, args) -> None:
    cmd = [
        stt_build,
        "--input", str(parquet),
        "--output", str(stt_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", str(args.min_zoom),
        "--max-zoom", str(args.max_zoom),
        "--temporal-bucket", args.temporal_bucket,
        "--compression", "zstd",
        # Playback safety: explicit time-major (auto may pick spatial → stall).
        "--blob-ordering", "time-major",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {stt_dir}")


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--station", default=STATION_ID)
    ap.add_argument("--cache", type=Path, default=here / "data" / "storm4d" / "sounding")
    ap.add_argument("--out", type=Path, required=True,
                    help="output packed dataset dir (or .parquet to stop there)")
    ap.add_argument("--min-zoom", type=int, default=3)
    ap.add_argument("--max-zoom", type=int, default=9)
    ap.add_argument("--temporal-bucket", default="1m")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--stt-build", default=str(
        here.parents[1] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    import pyarrow.parquet as pq

    html = fetch_sounding(args.cache, LAUNCH_UTC, args.station)
    cols = parse_levels(html)
    table = build_table(cols)

    if args.out.suffix == ".parquet":
        pq_path = args.out
    else:
        args.cache.mkdir(parents=True, exist_ok=True)
        pq_path = args.cache / (args.out.name + ".parquet")
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, pq_path, compression="snappy")
    print(f"Wrote {table.num_rows} points → {pq_path}")

    if args.skip_build or args.out.suffix == ".parquet":
        print("GeoParquet only (--skip-build).")
        return 0
    run_stt_build(pq_path, args.out.with_suffix(""), args.stt_build, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
