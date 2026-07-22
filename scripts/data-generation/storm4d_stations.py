#!/usr/bin/env python3
"""Surface-station wind/gust points for the Storm-4D Greenfield demo.

`storm4d-stations` (storm-4d-greenfield contract §9.1): one POINT per
(station, observation) across the 2024-05-21 Greenfield EF4 window, driving the
showcase station layer (radius by `gust_kt`, colors by `gust_band`).

Sources (IEM, all anonymous HTTPS):

  1. Station inventory — per-network GeoJSON
       https://mesonet.agron.iastate.edu/geojson/network/<NET>.geojson
     for IA_ASOS + border networks (NE_ASOS, MO_ASOS, IA_AWOS), filtered to the
     contract bbox [-98, 39, -90, 44] (min_lon, min_lat, max_lon, max_lat).
  2. 1-minute ASOS — `cgi-bin/request/asos1min.py` (multi-station, sample=1min,
     gis=yes). Only a minority of sites are in the 1-min network, so we request
     ALL inventory stations (chunked) and keep whichever return data.
  3. METAR fallback — `cgi-bin/request/asos.py` at native cadence (5/20-min
     AWOS + hourly METAR + specials, report_type 3+4) for every inventory
     station that returned NO 1-minute data.

Emitted columns (contract §9.1, byte-for-byte band labels):

    lon (f64)  lat (f64)  timestamp (i64 unix-ms)  station (str)
    wind_kt (f32)  gust_kt (f32)  drct_deg (f32)  tmpf (f32)  gust_band (str)

Gust semantics — `gust_kt` is the MAX of the reported gust and the sustained
speed for the observation (1-min `gust_sknt` vs `sknt`; METAR `gust` vs
`sknt`): when no gust is reported the sustained wind stands in, so `gust_kt`
and `gust_band` are defined for every emitted row. `wind_kt` is the sustained
speed alone and is left null when the sensor dropped it (nulls become an
omitted property per-feature in stt-build — no sentinel-0 coercion). Rows with
neither a sustained speed nor a gust are dropped (no wind information at all);
missing `drct_deg` / `tmpf` are nulls. `pres1` is requested from the 1-min
service (kept in the raw cache for later use) but NOT emitted — the §9.1
schema carries no pressure column.

gust_band (kt): "calm" (<15), "breezy" [15,25), "windy" [25,35),
"severe" [35,50), "extreme" (>=50).

Knobs: --start/--end (window), --bbox, --networks, --chunk (stations per IEM
request). Raw responses cache under data/storm4d/stations/ keyed by
station-set + window (idempotent: re-runs skip files already on disk; a
header-only CSV is a legitimate "no data" result and is cached too).

Build (contract): stt-build --temporal-bucket 1m, zoom 3-9, time-major
blob ordering. Use --skip-build to stop at the GeoParquet.

Sanity anchor (verified live 2026-07-22): DSM reports a 31 kt gust at
2024-05-21 20:04Z — the script prints the DSM 20:00-20:10Z max gust so a
regen that loses it is loud.

Full build:
    venv-storm4d/bin/python storm4d_stations.py \
        --out ../../examples/showcase/public/data/storm4d-stations --publish
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IEM = "https://mesonet.agron.iastate.edu"
NETWORKS = ["IA_ASOS", "NE_ASOS", "MO_ASOS", "IA_AWOS"]
# Contract bbox (min_lon, min_lat, max_lon, max_lat) — Iowa + border sites.
BBOX = (-98.0, 39.0, -90.0, 44.0)
WINDOW_START = "2024-05-21T17:30Z"
WINDOW_END = "2024-05-22T03:00Z"
ONE_MIN_VARS = ["tmpf", "sknt", "drct", "gust_sknt", "pres1"]
METAR_VARS = ["tmpf", "drct", "sknt", "gust"]

# Contract §9.1 gust_band edges (kt) — labels must match FE colorMapping keys
# byte-for-byte.
GUST_BANDS = [
    (15.0, "calm"),
    (25.0, "breezy"),
    (35.0, "windy"),
    (50.0, "severe"),
    (float("inf"), "extreme"),
]


def gust_band(gust_kt: float) -> str:
    for edge, label in GUST_BANDS:
        if gust_kt < edge:
            return label
    return GUST_BANDS[-1][1]


def parse_when(spec: str) -> datetime:
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_bbox(spec: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 4:
        sys.exit("--bbox must be min_lon,min_lat,max_lon,max_lat")
    return parts[0], parts[1], parts[2], parts[3]


def to_float(raw: str | None) -> float | None:
    """IEM CSV value → float, treating '', 'M' (missing) and 'T' (trace) as null."""
    if raw is None:
        return None
    s = raw.strip()
    if s in ("", "M", "T", "None"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


# ── HTTP (anonymous, retried with backoff, cached atomically) ─────────────────
def http_get(url: str, retries: int = 6, timeout: int = 180) -> bytes:
    """GET with exponential backoff — the IEM request CGIs 429 under load."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(60.0, 5.0 * 2**attempt))
                continue
            break  # 4xx other than rate-limit won't heal on retry
        except Exception as e:  # noqa: BLE001 — retry any transport error
            last = e
            time.sleep(2.0)
    raise RuntimeError(f"GET failed after {retries} tries: {url}\n  last error: {last}")


def fetch_cached(url: str, path: Path, validate=None) -> tuple[bytes, bool]:
    """Idempotent download: reuse `path` if present, else GET → validate → cache.
    Returns (body, fetched_over_network) so callers can pace live requests."""
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes(), False
    body = http_get(url)
    if validate is not None:
        validate(body, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.rename(path)
    return body, True


def validate_csv(body: bytes, url: str) -> None:
    """IEM request CGIs return a `station,...` CSV header on success; anything
    else (HTML error page, "ERROR: ..." text) must not poison the cache."""
    head = body[:200].decode("utf-8", "replace").lstrip()
    if not head.lower().startswith("station"):
        raise RuntimeError(f"unexpected response (not station CSV) from {url}:\n  {head[:120]!r}")


def chunk_key(stations: list[str], start: datetime, end: datetime) -> str:
    """Stable cache key for one multi-station request (station set + window)."""
    blob = ",".join(sorted(stations)) + f"|{start:%Y%m%d%H%M}|{end:%Y%m%d%H%M}"
    return hashlib.md5(blob.encode()).hexdigest()[:10]


# ── station inventory ─────────────────────────────────────────────────────────
def load_inventory(networks: list[str], bbox, cache: Path) -> dict[str, dict]:
    """station id → {lon, lat, network, name} for inventory sites inside bbox."""
    import json

    min_lon, min_lat, max_lon, max_lat = bbox
    stations: dict[str, dict] = {}
    for net in networks:
        url = f"{IEM}/geojson/network/{net}.geojson"
        try:
            body, _ = fetch_cached(url, cache / f"network_{net}.geojson")
            feats = json.loads(body).get("features", [])
        except Exception as e:  # noqa: BLE001 — a missing border net is non-fatal
            print(f"  WARN network {net}: {e}")
            continue
        kept = 0
        for f in feats:
            coords = (f.get("geometry") or {}).get("coordinates")
            sid = (f.get("properties") or {}).get("sid") or f.get("id")
            if not coords or not sid:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                continue
            if sid in stations:
                continue  # first network wins (IA_ASOS listed first)
            stations[sid] = {
                "lon": lon,
                "lat": lat,
                "network": net,
                "name": (f.get("properties") or {}).get("sname", ""),
            }
            kept += 1
        print(f"  {net}: {len(feats)} site(s), {kept} inside bbox")
    return stations


# ── 1-minute ASOS ─────────────────────────────────────────────────────────────
def fetch_asos1min(stations: list[str], start: datetime, end: datetime, cache: Path) -> str:
    params = [("station", s) for s in stations]
    params += [("vars", v) for v in ONE_MIN_VARS]
    params += [
        ("sts", f"{start:%Y-%m-%dT%H:%M}Z"),
        ("ets", f"{end:%Y-%m-%dT%H:%M}Z"),
        ("sample", "1min"),
        ("what", "download"),
        ("tz", "UTC"),
        ("delim", "comma"),
        ("gis", "yes"),
    ]
    url = f"{IEM}/cgi-bin/request/asos1min.py?{urlencode(params)}"
    path = cache / f"asos1min_{chunk_key(stations, start, end)}.csv"
    body, fetched = fetch_cached(url, path, validate_csv)
    if fetched:
        time.sleep(1.0)  # polite pacing — the request CGIs 429 when hammered
    return body.decode("utf-8", "replace")


def parse_obs_time(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_asos1min(text: str, start: datetime, end: datetime) -> list[dict]:
    """CSV → observation dicts. Header: station,station_name,lat,lon,
    valid(UTC),tmpf,sknt,drct,gust_sknt,pres1."""
    rows: list[dict] = []
    for rec in csv.DictReader(io.StringIO(text)):
        t = parse_obs_time(rec.get("valid(UTC)") or rec.get("valid") or "")
        if t is None or not (start <= t <= end):
            continue
        rows.append(
            {
                "station": (rec.get("station") or "").strip(),
                "t": t,
                "lon": to_float(rec.get("lon")),
                "lat": to_float(rec.get("lat")),
                "sknt": to_float(rec.get("sknt")),
                "gust": to_float(rec.get("gust_sknt")),
                "drct": to_float(rec.get("drct")),
                "tmpf": to_float(rec.get("tmpf")),
            }
        )
    return rows


# ── METAR fallback (native cadence) ───────────────────────────────────────────
def fetch_metar(stations: list[str], start: datetime, end: datetime, cache: Path) -> str:
    params = [("station", s) for s in stations]
    params += [("data", v) for v in METAR_VARS]
    params += [
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("hour1", start.hour), ("minute1", start.minute),
        ("year2", end.year), ("month2", end.month), ("day2", end.day),
        ("hour2", end.hour), ("minute2", end.minute),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("missing", "empty"),
        ("trace", "empty"),
        ("direct", "no"),
        # routine (3) + special (4) reports — native METAR cadence
        ("report_type", 3), ("report_type", 4),
    ]
    url = f"{IEM}/cgi-bin/request/asos.py?{urlencode(params)}"
    path = cache / f"metar_{chunk_key(stations, start, end)}.csv"
    body, fetched = fetch_cached(url, path, validate_csv)
    if fetched:
        time.sleep(1.0)  # polite pacing — asos.py 429s under parallel load
    return body.decode("utf-8", "replace")


def parse_metar(text: str, start: datetime, end: datetime) -> list[dict]:
    """CSV → observation dicts. Header: station,valid,lon,lat,tmpf,drct,sknt,gust."""
    rows: list[dict] = []
    for rec in csv.DictReader(io.StringIO(text)):
        t = parse_obs_time(rec.get("valid") or "")
        if t is None or not (start <= t <= end):
            continue
        rows.append(
            {
                "station": (rec.get("station") or "").strip(),
                "t": t,
                "lon": to_float(rec.get("lon")),
                "lat": to_float(rec.get("lat")),
                "sknt": to_float(rec.get("sknt")),
                "gust": to_float(rec.get("gust")),
                "drct": to_float(rec.get("drct")),
                "tmpf": to_float(rec.get("tmpf")),
            }
        )
    return rows


def chunked(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


# ── assemble + parquet ────────────────────────────────────────────────────────
def assemble(raw: list[dict], inventory: dict[str, dict], bbox) -> list[dict]:
    """Filter/derive per-observation rows: require some wind info, compute
    gust_kt = max(sustained, gust) and its contract band, dedupe, sort."""
    min_lon, min_lat, max_lon, max_lat = bbox
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for r in raw:
        sid = r["station"]
        if not sid:
            continue
        lon, lat = r["lon"], r["lat"]
        if lon is None or lat is None:
            inv = inventory.get(sid)
            if inv is None:
                continue
            lon, lat = inv["lon"], inv["lat"]
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            continue
        sknt, gust = r["sknt"], r["gust"]
        # Negative speeds are sensor garbage; a row with no wind at all carries
        # nothing for this layer (radius = gust) — drop it.
        if sknt is not None and sknt < 0:
            sknt = None
        if gust is not None and gust < 0:
            gust = None
        if sknt is None and gust is None:
            continue
        gust_kt = max(v for v in (sknt, gust) if v is not None)
        t_ms = int(r["t"].timestamp() * 1000)
        key = (sid, t_ms)
        if key in seen:
            continue  # e.g. METAR routine+special duplicate at the same valid
        seen.add(key)
        drct = r["drct"]
        if drct is not None and not (0.0 <= drct <= 360.0):
            drct = None
        out.append(
            {
                "lon": lon,
                "lat": lat,
                "timestamp": t_ms,
                "station": sid,
                "wind_kt": sknt,
                "gust_kt": gust_kt,
                "drct_deg": drct,
                "tmpf": r["tmpf"],
                "gust_band": gust_band(gust_kt),
            }
        )
    out.sort(key=lambda r: (r["timestamp"], r["station"]))
    return out


def write_parquet(rows: list[dict], out: Path) -> None:
    table = pa.table(
        {
            "lon": pa.array([r["lon"] for r in rows], type=pa.float64()),
            "lat": pa.array([r["lat"] for r in rows], type=pa.float64()),
            "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.int64()),
            "station": pa.array([r["station"] for r in rows], type=pa.string()),
            "wind_kt": pa.array([r["wind_kt"] for r in rows], type=pa.float32()),
            "gust_kt": pa.array([r["gust_kt"] for r in rows], type=pa.float32()),
            "drct_deg": pa.array([r["drct_deg"] for r in rows], type=pa.float32()),
            "tmpf": pa.array([r["tmpf"] for r in rows], type=pa.float32()),
            "gust_band": pa.array([r["gust_band"] for r in rows], type=pa.string()),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy")
    t0 = datetime.fromtimestamp(rows[0]["timestamp"] / 1000, tz=timezone.utc)
    t1 = datetime.fromtimestamp(rows[-1]["timestamp"] / 1000, tz=timezone.utc)
    print(f"\nWrote {table.num_rows} observations → {out}")
    print(f"  time span: {t0:%Y-%m-%d %H:%M} … {t1:%Y-%m-%d %H:%M} UTC")


def print_summary(rows: list[dict], one_min_ids: set[str]) -> None:
    by_station: dict[str, int] = {}
    band_counts: dict[str, int] = {}
    for r in rows:
        by_station[r["station"]] = by_station.get(r["station"], 0) + 1
        band_counts[r["gust_band"]] = band_counts.get(r["gust_band"], 0) + 1
    print(f"  stations emitted: {len(by_station)} "
          f"({len(one_min_ids)} at 1-min cadence, "
          f"{len(by_station) - len(one_min_ids)} METAR fallback)")
    print(f"  gust_band counts: "
          + ", ".join(f"{k}={band_counts.get(k, 0)}" for _, k in GUST_BANDS))
    top = sorted(rows, key=lambda r: -r["gust_kt"])[:5]
    print("  top gusts: " + ", ".join(
        f"{r['station']} {r['gust_kt']:.0f}kt@" +
        datetime.fromtimestamp(r["timestamp"] / 1000, tz=timezone.utc).strftime("%H:%MZ")
        for r in top))
    # Sanity anchor: DSM 31 kt gust near 20:04Z (verified live 2026-07-22).
    lo = int(datetime(2024, 5, 21, 20, 0, tzinfo=timezone.utc).timestamp() * 1000)
    hi = int(datetime(2024, 5, 21, 20, 10, tzinfo=timezone.utc).timestamp() * 1000)
    dsm = [r["gust_kt"] for r in rows if r["station"] == "DSM" and lo <= r["timestamp"] <= hi]
    if dsm:
        print(f"  sanity DSM 20:00-20:10Z max gust: {max(dsm):.0f} kt (expect ~31)")
    elif any(r["station"] == "DSM" for r in rows):
        print("  sanity: DSM present but no rows in 20:00-20:10Z (window subset?)")
    else:
        print("  sanity: WARNING — no DSM rows at all")


# ── stt-build ─────────────────────────────────────────────────────────────────
def run_stt_build(parquet: Path, stt_dir: Path, args) -> None:
    cmd = [
        args.stt_build,
        "--input", str(parquet),
        "--output", str(stt_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        # Contract §9.1: zoom 3-9, 1-minute buckets.
        "--min-zoom", str(args.min_zoom),
        "--max-zoom", str(args.max_zoom),
        "--temporal-bucket", args.temporal_bucket,
        "--compression", "zstd",
        # Multi-cell time PLAYBACK requires time-major ordering — `auto` can
        # pick spatial and silently stall the playhead (blob-ordering gotcha).
        "--blob-ordering", "time-major",
        "--quantize-coords", "100",
        "--quantize-attrs-auto",
    ]
    if args.publish:
        cmd += ["--publish"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {stt_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", default=WINDOW_START,
                    help=f"window start, ISO-8601 UTC (default {WINDOW_START})")
    ap.add_argument("--end", default=WINDOW_END,
                    help=f"window end, ISO-8601 UTC (default {WINDOW_END})")
    ap.add_argument("--bbox", default=",".join(str(x) for x in BBOX),
                    help="min_lon,min_lat,max_lon,max_lat (contract default)")
    ap.add_argument("--networks", default=",".join(NETWORKS),
                    help="comma-separated IEM network ids")
    ap.add_argument("--chunk", type=int, default=20,
                    help="stations per IEM request (URL-length / service-load knob)")
    ap.add_argument("--cache", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "storm4d" / "stations")
    ap.add_argument("--out", type=Path, required=True,
                    help="output packed-archive directory (or .parquet with --skip-build)")
    ap.add_argument("--parquet", type=Path, default=None,
                    help="intermediate GeoParquet path (default: <cache>/<out-stem>.parquet)")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel 1-min IEM requests (higher trips the 429 rate limit)")
    ap.add_argument("--min-zoom", type=int, default=3)
    ap.add_argument("--max-zoom", type=int, default=9)
    ap.add_argument("--temporal-bucket", default="1m")
    ap.add_argument("--publish", action="store_true", help="zstd-19 deploy build")
    ap.add_argument("--skip-build", action="store_true", help="stop at parquet")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = parse_when(args.start), parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    bbox = parse_bbox(args.bbox)
    networks = [n.strip() for n in args.networks.split(",") if n.strip()]
    cache: Path = args.cache

    print(f"Station inventory ({', '.join(networks)}) → bbox {bbox} …")
    inventory = load_inventory(networks, bbox, cache)
    if not inventory:
        sys.exit("No inventory stations inside bbox — check --networks/--bbox.")
    all_ids = sorted(inventory)
    print(f"  {len(all_ids)} candidate station(s)")

    # 1-minute pass: request EVERY inventory station, keep whichever have data.
    print(f"\n1-minute ASOS {start:%Y-%m-%dT%H:%MZ} → {end:%Y-%m-%dT%H:%MZ} …")
    one_min_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for text in ex.map(
            lambda c: fetch_asos1min(c, start, end, cache), chunked(all_ids, args.chunk)
        ):
            one_min_rows.extend(parse_asos1min(text, start, end))
    one_min_ids = {r["station"] for r in one_min_rows}
    print(f"  {len(one_min_rows)} obs from {len(one_min_ids)} 1-min station(s): "
          + ", ".join(sorted(one_min_ids)))

    # METAR fallback for everything that returned no 1-min data. Serial on
    # purpose: asos.py rate-limits (429) parallel requests.
    fallback_ids = [s for s in all_ids if s not in one_min_ids]
    print(f"\nMETAR fallback for {len(fallback_ids)} station(s) …")
    metar_rows: list[dict] = []
    for c in chunked(fallback_ids, args.chunk):
        metar_rows.extend(parse_metar(fetch_metar(c, start, end, cache), start, end))
    print(f"  {len(metar_rows)} obs from "
          f"{len({r['station'] for r in metar_rows})} METAR station(s)")

    rows = assemble(one_min_rows + metar_rows, inventory, bbox)
    if not rows:
        sys.exit("No observations after filtering — nothing to build.")
    print_summary(rows, one_min_ids)

    if args.out.suffix == ".parquet":
        pq_path = args.out
    elif args.parquet is not None:
        pq_path = args.parquet
    else:
        pq_path = cache / f"{args.out.name.removesuffix('.stt')}.parquet"
    write_parquet(rows, pq_path)

    if args.skip_build or args.out.suffix == ".parquet":
        stt_dir = args.out.with_suffix("")
        print(f"\nGeoParquet only. Build with:\n"
              f"  {args.stt_build} --input {pq_path} --output {stt_dir} "
              f"--time-field timestamp --time-format unix-ms "
              f"--min-zoom {args.min_zoom} --max-zoom {args.max_zoom} "
              f"--temporal-bucket {args.temporal_bucket} "
              f"--blob-ordering time-major --quantize-coords 100 --quantize-attrs-auto")
        return 0

    run_stt_build(pq_path, args.out.with_suffix(""), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
