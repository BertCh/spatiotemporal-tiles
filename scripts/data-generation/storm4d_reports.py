#!/usr/bin/env python3
"""Local storm reports (LSR) points for the Storm-4D Greenfield demo.

`storm4d-reports` archive (storm-4d-greenfield campaign, contract §9.1):
POINTS, 1-minute bucket, zoom 3-9 — the damage wave arriving *behind* the
storm. Source: IEM LSR GeoJSON (every WFO, then cropped to the scene bbox):

    https://mesonet.agron.iastate.edu/geojson/lsr.geojson?sts=...&ets=...

Contract kind mapping from the LSR `typetext`:

    TORNADO                                   → tornado
    HAIL                                      → hail
    TSTM WND GST | NON-TSTM WND GST
      | MARINE TSTM WIND                      → wind   (gust, converted to kt)
    TSTM WND DMG | NON-TSTM WND DMG           → damage
    FLOOD | FLASH FLOOD                       → flood
    anything else (rain, funnel cloud, …)     → other

`magnitude` is hail size in inches or wind gust in KNOTS (IEM reports gusts in
MPH — converted here), 0 where not applicable. `remark` is clipped to 120 chars.

Cross-check (printed, never merged): SPC filtered storm reports CSV
`spc.noaa.gov/climo/reports/<yymmdd>_rpts_filtered.csv` — the SPC "day" runs
12Z→12Z, so HHMM ≥ 1200 belongs to the CSV's date and HHMM < 1200 to the next
day. Tornado counts differ from the LSR feed by design (SPC filters
duplicates; LSR carries every report).

Output GeoParquet → `stt-build`:

    lon (Float64)   lat (Float64)   timestamp (Int64 unix ms)
    kind (str)   magnitude (Float64)   remark (str ≤120)

Raw responses cached under --cache (idempotent). --skip-build stops at parquet.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

IEM_LSR = "https://mesonet.agron.iastate.edu/geojson/lsr.geojson"
SPC_REPORTS = "https://www.spc.noaa.gov/climo/reports"

# Scene bbox (min_lon, min_lat, max_lon, max_lat) — matches storm4d-warnings.
DEFAULT_BBOX = (-98.0, 39.0, -90.0, 44.0)

MPH_TO_KT = 0.868976

KIND_BY_TYPETEXT = {
    "TORNADO": "tornado",
    "HAIL": "hail",
    "TSTM WND GST": "wind",
    "NON-TSTM WND GST": "wind",
    "MARINE TSTM WIND": "wind",
    "TSTM WND DMG": "damage",
    "NON-TSTM WND DMG": "damage",
    "FLOOD": "flood",
    "FLASH FLOOD": "flood",
}

REMARK_MAX = 120


def parse_when(spec: str) -> datetime:
    """Parse an ISO-8601 UTC instant (with or without trailing Z)."""
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_bbox(spec: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 4:
        sys.exit("--bbox must be min_lon,min_lat,max_lon,max_lat")
    return parts[0], parts[1], parts[2], parts[3]


# ── fetch (cached, atomic) ────────────────────────────────────────────────────
def fetch(url: str, path: Path, what: str) -> Path:
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached {path}")
        return path
    print(f"Fetching {what}: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=300) as r:
        data = r.read()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)
    print(f"  {len(data)} bytes → {path}")
    return path


# ── LSR extraction ────────────────────────────────────────────────────────────
def gust_kt(magf: float, unit: str | None) -> float:
    u = (unit or "").strip().upper()
    if u in ("KT", "KTS", "KNOTS"):
        return magf
    # IEM reports wind gusts in MPH; treat unknown units as MPH too.
    return magf * MPH_TO_KT


def read_lsrs(path: Path, bbox, start: datetime, end: datetime):
    """Return rows (lon, lat, t_ms, kind, magnitude, remark) inside bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    feats = json.load(open(path))["features"]
    rows = []
    for f in feats:
        lon, lat = f["geometry"]["coordinates"][:2]
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            continue
        p = f["properties"]
        t_ms = int(parse_when(p["valid"]).timestamp() * 1000)
        if not (start_ms <= t_ms <= end_ms):
            continue
        typetext = (p.get("typetext") or "").upper()
        kind = KIND_BY_TYPETEXT.get(typetext, "other")
        magf = p.get("magf")
        if magf is None or not isinstance(magf, (int, float)):
            magnitude = 0.0
        elif kind == "wind":
            magnitude = gust_kt(float(magf), p.get("unit"))
        elif kind == "hail":
            magnitude = float(magf)  # inches
        else:
            magnitude = 0.0
        remark = (p.get("remark") or "").strip()
        if len(remark) > REMARK_MAX:
            remark = remark[: REMARK_MAX - 1].rstrip() + "…"
        rows.append((float(lon), float(lat), t_ms, kind, magnitude, remark))
    print(f"  kept {len(rows)} of {len(feats)} LSR(s) inside bbox/window")
    return rows


def build_parquet(rows, out: Path) -> int:
    if not rows:
        sys.exit("No LSRs in window/bbox — nothing to build.")
    rows.sort(key=lambda r: r[2])
    table = pa.table(
        {
            "lon": pa.array([r[0] for r in rows], type=pa.float64()),
            "lat": pa.array([r[1] for r in rows], type=pa.float64()),
            "timestamp": pa.array([r[2] for r in rows], type=pa.int64()),
            "kind": pa.array([r[3] for r in rows], type=pa.string()),
            "magnitude": pa.array([r[4] for r in rows], type=pa.float64()),
            "remark": pa.array([r[5] for r in rows], type=pa.string()),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy")
    t0 = datetime.fromtimestamp(rows[0][2] / 1000, tz=timezone.utc)
    t1 = datetime.fromtimestamp(rows[-1][2] / 1000, tz=timezone.utc)
    print(f"\nWrote {table.num_rows} report(s) → {out}")
    print(f"  by kind: {dict(Counter(r[3] for r in rows))}")
    print(f"  time span: {t0:%Y-%m-%d %H:%M} … {t1:%Y-%m-%d %H:%M} UTC")
    return table.num_rows


# ── SPC cross-check (report-only; duplicates are NOT merged) ──────────────────
def spc_cross_check(rows, bbox, start: datetime, end: datetime, cache: Path) -> None:
    """Compare tornado LSR count against the SPC filtered-reports CSV."""
    # The SPC day runs 12Z→12Z; anchor on the date containing `start` 12Z.
    anchor = (start - timedelta(hours=12)).date()
    url = f"{SPC_REPORTS}/{anchor:%y%m%d}_rpts_filtered.csv"
    try:
        path = fetch(url, cache / f"{anchor:%y%m%d}_rpts_filtered.csv", "SPC reports")
        text = path.read_text(errors="replace")
    except Exception as e:  # cross-check only — never fail the build
        print(f"SPC cross-check skipped ({e})")
        return
    min_lon, min_lat, max_lon, max_lat = bbox
    section = None
    spc_tor = 0
    for line in text.splitlines():
        if line.startswith("Time,"):
            section = "tornado" if "F_Scale" in line else "other"
            continue
        if section != "tornado" or not line.strip():
            continue
        parts = line.split(",", 7)  # comments may contain commas — split last
        if len(parts) < 7:
            continue
        try:
            hhmm = int(parts[0])
            lat, lon = float(parts[5]), float(parts[6])
        except ValueError:
            continue
        day = anchor if hhmm >= 1200 else anchor + timedelta(days=1)
        t = datetime(day.year, day.month, day.day, hhmm // 100, hhmm % 100,
                     tzinfo=timezone.utc)
        if not (start <= t <= end):
            continue
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            spc_tor += 1
    lsr_tor = sum(1 for r in rows if r[3] == "tornado")
    print(f"\nSPC cross-check (window ∩ bbox): {spc_tor} SPC filtered tornado "
          f"report(s) vs {lsr_tor} tornado LSR(s).")
    if spc_tor != lsr_tor:
        print("  Counts differ by design (SPC filters duplicate reports; the LSR "
              "feed carries every report). Not merged.")


# ── stt-build ─────────────────────────────────────────────────────────────────
def run_stt_build(parquet: Path, stt_dir: Path, args) -> None:
    cmd = [
        args.stt_build,
        "--input", str(parquet),
        "--output", str(stt_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", str(args.min_zoom),
        "--max-zoom", str(args.max_zoom),
        "--temporal-bucket", args.temporal_bucket,
        "--compression", "zstd",
        # Playback-safe ordering (blob-ordering gotcha: `auto` may pick spatial).
        "--blob-ordering", "time-major",
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
    ap.add_argument("--start", default="2024-05-21T17:30Z",
                    help="window start, ISO-8601 UTC (contract §9.0 default)")
    ap.add_argument("--end", default="2024-05-22T03:00Z",
                    help="window end, ISO-8601 UTC")
    ap.add_argument("--bbox", default=",".join(str(x) for x in DEFAULT_BBOX),
                    help="scene crop min_lon,min_lat,max_lon,max_lat")
    ap.add_argument("--cache", type=Path, default=Path("data/storm4d/reports"))
    ap.add_argument("--out", type=Path, required=True,
                    help="output .stt directory (or .parquet with --skip-build)")
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

    lsr_url = f"{IEM_LSR}?sts={start:%Y-%m-%dT%H:%M}Z&ets={end:%Y-%m-%dT%H:%M}Z"
    lsr_path = fetch(
        lsr_url,
        args.cache / f"lsr_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.geojson",
        "IEM LSRs",
    )
    print(f"Reading LSRs from {lsr_path.name} …")
    rows = read_lsrs(lsr_path, bbox, start, end)

    pq_path = args.out if args.out.suffix == ".parquet" else args.out.with_suffix(".parquet")
    build_parquet(rows, pq_path)
    spc_cross_check(rows, bbox, start, end, args.cache)

    if args.skip_build or args.out.suffix == ".parquet":
        print(f"\nGeoParquet only. Build with:\n"
              f"  {args.stt_build} --input {pq_path} --output {args.out.with_suffix('')} "
              f"--time-field timestamp --time-format unix-ms "
              f"--min-zoom {args.min_zoom} --max-zoom {args.max_zoom} "
              f"--temporal-bucket {args.temporal_bucket} "
              f"--blob-ordering time-major --quantize-attrs-auto")
        return 0

    run_stt_build(pq_path, args.out.with_suffix(""), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
