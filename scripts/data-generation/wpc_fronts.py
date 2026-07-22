#!/usr/bin/env python3
"""Surface frontal analysis (cold/warm/occluded/stationary fronts + troughs)
from the NWS Weather Prediction Center coded surface bulletins.

The synoptic member of the weather suite. Source: WPC "Coded Surface Bulletin"
(CODSUS) — every 3 hours a meteorologist-analyzed snapshot of surface fronts,
troughs, and high/low pressure centers. The hi-res variant (WMO id ASUS02) has
0.1° precision. Bulletins are fetched from the Iowa Environmental Mesonet AFOS
archive (the WPC itself only keeps a rolling 2-week window):

    https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=CODSUS&...

Front lines code as `KIND [STRENGTH] p1 p2 …` with 7-digit points (LLLNNNN =
lat, west-lon in tenths); HIGHS/LOWS code as `pressure position` pairs. From
one bulletin per 3-hour analysis this bakes:

    <out>            front polylines (categorical `front_type`+`render_class`);
                     troughs pre-DASHED (the renderer has no path-dash prop)
    <out>-pips       the classic frontal notation — filled triangles (cold),
                     semicircles (warm), alternating both (occluded/stationary)
                     — as small ORIENTED GEOGRAPHIC POLYGONS riding the lines,
                     so they rotate and scale with the map like the paper map
    <out>-centers    H/L pressure centers (points; opt-in via --centers)

Pips sit on the ADVANCING side of the front. The bulletin does not encode it,
so it is derived: each front is matched to its counterpart in the next (else
previous) analysis by nearest-vertex distance and the mean displacement is
projected onto the front's normal; unmatched fronts (and stationary fronts,
which do not advance) fall back to right-of-walk. Stationary fronts are split
into alternating chunks — blue chunk/triangle one side, red chunk/semicircle
the other — so the classic red/blue alternation is a plain categorical color
mapping downstream.

Each feature is valid [analysis, next analysis + fade pad] so the renderer can
CROSS-DISSOLVE consecutive analyses: with the layer's fadeIn/fadeOutDuration
equal to the pad, the outgoing front ramps down exactly while its successor
ramps up (constant total alpha — no pop, no pulse). The analyst's 0.1° control
points are Catmull-Rom smoothed (originals kept). Bulletin text caches under
<cache>/ (resumable). Use --skip-build to stop at the parquet, --skip-fetch to
reuse the cache, or --stt-build /path to point at the binary.
"""

from __future__ import annotations

import argparse
import bisect
import math
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
CADENCE = timedelta(hours=3)  # WPC surface analysis interval
FRONT_KINDS = {"COLD", "WARM", "STNRY", "OCFNT", "TROF"}
CENTER_KINDS = {"HIGHS": "H", "LOWS": "L"}
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320
# Wider than CONUS: fronts sweeping in from the Pacific/Atlantic/Canada give
# the composite context at its edges. A front is kept if ANY point is inside.
DEFAULT_BOUNDS = (20.0, -135.0, 55.0, -55.0)


def parse_bounds(spec: str) -> tuple[float, float, float, float]:
    p = [float(x) for x in spec.split(",")]
    if len(p) != 4:
        sys.exit("--bounds must be min_lat,min_lon,max_lat,max_lon")
    return p[0], p[1], p[2], p[3]


def parse_when(spec: str) -> datetime:
    s = spec.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── IEM AFOS fetch (per UTC day, resumable) ───────────────────────────────────
def fetch_day(day: datetime, cache: Path, retries: int = 3) -> Path | None:
    path = cache / f"codsus_{day:%Y%m%d}.txt"
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    nxt = day + timedelta(days=1)
    url = (f"{IEM}?pil=CODSUS&sdate={day:%Y-%m-%d}&edate={nxt:%Y-%m-%d}"
           f"&fmt=text&limit=200")
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=120) as r:
                data = r.read()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)
            return path
        except Exception:
            if attempt == retries - 1:
                return None
    return None


# ── bulletin parsing ──────────────────────────────────────────────────────────
def decode_pt(tok: str) -> tuple[float, float]:
    """7-digit hi-res point LLLNNNN → (lon, lat). Lat/lon in tenths, lon west."""
    return -int(tok[3:]) / 10.0, int(tok[:3]) / 10.0


def split_products(text: str, year: int) -> list[tuple[datetime, str]]:
    """Split an AFOS stream into hi-res (ASUS02) bulletins → (valid, body)."""
    out = []
    text = text.replace("\x01", "").replace("\x03", "")
    for part in re.split(r"(?=ASUS0[12] KWBC \d{6})", text):
        hdr = re.match(r"ASUS0(\d) KWBC \d{6}", part)
        m = re.search(r"VALID (\d{2})(\d{2})(\d{2})Z", part)
        if not (hdr and m) or hdr.group(1) != "2":
            continue  # hi-res only: 0.1° vs the whole-degree ASUS01
        mo, dy, hr = (int(g) for g in m.groups())
        out.append((datetime(year, mo, dy, hr, tzinfo=timezone.utc), part))
    return out


def parse_bulletin(body: str):
    """→ (fronts: [(kind, strength, [(lon,lat)…])], centers: [(type, mb|None, lon, lat)])."""
    merged: list[str] = []
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln or re.fullmatch(r"\d{1,3}", ln):
            continue  # blank, or a bare transmission sequence number
        first = ln.split()[0]
        if first in FRONT_KINDS or first in CENTER_KINDS:
            merged.append(ln)
        elif merged and re.fullmatch(r"[\d ]+", ln):
            merged[-1] += " " + ln  # wrapped continuation
    fronts, centers = [], []
    for ln in merged:
        toks = ln.split()
        kind, rest = toks[0], toks[1:]
        if kind in CENTER_KINDS:
            # `pressure position` pairs, but WPC occasionally omits a pressure —
            # walk tokens instead of zipping: 7 digits = position, 3-4 = mb.
            pending = None
            for t in rest:
                if len(t) == 7:
                    centers.append((CENTER_KINDS[kind], pending, *decode_pt(t)))
                    pending = None
                elif len(t) in (3, 4) and 850 <= int(t) <= 1090:
                    pending = int(t)
                else:
                    print(f"    ⚠️  odd center token {t!r} in: {ln[:60]}…")
        else:
            strength = ""
            if rest and rest[0].isalpha():
                strength, rest = rest[0], rest[1:]
            pts = [decode_pt(t) for t in rest if len(t) == 7]
            if len(pts) != len(rest):
                print(f"    ⚠️  dropped {len(rest) - len(pts)} odd token(s) in: {ln[:60]}…")
            # consecutive duplicates appear in real bulletins; they break the
            # centripetal spline (zero-length segment)
            pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
            if len(pts) >= 2:
                fronts.append((kind, strength, pts))
    return fronts, centers


# ── geometry: spline smoothing ────────────────────────────────────────────────
def catmull_rom(pts, subdiv: int):
    """Centripetal Catmull-Rom through all control points (originals kept)."""
    if len(pts) < 3 or subdiv < 2:
        return pts
    ext = [pts[0]] + pts + [pts[-1]]
    out = [pts[0]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        # centripetal knots (alpha = 0.5) — no loops/overshoot on uneven spacing
        t0 = 0.0
        t1 = t0 + math.dist(p0, p1) ** 0.5 or t0 + 1e-6
        t2 = t1 + math.dist(p1, p2) ** 0.5
        t3 = t2 + math.dist(p2, p3) ** 0.5 or t2 + 1e-6
        for s in range(1, subdiv):
            t = t1 + (t2 - t1) * s / subdiv
            a1 = _lerp2(p0, p1, t0, t1, t)
            a2 = _lerp2(p1, p2, t1, t2, t)
            a3 = _lerp2(p2, p3, t2, t3, t)
            b1 = _lerp2(a1, a2, t0, t2, t)
            b2 = _lerp2(a2, a3, t1, t3, t)
            out.append(_lerp2(b1, b2, t1, t2, t))
        out.append(p2)
    return out


def _lerp2(pa_, pb, ta, tb, t):
    if tb == ta:
        return pa_
    w = (t - ta) / (tb - ta)
    return (pa_[0] + (pb[0] - pa_[0]) * w, pa_[1] + (pb[1] - pa_[1]) * w)


def haversine(lon1, lat1, lon2, lat2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ── arc-length walking (shared by chunks, dashes, pip placement) ──────────────
def _cumlen_km(pts):
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + haversine(*a, *b) / 1000.0)
    return cum


def _at(pts, cum, pos_km):
    """Point + unit tangent (local-km frame) at arc position `pos_km`."""
    i = min(bisect.bisect_right(cum, pos_km), len(pts) - 1) - 1
    i = max(0, i)
    seg = cum[i + 1] - cum[i]
    w = 0.0 if seg <= 0 else (pos_km - cum[i]) / seg
    a, b = pts[i], pts[i + 1]
    p = (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)
    cosl = math.cos(math.radians(p[1])) or 1e-6
    tx = (b[0] - a[0]) * KM_PER_DEG_LON_EQ * cosl
    ty = (b[1] - a[1]) * KM_PER_DEG_LAT
    n = math.hypot(tx, ty) or 1e-6
    return p, (tx / n, ty / n)


def alternate_chunks(pts, dash_km: float):
    """Split a stationary front into alternating-length chunks (vertex-snapped);
    downstream colors STNRY_COLD blue / STNRY_WARM red — the classic notation."""
    chunks, cur, acc = [], [pts[0]], 0.0
    for a, b in zip(pts, pts[1:]):
        cur.append(b)
        acc += haversine(*a, *b) / 1000.0
        if acc >= dash_km:
            chunks.append(cur)
            cur, acc = [b], 0.0
    if len(cur) >= 2:
        chunks.append(cur)
    elif chunks:
        chunks[-1] += cur[1:]
    return [("STNRY_COLD" if i % 2 == 0 else "STNRY_WARM", c)
            for i, c in enumerate(chunks)]


def dash_polyline(pts, dash_km: float, gap_km: float):
    """Cut a trough into dash sub-polylines (exact interpolated boundaries).
    Data-side because the path layer exposes no dash-array prop; geographic
    dashes also scale with zoom like the hand-drawn map's."""
    out, cur = [], [pts[0]]
    drawing, remaining = True, dash_km
    for a, b in zip(pts, pts[1:]):
        seg = haversine(*a, *b) / 1000.0
        pos = 0.0
        while seg - pos > remaining + 1e-9:
            frac = (pos + remaining) / seg
            cut = (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)
            if drawing:
                cur.append(cut)
                out.append(cur)
                cur = []
            else:
                cur = [cut]
            pos += remaining
            drawing = not drawing
            remaining = dash_km if drawing else gap_km
        remaining -= seg - pos
        if drawing:
            cur.append(b)
    if drawing and len(cur) >= 2:
        out.append(cur)
    return [c for c in out if len(c) >= 2]


# ── advancing-side derivation + pip construction ──────────────────────────────
def _to_km(pts, cosl):
    return [(lo * KM_PER_DEG_LON_EQ * cosl, la * KM_PER_DEG_LAT) for lo, la in pts]


def _match_stats(fk, ck):
    """(mean nearest-vertex distance, mean displacement along LEFT normal) of
    up to 12 samples of front `fk` against candidate `ck` (both km frames)."""
    n = len(fk)
    idxs = sorted({round(j * (n - 1) / 11) for j in range(12)})
    dists, dots = [], []
    for i in idxs:
        p = fk[i]
        a, b = fk[max(0, i - 1)], fk[min(n - 1, i + 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        tl = math.hypot(tx, ty)
        if tl <= 0:
            continue
        q = min(ck, key=lambda c: (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2)
        dx, dy = q[0] - p[0], q[1] - p[1]
        dists.append(math.hypot(dx, dy))
        dots.append((dx * -ty + dy * tx) / tl)  # dot with left normal
    if not dists:
        return 1e18, 0.0
    return sum(dists) / len(dists), sum(dots) / len(dots)


def advancing_side(pts, kind, nxt_fronts, prv_fronts) -> float:
    """+1 = pips LEFT of walk direction, -1 = right. Matches the front to the
    same-kind front nearest in the next (else previous) analysis and reads the
    sign of its normal displacement; falls back to right-of-walk."""
    cosl = math.cos(math.radians(sum(la for _, la in pts) / len(pts)))
    fk = _to_km(pts, cosl)
    for cands, sign in ((nxt_fronts, 1.0), (prv_fronts, -1.0)):
        best_d, best_disp = 1e18, 0.0
        for k2, _s, c in cands:
            if k2 != kind:
                continue
            d, disp = _match_stats(fk, _to_km(c, cosl))
            if d < best_d:
                best_d, best_disp = d, disp
        # ≤350 km mean offset = same front; ≥15 km normal motion = trustworthy
        if best_d <= 350.0 and abs(best_disp) >= 15.0:
            return (1.0 if best_disp > 0 else -1.0) * sign
    return -1.0


def pip_ring(p, t, side, shape, base_km, height_km):
    """One classic pip as a closed ring: TRIANGLE (cold) or SEMICIRCLE (warm)
    with its base ON the front line, pointing to `side` (+1 left of walk)."""
    lon, lat = p
    cosl = math.cos(math.radians(lat)) or 1e-6
    tx, ty = t
    nx, ny = side * -ty, side * tx  # unit normal toward the pip side
    if shape == "TRIANGLE":
        off = [
            (-base_km / 2 * tx, -base_km / 2 * ty),
            (base_km / 2 * tx, base_km / 2 * ty),
            (height_km * nx, height_km * ny),
        ]
    else:  # SEMICIRCLE — radius = half the triangle base, dome toward n
        r = base_km / 2
        off = [
            (r * math.cos(th) * tx + r * math.sin(th) * nx,
             r * math.cos(th) * ty + r * math.sin(th) * ny)
            for th in (j * math.pi / 8 for j in range(9))
        ]
    return [
        (lon + x / (KM_PER_DEG_LON_EQ * cosl), lat + y / KM_PER_DEG_LAT)
        for x, y in off
    ]


def pip_positions(cum, spacing_km):
    """Centered arc positions, half-spacing margins; one mid-pip if short."""
    total = cum[-1]
    usable = total - spacing_km * 0.9
    if usable <= 0:
        return [total / 2] if total >= spacing_km * 0.4 else []
    n = int(usable // spacing_km) + 1
    start = (total - (n - 1) * spacing_km) / 2
    return [start + j * spacing_km for j in range(n)]


# ── parquet writers ───────────────────────────────────────────────────────────
def write_fronts(rows, out: Path):
    from shapely import wkb as shp_wkb
    from shapely.geometry import LineString

    geom, ts, ets, ftype, rclass, strength = [], [], [], [], [], []
    for t, et, k, rc, st, pts in rows:
        geom.append(shp_wkb.dumps(LineString(pts)))
        ts.append(t)
        ets.append(et)
        ftype.append(k)
        rclass.append(rc)
        strength.append(st)
    tbl = pa.table(
        {
            "geometry": pa.array(geom, type=pa.binary()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "end_timestamp": pa.array(ets, type=pa.int64()),
            "front_type": pa.array(ftype, type=pa.string()),
            "render_class": pa.array(rclass, type=pa.string()),
            "strength": pa.array(strength, type=pa.string()),
        }
    )
    pq.write_table(tbl, out, compression="snappy")
    return tbl.num_rows


def write_pips(rows, out: Path):
    from shapely import wkb as shp_wkb
    from shapely.geometry import Polygon

    geom, ts, ets, ftype, rclass, ptype = [], [], [], [], [], []
    for t, et, k, rc, shape, ring in rows:
        geom.append(shp_wkb.dumps(Polygon(ring)))
        ts.append(t)
        ets.append(et)
        ftype.append(k)
        rclass.append(rc)
        ptype.append(shape)
    tbl = pa.table(
        {
            "geometry": pa.array(geom, type=pa.binary()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "end_timestamp": pa.array(ets, type=pa.int64()),
            "front_type": pa.array(ftype, type=pa.string()),
            "render_class": pa.array(rclass, type=pa.string()),
            "pip_type": pa.array(ptype, type=pa.string()),
        }
    )
    pq.write_table(tbl, out, compression="snappy")
    return tbl.num_rows


def write_centers(rows, out: Path):
    lon, lat, ts, ets, ctype, mb = [], [], [], [], [], []
    for t, et, c, p, clon, clat in rows:
        lon.append(clon)
        lat.append(clat)
        ts.append(t)
        ets.append(et)
        ctype.append(c)
        mb.append(p)
    tbl = pa.table(
        {
            "lon": pa.array(lon, type=pa.float64()),
            "lat": pa.array(lat, type=pa.float64()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "end_timestamp": pa.array(ets, type=pa.int64()),
            "center_type": pa.array(ctype, type=pa.string()),
            "pressure_mb": pa.array(mb, type=pa.int64()),
        }
    )
    pq.write_table(tbl, out, compression="snappy")
    return tbl.num_rows


# ── stt-build ─────────────────────────────────────────────────────────────────
def build(parquet: Path, out_dir: Path, args):
    cmd = [
        args.stt_build,
        "--input", str(parquet),
        "--output", str(out_dir),
        "--time-field", "timestamp",
        "--end-time-field", "end_timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", "0",
        "--max-zoom", str(args.max_zoom),
        # One bucket per analysis: fronts are synoptic, a finer bucket would
        # only shatter tiles (cf. the glm 1m-bucket lesson).
        "--temporal-bucket", "3h",
        "--blob-ordering", "time-major",
        "--compression", "zstd",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", required=True, help="ISO-8601 UTC")
    ap.add_argument("--end", required=True, help="ISO-8601 UTC")
    ap.add_argument("--bounds", default=",".join(str(x) for x in DEFAULT_BOUNDS))
    ap.add_argument("--smooth-subdiv", type=int, default=6,
                    help="spline points per control segment (<2 disables)")
    ap.add_argument("--stnry-dash-km", type=float, default=180.0,
                    help="alternation length for stationary fronts")
    # Pip geometry (km, so pips scale with zoom like the paper map). At the
    # composite's z4.2 home view: 110 km spacing ≈ 28 px, 60 km base ≈ 15 px.
    ap.add_argument("--pip-spacing-km", type=float, default=110.0)
    ap.add_argument("--pip-base-km", type=float, default=60.0)
    ap.add_argument("--pip-height-km", type=float, default=45.0)
    ap.add_argument("--trof-dash-km", type=float, default=90.0)
    ap.add_argument("--trof-gap-km", type=float, default=60.0)
    # Must equal the renderer's fadeIn/fadeOutDuration on the fronts layers for
    # a constant-alpha cross-dissolve (see module docstring).
    ap.add_argument("--fade-pad-min", type=int, default=45)
    ap.add_argument("--centers", action="store_true",
                    help="also build the <out>-centers H/L pressure archive")
    ap.add_argument("--cache", type=Path, default=Path("data/wpc-fronts"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-zoom", type=int, default=6)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    start, end = parse_when(args.start), parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    min_lat, min_lon, max_lat, max_lon = parse_bounds(args.bounds)
    pad_ms = args.fade_pad_min * 60_000

    # One analysis of run-in so the window start already has fronts on screen.
    fetch_from = start - CADENCE - timedelta(hours=3)  # + issuance lag
    days = []
    d = fetch_from.replace(hour=0, minute=0, second=0, microsecond=0)
    while d <= end:
        days.append(d)
        d += timedelta(days=1)

    print(f"WPC CODSUS fronts: {start:%Y-%m-%d %H:%M} … {end:%Y-%m-%d %H:%M}Z")
    if not args.skip_fetch:
        print(f"Fetching {len(days)} day(s) of bulletins into {args.cache} …")
    by_valid: dict[datetime, str] = {}
    for day in days:
        path = fetch_day(day, args.cache) if not args.skip_fetch \
            else args.cache / f"codsus_{day:%Y%m%d}.txt"
        if path is None or not path.exists():
            print(f"    ⚠️  no bulletins for {day:%Y-%m-%d}")
            continue
        # stream is ascending by issuance: later product per valid time wins
        for valid, body in split_products(path.read_text(errors="replace"), day.year):
            if start - CADENCE <= valid <= end:
                by_valid[valid] = body
    if not by_valid:
        sys.exit("No hi-res (ASUS02) bulletins found for that window.")
    valids = sorted(by_valid)
    gaps = [b for a, b in zip(valids, valids[1:]) if b - a > CADENCE]
    if gaps:
        print(f"  ⚠️  {len(gaps)} gap(s) in the 3-hourly sequence, e.g. before {gaps[0]}")

    parsed = {v: parse_bulletin(by_valid[v]) for v in valids}
    front_rows, pip_rows, center_rows = [], [], []
    n_fronts_raw = 0
    for i, valid in enumerate(valids):
        fronts, centers = parsed[valid]
        nxt = parsed[valids[i + 1]][0] if i + 1 < len(valids) else []
        prv = parsed[valids[i - 1]][0] if i > 0 else []
        t_ms = int(valid.timestamp() * 1000)
        et_ms = int((valid + CADENCE).timestamp() * 1000) + pad_ms

        def inside(pts):
            return any(
                min_lat <= la <= max_lat and min_lon <= lo <= max_lon
                for lo, la in pts
            )

        for kind, strength, pts in fronts:
            n_fronts_raw += 1
            if not inside(pts):
                continue
            sm = catmull_rom(pts, args.smooth_subdiv)

            if kind == "TROF":
                for dash in dash_polyline(sm, args.trof_dash_km, args.trof_gap_km):
                    front_rows.append((t_ms, et_ms, kind, "TROF", strength, dash))
                continue

            if kind == "STNRY":
                # Chunk pips pair with the chunk colors: blue chunk → blue
                # triangle right of walk, red chunk → red semicircle on the
                # OPPOSITE side (no advance ⇒ no motion-derived side).
                for rclass, piece in alternate_chunks(sm, args.stnry_dash_km):
                    front_rows.append((t_ms, et_ms, kind, rclass, strength, piece))
                    cum = _cumlen_km(piece)
                    if cum[-1] < args.pip_base_km * 1.2:
                        continue
                    p, t = _at(piece, cum, cum[-1] / 2)
                    shape, side = (
                        ("TRIANGLE", -1.0) if rclass == "STNRY_COLD"
                        else ("SEMICIRCLE", 1.0)
                    )
                    ring = pip_ring(p, t, side, shape,
                                    args.pip_base_km, args.pip_height_km)
                    pip_rows.append((t_ms, et_ms, kind, rclass, shape, ring))
                continue

            front_rows.append((t_ms, et_ms, kind, kind, strength, sm))
            side = advancing_side(pts, kind, nxt, prv)
            cum = _cumlen_km(sm)
            for j, pos in enumerate(pip_positions(cum, args.pip_spacing_km)):
                p, t = _at(sm, cum, pos)
                # cold = all triangles, warm = all semicircles, occluded =
                # alternating both on the same side
                shape = (
                    "SEMICIRCLE"
                    if kind == "WARM" or (kind == "OCFNT" and j % 2 == 1)
                    else "TRIANGLE"
                )
                ring = pip_ring(p, t, side, shape,
                                args.pip_base_km, args.pip_height_km)
                pip_rows.append((t_ms, et_ms, kind, kind, shape, ring))

        for ctype, mb, lon, lat in centers:
            if mb is not None and inside([(lon, lat)]):
                center_rows.append((t_ms, et_ms, ctype, mb, lon, lat))

    print(f"\nBaked {len(front_rows)} front segments + {len(pip_rows)} pips "
          f"({n_fronts_raw} coded lines, {len(valids)} analyses) "
          f"+ {len(center_rows)} pressure centers.")

    args.cache.mkdir(parents=True, exist_ok=True)
    base = args.out
    fronts_pq = args.cache / (base.name + ".parquet")
    pips_pq = args.cache / (base.name + "-pips.parquet")
    write_fronts(front_rows, fronts_pq)
    write_pips(pip_rows, pips_pq)
    centers_pq = args.cache / (base.name + "-centers.parquet")
    if args.centers:
        write_centers(center_rows, centers_pq)
    print(f"Wrote parquet: {fronts_pq.name}, {pips_pq.name}" +
          (f", {centers_pq.name}" if args.centers else ""))

    if args.skip_build:
        print("--skip-build set; stopping at parquet.")
        return 0

    build(fronts_pq, base, args)
    build(pips_pq, base.with_name(base.name + "-pips"), args)
    if args.centers:
        build(centers_pq, base.with_name(base.name + "-centers"), args)
    print(f"\n✅ Built {base.name} / {base.name}-pips" +
          (f" / {base.name}-centers" if args.centers else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
