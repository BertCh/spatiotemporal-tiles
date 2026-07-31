#!/usr/bin/env python3
"""County power outages for the Storm-4D Greenfield demo (EAGLE-I + Census).

Binding contract: docs/roadmap/storm-4d-greenfield-2026-07.md §9.1 `storm4d-outages`
— POLYGONS, bucket 15m, zoom 3-8; one feature per (county, 15-min interval) where
customers_out > 0; columns `timestamp`, `end_timestamp` (+15 min), `county` str,
`fips` str, `customers_out` i32. The FE extrudes prisms by `customers_out`.

Sources (both cached, downloads are idempotent):
- ORNL EAGLE-I 2024 county outage CSV (CC-BY 4.0), 1.44 GB, direct figshare:
      https://ndownloader.figshare.com/files/53581661
  Columns: fips_code,county,state,customers_out,run_start_time,total_customers —
  one row per (county, 15-min collection run), run_start_time in **UTC** (verified
  against the Greenfield timeline: Adair Co. IA outages jump right after the
  ~20:26-20:32Z tornado hit). The full-file scan is slow, so the state+window
  subset is cached as a parquet next to the CSV and reused on re-runs.
- Census 2023 cartographic county boundaries (20m simplified), pyshp-parsed from
      https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip
  matched on 5-digit FIPS (GEOID). MultiPolygon counties are split into
  individual Polygons so they tile cleanly (same rule as mrms/precip isobands).

Knobs (defaults = the ratified demo):
- raw filter window --start/--end   2024-05-21T12:00Z → 2024-05-22T12:00Z (the
  on-disk cached subset; wide so the demo window can be re-trimmed for free)
- emit window --emit-start/--emit-end  17:30Z → 03:00Z (§9.0 demo window; rows
  are kept when run_start_time ∈ [emit-start, emit-end), so the last feature
  ends exactly at the demo end)
- --states IA,MO and --bbox -98,39,-90,44 (counties whose geometry intersects)

Use --skip-build to stop at the parquet; --stt-build to point at the binary.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapefile  # pyshp
from shapely import to_wkb
from shapely.geometry import MultiPolygon, Polygon, box, shape

EAGLEI_URL = "https://ndownloader.figshare.com/files/53581661"
EAGLEI_NAME = "eaglei_outages_2024.csv"
COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip"

# CONUS. EAGLE-I names states in full; the Census cartographic-boundary file
# keys them by FIPS. Both tables are needed because the outage feed and the
# county geometry are joined across that gap.
STATE_NAMES = {
    "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "IA": "Iowa",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts",
    "MD": "Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota",
    "MO": "Missouri", "MS": "Mississippi", "MT": "Montana",
    "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NV": "Nevada", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VA": "Virginia", "VT": "Vermont",
    "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia",
    "WY": "Wyoming",
}
STATE_FIPS = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41",
    "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48",
    "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}

FIFTEEN_MIN_MS = 15 * 60 * 1000


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


# ── downloads (idempotent, atomic, Range-resumable) ───────────────────────────
def fetch(url: str, dest: Path, label: str) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"{label}: using cached {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = Request(url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
        print(f"{label}: resuming {url} at {offset / 1e6:.0f} MB → {dest} …")
    else:
        print(f"{label}: downloading {url} → {dest} …")
    with urlopen(req, timeout=120) as r:
        # 206 = server honored the Range; anything else restarts from scratch.
        mode = "ab" if offset and getattr(r, "status", 200) == 206 else "wb"
        with open(part, mode) as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    part.rename(dest)
    print(f"{label}: fetched {dest.stat().st_size / 1e6:.0f} MB")
    return dest


# ── EAGLE-I subset (chunked scan of the 1.44 GB CSV, cached as parquet) ───────
def load_subset(csv_path: Path, states: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    tag = f"{start:%Y%m%dT%H%M}-{end:%Y%m%dT%H%M}_{'-'.join(sorted(states))}"
    cache = csv_path.with_name(f"eaglei_subset_{tag}.parquet")
    if cache.exists():
        print(f"Subset: using cached {cache}")
        return pd.read_parquet(cache)

    names = {STATE_NAMES[s] for s in states}
    # Lexicographic compare works on the CSV's "YYYY-MM-DD HH:MM:SS" strings.
    lo, hi = f"{start:%Y-%m-%d %H:%M:%S}", f"{end:%Y-%m-%d %H:%M:%S}"
    print(f"Subset: scanning {csv_path.name} for {sorted(names)} in [{lo}, {hi}) …")
    kept: list[pd.DataFrame] = []
    rows = 0
    for chunk in pd.read_csv(
        csv_path, dtype={"fips_code": str, "customers_out": "Int64"}, chunksize=2_000_000
    ):
        rows += len(chunk)
        m = (
            chunk["state"].isin(names)
            & (chunk["run_start_time"] >= lo)
            & (chunk["run_start_time"] < hi)
        )
        if m.any():
            kept.append(chunk.loc[m])
        print(f"  scanned {rows / 1e6:.0f} M rows, kept {sum(len(k) for k in kept)}")
    if not kept:
        sys.exit("No EAGLE-I rows matched the states/window filter.")
    df = pd.concat(kept, ignore_index=True)
    df["fips_code"] = df["fips_code"].str.zfill(5)
    # The file should be unique per (county, run); guard against exact dupes.
    df = df.drop_duplicates(subset=["fips_code", "run_start_time"], ignore_index=True)
    df.to_parquet(cache, index=False)
    print(f"Subset: {len(df)} rows → cached {cache}")
    return df


# ── county polygons (Census cartographic 20m, pyshp from the zip) ─────────────
def load_counties(
    zip_path: Path, states: list[str], bbox: tuple[float, float, float, float]
) -> dict[str, tuple[str, list[bytes]]]:
    """Return fips → (name, [polygon-part WKBs]) for counties intersecting bbox."""
    state_fps = {STATE_FIPS[s] for s in states}
    clip = box(*bbox)
    out: dict[str, tuple[str, list[bytes]]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        stem = next(n[:-4] for n in zf.namelist() if n.endswith(".shp"))
        rdr = shapefile.Reader(
            shp=BytesIO(zf.read(stem + ".shp")),
            shx=BytesIO(zf.read(stem + ".shx")),
            dbf=BytesIO(zf.read(stem + ".dbf")),
        )
        fields = [f[0] for f in rdr.fields[1:]]
        i_geoid, i_name, i_statefp = (fields.index(k) for k in ("GEOID", "NAME", "STATEFP"))
        for sr in rdr.iterShapeRecords():
            if sr.record[i_statefp] not in state_fps:
                continue
            geom = shape(sr.shape.__geo_interface__)
            if not geom.intersects(clip):
                continue
            parts: list[Polygon]
            if isinstance(geom, MultiPolygon):
                parts = list(geom.geoms)
            elif isinstance(geom, Polygon):
                parts = [geom]
            else:
                continue
            out[sr.record[i_geoid]] = (sr.record[i_name], [to_wkb(p) for p in parts])
    print(f"Counties: {len(out)} in {'+'.join(states)} intersect bbox {bbox}")
    return out


# ── feature emission ──────────────────────────────────────────────────────────
def emit(
    df: pd.DataFrame,
    counties: dict[str, tuple[str, list[bytes]]],
    emit_start: datetime,
    emit_end: datetime,
    out_parquet: Path,
) -> int:
    lo, hi = f"{emit_start:%Y-%m-%d %H:%M:%S}", f"{emit_end:%Y-%m-%d %H:%M:%S}"
    m = (
        (df["run_start_time"] >= lo)
        & (df["run_start_time"] < hi)
        & (df["customers_out"].fillna(0) > 0)
        & df["fips_code"].isin(counties)
    )
    sel = df.loc[m].copy()
    dropped = df.loc[
        (df["run_start_time"] >= lo) & (df["run_start_time"] < hi) & ~df["fips_code"].isin(counties)
    ]
    if len(dropped):
        skipped = sorted(dropped["fips_code"].unique())
        print(f"  note: {len(skipped)} FIPS outside bbox/boundary file skipped: {skipped}")
    if sel.empty:
        sys.exit("No (county, interval) rows with customers_out > 0 in the emit window.")
    t = pd.to_datetime(sel["run_start_time"], utc=True)
    # pandas >=3 infers datetime64[us]/[s] resolution from the strings, so a raw
    # astype("int64") is NOT nanoseconds — pin the unit to ms explicitly.
    sel["timestamp"] = t.dt.as_unit("ms").astype("int64")
    sel = sel.sort_values(["timestamp", "fips_code"], kind="stable")

    geom, ts, ets, county, fips, cust = [], [], [], [], [], []
    for row in sel.itertuples(index=False):
        name, parts = counties[row.fips_code]
        for wkb in parts:
            geom.append(wkb)
            ts.append(int(row.timestamp))
            ets.append(int(row.timestamp) + FIFTEEN_MIN_MS)
            county.append(name)
            fips.append(row.fips_code)
            cust.append(int(row.customers_out))
    tbl = pa.table(
        {
            "geometry": pa.array(geom, type=pa.binary()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "end_timestamp": pa.array(ets, type=pa.int64()),
            "county": pa.array(county, type=pa.string()),
            "fips": pa.array(fips, type=pa.string()),
            "customers_out": pa.array(cust, type=pa.int32()),
        }
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out_parquet, compression="snappy")
    print(
        f"Emitted {tbl.num_rows} polygon features "
        f"({len(sel)} county-intervals, {sel['fips_code'].nunique()} counties) → {out_parquet}"
    )

    # Sanity: Adair County IA (19001) should light up right after the ~20:26-20:32Z hit.
    adair = sel.loc[sel["fips_code"] == "19001"]
    if len(adair):
        first = adair["run_start_time"].min()
        peak = int(adair["customers_out"].max())
        print(f"  sanity 19001 Adair IA: first outage interval {first}Z, peak {peak} customers")
    else:
        print("  sanity 19001 Adair IA: NO outage rows in emit window (UNEXPECTED)")
    # Peak statewide customers_out (sum across counties per 15-min interval).
    sel["state_abbr"] = sel["fips_code"].str[:2].map({v: k for k, v in STATE_FIPS.items()})
    per_t = sel.groupby(["state_abbr", "run_start_time"])["customers_out"].sum()
    for st, series in per_t.groupby(level=0):
        t_peak = series.idxmax()[1]
        print(f"  peak statewide {st}: {int(series.max())} customers out at {t_peak}Z")
    combined = sel.groupby("run_start_time")["customers_out"].sum()
    print(f"  peak combined: {int(combined.max())} customers out at {combined.idxmax()}Z")
    return tbl.num_rows


# ── stt-build ─────────────────────────────────────────────────────────────────
def run_stt_build(parquet: Path, out_dir: Path, args) -> None:
    cmd = [
        args.stt_build,
        "--input", str(parquet),
        "--output", str(out_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        "--end-time-field", "end_timestamp",
        "--min-zoom", str(args.min_zoom),
        "--max-zoom", str(args.max_zoom),
        "--temporal-bucket", args.temporal_bucket,
        "--compression", "zstd",
        # Multi-cell time PLAYBACK requires time-major ordering (blob-ordering gotcha).
        "--blob-ordering", "time-major",
    ]
    if args.publish:
        cmd += ["--publish"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", default="2024-05-21T12:00Z",
                    help="raw filter window start, ISO-8601 UTC (cached subset)")
    ap.add_argument("--end", default="2024-05-22T12:00Z", help="raw filter window end")
    ap.add_argument("--emit-start", default="2024-05-21T17:30Z",
                    help="demo emit window start (§9.0)")
    ap.add_argument("--emit-end", default="2024-05-22T03:00Z", help="demo emit window end")
    ap.add_argument("--states", default="IA,MO", help="comma list of state abbrs")
    ap.add_argument("--bbox", default="-98,39,-90,44",
                    help="min_lon,min_lat,max_lon,max_lat county-intersect filter")
    ap.add_argument("--cache", type=Path, default=Path("data/storm4d/eaglei"))
    ap.add_argument("--county-zip", type=Path,
                    default=Path("data/storm4d/census/cb_2023_us_county_20m.zip"))
    ap.add_argument("--out", type=Path, required=True,
                    help="output .stt directory (or .parquet with --skip-build)")
    ap.add_argument("--min-zoom", type=int, default=3)
    ap.add_argument("--max-zoom", type=int, default=8)
    ap.add_argument("--temporal-bucket", default="15m")
    ap.add_argument("--publish", action="store_true", help="zstd-19 deploy build")
    ap.add_argument("--skip-build", action="store_true", help="stop at parquet")
    ap.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target" / "release" / "stt-build"))
    args = ap.parse_args()

    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    unknown = [s for s in states if s not in STATE_NAMES]
    if unknown:
        sys.exit(f"Unsupported state(s) {unknown} — extend STATE_NAMES/STATE_FIPS.")
    start, end = parse_when(args.start), parse_when(args.end)
    emit_start, emit_end = parse_when(args.emit_start), parse_when(args.emit_end)
    if not (start <= emit_start < emit_end <= end):
        sys.exit("emit window must sit inside the raw filter window")

    csv_path = fetch(EAGLEI_URL, args.cache / EAGLEI_NAME, "EAGLE-I")
    zip_path = fetch(COUNTY_URL, args.county_zip, "Census counties")

    df = load_subset(csv_path, states, start, end)
    counties = load_counties(zip_path, states, parse_bbox(args.bbox))

    if args.out.suffix == ".parquet":
        pq_path = args.out
    else:
        # Intermediates stay in the raw-download cache, never in public/data (§9.0).
        pq_path = args.cache / f"{args.out.name}.parquet"
    n = emit(df, counties, emit_start, emit_end, pq_path)
    if n == 0:
        sys.exit("Nothing to build.")

    if args.skip_build or args.out.suffix == ".parquet":
        print(f"\nGeoParquet only. Build with:\n"
              f"  {args.stt_build} --input {pq_path} --output {args.out.with_suffix('')} "
              f"--time-field timestamp --time-format unix-ms "
              f"--end-time-field end_timestamp --min-zoom {args.min_zoom} "
              f"--max-zoom {args.max_zoom} --temporal-bucket {args.temporal_bucket} "
              f"--blob-ordering time-major --compression zstd")
        return 0

    run_stt_build(pq_path, args.out.with_suffix(""), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
