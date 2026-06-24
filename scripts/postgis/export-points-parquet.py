#!/usr/bin/env python3
"""Convert a CSV dump of hurricane_obs into a Parquet file stt-build can tile.

stt-build reads the `lon`/`lat` columns directly (no WKB / geo metadata
required), so this produces the file-ingest baseline that is byte-for-byte the
same point set as the PostGIS source — giving a clean file-vs-PostGIS A/B.

Usage: export-points-parquet.py <input.csv> <output.parquet>
The CSV is expected to have a header with columns:
  lon, lat, iso_time (YYYY-MM-DDTHH:MM:SS, UTC), sid, season, basin, name,
  nature, wmo_wind, wmo_pres, usa_sshs
(the full hurricane_obs non-geometry column set, so the file-vs-PostGIS ingest
A/B compares an identical column set).
"""
import csv
import datetime as dt
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def fnum(x):
    return float(x) if x not in ("", None) else None


def inum(x):
    return int(x) if x not in ("", None) else None


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    lon, lat, ts, sid, season, basin, name, nature, wind, pres, sshs = (
        [] for _ in range(11)
    )
    with open(src, newline="") as fh:
        for row in csv.DictReader(fh):
            lon.append(fnum(row["lon"]))
            lat.append(fnum(row["lat"]))
            ts.append(dt.datetime.fromisoformat(row["iso_time"]))
            sid.append(row["sid"] or None)
            season.append(inum(row.get("season")))
            basin.append(row.get("basin") or None)
            name.append(row["name"] or None)
            nature.append(row.get("nature") or None)
            wind.append(fnum(row["wmo_wind"]))
            pres.append(fnum(row["wmo_pres"]))
            sshs.append(inum(row["usa_sshs"]))

    table = pa.table(
        {
            "lon": pa.array(lon, pa.float64()),
            "lat": pa.array(lat, pa.float64()),
            "iso_time": pa.array(ts, pa.timestamp("ms")),
            "sid": pa.array(sid, pa.string()),
            "season": pa.array(season, pa.int32()),
            "basin": pa.array(basin, pa.string()),
            "name": pa.array(name, pa.string()),
            "nature": pa.array(nature, pa.string()),
            "wmo_wind": pa.array(wind, pa.float64()),
            "wmo_pres": pa.array(pres, pa.float64()),
            "usa_sshs": pa.array(sshs, pa.int32()),
        }
    )
    pq.write_table(table, dst)
    print(f"wrote {table.num_rows} rows -> {dst}")


if __name__ == "__main__":
    main()
