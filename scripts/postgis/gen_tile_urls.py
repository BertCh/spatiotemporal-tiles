#!/usr/bin/env python3
"""Generate tile request paths (/tiles/{z}/{x}/{y}/{t}.stt) from a CSV sample of
points, so the serve benchmark hits tiles that actually contain data.

For each sampled (lon, lat, ms) row, emit the tile path at every requested zoom,
with `t` snapped into the temporal bucket. Deduplicated.

Usage: gen_tile_urls.py <sample.csv> <bucket_ms> <zoom1,zoom2,...>
CSV columns: lon, lat, ms
"""
import csv
import math
import sys


def tile_xy(lon: float, lat: float, z: int):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    x = min(max(x, 0), n - 1)
    y = min(max(y, 0), n - 1)
    return x, y


def main() -> None:
    sample, bucket_ms, zooms = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    zooms = [int(z) for z in zooms.split(",")]
    seen = set()
    out = []
    with open(sample, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                lon, lat, ms = float(row["lon"]), float(row["lat"]), int(row["ms"])
            except (KeyError, ValueError):
                continue
            t = ms // bucket_ms * bucket_ms
            for z in zooms:
                x, y = tile_xy(lon, lat, z)
                key = (z, x, y, t)
                if key in seen:
                    continue
                seen.add(key)
                out.append(f"/tiles/{z}/{x}/{y}/{t}.stt")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
