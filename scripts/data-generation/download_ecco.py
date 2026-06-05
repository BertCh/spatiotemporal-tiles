#!/usr/bin/env python3
"""Download ECCO V4r4 ocean-velocity granules from NASA PO.DAAC via earthaccess.

Auth: needs a free Earthdata login. Either create ~/.netrc with a
`machine urs.earthdata.nasa.gov` entry, or run once interactively:

    ./venv-dl/bin/python -c "import earthaccess; earthaccess.login(persist=True)"

(the password prompt is hidden and the credentials are written to ~/.netrc).

Then:

    ./venv-dl/bin/python download_ecco.py \
      --short-name ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4 \
      --start 2017-01-01 --end 2017-12-31 --dest ./ecco-vel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import earthaccess


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--short-name", default="ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4",
                   help="PO.DAAC collection short name "
                        "(…_MONTHLY_V4R4 small/smooth, …_DAILY_V4R4 full-res)")
    p.add_argument("--start", default="2017-01-01", help="YYYY-MM-DD inclusive")
    p.add_argument("--end", default="2017-12-31", help="YYYY-MM-DD inclusive")
    p.add_argument("--dest", type=Path, default=Path("./ecco-vel"))
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args()

    print("Authenticating with Earthdata …")
    auth = earthaccess.login()  # tries env vars, then ~/.netrc
    if not auth or not getattr(auth, "authenticated", False):
        sys.exit("Earthdata auth failed. Create ~/.netrc or run:\n"
                 "  python -c \"import earthaccess; earthaccess.login(persist=True)\"")

    print(f"Searching {args.short_name}  {args.start} → {args.end} …")
    results = earthaccess.search_data(
        short_name=args.short_name,
        temporal=(args.start, args.end),
    )
    print(f"  {len(results)} granule(s) found.")
    if not results:
        sys.exit("No granules — check the short name and date range.")

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {args.dest} …")
    files = earthaccess.download(results, str(args.dest), threads=args.threads)
    print(f"Downloaded {len(files)} file(s).")
    ncs = sorted(args.dest.glob("*.nc"))
    print(f"{len(ncs)} .nc file(s) now in {args.dest}.")


if __name__ == "__main__":
    main()
