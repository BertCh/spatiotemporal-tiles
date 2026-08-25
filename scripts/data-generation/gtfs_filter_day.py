#!/usr/bin/env python3
"""Filter a GTFS feed down to the trips active on one service date.

Usage: gtfs_filter_day.py <feed-dir> <out-dir> <YYYYMMDD>

Purpose: shrink a whole-timetable-year national feed to the single service day
a showcase demo renders, so pfaedle (GTFS→OSM shape map-matching, used by the
gtfs-ch rebuild — see datasets.ts) works on ~1/8th the rows. The gtfs-ch feed
goes 2.2 GB → ~250 MB and pfaedle finishes in minutes.

Writes a valid, much smaller feed: trips/stop_times are filtered to the active
service set, calendar/calendar_dates restricted to it (so `stt-generate gtfs
--date` still resolves services), stops/routes/agency/feed_info copied
verbatim. transfers.txt is deliberately NOT copied — its trip-to-trip rows
reference trips outside the day and pfaedle validates referential integrity
(neither pfaedle's matching nor stt-generate needs transfers). frequencies.txt
is dropped too (on-demand templates; both consumers ignore them).

Handles the Swiss feed's quirks: UTF-8 BOM, CRLF, quoted CSV.
"""
import csv
import datetime as _dt
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
DATE = sys.argv[3]  # YYYYMMDD
WEEKDAY_COLS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

d = _dt.datetime.strptime(DATE, "%Y%m%d").date()
weekday_col = WEEKDAY_COLS[d.weekday()]
DST.mkdir(parents=True, exist_ok=True)


def rows(name):
    p = SRC / name
    if not p.exists():
        return None, None
    f = open(p, newline="", encoding="utf-8-sig")
    r = csv.reader(f)
    header = [h.strip() for h in next(r)]
    return header, r


# 1. Active service ids: calendar weekday window, then calendar_dates exceptions.
active = set()
h, r = rows("calendar.txt")
kept_calendar = []
if h:
    i_sid, i_start, i_end, i_day = (
        h.index("service_id"),
        h.index("start_date"),
        h.index("end_date"),
        h.index(weekday_col),
    )
    for rec in r:
        if rec[i_start] <= DATE <= rec[i_end] and rec[i_day] == "1":
            active.add(rec[i_sid])
            kept_calendar.append(rec)

h2, r2 = rows("calendar_dates.txt")
kept_caldates = []
if h2:
    i_sid, i_date, i_ex = (
        h2.index("service_id"),
        h2.index("date"),
        h2.index("exception_type"),
    )
    for rec in r2:
        if rec[i_date] != DATE:
            continue
        kept_caldates.append(rec)
        if rec[i_ex] == "1":
            active.add(rec[i_sid])
        elif rec[i_ex] == "2":
            active.discard(rec[i_sid])
print(f"active services on {DATE}: {len(active)}", flush=True)

# 2. Trips on those services.
h, r = rows("trips.txt")
i_sid, i_tid = h.index("service_id"), h.index("trip_id")
trip_ids = set()
with open(DST / "trips.txt", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(h)
    n = 0
    for rec in r:
        if rec[i_sid] in active:
            trip_ids.add(rec[i_tid])
            w.writerow(rec)
            n += 1
print(f"trips kept: {n}", flush=True)

# 3. stop_times for those trips (streamed; the multi-GB file).
h, r = rows("stop_times.txt")
i_tid = h.index("trip_id")
with open(DST / "stop_times.txt", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(h)
    n = 0
    for rec in r:
        if rec[i_tid] in trip_ids:
            w.writerow(rec)
            n += 1
print(f"stop_times kept: {n}", flush=True)

# 4. Calendar files restricted to the active set (so --date still resolves).
with open(DST / "calendar.txt", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    hh, _ = rows("calendar.txt")
    w.writerow(hh)
    for rec in kept_calendar:
        w.writerow(rec)
with open(DST / "calendar_dates.txt", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    hh, _ = rows("calendar_dates.txt")
    w.writerow(hh)
    i_sid = hh.index("service_id")
    for rec in kept_caldates:
        if rec[i_sid] in active:
            w.writerow(rec)

# 5. Verbatim copies. transfers.txt and frequencies.txt deliberately absent —
#    see the module docstring.
for name in ["stops.txt", "routes.txt", "agency.txt", "feed_info.txt"]:
    if (SRC / name).exists():
        shutil.copy(SRC / name, DST / name)
print("done", flush=True)
