#!/usr/bin/env bash
# One-off, LEGACY (single-file `.stt`): re-process feasible direct-API showcase
# datasets through the updated pipeline (fresh build → optimize/repack → measure),
# staging via .new and promoting only on success. Logs a results table to stdout.
#
# Superseded by the packed flow: `stt-build` now emits packed datasets directly
# (blobs ordered at pack-cutting time), so the optimize/repack step here applies
# only to single-file archives. Kept for single-file measurement.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA=examples/showcase/public/data
GEN=target/release/stt-generate
SIM=target/release/examples/simulate_layout
cost() { STT_SIM_GAP=524288 STT_SIM_MAXREQ=1024 "$SIM" "$1" 2>/dev/null | sed -n '/=== RANKING/,$p' | grep 'REAL-DISK' | awk '{print $NF}'; }
covline() { "$SIM" "$1" 2>/dev/null | sed -n '2p' | sed 's/^  covering: //'; }

# name | live filename | generator args
RUNS=(
  "hurricanes|hurricanes.stt|hurricanes"
  "wildfires|wildfires.stt|wildfires"
  "animals|animals.stt|animals"
  "flights|flights.stt|flights --date 2020-01-06"
  "drifters|drifters.stt|drifters --start 1979-01-01 --end 2022-11-01 --temporal-bucket 7d --max-zoom 4"
  "ais|ais-all-us.stt|ais --date 2023-01-09"
)

printf "%-12s %-10s %12s %12s %10s %s\n" DATASET RESULT OLD_COST NEW_COST DSIZE COVERING
for r in "${RUNS[@]}"; do
  IFS='|' read -r name live args <<<"$r"
  new="$DATA/${live%.stt}.new.stt"
  old="$DATA/$live"
  echo ">>> [$name] generating: $GEN $args --output $new" >&2
  if ! $GEN $args --output "$new" >/tmp/reprocess-$name.log 2>&1; then
    printf "%-12s %-10s %12s %12s %10s %s\n" "$name" "GEN-FAIL" - - - "(see /tmp/reprocess-$name.log)"
    rm -f "$new"; continue
  fi
  if [[ ! -s "$new" ]]; then
    printf "%-12s %-10s %12s %12s %10s %s\n" "$name" "EMPTY" - - - "-"; rm -f "$new"; continue
  fi
  oldcost="-"; [[ -s "$old" ]] && oldcost=$(cost "$old")
  echo ">>> [$name] optimizing $new" >&2
  scripts/optimize-tiles.sh "$new" >/tmp/reprocess-$name-opt.log 2>&1 || true
  newcost=$(cost "$new")
  cov=$(covline "$new")
  osz=0; [[ -s "$old" ]] && osz=$(wc -c < "$old")
  nsz=$(wc -c < "$new")
  dsz=$(awk "BEGIN{if($osz>0) printf \"%+.1f%%\", 100*($nsz-$osz)/$osz; else print \"new\"}")
  mv -f "$new" "$old"
  printf "%-12s %-10s %12s %12s %10s %s\n" "$name" "OK" "$oldcost" "$newcost" "$dsz" "$cov"
done
echo ">>> reprocess complete" >&2
