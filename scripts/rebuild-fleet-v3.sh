#!/usr/bin/env bash
#
# Rebuild the API-sourced half of the showcase fleet at packed formatVersion 3.
#
# WHY A STAGING TREE: archives are content-addressed, so a v3 rebuild shares no
# pack names with its v2 predecessor. Building into `public/data-v3/` leaves the
# live `public/data/` tree untouched and serving, which is what makes the R2
# ordering rule (upload immutable objects → deploy the reader → flip manifests)
# possible at all. Nothing here uploads.
#
# WHY ONLY THESE: every dataset below is rebuilt by `stt-generate` from a public
# API with no staged input, so it is reproducible from a clean checkout. The
# weather / storm / AV families come from the Python generators in
# scripts/data-generation and from staged multi-GB inputs; they are a separate
# pass and are NOT attempted here. Four datasets have no rebuild path at all
# (lines-v2 is synthetic with no recipe; nyc-taxi-od-summary has no generator;
# osm-nyc-nodes needs a login-gated Geofabrik full-history extract) — they stay
# formatVersion 2, which is exactly why the reader kept its v2 read window.
#
# Recipes are the ones recorded in examples/showcase/src/datasets.ts; the
# non-obvious arguments are commented at their call site below.
#
# Usage: scripts/rebuild-fleet-v3.sh [stem ...]     (default: all of them)

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/examples/showcase/public/data-v3"
GEN="$REPO/tools/stt-generate/target/release/stt-generate"
VALIDATE="$REPO/target/release/stt-validate"
LOG="$OUT/_rebuild.log"

mkdir -p "$OUT"
cd "$OUT" || exit 1

say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# One dataset: run the generator, time it, then validate the result. A failure
# is recorded and the run CONTINUES — one dead upstream API must not cost the
# other twelve rebuilds.
run_one() {
  local stem="$1"; shift
  local start elapsed
  start=$(date +%s)
  say "START $stem"
  if ! "$GEN" "$@" >>"$LOG" 2>&1; then
    say "FAIL  $stem (generator exited non-zero)"
    return 1
  fi
  elapsed=$(( $(date +%s) - start ))
  if [ ! -f "$OUT/$stem/manifest.json" ]; then
    say "FAIL  $stem (no manifest at $stem/manifest.json after ${elapsed}s)"
    return 1
  fi
  local ver
  ver=$(python3 -c "import json;print(json.load(open('$OUT/$stem/manifest.json')).get('formatVersion'))" 2>/dev/null)
  if [ "$ver" != "3" ]; then
    say "FAIL  $stem (formatVersion $ver, expected 3)"
    return 1
  fi
  # Structure + decode. Not fatal on its own — a drift NOTE is expected on
  # clipped-trajectory archives (the writer mints ids per clipped segment), so
  # the exit code is reported rather than obeyed.
  "$VALIDATE" "$OUT/$stem/manifest.json" >>"$LOG" 2>&1
  local vexit=$?

  # ACCEPTANCE: compare feature_count against the archive currently serving.
  #
  # This gate exists because the first run of this script produced a wildfires
  # archive with 175 features against the live 4,600 — the NIFC generator now
  # fetches 15 source perimeters where it once fetched hundreds — and BOTH
  # earlier checks passed it: the manifest said formatVersion 3, and
  # stt-validate exited 0 because the archive is perfectly well-formed. It is
  # an honest encoding of almost no data. A version migration must never be
  # able to quietly empty a demo, so the previous archive is the baseline and a
  # material shortfall fails the dataset.
  local old="$REPO/examples/showcase/public/data/$stem/manifest.json"
  if [ -f "$old" ]; then
    local cmp
    cmp=$(python3 - "$old" "$OUT/$stem/manifest.json" <<'PY'
import json,sys
def n(p):
    return (json.load(open(p)).get('metadata') or {}).get('feature_count') or 0
o,w = n(sys.argv[1]), n(sys.argv[2])
ratio = (w/o) if o else 1.0
print(f"{o} {w} {ratio:.3f} {'SHORTFALL' if ratio < 0.95 else 'ok'}")
PY
)
    set -- $cmp
    if [ "$4" = "SHORTFALL" ]; then
      say "FAIL  $stem  ${elapsed}s  features $2 vs live $1 (${3}x) — REBUILD IS NOT A REPLACEMENT"
      return 1
    fi
    say "OK    $stem  ${elapsed}s  validate=$vexit  features $2 vs live $1 (${3}x)"
    return 0
  fi
  say "OK    $stem  ${elapsed}s  validate=$vexit  (no live baseline to compare)"
}

declare -a WANT=("$@")
want() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  local s
  for s in "${WANT[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

say "=== fleet rebuild → $OUT ==="

# Cheapest first, so a broken toolchain shows up in seconds rather than hours.
want wildfires        && run_one wildfires      wildfires
want hurricanes       && run_one hurricanes     hurricanes
# The demo wants raw points, NOT a summary tier, on this small dataset; and the
# stem is earthquakes-v2 while the subcommand is `earthquakes`.
want earthquakes-v2   && run_one earthquakes-v2 earthquakes --output earthquakes-v2.stt
# GBIF season-folds every track onto --ref-year; the demo's timeRange is 2024.
want animals          && run_one animals        animals
# CelesTrak propagates from NOW, so datasets.ts's timeRange needs re-syncing
# after this one — see the entry's comment there.
want satellites       && run_one satellites     satellites
# OpenSky publishes historical days on Mondays 2017-2020 only; the demo config
# is pinned to this date.
want flights          && run_one flights        flights --date 2020-01-06
# The ship-traffic demo is pinned to 2023-01-09; any other date renders blank
# (disjoint time window). ~1.1 GB download.
want ais-all-us       && run_one ais-all-us     ais --date 2023-01-09 --output ais-all-us.stt
# The drifters demo wants the FULL 1979→2022 GDP record. The generator default
# is 2021-only, which leaves the 43-year timeline almost empty and reads as a
# broken demo.
want drifters         && run_one drifters       drifters --start 1979-01-01 --end 2022-11-01 \
                                                --temporal-bucket 7d --max-zoom 4

say "=== done ==="
grep -cE '^\S+ OK ' "$LOG" 2>/dev/null | sed 's/^/ok: /' | tee -a "$LOG"
