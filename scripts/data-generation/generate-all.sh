#!/bin/bash
# Generate the self-contained showcase datasets with stt-generate.
#
# There is NO `stt-generate all` subcommand — the CLI is one subcommand per
# dataset (`stt-generate --help` lists them). This script is the loop that used
# to be claimed by that phantom subcommand: it drives the datasets that need
# nothing but a network connection, one `stt-generate <subcommand>` at a time.
#
# Usage:
#   ./generate-all.sh                     # every self-contained dataset
#   ./generate-all.sh earthquakes flights # just these
#   ./generate-all.sh --list              # names + the command each one runs
#
# For individual datasets or custom options, use stt-generate directly:
#   stt-generate earthquakes --output earthquakes.stt
#   stt-generate ais --date 2024-01-01 --output ais-traffic.stt
#   stt-generate flights --date 2020-01-06 --output flights.stt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-data-fleet}"

# The datasets that run unattended: each downloads its own source data and
# needs no local input file and no local service. `name<TAB>args...` — the args
# are passed to `stt-generate <name>` verbatim.
#
# NOT listed here, because each needs something this script cannot supply
# (see the closing message):
#   bixi (--input), gtfs (--feed/--date), osm-edits (--input),
#   nwm (a local NHDPlus flowlines parquet), nyc-taxi-points (an existing
#   nyc-taxi-paths archive), nyc-rideshare (TLC input + a local OSRM server),
#   drifters-hourly (EXPERIMENTAL, very large).
DATASETS=(
  "earthquakes	--output $OUTPUT_DIR/earthquakes.stt"
  "hurricanes	--output $OUTPUT_DIR/hurricanes.stt"
  "wildfires	--output $OUTPUT_DIR/wildfires.stt"
  "satellites	--output $OUTPUT_DIR/satellites.stt"
  "drifters	--output $OUTPUT_DIR/drifters.stt"
  "animals	--output $OUTPUT_DIR/animals.stt"
  "flights	--date 2020-01-06 --output $OUTPUT_DIR/flights.stt"
  "ais	--date 2024-01-01 --output $OUTPUT_DIR/ais-traffic.stt"
  # storms writes THREE archives (storm-field / storm-cells / storm-tracks)
  # into a directory, so its --output is the directory, not a file.
  "storms	--output $OUTPUT_DIR"
)

dataset_names() {
  local entry
  for entry in "${DATASETS[@]}"; do
    printf '%s\n' "${entry%%	*}"
  done
}

dataset_args() {
  local entry
  for entry in "${DATASETS[@]}"; do
    if [ "${entry%%	*}" = "$1" ]; then
      printf '%s\n' "${entry#*	}"
      return 0
    fi
  done
  return 1
}

if [ "${1:-}" = "--list" ]; then
  echo "Self-contained datasets (stt-generate <subcommand> ...):"
  for entry in "${DATASETS[@]}"; do
    printf '  stt-generate %s %s\n' "${entry%%	*}" "${entry#*	}"
  done
  echo ""
  echo "Every registered subcommand: stt-generate --help"
  exit 0
fi

# Positional args select a subset; unknown names fail loudly rather than
# silently generating nothing.
SELECTED=()
if [ "$#" -gt 0 ]; then
  for name in "$@"; do
    if ! dataset_args "$name" >/dev/null; then
      echo "❌ '$name' is not a self-contained dataset this script drives." >&2
      echo "   Available here: $(dataset_names | tr '\n' ' ')" >&2
      echo "   Other subcommands need local inputs — run them directly:" >&2
      echo "     stt-generate --help" >&2
      exit 2
    fi
    SELECTED+=("$name")
  done
else
  while IFS= read -r name; do SELECTED+=("$name"); done < <(dataset_names)
fi

echo "🚀 STT Data Generation Pipeline"
echo "================================"
echo ""

# stt-build first: stt-generate SHELLS OUT to it, resolving (in order)
# $STT_BUILD_BIN, a sibling of its own exe, then this repo's
# target/release/stt-build. Without it the lookup falls through to PATH and can
# silently drive a stale INSTALLED stt-build.
echo "🔨 Building stt-build (root workspace)..."
cargo build --release -p spatiotemporal-tiles --bin stt-build

# stt-generate is NOT a member of the root workspace — it declares its own
# [workspace] and lockfile under tools/ — so `-p stt-generate` from the root
# does not resolve. It has to be built through its own manifest, which also
# means its binary lands in tools/stt-generate/target, not ./target.
echo "🔨 Building stt-generate (tools/stt-generate workspace)..."
cargo build --release --manifest-path tools/stt-generate/Cargo.toml
STT_GENERATE="$REPO_ROOT/tools/stt-generate/target/release/stt-generate"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "📊 Generating ${#SELECTED[@]} dataset(s) into $OUTPUT_DIR ..."
echo ""

# One dataset failing (a source API down, a rate limit) must not discard the
# ones that already succeeded, so failures are collected and reported at the
# end instead of aborting the run.
FAILED=()
for name in "${SELECTED[@]}"; do
  args="$(dataset_args "$name")"
  echo "── stt-generate $name $args"
  # shellcheck disable=SC2086 # args are a deliberate word-split argument list
  if "$STT_GENERATE" "$name" $args; then
    echo "   ✅ $name"
  else
    echo "   ❌ $name failed" >&2
    FAILED+=("$name")
  fi
  echo ""
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "❌ ${#FAILED[@]} dataset(s) failed: ${FAILED[*]}"
  echo "   Re-run just those: $0 ${FAILED[*]}"
  exit 1
fi

echo "✅ All selected datasets generated successfully!"
echo ""
echo "📁 Output location: $OUTPUT_DIR"
echo ""
echo "Datasets NOT generated here, because each needs a local input or service:"
echo "  stt-generate bixi --input <trips.csv> --output $OUTPUT_DIR/bixi-flowmap"
echo "  stt-generate gtfs --feed <feed.zip> --date <YYYY-MM-DD>"
echo "  stt-generate osm-edits --input <history.osm.pbf>"
echo "  stt-generate nwm --flowlines <nhd-flowlines.parquet>"
echo "  stt-generate nyc-taxi-points --input $OUTPUT_DIR/nyc-taxi-paths"
echo "  stt-generate nyc-rideshare --input <tlc.parquet>   # needs a local OSRM"
echo "  stt-generate drifters-hourly                       # EXPERIMENTAL, large"
echo ""
echo "For per-dataset options:"
echo "  stt-generate --help"
echo "  stt-generate earthquakes --help"
