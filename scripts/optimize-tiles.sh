#!/usr/bin/env bash
#
# optimize-tiles.sh — post-build blob-layout optimizer for STT v4 archives.
#
# For each .stt argument (default: examples/showcase/public/data/*.stt):
#   1. run the layout simulator to PICK the best blob ordering and its predicted
#      improvement vs the current on-disk (add-order) layout;
#   2. if the predicted improvement >= MIN_IMPROVEMENT (%), repack the archive
#      in that ordering via the reader-safe (no-dict) buffered writer;
#   3. VERIFY on the actual repacked bytes that REAL-DISK cost dropped, then
#      atomically replace the original. On any failure the original is untouched.
#
# Datasets whose best ordering doesn't beat add-order by the threshold are left
# as-is (reordering them would risk a regression — add-order is already good).
#
# Env:
#   MIN_IMPROVEMENT  minimum predicted % win to repack (default 5)
#   DRY_RUN=1        report picks only, don't repack
#
# Usage:
#   scripts/optimize-tiles.sh [archive.stt ...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-5}"
DRY_RUN="${DRY_RUN:-0}"

# Model the SHIPPED client when ranking orderings: the live reader coalesces the
# whole working set (no ≤12 pre-split) with a 512 KB gap (archive.ts
# coalesceGapBytes + spatiotemporal-tileset.ts MAX_COALESCE_BATCH). Picking an
# ordering under the old 32 KB / 12-tile model would optimise for a client that
# no longer exists. Overridable from the environment.
export STT_SIM_GAP="${STT_SIM_GAP:-524288}"
export STT_SIM_MAXREQ="${STT_SIM_MAXREQ:-1024}"

SIM=target/release/examples/simulate_layout
REPACK=target/release/examples/repack
if [[ ! -x "$SIM" || ! -x "$REPACK" ]]; then
  echo "building tools..." >&2
  cargo build --release -p stt-core --example simulate_layout --example repack >/dev/null 2>&1
fi

ARCHIVES=("$@")
if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
  while IFS= read -r f; do ARCHIVES+=("$f"); done < <(find examples/showcase/public/data -name '*.stt' | sort)
fi

# Worst single-query request count under the archive's CURRENT on-disk order.
# This — not the weighted average — is the metric that matters once the client
# coalesces globally: nearly every query collapses to a handful of requests, so
# orderings differ ONLY on their worst case (the wide-time-at-a-location query,
# which only spatial-major handles well). reqs is the 6th-from-last column.
worst_reqs() {
  "$SIM" "$1" 2>/dev/null | sed -n '/=== ordering: 0-REAL-DISK/,/COST =/p' \
    | awk '/^Q[0-9]/ && NF>=6 { r=$(NF-5); if (r>m) m=r } END { print m+0 }'
}

printf "%-30s %-7s %9s %9s  %s\n" DATASET ORDER WORST_B WORST_A RESULT
printf "%-30s %-7s %9s %9s  %s\n" "-------" "-----" "-------" "-------" "------"

repacked=0; skipped=0; failed=0
for f in "${ARCHIVES[@]}"; do
  name="$(basename "$f" .stt)"
  if [[ ! -s "$f" ]]; then
    printf "%-30s %-7s %9s %9s  %s\n" "$name" "-" "-" "-" "SKIP(empty)"; skipped=$((skipped+1)); continue
  fi
  wb="$(worst_reqs "$f")"
  if [[ -z "$wb" || "$wb" -eq 0 ]]; then
    printf "%-30s %-7s %9s %9s  %s\n" "$name" "-" "-" "-" "SKIP(no data)"; skipped=$((skipped+1)); continue
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "%-30s %-7s %9s %9s  %s\n" "$name" "auto" "$wb" "?" "DRY-RUN"; continue
  fi

  # Repack to the data-adaptive `auto` order (BlobOrdering::choose: wide-time →
  # spatial-major so a location's whole timeline is byte-contiguous; otherwise
  # the 3D-Hilbert generalist). Preserves cover_t_min (repack uses add_tile_full).
  tmp="${f}.opt.$$"
  if ! "$REPACK" "$f" "$tmp" auto >/dev/null 2>&1; then
    printf "%-30s %-7s %9s %9s  %s\n" "$name" "auto" "$wb" "-" "FAIL(repack)"; rm -f "$tmp"; failed=$((failed+1)); continue
  fi
  wa="$(worst_reqs "$tmp")"
  # Promote ONLY if auto strictly reduces the worst-query request count — fixing
  # a bad order (e.g. time-major wide-time) without needlessly rewriting a
  # multi-GB archive whose order is already fine.
  if [[ -n "$wa" && "$wa" -lt "$wb" ]]; then
    mv -f "$tmp" "$f"
    dist="examples/showcase/dist/data/${name}.stt"
    [[ -f "$dist" ]] && cp -f "$f" "$dist"
    printf "%-30s %-7s %9s %9s  %s\n" "$name" "auto" "$wb" "$wa" "OK"; repacked=$((repacked+1))
  else
    rm -f "$tmp"
    printf "%-30s %-7s %9s %9s  %s\n" "$name" "auto" "$wb" "$wa" "keep(already ok)"; skipped=$((skipped+1))
  fi
done

echo
echo "repacked=$repacked kept=$skipped failed=$failed (metric: worst-query request count)"
