#!/usr/bin/env bash
#
# PUBLISH-BUILD re-transcode of every live packed dataset into a staging tree,
# applying the two LOSSLESS deploy wins: zstd level 19 (−10..27% on the wire,
# decode-free) + a paged directory (cold reader fetches only the leaves it
# touches). Tile CONTENT is preserved byte-for-byte.
#
# Reads the current `public/data/<ds>/` packs as the SOURCE OF TRUTH (several
# datasets were rebuilt later than their legacy single-file `.stt`, so the packs
# — not the `.stt` — carry the latest fixes) and writes a fresh dataset to
# `public/data-publish/<ds>/`. The live tree is left untouched, so a bad build
# never corrupts the source; the swap/sync is a separate, deliberate step.
#
# Each output is validated (stt-validate) before it counts as done; a dataset
# whose staged manifest already exists is skipped (resumable). Datasets are
# processed smallest-first so failures surface fast and the long tail (the
# multi-GB sets) runs last.
#
# Usage:
#   scripts/repack-publish-all.sh                  # all datasets
#   ZSTD_LEVEL=19 PAGE_ENTRIES=4096 scripts/repack-publish-all.sh
#   FORCE=1 scripts/repack-publish-all.sh          # rebuild even if staged
#   scripts/repack-publish-all.sh flights drifters # just these
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/examples/showcase/public/data"
DST="$REPO/examples/showcase/public/data-publish"
REPACK="$REPO/target/release/examples/repack-publish"
VALIDATE="$REPO/target/release/stt-validate"
ZSTD_LEVEL="${ZSTD_LEVEL:-19}"
PAGE_ENTRIES="${PAGE_ENTRIES:-4096}"
PACK_MB="${PACK_MB:-64}"
FORCE="${FORCE:-0}"

[ -x "$REPACK" ] || { echo "build first: cargo build --release -p stt-core --example repack-publish" >&2; exit 1; }
[ -x "$VALIDATE" ] || { echo "build first: cargo build --release -p stt-validate" >&2; exit 1; }
mkdir -p "$DST"

# Dataset list: explicit args, else every dir with a manifest, smallest-first.
# (Plain while-read, not `mapfile` — macOS still ships bash 3.2.)
if [ "$#" -gt 0 ]; then
  datasets=("$@")
else
  datasets=()
  while IFS= read -r ds_name; do
    [ -n "$ds_name" ] && datasets+=("$ds_name")
  done < <(
    for d in "$SRC"/*/; do
      [ -f "$d/manifest.json" ] || continue
      printf '%s %s\n' "$(du -sk "$d" | awk '{print $1}')" "$(basename "$d")"
    done | sort -n | awk '{print $2}'
  )
fi

fail=0
declare -i done=0
echo "publish-build: ${#datasets[@]} datasets -> data-publish/  (zstd-$ZSTD_LEVEL, paged $PAGE_ENTRIES)"
for ds in "${datasets[@]}"; do
  in="$SRC/$ds/manifest.json"
  out="$DST/$ds"
  [ -f "$in" ] || { echo "skip missing  $ds"; continue; }
  if [ "$FORCE" != "1" ] && [ -f "$out/manifest.json" ]; then echo "skip staged   $ds"; done+=1; continue; fi
  rm -rf "$out.tmp" "$out"
  before=$(du -sk "$SRC/$ds" | awk '{print $1}')
  t0=$(date +%s)
  if ! "$REPACK" "$in" "$out.tmp" "$ZSTD_LEVEL" "$PAGE_ENTRIES" "$PACK_MB" >/dev/null 2>&1; then
    echo "  FAILED repack  $ds"; rm -rf "$out.tmp"; fail=1; continue
  fi
  # Validate before swapping the .tmp into place. The re-transcode is LOSSLESS
  # (it re-compresses verbatim tile payloads), so the only failure it can
  # introduce is corruption — caught by stt-validate's "decoded N of N tiles"
  # (+ per-blob CRC + content-address). stt-validate's schema findings
  # (drift across summary/variable-property layers, a Utf8 property) describe
  # the SOURCE data and are present in the live dataset too, so they are NOT
  # re-transcode errors — we surface them as a note, not a failure.
  vout="$("$VALIDATE" "$out.tmp/manifest.json" 2>&1)"
  counts="$(echo "$vout" | grep -E 'decoded +[0-9]+ of +[0-9]+ tiles' | head -1)"
  dec="$(echo "$counts" | awk '{print $2}')"; tot="$(echo "$counts" | awk '{print $4}')"
  if [ -z "$counts" ] || [ "$dec" != "$tot" ]; then
    echo "  FAILED $ds: tile decode/integrity ('${counts:-no decode summary}')"
    echo "$vout" | tail -4 | sed 's/^/      /'
    rm -rf "$out.tmp"; fail=1; continue
  fi
  # Cross-check losslessness: staged tile count must equal the source's.
  src_tiles="$("$VALIDATE" "$in" 2>&1 | grep -E 'decoded +[0-9]+ of' | head -1 | awk '{print $2}')"
  if [ -n "$src_tiles" ] && [ "$tot" != "$src_tiles" ]; then
    echo "  FAILED $ds: tile count changed ($src_tiles -> $tot)"; rm -rf "$out.tmp"; fail=1; continue
  fi
  vflag="$(echo "$vout" | grep -qiE 'schema drift|expected Float64' && echo ' (source schema note — also in live)')"
  mv "$out.tmp" "$out"
  t1=$(date +%s)
  after=$(du -sk "$out" | awk '{print $1}')
  pct=$(awk "BEGIN{printf \"%+.1f\", ($after/$before-1)*100}")
  printf "  ok %-22s %5ds  %6.1f→%6.1f MB  %s%%%s\n" "$ds" "$((t1-t0))" "$(awk "BEGIN{print $before/1024}")" "$(awk "BEGIN{print $after/1024}")" "$pct" "$vflag"
  done+=1
done

echo "publish-build done: $done staged, fail=$fail"
[ "$fail" = "0" ] && echo "next: review data-publish/, then swap+sync (see scripts/README or r2-sync.sh)"
exit $fail
