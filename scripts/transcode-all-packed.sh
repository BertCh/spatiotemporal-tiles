#!/usr/bin/env bash
#
# ONE-TIME MIGRATION. Transcode every legacy single-file v4 .stt in the showcase
# data dir into the packed format (manifest.json + index/<hash>.sttd +
# packs/<hash>.sttp). No generators are re-run — the existing .stt files are read
# directly. Skips empty/scratch files and datasets already transcoded.
#
# This is NOT part of the steady-state build/deploy flow: `stt-build` now emits
# packed datasets directly. Use this only to migrate pre-existing single-file
# `.stt` archives (e.g. the original showcase data) to packed.
#
# Usage:
#   scripts/transcode-all-packed.sh            # transcode all
#   PACK_MB=32 scripts/transcode-all-packed.sh # custom pack target
#   FORCE=1 scripts/transcode-all-packed.sh    # re-transcode even if present
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$REPO/examples/showcase/public/data"
BIN="$REPO/target/release/examples/pack-transcode"
PACK_MB="${PACK_MB:-64}"
FORCE="${FORCE:-0}"

[ -x "$BIN" ] || { echo "error: build it first: cargo build --release -p stt-core --example pack-transcode" >&2; exit 1; }

fail=0
for f in "$DATA"/*.stt; do
  [ -e "$f" ] || continue
  stem="$(basename "$f" .stt)"
  case "$stem" in *.new) echo "skip scratch  $stem"; continue;; esac
  if [ ! -s "$f" ]; then echo "skip empty    $stem (0 bytes)"; continue; fi
  out="$DATA/$stem"
  if [ "$FORCE" != "1" ] && [ -f "$out/manifest.json" ]; then
    echo "skip done     $stem"; continue
  fi
  sz="$(du -h "$f" | awk '{print $1}')"
  echo ">> transcoding $stem ($sz) -> $stem/"
  t0=$(date +%s)
  if "$BIN" "$f" "$out" "$PACK_MB" >/dev/null; then
    t1=$(date +%s)
    echo "   ok $stem  ${t1}s  $((t1-t0))s elapsed  $(ls "$out/packs" 2>/dev/null | wc -l | tr -d ' ') packs"
  else
    echo "   FAILED $stem"; fail=1
  fi
done
echo "done (fail=$fail)."
exit $fail
