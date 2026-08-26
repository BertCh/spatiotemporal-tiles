#!/usr/bin/env bash
#
# Fleet blob-ordering re-audit (WM-3's acceptance gate).
#
# Runs `stt-optimize order-audit --format json` over every packed archive under
# the given roots and emits ONE newline-delimited JSON stream on stdout: one
# object per archive, with the archive's path folded in beside the audit report.
#
#   tools/fleet-order-audit.sh [ROOT ...] > audit.ndjson
#
# Defaults to the local showcase archive set (`data-fleet`)
# plus `data/`. Archives are discovered by their `manifest.json` and visited in
# LC_ALL=C sorted path order, so two runs over an unchanged tree produce
# byte-identical output — the "zero nondeterminism" half of WM-3's acceptance is
# checked by simply diffing two runs:
#
#   tools/fleet-order-audit.sh > a.ndjson && tools/fleet-order-audit.sh > b.ndjson
#   diff a.ndjson b.ndjson && echo "deterministic"
#
# Summarise picks with jq, e.g.:
#
#   jq -r '[.archive, .current // "-", .recommended, .auto_choice] | @tsv' audit.ndjson
#   jq -s 'group_by(.recommended) | map({(.[0].recommended): length}) | add' audit.ndjson
#
# The before/after comparison the ordering-default flip is gated on is two runs
# under the two weightings — `legacy` is the pre-2026-08 scrub+pan model:
#
#   STT_ORDER_AUDIT_WORKLOAD=legacy  tools/fleet-order-audit.sh > legacy.ndjson
#   STT_ORDER_AUDIT_WORKLOAD=derived tools/fleet-order-audit.sh > derived.ndjson
#
# Dot-prefixed directories (`.foo.bak-drift` and friends) are skipped: they are
# local scratch copies, not fleet members.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
roots=("$@")
if [ ${#roots[@]} -eq 0 ]; then
  roots=("$repo_root/data-fleet" "$repo_root/data")
fi

bin="${STT_OPTIMIZE_BIN:-}"
if [ -z "$bin" ]; then
  # Prefer an already-built binary; fall back to building one in release mode.
  for candidate in "$repo_root/target/release/stt-optimize" "$repo_root/target/debug/stt-optimize"; do
    if [ -x "$candidate" ]; then bin="$candidate"; break; fi
  done
fi
if [ -z "$bin" ]; then
  cargo build --release -p spatiotemporal-tiles --bin stt-optimize >&2
  bin="$repo_root/target/release/stt-optimize"
fi

workload="${STT_ORDER_AUDIT_WORKLOAD:-derived}"

manifests=()
for root in "${roots[@]}"; do
  [ -d "$root" ] || continue
  while IFS= read -r m; do
    case "$m" in
      */.*/*) continue ;;  # scratch/backup copies
    esac
    manifests+=("$m")
  done < <(find "$root" -name manifest.json -type f | LC_ALL=C sort)
done

audited=0
failed=0
for manifest in "${manifests[@]}"; do
  archive="$(dirname "$manifest")"
  name="${archive#"$repo_root"/}"
  if report="$("$bin" order-audit --archive "$archive" --format json \
      --ordering-workload "$workload" 2>/dev/null)"; then
    printf '%s\n' "$report" | python3 -c '
import json, sys
report = json.load(sys.stdin)
report["archive"] = sys.argv[1]
print(json.dumps(report, sort_keys=True, separators=(",", ":")))
' "$name"
    audited=$((audited + 1))
  else
    printf 'skipped (order-audit failed): %s\n' "$name" >&2
    failed=$((failed + 1))
  fi
done

printf 'audited %d archive(s), %d skipped\n' "$audited" "$failed" >&2
