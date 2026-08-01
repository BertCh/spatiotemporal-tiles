#!/usr/bin/env bash
#
# Sync showcase tile datasets to a Cloudflare R2 bucket.
#
# Each dataset is now a packed DIRECTORY (not a single `.stt` file):
#
#   data/<stem>/
#     manifest.json          tiny, MUTABLE   → short TTL (the only deploy-fresh object)
#     index/<blake3>.sttd     directory blob  → IMMUTABLE (content-addressed)
#     packs/<blake3>.sttp     tile blob data  → IMMUTABLE (content-addressed)
#
# Because packs/ and index/ are content-addressed (blake3 → filename), their
# bytes never change without their name changing. They ship `immutable` and are
# cached forever: a re-sync skips unchanged objects (rclone size+mtime/checksum)
# and they NEVER need a cache purge. Only `manifest.json` is mutable, so it gets
# a short TTL — it's the one (tiny, cheap) object a deploy must refresh/purge.
#
# rclone applies a single `--header-upload` per invocation, so the two
# Cache-Control regimes are TWO passes:
#   (a) immutable pass — `packs/**` + `index/**` with max-age=1y, immutable
#   (b) manifest pass  — `manifest.json` with max-age=60, must-revalidate
#
# Both passes use `rclone copy` (NEVER `sync`): a deploy must not delete the
# previous deploy's packs out from under live sessions. Manifests are cached
# up to 60 s, and an open tab holds its manifest in memory for the whole
# session — a pruned-on-sync pack 404s exactly those readers.
#
# Garbage collection is a separate, retention-aware pass (c): an immutable
# object (packs/** or index/**) is deleted only when it is ALL of
#   - unreferenced by the dataset's CURRENT local manifest, AND
#   - unreferenced by the CURRENTLY-DEPLOYED remote manifest, which is read
#     BEFORE anything uploads (the one-deploy "grace rule" of the format
#     spec §2: a republish that changes every object — a format-version
#     bump, a full rebuild — must not reap the previous deploy while ≤60 s
#     edge-cached manifests and open sessions still resolve it), AND
#   - older than the retention window (R2_PRUNE_RETENTION, default 7d) —
#     long enough for every cached manifest and live session to drain.
# If the deployed manifest(s) cannot be READ (as opposed to being absent —
# a first deploy has none and proceeds normally), the prune pass is SKIPPED
# for that run: without them the script cannot prove an object unreferenced.
# `--prune-now` drops only the age gate (the grace rule still protects the
# previous deploy's references); it is for disaster cleanup and CAN 404
# sessions older than the previous deploy. `--no-prune` skips pass (c)
# entirely and is the RECOMMENDED mode for major republishes: upload
# additively, then let a later default sync GC once the retention window
# has passed. `manifest.json` itself is never pruned — which also means a
# dataset REMOVED locally keeps protecting its remote objects (its deployed
# manifest is re-read every sync); to retire one, delete its remote
# manifest.json first, then let GC collect the rest.
#
# Reads credentials from .env (see .env.r2.example). rclone is configured
# entirely from env vars, so nothing is written to ~/.config/rclone/rclone.conf
# and no secret is persisted on disk by this script.
#
# Usage:
#   scripts/r2-sync.sh                 # copy all of public/data -> r2:<bucket>/data, then GC
#   scripts/r2-sync.sh --dry-run       # show what would change, transfer/delete nothing
#                                      # (the grace-rule manifest read still runs: read-only,
#                                      #  and the prune preview is wrong without it)
#   scripts/r2-sync.sh flights         # one dataset dir data/flights/ (e.g. after regen)
#   scripts/r2-sync.sh --no-prune      # copy only, skip GC — RECOMMENDED for major republishes
#   scripts/r2-sync.sh --prune-now     # GC ignores the retention window (grace rule still applies)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
# Source tree to sync. Defaults to the live showcase data dir; override with
# STT_DATA_DIR to deploy an alternate staging tree straight to R2 without a
# destructive local swap, e.g.
#   STT_DATA_DIR=examples/showcase/public/data-staging scripts/r2-sync.sh
# NOTE: the tree must be COMPLETE — this script GCs remote objects that the
# local tree doesn't reference, so syncing a partial tree deletes the rest.
DATA_DIR="${STT_DATA_DIR:-${REPO_ROOT}/examples/showcase/public/data}"
case "$DATA_DIR" in /*) ;; *) DATA_DIR="${REPO_ROOT}/${DATA_DIR}";; esac

# Tuning for multi-GB pack objects: large multipart chunks, parallel parts/files.
# NOTE: no --header-upload here — Cache-Control is per-pass (see below) since
# rclone honours only one --header-upload per invocation.
COMMON_FLAGS=(
  --progress
  --transfers 4
  --checkers 8
  --s3-chunk-size 64M
  --s3-upload-concurrency 8
  --s3-no-check-bucket
)

# License gate: Waymo Open Dataset terms forbid redistribution, so waymo-*
# dataset bundles must never reach the public bucket. The all-datasets walk
# excludes them via ordered `--filter` rules (NOT --exclude: rclone documents
# the relative order of mixed --include/--exclude flags as INDETERMINATE, and
# v1.74 actually places includes first — which silently defeated this gate).
# Only `--filter` rules are guaranteed first-match-wins in flag order, so every
# copy pass below builds its whole filter list from `--filter` alone, leading
# with these excludes and ending with an explicit "- **". Single-stem sync
# refuses waymo outright, and the prune pass treats any remote waymo-* object
# as unreferenced so an accidental prior upload self-heals (gets deleted once
# past the retention window). Hidden dot-dirs (e.g. `.foo-bundle.bak` backup
# trees) are local-only scratch and never ship.
LICENSE_EXCLUDE_FLAGS=(--filter "- waymo-*/**" --filter "- .*/**")

# Cache-Control headers, one per regime.
#   immutable: content-addressed packs/index never change under a stable name →
#     cache forever, never purge. A re-sync simply skips unchanged objects.
#   manifest: the only mutable object → short TTL so a redeploy is visible fast;
#     tiny (a few KB), so a purge (if you choose to) is cheap.
IMMUTABLE_HEADER='Cache-Control: public, max-age=31536000, immutable'
MANIFEST_HEADER='Cache-Control: public, max-age=60, must-revalidate'

# GC retention window: an unreferenced immutable object younger than this is
# kept (cached manifests + live sessions may still reference it).
PRUNE_RETENTION="${R2_PRUNE_RETENTION:-7d}"

# Defaults for the flag surface (parsed after the lib-only hook below).
EXTRA_FLAGS=()
SINGLE_STEM=""
PRUNE=1
PRUNE_NOW=0
PRUNE_SKIP_REASON="--no-prune"

# ---------------------------------------------------------------------------
# Referenced-set computation.
# The first two helpers are pure (no rclone, no .env, no network) so they can
# be unit-tested by sourcing this script with R2_SYNC_LIB_ONLY=1 (see below).
# ---------------------------------------------------------------------------

# emit_manifest_refs REL   (manifest JSON on stdin)
# Print one rclone exclude line ("- /<REL><key>") per immutable object the
# manifest references: directory.key + packs[].key + schemas[].key — the last
# for forward-compat with manifest versions that reference external schema
# objects; today's manifests have no such field and its absence is fine.
# Parsing is deliberately defensive (manifests permit unknown fields at every
# envelope level): missing/oddly-typed fields are skipped, only invalid JSON
# fails (exit 1) so a corrupt manifest never silently protects nothing.
emit_manifest_refs() {
  local rel="$1"
  # NB: program via -c, NOT `python3 -` + heredoc — the manifest arrives on
  # stdin (works for both `< file` and `rclone cat |`), so stdin must stay free.
  python3 -c '
import json, sys

rel = sys.argv[1]
try:
    m = json.load(sys.stdin)
except ValueError as e:
    sys.stderr.write("emit_manifest_refs: invalid manifest JSON: %s\n" % e)
    sys.exit(1)
if not isinstance(m, dict):
    sys.exit(0)

keys = []
directory = m.get("directory")
if isinstance(directory, dict) and isinstance(directory.get("key"), str):
    keys.append(directory["key"])
for field in ("packs", "schemas"):
    entries = m.get(field)
    if isinstance(entries, list):
        keys.extend(e["key"] for e in entries
                    if isinstance(e, dict) and isinstance(e.get("key"), str))
for key in keys:
    print("- /%s%s" % (rel, key))
' "${rel}"
}

# emit_local_refs SRC
# Walk every manifest.json under the local tree SRC and emit exclude lines for
# everything they reference, anchored at each manifest's directory relative to
# the transfer root (handles both call shapes: all-datasets root and single
# dataset, plus nested manifests like <stem>/lidar/manifest.json).
# License gate: a local waymo-* manifest must not protect remote objects —
# nothing waymo should exist on R2, so leaving them unreferenced lets the
# prune pass clean up any accidental upload. Hidden dot-dir manifests (local
# `.foo.bak` backup trees) likewise never ship, and a stale backup manifest
# must not protect superseded packs from GC.
emit_local_refs() {
  local src="$1" manifest rel
  while IFS= read -r manifest; do
    rel="$(dirname "${manifest#"${src}"/}")"
    [[ "${rel}" == "." ]] && rel=""
    [[ -n "${rel}" ]] && rel="${rel}/"
    case "${rel}" in waymo-*|.*) continue ;; esac
    emit_manifest_refs "${rel}" < "${manifest}"
  done < <(find "${src}" -name manifest.json)
}

# fetch_remote_refs DST OUT
# Grace rule (format spec §2): read every CURRENTLY-DEPLOYED manifest.json
# under DST and append exclude lines for the objects they reference to the
# file OUT, so the prune pass keeps the previous deploy resolvable for one
# deploy cycle (≤60 s edge-cached manifests and open sessions hold the OLD
# manifest and still fetch its packs). MUST run BEFORE the manifest upload
# pass overwrites the deployed manifests. No remote manifests (first deploy)
# = success with empty OUT. Any listing/fetch/parse failure returns non-zero;
# the caller then SKIPS pruning rather than reap objects it could not prove
# unreferenced. Same waymo-*/dot-dir gate as emit_local_refs: a remote waymo
# manifest must not protect objects that should never have shipped.
fetch_remote_refs() {
  local dst="$1" out="$2"
  local listing path rel
  if ! listing="$(rclone lsf -R --files-only \
      --filter "+ manifest.json" --filter "+ **/manifest.json" --filter "- **" \
      "${dst}")"; then
    return 1
  fi
  [[ -z "${listing}" ]] && return 0
  while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    rel="$(dirname "${path}")"
    [[ "${rel}" == "." ]] && rel=""
    [[ -n "${rel}" ]] && rel="${rel}/"
    case "${rel}" in waymo-*|.*) continue ;; esac
    if ! rclone cat "${dst}/${path}" | emit_manifest_refs "${rel}" >> "${out}"; then
      return 1
    fi
  done <<< "${listing}"
  return 0
}

# Copy one local tree -> one remote prefix in two header passes, then GC.
#   pass (0): grace-rule read of the deployed manifests (prune protection)
#   pass (a): immutable assets — every */packs/** and */index/** (copy, additive)
#   pass (b): mutable manifest — every */manifest.json (copy overwrites changed)
#   pass (c): retention-aware prune of unreferenced immutable objects
# Each pass is scoped by --include/--filter so it carries exactly the objects
# that pass's --header-upload (or deletion policy) applies to.
#
# The include globs are written to match at ANY depth so one function works for
# both call shapes:
#   - all-datasets root (src = .../data):   matches <stem>/packs/**, <stem>/index/**, <stem>/manifest.json
#   - single dataset    (src = .../<stem>): matches packs/**, index/**, manifest.json
# rclone's `**` spans path separators, so `**/packs/**` covers the leading-dir
# case while `packs/**` covers the tree-root case.
sync_tree() {
  local src="$1" dst="$2"

  # Pass (0): capture what the CURRENTLY-DEPLOYED manifests reference BEFORE
  # any upload — pass (b) below overwrites them. Read-only, runs even under
  # --dry-run (without it the prune preview would show grace-protected
  # objects as deletions).
  local grace_refs=""
  if [[ "${PRUNE}" -eq 1 ]]; then
    grace_refs="$(mktemp)"
    echo ">> [grace]     reading currently-deployed manifest(s) under ${dst}"
    if ! fetch_remote_refs "${dst}" "${grace_refs}"; then
      PRUNE=0
      PRUNE_SKIP_REASON="failed to read the currently-deployed remote manifest(s); fail-safe"
      {
        echo "!! [grace]     WARNING: could not read the deployed manifest(s) under ${dst}."
        echo "!!             Pruning is SKIPPED this run: without them the prune pass cannot"
        echo "!!             honor the one-deploy grace rule (format spec §2) and could 404"
        echo "!!             live sessions. Uploads proceed; re-run later to GC."
      } >&2
    fi
  fi

  echo ">> [immutable] ${src} (packs/index) -> ${dst}"
  rclone copy "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    "${LICENSE_EXCLUDE_FLAGS[@]}" \
    --header-upload "${IMMUTABLE_HEADER}" \
    --filter "+ packs/**" --filter "+ index/**" \
    --filter "+ **/packs/**" --filter "+ **/index/**" \
    --filter "- **" \
    "${src}" "${dst}"

  echo ">> [manifest]  ${src} (manifest.json) -> ${dst}"
  rclone copy "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    "${LICENSE_EXCLUDE_FLAGS[@]}" \
    --header-upload "${MANIFEST_HEADER}" \
    --filter "+ manifest.json" --filter "+ **/manifest.json" \
    --filter "- **" \
    "${src}" "${dst}"

  # AV "scene bundle" sidecars (type:'av' datasets): scene.json / telemetry.json /
  # cameras.json + the camera JPGs under cam/. These are NOT content-addressed
  # (stable filenames whose bytes change on a re-extract), so they ride the
  # mutable/short-TTL regime like manifest.json. No-op for non-AV datasets (none
  # match these globs). The prune pass below only deletes packs/index, so these
  # are never garbage-collected.
  echo ">> [av-sidecar] ${src} (scene/telemetry/cameras/cam) -> ${dst}"
  rclone copy "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    "${LICENSE_EXCLUDE_FLAGS[@]}" \
    --header-upload "${MANIFEST_HEADER}" \
    --filter "+ scene.json" --filter "+ **/scene.json" \
    --filter "+ telemetry.json" --filter "+ **/telemetry.json" \
    --filter "+ cameras.json" --filter "+ **/cameras.json" \
    --filter "+ cam/**" --filter "+ **/cam/**" \
    --filter "- **" \
    "${src}" "${dst}"

  # Cosmos-Drive-Dreams "/worlds" bundle sidecars (type:'worlds'): the
  # scenario index `worlds.json` at the bundle root plus the per-scenario
  # Cosmos videos under `videos/`. Like the AV sidecars these are NOT
  # content-addressed (stable filenames whose bytes change on a re-generate),
  # so they ride the mutable/short-TTL regime — a re-upload must never serve a
  # stale world index or the wrong video. The hero LiDAR under `heroes/*/lidar/`
  # is packed (manifest/index/packs) and already rides the immutable+manifest
  # passes above, so it needs no rule here. No-op for every non-worlds dataset.
  # (Videos are large; if 60 s revalidation becomes a cost, give `videos/**` a
  # dedicated longer-TTL header — they change far less often than worlds.json.)
  echo ">> [worlds]    ${src} (worlds.json + videos/) -> ${dst}"
  rclone copy "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    "${LICENSE_EXCLUDE_FLAGS[@]}" \
    --header-upload "${MANIFEST_HEADER}" \
    --filter "+ worlds.json" --filter "+ **/worlds.json" \
    --filter "+ videos/**" --filter "+ **/videos/**" \
    --filter "- **" \
    "${src}" "${dst}"

  # Root-level dataset sidecars: `<stem>.meta.json` and the `<stem>-*.bin`
  # typed-array blobs. The Neural-State Atlas is the first surface to use them
  # — its node index and per-node series are plain binary buffers fetched
  # directly, not packed archives — and NO pass above matches them: they sit at
  # the data root rather than under `packs/`, `index/` or a bundle directory,
  # so every filter list ends up rejecting them at its trailing `- **`.
  # Discovered 2026-07-31, when the three atlas ARCHIVES synced and the two
  # `.bin` sidecars did not, which is precisely the partial-sync state
  # ATLAS_ARCHIVES_SYNCED exists to keep off the public deploy.
  # Mutable regime (stable names, bytes change on regenerate), so they take the
  # short-TTL header like manifest.json — never the immutable one.
  echo ">> [sidecar]   ${src} (*.meta.json + *.bin) -> ${dst}"
  rclone copy "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    "${LICENSE_EXCLUDE_FLAGS[@]}" \
    --header-upload "${MANIFEST_HEADER}" \
    --filter "+ *.meta.json" --filter "+ *.bin" \
    --filter "- **" \
    "${src}" "${dst}"

  if [[ "${PRUNE}" -eq 1 ]]; then
    prune_tree "${src}" "${dst}" "${grace_refs}"
  else
    echo ">> [prune]     skipped (${PRUNE_SKIP_REASON})"
  fi
  if [[ -n "${grace_refs}" ]]; then rm -f "${grace_refs}"; fi
}

# Pass (c): delete remote packs/index objects that neither the CURRENT local
# manifests nor the PREVIOUSLY-DEPLOYED manifests (grace rule — arg 3, the
# refs captured by pass (0)) reference, and only once they are older than the
# retention window (so cached manifests / open sessions referencing even
# earlier deploys keep resolving). The filter file lists every referenced
# object as an exclusion (first match wins in rclone filters), then includes
# the immutable trees, then excludes everything else — `manifest.json` is
# therefore never deleted.
prune_tree() {
  local src="$1" dst="$2" grace_refs="${3:-}"
  local filter_file local_refs
  filter_file="$(mktemp)"
  local_refs="$(mktemp)"

  # Referenced set, part 1: everything the CURRENT local manifests reference.
  emit_local_refs "${src}" > "${local_refs}"
  cat "${local_refs}" > "${filter_file}"

  # Referenced set, part 2 (grace rule): everything the manifests deployed
  # BEFORE this sync reference. Protected for this one deploy cycle; on the
  # NEXT sync the deployed manifests are the ones just uploaded, so these
  # become unreferenced and --min-age ages them out.
  local grace_only=0
  if [[ -n "${grace_refs}" && -s "${grace_refs}" ]]; then
    grace_only="$(comm -23 <(sort -u "${grace_refs}") <(sort -u "${local_refs}") | wc -l | tr -d ' ')"
    cat "${grace_refs}" >> "${filter_file}"
    if [[ "${grace_only}" -gt 0 ]]; then
      echo ">> [grace]     protecting ${grace_only} object(s) referenced only by the previously-deployed manifest(s)"
    fi
  fi

  {
    echo "+ /packs/**"
    echo "+ /index/**"
    echo "+ /**/packs/**"
    echo "+ /**/index/**"
    echo "- **"
  } >> "${filter_file}"

  local age_flags=(--min-age "${PRUNE_RETENTION}")
  if [[ "${PRUNE_NOW}" -eq 1 ]]; then
    age_flags=()
    if [[ "${grace_only}" -gt 0 ]]; then
      {
        echo "!! ------------------------------------------------------------------------"
        echo "!! --prune-now: the previously-deployed manifest(s) still reference"
        echo "!! ${grace_only} object(s) that this deploy does not. The grace rule keeps those"
        echo "!! protected, but every OTHER unreferenced object is deleted IMMEDIATELY —"
        echo "!! no retention window. Live sessions and cached manifests from deploys"
        echo "!! older than the previous one CAN 404. For a major republish prefer"
        echo "!! --no-prune and let a later default sync GC after R2_PRUNE_RETENTION"
        echo "!! (currently ${PRUNE_RETENTION})."
        echo "!! ------------------------------------------------------------------------"
      } >&2
    fi
    echo ">> [prune]     ${dst}: deleting ALL unreferenced packs/index objects (--prune-now)"
  else
    echo ">> [prune]     ${dst}: deleting unreferenced packs/index objects older than ${PRUNE_RETENTION}"
  fi
  rclone delete ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    ${age_flags[@]+"${age_flags[@]}"} \
    --filter-from "${filter_file}" \
    "${dst}"
  rm -f "${filter_file}" "${local_refs}"
}

# Test hook: `R2_SYNC_LIB_ONLY=1 source scripts/r2-sync.sh` stops here — the
# helpers above become callable without .env, credentials, or a configured
# remote (used by scripts/tests for the referenced-set/prune logic; rclone
# accepts plain local paths as src/dst, so sync_tree is exercisable against
# fixture directories).
if [[ "${R2_SYNC_LIB_ONLY:-0}" == "1" ]]; then
  # shellcheck disable=SC2317  # `exit` is the executed-not-sourced fallback
  return 0 2>/dev/null || exit 0
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "error: ${ENV_FILE} not found. See .env.r2.example for the required keys." >&2
  exit 1
fi
set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

# Accept either the canonical names or the shorter aliases used in .env.
R2_ACCOUNT_ID="${R2_ACCOUNT_ID:-${R2_ID:-}}"
R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-${R2_KEY_ID:-}}"
R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-${R2_SECRET_KEY:-}}"
R2_BUCKET="${R2_BUCKET:-stt-tiles}"

: "${R2_ACCOUNT_ID:?set R2_ID (account id) in .env}"
: "${R2_ACCESS_KEY_ID:?set R2_KEY_ID (S3 Access Key ID) in .env}"
: "${R2_SECRET_ACCESS_KEY:?set R2_SECRET_KEY (S3 Secret Access Key) in .env}"

# Configure the "r2" remote purely via environment (no persisted config).
export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
export RCLONE_CONFIG_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export RCLONE_CONFIG_R2_REGION=auto
# R2 has no ACLs; suppress the no-op ACL header.
export RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true

# Pass any unrecognized flags (e.g. --dry-run) straight through to rclone;
# treat a bare arg as a single dataset STEM (its directory data/<stem>/).
for arg in "$@"; do
  case "${arg}" in
    --no-prune)  PRUNE=0 ;;
    --prune-now) PRUNE_NOW=1 ;;
    --*) EXTRA_FLAGS+=("${arg}") ;;
    *)   SINGLE_STEM="${arg%/}" ;;  # tolerate a trailing slash
  esac
done

if [[ -n "${SINGLE_STEM}" ]]; then
  case "${SINGLE_STEM}" in
    waymo-*)
      echo "error: '${SINGLE_STEM}' is a Waymo Open Dataset bundle — the license forbids" >&2
      echo "       redistribution, so it must never be synced to the public bucket." >&2
      exit 1
      ;;
  esac
  if [[ ! -d "${DATA_DIR}/${SINGLE_STEM}" ]]; then
    echo "error: ${DATA_DIR}/${SINGLE_STEM} is not a directory. Pass a dataset STEM" >&2
    echo "       (the dir name under public/data, e.g. 'flights'), not a .stt file." >&2
    exit 1
  fi
  echo ">> syncing dataset '${SINGLE_STEM}' -> r2:${R2_BUCKET}/data/${SINGLE_STEM}"
  sync_tree "${DATA_DIR}/${SINGLE_STEM}" "r2:${R2_BUCKET}/data/${SINGLE_STEM}"
else
  # Walk all dataset directories recursively. The `**` globs match at any depth,
  # so the two passes correctly classify every dataset's packs/index vs manifest.
  echo ">> syncing all datasets ${DATA_DIR} -> r2:${R2_BUCKET}/data"
  sync_tree "${DATA_DIR}" "r2:${R2_BUCKET}/data"
fi

echo ">> done."
