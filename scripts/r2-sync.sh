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
# Reads credentials from .env (see .env.r2.example). rclone is configured
# entirely from env vars, so nothing is written to ~/.config/rclone/rclone.conf
# and no secret is persisted on disk by this script.
#
# Usage:
#   scripts/r2-sync.sh                 # sync all of public/data -> r2:<bucket>/data
#   scripts/r2-sync.sh --dry-run       # show what would change, transfer nothing
#   scripts/r2-sync.sh flights         # sync a single dataset dir data/flights/ (e.g. after regen)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
DATA_DIR="${REPO_ROOT}/examples/showcase/public/data"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "error: ${ENV_FILE} not found. See .env.r2.example for the required keys." >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a; source "${ENV_FILE}"; set +a

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

# Cache-Control headers, one per regime.
#   immutable: content-addressed packs/index never change under a stable name →
#     cache forever, never purge. A re-sync simply skips unchanged objects.
#   manifest: the only mutable object → short TTL so a redeploy is visible fast;
#     tiny (a few KB), so a purge (if you choose to) is cheap.
IMMUTABLE_HEADER='Cache-Control: public, max-age=31536000, immutable'
MANIFEST_HEADER='Cache-Control: public, max-age=60, must-revalidate'

# Pass any flags (e.g. --dry-run) straight through; treat a bare arg as a
# single dataset STEM (its directory data/<stem>/).
EXTRA_FLAGS=()
SINGLE_STEM=""
for arg in "$@"; do
  case "${arg}" in
    --*) EXTRA_FLAGS+=("${arg}") ;;
    *)   SINGLE_STEM="${arg%/}" ;;  # tolerate a trailing slash
  esac
done

# Sync one local tree -> one remote prefix in two header passes.
#   pass (a): immutable assets — every */packs/** and */index/**
#   pass (b): mutable manifest — every */manifest.json
# Each `rclone sync` is scoped by --include so it carries exactly the objects
# that pass's --header-upload applies to. `sync` (not `copy`) so a removed pack
# on the source is pruned remotely; the per-pass --include keeps each pass from
# deleting the OTHER pass's objects.
#
# The include globs are written to match at ANY depth so one function works for
# both call shapes:
#   - all-datasets root (src = .../data):   matches <stem>/packs/**, <stem>/index/**, <stem>/manifest.json
#   - single dataset    (src = .../<stem>): matches packs/**, index/**, manifest.json
# rclone's `**` spans path separators, so `**/packs/**` covers the leading-dir
# case while `packs/**` covers the tree-root case.
sync_tree() {
  local src="$1" dst="$2"

  echo ">> [immutable] ${src} (packs/index) -> ${dst}"
  rclone sync "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    --header-upload "${IMMUTABLE_HEADER}" \
    --include "packs/**" --include "index/**" \
    --include "**/packs/**" --include "**/index/**" \
    "${src}" "${dst}"

  echo ">> [manifest]  ${src} (manifest.json) -> ${dst}"
  rclone sync "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    --header-upload "${MANIFEST_HEADER}" \
    --include "manifest.json" --include "**/manifest.json" \
    "${src}" "${dst}"
}

if [[ -n "${SINGLE_STEM}" ]]; then
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
