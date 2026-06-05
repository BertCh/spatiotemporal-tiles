#!/usr/bin/env bash
#
# Sync showcase tile archives to a Cloudflare R2 bucket.
#
# Reads credentials from .env.r2 (see .env.r2.example). rclone is configured
# entirely from env vars, so nothing is written to ~/.config/rclone/rclone.conf
# and no secret is persisted on disk by this script.
#
# Usage:
#   scripts/r2-sync.sh                 # sync all of public/data -> r2:<bucket>/data
#   scripts/r2-sync.sh --dry-run       # show what would change, transfer nothing
#   scripts/r2-sync.sh flights.stt     # upload a single archive (e.g. after regen)
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

# Tuning for multi-GB .stt files: large multipart chunks, parallel parts/files.
COMMON_FLAGS=(
  --progress
  --transfers 4
  --checkers 8
  --s3-chunk-size 64M
  --s3-upload-concurrency 8
  --s3-no-check-bucket
  # .stt archives change only when regenerated. Cache a day in the browser and
  # let the Cloudflare edge revalidate via ETag. After re-syncing a CHANGED file,
  # purge it from the Cloudflare cache (filenames are reused, so the edge would
  # otherwise serve stale bytes until TTL). Bump max-age if filenames get versioned.
  --header-upload "Cache-Control: public, max-age=86400"
)

# Pass any flags (e.g. --dry-run) straight through; treat a bare *.stt arg as a
# single-file upload.
EXTRA_FLAGS=()
SINGLE_FILE=""
for arg in "$@"; do
  case "${arg}" in
    --*) EXTRA_FLAGS+=("${arg}") ;;
    *)   SINGLE_FILE="${arg}" ;;
  esac
done

if [[ -n "${SINGLE_FILE}" ]]; then
  echo ">> uploading ${SINGLE_FILE} -> r2:${R2_BUCKET}/data/"
  rclone copyto "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    "${DATA_DIR}/${SINGLE_FILE}" "r2:${R2_BUCKET}/data/${SINGLE_FILE}"
else
  echo ">> syncing ${DATA_DIR} -> r2:${R2_BUCKET}/data"
  rclone sync "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    "${DATA_DIR}" "r2:${R2_BUCKET}/data"
fi

echo ">> done."
