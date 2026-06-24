#!/usr/bin/env bash
# Bring up a local PostGIS for the integration benchmark: start Colima (if it
# isn't running) and a native-arch postgis container, then enable PostGIS.
# Idempotent — safe to re-run. Teardown: docker rm -f stt-postgis
#
# Usage: scripts/postgis/setup.sh
# Env:   CONTAINER (default stt-postgis), PORT (5432), DB (stt),
#        IMAGE (imresamu/postgis:16-3.4 — multi-arch incl. arm64; the official
#        postgis/postgis is amd64-only and runs emulated on Apple Silicon)
set -euo pipefail

CONTAINER="${CONTAINER:-stt-postgis}"
PORT="${PORT:-5432}"
DB="${DB:-stt}"
IMAGE="${IMAGE:-imresamu/postgis:16-3.4}"

if command -v colima >/dev/null 2>&1; then
  if ! docker info >/dev/null 2>&1; then
    echo "==> Starting Colima VM"
    colima start --cpu 4 --memory 6
  fi
fi
docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "==> Container $CONTAINER exists; (re)starting"
  docker start "$CONTAINER" >/dev/null
else
  echo "==> Launching $IMAGE as $CONTAINER on :$PORT"
  docker run -d --name "$CONTAINER" --platform linux/arm64 \
    -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB="$DB" \
    -p "$PORT:5432" "$IMAGE" >/dev/null
fi

echo -n "==> Waiting for Postgres"
for _ in $(seq 1 60); do
  if docker exec "$CONTAINER" pg_isready -U postgres -d "$DB" >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  echo -n "."
  sleep 1
done

docker exec "$CONTAINER" psql -U postgres -d "$DB" -c "CREATE EXTENSION IF NOT EXISTS postgis;" >/dev/null
echo "==> PostGIS: $(docker exec "$CONTAINER" psql -U postgres -d "$DB" -tAc 'SELECT postgis_version();')"
echo "==> Connection: postgresql://postgres:postgres@localhost:$PORT/$DB"
