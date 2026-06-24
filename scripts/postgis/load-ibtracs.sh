#!/usr/bin/env bash
# Load data/ibtracs.csv (NOAA IBTrACS storm best-track) into the PostGIS
# `hurricane_obs` table used by the PostGIS-integration benchmark.
#
# IBTrACS CSV has TWO header rows (column names, then units) and 163 columns.
# We stage every column as text, then project the handful we need into a typed
# table with a 4326 Point geometry + spatial / temporal indexes.
#
# Usage: scripts/postgis/load-ibtracs.sh
# Env:   PGURL (default postgresql://postgres:postgres@localhost:5432/stt)
#        CSV   (default data/ibtracs.csv)
#        CONTAINER (default stt-postgis) — used for `docker exec` psql
set -euo pipefail

CSV="${CSV:-data/ibtracs.csv}"
CONTAINER="${CONTAINER:-stt-postgis}"
PSQL=(docker exec -i "$CONTAINER" psql -U postgres -d stt -v ON_ERROR_STOP=1)

[ -f "$CSV" ] || { echo "missing $CSV" >&2; exit 1; }

echo "==> Building staging schema from $CSV header (163 text columns)"
# Lowercase header, comma -> newline, wrap each as "name" text, comma-join.
COLS=$(head -1 "$CSV" \
  | tr 'A-Z,' 'a-z\n' \
  | awk 'NF{printf "%s\"%s\" text", (NR>1?", ":""), $0}')

"${PSQL[@]}" <<SQL
DROP TABLE IF EXISTS ibtracs_raw;
CREATE TABLE ibtracs_raw ($COLS);
SQL

echo "==> Streaming rows (skipping the 2 header lines) into ibtracs_raw"
# tail -n +3 drops the column-name row AND the units row.
tail -n +3 "$CSV" | "${PSQL[@]}" -c "\copy ibtracs_raw FROM STDIN WITH (FORMAT csv)"

echo "==> Projecting typed hurricane_obs (Point/4326) + indexes"
"${PSQL[@]}" <<'SQL'
DROP TABLE IF EXISTS hurricane_obs;
CREATE TABLE hurricane_obs AS
SELECT
    sid,
    season,
    basin,
    nullif(trim(name), '')              AS name,
    nature,
    iso_time::timestamptz               AS iso_time,
    nullif(trim(lat), '')::float8       AS lat,
    nullif(trim(lon), '')::float8       AS lon,
    nullif(trim(wmo_wind), '')::float8  AS wmo_wind,
    nullif(trim(wmo_pres), '')::float8  AS wmo_pres,
    nullif(trim(usa_sshs), '')::int     AS usa_sshs
FROM ibtracs_raw
WHERE nullif(trim(lat), '') IS NOT NULL
  AND nullif(trim(lon), '') IS NOT NULL
  AND nullif(trim(iso_time), '') IS NOT NULL;

ALTER TABLE hurricane_obs ADD COLUMN geom geometry(Point, 4326);
UPDATE hurricane_obs SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);

CREATE INDEX hurricane_obs_geom_gist ON hurricane_obs USING gist (geom);
CREATE INDEX hurricane_obs_time_btree ON hurricane_obs (iso_time);
ANALYZE hurricane_obs;

DROP TABLE ibtracs_raw;
SQL

echo "==> Loaded:"
"${PSQL[@]}" -tAc "SELECT count(*) || ' obs, ' ||
  to_char(min(iso_time),'YYYY-MM-DD') || ' .. ' ||
  to_char(max(iso_time),'YYYY-MM-DD') AS summary FROM hurricane_obs;"
"${PSQL[@]}" -tAc "SELECT 'bbox ' || ST_Extent(geom)::text FROM hurricane_obs;"
