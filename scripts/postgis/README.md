# PostGIS integration — local benchmark harness

Reproduces the numbers in [`docs/roadmap/postgis-integration.md`](../../docs/roadmap/postgis-integration.md):
PostGIS as an `stt-build` input source, and `stt-serve` generating STT tiles on
the fly from a live PostGIS table.

## Prerequisites

- Colima (or any Docker) — `setup.sh` starts a native-arm64 PostGIS container.
- Release binaries built with the `postgres` feature:
  ```bash
  cargo build --release --bin stt-build --features postgres
  cargo build --release -p stt-serve -p stt-validate
  cargo build --release -p stt-build --example tile_info
  ```
- The points-Parquet baseline needs `pyarrow` (any venv): `pip install pyarrow`.

## Workflow

```bash
export PGURL="postgresql://postgres:postgres@localhost:5432/stt"

# 1. PostGIS up + IBTrACS loaded into hurricane_obs
scripts/postgis/setup.sh
scripts/postgis/load-ibtracs.sh

# 2. Parquet baseline for the ingest A/B (file vs PostGIS, identical columns)
docker exec stt-postgis psql -U postgres -d stt -tAc "\copy (SELECT lon, lat,
  to_char(iso_time,'YYYY-MM-DD\"T\"HH24:MI:SS') AS iso_time, sid, season, basin,
  name, nature, wmo_wind, wmo_pres, usa_sshs FROM hurricane_obs
  WHERE iso_time >= '1970-01-01' ORDER BY sid, iso_time)
  TO STDOUT WITH (FORMAT csv, HEADER true)" > /tmp/h.csv
python scripts/postgis/export-points-parquet.py /tmp/h.csv scratch-postgis/hurricane_points.parquet

# 3. Ingest A/B (parity + timing)
SCRATCH=scratch-postgis scripts/postgis/bench-ingest.sh

# 4. Dynamic serve benchmark — start the server, then bench it vs static files
stt-serve --postgres "$PGURL" --table hurricane_obs --geom-column geom \
  --time-field iso_time --temporal-bucket 7d --min-zoom 3 --max-zoom 8 &
SCRATCH=scratch-postgis scripts/postgis/bench-serve.sh
```

`SCRATCH` is where intermediate archives / tiles / the Parquet baseline live
(any scratch dir). Teardown: `docker rm -f stt-postgis`.

## Files

| file | role |
|---|---|
| `setup.sh` | Colima + PostGIS container + `CREATE EXTENSION postgis` |
| `load-ibtracs.sh` | `data/ibtracs.csv` → typed `hurricane_obs` (Point/4326 + indexes) |
| `export-points-parquet.py` | CSV dump → Parquet (the file-ingest baseline) |
| `bench-ingest.sh` | file vs PostGIS ingest: parity (via `stt-validate`) + wall-clock |
| `bench-serve.sh` | materialize tiles, then load-test dynamic vs static |
| `bench_serve.py` | concurrent HTTP latency driver (percentiles, server-gen time) |
| `gen_tile_urls.py` | sample points → `(z,x,y,t)` tile paths that contain data |
