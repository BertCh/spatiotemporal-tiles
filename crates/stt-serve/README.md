# stt-serve

Dynamic STT tile server — the `ST_AsMVT` analog for the STT format. It
generates spatiotemporal tiles **on the fly** from a live **PostGIS** table
or a **DuckDB** database file, with no pre-bake step: each
`GET /tiles/{z}/{x}/{y}/{t}.stt` request runs a bbox + time-bucket query and
encodes the rows through the same per-tile encoder `stt-build` uses, so a
served tile is byte-identical to the offline-built tile for the same source
rows. Use it while data is still changing; bake with `stt-build` when it
stops.

> **Not yet published to crates.io** — build from the repo with a backend
> feature (there is no default backend):
>
> ```bash
> git clone https://github.com/BertCh/spatiotemporal-tiles
> cd spatiotemporal-tiles
> cargo build --release -p stt-serve --features postgres   # or: duckdb
> ```

## Example

```bash
# PostGIS backend:
stt-serve --postgres "$DATABASE_URL" --table hurricane_obs --geom-column geom \
          --time-field iso_time --temporal-bucket 7d --min-zoom 3 --max-zoom 8

# DuckDB backend (no database server needed — point at a .duckdb file):
stt-serve --duckdb hurricane.duckdb --table hurricane_obs --geom-column geom \
          --time-field iso_time --temporal-bucket 7d

# Multiple datasets from one process, each under /{name}/…:
stt-serve --postgres "$DATABASE_URL" --config datasets.json
```

Endpoints: `GET /tiles/{z}/{x}/{y}/{t}.stt` (204 for below-threshold tiles),
`GET /metadata.json`, `GET /health`; with `--config`, per-dataset routes plus
a `GET /datasets` catalog. Most of `stt-build`'s per-tile and encoder flags
(`--simplify`, `--quantize-coords`, `--temporal-lod`, budgets, …) are
accepted and applied identically; whole-dataset passes (`--summary-tier`,
`--adaptive-temporal`) are rejected at startup — pre-bake those.

## Relation to the other crates

Shares [`stt-build`](../stt-build)'s tile encoder and `build_options`
module (which is how flag parity is kept), which in turn sits on
[`stt-core`](../stt-core). The offline equivalents of its DB backends are
`stt-build --postgres` / `--duckdb`.

## Docs

- [CLI reference](../../docs/api/cli-reference.md#stt-serve)
- [HTTP protocol spec](../../docs/spec/stt-serve-protocol.md)
- [Database adaptors design + benchmarks](../../docs/roadmap/db-input-adaptors.md)

License: MIT.
