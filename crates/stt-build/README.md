# stt-build

The CLI that builds **SpatioTemporal Tiles (STT)** datasets. It reads a
GeoParquet file with a WKB geometry column (or separate `lon`/`lat` columns)
plus a timestamp column — or, behind opt-in cargo features, a live
**PostGIS** or **DuckDB** query — tiles the features across zooms and
temporal buckets, and writes the packed format: a dataset directory of
`manifest.json`, one `index/<blake3>.sttd` directory object, and
content-addressed `packs/<blake3>.sttp` objects (immutable,
forever-cacheable). Simplification, temporal LOD pyramids, summary tiers
(H3/Quadbin), coordinate/attribute quantization, and per-tile budgets are
all flags away.

> **Internal implementation crate** of
> [`spatiotemporal-tiles`](https://crates.io/crates/spatiotemporal-tiles):
> this is the tiler/encoder *library* (the facade's `build` module). The
> `stt-build` CLI ships with the facade:
>
> ```bash
> cargo install spatiotemporal-tiles --features cli
> ```
>
> Database inputs stay feature-gated (`postgres` / `duckdb`) on both the
> library and the facade.

## Example

```bash
# Auto-tuned build (zoom range + temporal bucket picked by stt-optimize):
stt-build -i earthquakes.parquet -o earthquakes.stt \
  --time-field time --time-format unix-ms \
  --auto

# Trajectories with simplification and a temporal-LOD pyramid:
stt-build -i taxi-trips.parquet -o taxi-trips.stt \
  --time-field start_time --end-time-field end_time \
  --time-format unix-ms \
  --simplify \
  --temporal-bucket 30m \
  --temporal-lod 1d@8,30d@4

# Straight from PostGIS (--features postgres), no export step:
stt-build --postgres "$DATABASE_URL" --table public.hurricane_obs \
  --geom-column geom --time-field iso_time \
  -o hurricanes.stt
```

`-o foo.stt` strips the extension and produces `foo/{manifest.json,index/,packs/}`.
Pass `--publish` for a deploy-ready build (zstd 19).

## Relation to the other crates

Encoding, packing, and the directory come from [`stt-core`](../stt-core);
`--auto` calls into [`stt-optimize`](../stt-optimize);
[`stt-generate`](../stt-generate) shells out to this binary for every bundled
dataset; [`stt-serve`](../stt-serve) reuses the same per-tile encoder to
serve tiles on the fly; gate outputs in CI with
[`stt-validate`](../stt-validate).

## Docs

- [Full flag reference](../../docs/api/cli-reference.md#stt-build)
- [Packed format spec](../../docs/spec/stt-packed-format.md)
- [Database input adaptors](../../docs/roadmap/db-input-adaptors.md)
- [Data generation guide](../../docs/guides/data-generation.md)

License: MIT.
