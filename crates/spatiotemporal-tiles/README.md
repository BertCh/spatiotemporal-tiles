# spatiotemporal-tiles

**SpatioTemporal Tiles (STT)** — a packed, content-addressed tile format for
*animated* geospatial data: every feature carries time, so a renderer can scrub
millions of moving points, trips, and polygons straight off static file
hosting. This is the umbrella crate for the Rust toolchain; the JavaScript
renderers live on npm under [`@poopdeck.gl`](https://www.npmjs.com/org/poopdeck.gl).

## Library

```sh
cargo add spatiotemporal-tiles
```

```rust
use spatiotemporal_tiles as stt;

let reader = stt::core::PackedReader::open("dataset/")?;
```

| feature | adds |
|---|---|
| *(default)* | `stt::core` — archive/tile format reader + writer |
| `build` | `stt::build` — the tiler/encoder library |
| `optimize` | `stt::optimize` — dataset analysis + encoding recommendations |
| `postgres`, `duckdb` | database input sources for `build` |
| `projection` | advanced CRS support (requires system libproj) |

The implementation crates ([`stt-core`](https://crates.io/crates/stt-core),
[`stt-build`](https://crates.io/crates/stt-build),
[`stt-optimize`](https://crates.io/crates/stt-optimize)) are published as
internal dependencies of this facade — like `bevy_ecs` under `bevy`. Depend on
them directly only if you need exactly one piece; they track this crate's
version in lockstep.

## CLI tools

```sh
cargo install spatiotemporal-tiles --features cli
```

installs the four binaries (prebuilt binaries and a shell installer are on the
[GitHub releases page](https://github.com/BertCh/spatiotemporal-tiles/releases)):

| binary | role |
|---|---|
| `stt-build` | build packed STT archives from GeoJSON / GeoParquet / PostGIS / DuckDB |
| `stt-optimize` | analyze a dataset and recommend encoder settings |
| `stt-validate` | validate archives: header, content hashes, Arrow IPC decode, schema |
| `stt-serve` | dynamic per-request tile server over live PostGIS/DuckDB (the `ST_AsMVT` analog) |

A lighter `stt-serve` without the bundled-DuckDB compile:
`cargo install spatiotemporal-tiles --features build-cli,optimize-cli,validate-cli,serve-postgres`.

See `docs/api/cli-reference.md` in the repository for every flag, and
`docs/spec/stt-packed-format.md` for the format specification.

## License

MIT © Robert Christie
