# spatiotemporal-tiles

**SpatioTemporal Tiles (STT)** — a packed, content-addressed tile format for
*animated* geospatial data: every feature carries time, so a renderer can scrub
millions of moving points, trips, and polygons straight off static file
hosting. This is the umbrella crate for the Rust toolchain; the JavaScript
renderers live on npm under [`@poopdeck.gl`](https://www.npmjs.com/org/poopdeck.gl).

## Library

The default features exist so `cargo install` ships the CLIs — as a
*dependency*, disable them and opt into just the pieces you need:

```sh
cargo add spatiotemporal-tiles --no-default-features
```

```rust
use spatiotemporal_tiles as stt;

let reader = stt::core::PackedReader::open("dataset/")?;
```

| feature | adds |
|---|---|
| *(always)* | `stt::core` — archive/tile format reader + writer |
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
cargo install spatiotemporal-tiles
```

installs the four binaries (prebuilt binaries and a shell installer are on the
[GitHub releases page](https://github.com/BertCh/spatiotemporal-tiles/releases)):

| binary | role |
|---|---|
| `stt-build` | build packed STT archives from GeoParquet / PostGIS / DuckDB |
| `stt-optimize` | analyze a dataset and recommend encoder settings; inspect/diff/doctor built tilesets |
| `stt-validate` | validate archives: header, content hashes, Arrow IPC decode, schema |
| `stt-serve` | dynamic per-request tile server over live PostGIS/DuckDB (the `ST_AsMVT` analog) |

By default `stt-serve` gets the PostGIS backend; add the embedded-DuckDB
backend (a heavy C++ compile) with `--features cli`.

Upgrading from a pre-0.1.0 source checkout that installed via
`cargo install --path crates/stt-build`? Those binary names are now owned by
this crate — run `cargo uninstall stt-build stt-optimize stt-validate` once
(or pass `--force`).

New to STT? `docs/guides/csv-quickstart.md` in the repository walks the
whole path — CSV → GeoParquet (one DuckDB command) → `stt-build --auto` →
animated deck.gl map; `docs/guides/tuning-tiles.md` then covers the
measure → interpret → decide tuning loop. See `docs/api/cli-reference.md`
for every flag, and `docs/spec/stt-packed-format.md` for the format
specification.

## License

MIT © Robert Christie
