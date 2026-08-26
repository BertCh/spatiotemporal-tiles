# SpatioTemporal Tiles documentation

SpatioTemporal Tiles (STT) is an open format and Rust toolchain for streaming
vector features by map viewport and time window. The tools here build, inspect,
validate, bundle, and serve STT archives.

Rendering is a separate project: **[poopdeck.gl][pd]** provides the
`@poopdeck.gl/*` TypeScript packages (deck.gl, Three.js, MapLibre, Cesium) and
the [live showcase](https://poopdeck.gl), which is also where the **complete
documentation corpus** is published — this repository's pages and the
renderer's, served together. Several pages below are authored here and vendored
there so the site has one copy of each; the reverse never happens.

[pd]: https://github.com/BertCh/poopdeck.gl

New to the project? Follow these in order:

1. Get an animated map running in five minutes with the
   [quickstart](https://poopdeck.gl/docs/intro/quickstart) — against a hosted
   dataset, no toolchain required.
2. [Choose whether STT fits](./intro/choosing.md).
3. Read the [core concepts](./intro/concepts.md).
4. Build and display your own data with the
   [CSV quickstart](./guides/csv-quickstart.md).

The [glossary](./intro/glossary.md) defines project names and format terms.

## Build data

- [CLI reference](./api/cli-reference.md) — canonical commands and flags for
  `stt-build`, `stt-optimize`, `stt-validate`, `stt-bundle`, `stt-serve`, and
  the repository-only `stt-generate`.
- [CSV quickstart](./guides/csv-quickstart.md) — CSV → GeoParquet → archive →
  animated map.
- [Python guide](./guides/python.md) — GeoPandas, DuckDB, and pyarrow input
  workflows.
- [Data generation](./guides/data-generation.md) — rebuild the bundled showcase
  datasets with `tools/stt-generate`.
- [Tile tuning](./guides/tuning-tiles.md) — analyze and improve archive layout
  without silently dropping data.

Default and `--auto` builds preserve every usable feature. Summary and raster
tiers are explicit coarse-zoom additions, not replacements for the raw tier.

## Deploy and operate

- [Deploying archives](./guides/deploying.md) — object storage, cache policy,
  CORS, and safe publication order.
- [`stt-serve` protocol](./spec/stt-serve-protocol.md) — dynamic service routes,
  response headers, and metadata.
- [Export](./guides/export.md) — move data out of STT-compatible workflows.
- [WebAssembly](./guides/wasm.md) — optional decoder build and integration.

## Architecture

- [System overview](./architecture/system-overview.md) — end-to-end build,
  storage, loading, and rendering pipeline across both repositories.
- [Packed archive performance](./architecture/archive-format-performance.md) —
  layout and generation decisions.
- [Tile payload](./architecture/data-format.md) — Arrow IPC and GeoArrow layer
  frames.

## Normative specification

- [Packed format](./spec/stt-packed-format.md) and
  [manifest schema](./spec/manifest.schema.json)
- [Time model](./spec/time-model.md) and
  [tile matrix set](./spec/tile-matrix-set.json)
- [Tile payload](./architecture/data-format.md)
- [Sidecar assets](./spec/sidecar-assets.md), the
  [scene schema](./spec/scene.schema.json), and the
  [AV palette contract](./spec/av-palettes.json)
- [Conformance](./spec/conformance.md), with portable vectors in
  [`conformance/`](../conformance/README.md)

The specification is authoritative for wire behavior. Current writers emit
packed format v3 and directory codec v6. Reference readers additionally accept
format v2 with directory v5 read-only.

## Rendering

Renderer documentation — the layer catalog, extensions, the tileset and player
APIs, backend capability matrix, and the deck.gl integration — lives with the
code, in the [poopdeck.gl repository][pd] and on
[poopdeck.gl/docs](https://poopdeck.gl/docs).

## AI integration

`@poopdeck.gl/mcp` exposes this toolchain to agents: dataset discovery,
analysis, map composition, and gated `stt-build` / `stt-validate` /
`stt-generate` operations. It ships from the [poopdeck.gl repository][pd] and
resolves the `stt-*` binaries off `PATH`, so it works against any installed
version of this toolchain. See
[the AI suite guide](https://poopdeck.gl/docs/guides/ai-suite).

The repository's [`AGENTS.md`](../AGENTS.md) is the orientation and routing
document for coding agents.
