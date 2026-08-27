# SpatioTemporal Tiles documentation

SpatioTemporal Tiles (STT) is an open format and a Rust toolchain for streaming
vector features by map viewport **and** time window. Five CLIs build, analyze,
validate, bundle and serve STT archives; the repository-only `stt-generate`
rebuilds the reference datasets.

**Rendering is a separate project.** The `@poopdeck.gl/*` TypeScript packages
(deck.gl, Three.js, MapLibre, Cesium), the playback clock and the
[live showcase](https://poopdeck.gl) live in [poopdeck.gl][pd] — which is also
where the **complete documentation corpus** is published, this repository's
pages and the renderer's served together. The pages listed in
[`.corpus.json`](./.corpus.json) are authored here and vendored there; the
reverse never happens.

[pd]: https://github.com/BertCh/poopdeck.gl

## New here

1. Get an animated map running against a hosted dataset, no toolchain required —
   the [five-minute quickstart](https://poopdeck.gl/docs/intro/quickstart).
2. [Choose STT and a deployment](./intro/choosing.md).
3. Read the [core concepts](./intro/concepts.md) — the `(z, x, y, bucket)` tile,
   the packed archive, playback.
4. Build and display your own data with the
   [CSV quickstart](./guides/csv-quickstart.md).

The [glossary](./intro/glossary.md) fixes the spelling and meaning of every
project, format, archive and API name.

## Build data

- [CLI reference](./api/cli-reference.md) — every flag of `stt-build`,
  `stt-optimize`, `stt-validate`, `stt-bundle`, `stt-serve`, and `stt-generate`.
- [CSV quickstart](./guides/csv-quickstart.md) — CSV → GeoParquet → archive →
  animated map, with no tile server anywhere.
- [Python guide](./guides/python.md) — three ways to produce the GeoParquet
  `stt-build` reads (GeoPandas, plain DuckDB, pyarrow), plus trajectories.
- [Tile tuning](./guides/tuning-tiles.md) — the measured loop: `analyze` /
  `recommend`, `--auto`, then `inspect` / `doctor` / `diff` / `order-audit`,
  and baking render defaults with `--style-hints`.
- [Data generation](./guides/data-generation.md) — the bundled `stt-generate`
  datasets (inventory:
  [`stt-generate-datasets.json`](./spec/stt-generate-datasets.json)), custom
  data through `stt-build`, and the AV scene-bundle extractors.
- [Export](./guides/export.md) — `stt-optimize export` reads a built archive
  back out as GeoParquet 1.1, so an archive is a render tier over your
  lakehouse rather than a place data goes to die.

## Deploy and serve

- [Deploying a dataset](./guides/deploying.md) — the two `Cache-Control`
  regimes, the Cloudflare R2 reference deploy, CORS, how to verify the edge,
  and the packs-first / manifest-last publication order.
- [`stt-serve` protocol](./spec/stt-serve-protocol.md) — the HTTP surface of the
  per-request tile server over PostGIS or DuckDB: routes, status codes,
  headers, `/metadata.json`, and its generation-parity contract with
  `stt-build`.
- [WebAssembly](./guides/wasm.md) — `stt-core` compiled to a browser decoder:
  the three entry points, the size budget, and what it deliberately does not do.

## Architecture

- [System overview](./architecture/system-overview.md) — the Rust toolchain and
  the TypeScript stack end to end, across both repositories.
- [Archive layout and generation policy](./architecture/archive-format-performance.md) —
  the one current contract, the physical shape each workload gets, and the
  no-thinning generation rules.

## Normative specification

- [Packed format](./spec/stt-packed-format.md) — on-bucket layout, the
  `manifest.json` schema, directory codec v6, pack-cutting, reader flow, and
  versioning. Machine-checkable:
  [`manifest.schema.json`](./spec/manifest.schema.json).
- [Time model](./spec/time-model.md) — the time base, instants vs intervals,
  fixed-width buckets, the coarser-bucket LOD pyramid, and read-time pruning.
  The spatial grid is [`tile-matrix-set.json`](./spec/tile-matrix-set.json).
- [Tile payload](./architecture/data-format.md) — the Arrow IPC / GeoArrow layer
  frame, its columns, and its quantization forms.
- [Sidecar assets](./spec/sidecar-assets.md) — the scene-bundle profile for
  co-registered multi-stream and approximately-georeferenced data, with
  [`scene.schema.json`](./spec/scene.schema.json) and the
  [AV palette contract](./spec/av-palettes.json).
- [Conformance](./spec/conformance.md) — what a conformant writer and reader
  must do, plus the portable kit in [`conformance/`](../conformance/README.md)
  that checks one.

The specification is authoritative for wire behavior, and it describes exactly
one current shape: **packed format v3, directory codec v6, layer frame v2** —
three independently-versioned axes, not one number.

## Roadmap and open work

- [Roadmap register](./roadmap/README.md) — the decision records (rationale,
  measured baselines, negative results) and the single backlog of open work for
  the format, the tiler, the optimizer and the published data fleet. Renderer
  and playback work lives in poopdeck.gl's own register.
- [Repository split](./roadmap/repo-split-2026-08.md) — the two-repository
  contract: what each side owns, what crosses the seam, and why roadmap pages
  are dated registers rather than descriptions of current behavior.

## Rendering and agents

Renderer documentation — the layer catalog, extensions, the tileset and player
APIs, the backend capability matrix, and the deck.gl integration — lives with
the code, in the [poopdeck.gl repository][pd] and on
[poopdeck.gl/docs](https://poopdeck.gl/docs).

`@poopdeck.gl/mcp` and the `poopdeck-ai` skills expose this toolchain to agents
— dataset discovery, analysis, map composition, and gated `stt-build` /
`stt-validate` / `stt-generate` runs. They ship from the
[poopdeck.gl repository][pd] and resolve the `stt-*` binaries off `PATH`, so
they work against any installed version of this toolchain. This repository's
[`AGENTS.md`](../AGENTS.md) is the orientation and routing document for coding
agents working in this checkout.
