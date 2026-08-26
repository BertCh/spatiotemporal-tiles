# SpatioTemporal Tiles (STT)

> Stream full-fidelity vector data by map viewport **and** time window from a
> static host or live database.

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](./LICENSE)
[![MSRV](https://img.shields.io/crates/msrv/spatiotemporal-tiles?label=msrv)](./Cargo.toml)

**STT** is the open format and Rust toolchain — this repository.
**[poopdeck.gl](https://github.com/BertCh/poopdeck.gl)** is the TypeScript
rendering ecosystem and [live showcase](https://poopdeck.gl). Together they turn
GeoParquet, PostGIS, or DuckDB data into interactive points, paths, polygons,
trips, flows, and events over time. They live apart on purpose, and the seam
between them is the archive on disk: a format change reaches every renderer at
once, and a renderer change needs no format release
([the split record](./docs/roadmap/repo-split-2026-08.md)).

An archive is a small `manifest.json`, a compact directory, and immutable,
content-addressed packs. A client requests only the spatial tiles and time
buckets it needs; ordinary HTTP Range support and CDN caching are enough for a
static deployment.

## Is STT a fit?

Choose STT when you need to:

- animate or scrub large **vector** datasets without loading the full history;
- publish immutable data to object storage or a CDN;
- serve the same archive to deck.gl, Three.js, or MapLibre applications; or
- build tiles dynamically from PostGIS or DuckDB with the same client contract.

Choose another format when your data is a time-varying raster or datacube (use
GeoZarr or COG), a small GeoJSON file already performs well, or you need edits
to individual records rather than immutable published snapshots. See
[Choosing STT](./docs/intro/choosing.md) for a fuller decision guide.

## Quick start

If you want to _see_ one first, the
[five-minute Quickstart](https://poopdeck.gl/docs/intro/quickstart) renders a
hosted dataset — half a million streaming earthquakes — inside an ordinary app,
with no account, no tile server and no Rust toolchain. It is the recommended
front door, and it needs nothing from this repository.

To _build_ one, read on.

### Build an archive

[From CSV to an Animated Map](./docs/guides/csv-quickstart.md) is the complete
tutorial. The short version:

```bash
cargo install spatiotemporal-tiles

stt-build \
  --input data.parquet \
  --output tiles \
  --time-field timestamp \
  --time-format unix-ms \
  --min-zoom 0 \
  --max-zoom 8 \
  --temporal-bucket 1h

stt-validate tiles
```

Use `stt-build --auto` when you want the toolchain to recommend the zoom range,
temporal bucket, and compression. Explicit flags still win.

Builds preserve every usable feature by default. STT does **not** silently
sample, thin, or aggregate data to meet a byte target. Control size first with
an honest maximum zoom and temporal bucketing. Summary and raster tiers are
explicit, coarse-zoom additions; they do not replace the raw tier.

The installed package provides five CLIs:

| Command        | Purpose                                                |
| -------------- | ------------------------------------------------------ |
| `stt-build`    | Build packed archives from GeoParquet/PostGIS/DuckDB   |
| `stt-optimize` | Analyze inputs and inspect, lint, or compare archives  |
| `stt-validate` | Verify integrity, schemas, decoding, and time metadata |
| `stt-bundle`   | Pack or unpack a single-file `.sttb` interchange file  |
| `stt-serve`    | Serve STT tiles dynamically from a live database       |

`stt-generate` is a separate, repository-only tool for rebuilding the reference
datasets; it lives in [`tools/stt-generate`](./tools/stt-generate) and has its
own cargo workspace.

### Render it

Rendering is [poopdeck.gl](https://github.com/BertCh/poopdeck.gl): seven
published `@poopdeck.gl/*` packages with deck.gl, Three.js/WebGPU, MapLibre and
Cesium backends, a playback clock, and React bindings. An archive built here
streams into any of them unchanged — the manifest is the whole contract.

```bash
npm install @poopdeck.gl/layers @poopdeck.gl/playback
```

## How the pieces fit

```text
   ─── this repository ─────────────┐  ─── poopdeck.gl ──────────────
                                    │
GeoParquet / PostGIS / DuckDB       │
            │                       │
            ├─ stt-build ──> static packed archive ──> CDN/object storage
            └─ stt-serve ──> dynamic tile endpoint
                                    │            │
                                    │            v
                                    │    @poopdeck.gl/core
                                    │            │
                                    │  deck.gl / Three.js / MapLibre / Cesium
                                    │            │
                                    │    playback + React UI
```

The archive manifest is the contract. It declares the temporal model,
capabilities, pack table, and optional style hints; clients should inspect it
rather than infer dataset behavior. Packs and directories are content-addressed
and must never be rewritten in place.

## Project status

The current release line is **0.7.0** and remains pre-1.0. Writers produce
packed format v3 with directory codec v6; reference readers also open published
format-v2/directory-v5 archives read-only.
[`project-status.json`](./project-status.json) is the machine-readable version
of that paragraph, and every field in it is proved against its source by a CI
gate.

> Since the 2026-08-26 split, the crates.io and npm version numbers are **not**
> in lockstep. They agree at 0.7.0 by history, not by promise; what relates the
> two stacks is the archive's `formatVersion`.

See [Status, support, and compatibility](https://poopdeck.gl/docs/intro/status-and-support)
before depending on a pre-1.0 API, and the [project changelog](./CHANGELOG.md)
before upgrading.

## Documentation

- [Documentation index](./docs/README.md)
- [Core concepts](./docs/intro/concepts.md)
- [Choose a deployment, backend, and layer](./docs/intro/choosing.md)
- [System overview](./docs/architecture/system-overview.md)
- [CLI reference](./docs/api/cli-reference.md)
- [Packed-format specification](./docs/spec/stt-packed-format.md) and
  [conformance vectors](./conformance/README.md)
- [Deployment guide](./docs/guides/deploying.md)
- [Renderer documentation](https://poopdeck.gl/docs) ·
  [Live demos](https://poopdeck.gl/demos)

The published site serves this repository's pages and the renderer's together;
several of the pages above are authored here and vendored downstream so there is
one copy of each.

AI coding agents should start with [`AGENTS.md`](./AGENTS.md). It contains the
repository map, invariant rules, and routing table to canonical documentation.

## Development and contributing

The root is a Cargo workspace. The common verification commands are:

```bash
cargo test --workspace
cargo test --workspace --all-features     # incl. duckdb, postgres, projection
cargo fmt --all -- --check

# The reference-dataset generator has its own workspace.
cargo test --manifest-path tools/stt-generate/Cargo.toml

# Repository gates (Node; no packages are built here)
pnpm install
pnpm project:check && pnpm docs:links && pnpm versions:check && pnpm citations
```

Read [CONTRIBUTING.md](./CONTRIBUTING.md) before submitting a change. Project
policies cover [support](./SUPPORT.md), [governance](./GOVERNANCE.md), the
[Code of Conduct](./CODE_OF_CONDUCT.md), and private
[security reporting](./SECURITY.md).

## License

MIT © Robert Christie
