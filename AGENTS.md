# AGENTS.md

Orientation for any AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, …)
dropped into this checkout. `AGENTS.md` is the cross-harness convention that
Cursor, Codex, and Gemini CLI read automatically; Claude Code users additionally
get the `poopdeck:poopdeck-ai` plugin (see [How agents get help](#how-agents-get-help)).
Read this first, then follow the routing table to the canonical docs.

## What this repo is

**SpatioTemporal Tiles (STT)** — the open **format** and the Rust **toolchain**
that writes it. A dataset is a tiny `manifest.json` plus many immutable,
content-addressed **pack** objects (`.stt` archive) that combine a spatial tile
pyramid with a temporal axis — each tile is addressed by `(zoom, x, y,
time-bucket)`. Five Rust CLIs (`stt-build`, `stt-optimize`, `stt-serve`,
`stt-validate`, `stt-bundle`) build / analyze / serve those archives, and the
repo-only `stt-generate` rebuilds the reference datasets. Scope is **vector**
spatiotemporal data (points, paths, polygons, trips, flows, events);
time-varying rasters/datacubes are out of scope.

> **⚠️ The renderers are NOT in this repository.** The `@poopdeck.gl/*`
> TypeScript packages — the deck.gl / Three.js / MapLibre / Cesium backends, the
> playback clock, the React bindings, the MCP server and the showcase site —
> live in [BertCh/poopdeck.gl][pd]. Do not look for `packages/` or
> `examples/showcase` here, and do not propose fixing a rendering bug from this
> checkout. The two repositories meet at the archive on disk;
> `docs/roadmap/repo-split-2026-08.md` is the contract.
>
> **What this repo owes downstream, it owes as artifacts.** 24 doc pages,
> `conformance/vectors/`, `docs/spec/av-palettes.json`,
> `docs/spec/stt-generate-datasets.json` and `project-status.json` are vendored
> into poopdeck.gl and byte-gated there. Change them here; the downstream copy
> follows. Never the other way round.

[pd]: https://github.com/BertCh/poopdeck.gl

## Mental model (the pipeline)

```
GeoParquet / PostGIS / DuckDB          ─── this repository ──────────────────
        │  stt-build      → packed STT archive (manifest.json + index/*.sttd + packs/*.sttp)
        │  stt-optimize   → analyze / recommend / inspect / doctor / diff / lint the archive
        │  stt-serve      → dynamic per-request tile server (or publish the static dir to R2/CDN)
        ▼                                  ─── BertCh/poopdeck.gl ───────────
  @poopdeck.gl/core       → STTArchive reader (HTTP Range) + decoder pool + tileset + render kernel
  @poopdeck.gl/layers     → SpatioTemporalLayer (deck.gl)   ← the primary renderer
  @poopdeck.gl/{three,maplibre,cesium}                       ← alternate backends
  @poopdeck.gl/playback + /react                             ← scrub / play clock + UI
```

A cold client load is 1 `manifest.json` + 1 directory object + N pack Range
requests; warm is served entirely from edge cache. Tile payloads are Apache
Arrow IPC with GeoArrow-encoded geometry.

## Ground rules (read before recommending anything)

- **NO DEFAULT THINNING.** Default and `--auto` builds preserve every usable
  feature; never recommend thinning, sampling, or aggregation merely to hit a
  byte budget. To manage size, first **clamp the zoom range** (keep
  `--max-zoom` honest) and use **temporal bucketing** (`--temporal-bucket`).
  Expert users may explicitly opt into the documented per-tile budget controls,
  but those controls must remain off by default and report what they remove.
  The **summary** (H3/Quadbin) and **raster** tiers are _opt-in coarse-zoom aids_
  for very large datasets, **not** a substitute for the raw tier.
- **The archive/manifest is the contract.** `manifest.json` carries capabilities,
  the temporal block, the pack table, and (if built with `--style-hints`)
  per-property percentiles. Read it before guessing (`describe_dataset`, or open
  the file).
- Packs and the directory are **immutable and content-addressed** (blake3 =
  filename); only the small `manifest.json` is mutable. Never rewrite a pack in
  place.

## Routing table — "to do X, look here"

| You want to…                                                              | Tool / package                                                   | Canonical docs                                                 |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------- |
| Turn **your own** GeoParquet / PostGIS / DuckDB into a `.stt`             | `stt-build` (`--auto` to infer knobs)                            | `docs/guides/csv-quickstart.md`, `docs/api/cli-reference.md`   |
| Get build knobs recommended from a source file                            | `stt-optimize` (powers `stt-build --auto`)                       | `docs/api/cli-reference.md`, `docs/guides/tuning-tiles.md`     |
| Shrink / lint / diff / audit a **built** archive                          | `stt-optimize inspect`/`doctor`/`diff`/`order-audit`             | `docs/guides/tuning-tiles.md`                                  |
| Serve tiles dynamically off a live DB                                     | `stt-serve` (PostGIS/DuckDB, axum)                               | `docs/api/cli-reference.md`, `docs/spec/stt-serve-protocol.md` |
| Publish a static archive to a CDN/R2                                      | sync the dir tree; `scripts/r2-sync.sh` sets cache headers       | `docs/guides/deploying.md`                                     |
| Find the built archive fleet on this machine                              | `data-fleet/` (76 GB, untracked; was the showcase's public/data) | `docs/guides/deploying.md`                                     |
| Check integrity / decode / schema / temporal consistency                  | `stt-validate` (CI-suitable)                                     | `docs/api/cli-reference.md`                                    |
| Pack/unpack a single-file `.sttb` interchange bundle                      | `stt-bundle`                                                     | `docs/api/cli-reference.md`                                    |
| Generate a **bundled reference** dataset (earthquakes, drifters, GTFS, …) | `stt-generate`                                                   | `docs/guides/data-generation.md`                               |
| Regenerate the conformance vectors                                        | `conformance/make-vectors.sh`                                    | `conformance/README.md`, `docs/spec/conformance.md`            |
| **Anything about rendering** — read, draw, animate, pick, the showcase    | the [poopdeck.gl repository][pd]                                 | <https://poopdeck.gl/docs>                                     |

The `stt-*` CLIs are the workhorses (think `wrangler`); their canonical flag
surface is **`docs/api/cli-reference.md`**, which is vendored downstream — edit
it here.

## Where the code lives

```
crates/                 # Rust workspace (5 members)
  stt-core/             # archive + Arrow tile format library (every CLI links it)
  stt-build/            # GeoParquet / PostGIS / DuckDB → packed .stt (library)
  stt-optimize/         # input analysis + archive inspect/doctor/diff (powers --auto)
  stt-wasm/             # stt-core → WebAssembly decoder; publish = false (docs/guides/wasm.md)
  spatiotemporal-tiles/ # umbrella crate: re-exports the libs + ships the published CLIs
    src/bin/            #   stt-build, stt-optimize, stt-validate, stt-bundle, stt-serve
tools/stt-generate/     # bundled showcase-dataset generators. NOT a root-workspace
                        #   member: its own [workspace] + lockfile keep its higher
                        #   MSRV off the published crates, so `-p stt-generate` from
                        #   the root does not resolve — `cargo install --path` it
packages/               # TypeScript (@poopdeck.gl/*)
  core/                 # STTArchive reader, decoder pool, OPFS cache, render kernel
  layers/               # deck.gl backend (primary)
  three/ maplibre/ cesium/   # alternate renderer backends
  playback/ react/      # clock + governor + React UI
  mcp/                  # MCP server (@poopdeck.gl/mcp)
poopdeck:examples/showcase/      # interactive demo app (deck.gl + MapLibre + Three)
docs/                   # spec, API reference, guides, architecture
poopdeck:poopdeck-ai/            # Claude Code plugin (Agent Skills + MCP server wiring)
```

## Where the docs live — key entry points

- `docs/intro/concepts.md` — the space×time tile model, packed archives, playback.
- `docs/intro/choosing.md` — static vs served; which renderer.
- `docs/intro/status-and-support.md` — maturity tiers, compatibility window, and
  support expectations; `project-status.json` is the machine-readable summary.
- `docs/intro/glossary.md` — canonical product, format, archive, and API names.
- `docs/architecture/system-overview.md` — how the pieces fit end to end.
- `docs/api/cli-reference.md` — canonical flags for every `stt-*` CLI.
- `docs/spec/conformance.md` + `conformance/README.md` — what a conformant
  reader/writer is, and the portable vectors that check one.
- `docs/spec/stt-packed-format.md` — the on-disk format (manifest + packs +
  directory v6; machine-checkable schema: `docs/spec/manifest.schema.json`).
- `docs/architecture/data-format.md` — the per-tile Arrow layer-frame encoding.
- `docs/guides/` — task guides (`csv-quickstart`, `tuning-tiles`, `deploying`,
  `export`, `python`, `data-generation`, `wasm`, `ai-suite`).

## Build / test basics

One **cargo** workspace, plus a handful of Node gates. Infer exact flags from
`Cargo.toml` / `package.json`; the verified commands are:

```bash
cargo build --release          # CLI binaries → target/release/{stt-build,stt-optimize,...}
cargo test --workspace
cargo test --workspace --all-features       # incl. duckdb, postgres, projection
cargo test --manifest-path tools/stt-generate/Cargo.toml   # its own workspace

pnpm install                   # only for the gates below; nothing is built
pnpm project:check             # project-status.json vs Cargo.toml + the constants
pnpm docs:links
pnpm versions:check
pnpm citations
pnpm pins                      # golden byte pins
```

The published CLIs also install via `cargo install spatiotemporal-tiles`. Only
run commands you can verify from the manifests — do not invent scripts.

## How agents get help

- **Claude Code:** install the **`poopdeck:poopdeck-ai` plugin** from this repo root — it
  wires an **MCP server** (`@poopdeck.gl/mcp`, live dataset discovery / analysis /
  `@deck.gl/json` map composition / gated build+validate) _and_ a set of **Agent
  Skills** that route between the CLIs and MCP tools. Start with the
  `poopdeck-overview` skill (the router). See `poopdeck:poopdeck-ai/README.md`.
- **Other harnesses (Cursor / Codex / Gemini CLI):** the Skills are authored to
  the [agentskills.io](https://agentskills.io) open standard (portable frontmatter)
  and load in any skill-aware harness. The MCP server also exposes
  `stt://docs/*` **resources** plus `search_docs` / `get_doc` tools.
- **Published docs for retrieval:** <https://poopdeck.gl/llms.txt> (index) and
  <https://poopdeck.gl/llms-full.txt> (full corpus). The in-repo `llms.txt`
  mirrors the index.
- This `AGENTS.md` is the cross-harness entry point; it complements the plugin
  rather than replacing it.

## Working conventions

- Prefer the routing table over guessing; open the referenced doc before writing
  code against an API.
- Respect the ground rules above — they are load-bearing invariants, not style
  preferences.
- **`poopdeck:` prefixes a path in the OTHER repository.** A comment citing a
  renderer file writes `poopdeck:packages/core/src/archive.ts` or
  `poopdeck:docs/roadmap/renderer-architecture.md §2.9`; a bare path would look
  local and resolve to nothing. `.github/scripts/check-roadmap-citations.mjs`
  counts the prefixed doc citations, so a cross-repo pointer stays a declared
  thing rather than a silent skip. Do not add the prefix inside a **vendored**
  file — that file is read downstream too, where the bare path is the right one.
