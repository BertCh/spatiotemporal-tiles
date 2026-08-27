# AGENTS.md

Orientation for any AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, …)
dropped into this checkout. `AGENTS.md` is the cross-harness convention that
Cursor, Codex, and Gemini CLI read automatically; Claude Code users additionally
get the `poopdeck-ai` plugin (see [How agents get help](#how-agents-get-help)).
Read this first, then follow the routing table to the canonical docs.

## What this repo is

**SpatioTemporal Tiles (STT)** — the open **format** and the Rust **toolchain**
that writes it. A dataset is a tiny `manifest.json` plus many immutable,
content-addressed **pack** objects — a directory, not a single file — that
combine a spatial tile pyramid with a temporal axis — each tile is addressed by `(zoom, x, y,
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
> **What this repo owes downstream, it owes as artifacts.** The pages and
> generated contracts listed in `docs/.corpus.json` (`vendoredDownstream`),
> `conformance/vectors/`, and `project-status.json` are vendored into
> poopdeck.gl and byte-gated there. Change them here; the downstream copy
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
| Turn **your own** GeoParquet / PostGIS / DuckDB into an archive           | `stt-build` (`--auto` to infer knobs)                            | `docs/guides/csv-quickstart.md`, `docs/api/cli-reference.md`   |
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
  stt-build/            # GeoParquet / PostGIS / DuckDB → packed archive (library)
  stt-optimize/         # input analysis + archive inspect/doctor/diff (powers --auto)
  stt-wasm/             # stt-core → WebAssembly decoder; publish = false (docs/guides/wasm.md)
  spatiotemporal-tiles/ # umbrella crate: re-exports the libs + ships the published CLIs
    src/bin/            #   stt-build, stt-optimize, stt-validate, stt-bundle, stt-serve
tools/stt-generate/     # bundled showcase-dataset generators. NOT a root-workspace
                        #   member: its own [workspace] + lockfile keep its dep tree
                        #   (osmpbf, nexrad-*, tokio, reqwest) and whatever MSRV that
                        #   tree demands off the four published crates, so
                        #   `-p stt-generate` from the root does not resolve —
                        #   `cargo install --path` it
conformance/            # portable reader vectors + make-vectors.sh
docs/                   # spec, API reference, guides, architecture
data-fleet/             # built archives, untracked
poopdeck:packages/, poopdeck:examples/showcase/, poopdeck:poopdeck-ai/  # the OTHER repository
```

## Where the docs live

`docs/README.md` indexes every page in this repository. The routing table above
already names the doc for each task; these are the ones it does not, and the
`docs/spec/` pages are **normative** — they win over any roadmap record:

- `docs/spec/stt-packed-format.md` — the container: manifest + packs +
  directory v6 (machine-checkable schema: `docs/spec/manifest.schema.json`).
- `docs/spec/time-model.md` — buckets, the temporal LOD pyramid, read-time
  pruning.
- `docs/architecture/data-format.md` — the per-tile Arrow layer-frame encoding.
- `docs/spec/sidecar-assets.md` — the scene-bundle profile (AV cockpit, and any
  multi-stream or `anchored-local` dataset).
- `docs/architecture/{system-overview,archive-format-performance}.md` — how the
  pieces fit end to end, and the no-thinning generation policy.
- `docs/intro/{concepts,choosing,glossary}.md` — the tile model, the
  static-vs-served choice, and the canonical spelling of every name.
- `docs/roadmap/README.md` — the decision-record index **and the only backlog**;
  open work is not tracked anywhere else in this repository.
- <https://poopdeck.gl/docs/intro/status-and-support> — maturity tiers,
  compatibility window, support expectations; `project-status.json` is this
  repository's machine-readable half.

## Build / test basics

One **cargo** workspace, plus a handful of Node gates. Infer exact flags from
`Cargo.toml` / `package.json`; the verified commands are:

```bash
cargo build --release          # CLI binaries → target/release/{stt-build,stt-optimize,...}
cargo test --workspace
cargo test --workspace --all-features       # incl. duckdb, postgres, both serve backends
cargo test --manifest-path tools/stt-generate/Cargo.toml   # its own workspace

pnpm install                   # only for the gates below; nothing is built
pnpm project:check             # project-status.json vs Cargo.toml + the constants
pnpm docs:links
pnpm versions:check
pnpm citations
pnpm pins                      # golden byte pins
pnpm lint                      # oxlint
pnpm format:check              # oxfmt
```

The published CLIs also install via `cargo install spatiotemporal-tiles`. Only
run commands you can verify from the manifests — do not invent scripts.

## How agents get help

- **Claude Code:** install the **`poopdeck-ai` plugin**. It lives in the OTHER
  repository — that checkout's root is the marketplace, not this one
  (`/plugin marketplace add /path/to/poopdeck.gl`, then
  `/plugin install poopdeck-ai`). It wires an **MCP server**
  (`@poopdeck.gl/mcp`, live dataset discovery / analysis / `@deck.gl/json` map
  composition / gated build+validate) _and_ ten **Agent Skills** that route
  between the CLIs and MCP tools. Start with the `poopdeck-overview` skill (the
  router). See `poopdeck:poopdeck-ai/README.md`.
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
