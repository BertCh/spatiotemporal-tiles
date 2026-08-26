# Contributing

Thanks for helping out. This file is the short version: layout, the commands
that actually exist, and the invariants you must not break. For the deeper
"to do X, look here" routing table — which CLI, which package, which doc —
read [`AGENTS.md`](AGENTS.md). It is written for AI agents but works fine for
humans, and it is the canonical map; this file deliberately does not duplicate
it.

## Layout

One cargo workspace. The TypeScript renderers that CONSUME archives live in a
separate repository, [BertCh/poopdeck.gl][pd]; this one writes them.

```
crates/                 # cargo workspace — the 4 PUBLISHED crates
  stt-core/             #   archive + Arrow tile format library
  stt-build/            #   GeoParquet / PostGIS / DuckDB → packed .stt
  stt-optimize/         #   input analysis + archive inspect/doctor/diff
  spatiotemporal-tiles/ #   umbrella crate: re-exports the libs + ships the CLIs
    src/bin/            #     stt-build, stt-optimize, stt-validate,
                        #     stt-bundle, stt-serve
packages/               # pnpm workspace — 7 published packages + Cesium preview
  core/                 #   reader, decoder pool, cache, render kernel
  layers/               #   deck.gl backend (primary)
  three/ maplibre/       #   published alternate renderer backends
  cesium/                #   experimental workspace-only backend (private)
  playback/ react/      #   clock + governor + React UI
  mcp/                  #   published MCP server (`stt-mcp`)
poopdeck:examples/showcase/      # React Router demo site (dozens of real datasets)
tools/                  # bench, perf, render-test harnesses
  stt-generate/         #   reference-dataset generators — its OWN cargo
                        #   workspace, unpublished, off the root MSRV
docs/                   # spec, API reference, guides, architecture
poopdeck:poopdeck-ai/            # Claude Code plugin (MCP server + Agent Skills)
```

`stt-generate` is deliberately outside the root workspace (`cargo test
--manifest-path tools/stt-generate/Cargo.toml` to run it); the five shipped CLIs
are bins inside `spatiotemporal-tiles` and are feature-gated (a bare
`cargo install spatiotemporal-tiles` gets all five).

## Setup

- **Rust 1.88+** — the MSRV in `[workspace.package]`, enforced by a CI job that
  checks the four published crates on exactly that toolchain.
- **Node 24+** and **pnpm**, for the repository gates only — nothing is built or
  published from `package.json`. The Node major is pinned by `.node-version` and
  pnpm by `packageManager`, so `corepack enable` is enough.
- **Python 3.12** if you touch `scripts/data-generation/`; each extractor keeps
  its own venv (`venv-*`, never committed).

```bash
cargo build --release
pnpm install     # gates only
```

## Repository gates

```bash
pnpm project:check   # project-status.json vs Cargo.toml and the version constants
pnpm docs:links      # every relative Markdown link resolves
pnpm versions:check  # the workspace version and every internal path-dep agree
pnpm citations       # every cited roadmap doc and §section exists
pnpm pins            # golden byte pins moved only inside a declared window
pnpm lint && pnpm format:check
node --test .github/scripts/check-golden-pins.test.mjs
node scripts/gen-generate-datasets.mjs --check
```

They are pure Node — no build, no packages, seconds to run. `pnpm pins` needs
full history (it diffs against a merge-base) and fails closed rather than
checking nothing.

## Rust

```bash
cargo test --workspace --locked                  # default features
cargo test --workspace --all-features --locked   # incl. duckdb, projection
cargo build --release                            # binaries → target/release/

# The generator is its own workspace — `--workspace` above does NOT cover it.
cargo test --manifest-path tools/stt-generate/Cargo.toml
```

CI additionally compiles each published feature combination on its own
(workspace feature unification would otherwise hide missing `cfg` gates), runs
`cargo package --workspace --exclude stt-generate --locked` as a standing
publishability check, and runs the `#[ignore]`d DuckDB spatial test (the only
one that exercises the SQL `stt-serve` emits). If you add a feature, add its
lane to `.github/workflows/ci.yml`.

## Lint & format

Rust is `cargo fmt` + the curated clippy deny set (see the `rust-lint` CI job).
For the `.mjs` gates and the JSON artifacts it is **oxlint + oxfmt; Prettier was
removed — do not add it back.** Config lives in `.oxlintrc.json` /
`.oxfmtrc.json`; the house style is single quotes, 2-space indent, 80-column
print width. The generated JSON artifacts are in `ignorePatterns`: reformatting
one would break the byte comparison downstream.

```bash
cargo fmt --all
pnpm lint          # oxlint
pnpm lint:fix
pnpm format        # oxfmt (writes)
pnpm format:check
```

Markdown in `docs/` and at the repo root is prose-wrapped at ~80 columns — match
the file you are editing.

## Releasing

npm and crates.io versions move in **lockstep** — one number, both registries.
There is one release system: **changesets drives the version, cargo-dist builds
the binaries.** (There used to be three; two of them had never produced a
release. release-plz was deleted rather than left as config nobody runs.)

Add `pnpm changeset` to any PR that changes a published package. The
`@poopdeck.gl/*` packages are a `fixed` group, so they all bump together.

To release:

1. **npm.** `release-npm.yml` opens a Version Packages PR from the accumulated
   changesets; merging it runs `pnpm version-packages` → `pnpm release`
   (build + `smoke-pack` + publish). That is what sets the canonical number.
2. **Everything else follows the canonical number.** Run
   `node scripts/sync-versions.mjs` — it rewrites `[workspace.package] version`
   in `Cargo.toml`, the Claude Code plugin manifest, the marketplace entry, and
   each skill's frontmatter `metadata.version` to match
   `poopdeck:packages/core/package.json`. `--check` reports drift and exits non-zero;
   CI runs it on every PR. **This is the gate that keeps crates.io and npm from
   diverging** — they did exactly that (0.4.0 vs 0.5.0) back when Cargo.toml
   was a hand edit no check covered.
3. **crates.io.** Publish by hand, in dependency order:
   `stt-core` → `stt-optimize` → `stt-build` → `spatiotemporal-tiles`.
   `stt-generate` is `publish = false`. Note the standing constraint in
   `docs/roadmap/shipping.md`: `cargo publish` stalls on upload from the
   author's network (h2 `PUT`-with-body hangs) — publish from elsewhere.
4. **Binaries.** Tag `v{version}` and push it, then run the **Release**
   workflow (Actions → Release → Run workflow) with that tag. cargo-dist is
   configured with `dispatch-releases = true`, so pushing the tag alone does
   **not** build anything.

Changelogs: `packages/*/CHANGELOG.md` are written by changesets — do not
hand-edit them. The crates have **no** changelog files, deliberately; the
GitHub Release cargo-dist creates in step 4 is the crate-side record.

## Invariants — do not break these

These are load-bearing design decisions, not style preferences. A change that
violates one will be sent back regardless of how well it is implemented.

- **No thinning.** Never thin, sample, or aggregate features to hit a byte
  budget — comprehensive data is the whole point of the format. Shrink by
  clamping the zoom range and using temporal bucketing. The H3/Quadbin
  **summary** tier is an opt-in coarse-zoom aid, never a replacement for the
  base tier.
- **The manifest/archive is the contract**, and since the split it is the
  contract _between two repositories_. `manifest.json` carries capabilities, the
  temporal block, the pack table, and optional per-property style hints. Readers,
  CLIs, and renderers all negotiate through it — read it before guessing, keep
  `docs/spec/manifest.schema.json` in step with any change, and remember that a
  reader you cannot see is bound by it too.
- **Packs and the directory are immutable and content-addressed** (the hash is
  the filename). Only the small `manifest.json` is mutable. Never rewrite a pack
  in place — write a new one and update the manifest.
- **Golden bytes move once, reviewed.** Both fixture trees
  (`crates/stt-core/tests/fixtures/v2-golden/`, `conformance/vectors/`) are the
  encoder's determinism oracle; a change needs a `Rebuild-Window: R1` commit
  trailer, and a byte-identical relocation needs `Pin-Relocation: old -> new`,
  which the gate verifies against the object database rather than believing.

## Pull requests

Keep the diff scoped, run the relevant suite above, and describe _why_ in the PR
body. New behavior in a CLI belongs in `docs/api/cli-reference.md`; new wire
behavior belongs in `docs/spec/`. If the change alters what a reader must
accept, say so — a downstream reader has to follow, and the conformance vectors
are how it finds out.
