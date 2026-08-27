# Shipping & distribution — decision record

_How this project is published, why the scheme looks the way it does, and what
was counted out. Behavior of the CLIs themselves lives in
[../api/cli-reference.md](../api/cli-reference.md)._

> **Status note (2026-08-26).** This is a decision record, so dated registry
> observations below are retained as history. For the current release procedure,
> use [CONTRIBUTING.md](../../CONTRIBUTING.md#releasing). The checked-in manifests
> currently define Rust **0.8.0** with MSRV **1.88**. Rust crates are published
> manually in dependency order; cargo-dist binaries require a `v{version}` tag
> followed by a manual **Release** workflow dispatch. The deleted release-plz
> workflow is not part of the current release path.

**Historical registry snapshot (verified 2026-07-26).** crates.io
`spatiotemporal-tiles` max_version was **0.4.0**, published 2026-07-06, three
minor versions behind what the same tree had already shipped to npm. See
"Historical release defect" below.

## Distribution pathways (decided)

| channel               | artifact                                                                                                                             | mechanism                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| crates.io             | **one user-facing name**: `spatiotemporal-tiles` (facade lib + all 5 CLI bins, feature-gated) over 3 published implementation crates | manual publish in dependency order; `sync-versions.mjs --check` gates the internal path-deps |
| GitHub Releases       | prebuilt `stt-*` binaries, 5 targets + shell/powershell installers                                                                   | cargo-dist 0.32.0 via manual `release.yml` dispatch for an existing tag                      |
| Cloudflare (existing) | R2 tile data (the showcase site deploys from poopdeck.gl)                                                                            | `scripts/r2-sync.sh`                                                                         |

## Naming rationale

- crates.io has a flat namespace; the bare `stt` name is squatted by an
  abandoned whisper lib, and "stt" reads as speech-to-text anyway. RFC 3243
  (`spatiotemporal-tiles::core`-style namespaces) is a 2026 Rust project goal
  but **not yet publishable** — owning the `spatiotemporal-tiles` base name
  reserves that namespace for when it lands.
- The facade is the bevy/gix model: users type ONE name
  (`cargo add spatiotemporal-tiles`, `cargo install spatiotemporal-tiles`);
  `stt-core/build/optimize` publish only because cargo requires published deps,
  and carry "internal implementation crate" banners.
- `crates/stt-validate` and `crates/stt-serve` were dissolved into the facade
  as feature-gated bin targets (binary names unchanged: `stt-build`,
  `stt-optimize`, `stt-validate`, `stt-serve`; `stt-bundle` joined later).
  `stt-generate` stays `publish = false` (RC nexrad deps; internal demo
  tooling).

## Feature/install matrix (facade)

- **Defaults ship the CLIs, not a slim lib.** `default = ["build-cli",
"optimize-cli", "validate-cli", "bundle-cli", "serve-postgres"]` — a bare
  `cargo install spatiotemporal-tiles` gets all five binaries with the light
  pure-Rust dep set (PostGIS serve backend only; bundled DuckDB is a heavy C++
  build). Library consumers are told to depend with `default-features = false`;
  docs.rs is configured that way too (`no-default-features`, features
  `build, optimize, postgres, duckdb`).
- `build` / `optimize` → library re-exports; `postgres` / `duckdb` forward to
  stt-build's input sources.
- `cli` → all five bins, both stt-serve backends; it is what
  `[package.metadata.dist]` builds. Lighter serve:
  `--features build-cli,optimize-cli,validate-cli,serve-postgres` skips the
  ~10-min bundled-DuckDB compile. `serve-core` alone is a deliberate
  `compile_error!` (needs ≥1 backend). `serve-postgres-tls` adds
  `sslmode=require` and is never a default — it would pull native-tls into the
  light dep set and disturb the local-dev NoTls path.
- stt-build's `geoparquet` feature is a **no-op back-compat alias** — the
  encode path needs arrow/parquet unconditionally (the optional-feature split
  never compiled and was untested before this work).

## Version/tag scheme

- **The two registries are not lockstep.** crates.io and npm last agreed at
  0.7.0 by history, not by promise; from here each moves on its own cadence. What
  relates the two stacks is the archive's `formatVersion`, declared in
  `project-status.json` on each side — read that, not the version string. The
  four published crates share one `[workspace.package] version`, currently
  **0.8.0**, and `sync-versions.mjs --check` gates the internal path-dep pins
  against it.
- Rust tag: `v{version}` (created manually; cargo-dist consumes it when the
  Release workflow is dispatched). npm's tags live downstream and cannot collide
  with it.
- MSRV: `rust-version = 1.88`, enforced by the CI `rust-msrv` job. It briefly
  read 1.87 (2026-08-24 to 2026-08-26, after `stt-generate` took its
  home→osmpbf floor into its own workspace) and that was wrong: `geo` 0.33 had
  already raised its own floor, so the locked tree needed 1.88 and
  `cargo install` on 1.87 failed a resolver check the manifest said would pass.
  Bisect an MSRV against the **lockfile**, not just against our own source.

## Historical release defect — the lockstep was broken

This was **B3** in the [roadmap README](./README.md), discharged by the 0.6.0
release on 2026-08-13. What belongs here is the two _structural_ reasons the
lockstep broke, both of which outlive any one release:

- **An unbuilt tag is a dead install path.** cargo-dist now uses
  `dispatch-releases = true`: create the `v{version}` tag, then manually dispatch
  `.github/workflows/release.yml` with that tag. A version that skips either step
  has no prebuilt binaries.
- **Every publish so far has been manual.** All commits in the repo are the
  author's; not one bot commit, not one `chore: release` PR merge. The lockstep
  rule ("crate and npm versions stay in step") has therefore always been
  aspirational rather than enforced — crates.io never even got a 0.2.0 or 0.5.0.
  That history is why the current procedure gates versions with
  `sync-versions.mjs --check` and requires an explicit Release workflow dispatch.

## Release systems: three became two (the deletion happened)

The repo used to carry **three** independent release mechanisms and used none of
them end-to-end. `release-plz.toml` and `.github/workflows/release-plz.yml` are
now **gone** (verified 2026-07-26): changesets owns npm from the renderer
repository, while cargo-dist builds Rust binaries from a manually dispatched
workflow for an existing `v{version}` tag. The evidence that forced the deletion
is kept because it is the argument against ever adding a third:

- `crates/*/CHANGELOG.md` **do not exist**, despite `release-plz.toml` setting
  `changelog_update = true`.
- 0.5.0 landed as a hand-edited bump commit (`8bdc01d`, "bump workspace and npm
  packages to 0.5.0"), not as output of any of the three.

**The reasoning that settled it, kept as the standing rule.** release-plz's value
is the release-PR/changelog workflow, and that workflow produced zero PRs and zero
changelogs here — a third system whose only effect was to make the crate
changelogs look _missing_ rather than absent-by-choice. cargo-dist keeps Rust
binary distribution. **The release ritual must end with a `v{version}` tag and a
successful manual Release workflow dispatch, or the binaries silently stop
existing** — that is the failure mode the deletion did not remove.

## Operational constraint: publishing crates from this network

`cargo publish` **stalls on upload from the author's network** — the h2
`PUT`-with-body hangs; plain HTTP/1.1 to crates.io is fine, and setting
`multiplexing = false` does **not** help. It is not cargo, not the sandbox, and
not crates.io being down. **Publish the crate from a different network** (or
from a CI runner). Budget for this when planning a release; it has eaten a
release window before.

## Publishing the tile fleet to R2 — the ordering is the whole procedure

The crates half above ships code; this half ships the ~65 GiB of archives the
code reads. It has its own release ritual, learned from the 2026-07-31 republish
of the whole fleet (29.3 GiB, 1,324 objects, 68/68 manifests flipped).

**The two halves are versioned against each other, so both naive orders break the
live site:**

- Push the frontend first and the new reader meets the old manifests →
  `unsupported formatVersion 1`.
- Flip the manifests first and the DEPLOYED older reader meets manifests declaring
  a capability it lacks → `requires capabilities this reader does not implement`.
  Measured at the break: **424 of 474** local manifests declared a post-break
  capability, so this order is far the worse of the two.

**The resolution is that packs are content-addressed.** The immutable pass writes
them under names nothing references yet and is invisible to the live site; only
`manifest.json` is the switch. So: **upload packs → push the frontend → flip the
manifests the moment the Pages deploy goes live.** That puts the entire exposure
inside the manifest pass — measured at **15 seconds**, affecting only the archives
whose format actually changed.

Three standing rules that fall out of it:

- **`--no-prune` on any republish that shares nothing with the previous
  manifest.** Prune grace is computed from the previously-deployed refs; a full
  re-address shares none of them, so the grace rule does not cover it. Let the
  retention window pass, then let a later default sync GC. Rollback until then =
  re-upload the previous manifests **and** pin the previous reader.
- ⚠️ **Never probe a URL before uploading it.** Probing not-yet-uploaded sidecars
  cached a negative response at the edge under `cache-control: max-age=14400` — a
  **4-hour 404 on an object that is present and correct in the bucket**. `HEAD`
  bypasses the cache and returns 200, which makes the symptom look inconsistent;
  `cf-cache-status: HIT` on the GET is the tell. Verify with a plain **GET**.
  Neither Cloudflare token in `.env` carries the Cache Purge permission.
- **A pass that matches nothing fails silently.** `r2-sync.sh` is a list of
  filtered `rclone copy` passes, each ending in `- **`; a file that matches no
  pass is simply never uploaded and nothing reports it. This has now bitten twice
  on the same root-level-sidecar shape (L1 in the [roadmap README](./README.md)).
  After adding any new non-packed artefact, probe its URL — after the upload, per
  the rule above.

## Auth lifecycle (token → OIDC)

The bootstrap token lives in the gitignored root `.env` (`CRATES_TOKEN`). Rust
publishing remains the manual dependency-order procedure in `CONTRIBUTING.md`;
there is no release-plz workflow to name as a trusted publisher, so there is
nothing here to migrate to OIDC until one exists.

## CI gates that keep publishability true

> ⚠️ **The gates have always run; nobody was reading them.** This section spent a
> month asserting that GitHub Actions had never run here, inferred from zero bot
> commits. It was wrong — 184 runs, with a standing red on `CI`. "Nobody reads the
> runs" is the worse failure of the two, because the gates were reporting real
> defects into an empty room for a month: `rust-msrv` had been red since `geo`
> 0.33 raised the locked tree's floor above the `rust-version` this file
> promised, and the default-features lane was dying in the linker. Both are
> fixed; `CI` and `Release` are green at `35cc8fe`. The 2026-07-31 by-hand run
> keeps its place in the record, because it found four red and is how the feature
> lanes below earned their keep — see
> [db-input-adaptors.md §5](./db-input-adaptors.md).

Ten jobs in `.github/workflows/ci.yml`:

- `rust` / `rust-all-features`: `cargo test --workspace --locked` at default
  features (what `cargo install` users get) and at `--all-features` — tests, not
  just builds, so the feature-gated tests actually run. Both set
  `CARGO_PROFILE_*_DEBUG: '0'`; full debuginfo OOMs the 7 GB runner's linker.
- `rust-lint`: `cargo fmt --all -- --check` plus a curated correctness clippy
  set the workspace satisfies in full, so every hit is a fresh regression.
- `rust-feature-lanes`: six solo compiles — facade lib with no features, facade
  bare `serve` (both backends), `serve-postgres`, `serve-duckdb`, full `cli`, and
  stt-build `--no-default-features`. Workspace feature unification would
  otherwise hide a missing cfg gate until a user hit it.
- `rust-package`: `cargo package --workspace --exclude stt-generate --locked` on
  every PR, so package excludes and versioned path-deps fail here rather than at
  release time.
- `rust-msrv`: `cargo check --locked` on 1.88 over the four **published** crates
  by name — not `--workspace`, which dragged the unpublished `stt-generate`'s
  osmpbf/home/delaunator floor into the number users see.
- `rust-duckdb`: `cargo test -p stt-build --features duckdb`, plus the `#[ignore]`d
  `spatial_roundtrip_smoke` — the only test that runs the SQL the server emits.
- `rust-postgres-parity`: the ignored PostGIS source-parity tests against a
  `postgis/postgis:16-3.4` service container.
- `python`: the data-generation tests, and `emit_av_palettes.py --check` — the
  seam gate that keeps `docs/spec/av-palettes.json` current, since the renderer
  now asserts against that artifact instead of reading `av_common.py`
  ([repo-split-2026-08.md §4.3](./repo-split-2026-08.md)).
- `gates` (Node): `sync-versions.mjs --check`, `check-project-status.mjs`,
  `check-doc-links.mjs`, `check-roadmap-citations.mjs`, `check-golden-pins.mjs`
  and its own self-test, `gen-generate-datasets.mjs --check` (the second seam
  gate — the generator subcommand inventory the MCP server binds its enum to,
  [repo-split-2026-08.md §8.2](./repo-split-2026-08.md)), then `pnpm lint` and
  `pnpm format:check`.

## Explicit non-goals (counted out, with revival triggers)

- **DB extensions** (pgrx Postgres extension, DuckDB community extension):
  the DB story is input adaptors + the `stt-serve` binary (the `ST_AsMVT`
  analog, per `db-input-adaptors.md`). Nothing runs _inside_ a database.
- **Python packaging**: `scripts/data-generation/*` stay internal scripts
  (dataset-specific licenses, per-dataset venvs).
- **Docker image / Homebrew tap**: cargo-dist's installers + `cargo install`
  cover it. Revive with a ghcr.io image for `stt-serve` if someone actually
  asks to run the server as a service.
- Node 20 and earlier (EOL; `.node-version` and the `gates` job require Node
  24+).

## Known risks / fallbacks

- cargo-dist is community-maintained post-axo: version pinned (0.32.0),
  generated `release.yml` treated as vendored; fallback is
  `taiki-e/upload-rust-binary-action`.
- docs.rs timeout on the bundled-duckdb doc build: drop `duckdb` from the
  affected `[package.metadata.docs.rs]` features in a patch release.
- Windows dist job (fat archives + DuckDB) is the wall-clock long pole;
  `[profile.dist]` already uses thin-LTO as the relief valve.
