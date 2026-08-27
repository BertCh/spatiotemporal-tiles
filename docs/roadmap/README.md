# Roadmap — decision records

A decision record here holds **rationale, measured baselines, negative results,
and counted-out items with revival triggers**. It is deliberately _not_ a
description of current behavior — the spec (`docs/spec/`) owns that, and a
record that restates it goes stale the moment the code moves. Nor is it a
campaign diary: how the work was sequenced is in git history, not here.

> **Two registers since 2026-08-26.** The repository split
> ([repo-split-2026-08.md](./repo-split-2026-08.md)) gave the renderer its own
> home. This register is complete **for the format, the tiler, the optimizer and
> the published data fleet**; open work on the TypeScript renderers, playback,
> the MCP server and the showcase lives in the poopdeck.gl repository's
> [own register](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/README.md).
> Records that moved are linked below by URL rather than repeated, and source
> comments that cite them carry an explicit `poopdeck:` prefix so the citation
> gate can tell a declared cross-repo pointer from drift.

> **Current-state rule (2026-08-24).** This directory preserves dated evidence
> and decision history; it is not the source of truth for versions, supported
> features, or release commands. Use `Cargo.toml` for versions and toolchain
> floors, `CONTRIBUTING.md` for the release procedure, and `docs/spec/` for
> shipped behavior. An older claim below remains historical unless its item has
> an explicit later status line.

Three house rules:

- **Every measurement keeps its units and its source.** If a number cannot be
  traced to the run that produced it, the claim is dropped rather than restated.
- **Open work lives in exactly one place — the backlog below.** Unbuilt or
  declined work is _not_ listed there; it lives as a counted-out bullet with a
  revival trigger inside the record that owns it. A record that is not indexed
  below is not findable, so a new record earns its index line in the same pass
  that creates it.
- **A closed item leaves one line, not its history.** When a backlog item is
  done it collapses to a single line in the discharged ledger, and any durable
  lesson it produced moves into the record that owns the subject. The story of
  how it closed is in git history.

## Records

- [**repo-split-2026-08.md**](./repo-split-2026-08.md) — the **two-repository
  contract**: what stayed here and what moved to poopdeck.gl, why
  `@poopdeck.gl/core` was not split in half, the three vendored seam artifacts
  (docs, conformance vectors, AV palettes) and their drift gates, and the costs
  accepted.
- [**shipping.md**](./shipping.md) — versioning and the crates.io registry, the
  naming rationale, the feature/install matrix, publish auth, the R2
  fleet-publish ordering rule, and the explicit non-goals. The npm half moved
  downstream with the packages.
- [**stt-packed-format-decisions.md**](./stt-packed-format-decisions.md) —
  format, build and optimizer decisions: the measured baselines, the paged
  directory, the frozen wire-token invariants, the two byte-break events
  (packed-v2 and the 2026-07-26 payload break), and the negative results
  (lightweight encodings, coordinate transforms, inter-timestep delta chains).
- [**db-input-adaptors.md**](./db-input-adaptors.md) — PostGIS/DuckDB as
  `stt-build` inputs and `stt-serve` backends: the seven encoder-seam lessons,
  the ingest/serve benchmarks, and the static-vs-DB verdict.
- [**demos-and-datasets.md**](./demos-and-datasets.md) — the dataset licence
  register, the BLOCKED list, the operational time-bombs, and the per-demo build
  gotchas.
- [**av-cockpit.md**](./av-cockpit.md) — the `/drive` AV **data contract**
  (normative: the extractors cite its numbered sections), the georeferencing
  gotchas, the palette-lockstep rule — now enforced through
  [`docs/spec/av-palettes.json`](../spec/av-palettes.json) — and the measured
  LiDAR compression story.
- [**storm-4d-greenfield-2026-07.md**](./storm-4d-greenfield-2026-07.md) — the
  storm-4d **archive/field contract** (§9.1, normative: eight generators cite
  it), the no-thinning verdict, and the Py-ART-not-Rust rationale.
- [**neural-atlas-2026-07.md**](./neural-atlas-2026-07.md) — a transformer's
  latent space on a token clock: the generator contract and the geometry
  rebuild.
- [**optimization-problems-2026-08.md**](./optimization-problems-2026-08.md),
  [**-informed-design**](./optimization-informed-design-2026-08.md),
  [**-implementation-plan**](./optimization-implementation-plan-2026-08.md) and
  [**-conformance**](./optimization-conformance-2026-08.md) — the optimization
  program: the formal problem statements, the six defects and the design that
  answers them, the work items, and the conformance obligations.
- [**formal-semantics-2026-08.md**](./formal-semantics-2026-08.md) — the
  semantic layer under the format trilogy: the enforced laws and the
  adjudication register.

**Moved to the poopdeck.gl repository** (linked, not repeated): the renderer
architecture record, playback-and-loading, both tile-loading records and the
loader audit, both measurement records, the AI-suite record, launch readiness,
and the OpenUSD evaluation. Source comments here cite them as
`poopdeck:docs/roadmap/<file>.md §N`.

---

## The backlog

The single source of open work **for this repository** at the time of each dated
update. Items carry the check that proves them and the condition that closes
them. Ordered by what blocks what, not by size.

**Where this actually stands.** The register that ran from 2026-07-26 to
2026-08-26 covered both stacks; on 2026-08-26 it was split along with the
repositories. What stayed here is the format's, the tiler's and the data
fleet's queue. The B1 → B2 → B3 chain that dominated the last four registers is
discharged, and B4 — the proposed fleet rebuild at `formatVersion: 3` — was
discharged on 2026-08-14 by a container-only v2→v3 migration. Its residue is
the B section below; the rebuild investigation behind it moved to the records
that own each half.

Items about the renderers, playback, the MCP server and the showcase moved
downstream — including L2's browser-verify queue, DX1/DX4, K3–K6, K8, K10 and
the whole TL tile-loading section.

The last whole-repo green baseline recorded here (2026-07-31) covered both
stacks: **45 Rust test targets at `--all-features` (1,264 tests), the six
feature lanes, the curated clippy set, `cargo fmt --check`, the MSRV check, 35
Python tests, oxlint, `oxfmt --check`, the version-sync gate, the
roadmap-citation gate, the golden-pin gate and its own 41 tests, `smoke-pack`,
and 6,240 package + showcase tests.** Everything up to and including the Python
tests is this repository's half; the rest is downstream's. The 2026-07-31 lesson
behind that phrasing — that `cargo test --workspace` alone was hiding four red
jobs — is recorded in [db-input-adaptors.md §5](./db-input-adaptors.md) and in
[shipping.md](./shipping.md).

### S — The split's own tail

**S3. Half the rehomed fleet's tooling is still unrun.** The 76 GB local fleet
lived at `poopdeck:examples/showcase/public/data`, a path in a repository that no
longer exists here; it is now `data-fleet/`, and `r2-sync.sh`,
`rebuild-fleet-v3.sh`, `patch-manifest-metadata.mjs`,
`tools/fleet-order-audit.sh` and every data-generation script were repointed by
rewrite, not by running them. `r2-sync.sh` has since been run for real against
the new root (DX2, discharged below); the read side has not. **Accept:** one
`stt-optimize inspect --sample 0` over a stem under the new root, from a clean
checkout.

### B — The v3 residue

B4's fleet-rebuild framing was discharged on 2026-08-14 by not rebuilding (one
line in the discharged ledger below). What the migration could not reach is
this section, and it is small.

**B5. Five summary-tier archives are still `formatVersion: 2`.**
`earthquakes-summary`, `goes-glm-lightning`, `nyc-od-quadbin`,
`nyc-taxi-od-summary` and `osm-nyc-changesets`. The migration refuses them on
purpose: a v2 directory has no column saying which entries are aggregates, so
the raw/summary split v3 needs cannot be recovered from one. They serve
correctly through the reader's v2 read window, so this is not an outage — but
that window is a property of the reference readers, not of the format
([packed spec §9.1](../spec/stt-packed-format.md)), and it is the only thing
keeping them alive. **Accept:** each is rebuilt at v3, or this register records
the decision to leave them on v2 permanently and the reader's obligation is
restated as deliberate rather than transitional.

**B6. Two staged rebuilds await a density decision, not a technical one.**
`ais-all-us` and `flights` are built, validated and **not** published, at 6.45×
and 5.93× the features of what ships (2.2 GB vs 0.51 GB; 4.2 GB vs 0.81 GB).
Neither is a defect: today's defaults preserve every usable row at every zoom —
the no-thinning ground rule — while the shipped archives were built thinned, so
the rebuilds are the more honest artifacts and also ~5× the bytes against a
fleet that is ~18 GB today. The per-dataset analysis and the feature-count gate
that catches this class live in
[demos-and-datasets.md §1.5](./demos-and-datasets.md); `wildfires` and
`satellites` are refused/parked there for source-side reasons. **Accept:** the
density call is made per dataset and written down, then each is published or
deleted — not left staged.

### L — Live defects on the published fleet

**L1. One atlas sidecar is missing from R2, and `r2-sync.sh` structurally cannot
upload it.** The Neural-State Atlas is still gated
(`ATLAS_ARCHIVES_SYNCED = false`,
`poopdeck:examples/showcase/src/datasets.ts:5499`) and the reason is now
precise. Probing
2026-08-03: all three `neural-atlas-*` archives, all three `<stem>.meta.json`
files and both `neural-atlas-node-{index,series}.bin` blobs return **200** — the
2026-07-31 negative-cache window has expired and the `[sidecar]` pass works. But
**`/data/neural-atlas.json` returns 404**, and that is the file
`poopdeck:examples/showcase/src/pages/NeuralAtlasImpl.tsx:80` fetches _first_
(`SIDECAR_URL` — the generator sidecar
carrying the pin, the framing contract and the two `.bin` URLs). The `[sidecar]`
pass filters on `+ *.meta.json` and `+ *.bin`; a bare root-level `.json` that is
not `manifest.json` matches no pass in the script and is rejected by every
trailing `- **`. This is the same hole that swallowed the `.bin` files, one
extension wider. **Accept:** the `[sidecar]` filter covers the generator sidecar,
`/data/neural-atlas.json` returns 200 on a plain **GET** (not `HEAD` — it bypasses
the edge cache and will lie to you), and `ATLAS_ARCHIVES_SYNCED` flips to `true`.

> **L2 moved downstream.** The browser-verify queue is a renderer gate; it is
> now L2 in the
> [poopdeck.gl register](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/README.md).

### DX — Onboarding review (2026-08-26)

A walk of the poopdeck.gl onboarding path exactly as a newcomer takes it —
`npm install` from the public registry, the quickstart copied verbatim, rendered
in headless Chromium against `tiles.poopdeck.gl`. Thirteen findings. **Nine are
fixed in the tree**; of the four the fix could not reach, two were this
repository's manifest-and-fleet defects — DX2 is discharged below, leaving DX3.
DX1 (a Vite 8 bundler bug) and DX4 (an intermittent deck.gl assertion) moved
downstream with the renderer.

Fixed and not repeated here: the unclickable play button and the light-only
transport bar in the quickstart's React sample (F1, F6 — the bar now ships a
dark token set and a `data-stt-theme` pin); the two TypeScript errors in
copy-pasteable samples plus a CI gate that typechecks the doc snippets so they
cannot drift again (F3, `scripts/check-doc-snippets.mjs`); the Node-24 engine
floor on six browser packages (F4); the Float32 precision warning that fired on
the canonical path (F5); the root README pointing at the CSV guide instead of
the quickstart (F7); the empty column inventory in `ArchiveMetadata`, now
derived from the manifest's own schema templates and exposed through a base
`onMetadataLoad` (F8); the production console warning on the live showcase
(F10); and the missing basemap and bundle-size notes in the quickstart (F11,
F12).

**DX3. The metadata gap is fleet-wide, not one dataset.** `--scan` over the 70
local packed datasets: **25 carry a build-scratch name** (`ais-all-us-new`,
`hurricanes.new`, `lines v2`, …), **65 have no description**, and **66 have no
attribution** — for datasets whose licences are the reason several of them can be
published at all. The curated copy already exists in
`poopdeck:examples/showcase/src/content/demoMeta.ts` (`tagline` + `dataSources`
with name, url and licence per demo), so this is a mapping job plus a
manifest-only republish, not a rebuild. **Accept:** `patch-manifest-metadata.mjs --scan` shows
no scratch names and no empty attribution across the shipped stems, and the
manifest pass is synced.

### K — Known defects with a named fix

Each is small, real, and has its analysis written down where it belongs. None
blocked the completed 0.6.0 release. K3–K6, K8 and K10 are renderer defects and
moved downstream; the numbering is left alone so existing references keep
resolving.

**K2. `stt-validate` reports structural drift on correct archives.** The per-tile
exact-integer quantizer refuses a column on outlier-inflated inputs, which
changes the column _set_ tile to tile; `part_offsets` adds
`<absent> vs List<UInt32>` entries for the same reason. The archive is right and
the report is a false positive. **The named fix has shipped.** The two-pass build
is the default path, and pass 1's dataset-global statistics scan pins both the
attribute range and the dictionary verdict over the whole dataset
(`crates/stt-build/src/dataset_stats.rs`; `--single-pass` is the documented
rollback). What is left is a re-measurement, not a fix: re-run `stt-validate
--sample 300` over the fleet's two-pass-rebuilt archives and either close K2 or
restate the residual false positive with fresh counts.
([format §10.3–§10.4](./stt-packed-format-decisions.md))

⚠️ **Not every drift report is this false positive.** Validating all 64 local
archives at `--sample 300` (2026-07-29) found **real** drift in six —
`wildfires` (163), `ais-all-us` (82), `animals` (66), `flights` (13),
`drifters` (3), `osm-nyc-changesets` (2) — all predating the property-kind fixes
(`2c020da`, `c13970a`), and all cleared by a rebuild with the current builder.
So: treat a drift **error** as real until a rebuild says otherwise; K2 covers the
benign `adaptive encoding width varies by tile` **warning**, not the error. Five
of the six are rebuilt and installed (old copies kept as
`.<name>.bak-drift`); `wildfires` is not — see K9.

**K9. `wildfires` cannot be regenerated: the upstream data is gone.** The shipped
archive holds ~460 fires. The NIFC service on 2026-07-29 returns 98,168 records
overall but only 297 for 2020–2023, of which **10** clear `--min-acres 1000` as
wildfires. Two real query bugs were fixed in the same pass (`FIRE_YEAR` is an
`esriFieldTypeString`, so the year range was a string comparison — the typed field
is `FIRE_YEAR_INT`; and `FEATURE_CA = 'Wildfire'` matched one of five values,
dropping `Wildfire Final Fire Perimeter` and `Wildfire for Resource Benefit`),
which took the match count 1 → 10 and confirms the filter was also wrong — but the
data is not there to recover. The archive keeps its 163-error K2 drift because
regenerating would destroy it. **Re-source before touching it.**
_(The sibling half is closed: `stt-generate` no longer silently drives whatever
`stt-build` is on `PATH`.)_

**K11. The builder is fixed; the published fleet still declares centroid bounds
and no `z_range`.** `stt-build` now takes `metadata.bounds` over every geometry
**vertex** — `BoundsMode::Vertex`, the default since R1
(`crates/stt-build/src/input.rs:959`; `--bounds-mode centroid` is the documented
rollback) — and stamps the `bounds_mode` attestation `stt-validate` check 13
reads. `z_range` landed the same way: an additive
`skip_serializing_if = "Option::is_none"` field on both `Metadata` and the
fingerprint (`crates/stt-core/src/metadata.rs:357`, `:1235`), so existing
manifests round-trip byte-identically, populated from 3D geometry or from
`--point-elevation-column`.

None of that has reached the wire. Every manifest on R2 was written pre-R1: the
declared bbox is a bbox of **centroids**, which provably under-states the extent
of any line / polygon / multi-point archive because the tiler addresses tiles by
vertex, and there is no `z_range` at all. Blast radius unchanged: the showcase
frames its opening camera from these bounds, and `stt-validate` and the MCP
`describe_dataset` both report them as the dataset's bbox — all three understate
the true extent, and no consumer can discover that a dataset is volumetric, so
altitude-aware selection stays hand-configured per demo. This was recorded as
"fold into B2 rather than schedule a republish"; B2 shipped without it.
**Accept:** every republished manifest carries
`properties.bounds_mode = "vertex"` — 2 of 64 local manifests do today — and the
pass rides an existing rebuild window rather than a dedicated one.
([tile-loading-3d-2026-07.md §6, §6b](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/tile-loading-3d-2026-07.md))

**K12. The producer-side LOD assigner covers points at exactly one zoom — not
lines, polygons, or a progressive floor.** `stt-build` reads a feature's LOD floor
from a property named by `--min-zoom-field` (`feature_min_zoom`,
`crates/stt-build/src/tiler.rs:733`) and confines the feature to a `[min_zoom,
max_zoom]` band from there. It used to compute that floor nowhere: the values were
baked upstream by the generation scripts (`scripts/data-generation/mrms_refloor.py`,
`mrms_volume.py`, `nexrad_volume.py`, the AV extractors) against a thinning grid whose
temporal bucket is a constant in a second codebase. Nothing asserted the two equal, and
when they diverged on 2026-07-28 the archive still built, still validated and still
rendered — at a median 13 % (worst 0 %) of the features visible in the displayed
bucket, with no decode-time signal at all. `crates/stt-build/src/lod_bucket.rs` (SH-4)
closed that hole by stamping the grid's bucket width into the Parquet footer and
hard-erroring on a mismatch. That is the right guard for the architecture, and its two
warn paths — footer absent, column absent — are exactly the cases it cannot check.

**DT-2 built the assigner** (`crates/stt-build/src/home_zoom.rs`, driven by
`stt-build --additive-lod [S_PX]`), with two of the three properties the named fix
asked for: voxel cells keyed on the **resolved** temporal bucket, and a total order on
candidates — `(importance desc, feature id asc)` — so assignment is deterministic under
input reordering and builds stay byte-reproducible. It routes through the existing band
mechanism rather than touching the placement authority: an assigned `home_zoom` wins
over `min_zoom_field` in `tiler.rs`. **Two gaps remain, and they are what keeps the
Python in the loop:**

- **No extent gate.** `home_zoom.rs` has no bbox or diagonal logic at all, so a line or
  polygon competes for a voxel on its anchor exactly like a point. COGP's
  `assign_levels` enters a feature as soon as its bbox diagonal is resolvable
  ([stt-packed-format-decisions.md §6](./stt-packed-format-decisions.md)); that half was
  not built.
- **One zoom, not a band.** Under `--additive-lod` a feature's ceiling equals its floor
  (`crates/stt-build/src/tiler.rs:745`) — which is what makes the partition additive,
  O(N) rather than O(|Z|·N) — and the flag refuses to run alongside `--min-zoom-field`
  (`crates/spatiotemporal-tiles/src/bin/stt-build.rs:1636`). So there is still no
  producer-side **replicated / progressive** floor, which is precisely the shape the
  generation scripts bake, and a thinned build of that shape stays reachable only by
  users who also run our Python.

**Accept:** an archive built from a floor-free input carries per-zoom feature counts
equal to one built from a baked column on the same source; the SH-4 warn paths become
the legacy path rather than the normal one; and `lod_bucket.rs` keeps policing
externally baked columns without being the only way to obtain one.

### Discharged (post-split)

Closed since the 2026-08-26 split. This section is owned by this register alone;
the downstream register keeps its own.

- **S1 — the vendor pin is exercised.** poopdeck.gl's `.stt-sync.json` pins
  `f4c4a95`, and its CI ran `sync-stt.mjs --check` green with no sibling
  checkout. All three riding artifacts are covered: the vendored doc list,
  `conformance/vectors/` and this repository's `project-status.json` block.
- **S2 — 0.8.0 was cut independently on both sides** (crates.io and npm), and
  the CHANGELOG states the end of lockstep in as many words: what relates the
  two stacks is the archive's `formatVersion`, declared in
  `project-status.json` on each side, not a shared version string.
- **T1 — both repositories are public.** The `repository`/`homepage` URLs on all
  four published crates resolve, and `sync-stt.mjs`'s pinned-tarball fetch
  resolves anonymously.
- **T2 — CI and Release are both green on GitHub's own runners** at `35cc8fe`
  here, and CI is green at `027e2f6` downstream. The two standing reds the
  corrected item named are fixed: `rust-version` is 1.88 (bisected against the
  lockfile) and the default-features lane now carries the same
  `CARGO_PROFILE_*_DEBUG: '0'` linker workaround the all-features lane had. Both
  durable lessons — bisect an MSRV against the lockfile, and that "nobody was
  reading the runs" is worse than "the gates do not run" — moved to
  [shipping.md](./shipping.md).
- **DX2 — the flagship manifest is deployed.** A plain GET of
  `/data/earthquakes-v2/manifest.json` returns a real name, a description and the
  ComCat attribution. It also discharges half of S3: `scripts/r2-sync.sh` has now
  been run for real, with R2 credentials, against the `data-fleet/` root.

### Discharged (pre-split)

One line each, so the ledger is auditable — not to imply they are still open.

> Everything below closed **before** the 2026-08-26 repository split, when one
> register covered both stacks. It is kept verbatim in both repositories rather
> than bisected: the work was done in one tree, and splitting a frozen history
> along a boundary that did not exist at the time would misdescribe it.
> Post-split discharges are recorded only in the register that owns them.

**Since the 2026-07-26 register.**

- **B3 — 0.6.0 shipped (2026-08-13).** crates.io `stt-core` / `stt-optimize` /
  `stt-build` / `spatiotemporal-tiles` and the seven published `@poopdeck.gl`
  packages all read **0.6.0**, `v0.6.0` is tagged and pushed, and the registries
  are level again (crates.io skips 0.5.0, which was cut on npm but never
  published there). The HTTP/2 publish stall [shipping.md](./shipping.md) budgets
  for did **not** recur; `CARGO_HTTP_TIMEOUT=900` and publishing in dependency
  order was enough. The cut's reserved public-API decision was resolved by
  **removal**: `emitGLSL300` and Cesium's `timeFilterAlphaGlsl` are gone,
  `ALPHA_EXPR`/`evalExpr` stay, and `render-spec.json` now declares an empty
  emitter list with a contract test pinning both names absent. Two gate holes
  surfaced in the process and were closed rather than worked around —
  `sync-versions.mjs` never covered the internal cargo path-dep `version` pins
  (cargo refuses to update a lockfile when they lag, which is how it was found),
  and `smoke-pack` left `@deck.gl/widgets` unpinned, so an upstream 9.3.10
  release could redden the publish gate with nothing in the repo changing.
  ([renderer-architecture §5.1](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/renderer-architecture.md))
- **B1 — the 2026-07-26 payload byte break landed** as `a7b57dc`, one commit as
  the accept condition required, and is **pushed**: HEAD is level with
  `origin/main`. ([format §10](./stt-packed-format-decisions.md))
- **B2 — the fleet is republished and verified.** 29.3 GiB of content-addressed
  packs and index objects (1,324) went up, then the manifests flipped; re-probing
  all 68 registered manifest URLs returns **68/68 at `formatVersion: 2`**, from
  35 v2 / 24 v1 / 9 × 404 that morning. Seventeen were additionally decoded
  end-to-end. The ordering rule this produced — packs first, then the frontend,
  then manifests, which put the whole exposure inside a **15-second** window — is
  now a standing procedure in [shipping.md](./shipping.md).
- **B4 — the fleet reached `formatVersion: 3` on 2026-08-14 without a rebuild.**
  A container-only migration (`stt_core::pack::migrate_dataset_v2_to_v3`)
  re-encoded 59 of 64 local datasets' directories under codec v6 and rewrote
  their manifests, touching **zero packs** — 76 MB of new directory objects
  against days of compute and a ~5× larger fleet, and it reached two archives no
  rebuild could (`lines-v2`, synthetic; `osm-nyc-nodes`, login-gated). Verified
  by decoding entry-by-entry against pre-migration content, client deployed
  before the manifests flipped, 59/59 synced and probed. Why touching no pack is
  legal is in [stt-packed-format-decisions.md §9](./stt-packed-format-decisions.md);
  the per-dataset drift the abandoned rebuild measured is in
  [demos-and-datasets.md §1.5](./demos-and-datasets.md); the residue is B5/B6.
- **The AV rebuild is done.** A `reoptimize` sweep had flattened and scrambled 106
  argoverse/waymo archives by reading a 3-wide `xyz` leaf at a 2-wide stride; the
  example is deleted, six generator call sites no longer fold, the bundles are
  rebuilt, and the cheap positive-proof check (`--sample 0`; one z14 tile per
  temporal bucket) is recorded in [av-cockpit.md §3](./av-cockpit.md). The live
  CDN fleet was never affected.
- **L0 — 3D tile selection is fixed.** The chassis derived its lon/lat box from
  two opposite screen corners and had no horizon guard, so the shipped cameras
  missed 20–44% of on-screen tiles and inverted past bearing ≈32°. Waves 1+2
  landed on the shared `core/geo/viewport-bounds.ts` primitive and the
  pitch×bearing matrix test is green; the browser half is now a line in L2.
  ([tile-loading-3d-2026-07.md](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/tile-loading-3d-2026-07.md))
- **L1 (fronts) — `wpc-fronts` and `wpc-fronts-pips` are synced.** Both return 200
  and decode, so `severe-weather-2024`'s overlay no longer 404-stalls.
  `LOCAL_ONLY_DATASETS` is now **empty**: `storm-4d-isolines`, `rain-flood-2019`,
  `gtfs-ch` and `storm-3d-conus` are all un-gated and verified. The gate mechanism
  stays for the next pre-sync dataset.
- **L1b — the storm-4d radar LOD pyramid has a time axis and is live.** The
  thinning grid was space-only over the whole 9.5-hour window, so z8 showed a
  median 13% of the visible bucket (0% at worst); the cell is now keyed on
  `--temporal-bucket` and z8 shows a median 65%.
  ([storm-4d §11.4](./storm-4d-greenfield-2026-07.md))
- **K1 — `stt-serve` has a capability channel.** `/metadata.json` carries a
  `capabilities` array derived from the same `EncoderSettings::required_capabilities()`
  the offline build declares with. It is **always present**, empty when the server
  encodes the capability-free shape, so its absence unambiguously means "server
  predates this key" rather than "declares nothing".
- **K7 — roadmap citations resolve by anchor, not just filename.** The gate parses
  numbered headings per document: **273 citations, 107 anchored, all resolve**
  (from 94 filename-only). A follow-on false positive is fixed too — the citation
  regex captured a lettered anchor (`§8.5a`) but the heading parser did not, so a
  real subsection read as missing. ⚠️ **A gate over `git grep` cannot see
  untracked work**, which is why the first commit of a new doc is exactly when to
  re-run it.

**Earlier waves.** Cloudflare _is_ caching the packs (`MISS` then `HIT`,
`max-age=31536000, immutable` — the earlier "not caching" claim was wrong and the
cold-start figures are edge figures) · the shipped plugin config launches
`npx -y @poopdeck.gl/mcp` with no `--allow-cli` · three release systems became
two · the showcase runs maplibre-gl v5 · the polygon outline draws per-ring ·
JSON Schemas resolve at their own `$id` · `packages/core`'s `clean` no longer
leaves a stale build stamp · the Mercator limit is one constant across both tiers
· v2 has a byte golden · `stt-validate` and `stt-serve` have real tests · cold
start is measured.

---

## Consolidation ledger (pre-split)

> Also kept verbatim in both repositories, for the same reason as the
> discharged ledger above. The retired-doc mapping is what makes an old
> citation recoverable, and an old citation can appear on either side.

Three consolidations have run: 2026-07-24 (26 records → 10), 2026-07-26 (re-verified
every open claim and rewrote the register as the backlog above), and 2026-08-03
(collapsed the discharged chain to one line each and rehomed its durable lessons).
**Git history preserves every retired file verbatim** — nothing was lost, only
de-duplicated, re-verified, and stripped of wave logs, agent-process narration,
and dated external SoTA surveys.

| Retired                                                                                                            | Durable content now in                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `space-time-lod-2026-07.md`, `preprocessing-framework.md`, `stt-optimize-intelligence-2026-07.md`                  | [stt-packed-format-decisions.md](./stt-packed-format-decisions.md) — measured baselines, the advisor "measure, don't model" evidence, and both programs as counted-out entries with triggers      |
| `naming-types-consistency-2026-06.md`                                                                              | format decisions (frozen wire tokens) + [renderer-architecture.md](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/renderer-architecture.md) (codegen CI-diff gate)                  |
| `sedona-integration-2026-07.md`                                                                                    | [db-input-adaptors.md](./db-input-adaptors.md) §8 — counted out, with the arrow-57-vs-59 containment note and a capability-shaped revival trigger                                                 |
| `kind-parity-campaign-2026-07.md`, `maplibre-parity-campaign-2026-07.md`, `three-backend-sota-campaign-2026-07.md` | [renderer-architecture.md](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/renderer-architecture.md) — backend tiering, the ratified adopt-or-cut verdicts, and the reusable gotchas |
| `full-ecosystem-audit-2026-07.md`                                                                                  | retired: §1 criticals closed; the backend parity matrix is now CI-generated (renderer-architecture §4); the untriaged backlog was not carried forward                                             |
| `scrub-lod-2026-07.md`                                                                                             | [playback-and-loading.md](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/playback-and-loading.md) §7 — the correctness contract, the G5 negative result, and the QoE criteria       |
| `cosmos-drive-dreams.md`, `rain-flood-demo-2026-07.md`, `dataset-candidates-2026-07.md`                            | [demos-and-datasets.md](./demos-and-datasets.md) — licence register, BLOCKED list, time-bombs, per-demo gotchas                                                                                   |
| `ai-suite-skills-mcp-2026-07.md`                                                                                   | [ai-suite.md](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/ai-suite.md)                                                                                                           |
| `shipping-2026-07.md`                                                                                              | [shipping.md](./shipping.md)                                                                                                                                                                      |
| `evaluations/` (4 files)                                                                                           | deleted — reference-only third-party model reviews from December 2025, written against a tree that predates the `packages/layers` rename, packed v2, and the render-kernel abstraction            |

### The contract rule

Three per-demo/per-campaign docs survived consolidation because they are **not
campaign logs — they are live contracts that source code cites as normative**,
with section anchors:

- **`av-cockpit.md`** — **41** section-anchored citations in this tree, almost
  all in `scripts/data-generation/*.py`, plus downstream ones in
  `poopdeck:packages/layers/src/layers/core/animated-bounding-box-layer.ts` and
  `poopdeck:examples/showcase/src/components/av/*`.
  `scripts/data-generation/av_common.py:8` instructs extractor authors **not to
  deviate from** its §2 data contract.
- **`storm-4d-greenfield-2026-07.md`** — its §9.1 per-archive layer/field schema
  is called "the binding contract" by eight generators (`nexrad_volume.py`,
  `goes_cloudtop.py`, `storm4d_outages.py`, `storm4d_sounding.py`,
  `storm4d_wind3d.py`, …), **13** citations in this tree, and by
  `poopdeck:examples/showcase/src/datasets.ts:892`.
- **`tile-loading-3d-2026-07.md`** is now poopdeck.gl's contract, cited there by
  the source and test files that read its §4 bounds contract and the F/A change
  identifiers in §5. The rule below is why it survived the split rather than
  being deleted.

**The rule going forward:** before retiring a record, `git grep` its filename. If
source code cites it as binding, it is a contract — move the contract to a spec
page or keep the record; do not delete it. The same rule applies to **section
numbers**: renumbering a cited heading breaks the anchor gate, so compact within
a section rather than resequencing it. Both filename and anchor halves of the
check are now automated (`.github/scripts/check-roadmap-citations.mjs`).
