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
discharged on 2026-08-14 by a container-only v2→v3 migration; its investigation
is retained below because it explains why rebuilding was rejected.

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
jobs — is recorded in T2 and in [db-input-adaptors.md §5](./db-input-adaptors.md).

### S — The split's own tail

**S1. The downstream vendor pin has never been exercised.** poopdeck.gl's
`.stt-sync.json` ships at `"ref": "UNPINNED"`, so its `pnpm stt:check` refuses
to report success without a local checkout of this repository — correct (a gate
that cannot verify must not pass), but it means the CI half has never run. Three
artifacts ride on it: the 24 vendored doc pages, `conformance/vectors/`, and
this repository's `project-status.json`. **Accept:** the split commit here is
pushed, `pnpm stt:sync --ref <sha>` downstream records it, and one CI run of
`pnpm stt:check` passes with no sibling checkout.

**S2. The first post-split release has not been cut.** crates.io and npm both
read 0.7.0, which is now a coincidence rather than a promise. **Accept:** `0.8.0`
is cut here on its own, with a CHANGELOG entry stating the new relationship
explicitly so a consumer reading two version numbers does not infer lockstep
that no longer exists ([repo-split-2026-08.md §2.3](./repo-split-2026-08.md)).

**S3. The archive fleet moved and nothing has re-verified it.** The 76 GB local
fleet lived at `poopdeck:examples/showcase/public/data`, a path in a repository
that no longer exists here; it is now `data-fleet/`, and `r2-sync.sh`,
`rebuild-fleet-v3.sh`, `patch-manifest-metadata.mjs`,
`tools/fleet-order-audit.sh` and every data-generation script were repointed by
rewrite, not by running them. **Accept:** one `stt-optimize inspect --sample 0`
over a stem under the new root, and one `r2-sync.sh` dry run, both from a clean
checkout.

### B — Blocking (retained discharge record)

**B4 (discharged 2026-08-14). Historical premise: the published fleet was still
`formatVersion: 2`, and rebuilding it was a program, not a batch job.** 0.6.0's
writer emits v3 only; its reader opens v2
and v3, so the live fleet keeps working untouched and this is an upgrade rather
than an outage. 54 of the 60 referenced archives are v2 (~18 GB, ~380 M
features); `storm-cells`, `storm-field` and `storm-tracks` are already v3.

**A rebuild is not automatically a replacement, and the checks that existed did
not say so.** The first run produced a `wildfires` archive holding 175 features
against the live 4,600 — and it passed both gates, because the manifest really
did say `formatVersion: 3` and `stt-validate` really did exit 0 on a flawless
encoding of almost no data. NIFC now returns 15 source perimeters for 2020–2023
where it once returned hundreds; `tools/stt-generate/src/datasets/wildfires.rs`
has carried a warning about exactly this since 2026-07-29. `scripts/rebuild-fleet-v3.sh`
therefore compares `feature_count` against the archive currently serving and
fails a dataset on a material shortfall, and reports anything above 1.5× for
review rather than passing it silently.

**The API tier, measured (2026-08-14).** Eight datasets, and only three were the
boring case:

| dataset          | wall   | features vs live         | verdict                              |
| ---------------- | ------ | ------------------------ | ------------------------------------ |
| `earthquakes-v2` | 231 s  | 522,978 / 522,982 1.000× | accepted                             |
| `hurricanes`     | 370 s  | 200,074 / 193,020 1.037× | accepted (fresh IBTrACS storms)      |
| `animals`        | 1227 s | 11.19 M / 10.58 M 1.058× | accepted                             |
| `drifters`       | 3006 s | 9.64 M / 9.05 M 1.066×   | accepted                             |
| `ais-all-us`     | 1916 s | 124.8 M / 19.3 M 6.453×  | **REVIEW** — recipe drift, see below |
| `satellites`     | 66 s   | 8.28 M / 24.2 M 0.342×   | **refused** — bucket drift, not loss |
| `wildfires`      | 2 s    | 175 / 4,600 0.038×       | **refused** — genuine upstream loss  |
| `flights`        | 27 min | 257.9 M / 43.5 M 5.925×  | **REVIEW** — needed `--streaming`    |

Three distinct failure modes, none of which is "the format migration went
wrong":

- **`wildfires` — the data is gone upstream.** Unfixable here; it stays v2.
- **`satellites` — the shipped archive used a 5-minute temporal bucket and
  today's builder auto-picks 1 hour**, so the same 16,087 satellites (MORE than
  the live archive's ~12,700) land in 2,125 tiles instead of 24,480. The
  generator does not expose `--temporal-bucket`, so reproducing it needs
  `--skip-build` plus a direct `stt-build`, AND a pinned `--start-time` with a
  matching `datasets.ts` `timeRange` — propagation runs from _now_, and a
  mismatched range renders the demo empty.
- **`ais-all-us` — the opposite drift.** Today's default `--sample-minutes 0`
  preserves every usable row, which is this document's own no-thinning ground
  rule; the shipped archive was built thinned. 6.45× the features and 2.2 GB
  against 0.51 GB. The rebuild is arguably the more correct artifact and the
  size is the reason it is a decision rather than a detail.

**Both of the two biggest archives come back 5–6× larger, and that is the
finding that decides this item's scope.** `ais-all-us` 6.45× / 2.2 GB vs
0.51 GB; `flights` 5.93× / 4.2 GB vs 0.81 GB. Neither is a defect — today's
defaults preserve every usable row at every zoom, which is this document's own
no-thinning ground rule, while the shipped archives were built when thinning was
applied at the source. The rebuilds are the more honest artifacts. They are also
**~5× the bytes** against a fleet that is ~18 GB today, so a wholesale rebuild at
current defaults plausibly lands near 60–90 GB of R2 storage, with a
correspondingly heavier client load on those demos. Decide the density per
dataset BEFORE uploading; the gate prints the ratio precisely so that choice is
explicit rather than discovered on the bill.

**Peak RAM, not disk, is what kills a large build.** `flights` (23.4 M features)
was SIGKILLed at zoom 8 twice — first with 19 GB free, then again with 28 GB
free, which is what disproved the disk theory the first failure suggested.
`--streaming` (write each zoom as it completes, at some parallelism cost) cleared
zooms 8/9/10 on the first attempt and finished in 27 minutes. Carry the trade the
conformance record notes: `--streaming` builds emit no content fingerprint.
`stt-build` separately spills pack payloads past `--pack-memory-budget` to a temp
file in the OUTPUT dir, so a large build still needs far more transient disk than
its finished archive implies; the driver refuses below 25 GB rather than dying
mid-build.

Three tiers, and they are not equally tractable:

- **API-sourced** (`stt-generate`, no staged input) — reproducible from a clean
  checkout. Rebuild times scale with tiles, not bytes: 231 s for 523 K
  earthquake points, 370 s for the hurricane tracks.
- **Staged-source** — the weather / storm / AV families come from the Python
  generators in `scripts/data-generation` over ~26 GB of staged inputs, not from
  `stt-generate`. A separate pass with its own toolchain.
- **No rebuild path at all** — `lines-v2` (synthetic, no recipe anywhere),
  `nyc-taxi-od-summary` (no generator), `osm-nyc-nodes` (needs a login-gated
  Geofabrik full-history extract), and now `wildfires` (upstream coverage
  collapsed). These stay v2. That is not a gap in the plan; it is the case the
  reader's v2 read window was kept for, and the argument for keeping it is
  written into [the packed spec §9.1](../spec/stt-packed-format.md).

**Ordering, which is load-bearing.** Packs are content-addressed, so uploading
them is invisible to the live site; only `manifest.json` is the switch. Upload
immutable objects → deploy the reader → flip manifests. The reverse order hands
a v3 manifest to a deployed reader that knows only v2. A rebuilt polygon archive
additionally declares `triangles-partial`, which a client without the decoder
backfill refuses at open — loud by design, and another reason the client ships
first.

**DISCHARGED 2026-08-14, by not rebuilding.** The rebuild framing above was
wrong, and the table that proves it is the reason: v2 → v3 is **container-only**,
so an archive can be promoted without re-deriving anything. `stt-core`'s
`migrate_dataset_v2_to_v3` re-encodes the directory under codec v6 and rewrites
the manifest, and **touches no pack** — legal because a reader accepts object
magic `2..=3` on every `.sttp` independently of `formatVersion` (packed spec
§9.4, written in the same pass).

Result: **59 of 64 local datasets migrated, 76 MB of new directory objects, zero
packs rewritten**, versus days of compute and a ~5× larger fleet. It also reached
two archives a rebuild never could — `lines-v2` (synthetic, no recipe) and
`osm-nyc-nodes` (login-gated source). Verification was decode-based, not
parse-based: every dataset reopened and decoded entry-by-entry against its
pre-migration content, the run aborting on the first difference.

Live as of 2026-08-14: client deployed first (it reads v2 AND v3, so there was
no exposure window), then 59/59 datasets synced and probed — all serving
`formatVersion: 3` + directory v6 + a variants registry, with pack objects still
answering `STTP\x02`, which is the migration's whole thesis visible on the wire.
The 12-demo render probe passes against live R2.

**What remains open, and it is small:**

- **The five summary-tier archives stay v2** (`earthquakes-summary`,
  `goes-glm-lightning`, `nyc-od-quadbin`, `nyc-taxi-od-summary`,
  `osm-nyc-changesets`). Migration refuses them on purpose — a v2 directory has
  no column saying which entries are aggregates, so the raw/summary split v3
  needs cannot be recovered. They serve correctly through the reader's v2
  window; reaching v3 requires a rebuild.
- **Two staged rebuilds await a density decision**, not a technical one:
  `ais-all-us` 6.45× and `flights` 5.93× against what ships, because today's
  defaults honour the no-thinning rule and the shipped archives were built
  thinned. Both are built, validated and NOT published.
- **`wildfires` and `satellites` rebuilds are refused/parked** per the table
  above; both remain live and correct at their migrated v3 container.

### L — Live defects on the published fleet

**L1. One atlas sidecar is missing from R2, and `r2-sync.sh` structurally cannot
upload it.** The Neural-State Atlas is still gated
(`ATLAS_ARCHIVES_SYNCED = false`) and the reason is now precise. Probing
2026-08-03: all three `neural-atlas-*` archives, all three `<stem>.meta.json`
files and both `neural-atlas-node-{index,series}.bin` blobs return **200** — the
2026-07-31 negative-cache window has expired and the `[sidecar]` pass works. But
**`/data/neural-atlas.json` returns 404**, and that is the file
`NeuralAtlasImpl.tsx:79` fetches _first_ (`SIDECAR_URL` — the generator sidecar
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
fixed in the tree**; of the four the fix could not reach, the two below are
this repository's — both are manifest-and-fleet defects. DX1 (a Vite 8 bundler
bug) and DX4 (an intermittent deck.gl assertion) moved downstream with the
renderer.

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

**DX2. The flagship dataset's published manifest is patched locally but not
deployed.** `data/earthquakes-v2/manifest.json` — the first file anyone
following the quickstart fetches — served `"name": "earthquakes-v2.new"` with an
empty `description` and an empty `attribution`, uncredited USGS data. The local
manifest now carries a real name, a one-line description and the ComCat
attribution (`scripts/patch-manifest-metadata.mjs`), verified to open through
`STTArchive` unchanged; every content-addressed object is untouched, so
publishing is a one-object manifest pass. **Accept:** `scripts/r2-sync.sh
earthquakes-v2` run with R2 credentials, and a plain GET of the live manifest
shows the three fields.

**DX3. The metadata gap is fleet-wide, not one dataset.** `--scan` over the 70
local packed datasets: **25 carry a build-scratch name** (`ais-all-us-new`,
`hurricanes.new`, `lines v2`, …), **66 have no description**, and **67 have no
attribution** — for datasets whose licences are the reason several of them can be
published at all. The curated copy already exists in
`examples/showcase/src/content/demoMeta.ts` (`tagline` + `dataSources` with name,
url and licence per demo), so this is a mapping job plus a manifest-only
republish, not a rebuild. **Accept:** `patch-manifest-metadata.mjs --scan` shows
no scratch names and no empty attribution across the shipped stems, and the
manifest pass is synced.

### T — Claims the repo makes that the world does not back

**T1. The published repository URL 404s — and the split doubled the problem.**
`https://github.com/BertCh/spatiotemporal-tiles` returns **404** (re-verified
2026-08-24; the repo is private). It is the `repository`/`homepage`/`bugs` on
all four published crates and the releases page both READMEs send
`cargo install` users to. Since 2026-08-26 there is a **second** URL in the same
state: `https://github.com/BertCh/poopdeck.gl`, which does not exist on the
remote at all yet, and which this repository's docs, the vendor-sync fetch path
and every downstream package manifest now name. Every ordering constraint that
once made this awkward is discharged. **Accept:** both repositories exist and
are public, and `sync-stt.mjs`'s tarball fetch resolves without credentials.

**T2. GitHub Actions has never run; the CI gates are config that only ever
executed by hand.** Zero bot commits across the repo's history and no release PR.
_(Not re-verifiable here: no `gh` CLI in this environment. Last verified
2026-07-24.)_ `ci.yml` now carries `cargo fmt --check`, the curated clippy deny
set, `oxlint` and `oxfmt --check` — the older claim that the gates were
"deliberately absent" no longer describes the file. Running every job locally on
2026-07-31 found **four red**, each invisible to `cargo test --workspace`:
`rust-feature-lanes` (3 of 6), `rust-all-features`, `rust-lint` (2 files) and
`ts-lint` (11 files). All are fixed. The durable half of that finding — that two
DB input adaptors sit behind non-default features where the default suite will
never report a shared-type change — lives in
[db-input-adaptors.md §5](./db-input-adaptors.md). **Accept:** one green run on
GitHub's own runners.

### K — Known defects with a named fix

Each is small, real, and has its analysis written down where it belongs. None
blocked the completed 0.6.0 release. K3–K6, K8 and K10 are renderer defects and
moved downstream; the numbering is left alone so existing references keep
resolving.

**K2. `stt-validate` reports structural drift on correct archives.** The per-tile
exact-integer quantizer refuses a column on outlier-inflated inputs, which
changes the column _set_ tile to tile; `part_offsets` adds
`<absent> vs List<UInt32>` entries for the same reason. The archive is right and
the report is a false positive. The proper fix is the dataset-global
attribute-range pin, which shares the two-pass-build prerequisite with the
dictionary hoist. ([format §10.3–§10.4](./stt-packed-format-decisions.md))

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

**K11. `metadata.bounds` is a bbox of CENTROIDS and does not bound the data — and
the republish it was meant to ride has now passed.** `stt-build` fills manifest
bounds from `input::calculate_bounds`, which takes min/max of each
`ParsedFeature.lon`/`.lat` — the geometry's **centroid** (verified still true
2026-08-03). The tiler addresses tiles by **vertex**, so on any line / polygon /
multi-point archive the occupied tiles provably extend past the declared extent.
(`calculate_bounds` separately skips the `(0,0)` sentinel, so even a pure-point
archive can hold data outside its bounds.) Blast radius: the showcase frames its
opening camera from these bounds, and `stt-validate` and the MCP
`describe_dataset` both report them as the dataset's bbox — all three understate
the true extent. The sibling item is the same shape: `zRange` exists as a layer
prop but there is **no `z_range` on `Metadata`/`ArchiveMetadata`** and no
`--elevation-column` to populate it, so no consumer can discover that a dataset is
volumetric and altitude-aware selection stays hand-configured per demo. Both were
recorded as "fold into B2 rather than schedule a republish" — B2 shipped without
them, so they now need their own rebuild window. **Accept:** the builder computes
the real geometry bbox, `z_range` lands as an additive
`skip_serializing_if = "Option::is_none"` field (existing manifests round-trip
byte-identically), and both ride the next rebuild rather than a dedicated one.
([tile-loading-3d-2026-07.md §6, §6b](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/tile-loading-3d-2026-07.md))

**K12. The per-feature LOD floor is consumed but never computed, so a cross-codebase
constant has to be policed instead.** `stt-build` reads a feature's LOD floor from a
property named by `--min-zoom-field` (`feature_min_zoom`,
`crates/stt-build/src/tiler.rs:731`) and confines the feature to a `[min_zoom,
max_zoom]` band from there. It computes that floor **nowhere**: the values are baked
upstream by the generation scripts (`scripts/data-generation/mrms_refloor.py`,
`mrms_volume.py`, `nexrad_volume.py`, the AV extractors) against a thinning grid whose
temporal bucket is a constant in a second codebase. Nothing asserted the two equal, and
when they diverged on 2026-07-28 the archive still built, still validated and still
rendered — at a median 13 % (worst 0 %) of the features visible in the displayed
bucket, with no decode-time signal at all. `crates/stt-build/src/lod_bucket.rs` (SH-4)
closed that hole by stamping the grid's bucket width into the Parquet footer and
hard-erroring on a mismatch. That is the right guard for the architecture, but it
leaves the architecture in place: a progressive or thinned build is reachable only by
users who also run our Python, and the guard's two warn paths — footer absent, column
absent — are exactly the cases it cannot check.

**Named fix:** a producer-side assigner inside `stt-build` that computes the floor from
the features it is already parsing, so the floor and the bucket come from one place and
cannot disagree. Shape, three quarters of it taken from COGP's `assign_levels`
([stt-packed-format-decisions.md §6](./stt-packed-format-decisions.md)): grid cells
keyed on the **resolved** temporal bucket (the SH-4 lesson is a precondition here, not
a follow-up — COGP's own grid is space-only because it has no time axis); an extent gate
for lines and polygons (enter as soon as the bbox diagonal is resolvable) against cell
competition for points; and a **hashed row index as the final tiebreak** after sort key
and bbox diagonal, so assignment is deterministic under input reordering and builds stay
byte-reproducible. **Accept:** an archive built from a floor-free input carries per-zoom
feature counts equal to one built from a baked column on the same source; the SH-4
warn paths become the legacy path rather than the normal one; and `lod_bucket.rs`
keeps policing externally baked columns without being the only way to obtain one.

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

- **`av-cockpit.md`** — 44 section-anchored citations across
  `scripts/data-generation/*.py`, `packages/layers/src/layers/core/animated-bounding-box-layer.ts`,
  and `examples/showcase/src/components/av/*`. `scripts/data-generation/av_common.py:8`
  instructs extractor authors **not to deviate from** its §2 data contract.
- **`storm-4d-greenfield-2026-07.md`** — its §9.1 per-archive layer/field schema
  is called "the binding contract" by eight generators (`nexrad_volume.py`,
  `goes_cloudtop.py`, `storm4d_outages.py`, `storm4d_sounding.py`,
  `storm4d_wind3d.py`, …) and by `examples/showcase/src/datasets.ts:3122`.
- **`tile-loading-3d-2026-07.md`** — its §4 bounds contract and the F/A change
  identifiers in §5 are cited by **35** source and test files across `core`,
  `layers`, `three`, `maplibre`, `cesium` and the showcase.

**The rule going forward:** before retiring a record, `git grep` its filename. If
source code cites it as binding, it is a contract — move the contract to a spec
page or keep the record; do not delete it. The same rule applies to **section
numbers**: renumbering a cited heading breaks the anchor gate, so compact within
a section rather than resequencing it. Both filename and anchor halves of the
check are now automated (`.github/scripts/check-roadmap-citations.mjs`).
