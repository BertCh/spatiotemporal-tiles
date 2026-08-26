# Changelog

This file summarizes changes that affect the STT **format and toolchain**. The
GitHub Release that cargo-dist creates for each tag is the per-crate record.
Renderer changes are recorded downstream, in the
[poopdeck.gl changelog](https://github.com/BertCh/poopdeck.gl/blob/main/CHANGELOG.md).

## 0.8.0 — 2026-08-26

### Repository split

- The `@poopdeck.gl/*` TypeScript renderers, the showcase site and the
  measurement harnesses moved to their own repository,
  [BertCh/poopdeck.gl](https://github.com/BertCh/poopdeck.gl), with their full
  commit history. **Nothing about the format, the CLIs or the crates changed** —
  same names, same flags, same bytes on disk.
- New `conformance/` directory: the six reader-side golden vectors (previously
  in the TypeScript reader's test tree, though always produced by the Rust
  writer) plus `make-vectors.sh` and a README explaining how a third-party
  reader consumes them. They are byte-pinned exactly as before.
- New generated contracts under `docs/spec/`: `av-palettes.json` (the AV
  cockpit's cross-language palette contract, from `av_common.py`) and
  `stt-generate-datasets.json` (the reference generator's subcommand
  inventory). Both replace tests that used to read across the two trees, and
  both are now readable by anyone.
- The built archive fleet moved from `examples/showcase/public/data` to
  `data-fleet/`; the publishing and generation scripts follow it.
- The golden-pin gate gained a second, **verified** declaration —
  `Pin-Relocation: old -> new` — so a byte-identical move of a pinned tree stops
  having to spend the `Rebuild-Window: R1` signal, which means "the fleet needs
  re-uploading".
- Rationale, the full inventory, and the costs accepted:
  [repo-split-2026-08.md](docs/roadmap/repo-split-2026-08.md).

### Fixed

- **The declared MSRV was wrong.** `rust-version` promised 1.87 while the locked
  tree required 1.88 (`geo` 0.33 had raised its own floor), so
  `cargo install spatiotemporal-tiles` on 1.87 failed a resolver check the
  manifest said would pass. It is now 1.88, bisected against the lockfile rather
  than against this repository's own source. If you pinned 1.87 on the strength
  of the old manifest, this release is the correction, not a new requirement —
  1.87 has not worked since `geo` 0.33 landed.

### Nothing else changed

No format change, no CLI change, no behaviour change: archives written by 0.7.0
and 0.8.0 are byte-identical, and the version exists to mark the repository
split. `formatVersion` stays 3 and `directoryVersion` stays 6.

**crates.io and npm are no longer released in lockstep.** They last agreed at
0.7.0 by history; from here each moves on its own. What relates the two stacks
is the archive's `formatVersion`, declared in `project-status.json` on both
sides — read that, not the version string.

## 0.6.0 — 2026-08-13

### Format and compatibility

- Writers now emit packed `formatVersion: 3` with directory codec v6.
- The manifest carries a required variant registry, and directory entries carry
  `variant_id`, preventing raw and summary tiles at the same space/time address
  from colliding.
- Readers retain a read-only compatibility window for packed v2/directory v5.
  Format v1 remains unsupported.
- Existing raw-only v2 archives can be migrated container-only; summary-tier v2
  archives must be rebuilt because their missing variant identity cannot be
  inferred safely.

### Toolchain and packages

- The Rust workspace and seven public `@poopdeck.gl/*` packages are aligned on
  0.6.0. _(The two registries were still in lockstep at this release; they were
  separated by the 2026-08-26 repository split above.)_
- The umbrella Rust crate installs five binaries: `stt-build`, `stt-optimize`,
  `stt-validate`, `stt-bundle`, and `stt-serve`.
- The Cesium backend remains experimental and private in the workspace after its
  last npm release at 0.5.0.
- Public Three.js renderer exports use `STT*` names to avoid collisions with
  deck.gl classes.

### Upgrade notes

- Rebuild new archives with the 0.6 toolchain. Existing packed v2 archives
  continue to open read-only.
- Tile cache keys include the variant axis; the first browser load after an
  upgrade can be cold.
- Review the package changelogs for any public API used directly; they live in
  [the poopdeck.gl repository](https://github.com/BertCh/poopdeck.gl/tree/main/packages).

## Earlier releases

Earlier package histories are recorded in the package changelogs and Git tags.
The project is pre-1.0; read the
[status and compatibility policy](https://github.com/BertCh/poopdeck.gl/blob/main/docs/intro/status-and-support.md) before
upgrading across a minor release.
