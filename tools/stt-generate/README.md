# stt-generate

Unified CLI for generating the project's bundled showcase datasets. Each
subcommand fetches a real public data source (USGS earthquakes, NOAA
IBTrACS hurricanes and AIS vessel tracks, OpenSky flights, NYC TLC trips,
Montréal BIXI rides, CelesTrak satellites, GBIF animal tracking, OSM edit
history, NEXRAD storm radar, static GTFS transit feeds, NOAA National Water
Model river discharge, …), normalises it into GeoParquet, and shells
out to `stt-build` — so every output is a packed STT dataset directory,
built publish-quality.

> **Not published to crates.io, and not a member of the root workspace** —
> this directory is its own workspace with its own `Cargo.lock`, so its heavy
> dep tree (osmpbf, nexrad-\*, reqwest, tokio) and its higher MSRV stay off the
> four published crates. Build from the repo:
>
> ```bash
> git clone https://github.com/BertCh/spatiotemporal-tiles
> cd spatiotemporal-tiles
> cargo install --path tools/stt-generate
> cargo install --path crates/spatiotemporal-tiles  # stt-generate shells out to stt-build
> ```
>
> `cargo build` / `cargo test` at the repo root do NOT cover this crate. Run
> them from `tools/stt-generate` (or with `--manifest-path
tools/stt-generate/Cargo.toml`).

## Example

```bash
# One dataset with its own parameters:
stt-generate earthquakes \
  --start-date 2020-01-01 --end-date 2024-12-31 \
  --min-magnitude 4.0 \
  --output data-fleet/earthquakes.stt

# Per-dataset flags:
stt-generate flights --help
```

There is deliberately no "build everything" subcommand: every generator needs
its own per-run parameters (dates, OSRM routing endpoints, local inputs like
the BIXI zip), so datasets are run individually — see the per-subcommand
`--help`.

## Relation to the other crates

A convenience layer over [`stt-build`](../../crates/stt-build) (which does the
actual tiling/encoding via [`stt-core`](../../crates/stt-core)). Outputs are
verified with `stt-validate` (a binary of
[`spatiotemporal-tiles`](../../crates/spatiotemporal-tiles)) and consumed by the
showcase app in `examples/showcase`.

## Docs

- [Subcommand catalog](../../docs/api/cli-reference.md#stt-generate)
- [Data generation guide](../../docs/guides/data-generation.md)
- [Packed format spec](../../docs/spec/stt-packed-format.md)

License: MIT.
