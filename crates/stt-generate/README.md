# stt-generate

Unified CLI for generating the project's bundled showcase datasets. Each
subcommand fetches a real public data source (USGS earthquakes, NOAA
IBTrACS hurricanes and AIS vessel tracks, OpenSky flights, NYC TLC trips,
Montréal BIXI rides, CelesTrak satellites, GBIF animal tracking, OSM edit
history, NEXRAD storm radar, …), normalises it into GeoParquet, and shells
out to `stt-build` — so every output is a packed STT dataset directory,
built publish-quality.

> **Not yet published to crates.io** — build from the repo:
>
> ```bash
> git clone https://github.com/BertCh/spatiotemporal-tiles
> cd spatiotemporal-tiles
> cargo install --path crates/stt-generate
> cargo install --path crates/stt-build   # stt-generate shells out to it
> ```

## Example

```bash
# The no-extra-setup trio (earthquakes, hurricanes, wildfires):
stt-generate all --output-dir examples/showcase/public/data --skip-existing

# One dataset with its own parameters:
stt-generate earthquakes \
  --start-date 2020-01-01 --end-date 2024-12-31 \
  --min-magnitude 4.0 \
  --output examples/showcase/public/data/earthquakes.stt

# Per-dataset flags:
stt-generate flights --help
```

`all` builds only the datasets that need no per-run parameters; the rest
(dates, OSRM routing endpoints, local inputs like the BIXI zip) are run
individually — see the per-subcommand `--help`.

## Relation to the other crates

A convenience layer over [`stt-build`](../stt-build) (which does the actual
tiling/encoding via [`stt-core`](../stt-core)). Outputs are verified with
[`stt-validate`](../stt-validate) and consumed by the showcase app in
`examples/showcase`.

## Docs

- [Subcommand catalog](../../docs/api/cli-reference.md#stt-generate)
- [Data generation guide](../../docs/guides/data-generation.md)
- [Packed format spec](../../docs/spec/stt-packed-format.md)

License: MIT.
