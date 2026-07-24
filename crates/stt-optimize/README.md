# stt-optimize

Analyzer, flag-recommender, and tileset profiler for STT builds. It reads a
source before the build and the packed dataset after it:

| Subcommand              | Reads          | Reports                                                                   |
| ----------------------- | -------------- | ------------------------------------------------------------------------- |
| `analyze` / `recommend` | GeoParquet     | Recommended zoom range and temporal bucket, plus the evidence behind them |
| `inspect`               | packed dataset | Per-zoom directory stats, dedup and compression ratios, per-column cost   |
| `doctor`                | packed dataset | Severity-ranked findings with the remediation flag for each               |
| `diff`                  | two datasets   | Total / per-zoom / per-column deltas, with a `--fail-on-growth` size gate |
| `order-audit`           | packed dataset | Simulated range-read cost per blob ordering                               |

`analyze`/`recommend` profile spatial extent, temporal distribution, geometry
mix, and density, and trial-encode a deterministic sample through the real
encoder plus zstd to calibrate the size estimates. The same logic runs inside
`stt-build --auto`; the CLI exists to run it standalone, inspect the reasoning
(`--verbose`), or emit machine-readable reports.

**Advisors.** Beyond zoom range and bucket, an advisor layer suggests flags
across the wider `stt-build` surface (coordinate/attribute quantization,
temporal LOD, wire layout, per-tile budgets). Where it matters the projection is
measured rather than extrapolated, and each suggestion carries the
dataset-specific rationale and a confidence grade. Anything that discards or
degrades data is marked lossy and stays opt-in: lossy levers never join the
suggested command (`recommend --show-command`) and are never auto-applied by
`stt-build --auto`. Only the reversible byte-level levers are applied, and only
under `--auto encode`. `recommend --explain` prints the full evidence table.

**`doctor`** lints a built tileset, citing that tileset's own measured numbers in
each finding. The rule catalog covers raw Float64 property columns,
near-incompressible hash-like feature ids, constant/all-null columns,
shallow-pyramid "z0 bombs", whole-load directories past 10k entries, oversized
tiles, and missing summary tiers. `--strict` exits non-zero on any
Warning-or-worse finding — the CI gate counterpart to `diff --fail-on-growth`.

**Style hints.** The crate also houses the profiler behind `stt-build
--style-hints` (`analysis::properties::profile_properties`): bounded per-property
profiles (numeric percentiles with a `[min, ~p97]` `suggested_domain`,
categorical cardinality) plus a suggested playback duration and a layer-type
hint, baked into archive metadata as a versioned `style_hints` block. Hints are
render defaults a reader may override; old readers are unaffected.

> **Internal implementation crate** of
> [`spatiotemporal-tiles`](https://crates.io/crates/spatiotemporal-tiles):
> the analysis _library_ (the facade's `optimize` module). The
> `stt-optimize` CLI ships with the facade:
>
> ```bash
> cargo install spatiotemporal-tiles
> ```

## Example

```bash
# Analyze a GeoParquet input:
stt-optimize analyze --input data.parquet --time-field timestamp \
  --time-format unix-ms

# Print a copy-pasteable stt-build invocation:
stt-optimize recommend --input data.parquet --time-field timestamp \
  --time-format unix-ms --show-command

# Machine-readable, for pipelines:
stt-optimize analyze --input data.parquet --time-field timestamp \
  --time-format unix-ms --format json -o report.json

# Profile a built packed dataset (per-zoom, dedup, per-column costs):
stt-optimize inspect --archive my-dataset/ --sample 200

# Compare two builds; fail CI if the re-encode grew more than 5%:
stt-optimize diff --before old-dataset/ --after new-dataset/ --fail-on-growth 5

# Lint a built dataset; fail CI on any Warning-or-worse finding:
stt-optimize doctor --archive my-dataset/ --strict
```

## Relation to the other crates

Reads inputs and packed datasets via [`stt-core`](../stt-core);
[`stt-build`](../stt-build) calls the library entry point (`recommend_for`)
when invoked with `--auto`, applying the zoom-range and temporal-bucket
recommendations (compression is not applied — the packed format is
zstd-only). `stt-build --auto encode` additionally applies the advisors'
non-lossy byte-level levers; lossy advice is only ever logged as a
suggestion. `stt-build --style-hints` calls this crate's property profiler
to bake the `style_hints` metadata block.

## Docs

- [CLI reference](../../docs/api/cli-reference.md#stt-optimize)
- [Tuning your tiles](../../docs/guides/tuning-tiles.md) — the
  measure → interpret → decide loop, end to end
- [`stt-build` flag reference](../../docs/api/cli-reference.md#stt-build)
- [Packed format spec](../../docs/spec/stt-packed-format.md)

License: MIT.
