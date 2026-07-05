# stt-optimize

Analyzer, flag-recommender, and tileset profiler for STT builds. On the
input side (`analyze`/`recommend`) it inspects a source's spatial extent,
temporal distribution, geometry mix, and density — plus a measured sample
encoding through the real encoder + zstd — then recommends `stt-build`
settings (zoom range, temporal bucket) so you don't hand-tune them per
dataset. The same logic is what runs inside `stt-build --auto`; the CLI
exists to run the analysis standalone, inspect the reasoning (`--verbose`),
or emit machine-readable reports. On the output side (`inspect`/`diff`/
`doctor`) it profiles **built** packed datasets — per-zoom directory stats,
dedup and compression ratios, per-column compressed cost — and compares two
builds with a CI-friendly `--fail-on-growth` size gate.

On top of the basics sits an **advisor layer**: evidence-based suggestions
for the wider `stt-build` flag surface (coordinate/attribute quantization,
temporal LOD, wire layout, per-tile budgets). Advisors never speak from
folklore — where it matters they trial-encode the loader sample through the
real encoder and report the measured delta; each suggestion carries the
dataset-specific rationale, a projection, and a confidence grade. Anything
that discards or degrades data is marked **lossy** and stays a per-dataset
opt-in: lossy levers never join the suggested command (`recommend
--show-command`) and are never auto-applied by `stt-build --auto` — only the
reversible byte-level levers are applied, and only under `--auto encode`.
Inspect the full evidence with `recommend --explain`.

**`doctor`** turns the inspect numbers into a lint pass over a built
tileset: severity-ranked findings (`CRITICAL`/`WARNING`/`INFO`), each citing
the tileset's measured numbers, with the concrete remediation flag(s) and —
where derivable from the measured column costs — a labeled projected win.
The rule catalog productizes this repo's recurring manual optimization
passes: raw Float64 property columns, near-incompressible hash-like feature
ids, constant/all-null columns, shallow-pyramid "z0 bombs", whole-load
directories past 10k entries, oversized tiles, and missing summary tiers.
`doctor --strict` exits non-zero on any Warning-or-worse finding — a CI
gate, like `diff --fail-on-growth`.

The crate also houses the **style-hints profiler**
(`analysis::properties::profile_properties`) behind `stt-build
--style-hints`: bounded per-property value profiles (numeric percentiles
with a `[min, ~p97]` `suggested_domain`, categorical cardinality) plus a
suggested playback duration and a layer-type hint, baked into archive
metadata as a versioned `style_hints` block. Hints are render *defaults*
readers may always override; old readers are unaffected.

> **Internal implementation crate** of
> [`spatiotemporal-tiles`](https://crates.io/crates/spatiotemporal-tiles):
> the analysis *library* (the facade's `optimize` module). The
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
