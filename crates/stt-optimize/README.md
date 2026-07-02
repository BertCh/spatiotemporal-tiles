# stt-optimize

Analyzer and flag-recommender for STT builds. It inspects an input's
spatial extent, temporal distribution, geometry mix, and density, then
recommends `stt-build` settings — zoom range, temporal bucket — so you
don't hand-tune them per dataset. The same logic is what runs inside
`stt-build --auto`; the CLI exists to run the analysis standalone, inspect
the reasoning (`--verbose`), or emit machine-readable reports.

> **Not yet published to crates.io** — build from the repo:
>
> ```bash
> git clone https://github.com/BertCh/spatiotemporal-tiles
> cd spatiotemporal-tiles
> cargo install --path crates/stt-optimize
> ```

## Example

```bash
# Analyze an input (also accepts an existing single-file archive via --stt):
stt-optimize analyze --input data.parquet --time-field timestamp \
  --time-format unix-ms

# Print a copy-pasteable stt-build invocation:
stt-optimize recommend --input data.parquet --time-field timestamp \
  --time-format unix-ms --show-command

# Machine-readable, for pipelines:
stt-optimize analyze --input data.parquet --time-field timestamp \
  --time-format unix-ms --format json -o report.json
```

## Relation to the other crates

Reads inputs via [`stt-core`](../stt-core); [`stt-build`](../stt-build)
calls the library entry point (`recommend_for`) when invoked with `--auto`,
applying the zoom-range and temporal-bucket recommendations (compression is
not applied — the packed format is zstd-only).

## Docs

- [CLI reference](../../docs/api/cli-reference.md#stt-optimize)
- [`stt-build` flag reference](../../docs/api/cli-reference.md#stt-build)
- [Packed format spec](../../docs/spec/stt-packed-format.md)

License: MIT.
