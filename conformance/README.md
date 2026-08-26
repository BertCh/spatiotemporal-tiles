# Conformance vectors

Six small packed archives, produced by the reference **writer** (`stt-build`),
that a candidate **reader** must be able to open. They are the portable half of
[`docs/spec/conformance.md`](../docs/spec/conformance.md): the spec says what
conformance means, and these say what it looks like in bytes.

| Vector                 | What it exercises                                                                                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v2-golden/`           | Points; numeric column with nulls, two dictionary columns with nulls, 1 m coord quantization, per-tile `qa` attribute affines, 2 zooms × 2 time buckets, **paged** directory |
| `v2-golden-tracks/`    | LineString trajectories with duration: interpolated u16-delta `vertex_time`, TILE_META `vt`, Float64 coords, **single** directory                                            |
| `packed-golden/`       | The plain packed shape                                                                                                                                                       |
| `paged-golden/`        | Paged directory                                                                                                                                                              |
| `paged-golden-single/` | Paged directory holding a single page                                                                                                                                        |
| `legacy-shape/`        | A published shape a conformant reader still has to open read-only                                                                                                            |

## Using them

Point your reader at each `manifest.json` and open every tile the directory
lists. A conformant reader opens all six without special-casing any of them —
both directory layouts, every `vertex_time` width, both `TILE_META` time forms,
quantized and raw per-vertex value columns, and unknown additive columns and
fields (§1 of the conformance page).

The vectors are **byte-pinned**. They change only inside a declared rebuild
window, flagged with a `Rebuild-Window: R1` commit trailer and enforced by
`.github/scripts/check-golden-pins.mjs` — because a golden test going red has
exactly two causes that want opposite responses, and re-blessing the fixture
destroys the evidence that tells them apart.

## Regenerating

```bash
conformance/make-vectors.sh
```

Needs `cargo` (it builds `spatiotemporal-tiles --features duckdb`) and, on the
first run, network for DuckDB's `INSTALL spatial`. Builds are byte-reproducible,
so a re-run is a no-op diff unless the writer's bytes intentionally changed. The
source data is synthetic and lives in the script itself as SQL `VALUES` — no
input files, no downloads.

## Who consumes them

- `crates/stt-core` has its own **writer**-side pin at
  `crates/stt-core/tests/fixtures/v2-golden/`, checked by
  `crates/stt-core/tests/v2_golden.rs`. That one asks "does the writer still
  emit these bytes?"; this tree asks "can a reader still open them?"
- `@poopdeck.gl/core` in the [poopdeck.gl repository][pd] vendors this tree
  verbatim to `packages/core/test/fixtures/` and byte-gates it with
  `pnpm stt:check`. Re-blessing it there is incoherent — the copy would then
  disagree with the writer that defines it.
- Anyone else: copy the directory. It is MIT-licensed like the rest of the
  repository, and it is deliberately small (~570 KB) so vendoring is reasonable.

[pd]: https://github.com/BertCh/poopdeck.gl
