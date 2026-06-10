# stt-core examples

Run with `cargo run -p stt-core --example <name> -- <args>`.

## Packed format (current)

| Example | Purpose |
| --- | --- |
| `pack-transcode` | Transcode a legacy single-file `.stt` into a packed dataset (`manifest.json` + `index/*.sttd` + `packs/*.sttp`). Used by `scripts/transcode-all-packed.sh`. |
| `verify-packed` | Verify a packed dataset decodes byte-for-byte against its single-file source. |

## Legacy / single-file (measurement & fixtures)

These operate on the **single-file** `.stt` container, which is no longer a
deployment target (it survives only as the `--streaming-arrow` intermediate).
They remain useful for measurement and for regenerating committed test fixtures.

| Example | Purpose | Notes |
| --- | --- | --- |
| `simulate_layout` | Measure blob-ordering locality (the request-cost oracle). | Still the canonical packing measurement tool; used by `scripts/optimize-tiles.sh` and `reprocess-run.sh`. |
| `repack` | Rewrite a single-file archive in a new blob order. | Single-file only; packed datasets order at pack-cutting time. Used by `optimize-tiles.sh`. |
| `tile_stats` | Print per-tile statistics for a single-file archive. | Inspection only. |
| `make_sample_fixture` | Generate the committed `sample.stt` test fixture. | Fixture regeneration. |
| `make-golden-fixture` | Generate the cross-impl golden fixture. | Fixture regeneration. |
