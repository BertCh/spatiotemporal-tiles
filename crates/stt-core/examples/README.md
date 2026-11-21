# stt-core examples

Run with `cargo run -p stt-core --example <name> -- <args>`.

All examples operate on the **packed format** (`manifest.json` + `index/*.sttd` +
`packs/*.sttp`). The single-file `.stt` container has been removed — datasets are
built and analysed as packed directories.

| Example               | Purpose                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| `packed-stats`        | Size / layout statistics for a packed dataset.                                                      |
| `point_column_stats`  | Per-column byte breakdown for a point dataset.                                                      |
| `pack-cover`          | Backfill the tight `cover_t_min` covering bound on a packed dataset (tile payload bytes untouched). |
| `encoding-experiment` | Measure candidate per-column encodings against the packed baseline.                                 |
| `trajectory-codec`    | Tile-local predictive coding of trajectory geometry + `vertex_time` vs. the stored columns.         |
| `trajectory-lossy`    | Rate–distortion of quantization, TD-TR, greedy linear and Hermite keyframes at measured SED error.  |
| `make-golden-fixture` | Regenerate the cross-impl golden packed fixture.                                                    |
