# Path codec — wire design (2026-11)

**Status:** landed on branch `trajectory`; normative text:
[docs/spec/stt-packed-format.md §5.2.7](../spec/stt-packed-format.md#527-path-codec-section-path_codec--capability-path-codec).
The evidence is in [trajectory-codec-2026-11.md](./trajectory-codec-2026-11.md).

The opt-in path codec is specified in packed-format §5.2.7. This record keeps only the
rationale for the wire choices.

## 8. Why these choices

Measured on real tile payloads (`trajectory-frontier`, AIS, 1/16 sample):

| choice                                          | against the alternative                        |
| ----------------------------------------------- | ---------------------------------------------- |
| side section, not Arrow `Binary` columns        | −4.8 % (per-feature `i32` offsets)             |
| per-feature predictor tag, not one per layer    | −1.6 %                                         |
| first vertex relative to the previous feature's | −1.9 %                                         |
| three predictors, not delta only                | −5.2 %                                         |
| quotient/remainder times, not ms deltas         | −20 % on the time stream, and exact            |
| empty list columns, not dropped columns         | no template fork; decoder substitutes in place |

A shared zstd dictionary would add a further −8 % (−15 % at the deepest zoom),
but the TypeScript reader's zstd decoder cannot take one. It is recorded in the
investigation record as a separate lever, not part of this codec.
