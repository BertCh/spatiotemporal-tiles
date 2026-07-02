# stt-validate

Validator for STT datasets — the CI gate for anything that ships. Pass a
packed dataset directory (or its `manifest.json`), or a legacy single-file
`.stt` archive. It verifies that every pack and the directory object
blake3-hash to the names the manifest gave them, that the index decodes with
the columns the schema promises, that every tile blob round-trips its
content hash and decompresses to its declared size, that every payload
decodes as Arrow IPC layer frames, that feature counts match the decoded
rows, and that tile temporal extents lie inside the dataset's time range.
Exits non-zero on any failure.

> **Not yet published to crates.io** — build from the repo:
>
> ```bash
> git clone https://github.com/BertCh/spatiotemporal-tiles
> cd spatiotemporal-tiles
> cargo install --path crates/stt-validate
> ```

## Example

```bash
# Full validation of a packed dataset:
stt-validate my-dataset/

# CI: machine-readable, stop at the first failure:
stt-validate my-dataset/manifest.json --json --fail-fast

# Very large archive: integrity checks over ALL tiles, expensive
# Arrow-decode over a deterministic 500-tile sample:
stt-validate my-dataset/ --sample 500

# Header/integrity/index only:
stt-validate my-dataset/ --skip-decode
```

## Relation to the other crates

Decodes with the same [`stt-core`](../stt-core) reader the renderers mirror,
so a green run means the archive is readable end to end. Point it at the
output of [`stt-build`](../stt-build) or [`stt-generate`](../stt-generate);
the checks encode the invariants of the packed format spec.

## Docs

- [CLI reference](../../docs/api/cli-reference.md#stt-validate)
- [Packed format spec](../../docs/spec/stt-packed-format.md)
- [Conformance suite](../../docs/spec/conformance.md)

License: MIT.
