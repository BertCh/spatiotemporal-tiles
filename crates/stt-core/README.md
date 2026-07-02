# stt-core

Core Rust library for the **SpatioTemporal Tiles (STT)** format. The
canonical container is the **packed format**: a per-dataset `manifest.json`
(tiny, mutable) plus content-addressed, immutable pack objects
(`packs/<blake3>.sttp`) and one directory object (`index/<blake3>.sttd`) —
small immutable objects are what make a dataset edge-cacheable on a plain
CDN. This crate owns the format end to end: `PackWriter` / `PackedReader`
and the `Manifest`, Arrow-IPC tile payloads with GeoArrow geometry
(`arrow_tile`), the run-length + paged tile directory, Hilbert/temporal blob
ordering (`curve`), zstd/gzip compression, and timestamp normalization.

> **Not yet published to crates.io** — depend on it by path from the
> workspace, or via a git dependency:
>
> ```toml
> [dependencies]
> stt-core = { git = "https://github.com/BertCh/spatiotemporal-tiles" }
> ```

## Example

`stt-core` is a library (the CLIs live in the sibling crates), but it ships
operational one-offs as cargo examples:

```bash
# Losslessly re-pack a dataset, backfilling the tight per-tile temporal
# bound (payload bytes untouched):
cargo run --release -p stt-core --example pack-cover -- \
  my-dataset/manifest.json my-dataset-v2 64 auto

# Size/layout stats for a packed dataset:
cargo run --release -p stt-core --example packed-stats -- my-dataset/manifest.json
```

See [`examples/README.md`](./examples/README.md) for the full list
(`repack`, `packed-stats`, `pack-transcode`, `simulate_layout`, …).

## Relation to the other crates

Every other crate builds on this one: [`stt-build`](../stt-build) writes
packed datasets through `PackWriter`, [`stt-serve`](../stt-serve) encodes
single tiles with the same encoder, [`stt-validate`](../stt-validate) checks
archives with `verify_packed_objects` + the tile decoder, and
[`stt-optimize`](../stt-optimize) reads archives for analysis. The
TypeScript reader (`@poopdeck.gl/core`) is the browser-side mirror.

## Docs

- [Packed format spec](../../docs/spec/stt-packed-format.md)
- [Time model](../../docs/spec/time-model.md)
- [Conformance suite](../../docs/spec/conformance.md)
- [Maintenance tools](../../docs/api/cli-reference.md#maintenance-tools-cargo-examples)

License: MIT.
