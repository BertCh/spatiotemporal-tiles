# stt-wasm

> **Status: experimental and repository-only.** See the
> [support policy](https://github.com/BertCh/poopdeck.gl/blob/main/docs/intro/status-and-support.md).

WebAssembly decoder for the **SpatioTemporal Tiles (STT)** packed format:
open a `manifest.json` from bytes, list tiles, decode a tile blob to Arrow
IPC. It wraps `stt-core`'s reader with `wasm-bindgen` and adds no decoding
logic of its own.

> **Not published** (`publish = false`). Build it yourself — see
> [`docs/guides/wasm.md`](../../docs/guides/wasm.md) for the build recipe,
> the JS and Python usage, and what it does not do yet.

## Why

The format has a Rust reader and a TypeScript reader, both written by the
same author. Anything else — a Python notebook, a Go service, a future GDAL
or Martin path — is not going to port the current v6 varint directory, but
every one of them already reads Arrow IPC. This crate is the cheapest bridge
to that audience: same reader, compiled small, bytes in and Arrow out.

## I/O is inverted

There are no file paths in a browser and `PackedReader` is built on
`Mmap::map`, so this crate never does I/O. It reports which object key and
which byte range it needs next; the host — `fetch`, `requests`, `fsspec`, a
local file — hands the bytes back. Range-request policy (auth, retries,
caching, coalescing) stays where the host already solved it, the artifact
stays small, and the decode stays testable without a browser runner.

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo build -p stt-wasm --target wasm32-unknown-unknown --release
# then wasm-bindgen/wasm-pack for the JS glue — see docs/guides/wasm.md
```

On macOS the build needs a C compiler with the WebAssembly LLVM backend
(Apple's `/usr/bin/clang` has none), because zstd is a C library. The guide
has the one-line fix.

## Tests

`cargo test -p stt-wasm` runs the exported API on the host — including a
parity suite that decodes every tile of a freshly written archive through
both this reader and `stt_core::PackedReader` and compares the Arrow record
batches.
