//! Arrow-based tile payload format.
//!
//! A tile's payload is one or more *layers*. Each layer is a single Arrow
//! [`RecordBatch`](arrow::array::RecordBatch) serialised as an Arrow IPC
//! **stream**. Geometry is encoded
//! using the GeoArrow interleaved-coordinate convention so the payload can be
//! consumed directly by GeoArrow-aware renderers (e.g. `@geoarrow/deck.gl`).
//!
//! ## Per-layer schema
//!
//! | column        | type                                    | notes                         |
//! |---------------|-----------------------------------------|-------------------------------|
//! | `id`          | `UInt64`                                | feature id                    |
//! | `start_time`  | `Int64`                                 | Unix ms, absolute             |
//! | `end_time`    | `Int64`                                 | Unix ms, absolute             |
//! | `geometry`    | GeoArrow point / linestring / polygon   | interleaved f64 lon/lat       |
//! | `vertex_time` | `List<UInt16>` deltas or `List<Int64>` (nullable) | per-vertex Unix ms (optional; see [`build_vertex_time_array`]) |
//! | `vertex_value`| `List<Float32>` (nullable)              | per-vertex scalar, e.g. SST (optional) |
//! | `triangles`   | `List<UInt16>` or `List<UInt32>`        | pre-baked earcut indices, feature-local (optional; polygon only) |
//! | `<property>`  | `Float64` or `Dictionary<UInt16,Utf8>`   | one column per property       |
//!
//! All layers in one tile are concatenated with a tiny frame so a tile can
//! carry, say, a linestring layer and a point layer side by side. Two frame
//! shapes exist, selected by [`EncoderConfig::format_version`] (the payload
//! side of the packed format's `manifest.formatVersion` — the two are bumped
//! in lockstep, see `docs/spec/stt-packed-format.md` §9):
//!
//! **v1** (`format_version: 1`, the 0.3.x wire shape — frozen, byte-identical):
//!
//! ```text
//! [u16 layer_count | ALIGNED_FRAME_FLAG]
//!   repeated: [u16 name_len][name utf8][u32 ipc_len][pad to 8][ipc stream bytes]
//! ```
//!
//! The leading u16's top bit ([`ALIGNED_FRAME_FLAG`]) marks the *aligned*
//! frame: zero padding after each `ipc_len` places every Arrow IPC stream at
//! an 8-byte boundary relative to the payload start, so readers can hand the
//! stream to an Arrow implementation zero-copy (Arrow buffers are 8-byte
//! aligned *within* a stream; the stream itself must start aligned for that
//! to survive). The pad length is not stored — readers derive it as
//! `(8 - pos % 8) % 8` from the position after `ipc_len`. Frames without the
//! flag (every archive written before the flag existed) carry no padding and
//! decode exactly as before.
//!
//! **v2** (`format_version: 2`, default for new `stt-build` output): a
//! sectioned frame that hoists each layer's Arrow IPC *schema message* into a
//! per-dataset **template** (referenced by blake3-128 hash, resolved through
//! the manifest's `schemas` table) so the per-tile schema tax disappears, and
//! carries the per-tile-varying metadata in a compact [`TILE_META`
//! section](TileMeta) instead of the schema:
//!
//! ```text
//! u16  0xFFFF                    # v2 escape (see FRAME_V2_ESCAPE)
//! u8   frame_version = 2
//! u8   flags = 0                 # reserved, MUST be 0
//! u16  layer_count
//! per layer:
//!   u16  name_len, [name utf8]
//!   u8   ref_kind_core           # 0 = INLINE_SCHEMA_CORE section present;
//!                                # 1 = the next 16 bytes are the template hash
//!   [16] core template hash     # iff ref_kind_core == 1
//!   u8   ref_kind_props          # 0/1 as above; 2 = NO props sections at all
//!   [16] props template hash    # iff ref_kind_props == 1
//!   u8   section_count
//!   per section (TOC): u8 tag, u32 length     # at-rest bytes, pad excluded
//!   [pad to 8, derived]
//!   per section: [section bytes][pad to 8, derived]
//! ```
//!
//! Reserved columns (`id`/times/geometry/vertex_*/`triangles`) form the CORE
//! batch; property columns form the PROPS batch with its own schema/template
//! (emitted only when properties exist). Each `*_BATCH` section is the IPC
//! stream's **tail** — dictionary batch(es) + record batch + end-of-stream —
//! and a reader materialises `concat(template, tail)` for a stock Arrow
//! reader. Unknown section tags are skippable via the TOC. Rows are
//! stable-sorted by `start_time` at encode (after id assignment), declared by
//! `TILE_META.sorted`. The v2 escape is unreachable from the v1 writer: the
//! v1 path caps `layer_count` below `0x7fff`, so an aligned v1 frame can
//! never start with `0xFFFF`.
//!
//! ## Module layout
//!
//! This was one 4.8k-line file; it is now split by concern into private
//! submodules, with every item re-exported here — `stt_core::arrow_tile::X`
//! resolves exactly as it always did:
//!
//! - `frame` — wire constants both directions agree on, the v2 schema
//!   template tables ([`TemplateCollector`] / [`TemplateRegistry`]) and
//!   [`TileMeta`].
//! - `layer` — the in-memory model ([`ColumnarLayer`] and its columns).
//! - `quantize` — coordinate ([`QuantAffine`]) and attribute ([`AttrQuant`])
//!   fixed-point encoding.
//! - `columns` — Arrow array construction, shared by both frame versions.
//! - `config` — [`EncoderConfig`] and the surviving encoder globals.
//! - `encode` — layer parts + v1/v2 frame assembly.
//! - `decode` — the v1/v2 frame walks.

mod columns;
mod config;
mod decode;
mod encode;
mod frame;
mod layer;
mod quantize;

pub use self::columns::*;
pub use self::config::*;
pub use self::decode::*;
pub use self::encode::*;
pub use self::frame::*;
pub use self::layer::*;
pub use self::quantize::*;

#[cfg(test)]
mod tests;
