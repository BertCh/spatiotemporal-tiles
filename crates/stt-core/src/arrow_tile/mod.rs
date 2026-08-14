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
//! | `vertex_value`| `List<Float32>`, or `List<UInt16>` under a `TILE_META.vq` affine (nullable) | per-vertex scalar, e.g. SST (optional) |
//! | `vertex_value_matrix` | as `vertex_value`               | per-vertex × per-bucket series (optional) |
//! | `triangles`   | `List<UInt16>` or `List<UInt32>`        | pre-baked earcut indices, feature-local (optional; polygon only) |
//! | `part_offsets`| `List<UInt32>`                          | per-feature MultiPolygon part boundaries as ring indices (optional; polygon only — absent ⇒ every feature is single-part) |
//! | `<property>`  | `Float64`, `Utf8`, or `Dictionary<UInt16,Utf8>` | one column per property; categorical representation is chosen per tile |
//!
//! All layers in one tile are concatenated with a tiny frame so a tile can
//! carry, say, a linestring layer and a point layer side by side.
//!
//! The layer frame ([`LAYER_FRAME_VERSION`] — the payload-side version axis,
//! pinned to the packed format's `manifest.formatVersion` but counted
//! separately; see `docs/spec/stt-packed-format.md` §9) is a
//! sectioned frame that hoists each layer's Arrow IPC *schema message* into a
//! per-dataset **template** (referenced by blake3-128 hash, resolved through
//! the manifest's `schemas` table) so the per-tile schema tax disappears, and
//! carries the per-tile-varying metadata in a compact [`TILE_META`
//! section](TileMeta) instead of the schema:
//!
//! ```text
//! u16  0xFFFF                    # layer-frame escape (see FRAME_V2_ESCAPE)
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
//! Reserved columns (`id`/times/geometry/vertex_*/`triangles`/`part_offsets`)
//! form the CORE
//! batch; property columns form the PROPS batch with its own schema/template
//! (emitted only when properties exist). Each `*_BATCH` section is the IPC
//! stream's **tail** — dictionary batch(es) + record batch + end-of-stream —
//! and a reader materialises `concat(template, tail)` for a stock Arrow
//! reader. Unknown section tags are skippable via the TOC. Rows are
//! stable-sorted by `start_time` at encode (after id assignment), declared by
//! `TILE_META.sorted`. The leading `0xFFFF` escape identifies a payload as a
//! frame at all, so a truncated read or an error body is rejected rather than
//! misparsed.
//!
//! ## Module layout
//!
//! Concerns live in private submodules and every item is re-exported here, so
//! `stt_core::arrow_tile::X` is the single public path whichever submodule
//! defines `X`:
//!
//! - `frame` — wire constants both directions agree on, the schema
//!   template tables ([`TemplateCollector`] / [`TemplateRegistry`]) and
//!   [`TileMeta`].
//! - `layer` — the in-memory model ([`ColumnarLayer`] and its columns).
//! - `quantize` — coordinate ([`QuantAffine`]) and attribute ([`AttrQuant`])
//!   fixed-point encoding.
//! - `columns` — Arrow array construction, shared by both frame versions.
//! - `config` — [`EncoderConfig`] and the surviving encoder globals.
//! - `encode` — layer parts, the standalone single-layer IPC stream, and
//!   layer-frame assembly.
//! - `decode` — the standalone single-layer IPC decode and the layer-frame
//!   walk.

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
