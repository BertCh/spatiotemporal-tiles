//! Layer-frame wire constants and the schema-template tables.
//!
//! The bytes both the encoder and the decoder have to agree on, with no
//! dependency on either direction: the frame alignment, the frame escape /
//! section tags / `ref_kind` discriminants, the layer-frame version, the Arrow
//! schema-metadata key names the decoder re-injects, the encode-side
//! [`TemplateCollector`] and decode-side [`TemplateRegistry`], and the
//! [`TileMeta`] section payload.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

/// Alignment (bytes) of each layer's Arrow IPC stream within the frame.
pub(crate) const FRAME_ALIGN: usize = 8;
// ----------------------------------------------------------------------------
// Layer frame — see the module docs.
// ----------------------------------------------------------------------------

/// Leading u16 of a layer frame: the escape that marks a payload as a frame at
/// all. The manifest's `formatVersion` stays the authoritative discriminator: a
/// reader dispatches on it, never on this escape, so a layer frame that
/// disagrees with the manifest is a hard error instead of a best-effort decode.
/// The escape is defense-in-depth against a payload that is not a frame at all.
pub const FRAME_V2_ESCAPE: u16 = 0xFFFF;

/// The layer-frame version as it appears on the wire: the `frame_version` byte
/// two bytes into every frame. The same axis as [`LAYER_FRAME_VERSION`],
/// narrowed to the u8 the header carries, so the two move together.
pub(crate) const FRAME_V2_VERSION: u8 = 2;

/// Layer-frame section tag: full IPC schema prefix for the CORE batch (self-contained
/// mode, `ref_kind == REF_KIND_INLINE`).
pub const SECTION_INLINE_SCHEMA_CORE: u8 = 0x01;
/// Layer-frame section tag: the canonical [`TileMeta`] JSON.
pub const SECTION_TILE_META: u8 = 0x02;
/// Layer-frame section tag: CORE batch IPC tail (dict batches + record batch + EOS).
pub const SECTION_CORE_BATCH: u8 = 0x03;
/// Layer-frame section tag: full IPC schema prefix for the PROPS batch (as 0x01).
pub const SECTION_INLINE_SCHEMA_PROPS: u8 = 0x04;
/// Layer-frame section tag: PROPS batch IPC tail (as 0x03, props schema).
pub const SECTION_PROPS_BATCH: u8 = 0x05;

/// Layer-frame `ref_kind`: the schema rides inline in an `INLINE_SCHEMA_*` section
/// (self-contained blob — no template registry needed to decode).
pub(crate) const REF_KIND_INLINE: u8 = 0;
/// Layer-frame `ref_kind`: the next 16 bytes are the blake3-128 template hash,
/// resolved against the dataset's [`TemplateRegistry`].
pub(crate) const REF_KIND_TEMPLATE_HASH: u8 = 1;
/// Layer-frame `ref_kind_props`: the layer has no property columns — no props
/// schema/template and no `PROPS_BATCH` section.
pub(crate) const REF_KIND_NO_PROPS: u8 = 2;

/// Version of the LAYER-FRAME axis: the wire format of a tile payload (the
/// sectioned, template-referencing frame). The only layer-frame version this
/// codebase emits or reads, and the only value
/// [`EncoderConfig::format_version`](crate::arrow_tile::EncoderConfig::format_version)
/// accepts.
///
/// A different axis from — though numerically equal to —
/// [`crate::pack::PACKED_FORMAT_VERSION`], which versions an archive's MANIFEST
/// schema. The two are pinned to each other at both ends: a writer refuses a
/// frame of one version under a manifest declaring the other, and a reader
/// refuses to decode such a dataset. Bumping either alone therefore requires
/// teaching those checks the new pairing. `FRAME_V2_VERSION` is this same value
/// as it appears in the frame header.
pub const LAYER_FRAME_VERSION: u32 = 2;
/// blake3 content hash truncated to 128 bits — the layer-frame template reference
/// (16 raw bytes in the frame; lowercase hex in `manifest.schemas`).
pub(crate) fn blake3_128(bytes: &[u8]) -> [u8; 16] {
    let hash = blake3::hash(bytes);
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    out
}
/// Thread-safe encode-side sink for the schema templates a packed build
/// produces.
///
/// The encoder [`record`](Self::record)s each layer's stripped schema prefix
/// and embeds the returned hash in the frame; `PackWriter::finalize` snapshots
/// the collector into the manifest's `schemas` table. Content-addressed keys
/// make the result independent of encode parallelism/order — two tiles sharing
/// a schema record the same entry, and the snapshot is sorted by hash — which
/// is what keeps a rebuild of the same input byte-reproducible.
#[derive(Debug, Default)]
pub struct TemplateCollector {
    templates: Mutex<BTreeMap<[u8; 16], Vec<u8>>>,
}

impl TemplateCollector {
    /// New, empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a template's bytes (idempotent), returning its blake3-128 hash —
    /// the 16-byte reference the layer frame carries.
    pub fn record(&self, template: &[u8]) -> [u8; 16] {
        let hash = blake3_128(template);
        self.templates
            .lock()
            .unwrap()
            .entry(hash)
            .or_insert_with(|| template.to_vec());
        hash
    }

    /// Snapshot of every recorded `(hash, template bytes)`, sorted by hash
    /// (deduped by construction) — the byte-reproducible manifest order.
    pub fn snapshot(&self) -> Vec<([u8; 16], Vec<u8>)> {
        self.templates
            .lock()
            .unwrap()
            .iter()
            .map(|(h, b)| (*h, b.clone()))
            .collect()
    }

    /// Number of distinct templates recorded so far.
    pub fn len(&self) -> usize {
        self.templates.lock().unwrap().len()
    }

    /// Whether no template has been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Decode-side template lookup: blake3-128 hash → raw template bytes.
///
/// Built once per dataset from `manifest.schemas` (each entry hash-validated
/// at open) and threaded into
/// [`decode_tile_with_templates`](crate::arrow_tile::decode_tile_with_templates).
#[derive(Debug, Default, Clone)]
pub struct TemplateRegistry {
    templates: HashMap<[u8; 16], Arc<Vec<u8>>>,
}

impl TemplateRegistry {
    /// New, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a template, returning its blake3-128 hash.
    pub fn insert(&mut self, template: Vec<u8>) -> [u8; 16] {
        let hash = blake3_128(&template);
        self.templates.insert(hash, Arc::new(template));
        hash
    }

    /// Look a template up by its 16-byte hash.
    pub fn get(&self, hash: &[u8; 16]) -> Option<&[u8]> {
        self.templates.get(hash).map(|t| t.as_slice())
    }

    /// Iterate every `(hash, template bytes)` pair (arbitrary order). Lets a
    /// verbatim-repack tool seed a [`TemplateCollector`] from a source
    /// dataset's registry ([`crate::pack::PackWriter::with_seeded_templates`]).
    pub fn iter(&self) -> impl Iterator<Item = (&[u8; 16], &[u8])> {
        self.templates.iter().map(|(h, t)| (h, t.as_slice()))
    }

    /// Number of templates registered.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}
/// `TILE_META.st` — how the layer's `start_time` column is encoded.
///
/// ABSENT means the column is the historical absolute non-null `Int64`. A
/// reader MUST branch on this key, never on the Arrow `DataType` alone: that
/// is the format's established convention for a re-typed column (see
/// `stt:quant` / `stt:qa`), and it is what lets the encoder pick per layer.
/// An unknown VALUE is a hard decode error rather than a silent misread — the
/// `time-delta` manifest capability is the version gate that keeps a reader
/// from ever seeing one it does not know.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StartTimeForm {
    /// Non-null `UInt32` offsets from `t0`: `absolute = t0 + offset`. `t0` is
    /// therefore REQUIRED (load-bearing, not an optimization) whenever this
    /// value is present.
    #[serde(rename = "u32")]
    U32Offset,
}

/// `TILE_META.et` — how the layer's `end_time` column is encoded.
///
/// ABSENT means the column is the historical absolute non-null `Int64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndTimeForm {
    /// Non-null `UInt32` DURATIONS against each feature's own start:
    /// `absolute_end = start_time + dur`.
    #[serde(rename = "dur32")]
    Dur32,
    /// The `end_time` column is OMITTED from the CORE batch entirely —
    /// `end == start` for every feature in the layer. The reader synthesizes
    /// it back from `start_time`.
    #[serde(rename = "zero")]
    Zero,
}

/// The per-tile-varying metadata a layer frame carries in its `TILE_META`
/// section instead of the (dataset-constant, template-resident) Arrow schema
/// metadata. Canonical serialization: JSON, keys sorted
/// — field order below IS alphabetical and the `qa` map is a `BTreeMap` — no
/// whitespace. Readers MUST ignore unknown keys (serde's default here), so
/// the section can evolve additively. Presence rules: a key is present iff
/// the corresponding feature is (`qa` omits non-quantized columns entirely;
/// `t0` iff a start-time column exists; `vt` iff delta-encoded vertex_time;
/// `vb` iff a value matrix; `st`/`et` iff the time columns are compact;
/// `vq` iff a per-vertex value column ships quantized).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TileMeta {
    /// Compact `end_time` encoding (see [`EndTimeForm`]); absent ⇒ an
    /// absolute `Int64` `end_time` column is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub et: Option<EndTimeForm>,
    /// Per-property attribute-quantization affines, `column → [o, s]`
    /// (`value = o + q*s`) — the `stt:qa` Arrow field metadata, hoisted out
    /// of the schema so the schema itself stays dataset-constant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qa: Option<BTreeMap<String, (f64, f64)>>,
    /// Rows are stable-sorted by `start_time` (always `true` from this
    /// writer; carried as a flag so a writer could demote the sort without
    /// bumping the layer-frame version).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sorted: Option<bool>,
    /// Compact `start_time` encoding (see [`StartTimeForm`]); absent ⇒ an
    /// absolute `Int64` `start_time` column.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub st: Option<StartTimeForm>,
    /// The `stt:time_offset_ms` Arrow schema-metadata key: minimum feature
    /// start-time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t0: Option<i64>,
    /// The `stt:vertex_value_buckets` Arrow schema-metadata key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vb: Option<u32>,
    /// Per-vertex-value quantization affines, `column → [o, s]`
    /// (`value = o + q*s`) — the `vertex-value-quant` capability. Keys are a
    /// subset of `{"vertex_value", "vertex_value_matrix"}`: whichever of the
    /// two ships its leaf as `UInt16` instead of `Float32` on this tile. The
    /// reserved index [`VERTEX_VALUE_QUANT_SENTINEL`](crate::arrow_tile::VERTEX_VALUE_QUANT_SENTINEL)
    /// decodes to `NaN` (the format's "this vertex has no value" marker).
    /// A `BTreeMap` so the canonical JSON's inner keys are sorted too.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vq: Option<BTreeMap<String, (f64, f64)>>,
    /// The `stt:vertex_time_origin_ms` / `stt:vertex_time_step_ms`
    /// schema-metadata pair,
    /// as `[origin_ms, step_ms]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vt: Option<(i64, u32)>,
}
/// Arrow schema-metadata keys for the DELTA per-vertex time encoding: the
/// absolute origin (Unix ms) the deltas are measured from, and the ms step each
/// delta counts. Both are present exactly when `vertex_time` ships as
/// `List<UInt16>` deltas; without them those deltas cannot be resolved back to
/// absolute times.
pub(crate) const VERTEX_TIME_ORIGIN_KEY: &str = "stt:vertex_time_origin_ms";
pub(crate) const VERTEX_TIME_STEP_KEY: &str = "stt:vertex_time_step_ms";
/// Number of time buckets packed into each row of the `vertex_value_matrix`
/// column. The renderer reshapes the flat vertex-major list back into a
/// `[vertex][bucket]` grid using this count.
pub(crate) const VERTEX_VALUE_BUCKETS_KEY: &str = "stt:vertex_value_buckets";
/// Baked minimum feature start-time (integer Unix ms) for the layer. The TS
/// decoder relativizes every start/end time against this value so the times
/// fit an f32; baking it here lets the decoder skip its client-side min-scan
/// over the whole start-time column. Mirrors exactly what the decoder computes
/// (the min of the `start_time` column) — see `packages/core/src/tile.ts`.
pub(crate) const TIME_OFFSET_MS_KEY: &str = "stt:time_offset_ms";
