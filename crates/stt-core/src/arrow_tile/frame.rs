//! Layer-frame wire constants and the v2 schema-template tables.
//!
//! The bytes both the encoder and the decoder have to agree on, with no
//! dependency on either direction: the v1 aligned-frame flag, the v2 escape /
//! section tags / `ref_kind` discriminants, the `formatVersion` values, the
//! v1 schema-metadata key names, the encode-side [`TemplateCollector`] and
//! decode-side [`TemplateRegistry`], and the v2 [`TileMeta`] section payload.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

/// Alignment (bytes) of each layer's Arrow IPC stream within the frame.
pub(crate) const FRAME_ALIGN: usize = 8;
// ----------------------------------------------------------------------------
// Layer frame — see the module docs.
// ----------------------------------------------------------------------------

/// Leading u16 of a layer frame: the escape that marks a payload as a frame at
/// all. Manifest `formatVersion` remains the authoritative discriminator
/// (design ★F6); this escape is defense-in-depth against a non-frame payload.
pub const FRAME_V2_ESCAPE: u16 = 0xFFFF;

/// `frame_version` byte of the v2 frame.
pub(crate) const FRAME_V2_VERSION: u8 = 2;

/// v2 section tag: full IPC schema prefix for the CORE batch (self-contained
/// mode, `ref_kind == REF_KIND_INLINE`).
pub const SECTION_INLINE_SCHEMA_CORE: u8 = 0x01;
/// v2 section tag: the canonical [`TileMeta`] JSON.
pub const SECTION_TILE_META: u8 = 0x02;
/// v2 section tag: CORE batch IPC tail (dict batches + record batch + EOS).
pub const SECTION_CORE_BATCH: u8 = 0x03;
/// v2 section tag: full IPC schema prefix for the PROPS batch (as 0x01).
pub const SECTION_INLINE_SCHEMA_PROPS: u8 = 0x04;
/// v2 section tag: PROPS batch IPC tail (as 0x03, props schema).
pub const SECTION_PROPS_BATCH: u8 = 0x05;

/// v2 `ref_kind`: the schema rides inline in an `INLINE_SCHEMA_*` section
/// (self-contained blob — no template registry needed to decode).
pub(crate) const REF_KIND_INLINE: u8 = 0;
/// v2 `ref_kind`: the next 16 bytes are the blake3-128 template hash,
/// resolved against the dataset's [`TemplateRegistry`].
pub(crate) const REF_KIND_TEMPLATE_HASH: u8 = 1;
/// v2 `ref_kind_props`: the layer has no property columns — no props
/// schema/template and no `PROPS_BATCH` section.
pub(crate) const REF_KIND_NO_PROPS: u8 = 2;

/// The layer-frame format: the sectioned, template-referencing frame. The only
/// version this codebase emits or reads — the transitional v1 (0.3.x) frame was
/// removed once the published fleet was migrated.
pub const FORMAT_VERSION: u32 = 2;
/// blake3 content hash truncated to 128 bits — the v2 template reference
/// (16 raw bytes in the frame; lowercase hex in `manifest.schemas`).
pub(crate) fn blake3_128(bytes: &[u8]) -> [u8; 16] {
    let hash = blake3::hash(bytes);
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    out
}
/// Thread-safe encode-side sink for the schema templates a v2 build produces.
///
/// The encoder [`record`](Self::record)s each layer's stripped schema prefix
/// and embeds the returned hash in the frame; `PackWriter::finalize` snapshots
/// the collector into the manifest's `schemas` table. Content-addressed keys
/// make the result independent of encode parallelism/order (design ★F1/E1):
/// two tiles sharing a schema record the same entry, and the snapshot is
/// sorted by hash.
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
    /// the 16-byte reference the v2 frame carries.
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
/// The per-tile-varying metadata a v2 frame carries in its `TILE_META`
/// section instead of the (now dataset-constant, template-resident) Arrow
/// schema metadata. Canonical serialization (design §4.3): JSON, keys sorted
/// — field order below IS alphabetical and the `qa` map is a `BTreeMap` — no
/// whitespace. Readers MUST ignore unknown keys (serde's default here), so
/// the section can evolve additively. Presence rules: a key is present iff
/// the corresponding feature is (`qa` omits non-quantized columns entirely;
/// `t0` iff a start-time column exists; `vt` iff delta-encoded vertex_time;
/// `vb` iff a value matrix).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TileMeta {
    /// Per-property attribute-quantization affines, `column → [o, s]`
    /// (`value = o + q*s`) — the v1 `stt:qa` field metadata, hoisted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qa: Option<BTreeMap<String, (f64, f64)>>,
    /// Rows are stable-sorted by `start_time` (always `true` from this
    /// writer; a flag so Stage III can demote the sort without a frame bump).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sorted: Option<bool>,
    /// The v1 `stt:time_offset_ms` schema key: minimum feature start-time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t0: Option<i64>,
    /// The v1 `stt:vertex_value_buckets` schema key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vb: Option<u32>,
    /// The v1 `stt:vertex_time_origin_ms` / `stt:vertex_time_step_ms` pair,
    /// as `[origin_ms, step_ms]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vt: Option<(i64, u32)>,
}
/// Schema metadata keys for the v3 per-vertex time encoding.
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
