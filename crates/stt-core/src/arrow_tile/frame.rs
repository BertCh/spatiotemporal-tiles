//! Layer-frame wire constants and the schema-template tables.
//!
//! The bytes both the encoder and the decoder have to agree on, with no
//! dependency on either direction: the frame alignment, the frame escape /
//! section tags / `ref_kind` discriminants, the layer-frame version, the Arrow
//! schema-metadata key names the decoder re-injects, the encode-side
//! [`TemplateCollector`] and decode-side [`TemplateRegistry`], and the
//! [`TileMeta`] section payload.

use arrow::datatypes::DataType;
use arrow::ipc::reader::StreamReader;
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

    /// The |T| ACCOUNTING report: one row per distinct schema SHAPE, as
    /// `(shape description, templates carrying it, total template bytes)`,
    /// sorted by description.
    ///
    /// This is the instrument defect D6 lacked. Every per-tile encoding choice
    /// — the dictionary-vs-`Utf8` verdict, the `UInt16`/`Int32` leaf width, the
    /// `EndTimeForm::Zero` omission, whether a sidecar column is present — forks
    /// a schema template when it differs from tile to tile, and until now no
    /// encoder could see the fork it caused, let alone price it. The report is
    /// what makes `|T|` observable at the end of a build, and it is the evidence
    /// [`template_cost_justifies`] is meant to be calibrated against.
    ///
    /// The description is a canonicalization of the template's Arrow schema
    /// (sorted schema metadata + the field list as `name:type`), NOT of its
    /// bytes: a `count` above 1 therefore means two templates that describe the
    /// same shape still hash apart — a byte-level fork the shape vocabulary
    /// cannot explain (the one known candidate is Arrow's `HashMap`-backed
    /// schema metadata, whose iteration order reaches the flatbuffer, and which
    /// the encoder's byte-reproducibility tests track separately). Surfacing
    /// that is the point; `count == 1` everywhere is the healthy state.
    ///
    /// Deterministic: the underlying snapshot is hash-sorted and deduped, the
    /// grouping key is a pure function of the template bytes, and the rows come
    /// out in `BTreeMap` (description) order.
    pub fn template_report(&self) -> Vec<(String, usize, u64)> {
        let mut by_shape: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for (_, bytes) in self.snapshot() {
            let entry = by_shape
                .entry(describe_template_shape(&bytes))
                .or_insert((0, 0));
            entry.0 += 1;
            entry.1 += bytes.len() as u64;
        }
        by_shape
            .into_iter()
            .map(|(shape, (count, bytes))| (shape, count, bytes))
            .collect()
    }

    /// Total bytes of every distinct template recorded — the `Σ|τ|` term of the
    /// §3.8 objective, as opposed to the per-tile tail term.
    pub fn total_template_bytes(&self) -> u64 {
        self.templates
            .lock()
            .unwrap()
            .values()
            .map(|t| t.len() as u64)
            .sum()
    }

    /// Measured marginal cost of one more template: the mean recorded template
    /// size, or `None` when nothing has been recorded yet.
    ///
    /// This is the "template size read from a trial [`TemplateCollector`]
    /// snapshot" a width/type decision should price itself against — pass it to
    /// [`template_cost_justifies_with`] in preference to the
    /// [`DEFAULT_TEMPLATE_COST_BYTES`] fallback whenever a real collector is in
    /// hand, because a dataset with wide property schemas pays far more per fork
    /// than the fixture-derived default assumes.
    pub fn mean_template_bytes(&self) -> Option<u64> {
        // One lock for both the count and the sum: a collector is still being
        // written to during a parallel encode, and a mean assembled from two
        // different instants of it would be a number nothing produced.
        let templates = self.templates.lock().unwrap();
        let n = templates.len() as u64;
        let total: u64 = templates.values().map(|t| t.len() as u64).sum();
        (n > 0).then(|| total / n)
    }

    /// Cross-check the REALIZED template count against pass 1's PREDICTED shape
    /// count, returning a warning line when the build forked more shapes than
    /// were predicted (and `None` when it did not).
    ///
    /// The assertion behind mechanism M2: once the dataset-global pins decide
    /// each column's Arrow type and leaf width, a per-tile encoder must not be
    /// able to invent a shape the pass-1 scan did not foresee. An overshoot is
    /// not a hard error — an unpredicted fork costs bytes, never correctness,
    /// and failing a long build over it would be the wrong trade — so this
    /// hands back a message for the caller to log loudly.
    ///
    /// Realized BELOW the prediction is fine and silent: a dataset whose tiles
    /// never realize some predicted shape (an empty layer kind, say) simply
    /// costs less than forecast.
    pub fn unpredicted_fork_warning(&self, predicted: usize) -> Option<String> {
        // Counted off the SAME snapshot the detail lines come from, so the
        // headline number and the shapes below it can never disagree.
        let report = self.template_report();
        let realized: usize = report.iter().map(|(_, count, _)| *count).sum();
        if realized <= predicted {
            return None;
        }
        let forks: Vec<String> = report
            .iter()
            .filter(|(_, count, _)| *count > 1)
            .map(|(shape, count, bytes)| format!("{count}× {shape} ({bytes} B)"))
            .collect();
        let detail = if forks.is_empty() {
            format!("{} distinct shapes, none duplicated", report.len())
        } else {
            format!("duplicated shapes: {}", forks.join("; "))
        };
        Some(format!(
            "schema templates: {realized} realized vs {predicted} predicted by pass 1 \
             (+{} unpredicted fork(s)); {detail}",
            realized - predicted
        ))
    }
}

// ----------------------------------------------------------------------------
// The schema-template cost term |T| — mechanism M2 / defect D6.
//
// §3.8's objective is `Σ_τ |τ| + Σ_t (|TILEMETA_t| + 16·refs + |tail_t|)`: every
// distinct realized shape costs a whole template, and that second-order cost is
// what every per-tile encoding choice externalizes. The partition of keys
// between TEMPLATE and TILE_META is NOT in question here — it is exactly optimal
// given the known variability structure and this file leaves it alone. What is
// added is the price tag a choice can consult before it forks a shape.
// ----------------------------------------------------------------------------

/// Default marginal cost of one extra schema template, in bytes — the `c_tmpl`
/// constant of the §3.8 objective when no measured collector is available.
///
/// Measured over the v2 golden fixture's `manifest.schemas`: five templates of
/// 128, 192, 720, 728 and 888 raw bytes, mean 531. A deliberate UNDER-estimate
/// of the true archive cost, on two counts — `manifest.schemas` stores each
/// template base64'd (×4/3), and a manifest is fetched by every client on open
/// while a per-row saving is paid once — because under-pricing templates biases
/// [`template_cost_justifies`] toward the INCUMBENT encoding. A cost term that
/// only ever fires when the win is unambiguous is the right default for a term
/// whose whole job is to stop unpriced churn.
///
/// Callers holding a real [`TemplateCollector`] should prefer its measured
/// [`mean_template_bytes`](TemplateCollector::mean_template_bytes).
pub const DEFAULT_TEMPLATE_COST_BYTES: u64 = 531;

/// Is an encoding choice that adds `shape_delta` schema templates justified by
/// `byte_savings` bytes of payload it saves?
///
/// **The gate every future menu extension must call.** §2.6's narrower
/// compact-time tiers (`UInt16` offsets, bucket-anchored anchors) are the
/// motivating case and are deliberately NOT in this window: adding a form to the
/// menu adds a realized core shape, and the incumbent's answer to that coupling
/// is one hand-rolled special case — the empty-layer pin, which refuses to
/// compact a zero-feature layer because "a compact verdict over zero features
/// would fork the layer's schema template". That special case is exactly
/// `template_cost_justifies(1, 0) == false`; this function generalizes it so the
/// next extension states its trade instead of re-deriving one.
///
/// What it does NOT do is re-open the choices already made. The per-layer
/// `EndTimeForm::Zero` / `Dur32` verdicts stay per layer: they are
/// feasibility-gated and within-family optimal (0 < 4 < 8 bytes a row, feasibility
/// monotone), so the cost term is offered to them and their extensions rather
/// than imposed on them.
///
/// Strict inequality: a break-even fork is not worth taking, and saturating
/// arithmetic keeps a nonsense `shape_delta` from wrapping into a cheap verdict.
pub fn template_cost_justifies(shape_delta: usize, byte_savings: u64) -> bool {
    template_cost_justifies_with(shape_delta, byte_savings, DEFAULT_TEMPLATE_COST_BYTES)
}

/// [`template_cost_justifies`] against a MEASURED per-template cost — normally
/// [`TemplateCollector::mean_template_bytes`] off a trial encode of the dataset.
pub fn template_cost_justifies_with(
    shape_delta: usize,
    byte_savings: u64,
    template_cost_bytes: u64,
) -> bool {
    byte_savings > (shape_delta as u64).saturating_mul(template_cost_bytes)
}

/// Canonical, human-readable SHAPE of a schema template: sorted schema
/// metadata plus the field list as `name:type`.
///
/// The grouping key of [`TemplateCollector::template_report`], and usable
/// directly against a [`TemplateRegistry`]'s entries by a tool inspecting a
/// built archive. Schema metadata is sorted because Arrow carries it in a
/// `HashMap`, whose iteration order must never leak into a report that a build
/// compares across runs.
///
/// A template that will not parse yields a fixed sentinel rather than the parse
/// error, so the grouping key stays stable (and so a corrupt entry shows up as
/// one row instead of scattering).
pub fn describe_template_shape(template: &[u8]) -> String {
    let Ok(reader) = StreamReader::try_new(std::io::Cursor::new(template), None) else {
        return "<unparseable schema template>".to_string();
    };
    let schema = reader.schema();
    let mut meta: Vec<(&str, &str)> = schema
        .metadata()
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    meta.sort_unstable();
    let fields = schema
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), compact_type_name(f.data_type())))
        .collect::<Vec<_>>()
        .join(", ");
    if meta.is_empty() {
        format!("[{fields}]")
    } else {
        let meta = meta
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("{{{meta}}} [{fields}]")
    }
}

/// Compact spelling of an Arrow type for a shape description.
///
/// Hand-written rather than `Debug`-derived for the nested types: arrow's
/// `Debug` for a `List` prints the whole inner `Field` (metadata, nullability,
/// dict id), which would make one shape row unreadable AND make the grouping key
/// sensitive to details that are not the shape. Leaf types fall through to
/// `Debug`, which prints exactly `Float64` / `UInt16` / `Utf8`.
fn compact_type_name(dt: &DataType) -> String {
    match dt {
        DataType::List(f) => format!("List<{}>", compact_type_name(f.data_type())),
        DataType::LargeList(f) => format!("LargeList<{}>", compact_type_name(f.data_type())),
        DataType::FixedSizeList(f, n) => {
            format!("FixedSizeList<{},{n}>", compact_type_name(f.data_type()))
        }
        DataType::Dictionary(k, v) => {
            format!(
                "Dictionary<{},{}>",
                compact_type_name(k),
                compact_type_name(v)
            )
        }
        DataType::Struct(fields) => {
            let inner = fields
                .iter()
                .map(|f| format!("{}:{}", f.name(), compact_type_name(f.data_type())))
                .collect::<Vec<_>>()
                .join(",");
            format!("Struct<{inner}>")
        }
        leaf => format!("{leaf:?}"),
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
    /// The `stt:vertex_time_feature_step_ms` schema-metadata value (TB-11
    /// extension 2): the step for FEATURE-ANCHORED per-vertex time deltas.
    ///
    /// A separate key rather than a reshaped [`Self::vt`], because TILE_META is
    /// EXTENDED and never mutated: a reader that understands only `vt` would
    /// otherwise resolve feature-anchored deltas against a layer origin that
    /// does not exist, placing every vertex at the wrong instant with no error.
    /// Emitting it therefore also declares the `vertex-time-feature-anchor`
    /// capability, which makes that a refusal at open instead.
    ///
    /// Mutually exclusive with `vt`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vtf: Option<u32>,
}
/// Arrow schema-metadata keys for the DELTA per-vertex time encoding: the
/// absolute origin (Unix ms) the deltas are measured from, and the ms step each
/// delta counts. Both are present exactly when `vertex_time` ships as
/// `List<UInt16>` deltas; without them those deltas cannot be resolved back to
/// absolute times.
pub(crate) const VERTEX_TIME_ORIGIN_KEY: &str = "stt:vertex_time_origin_ms";
pub(crate) const VERTEX_TIME_STEP_KEY: &str = "stt:vertex_time_step_ms";
/// Arrow schema-metadata key for the FEATURE-ANCHORED per-vertex time encoding
/// (TB-11 extension 2): the ms step each `List<UInt16>` delta counts, measured
/// from **each feature's own `start_time`** rather than a layer-wide origin.
///
/// There is deliberately no companion origin key — the anchor already ships in
/// the CORE `start_time` column, which is what makes this tier free on the wire.
/// Present exactly when [`TileMeta::vtf`] is, and never together with
/// [`VERTEX_TIME_ORIGIN_KEY`]/[`VERTEX_TIME_STEP_KEY`].
pub(crate) const VERTEX_TIME_FEATURE_STEP_KEY: &str = "stt:vertex_time_feature_step_ms";
/// Number of time buckets packed into each row of the `vertex_value_matrix`
/// column. The renderer reshapes the flat vertex-major list back into a
/// `[vertex][bucket]` grid using this count.
pub(crate) const VERTEX_VALUE_BUCKETS_KEY: &str = "stt:vertex_value_buckets";
/// Baked minimum feature start-time (integer Unix ms) for the layer. The TS
/// decoder relativizes every start/end time against this value so the times
/// fit an f32; baking it here lets the decoder skip its client-side min-scan
/// over the whole start-time column. Mirrors exactly what the decoder computes
/// (the min of the `start_time` column) — see `poopdeck:packages/core/src/tile.ts`.
pub(crate) const TIME_OFFSET_MS_KEY: &str = "stt:time_offset_ms";

#[cfg(test)]
mod template_cost_tests {
    use super::*;
    use crate::arrow_tile::{
        encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn, PropertyColumn,
    };

    fn point_layer() -> ColumnarLayer {
        ColumnarLayer {
            name: "points".to_string(),
            feature_ids: vec![1, 2, 3],
            start_times: vec![1000, 2000, 3000],
            end_times: vec![1500, 2500, 3500],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8], [-122.6, 37.9]]),
            vertex_times: None,
            vertex_values: None,
            vertex_value_matrix: None,
            triangles: None,
            polygon_parts: None,
            properties: vec![(
                "speed".to_string(),
                PropertyColumn::Numeric(vec![Some(10.0), None, Some(30.0)]),
            )],
        }
    }

    fn line_layer() -> ColumnarLayer {
        ColumnarLayer {
            name: "tracks".to_string(),
            feature_ids: vec![10, 11],
            start_times: vec![0, 100],
            end_times: vec![50, 200],
            geometry: GeometryColumn::LineString(vec![
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                vec![[5.0, 5.0], [6.0, 6.0]],
            ]),
            vertex_times: None,
            vertex_values: None,
            vertex_value_matrix: None,
            triangles: None,
            polygon_parts: None,
            properties: vec![],
        }
    }

    fn collecting_config() -> (EncoderConfig, Arc<TemplateCollector>) {
        let collector = Arc::new(TemplateCollector::new());
        let cfg = EncoderConfig {
            template_collector: Some(collector.clone()),
            ..EncoderConfig::default()
        };
        (cfg, collector)
    }

    /// The gate prices a fork strictly, saturates safely, and reproduces the
    /// incumbent's one hand-rolled special case: an EMPTY layer saves nothing by
    /// compacting its time columns, so the fork it would cost is not justified.
    #[test]
    fn template_cost_justifies_generalizes_the_empty_layer_pin() {
        // The empty-layer pin (encode.rs `choose_time_forms`): one extra core
        // shape, zero features, therefore zero bytes saved.
        assert!(!template_cost_justifies(1, 0));
        // A layer with 10 000 features saving 4 B on each does justify it.
        assert!(template_cost_justifies(1, 10_000 * 4));
        // Break-even is not worth a fork; one byte past it is.
        assert!(!template_cost_justifies(1, DEFAULT_TEMPLATE_COST_BYTES));
        assert!(template_cost_justifies(1, DEFAULT_TEMPLATE_COST_BYTES + 1));
        // Forking nothing is justified by any positive saving, and by nothing
        // less — a neutral change is not a reason to churn bytes.
        assert!(template_cost_justifies(0, 1));
        assert!(!template_cost_justifies(0, 0));
        // Two shapes cost two templates.
        assert!(!template_cost_justifies_with(2, 2_000, 1_000));
        assert!(template_cost_justifies_with(2, 2_001, 1_000));
        // Nonsense inputs saturate instead of wrapping into a cheap verdict.
        assert!(!template_cost_justifies_with(
            usize::MAX,
            u64::MAX - 1,
            u64::MAX
        ));
    }

    /// A hand-built two-shape dataset: three distinct templates, one row each,
    /// bytes accounted exactly.
    #[test]
    fn template_report_counts_a_two_shape_dataset() {
        let (cfg, collector) = collecting_config();
        // Two geometry kinds — the deliberately-forking build. The point layer
        // carries a property column (CORE + PROPS templates); the line layer has
        // none (CORE only).
        encode_tile_with(&[point_layer()], &cfg).unwrap();
        encode_tile_with(&[line_layer()], &cfg).unwrap();

        let report = collector.template_report();
        assert_eq!(collector.len(), 3, "point CORE + point PROPS + line CORE");
        assert_eq!(report.len(), 3, "three distinct shapes");
        for (shape, count, bytes) in &report {
            assert_eq!(*count, 1, "no shape forked a second template: {shape}");
            assert!(*bytes > 0);
        }
        assert_eq!(
            report.iter().map(|(_, _, b)| *b).sum::<u64>(),
            collector.total_template_bytes()
        );
        assert_eq!(
            collector.mean_template_bytes(),
            Some(collector.total_template_bytes() / 3)
        );

        let shapes: Vec<&str> = report.iter().map(|(s, _, _)| s.as_str()).collect();
        assert_eq!(
            shapes
                .iter()
                .filter(|s| s.contains("stt:geometry=geoarrow.point"))
                .count(),
            1
        );
        assert_eq!(
            shapes
                .iter()
                .filter(|s| s.contains("stt:geometry=geoarrow.linestring"))
                .count(),
            1
        );
        // The PROPS template carries no schema-level metadata at all — every
        // dataset-constant key lives on the CORE template.
        let props: Vec<&&str> = shapes.iter().filter(|s| s.starts_with('[')).collect();
        assert_eq!(props.len(), 1, "exactly one PROPS shape");
        assert!(
            props[0].contains("speed:Float64"),
            "PROPS shape names its column and type: {}",
            props[0]
        );

        // Re-encoding the SAME shapes records nothing new: |T| counts distinct
        // shapes, not tiles.
        encode_tile_with(&[point_layer()], &cfg).unwrap();
        encode_tile_with(&[line_layer()], &cfg).unwrap();
        assert_eq!(collector.template_report(), report);
    }

    /// The report is a pure function of the recorded SET — same rows whichever
    /// order the templates arrived in, and stable across repeated calls.
    #[test]
    fn template_report_is_deterministic_and_order_independent() {
        let (forward_cfg, forward) = collecting_config();
        encode_tile_with(&[point_layer()], &forward_cfg).unwrap();
        encode_tile_with(&[line_layer()], &forward_cfg).unwrap();

        let (reverse_cfg, reverse) = collecting_config();
        encode_tile_with(&[line_layer()], &reverse_cfg).unwrap();
        encode_tile_with(&[point_layer()], &reverse_cfg).unwrap();

        assert_eq!(forward.template_report(), reverse.template_report());
        assert_eq!(
            forward.total_template_bytes(),
            reverse.total_template_bytes()
        );
        // Idempotent: re-running the report cannot move a byte of it.
        let once = forward.template_report();
        assert_eq!(once, forward.template_report());
        assert_eq!(once, forward.template_report());
        // Rows come out sorted by shape description — a total order, so a build
        // that logs the report emits the same lines every run.
        let mut sorted = once.clone();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(once, sorted);
    }

    /// The finalize-time cross-check: silent when the realized shape count is
    /// within the pass-1 prediction, loud and specific when it is not.
    #[test]
    fn unpredicted_fork_warning_fires_only_on_an_overshoot() {
        let (cfg, collector) = collecting_config();
        encode_tile_with(&[point_layer()], &cfg).unwrap();
        encode_tile_with(&[line_layer()], &cfg).unwrap();
        assert_eq!(collector.len(), 3);

        assert_eq!(
            collector.unpredicted_fork_warning(3),
            None,
            "exact forecast"
        );
        assert_eq!(
            collector.unpredicted_fork_warning(4),
            None,
            "under-realizing a forecast is free and silent"
        );
        let warning = collector
            .unpredicted_fork_warning(2)
            .expect("an overshoot must warn");
        assert!(warning.contains("3 realized vs 2 predicted"), "{warning}");
        assert!(warning.contains("+1 unpredicted fork"), "{warning}");
        assert!(warning.contains("none duplicated"), "{warning}");

        // An empty collector never warns.
        assert_eq!(TemplateCollector::new().unpredicted_fork_warning(0), None);
        assert_eq!(TemplateCollector::new().mean_template_bytes(), None);
        assert_eq!(TemplateCollector::new().total_template_bytes(), 0);
    }

    /// The shape description is derived from the schema, sorts its metadata, and
    /// degrades to one stable key rather than an error string.
    #[test]
    fn describe_template_shape_is_schema_derived_and_stable() {
        let (cfg, collector) = collecting_config();
        encode_tile_with(&[point_layer()], &cfg).unwrap();
        let snapshot = collector.snapshot();
        let core = snapshot
            .iter()
            .map(|(_, bytes)| describe_template_shape(bytes))
            .find(|s| s.contains("stt:geometry"))
            .expect("a CORE template");
        // Metadata block first, sorted: `stt:geometry` before `stt:layer`.
        let geometry_at = core.find("stt:geometry").unwrap();
        let layer_at = core.find("stt:layer").unwrap();
        assert!(geometry_at < layer_at, "metadata is sorted: {core}");
        assert!(core.contains("stt:layer=points"), "{core}");
        // The reserved core columns, named with compact type spellings.
        for expected in [
            "id:UInt64",
            "start_time:",
            "geometry:FixedSizeList<Float64,2>",
        ] {
            assert!(core.contains(expected), "{core} is missing {expected}");
        }

        // Not a schema template at all → one stable sentinel key, so a corrupt
        // entry groups as a single report row instead of scattering.
        assert_eq!(
            describe_template_shape(&[0u8, 1, 2, 3]),
            "<unparseable schema template>"
        );
        assert_eq!(
            describe_template_shape(&[]),
            "<unparseable schema template>"
        );
    }
}
