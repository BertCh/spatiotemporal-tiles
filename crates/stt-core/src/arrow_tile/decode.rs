//! Tile decoding — the layer-frame walk plus the standalone single-layer IPC
//! decode.
//!
//! Bounds-checked byte reads throughout, so arbitrary or truncated input
//! errors instead of panicking. A framed layer is spliced back together from
//! its schema template + batch tail and merged into the single flat
//! [`RecordBatch`] every downstream consumer already expects — the same shape
//! [`decode_layer`] returns.

use super::frame::{
    EndTimeForm, StartTimeForm, TemplateRegistry, TileMeta, FRAME_ALIGN, FRAME_V2_ESCAPE,
    FRAME_V2_VERSION, REF_KIND_INLINE, REF_KIND_NO_PROPS, REF_KIND_TEMPLATE_HASH,
    SECTION_CORE_BATCH, SECTION_INLINE_SCHEMA_CORE, SECTION_INLINE_SCHEMA_PROPS,
    SECTION_PROPS_BATCH, SECTION_TILE_META, TIME_OFFSET_MS_KEY, VERTEX_TIME_ORIGIN_KEY,
    VERTEX_TIME_STEP_KEY, VERTEX_VALUE_BUCKETS_KEY,
};
use super::quantize::{AttrQuant, STT_QUANT_ATTR_META_KEY, VERTEX_VALUE_QUANT_SENTINEL};
use crate::error::{Error, Result};
use arrow::array::{
    Array, ArrayRef, Float32Array, Int64Array, ListArray, RecordBatch, UInt16Array, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

// ----------------------------------------------------------------------------
// Decoding
// ----------------------------------------------------------------------------

/// A decoded tile layer: its name and the raw Arrow [`RecordBatch`].
#[derive(Debug, Clone)]
pub struct DecodedLayer {
    /// Layer name from the layer frame.
    pub name: String,
    /// The decoded Arrow record batch.
    pub batch: RecordBatch,
}

/// Decode a single-layer Arrow IPC stream into a [`RecordBatch`].
pub fn decode_layer(ipc: &[u8]) -> Result<RecordBatch> {
    let reader = StreamReader::try_new(ipc, None)
        .map_err(|e| Error::Other(format!("Arrow IPC reader init failed: {e}")))?;
    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|e| Error::Other(format!("Arrow IPC read failed: {e}")))?);
    }
    match batches.len() {
        0 => Err(Error::Other(
            "tile layer IPC contained no record batch".into(),
        )),
        1 => Ok(batches.into_iter().next().unwrap()),
        // A layer is written as exactly one batch; concatenating is the safe
        // fallback if a producer ever splits it.
        _ => arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .map_err(|e| Error::Other(format!("failed to concat tile batches: {e}"))),
    }
}

/// Whether a tile payload carries the sectioned layer frame (leading u16 is
/// the [`FRAME_V2_ESCAPE`]). The manifest's `formatVersion` remains
/// authoritative — a reader dispatches on it, so a frame disagreeing with the
/// manifest is a hard error — while this sniff is defense-in-depth at the
/// payload level, rejecting a payload that is not a frame at all.
pub fn is_frame_v2(payload: &[u8]) -> bool {
    payload.len() >= 2 && u16::from_le_bytes([payload[0], payload[1]]) == FRAME_V2_ESCAPE
}

/// Decode a full tile payload (the layer frame) into its layers.
///
/// Accepts **self-contained** frames (inline schema sections). A frame that
/// references templates by hash needs the dataset's registry: use
/// [`decode_tile_with_templates`] (a hash reference here is a descriptive
/// error, not a panic).
pub fn decode_tile(payload: &[u8]) -> Result<Vec<DecodedLayer>> {
    if !is_frame_v2(payload) {
        return Err(Error::Other(
            "tile payload is not a layer frame (missing the frame escape)".into(),
        ));
    }
    decode_tile_v2(payload, None)
}

/// [`decode_tile`] with the dataset's [`TemplateRegistry`], so frames can
/// resolve their 16-byte template-hash references.
pub fn decode_tile_with_templates(
    payload: &[u8],
    templates: &TemplateRegistry,
) -> Result<Vec<DecodedLayer>> {
    if !is_frame_v2(payload) {
        return Err(Error::Other(
            "tile payload is not a layer frame (missing the frame escape)".into(),
        ));
    }
    decode_tile_v2(payload, Some(templates))
}

/// Splice a template onto a section tail and decode the resulting stream.
///
/// Normative guards: the tail is EXACTLY the TOC-declared bytes
/// (the caller sliced it that way) and MUST begin with the `0xFFFFFFFF`
/// continuation marker — stray zero bytes make arrow-rs silently return an
/// EMPTY tile (they parse as a legacy 4-byte end-of-stream), so a malformed
/// section must error loudly instead. The template gets the same check
/// (a corrupt manifest entry hashes consistently but still must not splice).
pub(crate) fn splice_decode(template: &[u8], tail: &[u8], what: &str) -> Result<RecordBatch> {
    if template.len() < 4 || template[0..4] != [0xFF, 0xFF, 0xFF, 0xFF] {
        return Err(Error::Other(format!(
            "{what}: schema template does not start with an encapsulated Arrow message"
        )));
    }
    if tail.len() < 4 || tail[0..4] != [0xFF, 0xFF, 0xFF, 0xFF] {
        return Err(Error::Other(format!(
            "{what}: batch section does not start with the 0xFFFFFFFF continuation marker \
             (corrupt or misaligned section)"
        )));
    }
    let mut buf = Vec::with_capacity(template.len() + tail.len());
    buf.extend_from_slice(template);
    buf.extend_from_slice(tail);
    decode_layer(&buf)
}

/// Resolve a framed layer's schema template: inline section or registry lookup.
fn resolve_template<'a>(
    ref_kind: u8,
    hash: Option<[u8; 16]>,
    inline: Option<&'a [u8]>,
    registry: Option<&'a TemplateRegistry>,
    what: &str,
) -> Result<&'a [u8]> {
    match ref_kind {
        REF_KIND_INLINE => inline.ok_or_else(|| {
            Error::Other(format!(
                "{what}: inline schema section missing from the frame"
            ))
        }),
        REF_KIND_TEMPLATE_HASH => {
            let hash = hash.expect("hash read for ref_kind 1");
            let registry = registry.ok_or_else(|| {
                Error::Other(format!(
                    "{what}: frame references schema template {} but no template registry \
                     was provided — open the dataset through its manifest (or use \
                     decode_tile_with_templates)",
                    hex_16(&hash)
                ))
            })?;
            registry.get(&hash).ok_or_else(|| {
                Error::Other(format!(
                    "{what}: schema template {} is not in the dataset's registry \
                     (manifest.schemas is incomplete or the frame is corrupt)",
                    hex_16(&hash)
                ))
            })
        }
        other => Err(Error::Other(format!(
            "{what}: unknown schema ref_kind {other} (this reader knows 0..=2)"
        ))),
    }
}

fn hex_16(hash: &[u8; 16]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// Re-inflate the compact feature-time columns declared by `TILE_META.st` /
/// `.et` into the absolute, non-null `Int64` `start_time` / `end_time` pair
/// every consumer of a decoded layer already expects.
///
/// This is the whole reason the compact encoding needs no downstream change:
/// the merged batch is INDISTINGUISHABLE from a non-compact tile's, right down
/// to column order — a synthesized `end_time` (`et: "zero"`) is inserted
/// immediately after `start_time`, its canonical index, rather than appended.
///
/// Operates on the CORE prefix, before property columns are appended, and is a
/// no-op for any tile that declares neither key.
fn reinflate_compact_times(
    fields: &mut Vec<Arc<Field>>,
    columns: &mut Vec<ArrayRef>,
    meta: &TileMeta,
) -> Result<()> {
    if meta.st.is_none() && meta.et.is_none() {
        return Ok(());
    }
    let start_idx = fields
        .iter()
        .position(|f| f.name() == "start_time")
        .ok_or_else(|| {
            Error::Other(
                "TILE_META declares a compact time encoding but the layer has no \
                 'start_time' column"
                    .into(),
            )
        })?;

    if meta.st == Some(StartTimeForm::U32Offset) {
        let t0 = meta.t0.ok_or_else(|| {
            Error::Other(
                "TILE_META declares st=\"u32\" (start_time as a u32 offset) but carries \
                 no 't0' anchor to reconstruct against"
                    .into(),
            )
        })?;
        let offsets = columns[start_idx]
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                Error::Other(format!(
                    "TILE_META declares st=\"u32\" but 'start_time' is {}",
                    columns[start_idx].data_type()
                ))
            })?;
        // `saturating_add`, not `+`: a crafted TILE_META (`t0` near i64::MAX)
        // must not panic a debug-build decoder on an overflowing sum.
        let values: Vec<i64> = offsets
            .values()
            .iter()
            .map(|&o| t0.saturating_add(o as i64))
            .collect();
        columns[start_idx] = Arc::new(Int64Array::new(values.into(), offsets.nulls().cloned()));
        fields[start_idx] = Arc::new(
            fields[start_idx]
                .as_ref()
                .clone()
                .with_data_type(DataType::Int64),
        );
    }

    let Some(et) = meta.et else {
        return Ok(());
    };
    // Both end forms are relative to the (now absolute) start column.
    let start_col = columns[start_idx].clone();
    let starts = start_col
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            Error::Other(format!(
                "TILE_META declares a compact 'end_time' but 'start_time' is {} \
                 (expected Int64 after re-inflation)",
                start_col.data_type()
            ))
        })?;

    match et {
        EndTimeForm::Zero => {
            if fields.iter().any(|f| f.name() == "end_time") {
                return Err(Error::Other(
                    "TILE_META declares et=\"zero\" (the end_time column is omitted) but \
                     the layer carries an 'end_time' column"
                        .into(),
                ));
            }
            fields.insert(
                start_idx + 1,
                Arc::new(Field::new("end_time", DataType::Int64, false)),
            );
            // `end == start` for every feature, so the reconstructed column IS
            // the start column — shared, not copied.
            columns.insert(start_idx + 1, start_col.clone());
        }
        EndTimeForm::Dur32 => {
            let end_idx = fields
                .iter()
                .position(|f| f.name() == "end_time")
                .ok_or_else(|| {
                    Error::Other(
                        "TILE_META declares et=\"dur32\" but the layer has no 'end_time' \
                         column"
                            .into(),
                    )
                })?;
            let durations = columns[end_idx]
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    Error::Other(format!(
                        "TILE_META declares et=\"dur32\" but 'end_time' is {}",
                        columns[end_idx].data_type()
                    ))
                })?;
            if durations.len() != starts.len() {
                return Err(Error::Other(format!(
                    "compact time columns disagree on length: start_time {} vs end_time {}",
                    starts.len(),
                    durations.len()
                )));
            }
            let values: Vec<i64> = starts
                .values()
                .iter()
                .zip(durations.values())
                .map(|(&s, &d)| s.saturating_add(d as i64))
                .collect();
            columns[end_idx] = Arc::new(Int64Array::new(values.into(), durations.nulls().cloned()));
            fields[end_idx] = Arc::new(
                fields[end_idx]
                    .as_ref()
                    .clone()
                    .with_data_type(DataType::Int64),
            );
        }
    }
    Ok(())
}

/// The `TILE_META.vq` keys a reader will act on. Anything else in the map is a
/// crafted/corrupt section, not an additive extension: the affine RE-TYPES a
/// named column, so applying one to a column outside this set could only
/// corrupt it.
const QUANTIZABLE_VERTEX_VALUE_COLUMNS: [&str; 2] = ["vertex_value", "vertex_value_matrix"];

/// Re-inflate the per-vertex value columns declared by `TILE_META.vq` from
/// their `UInt16` leaf back to the `List<Float32>` shape every consumer of a
/// decoded layer already expects (`value = o + q*s`, with the reserved
/// [`VERTEX_VALUE_QUANT_SENTINEL`] index becoming `NaN`).
///
/// The sibling of [`reinflate_compact_times`], and the reason the quantization
/// needs no downstream change: the merged batch is INDISTINGUISHABLE from a
/// non-quantized tile's — same column, same position, same Arrow type, same
/// list offsets and null buffers.
///
/// Operates on the CORE prefix, and is a no-op for every archive built without
/// the flag (`vq` absent).
fn reinflate_quantized_vertex_values(
    fields: &mut [Arc<Field>],
    columns: &mut [ArrayRef],
    meta: &TileMeta,
) -> Result<()> {
    let Some(vq) = meta.vq.as_ref() else {
        return Ok(());
    };
    for (name, &(o, s)) in vq {
        if !QUANTIZABLE_VERTEX_VALUE_COLUMNS.contains(&name.as_str()) {
            return Err(Error::Other(format!(
                "TILE_META.vq names column '{name}', which is not a per-vertex value column \
                 (this reader knows {QUANTIZABLE_VERTEX_VALUE_COLUMNS:?})"
            )));
        }
        let idx = fields
            .iter()
            .position(|f| f.name() == name)
            .ok_or_else(|| {
                Error::Other(format!(
                    "TILE_META.vq carries an affine for '{name}' but the layer has no such \
                     column"
                ))
            })?;
        let list = columns[idx]
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                Error::Other(format!(
                    "TILE_META.vq declares '{name}' quantized but the column is {} \
                     (expected a List)",
                    columns[idx].data_type()
                ))
            })?;
        let child = list
            .values()
            .as_any()
            .downcast_ref::<UInt16Array>()
            .ok_or_else(|| {
                Error::Other(format!(
                    "TILE_META.vq declares '{name}' quantized but its list leaf is {} \
                     (expected UInt16)",
                    list.values().data_type()
                ))
            })?;
        // The sentinel is the format's `NaN` (no value at this vertex); every
        // other index is the affine applied in f64 and narrowed once, exactly
        // mirroring the encoder.
        let values: Vec<f32> = child
            .values()
            .iter()
            .map(|&q| {
                if q == VERTEX_VALUE_QUANT_SENTINEL {
                    f32::NAN
                } else {
                    (o + q as f64 * s) as f32
                }
            })
            .collect();
        let child_field = Arc::new(Field::new("item", DataType::Float32, true));
        let inflated = ListArray::new(
            child_field,
            list.offsets().clone(),
            Arc::new(Float32Array::new(values.into(), child.nulls().cloned())),
            list.nulls().cloned(),
        );
        fields[idx] = Arc::new(
            fields[idx]
                .as_ref()
                .clone()
                .with_data_type(inflated.data_type().clone()),
        );
        columns[idx] = Arc::new(inflated);
    }
    Ok(())
}

/// Merge a framed layer's decoded CORE + PROPS batches back into the single
/// flat batch, RE-INJECTING the TILE_META values into schema/field metadata
/// with the exact formatting the standalone `encode_layer` path writes — so
/// every downstream consumer (validator, stt-optimize, tests, renderers) sees
/// ONE decoded-layer shape and never branches on how the tile was framed.
pub(crate) fn merge_v2_layer(
    core: RecordBatch,
    props: Option<RecordBatch>,
    meta: &TileMeta,
) -> Result<RecordBatch> {
    let mut fields: Vec<Arc<Field>> = core.schema().fields().iter().cloned().collect();
    let mut columns: Vec<ArrayRef> = core.columns().to_vec();
    let mut schema_meta: HashMap<String, String> = core.schema().metadata().clone();

    // Re-inflate the compact time columns FIRST, while `fields`/`columns` are
    // still exactly the CORE prefix, so a synthesized `end_time` lands at its
    // canonical index instead of after the property columns.
    reinflate_compact_times(&mut fields, &mut columns, meta)?;
    // Index-independent (it looks its columns up by name), so it is unaffected
    // by the `end_time` the step above may have inserted.
    reinflate_quantized_vertex_values(&mut fields, &mut columns, meta)?;

    if let Some(props) = props {
        if props.num_rows() != core.num_rows() {
            return Err(Error::Other(format!(
                "tile layer CORE/PROPS row counts disagree: {} vs {}",
                core.num_rows(),
                props.num_rows()
            )));
        }
        for (field, column) in props.schema().fields().iter().zip(props.columns()) {
            // Re-inject the hoisted attribute-quantization affine
            // (byte-identical to the Arrow field metadata the standalone
            // encode path writes: `to_json` is a pure
            // function of the `[o, s]` pair TILE_META carries).
            let field = match meta.qa.as_ref().and_then(|qa| qa.get(field.name())) {
                Some(&(o, s)) => {
                    let mut m = field.metadata().clone();
                    m.insert(
                        STT_QUANT_ATTR_META_KEY.to_string(),
                        AttrQuant { o, s }.to_json(),
                    );
                    Arc::new(field.as_ref().clone().with_metadata(m))
                }
                None => field.clone(),
            };
            fields.push(field);
            columns.push(column.clone());
        }
    }

    // Schema-level re-injection, mirroring the standalone assembler's formatting.
    if let Some(t0) = meta.t0 {
        schema_meta.insert(TIME_OFFSET_MS_KEY.to_string(), t0.to_string());
    }
    if let Some((origin, step)) = meta.vt {
        schema_meta.insert(VERTEX_TIME_ORIGIN_KEY.to_string(), origin.to_string());
        schema_meta.insert(VERTEX_TIME_STEP_KEY.to_string(), step.to_string());
    }
    if let Some(buckets) = meta.vb {
        schema_meta.insert(VERTEX_VALUE_BUCKETS_KEY.to_string(), buckets.to_string());
    }

    let schema = Arc::new(Schema::new_with_metadata(fields, schema_meta));
    RecordBatch::try_new(schema, columns).map_err(|e| {
        Error::Other(format!(
            "failed to merge layer-frame CORE/PROPS batches: {e}"
        ))
    })
}

/// The layer-frame walk. Bounds-checked byte reads throughout (`read_slice`), so
/// arbitrary/truncated input errors instead of panicking; unknown section
/// tags are skipped via their TOC length (additive evolution).
fn decode_tile_v2(
    payload: &[u8],
    registry: Option<&TemplateRegistry>,
) -> Result<Vec<DecodedLayer>> {
    let mut pos = 0usize;
    let escape = read_u16(payload, &mut pos)?;
    debug_assert_eq!(escape, FRAME_V2_ESCAPE, "caller dispatched on the escape");
    let frame_version = read_slice(payload, &mut pos, 1)?[0];
    if frame_version != FRAME_V2_VERSION {
        return Err(Error::Other(format!(
            "unsupported layer-frame version {frame_version} (this reader knows v2)"
        )));
    }
    let flags = read_slice(payload, &mut pos, 1)?[0];
    if flags != 0 {
        return Err(Error::Other(format!(
            "reserved v2 layer-frame flags must be 0, got {flags:#04x}"
        )));
    }
    let count = read_u16(payload, &mut pos)? as usize;
    let mut layers = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let name_len = read_u16(payload, &mut pos)? as usize;
        let name = read_slice(payload, &mut pos, name_len)?;
        let name = String::from_utf8(name.to_vec())
            .map_err(|e| Error::Other(format!("layer name not utf8: {e}")))?;

        let mut read_ref = |what: &str| -> Result<(u8, Option<[u8; 16]>)> {
            let kind = read_slice(payload, &mut pos, 1)?[0];
            let hash = if kind == REF_KIND_TEMPLATE_HASH {
                let mut h = [0u8; 16];
                h.copy_from_slice(read_slice(payload, &mut pos, 16)?);
                Some(h)
            } else if kind == REF_KIND_INLINE || kind == REF_KIND_NO_PROPS {
                None
            } else {
                return Err(Error::Other(format!(
                    "layer '{name}' {what}: unknown schema ref_kind {kind} \
                     (this reader knows 0..=2)"
                )));
            };
            Ok((kind, hash))
        };
        let (ref_core, core_hash) = read_ref("core")?;
        if ref_core == REF_KIND_NO_PROPS {
            return Err(Error::Other(format!(
                "layer '{name}': ref_kind_core 2 is invalid (every layer has a CORE batch)"
            )));
        }
        let (ref_props, props_hash) = read_ref("props")?;

        let section_count = read_slice(payload, &mut pos, 1)?[0] as usize;
        let mut toc: Vec<(u8, usize)> = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            let tag = read_slice(payload, &mut pos, 1)?[0];
            let len = read_u32(payload, &mut pos)? as usize;
            toc.push((tag, len));
        }
        let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
        read_slice(payload, &mut pos, pad)?;

        let mut sections: BTreeMap<u8, &[u8]> = BTreeMap::new();
        for (tag, len) in toc {
            let bytes = read_slice(payload, &mut pos, len)?;
            let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
            read_slice(payload, &mut pos, pad)?;
            if sections.insert(tag, bytes).is_some() {
                return Err(Error::Other(format!(
                    "layer '{name}': duplicate section tag 0x{tag:02x} in the TOC"
                )));
            }
        }

        // TILE_META: canonical JSON, unknown keys ignored (additive contract).
        let tile_meta: TileMeta = match sections.get(&SECTION_TILE_META) {
            Some(bytes) => serde_json::from_slice(bytes).map_err(|e| {
                Error::Other(format!("layer '{name}': TILE_META JSON decode failed: {e}"))
            })?,
            None => TileMeta::default(),
        };

        let core_template = resolve_template(
            ref_core,
            core_hash,
            sections.get(&SECTION_INLINE_SCHEMA_CORE).copied(),
            registry,
            &format!("layer '{name}' core"),
        )?;
        let core_tail = sections
            .get(&SECTION_CORE_BATCH)
            .ok_or_else(|| Error::Other(format!("layer '{name}': CORE_BATCH section missing")))?;
        let core = splice_decode(core_template, core_tail, &format!("layer '{name}' core"))?;

        let props = if ref_props == REF_KIND_NO_PROPS {
            if sections.contains_key(&SECTION_PROPS_BATCH) {
                return Err(Error::Other(format!(
                    "layer '{name}': PROPS_BATCH section present but ref_kind_props \
                     declares no props"
                )));
            }
            None
        } else {
            let template = resolve_template(
                ref_props,
                props_hash,
                sections.get(&SECTION_INLINE_SCHEMA_PROPS).copied(),
                registry,
                &format!("layer '{name}' props"),
            )?;
            let tail = sections.get(&SECTION_PROPS_BATCH).ok_or_else(|| {
                Error::Other(format!("layer '{name}': PROPS_BATCH section missing"))
            })?;
            Some(splice_decode(
                template,
                tail,
                &format!("layer '{name}' props"),
            )?)
        };

        let batch = merge_v2_layer(core, props, &tile_meta)?;
        layers.push(DecodedLayer { name, batch });
    }
    Ok(layers)
}

/// Walk ONLY a layer frame's header structure — escape/version/flags/count,
/// per-layer name + schema ref_kinds (+ their 16-byte hashes), TOC-driven
/// section skips; **no Arrow decode, no section parse** — and return every
/// template hash the frame references. The packed-format validators use this
/// to prove each referenced hash resolves in `manifest.schemas` without
/// decoding tiles. Errors (never panics) on truncated/malformed headers.
pub fn frame_v2_template_refs(payload: &[u8]) -> Result<Vec<[u8; 16]>> {
    if !is_frame_v2(payload) {
        return Err(Error::Other("not a v2 layer frame (missing escape)".into()));
    }
    let mut pos = 2usize; // past the escape
    let frame_version = read_slice(payload, &mut pos, 1)?[0];
    if frame_version != FRAME_V2_VERSION {
        return Err(Error::Other(format!(
            "unsupported layer-frame version {frame_version} (this reader knows v2)"
        )));
    }
    let flags = read_slice(payload, &mut pos, 1)?[0];
    if flags != 0 {
        return Err(Error::Other(format!(
            "reserved v2 layer-frame flags must be 0, got {flags:#04x}"
        )));
    }
    let count = read_u16(payload, &mut pos)? as usize;
    let mut refs: Vec<[u8; 16]> = Vec::new();
    for _ in 0..count {
        let name_len = read_u16(payload, &mut pos)? as usize;
        read_slice(payload, &mut pos, name_len)?;
        for what in ["core", "props"] {
            let kind = read_slice(payload, &mut pos, 1)?[0];
            if kind == REF_KIND_TEMPLATE_HASH {
                let mut h = [0u8; 16];
                h.copy_from_slice(read_slice(payload, &mut pos, 16)?);
                refs.push(h);
            } else if kind != REF_KIND_INLINE && kind != REF_KIND_NO_PROPS {
                return Err(Error::Other(format!(
                    "{what}: unknown schema ref_kind {kind} (this reader knows 0..=2)"
                )));
            }
        }
        let section_count = read_slice(payload, &mut pos, 1)?[0] as usize;
        let mut toc: Vec<usize> = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            read_slice(payload, &mut pos, 1)?; // tag
            toc.push(read_u32(payload, &mut pos)? as usize);
        }
        let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
        read_slice(payload, &mut pos, pad)?;
        for len in toc {
            read_slice(payload, &mut pos, len)?;
            let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
            read_slice(payload, &mut pos, pad)?;
        }
    }
    Ok(refs)
}

fn read_u16(buf: &[u8], pos: &mut usize) -> Result<u16> {
    let s = read_slice(buf, pos, 2)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let s = read_slice(buf, pos, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_slice<'a>(buf: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::Other("tile frame length overflow".into()))?;
    if end > buf.len() {
        return Err(Error::Other("tile frame truncated".into()));
    }
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}
