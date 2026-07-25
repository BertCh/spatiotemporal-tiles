//! Tile decoding — the v1 and v2 frame walks.
//!
//! Bounds-checked byte reads throughout, so arbitrary or truncated input
//! errors instead of panicking. A v2 layer is spliced back together from its
//! schema template + batch tail and merged into the v1-shaped single
//! [`RecordBatch`] every downstream consumer already expects.

use super::frame::{
    TemplateRegistry, TileMeta, ALIGNED_FRAME_FLAG, FRAME_ALIGN, FRAME_V2_ESCAPE, FRAME_V2_VERSION,
    REF_KIND_INLINE, REF_KIND_NO_PROPS, REF_KIND_TEMPLATE_HASH, SECTION_CORE_BATCH,
    SECTION_INLINE_SCHEMA_CORE, SECTION_INLINE_SCHEMA_PROPS, SECTION_PROPS_BATCH,
    SECTION_TILE_META, TIME_OFFSET_MS_KEY, VERTEX_TIME_ORIGIN_KEY, VERTEX_TIME_STEP_KEY,
    VERTEX_VALUE_BUCKETS_KEY,
};
use super::quantize::{AttrQuant, STT_QUANT_ATTR_META_KEY};
use crate::error::{Error, Result};
use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{Field, Schema};
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

/// Whether a tile payload carries the v2 sectioned frame (leading u16 is the
/// [`FRAME_V2_ESCAPE`]). The manifest's `formatVersion` remains authoritative
/// (★F6) — this sniff is defense-in-depth for the payload level.
pub fn is_frame_v2(payload: &[u8]) -> bool {
    payload.len() >= 2 && u16::from_le_bytes([payload[0], payload[1]]) == FRAME_V2_ESCAPE
}

/// Decode a full tile payload (the layer frame) into its layers.
///
/// Accepts the v1 frame in both shapes — aligned ([`ALIGNED_FRAME_FLAG`] set,
/// derived padding before each IPC stream) and the legacy unpadded frame
/// written before the flag existed — plus **self-contained** v2 frames
/// (inline schema sections). A v2 frame that references templates by hash
/// needs the dataset's registry: use [`decode_tile_with_templates`] (a hash
/// reference here is a descriptive error, not a panic).
pub fn decode_tile(payload: &[u8]) -> Result<Vec<DecodedLayer>> {
    if is_frame_v2(payload) {
        return decode_tile_v2(payload, None);
    }
    decode_tile_v1(payload)
}

/// [`decode_tile`] with the dataset's [`TemplateRegistry`], so v2 frames can
/// resolve their 16-byte template-hash references. v1 frames decode
/// unchanged (the registry is simply unused).
pub fn decode_tile_with_templates(
    payload: &[u8],
    templates: &TemplateRegistry,
) -> Result<Vec<DecodedLayer>> {
    if is_frame_v2(payload) {
        return decode_tile_v2(payload, Some(templates));
    }
    decode_tile_v1(payload)
}

/// The v1 frame walk — UNCHANGED from the 0.3.x reader.
fn decode_tile_v1(payload: &[u8]) -> Result<Vec<DecodedLayer>> {
    if payload.len() < 2 {
        return Err(Error::Other(
            "tile payload too short for layer frame".into(),
        ));
    }
    let raw_count = u16::from_le_bytes([payload[0], payload[1]]);
    let aligned = raw_count & ALIGNED_FRAME_FLAG != 0;
    let count = (raw_count & !ALIGNED_FRAME_FLAG) as usize;
    let mut pos = 2usize;
    // Cap the pre-allocation: `count` is attacker-controlled (up to 0x7FFE) and
    // each real layer costs many wire bytes, so a doctored count must not force
    // a giant up-front allocation. Mirrors the v2 walker.
    let mut layers = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let name_len = read_u16(payload, &mut pos)? as usize;
        let name = read_slice(payload, &mut pos, name_len)?;
        let name = String::from_utf8(name.to_vec())
            .map_err(|e| Error::Other(format!("layer name not utf8: {e}")))?;
        let ipc_len = read_u32(payload, &mut pos)? as usize;
        if aligned {
            let pad = (FRAME_ALIGN - pos % FRAME_ALIGN) % FRAME_ALIGN;
            read_slice(payload, &mut pos, pad)?;
        }
        let ipc = read_slice(payload, &mut pos, ipc_len)?;
        let batch = decode_layer(ipc)?;
        layers.push(DecodedLayer { name, batch });
    }
    Ok(layers)
}

/// Splice a template onto a section tail and decode the resulting stream.
///
/// Normative guards (design §3.4): the tail is EXACTLY the TOC-declared bytes
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

/// Resolve a v2 layer's schema template: inline section or registry lookup.
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

/// Merge a v2 layer's decoded CORE + PROPS batches back into the v1-shaped
/// single batch, RE-INJECTING the TILE_META values into schema/field metadata
/// with the exact v1 formatting — so every downstream consumer (validator,
/// stt-optimize, tests, renderers) sees v1-equivalent decoded layers with
/// zero changes.
pub(crate) fn merge_v2_layer(
    core: RecordBatch,
    props: Option<RecordBatch>,
    meta: &TileMeta,
) -> Result<RecordBatch> {
    let mut fields: Vec<Arc<Field>> = core.schema().fields().iter().cloned().collect();
    let mut columns: Vec<ArrayRef> = core.columns().to_vec();
    let mut schema_meta: HashMap<String, String> = core.schema().metadata().clone();

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
            // (byte-identical to the v1 field metadata: `to_json` is a pure
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

    // Schema-level re-injection, mirroring the v1 assembler's formatting.
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
    RecordBatch::try_new(schema, columns)
        .map_err(|e| Error::Other(format!("failed to merge v2 CORE/PROPS batches: {e}")))
}

/// The v2 frame walk. Bounds-checked byte reads throughout (`read_slice`), so
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

/// Walk ONLY a v2 frame's header structure — escape/version/flags/count,
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
