//! STT **packed format** — a multi-object, edge-cacheable container.
//!
//! Replaces the single-file v4 archive (`ArchiveWriter`) as
//! the canonical write path. A single-file archive cannot be edge-cached once it
//! exceeds the CDN per-object limit (Cloudflare = 512 MB): every range request
//! hits origin forever. The packed format makes the *cacheable unit a small
//! object, not the whole dataset* — tile blobs are split into many
//! content-addressed **pack** objects (each `≤ pack_target_bytes`) plus a tiny
//! mutable manifest. A dumb CDN caches each immutable pack natively; no Worker,
//! no vendor lock-in.
//!
//! ## On-disk layout (per dataset)
//!
//! ```text
//! <out_dir>/
//!   manifest.json            tiny, MUTABLE   → short TTL
//!   index/<blake3>.sttd      directory blob  → IMMUTABLE (content-addressed)
//!   packs/<blake3>.sttp      tile blob data  → IMMUTABLE (content-addressed)
//!   packs/<blake3>.sttp
//!   ...
//! ```
//!
//! Packs and the directory are content-addressed (blake3, 128-bit → 32 hex
//! chars) so their bytes never change without their name changing. The directory
//! is the v6 codec ([`crate::directory`]): per-run `pack_id` column +
//! pack-relative offsets. See `docs/spec/stt-packed-format.md` for the full
//! contract.
//!
//! [`PackWriter`] reuses the dedup / per-blob-zstd / curve-ordering logic of
//! `ArchiveWriter::finalize_buffered`, then cuts the ordered, deduped blob stream
//! into packs. [`PackedReader`] is the local-file reader (the TS reader handles
//! remote/HTTP); it mirrors `ArchiveReader` semantics.

use crate::arrow_tile::{TemplateCollector, TemplateRegistry};
use crate::compression;
use crate::curve::BlobOrdering;
use crate::directory::TileEntry;
use crate::error::{Error, Result};
use crate::metadata::Metadata;
use crate::tile::TileId;
use crate::types::Compression;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `format` discriminator written into every packed manifest.
pub const PACKED_FORMAT: &str = "stt-packed";

/// Packed-format version (the manifest schema, distinct from the directory
/// codec's [`crate::directory::DIRECTORY_VERSION`]) this toolchain emits and
/// accepts: `STTP`/`STTD` object magic, object-absolute blob offsets, and
/// sectioned layer frames referencing manifest-embedded schema templates
/// (`manifest.schemas`). See
/// `docs/roadmap/stt-packed-format-decisions.md`.
///
/// Any manifest version other than v3 is a hard open error. Refusing loudly
/// prevents legacy or future bytes from being misdecoded as a v3 dataset.
pub const PACKED_FORMAT_VERSION: u32 = 3;
/// Oldest manifest `formatVersion` the READER accepts.
///
/// v2 differs from v3 in the CONTAINER only: it has no `variants` registry (the
/// variant axis did not exist, so every payload is raw) and its directory codec
/// is v5 (no per-entry `variant_id`). The tile payloads are byte-identical in
/// shape — [`crate::arrow_tile::LAYER_FRAME_VERSION`] is 2 for both — so no
/// decode path forks below the container.
///
/// Read-only: [`PACKED_FORMAT_VERSION`] is what every writer emits, so this
/// cannot put a v2 manifest into the world. It exists because a published
/// archive is a durable artifact and some have no reproducible source.
pub const MIN_PACKED_FORMAT_VERSION: u32 = 2;

// ----------------------------------------------------------------------------
// Object magic (`.sttp` / `.sttd` self-identification, spec §9.2)
// ----------------------------------------------------------------------------

/// First 4 bytes of a formatVersion-3 pack object.
pub const PACK_MAGIC: [u8; 4] = *b"STTP";
/// First 4 bytes of a formatVersion-3 directory object.
pub const DIRECTORY_MAGIC: [u8; 4] = *b"STTD";
/// Length of the object magic prelude: 4-byte tag + u8 object version + 3 zero
/// bytes. Blob offsets in a formatVersion-3 directory are object-absolute, so a
/// pack's first blob sits at offset 8; the manifest `length` fields and every
/// blake3 content address cover the ENTIRE object, prelude included, so a reader
/// that hashed only the post-prelude bytes would reject every object it opens.
/// Historical pre-v3 objects are not accepted by the current reader.
pub const OBJECT_MAGIC_LEN: usize = 8;
/// Version of the OBJECT-PRELUDE axis: the byte at offset 4 of every `.sttp` /
/// `.sttd`. Counted independently of [`PACKED_FORMAT_VERSION`] (manifest schema)
/// and of the layer-frame version, so a prelude revision need not disturb either.
const OBJECT_MAGIC_VERSION: u8 = 3;
/// Oldest object prelude the READER accepts. The prelude is a pure envelope —
/// a 4-byte kind tag, this version byte, three reserved zeros — so an older
/// value changes nothing about how the bytes AFTER it are parsed. Accepting it
/// costs one comparison and is what keeps published archives (several of which
/// have no reproducible source) openable.
///
/// The writer always emits [`OBJECT_MAGIC_VERSION`]; this bound is read-only.
const MIN_OBJECT_MAGIC_VERSION: u8 = 2;

/// The full 8-byte magic prelude for an object kind.
fn object_magic(kind: [u8; 4]) -> [u8; OBJECT_MAGIC_LEN] {
    [
        kind[0],
        kind[1],
        kind[2],
        kind[3],
        OBJECT_MAGIC_VERSION,
        0,
        0,
        0,
    ]
}

/// Validate an object's 8-byte magic prelude and return the bytes after it.
fn strip_object_magic<'a>(bytes: &'a [u8], kind: [u8; 4], what: &str) -> Result<&'a [u8]> {
    let tag = std::str::from_utf8(&kind).expect("magic tags are ASCII");
    if bytes.len() < OBJECT_MAGIC_LEN || bytes[0..4] != kind {
        return Err(Error::InvalidArchive(format!(
            "{what}: missing {tag:?} magic (formatVersion-3 objects start with an \
             8-byte magic prelude)"
        )));
    }
    if !(MIN_OBJECT_MAGIC_VERSION..=OBJECT_MAGIC_VERSION).contains(&bytes[4]) {
        return Err(Error::InvalidArchive(format!(
            "{what}: unsupported {tag} object version {} (this reader supports \
             {MIN_OBJECT_MAGIC_VERSION}..={OBJECT_MAGIC_VERSION})",
            bytes[4]
        )));
    }
    if bytes[5..OBJECT_MAGIC_LEN] != [0, 0, 0] {
        return Err(Error::InvalidArchive(format!(
            "{what}: reserved {tag} magic bytes must be zero"
        )));
    }
    Ok(&bytes[OBJECT_MAGIC_LEN..])
}

/// A v3 directory object's codec bytes: the post-magic slice. (`rootLength`
/// keeps meaning the root frame's
/// at-rest length — a paged root fetch is `bytes=0..(8+rootLength-1)` — so
/// all downstream paged math is unchanged once the magic is stripped.)
/// Public so out-of-band directory consumers (e.g. `stt-validate`'s direct
/// paged-structure re-check) apply the same version-aware unwrap.
pub fn directory_codec_bytes(bytes: &[u8], format_version: u32) -> Result<&[u8]> {
    if !(MIN_PACKED_FORMAT_VERSION..=PACKED_FORMAT_VERSION).contains(&format_version) {
        return Err(Error::InvalidArchive(format!(
            "unsupported packed formatVersion {format_version} \
             (this reader supports {MIN_PACKED_FORMAT_VERSION}..={PACKED_FORMAT_VERSION})"
        )));
    }
    strip_object_magic(bytes, DIRECTORY_MAGIC, "directory object")
}

/// `manifest.capabilities` entry: coordinate quantization (`stt:quant`
/// re-types the existing `geometry` column to fixed-point `Int32`).
pub const CAPABILITY_COORD_QUANT: &str = "coord-quant";
/// `manifest.capabilities` entry: numeric-attribute quantization (`stt:qa`
/// re-types existing property columns to fixed-point integers).
pub const CAPABILITY_ATTR_QUANT: &str = "attr-quant";
/// `manifest.capabilities` entry: point-elevation fold (a property folded into
/// POINT geometry z — the point leaf becomes 3 components instead of 2).
pub const CAPABILITY_ELEVATION_FOLD: &str = "elevation-fold";
/// `manifest.capabilities` entry: compact feature times (`TILE_META.st` /
/// `.et` re-type `start_time` to a `UInt32` offset from `t0`, and `end_time`
/// to a `UInt32` duration — or omit it entirely when every feature is
/// instantaneous). A reader that ignores the two keys would read millisecond
/// offsets as absolute Unix times, i.e. place every feature in January 1970.
pub const CAPABILITY_TIME_DELTA: &str = "time-delta";
/// `manifest.capabilities` entry: per-vertex value quantization
/// (`TILE_META.vq` re-types the `vertex_value` / `vertex_value_matrix` list
/// leaf from `Float32` to `UInt16` indices). A reader that ignores the key
/// would render raw 0..65534 indices as physical values — e.g. a −2..32 °C
/// sea-surface temperature ramp saturated everywhere.
pub const CAPABILITY_VERTEX_VALUE_QUANT: &str = "vertex-value-quant";
/// `manifest.capabilities` entry: PER-FEATURE triangle emission (TB-12). A
/// polygon layer's `triangles` column bakes indices only for the features a
/// renderer's own single-boundary earcut cannot reproduce (holes, multi-part)
/// and leaves every other feature's list EMPTY, expecting the reader to earcut
/// it at decode.
///
/// The pre-TB-12 readers all branch on whether the LAYER has a triangle column
/// and then trust each feature's slice verbatim, so against them an empty list
/// means "draw nothing": every single-ring polygon would silently VANISH. That
/// is the silent-misdecode class this registry exists to turn into a refusal at
/// open, which is why per-feature emission is a capability and not a quiet
/// byte saving.
pub const CAPABILITY_TRIANGLES_PARTIAL: &str = "triangles-partial";
/// `manifest.capabilities` entry: FEATURE-ANCHORED per-vertex times (TB-11
/// extension 2). `TILE_META.vtf` re-types the `vertex_time` list leaf to
/// `UInt16` deltas measured from **each feature's own `start_time`** instead of
/// a layer-wide origin.
///
/// A reader that knows only the layer-anchored `vt` form sees a `List<UInt16>`
/// with no origin/step pair. Depending on how forgiving it is, it either treats
/// the raw deltas as absolute Unix ms — putting every vertex in January 1970 —
/// or anchors them against an origin it invented. Both are silent, which is why
/// the form is gated rather than merely additive.
pub const CAPABILITY_VERTEX_TIME_FEATURE_ANCHOR: &str = "vertex-time-feature-anchor";
/// Every `manifest.capabilities` value this toolchain implements — the
/// required-to-understand feature registry (`docs/spec/stt-packed-format.md`
/// §3.1). Each capability RE-TYPES existing columns, so a reader that lacks it
/// wouldn't error downstream — it would silently misdecode (e.g. `Int32` grid
/// indices read as microscopic lon/lat degrees). A reader MUST therefore
/// refuse, at open, any dataset declaring a capability outside its own set.
pub const KNOWN_CAPABILITIES: &[&str] = &[
    CAPABILITY_COORD_QUANT,
    CAPABILITY_ATTR_QUANT,
    CAPABILITY_ELEVATION_FOLD,
    CAPABILITY_TIME_DELTA,
    CAPABILITY_VERTEX_VALUE_QUANT,
    CAPABILITY_TRIANGLES_PARTIAL,
    CAPABILITY_VERTEX_TIME_FEATURE_ANCHOR,
];

/// Default pack target size — 64 MiB. Well under the 512 MB CDN per-object cap,
/// with enough granularity for fine cache + parallel range reads.
pub const DEFAULT_PACK_TARGET_BYTES: u64 = 64 * 1024 * 1024;

/// CRC32C integrity tag for a compressed blob (mirrors the archive writer).
fn crc32c_tag(bytes: &[u8]) -> u32 {
    crc32c::crc32c(bytes)
}

/// blake3 content address, 128-bit → 32 lowercase hex chars.
fn blake3_128_hex(bytes: &[u8]) -> String {
    let hash = blake3::hash(bytes);
    // Take the first 16 bytes (128 bits) of the 256-bit digest.
    hash.as_bytes()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ----------------------------------------------------------------------------
// Manifest
// ----------------------------------------------------------------------------

/// At-rest encoding value for a zstd-compressed directory object.
pub const DIRECTORY_ENCODING_ZSTD: &str = "zstd";

/// `directory.layout` value for the paged container (root page + leaf pages,
/// each independently framed). Absent or `"single"` = the whole-load v6 object.
pub const DIRECTORY_LAYOUT_PAGED: &str = "paged";
/// `directory.layout` value for the single whole-load object (the default).
pub const DIRECTORY_LAYOUT_SINGLE: &str = "single";

/// Pointer to the encoded directory object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryRef {
    /// Object key, relative to the dataset root (e.g. `index/<hash>.sttd`).
    pub key: String,
    /// Directory object length in bytes (the at-rest object, i.e. the
    /// compressed length when `encoding` is set).
    pub length: u64,
    /// Directory codec version (`6` for the packed format). The leaf pages of a
    /// paged directory are this same v6 codec — `layout` (below), not this
    /// version, discriminates the container shape.
    #[serde(rename = "directoryVersion")]
    pub directory_version: u8,
    /// At-rest encoding of the directory object. `Some("zstd")` means the
    /// object bytes are a zstd frame wrapping the codec bytes; absent means
    /// raw codec bytes. For a
    /// paged directory it describes the framing of **each page** (root + every
    /// leaf), not one frame over the whole object. The content address and
    /// `length` always describe the at-rest bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// Container layout. `Some("paged")` = a root page + leaf pages;
    /// absent or `Some("single")` = the single whole-load object. Readers that
    /// don't know `"paged"` fail loudly (the root's first byte isn't a valid v6
    /// directory version), which is why readers ship before any paged dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
    /// At-rest byte length of the root page (a prefix of the object). Present
    /// iff `layout == "paged"`; the reader range-GETs `bytes=0-(8+rootLength-1)`
    /// for the root (the 8-byte object magic shifts it), then leaf ranges on
    /// demand.
    #[serde(
        default,
        rename = "rootLength",
        skip_serializing_if = "Option::is_none"
    )]
    pub root_length: Option<u64>,
    /// Number of leaf pages (informational / validation). Paged only.
    #[serde(default, rename = "pageCount", skip_serializing_if = "Option::is_none")]
    pub page_count: Option<u64>,
    /// Nominal entries-per-page used at build (informational). Paged only.
    #[serde(
        default,
        rename = "pageEntries",
        skip_serializing_if = "Option::is_none"
    )]
    pub page_entries: Option<u64>,
    /// Required-on-paged blake3-128 of the exact at-rest root frame (excluding
    /// the 8-byte `.sttd` object magic).
    #[serde(default, rename = "rootHash", skip_serializing_if = "Option::is_none")]
    pub root_hash: Option<String>,
    /// Required-on-paged blake3-128 hashes of the exact at-rest leaf frames, in
    /// page descriptor order.
    #[serde(
        default,
        rename = "pageHashes",
        skip_serializing_if = "Option::is_none"
    )]
    pub page_hashes: Option<Vec<String>>,
}

impl DirectoryRef {
    /// Is this a paged-container directory?
    pub fn is_paged(&self) -> bool {
        self.layout.as_deref() == Some(DIRECTORY_LAYOUT_PAGED)
    }
}

/// Pointer to one pack object. The position in `Manifest::packs` **is** the
/// `pack_id` the directory references.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackRef {
    /// Object key, relative to the dataset root (e.g. `packs/<hash>.sttp`).
    pub key: String,
    /// Pack object length in bytes.
    pub length: u64,
}

/// One schema-template entry of a formatVersion-3 manifest's `schemas` table:
/// the blake3-128 content hash of the raw template bytes (32 lowercase hex
/// chars — the hex form of the 16-byte reference v2 layer frames embed) and
/// the raw template bytes, base64-encoded (standard alphabet, padded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaTemplateRef {
    /// blake3-128 of the RAW (decoded) template bytes, lowercase hex.
    pub hash: String,
    /// The raw template bytes, base64.
    pub data: String,
}

/// Stable logical role of an independently addressable tile representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VariantKind {
    Raw,
    Summary,
    /// A REDUCED representation: fewer features than the base tier, derived by
    /// a declared `method` (DT-1). Explicitly not lossless.
    ///
    /// Needs no must-understand capability: variant 0 stays complete and a
    /// reader never fetches a variant it did not ask for (the summary-tier
    /// precedent).
    Reduced,
}

/// One entry in the manifest's required variant registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestVariant {
    /// Numeric id stored in every directory entry and carried by every TileId.
    pub id: u32,
    /// Semantic representation role.
    pub kind: VariantKind,
    /// Layer name used by this variant when it has one canonical layer.
    #[serde(default, rename = "layerName", skip_serializing_if = "Option::is_none")]
    pub layer_name: Option<String>,
    /// Reduction method — REQUIRED when `kind` is [`VariantKind::Reduced`],
    /// and absent otherwise (DT-1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<crate::metadata::ReductionMethod>,
    /// Method-specific parameters, required alongside `method`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// Decode + hash-validate a manifest's `schemas` table into the decode-side
/// [`TemplateRegistry`]. Every entry must base64-decode and blake3-hash to
/// its declared `hash` — the loud, dataset-level failure mode for corrupt
/// manifests, raised before a single tile is fetched.
fn build_template_registry(schemas: &[SchemaTemplateRef]) -> Result<TemplateRegistry> {
    use base64::Engine as _;
    let mut registry = TemplateRegistry::new();
    for (i, s) in schemas.iter().enumerate() {
        let data = base64::engine::general_purpose::STANDARD
            .decode(&s.data)
            .map_err(|e| {
                Error::InvalidArchive(format!(
                    "manifest schemas[{i}] ({}): base64 decode failed: {e}",
                    s.hash
                ))
            })?;
        let actual = blake3_128_hex(&data);
        if actual != s.hash {
            return Err(Error::InvalidArchive(format!(
                "manifest schemas[{i}]: template bytes hash to {actual}, declared {}",
                s.hash
            )));
        }
        registry.insert(data);
    }
    Ok(registry)
}

/// The packed-format `manifest.json` — the only mutable object per dataset.
///
/// Folds the metadata, directory pointer and pack table into one tiny JSON so a
/// cold reader needs exactly one manifest + one directory + N pack-range fetches
/// (no separate header or metadata object).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Always [`PACKED_FORMAT`] (`"stt-packed"`).
    pub format: String,
    /// Manifest schema version.
    #[serde(rename = "formatVersion")]
    pub format_version: u32,
    /// Required-to-understand feature declarations (spec §3.1). Each entry
    /// names a feature the writer used that RE-TYPES existing tile columns
    /// (registry: [`KNOWN_CAPABILITIES`]); a reader MUST refuse a dataset
    /// declaring a capability it does not implement. Empty when no such
    /// feature was used — omitted from the JSON so pre-capabilities builds
    /// stay byte-identical. Additive columns never need a capability.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// formatVersion ≥ 2: the dataset's Arrow schema **templates**, embedded
    /// directly in the manifest (no extra object class — every session
    /// already fetches the manifest). Each entry is a layer schema's IPC
    /// prefix, referenced from v2 layer frames by blake3-128 hash. Sorted by
    /// `hash` and deduped (byte-reproducible manifests); a reader validates
    /// `blake3(data) == hash` for every entry at open, so a corrupt manifest
    /// fails loudly, dataset-level, before any tile fetch. Absent/empty when
    /// every frame carries its schema inline (the key is then omitted).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schemas: Vec<SchemaTemplateRef>,
    /// Required registry of independently addressable tile representations.
    /// Every directory `variant_id` MUST resolve here.
    ///
    /// `#[serde(default)]` is deliberate and does NOT relax that requirement:
    /// the field is required by `formatVersion` 3, and the version gate rejects
    /// anything older. Without the default, a pre-v3 manifest died inside serde
    /// with `missing field 'variants'` — a decode error that says nothing about
    /// the real problem — *before* the version check could produce the message
    /// that actually helps ("unsupported formatVersion N"). Defaulting to empty
    /// lets the version gate speak first; a v3 manifest that genuinely omits
    /// `variants` still fails validation on the `variant_id` resolution below.
    #[serde(default)]
    pub variants: Vec<ManifestVariant>,
    /// Blob compression codec (always `"zstd"`, per-blob, no shared dict).
    pub compression: String,
    /// The concrete blob byte-ordering the writer resolved and laid down
    /// (`spatial` | `time-major` | `hilbert3` | `morton3`) — see
    /// [`crate::curve::BlobOrdering`]. Additive/optional: pre-2026-07 archives
    /// omit it (a reader infers the order from the `(pack_id, offset)` layout).
    /// Never `auto`/`measured` — those resolve to a concrete order at build.
    /// Omitted from the JSON when `None` so pre-field builds stay byte-identical.
    #[serde(
        default,
        rename = "blobOrdering",
        skip_serializing_if = "Option::is_none"
    )]
    pub blob_ordering: Option<String>,
    /// **TB-10 shared adaptive boundaries.** The dataset-wide candidate instants
    /// that `--adaptive-temporal` window keys were snapped down onto, ascending
    /// and deduplicated.
    ///
    /// Adaptive mode derives each tile's time key from its own cell's feature
    /// distribution, so before snapping a client had no way to ENUMERATE the
    /// keys a viewport would need — it could only discover them from the
    /// directory, which is what broke multi-cell prefetch. Publishing the
    /// candidate set restores enumeration: every window key is drawn from this
    /// list (bar the rare collision fallback, which keeps its exact timestamp).
    ///
    /// ADDITIVE, never a capability: a reader that ignores the key decodes every
    /// tile correctly and merely prefetches less well. Omitted from the JSON
    /// when empty, so non-adaptive builds stay byte-identical.
    #[serde(
        default,
        rename = "adaptiveBoundaries",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub adaptive_boundaries: Vec<i64>,
    /// **Co-versioning for [`Self::blob_ordering`]** (SH-3 / WM-2): the workload
    /// model the layout was chosen under, including the reader-mirroring
    /// coalescing gap the simulation priced at. See
    /// [`crate::metadata::OrderingWorkload`].
    ///
    /// `blob_ordering` alone says WHAT the layout is; it cannot say whether the
    /// layout is still optimal. Two things invalidate a simulated layout without
    /// moving a single archive byte — a re-fit of the query weights, and a change
    /// to the reader's coalescing gap — and both are recorded here so
    /// `stt-optimize order-audit` can flag drift instead of silently comparing
    /// against today's constants.
    ///
    /// Present on exactly the archives whose ordering was resolved by SIMULATION
    /// (`--blob-ordering measured` with enough tiles to simulate) and absent
    /// everywhere else; that presence/absence is itself the signal, because the
    /// ordering STRING only ever names the concrete winner
    /// (`"spatial" | "time-major" | …`) and so cannot distinguish a fitted layout
    /// from a declared one. Omitted from the JSON when `None`, so every
    /// non-measured build stays byte-identical to a pre-field build.
    ///
    /// ⚠️ **Mirrored, deliberately.** The identical object is also written to
    /// `metadata.ordering_workload` (see
    /// [`crate::metadata::Metadata::ordering_workload`]). This top-level key is
    /// canonical — it belongs beside the layout fact it co-versions, not inside
    /// the content-description block — but the shipped TS reader
    /// (`packages/core/src/archive.ts`, `manifestBuildAssumedGapBytes`) reads the
    /// `metadata` copy, so removing the mirror would silently disable the
    /// adaptive-coalesce co-versioning guard. Both copies are written from one
    /// value at one site in [`PackWriter::finalize`] and are asserted equal by
    /// `manifest_records_the_ordering_workload_at_both_pinned_keys`.
    /// **Removal trigger:** when the TS reader reads `orderingWorkload`, drop the
    /// `metadata` mirror.
    #[serde(
        default,
        rename = "orderingWorkload",
        skip_serializing_if = "Option::is_none"
    )]
    pub ordering_workload: Option<crate::metadata::OrderingWorkload>,
    /// Pointer to the encoded directory object.
    pub directory: DirectoryRef,
    /// Pack table. Index == `pack_id`.
    pub packs: Vec<PackRef>,
    /// The full `crate::metadata::Metadata` JSON, verbatim.
    pub metadata: Metadata,
}

impl Manifest {
    /// Parse a manifest from its JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| Error::InvalidArchive(format!("manifest JSON decode failed: {e}")))
    }

    /// Serialise the manifest to pretty JSON bytes.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self)
            .map_err(|e| Error::Other(format!("manifest JSON encode failed: {e}")))
    }
}

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

/// A tile buffered for deferred ordering + per-blob compression + pack-cutting.
struct PendingTile {
    z: u8,
    x: u32,
    y: u32,
    hilbert: u64,
    time_start: i64,
    time_end: i64,
    variant_id: u32,
    cover_t_min: Option<i64>,
    feature_count: u32,
    temporal_bucket_ms: Option<u64>,
    payload: PendingPayload,
}

/// Where a pending tile's uncompressed payload lives: in RAM (the legacy
/// behaviour, and always the first `memory_budget` bytes), or appended to the
/// writer's temp spill file as an `(offset, len)` record once the in-memory
/// budget is exhausted. Either way the ~100 B of per-tile metadata above stays
/// in RAM — only the payload bytes change medium, so the finalize sort/dedup/
/// cut logic (and therefore every output byte) is identical in both modes.
enum PendingPayload {
    Mem(Vec<u8>),
    /// `len` is u64 to match [`SpillFile`]'s u64 offsets — a narrower field
    /// here would silently truncate what the spill file faithfully stored.
    /// (Payloads over u32::MAX never get this far: [`check_payload_len`]
    /// rejects them at [`PackWriter::add_tile_full`].)
    Spilled {
        offset: u64,
        len: u64,
    },
}

impl PendingPayload {
    /// Uncompressed payload length in bytes.
    fn len(&self) -> usize {
        match self {
            PendingPayload::Mem(v) => v.len(),
            PendingPayload::Spilled { len, .. } => *len as usize,
        }
    }
}

/// Reject payloads the packed format cannot represent: the directory's
/// `uncompressed_size` is a u32 field, so a tile payload of 4 GiB or more
/// must be a loud, descriptive error on BOTH storage paths (in-RAM and
/// spilled) — never a silent length truncation that surfaces as corruption
/// at read time. Extracted so the bound is unit-testable without a 4 GiB
/// allocation.
fn check_payload_len(id: &TileId, len: u64) -> Result<()> {
    if len > u32::MAX as u64 {
        return Err(Error::Other(format!(
            "tile {id:?}: payload is {len} bytes, exceeding the directory's u32 \
             uncompressed_size field (the packed format cannot represent tiles of \
             4 GiB or larger) — split the tile"
        )));
    }
    Ok(())
}

/// Temp file holding spilled tile payloads, created lazily in the OUTPUT
/// directory (same filesystem as the final objects — no /tmp capacity or
/// cross-device concerns). Append-only during the add phase; random-access
/// reads during finalize's chunked compression pass. Removed on drop, which
/// covers success, error and abandoned-writer paths alike.
struct SpillFile {
    file: File,
    path: PathBuf,
    /// Bytes appended so far == the next record's offset.
    len: u64,
}

impl SpillFile {
    fn create(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = dir.join(format!(".spill-{}-{nanos:x}", std::process::id()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self { file, path, len: 0 })
    }

    /// Append one payload, returning its record offset.
    fn append(&mut self, payload: &[u8]) -> Result<u64> {
        use std::io::{Seek, SeekFrom};
        let offset = self.len;
        let mut f = &self.file;
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(payload)?;
        self.len += payload.len() as u64;
        Ok(offset)
    }

    /// Read one payload record back. Only called from finalize's sequential
    /// materialisation pass (after all appends), so sharing the file cursor
    /// is safe.
    fn read(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = &self.file;
        f.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Writer for the multi-object packed format.
///
/// Consumes an `(id, payload)` stream like `ArchiveWriter`,
/// buffering every tile until [`finalize`](Self::finalize). At finalize it
/// reuses the buffered-writer pipeline — resolve [`BlobOrdering::Auto`], sort by
/// the space-time curve key, per-blob zstd (NO shared dictionary, so the fzstd
/// TS reader can decode), byte-identical dedup via blake3 — then cuts the
/// ordered/deduped blob stream into packs of `≤ pack_target_bytes`, never
/// splitting a blob, and writes content-addressed `packs/*.sttp` +
/// `index/*.sttd` + `manifest.json`.
pub struct PackWriter {
    out_dir: PathBuf,
    ordering: BlobOrdering,
    pack_target_bytes: u64,
    pending: Vec<PendingTile>,
    /// `Some(k)` → emit a **paged** directory (root + leaf pages of ≤ `k`
    /// entries; see [`crate::directory_page`]); `None` → the single whole-load
    /// v6 directory (the default). Set via [`with_paging`](Self::with_paging).
    page_entries: Option<usize>,
    /// Minimum directory entry count before paging activates. `0` means every
    /// non-empty directory is paged (the explicit `with_paging` behaviour).
    paging_min_entries: usize,
    /// zstd level for per-blob + directory compression. Defaults to
    /// [`compression::ZSTD_LEVEL`]; a publish build raises it via
    /// [`with_zstd_level`](Self::with_zstd_level). Decode is level-independent,
    /// so this only trades build CPU for smaller on-the-wire bytes.
    zstd_level: i32,
    /// `manifest.capabilities` — required-to-understand feature declarations.
    /// Set via [`with_capabilities`](Self::with_capabilities); empty (the
    /// default) omits the key from the manifest.
    capabilities: Vec<String>,
    /// `manifest.adaptiveBoundaries` — TB-10's shared candidate instants. Set
    /// via [`with_adaptive_boundaries`](Self::with_adaptive_boundaries); empty
    /// (the default) omits the key.
    adaptive_boundaries: Vec<i64>,
    /// Opt into the **measured** ordering picker: resolve the concrete on-disk
    /// order by simulating per-ordering range-read cost ([`crate::ordering_sim`])
    /// rather than the `auto` cardinality heuristic. Set via
    /// [`with_measured_ordering`](Self::with_measured_ordering); `false` (the
    /// default) keeps the `ordering` argument's behaviour.
    measured_ordering: bool,
    /// Which query weighting the measured picker runs under. `Derived` (the
    /// default) reads the dataset's `layer_hint` + bucket count; `Legacy`
    /// forces the pre-M4 two-query weights. Inert unless
    /// [`measured_ordering`](Self::with_measured_ordering) is on.
    ordering_workload: crate::ordering_sim::OrderingWorkloadMode,
    /// Schema-template sink: the encoder records each layer's stripped
    /// schema prefix here (wired automatically by
    /// [`encoder_config`](Self::encoder_config) → [`crate::arrow_tile::
    /// EncoderConfig::template_collector`]); finalize publishes the snapshot
    /// as `manifest.schemas`.
    templates: Arc<TemplateCollector>,
    /// Quantization / vector-grouping / elevation-fold settings the payloads
    /// written to THIS writer are encoded with (see
    /// [`with_encoder_config`](Self::with_encoder_config)). The frame version
    /// and template sink are NOT taken from here — `encoder_config()` always
    /// overrides them with the writer's own, so a dataset's frames and its
    /// manifest can never disagree — readers reject a mixed-version dataset
    /// outright. Default = the plain
    /// encoder, byte-identical to a build that sets nothing.
    encoder: crate::arrow_tile::EncoderConfig,
    /// In-memory budget (bytes) for buffered UNCOMPRESSED tile payloads.
    /// `0` (the default) = unlimited — every payload stays in RAM until
    /// finalize. Non-zero: once buffered payload bytes
    /// would exceed the budget, further payloads are appended to a temp spill
    /// file in `out_dir` and read back record-by-record during finalize.
    /// A pure memory-behaviour lever: output bytes are identical either way.
    /// Set via [`with_memory_budget`](Self::with_memory_budget).
    memory_budget: u64,
    /// Total bytes of payloads currently held in RAM (`PendingPayload::Mem`).
    pending_payload_bytes: u64,
    /// Total UNCOMPRESSED payload bytes ever handed to
    /// [`add_tile_full`](Self::add_tile_full), regardless of where they were
    /// stored. Unlike [`pending_payload_bytes`](Self::pending_payload_bytes)
    /// this is a monotonic total, not a residency gauge, so it is a stable
    /// answer to "how big is this dataset" at any point before finalize.
    /// Read-only, via [`payload_bytes`](Self::payload_bytes); it influences no
    /// output byte.
    total_payload_bytes: u64,
    /// Lazily created once the budget is exceeded; `None` = nothing spilled.
    spill: Option<SpillFile>,
}

impl PackWriter {
    fn merge_capabilities(&mut self, capabilities: impl IntoIterator<Item = String>) {
        self.capabilities.extend(capabilities);
        self.capabilities.sort_unstable();
        self.capabilities.dedup();
    }

    /// Create a packed-format writer targeting `out_dir`.
    ///
    /// `out_dir` (and its `index/` + `packs/` subdirs) are created on
    /// [`finalize`](Self::finalize). `ordering` controls the on-disk blob byte
    /// order ([`BlobOrdering::Auto`] resolves per-dataset at finalize);
    /// `pack_target_bytes` is the soft per-pack size cap — a single blob larger
    /// than the target gets its own pack rather than being split.
    pub fn create<P: AsRef<Path>>(
        out_dir: P,
        ordering: BlobOrdering,
        pack_target_bytes: u64,
    ) -> Result<Self> {
        Ok(Self {
            out_dir: out_dir.as_ref().to_path_buf(),
            ordering,
            pack_target_bytes: pack_target_bytes.max(1),
            pending: Vec::new(),
            page_entries: None,
            paging_min_entries: 0,
            zstd_level: compression::ZSTD_LEVEL,
            capabilities: Vec::new(),
            adaptive_boundaries: Vec::new(),
            measured_ordering: false,
            ordering_workload: crate::ordering_sim::OrderingWorkloadMode::default(),
            templates: Arc::new(TemplateCollector::new()),
            encoder: crate::arrow_tile::EncoderConfig::default(),
            memory_budget: 0,
            pending_payload_bytes: 0,
            total_payload_bytes: 0,
            spill: None,
        })
    }

    /// Cap the UNCOMPRESSED tile-payload bytes buffered in RAM between
    /// [`add_tile_full`](Self::add_tile_full) and [`finalize`](Self::finalize).
    /// Once the budget is reached, further payloads are appended to a temp
    /// spill file inside `out_dir` (same rename-safe filesystem as the final
    /// objects; removed on success, error and drop alike) while the ~100 B of
    /// per-tile directory metadata stays in RAM. Finalize reads spilled
    /// payloads back record-by-record inside its existing chunked
    /// parallel-compression pass, so the sort keys, dedup and pack-cut logic —
    /// and therefore every output byte — are identical to an unbounded build.
    /// `0` (the default) = unlimited / legacy all-in-RAM behaviour.
    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.memory_budget = bytes;
        self
    }

    /// The configured in-RAM payload budget in bytes (`0` = unlimited).
    /// Encode drivers (the stt-build tiler) read this so the ENCODED-but-not-
    /// yet-added payloads of a parallel encode batch honor the same cap as
    /// the writer's own buffered payloads.
    pub fn memory_budget(&self) -> u64 {
        self.memory_budget
    }

    /// Seed the writer's schema-template collector from an existing dataset's
    /// registry — the VERBATIM-repack path (pack-cover, repair tools). Copied
    /// payloads reference templates by 16-byte hash but are never
    /// re-encoded, so nothing records their templates; without seeding, the
    /// repacked `manifest.schemas` comes out empty and every tile read fails
    /// template resolution. Recording is content-addressed (sorted + deduped
    /// at finalize), so seeding order is irrelevant and templates the subset
    /// no longer references ride along harmlessly.
    pub fn with_seeded_templates(self, templates: &TemplateRegistry) -> Self {
        for (_, bytes) in templates.iter() {
            self.templates.record(bytes);
        }
        self
    }

    /// The manifest `formatVersion` this writer emits — always
    /// [`PACKED_FORMAT_VERSION`]. The layer-frame axis is intentionally
    /// independent and is pinned by [`encoder_config`](Self::encoder_config).
    ///
    /// There is no setter. The transitional format was withdrawn in
    /// `e084ccd`, and the writer emitting a version the reader refuses is not
    /// a capability worth keeping a checked channel open for.
    pub fn format_version(&self) -> u32 {
        PACKED_FORMAT_VERSION
    }

    /// The writer's schema-template sink. Normally taken care of by
    /// [`encoder_config`](Self::encoder_config), which installs it on
    /// [`crate::arrow_tile::EncoderConfig::template_collector`] so every
    /// template a layer frame references ends up in `manifest.schemas` at
    /// [`finalize`](Self::finalize).
    pub fn template_collector(&self) -> Arc<TemplateCollector> {
        Arc::clone(&self.templates)
    }

    /// Declare the encoder settings (coordinate + attribute quantization,
    /// vector grouping, point-elevation fold, vertex-time precision) the tile
    /// payloads handed to this writer are produced with, so an encode driver
    /// can recover them from the writer alone via
    /// [`encoder_config`](Self::encoder_config) instead of reaching for
    /// process-wide state. The `format_version` / `template_collector` fields
    /// of `cfg` are ignored — the writer's own always win.
    ///
    /// NOTE on `point_elevation_column`: it is a supported encoder setting, but
    /// folding elevation into the geometry's 3rd coordinate makes the tile
    /// unreadable to anything that walks point coords at the 2-wide `xy`
    /// stride. Prefer shipping elevation as a numeric COLUMN and lifting it
    /// client-side via the layer's `elevationProperty`.
    pub fn with_encoder_config(mut self, cfg: crate::arrow_tile::EncoderConfig) -> Self {
        let mut required = Vec::new();
        if cfg.quantize_coords_m.is_some() {
            required.push(CAPABILITY_COORD_QUANT.to_string());
        }
        if !cfg.quantize_attrs.is_empty() || cfg.quantize_attrs_auto {
            required.push(CAPABILITY_ATTR_QUANT.to_string());
        }
        if !cfg.point_elevation_column.is_empty() {
            required.push(CAPABILITY_ELEVATION_FOLD.to_string());
        }
        if cfg.compact_times {
            required.push(CAPABILITY_TIME_DELTA.to_string());
        }
        if cfg.quantize_vertex_values {
            required.push(CAPABILITY_VERTEX_VALUE_QUANT.to_string());
        }
        self.merge_capabilities(required);
        self.encoder = cfg;
        self
    }

    /// The config a payload for THIS writer must be encoded with: the settings
    /// from [`with_encoder_config`](Self::with_encoder_config), with the frame
    /// version and template sink forced to the writer's own.
    ///
    /// That override pins the independent layer-frame version and template
    /// registry in one place. A template-referencing frame whose templates
    /// were never recorded fails template resolution at read time.
    pub fn encoder_config(&self) -> crate::arrow_tile::EncoderConfig {
        crate::arrow_tile::EncoderConfig {
            format_version: crate::arrow_tile::LAYER_FRAME_VERSION,
            template_collector: Some(self.template_collector()),
            ..self.encoder.clone()
        }
    }

    /// Opt into a **paged** directory: the `.sttd` becomes a root page + leaf
    /// pages of ≤ `page_entries` entries each, so a cold reader fetches only the
    /// leaves its viewport/time-window touches. `None` (the default) emits the
    /// single whole-load v6 directory — byte-identical output to a build that
    /// never enables paging. `Some(0)` is clamped to 1.
    pub fn with_paging(mut self, page_entries: Option<usize>) -> Self {
        self.page_entries = page_entries.map(|k| k.max(1));
        self.paging_min_entries = 0;
        self
    }

    /// Enable paging only when the directory is large enough to benefit.
    /// Sparse archives stay a smaller single-frame index; large archives keep
    /// viewport-proportional leaf loading.
    pub fn with_adaptive_paging(mut self, page_entries: usize, min_entries: usize) -> Self {
        self.page_entries = Some(page_entries.max(1));
        self.paging_min_entries = min_entries.max(1);
        self
    }

    /// Set the zstd compression level for tile blobs and the directory.
    ///
    /// The packed format is write-once / serve-many, so the higher build CPU of
    /// a level like 19 is paid once while the smaller bytes are paid on every
    /// fetch (measured −10..19% vs the level-3 default; decode is unaffected).
    /// Clamped to zstd's valid 1..=22 range. At the default
    /// ([`compression::ZSTD_LEVEL`]) the bytes match a build that never calls
    /// this, so an already-published archive keeps its content addresses.
    pub fn with_zstd_level(mut self, level: i32) -> Self {
        self.zstd_level = level.clamp(1, compression::ZSTD_LEVEL_MAX);
        self
    }

    /// Declare the required-to-understand features this build used
    /// (`manifest.capabilities`, spec §3.1) — e.g. [`CAPABILITY_COORD_QUANT`]
    /// when coordinate quantization re-types the geometry column. Canonicalized
    /// (sorted + deduped) so the manifest bytes never depend on call order.
    /// Empty (the default) omits the key entirely, so the manifest bytes match
    /// a build that declares nothing.
    pub fn with_capabilities(mut self, mut capabilities: Vec<String>) -> Self {
        capabilities.sort_unstable();
        capabilities.dedup();
        self.merge_capabilities(capabilities);
        self
    }

    /// Publish TB-10's shared adaptive boundary set in the manifest.
    ///
    /// ADDITIVE: a reader that ignores `adaptiveBoundaries` still decodes every
    /// tile — it just cannot enumerate the adaptive key set ahead of a fetch,
    /// which is the prefetch problem the field exists to solve. Sorted and
    /// deduped here so the manifest bytes never depend on caller order.
    pub fn with_adaptive_boundaries(mut self, mut boundaries: Vec<i64>) -> Self {
        boundaries.sort_unstable();
        boundaries.dedup();
        self.adaptive_boundaries = boundaries;
        self
    }

    /// Declare one required-to-understand capability AFTER construction.
    ///
    /// The builder-style [`with_capabilities`](Self::with_capabilities) covers
    /// everything derivable from settings before any tile is encoded. Some
    /// capabilities are instead OBSERVED while encoding — TB-12's
    /// [`CAPABILITY_TRIANGLES_PARTIAL`] is only owed when a layer actually mixes
    /// empty and baked triangle lists — and those arrive after the writer
    /// already exists. Declaring one is idempotent and order-independent (the
    /// set is sorted and deduped), so it stays deterministic no matter which
    /// worker observed it first. Must be called before
    /// [`finalize`](Self::finalize), which is when the manifest is written.
    pub fn declare_capability(&mut self, capability: &str) {
        self.merge_capabilities([capability.to_string()]);
    }

    /// Opt into the **measured** ordering picker (`--blob-ordering measured`):
    /// at finalize, resolve the concrete on-disk blob order by simulating
    /// per-ordering range-read cost over the native-tier tiles
    /// ([`crate::ordering_sim`]) instead of the `auto` cardinality heuristic.
    /// When `on`, the `ordering` passed to [`create`](Self::create) is ignored.
    /// `false` (the default) emits the bytes a build that never calls this does.
    pub fn with_measured_ordering(mut self, on: bool) -> Self {
        self.measured_ordering = on;
        self
    }

    /// Choose the query weighting the **measured** picker ranks under
    /// (`stt-build --ordering-workload`).
    ///
    /// [`Derived`](crate::ordering_sim::OrderingWorkloadMode::Derived) — the
    /// default — derives per-dataset weights from the metadata's `layer_hint`
    /// and the native tier's distinct bucket count
    /// ([`crate::ordering_sim::workload_weights`]), so a playback-class dataset
    /// is not laid out for an access pattern it never issues.
    /// [`Legacy`](crate::ordering_sim::OrderingWorkloadMode::Legacy) is the
    /// documented escape hatch: it forces the pre-M4 scrub+pan weighting, which
    /// is what every archive built before the workload model used. Inert unless
    /// [`with_measured_ordering`](Self::with_measured_ordering) is on.
    pub fn with_ordering_workload(
        mut self,
        mode: crate::ordering_sim::OrderingWorkloadMode,
    ) -> Self {
        self.ordering_workload = mode;
        self
    }

    /// Add a tile carrying the full directory metadata. Same shape as
    /// `ArchiveWriter::add_tile_full`: `cover_t_min` is the
    /// tight lower covering bound (`None` to omit), `temporal_bucket_ms` tags the
    /// directory entry with the temporal bucket size the tile represents. The
    /// `payload` is the uncompressed tile frame.
    #[allow(clippy::too_many_arguments)]
    pub fn add_tile_full(
        &mut self,
        id: &TileId,
        time_start: i64,
        time_end: i64,
        cover_t_min: Option<i64>,
        feature_count: u32,
        temporal_bucket_ms: Option<u64>,
        payload: &[u8],
    ) -> Result<()> {
        check_payload_len(id, payload.len() as u64)?;
        // Monotonic dataset-size total (see `total_payload_bytes`). Counted
        // before the storage-medium branch so a spilled payload counts the same
        // as an in-RAM one, and counted pre-dedup/pre-compression because the
        // quantity it answers for is DECODED resident bytes.
        self.total_payload_bytes = self
            .total_payload_bytes
            .saturating_add(payload.len() as u64);
        // Frame/manifest version coherence: readers hard-reject mixed-version
        // datasets, so a layer frame of the OTHER version would brick the
        // output on first read. Refuse it at add time, naming the tile —
        // both directions.
        if !crate::arrow_tile::is_frame_v2(payload) {
            return Err(Error::Other(format!(
                "tile {id:?}: pre-v2 layer frame fed to a \
                 formatVersion-{PACKED_FORMAT_VERSION} writer (the transitional \
                 frame shape was withdrawn; encode payloads with \
                 PackWriter::encoder_config)"
            )));
        }
        // Payload storage medium: RAM until the (optional) memory budget is
        // exhausted, then the temp spill file. Metadata always stays in RAM.
        let payload = if self.memory_budget > 0
            && self.pending_payload_bytes + payload.len() as u64 > self.memory_budget
        {
            let spill = match &mut self.spill {
                Some(s) => s,
                None => self.spill.insert(SpillFile::create(&self.out_dir)?),
            };
            let offset = spill.append(payload)?;
            PendingPayload::Spilled {
                offset,
                len: payload.len() as u64,
            }
        } else {
            self.pending_payload_bytes += payload.len() as u64;
            PendingPayload::Mem(payload.to_vec())
        };
        self.pending.push(PendingTile {
            z: id.z,
            x: id.x,
            y: id.y,
            hilbert: id.hilbert_index(),
            time_start,
            time_end,
            variant_id: id.variant_id,
            cover_t_min,
            feature_count,
            temporal_bucket_ms,
            payload,
        });
        Ok(())
    }

    /// Number of tiles buffered so far.
    pub fn tile_count(&self) -> usize {
        self.pending.len()
    }

    /// Total UNCOMPRESSED tile-payload bytes added so far — the sum of every
    /// `payload.len()` handed to [`add_tile_full`](Self::add_tile_full),
    /// counted before dedup and before per-blob zstd.
    ///
    /// This is the "how many bytes does a client hold if it decodes everything"
    /// total, which is what the derived playback-window hint (BH-10) prices a
    /// memory budget against — not the on-the-wire compressed size. Purely
    /// observational: reading it cannot change an output byte, and a build that
    /// never reads it is byte-identical.
    pub fn payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }

    /// Finalise: order + dedup + per-blob zstd, cut packs, write
    /// `packs/*.sttp` + `index/*.sttd` + `manifest.json` into `out_dir`.
    ///
    /// Mirrors `ArchiveWriter::finalize_buffered`: the same time-bucket calc,
    /// [`BlobOrdering::Auto`] → [`BlobOrdering::choose`] resolution, space-time
    /// sort key, per-blob `compress_zstd_with_dict(_, None)` (no shared dict),
    /// and blake3 byte-identical dedup. THEN the ordered, deduped blob stream is
    /// cut into packs.
    pub fn finalize(self, metadata: &Metadata) -> Result<Manifest> {
        let PackWriter {
            out_dir,
            ordering,
            pack_target_bytes,
            mut pending,
            page_entries,
            paging_min_entries,
            zstd_level,
            capabilities,
            adaptive_boundaries,
            measured_ordering,
            ordering_workload,
            templates,
            // Encode-time only — finalize just stores what was handed to it.
            encoder: _,
            memory_budget,
            pending_payload_bytes: _,
            total_payload_bytes: _,
            spill,
        } = self;
        if pending.is_empty() {
            return Err(Error::Other(
                "cannot finalize a packed dataset with no tiles (the manifest requires at least one pack)"
                    .into(),
            ));
        }
        // (formatVersion-3 objects carry an 8-byte magic prelude and blob
        // offsets are object-absolute — the first blob of a pack sits at
        // offset 8, and content addresses cover the prelude too;
        // `PackStreamWriter` below owns that math.)

        // --- Blob ordering (identical to finalize_buffered) ---------------
        let base_bucket = metadata.temporal_bucket_ms.max(1) as i64;
        let tb = |p: &PendingTile| {
            let b = p
                .temporal_bucket_ms
                .map(|v| v as i64)
                .unwrap_or(base_bucket)
                .max(1);
            p.time_start.div_euclid(b)
        };
        let (tb_min, tb_max) = pending.iter().fold((i64::MAX, i64::MIN), |(lo, hi), p| {
            let t = tb(p);
            (lo.min(t), hi.max(t))
        });
        let tb_span = if pending.is_empty() {
            0
        } else {
            tb_max - tb_min
        };
        // Occupied spatial extent over the native (max-zoom) tier only — coarse
        // LOD tiers have artificially small x/y ranges. Measuring the OCCUPIED
        // bbox (not the raw 2^zoom grid) makes the space axis symmetric with the
        // time axis; the raw grid overstates space for sparse data and pushes
        // `choose` to the wrong ordering.
        let native_z = pending.iter().map(|p| p.z).max().unwrap_or(0);
        let (mut x_min, mut x_max, mut y_min, mut y_max) = (u32::MAX, 0u32, u32::MAX, 0u32);
        for p in pending.iter().filter(|p| p.z == native_z) {
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
        }
        let x_span = x_max.saturating_sub(x_min) as u64;
        let y_span = y_max.saturating_sub(y_min) as u64;
        // Resolve the concrete on-disk ordering. `measured` simulates per-ordering
        // range-read cost across the native tiles; otherwise `auto` uses the
        // (occupied-extent) cardinality heuristic and an explicit order passes
        // through.
        // `auto`'s cardinality heuristic, also the documented fallback the
        // measured picker degrades to when there is nothing worth simulating.
        let auto_choice = || {
            let space_bits = crate::curve::bits_for(x_span.max(y_span) + 1);
            let time_bits = crate::curve::bits_for((tb_span.max(0) + 1) as u64);
            BlobOrdering::choose(space_bits, time_bits)
        };
        // Set only when the ordering was actually resolved by simulation, so it
        // is exactly the archives whose layout the workload model explains.
        let mut resolved_workload: Option<crate::metadata::OrderingWorkload> = None;
        let ordering = if measured_ordering {
            let samples: Vec<crate::ordering_sim::TileSample> = pending
                .iter()
                .filter(|p| p.z == native_z)
                .map(|p| crate::ordering_sim::TileSample {
                    z: p.z,
                    x: p.x,
                    y: p.y,
                    hilbert: p.hilbert,
                    time_start: p.time_start,
                    tb: tb(p),
                    len: p.payload.len() as u64,
                })
                .collect();
            if samples.len() < crate::ordering_sim::MIN_TILES_TO_SIMULATE {
                // Below the floor the canonical band degenerates to one or two
                // cells and the ranking is decided by rounding, so `measured`
                // falls back to `auto` and records nothing. Same floor the
                // pre-build layout advisor gives up at.
                auto_choice()
            } else {
                // Per-dataset weights from build inputs the pipeline already
                // computed: the dominant layer kind and the native tier's
                // distinct bucket count. `Legacy` mode pins the pre-M4
                // two-query weighting instead.
                let distinct_buckets = {
                    let mut tbs: Vec<i64> = samples.iter().map(|s| s.tb).collect();
                    tbs.sort_unstable();
                    tbs.dedup();
                    tbs.len()
                };
                let weights = match ordering_workload {
                    crate::ordering_sim::OrderingWorkloadMode::Derived => {
                        crate::ordering_sim::workload_weights(
                            metadata
                                .style_hints
                                .as_ref()
                                .and_then(|h| h.layer_hint.as_deref()),
                            distinct_buckets,
                        )
                    }
                    crate::ordering_sim::OrderingWorkloadMode::Legacy => {
                        crate::ordering_sim::LEGACY_WEIGHTS
                    }
                };
                // Pass the archive's real pack target so the simulated per-pack
                // coalescing matches the packs this build will actually cut. (Blob
                // weights here are uncompressed payload lengths vs the writer's
                // compressed cut, so boundaries land a little early — but uniformly
                // across all orderings, so the RELATIVE ranking is preserved.)
                let opts = crate::ordering_sim::SimOptions {
                    pack_bytes: pack_target_bytes,
                    weights,
                    ..crate::ordering_sim::SimOptions::default()
                };
                resolved_workload = Some(crate::metadata::OrderingWorkload {
                    scrub: weights.scrub,
                    pan: weights.pan,
                    playback: weights.playback,
                    playback_window_buckets: opts.playback_window_buckets,
                    runway_multiplier: opts.runway_multiplier,
                    coalesce_gap_bytes: opts.coalesce_gap_bytes,
                });
                crate::ordering_sim::measured_ordering(&samples, opts)
            }
        } else {
            match ordering {
                BlobOrdering::Auto => auto_choice(),
                other => other,
            }
        };
        // The curve key alone is not total (base and temporal-LOD tiles of one
        // cell tie; the 21-bit cube cap can collide), and `pending` arrives in
        // whatever order the (possibly parallel) tiler produced — so a total
        // tiebreak makes the blob byte order, and therefore every content
        // address, reproducible across identical rebuilds. Immutable-pack CDN
        // caching depends on that: a rebuild of unchanged data must re-derive
        // the same pack names.
        pending.sort_by_key(|p| {
            (
                crate::curve::space_time_key(
                    ordering,
                    p.z,
                    p.x,
                    p.y,
                    p.hilbert,
                    p.time_start,
                    tb(p),
                    tb_min,
                    tb_span,
                ),
                p.z,
                p.x,
                p.y,
                p.time_start,
                p.temporal_bucket_ms,
            )
        });

        // Output directories exist BEFORE the compress/dedup pass: the pack
        // phase below streams each pack to a temp file in `packs/` as blobs
        // are deduped (write_atomic gives the same rename discipline for the
        // directory + manifest afterwards).
        fs::create_dir_all(&out_dir)?;
        let index_dir = out_dir.join("index");
        let packs_dir = out_dir.join("packs");
        fs::create_dir_all(&index_dir)?;
        fs::create_dir_all(&packs_dir)?;

        // --- Per-blob zstd + byte-identical dedup + streaming pack cut ----
        // Each pending tile is compressed (NO shared dictionary, so the fzstd TS
        // reader can decode). Byte-identical compressed blobs collapse to a
        // single physical blob. Packs are cut greedily in first-seen (curve)
        // order, so each NEW unique blob streams straight into the current
        // pack's temp file via `PackStreamWriter` — the archive's compressed
        // bytes are never all resident, regardless of `memory_budget`. Only
        // ~24 B of metadata per unique blob stays in RAM; placement is
        // assigned at first sight, so a tile shared across time buckets still
        // resolves to one (pack, offset).
        struct Blob {
            /// Compressed length (the directory's u32 `length` field).
            length: u32,
            uncompressed_size: u32,
            crc: u32,
        }
        /// Where a unique blob landed: `(pack_id, object-absolute offset)`.
        struct Placement {
            pack_id: u32,
            offset: u64,
        }
        let mut stream = PackStreamWriter::new(&out_dir, &packs_dir, pack_target_bytes);
        let mut blobs: Vec<Blob> = Vec::new();
        let mut placements: Vec<Placement> = Vec::new();
        // blake3(compressed) → blob index in `blobs`.
        let mut blob_dedup: HashMap<[u8; 32], usize> = HashMap::new();
        // Per pending tile (in sorted order): which blob it references.
        let mut tile_blob: Vec<usize> = Vec::with_capacity(pending.len());
        // Compress in parallel, dedup sequentially. zstd at a high level
        // (`--zstd-level`) is the build-time bottleneck and is embarrassingly
        // parallel per blob; the dedup/index assignment that follows stays
        // strictly sequential over the sorted order, so the output is
        // byte-identical to a single-threaded build. Chunking caps peak memory
        // at ~CHUNK compressed blobs (a whole-dataset parallel pass would hold
        // every pre-dedup blob at once — pathological on dedup-heavy datasets).
        //
        // Chunk BOUNDARIES can never change output bytes (compression is
        // per-blob; the dedup/index pass runs strictly sequentially across
        // boundaries) — they only bound how many payloads are resident at
        // once. Without a spill file the chunks are the legacy fixed 8192;
        // with one, a chunk is additionally capped at ~memory_budget bytes of
        // payload so the spilled read-back honours the same budget that
        // triggered spilling.
        const COMPRESS_CHUNK: usize = 8192;
        let chunk_byte_cap: u64 = if spill.is_some() && memory_budget > 0 {
            memory_budget
        } else {
            u64::MAX
        };
        let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
        {
            let mut start = 0usize;
            let mut bytes = 0u64;
            for (i, p) in pending.iter().enumerate() {
                let len = p.payload.len() as u64;
                if i > start && (i - start >= COMPRESS_CHUNK || bytes + len > chunk_byte_cap) {
                    chunk_ranges.push((start, i));
                    start = i;
                    bytes = 0;
                }
                bytes += len;
            }
            if start < pending.len() {
                chunk_ranges.push((start, pending.len()));
            }
        }
        for (chunk_start, chunk_end) in chunk_ranges {
            let chunk = &pending[chunk_start..chunk_end];
            // Materialise this chunk's spilled payloads (sequential reads
            // by (offset, len) record; in-memory payloads borrow in place).
            let materialized: Vec<Option<Vec<u8>>> = chunk
                .iter()
                .map(|p| match &p.payload {
                    PendingPayload::Mem(_) => Ok(None),
                    PendingPayload::Spilled { offset, len } => {
                        let s = spill
                            .as_ref()
                            .expect("spilled payload without a spill file");
                        s.read(*offset, *len as usize).map(Some)
                    }
                })
                .collect::<Result<_>>()?;
            let compressed_chunk: Vec<Vec<u8>> = chunk
                .par_iter()
                .zip(materialized.par_iter())
                .map(|(p, m)| {
                    let payload: &[u8] = match (&p.payload, m) {
                        (PendingPayload::Mem(v), _) => v,
                        (PendingPayload::Spilled { .. }, Some(v)) => v,
                        (PendingPayload::Spilled { .. }, None) => {
                            unreachable!("spilled payload was not materialised")
                        }
                    };
                    compression::compress_zstd_with_dict_level(payload, None, zstd_level)
                })
                .collect::<Result<Vec<_>>>()?;
            for (p, compressed) in chunk.iter().zip(compressed_chunk) {
                let key = *blake3::hash(&compressed).as_bytes();
                let idx = if let Some(&i) = blob_dedup.get(&key) {
                    i
                } else {
                    let i = blobs.len();
                    let crc = crc32c_tag(&compressed);
                    // Blob offsets are object-absolute: the magic prelude
                    // occupies [0, 8), so blobs start at 8 and readers slice
                    // `[offset..offset+length]` off the whole object unchanged.
                    let (pack_id, offset) = stream.append(&compressed)?;
                    let compressed_len = u32::try_from(compressed.len()).map_err(|_| {
                        Error::Other(format!(
                            "compressed tile blob is {} bytes, exceeding the directory's u32 length field",
                            compressed.len()
                        ))
                    })?;
                    let uncompressed_size =
                        u32::try_from(p.payload.len()).map_err(|_| {
                            Error::Other(format!(
                                "tile payload is {} bytes, exceeding the directory's u32 uncompressed_size field",
                                p.payload.len()
                            ))
                        })?;
                    blobs.push(Blob {
                        length: compressed_len,
                        uncompressed_size,
                        crc,
                    });
                    placements.push(Placement { pack_id, offset });
                    blob_dedup.insert(key, i);
                    i
                };
                tile_blob.push(idx);
            }
        }
        // Every payload has been compressed; delete the spill file now (via
        // Drop) so peak disk usage isn't spill + packs longer than necessary.
        drop(spill);

        // Seal the trailing pack: the pack table is complete (index ==
        // pack_id, matching every placement handed out above).
        let pack_refs: Vec<PackRef> = stream.finish()?;

        // --- Build directory entries ------------------------------------
        let mut entries: Vec<TileEntry> = Vec::with_capacity(pending.len());
        for (p, &bi) in pending.iter().zip(tile_blob.iter()) {
            let blob = &blobs[bi];
            let pl = &placements[bi];
            entries.push(TileEntry {
                zoom: p.z,
                x: p.x,
                y: p.y,
                time_start: p.time_start,
                time_end: p.time_end,
                variant_id: p.variant_id,
                pack_id: pl.pack_id,
                offset: pl.offset,
                length: blob.length,
                uncompressed_size: blob.uncompressed_size,
                feature_count: p.feature_count,
                hilbert: p.hilbert,
                crc32c: blob.crc,
                temporal_bucket_ms: p.temporal_bucket_ms,
                cover_t_min: p.cover_t_min,
            });
        }
        // Directory codec order is (zoom, hilbert, time_start); the extra
        // bucket key totalizes ties between a cell's base and temporal-LOD
        // entries so the encoded directory is byte-reproducible too.
        entries.sort_by_key(|e| {
            (
                e.zoom,
                e.hilbert,
                e.time_start,
                e.variant_id,
                e.temporal_bucket_ms,
            )
        });

        // The manifest's totals are derived from the directory itself, not
        // taken from the caller: `tile_count` = directory entries,
        // `feature_count` = sum of per-entry counts (the same total
        // stt-validate reports as `feature_count_index`). Deriving here keeps
        // manifest and directory consistent by construction for every writer
        // path; a caller that forgot to set them would otherwise ship 0s.
        let mut metadata = metadata.clone();
        // Co-version the layout with the workload it was optimised for. Only a
        // simulated (`measured`) ordering sets this, so every other build emits
        // byte-identical metadata to a build that predates the field.
        //
        // This is the READER-COMPAT MIRROR of `Manifest::ordering_workload`
        // (the canonical key, beside `blobOrdering`). Written from the same
        // `resolved_workload` in the same function so the two copies cannot
        // diverge; see the rustdoc on `Manifest::ordering_workload` for the
        // removal trigger.
        if let Some(workload) = resolved_workload {
            metadata.ordering_workload = Some(workload);
        }
        metadata.tile_count = entries.len() as u64;
        metadata.feature_count = entries.iter().map(|e| u64::from(e.feature_count)).sum();

        // --- Write the remaining objects ----------------------------------
        // (Packs already streamed to disk above — content-addressed via the
        // incremental hash, atomically renamed at each seal.)

        // Encode the directory. Both shapes are zstd-at-rest (declared via
        // `directory.encoding`) — directories compress ~2x and sit on the
        // cold-start critical path with no CDN content-encoding rescue:
        //   - single (default): one zstd frame over the whole v6 codec buffer.
        //   - paged (opt-in):   a root page + leaf pages, each its own zstd
        //     frame, so a cold reader fetches only the leaves it touches. The
        //     leaf codec is the same v6 directory.
        // The object is content-addressed over its at-rest bytes either way.
        let use_paging = page_entries.filter(|_| entries.len() >= paging_min_entries);
        let (index_bytes, directory_ref_fields): (Vec<u8>, _) = if let Some(k) = use_paging {
            let paged =
                crate::directory_page::encode_paged_directory_level(&entries, k, true, zstd_level)?;
            (
                paged.bytes,
                (
                    Some(DIRECTORY_LAYOUT_PAGED.to_string()),
                    Some(paged.root_length),
                    Some(paged.page_count as u64),
                    Some(paged.page_entries as u64),
                    Some(paged.root_hash),
                    Some(paged.page_hashes),
                ),
            )
        } else {
            let index_plain = crate::directory::encode_directory(&entries);
            let bytes = compression::compress_zstd_with_dict_level(&index_plain, None, zstd_level)?;
            (bytes, (None, None, None, None, None, None))
        };
        // formatVersion 3: the object is magic + frames; `rootLength` keeps meaning the
        // root frame's at-rest length (a paged reader fetches
        // `bytes=0..(8+rootLength-1)`), so the DirectoryRef fields above need
        // no adjustment — only the object bytes gain the prelude.
        let index_bytes: Vec<u8> = {
            let mut with_magic = Vec::with_capacity(OBJECT_MAGIC_LEN + index_bytes.len());
            with_magic.extend_from_slice(&object_magic(DIRECTORY_MAGIC));
            with_magic.extend_from_slice(&index_bytes);
            with_magic
        };
        let index_hex = blake3_128_hex(&index_bytes);
        let index_rel = format!("index/{index_hex}.sttd");
        let index_path = out_dir.join(&index_rel);
        write_atomic(&index_dir, &index_path, &index_bytes)?;

        let (layout, root_length, page_count, page_entries_field, root_hash, page_hashes) =
            directory_ref_fields;

        // Publish the collected schema templates. The collector snapshot is
        // sorted by hash and deduped by construction, so the manifest bytes are
        // independent of encode order/parallelism.
        let schemas: Vec<SchemaTemplateRef> = {
            use base64::Engine as _;
            templates
                .snapshot()
                .into_iter()
                .map(|(hash, data)| SchemaTemplateRef {
                    hash: hash.iter().map(|b| format!("{b:02x}")).collect(),
                    data: base64::engine::general_purpose::STANDARD.encode(&data),
                })
                .collect()
        };

        let mut variants = vec![ManifestVariant {
            id: crate::tile::RAW_VARIANT_ID,
            kind: VariantKind::Raw,
            layer_name: None,
            method: None,
            params: None,
        }];
        if let Some(summary) = &metadata.summary_tier {
            if summary.variant_id == crate::tile::RAW_VARIANT_ID {
                return Err(Error::Other(
                    "summary_tier.variant_id must not reuse raw variant 0".into(),
                ));
            }
            variants.push(ManifestVariant {
                id: summary.variant_id,
                kind: VariantKind::Summary,
                layer_name: Some(summary.layer_name.clone()),
                method: None,
                params: None,
            });
        }
        variants.sort_by_key(|variant| variant.id);
        variants.dedup_by_key(|variant| variant.id);
        for entry in &entries {
            if !variants
                .iter()
                .any(|variant| variant.id == entry.variant_id)
            {
                return Err(Error::Other(format!(
                    "tile variant {} is not declared by metadata/manifest",
                    entry.variant_id
                )));
            }
        }

        // Build + write the manifest.
        let manifest = Manifest {
            format: PACKED_FORMAT.to_string(),
            format_version: PACKED_FORMAT_VERSION,
            capabilities,
            schemas,
            variants,
            compression: "zstd".to_string(),
            // Record the concrete order actually laid down (never auto/measured)
            // and, beside it, the workload model that order was chosen under.
            // Both keys are omitted when nothing was simulated.
            blob_ordering: Some(ordering.as_str().to_string()),
            // TB-10: empty unless an adaptive build set it, and skipped from the
            // JSON when empty, so non-adaptive manifests are byte-unchanged.
            adaptive_boundaries,
            ordering_workload: resolved_workload,
            directory: DirectoryRef {
                key: index_rel,
                length: index_bytes.len() as u64,
                directory_version: crate::directory::DIRECTORY_VERSION,
                encoding: Some(DIRECTORY_ENCODING_ZSTD.to_string()),
                layout,
                root_length,
                page_count,
                page_entries: page_entries_field,
                root_hash,
                page_hashes,
            },
            packs: pack_refs,
            metadata,
        };
        let manifest_bytes = manifest.to_json_bytes()?;
        let manifest_path = out_dir.join("manifest.json");
        let mut f = File::create(&manifest_path)?;
        f.write_all(&manifest_bytes)?;
        f.flush()?;

        Ok(manifest)
    }
}

/// Unwrap a directory object's at-rest encoding into raw codec bytes.
/// `encoding` is the manifest's `directory.encoding`: absent = raw,
/// `"zstd"` = one zstd frame around the codec bytes.
fn decode_directory_object(
    bytes: &[u8],
    encoding: Option<&str>,
    max_output: usize,
) -> Result<Vec<u8>> {
    match encoding {
        None => {
            if bytes.len() > max_output {
                return Err(Error::InvalidArchive(format!(
                    "directory is {} bytes, exceeding the {max_output}-byte limit derived from tileCount",
                    bytes.len()
                )));
            }
            Ok(bytes.to_vec())
        }
        Some(DIRECTORY_ENCODING_ZSTD) => {
            compression::decompress_zstd_with_dict_bounded(bytes, None, max_output)
        }
        Some(other) => Err(Error::InvalidArchive(format!(
            "unknown directory encoding {other:?} (this reader supports absent or \"zstd\")"
        ))),
    }
}

/// Decode the full tile-entry list from a directory object's at-rest bytes,
/// branching on the container layout. The single (whole-load) shape unwraps the
/// at-rest encoding then runs the v6 codec; the paged shape decodes the root +
/// every leaf (the local load-all path — a mmap'd file has no cold-start cost,
/// so the paging *query* win lives in the TS HTTP reader). Used by the local
/// `PackedReader` and `verify_packed_objects`.
fn decode_directory_entries(
    bytes: &[u8],
    dref: &DirectoryRef,
    expected_entries: u64,
) -> Result<Vec<TileEntry>> {
    if dref.is_paged() {
        let root_length = dref.root_length.ok_or_else(|| {
            Error::InvalidArchive("paged directory: manifest missing rootLength".into())
        })?;
        let zstd = dref.encoding.as_deref() == Some(DIRECTORY_ENCODING_ZSTD);
        validate_paged_frame_hashes(bytes, dref, zstd)?;
        crate::directory_page::decode_paged_directory_bounded(
            bytes,
            root_length,
            zstd,
            dref.page_count,
        )
    } else {
        let entry_count = usize::try_from(expected_entries).map_err(|_| {
            Error::InvalidArchive(format!(
                "metadata tileCount {expected_entries} does not fit this platform"
            ))
        })?;
        let max_output = crate::directory_page::DIRECTORY_FRAME_OVERHEAD
            .checked_add(
                entry_count
                    .checked_mul(crate::directory_page::MAX_DECODED_BYTES_PER_ENTRY)
                    .ok_or_else(|| {
                        Error::InvalidArchive("directory decode limit overflows".into())
                    })?,
            )
            .ok_or_else(|| Error::InvalidArchive("directory decode limit overflows".into()))?
            .min(crate::directory_page::MAX_DECODED_DIRECTORY_BYTES);
        let raw = decode_directory_object(bytes, dref.encoding.as_deref(), max_output)?;
        crate::directory::decode_directory(&raw)
    }
}

fn validate_paged_frame_hashes(bytes: &[u8], dref: &DirectoryRef, zstd: bool) -> Result<()> {
    // Per-frame hashes are a v6 addition. A pre-v6 container never carried them,
    // so demanding them turns "this archive predates the feature" into "this
    // archive is corrupt". Skip only for those, and ONLY when they are genuinely
    // absent: a v6 directory missing its hashes really is corrupt, and a pre-v6
    // one that DOES carry them is still checked.
    if dref.directory_version < crate::directory::DIRECTORY_VERSION
        && dref.root_hash.is_none()
        && dref.page_hashes.is_none()
    {
        return Ok(());
    }
    let root_hash = dref
        .root_hash
        .as_deref()
        .ok_or_else(|| Error::InvalidArchive("paged directory missing rootHash".into()))?;
    let page_hashes = dref
        .page_hashes
        .as_deref()
        .ok_or_else(|| Error::InvalidArchive("paged directory missing pageHashes".into()))?;
    let root_length = dref
        .root_length
        .ok_or_else(|| Error::InvalidArchive("paged directory missing rootLength".into()))?;
    let root_len = usize::try_from(root_length)
        .map_err(|_| Error::InvalidArchive("paged rootLength does not fit this platform".into()))?;
    let root_frame = bytes
        .get(..root_len)
        .ok_or_else(|| Error::InvalidArchive("paged rootLength exceeds directory object".into()))?;
    let actual_root = blake3_128_hex(root_frame);
    if actual_root != root_hash {
        return Err(Error::InvalidArchive(format!(
            "paged directory root hash mismatch: bytes hash to {actual_root}, manifest declares {root_hash}"
        )));
    }

    let page_count = dref
        .page_count
        .ok_or_else(|| Error::InvalidArchive("paged directory missing pageCount".into()))?;
    let page_count_usize = usize::try_from(page_count)
        .map_err(|_| Error::InvalidArchive("paged pageCount does not fit this platform".into()))?;
    let root_limit = crate::directory_page::ROOT_HEADER_LEN
        .checked_add(
            page_count_usize
                .checked_mul(crate::directory_page::DESCRIPTOR_LEN)
                .ok_or_else(|| Error::InvalidArchive("paged root size overflows".into()))?,
        )
        .ok_or_else(|| Error::InvalidArchive("paged root size overflows".into()))?
        .min(crate::directory_page::MAX_DECODED_DIRECTORY_BYTES);
    let root_raw = crate::directory_page::unframe_bounded(root_frame, zstd, root_limit)?;
    let root = crate::directory_page::decode_root(&root_raw)?;
    for (index, (descriptor, expected_hash)) in
        root.pages.iter().zip(page_hashes.iter()).enumerate()
    {
        let start = root_length
            .checked_add(descriptor.rel_offset)
            .ok_or_else(|| Error::InvalidArchive("paged leaf offset overflows".into()))?;
        let end = start
            .checked_add(descriptor.length as u64)
            .ok_or_else(|| Error::InvalidArchive("paged leaf range overflows".into()))?;
        let start = usize::try_from(start).map_err(|_| {
            Error::InvalidArchive(format!(
                "paged leaf {index} start does not fit this platform"
            ))
        })?;
        let end = usize::try_from(end).map_err(|_| {
            Error::InvalidArchive(format!("paged leaf {index} end does not fit this platform"))
        })?;
        let frame = bytes
            .get(start..end)
            .ok_or_else(|| Error::InvalidArchive(format!("paged leaf {index} is out of bounds")))?;
        let actual = blake3_128_hex(frame);
        if actual != *expected_hash {
            return Err(Error::InvalidArchive(format!(
                "paged directory leaf {index} hash mismatch: bytes hash to {actual}, manifest declares {expected_hash}"
            )));
        }
    }
    Ok(())
}

/// Write `bytes` to a temp file inside `dir` then atomically rename to
/// `final_path` (content-addressed, so an existing identical object is a no-op
/// overwrite of the same bytes).
fn write_atomic(dir: &Path, final_path: &Path, bytes: &[u8]) -> Result<()> {
    // Unique temp name within the same directory so the rename is atomic.
    let tmp = dir.join(format!(
        ".tmp-{}-{}",
        std::process::id(),
        blake3_128_hex(bytes)
    ));
    {
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.flush()?;
    }
    fs::rename(&tmp, final_path)?;
    Ok(())
}

/// Streaming pack emitter for [`PackWriter::finalize`]: blobs are assigned to
/// packs greedily in first-seen (curve) order, so each new unique blob's
/// compressed bytes are appended straight to the current pack's TEMP file
/// while an INCREMENTAL blake3 hasher tracks the object's content address;
/// exceeding `pack_target_bytes` seals the pack — finalize the hash, rename
/// the temp to `packs/<hex>.sttp`. Only the current pack's file handle (not
/// the archive's compressed bytes) stays resident, so finalize's memory is
/// independent of dataset size. Same bytes, same order, same hashing as the
/// old buffer-everything path — output is byte-identical (pinned by the
/// golden/reproducibility tests). The open temp file is removed on drop
/// (error paths); sealed packs are content-addressed, so re-writes after a
/// failed run are idempotent.
/// One blob as the WM-5 planner sees it: its POST-DEDUP COMPRESSED length in
/// final layout order.
///
/// Using the post-dedup compressed length is the point — §6.5 gap 3 records
/// that the build-path simulation drifted by pricing pre-dedup *uncompressed*
/// bytes, which is not what the reader fetches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedBlob {
    pub len: u64,
}

/// Knobs for [`plan_pack_cuts`] (WM-5).
#[derive(Debug, Clone)]
pub struct PackCutOptions {
    /// Soft target pack size — the incumbent's capacity.
    pub target_bytes: u64,
    /// HARD per-object ceiling (Cloudflare's 512 MB), never exceeded except by
    /// the unsplittable-blob exemption.
    pub max_bytes: u64,
    /// Bytes-equivalent price of one pack boundary: a cut costs at most one
    /// extra request to any query that spans it, and P3 says price requests in
    /// bytes. Defaults to one coalesce gap.
    pub boundary_price_bytes: u64,
}

impl Default for PackCutOptions {
    fn default() -> Self {
        Self {
            target_bytes: DEFAULT_PACK_TARGET_BYTES,
            max_bytes: 512 * 1024 * 1024,
            boundary_price_bytes: 2 * 1024 * 1024,
        }
    }
}

/// WM-5 — plan pack cut positions by shortest path over candidate cuts.
///
/// Returns the indices to cut BEFORE, ascending. An empty plan reproduces
/// next-fit exactly.
///
/// ⚠️ REGISTER: the next-fit cut is pinned as EXACTLY OPTIMAL *for its own cost
/// model* (fewest segments under capacity), and this item does not touch that.
/// What changes is the cost MODEL: the objective gains a workload term, and the
/// solver for that different objective is this DP. Next-fit remains reachable
/// and semantically intact as the no-plan path.
///
/// Cost of a segment = `boundary_price_bytes` (one boundary) + the segment's
/// own bytes. Minimizing that over contiguous segments under the capacity
/// constraint is a shortest path on a DAG; with the monotone edge cost here it
/// is solved exactly by the O(m·C/ℓ_min) forward scan below.
///
/// Deterministic: integer arithmetic, no RNG, ties resolved toward the EARLIER
/// cut so the plan is a pure function of the blob lengths.
pub fn plan_pack_cuts(blobs: &[PlannedBlob], opts: &PackCutOptions) -> Vec<usize> {
    let n = blobs.len();
    if n == 0 {
        return Vec::new();
    }
    let cap = opts.max_bytes.max(1);
    let target = opts.target_bytes.clamp(1, cap);

    // best[i] = (cost of packing blobs[i..], index of the cut that achieves it)
    let mut best_cost = vec![u128::MAX; n + 1];
    let mut best_next = vec![n; n + 1];
    best_cost[n] = 0;

    for i in (0..n).rev() {
        let mut run = 0u64;
        let mut j = i;
        while j < n {
            let blen = blobs[j].len;
            // The unsplittable-blob exemption: a lone blob larger than the cap
            // still owns a pack rather than being dropped.
            if run > 0 && run.saturating_add(blen) > cap {
                break;
            }
            run = run.saturating_add(blen);
            j += 1;
            // A segment past the soft target buys nothing, so stop extending —
            // this is what bounds the scan to O(C/ℓ_min) per start.
            let seg_cost = u128::from(opts.boundary_price_bytes) + u128::from(run);
            if best_cost[j] != u128::MAX {
                let total = seg_cost.saturating_add(best_cost[j]);
                if total < best_cost[i] {
                    best_cost[i] = total;
                    best_next[i] = j;
                }
            }
            if run >= target {
                break;
            }
        }
        // Safety net: a start that could not reach any priced suffix takes the
        // singleton, which is always legal.
        if best_cost[i] == u128::MAX {
            let j = (i + 1).min(n);
            best_cost[i] = u128::from(opts.boundary_price_bytes) + u128::from(blobs[i].len);
            best_next[i] = j;
        }
    }

    let mut cuts = Vec::new();
    let mut i = best_next[0];
    while i < n {
        cuts.push(i);
        i = best_next[i];
    }
    cuts
}

struct PackStreamWriter {
    out_dir: PathBuf,
    packs_dir: PathBuf,
    pack_target_bytes: u64,
    /// Object-absolute offset of the first blob (`OBJECT_MAGIC_LEN`, past the
    /// magic prelude) — the "empty pack" length.
    data_start: u64,
    current: Option<OpenPack>,
    next_pack_id: u32,
    refs: Vec<PackRef>,
    /// WM-5: planned cut positions (cut BEFORE blob `i`), ascending. Empty =
    /// pure next-fit, which is the default and the documented rollback.
    cut_plan: Vec<usize>,
    /// Count of UNIQUE blobs appended so far — the index `cut_plan` refers to.
    /// Deduped blobs never reach `append`, so this is the layout position.
    appended_unique: usize,
}

/// The in-progress pack object: temp file + incremental hash + length so far.
struct OpenPack {
    file: File,
    tmp_path: PathBuf,
    hasher: blake3::Hasher,
    len: u64,
}

impl Drop for OpenPack {
    fn drop(&mut self) {
        // Sealing renames the temp away first, so this only fires for an
        // ABANDONED pack (finalize errored) — never a published object.
        let _ = fs::remove_file(&self.tmp_path);
    }
}

impl PackStreamWriter {
    fn new(out_dir: &Path, packs_dir: &Path, pack_target_bytes: u64) -> Self {
        Self {
            out_dir: out_dir.to_path_buf(),
            packs_dir: packs_dir.to_path_buf(),
            pack_target_bytes,
            data_start: OBJECT_MAGIC_LEN as u64,
            current: None,
            next_pack_id: 0,
            refs: Vec::new(),
            cut_plan: Vec::new(),
            appended_unique: 0,
        }
    }

    /// Append one blob's compressed bytes, returning its `(pack_id,
    /// object-absolute offset)` placement. Cuts BEFORE the blob if adding it
    /// would exceed the target and the current pack is non-empty, so a lone
    /// oversized blob still fits — it just owns a pack (the unsplittable-blob
    /// rule, unchanged).
    fn append(&mut self, compressed: &[u8]) -> Result<(u32, u64)> {
        let blen = compressed.len() as u64;
        let idx = self.appended_unique;
        self.appended_unique += 1;
        // WM-5: an explicit plan cuts here; otherwise next-fit decides.
        let planned_cut = self.cut_plan.binary_search(&idx).is_ok();
        if let Some(cur) = &self.current {
            let nonempty = cur.len > self.data_start;
            // Next-fit is retained VERBATIM as both the no-plan path and the
            // safety net: a planned cut may never leave an overfull pack, and
            // if a plan somehow would, next-fit still seals.
            if nonempty && (planned_cut || cur.len + blen > self.pack_target_bytes) {
                self.seal()?;
            }
        }
        if self.current.is_none() {
            self.open()?;
        }
        let cur = self.current.as_mut().expect("just opened");
        let offset = cur.len;
        cur.file.write_all(compressed)?;
        cur.hasher.update(compressed);
        cur.len += blen;
        Ok((self.next_pack_id, offset))
    }

    /// Start a new pack: unique temp file (pack ids are unique per run) with
    /// the object magic prelude already written and hashed.
    fn open(&mut self) -> Result<()> {
        let tmp_path = self.packs_dir.join(format!(
            ".tmp-{}-pack-{}",
            std::process::id(),
            self.next_pack_id
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        let mut hasher = blake3::Hasher::new();
        let magic = object_magic(PACK_MAGIC);
        file.write_all(&magic)?;
        hasher.update(&magic);
        let len = OBJECT_MAGIC_LEN as u64;
        self.current = Some(OpenPack {
            file,
            tmp_path,
            hasher,
            len,
        });
        Ok(())
    }

    /// Seal the current pack (if any): flush, finalize the incremental hash
    /// into the content address, atomically rename temp → final, record the
    /// [`PackRef`]. No-op when nothing is open.
    fn seal(&mut self) -> Result<()> {
        let Some(mut cur) = self.current.take() else {
            return Ok(());
        };
        cur.file.flush()?;
        let hash = cur.hasher.finalize();
        let hex: String = hash.as_bytes()[..16]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let rel = format!("packs/{hex}.sttp");
        fs::rename(&cur.tmp_path, self.out_dir.join(&rel))?;
        self.refs.push(PackRef {
            key: rel,
            length: cur.len,
        });
        self.next_pack_id += 1;
        Ok(())
    }

    /// Seal any trailing open pack and return the complete pack table
    /// (index == `pack_id`).
    fn finish(mut self) -> Result<Vec<PackRef>> {
        self.seal()?;
        Ok(std::mem::take(&mut self.refs))
    }
}

// ----------------------------------------------------------------------------
// Integrity
// ----------------------------------------------------------------------------

/// Verify the on-disk integrity of a packed dataset against its manifest.
///
/// Because packs and the directory are **content-addressed**, integrity is a
/// property anyone can check with no trusted side-channel: each object's bytes
/// must blake3-hash to the name the manifest gave it, and its on-disk length
/// must match the declared length. Additionally the directory must decode and
/// reference no `pack_id` outside the manifest's pack table.
///
/// Returns the list of violations (empty ⇒ clean). Returns `Err` only when the
/// manifest itself cannot be read or parsed (a missing referenced object is a
/// reported violation, not an `Err`, so a full report is produced in one pass).
pub fn verify_packed_objects<P: AsRef<Path>>(manifest_path: P) -> Result<Vec<String>> {
    let manifest_path = manifest_path.as_ref();
    let root = manifest_path
        .parent()
        .ok_or_else(|| Error::InvalidArchive("manifest path has no parent dir".into()))?;
    let manifest = Manifest::from_json_bytes(&fs::read(manifest_path)?)?;

    let mut issues = Vec::new();
    if let Err(error) = manifest_open_checks(&manifest) {
        issues.push(error.to_string());
        // Object keys are untrusted until the semantic checks pass. Do not
        // resolve even one referenced path for a malformed manifest.
        return Ok(issues);
    }

    if manifest.format != PACKED_FORMAT {
        issues.push(format!(
            "manifest format is {:?}, expected {PACKED_FORMAT:?}",
            manifest.format
        ));
    }
    if !(MIN_PACKED_FORMAT_VERSION..=PACKED_FORMAT_VERSION).contains(&manifest.format_version) {
        issues.push(format!(
            "manifest formatVersion is {}, expected \
             {MIN_PACKED_FORMAT_VERSION}..={PACKED_FORMAT_VERSION}",
            manifest.format_version
        ));
    }
    if !(crate::directory::MIN_DIRECTORY_VERSION..=crate::directory::DIRECTORY_VERSION)
        .contains(&manifest.directory.directory_version)
    {
        issues.push(format!(
            "directoryVersion is {}, expected {}",
            manifest.directory.directory_version,
            crate::directory::DIRECTORY_VERSION
        ));
    }
    verify_manifest_schemas(&manifest, &mut issues);

    // Each content-addressed object: name must equal blake3-128 of its bytes,
    // and on-disk length must equal the declared length. Content addresses
    // cover the ENTIRE object — magic prelude included on formatVersion 3 — so this check
    // is version-independent; the magic itself is validated separately below.
    fn check_object(
        root: &Path,
        key: &str,
        declared_len: u64,
        prefix: &str,
        ext: &str,
        issues: &mut Vec<String>,
    ) {
        match fs::read(root.join(key)) {
            Ok(bytes) => {
                if bytes.len() as u64 != declared_len {
                    issues.push(format!(
                        "{key}: on-disk length {} != manifest-declared {declared_len}",
                        bytes.len()
                    ));
                }
                let expected = format!("{prefix}/{}.{ext}", blake3_128_hex(&bytes));
                if key != expected {
                    issues.push(format!(
                        "{key}: content-address mismatch (bytes hash to {expected})"
                    ));
                }
            }
            Err(e) => issues.push(format!("{key}: cannot read object ({e})")),
        }
    }

    check_object(
        root,
        &manifest.directory.key,
        manifest.directory.length,
        "index",
        "sttd",
        &mut issues,
    );
    for p in &manifest.packs {
        check_object(root, &p.key, p.length, "packs", "sttp", &mut issues);
    }

    // Objects must self-identify: validate each object's magic prelude.
    {
        for (key, kind) in std::iter::once((&manifest.directory.key, DIRECTORY_MAGIC))
            .chain(manifest.packs.iter().map(|p| (&p.key, PACK_MAGIC)))
        {
            if let Ok(bytes) = fs::read(root.join(key)) {
                if let Err(e) = strip_object_magic(&bytes, kind, key) {
                    issues.push(e.to_string());
                }
            }
        }
    }

    // Directory must decode (through its magic prelude, at-rest encoding +
    // container layout) and reference only packs that exist. Under
    // formatVersion 3, the
    // decoded entries additionally drive the frame → manifest.schemas
    // reference check (a dataset whose frames reference a missing template
    // is undecodable and must not verify clean).
    match fs::read(root.join(&manifest.directory.key)) {
        Ok(dir_bytes) => {
            match directory_codec_bytes(&dir_bytes, manifest.format_version).and_then(|codec| {
                decode_directory_entries(codec, &manifest.directory, manifest.metadata.tile_count)
            }) {
                Ok(entries) => {
                    if let Some(max_pid) = entries.iter().map(|e| e.pack_id).max() {
                        if max_pid as usize >= manifest.packs.len() {
                            issues.push(format!(
                                "directory references pack_id {max_pid} but the manifest lists only {} pack(s)",
                                manifest.packs.len()
                            ));
                        }
                    }
                    verify_v2_frame_template_refs(
                        &manifest,
                        &entries,
                        |pid| {
                            manifest
                                .packs
                                .get(pid)
                                .and_then(|p| fs::read(root.join(&p.key)).ok())
                        },
                        &mut issues,
                    );
                }
                Err(e) => issues.push(format!("directory failed to decode: {e}")),
            }
        }
        // A read failure here is already reported by check_object above.
        Err(_) => {}
    }

    // Paged directories: validate the container structure beyond plain decode —
    // page descriptor bounds cover their leaf's entries (so a reader's prune
    // never drops a matching tile) and cross-page key order is monotonic.
    if manifest.directory.is_paged() {
        match manifest.directory.root_length {
            Some(rl) => {
                if let Ok(dir_bytes) = fs::read(root.join(&manifest.directory.key)) {
                    if let Ok(codec) = directory_codec_bytes(&dir_bytes, manifest.format_version) {
                        let zstd =
                            manifest.directory.encoding.as_deref() == Some(DIRECTORY_ENCODING_ZSTD);
                        match crate::directory_page::verify_paged_structure(codec, rl, zstd) {
                            Ok(mut more) => issues.append(&mut more),
                            Err(e) => issues.push(format!("paged structure check failed: {e}")),
                        }
                    }
                }
            }
            None => issues.push("paged directory: manifest missing rootLength".into()),
        }
    }

    Ok(issues)
}

/// Shared `manifest.schemas` checks for both verifiers: entries must
/// base64-decode, hash to their declared address, and arrive sorted-by-hash +
/// deduped (the byte-reproducibility contract).
fn verify_manifest_schemas(manifest: &Manifest, issues: &mut Vec<String>) {
    if let Err(e) = build_template_registry(&manifest.schemas) {
        issues.push(e.to_string());
    }
    let hashes: Vec<&str> = manifest.schemas.iter().map(|s| s.hash.as_str()).collect();
    if hashes.windows(2).any(|w| w[0] >= w[1]) {
        issues.push("manifest schemas are not sorted by hash (or contain duplicates)".into());
    }
}

/// Parse a 32-lower-hex schema hash into its raw 16 bytes.
fn parse_schema_hash_hex(s: &str) -> Option<[u8; 16]> {
    if s.len() != 32 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// Layer-frame deep check shared by both verifiers: every template hash a tile frame
/// references must resolve in `manifest.schemas` — otherwise the dataset
/// "verifies clean" while no tile can decode. Walks each UNIQUE blob (dedup by
/// `(pack_id, offset)`), decompresses it, parses ONLY the layer-frame header
/// (escape/version/count, per-layer ref_kinds + 16-byte hashes — no Arrow
/// decode; [`crate::arrow_tile::frame_v2_template_refs`]) and reports each
/// missing hash once, with an example tile. `pack_object` returns a pack's
/// full object bytes by `pack_id` (`None` when the object is unreadable —
/// already reported by the content-address checks).
fn verify_v2_frame_template_refs<F>(
    manifest: &Manifest,
    entries: &[TileEntry],
    mut pack_object: F,
    issues: &mut Vec<String>,
) where
    F: FnMut(usize) -> Option<Vec<u8>>,
{
    if !(MIN_PACKED_FORMAT_VERSION..=PACKED_FORMAT_VERSION).contains(&manifest.format_version) {
        return;
    }
    // Unparseable hex entries simply can't match any frame reference; their
    // own malformation is reported by `verify_manifest_schemas`.
    let known: std::collections::HashSet<[u8; 16]> = manifest
        .schemas
        .iter()
        .filter_map(|s| parse_schema_hash_hex(&s.hash))
        .collect();

    let mut seen_blobs: std::collections::HashSet<(u32, u64)> = std::collections::HashSet::new();
    let mut pack_cache: HashMap<u32, Option<Vec<u8>>> = HashMap::new();
    // hex hash → example tile (BTreeMap for deterministic issue order).
    let mut missing: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for e in entries {
        if !seen_blobs.insert((e.pack_id, e.offset)) {
            continue;
        }
        let Some(pack) = pack_cache
            .entry(e.pack_id)
            .or_insert_with(|| pack_object(e.pack_id as usize))
            .as_deref()
        else {
            continue;
        };
        // Compute the exclusive end in u64: a corrupt offset near u64::MAX
        // would wrap the `usize` add and slip past the `end > pack.len()`
        // guard into a slice panic.
        let start = e.offset as usize;
        let end = match e.offset.checked_add(e.length as u64) {
            Some(end) if end <= pack.len() as u64 => end as usize,
            _ => {
                issues.push(format!(
                    "tile {:?} blob range at offset {} (length {}) exceeds pack {} size {}",
                    e.tile_id(),
                    e.offset,
                    e.length,
                    e.pack_id,
                    pack.len()
                ));
                continue;
            }
        };
        let payload = match manifest.compression.as_str() {
            "zstd" => match compression::decompress_zstd_with_dict(&pack[start..end], None) {
                Ok(p) => p,
                Err(err) => {
                    issues.push(format!(
                        "tile {:?}: blob failed to decompress during the schema-reference \
                         check: {err}",
                        e.tile_id()
                    ));
                    continue;
                }
            },
            _ => pack[start..end].to_vec(),
        };
        if !crate::arrow_tile::is_frame_v2(&payload) {
            issues.push(format!(
                "tile {:?}: v1 layer frame inside a formatVersion-3 dataset",
                e.tile_id()
            ));
            continue;
        }
        match crate::arrow_tile::frame_v2_template_refs(&payload) {
            Ok(refs) => {
                for hash in refs {
                    if !known.contains(&hash) {
                        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
                        missing
                            .entry(hex)
                            .or_insert_with(|| format!("{:?}", e.tile_id()));
                    }
                }
            }
            Err(err) => issues.push(format!(
                "tile {:?}: layer-frame header parse failed: {err}",
                e.tile_id()
            )),
        }
    }
    for (hex, tile) in missing {
        issues.push(format!(
            "tile frames reference schema template {hex}, which is absent from \
             manifest.schemas (e.g. tile {tile}) — no such tile can decode"
        ));
    }
}

// ----------------------------------------------------------------------------
// Reader
// ----------------------------------------------------------------------------

/// One mapped pack, lazily loaded on first access.
struct LoadedPack {
    /// `None` until first read of a tile in this pack.
    mmap: Option<Mmap>,
    /// Absolute path to the pack object.
    path: PathBuf,
    /// Declared length from the manifest (for a bounds sanity check).
    length: u64,
}

/// Reader for a **local** packed dataset. Remote/HTTP reads are the TS reader's
/// job; this opens objects from the filesystem.
///
/// Mirrors `ArchiveReader`: [`entries`](Self::entries) returns
/// the decoded v6 directory, [`metadata`](Self::metadata) the folded metadata,
/// and [`read_payload`](Self::read_payload) selects the entry's pack, slices
/// `[offset..offset+length]`, verifies CRC32C and decompresses the per-blob
/// zstd. Packs are mmap'd lazily by `pack_id`.
pub struct PackedReader {
    entries: Vec<TileEntry>,
    metadata: Metadata,
    compression: Compression,
    packs: Vec<std::cell::RefCell<LoadedPack>>,
    capabilities: Vec<String>,
    /// The manifest's authoritative packed `formatVersion` (3).
    format_version: u32,
    /// The hash-validated schema-template registry decoded from
    /// `manifest.schemas` at open.
    templates: Option<TemplateRegistry>,
    /// `Some` iff this reader was opened via
    /// [`open_bundle`](Self::open_bundle): the whole `.sttb` is one mapping
    /// and each pack is an (offset, length) window into it. `None` for the
    /// exploded-directory [`open`](Self::open) path, which keeps using
    /// `packs` above.
    bundle: Option<BundleBacking>,
}

/// Bundle backing for a [`PackedReader`] opened from a single-file `.sttb`:
/// one mmap of the whole bundle plus each pack's window into it.
struct BundleBacking {
    mmap: Mmap,
    /// Per-pack `(absolute_offset, length)` window into `mmap`. Index ==
    /// `pack_id` (manifest pack-table order). Bounds-checked at open.
    windows: Vec<(u64, u64)>,
}

/// Open-time manifest checks shared by [`PackedReader::open`] and
/// [`PackedReader::open_bundle`]: format tag, `formatVersion`, the
/// required-to-understand capability registry, and the compression codec.
/// Returns the decoded compression so both open paths stay in lockstep.
fn is_lower_blake3_128(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_object_key(key: &str, directory: &str, extension: &str) -> Result<()> {
    let mut parts = key.split('/');
    let prefix = parts.next();
    let file = parts.next();
    if prefix != Some(directory) || parts.next().is_some() {
        return Err(Error::InvalidArchive(format!(
            "unsafe object key {key:?}: expected {directory}/<32-lowercase-hex>.{extension}"
        )));
    }
    let Some(file) = file else {
        return Err(Error::InvalidArchive(format!("unsafe object key {key:?}")));
    };
    let suffix = format!(".{extension}");
    let Some(hash) = file.strip_suffix(&suffix) else {
        return Err(Error::InvalidArchive(format!(
            "unsafe object key {key:?}: expected .{extension} suffix"
        )));
    };
    if !is_lower_blake3_128(hash) {
        return Err(Error::InvalidArchive(format!(
            "unsafe object key {key:?}: content address must be 32 lowercase hex characters"
        )));
    }
    Ok(())
}

/// The variant registry to validate a manifest's tiles against.
///
/// A pre-v3 manifest carries no `variants` key at all, because the variant axis
/// did not exist when it was written — which means every payload in such an
/// archive is raw, and its directory decodes every entry to
/// [`crate::tile::RAW_VARIANT_ID`]. So the absent registry is not missing
/// information; it is the IMPLICIT raw-only registry, and this returns it
/// rather than mutating the parsed manifest (parse stays a faithful record of
/// the bytes on disk).
///
/// A v3 manifest is returned borrowed and unchanged — `variants` is required
/// there, and an empty one stays the hard error it should be.
fn effective_variants(manifest: &Manifest) -> std::borrow::Cow<'_, [ManifestVariant]> {
    if manifest.format_version >= PACKED_FORMAT_VERSION || !manifest.variants.is_empty() {
        return std::borrow::Cow::Borrowed(&manifest.variants);
    }
    let mut implied = vec![ManifestVariant {
        id: crate::tile::RAW_VARIANT_ID,
        kind: VariantKind::Raw,
        layer_name: None,
        method: None,
        params: None,
    }];
    // A legacy summary tier named its variant in `metadata.summary_tier` — the
    // only place it could — so honour that too, or every summary tile in the
    // archive would read as an undeclared variant.
    if let Some(summary) = &manifest.metadata.summary_tier {
        if summary.variant_id != crate::tile::RAW_VARIANT_ID {
            implied.push(ManifestVariant {
                id: summary.variant_id,
                kind: VariantKind::Summary,
                layer_name: None,
                method: None,
                params: None,
            });
        }
    }
    std::borrow::Cow::Owned(implied)
}

/// Semantic checks that JSON Schema alone cannot express and that every open
/// path must run before resolving any referenced object.
fn validate_manifest_semantics(manifest: &Manifest) -> Result<()> {
    let variants = effective_variants(manifest);
    if variants.is_empty() {
        return Err(Error::InvalidArchive(
            "manifest variants must contain at least the raw variant".into(),
        ));
    }
    let mut variant_ids = HashSet::with_capacity(variants.len());
    for variant in variants.iter() {
        if !variant_ids.insert(variant.id) {
            return Err(Error::InvalidArchive(format!(
                "manifest variant id {} is duplicated",
                variant.id
            )));
        }
    }
    if !variants.iter().any(|variant| {
        variant.id == crate::tile::RAW_VARIANT_ID && variant.kind == VariantKind::Raw
    }) {
        return Err(Error::InvalidArchive(
            "manifest variant 0 must be declared with kind raw".into(),
        ));
    }
    if let Some(summary) = &manifest.metadata.summary_tier {
        // A LEGACY manifest has no variant axis at all, so its `summary_tier`
        // carries no `variant_id` and reads back as the raw variant. That is
        // exactly what `effective_variants` above encodes — it deliberately
        // declares NO summary variant in that case, because a v2 directory has
        // no column that could distinguish summary entries from raw ones.
        // Demanding a summary declaration here therefore contradicted the
        // function two screens up and made every Rust tool refuse a v2 archive
        // with a summary tier ("variant 0 is not declared as kind summary"),
        // while the TypeScript reader opened the same archive without
        // complaint. Six published archives sit in that hole.
        //
        // v3 keeps the strict rule: there `variants` is required and
        // authoritative, so a summary tier that names an undeclared variant is
        // real drift.
        let legacy_untagged_summary = manifest.format_version < PACKED_FORMAT_VERSION
            && summary.variant_id == crate::tile::RAW_VARIANT_ID;
        if !legacy_untagged_summary
            && !variants
                .iter()
                .any(|v| v.id == summary.variant_id && v.kind == VariantKind::Summary)
        {
            return Err(Error::InvalidArchive(format!(
                "metadata summary_tier variant {} is not declared as kind summary",
                summary.variant_id
            )));
        }
    }
    // The directory codec is COUPLED to the manifest version, not independently
    // ranged. A current archive must carry the current codec — a v3 manifest
    // claiming directory v4 is drift, not history, and stays a hard error. Only
    // a genuinely legacy manifest may carry a legacy codec.
    let expected_directory = if manifest.format_version >= PACKED_FORMAT_VERSION {
        crate::directory::DIRECTORY_VERSION..=crate::directory::DIRECTORY_VERSION
    } else {
        crate::directory::MIN_DIRECTORY_VERSION..=crate::directory::DIRECTORY_VERSION
    };
    if !expected_directory.contains(&manifest.directory.directory_version) {
        return Err(Error::InvalidArchive(format!(
            "directoryVersion is {}, expected {}..={} for formatVersion {}",
            manifest.directory.directory_version,
            expected_directory.start(),
            expected_directory.end(),
            manifest.format_version
        )));
    }
    validate_object_key(&manifest.directory.key, "index", "sttd")?;
    if manifest.directory.length <= OBJECT_MAGIC_LEN as u64 {
        return Err(Error::InvalidArchive(format!(
            "directory {:?} length {} cannot contain magic plus a codec frame",
            manifest.directory.key, manifest.directory.length
        )));
    }
    match manifest.directory.encoding.as_deref() {
        None | Some(DIRECTORY_ENCODING_ZSTD) => {}
        Some(other) => {
            return Err(Error::InvalidArchive(format!(
                "unknown directory encoding {other:?}"
            )))
        }
    }

    match manifest.directory.layout.as_deref() {
        None | Some(DIRECTORY_LAYOUT_SINGLE) => {
            if manifest.directory.root_length.is_some()
                || manifest.directory.page_count.is_some()
                || manifest.directory.page_entries.is_some()
                || manifest.directory.root_hash.is_some()
                || manifest.directory.page_hashes.is_some()
            {
                return Err(Error::InvalidArchive(
                    "single directory must not declare paged-only fields".into(),
                ));
            }
        }
        Some(DIRECTORY_LAYOUT_PAGED) => {
            let root_length = manifest.directory.root_length.ok_or_else(|| {
                Error::InvalidArchive("paged directory is missing rootLength".into())
            })?;
            let page_count = manifest.directory.page_count.ok_or_else(|| {
                Error::InvalidArchive("paged directory is missing pageCount".into())
            })?;
            let page_entries = manifest.directory.page_entries.ok_or_else(|| {
                Error::InvalidArchive("paged directory is missing pageEntries".into())
            })?;
            if root_length == 0
                || root_length
                    > manifest
                        .directory
                        .length
                        .saturating_sub(OBJECT_MAGIC_LEN as u64)
            {
                return Err(Error::InvalidArchive(format!(
                    "paged rootLength {root_length} is outside the directory payload"
                )));
            }
            if page_count == 0 || page_entries == 0 {
                return Err(Error::InvalidArchive(
                    "paged pageCount and pageEntries must both be greater than zero".into(),
                ));
            }
            // Per-frame hashes arrived with directory v6. A pre-v6 container never
            // wrote them, so requiring them would report "corrupt" for an archive
            // that is merely older. Everything else about the paged shape above —
            // rootLength, pageCount, pageEntries and their bounds — is still
            // enforced, and a v6 directory missing its hashes remains an error.
            let legacy_unhashed = manifest.directory.directory_version
                < crate::directory::DIRECTORY_VERSION
                && manifest.directory.root_hash.is_none()
                && manifest.directory.page_hashes.is_none();
            if legacy_unhashed {
                return Ok(());
            }
            let root_hash = manifest.directory.root_hash.as_deref().ok_or_else(|| {
                Error::InvalidArchive("paged directory is missing rootHash".into())
            })?;
            let page_hashes = manifest.directory.page_hashes.as_deref().ok_or_else(|| {
                Error::InvalidArchive("paged directory is missing pageHashes".into())
            })?;
            if !is_lower_blake3_128(root_hash) {
                return Err(Error::InvalidArchive(
                    "directory rootHash must be 32 lowercase hex characters".into(),
                ));
            }
            if page_hashes.len() as u64 != page_count {
                return Err(Error::InvalidArchive(format!(
                    "directory pageHashes has {} values, pageCount is {page_count}",
                    page_hashes.len()
                )));
            }
            if page_hashes.iter().any(|h| !is_lower_blake3_128(h)) {
                return Err(Error::InvalidArchive(
                    "every directory pageHashes value must be 32 lowercase hex characters".into(),
                ));
            }
        }
        Some(other) => {
            return Err(Error::InvalidArchive(format!(
                "unknown directory layout {other:?}"
            )))
        }
    }

    if manifest.packs.is_empty() {
        return Err(Error::InvalidArchive(
            "packed manifest must contain at least one pack".into(),
        ));
    }
    let mut keys = HashSet::with_capacity(manifest.packs.len() + 1);
    keys.insert(manifest.directory.key.as_str());
    for pack in &manifest.packs {
        validate_object_key(&pack.key, "packs", "sttp")?;
        if pack.length < OBJECT_MAGIC_LEN as u64 {
            return Err(Error::InvalidArchive(format!(
                "pack {:?} length {} is smaller than its object magic",
                pack.key, pack.length
            )));
        }
        if !keys.insert(pack.key.as_str()) {
            return Err(Error::InvalidArchive(format!(
                "manifest object key {:?} is duplicated",
                pack.key
            )));
        }
    }

    let mut capabilities = HashSet::with_capacity(manifest.capabilities.len());
    if let Some(duplicate) = manifest
        .capabilities
        .iter()
        .find(|capability| !capabilities.insert(capability.as_str()))
    {
        return Err(Error::InvalidArchive(format!(
            "manifest capability {duplicate:?} is duplicated"
        )));
    }
    Ok(())
}

fn manifest_open_checks(manifest: &Manifest) -> Result<Compression> {
    if manifest.format != PACKED_FORMAT {
        return Err(Error::InvalidArchive(format!(
            "not a packed manifest: format={:?} (expected {PACKED_FORMAT:?})",
            manifest.format
        )));
    }
    // A future breaking manifest revision must fail loudly at open, not
    // misdecode downstream (the directory codec has its own version byte;
    // this guards the manifest schema itself). `formatVersion` is the
    // AUTHORITATIVE discriminator — the layer-frame escape is only
    // defense-in-depth.
    //
    // The accepted range bottoms out at [`MIN_PACKED_FORMAT_VERSION`]: v2 is a
    // CONTAINER-only difference (no `variants` registry, directory codec v5) and
    // its payloads share this toolchain's layer-frame version, so reading it
    // forks nothing below the container. Writers remain v3-only.
    if !(MIN_PACKED_FORMAT_VERSION..=PACKED_FORMAT_VERSION).contains(&manifest.format_version) {
        return Err(Error::InvalidArchive(format!(
            "unsupported packed formatVersion {} (this reader supports \
             {MIN_PACKED_FORMAT_VERSION}..={PACKED_FORMAT_VERSION})",
            manifest.format_version
        )));
    }
    validate_manifest_semantics(manifest)?;
    // Required-to-understand capabilities (spec §3.1): each one re-types
    // EXISTING columns, so a reader that lacks it wouldn't fail later — it
    // would silently misdecode, per tile. Refuse loudly at open instead.
    let unknown: Vec<&str> = manifest
        .capabilities
        .iter()
        .map(String::as_str)
        .filter(|c| !KNOWN_CAPABILITIES.contains(c))
        .collect();
    if !unknown.is_empty() {
        return Err(Error::InvalidArchive(format!(
            "dataset requires capabilities this reader does not implement: {} \
             (implemented: {})",
            unknown.join(", "),
            KNOWN_CAPABILITIES.join(", ")
        )));
    }
    match manifest.compression.as_str() {
        "zstd" => Ok(Compression::Zstd),
        "none" => Ok(Compression::None),
        other => Err(Error::InvalidArchive(format!(
            "unknown packed compression {other:?}"
        ))),
    }
}

fn validate_decoded_entries(manifest: &Manifest, entries: &[TileEntry]) -> Result<()> {
    if entries.len() as u64 != manifest.metadata.tile_count {
        return Err(Error::InvalidArchive(format!(
            "directory decoded {} entries, metadata.tileCount declares {}",
            entries.len(),
            manifest.metadata.tile_count
        )));
    }
    let variants = effective_variants(manifest);
    for entry in entries {
        if !variants
            .iter()
            .any(|variant| variant.id == entry.variant_id)
        {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} references undeclared variant {}",
                entry.tile_id(),
                entry.variant_id
            )));
        }
        let pack = manifest.packs.get(entry.pack_id as usize).ok_or_else(|| {
            Error::InvalidArchive(format!(
                "tile {:?} references pack {} but the manifest contains {} pack(s)",
                entry.tile_id(),
                entry.pack_id,
                manifest.packs.len()
            ))
        })?;
        let end = entry
            .offset
            .checked_add(entry.length as u64)
            .ok_or_else(|| {
                Error::InvalidArchive(format!(
                    "tile {:?}: blob offset+length overflows",
                    entry.tile_id()
                ))
            })?;
        if entry.offset < OBJECT_MAGIC_LEN as u64 || end > pack.length {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} blob range {}..{end} is outside declared pack {:?} length {}",
                entry.tile_id(),
                entry.offset,
                pack.key,
                pack.length
            )));
        }
    }
    Ok(())
}

impl PackedReader {
    /// Open a packed dataset by its `manifest.json` path. The directory and pack
    /// objects are resolved relative to the manifest's parent directory.
    pub fn open<P: AsRef<Path>>(manifest_path: P) -> Result<Self> {
        let manifest_path = manifest_path.as_ref();
        let root = manifest_path
            .parent()
            .ok_or_else(|| Error::InvalidArchive("manifest path has no parent dir".into()))?
            .to_path_buf();

        let manifest_bytes = fs::read(manifest_path)?;
        let manifest = Manifest::from_json_bytes(&manifest_bytes)?;
        let compression = manifest_open_checks(&manifest)?;
        // formatVersion 3: decode + hash-validate the manifest-embedded schema templates
        // into the registry every tile decode resolves against. Fails the
        // whole open (dataset-level) on any corrupt entry.
        // `manifest_open_checks` above already refused anything that is not
        // PACKED_FORMAT_VERSION, so the registry is unconditional here.
        let templates = Some(build_template_registry(&manifest.schemas)?);

        // Load + decode the directory object (validating + stripping its
        // object magic prelude first). The single (whole-load) shape unwraps the
        // at-rest encoding then runs the v6 codec; the paged shape decodes the
        // root + every leaf (local load-all — a mmap'd file has no cold-start
        // cost). Both branches return the same full entry list.
        let dir_path = root.join(&manifest.directory.key);
        let dir_bytes = fs::read(&dir_path)?;
        if dir_bytes.len() as u64 != manifest.directory.length {
            return Err(Error::InvalidArchive(format!(
                "directory {} is {} bytes, manifest declared {}",
                dir_path.display(),
                dir_bytes.len(),
                manifest.directory.length
            )));
        }
        let codec_bytes = directory_codec_bytes(&dir_bytes, manifest.format_version)?;
        let entries = decode_directory_entries(
            codec_bytes,
            &manifest.directory,
            manifest.metadata.tile_count,
        )?;
        validate_decoded_entries(&manifest, &entries)?;

        // Prepare (lazy) pack handles in pack_id order.
        let packs = manifest
            .packs
            .iter()
            .map(|p| {
                std::cell::RefCell::new(LoadedPack {
                    mmap: None,
                    path: root.join(&p.key),
                    length: p.length,
                })
            })
            .collect();

        Ok(Self {
            entries,
            metadata: manifest.metadata,
            compression,
            packs,
            capabilities: manifest.capabilities,
            format_version: manifest.format_version,
            templates,
            bundle: None,
        })
    }

    /// Open a single-file `.sttb` **bundle** (spec §13, the interchange
    /// profile). The bundle is mmap'd once; the manifest comes verbatim from
    /// the header, each pack becomes an `(offset, length)` window into the
    /// mapping, and everything below [`read_payload`](Self::read_payload) is
    /// shared with the exploded-directory [`open`](Self::open) path —
    /// including the manifest format/version/capability refusals.
    pub fn open_bundle<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: read-only mapping of a file we never write through; the
        // mapping is owned by the reader for its lifetime.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| Error::Other(format!("mmap failed: {e}")))?;
        let (header, header_end) = parse_bundle_header(&mmap)?;
        let manifest = Manifest::from_json_bytes(header.manifest.get().as_bytes())?;
        let compression = manifest_open_checks(&manifest)?;

        // Object table: key → window, every window bounds-checked against the
        // file before anything dereferences it.
        let mut by_key: HashMap<&str, (u64, u64)> = HashMap::with_capacity(header.objects.len());
        for o in &header.objects {
            let end = o.offset.checked_add(o.length).ok_or_else(|| {
                Error::InvalidArchive(format!(
                    "bundle object {:?}: offset+length overflows",
                    o.key
                ))
            })?;
            if o.offset < header_end || end > mmap.len() as u64 {
                return Err(Error::InvalidArchive(format!(
                    "bundle object {:?} window {}..{end} outside the data region ({header_end}..{})",
                    o.key,
                    o.offset,
                    mmap.len()
                )));
            }
            if by_key
                .insert(o.key.as_str(), (o.offset, o.length))
                .is_some()
            {
                return Err(Error::InvalidArchive(format!(
                    "bundle header lists object key {:?} twice",
                    o.key
                )));
            }
        }
        let lookup = |key: &str, declared: u64, what: &str| -> Result<(u64, u64)> {
            let &(offset, length) = by_key.get(key).ok_or_else(|| {
                Error::InvalidArchive(format!("bundle header lists no object for {what} {key:?}"))
            })?;
            if length != declared {
                return Err(Error::InvalidArchive(format!(
                    "{what} {key:?} is {length} bytes in the bundle, manifest declared {declared}"
                )));
            }
            Ok((offset, length))
        };

        // formatVersion 3: registry from the embedded manifest, then per-object magic
        // validation on every window before anything decodes through it.
        // `manifest_open_checks` above already refused anything that is not
        // PACKED_FORMAT_VERSION, so the registry is unconditional here.
        let templates = Some(build_template_registry(&manifest.schemas)?);

        let (dir_off, dir_len) = lookup(
            &manifest.directory.key,
            manifest.directory.length,
            "directory",
        )?;
        let dir_object = &mmap[dir_off as usize..(dir_off + dir_len) as usize];
        let codec_bytes = directory_codec_bytes(dir_object, manifest.format_version)?;
        let entries = decode_directory_entries(
            codec_bytes,
            &manifest.directory,
            manifest.metadata.tile_count,
        )?;
        validate_decoded_entries(&manifest, &entries)?;
        let windows = manifest
            .packs
            .iter()
            .map(|p| lookup(&p.key, p.length, "pack"))
            .collect::<Result<Vec<_>>>()?;
        for (&(off, len), p) in windows.iter().zip(&manifest.packs) {
            strip_object_magic(
                &mmap[off as usize..(off + len) as usize],
                PACK_MAGIC,
                &format!("bundle pack {:?}", p.key),
            )?;
        }

        Ok(Self {
            entries,
            metadata: manifest.metadata,
            compression,
            packs: Vec::new(),
            capabilities: manifest.capabilities,
            format_version: manifest.format_version,
            templates,
            bundle: Some(BundleBacking { mmap, windows }),
        })
    }

    /// All directory entries (sorted by zoom then Hilbert index).
    pub fn entries(&self) -> &[TileEntry] {
        &self.entries
    }

    /// Dataset metadata.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// The manifest's `capabilities` declarations (spec §3.1). A tool that
    /// repacks this dataset's tiles verbatim MUST carry these forward via
    /// [`PackWriter::with_capabilities`] — dropping them re-arms the
    /// silent-misdecode hazard the declaration exists to prevent.
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    /// Read and decompress a tile's raw payload bytes, verifying its CRC32C.
    ///
    /// Selects the pack by `entry.pack_id`, slices `[offset..offset+length]`
    /// within it, checks the CRC, then decompresses the per-blob zstd. Mirrors
    /// `ArchiveReader::read_payload`.
    pub fn read_payload(&self, entry: &TileEntry) -> Result<Vec<u8>> {
        // Bundle-backed reader: the pack is an (offset, length) window into
        // the bundle mapping — slice it, then the shared CRC/decompress tail.
        if let Some(bundle) = &self.bundle {
            let &(pack_offset, pack_length) =
                bundle.windows.get(entry.pack_id as usize).ok_or_else(|| {
                    Error::InvalidArchive(format!(
                        "tile {:?} references pack {} but the bundle holds {} pack(s)",
                        entry.tile_id(),
                        entry.pack_id,
                        bundle.windows.len()
                    ))
                })?;
            let end_in_pack = entry
                .offset
                .checked_add(entry.length as u64)
                .ok_or_else(|| {
                    Error::InvalidArchive(format!(
                        "tile {:?}: blob offset+length overflows",
                        entry.tile_id()
                    ))
                })?;
            if end_in_pack > pack_length {
                return Err(Error::InvalidArchive(format!(
                    "tile {:?} blob range {}..{end_in_pack} exceeds pack size {pack_length}",
                    entry.tile_id(),
                    entry.offset
                )));
            }
            // Windows were bounds-checked against the mapping at open.
            let start = (pack_offset + entry.offset) as usize;
            let end = start + entry.length as usize;
            return self.decode_blob(entry, &bundle.mmap[start..end]);
        }

        let cell = self.packs.get(entry.pack_id as usize).ok_or_else(|| {
            Error::InvalidArchive(format!(
                "tile {:?} references pack {} but only {} packs exist",
                entry.tile_id(),
                entry.pack_id,
                self.packs.len()
            ))
        })?;

        let mut pack = cell.borrow_mut();
        if pack.mmap.is_none() {
            let file = File::open(&pack.path)?;
            // SAFETY: read-only mapping of a file we never write through; the
            // mapping is owned by the reader for its lifetime.
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| Error::Other(format!("mmap failed: {e}")))?;
            if mmap.len() as u64 != pack.length {
                return Err(Error::InvalidArchive(format!(
                    "pack {} is {} bytes, manifest declared {}",
                    pack.path.display(),
                    mmap.len(),
                    pack.length
                )));
            }
            // Pack objects self-identify; validate once per mapping. (Blob
            // offsets are object-absolute, so the slice math below reads past
            // the prelude unchanged.)
            strip_object_magic(&mmap, PACK_MAGIC, &format!("pack {}", pack.path.display()))?;
            pack.mmap = Some(mmap);
        }
        let mmap = pack.mmap.as_ref().expect("just loaded");

        // Compute the exclusive end in u64 first: a corrupt entry offset near
        // u64::MAX would wrap a `usize` add and bypass the length guard, then
        // panic in the slice. Mirror the bundle path above.
        let end = entry
            .offset
            .checked_add(entry.length as u64)
            .ok_or_else(|| {
                Error::InvalidArchive(format!(
                    "tile {:?}: blob offset+length overflows",
                    entry.tile_id()
                ))
            })?;
        if end > mmap.len() as u64 {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} blob range {}..{end} exceeds pack size {}",
                entry.tile_id(),
                entry.offset,
                mmap.len()
            )));
        }
        let start = entry.offset as usize;
        let end = end as usize;
        self.decode_blob(entry, &mmap[start..end])
    }

    /// Shared read tail: CRC32C-check, decompress and size-check one tile's
    /// compressed blob bytes (whichever backing they were sliced from).
    fn decode_blob(&self, entry: &TileEntry, compressed: &[u8]) -> Result<Vec<u8>> {
        if crc32c_tag(compressed) != entry.crc32c {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} failed integrity check (corrupt pack)",
                entry.tile_id()
            )));
        }

        let payload = if self.compression == Compression::Zstd {
            compression::decompress_zstd_with_dict_bounded(
                compressed,
                None,
                entry.uncompressed_size as usize,
            )?
        } else {
            if compressed.len() != entry.uncompressed_size as usize {
                return Err(Error::InvalidArchive(format!(
                    "tile {:?} uncompressed blob is {} bytes, expected {}",
                    entry.tile_id(),
                    compressed.len(),
                    entry.uncompressed_size
                )));
            }
            compression::decompress(compressed, self.compression)?
        };

        if payload.len() != entry.uncompressed_size as usize {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} decompressed to {} bytes, expected {}",
                entry.tile_id(),
                payload.len(),
                entry.uncompressed_size
            )));
        }
        Ok(payload)
    }

    /// Read and decode a tile into its Arrow layers.
    pub fn read_layers(&self, entry: &TileEntry) -> Result<Vec<crate::arrow_tile::DecodedLayer>> {
        let payload = self.read_payload(entry)?;
        self.decode_payload(&payload)
    }

    /// Decode a tile payload. Every open dataset is
    /// [`PACKED_FORMAT_VERSION`] (`manifest_open_checks` refuses anything
    /// else), so a pre-v2 layer frame here means the manifest and the frames
    /// disagree — a hard error, never a silent best-effort decode. Template
    /// references resolve through the registry built at open.
    pub fn decode_payload(&self, payload: &[u8]) -> Result<Vec<crate::arrow_tile::DecodedLayer>> {
        if !crate::arrow_tile::is_frame_v2(payload) {
            return Err(Error::InvalidArchive(
                "pre-v2 layer frame inside a formatVersion-3 dataset (the manifest is \
                 authoritative; mixed-version datasets are invalid)"
                    .into(),
            ));
        }
        let templates = self
            .templates
            .as_ref()
            .expect("open always builds the template registry");
        crate::arrow_tile::decode_tile_with_templates(payload, templates)
    }

    /// The manifest's `formatVersion` — always [`PACKED_FORMAT_VERSION`], since
    /// open refuses anything else.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// The schema-template registry decoded from `manifest.schemas`. Exposed
    /// so tools that fetch payloads through
    /// [`read_payload`](Self::read_payload) can decode them out-of-band via
    /// [`crate::arrow_tile::decode_tile_with_templates`].
    pub fn templates(&self) -> Option<&TemplateRegistry> {
        self.templates.as_ref()
    }
}

// ----------------------------------------------------------------------------
// Bundle profile (`.sttb`) — single-file interchange (spec §13, DRAFT)
// ----------------------------------------------------------------------------

mod bundle;
mod migrate;
pub use migrate::{migrate_dataset_v2_to_v3, MigrationReport};

use self::bundle::parse_bundle_header;
pub use self::bundle::{
    read_bundle_manifest, unpack_bundle, verify_bundle_objects, write_bundle, BundleObject,
    BundleSummary, BUNDLE_MAGIC, BUNDLE_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_tile::{
        encode_tile, encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn,
        PropertyColumn, LAYER_FRAME_VERSION,
    };

    fn point_layer(name: &str, ids: Vec<u64>, t0: i64) -> ColumnarLayer {
        let n = ids.len();
        ColumnarLayer {
            polygon_parts: None,
            name: name.to_string(),
            feature_ids: ids,
            start_times: vec![t0; n],
            end_times: vec![t0 + 100; n],
            geometry: GeometryColumn::Point(vec![[-122.4, 37.7]; n]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        }
    }

    /// Distinct-payload tile so each blob is unique (forces many packs at a tiny
    /// target). `seed` perturbs the feature ids so compressed bytes differ.
    pub(super) fn distinct_tile(seed: u64) -> Vec<u8> {
        let ids: Vec<u64> = (0..6).map(|i| seed * 100 + i).collect();
        encode_tile(&[point_layer("default", ids, (seed as i64) * 7)]).unwrap()
    }

    /// Plain writer for the tests whose payloads are bare `encode_tile` frames
    /// (or opaque bytes) rather than a full template-collecting build.
    pub(super) fn test_writer(
        out: &Path,
        ordering: BlobOrdering,
        pack_target_bytes: u64,
    ) -> PackWriter {
        PackWriter::create(out, ordering, pack_target_bytes).unwrap()
    }

    #[test]
    fn blake3_128_hex_is_32_chars() {
        let h = blake3_128_hex(b"hello");
        assert_eq!(h.len(), 32);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn manifest_json_roundtrips() {
        let m = Manifest {
            format: PACKED_FORMAT.to_string(),
            format_version: PACKED_FORMAT_VERSION,
            capabilities: Vec::new(),
            schemas: Vec::new(),
            variants: vec![ManifestVariant {
                id: crate::tile::RAW_VARIANT_ID,
                kind: VariantKind::Raw,
                layer_name: None,
                method: None,
                params: None,
            }],
            compression: "zstd".to_string(),
            blob_ordering: None,
            adaptive_boundaries: Vec::new(),
            ordering_workload: None,
            directory: DirectoryRef {
                key: "index/abc.sttd".to_string(),
                length: 42,
                directory_version: crate::directory::DIRECTORY_VERSION,
                encoding: Some(DIRECTORY_ENCODING_ZSTD.to_string()),
                layout: None,
                root_length: None,
                page_count: None,
                page_entries: None,
                root_hash: None,
                page_hashes: None,
            },
            packs: vec![
                PackRef {
                    key: "packs/a.sttp".to_string(),
                    length: 100,
                },
                PackRef {
                    key: "packs/b.sttp".to_string(),
                    length: 200,
                },
            ],
            metadata: Metadata::new("manifest-test"),
        };
        let bytes = m.to_json_bytes().unwrap();
        // The spec keys must be camelCase where renamed.
        let s = String::from_utf8(bytes.clone()).unwrap();
        assert!(s.contains("\"formatVersion\""), "{s}");
        assert!(s.contains("\"directoryVersion\""), "{s}");
        assert!(s.contains("\"stt-packed\""), "{s}");
        // Empty capabilities are OMITTED — every pre-capabilities manifest
        // (and every non-quantized build) stays byte-identical.
        assert!(!s.contains("\"capabilities\""), "{s}");
        // blob_ordering None is OMITTED too, so pre-field builds stay byte-identical.
        assert!(!s.contains("\"blobOrdering\""), "{s}");
        let back = Manifest::from_json_bytes(&bytes).unwrap();
        assert_eq!(back.format, m.format);
        assert_eq!(back.format_version, m.format_version);
        assert_eq!(back.packs.len(), 2);
        assert_eq!(
            back.directory.directory_version,
            crate::directory::DIRECTORY_VERSION
        );
        assert_eq!(back.directory.encoding.as_deref(), Some("zstd"));
        assert_eq!(back.metadata.name, "manifest-test");

        // A raw directory omits the encoding key.
        let mut legacy_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        legacy_json["directory"]
            .as_object_mut()
            .unwrap()
            .remove("encoding");
        let legacy_back =
            Manifest::from_json_bytes(&serde_json::to_vec(&legacy_json).unwrap()).unwrap();
        assert_eq!(legacy_back.directory.encoding, None);
        // An omitted capabilities key remains the canonical empty list.
        assert!(legacy_back.capabilities.is_empty());
        let legacy_out = String::from_utf8(legacy_back.to_json_bytes().unwrap()).unwrap();
        assert!(!legacy_out.contains("\"encoding\""), "{legacy_out}");
    }

    /// Full round-trip: 30 synthetic tiles (a couple byte-identical to exercise
    /// dedup) across 2 zooms + several time buckets, written with a tiny pack
    /// target to force multiple packs. Every payload must decode byte-identical.
    #[test]
    fn packwriter_roundtrips_through_multiple_packs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = test_writer(&out, BlobOrdering::Auto, 8 * 1024);

        // Build 30 tiles. Tiles 0..28 distinct; the static one is reused so two
        // entries point at one byte-identical blob (dedup).
        let static_payload = encode_tile(&[point_layer("default", vec![7, 8, 9], 0)]).unwrap();
        let mut expected: Vec<(TileId, i64, i64, Vec<u8>)> = Vec::new();
        let bucket = 3_600_000i64;
        for k in 0..30u64 {
            let zoom = if k % 2 == 0 { 9u8 } else { 10u8 };
            let b = (k % 5) as i64; // several time buckets
            let t = b * bucket;
            let payload = if k == 13 || k == 27 {
                static_payload.clone() // two byte-identical → dedup
            } else {
                distinct_tile(k)
            };
            let id = TileId::new(zoom, (k % 7) as u32, (k / 7) as u32, t as u64);
            w.add_tile_full(
                &id,
                t,
                t + bucket - 1,
                Some(t),
                6,
                Some(bucket as u64),
                &payload,
            )
            .unwrap();
            expected.push((id, t, t + bucket - 1, payload));
        }
        assert_eq!(w.tile_count(), 30);

        let meta = Metadata::new("packed-roundtrip").with_temporal_bucket_ms(bucket as u64);
        let manifest = w.finalize(&meta).unwrap();

        // >1 pack produced at the tiny target.
        assert!(
            manifest.packs.len() > 1,
            "expected multiple packs, got {}",
            manifest.packs.len()
        );

        // Every pack file ≤ target, except a lone oversized blob owning a pack.
        for p in &manifest.packs {
            let bytes = fs::read(out.join(&p.key)).unwrap();
            assert_eq!(bytes.len() as u64, p.length);
            // A pack over target must contain exactly one blob (oversized loner).
            // We can't see blob boundaries here, but the cut rule guarantees a
            // pack only exceeds the target when it holds a single blob; the
            // round-trip below proves correctness regardless.
        }

        // Pack/dir filenames are blake3-128 hex of their bytes.
        for p in &manifest.packs {
            let bytes = fs::read(out.join(&p.key)).unwrap();
            let hex = blake3_128_hex(&bytes);
            assert_eq!(p.key, format!("packs/{hex}.sttp"));
        }
        let dir_bytes = fs::read(out.join(&manifest.directory.key)).unwrap();
        assert_eq!(
            manifest.directory.key,
            format!("index/{}.sttd", blake3_128_hex(&dir_bytes))
        );

        // pack_ids are contiguous from 0 across all entries.
        let reader = PackedReader::open(out.join("manifest.json")).unwrap();
        let max_pack = reader.entries().iter().map(|e| e.pack_id).max().unwrap();
        assert_eq!(max_pack as usize, manifest.packs.len() - 1);
        let observed: std::collections::BTreeSet<u32> =
            reader.entries().iter().map(|e| e.pack_id).collect();
        for pid in 0..manifest.packs.len() as u32 {
            assert!(observed.contains(&pid), "pack_id {pid} unused");
        }

        // Metadata round-trips.
        assert_eq!(reader.metadata().name, "packed-roundtrip");
        assert_eq!(reader.metadata().temporal_bucket_ms, bucket as u64);
        assert_eq!(reader.entries().len(), 30);

        // The manifest totals are derived from the directory at finalize —
        // the caller's Metadata left them 0 and finalize must overwrite them.
        assert_eq!(manifest.metadata.tile_count, 30);
        assert_eq!(manifest.metadata.feature_count, 30 * 6);
        assert_eq!(reader.metadata().tile_count, 30);
        assert_eq!(reader.metadata().feature_count, 30 * 6);

        // Every tile's decompressed payload is byte-identical.
        for (id, ts, _te, payload) in &expected {
            let e = reader
                .entries()
                .iter()
                .find(|e| e.zoom == id.z && e.x == id.x && e.y == id.y && e.time_start == *ts)
                .expect("entry present");
            let got = reader.read_payload(e).unwrap();
            assert_eq!(&got, payload, "payload mismatch for {id:?}");
        }

        // Dedup: the two static tiles share one (pack, offset).
        let static_entries: Vec<&TileEntry> = reader
            .entries()
            .iter()
            .filter(|e| {
                let g = reader.read_payload(e).unwrap();
                g == static_payload
            })
            .collect();
        assert_eq!(static_entries.len(), 2);
        assert_eq!(static_entries[0].pack_id, static_entries[1].pack_id);
        assert_eq!(static_entries[0].offset, static_entries[1].offset);
    }

    /// `auto` measures the OCCUPIED spatial extent, not the raw max zoom. A
    /// dataset at a high zoom but a tiny spatial bbox over a deep-ish timeline
    /// resolves to spatial-major — which only holds when `choose` sees the
    /// occupied bbox (~2 bits) rather than the 14-bit zoom (which would pick
    /// hilbert3). The resolved order is also recorded in the manifest.
    #[test]
    fn auto_resolves_from_occupied_extent_not_max_zoom() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("f1");
        let mut w = test_writer(&out, BlobOrdering::Auto, 1 << 20);
        let bucket = 3_600_000i64;
        for x in 0..4u32 {
            for b in 0..64i64 {
                let t = b * bucket;
                let id = TileId::new(14, 10_000 + x, 20_000, t as u64);
                let payload = distinct_tile(u64::from(x) * 1_000 + b as u64);
                w.add_tile_full(&id, t, t + bucket - 1, Some(t), 1, None, &payload)
                    .unwrap();
            }
        }
        let meta = Metadata::new("f1").with_temporal_bucket_ms(bucket as u64);
        let manifest = w.finalize(&meta).unwrap();
        // occupied space bits = bits_for(4) = 2; time bits = bits_for(64) = 6;
        // 6 > 2 + 3 → spatial. (Old raw-max-zoom rule: choose(14, 6) → hilbert3.)
        assert_eq!(manifest.blob_ordering.as_deref(), Some("spatial"));
    }

    /// The opt-in measured picker resolves to a concrete, deterministic order
    /// and records it in the manifest (never `auto`/`measured`).
    #[test]
    fn measured_ordering_records_concrete_order() {
        let build = || {
            let dir = tempfile::tempdir().unwrap();
            let out = dir.path().join("m");
            let mut w = test_writer(&out, BlobOrdering::Auto, 1 << 20).with_measured_ordering(true);
            let bucket = 3_600_000i64;
            for x in 0..2u32 {
                for b in 0..24i64 {
                    let t = b * bucket;
                    let id = TileId::new(10, 5_000 + x, 6_000, t as u64);
                    let payload = distinct_tile(u64::from(x) * 1_000 + b as u64);
                    w.add_tile_full(&id, t, t + bucket - 1, Some(t), 1, None, &payload)
                        .unwrap();
                }
            }
            let meta = Metadata::new("m").with_temporal_bucket_ms(bucket as u64);
            w.finalize(&meta).unwrap().blob_ordering
        };
        let a = build();
        let b = build();
        assert_eq!(a, b, "measured resolution must be deterministic");
        let o = a.expect("measured records a concrete order");
        assert!(matches!(
            o.as_str(),
            "spatial" | "time-major" | "hilbert3" | "morton3"
        ));
    }

    /// Below the simulation floor there is nothing to measure — the canonical
    /// band collapses to one or two cells and the ranking is decided by
    /// rounding — so `measured` degrades to `auto`'s cardinality heuristic and
    /// records NO workload (there was no simulation to co-version).
    #[test]
    fn measured_below_the_simulation_floor_falls_back_to_auto() {
        let bucket = 3_600_000i64;
        // Same input built twice: once via `measured` (which must give up), once
        // via `auto` outright. Six tiles — under MIN_TILES_TO_SIMULATE = 8.
        let build = |out: &Path, measured: bool| {
            let mut w =
                test_writer(out, BlobOrdering::Auto, 1 << 20).with_measured_ordering(measured);
            for b in 0..6i64 {
                let t = b * bucket;
                let id = TileId::new(9, 100, 200 + b as u32, t as u64);
                let payload = distinct_tile(b as u64);
                w.add_tile_full(&id, t, t + bucket - 1, Some(t), 1, None, &payload)
                    .unwrap();
            }
            let meta = Metadata::new("floor")
                .with_temporal_bucket_ms(bucket as u64)
                .with_zoom_levels(0, 9);
            w.finalize(&meta).unwrap()
        };
        let dir = tempfile::tempdir().unwrap();
        let measured = build(&dir.path().join("m"), true);
        let auto = build(&dir.path().join("a"), false);
        assert!(crate::ordering_sim::MIN_TILES_TO_SIMULATE > 6);
        assert_eq!(
            measured.blob_ordering, auto.blob_ordering,
            "below the floor, measured must resolve to auto's choice"
        );
        assert_eq!(
            measured.metadata.ordering_workload, None,
            "a fallback pick has no workload to record"
        );
        // The whole manifest is byte-identical, so the fallback really is the
        // incumbent path and not a look-alike.
        assert_eq!(
            measured.to_json_bytes().unwrap(),
            auto.to_json_bytes().unwrap()
        );
    }

    /// The workload block is written ONLY by a simulated ordering, so every
    /// `auto`/explicit build — which is every golden fixture and every archive
    /// in the published fleet — emits byte-identical manifest JSON to a build
    /// that predates the field.
    #[test]
    fn ordering_workload_is_omitted_unless_the_ordering_was_simulated() {
        let bucket = 3_600_000i64;
        let build = |out: &Path, ordering: BlobOrdering, measured: bool| {
            let mut w = test_writer(out, ordering, 1 << 20).with_measured_ordering(measured);
            for x in 0..3u32 {
                for b in 0..8i64 {
                    let t = b * bucket;
                    let id = TileId::new(10, 700 + x, 800, t as u64);
                    let payload = distinct_tile(u64::from(x) * 100 + b as u64);
                    w.add_tile_full(&id, t, t + bucket - 1, Some(t), 1, None, &payload)
                        .unwrap();
                }
            }
            let meta = Metadata::new("omit")
                .with_temporal_bucket_ms(bucket as u64)
                .with_zoom_levels(0, 10);
            w.finalize(&meta).unwrap()
        };
        let dir = tempfile::tempdir().unwrap();
        for (name, ordering) in [
            ("auto", BlobOrdering::Auto),
            ("spatial", BlobOrdering::SpatialMajor),
            ("time-major", BlobOrdering::TimeMajor),
            ("hilbert3", BlobOrdering::Hilbert3),
        ] {
            let m = build(&dir.path().join(name), ordering, false);
            assert_eq!(m.metadata.ordering_workload, None, "{name}");
            let json = String::from_utf8(m.to_json_bytes().unwrap()).unwrap();
            assert!(
                !json.contains("orderingWorkload"),
                "{name} emitted the key: {json}"
            );
        }
        // ...and a measured build DOES record it, at BOTH pinned keys.
        let m = build(&dir.path().join("measured"), BlobOrdering::Auto, true);
        let w = m
            .ordering_workload
            .expect("measured records a workload top-level");
        assert_eq!(
            m.metadata.ordering_workload,
            Some(w),
            "the metadata mirror must carry the identical object"
        );
        assert_eq!(
            w.coalesce_gap_bytes,
            crate::ordering_sim::DEFAULT_COALESCE_GAP_BYTES
        );
        assert_eq!(
            w.runway_multiplier,
            crate::ordering_sim::DEFAULT_RUNWAY_MULTIPLIER
        );
        // Top-level key is camelCase (`orderingWorkload`, beside `blobOrdering`);
        // the mirror inside the folded `metadata` block is snake_case like every
        // other key there. The OBJECT's own keys are snake_case in both places,
        // so the two copies are byte-identical JSON.
        let json = String::from_utf8(m.to_json_bytes().unwrap()).unwrap();
        assert!(json.contains("\"orderingWorkload\""), "{json}");
        assert!(json.contains("\"ordering_workload\""), "{json}");
        assert!(json.contains("\"coalesce_gap_bytes\""), "{json}");
        assert!(json.contains("\"playback_window_buckets\""), "{json}");
        assert!(json.contains("\"runway_multiplier\""), "{json}");
    }

    /// SH-3 / WM-2: the workload is co-versioned at **two** manifest keys and
    /// the two must never disagree.
    ///
    /// `Manifest::ordering_workload` (top-level `orderingWorkload`, beside the
    /// `blobOrdering` it co-versions) is canonical. `metadata.ordering_workload`
    /// is the reader-compat mirror the shipped TS reader resolves the
    /// build-assumed coalescing gap through
    /// (`archive.ts::manifestBuildAssumedGapBytes`) — deleting it would silently
    /// disable the adaptive-coalesce co-versioning guard instead of failing
    /// loudly, so this test is what keeps the mirror honest until the reader
    /// moves. Both survive a JSON round-trip: an audit reads them BACK off disk.
    #[test]
    fn manifest_records_the_ordering_workload_at_both_pinned_keys() {
        let bucket = 3_600_000i64;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("both");
        let mut w = test_writer(&out, BlobOrdering::Auto, 1 << 20).with_measured_ordering(true);
        for x in 0..3u32 {
            for b in 0..8i64 {
                let t = b * bucket;
                let id = TileId::new(10, 700 + x, 800, t as u64);
                let payload = distinct_tile(u64::from(x) * 100 + b as u64);
                w.add_tile_full(&id, t, t + bucket - 1, Some(t), 1, None, &payload)
                    .unwrap();
            }
        }
        let meta = Metadata::new("both")
            .with_temporal_bucket_ms(bucket as u64)
            .with_zoom_levels(0, 10);
        let manifest = w.finalize(&meta).unwrap();

        let top = manifest.ordering_workload.expect("top-level key");
        assert_eq!(manifest.metadata.ordering_workload, Some(top));

        // Round-trip through the on-disk bytes — the audit path.
        let bytes = std::fs::read(out.join("manifest.json")).unwrap();
        let reparsed = Manifest::from_json_bytes(&bytes).unwrap();
        assert_eq!(reparsed.ordering_workload, Some(top));
        assert_eq!(reparsed.metadata.ordering_workload, Some(top));

        // The two serialized objects are literally the same JSON value.
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["orderingWorkload"], v["metadata"]["ordering_workload"]);
        assert!(v["orderingWorkload"]["coalesce_gap_bytes"].is_u64());
    }

    /// A **paged** directory build round-trips end-to-end through
    /// `PackedReader`: every input tile's payload decodes byte-identically, the
    /// manifest carries the container fields, and content-address verification
    /// is clean. Separately, a paged and a single build of the same input must
    /// agree byte-for-byte on everything except the directory object's
    /// container shape: identical pack content addresses and identical decoded
    /// entries. Cross-finalize blob bytes ARE reproducible — the encoder feeds
    /// sorted metadata to Arrow ≥59's sorted-order IPC writer, guarded by
    /// `reproducible_build.rs` — so this compares the content addresses
    /// themselves, not a weaker key-set match.
    #[test]
    fn paged_directory_writer_roundtrips_and_matches_single() {
        let bucket = 3_600_000i64;
        // Deterministic input tiles: (id, time_start, time_end, payload).
        let mut input: Vec<(TileId, i64, i64, Vec<u8>)> = Vec::new();
        for k in 0..120u64 {
            let zoom = [6u8, 10, 13][(k % 3) as usize];
            let b = (k % 4) as i64;
            let t = b * bucket;
            let id = TileId::new(zoom, (k % 11) as u32, (k / 11) as u32, t as u64);
            input.push((id, t, t + bucket - 1, distinct_tile(k)));
        }
        let build = |out: &Path, page_entries: Option<usize>| -> Manifest {
            let mut w = test_writer(out, BlobOrdering::Auto, 16 * 1024).with_paging(page_entries);
            for (id, ts, te, payload) in &input {
                w.add_tile_full(id, *ts, *te, Some(*ts), 6, Some(bucket as u64), payload)
                    .unwrap();
            }
            let meta = Metadata::new("paged-roundtrip").with_temporal_bucket_ms(bucket as u64);
            w.finalize(&meta).unwrap()
        };

        let dir = tempfile::tempdir().unwrap();
        let single_out = dir.path().join("single");
        let paged_out = dir.path().join("paged");
        let single = build(&single_out, None);
        // Small page size to force several leaf pages over 120 entries.
        let paged = build(&paged_out, Some(16));

        // Paged manifest carries the container fields; single does not.
        assert!(single.directory.layout.is_none());
        assert_eq!(
            paged.directory.layout.as_deref(),
            Some(DIRECTORY_LAYOUT_PAGED)
        );
        assert!(paged.directory.root_length.unwrap() > 0);
        assert!(
            paged.directory.page_count.unwrap() >= 2,
            "expected multiple leaf pages"
        );
        assert_eq!(paged.directory.page_entries, Some(16));
        assert!(paged.directory.root_hash.is_some());
        assert_eq!(
            paged.directory.page_hashes.as_ref().unwrap().len() as u64,
            paged.directory.page_count.unwrap()
        );
        assert_eq!(
            paged.directory.directory_version,
            crate::directory::DIRECTORY_VERSION
        );
        assert_eq!(
            paged.directory.encoding.as_deref(),
            Some(DIRECTORY_ENCODING_ZSTD)
        );

        let r_single = PackedReader::open(single_out.join("manifest.json")).unwrap();
        let r_paged = PackedReader::open(paged_out.join("manifest.json")).unwrap();
        assert_eq!(r_paged.entries().len(), 120);
        assert_eq!(r_single.entries().len(), 120);

        // Cross-build agreement is byte-level: identical pack content addresses
        // (blob bytes are reproducible across finalize runs) and identical
        // decoded directory entries — the paged build differs from the single
        // build ONLY in the directory object's container shape.
        assert_eq!(
            single
                .packs
                .iter()
                .map(|p| (&p.key, p.length))
                .collect::<Vec<_>>(),
            paged
                .packs
                .iter()
                .map(|p| (&p.key, p.length))
                .collect::<Vec<_>>(),
            "pack content addresses must match across single vs paged builds"
        );
        assert_eq!(r_single.entries(), r_paged.entries());

        // Every INPUT tile decodes byte-identically through the paged reader.
        for (id, ts, _te, payload) in &input {
            let e = r_paged
                .entries()
                .iter()
                .find(|x| x.zoom == id.z && x.x == id.x && x.y == id.y && x.time_start == *ts)
                .expect("paged entry present");
            assert_eq!(
                &r_paged.read_payload(e).unwrap(),
                payload,
                "payload mismatch {id:?}"
            );
        }

        // Content-address integrity verifies clean on the paged dataset.
        let issues = verify_packed_objects(paged_out.join("manifest.json")).unwrap();
        assert!(issues.is_empty(), "paged verify issues: {issues:?}");

        // The v3 cutover requires independent page authentication; there is
        // no unhashed paged-directory compatibility shape.
        let mut unhashed = paged.clone();
        unhashed.directory.root_hash = None;
        unhashed.directory.page_hashes = None;
        fs::write(
            paged_out.join("manifest.json"),
            unhashed.to_json_bytes().unwrap(),
        )
        .unwrap();
        let error = PackedReader::open(paged_out.join("manifest.json"))
            .err()
            .expect("unhashed paged v3 manifest must be rejected");
        assert!(error.to_string().contains("rootHash"));
    }

    #[test]
    fn adaptive_paging_uses_single_below_threshold_and_paged_at_threshold() {
        let build = |out: &Path, min_entries: usize| {
            let mut writer = test_writer(out, BlobOrdering::SpatialMajor, 1 << 20)
                .with_adaptive_paging(2, min_entries);
            for x in 0..4u32 {
                let id = TileId::new(2, x, 0, 0);
                writer
                    .add_tile_full(&id, 0, 999, Some(0), 1, None, &distinct_tile(x as u64))
                    .unwrap();
            }
            writer
                .finalize(&Metadata::new("adaptive").with_temporal_bucket_ms(1_000))
                .unwrap()
        };

        let dir = tempfile::tempdir().unwrap();
        let single = build(&dir.path().join("single"), 5);
        let paged = build(&dir.path().join("paged"), 4);
        assert!(single.directory.layout.is_none());
        assert_eq!(
            paged.directory.layout.as_deref(),
            Some(DIRECTORY_LAYOUT_PAGED)
        );
        assert_eq!(paged.directory.page_count, Some(2));
    }

    #[test]
    fn paged_frame_hash_rejects_tampered_leaf_before_decode() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut writer = test_writer(&out, BlobOrdering::Auto, 64 * 1024).with_paging(Some(2));
        for k in 0..4u64 {
            writer
                .add_tile_full(
                    &TileId::new(8, k as u32, 0, 0),
                    0,
                    100,
                    None,
                    6,
                    None,
                    &distinct_tile(k),
                )
                .unwrap();
        }
        let manifest = writer.finalize(&Metadata::new("page-hash")).unwrap();
        let path = out.join(&manifest.directory.key);
        let mut object = fs::read(&path).unwrap();
        let first_leaf = OBJECT_MAGIC_LEN + manifest.directory.root_length.unwrap() as usize;
        object[first_leaf] ^= 0x01;
        fs::write(path, object).unwrap();

        let error = PackedReader::open(out.join("manifest.json"))
            .err()
            .expect("tampered page must fail");
        assert!(
            error.to_string().contains("leaf 0 hash mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn open_rejects_unsafe_manifest_key_before_resolving_it() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut writer = test_writer(&out, BlobOrdering::Auto, 64 * 1024);
        writer
            .add_tile_full(
                &TileId::new(8, 0, 0, 0),
                0,
                100,
                None,
                6,
                None,
                &distinct_tile(0),
            )
            .unwrap();
        let mut manifest = writer.finalize(&Metadata::new("unsafe-key")).unwrap();
        manifest.directory.key = "../outside.sttd".into();
        fs::write(out.join("manifest.json"), manifest.to_json_bytes().unwrap()).unwrap();

        let error = PackedReader::open(out.join("manifest.json"))
            .err()
            .expect("unsafe key must fail");
        assert!(
            error.to_string().contains("unsafe object key"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn open_rejects_manifest_layout_and_version_drift() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut writer = test_writer(&out, BlobOrdering::Auto, 64 * 1024);
        writer
            .add_tile_full(
                &TileId::new(8, 0, 0, 0),
                0,
                100,
                None,
                6,
                None,
                &distinct_tile(0),
            )
            .unwrap();
        let manifest = writer.finalize(&Metadata::new("semantics")).unwrap();
        let manifest_path = out.join("manifest.json");

        let mut wrong_version = manifest.clone();
        wrong_version.directory.directory_version = 4;
        fs::write(&manifest_path, wrong_version.to_json_bytes().unwrap()).unwrap();
        let error = PackedReader::open(&manifest_path)
            .err()
            .expect("directoryVersion 4 must fail");
        assert!(error.to_string().contains("directoryVersion"));

        let mut unknown_layout = manifest.clone();
        unknown_layout.directory.layout = Some("future-layout".into());
        fs::write(&manifest_path, unknown_layout.to_json_bytes().unwrap()).unwrap();
        let error = PackedReader::open(&manifest_path)
            .err()
            .expect("unknown layout must fail");
        assert!(error.to_string().contains("unknown directory layout"));

        let mut no_packs = manifest;
        no_packs.packs.clear();
        fs::write(&manifest_path, no_packs.to_json_bytes().unwrap()).unwrap();
        let error = PackedReader::open(&manifest_path)
            .err()
            .expect("empty pack table must fail");
        assert!(error.to_string().contains("at least one pack"));
    }

    /// A single blob larger than the pack target gets its own pack (never split).
    #[test]
    fn oversized_blob_gets_its_own_pack() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        // One big tile (lots of distinct points → big compressed blob) plus a
        // few small ones, with a target smaller than the big blob.
        let big_ids: Vec<u64> = (0..4000).collect();
        let big = encode_tile(&[ColumnarLayer {
            polygon_parts: None,
            name: "default".to_string(),
            feature_ids: big_ids.clone(),
            start_times: vec![0; big_ids.len()],
            end_times: vec![100; big_ids.len()],
            geometry: GeometryColumn::Point(
                (0..big_ids.len())
                    .map(|i| [i as f64 * 0.01, i as f64 * 0.013])
                    .collect(),
            ),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        }])
        .unwrap();
        let big_compressed_len = compression::compress_zstd_with_dict(&big, None)
            .unwrap()
            .len();

        let mut w = test_writer(&out, BlobOrdering::SpatialMajor, 4 * 1024);
        // target 4 KiB < big blob.
        assert!(big_compressed_len as u64 > 4 * 1024);
        w.add_tile_full(&TileId::new(10, 0, 0, 0), 0, 100, None, 4000, None, &big)
            .unwrap();
        for k in 1..4u64 {
            let p = distinct_tile(k);
            w.add_tile_full(&TileId::new(10, k as u32, 0, 0), 0, 100, None, 6, None, &p)
                .unwrap();
        }
        let _manifest = w.finalize(&Metadata::new("big")).unwrap();

        // At least one pack exceeds the target (the loner). Reading proves it.
        let reader = PackedReader::open(out.join("manifest.json")).unwrap();
        let big_entry = reader.entries().iter().find(|e| e.x == 0).unwrap();
        assert_eq!(reader.read_payload(big_entry).unwrap(), big);
    }

    /// Read-path integrity: flipping a byte inside a pack object makes
    /// [`PackedReader::read_payload`] fail the per-blob CRC32C check instead of
    /// returning silently-wrong bytes. Guards the integrity check in
    /// `read_payload` (replaces the archive-era `corrupt_blob_is_detected`).
    #[test]
    fn corrupt_pack_blob_is_detected_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = test_writer(&out, BlobOrdering::Auto, 64 * 1024 * 1024);
        for k in 0..4u64 {
            let p = distinct_tile(k);
            w.add_tile_full(&TileId::new(10, k as u32, 0, 0), 0, 100, None, 6, None, &p)
                .unwrap();
        }
        let manifest = w.finalize(&Metadata::new("crc")).unwrap();

        // Clean read before corruption.
        let entry = PackedReader::open(out.join("manifest.json"))
            .unwrap()
            .entries()[0]
            .clone();
        assert!(PackedReader::open(out.join("manifest.json"))
            .unwrap()
            .read_payload(&entry)
            .is_ok());

        // Flip the first byte of that tile's compressed blob inside its pack.
        let pack_path = out.join(&manifest.packs[entry.pack_id as usize].key);
        let mut bytes = fs::read(&pack_path).unwrap();
        bytes[entry.offset as usize] ^= 0xff;
        fs::write(&pack_path, &bytes).unwrap();

        // A fresh reader must now reject that tile's payload (CRC32C mismatch).
        let reader = PackedReader::open(out.join("manifest.json")).unwrap();
        assert!(
            reader.read_payload(&entry).is_err(),
            "corrupt pack blob must fail the read-path CRC32C check"
        );
    }

    /// A corrupt entry whose `offset` sits near `u64::MAX` must make
    /// `read_payload` return `Err`, not overflow the `offset + length` slice
    /// math and panic. Guards the checked-add in the non-bundle read path.
    #[test]
    fn read_payload_offset_overflow_errors() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = test_writer(&out, BlobOrdering::Auto, 64 * 1024 * 1024);
        w.add_tile_full(
            &TileId::new(10, 0, 0, 0),
            0,
            100,
            None,
            6,
            None,
            &distinct_tile(0),
        )
        .unwrap();
        w.finalize(&Metadata::new("ovf")).unwrap();

        let reader = PackedReader::open(out.join("manifest.json")).unwrap();
        let mut entry = reader.entries()[0].clone();
        // offset + length overflows u64: a wrapping `usize` add would slip past
        // the `end > pack.len()` guard into a slice panic pre-fix.
        entry.offset = u64::MAX - 3;
        entry.length = 16;
        assert!(
            reader.read_payload(&entry).is_err(),
            "an entry offset near u64::MAX must error, not panic"
        );
    }

    /// Two builds of the same input — added in different orders, including
    /// curve-key ties (a base + temporal-LOD entry on one cell) — must produce
    /// byte-identical objects: same pack hashes, same directory hash. This is
    /// the immutable-pack CDN contract (a rebuild of unchanged data must not
    /// invalidate the edge cache).
    #[test]
    fn rebuilds_are_byte_reproducible() {
        // Tiles including a tie pair: same (z, x, y, time_start), one base
        // (bucket None) and one LOD (bucket Some) — the curve key alone can't
        // order them, the tiebreak must.
        let bucket = 3_600_000i64;
        let mut tiles: Vec<(TileId, i64, Option<u64>, Vec<u8>)> = Vec::new();
        for k in 0..10u64 {
            let t = (k % 4) as i64 * bucket;
            tiles.push((
                TileId::new(9, (k % 5) as u32, (k / 5) as u32, t as u64),
                t,
                None,
                distinct_tile(k),
            ));
        }
        // The tie pair on cell (1, 0): base + LOD aggregate at one time_start.
        tiles.push((TileId::new(9, 1, 0, 0), 0, None, distinct_tile(100)));
        tiles.push((
            TileId::new(9, 1, 0, 0),
            0,
            Some(24 * bucket as u64),
            distinct_tile(101),
        ));

        let meta = Metadata::new("repro").with_temporal_bucket_ms(bucket as u64);
        let build = |order: &[usize]| {
            let dir = tempfile::tempdir().unwrap();
            let out = dir.path().join("dataset");
            let mut w = test_writer(&out, BlobOrdering::Auto, 8 * 1024);
            for &i in order {
                let (id, t, b, payload) = &tiles[i];
                w.add_tile_full(id, *t, t + bucket - 1, Some(*t), 6, *b, payload)
                    .unwrap();
            }
            let manifest = w.finalize(&meta).unwrap();
            (dir, manifest)
        };

        let forward: Vec<usize> = (0..tiles.len()).collect();
        let reverse: Vec<usize> = (0..tiles.len()).rev().collect();
        let (_d1, m1) = build(&forward);
        let (_d2, m2) = build(&reverse);

        assert_eq!(
            m1.directory.key, m2.directory.key,
            "directory hash must be stable"
        );
        assert_eq!(
            m1.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            m2.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            "pack hashes must be stable across rebuilds"
        );
        assert_eq!(m1.to_json_bytes().unwrap(), m2.to_json_bytes().unwrap());
    }

    /// Spilling is a MEMORY-behaviour lever only: a build whose payloads spill
    /// to disk (tiny budget) must produce byte-identical objects — same pack
    /// hashes, same directory hash, same manifest — as the unlimited all-in-RAM
    /// build, and the temp spill file must be gone afterwards (success path)
    /// as well as when the writer is dropped without finalize (abandon path).
    #[test]
    fn spilled_build_is_byte_identical_to_in_memory() {
        let bucket = 3_600_000i64;
        let mut tiles: Vec<(TileId, i64, Option<u64>, Vec<u8>)> = Vec::new();
        for k in 0..40u64 {
            let t = (k % 4) as i64 * bucket;
            tiles.push((
                TileId::new(9, (k % 8) as u32, (k / 8) as u32, t as u64),
                t,
                None,
                distinct_tile(k),
            ));
        }
        // Duplicate payload pair so dedup is exercised across the spill.
        tiles.push((TileId::new(9, 7, 7, 0), 0, None, distinct_tile(3)));

        let meta = Metadata::new("spill").with_temporal_bucket_ms(bucket as u64);
        let build = |budget: u64| {
            let dir = tempfile::tempdir().unwrap();
            let out = dir.path().join("dataset");
            let mut w = test_writer(&out, BlobOrdering::Auto, 8 * 1024).with_memory_budget(budget);
            for (id, t, b, payload) in &tiles {
                w.add_tile_full(id, *t, t + bucket - 1, Some(*t), 6, *b, payload)
                    .unwrap();
            }
            let manifest = w.finalize(&meta).unwrap();
            // No `.spill-*` residue in the output dir on success.
            let residue: Vec<_> = fs::read_dir(&out)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with(".spill-"))
                .collect();
            assert!(residue.is_empty(), "spill file left behind: {residue:?}");
            (dir, manifest)
        };

        // 1 KiB budget: virtually every payload spills. 0 = legacy unlimited.
        let (d1, spilled) = build(1024);
        let (_d2, in_mem) = build(0);
        assert_eq!(
            spilled.directory.key, in_mem.directory.key,
            "directory hash must not depend on the payload storage medium"
        );
        assert_eq!(
            spilled.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            in_mem.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            "pack hashes must not depend on the payload storage medium"
        );
        assert_eq!(
            spilled.to_json_bytes().unwrap(),
            in_mem.to_json_bytes().unwrap()
        );

        // Multi-pack at the tiny target, so the STREAMING pack phase
        // (incremental blake3, per-seal temp→content-address renames) crossed
        // pack cuts under both budgets and still matched byte-for-byte; its
        // sealed objects verify clean and leave no `.tmp-*` residue.
        assert!(spilled.packs.len() > 1, "fixture must span multiple packs");
        let spilled_out = d1.path().join("dataset");
        assert!(verify_packed_objects(spilled_out.join("manifest.json"))
            .unwrap()
            .is_empty());
        let tmp_residue: Vec<_> = fs::read_dir(spilled_out.join("packs"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp-"))
            .collect();
        assert!(
            tmp_residue.is_empty(),
            "streaming pack temp left behind: {tmp_residue:?}"
        );

        // Abandon path: dropping a writer that spilled (no finalize) must
        // remove the spill file too.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        {
            let mut w = test_writer(&out, BlobOrdering::Auto, 8 * 1024).with_memory_budget(1);
            for (id, t, b, payload) in tiles.iter().take(4) {
                w.add_tile_full(id, *t, t + bucket - 1, Some(*t), 6, *b, payload)
                    .unwrap();
            }
            // Spill file exists while the writer is live.
            assert!(fs::read_dir(&out)
                .unwrap()
                .filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().starts_with(".spill-")));
        }
        assert!(!fs::read_dir(&out)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().starts_with(".spill-")));
    }

    /// The directory's `uncompressed_size` is a u32 field, so a ≥ 4 GiB
    /// payload must be a loud error at `add_tile_full` — never a silent
    /// length truncation (in-RAM path) or a spill-length mismatch. The bound
    /// is tested on the extracted check (no 4 GiB allocation).
    #[test]
    fn payload_len_guard_rejects_4gib_payloads() {
        let id = TileId::new(5, 1, 2, 0);
        assert!(check_payload_len(&id, 0).is_ok());
        assert!(
            check_payload_len(&id, u32::MAX as u64).is_ok(),
            "exact u32::MAX still fits"
        );
        let err = check_payload_len(&id, u32::MAX as u64 + 1).unwrap_err();
        assert!(
            err.to_string().contains("uncompressed_size") && err.to_string().contains("4 GiB"),
            "got: {err}"
        );
        assert!(check_payload_len(&id, u64::MAX).is_err());
    }

    /// The writer chooses zstd at rest; the v3 contract also permits an
    /// explicitly raw directory by omitting `directory.encoding`.
    #[test]
    fn directory_encoding_compressed_and_raw_both_read() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 8 * 1024).unwrap();
        for k in 0..8u64 {
            let p = distinct_tile(k);
            w.add_tile_full(&TileId::new(10, k as u32, 0, 0), 0, 100, None, 6, None, &p)
                .unwrap();
        }
        let manifest = w.finalize(&Metadata::new("dir-enc")).unwrap();
        let manifest_path = out.join("manifest.json");

        // Fresh output declares the encoding and the at-rest bytes are a
        // valid zstd frame that inflates to the codec bytes.
        assert_eq!(manifest.directory.encoding.as_deref(), Some("zstd"));
        let at_rest = fs::read(out.join(&manifest.directory.key)).unwrap();
        assert_eq!(at_rest.len() as u64, manifest.directory.length);
        // The object self-identifies (spec §9.2): strip the magic prelude
        // before the codec bytes.
        let codec = directory_codec_bytes(&at_rest, manifest.format_version).unwrap();
        let raw = compression::decompress_zstd_with_dict(codec, None).unwrap();
        assert!(crate::directory::decode_directory(&raw).is_ok());
        assert!(verify_packed_objects(&manifest_path).unwrap().is_empty());
        let entries_compressed = PackedReader::open(&manifest_path)
            .unwrap()
            .entries()
            .to_vec();

        // Rewrite the dataset with a RAW (uncompressed) directory and no
        // `encoding` key — the key is optional and absent means raw. The object
        // still carries its `STTD` prelude; only the codec changes.
        let mut raw_obj = object_magic(DIRECTORY_MAGIC).to_vec();
        raw_obj.extend_from_slice(&raw);
        let raw_hex = blake3_128_hex(&raw_obj);
        let raw_rel = format!("index/{raw_hex}.sttd");
        fs::write(out.join(&raw_rel), &raw_obj).unwrap();
        let mut legacy = manifest.clone();
        legacy.directory = DirectoryRef {
            key: raw_rel,
            length: raw_obj.len() as u64,
            directory_version: crate::directory::DIRECTORY_VERSION,
            encoding: None,
            layout: None,
            root_length: None,
            page_count: None,
            page_entries: None,
            root_hash: None,
            page_hashes: None,
        };
        fs::write(&manifest_path, legacy.to_json_bytes().unwrap()).unwrap();

        assert!(verify_packed_objects(&manifest_path).unwrap().is_empty());
        let entries_raw = PackedReader::open(&manifest_path)
            .unwrap()
            .entries()
            .to_vec();
        assert_eq!(entries_raw, entries_compressed);

        // An unknown encoding must fail loudly, not decode garbage.
        legacy.directory.encoding = Some("br".to_string());
        fs::write(&manifest_path, legacy.to_json_bytes().unwrap()).unwrap();
        assert!(PackedReader::open(&manifest_path).is_err());
    }

    /// `PackedReader::open` must reject a manifest whose `formatVersion` is
    /// not the one this reader implements — a future breaking revision has to
    /// fail loudly at open instead of silently misdecoding.
    #[test]
    fn unknown_format_version_is_rejected_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = test_writer(&out, BlobOrdering::Auto, 8 * 1024);
        w.add_tile_full(
            &TileId::new(10, 0, 0, 0),
            0,
            100,
            None,
            6,
            None,
            &distinct_tile(1),
        )
        .unwrap();
        w.finalize(&Metadata::new("ver")).unwrap();
        let manifest_path = out.join("manifest.json");
        assert!(PackedReader::open(&manifest_path).is_ok());

        // Doctor the manifest to claim a future formatVersion.
        let mut v: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        v["formatVersion"] = serde_json::json!(PACKED_FORMAT_VERSION + 1);
        fs::write(&manifest_path, serde_json::to_vec(&v).unwrap()).unwrap();

        let err = match PackedReader::open(&manifest_path) {
            Ok(_) => panic!(
                "formatVersion {} must be rejected at open",
                PACKED_FORMAT_VERSION + 1
            ),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("formatVersion"),
            "unexpected error: {err}"
        );
    }

    /// `with_capabilities` declares required-to-understand features in the
    /// manifest: canonicalized (sorted + deduped) for byte-reproducibility,
    /// present in the JSON, and accepted by a reader that implements them.
    #[test]
    fn writer_declares_capabilities_canonically() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = test_writer(&out, BlobOrdering::Auto, 8 * 1024)
            // Unsorted + duplicated on purpose: the manifest must not depend
            // on the caller's flag order.
            .with_capabilities(vec![
                CAPABILITY_ELEVATION_FOLD.to_string(),
                CAPABILITY_COORD_QUANT.to_string(),
                CAPABILITY_COORD_QUANT.to_string(),
            ]);
        w.add_tile_full(
            &TileId::new(10, 0, 0, 0),
            0,
            100,
            None,
            6,
            None,
            &distinct_tile(1),
        )
        .unwrap();
        let manifest = w.finalize(&Metadata::new("caps")).unwrap();
        assert_eq!(
            manifest.capabilities,
            vec![
                CAPABILITY_COORD_QUANT.to_string(),
                CAPABILITY_ELEVATION_FOLD.to_string()
            ]
        );

        let manifest_path = out.join("manifest.json");
        let s = fs::read_to_string(&manifest_path).unwrap();
        assert!(s.contains("\"capabilities\""), "{s}");
        // This reader implements the whole registry, so open succeeds.
        assert!(PackedReader::open(&manifest_path).is_ok());
    }

    #[test]
    fn writer_derives_capabilities_from_its_encoder_config() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let cfg = EncoderConfig {
            quantize_coords_m: Some(1.0),
            compact_times: true,
            ..EncoderConfig::default()
        };
        let mut writer = test_writer(&out, BlobOrdering::Auto, 8 * 1024).with_encoder_config(cfg);
        let payload = encode_tile_with(
            &[point_layer("default", vec![1, 2], 0)],
            &writer.encoder_config(),
        )
        .unwrap();
        writer
            .add_tile_full(&TileId::new(10, 0, 0, 0), 0, 100, None, 2, None, &payload)
            .unwrap();
        let manifest = writer
            .finalize(&Metadata::new("derived-capabilities"))
            .unwrap();
        assert_eq!(
            manifest.capabilities,
            vec![
                CAPABILITY_COORD_QUANT.to_string(),
                CAPABILITY_TIME_DELTA.to_string()
            ]
        );
    }

    /// `PackedReader::open` must reject a dataset declaring a capability this
    /// reader does not implement, naming the unknown entries — a capability
    /// re-types existing columns, so proceeding would silently misdecode.
    #[test]
    fn unknown_capability_is_rejected_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = test_writer(&out, BlobOrdering::Auto, 8 * 1024);
        w.add_tile_full(
            &TileId::new(10, 0, 0, 0),
            0,
            100,
            None,
            6,
            None,
            &distinct_tile(1),
        )
        .unwrap();
        w.finalize(&Metadata::new("caps")).unwrap();
        let manifest_path = out.join("manifest.json");

        // Doctor the manifest to declare a capability from the future,
        // alongside one this reader DOES implement (only the unknown one may
        // be named in the error).
        let mut v: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        v["capabilities"] = serde_json::json!([CAPABILITY_COORD_QUANT, "from-the-future"]);
        fs::write(&manifest_path, serde_json::to_vec(&v).unwrap()).unwrap();

        let err = match PackedReader::open(&manifest_path) {
            Ok(_) => panic!("unknown capability must be rejected at open"),
            Err(e) => e,
        };
        let msg = err.to_string();
        // Exactly the unknown entry is named (coord-quant is implemented).
        assert!(
            msg.contains("does not implement: from-the-future ("),
            "unexpected error: {msg}"
        );
    }

    /// A clean packed dataset verifies with no issues; corrupting a pack's
    /// bytes (without changing its length) breaks the content address and is
    /// reported.
    #[test]
    fn verify_packed_objects_clean_then_detects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = test_writer(&out, BlobOrdering::Auto, 8 * 1024);
        for k in 0..12u64 {
            let p = distinct_tile(k);
            w.add_tile_full(&TileId::new(10, k as u32, 0, 0), 0, 100, None, 6, None, &p)
                .unwrap();
        }
        let manifest = w.finalize(&Metadata::new("verify")).unwrap();
        let manifest_path = out.join("manifest.json");

        // Clean dataset → no integrity violations.
        assert!(verify_packed_objects(&manifest_path).unwrap().is_empty());

        // Flip a byte in pack 0: same length, but blake3 no longer matches the
        // filename it was addressed by.
        let pack0 = out.join(&manifest.packs[0].key);
        let mut bytes = fs::read(&pack0).unwrap();
        bytes[0] ^= 0xff;
        fs::write(&pack0, &bytes).unwrap();

        let issues = verify_packed_objects(&manifest_path).unwrap();
        assert!(
            issues
                .iter()
                .any(|s| s.contains("content-address mismatch")),
            "expected a content-address mismatch, got {issues:?}"
        );
    }

    // ---- Bundle profile (`.sttb`) ------------------------------------------

    /// Build a small multi-pack dataset (with a dedup pair) into `out` and
    /// return its manifest — shared fixture for the bundle tests.
    fn build_bundle_fixture(out: &Path) -> Manifest {
        let mut w = PackWriter::create(out, BlobOrdering::Auto, 8 * 1024).unwrap();
        let bucket = 3_600_000i64;
        let static_payload = encode_tile(&[point_layer("default", vec![7, 8, 9], 0)]).unwrap();
        for k in 0..20u64 {
            let t = (k % 4) as i64 * bucket;
            let payload = if k == 3 || k == 17 {
                static_payload.clone() // byte-identical pair → dedup
            } else {
                distinct_tile(k)
            };
            let id = TileId::new(10, (k % 5) as u32, (k / 5) as u32, t as u64);
            w.add_tile_full(
                &id,
                t,
                t + bucket - 1,
                Some(t),
                6,
                Some(bucket as u64),
                &payload,
            )
            .unwrap();
        }
        let meta = Metadata::new("bundle-fixture").with_temporal_bucket_ms(bucket as u64);
        w.finalize(&meta).unwrap()
    }

    /// pack → unpack round-trips every object byte-identically (blake3 names
    /// re-verify via `verify_packed_objects`), `manifest.json` comes back
    /// verbatim, and packing is deterministic (same dataset ⇒ same bytes).
    #[test]
    fn bundle_pack_unpack_roundtrips_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("dataset");
        let manifest = build_bundle_fixture(&src);
        assert!(
            manifest.packs.len() > 1,
            "fixture should span multiple packs"
        );
        let manifest_path = src.join("manifest.json");

        // Deterministic pack: two runs produce byte-identical bundles.
        let bundle_a = dir.path().join("a.sttb");
        let bundle_b = dir.path().join("b.sttb");
        let summary = write_bundle(&manifest_path, &bundle_a).unwrap();
        write_bundle(&manifest_path, &bundle_b).unwrap();
        let bytes_a = fs::read(&bundle_a).unwrap();
        assert_eq!(
            bytes_a,
            fs::read(&bundle_b).unwrap(),
            "bundling must be deterministic"
        );
        assert_eq!(bytes_a.len() as u64, summary.bytes);
        assert_eq!(summary.objects, 1 + manifest.packs.len());

        // Magic prelude + 8-aligned object offsets in canonical order.
        assert_eq!(&bytes_a[0..4], b"STTB");
        assert_eq!(bytes_a[4], BUNDLE_VERSION);
        assert_eq!(&bytes_a[5..8], &[0, 0, 0]);
        let (header, header_end) = parse_bundle_header(&bytes_a).unwrap();
        assert_eq!(header.objects.len(), summary.objects);
        assert_eq!(
            header.objects[0].key, manifest.directory.key,
            "directory first"
        );
        for (o, p) in header.objects[1..].iter().zip(&manifest.packs) {
            assert_eq!(o.key, p.key, "packs in pack_id order");
        }
        for o in &header.objects {
            assert_eq!(o.offset % 8, 0, "object {} not 8-aligned", o.key);
            assert!(o.offset >= header_end);
        }
        // The embedded manifest is VERBATIM.
        assert_eq!(
            header.manifest.get().as_bytes(),
            fs::read(&manifest_path).unwrap(),
            "header manifest must be the manifest.json bytes verbatim"
        );

        // Unpack: byte-identical objects + manifest, clean verification.
        let out = dir.path().join("unpacked");
        let back = unpack_bundle(&bundle_a, &out).unwrap();
        assert_eq!(back.objects, summary.objects);
        assert!(
            verify_packed_objects(out.join("manifest.json"))
                .unwrap()
                .is_empty(),
            "unpacked dataset must re-verify its content addresses"
        );
        assert_eq!(
            fs::read(out.join("manifest.json")).unwrap(),
            fs::read(&manifest_path).unwrap()
        );
        for key in
            std::iter::once(&manifest.directory.key).chain(manifest.packs.iter().map(|p| &p.key))
        {
            assert_eq!(
                fs::read(out.join(key)).unwrap(),
                fs::read(src.join(key)).unwrap(),
                "object {key} must round-trip byte-identical"
            );
        }

        // And the bundle's own verifier is clean.
        assert!(verify_bundle_objects(&bundle_a).unwrap().is_empty());
    }

    /// `open_bundle` reads every tile byte-identical to the exploded-dir
    /// reader, over both directory container shapes (single and paged).
    #[test]
    fn open_bundle_reads_tiles_byte_identical_to_exploded_dir() {
        for paged in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let src = dir.path().join("dataset");
            let mut w =
                test_writer(&src, BlobOrdering::Auto, 8 * 1024).with_paging(paged.then_some(4));
            for k in 0..20u64 {
                w.add_tile_full(
                    &TileId::new(10, (k % 5) as u32, (k / 5) as u32, 0),
                    0,
                    100,
                    None,
                    6,
                    None,
                    &distinct_tile(k),
                )
                .unwrap();
            }
            w.finalize(&Metadata::new("bundle-read")).unwrap();

            let bundle = dir.path().join("dataset.sttb");
            write_bundle(src.join("manifest.json"), &bundle).unwrap();

            let exploded = PackedReader::open(src.join("manifest.json")).unwrap();
            let bundled = PackedReader::open_bundle(&bundle).unwrap();
            assert_eq!(exploded.entries(), bundled.entries(), "paged={paged}");
            assert_eq!(exploded.metadata().name, bundled.metadata().name);
            for (e_dir, e_bun) in exploded.entries().iter().zip(bundled.entries()) {
                assert_eq!(
                    exploded.read_payload(e_dir).unwrap(),
                    bundled.read_payload(e_bun).unwrap(),
                    "payload mismatch (paged={paged}) for {:?}",
                    e_dir.tile_id()
                );
            }
        }
    }

    /// Corrupting a blob inside the bundle is caught twice: by
    /// `verify_bundle_objects` (content-address mismatch) and by the
    /// bundle reader's per-blob CRC on read.
    #[test]
    fn corrupt_bundle_blob_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("dataset");
        build_bundle_fixture(&src);
        let bundle = dir.path().join("dataset.sttb");
        write_bundle(src.join("manifest.json"), &bundle).unwrap();
        assert!(verify_bundle_objects(&bundle).unwrap().is_empty());

        // Flip the first BLOB byte of the first pack's window — past the
        // 8-byte `STTP` prelude, so the corruption is caught by the content
        // address / blob CRC rather than by the object's magic check.
        let bytes = fs::read(&bundle).unwrap();
        let (header, _) = parse_bundle_header(&bytes).unwrap();
        let pack0 = header
            .objects
            .iter()
            .find(|o| o.key.starts_with("packs/"))
            .expect("a pack object");
        let mut corrupt = bytes.clone();
        corrupt[pack0.offset as usize + OBJECT_MAGIC_LEN] ^= 0xff;
        fs::write(&bundle, &corrupt).unwrap();

        let issues = verify_bundle_objects(&bundle).unwrap();
        assert!(
            issues
                .iter()
                .any(|s| s.contains("content-address mismatch")),
            "expected a content-address mismatch, got {issues:?}"
        );
        let reader = PackedReader::open_bundle(&bundle).unwrap();
        let bad = reader
            .entries()
            .iter()
            .find(|e| reader.read_payload(e).is_err());
        assert!(
            bad.is_some(),
            "some tile in the corrupted pack must fail its CRC"
        );
    }

    /// Truncated / doctored bundles error loudly — never panic, never
    /// silently succeed. Mirrors the adversarial-decode style.
    #[test]
    fn truncated_and_doctored_bundles_error_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("dataset");
        build_bundle_fixture(&src);
        let bundle = dir.path().join("dataset.sttb");
        write_bundle(src.join("manifest.json"), &bundle).unwrap();
        let good = fs::read(&bundle).unwrap();

        let open_err = |bytes: &[u8]| {
            let p = dir.path().join("bad.sttb");
            fs::write(&p, bytes).unwrap();
            assert!(
                PackedReader::open_bundle(&p).is_err(),
                "must reject {} bytes",
                bytes.len()
            );
            assert!(unpack_bundle(&p, dir.path().join("bad-out")).is_err());
        };

        // Shorter than the prelude.
        open_err(&good[..5]);
        // Bad magic.
        let mut bad = good.clone();
        bad[0] = b'X';
        open_err(&bad);
        // Unknown bundle version.
        let mut bad = good.clone();
        bad[4] = 9;
        open_err(&bad);
        // Nonzero reserved bytes.
        let mut bad = good.clone();
        bad[5] = 1;
        open_err(&bad);
        // header_len pointing past EOF.
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        open_err(&bad);
        // Truncated mid-pack: an object window now falls outside the file.
        open_err(&good[..good.len() - 32]);
        // Garbage header JSON.
        let mut bad = good.clone();
        let hl = u32::from_le_bytes(bad[8..12].try_into().unwrap()) as usize;
        bad[12..12 + hl].fill(b'!');
        open_err(&bad);

        // A handcrafted header with a path-traversal key must be refused
        // before anything is written.
        let evil_header = r#"{"manifest":{"format":"stt-packed"},"objects":[{"key":"../evil","offset":128,"length":0}]}"#;
        let mut evil = Vec::new();
        evil.extend_from_slice(&BUNDLE_MAGIC);
        evil.extend_from_slice(&[BUNDLE_VERSION, 0, 0, 0]);
        evil.extend_from_slice(&(evil_header.len() as u32).to_le_bytes());
        evil.extend_from_slice(evil_header.as_bytes());
        evil.resize(256, 0);
        let p = dir.path().join("evil.sttb");
        fs::write(&p, &evil).unwrap();
        let out = dir.path().join("evil-out");
        let err = unpack_bundle(&p, &out).unwrap_err();
        assert!(
            err.to_string().contains("unsafe bundle object key"),
            "got: {err}"
        );
        assert!(
            !dir.path().join("evil-out").exists(),
            "nothing may be written"
        );
        assert!(
            !dir.path().join("evil").exists(),
            "traversal target must not exist"
        );
    }

    // ---- formatVersion 3 (template-referencing frames + object magic) ------

    /// A quantized point layer with numeric + categorical properties — the
    /// shape whose per-tile `stt:qa`/`stt:time_offset_ms` variance the layer
    /// frame hoists into TILE_META. `seed` shifts values so every tile's
    /// affines differ.
    fn v2_point_layer(seed: u64, n: usize) -> ColumnarLayer {
        let base = 1_700_000_000_000i64 + seed as i64 * 60_000;
        ColumnarLayer {
            polygon_parts: None,
            name: "default".to_string(),
            feature_ids: (0..n as u64).map(|i| seed * 100 + i).collect(),
            start_times: (0..n as i64).map(|i| base + i * 1000).collect(),
            end_times: (0..n as i64).map(|i| base + i * 1000 + 500).collect(),
            geometry: GeometryColumn::Point(
                (0..n)
                    .map(|i| [-122.4 + seed as f64 * 0.01, 37.7 + i as f64 * 1e-4])
                    .collect(),
            ),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "speed".to_string(),
                    PropertyColumn::Numeric(
                        (0..n).map(|i| Some(seed as f64 * 3.0 + i as f64)).collect(),
                    ),
                ),
                (
                    "kind".to_string(),
                    PropertyColumn::Categorical(
                        (0..n)
                            .map(|i| Some(["car", "bus"][i % 2].to_string()))
                            .collect(),
                    ),
                ),
            ],
        }
    }

    /// The layer set the formatVersion-3 fixture builds: quantized point tiles with
    /// per-tile-varying affines plus an EMPTY tile (0 rows, dictionary
    /// column intact).
    fn v2_fixture_layers() -> Vec<ColumnarLayer> {
        let mut layers: Vec<ColumnarLayer> = (0..5).map(|k| v2_point_layer(k, 4)).collect();
        layers.push(v2_point_layer(99, 0)); // empty-bucket tile
        layers
    }

    fn v2_encoder_cfg(w: &PackWriter) -> EncoderConfig {
        EncoderConfig {
            quantize_coords_m: Some(1.0),
            quantize_attrs_auto: true,
            format_version: LAYER_FRAME_VERSION,
            template_collector: Some(w.template_collector()),
            ..EncoderConfig::default()
        }
    }

    /// Build a small formatVersion-3 dataset (encoder wired to the writer's template
    /// collector) into `out`, returning the manifest and the input layers in
    /// tile order.
    fn build_v2_dataset(out: &Path, paging: Option<usize>) -> (Manifest, Vec<ColumnarLayer>) {
        let mut w = PackWriter::create(out, BlobOrdering::Auto, 8 * 1024)
            .unwrap()
            .with_paging(paging)
            .with_capabilities(vec![
                CAPABILITY_COORD_QUANT.to_string(),
                CAPABILITY_ATTR_QUANT.to_string(),
            ]);
        let cfg = v2_encoder_cfg(&w);
        let layers = v2_fixture_layers();
        for (k, layer) in layers.iter().enumerate() {
            let payload = encode_tile_with(std::slice::from_ref(layer), &cfg).unwrap();
            w.add_tile_full(
                &TileId::new(10, k as u32, 0, 0),
                0,
                100,
                None,
                layer.feature_count() as u32,
                None,
                &payload,
            )
            .unwrap();
        }
        let manifest = w.finalize(&Metadata::new("v2-fixture")).unwrap();
        (manifest, layers)
    }

    /// End-to-end formatVersion 3: magic on every object, schemas embedded
    /// (sorted, hash-valid), verify clean, and every tile decodes through the
    /// registry to EXACTLY what the same layer's SELF-CONTAINED (inline-schema)
    /// frame decodes to — the metadata re-injection contract.
    #[test]
    fn v2_dataset_roundtrips_with_templates_magic_and_v1_equivalence() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let (manifest, layers) = build_v2_dataset(&out, None);

        assert_eq!(manifest.format_version, PACKED_FORMAT_VERSION);
        // Templates collected: TWO CORE + one PROPS schema. The five
        // populated tiles share one CORE template (constancy: they differ
        // only in per-tile metadata, which lives in TILE_META); the
        // empty-bucket tile has a second one because a zero-feature layer has
        // no `t0` to anchor compact `start_time` offsets against and so keeps
        // the absolute Int64 pair (see `choose_time_forms`). All six share the
        // single PROPS template.
        assert_eq!(manifest.schemas.len(), 3, "expected core+props templates");
        let hashes: Vec<&str> = manifest.schemas.iter().map(|s| s.hash.as_str()).collect();
        assert!(
            hashes.windows(2).all(|w| w[0] < w[1]),
            "schemas sorted by hash"
        );

        // Object magic on every object; content addresses cover it.
        let dir_bytes = fs::read(out.join(&manifest.directory.key)).unwrap();
        assert_eq!(&dir_bytes[0..8], &object_magic(DIRECTORY_MAGIC));
        assert_eq!(
            manifest.directory.key,
            format!("index/{}.sttd", blake3_128_hex(&dir_bytes))
        );
        for p in &manifest.packs {
            let bytes = fs::read(out.join(&p.key)).unwrap();
            assert_eq!(&bytes[0..8], &object_magic(PACK_MAGIC));
            assert_eq!(p.key, format!("packs/{}.sttp", blake3_128_hex(&bytes)));
        }

        // Blob offsets are object-absolute: nothing points into the magic.
        let reader = PackedReader::open(out.join("manifest.json")).unwrap();
        assert_eq!(reader.format_version(), PACKED_FORMAT_VERSION);
        assert!(reader.templates().is_some());
        for e in reader.entries() {
            assert!(
                e.offset >= OBJECT_MAGIC_LEN as u64,
                "offset {} inside magic",
                e.offset
            );
        }

        assert!(verify_packed_objects(out.join("manifest.json"))
            .unwrap()
            .is_empty());

        // Decode parity: each tile equals the self-contained (inline-schema)
        // decode of the same layer.
        for (k, layer) in layers.iter().enumerate() {
            let e = reader
                .entries()
                .iter()
                .find(|e| e.x == k as u32)
                .expect("entry present");
            let got = reader.read_layers(e).unwrap();
            let inline = crate::arrow_tile::decode_tile(
                &encode_tile_with(
                    std::slice::from_ref(layer),
                    &EncoderConfig {
                        quantize_coords_m: Some(1.0),
                        quantize_attrs_auto: true,
                        ..EncoderConfig::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(got.len(), inline.len());
            assert_eq!(got[0].name, inline[0].name);
            assert_eq!(
                got[0].batch, inline[0].batch,
                "tile {k}: template-referencing decode must equal the inline-schema \
                 decode after re-injection"
            );
        }
    }

    /// A doctored `schemas` entry (bytes not matching the declared hash) is a
    /// loud, dataset-level failure at open — before any tile fetch.
    #[test]
    fn corrupt_manifest_schema_template_fails_open() {
        use base64::Engine as _;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_v2_dataset(&out, None);
        let manifest_path = out.join("manifest.json");

        let mut v: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        let tampered = base64::engine::general_purpose::STANDARD.encode(b"not a template");
        v["schemas"][0]["data"] = serde_json::json!(tampered);
        fs::write(&manifest_path, serde_json::to_vec(&v).unwrap()).unwrap();

        let err = match PackedReader::open(&manifest_path) {
            Ok(_) => panic!("corrupt template must fail the open"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("hash"), "got: {err}");
        let issues = verify_packed_objects(&manifest_path).unwrap();
        assert!(
            issues.iter().any(|i| i.contains("hash")),
            "verify must report the schema corruption: {issues:?}"
        );
    }

    /// Corrupting an object's magic prelude is caught at open/read AND by
    /// the verifier (which also still reports the content-address mismatch).
    #[test]
    fn corrupt_v2_object_magic_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let (manifest, _) = build_v2_dataset(&out, None);
        let manifest_path = out.join("manifest.json");

        // Pack magic: flipping byte 0 breaks STTP → read path + verifier.
        let pack0 = out.join(&manifest.packs[0].key);
        let mut bytes = fs::read(&pack0).unwrap();
        bytes[0] ^= 0xff;
        fs::write(&pack0, &bytes).unwrap();
        let reader = PackedReader::open(&manifest_path).unwrap();
        let entry = reader.entries().iter().find(|e| e.pack_id == 0).unwrap();
        let err = reader.read_payload(entry).unwrap_err();
        assert!(err.to_string().contains("STTP"), "got: {err}");
        let issues = verify_packed_objects(&manifest_path).unwrap();
        assert!(issues.iter().any(|i| i.contains("STTP")), "{issues:?}");
        assert!(
            issues
                .iter()
                .any(|i| i.contains("content-address mismatch")),
            "{issues:?}"
        );
        fs::write(&pack0, {
            let mut b = fs::read(&pack0).unwrap();
            b[0] ^= 0xff;
            b
        })
        .unwrap();

        // Directory magic: same treatment, caught at open.
        let dir_path = out.join(&manifest.directory.key);
        let mut bytes = fs::read(&dir_path).unwrap();
        bytes[0] ^= 0xff;
        fs::write(&dir_path, &bytes).unwrap();
        let err = match PackedReader::open(&manifest_path) {
            Ok(_) => panic!("corrupt directory magic must fail the open"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("STTD"), "got: {err}");
    }

    /// The paged directory container works identically under formatVersion 3 (magic
    /// stripped before the root/leaf math; `rootLength` still means the root
    /// frame's at-rest length).
    #[test]
    fn v2_paged_directory_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let (manifest, layers) = build_v2_dataset(&out, Some(2));
        assert_eq!(
            manifest.directory.layout.as_deref(),
            Some(DIRECTORY_LAYOUT_PAGED)
        );
        assert!(verify_packed_objects(out.join("manifest.json"))
            .unwrap()
            .is_empty());
        let reader = PackedReader::open(out.join("manifest.json")).unwrap();
        assert_eq!(reader.entries().len(), layers.len());
        for e in reader.entries() {
            assert!(reader.read_layers(e).is_ok());
        }
    }

    /// Bundles: pack → verify → open → unpack → re-verify, all driven by the
    /// manifest embedded in the bundle header.
    #[test]
    fn v2_bundle_roundtrips_and_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("dataset");
        let (_, layers) = build_v2_dataset(&src, None);

        let bundle = dir.path().join("dataset.sttb");
        write_bundle(src.join("manifest.json"), &bundle).unwrap();
        assert!(verify_bundle_objects(&bundle).unwrap().is_empty());

        let exploded = PackedReader::open(src.join("manifest.json")).unwrap();
        let bundled = PackedReader::open_bundle(&bundle).unwrap();
        assert_eq!(bundled.format_version(), PACKED_FORMAT_VERSION);
        assert_eq!(exploded.entries(), bundled.entries());
        for (e_dir, e_bun) in exploded.entries().iter().zip(bundled.entries()) {
            let a = exploded.read_layers(e_dir).unwrap();
            let b = bundled.read_layers(e_bun).unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.batch, y.batch);
            }
        }
        assert_eq!(bundled.entries().len(), layers.len());

        let unpacked = dir.path().join("unpacked");
        unpack_bundle(&bundle, &unpacked).unwrap();
        assert!(
            verify_packed_objects(unpacked.join("manifest.json"))
                .unwrap()
                .is_empty(),
            "unpacked dataset must re-verify"
        );
    }

    /// The verbatim-repack path (pack-cover / repair tools): payloads copied
    /// from a formatVersion-3 source MUST carry the source's formatVersion forward and
    /// seed the new writer's template collector from the source registry —
    /// the repacked dataset round-trips readable, tile for tile.
    #[test]
    fn v2_verbatim_repack_roundtrips_readable() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        let (src_manifest, layers) = build_v2_dataset(&src, None);
        let reader = PackedReader::open(src.join("manifest.json")).unwrap();

        let out = dir.path().join("repacked");
        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 8 * 1024)
            .unwrap()
            .with_capabilities(reader.capabilities().to_vec())
            .with_seeded_templates(
                reader
                    .templates()
                    .expect("formatVersion-3 source has a registry"),
            );
        for e in reader.entries() {
            let payload = reader.read_payload(e).unwrap();
            w.add_tile_full(
                &TileId::new(e.zoom, e.x, e.y, e.time_start.max(0) as u64),
                e.time_start,
                e.time_end,
                e.cover_t_min,
                e.feature_count,
                e.temporal_bucket_ms,
                &payload,
            )
            .unwrap();
        }
        let manifest = w.finalize(reader.metadata()).unwrap();
        assert_eq!(manifest.format_version, PACKED_FORMAT_VERSION);
        // The seeded templates published (sorted + deduped by construction).
        assert_eq!(
            manifest.schemas.iter().map(|s| &s.hash).collect::<Vec<_>>(),
            src_manifest
                .schemas
                .iter()
                .map(|s| &s.hash)
                .collect::<Vec<_>>(),
            "seeded schemas must match the source's table"
        );

        assert!(verify_packed_objects(out.join("manifest.json"))
            .unwrap()
            .is_empty());
        let re = PackedReader::open(out.join("manifest.json")).unwrap();
        assert_eq!(re.entries().len(), layers.len());
        for e in re.entries() {
            re.read_layers(e)
                .unwrap_or_else(|err| panic!("repacked tile {:?} must decode: {err}", e.tile_id()));
        }
    }

    /// A formatVersion-3 dataset whose frames reference a template MISSING from
    /// `manifest.schemas` is undecodable — both validators must flag the
    /// absent hash instead of verifying clean (every object still hashes to
    /// its content address and the remaining schemas table is valid).
    #[test]
    fn verify_flags_missing_frame_referenced_schema_template() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let (manifest, _) = build_v2_dataset(&out, None);
        // Two CORE (populated + empty-bucket, see the roundtrip test) + one
        // PROPS template.
        assert_eq!(manifest.schemas.len(), 3, "core + props templates");
        let manifest_path = out.join("manifest.json");
        assert!(verify_packed_objects(&manifest_path).unwrap().is_empty());

        // Remove ONE schemas entry: the manifest still parses, the remaining
        // table is hash-valid + sorted, every object address still matches —
        // pre-fix this dataset "verified clean" while no tile could decode.
        let mut v: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        let removed = v["schemas"].as_array_mut().unwrap().remove(0);
        let removed_hash = removed["hash"].as_str().unwrap().to_string();
        fs::write(&manifest_path, serde_json::to_vec(&v).unwrap()).unwrap();

        let issues = verify_packed_objects(&manifest_path).unwrap();
        assert!(
            issues
                .iter()
                .any(|i| i.contains(&removed_hash) && i.contains("manifest.schemas")),
            "expected the missing template hash {removed_hash} to be reported: {issues:?}"
        );

        // The bundle validator shares the check (the doctored manifest rides
        // the bundle header verbatim).
        let bundle = dir.path().join("dataset.sttb");
        write_bundle(&manifest_path, &bundle).unwrap();
        let issues = verify_bundle_objects(&bundle).unwrap();
        assert!(
            issues.iter().any(|i| i.contains(&removed_hash)),
            "bundle verify must report the missing template: {issues:?}"
        );
    }

    // ------------------------------------------------------------------
    // WM-5 — pack-boundary planning (§6.5)
    // ------------------------------------------------------------------

    fn blobs(lens: &[u64]) -> Vec<PlannedBlob> {
        lens.iter().map(|&len| PlannedBlob { len }).collect()
    }

    /// Cuts partition the blob sequence: ascending, in range, and never empty
    /// segments. Anything else would produce a pack that is not a contiguous
    /// slice of layout order, which content addressing depends on.
    #[test]
    fn planned_cuts_are_ascending_and_in_range() {
        let b = blobs(&[10, 20, 30, 40, 50, 60, 70]);
        let opts = PackCutOptions {
            target_bytes: 100,
            max_bytes: 150,
            boundary_price_bytes: 8,
        };
        let cuts = plan_pack_cuts(&b, &opts);
        let mut prev = 0usize;
        for &c in &cuts {
            assert!(c > prev && c < b.len(), "cuts={cuts:?}");
            prev = c;
        }
    }

    /// ⚠️ THE HARD CONSTRAINT: no planned segment may exceed the per-object
    /// ceiling, except the unsplittable-blob exemption (a lone blob larger than
    /// the cap owns its pack rather than being dropped).
    #[test]
    fn no_planned_segment_exceeds_the_hard_cap() {
        let opts = PackCutOptions {
            target_bytes: 100,
            max_bytes: 120,
            boundary_price_bytes: 8,
        };
        // Includes a blob LARGER than the cap — the exemption case.
        let b = blobs(&[30, 40, 200, 10, 50, 60, 25]);
        let cuts = plan_pack_cuts(&b, &opts);

        let mut bounds = vec![0usize];
        bounds.extend(cuts.iter().copied());
        bounds.push(b.len());
        for w in bounds.windows(2) {
            let seg = &b[w[0]..w[1]];
            let total: u64 = seg.iter().map(|x| x.len).sum();
            let singleton_oversized = seg.len() == 1 && seg[0].len > opts.max_bytes;
            assert!(
                total <= opts.max_bytes || singleton_oversized,
                "segment {:?} totals {total}, over the {} cap and not a singleton exemption",
                w,
                opts.max_bytes
            );
        }
    }

    /// Deterministic: a pure function of the blob lengths and options.
    #[test]
    fn pack_cut_planning_is_deterministic() {
        let b = blobs(&[7, 13, 29, 3, 61, 5, 41, 17, 23, 11]);
        let opts = PackCutOptions {
            target_bytes: 64,
            max_bytes: 96,
            boundary_price_bytes: 4,
        };
        assert_eq!(plan_pack_cuts(&b, &opts), plan_pack_cuts(&b, &opts));
    }

    /// An EMPTY plan is the documented rollback: the writer falls through to
    /// next-fit, which the register pins as exact for its own cost model.
    #[test]
    fn an_empty_plan_leaves_next_fit_in_charge() {
        assert!(plan_pack_cuts(&[], &PackCutOptions::default()).is_empty());
        // One blob under target needs no cut at all.
        let b = blobs(&[10]);
        assert!(plan_pack_cuts(&b, &PackCutOptions::default()).is_empty());
    }
}

#[cfg(test)]
mod legacy_read_tests {
    use super::*;
    use crate::metadata::Metadata;
    use crate::tile::TileId;
    use std::fs;

    /// Build a real v3 archive, then rewrite its manifest into the v2 SHAPE the
    /// published fleet actually has: `formatVersion: 2`, no `variants` registry,
    /// and no paged per-frame hashes. The objects on disk are untouched, which is
    /// the point — v2 and v3 differ only in the container.
    fn v2_shaped_manifest(manifest: &Manifest) -> serde_json::Value {
        let mut j = serde_json::to_value(manifest).unwrap();
        let obj = j.as_object_mut().unwrap();
        obj.insert("formatVersion".into(), serde_json::json!(2));
        obj.remove("variants");
        let dir = obj.get_mut("directory").unwrap().as_object_mut().unwrap();
        dir.insert("directoryVersion".into(), serde_json::json!(5));
        dir.remove("rootHash");
        dir.remove("pageHashes");
        j
    }

    #[test]
    fn a_v2_shaped_manifest_still_opens() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut writer = super::tests::test_writer(&out, BlobOrdering::Auto, 64 * 1024);
        for i in 0..4u32 {
            writer
                .add_tile_full(
                    &TileId::new(8, i, 0, 0),
                    0,
                    100,
                    None,
                    6,
                    None,
                    &super::tests::distinct_tile(u64::from(i)),
                )
                .unwrap();
        }
        let manifest = writer.finalize(&Metadata::new("legacy")).unwrap();
        let path = out.join("manifest.json");

        // Sanity: it opens as v3 first, so the test cannot pass vacuously.
        let v3 = PackedReader::open(&path).expect("v3 opens");
        let v3_entries = v3.entries().len();
        assert_eq!(v3_entries, 4);

        // Now the v2 container shape.
        fs::write(
            &path,
            serde_json::to_vec_pretty(&v2_shaped_manifest(&manifest)).unwrap(),
        )
        .unwrap();
        let legacy = PackedReader::open(&path).expect("a v2-shaped manifest must still open");
        assert_eq!(
            legacy.entries().len(),
            v3_entries,
            "the same tiles must be visible through the legacy container"
        );
        // Every entry reads as the raw variant, which is what a pre-variant
        // archive meant.
        assert!(legacy
            .entries()
            .iter()
            .all(|e| e.variant_id == crate::tile::RAW_VARIANT_ID));
        // ...and payloads still resolve.
        let payload = legacy.read_payload(&legacy.entries()[0]).expect("payload");
        assert!(!payload.is_empty());
    }

    /// The relaxation is bounded on BOTH sides: it must not become a licence to
    /// open anything.
    #[test]
    fn the_legacy_window_has_hard_edges() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut writer = super::tests::test_writer(&out, BlobOrdering::Auto, 64 * 1024);
        writer
            .add_tile_full(
                &TileId::new(8, 0, 0, 0),
                0,
                100,
                None,
                6,
                None,
                &super::tests::distinct_tile(0),
            )
            .unwrap();
        let manifest = writer.finalize(&Metadata::new("edges")).unwrap();
        let path = out.join("manifest.json");

        // Below the window: v1 is still refused.
        let mut too_old = v2_shaped_manifest(&manifest);
        too_old["formatVersion"] = serde_json::json!(1);
        fs::write(&path, serde_json::to_vec_pretty(&too_old).unwrap()).unwrap();
        let err = PackedReader::open(&path)
            .err()
            .expect("formatVersion 1 must still be refused");
        assert!(
            err.to_string().contains("formatVersion"),
            "unexpected: {err}"
        );

        // Above the window: a future version is still refused.
        let mut too_new = v2_shaped_manifest(&manifest);
        too_new["formatVersion"] = serde_json::json!(4);
        fs::write(&path, serde_json::to_vec_pretty(&too_new).unwrap()).unwrap();
        assert!(PackedReader::open(&path).is_err(), "v4 must be refused");

        // DRIFT, not history: a CURRENT manifest claiming a legacy directory
        // codec is a mismatch and must stay a hard error, or the legacy window
        // would silently excuse corruption in new archives.
        let mut drifted = serde_json::to_value(&manifest).unwrap();
        drifted["directory"]["directoryVersion"] = serde_json::json!(5);
        fs::write(&path, serde_json::to_vec_pretty(&drifted).unwrap()).unwrap();
        let err = PackedReader::open(&path)
            .err()
            .expect("v3 manifest + v5 directory must fail");
        assert!(
            err.to_string().contains("directoryVersion"),
            "unexpected: {err}"
        );

        // A LEGACY manifest that declares hashes must still have them verified —
        // the skip is only for genuinely absent ones.
        let mut legacy_bad_hash = v2_shaped_manifest(&manifest);
        legacy_bad_hash["directory"]["rootHash"] =
            serde_json::json!("00000000000000000000000000000000");
        fs::write(&path, serde_json::to_vec_pretty(&legacy_bad_hash).unwrap()).unwrap();
        assert!(
            PackedReader::open(&path).is_err(),
            "a declared-but-wrong rootHash must still fail"
        );
    }

    /// The WRITER never emits anything legacy, whatever the reader tolerates.
    #[test]
    fn the_writer_stays_current_only() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut writer = super::tests::test_writer(&out, BlobOrdering::Auto, 64 * 1024);
        writer
            .add_tile_full(
                &TileId::new(8, 0, 0, 0),
                0,
                100,
                None,
                6,
                None,
                &super::tests::distinct_tile(0),
            )
            .unwrap();
        let manifest = writer.finalize(&Metadata::new("writer")).unwrap();
        assert_eq!(manifest.format_version, PACKED_FORMAT_VERSION);
        assert_eq!(
            manifest.directory.directory_version,
            crate::directory::DIRECTORY_VERSION
        );
        assert!(!manifest.variants.is_empty(), "v3 always declares variants");
        // And the object prelude is the current one.
        let sttd = fs::read(out.join(&manifest.directory.key)).unwrap();
        assert_eq!(sttd[4], OBJECT_MAGIC_VERSION);
    }
}
