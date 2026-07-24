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
//! is the v5 codec ([`crate::directory`]): per-run `pack_id` column +
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
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `format` discriminator written into every packed manifest.
pub const PACKED_FORMAT: &str = "stt-packed";

/// The 0.3.x packed format: no object magic, v1 layer frames, no manifest
/// `schemas` table. Frozen — [`PackWriter::with_format_version`]`(1)` MUST
/// keep reproducing it byte-identically (pinned by `tests/v1_golden.rs`).
pub const PACKED_FORMAT_VERSION_V1: u32 = 1;
/// The 2026-07 coordinated byte break
/// (`docs/roadmap/stt-packed-format-decisions.md`): `STTP`/`STTD` object
/// magic, object-absolute blob offsets, and v2 sectioned layer frames
/// referencing manifest-embedded schema templates (`manifest.schemas`).
pub const PACKED_FORMAT_VERSION_V2: u32 = 2;
/// Packed-format version (the manifest schema, distinct from the directory
/// codec's [`crate::directory::DIRECTORY_VERSION`]) this writer emits by
/// default. `stt-build --format-version 1` is the transitional kill switch.
pub const PACKED_FORMAT_VERSION: u32 = PACKED_FORMAT_VERSION_V2;
/// Every `manifest.formatVersion` this toolchain's readers accept.
pub const SUPPORTED_PACKED_FORMAT_VERSIONS: [u32; 2] =
    [PACKED_FORMAT_VERSION_V1, PACKED_FORMAT_VERSION_V2];

// ----------------------------------------------------------------------------
// v2 object magic (`.sttp` / `.sttd` self-identification, spec §9.2)
// ----------------------------------------------------------------------------

/// First 4 bytes of a formatVersion-2 pack object.
pub const PACK_MAGIC: [u8; 4] = *b"STTP";
/// First 4 bytes of a formatVersion-2 directory object.
pub const DIRECTORY_MAGIC: [u8; 4] = *b"STTD";
/// Length of the v2 object magic prelude: 4-byte tag + u8 version(2) + 3 zero
/// bytes. Blob offsets in a v2 directory are object-absolute, so a pack's
/// first blob sits at offset 8; the manifest `length` fields and every blake3
/// content address cover the ENTIRE object including this prelude (★F5).
/// v1 objects carry no magic (byte-frozen).
pub const OBJECT_MAGIC_LEN: usize = 8;
/// Version byte inside the v2 object magic prelude.
const OBJECT_MAGIC_VERSION: u8 = 2;

/// The full 8-byte magic prelude for a v2 object kind.
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

/// Validate a v2 object's 8-byte magic prelude and return the bytes after it.
fn strip_object_magic<'a>(bytes: &'a [u8], kind: [u8; 4], what: &str) -> Result<&'a [u8]> {
    let tag = std::str::from_utf8(&kind).expect("magic tags are ASCII");
    if bytes.len() < OBJECT_MAGIC_LEN || bytes[0..4] != kind {
        return Err(Error::InvalidArchive(format!(
            "{what}: missing {tag:?} magic (formatVersion-2 objects start with an \
             8-byte magic prelude)"
        )));
    }
    if bytes[4] != OBJECT_MAGIC_VERSION {
        return Err(Error::InvalidArchive(format!(
            "{what}: unsupported {tag} object version {} (this reader supports \
             {OBJECT_MAGIC_VERSION})",
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

/// A directory object's codec bytes: the whole object under v1, the
/// post-magic slice under v2. (`rootLength` keeps meaning the root frame's
/// at-rest length — a paged root fetch is `bytes=0..(8+rootLength-1)` — so
/// all downstream paged math is unchanged once the magic is stripped.)
/// Public so out-of-band directory consumers (e.g. `stt-validate`'s direct
/// paged-structure re-check) apply the same version-aware unwrap.
pub fn directory_codec_bytes(bytes: &[u8], format_version: u32) -> Result<&[u8]> {
    if format_version == PACKED_FORMAT_VERSION_V2 {
        strip_object_magic(bytes, DIRECTORY_MAGIC, "directory object")
    } else {
        Ok(bytes)
    }
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
/// each independently framed). Absent or `"single"` = the whole-load v5 object.
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
    /// Directory codec version (`5` for the packed format). The leaf pages of a
    /// paged directory are this same v5 codec — `layout` (below), not this
    /// version, discriminates the container shape.
    #[serde(rename = "directoryVersion")]
    pub directory_version: u8,
    /// At-rest encoding of the directory object. `Some("zstd")` means the
    /// object bytes are a zstd frame wrapping the codec bytes; absent (the
    /// shape every pre-encoding manifest has) means raw codec bytes. For a
    /// paged directory it describes the framing of **each page** (root + every
    /// leaf), not one frame over the whole object. The content address and
    /// `length` always describe the at-rest bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// Container layout. `Some("paged")` = a root page + leaf pages (Wave 2);
    /// absent or `Some("single")` = the single whole-load object. Readers that
    /// don't know `"paged"` fail loudly (the root's first byte isn't a valid v5
    /// directory version), which is why readers ship before any paged dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
    /// At-rest byte length of the root page (a prefix of the object). Present
    /// iff `layout == "paged"`; the reader range-GETs `bytes=0-(rootLength-1)`
    /// for the root under formatVersion 1, or `bytes=0-(8+rootLength-1)` under
    /// formatVersion 2 (the 8-byte object magic shifts the root), then leaf
    /// ranges on demand.
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

/// One schema-template entry of a formatVersion-2 manifest's `schemas` table:
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

/// Decode + hash-validate a manifest's `schemas` table into the decode-side
/// [`TemplateRegistry`]. Every entry must base64-decode and blake3-hash to
/// its declared `hash` — the loud, dataset-level failure mode for corrupt
/// manifests (design §2.2).
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
    /// fails loudly, dataset-level, before any tile fetch. Absent/empty on
    /// formatVersion-1 manifests (the key is omitted).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schemas: Vec<SchemaTemplateRef>,
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
    /// v5 directory (the default). Set via [`with_paging`](Self::with_paging).
    page_entries: Option<usize>,
    /// zstd level for per-blob + directory compression. Defaults to
    /// [`compression::ZSTD_LEVEL`]; a publish build raises it via
    /// [`with_zstd_level`](Self::with_zstd_level). Decode is level-independent,
    /// so this only trades build CPU for smaller on-the-wire bytes.
    zstd_level: i32,
    /// `manifest.capabilities` — required-to-understand feature declarations.
    /// Set via [`with_capabilities`](Self::with_capabilities); empty (the
    /// default) omits the key from the manifest.
    capabilities: Vec<String>,
    /// Opt into the **measured** ordering picker: resolve the concrete on-disk
    /// order by simulating per-ordering range-read cost ([`crate::ordering_sim`])
    /// rather than the `auto` cardinality heuristic. Set via
    /// [`with_measured_ordering`](Self::with_measured_ordering); `false` (the
    /// default) keeps the `ordering` argument's behaviour.
    measured_ordering: bool,
    /// Packed format version to emit ([`PACKED_FORMAT_VERSION_V1`] |
    /// [`PACKED_FORMAT_VERSION_V2`]). Defaults to v2; v1 is the transitional
    /// kill switch and MUST reproduce the 0.3.x bytes exactly (no object
    /// magic, `formatVersion: 1`, no `schemas` — pinned by
    /// `tests/v1_golden.rs`). Set via
    /// [`with_format_version`](Self::with_format_version).
    format_version: u32,
    /// v2 schema-template sink: the encoder records each layer's stripped
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
    /// manifest can never disagree (design §1 ★F6). Default = the plain
    /// encoder, byte-identical to a build that sets nothing.
    encoder: crate::arrow_tile::EncoderConfig,
    /// In-memory budget (bytes) for buffered UNCOMPRESSED tile payloads.
    /// `0` (the default) = unlimited — every payload stays in RAM until
    /// finalize, the legacy behaviour. Non-zero: once buffered payload bytes
    /// would exceed the budget, further payloads are appended to a temp spill
    /// file in `out_dir` and read back record-by-record during finalize.
    /// A pure memory-behaviour lever: output bytes are identical either way.
    /// Set via [`with_memory_budget`](Self::with_memory_budget).
    memory_budget: u64,
    /// Total bytes of payloads currently held in RAM (`PendingPayload::Mem`).
    pending_payload_bytes: u64,
    /// Lazily created once the budget is exceeded; `None` = nothing spilled.
    spill: Option<SpillFile>,
}

impl PackWriter {
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
            zstd_level: compression::ZSTD_LEVEL,
            capabilities: Vec::new(),
            measured_ordering: false,
            format_version: PACKED_FORMAT_VERSION,
            templates: Arc::new(TemplateCollector::new()),
            encoder: crate::arrow_tile::EncoderConfig::default(),
            memory_budget: 0,
            pending_payload_bytes: 0,
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
    /// v2 payloads reference templates by 16-byte hash but are never
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

    /// Select the packed format version to emit (`1` | `2`; default 2 =
    /// [`PACKED_FORMAT_VERSION`]). v1 is the one-release transitional kill
    /// switch (`stt-build --format-version 1`) and reproduces the 0.3.x
    /// output byte-for-byte. Anything else errors at
    /// [`finalize`](Self::finalize). The caller is responsible for feeding
    /// payloads of the MATCHING frame version — readers hard-reject
    /// mixed-version datasets (design §1 ★F6).
    pub fn with_format_version(mut self, format_version: u32) -> Self {
        self.format_version = format_version;
        self
    }

    /// The packed format version this writer will emit (`1` | `2`). Encode
    /// callers derive their frame version from this so a dataset can never
    /// mix frame and manifest versions (design §1 ★F6).
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// The writer's schema-template sink for a v2 build. Normally taken care
    /// of by [`encoder_config`](Self::encoder_config), which installs it on
    /// [`crate::arrow_tile::EncoderConfig::template_collector`] so every
    /// template a v2 frame references ends up in `manifest.schemas` at
    /// [`finalize`](Self::finalize). Unused (and empty) under v1.
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
    pub fn with_encoder_config(mut self, cfg: crate::arrow_tile::EncoderConfig) -> Self {
        self.encoder = cfg;
        self
    }

    /// The config a payload for THIS writer must be encoded with: the settings
    /// from [`with_encoder_config`](Self::with_encoder_config), with the frame
    /// version and template sink forced to the writer's own.
    ///
    /// That override is the single coupling point keeping a dataset's frames
    /// and its manifest declaration in lockstep — mixed versions are a reader
    /// hard error (packed-v2 design §1 ★F6), and a v2 frame whose templates
    /// were never recorded fails template resolution at read time.
    pub fn encoder_config(&self) -> crate::arrow_tile::EncoderConfig {
        crate::arrow_tile::EncoderConfig {
            format_version: self.format_version,
            template_collector: (self.format_version == PACKED_FORMAT_VERSION_V2)
                .then(|| self.template_collector()),
            ..self.encoder.clone()
        }
    }

    /// Opt into a **paged** directory: the `.sttd` becomes a root page + leaf
    /// pages of ≤ `page_entries` entries each, so a cold reader fetches only the
    /// leaves its viewport/time-window touches (Wave 2). `None` (the default)
    /// emits the single whole-load v5 directory — byte-identical output to a
    /// pre-paging build. `Some(0)` is clamped to 1.
    pub fn with_paging(mut self, page_entries: Option<usize>) -> Self {
        self.page_entries = page_entries.map(|k| k.max(1));
        self
    }

    /// Set the zstd compression level for tile blobs and the directory.
    ///
    /// The packed format is write-once / serve-many, so the higher build CPU of
    /// a level like 19 is paid once while the smaller bytes are paid on every
    /// fetch (measured −10..19% vs the level-3 default; decode is unaffected).
    /// Clamped to zstd's valid 1..=22 range. The default
    /// ([`compression::ZSTD_LEVEL`]) reproduces byte-identical pre-existing builds.
    pub fn with_zstd_level(mut self, level: i32) -> Self {
        self.zstd_level = level.clamp(1, compression::ZSTD_LEVEL_MAX);
        self
    }

    /// Declare the required-to-understand features this build used
    /// (`manifest.capabilities`, spec §3.1) — e.g. [`CAPABILITY_COORD_QUANT`]
    /// when coordinate quantization re-types the geometry column. Canonicalized
    /// (sorted + deduped) so the manifest bytes never depend on call order.
    /// Empty (the default) omits the key entirely, keeping the manifest
    /// byte-identical to a pre-capabilities build.
    pub fn with_capabilities(mut self, mut capabilities: Vec<String>) -> Self {
        capabilities.sort_unstable();
        capabilities.dedup();
        self.capabilities = capabilities;
        self
    }

    /// Opt into the **measured** ordering picker (`--blob-ordering measured`):
    /// at finalize, resolve the concrete on-disk blob order by simulating
    /// per-ordering range-read cost over the native-tier tiles
    /// ([`crate::ordering_sim`]) instead of the `auto` cardinality heuristic.
    /// When `on`, the `ordering` passed to [`create`](Self::create) is ignored.
    /// `false` (the default) reproduces byte-identical pre-existing builds.
    pub fn with_measured_ordering(mut self, on: bool) -> Self {
        self.measured_ordering = on;
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
        // Frame/manifest version coherence (design §1 ★F6): readers hard-
        // reject mixed-version datasets, so a frame of the OTHER version
        // would brick the output on first read. Refuse it at add time,
        // naming the tile — both directions.
        let v2_frame = crate::arrow_tile::is_frame_v2(payload);
        if v2_frame != (self.format_version == PACKED_FORMAT_VERSION_V2) {
            return Err(Error::Other(format!(
                "tile {id:?}: {} layer frame fed to a formatVersion-{} writer \
                 (mixed-version datasets are unreadable; encode payloads with \
                 the writer's PackWriter::format_version)",
                if v2_frame { "v2" } else { "v1" },
                self.format_version,
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
            zstd_level,
            capabilities,
            measured_ordering,
            format_version,
            templates,
            // Encode-time only — finalize just stores what was handed to it.
            encoder: _,
            memory_budget,
            pending_payload_bytes: _,
            spill,
        } = self;
        if !SUPPORTED_PACKED_FORMAT_VERSIONS.contains(&format_version) {
            return Err(Error::Other(format!(
                "unsupported packed formatVersion {format_version} (this writer emits 1 or 2)"
            )));
        }
        // (v2 objects carry an 8-byte magic prelude and blob offsets are
        // object-absolute — the first blob of a pack sits at offset 8 (★F5);
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
        // time axis, so `choose` no longer overstates space for sparse data (F1).
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
            // Pass the archive's real pack target so the simulated per-pack
            // coalescing matches the packs this build will actually cut. (Blob
            // weights here are uncompressed payload lengths vs the writer's
            // compressed cut, so boundaries land a little early — but uniformly
            // across all orderings, so the RELATIVE ranking is preserved.)
            crate::ordering_sim::measured_ordering(
                &samples,
                crate::ordering_sim::SimOptions {
                    pack_bytes: pack_target_bytes,
                    ..crate::ordering_sim::SimOptions::default()
                },
            )
        } else {
            match ordering {
                BlobOrdering::Auto => {
                    let space_bits = crate::curve::bits_for(x_span.max(y_span) + 1);
                    let time_bits = crate::curve::bits_for((tb_span.max(0) + 1) as u64);
                    BlobOrdering::choose(space_bits, time_bits)
                }
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
        let mut stream =
            PackStreamWriter::new(&out_dir, &packs_dir, format_version, pack_target_bytes);
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
                    // v1: offsets are pack-relative from 0. v2: object-
                    // absolute — the magic prelude occupies [0, 8), so blobs
                    // start at 8 and readers slice `[offset..offset+length]`
                    // off the whole object unchanged.
                    let (pack_id, offset) = stream.append(&compressed)?;
                    blobs.push(Blob {
                        length: compressed.len() as u32,
                        uncompressed_size: p.payload.len() as u32,
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
        entries.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start, e.temporal_bucket_ms));

        // The manifest's totals are derived from the directory itself, not
        // taken from the caller: `tile_count` = directory entries,
        // `feature_count` = sum of per-entry counts (the same total
        // stt-validate reports as `feature_count_index`). Deriving here keeps
        // manifest and directory consistent by construction for every writer
        // path — historically no caller set these and manifests shipped 0s.
        let mut metadata = metadata.clone();
        metadata.tile_count = entries.len() as u64;
        metadata.feature_count = entries.iter().map(|e| u64::from(e.feature_count)).sum();

        // --- Write the remaining objects ----------------------------------
        // (Packs already streamed to disk above — content-addressed via the
        // incremental hash, atomically renamed at each seal.)

        // Encode the directory. Both shapes are zstd-at-rest (declared via
        // `directory.encoding`) — directories compress ~2x and sit on the
        // cold-start critical path with no CDN content-encoding rescue:
        //   - single (default): one zstd frame over the whole v5 codec buffer.
        //   - paged (opt-in):   a root page + leaf pages, each its own zstd
        //     frame, so a cold reader fetches only the leaves it touches. The
        //     leaf codec is the same v5 directory.
        // The object is content-addressed over its at-rest bytes either way.
        let (index_bytes, directory_ref_fields): (Vec<u8>, _) = if let Some(k) = page_entries {
            let paged =
                crate::directory_page::encode_paged_directory_level(&entries, k, true, zstd_level)?;
            (
                paged.bytes,
                (
                    Some(DIRECTORY_LAYOUT_PAGED.to_string()),
                    Some(paged.root_length),
                    Some(paged.page_count as u64),
                    Some(paged.page_entries as u64),
                ),
            )
        } else {
            let index_plain = crate::directory::encode_directory(&entries);
            let bytes = compression::compress_zstd_with_dict_level(&index_plain, None, zstd_level)?;
            (bytes, (None, None, None, None))
        };
        // v2: the object is magic + frames; `rootLength` keeps meaning the
        // root frame's at-rest length (a paged reader fetches
        // `bytes=0..(8+rootLength-1)`), so the DirectoryRef fields above need
        // no adjustment — only the object bytes gain the prelude.
        let index_bytes: Vec<u8> = if format_version == PACKED_FORMAT_VERSION_V2 {
            let mut with_magic = Vec::with_capacity(OBJECT_MAGIC_LEN + index_bytes.len());
            with_magic.extend_from_slice(&object_magic(DIRECTORY_MAGIC));
            with_magic.extend_from_slice(&index_bytes);
            with_magic
        } else {
            index_bytes
        };
        let index_hex = blake3_128_hex(&index_bytes);
        let index_rel = format!("index/{index_hex}.sttd");
        let index_path = out_dir.join(&index_rel);
        write_atomic(&index_dir, &index_path, &index_bytes)?;

        let (layout, root_length, page_count, page_entries_field) = directory_ref_fields;

        // v2: publish the collected schema templates. The collector snapshot
        // is sorted by hash and deduped by construction, so the manifest
        // bytes are independent of encode order/parallelism (★F1).
        let schemas: Vec<SchemaTemplateRef> = if format_version == PACKED_FORMAT_VERSION_V2 {
            use base64::Engine as _;
            templates
                .snapshot()
                .into_iter()
                .map(|(hash, data)| SchemaTemplateRef {
                    hash: hash.iter().map(|b| format!("{b:02x}")).collect(),
                    data: base64::engine::general_purpose::STANDARD.encode(&data),
                })
                .collect()
        } else {
            if !templates.is_empty() {
                return Err(Error::Other(
                    "template collector is non-empty on a formatVersion-1 build: v2-encoded \
                     payloads were fed to a v1 writer (mixed-version dataset)"
                        .into(),
                ));
            }
            Vec::new()
        };

        // Build + write the manifest.
        let manifest = Manifest {
            format: PACKED_FORMAT.to_string(),
            format_version,
            capabilities,
            schemas,
            compression: "zstd".to_string(),
            // Record the concrete order actually laid down (never auto/measured).
            blob_ordering: Some(ordering.as_str().to_string()),
            directory: DirectoryRef {
                key: index_rel,
                length: index_bytes.len() as u64,
                directory_version: crate::directory::DIRECTORY_VERSION,
                encoding: Some(DIRECTORY_ENCODING_ZSTD.to_string()),
                layout,
                root_length,
                page_count,
                page_entries: page_entries_field,
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
/// `encoding` is the manifest's `directory.encoding`: absent = raw (every
/// pre-encoding manifest), `"zstd"` = one zstd frame around the codec bytes.
fn decode_directory_object(bytes: &[u8], encoding: Option<&str>) -> Result<Vec<u8>> {
    match encoding {
        None => Ok(bytes.to_vec()),
        Some(DIRECTORY_ENCODING_ZSTD) => compression::decompress_zstd_with_dict(bytes, None),
        Some(other) => Err(Error::InvalidArchive(format!(
            "unknown directory encoding {other:?} (this reader supports absent or \"zstd\")"
        ))),
    }
}

/// Decode the full tile-entry list from a directory object's at-rest bytes,
/// branching on the container layout. The single (whole-load) shape unwraps the
/// at-rest encoding then runs the v5 codec; the paged shape decodes the root +
/// every leaf (the local load-all path — a mmap'd file has no cold-start cost,
/// so the paging *query* win lives in the TS HTTP reader). Used by the local
/// `PackedReader` and `verify_packed_objects`.
fn decode_directory_entries(bytes: &[u8], dref: &DirectoryRef) -> Result<Vec<TileEntry>> {
    if dref.is_paged() {
        let root_length = dref.root_length.ok_or_else(|| {
            Error::InvalidArchive("paged directory: manifest missing rootLength".into())
        })?;
        let zstd = dref.encoding.as_deref() == Some(DIRECTORY_ENCODING_ZSTD);
        crate::directory_page::decode_paged_directory(bytes, root_length, zstd)
    } else {
        let raw = decode_directory_object(bytes, dref.encoding.as_deref())?;
        crate::directory::decode_directory(&raw)
    }
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
struct PackStreamWriter {
    out_dir: PathBuf,
    packs_dir: PathBuf,
    format_version: u32,
    pack_target_bytes: u64,
    /// Object-absolute offset of the first blob (`OBJECT_MAGIC_LEN` under
    /// v2, 0 under v1) — the "empty pack" length.
    data_start: u64,
    current: Option<OpenPack>,
    next_pack_id: u32,
    refs: Vec<PackRef>,
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
    fn new(out_dir: &Path, packs_dir: &Path, format_version: u32, pack_target_bytes: u64) -> Self {
        Self {
            out_dir: out_dir.to_path_buf(),
            packs_dir: packs_dir.to_path_buf(),
            format_version,
            pack_target_bytes,
            data_start: if format_version == PACKED_FORMAT_VERSION_V2 {
                OBJECT_MAGIC_LEN as u64
            } else {
                0
            },
            current: None,
            next_pack_id: 0,
            refs: Vec::new(),
        }
    }

    /// Append one blob's compressed bytes, returning its `(pack_id,
    /// object-absolute offset)` placement. Cuts BEFORE the blob if adding it
    /// would exceed the target and the current pack is non-empty, so a lone
    /// oversized blob still fits — it just owns a pack (the unsplittable-blob
    /// rule, unchanged).
    fn append(&mut self, compressed: &[u8]) -> Result<(u32, u64)> {
        let blen = compressed.len() as u64;
        if let Some(cur) = &self.current {
            if cur.len > self.data_start && cur.len + blen > self.pack_target_bytes {
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
    /// the v2 magic prelude already written and hashed.
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
        let mut len = 0u64;
        if self.format_version == PACKED_FORMAT_VERSION_V2 {
            let magic = object_magic(PACK_MAGIC);
            file.write_all(&magic)?;
            hasher.update(&magic);
            len = OBJECT_MAGIC_LEN as u64;
        }
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

    if manifest.format != PACKED_FORMAT {
        issues.push(format!(
            "manifest format is {:?}, expected {PACKED_FORMAT:?}",
            manifest.format
        ));
    }
    if !SUPPORTED_PACKED_FORMAT_VERSIONS.contains(&manifest.format_version) {
        issues.push(format!(
            "manifest formatVersion is {}, expected one of {SUPPORTED_PACKED_FORMAT_VERSIONS:?}",
            manifest.format_version
        ));
    }
    if manifest.directory.directory_version != crate::directory::DIRECTORY_VERSION {
        issues.push(format!(
            "directoryVersion is {}, expected {}",
            manifest.directory.directory_version,
            crate::directory::DIRECTORY_VERSION
        ));
    }
    verify_manifest_schemas(&manifest, &mut issues);

    // Each content-addressed object: name must equal blake3-128 of its bytes,
    // and on-disk length must equal the declared length. Content addresses
    // cover the ENTIRE object — magic prelude included on v2 — so this check
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

    // v2 objects must self-identify: validate each object's magic prelude.
    if manifest.format_version == PACKED_FORMAT_VERSION_V2 {
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

    // Directory must decode (through its v2 magic prelude, at-rest encoding +
    // container layout) and reference only packs that exist. Under v2, the
    // decoded entries additionally drive the frame → manifest.schemas
    // reference check (a dataset whose frames reference a missing template
    // is undecodable and must not verify clean).
    match fs::read(root.join(&manifest.directory.key)) {
        Ok(dir_bytes) => {
            match directory_codec_bytes(&dir_bytes, manifest.format_version)
                .and_then(|codec| decode_directory_entries(codec, &manifest.directory))
            {
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

/// Shared `manifest.schemas` checks for both verifiers: v1 must not carry the
/// table; v2 entries must base64-decode, hash to their declared address, and
/// arrive sorted-by-hash + deduped (the byte-reproducibility contract).
fn verify_manifest_schemas(manifest: &Manifest, issues: &mut Vec<String>) {
    if manifest.format_version == PACKED_FORMAT_VERSION_V1 {
        if !manifest.schemas.is_empty() {
            issues.push("formatVersion-1 manifest carries a schemas table (v2-only)".into());
        }
        return;
    }
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

/// v2 deep check shared by both verifiers: every template hash any tile frame
/// references must resolve in `manifest.schemas` — otherwise the dataset
/// "verifies clean" while no tile can decode. Walks each UNIQUE blob (dedup by
/// `(pack_id, offset)`), decompresses it, parses ONLY the v2 frame header
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
    if manifest.format_version != PACKED_FORMAT_VERSION_V2 {
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
                "tile {:?}: v1 layer frame inside a formatVersion-2 dataset",
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
                "tile {:?}: v2 frame header parse failed: {err}",
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
/// the decoded v5 directory, [`metadata`](Self::metadata) the folded metadata,
/// and [`read_payload`](Self::read_payload) selects the entry's pack, slices
/// `[offset..offset+length]`, verifies CRC32C and decompresses the per-blob
/// zstd. Packs are mmap'd lazily by `pack_id`.
pub struct PackedReader {
    entries: Vec<TileEntry>,
    metadata: Metadata,
    compression: Compression,
    packs: Vec<std::cell::RefCell<LoadedPack>>,
    capabilities: Vec<String>,
    /// The manifest's authoritative `formatVersion` (1 | 2). Decides object
    /// magic expectations and which layer-frame shape
    /// [`read_layers`](Self::read_layers) accepts — mixed-version datasets
    /// are hard errors both ways (design §1 ★F6).
    format_version: u32,
    /// v2 only: the hash-validated schema-template registry decoded from
    /// `manifest.schemas` at open. `None` on v1 datasets.
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
    // AUTHORITATIVE v1/v2 discriminator (★F6) — the frame escape is only
    // defense-in-depth.
    if !SUPPORTED_PACKED_FORMAT_VERSIONS.contains(&manifest.format_version) {
        return Err(Error::InvalidArchive(format!(
            "unsupported packed formatVersion {} (this reader supports 1 and 2)",
            manifest.format_version
        )));
    }
    // A schemas table on a v1 manifest is a mixed-version artifact — refuse
    // rather than guess which half to believe.
    if manifest.format_version == PACKED_FORMAT_VERSION_V1 && !manifest.schemas.is_empty() {
        return Err(Error::InvalidArchive(
            "formatVersion-1 manifest carries a schemas table (v2-only)".into(),
        ));
    }
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
        // v2: decode + hash-validate the manifest-embedded schema templates
        // into the registry every tile decode resolves against. Fails the
        // whole open (dataset-level) on any corrupt entry.
        let templates = (manifest.format_version == PACKED_FORMAT_VERSION_V2)
            .then(|| build_template_registry(&manifest.schemas))
            .transpose()?;

        // Load + decode the directory object (validating + stripping the v2
        // magic prelude first). The single (whole-load) shape unwraps the
        // at-rest encoding then runs the v5 codec; the paged shape decodes the
        // root + every leaf (local load-all — a mmap'd file has no cold-start
        // cost). Both branches return the same full entry list.
        let dir_path = root.join(&manifest.directory.key);
        let dir_bytes = fs::read(&dir_path)?;
        let codec_bytes = directory_codec_bytes(&dir_bytes, manifest.format_version)?;
        let entries = decode_directory_entries(codec_bytes, &manifest.directory)?;

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

        // v2: registry from the embedded manifest, then per-object magic
        // validation on every window before anything decodes through it.
        let templates = (manifest.format_version == PACKED_FORMAT_VERSION_V2)
            .then(|| build_template_registry(&manifest.schemas))
            .transpose()?;

        let (dir_off, dir_len) = lookup(
            &manifest.directory.key,
            manifest.directory.length,
            "directory",
        )?;
        let dir_object = &mmap[dir_off as usize..(dir_off + dir_len) as usize];
        let codec_bytes = directory_codec_bytes(dir_object, manifest.format_version)?;
        let entries = decode_directory_entries(codec_bytes, &manifest.directory)?;
        let windows = manifest
            .packs
            .iter()
            .map(|p| lookup(&p.key, p.length, "pack"))
            .collect::<Result<Vec<_>>>()?;
        if manifest.format_version == PACKED_FORMAT_VERSION_V2 {
            for (&(off, len), p) in windows.iter().zip(&manifest.packs) {
                strip_object_magic(
                    &mmap[off as usize..(off + len) as usize],
                    PACK_MAGIC,
                    &format!("bundle pack {:?}", p.key),
                )?;
            }
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
            // v2 objects self-identify; validate once per mapping. (Blob
            // offsets are object-absolute under v2, so the slice math below
            // is version-independent.)
            if self.format_version == PACKED_FORMAT_VERSION_V2 {
                strip_object_magic(&mmap, PACK_MAGIC, &format!("pack {}", pack.path.display()))?;
            }
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
            compression::decompress_zstd_with_dict(compressed, None)?
        } else {
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

    /// Decode a tile payload under this dataset's declared `formatVersion`.
    ///
    /// The manifest is authoritative (design §1 ★F6): a v2 frame inside a
    /// v1-declared dataset is a hard error, and vice versa — never a silent
    /// best-effort decode of a mixed-version dataset. v2 frames resolve their
    /// template references through the registry built at open.
    pub fn decode_payload(&self, payload: &[u8]) -> Result<Vec<crate::arrow_tile::DecodedLayer>> {
        let v2_frame = crate::arrow_tile::is_frame_v2(payload);
        if self.format_version == PACKED_FORMAT_VERSION_V2 {
            if !v2_frame {
                return Err(Error::InvalidArchive(
                    "v1 layer frame inside a formatVersion-2 dataset (the manifest is \
                     authoritative; mixed-version datasets are invalid)"
                        .into(),
                ));
            }
            let templates = self
                .templates
                .as_ref()
                .expect("v2 open always builds the template registry");
            crate::arrow_tile::decode_tile_with_templates(payload, templates)
        } else {
            if v2_frame {
                return Err(Error::InvalidArchive(
                    "v2 layer frame inside a formatVersion-1 dataset (the manifest is \
                     authoritative; mixed-version datasets are invalid)"
                        .into(),
                ));
            }
            crate::arrow_tile::decode_tile(payload)
        }
    }

    /// The manifest's authoritative `formatVersion` (1 | 2). A tool that
    /// repacks this dataset's payloads verbatim MUST write the same version
    /// (`PackWriter::with_format_version`) — frames and manifest declarations
    /// may not mix.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// The v2 schema-template registry decoded from `manifest.schemas`
    /// (`None` on v1 datasets). Exposed so tools that fetch payloads through
    /// [`read_payload`](Self::read_payload) can decode them out-of-band via
    /// [`crate::arrow_tile::decode_tile_with_templates`].
    pub fn templates(&self) -> Option<&TemplateRegistry> {
        self.templates.as_ref()
    }
}

// ----------------------------------------------------------------------------
// Bundle profile (`.sttb`) — single-file interchange (spec §13, DRAFT)
// ----------------------------------------------------------------------------
//
// A bundle is the packed dataset as ONE file, restoring the "download one
// file" usability property the exploded layout gave up. Strictly an
// interchange profile: the CDN story remains the exploded layout — nothing
// serves bundles over HTTP ranges. Layout:
//
// ```text
// "STTB"  u8 version(1)  [3 × 0x00]   # 8-byte magic prelude
// u32     header_len                  # little-endian
// [header JSON]                       # { manifest: <verbatim>, objects: [..] }
// [zero pad to 8]
// [objects back-to-back at 8-byte-aligned offsets]
// ```
//
// The container is object-agnostic (keys + bytes are opaque), so it works
// for `formatVersion` 1 today and future manifest revisions unchanged.

/// First 4 bytes of a `.sttb` bundle file.
pub const BUNDLE_MAGIC: [u8; 4] = *b"STTB";
/// Bundle container version this toolchain writes and reads.
pub const BUNDLE_VERSION: u8 = 1;
/// Fixed bundle prelude: 8-byte magic (`"STTB"`, version, 3 × 0x00) + the
/// little-endian `u32` header length.
const BUNDLE_PRELUDE_LEN: usize = 12;

/// One object entry in the bundle header's `objects` table.
///
/// `offset`/`length` are byte positions within the bundle file, emitted as
/// JSON numbers — exact for u64 in Rust, and fine to 2^53 for JS readers
/// (a 9-PB bundle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleObject {
    /// Object key, verbatim from the manifest (e.g. `packs/<hash>.sttp`).
    pub key: String,
    /// Absolute byte offset of the object within the bundle (8-aligned).
    pub offset: u64,
    /// Object length in bytes (at-rest bytes; padding excluded).
    pub length: u64,
}

/// The bundle header JSON: the dataset's `manifest.json` VERBATIM (so unpack
/// reproduces it byte-identically) plus the object table.
#[derive(Serialize, Deserialize)]
struct BundleHeader {
    manifest: Box<serde_json::value::RawValue>,
    objects: Vec<BundleObject>,
}

/// Summary of a bundle pack/unpack for CLI reporting.
#[derive(Debug, Clone, Copy)]
pub struct BundleSummary {
    /// Objects carried (directory + packs + any future manifest-listed
    /// kinds); `manifest.json` rides in the header and is not counted.
    pub objects: usize,
    /// Pack: the final bundle file size. Unpack: total bytes written
    /// (objects + `manifest.json`).
    pub bytes: u64,
}

/// Round `n` up to the next 8-byte boundary (bundle object alignment —
/// mirrors the layer-frame alignment rule).
fn align8_u64(n: u64) -> u64 {
    n.div_ceil(8) * 8
}

/// Reject bundle/manifest object keys that could escape the dataset root
/// when joined to a filesystem path — spec §11's path-traversal guard,
/// applied to the bundle header exactly as readers apply it to the manifest.
fn validate_bundle_key(key: &str) -> Result<()> {
    let bad = key.is_empty()
        || key.starts_with('/')
        || key.contains('\\')
        || key.contains(':')
        || key
            .split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == "..");
    if bad {
        return Err(Error::InvalidArchive(format!(
            "unsafe bundle object key {key:?} (keys must be relative paths \
             with no empty/'.'/'..' segments)"
        )));
    }
    Ok(())
}

/// If `key` names a content-addressed object (`<dir>/<32-lower-hex>.<ext>`),
/// return the hex stem so callers can verify blake3(bytes) against it.
/// Unknown/future key shapes return `None` and skip the hash check.
fn content_address_stem(key: &str) -> Option<&str> {
    let name = key.rsplit('/').next()?;
    let stem = name.split('.').next()?;
    (stem.len() == 32 && stem.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .then_some(stem)
}

/// One manifest-referenced object in canonical bundle order.
struct ManifestObjectKey {
    key: String,
    /// Manifest-declared at-rest length, when the manifest carries one.
    declared_length: Option<u64>,
}

/// Enumerate a packed manifest's object keys in the **canonical bundle
/// order**: the directory first, then packs in `pack_id` order, then any
/// future manifest object tables (a `formatVersion: 2` `schemas` array) in
/// listed order. Operates on the raw JSON value so bundling stays
/// object-agnostic across manifest revisions. Keys are traversal-validated
/// and must be unique.
fn manifest_object_keys(manifest: &serde_json::Value) -> Result<Vec<ManifestObjectKey>> {
    let obj = manifest
        .as_object()
        .ok_or_else(|| Error::InvalidArchive("manifest is not a JSON object".into()))?;
    if obj.get("format").and_then(|v| v.as_str()) != Some(PACKED_FORMAT) {
        return Err(Error::InvalidArchive(format!(
            "not a packed manifest: format={:?} (expected {PACKED_FORMAT:?})",
            obj.get("format")
        )));
    }
    let entry = |v: &serde_json::Value, what: &str| -> Result<ManifestObjectKey> {
        let o = v
            .as_object()
            .ok_or_else(|| Error::InvalidArchive(format!("manifest {what} is not an object")))?;
        let key = o
            .get("key")
            .and_then(|k| k.as_str())
            .ok_or_else(|| Error::InvalidArchive(format!("manifest {what} has no string `key`")))?;
        Ok(ManifestObjectKey {
            key: key.to_string(),
            declared_length: o.get("length").and_then(|l| l.as_u64()),
        })
    };

    let mut out = Vec::new();
    out.push(entry(
        obj.get("directory")
            .ok_or_else(|| Error::InvalidArchive("manifest has no `directory`".into()))?,
        "directory",
    )?);
    let packs = obj
        .get("packs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::InvalidArchive("manifest has no `packs` array".into()))?;
    for (i, p) in packs.iter().enumerate() {
        out.push(entry(p, &format!("packs[{i}]"))?);
    }
    // formatVersion 2 embeds its schema templates INSIDE the manifest
    // (`schemas: [{hash, data}]` — base64, no object keys), so they ride the
    // header's verbatim manifest and the bundle carries no schema objects.
    // (An earlier draft planned external `schemas/<hash>.sttt` objects; the
    // frozen v2 design dissolved that object class.)

    let mut seen = std::collections::HashSet::with_capacity(out.len());
    for k in &out {
        validate_bundle_key(&k.key)?;
        if !seen.insert(k.key.as_str()) {
            return Err(Error::InvalidArchive(format!(
                "manifest lists object key {:?} twice",
                k.key
            )));
        }
    }
    Ok(out)
}

/// Parse and validate a bundle's fixed prelude + JSON header. Returns the
/// header and the byte offset where the header region ends (object windows
/// must start at or after it). Errors — never panics — on truncation, bad
/// magic, unknown version, nonzero reserved bytes, or malformed header JSON.
fn parse_bundle_header(bytes: &[u8]) -> Result<(BundleHeader, u64)> {
    if bytes.len() < BUNDLE_PRELUDE_LEN {
        return Err(Error::InvalidArchive(format!(
            "not an STT bundle: {} bytes is shorter than the {BUNDLE_PRELUDE_LEN}-byte prelude",
            bytes.len()
        )));
    }
    if bytes[0..4] != BUNDLE_MAGIC {
        return Err(Error::InvalidArchive(
            "not an STT bundle (bad magic; expected \"STTB\")".into(),
        ));
    }
    if bytes[4] != BUNDLE_VERSION {
        return Err(Error::InvalidArchive(format!(
            "unsupported bundle version {} (this reader supports {BUNDLE_VERSION})",
            bytes[4]
        )));
    }
    if bytes[5..8] != [0, 0, 0] {
        return Err(Error::InvalidArchive(
            "malformed bundle: reserved prelude bytes must be zero".into(),
        ));
    }
    let header_len = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as u64;
    let header_end = BUNDLE_PRELUDE_LEN as u64 + header_len;
    if header_end > bytes.len() as u64 {
        return Err(Error::InvalidArchive(format!(
            "truncated bundle: header claims {header_len} bytes but only {} remain",
            bytes.len() - BUNDLE_PRELUDE_LEN
        )));
    }
    let header: BundleHeader =
        serde_json::from_slice(&bytes[BUNDLE_PRELUDE_LEN..header_end as usize])
            .map_err(|e| Error::InvalidArchive(format!("bundle header JSON decode failed: {e}")))?;
    Ok((header, header_end))
}

/// Pack an exploded packed dataset into a single `.sttb` bundle file.
///
/// The manifest JSON is embedded **verbatim** in the header (so
/// [`unpack_bundle`] reproduces `manifest.json` byte-identically) and the
/// objects are laid out back-to-back at 8-byte-aligned offsets in canonical
/// manifest order — deterministic: the same dataset packs to byte-identical
/// bundle bytes. Every content-addressed object is re-hashed on the way in;
/// a mismatch aborts the pack (never emit a bundle that cannot verify).
pub fn write_bundle<P: AsRef<Path>, Q: AsRef<Path>>(
    manifest_path: P,
    out_path: Q,
) -> Result<BundleSummary> {
    let manifest_path = manifest_path.as_ref();
    let out_path = out_path.as_ref();
    let root = manifest_path
        .parent()
        .ok_or_else(|| Error::InvalidArchive("manifest path has no parent dir".into()))?;

    let manifest_raw = fs::read(manifest_path)?;
    let manifest_text = std::str::from_utf8(&manifest_raw)
        .map_err(|_| Error::InvalidArchive("manifest.json is not UTF-8".into()))?
        .trim()
        .to_string();
    let manifest_value: serde_json::Value = serde_json::from_str(&manifest_text)
        .map_err(|e| Error::InvalidArchive(format!("manifest JSON decode failed: {e}")))?;
    let keys = manifest_object_keys(&manifest_value)?;
    let manifest_rv = serde_json::value::RawValue::from_string(manifest_text)
        .map_err(|e| Error::InvalidArchive(format!("manifest is not a JSON value: {e}")))?;

    // Plan the layout from on-disk lengths (verified again at copy time, and
    // against the manifest-declared lengths where the manifest carries one).
    let mut lengths: Vec<u64> = Vec::with_capacity(keys.len());
    for k in &keys {
        let len = fs::metadata(root.join(&k.key))
            .map_err(|e| Error::InvalidArchive(format!("{}: cannot stat object ({e})", k.key)))?
            .len();
        if let Some(declared) = k.declared_length {
            if len != declared {
                return Err(Error::InvalidArchive(format!(
                    "{}: on-disk length {len} != manifest-declared {declared}",
                    k.key
                )));
            }
        }
        lengths.push(len);
    }
    let mut rel_offsets: Vec<u64> = Vec::with_capacity(lengths.len());
    let mut data_len = 0u64;
    for len in &lengths {
        rel_offsets.push(data_len);
        data_len = align8_u64(data_len + len);
    }

    // Fixed point on the header length: absolute offsets appear as decimal
    // digits inside the header JSON, so the data start depends on the header
    // length and vice versa. `data_start` only ever grows across iterations
    // (offsets grow → digits grow → header grows), so this converges —
    // typically on the second pass.
    let mut data_start = align8_u64(BUNDLE_PRELUDE_LEN as u64);
    let header_bytes = loop {
        let objects: Vec<BundleObject> = keys
            .iter()
            .zip(&rel_offsets)
            .zip(&lengths)
            .map(|((k, rel), len)| BundleObject {
                key: k.key.clone(),
                offset: data_start + rel,
                length: *len,
            })
            .collect();
        let header = BundleHeader {
            manifest: manifest_rv.clone(),
            objects,
        };
        let bytes = serde_json::to_vec(&header)
            .map_err(|e| Error::Other(format!("bundle header encode failed: {e}")))?;
        if bytes.len() as u64 > u32::MAX as u64 {
            return Err(Error::InvalidArchive(format!(
                "bundle header is {} bytes, exceeding the u32 header_len cap",
                bytes.len()
            )));
        }
        let next = align8_u64(BUNDLE_PRELUDE_LEN as u64 + bytes.len() as u64);
        if next == data_start {
            break bytes;
        }
        data_start = next;
    };

    // Stream out via a same-directory temp file + atomic rename (the
    // content-addressed-object write discipline, applied to the bundle).
    let out_dir = match out_path.parent() {
        Some(p) if p != Path::new("") => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    fs::create_dir_all(&out_dir)?;
    let tmp = out_dir.join(format!(".tmp-sttb-{}", std::process::id()));
    let total = {
        use std::io::BufWriter;
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        let mut w = BufWriter::new(file);
        const ZEROS: [u8; 8] = [0u8; 8];

        w.write_all(&BUNDLE_MAGIC)?;
        w.write_all(&[BUNDLE_VERSION, 0, 0, 0])?;
        w.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        w.write_all(&header_bytes)?;
        let mut pos = BUNDLE_PRELUDE_LEN as u64 + header_bytes.len() as u64;
        w.write_all(&ZEROS[..(data_start - pos) as usize])?;
        pos = data_start;

        for ((k, rel), len) in keys.iter().zip(&rel_offsets).zip(&lengths) {
            debug_assert_eq!(pos, data_start + rel);
            let bytes = fs::read(root.join(&k.key))?;
            if bytes.len() as u64 != *len {
                return Err(Error::InvalidArchive(format!(
                    "{}: object changed size during bundling ({} != planned {len})",
                    k.key,
                    bytes.len()
                )));
            }
            // Content-addressed keys must hash to their own name — never
            // emit a bundle that cannot verify.
            if let Some(stem) = content_address_stem(&k.key) {
                let hex = blake3_128_hex(&bytes);
                if hex != stem {
                    return Err(Error::InvalidArchive(format!(
                        "{}: content-address mismatch (bytes hash to {hex})",
                        k.key
                    )));
                }
            }
            w.write_all(&bytes)?;
            pos += *len;
            let padded = align8_u64(pos);
            w.write_all(&ZEROS[..(padded - pos) as usize])?;
            pos = padded;
        }
        w.flush()?;
        pos
    };
    fs::rename(&tmp, out_path)?;

    Ok(BundleSummary {
        objects: keys.len(),
        bytes: total,
    })
}

/// Explode a `.sttb` bundle back into a packed dataset directory:
/// `manifest.json` (the verbatim header bytes) plus every object at its key
/// path. Keys are traversal-validated and windows bounds-checked **before**
/// anything is written. Objects are content-addressed, so callers can (and
/// `stt-bundle unpack` does) run [`verify_packed_objects`] on the result to
/// prove the round-trip byte-identical.
pub fn unpack_bundle<P: AsRef<Path>, Q: AsRef<Path>>(
    bundle_path: P,
    out_dir: Q,
) -> Result<BundleSummary> {
    let out_dir = out_dir.as_ref();
    let file = File::open(bundle_path.as_ref())?;
    // SAFETY: read-only mapping of a file we never write through, held only
    // for the duration of this call.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| Error::Other(format!("mmap failed: {e}")))?;
    let (header, header_end) = parse_bundle_header(&mmap)?;

    // Validate everything before the first write.
    let mut seen = std::collections::HashSet::with_capacity(header.objects.len());
    for o in &header.objects {
        validate_bundle_key(&o.key)?;
        if !seen.insert(o.key.as_str()) {
            return Err(Error::InvalidArchive(format!(
                "bundle header lists object key {:?} twice",
                o.key
            )));
        }
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
    }
    // Every manifest-referenced object must be present — fail before writing
    // a dataset that could never verify.
    let manifest_value: serde_json::Value = serde_json::from_str(header.manifest.get())
        .map_err(|e| Error::InvalidArchive(format!("bundle manifest JSON decode failed: {e}")))?;
    for k in manifest_object_keys(&manifest_value)? {
        if !seen.contains(k.key.as_str()) {
            return Err(Error::InvalidArchive(format!(
                "bundle header carries no object for manifest key {:?}",
                k.key
            )));
        }
    }

    fs::create_dir_all(out_dir)?;
    let mut bytes_written = 0u64;
    for o in &header.objects {
        let path = out_dir.join(&o.key);
        let parent = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| out_dir.to_path_buf());
        fs::create_dir_all(&parent)?;
        write_atomic(
            &parent,
            &path,
            &mmap[o.offset as usize..(o.offset + o.length) as usize],
        )?;
        bytes_written += o.length;
    }
    // Manifest last: its presence marks the exploded dataset complete.
    let manifest_bytes = header.manifest.get().as_bytes();
    write_atomic(out_dir, &out_dir.join("manifest.json"), manifest_bytes)?;
    bytes_written += manifest_bytes.len() as u64;

    Ok(BundleSummary {
        objects: header.objects.len(),
        bytes: bytes_written,
    })
}

/// Read just the (typed) manifest out of a `.sttb` bundle header.
pub fn read_bundle_manifest<P: AsRef<Path>>(bundle_path: P) -> Result<Manifest> {
    let file = File::open(bundle_path.as_ref())?;
    // SAFETY: read-only mapping, held only for the duration of this call.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| Error::Other(format!("mmap failed: {e}")))?;
    let (header, _) = parse_bundle_header(&mmap)?;
    Manifest::from_json_bytes(header.manifest.get().as_bytes())
}

/// Bundle analog of [`verify_packed_objects`]: verify a `.sttb`'s contents
/// against its embedded manifest with no trusted side-channel — each
/// content-addressed object's bytes must blake3-hash to its key, in-bundle
/// lengths must match the manifest-declared lengths, the directory must
/// decode (paged structure included), and no `pack_id` may fall outside the
/// pack table.
///
/// Returns the list of violations (empty ⇒ clean). Returns `Err` only when
/// the bundle container itself cannot be read or parsed (mirroring the
/// manifest-unreadable case of the exploded verifier).
pub fn verify_bundle_objects<P: AsRef<Path>>(bundle_path: P) -> Result<Vec<String>> {
    let file = File::open(bundle_path.as_ref())?;
    // SAFETY: read-only mapping, held only for the duration of this call.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| Error::Other(format!("mmap failed: {e}")))?;
    let (header, header_end) = parse_bundle_header(&mmap)?;
    let manifest = Manifest::from_json_bytes(header.manifest.get().as_bytes())?;

    let mut issues = Vec::new();

    if manifest.format != PACKED_FORMAT {
        issues.push(format!(
            "manifest format is {:?}, expected {PACKED_FORMAT:?}",
            manifest.format
        ));
    }
    if !SUPPORTED_PACKED_FORMAT_VERSIONS.contains(&manifest.format_version) {
        issues.push(format!(
            "manifest formatVersion is {}, expected one of {SUPPORTED_PACKED_FORMAT_VERSIONS:?}",
            manifest.format_version
        ));
    }
    if manifest.directory.directory_version != crate::directory::DIRECTORY_VERSION {
        issues.push(format!(
            "directoryVersion is {}, expected {}",
            manifest.directory.directory_version,
            crate::directory::DIRECTORY_VERSION
        ));
    }
    verify_manifest_schemas(&manifest, &mut issues);

    // Object table: key/window sanity, then key → slice for the checks below.
    let mut by_key: HashMap<&str, &[u8]> = HashMap::with_capacity(header.objects.len());
    for o in &header.objects {
        if let Err(e) = validate_bundle_key(&o.key) {
            issues.push(e.to_string());
            continue;
        }
        let end = match o.offset.checked_add(o.length) {
            Some(end) if o.offset >= header_end && end <= mmap.len() as u64 => end,
            _ => {
                issues.push(format!(
                    "bundle object {:?} window {}+{} outside the data region ({header_end}..{})",
                    o.key,
                    o.offset,
                    o.length,
                    mmap.len()
                ));
                continue;
            }
        };
        if by_key
            .insert(o.key.as_str(), &mmap[o.offset as usize..end as usize])
            .is_some()
        {
            issues.push(format!("bundle header lists object key {:?} twice", o.key));
        }
    }

    // Each content-addressed object: bytes must hash to the key's name and
    // match the manifest-declared length — exactly the exploded-directory
    // checks, sourced from bundle windows instead of files.
    fn check_object(
        by_key: &HashMap<&str, &[u8]>,
        key: &str,
        declared_len: u64,
        prefix: &str,
        ext: &str,
        issues: &mut Vec<String>,
    ) {
        match by_key.get(key) {
            Some(bytes) => {
                if bytes.len() as u64 != declared_len {
                    issues.push(format!(
                        "{key}: in-bundle length {} != manifest-declared {declared_len}",
                        bytes.len()
                    ));
                }
                let expected = format!("{prefix}/{}.{ext}", blake3_128_hex(bytes));
                if key != expected {
                    issues.push(format!(
                        "{key}: content-address mismatch (bytes hash to {expected})"
                    ));
                }
            }
            None => issues.push(format!("{key}: bundle header carries no such object")),
        }
    }

    check_object(
        &by_key,
        &manifest.directory.key,
        manifest.directory.length,
        "index",
        "sttd",
        &mut issues,
    );
    for p in &manifest.packs {
        check_object(&by_key, &p.key, p.length, "packs", "sttp", &mut issues);
    }

    // v2 objects must self-identify: validate each window's magic prelude.
    if manifest.format_version == PACKED_FORMAT_VERSION_V2 {
        for (key, kind) in std::iter::once((&manifest.directory.key, DIRECTORY_MAGIC))
            .chain(manifest.packs.iter().map(|p| (&p.key, PACK_MAGIC)))
        {
            if let Some(bytes) = by_key.get(key.as_str()) {
                if let Err(e) = strip_object_magic(bytes, kind, key) {
                    issues.push(e.to_string());
                }
            }
        }
    }

    // Directory must decode (through its v2 magic prelude, at-rest encoding +
    // container layout) and reference only packs the manifest lists; a paged
    // directory additionally passes the structural covering/order checks.
    if let Some(dir_bytes) = by_key.get(manifest.directory.key.as_str()) {
        match directory_codec_bytes(dir_bytes, manifest.format_version) {
            Ok(codec) => {
                match decode_directory_entries(codec, &manifest.directory) {
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
                                    .and_then(|p| by_key.get(p.key.as_str()))
                                    .map(|b| b.to_vec())
                            },
                            &mut issues,
                        );
                    }
                    Err(e) => issues.push(format!("directory failed to decode: {e}")),
                }
                if manifest.directory.is_paged() {
                    match manifest.directory.root_length {
                        Some(rl) => {
                            let zstd = manifest.directory.encoding.as_deref()
                                == Some(DIRECTORY_ENCODING_ZSTD);
                            match crate::directory_page::verify_paged_structure(codec, rl, zstd) {
                                Ok(mut more) => issues.append(&mut more),
                                Err(e) => issues.push(format!("paged structure check failed: {e}")),
                            }
                        }
                        None => issues.push("paged directory: manifest missing rootLength".into()),
                    }
                }
            }
            // The magic check above already reported the malformed prelude.
            Err(_) => {}
        }
    }

    Ok(issues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_tile::{
        encode_tile, encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn,
        PropertyColumn, FORMAT_VERSION_V2,
    };

    fn point_layer(name: &str, ids: Vec<u64>, t0: i64) -> ColumnarLayer {
        let n = ids.len();
        ColumnarLayer {
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
    fn distinct_tile(seed: u64) -> Vec<u8> {
        let ids: Vec<u64> = (0..6).map(|i| seed * 100 + i).collect();
        encode_tile(&[point_layer("default", ids, (seed as i64) * 7)]).unwrap()
    }

    /// v1-pinned writer for the tests whose payloads are bare `encode_tile`
    /// v1 frames (or opaque bytes): `add_tile_full` enforces frame/manifest
    /// version coherence, so these fixtures must not inherit the v2 default.
    fn v1_writer(out: &Path, ordering: BlobOrdering, pack_target_bytes: u64) -> PackWriter {
        PackWriter::create(out, ordering, pack_target_bytes)
            .unwrap()
            .with_format_version(PACKED_FORMAT_VERSION_V1)
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
            compression: "zstd".to_string(),
            blob_ordering: None,
            directory: DirectoryRef {
                key: "index/abc.sttd".to_string(),
                length: 42,
                directory_version: 5,
                encoding: Some(DIRECTORY_ENCODING_ZSTD.to_string()),
                layout: None,
                root_length: None,
                page_count: None,
                page_entries: None,
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
        assert_eq!(back.directory.directory_version, 5);
        assert_eq!(back.directory.encoding.as_deref(), Some("zstd"));
        assert_eq!(back.metadata.name, "manifest-test");

        // Backward compat: a pre-encoding manifest (no `encoding` key — the
        // shape of every deployed dataset) must parse with `encoding: None`,
        // and a None encoding must serialize without the key.
        let mut legacy_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        legacy_json["directory"]
            .as_object_mut()
            .unwrap()
            .remove("encoding");
        let legacy_back =
            Manifest::from_json_bytes(&serde_json::to_vec(&legacy_json).unwrap()).unwrap();
        assert_eq!(legacy_back.directory.encoding, None);
        // A pre-capabilities manifest (no `capabilities` key — the shape of
        // every deployed dataset) parses to an empty list.
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

        let mut w = v1_writer(&out, BlobOrdering::Auto, 8 * 1024);

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

    /// F1: `auto` measures the OCCUPIED spatial extent, not the raw max zoom. A
    /// dataset at a high zoom but a tiny spatial bbox over a deep-ish timeline
    /// resolves to spatial-major — which only holds when choose() sees the
    /// occupied bbox (~2 bits) rather than the 14-bit zoom (which would pick
    /// hilbert3). Also records the resolved order in the manifest (F4a).
    #[test]
    fn auto_resolves_from_occupied_extent_not_max_zoom() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("f1");
        let mut w = v1_writer(&out, BlobOrdering::Auto, 1 << 20);
        let bucket = 3_600_000i64;
        for x in 0..4u32 {
            for b in 0..64i64 {
                let t = b * bucket;
                let id = TileId::new(14, 10_000 + x, 20_000, t as u64);
                let payload = format!("f1-{x}-{b}").into_bytes();
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
            let mut w = v1_writer(&out, BlobOrdering::Auto, 1 << 20).with_measured_ordering(true);
            let bucket = 3_600_000i64;
            for x in 0..2u32 {
                for b in 0..24i64 {
                    let t = b * bucket;
                    let id = TileId::new(10, 5_000 + x, 6_000, t as u64);
                    let payload = format!("m-{x}-{b}").into_bytes();
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

    /// A **paged** directory build round-trips end-to-end through
    /// `PackedReader`: every input tile's payload decodes byte-identically, the
    /// manifest carries the container fields, and content-address verification
    /// is clean. Separately, a paged and a single build of the same input must
    /// agree byte-for-byte on everything except the directory object's
    /// container shape: identical pack content addresses and identical decoded
    /// entries. (Cross-finalize blob bytes ARE reproducible — the encoder feeds
    /// sorted metadata to Arrow ≥59's sorted-order IPC writer, guarded by
    /// `reproducible_build.rs` — so the old arrow-54 "compare only address
    /// keys" softening no longer applies.)
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
            let mut w = v1_writer(out, BlobOrdering::Auto, 16 * 1024).with_paging(page_entries);
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

        let mut w = v1_writer(&out, BlobOrdering::SpatialMajor, 4 * 1024);
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

        let mut w = v1_writer(&out, BlobOrdering::Auto, 64 * 1024 * 1024);
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

        let mut w = v1_writer(&out, BlobOrdering::Auto, 64 * 1024 * 1024);
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
            let mut w = v1_writer(&out, BlobOrdering::Auto, 8 * 1024);
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
            let mut w = v1_writer(&out, BlobOrdering::Auto, 8 * 1024).with_memory_budget(budget);
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
            let mut w = v1_writer(&out, BlobOrdering::Auto, 8 * 1024).with_memory_budget(1);
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

    /// ★F6 at WRITE time: `add_tile_full` refuses a layer frame whose version
    /// disagrees with the writer's declared formatVersion — in BOTH
    /// directions — so a mixed-version dataset can never be produced (it
    /// would brick on first read; readers hard-reject).
    #[test]
    fn add_tile_full_rejects_mixed_frame_and_writer_versions() {
        let dir = tempfile::tempdir().unwrap();
        let id = TileId::new(10, 0, 0, 0);
        let v1_payload = encode_tile_with(
            &[point_layer("default", vec![1, 2], 0)],
            &EncoderConfig::default(),
        )
        .unwrap();
        let v2_payload = encode_tile_with(
            &[point_layer("default", vec![1, 2], 0)],
            &EncoderConfig {
                format_version: FORMAT_VERSION_V2,
                ..EncoderConfig::default()
            },
        )
        .unwrap();

        // v1 frame → v2 (default) writer.
        let mut w2 =
            PackWriter::create(dir.path().join("v2"), BlobOrdering::Auto, 8 * 1024).unwrap();
        let err = w2
            .add_tile_full(&id, 0, 100, None, 2, None, &v1_payload)
            .unwrap_err();
        assert!(
            err.to_string().contains("v1 layer frame")
                && err.to_string().contains("formatVersion-2"),
            "got: {err}"
        );

        // v2 frame → v1 writer.
        let mut w1 = v1_writer(&dir.path().join("v1"), BlobOrdering::Auto, 8 * 1024);
        let err = w1
            .add_tile_full(&id, 0, 100, None, 2, None, &v2_payload)
            .unwrap_err();
        assert!(
            err.to_string().contains("v2 layer frame")
                && err.to_string().contains("formatVersion-1"),
            "got: {err}"
        );

        // Matching versions pass (both directions).
        w2.add_tile_full(&id, 0, 100, None, 2, None, &v2_payload)
            .unwrap();
        w1.add_tile_full(&id, 0, 100, None, 2, None, &v1_payload)
            .unwrap();
    }

    /// The directory ships zstd-compressed at rest (declared via
    /// `directory.encoding`), and a legacy manifest with a RAW directory and
    /// no `encoding` key — the shape of every deployed dataset — must still
    /// open and verify. formatVersion 1 throughout: the raw-directory legacy
    /// shape predates v2 (whose objects always carry the magic prelude).
    #[test]
    fn directory_encoding_compressed_and_raw_both_read() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 8 * 1024)
            .unwrap()
            .with_format_version(PACKED_FORMAT_VERSION_V1);
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
        let raw = compression::decompress_zstd_with_dict(&at_rest, None).unwrap();
        assert!(crate::directory::decode_directory(&raw).is_ok());
        assert!(verify_packed_objects(&manifest_path).unwrap().is_empty());
        let entries_compressed = PackedReader::open(&manifest_path)
            .unwrap()
            .entries()
            .to_vec();

        // Rewrite the dataset as a legacy (raw-directory) manifest: store the
        // raw codec bytes under their own content address and drop `encoding`.
        let raw_hex = blake3_128_hex(&raw);
        let raw_rel = format!("index/{raw_hex}.sttd");
        fs::write(out.join(&raw_rel), &raw).unwrap();
        let mut legacy = manifest.clone();
        legacy.directory = DirectoryRef {
            key: raw_rel,
            length: raw.len() as u64,
            directory_version: crate::directory::DIRECTORY_VERSION,
            encoding: None,
            layout: None,
            root_length: None,
            page_count: None,
            page_entries: None,
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

        let mut w = v1_writer(&out, BlobOrdering::Auto, 8 * 1024);
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
        v["formatVersion"] = serde_json::json!(3);
        fs::write(&manifest_path, serde_json::to_vec(&v).unwrap()).unwrap();

        let err = match PackedReader::open(&manifest_path) {
            Ok(_) => panic!("formatVersion 3 must be rejected at open"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("formatVersion"),
            "unexpected error: {err}"
        );
    }

    /// The manifest's `formatVersion` is AUTHORITATIVE (design §1 ★F6):
    /// doctoring a v1 dataset's manifest to claim v2 must fail loudly at open
    /// (the v1 objects carry no magic prelude), never best-effort decode.
    #[test]
    fn v1_dataset_doctored_to_claim_v2_is_rejected_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 8 * 1024)
            .unwrap()
            .with_format_version(PACKED_FORMAT_VERSION_V1);
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

        let mut v: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        v["formatVersion"] = serde_json::json!(2);
        fs::write(&manifest_path, serde_json::to_vec(&v).unwrap()).unwrap();

        let err = match PackedReader::open(&manifest_path) {
            Ok(_) => panic!("v2-declared manifest over magic-less v1 objects must be rejected"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("STTD"), "unexpected error: {err}");
    }

    /// `with_capabilities` declares required-to-understand features in the
    /// manifest: canonicalized (sorted + deduped) for byte-reproducibility,
    /// present in the JSON, and accepted by a reader that implements them.
    #[test]
    fn writer_declares_capabilities_canonically() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = v1_writer(&out, BlobOrdering::Auto, 8 * 1024)
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

    /// `PackedReader::open` must reject a dataset declaring a capability this
    /// reader does not implement, naming the unknown entries — a capability
    /// re-types existing columns, so proceeding would silently misdecode.
    #[test]
    fn unknown_capability_is_rejected_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = v1_writer(&out, BlobOrdering::Auto, 8 * 1024);
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

        let mut w = v1_writer(&out, BlobOrdering::Auto, 8 * 1024);
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
        // v1: the fixture's payloads are v1 frames (bare `encode_tile`), and
        // the bundle container is deliberately format-version-agnostic — v2
        // bundle coverage lives in `v2_bundle_roundtrips_and_verifies`.
        let mut w = PackWriter::create(out, BlobOrdering::Auto, 8 * 1024)
            .unwrap()
            .with_format_version(PACKED_FORMAT_VERSION_V1);
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
                v1_writer(&src, BlobOrdering::Auto, 8 * 1024).with_paging(paged.then_some(4));
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

        // Flip the first byte of the first pack's window.
        let bytes = fs::read(&bundle).unwrap();
        let (header, _) = parse_bundle_header(&bytes).unwrap();
        let pack0 = header
            .objects
            .iter()
            .find(|o| o.key.starts_with("packs/"))
            .expect("a pack object");
        let mut corrupt = bytes.clone();
        corrupt[pack0.offset as usize] ^= 0xff;
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

    // ---- formatVersion 2 (template-referencing frames + object magic) ------

    /// A quantized point layer with numeric + categorical properties — the
    /// shape whose per-tile `stt:qa`/`stt:time_offset_ms` variance v2 hoists
    /// into TILE_META. `seed` shifts values so every tile's affines differ.
    fn v2_point_layer(seed: u64, n: usize) -> ColumnarLayer {
        let base = 1_700_000_000_000i64 + seed as i64 * 60_000;
        ColumnarLayer {
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

    /// The layer set the v2 fixture builds: quantized point tiles with
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
            format_version: FORMAT_VERSION_V2,
            template_collector: Some(w.template_collector()),
            ..EncoderConfig::default()
        }
    }

    /// Build a small v2 dataset (encoder wired to the writer's template
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

    /// End-to-end v2: magic on every object, schemas embedded (sorted,
    /// hash-valid), verify clean, and every tile decodes through the registry
    /// to EXACTLY what the v1 frame of the same layer decodes to (metadata
    /// re-injection contract).
    #[test]
    fn v2_dataset_roundtrips_with_templates_magic_and_v1_equivalence() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let (manifest, layers) = build_v2_dataset(&out, None);

        assert_eq!(manifest.format_version, PACKED_FORMAT_VERSION_V2);
        // Templates collected: one CORE + one PROPS schema for the shared
        // layer shape (constancy: five differing tiles, ONE template pair).
        assert_eq!(manifest.schemas.len(), 2, "expected core+props templates");
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
        assert_eq!(reader.format_version(), PACKED_FORMAT_VERSION_V2);
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

        // Decode parity: each tile equals the v1 decode of the same layer.
        for (k, layer) in layers.iter().enumerate() {
            let e = reader
                .entries()
                .iter()
                .find(|e| e.x == k as u32)
                .expect("entry present");
            let got = reader.read_layers(e).unwrap();
            let v1 = crate::arrow_tile::decode_tile(
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
            assert_eq!(got.len(), v1.len());
            assert_eq!(got[0].name, v1[0].name);
            assert_eq!(
                got[0].batch, v1[0].batch,
                "tile {k}: v2 decode must equal the v1 decode after re-injection"
            );
        }
    }

    /// ★F6: the manifest's formatVersion is authoritative — a frame of the
    /// OTHER version inside a dataset is a hard error in both directions.
    #[test]
    fn mixed_frame_and_manifest_versions_hard_error() {
        let dir = tempfile::tempdir().unwrap();

        // v2 dataset refuses a v1 frame…
        let v2_out = dir.path().join("v2");
        build_v2_dataset(&v2_out, None);
        let v2_reader = PackedReader::open(v2_out.join("manifest.json")).unwrap();
        let v1_payload = encode_tile(&[point_layer("default", vec![1, 2], 0)]).unwrap();
        let err = v2_reader.decode_payload(&v1_payload).unwrap_err();
        assert!(err.to_string().contains("v1 layer frame"), "got: {err}");

        // …and a v1 dataset refuses a v2 frame.
        let v1_out = dir.path().join("v1");
        let mut w = PackWriter::create(&v1_out, BlobOrdering::Auto, 8 * 1024)
            .unwrap()
            .with_format_version(PACKED_FORMAT_VERSION_V1);
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
        w.finalize(&Metadata::new("v1")).unwrap();
        let v1_reader = PackedReader::open(v1_out.join("manifest.json")).unwrap();
        let v2_payload = encode_tile_with(
            &[v2_point_layer(0, 3)],
            &EncoderConfig {
                format_version: FORMAT_VERSION_V2,
                ..EncoderConfig::default()
            },
        )
        .unwrap();
        let err = v1_reader.decode_payload(&v2_payload).unwrap_err();
        assert!(err.to_string().contains("v2 layer frame"), "got: {err}");
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

    /// Corrupting a v2 object's magic prelude is caught at open/read AND by
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

    /// The paged directory container works identically under v2 (magic
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

    /// v2 bundles: pack → verify → open → unpack → re-verify, all through
    /// the embedded v2 manifest (design §6's open item).
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
        assert_eq!(bundled.format_version(), PACKED_FORMAT_VERSION_V2);
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
            "unpacked v2 dataset must re-verify"
        );
    }

    /// The verbatim-repack path (pack-cover / repair tools): payloads copied
    /// from a v2 source MUST carry the source's formatVersion forward and
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
            .with_format_version(reader.format_version())
            .with_capabilities(reader.capabilities().to_vec())
            .with_seeded_templates(reader.templates().expect("v2 source has a registry"));
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
        assert_eq!(manifest.format_version, PACKED_FORMAT_VERSION_V2);
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

    /// F4: a v2 dataset whose frames reference a template MISSING from
    /// `manifest.schemas` is undecodable — both validators must flag the
    /// absent hash instead of verifying clean (every object still hashes to
    /// its content address and the remaining schemas table is valid).
    #[test]
    fn verify_flags_missing_frame_referenced_schema_template() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let (manifest, _) = build_v2_dataset(&out, None);
        assert_eq!(manifest.schemas.len(), 2, "core + props templates");
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
}
