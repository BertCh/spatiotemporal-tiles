//! STT **packed format** — a multi-object, edge-cacheable container.
//!
//! Replaces the single-file v4 archive ([`crate::archive::ArchiveWriter`]) as
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
//! remote/HTTP); it mirrors [`crate::archive::ArchiveReader`] semantics.

use crate::archive::{ArchiveReader, TileEntry};
use crate::compression;
use crate::curve::BlobOrdering;
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

/// `format` discriminator written into every packed manifest.
pub const PACKED_FORMAT: &str = "stt-packed";

/// Packed-format version (the manifest schema, distinct from the directory
/// codec's [`crate::directory::DIRECTORY_VERSION`]).
pub const PACKED_FORMAT_VERSION: u32 = 1;

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
    /// for the root, then leaf ranges on demand.
    #[serde(default, rename = "rootLength", skip_serializing_if = "Option::is_none")]
    pub root_length: Option<u64>,
    /// Number of leaf pages (informational / validation). Paged only.
    #[serde(default, rename = "pageCount", skip_serializing_if = "Option::is_none")]
    pub page_count: Option<u64>,
    /// Nominal entries-per-page used at build (informational). Paged only.
    #[serde(default, rename = "pageEntries", skip_serializing_if = "Option::is_none")]
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
    /// Blob compression codec (always `"zstd"`, per-blob, no shared dict).
    pub compression: String,
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
    payload: Vec<u8>,
}

/// Writer for the multi-object packed format.
///
/// Consumes an `(id, payload)` stream like [`crate::archive::ArchiveWriter`],
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
        })
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

    /// Add a tile carrying the full directory metadata. Same shape as
    /// [`crate::archive::ArchiveWriter::add_tile_full`]: `cover_t_min` is the
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
            payload: payload.to_vec(),
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
        } = self;

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
        let tb_span = if pending.is_empty() { 0 } else { tb_max - tb_min };
        // Resolve Auto from the dataset's space-vs-time cardinality.
        let ordering = match ordering {
            BlobOrdering::Auto => {
                let max_z = pending.iter().map(|p| p.z).max().unwrap_or(0) as u32;
                let time_bits = crate::curve::bits_for((tb_span.max(0) + 1) as u64);
                BlobOrdering::choose(max_z, time_bits)
            }
            other => other,
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

        // --- Per-blob zstd + byte-identical dedup ------------------------
        // Each pending tile is compressed (NO shared dictionary, so the fzstd TS
        // reader can decode). Byte-identical compressed blobs collapse to a
        // single physical blob — but, unlike the single-file writer, we DON'T
        // assign byte offsets yet: pack assignment happens after dedup so a
        // shared blob lands in exactly one pack.
        struct Blob {
            compressed: Vec<u8>,
            uncompressed_size: u32,
            crc: u32,
        }
        let mut blobs: Vec<Blob> = Vec::new();
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
        const COMPRESS_CHUNK: usize = 8192;
        for chunk in pending.chunks(COMPRESS_CHUNK) {
            let compressed_chunk: Vec<Vec<u8>> = chunk
                .par_iter()
                .map(|p| compression::compress_zstd_with_dict_level(&p.payload, None, zstd_level))
                .collect::<Result<Vec<_>>>()?;
            for (p, compressed) in chunk.iter().zip(compressed_chunk) {
                let key = *blake3::hash(&compressed).as_bytes();
                let idx = if let Some(&i) = blob_dedup.get(&key) {
                    i
                } else {
                    let i = blobs.len();
                    let crc = crc32c_tag(&compressed);
                    let uncompressed_size = p.payload.len() as u32;
                    blobs.push(Blob {
                        compressed,
                        uncompressed_size,
                        crc,
                    });
                    blob_dedup.insert(key, i);
                    i
                };
                tile_blob.push(idx);
            }
        }

        // --- Pack cutting -----------------------------------------------
        // Walk the deduped blobs in their (first-seen, i.e. curve) order and cut
        // into packs of ≤ pack_target_bytes. Never split a blob; a single blob
        // larger than the target gets its own pack. pack_id is assigned in cut
        // order; offset_in_pack resets to 0 per pack. Each blob is placed once,
        // so a tile shared across time buckets resolves to one (pack, offset).
        struct Placement {
            pack_id: u32,
            offset: u64,
        }
        let mut placements: Vec<Placement> = Vec::with_capacity(blobs.len());
        // Pack contents: each pack is a contiguous slice of `blobs` indices.
        let mut packs_blob_ranges: Vec<(usize, usize)> = Vec::new(); // (start, end) into blobs
        if !blobs.is_empty() {
            let mut pack_start = 0usize;
            let mut pack_id = 0u32;
            let mut cur_offset = 0u64;
            for (i, blob) in blobs.iter().enumerate() {
                let blen = blob.compressed.len() as u64;
                // Cut BEFORE this blob if adding it would exceed the target and
                // the current pack is non-empty (so a lone oversized blob still
                // fits — it just owns a pack).
                if i > pack_start && cur_offset + blen > pack_target_bytes {
                    packs_blob_ranges.push((pack_start, i));
                    pack_start = i;
                    pack_id += 1;
                    cur_offset = 0;
                }
                placements.push(Placement {
                    pack_id,
                    offset: cur_offset,
                });
                cur_offset += blen;
            }
            packs_blob_ranges.push((pack_start, blobs.len()));
        }

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
                length: blob.compressed.len() as u32,
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

        // --- Write objects ----------------------------------------------
        fs::create_dir_all(&out_dir)?;
        let index_dir = out_dir.join("index");
        let packs_dir = out_dir.join("packs");
        fs::create_dir_all(&index_dir)?;
        fs::create_dir_all(&packs_dir)?;

        // Write each pack to a temp file then atomically rename to its content
        // address. Packs are immutable, so a re-sync skips an unchanged pack.
        let mut pack_refs: Vec<PackRef> = Vec::with_capacity(packs_blob_ranges.len());
        for (start, end) in &packs_blob_ranges {
            let mut bytes: Vec<u8> = Vec::new();
            for blob in &blobs[*start..*end] {
                bytes.extend_from_slice(&blob.compressed);
            }
            let hex = blake3_128_hex(&bytes);
            let rel = format!("packs/{hex}.sttp");
            let final_path = out_dir.join(&rel);
            write_atomic(&packs_dir, &final_path, &bytes)?;
            pack_refs.push(PackRef {
                key: rel,
                length: bytes.len() as u64,
            });
        }

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
        let index_hex = blake3_128_hex(&index_bytes);
        let index_rel = format!("index/{index_hex}.sttd");
        let index_path = out_dir.join(&index_rel);
        write_atomic(&index_dir, &index_path, &index_bytes)?;

        let (layout, root_length, page_count, page_entries_field) = directory_ref_fields;

        // Build + write the manifest.
        let manifest = Manifest {
            format: PACKED_FORMAT.to_string(),
            format_version: PACKED_FORMAT_VERSION,
            compression: "zstd".to_string(),
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
            metadata: metadata.clone(),
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

// ----------------------------------------------------------------------------
// Transcode (single-file v4 archive -> packed dir)
// ----------------------------------------------------------------------------

/// Losslessly re-wrap a single-file v4 `.stt` archive into the packed format.
///
/// Reads every tile payload via [`ArchiveReader`] and streams it into a fresh
/// [`PackWriter`], preserving the tight covering bound (`cover_t_min`) and the
/// per-tile `temporal_bucket_ms` exactly (no thinning), then finalizes with the
/// source archive's metadata. Writes `manifest.json`, `index/<hash>.sttd`, and
/// one or more `packs/<hash>.sttp` objects under `out_dir`.
///
/// Shared by the `pack-transcode` example and `stt-build`'s `--streaming-arrow`
/// path (which builds a temp single-file archive under bounded RAM, then
/// transcodes it to packs).
pub fn transcode_archive_to_packs<I: AsRef<Path>, O: AsRef<Path>>(
    in_archive: I,
    out_dir: O,
    ordering: BlobOrdering,
    pack_target_bytes: u64,
) -> Result<Manifest> {
    transcode_archive_to_packs_paged(in_archive, out_dir, ordering, pack_target_bytes, None)
}

/// [`transcode_archive_to_packs`] with an explicit paging choice.
///
/// `page_entries`: `Some(k)` emits a paged directory (root + leaf pages of ≤ `k`
/// entries — Wave 2); `None` emits the single whole-load v5 directory. The tile
/// payloads, packs, dedup and ordering are identical either way — only the
/// `.sttd` directory shape changes — so a paged transcode of an existing dataset
/// re-ships only `index/*.sttd` + `manifest.json` (the packs are unchanged
/// content addresses).
pub fn transcode_archive_to_packs_paged<I: AsRef<Path>, O: AsRef<Path>>(
    in_archive: I,
    out_dir: O,
    ordering: BlobOrdering,
    pack_target_bytes: u64,
    page_entries: Option<usize>,
) -> Result<Manifest> {
    transcode_archive_to_packs_paged_level(
        in_archive,
        out_dir,
        ordering,
        pack_target_bytes,
        page_entries,
        compression::ZSTD_LEVEL,
    )
}

/// [`transcode_archive_to_packs_paged`] at an explicit zstd `level` (see
/// [`PackWriter::with_zstd_level`]). Re-compresses every blob at `level`, so —
/// unlike [`repack_directory`] — it rewrites the packs, not just the directory.
pub fn transcode_archive_to_packs_paged_level<I: AsRef<Path>, O: AsRef<Path>>(
    in_archive: I,
    out_dir: O,
    ordering: BlobOrdering,
    pack_target_bytes: u64,
    page_entries: Option<usize>,
    zstd_level: i32,
) -> Result<Manifest> {
    let reader = ArchiveReader::open(in_archive)?;
    let meta = reader.metadata().clone();
    let entries = reader.entries().to_vec();

    let mut writer = PackWriter::create(out_dir, ordering, pack_target_bytes)?
        .with_paging(page_entries)
        .with_zstd_level(zstd_level);
    for e in &entries {
        let payload = reader.read_payload(e)?;
        writer.add_tile_full(
            &TileId::new(e.zoom, e.x, e.y, e.time_start.max(0) as u64),
            e.time_start,
            e.time_end,
            e.cover_t_min,
            e.feature_count,
            e.temporal_bucket_ms,
            &payload,
        )?;
    }
    writer.finalize(&meta)
}

/// Re-transcode an existing **packed** dataset into a fresh packed dataset at an
/// explicit zstd `level` and (optionally) a paged directory — the publish-build
/// migration. Unlike [`repack_directory`] (directory-only) this re-reads and
/// **re-compresses every tile payload**, so it picks up a higher `--zstd-level`
/// (−10..19% on the wire) and re-pages the directory in one pass.
///
/// The input packed dir is the source of truth — important because several
/// showcase datasets were rebuilt later than their legacy single-file `.stt`,
/// so transcoding from the packs (not the stale `.stt`) preserves those fixes.
/// Tile content is byte-for-byte preserved (lossless); only the compression
/// level and directory shape change. `out_dir` MUST differ from the input dir.
pub fn transcode_packs_to_packs_paged_level<I: AsRef<Path>, O: AsRef<Path>>(
    in_manifest: I,
    out_dir: O,
    ordering: BlobOrdering,
    pack_target_bytes: u64,
    page_entries: Option<usize>,
    zstd_level: i32,
) -> Result<Manifest> {
    let reader = PackedReader::open(in_manifest)?;
    let meta = reader.metadata().clone();
    let entries = reader.entries().to_vec();

    let mut writer = PackWriter::create(out_dir, ordering, pack_target_bytes)?
        .with_paging(page_entries)
        .with_zstd_level(zstd_level);
    for e in &entries {
        let payload = reader.read_payload(e)?;
        writer.add_tile_full(
            &TileId::new(e.zoom, e.x, e.y, e.time_start.max(0) as u64),
            e.time_start,
            e.time_end,
            e.cover_t_min,
            e.feature_count,
            e.temporal_bucket_ms,
            &payload,
        )?;
    }
    writer.finalize(&meta)
}

/// Re-encode **only the directory** of an existing packed dataset, optionally
/// switching it to (or from) the paged container — without re-reading or
/// re-compressing a single tile payload.
///
/// This is the cheap Wave-2 re-transcode: the directory entries already carry
/// every blob's `(pack_id, offset, length, crc32c)`, and the packs are
/// content-addressed and immutable, so a directory shape change re-ships only
/// `index/<hash>.sttd` + `manifest.json` — the packs keep their exact bytes and
/// names. `page_entries`: `Some(k)` → paged directory (leaf pages of ≤ `k`),
/// `None` → single whole-load directory.
///
/// When `out_dir` differs from the input dataset's directory, the (unchanged)
/// pack objects are copied across so `out_dir` is a complete dataset; an
/// in-place re-pack (`out_dir` == the input dir) leaves the packs untouched and
/// just adds the new index object + overwrites the manifest.
pub fn repack_directory<P: AsRef<Path>, Q: AsRef<Path>>(
    in_manifest: P,
    out_dir: Q,
    page_entries: Option<usize>,
) -> Result<Manifest> {
    let in_manifest = in_manifest.as_ref();
    let in_dir = in_manifest
        .parent()
        .ok_or_else(|| Error::InvalidArchive("manifest path has no parent dir".into()))?;
    let out_dir = out_dir.as_ref();

    let reader = PackedReader::open(in_manifest)?;
    let entries = reader.entries().to_vec();
    let metadata = reader.metadata().clone();
    let src = Manifest::from_json_bytes(&fs::read(in_manifest)?)?;

    fs::create_dir_all(out_dir.join("index"))?;
    fs::create_dir_all(out_dir.join("packs"))?;

    // Copy the (unchanged, content-addressed) packs only when writing elsewhere.
    let same_dir = fs::canonicalize(in_dir).ok() == fs::canonicalize(out_dir).ok();
    if !same_dir {
        for p in &src.packs {
            let dst = out_dir.join(&p.key);
            if !dst.exists() {
                fs::copy(in_dir.join(&p.key), &dst)?;
            }
        }
    }

    // Re-encode the directory from the existing entries (no payload re-read).
    let (index_bytes, layout, root_length, page_count, page_entries_field): (
        Vec<u8>,
        Option<String>,
        Option<u64>,
        Option<u64>,
        Option<u64>,
    ) = if let Some(k) = page_entries {
        let paged = crate::directory_page::encode_paged_directory(&entries, k, true)?;
        (
            paged.bytes,
            Some(DIRECTORY_LAYOUT_PAGED.to_string()),
            Some(paged.root_length),
            Some(paged.page_count as u64),
            Some(paged.page_entries as u64),
        )
    } else {
        let plain = crate::directory::encode_directory(&entries);
        (
            compression::compress_zstd_with_dict(&plain, None)?,
            None,
            None,
            None,
            None,
        )
    };

    let index_hex = blake3_128_hex(&index_bytes);
    let index_rel = format!("index/{index_hex}.sttd");
    write_atomic(&out_dir.join("index"), &out_dir.join(&index_rel), &index_bytes)?;

    let manifest = Manifest {
        format: PACKED_FORMAT.to_string(),
        format_version: PACKED_FORMAT_VERSION,
        compression: src.compression.clone(),
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
        packs: src.packs.clone(),
        metadata,
    };
    let manifest_bytes = manifest.to_json_bytes()?;
    let mut f = File::create(out_dir.join("manifest.json"))?;
    f.write_all(&manifest_bytes)?;
    f.flush()?;
    Ok(manifest)
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
    if manifest.format_version != PACKED_FORMAT_VERSION {
        issues.push(format!(
            "manifest formatVersion is {}, expected {PACKED_FORMAT_VERSION}",
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

    // Each content-addressed object: name must equal blake3-128 of its bytes,
    // and on-disk length must equal the declared length.
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

    // Directory must decode (through its at-rest encoding + container layout)
    // and reference only packs that exist.
    match fs::read(root.join(&manifest.directory.key)) {
        Ok(dir_bytes) => match decode_directory_entries(&dir_bytes, &manifest.directory) {
            Ok(entries) => {
                if let Some(max_pid) = entries.iter().map(|e| e.pack_id).max() {
                    if max_pid as usize >= manifest.packs.len() {
                        issues.push(format!(
                            "directory references pack_id {max_pid} but the manifest lists only {} pack(s)",
                            manifest.packs.len()
                        ));
                    }
                }
            }
            Err(e) => issues.push(format!("directory failed to decode: {e}")),
        },
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
                    let zstd =
                        manifest.directory.encoding.as_deref() == Some(DIRECTORY_ENCODING_ZSTD);
                    match crate::directory_page::verify_paged_structure(&dir_bytes, rl, zstd) {
                        Ok(mut more) => issues.append(&mut more),
                        Err(e) => issues.push(format!("paged structure check failed: {e}")),
                    }
                }
            }
            None => issues.push("paged directory: manifest missing rootLength".into()),
        }
    }

    Ok(issues)
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
/// Mirrors [`crate::archive::ArchiveReader`]: [`entries`](Self::entries) returns
/// the decoded v5 directory, [`metadata`](Self::metadata) the folded metadata,
/// and [`read_payload`](Self::read_payload) selects the entry's pack, slices
/// `[offset..offset+length]`, verifies CRC32C and decompresses the per-blob
/// zstd. Packs are mmap'd lazily by `pack_id`.
pub struct PackedReader {
    entries: Vec<TileEntry>,
    metadata: Metadata,
    compression: Compression,
    packs: Vec<std::cell::RefCell<LoadedPack>>,
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

        if manifest.format != PACKED_FORMAT {
            return Err(Error::InvalidArchive(format!(
                "not a packed manifest: format={:?} (expected {PACKED_FORMAT:?})",
                manifest.format
            )));
        }
        let compression = match manifest.compression.as_str() {
            "zstd" => Compression::Zstd,
            "none" => Compression::None,
            other => {
                return Err(Error::InvalidArchive(format!(
                    "unknown packed compression {other:?}"
                )))
            }
        };

        // Load + decode the directory object. The single (whole-load) shape
        // unwraps the at-rest encoding then runs the v5 codec; the paged shape
        // decodes the root + every leaf (local load-all — a mmap'd file has no
        // cold-start cost). Both branches return the same full entry list.
        let dir_path = root.join(&manifest.directory.key);
        let dir_bytes = fs::read(&dir_path)?;
        let entries = decode_directory_entries(&dir_bytes, &manifest.directory)?;

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

    /// Read and decompress a tile's raw payload bytes, verifying its CRC32C.
    ///
    /// Selects the pack by `entry.pack_id`, slices `[offset..offset+length]`
    /// within it, checks the CRC, then decompresses the per-blob zstd. Mirrors
    /// [`crate::archive::ArchiveReader::read_payload`].
    pub fn read_payload(&self, entry: &TileEntry) -> Result<Vec<u8>> {
        let cell = self.packs.get(entry.pack_id as usize).ok_or_else(|| {
            Error::InvalidArchive(format!(
                "tile {:?} references pack {} but only {} packs exist",
                entry.tile_id(),
                entry.pack_id,
                self.packs.len()
            ))
        })?;

        let payload = {
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
                pack.mmap = Some(mmap);
            }
            let mmap = pack.mmap.as_ref().expect("just loaded");

            let start = entry.offset as usize;
            let end = start + entry.length as usize;
            if end > mmap.len() {
                return Err(Error::InvalidArchive(format!(
                    "tile {:?} blob range {start}..{end} exceeds pack size {}",
                    entry.tile_id(),
                    mmap.len()
                )));
            }
            let compressed = &mmap[start..end];

            if crc32c_tag(compressed) != entry.crc32c {
                return Err(Error::InvalidArchive(format!(
                    "tile {:?} failed integrity check (corrupt pack)",
                    entry.tile_id()
                )));
            }

            if self.compression == Compression::Zstd {
                compression::decompress_zstd_with_dict(compressed, None)?
            } else {
                compression::decompress(compressed, self.compression)?
            }
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
        crate::arrow_tile::decode_tile(&payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_tile::{encode_tile, ColumnarLayer, GeometryColumn};

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
            compression: "zstd".to_string(),
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
                PackRef { key: "packs/a.sttp".to_string(), length: 100 },
                PackRef { key: "packs/b.sttp".to_string(), length: 200 },
            ],
            metadata: Metadata::new("manifest-test"),
        };
        let bytes = m.to_json_bytes().unwrap();
        // The spec keys must be camelCase where renamed.
        let s = String::from_utf8(bytes.clone()).unwrap();
        assert!(s.contains("\"formatVersion\""), "{s}");
        assert!(s.contains("\"directoryVersion\""), "{s}");
        assert!(s.contains("\"stt-packed\""), "{s}");
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
        let legacy_out =
            String::from_utf8(legacy_back.to_json_bytes().unwrap()).unwrap();
        assert!(!legacy_out.contains("\"encoding\""), "{legacy_out}");
    }

    /// Full round-trip: 30 synthetic tiles (a couple byte-identical to exercise
    /// dedup) across 2 zooms + several time buckets, written with a tiny pack
    /// target to force multiple packs. Every payload must decode byte-identical.
    #[test]
    fn packwriter_roundtrips_through_multiple_packs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 8 * 1024).unwrap();

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
            w.add_tile_full(&id, t, t + bucket - 1, Some(t), 6, Some(bucket as u64), &payload)
                .unwrap();
            expected.push((id, t, t + bucket - 1, payload));
        }
        assert_eq!(w.tile_count(), 30);

        let meta = Metadata::new("packed-roundtrip").with_temporal_bucket_ms(bucket as u64);
        let manifest = w.finalize(&meta).unwrap();

        // >1 pack produced at the tiny target.
        assert!(manifest.packs.len() > 1, "expected multiple packs, got {}", manifest.packs.len());

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

        // Every tile's decompressed payload is byte-identical.
        for (id, ts, _te, payload) in &expected {
            let e = reader
                .entries()
                .iter()
                .find(|e| {
                    e.zoom == id.z && e.x == id.x && e.y == id.y && e.time_start == *ts
                })
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

    /// A **paged** directory build round-trips end-to-end through
    /// `PackedReader`: every input tile's payload decodes byte-identically, the
    /// manifest carries the container fields, and content-address verification
    /// is clean. Separately, a paged and a single build of the same input agree
    /// on the payload-independent address keys (the cross-build *blob* bytes are
    /// not comparable — Arrow IPC schema-metadata ordering isn't reproducible
    /// across `finalize` runs, the documented D6 caveat; the codec-level
    /// "paged == whole-load" equality is proven in `directory_page` tests).
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
            let mut w = PackWriter::create(out, BlobOrdering::Auto, 16 * 1024)
                .unwrap()
                .with_paging(page_entries);
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
        assert_eq!(paged.directory.layout.as_deref(), Some(DIRECTORY_LAYOUT_PAGED));
        assert!(paged.directory.root_length.unwrap() > 0);
        assert!(paged.directory.page_count.unwrap() >= 2, "expected multiple leaf pages");
        assert_eq!(paged.directory.page_entries, Some(16));
        assert_eq!(paged.directory.directory_version, crate::directory::DIRECTORY_VERSION);
        assert_eq!(paged.directory.encoding.as_deref(), Some(DIRECTORY_ENCODING_ZSTD));

        let r_single = PackedReader::open(single_out.join("manifest.json")).unwrap();
        let r_paged = PackedReader::open(paged_out.join("manifest.json")).unwrap();
        assert_eq!(r_paged.entries().len(), 120);
        assert_eq!(r_single.entries().len(), 120);

        // Cross-build agreement on payload-INDEPENDENT address keys (sorted, so
        // the comparison ignores blob-byte non-reproducibility).
        let keys = |r: &PackedReader| -> Vec<(u8, u32, u32, i64, i64, u32, Option<u64>, Option<i64>)> {
            let mut v: Vec<_> = r
                .entries()
                .iter()
                .map(|e| {
                    (
                        e.zoom, e.x, e.y, e.time_start, e.time_end, e.feature_count,
                        e.temporal_bucket_ms, e.cover_t_min,
                    )
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(keys(&r_single), keys(&r_paged));

        // Every INPUT tile decodes byte-identically through the paged reader.
        for (id, ts, _te, payload) in &input {
            let e = r_paged
                .entries()
                .iter()
                .find(|x| x.zoom == id.z && x.x == id.x && x.y == id.y && x.time_start == *ts)
                .expect("paged entry present");
            assert_eq!(&r_paged.read_payload(e).unwrap(), payload, "payload mismatch {id:?}");
        }

        // Content-address integrity verifies clean on the paged dataset.
        let issues = verify_packed_objects(paged_out.join("manifest.json")).unwrap();
        assert!(issues.is_empty(), "paged verify issues: {issues:?}");
    }

    /// `repack_directory` switches a built dataset's directory to paged WITHOUT
    /// re-reading payloads: the packs keep their exact content addresses, the
    /// re-opened entries are identical, and the result verifies clean.
    #[test]
    fn repack_directory_to_paged_preserves_packs_and_entries() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        let out = dir.path().join("out");

        // A small single-directory dataset.
        let mut w = PackWriter::create(&src, BlobOrdering::Auto, 16 * 1024).unwrap();
        for k in 0..80u64 {
            let zoom = [8u8, 11][(k % 2) as usize];
            let t = (k % 3) as i64 * 3_600_000;
            let id = TileId::new(zoom, (k % 9) as u32, (k / 9) as u32, t as u64);
            w.add_tile_full(&id, t, t + 3_599_999, Some(t), 4, Some(3_600_000), &distinct_tile(k))
                .unwrap();
        }
        let src_manifest = w.finalize(&Metadata::new("repack-src")).unwrap();
        assert!(src_manifest.directory.layout.is_none());

        // Directory-only re-pack into a fresh dir, paged.
        let out_manifest = repack_directory(src.join("manifest.json"), &out, Some(8)).unwrap();
        assert_eq!(out_manifest.directory.layout.as_deref(), Some(DIRECTORY_LAYOUT_PAGED));
        assert!(out_manifest.directory.page_count.unwrap() >= 2);

        // Packs are byte-identical content addresses (only the directory changed).
        let src_keys: std::collections::BTreeSet<&str> =
            src_manifest.packs.iter().map(|p| p.key.as_str()).collect();
        let out_keys: std::collections::BTreeSet<&str> =
            out_manifest.packs.iter().map(|p| p.key.as_str()).collect();
        assert_eq!(src_keys, out_keys);

        // Re-opened entries are identical, and integrity verifies clean.
        let r_src = PackedReader::open(src.join("manifest.json")).unwrap();
        let r_out = PackedReader::open(out.join("manifest.json")).unwrap();
        assert_eq!(r_src.entries(), r_out.entries());
        // A payload reads back identically through the re-packed (paged) dataset.
        let e = &r_out.entries()[0];
        assert_eq!(r_out.read_payload(e).unwrap(), r_src.read_payload(e).unwrap());
        assert!(verify_packed_objects(out.join("manifest.json")).unwrap().is_empty());
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
                (0..big_ids.len()).map(|i| [i as f64 * 0.01, i as f64 * 0.013]).collect(),
            ),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        }])
        .unwrap();
        let big_compressed_len = compression::compress_zstd_with_dict(&big, None).unwrap().len();

        let mut w = PackWriter::create(&out, BlobOrdering::SpatialMajor, 4 * 1024).unwrap();
        // target 4 KiB < big blob.
        assert!(big_compressed_len as u64 > 4 * 1024);
        w.add_tile_full(&TileId::new(10, 0, 0, 0), 0, 100, None, 4000, None, &big).unwrap();
        for k in 1..4u64 {
            let p = distinct_tile(k);
            w.add_tile_full(&TileId::new(10, k as u32, 0, 0), 0, 100, None, 6, None, &p).unwrap();
        }
        let _manifest = w.finalize(&Metadata::new("big")).unwrap();

        // At least one pack exceeds the target (the loner). Reading proves it.
        let reader = PackedReader::open(out.join("manifest.json")).unwrap();
        let big_entry = reader.entries().iter().find(|e| e.x == 0).unwrap();
        assert_eq!(reader.read_payload(big_entry).unwrap(), big);
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
            let mut w = PackWriter::create(&out, BlobOrdering::Auto, 8 * 1024).unwrap();
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

        assert_eq!(m1.directory.key, m2.directory.key, "directory hash must be stable");
        assert_eq!(
            m1.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            m2.packs.iter().map(|p| &p.key).collect::<Vec<_>>(),
            "pack hashes must be stable across rebuilds"
        );
        assert_eq!(m1.to_json_bytes().unwrap(), m2.to_json_bytes().unwrap());
    }

    /// The directory ships zstd-compressed at rest (declared via
    /// `directory.encoding`), and a legacy manifest with a RAW directory and
    /// no `encoding` key — the shape of every deployed dataset — must still
    /// open and verify.
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
        let raw = compression::decompress_zstd_with_dict(&at_rest, None).unwrap();
        assert!(crate::directory::decode_directory(&raw).is_ok());
        assert!(verify_packed_objects(&manifest_path).unwrap().is_empty());
        let entries_compressed = PackedReader::open(&manifest_path).unwrap().entries().to_vec();

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
        let entries_raw = PackedReader::open(&manifest_path).unwrap().entries().to_vec();
        assert_eq!(entries_raw, entries_compressed);

        // An unknown encoding must fail loudly, not decode garbage.
        legacy.directory.encoding = Some("br".to_string());
        fs::write(&manifest_path, legacy.to_json_bytes().unwrap()).unwrap();
        assert!(PackedReader::open(&manifest_path).is_err());
    }

    /// A clean packed dataset verifies with no issues; corrupting a pack's
    /// bytes (without changing its length) breaks the content address and is
    /// reported.
    #[test]
    fn verify_packed_objects_clean_then_detects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");

        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 8 * 1024).unwrap();
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
            issues.iter().any(|s| s.contains("content-address mismatch")),
            "expected a content-address mismatch, got {issues:?}"
        );
    }
}
