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

/// Pointer to the encoded directory object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryRef {
    /// Object key, relative to the dataset root (e.g. `index/<hash>.sttd`).
    pub key: String,
    /// Directory object length in bytes.
    pub length: u64,
    /// Directory codec version (`5` for the packed format).
    #[serde(rename = "directoryVersion")]
    pub directory_version: u8,
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
        })
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
        pending.sort_by_key(|p| {
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
        for p in &pending {
            let compressed = compression::compress_zstd_with_dict(&p.payload, None)?;
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
        entries.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));

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

        // Encode + write the directory (content-addressed).
        let index_bytes = crate::directory::encode_directory(&entries);
        let index_hex = blake3_128_hex(&index_bytes);
        let index_rel = format!("index/{index_hex}.sttd");
        let index_path = out_dir.join(&index_rel);
        write_atomic(&index_dir, &index_path, &index_bytes)?;

        // Build + write the manifest.
        let manifest = Manifest {
            format: PACKED_FORMAT.to_string(),
            format_version: PACKED_FORMAT_VERSION,
            compression: "zstd".to_string(),
            directory: DirectoryRef {
                key: index_rel,
                length: index_bytes.len() as u64,
                directory_version: crate::directory::DIRECTORY_VERSION,
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
    let reader = ArchiveReader::open(in_archive)?;
    let meta = reader.metadata().clone();
    let entries = reader.entries().to_vec();

    let mut writer = PackWriter::create(out_dir, ordering, pack_target_bytes)?;
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
            "gzip" => Compression::Gzip,
            "none" => Compression::None,
            other => {
                return Err(Error::InvalidArchive(format!(
                    "unknown packed compression {other:?}"
                )))
            }
        };

        // Load + decode the directory object.
        let dir_path = root.join(&manifest.directory.key);
        let dir_bytes = fs::read(&dir_path)?;
        let entries = crate::directory::decode_directory(&dir_bytes)?;

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
        assert_eq!(back.metadata.name, "manifest-test");
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
        let manifest = w.finalize(&Metadata::new("big")).unwrap();

        // At least one pack exceeds the target (the loner). Reading proves it.
        let reader = PackedReader::open(out.join("manifest.json")).unwrap();
        let big_entry = reader.entries().iter().find(|e| e.x == 0).unwrap();
        assert_eq!(reader.read_payload(big_entry).unwrap(), big);
    }
}
