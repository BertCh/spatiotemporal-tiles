//! STT archive format — a single-file, range-request-friendly container.
//!
//! ## Layout
//!
//! ```text
//! [ 64-byte header ]
//! [ tile blobs ... ]            compressed tile payloads (zstd, optionally
//!                               against a shared dictionary; or gzip / none)
//! [ dictionary    ]            optional shared zstd training dictionary
//! [ directory     ]            the compact run-length tile index
//! [ metadata      ]            UTF-8 JSON (also renders TileJSON 3.0 + STAC)
//! ```
//!
//! The header records the byte ranges of the dictionary, directory and
//! metadata so a reader can fetch them with at most three HTTP range requests,
//! then each tile blob with one more.
//!
//! ## Format
//!
//! There is a single archive version, **v4** (`STT\x04`): blobs are integrity-
//! tagged with CRC32C, the tile index is the compact columnar run-length
//! [`crate::directory`], and a shared zstd dictionary slot is available. The
//! legacy v2 (gzip/blake3) and v3 (Arrow-IPC index) formats have been removed —
//! nothing is deployed, so archives regenerate from source.

use crate::compression;
use crate::error::{Error, Result};
use crate::tile::TileId;
use crate::types::Compression;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic prefix for STT archives: `"STT"` + the format version byte.
const MAGIC: &[u8; 4] = b"STT\x04";

/// The one and only archive format version.
pub const FORMAT_VERSION: u8 = 4;

/// Fixed header size in bytes.
const HEADER_SIZE: u64 = 64;

/// Target size of the trained zstd dictionary. ~110 KB is zstd's typical sweet
/// spot for corpora of many small, structurally-similar payloads (tiles).
const DICT_MAX_SIZE: usize = 112 * 1024;
/// Cap the dictionary-training sample so builds stay bounded on huge archives.
const DICT_SAMPLE_MAX_TILES: usize = 8192;
/// Byte cap on the dictionary-training sample.
const DICT_SAMPLE_MAX_BYTES: usize = 32 * 1024 * 1024;

/// One directory entry: where a tile lives and what it covers.
#[derive(Debug, Clone, PartialEq)]
pub struct TileEntry {
    /// Zoom level.
    pub zoom: u8,
    /// Tile X coordinate.
    pub x: u32,
    /// Tile Y coordinate.
    pub y: u32,
    /// Inclusive temporal start (Unix ms).
    pub time_start: i64,
    /// Inclusive temporal end (Unix ms).
    pub time_end: i64,
    /// Byte offset of the compressed blob within the archive.
    pub offset: u64,
    /// Compressed blob length in bytes.
    pub length: u32,
    /// Uncompressed payload length in bytes.
    pub uncompressed_size: u32,
    /// Total feature count across the tile's layers.
    pub feature_count: u32,
    /// Hilbert index of `(zoom, x, y)` — directory sort key.
    pub hilbert: u64,
    /// CRC32C integrity tag of the compressed blob.
    pub crc32c: u32,
    /// Temporal bucket size in milliseconds this tile occupies.
    ///
    /// Base tiles record the archive's base bucket size; temporal-LOD tiles
    /// record their coarser bucket. `None` when the archive carries no LOD
    /// pyramid (readers fall back to `Metadata::temporal_bucket_ms`).
    pub temporal_bucket_ms: Option<u64>,
    /// **Tight lower covering bound** — the minimum feature *start* time actually
    /// present in the tile. `time_start` is the addressable *bucket boundary*
    /// (kept loose so `(z,x,y,bucket)` lookup works); `cover_t_min` is the real
    /// earliest data, which a client uses to prune a tile whose features all lie
    /// after a query window (`time_end` already gives the tight upper bound).
    /// `None` when not computed — repacked/transcoded archives and pre-covering
    /// builds — in which case clients fall back to `time_start`.
    pub cover_t_min: Option<i64>,
}

impl TileEntry {
    /// The tile's identity.
    pub fn tile_id(&self) -> TileId {
        TileId::new(self.zoom, self.x, self.y, self.time_start.max(0) as u64)
    }
}

/// Parsed archive header.
#[derive(Debug, Clone)]
pub struct ArchiveHeader {
    /// Format version (always [`FORMAT_VERSION`]).
    pub version: u8,
    /// Compression applied to every tile blob.
    pub compression: Compression,
    /// Byte offset of the directory.
    pub index_offset: u64,
    /// Directory length in bytes.
    pub index_length: u64,
    /// Byte offset of the metadata JSON.
    pub metadata_offset: u64,
    /// Metadata JSON length in bytes.
    pub metadata_length: u64,
    /// Byte offset of the optional shared zstd dictionary. Zero if absent.
    pub dictionary_offset: u64,
    /// Length of the optional shared zstd dictionary in bytes. Zero if absent.
    pub dictionary_length: u64,
}

fn compression_to_byte(c: Compression) -> u8 {
    match c {
        Compression::None => 0,
        Compression::Gzip => 1,
        Compression::Zstd => 2,
    }
}

fn compression_from_byte(b: u8) -> Result<Compression> {
    match b {
        0 => Ok(Compression::None),
        1 => Ok(Compression::Gzip),
        2 => Ok(Compression::Zstd),
        other => Err(Error::InvalidArchive(format!(
            "unknown compression code {other}"
        ))),
    }
}

impl ArchiveHeader {
    /// Read and validate a v4 header.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::InvalidMagic);
        }
        let version = reader.read_u8()?;
        // The 4th magic byte doubles as a version field; both must be the one
        // supported version.
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let compression = compression_from_byte(reader.read_u8()?)?;
        let index_offset = reader.read_u64::<LittleEndian>()?;
        let index_length = reader.read_u64::<LittleEndian>()?;
        let metadata_offset = reader.read_u64::<LittleEndian>()?;
        let metadata_length = reader.read_u64::<LittleEndian>()?;
        let dictionary_offset = reader.read_u64::<LittleEndian>()?;
        let dictionary_length = reader.read_u64::<LittleEndian>()?;
        // Drain the remaining reserved bytes (HEADER_SIZE - 54).
        let mut reserved = [0u8; (HEADER_SIZE as usize) - 54];
        reader.read_exact(&mut reserved)?;
        Ok(Self {
            version,
            compression,
            index_offset,
            index_length,
            metadata_offset,
            metadata_length,
            dictionary_offset,
            dictionary_length,
        })
    }

    /// Write the header.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(MAGIC)?;
        writer.write_u8(FORMAT_VERSION)?;
        writer.write_u8(compression_to_byte(self.compression))?;
        writer.write_u64::<LittleEndian>(self.index_offset)?;
        writer.write_u64::<LittleEndian>(self.index_length)?;
        writer.write_u64::<LittleEndian>(self.metadata_offset)?;
        writer.write_u64::<LittleEndian>(self.metadata_length)?;
        writer.write_u64::<LittleEndian>(self.dictionary_offset)?;
        writer.write_u64::<LittleEndian>(self.dictionary_length)?;
        writer.write_all(&[0u8; (HEADER_SIZE as usize) - 54])?;
        Ok(())
    }
}

/// CRC32C integrity tag for a compressed blob.
fn crc32c_tag(bytes: &[u8]) -> u32 {
    crc32c::crc32c(bytes)
}

// ----------------------------------------------------------------------------
// In-memory temporal lookup
// ----------------------------------------------------------------------------

/// Boundary-bucketed temporal index built in memory when an archive is opened.
///
/// `boundaries` holds every distinct `time_start`/`time_end`, ascending. For
/// the largest boundary `<= t`, `buckets` lists the indices of every tile
/// whose `[time_start, time_end]` interval contains that boundary — so
/// "tiles overlapping time T" is an O(log n + k) lookup.
#[derive(Debug, Default)]
struct TemporalLookup {
    boundaries: Vec<i64>,
    bucket_offsets: Vec<u32>,
    bucket_refs: Vec<u32>,
}

impl TemporalLookup {
    /// Build the temporal lookup in `O(N log N + total bucket size)` time via a
    /// sweep over half-open interval events plus an incremental active set.
    ///
    /// Snapshot semantics:
    /// - `boundaries[i]` is a half-open boundary after which the active set is
    ///   stable until `boundaries[i+1]` (or +∞ for the last entry).
    /// - A query for time `t` uses `partition_point(|&b| b <= t)` to find the
    ///   bucket index `b`. `b == 0` means the query is before any tile
    ///   started → empty.
    fn build(entries: &[TileEntry]) -> Self {
        if entries.is_empty() {
            return Self::default();
        }

        // Half-open events: START at `time_start`, END at `time_end + 1`
        // (saturating so an interval ending at `i64::MAX` still produces a
        // well-defined event time). STARTs sort BEFORE ENDs at ties, so an
        // instantaneous interval `[t, t]` is active exactly at time `t`.
        const START: u8 = 0;
        const END: u8 = 1;
        let mut events: Vec<(i64, u8, u32)> = Vec::with_capacity(entries.len() * 2);
        for (i, e) in entries.iter().enumerate() {
            events.push((e.time_start, START, i as u32));
            events.push((e.time_end.saturating_add(1), END, i as u32));
        }
        events.sort_unstable_by_key(|&(t, kind, _)| (t, kind));

        let mut boundaries: Vec<i64> = Vec::new();
        let mut bucket_offsets: Vec<u32> = Vec::new();
        let mut bucket_refs: Vec<u32> = Vec::new();
        let mut active: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

        let mut i = 0;
        let mut prev_bucket_start: usize = 0;
        while i < events.len() {
            let t = events[i].0;
            while i < events.len() && events[i].0 == t {
                let (_, kind, idx) = events[i];
                if kind == START {
                    active.insert(idx);
                } else {
                    active.remove(&idx);
                }
                i += 1;
            }

            // Skip when the active set matches the previous boundary exactly
            // (STARTs and ENDs cancelled out at this time).
            let prev_bucket_end = bucket_refs.len();
            let prev_active_len = prev_bucket_end - prev_bucket_start;
            if active.len() == prev_active_len {
                let mut matches = true;
                let prev_slice = &bucket_refs[prev_bucket_start..prev_bucket_end];
                for (a, b) in active.iter().zip(prev_slice.iter()) {
                    if a != b {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    continue;
                }
            }
            boundaries.push(t);
            bucket_offsets.push(prev_bucket_end as u32);
            prev_bucket_start = prev_bucket_end;
            bucket_refs.extend(active.iter().copied());
        }
        // Sentinel offset so queries can slice `[bucket_offsets[b]..bucket_offsets[b+1]]`.
        bucket_offsets.push(bucket_refs.len() as u32);

        Self {
            boundaries,
            bucket_offsets,
            bucket_refs,
        }
    }

    /// Indices of every tile whose interval contains `t`.
    fn at(&self, t: i64) -> &[u32] {
        if self.boundaries.is_empty() {
            return &[];
        }
        let bucket = self.boundaries.partition_point(|&b| b <= t);
        if bucket == 0 {
            return &[];
        }
        let bucket = bucket - 1;
        let start = self.bucket_offsets[bucket] as usize;
        let end = self.bucket_offsets[bucket + 1] as usize;
        &self.bucket_refs[start..end]
    }
}

// ----------------------------------------------------------------------------
// Reader
// ----------------------------------------------------------------------------

/// Reader for an STT archive.
///
/// The archive file is memory-mapped on open; tile payload reads serve slices
/// directly out of the mapping rather than allocating + `pread`-ing a fresh
/// `Vec<u8>` per call. For warm caches this is allocation-free.
pub struct ArchiveReader {
    /// Owned file handle keeps the mmap alive even after `open` returns.
    #[allow(dead_code)]
    file: File,
    /// Memory map covering the entire archive.
    mmap: Mmap,
    header: ArchiveHeader,
    entries: Vec<TileEntry>,
    metadata: crate::metadata::Metadata,
    temporal: TemporalLookup,
    /// Shared zstd dictionary loaded from the dictionary slot, if present.
    dictionary: Option<Vec<u8>>,
}

impl ArchiveReader {
    /// Open an archive for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: we treat the mmap as read-only. The file is kept open for the
        // lifetime of the reader, and we never write through the map.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| Error::Other(format!("mmap failed: {e}")))?;

        let mut header_cursor = std::io::Cursor::new(&mmap[..HEADER_SIZE as usize]);
        let header = ArchiveHeader::read(&mut header_cursor)?;

        let index_end = (header.index_offset + header.index_length) as usize;
        if index_end > mmap.len() {
            return Err(Error::InvalidArchive(format!(
                "directory range {}..{} exceeds archive size {}",
                header.index_offset,
                index_end,
                mmap.len()
            )));
        }
        let entries =
            crate::directory::decode_directory(&mmap[header.index_offset as usize..index_end])?;

        let metadata_end = (header.metadata_offset + header.metadata_length) as usize;
        if metadata_end > mmap.len() {
            return Err(Error::InvalidArchive(format!(
                "metadata range {}..{} exceeds archive size {}",
                header.metadata_offset,
                metadata_end,
                mmap.len()
            )));
        }
        let metadata = crate::metadata::Metadata::from_json_bytes(
            &mmap[header.metadata_offset as usize..metadata_end],
        )?;

        let dictionary = if header.dictionary_length > 0 {
            let dict_end = (header.dictionary_offset + header.dictionary_length) as usize;
            if dict_end > mmap.len() {
                return Err(Error::InvalidArchive(format!(
                    "dictionary range {}..{} exceeds archive size {}",
                    header.dictionary_offset,
                    dict_end,
                    mmap.len()
                )));
            }
            Some(mmap[header.dictionary_offset as usize..dict_end].to_vec())
        } else {
            None
        };

        let temporal = TemporalLookup::build(&entries);

        Ok(Self {
            file,
            mmap,
            header,
            entries,
            metadata,
            temporal,
            dictionary,
        })
    }

    /// Archive metadata.
    pub fn metadata(&self) -> &crate::metadata::Metadata {
        &self.metadata
    }

    /// Parsed header.
    pub fn header(&self) -> &ArchiveHeader {
        &self.header
    }

    /// Shared zstd dictionary, if the archive ships one.
    pub fn dictionary(&self) -> Option<&[u8]> {
        self.dictionary.as_deref()
    }

    /// All directory entries (sorted by zoom then Hilbert index).
    pub fn entries(&self) -> &[TileEntry] {
        &self.entries
    }

    /// Directory entries of every tile whose interval contains time `t`.
    pub fn tiles_at_time(&self, t: i64) -> Vec<&TileEntry> {
        self.temporal
            .at(t)
            .iter()
            .filter_map(|&idx| self.entries.get(idx as usize))
            // The bucket is keyed on the largest boundary <= t and is a
            // superset; confirm the tile's interval actually contains t.
            .filter(|e| e.time_start <= t && t <= e.time_end)
            .collect()
    }

    /// Look up one tile by exact `(zoom, x, y, time)`.
    pub fn find_tile(&self, zoom: u8, x: u32, y: u32, t: i64) -> Option<&TileEntry> {
        self.tiles_at_time(t)
            .into_iter()
            .find(|e| e.zoom == zoom && e.x == x && e.y == y)
    }

    /// Read and decompress a tile's raw payload bytes, verifying its CRC32C.
    pub fn read_payload(&self, entry: &TileEntry) -> Result<Vec<u8>> {
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        if end > self.mmap.len() {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} blob range {start}..{end} exceeds archive size {}",
                entry.tile_id(),
                self.mmap.len()
            )));
        }
        let compressed = &self.mmap[start..end];

        if crc32c_tag(compressed) != entry.crc32c {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} failed integrity check (corrupt archive)",
                entry.tile_id()
            )));
        }

        let payload = if self.header.compression == Compression::Zstd {
            compression::decompress_zstd_with_dict(compressed, self.dictionary.as_deref())?
        } else {
            compression::decompress(compressed, self.header.compression)?
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

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

/// A tile buffered for deferred, dictionary-trained compression.
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

/// Writer for creating an STT archive.
///
/// Two modes:
/// - [`create`](Self::create) — *eager*: each tile is compressed and written as
///   it arrives, so peak memory is bounded (good for streaming huge builds).
/// - [`create_optimized`](Self::create_optimized) — *buffered*: tiles are held
///   until [`finalize`](Self::finalize), where a shared zstd dictionary is
///   trained, byte-identical blobs are deduplicated, and blobs are written in
///   Hilbert order. Smaller archives at the cost of holding payloads in memory.
pub struct ArchiveWriter {
    file: File,
    compression: Compression,
    current_offset: u64,
    entries: Vec<TileEntry>,
    /// `Some` in the buffered/optimized mode (tiles parked until finalize).
    pending: Option<Vec<PendingTile>>,
    /// Blob byte-order applied at `finalize_buffered` (ignored in eager mode,
    /// which never reorders blobs).
    ordering: crate::curve::BlobOrdering,
    /// Train + ship a shared zstd dictionary (buffered mode). OFF keeps the
    /// archive decodable by readers without a dictionary-capable zstd (the
    /// current TS reader throws on a dictionary), so blob *reordering* — which
    /// is reader-safe on its own — can ship independently of the dict.
    train_dict: bool,
}

impl ArchiveWriter {
    /// Create an archive in eager (streaming) mode. Tile blobs use `compression`.
    pub fn create<P: AsRef<Path>>(path: P, compression: Compression) -> Result<Self> {
        Self::open_file(path, compression, None, crate::curve::BlobOrdering::default(), false)
    }

    /// Create an archive in buffered "optimized" mode: a shared zstd dictionary
    /// is trained over the tile corpus, byte-identical blobs are deduplicated,
    /// and blobs are laid down in 2D-spatial-Hilbert order for range-read
    /// locality. Requires zstd. See [`create_optimized_with_ordering`] to pick a
    /// different (e.g. space-time) blob order.
    ///
    /// [`create_optimized_with_ordering`]: Self::create_optimized_with_ordering
    pub fn create_optimized<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::create_optimized_with_ordering(path, crate::curve::BlobOrdering::SpatialMajor)
    }

    /// Buffered "optimized" mode with an explicit blob byte-[`ordering`]. The
    /// directory index stays `(zoom, hilbert, time_start)` regardless; only the
    /// physical blob layout changes, so readers are unaffected. The default
    /// ([`BlobOrdering::Hilbert3`]) minimises range requests across the widest
    /// range of datasets (see `examples/simulate_layout.rs`).
    ///
    /// [`ordering`]: crate::curve::BlobOrdering
    /// [`BlobOrdering::Hilbert3`]: crate::curve::BlobOrdering::Hilbert3
    pub fn create_optimized_with_ordering<P: AsRef<Path>>(
        path: P,
        ordering: crate::curve::BlobOrdering,
    ) -> Result<Self> {
        Self::open_file(path, Compression::Zstd, Some(Vec::new()), ordering, true)
    }

    /// Buffered mode that **reorders blobs but ships NO shared dictionary**, so
    /// the result stays decodable by readers without a dictionary-capable zstd
    /// (today's TS reader). Per-blob zstd + byte-identical dedup still apply.
    /// This is the reader-safe way to get the range-request win from blob
    /// [`ordering`](crate::curve::BlobOrdering) without the dict that the TS
    /// reader can't yet decode.
    pub fn create_reordered<P: AsRef<Path>>(
        path: P,
        ordering: crate::curve::BlobOrdering,
    ) -> Result<Self> {
        Self::open_file(path, Compression::Zstd, Some(Vec::new()), ordering, false)
    }

    fn open_file<P: AsRef<Path>>(
        path: P,
        compression: Compression,
        pending: Option<Vec<PendingTile>>,
        ordering: crate::curve::BlobOrdering,
        train_dict: bool,
    ) -> Result<Self> {
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.seek(SeekFrom::Start(HEADER_SIZE))?;
        Ok(Self {
            file,
            compression,
            current_offset: HEADER_SIZE,
            entries: Vec::new(),
            pending,
            ordering,
            train_dict,
        })
    }

    /// Add a tile. `payload` is the uncompressed tile payload (the layer frame
    /// produced by [`crate::arrow_tile::encode_tile`]).
    pub fn add_tile(
        &mut self,
        id: &TileId,
        time_start: i64,
        time_end: i64,
        feature_count: u32,
        payload: &[u8],
    ) -> Result<()> {
        self.add_tile_with_bucket(id, time_start, time_end, feature_count, None, payload)
    }

    /// Add a tile, tagging the directory entry with the temporal bucket size it
    /// represents (used when emitting temporal-LOD aggregate tiles).
    pub fn add_tile_with_bucket(
        &mut self,
        id: &TileId,
        time_start: i64,
        time_end: i64,
        feature_count: u32,
        temporal_bucket_ms: Option<u64>,
        payload: &[u8],
    ) -> Result<()> {
        self.add_tile_full(id, time_start, time_end, None, feature_count, temporal_bucket_ms, payload)
    }

    /// Add a tile carrying the full directory metadata, including the tight
    /// lower covering bound `cover_t_min` (the earliest feature *start* time
    /// actually present — see [`TileEntry::cover_t_min`]). `None` leaves the
    /// entry without a covering bound (clients fall back to `time_start`).
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
        if let Some(pending) = self.pending.as_mut() {
            pending.push(PendingTile {
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
            return Ok(());
        }

        let uncompressed_size = payload.len() as u32;
        let compressed = compression::compress(payload, self.compression)?;
        let crc = crc32c_tag(&compressed);
        let offset = self.current_offset;
        let length = compressed.len() as u32;
        self.file.write_all(&compressed)?;
        self.current_offset += length as u64;

        self.entries.push(TileEntry {
            zoom: id.z,
            x: id.x,
            y: id.y,
            time_start,
            time_end,
            offset,
            length,
            uncompressed_size,
            feature_count,
            hilbert: id.hilbert_index(),
            crc32c: crc,
            temporal_bucket_ms,
            cover_t_min,
        });
        Ok(())
    }

    /// Number of tiles staged so far (buffered or written).
    pub fn tile_count(&self) -> usize {
        match &self.pending {
            Some(p) => p.len(),
            None => self.entries.len(),
        }
    }

    /// Finalise: write the dictionary (if any), directory, metadata, and header.
    pub fn finalize(self, metadata: &crate::metadata::Metadata) -> Result<()> {
        if self.pending.is_some() {
            self.finalize_buffered(metadata)
        } else {
            self.finalize_eager(metadata)
        }
    }

    /// Eager finalize: blobs are already written; lay down the directory,
    /// metadata and header (no shared dictionary).
    fn finalize_eager(mut self, metadata: &crate::metadata::Metadata) -> Result<()> {
        self.entries.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
        self.write_tail(metadata, 0, 0)
    }

    /// Buffered finalize: train a shared zstd dictionary over a sample of the
    /// buffered payloads, compress every tile against it, deduplicate byte-
    /// identical blobs, write blobs in Hilbert order, then the dictionary,
    /// directory, metadata and header.
    ///
    /// Degrades gracefully to plain zstd (empty dictionary slot) when the
    /// corpus is too small/sparse to train a useful dictionary.
    fn finalize_buffered(mut self, metadata: &crate::metadata::Metadata) -> Result<()> {
        let mut pending = self.pending.take().unwrap_or_default();

        // Order the blobs (not just the index) so a viewport maps to a
        // near-contiguous byte range — what lets the client's range coalescer
        // merge neighbouring tiles into one HTTP request. The byte order is
        // decoupled from the directory index (which stays (zoom,hilbert,t)); the
        // best order depends on the dataset's space-vs-time shape, so it's a
        // knob (default 3D Hilbert). See crate::curve.
        let base_bucket = metadata.temporal_bucket_ms.max(1) as i64;
        let tb = |p: &PendingTile| {
            let b = p.temporal_bucket_ms.map(|v| v as i64).unwrap_or(base_bucket).max(1);
            p.time_start.div_euclid(b)
        };
        let (tb_min, tb_max) = pending.iter().fold((i64::MAX, i64::MIN), |(lo, hi), p| {
            let t = tb(p);
            (lo.min(t), hi.max(t))
        });
        let tb_span = if pending.is_empty() { 0 } else { tb_max - tb_min };
        // Resolve an `Auto` ordering to a concrete one from the dataset's
        // space-vs-time cardinality (wide-time → spatial-major for playback
        // locality; otherwise the 3D-Hilbert generalist).
        let ordering = match self.ordering {
            crate::curve::BlobOrdering::Auto => {
                let max_z = pending.iter().map(|p| p.z).max().unwrap_or(0) as u32;
                let time_bits = crate::curve::bits_for((tb_span.max(0) + 1) as u64);
                crate::curve::BlobOrdering::choose(max_z, time_bits)
            }
            other => other,
        };
        pending.sort_by_key(|p| {
            crate::curve::space_time_key(
                ordering, p.z, p.x, p.y, p.hilbert, p.time_start, tb(p), tb_min, tb_span,
            )
        });

        // Train a shared dictionary from a bounded sample of payloads (only when
        // requested). Bytes shared across tiles (Arrow schema, dictionary key
        // names, common coordinate prefixes) get hoisted into the dictionary
        // once instead of repeating in every compressed blob. Skipped by the
        // reorder-only path so the archive stays decodable by readers without a
        // dictionary-capable zstd; `compress_zstd_with_dict(_, None)` then emits
        // plain zstd and the dictionary section is omitted.
        let dictionary = if self.train_dict {
            let mut sample: Vec<&[u8]> = Vec::new();
            let mut sample_bytes = 0usize;
            for p in &pending {
                if sample.len() >= DICT_SAMPLE_MAX_TILES || sample_bytes >= DICT_SAMPLE_MAX_BYTES {
                    break;
                }
                sample_bytes += p.payload.len();
                sample.push(p.payload.as_slice());
            }
            compression::train_zstd_dictionary_from_slices(&sample, DICT_MAX_SIZE)
        } else {
            None
        };

        // Stream the dictionary-compressed blobs, deduplicating byte-identical
        // ones (a static cell repeated across time buckets writes its blob
        // once); the directory's run-length encoding then collapses the shared
        // references into a single run.
        self.file.seek(SeekFrom::Start(HEADER_SIZE))?;
        self.current_offset = HEADER_SIZE;
        let mut blob_dedup: std::collections::HashMap<[u8; 32], (u64, u32, u32, u32)> =
            std::collections::HashMap::new();
        for p in &pending {
            let compressed =
                compression::compress_zstd_with_dict(&p.payload, dictionary.as_deref())?;
            let uncompressed_size = p.payload.len() as u32;
            // Strong hash of the *compressed* bytes — deterministic zstd makes
            // identical payloads identical blobs, so this dedup is exact.
            let key = *blake3::hash(&compressed).as_bytes();
            let (offset, length, crc) = if let Some(&(off, len, _unc, crc)) = blob_dedup.get(&key) {
                (off, len, crc)
            } else {
                let offset = self.current_offset;
                let length = compressed.len() as u32;
                let crc = crc32c_tag(&compressed);
                self.file.write_all(&compressed)?;
                self.current_offset += length as u64;
                blob_dedup.insert(key, (offset, length, uncompressed_size, crc));
                (offset, length, crc)
            };
            self.entries.push(TileEntry {
                zoom: p.z,
                x: p.x,
                y: p.y,
                time_start: p.time_start,
                time_end: p.time_end,
                offset,
                length,
                uncompressed_size,
                feature_count: p.feature_count,
                hilbert: p.hilbert,
                crc32c: crc,
                temporal_bucket_ms: p.temporal_bucket_ms,
                cover_t_min: p.cover_t_min,
            });
        }

        // Dictionary section sits between the tile data and the directory so a
        // reader can coalesce it with the directory range request.
        let (dictionary_offset, dictionary_length) = if let Some(dict) = &dictionary {
            let off = self.current_offset;
            self.file.write_all(dict)?;
            self.current_offset += dict.len() as u64;
            (off, dict.len() as u64)
        } else {
            (0, 0)
        };

        self.entries.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
        self.write_tail(metadata, dictionary_offset, dictionary_length)
    }

    /// Shared tail writer: directory + metadata + header. Assumes blobs (and the
    /// dictionary, if any) are already written and `self.entries` is sorted.
    fn write_tail(
        mut self,
        metadata: &crate::metadata::Metadata,
        dictionary_offset: u64,
        dictionary_length: u64,
    ) -> Result<()> {
        let index_bytes = crate::directory::encode_directory(&self.entries);
        let index_offset = self.current_offset;
        let index_length = index_bytes.len() as u64;
        self.file.write_all(&index_bytes)?;
        self.current_offset += index_length;

        let metadata_bytes = metadata.to_json_bytes()?;
        let metadata_offset = self.current_offset;
        let metadata_length = metadata_bytes.len() as u64;
        self.file.write_all(&metadata_bytes)?;
        self.current_offset += metadata_length;

        self.file.flush()?;
        self.file.set_len(self.current_offset)?;

        let header = ArchiveHeader {
            version: FORMAT_VERSION,
            compression: self.compression,
            index_offset,
            index_length,
            metadata_offset,
            metadata_length,
            dictionary_offset,
            dictionary_length,
        };
        self.file.seek(SeekFrom::Start(0))?;
        header.write(&mut self.file)?;
        self.file.flush()?;
        Ok(())
    }
}

/// High-level archive entry points.
pub struct Archive;

impl Archive {
    /// Open an existing archive.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<ArchiveReader> {
        ArchiveReader::open(path)
    }

    /// Create a new archive (eager mode).
    pub fn create<P: AsRef<Path>>(path: P, compression: Compression) -> Result<ArchiveWriter> {
        ArchiveWriter::create(path, compression)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_tile::{ColumnarLayer, GeometryColumn};
    use tempfile::NamedTempFile;

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

    #[test]
    fn header_roundtrips() {
        let header = ArchiveHeader {
            version: FORMAT_VERSION,
            compression: Compression::Zstd,
            index_offset: 1234,
            index_length: 56,
            metadata_offset: 1290,
            metadata_length: 78,
            dictionary_offset: 1100,
            dictionary_length: 32,
        };
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE as usize);
        assert_eq!(&buf[..4], MAGIC);
        let read = ArchiveHeader::read(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(read.version, FORMAT_VERSION);
        assert_eq!(read.index_offset, 1234);
        assert_eq!(read.dictionary_offset, 1100);
        assert_eq!(read.dictionary_length, 32);
        assert_eq!(read.compression, Compression::Zstd);
    }

    #[test]
    fn rejects_foreign_magic() {
        let mut buf = vec![b'X', b'Y', b'Z', 9];
        buf.resize(HEADER_SIZE as usize, 0);
        assert!(ArchiveHeader::read(&mut std::io::Cursor::new(buf)).is_err());
    }

    #[test]
    fn eager_archive_roundtrips_tiles_and_temporal_lookup() {
        let path = NamedTempFile::new().unwrap().into_temp_path();

        let mut writer = ArchiveWriter::create(&path, Compression::Zstd).unwrap();
        let tile_a = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 1000)]).unwrap();
        let tile_b = crate::arrow_tile::encode_tile(&[point_layer("default", vec![2, 3], 1500)]).unwrap();
        let tile_c = crate::arrow_tile::encode_tile(&[point_layer("default", vec![4], 5000)]).unwrap();
        writer.add_tile(&TileId::new(10, 1, 1, 1000), 1000, 2000, 1, &tile_a).unwrap();
        writer.add_tile(&TileId::new(10, 2, 2, 1500), 1500, 3000, 2, &tile_b).unwrap();
        writer.add_tile(&TileId::new(11, 4, 4, 5000), 5000, 6000, 1, &tile_c).unwrap();
        writer.finalize(&crate::metadata::Metadata::new("test-archive")).unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        assert_eq!(reader.header().version, FORMAT_VERSION);
        assert_eq!(reader.entries().len(), 3);
        assert_eq!(reader.metadata().name, "test-archive");

        assert_eq!(reader.tiles_at_time(1800).len(), 2);
        let at = reader.tiles_at_time(2500);
        assert_eq!(at.len(), 1);
        assert_eq!(at[0].x, 2);
        assert!(reader.tiles_at_time(4000).is_empty());

        let entry = reader.find_tile(10, 2, 2, 2000).unwrap().clone();
        let layers = reader.read_layers(&entry).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "default");
        assert_eq!(layers[0].batch.num_rows(), 2);
    }

    #[test]
    fn corrupt_blob_is_detected() {
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create(&path, Compression::None).unwrap();
        let payload = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 1000)]).unwrap();
        writer.add_tile(&TileId::new(5, 0, 0, 1000), 1000, 2000, 1, &payload).unwrap();
        writer.finalize(&crate::metadata::Metadata::new("corrupt")).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_SIZE as usize] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        let entry = reader.entries()[0].clone();
        assert!(reader.read_payload(&entry).is_err());
    }

    /// `TemporalLookup` build correctness + worst-case timing regression
    /// (50k entries must build in well under a second).
    #[test]
    fn temporal_lookup_build_is_fast_and_correct() {
        let bucket = 3_600_000i64; // 1 hour
        let mut entries = Vec::with_capacity(50_000);
        for cell in 0..200u32 {
            for b in 0..250u32 {
                let t_start = (b as i64) * bucket;
                entries.push(TileEntry {
                    zoom: 10,
                    x: cell,
                    y: 0,
                    time_start: t_start,
                    time_end: t_start + bucket - 1,
                    offset: 0,
                    length: 0,
                    uncompressed_size: 0,
                    feature_count: 0,
                    hilbert: 0,
                    crc32c: 0,
                    temporal_bucket_ms: None,
                    cover_t_min: None,
                });
            }
        }

        let started = std::time::Instant::now();
        let lookup = TemporalLookup::build(&entries);
        let elapsed_ms = started.elapsed().as_millis();
        assert!(elapsed_ms < 1000, "TemporalLookup::build took {elapsed_ms}ms for 50k entries");

        let query_t = bucket * 123 + bucket / 2;
        let got: std::collections::BTreeSet<u32> = lookup.at(query_t).iter().copied().collect();
        let expected: std::collections::BTreeSet<u32> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.time_start <= query_t && query_t <= e.time_end)
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(got, expected);
        assert!(lookup.at(-1).is_empty());
        assert!(lookup.at(bucket * 1_000_000).is_empty());
    }

    /// Optimized (buffered) archive: byte-identical tiles across time dedup to a
    /// single blob and the run-length directory round-trips every entry.
    #[test]
    fn optimized_archive_dedups_and_roundtrips() {
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create_optimized(&path).unwrap();

        let static_payload =
            crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 0)]).unwrap();
        for b in 0..5i64 {
            let t = b * 3_600_000;
            writer
                .add_tile(&TileId::new(9, 3, 4, t as u64), t, t + 3_599_999, 1, &static_payload)
                .unwrap();
        }
        let other =
            crate::arrow_tile::encode_tile(&[point_layer("default", vec![2, 3], 1000)]).unwrap();
        writer.add_tile(&TileId::new(9, 5, 6, 0), 0, 1000, 2, &other).unwrap();
        writer.finalize(&crate::metadata::Metadata::new("opt")).unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        assert_eq!(reader.header().version, FORMAT_VERSION);
        assert_eq!(reader.entries().len(), 6);

        let static_offsets: std::collections::BTreeSet<u64> = reader
            .entries()
            .iter()
            .filter(|e| e.x == 3 && e.y == 4)
            .map(|e| e.offset)
            .collect();
        assert_eq!(static_offsets.len(), 1, "static-across-time tiles should dedup to one blob");

        let mid = reader.find_tile(9, 3, 4, 3 * 3_600_000).unwrap().clone();
        assert_eq!(reader.read_layers(&mid).unwrap()[0].batch.num_rows(), 1);
        let distinct = reader.entries().iter().find(|e| e.x == 5).unwrap().clone();
        assert_eq!(reader.read_layers(&distinct).unwrap()[0].batch.num_rows(), 2);
    }

    /// The reorder-only buffered path applies the chosen blob ordering but ships
    /// NO shared dictionary, so the archive stays decodable by readers without a
    /// dictionary-capable zstd (today's TS reader throws on a dictionary). This
    /// is the reader-safety contract behind `stt-build --blob-ordering`.
    #[test]
    fn reordered_archive_has_no_dict_and_roundtrips() {
        use crate::curve::{space_time_key, BlobOrdering};
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create_reordered(&path, BlobOrdering::Hilbert3).unwrap();

        // A small space×time grid with all-distinct payloads (no dedup) so blob
        // order is a 1:1 image of the curve order.
        let mut expected = 0usize;
        for x in 0..4u32 {
            for y in 0..4u32 {
                for b in 0..3i64 {
                    let t = b * 3_600_000;
                    let p = crate::arrow_tile::encode_tile(&[point_layer(
                        "default",
                        vec![(x * 16 + y) as u64 + b as u64 * 1000],
                        1000,
                    )])
                    .unwrap();
                    writer
                        .add_tile(&TileId::new(10, x, y, t as u64), t, t + 3_599_999, 1, &p)
                        .unwrap();
                    expected += 1;
                }
            }
        }
        writer
            .finalize(&crate::metadata::Metadata::new("reordered"))
            .unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        // Reader-safety contract: NO shared dictionary section.
        assert!(reader.dictionary().is_none(), "reorder path must not ship a dict");
        assert_eq!(reader.entries().len(), expected);
        // Every tile still decodes.
        for e in reader.entries().to_vec() {
            assert_eq!(reader.read_layers(&e).unwrap()[0].batch.num_rows(), 1);
        }
        // Blobs are physically laid down in Hilbert order: walking entries by
        // ascending byte offset yields a non-decreasing curve key.
        let mut ents = reader.entries().to_vec();
        ents.sort_by_key(|e| e.offset);
        let key = |e: &TileEntry| {
            space_time_key(
                BlobOrdering::Hilbert3,
                e.zoom,
                e.x,
                e.y,
                e.hilbert,
                e.time_start,
                e.time_start.div_euclid(3_600_000),
                0,
                2,
            )
        };
        for w in ents.windows(2) {
            assert!(key(&w[0]) <= key(&w[1]), "blobs must be in Hilbert order on disk");
        }
    }

    /// The shared dictionary shrinks a corpus of many small similar tiles and
    /// every tile still decodes against the embedded dictionary.
    #[test]
    fn optimized_dictionary_shrinks_and_roundtrips() {
        use crate::arrow_tile::{ColumnarLayer, GeometryColumn, PropertyColumn};

        let kinds = ["bike", "car", "bus", "scooter"];
        let make_payload = |seed: u64| -> Vec<u8> {
            let f = 8usize;
            let mut feature_ids = Vec::with_capacity(f);
            let mut start_times = Vec::with_capacity(f);
            let mut end_times = Vec::with_capacity(f);
            let mut paths: Vec<Vec<[f64; 2]>> = Vec::with_capacity(f);
            let mut vtimes: Vec<Vec<i64>> = Vec::with_capacity(f);
            let mut kind: Vec<Option<String>> = Vec::with_capacity(f);
            for i in 0..f {
                feature_ids.push(seed * 100 + i as u64);
                let t0 = i as i64 * 1000;
                start_times.push(t0);
                end_times.push(t0 + 1000);
                paths.push(
                    (0..16)
                        .map(|j| {
                            [
                                -122.4 + i as f64 * 0.001 + j as f64 * 0.0001,
                                37.7 + seed as f64 * 0.0001 + j as f64 * 0.0001,
                            ]
                        })
                        .collect(),
                );
                vtimes.push((0..16).map(|j| t0 + j * 60).collect());
                kind.push(Some(kinds[(seed as usize + i) % kinds.len()].to_string()));
            }
            crate::arrow_tile::encode_tile(&[ColumnarLayer {
                name: "trips".to_string(),
                feature_ids,
                start_times,
                end_times,
                geometry: GeometryColumn::LineString(paths),
                vertex_times: Some(vtimes),
                vertex_values: None,
                triangles: None,
                properties: vec![("kind".to_string(), PropertyColumn::Categorical(kind))],
            }])
            .unwrap()
        };

        let dir = tempfile::tempdir().unwrap();
        let opt_path = dir.path().join("opt.stt");
        let plain_path = dir.path().join("plain.stt");

        let mut w = ArchiveWriter::create_optimized(&opt_path).unwrap();
        for s in 0..512u64 {
            w.add_tile(&TileId::new(10, (s % 32) as u32, (s / 32) as u32, 0), 0, 8000, 8, &make_payload(s))
                .unwrap();
        }
        w.finalize(&crate::metadata::Metadata::new("opt")).unwrap();

        let mut w2 = ArchiveWriter::create(&plain_path, Compression::Zstd).unwrap();
        for s in 0..512u64 {
            w2.add_tile(&TileId::new(10, (s % 32) as u32, (s / 32) as u32, 0), 0, 8000, 8, &make_payload(s))
                .unwrap();
        }
        w2.finalize(&crate::metadata::Metadata::new("plain")).unwrap();

        let opt_size = std::fs::metadata(&opt_path).unwrap().len();
        let plain_size = std::fs::metadata(&plain_path).unwrap().len();
        eprintln!("optimized: {opt_size} bytes, eager-zstd: {plain_size} bytes");

        let reader = ArchiveReader::open(&opt_path).unwrap();
        assert!(reader.dictionary().is_some(), "expected a trained dictionary");
        let entry = reader.entries()[0].clone();
        assert_eq!(reader.read_layers(&entry).unwrap()[0].batch.num_rows(), 8);
        assert!(opt_size < plain_size, "dictionary should shrink: opt={opt_size}, plain={plain_size}");
    }
}
