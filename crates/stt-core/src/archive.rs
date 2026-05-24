//! STT archive format — a single-file, range-request-friendly container.
//!
//! ## Layout (v3)
//!
//! ```text
//! [ 64-byte header ]
//! [ tile blobs ... ]            zstd-compressed Arrow-IPC tile payloads
//! [ dictionary    ]            optional zstd training dictionary
//! [ index table   ]            Arrow IPC: one row per tile (the directory)
//! [ metadata      ]            UTF-8 JSON
//! ```
//!
//! The header records the byte ranges of the dictionary, index and metadata
//! so a reader can fetch them with at most three HTTP range requests, then
//! each tile with one more.
//!
//! ## Versioning
//!
//! - `STT\x02` archives are gzip-only, blake3-64 content addressed, and use
//!   a single 11-column directory.
//! - `STT\x03` archives default to zstd-3, use CRC32C for integrity, and
//!   carry the optional dictionary slot. The directory drops the dedup
//!   `content_hash` column in favour of a 32-bit `crc32c` integrity tag.
//!
//! Both versions share the 64-byte header layout — v2 simply ignores the
//! `dictionary_*` slots (which were reserved bytes in v2) on read.

use crate::compression;
use crate::error::{Error, Result};
use crate::tile::TileId;
use crate::types::Compression;
use arrow::array::{
    Array, ArrayRef, Int64Array, RecordBatch, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

/// Magic prefix for v2 archives.
const MAGIC_V2: &[u8; 4] = b"STT\x02";
/// Magic prefix for v3 archives.
const MAGIC_V3: &[u8; 4] = b"STT\x03";

/// Latest archive format version.
pub const FORMAT_VERSION: u8 = 3;

/// Fixed header size in bytes.
const HEADER_SIZE: u64 = 64;

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
    /// Integrity tag for the compressed blob. In v3 this is a CRC32C zero-
    /// extended to 64 bits; in v2 it is the leading 64 bits of a blake3
    /// digest (which is also the dedup key on the write side).
    pub content_hash: u64,
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
    /// Format version (2 or 3).
    pub version: u8,
    /// Compression applied to every tile blob.
    pub compression: Compression,
    /// Byte offset of the index table.
    pub index_offset: u64,
    /// Index table length in bytes.
    pub index_length: u64,
    /// Byte offset of the metadata JSON.
    pub metadata_offset: u64,
    /// Metadata JSON length in bytes.
    pub metadata_length: u64,
    /// Byte offset of the optional zstd training dictionary (v3 only).
    /// Zero if no dictionary is present.
    pub dictionary_offset: u64,
    /// Length of the optional zstd training dictionary in bytes (v3 only).
    /// Zero if no dictionary is present.
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
    /// Read and validate a header. Accepts both v2 and v3 magic numbers.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let version = if &magic == MAGIC_V3 {
            3
        } else if &magic == MAGIC_V2 {
            2
        } else {
            return Err(Error::InvalidMagic);
        };
        let version_byte = reader.read_u8()?;
        if version_byte != version {
            return Err(Error::InvalidArchive(format!(
                "header magic says v{version} but version byte is {version_byte}"
            )));
        }
        if version > FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version_byte));
        }
        let compression = compression_from_byte(reader.read_u8()?)?;
        let index_offset = reader.read_u64::<LittleEndian>()?;
        let index_length = reader.read_u64::<LittleEndian>()?;
        let metadata_offset = reader.read_u64::<LittleEndian>()?;
        let metadata_length = reader.read_u64::<LittleEndian>()?;
        // v3 stores dictionary_offset / dictionary_length immediately after
        // the metadata fields. v2 has reserved zeros there — reading them as
        // 0 is the correct fallback (no dictionary).
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

    /// Write the header. Picks v2 or v3 magic based on `version`.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        let magic: &[u8; 4] = match self.version {
            2 => MAGIC_V2,
            3 => MAGIC_V3,
            other => {
                return Err(Error::InvalidArchive(format!(
                    "cannot write unknown header version {other}"
                )))
            }
        };
        writer.write_all(magic)?;
        writer.write_u8(self.version)?;
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

// ----------------------------------------------------------------------------
// Index table (Arrow) encode / decode
// ----------------------------------------------------------------------------

/// Directory schema for v2 archives. Carries a 64-bit blake3 prefix that
/// doubles as dedup key + integrity tag.
fn index_schema_v2() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("zoom", DataType::UInt8, false),
        Field::new("x", DataType::UInt32, false),
        Field::new("y", DataType::UInt32, false),
        Field::new("time_start", DataType::Int64, false),
        Field::new("time_end", DataType::Int64, false),
        Field::new("offset", DataType::UInt64, false),
        Field::new("length", DataType::UInt32, false),
        Field::new("uncompressed_size", DataType::UInt32, false),
        Field::new("feature_count", DataType::UInt32, false),
        Field::new("hilbert", DataType::UInt64, false),
        Field::new("content_hash", DataType::UInt64, false),
    ]))
}

/// Directory schema for v3 archives. Replaces the 8-byte blake3 prefix with
/// a 4-byte CRC32C tag (no dedup, just integrity).
fn index_schema_v3() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("zoom", DataType::UInt8, false),
        Field::new("x", DataType::UInt32, false),
        Field::new("y", DataType::UInt32, false),
        Field::new("time_start", DataType::Int64, false),
        Field::new("time_end", DataType::Int64, false),
        Field::new("offset", DataType::UInt64, false),
        Field::new("length", DataType::UInt32, false),
        Field::new("uncompressed_size", DataType::UInt32, false),
        Field::new("feature_count", DataType::UInt32, false),
        Field::new("hilbert", DataType::UInt64, false),
        Field::new("crc32c", DataType::UInt32, false),
    ]))
}

fn encode_index(entries: &[TileEntry], version: u8) -> Result<Vec<u8>> {
    let schema = if version == 3 {
        index_schema_v3()
    } else {
        index_schema_v2()
    };

    let zoom = Arc::new(UInt8Array::from(entries.iter().map(|e| e.zoom).collect::<Vec<_>>())) as ArrayRef;
    let x = Arc::new(UInt32Array::from(entries.iter().map(|e| e.x).collect::<Vec<_>>())) as ArrayRef;
    let y = Arc::new(UInt32Array::from(entries.iter().map(|e| e.y).collect::<Vec<_>>())) as ArrayRef;
    let time_start = Arc::new(Int64Array::from(entries.iter().map(|e| e.time_start).collect::<Vec<_>>())) as ArrayRef;
    let time_end = Arc::new(Int64Array::from(entries.iter().map(|e| e.time_end).collect::<Vec<_>>())) as ArrayRef;
    let offset = Arc::new(UInt64Array::from(entries.iter().map(|e| e.offset).collect::<Vec<_>>())) as ArrayRef;
    let length = Arc::new(UInt32Array::from(entries.iter().map(|e| e.length).collect::<Vec<_>>())) as ArrayRef;
    let uncompressed = Arc::new(UInt32Array::from(entries.iter().map(|e| e.uncompressed_size).collect::<Vec<_>>())) as ArrayRef;
    let feature_count = Arc::new(UInt32Array::from(entries.iter().map(|e| e.feature_count).collect::<Vec<_>>())) as ArrayRef;
    let hilbert = Arc::new(UInt64Array::from(entries.iter().map(|e| e.hilbert).collect::<Vec<_>>())) as ArrayRef;

    let columns: Vec<ArrayRef> = if version == 3 {
        let crc = Arc::new(UInt32Array::from(
            entries.iter().map(|e| e.content_hash as u32).collect::<Vec<_>>(),
        )) as ArrayRef;
        vec![zoom, x, y, time_start, time_end, offset, length, uncompressed, feature_count, hilbert, crc]
    } else {
        let hash = Arc::new(UInt64Array::from(
            entries.iter().map(|e| e.content_hash).collect::<Vec<_>>(),
        )) as ArrayRef;
        vec![zoom, x, y, time_start, time_end, offset, length, uncompressed, feature_count, hilbert, hash]
    };

    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Error::Other(format!("failed to build index batch: {e}")))?;
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)
            .map_err(|e| Error::Other(format!("index IPC writer init failed: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| Error::Other(format!("index IPC write failed: {e}")))?;
        writer
            .finish()
            .map_err(|e| Error::Other(format!("index IPC finish failed: {e}")))?;
    }
    Ok(buf)
}

fn decode_index(bytes: &[u8], version: u8) -> Result<Vec<TileEntry>> {
    let reader = StreamReader::try_new(bytes, None)
        .map_err(|e| Error::InvalidArchive(format!("index IPC reader init failed: {e}")))?;
    let mut entries = Vec::new();
    for batch in reader {
        let batch =
            batch.map_err(|e| Error::InvalidArchive(format!("index IPC read failed: {e}")))?;
        let col = |name: &str| -> Result<&ArrayRef> {
            batch
                .column_by_name(name)
                .ok_or_else(|| Error::InvalidArchive(format!("index missing column '{name}'")))
        };
        macro_rules! cast {
            ($name:literal, $ty:ty) => {
                col($name)?
                    .as_any()
                    .downcast_ref::<$ty>()
                    .ok_or_else(|| {
                        Error::InvalidArchive(format!("index column '{}' has wrong type", $name))
                    })?
            };
        }
        let zoom = cast!("zoom", UInt8Array);
        let x = cast!("x", UInt32Array);
        let y = cast!("y", UInt32Array);
        let time_start = cast!("time_start", Int64Array);
        let time_end = cast!("time_end", Int64Array);
        let offset = cast!("offset", UInt64Array);
        let length = cast!("length", UInt32Array);
        let uncompressed_size = cast!("uncompressed_size", UInt32Array);
        let feature_count = cast!("feature_count", UInt32Array);
        let hilbert = cast!("hilbert", UInt64Array);

        // The integrity column changes name and width between versions; pick
        // the one that's actually in the schema rather than the requested
        // version, so a v3 archive built without the crc column (e.g. a
        // future variant) still decodes its identifying columns.
        let hash_u64 = batch
            .column_by_name("content_hash")
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
        let crc_u32 = batch
            .column_by_name("crc32c")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());

        for i in 0..batch.num_rows() {
            let content_hash = if let Some(arr) = hash_u64 {
                arr.value(i)
            } else if let Some(arr) = crc_u32 {
                arr.value(i) as u64
            } else if version == 3 {
                return Err(Error::InvalidArchive(
                    "v3 index missing both content_hash and crc32c columns".into(),
                ));
            } else {
                return Err(Error::InvalidArchive("v2 index missing content_hash".into()));
            };

            entries.push(TileEntry {
                zoom: zoom.value(i),
                x: x.value(i),
                y: y.value(i),
                time_start: time_start.value(i),
                time_end: time_end.value(i),
                offset: offset.value(i),
                length: length.value(i),
                uncompressed_size: uncompressed_size.value(i),
                feature_count: feature_count.value(i),
                hilbert: hilbert.value(i),
                content_hash,
            });
        }
    }
    Ok(entries)
}

/// 64-bit blake3 prefix used for v2 dedup + integrity.
fn blake3_prefix(bytes: &[u8]) -> u64 {
    let digest = blake3::hash(bytes);
    let b = digest.as_bytes();
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// CRC32C integrity tag used for v3.
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
    fn build(entries: &[TileEntry]) -> Self {
        let mut boundaries: Vec<i64> = entries
            .iter()
            .flat_map(|e| [e.time_start, e.time_end])
            .collect();
        boundaries.sort_unstable();
        boundaries.dedup();

        let mut bucket_offsets = Vec::with_capacity(boundaries.len() + 1);
        let mut bucket_refs = Vec::new();
        for &b in &boundaries {
            bucket_offsets.push(bucket_refs.len() as u32);
            for (idx, e) in entries.iter().enumerate() {
                if e.time_start <= b && b <= e.time_end {
                    bucket_refs.push(idx as u32);
                }
            }
        }
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
pub struct ArchiveReader {
    file: File,
    header: ArchiveHeader,
    entries: Vec<TileEntry>,
    metadata: crate::metadata::Metadata,
    temporal: TemporalLookup,
    /// Optional zstd training dictionary loaded from the v3 dictionary slot.
    /// `None` for v2 archives and v3 archives written without a dictionary.
    dictionary: Option<Vec<u8>>,
}

impl ArchiveReader {
    /// Open an archive for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let header = ArchiveHeader::read(&mut file)?;

        file.seek(SeekFrom::Start(header.index_offset))?;
        let mut index_bytes = vec![0u8; header.index_length as usize];
        file.read_exact(&mut index_bytes)?;
        let entries = decode_index(&index_bytes, header.version)?;

        file.seek(SeekFrom::Start(header.metadata_offset))?;
        let mut metadata_bytes = vec![0u8; header.metadata_length as usize];
        file.read_exact(&mut metadata_bytes)?;
        let metadata = crate::metadata::Metadata::from_json_bytes(&metadata_bytes)?;

        let dictionary = if header.version >= 3 && header.dictionary_length > 0 {
            file.seek(SeekFrom::Start(header.dictionary_offset))?;
            let mut buf = vec![0u8; header.dictionary_length as usize];
            file.read_exact(&mut buf)?;
            Some(buf)
        } else {
            None
        };

        let temporal = TemporalLookup::build(&entries);

        Ok(Self {
            file,
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

    /// Optional zstd training dictionary. `None` for v2 archives and v3
    /// archives written without a dictionary.
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

    /// Read and decompress a tile's raw payload bytes. Verifies the per-tile
    /// integrity tag (CRC32C for v3, blake3 prefix for v2).
    pub fn read_payload(&mut self, entry: &TileEntry) -> Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(entry.offset))?;
        let mut compressed = vec![0u8; entry.length as usize];
        self.file.read_exact(&mut compressed)?;

        let ok = if self.header.version >= 3 {
            crc32c_tag(&compressed) as u64 == entry.content_hash
        } else {
            blake3_prefix(&compressed) == entry.content_hash
        };
        if !ok {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} failed integrity check (corrupt archive)",
                entry.tile_id()
            )));
        }

        let payload = if self.header.version >= 3 && self.header.compression == Compression::Zstd {
            compression::decompress_zstd_with_dict(&compressed, self.dictionary.as_deref())?
        } else {
            compression::decompress(&compressed, self.header.compression)?
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
    pub fn read_layers(
        &mut self,
        entry: &TileEntry,
    ) -> Result<Vec<crate::arrow_tile::DecodedLayer>> {
        let payload = self.read_payload(entry)?;
        crate::arrow_tile::decode_tile(&payload)
    }
}

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

/// Writer for creating an STT archive. Always writes the latest version
/// (currently v3); use the [`ArchiveWriter::create_v2`] constructor to emit a
/// legacy v2 archive (kept for migration tools and tests).
pub struct ArchiveWriter {
    file: File,
    compression: Compression,
    version: u8,
    current_offset: u64,
    entries: Vec<TileEntry>,
    /// v2: maps blake3 prefix -> already-written `(offset, length)`. v3 has
    /// no dedup (it's near-zero on real-world continuous data) so this stays
    /// empty.
    dedup: std::collections::HashMap<u64, (u64, u32)>,
}

impl ArchiveWriter {
    /// Create a new v3 archive. Tile blobs use `compression`.
    pub fn create<P: AsRef<Path>>(path: P, compression: Compression) -> Result<Self> {
        Self::create_with_version(path, compression, FORMAT_VERSION)
    }

    /// Create a legacy v2 archive (gzip-only, blake3 dedup). Useful for
    /// migration scripts and the v2-compat test fixtures.
    pub fn create_v2<P: AsRef<Path>>(path: P, compression: Compression) -> Result<Self> {
        Self::create_with_version(path, compression, 2)
    }

    fn create_with_version<P: AsRef<Path>>(
        path: P,
        compression: Compression,
        version: u8,
    ) -> Result<Self> {
        if version != 2 && version != 3 {
            return Err(Error::InvalidArchive(format!(
                "cannot create archive at version {version}"
            )));
        }
        let file = OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut file = file;
        file.seek(SeekFrom::Start(HEADER_SIZE))?;
        Ok(Self {
            file,
            compression,
            version,
            current_offset: HEADER_SIZE,
            entries: Vec::new(),
            dedup: std::collections::HashMap::new(),
        })
    }

    /// Add a tile. `payload` is the uncompressed tile payload (the layer
    /// frame produced by [`crate::arrow_tile::encode_tile`]). For v2,
    /// byte-identical compressed blobs are written once and shared via the
    /// dedup table; v3 always appends.
    pub fn add_tile(
        &mut self,
        id: &TileId,
        time_start: i64,
        time_end: i64,
        feature_count: u32,
        payload: &[u8],
    ) -> Result<()> {
        let uncompressed_size = payload.len() as u32;
        let compressed = compression::compress(payload, self.compression)?;

        let (offset, length, hash_value) = if self.version >= 3 {
            // v3: CRC32C, no dedup. Real-data dedup hit rates are sub-1% on
            // continuous datasets — the blake3 + HashMap overhead is pure
            // cost. Skip both.
            let crc = crc32c_tag(&compressed) as u64;
            let offset = self.current_offset;
            let length = compressed.len() as u32;
            self.file.write_all(&compressed)?;
            self.current_offset += length as u64;
            (offset, length, crc)
        } else {
            let hash = blake3_prefix(&compressed);
            let (offset, length) = match self.dedup.get(&hash) {
                Some(&existing) => existing,
                None => {
                    let offset = self.current_offset;
                    let length = compressed.len() as u32;
                    self.file.write_all(&compressed)?;
                    self.current_offset += length as u64;
                    self.dedup.insert(hash, (offset, length));
                    (offset, length)
                }
            };
            (offset, length, hash)
        };

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
            content_hash: hash_value,
        });
        Ok(())
    }

    /// Number of tiles added so far.
    pub fn tile_count(&self) -> usize {
        self.entries.len()
    }

    /// Finalise: write the index table, metadata, and header.
    pub fn finalize(self, metadata: &crate::metadata::Metadata) -> Result<()> {
        self.finalize_with_dictionary(metadata, None)
    }

    /// Finalise, optionally embedding a zstd training dictionary.
    ///
    /// The dictionary slot is a v3-only feature: passing a dictionary on a
    /// v2 archive is an error (v2 has no on-disk slot for it).
    pub fn finalize_with_dictionary(
        mut self,
        metadata: &crate::metadata::Metadata,
        dictionary: Option<&[u8]>,
    ) -> Result<()> {
        if dictionary.is_some() && self.version < 3 {
            return Err(Error::InvalidArchive(
                "zstd training dictionary requires a v3 archive".into(),
            ));
        }

        // Sort the directory by (zoom, Hilbert index) for spatial locality.
        self.entries.sort_by_key(|e| (e.zoom, e.hilbert));

        // Optional dictionary section sits between the tile data and the
        // index so a reader can fetch it alongside the index in a coalesced
        // range request.
        let (dictionary_offset, dictionary_length) = if let Some(dict) = dictionary {
            let off = self.current_offset;
            self.file.write_all(dict)?;
            self.current_offset += dict.len() as u64;
            (off, dict.len() as u64)
        } else {
            (0, 0)
        };

        let index_bytes = encode_index(&self.entries, self.version)?;
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
            version: self.version,
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

    /// Create a new (v3) archive.
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
            properties: vec![],
        }
    }

    #[test]
    fn header_v3_roundtrips() {
        let header = ArchiveHeader {
            version: 3,
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
        assert_eq!(&buf[..4], MAGIC_V3);
        let read = ArchiveHeader::read(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(read.version, 3);
        assert_eq!(read.index_offset, 1234);
        assert_eq!(read.dictionary_offset, 1100);
        assert_eq!(read.dictionary_length, 32);
        assert_eq!(read.compression, Compression::Zstd);
    }

    #[test]
    fn header_v2_roundtrips() {
        let header = ArchiveHeader {
            version: 2,
            compression: Compression::Gzip,
            index_offset: 100,
            index_length: 20,
            metadata_offset: 120,
            metadata_length: 30,
            dictionary_offset: 0,
            dictionary_length: 0,
        };
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();
        assert_eq!(&buf[..4], MAGIC_V2);
        let read = ArchiveHeader::read(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(read.version, 2);
        assert_eq!(read.dictionary_length, 0);
    }

    #[test]
    fn archive_v3_roundtrips_tiles_and_temporal_lookup() {
        let path = NamedTempFile::new().unwrap().into_temp_path();

        let mut writer = ArchiveWriter::create(&path, Compression::Zstd).unwrap();
        let tile_a = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 1000)]).unwrap();
        let tile_b = crate::arrow_tile::encode_tile(&[point_layer("default", vec![2, 3], 1500)]).unwrap();
        let tile_c = crate::arrow_tile::encode_tile(&[point_layer("default", vec![4], 5000)]).unwrap();
        writer.add_tile(&TileId::new(10, 1, 1, 1000), 1000, 2000, 1, &tile_a).unwrap();
        writer.add_tile(&TileId::new(10, 2, 2, 1500), 1500, 3000, 2, &tile_b).unwrap();
        writer.add_tile(&TileId::new(11, 4, 4, 5000), 5000, 6000, 1, &tile_c).unwrap();

        let metadata = crate::metadata::Metadata::new("test-archive");
        writer.finalize(&metadata).unwrap();

        let mut reader = ArchiveReader::open(&path).unwrap();
        assert_eq!(reader.header().version, 3);
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
    fn archive_v2_still_reads_back() {
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create_v2(&path, Compression::Gzip).unwrap();
        let payload = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1, 2], 0)]).unwrap();
        writer.add_tile(&TileId::new(8, 1, 1, 0), 0, 1000, 2, &payload).unwrap();
        writer.finalize(&crate::metadata::Metadata::new("legacy")).unwrap();

        let mut reader = ArchiveReader::open(&path).unwrap();
        assert_eq!(reader.header().version, 2);
        let entry = reader.entries()[0].clone();
        let layers = reader.read_layers(&entry).unwrap();
        assert_eq!(layers[0].batch.num_rows(), 2);
    }

    #[test]
    fn v2_dedup_still_works() {
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create_v2(&path, Compression::Gzip).unwrap();
        let payload = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 1000)]).unwrap();
        writer.add_tile(&TileId::new(10, 1, 1, 1000), 1000, 2000, 1, &payload).unwrap();
        writer.add_tile(&TileId::new(10, 2, 2, 1000), 1000, 2000, 1, &payload).unwrap();
        writer.finalize(&crate::metadata::Metadata::new("dedup")).unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        let offsets: Vec<u64> = reader.entries().iter().map(|e| e.offset).collect();
        assert_eq!(offsets[0], offsets[1], "v2 dedup should share the blob");
    }

    #[test]
    fn corrupt_blob_is_detected_v3() {
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create(&path, Compression::None).unwrap();
        let payload = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 1000)]).unwrap();
        writer.add_tile(&TileId::new(5, 0, 0, 1000), 1000, 2000, 1, &payload).unwrap();
        writer.finalize(&crate::metadata::Metadata::new("corrupt")).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_SIZE as usize] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let mut reader = ArchiveReader::open(&path).unwrap();
        let entry = reader.entries()[0].clone();
        assert!(reader.read_payload(&entry).is_err());
    }

    /// Build a small but representative tile (point + categorical + path
     /// with per-vertex times) and assert that the v3 archive is at least
     /// 40% smaller than the v2 equivalent. The shrink is the combined
     /// effect of zstd-3 over gzip-6, u16-delta vertex times, dictionary
     /// categoricals, and the CRC32C column on the directory.
    #[test]
    fn v3_archives_are_substantially_smaller_than_v2() {
        use crate::arrow_tile::{ColumnarLayer, GeometryColumn, PropertyColumn};

        // 16 path-like layers per tile across 32 tiles. Each layer has
        // 64 features with ~16 vertices and a categorical "kind".
        let kinds = ["bike", "car", "scooter", "bus"];
        let make_tile = |seed: u64| -> Vec<u8> {
            let mut layers = Vec::new();
            for layer_i in 0..16u64 {
                let n = 64usize;
                let mut feature_ids = Vec::with_capacity(n);
                let mut start_times = Vec::with_capacity(n);
                let mut end_times = Vec::with_capacity(n);
                let mut paths: Vec<Vec<[f64; 2]>> = Vec::with_capacity(n);
                let mut vertex_times: Vec<Vec<i64>> = Vec::with_capacity(n);
                let mut kind_col: Vec<Option<String>> = Vec::with_capacity(n);
                for i in 0..n {
                    feature_ids.push(seed * 1000 + layer_i * 100 + i as u64);
                    let t0 = (i as i64) * 1000;
                    start_times.push(t0);
                    end_times.push(t0 + 1000);
                    let vertices: Vec<[f64; 2]> = (0..16)
                        .map(|j| {
                            [
                                -122.4 + (i as f64) * 0.001 + (j as f64) * 0.0001,
                                37.7 + (layer_i as f64) * 0.001 + (j as f64) * 0.0001,
                            ]
                        })
                        .collect();
                    let times: Vec<i64> = (0..16).map(|j| t0 + j * 60).collect();
                    paths.push(vertices);
                    vertex_times.push(times);
                    kind_col.push(Some(kinds[(i + layer_i as usize) % kinds.len()].to_string()));
                }
                layers.push(ColumnarLayer {
                    name: format!("layer_{layer_i}"),
                    feature_ids,
                    start_times,
                    end_times,
                    geometry: GeometryColumn::LineString(paths),
                    vertex_times: Some(vertex_times),
                    properties: vec![("kind".into(), PropertyColumn::Categorical(kind_col))],
                });
            }
            crate::arrow_tile::encode_tile(&layers).unwrap()
        };

        let build_archive = |path: &std::path::Path, version: u8| -> u64 {
            let mut writer = if version == 2 {
                ArchiveWriter::create_v2(path, Compression::Gzip).unwrap()
            } else {
                ArchiveWriter::create(path, Compression::Zstd).unwrap()
            };
            for seed in 0..32u64 {
                let payload = make_tile(seed);
                writer
                    .add_tile(
                        &TileId::new(8, seed as u32, 0, 0),
                        0,
                        1_000_000,
                        64 * 16,
                        &payload,
                    )
                    .unwrap();
            }
            writer
                .finalize(&crate::metadata::Metadata::new("size-comparison"))
                .unwrap();
            std::fs::metadata(path).unwrap().len()
        };

        let dir = tempfile::tempdir().unwrap();
        let v2_size = build_archive(&dir.path().join("v2.stt"), 2);
        let v3_size = build_archive(&dir.path().join("v3.stt"), 3);

        let ratio = v3_size as f64 / v2_size as f64;
        eprintln!(
            "v3 archive shrink: v2={} bytes, v3={} bytes, ratio={:.3}",
            v2_size, v3_size, ratio
        );
        assert!(
            ratio <= 0.60,
            "expected v3 archive to be at most 60% the size of v2, got ratio={:.3} (v2={}, v3={})",
            ratio,
            v2_size,
            v3_size
        );
    }

    #[test]
    fn v3_dictionary_slot_roundtrips() {
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create(&path, Compression::Zstd).unwrap();
        let payload = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 0)]).unwrap();
        writer.add_tile(&TileId::new(5, 0, 0, 0), 0, 1, 1, &payload).unwrap();
        // Embed an arbitrary "dictionary" blob in the slot (the reader does
        // not validate its zstd-magic, just the size; production builds use
        // a real trained dictionary).
        let fake_dict = b"DICTIONARY-BYTES-GO-HERE".to_vec();
        writer
            .finalize_with_dictionary(&crate::metadata::Metadata::new("dict"), Some(&fake_dict))
            .unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        assert_eq!(reader.dictionary(), Some(fake_dict.as_slice()));
    }
}
