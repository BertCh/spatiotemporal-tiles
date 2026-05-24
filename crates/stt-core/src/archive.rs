//! STT archive format — a single-file, range-request-friendly container.
//!
//! ## Layout
//!
//! ```text
//! [ 64-byte header ]
//! [ tile blobs ... ]            gzip-compressed Arrow-IPC tile payloads
//! [ index table   ]            Arrow IPC: one row per tile (the directory)
//! [ metadata      ]            UTF-8 JSON
//! ```
//!
//! The header records the byte ranges of the index and metadata so a reader
//! can fetch them with two HTTP range requests, then each tile with one more.
//!
//! Tiles are content-addressed: identical compressed blobs are written once
//! and shared by every directory row that references them (dedup). Each row
//! also carries a 64-bit content hash for cheap corruption detection.

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
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

/// Magic number for STT archives. The trailing byte is the format version.
const MAGIC: &[u8; 4] = b"STT\x02";

/// Current archive format version.
pub const FORMAT_VERSION: u8 = 2;

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
    /// 64-bit content hash of the compressed blob (dedup + integrity).
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
    /// Format version.
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
    /// Read and validate a header.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::InvalidMagic);
        }
        let version = reader.read_u8()?;
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let compression = compression_from_byte(reader.read_u8()?)?;
        let index_offset = reader.read_u64::<LittleEndian>()?;
        let index_length = reader.read_u64::<LittleEndian>()?;
        let metadata_offset = reader.read_u64::<LittleEndian>()?;
        let metadata_length = reader.read_u64::<LittleEndian>()?;
        // Drain reserved bytes.
        let mut reserved = [0u8; (HEADER_SIZE as usize) - 38];
        reader.read_exact(&mut reserved)?;
        Ok(Self {
            version,
            compression,
            index_offset,
            index_length,
            metadata_offset,
            metadata_length,
        })
    }

    /// Write the header.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(MAGIC)?;
        writer.write_u8(self.version)?;
        writer.write_u8(compression_to_byte(self.compression))?;
        writer.write_u64::<LittleEndian>(self.index_offset)?;
        writer.write_u64::<LittleEndian>(self.index_length)?;
        writer.write_u64::<LittleEndian>(self.metadata_offset)?;
        writer.write_u64::<LittleEndian>(self.metadata_length)?;
        writer.write_all(&[0u8; (HEADER_SIZE as usize) - 38])?;
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Index table (Arrow) encode / decode
// ----------------------------------------------------------------------------

fn index_schema() -> Arc<Schema> {
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

fn encode_index(entries: &[TileEntry]) -> Result<Vec<u8>> {
    let schema = index_schema();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(entries.iter().map(|e| e.zoom).collect::<Vec<_>>())),
        Arc::new(UInt32Array::from(entries.iter().map(|e| e.x).collect::<Vec<_>>())),
        Arc::new(UInt32Array::from(entries.iter().map(|e| e.y).collect::<Vec<_>>())),
        Arc::new(Int64Array::from(entries.iter().map(|e| e.time_start).collect::<Vec<_>>())),
        Arc::new(Int64Array::from(entries.iter().map(|e| e.time_end).collect::<Vec<_>>())),
        Arc::new(UInt64Array::from(entries.iter().map(|e| e.offset).collect::<Vec<_>>())),
        Arc::new(UInt32Array::from(entries.iter().map(|e| e.length).collect::<Vec<_>>())),
        Arc::new(UInt32Array::from(
            entries.iter().map(|e| e.uncompressed_size).collect::<Vec<_>>(),
        )),
        Arc::new(UInt32Array::from(
            entries.iter().map(|e| e.feature_count).collect::<Vec<_>>(),
        )),
        Arc::new(UInt64Array::from(entries.iter().map(|e| e.hilbert).collect::<Vec<_>>())),
        Arc::new(UInt64Array::from(
            entries.iter().map(|e| e.content_hash).collect::<Vec<_>>(),
        )),
    ];
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

fn decode_index(bytes: &[u8]) -> Result<Vec<TileEntry>> {
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
        let content_hash = cast!("content_hash", UInt64Array);

        for i in 0..batch.num_rows() {
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
                content_hash: content_hash.value(i),
            });
        }
    }
    Ok(entries)
}

/// 64-bit content hash used for dedup and corruption detection.
fn content_hash(bytes: &[u8]) -> u64 {
    let digest = blake3::hash(bytes);
    let b = digest.as_bytes();
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
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
    /// Build the temporal lookup in `O(N log N + total bucket size)` time.
    ///
    /// The previous implementation enumerated every distinct boundary and
    /// scanned every entry for each — `O(N · B)`, with `B` up to `2N`. At a
    /// million tiles that was effectively `O(N²)` and `archive open` could
    /// take minutes on the larger archives. The sweep below produces the
    /// same snapshot semantics with an event-sorted pass plus an incremental
    /// `BTreeSet` of currently-active intervals.
    ///
    /// Snapshot semantics (unchanged from the previous impl):
    /// - `boundaries[i]` is a half-open boundary after which the active set
    ///   is stable until `boundaries[i+1]` (or +∞ for the last entry).
    /// - A query for time `t` uses `partition_point(|&b| b <= t)` to find the
    ///   bucket index `b`. `b == 0` means the query is before any tile
    ///   started → empty.
    fn build(entries: &[TileEntry]) -> Self {
        if entries.is_empty() {
            return Self::default();
        }

        // Half-open events: START at `time_start`, END at `time_end + 1`
        // (using `saturating_add` so an interval ending at `i64::MAX` still
        // produces a well-defined event time). Tag bytes pack the event
        // kind so the sort is a single `(t, kind, idx)` key — STARTs sort
        // BEFORE ENDs at ties, so an instantaneous interval `[t, t]` is
        // active exactly at time `t`.
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
        // Track the previous boundary's bucket so we can dedup consecutive
        // snapshots — an event-time where STARTs and ENDs cancel produces
        // the same active set, and emitting a duplicate just bloats the
        // `bucket_refs` storage and slows down queries marginally.
        let mut prev_bucket_start: usize = 0;
        while i < events.len() {
            let t = events[i].0;
            // Apply every event sharing this timestamp.
            while i < events.len() && events[i].0 == t {
                let (_, kind, idx) = events[i];
                if kind == START {
                    active.insert(idx);
                } else {
                    active.remove(&idx);
                }
                i += 1;
            }

            // Snapshot the active set. Skip when it matches the previous
            // boundary exactly (STARTs and ENDs cancelled out at this time)
            // so the boundary list stays minimal.
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
pub struct ArchiveReader {
    file: File,
    header: ArchiveHeader,
    entries: Vec<TileEntry>,
    metadata: crate::metadata::Metadata,
    temporal: TemporalLookup,
}

impl ArchiveReader {
    /// Open an archive for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let header = ArchiveHeader::read(&mut file)?;

        file.seek(SeekFrom::Start(header.index_offset))?;
        let mut index_bytes = vec![0u8; header.index_length as usize];
        file.read_exact(&mut index_bytes)?;
        let entries = decode_index(&index_bytes)?;

        file.seek(SeekFrom::Start(header.metadata_offset))?;
        let mut metadata_bytes = vec![0u8; header.metadata_length as usize];
        file.read_exact(&mut metadata_bytes)?;
        let metadata = crate::metadata::Metadata::from_json_bytes(&metadata_bytes)?;

        let temporal = TemporalLookup::build(&entries);

        Ok(Self {
            file,
            header,
            entries,
            metadata,
            temporal,
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

    /// Read and decompress a tile's raw payload bytes.
    pub fn read_payload(&mut self, entry: &TileEntry) -> Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(entry.offset))?;
        let mut compressed = vec![0u8; entry.length as usize];
        self.file.read_exact(&mut compressed)?;

        if content_hash(&compressed) != entry.content_hash {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} failed content-hash check (corrupt archive)",
                entry.tile_id()
            )));
        }

        let payload = compression::decompress(&compressed, self.header.compression)?;
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

/// Writer for creating an STT archive.
pub struct ArchiveWriter {
    file: File,
    compression: Compression,
    current_offset: u64,
    entries: Vec<TileEntry>,
    /// Maps a blob's content hash to its already-written `(offset, length)`.
    dedup: HashMap<u64, (u64, u32)>,
}

impl ArchiveWriter {
    /// Create a new archive. Tile blobs use `compression`.
    pub fn create<P: AsRef<Path>>(path: P, compression: Compression) -> Result<Self> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.seek(SeekFrom::Start(HEADER_SIZE))?;
        Ok(Self {
            file,
            compression,
            current_offset: HEADER_SIZE,
            entries: Vec::new(),
            dedup: HashMap::new(),
        })
    }

    /// Add a tile. `payload` is the uncompressed tile payload (the layer frame
    /// produced by [`crate::arrow_tile::encode_tile`]). Identical compressed
    /// blobs are written only once.
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
        let hash = content_hash(&compressed);

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
            content_hash: hash,
        });
        Ok(())
    }

    /// Number of tiles added so far.
    pub fn tile_count(&self) -> usize {
        self.entries.len()
    }

    /// Finalise: write the index table, metadata, and header.
    pub fn finalize(mut self, metadata: &crate::metadata::Metadata) -> Result<()> {
        // Sort the directory by (zoom, Hilbert index) for spatial locality.
        self.entries.sort_by_key(|e| (e.zoom, e.hilbert));

        let index_bytes = encode_index(&self.entries)?;
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

    /// Create a new archive.
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
    fn header_roundtrips() {
        let header = ArchiveHeader {
            version: FORMAT_VERSION,
            compression: Compression::Gzip,
            index_offset: 1234,
            index_length: 56,
            metadata_offset: 1290,
            metadata_length: 78,
        };
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE as usize);
        let read = ArchiveHeader::read(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(read.index_offset, 1234);
        assert_eq!(read.metadata_length, 78);
        assert_eq!(read.compression, Compression::Gzip);
    }

    #[test]
    fn archive_roundtrips_tiles_and_temporal_lookup() {
        let path = NamedTempFile::new().unwrap().into_temp_path();

        let mut writer = ArchiveWriter::create(&path, Compression::Gzip).unwrap();
        // Three tiles with overlapping temporal intervals.
        let tile_a = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 1000)]).unwrap();
        let tile_b = crate::arrow_tile::encode_tile(&[point_layer("default", vec![2, 3], 1500)]).unwrap();
        let tile_c = crate::arrow_tile::encode_tile(&[point_layer("default", vec![4], 5000)]).unwrap();
        writer.add_tile(&TileId::new(10, 1, 1, 1000), 1000, 2000, 1, &tile_a).unwrap();
        writer.add_tile(&TileId::new(10, 2, 2, 1500), 1500, 3000, 2, &tile_b).unwrap();
        writer.add_tile(&TileId::new(11, 4, 4, 5000), 5000, 6000, 1, &tile_c).unwrap();

        let metadata = crate::metadata::Metadata::new("test-archive");
        writer.finalize(&metadata).unwrap();

        let mut reader = ArchiveReader::open(&path).unwrap();
        assert_eq!(reader.entries().len(), 3);
        assert_eq!(reader.metadata().name, "test-archive");

        // T=1800 overlaps tiles A (1000-2000) and B (1500-3000).
        assert_eq!(reader.tiles_at_time(1800).len(), 2);
        // T=2500 overlaps only B.
        let at = reader.tiles_at_time(2500);
        assert_eq!(at.len(), 1);
        assert_eq!(at[0].x, 2);
        // T=4000 falls in a gap.
        assert!(reader.tiles_at_time(4000).is_empty());

        // Read a tile back and decode its layer.
        let entry = reader.find_tile(10, 2, 2, 2000).unwrap().clone();
        let layers = reader.read_layers(&entry).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "default");
        assert_eq!(layers[0].batch.num_rows(), 2);
    }

    #[test]
    fn identical_tiles_are_deduplicated() {
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create(&path, Compression::Gzip).unwrap();

        // Two tiles at different coords but byte-identical payloads.
        let payload = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 1000)]).unwrap();
        writer.add_tile(&TileId::new(10, 1, 1, 1000), 1000, 2000, 1, &payload).unwrap();
        writer.add_tile(&TileId::new(10, 2, 2, 1000), 1000, 2000, 1, &payload).unwrap();
        writer.finalize(&crate::metadata::Metadata::new("dedup")).unwrap();

        let reader = ArchiveReader::open(&path).unwrap();
        let offsets: Vec<u64> = reader.entries().iter().map(|e| e.offset).collect();
        // Both directory rows point at the same blob.
        assert_eq!(offsets[0], offsets[1]);
    }

    /// `TemporalLookup` build correctness + worst-case timing regression.
    ///
    /// Builds 50k bucket-aligned entries (representative of mid-size archives)
    /// and asserts:
    /// 1. The lookup returns exactly the entries whose `[start, end]` spans
    ///    a sample query time.
    /// 2. Build completes in well under a second — the previous O(N · B)
    ///    construction took 30+ s at this size and was a real archive-open
    ///    bottleneck for the AIS / NYC-taxi datasets.
    #[test]
    fn temporal_lookup_build_is_fast_and_correct() {
        let bucket = 3_600_000i64; // 1 hour
        let mut entries = Vec::with_capacity(50_000);
        // 200 spatial cells × 250 temporal buckets each = 50k entries.
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
                    content_hash: 0,
                });
            }
        }

        let started = std::time::Instant::now();
        let lookup = TemporalLookup::build(&entries);
        let elapsed_ms = started.elapsed().as_millis();
        assert!(
            elapsed_ms < 1000,
            "TemporalLookup::build took {elapsed_ms}ms for 50k entries — \
             the sweep-line build is meant to be sub-second",
        );

        // Sample query mid-archive: every entry whose interval contains this
        // time must be in the result, and nothing else.
        let query_t = bucket * 123 + bucket / 2;
        let got: std::collections::BTreeSet<u32> = lookup.at(query_t).iter().copied().collect();
        let expected: std::collections::BTreeSet<u32> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.time_start <= query_t && query_t <= e.time_end)
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(got, expected);

        // Before-everything and after-everything queries return empty.
        assert!(lookup.at(-1).is_empty());
        assert!(lookup.at(bucket * 1_000_000).is_empty());
    }

    #[test]
    fn corrupt_blob_is_detected() {
        let path = NamedTempFile::new().unwrap().into_temp_path();
        let mut writer = ArchiveWriter::create(&path, Compression::None).unwrap();
        let payload = crate::arrow_tile::encode_tile(&[point_layer("default", vec![1], 1000)]).unwrap();
        writer.add_tile(&TileId::new(5, 0, 0, 1000), 1000, 2000, 1, &payload).unwrap();
        writer.finalize(&crate::metadata::Metadata::new("corrupt")).unwrap();

        // Flip a byte inside the first tile blob (just past the 64-byte header).
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_SIZE as usize] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let mut reader = ArchiveReader::open(&path).unwrap();
        let entry = reader.entries()[0].clone();
        assert!(reader.read_payload(&entry).is_err());
    }
}
