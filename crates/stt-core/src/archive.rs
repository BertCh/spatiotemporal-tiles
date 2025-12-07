//! Archive format and I/O operations

use crate::compression;
use crate::error::{Error, Result};
use crate::tile::TileId;
use crate::types::{BoundingBox, Compression, TimeRange};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use prost::Message;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic number for STT archives: "STT\x01"
const MAGIC: &[u8; 4] = b"STT\x01";

/// Current format version
const VERSION: u8 = 1;

/// Archive header size (fixed 53 bytes: 4 magic + 1 version + 32 u64s + 16 reserved)
const HEADER_SIZE: u64 = 53;

/// Archive header structure
#[derive(Debug, Clone)]
pub struct ArchiveHeader {
    pub version: u8,
    pub index_offset: u64,
    pub index_length: u64,
    pub metadata_offset: u64,
    pub metadata_length: u64,
}

impl ArchiveHeader {
    /// Read header from a file
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        // Read and verify magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::InvalidMagic);
        }

        // Read version
        let version = reader.read_u8()?;
        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }

        // Read offsets and lengths
        let index_offset = reader.read_u64::<LittleEndian>()?;
        let index_length = reader.read_u64::<LittleEndian>()?;
        let metadata_offset = reader.read_u64::<LittleEndian>()?;
        let metadata_length = reader.read_u64::<LittleEndian>()?;

        // Skip reserved bytes
        let mut _reserved = [0u8; 16];
        reader.read_exact(&mut _reserved)?;

        Ok(Self {
            version,
            index_offset,
            index_length,
            metadata_offset,
            metadata_length,
        })
    }

    /// Write header to a file
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write magic number
        writer.write_all(MAGIC)?;

        // Write version
        writer.write_u8(self.version)?;

        // Write offsets and lengths
        writer.write_u64::<LittleEndian>(self.index_offset)?;
        writer.write_u64::<LittleEndian>(self.index_length)?;
        writer.write_u64::<LittleEndian>(self.metadata_offset)?;
        writer.write_u64::<LittleEndian>(self.metadata_length)?;

        // Write reserved bytes (zeros)
        writer.write_all(&[0u8; 16])?;

        Ok(())
    }
}

/// Reader for STT archives
pub struct ArchiveReader {
    file: File,
    header: ArchiveHeader,
    index: crate::proto::Index,
    metadata: crate::proto::Metadata,
}

impl ArchiveReader {
    /// Open an existing archive for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;

        // Read header
        let header = ArchiveHeader::read(&mut file)?;

        // Read index
        file.seek(SeekFrom::Start(header.index_offset))?;
        let mut index_bytes = vec![0u8; header.index_length as usize];
        file.read_exact(&mut index_bytes)?;
        let index = crate::proto::Index::decode(&index_bytes[..])?;

        // Read metadata
        file.seek(SeekFrom::Start(header.metadata_offset))?;
        let mut metadata_bytes = vec![0u8; header.metadata_length as usize];
        file.read_exact(&mut metadata_bytes)?;
        let metadata = crate::proto::Metadata::decode(&metadata_bytes[..])?;

        Ok(Self {
            file,
            header,
            index,
            metadata,
        })
    }

    /// Get archive metadata
    pub fn metadata(&self) -> &crate::proto::Metadata {
        &self.metadata
    }

    /// Get the archive index
    pub fn index(&self) -> &crate::proto::Index {
        &self.index
    }

    /// Get tile by ID
    pub fn get_tile(&mut self, id: &TileId) -> Result<Option<crate::tile::Tile>> {
        // Find tile entry in index
        let entry = self.index.tiles.iter().find(|e| {
            e.zoom == id.z as u32
                && e.x == id.x
                && e.y == id.y
                && e.time_start <= id.t
                && e.time_end >= id.t
        });

        let entry = match entry {
            Some(e) => e,
            None => return Ok(None),
        };

        // Read compressed tile data
        self.file.seek(SeekFrom::Start(entry.offset))?;
        let mut compressed = vec![0u8; entry.length as usize];
        self.file.read_exact(&mut compressed)?;

        // Decompress
        let compression = Compression::from_proto(entry.compression);
        let data = compression::decompress(&compressed, compression)?;

        // Decode Protocol Buffer
        let proto_tile = crate::proto::Tile::decode(&data[..])?;

        // Convert to internal tile format
        let tile = self.proto_to_tile(*id, proto_tile)?;

        Ok(Some(tile))
    }

    /// Get tiles in a bounding box and time range
    pub fn get_tiles_in_bounds(
        &mut self,
        bounds: &BoundingBox,
        zoom: u8,
        time_range: &TimeRange,
    ) -> Result<Vec<crate::tile::Tile>> {
        // Convert bounding box to tile coordinates
        let tile_coords = bounds_to_tiles(bounds, zoom);

        let mut tiles = Vec::new();
        for (x, y) in tile_coords {
            // Use time_start as the query time (could be improved with range queries)
            let id = TileId::new(zoom, x, y, time_range.start);
            if let Some(tile) = self.get_tile(&id)? {
                tiles.push(tile);
            }
        }

        Ok(tiles)
    }

    fn proto_to_tile(
        &self,
        id: TileId,
        proto_tile: crate::proto::Tile,
    ) -> Result<crate::tile::Tile> {
        use crate::tile::{Feature, Layer, Position, Tile};
        use crate::types::GeometryType;

        let layers = proto_tile
            .layers
            .into_iter()
            .map(|proto_layer| {
                let features = proto_layer
                    .features
                    .into_iter()
                    .map(|proto_feature| {
                        let positions = proto_feature
                            .positions
                            .into_iter()
                            .map(|p| Position {
                                lon: p.lon,
                                lat: p.lat,
                            })
                            .collect();

                        let properties = proto_feature
                            .properties
                            .into_iter()
                            .map(|(key, val)| (key, proto_value_to_value(&val)))
                            .collect();

                        Feature {
                            id: proto_feature.id,
                            geometry_type: GeometryType::from_proto(proto_feature.r#type),
                            positions,
                            properties,
                            time_range: if proto_feature.valid_from > 0
                                && proto_feature.valid_to > 0
                            {
                                Some(TimeRange::new(
                                    proto_feature.valid_from,
                                    proto_feature.valid_to,
                                ))
                            } else {
                                None
                            },
                        }
                    })
                    .collect();

                Layer {
                    name: proto_layer.name,
                    extent: proto_layer.extent,
                    features,
                }
            })
            .collect();

        Ok(Tile {
            id,
            time_range: TimeRange::new(proto_tile.time_start, proto_tile.time_end),
            layers,
        })
    }
}

fn proto_value_to_value(proto_value: &crate::proto::Value) -> crate::tile::Value {
    use crate::proto::value::ValueType;
    use crate::tile::Value;

    match &proto_value.value_type {
        Some(ValueType::StringValue(s)) => Value::String(s.clone()),
        Some(ValueType::DoubleValue(d)) => Value::Double(*d),
        Some(ValueType::FloatValue(f)) => Value::Float(*f),
        Some(ValueType::IntValue(i)) => Value::Int(*i),
        Some(ValueType::UintValue(u)) => Value::UInt(*u),
        Some(ValueType::SintValue(s)) => Value::Int(*s),
        Some(ValueType::BoolValue(b)) => Value::Bool(*b),
        None => Value::String(String::new()),
    }
}

/// Convert geographic bounding box to tile coordinates
///
/// Now uses the standardized projection module.
fn bounds_to_tiles(bounds: &BoundingBox, zoom: u8) -> Vec<(u32, u32)> {
    use crate::projection::lonlat_to_tile;

    let n = 1u32 << zoom;

    // Convert lon/lat to tile coordinates using projection module
    let (min_x, min_y) = lonlat_to_tile(bounds.min_lon, bounds.max_lat, zoom).unwrap_or((0, 0));
    let (max_x, max_y) =
        lonlat_to_tile(bounds.max_lon, bounds.min_lat, zoom).unwrap_or((n - 1, n - 1));

    let mut tiles = Vec::new();
    for x in min_x..=max_x.min(n - 1) {
        for y in min_y..=max_y.min(n - 1) {
            tiles.push((x, y));
        }
    }
    tiles
}

/// Writer for creating STT archives
pub struct ArchiveWriter {
    file: File,
    current_offset: u64,
    tiles: Vec<crate::proto::TileEntry>,
}

impl ArchiveWriter {
    /// Create a new archive for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Reserve space for header (will be written at the end)
        file.seek(SeekFrom::Start(HEADER_SIZE))?;

        Ok(Self {
            file,
            current_offset: HEADER_SIZE,
            tiles: Vec::new(),
        })
    }

    /// Add a tile to the archive
    pub fn add_tile(
        &mut self,
        id: &TileId,
        tile: &crate::proto::Tile,
        compression: Compression,
    ) -> Result<()> {
        // Encode to Protocol Buffer
        let mut uncompressed = Vec::new();
        tile.encode(&mut uncompressed)?;

        let uncompressed_size = uncompressed.len() as u32;

        // Compress
        let compressed = compression::compress(&uncompressed, compression)?;

        // Write to file
        let offset = self.current_offset;
        let length = compressed.len() as u32;
        self.file.write_all(&compressed)?;
        self.current_offset += length as u64;

        // Add to index
        self.tiles.push(crate::proto::TileEntry {
            zoom: id.z as u32,
            x: id.x,
            y: id.y,
            time_start: tile.time_start,
            time_end: tile.time_end,
            offset,
            length,
            feature_count: tile.layers.iter().map(|l| l.features.len() as u32).sum(),
            compression: compression.to_proto(),
            uncompressed_size,
        });

        Ok(())
    }

    /// Finalize the archive (write index and metadata)
    pub fn finalize(mut self, metadata: &crate::proto::Metadata) -> Result<()> {
        // Build index
        let index = self.build_index();

        // Encode index
        let mut index_bytes = Vec::new();
        index.encode(&mut index_bytes)?;
        let index_offset = self.current_offset;
        let index_length = index_bytes.len() as u64;
        self.file.write_all(&index_bytes)?;
        self.current_offset += index_length;

        // Encode metadata
        let mut metadata_bytes = Vec::new();
        metadata.encode(&mut metadata_bytes)?;
        let metadata_offset = self.current_offset;
        let metadata_length = metadata_bytes.len() as u64;
        self.file.write_all(&metadata_bytes)?;
        self.current_offset += metadata_length;

        // Flush before truncating to ensure all data is written
        self.file.flush()?;

        // Truncate file to current position (remove any extra data)
        self.file.set_len(self.current_offset)?;

        // Write header at the beginning
        let header = ArchiveHeader {
            version: VERSION,
            index_offset,
            index_length,
            metadata_offset,
            metadata_length,
        };
        self.file.seek(SeekFrom::Start(0))?;
        header.write(&mut self.file)?;

        // Final flush to ensure everything is written to disk
        self.file.flush()?;

        Ok(())
    }

    fn build_index(&self) -> crate::proto::Index {
        // Sort tiles by Hilbert curve
        let mut sorted_tiles = self.tiles.clone();
        sorted_tiles.sort_by_key(|t| {
            let id = TileId::new(t.zoom as u8, t.x, t.y, t.time_start);
            id.hilbert_index()
        });

        // Build spatial index
        let mut hilbert_ids = Vec::new();
        let mut tile_indices = Vec::new();
        for (idx, tile) in sorted_tiles.iter().enumerate() {
            let id = TileId::new(tile.zoom as u8, tile.x, tile.y, tile.time_start);
            hilbert_ids.push(id.hilbert_index());
            tile_indices.push(idx as u32);
        }

        let spatial = crate::proto::SpatialIndex {
            hilbert_ids,
            tile_indices,
            zoom_offsets: vec![], // TODO: Calculate zoom offsets
        };

        // Build temporal index (simplified version)
        let mut timestamps: Vec<u64> = sorted_tiles.iter().map(|t| t.time_start).collect();
        timestamps.sort();
        timestamps.dedup();

        let temporal = crate::proto::TemporalIndex {
            timestamps,
            tile_ref_offsets: vec![],
            tile_refs: vec![],
        };

        crate::proto::Index {
            tiles: sorted_tiles,
            spatial: Some(spatial),
            temporal: Some(temporal),
        }
    }
}

/// Archive struct providing high-level access
pub struct Archive;

impl Archive {
    /// Open an existing archive
    pub fn open<P: AsRef<Path>>(path: P) -> Result<ArchiveReader> {
        ArchiveReader::open(path)
    }

    /// Create a new archive
    pub fn create<P: AsRef<Path>>(path: P) -> Result<ArchiveWriter> {
        ArchiveWriter::create(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_header_roundtrip() {
        let header = ArchiveHeader {
            version: 1,
            index_offset: 1000,
            index_length: 200,
            metadata_offset: 1200,
            metadata_length: 100,
        };

        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();

        let mut cursor = std::io::Cursor::new(buffer);
        let read_header = ArchiveHeader::read(&mut cursor).unwrap();

        assert_eq!(header.version, read_header.version);
        assert_eq!(header.index_offset, read_header.index_offset);
        assert_eq!(header.index_length, read_header.index_length);
    }
}
