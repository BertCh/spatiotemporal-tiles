//! Core library for spatiotemporal tile format
//!
//! This crate provides the fundamental types and operations for working with
//! SpatioTemporal Tiles (STT), including:
//!
//! - Tile encoding and decoding (Protocol Buffers)
//! - Spatial indexing (Hilbert curve)
//! - Temporal indexing (sorted timestamps)
//! - Archive I/O operations
//! - Compression (Brotli, Gzip)

pub mod analyzer;
pub mod archive;
pub mod budget;
pub mod compression;
pub mod error;
pub mod geometry;
pub mod index;
pub mod metadata;
pub mod projection;
pub mod tile;
pub mod types;

// Re-export commonly used types
pub use archive::{Archive, ArchiveReader, ArchiveWriter};
pub use error::{Error, Result};
pub use tile::{Feature, Layer, Tile, TileId};
pub use types::{BoundingBox, TimeRange};

// Include generated Protocol Buffer code
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/stt.rs"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_id_ordering() {
        let t1 = TileId::new(10, 512, 384, 1609459200000);
        let t2 = TileId::new(10, 512, 384, 1609545600000);
        assert!(t1 < t2);
    }
}
