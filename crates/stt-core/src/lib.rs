//! Core library for the SpatioTemporal Tiles (STT) format.
//!
//! This crate provides the fundamental types and operations for working with
//! STT archives, including:
//!
//! - Arrow-IPC tile payloads with GeoArrow geometry ([`arrow_tile`])
//! - The single-file [`archive`] container (Arrow index, JSON metadata)
//! - Spatial (Hilbert curve) and temporal indexing
//! - Gzip compression

pub mod analyzer;
pub mod archive;
pub mod arrow_tile;
pub mod budget;
pub mod compression;
pub mod curve;
pub mod directory;
pub mod error;
pub mod geometry;
pub mod index;
pub mod metadata;
pub mod projection;
pub mod tile;
pub mod types;

// Re-export commonly used types
pub use archive::{Archive, ArchiveReader, ArchiveWriter};
pub use curve::BlobOrdering;
pub use error::{Error, Result};
pub use tile::TileId;
pub use types::{BoundingBox, TimeRange};

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
