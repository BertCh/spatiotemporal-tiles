//! Core library for the **SpatioTemporal Tiles (STT)** format.
//!
//! The canonical container is the **packed format** ([`pack`]): a per-dataset
//! `manifest.json` plus many content-addressed, immutable pack objects
//! (`packs/<blake3>.sttp`) and one directory object (`index/<blake3>.sttd`).
//! Splitting a dataset into small, immutable objects is what makes it
//! edge-cacheable on a plain CDN — see `docs/spec/stt-packed-format.md`.
//!
//! Building blocks:
//! - [`pack`] — the packed format: [`PackWriter`], [`PackedReader`], [`Manifest`],
//!   [`transcode_archive_to_packs`], [`verify_packed_objects`].
//! - [`arrow_tile`] — Arrow-IPC tile payloads with GeoArrow geometry.
//! - [`directory`] — the run-length tile directory (v5 = packed; v4 = legacy single-file).
//! - [`curve`] — spatial (Hilbert) + temporal blob ordering.
//! - [`metadata`] — dataset metadata (folded verbatim into the manifest).
//! - [`compression`] — per-blob zstd / gzip.
//!
//! The single-file [`archive`] container ([`ArchiveWriter`] / [`ArchiveReader`])
//! predates the packed format and is **no longer a deployment target**. It
//! survives only as (a) the bounded-RAM streaming *intermediate* that
//! `stt-build --streaming-arrow` transcodes into packs, and (b) the read side
//! that [`transcode_archive_to_packs`] consumes when migrating old archives.

pub mod archive;
pub mod arrow_tile;
pub mod budget;
pub mod compression;
pub mod curve;
pub mod directory;
pub mod error;
pub mod geometry;
pub mod metadata;
pub mod pack;
pub mod projection;
pub mod tile;
pub mod types;

// Re-export commonly used types.
// Packed format — the canonical container.
pub use pack::{
    transcode_archive_to_packs, verify_packed_objects, Manifest, PackWriter, PackedReader,
};
// Single-file archive — legacy/intermediate surface (see the module doc above).
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
