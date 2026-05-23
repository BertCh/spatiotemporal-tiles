//! Error types for the STT library

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid tile coordinates: z={0}, x={1}, y={2}")]
    InvalidCoordinates(u8, u32, u32),

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(u64),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Invalid archive format: {0}")]
    InvalidArchive(String),

    #[error("Tile not found: {0:?}")]
    TileNotFound(crate::tile::TileId),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Geometry error: {0}")]
    Geometry(String),

    #[error("Invalid magic number")]
    InvalidMagic,

    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u8),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
