//! Error types for the STT library
//!
//! Every variant here has a live construction site. Six speculative ones
//! (`InvalidTimestamp`, `TileNotFound`, `Index`, `Geometry`, `InvalidMagic`,
//! `UnsupportedVersion`) were removed: nothing ever built them, so matching on
//! them read as handling a case that could not occur. Magic-number and
//! format-version failures are reported through `InvalidArchive` with the
//! offending bytes in the message, which is what the readers actually do.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid tile coordinates: z={0}, x={1}, y={2}")]
    InvalidCoordinates(u8, u32, u32),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Invalid archive format: {0}")]
    InvalidArchive(String),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
