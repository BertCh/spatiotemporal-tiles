//! Compression utilities for tiles

use crate::error::{Error, Result};
use crate::types::Compression;
use flate2::read::{GzDecoder, GzEncoder};
use flate2::Compression as GzipLevel;
use std::io::Read;

/// Compress data using the specified compression method
pub fn compress(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::Gzip => compress_gzip(data),
    }
}

/// Decompress data using the specified compression method
pub fn decompress(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::Gzip => decompress_gzip(data),
    }
}

fn compress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(data, GzipLevel::best());
    let mut compressed = Vec::new();
    encoder
        .read_to_end(&mut compressed)
        .map_err(|e| Error::Compression(e.to_string()))?;
    Ok(compressed)
}

fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| Error::Decompression(e.to_string()))?;
    Ok(decompressed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gzip_roundtrip() {
        let data = b"Hello, world! This is a test string that should compress well.";
        let compressed = compress(data, Compression::Gzip).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = decompress(&compressed, Compression::Gzip).unwrap();
        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_none_roundtrip() {
        let data = b"No compression";
        let compressed = compress(data, Compression::None).unwrap();
        assert_eq!(&compressed, data);

        let decompressed = decompress(&compressed, Compression::None).unwrap();
        assert_eq!(&decompressed, data);
    }
}

