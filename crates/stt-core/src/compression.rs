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
        Compression::Zstd => compress_zstd(data),
    }
}

/// Decompress data using the specified compression method
pub fn decompress(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::Gzip => decompress_gzip(data),
        Compression::Zstd => decompress_zstd(data),
    }
}

fn compress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    // Level 6 (zlib default): ~3-5x faster than level 9 for <1% larger output
    // — the right trade-off when compressing thousands of tiles.
    let mut encoder = GzEncoder::new(data, GzipLevel::new(6));
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

/// Default zstd level. Level 3 is zstd's documented "fast" sweet spot —
/// roughly gzip-6 ratio at ~5x the encode speed; higher levels save a few
/// percent at significant CPU cost.
const ZSTD_LEVEL: i32 = 3;

fn compress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    zstd::stream::encode_all(data, ZSTD_LEVEL).map_err(|e| Error::Compression(e.to_string()))
}

fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    zstd::stream::decode_all(data).map_err(|e| Error::Decompression(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gzip_roundtrip() {
        // Use a payload large and repetitive enough to overcome gzip's
        // fixed ~18-byte header/footer overhead. A short non-repetitive
        // string would (correctly) fail to shrink.
        let data = b"Hello, world! This is a test string that should compress well."
            .repeat(50);
        let compressed = compress(&data, Compression::Gzip).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = decompress(&compressed, Compression::Gzip).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_zstd_roundtrip() {
        let data = b"Hello, world! This is a test string that should compress well."
            .repeat(50);
        let compressed = compress(&data, Compression::Zstd).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = decompress(&compressed, Compression::Zstd).unwrap();
        assert_eq!(decompressed, data);
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
