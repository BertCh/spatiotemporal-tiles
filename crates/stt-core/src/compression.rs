//! Compression for tile payloads. The format is **zstd-only**: every blob is
//! compressed independently (no shared dictionary) so any one decodes in
//! isolation, and [`Compression::None`] covers already-incompressible blobs.
//! zstd beats DEFLATE on both ratio and browser-side decode, so nothing else
//! is offered.
//!
//! A packed archive names its codec as a **string** in `manifest.json` —
//! `"zstd"` or `"none"`, enumerated in `docs/spec/manifest.schema.json`. No
//! live STT format encodes the codec as a byte.
//!
//! # gzip
//!
//! The canonical account. [`crate::types::Compression`], `stt-build
//! --compression`, and `@poopdeck.gl/core` defer here; three claims that are
//! easy to conflate are kept apart:
//!
//! - **It was a real codec, not merely an accepted flag value.** The
//!   single-file `.stt` archive that preceded the packed format defaulted to
//!   gzip and wrote true DEFLATE frames, tagged `1` in that header's
//!   compression byte.
//! - **No release and no packed archive ever carried it.** The single-file
//!   format is gone, its codec with it, and no packed writer has ever put
//!   anything but `"zstd"` in a manifest. This crate has no DEFLATE
//!   implementation — the gzip elsewhere in the workspace decompresses gzipped
//!   *input* files (Parquet, CSV), never tile bytes.
//! - **It cannot reappear silently.** `--compression gzip` is a hard error, and
//!   a manifest claiming `"gzip"` is refused at open rather than assumed to be
//!   zstd, which would fail late and opaquely inside the decoder.
//!
//! The codec number `1` stays unassigned: reusing it would let a byte salvaged
//! from a single-file archive decode as a live codec.

use crate::error::{Error, Result};
use crate::types::Compression;

/// Compress data using the specified compression method
pub fn compress(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::Zstd => compress_zstd(data),
    }
}

/// Decompress data using the specified compression method
pub fn decompress(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::Zstd => decompress_zstd(data),
    }
}

/// Default zstd level for the packed format. Level 3 is zstd's documented
/// "fast" sweet spot — roughly DEFLATE-6 ratio at ~5x the encode speed; higher
/// levels save a few percent at significant CPU cost. `compress_zstd_with_dict`
/// and the writers use this unless an explicit level is threaded through (e.g.
/// `stt-build --zstd-level`).
pub const ZSTD_LEVEL: i32 = 3;

/// Highest level we expose. zstd defines 1..=22; level 19 is the practical
/// ceiling (19 ≈ 22 on STT tiles, measured) but we accept up to 22.
pub const ZSTD_LEVEL_MAX: i32 = 22;

fn compress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    zstd::stream::encode_all(data, ZSTD_LEVEL).map_err(|e| Error::Compression(e.to_string()))
}

fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    zstd::stream::decode_all(data).map_err(|e| Error::Decompression(e.to_string()))
}

/// Compress a payload with zstd using an optional pre-shared dictionary, at the
/// default level ([`ZSTD_LEVEL`]). Thin wrapper over
/// [`compress_zstd_with_dict_level`] — kept as the stable call site for the
/// many writers that don't tune the level.
pub fn compress_zstd_with_dict(data: &[u8], dict: Option<&[u8]>) -> Result<Vec<u8>> {
    compress_zstd_with_dict_level(data, dict, ZSTD_LEVEL)
}

/// Compress a payload with zstd at an explicit `level`, with an optional
/// pre-shared dictionary.
///
/// The packed format is **write-once / serve-many**: build CPU is paid once,
/// offline, while the bytes are paid on every client fetch. Decompression speed
/// is independent of the level the frame was written at, so a higher build-time
/// level is a pure win on the wire (measured −10..19% at level 19 vs 3). When
/// `dict` is `Some`, the encoder primes itself with the dictionary — the decoder
/// needs the same dictionary to round-trip.
pub fn compress_zstd_with_dict_level(
    data: &[u8],
    dict: Option<&[u8]>,
    level: i32,
) -> Result<Vec<u8>> {
    let level = level.clamp(1, ZSTD_LEVEL_MAX);
    let Some(dict) = dict else {
        return zstd::stream::encode_all(data, level)
            .map_err(|e| Error::Compression(e.to_string()));
    };
    let mut out = Vec::with_capacity(data.len() / 2 + 64);
    let mut encoder = zstd::stream::Encoder::with_dictionary(&mut out, level, dict)
        .map_err(|e| Error::Compression(e.to_string()))?;
    use std::io::Write;
    encoder
        .write_all(data)
        .map_err(|e| Error::Compression(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| Error::Compression(e.to_string()))?;
    Ok(out)
}

/// Decompress zstd data, applying a pre-shared dictionary when supplied.
pub fn decompress_zstd_with_dict(data: &[u8], dict: Option<&[u8]>) -> Result<Vec<u8>> {
    let Some(dict) = dict else {
        return decompress_zstd(data);
    };
    let mut decoder = zstd::stream::Decoder::with_dictionary(data, dict)
        .map_err(|e| Error::Decompression(e.to_string()))?;
    let mut out = Vec::new();
    use std::io::Read as _;
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::Decompression(e.to_string()))?;
    Ok(out)
}

/// Train a zstd dictionary from a sample of payloads.
///
/// Returns `None` when the sample is too small for zstd's dictionary builder
/// to produce a useful dictionary; callers should treat this as "ship without
/// a dictionary" rather than an error.
pub fn train_zstd_dictionary(samples: &[Vec<u8>], max_size: usize) -> Option<Vec<u8>> {
    let slices: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
    train_zstd_dictionary_from_slices(&slices, max_size)
}

/// Train a zstd dictionary from borrowed payload slices (no cloning).
///
/// Same contract as [`train_zstd_dictionary`]: returns `None` when the sample
/// is too small or too sparse for zstd's builder to produce a useful
/// dictionary, so callers treat that as "ship without a dictionary".
pub fn train_zstd_dictionary_from_slices(samples: &[&[u8]], max_size: usize) -> Option<Vec<u8>> {
    if samples.is_empty() {
        return None;
    }
    match zstd::dict::from_samples(samples, max_size) {
        Ok(dict) if !dict.is_empty() => Some(dict),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_roundtrip() {
        let data = b"Hello, world! This is a test string that should compress well.".repeat(50);
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
