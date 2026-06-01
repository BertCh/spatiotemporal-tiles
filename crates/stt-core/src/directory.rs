//! STT v4 directory — a compact, range-request-friendly tile index.
//!
//! Replaces the v2/v3 Arrow-IPC index (fixed-width columns + IPC framing) with
//! a columnar binary encoding inspired by PMTiles v3:
//!
//! - **Columnar + delta + zig-zag varints.** Entries are sorted by
//!   `(zoom, hilbert, time_start)`, so each column (zoom, hilbert, x, y,
//!   time_start) is near-monotonic and delta-codes to ~1 byte per entry.
//! - **Blob-run RLE.** Consecutive entries that point at the *same physical
//!   blob* (a spatial cell whose content is identical across consecutive time
//!   buckets — the temporal analogue of PMTiles' ocean tiles) collapse into one
//!   run. The heavy per-blob columns (offset/length/uncompressed/crc) are then
//!   stored once per *run* instead of once per *entry*.
//! - **Offset contiguity sentinel.** A run whose blob immediately follows the
//!   previous run's blob stores offset `0`; otherwise `offset + 1`. Sequential
//!   archives (the common case) cost ~1 byte for the whole offset column.
//!
//! The directory is self-describing (leading version byte + entry/run counts)
//! and decodes to exactly the `TileEntry` list that was encoded, provided the
//! input was already (or is internally re-)sorted into directory order.
//!
//! This module is pure (no I/O): `encode_directory` / `decode_directory` map
//! `&[TileEntry] ⇆ Vec<u8>`. The archive writer/reader own where the buffer
//! lives in the file (it occupies the header's `index_*` byte range for v4).

use crate::archive::TileEntry;
use crate::error::{Error, Result};

/// Directory format tag (first byte of the buffer). Bumped independently of the
/// archive `FORMAT_VERSION` so the directory codec can evolve on its own.
pub const DIRECTORY_VERSION: u8 = 4;

// ----------------------------------------------------------------------------
// LEB128 varints
// ----------------------------------------------------------------------------

fn put_uvarint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            buf.push(byte | 0x80);
        } else {
            buf.push(byte);
            break;
        }
    }
}

fn get_uvarint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf
            .get(*pos)
            .ok_or_else(|| Error::InvalidArchive("directory: truncated varint".into()))?;
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::InvalidArchive("directory: varint exceeds 64 bits".into()));
        }
    }
    Ok(result)
}

#[inline]
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

fn put_ivarint(buf: &mut Vec<u8>, v: i64) {
    put_uvarint(buf, zigzag(v));
}

fn get_ivarint(buf: &[u8], pos: &mut usize) -> Result<i64> {
    Ok(unzigzag(get_uvarint(buf, pos)?))
}

// ----------------------------------------------------------------------------
// Encode
// ----------------------------------------------------------------------------

/// Encode tile entries into the v4 directory buffer.
///
/// Entries are sorted into directory order `(zoom, hilbert, time_start)` first,
/// so the caller need not pre-sort. Two entries are considered to share a blob
/// (and so RLE-collapse) when their `(offset, length, uncompressed_size,
/// crc32c)` all match — which is exactly what the dedup-on-write path produces
/// for byte-identical tiles.
pub fn encode_directory(entries: &[TileEntry]) -> Vec<u8> {
    let mut sorted: Vec<&TileEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
    let n = sorted.len();

    // Compute blob runs up front so we can write the run count into the header.
    // A run is a maximal stretch of consecutive entries pointing at one blob.
    let mut runs: Vec<(usize, u64, u32, u32, u32)> = Vec::new();
    let mut i = 0;
    while i < n {
        let head = sorted[i];
        let crc = head.crc32c;
        let mut j = i + 1;
        while j < n {
            let e = sorted[j];
            if e.offset == head.offset
                && e.length == head.length
                && e.uncompressed_size == head.uncompressed_size
                && e.crc32c == crc
            {
                j += 1;
            } else {
                break;
            }
        }
        runs.push((j - i, head.offset, head.length, head.uncompressed_size, crc));
        i = j;
    }

    let mut buf = Vec::with_capacity(n * 8 + runs.len() * 8 + 16);
    buf.push(DIRECTORY_VERSION);
    put_uvarint(&mut buf, n as u64);
    put_uvarint(&mut buf, runs.len() as u64);

    // Per-entry key columns (delta / zig-zag coded against the previous entry).
    let mut prev_zoom = 0i64;
    let mut prev_hilbert = 0i64;
    let mut prev_x = 0i64;
    let mut prev_y = 0i64;
    let mut prev_t = 0i64;
    for e in &sorted {
        put_ivarint(&mut buf, (e.zoom as i64).wrapping_sub(prev_zoom));
        prev_zoom = e.zoom as i64;
        put_ivarint(&mut buf, (e.hilbert as i64).wrapping_sub(prev_hilbert));
        prev_hilbert = e.hilbert as i64;
        put_ivarint(&mut buf, (e.x as i64).wrapping_sub(prev_x));
        prev_x = e.x as i64;
        put_ivarint(&mut buf, (e.y as i64).wrapping_sub(prev_y));
        prev_y = e.y as i64;
        put_ivarint(&mut buf, e.time_start.wrapping_sub(prev_t));
        prev_t = e.time_start;
        // duration may legitimately be 0; store signed so end<start round-trips too.
        put_ivarint(&mut buf, e.time_end.wrapping_sub(e.time_start));
        put_uvarint(&mut buf, e.feature_count as u64);
        // temporal_bucket_ms: a presence flag (0 = None, 1 = Some) followed by
        // the raw value when present — so every u64 (incl. u64::MAX) round-trips
        // without colliding with the None sentinel.
        match e.temporal_bucket_ms {
            Some(v) => {
                put_uvarint(&mut buf, 1);
                put_uvarint(&mut buf, v);
            }
            None => put_uvarint(&mut buf, 0),
        }
    }

    // Per-run blob columns with offset contiguity.
    let mut expected_offset = 0u64;
    for (run_len, offset, length, uncompressed, crc) in &runs {
        put_uvarint(&mut buf, *run_len as u64);
        // Offset: 0 = contiguous (== expected); else a `1` flag followed by the
        // raw offset, so a real u64::MAX offset can't collide with the
        // contiguity sentinel.
        if *offset == expected_offset {
            put_uvarint(&mut buf, 0);
        } else {
            put_uvarint(&mut buf, 1);
            put_uvarint(&mut buf, *offset);
        }
        put_uvarint(&mut buf, *length as u64);
        put_uvarint(&mut buf, *uncompressed as u64);
        buf.extend_from_slice(&crc.to_le_bytes());
        expected_offset = offset.wrapping_add(*length as u64);
    }

    buf
}

// ----------------------------------------------------------------------------
// Decode
// ----------------------------------------------------------------------------

/// Decode a v4 directory buffer back into tile entries (in directory order).
pub fn decode_directory(bytes: &[u8]) -> Result<Vec<TileEntry>> {
    let mut pos = 0usize;
    let version = *bytes
        .first()
        .ok_or_else(|| Error::InvalidArchive("directory: empty buffer".into()))?;
    pos += 1;
    if version != DIRECTORY_VERSION {
        return Err(Error::InvalidArchive(format!(
            "directory: unsupported version {version} (expected {DIRECTORY_VERSION})"
        )));
    }
    let n = get_uvarint(bytes, &mut pos)? as usize;
    let run_count = get_uvarint(bytes, &mut pos)? as usize;

    // Decode the per-entry key columns into a scratch buffer; blob fields are
    // filled in during the run expansion below.
    struct Key {
        zoom: u8,
        hilbert: u64,
        x: u32,
        y: u32,
        time_start: i64,
        time_end: i64,
        feature_count: u32,
        temporal_bucket_ms: Option<u64>,
    }
    let mut keys: Vec<Key> = Vec::with_capacity(n);
    let mut prev_zoom = 0i64;
    let mut prev_hilbert = 0i64;
    let mut prev_x = 0i64;
    let mut prev_y = 0i64;
    let mut prev_t = 0i64;
    for _ in 0..n {
        prev_zoom = prev_zoom.wrapping_add(get_ivarint(bytes, &mut pos)?);
        prev_hilbert = prev_hilbert.wrapping_add(get_ivarint(bytes, &mut pos)?);
        prev_x = prev_x.wrapping_add(get_ivarint(bytes, &mut pos)?);
        prev_y = prev_y.wrapping_add(get_ivarint(bytes, &mut pos)?);
        prev_t = prev_t.wrapping_add(get_ivarint(bytes, &mut pos)?);
        let duration = get_ivarint(bytes, &mut pos)?;
        let feature_count_raw = get_uvarint(bytes, &mut pos)?;
        let temporal_bucket_ms = if get_uvarint(bytes, &mut pos)? == 0 {
            None
        } else {
            Some(get_uvarint(bytes, &mut pos)?)
        };
        // Validate the spatial / feature columns fit their target widths, so a
        // corrupt (or foreign mis-encoded) directory errors loudly instead of
        // silently truncating through `as u8` / `as u32`.
        if !(0..=u8::MAX as i64).contains(&prev_zoom) {
            return Err(Error::InvalidArchive(format!(
                "directory: zoom {prev_zoom} out of u8 range"
            )));
        }
        if !(0..=u32::MAX as i64).contains(&prev_x) {
            return Err(Error::InvalidArchive(format!(
                "directory: x {prev_x} out of u32 range"
            )));
        }
        if !(0..=u32::MAX as i64).contains(&prev_y) {
            return Err(Error::InvalidArchive(format!(
                "directory: y {prev_y} out of u32 range"
            )));
        }
        if feature_count_raw > u32::MAX as u64 {
            return Err(Error::InvalidArchive(format!(
                "directory: feature_count {feature_count_raw} out of u32 range"
            )));
        }
        keys.push(Key {
            zoom: prev_zoom as u8,
            hilbert: prev_hilbert as u64,
            x: prev_x as u32,
            y: prev_y as u32,
            time_start: prev_t,
            time_end: prev_t.wrapping_add(duration),
            feature_count: feature_count_raw as u32,
            temporal_bucket_ms,
        });
    }

    // Expand runs over the keys, assigning each run's shared blob fields.
    let mut entries = Vec::with_capacity(n);
    let mut cursor = 0usize;
    let mut expected_offset = 0u64;
    for _ in 0..run_count {
        let run_len = get_uvarint(bytes, &mut pos)? as usize;
        let offset = if get_uvarint(bytes, &mut pos)? == 0 {
            expected_offset
        } else {
            get_uvarint(bytes, &mut pos)?
        };
        let length = get_uvarint(bytes, &mut pos)? as u32;
        let uncompressed_size = get_uvarint(bytes, &mut pos)? as u32;
        let crc = u32::from_le_bytes(
            bytes
                .get(pos..pos + 4)
                .ok_or_else(|| Error::InvalidArchive("directory: truncated crc".into()))?
                .try_into()
                .unwrap(),
        );
        pos += 4;

        if cursor + run_len > n {
            return Err(Error::InvalidArchive(
                "directory: run length exceeds entry count".into(),
            ));
        }
        for _ in 0..run_len {
            let k = &keys[cursor];
            cursor += 1;
            entries.push(TileEntry {
                zoom: k.zoom,
                x: k.x,
                y: k.y,
                time_start: k.time_start,
                time_end: k.time_end,
                offset,
                length,
                uncompressed_size,
                feature_count: k.feature_count,
                hilbert: k.hilbert,
                crc32c: crc,
                temporal_bucket_ms: k.temporal_bucket_ms,
            });
        }
        expected_offset = offset.wrapping_add(length as u64);
    }

    if cursor != n {
        return Err(Error::InvalidArchive(format!(
            "directory: runs covered {cursor} entries, expected {n}"
        )));
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        zoom: u8,
        x: u32,
        y: u32,
        hilbert: u64,
        ts: i64,
        te: i64,
        offset: u64,
        length: u32,
        unc: u32,
        fc: u32,
        crc: u32,
        tb: Option<u64>,
    ) -> TileEntry {
        TileEntry {
            zoom,
            x,
            y,
            time_start: ts,
            time_end: te,
            offset,
            length,
            uncompressed_size: unc,
            feature_count: fc,
            hilbert,
            crc32c: crc,
            temporal_bucket_ms: tb,
        }
    }

    #[test]
    fn empty_roundtrips() {
        let bytes = encode_directory(&[]);
        let back = decode_directory(&bytes).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn single_entry_roundtrips() {
        let e = entry(10, 5, 7, 42, 1000, 2000, 64, 128, 256, 3, 0xDEAD_BEEF, Some(3_600_000));
        let bytes = encode_directory(std::slice::from_ref(&e));
        let back = decode_directory(&bytes).unwrap();
        assert_eq!(back, vec![e]);
    }

    /// Distinct contiguous blobs: the offset column should ride the contiguity
    /// sentinel, and every field must round-trip exactly.
    #[test]
    fn contiguous_distinct_blobs_roundtrip() {
        let mut entries = Vec::new();
        let mut offset = 64u64;
        for i in 0..50u32 {
            let len = 100 + i;
            entries.push(entry(
                12,
                i,
                i,
                i as u64, // hilbert monotonic → already in directory order
                (i as i64) * 1000,
                (i as i64) * 1000 + 500,
                offset,
                len,
                len * 2,
                i,
                0x1000 + i,
                None,
            ));
            offset += len as u64;
        }
        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();
        assert_eq!(back, entries);
    }

    /// Temporal RLE: one spatial cell whose content is identical across many
    /// time buckets (same blob → same offset/length/crc). The encoding must
    /// collapse these into a single run yet decode back to every entry.
    #[test]
    fn identical_across_time_collapses_to_one_run_and_roundtrips() {
        let crc = 0xABCD_1234u32;
        let mut entries = Vec::new();
        // 100 consecutive hourly buckets of one static cell, all the same blob.
        for b in 0..100u64 {
            entries.push(entry(
                9,
                3,
                4,
                77, // same hilbert (same cell)
                (b as i64) * 3_600_000,
                (b as i64) * 3_600_000 + 3_599_999,
                4096,   // same offset (deduped blob)
                512,    // same length
                1024,   // same uncompressed
                64,
                crc,    // same crc
                Some(3_600_000),
            ));
        }
        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();
        assert_eq!(back, entries);

        // The headline RLE win is on the per-blob columns (offset/length/
        // uncompressed/crc): the static run stores them once instead of 100×.
        // Compare against the same corpus with *distinct* blobs, where every
        // entry needs its own run.
        let mut distinct = entries.clone();
        for (i, e) in distinct.iter_mut().enumerate() {
            e.offset = 4096 + i as u64 * 512;
            e.crc32c = 0x9000 + i as u32;
        }
        let distinct_bytes = encode_directory(&distinct);
        eprintln!(
            "static-cell RLE directory: {} bytes vs distinct-blob: {} bytes",
            bytes.len(),
            distinct_bytes.len()
        );
        // The blob columns are ~10 bytes/run. Collapsing 100 runs → 1 must save
        // close to the full 99 × 10 bytes the distinct encoding spends on them.
        assert!(
            distinct_bytes.len() >= bytes.len() + 99 * 8,
            "RLE should reclaim the per-blob columns: rle={}, distinct={}",
            bytes.len(),
            distinct_bytes.len()
        );
    }

    /// A mixed corpus across several zooms, cells, times, with some shared
    /// blobs and some unsorted input — the codec must sort, RLE, and round-trip.
    #[test]
    fn mixed_unsorted_corpus_roundtrips() {
        let mut entries = Vec::new();
        let mut off = 64u64;
        for zoom in [4u8, 8, 12] {
            for cell in 0..20u64 {
                for t in 0..5i64 {
                    let len = 80 + (cell as u32 % 7);
                    // Every 3rd (cell,t) reuses the previous blob to exercise RLE.
                    let shared = t > 0 && t % 3 != 0;
                    let (offset, crc) = if shared {
                        (off, 0x5555)
                    } else {
                        off += len as u64;
                        (off, 0x6000 + cell as u32 + t as u32)
                    };
                    entries.push(entry(
                        zoom,
                        cell as u32,
                        (cell * 2) as u32,
                        cell * 10 + zoom as u64, // arbitrary but stable hilbert
                        t * 1000 + cell as i64,
                        t * 1000 + cell as i64 + 250,
                        offset,
                        len,
                        len * 3,
                        (cell + t as u64) as u32,
                        crc,
                        if zoom == 4 { Some(86_400_000) } else { None },
                    ));
                }
            }
        }
        // Shuffle deterministically so encode must sort.
        entries.reverse();

        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();

        // Expected = the same entries in directory order.
        let mut expected = entries.clone();
        expected.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
        assert_eq!(back, expected);
    }

    #[test]
    fn negative_and_extreme_times_roundtrip() {
        let entries = vec![
            entry(0, 0, 0, 0, i64::MIN + 1, i64::MIN + 10, 64, 8, 16, 1, 1, None),
            entry(0, 0, 0, 0, -5000, -1000, 72, 8, 16, 1, 2, Some(1)),
            entry(0, 0, 0, 0, 0, 0, 80, 8, 16, 1, 3, None),
            entry(0, 0, 0, 0, i64::MAX - 10, i64::MAX, 88, 8, 16, 1, 4, None),
        ];
        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();
        assert_eq!(back, entries);
    }

    #[test]
    fn truncated_buffer_errors() {
        let e = entry(10, 5, 7, 42, 1000, 2000, 64, 128, 256, 3, 7, None);
        let bytes = encode_directory(std::slice::from_ref(&e));
        // Lop off the trailing crc bytes.
        let truncated = &bytes[..bytes.len() - 2];
        assert!(decode_directory(truncated).is_err());
    }

    #[test]
    fn wrong_version_errors() {
        let mut bytes = encode_directory(&[]);
        bytes[0] = 99;
        assert!(decode_directory(&bytes).is_err());
    }

    /// A foreign / corrupt directory whose zoom delta overflows u8 must error
    /// rather than silently truncate via `as u8`.
    #[test]
    fn decode_rejects_out_of_range_columns() {
        let mut buf = Vec::new();
        buf.push(DIRECTORY_VERSION);
        put_uvarint(&mut buf, 1); // N
        put_uvarint(&mut buf, 1); // R
        put_ivarint(&mut buf, 300); // Δzoom (out of u8 range)
        put_ivarint(&mut buf, 0); // Δhilbert
        put_ivarint(&mut buf, 0); // Δx
        put_ivarint(&mut buf, 0); // Δy
        put_ivarint(&mut buf, 0); // Δtime_start
        put_ivarint(&mut buf, 0); // duration
        put_uvarint(&mut buf, 0); // feature_count
        put_uvarint(&mut buf, 0); // bucket present = 0
        put_uvarint(&mut buf, 1); // run_len
        put_uvarint(&mut buf, 0); // offset flag (contiguous)
        put_uvarint(&mut buf, 0); // length
        put_uvarint(&mut buf, 0); // uncompressed
        buf.extend_from_slice(&0u32.to_le_bytes()); // crc
        assert!(decode_directory(&buf).is_err());
    }

    /// Boundary: u64::MAX must round-trip for both the blob offset and the
    /// temporal bucket — neither sentinel may collide with a real value.
    #[test]
    fn u64_max_offset_and_bucket_roundtrip() {
        let entries = vec![
            entry(5, 1, 1, 10, 0, 100, u64::MAX, 8, 16, 1, 7, Some(u64::MAX)),
            entry(5, 2, 2, 11, 0, 100, 64, 8, 16, 1, 8, Some(3_600_000)),
            entry(5, 3, 3, 12, 0, 100, 0, 8, 16, 1, 9, None),
        ];
        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();
        assert_eq!(back, entries);
    }
}
