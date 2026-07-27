//! The STT tile directory at directory-codec version 5 (`DIRECTORY_VERSION`) —
//! a compact, range-request-friendly tile index. Every bare vN below names a
//! version on THAT axis, not the manifest's `formatVersion`.
//!
//! A columnar binary encoding (directory versions 2 and 3 were an Arrow-IPC
//! index: fixed-width columns + IPC framing), inspired by PMTiles v3:
//!
//! - **Columnar + delta + zig-zag varints.** Entries are sorted by
//!   `(zoom, hilbert, time_start)`, so each column (zoom, hilbert, x, y,
//!   time_start) is near-monotonic and delta-codes to ~1 byte per entry.
//! - **Blob-run RLE.** Consecutive entries that point at the *same physical
//!   blob* (a spatial cell whose content is identical across consecutive time
//!   buckets — the temporal analogue of PMTiles' ocean tiles) collapse into one
//!   run. The heavy per-blob columns (offset/length/uncompressed/crc) are then
//!   stored once per *run* instead of once per *entry*.
//! - **Per-run pack id (v5).** Each run carries a `pack_id` (which packed-format
//!   object holds the blob), delta+zig-zag coded against the previous run's
//!   pack id — packs are near-monotonic in directory order, so ~1 byte/run. A
//!   single-file archive has `pack_id == 0` on every run.
//! - **Pack-relative offset contiguity sentinel.** A run whose blob immediately
//!   follows the previous run's blob *in the same pack* stores offset `0`;
//!   otherwise a `1` flag + the raw offset. The contiguity expectation resets to
//!   `0` whenever the pack id changes between consecutive runs, so the first run
//!   of every pack still hits the cheap `0` sentinel. Sequential archives (the
//!   common case) cost ~1 byte for the whole offset column.
//!
//! The directory is self-describing (leading version byte + entry/run counts)
//! and decodes to exactly the `TileEntry` list that was encoded, provided the
//! input was already (or is internally re-)sorted into directory order.
//!
//! This module is pure (no I/O): `encode_directory` / `decode_directory` map
//! `&[TileEntry] ⇆ Vec<u8>`. The archive writer/reader own where the buffer
//! lives in the file (single-file v4) or object (packed `index/<hash>.sttd`).
//!
//! ## v5 wire format (per-run columns, in order)
//!
//! Each run, in directory order, writes:
//! 1. `run_len`         — uvarint
//! 2. `Δpack_id`        — ivarint (zig-zag of `pack_id - prev_pack_id`)
//! 3. offset sentinel   — uvarint `0` (== `expected_offset`) **or** `1` then a
//!    uvarint raw offset. `expected_offset` is reset to `0` before this run when
//!    `pack_id != prev_pack_id`.
//! 4. `length`          — uvarint
//! 5. `uncompressed`    — uvarint
//! 6. `crc32c`          — 4 raw little-endian bytes
//!
//! The `Δpack_id` column is **new in v5** and sits immediately after `run_len`,
//! before the offset sentinel. Per-entry key columns and the trailing
//! `COVER_SECTION_TMIN` section are byte-identical to v4.

use crate::error::{Error, Result};
use crate::tile::TileId;

/// One directory entry: where a tile lives and what it covers.
#[derive(Debug, Clone, PartialEq)]
pub struct TileEntry {
    /// Zoom level.
    pub zoom: u8,
    /// Tile X coordinate.
    pub x: u32,
    /// Tile Y coordinate.
    pub y: u32,
    /// Inclusive temporal start (Unix ms).
    pub time_start: i64,
    /// Inclusive temporal end (Unix ms).
    pub time_end: i64,
    /// Index of the pack object holding this tile's blob (packed format).
    ///
    /// This selects `manifest.packs[pack_id]` and [`offset`](Self::offset) is
    /// **pack-relative**. A per-blob (per-run) property, not a per-entry key —
    /// entries RLE-collapse only within one pack.
    pub pack_id: u32,
    /// Byte offset of the compressed blob, relative to `packs[pack_id]`.
    pub offset: u64,
    /// Compressed blob length in bytes.
    pub length: u32,
    /// Uncompressed payload length in bytes.
    pub uncompressed_size: u32,
    /// Total feature count across the tile's layers.
    pub feature_count: u32,
    /// Hilbert index of `(zoom, x, y)` — directory sort key.
    pub hilbert: u64,
    /// CRC32C integrity tag of the compressed blob.
    pub crc32c: u32,
    /// Temporal bucket size in milliseconds this tile occupies.
    ///
    /// Base tiles record the archive's base bucket size; temporal-LOD tiles
    /// record their coarser bucket. `None` when the archive carries no LOD
    /// pyramid (readers fall back to `Metadata::temporal_bucket_ms`).
    pub temporal_bucket_ms: Option<u64>,
    /// **Tight lower covering bound** — the minimum feature *start* time actually
    /// present in the tile. `time_start` is the addressable *bucket boundary*
    /// (kept loose so `(z,x,y,bucket)` lookup works); `cover_t_min` is the real
    /// earliest data, which a client uses to prune a tile whose features all lie
    /// after a query window (`time_end` already gives the tight upper bound).
    /// `None` when not computed (pre-covering builds), in which case clients
    /// fall back to `time_start`.
    pub cover_t_min: Option<i64>,
}

impl TileEntry {
    /// The tile's identity.
    pub fn tile_id(&self) -> TileId {
        TileId::new(self.zoom, self.x, self.y, self.time_start.max(0) as u64)
    }
}

/// Version of the DIRECTORY-CODEC axis, and the first byte of every directory
/// buffer. Counted independently of the manifest's
/// [`PACKED_FORMAT_VERSION`](crate::pack::PACKED_FORMAT_VERSION) and of the
/// layer-frame version, so the directory codec can evolve without churning
/// either.
///
/// v5 adds the per-run `pack_id` column and makes the offset contiguity sentinel
/// pack-relative (reset on every pack change). See the module docs.
pub const DIRECTORY_VERSION: u8 = 5;

/// Tag for the optional trailing **covering** section: one signed varint per
/// entry (in directory order) giving `cover_t_min - time_start`, the tight
/// lower temporal bound. Backward-compatible — a pre-covering archive's buffer
/// simply ends after the per-run blob columns, and the decoder leaves
/// `cover_t_min = None`. Forward-compatible — a decoder that doesn't recognise a
/// trailing tag stops reading it. See [`TileEntry::cover_t_min`].
const COVER_SECTION_TMIN: u8 = 1;

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
        if shift == 63 {
            // 10th continuation byte: only bit 0 fits into a u64. A value with
            // any higher bit set (or a further continuation) is an over-long /
            // oversized varint — reject it instead of silently truncating.
            if byte & 0x80 != 0 || byte > 1 {
                return Err(Error::InvalidArchive(
                    "directory: varint exceeds 64 bits".into(),
                ));
            }
        }
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::InvalidArchive(
                "directory: varint exceeds 64 bits".into(),
            ));
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

/// Encode tile entries into the v5 directory buffer.
///
/// Entries are sorted into directory order `(zoom, hilbert, time_start)` first,
/// so the caller need not pre-sort. Two entries are considered to share a blob
/// (and so RLE-collapse) when their `(pack_id, offset, length,
/// uncompressed_size, crc32c)` all match — which is exactly what the
/// dedup-on-write path produces for byte-identical tiles within one pack. The
/// `pack_id` is part of the run identity (v5): two entries collapse only when
/// they live in the same pack *and* point at the same blob.
pub fn encode_directory(entries: &[TileEntry]) -> Vec<u8> {
    let mut sorted: Vec<&TileEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
    let n = sorted.len();

    // Compute blob runs up front so we can write the run count into the header.
    // A run is a maximal stretch of consecutive entries pointing at one blob in
    // one pack (pack_id is part of the run identity in v5).
    let mut runs: Vec<(usize, u32, u64, u32, u32, u32)> = Vec::new();
    let mut i = 0;
    while i < n {
        let head = sorted[i];
        let crc = head.crc32c;
        let mut j = i + 1;
        while j < n {
            let e = sorted[j];
            if e.pack_id == head.pack_id
                && e.offset == head.offset
                && e.length == head.length
                && e.uncompressed_size == head.uncompressed_size
                && e.crc32c == crc
            {
                j += 1;
            } else {
                break;
            }
        }
        runs.push((
            j - i,
            head.pack_id,
            head.offset,
            head.length,
            head.uncompressed_size,
            crc,
        ));
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

    // Per-run blob columns: run_len, Δpack_id, offset (pack-relative
    // contiguity), length, uncompressed, crc. The pack_id is delta+zig-zag coded
    // against the previous run (packs are near-monotonic → ~1 byte/run), and the
    // offset contiguity expectation resets to 0 whenever the pack changes so the
    // first run of each pack hits the cheap `0` sentinel.
    let mut expected_offset = 0u64;
    let mut prev_pack_id = 0i64;
    for (run_len, pack_id, offset, length, uncompressed, crc) in &runs {
        put_uvarint(&mut buf, *run_len as u64);
        // Δpack_id (zig-zag). When the pack changes, reset the offset contiguity
        // expectation so this run's first blob is "contiguous from 0".
        let pid = *pack_id as i64;
        if pid != prev_pack_id {
            expected_offset = 0;
        }
        put_ivarint(&mut buf, pid.wrapping_sub(prev_pack_id));
        prev_pack_id = pid;
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

    // Optional trailing covering section. Emitted only when EVERY entry carries
    // a tight lower bound, so it's exactly N signed varints indexable 1:1 with
    // the key rows. A mixed or all-`None` corpus writes nothing (the common case
    // for repacked/transcoded archives), keeping the buffer byte-identical to a
    // pre-covering directory.
    if n > 0 && sorted.iter().all(|e| e.cover_t_min.is_some()) {
        buf.push(COVER_SECTION_TMIN);
        for e in &sorted {
            // cover_t_min - time_start; signed because a feature can start
            // before its bucket boundary. Small magnitude → ~1-2 bytes.
            let delta = e.cover_t_min.unwrap().wrapping_sub(e.time_start);
            put_ivarint(&mut buf, delta);
        }
    }

    buf
}

// ----------------------------------------------------------------------------
// Decode
// ----------------------------------------------------------------------------

/// Lowest directory version this decoder accepts. v4 (single-file archives) has
/// no per-run `pack_id` column and whole-file offsets — decoded as `pack_id = 0`
/// with no pack-relative reset. v5 adds the `pack_id` column. The encoder always
/// writes [`DIRECTORY_VERSION`] (v5); v4 read support keeps the v4 single-file
/// `ArchiveReader` working as the transcode input.
const MIN_DIRECTORY_VERSION: u8 = 4;

/// Decode a v4/v5 directory buffer back into tile entries (in directory order).
///
/// Both versions share the per-entry key columns and the trailing cover section.
/// v5 prepends a `Δpack_id` (zig-zag) varint to each run's columns and resets
/// the offset contiguity expectation on every pack change; v4 omits the column
/// and never resets (one implicit pack, `pack_id = 0`).
pub fn decode_directory(bytes: &[u8]) -> Result<Vec<TileEntry>> {
    let mut pos = 0usize;
    let version = *bytes
        .first()
        .ok_or_else(|| Error::InvalidArchive("directory: empty buffer".into()))?;
    pos += 1;
    if !(MIN_DIRECTORY_VERSION..=DIRECTORY_VERSION).contains(&version) {
        return Err(Error::InvalidArchive(format!(
            "directory: unsupported version {version} (expected {MIN_DIRECTORY_VERSION}..={DIRECTORY_VERSION})"
        )));
    }
    let has_pack_id = version >= 5;
    let n = get_uvarint(bytes, &mut pos)? as usize;
    let run_count = get_uvarint(bytes, &mut pos)? as usize;
    // Adversarial-input guard: every entry costs ≥ 8 wire bytes (8 one-byte-
    // minimum varints) and every run ≥ 8 (v4) / 9 (v5, with the Δpack_id
    // column), and entries + runs share this one buffer — so
    // 8·(n + run_count) ≤ bytes.len() must hold for any decodable input
    // (/8 blanket keeps degenerate zero-length-run v4 buffers decodable).
    // Rejecting BEFORE the `with_capacity` calls below keeps a doctored
    // header from forcing a huge allocation: the scratch `Key` is 56 B/entry,
    // so the un-divided `n > bytes.len()` form of this guard still allowed a
    // 56× amplification (a 100 MB object claiming n = 100 M entries forced a
    // 5.6 GB allocation). Post-division the worst case is 56·(len/8) = 7×
    // the buffer — the same as a maximally compact legitimate directory,
    // i.e. tight without parsing (guarded by tests/adversarial_decode.rs).
    if n.saturating_add(run_count) > bytes.len() / 8 {
        return Err(Error::InvalidArchive(format!(
            "directory: header claims {n} entries + {run_count} runs, more than the \
             {}-byte buffer can hold at ≥8 bytes per record",
            bytes.len()
        )));
    }

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

    // Expand runs over the keys, assigning each run's shared blob fields. The
    // pack_id is delta+zig-zag against the previous run, and the offset
    // contiguity expectation resets to 0 on every pack change — symmetric with
    // the encoder.
    let mut entries = Vec::with_capacity(n);
    let mut cursor = 0usize;
    let mut expected_offset = 0u64;
    let mut prev_pack_id = 0i64;
    for _ in 0..run_count {
        let run_len = get_uvarint(bytes, &mut pos)? as usize;
        // v5 carries a Δpack_id column (zig-zag) and resets the offset
        // contiguity expectation on every pack change. v4 has neither: one
        // implicit pack (pack_id = 0), whole-file-contiguous offsets.
        let pid = if has_pack_id {
            prev_pack_id.wrapping_add(get_ivarint(bytes, &mut pos)?)
        } else {
            0
        };
        if pid != prev_pack_id {
            expected_offset = 0;
        }
        prev_pack_id = pid;
        if !(0..=u32::MAX as i64).contains(&pid) {
            return Err(Error::InvalidArchive(format!(
                "directory: pack_id {pid} out of u32 range"
            )));
        }
        let pack_id = pid as u32;
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

        // Phrased as a subtraction (cursor ≤ n is a loop invariant) so a
        // doctored run_len near u64::MAX can't overflow `cursor + run_len`.
        if run_len > n - cursor {
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
                pack_id,
                offset,
                length,
                uncompressed_size,
                feature_count: k.feature_count,
                hilbert: k.hilbert,
                crc32c: crc,
                temporal_bucket_ms: k.temporal_bucket_ms,
                cover_t_min: None,
            });
        }
        expected_offset = offset.wrapping_add(length as u64);
    }

    if cursor != n {
        return Err(Error::InvalidArchive(format!(
            "directory: runs covered {cursor} entries, expected {n}"
        )));
    }

    // Optional trailing covering section(s). A pre-covering archive's buffer
    // ends here; if bytes remain, read tagged sections. Unknown tags stop the
    // scan (forward-compat) rather than erroring.
    if pos < bytes.len() {
        let tag = bytes[pos];
        pos += 1;
        if tag == COVER_SECTION_TMIN {
            for e in entries.iter_mut() {
                let delta = get_ivarint(bytes, &mut pos)?;
                e.cover_t_min = Some(e.time_start.wrapping_add(delta));
            }
        }
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
            pack_id: 0,
            offset,
            length,
            uncompressed_size: unc,
            feature_count: fc,
            hilbert,
            crc32c: crc,
            temporal_bucket_ms: tb,
            cover_t_min: None,
        }
    }

    /// Build an entry with an explicit `pack_id` (v5 packed format).
    #[allow(clippy::too_many_arguments)]
    fn entry_pack(
        pack_id: u32,
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
        let mut e = entry(
            zoom, x, y, hilbert, ts, te, offset, length, unc, fc, crc, tb,
        );
        e.pack_id = pack_id;
        e
    }

    #[test]
    fn empty_roundtrips() {
        let bytes = encode_directory(&[]);
        let back = decode_directory(&bytes).unwrap();
        assert!(back.is_empty());
    }

    /// The uvarint decoder must round-trip every representative value — crucially
    /// including `u64::MAX`, whose 10th byte is `0x01` — and must REJECT an
    /// over-long / oversized encoding (a 10th byte with bits above bit 0, or a
    /// further continuation) instead of silently truncating the top bits.
    #[test]
    fn uvarint_rejects_oversized_encoding() {
        // Valid round-trips, including the max value (10-byte encoding).
        for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            let mut buf = Vec::new();
            put_uvarint(&mut buf, v);
            let mut pos = 0;
            assert_eq!(get_uvarint(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
        }
        // 10th byte carries bits above bit 0 (0x02): pre-fix this silently
        // truncated to 0; now it must error.
        let oversized = [0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02];
        let mut pos = 0;
        assert!(get_uvarint(&oversized, &mut pos).is_err());
        // 10th byte still sets the continuation bit (over-long): must error.
        let overlong = [
            0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x81, 0x00,
        ];
        let mut pos = 0;
        assert!(get_uvarint(&overlong, &mut pos).is_err());
    }

    /// The optional covering section round-trips `cover_t_min` exactly (incl. a
    /// value BELOW `time_start` — a feature starting before its bucket edge).
    #[test]
    fn cover_t_min_section_roundtrips() {
        let mut entries = Vec::new();
        for i in 0..20u32 {
            let mut e = entry(
                10,
                i,
                i,
                i as u64,
                (i as i64) * 1000,
                (i as i64) * 1000 + 900,
                64 + i as u64 * 50,
                50,
                100,
                i,
                0x100 + i,
                Some(1000),
            );
            // tight lower bound: usually inside the bucket, but entry 0 starts
            // 500ms BEFORE its bucket boundary to exercise the signed delta.
            e.cover_t_min = Some(if i == 0 {
                -500
            } else {
                (i as i64) * 1000 + 250
            });
            entries.push(e);
        }
        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();
        let mut expected = entries.clone();
        expected.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
        assert_eq!(back, expected);
        assert!(back.iter().all(|e| e.cover_t_min.is_some()));
    }

    /// A directory with NO covering bounds must produce a buffer byte-identical
    /// to the pre-covering codec (no trailing section), and decode to `None`.
    #[test]
    fn absent_cover_section_is_backward_compatible() {
        let e = entry(10, 5, 7, 42, 1000, 2000, 64, 128, 256, 3, 7, None);
        let bytes = encode_directory(std::slice::from_ref(&e));
        // No trailing tag byte: the last byte is the run's crc (4 LE bytes),
        // never a lone COVER_SECTION_TMIN tag.
        let back = decode_directory(&bytes).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].cover_t_min, None);
    }

    #[test]
    fn single_entry_roundtrips() {
        let e = entry(
            10,
            5,
            7,
            42,
            1000,
            2000,
            64,
            128,
            256,
            3,
            0xDEAD_BEEF,
            Some(3_600_000),
        );
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
                4096, // same offset (deduped blob)
                512,  // same length
                1024, // same uncompressed
                64,
                crc, // same crc
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
            entry(
                0,
                0,
                0,
                0,
                i64::MIN + 1,
                i64::MIN + 10,
                64,
                8,
                16,
                1,
                1,
                None,
            ),
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
        put_ivarint(&mut buf, 0); // Δpack_id (v5)
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

    /// v5: entries spread across multiple packs round-trip with the correct
    /// `pack_id`, and each pack's offsets are pack-relative (every pack starts
    /// from a small offset, riding the contiguity sentinel reset).
    #[test]
    fn multi_pack_offsets_are_pack_relative_and_roundtrip() {
        let mut entries = Vec::new();
        // Three packs, each with a fresh offset run starting at 0. Distinct
        // blobs within a pack are contiguous (offset += length).
        for pack_id in 0..3u32 {
            let mut off = 0u64;
            for i in 0..6u32 {
                let hil = (pack_id as u64) * 100 + i as u64; // monotone in dir order
                let len = 40 + i;
                entries.push(entry_pack(
                    pack_id,
                    9,
                    pack_id * 10 + i,
                    i,
                    hil,
                    (i as i64) * 1000,
                    (i as i64) * 1000 + 500,
                    off,
                    len,
                    len * 2,
                    i,
                    0x2000 + pack_id * 100 + i,
                    Some(3_600_000),
                ));
                off += len as u64;
            }
        }
        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();
        let mut expected = entries.clone();
        expected.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
        assert_eq!(back, expected);
        // Every pack contributes entries, and the first run of each pack has
        // offset 0 (pack-relative reset). pack_ids observed are exactly {0,1,2}.
        let packs: std::collections::BTreeSet<u32> = back.iter().map(|e| e.pack_id).collect();
        assert_eq!(packs, [0u32, 1, 2].into_iter().collect());
        for p in 0..3u32 {
            assert!(back.iter().any(|e| e.pack_id == p && e.offset == 0));
        }
    }

    /// v5: RLE within a pack still collapses a static cell across time, but two
    /// blobs that are byte-identical in *different* packs must NOT collapse
    /// (pack_id is part of the run identity), and both decode to their own pack.
    #[test]
    fn rle_collapses_within_pack_but_not_across_packs() {
        let crc = 0x55AA_55AAu32;
        let mut entries = Vec::new();
        // Pack 0: one static cell across 50 hourly buckets — one run.
        for b in 0..50u64 {
            entries.push(entry_pack(
                0,
                9,
                3,
                4,
                77,
                (b as i64) * 3_600_000,
                (b as i64) * 3_600_000 + 3_599_999,
                4096,
                512,
                1024,
                64,
                crc,
                Some(3_600_000),
            ));
        }
        // Pack 1: same blob fields (offset/len/unc/crc) but a different pack —
        // a separate physical object, so it must stay its own run.
        for b in 0..50u64 {
            entries.push(entry_pack(
                1,
                9,
                5,
                6,
                99,
                (b as i64) * 3_600_000,
                (b as i64) * 3_600_000 + 3_599_999,
                4096,
                512,
                1024,
                64,
                crc,
                Some(3_600_000),
            ));
        }
        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();
        let mut expected = entries.clone();
        expected.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
        assert_eq!(back, expected);
        assert!(back.iter().any(|e| e.pack_id == 0));
        assert!(back.iter().any(|e| e.pack_id == 1));
    }

    /// v5: the cover section still round-trips alongside multi-pack pack_ids.
    #[test]
    fn cover_section_roundtrips_with_pack_ids() {
        let mut entries = Vec::new();
        for i in 0..12u32 {
            let pack_id = i / 4; // packs 0,1,2
            let mut e = entry_pack(
                pack_id,
                10,
                i,
                i,
                i as u64,
                (i as i64) * 1000,
                (i as i64) * 1000 + 900,
                (i % 4) as u64 * 50,
                50,
                100,
                i,
                0x300 + i,
                Some(1000),
            );
            e.cover_t_min = Some((i as i64) * 1000 + 250);
            entries.push(e);
        }
        let bytes = encode_directory(&entries);
        let back = decode_directory(&bytes).unwrap();
        let mut expected = entries.clone();
        expected.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
        assert_eq!(back, expected);
        assert!(back.iter().all(|e| e.cover_t_min.is_some()));
        let packs: std::collections::BTreeSet<u32> = back.iter().map(|e| e.pack_id).collect();
        assert_eq!(packs, [0u32, 1, 2].into_iter().collect());
    }
}
