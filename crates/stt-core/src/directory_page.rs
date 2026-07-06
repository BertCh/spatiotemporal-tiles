//! Paged-directory container (Wave 2 — COPC/GeoParquet-1.1 steal).
//!
//! The v5 directory ([`crate::directory`]) is a single whole-load blob: a cold
//! reader must fetch and decode *every* entry before it can request one tile.
//! This module wraps that codec in a **paged container** so a reader fetches a
//! tiny root page plus only the leaf pages its viewport/time-window actually
//! touches — cold start proportional to the query footprint, not dataset size.
//!
//! ## Container layout
//!
//! ```text
//! .sttd  =  [root page frame][leaf 0 frame][leaf 1 frame] ...
//! ```
//!
//! - A **leaf page** is the *unchanged* v5 codec ([`encode_directory`]) over a
//!   contiguous slice of directory order `(zoom, hilbert, time_start)`. Slicing
//!   resets delta state and splits RLE runs at boundaries — the only cost of
//!   paging (sim-measured +6–19% at-rest, paid once by the immutable CDN object,
//!   not per session). Reusing the proven codec for leaves means the only new
//!   bytes are the root + framing.
//! - The **root page** is a fixed-width table of [`PageDescriptor`]s — one per
//!   leaf — carrying each leaf's byte range and its pruning bounds: a
//!   **geographic bbox** (lon/lat × 1e7), a **zoom range**, and a **temporal
//!   `[t_min, t_max]`** (with `t_min` taken from `cover_t_min` when present).
//!   A reader prunes whole leaves by `zoom ∈ [min,max]` ∧ bbox∩viewport ∧
//!   `[t_min,t_max]`∩window — *without fetching them*. (The geo-bbox descriptor
//!   was frozen over the Hilbert-key-range alternative by the step-0 A/B sim;
//!   it is zoom-correct and needs no Hilbert port in the TS reader.)
//!
//! Each page (root and leaves) is an **independent frame** so it is fetchable in
//! isolation: when `zstd` framing is on (the writer default) every frame is its
//! own zstd frame; raw framing stores the codec bytes directly. There is no
//! shared dictionary, so the fzstd TS path decodes every page.
//!
//! Offsets in the root are **relative to the end of the root page**
//! (`absolute = root_length + rel_offset`), so the root can be encoded without
//! knowing its own at-rest length first.
//!
//! This module is pure (no I/O). [`encode_paged_directory`] maps
//! `&[TileEntry] → bytes + root_length`; [`decode_paged_directory`] is the
//! load-all inverse (root + every leaf) used by the local Rust reader and the
//! round-trip tests. The HTTP reader (TS) decodes the root and fetches leaves on
//! demand — that is where the cold-start win is realized.

use crate::directory::TileEntry;
use crate::compression;
use crate::directory::{decode_directory, encode_directory};
use crate::error::{Error, Result};
use crate::projection::tile_geo_bounds;

/// Root container version (first byte of the decoded root page). Bumped
/// independently of the leaf codec's [`crate::directory::DIRECTORY_VERSION`].
pub const PAGED_ROOT_VERSION: u8 = 1;

/// Descriptor-kind tag: geographic bbox + zoom range + temporal bounds.
pub const DESCRIPTOR_GEO_BBOX: u8 = 0;

/// Default entries per leaf page — the 1024–4096 sweet spot (sim-validated:
/// ~60–130 KB zstd pages, the range coalescer's comfort zone).
pub const DEFAULT_PAGE_ENTRIES: usize = 4096;

/// Fixed-width root header (bytes): version, kind, reserved u16, page_count u32,
/// page_entries u32.
const ROOT_HEADER_LEN: usize = 12;
/// Fixed-width per-page descriptor (bytes). See [`PageDescriptor`] field order.
const DESCRIPTOR_LEN: usize = 52;

/// One leaf page's pruning descriptor, stored fixed-width in the root page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageDescriptor {
    /// Byte offset of the leaf frame **relative to the end of the root page**
    /// (i.e. relative to `root_length`). Absolute = `root_length + rel_offset`.
    /// Relative so the root encodes without knowing its own at-rest length.
    pub rel_offset: u64,
    /// At-rest (framed) byte length of the leaf.
    pub length: u32,
    /// Entries in this leaf (Σ over pages == N).
    pub entry_count: u32,
    /// Inclusive zoom range of the leaf's entries.
    pub min_zoom: u8,
    pub max_zoom: u8,
    /// Geographic bbox of the leaf's tiles, lon/lat × 1e7. Mins are floored and
    /// maxes ceiled so the integer bbox never under-covers the true bbox (no
    /// false-negative page prune from rounding).
    pub min_lon_e7: i32,
    pub min_lat_e7: i32,
    pub max_lon_e7: i32,
    pub max_lat_e7: i32,
    /// Subtree temporal bounds: `min(cover_t_min ?? time_start)` ..
    /// `max(time_end)` — the COPC-temporal page-pointer prune.
    pub t_min: i64,
    pub t_max: i64,
}

impl PageDescriptor {
    /// Does this leaf overlap a viewport query? `zoom` membership ∧ geo-bbox
    /// overlap ∧ temporal overlap. Query bbox is lon/lat × 1e7 (same fixed
    /// point as the descriptor). Mirrors the TS reader's page selection exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn overlaps(
        &self,
        zoom: u8,
        q_min_lon_e7: i32,
        q_min_lat_e7: i32,
        q_max_lon_e7: i32,
        q_max_lat_e7: i32,
        t_start: i64,
        t_end: i64,
    ) -> bool {
        zoom >= self.min_zoom
            && zoom <= self.max_zoom
            && self.min_lon_e7 <= q_max_lon_e7
            && q_min_lon_e7 <= self.max_lon_e7
            && self.min_lat_e7 <= q_max_lat_e7
            && q_min_lat_e7 <= self.max_lat_e7
            && self.t_max >= t_start
            && self.t_min <= t_end
    }
}

/// The decoded root page: the descriptor table plus its header fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedRoot {
    pub descriptor_kind: u8,
    /// Nominal entries-per-page used at build (informational).
    pub page_entries: u32,
    pub pages: Vec<PageDescriptor>,
}

/// Output of [`encode_paged_directory`]: the whole `.sttd` bytes plus the root
/// length (→ manifest `directory.rootLength`) and page bookkeeping.
pub struct EncodedPagedDirectory {
    /// `[root frame][leaf 0 frame] ...` — the content-addressed `.sttd` bytes.
    pub bytes: Vec<u8>,
    /// At-rest length of the root frame (manifest `directory.rootLength`).
    pub root_length: u64,
    pub page_count: usize,
    pub page_entries: usize,
}

#[inline]
fn floor_e7(deg: f64) -> i32 {
    (deg * 1e7).floor().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

#[inline]
fn ceil_e7(deg: f64) -> i32 {
    (deg * 1e7).ceil().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Encode the root page (descriptor table) to its raw (pre-framing) bytes.
pub fn encode_root(page_entries: u32, descriptors: &[PageDescriptor]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ROOT_HEADER_LEN + descriptors.len() * DESCRIPTOR_LEN);
    buf.push(PAGED_ROOT_VERSION);
    buf.push(DESCRIPTOR_GEO_BBOX);
    put_u16(&mut buf, 0); // reserved
    put_u32(&mut buf, descriptors.len() as u32);
    put_u32(&mut buf, page_entries);
    for d in descriptors {
        put_u64(&mut buf, d.rel_offset);
        put_u32(&mut buf, d.length);
        put_u32(&mut buf, d.entry_count);
        buf.push(d.min_zoom);
        buf.push(d.max_zoom);
        put_u16(&mut buf, 0); // reserved
        put_i32(&mut buf, d.min_lon_e7);
        put_i32(&mut buf, d.min_lat_e7);
        put_i32(&mut buf, d.max_lon_e7);
        put_i32(&mut buf, d.max_lat_e7);
        put_i64(&mut buf, d.t_min);
        put_i64(&mut buf, d.t_max);
    }
    buf
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos + n;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::InvalidArchive("paged root: truncated".into()))?;
        self.pos = end;
        Ok(slice)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// Decode the root page from its raw (already-unframed) bytes.
pub fn decode_root(bytes: &[u8]) -> Result<PagedRoot> {
    let mut r = Reader { bytes, pos: 0 };
    let version = *bytes
        .first()
        .ok_or_else(|| Error::InvalidArchive("paged root: empty".into()))?;
    r.pos = 1;
    if version != PAGED_ROOT_VERSION {
        return Err(Error::InvalidArchive(format!(
            "paged root: unsupported version {version} (expected {PAGED_ROOT_VERSION})"
        )));
    }
    let descriptor_kind = *bytes.get(1).unwrap();
    r.pos = 2;
    if descriptor_kind != DESCRIPTOR_GEO_BBOX {
        return Err(Error::InvalidArchive(format!(
            "paged root: unsupported descriptor kind {descriptor_kind}"
        )));
    }
    let _reserved = r.u16()?;
    let page_count = r.u32()? as usize;
    let page_entries = r.u32()?;
    // Adversarial-input guard: descriptors are fixed-width, so a count beyond
    // what the remaining bytes can hold is corrupt; reject before allocating
    // the table (guarded by tests/adversarial_decode.rs).
    if bytes.len().saturating_sub(r.pos) < page_count.saturating_mul(DESCRIPTOR_LEN) {
        return Err(Error::InvalidArchive(format!(
            "paged root: header claims {page_count} pages but only {} bytes remain",
            bytes.len().saturating_sub(r.pos)
        )));
    }
    let mut pages = Vec::with_capacity(page_count);
    for _ in 0..page_count {
        let rel_offset = r.u64()?;
        let length = r.u32()?;
        let entry_count = r.u32()?;
        let min_zoom = r.take(1)?[0];
        let max_zoom = r.take(1)?[0];
        let _res = r.u16()?;
        let min_lon_e7 = r.i32()?;
        let min_lat_e7 = r.i32()?;
        let max_lon_e7 = r.i32()?;
        let max_lat_e7 = r.i32()?;
        let t_min = r.i64()?;
        let t_max = r.i64()?;
        pages.push(PageDescriptor {
            rel_offset,
            length,
            entry_count,
            min_zoom,
            max_zoom,
            min_lon_e7,
            min_lat_e7,
            max_lon_e7,
            max_lat_e7,
            t_min,
            t_max,
        });
    }
    Ok(PagedRoot {
        descriptor_kind,
        page_entries,
        pages,
    })
}

fn frame(raw: Vec<u8>, zstd: bool, level: i32) -> Result<Vec<u8>> {
    if zstd {
        compression::compress_zstd_with_dict_level(&raw, None, level)
    } else {
        Ok(raw)
    }
}

fn unframe(frame: &[u8], zstd: bool) -> Result<Vec<u8>> {
    if zstd {
        compression::decompress_zstd_with_dict(frame, None)
    } else {
        Ok(frame.to_vec())
    }
}

/// Encode tile entries into the paged container.
///
/// `entries` need not be pre-sorted; they are sorted into directory order
/// `(zoom, hilbert, time_start, temporal_bucket_ms)` — matching the writer and
/// the v5 codec — then sliced into pages of `page_entries`. `zstd` selects
/// per-page zstd framing (the writer default) vs raw codec bytes.
///
/// The covering section is decided **globally**: if any entry lacks
/// `cover_t_min`, it is stripped from every leaf, so a paged directory decodes
/// byte-for-identical-entries to a whole-load v5 directory of the same corpus
/// (the per-page codec would otherwise emit a cover section for an all-`Some`
/// page even when the corpus is mixed).
pub fn encode_paged_directory(
    entries: &[TileEntry],
    page_entries: usize,
    zstd: bool,
) -> Result<EncodedPagedDirectory> {
    encode_paged_directory_level(entries, page_entries, zstd, compression::ZSTD_LEVEL)
}

/// [`encode_paged_directory`] at an explicit zstd `level` (ignored when
/// `zstd` is false). The directory sits on the cold-start critical path, so a
/// publish build can spend a higher level here too — decode is level-independent.
pub fn encode_paged_directory_level(
    entries: &[TileEntry],
    page_entries: usize,
    zstd: bool,
    level: i32,
) -> Result<EncodedPagedDirectory> {
    let page_entries = page_entries.max(1);
    let mut sorted: Vec<&TileEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start, e.temporal_bucket_ms));

    // Global cover decision (mirror the whole-load codec's all-or-nothing rule).
    let all_cover = !sorted.is_empty() && sorted.iter().all(|e| e.cover_t_min.is_some());

    let mut descriptors: Vec<PageDescriptor> = Vec::new();
    let mut leaf_frames: Vec<Vec<u8>> = Vec::new();
    let mut rel_offset = 0u64;

    for chunk in sorted.chunks(page_entries) {
        // Materialize the slice for encode_directory, applying the global cover
        // decision so per-page emission matches whole-load.
        let owned: Vec<TileEntry> = chunk
            .iter()
            .map(|&e| {
                let mut c = e.clone();
                if !all_cover {
                    c.cover_t_min = None;
                }
                c
            })
            .collect();
        let raw = encode_directory(&owned);
        let leaf = frame(raw, zstd, level)?;

        let mut geo = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        let (mut zmin, mut zmax) = (u8::MAX, 0u8);
        let (mut tmin, mut tmax) = (i64::MAX, i64::MIN);
        for &e in chunk {
            let b = tile_geo_bounds(e.zoom, e.x, e.y);
            geo.0 = geo.0.min(b.0);
            geo.1 = geo.1.min(b.1);
            geo.2 = geo.2.max(b.2);
            geo.3 = geo.3.max(b.3);
            zmin = zmin.min(e.zoom);
            zmax = zmax.max(e.zoom);
            tmin = tmin.min(e.cover_t_min.unwrap_or(e.time_start));
            tmax = tmax.max(e.time_end);
        }

        descriptors.push(PageDescriptor {
            rel_offset,
            length: leaf.len() as u32,
            entry_count: chunk.len() as u32,
            min_zoom: zmin,
            max_zoom: zmax,
            min_lon_e7: floor_e7(geo.0),
            min_lat_e7: floor_e7(geo.1),
            max_lon_e7: ceil_e7(geo.2),
            max_lat_e7: ceil_e7(geo.3),
            t_min: tmin,
            t_max: tmax,
        });
        rel_offset += leaf.len() as u64;
        leaf_frames.push(leaf);
    }

    let root_raw = encode_root(page_entries as u32, &descriptors);
    let root_frame = frame(root_raw, zstd, level)?;
    let root_length = root_frame.len() as u64;
    let page_count = descriptors.len();

    let mut bytes = root_frame;
    for f in &leaf_frames {
        bytes.extend_from_slice(f);
    }

    Ok(EncodedPagedDirectory {
        bytes,
        root_length,
        page_count,
        page_entries,
    })
}

/// Decode a whole paged `.sttd` (root + every leaf) back into the full entry
/// list, in directory order — the load-all inverse of [`encode_paged_directory`].
///
/// `root_length` is the manifest's `directory.rootLength`; `zstd` the per-page
/// framing flag (`directory.encoding == "zstd"`). Used by the local Rust reader
/// (no cold-start cost on a mmap'd file) and the round-trip tests. The HTTP
/// reader decodes only the root and fetches leaves on demand instead.
pub fn decode_paged_directory(
    bytes: &[u8],
    root_length: u64,
    zstd: bool,
) -> Result<Vec<TileEntry>> {
    let rl = root_length as usize;
    if rl > bytes.len() {
        return Err(Error::InvalidArchive(format!(
            "paged directory: rootLength {rl} exceeds object size {}",
            bytes.len()
        )));
    }
    let root_raw = unframe(&bytes[..rl], zstd)?;
    let root = decode_root(&root_raw)?;
    let mut entries = Vec::new();
    for d in &root.pages {
        // Checked range arithmetic: a corrupt root's rel_offset/length must
        // error, not overflow (guarded by tests/adversarial_decode.rs).
        let start = root_length.checked_add(d.rel_offset);
        let end = start.and_then(|s| s.checked_add(d.length as u64));
        let frame = match (start, end) {
            (Some(s), Some(e)) if e <= bytes.len() as u64 => &bytes[s as usize..e as usize],
            _ => {
                return Err(Error::InvalidArchive(format!(
                    "paged directory: leaf range rootLength+{}..+{} exceeds object size {}",
                    d.rel_offset,
                    d.length,
                    bytes.len()
                )))
            }
        };
        let raw = unframe(frame, zstd)?;
        let mut page = decode_directory(&raw)?;
        if page.len() != d.entry_count as usize {
            return Err(Error::InvalidArchive(format!(
                "paged directory: leaf declared {} entries, decoded {}",
                d.entry_count,
                page.len()
            )));
        }
        entries.append(&mut page);
    }
    Ok(entries)
}

/// Structural validation of a paged `.sttd`, beyond what plain decode checks.
///
/// `decode_paged_directory` already verifies the root frame, leaf byte-ranges
/// and per-leaf entry counts. This adds the paged-specific invariants a
/// validator wants:
/// - each page descriptor's **bounds cover** every entry in its leaf (geo-bbox,
///   zoom range, temporal `[t_min,t_max]`) — so a reader's prune never drops a
///   matching tile;
/// - **cross-page key order** is monotonic in `(zoom, hilbert, time_start)` — the
///   leaves are contiguous slices of one global sort;
/// - declared vs decoded entry totals agree.
///
/// Returns the list of violations (empty ⇒ clean). Issue output is capped so a
/// badly-corrupt directory can't flood the report.
pub fn verify_paged_structure(bytes: &[u8], root_length: u64, zstd: bool) -> Result<Vec<String>> {
    const MAX_ISSUES: usize = 25;
    let mut issues: Vec<String> = Vec::new();
    let push = |issues: &mut Vec<String>, msg: String| {
        if issues.len() < MAX_ISSUES {
            issues.push(msg);
        } else if issues.len() == MAX_ISSUES {
            issues.push("… (further paged-structure issues truncated)".into());
        }
    };

    let rl = root_length as usize;
    if rl > bytes.len() {
        return Ok(vec![format!(
            "paged: rootLength {rl} exceeds object size {}",
            bytes.len()
        )]);
    }
    let root = decode_root(&unframe(&bytes[..rl], zstd)?)?;

    let mut decoded_total = 0usize;
    let mut declared_total = 0usize;
    let mut prev_last: Option<(u8, u64, i64)> = None;
    for (pi, d) in root.pages.iter().enumerate() {
        declared_total += d.entry_count as usize;
        // Checked range arithmetic, mirroring `decode_paged_directory`: a
        // corrupt descriptor must be reported, not overflow.
        let start = root_length.checked_add(d.rel_offset);
        let end = start.and_then(|s| s.checked_add(d.length as u64));
        let frame = match (start, end) {
            (Some(s), Some(e)) if e <= bytes.len() as u64 => &bytes[s as usize..e as usize],
            _ => {
                push(
                    &mut issues,
                    format!(
                        "page {pi}: leaf range rootLength+{}..+{} exceeds object size {}",
                        d.rel_offset,
                        d.length,
                        bytes.len()
                    ),
                );
                continue;
            }
        };
        let leaf = match unframe(frame, zstd).and_then(|raw| decode_directory(&raw)) {
            Ok(l) => l,
            Err(e) => {
                push(&mut issues, format!("page {pi}: leaf decode failed: {e}"));
                continue;
            }
        };
        if leaf.len() != d.entry_count as usize {
            push(
                &mut issues,
                format!("page {pi}: descriptor declares {} entries, leaf decoded {}", d.entry_count, leaf.len()),
            );
        }
        decoded_total += leaf.len();

        for e in &leaf {
            if e.zoom < d.min_zoom || e.zoom > d.max_zoom {
                push(
                    &mut issues,
                    format!("page {pi}: entry zoom {} outside descriptor [{}, {}]", e.zoom, d.min_zoom, d.max_zoom),
                );
            }
            let b = tile_geo_bounds(e.zoom, e.x, e.y);
            if floor_e7(b.0) < d.min_lon_e7
                || floor_e7(b.1) < d.min_lat_e7
                || ceil_e7(b.2) > d.max_lon_e7
                || ceil_e7(b.3) > d.max_lat_e7
            {
                push(
                    &mut issues,
                    format!("page {pi}: descriptor bbox does not cover tile {}/{}/{}", e.zoom, e.x, e.y),
                );
            }
            let lo = e.cover_t_min.unwrap_or(e.time_start);
            if lo < d.t_min || e.time_end > d.t_max {
                push(
                    &mut issues,
                    format!("page {pi}: descriptor t-bounds [{}, {}] do not cover entry [{}, {}]", d.t_min, d.t_max, lo, e.time_end),
                );
            }
        }

        // Cross-page (and within-page) directory-order monotonicity.
        if let (Some(first), Some(last)) = (leaf.first(), leaf.last()) {
            let fk = (first.zoom, first.hilbert, first.time_start);
            if let Some(pl) = prev_last {
                if pl > fk {
                    push(
                        &mut issues,
                        format!("page {pi}: first key {fk:?} precedes previous page's last key {pl:?} (directory order violated)"),
                    );
                }
            }
            prev_last = Some((last.zoom, last.hilbert, last.time_start));
        }
    }

    if decoded_total != declared_total {
        push(
            &mut issues,
            format!("paged: descriptors declare {declared_total} entries, leaves decoded {decoded_total}"),
        );
    }
    Ok(issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(z: u8, x: u32, y: u32, ts: i64, te: i64, cover: Option<i64>) -> TileEntry {
        TileEntry {
            zoom: z,
            x,
            y,
            time_start: ts,
            time_end: te,
            pack_id: 0,
            offset: (x as u64) * 64,
            length: 50 + x,
            uncompressed_size: 100 + x,
            feature_count: x,
            hilbert: crate::tile::TileId::new(z, x, y, 0).hilbert_index(),
            crc32c: 0x1000 + x,
            temporal_bucket_ms: Some(3_600_000),
            cover_t_min: cover,
        }
    }

    /// A spread-out synthetic corpus across zooms / cells / time buckets.
    fn corpus(all_cover: bool) -> Vec<TileEntry> {
        let mut v = Vec::new();
        for z in [4u8, 8, 12] {
            let n = 1u32 << z;
            for i in 0..40u32 {
                let x = (i * 7) % n;
                let y = (i * 13) % n;
                for b in 0..3i64 {
                    let ts = b * 3_600_000 + i as i64 * 1000;
                    let cover = if all_cover { Some(ts - 500) } else { None };
                    v.push(entry(z, x, y, ts, ts + 3_599_000, cover));
                }
            }
        }
        v
    }

    /// The whole-load decode of a corpus (the ground truth paged must match).
    fn whole_load(entries: &[TileEntry]) -> Vec<TileEntry> {
        decode_directory(&encode_directory(entries)).unwrap()
    }

    #[test]
    fn paged_decode_equals_whole_load_all_cover() {
        let c = corpus(true);
        let want = whole_load(&c);
        for pe in [1usize, 2, 7, 50, 4096, 100_000] {
            for zstd in [true, false] {
                let enc = encode_paged_directory(&c, pe, zstd).unwrap();
                let got = decode_paged_directory(&enc.bytes, enc.root_length, zstd).unwrap();
                assert_eq!(got, want, "page_entries={pe} zstd={zstd}");
            }
        }
    }

    #[test]
    fn paged_decode_equals_whole_load_mixed_cover() {
        // A MIXED corpus: whole-load drops the cover section entirely (all
        // None). Paged must match — its per-page codec must NOT emit cover for
        // an accidentally-all-Some page.
        let mut c = corpus(true);
        c[0].cover_t_min = None; // make it mixed
        let want = whole_load(&c);
        assert!(want.iter().all(|e| e.cover_t_min.is_none()));
        for pe in [1usize, 3, 4096] {
            let enc = encode_paged_directory(&c, pe, true).unwrap();
            let got = decode_paged_directory(&enc.bytes, enc.root_length, true).unwrap();
            assert_eq!(got, want, "page_entries={pe}");
            assert!(got.iter().all(|e| e.cover_t_min.is_none()));
        }
    }

    #[test]
    fn empty_roundtrips() {
        let enc = encode_paged_directory(&[], 4096, true).unwrap();
        assert_eq!(enc.page_count, 0);
        let got = decode_paged_directory(&enc.bytes, enc.root_length, true).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn small_dataset_is_one_page() {
        let c = corpus(true);
        let enc = encode_paged_directory(&c, 100_000, true).unwrap();
        assert_eq!(enc.page_count, 1);
    }

    #[test]
    fn page_count_matches_chunking() {
        let c = corpus(true); // 3 zooms × 40 × 3 = 360 entries
        assert_eq!(c.len(), 360);
        let enc = encode_paged_directory(&c, 100, true).unwrap();
        assert_eq!(enc.page_count, 4); // ceil(360/100)
    }

    #[test]
    fn descriptor_bounds_cover_their_entries() {
        let c = corpus(true);
        let enc = encode_paged_directory(&c, 37, true).unwrap();
        let root = decode_root(&unframe(&enc.bytes[..enc.root_length as usize], true).unwrap())
            .unwrap();
        // Re-derive the sorted slices the encoder used.
        let mut sorted = c.clone();
        sorted.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start, e.temporal_bucket_ms));
        let mut total = 0u32;
        for (page, chunk) in root.pages.iter().zip(sorted.chunks(37)) {
            assert_eq!(page.entry_count as usize, chunk.len());
            total += page.entry_count;
            for e in chunk {
                let b = tile_geo_bounds(e.zoom, e.x, e.y);
                assert!(e.zoom >= page.min_zoom && e.zoom <= page.max_zoom);
                // The entry's true bbox must sit inside the (floored/ceiled)
                // fixed-point descriptor bbox.
                assert!(floor_e7(b.0) >= page.min_lon_e7, "min_lon under-covered");
                assert!(floor_e7(b.1) >= page.min_lat_e7, "min_lat under-covered");
                assert!(ceil_e7(b.2) <= page.max_lon_e7, "max_lon under-covered");
                assert!(ceil_e7(b.3) <= page.max_lat_e7, "max_lat under-covered");
                let lo = e.cover_t_min.unwrap_or(e.time_start);
                assert!(lo >= page.t_min && e.time_end <= page.t_max);
            }
        }
        assert_eq!(total as usize, c.len());
    }

    #[test]
    fn cross_page_key_order_is_monotonic() {
        let c = corpus(true);
        let got = decode_paged_directory(
            &encode_paged_directory(&c, 29, true).unwrap().bytes,
            encode_paged_directory(&c, 29, true).unwrap().root_length,
            true,
        )
        .unwrap();
        // Decoded entries must be globally non-decreasing in (zoom, hilbert, t).
        for w in got.windows(2) {
            let a = (w[0].zoom, w[0].hilbert, w[0].time_start);
            let b = (w[1].zoom, w[1].hilbert, w[1].time_start);
            assert!(a <= b, "directory order violated across pages: {a:?} > {b:?}");
        }
    }

    #[test]
    fn root_encode_decode_roundtrips() {
        let descs = vec![
            PageDescriptor {
                rel_offset: 0,
                length: 1234,
                entry_count: 4096,
                min_zoom: 8,
                max_zoom: 9,
                min_lon_e7: -740_000_000,
                min_lat_e7: 400_000_000,
                max_lon_e7: -730_000_000,
                max_lat_e7: 410_000_000,
                t_min: 1_000,
                t_max: 9_999,
            },
            PageDescriptor {
                rel_offset: 1234,
                length: 5678,
                entry_count: 12,
                min_zoom: 0,
                max_zoom: 14,
                min_lon_e7: -1_800_000_000,
                min_lat_e7: -850_000_000,
                max_lon_e7: 1_800_000_000,
                max_lat_e7: 850_000_000,
                t_min: i64::MIN + 1,
                t_max: i64::MAX - 1,
            },
        ];
        let raw = encode_root(4096, &descs);
        let root = decode_root(&raw).unwrap();
        assert_eq!(root.page_entries, 4096);
        assert_eq!(root.pages, descs);
    }

    #[test]
    fn overlaps_selects_the_right_pages() {
        // Two well-separated pages; a query over page A must not select page B.
        let a = PageDescriptor {
            rel_offset: 0,
            length: 1,
            entry_count: 1,
            min_zoom: 10,
            max_zoom: 10,
            min_lon_e7: -740_000_000,
            min_lat_e7: 400_000_000,
            max_lon_e7: -730_000_000,
            max_lat_e7: 410_000_000,
            t_min: 0,
            t_max: 1000,
        };
        let b = PageDescriptor {
            min_lon_e7: 1_000_000_000,
            max_lon_e7: 1_010_000_000,
            min_lat_e7: -100_000_000,
            max_lat_e7: -90_000_000,
            t_min: 5000,
            t_max: 6000,
            ..a.clone()
        };
        // Query box over A's region, A's time.
        assert!(a.overlaps(10, -735_000_000, 405_000_000, -731_000_000, 408_000_000, 0, 1000));
        assert!(!b.overlaps(10, -735_000_000, 405_000_000, -731_000_000, 408_000_000, 0, 1000));
        // Wrong zoom prunes A.
        assert!(!a.overlaps(9, -735_000_000, 405_000_000, -731_000_000, 408_000_000, 0, 1000));
        // Disjoint time prunes A.
        assert!(!a.overlaps(10, -735_000_000, 405_000_000, -731_000_000, 408_000_000, 2000, 3000));
    }

    #[test]
    fn verify_paged_structure_clean_then_detects_corruption() {
        let c = corpus(true);
        let enc = encode_paged_directory(&c, 19, true).unwrap();
        // A faithful build verifies clean.
        let issues = verify_paged_structure(&enc.bytes, enc.root_length, true).unwrap();
        assert!(issues.is_empty(), "clean build had issues: {issues:?}");

        // Corrupt the root: shrink page 0's bbox so it no longer covers its
        // entries (decode the root, tighten a bound, re-encode the root frame in
        // place). Build a fresh paged object with a deliberately-too-small bbox.
        let mut sorted = c.clone();
        sorted.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start, e.temporal_bucket_ms));
        let root_raw = unframe(&enc.bytes[..enc.root_length as usize], true).unwrap();
        let mut root = decode_root(&root_raw).unwrap();
        // Make page 0 claim an empty bbox far from its tiles → cover violations.
        root.pages[0].min_lon_e7 = 1_790_000_000;
        root.pages[0].max_lon_e7 = 1_800_000_000;
        // Also break temporal coverage on page 1.
        root.pages[1].t_max = i64::MIN;
        let bad_root = encode_root(root.page_entries, &root.pages);
        let bad_root_frame = frame(bad_root, true, compression::ZSTD_LEVEL).unwrap();
        // Reassemble: bad root frame + original leaves. rootLength changes.
        let mut bad = bad_root_frame.clone();
        bad.extend_from_slice(&enc.bytes[enc.root_length as usize..]);
        let bad_issues =
            verify_paged_structure(&bad, bad_root_frame.len() as u64, true).unwrap();
        assert!(
            bad_issues.iter().any(|s| s.contains("bbox does not cover")),
            "expected a bbox-cover violation, got {bad_issues:?}"
        );
        assert!(
            bad_issues.iter().any(|s| s.contains("t-bounds")),
            "expected a t-bounds violation, got {bad_issues:?}"
        );
    }

    #[test]
    fn truncated_root_errors() {
        let c = corpus(true);
        let enc = encode_paged_directory(&c, 50, false).unwrap();
        // rootLength claims more than the object holds.
        assert!(decode_paged_directory(&enc.bytes, enc.bytes.len() as u64 + 1, false).is_err());
        // A corrupt root version.
        let mut raw = encode_root(50, &[]);
        raw[0] = 99;
        assert!(decode_root(&raw).is_err());
    }
}
