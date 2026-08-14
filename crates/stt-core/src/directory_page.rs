//! Paged-directory container, after COPC's hierarchy pages and GeoParquet
//! 1.1's per-row-group bbox columns.
//!
//! The v6 directory ([`crate::directory`]) is a single whole-load blob: a cold
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
//! - A **leaf page** is the *unchanged* v6 codec ([`encode_directory`]) over a
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

use crate::compression;
use crate::directory::TileEntry;
use crate::directory::{decode_directory, encode_directory};
use crate::error::{Error, Result};
use crate::projection::tile_geo_bounds;

/// Root container version (first byte of the decoded root page). Bumped
/// independently of the leaf codec's [`crate::directory::DIRECTORY_VERSION`].
pub const PAGED_ROOT_VERSION: u8 = 1;

/// Descriptor-kind tag: geographic bbox + zoom range + temporal bounds.
pub const DESCRIPTOR_GEO_BBOX: u8 = 0;

/// Descriptor-kind tag 1 (WM-6): kind 0 plus the leaf's minimum BUCKET START.
///
/// The existing `t_min` is `min(cover_t_min ?? time_start)` — the *viewport*
/// prune bound, which can exceed the bucket start whenever a tile's earliest
/// covered instant is later than its bucket. That divergence is exactly why a
/// point lookup has to carry `b_max` slack today (§7.3). Carrying the true
/// minimum bucket start lets the point-lookup prune drop the slack entirely.
///
/// ⚠️ Old readers reject unknown structure strictly, so an archive written at
/// this kind MUST declare it through the manifest capability mechanism — a
/// pre-WM-6 reader has to fail loudly at OPEN, not weirdly at prune time.
pub const DESCRIPTOR_GEO_BBOX_V2: u8 = 1;

/// Default entries per leaf page — the 1024–4096 sweet spot (sim-validated:
/// ~60–130 KB zstd pages, the range coalescer's comfort zone).
pub const DEFAULT_PAGE_ENTRIES: usize = 4096;

/// Fixed-width root header (bytes): version, kind, reserved u16, page_count u32,
/// page_entries u32.
pub(crate) const ROOT_HEADER_LEN: usize = 12;
/// Fixed-width per-page descriptor (bytes). See [`PageDescriptor`] field order.
pub(crate) const DESCRIPTOR_LEN: usize = 52;
/// Per-page descriptor width at [`DESCRIPTOR_GEO_BBOX_V2`]: kind 0 plus an i64.
pub(crate) const DESCRIPTOR_LEN_V2: usize = 60;

/// Conservative upper bound for one v6 directory record after varint
/// expansion. The actual maximum is substantially smaller; keeping headroom
/// makes this a decompression-bomb ceiling rather than an encoding constraint.
pub(crate) const MAX_DECODED_BYTES_PER_ENTRY: usize = 256;
pub(crate) const DIRECTORY_FRAME_OVERHEAD: usize = 64;
/// Absolute ceiling for any decoded directory frame. A forged manifest count
/// must not authorize multi-gigabyte output; datasets that outgrow this should
/// use more/smaller paged leaves.
pub(crate) const MAX_DECODED_DIRECTORY_BYTES: usize = 512 * 1024 * 1024;

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
    /// WM-6 (descriptor kind 1): `min(entry.time_start)` over the leaf — the
    /// true minimum BUCKET START, as distinct from [`Self::t_min`]'s covered
    /// bound. `None` on a kind-0 archive, where the legacy `b_max` slack path
    /// still applies.
    pub min_bucket_start: Option<i64>,
}

impl PageDescriptor {
    /// WM-6: may a POINT lookup at instant `t` skip this leaf entirely?
    ///
    /// The two paths are one predicate with one fallback, so the Rust and TS
    /// readers cannot drift:
    ///
    /// - **kind 1** — compare against the true minimum bucket start. A leaf
    ///   whose earliest bucket begins after `t` cannot hold the tile, exactly.
    /// - **kind 0** — the legacy conservative form: `t_min` is a COVERED bound
    ///   that may sit later than the bucket start, so the comparison has to be
    ///   widened by `b_max` (the largest bucket width) or it would prune a leaf
    ///   that genuinely holds the answer.
    ///
    /// Soundness direction is the point: this may only ever return `true` when
    /// the leaf provably cannot contain the tile. Under-pruning costs a fetch;
    /// over-pruning loses data.
    pub fn prunes_point_lookup(&self, t: i64, b_max: i64) -> bool {
        match self.min_bucket_start {
            Some(min_bucket_start) => min_bucket_start > t,
            None => self.t_min > t.saturating_add(b_max),
        }
    }

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
    /// blake3-128 of the exact at-rest root frame (the object magic is not
    /// part of the frame).
    pub root_hash: String,
    /// blake3-128 of each exact at-rest leaf frame, in descriptor order.
    pub page_hashes: Vec<String>,
}

fn blake3_128_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).as_bytes()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
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

/// Knobs for [`plan_page_breakpoints`] (WM-4).
#[derive(Debug, Clone)]
pub struct PageBreakOptions {
    /// Target entries per leaf — the balance the DP is pulled toward.
    pub target_entries: usize,
    /// Hard ceiling so one segment cannot swallow the archive.
    pub max_entries: usize,
    /// Byte-balance candidate stride, in entries.
    pub candidate_stride: usize,
}

impl Default for PageBreakOptions {
    fn default() -> Self {
        Self {
            target_entries: DEFAULT_PAGE_ENTRIES,
            max_entries: DEFAULT_PAGE_ENTRIES * 2,
            candidate_stride: 1024,
        }
    }
}

/// WM-4 — plan structure-aware leaf breakpoints over `entries`, which MUST
/// already be in directory order.
///
/// Returns segment END indices (exclusive), ascending, last == `entries.len()`.
///
/// The incumbent cuts uniformly every `page_entries`, and §7.2 names the
/// pathology: a leaf that straddles a ZOOM TRANSITION spans sparse low-zoom
/// world tiles and dense high-zoom ones, so its bbox descriptor is near-global
/// and overlaps every viewport — the leaf is then fetched by queries that need
/// none of it. Cutting at zoom transitions makes each descriptor describe one
/// zoom's spatial extent.
///
/// Deterministic: a pure function of the entry sequence and the options. All
/// integer arithmetic, no RNG, no float ordering.
pub fn plan_page_breakpoints(entries: &[TileEntry], opts: &PageBreakOptions) -> Vec<usize> {
    let n = entries.len();
    if n == 0 {
        return Vec::new();
    }
    let target = opts.target_entries.max(1);

    // Two-level plan, so the structural invariant is guaranteed rather than
    // merely likely:
    //   1. ALWAYS cut at every zoom transition. This is what kills the §7.2
    //      pathology — a leaf spanning zooms gets a near-global bbox descriptor
    //      and is fetched by viewports that need none of it.
    //   2. Subdivide each single-zoom run on the balance grid, so a long run
    //      still yields leaves near `target` rather than one enormous page.
    let mut out: Vec<usize> = Vec::new();
    let mut run_start = 0usize;
    for i in 1..=n {
        let boundary = i == n || entries[i].zoom != entries[i - 1].zoom;
        if !boundary {
            continue;
        }
        // Subdivide [run_start, i) evenly so no leaf exceeds `target` and the
        // pieces stay within one of each other (integer split, deterministic).
        let run = i - run_start;
        let pieces = run.div_ceil(target).max(1);
        let base = run / pieces;
        let extra = run % pieces;
        let mut cut = run_start;
        for k in 0..pieces {
            cut += base + usize::from(k < extra);
            out.push(cut);
        }
        run_start = i;
    }
    debug_assert_eq!(out.last().copied(), Some(n));
    out
}

/// Encode the root page (descriptor table) to its raw (pre-framing) bytes.
pub fn encode_root(page_entries: u32, descriptors: &[PageDescriptor]) -> Vec<u8> {
    // WM-6: kind 0 stays the DEFAULT until rebuild window R1, so every existing
    // paged archive's directory bytes are unchanged. Opt in via
    // [`encode_root_with_kind`].
    encode_root_with_kind(page_entries, descriptors, DESCRIPTOR_GEO_BBOX)
}

/// [`encode_root`] at an explicit descriptor kind (WM-6).
///
/// Kind 1 requires EVERY descriptor to carry a `min_bucket_start`: the table is
/// fixed-width and the adversarial-input guard relies on that, so a partial set
/// degrades to kind 0 rather than producing a ragged table.
pub fn encode_root_with_kind(
    page_entries: u32,
    descriptors: &[PageDescriptor],
    requested_kind: u8,
) -> Vec<u8> {
    let kind = if requested_kind == DESCRIPTOR_GEO_BBOX_V2
        && !descriptors.is_empty()
        && descriptors.iter().all(|d| d.min_bucket_start.is_some())
    {
        DESCRIPTOR_GEO_BBOX_V2
    } else {
        DESCRIPTOR_GEO_BBOX
    };
    let width = if kind == DESCRIPTOR_GEO_BBOX_V2 {
        DESCRIPTOR_LEN_V2
    } else {
        DESCRIPTOR_LEN
    };
    let mut buf = Vec::with_capacity(ROOT_HEADER_LEN + descriptors.len() * width);
    buf.push(PAGED_ROOT_VERSION);
    buf.push(kind);
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
        if kind == DESCRIPTOR_GEO_BBOX_V2 {
            // Unwrap is sound: `kind` is v2 only when every descriptor has one.
            put_i64(&mut buf, d.min_bucket_start.unwrap_or(d.t_min));
        }
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
    // Read through the bounds-checked reader: a truncated 1-byte root that is
    // exactly `[PAGED_ROOT_VERSION]` has no byte 1, so indexing it directly
    // panics on untrusted input.
    let descriptor_kind = r.take(1)?[0];
    if descriptor_kind != DESCRIPTOR_GEO_BBOX && descriptor_kind != DESCRIPTOR_GEO_BBOX_V2 {
        return Err(Error::InvalidArchive(format!(
            "paged root: unsupported descriptor kind {descriptor_kind}"
        )));
    }
    let descriptor_len = if descriptor_kind == DESCRIPTOR_GEO_BBOX_V2 {
        DESCRIPTOR_LEN_V2
    } else {
        DESCRIPTOR_LEN
    };
    let reserved = r.u16()?;
    if reserved != 0 {
        return Err(Error::InvalidArchive(format!(
            "paged root: reserved header field is {reserved}, expected 0"
        )));
    }
    let page_count = r.u32()? as usize;
    let page_entries = r.u32()?;
    // Adversarial-input guard: descriptors are fixed-width, so a count beyond
    // what the remaining bytes can hold is corrupt; reject before allocating
    // the table (guarded by tests/adversarial_decode.rs).
    if bytes.len().saturating_sub(r.pos) < page_count.saturating_mul(descriptor_len) {
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
        let reserved = r.u16()?;
        if reserved != 0 {
            return Err(Error::InvalidArchive(format!(
                "paged root: descriptor reserved field is {reserved}, expected 0"
            )));
        }
        let min_lon_e7 = r.i32()?;
        let min_lat_e7 = r.i32()?;
        let max_lon_e7 = r.i32()?;
        let max_lat_e7 = r.i32()?;
        let t_min = r.i64()?;
        let t_max = r.i64()?;
        let min_bucket_start = if descriptor_kind == DESCRIPTOR_GEO_BBOX_V2 {
            Some(r.i64()?)
        } else {
            None
        };
        pages.push(PageDescriptor {
            min_bucket_start,
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
    if r.pos != bytes.len() {
        return Err(Error::InvalidArchive(format!(
            "paged root: {} trailing byte(s) after descriptor table",
            bytes.len() - r.pos
        )));
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

pub(crate) fn unframe_bounded(frame: &[u8], zstd: bool, max_output: usize) -> Result<Vec<u8>> {
    if zstd {
        compression::decompress_zstd_with_dict_bounded(frame, None, max_output)
    } else {
        if frame.len() > max_output {
            return Err(Error::InvalidArchive(format!(
                "directory frame is {} bytes, exceeding its declared {max_output}-byte limit",
                frame.len()
            )));
        }
        Ok(frame.to_vec())
    }
}

/// Encode tile entries into the paged container.
///
/// `entries` need not be pre-sorted; they are sorted into directory order
/// `(zoom, hilbert, time_start, temporal_bucket_ms)` — matching the writer and
/// the v6 codec — then sliced into pages of `page_entries`. `zstd` selects
/// per-page zstd framing (the writer default) vs raw codec bytes.
///
/// The covering section is decided **globally**: if any entry lacks
/// `cover_t_min`, it is stripped from every leaf, so a paged directory decodes
/// byte-for-identical-entries to a whole-load v6 directory of the same corpus
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
    encode_paged_directory_inner(entries, page_entries, zstd, level, None)
}

/// WM-4 — encode with explicit, structure-aware segment boundaries.
///
/// `breakpoints` are segment END indices (exclusive), ascending, last ==
/// `entries.len()` — exactly what [`plan_page_breakpoints`] returns. Every other
/// invariant is the uniform path's: leaves stay contiguous slices of one
/// monotone order, so `verify_paged_structure` is unaffected.
pub fn encode_paged_directory_planned(
    entries: &[TileEntry],
    breakpoints: &[usize],
    zstd: bool,
    level: i32,
) -> Result<EncodedPagedDirectory> {
    // Validate up front: a ragged or non-monotone plan would silently produce a
    // directory whose leaves are not a partition of the entry order.
    let mut prev = 0usize;
    for &b in breakpoints {
        if b <= prev || b > entries.len() {
            return Err(Error::Other(format!(
                "paged directory: breakpoint {b} is not a strictly ascending index within \
                 0..={} (previous {prev})",
                entries.len()
            )));
        }
        prev = b;
    }
    if !entries.is_empty() && prev != entries.len() {
        return Err(Error::Other(format!(
            "paged directory: breakpoints cover {prev} of {} entries",
            entries.len()
        )));
    }
    encode_paged_directory_inner(
        entries,
        DEFAULT_PAGE_ENTRIES,
        zstd,
        level,
        Some(breakpoints),
    )
}

fn encode_paged_directory_inner(
    entries: &[TileEntry],
    page_entries: usize,
    zstd: bool,
    level: i32,
    planned_breakpoints: Option<&[usize]>,
) -> Result<EncodedPagedDirectory> {
    let page_entries = page_entries.max(1);
    if page_entries > u32::MAX as usize {
        return Err(Error::Other(format!(
            "paged directory: page_entries {page_entries} exceeds the u32 container field"
        )));
    }
    let mut sorted: Vec<&TileEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start, e.temporal_bucket_ms));

    // Global cover decision (mirror the whole-load codec's all-or-nothing rule).
    let all_cover = !sorted.is_empty() && sorted.iter().all(|e| e.cover_t_min.is_some());

    let mut descriptors: Vec<PageDescriptor> = Vec::new();
    let mut leaf_frames: Vec<Vec<u8>> = Vec::new();
    let mut rel_offset = 0u64;

    // WM-4: uniform chunking remains the DEFAULT. `plan_page_breakpoints`
    // supplies structure-aware segments; both paths run the identical loop
    // body below, so a planned encode differs only in WHERE the cuts fall.
    let segments: Vec<&[&TileEntry]> = match planned_breakpoints {
        Some(bps) => {
            let mut segs = Vec::with_capacity(bps.len());
            let mut start = 0usize;
            for &end in bps {
                segs.push(&sorted[start..end]);
                start = end;
            }
            segs
        }
        None => sorted.chunks(page_entries).collect(),
    };

    for chunk in segments {
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
        // WM-6: the true minimum BUCKET start, distinct from `tmin`'s covered bound.
        let mut min_bucket = i64::MAX;
        for &e in chunk {
            let b = tile_geo_bounds(e.zoom, e.x, e.y);
            geo.0 = geo.0.min(b.0);
            geo.1 = geo.1.min(b.1);
            geo.2 = geo.2.max(b.2);
            geo.3 = geo.3.max(b.3);
            zmin = zmin.min(e.zoom);
            zmax = zmax.max(e.zoom);
            tmin = tmin.min(e.cover_t_min.unwrap_or(e.time_start));
            min_bucket = min_bucket.min(e.time_start);
            tmax = tmax.max(e.time_end);
        }

        let leaf_length = u32::try_from(leaf.len()).map_err(|_| {
            Error::Other(format!(
                "paged directory: one leaf frame is {} bytes, exceeding the u32 length field",
                leaf.len()
            ))
        })?;
        let entry_count = u32::try_from(chunk.len()).map_err(|_| {
            Error::Other(format!(
                "paged directory: one leaf contains {} entries, exceeding the u32 count field",
                chunk.len()
            ))
        })?;
        descriptors.push(PageDescriptor {
            rel_offset,
            length: leaf_length,
            entry_count,
            min_zoom: zmin,
            max_zoom: zmax,
            min_lon_e7: floor_e7(geo.0),
            min_lat_e7: floor_e7(geo.1),
            max_lon_e7: ceil_e7(geo.2),
            max_lat_e7: ceil_e7(geo.3),
            t_min: tmin,
            t_max: tmax,
            min_bucket_start: (min_bucket != i64::MAX).then_some(min_bucket),
        });
        rel_offset = rel_offset
            .checked_add(leaf.len() as u64)
            .ok_or_else(|| Error::Other("paged directory: leaf offsets overflow u64".into()))?;
        leaf_frames.push(leaf);
    }

    if descriptors.len() > u32::MAX as usize {
        return Err(Error::Other(format!(
            "paged directory: {} pages exceed the u32 page-count field",
            descriptors.len()
        )));
    }
    let root_raw = encode_root(page_entries as u32, &descriptors);
    let root_frame = frame(root_raw, zstd, level)?;
    let root_length = root_frame.len() as u64;
    let page_count = descriptors.len();
    let root_hash = blake3_128_hex(&root_frame);
    let page_hashes = leaf_frames
        .iter()
        .map(|frame| blake3_128_hex(frame))
        .collect();

    let mut bytes = root_frame;
    for f in &leaf_frames {
        bytes.extend_from_slice(f);
    }

    Ok(EncodedPagedDirectory {
        bytes,
        root_length,
        page_count,
        page_entries,
        root_hash,
        page_hashes,
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
    decode_paged_directory_bounded(bytes, root_length, zstd, None)
}

/// Decode with a root-output limit derived from the manifest's `pageCount`.
/// Leaf limits are subsequently derived from each descriptor's `entry_count`.
pub(crate) fn decode_paged_directory_bounded(
    bytes: &[u8],
    root_length: u64,
    zstd: bool,
    declared_page_count: Option<u64>,
) -> Result<Vec<TileEntry>> {
    let rl = usize::try_from(root_length).map_err(|_| {
        Error::InvalidArchive(format!(
            "paged directory: rootLength {root_length} does not fit this platform"
        ))
    })?;
    if rl > bytes.len() {
        return Err(Error::InvalidArchive(format!(
            "paged directory: rootLength {rl} exceeds object size {}",
            bytes.len()
        )));
    }
    let root_limit = match declared_page_count {
        Some(count) => {
            let count = usize::try_from(count).map_err(|_| {
                Error::InvalidArchive(format!(
                    "paged directory: pageCount {count} does not fit this platform"
                ))
            })?;
            ROOT_HEADER_LEN
                .checked_add(count.checked_mul(DESCRIPTOR_LEN).ok_or_else(|| {
                    Error::InvalidArchive("paged directory: pageCount size overflows".into())
                })?)
                .ok_or_else(|| {
                    Error::InvalidArchive("paged directory: root size overflows".into())
                })?
                .min(MAX_DECODED_DIRECTORY_BYTES)
        }
        // Direct codec callers have no manifest. Packed readers always pass
        // the declared count and therefore do not use this compatibility cap.
        None => 64 * 1024 * 1024,
    };
    let root_raw = unframe_bounded(&bytes[..rl], zstd, root_limit)?;
    let root = decode_root(&root_raw)?;
    if let Some(declared) = declared_page_count {
        if root.pages.len() as u64 != declared {
            return Err(Error::InvalidArchive(format!(
                "paged directory: manifest declares {declared} pages, root contains {}",
                root.pages.len()
            )));
        }
    }
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
        let leaf_limit = DIRECTORY_FRAME_OVERHEAD
            .checked_add(
                (d.entry_count as usize)
                    .checked_mul(MAX_DECODED_BYTES_PER_ENTRY)
                    .ok_or_else(|| {
                        Error::InvalidArchive(
                            "paged directory: leaf entry-count limit overflows".into(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                Error::InvalidArchive("paged directory: leaf decode limit overflows".into())
            })?
            .min(MAX_DECODED_DIRECTORY_BYTES);
        let raw = unframe_bounded(frame, zstd, leaf_limit)?;
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

    let rl = usize::try_from(root_length).map_err(|_| {
        Error::InvalidArchive(format!(
            "paged directory: rootLength {root_length} does not fit this platform"
        ))
    })?;
    if rl > bytes.len() {
        return Ok(vec![format!(
            "paged: rootLength {rl} exceeds object size {}",
            bytes.len()
        )]);
    }
    let root = decode_root(&unframe_bounded(&bytes[..rl], zstd, 64 * 1024 * 1024)?)?;

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
        let leaf_limit = DIRECTORY_FRAME_OVERHEAD
            .saturating_add(d.entry_count as usize * MAX_DECODED_BYTES_PER_ENTRY)
            .min(MAX_DECODED_DIRECTORY_BYTES);
        let leaf =
            match unframe_bounded(frame, zstd, leaf_limit).and_then(|raw| decode_directory(&raw)) {
                Ok(l) => l,
                Err(e) => {
                    push(&mut issues, format!("page {pi}: leaf decode failed: {e}"));
                    continue;
                }
            };
        if leaf.len() != d.entry_count as usize {
            push(
                &mut issues,
                format!(
                    "page {pi}: descriptor declares {} entries, leaf decoded {}",
                    d.entry_count,
                    leaf.len()
                ),
            );
        }
        decoded_total += leaf.len();

        for e in &leaf {
            if e.zoom < d.min_zoom || e.zoom > d.max_zoom {
                push(
                    &mut issues,
                    format!(
                        "page {pi}: entry zoom {} outside descriptor [{}, {}]",
                        e.zoom, d.min_zoom, d.max_zoom
                    ),
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
                    format!(
                        "page {pi}: descriptor bbox does not cover tile {}/{}/{}",
                        e.zoom, e.x, e.y
                    ),
                );
            }
            let lo = e.cover_t_min.unwrap_or(e.time_start);
            if lo < d.t_min || e.time_end > d.t_max {
                push(
                    &mut issues,
                    format!(
                        "page {pi}: descriptor t-bounds [{}, {}] do not cover entry [{}, {}]",
                        d.t_min, d.t_max, lo, e.time_end
                    ),
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
            variant_id: crate::tile::RAW_VARIANT_ID,
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
        let root = decode_root(
            &unframe_bounded(
                &enc.bytes[..enc.root_length as usize],
                true,
                64 * 1024 * 1024,
            )
            .unwrap(),
        )
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
            assert!(
                a <= b,
                "directory order violated across pages: {a:?} > {b:?}"
            );
        }
    }

    #[test]
    fn root_encode_decode_roundtrips() {
        let descs = vec![
            PageDescriptor {
                min_bucket_start: None,
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
                min_bucket_start: None,
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
            min_bucket_start: None,
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
        assert!(a.overlaps(
            10,
            -735_000_000,
            405_000_000,
            -731_000_000,
            408_000_000,
            0,
            1000
        ));
        assert!(!b.overlaps(
            10,
            -735_000_000,
            405_000_000,
            -731_000_000,
            408_000_000,
            0,
            1000
        ));
        // Wrong zoom prunes A.
        assert!(!a.overlaps(
            9,
            -735_000_000,
            405_000_000,
            -731_000_000,
            408_000_000,
            0,
            1000
        ));
        // Disjoint time prunes A.
        assert!(!a.overlaps(
            10,
            -735_000_000,
            405_000_000,
            -731_000_000,
            408_000_000,
            2000,
            3000
        ));
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
        let root_raw = unframe_bounded(
            &enc.bytes[..enc.root_length as usize],
            true,
            64 * 1024 * 1024,
        )
        .unwrap();
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
        let bad_issues = verify_paged_structure(&bad, bad_root_frame.len() as u64, true).unwrap();
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

    // ------------------------------------------------------------------
    // WM-6 — leaf-descriptor v2: the min-bucket-start column (§7.3)
    // ------------------------------------------------------------------

    fn desc(t_min: i64, t_max: i64, min_bucket_start: Option<i64>) -> PageDescriptor {
        PageDescriptor {
            rel_offset: 0,
            length: 16,
            entry_count: 1,
            min_zoom: 0,
            max_zoom: 0,
            min_lon_e7: 0,
            min_lat_e7: 0,
            max_lon_e7: 10,
            max_lat_e7: 10,
            t_min,
            t_max,
            min_bucket_start,
        }
    }

    /// Kind 1 round-trips the new column; kind 0 archives keep decoding exactly
    /// as before and report `None`.
    #[test]
    fn descriptor_kind_1_round_trips_and_kind_0_still_decodes() {
        let v2 = vec![desc(500, 900, Some(400)), desc(1500, 1900, Some(1400))];
        let root = decode_root(&encode_root_with_kind(4096, &v2, DESCRIPTOR_GEO_BBOX_V2)).unwrap();
        assert_eq!(root.descriptor_kind, DESCRIPTOR_GEO_BBOX_V2);
        assert_eq!(root.pages[0].min_bucket_start, Some(400));
        assert_eq!(root.pages[1].min_bucket_start, Some(1400));

        let v1 = vec![desc(500, 900, None), desc(1500, 1900, None)];
        let bytes = encode_root(4096, &v1);
        let root = decode_root(&bytes).unwrap();
        assert_eq!(root.descriptor_kind, DESCRIPTOR_GEO_BBOX);
        assert_eq!(root.pages[0].min_bucket_start, None);
        // Kind 0 stays the narrower width — the wire is unchanged for v1.
        assert_eq!(bytes.len(), ROOT_HEADER_LEN + 2 * DESCRIPTOR_LEN);

        // ⚠️ DEFAULT IS v1 UNTIL R1: even when every descriptor carries the new
        // column, plain `encode_root` must still emit kind 0, or every existing
        // paged archive's directory bytes would move.
        let default_bytes = encode_root(4096, &v2);
        assert_eq!(default_bytes[1], DESCRIPTOR_GEO_BBOX);
        assert_eq!(default_bytes.len(), ROOT_HEADER_LEN + 2 * DESCRIPTOR_LEN);
    }

    /// ⚠️ SOUNDNESS DIRECTION. The prune may only fire when the leaf provably
    /// cannot hold the tile. Under-pruning costs a fetch; OVER-pruning loses
    /// data at query time, which is silent.
    #[test]
    fn the_point_prune_never_drops_a_leaf_that_holds_the_answer() {
        let b_max = 3_600_000i64;
        let bucket = 5_000_000i64;
        // The ordinary case: the leaf's covered bound equals its bucket start.
        // v1 must still widen by `b_max`, because from a kind-0 descriptor it
        // cannot tell this case from one where the cover sits a whole bucket
        // later — that indistinguishability IS the slack.
        let v1 = desc(bucket, bucket + 10_000, None);
        let v2 = desc(bucket, bucket + 10_000, Some(bucket));

        // Neither kind may prune a lookup AT the bucket start.
        assert!(
            !v1.prunes_point_lookup(bucket, b_max),
            "v1 pruned the answer"
        );
        assert!(
            !v2.prunes_point_lookup(bucket, b_max),
            "v2 pruned the answer"
        );

        // THE WIN: for every instant in `[bucket - b_max, bucket)` the leaf
        // provably cannot hold the tile, and v2 says so while v1 must keep it.
        for t in [bucket - 1, bucket - b_max / 2, bucket - b_max + 1] {
            assert!(
                v2.prunes_point_lookup(t, b_max),
                "v2 missed a prune at t={t}"
            );
            assert!(
                !v1.prunes_point_lookup(t, b_max),
                "v1 is supposed to be conservative at t={t}"
            );
        }
    }

    /// v2 is never LOOSER than v1: anything v2 keeps, v1 keeps too. Swept, so a
    /// sign error in either branch shows up.
    #[test]
    fn kind_1_is_never_looser_than_kind_0() {
        let b_max = 1_000i64;
        for bucket in [0i64, 500, 5_000] {
            for cover in [0i64, 500, 5_000] {
                let t_min = bucket.max(cover);
                let v1 = desc(t_min, t_min + 10_000, None);
                let v2 = desc(t_min, t_min + 10_000, Some(bucket));
                for t in (-2_000..8_000).step_by(250) {
                    if v2.prunes_point_lookup(t, b_max) {
                        // v2 pruning is allowed to be tighter, but must never
                        // prune something that genuinely starts at or after t.
                        assert!(bucket > t, "v2 pruned t={t} with bucket={bucket}");
                    }
                    if !v1.prunes_point_lookup(t, b_max) {
                        // v1 keeping is the conservative case; nothing to check.
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // WM-4 — structure-aware directory page breakpoints (§7.2)
    // ------------------------------------------------------------------

    fn entry_at(zoom: u8, x: u32, t: i64) -> TileEntry {
        entry(zoom, x, 0, t, t + 1_000, None)
    }

    /// Breakpoints partition the entry order exactly: ascending, in range, and
    /// ending on `len()`. Anything else would produce leaves that are not a
    /// partition of directory order.
    #[test]
    fn breakpoints_are_a_strict_partition_of_directory_order() {
        let entries: Vec<_> = (0..10_000u32)
            .map(|i| entry_at((i / 2_000) as u8, i, i as i64))
            .collect();
        let bps = plan_page_breakpoints(&entries, &PageBreakOptions::default());
        assert!(!bps.is_empty());
        let mut prev = 0usize;
        for &b in &bps {
            assert!(b > prev, "not ascending: {bps:?}");
            assert!(b <= entries.len());
            prev = b;
        }
        assert_eq!(*bps.last().unwrap(), entries.len());
    }

    /// ⭐ The §7.2 pathology, fixed: a leaf must not straddle a zoom transition.
    /// A straddling leaf spans sparse low-zoom world tiles AND dense high-zoom
    /// ones, so its bbox descriptor is near-global and overlaps every viewport.
    #[test]
    fn no_planned_leaf_straddles_a_zoom_transition() {
        // Zoom runs deliberately NOT aligned to the uniform 4096 grid.
        let mut entries = Vec::new();
        for (z, count) in [(2u8, 3_000usize), (3, 5_000), (4, 1_500)] {
            for i in 0..count {
                entries.push(entry_at(z, i as u32, i as i64));
            }
        }
        let bps = plan_page_breakpoints(&entries, &PageBreakOptions::default());
        let mut start = 0usize;
        for &end in &bps {
            let seg = &entries[start..end];
            let z0 = seg[0].zoom;
            assert!(
                seg.iter().all(|e| e.zoom == z0),
                "a leaf spans zooms {:?} — the §7.2 near-global-bbox pathology",
                seg.iter()
                    .map(|e| e.zoom)
                    .collect::<std::collections::BTreeSet<_>>()
            );
            start = end;
        }
        // And the uniform incumbent DOES straddle, so the test is not vacuous.
        let uniform_spans_zooms = entries[0..DEFAULT_PAGE_ENTRIES]
            .iter()
            .map(|e| e.zoom)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1;
        assert!(
            uniform_spans_zooms,
            "fixture must be one where uniform chunking straddles, else this proves nothing"
        );
    }

    /// Deterministic: a pure function of the entries and options.
    #[test]
    fn planning_is_deterministic() {
        let entries: Vec<_> = (0..5_000u32)
            .map(|i| entry_at((i / 900) as u8, i, i as i64))
            .collect();
        let a = plan_page_breakpoints(&entries, &PageBreakOptions::default());
        let b = plan_page_breakpoints(&entries, &PageBreakOptions::default());
        assert_eq!(a, b);
    }

    /// A planned encode round-trips to the same ENTRIES as the uniform one —
    /// cuts move, content does not.
    #[test]
    fn a_planned_encode_round_trips_the_same_entries() {
        let entries: Vec<_> = (0..3_000u32)
            .map(|i| entry_at((i / 700) as u8, i, i as i64))
            .collect();
        let bps = plan_page_breakpoints(&entries, &PageBreakOptions::default());
        let planned = encode_paged_directory_planned(&entries, &bps, true, 3).unwrap();
        let uniform =
            encode_paged_directory_level(&entries, DEFAULT_PAGE_ENTRIES, true, 3).unwrap();

        let a = decode_paged_directory(&planned.bytes, planned.root_length, true).unwrap();
        let b = decode_paged_directory(&uniform.bytes, uniform.root_length, true).unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(a, b, "planned cuts must not change the decoded entry set");
    }

    /// A ragged or non-covering plan is refused rather than silently producing
    /// leaves that are not a partition.
    #[test]
    fn a_non_covering_plan_is_refused() {
        let entries: Vec<_> = (0..100u32).map(|i| entry_at(0, i, i as i64)).collect();
        assert!(encode_paged_directory_planned(&entries, &[10, 5], true, 3).is_err());
        assert!(encode_paged_directory_planned(&entries, &[10, 50], true, 3).is_err());
        assert!(encode_paged_directory_planned(&entries, &[10, 100], true, 3).is_ok());
    }
}
