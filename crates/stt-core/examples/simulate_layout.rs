//! Build-time STT blob-layout simulator.
//!
//! Opens a real `.stt` via the public reader API, re-linearizes blobs under
//! several candidate orderings (assigning synthetic byte offsets by prefix-
//! summing real compressed lengths in candidate order, with first-occurrence
//! dedup), then for a synthetic query set replays the EXACT client request
//! path: split each query's working set into <=maxRequests batches in priority
//! order (tileset.ts:805-810), coalesce each batch (archive.ts:660-682), and
//! report requests / fetched / waste / decode plus an R2-weighted scalar cost.
//!
//! This answers the v4 roadmap's open Decision #1 (spatial-major vs space-time
//! ordering) with measured numbers instead of intuition.
//!
//! Run:
//!   cargo run --release -p stt-core --example simulate_layout -- \
//!     examples/showcase/public/data/flights.stt
//!
//! VALIDATION GATE: under the baseline ordering, synthetic offsets MUST equal
//! the real on-disk offsets. If "BASELINE OFFSET MATCH" prints FAIL, every
//! downstream number is invalid -- fix the model before reading rankings.

use std::collections::HashMap;
use std::error::Error;

use stt_core::archive::TileEntry; // NOT re-exported; archive.rs:55
// Shared, tested ordering logic — the SAME code the writer uses, so the sim's
// prediction for an ordering equals what create_optimized_with_ordering produces.
use stt_core::curve::{self, BlobOrdering};
use stt_core::metadata::Metadata; // metadata.rs:188
use stt_core::projection::{lonlat_to_tile, tile_to_lonlat}; // projection.rs
use stt_core::ArchiveReader; // re-exported, lib.rs:26

// ---- constants that mirror the live runtime code ----
const HEADER_SIZE: u64 = 64; // archive.rs:43 (private; validated by the baseline gate)
// Client knobs, overridable from the environment so a build can SWEEP them:
//   STT_SIM_GAP=<bytes>     range-coalesce gap (archive.ts `coalesceGapBytes`)
//   STT_SIM_MAXREQ=<count>  per-batch tile cap before coalescing (tileset.ts)
// A very large STT_SIM_MAXREQ (e.g. 1000000) models the post-P0 client, which
// drops the ≤12 pre-split and coalesces the WHOLE working set globally.
const DEFAULT_RANGE_COALESCE_GAP: u64 = 32 * 1024; // archive.ts:52 (pre-P0 default)
const DEFAULT_MAX_REQUESTS: usize = 12; // tileset.ts (pre-P0 default)

/// The two client knobs, resolved once from the environment.
#[derive(Clone, Copy)]
struct ClientCfg {
    gap: u64,
    max_requests: usize,
}

impl ClientCfg {
    fn from_env() -> Self {
        let gap = std::env::var("STT_SIM_GAP")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(DEFAULT_RANGE_COALESCE_GAP);
        let max_requests = std::env::var("STT_SIM_MAXREQ")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MAX_REQUESTS);
        Self { gap, max_requests }
    }
}

// ---- R2-aware cost weights (seconds-domain; requests in RTTs) ----
const W_REQ: f64 = 0.06; // s per range request (~one RTT) -- dominant on R2 (egress free)
const W_DL: f64 = 1.0; // bytesFetched / BW (egress free, but wall-clock)
const W_DEC: f64 = 1.0; // decodeBytes / DEC
const W_TAIL: f64 = 0.5; // maxReqBytes / BW (TTFB / first-paint stall)
const BW: f64 = 50.0e6; // 50 MB/s download
const DEC: f64 = 500.0e6; // 500 MB/s zstd decode

// ---------------------------------------------------------------------------
// Working representation
// ---------------------------------------------------------------------------

/// Dedup identity = the directory's RLE/shared-blob key. The directory already
/// collapsed byte-identical blobs to a shared offset (archive.rs:673-684), so
/// grouping by these fields recovers the exact distinct-blob set.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BlobId {
    offset: u64,
    length: u32,
    uncompressed_size: u32,
    crc32c: u32,
}

#[derive(Clone)]
struct Cell {
    zoom: u8,
    x: u32,
    y: u32,
    hilbert: u64,
    time_start: i64,
    time_end: i64,
    cover_t_min: i64, // tight lower bound (== time_start when no covering)
    uncompressed_size: u32,
    tb: i64, // floor(time_start / bucket)  (bucket index)
    blob: BlobId,
    // assigned per candidate during linearize():
    syn_off: u64,
    syn_len: u32,
}

struct Query {
    name: &'static str,
    z0: u8,
    fallback: bool,
    min_lon: f64,
    min_lat: f64,
    max_lon: f64,
    max_lat: f64,
    t_start: i64,
    t_end: i64,
    freq: f64,
    cap_to_primary_zoom: bool, // Q6: zoom=z0 only, no parent fallback
}

#[derive(Default, Clone)]
struct Metrics {
    requests: u64,
    fetched: u64,
    useful: u64,
    waste: u64,
    max_req: u64,
    decode: u64,
}

// ---------------------------------------------------------------------------
// Candidate orderings. The 4 shipping orders delegate to stt_core::curve (the
// SAME code the writer uses); RealDisk and the experimental Aniso are sim-only.
// Zoom is ALWAYS the leading key component (directory codec requirement).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Candidate {
    RealDisk,    // the actual current on-disk byte order (from real offsets)
    SpatialMajor, // (zoom,hilbert,time_start) -- what finalize_buffered / a reorder pass produces
    TimeMajor,
    Hilbert3,
    Morton3,
    Aniso(u32),
}

impl Candidate {
    fn name(&self) -> String {
        match self {
            Candidate::RealDisk => "0-REAL-DISK (current, add-order)".into(),
            Candidate::SpatialMajor => "a-spatial-major (zoom,hilbert,t)".into(),
            Candidate::TimeMajor => "b-time-major (zoom,tb,hilbert)".into(),
            Candidate::Hilbert3 => "c-hilbert3 (zoom,H3(x,y,t))".into(),
            Candidate::Morton3 => "d-morton3 (zoom,Z3(x,y,t))".into(),
            Candidate::Aniso(g) => format!("e-aniso (zoom,hilbert>>{g},tb,hilbert)"),
        }
    }
    /// Sort key as (zoom, primary:u64, secondary:u64, tertiary:u64).
    /// (RealDisk is special-cased in linearize and never keyed here.)
    fn key(&self, c: &Cell, tb_min: i64, tb_span: i64) -> (u8, u64, u64, u64) {
        let st = |o: BlobOrdering| {
            curve::space_time_key(o, c.zoom, c.x, c.y, c.hilbert, c.time_start, c.tb, tb_min, tb_span)
        };
        match self {
            Candidate::RealDisk => (0, c.blob.offset, 0, 0),
            Candidate::SpatialMajor => st(BlobOrdering::SpatialMajor),
            Candidate::TimeMajor => st(BlobOrdering::TimeMajor),
            Candidate::Hilbert3 => st(BlobOrdering::Hilbert3),
            Candidate::Morton3 => st(BlobOrdering::Morton3),
            Candidate::Aniso(g) => {
                let g = *g;
                (c.zoom, c.hilbert >> g, c.tb as u64, c.hilbert)
            }
        }
    }
    /// The `stt-build --blob-ordering` / repack token for orderings the WRITER
    /// can actually produce (None for RealDisk and the experimental Aniso).
    fn cli_name(&self) -> Option<&'static str> {
        match self {
            Candidate::SpatialMajor => Some("spatial"),
            Candidate::TimeMajor => Some("time-major"),
            Candidate::Hilbert3 => Some("hilbert3"),
            Candidate::Morton3 => Some("morton3"),
            Candidate::RealDisk | Candidate::Aniso(_) => None,
        }
    }
    /// Tie-break preference when costs are within noise: favour the robust
    /// generalist (Hilbert) then time-major.
    fn pick_rank(&self) -> u8 {
        match self {
            Candidate::Hilbert3 => 0,
            Candidate::TimeMajor => 1,
            Candidate::SpatialMajor => 2,
            Candidate::Morton3 => 3,
            _ => 9,
        }
    }
}

// ---------------------------------------------------------------------------
// Linearize: assign synthetic offsets. Models what create_reordered produces:
// CONTENT-dedup (the writer hashes the recompressed payload with blake3; we
// proxy that with the content key (crc32c,length,uncompressed_size) — identical
// payloads compress identically, so this matches) with FIRST-occurrence in
// candidate sort order claiming the slot. This is what makes the prediction
// match reality on high-dedup datasets (e.g. earthquakes, 76% identical tiles)
// where dedup itself can hurt range locality.
// ---------------------------------------------------------------------------
fn content_key(b: &BlobId) -> (u32, u32, u32) {
    (b.crc32c, b.length, b.uncompressed_size)
}

fn linearize(cells: &mut [Cell], cand: &Candidate, tb_min: i64, tb_span: i64) {
    // RealDisk = the actual current layout: synthetic offsets ARE the real
    // offsets, byte-for-byte. No repack, no assumptions -- this is ground truth.
    if *cand == Candidate::RealDisk {
        for c in cells.iter_mut() {
            c.syn_off = c.blob.offset;
            c.syn_len = c.blob.length;
        }
        return;
    }
    // index permutation sorted by candidate key, tie-broken by real offset
    // (determinism + reproduces baseline first-occurrence exactly).
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by(|&a, &b| {
        cand.key(&cells[a], tb_min, tb_span)
            .cmp(&cand.key(&cells[b], tb_min, tb_span))
            .then(cells[a].blob.offset.cmp(&cells[b].blob.offset))
    });
    let mut cursor = HEADER_SIZE;
    let mut slot: HashMap<(u32, u32, u32), (u64, u32)> = HashMap::new();
    for &i in &order {
        let id = cells[i].blob;
        slot.entry(content_key(&id)).or_insert_with(|| {
            let o = cursor;
            cursor += id.length as u64;
            (o, id.length)
        });
    }
    for c in cells.iter_mut() {
        let (o, l) = slot[&content_key(&c.blob)];
        c.syn_off = o;
        c.syn_len = l;
    }
}

// ---------------------------------------------------------------------------
// Tile-box selection: mirror boundsToTiles (archive.ts:1004-1017). lonlat_to_tile
// ERRORS outside lat +-85.0511 / lon +-180, so clamp first. Y is flipped
// (maxLat -> smaller y), resolved by min/max on both corners.
// ---------------------------------------------------------------------------
fn box_tiles(q: &Query, z: u8) -> (u32, u32, u32, u32) {
    let clon = |l: f64| l.clamp(-180.0, 180.0);
    let clat = |l: f64| l.clamp(-85.0511, 85.0511);
    let (x0, ya) = lonlat_to_tile(clon(q.min_lon), clat(q.max_lat), z).unwrap_or((0, 0));
    let (x1, yb) = lonlat_to_tile(clon(q.max_lon), clat(q.min_lat), z).unwrap_or((0, 0));
    (x0.min(x1), ya.min(yb), x0.max(x1), ya.max(yb))
}

// ---------------------------------------------------------------------------
// Evaluate one query: build the working set in PRIORITY ORDER (primary zoom
// first, then parents -- tileset.ts:569-573), chunk into <=MAX_REQUESTS batches
// (tileset.ts:805-810), coalesce each batch, sum metrics.
// ---------------------------------------------------------------------------
fn eval_query(cells: &[Cell], q: &Query, min_zoom: u8, max_zoom: u8, cfg: ClientCfg) -> Metrics {
    // zoom list in PRIORITY order: primary z0 first, then z0-1..z0-4 (parents)
    let mut zooms: Vec<u8> = vec![q.z0];
    if q.fallback && !q.cap_to_primary_zoom {
        for i in 1..=4i32 {
            let z = q.z0 as i32 - i;
            if z < min_zoom as i32 {
                break;
            }
            zooms.push(z as u8);
        }
    }
    zooms.retain(|&z| z >= min_zoom && z <= max_zoom);

    // Working set, in priority order (zoom-major). Within a zoom, the client
    // enqueues in x-major grid order -- getTileIdsInBounds iterates outer x,
    // inner y (boundsToTiles), pushing each tile's overlapping time buckets in
    // ascending order. This is the BUG-3 fix: batch MEMBERSHIP follows this
    // enqueue order (NOT Hilbert); only within-batch coalescing uses byte order.
    let mut sel: Vec<&Cell> = Vec::new();
    for &z in &zooms {
        let (xmin, ymin, xmax, ymax) = box_tiles(q, z);
        let mut zsel: Vec<&Cell> = cells
            .iter()
            .filter(|c| {
                c.zoom == z
                    && c.x >= xmin
                    && c.x <= xmax
                    && c.y >= ymin
                    && c.y <= ymax
                    // time OVERLAP (tileset.ts): intervals intersect. Lower
                    // bound uses the tight covering `cover_t_min` (== time_start
                    // when the archive has no covering section) so tiles whose
                    // data is entirely after the window are pruned.
                    && !(c.time_end < q.t_start || c.cover_t_min > q.t_end)
            })
            .collect();
        zsel.sort_by_key(|c| (c.x, c.y, c.time_start));
        sel.extend(zsel);
    }
    if sel.is_empty() {
        return Metrics::default();
    }

    // Chunk into <=MAX_REQUESTS batches; coalesce each independently; sum.
    let mut m = Metrics::default();
    let mut counted_unc: HashMap<u64, u32> = HashMap::new(); // syn_off -> uncompressed (decode dedup across batches)
    let mut counted_len: HashMap<u64, u32> = HashMap::new(); // syn_off -> compressed len (useful dedup across batches)
    for batch in sel.chunks(cfg.max_requests) {
        // distinct synthetic ranges in this batch (dedup-shared blobs collapse)
        let mut seen: HashMap<u64, (u64, u32, u32)> = HashMap::new();
        for c in batch {
            seen.entry(c.syn_off)
                .or_insert((c.syn_off, c.syn_len, c.uncompressed_size));
        }
        let mut ranges: Vec<(u64, u64)> = seen
            .values()
            .map(|&(o, l, _)| (o, o + l as u64 - 1))
            .collect();
        ranges.sort_by_key(|r| r.0); // archive.ts:653
                                     // coalesce
        let mut groups: Vec<(u64, u64)> = Vec::new();
        for (s, e) in ranges {
            if let Some(last) = groups.last_mut() {
                if s.saturating_sub(last.1 + 1) <= cfg.gap {
                    // archive.ts:664
                    last.1 = last.1.max(e); // archive.ts:666
                    continue;
                }
            }
            groups.push((s, e));
        }
        m.requests += groups.len() as u64;
        let batch_fetched: u64 = groups.iter().map(|(s, e)| e - s + 1).sum();
        m.fetched += batch_fetched;
        let batch_max = groups.iter().map(|(s, e)| e - s + 1).max().unwrap_or(0);
        m.max_req = m.max_req.max(batch_max);
        // useful + decode counted ONCE per distinct blob across the whole query
        for (&off, &(_, len, unc)) in &seen {
            counted_len.entry(off).or_insert(len);
            counted_unc.entry(off).or_insert(unc);
        }
    }
    m.useful = counted_len.values().map(|&l| l as u64).sum();
    m.decode = counted_unc.values().map(|&u| u as u64).sum();
    m.waste = m.fetched.saturating_sub(m.useful);
    m
}

// ---------------------------------------------------------------------------
// Query set, anchored to a DENSE populated (cell, time) in the real data so it
// exercises every dataset — including sparse/event ones where bounds-center is
// empty ocean. `a_lon/a_lat/a_z` is the anchor cell center+zoom; `t_anchor` a
// populated instant.
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
fn build_queries(
    a_lon: f64,
    a_lat: f64,
    a_z: u8,
    t_anchor: i64,
    lon0: f64,
    lat0: f64,
    lon1: f64,
    lat1: f64,
    t0: i64,
    t1: i64,
    b: i64,
    min_z: u8,
    max_z: u8,
) -> Vec<Query> {
    let clon = a_lon.clamp(-179.0, 179.0);
    let clat = a_lat.clamp(-84.0, 84.0);
    let z0 = a_z.clamp(min_z, max_z);
    let z_zoomout = a_z.saturating_sub(3).clamp(min_z, max_z);
    let n0 = (1u64 << z0) as f64;
    let dlon = 360.0 / n0 * 1.5; // ~3 tiles wide at z0
    let dlat = 170.0 / n0 * 1.5;
    let tmid = t_anchor;
    // Playback window: a day, or several buckets for coarse-bucket datasets.
    let day = (b * 8).max(86_400_000i64);
    let n_frames = ((t1 - t0).max(1) / b.max(1)) as f64;
    vec![
        Query {
            name: "Q1 parked 1-frame",
            z0,
            fallback: true,
            cap_to_primary_zoom: false,
            min_lon: clon - dlon,
            min_lat: clat - dlat,
            max_lon: clon + dlon,
            max_lat: clat + dlat,
            t_start: tmid,
            t_end: tmid + b,
            freq: 1.0,
        },
        Query {
            name: "Q2 parked-playback",
            z0,
            fallback: true,
            cap_to_primary_zoom: false,
            min_lon: clon - dlon,
            min_lat: clat - dlat,
            max_lon: clon + dlon,
            max_lat: clat + dlat,
            t_start: tmid - day / 2,
            t_end: tmid + day / 2,
            freq: n_frames.min(1000.0),
        },
        Query {
            name: "Q3 pan fixed-time",
            z0,
            fallback: true,
            cap_to_primary_zoom: false,
            min_lon: clon - dlon * 6.0,
            min_lat: clat - dlat,
            max_lon: clon + dlon * 6.0,
            max_lat: clat + dlat,
            t_start: tmid,
            t_end: tmid + b,
            freq: 20.0,
        },
        Query {
            name: "Q4 zoom-out",
            z0: z_zoomout,
            fallback: true,
            cap_to_primary_zoom: false,
            min_lon: clon - dlon * 8.0,
            min_lat: clat - dlat * 8.0,
            max_lon: clon + dlon * 8.0,
            max_lat: clat + dlat * 8.0,
            t_start: tmid - day / 2,
            t_end: tmid + day / 2,
            freq: 5.0,
        },
        Query {
            name: "Q5 wide-TIME 1tile",
            z0,
            fallback: false,
            cap_to_primary_zoom: false,
            min_lon: clon,
            min_lat: clat,
            max_lon: clon,
            max_lat: clat,
            t_start: t0,
            t_end: t1,
            freq: 3.0,
        },
        Query {
            name: "Q6 wide-SPACE instant",
            z0,
            fallback: false,
            cap_to_primary_zoom: true,
            min_lon: lon0,
            min_lat: lat0,
            max_lon: lon1,
            max_lat: lat1,
            t_start: tmid,
            t_end: tmid + b,
            freq: 3.0,
        },
    ]
}

// ---------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: simulate_layout <archive.stt> [aniso_g=6]")?;
    let aniso_g: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);

    let cfg = ClientCfg::from_env();
    let reader = ArchiveReader::open(&path)?;
    let meta: &Metadata = reader.metadata();
    let b_default = (meta.temporal_bucket_ms.max(1)) as i64;

    let mut cells: Vec<Cell> = reader
        .entries()
        .iter()
        .map(|e: &TileEntry| {
            let b = e
                .temporal_bucket_ms
                .map(|v| v as i64)
                .unwrap_or(b_default)
                .max(1);
            Cell {
                zoom: e.zoom,
                x: e.x,
                y: e.y,
                hilbert: e.hilbert,
                time_start: e.time_start,
                time_end: e.time_end,
                cover_t_min: e.cover_t_min.unwrap_or(e.time_start),
                uncompressed_size: e.uncompressed_size,
                tb: e.time_start.div_euclid(b),
                blob: BlobId {
                    offset: e.offset,
                    length: e.length,
                    uncompressed_size: e.uncompressed_size,
                    crc32c: e.crc32c,
                },
                syn_off: 0,
                syn_len: 0,
            }
        })
        .collect();

    if cells.is_empty() {
        return Err("archive has no entries".into());
    }

    let tb_min = cells.iter().map(|c| c.tb).min().unwrap();
    let tb_max = cells.iter().map(|c| c.tb).max().unwrap();
    let tb_span = tb_max - tb_min;

    let (lon0, lat0, lon1, lat1) = (
        meta.bounds.min_lon,
        meta.bounds.min_lat,
        meta.bounds.max_lon,
        meta.bounds.max_lat,
    );
    let (t0, t1) = (meta.time_range.start as i64, meta.time_range.end as i64); // u64 -> i64

    // Data-driven query anchor: the zoom with the most distinct populated cells,
    // its densest (x,y) cell, and a populated instant (median time there). This
    // makes the query set hit real data even for sparse/event datasets where the
    // bbox center is empty.
    let mut per_zoom: HashMap<u8, std::collections::HashSet<(u32, u32)>> = HashMap::new();
    for c in &cells {
        per_zoom.entry(c.zoom).or_default().insert((c.x, c.y));
    }
    let a_z = per_zoom
        .iter()
        .max_by_key(|(z, s)| (s.len(), **z))
        .map(|(z, _)| *z)
        .unwrap_or(meta.max_zoom);
    let mut cell_times: HashMap<(u32, u32), Vec<i64>> = HashMap::new();
    for c in &cells {
        if c.zoom == a_z {
            cell_times.entry((c.x, c.y)).or_default().push(c.time_start);
        }
    }
    // Tie-break on (x,y) so the anchor is deterministic regardless of HashMap
    // iteration order (otherwise the pick/numbers wobble run-to-run on ties).
    let ((cx, cy), mut times) = cell_times
        .into_iter()
        .max_by_key(|((x, y), ts)| (ts.len(), *x, *y))
        .unwrap();
    times.sort_unstable();
    let t_anchor = times[times.len() / 2];
    let nw = tile_to_lonlat(cx, cy, a_z);
    let se = tile_to_lonlat(cx.wrapping_add(1), cy.wrapping_add(1), a_z);
    let a_lon = (nw.x() + se.x()) / 2.0;
    let a_lat = (nw.y() + se.y()) / 2.0;

    let queries = build_queries(
        a_lon,
        a_lat,
        a_z,
        t_anchor,
        lon0,
        lat0,
        lon1,
        lat1,
        t0,
        t1,
        b_default,
        meta.min_zoom,
        meta.max_zoom,
    );

    println!("archive: {}", path);
    {
        let n = reader.entries().len();
        let with_cover = reader.entries().iter().filter(|e| e.cover_t_min.is_some()).count();
        let tighter = reader
            .entries()
            .iter()
            .filter(|e| e.cover_t_min.is_some_and(|c| c > e.time_start))
            .count();
        println!(
            "  covering: cover_t_min present on {}/{} entries ({} tighter than bucket edge)",
            with_cover, n, tighter
        );
    }
    println!(
        "  client: coalesce_gap={}B max_requests={}{}",
        cfg.gap,
        cfg.max_requests,
        if cfg.max_requests >= 1_000_000 { " (global-coalesce / post-P0)" } else { "" }
    );
    let tbits = curve::bits_for((tb_span.max(0) + 1) as u64);
    println!(
        "  entries={} bucket_ms={} zoom=[{}..{}] time_bits={} curve_side@maxzoom={}b (space {}b vs time {}b)",
        cells.len(),
        meta.temporal_bucket_ms,
        meta.min_zoom,
        meta.max_zoom,
        tbits,
        curve::curve_bits(meta.max_zoom, tb_span),
        meta.max_zoom,
        tbits
    );
    println!(
        "  bounds=[{:.3},{:.3},{:.3},{:.3}] time=[{}..{}] buckets=[{}..{}] ({} buckets)",
        lon0,
        lat0,
        lon1,
        lat1,
        t0,
        t1,
        tb_min,
        tb_max,
        tb_span + 1
    );
    if meta.max_zoom as u32 > curve::CURVE_BITS_CAP || tbits > curve::CURVE_BITS_CAP {
        println!(
            "  WARNING: axis exceeds curve cap {} (max_zoom={}, time_bits={}) -> 3D-curve low bits dropped",
            curve::CURVE_BITS_CAP, meta.max_zoom, tbits
        );
    }

    // ---- layout diagnostics (ground truth about the real archive) ----
    {
        use std::collections::HashSet;
        let mut by_offset: HashSet<u64> = HashSet::new();
        let mut by_content: HashSet<(u32, u32, u32)> = HashSet::new();
        let mut blob_bytes: u64 = 0;
        let mut max_end: u64 = 0;
        let mut min_off: u64 = u64::MAX;
        for c in &cells {
            if by_offset.insert(c.blob.offset) {
                blob_bytes += c.blob.length as u64; // distinct physical blob bytes
            }
            by_content.insert((c.blob.crc32c, c.blob.length, c.blob.uncompressed_size));
            max_end = max_end.max(c.blob.offset + c.blob.length as u64);
            min_off = min_off.min(c.blob.offset);
        }
        let span = max_end - min_off;
        let contiguous = blob_bytes == span;
        println!("\n[diagnostics]");
        println!(
            "  query anchor: densest cell (z{} x{} y{}) center=({:.3},{:.3}) t_anchor={}",
            a_z, cx, cy, a_lon, a_lat, t_anchor
        );
        println!(
            "  distinct physical blobs (current dedup): {} / {} entries",
            by_offset.len(),
            cells.len()
        );
        println!(
            "  byte-identical content groups (latent dedup ceiling): {} / {} entries  -> up to {:.1}% blobs removable by create_optimized",
            by_content.len(),
            cells.len(),
            100.0 * (cells.len() - by_content.len()) as f64 / cells.len() as f64
        );
        println!(
            "  blob region: first_off={} max_end={} distinct_bytes={} span={} contiguous={}",
            min_off, max_end, blob_bytes, span, contiguous
        );
        // Does the real disk order already equal spatial-major(zoom,hilbert,t)?
        linearize(&mut cells, &Candidate::SpatialMajor, tb_min, tb_span);
        let already_spatial = cells.iter().all(|c| c.syn_off == c.blob.offset);
        println!(
            "  real disk order == spatial-major(zoom,hilbert,t)?  {}  ({})",
            already_spatial,
            if already_spatial {
                "build already lays bytes spatial-major"
            } else {
                "build writes add-order; bytes are NOT spatial-sorted (reorder pass would change layout)"
            }
        );
    }

    let candidates = [
        Candidate::RealDisk,
        Candidate::SpatialMajor,
        Candidate::TimeMajor,
        Candidate::Hilbert3,
        Candidate::Morton3,
        Candidate::Aniso(aniso_g),
    ];

    let mut ranking: Vec<(String, f64)> = Vec::new();
    let mut writer_picks: Vec<(&'static str, u8, f64)> = Vec::new(); // (cli, pick_rank, cost)
    let mut real_cost = f64::NAN;
    for cand in candidates.iter() {
        linearize(&mut cells, cand, tb_min, tb_span);
        println!("\n=== ordering: {} ===", cand.name());
        println!(
            "{:<26} {:>6} {:>13} {:>13} {:>7} {:>11} {:>13}",
            "query", "reqs", "fetched", "useful", "waste%", "maxReq", "decode"
        );
        let mut total = 0.0;
        for q in &queries {
            let m = eval_query(&cells, q, meta.min_zoom, meta.max_zoom, cfg);
            let wpct = if m.fetched > 0 {
                100.0 * m.waste as f64 / m.fetched as f64
            } else {
                0.0
            };
            println!(
                "{:<26} {:>6} {:>13} {:>13} {:>6.1} {:>11} {:>13}",
                q.name, m.requests, m.fetched, m.useful, wpct, m.max_req, m.decode
            );
            total += q.freq
                * (W_REQ * m.requests as f64
                    + W_DL * m.fetched as f64 / BW
                    + W_DEC * m.decode as f64 / DEC
                    + W_TAIL * m.max_req as f64 / BW);
        }
        println!("  COST = {:.4} s (weighted)", total);
        ranking.push((cand.name(), total));
        if let Some(cli) = cand.cli_name() {
            writer_picks.push((cli, cand.pick_rank(), total));
        }
        if matches!(cand, Candidate::RealDisk) {
            real_cost = total;
        }
    }

    ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("\n=== RANKING (ascending weighted cost; lower is better) ===");
    for (n, c) in &ranking {
        println!("  {:<32} {:.4}", n, c);
    }

    // Auto-pick: lowest-cost ordering the WRITER can produce (ties -> Hilbert).
    // Machine-readable line for the batch optimizer to parse.
    writer_picks.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap().then(a.1.cmp(&b.1)));
    if let Some(&(cli, _, cost)) = writer_picks.first() {
        let imp = if real_cost.is_finite() && real_cost > 0.0 {
            100.0 * (real_cost - cost) / real_cost
        } else {
            0.0
        };
        println!(
            "\nPICK {} cost {:.4} real {:.4} improvement {:.1}%",
            cli, cost, real_cost, imp
        );
    }
    Ok(())
}
