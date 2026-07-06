//! Measured blob-ordering picker: simulate per-ordering HTTP range-read cost.
//!
//! [`BlobOrdering::choose`](crate::curve::BlobOrdering::choose) is a cheap
//! cardinality heuristic. This module is the honest alternative it references:
//! over the real native-tier tiles, lay each candidate ordering's byte string
//! down (with the writer's exact total sort key), simulate the two canonical
//! access patterns — *scrub a viewport across all time* and *pan one instant
//! across space* — under HTTP range coalescing, and rank the orderings by the
//! range-read cost they actually incur. It is a faithful Rust port of the
//! showcase probe (`examples/showcase/src/lib/spaceCurve.ts`), sharpened to sort
//! with the production [`crate::curve::space_time_key`] so the simulated layout
//! is byte-identical to what the writer lays down.
//!
//! Two callers share it: the opt-in `--blob-ordering measured` build mode
//! (`PackWriter::with_measured_ordering`) and the `stt-optimize order-audit`
//! subcommand. It is a **pure, deterministic** function of its inputs — all
//! integer arithmetic, stable sorts, no RNG — because the resolved ordering
//! sets every content-addressed pack name; any nondeterminism would churn pack
//! hashes and break immutable-CDN caching.

use std::collections::{BTreeMap, HashSet};

use crate::curve::{self, BlobOrdering};

/// Default range-read coalescing gap: two needed blobs fuse into one request
/// when at most this many **unneeded** bytes lie between them. Matches the
/// production reader's `DEFAULT_RANGE_COALESCE_GAP` (packages/core, archive.ts)
/// so the simulation models what the client actually does.
pub const DEFAULT_COALESCE_GAP_BYTES: u64 = 2 * 1024 * 1024;

/// The orderings evaluated, in **final-tiebreak priority order**: a genuine
/// cost tie resolves to the earliest entry, so `SpatialMajor` wins ties (which
/// reproduces the shallow-time `choose` rule for free) and `Morton3` sits
/// dead-last and can never win a tie.
pub const CANDIDATES: [BlobOrdering; 4] = [
    BlobOrdering::SpatialMajor,
    BlobOrdering::Hilbert3,
    BlobOrdering::TimeMajor,
    BlobOrdering::Morton3,
];

/// One tile as the simulator sees it. `len` is the blob byte weight — the
/// uncompressed `payload.len()` at finalize, or the compressed `entry.length`
/// in the advisor; because range *count* (the primary cost) is weight-insensitive
/// both call sites pick the same winner.
#[derive(Debug, Clone, Copy)]
pub struct TileSample {
    pub z: u8,
    pub x: u32,
    pub y: u32,
    pub hilbert: u64,
    pub time_start: i64,
    /// Time-bucket index (`time_start / bucket_ms`).
    pub tb: i64,
    pub len: u64,
}

/// Simulation knobs. The build path always uses the default so pack hashes stay
/// reproducible; `coalesce_gap_bytes` is exposed for offline sweeps.
#[derive(Debug, Clone, Copy)]
pub struct SimOptions {
    pub coalesce_gap_bytes: u64,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self { coalesce_gap_bytes: DEFAULT_COALESCE_GAP_BYTES }
    }
}

/// Cost of one access pattern under one ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryCost {
    /// Coalesced range requests.
    pub reads: u64,
    /// Bytes actually transferred (needed + over-read swept up inside fused runs).
    pub bytes_read: u64,
    /// Bytes the query genuinely needs (ordering-independent).
    pub bytes_needed: u64,
}

impl QueryCost {
    const ZERO: QueryCost = QueryCost { reads: 0, bytes_read: 0, bytes_needed: 0 };
}

/// Aggregate cost of one ordering across the canonical query mix.
#[derive(Debug, Clone, Copy)]
pub struct OrderingCost {
    pub ordering: BlobOrdering,
    pub scrub: QueryCost,
    pub pan: QueryCost,
    pub total_reads: u64,
    pub total_bytes_read: u64,
    /// Blended cost in bytes-equivalent: `total_bytes_read + total_reads * gap`.
    /// The reader over-reads up to `gap` bytes to save one request, so it prices
    /// a request at exactly `gap` bytes — this is the same trade the coalescer
    /// makes, so ranking by it is self-consistent. Request count *alone* is a
    /// broken proxy (fusing an entire archive into "1 read" of hundreds of MiB
    /// would look free); bytes *alone* ignores per-request latency.
    pub cost: u64,
}

/// Orderings the pickers may *select*. `Morton3` is deliberately absent — it is
/// offered only as an explicit `--blob-ordering morton3` for research; its long
/// jumps hurt locality and its rare measured "wins" are marginal over-read edges
/// on tiny datasets. It is still `evaluate`d and reported for comparison.
pub const SELECTABLE: [BlobOrdering; 3] = [
    BlobOrdering::SpatialMajor,
    BlobOrdering::Hilbert3,
    BlobOrdering::TimeMajor,
];

/// Rank every candidate ordering cheapest-first for these tiles by the blended
/// [`OrderingCost::cost`] (bytes read + reads × gap), with the stable sort
/// keeping `CANDIDATES` order as the final tiebreak (so `SpatialMajor` wins
/// genuine ties). `Morton3` is included for reporting but never *selected* by
/// [`measured_ordering`] (see [`SELECTABLE`]).
pub fn evaluate(samples: &[TileSample], opts: SimOptions) -> Vec<OrderingCost> {
    let gap = opts.coalesce_gap_bytes;
    if samples.is_empty() {
        return CANDIDATES
            .iter()
            .map(|&ordering| OrderingCost {
                ordering,
                scrub: QueryCost::ZERO,
                pan: QueryCost::ZERO,
                total_reads: 0,
                total_bytes_read: 0,
                cost: 0,
            })
            .collect();
    }

    let (tb_min, tb_max) = samples
        .iter()
        .fold((i64::MAX, i64::MIN), |(lo, hi), s| (lo.min(s.tb), hi.max(s.tb)));
    let tb_span = tb_max - tb_min;

    let scrub = scrub_hits(samples);
    let pan = pan_hits(samples);

    let mut costs: Vec<OrderingCost> = CANDIDATES
        .iter()
        .map(|&ordering| {
            let order = linearize(samples, ordering, tb_min, tb_span);
            let sc = simulate_query(&order, samples, &scrub, gap);
            let pn = simulate_query(&order, samples, &pan, gap);
            let total_reads = sc.reads + pn.reads;
            let total_bytes_read = sc.bytes_read + pn.bytes_read;
            OrderingCost {
                ordering,
                scrub: sc,
                pan: pn,
                total_reads,
                total_bytes_read,
                cost: total_bytes_read.saturating_add(total_reads.saturating_mul(gap)),
            }
        })
        .collect();

    // Stable sort by blended cost → ties keep CANDIDATES order (Spatial first).
    costs.sort_by(|a, b| a.cost.cmp(&b.cost));
    costs
}

/// The measured-best **selectable** ordering for these tiles — the cheapest that
/// is not `Morton3` (`SpatialMajor` on empty input). `Morton3` is reported by
/// [`evaluate`] but never chosen (see [`SELECTABLE`]).
pub fn measured_ordering(samples: &[TileSample], opts: SimOptions) -> BlobOrdering {
    evaluate(samples, opts)
        .into_iter()
        .map(|c| c.ordering)
        .find(|o| *o != BlobOrdering::Morton3)
        .unwrap_or(BlobOrdering::SpatialMajor)
}

/// Index permutation of `samples` in `ordering` byte order, using the writer's
/// exact total sort key (`crates/stt-core/src/pack.rs` finalize) so the
/// simulated layout is byte-identical to what the writer lays down.
fn linearize(samples: &[TileSample], ordering: BlobOrdering, tb_min: i64, tb_span: i64) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..samples.len() as u32).collect();
    idx.sort_by_key(|&i| {
        let s = &samples[i as usize];
        (
            curve::space_time_key(
                ordering, s.z, s.x, s.y, s.hilbert, s.time_start, s.tb, tb_min, tb_span,
            ),
            s.z,
            s.x,
            s.y,
            s.time_start,
        )
    });
    idx
}

/// Simulate one access pattern: walk the linearized blobs, fusing two needed
/// blobs into one request when at most `gap_bytes` unneeded bytes lie between
/// them (else a new request). `bytes_read` sums every blob inside a fused run.
fn simulate_query(order: &[u32], samples: &[TileSample], hit: &[bool], gap_bytes: u64) -> QueryCost {
    let mut reads = 0u64;
    let mut bytes_read = 0u64;
    let mut bytes_needed = 0u64;

    let mut in_run = false;
    let mut run_bytes = 0u64; // needed + swept over-read in the current fused run
    let mut gap_since_hit = 0u64; // unneeded bytes since the last hit (fuse candidate)

    for &i in order {
        let s = &samples[i as usize];
        let h = hit[i as usize];
        if h {
            bytes_needed += s.len;
        }
        if !in_run {
            if h {
                in_run = true;
                run_bytes = s.len;
                gap_since_hit = 0;
            }
            // leading unneeded blobs are simply skipped
        } else if h {
            // fuse: the pending gap becomes over-read, plus this blob
            run_bytes += gap_since_hit + s.len;
            gap_since_hit = 0;
        } else {
            gap_since_hit += s.len;
            if gap_since_hit > gap_bytes {
                // gap too wide — close the run (trailing gap is not transferred)
                reads += 1;
                bytes_read += run_bytes;
                in_run = false;
                run_bytes = 0;
                gap_since_hit = 0;
            }
        }
    }
    if in_run {
        reads += 1;
        bytes_read += run_bytes;
    }
    QueryCost { reads, bytes_read, bytes_needed }
}

/// "Scrub a viewport": a contiguous 2D-Hilbert spatial band (the central quarter
/// of the distinct occupied cells in Hilbert order) read across ALL time — a
/// neighbourhood you zoom to and play through. Rewards keeping each cell's whole
/// timeline byte-contiguous (`SpatialMajor`).
fn scrub_hits(samples: &[TileSample]) -> Vec<bool> {
    let mut hs: Vec<u64> = samples.iter().map(|s| s.hilbert).collect();
    hs.sort_unstable();
    hs.dedup();
    let m = hs.len();
    if m == 0 {
        return vec![false; samples.len()];
    }
    // Central quarter by rank: [m*3/8 .. max(m*5/8, m*3/8 + 1)) — integer math,
    // equal to the FE's floor(m*0.375)..floor(m*0.625) (3/8, 5/8 are exact).
    let lo = m * 3 / 8;
    let hi = (m * 5 / 8).max(lo + 1);
    let band: HashSet<u64> = hs[lo..hi].iter().copied().collect();
    samples.iter().map(|s| band.contains(&s.hilbert)).collect()
}

/// "Pan at one instant": the single densest time bucket (by summed bytes; ties
/// resolve to the smallest bucket for determinism) read across the whole map.
/// Rewards keeping one instant's whole map byte-contiguous (`TimeMajor`).
fn pan_hits(samples: &[TileSample]) -> Vec<bool> {
    let mut by_tb: BTreeMap<i64, u64> = BTreeMap::new();
    for s in samples {
        *by_tb.entry(s.tb).or_insert(0) += s.len;
    }
    let mut best_tb = 0i64;
    let mut best_w = 0u64;
    let mut found = false;
    for (&tb, &w) in by_tb.iter() {
        // BTreeMap iterates ascending tb; strict `>` keeps the smallest on ties.
        if !found || w > best_w {
            best_w = w;
            best_tb = tb;
            found = true;
        }
    }
    samples.iter().map(|s| s.tb == best_tb).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A gap small enough (relative to the fixtures' 1000-byte blobs) that
    // coalescing engages but over-read is penalised — so the blended cost
    // actually discriminates orderings (the 2 MiB default would fuse tiny
    // fixtures to ~1 read for everything).
    const OPTS: SimOptions = SimOptions { coalesce_gap_bytes: 1500 };
    const LEN: u64 = 1000;

    fn s(x: u32, y: u32, hilbert: u64, tb: i64, len: u64) -> TileSample {
        TileSample { z: 8, x, y, hilbert, time_start: tb, tb, len }
    }

    #[test]
    fn deep_time_playback_prefers_spatial() {
        // Two cells, a long shared timeline: playback (scrub a cell across all
        // time) wants each cell's whole timeline byte-contiguous → SpatialMajor.
        // Under time-major the band scatters and coalescing over-reads the whole
        // archive, so the blended cost correctly penalises it.
        let mut samples = Vec::new();
        for t in 0..24i64 {
            samples.push(s(0, 0, 0, t, LEN));
            samples.push(s(1, 0, 1, t, LEN));
        }
        assert_eq!(measured_ordering(&samples, OPTS), BlobOrdering::SpatialMajor);
    }

    #[test]
    fn single_bucket_resolves_to_spatial() {
        // One time bucket → SpatialMajor (pure 2D Hilbert); the degenerate 3D
        // time axis never helps. (Measured-path parity with the F2 choose rule.)
        let samples: Vec<TileSample> = (0..16u32).map(|i| s(i, 0, i as u64, 0, LEN)).collect();
        assert_eq!(measured_ordering(&samples, OPTS), BlobOrdering::SpatialMajor);
    }

    #[test]
    fn morton3_is_reported_but_never_selected() {
        // A range of shapes; whatever `evaluate` ranks, the selected ordering is
        // the cheapest NON-morton3, and morton3 still appears in the report.
        for spread in [1u32, 8, 32] {
            let mut samples = Vec::new();
            for t in 0..8i64 {
                for x in 0..spread {
                    samples.push(s(x, x % 3, (x as u64) * 7 + 1, t, LEN + x as u64));
                }
            }
            let ranked = evaluate(&samples, OPTS);
            assert!(ranked.iter().any(|c| c.ordering == BlobOrdering::Morton3), "morton3 reported");
            let expected = ranked
                .iter()
                .map(|c| c.ordering)
                .find(|o| *o != BlobOrdering::Morton3)
                .unwrap();
            assert_eq!(measured_ordering(&samples, OPTS), expected);
            assert_ne!(measured_ordering(&samples, OPTS), BlobOrdering::Morton3);
        }
    }

    #[test]
    fn cost_is_bytes_plus_reads_times_gap() {
        // The ranking key is the blended cost, not raw request count.
        let samples: Vec<TileSample> = (0..12u32).map(|i| s(i, 0, i as u64, (i % 4) as i64, LEN)).collect();
        for c in evaluate(&samples, OPTS) {
            assert_eq!(c.cost, c.total_bytes_read + c.total_reads * OPTS.coalesce_gap_bytes);
        }
    }

    #[test]
    fn evaluate_is_deterministic() {
        let mut samples = Vec::new();
        for t in 0..10i64 {
            for x in 0..5u32 {
                samples.push(s(x, x % 2, (x as u64) * 3 + 1, t, 70 + t as u64));
            }
        }
        let a = evaluate(&samples, SimOptions::default());
        let b = evaluate(&samples, SimOptions::default());
        let key = |v: &[OrderingCost]| {
            v.iter().map(|c| (c.ordering.as_str(), c.cost)).collect::<Vec<_>>()
        };
        assert_eq!(key(&a), key(&b));
    }

    #[test]
    fn empty_input_is_safe_and_spatial() {
        assert_eq!(measured_ordering(&[], SimOptions::default()), BlobOrdering::SpatialMajor);
        assert_eq!(evaluate(&[], SimOptions::default()).len(), 4);
    }

    #[test]
    fn linearize_matches_spatial_major_hand_sort() {
        // SpatialMajor key = (hilbert, time_start): grouped by hilbert then time.
        let samples = vec![s(1, 0, 5, 2, 10), s(0, 0, 1, 1, 10), s(1, 0, 5, 0, 10)];
        let order = linearize(&samples, BlobOrdering::SpatialMajor, 0, 2);
        // hilbert 1 (idx1) first; then hilbert 5 by time_start: idx2 (t0) then idx0 (t2).
        assert_eq!(order, vec![1, 2, 0]);
    }
}
