//! Measured blob-ordering picker: simulate per-ordering HTTP range-read cost.
//!
//! [`BlobOrdering::choose`](crate::curve::BlobOrdering::choose) is a cheap
//! cardinality heuristic. This module is the honest alternative it references:
//! over the real native-tier tiles, lay each candidate ordering's byte string
//! down (with the writer's exact total sort key), simulate the three canonical
//! access patterns — *scrub a viewport across all time*, *pan one instant
//! across space*, and *play a sliding time window through that viewport* —
//! under HTTP range coalescing, and rank the orderings by the range-read cost
//! they actually incur. It is a faithful Rust port of the showcase probe
//! (`poopdeck:examples/showcase/src/lib/spaceCurve.ts`), sharpened to sort with the
//! production [`crate::curve::space_time_key`] so the simulated layout is
//! byte-identical to what the writer lays down.
//!
//! The third (playback) query is the M4 workload-model addition. Until it was
//! added the picker priced only scrub + pan, and the omission was patched by
//! prose — `stt-optimize order-audit` printed a "spatial is not playback-optimal"
//! caveat next to the ranking instead of inside it. It is now a cost row:
//! [`simulate_playback`] prices the delta-fetch a player issues each time the
//! playhead advances one bucket, and [`PlaybackCost::worst_advance_cost`] is the
//! stall proxy (the buffered-runway term). Weights come from
//! [`workload_weights`], a pure function of the dataset's `layer_hint` and its
//! distinct bucket count.
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
/// production reader's `DEFAULT_RANGE_COALESCE_GAP` (poopdeck:packages/core, archive.ts)
/// so the simulation models what the client actually does.
pub const DEFAULT_COALESCE_GAP_BYTES: u64 = 2 * 1024 * 1024;

/// Default pack-object target: the `stt-build` default (`--pack-size 64`). The
/// reader coalesces **per pack** — a range request can never bridge two pack
/// objects (`poopdeck:packages/core`, archive.ts: "Coalescing is per-pack") — so the
/// simulator force-closes a coalesced run at every pack boundary. Without this
/// a scattered query fused into a single archive-spanning read, which no client
/// can issue (it modelled spatial-major's whole-map pan as one 500 MB+ read and
/// so systematically over-preferred time-major on space-wide datasets).
pub const DEFAULT_PACK_BYTES: u64 = 64 * 1024 * 1024;

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

/// Default sliding-window width for the playback query, in time buckets: the
/// window the playhead holds resident. `1` = the minimal player (one bucket on
/// screen, one bucket fetched per advance). Widening it only changes the FIRST
/// advance (the window prime); steady-state advances are always one bucket.
pub const DEFAULT_PLAYBACK_WINDOW_BUCKETS: u64 = 1;

/// Default weight on [`PlaybackCost::worst_advance_cost`] relative to the
/// per-query blended cost: the production reader keeps a ~4×`timeWindow`
/// prefetch runway ahead of the playhead (the flow-and-riders campaign's
/// measured runway), so a single advance that cannot complete inside one bucket
/// of playhead time costs the player about that many windows of stall risk.
///
/// A fit target, not a law — it is recorded in the archive's
/// `orderingWorkload` block precisely so a later re-fit can tell which archives
/// were laid out under which constant.
pub const DEFAULT_RUNWAY_MULTIPLIER: u64 = 4;

/// Below this many native-tier tiles the measured picker has nothing to
/// measure — the canonical band degenerates to one or two cells and the
/// ranking is decided by rounding. `measured` falls back to `auto`'s
/// cardinality heuristic there. Mirrors the pre-build layout advisor's own
/// `MIN_TILES_TO_SIMULATE` so both pickers give up at the same size.
pub const MIN_TILES_TO_SIMULATE: usize = 8;

/// Relative weights of the canonical query families in the blended ranking.
///
/// Never a zero-weight scrub/pan pair: every dataset pays its pan and scrub
/// terms, so no ordering can win by starving a query family. Only `playback`
/// may be zero, and only for a dataset with no playback dimension at all
/// (a single time bucket).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryWeights {
    pub scrub: u32,
    pub pan: u32,
    pub playback: u32,
}

/// The two-query weighting every pick used before the M4 workload model:
/// scrub + pan, playback unpriced. Kept as the documented fallback — it is what
/// `--ordering-workload legacy` selects and what [`SimOptions::default`]
/// still means, so an unmodified caller's ranking is bit-for-bit unchanged.
pub const LEGACY_WEIGHTS: QueryWeights = QueryWeights {
    scrub: 1,
    pan: 1,
    playback: 0,
};

impl Default for QueryWeights {
    fn default() -> Self {
        LEGACY_WEIGHTS
    }
}

/// Which weighting the writer applies when it resolves a `measured` ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrderingWorkloadMode {
    /// Derive the weights per dataset from `layer_hint` + bucket count
    /// ([`workload_weights`]). The default.
    #[default]
    Derived,
    /// Force [`LEGACY_WEIGHTS`] — the pre-M4 two-query ranking. The escape
    /// hatch (`stt-build --ordering-workload legacy`) for reproducing a pick
    /// made before the workload model existed.
    Legacy,
}

/// Per-dataset query weights, as a **pure function of build inputs the pipeline
/// already computes**: the dominant produced layer kind (`layer_hint`, baked
/// into `metadata.style_hints` on every non-streaming build) and the number of
/// distinct time buckets in the native tier.
///
/// The table (constants are fit targets, not laws):
///
/// | `distinct_buckets` | `layer_hint`              | weights `(scrub, pan, playback)` |
/// |---|---|---|
/// | `<= 1`             | any                       | `(1, 1, 0)` — no playback dimension exists |
/// | `> 1`              | `trips` \| `points`       | `(1, 1, 2)` — playback-dominant, the project's headline use case |
/// | `> 1`              | `paths` \| `polygons` \| absent \| unknown | `(1, 1, 1)` — generalist |
///
/// The `layer_hint` vocabulary is deliberately open (a newer writer may emit a
/// kind an older reader has never seen), so every unrecognised string falls to
/// the generalist row rather than erroring.
///
/// Determinism: `layer_hint` is itself a deterministic function of the input
/// features (majority kind with a fixed tie order), so identical rebuilds
/// derive identical weights and therefore identical pack names.
pub fn workload_weights(layer_hint: Option<&str>, distinct_buckets: usize) -> QueryWeights {
    if distinct_buckets <= 1 {
        // Same rule the order-audit caveat used: one bucket = nothing to play.
        return LEGACY_WEIGHTS;
    }
    match layer_hint {
        Some("trips") | Some("points") => QueryWeights {
            scrub: 1,
            pan: 1,
            playback: 2,
        },
        _ => QueryWeights {
            scrub: 1,
            pan: 1,
            playback: 1,
        },
    }
}

/// Simulation knobs. The build path passes the archive's real pack target so the
/// simulated layout matches the packs the writer cuts; `coalesce_gap_bytes` is
/// exposed for offline sweeps.
///
/// `coalesce_gap_bytes` is **reader-mirroring**, not a tuning dial: it is the
/// gap the production client actually fuses across, and the simulator prices at
/// the BUILD-ASSUMED gap so a layout and the reader that reads it agree. It is
/// recorded alongside the weights in the archive so a later reader-side gap
/// change can be detected as drift rather than silently invalidating the layout.
#[derive(Debug, Clone, Copy)]
pub struct SimOptions {
    pub coalesce_gap_bytes: u64,
    /// Pack-object target (bytes). Coalesced runs are force-closed at each pack
    /// boundary because the reader never bridges two pack objects. Defaults to
    /// [`DEFAULT_PACK_BYTES`]; the writer/audit pass the archive's actual target.
    pub pack_bytes: u64,
    /// Relative weights of the three canonical queries in the blended ranking.
    /// Defaults to [`LEGACY_WEIGHTS`] so a caller that constructs `SimOptions`
    /// by functional update gets exactly the pre-M4 ranking.
    pub weights: QueryWeights,
    /// Sliding-window width (time buckets) for the playback query. See
    /// [`DEFAULT_PLAYBACK_WINDOW_BUCKETS`].
    pub playback_window_buckets: u64,
    /// Multiplier on [`PlaybackCost::worst_advance_cost`] — the buffered-runway
    /// term. See [`DEFAULT_RUNWAY_MULTIPLIER`].
    pub runway_multiplier: u64,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self {
            coalesce_gap_bytes: DEFAULT_COALESCE_GAP_BYTES,
            pack_bytes: DEFAULT_PACK_BYTES,
            weights: LEGACY_WEIGHTS,
            playback_window_buckets: DEFAULT_PLAYBACK_WINDOW_BUCKETS,
            runway_multiplier: DEFAULT_RUNWAY_MULTIPLIER,
        }
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
    const ZERO: QueryCost = QueryCost {
        reads: 0,
        bytes_read: 0,
        bytes_needed: 0,
    };
}

/// Cost of the playback query: the summed cost of every playhead advance, plus
/// the worst single advance (the buffered-runway / stall term).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaybackCost {
    /// Summed over every advance. `bytes_needed` equals the band's total bytes
    /// exactly once — playback never refetches an already-buffered bucket.
    pub total: QueryCost,
    /// `max` over advances of the blended cost `bytes_k + gap * reads_k`.
    ///
    /// The player must complete each advance inside one bucket of playhead time
    /// while holding a multi-window runway, so the *worst* advance — not the
    /// average — is what stalls the playhead. This is the corrective the
    /// register demands: the rejected alternative is a whole-map-instant
    /// surrogate, which prices an access no client ever issues.
    pub worst_advance_cost: u64,
}

/// Aggregate cost of one ordering across the canonical query mix.
#[derive(Debug, Clone, Copy)]
pub struct OrderingCost {
    pub ordering: BlobOrdering,
    pub scrub: QueryCost,
    pub pan: QueryCost,
    /// The playback query, summed over playhead advances. Computed and reported
    /// **unconditionally**, even at `weights.playback == 0`, so an audit always
    /// has the number before any weighting flip.
    pub playback: QueryCost,
    /// [`PlaybackCost::worst_advance_cost`] for this ordering.
    pub playback_worst_advance: u64,
    /// The legacy two-query read total (scrub + pan). Retained verbatim for
    /// report continuity; the playback family is reported in its own columns
    /// and never folded in here.
    pub total_reads: u64,
    /// The legacy two-query byte total (scrub + pan) — see [`Self::total_reads`].
    pub total_bytes_read: u64,
    /// Blended, weighted cost in bytes-equivalent:
    /// `Σ_q w_q·(bytes_q + gap·reads_q) + w_playback·runway_multiplier·worst_advance_cost`.
    ///
    /// The reader over-reads up to `gap` bytes to save one request, so it prices
    /// a request at exactly `gap` bytes — this is the same trade the coalescer
    /// makes, so ranking by it is self-consistent. Request count *alone* is a
    /// broken proxy (fusing an entire archive into "1 read" of hundreds of MiB
    /// would look free — the 669 MiB incident); bytes *alone* ignores
    /// per-request latency. Weighting scales the blended terms; it never
    /// reintroduces a raw-count ranking.
    ///
    /// With [`LEGACY_WEIGHTS`] this reduces **exactly** to the pre-M4 formula
    /// `total_bytes_read + total_reads * gap`.
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
                playback: QueryCost::ZERO,
                playback_worst_advance: 0,
                total_reads: 0,
                total_bytes_read: 0,
                cost: 0,
            })
            .collect();
    }

    let (tb_min, tb_max) = samples.iter().fold((i64::MAX, i64::MIN), |(lo, hi), s| {
        (lo.min(s.tb), hi.max(s.tb))
    });
    let tb_span = tb_max - tb_min;

    // All three canonical queries read the SAME bounded viewport (the central
    // spatial band): scrub sweeps all time within it, pan reads one instant
    // within it, playback walks a sliding window through it. Keeping their
    // spatial extent symmetric is what stops pan from dominating, and is what
    // makes the playback query client-realizable rather than a whole-map
    // surrogate (see `central_band`).
    let band = central_band(samples);
    let scrub = scrub_hits(samples, &band);
    let pan = pan_hits(samples, &band);

    let window = opts.playback_window_buckets.max(1);
    let w = opts.weights;
    let pack_bytes = opts.pack_bytes.max(1);
    let mut costs: Vec<OrderingCost> = CANDIDATES
        .iter()
        .map(|&ordering| {
            let order = linearize(samples, ordering, tb_min, tb_span);
            // Pack assignment for THIS ordering: the writer cuts packs greedily
            // in byte order, so a coalesced run can't span a pack boundary.
            let packs = pack_assignment(&order, samples, pack_bytes);
            let sc = simulate_query(&order, samples, &scrub, gap, &packs);
            let pn = simulate_query(&order, samples, &pan, gap, &packs);
            let pb = simulate_playback(&order, samples, &band, gap, &packs, window);
            let total_reads = sc.reads + pn.reads;
            let total_bytes_read = sc.bytes_read + pn.bytes_read;
            let cost = blend(&sc, gap)
                .saturating_mul(u64::from(w.scrub))
                .saturating_add(blend(&pn, gap).saturating_mul(u64::from(w.pan)))
                .saturating_add(blend(&pb.total, gap).saturating_mul(u64::from(w.playback)))
                .saturating_add(
                    u64::from(w.playback)
                        .saturating_mul(opts.runway_multiplier)
                        .saturating_mul(pb.worst_advance_cost),
                );
            OrderingCost {
                ordering,
                scrub: sc,
                pan: pn,
                playback: pb.total,
                playback_worst_advance: pb.worst_advance_cost,
                total_reads,
                total_bytes_read,
                cost,
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
fn linearize(
    samples: &[TileSample],
    ordering: BlobOrdering,
    tb_min: i64,
    tb_span: i64,
) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..samples.len() as u32).collect();
    idx.sort_by_key(|&i| {
        let s = &samples[i as usize];
        (
            curve::space_time_key(
                ordering,
                s.z,
                s.x,
                s.y,
                s.hilbert,
                s.time_start,
                s.tb,
                tb_min,
                tb_span,
            ),
            s.z,
            s.x,
            s.y,
            s.time_start,
        )
    });
    idx
}

/// Greedy pack index per linearized position: the writer streams blobs into the
/// current pack in byte order and cuts a new one when the next blob would exceed
/// `pack_bytes` (matches `PackStreamWriter` in pack.rs finalize). `packs[k]` is
/// the pack holding `order[k]`. Dedup is not modelled here (nor anywhere in this
/// simulator), so this is the same cut the writer makes on the pre-dedup stream.
fn pack_assignment(order: &[u32], samples: &[TileSample], pack_bytes: u64) -> Vec<u64> {
    let mut packs = vec![0u64; order.len()];
    let mut cur_pack = 0u64;
    let mut fill = 0u64;
    for (k, &i) in order.iter().enumerate() {
        let len = samples[i as usize].len;
        if fill > 0 && fill + len > pack_bytes {
            cur_pack += 1;
            fill = 0;
        }
        packs[k] = cur_pack;
        fill += len;
    }
    packs
}

/// Simulate one access pattern: walk the linearized blobs, fusing two needed
/// blobs into one request when at most `gap_bytes` unneeded bytes lie between
/// them (else a new request). A run is ALSO force-closed at every pack boundary
/// (`packs[k]` changes) because the reader coalesces per-pack and a range can
/// never bridge two pack objects. `bytes_read` sums every blob inside a fused run.
fn simulate_query(
    order: &[u32],
    samples: &[TileSample],
    hit: &[bool],
    gap_bytes: u64,
    packs: &[u64],
) -> QueryCost {
    let mut reads = 0u64;
    let mut bytes_read = 0u64;
    let mut bytes_needed = 0u64;

    let mut in_run = false;
    let mut run_bytes = 0u64; // needed + swept over-read in the current fused run
    let mut gap_since_hit = 0u64; // unneeded bytes since the last hit (fuse candidate)
    let mut run_pack = 0u64; // pack the current run started in

    for (k, &i) in order.iter().enumerate() {
        let s = &samples[i as usize];
        let h = hit[i as usize];
        // A coalesced range can't bridge two pack objects: crossing into a new
        // pack closes any open run before this blob is considered.
        if in_run && packs[k] != run_pack {
            reads += 1;
            bytes_read += run_bytes;
            in_run = false;
            run_bytes = 0;
            gap_since_hit = 0;
        }
        if h {
            bytes_needed += s.len;
        }
        if !in_run {
            if h {
                in_run = true;
                run_bytes = s.len;
                gap_since_hit = 0;
                run_pack = packs[k];
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
    QueryCost {
        reads,
        bytes_read,
        bytes_needed,
    }
}

/// The blended bytes-equivalent cost of one query: `bytes_read + reads * gap`.
/// The single place the request/byte trade is priced — ranking by request count
/// alone is a standing rejection (it once made a 669 MiB whole-pack over-read
/// look like the cheapest layout because it was "2 reads").
fn blend(q: &QueryCost, gap_bytes: u64) -> u64 {
    q.bytes_read
        .saturating_add(q.reads.saturating_mul(gap_bytes))
}

/// Simulate the **playback** query: the central `band` played through as a
/// sliding window of `window_buckets` consecutive time buckets, advanced one
/// bucket at a time from the band's first occupied bucket to its last.
///
/// Structure, and how it differs from [`simulate_query`]'s one-shot patterns:
///
/// * **Delta fetch.** Advance *k* fetches exactly the band's tiles in the
///   bucket(s) newly entering the window — the trailing buckets are already
///   buffered and are never refetched. Summed over every advance, each band
///   blob is needed exactly once. (The first advance is the window *prime*: it
///   admits all `window_buckets` starting buckets at once. Every later advance
///   admits exactly one.)
/// * **No cross-bucket coalescing.** A range request cannot fuse across a
///   bucket the playhead has not reached yet, because that fetch has not been
///   issued yet. So each advance is costed as its own coalesced read set — the
///   whole-map-instant surrogate (fuse everything, call it one read) is exactly
///   the rejected model this replaces.
///
/// Cost is `O(n + Σ hits)`, not `O(n × buckets)`: one pass builds prefix sums of
/// blob lengths and per-bucket hit positions, and each advance's coalesced runs
/// are then read off those prefix sums in `O(hits)`.
///
/// `packs[k]` is the pack holding `order[k]` (see `pack_assignment`); runs are
/// force-closed wherever it changes, because the reader coalesces per pack.
pub fn simulate_playback(
    order: &[u32],
    samples: &[TileSample],
    band: &HashSet<u64>,
    gap_bytes: u64,
    packs: &[u64],
    window_buckets: u64,
) -> PlaybackCost {
    // prefix[k] = total blob bytes strictly before linearized position k, so the
    // unneeded bytes between two hits a < b are `prefix[b] - prefix[a + 1]` and
    // a fused run a..=b transfers `prefix[b + 1] - prefix[a]`.
    let mut prefix: Vec<u64> = Vec::with_capacity(order.len() + 1);
    prefix.push(0);
    // boundaries[k] = number of pack changes at or before position k, so "no
    // pack boundary lies between a and b" is the O(1) test
    // `boundaries[a] == boundaries[b]`.
    let mut boundaries: Vec<u32> = Vec::with_capacity(order.len());
    let mut seen = 0u32;
    let mut prev_pack: Option<u64> = None;
    for (k, &i) in order.iter().enumerate() {
        let len = samples[i as usize].len;
        prefix.push(prefix[k].saturating_add(len));
        let p = packs.get(k).copied().unwrap_or(0);
        if prev_pack.is_some_and(|q| q != p) {
            seen += 1;
        }
        prev_pack = Some(p);
        boundaries.push(seen);
    }

    // Band hits grouped by time bucket. One pass over the linearized order, so
    // each bucket's position list comes out already ascending.
    let mut by_bucket: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (k, &i) in order.iter().enumerate() {
        let s = &samples[i as usize];
        if band.contains(&s.hilbert) {
            by_bucket.entry(s.tb).or_default().push(k);
        }
    }
    let Some(&first_tb) = by_bucket.keys().next() else {
        return PlaybackCost {
            total: QueryCost::ZERO,
            worst_advance_cost: 0,
        };
    };

    // The window prime covers [first_tb, first_tb + window - 1]; buckets with no
    // band tiles simply contribute no advance (a zero-cost advance can never be
    // the worst one, so skipping them is exact, not an approximation).
    let span = i64::try_from(window_buckets.max(1) - 1).unwrap_or(i64::MAX);
    let prime_end = first_tb.saturating_add(span);
    let mut prime: Vec<usize> = Vec::new();
    let mut advances: Vec<Vec<usize>> = Vec::new();
    for (tb, positions) in by_bucket {
        if tb <= prime_end {
            prime.extend(positions);
        } else {
            advances.push(positions);
        }
    }
    // The prime merges several already-ascending lists; sort restores linearized
    // order. Positions are distinct, so the sort is total (no tiebreak needed).
    prime.sort_unstable();

    let mut total = QueryCost::ZERO;
    let mut worst = 0u64;
    let mut price = |positions: &[usize]| {
        let c = coalesced_cost(positions, order, samples, &prefix, &boundaries, gap_bytes);
        total.reads = total.reads.saturating_add(c.reads);
        total.bytes_read = total.bytes_read.saturating_add(c.bytes_read);
        total.bytes_needed = total.bytes_needed.saturating_add(c.bytes_needed);
        worst = worst.max(blend(&c, gap_bytes));
    };
    price(&prime);
    for positions in &advances {
        price(positions);
    }

    PlaybackCost {
        total,
        worst_advance_cost: worst,
    }
}

/// Coalesced cost of one hit set given as **ascending linearized positions**.
///
/// Reproduces [`simulate_query`]'s fuse rule exactly, read off prefix sums
/// instead of walked blob-by-blob: two consecutive hits fuse when no pack
/// boundary lies between them AND at most `gap_bytes` unneeded bytes do.
/// (`simulate_query` closes the run the instant its running gap exceeds the
/// budget; since that running sum is monotone, "closed somewhere in between" is
/// exactly "total unneeded bytes > gap_bytes".)
fn coalesced_cost(
    positions: &[usize],
    order: &[u32],
    samples: &[TileSample],
    prefix: &[u64],
    boundaries: &[u32],
    gap_bytes: u64,
) -> QueryCost {
    let Some(&first) = positions.first() else {
        return QueryCost::ZERO;
    };
    let len_at = |k: usize| samples[order[k] as usize].len;

    let mut reads = 0u64;
    let mut bytes_read = 0u64;
    let mut bytes_needed = len_at(first);
    let mut run_start = first;
    let mut run_last = first;
    for &p in &positions[1..] {
        bytes_needed = bytes_needed.saturating_add(len_at(p));
        let same_pack = boundaries[p] == boundaries[run_last];
        let unneeded = prefix[p].saturating_sub(prefix[run_last + 1]);
        if same_pack && unneeded <= gap_bytes {
            run_last = p;
        } else {
            reads += 1;
            bytes_read = bytes_read.saturating_add(prefix[run_last + 1] - prefix[run_start]);
            run_start = p;
            run_last = p;
        }
    }
    reads += 1;
    bytes_read = bytes_read.saturating_add(prefix[run_last + 1] - prefix[run_start]);
    QueryCost {
        reads,
        bytes_read,
        bytes_needed,
    }
}

/// The bounded viewport all three canonical queries read: the central quarter of
/// the distinct occupied cells in 2D-Hilbert order (a contiguous Hilbert band —
/// the neighbourhood you'd zoom to). ALL queries share it so they stay symmetric
/// in spatial extent. The earlier `pan` read one instant across the ENTIRE map — an
/// access no client issues (a viewport never holds the whole map's tiles at
/// once); under per-pack coalescing that scattered whole-map read fused into
/// whole-pack over-reads, which systematically over-preferred `time-major` on
/// every space-wide dataset. Bounding pan to the same viewport removes that bias.
///
/// Integer math equal to the FE's `floor(m*0.375)..floor(m*0.625)` (3/8, 5/8 are
/// exact) so this and the showcase probe pick the same band.
pub fn central_band(samples: &[TileSample]) -> HashSet<u64> {
    let mut hs: Vec<u64> = samples.iter().map(|s| s.hilbert).collect();
    hs.sort_unstable();
    hs.dedup();
    let m = hs.len();
    if m == 0 {
        return HashSet::new();
    }
    let lo = m * 3 / 8;
    let hi = (m * 5 / 8).max(lo + 1);
    hs[lo..hi].iter().copied().collect()
}

/// "Scrub a viewport": the central spatial `band` read across ALL time — a
/// neighbourhood you zoom to and play through. Rewards keeping each cell's whole
/// timeline byte-contiguous (`SpatialMajor`).
fn scrub_hits(samples: &[TileSample], band: &HashSet<u64>) -> Vec<bool> {
    samples.iter().map(|s| band.contains(&s.hilbert)).collect()
}

/// "Pan at one instant": the SAME viewport `band` at its single densest time
/// bucket (by summed bytes; ties resolve to the smallest bucket for determinism)
/// — dragging the map at a fixed time, a bounded viewport rather than the whole
/// planet. Rewards keeping one instant's viewport byte-contiguous (`TimeMajor`).
fn pan_hits(samples: &[TileSample], band: &HashSet<u64>) -> Vec<bool> {
    let mut by_tb: BTreeMap<i64, u64> = BTreeMap::new();
    for s in samples {
        if band.contains(&s.hilbert) {
            *by_tb.entry(s.tb).or_insert(0) += s.len;
        }
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
    samples
        .iter()
        .map(|s| band.contains(&s.hilbert) && s.tb == best_tb)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A gap small enough (relative to the fixtures' 1000-byte blobs) that
    // coalescing engages but over-read is penalised — so the blended cost
    // actually discriminates orderings (the 2 MiB default would fuse tiny
    // fixtures to ~1 read for everything).
    const OPTS: SimOptions = SimOptions {
        coalesce_gap_bytes: 1500,
        pack_bytes: 1 << 30,
        weights: LEGACY_WEIGHTS,
        playback_window_buckets: DEFAULT_PLAYBACK_WINDOW_BUCKETS,
        runway_multiplier: DEFAULT_RUNWAY_MULTIPLIER,
    };
    const LEN: u64 = 1000;

    /// `OPTS` with the playback family weighted in.
    fn opts_with_playback(playback: u32) -> SimOptions {
        SimOptions {
            weights: QueryWeights {
                scrub: 1,
                pan: 1,
                playback,
            },
            ..OPTS
        }
    }

    /// A grid of shapes the whole suite reuses: `(cells, buckets)` fixtures
    /// spanning single-bucket, narrow-space/deep-time, and wide-space cases.
    fn fixture(cells: u32, buckets: i64) -> Vec<TileSample> {
        let mut out = Vec::new();
        for t in 0..buckets {
            for x in 0..cells {
                // Varying lengths keep the byte terms from degenerating to
                // "count × LEN" (which would let a count-only ranking pass).
                out.push(s(x, x % 4, x as u64, t, LEN + u64::from(x % 3) * 137));
            }
        }
        out
    }

    fn fixture_grid() -> Vec<Vec<TileSample>> {
        vec![
            fixture(2, 1),
            fixture(8, 1),
            fixture(2, 24),
            fixture(8, 24),
            fixture(16, 6),
            fixture(5, 37),
            fixture(9, 9),
        ]
    }

    fn s(x: u32, y: u32, hilbert: u64, tb: i64, len: u64) -> TileSample {
        TileSample {
            z: 8,
            x,
            y,
            hilbert,
            time_start: tb,
            tb,
            len,
        }
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
        assert_eq!(
            measured_ordering(&samples, OPTS),
            BlobOrdering::SpatialMajor
        );
    }

    #[test]
    fn single_bucket_resolves_to_spatial() {
        // One time bucket → SpatialMajor (pure 2D Hilbert); the degenerate 3D
        // time axis never helps. (Measured-path parity with the F2 choose rule.)
        let samples: Vec<TileSample> = (0..16u32).map(|i| s(i, 0, i as u64, 0, LEN)).collect();
        assert_eq!(
            measured_ordering(&samples, OPTS),
            BlobOrdering::SpatialMajor
        );
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
            assert!(
                ranked.iter().any(|c| c.ordering == BlobOrdering::Morton3),
                "morton3 reported"
            );
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
        let samples: Vec<TileSample> = (0..12u32)
            .map(|i| s(i, 0, i as u64, (i % 4) as i64, LEN))
            .collect();
        for c in evaluate(&samples, OPTS) {
            assert_eq!(
                c.cost,
                c.total_bytes_read + c.total_reads * OPTS.coalesce_gap_bytes
            );
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
        // Every reported field, at legacy AND at nonzero playback weights: the
        // ordering names every content-addressed pack, so a single unstable
        // byte here would churn the whole fleet.
        let key = |v: &[OrderingCost]| {
            v.iter()
                .map(|c| {
                    (
                        c.ordering.as_str(),
                        c.cost,
                        c.total_reads,
                        c.total_bytes_read,
                        c.playback.reads,
                        c.playback.bytes_read,
                        c.playback.bytes_needed,
                        c.playback_worst_advance,
                    )
                })
                .collect::<Vec<_>>()
        };
        for opts in [
            SimOptions::default(),
            opts_with_playback(1),
            opts_with_playback(2),
            SimOptions {
                playback_window_buckets: 4,
                ..opts_with_playback(3)
            },
        ] {
            let a = evaluate(&samples, opts);
            let b = evaluate(&samples, opts);
            assert_eq!(key(&a), key(&b));
            assert_eq!(
                measured_ordering(&samples, opts),
                measured_ordering(&samples, opts)
            );
        }
    }

    // ---- WM-1: the playback canonical query ----------------------------

    #[test]
    fn playback_delta_fetch_never_refetches() {
        // Summed over every advance, the playback query needs each band blob
        // exactly once — a trailing bucket that is already buffered is never
        // refetched. That total is ordering-INDEPENDENT (it is the band), so it
        // is also the guard that no ordering is credited with fetching less.
        for samples in fixture_grid() {
            let band = central_band(&samples);
            let band_bytes: u64 = samples
                .iter()
                .filter(|s| band.contains(&s.hilbert))
                .map(|s| s.len)
                .sum();
            for c in evaluate(&samples, opts_with_playback(1)) {
                assert_eq!(
                    c.playback.bytes_needed,
                    band_bytes,
                    "{} refetched or dropped band bytes",
                    c.ordering.as_str()
                );
            }
        }
    }

    #[test]
    fn playback_hits_are_confined_to_band() {
        // Guard on the whole-map-instant ban: every blob the playback query
        // counts is inside `central_band`, so its needed set is EXACTLY the
        // scrub set (same viewport, walked in time) and strictly smaller than
        // the archive whenever the band is a proper subset.
        for samples in fixture_grid() {
            let total_bytes: u64 = samples.iter().map(|s| s.len).sum();
            let band = central_band(&samples);
            let distinct_cells = {
                let mut hs: Vec<u64> = samples.iter().map(|s| s.hilbert).collect();
                hs.sort_unstable();
                hs.dedup();
                hs.len()
            };
            for c in evaluate(&samples, opts_with_playback(1)) {
                assert_eq!(
                    c.playback.bytes_needed, c.scrub.bytes_needed,
                    "playback must read the same viewport as scrub"
                );
                if distinct_cells > band.len() {
                    assert!(
                        c.playback.bytes_needed < total_bytes,
                        "playback read the whole map — the rejected surrogate"
                    );
                }
            }
        }
    }

    #[test]
    fn spatial_ordering_pays_worst_advance() {
        // The order-audit caveat, now a number. Eight cells over 24 buckets:
        // the central band holds TWO cells (a one-cell band makes every advance
        // a single blob and cannot discriminate — `central_band` takes the
        // middle quarter, so the fixture needs >= 8 distinct cells for that).
        //
        // Under `spatial` each cell's whole timeline is contiguous, so the two
        // band cells sit 24 blobs apart and EVERY advance pays two reads.
        // Under `time-major` the two cells are adjacent within their bucket and
        // every advance fuses into one.
        let samples: Vec<TileSample> = (0..24i64)
            .flat_map(|t| (0..8u32).map(move |x| s(x, 0, u64::from(x), t, LEN)))
            .collect();
        let band = central_band(&samples);
        assert_eq!(band.len(), 2, "fixture must give a two-cell band");

        let ranked = evaluate(&samples, opts_with_playback(2));
        let by = |o: BlobOrdering| ranked.iter().find(|c| c.ordering == o).unwrap();
        let spatial = by(BlobOrdering::SpatialMajor);
        let time_major = by(BlobOrdering::TimeMajor);
        assert!(
            spatial.playback_worst_advance > time_major.playback_worst_advance,
            "spatial {} !> time-major {}",
            spatial.playback_worst_advance,
            time_major.playback_worst_advance
        );
        // ... and the concrete numbers, so a silent model change is caught:
        // spatial 2 reads + 2000 B; time-major 1 read + 2000 B, gap 1500.
        assert_eq!(spatial.playback_worst_advance, 2000 + 2 * 1500);
        assert_eq!(time_major.playback_worst_advance, 2000 + 1500);
    }

    #[test]
    fn default_weights_reproduce_legacy_cost() {
        // BYTE-NEUTRALITY of WM-1 as shipped: with the default (1, 1, 0)
        // weights the blended cost and the ranking are bit-for-bit the pre-M4
        // two-query formula, for every fixture shape.
        for samples in fixture_grid() {
            let ranked = evaluate(&samples, SimOptions::default());
            let gap = SimOptions::default().coalesce_gap_bytes;
            for c in &ranked {
                assert_eq!(
                    c.cost,
                    c.total_bytes_read + c.total_reads * gap,
                    "{} cost drifted from the legacy formula",
                    c.ordering.as_str()
                );
            }
            for pair in ranked.windows(2) {
                assert!(
                    pair[0].cost <= pair[1].cost,
                    "ranking is not cheapest-first"
                );
            }
            // The legacy ranking is also what the tight-gap OPTS produce.
            let tight = evaluate(&samples, OPTS);
            for c in &tight {
                assert_eq!(
                    c.cost,
                    c.total_bytes_read + c.total_reads * OPTS.coalesce_gap_bytes
                );
            }
        }
    }

    /// Reference implementation: run the incumbent `simulate_query` once per
    /// advance, an honest O(n × buckets) walk, and compare it to the prefix-sum
    /// fast path. This is the property that keeps the optimisation from drifting
    /// away from the fuse rule it claims to reproduce.
    fn brute_force_playback(
        order: &[u32],
        samples: &[TileSample],
        band: &HashSet<u64>,
        gap: u64,
        packs: &[u64],
        window: u64,
    ) -> PlaybackCost {
        let mut buckets: Vec<i64> = samples
            .iter()
            .filter(|s| band.contains(&s.hilbert))
            .map(|s| s.tb)
            .collect();
        buckets.sort_unstable();
        buckets.dedup();
        if buckets.is_empty() {
            return PlaybackCost {
                total: QueryCost::ZERO,
                worst_advance_cost: 0,
            };
        }
        let prime_end = buckets[0].saturating_add(i64::try_from(window.max(1) - 1).unwrap());
        let mut advance_sets: Vec<Vec<i64>> = Vec::new();
        let prime: Vec<i64> = buckets
            .iter()
            .copied()
            .filter(|t| *t <= prime_end)
            .collect();
        advance_sets.push(prime);
        for &t in buckets.iter().filter(|t| **t > prime_end) {
            advance_sets.push(vec![t]);
        }

        let mut total = QueryCost::ZERO;
        let mut worst = 0u64;
        for set in advance_sets {
            let hits: Vec<bool> = samples
                .iter()
                .map(|s| band.contains(&s.hilbert) && set.contains(&s.tb))
                .collect();
            let c = simulate_query(order, samples, &hits, gap, packs);
            total.reads += c.reads;
            total.bytes_read += c.bytes_read;
            total.bytes_needed += c.bytes_needed;
            worst = worst.max(c.bytes_read + c.reads * gap);
        }
        PlaybackCost {
            total,
            worst_advance_cost: worst,
        }
    }

    #[test]
    fn playback_fast_path_matches_bruteforce_per_bucket_simulation() {
        // Deterministic pseudo-random shapes (a fixed LCG — no RNG crate, no
        // seed-of-the-day) across window widths, pack sizes and gaps.
        let mut state: u64 = 0x2026_0810;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for case in 0..24 {
            let cells = 1 + next() % 12;
            let buckets = 1 + (next() % 15) as i64;
            let mut samples = Vec::new();
            for t in 0..buckets {
                for x in 0..cells {
                    // Sparse occupancy: some (cell, bucket) pairs are missing,
                    // which is what produces the empty-advance edge cases.
                    if next() % 5 == 0 {
                        continue;
                    }
                    samples.push(s(x, x % 3, u64::from(x), t, 60 + u64::from(next() % 900)));
                }
            }
            if samples.is_empty() {
                continue;
            }
            let (tb_min, tb_max) = samples.iter().fold((i64::MAX, i64::MIN), |(lo, hi), s| {
                (lo.min(s.tb), hi.max(s.tb))
            });
            let band = central_band(&samples);
            for &window in &[1u64, 2, 5] {
                for &gap in &[0u64, 700, 4000] {
                    for &pack_bytes in &[1u64 << 30, 4000, 900] {
                        for ordering in CANDIDATES {
                            let order = linearize(&samples, ordering, tb_min, tb_max - tb_min);
                            let packs = pack_assignment(&order, &samples, pack_bytes);
                            let fast =
                                simulate_playback(&order, &samples, &band, gap, &packs, window);
                            let slow =
                                brute_force_playback(&order, &samples, &band, gap, &packs, window);
                            assert_eq!(
                                fast, slow,
                                "case {case} {ordering:?} window {window} gap {gap} pack {pack_bytes}"
                            );
                        }
                    }
                }
            }
        }
    }

    // ---- WM-2: the workload weight table -------------------------------

    #[test]
    fn weights_table_is_total() {
        // Every (hint, buckets) combination returns a defined triple, scrub and
        // pan are never starved, and an UNKNOWN hint string — the vocabulary is
        // deliberately open — falls to the generalist row rather than panicking.
        let hints = [
            None,
            Some("points"),
            Some("trips"),
            Some("paths"),
            Some("polygons"),
            Some("hexagons"),
            Some(""),
            Some("Points"),
        ];
        for hint in hints {
            for buckets in [0usize, 1, 2, 5, 4096] {
                let w = workload_weights(hint, buckets);
                assert!(
                    w.scrub >= 1 && w.pan >= 1,
                    "{hint:?}/{buckets} starved a family"
                );
                if buckets <= 1 {
                    assert_eq!(w, LEGACY_WEIGHTS, "{hint:?} single-bucket must be legacy");
                }
            }
        }
        // The playback-dominant row is exactly {trips, points}; case-sensitive,
        // because `layer_hint` is emitted lowercase by the builder.
        assert_eq!(workload_weights(Some("trips"), 9).playback, 2);
        assert_eq!(workload_weights(Some("points"), 9).playback, 2);
        assert_eq!(workload_weights(Some("paths"), 9).playback, 1);
        assert_eq!(workload_weights(Some("polygons"), 9).playback, 1);
        assert_eq!(workload_weights(None, 9).playback, 1);
        assert_eq!(workload_weights(Some("hexagons"), 9).playback, 1);
        assert_eq!(workload_weights(Some("Points"), 9).playback, 1);
    }

    #[test]
    fn single_bucket_zeroes_playback() {
        // A snapshot dataset has no playback dimension to price, so the weights
        // collapse to legacy and the cost is bit-for-bit the two-query formula.
        for hint in [None, Some("trips"), Some("points"), Some("polygons")] {
            assert_eq!(workload_weights(hint, 1), LEGACY_WEIGHTS);
            assert_eq!(workload_weights(hint, 0), LEGACY_WEIGHTS);
        }
        let samples = fixture(8, 1);
        let weighted = SimOptions {
            weights: workload_weights(Some("trips"), 1),
            ..OPTS
        };
        for (a, b) in evaluate(&samples, weighted)
            .iter()
            .zip(evaluate(&samples, OPTS).iter())
        {
            assert_eq!(a.ordering, b.ordering);
            assert_eq!(a.cost, b.cost);
        }
    }

    #[test]
    fn weighted_pick_flips_on_playback_dominant_fixture() {
        // Eight cells × 24 buckets: scrub loves `spatial` (each cell's whole
        // timeline is one fused run), so the legacy two-query pick is spatial —
        // and the prose caveat then had to warn that it stalls playback. With
        // the playback family priced IN, the same fixture picks a
        // playback-safe ordering instead. This is the caveat becoming a cost.
        let samples: Vec<TileSample> = (0..24i64)
            .flat_map(|t| (0..8u32).map(move |x| s(x, 0, u64::from(x), t, LEN)))
            .collect();
        assert_eq!(
            measured_ordering(&samples, OPTS),
            BlobOrdering::SpatialMajor,
            "legacy weights must still pick spatial for this fixture"
        );
        let weighted = SimOptions {
            weights: workload_weights(Some("trips"), 24),
            ..OPTS
        };
        assert_eq!(workload_weights(Some("trips"), 24).playback, 2);
        assert_ne!(
            measured_ordering(&samples, weighted),
            BlobOrdering::SpatialMajor,
            "playback-dominant weights must move off the stalling ordering"
        );
    }

    #[test]
    fn playback_weight_never_rescues_the_costliest_playback_ordering() {
        // Monotonicity. `cost_w(o) = base(o) + w·P(o)` where
        // `P(o) = blend(playback) + runway·worst_advance`, so if `P(o*)` is
        // maximal then `cost_w(o*) - cost_w(o)` is non-decreasing in `w` for
        // every `o`: the ordering that costs the most playback can never
        // IMPROVE its rank as playback is weighted more heavily.
        for samples in fixture_grid() {
            let base = evaluate(&samples, opts_with_playback(0));
            let worst_ordering = base
                .iter()
                .max_by_key(|c| {
                    (
                        blend(&c.playback, OPTS.coalesce_gap_bytes)
                            + OPTS.runway_multiplier * c.playback_worst_advance,
                        // Total tiebreak so the choice itself is deterministic.
                        std::cmp::Reverse(
                            CANDIDATES.iter().position(|o| *o == c.ordering).unwrap(),
                        ),
                    )
                })
                .unwrap()
                .ordering;
            let mut prev = 0usize;
            for w in 0..5u32 {
                let ranked = evaluate(&samples, opts_with_playback(w));
                let mine = ranked
                    .iter()
                    .find(|c| c.ordering == worst_ordering)
                    .unwrap()
                    .cost;
                let strictly_cheaper = ranked.iter().filter(|c| c.cost < mine).count();
                assert!(
                    strictly_cheaper >= prev,
                    "{worst_ordering:?} improved from rank {prev} to {strictly_cheaper} at w={w}"
                );
                prev = strictly_cheaper;
            }
        }
    }

    #[test]
    fn empty_input_is_safe_and_spatial() {
        assert_eq!(
            measured_ordering(&[], SimOptions::default()),
            BlobOrdering::SpatialMajor
        );
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
