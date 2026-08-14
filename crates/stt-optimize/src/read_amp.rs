//! DT-5 — interval-segregation read-amplification instrumentation (§2.2).
//!
//! The pathology: single start-bucket placement plus an exact `time_end` bound
//! means one decade-long interval makes its whole tile a permanent fetch for
//! every later window, dragging every co-resident short feature with it.
//!
//! This is the TRIGGER instrument, built before any erratum is drafted. It is a
//! directory-only walk (the fast `--sample 0` class): deterministic, no decode,
//! integer arithmetic throughout.

use serde::{Deserialize, Serialize};

/// Read-amplification report for one sliding-window size.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadAmpReport {
    /// Sliding window width in milliseconds.
    pub window_ms: u64,
    /// Number of windows swept.
    pub windows: u64,
    /// Expected fetched bytes over the sweep, summed across windows.
    pub fetched_bytes: u128,
    /// Of those, bytes attributable to tiles kept resident SOLELY by a
    /// long-lived `time_end` — i.e. the window no longer overlaps the tile's
    /// start bucket, and only its trailing interval keeps it in the answer.
    pub long_lived_bytes: u128,
    /// `long_lived_bytes / fetched_bytes`. The §2.2 trigger quantity.
    pub amplification_share: f64,
    /// Residual-lifetime histogram, in whole windows: `bins[i]` counts entries
    /// whose `time_end` reaches `i` windows past their own bucket start.
    /// The last bin is saturating.
    pub residual_lifetime_bins: Vec<u64>,
}

/// One directory entry's temporal footprint, as the walk needs it.
#[derive(Debug, Clone, Copy)]
pub struct EntrySpan {
    pub time_start: i64,
    pub time_end: i64,
    /// Earliest instant any feature in the tile actually covers, when the
    /// writer recorded it. Absent = `time_start`.
    pub cover_t_min: Option<i64>,
    pub length: u64,
}

/// Number of residual-lifetime bins; the last one saturates.
pub const RESIDUAL_BINS: usize = 16;

/// Compute read amplification over a sliding window of `window_ms`.
///
/// A tile answers a window `[w, w+W)` when its own covered span intersects it.
/// The share attributes bytes to the LONG-LIVED cause when the window starts
/// after the tile's start bucket has ended — that is, the tile is in the answer
/// only because its `time_end` trails into the window.
///
/// Deterministic: integer arithmetic, input order irrelevant (every term is a
/// sum over entries).
pub fn read_amplification(entries: &[EntrySpan], window_ms: u64, bucket_ms: u64) -> ReadAmpReport {
    let mut report = ReadAmpReport {
        window_ms,
        windows: 0,
        fetched_bytes: 0,
        long_lived_bytes: 0,
        amplification_share: 0.0,
        residual_lifetime_bins: vec![0; RESIDUAL_BINS],
    };
    if entries.is_empty() || window_ms == 0 {
        return report;
    }
    let bucket = bucket_ms.max(1) as i64;
    let w = window_ms as i64;

    let t_min = entries
        .iter()
        .map(|e| e.cover_t_min.unwrap_or(e.time_start))
        .min()
        .unwrap();
    let t_max = entries.iter().map(|e| e.time_end).max().unwrap();
    if t_max <= t_min {
        return report;
    }
    let windows = ((t_max - t_min) / w).max(1) as u64;
    report.windows = windows;

    for e in entries {
        let cover_from = e.cover_t_min.unwrap_or(e.time_start);
        // The tile's own bucket: where single start-bucket placement put it.
        let bucket_end = e.time_start.saturating_add(bucket);

        // Windows this entry is fetched by, and the subset where only its
        // trailing `time_end` keeps it resident.
        let first = (cover_from - t_min).div_euclid(w);
        let last = (e.time_end - t_min).div_euclid(w);
        let hits = (last - first + 1).max(0) as u128;
        report.fetched_bytes += hits * u128::from(e.length);

        // A window starting at or after the entry's own bucket end is answered
        // ONLY because of the trailing interval.
        let trailing_first = (bucket_end - t_min).div_euclid(w).max(first);
        let trailing = (last - trailing_first + 1).max(0) as u128;
        report.long_lived_bytes += trailing * u128::from(e.length);

        // Residual lifetime, in whole windows past the entry's own bucket.
        let residual = ((e.time_end - bucket_end).max(0) / w) as usize;
        let bin = residual.min(RESIDUAL_BINS - 1);
        report.residual_lifetime_bins[bin] += 1;
    }

    if report.fetched_bytes > 0 {
        report.amplification_share = report.long_lived_bytes as f64 / report.fetched_bytes as f64;
    }
    report
}

/// The §2.2 adoption threshold, proposed and explicitly not yet fit (P1).
/// No number, no erratum.
pub const READ_AMP_TRIGGER_SHARE: f64 = 0.25;

#[cfg(test)]
mod tests {
    use super::*;

    fn span(start: i64, end: i64, len: u64) -> EntrySpan {
        EntrySpan {
            time_start: start,
            time_end: end,
            cover_t_min: None,
            length: len,
        }
    }

    /// Short-lived features confined to their own bucket amplify nothing.
    #[test]
    fn tightly_bounded_entries_show_no_amplification() {
        let bucket = 1_000;
        let e: Vec<_> = (0..10)
            .map(|i| span(i * bucket, i * bucket + bucket - 1, 100))
            .collect();
        let r = read_amplification(&e, bucket as u64, bucket as u64);
        assert_eq!(r.long_lived_bytes, 0, "{r:?}");
        assert_eq!(r.amplification_share, 0.0);
        // Every entry sits in bin 0 — no residual lifetime.
        assert_eq!(r.residual_lifetime_bins[0], 10);
    }

    /// One decade-long interval drags its bytes across every later window —
    /// the §2.2 pathology, made visible.
    #[test]
    fn one_long_lived_interval_dominates_the_share() {
        let bucket = 1_000;
        let mut e: Vec<_> = (0..10)
            .map(|i| span(i * bucket, i * bucket + bucket - 1, 100))
            .collect();
        // A single feature starting in bucket 0 and ending far past the end.
        e.push(span(0, 100 * bucket, 100));
        let r = read_amplification(&e, bucket as u64, bucket as u64);
        assert!(r.long_lived_bytes > 0, "{r:?}");
        assert!(
            r.amplification_share > READ_AMP_TRIGGER_SHARE,
            "one decade-long interval should dominate: {r:?}"
        );
        // And it lands in the saturating top bin.
        assert_eq!(r.residual_lifetime_bins[RESIDUAL_BINS - 1], 1);
    }

    /// Deterministic and order-independent: every term is a sum over entries.
    #[test]
    fn the_walk_is_order_independent_and_reproducible() {
        let bucket = 500;
        let mut e: Vec<_> = (0..25)
            .map(|i| span(i * bucket, i * bucket + (i % 7) * bucket, 64 + i as u64))
            .collect();
        let a = read_amplification(&e, bucket as u64, bucket as u64);
        e.reverse();
        let b = read_amplification(&e, bucket as u64, bucket as u64);
        assert_eq!(a, b);
        // And re-running is byte-identical.
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }

    /// Degenerate inputs never panic and never divide by zero.
    #[test]
    fn degenerate_inputs_are_inert() {
        assert_eq!(read_amplification(&[], 1000, 1000).amplification_share, 0.0);
        // A zero-width dataset spans no windows at all.
        let point = vec![span(0, 0, 10)];
        assert_eq!(read_amplification(&point, 1000, 1000).windows, 0);
        // A zero window width is inert rather than a division by zero.
        assert_eq!(read_amplification(&point, 0, 1000).amplification_share, 0.0);
        // A zero bucket width falls back to 1 ms rather than dividing by zero.
        let spanned = vec![span(0, 5_000, 10)];
        assert_eq!(read_amplification(&spanned, 1000, 0).windows, 5);
    }
}
