//! Tile size budget enforcement and intelligent feature dropping
//!
//! This module provides strategies for keeping tiles within size limits
//! by intelligently dropping less important features.
//!
//! Everything here is INDEX-based ([`TileBudget::enforce_indexed`] +
//! [`TileBudget::score_signals`]): the caller keeps its own feature
//! representation and the budget only ever sees `(vertex_count,
//! property_count, bytes)`. An earlier owned-`Feature` API existed alongside
//! it, but no build path ever built a `Feature` to hand it (see
//! `stt_build::tiler::tile_feature_signals`), so it was removed.

/// Feature importance scorer
#[derive(Debug, Clone, Copy)]
pub enum ImportanceScorer {
    /// Score by geometry size (larger = more important)
    GeometrySize,
    /// Score by property count (more properties = more important)
    PropertyCount,
    /// Score by feature ID (useful for debugging)
    FeatureId,
    /// Random scoring (for testing)
    Random,
    /// Combined scoring (geometry + properties)
    Combined,
}

/// Tile size budget configuration
///
/// ## Why there is no compressed-byte cap
///
/// A `max_compressed_size` field lived here until 2026-08. It was stored, and
/// `build_tile_budget` passed the SAME `--maximum-tile-bytes` value twice to
/// fill it — but nothing ever read it: [`Self::enforce_indexed`] binds only on
/// `max_uncompressed_size` and `max_feature_count`. It was deleted rather than
/// enforced, deliberately:
///
/// * Enforcing a compressed cap needs a drop → encode → zstd → check fixpoint
///   per tile, which this module's index-based, never-materialize design exists
///   to avoid (the caller keeps its own feature representation; the budget only
///   ever sees `(vertex_count, property_count, bytes)`).
/// * The correct form of a compressed cap is a **measured** post-zstd byte
///   objective from the sample encoder, which is the measurement-oracle work
///   feeding `--target-size` — not a stored constant.
///
/// So the CLI surface, the docs and the behavior now agree:
/// `--maximum-tile-bytes` is an UNCOMPRESSED cap and says so. Re-proposing a
/// compressed cap here means implementing the measured version; do not
/// resurrect the dead field.
#[derive(Debug, Clone)]
pub struct TileBudget {
    /// Maximum uncompressed size in bytes
    pub max_uncompressed_size: usize,
    /// Maximum number of features
    pub max_feature_count: usize,
    /// Importance scorer to use for feature dropping
    pub scorer: ImportanceScorer,
}

impl Default for TileBudget {
    fn default() -> Self {
        Self {
            max_uncompressed_size: 500 * 1024, // 500KB uncompressed
            max_feature_count: 10_000,
            scorer: ImportanceScorer::Combined,
        }
    }
}

impl TileBudget {
    /// Create a new tile budget with custom limits.
    ///
    /// Both caps are what [`Self::enforce_indexed`] actually binds on — there is
    /// no third, unenforced cap (see the type docs).
    pub fn new(max_uncompressed_size: usize, max_feature_count: usize) -> Self {
        Self {
            max_uncompressed_size,
            max_feature_count,
            scorer: ImportanceScorer::Combined,
        }
    }

    /// Set the importance scorer
    pub fn with_scorer(mut self, scorer: ImportanceScorer) -> Self {
        self.scorer = scorer;
        self
    }

    /// Enforce the budget over an *opaque, externally-owned* collection of
    /// `count` items, identified only by index.
    ///
    /// This is the type-impedance bridge used by stt-build: the tiler's
    /// per-tile feature representation (`TileFeature` — an enum over a borrowed
    /// `ParsedFeature` or an owned clipped segment) is columnar-backed, and
    /// materialising an owned feature struct (and going back) would be both
    /// lossy and wasteful. Instead the caller supplies two closures over its
    /// own collection:
    ///
    /// * `score(i)` — importance of item `i` (higher = keep). Use
    ///   [`Self::score_signals`] to mirror the [`ImportanceScorer`] semantics
    ///   for the chosen strategy.
    /// * `size(i)`  — estimated uncompressed bytes of item `i`.
    ///
    /// The drop policy is two-stage:
    /// 1. **Count cap** (`max_feature_count`): keep the highest-scored items.
    /// 2. **Size cap** (`max_uncompressed_size`): greedily keep items by
    ///    descending importance-per-byte until the target (90% of the cap) is
    ///    reached.
    ///
    /// Returns the indices to KEEP, **sorted ascending** so the caller can
    /// preserve its original feature order. Nothing is dropped (every index is
    /// returned) when the collection already fits — guaranteeing the
    /// default-off / under-budget path is a no-op.
    pub fn enforce_indexed<S, Z>(&self, count: usize, score: S, size: Z) -> Vec<usize>
    where
        S: Fn(usize) -> f64,
        Z: Fn(usize) -> usize,
    {
        // Fast path: a collection within BOTH caps is returned untouched. This
        // is what makes a tile under the budget byte-for-byte identical to a
        // build with no budget at all.
        let total_size: usize = (0..count).map(&size).sum();
        if count <= self.max_feature_count && total_size <= self.max_uncompressed_size {
            return (0..count).collect();
        }

        // Pre-score every item once.
        let mut scored: Vec<(usize, f64, usize)> =
            (0..count).map(|i| (i, score(i), size(i))).collect();

        // Step 1: count cap — keep the highest-scored items.
        if scored.len() > self.max_feature_count {
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(self.max_feature_count);
        }

        // Step 2: size cap — greedy keep by importance-per-byte, target = 90%
        // of the cap with a slight-overshoot tolerance so a nearly-full budget
        // is not left with an unusable remainder.
        let kept_size: usize = scored.iter().map(|(_, _, s)| *s).sum();
        let mut keep: Vec<usize> = if kept_size > self.max_uncompressed_size {
            let target = (self.max_uncompressed_size as f64 * 0.9) as usize;
            scored.sort_by(|a, b| {
                let ratio_a = a.1 / (a.2.max(1) as f64);
                let ratio_b = b.1 / (b.2.max(1) as f64);
                ratio_b
                    .partial_cmp(&ratio_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut kept = Vec::new();
            let mut total = 0usize;
            for (i, _, s) in &scored {
                if total + s <= target {
                    total += s;
                    kept.push(*i);
                } else if total < target * 95 / 100 && total + s <= target * 105 / 100 {
                    // Within 5% of budget — allow a slight overshoot to fill it.
                    total += s;
                    kept.push(*i);
                }
            }
            kept
        } else {
            scored.into_iter().map(|(i, _, _)| i).collect()
        };

        // Restore the caller's original order so feature emission is stable.
        keep.sort_unstable();
        keep
    }

    /// Score one *opaque* item for the budget's configured scorer, given the
    /// raw signals the tiler can cheaply produce: geometry vertex count and
    /// property count. `Random`/`FeatureId` are not reachable via this path
    /// (the budget always uses a deterministic geometry/combined strategy from
    /// stt-build), so they fall back to the combined formula.
    pub fn score_signals(&self, vertex_count: usize, property_count: usize) -> f64 {
        match self.scorer {
            ImportanceScorer::GeometrySize => vertex_count as f64,
            ImportanceScorer::PropertyCount => property_count as f64,
            ImportanceScorer::Combined | ImportanceScorer::FeatureId | ImportanceScorer::Random => {
                vertex_count as f64 + property_count as f64 * 10.0
            }
        }
    }
}

/// Statistics about budget enforcement
#[derive(Debug, Clone, Default)]
pub struct BudgetStats {
    pub tiles_processed: usize,
    pub tiles_reduced: usize,
    pub features_dropped: usize,
    pub original_size: usize,
    pub reduced_size: usize,
}

impl BudgetStats {
    /// Add stats from processing a tile
    pub fn add_tile(
        &mut self,
        original_count: usize,
        final_count: usize,
        original_size: usize,
        final_size: usize,
    ) {
        self.tiles_processed += 1;
        if final_count < original_count {
            self.tiles_reduced += 1;
        }
        self.features_dropped += original_count - final_count;
        self.original_size += original_size;
        self.reduced_size += final_size;
    }

    /// Get the average reduction ratio
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.reduced_size as f64 / self.original_size as f64
    }

    /// Get the percentage of features dropped
    pub fn features_dropped_pct(&self) -> f64 {
        if self.tiles_processed == 0 {
            return 0.0;
        }
        (self.features_dropped as f64 / self.tiles_processed as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enforce_indexed_under_budget_is_noop() {
        // Well within both caps -> every index returned, in order.
        let budget = TileBudget::new(1_000_000, 1000);
        let sizes = [100usize, 50, 200, 75];
        let keep = budget.enforce_indexed(sizes.len(), |i| sizes[i] as f64, |i| sizes[i]);
        assert_eq!(keep, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_enforce_indexed_count_cap_keeps_highest_scored() {
        // Cap at 2 features; scorer = geometry size (here, size doubles as both
        // score and bytes). The two largest (idx 2 and 3) survive, returned in
        // ascending index order.
        let budget = TileBudget::new(1_000_000, 2);
        let sizes = [10usize, 50, 200, 100];
        let mut keep = budget.enforce_indexed(sizes.len(), |i| sizes[i] as f64, |i| sizes[i]);
        keep.sort_unstable();
        assert_eq!(keep, vec![2, 3]); // 200 and 100 are largest
    }

    #[test]
    fn test_enforce_indexed_size_cap_drops_to_fit() {
        // Tiny byte cap forces a size-based drop; result must fit under cap.
        let budget = TileBudget::new(150, 10_000);
        let sizes = [100usize, 100, 100, 100];
        let keep = budget.enforce_indexed(sizes.len(), |i| sizes[i] as f64, |i| sizes[i]);
        let kept_bytes: usize = keep.iter().map(|&i| sizes[i]).sum();
        assert!(keep.len() < sizes.len(), "expected some features dropped");
        // 90%-of-cap target with slight-overshoot tolerance (<=105%).
        assert!(kept_bytes <= 150 * 105 / 100);
    }

    /// SH-5 neutrality snapshot. Deleting `max_compressed_size` must not move a
    /// single kept index: it was never read, so an over-budget fixture's keep-set
    /// is pinned here as a literal. If a future change to the greedy moves this,
    /// that is a real behavior change and must be argued for on its own terms —
    /// it is NOT a side effect of the field deletion.
    #[test]
    fn enforce_indexed_keep_set_is_pinned_across_the_compressed_cap_deletion() {
        // 12 items, byte cap 1000 (target 900), count cap 8. Scores are chosen
        // so the count stage and the density stage both bind.
        let sizes: [usize; 12] = [300, 120, 90, 400, 60, 250, 30, 180, 75, 500, 45, 210];
        let scores: [f64; 12] = [
            9.0, 7.0, 8.0, 12.0, 3.0, 6.0, 1.0, 11.0, 4.0, 15.0, 2.0, 10.0,
        ];
        let budget = TileBudget::new(1000, 8);
        let keep = budget.enforce_indexed(sizes.len(), |i| scores[i], |i| sizes[i]);
        assert_eq!(keep, vec![0, 1, 2, 7, 11]);
        // Ascending order is the caller's emission-order guarantee.
        assert!(keep.windows(2).all(|w| w[0] < w[1]));
    }

    /// Determinism: the same input re-run yields the identical keep-set. No
    /// RNG, no hash-map iteration, no wall clock in the selection path.
    #[test]
    fn enforce_indexed_is_deterministic_across_reruns() {
        let sizes: [usize; 12] = [300, 120, 90, 400, 60, 250, 30, 180, 75, 500, 45, 210];
        let scores: [f64; 12] = [
            9.0, 7.0, 8.0, 12.0, 3.0, 6.0, 1.0, 11.0, 4.0, 15.0, 2.0, 10.0,
        ];
        let budget = TileBudget::new(1000, 8);
        let a = budget.enforce_indexed(sizes.len(), |i| scores[i], |i| sizes[i]);
        for _ in 0..8 {
            let b = budget.enforce_indexed(sizes.len(), |i| scores[i], |i| sizes[i]);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_score_signals_matches_scorer() {
        let geo = TileBudget::default().with_scorer(ImportanceScorer::GeometrySize);
        assert_eq!(geo.score_signals(10, 5), 10.0);
        let combined = TileBudget::default().with_scorer(ImportanceScorer::Combined);
        assert_eq!(combined.score_signals(10, 5), 10.0 + 50.0);
    }

    #[test]
    fn test_budget_stats() {
        let mut stats = BudgetStats::default();

        stats.add_tile(1000, 800, 100_000, 80_000);
        stats.add_tile(500, 500, 50_000, 50_000);

        assert_eq!(stats.tiles_processed, 2);
        assert_eq!(stats.tiles_reduced, 1);
        assert_eq!(stats.features_dropped, 200);
        assert!((stats.reduction_ratio() - 0.8666).abs() < 0.01);
    }
}
