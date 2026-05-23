//! Tile size budget enforcement and intelligent feature dropping
//!
//! This module provides strategies for keeping tiles within size limits
//! by intelligently dropping less important features.

use crate::tile::Feature;
use crate::types::GeometryType;

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

impl ImportanceScorer {
    /// Calculate importance score for a feature (higher = more important)
    pub fn score(&self, feature: &Feature) -> f64 {
        match self {
            Self::GeometrySize => feature.positions.len() as f64,
            Self::PropertyCount => {
                // Features with more properties might be more important
                feature.properties.len() as f64
            }
            Self::FeatureId => {
                // Lower IDs = higher importance (arbitrary but deterministic)
                1.0 / (feature.id as f64 + 1.0)
            }
            Self::Random => {
                // Random for testing
                use std::collections::hash_map::RandomState;
                use std::hash::{BuildHasher, Hash, Hasher};
                let mut hasher = RandomState::new().build_hasher();
                feature.id.hash(&mut hasher);
                (hasher.finish() as f64) / (u64::MAX as f64)
            }
            Self::Combined => {
                let geom_score = feature.positions.len() as f64;
                let prop_score = feature.properties.len() as f64 * 10.0;
                geom_score + prop_score
            }
        }
    }
}

/// Tile size budget configuration
#[derive(Debug, Clone)]
pub struct TileBudget {
    /// Maximum uncompressed size in bytes
    pub max_uncompressed_size: usize,
    /// Maximum compressed size in bytes (after compression)
    pub max_compressed_size: usize,
    /// Maximum number of features
    pub max_feature_count: usize,
    /// Importance scorer to use for feature dropping
    pub scorer: ImportanceScorer,
}

impl Default for TileBudget {
    fn default() -> Self {
        Self {
            max_uncompressed_size: 500 * 1024, // 500KB uncompressed
            max_compressed_size: 128 * 1024,   // 128KB compressed
            max_feature_count: 10_000,
            scorer: ImportanceScorer::Combined,
        }
    }
}

impl TileBudget {
    /// Create a new tile budget with custom limits
    pub fn new(
        max_uncompressed_size: usize,
        max_compressed_size: usize,
        max_feature_count: usize,
    ) -> Self {
        Self {
            max_uncompressed_size,
            max_compressed_size,
            max_feature_count,
            scorer: ImportanceScorer::Combined,
        }
    }

    /// Set the importance scorer
    pub fn with_scorer(mut self, scorer: ImportanceScorer) -> Self {
        self.scorer = scorer;
        self
    }

    /// Estimate the uncompressed size of features
    pub fn estimate_size(features: &[Feature]) -> usize {
        let mut size = 0;

        for feature in features {
            // Rough geometry size (16 bytes per lon/lat pair)
            size += feature.positions.len() * 16;

            // Properties size (rough estimate)
            for (key, value) in &feature.properties {
                size += key.len();
                size += Self::estimate_value_size(value);
            }

            // Feature metadata overhead
            size += 32; // ID, type, time range, etc.
        }

        size
    }

    /// Estimate the size of a property value
    fn estimate_value_size(value: &crate::tile::Value) -> usize {
        match value {
            crate::tile::Value::String(s) => s.len(),
            crate::tile::Value::Double(_) => 8,
            crate::tile::Value::Float(_) => 4,
            crate::tile::Value::Int(_) => 8,
            crate::tile::Value::UInt(_) => 8,
            crate::tile::Value::Bool(_) => 1,
        }
    }

    /// Drop features to fit within budget constraints
    /// Returns (kept_features, dropped_count)
    pub fn enforce(&self, mut features: Vec<Feature>) -> (Vec<Feature>, usize) {
        let original_count = features.len();

        // Check feature count limit
        if features.len() > self.max_feature_count {
            features = self.drop_by_count(features, self.max_feature_count);
        }

        // Check size limit
        let size = Self::estimate_size(&features);
        if size > self.max_uncompressed_size {
            let target_size = (self.max_uncompressed_size as f64 * 0.9) as usize; // 90% of budget
            features = self.drop_by_size(features, target_size);
        }

        let dropped_count = original_count - features.len();
        (features, dropped_count)
    }

    /// Drop features to meet a count limit
    fn drop_by_count(&self, mut features: Vec<Feature>, max_count: usize) -> Vec<Feature> {
        if features.len() <= max_count {
            return features;
        }

        // Score all features
        let mut scored: Vec<(Feature, f64)> = features
            .into_iter()
            .map(|f| {
                let score = self.scorer.score(&f);
                (f, score)
            })
            .collect();

        // Sort by score (descending - highest score first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only the most important features
        scored.truncate(max_count);
        scored.into_iter().map(|(f, _)| f).collect()
    }

    /// Drop features to meet a size limit
    fn drop_by_size(&self, mut features: Vec<Feature>, target_size: usize) -> Vec<Feature> {
        let current_size = Self::estimate_size(&features);
        if current_size <= target_size {
            return features;
        }

        // Score all features
        let mut scored: Vec<(Feature, f64, usize)> = features
            .into_iter()
            .map(|f| {
                let score = self.scorer.score(&f);
                let size = self.estimate_feature_size(&f);
                (f, score, size)
            })
            .collect();

        // Sort by score/size ratio (importance per byte - higher is better)
        scored.sort_by(|a, b| {
            let ratio_a = a.1 / (a.2 as f64);
            let ratio_b = b.1 / (b.2 as f64);
            ratio_b
                .partial_cmp(&ratio_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedily keep features until we hit the budget
        let mut kept = Vec::new();
        let mut total_size = 0;

        for (feature, _, size) in scored {
            if total_size + size <= target_size {
                total_size += size;
                kept.push(feature);
            } else if total_size < target_size * 95 / 100 {
                // If we're more than 5% under budget, try to fit this feature
                // even if it goes slightly over
                if total_size + size <= target_size * 105 / 100 {
                    total_size += size;
                    kept.push(feature);
                }
            }
        }

        kept
    }

    /// Estimate the size of a single feature
    fn estimate_feature_size(&self, feature: &Feature) -> usize {
        let mut size = feature.positions.len() * 16;

        for (key, value) in &feature.properties {
            size += key.len();
            size += Self::estimate_value_size(value);
        }

        size + 32 // metadata overhead
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
    use std::collections::HashMap;

    fn create_test_feature(id: u64, geometry_size: usize, property_count: usize) -> Feature {
        let positions = vec![crate::tile::Position { lon: 0.0, lat: 0.0 }; geometry_size];
        let mut properties = HashMap::new();

        for i in 0..property_count {
            properties.insert(format!("key_{}", i), crate::tile::Value::Int(i as i64));
        }

        Feature {
            id,
            geometry_type: GeometryType::Point,
            positions,
            properties,
            time_range: None,
        }
    }

    #[test]
    fn test_importance_scoring() {
        let scorer = ImportanceScorer::GeometrySize;

        let feature1 = create_test_feature(1, 100, 0);
        let feature2 = create_test_feature(2, 50, 0);

        assert!(scorer.score(&feature1) > scorer.score(&feature2));
    }

    #[test]
    fn test_budget_enforcement_by_count() {
        let budget = TileBudget::default().with_scorer(ImportanceScorer::GeometrySize);

        let features = vec![
            create_test_feature(1, 100, 0),
            create_test_feature(2, 50, 0),
            create_test_feature(3, 75, 0),
            create_test_feature(4, 200, 0), // Largest, should be kept
        ];

        let original_count = features.len();
        let kept = budget.drop_by_count(features, 2);
        let dropped = original_count - kept.len();

        assert_eq!(kept.len(), 2);
        assert_eq!(dropped, 2);

        // Should keep the largest geometries
        assert_eq!(kept[0].id, 4); // 200
        assert_eq!(kept[1].id, 1); // 100
    }

    #[test]
    fn test_size_estimation() {
        let feature = create_test_feature(1, 100, 5);
        let size = TileBudget::estimate_size(&[feature]);

        assert!(size > 1000);
        assert!(size < 3000);
    }

    #[test]
    fn test_budget_enforcement_no_drop() {
        let budget = TileBudget::default();

        let features = vec![create_test_feature(1, 10, 1), create_test_feature(2, 10, 1)];

        let original_count = features.len();
        let (kept, dropped) = budget.enforce(features);

        assert_eq!(kept.len(), original_count);
        assert_eq!(dropped, 0);
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
