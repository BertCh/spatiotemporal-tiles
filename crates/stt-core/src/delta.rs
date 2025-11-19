//! Delta encoding for temporal features
//!
//! This module provides efficient delta encoding to avoid storing
//! identical features across multiple temporal frames.

use crate::tile::Feature;
use crate::types::GeometryType;
use std::collections::HashMap;
use blake3;

/// Hash of a feature's geometry and properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureHash([u8; 32]);

impl FeatureHash {
    /// Create a new feature hash from raw bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the hash as a u64 for compact storage
    pub fn to_u64(&self) -> u64 {
        u64::from_le_bytes([
            self.0[0], self.0[1], self.0[2], self.0[3],
            self.0[4], self.0[5], self.0[6], self.0[7],
        ])
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Compute a hash of a feature's geometry and properties
pub fn hash_feature(feature: &Feature) -> FeatureHash {
    let mut hasher = blake3::Hasher::new();
    
    // Hash geometry type
    hasher.update(&[feature.geometry_type.to_proto() as u8]);
    
    // Hash geometry coordinates
    for coord in &feature.geometry {
        hasher.update(&coord.to_le_bytes());
    }
    
    // Hash properties (sorted by key for consistency)
    let mut sorted_props: Vec<_> = feature.properties.iter().collect();
    sorted_props.sort_by_key(|(k, _)| *k);
    
    for (key, value) in sorted_props {
        hasher.update(key.as_bytes());
        hasher.update(&value_to_bytes(value));
    }
    
    FeatureHash(hasher.finalize().into())
}

/// Convert a value to bytes for hashing
fn value_to_bytes(value: &crate::tile::Value) -> Vec<u8> {
    match value {
        crate::tile::Value::String(s) => s.as_bytes().to_vec(),
        crate::tile::Value::Double(d) => d.to_le_bytes().to_vec(),
        crate::tile::Value::Float(f) => f.to_le_bytes().to_vec(),
        crate::tile::Value::Int(i) => i.to_le_bytes().to_vec(),
        crate::tile::Value::UInt(u) => u.to_le_bytes().to_vec(),
        crate::tile::Value::Bool(b) => vec![*b as u8],
    }
}

/// Track feature changes across temporal frames
#[derive(Debug)]
pub struct TemporalDeltaTracker {
    /// Map from feature ID to its hash in the previous frame
    previous_hashes: HashMap<u64, FeatureHash>,
    /// Map from hash to reference feature (for deduplication)
    hash_to_feature: HashMap<FeatureHash, Feature>,
    /// Statistics
    pub stats: DeltaStats,
}

#[derive(Debug, Default, Clone)]
pub struct DeltaStats {
    pub total_features: usize,
    pub unchanged_features: usize,
    pub modified_features: usize,
    pub new_features: usize,
    pub deleted_features: usize,
    pub deduplicated_count: usize,
}

impl TemporalDeltaTracker {
    /// Create a new delta tracker
    pub fn new() -> Self {
        Self {
            previous_hashes: HashMap::new(),
            hash_to_feature: HashMap::new(),
            stats: DeltaStats::default(),
        }
    }

    /// Process features for a new temporal frame
    /// Returns (features_to_encode, change_type_per_feature)
    pub fn process_frame(
        &mut self,
        features: Vec<Feature>,
    ) -> Vec<(Feature, ChangeType)> {
        let mut result = Vec::new();
        let mut current_ids = std::collections::HashSet::new();

        for feature in features {
            current_ids.insert(feature.id);
            let hash = hash_feature(&feature);
            
            let change_type = if let Some(&previous_hash) = self.previous_hashes.get(&feature.id) {
                if previous_hash == hash {
                    // Feature unchanged - reference previous
                    self.stats.unchanged_features += 1;
                    ChangeType::Unchanged(previous_hash)
                } else {
                    // Feature modified
                    self.stats.modified_features += 1;
                    self.previous_hashes.insert(feature.id, hash);
                    self.hash_to_feature.insert(hash, feature.clone());
                    ChangeType::Modified
                }
            } else {
                // New feature
                self.stats.new_features += 1;
                self.previous_hashes.insert(feature.id, hash);
                self.hash_to_feature.insert(hash, feature.clone());
                ChangeType::Created
            };

            self.stats.total_features += 1;
            result.push((feature, change_type));
        }

        // Detect deleted features
        let deleted: Vec<_> = self.previous_hashes
            .keys()
            .filter(|id| !current_ids.contains(id))
            .copied()
            .collect();
        
        self.stats.deleted_features += deleted.len();
        
        for id in deleted {
            self.previous_hashes.remove(&id);
        }

        result
    }

    /// Get a feature by its hash (for reconstructing unchanged features)
    pub fn get_feature_by_hash(&self, hash: &FeatureHash) -> Option<&Feature> {
        self.hash_to_feature.get(hash)
    }

    /// Clear the tracker (start fresh)
    pub fn clear(&mut self) {
        self.previous_hashes.clear();
        self.hash_to_feature.clear();
        self.stats = DeltaStats::default();
    }
}

/// Change type for a feature
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// Feature is unchanged from previous frame (includes hash reference)
    Unchanged(FeatureHash),
    /// Feature was created in this frame
    Created,
    /// Feature was modified in this frame
    Modified,
    /// Feature was deleted in this frame
    Deleted,
}

impl ChangeType {
    /// Convert to proto enum value
    pub fn to_proto(&self) -> i32 {
        match self {
            Self::Unchanged(_) => 0, // UNCHANGED
            Self::Created => 1,      // CREATED
            Self::Modified => 2,     // MODIFIED
            Self::Deleted => 3,      // DELETED
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_feature(id: u64, x: u32, y: u32) -> Feature {
        Feature {
            id,
            geometry_type: GeometryType::Point,
            geometry: vec![x, y],
            properties: std::collections::HashMap::new(),
            time_range: None,
        }
    }

    #[test]
    fn test_feature_hashing() {
        let feature1 = create_test_feature(1, 100, 200);
        let feature2 = create_test_feature(2, 100, 200); // Different ID, same geometry
        
        let hash1 = hash_feature(&feature1);
        let hash2 = hash_feature(&feature2);
        
        // Same geometry should have same hash regardless of ID
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_delta_tracking() {
        let mut tracker = TemporalDeltaTracker::new();

        // Frame 1: Create features
        let frame1 = vec![
            create_test_feature(1, 100, 200),
            create_test_feature(2, 150, 250),
        ];
        
        let result = tracker.process_frame(frame1);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].1, ChangeType::Created));
        assert!(matches!(result[1].1, ChangeType::Created));

        // Frame 2: One unchanged, one modified
        let frame2 = vec![
            create_test_feature(1, 100, 200), // Unchanged
            create_test_feature(2, 160, 260), // Modified
        ];
        
        let result = tracker.process_frame(frame2);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].1, ChangeType::Unchanged(_)));
        assert!(matches!(result[1].1, ChangeType::Modified));

        assert_eq!(tracker.stats.unchanged_features, 1);
        assert_eq!(tracker.stats.modified_features, 1);
    }

    #[test]
    fn test_feature_deletion() {
        let mut tracker = TemporalDeltaTracker::new();

        // Frame 1: Create features
        let frame1 = vec![
            create_test_feature(1, 100, 200),
            create_test_feature(2, 150, 250),
        ];
        tracker.process_frame(frame1);

        // Frame 2: Delete feature 2
        let frame2 = vec![
            create_test_feature(1, 100, 200),
        ];
        tracker.process_frame(frame2);

        assert_eq!(tracker.stats.deleted_features, 1);
    }
}

