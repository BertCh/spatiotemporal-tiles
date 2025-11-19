//! Property deduplication for efficient value storage
//!
//! This module provides advanced property value deduplication
//! to minimize redundant data in tiles.

use std::collections::HashMap;
use crate::tile::Value;

/// Property value deduplicator
pub struct PropertyDeduplicator {
    /// Map from value hash to index
    value_to_index: HashMap<ValueHash, u32>,
    /// Deduplicated values in order
    values: Vec<Value>,
    /// Statistics
    pub stats: DeduplicationStats,
}

/// Hash of a property value for deduplication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ValueHash(u64);

impl ValueHash {
    fn from_value(value: &Value) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        match value {
            Value::String(s) => {
                0u8.hash(&mut hasher);
                s.hash(&mut hasher);
            }
            Value::Double(d) => {
                1u8.hash(&mut hasher);
                d.to_bits().hash(&mut hasher);
            }
            Value::Float(f) => {
                2u8.hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            }
            Value::Int(i) => {
                3u8.hash(&mut hasher);
                i.hash(&mut hasher);
            }
            Value::UInt(u) => {
                4u8.hash(&mut hasher);
                u.hash(&mut hasher);
            }
            Value::Bool(b) => {
                5u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
        }
        
        ValueHash(hasher.finish())
    }
}

#[derive(Debug, Default, Clone)]
pub struct DeduplicationStats {
    pub total_values: usize,
    pub unique_values: usize,
    pub duplicate_values: usize,
}

impl DeduplicationStats {
    pub fn deduplication_ratio(&self) -> f64 {
        if self.total_values == 0 {
            return 1.0;
        }
        self.unique_values as f64 / self.total_values as f64
    }
}

impl PropertyDeduplicator {
    /// Create a new property deduplicator
    pub fn new() -> Self {
        Self {
            value_to_index: HashMap::new(),
            values: Vec::new(),
            stats: DeduplicationStats::default(),
        }
    }

    /// Add a value and get its index
    /// Returns the index and whether it was newly inserted
    pub fn add_value(&mut self, value: Value) -> (u32, bool) {
        self.stats.total_values += 1;
        
        let hash = ValueHash::from_value(&value);
        
        if let Some(&index) = self.value_to_index.get(&hash) {
            self.stats.duplicate_values += 1;
            (index, false)
        } else {
            let index = self.values.len() as u32;
            self.values.push(value);
            self.value_to_index.insert(hash, index);
            self.stats.unique_values += 1;
            (index, true)
        }
    }

    /// Get all deduplicated values
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Get value by index
    pub fn get_value(&self, index: u32) -> Option<&Value> {
        self.values.get(index as usize)
    }

    /// Clear the deduplicator
    pub fn clear(&mut self) {
        self.value_to_index.clear();
        self.values.clear();
        self.stats = DeduplicationStats::default();
    }

    /// Get the number of unique values
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the deduplicator is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Key deduplicator (similar to property deduplicator but for keys)
pub struct KeyDeduplicator {
    /// Map from key to index
    key_to_index: HashMap<String, u32>,
    /// Deduplicated keys in order
    keys: Vec<String>,
}

impl KeyDeduplicator {
    /// Create a new key deduplicator
    pub fn new() -> Self {
        Self {
            key_to_index: HashMap::new(),
            keys: Vec::new(),
        }
    }

    /// Add a key and get its index
    /// Returns the index and whether it was newly inserted
    pub fn add_key(&mut self, key: String) -> (u32, bool) {
        if let Some(&index) = self.key_to_index.get(&key) {
            (index, false)
        } else {
            let index = self.keys.len() as u32;
            self.key_to_index.insert(key.clone(), index);
            self.keys.push(key);
            (index, true)
        }
    }

    /// Get all deduplicated keys
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Get key by index
    pub fn get_key(&self, index: u32) -> Option<&String> {
        self.keys.get(index as usize)
    }

    /// Clear the deduplicator
    pub fn clear(&mut self) {
        self.key_to_index.clear();
        self.keys.clear();
    }

    /// Get the number of unique keys
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if the deduplicator is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_deduplication() {
        let mut dedup = PropertyDeduplicator::new();

        // Add same string multiple times
        let (idx1, new1) = dedup.add_value(Value::String("test".to_string()));
        let (idx2, new2) = dedup.add_value(Value::String("test".to_string()));
        let (idx3, new3) = dedup.add_value(Value::String("test".to_string()));

        assert!(new1); // First one is new
        assert!(!new2); // Second is duplicate
        assert!(!new3); // Third is duplicate
        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx3);
        
        assert_eq!(dedup.len(), 1);
        assert_eq!(dedup.stats.total_values, 3);
        assert_eq!(dedup.stats.unique_values, 1);
        assert_eq!(dedup.stats.duplicate_values, 2);
    }

    #[test]
    fn test_different_values() {
        let mut dedup = PropertyDeduplicator::new();

        let (idx1, _) = dedup.add_value(Value::String("test1".to_string()));
        let (idx2, _) = dedup.add_value(Value::String("test2".to_string()));
        let (idx3, _) = dedup.add_value(Value::Int(42));

        assert_ne!(idx1, idx2);
        assert_ne!(idx2, idx3);
        assert_ne!(idx1, idx3);
        
        assert_eq!(dedup.len(), 3);
    }

    #[test]
    fn test_numeric_deduplication() {
        let mut dedup = PropertyDeduplicator::new();

        let (idx1, _) = dedup.add_value(Value::Int(42));
        let (idx2, _) = dedup.add_value(Value::Int(42));
        let (idx3, _) = dedup.add_value(Value::Int(43));

        assert_eq!(idx1, idx2);
        assert_ne!(idx1, idx3);
        
        assert_eq!(dedup.len(), 2);
    }

    #[test]
    fn test_key_deduplication() {
        let mut dedup = KeyDeduplicator::new();

        let (idx1, new1) = dedup.add_key("name".to_string());
        let (idx2, new2) = dedup.add_key("name".to_string());
        let (idx3, new3) = dedup.add_key("age".to_string());

        assert!(new1);
        assert!(!new2);
        assert!(new3);
        
        assert_eq!(idx1, idx2);
        assert_ne!(idx1, idx3);
        
        assert_eq!(dedup.len(), 2);
    }

    #[test]
    fn test_deduplication_ratio() {
        let mut dedup = PropertyDeduplicator::new();

        // Add 10 values, but only 2 unique
        for _ in 0..5 {
            dedup.add_value(Value::String("a".to_string()));
            dedup.add_value(Value::String("b".to_string()));
        }

        assert_eq!(dedup.stats.total_values, 10);
        assert_eq!(dedup.stats.unique_values, 2);
        assert_eq!(dedup.stats.duplicate_values, 8);
        
        let ratio = dedup.stats.deduplication_ratio();
        assert!((ratio - 0.2).abs() < 0.01); // 2/10 = 0.2
    }
}

