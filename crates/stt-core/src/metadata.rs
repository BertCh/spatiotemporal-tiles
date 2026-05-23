//! Archive metadata structures
//!
//! This module provides types for storing and managing archive metadata.

use crate::error::{Error, Result};
use crate::types::{BoundingBox, TimeRange};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Archive metadata.
///
/// Stored in the archive as UTF-8 JSON — small, human-inspectable, and
/// versionless thanks to serde's field defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Archive name
    pub name: String,
    /// Description
    pub description: String,
    /// Attribution text
    pub attribution: String,
    /// Bounding box
    pub bounds: BoundingBox,
    /// Time range
    pub time_range: TimeRange,
    /// Minimum zoom level
    pub min_zoom: u8,
    /// Maximum zoom level
    pub max_zoom: u8,
    /// Total number of tiles
    pub tile_count: u64,
    /// Total number of features
    pub feature_count: u64,
    /// Layer names
    pub layers: Vec<String>,
    /// Custom properties
    pub properties: HashMap<String, String>,
    /// Temporal bucket size in milliseconds for tile chunking
    /// Tiles are organized into fixed temporal intervals (e.g., 3600000 = 1 hour)
    pub temporal_bucket_ms: u64,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            attribution: String::new(),
            bounds: BoundingBox {
                min_lon: -180.0,
                min_lat: -85.0511,
                max_lon: 180.0,
                max_lat: 85.0511,
            },
            time_range: TimeRange::new(0, 0),
            min_zoom: 0,
            max_zoom: 14,
            tile_count: 0,
            feature_count: 0,
            layers: vec!["default".to_string()],
            properties: HashMap::new(),
            temporal_bucket_ms: 3600 * 1000, // 1 hour default
        }
    }
}

impl Metadata {
    /// Create a new metadata object
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set attribution
    pub fn with_attribution(mut self, attribution: impl Into<String>) -> Self {
        self.attribution = attribution.into();
        self
    }

    /// Set bounds
    pub fn with_bounds(mut self, bounds: BoundingBox) -> Self {
        self.bounds = bounds;
        self
    }

    /// Set time range
    pub fn with_time_range(mut self, time_range: TimeRange) -> Self {
        self.time_range = time_range;
        self
    }

    /// Set zoom levels
    pub fn with_zoom_levels(mut self, min_zoom: u8, max_zoom: u8) -> Self {
        self.min_zoom = min_zoom;
        self.max_zoom = max_zoom;
        self
    }

    /// Set temporal bucket size in milliseconds
    pub fn with_temporal_bucket_ms(mut self, temporal_bucket_ms: u64) -> Self {
        self.temporal_bucket_ms = temporal_bucket_ms;
        self
    }

    /// Add a custom property
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Serialise to the JSON byte form stored in an archive.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| Error::Other(format!("metadata JSON encode failed: {e}")))
    }

    /// Parse from the JSON byte form stored in an archive.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| Error::InvalidArchive(format!("metadata JSON decode failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_json_roundtrip() {
        let metadata = Metadata::new("json-test")
            .with_description("desc")
            .with_zoom_levels(2, 12)
            .with_temporal_bucket_ms(3_600_000)
            .with_property("source", "unit-test");
        let bytes = metadata.to_json_bytes().unwrap();
        let decoded = Metadata::from_json_bytes(&bytes).unwrap();
        assert_eq!(decoded.name, "json-test");
        assert_eq!(decoded.min_zoom, 2);
        assert_eq!(decoded.max_zoom, 12);
        assert_eq!(decoded.temporal_bucket_ms, 3_600_000);
        assert_eq!(decoded.properties.get("source").map(String::as_str), Some("unit-test"));
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = Metadata::new("test")
            .with_description("Test archive")
            .with_attribution("Test data")
            .with_zoom_levels(0, 14)
            .with_property("key", "value");

        assert_eq!(metadata.name, "test");
        assert_eq!(metadata.description, "Test archive");
        assert_eq!(metadata.min_zoom, 0);
        assert_eq!(metadata.max_zoom, 14);
        assert_eq!(metadata.properties.get("key"), Some(&"value".to_string()));
    }
}
