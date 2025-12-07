//! Archive metadata structures
//!
//! This module provides types for storing and managing archive metadata.

use crate::error::Result;
use crate::types::{BoundingBox, TimeRange};
use std::collections::HashMap;

/// Archive metadata
#[derive(Debug, Clone)]
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

    /// Add a custom property
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Convert to Protocol Buffer Metadata
    pub fn to_proto(&self) -> crate::proto::Metadata {
        crate::proto::Metadata {
            version: 1,
            name: self.name.clone(),
            description: self.description.clone(),
            attribution: self.attribution.clone(),
            bounds: Some(crate::proto::BoundingBox {
                min_lon: self.bounds.min_lon,
                min_lat: self.bounds.min_lat,
                max_lon: self.bounds.max_lon,
                max_lat: self.bounds.max_lat,
            }),
            time_range: Some(crate::proto::TimeRange {
                start: self.time_range.start,
                end: self.time_range.end,
                interval: 0, // No regular interval specified
            }),
            min_zoom: self.min_zoom as u32,
            max_zoom: self.max_zoom as u32,
            layers: self
                .layers
                .iter()
                .map(|name| crate::proto::LayerInfo {
                    name: name.clone(),
                    description: String::new(),
                    properties: vec![],
                    geometry_types: vec![],
                })
                .collect(),
            generation: None,
            stats: Some(crate::proto::Statistics {
                total_tiles: self.tile_count,
                total_features: self.feature_count,
                total_size: 0,
                uncompressed_size: 0,
                compression_ratio: 0.0,
                zoom_stats: vec![],
            }),
        }
    }

    /// Convert from Protocol Buffer Metadata
    pub fn from_proto(proto: &crate::proto::Metadata) -> Result<Self> {
        let bounds = proto
            .bounds
            .as_ref()
            .map(|b| BoundingBox {
                min_lon: b.min_lon,
                min_lat: b.min_lat,
                max_lon: b.max_lon,
                max_lat: b.max_lat,
            })
            .unwrap_or(BoundingBox {
                min_lon: -180.0,
                min_lat: -85.0511,
                max_lon: 180.0,
                max_lat: 85.0511,
            });

        let time_range = proto
            .time_range
            .as_ref()
            .map(|tr| TimeRange::new(tr.start, tr.end))
            .unwrap_or(TimeRange::new(0, 0));

        let stats = proto.stats.as_ref();
        let tile_count = stats.map(|s| s.total_tiles).unwrap_or(0);
        let feature_count = stats.map(|s| s.total_features).unwrap_or(0);

        Ok(Self {
            name: proto.name.clone(),
            description: proto.description.clone(),
            attribution: proto.attribution.clone(),
            bounds,
            time_range,
            min_zoom: proto.min_zoom as u8,
            max_zoom: proto.max_zoom as u8,
            tile_count,
            feature_count,
            layers: proto.layers.iter().map(|l| l.name.clone()).collect(),
            properties: HashMap::new(), // No custom properties in proto
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_metadata_proto_roundtrip() {
        let metadata = Metadata::new("test").with_description("Test archive");

        let proto = metadata.to_proto();
        let decoded = Metadata::from_proto(&proto).unwrap();

        assert_eq!(decoded.name, metadata.name);
        assert_eq!(decoded.description, metadata.description);
    }
}
