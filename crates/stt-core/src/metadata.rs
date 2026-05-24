//! Archive metadata structures
//!
//! This module provides types for storing and managing archive metadata.

use crate::error::{Error, Result};
use crate::types::{BoundingBox, TimeRange};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Aggregation scheme for the optional pre-aggregated summary tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SummaryScheme {
    /// Uber H3 hexagonal cells.
    H3,
    /// CARTO Quadbin (Z/X/Y quad-key encoded as u64).
    Quadbin,
}

/// One aggregated column in a summary tier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryAggregation {
    /// Total count per cell (always present, computed implicitly).
    Count,
    /// Sum of the source column per cell.
    Sum,
    /// Mean (sum / count) of the source column per cell.
    Mean,
    /// Min of the source column per cell.
    Min,
    /// Max of the source column per cell.
    Max,
}

/// Description of a single column emitted by the summary tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryColumn {
    /// Source property name in the raw features (ignored for `Count`).
    pub name: String,
    /// Aggregation function applied across the cell.
    pub agg: SummaryAggregation,
}

/// Description of the optional pre-aggregated summary tier.
///
/// When present, the archive carries a parallel set of tiles whose features
/// are cell aggregates (one row per cell, with `count` and the configured
/// `columns`) rather than raw points. Readers dispatch to the summary tier
/// at low zooms where raw rendering would saturate the GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryTier {
    /// Aggregation cell scheme (h3, quadbin, ...).
    pub scheme: SummaryScheme,
    /// Lowest zoom level at which summary tiles exist.
    pub min_zoom: u8,
    /// Highest zoom level at which summary tiles exist.
    pub max_zoom: u8,
    /// One cell-resolution per zoom in `[min_zoom..=max_zoom]`. For H3 this
    /// is the H3 resolution (0..=15); for Quadbin it is the quadbin zoom.
    pub cell_resolution_per_zoom: Vec<u8>,
    /// Columns aggregated into every summary tile.
    pub columns: Vec<SummaryColumn>,
    /// Layer name used for summary tile layers (matches the
    /// `arrow_tile` layer name a reader looks for to identify summary
    /// payloads). Defaults to `"summary"`.
    #[serde(default = "default_summary_layer_name")]
    pub layer_name: String,
}

fn default_summary_layer_name() -> String {
    "summary".to_string()
}

impl SummaryTier {
    /// Resolution to use at a given zoom. Falls back to the closest mapped
    /// resolution if the zoom is outside `[min_zoom, max_zoom]`.
    pub fn resolution_for_zoom(&self, zoom: u8) -> u8 {
        if self.cell_resolution_per_zoom.is_empty() {
            return zoom;
        }
        if zoom <= self.min_zoom {
            return self.cell_resolution_per_zoom[0];
        }
        if zoom >= self.max_zoom {
            return *self
                .cell_resolution_per_zoom
                .last()
                .expect("non-empty per check");
        }
        let idx = (zoom - self.min_zoom) as usize;
        self.cell_resolution_per_zoom
            .get(idx)
            .copied()
            .unwrap_or_else(|| {
                *self.cell_resolution_per_zoom.last().unwrap()
            })
    }
}

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
    /// Optional server-side aggregated summary tier. When present the archive
    /// contains BOTH raw tiles (covering `min_zoom..=max_zoom`) and summary
    /// tiles (covering `summary_tier.min_zoom..=summary_tier.max_zoom`),
    /// addressable through the same `(zoom, x, y, time)` directory but with
    /// a distinct layer name (`summary_tier.layer_name`).
    ///
    /// `#[serde(default)]` so v2/v3 archives without a summary tier round-
    /// trip cleanly through the metadata decoder.
    #[serde(default)]
    pub summary_tier: Option<SummaryTier>,
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
            summary_tier: None,
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

    /// Attach an aggregated summary-tier descriptor.
    pub fn with_summary_tier(mut self, tier: SummaryTier) -> Self {
        self.summary_tier = Some(tier);
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
    fn test_metadata_summary_tier_roundtrip() {
        let tier = SummaryTier {
            scheme: SummaryScheme::H3,
            min_zoom: 0,
            max_zoom: 4,
            cell_resolution_per_zoom: vec![0, 1, 2, 3, 4],
            columns: vec![
                SummaryColumn {
                    name: "magnitude".to_string(),
                    agg: SummaryAggregation::Mean,
                },
                SummaryColumn {
                    name: "magnitude".to_string(),
                    agg: SummaryAggregation::Max,
                },
            ],
            layer_name: "summary".to_string(),
        };
        let metadata = Metadata::new("summary-test").with_summary_tier(tier.clone());
        let bytes = metadata.to_json_bytes().unwrap();
        let decoded = Metadata::from_json_bytes(&bytes).unwrap();
        let dt = decoded.summary_tier.unwrap();
        assert_eq!(dt.scheme, SummaryScheme::H3);
        assert_eq!(dt.min_zoom, 0);
        assert_eq!(dt.max_zoom, 4);
        assert_eq!(dt.cell_resolution_per_zoom.len(), 5);
        assert_eq!(dt.columns.len(), 2);
        assert_eq!(dt.resolution_for_zoom(2), 2);
        // Out-of-range zooms clamp to the endpoints.
        assert_eq!(dt.resolution_for_zoom(10), 4);
    }

    #[test]
    fn test_metadata_without_summary_tier_decodes() {
        // A pre-summary-tier archive's metadata JSON has no `summary_tier`
        // field at all. serde's `#[default]` must accept it.
        let json = br#"{
            "name": "old",
            "description": "",
            "attribution": "",
            "bounds": {"min_lon":-180.0,"min_lat":-85.0,"max_lon":180.0,"max_lat":85.0},
            "time_range": {"start":0,"end":1000},
            "min_zoom": 0,
            "max_zoom": 8,
            "tile_count": 0,
            "feature_count": 0,
            "layers": ["default"],
            "properties": {},
            "temporal_bucket_ms": 3600000
        }"#;
        let m = Metadata::from_json_bytes(json).unwrap();
        assert!(m.summary_tier.is_none());
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
