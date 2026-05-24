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
    Count,
    Sum,
    Mean,
    Min,
    Max,
}

/// Description of a single column emitted by the summary tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryColumn {
    pub name: String,
    pub agg: SummaryAggregation,
}

/// Description of the optional pre-aggregated summary tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryTier {
    pub scheme: SummaryScheme,
    pub min_zoom: u8,
    pub max_zoom: u8,
    pub cell_resolution_per_zoom: Vec<u8>,
    pub columns: Vec<SummaryColumn>,
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

/// One level of a temporal LOD pyramid (orthogonal to the summary tier above).
///
/// At any tile-zoom-level `z` such that `z <= max_zoom_level`, a client that
/// is currently displaying a time range too wide to render the base
/// `temporal_bucket_ms` tiles efficiently can fetch coarser tiles from this
/// level instead. Each level uses `bucket_ms` as its temporal bucket size
/// (which must be a multiple of the archive's base `temporal_bucket_ms`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalLodLevel {
    /// Temporal bucket size in milliseconds for tiles at this level.
    pub bucket_ms: u64,
    /// Inclusive upper bound on the spatial zoom level where this LOD applies.
    pub max_zoom_level: u8,
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
    /// Optional server-side aggregated summary tier. v2/v3 archives without
    /// a summary tier round-trip cleanly via the field default.
    #[serde(default)]
    pub summary_tier: Option<SummaryTier>,

    /// Optional temporal LOD pyramid (orthogonal to summary tier).
    /// When present, the archive carries aggregate tiles at coarser temporal
    /// granularities so a reader animating decades of data at "year scale"
    /// can fetch coarser tiles instead of streaming per-hour base tiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_lod: Option<Vec<TemporalLodLevel>>,
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
            temporal_lod: None,
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

    /// Attach a temporal LOD pyramid.
    ///
    /// Each level's `bucket_ms` MUST be a strict multiple of the archive's
    /// base `temporal_bucket_ms` and MUST be strictly greater than it; levels
    /// MUST be sorted by ascending `bucket_ms`. Returns `Err` if the input
    /// breaks any of those invariants — the build pipeline relies on them
    /// when re-bucketing features into LOD aggregates.
    pub fn with_temporal_lod(mut self, levels: Vec<TemporalLodLevel>) -> Result<Self> {
        validate_temporal_lod(self.temporal_bucket_ms, &levels)?;
        self.temporal_lod = if levels.is_empty() { None } else { Some(levels) };
        Ok(self)
    }

    /// Return the LOD level that applies at `zoom`, if any. The largest
    /// matching `bucket_ms` (coarsest level) wins — at a global zoom, you
    /// want the coarsest available aggregate, not the finest.
    pub fn temporal_lod_for_zoom(&self, zoom: u8) -> Option<&TemporalLodLevel> {
        let levels = self.temporal_lod.as_ref()?;
        levels
            .iter()
            .filter(|l| zoom <= l.max_zoom_level)
            .max_by_key(|l| l.bucket_ms)
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

/// Verify the LOD invariants: ascending bucket order, multiples of the base
/// bucket, strictly coarser than base, distinct bucket sizes.
fn validate_temporal_lod(base_bucket_ms: u64, levels: &[TemporalLodLevel]) -> Result<()> {
    if base_bucket_ms == 0 {
        return Err(Error::Other(
            "temporal_bucket_ms must be non-zero when declaring a LOD pyramid".into(),
        ));
    }
    let mut prev: Option<u64> = None;
    for (i, level) in levels.iter().enumerate() {
        if level.bucket_ms == 0 {
            return Err(Error::Other(format!(
                "temporal_lod[{i}].bucket_ms must be non-zero"
            )));
        }
        if level.bucket_ms <= base_bucket_ms {
            return Err(Error::Other(format!(
                "temporal_lod[{i}].bucket_ms ({}) must be > base bucket ({})",
                level.bucket_ms, base_bucket_ms
            )));
        }
        if level.bucket_ms % base_bucket_ms != 0 {
            return Err(Error::Other(format!(
                "temporal_lod[{i}].bucket_ms ({}) must be a multiple of base bucket ({})",
                level.bucket_ms, base_bucket_ms
            )));
        }
        if let Some(p) = prev {
            if level.bucket_ms <= p {
                return Err(Error::Other(format!(
                    "temporal_lod must be sorted by ascending bucket_ms; got {} after {}",
                    level.bucket_ms, p
                )));
            }
        }
        prev = Some(level.bucket_ms);
    }
    Ok(())
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

    // ------------------------------------------------------------------
    // temporal_lod
    // ------------------------------------------------------------------

    fn hour() -> u64 {
        3_600_000
    }
    fn day() -> u64 {
        24 * hour()
    }
    fn thirty_days() -> u64 {
        30 * day()
    }

    #[test]
    fn temporal_lod_roundtrips_through_json() {
        let levels = vec![
            TemporalLodLevel {
                bucket_ms: day(),
                max_zoom_level: 8,
            },
            TemporalLodLevel {
                bucket_ms: thirty_days(),
                max_zoom_level: 4,
            },
        ];
        let metadata = Metadata::new("lod")
            .with_temporal_bucket_ms(hour())
            .with_temporal_lod(levels.clone())
            .unwrap();
        let bytes = metadata.to_json_bytes().unwrap();
        let decoded = Metadata::from_json_bytes(&bytes).unwrap();
        assert_eq!(decoded.temporal_lod.as_deref(), Some(levels.as_slice()));
    }

    #[test]
    fn temporal_lod_field_omitted_when_unset() {
        // Older readers that don't know about temporal_lod must still parse
        // a freshly-written archive; the field is skipped when None.
        let metadata = Metadata::new("no-lod").with_temporal_bucket_ms(hour());
        let s = String::from_utf8(metadata.to_json_bytes().unwrap()).unwrap();
        assert!(!s.contains("temporal_lod"), "got: {s}");
    }

    #[test]
    fn temporal_lod_missing_field_decodes_back_compat() {
        // A v3 archive built before this feature has no `temporal_lod` key
        // in its metadata JSON; the new field must default to None.
        let legacy = r#"{
            "name": "legacy",
            "description": "",
            "attribution": "",
            "bounds": {"min_lon": -180, "min_lat": -85, "max_lon": 180, "max_lat": 85},
            "time_range": {"start": 0, "end": 1},
            "min_zoom": 0,
            "max_zoom": 14,
            "tile_count": 0,
            "feature_count": 0,
            "layers": ["default"],
            "properties": {},
            "temporal_bucket_ms": 3600000
        }"#;
        let m = Metadata::from_json_bytes(legacy.as_bytes()).unwrap();
        assert!(m.temporal_lod.is_none());
    }

    #[test]
    fn temporal_lod_rejects_non_multiple_bucket() {
        let res = Metadata::new("bad").with_temporal_bucket_ms(hour()).with_temporal_lod(vec![
            TemporalLodLevel { bucket_ms: hour() + 7, max_zoom_level: 5 },
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn temporal_lod_rejects_bucket_smaller_than_or_equal_to_base() {
        let res = Metadata::new("bad")
            .with_temporal_bucket_ms(day())
            .with_temporal_lod(vec![TemporalLodLevel { bucket_ms: hour(), max_zoom_level: 5 }]);
        assert!(res.is_err());

        let res = Metadata::new("bad")
            .with_temporal_bucket_ms(hour())
            .with_temporal_lod(vec![TemporalLodLevel { bucket_ms: hour(), max_zoom_level: 5 }]);
        assert!(res.is_err());
    }

    #[test]
    fn temporal_lod_rejects_unsorted_levels() {
        let res = Metadata::new("bad").with_temporal_bucket_ms(hour()).with_temporal_lod(vec![
            TemporalLodLevel { bucket_ms: thirty_days(), max_zoom_level: 4 },
            TemporalLodLevel { bucket_ms: day(), max_zoom_level: 8 },
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn temporal_lod_for_zoom_picks_coarsest_applicable() {
        let m = Metadata::new("lod")
            .with_temporal_bucket_ms(hour())
            .with_temporal_lod(vec![
                TemporalLodLevel { bucket_ms: day(), max_zoom_level: 8 },
                TemporalLodLevel { bucket_ms: thirty_days(), max_zoom_level: 4 },
            ])
            .unwrap();
        // Very-zoomed-out: both levels apply, pick the coarser (30d).
        assert_eq!(
            m.temporal_lod_for_zoom(0).map(|l| l.bucket_ms),
            Some(thirty_days())
        );
        // Mid zoom: only the day level applies.
        assert_eq!(m.temporal_lod_for_zoom(6).map(|l| l.bucket_ms), Some(day()));
        // High zoom: no LOD — fall back to base bucket.
        assert!(m.temporal_lod_for_zoom(12).is_none());
    }

    #[test]
    fn temporal_lod_for_zoom_is_none_when_unset() {
        let m = Metadata::new("plain").with_temporal_bucket_ms(hour());
        assert!(m.temporal_lod_for_zoom(0).is_none());
    }

    #[test]
    fn temporal_lod_empty_vec_clears_to_none() {
        // Passing an empty list is treated as "no LOD" rather than an error,
        // so callers can compute the level set unconditionally.
        let m = Metadata::new("empty")
            .with_temporal_bucket_ms(hour())
            .with_temporal_lod(vec![])
            .unwrap();
        assert!(m.temporal_lod.is_none());
    }
}
