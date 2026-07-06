//! Archive metadata structures
//!
//! This module provides types for storing and managing archive metadata.

use crate::error::{Error, Result};
use crate::types::{BoundingBox, TimeRange};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    /// Number of fine-grained sub-buckets per outer time-bucket emitted
    /// at build time. `1` (or absent) = legacy single-count behaviour.
    /// When > 1, each cell row carries N additional numeric columns named
    /// `bucket_0`..`bucket_<N-1>` and the renderer animates inside a tile
    /// by switching which column drives the per-cell colour — zero data
    /// re-upload between frames.
    #[serde(default = "default_sub_buckets")]
    pub sub_buckets: u32,
}

fn default_sub_buckets() -> u32 {
    1
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

/// Bake-time per-class intensity domain for the GPU-splat HeatmapLayer.
///
/// The HeatmapLayer maps `(weight × gaussian_falloff × intensity)` through a
/// palette LUT. Without a pinned domain the renderer would either bake `[0,1]`
/// in (saturating immediately when `weightProperty` carries large values like
/// earthquake magnitudes) or trigger a runtime GPU readback to auto-detect
/// the max. Computing the domain at build time gives the renderer a stable
/// ramp with zero runtime cost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeatmapClassDomain {
    /// Class id — `"default"` for the un-classified single-channel mode,
    /// otherwise matches the FE channel id (typically a categorical value).
    pub id: String,
    /// Inclusive minimum splat intensity for this class.
    pub min: f64,
    /// Inclusive maximum splat intensity for this class. For the
    /// un-weighted default this is 1.0 (the gaussian peak). For a
    /// weight-property-driven layer this is the 95th-percentile weight
    /// across all features (95p is more visually useful than absolute max,
    /// which lets a single outlier dim the whole ramp).
    pub max: f64,
    /// Source weight property the domain was computed from, if any.
    /// `None` = constant unit weight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub property: Option<String>,
}

/// Container for the build-time HeatmapLayer domain metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeatmapDomain {
    pub classes: Vec<HeatmapClassDomain>,
}

/// Build-time statistics + rendering defaults for ONE property column,
/// carried inside [`StyleHints`].
///
/// Two shapes share this struct:
/// * **numeric** properties carry `min`/`p50`/`p90`/`p95`/`p97`/`p99`/`max`
///   and `suggested_domain` (`cardinality` absent);
/// * **categorical** (string) properties carry ONLY `name` + `cardinality`
///   (the numeric fields are absent from the JSON, not null-filled).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PropertyStyleHint {
    /// Property (tile column) name.
    pub name: String,
    /// Minimum observed value (numeric only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// 50th percentile (numeric only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p50: Option<f64>,
    /// 90th percentile (numeric only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p90: Option<f64>,
    /// 95th percentile (numeric only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95: Option<f64>,
    /// 97th percentile (numeric only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p97: Option<f64>,
    /// 99th percentile (numeric only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p99: Option<f64>,
    /// Maximum observed value (numeric only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Suggested render domain `[min, p97]`, each endpoint rounded OUTWARD to
    /// 2 significant figures. p97 (not max) encodes the project's manual
    /// "domain clamps at ~p97" tuning convention — a single outlier must not
    /// dim the whole ramp (numeric only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_domain: Option<[f64; 2]>,
    /// Distinct-value count (categorical only; the profiler caps it at
    /// 10 000 — "at least 10k" is already actionable for palette sizing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality: Option<u32>,
}

/// Optional build-time "style hints" block: per-property statistics plus
/// archive-level rendering defaults, computed by the opt-in
/// `stt-build --style-hints` profiler so a fresh dataset renders sensibly
/// without hand-tuning.
///
/// Hints are DEFAULTS — a renderer or user can always override them. The
/// whole block is additive: archives without it (and readers that don't know
/// it) are unaffected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleHints {
    /// Style-hints block schema version. Currently `1`.
    pub version: u32,
    /// Per-property statistics (numeric percentiles or categorical
    /// cardinality — see [`PropertyStyleHint`]).
    pub properties: Vec<PropertyStyleHint>,
    /// Suggested playback duration in seconds:
    /// `clamp(round(sqrt(bucket_count)), 20, 90)` where `bucket_count` is the
    /// time-range duration divided by `temporal_bucket_ms`. Absent when the
    /// bucket size is unknown/zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_playback_seconds: Option<u32>,
    /// Dominant produced layer kind: one of `"points"`, `"paths"`, `"trips"`,
    /// or `"polygons"` (absent when no kind could be derived). Kept as a
    /// string (not an enum) so a newer writer's vocabulary can't fail an
    /// older reader's decode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_hint: Option<String>,
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
    /// Total number of tiles. For packed manifests this is derived from the
    /// directory at write time (`PackWriter::finalize`); caller-set values are
    /// ignored there.
    pub tile_count: u64,
    /// Total feature records summed across tiles — a feature that lands in N
    /// tiles (zoom pyramid, clipping, temporal LOD) counts N times. Matches
    /// stt-validate's `feature_count_index`; derived from the directory at
    /// write time for packed manifests.
    pub feature_count: u64,
    /// Layer names
    pub layers: Vec<String>,
    /// Custom properties. A `BTreeMap` so the serialized JSON key order is
    /// deterministic across processes — the packed manifest embeds this map,
    /// and byte-reproducible builds must not depend on hash-map iteration order.
    pub properties: BTreeMap<String, String>,
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

    /// Optional bake-time HeatmapLayer intensity domains. When set, the
    /// renderer's HeatmapLayer skips its runtime `colorDomain` default of
    /// `[0, 1]` and uses these per-class entries instead — vital when the
    /// configured `weightProperty` carries values far outside that range
    /// (earthquake magnitudes, AIS speed, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heatmap_domain: Option<HeatmapDomain>,

    /// Optional build-time style hints (per-property statistics + suggested
    /// rendering defaults). Baked by `stt-build --style-hints`; always
    /// overridable by the renderer/user, ignored by older readers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style_hints: Option<StyleHints>,
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
            properties: BTreeMap::new(),
            temporal_bucket_ms: 3600 * 1000, // 1 hour default
            summary_tier: None,
            temporal_lod: None,
            heatmap_domain: None,
            style_hints: None,
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

    /// Attach a bake-time HeatmapLayer intensity-domain block.
    pub fn with_heatmap_domain(mut self, domain: HeatmapDomain) -> Self {
        self.heatmap_domain = Some(domain);
        self
    }

    /// Attach a build-time style-hints block.
    pub fn with_style_hints(mut self, hints: StyleHints) -> Self {
        self.style_hints = Some(hints);
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

    /// Render a TileJSON 3.0 descriptor (with a STAC-style `temporal`
    /// extension) for this archive.
    ///
    /// This is the self-describing, ecosystem-recognisable face of the archive:
    /// MapLibre / Leaflet / OpenLayers understand the core TileJSON fields, and
    /// the additive `temporal` block — an ISO-8601 `interval` à la STAC, plus
    /// the bucket size/step and any LOD pyramid — carries the time dimension the
    /// spatial-only standard lacks (unknown keys are ignored by existing
    /// clients). `tile_url_template` is the `{z}/{x}/{y}/{t}` URL the host serves
    /// tiles at; when `None`, a relative template is emitted.
    pub fn to_tilejson(&self, tile_url_template: Option<&str>) -> serde_json::Value {
        use serde_json::json;
        let tiles = tile_url_template.unwrap_or("{z}/{x}/{y}/{t}");
        let center_lon = (self.bounds.min_lon + self.bounds.max_lon) / 2.0;
        let center_lat = (self.bounds.min_lat + self.bounds.max_lat) / 2.0;

        let to_iso = |ms: u64| -> serde_json::Value {
            // Guard the u64→i64 cast: a timestamp beyond i64::MAX ms (year ~292M)
            // would wrap negative; surface it as a null open bound instead.
            if ms > i64::MAX as u64 {
                return serde_json::Value::Null;
            }
            chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms as i64)
                .map(|dt| json!(dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)))
                .unwrap_or(serde_json::Value::Null)
        };

        let vector_layers: Vec<serde_json::Value> = self
            .layers
            .iter()
            .map(|name| {
                json!({
                    "id": name,
                    "fields": {},
                    "minzoom": self.min_zoom,
                    "maxzoom": self.max_zoom,
                })
            })
            .collect();

        // STAC temporal extent: interval is an array of [start, end] pairs with
        // `null` for an open end. We carry the bucket size + ISO-8601 step and
        // any LOD pyramid as additive keys.
        let mut temporal = json!({
            "interval": [[to_iso(self.time_range.start), to_iso(self.time_range.end)]],
            "bucket_ms": self.temporal_bucket_ms,
        });
        if let Some(step) = iso8601_duration(self.temporal_bucket_ms) {
            temporal["step"] = json!(step);
        }
        if let Some(levels) = &self.temporal_lod {
            temporal["lod"] = json!(levels
                .iter()
                .map(|l| json!({ "bucket_ms": l.bucket_ms, "max_zoom": l.max_zoom_level }))
                .collect::<Vec<_>>());
        }

        json!({
            "tilejson": "3.0.0",
            "tiles": [tiles],
            "name": self.name,
            "description": self.description,
            "attribution": self.attribution,
            "scheme": "xyz",
            "version": "1.0.0",
            "minzoom": self.min_zoom,
            "maxzoom": self.max_zoom,
            "bounds": [
                self.bounds.min_lon,
                self.bounds.min_lat,
                self.bounds.max_lon,
                self.bounds.max_lat
            ],
            "center": [center_lon, center_lat, self.min_zoom],
            "vector_layers": vector_layers,
            "temporal": temporal,
        })
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

/// Format a millisecond duration as an ISO-8601 duration string for the
/// TileJSON `temporal.step` field (best-effort: days / hours / minutes /
/// seconds). Returns `None` for a zero bucket.
fn iso8601_duration(ms: u64) -> Option<String> {
    if ms == 0 {
        return None;
    }
    Some(if ms % 86_400_000 == 0 {
        format!("P{}D", ms / 86_400_000)
    } else if ms % 3_600_000 == 0 {
        format!("PT{}H", ms / 3_600_000)
    } else if ms % 60_000 == 0 {
        format!("PT{}M", ms / 60_000)
    } else if ms % 1000 == 0 {
        format!("PT{}S", ms / 1000)
    } else {
        format!("PT{:.3}S", ms as f64 / 1000.0)
    })
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
            sub_buckets: 1,
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
    fn test_metadata_summary_tier_quadbin_and_sub_buckets_roundtrip() {
        // The CARTO `quadbin` scheme and a non-default `sub_buckets` count must
        // survive the manifest JSON round-trip, and the scheme must serialize
        // with its documented lowercase spelling. (Moved here from
        // tests/spec_conformance.rs, whose remit is the per-layer Arrow schema.)
        let tier = SummaryTier {
            scheme: SummaryScheme::Quadbin,
            min_zoom: 0,
            max_zoom: 6,
            cell_resolution_per_zoom: vec![0, 1, 2, 3, 4, 5, 6],
            columns: vec![SummaryColumn {
                name: "magnitude".to_string(),
                agg: SummaryAggregation::Mean,
            }],
            layer_name: "summary".to_string(),
            sub_buckets: 24,
        };
        let bytes = Metadata::new("q")
            .with_summary_tier(tier)
            .to_json_bytes()
            .expect("serialize metadata");
        let decoded = Metadata::from_json_bytes(&bytes).expect("deserialize metadata");
        let dt = decoded.summary_tier.expect("summary tier round-trips");
        assert_eq!(dt.scheme, SummaryScheme::Quadbin, "quadbin scheme survives");
        assert_eq!(dt.sub_buckets, 24, "sub_buckets survives the round-trip");
        assert_eq!(dt.max_zoom, 6);

        let json = String::from_utf8(bytes).unwrap();
        assert!(json.contains("quadbin"), "scheme serializes lowercase: {json}");
        assert!(json.contains("sub_buckets"), "sub_buckets is a manifest field: {json}");
    }

    #[test]
    fn test_metadata_summary_tier_defaults_when_subfields_absent() {
        // A pre-`sub_buckets` manifest (tier present, but the `sub_buckets` and
        // `layer_name` fields absent) must decode to the legacy single-count
        // behaviour via serde's documented defaults.
        let json = br#"{
            "name": "old-summary",
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
            "temporal_bucket_ms": 3600000,
            "summary_tier": {
                "scheme": "h3",
                "min_zoom": 0,
                "max_zoom": 4,
                "cell_resolution_per_zoom": [0,1,2,3,4],
                "columns": [{"name":"magnitude","agg":"mean"}]
            }
        }"#;
        let m = Metadata::from_json_bytes(json).expect("legacy summary tier decodes");
        let dt = m.summary_tier.expect("summary tier present");
        assert_eq!(dt.scheme, SummaryScheme::H3);
        assert_eq!(dt.sub_buckets, 1, "absent sub_buckets defaults to 1");
        assert_eq!(dt.layer_name, "summary", "absent layer_name defaults to 'summary'");
    }

    #[test]
    fn test_properties_serialize_in_sorted_key_order() {
        // `properties` is a BTreeMap so the manifest JSON key order is
        // deterministic across processes (no HashMap iteration order in
        // anything serialized) — insertion order must NOT leak through.
        let metadata = Metadata::new("ord")
            .with_property("zebra", "1")
            .with_property("alpha", "2")
            .with_property("mid", "3");
        let s = String::from_utf8(metadata.to_json_bytes().unwrap()).unwrap();
        let a = s.find("\"alpha\"").unwrap();
        let m = s.find("\"mid\"").unwrap();
        let z = s.find("\"zebra\"").unwrap();
        assert!(a < m && m < z, "properties must serialize sorted: {s}");
    }

    #[test]
    fn test_metadata_ignores_unknown_fields() {
        // Forward compat: metadata written by a newer builder (or carrying
        // since-removed fields like the old `raster_tier` scaffold) must
        // still decode — serde skips unknown keys by default.
        let metadata = Metadata::new("fwd");
        let mut v: serde_json::Value =
            serde_json::from_slice(&metadata.to_json_bytes().unwrap()).unwrap();
        v["raster_tier"] = serde_json::json!({ "min_zoom": 0, "max_zoom": 5 });
        let decoded = Metadata::from_json_bytes(&serde_json::to_vec(&v).unwrap()).unwrap();
        assert_eq!(decoded.name, "fwd");
    }

    #[test]
    fn test_heatmap_domain_roundtrip() {
        let domain = HeatmapDomain {
            classes: vec![
                HeatmapClassDomain {
                    id: "pickup".to_string(),
                    min: 0.0,
                    max: 7.5,
                    property: Some("intensity".to_string()),
                },
                HeatmapClassDomain {
                    id: "dropoff".to_string(),
                    min: 0.0,
                    max: 9.0,
                    property: None,
                },
            ],
        };
        let metadata = Metadata::new("heat-test").with_heatmap_domain(domain.clone());
        let bytes = metadata.to_json_bytes().unwrap();
        let decoded = Metadata::from_json_bytes(&bytes).unwrap();
        let d = decoded.heatmap_domain.unwrap();
        assert_eq!(d.classes.len(), 2);
        assert_eq!(d.classes[0].id, "pickup");
        assert_eq!(d.classes[0].max, 7.5);
        assert_eq!(d.classes[0].property.as_deref(), Some("intensity"));
        assert_eq!(d.classes[1].id, "dropoff");
        assert_eq!(d.classes[1].property, None);
    }

    #[test]
    fn test_heatmap_domain_field_omitted_when_unset() {
        let metadata = Metadata::new("no-heat");
        let s = String::from_utf8(metadata.to_json_bytes().unwrap()).unwrap();
        assert!(!s.contains("heatmap_domain"), "got: {s}");
    }

    #[test]
    fn test_style_hints_roundtrip() {
        let hints = StyleHints {
            version: 1,
            properties: vec![
                PropertyStyleHint {
                    name: "magnitude".to_string(),
                    min: Some(0.1),
                    p50: Some(2.0),
                    p90: Some(4.1),
                    p95: Some(4.9),
                    p97: Some(5.3),
                    p99: Some(6.2),
                    max: Some(9.1),
                    suggested_domain: Some([0.1, 5.3]),
                    cardinality: None,
                },
                // Categorical: ONLY name + cardinality.
                PropertyStyleHint {
                    name: "category".to_string(),
                    cardinality: Some(7),
                    ..Default::default()
                },
            ],
            suggested_playback_seconds: Some(45),
            layer_hint: Some("points".to_string()),
        };
        let metadata = Metadata::new("hints-test").with_style_hints(hints.clone());
        let bytes = metadata.to_json_bytes().unwrap();

        // Wire shape: the categorical entry must carry NO numeric keys (absent,
        // not null-filled) — pinned cross-language contract with the TS reader.
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let cat = &v["style_hints"]["properties"][1];
        assert_eq!(cat["name"], "category");
        assert_eq!(cat["cardinality"], 7);
        let cat_keys: Vec<&String> = cat.as_object().unwrap().keys().collect();
        assert_eq!(cat_keys.len(), 2, "categorical carries only name+cardinality: {cat_keys:?}");
        // Numeric entry: no cardinality key.
        assert!(v["style_hints"]["properties"][0].get("cardinality").is_none());
        assert_eq!(v["style_hints"]["suggested_playback_seconds"], 45);
        assert_eq!(v["style_hints"]["layer_hint"], "points");

        let decoded = Metadata::from_json_bytes(&bytes).unwrap();
        assert_eq!(decoded.style_hints, Some(hints));
    }

    #[test]
    fn test_style_hints_field_omitted_when_unset() {
        // Old readers must be unaffected: an archive built without
        // --style-hints carries no `style_hints` key at all, and a legacy
        // metadata JSON without the key decodes to None (see the
        // `test_metadata_without_summary_tier_decodes` fixture, which also
        // lacks style_hints).
        let metadata = Metadata::new("no-hints");
        let s = String::from_utf8(metadata.to_json_bytes().unwrap()).unwrap();
        assert!(!s.contains("style_hints"), "got: {s}");
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

    #[test]
    fn tilejson_descriptor_has_core_fields_and_temporal_extension() {
        let m = Metadata::new("quakes")
            .with_description("USGS earthquakes")
            .with_attribution("USGS")
            .with_zoom_levels(0, 10)
            .with_temporal_bucket_ms(hour())
            .with_time_range(TimeRange::new(1_700_000_000_000, 1_700_086_400_000))
            .with_temporal_lod(vec![TemporalLodLevel {
                bucket_ms: day(),
                max_zoom_level: 6,
            }])
            .unwrap();
        let tj = m.to_tilejson(Some("https://cdn/{z}/{x}/{y}/{t}.stt"));

        // Core TileJSON 3.0 fields every web client recognises.
        assert_eq!(tj["tilejson"], "3.0.0");
        assert_eq!(tj["tiles"][0], "https://cdn/{z}/{x}/{y}/{t}.stt");
        assert_eq!(tj["minzoom"], 0);
        assert_eq!(tj["maxzoom"], 10);
        assert_eq!(tj["scheme"], "xyz");
        assert_eq!(tj["vector_layers"][0]["id"], "default");
        assert_eq!(tj["bounds"].as_array().unwrap().len(), 4);

        // Additive STAC-style temporal extension.
        assert_eq!(tj["temporal"]["step"], "PT1H");
        assert_eq!(tj["temporal"]["bucket_ms"], 3_600_000u64);
        let interval = &tj["temporal"]["interval"][0];
        assert!(interval[0].as_str().unwrap().starts_with("2023-11-"));
        assert!(interval[1].is_string());
        assert_eq!(tj["temporal"]["lod"][0]["bucket_ms"], day());
    }

    #[test]
    fn iso8601_duration_formats_common_buckets() {
        assert_eq!(iso8601_duration(hour()).as_deref(), Some("PT1H"));
        assert_eq!(iso8601_duration(day()).as_deref(), Some("P1D"));
        assert_eq!(iso8601_duration(60_000).as_deref(), Some("PT1M"));
        assert_eq!(iso8601_duration(0), None);
    }

    #[test]
    fn tilejson_time_beyond_i64_is_null_not_garbage() {
        // A timestamp past i64::MAX ms must surface as a null open bound rather
        // than wrapping to a negative (bogus) date.
        let m = Metadata::new("x").with_time_range(TimeRange::new(u64::MAX, u64::MAX));
        let tj = m.to_tilejson(None);
        assert!(tj["temporal"]["interval"][0][0].is_null());
        assert!(tj["temporal"]["interval"][0][1].is_null());
    }
}
