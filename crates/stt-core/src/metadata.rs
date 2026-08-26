//! Archive metadata structures
//!
//! This module provides types for storing and managing archive metadata.

use crate::arrow_tile::DecodedLayer;
use crate::error::{Error, Result};
use crate::types::{BoundingBox, TimeRange};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

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
    /// Directory/manifest variant that stores this summary representation.
    ///
    /// `#[serde(default)]` is load-bearing for the v2 read window, not tidiness.
    /// A formatVersion-2 manifest predates the variant axis entirely, so its
    /// `summary_tier` carries no `variant_id` — and without a default, serde
    /// failed the WHOLE manifest ("missing field `variant_id`"), which meant no
    /// Rust tool could open a v2 archive that happened to have a summary tier.
    /// That is six of the published archives, and it made `PackedReader`,
    /// `stt-validate` and `stt-optimize` all refuse them while the TypeScript
    /// reader opened them happily.
    ///
    /// Defaulting to 0 is the legacy reading, not an invention: a v2 directory
    /// has no variant column, so every entry in such an archive — summary tiles
    /// included — decodes as variant 0, which is exactly what
    /// `effective_variants` and the TS `effectiveVariants` already infer. A v3
    /// writer always states the value explicitly (1 is canonical for summary).
    #[serde(default)]
    pub variant_id: u32,
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
            .unwrap_or_else(|| *self.cell_resolution_per_zoom.last().unwrap())
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
    /// Suggested playback duration in seconds. Absent when the bucket size is
    /// unknown/zero.
    ///
    /// Two formulas exist; the archive does not say which produced the value,
    /// because both answer the same question and a reader only ever consumes the
    /// number:
    ///
    /// * **Legacy (the default emission today):**
    ///   `clamp(round(sqrt(bucket_count)), 20, 90)`.
    /// * **Frame-rate refit** (`stt-build --derived-playback-params`, becoming the
    ///   default at the next rebuild window): `clamp(K / 20, K/30, K/12)` further
    ///   clamped to `[5, 300]`, where `K` = `bucket_count`. This holds the implied
    ///   data frame rate near 20 frames/s instead of letting a 10-bucket archive
    ///   crawl at one frame per 2 s or a 100 000-bucket archive imply 3 333
    ///   data-fps.
    ///
    /// `bucket_count` is the time-range duration divided by `temporal_bucket_ms`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_playback_seconds: Option<u32>,
    /// Suggested playback WINDOW width in milliseconds — how much time the
    /// player should show at once, the companion to
    /// [`Self::suggested_playback_seconds`] (which is how fast it sweeps).
    ///
    /// The widest window whose resident payload still fits a reference client
    /// memory budget, capped at 24 temporal buckets and floored at one:
    /// `min(argmax{W : β̄·W ≤ M_REF}, 24·Δ)`, where `β̄` is the archive's mean
    /// payload bytes per millisecond of span, `Δ` is `temporal_bucket_ms`, and
    /// `M_REF` is the documented reference budget (see
    /// `stt_optimize::analysis::properties::M_REF`).
    ///
    /// A DEFAULT, like every other hint: an explicit reader/user `timeWindow`
    /// always wins, and an authored sub-range is respected verbatim. Absent on
    /// every archive built before the field existed and on any build where the
    /// payload total was not available (streaming builds), so a reader must have
    /// a fallback — which is the bucket-multiple default it already had.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_time_window_ms: Option<u64>,
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
    /// What a reader may assume about this tier's CONTENTS (DT-1).
    ///
    /// Absent = [`TierContract::Union`] = the existing normative MUST from
    /// `docs/spec/time-model.md` §4: exactly the base features, re-bucketed,
    /// with no reduction, aggregation or thinning. Every manifest written
    /// before DT-1 therefore stays valid and byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract: Option<TierContract>,
    /// Reduction method, REQUIRED when `contract` is
    /// [`TierContract::Reduced`] and meaningless otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<ReductionMethod>,
}

/// What a declared tier's tiles contain relative to the base tier (DT-1).
///
/// This is the single concept that replaces four separately-grown tier
/// mechanisms. A reader MUST NOT substitute a non-base tier for base content
/// unless it understands the declared contract (and `method`, if reduced) —
/// an undeclared or unrecognized tier is simply never substituted, which keeps
/// conservative-superset soundness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierContract {
    /// Exactly the base features, re-bucketed. Lossless; the default.
    Union,
    /// FEWER features than the base tier, by a declared `method`. Explicitly
    /// not lossless — the base tier stays complete and addressable beside it.
    Reduced,
}

/// How a `reduced` tier was derived (DT-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReductionMethod {
    /// M4 aggregation.
    M4,
    /// MinMaxLTTB downsampling.
    MinMaxLttb,
}

/// How base-tier features are distributed across zooms (DT-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Partition {
    /// Today's behaviour: every feature is replicated at every zoom in its band.
    Replicated,
    /// Each feature lives at exactly one home zoom; the reader unions across
    /// `[minZoom..z]`.
    ///
    /// ⚠️ A manifest declaring this MUST also list the must-understand
    /// capability [`CAPABILITY_ADDITIVE_PARTITION`]: home-zoom changes what the
    /// base tier at ONE zoom contains, so a parent-fallback reader that does
    /// not understand it would render a sparse slice as if it were complete —
    /// exactly the silent-misdecode class capabilities exist to turn into a
    /// loud refusal.
    HomeZoom,
}

/// Must-understand capability required alongside [`Partition::HomeZoom`].
pub const CAPABILITY_ADDITIVE_PARTITION: &str = "additive-partition";

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
    /// Bounding box.
    ///
    /// **MUST contain every vertex the archive decodes to** — it is a
    /// conservative superset of the data's horizontal extent, never tighter.
    /// Consumers pre-intersect query boxes against it (tile selection, frustum
    /// pre-culling, the opening camera), so an under-stated box silently
    /// discards tiles that really do carry data. Writers therefore compute it
    /// from geometry **vertices**, not from feature centroids (see
    /// `stt_build::input::BoundsMode`; backlog K11). An antimeridian-spanning
    /// dataset reports the loose full-width box rather than a wrapped
    /// interval — looser, still sound.
    pub bounds: BoundingBox,
    /// OPTIONAL vertical extent `[min_z, max_z]` of the dataset, in the units
    /// its elevations are authored in (metres for every current archive).
    ///
    /// Additive and purely descriptive: it makes a volumetric dataset
    /// *discoverable* (and unblocks altitude-aware selection) without changing
    /// how a single tile decodes. Present only when the source carried
    /// altitude — a 3-element geometry position, or a property column the
    /// build declared as its elevation source. **Absent** on every 2D dataset
    /// and on every archive written before this field existed, which is what
    /// keeps their manifest bytes identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_range: Option<[f64; 2]>,
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
    /// Count of DISTINCT source features (before tile placement, clipping and
    /// pyramid/LOD replication). Set by the builder from the ingested feature
    /// set; unlike [`Self::feature_count`] it does NOT double-count a feature
    /// that spans several tiles, so it is the correct number for a user-facing
    /// "N features" total. `None` for archives written before this field
    /// existed (and for v1 kill-switch builds) — consumers fall back to the
    /// index-weighted `feature_count` then, ideally labelling it as such.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct_feature_count: Option<u64>,
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
    /// How base-tier features are distributed across zooms (DT-1). Absent =
    /// [`Partition::Replicated`] = today's behaviour, so every existing
    /// manifest stays valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition: Option<Partition>,

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

    /// The workload model the **measured** blob-ordering picker was run under,
    /// recorded so a layout can be re-derived, re-audited, and detected as
    /// stale. See [`OrderingWorkload`]. `None` — and therefore omitted from the
    /// JSON — on every build that did not resolve its ordering by simulation
    /// (`auto` and explicit orderings), so pre-field archives are byte-identical.
    ///
    /// ⚠️ **This is the reader-compat MIRROR, not the canonical key.** The
    /// canonical home is `Manifest::ordering_workload` (serialized top-level as
    /// `orderingWorkload`, beside `blobOrdering` — the layout fact it
    /// co-versions). This copy exists because the shipped TS reader
    /// (`poopdeck:packages/core/src/archive.ts`, `manifestBuildAssumedGapBytes`) resolves
    /// the build-assumed coalescing gap through
    /// `metadata.ordering_workload.coalesce_gap_bytes`; dropping it would
    /// silently disable the adaptive-coalesce co-versioning guard rather than
    /// fail loudly. Both copies are written from one value at one site in
    /// `PackWriter::finalize`. **Removal trigger:** when the TS reader reads the
    /// top-level `orderingWorkload`, delete this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordering_workload: Option<OrderingWorkload>,

    /// OPTIONAL semantic fingerprint of the dataset's DECODED CONTENT (SH-1).
    ///
    /// Structural validation is sound but incomplete: an archive whose
    /// coordinates have been silently scrambled hashes, decodes and
    /// schema-checks perfectly. This block records replication-invariant
    /// statistics of the source features so `stt-validate` can recompute them
    /// from the decoded tiles and compare — see [`ContentFingerprint`].
    ///
    /// Additive: absent on every archive written before it existed (which is
    /// every published archive), and the validator warns rather than errors
    /// when it is missing, mirroring the CRS84 precedent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_fingerprint: Option<ContentFingerprint>,
}

/// The blob-ordering workload model an archive was laid out under.
///
/// Co-versioning, not decoration. Two independent things can invalidate a
/// `measured` layout after the fact:
///
/// 1. **The weights change.** A later re-fit of the query weights would pick a
///    different ordering for the same data. Recording the triple lets
///    `stt-optimize order-audit` say *which* table an archive was laid out
///    under instead of silently comparing against today's.
/// 2. **The reader's coalescing gap changes.** The simulator prices at the
///    BUILD-ASSUMED gap; if the client later fuses across a different gap, the
///    layout was optimised for a reader that no longer exists. Recording
///    `coalesce_gap_bytes` turns that from a silent mismatch into a flagged
///    drift.
///
/// Additive and optional: absent on every archive whose ordering was not
/// simulated, and ignored by every existing reader.
/// Keys are snake_case, matching every other key in the folded `metadata`
/// block (the manifest's own top-level keys are camelCase; `metadata` is
/// pinned as snake_case by `docs/spec/manifest.schema.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderingWorkload {
    /// Weight on the "scrub a viewport across all time" query.
    pub scrub: u32,
    /// Weight on the "pan at one instant" query.
    pub pan: u32,
    /// Weight on the "play a sliding time window" query. `0` means the dataset
    /// has no playback dimension (a single time bucket).
    pub playback: u32,
    /// Sliding-window width, in time buckets, the playback query was priced at.
    pub playback_window_buckets: u64,
    /// Multiplier applied to the worst single playhead advance (the
    /// buffered-runway / stall term).
    pub runway_multiplier: u64,
    /// The range-read coalescing gap the simulation assumed, in bytes — the
    /// reader-mirroring constant. Drift here invalidates the layout's premise.
    pub coalesce_gap_bytes: u64,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            attribution: String::new(),
            // Same rounded literals as `BoundingBox::default()`, and for the
            // same reason: a serialized whole-world placeholder pinned by the
            // golden manifests, NOT the projection clamp
            // (`projection::MERCATOR_MAX_LAT`).
            bounds: BoundingBox {
                min_lon: -180.0,
                min_lat: -85.0511,
                max_lon: 180.0,
                max_lat: 85.0511,
            },
            // 2D until a build proves otherwise: an unset vertical extent is
            // omitted from the JSON entirely, so default manifests keep their
            // pinned bytes.
            z_range: None,
            time_range: TimeRange::new(0, 0),
            min_zoom: 0,
            max_zoom: 14,
            tile_count: 0,
            feature_count: 0,
            distinct_feature_count: None,
            layers: vec!["default".to_string()],
            properties: BTreeMap::new(),
            temporal_bucket_ms: 3600 * 1000, // 1 hour default
            summary_tier: None,
            temporal_lod: None,
            partition: None,
            heatmap_domain: None,
            style_hints: None,
            ordering_workload: None,
            // Emitted only by an explicit `stt-build --content-fingerprint`,
            // so default manifests keep their pinned bytes.
            content_fingerprint: None,
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

    /// Set bounds.
    ///
    /// The value MUST contain every vertex the archive decodes to (see
    /// [`Self::bounds`]); passing a centroid-derived box under-states the
    /// extent and makes downstream pre-intersection unsound.
    pub fn with_bounds(mut self, bounds: BoundingBox) -> Self {
        self.bounds = bounds;
        self
    }

    /// Declare the dataset's vertical extent `[min_z, max_z]`.
    ///
    /// Ignored (and omitted from the JSON) when the range is `None` or
    /// non-finite: a NaN altitude is not a claim worth publishing, and an
    /// omitted key is what every 2D archive already writes. An inverted input
    /// is normalised rather than rejected, so the declared range always
    /// contains the observed one.
    pub fn with_z_range(mut self, z_range: Option<[f64; 2]>) -> Self {
        self.z_range = match z_range {
            Some([lo, hi]) if lo.is_finite() && hi.is_finite() => Some([lo.min(hi), lo.max(hi)]),
            _ => None,
        };
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
    /// when losslessly re-bucketing features into coarser request groups.
    pub fn with_temporal_lod(mut self, levels: Vec<TemporalLodLevel>) -> Result<Self> {
        validate_temporal_lod(self.temporal_bucket_ms, &levels)?;
        self.temporal_lod = if levels.is_empty() {
            None
        } else {
            Some(levels)
        };
        Ok(self)
    }

    /// Return the LOD level that applies at `zoom`, if any. The largest
    /// matching `bucket_ms` (coarsest level) wins — at a global zoom, you
    /// want the coarsest available request bucket, not the finest.
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

    /// Record the workload model a **measured** blob ordering was resolved
    /// under (see [`OrderingWorkload`]). Set by `PackWriter::finalize`; callers
    /// do not normally need it, and a build that never resolves an ordering by
    /// simulation leaves it `None` so its manifest bytes do not move.
    pub fn with_ordering_workload(mut self, workload: OrderingWorkload) -> Self {
        self.ordering_workload = Some(workload);
        self
    }

    /// Attach the semantic [`ContentFingerprint`] of the source features.
    ///
    /// ⚠️ A tool that transforms an archive **losslessly** (reorder, repack,
    /// re-optimize) MUST carry the existing fingerprint through verbatim and
    /// MUST NOT recompute it from its own output — see
    /// [`ContentFingerprint`]'s "relapse-proof rule".
    pub fn with_content_fingerprint(mut self, fingerprint: ContentFingerprint) -> Self {
        self.content_fingerprint = Some(fingerprint);
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
            // Four elements, always: TileJSON 3.0 defines `bounds` as exactly
            // [west, south, east, north], and a 6-element form would be
            // rejected by the very clients this view exists for. A dataset's
            // vertical extent (`z_range`) therefore does NOT appear here — it
            // surfaces in the STAC Item's 2n-element bbox, which does permit it.
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
        // DT-1 rule (1): `method` is meaningful only on a `reduced` tier, and
        // is REQUIRED there. A reduced tier with no method is unreadable — a
        // reader cannot know what was dropped.
        match level.contract {
            Some(TierContract::Reduced) => {
                if level.method.is_none() {
                    return Err(Error::Other(format!(
                        "temporal_lod[{i}] declares contract \"reduced\" but no method; a reader \
                         cannot substitute a reduced tier whose reduction it does not understand"
                    )));
                }
            }
            _ => {
                if level.method.is_some() {
                    return Err(Error::Other(format!(
                        "temporal_lod[{i}] declares a reduction method but its contract is \
                         \"union\" (or absent); a union tier is exactly the base features \
                         re-bucketed, so a method is meaningless"
                    )));
                }
            }
        }
        prev = Some(level.bucket_ms);
    }
    Ok(())
}

/// DT-1 rule (3): a manifest declaring [`Partition::HomeZoom`] MUST also list
/// the must-understand capability [`CAPABILITY_ADDITIVE_PARTITION`].
///
/// Home-zoom changes what the base tier at ONE zoom contains. A reader that
/// does not understand it would render a sparse slice as if it were complete —
/// the silent-misdecode class that capabilities exist to convert into a loud
/// refusal at open.
pub fn validate_partition_capability(
    partition: Option<Partition>,
    capabilities: &[String],
) -> Result<()> {
    if partition == Some(Partition::HomeZoom)
        && !capabilities
            .iter()
            .any(|c| c == CAPABILITY_ADDITIVE_PARTITION)
    {
        return Err(Error::Other(format!(
            "metadata.partition is \"home-zoom\" but capabilities does not list \
             {CAPABILITY_ADDITIVE_PARTITION:?}; without it an older reader would silently render \
             a sparse per-zoom slice as if it were the complete base tier"
        )));
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

// =============================================================================
// Semantic content fingerprints (SH-1)
// =============================================================================
//
// WHY THIS EXISTS. `stt-validate` verifies *structure*: content addressing,
// CRCs, Arrow schemas, counts, temporal bounds. All of that passed on 106 AV
// archives whose 3D `xyz` coordinate leaves had been read back with a stride of
// 2 — flattened and scrambled. The corruption was found by a human eyeballing
// `stt-optimize export` bboxes. The fingerprint turns that human step into a
// check the validator runs from the archive alone.
//
// WHAT GOES IN, AND WHY ONLY THIS. A source feature lands in N tiles (zoom
// pyramid, tile clipping, temporal LOD), so only **replication-invariant**
// statistics survive tiling unchanged: min/max and distinct counts do, sums and
// means do not. Clip-synthesised vertices lie ON original segments, so the
// decoded vertex bbox equals the source vertex bbox under a full decode and is
// contained by it on any subset. That containment property is what makes a
// `--sample`d run a sound (if weaker) check rather than a guess.

/// Schema version of [`ContentFingerprint`] this build emits and understands.
/// A validator that meets a *higher* version warns and skips the comparison
/// rather than mis-checking it against rules it does not know.
pub const CONTENT_FINGERPRINT_VERSION: u32 = 1;

/// Cap on the per-column distinct-value counts recorded in
/// [`ContentFingerprint::categorical_cardinality`] — the same 10 000 ceiling
/// [`PropertyStyleHint::cardinality`] uses ("at least 10k" is already
/// actionable, and an uncapped set is an unbounded-memory hazard).
pub const FINGERPRINT_CARDINALITY_CAP: u32 = 10_000;

/// HyperLogLog precision: `2^14 = 16 384` registers, 16 KiB of state.
const HLL_PRECISION: u32 = 14;
const HLL_REGISTERS: usize = 1 << HLL_PRECISION;

/// Standard error of the distinct-id sketch, `1.04 / sqrt(2^14)` ≈ 0.81 %.
pub const HLL_STANDARD_ERROR: f64 = 1.04 / 128.0;

/// The band a distinct-count comparison must clear before it is reported:
/// three standard errors (≈ 2.4 %). The sketch is the ONE approximation in an
/// otherwise exact validator, so any finding derived from it prints this bound.
///
/// This is also the threshold a **shortfall ERROR** must clear — deliberately
/// the sketch's own bound rather than a hand-picked percentage.
pub const HLL_ERROR_BOUND: f64 = 3.0 * HLL_STANDARD_ERROR;

/// Distinct-id count at or below which [`FingerprintAccumulator`] counts ids
/// **exactly** (a `BTreeSet<u64>`) instead of reading the sketch.
///
/// WHY. The HLL's guarantee is *relative*; its resolution at small cardinality
/// is not. One hash collision costs one id, which is 3.3 % at n = 30 — wider
/// than [`HLL_ERROR_BOUND`] — so a shortfall ERROR keyed to the relative band
/// would fire on ~2.7 % of honest 30-feature archives
/// (`P(collision) ≈ 1 - exp(-n²/2m)`). Counting exactly below the cap removes
/// the estimator noise entirely and leaves the relative band doing the one job
/// it is sound for: tolerating a handful of genuinely colliding source ids.
///
/// The ceiling is [`FINGERPRINT_CARDINALITY_CAP`]'s, for the same reason
/// (bounded memory: ≤ 10 000 `u64` ≈ 200 KiB of B-tree, and it STOPS growing —
/// the national radar archive's 44 M ids fall straight through to the sketch).
/// Above it a single collision is ≤ 0.01 % of the count and the sketch's
/// relative bound is sound on its own.
pub const FINGERPRINT_EXACT_ID_CAP: usize = FINGERPRINT_CARDINALITY_CAP as usize;

/// Prefix of the distinct-count **shortfall** error, so a caller can recognise
/// that one finding without re-parsing prose. `stt-validate` matches on it to
/// service `--allow-distinct-shortfall`; nothing else may key off it.
pub const FEATURE_LOSS_PREFIX: &str = "FEATURE LOSS:";

// ---------------------------------------------------------------------------
// Distinct-id BASIS — what `distinct_feature_count` may be compared against.
// ---------------------------------------------------------------------------

/// `manifest.capabilities` entry by which a writer declares that the `id`
/// column is a GLOBALLY distinct key — one wire id per source feature, stable
/// across every tile and every pyramid level.
///
/// **Nothing emits this today.** It is the forward seam for TB-5 (global dense
/// id renumbering): once the writer renumbers ids into a dataset-wide bijection
/// it can declare this, and the distinct-id comparison below becomes sound
/// enough to be an error again.
///
/// ⚠️ Cross-crate note. The rest of the capability vocabulary lives in
/// `crate::pack` alongside [`KNOWN_CAPABILITIES`](crate::pack::KNOWN_CAPABILITIES),
/// and a reader **refuses at open** any archive declaring a capability outside
/// that set. This constant is deliberately NOT in that list: adding it would be
/// a promise the toolchain has not yet earned, and emitting it before the list
/// grows would make every attested archive unopenable. TB-5 owns both halves —
/// it must append this value to `KNOWN_CAPABILITIES` in the same change that
/// starts emitting it. Until then the *property* seam below is the only live
/// arming path, exactly as check 13 arms on `bounds_mode`.
pub const CAPABILITY_GLOBAL_FEATURE_IDS: &str = "global-feature-ids";

/// The `metadata.properties` key by which a writer attests the scope of its
/// `id` column.
///
/// `properties` is the free-form `BTreeMap<String, String>` every manifest
/// already carries — inert to every reader that does not look for it, which is
/// exactly why check 13's `bounds_mode` attestation uses the same seam. A
/// capability would instead be *rejected* at open by any reader that does not
/// implement it (see [`CAPABILITY_GLOBAL_FEATURE_IDS`]).
pub const FEATURE_ID_SCOPE_PROPERTY: &str = "feature_id_scope";

/// The [`FEATURE_ID_SCOPE_PROPERTY`] value that attests globally distinct ids.
pub const FEATURE_ID_SCOPE_GLOBAL: &str = "global";

/// The [`FEATURE_ID_SCOPE_PROPERTY`] value that records the opposite: wire ids
/// are unique only WITHIN a tile. Stamped explicitly (rather than omitted) so a
/// build that considered the question and answered "no" is distinguishable from
/// one that predates the question entirely.
pub const FEATURE_ID_SCOPE_LOCAL: &str = "local";

/// The `metadata.properties` key by which a writer records **how** it built the
/// wire `id` column — the FACT the basis is keyed on, as opposed to
/// [`FEATURE_ID_SCOPE_PROPERTY`], which is an assertion.
///
/// # Why a fact and not an assertion (BLOCKER 1)
///
/// `feature_id_scope` was reachable only through `--feature-id-scope auto`
/// proving that *every source feature carries an id*, and no ingest path in
/// `stt-build` populates `ParsedFeature.geojson.id` — the GeoParquet, PostGIS
/// and DuckDB readers all construct `Feature { id: None, .. }`. So `auto` was
/// structurally always `local`, the strict comparison had no live producer, and
/// feature loss on line/polygon archives went undetected (the per-zoom row floor
/// is loose exactly where clipping replicates a feature across tiles). Worse,
/// the reported basis then *described* the point-archive row-index mechanism on
/// archives that never used it.
///
/// The writer does not need to be asked: it knows which id-construction path it
/// took. This key carries that, and [`distinct_id_basis`] reads it.
///
/// Rides `properties` — the free-form `BTreeMap<String, String>` every manifest
/// already carries — for the same reason `bounds_mode` and `feature_id_scope`
/// do: a reader **rejects** an archive declaring a capability it does not
/// implement, so minting one here would make every honest archive unopenable by
/// the deployed fleet, while a property is inert to any reader that ignores it.
pub const FEATURE_ID_CONSTRUCTION_PROPERTY: &str = "feature_id_construction";

/// [`FEATURE_ID_CONSTRUCTION_PROPERTY`]: the wire id is the source feature's own
/// id (integers verbatim, strings through the spec-fixed FNV-1a-64). **A
/// dataset-wide key.**
pub const FEATURE_ID_CONSTRUCTION_SOURCE: &str = "source";

/// [`FEATURE_ID_CONSTRUCTION_PROPERTY`]: the wire id is the whole-feature ANCHOR
/// HASH `FNV(timestamp, lon, lat)`, which the tiler copies into every clipped
/// piece so one source feature keeps one id across every tile and zoom. **A
/// dataset-wide key** — this is the value that arms the comparison on line and
/// polygon archives, where nothing armed it before.
pub const FEATURE_ID_CONSTRUCTION_ANCHOR_HASH: &str = "anchor-hash";

/// [`FEATURE_ID_CONSTRUCTION_PROPERTY`]: the wire id is the PER-TILE ROW INDEX
/// (`build_point_layer`'s measured saving — the synthetic hash id was ~40 % of a
/// point's compressed bytes). Unique within a tile, repeated across tiles: **not
/// a key.**
pub const FEATURE_ID_CONSTRUCTION_ROW_INDEX: &str = "row-index";

/// [`FEATURE_ID_CONSTRUCTION_PROPERTY`]: the wire id is minted per CLIPPED
/// SEGMENT of a trajectory, so one source feature becomes many ids and the
/// writer cannot enumerate them before tiling: **not a key.**
pub const FEATURE_ID_CONSTRUCTION_SEGMENT_HASH: &str = "segment-hash";

/// Does a [`FEATURE_ID_CONSTRUCTION_PROPERTY`] value name a construction that
/// puts exactly one distinct wire id on each source feature?
///
/// Unknown spellings — a future construction this build does not know, a typo,
/// an absent key — answer `false`. A basis is a licence to turn a deviation into
/// an ERROR, so it must never be granted by a value this build cannot interpret.
///
/// # ⚠️ The writer's obligation
///
/// `source` and `anchor-hash` name constructions that put one id on each source
/// feature *provided no two of them collide*, and collide they can: the anchor
/// hash is `FNV(timestamp, lon, lat)`, so two features sharing a timestamp and a
/// representative point map to one id. A writer stamping either value is
/// therefore asserting it has CHECKED pairwise distinctness, exactly as a writer
/// stamping `bounds_mode = vertex` asserts it folded vertices.
///
/// `stt-build` discharges that by proving it (`input::feature_id_report`), and
/// when the proof declines — a collision, or a feature set above the attestation
/// cap — it records the construction honestly and stamps
/// `feature_id_scope = local` beside it, which [`distinct_id_basis`] resolves
/// FIRST and which disarms the comparison. So the honest label and the safe
/// verdict coexist; only a third-party writer that stamps a key construction it
/// has not verified could get this wrong, and it would be wrong in its own
/// manifest.
pub fn construction_is_a_dataset_wide_key(value: &str) -> bool {
    value.eq_ignore_ascii_case(FEATURE_ID_CONSTRUCTION_SOURCE)
        || value.eq_ignore_ascii_case(FEATURE_ID_CONSTRUCTION_ANCHOR_HASH)
}

/// What [`ContentFingerprint::distinct_feature_count`] may legitimately be
/// compared against.
///
/// # The category error this type exists to stop
///
/// `distinct_feature_count` counts **source features**. The decoded `id` column
/// does not: `build_point_layer` substitutes the **per-tile row index** for
/// every point whose source carried no id (a deliberate, measured saving — the
/// synthetic FNV id was ~40 % of a point's compressed bytes on Waymo LiDAR).
/// Archive-wide, the number of distinct ids on such an archive is therefore
/// `max over tiles of (rows in that tile)`, not the feature count.
///
/// Measured, on a real 600-feature CONUS build (`--min-zoom 2 --max-zoom 8`):
/// **5** distinct decoded ids against a declared 600, which the pre-basis check
/// reported as "99.2 % of the declared features are MISSING". Nothing had been
/// dropped. The two numbers were never the same quantity.
///
/// # But NOT on every archive — the row index is one of four constructions
///
/// The paragraph above is the POINT-archive story, and it was for a while told
/// about every archive. It is false for the other three quarters of the writer's
/// behaviour: an id-less LINE or POLYGON gets the whole-feature ANCHOR HASH,
/// which the tiler copies into every clipped piece precisely so one source
/// feature keeps one id everywhere. On those archives the two counts ARE the
/// same quantity, and refusing to compare them is what let a measured 50–78 %
/// feature loss through with exit 0.
///
/// # The rule
///
/// The validator must NEVER infer [`DistinctIdBasis::DecodedIds`] from the two
/// numbers happening to agree — that is how a check learns to pass. The basis
/// comes from the WRITER's recorded id construction (or an explicit
/// attestation); see [`distinct_id_basis`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum DistinctIdBasis {
    /// The conservative basis. `distinct_feature_count` counts source features,
    /// the `id` column counts something else (a per-tile row index, or one id
    /// per clipped trajectory segment), and a deviation between them is a
    /// NOTE — not a warning, and certainly not an error.
    #[default]
    SourceFeatures,
    /// The writer's id construction puts one wire id on each source feature,
    /// dataset-wide, so the two counts ARE the same quantity and a shortfall on
    /// a complete decode is real feature loss.
    DecodedIds,
}

/// Which basis a given archive's declaration may be judged on.
///
/// Three inputs, in this precedence — the first that answers, wins:
///
/// 1. `metadata.properties["feature_id_scope"]`, the OPERATOR's assertion, in
///    **both** directions. `global` arms the comparison without proof
///    (`stt-build --feature-id-scope global`), and `local` disarms it whatever
///    the construction says — that is the documented rollback, and a rollback
///    that could be overridden would not be one.
///    ⚠️ `local` outranks the capability below as well. Nothing emits that
///    capability today, so this is contrived — but the rollback is the one way
///    an operator turns the comparison off, and nothing may outvote it.
/// 2. [`CAPABILITY_GLOBAL_FEATURE_IDS`] in `manifest.capabilities` — TB-5's
///    seam; nothing emits it today.
/// 3. ⭐ `metadata.properties["feature_id_construction"]`, the WRITER's fact:
///    `source` or `anchor-hash` ⇒ [`DistinctIdBasis::DecodedIds`]. This is the
///    live arming path, and the one that needs no operator in the loop — see
///    [`FEATURE_ID_CONSTRUCTION_PROPERTY`] for why keying the basis on an
///    assertion instead left it structurally unreachable.
///
/// Anything else — absent, `row-index`, `segment-hash`, a spelling this build
/// does not know, an archive that predates the keys — is
/// [`DistinctIdBasis::SourceFeatures`]. Silence is never an attestation.
pub fn distinct_id_basis(metadata: &Metadata, capabilities: &[String]) -> DistinctIdBasis {
    // (1) The operator's assertion, both ways. `local` is the rollback and must
    // win over the writer's fact below; `global` is "assert without proof".
    if let Some(scope) = metadata.properties.get(FEATURE_ID_SCOPE_PROPERTY) {
        if scope.eq_ignore_ascii_case(FEATURE_ID_SCOPE_GLOBAL) {
            return DistinctIdBasis::DecodedIds;
        }
        if scope.eq_ignore_ascii_case(FEATURE_ID_SCOPE_LOCAL) {
            return DistinctIdBasis::SourceFeatures;
        }
        // An unrecognised spelling asserts nothing; fall through to the fact.
    }
    // (2) The capability seam TB-5 will use.
    if capabilities
        .iter()
        .any(|c| c == CAPABILITY_GLOBAL_FEATURE_IDS)
    {
        return DistinctIdBasis::DecodedIds;
    }
    // (3) The writer's own id construction.
    if metadata
        .properties
        .get(FEATURE_ID_CONSTRUCTION_PROPERTY)
        .is_some_and(|kind| construction_is_a_dataset_wide_key(kind))
    {
        return DistinctIdBasis::DecodedIds;
    }
    DistinctIdBasis::SourceFeatures
}

/// Columns the tile schema reserves; everything else in a decoded layer is a
/// property column. Mirrors `stt_core::arrow_tile`'s reserved set.
const RESERVED_TILE_COLUMNS: &[&str] = &[
    "id",
    "start_time",
    "end_time",
    "geometry",
    "vertex_time",
    "vertex_value",
    "vertex_value_matrix",
    "triangles",
    "part_offsets",
];

/// A semantic fingerprint of an archive's DECODED CONTENT, computed by the
/// writer from the **source features, before tiling and encode**, and
/// recomputed by `stt-validate` from the decoded tiles.
///
/// # The relapse-proof rule (normative)
///
/// A tool that transforms an archive **losslessly** — reorder, repack,
/// re-optimize — MUST carry this block through **verbatim** and MUST NOT
/// recompute it from its own output. Recomputing is exactly how the
/// 106-archive defect would have self-certified: the corrupting tool would
/// have cheerfully fingerprinted its own scrambled output and declared it
/// valid. Acceptance for such a transform is
/// `stt-validate <after> --expect-fingerprint <captured-before.json>`.
///
/// # Determinism
///
/// Every field is an order-independent fold (`min`/`max`, a capped
/// `BTreeSet`, a max-merged HLL register array) over the feature set, with
/// non-finite values skipped so no NaN can poison a comparison chain. There is
/// no RNG, no clock, and no hash-map iteration order — `BTreeMap` throughout —
/// so two builds of one input emit byte-identical JSON, and the validator's
/// recomputation is independent of tile visit order, thread count and rayon
/// emission order.
///
/// # Floats on the wire
///
/// The f64 fields are serialised by `serde_json` (shortest round-trip / ryu),
/// **not** rounded or quantized: the values are exact IEEE-754 min/max of
/// values that were themselves exact, so re-parsing recovers the same bits.
/// Nothing here is hashed, so there is no digest for a formatting change to
/// perturb — the block *is* the fingerprint.
///
/// # Out of scope for v1
///
/// Per-vertex value quantization (`TILE_META.vq`) is explicitly NOT covered:
/// `vertex_value` / `vertex_value_matrix` contribute no fingerprint statistic,
/// so their quantization needs no tolerance. Only `coord-quant` (geometry) and
/// `attr-quant` (numeric property columns) move values a fingerprint compares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentFingerprint {
    /// Block schema version — [`CONTENT_FINGERPRINT_VERSION`].
    pub version: u32,
    /// Decoded-**vertex** bbox `[min_lon, min_lat, max_lon, max_lat]`.
    ///
    /// Unlike [`Metadata::bounds`] this deliberately includes null-island
    /// `(0, 0)` features: `bounds` is a presentation quantity (the opening
    /// camera must not zoom to the whole globe over one coerced row), whereas
    /// this is a **containment claim** — excluding data the archive will later
    /// be asked to contain would make the claim false.
    pub bbox: [f64; 4],
    /// Vertical extent `[min_z, max_z]` when any vertex (or a declared
    /// elevation column) carried altitude. Pairs with [`Metadata::z_range`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_range: Option<[f64; 2]>,
    /// Count of DISTINCT source features — the same quantity as
    /// [`Metadata::distinct_feature_count`], repeated here so the fingerprint
    /// is self-contained when carried in a sidecar for transform acceptance.
    pub distinct_feature_count: u64,
    /// Per property column, the non-null `[min, max]` of its numeric values.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub numeric_ranges: BTreeMap<String, [f64; 2]>,
    /// Per property column, its distinct-value count, capped at
    /// [`FINGERPRINT_CARDINALITY_CAP`] (a value AT the cap means "at least").
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub categorical_cardinality: BTreeMap<String, u32>,
    /// Half-open slack, in degrees, allowed on every coordinate comparison.
    ///
    /// `0.0` unless the build quantizes coordinates. **Doubly gated**, because
    /// a tolerance is a request NOT to be checked and the whole bbox comparison
    /// is only as strong as its bound:
    ///
    /// 1. *Capability* — the validator rejects a non-zero value on an archive
    ///    that does not declare `coord-quant`, so a writer cannot declare slack
    ///    for an encoding it never applied.
    /// 2. *Magnitude* — the validator rejects a value materially larger than
    ///    the archive's OWN on-wire `stt:quant` step (see
    ///    [`COORD_TOLERANCE_STEP_FACTOR`]). Presence of the capability alone
    ///    used to admit any finite value, so `coord_tolerance_deg: 90.0` on a
    ///    1 m grid passed clean and made every bbox comparison vacuous.
    ///
    /// The on-wire step is admitted as slack regardless of what is declared, so
    /// bounding this field by it costs an honest archive nothing.
    pub coord_tolerance_deg: f64,
    /// Per-column slack for `attr-quant` columns; an absent entry means exact.
    /// **Capability-gated** exactly like [`Self::coord_tolerance_deg`], against
    /// `attr-quant`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub column_tolerance: BTreeMap<String, f64>,
}

impl ContentFingerprint {
    /// An empty v1 fingerprint over an inside-out bbox — the starting point a
    /// builder folds features into.
    pub fn empty() -> Self {
        Self {
            version: CONTENT_FINGERPRINT_VERSION,
            bbox: [0.0, 0.0, 0.0, 0.0],
            z_range: None,
            distinct_feature_count: 0,
            numeric_ranges: BTreeMap::new(),
            categorical_cardinality: BTreeMap::new(),
            coord_tolerance_deg: 0.0,
            column_tolerance: BTreeMap::new(),
        }
    }
}

/// The coordinate-quantization step, in degrees, a `--quantize-coords METERS`
/// build reconstructs on — i.e. the value a writer declares as
/// [`ContentFingerprint::coord_tolerance_deg`].
///
/// The world grid is anchored at `(-180, -90)` with a uniform step of
/// `meters / M_PER_DEG_LAT` degrees on BOTH axes
/// (`arrow_tile::quantize::world_grid_affine`). Reconstruction error is at most
/// half a step; the full step is declared so a boundary rounding can never
/// false-positive. `meters <= 0` (quantization off) yields `0.0`.
///
/// The metres-per-degree constant lives in exactly one place (`quantize.rs`,
/// crate-private) and is recovered here from the public
/// [`MIN_QUANTIZE_COORDS_M`](crate::arrow_tile::MIN_QUANTIZE_COORDS_M) floor,
/// which is defined as `360 * M_PER_DEG_LAT / i32::MAX` — pinned by
/// `coord_quant_step_deg_tracks_the_world_grid`.
pub fn coord_quant_step_deg(meters: f64) -> f64 {
    // NaN included: an unusable precision declares no slack rather than a
    // NaN tolerance, which would make every comparison vacuously true.
    if !meters.is_finite() || meters <= 0.0 {
        return 0.0;
    }
    let m_per_deg_lat = crate::arrow_tile::MIN_QUANTIZE_COORDS_M * (i32::MAX as f64) / 360.0;
    meters / m_per_deg_lat
}

/// How far a declared [`ContentFingerprint::coord_tolerance_deg`] may exceed the
/// archive's OWN on-wire `stt:quant` step before [`check_fingerprint`] rejects
/// it.
///
/// # Why a bound is needed at all
///
/// The capability gate is a *presence* test: it asks whether the archive
/// declares `coord-quant`, not whether the declared slack matches the grid it
/// was quantized on. An archive quantized at 1 m (step ≈ 9e-6°) that declared
/// `coord_tolerance_deg: 90.0` therefore validated clean — and every bbox
/// comparison in the run became vacuous, which is strictly worse than no
/// fingerprint at all because the report then *certifies* the content.
///
/// # Why bounding costs an honest archive nothing
///
/// [`ObservedFingerprint::coord_step_deg`] is read off the wire, and
/// [`check_fingerprint`] admits it as slack **whatever the writer declared**.
/// So the declared number is only load-bearing in the range where it exceeds
/// the wire's own step — exactly the range where it is a request not to be
/// checked. An honest writer declares [`coord_quant_step_deg`] of the same
/// precision the encoder used, which reproduces the wire's `sx`/`sy` to within
/// a float ULP.
///
/// # Why 2× and not 1×
///
/// The declared convention is the FULL step (reconstruction error is at most
/// half a step; the full step is declared so a boundary rounding cannot
/// false-positive). `2.0` admits the one defensible variant of that convention
/// — a full step *either side* — plus float slop, while still rejecting the
/// laundering case by seven orders of magnitude.
pub const COORD_TOLERANCE_STEP_FACTOR: f64 = 2.0;

/// ULPs of slack every fingerprint float comparison carries, on top of any
/// declared or on-wire tolerance.
///
/// # This is not "a bit of fuzz", it is a measured round-trip defect
///
/// The declared side of every comparison has been through
/// `f64 → JSON text → f64`, and **that round trip is not bit-exact**:
/// `serde_json`'s exact float parser lives behind its `float_roundtrip`
/// feature, which is not enabled, so a 17-significant-digit literal comes back
/// **one ULP off**. Measured on a real build: the writer emitted
/// `-122.39399999999999` into `manifest.json` and the validator re-read
/// `-122.394` — a different double. Most f64 coordinates need 17 digits, so
/// this is not an exotic input; a plain linestring fixture over San Francisco
/// hit it on the first try, and check 12's zero-tolerance equality then failed
/// a perfectly honest archive with the *scrambled-coordinates* error message.
///
/// A validator that cries wolf on honest archives is how the fleet learns to
/// ignore it, which is the same failure the absent-fingerprint warning exists
/// to avoid.
///
/// # Why it cannot hide anything
///
/// The slack is **relative**: four ULPs at the magnitude being compared. At a
/// longitude of 122° that is ~1.1e-13 degrees — about 12 nanometres, six orders
/// of magnitude finer than [`crate::arrow_tile::MIN_QUANTIZE_COORDS_M`], the
/// finest grid the format can even represent, and fourteen orders below the
/// tens-of-degrees displacement of the recorded stride-2 scramble.
///
/// ⚠️ The ROOT fix is enabling `serde_json`'s `float_roundtrip` feature in the
/// workspace manifest, which would make the round trip exact. That is a
/// dependency-configuration change outside this module; this constant makes the
/// comparison sound in the meantime and stays harmless afterwards.
pub const MANIFEST_FLOAT_ULP_SLACK: f64 = 4.0;

/// [`MANIFEST_FLOAT_ULP_SLACK`] ULPs at the magnitude of the two values being
/// compared, for use as a floor under any declared tolerance.
///
/// Public so the sibling `metadata.bounds` containment check can apply the
/// identical floor — it reads its declared side out of the same manifest and is
/// exposed to the same round-trip defect.
///
/// Returns `0.0` for non-finite inputs: a NaN slack would make every comparison
/// vacuously true, which is the exact failure mode this whole module exists to
/// prevent.
pub fn manifest_float_slack(a: f64, b: f64) -> f64 {
    let magnitude = a.abs().max(b.abs());
    if !magnitude.is_finite() {
        return 0.0;
    }
    MANIFEST_FLOAT_ULP_SLACK * magnitude * f64::EPSILON
}

/// What a decode actually observed, recomputed by the validator. The
/// counterpart of [`ContentFingerprint`]; never serialised into an archive.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObservedFingerprint {
    /// `[min_lon, min_lat, max_lon, max_lat]`, or `None` when no finite
    /// coordinate was decoded at all.
    pub bbox: Option<[f64; 4]>,
    /// Observed `[min_z, max_z]` from 3-component coordinates.
    pub z_range: Option<[f64; 2]>,
    /// Observed non-null `[min, max]` per property column.
    pub numeric_ranges: BTreeMap<String, [f64; 2]>,
    /// Observed distinct-value count per categorical column, capped at
    /// [`FINGERPRINT_CARDINALITY_CAP`].
    pub categorical_cardinality: BTreeMap<String, u32>,
    /// Categorical columns whose distinct set hit the cap — their counts are
    /// lower bounds, so a "> declared" comparison on them is not evidence.
    pub saturated_categoricals: BTreeSet<String>,
    /// HLL estimate of the distinct `id` values decoded, with relative error
    /// [`HLL_ERROR_BOUND`].
    pub distinct_ids_estimate: u64,
    /// EXACT distinct `id` count, present only while the decode stayed at or
    /// below [`FINGERPRINT_EXACT_ID_CAP`] ids. `None` above the cap, where the
    /// sketch takes over. Prefer [`Self::distinct_ids`] over reading either
    /// field directly.
    pub distinct_ids_exact: Option<u64>,
    /// Rows that contributed a non-null, decodable `id` to the count. A layer
    /// with no `id` column (or a non-`UInt64` one — already a schema error)
    /// contributes rows but no ids, so a shortfall over such a decode is an
    /// artefact of the walk rather than evidence of loss.
    pub id_rows: u64,
    /// Rows decoded (fingerprinted layers only).
    pub rows: u64,
    /// Rows decoded per TILE ZOOM (fingerprinted layers only), booked by
    /// [`FingerprintAccumulator::ingest_at_zoom`]. Empty when every fold went
    /// through the unattributed [`FingerprintAccumulator::ingest`].
    ///
    /// This is the input to [`Self::fullest_zoom`], which is the one row-count
    /// quantity that IS comparable against a declared source-feature count —
    /// see [`check_decoded_row_floor`].
    pub rows_by_zoom: BTreeMap<u8, u64>,
    /// Rows folded in with **no** zoom attribution (via
    /// [`FingerprintAccumulator::ingest`]). Non-zero makes
    /// [`Self::fullest_zoom`] return `None`, which makes the row floor inert
    /// rather than unsound.
    pub unattributed_rows: u64,
    /// Largest `stt:quant` xy step, in degrees, seen on the wire. **Read from
    /// the archive, not declared by the writer** — it is the exact affine the
    /// reader dequantizes with, so admitting it as slack cannot launder
    /// corruption while it does let an honest quantized archive pass without
    /// the writer having to declare anything.
    pub coord_step_deg: f64,
    /// Largest `stt:quant` z step (metres) seen on the wire; same rationale.
    pub z_step: f64,
    /// Largest `stt:qa` step seen on the wire, per property column; same
    /// rationale. Closes the `--quantize-attrs-auto` gap, whose per-tile
    /// range-adaptive step no writer could have declared up front.
    pub column_steps: BTreeMap<String, f64>,
    /// Layer names that contributed (summary tiers are excluded by the caller).
    pub layers_seen: BTreeSet<String>,
    /// Layers whose geometry could not be walked (unknown nesting, or Int32
    /// leaves with no affine). Their coordinates contributed nothing, so
    /// containment over them proves nothing — surfaced rather than swallowed.
    pub undecodable_geometry_layers: BTreeSet<String>,
}

impl ObservedFingerprint {
    /// The distinct feature-id count to compare a declaration against: the
    /// EXACT count while the decode stayed at or below
    /// [`FINGERPRINT_EXACT_ID_CAP`], the sketch estimate above it.
    pub fn distinct_ids(&self) -> u64 {
        self.distinct_ids_exact
            .unwrap_or(self.distinct_ids_estimate)
    }

    /// Whether [`Self::distinct_ids`] is an exact count rather than a sketch
    /// estimate — the difference between "off by up to ±2.4 %" and "off by
    /// nothing", which every distinct-count finding states.
    pub fn distinct_ids_are_exact(&self) -> bool {
        self.distinct_ids_exact.is_some()
    }

    /// The zoom that decoded the MOST rows, and how many — the quantity
    /// [`check_decoded_row_floor`] compares against a declared source-feature
    /// count.
    ///
    /// # Why per-zoom and not the total
    ///
    /// Measured on the healthy 600-feature CONUS fixture (`--min-zoom 2
    /// --max-zoom 8`): the archive decodes **4 200** rows in total, because the
    /// pyramid replicates every feature at all seven levels. A floor against
    /// the total would need >85 % of the dataset to vanish before it fired —
    /// replication masks the loss. Rows at the fullest single zoom is exactly
    /// **600**, so the floor is TIGHT: one dropped feature at the base tier
    /// takes it below the declaration.
    ///
    /// # Why the MAX over zooms is the sound pick
    ///
    /// Coarse zooms may legitimately carry fewer rows (per-feature `min_zoom`
    /// floors, LOD thinning tiers — the base tier stays lossless), so any one
    /// zoom is not a floor. The *deepest populated* tier holds the whole
    /// feature set, and taking the max finds it without having to know which
    /// tier that is.
    ///
    /// # Why `None` on any unattributed row
    ///
    /// [`FingerprintAccumulator::ingest`] books rows with no zoom, and it is
    /// the entry point every unit/property test outside this crate's validator
    /// uses. A partially-attributed observation would under-count the fullest
    /// zoom and manufacture loss findings, so it disables the check entirely.
    /// Inert, never unsound.
    ///
    /// Ties (two zooms with the same row count — the ordinary case for a fully
    /// replicated pyramid) resolve to the LOWEST zoom, purely so the reported
    /// number is deterministic; the count, which is what the check uses, is the
    /// same either way.
    pub fn fullest_zoom(&self) -> Option<(u8, u64)> {
        if self.unattributed_rows > 0 {
            return None;
        }
        self.rows_by_zoom
            .iter()
            // `max_by_key` keeps the LAST maximum; iterate high-zoom-first so
            // the retained one is the lowest zoom. Deterministic either way.
            .rev()
            .max_by_key(|(_, rows)| **rows)
            .map(|(zoom, rows)| (*zoom, *rows))
    }
}

/// Findings from [`check_fingerprint`], split by severity exactly like the
/// validator's schema findings: `errors` fail the run, `warnings` never do.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FingerprintFindings {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// Below a warning: an observation worth STATING but which is not a defect
    /// and is not actionable by the operator running the validator.
    ///
    /// The one member of this class today is the distinct-id deviation under
    /// [`DistinctIdBasis::SourceFeatures`] — a category difference between two
    /// quantities the archive never claimed were equal.
    ///
    /// # Why not a warning
    ///
    /// Warnings are the class the optimization program's O4 objective measures
    /// ("K2 noise → 0"). Filing a finding that is *expected on every archive in
    /// the fleet* as a warning would inflate exactly the number that is
    /// supposed to be driven to zero, and would train operators to skim the
    /// warning block — which is how the 106-archive defect survived. Notes are
    /// reported, never counted against that budget, and never affect the exit
    /// code.
    pub notes: Vec<String>,
}

impl FingerprintFindings {
    fn err(&mut self, msg: String) {
        self.errors.push(msg);
    }
    fn warn(&mut self, msg: String) {
        self.warnings.push(msg);
    }
    fn note(&mut self, msg: String) {
        self.notes.push(msg);
    }
}

/// Streaming recomputation of an [`ObservedFingerprint`] from decoded tiles.
///
/// Bounded memory by construction: running min/max, a capped distinct set per
/// categorical column, and a fixed 16 KiB HLL register array — never an id set
/// (the national radar archive holds 44 M points).
pub struct FingerprintAccumulator {
    skip_layers: BTreeSet<String>,
    min_lon: f64,
    min_lat: f64,
    max_lon: f64,
    max_lat: f64,
    saw_coord: bool,
    min_z: f64,
    max_z: f64,
    saw_z: bool,
    numeric: BTreeMap<String, [f64; 2]>,
    categorical: BTreeMap<String, BTreeSet<String>>,
    saturated: BTreeSet<String>,
    registers: Vec<u8>,
    /// Exact distinct ids while at or below [`FINGERPRINT_EXACT_ID_CAP`];
    /// dropped to `None` the moment the union crosses the cap, after which the
    /// sketch is the only distinct-id state kept.
    exact_ids: Option<BTreeSet<u64>>,
    id_rows: u64,
    rows: u64,
    /// Rows booked against the zoom of the tile they came from
    /// ([`Self::ingest_at_zoom`]). One `u64` per zoom level — 24 entries at the
    /// absolute worst, so this adds nothing to the bounded-memory contract.
    rows_by_zoom: BTreeMap<u8, u64>,
    /// Rows booked through [`Self::ingest`], which carries no zoom.
    unattributed_rows: u64,
    coord_step_deg: f64,
    z_step: f64,
    column_steps: BTreeMap<String, f64>,
    layers_seen: BTreeSet<String>,
    undecodable_geometry_layers: BTreeSet<String>,
}

impl Default for FingerprintAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprintAccumulator {
    pub fn new() -> Self {
        Self {
            skip_layers: BTreeSet::new(),
            min_lon: 0.0,
            min_lat: 0.0,
            max_lon: 0.0,
            max_lat: 0.0,
            saw_coord: false,
            min_z: 0.0,
            max_z: 0.0,
            saw_z: false,
            numeric: BTreeMap::new(),
            categorical: BTreeMap::new(),
            saturated: BTreeSet::new(),
            registers: vec![0u8; HLL_REGISTERS],
            exact_ids: Some(BTreeSet::new()),
            id_rows: 0,
            rows: 0,
            rows_by_zoom: BTreeMap::new(),
            unattributed_rows: 0,
            coord_step_deg: 0.0,
            z_step: 0.0,
            column_steps: BTreeMap::new(),
            layers_seen: BTreeSet::new(),
            undecodable_geometry_layers: BTreeSet::new(),
        }
    }

    /// Ignore the named layers.
    ///
    /// The one real caller is the **summary tier**: its cells are derived
    /// aggregates keyed by H3/Quadbin index, not source geometry — a cell
    /// centroid can legitimately sit a cell-radius outside the source bbox, and
    /// its `count` column is an aggregate no source feature carries. Comparing
    /// them against a source-feature fingerprint would manufacture findings.
    pub fn skipping_layers(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.skip_layers.extend(names);
        self
    }

    /// Fold one decoded tile's layers in, with **no zoom attribution**.
    ///
    /// Never panics on a malformed layer: an unrecognised geometry nesting,
    /// a missing quantization affine or an unexpected column type is recorded
    /// (or skipped) rather than unwrapped.
    ///
    /// Rows folded here are booked as unattributed, which makes
    /// [`ObservedFingerprint::fullest_zoom`] — and therefore
    /// [`check_decoded_row_floor`] — INERT for the whole observation. That is
    /// deliberate: this entry point is what the crate's own property tests and
    /// out-of-crate callers use on hand-built layers that have no tile, and a
    /// half-attributed row census would manufacture feature-loss findings.
    /// Callers that do know the tile's zoom should use [`Self::ingest_at_zoom`].
    pub fn ingest(&mut self, layers: &[DecodedLayer]) {
        self.ingest_inner(layers, None);
    }

    /// [`Self::ingest`], with the rows booked against the zoom of the tile they
    /// were decoded from.
    ///
    /// Every other statistic is folded identically — the zoom only ever adds a
    /// row census. Mixing the two entry points in one accumulator is safe and
    /// simply disables the row floor (any unattributed row does).
    pub fn ingest_at_zoom(&mut self, zoom: u8, layers: &[DecodedLayer]) {
        self.ingest_inner(layers, Some(zoom));
    }

    fn ingest_inner(&mut self, layers: &[DecodedLayer], zoom: Option<u8>) {
        for layer in layers {
            if self.skip_layers.contains(&layer.name) {
                continue;
            }
            self.layers_seen.insert(layer.name.clone());
            let rows = layer.batch.num_rows() as u64;
            self.rows += rows;
            match zoom {
                Some(z) => *self.rows_by_zoom.entry(z).or_insert(0) += rows,
                None => self.unattributed_rows += rows,
            }
            self.ingest_geometry(layer);
            self.ingest_ids(layer);
            self.ingest_properties(layer);
        }
    }

    /// Merge another accumulator (max/min folds and register-wise max), so a
    /// parallel decode composes to the identical result as a sequential one.
    pub fn merge(&mut self, other: &FingerprintAccumulator) {
        if other.saw_coord {
            self.push_coord(other.min_lon, other.min_lat);
            self.push_coord(other.max_lon, other.max_lat);
        }
        if other.saw_z {
            self.push_z(other.min_z);
            self.push_z(other.max_z);
        }
        for (name, [lo, hi]) in &other.numeric {
            self.push_numeric(name, *lo);
            self.push_numeric(name, *hi);
        }
        for (name, values) in &other.categorical {
            for v in values {
                self.push_categorical(name, v);
            }
        }
        self.saturated.extend(other.saturated.iter().cloned());
        for (i, r) in other.registers.iter().enumerate() {
            if *r > self.registers[i] {
                self.registers[i] = *r;
            }
        }
        // Set union, then the same cap. Order-independent: a partial union is a
        // subset of the total, so the result is `Some` iff the TOTAL distinct
        // count is at or below the cap, whatever order the merges happen in.
        match (self.exact_ids.as_mut(), other.exact_ids.as_ref()) {
            (Some(mine), Some(theirs)) => {
                mine.extend(theirs.iter().copied());
                if mine.len() > FINGERPRINT_EXACT_ID_CAP {
                    self.exact_ids = None;
                }
            }
            _ => self.exact_ids = None,
        }
        self.id_rows += other.id_rows;
        self.rows += other.rows;
        // Per-zoom row census: a plain sum per key, so a parallel decode that
        // splits one zoom across shards composes to the identical total as a
        // sequential one.
        for (zoom, rows) in &other.rows_by_zoom {
            *self.rows_by_zoom.entry(*zoom).or_insert(0) += *rows;
        }
        self.unattributed_rows += other.unattributed_rows;
        self.coord_step_deg = self.coord_step_deg.max(other.coord_step_deg);
        self.z_step = self.z_step.max(other.z_step);
        for (name, step) in &other.column_steps {
            let slot = self.column_steps.entry(name.clone()).or_insert(0.0);
            *slot = slot.max(*step);
        }
        self.layers_seen.extend(other.layers_seen.iter().cloned());
        self.undecodable_geometry_layers
            .extend(other.undecodable_geometry_layers.iter().cloned());
    }

    pub fn finish(self) -> ObservedFingerprint {
        ObservedFingerprint {
            bbox: self.saw_coord.then_some([
                self.min_lon,
                self.min_lat,
                self.max_lon,
                self.max_lat,
            ]),
            z_range: self.saw_z.then_some([self.min_z, self.max_z]),
            numeric_ranges: self.numeric,
            categorical_cardinality: self
                .categorical
                .iter()
                .map(|(k, v)| (k.clone(), v.len() as u32))
                .collect(),
            saturated_categoricals: self.saturated,
            distinct_ids_estimate: hll_estimate(&self.registers),
            distinct_ids_exact: self.exact_ids.as_ref().map(|ids| ids.len() as u64),
            id_rows: self.id_rows,
            rows: self.rows,
            rows_by_zoom: self.rows_by_zoom,
            unattributed_rows: self.unattributed_rows,
            coord_step_deg: self.coord_step_deg,
            z_step: self.z_step,
            column_steps: self.column_steps,
            layers_seen: self.layers_seen,
            undecodable_geometry_layers: self.undecodable_geometry_layers,
        }
    }

    fn push_coord(&mut self, lon: f64, lat: f64) {
        if !lon.is_finite() || !lat.is_finite() {
            return;
        }
        if self.saw_coord {
            self.min_lon = self.min_lon.min(lon);
            self.max_lon = self.max_lon.max(lon);
            self.min_lat = self.min_lat.min(lat);
            self.max_lat = self.max_lat.max(lat);
        } else {
            self.min_lon = lon;
            self.max_lon = lon;
            self.min_lat = lat;
            self.max_lat = lat;
            self.saw_coord = true;
        }
    }

    fn push_z(&mut self, z: f64) {
        if !z.is_finite() {
            return;
        }
        if self.saw_z {
            self.min_z = self.min_z.min(z);
            self.max_z = self.max_z.max(z);
        } else {
            self.min_z = z;
            self.max_z = z;
            self.saw_z = true;
        }
    }

    fn push_numeric(&mut self, name: &str, value: f64) {
        if !value.is_finite() {
            return;
        }
        match self.numeric.get_mut(name) {
            Some(slot) => {
                slot[0] = slot[0].min(value);
                slot[1] = slot[1].max(value);
            }
            None => {
                self.numeric.insert(name.to_string(), [value, value]);
            }
        }
    }

    fn push_categorical(&mut self, name: &str, value: &str) {
        let set = self.categorical.entry(name.to_string()).or_default();
        if set.len() >= FINGERPRINT_CARDINALITY_CAP as usize {
            if !set.contains(value) {
                self.saturated.insert(name.to_string());
            }
            return;
        }
        set.insert(value.to_string());
    }

    fn ingest_ids(&mut self, layer: &DecodedLayer) {
        use arrow::array::{Array, UInt64Array};
        let Some(column) = layer.batch.column_by_name("id") else {
            return;
        };
        let Some(ids) = column.as_any().downcast_ref::<UInt64Array>() else {
            return;
        };
        for i in 0..ids.len() {
            if ids.is_null(i) {
                continue;
            }
            let id = ids.value(i);
            self.id_rows += 1;
            hll_add(&mut self.registers, id);
            if let Some(exact) = self.exact_ids.as_mut() {
                exact.insert(id);
                if exact.len() > FINGERPRINT_EXACT_ID_CAP {
                    // Past the cap the sketch is the only distinct-id state we
                    // keep, and its relative bound is sound at that size.
                    self.exact_ids = None;
                }
            }
        }
    }

    fn ingest_geometry(&mut self, layer: &DecodedLayer) {
        let schema = layer.batch.schema();
        let Ok(index) = schema.index_of("geometry") else {
            return;
        };
        let field = schema.field(index);
        let array = layer.batch.column(index);
        let affine = field
            .metadata()
            .get(crate::arrow_tile::STT_QUANT_META_KEY)
            .and_then(|json| crate::arrow_tile::QuantAffine::from_json(json));
        if let Some(q) = &affine {
            self.coord_step_deg = self.coord_step_deg.max(q.sx.abs()).max(q.sy.abs());
            if let Some(sz) = q.sz {
                self.z_step = self.z_step.max(sz.abs());
            }
        }
        let mut coords: Vec<(f64, f64, Option<f64>)> = Vec::new();
        if !collect_leaf_coords(array, affine.as_ref(), &mut coords) {
            self.undecodable_geometry_layers.insert(layer.name.clone());
            return;
        }
        for (x, y, z) in coords {
            self.push_coord(x, y);
            if let Some(z) = z {
                self.push_z(z);
            }
        }
    }

    fn ingest_properties(&mut self, layer: &DecodedLayer) {
        use arrow::array::{
            Array, DictionaryArray, Float64Array, Int32Array, LargeStringArray, StringArray,
            UInt16Array,
        };
        use arrow::datatypes::{DataType, UInt16Type};

        let schema = layer.batch.schema();
        for (index, field) in schema.fields().iter().enumerate() {
            let name = field.name().as_str();
            if RESERVED_TILE_COLUMNS.contains(&name) {
                continue;
            }
            let column = layer.batch.column(index);
            let attr_quant = field
                .metadata()
                .get(crate::arrow_tile::STT_QUANT_ATTR_META_KEY)
                .and_then(|json| crate::arrow_tile::AttrQuant::from_json(json));
            if let Some(q) = &attr_quant {
                let slot = self.column_steps.entry(name.to_string()).or_insert(0.0);
                *slot = slot.max(q.s.abs());
            }
            match field.data_type() {
                DataType::Float64 => {
                    let Some(values) = column.as_any().downcast_ref::<Float64Array>() else {
                        continue;
                    };
                    for i in 0..values.len() {
                        if !values.is_null(i) {
                            self.push_numeric(name, values.value(i));
                        }
                    }
                }
                DataType::UInt16 => {
                    let Some(values) = column.as_any().downcast_ref::<UInt16Array>() else {
                        continue;
                    };
                    // A UInt16 property column is only meaningful with its
                    // `stt:qa` affine; without one the schema check already
                    // errors, and guessing here would invent a range.
                    let Some(q) = &attr_quant else { continue };
                    for i in 0..values.len() {
                        if !values.is_null(i) {
                            self.push_numeric(name, q.value(i64::from(values.value(i))));
                        }
                    }
                }
                DataType::Int32 => {
                    let Some(values) = column.as_any().downcast_ref::<Int32Array>() else {
                        continue;
                    };
                    let Some(q) = &attr_quant else { continue };
                    for i in 0..values.len() {
                        if !values.is_null(i) {
                            self.push_numeric(name, q.value(i64::from(values.value(i))));
                        }
                    }
                }
                DataType::Utf8 => {
                    let Some(values) = column.as_any().downcast_ref::<StringArray>() else {
                        continue;
                    };
                    for i in 0..values.len() {
                        if !values.is_null(i) {
                            self.push_categorical(name, values.value(i));
                        }
                    }
                }
                DataType::LargeUtf8 => {
                    let Some(values) = column.as_any().downcast_ref::<LargeStringArray>() else {
                        continue;
                    };
                    for i in 0..values.len() {
                        if !values.is_null(i) {
                            self.push_categorical(name, values.value(i));
                        }
                    }
                }
                DataType::Dictionary(_, _) => {
                    let Some(dict) = column
                        .as_any()
                        .downcast_ref::<DictionaryArray<UInt16Type>>()
                    else {
                        continue;
                    };
                    let Some(values) = dict.values().as_any().downcast_ref::<StringArray>() else {
                        continue;
                    };
                    // Walk the KEYS, not the dictionary: a dictionary can carry
                    // entries no row references (concat, or a producer that
                    // reuses one dictionary across tiles), and counting those
                    // would over-state cardinality.
                    let keys = dict.keys();
                    for i in 0..keys.len() {
                        if keys.is_null(i) {
                            continue;
                        }
                        let k = keys.value(i) as usize;
                        if k < values.len() && !values.is_null(k) {
                            self.push_categorical(name, values.value(k));
                        }
                    }
                }
                // Vector-group columns (FixedSizeList) and anything else carry
                // no v1 fingerprint statistic.
                _ => {}
            }
        }
    }
}

/// Peel a decoded GeoArrow `geometry` column down to its coordinate leaf and
/// push every coordinate into `out`. Returns `false` when the nesting or leaf
/// type is not one this build understands (never panics).
///
/// Only the leaf `FixedSizeList` is walked: the list offsets above it merely
/// group coordinates into features, and grouping is irrelevant to a min/max
/// fold. A decoded layer's leaf child array is exactly the coordinates its
/// offsets reference, so this is the same coordinate set the offsets walk would
/// visit — and in any pathological case it is a SUPERSET, which is the safe
/// direction for a containment claim about the archive's own bytes.
fn collect_leaf_coords(
    array: &arrow::array::ArrayRef,
    affine: Option<&crate::arrow_tile::QuantAffine>,
    out: &mut Vec<(f64, f64, Option<f64>)>,
) -> bool {
    use arrow::array::{Array, FixedSizeListArray, Float64Array, Int32Array, ListArray};
    use arrow::datatypes::DataType;

    // Descend at most two List levels: Point is FixedSizeList, LineString is
    // List<FixedSizeList>, Polygon is List<List<FixedSizeList>>.
    let mut current = array.clone();
    for _ in 0..3 {
        let next = match current.data_type() {
            DataType::FixedSizeList(_, _) => break,
            DataType::List(_) => {
                let Some(list) = current.as_any().downcast_ref::<ListArray>() else {
                    return false;
                };
                // `values()` returns the child unsliced; that is what we want.
                list.values().clone()
            }
            _ => return false,
        };
        current = next;
    }
    let Some(leaf) = current.as_any().downcast_ref::<FixedSizeListArray>() else {
        return false;
    };
    let width = leaf.value_length() as usize;
    if width != 2 && width != 3 {
        return false;
    }
    let values = leaf.values();
    match values.data_type() {
        DataType::Float64 => {
            let Some(v) = values.as_any().downcast_ref::<Float64Array>() else {
                return false;
            };
            let n = v.len() / width;
            out.reserve(n);
            for i in 0..n {
                let base = i * width;
                out.push((
                    v.value(base),
                    v.value(base + 1),
                    (width == 3).then(|| v.value(base + 2)),
                ));
            }
            true
        }
        DataType::Int32 => {
            // Quantized grid indices are meaningless without their affine;
            // refuse rather than fold raw indices in as if they were degrees
            // (that would be the very confusion this check exists to catch).
            let Some(q) = affine else { return false };
            let Some(v) = values.as_any().downcast_ref::<Int32Array>() else {
                return false;
            };
            let n = v.len() / width;
            out.reserve(n);
            let z0 = q.z0.unwrap_or(0.0);
            let sz = q.sz.unwrap_or(1.0);
            for i in 0..n {
                let base = i * width;
                out.push((
                    q.lon(v.value(base)),
                    q.lat(v.value(base + 1)),
                    (width == 3).then(|| z0 + f64::from(v.value(base + 2)) * sz),
                ));
            }
            true
        }
        _ => false,
    }
}

/// Fold one u64 id into the HLL registers.
///
/// The id is hashed with **blake3** (already a dependency, and the archive's
/// own content-addressing hash) so the register assignment is identical on
/// every platform and every toolchain version — a `DefaultHasher` would make
/// the estimate machine-dependent.
fn hll_add(registers: &mut [u8], id: u64) {
    let digest = blake3::hash(&id.to_le_bytes());
    let bytes = digest.as_bytes();
    let x = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let index = (x >> (64 - HLL_PRECISION)) as usize;
    // Shift the index bits out and set a terminator bit so `leading_zeros`
    // is bounded by the remaining width.
    let w = (x << HLL_PRECISION) | (1u64 << (HLL_PRECISION - 1));
    let rank = (w.leading_zeros() + 1) as u8;
    if rank > registers[index] {
        registers[index] = rank;
    }
}

/// Classic HyperLogLog estimator with the small-range linear-counting
/// correction. A pure function of the register array — which is itself
/// order-independent (register-wise max) — so the estimate does not depend on
/// tile visit order or thread count.
fn hll_estimate(registers: &[u8]) -> u64 {
    let m = registers.len() as f64;
    let alpha = 0.7213 / (1.0 + 1.079 / m);
    let mut sum = 0.0f64;
    let mut zeros = 0usize;
    for &r in registers {
        sum += (-(f64::from(r))).exp2();
        if r == 0 {
            zeros += 1;
        }
    }
    let raw = alpha * m * m / sum;
    if raw <= 2.5 * m && zeros > 0 {
        (m * (m / zeros as f64).ln()).round() as u64
    } else {
        raw.round() as u64
    }
}

/// Floating-point noise floor under a coordinate comparison, in degrees.
///
/// **Not** slack for imprecision the format allows — that is what the wire's
/// `stt:quant` step and the declared [`ContentFingerprint::coord_tolerance_deg`]
/// are for, and both are gated above. This is purely IEEE-754 noise: tile
/// clipping synthesises vertices by interpolating along original segments, and
/// an interpolated endpoint can land an ULP or two past the segment's own
/// extreme (a real LineString build decodes `-122.39399999999999` where the
/// source read `-122.394`). One f64 ULP near 180° is ≈2.8e-14 deg, so 1e-9 deg
/// — about 0.11 mm on the ground — swallows accumulated ULP noise while staying
/// many orders of magnitude below the smallest bbox error anyone could care
/// about, and fourteen orders below the tens-of-degrees displacement of the
/// recorded stride-2 scramble.
///
/// Public because the two bbox-containment checks that face the antimeridian —
/// check 12 ([`check_fingerprint_with`], against the fingerprint's declared
/// bbox) and check 13 (the validator's `metadata.bounds` check) — must apply the
/// *identical* floor when asking whether an edge sits ON the ±180° seam.
/// Two floors would mean two answers to one question.
pub const COORD_NOISE_FLOOR_DEG: f64 = 1e-9;

/// Is the *entire* bbox-containment escape explained by the builder's
/// ANTIMERIDIAN SPLIT?
///
/// # The behaviour this exists for (measured, not assumed)
///
/// `stt-build` **splits** a ±180°-crossing ring rather than dropping it
/// (`stt_build::clip::split_polygon_at_antimeridian`, campaign A1), and the
/// split **synthesises seam vertices at exactly ±180°** that no source vertex
/// carried. Both declared boxes it is checked against, meanwhile, are plain
/// unwrapped min/max folds over the SOURCE vertices, taken before tiling:
/// [`ContentFingerprint::bbox`] (`stt_build::input::content_fingerprint`) and
/// `metadata.bounds` (`stt_build::input::profile_features_with`, whose
/// antimeridian paragraph documents that fold as deliberate and unchanged).
/// A polygon whose extreme source vertices sit at 178°E and 178°W therefore
/// declares `[-178, 178]` and decodes to `[-180, 180]`: a real containment
/// failure produced by *correct* writer behaviour, on every seam-crossing
/// non-point dataset. Pinned builder-side by
/// `seam_split_synthesises_pm180_vertices_the_declared_source_bbox_cannot_contain`
/// (`crates/stt-build/tests/antimeridian_polygon.rs`).
///
/// Without this narrowing, an honest dateline-crossing polygon archive built
/// `--content-fingerprint` fails check 12 with the *corruption* message — while
/// check 13, sitting beside it, calmly explains that it is not corruption. The
/// operator gets a red build and a warning that contradicts it. It is a
/// **warning**, not a pass: the escape is still printed, still names the seam,
/// and the fix (widen the declared longitude interval to the full `[-180, 180]`
/// when the source straddles) still lands at the writer.
///
/// # Why it cannot launder real corruption
///
/// Three conditions, **all** required, and every one of them is a property of
/// the seam specifically rather than of "the bbox is wrong somehow":
///
/// 1. **No latitude escape.** A meridian cannot move latitude, so the latitude
///    axis keeps full severity. Any `lat` escape ⇒ no exemption at all. This is
///    what keeps the recorded stride-2 `xyz` fold caught *on a seam-crossing
///    archive*: reading `[x0,y0,z0,x1,y1,z1,…]` as 2-strided pairs puts source
///    LONGITUDES (±178) into the latitude slot, so latitude escapes by ~160°.
/// 2. **Each escaping longitude edge lands ON the seam**, to within the same
///    slack the containment test itself uses (the wire's `stt:quant` step, any
///    capability-gated declared tolerance, and [`COORD_NOISE_FLOOR_DEG`]). An
///    escape to −179, or past −180, is not the splitter's signature and stays an
///    error. This is the same shape as the null-island rule, which tests for
///    *exactly* `0.0` rather than "escapes origin-ward".
/// 3. **The declared interval is already wider than 180°.** A dataset that
///    genuinely straddles ±180 has source vertices on both sides of it, so the
///    unwrapped min/max fold yields a nearly-full-width box by construction.
///    A compact regional archive whose tiles decoded to the world edges — the
///    scrambled-coordinate class these checks exist to catch — declares a narrow
///    box and is **still an error**.
///
/// The residual: a genuinely global archive (declared width already ≈360°)
/// whose coordinates were scrambled outward to the world edges, with latitude
/// left intact, is demoted to a warning on the bbox edges. That case costs
/// little, because a full-width declared box contains essentially any scramble
/// anyway — and nothing else is demoted with it: the per-column numeric ranges,
/// categorical cardinalities, `z_range` and distinct-feature-count comparisons
/// in [`check_fingerprint_with`] are untouched by this predicate and are what
/// cover a within-bbox permutation.
#[allow(clippy::too_many_arguments)]
pub fn explained_by_antimeridian_seam(
    declared_min_lon: f64,
    declared_max_lon: f64,
    observed: [f64; 4],
    tol: f64,
    lon_lo: bool,
    lon_hi: bool,
    lat_lo: bool,
    lat_hi: bool,
) -> bool {
    // Every test below is written to fail CLOSED on a non-finite input — an
    // exemption is a licence to not-error, so `NaN` must never buy one. (Both
    // callers already reject a non-finite declaration and neither accumulator
    // folds a non-finite vertex; this is the belt for when one of those stops
    // being true.)
    let seam_distance = |edge: f64, seam: f64| -> Option<f64> {
        let d = (edge - seam).abs();
        d.is_finite().then_some(d)
    };

    // (1) latitude is never excused, and there must be something to excuse.
    if lat_lo || lat_hi || !(lon_lo || lon_hi) {
        return false;
    }
    // (3) only a straddling fold produces a wider-than-half-the-world box.
    let declared_width = declared_max_lon - declared_min_lon;
    if !declared_width.is_finite() || declared_width <= 180.0 {
        return false;
    }
    // (2) every escaping edge sits ON the seam.
    if lon_lo && seam_distance(observed[0], -180.0).is_none_or(|d| d > tol) {
        return false;
    }
    if lon_hi && seam_distance(observed[2], 180.0).is_none_or(|d| d > tol) {
        return false;
    }
    true
}

/// Compare a declared fingerprint against what a decode observed.
///
/// # Semantics by decode mode
///
/// * **Sampled / partial decode** — *containment* only: every observed vertex
///   and value must lie INSIDE the declared bbox/ranges (within tolerance) and
///   every observed cardinality must be `<=` its declared one. Containment is
///   what makes sampling sufficient for the recorded regression class: the AV
///   defect read metric `z` values as longitudes, which land tens of degrees
///   outside any real bbox, so `--sample 2` catches it.
/// * **Full decode** (`decode_complete`) — containment PLUS equality within
///   tolerance on the bbox edges and per-column min/max. A declared box wider
///   than the decoded content means the manifest describes data the archive
///   does not contain.
///
/// The distinct-feature-count comparison follows the same split and is
/// documented on [`check_distinct_feature_count`]: a SHORTFALL on a complete
/// decode is an ERROR (a rebuild dropped features), a shortfall on a sampled
/// decode and any OVERSHOOT are warnings.
///
/// # What sampling cannot catch
///
/// A defect that keeps every value inside the declared ranges — a *within-bbox
/// permutation*, say — is invisible to containment. Full-decode equality and
/// the verbatim carry-through rule are the backstops; a sampled run must never
/// be presented as a full verification (the report already refuses to).
///
/// # The one relaxation on the bbox
///
/// A bbox escape that is *entirely* the builder's ±180° ring split showing
/// through is reported as a WARNING naming the seam rather than as the
/// corruption error — see [`explained_by_antimeridian_seam`] for the three
/// conditions, which between them keep the recorded stride-2 scramble an error
/// even on a seam-crossing archive. It is the same predicate, on the same
/// terms, that the sibling `metadata.bounds` check (check 13) applies; before
/// it was shared, an honest dateline-crossing polygon failed one check while
/// the other calmly explained why it should not.
///
/// `capabilities` is the archive's `manifest.capabilities`: a declared
/// tolerance without its capability is an ERROR, so a writer cannot buy slack
/// it did not earn — and a declared tolerance materially wider than the
/// archive's own on-wire quantization step is an ERROR too
/// ([`COORD_TOLERANCE_STEP_FACTOR`]), so a writer that *does* hold the
/// capability still cannot inflate its way out of being checked.
pub fn check_fingerprint(
    declared: &ContentFingerprint,
    observed: &ObservedFingerprint,
    decode_complete: bool,
    capabilities: &[String],
) -> FingerprintFindings {
    check_fingerprint_with(
        declared,
        observed,
        decode_complete,
        capabilities,
        DistinctIdBasis::default(),
    )
}

/// [`check_fingerprint`], with the distinct-id [`DistinctIdBasis`] spelled out.
///
/// # Why the 4-argument form stays
///
/// It is the surface the crate's own integration tests
/// (`tests/adversarial_decode.rs`, `tests/spec_conformance.rs`) call, and it is
/// the conservative default — `SourceFeatures` never turns anything into an
/// error that the basis-aware form would not. Callers that can resolve the
/// archive's attestation ([`distinct_id_basis`]) should use this function; the
/// validator does.
///
/// Note the deliberate absence of a new [`ContentFingerprint`] field: the basis
/// is a property of the ARCHIVE (its capabilities and manifest properties), not
/// of the fingerprint block, and adding a field to that struct would break
/// every exhaustive struct literal in the test suite for no semantic gain.
pub fn check_fingerprint_with(
    declared: &ContentFingerprint,
    observed: &ObservedFingerprint,
    decode_complete: bool,
    capabilities: &[String],
    basis: DistinctIdBasis,
) -> FingerprintFindings {
    let mut findings = FingerprintFindings::default();

    if declared.version != CONTENT_FINGERPRINT_VERSION {
        findings.warn(format!(
            "declared fingerprint version {} is not the version {} this build understands — \
             the semantic comparison was SKIPPED (upgrade the toolchain to check it)",
            declared.version, CONTENT_FINGERPRINT_VERSION
        ));
        return findings;
    }

    // --- tolerance gating: capability, THEN magnitude -----------------------
    //
    // Both halves are needed. The capability test asks whether the archive is
    // entitled to any slack at all; the magnitude test asks whether the amount
    // it declared is the amount its own encoding implies. Presence alone let
    // `coord_tolerance_deg: 90.0` ride on a 1 m grid, which made every bbox
    // comparison below vacuous while the report still said "content verified".
    let has = |c: &str| capabilities.iter().any(|k| k == c);
    if !declared.coord_tolerance_deg.is_finite() || declared.coord_tolerance_deg < 0.0 {
        findings.err(format!(
            "coord_tolerance_deg {} is not a finite, non-negative number",
            declared.coord_tolerance_deg
        ));
    } else if declared.coord_tolerance_deg > 0.0 && !has(crate::pack::CAPABILITY_COORD_QUANT) {
        findings.err(format!(
            "declares coord_tolerance_deg {} but the manifest does not declare the \
             '{}' capability — a tolerance without its capability is how a writer would \
             launder corruption, so it is rejected",
            declared.coord_tolerance_deg,
            crate::pack::CAPABILITY_COORD_QUANT
        ));
    } else if declared.coord_tolerance_deg > 0.0 {
        // Capability present, so the archive really is quantized and the wire
        // carries the exact affine the reader dequantizes with. Bound the
        // declaration by it.
        let admissible = observed.coord_step_deg * COORD_TOLERANCE_STEP_FACTOR;
        if declared.coord_tolerance_deg > admissible {
            // Only an ERROR when the decode could actually account for the
            // archive's quantization. A layer whose geometry could not be
            // walked (unsupported nesting, Int32 leaves with no affine)
            // contributes no step, so its absence is ignorance, not evidence —
            // and nothing was decoded at all means there is no wire to compare
            // against.
            let wire_is_authoritative =
                observed.bbox.is_some() && observed.undecodable_geometry_layers.is_empty();
            let message = format!(
                "declares coord_tolerance_deg {} but the archive's own on-wire 'stt:quant' \
                 step is {} deg (at most {admissible} is admissible, {COORD_TOLERANCE_STEP_FACTOR}x \
                 the step). A tolerance wider than the encoding that earned it is a request \
                 NOT to be checked: it makes every bbox comparison below vacuous while the \
                 report still certifies the content. The on-wire step is admitted as slack \
                 anyway, so an honest archive loses nothing by declaring it exactly",
                declared.coord_tolerance_deg, observed.coord_step_deg
            );
            if wire_is_authoritative {
                findings.err(message);
            } else {
                findings.warn(format!(
                    "{message} — reported as a WARNING only because the decode could not \
                     account for every layer's geometry, so the observed step may understate \
                     the archive"
                ));
            }
        }
    }
    for (column, tolerance) in &declared.column_tolerance {
        if !tolerance.is_finite() || *tolerance < 0.0 {
            findings.err(format!(
                "column_tolerance['{column}'] {tolerance} is not a finite, non-negative number"
            ));
        } else if *tolerance > 0.0 && !has(crate::pack::CAPABILITY_ATTR_QUANT) {
            findings.err(format!(
                "declares column_tolerance['{column}'] {tolerance} but the manifest does not \
                 declare the '{}' capability — rejected",
                crate::pack::CAPABILITY_ATTR_QUANT
            ));
        }
    }

    for layer in &observed.undecodable_geometry_layers {
        findings.warn(format!(
            "layer '{layer}': geometry could not be walked for the fingerprint (unsupported \
             nesting, or quantized Int32 coordinates with no 'stt:quant' affine) — its \
             coordinates were NOT checked"
        ));
    }

    // --- coordinates -------------------------------------------------------
    // Slack is the wider of what the writer declared and the step the archive
    // itself carries on the wire. The declaration is both capability-gated and
    // magnitude-bounded above, so the `max` can now only pick a value the wire
    // itself justifies — it can never open the comparison up beyond
    // `COORD_TOLERANCE_STEP_FACTOR` times the real grid.
    let coord_tol = declared.coord_tolerance_deg.max(observed.coord_step_deg);
    // …floored, per comparison, by the manifest's own JSON float round-trip
    // error. The declared side came back through `serde_json` one ULP off on
    // any 17-digit literal, so an exact `>` here fails honest archives. See
    // [`MANIFEST_FLOAT_ULP_SLACK`].
    let edge_tol = |o: f64, d: f64| coord_tol.max(manifest_float_slack(o, d));
    let d = declared.bbox;
    match observed.bbox {
        None => {
            if observed.rows > 0 {
                findings.warn(
                    "no finite coordinate was decoded, so the fingerprint's bbox could not be \
                     checked"
                        .to_string(),
                );
            }
        }
        Some(o) => {
            // Per EDGE, not as one boolean: the seam exemption below has to know
            // *which* edge escaped, because the ring splitter can only ever pull
            // an edge to exactly ±180 on the LONGITUDE axis.
            let lon_lo = o[0] < d[0] - edge_tol(o[0], d[0]);
            let lat_lo = o[1] < d[1] - edge_tol(o[1], d[1]);
            let lon_hi = o[2] > d[2] + edge_tol(o[2], d[2]);
            let lat_hi = o[3] > d[3] + edge_tol(o[3], d[3]);
            let escapes = lon_lo || lat_lo || lon_hi || lat_hi;
            // Is the WHOLE escape the builder's antimeridian split showing
            // through? `declared.bbox` is a min/max fold over SOURCE vertices
            // taken before tiling, and the split synthesises ±180° vertices the
            // source never carried — so a seam-crossing ring makes an honest
            // writer contradict itself. Deliberate writer behaviour must not be
            // reported as the corruption class, but it is still said out loud,
            // and only under the three conditions the predicate documents (no
            // latitude escape; every escaping edge ON the seam; a declared
            // interval already wider than half the world).
            //
            // The seam distance is measured against the SEAM, so its slack is
            // the containment slack evaluated at ±180 rather than at the
            // declared edge, floored by the shared clip-interpolation noise
            // floor. Nothing here widens the containment test itself.
            let seam_tol = edge_tol(o[0], -180.0)
                .max(edge_tol(o[2], 180.0))
                .max(COORD_NOISE_FLOOR_DEG);
            let explained_by_seam = escapes
                && explained_by_antimeridian_seam(
                    d[0], d[2], o, seam_tol, lon_lo, lon_hi, lat_lo, lat_hi,
                );
            if explained_by_seam {
                findings.warn(format!(
                    "decoded vertex bbox [{}, {}, {}, {}] escapes the declared bbox \
                     [{}, {}, {}, {}], but the entire escape is to the ±180° ANTIMERIDIAN SEAM \
                     on a dataset whose declared longitude interval already spans {:.3}°. \
                     stt-build SPLITS a seam-crossing ring and synthesises vertices at exactly \
                     ±180 that the source geometry never carried, while the fingerprint's bbox \
                     is a plain min/max fold over those source vertices — so the escape is the \
                     split showing through, not the stride-2 scramble. Reported, not failed; \
                     latitude, z_range and every per-column range were still enforced. Fix it at \
                     the writer by widening the declared longitude interval to the full \
                     [-180, 180] when the source straddles ±180",
                    o[0],
                    o[1],
                    o[2],
                    o[3],
                    d[0],
                    d[1],
                    d[2],
                    d[3],
                    d[2] - d[0]
                ));
            } else if escapes {
                findings.err(format!(
                    "decoded vertex bbox [{}, {}, {}, {}] escapes the declared bbox \
                     [{}, {}, {}, {}] (tolerance {coord_tol} deg). Coordinates that are \
                     structurally valid but semantically wrong — the recorded \
                     stride-2 xyz fold that flattened 106 archives — land here",
                    o[0], o[1], o[2], o[3], d[0], d[1], d[2], d[3]
                ));
            }
            // Unchanged: over-statement is only meaningful when nothing escaped
            // at all. A seam-split archive's declared box is genuinely narrower
            // than its decode on longitude, so running the equality arm here
            // would manufacture a second, contradictory finding.
            if decode_complete && !escapes {
                let short = (o[0] - d[0]).abs() > edge_tol(o[0], d[0])
                    || (o[1] - d[1]).abs() > edge_tol(o[1], d[1])
                    || (o[2] - d[2]).abs() > edge_tol(o[2], d[2])
                    || (o[3] - d[3]).abs() > edge_tol(o[3], d[3]);
                if short {
                    findings.err(format!(
                        "full decode: declared bbox [{}, {}, {}, {}] is WIDER than the decoded \
                         vertex bbox [{}, {}, {}, {}] — the manifest describes content the \
                         archive does not contain (a dropped tier, --min-features-per-tile, or \
                         a zoom clamp will do this; so will data loss)",
                        d[0], d[1], d[2], d[3], o[0], o[1], o[2], o[3]
                    ));
                }
            }
        }
    }

    // --- vertical extent ---------------------------------------------------
    // Same manifest-round-trip floor as the bbox: `z_range` is declared as f64
    // JSON and comes back up to one ULP off.
    let z_tol = observed.z_step.max(0.0);
    match (declared.z_range, observed.z_range) {
        (Some([dlo, dhi]), Some([olo, ohi])) => {
            let lo_tol = z_tol.max(manifest_float_slack(olo, dlo));
            let hi_tol = z_tol.max(manifest_float_slack(ohi, dhi));
            if olo < dlo - lo_tol || ohi > dhi + hi_tol {
                findings.err(format!(
                    "decoded z range [{olo}, {ohi}] escapes the declared z_range \
                     [{dlo}, {dhi}] (tolerance {z_tol})"
                ));
            } else if decode_complete && ((olo - dlo).abs() > lo_tol || (ohi - dhi).abs() > hi_tol)
            {
                findings.err(format!(
                    "full decode: declared z_range [{dlo}, {dhi}] is wider than the decoded \
                     z range [{olo}, {ohi}]"
                ));
            }
        }
        (None, Some([olo, ohi])) => findings.warn(format!(
            "the archive decodes 3D coordinates (z in [{olo}, {ohi}]) but the fingerprint \
             declares no z_range — rebuild with a writer that records it"
        )),
        (Some(_), None) => {
            if decode_complete {
                findings.warn(
                    "the fingerprint declares a z_range but no decoded coordinate carried a \
                     third ordinate"
                        .to_string(),
                );
            }
        }
        (None, None) => {}
    }

    // --- numeric property ranges -------------------------------------------
    for (column, [olo, ohi]) in &observed.numeric_ranges {
        let Some([dlo, dhi]) = declared.numeric_ranges.get(column) else {
            findings.warn(format!(
                "column '{column}' decodes numeric values in [{olo}, {ohi}] but the fingerprint \
                 declares no range for it"
            ));
            continue;
        };
        let tol = declared
            .column_tolerance
            .get(column)
            .copied()
            .unwrap_or(0.0)
            .max(observed.column_steps.get(column).copied().unwrap_or(0.0));
        // …and again the manifest-round-trip floor, per bound: a declared
        // numeric range is f64 JSON exactly like the bbox is.
        let lo_tol = tol.max(manifest_float_slack(*olo, *dlo));
        let hi_tol = tol.max(manifest_float_slack(*ohi, *dhi));
        if *olo < dlo - lo_tol || *ohi > dhi + hi_tol {
            findings.err(format!(
                "column '{column}' decodes values in [{olo}, {ohi}], outside its declared range \
                 [{dlo}, {dhi}] (tolerance {tol})"
            ));
        } else if decode_complete && ((olo - dlo).abs() > lo_tol || (ohi - dhi).abs() > hi_tol) {
            findings.err(format!(
                "full decode: column '{column}' declares range [{dlo}, {dhi}] but decodes only \
                 [{olo}, {ohi}]"
            ));
        }
    }
    for column in declared.numeric_ranges.keys() {
        if !observed.numeric_ranges.contains_key(column) {
            findings.warn(format!(
                "column '{column}' is in the declared fingerprint but no decoded tile carried it \
                 (an all-null column drop, a property filter or a vector-group fold does this)"
            ));
        }
    }

    // --- categorical cardinality -------------------------------------------
    for (column, observed_count) in &observed.categorical_cardinality {
        let Some(declared_count) = declared.categorical_cardinality.get(column) else {
            continue;
        };
        // A saturated observation is a lower bound, so "more than declared" is
        // not evidence unless the declared count is itself below the cap.
        let saturated = observed.saturated_categoricals.contains(column);
        if *observed_count > *declared_count
            && !(saturated && *declared_count >= FINGERPRINT_CARDINALITY_CAP)
        {
            findings.err(format!(
                "column '{column}' decodes {observed_count} distinct values, more than the \
                 {declared_count} the fingerprint declares — attribute content changed"
            ));
        }
    }

    check_distinct_feature_count(declared, observed, decode_complete, basis, &mut findings);
    check_decoded_row_floor(declared, observed, decode_complete, &mut findings);

    findings
}

/// The distinct-feature-count comparison.
///
/// # ⚠️ Read [`DistinctIdBasis`] first
///
/// Under the default [`DistinctIdBasis::SourceFeatures`] — every archive in the
/// fleet — the declared count and the decoded id count are DIFFERENT QUANTITIES
/// and every finding below is a NOTE. The loss detection that used to live here
/// moved to [`check_decoded_row_floor`], which compares a quantity that really
/// is comparable. Only an archive that ATTESTS globally distinct ids reaches
/// the warning/error paths.
///
/// # The two directions are not the same finding
///
/// * **OVERSHOOT** (materially more distinct ids than declared) → WARNING.
///   `segment_feature_id` mints a fresh id per clipped segment of a source
///   feature that carries no id of its own, so a line/polygon dataset
///   legitimately decodes more distinct ids than it has source features.
///   Nothing that *drops* data adds ids, so an overshoot can never indicate
///   loss.
/// * **SHORTFALL** (materially fewer) on a **complete** decode → ERROR. Every
///   tile was read, so the observation cannot be short for a benign reason; the
///   archive does not contain the feature set its manifest declares. This is
///   the gap a demonstrated 74 % feature loss walked through with exit 0.
/// * **SHORTFALL on an INCOMPLETE decode** (`--sample`, `--skip-decode`) →
///   WARNING, worded so it cannot be read as the error above: a subset of tiles
///   carries a subset of the ids, and that is exactly what was asked for.
///
/// # The threshold is the sketch's, not a hand-picked percentage
///
/// [`HLL_ERROR_BOUND`] — three standard errors of the p=14 sketch, ≈ 2.4 % —
/// is what a deviation must clear in either direction. Below
/// [`FINGERPRINT_EXACT_ID_CAP`] the observed count is EXACT rather than
/// estimated (see that constant for why the relative band alone is unsound at
/// small n), and the same band then serves as the only slack it is sound for:
/// a handful of source features whose ids genuinely collide.
fn check_distinct_feature_count(
    declared: &ContentFingerprint,
    observed: &ObservedFingerprint,
    decode_complete: bool,
    basis: DistinctIdBasis,
    findings: &mut FingerprintFindings,
) {
    if declared.distinct_feature_count == 0 || observed.rows == 0 {
        return;
    }
    let declared_n = declared.distinct_feature_count as f64;
    let observed_ids = observed.distinct_ids();
    // Signed: positive is an overshoot, negative a shortfall.
    let deviation = (observed_ids as f64 - declared_n) / declared_n;
    let band = HLL_ERROR_BOUND * 100.0;
    let count_basis = if observed.distinct_ids_are_exact() {
        format!("counted EXACTLY, at or below the {FINGERPRINT_EXACT_ID_CAP}-id exact-count cap")
    } else {
        format!("HyperLogLog p={HLL_PRECISION} estimate")
    };

    // ------------------------------------------------------------------
    // THE BASIS GATE.
    //
    // Under `SourceFeatures` the declared count and the decoded id count are
    // different quantities, so no comparison between them is evidence of
    // anything — in either direction. One NOTE states the category difference
    // and names what would arm the real check. Nothing on this path is a
    // warning (that budget belongs to findings an operator can act on) and
    // nothing is an error.
    //
    // Silence when the numbers happen to agree is NOT an attestation and must
    // never be upgraded into one: agreement is a coincidence of a dataset whose
    // largest tile happens to hold every feature.
    // ------------------------------------------------------------------
    if basis == DistinctIdBasis::SourceFeatures {
        if observed.id_rows == 0 {
            findings.note(format!(
                "no decoded row carried a usable UInt64 `id`; the declared \
                 distinct_feature_count {} was not compared against it (the schema check owns \
                 the malformed-id-column finding, and on this archive the two counts are \
                 different quantities anyway — see below)",
                declared.distinct_feature_count
            ));
            return;
        }
        if deviation.abs() <= HLL_ERROR_BOUND {
            return;
        }
        let direction = if deviation > 0.0 { "MORE" } else { "FEWER" };
        findings.note(format!(
            "distinct-id deviation is NOT a content finding on this archive: {observed_ids} \
             distinct feature ids decoded over {} rows, {:.1}% {direction} than the declared \
             distinct_feature_count {} [{count_basis}; band ±{band:.2}%]. The two numbers are \
             DIFFERENT QUANTITIES here. `distinct_feature_count` counts SOURCE FEATURES, while \
             this archive's wire `id` is not a dataset-wide key — the writer records WHICH of \
             its constructions it used in \
             `metadata.properties.{FEATURE_ID_CONSTRUCTION_PROPERTY}`, and the two that land \
             here are '{FEATURE_ID_CONSTRUCTION_ROW_INDEX}' (an id-less point is written with \
             the PER-TILE ROW INDEX — a deliberate, measured saving, since the synthetic hash \
             id was ~40% of a point's compressed bytes — so archive-wide the distinct-id count \
             is roughly the row count of the single largest tile) and \
             '{FEATURE_ID_CONSTRUCTION_SEGMENT_HASH}' (an id-less clipped trajectory mints a \
             FRESH id per segment, so one source feature becomes many ids). Nothing is missing \
             and nothing is duplicated. The comparison arms itself on any archive whose \
             construction IS a key ('{FEATURE_ID_CONSTRUCTION_SOURCE}' or \
             '{FEATURE_ID_CONSTRUCTION_ANCHOR_HASH}' — every id-less line and polygon build), \
             and can also be forced on with TB-5's `manifest.capabilities` \
             '{CAPABILITY_GLOBAL_FEATURE_IDS}' or the \
             '{FEATURE_ID_SCOPE_PROPERTY}={FEATURE_ID_SCOPE_GLOBAL}' property. Feature LOSS on \
             this archive is caught by the decoded-row floor instead, which compares quantities \
             that are comparable",
            observed.rows,
            deviation.abs() * 100.0,
            declared.distinct_feature_count
        ));
        return;
    }

    // ------------------------------------------------------------------
    // `DecodedIds`: the writer attests one wire id per source feature, so the
    // two counts ARE the same quantity and the original severities apply.
    // ------------------------------------------------------------------

    // No decoded row carried a usable `id`, so there is no count to compare.
    // (The encoder always emits a non-nullable UInt64 `id`, so this means a
    // malformed archive — already an error from the schema check. Reporting a
    // "100 % loss" on top of it would be noise, not evidence.)
    if observed.id_rows == 0 {
        findings.warn(format!(
            "no decoded row carried a usable UInt64 `id`, so the declared \
             distinct_feature_count {} could NOT be checked (the schema check owns the \
             malformed-id-column finding)",
            declared.distinct_feature_count
        ));
        return;
    }

    if deviation > HLL_ERROR_BOUND {
        findings.warn(format!(
            "distinct-id OVERSHOOT (cannot indicate loss): {observed_ids} distinct feature ids \
             decoded, {:.1}% MORE than the declared distinct_feature_count {} [{count_basis}; \
             band ±{band:.2}%]. Clipped segments of an id-less source feature mint fresh ids, so \
             an overshoot is expected on line/polygon datasets",
            deviation * 100.0,
            declared.distinct_feature_count
        ));
        return;
    }
    if -deviation <= HLL_ERROR_BOUND {
        return;
    }

    let missing = -deviation * 100.0;
    if !decode_complete {
        findings.warn(format!(
            "distinct-id shortfall under an INCOMPLETE decode — SAMPLING NOISE, NOT feature \
             loss: {observed_ids} of the declared {} distinct feature ids were seen ({missing:.1}% \
             fewer) [{count_basis}; band ±{band:.2}%]. A --sample / --skip-decode run reads a \
             subset of the tiles and therefore a subset of the ids, so a shortfall here is \
             expected and is reported as a warning. Re-run a FULL decode to make a real shortfall \
             an error",
            declared.distinct_feature_count
        ));
        return;
    }
    if observed.id_rows < observed.rows {
        findings.warn(format!(
            "distinct-id shortfall NOT judged as loss: only {} of {} decoded rows carried a \
             usable UInt64 `id`, so the observed count of {observed_ids} against the declared {} \
             under-counts by construction [{count_basis}; band ±{band:.2}%]",
            observed.id_rows, observed.rows, declared.distinct_feature_count
        ));
        return;
    }

    findings.err(format!(
        "{FEATURE_LOSS_PREFIX} the archive decodes {observed_ids} distinct feature ids but the \
         fingerprint declares {} — {missing:.1}% of the declared features are MISSING. The wire \
         `id` on this archive IS a dataset-wide key (its \
         `metadata.properties.{FEATURE_ID_CONSTRUCTION_PROPERTY}` is \
         '{FEATURE_ID_CONSTRUCTION_SOURCE}' or '{FEATURE_ID_CONSTRUCTION_ANCHOR_HASH}', or it \
         declares the '{CAPABILITY_GLOBAL_FEATURE_IDS}' capability or the \
         '{FEATURE_ID_SCOPE_PROPERTY}={FEATURE_ID_SCOPE_GLOBAL}' property), so one source feature \
         carries ONE id into every tile and zoom it reaches and the two counts are the same \
         quantity. The decode was COMPLETE (every tile read; no --sample, no --skip-decode) over \
         {} rows, so it cannot under-count [{count_basis}; the shortfall is outside the \
         ±{band:.2}% band]. Note this is TIGHT where the decoded-row floor is loose: clipping \
         replicates a surviving line or polygon across tiles, so its extra ROWS cover for the \
         missing features while the distinct-id count does not move. A rebuild that silently \
         drops features lands here, and so does a fingerprint carried over from a larger dataset; \
         so do features the tiler could not place at all (empty or non-finite geometry) and the \
         opt-in per-tile budgets (--maximum-tile-features / --maximum-tile-bytes), which are the \
         benign causes worth checking the build log for. This is NOT the sampling warning — a \
         partial decode reports a shortfall as a warning and says so. Suppress only with \
         --allow-distinct-shortfall, and only once you know why the ids collapsed",
        declared.distinct_feature_count, observed.rows
    ));
}

/// RETAINED LOSS DETECTION — the decoded-**row** floor.
///
/// # Why this check exists
///
/// Moving the distinct-id comparison behind [`DistinctIdBasis`] would otherwise
/// have deleted the validator's only defence against a rebuild that silently
/// drops features — the gap a demonstrated 74 % feature loss once walked
/// through with exit 0. This restores it using a quantity that is genuinely
/// comparable to a source-feature count.
///
/// # The claim
///
/// Every source feature is placed in **at least one** tile at every zoom it
/// survives to, and the deepest populated tier carries the whole set (coarse
/// tiers may legitimately carry fewer — per-feature `min_zoom` floors, LOD
/// thinning tiers; the base tier stays lossless). So on a COMPLETE decode:
///
/// ```text
/// max over zooms of (decoded rows at that zoom)  >=  distinct_feature_count
/// ```
///
/// Clipping only ever pushes the left side UP (one source line crossing three
/// tiles contributes three rows), which is why this is a floor and not an
/// equality — an overshoot is never evidence of anything.
///
/// # Why per-zoom and not the archive total (measured)
///
/// The healthy 600-feature CONUS fixture decodes **4 200** rows in total across
/// its seven pyramid levels. A floor against the total would need >85 % of the
/// dataset to vanish before firing; pyramid replication masks the loss. Rows at
/// the fullest single zoom is exactly **600**, so this floor is TIGHT.
///
/// # No new manifest field
///
/// The declared side is the `distinct_feature_count` the fingerprint already
/// carries; the observed side is a census the accumulator already had to walk.
///
/// # Exactness
///
/// Both sides are exact integer counts — no sketch, so no
/// [`HLL_ERROR_BOUND`] band. Borrowing the sketch's tolerance here would only
/// blunt the check.
///
/// # Severity
///
/// * **Complete decode** → `FEATURE_LOSS` ERROR (same prefix, so
///   `--allow-distinct-shortfall` downgrades it the same way).
/// * **`--sample` / partial decode** → the SAMPLING NOISE warning: a stride of
///   tiles carries a subset of the rows, which is exactly what was asked for.
/// * **Any unattributed row** (a caller that used
///   [`FingerprintAccumulator::ingest`] rather than
///   [`FingerprintAccumulator::ingest_at_zoom`]) → the check is skipped
///   entirely. Inert, never unsound.
pub fn check_decoded_row_floor(
    declared: &ContentFingerprint,
    observed: &ObservedFingerprint,
    decode_complete: bool,
    findings: &mut FingerprintFindings,
) {
    let declared_n = declared.distinct_feature_count;
    if declared_n == 0 || observed.rows == 0 {
        return;
    }
    // No zoom attribution ⇒ no floor. See `ObservedFingerprint::fullest_zoom`.
    let Some((zoom, rows)) = observed.fullest_zoom() else {
        return;
    };
    if rows >= declared_n {
        return;
    }

    let missing = (declared_n - rows) as f64 / declared_n as f64 * 100.0;
    if !decode_complete {
        findings.warn(format!(
            "decoded-row shortfall under an INCOMPLETE decode — SAMPLING NOISE, NOT feature \
             loss: the fullest decoded zoom (z{zoom}) carried {rows} rows against a declared \
             distinct_feature_count of {declared_n} ({missing:.1}% fewer). A --sample / \
             --skip-decode run reads a stride of the tiles and therefore a subset of the rows, \
             so a shortfall here is expected. Re-run a FULL decode to make a real shortfall an \
             error"
        ));
        return;
    }

    findings.err(format!(
        "{FEATURE_LOSS_PREFIX} the fullest decoded zoom (z{zoom}) carries only {rows} rows, but \
         the fingerprint declares {declared_n} distinct source features — {missing:.1}% of them \
         are MISSING. The decode was COMPLETE (every tile read; no --sample, no --skip-decode), \
         and every source feature must appear at least once at the deepest tier that survives \
         it, so this count cannot be short for a benign reason: clipping only ever ADDS rows. \
         Both sides are exact integer counts, not estimates. A rebuild that silently drops \
         features lands here, and so does a fingerprint carried over from a larger dataset; \
         features the tiler could not place at all (empty or non-finite geometry) land here too \
         and are the one benign cause worth checking the build log for. Suppress only with \
         --allow-distinct-shortfall, and only once you know where the rows went"
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⭐ BLOCKER 1 — [`distinct_id_basis`]'s full decision table.
    ///
    /// The whole defect was that the basis keyed on an attestation nothing
    /// emitted, so every row here that resolves through
    /// [`FEATURE_ID_CONSTRUCTION_PROPERTY`] is a case that used to be
    /// unreachable — and every row that does NOT is a case that must stay
    /// exactly as conservative as it was.
    #[test]
    fn distinct_id_basis_is_keyed_on_the_writers_construction() {
        let basis = |props: &[(&str, &str)], caps: &[&str]| {
            let mut m = Metadata::new("basis");
            for (k, v) in props {
                m.properties.insert((*k).to_string(), (*v).to_string());
            }
            let caps: Vec<String> = caps.iter().map(|c| (*c).to_string()).collect();
            distinct_id_basis(&m, &caps)
        };
        let c = FEATURE_ID_CONSTRUCTION_PROPERTY;
        let s = FEATURE_ID_SCOPE_PROPERTY;

        // (3) THE LIVE PATH. The writer's fact alone arms the comparison —
        // no operator, no capability, no attestation.
        for key in [
            FEATURE_ID_CONSTRUCTION_SOURCE,
            FEATURE_ID_CONSTRUCTION_ANCHOR_HASH,
        ] {
            assert_eq!(
                basis(&[(c, key)], &[]),
                DistinctIdBasis::DecodedIds,
                "{key} is a dataset-wide key and must arm the comparison on its own"
            );
        }
        // …and the two that are NOT keys stay conservative.
        for key in [
            FEATURE_ID_CONSTRUCTION_ROW_INDEX,
            FEATURE_ID_CONSTRUCTION_SEGMENT_HASH,
        ] {
            assert_eq!(
                basis(&[(c, key)], &[]),
                DistinctIdBasis::SourceFeatures,
                "{key} is not a key"
            );
        }
        // A construction this build cannot interpret grants nothing. A basis is
        // a licence to turn a deviation into an ERROR.
        assert_eq!(
            basis(&[(c, "dense-renumbered")], &[]),
            DistinctIdBasis::SourceFeatures
        );
        // Case-insensitive, like every other cross-crate property value.
        assert_eq!(
            basis(&[(c, "Anchor-Hash")], &[]),
            DistinctIdBasis::DecodedIds
        );

        // (1) The OPERATOR's assertion wins over the writer's fact, BOTH ways.
        // `local` is the documented rollback and a rollback that could be
        // overridden would not be one.
        assert_eq!(
            basis(
                &[(c, FEATURE_ID_CONSTRUCTION_ANCHOR_HASH), (s, "local")],
                &[]
            ),
            DistinctIdBasis::SourceFeatures,
            "--feature-id-scope local must disarm an archive the construction would arm"
        );
        assert_eq!(
            basis(
                &[(c, FEATURE_ID_CONSTRUCTION_ROW_INDEX), (s, "global")],
                &[]
            ),
            DistinctIdBasis::DecodedIds,
            "--feature-id-scope global must still assert without proof"
        );
        // An unrecognised scope spelling asserts nothing and falls through.
        assert_eq!(
            basis(
                &[(c, FEATURE_ID_CONSTRUCTION_ANCHOR_HASH), (s, "maybe")],
                &[]
            ),
            DistinctIdBasis::DecodedIds
        );

        // (2) TB-5's capability seam, still wired and still unemitted.
        assert_eq!(
            basis(&[], &[CAPABILITY_GLOBAL_FEATURE_IDS]),
            DistinctIdBasis::DecodedIds
        );
        // …and the rollback outranks even that. Deliberate: `local` is the ONE
        // way an operator turns this off, so nothing may outvote it. (Contrived
        // today — nothing emits the capability — but pinned so the precedence is
        // a decision rather than an accident when TB-5 starts emitting it.)
        assert_eq!(
            basis(&[(s, "local")], &[CAPABILITY_GLOBAL_FEATURE_IDS]),
            DistinctIdBasis::SourceFeatures
        );
        assert!(
            !crate::pack::KNOWN_CAPABILITIES.contains(&CAPABILITY_GLOBAL_FEATURE_IDS),
            "emitting this capability before KNOWN_CAPABILITIES grows would make every \
             attested archive unopenable by the deployed fleet — TB-5 owns both halves"
        );

        // The fleet's state: neither key present, nothing armed. Silence is
        // never an attestation.
        assert_eq!(basis(&[], &[]), DistinctIdBasis::SourceFeatures);
        assert_eq!(DistinctIdBasis::default(), DistinctIdBasis::SourceFeatures);
    }

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
        assert_eq!(
            decoded.properties.get("source").map(String::as_str),
            Some("unit-test")
        );
    }

    #[test]
    fn test_metadata_summary_tier_roundtrip() {
        let tier = SummaryTier {
            variant_id: crate::tile::SUMMARY_VARIANT_ID,
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
        // with its documented lowercase spelling. Manifest-level, so it belongs
        // here and not in tests/spec_conformance.rs, whose remit is the
        // per-layer Arrow schema.
        let tier = SummaryTier {
            variant_id: crate::tile::SUMMARY_VARIANT_ID,
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
        assert!(
            json.contains("quadbin"),
            "scheme serializes lowercase: {json}"
        );
        assert!(
            json.contains("sub_buckets"),
            "sub_buckets is a manifest field: {json}"
        );
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
                "variant_id": 1,
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
        assert_eq!(
            dt.layer_name, "summary",
            "absent layer_name defaults to 'summary'"
        );
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
        // Forward compat: metadata carrying a key this build does not know
        // (`raster_tier` here) must still decode — serde skips unknown keys by
        // default.
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
            suggested_time_window_ms: Some(86_400_000),
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
        assert_eq!(
            cat_keys.len(),
            2,
            "categorical carries only name+cardinality: {cat_keys:?}"
        );
        // Numeric entry: no cardinality key.
        assert!(v["style_hints"]["properties"][0]
            .get("cardinality")
            .is_none());
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
                contract: None,
                method: None,
            },
            TemporalLodLevel {
                bucket_ms: thirty_days(),
                max_zoom_level: 4,
                contract: None,
                method: None,
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

    // --- SH-2 / K11: the additive vertical extent -------------------------

    #[test]
    fn z_range_roundtrips_through_json() {
        let metadata = Metadata::new("volumetric").with_z_range(Some([-120.5, 18_000.0]));
        let bytes = metadata.to_json_bytes().unwrap();
        let decoded = Metadata::from_json_bytes(&bytes).unwrap();
        assert_eq!(decoded.z_range, Some([-120.5, 18_000.0]));
    }

    /// K11's byte-identity accept criterion, as an explicit assertion: a 2D
    /// dataset's manifest must not gain a key. This is what lets the field land
    /// additively without touching a single published archive.
    #[test]
    fn z_range_field_omitted_when_unset() {
        let metadata = Metadata::new("flat");
        assert!(metadata.z_range.is_none());
        let s = String::from_utf8(metadata.to_json_bytes().unwrap()).unwrap();
        assert!(!s.contains("z_range"), "got: {s}");
    }

    #[test]
    fn z_range_missing_field_decodes_back_compat() {
        // Every archive written before this field existed simply has no key.
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
        assert!(m.z_range.is_none());
    }

    /// A declared range must contain the observed one, so an inverted input is
    /// normalised rather than published inside-out; a non-finite one is refused
    /// outright (an omitted key beats a NaN claim).
    #[test]
    fn z_range_is_normalised_and_refuses_non_finite() {
        assert_eq!(
            Metadata::new("x")
                .with_z_range(Some([900.0, -40.0]))
                .z_range,
            Some([-40.0, 900.0])
        );
        assert_eq!(
            Metadata::new("x")
                .with_z_range(Some([f64::NAN, 3.0]))
                .z_range,
            None
        );
        assert_eq!(
            Metadata::new("x")
                .with_z_range(Some([0.0, f64::INFINITY]))
                .z_range,
            None
        );
        assert_eq!(Metadata::new("x").with_z_range(None).z_range, None);
    }

    /// TileJSON 3.0 defines `bounds` as exactly four elements; the vertical
    /// extent must NOT leak into it (the STAC Item is where 2n-element bboxes
    /// are legal).
    #[test]
    fn tilejson_bounds_stay_four_elements_with_a_z_range_declared() {
        let metadata = Metadata::new("volumetric").with_z_range(Some([0.0, 12_000.0]));
        let tj = metadata.to_tilejson(None);
        assert_eq!(tj["bounds"].as_array().unwrap().len(), 4);
        assert!(tj.get("z_range").is_none());
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
        let res = Metadata::new("bad")
            .with_temporal_bucket_ms(hour())
            .with_temporal_lod(vec![TemporalLodLevel {
                bucket_ms: hour() + 7,
                max_zoom_level: 5,
                contract: None,
                method: None,
            }]);
        assert!(res.is_err());
    }

    #[test]
    fn temporal_lod_rejects_bucket_smaller_than_or_equal_to_base() {
        let res = Metadata::new("bad")
            .with_temporal_bucket_ms(day())
            .with_temporal_lod(vec![TemporalLodLevel {
                bucket_ms: hour(),
                max_zoom_level: 5,
                contract: None,
                method: None,
            }]);
        assert!(res.is_err());

        let res = Metadata::new("bad")
            .with_temporal_bucket_ms(hour())
            .with_temporal_lod(vec![TemporalLodLevel {
                bucket_ms: hour(),
                max_zoom_level: 5,
                contract: None,
                method: None,
            }]);
        assert!(res.is_err());
    }

    #[test]
    fn temporal_lod_rejects_unsorted_levels() {
        let res = Metadata::new("bad")
            .with_temporal_bucket_ms(hour())
            .with_temporal_lod(vec![
                TemporalLodLevel {
                    bucket_ms: thirty_days(),
                    max_zoom_level: 4,
                    contract: None,
                    method: None,
                },
                TemporalLodLevel {
                    bucket_ms: day(),
                    max_zoom_level: 8,
                    contract: None,
                    method: None,
                },
            ]);
        assert!(res.is_err());
    }

    #[test]
    fn temporal_lod_for_zoom_picks_coarsest_applicable() {
        let m = Metadata::new("lod")
            .with_temporal_bucket_ms(hour())
            .with_temporal_lod(vec![
                TemporalLodLevel {
                    bucket_ms: day(),
                    max_zoom_level: 8,
                    contract: None,
                    method: None,
                },
                TemporalLodLevel {
                    bucket_ms: thirty_days(),
                    max_zoom_level: 4,
                    contract: None,
                    method: None,
                },
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
                contract: None,
                method: None,
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

    // ------------------------------------------------------------------
    // Semantic content fingerprint (SH-1)
    // ------------------------------------------------------------------

    use crate::arrow_tile::{decode_tile, encode_tile, ColumnarLayer, PropertyColumn};
    use crate::pack::{CAPABILITY_ATTR_QUANT, CAPABILITY_COORD_QUANT};

    /// The declared fingerprint the checks below are exercised against: a tiny
    /// honest archive spanning `[-1, -1] .. [1, 1]`.
    fn honest_fingerprint() -> ContentFingerprint {
        ContentFingerprint {
            version: CONTENT_FINGERPRINT_VERSION,
            bbox: [-1.0, -1.0, 1.0, 1.0],
            z_range: Some([0.0, 10.0]),
            distinct_feature_count: 3,
            numeric_ranges: BTreeMap::from([("speed".to_string(), [0.0, 30.0])]),
            categorical_cardinality: BTreeMap::from([("kind".to_string(), 2)]),
            coord_tolerance_deg: 0.0,
            column_tolerance: BTreeMap::new(),
        }
    }

    /// An observation of the same three features [`honest_fingerprint`]
    /// declares, over `bbox`. The id fields are filled in as a real decode
    /// would fill them (three rows, three ids, exactly counted) so these
    /// fixtures exercise the geometry/column checks WITHOUT tripping the
    /// distinct-count comparison — which has its own tests below.
    fn observed_at(bbox: [f64; 4]) -> ObservedFingerprint {
        ObservedFingerprint {
            bbox: Some(bbox),
            rows: 3,
            id_rows: 3,
            distinct_ids_estimate: 3,
            distinct_ids_exact: Some(3),
            ..Default::default()
        }
    }

    /// A one-layer point tile, decoded — the shape the accumulator ingests.
    fn decoded_points(
        name: &str,
        points: Vec<[f64; 2]>,
        speeds: Vec<Option<f64>>,
        kinds: Vec<Option<String>>,
    ) -> Vec<DecodedLayer> {
        let n = points.len();
        let layer = ColumnarLayer {
            polygon_parts: None,
            name: name.to_string(),
            feature_ids: (0..n as u64).map(|i| i + 1).collect(),
            start_times: vec![1_000; n],
            end_times: vec![2_000; n],
            geometry: crate::arrow_tile::GeometryColumn::Point(points),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                ("speed".to_string(), PropertyColumn::Numeric(speeds)),
                ("kind".to_string(), PropertyColumn::Categorical(kinds)),
            ],
        };
        let payload = encode_tile(std::slice::from_ref(&layer)).expect("encode");
        decode_tile(&payload).expect("decode")
    }

    #[test]
    fn content_fingerprint_roundtrips_through_json() {
        let fingerprint = honest_fingerprint();
        let metadata = Metadata::new("fp").with_content_fingerprint(fingerprint.clone());
        let bytes = metadata.to_json_bytes().unwrap();
        let decoded = Metadata::from_json_bytes(&bytes).unwrap();
        assert_eq!(decoded.content_fingerprint, Some(fingerprint));

        // Wire shape: the empty optional maps are ABSENT, not null-filled, so a
        // minimal fingerprint stays small in the manifest.
        let bare = ContentFingerprint {
            numeric_ranges: BTreeMap::new(),
            categorical_cardinality: BTreeMap::new(),
            z_range: None,
            ..honest_fingerprint()
        };
        let json = String::from_utf8(
            Metadata::new("bare")
                .with_content_fingerprint(bare)
                .to_json_bytes()
                .unwrap(),
        )
        .unwrap();
        assert!(!json.contains("numeric_ranges"), "got: {json}");
        assert!(!json.contains("categorical_cardinality"), "got: {json}");
        assert!(!json.contains("\"z_range\":null"), "got: {json}");
    }

    #[test]
    fn content_fingerprint_field_omitted_when_unset() {
        // The byte-identity criterion: an archive built WITHOUT
        // --content-fingerprint carries no `content_fingerprint` key at all, so
        // no manifest in the fleet moves until a rebuild deliberately asks.
        let s = String::from_utf8(Metadata::new("no-fp").to_json_bytes().unwrap()).unwrap();
        assert!(!s.contains("content_fingerprint"), "got: {s}");
    }

    #[test]
    fn legacy_metadata_without_fingerprint_decodes() {
        let legacy = br#"{
            "name": "legacy", "description": "", "attribution": "",
            "bounds": {"min_lon":-180.0,"min_lat":-85.0,"max_lon":180.0,"max_lat":85.0},
            "time_range": {"start":0,"end":1000},
            "min_zoom": 0, "max_zoom": 8, "tile_count": 0, "feature_count": 0,
            "layers": ["default"], "properties": {}, "temporal_bucket_ms": 3600000
        }"#;
        let m = Metadata::from_json_bytes(legacy).unwrap();
        assert!(m.content_fingerprint.is_none());
    }

    /// The world grid's metres-per-degree constant is crate-private; the
    /// tolerance helper recovers it from the public floor. Pin the recovery so a
    /// change to either side is caught here rather than as a mystery tolerance.
    #[test]
    fn coord_quant_step_deg_tracks_the_world_grid() {
        // 111 320 m is exactly one degree of latitude on the grid.
        assert!((coord_quant_step_deg(111_320.0) - 1.0).abs() < 1e-9);
        // 1 m ≈ 8.983e-6 deg.
        assert!((coord_quant_step_deg(1.0) - 1.0 / 111_320.0).abs() < 1e-15);
        // Quantization off ⇒ exact coordinates ⇒ zero slack.
        assert_eq!(coord_quant_step_deg(0.0), 0.0);
        assert_eq!(coord_quant_step_deg(-1.0), 0.0);
    }

    /// The accumulator folds real decoded layers: bbox, per-column ranges,
    /// cardinality and distinct ids.
    #[test]
    fn accumulator_folds_decoded_layers() {
        let mut acc = FingerprintAccumulator::new();
        acc.ingest(&decoded_points(
            "default",
            vec![[-1.0, -0.5], [0.5, 1.0]],
            vec![Some(2.0), Some(30.0)],
            vec![Some("a".into()), Some("b".into())],
        ));
        let observed = acc.finish();
        assert_eq!(observed.bbox, Some([-1.0, -0.5, 0.5, 1.0]));
        assert_eq!(observed.numeric_ranges.get("speed"), Some(&[2.0, 30.0]));
        assert_eq!(observed.categorical_cardinality.get("kind"), Some(&2));
        assert_eq!(observed.rows, 2);
        assert_eq!(observed.layers_seen.len(), 1);
        assert!(observed.z_range.is_none(), "2D points carry no z");
    }

    /// min/max merge across layers and across tiles, and the result does not
    /// depend on the order the tiles were visited in — the determinism
    /// invariant that lets a parallel decode compose.
    #[test]
    fn accumulator_merge_is_order_independent() {
        let west = decoded_points(
            "default",
            vec![[-1.0, -1.0]],
            vec![Some(0.0)],
            vec![Some("a".into())],
        );
        let east = decoded_points(
            "default",
            vec![[1.0, 1.0]],
            vec![Some(30.0)],
            vec![Some("b".into())],
        );

        let mut forward = FingerprintAccumulator::new();
        forward.ingest(&west);
        forward.ingest(&east);

        let mut reverse = FingerprintAccumulator::new();
        reverse.ingest(&east);
        reverse.ingest(&west);

        let mut merged = FingerprintAccumulator::new();
        let mut half = FingerprintAccumulator::new();
        half.ingest(&east);
        merged.ingest(&west);
        merged.merge(&half);

        let a = forward.finish();
        let b = reverse.finish();
        let c = merged.finish();
        assert_eq!(a, b, "ingest order must not change the observation");
        assert_eq!(a.bbox, c.bbox, "merge must equal sequential ingest");
        assert_eq!(a.numeric_ranges, c.numeric_ranges);
        assert_eq!(a.distinct_ids_estimate, c.distinct_ids_estimate);
        assert_eq!(a.distinct_ids_exact, c.distinct_ids_exact);
        assert_eq!(a.id_rows, c.id_rows);
    }

    /// Named layers are skipped wholesale — the summary tier's derived cells
    /// must not be compared against a source-feature fingerprint.
    #[test]
    fn accumulator_skips_named_layers() {
        let mut acc = FingerprintAccumulator::new().skipping_layers(["summary".to_string()]);
        acc.ingest(&decoded_points(
            "summary",
            vec![[170.0, 80.0]],
            vec![Some(999.0)],
            vec![Some("cell".into())],
        ));
        let observed = acc.finish();
        assert_eq!(observed.bbox, None, "skipped layer contributed nothing");
        assert_eq!(observed.rows, 0);
        assert!(observed.numeric_ranges.is_empty());
    }

    /// Containment is checked in EVERY mode; equality only under a full decode.
    /// This is the sampled-vs-full mode selection.
    #[test]
    fn containment_always_equality_only_on_full_decode() {
        let declared = honest_fingerprint();

        // Strictly inside the declared box: containment holds.
        let inside = observed_at([-0.5, -0.5, 0.5, 0.5]);
        let sampled = check_fingerprint(&declared, &inside, false, &[]);
        assert!(
            sampled.errors.is_empty(),
            "a sampled decode must not demand equality: {:?}",
            sampled.errors
        );

        // The SAME observation under a full decode is a manifest overstating
        // its content.
        let full = check_fingerprint(&declared, &inside, true, &[]);
        assert_eq!(full.errors.len(), 1, "errors were {:?}", full.errors);
        assert!(full.errors[0].contains("WIDER"), "got {:?}", full.errors);

        // Exactly the declared box: clean in both modes.
        let exact = observed_at([-1.0, -1.0, 1.0, 1.0]);
        assert!(check_fingerprint(&declared, &exact, true, &[])
            .errors
            .is_empty());
    }

    /// THE ACCEPTANCE CASE, at the unit level: a vertex outside the declared
    /// bbox is an ERROR in every decode mode.
    #[test]
    fn escaping_vertex_is_an_error_even_when_sampled() {
        let declared = honest_fingerprint();
        // A metric z value (37.79) read into a longitude slot — the recorded
        // stride-2 xyz fold's signature.
        let scrambled = observed_at([-1.0, -1.0, 37.79, 1.0]);
        for complete in [false, true] {
            let findings = check_fingerprint(&declared, &scrambled, complete, &[]);
            assert!(
                findings
                    .errors
                    .iter()
                    .any(|e| e.contains("escapes the declared bbox")),
                "complete={complete}: {:?}",
                findings.errors
            );
        }
    }

    /// A declared tolerance without its capability is rejected — otherwise a
    /// writer could declare a globe-wide tolerance and launder any corruption.
    #[test]
    fn tolerance_without_its_capability_is_rejected() {
        let mut declared = honest_fingerprint();
        declared.coord_tolerance_deg = 5.0;
        declared.column_tolerance.insert("speed".to_string(), 0.5);
        let observed = observed_at([-1.0, -1.0, 1.0, 1.0]);

        let ungated = check_fingerprint(&declared, &observed, false, &[]);
        assert_eq!(ungated.errors.len(), 2, "errors were {:?}", ungated.errors);
        assert!(ungated.errors.iter().any(|e| e.contains("coord-quant")));
        assert!(ungated.errors.iter().any(|e| e.contains("attr-quant")));

        // Capabilities present AND the declaration matches the archive's own
        // on-wire step: legitimate, and still clean.
        let mut earned = observed.clone();
        earned.coord_step_deg = 5.0;
        let gated = check_fingerprint(
            &declared,
            &earned,
            false,
            &[
                CAPABILITY_COORD_QUANT.to_string(),
                CAPABILITY_ATTR_QUANT.to_string(),
            ],
        );
        assert!(
            gated.errors.is_empty(),
            "a declared+capable tolerance that matches the wire is legitimate: {:?}",
            gated.errors
        );

        // A non-finite tolerance is rejected whatever the capabilities say.
        let mut poisoned = honest_fingerprint();
        poisoned.coord_tolerance_deg = f64::NAN;
        assert!(check_fingerprint(
            &poisoned,
            &observed,
            false,
            &[CAPABILITY_COORD_QUANT.to_string()]
        )
        .errors
        .iter()
        .any(|e| e.contains("finite")));
    }

    /// ⭐ The capability gate is a PRESENCE test, and presence alone used to
    /// admit any finite tolerance. An archive quantized at 1 m that declares
    /// `coord_tolerance_deg: 90.0` — capability and all — makes every bbox
    /// comparison vacuous, so the recorded stride-2 scramble rides through
    /// while the report certifies the content. Both halves are asserted: that
    /// the inflation is rejected, and that it WOULD otherwise have laundered
    /// the defect.
    #[test]
    fn inflated_coord_tolerance_is_rejected_rather_than_making_the_check_vacuous() {
        let mut declared = honest_fingerprint();
        declared.coord_tolerance_deg = 90.0;

        // The archive really is quantized — at a 1 m grid, seven orders of
        // magnitude below what it declared.
        let step = 1.0 / 111_320.0;
        let mut scrambled = observed_at([-1.0, -1.0, 37.79, 1.0]);
        scrambled.coord_step_deg = step;

        let findings = check_fingerprint(
            &declared,
            &scrambled,
            false,
            &[CAPABILITY_COORD_QUANT.to_string()],
        );
        assert!(
            findings
                .errors
                .iter()
                .any(|e| e.contains("on-wire 'stt:quant' step")),
            "the inflated declaration must be named: {:?}",
            findings.errors
        );

        // The vacuity is real: with the inflated tolerance ADMITTED, the
        // scramble that this whole check exists to catch produces no bbox
        // finding at all. That is what the rejection above is protecting.
        let laundered: Vec<&String> = findings
            .errors
            .iter()
            .filter(|e| e.contains("escapes the declared bbox"))
            .collect();
        assert!(
            laundered.is_empty(),
            "fixture check: a 90 deg tolerance is supposed to swallow this bbox \
             violation — if it does not, the test is no longer demonstrating the \
             vacuity: {laundered:?}"
        );
    }

    /// The bound is a MAGNITUDE test, not an equality test: the honest
    /// declaration (the full grid step) and the one defensible variant of it (a
    /// full step either side) both pass, and only a materially wider claim
    /// fails. Pins [`COORD_TOLERANCE_STEP_FACTOR`] as the boundary.
    #[test]
    fn declared_tolerance_is_admitted_up_to_the_step_factor() {
        let step = 1.0 / 111_320.0;
        let mut observed = observed_at([-1.0, -1.0, 1.0, 1.0]);
        observed.coord_step_deg = step;
        let caps = [CAPABILITY_COORD_QUANT.to_string()];

        let tolerance_errors = |tolerance: f64| -> Vec<String> {
            let mut declared = honest_fingerprint();
            declared.coord_tolerance_deg = tolerance;
            check_fingerprint(&declared, &observed, false, &caps)
                .errors
                .into_iter()
                .filter(|e| e.contains("on-wire 'stt:quant' step"))
                .collect()
        };

        for admitted in [step, step * COORD_TOLERANCE_STEP_FACTOR] {
            assert!(
                tolerance_errors(admitted).is_empty(),
                "tolerance {admitted} is within {COORD_TOLERANCE_STEP_FACTOR}x the {step} deg \
                 step and must be admitted"
            );
        }
        assert!(
            !tolerance_errors(step * (COORD_TOLERANCE_STEP_FACTOR + 1.0)).is_empty(),
            "a tolerance past the factor must be rejected"
        );
    }

    /// Ignorance is not evidence. When a layer's geometry could not be walked,
    /// the observed step may understate the archive, so the magnitude bound
    /// degrades to a WARNING rather than failing a run it cannot justify
    /// failing. Same rule when nothing decoded a coordinate at all.
    #[test]
    fn unaccountable_geometry_downgrades_the_tolerance_bound_to_a_warning() {
        let mut declared = honest_fingerprint();
        declared.coord_tolerance_deg = 90.0;
        let caps = [CAPABILITY_COORD_QUANT.to_string()];

        let mut blind = observed_at([-1.0, -1.0, 1.0, 1.0]);
        blind
            .undecodable_geometry_layers
            .insert("opaque".to_string());
        let findings = check_fingerprint(&declared, &blind, false, &caps);
        assert!(
            findings.errors.is_empty(),
            "an unwalkable layer must not turn the bound into a failure: {:?}",
            findings.errors
        );
        assert!(
            findings
                .warnings
                .iter()
                .any(|w| w.contains("on-wire 'stt:quant' step")),
            "...but the inflation must still be visible: {:?}",
            findings.warnings
        );

        let mut nothing = ObservedFingerprint {
            rows: 3,
            ..Default::default()
        };
        nothing.bbox = None;
        let findings = check_fingerprint(&declared, &nothing, false, &caps);
        assert!(
            findings.errors.is_empty(),
            "no decoded coordinate means no wire to bound against: {:?}",
            findings.errors
        );
    }

    /// ⭐ A MANIFEST ROUND TRIP IS NOT BIT-EXACT, and an exact comparison
    /// therefore fails honest archives.
    ///
    /// `serde_json`'s exact float parser is behind its `float_roundtrip`
    /// feature, which is not enabled, so a 17-significant-digit literal comes
    /// back one ULP off. The pair below is the MEASURED case: a linestring
    /// build wrote `-122.39399999999999` into `manifest.json`, the validator
    /// re-read `-122.394`, and check 12 reported the honest archive with the
    /// *scrambled coordinates* error. Most f64 coordinates need 17 digits, so
    /// this is the common case, not an exotic one.
    #[test]
    fn one_ulp_manifest_round_trip_is_not_a_finding() {
        // The two really are different doubles — if a toolchain ever made them
        // equal this test would silently stop testing anything.
        let written = -122.39399999999999_f64;
        let reread = -122.394_f64;
        assert_ne!(written.to_bits(), reread.to_bits());

        let mut declared = honest_fingerprint();
        declared.bbox = [-122.405, 37.79, reread, 37.801];
        let mut observed = observed_at([-122.405, 37.79, written, 37.801]);
        observed.z_range = declared.z_range;

        for complete in [false, true] {
            let findings = check_fingerprint(&declared, &observed, complete, &[]);
            assert!(
                findings.errors.is_empty(),
                "complete={complete}: a one-ULP JSON round-trip is not corruption: {:?}",
                findings.errors
            );
        }
    }

    /// …and the slack that buys that is far too small to hide anything the
    /// format can even express. Pinned against the FINEST grid the encoder
    /// admits, which is itself six orders of magnitude coarser than the slack.
    #[test]
    fn manifest_float_slack_cannot_hide_a_representable_displacement() {
        let finest_step_deg = coord_quant_step_deg(crate::arrow_tile::MIN_QUANTIZE_COORDS_M);
        assert!(
            manifest_float_slack(-122.394, -122.394) < finest_step_deg / 1.0e5,
            "the ULP floor ({}) must stay orders of magnitude below the finest \
             representable quantization step ({finest_step_deg})",
            manifest_float_slack(-122.394, -122.394)
        );

        let mut declared = honest_fingerprint();
        declared.bbox = [-122.405, 37.79, -122.394, 37.801];
        let nudged = observed_at([-122.405, 37.79, -122.394 + finest_step_deg, 37.801]);
        assert!(
            !check_fingerprint(&declared, &nudged, false, &[])
                .errors
                .is_empty(),
            "a displacement of one FINEST-grid step must still be caught"
        );
    }

    /// The wire-sourced quantization step is admitted as slack even when the
    /// writer declared none: it is the exact affine the reader dequantizes
    /// with, so an honest quantized archive passes without the writer having to
    /// declare anything — and it cannot launder the scramble class, whose
    /// displacement is tens of degrees.
    #[test]
    fn on_wire_quant_step_is_admitted_as_slack() {
        let declared = honest_fingerprint();
        let mut observed = observed_at([-1.000004, -1.0, 1.0, 1.0]);
        observed.coord_step_deg = 1.0 / 111_320.0; // a 1 m grid
        assert!(
            check_fingerprint(&declared, &observed, false, &[])
                .errors
                .is_empty(),
            "a half-step overshoot on a 1 m grid is not corruption"
        );

        // ...but the scramble is still caught with the same slack in place.
        let mut scrambled = observed_at([-1.0, -1.0, 37.79, 1.0]);
        scrambled.coord_step_deg = 1.0 / 111_320.0;
        assert!(!check_fingerprint(&declared, &scrambled, false, &[])
            .errors
            .is_empty());
    }

    /// A numeric column whose values escape its declared range fails; the
    /// column's own `stt:qa` step is admitted as slack.
    #[test]
    fn numeric_column_containment() {
        let declared = honest_fingerprint();
        let mut observed = observed_at([-1.0, -1.0, 1.0, 1.0]);
        observed
            .numeric_ranges
            .insert("speed".to_string(), [0.0, 300.0]);
        let findings = check_fingerprint(&declared, &observed, false, &[]);
        assert!(
            findings
                .errors
                .iter()
                .any(|e| e.contains("column 'speed'") && e.contains("outside its declared range")),
            "{:?}",
            findings.errors
        );

        // Within the wire step: not a finding.
        let mut nudged = observed_at([-1.0, -1.0, 1.0, 1.0]);
        nudged
            .numeric_ranges
            .insert("speed".to_string(), [0.0, 30.2]);
        nudged.column_steps.insert("speed".to_string(), 0.5);
        assert!(check_fingerprint(&declared, &nudged, false, &[])
            .errors
            .is_empty());
    }

    /// A newer fingerprint version is a warning and a SKIP, never a
    /// misinterpretation under this build's rules.
    #[test]
    fn unknown_fingerprint_version_warns_and_skips() {
        let mut declared = honest_fingerprint();
        declared.version = CONTENT_FINGERPRINT_VERSION + 1;
        let findings = check_fingerprint(
            &declared,
            &observed_at([-99.0, -99.0, 99.0, 99.0]),
            true,
            &[],
        );
        assert!(findings.errors.is_empty(), "{:?}", findings.errors);
        assert_eq!(findings.warnings.len(), 1);
        assert!(findings.warnings[0].contains("SKIPPED"));
    }

    /// An OVERSHOOT stays a WARNING in every decode mode **on an archive that
    /// attests globally distinct ids** — clipped segments of an id-less source
    /// feature legitimately mint fresh ids, and nothing that DROPS data adds
    /// ids — and it always prints the band it was judged against.
    ///
    /// (Was `distinct_count_drift_warns_with_its_error_bound`, which pinned the
    /// old contract where a shortfall warned identically. The overshoot half of
    /// that contract is unchanged; the shortfall half moved to
    /// `distinct_count_shortfall_is_an_error_only_on_a_complete_decode`. The
    /// BASIS gate is newer still — under the default `SourceFeatures` the same
    /// observation is a note, pinned by
    /// `source_features_basis_downgrades_every_distinct_id_finding_to_a_note`.)
    #[test]
    fn distinct_count_overshoot_warns_with_its_error_bound() {
        let declared = honest_fingerprint(); // declares 3
        let mut observed = observed_at([-1.0, -1.0, 1.0, 1.0]);
        observed.distinct_ids_estimate = 9;
        observed.distinct_ids_exact = Some(9);
        for complete in [false, true] {
            let findings = check_fingerprint_with(
                &declared,
                &observed,
                complete,
                &[],
                DistinctIdBasis::DecodedIds,
            );
            assert!(findings.errors.is_empty(), "{:?}", findings.errors);
            assert!(
                findings
                    .warnings
                    .iter()
                    .any(|w| w.contains("OVERSHOOT") && w.contains("band ±")),
                "complete={complete}: {:?}",
                findings.warnings
            );
        }
    }

    /// ⭐ THE BLOCKER-1 CASE at the unit level. A distinct-count SHORTFALL —
    /// the archive holding materially fewer features than the manifest declares
    /// — is an ERROR on a complete decode, a WARNING on a sampled one, and the
    /// two messages cannot be mistaken for one another.
    ///
    /// ⚠️ This is now the `DecodedIds` contract: it needs the writer's
    /// attestation, because without it the two counts are not the same
    /// quantity (see [`DistinctIdBasis`]).
    #[test]
    fn distinct_count_shortfall_is_an_error_only_on_a_complete_decode() {
        let mut declared = honest_fingerprint();
        declared.distinct_feature_count = 100;

        // 26 of 100 survived: a 74 % loss, the fraction the reviewer walked
        // through the old warning.
        let mut lossy = observed_at([-1.0, -1.0, 1.0, 1.0]);
        lossy.rows = 26;
        lossy.id_rows = 26;
        lossy.distinct_ids_estimate = 26;
        lossy.distinct_ids_exact = Some(26);

        let full =
            check_fingerprint_with(&declared, &lossy, true, &[], DistinctIdBasis::DecodedIds);
        let loss: Vec<&String> = full
            .errors
            .iter()
            .filter(|e| e.starts_with(FEATURE_LOSS_PREFIX))
            .collect();
        assert_eq!(loss.len(), 1, "errors were {:?}", full.errors);
        assert!(
            loss[0].contains("74.0%") && loss[0].contains("MISSING"),
            "the error must name the loss: {}",
            loss[0]
        );

        // The SAME observation from a sampled decode is expected, not evidence.
        let sampled =
            check_fingerprint_with(&declared, &lossy, false, &[], DistinctIdBasis::DecodedIds);
        assert!(
            sampled.errors.is_empty(),
            "a sampled decode legitimately sees fewer ids: {:?}",
            sampled.errors
        );
        let noise: Vec<&String> = sampled
            .warnings
            .iter()
            .filter(|w| w.contains("SAMPLING NOISE"))
            .collect();
        assert_eq!(noise.len(), 1, "warnings were {:?}", sampled.warnings);
        assert!(
            !noise[0].contains(FEATURE_LOSS_PREFIX),
            "the sampled wording must never read as the loss error: {}",
            noise[0]
        );

        // Inside the band: no finding at all, in either mode.
        let mut within = lossy.clone();
        within.distinct_ids_estimate = 99;
        within.distinct_ids_exact = Some(99);
        for complete in [false, true] {
            let findings = check_fingerprint_with(
                &declared,
                &within,
                complete,
                &[],
                DistinctIdBasis::DecodedIds,
            );
            assert!(findings.errors.is_empty(), "{:?}", findings.errors);
            assert!(
                !findings.warnings.iter().any(|w| w.contains("distinct-id")),
                "1 % is inside the ±{:.2}% band: {:?}",
                HLL_ERROR_BOUND * 100.0,
                findings.warnings
            );
        }
    }

    /// A shortfall over rows that carried no usable `id` is an artefact of the
    /// walk, not loss — so it must not fire the error even on a full decode.
    /// (The encoder always emits a non-nullable UInt64 `id`; a decode that sees
    /// none is a malformed archive the schema check already fails.)
    #[test]
    fn shortfall_without_decodable_ids_is_never_the_loss_error() {
        let mut declared = honest_fingerprint();
        declared.distinct_feature_count = 100;

        // No id column anywhere.
        let mut idless = observed_at([-1.0, -1.0, 1.0, 1.0]);
        idless.rows = 100;
        idless.id_rows = 0;
        idless.distinct_ids_estimate = 0;
        idless.distinct_ids_exact = Some(0);
        let findings =
            check_fingerprint_with(&declared, &idless, true, &[], DistinctIdBasis::DecodedIds);
        assert!(findings.errors.is_empty(), "{:?}", findings.errors);
        assert!(
            findings
                .warnings
                .iter()
                .any(|w| w.contains("could NOT be checked")),
            "{:?}",
            findings.warnings
        );

        // Some rows carried ids, some did not: the count under-counts by
        // construction, so the shortfall is reported but not as loss.
        let mut partial = idless.clone();
        partial.id_rows = 40;
        partial.distinct_ids_estimate = 40;
        partial.distinct_ids_exact = Some(40);
        let findings =
            check_fingerprint_with(&declared, &partial, true, &[], DistinctIdBasis::DecodedIds);
        assert!(findings.errors.is_empty(), "{:?}", findings.errors);
        assert!(
            findings
                .warnings
                .iter()
                .any(|w| w.contains("NOT judged as loss")),
            "{:?}",
            findings.warnings
        );
    }

    // -----------------------------------------------------------------------
    // BLOCKER A — the distinct-id BASIS, and the decoded-row floor that keeps
    // loss detection alive without it.
    // -----------------------------------------------------------------------

    /// ⭐ THE CATEGORY ERROR, at the unit level. Under the default
    /// `SourceFeatures` basis a distinct-id deviation — in EITHER direction,
    /// however large — is a NOTE. Not an error (it is not loss) and not a
    /// warning (warnings are the budget O4 drives to zero, and this fires on
    /// every id-less archive in the fleet).
    ///
    /// The 5-of-600 numbers are the ones the real CONUS build produced.
    #[test]
    fn source_features_basis_downgrades_every_distinct_id_finding_to_a_note() {
        let mut declared = honest_fingerprint();
        declared.distinct_feature_count = 600;

        // What a 600-feature, 7-level CONUS pyramid actually decodes: 4 200
        // rows, and 5 distinct ids (the row count of the largest single tile).
        let mut observed = observed_at([-1.0, -1.0, 1.0, 1.0]);
        observed.rows = 4_200;
        observed.id_rows = 4_200;
        observed.distinct_ids_estimate = 5;
        observed.distinct_ids_exact = Some(5);

        for complete in [false, true] {
            let findings = check_fingerprint(&declared, &observed, complete, &[]);
            assert!(
                findings.errors.is_empty(),
                "complete={complete}: a per-tile row index is not feature loss: {:?}",
                findings.errors
            );
            assert!(
                !findings.warnings.iter().any(|w| w.contains("distinct-id")),
                "complete={complete}: it must not spend the WARNING budget either: {:?}",
                findings.warnings
            );
            assert_eq!(findings.notes.len(), 1, "notes: {:?}", findings.notes);
            let note = &findings.notes[0];
            assert!(
                note.contains("DIFFERENT QUANTITIES") && note.contains("PER-TILE ROW INDEX"),
                "the note must state the category error: {note}"
            );
            assert!(
                note.contains(CAPABILITY_GLOBAL_FEATURE_IDS)
                    && note.contains(FEATURE_ID_SCOPE_PROPERTY),
                "the note must name what would arm the real check (TB-5): {note}"
            );
        }

        // An OVERSHOOT is a note on this basis too.
        let mut over = observed.clone();
        over.distinct_ids_estimate = 5_000;
        over.distinct_ids_exact = Some(5_000);
        let findings = check_fingerprint(&declared, &over, true, &[]);
        assert!(findings.errors.is_empty(), "{:?}", findings.errors);
        assert!(!findings.warnings.iter().any(|w| w.contains("distinct-id")));
        assert_eq!(findings.notes.len(), 1, "notes: {:?}", findings.notes);
    }

    /// The basis is an ATTESTATION, never an inference. Two archives whose
    /// declared and decoded counts agree exactly get the SAME (default) basis
    /// as one that does not: agreement is a property of the dataset, not a
    /// promise by the writer.
    #[test]
    fn agreement_between_the_counts_never_arms_the_strict_basis() {
        let metadata = Metadata::new("plain");
        assert_eq!(
            distinct_id_basis(&metadata, &[]),
            DistinctIdBasis::SourceFeatures
        );

        // Only the two attestations arm it.
        assert_eq!(
            distinct_id_basis(&metadata, &[CAPABILITY_GLOBAL_FEATURE_IDS.to_string()]),
            DistinctIdBasis::DecodedIds
        );
        let attested = Metadata::new("attested")
            .with_property(FEATURE_ID_SCOPE_PROPERTY, FEATURE_ID_SCOPE_GLOBAL);
        assert_eq!(
            distinct_id_basis(&attested, &[]),
            DistinctIdBasis::DecodedIds
        );

        // Explicitly local, an unrelated capability, and a near-miss value all
        // stay on the default.
        let local =
            Metadata::new("local").with_property(FEATURE_ID_SCOPE_PROPERTY, FEATURE_ID_SCOPE_LOCAL);
        assert_eq!(
            distinct_id_basis(&local, &[crate::pack::CAPABILITY_COORD_QUANT.to_string()]),
            DistinctIdBasis::SourceFeatures
        );
        let nearly = Metadata::new("nearly").with_property(FEATURE_ID_SCOPE_PROPERTY, "globalish");
        assert_eq!(
            distinct_id_basis(&nearly, &[]),
            DistinctIdBasis::SourceFeatures
        );

        // Case-insensitive, like check 13's bounds_mode attestation.
        let shouty = Metadata::new("shouty").with_property(FEATURE_ID_SCOPE_PROPERTY, "GLOBAL");
        assert_eq!(distinct_id_basis(&shouty, &[]), DistinctIdBasis::DecodedIds);
    }

    /// The 4-argument `check_fingerprint` is load-bearing surface (two
    /// integration-test files call it) and must stay the conservative default.
    #[test]
    fn check_fingerprint_defaults_to_the_source_features_basis() {
        let mut declared = honest_fingerprint();
        declared.distinct_feature_count = 600;
        let mut observed = observed_at([-1.0, -1.0, 1.0, 1.0]);
        observed.rows = 4_200;
        observed.id_rows = 4_200;
        observed.distinct_ids_estimate = 5;
        observed.distinct_ids_exact = Some(5);

        assert_eq!(
            check_fingerprint(&declared, &observed, true, &[]),
            check_fingerprint_with(
                &declared,
                &observed,
                true,
                &[],
                DistinctIdBasis::SourceFeatures
            )
        );
        assert_eq!(DistinctIdBasis::default(), DistinctIdBasis::SourceFeatures);
    }

    /// ⭐ RETAINED LOSS DETECTION. The row floor is what still catches a
    /// rebuild that drops features once the distinct-id comparison is off.
    #[test]
    fn decoded_row_floor_is_the_retained_loss_check() {
        let mut declared = honest_fingerprint();
        declared.distinct_feature_count = 600;

        // Healthy: 600 rows at each of seven pyramid levels.
        let mut healthy = observed_at([-1.0, -1.0, 1.0, 1.0]);
        healthy.rows = 4_200;
        healthy.id_rows = 4_200;
        healthy.distinct_ids_estimate = 5;
        healthy.distinct_ids_exact = Some(5);
        healthy.rows_by_zoom = (2u8..=8).map(|z| (z, 600)).collect();
        let findings = check_fingerprint(&declared, &healthy, true, &[]);
        assert!(findings.errors.is_empty(), "{:?}", findings.errors);

        // WHY PER-ZOOM: the same archive missing 100 features still decodes
        // 3 500 rows in TOTAL, which is 83 % of the declared 4 200 — a total-row
        // floor keyed to `distinct_feature_count` would not even look at it, and
        // one keyed to the total would need >85 % loss to fire. Per zoom it is
        // 500 against 600 and the floor bites.
        let mut lossy = healthy.clone();
        lossy.rows = 3_500;
        lossy.id_rows = 3_500;
        lossy.rows_by_zoom = (2u8..=8).map(|z| (z, 500)).collect();
        let full = check_fingerprint(&declared, &lossy, true, &[]);
        let loss: Vec<&String> = full
            .errors
            .iter()
            .filter(|e| e.starts_with(FEATURE_LOSS_PREFIX))
            .collect();
        assert_eq!(loss.len(), 1, "errors were {:?}", full.errors);
        assert!(
            loss[0].contains("16.7%") && loss[0].contains("MISSING"),
            "the error must name the loss: {}",
            loss[0]
        );
        // A TOTAL-row floor would have seen 3 500 >= 600 and said nothing.
        assert!(lossy.rows > declared.distinct_feature_count);

        // The same observation from a sampled decode is expected, not evidence.
        let sampled = check_fingerprint(&declared, &lossy, false, &[]);
        assert!(sampled.errors.is_empty(), "{:?}", sampled.errors);
        assert!(
            sampled
                .warnings
                .iter()
                .any(|w| w.contains("SAMPLING NOISE") && w.contains("decoded-row")),
            "{:?}",
            sampled.warnings
        );

        // One feature short of the declaration is still short — both sides are
        // exact counts, so there is no band to hide in.
        let mut off_by_one = healthy.clone();
        off_by_one.rows_by_zoom.insert(8, 599);
        assert!(
            check_fingerprint(&declared, &off_by_one, true, &[])
                .errors
                .is_empty(),
            "z8 dropped to 599 but z2..z7 still carry 600, so the MAX is still 600"
        );
        let mut all_short = healthy.clone();
        all_short.rows_by_zoom = (2u8..=8).map(|z| (z, 599)).collect();
        assert!(
            check_fingerprint(&declared, &all_short, true, &[])
                .errors
                .iter()
                .any(|e| e.starts_with(FEATURE_LOSS_PREFIX)),
            "one missing feature at EVERY zoom is one missing feature"
        );

        // Clipping only ever adds rows, so an overshoot is never a finding.
        let mut clipped = healthy.clone();
        clipped.rows_by_zoom = (2u8..=8).map(|z| (z, 1_800)).collect();
        assert!(check_fingerprint(&declared, &clipped, true, &[])
            .errors
            .is_empty());
    }

    /// The floor is INERT — never unsound — for any observation that carries an
    /// unattributed row, which is every caller that uses the plain `ingest`.
    #[test]
    fn decoded_row_floor_is_inert_without_zoom_attribution() {
        let mut declared = honest_fingerprint();
        declared.distinct_feature_count = 600;

        // Wholly unattributed (the `ingest` path).
        let mut plain = observed_at([-1.0, -1.0, 1.0, 1.0]);
        plain.rows = 26;
        plain.id_rows = 26;
        plain.unattributed_rows = 26;
        assert_eq!(plain.fullest_zoom(), None);
        assert!(check_fingerprint(&declared, &plain, true, &[])
            .errors
            .is_empty());

        // PARTIALLY attributed is the dangerous one: the attributed half would
        // look like a 99 % shortfall. It must be inert too.
        let mut mixed = plain.clone();
        mixed.rows_by_zoom = BTreeMap::from([(8u8, 4)]);
        assert_eq!(mixed.fullest_zoom(), None);
        assert!(check_fingerprint(&declared, &mixed, true, &[])
            .errors
            .is_empty());
    }

    /// `fullest_zoom` picks the MAX row count — the deepest fully populated
    /// tier — and breaks ties deterministically.
    #[test]
    fn fullest_zoom_takes_the_max_and_breaks_ties_deterministically() {
        // A coarse-thinned pyramid: only the base tier is complete.
        let thinned = ObservedFingerprint {
            rows: 40 + 120 + 600,
            rows_by_zoom: BTreeMap::from([(2u8, 40), (5u8, 120), (8u8, 600)]),
            ..Default::default()
        };
        assert_eq!(thinned.fullest_zoom(), Some((8, 600)));

        // Fully replicated: every zoom ties. The count is what the check uses;
        // the reported zoom is pinned only so the message is stable.
        let flat = ObservedFingerprint {
            rows: 1_800,
            rows_by_zoom: BTreeMap::from([(2u8, 600), (3u8, 600), (4u8, 600)]),
            ..Default::default()
        };
        assert_eq!(flat.fullest_zoom(), Some((2, 600)));

        assert_eq!(ObservedFingerprint::default().fullest_zoom(), None);
    }

    /// `ingest_at_zoom` books the row census; `ingest` books the same rows as
    /// unattributed; `merge` composes both so a parallel decode is identical to
    /// a sequential one.
    #[test]
    fn row_census_is_booked_by_zoom_and_merges_associatively() {
        let a = decoded_points(
            "default",
            vec![[0.0, 0.0], [1.0, 1.0]],
            vec![Some(1.0), Some(2.0)],
            vec![Some("car".to_string()), Some("bus".to_string())],
        );
        let b = decoded_points(
            "default",
            vec![[2.0, 2.0]],
            vec![Some(3.0)],
            vec![Some("car".to_string())],
        );

        let mut sequential = FingerprintAccumulator::new();
        sequential.ingest_at_zoom(8, &a);
        sequential.ingest_at_zoom(8, &b);
        sequential.ingest_at_zoom(2, &a);
        let sequential = sequential.finish();
        assert_eq!(
            sequential.rows_by_zoom,
            BTreeMap::from([(2u8, 2), (8u8, 3)])
        );
        assert_eq!(sequential.unattributed_rows, 0);
        assert_eq!(sequential.rows, 5);

        let mut left = FingerprintAccumulator::new();
        left.ingest_at_zoom(8, &a);
        let mut right = FingerprintAccumulator::new();
        right.ingest_at_zoom(8, &b);
        right.ingest_at_zoom(2, &a);
        left.merge(&right);
        let parallel = left.finish();
        assert_eq!(parallel.rows_by_zoom, sequential.rows_by_zoom);
        assert_eq!(parallel.rows, sequential.rows);

        // The unattributed entry point keeps working and poisons only the floor.
        let mut legacy = FingerprintAccumulator::new();
        legacy.ingest(&a);
        let legacy = legacy.finish();
        assert_eq!(legacy.rows, 2);
        assert_eq!(legacy.unattributed_rows, 2);
        assert!(legacy.rows_by_zoom.is_empty());
        assert_eq!(legacy.fullest_zoom(), None);
    }

    /// The distinct count is EXACT below the cap and falls back to the sketch
    /// above it. Without the exact count a single hash collision is a 3.3 %
    /// shortfall at n = 30 — wider than the band — so the loss error would fire
    /// on honest small archives.
    #[test]
    fn distinct_ids_are_exact_below_the_cap_and_estimated_above_it() {
        let small: Vec<[f64; 2]> = (0..30).map(|i| [i as f64 * 0.01, 0.0]).collect();
        let n = small.len();
        let mut acc = FingerprintAccumulator::new();
        acc.ingest(&decoded_points(
            "default",
            small,
            vec![None; n],
            vec![None; n],
        ));
        let observed = acc.finish();
        assert_eq!(observed.distinct_ids_exact, Some(30));
        assert_eq!(observed.distinct_ids(), 30);
        assert!(observed.distinct_ids_are_exact());
        assert_eq!(observed.id_rows, 30);
        assert_eq!(observed.id_rows, observed.rows);

        // Past the cap the set is dropped and the sketch takes over — still
        // inside its stated relative bound.
        let big: Vec<[f64; 2]> = (0..FINGERPRINT_EXACT_ID_CAP + 50)
            .map(|i| [(i % 360) as f64 * 0.001, 0.0])
            .collect();
        let n = big.len();
        let mut acc = FingerprintAccumulator::new();
        acc.ingest(&decoded_points(
            "default",
            big,
            vec![None; n],
            vec![None; n],
        ));
        let observed = acc.finish();
        assert_eq!(
            observed.distinct_ids_exact, None,
            "the cap must drop the set"
        );
        assert!(!observed.distinct_ids_are_exact());
        let error = (observed.distinct_ids() as f64 - n as f64).abs() / n as f64;
        assert!(
            error <= HLL_ERROR_BOUND,
            "estimate {} for {n} ids is off by {:.2}%",
            observed.distinct_ids(),
            error * 100.0
        );
    }

    /// The HLL is deterministic and roughly right: same ids ⇒ same estimate,
    /// order-independent, and inside its stated band.
    #[test]
    fn hll_is_deterministic_and_within_its_error_bound() {
        let n = 50_000u64;
        let mut forward = vec![0u8; HLL_REGISTERS];
        let mut backward = vec![0u8; HLL_REGISTERS];
        for i in 0..n {
            hll_add(&mut forward, i.wrapping_mul(2_654_435_761));
        }
        for i in (0..n).rev() {
            hll_add(&mut backward, i.wrapping_mul(2_654_435_761));
        }
        assert_eq!(forward, backward, "register state is order-independent");
        let estimate = hll_estimate(&forward) as f64;
        let error = (estimate - n as f64).abs() / n as f64;
        assert!(
            error <= HLL_ERROR_BOUND,
            "estimate {estimate} for {n} ids is off by {:.2}% (bound {:.2}%)",
            error * 100.0,
            HLL_ERROR_BOUND * 100.0
        );
        // Empty and tiny cardinalities land on the linear-counting branch.
        assert_eq!(hll_estimate(&vec![0u8; HLL_REGISTERS]), 0);
    }

    /// The whole point of the check, end to end at the unit level: the SAME
    /// features, encoded once honestly and once with the stride-2 xyz fold,
    /// under one honest declared fingerprint.
    #[test]
    fn stride_two_xyz_fold_escapes_the_declared_bbox() {
        // An AV-style scene: real lon/lat near San Francisco, z in metres.
        let xyz: Vec<f64> = vec![
            -122.4050, 37.7900, 0.0, //
            -122.4030, 37.7920, 12.5, //
            -122.4010, 37.7940, 30.0,
        ];
        let honest: Vec<[f64; 2]> = xyz.chunks_exact(3).map(|c| [c[0], c[1]]).collect();
        // The defect: the same flat buffer re-read with a stride of 2, which
        // both FLATTENS (z vanishes) and SCRAMBLES (metres land in lon/lat
        // slots and the trailing ordinate is dropped).
        let scrambled: Vec<[f64; 2]> = xyz.chunks_exact(2).map(|c| [c[0], c[1]]).collect();

        let declared = ContentFingerprint {
            version: CONTENT_FINGERPRINT_VERSION,
            bbox: [-122.4050, 37.7900, -122.4010, 37.7940],
            z_range: Some([0.0, 30.0]),
            distinct_feature_count: 3,
            numeric_ranges: BTreeMap::new(),
            categorical_cardinality: BTreeMap::new(),
            coord_tolerance_deg: 0.0,
            column_tolerance: BTreeMap::new(),
        };

        let mut good = FingerprintAccumulator::new();
        good.ingest(&decoded_points(
            "default",
            honest,
            vec![None, None, None],
            vec![None, None, None],
        ));
        assert!(
            check_fingerprint(&declared, &good.finish(), true, &[])
                .errors
                .is_empty(),
            "the honest encoding must pass — otherwise the negative proves nothing"
        );

        let n = scrambled.len();
        let mut bad = FingerprintAccumulator::new();
        bad.ingest(&decoded_points(
            "default",
            scrambled,
            vec![None; n],
            vec![None; n],
        ));
        let findings = check_fingerprint(&declared, &bad.finish(), true, &[]);
        assert!(
            findings
                .errors
                .iter()
                .any(|e| e.contains("escapes the declared bbox")),
            "the stride-2 fold must be caught: {:?}",
            findings.errors
        );
    }

    // ------------------------------------------------------------------
    // DT-1 — unified tier declaration
    // ------------------------------------------------------------------

    /// Absent `contract` means `union` — every pre-DT-1 manifest stays valid
    /// and round-trips without gaining a key.
    #[test]
    fn an_absent_contract_is_the_union_default_and_is_not_serialized() {
        let level = TemporalLodLevel {
            bucket_ms: 3_600_000,
            max_zoom_level: 6,
            contract: None,
            method: None,
        };
        let json = serde_json::to_string(&level).unwrap();
        assert!(!json.contains("contract"), "{json}");
        assert!(!json.contains("method"), "{json}");
        let back: TemporalLodLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(back, level);
        // And a legacy object with neither key still decodes.
        let legacy: TemporalLodLevel =
            serde_json::from_str(r#"{"bucket_ms":3600000,"max_zoom_level":6}"#).unwrap();
        assert_eq!(legacy.contract, None);
    }

    /// A `reduced` tier without a method is unreadable, so it is refused.
    #[test]
    fn a_reduced_tier_must_name_its_method() {
        let bad = vec![TemporalLodLevel {
            bucket_ms: 7_200_000,
            max_zoom_level: 4,
            contract: Some(TierContract::Reduced),
            method: None,
        }];
        let err = validate_temporal_lod(3_600_000, &bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reduced"), "{err}");
        assert!(err.contains("method"), "{err}");

        let good = vec![TemporalLodLevel {
            method: Some(ReductionMethod::MinMaxLttb),
            ..bad[0]
        }];
        validate_temporal_lod(3_600_000, &good).unwrap();
    }

    /// A method on a UNION tier is a category error: a union tier is exactly
    /// the base features re-bucketed, so nothing was reduced.
    #[test]
    fn a_union_tier_may_not_name_a_reduction_method() {
        let bad = vec![TemporalLodLevel {
            bucket_ms: 7_200_000,
            max_zoom_level: 4,
            contract: Some(TierContract::Union),
            method: Some(ReductionMethod::M4),
        }];
        let err = validate_temporal_lod(3_600_000, &bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("meaningless"), "{err}");
    }

    /// Home-zoom without its must-understand capability is refused — the
    /// silent-misdecode class this capability exists to prevent.
    #[test]
    fn home_zoom_requires_its_must_understand_capability() {
        let err = validate_partition_capability(Some(Partition::HomeZoom), &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains(CAPABILITY_ADDITIVE_PARTITION), "{err}");

        validate_partition_capability(
            Some(Partition::HomeZoom),
            &[CAPABILITY_ADDITIVE_PARTITION.to_string()],
        )
        .unwrap();
        // Replicated (and absent) need nothing.
        validate_partition_capability(Some(Partition::Replicated), &[]).unwrap();
        validate_partition_capability(None, &[]).unwrap();
    }

    /// The declared vocabulary serializes to the spelling the spec fixes.
    #[test]
    fn the_declaration_vocabulary_uses_its_spec_spelling() {
        assert_eq!(
            serde_json::to_string(&TierContract::Union).unwrap(),
            "\"union\""
        );
        assert_eq!(
            serde_json::to_string(&TierContract::Reduced).unwrap(),
            "\"reduced\""
        );
        assert_eq!(
            serde_json::to_string(&ReductionMethod::M4).unwrap(),
            "\"m4\""
        );
        assert_eq!(
            serde_json::to_string(&ReductionMethod::MinMaxLttb).unwrap(),
            "\"minmaxlttb\""
        );
        assert_eq!(
            serde_json::to_string(&Partition::HomeZoom).unwrap(),
            "\"home-zoom\""
        );
        assert_eq!(
            serde_json::to_string(&Partition::Replicated).unwrap(),
            "\"replicated\""
        );
    }
}
