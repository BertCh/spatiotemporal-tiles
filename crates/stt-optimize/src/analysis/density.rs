//! Tile density analysis
//!
//! Buckets features into (zoom, x, y, time-bucket) tiles — the same cut a
//! real `stt-build` run makes with `--temporal-bucket` — to predict tile
//! counts, per-tile feature loads, and archive size, and to flag issues.
//!
//! # Two things a size estimate here has to get right
//!
//! Both were wrong before, and both were wrong in the same direction (too
//! small), which is how a `--target-size` recipe could project 41% of a budget
//! and then build 307% of it.
//!
//! 1. **Per-tile framing is not free.** The estimate used to be
//!    `features × bytes_per_feature` — a zero-intercept line in the feature
//!    count with no tile term at all. On a real archive that is catastrophic
//!    where tiles are thin: `osm-nyc-changesets` builds **301,406 tiles for
//!    380,007 features** (1.26 features/tile) and the realized archive is
//!    105,498,977 B, i.e. **350 B per tile**, nearly all of it Arrow frame,
//!    section TOC and zstd frame overhead that a per-feature rate cannot see.
//!    [`SizeModel`] therefore carries TWO measured coefficients — bytes per
//!    tile and bytes per feature — and both come from real encodes of the
//!    run's own sample. Neither is assumed, and no closed-form size formula is
//!    fitted: the model is two measured points on the same encoder the build
//!    uses (see the standing rejection of analytic size models).
//!
//! 2. **A feature can land in more than one tile.** The centroid scan below
//!    files every feature into exactly one `(x, y, t)` cell. The builder does
//!    not: it clips trajectories and polygons, so `wpc-fronts` turns 4,156
//!    input features into **11,156 tiled instances across 7,405 tiles**, while
//!    the centroid scan sees 3,698 cells. [`sample_replication`] measures that
//!    inflation from the loader's retained geometries by walking the same tile
//!    grid the clipper does, and the per-zoom rows below are scaled by it.
//!    Measured against the two real builds this reproduces the tile count
//!    EXACTLY (7,405 and 301,406) and the instance count to within 0.04%.
//!
//! The residual risk is unchanged and is stated rather than hidden: the
//! coefficients are measured on a sample of at most 5,000 features and then
//! carried to an archive of hundreds of thousands of tiles. That extrapolation
//! is what [`SizeModel::bytes_per_tile_stderr`] quantifies — measured, from the
//! probe's own curvature — and if it ever misses badly the recorded fallback is
//! to WIDEN THE SAMPLE, never to fit a formula.

use crate::analysis::spatial::SpatialAnalysis;
use crate::analysis::temporal::TemporalAnalysis;
use crate::loader::{LoadedData, PropValue, SampledFeature};
use crate::measure::{MeasureSettings, MeasuredEncoding};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use stt_core::arrow_tile::{
    encode_tile_with, ColumnarLayer, Coord, EncoderConfig, GeometryColumn, PropertyColumn,
    TemplateCollector,
};
use stt_core::compression::compress_zstd_with_dict_level;
use stt_core::projection;

/// Last-resort zstd compression ratio, used ONLY when the sample is too small to
/// measure a real ratio (see [`crate::measure::measure_sample`]). A deliberately
/// conservative rough guess — real STT tiles routinely compress 3–6× — kept as a
/// named constant so this is clearly an unmeasured estimate, not a fact. When a
/// measurement exists the real [`MeasuredEncoding::zstd_ratio`] is used instead.
const FALLBACK_ZSTD_RATIO: f64 = 3.0;

/// Synthetic tiles the framing probe cuts the sample into at its FINEST point.
///
/// 512, and the number is a cost/accuracy measurement rather than a taste:
/// against the two real builds this file quotes, a probe at 512 tiles projects
/// `wpc-fronts` to 0.95× and `osm-nyc` to 1.02× of realized bytes, while 256
/// tiles lands at 1.02×/1.21× and 1024 tiles at 0.92×/0.89×. The probe costs
/// ~50–90 ms at the build-default zstd level 3 (~3 s at 19), which is what keeps
/// it affordable inside [`analyze`].
const PROBE_TILES: usize = 512;

/// Fewest features a probe tile may hold.
///
/// Mirrors `measure::MIN_TILE_FEATURES`: below it a "tile" is mostly frame and
/// the split stops describing occupancy. The probe reduces its tile count rather
/// than measuring noise.
const MIN_PROBE_TILE_FEATURES: usize = 8;

/// Fewest sampled features the framing probe needs before it will report a
/// model.
///
/// 64 gives the finest cut 8 tiles, which is the smallest ladder on which the
/// three-point curvature estimate ([`SizeModel::bytes_per_tile_stderr`]) means
/// anything. Below it the honest answer is `None` — no model, and the caller
/// says so — rather than a coefficient fitted to four points.
const MIN_PROBE_FEATURES: usize = 64;

/// Encoder column name the geometry payload is attributed to.
const GEOMETRY_COLUMN: &str = "geometry";

/// Density analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityAnalysis {
    /// Per-zoom tile statistics across the recommended zoom range
    pub per_zoom: Vec<ZoomDensity>,
    /// Estimated total tile count at recommended settings (summed over zooms)
    pub estimated_tile_count: usize,
    /// Estimated archive size in bytes (compressed)
    pub estimated_archive_size: usize,
    /// Potential issues identified
    pub issues: Vec<DensityIssue>,
}

impl DensityAnalysis {
    /// Estimated tiled FEATURE INSTANCES at recommended settings (summed over
    /// zooms) — a feature clipped across N tiles counts N times, which is what
    /// the builder reports as `metadata.feature_count` and what the archive
    /// actually pays for.
    ///
    /// A method rather than a field: `DensityAnalysis` is built by struct
    /// literal in six modules this change does not own, and a new required field
    /// would break every one of them. The per-zoom rows carry the number
    /// losslessly (see [`ZoomDensity::tiled_instances`]).
    pub fn estimated_tiled_instances(&self) -> usize {
        self.per_zoom.iter().map(|z| z.tiled_instances()).sum()
    }
}

/// A measured two-coefficient archive-size model: framing per tile, content per
/// feature.
///
/// # Why two coefficients and not one
///
/// The single-coefficient predecessor was `Ŝ = features × bytes_per_feature`.
/// Its form — proportional to feature count, intercept zero — is an ASSUMPTION,
/// and on real archives it is the wrong one: `osm-nyc-changesets` spends
/// ~105 MB on 380,007 features spread over 301,406 tiles, of which the
/// per-feature content is ~9 MB and the per-tile framing is ~96 MB. A model
/// blind to tiles under-projected it by 10×.
///
/// # Why this is not the analytic size model the register rejects
///
/// Neither coefficient is derived, guessed, or curve-fitted from dataset
/// statistics. Both are read off REAL ENCODES of this run's own sample through
/// the production encoder and the production compressor
/// ([`measure_size_model`]): `bytes_per_feature` is a one-tile encode divided by
/// its feature count, `bytes_per_tile` is the difference between two encodes of
/// the SAME features cut into different numbers of tiles, divided by the tile
/// difference. The only structure imposed is that splitting features across more
/// tiles costs a roughly constant amount per extra tile — and
/// [`Self::bytes_per_tile_stderr`] MEASURES how badly that holds, from a third
/// encode, instead of asserting it.
///
/// # The encode is in packed mode
///
/// Every probe encode installs a [`TemplateCollector`], which is what a real
/// `stt-build` run does: formatVersion 3 hoists Arrow schema templates into the
/// manifest and tile frames carry 16-byte hash references instead. Measuring
/// with inline schemas — which is what `measure::measure_sample_layout` does,
/// correctly, for its own purposes — charges every tile a compressed copy of the
/// schema and overstates `bytes_per_tile` by ~2× on both datasets checked.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SizeModel {
    /// Measured marginal compressed bytes each additional tile costs, at the
    /// probe's finest cut. Framing: Arrow frame header, per-layer section TOC,
    /// schema-template references, and the zstd frame the tile is compressed in.
    pub bytes_per_tile: f64,
    /// Measured spread of [`Self::bytes_per_tile`] between the probe's two
    /// finest chords — the model's own curvature, i.e. how much the "constant
    /// per tile" premise bends over the range that was actually measured.
    ///
    /// `0.0` means the ladder was too short to see curvature, never "no error".
    /// This is one half of the projection's honest error bar: it is multiplied
    /// by the projected TILE COUNT, so it grows exactly where the extrapolation
    /// is longest.
    pub bytes_per_tile_stderr: f64,
    /// The model's worst RELATIVE residual against the cuts it was measured on
    /// — the other half of the error bar, and the one that says how well the
    /// two-term FORM fits this dataset at all.
    ///
    /// Computed by replaying the finished model against every probe encode
    /// (1, k/4, k/2 and k tiles) and taking the largest `|projected − measured| /
    /// measured`. A dataset whose bytes really are framing + content reports a
    /// few percent; one that is not reports a lot, and the bar widens instead of
    /// the projection quietly being wrong. It is measured, not chosen.
    pub fit_residual: f64,
    /// Measured compressed bytes per feature when the whole sample sits in ONE
    /// tile — content with framing amortised away.
    pub bytes_per_feature: f64,
    /// Measured share of encoded bytes attributed to the geometry column.
    ///
    /// Used to split the content term: geometry is (approximately) CONSERVED
    /// when the clipper cuts a feature across tiles — the vertices are
    /// redistributed, not duplicated — while ids, times and property values are
    /// re-emitted once per tiled instance. So the content charge is
    /// `bytes_per_feature × (share × source_features + (1 − share) × instances)`.
    /// `1.0` (the default when no attribution is available) means "assume
    /// conserved", which is exactly right for point data, where instances and
    /// source features are the same number anyway.
    pub geometry_share: f64,
    /// Tiles the finest probe cut resolved to.
    pub probe_tiles: usize,
    /// Features the probe encoded (the dominant-geometry-kind subset).
    pub probe_features: usize,
    /// zstd level the probe measured at.
    pub zstd_level: i32,
}

impl SizeModel {
    /// Project compressed archive bytes for one zoom's cut.
    ///
    /// `tiles` is the tile count INCLUDING clip replication, `instances` the
    /// tiled feature instances, `source_features` the distinct input features.
    /// The first tile's framing is already inside `bytes_per_feature` (the
    /// one-tile probe), so only `tiles − 1` is charged.
    pub fn project(&self, tiles: u64, instances: u64, source_features: u64) -> u64 {
        let instances = instances.max(source_features) as f64;
        let share = self.geometry_share.clamp(0.0, 1.0);
        let content =
            self.bytes_per_feature * (share * source_features as f64 + (1.0 - share) * instances);
        let framing = self.bytes_per_tile * tiles.saturating_sub(1) as f64;
        let total = content + framing;
        if !total.is_finite() || total <= 0.0 {
            return 0;
        }
        total.round() as u64
    }

    /// The projection's own uncertainty: the measured curvature of
    /// `bytes_per_tile` carried out to the projected tile count, combined in
    /// quadrature with the model's measured misfit against its own probe cuts.
    ///
    /// Both terms are measured. The first grows with the REACH of the
    /// extrapolation, the second with how poorly the two-term form describes
    /// this dataset — which are the two ways this projection can be wrong.
    pub fn projection_stderr(&self, tiles: u64, projected: u64) -> f64 {
        let curvature = self.bytes_per_tile_stderr * tiles.saturating_sub(1) as f64;
        let misfit = self.fit_residual * projected as f64;
        let total = (curvature * curvature + misfit * misfit).sqrt();
        if total.is_finite() {
            total
        } else {
            0.0
        }
    }

    /// How far past the probe's own tile count this projection reaches.
    ///
    /// `1.0` means the archive is no bigger than what was measured; `589.0` is
    /// what `osm-nyc` asks for. Reported in words so nobody reads a projection
    /// as a measurement.
    pub fn extrapolation_factor(&self, tiles: u64) -> f64 {
        if self.probe_tiles == 0 {
            return f64::INFINITY;
        }
        tiles as f64 / self.probe_tiles as f64
    }
}

/// How much the builder's clip-and-place step multiplies a centroid scan.
///
/// Measured from the loader's retained geometries, never assumed: point data
/// yields `(1.0, 1.0)` and every number downstream is byte-identical to the
/// pre-replication behaviour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Replication {
    /// Mean tiled instances per source feature at this `(zoom, bucket)`.
    pub instances_per_feature: f64,
    /// Factor by which geometry coverage inflates the DISTINCT `(x, y, t)` cell
    /// count over the centroid scan.
    pub cell_inflation: f64,
}

impl Replication {
    /// The identity: one instance per feature, no extra cells. What point data
    /// measures, and the fallback when there is no sample to measure.
    pub fn none() -> Self {
        Self {
            instances_per_feature: 1.0,
            cell_inflation: 1.0,
        }
    }

    /// Whether this replication moves anything.
    pub fn is_none(&self) -> bool {
        self.instances_per_feature <= 1.0 && self.cell_inflation <= 1.0
    }
}

/// Tile statistics for one zoom level, with features split into
/// (x, y, time-bucket) tiles by the recommended temporal bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomDensity {
    /// Zoom level
    pub zoom: u8,
    /// Number of (x, y, time-bucket) tiles at this zoom, INCLUDING the extra
    /// cells a clipped trajectory or polygon occupies (see
    /// [`sample_replication`]). For point data this is the exact centroid-scan
    /// count, unchanged.
    pub tile_count: usize,
    /// Average tiled instances per tile — the replication-aware mean.
    pub avg_features_per_tile: f64,
    /// Median features per tile
    ///
    /// ⚠️ Computed on the CENTROID scan (one cell per feature), so on clipped
    /// geometry it describes the distribution of whole features over cells
    /// rather than of clipped pieces over tiles. [`Self::avg_features_per_tile`]
    /// is the replication-aware one. Kept centroid-based deliberately: it is
    /// what [`crate::measure::SyntheticLayout::from_density`] reads, and that
    /// layout must stay a pure function of the occupancy scan.
    pub median_features_per_tile: usize,
    /// Maximum features in any tile
    pub max_features_per_tile: usize,
    /// 95th-percentile features per tile — a skew-robust "typical busy tile"
    /// that, unlike [`Self::max_features_per_tile`], is not pinned by a single
    /// outlier cell.
    #[serde(default)]
    pub p95_features_per_tile: usize,
    /// Share of ALL features that fall in the top 1% most-populated tiles
    /// (0.0–1.0) — a Gini-style spatial-concentration signal. ~0.01 = uniform;
    /// toward 1.0 = a few cells hold almost everything (a hotspot a raw tier
    /// can't serve but a summary tier can). This is the skew signal the
    /// summary-tier advisor triggers on beyond raw totals/averages.
    #[serde(default)]
    pub top1pct_feature_share: f64,
    /// Number of oversized tiles (> 10,000 features)
    pub oversized_tiles: usize,
    /// Number of undersized tiles (< 10 features)
    pub undersized_tiles: usize,
    /// Estimated total size at this zoom, uncompressed (measured-sample
    /// calibrated when a measurement is available, else summed per-feature
    /// formula estimates)
    pub estimated_size_uncompressed: usize,
    /// Estimated total size at this zoom, compressed.
    ///
    /// With a [`SizeModel`] this is `bytes_per_tile × (tiles − 1) + content`
    /// — the two-term projection. Without one (no measurement, or a sample below
    /// [`MIN_PROBE_FEATURES`]) it degrades to the historical per-instance rate
    /// with NO tile term, which under-projects thin-tiled archives badly. Call
    /// [`measure_size_model`] on the same sample to see which of the two you
    /// were handed.
    pub estimated_size_compressed: usize,
}

impl ZoomDensity {
    /// Tiled feature INSTANCES at this zoom: a feature clipped across N tiles
    /// counts N times.
    ///
    /// Derived from [`Self::avg_features_per_tile`] × [`Self::tile_count`],
    /// which is exactly how it was computed — a method rather than a field
    /// because `ZoomDensity` is built by struct literal in modules this change
    /// does not own.
    pub fn tiled_instances(&self) -> usize {
        let n = self.avg_features_per_tile * self.tile_count as f64;
        if !n.is_finite() || n <= 0.0 {
            return 0;
        }
        n.round() as usize
    }
}

/// A potential density issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Suggested fix
    pub suggestion: String,
}

/// Issue severity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Info => write!(f, "INFO"),
            IssueSeverity::Warning => write!(f, "WARNING"),
            IssueSeverity::Error => write!(f, "ERROR"),
        }
    }
}

/// Analyze tile density across the recommended zoom range.
///
/// Models the real build: at every zoom in the recommended
/// `[min_zoom, max_zoom]` range each feature lands in its containing (x, y)
/// tile, split by fixed `--temporal-bucket`-sized time buckets (the
/// recommended bucket from the temporal analysis), and non-point geometry is
/// replicated across every tile it covers exactly as the builder's clipper does
/// ([`sample_replication`]). This is the cut `stt-build` actually makes, so
/// predicted tile counts and sizes track a real build.
///
/// When a `measured` sample encoding is present the run also probes the encoder
/// for a [`SizeModel`] — measured framing per tile and measured content per
/// feature — and the size estimate is that two-term projection. Without a
/// measurement the per-feature formula estimate with an assumed 3x compression
/// ratio is the fallback, and [`DensityAnalysis::size_model`] is `None` so the
/// difference is visible rather than implied.
///
/// # Cost
///
/// The framing probe adds ~4 sample encodes' worth of work at the build-default
/// zstd level (measured: 50 ms on a 4,156-feature LineString sample, 90 ms on a
/// 4,936-feature point sample). It runs only when `measured` is `Some`, so the
/// occupancy pre-pass in `analyze_source_with` — which passes `None` — costs
/// exactly what it did before.
///
/// ⚠️ The probe measures at [`MeasureSettings::default`]'s zstd level, which is
/// the level every caller in this crate takes `measured` at. A caller that
/// measured at another level gets a `size_model` for the DEFAULT level; the
/// field records `zstd_level` so that is checkable rather than silent.
pub fn analyze(
    data: &LoadedData,
    spatial: &SpatialAnalysis,
    temporal: &TemporalAnalysis,
    measured: Option<&MeasuredEncoding>,
) -> Result<DensityAnalysis> {
    let bucket_ms = temporal.recommended_bucket_ms;
    let zooms: Vec<u8> = (spatial.recommended_min_zoom..=spatial.recommended_max_zoom).collect();
    tracing::debug!(
        "density: bucketing {} features into (x, y, t/{}ms) tiles at zooms {:?}",
        data.features.len(),
        bucket_ms,
        zooms
    );

    // The size model is measured once per run and shared by every zoom: framing
    // and content rates are properties of the ENCODER at a zstd level, not of
    // the cut. What the cut changes is the tile and instance counts the model is
    // evaluated at, and those are counted per zoom below.
    let size_model = match measured {
        Some(m) => measure_size_model(
            &data.sample,
            MeasureSettings::default().zstd_level,
            geometry_share(m),
        )
        .context("density: framing probe")?,
        None => None,
    };

    let mut per_zoom = Vec::with_capacity(zooms.len());
    for &zoom in &zooms {
        per_zoom.push(bucket_zoom(
            data,
            zoom,
            bucket_ms,
            measured,
            size_model.as_ref(),
        ));
    }

    let estimated_tile_count = per_zoom.iter().map(|z| z.tile_count).sum();
    let estimated_archive_size = per_zoom.iter().map(|z| z.estimated_size_compressed).sum();

    let issues = identify_issues(
        data,
        spatial,
        &per_zoom,
        estimated_tile_count,
        estimated_archive_size,
    );

    Ok(DensityAnalysis {
        per_zoom,
        estimated_tile_count,
        estimated_archive_size,
        issues,
    })
}

/// The measured share of encoded bytes attributed to the geometry column, or
/// `1.0` when the measurement carries no attribution.
///
/// `1.0` = "content is conserved when the clipper splits a feature", the
/// behaviour that is exactly right for points and the conservative reading for
/// everything else (it declines to inflate the content term on the strength of
/// an attribution that was not measured).
fn geometry_share(measured: &MeasuredEncoding) -> f64 {
    let share: f64 = measured
        .per_column
        .iter()
        .filter(|c| c.name == GEOMETRY_COLUMN)
        .map(|c| c.share)
        .sum();
    if share.is_finite() && share > 0.0 {
        share.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Bucket every feature into its (x, y, time-bucket) tile at one zoom and
/// compute per-tile statistics. `bucket_ms == 0` (no temporal bucketing, e.g.
/// an instantaneous dataset) collapses to a single time bucket per tile.
fn bucket_zoom(
    data: &LoadedData,
    zoom: u8,
    bucket_ms: u64,
    measured: Option<&MeasuredEncoding>,
    model: Option<&SizeModel>,
) -> ZoomDensity {
    // (feature count, estimated bytes) per (x, y, t_bucket) tile.
    let mut tiles: HashMap<(u32, u32, u64), (usize, usize)> = HashMap::new();

    // The scan itself files each feature into the single tile containing its
    // CENTROID — exact for points, and the only thing it can do, because
    // `AnalyzableFeature` carries a centroid rather than a geometry. The
    // builder's trajectory/polygon clipping is then applied on top as a MEASURED
    // inflation (`replication`, below) taken from the loader's retained
    // geometries, so `tile_count` and `tiled_instances` describe the real cut
    // rather than the centroid one.
    for feature in &data.features {
        if let Ok((x, y)) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
            let t_bucket = if bucket_ms > 0 {
                feature.timestamp / bucket_ms
            } else {
                0
            };
            let entry = tiles.entry((x, y, t_bucket)).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += feature.estimated_size;
        }
    }

    let mut feature_counts: Vec<usize> = tiles.values().map(|&(count, _)| count).collect();
    feature_counts.sort_unstable();
    let total_uncompressed: usize = tiles.values().map(|&(_, bytes)| bytes).sum();

    let centroid_tile_count = feature_counts.len();
    let total_features: usize = feature_counts.iter().sum();

    // Clip replication, measured on the retained geometries. Point data — and
    // any run whose loader kept no sample — yields the identity, so every number
    // below is bit-for-bit what the centroid scan produced before this existed.
    let replication = sample_replication(&data.sample, zoom, bucket_ms);
    let tiled_instances = scale_count(total_features as u64, replication.instances_per_feature);
    // A cell cannot hold fewer than one instance, and coverage can only ADD
    // cells, so the inflated count is clamped into [centroid cells, instances].
    let tile_count = scale_count(centroid_tile_count as u64, replication.cell_inflation).clamp(
        centroid_tile_count as u64,
        tiled_instances.max(centroid_tile_count as u64),
    ) as usize;
    let tiled_instances = tiled_instances as usize;

    let avg_features_per_tile = if tile_count > 0 {
        tiled_instances as f64 / tile_count as f64
    } else {
        0.0
    };
    let median_features_per_tile = feature_counts
        .get(centroid_tile_count / 2)
        .copied()
        .unwrap_or(0);
    let max_features_per_tile = feature_counts.last().copied().unwrap_or(0);
    // Skew signals (feature_counts is sorted ascending, so the top-k tiles are
    // the last k). p95 = a busy-but-not-outlier tile; top1pct share = how much
    // of the data concentrates in the densest 1% of cells.
    let p95_features_per_tile = if centroid_tile_count > 0 {
        let idx = ((centroid_tile_count as f64 * 0.95).ceil() as usize)
            .min(centroid_tile_count)
            .saturating_sub(1);
        feature_counts[idx]
    } else {
        0
    };
    let top1pct_feature_share = if total_features > 0 && centroid_tile_count > 0 {
        let k = ((centroid_tile_count as f64 * 0.01).ceil() as usize).clamp(1, centroid_tile_count);
        let top_sum: usize = feature_counts[centroid_tile_count - k..].iter().sum();
        top_sum as f64 / total_features as f64
    } else {
        0.0
    };
    // 10,000-feature "oversized" threshold is a rough rule of thumb for a tile
    // that will be slow to decode/render; it is not a hard format limit.
    let oversized_tiles = feature_counts.iter().filter(|&&c| c > 10_000).count();
    let undersized_tiles = feature_counts.iter().filter(|&&c| c < 10).count();

    // Size estimates, best evidence first:
    //
    //  1. a measured two-coefficient `SizeModel` — framing × tiles + content ×
    //     features. The only path with a tile term, and the only one that can
    //     project a thin-tiled archive.
    //  2. a measured `bytes_per_feature` with NO tile term (the pre-SizeModel
    //     behaviour, kept as the rollback for samples too small to probe). It
    //     now charges TILED INSTANCES rather than source features, so clip
    //     replication is at least counted once.
    //  3. no measurement at all: the summed per-feature formula and a rough
    //     assumed compression ratio.
    let (estimated_size_uncompressed, estimated_size_compressed) = match (model, measured) {
        (Some(model), m) => {
            let compressed = model.project(
                tile_count as u64,
                tiled_instances as u64,
                total_features as u64,
            ) as usize;
            let ratio = m.map(|m| m.zstd_ratio).unwrap_or(FALLBACK_ZSTD_RATIO);
            let uncompressed = (compressed as f64 * ratio).round() as usize;
            (uncompressed, compressed)
        }
        (None, Some(m)) => {
            let compressed = (tiled_instances as f64 * m.bytes_per_feature).round() as usize;
            let uncompressed = (compressed as f64 * m.zstd_ratio).round() as usize;
            (uncompressed, compressed)
        }
        (None, None) => (
            total_uncompressed,
            (total_uncompressed as f64 / FALLBACK_ZSTD_RATIO).round() as usize,
        ),
    };

    ZoomDensity {
        zoom,
        tile_count,
        avg_features_per_tile,
        median_features_per_tile,
        max_features_per_tile,
        p95_features_per_tile,
        top1pct_feature_share,
        oversized_tiles,
        undersized_tiles,
        estimated_size_uncompressed,
        estimated_size_compressed,
    }
}

/// Scale a count by a measured replication factor, saturating and never below
/// the input.
///
/// A factor is a measured ratio, so it is `>= 1.0` by construction; the clamp is
/// a guard against a degenerate sample producing a fraction, not a correction.
/// Shared with [`crate::budget_solver`] so the solver's own occupancy scan and
/// this one cannot drift apart.
pub(crate) fn scale_count(count: u64, factor: f64) -> u64 {
    if !factor.is_finite() || factor <= 1.0 {
        return count;
    }
    let scaled = (count as f64 * factor).round();
    if !scaled.is_finite() || scaled >= u64::MAX as f64 {
        return count;
    }
    (scaled as u64).max(count)
}

// ----------------------------------------------------------------------------
// Clip replication — measured from the retained geometries
// ----------------------------------------------------------------------------

/// Measure how far the builder's clipper spreads this dataset's geometry beyond
/// its centroid cells, at one `(zoom, bucket_ms)` cut.
///
/// # What it walks
///
/// The loader retains a deterministic stride sample of FULL geometries. For each
/// one this counts the `(x, y)` tiles the geometry actually touches, using the
/// same rule the builder uses to decide where a piece is emitted:
///
/// | geometry | rule | fidelity |
/// |---|---|---|
/// | Point / MultiPoint | one cell per member | exact (`tiler::place_multipoint`) |
/// | Line / LineString / MultiLineString | supercover walk of every segment through the tile grid | exact — measured 7,405 cells against `wpc-fronts`' real 7,405 |
/// | Polygon / MultiPolygon / Rect / Triangle | every cell the ring bbox covers | UPPER BOUND — the builder clips rings per tile and drops empty pieces, so a concave or diagonal polygon is over-counted here |
///
/// # What it returns
///
/// Two ratios rather than absolute counts, because the sample is a fraction of
/// the dataset: `instances_per_feature` scales linearly and is unbiased, while
/// `cell_inflation` is the sample's own expanded-vs-centroid distinct-cell ratio
/// and is an OVER-estimate on a sparse sample (sampled features share fewer
/// cells with each other than the full population does). The caller clamps the
/// inflated count at the instance count, which is the true ceiling.
///
/// # Determinism
///
/// Pure counting over a `HashSet`: only cardinalities are read, never iteration
/// order. Same file in, same ratios out.
pub fn sample_replication(sample: &[SampledFeature], zoom: u8, bucket_ms: u64) -> Replication {
    if sample.is_empty() {
        return Replication::none();
    }
    // Points cannot replicate, so a point-only sample skips the scan entirely
    // and returns the identity — the pre-replication behaviour, for free.
    if sample.iter().all(|f| is_point_geometry(&f.geometry)) {
        return Replication::none();
    }

    let mut covered: HashSet<(u32, u32, u64)> = HashSet::new();
    let mut centroid: HashSet<(u32, u32, u64)> = HashSet::new();
    let mut instances = 0usize;
    let mut placed = 0usize;
    let mut cells: Vec<(u32, u32)> = Vec::new();
    for feature in sample {
        let t_bucket = if bucket_ms > 0 {
            feature.timestamp_ms / bucket_ms
        } else {
            0
        };
        cells.clear();
        covered_cells(&feature.geometry, zoom, &mut cells);
        if cells.is_empty() {
            continue;
        }
        placed += 1;
        instances += cells.len();
        for &(x, y) in &cells {
            covered.insert((x, y, t_bucket));
        }
        if let Some((lon, lat)) = centroid_of(&feature.geometry) {
            if let Ok((x, y)) = projection::lonlat_to_tile(lon, lat, zoom) {
                centroid.insert((x, y, t_bucket));
            }
        }
    }
    if placed == 0 || centroid.is_empty() {
        return Replication::none();
    }

    let instances_per_feature = (instances as f64 / placed as f64).max(1.0);
    let cell_inflation = (covered.len() as f64 / centroid.len() as f64)
        .max(1.0)
        .min(instances_per_feature);
    Replication {
        instances_per_feature,
        cell_inflation,
    }
}

/// Whether a geometry is point-kind (cannot be clipped across tiles).
fn is_point_geometry(geometry: &geo_types::Geometry<f64>) -> bool {
    use geo_types::Geometry as G;
    matches!(geometry, G::Point(_) | G::MultiPoint(_))
}

/// The geometry's representative lon/lat — the same rule the loader files an
/// `AnalyzableFeature` under, so the two scans are comparable.
fn centroid_of(geometry: &geo_types::Geometry<f64>) -> Option<(f64, f64)> {
    use geo::algorithm::centroid::Centroid;
    geometry.centroid().map(|p| (p.x(), p.y()))
}

/// Append every `(x, y)` tile `geometry` occupies at `zoom` to `out`
/// (deduplicated, sorted).
fn covered_cells(geometry: &geo_types::Geometry<f64>, zoom: u8, out: &mut Vec<(u32, u32)>) {
    use geo_types::Geometry as G;
    match geometry {
        G::Point(p) => push_cell(p.x(), p.y(), zoom, out),
        G::MultiPoint(mp) => {
            for p in &mp.0 {
                push_cell(p.x(), p.y(), zoom, out);
            }
        }
        G::Line(l) => walk_segment(l.start.x, l.start.y, l.end.x, l.end.y, zoom, out),
        G::LineString(ls) => walk_line(&ls.0, zoom, out),
        G::MultiLineString(mls) => {
            for ls in &mls.0 {
                walk_line(&ls.0, zoom, out);
            }
        }
        // Polygon coverage: the builder sweeps the ring bbox and clips per tile,
        // keeping whatever survives. Sweeping the bbox here matches that sweep
        // and over-counts only where the clip finds nothing.
        G::Polygon(p) => push_ring_bbox(&p.exterior().0, zoom, out),
        G::MultiPolygon(mp) => {
            for p in &mp.0 {
                push_ring_bbox(&p.exterior().0, zoom, out);
            }
        }
        G::Rect(r) => push_ring_bbox(&r.to_polygon().exterior().0, zoom, out),
        G::Triangle(t) => push_ring_bbox(&t.to_polygon().exterior().0, zoom, out),
        // A GeometryCollection has no encoder bucket; the measurement excludes
        // it exactly as `measure::kind_of` does.
        G::GeometryCollection(_) => {}
    }
    out.sort_unstable();
    out.dedup();
}

fn push_cell(lon: f64, lat: f64, zoom: u8, out: &mut Vec<(u32, u32)>) {
    if let Ok(cell) = projection::lonlat_to_tile(lon, lat, zoom) {
        out.push(cell);
    }
}

fn walk_line(coords: &[geo_types::Coord<f64>], zoom: u8, out: &mut Vec<(u32, u32)>) {
    match coords.len() {
        0 => {}
        1 => push_cell(coords[0].x, coords[0].y, zoom, out),
        _ => {
            for pair in coords.windows(2) {
                walk_segment(pair[0].x, pair[0].y, pair[1].x, pair[1].y, zoom, out);
            }
        }
    }
}

/// Web-Mercator world coordinates (tile units at `zoom`), clamped at the poles
/// exactly as [`projection::lonlat_to_tile`] clamps.
fn world_coords(lon: f64, lat: f64, zoom: u8) -> Option<(f64, f64)> {
    if !lon.is_finite() || !lat.is_finite() || !(-180.0..=180.0).contains(&lon) {
        return None;
    }
    let lat = lat.clamp(-projection::MERCATOR_MAX_LAT, projection::MERCATOR_MAX_LAT);
    let n = (1u64 << zoom) as f64;
    let x = (lon + 180.0) / 360.0 * n;
    let y = (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n;
    Some((x, y))
}

/// Supercover DDA: every tile the segment passes through, not merely the ones
/// its endpoints land in.
///
/// This is the walk the trajectory clipper's output implies — a clipped segment
/// is emitted wherever the polyline enters a tile — and it is what makes the
/// projection match reality on line data: measured against the real
/// `wpc-fronts` build it reproduces 7,405 tiles out of 7,405 and 11,160 tiled
/// instances against 11,156.
///
/// ⚠️ An ANTIMERIDIAN-crossing edge (`|Δlon| > 180°`) contributes its two
/// endpoint cells only. The builder splits such a polyline at ±180 before
/// clipping (`clip::split_polyline_at_antimeridian`), so walking the segment as
/// drawn would sweep the whole world the wrong way round — 2^zoom columns of
/// tiles that the build never emits, at 2^zoom iterations. Same per-edge test
/// the splitter uses.
fn walk_segment(lon0: f64, lat0: f64, lon1: f64, lat1: f64, zoom: u8, out: &mut Vec<(u32, u32)>) {
    let (Some((ax, ay)), Some((bx, by))) = (
        world_coords(lon0, lat0, zoom),
        world_coords(lon1, lat1, zoom),
    ) else {
        return;
    };
    if (lon1 - lon0).abs() > 180.0 {
        push_cell(lon0, lat0, zoom, out);
        push_cell(lon1, lat1, zoom, out);
        return;
    }
    let n = 1i64 << zoom;
    let clamp = |v: i64| v.clamp(0, n - 1) as u32;

    let (mut cx, mut cy) = (ax.floor() as i64, ay.floor() as i64);
    let (ex, ey) = (bx.floor() as i64, by.floor() as i64);
    let (dx, dy) = (bx - ax, by - ay);
    let step_x: i64 = if dx > 0.0 { 1 } else { -1 };
    let step_y: i64 = if dy > 0.0 { 1 } else { -1 };
    // Parametric distance (in [0, 1] along the segment) to the next grid line on
    // each axis, and the distance between successive grid lines.
    let mut t_max_x = if dx == 0.0 {
        f64::INFINITY
    } else {
        let next = if dx > 0.0 { (cx + 1) as f64 } else { cx as f64 };
        (next - ax) / dx
    };
    let mut t_max_y = if dy == 0.0 {
        f64::INFINITY
    } else {
        let next = if dy > 0.0 { (cy + 1) as f64 } else { cy as f64 };
        (next - ay) / dy
    };
    let t_delta_x = if dx == 0.0 {
        f64::INFINITY
    } else {
        (1.0 / dx).abs()
    };
    let t_delta_y = if dy == 0.0 {
        f64::INFINITY
    } else {
        (1.0 / dy).abs()
    };

    // A segment can cross at most (Δx + Δy + 2) grid lines; the guard bounds the
    // walk against a NaN or a pathological antimeridian span rather than
    // trusting the float loop to terminate.
    let budget = ((dx.abs() + dy.abs()).min(1e6) as i64) + 2;
    for _ in 0..=budget {
        out.push((clamp(cx), clamp(cy)));
        if cx == ex && cy == ey {
            return;
        }
        if t_max_x < t_max_y {
            if t_max_x > 1.0 {
                return;
            }
            cx += step_x;
            t_max_x += t_delta_x;
        } else {
            if t_max_y > 1.0 {
                return;
            }
            cy += step_y;
            t_max_y += t_delta_y;
        }
    }
}

/// Most cells one polygon ring may claim.
///
/// A bound on the scan, not on the format: 1M cells is far beyond any real
/// polygon at a recommended zoom, and stopping there keeps a corrupt ring from
/// turning an O(1) analysis into an O(4^zoom) one. A ring that hits it falls
/// back to its vertex cells and is therefore UNDER-counted — the reason the
/// bound is set where no real geometry reaches it.
const MAX_RING_CELLS: u64 = 1 << 20;

/// Every cell in the bounding box of a polygon ring.
///
/// ⚠️ An ANTIMERIDIAN-crossing ring (any edge with `|Δlon| > 180°`) claims its
/// vertex cells only. The builder SPLITS such a ring per hemisphere before
/// clipping (`clip::split_polygon_at_antimeridian`), so its pieces land at both
/// edges of the world and never across the middle — while the raw bbox spans
/// every column in between. Same per-edge test the splitter uses, and the same
/// one `place_polygon` applies.
fn push_ring_bbox(ring: &[geo_types::Coord<f64>], zoom: u8, out: &mut Vec<(u32, u32)>) {
    let crosses = ring.windows(2).any(|w| (w[1].x - w[0].x).abs() > 180.0);
    if crosses {
        for c in ring {
            push_cell(c.x, c.y, zoom, out);
        }
        return;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    let mut any = false;
    for c in ring {
        if let Ok((x, y)) = projection::lonlat_to_tile(c.x, c.y, zoom) {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            any = true;
        }
    }
    if !any {
        return;
    }
    let cells = (x1 - x0 + 1) as u64 * (y1 - y0 + 1) as u64;
    if cells > MAX_RING_CELLS {
        for c in ring {
            push_cell(c.x, c.y, zoom, out);
        }
        return;
    }
    for x in x0..=x1 {
        for y in y0..=y1 {
            out.push((x, y));
        }
    }
}

// ----------------------------------------------------------------------------
// The framing probe — both coefficients measured, neither assumed
// ----------------------------------------------------------------------------

/// Measure a [`SizeModel`] by encoding the run's own sample at four different
/// tile counts through the production encoder and compressor.
///
/// # The four encodes
///
/// With `n` usable features and `k = min(PROBE_TILES, n / MIN_PROBE_TILE_FEATURES)`:
///
/// | cut | what it pins |
/// |---|---|
/// | 1 tile | `bytes_per_feature` — content with framing amortised away |
/// | `k/4` tiles | the coarser chord, for curvature |
/// | `k/2` tiles | shared endpoint of both chords |
/// | `k` tiles | `bytes_per_tile` = (bytes(k) − bytes(k/2)) / (k − k/2) |
///
/// `bytes_per_tile_stderr` is the gap between the two chords. It is the only
/// error term that knows how far the model is being carried, and it is
/// MEASURED: a dataset whose bytes-vs-tiles curve is straight over the probe's
/// range reports a small one, a dataset that bends reports a large one.
///
/// # Why the finest cut and not the coarsest
///
/// The marginal cost of a tile is not constant — it falls as tiles thin out and
/// the compressor loses cross-feature context — so the chord that matters is the
/// one closest to the archive's real occupancy (1–2 features/tile on the
/// datasets that motivated this). Taking the whole-range chord from 1 tile
/// instead over-charges by ~35%.
///
/// # Determinism
///
/// Contiguous integer splits (`n·i/k`), first-seen property order, a fresh
/// [`TemplateCollector`] per probe, no RNG and no clock. Same sample and level
/// in, byte-identical model out.
///
/// Returns `Ok(None)` — never an error — when the sample is below
/// [`MIN_PROBE_FEATURES`] or cannot fill four distinct cuts. A caller with no
/// model must say it has no model, not invent one.
pub fn measure_size_model(
    sample: &[SampledFeature],
    zstd_level: i32,
    geometry_share: f64,
) -> Result<Option<SizeModel>> {
    let Some((kind, subset)) = dominant_subset(sample) else {
        return Ok(None);
    };
    let n = subset.len();
    if n < MIN_PROBE_FEATURES {
        return Ok(None);
    }
    let k_fine = PROBE_TILES.min(n / MIN_PROBE_TILE_FEATURES);
    if k_fine < 8 {
        return Ok(None);
    }
    let k_mid = k_fine / 2;
    let k_quarter = k_fine / 4;

    let bytes_1 = encode_probe(&subset, kind, 1, zstd_level)?;
    let bytes_quarter = encode_probe(&subset, kind, k_quarter, zstd_level)?;
    let bytes_mid = encode_probe(&subset, kind, k_mid, zstd_level)?;
    let bytes_fine = encode_probe(&subset, kind, k_fine, zstd_level)?;

    let fine_chord = (bytes_fine as f64 - bytes_mid as f64) / (k_fine - k_mid) as f64;
    let coarse_chord = (bytes_mid as f64 - bytes_quarter as f64) / (k_mid - k_quarter) as f64;
    // A negative marginal means the split MADE the archive smaller, which zstd
    // can do on tiny frames; it is not evidence that tiles are free, so the
    // coefficient floors at zero and the curvature term still records the swing.
    let bytes_per_tile = fine_chord.max(0.0);
    let bytes_per_tile_stderr = (fine_chord - coarse_chord).abs();
    let bytes_per_feature = bytes_1 as f64 / n as f64;
    if !bytes_per_tile.is_finite()
        || !bytes_per_tile_stderr.is_finite()
        || !bytes_per_feature.is_finite()
    {
        return Ok(None);
    }

    let mut model = SizeModel {
        bytes_per_tile,
        bytes_per_tile_stderr,
        bytes_per_feature,
        fit_residual: 0.0,
        geometry_share: if geometry_share.is_finite() {
            geometry_share.clamp(0.0, 1.0)
        } else {
            1.0
        },
        probe_tiles: k_fine,
        probe_features: n,
        zstd_level,
    };
    // Goodness of fit, measured: replay the finished model against every encode
    // that produced it. This is the only term that notices when "framing +
    // content" is simply the wrong shape for a dataset, and it is what stops the
    // error bar from being narrower than the model's own known misses.
    let features = n as u64;
    model.fit_residual = [
        (1u64, bytes_1),
        (k_quarter as u64, bytes_quarter),
        (k_mid as u64, bytes_mid),
        (k_fine as u64, bytes_fine),
    ]
    .into_iter()
    .filter(|(_, measured)| *measured > 0)
    .map(|(tiles, measured)| {
        let projected = model.project(tiles, features, features) as f64;
        (projected - measured as f64).abs() / measured as f64
    })
    .fold(0.0f64, f64::max);
    if !model.fit_residual.is_finite() {
        model.fit_residual = 0.0;
    }

    Ok(Some(model))
}

/// The encoder geometry bucket a probe measures — the tiler emits one layer per
/// kind, so a synthetic layer must be single-kind. Mirrors `measure::GeomKind`
/// (a private type there) including its point > line > polygon tie-break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeKind {
    Point,
    Line,
    Polygon,
}

/// The dominant-geometry-kind subset of a sample, or `None` when nothing
/// classifiable is present.
fn dominant_subset(sample: &[SampledFeature]) -> Option<(ProbeKind, Vec<&SampledFeature>)> {
    use geo_types::Geometry as G;
    let kind_of = |g: &geo_types::Geometry<f64>| match g {
        G::Point(_) | G::MultiPoint(_) => Some(ProbeKind::Point),
        G::Line(_) | G::LineString(_) | G::MultiLineString(_) => Some(ProbeKind::Line),
        G::Polygon(_) | G::MultiPolygon(_) | G::Rect(_) | G::Triangle(_) => {
            Some(ProbeKind::Polygon)
        }
        G::GeometryCollection(_) => None,
    };
    let mut counts = [0usize; 3];
    for f in sample {
        match kind_of(&f.geometry) {
            Some(ProbeKind::Point) => counts[0] += 1,
            Some(ProbeKind::Line) => counts[1] += 1,
            Some(ProbeKind::Polygon) => counts[2] += 1,
            None => {}
        }
    }
    let mut dominant = ProbeKind::Point;
    let mut best = counts[0];
    for (kind, count) in [
        (ProbeKind::Line, counts[1]),
        (ProbeKind::Polygon, counts[2]),
    ] {
        if count > best {
            dominant = kind;
            best = count;
        }
    }
    if best == 0 {
        return None;
    }
    let subset: Vec<&SampledFeature> = sample
        .iter()
        .filter(|f| kind_of(&f.geometry) == Some(dominant))
        .collect();
    Some((dominant, subset))
}

/// Encode `subset` cut into `tiles` contiguous synthetic tiles and return the
/// summed compressed bytes.
///
/// ⚠️ A [`TemplateCollector`] is installed, which is what makes this a PACKED
/// measurement: schema templates are hoisted into `manifest.schemas` and tile
/// frames carry hash references, exactly as `stt-build` writes them. Measuring
/// with inline schemas charges every tile a compressed schema copy and doubles
/// the apparent per-tile cost.
fn encode_probe(
    subset: &[&SampledFeature],
    kind: ProbeKind,
    tiles: usize,
    zstd_level: i32,
) -> Result<usize> {
    let tiles = tiles.clamp(1, subset.len().max(1));
    let collector = Arc::new(TemplateCollector::new());
    let cfg = EncoderConfig {
        template_collector: Some(Arc::clone(&collector)),
        ..EncoderConfig::default()
    };
    let mut total = 0usize;
    for i in 0..tiles {
        let start = subset.len() * i / tiles;
        let end = subset.len() * (i + 1) / tiles;
        if end <= start {
            continue;
        }
        let layer = build_probe_layer(&subset[start..end], kind);
        let payload = encode_tile_with(&[layer], &cfg).context("probe tile encode failed")?;
        let compressed = compress_zstd_with_dict_level(&payload, None, zstd_level)
            .context("probe tile compression failed")?;
        total += compressed.len();
    }
    Ok(total)
}

/// Assemble one synthetic tile layer from sampled features.
///
/// Mirrors `measure::build_layer` (private there): first-seen property order,
/// Numeric vs Categorical from the first value seen, multi-part geometries
/// flattened to one row per part — the shape a tiler split produces.
fn build_probe_layer(subset: &[&SampledFeature], kind: ProbeKind) -> ColumnarLayer {
    use geo_types::Geometry as G;

    let mut names: Vec<String> = Vec::new();
    let mut is_numeric: Vec<bool> = Vec::new();
    for feature in subset {
        for (name, value) in &feature.properties {
            if !names.iter().any(|n| n == name) {
                names.push(name.clone());
                is_numeric.push(matches!(value, PropValue::Number(_)));
            }
        }
    }

    let ring_coords = |ls: &geo_types::LineString<f64>| -> Vec<Coord> {
        ls.0.iter().map(|c| [c.x, c.y]).collect()
    };
    let polygon_rings = |p: &geo_types::Polygon<f64>| -> Vec<Vec<Coord>> {
        std::iter::once(p.exterior())
            .chain(p.interiors().iter())
            .map(&ring_coords)
            .collect()
    };

    let mut ids: Vec<u64> = Vec::new();
    let mut starts: Vec<i64> = Vec::new();
    let mut ends: Vec<i64> = Vec::new();
    let mut points: Vec<Coord> = Vec::new();
    let mut lines: Vec<Vec<Coord>> = Vec::new();
    let mut polygons: Vec<Vec<Vec<Coord>>> = Vec::new();
    let mut numeric_cols: Vec<Vec<Option<f64>>> = vec![Vec::new(); names.len()];
    let mut categorical_cols: Vec<Vec<Option<String>>> = vec![Vec::new(); names.len()];

    for feature in subset {
        let parts = match (kind, &feature.geometry) {
            (ProbeKind::Point, G::Point(p)) => {
                points.push([p.x(), p.y()]);
                1
            }
            (ProbeKind::Point, G::MultiPoint(mp)) => {
                for p in &mp.0 {
                    points.push([p.x(), p.y()]);
                }
                mp.0.len()
            }
            (ProbeKind::Line, G::Line(l)) => {
                lines.push(vec![[l.start.x, l.start.y], [l.end.x, l.end.y]]);
                1
            }
            (ProbeKind::Line, G::LineString(ls)) => {
                lines.push(ring_coords(ls));
                1
            }
            (ProbeKind::Line, G::MultiLineString(mls)) => {
                for ls in &mls.0 {
                    lines.push(ring_coords(ls));
                }
                mls.0.len()
            }
            (ProbeKind::Polygon, G::Polygon(p)) => {
                polygons.push(polygon_rings(p));
                1
            }
            (ProbeKind::Polygon, G::MultiPolygon(mp)) => {
                for p in &mp.0 {
                    polygons.push(polygon_rings(p));
                }
                mp.0.len()
            }
            (ProbeKind::Polygon, G::Rect(r)) => {
                polygons.push(polygon_rings(&r.to_polygon()));
                1
            }
            (ProbeKind::Polygon, G::Triangle(t)) => {
                polygons.push(polygon_rings(&t.to_polygon()));
                1
            }
            _ => 0,
        };
        for _ in 0..parts {
            ids.push(ids.len() as u64);
            starts.push(feature.timestamp_ms as i64);
            ends.push(feature.timestamp_ms as i64);
            for (col, name) in names.iter().enumerate() {
                let value = feature
                    .properties
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, v)| v);
                if is_numeric[col] {
                    numeric_cols[col].push(match value {
                        Some(PropValue::Number(x)) => Some(*x),
                        _ => None,
                    });
                } else {
                    categorical_cols[col].push(match value {
                        Some(PropValue::Text(s)) => Some(s.clone()),
                        _ => None,
                    });
                }
            }
        }
    }

    let geometry = match kind {
        ProbeKind::Point => GeometryColumn::Point(points),
        ProbeKind::Line => GeometryColumn::LineString(lines),
        ProbeKind::Polygon => GeometryColumn::Polygon(polygons),
    };
    let properties = names
        .into_iter()
        .enumerate()
        .map(|(col, name)| {
            let column = if is_numeric[col] {
                PropertyColumn::Numeric(std::mem::take(&mut numeric_cols[col]))
            } else {
                PropertyColumn::Categorical(std::mem::take(&mut categorical_cols[col]))
            };
            (name, column)
        })
        .collect();

    ColumnarLayer {
        polygon_parts: None,
        name: "default".to_string(),
        feature_ids: ids,
        start_times: starts,
        end_times: ends,
        geometry,
        vertex_times: None,
        vertex_values: None,
        vertex_value_matrix: None,
        triangles: None,
        properties,
    }
}

/// Identify potential issues from the per-zoom breakdown. Every suggestion
/// names real `stt-build` flags; per-tile budgets are always described as
/// opt-in with a data-loss tradeoff.
fn identify_issues(
    data: &LoadedData,
    spatial: &SpatialAnalysis,
    per_zoom: &[ZoomDensity],
    total_tile_count: usize,
    estimated_archive_size: usize,
) -> Vec<DensityIssue> {
    let mut issues = Vec::new();

    let oversized: usize = per_zoom.iter().map(|z| z.oversized_tiles).sum();
    let undersized: usize = per_zoom.iter().map(|z| z.undersized_tiles).sum();
    let max_features = per_zoom
        .iter()
        .map(|z| z.max_features_per_tile)
        .max()
        .unwrap_or(0);

    // Check for oversized tiles
    if oversized > 0 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!(
                "{} tiles exceed 10,000 features (max: {})",
                oversized, max_features
            ),
            suggestion: "Use a finer --temporal-bucket to spread features over more time \
                         buckets, or opt into per-tile budgets (--maximum-tile-bytes / \
                         --maximum-tile-features, optionally --drop-densest-as-needed) — \
                         budgets drop features to fit, trading data loss for tile size. For \
                         very dense point sets, --summary-tier bakes aggregate overview tiles"
                .to_string(),
        });
    }

    // Check for many undersized tiles
    let undersized_pct = if total_tile_count > 0 {
        undersized as f64 / total_tile_count as f64 * 100.0
    } else {
        0.0
    };
    if undersized_pct > 20.0 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Info,
            description: format!(
                "{:.1}% of tiles have fewer than 10 features",
                undersized_pct
            ),
            suggestion: "Lower --max-zoom or use a coarser --temporal-bucket so tiles \
                         aggregate more features"
                .to_string(),
        });
    }

    // Check for very high tile count
    if total_tile_count > 50_000 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!(
                "High tile count ({}) may impact loading performance",
                total_tile_count
            ),
            suggestion: "Narrow the zoom range (--min-zoom / --max-zoom) or use a coarser \
                         --temporal-bucket"
                .to_string(),
        });
    }

    // Check for sparse data at high zooms
    if let Some(z_max) = spatial
        .zoom_coverage
        .iter()
        .find(|z| z.zoom == spatial.recommended_max_zoom)
    {
        if z_max.coverage_percent < 0.1 {
            issues.push(DensityIssue {
                severity: IssueSeverity::Info,
                description: format!(
                    "Only {:.2}% coverage at zoom {}",
                    z_max.coverage_percent, spatial.recommended_max_zoom
                ),
                suggestion: "Data is sparse at this zoom level; lower --max-zoom".to_string(),
            });
        }
    }

    // Check estimated archive size
    let size_mb = estimated_archive_size as f64 / 1_048_576.0;
    if size_mb > 500.0 {
        issues.push(DensityIssue {
            severity: IssueSeverity::Warning,
            description: format!("Large estimated archive size ({:.1} MB)", size_mb),
            suggestion: "Lower --max-zoom, or opt into per-tile budgets \
                         (--maximum-tile-bytes / --maximum-tile-features, optionally \
                         --drop-densest-as-needed) which drop features to fit (data loss). \
                         For very dense point sets, --summary-tier bakes aggregate overview \
                         tiles instead of full-resolution features"
                .to_string(),
        });
    }

    // Check for hotspot concentration
    if !spatial.hotspots.is_empty() {
        let top_hotspot = &spatial.hotspots[0];
        let hotspot_pct = top_hotspot.feature_count as f64 / data.features.len() as f64 * 100.0;
        if hotspot_pct > 50.0 {
            issues.push(DensityIssue {
                severity: IssueSeverity::Info,
                description: format!(
                    "{:.1}% of features concentrated in {}",
                    hotspot_pct,
                    top_hotspot.name.as_deref().unwrap_or("one region")
                ),
                suggestion: "Hotspot tiles will be large; opt-in per-tile budgets \
                             (--maximum-tile-bytes / --maximum-tile-features, which drop \
                             features to fit — data loss) cap them, or a per-feature \
                             --min-zoom-field keeps coarse zooms light by holding minor \
                             features back to deeper zooms"
                    .to_string(),
            });
        }
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{AnalyzableFeature, GeometryType};
    use stt_core::types::{BoundingBox, TimeRange};

    fn feature(lon: f64, lat: f64, timestamp: u64) -> AnalyzableFeature {
        AnalyzableFeature {
            lon,
            lat,
            timestamp,
            geometry_type: GeometryType::Point,
            vertex_count: 1,
            estimated_size: 150,
            property_count: 2,
        }
    }

    /// Build synthetic data spread over a small region with timestamps spread
    /// over ~n² seconds.
    fn make_grid_data(n_side: usize) -> LoadedData {
        let mut features = Vec::new();
        let mut min_lon = f64::MAX;
        let mut max_lon = f64::MIN;
        let mut min_lat = f64::MAX;
        let mut max_lat = f64::MIN;
        for i in 0..n_side {
            for j in 0..n_side {
                let lon = -100.0 + (i as f64) * 0.05;
                let lat = 40.0 + (j as f64) * 0.05;
                min_lon = min_lon.min(lon);
                max_lon = max_lon.max(lon);
                min_lat = min_lat.min(lat);
                max_lat = max_lat.max(lat);
                features.push(feature(lon, lat, (i * n_side + j) as u64 * 1000));
            }
        }
        LoadedData {
            features,
            bounds: BoundingBox::new(min_lon, min_lat, max_lon, max_lat),
            time_range: TimeRange::new(0, 1_000_000),
            sample: Vec::new(),
        }
    }

    #[test]
    fn test_bucket_zoom_splits_by_temporal_bucket() {
        // 100 features at ONE location spread over 100s: with a 10s bucket the
        // single spatial tile must split into 10 (x, y, t) tiles; with no
        // bucketing (bucket_ms = 0) it stays a single tile.
        let features: Vec<_> = (0..100u64)
            .map(|i| feature(-100.0, 40.0, i * 1_000))
            .collect();
        let data = LoadedData {
            features,
            bounds: BoundingBox::new(-100.0, 40.0, -100.0, 40.0),
            time_range: TimeRange::new(0, 100_000),
            sample: Vec::new(),
        };

        let bucketed = bucket_zoom(&data, 10, 10_000, None, None);
        assert_eq!(bucketed.tile_count, 10);
        assert_eq!(bucketed.max_features_per_tile, 10);
        assert_eq!(bucketed.estimated_size_uncompressed, 100 * 150);

        let unbucketed = bucket_zoom(&data, 10, 0, None, None);
        assert_eq!(unbucketed.tile_count, 1);
        assert_eq!(unbucketed.max_features_per_tile, 100);
    }

    #[test]
    fn test_measured_calibration_replaces_formula() {
        // A measured sample encoding must drive both size estimates (real
        // bytes/feature and zstd ratio), replacing the formula + /3 fallback.
        let features: Vec<_> = (0..100u64)
            .map(|i| feature(-100.0, 40.0, i * 1_000))
            .collect();
        let data = LoadedData {
            features,
            bounds: BoundingBox::new(-100.0, 40.0, -100.0, 40.0),
            time_range: TimeRange::new(0, 100_000),
            sample: Vec::new(),
        };
        let measured = MeasuredEncoding {
            features: 100,
            geometry_kind: "point".to_string(),
            bytes_total: 4_200,
            bytes_per_feature: 42.0,
            zstd_ratio: 2.0,
            tiles: 1,
            per_column: Vec::new(),
        };

        let calibrated = bucket_zoom(&data, 10, 0, Some(&measured), None);
        assert_eq!(calibrated.estimated_size_compressed, 100 * 42);
        assert_eq!(calibrated.estimated_size_uncompressed, 100 * 42 * 2);

        // The no-measurement fallback keeps the formula estimates + the named
        // rough compression ratio (no bare magic constant).
        let fallback = bucket_zoom(&data, 10, 0, None, None);
        assert_eq!(fallback.estimated_size_uncompressed, 100 * 150);
        assert_eq!(
            fallback.estimated_size_compressed,
            (100.0 * 150.0 / FALLBACK_ZSTD_RATIO).round() as usize,
        );
    }

    #[test]
    fn test_analyze_aggregates_across_zoom_range() {
        // End-to-end: analyze() must produce one ZoomDensity per zoom in the
        // recommended range, and the aggregates must sum the per-zoom stats.
        let data = make_grid_data(20); // 400 points
        let spatial = crate::analysis::spatial::analyze(&data).unwrap();
        let temporal = crate::analysis::temporal::analyze(&data).unwrap();
        let density = analyze(&data, &spatial, &temporal, None).unwrap();

        let expected_zooms = (spatial.recommended_min_zoom..=spatial.recommended_max_zoom).count();
        assert_eq!(density.per_zoom.len(), expected_zooms);
        assert_eq!(
            density.estimated_tile_count,
            density.per_zoom.iter().map(|z| z.tile_count).sum::<usize>()
        );
        assert!(density.estimated_archive_size > 0);
        assert!(density.per_zoom.iter().all(|z| z.tile_count > 0));
        // Point data: a deeper zoom can only split (x, y, t) tiles, never merge
        // them, so tile counts are non-decreasing across the range.
        for pair in density.per_zoom.windows(2) {
            assert!(
                pair[1].tile_count >= pair[0].tile_count,
                "z{} tile_count {} < z{} tile_count {}",
                pair[1].zoom,
                pair[1].tile_count,
                pair[0].zoom,
                pair[0].tile_count
            );
        }
    }

    #[test]
    fn test_oversized_issue_names_real_build_flags() {
        // >10k features in one (x, y, t) tile must yield an oversized warning
        // whose suggestion names real stt-build flags.
        let features: Vec<_> = (0..10_001).map(|_| feature(-100.0, 40.0, 0)).collect();
        let data = LoadedData {
            features,
            bounds: BoundingBox::new(-100.0, 40.0, -100.0, 40.0),
            time_range: TimeRange::new(0, 0),
            sample: Vec::new(),
        };
        let spatial = crate::analysis::spatial::analyze(&data).unwrap();
        let temporal = crate::analysis::temporal::analyze(&data).unwrap();
        let density = analyze(&data, &spatial, &temporal, None).unwrap();

        let oversized: usize = density.per_zoom.iter().map(|z| z.oversized_tiles).sum();
        assert!(oversized > 0, "expected oversized tiles");
        let issue = density
            .issues
            .iter()
            .find(|i| i.description.contains("10,000"))
            .expect("oversized issue present");
        assert!(issue.suggestion.contains("--maximum-tile-bytes"));
        assert!(issue.suggestion.contains("--maximum-tile-features"));
        assert!(issue.suggestion.contains("--temporal-bucket"));
    }

    // ==================================================================
    // The tile term and clip replication
    // ==================================================================

    use crate::loader::{PropValue, SampledFeature};
    use geo_types::{Coord as GeoCoord, Geometry, LineString, Point as GeoPoint, Polygon};

    /// Deterministic pseudo-noise in `[0, 1)` — high-entropy f64 mantissas with
    /// no RNG, so every fixture is byte-identical on every machine.
    fn noise(i: usize, salt: u64) -> f64 {
        ((i as u64).wrapping_add(salt).wrapping_mul(2_654_435_761) % 1_000_000) as f64 / 1e6
    }

    /// `n` scattered points, each with a high-entropy float and a repeating
    /// categorical — the shape the probe measures on real point datasets.
    fn point_sample(n: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| SampledFeature {
                geometry: Geometry::Point(GeoPoint::new(
                    -100.0 + (i as f64) * 0.017 % 30.0 + noise(i, 3) * 1e-3,
                    35.0 + (i % 23) as f64 * 0.31 + noise(i, 91) * 1e-3,
                )),
                timestamp_ms: 1_600_000_000_000 + i as u64 * 60_000,
                properties: vec![
                    (
                        "magnitude".to_string(),
                        PropValue::Number(1.0 + noise(i, 7) * 9.0),
                    ),
                    (
                        "region".to_string(),
                        PropValue::Text(format!("region-{}", i % 5)),
                    ),
                ],
            })
            .collect()
    }

    /// A `LoadedData` whose `features` are the centroids of `sample`, exactly as
    /// the loader derives them.
    fn loaded(sample: Vec<SampledFeature>) -> LoadedData {
        let features: Vec<AnalyzableFeature> = sample
            .iter()
            .map(|f| {
                let (lon, lat) = centroid_of(&f.geometry).unwrap();
                AnalyzableFeature {
                    lon,
                    lat,
                    timestamp: f.timestamp_ms,
                    geometry_type: GeometryType::Point,
                    vertex_count: 1,
                    estimated_size: 120,
                    property_count: f.properties.len(),
                }
            })
            .collect();
        let (mut min_lon, mut min_lat) = (f64::MAX, f64::MAX);
        let (mut max_lon, mut max_lat) = (f64::MIN, f64::MIN);
        let (mut t0, mut t1) = (u64::MAX, 0u64);
        for f in &features {
            min_lon = min_lon.min(f.lon);
            max_lon = max_lon.max(f.lon);
            min_lat = min_lat.min(f.lat);
            max_lat = max_lat.max(f.lat);
            t0 = t0.min(f.timestamp);
            t1 = t1.max(f.timestamp);
        }
        LoadedData {
            features,
            bounds: BoundingBox::new(min_lon, min_lat, max_lon, max_lat),
            time_range: TimeRange::new(t0, t1),
            sample,
        }
    }

    // ------------------------------------------------------------------
    // THE PIN: projected bytes vs bytes the real encoder actually emits
    // ------------------------------------------------------------------

    #[test]
    fn the_projection_agrees_with_what_the_real_encoder_emits_at_real_occupancy() {
        // The defect this pins: `Ŝ = features × bytes_per_feature` has no tile
        // term, so on a thin-tiled archive it under-projects by ~10×. Here the
        // REALIZED side is not a stand-in — it is the production encoder and the
        // production compressor run over every tile of the cut, summed. The
        // projection may not drift from it by more than a stated tolerance.
        //
        // (An end-to-end pin against a real `stt-build` archive cannot live in
        // this crate: `stt-build` DEPENDS on `stt-optimize`, so the dependency
        // cannot be reversed even for a dev-dependency. The realized numbers
        // from two real builds are recorded in the module docs and in the
        // campaign report.)
        let sample = point_sample(2_048);
        let level = MeasureSettings::default().zstd_level;
        let model = measure_size_model(&sample, level, 1.0)
            .unwrap()
            .expect("2048 features clears the probe floor");

        let (_, subset) = dominant_subset(&sample).unwrap();
        // Three real cuts, thin to fat, all far past the probe's own 256-tile
        // finest chord so the assertion is about EXTRAPOLATION, not about
        // re-reading a measurement.
        for tiles in [1_024usize, 512, 128] {
            let realized = encode_probe(&subset, ProbeKind::Point, tiles, level).unwrap() as f64;
            let projected =
                model.project(tiles as u64, subset.len() as u64, subset.len() as u64) as f64;
            let error = (projected - realized).abs() / realized;
            assert!(
                error <= 0.25,
                "at {tiles} tiles the projection is {projected:.0} B against {realized:.0} B \
                 realized ({:+.1}%), outside the ±25% the two-term model is pinned to. Model: \
                 {model:?}",
                (projected / realized - 1.0) * 100.0
            );
            // …and the bar has to cover the miss it actually made. A tight bar
            // over a wrong number is the failure mode this whole change exists
            // to remove: the incumbent reported ±0.94% on a projection the real
            // build missed by 7.4×.
            let bar = model.projection_stderr(tiles as u64, projected as u64);
            assert!(
                (projected - realized).abs() <= 2.0 * bar.max(1.0),
                "the ±{bar:.0} B bar does not cover a {:.0} B miss at {tiles} tiles: {model:?}",
                (projected - realized).abs()
            );
        }
    }

    #[test]
    fn the_tile_term_is_what_makes_the_projection_track_reality() {
        // The counterfactual, on the same data: the OLD one-term oracle
        // (`instances × bytes_per_feature`, no tile term) has to be dramatically
        // low at real occupancy, or the tile term this change adds would be
        // pointless.
        let sample = point_sample(2_048);
        let level = MeasureSettings::default().zstd_level;
        let model = measure_size_model(&sample, level, 1.0).unwrap().unwrap();
        let (_, subset) = dominant_subset(&sample).unwrap();

        let tiles = 1_024usize;
        let realized = encode_probe(&subset, ProbeKind::Point, tiles, level).unwrap() as f64;
        let one_term = model.bytes_per_feature * subset.len() as f64;
        let two_term = model.project(tiles as u64, subset.len() as u64, subset.len() as u64) as f64;

        assert!(
            one_term < realized * 0.6,
            "the tile-blind estimate {one_term:.0} B should badly under-project {realized:.0} B \
             at {:.2} features/tile",
            subset.len() as f64 / tiles as f64
        );
        assert!(
            (two_term - realized).abs() < (one_term - realized).abs(),
            "the two-term projection {two_term:.0} B must beat the one-term {one_term:.0} B \
             against {realized:.0} B realized"
        );
        assert!(model.bytes_per_tile > 0.0, "{model:?}");
    }

    #[test]
    fn the_probe_is_deterministic_and_declines_a_sample_it_cannot_measure() {
        let sample = point_sample(512);
        let level = MeasureSettings::default().zstd_level;
        let a = measure_size_model(&sample, level, 0.9).unwrap().unwrap();
        let b = measure_size_model(&sample, level, 0.9).unwrap().unwrap();
        assert_eq!(a, b, "the probe must be a pure function of its inputs");
        assert_eq!(a.probe_tiles, 64);
        assert_eq!(a.probe_features, 512);
        assert_eq!(a.zstd_level, level);

        // Below the floor there is no model — not a fabricated one.
        assert!(measure_size_model(&point_sample(32), level, 1.0)
            .unwrap()
            .is_none());
        assert!(measure_size_model(&[], level, 1.0).unwrap().is_none());
    }

    #[test]
    fn the_content_charge_counts_clip_replication_by_its_measured_row_share() {
        // Geometry is conserved when the clipper cuts a feature (vertices are
        // redistributed); ids, times and properties are re-emitted per tiled
        // instance. So a replicated dataset must cost MORE than an unreplicated
        // one, in proportion to the measured non-geometry share — and nothing at
        // all when the share says the bytes are all geometry.
        let model = SizeModel {
            bytes_per_tile: 100.0,
            bytes_per_tile_stderr: 0.0,
            bytes_per_feature: 10.0,
            fit_residual: 0.0,
            geometry_share: 0.5,
            probe_tiles: 64,
            probe_features: 512,
            zstd_level: 3,
        };
        // 100 source features, 300 tiled instances: content = 10 × (0.5·100 +
        // 0.5·300) = 2000, framing = 100 × (50 − 1) = 4900.
        assert_eq!(model.project(50, 300, 100), 2_000 + 4_900);
        // No replication → content = 10 × 100 = 1000.
        assert_eq!(model.project(50, 100, 100), 1_000 + 4_900);

        let all_geometry = SizeModel {
            geometry_share: 1.0,
            ..model
        };
        assert_eq!(
            all_geometry.project(50, 300, 100),
            all_geometry.project(50, 100, 100),
            "with every byte in geometry, clipping redistributes rather than duplicates"
        );
    }

    // ------------------------------------------------------------------
    // Clip replication
    // ------------------------------------------------------------------

    #[test]
    fn a_trajectory_is_counted_in_every_tile_it_crosses() {
        // The centroid scan sees one cell. The builder clips the line and emits
        // a piece in each tile it enters, which is what `sample_replication`
        // has to reproduce. At z6 one tile spans 5.625° of longitude, so a line
        // from -100° to -60° along a constant latitude crosses 8 columns.
        let line = Geometry::LineString(LineString(vec![
            GeoCoord { x: -100.0, y: 40.0 },
            GeoCoord { x: -60.0, y: 40.0 },
        ]));
        let mut cells = Vec::new();
        covered_cells(&line, 6, &mut cells);
        assert_eq!(cells.len(), 8, "{cells:?}");
        // Every cell is in the same row, consecutive columns — a supercover
        // walk, not a bbox fill (which would be identical here) and not two
        // endpoints (which would be 2).
        assert!(cells.windows(2).all(|w| w[0].1 == w[1].1));
        assert!(cells.windows(2).all(|w| w[1].0 == w[0].0 + 1));

        // A DIAGONAL is where a bbox fill and a supercover walk diverge: the
        // walk visits the staircase, the fill would visit the whole rectangle.
        let diagonal = Geometry::LineString(LineString(vec![
            GeoCoord { x: -100.0, y: 20.0 },
            GeoCoord { x: -60.0, y: 60.0 },
        ]));
        let mut diag_cells = Vec::new();
        covered_cells(&diagonal, 6, &mut diag_cells);
        let cols = diag_cells.iter().map(|c| c.0).collect::<HashSet<_>>().len();
        let rows = diag_cells.iter().map(|c| c.1).collect::<HashSet<_>>().len();
        assert!(
            diag_cells.len() < cols * rows,
            "a supercover walk must visit fewer cells than the {cols}×{rows} bbox: {diag_cells:?}"
        );
        assert!(diag_cells.len() >= cols.max(rows));
    }

    #[test]
    fn point_data_measures_no_replication_at_all() {
        // The regression pin for every existing point fixture: a point-only
        // sample must return the identity WITHOUT scanning, so nothing about a
        // point dataset's tile counts or size estimates moved.
        let sample = point_sample(200);
        let replication = sample_replication(&sample, 10, 60_000);
        assert_eq!(replication, Replication::none());
        assert!(replication.is_none());
        // No sample at all is also the identity — which is what keeps every
        // synthetic `LoadedData { sample: Vec::new(), .. }` fixture unchanged.
        assert_eq!(sample_replication(&[], 10, 60_000), Replication::none());
    }

    #[test]
    fn replication_inflates_the_tile_count_and_the_instance_count_together() {
        // A dataset of horizontal lines each crossing several z6 columns: the
        // centroid scan files one cell per line, the replication-aware scan has
        // to see the crossings in BOTH the tile count and the instance count.
        let sample: Vec<SampledFeature> = (0..64)
            .map(|i| SampledFeature {
                geometry: Geometry::LineString(LineString(vec![
                    GeoCoord {
                        x: -100.0,
                        y: 30.0 + i as f64 * 0.4,
                    },
                    GeoCoord {
                        x: -70.0,
                        y: 30.0 + i as f64 * 0.4,
                    },
                ])),
                timestamp_ms: 1_600_000_000_000 + i as u64 * 60_000,
                properties: vec![("kind".to_string(), PropValue::Text("front".to_string()))],
            })
            .collect();
        let data = loaded(sample);

        let plain = sample_replication(&[], 6, 0);
        let measured = sample_replication(&data.sample, 6, 0);
        assert!(plain.is_none());
        assert!(
            measured.instances_per_feature > 5.0,
            "a 30°-wide line at z6 spans ~6 columns: {measured:?}"
        );
        assert!(measured.cell_inflation > 1.0, "{measured:?}");
        assert!(
            measured.cell_inflation <= measured.instances_per_feature,
            "cells can never outnumber instances: {measured:?}"
        );

        // Against the SAME features with the sample removed — i.e. the pure
        // centroid scan, which is what this file did before — both counts must
        // grow, and the tile count must stay inside its structural bounds.
        let centroid_only = LoadedData {
            features: data.features.clone(),
            bounds: data.bounds,
            time_range: data.time_range,
            sample: Vec::new(),
        };
        let plain_row = bucket_zoom(&centroid_only, 6, 0, None, None);
        let row = bucket_zoom(&data, 6, 0, None, None);
        assert!(
            row.tile_count > plain_row.tile_count,
            "coverage must add cells the centroid scan cannot see: {} vs {}",
            row.tile_count,
            plain_row.tile_count
        );
        assert!(
            row.tile_count <= row.tiled_instances(),
            "a cell cannot hold fewer than one instance: {row:?}"
        );
        assert_eq!(
            row.tiled_instances(),
            scale_count(data.features.len() as u64, measured.instances_per_feature) as usize
        );
        assert!(
            row.tiled_instances() > data.features.len(),
            "clipping must produce more instances than input features: {row:?}"
        );
        assert_eq!(plain_row.tiled_instances(), data.features.len());
        // The published average is the replication-aware one, and the derived
        // instance count round-trips through it.
        assert!(
            (row.avg_features_per_tile - row.tiled_instances() as f64 / row.tile_count as f64)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn a_polygon_covers_its_bounding_box_the_way_the_clipper_sweeps_it() {
        // `place_polygon` clips the rings against every tile the bbox covers and
        // keeps whatever survives, so a bbox sweep is the matching (upper-bound)
        // count here.
        let square = Geometry::Polygon(Polygon::new(
            LineString(vec![
                GeoCoord { x: -100.0, y: 40.0 },
                GeoCoord { x: -80.0, y: 40.0 },
                GeoCoord { x: -80.0, y: 50.0 },
                GeoCoord { x: -100.0, y: 50.0 },
                GeoCoord { x: -100.0, y: 40.0 },
            ]),
            vec![],
        ));
        let mut cells = Vec::new();
        covered_cells(&square, 5, &mut cells);
        let cols = cells.iter().map(|c| c.0).collect::<HashSet<_>>().len();
        let rows = cells.iter().map(|c| c.1).collect::<HashSet<_>>().len();
        assert_eq!(cells.len(), cols * rows, "a bbox sweep fills the rectangle");
        assert!(cells.len() > 1);
    }

    #[test]
    fn the_zoom_row_carries_a_tile_term_only_when_a_model_was_measured() {
        // Both estimate paths on identical data, so the difference IS the tile
        // term. The fallback keeps the historical shape and is therefore the
        // rollback, not a second opinion.
        let data = loaded(point_sample(600));
        let measured = MeasuredEncoding {
            features: 600,
            geometry_kind: "point".to_string(),
            bytes_total: 12_000,
            bytes_per_feature: 20.0,
            zstd_ratio: 3.0,
            tiles: 8,
            per_column: Vec::new(),
        };
        let model = SizeModel {
            bytes_per_tile: 300.0,
            bytes_per_tile_stderr: 25.0,
            bytes_per_feature: 20.0,
            fit_residual: 0.0,
            geometry_share: 1.0,
            probe_tiles: 64,
            probe_features: 600,
            zstd_level: 3,
        };

        let without = bucket_zoom(&data, 12, 60_000, Some(&measured), None);
        let with = bucket_zoom(&data, 12, 60_000, Some(&measured), Some(&model));
        assert_eq!(without.tile_count, with.tile_count);
        assert_eq!(
            without.estimated_size_compressed,
            with.tiled_instances() * 20
        );
        assert_eq!(
            with.estimated_size_compressed,
            with.tiled_instances() * 20 + 300 * (with.tile_count - 1)
        );
        assert!(
            with.estimated_size_compressed > without.estimated_size_compressed * 2,
            "on {} tiles for {} features the tile term must dominate: {} vs {}",
            with.tile_count,
            with.tiled_instances(),
            with.estimated_size_compressed,
            without.estimated_size_compressed
        );
    }

    #[test]
    fn an_antimeridian_crossing_geometry_is_not_swept_the_wrong_way_round() {
        // The builder SPLITS a dateline-crossing line or ring at ±180 and clips
        // the pieces, so they land at both edges of the world. Walking the edge
        // as drawn would instead sweep every column in between — 2^zoom tiles
        // the build never emits, at 2^zoom iterations of the DDA. Both guards
        // use the same per-edge |Δlon| > 180° test the splitter does.
        let crossing = Geometry::LineString(LineString(vec![
            GeoCoord { x: 179.0, y: 10.0 },
            GeoCoord { x: -179.0, y: 10.0 },
        ]));
        let mut cells = Vec::new();
        covered_cells(&crossing, 10, &mut cells);
        assert_eq!(cells.len(), 2, "endpoint cells only: {cells:?}");
        // The columns are at opposite edges of the world, not adjacent.
        assert!(cells[0].0 < 8 && cells[1].0 > (1u32 << 10) - 8, "{cells:?}");

        let crossing_ring = Geometry::Polygon(Polygon::new(
            LineString(vec![
                GeoCoord { x: 179.0, y: 10.0 },
                GeoCoord { x: -179.0, y: 10.0 },
                GeoCoord { x: -179.0, y: 12.0 },
                GeoCoord { x: 179.0, y: 12.0 },
                GeoCoord { x: 179.0, y: 10.0 },
            ]),
            vec![],
        ));
        let mut ring_cells = Vec::new();
        covered_cells(&crossing_ring, 10, &mut ring_cells);
        assert!(
            ring_cells.len() <= 4,
            "a split ring claims its vertex cells, not the whole world: {}",
            ring_cells.len()
        );

        // A NON-crossing ring that merely spans a wide longitude range still
        // sweeps normally — the test is per-EDGE, not on the bbox width, which
        // is the same distinction `place_polygon` draws: "a polygon that merely
        // spans a wide (but < 360°) longitude range WITHOUT any wrapping edge
        // genuinely occupies those columns". Intermediate vertices keep every
        // edge under 180°.
        let wide = Geometry::Polygon(Polygon::new(
            LineString(vec![
                GeoCoord { x: -170.0, y: 10.0 },
                GeoCoord { x: 0.0, y: 10.0 },
                GeoCoord { x: 170.0, y: 10.0 },
                GeoCoord { x: 170.0, y: 12.0 },
                GeoCoord { x: 0.0, y: 12.0 },
                GeoCoord { x: -170.0, y: 12.0 },
                GeoCoord { x: -170.0, y: 10.0 },
            ]),
            vec![],
        ));
        let mut wide_cells = Vec::new();
        covered_cells(&wide, 4, &mut wide_cells);
        assert!(
            wide_cells.len() > 8,
            "a genuinely wide ring occupies its columns: {}",
            wide_cells.len()
        );
    }

    #[test]
    fn analyze_probes_a_model_only_when_it_was_handed_a_measurement() {
        // The occupancy pre-pass (`measured: None`) must not pay for encodes,
        // and the calibrated pass must produce an estimate with the tile term.
        let data = loaded(point_sample(600));
        let spatial = crate::analysis::spatial::analyze(&data).unwrap();
        let temporal = crate::analysis::temporal::analyze(&data).unwrap();

        let unmeasured = analyze(&data, &spatial, &temporal, None).unwrap();
        let layout = crate::measure::SyntheticLayout::from_density(&unmeasured);
        let measured = crate::measure::measure_sample_layout(
            &data.sample,
            &MeasureSettings::default(),
            &layout,
        )
        .unwrap()
        .unwrap();
        let calibrated = analyze(&data, &spatial, &temporal, Some(&measured)).unwrap();

        // Occupancy is identical between the passes — the layout contract MO-4
        // depends on ("the layout is a pure function of the occupancy scan").
        assert_eq!(
            unmeasured.estimated_tile_count,
            calibrated.estimated_tile_count
        );
        assert_eq!(
            layout,
            crate::measure::SyntheticLayout::from_density(&calibrated)
        );
        assert_eq!(
            calibrated.estimated_tiled_instances(),
            calibrated
                .per_zoom
                .iter()
                .map(|z| z.tiled_instances())
                .sum::<usize>()
        );
        // The calibrated estimate carries the tile term; the tile-blind rate
        // over the same instances is strictly smaller.
        let tile_blind = calibrated.estimated_tiled_instances() as f64 * measured.bytes_per_feature;
        assert!(
            calibrated.estimated_archive_size as f64 > tile_blind,
            "calibrated {} must exceed the tile-blind {tile_blind:.0}",
            calibrated.estimated_archive_size
        );
        // Deterministic across runs.
        let again = analyze(&data, &spatial, &temporal, Some(&measured)).unwrap();
        assert_eq!(
            again.estimated_archive_size,
            calibrated.estimated_archive_size
        );
    }
}
