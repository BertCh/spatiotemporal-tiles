//! Sample-encode measurement: push a deterministic sample of source features
//! through the real stt-core encoder (+ zstd) to replace formula-based size
//! estimates with measured bytes.
//!
//! The loader retains a stride sample of full geometries and property values
//! ([`crate::loader::SampledFeature`]); [`measure_sample_layout`] groups the
//! dominant geometry kind into a [`SyntheticLayout`] of synthetic tiles whose
//! sizes come from the OBSERVED per-`(x, y, t)` occupancy distribution, encodes
//! each through [`stt_core::arrow_tile::encode_tile_with`] (the production
//! encoder, driven by an explicit [`EncoderConfig`] so no process-globals are
//! touched), zstd-compresses each payload, and attributes per-column costs
//! through the shared [`attribution`] module.
//!
//! Why the layout matters: packing the whole 5000-feature sample into ONE
//! synthetic tile amortises the per-tile encoder + zstd framing overhead over
//! the entire sample instead of over a real tile's worth of features, so
//! bytes/feature comes out biased LOW. [`measure_sample`] is the single-tile
//! wrapper that preserves the old behaviour (and is the rollback).

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use stt_core::arrow_tile::{
    decode_tile, encode_tile_with, ColumnarLayer, Coord, EncoderConfig, GeometryColumn,
    PropertyColumn,
};
use stt_core::compression::compress_zstd_with_dict_level;

use crate::analysis::density::DensityAnalysis;
use crate::loader::{PropValue, SampledFeature};

// `attribution` is a top-level module (`pub mod attribution;` in `lib.rs`);
// MO-2/MO-3 could not declare it there, so it lived here behind a `#[path]`
// attribute until MO-5 — the next item to own `lib.rs` — moved the declaration
// up as its authors asked. These re-exports keep `crate::measure::attribution::*`
// resolving for every existing caller.
pub use crate::attribution::{
    self, attribute_columns, attribute_columns_loo, attribute_columns_singleton, AttributionDesign,
    ColumnBytes, LooCost, ShareDispersion,
};

/// Below this many usable sampled features the measurement is noise (zstd
/// framing and per-tile encoder overheads dominate the per-feature figure) and
/// [`measure_sample`] returns `None` — callers fall back to the formula
/// estimates. Set to 24: low enough that most real datasets get MEASURED bytes
/// (a real encode of two-dozen homogeneous features estimates bytes/feature and
/// the zstd ratio well within estimate tolerance), high enough that fixed frame
/// overhead doesn't materially skew the per-feature figure.
///
/// This applies PER MEASUREMENT (the whole sample subset), not per synthetic
/// tile — the per-tile floor is [`MIN_TILE_FEATURES`].
const MIN_MEASURE_FEATURES: usize = 24;

/// Floor on features per synthetic tile. Below it a "tile" is mostly frame
/// overhead and the split stops telling you anything about real occupancy, so
/// the layout is collapsed to fewer tiles rather than measuring noise.
const MIN_TILE_FEATURES: usize = 8;

/// Synthetic tiles a density-derived layout uses. Eight evenly-spaced quantiles
/// of the occupancy distribution: enough to carry the shape (a light tile and a
/// busy tile pay different framing overheads per feature), few enough that each
/// tile clears [`MIN_TILE_FEATURES`] on the loader's 5000-feature sample.
const LAYOUT_TILES: usize = 8;

/// How the sample is cut into synthetic tiles before it is encoded.
///
/// `tile_sizes` are RELATIVE weights, not absolute counts: a layout describes
/// the *shape* of real tile occupancy, and [`measure_sample_layout`] scales it
/// onto whatever sample it is handed. That is what makes one layout reusable
/// across a baseline and every trial.
///
/// ⚠️ INVARIANT: one layout per analysis run, shared by the baseline AND every
/// trial. A trial measured under a different layout confounds the lever with
/// the layout and its delta is meaningless. The layout is a pure deterministic
/// function of the density occupancy scan
/// ([`SyntheticLayout::from_density`]), and that scan does not depend on any
/// measurement — so any holder of an
/// [`AnalysisResult`](crate::analysis::AnalysisResult) can reconstruct the
/// exact layout its `measured` field was produced under with
/// `SyntheticLayout::from_density(&result.density)`. That reconstructability
/// IS the structural enforcement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticLayout {
    /// Relative sizes of the synthetic tiles, ascending.
    pub tile_sizes: Vec<usize>,
}

impl Default for SyntheticLayout {
    fn default() -> Self {
        Self::single()
    }
}

impl SyntheticLayout {
    /// The one-tile layout: the pre-MO-4 behaviour, and the rollback.
    pub fn single() -> Self {
        Self {
            tile_sizes: vec![1],
        }
    }

    /// Build a layout from an observed per-tile feature-count distribution.
    ///
    /// `counts` need not be sorted; the layout samples it at
    /// [`LAYOUT_TILES`] evenly-spaced quantiles `q_i = (i + 0.5)/T`, i.e. each
    /// synthetic tile stands for one equal slice of the tile population. Pure
    /// integer indexing into the sorted counts — no interpolation, no RNG.
    pub fn from_feature_counts(counts: &[usize]) -> Self {
        let mut sorted: Vec<usize> = counts.iter().copied().filter(|&c| c > 0).collect();
        if sorted.is_empty() {
            return Self::single();
        }
        sorted.sort_unstable();
        let len = sorted.len();
        let tile_sizes = (0..LAYOUT_TILES)
            .map(|i| {
                // idx = floor(q_i · len) with q_i = (2i+1)/(2T), in integers.
                let idx = (((2 * i + 1) * len) / (2 * LAYOUT_TILES)).min(len - 1);
                sorted[idx].max(1)
            })
            .collect();
        Self { tile_sizes }
    }

    /// Build the layout for an analysis run from its density occupancy scan.
    ///
    /// Reads ONLY occupancy statistics (`median` / `p95` / `max` features per
    /// tile at the deepest analysed zoom) — never a size estimate — so the
    /// layout is identical whether it is derived from the pre-measurement
    /// occupancy pass or from the final calibrated
    /// [`DensityAnalysis`]. That is what makes it reproducible by any later
    /// consumer holding the `AnalysisResult`.
    ///
    /// The quantile function is piecewise linear through the published
    /// anchors `Q(0)=1`, `Q(0.5)=median`, `Q(0.95)=p95`, `Q(1)=max`, sampled at
    /// the same `(i + 0.5)/T` fractions [`Self::from_feature_counts`] uses. At
    /// `T = 8` the deepest fraction is `0.9375`, so a single outlier cell never
    /// sets a synthetic tile size.
    pub fn from_density(density: &DensityAnalysis) -> Self {
        let Some(deepest) = density.per_zoom.iter().max_by_key(|z| z.zoom) else {
            return Self::single();
        };
        if deepest.tile_count == 0 {
            return Self::single();
        }
        // Monotone anchors: a producer that reports p95 > max (or median > p95)
        // must not invert the ladder.
        let median = deepest.median_features_per_tile.max(1) as f64;
        let p95 = (deepest.p95_features_per_tile.max(1) as f64).max(median);
        let max = (deepest.max_features_per_tile.max(1) as f64).max(p95);
        let tile_sizes = (0..LAYOUT_TILES)
            .map(|i| {
                let q = (2 * i + 1) as f64 / (2 * LAYOUT_TILES) as f64;
                let v = if q <= 0.5 {
                    1.0 + (median - 1.0) * (q / 0.5)
                } else if q <= 0.95 {
                    median + (p95 - median) * ((q - 0.5) / 0.45)
                } else {
                    p95 + (max - p95) * ((q - 0.95) / 0.05)
                };
                (v.round() as usize).max(1)
            })
            .collect();
        Self { tile_sizes }
    }

    /// Number of synthetic tiles this layout asks for.
    pub fn tiles(&self) -> usize {
        self.tile_sizes.len().max(1)
    }

    /// How many synthetic tiles this layout actually resolves to when it is
    /// scaled onto `usable_features` (the dominant-kind subset size).
    ///
    /// [`Self::tiles`] is what the layout ASKS for; this is what it GETS, after
    /// the [`MIN_TILE_FEATURES`] floor thins it. The two differ only on samples
    /// too small to fill every tile.
    pub fn resolved_tiles(&self, usable_features: usize) -> usize {
        layout_weights(self, usable_features).len()
    }

    /// Whether `measured` could have been produced under this layout.
    ///
    /// The MO-4 invariant — *one layout per analysis run, shared by the
    /// baseline and every trial* — is only worth stating if it can be CHECKED,
    /// and a [`MeasuredEncoding`] does not carry the layout it was cut under.
    /// It does carry the two numbers that pin the cut: the usable feature count
    /// and the resulting tile count, and [`Self::resolved_tiles`] is the
    /// function between them. So a measurement taken under a different cut is
    /// detectable, and a consumer handed one can re-measure instead of
    /// comparing a trial against a baseline that was never comparable.
    ///
    /// That check is not academic. A layout baseline against a single-tile
    /// trial reads as a ~32% shrink out of thin air (one synthetic tile
    /// amortises per-tile framing over the whole sample and so under-charges by
    /// a measured 1.47× on an 800-feature point sample) — which on the
    /// quantization advisors would fire BOTH lossy levers, at High confidence,
    /// on every dataset. See [`crate::advisors::quantize`].
    pub fn produced(&self, measured: &MeasuredEncoding) -> bool {
        measured.tiles == self.resolved_tiles(measured.features)
    }
}

/// The build-default baseline for an analysis run, guaranteed to have been
/// measured under `layout`.
///
/// `measured` is the run's own [`MeasuredEncoding`] when it has one
/// (`AnalysisResult::measured`); it is reused as-is only when
/// [`SyntheticLayout::produced`] confirms it was cut the same way, and
/// re-measured under `layout` otherwise. Advisors call this instead of trusting
/// `result.measured` blindly, so that "baseline and trials share one layout"
/// holds by construction rather than by convention — including for callers that
/// assembled the `AnalysisResult` themselves.
///
/// `Ok(None)` means the sample is below [`MIN_MEASURE_FEATURES`]: there is no
/// honest baseline, and every caller here treats that as "say nothing".
pub fn baseline_under(
    measured: Option<&MeasuredEncoding>,
    sample: &[SampledFeature],
    layout: &SyntheticLayout,
) -> Result<Option<MeasuredEncoding>> {
    if let Some(measured) = measured {
        if layout.produced(measured) {
            return Ok(Some(measured.clone()));
        }
    }
    measure_sample_layout(sample, &MeasureSettings::default(), layout)
}

/// Encoder/compression settings for a sample measurement — the same levers a
/// real `stt-build` run exposes (`--zstd-level`, `--quantize-coords`,
/// `--quantize-attr`, `--quantize-attrs-auto`, `--compact-times`).
///
/// Every field maps 1:1 onto an [`EncoderConfig`] knob, and every default is
/// TODAY'S build default — so a `MeasureSettings::default()` measurement is
/// byte-identical to one taken before any field was added. That is what makes
/// the trial oracle ([`crate::oracle::run_trials`]) able to move exactly one
/// lever at a time without disturbing the incumbent baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureSettings {
    /// zstd level applied to the encoded tile payload.
    pub zstd_level: i32,
    /// Fixed-point coordinate quantization ground precision in meters
    /// (`None` = Float64 coordinates, the build default).
    pub quantize_coords_m: Option<f64>,
    /// Range-adaptive `UInt16` quantization for Float64 numeric properties.
    pub quantize_attrs_auto: bool,
    /// Per-property explicit fixed-point precisions (`name → ground precision`),
    /// the `--quantize-attr name=step` lever. Empty (the default) leaves every
    /// numeric property `Float64`, exactly as
    /// [`EncoderConfig::default`](stt_core::arrow_tile::EncoderConfig) does.
    ///
    /// A `BTreeMap` rather than the encoder's `HashMap`: settings are compared,
    /// logged and (from MO-6 on) used as a measurement-cache key, and none of
    /// that may depend on hash iteration order. It is converted at the
    /// [`EncoderConfig`] boundary, where the encoder only ever does keyed
    /// lookups.
    pub quantize_attrs: BTreeMap<String, f64>,
    /// Compact the per-feature time columns (`TILE_META.st`/`.et`). `true` is
    /// the encoder default; it is a knob here because it is one of the five D4
    /// decision sites, and the recorded reason D4 exists at all is that compact
    /// times measured **+3.4% wire while −13.1% uncompressed** — a sign
    /// inversion no pre-compression model could have caught.
    pub compact_times: bool,
}

impl Default for MeasureSettings {
    /// The `stt-build` defaults: zstd level 3, no quantization, compact times
    /// on (which is [`EncoderConfig::default`](stt_core::arrow_tile::EncoderConfig)'s
    /// value, so the added fields are byte-neutral by construction).
    fn default() -> Self {
        Self {
            zstd_level: 3,
            quantize_coords_m: None,
            quantize_attrs_auto: false,
            quantize_attrs: BTreeMap::new(),
            compact_times: true,
        }
    }
}

/// Measured encoder output for a sample: real compressed bytes per feature
/// plus per-column cost attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredEncoding {
    /// Source features actually encoded (the dominant-geometry-kind subset of
    /// the sample; multi-part geometries count once even though they encode
    /// as one row per part).
    pub features: usize,
    /// Encoder geometry kind that was measured (`"point"` / `"line"` /
    /// `"polygon"`) — for mixed-geometry sources, the dominant kind.
    pub geometry_kind: String,
    /// Compressed size of the encoded synthetic tile(s) (tile payload + zstd),
    /// summed over [`Self::tiles`].
    pub bytes_total: usize,
    /// `bytes_total / features`.
    pub bytes_per_feature: f64,
    /// Uncompressed-payload / compressed ratio (replaces the assumed 3x).
    pub zstd_ratio: f64,
    /// Synthetic tiles the sample was split across. `1` is the single-tile
    /// wrapper ([`measure_sample`]); more means the per-tile encoder + zstd
    /// framing overhead is charged at realistic occupancy. Reports written
    /// before the field existed were all single-tile.
    #[serde(default = "one_tile")]
    pub tiles: usize,
    /// Per-column compressed cost, sorted descending by bytes.
    pub per_column: Vec<ColumnCost>,
}

/// serde default for [`MeasuredEncoding::tiles`].
fn one_tile() -> usize {
    1
}

/// Compressed cost of one encoded tile column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnCost {
    /// Column name (e.g. `geometry`, `start_time`, a property name).
    pub name: String,
    /// Bytes charged to this column under the active
    /// [`AttributionDesign`], summed over the measured synthetic tiles.
    ///
    /// Under the default leave-one-out design this is
    /// [`Self::marginal_bytes`]: the post-zstd bytes the tiles shed if the
    /// column vanished. Under the [`AttributionDesign::SingletonV1`] rollback
    /// it is the old proxy — the column IPC-encoded ALONE and zstd-compressed,
    /// which over-charges every column that shares information with another.
    pub compressed_bytes: usize,
    /// Fraction of the summed per-column bytes in `[0, 1]`.
    ///
    /// Under leave-one-out the shares are normalised marginals, so they sum to
    /// 1 and never over-spend the whole-tile budget (the efficiency property
    /// the singleton proxy broke).
    pub share: f64,
    /// Leave-one-out marginal bytes, summed over the measured tiles: the
    /// honest "what does dropping this column actually save" unit.
    ///
    /// `0` under the [`AttributionDesign::SingletonV1`] rollback, where no
    /// leave-one-out encode was performed. Reports written before the field
    /// existed were all singleton, so the serde default matches them.
    #[serde(default)]
    pub marginal_bytes: u64,
    /// Standard error of [`Self::share`], estimated across the measured
    /// synthetic tiles (the ratio-estimator form — see [`ShareDispersion`]).
    ///
    /// `0.0` for a single-tile measurement ([`measure_sample`]): one tile
    /// carries no dispersion evidence. With a multi-tile
    /// [`SyntheticLayout`] it is the tile-to-tile spread of the share, which
    /// is finite rather than zero even though every tile of the sample was
    /// measured — it answers "how far would this share move on comparable
    /// tiles", not "how much of the population did I miss".
    #[serde(default)]
    pub share_stderr: f64,
}

/// The encoder geometry bucket a sampled geometry falls into (the tiler emits
/// one layer per kind, so a synthetic layer must be single-kind).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeomKind {
    Point = 0,
    Line = 1,
    Polygon = 2,
}

impl GeomKind {
    fn name(self) -> &'static str {
        match self {
            GeomKind::Point => "point",
            GeomKind::Line => "line",
            GeomKind::Polygon => "polygon",
        }
    }
}

/// Classify a sampled geometry; `None` for GeometryCollection (no encoder
/// bucket — such features are excluded from the measurement).
fn kind_of(geom: &geo_types::Geometry<f64>) -> Option<GeomKind> {
    use geo_types::Geometry as G;
    match geom {
        G::Point(_) | G::MultiPoint(_) => Some(GeomKind::Point),
        G::Line(_) | G::LineString(_) | G::MultiLineString(_) => Some(GeomKind::Line),
        G::Polygon(_) | G::MultiPolygon(_) | G::Rect(_) | G::Triangle(_) => Some(GeomKind::Polygon),
        G::GeometryCollection(_) => None,
    }
}

/// Measure the real encoded + compressed size of a loader sample as ONE
/// synthetic tile.
///
/// The single-tile wrapper around [`measure_sample_layout`] — identical to the
/// pre-layout behaviour, and the documented rollback. Prefer
/// `measure_sample_layout` with the run's [`SyntheticLayout`]: a single tile
/// amortises per-tile framing over the whole sample and biases bytes/feature
/// low.
pub fn measure_sample(
    sample: &[SampledFeature],
    settings: &MeasureSettings,
) -> Result<Option<MeasuredEncoding>> {
    measure_sample_layout(sample, settings, &SyntheticLayout::single())
}

/// [`measure_sample_layout`] with the default ([`AttributionDesign::LeaveOneOutV2`])
/// per-column attribution.
pub fn measure_sample_layout(
    sample: &[SampledFeature],
    settings: &MeasureSettings,
    layout: &SyntheticLayout,
) -> Result<Option<MeasuredEncoding>> {
    measure_sample_layout_with(sample, settings, layout, AttributionDesign::default())
}

/// Measure the real encoded + compressed size of a loader sample, cut into the
/// synthetic tiles `layout` describes.
///
/// Takes the dominant geometry kind's subset (mixed-geometry samples measure
/// that subset only — [`MeasuredEncoding::geometry_kind`] records which),
/// splits it into `layout.tiles()` contiguous tiles whose sizes are
/// proportional to the layout weights, encodes each through the production
/// stt-core encoder with the given settings, compresses each with zstd, and
/// sums. Returns `Ok(None)` when the usable subset is smaller than
/// [`MIN_MEASURE_FEATURES`].
///
/// The tile count is reduced (never the layout mutated) when the subset cannot
/// give every tile [`MIN_TILE_FEATURES`] features, so the result is a pure
/// function of `(sample, settings, layout, attribution)`.
///
/// `attribution` selects the per-column cost definition; pass
/// [`AttributionDesign::default`] unless you are exercising the documented
/// singleton rollback. It has NO effect on `bytes_total` /
/// `bytes_per_feature` / `zstd_ratio` — those are whole-tile measurements.
pub fn measure_sample_layout_with(
    sample: &[SampledFeature],
    settings: &MeasureSettings,
    layout: &SyntheticLayout,
    attribution: AttributionDesign,
) -> Result<Option<MeasuredEncoding>> {
    // Dominant kind, ties broken toward point > line > polygon (deterministic).
    let mut counts = [0usize; 3];
    for f in sample {
        if let Some(kind) = kind_of(&f.geometry) {
            counts[kind as usize] += 1;
        }
    }
    let mut dominant = GeomKind::Point;
    for kind in [GeomKind::Line, GeomKind::Polygon] {
        if counts[kind as usize] > counts[dominant as usize] {
            dominant = kind;
        }
    }

    let subset: Vec<&SampledFeature> = sample
        .iter()
        .filter(|f| kind_of(&f.geometry) == Some(dominant))
        .collect();
    if subset.len() < MIN_MEASURE_FEATURES {
        return Ok(None);
    }

    let weights = layout_weights(layout, subset.len());
    let tile_counts = split_counts(subset.len(), &weights);

    let cfg = EncoderConfig {
        quantize_coords_m: settings.quantize_coords_m,
        quantize_attrs_auto: settings.quantize_attrs_auto,
        quantize_attrs: settings
            .quantize_attrs
            .iter()
            .map(|(name, step)| (name.clone(), *step))
            .collect(),
        compact_times: settings.compact_times,
        ..EncoderConfig::default()
    };

    let mut payload_bytes = 0usize;
    let mut compressed_bytes = 0usize;
    let mut column_acc: BTreeMap<String, ColAcc> = BTreeMap::new();
    let mut dispersion = ShareDispersion::default();
    let mut cursor = 0usize;
    for &take in &tile_counts {
        let slice = &subset[cursor..cursor + take];
        cursor += take;
        let layer = build_layer(slice, dominant);
        let payload = encode_tile_with(&[layer], &cfg).context("sample tile encode failed")?;
        let compressed = compress_zstd_with_dict_level(&payload, None, settings.zstd_level)
            .context("sample tile compression failed")?;
        payload_bytes += payload.len();
        compressed_bytes += compressed.len();
        accumulate_columns(
            &payload,
            settings.zstd_level,
            attribution,
            &mut column_acc,
            &mut dispersion,
        )?;
    }

    Ok(Some(MeasuredEncoding {
        features: subset.len(),
        geometry_kind: dominant.name().to_string(),
        bytes_total: compressed_bytes,
        bytes_per_feature: compressed_bytes as f64 / subset.len() as f64,
        zstd_ratio: payload_bytes as f64 / compressed_bytes.max(1) as f64,
        tiles: tile_counts.len(),
        per_column: finish_columns(column_acc, &dispersion),
    }))
}

/// The layout weights actually usable for `total` features: the layout's tile
/// sizes, thinned to at most `total / MIN_TILE_FEATURES` tiles by taking
/// evenly-spaced entries (midpoint rule) so the retained weights still span the
/// occupancy range rather than clustering at the light end.
fn layout_weights(layout: &SyntheticLayout, total: usize) -> Vec<usize> {
    let src: &[usize] = if layout.tile_sizes.is_empty() {
        &[1]
    } else {
        &layout.tile_sizes
    };
    let affordable = (total / MIN_TILE_FEATURES).max(1);
    let tiles = src.len().min(affordable);
    if tiles == src.len() {
        return src.to_vec();
    }
    (0..tiles)
        .map(|i| src[(((2 * i + 1) * src.len()) / (2 * tiles)).min(src.len() - 1)])
        .collect()
}

/// Split `total` features across `weights.len()` tiles: [`MIN_TILE_FEATURES`]
/// to every tile first, then the remainder by largest-remainder rounding on the
/// weights (ties broken by ascending tile index). Sums to exactly `total`.
///
/// Callers guarantee `total >= MIN_TILE_FEATURES * weights.len()` via
/// [`layout_weights`].
fn split_counts(total: usize, weights: &[usize]) -> Vec<usize> {
    let k = weights.len().max(1);
    if k == 1 {
        return vec![total];
    }
    let mut counts = vec![MIN_TILE_FEATURES; k];
    let rest = total - MIN_TILE_FEATURES * k;
    let w_sum: u128 = weights.iter().map(|&w| w as u128).sum();
    let (w, w_sum): (Vec<u128>, u128) = if w_sum == 0 {
        (vec![1u128; k], k as u128)
    } else {
        (weights.iter().map(|&x| x as u128).collect(), w_sum)
    };
    let mut rems: Vec<(u128, usize)> = Vec::with_capacity(k);
    let mut placed = 0usize;
    for i in 0..k {
        let num = rest as u128 * w[i];
        let base = (num / w_sum) as usize;
        counts[i] += base;
        placed += base;
        rems.push((num % w_sum, i));
    }
    let mut left = rest - placed;
    rems.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for &(_, i) in &rems {
        if left == 0 {
            break;
        }
        counts[i] += 1;
        left -= 1;
    }
    counts
}

fn line_coords(ls: &geo_types::LineString<f64>) -> Vec<Coord> {
    ls.0.iter().map(|c| [c.x, c.y]).collect()
}

fn polygon_rings(polygon: &geo_types::Polygon<f64>) -> Vec<Vec<Coord>> {
    std::iter::once(polygon.exterior())
        .chain(polygon.interiors().iter())
        .map(line_coords)
        .collect()
}

/// Point parts of a point-kind geometry (a MultiPoint flattens to one row per
/// point, the shape a tiler split produces).
fn point_parts(geom: &geo_types::Geometry<f64>) -> Vec<Coord> {
    use geo_types::Geometry as G;
    match geom {
        G::Point(p) => vec![[p.x(), p.y()]],
        G::MultiPoint(mp) => mp.0.iter().map(|p| [p.x(), p.y()]).collect(),
        _ => Vec::new(),
    }
}

/// Line parts of a line-kind geometry; empty parts are dropped.
fn line_parts(geom: &geo_types::Geometry<f64>) -> Vec<Vec<Coord>> {
    use geo_types::Geometry as G;
    let parts = match geom {
        G::Line(l) => vec![vec![[l.start.x, l.start.y], [l.end.x, l.end.y]]],
        G::LineString(ls) => vec![line_coords(ls)],
        G::MultiLineString(mls) => mls.0.iter().map(line_coords).collect(),
        _ => Vec::new(),
    };
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Polygon parts (ring lists) of a polygon-kind geometry; empty parts dropped.
fn polygon_parts(geom: &geo_types::Geometry<f64>) -> Vec<Vec<Vec<Coord>>> {
    use geo_types::Geometry as G;
    let parts = match geom {
        G::Polygon(p) => vec![polygon_rings(p)],
        G::MultiPolygon(mp) => mp.0.iter().map(polygon_rings).collect(),
        G::Rect(r) => vec![polygon_rings(&r.to_polygon())],
        G::Triangle(t) => vec![polygon_rings(&t.to_polygon())],
        _ => Vec::new(),
    };
    parts
        .into_iter()
        .filter(|rings| rings.iter().any(|ring| !ring.is_empty()))
        .collect()
}

/// Assemble the sampled subset into one synthetic tile layer. Multi-part
/// geometries flatten into one row per part, duplicating times and property
/// values (what a tiler's multi-geometry split produces), so their full cost
/// still lands on the one source feature.
fn build_layer(subset: &[&SampledFeature], kind: GeomKind) -> ColumnarLayer {
    // Property schema: names in first-seen order; Numeric vs Categorical from
    // the first value seen (all rows come from one Parquet schema, so mixed
    // types per name don't occur in practice — mismatches encode as null).
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

    let mut ids: Vec<u64> = Vec::new();
    let mut starts: Vec<i64> = Vec::new();
    let mut ends: Vec<i64> = Vec::new();
    let mut points: Vec<Coord> = Vec::new();
    let mut lines: Vec<Vec<Coord>> = Vec::new();
    let mut polygons: Vec<Vec<Vec<Coord>>> = Vec::new();
    let mut numeric_cols: Vec<Vec<Option<f64>>> = vec![Vec::new(); names.len()];
    let mut categorical_cols: Vec<Vec<Option<String>>> = vec![Vec::new(); names.len()];

    for feature in subset {
        let n_parts = match kind {
            GeomKind::Point => {
                let parts = point_parts(&feature.geometry);
                let n = parts.len();
                points.extend(parts);
                n
            }
            GeomKind::Line => {
                let parts = line_parts(&feature.geometry);
                let n = parts.len();
                lines.extend(parts);
                n
            }
            GeomKind::Polygon => {
                let parts = polygon_parts(&feature.geometry);
                let n = parts.len();
                polygons.extend(parts);
                n
            }
        };
        for _ in 0..n_parts {
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
        GeomKind::Point => GeometryColumn::Point(points),
        GeomKind::Line => GeometryColumn::LineString(lines),
        GeomKind::Polygon => GeometryColumn::Polygon(polygons),
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

/// Per-column accumulator across the measured synthetic tiles.
#[derive(Default)]
struct ColAcc {
    /// Bytes charged under the active design.
    bytes: u64,
    /// Leave-one-out marginal bytes (`0` under the singleton rollback).
    marginal: u64,
}

/// Per-column compressed-cost attribution for ONE synthetic tile: decode the
/// encoded tile, run the shared [`attribution`] pass over each layer's batch,
/// fold the bytes into `acc` by column name and record the tile as one
/// dispersion observation.
///
/// `BTreeMap`, not `HashMap`: the accumulator's iteration order is part of the
/// determinism contract, and the per-tile map is what feeds
/// [`ShareDispersion::observe_tile`].
fn accumulate_columns(
    payload: &[u8],
    zstd_level: i32,
    design: AttributionDesign,
    acc: &mut BTreeMap<String, ColAcc>,
    dispersion: &mut ShareDispersion,
) -> Result<()> {
    let layers = decode_tile(payload).context("sample tile decode failed")?;
    let mut per_tile: BTreeMap<String, u64> = BTreeMap::new();
    for layer in &layers {
        for column in attribute_columns(&layer.batch, zstd_level, design)? {
            let entry = acc.entry(column.name.clone()).or_default();
            entry.bytes += column.bytes;
            entry.marginal += column.marginal_bytes;
            *per_tile.entry(column.name).or_insert(0) += column.bytes;
        }
    }
    dispersion.observe_tile(&per_tile);
    Ok(())
}

/// Turn the accumulated per-column bytes into shares of the per-column sum,
/// sorted descending by bytes with name as the deterministic tiebreak, and
/// attach each share's standard error from `dispersion`.
fn finish_columns(acc: BTreeMap<String, ColAcc>, dispersion: &ShareDispersion) -> Vec<ColumnCost> {
    let total: u64 = acc.values().map(|c| c.bytes).sum();
    let mut out: Vec<ColumnCost> = acc
        .into_iter()
        .map(|(name, c)| ColumnCost {
            share: c.bytes as f64 / total.max(1) as f64,
            share_stderr: dispersion.stderr(&name),
            compressed_bytes: c.bytes as usize,
            marginal_bytes: c.marginal,
            name,
        })
        .collect();
    out.sort_by(|a, b| {
        b.compressed_bytes
            .cmp(&a.compressed_bytes)
            .then_with(|| a.name.cmp(&b.name))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry, LineString, Point};

    /// n spread-out points with one numeric and one string property. Knuth-hash
    /// jitter keeps the f64 coordinate mantissas high-entropy so quantization
    /// has real bytes to win.
    fn point_sample(n: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| {
                let jitter = |salt: u64| {
                    ((i as u64).wrapping_add(salt).wrapping_mul(2_654_435_761) % 100_000) as f64
                        * 1e-7
                };
                SampledFeature {
                    geometry: Geometry::Point(Point::new(
                        -73.5 + i as f64 * 0.0013 + jitter(0),
                        45.5 + (i % 7) as f64 * 0.0021 + jitter(17),
                    )),
                    timestamp_ms: 1_600_000_000_000 + i as u64 * 1_000,
                    properties: vec![
                        (
                            "magnitude".to_string(),
                            PropValue::Number(1.0 + (i % 90) as f64 * 0.137),
                        ),
                        (
                            "region".to_string(),
                            PropValue::Text(format!("region-{}", i % 5)),
                        ),
                    ],
                }
            })
            .collect()
    }

    #[test]
    fn measures_point_sample() {
        let sample = point_sample(200);
        let measured = measure_sample(&sample, &MeasureSettings::default())
            .unwrap()
            .expect("200 features is enough to measure");
        assert_eq!(measured.features, 200);
        assert_eq!(measured.geometry_kind, "point");
        assert!(measured.bytes_total > 0);
        assert!(measured.bytes_per_feature > 0.0);
        assert!(measured.zstd_ratio > 0.0);

        let share_sum: f64 = measured.per_column.iter().map(|c| c.share).sum();
        assert!((share_sum - 1.0).abs() < 1e-9, "shares sum to {share_sum}");
        for name in ["geometry", "magnitude", "region", "id", "start_time"] {
            assert!(
                measured.per_column.iter().any(|c| c.name == name),
                "missing column {name}"
            );
        }
        // Sorted descending by compressed bytes.
        for pair in measured.per_column.windows(2) {
            assert!(pair[0].compressed_bytes >= pair[1].compressed_bytes);
        }
    }

    #[test]
    fn quantized_coords_never_larger() {
        let sample = point_sample(500);
        let base = measure_sample(&sample, &MeasureSettings::default())
            .unwrap()
            .unwrap();
        let quantized = measure_sample(
            &sample,
            &MeasureSettings {
                quantize_coords_m: Some(0.1),
                ..MeasureSettings::default()
            },
        )
        .unwrap()
        .unwrap();
        assert!(
            quantized.bytes_total <= base.bytes_total,
            "quantized {} > unquantized {}",
            quantized.bytes_total,
            base.bytes_total
        );
    }

    #[test]
    fn mixed_geometry_measures_dominant_kind_subset() {
        let mut sample = point_sample(150);
        for i in 0..50 {
            sample.push(SampledFeature {
                geometry: Geometry::LineString(LineString::from(vec![
                    (0.0, 0.0),
                    (i as f64 * 0.01, 1.0),
                ])),
                timestamp_ms: 0,
                properties: vec![],
            });
        }
        let measured = measure_sample(&sample, &MeasureSettings::default())
            .unwrap()
            .unwrap();
        assert_eq!(measured.geometry_kind, "point");
        assert_eq!(measured.features, 150);
    }

    #[test]
    fn empty_or_tiny_sample_returns_none() {
        assert!(measure_sample(&[], &MeasureSettings::default())
            .unwrap()
            .is_none());
        let tiny = point_sample(MIN_MEASURE_FEATURES - 1);
        assert!(measure_sample(&tiny, &MeasureSettings::default())
            .unwrap()
            .is_none());
    }

    // ------------------------------------------------------------------
    // MO-4: synthetic-tile layout
    // ------------------------------------------------------------------

    use crate::analysis::density::ZoomDensity;

    /// A density result whose deepest zoom publishes the given occupancy
    /// quantiles — the only fields [`SyntheticLayout::from_density`] reads.
    fn density_with(zoom: u8, median: usize, p95: usize, max: usize) -> DensityAnalysis {
        DensityAnalysis {
            per_zoom: vec![ZoomDensity {
                zoom,
                tile_count: 1_000,
                avg_features_per_tile: median as f64,
                median_features_per_tile: median,
                max_features_per_tile: max,
                p95_features_per_tile: p95,
                top1pct_feature_share: 0.0,
                oversized_tiles: 0,
                undersized_tiles: 0,
                estimated_size_uncompressed: 0,
                estimated_size_compressed: 0,
            }],
            estimated_tile_count: 1_000,
            estimated_archive_size: 0,
            issues: Vec::new(),
        }
    }

    #[test]
    fn layout_quantiles_are_deterministic_and_hand_computable() {
        // Q(0) = 1, Q(0.5) = 100, Q(0.95) = 200, sampled at (2i+1)/16:
        //   0.0625 → 1 + 99·0.125   = 13.375  → 13
        //   0.1875 → 1 + 99·0.375   = 38.125  → 38
        //   0.3125 → 1 + 99·0.625   = 62.875  → 63
        //   0.4375 → 1 + 99·0.875   = 87.625  → 88
        //   0.5625 → 100 + 100·(0.0625/0.45) = 113.89 → 114
        //   0.6875 → 100 + 100·(0.1875/0.45) = 141.67 → 142
        //   0.8125 → 100 + 100·(0.3125/0.45) = 169.44 → 169
        //   0.9375 → 100 + 100·(0.4375/0.45) = 197.22 → 197
        let layout = SyntheticLayout::from_density(&density_with(12, 100, 200, 1_000));
        assert_eq!(layout.tile_sizes, vec![13, 38, 63, 88, 114, 142, 169, 197]);
        // The p99+ outlier (max = 1000) never sets a synthetic tile size.
        assert!(layout.tile_sizes.iter().all(|&s| s <= 200));
        // Pure function of the published stats.
        assert_eq!(
            layout,
            SyntheticLayout::from_density(&density_with(12, 100, 200, 1_000))
        );

        // Only occupancy is read: changing the SIZE estimates (the fields the
        // calibrated second density pass rewrites) cannot move the layout.
        let mut calibrated = density_with(12, 100, 200, 1_000);
        calibrated.per_zoom[0].estimated_size_compressed = 999_999;
        calibrated.per_zoom[0].estimated_size_uncompressed = 999_999;
        calibrated.estimated_archive_size = 999_999;
        assert_eq!(layout, SyntheticLayout::from_density(&calibrated));

        // Deepest zoom wins, not first-listed.
        let mut two = density_with(4, 10, 20, 30);
        two.per_zoom
            .push(density_with(12, 100, 200, 1_000).per_zoom[0].clone());
        assert_eq!(layout, SyntheticLayout::from_density(&two));

        // Degenerate density collapses to the single-tile rollback.
        let empty = DensityAnalysis {
            per_zoom: Vec::new(),
            estimated_tile_count: 0,
            estimated_archive_size: 0,
            issues: Vec::new(),
        };
        assert_eq!(
            SyntheticLayout::from_density(&empty),
            SyntheticLayout::single()
        );
    }

    #[test]
    fn layout_from_observed_counts_takes_evenly_spaced_quantiles() {
        // Eight counts, eight tiles → every observation is one synthetic tile.
        let layout = SyntheticLayout::from_feature_counts(&[8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(layout.tile_sizes, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        // Empty cells never become tiles.
        assert_eq!(
            SyntheticLayout::from_feature_counts(&[0, 0, 0]),
            SyntheticLayout::single()
        );
    }

    #[test]
    fn multi_tile_measurement_exceeds_single_tile_on_the_same_sample() {
        // THE bias MO-4 corrects: one synthetic tile amortises per-tile encoder
        // + zstd framing over the whole sample, so it under-charges. Splitting
        // the SAME features into real-occupancy tiles must cost more bytes.
        let sample = point_sample(800);
        let settings = MeasureSettings::default();
        let layout = SyntheticLayout::from_density(&density_with(12, 100, 200, 1_000));

        let single = measure_sample(&sample, &settings).unwrap().unwrap();
        let multi = measure_sample_layout(&sample, &settings, &layout)
            .unwrap()
            .unwrap();

        assert_eq!(single.tiles, 1);
        assert_eq!(multi.tiles, 8);
        assert_eq!(single.features, multi.features, "same features encoded");
        assert!(
            multi.bytes_total > single.bytes_total,
            "multi-tile {} must exceed single-tile {} (per-tile framing is real)",
            multi.bytes_total,
            single.bytes_total
        );
        assert!(multi.bytes_per_feature > single.bytes_per_feature);
        // Measured here: 17_635 B single-tile vs 25_884 B multi-tile (1.47x).
        // That gap is exactly why a baseline and a trial must share ONE layout:
        // measured across the cut it is a fake 32% shrink. It is now
        // structurally impossible to compare across cuts by accident —
        // `SyntheticLayout::produced` detects it and `baseline_under`
        // re-measures — which is what let `MeasurementMode::DensityLayout`
        // become the default in `lib.rs`.
        // Shares still normalise over the merged per-column accumulation.
        let share_sum: f64 = multi.per_column.iter().map(|c| c.share).sum();
        assert!((share_sum - 1.0).abs() < 1e-9, "shares sum to {share_sum}");
        for pair in multi.per_column.windows(2) {
            assert!(pair[0].compressed_bytes >= pair[1].compressed_bytes);
        }
    }

    #[test]
    fn measure_sample_is_the_single_tile_wrapper() {
        // The rollback must be exactly the old behaviour, byte for byte.
        let sample = point_sample(300);
        let settings = MeasureSettings::default();
        let wrapper = measure_sample(&sample, &settings).unwrap().unwrap();
        let explicit = measure_sample_layout(&sample, &settings, &SyntheticLayout::single())
            .unwrap()
            .unwrap();
        assert_eq!(wrapper.bytes_total, explicit.bytes_total);
        assert_eq!(wrapper.tiles, 1);
        assert_eq!(
            serde_json::to_string(&wrapper).unwrap(),
            serde_json::to_string(&explicit).unwrap()
        );
    }

    #[test]
    fn quantized_coords_never_larger_under_a_shared_layout() {
        // The baseline-vs-trial invariant: same sample, SAME layout, one lever
        // moved. Re-runs the existing quantization check through the layout
        // path so trial deltas cannot confound layout with lever.
        let sample = point_sample(500);
        let layout = SyntheticLayout::from_density(&density_with(12, 60, 90, 400));
        let base = measure_sample_layout(&sample, &MeasureSettings::default(), &layout)
            .unwrap()
            .unwrap();
        let quantized = measure_sample_layout(
            &sample,
            &MeasureSettings {
                quantize_coords_m: Some(0.1),
                ..MeasureSettings::default()
            },
            &layout,
        )
        .unwrap()
        .unwrap();
        assert_eq!(base.tiles, quantized.tiles, "same layout → same tile count");
        assert_eq!(base.features, quantized.features);
        assert!(
            quantized.bytes_total <= base.bytes_total,
            "quantized {} > unquantized {}",
            quantized.bytes_total,
            base.bytes_total
        );
    }

    #[test]
    fn layout_collapses_rather_than_measuring_undersized_tiles() {
        // 30 features cannot fill eight 8-feature tiles: the layout is thinned
        // to 3 tiles, never mutated, and every tile clears the floor.
        let sample = point_sample(30);
        let layout = SyntheticLayout::from_density(&density_with(12, 100, 200, 1_000));
        assert_eq!(layout.tiles(), 8);
        let measured = measure_sample_layout(&sample, &MeasureSettings::default(), &layout)
            .unwrap()
            .expect("30 >= MIN_MEASURE_FEATURES");
        assert_eq!(measured.tiles, 3);
        assert_eq!(measured.features, 30);

        // MIN_MEASURE_FEATURES still applies per measurement, layout or not.
        let tiny = point_sample(MIN_MEASURE_FEATURES - 1);
        assert!(
            measure_sample_layout(&tiny, &MeasureSettings::default(), &layout)
                .unwrap()
                .is_none()
        );
        assert!(
            measure_sample_layout(&[], &MeasureSettings::default(), &layout)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_measurement_knows_which_cut_it_came_from() {
        // The structural half of the MO-4 invariant: a measurement taken under
        // the WRONG layout is detectable, so nothing has to trust convention.
        let sample = point_sample(800);
        let settings = MeasureSettings::default();
        let layout = SyntheticLayout::from_density(&density_with(12, 100, 200, 1_000));
        let single = SyntheticLayout::single();

        let multi_m = measure_sample_layout(&sample, &settings, &layout)
            .unwrap()
            .unwrap();
        let single_m = measure_sample(&sample, &settings).unwrap().unwrap();

        assert!(layout.produced(&multi_m));
        assert!(single.produced(&single_m));
        // …and each rejects the other's cut, which is the whole point.
        assert!(!layout.produced(&single_m));
        assert!(!single.produced(&multi_m));

        // A thinned layout still recognises its own output: 30 features cannot
        // fill eight tiles, so the layout resolves to 3 — and `produced` asks
        // the RESOLVED question, not the nominal one.
        assert_eq!(layout.tiles(), 8);
        assert_eq!(layout.resolved_tiles(30), 3);
        let thinned = measure_sample_layout(&point_sample(30), &settings, &layout)
            .unwrap()
            .unwrap();
        assert_eq!(thinned.tiles, 3);
        assert!(layout.produced(&thinned));
    }

    #[test]
    fn baseline_under_reuses_a_matching_measurement_and_replaces_a_mismatched_one() {
        let sample = point_sample(800);
        let layout = SyntheticLayout::from_density(&density_with(12, 100, 200, 1_000));
        let settings = MeasureSettings::default();
        let under_layout = measure_sample_layout(&sample, &settings, &layout)
            .unwrap()
            .unwrap();
        let single = measure_sample(&sample, &settings).unwrap().unwrap();

        // Matching: reused verbatim, no encode performed.
        let reused = baseline_under(Some(&under_layout), &sample, &layout)
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::to_string(&reused).unwrap(),
            serde_json::to_string(&under_layout).unwrap()
        );

        // Mismatched (a single-tile measurement handed to a multi-tile run):
        // silently discarded and re-measured under the run's own layout, so a
        // trial can never be compared against the 1.47x-cheaper cut.
        let repaired = baseline_under(Some(&single), &sample, &layout)
            .unwrap()
            .unwrap();
        assert_eq!(repaired.tiles, under_layout.tiles);
        assert_eq!(repaired.bytes_total, under_layout.bytes_total);
        assert!(repaired.bytes_total > single.bytes_total);

        // Absent: measured under the layout.
        let fresh = baseline_under(None, &sample, &layout).unwrap().unwrap();
        assert_eq!(fresh.bytes_total, under_layout.bytes_total);

        // Below the floor: no honest baseline exists, layout or not.
        assert!(baseline_under(None, &point_sample(4), &layout)
            .unwrap()
            .is_none());
    }

    #[test]
    fn split_counts_is_exact_and_floored() {
        let counts = split_counts(100, &[1, 2, 3, 4]);
        assert_eq!(counts.iter().sum::<usize>(), 100);
        assert!(counts.iter().all(|&c| c >= MIN_TILE_FEATURES));
        // 68 to share over weights 1:2:3:4 → 6.8 / 13.6 / 20.4 / 27.2, floors
        // 6 / 13 / 20 / 27 = 66, two left over to the largest remainders
        // (.8 → tile 0, .6 → tile 1).
        assert_eq!(counts, vec![8 + 7, 8 + 14, 8 + 20, 8 + 27]);
        // Deterministic.
        assert_eq!(counts, split_counts(100, &[1, 2, 3, 4]));
        // Degenerate weights still allocate exactly.
        assert_eq!(split_counts(40, &[0, 0]).iter().sum::<usize>(), 40);
    }

    #[test]
    fn layout_measurement_is_byte_identical_across_runs() {
        let sample = point_sample(400);
        let layout = SyntheticLayout::from_density(&density_with(12, 100, 200, 1_000));
        let a = measure_sample_layout(&sample, &MeasureSettings::default(), &layout).unwrap();
        let b = measure_sample_layout(&sample, &MeasureSettings::default(), &layout).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }

    #[test]
    fn tiles_field_defaults_to_one_for_pre_layout_reports() {
        let json = r#"{"features":10,"geometry_kind":"point","bytes_total":100,
            "bytes_per_feature":10.0,"zstd_ratio":3.0,"per_column":[]}"#;
        let back: MeasuredEncoding = serde_json::from_str(json).unwrap();
        assert_eq!(back.tiles, 1);
    }

    // ------------------------------------------------------------------
    // MO-3: leave-one-out attribution
    // ------------------------------------------------------------------

    #[test]
    fn measured_columns_publish_leave_one_out_marginals() {
        let sample = point_sample(400);
        let measured = measure_sample(&sample, &MeasureSettings::default())
            .unwrap()
            .unwrap();

        // Efficiency: the per-column sum no longer over-spends the tile.
        let attributed: usize = measured.per_column.iter().map(|c| c.compressed_bytes).sum();
        assert!(
            attributed <= measured.bytes_total,
            "Σ marginals {attributed} must fit inside the tile's {} B",
            measured.bytes_total
        );
        let share_sum: f64 = measured.per_column.iter().map(|c| c.share).sum();
        assert!((share_sum - 1.0).abs() < 1e-9, "shares sum to {share_sum}");

        // Under the default design the charged cost IS the marginal.
        for c in &measured.per_column {
            assert_eq!(
                c.marginal_bytes as usize, c.compressed_bytes,
                "column {}: marginal must be the charged cost under loo-v2",
                c.name
            );
        }

        // The singleton rollback stays reachable and remains inefficient —
        // which is the whole reason it was replaced.
        let singleton = measure_sample_layout_with(
            &sample,
            &MeasureSettings::default(),
            &SyntheticLayout::single(),
            AttributionDesign::SingletonV1,
        )
        .unwrap()
        .unwrap();
        let singleton_sum: usize = singleton
            .per_column
            .iter()
            .map(|c| c.compressed_bytes)
            .sum();
        assert!(
            singleton_sum > singleton.bytes_total,
            "guard for the REJECTED proxy: singleton sum {singleton_sum} must exceed the tile's \
             {} B",
            singleton.bytes_total
        );
        assert!(singleton.per_column.iter().all(|c| c.marginal_bytes == 0));
        // Whole-tile numbers are attribution-independent.
        assert_eq!(singleton.bytes_total, measured.bytes_total);
        assert_eq!(singleton.features, measured.features);
    }

    #[test]
    fn duplicated_property_columns_are_not_double_charged() {
        // The recorded misattribution case, end to end through the sample
        // measurement: two properties carrying byte-identical high-entropy
        // values. The information is stored once, so dropping either costs
        // almost nothing — LOO charges each of them almost nothing, while the
        // singleton proxy bills both in full.
        let sample: Vec<SampledFeature> = (0..400)
            .map(|i| {
                let noisy = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64 / 1e6;
                SampledFeature {
                    geometry: Geometry::Point(Point::new(
                        -73.5 + i as f64 * 0.001,
                        45.5 + (i % 11) as f64 * 0.001,
                    )),
                    timestamp_ms: 1_600_000_000_000 + i as u64 * 1_000,
                    properties: vec![
                        ("dup_a".to_string(), PropValue::Number(noisy)),
                        ("dup_b".to_string(), PropValue::Number(noisy)),
                    ],
                }
            })
            .collect();

        let settings = MeasureSettings::default();
        let layout = SyntheticLayout::single();
        let loo = measure_sample_layout_with(
            &sample,
            &settings,
            &layout,
            AttributionDesign::LeaveOneOutV2,
        )
        .unwrap()
        .unwrap();
        let singleton =
            measure_sample_layout_with(&sample, &settings, &layout, AttributionDesign::SingletonV1)
                .unwrap()
                .unwrap();

        let share = |m: &MeasuredEncoding, name: &str| {
            m.per_column
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("column {name} missing"))
                .share
        };
        let loo_pair = share(&loo, "dup_a") + share(&loo, "dup_b");
        let singleton_pair = share(&singleton, "dup_a") + share(&singleton, "dup_b");
        assert!(
            singleton_pair > 0.4,
            "guard: the singleton proxy must over-charge the duplicated pair, got \
             {singleton_pair:.4}"
        );
        assert!(
            loo_pair < singleton_pair / 2.0,
            "leave-one-out pair share {loo_pair:.4} must be far below the proxy's \
             {singleton_pair:.4}"
        );
    }

    // ------------------------------------------------------------------
    // MO-2: dispersion published with every share
    // ------------------------------------------------------------------

    #[test]
    fn single_tile_measurement_publishes_zero_stderr() {
        // One synthetic tile carries NO dispersion evidence, so the honest
        // stderr is 0.0 — which makes every consumer's `share − 2·stderr` gate
        // behave exactly as it did before MO-2.
        let measured = measure_sample(&point_sample(300), &MeasureSettings::default())
            .unwrap()
            .unwrap();
        assert_eq!(measured.tiles, 1);
        assert!(measured.per_column.iter().all(|c| c.share_stderr == 0.0));
    }

    #[test]
    fn multi_tile_measurement_publishes_finite_positive_stderr() {
        // Every feature of the sample is encoded, so this is an "exhaustive"
        // measurement — and the stderr is still finite and non-zero, because it
        // describes tile-to-tile dispersion of the share, not how much of the
        // sample was missed.
        let sample = point_sample(800);
        let layout = SyntheticLayout::from_density(&density_with(12, 100, 200, 1_000));
        let measured = measure_sample_layout(&sample, &MeasureSettings::default(), &layout)
            .unwrap()
            .unwrap();
        assert_eq!(measured.tiles, 8);
        for c in &measured.per_column {
            assert!(
                c.share_stderr.is_finite() && c.share_stderr >= 0.0,
                "column {} stderr {}",
                c.name,
                c.share_stderr
            );
            // A published spread must never swamp the share it qualifies.
            assert!(
                c.share_stderr < 0.5,
                "column {} stderr {} is implausibly wide",
                c.name,
                c.share_stderr
            );
        }
        // The layout deliberately spans light and busy tiles, so at least one
        // column's share genuinely moves between them.
        assert!(
            measured.per_column.iter().any(|c| c.share_stderr > 0.0),
            "a mixed-occupancy layout must expose SOME dispersion: {:?}",
            measured.per_column
        );
    }

    #[test]
    fn attribution_fields_are_deterministic_and_serde_defaulted() {
        let sample = point_sample(400);
        let layout = SyntheticLayout::from_density(&density_with(12, 60, 90, 400));
        let a = measure_sample_layout(&sample, &MeasureSettings::default(), &layout).unwrap();
        let b = measure_sample_layout(&sample, &MeasureSettings::default(), &layout).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
            "stderr + marginal math must not depend on iteration order"
        );

        // Pre-MO-2/MO-3 reports (no marginal_bytes / share_stderr keys) still
        // deserialize, as singleton costs with no published dispersion.
        let json = r#"{"features":10,"geometry_kind":"point","bytes_total":100,
            "bytes_per_feature":10.0,"zstd_ratio":3.0,"tiles":1,
            "per_column":[{"name":"geometry","compressed_bytes":60,"share":0.6}]}"#;
        let back: MeasuredEncoding = serde_json::from_str(json).unwrap();
        assert_eq!(back.per_column[0].marginal_bytes, 0);
        assert_eq!(back.per_column[0].share_stderr, 0.0);
    }
}
