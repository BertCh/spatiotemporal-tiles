//! Pass 1 — the dataset-global statistics scan (mechanism M2).
//!
//! # Why a second pass exists at all
//!
//! Every encoding verdict the tile encoder makes today is made from ONE TILE's
//! sample, because one tile is all it can see. Five separate defects fall out of
//! that single cause:
//!
//! * the numeric-property **affine** (`o`, `s`) is fitted to the tile's own
//!   `[min, max]`, so the same source value decodes to a different number in
//!   different tiles;
//! * the **dictionary-vs-`Utf8`** verdict is a per-tile size comparison, so one
//!   categorical column is a dictionary in the dense tiles and a `Utf8` in the
//!   sparse ones — forking a PROPS schema template per variant;
//! * the exact-integer path's **leaf width** (`UInt16` vs `Int32`) follows the
//!   tile's span, forking a template again;
//! * each tile ships its **own copy of the category list**, which measured 43.9 %
//!   of uncompressed tile bytes on `earthquakes-v2`;
//! * synthetic **row-index ids** are per-tile, so an id column is neither dense
//!   nor monotone across the dataset and cannot be delta-coded.
//!
//! All five want the same missing thing: the column's DATASET domain. This
//! module computes it once, in one scan, and hands it to every consumer — the
//! "one investment, several customers" shape. The conformance-invariance rule
//! (spec §13.2) is what it buys: an encoding verdict must be a function of the
//! dataset's domain, never of which rows happened to land in a tile or of the
//! order tiles were produced in.
//!
//! # Where it sits, and what it costs
//!
//! Between `source.load(...)` and `generate_tiles*` — no new I/O, no second read
//! of the source. The build already materialises every feature in memory (all
//! three input backends return an owned `Vec<ParsedFeature>`, and `--streaming`
//! holds the same vector), and it already runs one whole-dataset property scan
//! there: `resolve_property_types` → [`fill_property_type_gaps`]. This is that
//! scan widened, not a new one bolted alongside.
//!
//! [`fill_property_type_gaps`]: crate::columnar::fill_property_type_gaps
//!
//! **Memory is O(columns + capped category sets), never O(rows).** That
//! constraint is architectural, not advisory: the reason properties are held
//! columnar at all is that the previous per-row `serde_json::Map` cost ~830 B
//! *even for a single key* (a `BTreeMap` allocates a whole leaf node), and
//! removing it took a 4 M-row build from 11.63 GB to 6.71 GB. A statistics pass
//! that cloned values into owned per-row collections would hand that back. So
//! the scan **borrows**: numeric cells are read as `f64` off the columnar cell,
//! string cells are borrowed as `&str`, and the only allocation is the first
//! insert of a genuinely new category. Number- and bool-backed categories format
//! into one reused scratch buffer.
//!
//! The two caps are explicit and their overflow behaviour is deterministic:
//!
//! | cap | default | on overflow |
//! |---|---|---|
//! | distinct categories per column | [`MAX_CATEGORIES`] (65 536 — the `UInt16` dictionary key space) | the set is DROPPED and `distinct_overflow` is set; the column is forced to `Utf8` everywhere |
//! | retained category bytes per column | [`DEFAULT_CATEGORY_BYTE_CAP`] (64 MiB) | same |
//!
//! Overflow is *sticky* and is a property of the DATASET, not of the chunking:
//! a chunk that overflows has strictly more distinct values than the cap, and a
//! merge that crosses the cap means the union does. Both directions are proved
//! by the chunk-invariance test below, which is why the caps do not reintroduce
//! sample dependence.
//!
//! # Determinism
//!
//! The scan is a parallel fold over ORDERED feature chunks with an associative
//! merge, so the result does not depend on `--workers`, on the rayon pool's
//! split points, or on which thread finished first:
//!
//! * every numeric statistic is an order-independent fold (`min`, `max`, `sum`,
//!   logical AND);
//! * per-zoom byte mass is a difference array of sums;
//! * the timestamp histogram is a vector of counts;
//! * category sets merge left-chunk-first, so **first-seen order is first in the
//!   dataset's own feature order** — never "first in whichever thread got there
//!   first". First-seen dictionary order is a pinned register entry; this widens
//!   it from per-tile to per-dataset without changing the rule.
//!
//! Every output-affecting map here is a `BTreeMap`. There is no `HashMap`, no
//! RNG, and no wall clock in this module.
//!
//! # The forward contract for a true streaming input
//!
//! Today every backend materialises, so one resident scan suffices. A future
//! genuinely-streaming input (the planned SedonaDB backend) cannot re-read the
//! feature vector, and has exactly two conforming options:
//!
//! 1. a **second source scan under a stable `ORDER BY`** — required, because
//!    first-seen category order is only well-defined against a fixed input
//!    order; or
//! 2. a **bounded property-stream spill**, i.e. tee the property columns to disk
//!    on the first pass and re-read them for the second.
//!
//! What is NOT conforming is deriving the pins from a prefix or a sample of the
//! stream: that is the sample-dependent type decision the design register
//! rejects, and it reintroduces exactly the drift this module removes.

use std::collections::BTreeMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::columnar::{
    prop_value_as_f64, with_category, AttributeFilter, PropertyKind, PropertyTypes,
};
use crate::input::ParsedFeature;
use stt_core::arrow_tile::{
    dataset_dictionary_is_smaller, AttrPinned, GlobalColumnPins, GlobalDictVerdict,
};

/// Distinct-category cap per column: the `UInt16` dictionary key space. A
/// column with more distinct values than this cannot be dictionary-encoded at
/// all (the encoder's own key type would overflow), so retaining the set past
/// the cap would buy nothing and cost unbounded memory.
pub const MAX_CATEGORIES: usize = 65_536;

/// Retained-category BYTE cap per column, on top of [`MAX_CATEGORIES`]. Guards
/// the pathological shape the count cap misses — 65 536 distinct values that
/// happen to be kilobyte blobs.
pub const DEFAULT_CATEGORY_BYTE_CAP: u64 = 64 * 1024 * 1024;

/// Largest category COUNT a column may have and still be pinned to a hoisted
/// dictionary — the resident-memory half of the verdict.
///
/// # Why a hoist cap exists at all
///
/// "It fits under the `UInt16` key space" was the whole test before this
/// constant, and it is the wrong test. A hoisted dictionary is stored ONCE on
/// the wire (it moves into `manifest.schemas` as part of the PROPS template),
/// but the reader pays for it ONCE PER RESIDENT TILE, three times over:
///
/// 1. `spliceIpc` concatenates `template + tail` into a fresh buffer per tile,
///    so the template's dictionary bytes are copied into every tile's stream;
/// 2. Arrow decodes that DictionaryBatch per tile, giving every tile its own
///    value buffer; and
/// 3. the reader materialises `categoricalProps[col].categories` — a JS
///    `string[]` — per tile.
///
/// None of the three is removable by the reader: (1) and (2) are inherent to
/// splicing a template onto a tail, and only (3) can be shared by identity.
/// So an unbounded hoist makes resident client memory grow as
/// `resident_tiles × category_bytes`, which is a **constraint-spine
/// violation** (bounded client memory), not a size/speed trade.
///
/// # Where the numbers come from (MEASURED, 2026-08-11)
///
/// A 380 007-feature OSM changeset build with a 14 653-category free-text
/// `user` column: 400 resident tiles cost **264.4 MB** of heap with the column
/// hoisted versus **6.7 MB** with it left per-tile — 39x, on tiles holding 408
/// features between them. Decomposed with the same harness: 82.2 MB spliced
/// IPC + 80.6 MB Arrow dictionary buffers + ~115 MB of JS category strings.
/// That is ~661 KB of realized heap per tile for a 151 KB category list, i.e.
/// roughly `24·K + 2·C` bytes (K = categories, C = category UTF-8 bytes).
///
/// [`MAX_HOISTED_CATEGORIES`] and [`MAX_HOISTED_CATEGORY_BYTES`] jointly bound
/// that at `24·1024 + 2·4096 ≈ 32 KiB` of realized heap per tile per hoisted
/// column. Against `@poopdeck.gl/core`'s DEFAULT resident tile budget —
/// `SpatioTemporalTileset`'s `maxCacheSize = 2000` — that is ≤ 64 MiB, i.e.
/// ~3 % of the matching default `maxCacheByteSize` of 2 GiB. The caps are
/// derived from the reader's own declared budget, not chosen by feel.
///
/// A column over either cap still gets a dataset-global *verdict*
/// ([`GlobalDictVerdict::Utf8`]) — one Arrow type in every tile — so TB-3's
/// template-fork collapse is fully preserved. Only TB-4's hoist is declined.
pub const MAX_HOISTED_CATEGORIES: usize = 1024;

/// Largest total category-UTF-8-byte size a column may have and still be
/// pinned to a hoisted dictionary. See [`MAX_HOISTED_CATEGORIES`] for the
/// derivation — this is the second half of the same bound, and it is the one
/// that catches "few categories, but each of them a paragraph".
pub const MAX_HOISTED_CATEGORY_BYTES: u64 = 4096;

/// Bins in the dataset timestamp histogram. Fixed-width over the observed
/// `[min, max]`, so the quantiles derived from it are integer arithmetic and
/// therefore reproducible bit-for-bit.
pub const DEFAULT_TIMESTAMP_BINS: usize = 4096;

/// Features per parallel fold chunk. Affects only how the work is split — the
/// result is chunk-invariant by construction (and by test).
pub const DEFAULT_STATS_CHUNK: usize = 8_192;

/// Per-feature payload estimate: 16 B per coordinate pair, 16 B per property,
/// 32 B of metadata overhead. Deliberately the SAME arithmetic
/// `tiler::tile_feature_size` applies per tile feature (and that the byte cap in
/// `stt_core::budget` is calibrated against), so pass 1's estimate and the
/// tiler's are comparable numbers rather than two unrelated models.
fn feature_byte_estimate(vertices: usize, properties: usize) -> u64 {
    (vertices as u64) * 16 + (properties as u64) * 16 + 32
}

/// Count the vertices in a GeoJSON feature's geometry.
///
/// Mirrors `tiler::geojson_vertex_count`, which is private to that module —
/// duplicated rather than re-exported because `tiler.rs` is not this item's to
/// edit. The two must agree: the shared consequence is that `b̂(z)` and the
/// tiler's own per-tile size estimate are on the same scale. A geometry-less
/// feature counts as 1, matching the tiler.
fn geojson_vertex_count(f: &geojson::Feature) -> usize {
    use geojson::Value as G;
    let Some(geom) = f.geometry.as_ref() else {
        return 1;
    };
    match &geom.value {
        G::Point(_) => 1,
        G::MultiPoint(pts) => pts.len(),
        G::LineString(c) => c.len(),
        G::MultiLineString(lines) => lines.iter().map(|l| l.len()).sum(),
        G::Polygon(rings) => rings.iter().map(|r| r.len()).sum(),
        G::MultiPolygon(polys) => polys.iter().flatten().map(|r| r.len()).sum(),
        G::GeometryCollection(_) => 1,
    }
}

// ----------------------------------------------------------------------------
// Output types
// ----------------------------------------------------------------------------

/// Dataset-wide statistics for one NUMERIC property column.
///
/// Exactly the statistics `quantize::numeric_column_stats` computes per tile
/// today, widened from a tile's sample to the column's whole domain — the same
/// five numbers, so the pin derived from them can reproduce the incumbent rule
/// tree branch for branch instead of inventing a new one.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericColStats {
    /// Smallest finite value, or `+inf` when the column has none.
    pub min: f64,
    /// Largest finite value, or `-inf` when the column has none.
    pub max: f64,
    /// Largest `|value|` over the finite values (`0.0` when there are none).
    /// The magnitude-refusal test reads this and nothing else.
    pub max_abs: f64,
    /// Every finite value is integer-valued (`v == v.trunc()`). Vacuously true
    /// for a column with no finite values.
    pub all_integer: bool,
    /// Cells holding a finite value.
    pub finite: u64,
    /// Cells holding no finite value — key absent, JSON null, or a non-finite
    /// float. Invariant: `finite + nulls == DatasetStats::features`.
    pub nulls: u64,
}

impl NumericColStats {
    /// The dataset span, or `0.0` when nothing finite was seen.
    pub fn span(&self) -> f64 {
        if self.finite == 0 {
            0.0
        } else {
            self.max - self.min
        }
    }
}

/// Dataset-wide statistics for one CATEGORICAL property column.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalColStats {
    /// Distinct categories in dataset-wide FIRST-SEEN order, or `None` when a
    /// cap overflowed. First-seen order is the pinned register rule, widened
    /// from per-tile to per-dataset — not re-decided.
    pub categories: Option<Vec<String>>,
    /// A cap overflowed: no global category list exists, so the column is
    /// forced to plain `Utf8` in every tile.
    pub distinct_overflow: bool,
    /// Total UTF-8 bytes of every non-null cell (NOT of the distinct set) —
    /// the `Utf8` side of the dataset-scale size comparison.
    pub total_value_bytes: u64,
    /// Non-null cells.
    pub values: u64,
}

impl CategoricalColStats {
    /// Distinct categories, or `None` when the set overflowed.
    pub fn distinct(&self) -> Option<usize> {
        self.categories.as_ref().map(Vec::len)
    }

    /// Total UTF-8 bytes of the retained distinct set (the dictionary side of
    /// the size comparison). `None` when the set overflowed.
    pub fn category_bytes(&self) -> Option<u64> {
        self.categories
            .as_ref()
            .map(|c| c.iter().map(|s| s.len() as u64).sum())
    }
}

/// Estimated uncompressed payload mass, dataset-wide and per zoom.
///
/// `per_zoom[z - zoom_range.0]` is `b̂(z)`: the estimated payload bytes the
/// features visible at zoom `z` contribute. "Visible" is the only per-zoom term
/// pass 1 can honestly evaluate — a feature's `min_zoom_field` /
/// `max_zoom_field` band, which is exactly how a baked LOD floor confines a
/// feature. Clip fan-out and line simplification are NOT modelled: both are
/// per-tile outcomes that would require running the clipper at every zoom, and
/// guessing them would be an analytic size model in place of measurement.
///
/// So `b̂(z)` is exact for point datasets with no zoom band, exact-up-to-clipping
/// for banded ones, and a lower bound for clipped trajectory layers. Consumers
/// must use it as a RELATIVE per-zoom weight, not as an absolute byte prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct ByteMassStats {
    /// `(min_zoom, max_zoom)` the per-zoom vector covers, inclusive.
    pub zoom_range: (u8, u8),
    /// Σ over every feature of its payload estimate, independent of zoom.
    pub total_feature_bytes: u64,
    /// `b̂(z)` for `z` in `zoom_range`, indexed from `zoom_range.0`.
    pub per_zoom: Vec<u64>,
}

impl ByteMassStats {
    /// `b̂(z)`, or `None` when `z` is outside the covered range.
    pub fn at_zoom(&self, z: u8) -> Option<u64> {
        if z < self.zoom_range.0 || z > self.zoom_range.1 {
            return None;
        }
        self.per_zoom.get((z - self.zoom_range.0) as usize).copied()
    }
}

/// The dataset's feature-start-time distribution, as a fixed-width histogram.
///
/// A histogram rather than the raw timestamps because the raw timestamps are
/// O(rows) and this module may not be. Fixed-width over the observed
/// `[min, max]` with integer bin arithmetic, so every derived quantile is
/// reproducible bit-for-bit and independent of chunking.
#[derive(Debug, Clone, PartialEq)]
pub struct TimestampStats {
    /// Earliest feature start, or `None` for an empty dataset.
    pub min: Option<u64>,
    /// Latest feature start, or `None` for an empty dataset.
    pub max: Option<u64>,
    /// Features counted.
    pub count: u64,
    /// Per-bin counts over `[min, max]`.
    pub bins: Vec<u64>,
}

impl TimestampStats {
    /// The `q`-quantile of the feature start times (`q` clamped to `[0, 1]`),
    /// resolved to the LOWER EDGE of the bin the quantile falls in.
    ///
    /// Bin-edge resolution rather than interpolation is deliberate: an
    /// interpolated boundary is not a value any feature has, and a temporal
    /// boundary that no feature sits on is a boundary the bucketer cannot snap
    /// to. `None` for an empty dataset.
    pub fn quantile(&self, q: f64) -> Option<u64> {
        let (min, max) = (self.min?, self.max?);
        if self.count == 0 {
            return None;
        }
        if max == min || self.bins.len() <= 1 {
            return Some(min);
        }
        let q = q.clamp(0.0, 1.0);
        // Ceil so q=0 lands on the first bin holding anything and q=1 on the
        // last; integer target, so the walk below is exact.
        let target = ((q * self.count as f64).ceil() as u64).clamp(1, self.count);
        let mut seen = 0u64;
        for (i, n) in self.bins.iter().enumerate() {
            seen += *n;
            if seen >= target {
                return Some(self.bin_lower_edge(i));
            }
        }
        Some(max)
    }

    /// `n + 1` evenly-spaced quantiles from `q = 0` to `q = 1` — the candidate
    /// boundary set an adaptive temporal partitioner searches over.
    pub fn quantiles(&self, n: usize) -> Vec<u64> {
        if n == 0 {
            return Vec::new();
        }
        (0..=n)
            .filter_map(|i| self.quantile(i as f64 / n as f64))
            .collect()
    }

    /// Lower timestamp edge of bin `i`.
    fn bin_lower_edge(&self, i: usize) -> u64 {
        let (Some(min), Some(max)) = (self.min, self.max) else {
            return 0;
        };
        if self.bins.len() <= 1 || max <= min {
            return min;
        }
        let span = (max - min) as u128 + 1;
        min + ((i as u128 * span) / self.bins.len() as u128) as u64
    }
}

/// Everything pass 1 knows about the dataset.
///
/// Consumed by the whole R1 chain — the global attribute-range pin, the global
/// dict-vs-`Utf8` verdict, the global dictionary hoist, dense id renumbering,
/// the schema-template cost term, the temporal-LOD zoom cutoffs and the adaptive
/// temporal partition all read from here rather than each running their own
/// scan.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetStats {
    /// Features scanned.
    pub features: u64,
    /// Per numeric property column, in sorted name order.
    pub numeric: BTreeMap<String, NumericColStats>,
    /// Per categorical property column, in sorted name order.
    pub categorical: BTreeMap<String, CategoricalColStats>,
    /// Estimated payload mass, dataset-wide and per zoom.
    pub byte_mass: ByteMassStats,
    /// Feature start-time distribution.
    pub timestamps: TimestampStats,
    /// The distinct-category cap this scan ran under.
    pub category_cap: usize,
    /// The category-byte cap this scan ran under.
    pub category_byte_cap: u64,
}

impl DatasetStats {
    /// Resolve the dataset-global encoder pins.
    ///
    /// Numeric columns get the AUTO-quantization affine derived from the whole
    /// domain ([`AttrPinned::derive_auto`] — the incumbent rule tree, evaluated
    /// over the dataset instead of over one tile). Categorical columns get a
    /// dictionary pin against the global first-seen category list, or `Utf8`
    /// when a cap overflowed, when the dictionary is not the smaller wire form,
    /// or when the list is too large to hoist.
    ///
    /// An encode that never reaches these pins (`--single-pass`, or any caller
    /// that leaves `EncoderConfig::global_pins` unset) takes the incumbent
    /// per-tile path and is byte-identical to a pre-M2 encode. That is the
    /// documented rollback, and it stays in the code.
    ///
    /// The dictionary side pins the LIST **and** a verdict: `Dictionary` only
    /// when the dictionary is measurably the smaller form on the wire AND its
    /// hoisted copy is small enough to be replicated into every resident tile
    /// on the reader. See [`HoistPolicy`] and
    /// [`DatasetStats::categorical_verdicts`] for the two gates and why "it
    /// fits under the `UInt16` cap" was never a sufficient test.
    pub fn to_pins(&self) -> GlobalColumnPins {
        self.to_pins_with(&HoistPolicy::default())
    }

    /// [`DatasetStats::to_pins`] with an explicit hoist policy.
    ///
    /// The rollback path for this item: [`HoistPolicy::unbounded`] restores the
    /// pre-fix behaviour (pin `Dictionary` for every column whose distinct set
    /// fits the key space, regardless of size), and `--single-pass` remains the
    /// whole-mechanism escape hatch.
    pub fn to_pins_with(&self, policy: &HoistPolicy) -> GlobalColumnPins {
        let mut attr = BTreeMap::new();
        for (name, st) in &self.numeric {
            attr.insert(
                name.clone(),
                AttrPinned::derive_auto(st.min, st.max, st.max_abs, st.all_integer, st.finite),
            );
        }
        let mut dict = BTreeMap::new();
        for (name, st) in &self.categorical {
            let verdict = match categorical_verdict(st, policy) {
                CategoricalVerdict::Dictionary => GlobalDictVerdict::Dictionary(Arc::new(
                    st.categories
                        .as_ref()
                        .expect("Dictionary implies a set")
                        .clone(),
                )),
                _ => GlobalDictVerdict::Utf8,
            };
            dict.insert(name.clone(), verdict);
        }
        GlobalColumnPins { attr, dict }
    }

    /// Why each categorical column got the verdict it did, in sorted name
    /// order — the diagnostic sibling of [`DatasetStats::to_pins_with`].
    ///
    /// Exists so a build can say *out loud* that it declined to hoist a column
    /// and why: a silent demotion to `Utf8` is exactly the kind of invisible
    /// policy that made the unbounded hoist ship in the first place. Wiring it
    /// into the CLI's pass-1 summary needs a line in `stt-build.rs`, which this
    /// item does not own — recorded as a cross-item dependency.
    pub fn categorical_verdicts(&self, policy: &HoistPolicy) -> Vec<(&str, CategoricalVerdict)> {
        self.categorical
            .iter()
            .map(|(name, st)| (name.as_str(), categorical_verdict(st, policy)))
            .collect()
    }

    /// Columns whose global magnitude forces a `Float64` refusal, and the
    /// `max_abs` that forced it — the loud warning pass 1 can now emit that a
    /// per-tile encoder could not.
    pub fn refused_numeric_columns(&self) -> Vec<(&str, f64)> {
        self.numeric
            .iter()
            .filter(|(_, st)| {
                AttrPinned::derive_auto(st.min, st.max, st.max_abs, st.all_integer, st.finite)
                    .refuse
            })
            .map(|(name, st)| (name.as_str(), st.max_abs))
            .collect()
    }

    /// Columns whose distinct set overflowed a cap, in sorted name order.
    pub fn overflowed_categorical_columns(&self) -> Vec<&str> {
        self.categorical
            .iter()
            .filter(|(_, st)| st.distinct_overflow)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

// ----------------------------------------------------------------------------
// The dict-vs-Utf8 verdict (TB-3), and the hoist's resident-memory gate
// ----------------------------------------------------------------------------

/// How large a hoisted dictionary this build is willing to replicate into every
/// resident tile on the READER.
///
/// The hoist's cost is asymmetric and that asymmetry is the whole reason this
/// type exists: the category list is written ONCE (into `manifest.schemas`) but
/// is resident ONCE PER TILE in the client, because `spliceIpc` concatenates
/// the template onto every tile's tail before Arrow ever sees it. "Bounded
/// client memory" is a constraint of this program, not an objective to trade
/// against wire bytes, so the size that may be hoisted is capped and the cap is
/// derived from the reader's own declared budget. See
/// [`MAX_HOISTED_CATEGORIES`] for the measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoistPolicy {
    /// Category-count ceiling; `usize::MAX` disables this half.
    pub max_categories: usize,
    /// Category-UTF-8-byte ceiling; `u64::MAX` disables this half.
    pub max_category_bytes: u64,
    /// Apply the dataset-scale wire surrogate
    /// ([`dataset_dictionary_is_smaller`]) as well. On by default; off restores
    /// "pin whatever fits the key space".
    pub require_wire_win: bool,
}

impl Default for HoistPolicy {
    fn default() -> Self {
        Self {
            max_categories: MAX_HOISTED_CATEGORIES,
            max_category_bytes: MAX_HOISTED_CATEGORY_BYTES,
            require_wire_win: true,
        }
    }
}

impl HoistPolicy {
    /// The pre-fix behaviour, kept as the documented fallback: pin `Dictionary`
    /// for every column whose distinct set fits the `UInt16` key space, with no
    /// size test on either side of the wire. Reproduces the bytes (and the
    /// client memory profile) of the first M2 landing exactly.
    pub fn unbounded() -> Self {
        Self {
            max_categories: usize::MAX,
            max_category_bytes: u64::MAX,
            require_wire_win: false,
        }
    }
}

/// One categorical column's dataset-global verdict, with the reason attached.
///
/// Every arm other than [`CategoricalVerdict::Dictionary`] resolves to
/// `GlobalDictVerdict::Utf8` — one Arrow type in every tile either way, so the
/// §13.2 conformance-invariance rule and TB-3's template-fork collapse hold on
/// all four. The arms differ only in what a build should TELL the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoricalVerdict {
    /// Pin the global list and hoist it into the schema template.
    Dictionary,
    /// A pass-1 cap overflowed, so no global list exists at all.
    Overflowed,
    /// The dataset-scale surrogate says plain `Utf8` is the smaller wire form
    /// (a near-unique column: the keys cost more than the values save).
    WireLoses,
    /// The dictionary IS smaller on the wire, but the list is too large to
    /// replicate into every resident tile on the reader. Carries `(categories,
    /// category_bytes)` so the message can name the numbers.
    TooLargeToHoist(usize, u64),
}

/// The verdict rule, in one place so [`DatasetStats::to_pins_with`] and
/// [`DatasetStats::categorical_verdicts`] cannot drift apart.
///
/// Order matters: overflow first (no list to measure), then the wire test, then
/// the resident test. Every input is a DATASET total, so the verdict is a
/// function of the dataset's domain and no property of any one tile can reach
/// it — the invariance rule the pin exists to enforce.
fn categorical_verdict(st: &CategoricalColStats, policy: &HoistPolicy) -> CategoricalVerdict {
    let Some(categories) = st.categories.as_ref().filter(|_| !st.distinct_overflow) else {
        return CategoricalVerdict::Overflowed;
    };
    let k = categories.len();
    let category_bytes = categories.iter().map(|s| s.len() as u64).sum::<u64>();
    // `st.values` counts NON-NULL cells while the surrogate's `values`
    // parameter is rows. Under-counting rows shrinks the `Utf8` side's offset
    // term faster than the dictionary's key term, so the substitution can only
    // ever move the verdict TOWARDS `Utf8` — conservative in the direction that
    // costs wire bytes rather than client memory, which is the safe direction
    // for a constraint that is not negotiable.
    if policy.require_wire_win
        && !dataset_dictionary_is_smaller(st.total_value_bytes, st.values, category_bytes, k as u64)
    {
        return CategoricalVerdict::WireLoses;
    }
    if k > policy.max_categories || category_bytes > policy.max_category_bytes {
        return CategoricalVerdict::TooLargeToHoist(k, category_bytes);
    }
    CategoricalVerdict::Dictionary
}

// ----------------------------------------------------------------------------
// Options
// ----------------------------------------------------------------------------

/// Knobs for [`collect_dataset_stats_with`]. Every field has a default that
/// reproduces [`collect_dataset_stats`].
#[derive(Debug, Clone)]
pub struct StatsOptions {
    /// Inclusive zoom range `b̂(z)` is computed over.
    pub zoom_range: (u8, u8),
    /// Property holding a feature's LOD floor (`--min-zoom-field`), if any.
    pub min_zoom_field: Option<String>,
    /// Property holding a feature's LOD ceiling (`--max-zoom-field`), if any.
    pub max_zoom_field: Option<String>,
    /// Distinct-category cap per column.
    pub category_cap: usize,
    /// Retained-category byte cap per column.
    pub category_byte_cap: u64,
    /// Timestamp histogram bins.
    pub timestamp_bins: usize,
    /// Features per parallel fold chunk. Result-invariant; affects only the
    /// work split (and is varied by the chunk-invariance test).
    pub chunk: usize,
}

impl Default for StatsOptions {
    fn default() -> Self {
        Self {
            zoom_range: (0, 14),
            min_zoom_field: None,
            max_zoom_field: None,
            category_cap: MAX_CATEGORIES,
            category_byte_cap: DEFAULT_CATEGORY_BYTE_CAP,
            timestamp_bins: DEFAULT_TIMESTAMP_BINS,
            chunk: DEFAULT_STATS_CHUNK,
        }
    }
}

impl StatsOptions {
    /// Zoom span the per-zoom vector covers (always ≥ 1 entry).
    fn zoom_len(&self) -> usize {
        let (lo, hi) = self.zoom_range;
        if hi < lo {
            1
        } else {
            (hi - lo) as usize + 1
        }
    }
}

// ----------------------------------------------------------------------------
// The scan
// ----------------------------------------------------------------------------

/// Pass 1 with default options.
///
/// `types` is the RESOLVED dataset-wide property-kind map — the input schema's
/// answers plus the gaps filled by the existing whole-dataset inference pass. A
/// key absent from it gets no statistics and therefore no pin, and its column
/// keeps the incumbent per-tile behaviour: that is the fallback that makes every
/// consumer additive. `filter` is the same [`AttributeFilter`] the tile builder
/// applies, so a property the build drops never reaches the statistics either.
pub fn collect_dataset_stats(
    features: &[ParsedFeature],
    filter: &AttributeFilter,
    types: &PropertyTypes,
) -> DatasetStats {
    collect_dataset_stats_with(features, filter, types, &StatsOptions::default())
}

/// Pass 1 with explicit options.
///
/// Two iterations over the resident feature slice, both parallel:
///
/// 1. the main fold — geometry vertex count, property cells, zoom band,
///    timestamp min/max;
/// 2. a timestamp-only fold that fills the histogram (it needs the range the
///    first pass establishes). It reads one `u64` field per feature and touches
///    no properties, so it is a small constant beside pass 1 proper.
pub fn collect_dataset_stats_with(
    features: &[ParsedFeature],
    filter: &AttributeFilter,
    types: &PropertyTypes,
    opts: &StatsOptions,
) -> DatasetStats {
    let schema = ColumnSchema::resolve(filter, types);
    let chunk = opts.chunk.max(1);

    let acc = features
        .par_chunks(chunk)
        .map(|chunk| {
            let mut acc = ScanAcc::new(&schema, opts);
            for feature in chunk {
                acc.observe(feature, opts);
            }
            acc
        })
        .reduce(|| ScanAcc::new(&schema, opts), |a, b| a.merge(b, opts));

    let timestamps = collect_timestamp_histogram(features, acc.ts_min, acc.ts_max, opts, chunk);
    acc.finish(opts, timestamps)
}

/// The timestamp histogram fold (step 2 above).
fn collect_timestamp_histogram(
    features: &[ParsedFeature],
    min: Option<u64>,
    max: Option<u64>,
    opts: &StatsOptions,
    chunk: usize,
) -> TimestampStats {
    let bins = opts.timestamp_bins.max(1);
    let (Some(min), Some(max)) = (min, max) else {
        return TimestampStats {
            min: None,
            max: None,
            count: 0,
            bins: vec![0; bins],
        };
    };
    let counts = features
        .par_chunks(chunk)
        .map(|chunk| {
            let mut local = vec![0u64; bins];
            for f in chunk {
                local[timestamp_bin(f.timestamp, min, max, bins)] += 1;
            }
            local
        })
        .reduce(
            || vec![0u64; bins],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );
    TimestampStats {
        min: Some(min),
        max: Some(max),
        count: features.len() as u64,
        bins: counts,
    }
}

/// Which fixed-width bin a timestamp falls in. Integer arithmetic throughout,
/// so the binning is reproducible and independent of evaluation order.
fn timestamp_bin(t: u64, min: u64, max: u64, bins: usize) -> usize {
    if bins <= 1 || max <= min {
        return 0;
    }
    let t = t.clamp(min, max);
    let span = (max - min) as u128 + 1;
    let idx = ((t - min) as u128 * bins as u128) / span;
    (idx as usize).min(bins - 1)
}

/// The columns the scan tracks, resolved once from the filter + kind map so the
/// per-row loop does no filtering work of its own.
struct ColumnSchema {
    /// Numeric column names, sorted.
    numeric: Vec<String>,
    /// Categorical column names, sorted.
    categorical: Vec<String>,
}

impl ColumnSchema {
    fn resolve(filter: &AttributeFilter, types: &PropertyTypes) -> Self {
        let mut numeric = Vec::new();
        let mut categorical = Vec::new();
        for (name, kind) in types {
            // A property the build drops never reaches a tile, so it must not
            // reach the statistics — otherwise a pin would be derived for a
            // column that does not exist in the archive.
            if !filter.keeps(name) {
                continue;
            }
            match kind {
                PropertyKind::Numeric => numeric.push(name.clone()),
                PropertyKind::Categorical => categorical.push(name.clone()),
            }
        }
        Self {
            numeric,
            categorical,
        }
    }
}

/// One chunk's running statistics.
///
/// Held as `BTreeMap`s keyed by column name, pre-seeded from the schema so the
/// per-row loop never allocates a key. Sizes are O(columns), plus the capped
/// category sets.
struct ScanAcc {
    features: u64,
    numeric: BTreeMap<String, NumAcc>,
    categorical: BTreeMap<String, CatAcc>,
    total_feature_bytes: u64,
    /// Difference array over the zoom range: `+bytes` at the feature's first
    /// visible zoom, `-bytes` one past its last. Prefix-summed at finish, so
    /// the per-feature cost is O(1) rather than O(zooms).
    zoom_delta: Vec<i64>,
    ts_min: Option<u64>,
    ts_max: Option<u64>,
    /// Reused formatting buffer for number/bool categorical cells — the reason
    /// the scan does not allocate per row.
    scratch: String,
}

/// Running numeric statistics for one column.
#[derive(Clone)]
struct NumAcc {
    min: f64,
    max: f64,
    max_abs: f64,
    all_integer: bool,
    finite: u64,
}

impl NumAcc {
    fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            max_abs: 0.0,
            all_integer: true,
            finite: 0,
        }
    }

    fn push(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        self.finite += 1;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
        self.max_abs = self.max_abs.max(v.abs());
        self.all_integer &= v == v.trunc();
    }

    fn merge(&mut self, other: &NumAcc) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.max_abs = self.max_abs.max(other.max_abs);
        self.all_integer &= other.all_integer;
        self.finite += other.finite;
    }
}

/// Running categorical statistics for one column.
///
/// The distinct set is `category → first-seen ordinal`, which stores each
/// string ONCE while still recovering first-seen order (the ordinals are dense,
/// so ordering is a bucket placement, not a sort).
#[derive(Clone)]
struct CatAcc {
    index: BTreeMap<String, u32>,
    /// Retained bytes of the distinct set (not of all values).
    index_bytes: u64,
    overflow: bool,
    total_value_bytes: u64,
    values: u64,
}

impl CatAcc {
    fn new() -> Self {
        Self {
            index: BTreeMap::new(),
            index_bytes: 0,
            overflow: false,
            total_value_bytes: 0,
            values: 0,
        }
    }

    /// Record one non-null cell. Allocates ONLY when `value` is a category this
    /// accumulator has not seen.
    fn push(&mut self, value: &str, opts: &StatsOptions) {
        self.values += 1;
        self.total_value_bytes += value.len() as u64;
        if self.overflow || self.index.contains_key(value) {
            return;
        }
        let ordinal = self.index.len() as u32;
        self.index.insert(value.to_string(), ordinal);
        self.index_bytes += value.len() as u64;
        self.check_caps(opts);
    }

    /// Drop the set once a cap is crossed. Sticky: an overflowed accumulator
    /// never re-populates, so the verdict cannot depend on what arrives after.
    fn check_caps(&mut self, opts: &StatsOptions) {
        if self.index.len() > opts.category_cap || self.index_bytes > opts.category_byte_cap {
            self.overflow = true;
            self.index = BTreeMap::new();
            self.index_bytes = 0;
        }
    }

    /// Distinct values in first-seen order. Dense ordinals ⇒ O(k) placement.
    fn ordered(&self) -> Vec<&str> {
        let mut out: Vec<&str> = vec![""; self.index.len()];
        for (value, ordinal) in &self.index {
            out[*ordinal as usize] = value.as_str();
        }
        out
    }

    /// Merge `other` (which covers the features immediately AFTER this
    /// accumulator's) into `self`, preserving left-first first-seen order.
    ///
    /// Associative, which is what makes the result independent of how rayon
    /// parenthesised the reduction: union-preserving-left-order is associative,
    /// the counters are sums, and the caps cannot disagree between
    /// parenthesisations because every intermediate union is a subset of the
    /// final one (so if the final union fits, no intermediate can overflow).
    fn merge(&mut self, other: CatAcc, opts: &StatsOptions) {
        self.values += other.values;
        self.total_value_bytes += other.total_value_bytes;
        if self.overflow || other.overflow {
            self.overflow = true;
            self.index = BTreeMap::new();
            self.index_bytes = 0;
            return;
        }
        for value in other.ordered() {
            if self.index.contains_key(value) {
                continue;
            }
            let ordinal = self.index.len() as u32;
            self.index.insert(value.to_string(), ordinal);
            self.index_bytes += value.len() as u64;
            self.check_caps(opts);
            if self.overflow {
                return;
            }
        }
    }
}

impl ScanAcc {
    fn new(schema: &ColumnSchema, opts: &StatsOptions) -> Self {
        Self {
            features: 0,
            numeric: schema
                .numeric
                .iter()
                .map(|n| (n.clone(), NumAcc::new()))
                .collect(),
            categorical: schema
                .categorical
                .iter()
                .map(|n| (n.clone(), CatAcc::new()))
                .collect(),
            total_feature_bytes: 0,
            // One extra slot so the difference array's closing `-bytes` always
            // has somewhere to land.
            zoom_delta: vec![0; opts.zoom_len() + 1],
            ts_min: None,
            ts_max: None,
            scratch: String::new(),
        }
    }

    /// Fold one feature in.
    ///
    /// A single pass over the feature's properties does all four jobs — numeric
    /// stats, categorical stats, the non-null property count for the byte
    /// estimate, and the zoom band — because `FeatureProperties::iter` returns a
    /// boxed iterator, so each extra traversal would cost an allocation per row.
    /// (That boxing is the one transient allocation this scan cannot avoid;
    /// removing it means changing `props.rs`, which is not this item's file.)
    fn observe(&mut self, feature: &ParsedFeature, opts: &StatsOptions) {
        self.features += 1;

        let mut property_count = 0usize;
        let mut feature_min_zoom: Option<u8> = None;
        let mut feature_max_zoom: Option<u8> = None;

        if let Some(props) = feature.shared_properties.as_ref() {
            for (key, value) in props.iter() {
                property_count += 1;

                // The zoom band, read exactly as `tiler::feature_zoom_bound`
                // reads it: real numbers only (a numeric-looking STRING is
                // deliberately not coerced there), rounded to the nearest zoom.
                if opts.min_zoom_field.as_deref() == Some(key) {
                    feature_min_zoom = value.as_f64().map(|z| z.round() as u8);
                }
                if opts.max_zoom_field.as_deref() == Some(key) {
                    feature_max_zoom = value.as_f64().map(|z| z.round() as u8);
                }

                if let Some(acc) = self.numeric.get_mut(key) {
                    if let Some(v) = prop_value_as_f64(value) {
                        acc.push(v);
                    }
                    continue;
                }
                if let Some(acc) = self.categorical.get_mut(key) {
                    with_category(value, &mut self.scratch, |s| acc.push(s, opts));
                }
            }
        }

        let bytes = feature_byte_estimate(geojson_vertex_count(&feature.geojson), property_count);
        self.total_feature_bytes += bytes;

        // Per-zoom mass: charge the feature to every zoom its band covers.
        let (lo_cfg, hi_cfg) = opts.zoom_range;
        let lo = feature_min_zoom.unwrap_or(lo_cfg).max(lo_cfg);
        let hi = feature_max_zoom.unwrap_or(hi_cfg).min(hi_cfg);
        if lo <= hi && hi_cfg >= lo_cfg {
            let from = (lo - lo_cfg) as usize;
            let to = (hi - lo_cfg) as usize + 1;
            self.zoom_delta[from] += bytes as i64;
            self.zoom_delta[to] -= bytes as i64;
        }

        self.ts_min = Some(match self.ts_min {
            Some(m) => m.min(feature.timestamp),
            None => feature.timestamp,
        });
        self.ts_max = Some(match self.ts_max {
            Some(m) => m.max(feature.timestamp),
            None => feature.timestamp,
        });
    }

    /// Associative merge of two adjacent chunks' accumulators, left first.
    fn merge(mut self, other: ScanAcc, opts: &StatsOptions) -> Self {
        self.features += other.features;
        self.total_feature_bytes += other.total_feature_bytes;
        for (i, d) in other.zoom_delta.iter().enumerate() {
            self.zoom_delta[i] += *d;
        }
        for (name, acc) in &other.numeric {
            self.numeric
                .entry(name.clone())
                .or_insert_with(NumAcc::new)
                .merge(acc);
        }
        for (name, acc) in other.categorical {
            self.categorical
                .entry(name.clone())
                .or_insert_with(CatAcc::new)
                .merge(acc, opts);
        }
        self.ts_min = match (self.ts_min, other.ts_min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.ts_max = match (self.ts_max, other.ts_max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self
    }

    fn finish(self, opts: &StatsOptions, timestamps: TimestampStats) -> DatasetStats {
        let features = self.features;
        let numeric = self
            .numeric
            .iter()
            .map(|(name, acc)| {
                (
                    name.clone(),
                    NumericColStats {
                        min: acc.min,
                        max: acc.max,
                        max_abs: acc.max_abs,
                        all_integer: acc.all_integer,
                        finite: acc.finite,
                        nulls: features.saturating_sub(acc.finite),
                    },
                )
            })
            .collect();
        let categorical = self
            .categorical
            .iter()
            .map(|(name, acc)| {
                (
                    name.clone(),
                    CategoricalColStats {
                        categories: (!acc.overflow)
                            .then(|| acc.ordered().into_iter().map(str::to_string).collect()),
                        distinct_overflow: acc.overflow,
                        total_value_bytes: acc.total_value_bytes,
                        values: acc.values,
                    },
                )
            })
            .collect();

        // Prefix-sum the difference array into b̂(z).
        let mut running = 0i64;
        let per_zoom = (0..opts.zoom_len())
            .map(|i| {
                running += self.zoom_delta[i];
                running.max(0) as u64
            })
            .collect();

        DatasetStats {
            features,
            numeric,
            categorical,
            byte_mass: ByteMassStats {
                zoom_range: opts.zoom_range,
                total_feature_bytes: self.total_feature_bytes,
                per_zoom,
            },
            timestamps,
            category_cap: opts.category_cap,
            category_byte_cap: opts.category_byte_cap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::props::FeatureProperties;
    use geojson::{Feature, Geometry, Value as GeomValue};
    use serde_json::json;

    fn feature(lon: f64, lat: f64, ts: u64, props: serde_json::Value) -> ParsedFeature {
        let map = props.as_object().cloned().unwrap_or_default();
        ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::Point(vec![lon, lat]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: FeatureProperties::from_map(map),
            timestamp: ts,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon,
            lat,
        }
    }

    /// A `DatasetStats` over no features — a base to drop hand-built column
    /// stats into when the rule under test reads only column totals.
    fn empty_stats() -> DatasetStats {
        collect_dataset_stats(&[], &AttributeFilter::KeepAll, &types(&[]))
    }

    fn line_feature(coords: Vec<Vec<f64>>, ts: u64, props: serde_json::Value) -> ParsedFeature {
        let map = props.as_object().cloned().unwrap_or_default();
        let (lon, lat) = (coords[0][0], coords[0][1]);
        ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::LineString(coords))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: FeatureProperties::from_map(map),
            timestamp: ts,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon,
            lat,
        }
    }

    fn types(pairs: &[(&str, PropertyKind)]) -> PropertyTypes {
        pairs.iter().map(|(n, k)| (n.to_string(), *k)).collect()
    }

    // ---- Unit: the statistics themselves -----------------------------------

    /// Hand-built features → the exact numbers a pin will be derived from.
    #[test]
    fn numeric_stats_match_a_hand_computed_domain() {
        let feats = vec![
            feature(0.0, 0.0, 10, json!({ "mag": 1.5, "cnt": 3 })),
            feature(1.0, 1.0, 20, json!({ "mag": -4.0, "cnt": 9 })),
            // NaN → no finite value: present in the row, absent from the domain.
            feature(2.0, 2.0, 30, json!({ "mag": f64::NAN, "cnt": 1 })),
            // `mag` missing entirely.
            feature(3.0, 3.0, 40, json!({ "cnt": 1 })),
        ];
        let t = types(&[
            ("mag", PropertyKind::Numeric),
            ("cnt", PropertyKind::Numeric),
        ]);
        let stats = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &t);

        assert_eq!(stats.features, 4);
        let mag = &stats.numeric["mag"];
        assert_eq!(mag.min, -4.0);
        assert_eq!(mag.max, 1.5);
        assert_eq!(mag.max_abs, 4.0);
        assert!(!mag.all_integer, "1.5 is not integer-valued");
        assert_eq!(mag.finite, 2);
        assert_eq!(mag.nulls, 2, "the NaN row and the absent row both count");
        assert_eq!(mag.finite + mag.nulls, stats.features);

        let cnt = &stats.numeric["cnt"];
        assert_eq!((cnt.min, cnt.max, cnt.max_abs), (1.0, 9.0, 9.0));
        assert!(cnt.all_integer);
        assert_eq!((cnt.finite, cnt.nulls), (4, 0));
        assert_eq!(cnt.span(), 8.0);
    }

    /// A column the schema declares but no feature carries still gets an entry
    /// — the pin has to exist for the all-null column every tile emits.
    #[test]
    fn a_declared_but_absent_column_still_gets_stats() {
        let feats = vec![feature(0.0, 0.0, 1, json!({ "present": 1 }))];
        let t = types(&[
            ("present", PropertyKind::Numeric),
            ("absent", PropertyKind::Numeric),
            ("absent_cat", PropertyKind::Categorical),
        ]);
        let stats = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &t);
        let absent = &stats.numeric["absent"];
        assert_eq!(absent.finite, 0);
        assert_eq!(absent.nulls, 1);
        assert_eq!(absent.min, f64::INFINITY);
        assert_eq!(absent.span(), 0.0, "no finite value ⇒ no span");
        let cat = &stats.categorical["absent_cat"];
        assert_eq!(cat.values, 0);
        assert_eq!(cat.categories.as_deref(), Some(&[][..]));
    }

    /// First-seen order is the DATASET's feature order, and it is what the
    /// dictionary pin carries. Not sorted order, not thread-arrival order.
    #[test]
    fn categorical_first_seen_order_is_dataset_order() {
        let feats = vec![
            feature(0.0, 0.0, 1, json!({ "kind": "zulu" })),
            feature(1.0, 1.0, 2, json!({ "kind": "alpha" })),
            feature(2.0, 2.0, 3, json!({ "kind": "zulu" })),
            feature(3.0, 3.0, 4, json!({ "kind": "mike" })),
        ];
        let t = types(&[("kind", PropertyKind::Categorical)]);
        // Chunk small enough to force several folds + merges.
        let opts = StatsOptions {
            chunk: 1,
            ..StatsOptions::default()
        };
        let stats = collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts);
        let kind = &stats.categorical["kind"];
        assert_eq!(
            kind.categories.as_deref().unwrap(),
            ["zulu".to_string(), "alpha".to_string(), "mike".to_string()]
        );
        assert!(!kind.distinct_overflow);
        assert_eq!(kind.values, 4);
        assert_eq!(kind.total_value_bytes, 4 + 5 + 4 + 4);
        assert_eq!(kind.distinct(), Some(3));
        assert_eq!(kind.category_bytes(), Some(4 + 5 + 4));

        // ...and the pin carries exactly that list, whenever it pins one.
        match &stats.to_pins_with(&HoistPolicy::unbounded()).dict["kind"] {
            GlobalDictVerdict::Dictionary(c) => {
                assert_eq!(c.as_slice(), &["zulu", "alpha", "mike"])
            }
            GlobalDictVerdict::Utf8 => panic!("a 3-category column must pin as a dictionary"),
        }
        // Under the DEFAULT policy this four-row dataset does not get one, and
        // that is correct rather than a regression: 17 bytes of values cannot
        // repay a dictionary's framing. "It fits under the cap" was never the
        // question — see `HoistPolicy`.
        assert_eq!(
            stats.categorical_verdicts(&HoistPolicy::default()),
            vec![("kind", CategoricalVerdict::WireLoses)]
        );
    }

    /// Cap overflow drops the set, marks the column, and forces `Utf8` — and it
    /// does so identically however the scan was chunked.
    #[test]
    fn category_cap_overflow_drops_the_set_and_forces_utf8() {
        let feats: Vec<ParsedFeature> = (0..10)
            .map(|i| feature(i as f64, 0.0, i as u64, json!({ "id": format!("v{i}") })))
            .collect();
        let t = types(&[("id", PropertyKind::Categorical)]);

        for chunk in [1usize, 2, 3, 7, 64] {
            let opts = StatsOptions {
                category_cap: 4,
                chunk,
                ..StatsOptions::default()
            };
            let stats = collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts);
            let id = &stats.categorical["id"];
            assert!(id.distinct_overflow, "chunk={chunk}");
            assert!(id.categories.is_none(), "chunk={chunk}: the set is dropped");
            // The plain-value counters survive overflow — they are what the
            // dataset-scale size comparison needs.
            assert_eq!(id.values, 10, "chunk={chunk}");
            assert_eq!(id.total_value_bytes, 20, "chunk={chunk}");
            assert!(matches!(
                stats.to_pins().dict["id"],
                GlobalDictVerdict::Utf8
            ));
        }

        // Exactly AT the cap is not overflow.
        let at_cap = StatsOptions {
            category_cap: 10,
            chunk: 3,
            ..StatsOptions::default()
        };
        let stats = collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &at_cap);
        assert!(!stats.categorical["id"].distinct_overflow);
        assert_eq!(stats.categorical["id"].distinct(), Some(10));
    }

    // ------------------------------------------------------------------
    // The verdict gates (wire surrogate + resident hoist cap)
    // ------------------------------------------------------------------

    /// Build `CategoricalColStats` directly — the verdict rule's only inputs
    /// are dataset totals, so the rule is testable without a feature scan.
    fn cat_stats(categories: &[&str], rows: u64, avg_value_len: u64) -> CategoricalColStats {
        CategoricalColStats {
            categories: Some(categories.iter().map(|s| (*s).to_string()).collect()),
            distinct_overflow: false,
            total_value_bytes: rows * avg_value_len,
            values: rows,
        }
    }

    /// A LOW-cardinality column over many rows is exactly what the hoist is
    /// for: both gates pass and it pins a dictionary.
    #[test]
    fn a_low_cardinality_column_over_many_rows_still_hoists() {
        let st = cat_stats(&["car", "bus", "tram"], 1_000_000, 4);
        assert_eq!(
            categorical_verdict(&st, &HoistPolicy::default()),
            CategoricalVerdict::Dictionary
        );
    }

    /// THE DEFECT, in its own test. A free-text column whose distinct set fits
    /// the `UInt16` key space used to pin a dictionary purely because it fit —
    /// and the reader then paid for that list in EVERY resident tile. The two
    /// gates are independent and this column is caught by the resident one:
    /// the wire surrogate still (correctly) says the dictionary is smaller.
    #[test]
    fn a_high_cardinality_free_text_column_is_declined_by_the_resident_gate() {
        // 14 653 usernames averaging 10 bytes, over 380 007 rows — the measured
        // shape of the OSM changeset `user` column.
        let categories: Vec<String> = (0..14_653).map(|i| format!("user-{i:06}")).collect();
        let refs: Vec<&str> = categories.iter().map(String::as_str).collect();
        let st = cat_stats(&refs, 380_007, 11);

        // The WIRE surrogate is not the thing that catches it: counted once,
        // dataset-wide, the dictionary genuinely is the smaller wire form.
        let category_bytes: u64 = categories.iter().map(|s| s.len() as u64).sum();
        assert!(
            dataset_dictionary_is_smaller(
                st.total_value_bytes,
                st.values,
                category_bytes,
                categories.len() as u64
            ),
            "the wire surrogate alone would have pinned this column"
        );

        // The RESIDENT gate is. `category_bytes` would be replicated into every
        // resident tile of the reader's 2000-tile default cache.
        assert_eq!(
            categorical_verdict(&st, &HoistPolicy::default()),
            CategoricalVerdict::TooLargeToHoist(14_653, category_bytes)
        );

        // ...and the verdict it resolves to is a dataset-global `Utf8`, so ONE
        // Arrow type still holds everywhere (TB-3's fork collapse survives).
        let mut stats = empty_stats();
        stats.categorical.insert("user".into(), st);
        assert!(matches!(
            stats.to_pins().dict["user"],
            GlobalDictVerdict::Utf8
        ));
    }

    /// A near-unique column loses on the WIRE too, and is caught first — the
    /// gates are ordered so the message names the real reason.
    #[test]
    fn a_near_unique_column_is_declined_by_the_wire_gate() {
        let categories: Vec<String> = (0..500).map(|i| format!("{i:03}")).collect();
        let refs: Vec<&str> = categories.iter().map(String::as_str).collect();
        // 500 distinct 3-byte values over 500 rows: every row is its own
        // category, so the keys cost more than the values save.
        let st = cat_stats(&refs, 500, 3);
        assert_eq!(
            categorical_verdict(&st, &HoistPolicy::default()),
            CategoricalVerdict::WireLoses
        );
    }

    /// The byte cap catches what the count cap misses: few categories, each of
    /// them enormous.
    #[test]
    fn the_resident_gate_has_a_byte_half_as_well_as_a_count_half() {
        let big: Vec<String> = (0..8)
            .map(|i| format!("{}{i}", "x".repeat(1_000)))
            .collect();
        let refs: Vec<&str> = big.iter().map(String::as_str).collect();
        let st = cat_stats(&refs, 1_000_000, 1_001);
        assert_eq!(
            categorical_verdict(&st, &HoistPolicy::default()),
            CategoricalVerdict::TooLargeToHoist(8, 8 * 1_001)
        );
        // Only 8 categories — the COUNT half would have waved it through.
        assert_eq!(
            categorical_verdict(
                &st,
                &HoistPolicy {
                    max_category_bytes: u64::MAX,
                    ..HoistPolicy::default()
                }
            ),
            CategoricalVerdict::Dictionary
        );
    }

    /// Overflow still wins over both gates: there is no list to measure.
    #[test]
    fn overflow_outranks_the_size_gates() {
        let mut st = cat_stats(&["a", "b"], 1_000_000, 1);
        st.distinct_overflow = true;
        for policy in [HoistPolicy::default(), HoistPolicy::unbounded()] {
            assert_eq!(
                categorical_verdict(&st, &policy),
                CategoricalVerdict::Overflowed
            );
        }
    }

    /// THE ROLLBACK: `HoistPolicy::unbounded` reproduces the pre-fix rule
    /// exactly — pin whatever fits the key space, measure nothing.
    #[test]
    fn the_unbounded_policy_restores_the_previous_behaviour() {
        let categories: Vec<String> = (0..14_653).map(|i| format!("user-{i:06}")).collect();
        let refs: Vec<&str> = categories.iter().map(String::as_str).collect();
        let cases = [
            cat_stats(&refs, 380_007, 11),     // resident-gated under the default
            cat_stats(&["a", "b", "c"], 4, 4), // wire-gated under the default
        ];
        for st in cases {
            assert_eq!(
                categorical_verdict(&st, &HoistPolicy::unbounded()),
                CategoricalVerdict::Dictionary
            );
        }
    }

    /// The verdict is a pure function of dataset totals, so `to_pins` and
    /// `categorical_verdicts` can never disagree, and repeated calls are
    /// identical (the determinism spine: no map iteration, no clock, no RNG).
    #[test]
    fn the_verdict_is_deterministic_and_agrees_with_the_pins() {
        let long = "x".repeat(9_000);
        let mut stats = empty_stats();
        stats
            .categorical
            .insert("kind".into(), cat_stats(&["car", "bus"], 1_000_000, 3));
        stats
            .categorical
            .insert("note".into(), cat_stats(&[&long], 1_000_000, 9_000));
        let policy = HoistPolicy::default();
        for _ in 0..3 {
            let pins = stats.to_pins_with(&policy);
            let verdicts: BTreeMap<&str, CategoricalVerdict> =
                stats.categorical_verdicts(&policy).into_iter().collect();
            assert_eq!(verdicts["kind"], CategoricalVerdict::Dictionary);
            assert!(matches!(
                pins.dict["kind"],
                GlobalDictVerdict::Dictionary(_)
            ));
            assert!(matches!(
                verdicts["note"],
                CategoricalVerdict::TooLargeToHoist(..)
            ));
            assert!(matches!(pins.dict["note"], GlobalDictVerdict::Utf8));
            assert_eq!(
                pins.to_canonical_json(),
                stats.to_pins_with(&policy).to_canonical_json()
            );
        }
    }

    /// The BYTE cap is the second, independent overflow trigger.
    #[test]
    fn category_byte_cap_overflow_is_chunk_invariant() {
        let feats: Vec<ParsedFeature> = (0..8)
            .map(|i| {
                feature(
                    i as f64,
                    0.0,
                    i as u64,
                    json!({ "blob": "x".repeat(10 + i) }),
                )
            })
            .collect();
        let t = types(&[("blob", PropertyKind::Categorical)]);
        for chunk in [1usize, 2, 5, 32] {
            let opts = StatsOptions {
                category_cap: MAX_CATEGORIES,
                category_byte_cap: 40,
                chunk,
                ..StatsOptions::default()
            };
            let stats = collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts);
            assert!(
                stats.categorical["blob"].distinct_overflow,
                "chunk={chunk}: 8 distinct blobs of 10..17 B exceed a 40 B cap"
            );
        }
        // A cap the set fits under does not fire, at any chunking.
        for chunk in [1usize, 3, 32] {
            let opts = StatsOptions {
                category_byte_cap: 4096,
                chunk,
                ..StatsOptions::default()
            };
            let stats = collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts);
            assert!(
                !stats.categorical["blob"].distinct_overflow,
                "chunk={chunk}"
            );
        }
    }

    /// The attribute filter keeps a column out of the statistics entirely — a
    /// pin for a column the archive does not contain would be a lie.
    #[test]
    fn attribute_filter_excludes_a_column_from_the_stats() {
        let feats = vec![
            feature(0.0, 0.0, 1, json!({ "keep": 1, "drop": 2, "kind": "a" })),
            feature(1.0, 1.0, 2, json!({ "keep": 5, "drop": 9, "kind": "b" })),
        ];
        let t = types(&[
            ("keep", PropertyKind::Numeric),
            ("drop", PropertyKind::Numeric),
            ("kind", PropertyKind::Categorical),
        ]);

        let excluded = AttributeFilter::Exclude(["drop".to_string()].into_iter().collect());
        let stats = collect_dataset_stats(&feats, &excluded, &t);
        assert!(stats.numeric.contains_key("keep"));
        assert!(!stats.numeric.contains_key("drop"));
        assert!(stats.categorical.contains_key("kind"));
        assert!(!stats.to_pins().attr.contains_key("drop"));

        let only = AttributeFilter::Include(["keep".to_string()].into_iter().collect());
        let stats = collect_dataset_stats(&feats, &only, &t);
        assert_eq!(stats.numeric.len(), 1);
        assert!(stats.categorical.is_empty());

        let none = collect_dataset_stats(&feats, &AttributeFilter::ExcludeAll, &t);
        assert!(none.numeric.is_empty() && none.categorical.is_empty());
        assert!(none.to_pins().is_empty());
    }

    /// A key the kind map does not list is UNPINNED — it keeps the incumbent
    /// per-tile behaviour rather than getting a half-informed global verdict.
    #[test]
    fn an_untyped_key_gets_no_stats_and_no_pin() {
        let feats = vec![feature(0.0, 0.0, 1, json!({ "typed": 1, "untyped": 2 }))];
        let t = types(&[("typed", PropertyKind::Numeric)]);
        let stats = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &t);
        assert!(stats.numeric.contains_key("typed"));
        assert!(!stats.numeric.contains_key("untyped"));
        assert!(!stats.to_pins().attr.contains_key("untyped"));
    }

    /// Numeric-looking STRINGS coerce into a numeric column's domain, exactly
    /// as the tile builder coerces them — the two share one coercion.
    #[test]
    fn numeric_strings_enter_the_domain_like_the_encoder_sees_them() {
        let feats = vec![
            feature(0.0, 0.0, 1, json!({ "alt": "1000.0" })),
            feature(1.0, 1.0, 2, json!({ "alt": 250 })),
            feature(2.0, 2.0, 3, json!({ "alt": "not a number" })),
        ];
        let t = types(&[("alt", PropertyKind::Numeric)]);
        let stats = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &t);
        let alt = &stats.numeric["alt"];
        assert_eq!((alt.min, alt.max), (250.0, 1000.0));
        assert_eq!(alt.finite, 2);
        assert!(alt.all_integer, "1000.0 and 250 are both integer-valued");
    }

    /// Booleans and numbers stringify into categories the same way the tile
    /// builder stringifies them.
    #[test]
    fn bool_and_number_categories_use_the_encoder_spelling() {
        let feats = vec![
            feature(0.0, 0.0, 1, json!({ "flag": true, "code": 7 })),
            feature(1.0, 1.0, 2, json!({ "flag": false, "code": 7.5 })),
        ];
        let t = types(&[
            ("flag", PropertyKind::Categorical),
            ("code", PropertyKind::Categorical),
        ]);
        let stats = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &t);
        assert_eq!(
            stats.categorical["flag"].categories.as_deref().unwrap(),
            ["true".to_string(), "false".to_string()]
        );
        assert_eq!(
            stats.categorical["code"].categories.as_deref().unwrap(),
            ["7".to_string(), "7.5".to_string()]
        );
    }

    // ---- Unit: byte mass + timestamps --------------------------------------

    /// `b̂(z)` follows the per-feature zoom band, and the totals are the tiler's
    /// own arithmetic.
    #[test]
    fn byte_mass_respects_the_per_feature_zoom_band() {
        let feats = vec![
            // 1 vertex, 2 props → 16 + 32 + 32 = 80, visible at z2..=z4
            feature(0.0, 0.0, 1, json!({ "min_zoom": 2, "max_zoom": 4 })),
            // 3 vertices, 0 props → 48 + 0 + 32 = 80, visible everywhere
            line_feature(
                vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]],
                2,
                json!({}),
            ),
        ];
        let t = types(&[
            ("min_zoom", PropertyKind::Numeric),
            ("max_zoom", PropertyKind::Numeric),
        ]);
        let opts = StatsOptions {
            zoom_range: (0, 6),
            min_zoom_field: Some("min_zoom".to_string()),
            max_zoom_field: Some("max_zoom".to_string()),
            ..StatsOptions::default()
        };
        let stats = collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts);
        assert_eq!(stats.byte_mass.total_feature_bytes, 160);
        assert_eq!(
            stats.byte_mass.per_zoom,
            vec![80, 80, 160, 160, 160, 80, 80]
        );
        assert_eq!(stats.byte_mass.at_zoom(3), Some(160));
        assert_eq!(stats.byte_mass.at_zoom(9), None);
    }

    /// Without zoom fields configured, every feature is charged to every zoom.
    #[test]
    fn byte_mass_is_flat_without_a_zoom_band() {
        let feats: Vec<ParsedFeature> = (0..5)
            .map(|i| feature(i as f64, 0.0, i as u64, json!({})))
            .collect();
        let opts = StatsOptions {
            zoom_range: (3, 5),
            ..StatsOptions::default()
        };
        let stats = collect_dataset_stats_with(
            &feats,
            &AttributeFilter::KeepAll,
            &PropertyTypes::new(),
            &opts,
        );
        // 5 features × (16 + 0 + 32) = 240
        assert_eq!(stats.byte_mass.total_feature_bytes, 240);
        assert_eq!(stats.byte_mass.per_zoom, vec![240, 240, 240]);
    }

    /// The timestamp histogram carries the range and its quantiles land on real
    /// bin edges inside it.
    #[test]
    fn timestamp_quantiles_are_monotone_and_in_range() {
        let feats: Vec<ParsedFeature> = (0..1000)
            .map(|i| feature(0.0, 0.0, 1_000 + i * 7, json!({})))
            .collect();
        let stats = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &PropertyTypes::new());
        assert_eq!(stats.timestamps.min, Some(1_000));
        assert_eq!(stats.timestamps.max, Some(1_000 + 999 * 7));
        assert_eq!(stats.timestamps.count, 1000);
        assert_eq!(stats.timestamps.bins.iter().sum::<u64>(), 1000);

        let qs = stats.timestamps.quantiles(8);
        assert_eq!(qs.len(), 9);
        for w in qs.windows(2) {
            assert!(w[0] <= w[1], "quantiles must be non-decreasing: {qs:?}");
        }
        assert!(qs[0] >= 1_000 && *qs.last().unwrap() <= 1_000 + 999 * 7);
        assert_eq!(stats.timestamps.quantile(0.0), Some(1_000));
    }

    /// An empty dataset produces empty, well-formed statistics rather than
    /// panicking or a half-filled struct.
    #[test]
    fn empty_dataset_is_well_formed() {
        let stats = collect_dataset_stats(&[], &AttributeFilter::KeepAll, &PropertyTypes::new());
        assert_eq!(stats.features, 0);
        assert_eq!(stats.byte_mass.total_feature_bytes, 0);
        assert!(stats.byte_mass.per_zoom.iter().all(|b| *b == 0));
        assert_eq!(stats.timestamps.count, 0);
        assert_eq!(stats.timestamps.quantile(0.5), None);
        assert!(stats.to_pins().is_empty());
    }

    // ---- Property: chunk-boundary invariance -------------------------------

    /// A deterministic pseudo-random dataset generator. A hand-rolled LCG
    /// rather than `proptest`, which is not a dev-dependency of this crate and
    /// cannot be added by this item (`Cargo.toml` is not its file). Fixed seeds
    /// keep the cases reproducible, which is what the property needs anyway.
    fn pseudo_dataset(seed: u64, n: usize) -> Vec<ParsedFeature> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 11
        };
        (0..n)
            .map(|_| {
                let kind = next() % 7;
                let mag = (next() % 10_000) as f64 / 100.0;
                let cnt = (next() % 500) as i64 - 250;
                let ts = 1_700_000_000_000 + (next() % 86_400_000);
                let mut props = serde_json::Map::new();
                if next() % 5 != 0 {
                    props.insert("mag".into(), json!(mag));
                }
                if next() % 4 != 0 {
                    props.insert("cnt".into(), json!(cnt));
                }
                if next() % 3 != 0 {
                    props.insert("kind".into(), json!(format!("k{kind}")));
                }
                feature(
                    (next() % 360) as f64 - 180.0,
                    (next() % 170) as f64 - 85.0,
                    ts,
                    serde_json::Value::Object(props),
                )
            })
            .collect()
    }

    /// **The merge is associative**: chunking the fold differently cannot move
    /// a single field of the result.
    ///
    /// This is the property that makes `--workers` (and rayon's own split
    /// points, which vary with machine load) invisible to the archive. If it
    /// failed, a "global" verdict would in fact depend on how the scan happened
    /// to be parallelised — the exact defect mechanism M2 exists to remove,
    /// reintroduced one layer down.
    #[test]
    fn stats_are_invariant_under_chunk_boundaries() {
        let t = types(&[
            ("mag", PropertyKind::Numeric),
            ("cnt", PropertyKind::Numeric),
            ("kind", PropertyKind::Categorical),
        ]);
        for seed in 0..12u64 {
            let feats = pseudo_dataset(seed, 137);
            let base = StatsOptions {
                zoom_range: (0, 8),
                chunk: 1,
                ..StatsOptions::default()
            };
            let reference =
                collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &base);
            for chunk in [2usize, 3, 5, 16, 64, 137, 1024] {
                let opts = StatsOptions {
                    chunk,
                    ..base.clone()
                };
                let got = collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts);
                assert_eq!(
                    got, reference,
                    "seed {seed}: stats moved at chunk size {chunk}"
                );
            }
        }
    }

    /// The same property with the caps ARMED, so the overflow decision itself
    /// is under the invariance claim rather than only the happy path.
    #[test]
    fn capped_stats_are_invariant_under_chunk_boundaries() {
        let t = types(&[("kind", PropertyKind::Categorical)]);
        for seed in 20..26u64 {
            let feats = pseudo_dataset(seed, 91);
            for cap in [1usize, 2, 3, 6, 7, 8] {
                let base = StatsOptions {
                    category_cap: cap,
                    chunk: 1,
                    ..StatsOptions::default()
                };
                let reference =
                    collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &base);
                for chunk in [2usize, 4, 13, 91, 512] {
                    let opts = StatsOptions {
                        chunk,
                        ..base.clone()
                    };
                    let got =
                        collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts);
                    assert_eq!(
                        got, reference,
                        "seed {seed}, cap {cap}: overflow verdict moved at chunk size {chunk}"
                    );
                }
            }
        }
    }

    // ---- Determinism -------------------------------------------------------

    /// Two scans of one input agree, and so do scans at 1 vs 8 rayon threads.
    ///
    /// The mandatory byte-identical-re-run test, at the statistics level: the
    /// pins are what will fix the archive's bytes, so if the pins move between
    /// two runs of the same build the archive moves with them and every
    /// content-addressed pack name — the immutable-CDN contract — moves too.
    #[test]
    fn stats_and_pins_are_thread_count_independent() {
        let t = types(&[
            ("mag", PropertyKind::Numeric),
            ("cnt", PropertyKind::Numeric),
            ("kind", PropertyKind::Categorical),
        ]);
        let feats = pseudo_dataset(99, 4_096);
        let opts = StatsOptions {
            zoom_range: (0, 10),
            chunk: 64,
            ..StatsOptions::default()
        };

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts)
                })
        };

        let one = run(1);
        let eight = run(8);
        let one_again = run(1);
        assert_eq!(one, one_again, "two scans of one input disagreed");
        assert_eq!(one, eight, "the scan depends on the rayon thread count");

        // ...and the derived pins, which are what actually reaches the encoder,
        // serialize to the same canonical bytes.
        assert_eq!(
            one.to_pins().to_canonical_json(),
            eight.to_pins().to_canonical_json()
        );
        // Guard against a vacuous pass: something must actually be pinned.
        assert!(!one.to_pins().is_empty());
        assert!(one.categorical["kind"].distinct().unwrap() >= 2);
    }

    /// Feature ORDER is data, not noise: reversing the input legitimately
    /// changes first-seen category order, and nothing else.
    ///
    /// Worth pinning because it is the boundary of the invariance claim. The
    /// scan must be independent of how the work was SPLIT and of which thread
    /// won a race — not of the dataset's own order, which is a build input.
    #[test]
    fn first_seen_order_follows_input_order_and_only_that() {
        let t = types(&[("kind", PropertyKind::Categorical)]);
        let feats = pseudo_dataset(7, 64);
        let mut reversed = feats.clone();
        reversed.reverse();

        let a = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &t);
        let b = collect_dataset_stats(&reversed, &AttributeFilter::KeepAll, &t);
        let (ca, cb) = (
            a.categorical["kind"].categories.clone().unwrap(),
            b.categorical["kind"].categories.clone().unwrap(),
        );
        assert_ne!(ca, cb, "reversing the dataset must reorder first-seen");
        let (mut sa, mut sb) = (ca, cb);
        sa.sort();
        sb.sort();
        assert_eq!(sa, sb, "...but the SET is the same");
        assert_eq!(a.numeric, b.numeric, "numeric stats are order-free");
        assert_eq!(a.byte_mass, b.byte_mass);
    }

    // ---- Pins --------------------------------------------------------------

    /// A huge-magnitude id column refuses globally — including on the tiles
    /// whose own sample would happily have quantized it.
    #[test]
    fn a_global_magnitude_outlier_refuses_the_whole_column() {
        let mut feats: Vec<ParsedFeature> = (0..50)
            .map(|i| feature(i as f64 / 10.0, 0.0, i as u64, json!({ "trip_id": i })))
            .collect();
        // One row past i32::MAX anywhere in the dataset is enough.
        feats.push(feature(
            9.0,
            0.0,
            99,
            json!({ "trip_id": 2_300_000_000_000_000_000i64 }),
        ));
        let t = types(&[("trip_id", PropertyKind::Numeric)]);
        let stats = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &t);
        assert!(
            stats.to_pins().attr["trip_id"].refuse,
            "the dataset max_abs is what decides, not any tile's"
        );
        assert_eq!(
            stats.refused_numeric_columns(),
            vec![("trip_id", 2.3e18)],
            "the refusal is reportable, so a build can say so out loud"
        );
    }

    /// A narrow integer column pins to the exact step-1 affine at the DATASET
    /// minimum — the same value therefore decodes identically in every tile.
    #[test]
    fn an_integer_column_pins_to_an_exact_step_one_affine() {
        let feats: Vec<ParsedFeature> = (0..100)
            .map(|i| feature(i as f64, 0.0, i as u64, json!({ "cnt": 1000 + i })))
            .collect();
        let t = types(&[("cnt", PropertyKind::Numeric)]);
        let pins = collect_dataset_stats(&feats, &AttributeFilter::KeepAll, &t).to_pins();
        let pin = pins.attr["cnt"];
        assert!(!pin.refuse);
        assert_eq!((pin.o, pin.s), (1000.0, 1.0));
        assert_eq!(pin.leaf, stt_core::arrow_tile::PinnedLeaf::U16);
    }

    /// Overflowed columns are reportable by name.
    #[test]
    fn overflowed_columns_are_listed() {
        let feats: Vec<ParsedFeature> = (0..6)
            .map(|i| {
                feature(
                    i as f64,
                    0.0,
                    i as u64,
                    json!({ "a": format!("{i}"), "b": "same" }),
                )
            })
            .collect();
        let t = types(&[
            ("a", PropertyKind::Categorical),
            ("b", PropertyKind::Categorical),
        ]);
        let opts = StatsOptions {
            category_cap: 3,
            ..StatsOptions::default()
        };
        let stats = collect_dataset_stats_with(&feats, &AttributeFilter::KeepAll, &t, &opts);
        assert_eq!(stats.overflowed_categorical_columns(), vec!["a"]);
    }
}
