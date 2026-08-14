//! The post-zstd trial-encode objective, exposed as a typed candidate API.
//!
//! Every size decision in this workspace used to be argued from a
//! *pre-compression* surrogate: raw buffer arithmetic
//! (`categorical_dictionary_is_smaller`'s `CATEGORICAL_DICTIONARY_IPC_OVERHEAD`),
//! uncompressed width ladders, or a fixed shrink prior. The recorded reason
//! that is not good enough: **compact times measured +3.4% on the wire while
//! −13.1% uncompressed.** A model that never runs the compressor cannot see a
//! sign inversion, and the wire side is the only side a client pays for.
//!
//! This module answers the question the honest way — run the candidate through
//! the real [`stt_core`] encoder and the real compressor on the run's
//! deterministic sample, and report the *post-zstd* delta.
//!
//! ```text
//! run_trials(sample, layout, baseline, candidates) -> Vec<TrialResult>
//!            └ the deterministic loader sample
//!                    └ ONE SyntheticLayout, shared by the baseline and every trial
//!                              └ the incumbent settings the deltas are against
//! ```
//!
//! # ⚠️ This oracle is deliberately NOT wired into the per-tile encoder
//!
//! [`stt_core::arrow_tile`]'s `choose_time_forms`, `build_quantized_numeric_auto`,
//! the dictionary-vs-Utf8 comparison and `TileBudget::enforce_indexed` keep
//! their current behaviour. Nothing here calls into them and nothing there
//! calls in here.
//!
//! That split is load-bearing, not an oversight. The conformance rule
//! (§13.2) **rejects sample-dependent type decisions**: two tiles of one
//! archive may not disagree about a column's Arrow type because a sample
//! happened to fall one way. So the oracle may only inform decisions that are
//! functions of the dataset DOMAIN — a single global verdict per column, per
//! dataset — never a per-tile flip. Its outputs are dataset-level by
//! construction: one number per candidate, measured on the run's one sample
//! under the run's one layout.
//!
//! The consumer that turns these verdicts into encoder behaviour is the M2
//! two-pass build, in rebuild window R1. **If you find yourself calling
//! [`run_trials`] from inside the encoder, that boundary has been broken.**
//!
//! # What is measured, and on what scale
//!
//! [`Candidate`]s come in two scopes ([`Candidate::scope`]):
//!
//! - [`TrialScope::WholeTile`] — the lever is an
//!   [`EncoderConfig`](stt_core::arrow_tile::EncoderConfig) knob, so the
//!   trial re-encodes the whole sample through
//!   [`measure_sample_layout`](crate::measure::measure_sample_layout) with that
//!   one knob moved. `bytes_total` is the compressed size of the whole measured
//!   sample; `delta_bytes` is against the same sample at `baseline` settings.
//! - [`TrialScope::Column`] — the lever has no config knob at all (the encoder
//!   decides dictionary-vs-Utf8 internally, from an unmeasured byte model), so
//!   the trial prices the column directly through the shared attribution
//!   surface ([`ipc_zstd_len`], the same Arrow-IPC-plus-zstd path MO-3's
//!   leave-one-out attribution uses — not a second encoder). `bytes_total` is
//!   that COLUMN's bytes under the chosen representation and `delta_bytes` is
//!   against the opposite representation.
//!
//! Mixing the two scales in one ranking is a bug; that is why the scope is
//! published rather than implied.
//!
//! # Invariants
//!
//! - **One layout.** Every whole-tile trial is measured under the layout it was
//!   handed, which must be the run's own
//!   ([`SyntheticLayout::from_density`](crate::measure::SyntheticLayout::from_density)
//!   of the analysis result). A trial measured under a different layout
//!   confounds the lever with the tile cut. The single exception is
//!   [`Candidate::FeatureBudgetBytes`], where the tile cut *is* the lever — and
//!   it says so.
//! - **zstd 19 is the ceiling.** [`MAX_ZSTD_LEVEL`] is a hard cap: level 22 is a
//!   standing rejection (19 ≈ 22 measured, at several times the encode cost).
//!   The cap is enforced *before* any candidate is evaluated, so an
//!   out-of-range level never reaches the compressor.
//! - **No thinning.** No candidate drops, samples or aggregates features.
//!   `FeatureBudgetBytes` prices an occupancy cap by RE-CUTTING the same
//!   features into more tiles, never by discarding any.
//! - **Determinism.** `Vec<TrialResult>` serialises byte-identically across
//!   runs: results come back in input order, every accumulator is a `BTreeMap`
//!   or a `Vec`, the dictionary order is first-seen (matching the encoder), and
//!   the dispersion sums are exact `u128` integers. This is required, not
//!   aspirational — the vector is solver input, and pack names are
//!   content-addressed.
//! - **No analytic size model.** Every number here comes out of the compressor.
//!
//! # Cost
//!
//! One trial costs ~2 sample measurements (the trial itself, plus
//! [`REPLICATE_BLOCKS`] single-tile block measurements that give `stderr` real
//! replicates). The baseline half is measured once and reused across the whole
//! candidate list. Attribution is run under
//! [`AttributionDesign::SingletonV1`] on every oracle measurement purely as a
//! cost choice: the oracle reads only whole-tile bytes, and
//! [`measure_sample_layout_with`] documents that the attribution design has no
//! effect on them — while the singleton pass compresses `m` one-column batches
//! instead of `m + 1` near-full ones.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow::array::{
    ArrayRef, DictionaryArray, RecordBatch, RecordBatchOptions, StringArray, UInt16Array,
};
use arrow::datatypes::{DataType, Field, Schema, UInt16Type};
use serde::{Deserialize, Serialize};
use stt_core::arrow_tile::MIN_QUANTIZE_COORDS_M;

use crate::attribution::{ipc_zstd_len, AttributionDesign};
use crate::loader::{PropValue, SampledFeature};
use crate::measure::{
    measure_sample_layout_with, MeasureSettings, MeasuredEncoding, SyntheticLayout,
};

/// Hard ceiling on any zstd level the oracle will evaluate.
///
/// zstd 22 is a **standing rejection**: it was measured against 19 and the wire
/// difference did not survive the encode-time cost, so the sweep caps here and
/// `COLUMN_ZSTD_LEVEL = 19` stays the fixed level every cross-dataset
/// comparison is taken at. [`Candidate::validate`] rejects anything above this
/// before a single byte is compressed.
pub const MAX_ZSTD_LEVEL: i32 = 19;

/// Floor on an evaluable zstd level (zstd's own valid range starts at 1; `0`
/// would mean "the library default", which is not a measurable point).
pub const MIN_ZSTD_LEVEL: i32 = 1;

/// Contiguous blocks of the sample used as replicates for [`TrialResult::stderr`].
///
/// Four is a deliberate compromise: enough replicates for a ratio-estimator
/// variance to mean something, few enough that each block still clears the
/// measurement floor on a modest sample, and cheap — the blocks partition the
/// sample, so measuring all of them costs about one extra sample encode.
pub const REPLICATE_BLOCKS: usize = 4;

/// A single lever the oracle can price, post-zstd, against a baseline.
///
/// The variants are the five D4 decision sites plus the two sweep levers the
/// advisors and the `--target-size` solver need:
///
/// | Variant | Site | Scope | Lossy |
/// |---|---|---|---|
/// | [`Candidate::ZstdLevel`] | §5.1 per-dataset sweep | whole-tile | no |
/// | [`Candidate::CompactTimes`] | §2.6 time-column menu | whole-tile | no |
/// | [`Candidate::QuantizeCoords`] | coordinate grid | whole-tile | **yes** |
/// | [`Candidate::QuantizeAttr`] | §3.3 per-column width/step ladder | whole-tile | **yes** |
/// | [`Candidate::QuantizeAttrsAuto`] | §3.3 auto ladder | whole-tile | **yes** |
/// | [`Candidate::CategoricalDict`] | §3.4 dictionary boundary | column | no |
/// | [`Candidate::FeatureBudgetBytes`] | §4.3 byte cap | whole-tile | no |
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Candidate {
    /// Compress the sample at this zstd level. Capped at [`MAX_ZSTD_LEVEL`].
    ZstdLevel(i32),
    /// Turn the compact per-feature time columns on or off. The lever whose
    /// recorded wire/uncompressed sign inversion is the reason this module
    /// exists.
    CompactTimes(bool),
    /// Fixed-point coordinate quantization at this ground precision in meters
    /// (`None` = Float64 GeoArrow coordinates). **Lossy.**
    QuantizeCoords(Option<f64>),
    /// Explicit fixed-point precision for one numeric property — one rung of
    /// the §3.3 width/step ladder. **Lossy.**
    QuantizeAttr {
        /// Property name.
        column: String,
        /// Ground precision (the quantization step), in the property's units.
        step: f64,
    },
    /// Range-adaptive `UInt16` quantization for every otherwise-raw `Float64`
    /// property. **Lossy.**
    QuantizeAttrsAuto(bool),
    /// Price one categorical column as a `Dictionary<UInt16, Utf8>`
    /// (`dict: true`) or as plain `Utf8` (`dict: false`), post-zstd.
    ///
    /// This is the §3.4 boundary call, and it is the one lever with no
    /// [`EncoderConfig`](stt_core::arrow_tile::EncoderConfig) knob: the encoder
    /// decides per tile from
    /// `categorical_dictionary_is_smaller`, an arithmetic model over raw buffer
    /// widths plus an *unmeasured* 192-byte IPC allowance. Post-zstd the two
    /// representations are far closer than that model claims — a heavy-repeat
    /// Utf8 column compresses almost as well as its own dictionary — which is
    /// exactly the kind of overstatement measurement is here to catch.
    CategoricalDict {
        /// Property name.
        column: String,
        /// `true` prices the dictionary form, `false` the plain `Utf8` form.
        dict: bool,
    },
    /// Price a per-tile occupancy cap of `max_features` features in WIRE bytes.
    ///
    /// §4.3's gap is that a *byte* budget was never enforceable because nobody
    /// had measured compressed bytes per feature at realistic occupancy. This
    /// candidate re-cuts the same sample into tiles holding at most
    /// `max_features` features and measures what that costs on the wire, so a
    /// caller can convert a byte cap into a feature cap from a measurement
    /// instead of a guess.
    ///
    /// **Nothing is dropped.** The cap is priced by re-cutting, never by
    /// thinning; `bytes_total / tiles` is the measured wire size of one capped
    /// tile.
    FeatureBudgetBytes {
        /// Maximum features per tile.
        max_features: usize,
    },
}

/// What a [`TrialResult`]'s `bytes_total` is measured over.
///
/// Published because the two scales are NOT comparable: ranking a column-scoped
/// result against a whole-tile one is a category error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrialScope {
    /// `bytes_total` is the compressed size of the whole measured sample.
    WholeTile,
    /// `bytes_total` is one column's compressed bytes under the chosen
    /// representation.
    Column,
}

impl Candidate {
    /// Which scale this candidate's [`TrialResult`] is reported on.
    pub fn scope(&self) -> TrialScope {
        match self {
            Candidate::CategoricalDict { .. } => TrialScope::Column,
            _ => TrialScope::WholeTile,
        }
    }

    /// Whether applying this lever would discard information.
    ///
    /// The oracle prices lossy levers happily — that is what a shadow price IS
    /// — but a lossy candidate may never join an auto-applied recipe. The
    /// lexicographic no-thinning filter downstream keys off exactly this.
    pub fn lossy(&self) -> bool {
        match self {
            Candidate::QuantizeCoords(step) => step.is_some(),
            Candidate::QuantizeAttr { .. } => true,
            Candidate::QuantizeAttrsAuto(on) => *on,
            // zstd level, compact times and the dictionary form are exact
            // re-encodings; the occupancy cap re-cuts without dropping.
            Candidate::ZstdLevel(_)
            | Candidate::CompactTimes(_)
            | Candidate::CategoricalDict { .. }
            | Candidate::FeatureBudgetBytes { .. } => false,
        }
    }

    /// Reject a candidate that cannot be evaluated, BEFORE anything is encoded.
    ///
    /// [`run_trials`] validates the whole list up front, so a rejected
    /// candidate means no candidate ran — in particular an out-of-range zstd
    /// level never reaches the compressor.
    pub fn validate(&self) -> Result<()> {
        match self {
            Candidate::ZstdLevel(level) => {
                if *level > MAX_ZSTD_LEVEL {
                    bail!(
                        "zstd level {level} exceeds the sweep cap of {MAX_ZSTD_LEVEL}: level 22 \
                         is a standing rejection (19 ~ 22 measured) and 19 is the fixed level \
                         cross-dataset column costs are compared at"
                    );
                }
                if *level < MIN_ZSTD_LEVEL {
                    bail!("zstd level {level} is below the evaluable floor of {MIN_ZSTD_LEVEL}");
                }
            }
            Candidate::QuantizeCoords(Some(meters)) => {
                if !meters.is_finite() || *meters < MIN_QUANTIZE_COORDS_M {
                    bail!(
                        "coordinate quantization precision {meters} m is below the world-grid \
                         floor of {MIN_QUANTIZE_COORDS_M} m"
                    );
                }
            }
            Candidate::QuantizeAttr { column, step } => {
                if column.is_empty() {
                    bail!("quantize-attr candidate needs a column name");
                }
                if !step.is_finite() || *step <= 0.0 {
                    bail!(
                        "quantize-attr step for `{column}` must be finite and positive, got {step}"
                    );
                }
            }
            Candidate::CategoricalDict { column, .. } => {
                if column.is_empty() {
                    bail!("categorical-dict candidate needs a column name");
                }
            }
            Candidate::FeatureBudgetBytes { max_features } => {
                if *max_features == 0 {
                    bail!(
                        "feature-budget candidate needs a positive cap: a cap of 0 would be \
                         thinning, which is structurally rejected"
                    );
                }
            }
            Candidate::QuantizeCoords(None)
            | Candidate::CompactTimes(_)
            | Candidate::QuantizeAttrsAuto(_) => {}
        }
        Ok(())
    }

    /// The settings this candidate measures under, derived from `baseline` by
    /// moving exactly one knob. `None` for candidates that are not
    /// [`MeasureSettings`] levers at all (the column-scoped ones).
    fn settings_from(&self, baseline: &MeasureSettings) -> Option<MeasureSettings> {
        let mut s = baseline.clone();
        match self {
            Candidate::ZstdLevel(level) => s.zstd_level = *level,
            Candidate::CompactTimes(on) => s.compact_times = *on,
            Candidate::QuantizeCoords(step) => s.quantize_coords_m = *step,
            Candidate::QuantizeAttr { column, step } => {
                s.quantize_attrs.insert(column.clone(), *step);
            }
            Candidate::QuantizeAttrsAuto(on) => s.quantize_attrs_auto = *on,
            // The cut is the lever, not the settings.
            Candidate::FeatureBudgetBytes { .. } => {}
            Candidate::CategoricalDict { .. } => return None,
        }
        Some(s)
    }
}

/// One priced candidate.
///
/// ⚠️ `bytes_total`'s scale depends on [`Candidate::scope`] — see the module
/// docs. `delta_bytes` is always signed against the candidate's own reference
/// arm (the baseline settings for a whole-tile candidate, the opposite
/// representation for a column-level one), so **negative means smaller**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialResult {
    /// The candidate that was priced, echoed back so a result carries its own
    /// question. Results come back in the order the candidates were given.
    pub candidate: Candidate,
    /// Measured compressed bytes under the candidate.
    pub bytes_total: usize,
    /// `bytes_total − reference_bytes`. Negative = the candidate is smaller.
    pub delta_bytes: i64,
    /// `delta_bytes / reference_bytes`. `0.0` when the reference measured zero.
    pub delta_frac: f64,
    /// Standard error of `delta_frac`, from [`REPLICATE_BLOCKS`] contiguous
    /// blocks of the sample measured under both arms (the textbook
    /// ratio-estimator form, the same estimator
    /// [`ShareDispersion`](crate::attribution::ShareDispersion) publishes for
    /// per-column shares).
    ///
    /// `0.0` means **no dispersion evidence**, not "no noise": fewer than two
    /// usable replicates. Consumers must gate conservatively — a shrink is
    /// believable when it exceeds a multiple of this, never because this is
    /// small.
    pub stderr: f64,
}

/// Which tile cut an arm of a trial is measured under.
#[derive(Debug, Clone, Copy)]
enum LayoutPlan {
    /// The run's shared layout — the invariant case.
    Shared,
    /// Equal tiles holding at most `max_features` features each. Used only by
    /// [`Candidate::FeatureBudgetBytes`], where the cut IS the lever.
    CapAt(usize),
}

/// Price every candidate against `baseline`, post-zstd, on the run's sample.
///
/// `layout` must be the run's own [`SyntheticLayout`] — the one
/// `result.measured` was taken under, reconstructible as
/// `SyntheticLayout::from_density(&result.density)`. Sharing it is what makes a
/// trial delta attributable to the lever rather than to the tile cut.
///
/// Returns one [`TrialResult`] per candidate, **in input order**.
///
/// # Fallback
///
/// An unmeasurable sample (too few usable features — the
/// `MIN_MEASURE_FEATURES` floor) yields an EMPTY vector rather than an error or
/// a fabricated number. Callers fall back to their own heuristics, which is the
/// documented rollback for every consumer of this module.
///
/// # Errors
///
/// - any candidate fails [`Candidate::validate`] — checked for the whole list
///   before anything is encoded, so nothing is evaluated;
/// - a named categorical column carries no `Utf8` values in the sample;
/// - the encoder or the compressor fails. A lever can be *infeasible* on a
///   given dataset without being invalid — a `QuantizeAttr` step finer than the
///   `Int32` leaf can address on that column's range is the recorded case — and
///   the error names the candidate so a caller sweeping a ladder knows exactly
///   which rung to drop. Infeasibility is a property of the data, not of the
///   candidate, so it cannot be caught by `validate`.
pub fn run_trials(
    sample: &[SampledFeature],
    layout: &SyntheticLayout,
    baseline: &MeasureSettings,
    candidates: &[Candidate],
) -> Result<Vec<TrialResult>> {
    // Cap enforcement happens FIRST, over the whole list: the guarantee is that
    // a level above MAX_ZSTD_LEVEL never reaches the compressor at all.
    for candidate in candidates {
        candidate.validate()?;
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let Some(base) = measure(sample, baseline, layout, LayoutPlan::Shared, 0)? else {
        return Ok(Vec::new());
    };
    // Baseline replicates, measured once and reused by every candidate. Always
    // the single-tile cut: they estimate how far the RATIO moves between
    // comparable stretches of the sample, and are never used for `bytes_total`.
    let blocks = block_ranges(sample.len(), REPLICATE_BLOCKS);
    let base_blocks = measure_blocks(sample, &blocks, baseline, layout, LayoutPlan::Shared, &base)?;

    let mut out = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        out.push(match candidate {
            Candidate::CategoricalDict { column, dict } => {
                column_trial(sample, baseline, &base, candidate, column, *dict)?
            }
            _ => whole_tile_trial(
                sample,
                layout,
                baseline,
                &base,
                base_blocks.as_deref(),
                &blocks,
                candidate,
            )?,
        });
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Whole-tile candidates: one EncoderConfig knob, measured through the real encoder
// ----------------------------------------------------------------------------

/// Measure the sample under `settings` and the resolved cut.
///
/// `usable` is the baseline's measured feature count, needed only to size a
/// [`LayoutPlan::CapAt`] cut; pass `0` when resolving the baseline itself
/// (which is always [`LayoutPlan::Shared`]).
fn measure(
    sample: &[SampledFeature],
    settings: &MeasureSettings,
    layout: &SyntheticLayout,
    plan: LayoutPlan,
    usable: usize,
) -> Result<Option<MeasuredEncoding>> {
    let resolved = match plan {
        LayoutPlan::Shared => layout.clone(),
        LayoutPlan::CapAt(max_features) => equal_tiles(div_ceil(usable, max_features)),
    };
    // SingletonV1 is a pure cost choice: only `bytes_total` is read here, and
    // the attribution design provably does not move it.
    measure_sample_layout_with(sample, settings, &resolved, AttributionDesign::SingletonV1)
}

/// Price one [`MeasureSettings`] lever.
#[allow(clippy::too_many_arguments)]
fn whole_tile_trial(
    sample: &[SampledFeature],
    layout: &SyntheticLayout,
    baseline: &MeasureSettings,
    base: &MeasuredEncoding,
    base_blocks: Option<&[u64]>,
    blocks: &[(usize, usize)],
    candidate: &Candidate,
) -> Result<TrialResult> {
    let settings = candidate
        .settings_from(baseline)
        .expect("whole-tile candidates always derive settings");
    let plan = match candidate {
        Candidate::FeatureBudgetBytes { max_features } => LayoutPlan::CapAt(*max_features),
        _ => LayoutPlan::Shared,
    };
    let trial = measure(sample, &settings, layout, plan, base.features)
        .with_context(|| format!("trial encode failed for candidate {candidate:?}"))?
        .with_context(|| {
            format!(
                "candidate {candidate:?} did not measure although the baseline did — the \
                 measurement floor is layout-independent, so this should be unreachable"
            )
        })?;

    // Replicates only when the baseline half exists; otherwise there is no
    // dispersion evidence and the honest answer is 0.0.
    let stderr = match base_blocks {
        Some(base_bytes) => match measure_blocks(sample, blocks, &settings, layout, plan, base)? {
            Some(trial_bytes) => ratio_stderr(base_bytes, &trial_bytes),
            None => 0.0,
        },
        None => 0.0,
    };

    Ok(finish(
        candidate.clone(),
        trial.bytes_total,
        base.bytes_total,
        stderr,
    ))
}

/// Measure each contiguous block of the sample on its own, for replicates.
///
/// `Ok(None)` when any block falls under the measurement floor — a partial set
/// of replicates would silently re-weight the ratio, so the whole estimate is
/// withdrawn instead.
fn measure_blocks(
    sample: &[SampledFeature],
    blocks: &[(usize, usize)],
    settings: &MeasureSettings,
    layout: &SyntheticLayout,
    plan: LayoutPlan,
    base: &MeasuredEncoding,
) -> Result<Option<Vec<u64>>> {
    if blocks.len() < 2 {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(blocks.len());
    for &(start, end) in blocks {
        // A block's usable count is not known before measuring it, so the cap
        // cut is sized from the baseline's usable FRACTION of the sample —
        // deterministic, and exact whenever the sample is single-kind.
        let usable = if sample.is_empty() {
            0
        } else {
            base.features * (end - start) / sample.len()
        };
        let single = SyntheticLayout::single();
        let block_layout = match plan {
            LayoutPlan::Shared => &single,
            LayoutPlan::CapAt(_) => layout,
        };
        let Some(m) = measure(&sample[start..end], settings, block_layout, plan, usable)? else {
            return Ok(None);
        };
        out.push(m.bytes_total as u64);
    }
    Ok(Some(out))
}

// ----------------------------------------------------------------------------
// Column candidates: priced through the shared attribution surface
// ----------------------------------------------------------------------------

/// Price one categorical column as dictionary vs plain `Utf8`, post-zstd.
///
/// The column is cut into the same number of chunks the baseline measurement
/// actually used, because the dictionary form repays its IPC dictionary message
/// PER TILE — charging it once over the whole sample is precisely the mistake
/// the 192-byte allowance in the encoder's pre-compression model is trying, and
/// failing, to approximate.
///
/// Both arms go through [`ipc_zstd_len`], the same IPC + zstd surface MO-3's
/// leave-one-out attribution uses. There is no second encoder here.
fn column_trial(
    sample: &[SampledFeature],
    baseline: &MeasureSettings,
    base: &MeasuredEncoding,
    candidate: &Candidate,
    column: &str,
    dict: bool,
) -> Result<TrialResult> {
    let values = categorical_values(sample, column);
    if values.iter().all(Option::is_none) {
        bail!(
            "column `{column}` carries no categorical (Utf8) values in the {}-feature sample, so \
             there is no dictionary boundary to price",
            sample.len()
        );
    }

    let chunks = block_ranges(values.len(), base.tiles.max(1));
    let mut plain = Vec::with_capacity(chunks.len());
    let mut dictionary = Vec::with_capacity(chunks.len());
    for &(start, end) in &chunks {
        let slice = &values[start..end];
        plain.push(ipc_zstd_len(
            &utf8_batch(column, slice)?,
            baseline.zstd_level,
        )?);
        dictionary.push(ipc_zstd_len(
            &dictionary_batch(column, slice)?,
            baseline.zstd_level,
        )?);
    }

    let (chosen, reference) = if dict {
        (&dictionary, &plain)
    } else {
        (&plain, &dictionary)
    };
    let chosen_bytes: u64 = chosen.iter().sum();
    let reference_bytes: u64 = reference.iter().sum();

    Ok(finish(
        candidate.clone(),
        chosen_bytes as usize,
        reference_bytes as usize,
        ratio_stderr(reference, chosen),
    ))
}

/// This column's value for every sampled feature, in sample order.
///
/// A feature that does not carry the property, or carries it as a number,
/// contributes a null — which is what the encoder would store for it too.
fn categorical_values(sample: &[SampledFeature], column: &str) -> Vec<Option<String>> {
    sample
        .iter()
        .map(|f| {
            f.properties
                .iter()
                .find(|(name, _)| name == column)
                .and_then(|(_, value)| match value {
                    PropValue::Text(s) => Some(s.clone()),
                    PropValue::Number(_) => None,
                })
        })
        .collect()
}

/// One-column `Utf8` batch.
fn utf8_batch(name: &str, values: &[Option<String>]) -> Result<RecordBatch> {
    let array: ArrayRef = Arc::new(StringArray::from_iter(values.iter().map(|v| v.as_deref())));
    one_column_batch(name, DataType::Utf8, array, values.len())
}

/// One-column `Dictionary<UInt16, Utf8>` batch, categories in FIRST-SEEN order.
///
/// First-seen order is the encoder's own dictionary contract (a pinned
/// invariant), so the arm being priced is the representation the encoder would
/// actually emit — not a re-sorted lookalike that compresses differently.
fn dictionary_batch(name: &str, values: &[Option<String>]) -> Result<RecordBatch> {
    let mut categories: Vec<String> = Vec::new();
    let mut lookup: BTreeMap<String, u16> = BTreeMap::new();
    let mut indices: Vec<Option<u16>> = Vec::with_capacity(values.len());
    for value in values {
        match value {
            None => indices.push(None),
            Some(s) => {
                if let Some(&idx) = lookup.get(s) {
                    indices.push(Some(idx));
                } else {
                    let idx = u16::try_from(categories.len()).with_context(|| {
                        format!(
                            "column `{name}` has more than {} distinct values in the sample, which \
                             a Dictionary<UInt16, Utf8> key cannot address — the encoder falls \
                             back to Utf8 for such a column, so there is no dictionary arm to price",
                            u16::MAX
                        )
                    })?;
                    categories.push(s.clone());
                    lookup.insert(s.clone(), idx);
                    indices.push(Some(idx));
                }
            }
        }
    }
    let dict = DictionaryArray::<UInt16Type>::try_new(
        UInt16Array::from(indices),
        Arc::new(StringArray::from(categories)) as ArrayRef,
    )
    .context("oracle: dictionary arm build failed")?;
    let dt = DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8));
    one_column_batch(name, dt, Arc::new(dict), values.len())
}

/// A one-column batch with an explicit row count (a fully-null column carries
/// no length of its own once projected).
fn one_column_batch(
    name: &str,
    data_type: DataType,
    array: ArrayRef,
    rows: usize,
) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(name, data_type, true)]));
    let options = RecordBatchOptions::new().with_row_count(Some(rows));
    RecordBatch::try_new_with_options(schema, vec![array], &options)
        .context("oracle: column batch build failed")
}

// ----------------------------------------------------------------------------
// Shared arithmetic
// ----------------------------------------------------------------------------

/// Assemble a result from a measured pair.
fn finish(
    candidate: Candidate,
    bytes_total: usize,
    reference_bytes: usize,
    stderr: f64,
) -> TrialResult {
    let delta_bytes = bytes_total as i64 - reference_bytes as i64;
    let delta_frac = if reference_bytes == 0 {
        0.0
    } else {
        delta_bytes as f64 / reference_bytes as f64
    };
    TrialResult {
        candidate,
        bytes_total,
        delta_bytes,
        delta_frac,
        stderr,
    }
}

/// `ceil(a / b)`, with `b == 0` treated as "one tile".
fn div_ceil(a: usize, b: usize) -> usize {
    if b == 0 {
        return 1;
    }
    a.div_ceil(b).max(1)
}

/// A layout of `k` equal-weight tiles.
fn equal_tiles(k: usize) -> SyntheticLayout {
    SyntheticLayout {
        tile_sizes: vec![1; k.max(1)],
    }
}

/// Split `total` items into `parts` contiguous ranges as evenly as possible,
/// the larger ones first. Deterministic, and exact: the ranges partition
/// `0..total`.
fn block_ranges(total: usize, parts: usize) -> Vec<(usize, usize)> {
    let parts = parts.max(1).min(total.max(1));
    let base = total / parts;
    let rem = total % parts;
    let mut out = Vec::with_capacity(parts);
    let mut cursor = 0usize;
    for i in 0..parts {
        let take = base + usize::from(i < rem);
        out.push((cursor, cursor + take));
        cursor += take;
    }
    out
}

/// Standard error of the ratio estimator `R̂ = Σy / Σx` over paired replicates.
///
/// ```text
/// se(R̂)² = Σ_i (y_i − R̂·x_i)² · n / ((n − 1) · (Σx)²)
/// ```
///
/// `R̂ − 1` is the relative delta, so `se(R̂)` is the standard error of
/// [`TrialResult::delta_frac`].
///
/// This is the same estimator
/// [`ShareDispersion`](crate::attribution::ShareDispersion) publishes, and it is
/// deliberately a separate function rather than a reuse: that one estimates one
/// column's SHARE of a total within a single measurement, this one estimates
/// the RATIO between two measurements of the same tiles. The numerator is
/// expanded exactly in `u128` — `(Σx)²·Σy² − 2·Σx·Σy·Σxy + (Σy)²·Σx²` — so
/// perfectly proportional replicates report EXACTLY zero rather than a float
/// cancellation residue, and the answer cannot depend on summation order.
/// Sums too large for the `u128` products fall back to the float expansion; the
/// choice is a pure function of the sums, so both paths stay deterministic.
///
/// `0.0` when fewer than two replicates, when the arms disagree in length, or
/// when the reference arm measured nothing: no dispersion evidence.
fn ratio_stderr(x: &[u64], y: &[u64]) -> f64 {
    if x.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    let n = x.len() as f64;
    let sum_x: u128 = x.iter().map(|&v| u128::from(v)).sum();
    let sum_y: u128 = y.iter().map(|&v| u128::from(v)).sum();
    if sum_x == 0 {
        return 0.0;
    }
    let sum_xx: u128 = x.iter().map(|&v| u128::from(v) * u128::from(v)).sum();
    let sum_yy: u128 = y.iter().map(|&v| u128::from(v) * u128::from(v)).sum();
    let sum_xy: u128 = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| u128::from(a) * u128::from(b))
        .sum();

    let exact = || -> Option<u128> {
        let a = sum_x.checked_mul(sum_x)?.checked_mul(sum_yy)?;
        let b = sum_x
            .checked_mul(sum_y)?
            .checked_mul(sum_xy)?
            .checked_mul(2)?;
        let c = sum_y.checked_mul(sum_y)?.checked_mul(sum_xx)?;
        // Cauchy–Schwarz guarantees a + c >= b, so this cannot wrap.
        a.checked_add(c)?.checked_sub(b)
    };
    let sx = sum_x as f64;
    let sse = match exact() {
        Some(numerator) => numerator as f64 / (sx * sx),
        None => {
            let r = sum_y as f64 / sx;
            ((sum_yy as f64) - 2.0 * r * (sum_xy as f64) + r * r * (sum_xx as f64)).max(0.0)
        }
    };
    // s² = SSE/(n−1) and n·x̄² = (Σx)²/n, so var = SSE·n / ((n−1)·(Σx)²).
    (sse * n / ((n - 1.0) * sx * sx)).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::density::{DensityAnalysis, ZoomDensity};
    use crate::measure::{measure_sample, measure_sample_layout};
    use geo_types::{Geometry, Point};

    /// The zstd level `stt-build` compresses at by default; every baseline
    /// below sits here so a level-19 trial reproduces the advisor's own
    /// `--publish` comparison.
    const DEFAULT_LEVEL: i32 = 3;

    /// A density result whose deepest zoom publishes the given occupancy
    /// quantiles — the only fields `SyntheticLayout::from_density` reads.
    fn density_with(median: usize, p95: usize, max: usize) -> DensityAnalysis {
        DensityAnalysis {
            per_zoom: vec![ZoomDensity {
                zoom: 12,
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

    fn run_layout() -> SyntheticLayout {
        SyntheticLayout::from_density(&density_with(100, 200, 1_000))
    }

    /// splitmix64 — full-entropy values with no RNG dependency.
    fn mix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `n` jittered points carrying a high-entropy numeric property and a
    /// heavy-repeat categorical one (five categories over the whole sample —
    /// exactly the shape the encoder's dictionary model is most confident
    /// about).
    fn point_sample(n: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| {
                let jitter = |salt: u64| (mix(i as u64 + salt) % 100_000) as f64 * 1e-7;
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
                            PropValue::Text(
                                ["downtown", "plateau", "verdun", "rosemont", "lachine"][i % 5]
                                    .to_string(),
                            ),
                        ),
                    ],
                }
            })
            .collect()
    }

    fn baseline() -> MeasureSettings {
        MeasureSettings {
            zstd_level: DEFAULT_LEVEL,
            ..MeasureSettings::default()
        }
    }

    fn only(results: &[TrialResult]) -> &TrialResult {
        assert_eq!(results.len(), 1, "expected one result: {results:?}");
        &results[0]
    }

    // ------------------------------------------------------------------
    // (1) the zstd sweep reproduces the advisor's publish trial
    // ------------------------------------------------------------------

    #[test]
    fn zstd_19_trial_reproduces_the_sign_of_the_publish_advisor() {
        let sample = point_sample(600);
        let layout = run_layout();
        let base = baseline();

        let trials = run_trials(
            &sample,
            &layout,
            &base,
            &[Candidate::ZstdLevel(19), Candidate::ZstdLevel(3)],
        )
        .unwrap();

        let at_19 = &trials[0];
        assert_eq!(at_19.candidate, Candidate::ZstdLevel(19));
        assert!(
            at_19.delta_bytes < 0 && at_19.delta_frac < 0.0,
            "zstd 19 must measure smaller than the level-3 baseline: {at_19:?}"
        );

        // Moving the lever to where it already is is a no-op, exactly.
        let at_3 = &trials[1];
        assert_eq!(at_3.delta_bytes, 0);
        assert_eq!(at_3.delta_frac, 0.0);

        // THE advisor's own comparison (advisors/layout.rs `publish_advice`):
        // measure the sample at 3 and at 19 and take the shrink. The oracle must
        // agree with it in sign and be the same order of magnitude — it is the
        // same measurement, taken under the run's layout instead of one tile.
        let advisor_3 = measure_sample(&sample, &base).unwrap().unwrap();
        let advisor_19 = measure_sample(
            &sample,
            &MeasureSettings {
                zstd_level: 19,
                ..base.clone()
            },
        )
        .unwrap()
        .unwrap();
        let advisor_frac = (advisor_19.bytes_total as f64 - advisor_3.bytes_total as f64)
            / advisor_3.bytes_total as f64;
        assert!(advisor_frac < 0.0, "advisor trial sign: {advisor_frac}");
        assert!(
            (at_19.delta_frac - advisor_frac).abs() < 0.10,
            "oracle {:.4} vs advisor {:.4}",
            at_19.delta_frac,
            advisor_frac
        );

        // And the whole-tile bytes ARE the run-layout measurement, not a
        // second, differently-cut one.
        let direct = measure_sample_layout(
            &sample,
            &MeasureSettings {
                zstd_level: 19,
                ..base
            },
            &layout,
        )
        .unwrap()
        .unwrap();
        assert_eq!(at_19.bytes_total, direct.bytes_total);
        assert_eq!(at_19.candidate.scope(), TrialScope::WholeTile);
    }

    // ------------------------------------------------------------------
    // (2) the recorded §3.4 overstatement, pinned as a guard
    // ------------------------------------------------------------------

    /// The encoder's pre-compression allowance for an Arrow IPC dictionary
    /// message (`CATEGORICAL_DICTIONARY_IPC_OVERHEAD`, arrow_tile/encode.rs).
    /// Mirrored here so the guard compares against the exact model that decides
    /// the boundary today; it is deliberately never *imported*, because this
    /// test exists to show the model is wrong post-zstd, not to depend on it.
    const MODEL_DICT_IPC_OVERHEAD: usize = 192;

    /// `categorical_dictionary_is_smaller`'s arithmetic, verbatim: the raw
    /// buffer widths the encoder compares BEFORE anything is compressed.
    fn model_gap(values: &[Option<String>]) -> i64 {
        let mut categories: Vec<&str> = Vec::new();
        for v in values.iter().flatten() {
            if !categories.contains(&v.as_str()) {
                categories.push(v.as_str());
            }
        }
        let plain_value_bytes: usize = values.iter().flatten().map(String::len).sum();
        let dictionary_value_bytes: usize = categories.iter().map(|c| c.len()).sum();
        let plain = plain_value_bytes + (values.len() + 1) * size_of::<i32>();
        let dictionary = dictionary_value_bytes
            + (categories.len() + 1) * size_of::<i32>()
            + values.len() * size_of::<u16>()
            + MODEL_DICT_IPC_OVERHEAD;
        dictionary as i64 - plain as i64
    }

    #[test]
    fn dictionary_gap_is_smaller_post_zstd_than_the_encoder_model_claims() {
        // A heavy-repeat Utf8 column: five long category strings over 600 rows.
        // The pre-compression model sees an enormous win for the dictionary,
        // because it charges the plain column the FULL concatenated string
        // bytes. zstd charges it once — so the real, post-zstd gap is a small
        // fraction of the modelled one. That overstatement is the recorded
        // §3.4 defect, and this test is its guard.
        let sample = point_sample(600);
        let layout = run_layout();
        let base = baseline();

        let trials = run_trials(
            &sample,
            &layout,
            &base,
            &[Candidate::CategoricalDict {
                column: "region".to_string(),
                dict: true,
            }],
        )
        .unwrap();
        let dict = only(&trials);
        assert_eq!(dict.candidate.scope(), TrialScope::Column);

        // The model's verdict, evaluated on the SAME chunking the trial used so
        // the per-tile dictionary overhead is charged the same number of times.
        let values = categorical_values(&sample, "region");
        let measured_base =
            measure_sample_layout_with(&sample, &base, &layout, AttributionDesign::SingletonV1)
                .unwrap()
                .unwrap();
        let modelled: i64 = block_ranges(values.len(), measured_base.tiles)
            .into_iter()
            .map(|(s, e)| model_gap(&values[s..e]))
            .sum();

        assert!(
            modelled < 0,
            "fixture guard: the pre-compression model must prefer the dictionary here \
             (gap {modelled} B)"
        );
        assert!(
            dict.delta_bytes.abs() < modelled.abs(),
            "THE §3.4 OVERSTATEMENT: the model claims a {} B gap, the compressor measures only \
             {} B — the oracle must report the measured one",
            modelled.abs(),
            dict.delta_bytes.abs()
        );

        // The two arms are exact mirrors of each other: pricing plain Utf8 must
        // report the same gap with the opposite sign.
        let plain = run_trials(
            &sample,
            &layout,
            &base,
            &[Candidate::CategoricalDict {
                column: "region".to_string(),
                dict: false,
            }],
        )
        .unwrap();
        assert_eq!(only(&plain).delta_bytes, -dict.delta_bytes);
        assert_eq!(
            only(&plain).bytes_total as i64 + dict.delta_bytes,
            dict.bytes_total as i64,
            "the plain arm's bytes plus the measured gap must be the dictionary arm's bytes"
        );

        // A column that is not categorical in the sample is an error, not a
        // silently-zero trial.
        let err = run_trials(
            &sample,
            &layout,
            &base,
            &[Candidate::CategoricalDict {
                column: "magnitude".to_string(),
                dict: true,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("no categorical"),
            "unexpected error: {err}"
        );
    }

    // ------------------------------------------------------------------
    // (3) compact times: both settings through the real encoder, signed delta,
    //     reported on the WIRE side
    // ------------------------------------------------------------------

    #[test]
    fn compact_times_trial_reports_the_signed_wire_delta() {
        let sample = point_sample(600);
        let layout = run_layout();
        let base = baseline();
        assert!(
            base.compact_times,
            "the build default is compact times ON — the baseline must be the incumbent"
        );

        let trials =
            run_trials(&sample, &layout, &base, &[Candidate::CompactTimes(false)]).unwrap();
        let off = only(&trials);

        // The reference is the incumbent (compact ON), the trial is compact OFF,
        // and both went through `encode_tile_with` under the SAME layout.
        let measured_off = measure_sample_layout(
            &sample,
            &MeasureSettings {
                compact_times: false,
                ..base.clone()
            },
            &layout,
        )
        .unwrap()
        .unwrap();
        let measured_on = measure_sample_layout(&sample, &base, &layout)
            .unwrap()
            .unwrap();
        assert_eq!(off.bytes_total, measured_off.bytes_total);
        assert_eq!(
            off.delta_bytes,
            measured_off.bytes_total as i64 - measured_on.bytes_total as i64,
            "the published delta must be the WIRE (post-zstd) difference"
        );
        assert_ne!(
            measured_off.bytes_total, measured_on.bytes_total,
            "the lever must actually move bytes on this fixture, or the test proves nothing"
        );

        // THE recorded inversion this whole mechanism exists for: compact times
        // measured +3.4% WIRE while −13.1% UNCOMPRESSED. `zstd_ratio` is
        // payload/compressed, so the uncompressed sizes are recoverable — and
        // the two sides are allowed to disagree. What is NOT allowed is the
        // oracle reporting the uncompressed side.
        let uncompressed = |m: &MeasuredEncoding| m.bytes_total as f64 * m.zstd_ratio;
        let wire_delta = off.delta_bytes as f64;
        let raw_delta = uncompressed(&measured_off) - uncompressed(&measured_on);
        assert_eq!(
            wire_delta < 0.0,
            off.delta_frac < 0.0,
            "delta_bytes and delta_frac must agree in sign"
        );
        // Even where the two sides AGREE in sign, the pre-compression view is
        // wildly out of scale: here compact times is worth 21.8% uncompressed
        // and 0.52% on the wire — a 42x overstatement. Reporting the raw side
        // would misprice the lever by more than an order of magnitude.
        assert!(
            wire_delta.abs() * 4.0 < raw_delta.abs(),
            "wire {wire_delta} vs uncompressed {raw_delta}: the oracle must report the WIRE side, \
             and these must not be interchangeable"
        );

        // Turning the lever to where it already is measures exactly zero.
        let noop = run_trials(&sample, &layout, &base, &[Candidate::CompactTimes(true)]).unwrap();
        assert_eq!(only(&noop).delta_bytes, 0);
    }

    #[test]
    fn compact_times_inverts_between_wire_and_uncompressed_and_the_oracle_follows_the_wire() {
        // THE RECORDED INVERSION, reproduced. On the fleet it was measured as
        // compact times **+3.4% wire / −13.1% uncompressed**; on this fixture,
        // under a layout of eight equal 75-feature tiles, it comes out as
        //
        //     wire:         22_444 B compact ON  vs 22_220 B compact OFF  → ON costs +1.0%
        //     uncompressed: 37_120 B compact ON  vs 45_184 B compact OFF  → ON saves −17.8%
        //
        // Same shape: the lever WINS on the pre-compression ledger and LOSES on
        // the wire, because the `end_time` column it removes was a byte-for-byte
        // copy of `start_time` that the compressor was matching away for free,
        // while the compact form adds per-tile `TILE_META` state. A model that
        // never runs the compressor cannot see this, which is the entire
        // justification for D4 — and the oracle must report the WIRE sign.
        let sample = point_sample(600);
        let layout = equal_tiles(8);
        let base = baseline();

        let on = measure_sample_layout(&sample, &base, &layout)
            .unwrap()
            .unwrap();
        let off = measure_sample_layout(
            &sample,
            &MeasureSettings {
                compact_times: false,
                ..base.clone()
            },
            &layout,
        )
        .unwrap()
        .unwrap();
        let raw = |m: &MeasuredEncoding| m.bytes_total as f64 * m.zstd_ratio;

        // The fixture really does invert; if this ever stops holding the guard
        // below is vacuous and must be re-based on a fixture that does.
        assert!(
            off.bytes_total < on.bytes_total,
            "fixture guard: compact times must be a WIRE LOSS here ({} on vs {} off)",
            on.bytes_total,
            off.bytes_total
        );
        assert!(
            raw(&on) < raw(&off),
            "fixture guard: compact times must be an UNCOMPRESSED WIN here ({:.0} on vs {:.0} off)",
            raw(&on),
            raw(&off)
        );

        // Baseline = compact ON (the build default); the trial turns it off. The
        // oracle must therefore report a NEGATIVE delta — turning the lever off
        // is a wire saving — which is the opposite of what the uncompressed
        // ledger says.
        let trials =
            run_trials(&sample, &layout, &base, &[Candidate::CompactTimes(false)]).unwrap();
        let trial = only(&trials);
        assert!(
            trial.delta_bytes < 0,
            "the oracle must follow the WIRE, which says turning compact times off saves bytes: \
             {trial:?}"
        );
        assert_eq!(
            trial.delta_bytes,
            off.bytes_total as i64 - on.bytes_total as i64
        );
        // …and the pre-compression ledger, had it been consulted, would have
        // reported the opposite sign.
        assert!(
            raw(&off) - raw(&on) > 0.0,
            "the uncompressed delta must point the other way — that IS the inversion"
        );
    }

    // ------------------------------------------------------------------
    // (4) input order, determinism, and the whole candidate surface
    // ------------------------------------------------------------------

    fn mixed_candidates() -> Vec<Candidate> {
        vec![
            Candidate::ZstdLevel(19),
            Candidate::CompactTimes(false),
            Candidate::QuantizeCoords(Some(0.1)),
            Candidate::QuantizeAttr {
                column: "magnitude".to_string(),
                step: 0.01,
            },
            Candidate::QuantizeAttrsAuto(true),
            Candidate::CategoricalDict {
                column: "region".to_string(),
                dict: true,
            },
            Candidate::FeatureBudgetBytes { max_features: 64 },
            Candidate::QuantizeCoords(None),
        ]
    }

    #[test]
    fn candidates_evaluate_in_input_order_and_deterministically() {
        let sample = point_sample(600);
        let layout = run_layout();
        let base = baseline();
        let candidates = mixed_candidates();

        let a = run_trials(&sample, &layout, &base, &candidates).unwrap();
        let b = run_trials(&sample, &layout, &base, &candidates).unwrap();

        assert_eq!(a.len(), candidates.len());
        for (result, candidate) in a.iter().zip(candidates.iter()) {
            assert_eq!(
                &result.candidate, candidate,
                "results must keep input order"
            );
            assert!(result.bytes_total > 0, "{result:?}");
            assert!(
                result.stderr.is_finite() && result.stderr >= 0.0,
                "{result:?}"
            );
            assert!(result.delta_frac.is_finite(), "{result:?}");
        }

        // THE determinism contract: this vector is solver input.
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
            "Vec<TrialResult> must serialise byte-identically across runs"
        );
        // …and it round-trips, so a report can carry it.
        //
        // ⚠️ The INTEGER fields — which are the ones a solver branches on —
        // round-trip exactly. The `f64` fields round-trip to within one ULP and
        // not always bit-exactly: `serde_json` 1.0.145 ships its fast float
        // parser by default (bit-exact parsing is behind its `float_roundtrip`
        // feature), so `to_string` → `from_str` can move the last mantissa bit.
        // That is a property of the JSON reader, not of this oracle: the
        // determinism contract asserted above is on the SERIALISED BYTES, which
        // are identical. A consumer must not compare re-parsed floats for exact
        // equality.
        let round: Vec<TrialResult> =
            serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(round.len(), a.len());
        for (back, orig) in round.iter().zip(a.iter()) {
            assert_eq!(back.candidate, orig.candidate);
            assert_eq!(back.bytes_total, orig.bytes_total);
            assert_eq!(back.delta_bytes, orig.delta_bytes);
            assert!(
                (back.delta_frac - orig.delta_frac).abs() <= orig.delta_frac.abs() * 1e-15,
                "delta_frac {} vs {}",
                back.delta_frac,
                orig.delta_frac
            );
            assert!(
                (back.stderr - orig.stderr).abs() <= orig.stderr.abs() * 1e-15,
                "stderr {} vs {}",
                back.stderr,
                orig.stderr
            );
        }

        // Lossy classification is a property of the lever, never of the result.
        assert!(Candidate::QuantizeCoords(Some(0.1)).lossy());
        assert!(!Candidate::QuantizeCoords(None).lossy());
        assert!(Candidate::QuantizeAttrsAuto(true).lossy());
        assert!(!Candidate::QuantizeAttrsAuto(false).lossy());
        assert!(Candidate::QuantizeAttr {
            column: "m".into(),
            step: 0.5
        }
        .lossy());
        assert!(!Candidate::ZstdLevel(19).lossy());
        assert!(!Candidate::CompactTimes(false).lossy());
        assert!(!Candidate::FeatureBudgetBytes { max_features: 64 }.lossy());
        assert!(!Candidate::CategoricalDict {
            column: "region".into(),
            dict: true
        }
        .lossy());
    }

    // ------------------------------------------------------------------
    // Register guards
    // ------------------------------------------------------------------

    #[test]
    fn the_zstd_sweep_hard_caps_at_19() {
        let sample = point_sample(200);
        let layout = run_layout();
        let base = baseline();

        for level in [20, 21, 22] {
            let err = run_trials(&sample, &layout, &base, &[Candidate::ZstdLevel(level)])
                .expect_err("level {level} must be rejected");
            let text = err.to_string();
            assert!(text.contains("19"), "level {level}: {text}");
            assert!(
                text.contains("standing rejection"),
                "level {level} must be refused by name: {text}"
            );
        }
        assert!(Candidate::ZstdLevel(MAX_ZSTD_LEVEL).validate().is_ok());
        assert!(Candidate::ZstdLevel(0).validate().is_err());
        assert!(Candidate::ZstdLevel(-1).validate().is_err());

        // The cap is enforced over the WHOLE list before anything is evaluated,
        // so one bad level cannot slip through beside good ones.
        let mixed = run_trials(
            &sample,
            &layout,
            &base,
            &[Candidate::ZstdLevel(9), Candidate::ZstdLevel(22)],
        );
        assert!(mixed.is_err(), "a capped list must be refused wholesale");
    }

    #[test]
    fn invalid_candidates_are_refused_before_anything_encodes() {
        assert!(Candidate::QuantizeCoords(Some(0.0)).validate().is_err());
        assert!(Candidate::QuantizeCoords(Some(f64::NAN))
            .validate()
            .is_err());
        assert!(Candidate::QuantizeCoords(Some(1e-9)).validate().is_err());
        assert!(Candidate::QuantizeCoords(Some(0.1)).validate().is_ok());
        assert!(Candidate::QuantizeCoords(None).validate().is_ok());
        assert!(Candidate::QuantizeAttr {
            column: String::new(),
            step: 0.5
        }
        .validate()
        .is_err());
        assert!(Candidate::QuantizeAttr {
            column: "m".into(),
            step: 0.0
        }
        .validate()
        .is_err());
        assert!(Candidate::CategoricalDict {
            column: String::new(),
            dict: true
        }
        .validate()
        .is_err());
        // A cap of zero features would be thinning by another name.
        let err = Candidate::FeatureBudgetBytes { max_features: 0 }
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("thinning"), "{err}");
    }

    #[test]
    fn the_feature_cap_re_cuts_and_never_drops() {
        // §4.3: a byte cap is priced by re-cutting the SAME features into more
        // tiles. More tiles cost more framing, so the delta is positive — and
        // the feature count is untouched, which is the structural no-thinning
        // guarantee.
        let sample = point_sample(600);
        let layout = run_layout();
        let base = baseline();
        let trials = run_trials(
            &sample,
            &layout,
            &base,
            &[Candidate::FeatureBudgetBytes { max_features: 32 }],
        )
        .unwrap();
        let capped = only(&trials);

        let baseline_measured =
            measure_sample_layout_with(&sample, &base, &layout, AttributionDesign::SingletonV1)
                .unwrap()
                .unwrap();
        let capped_measured = measure_sample_layout_with(
            &sample,
            &base,
            &equal_tiles(div_ceil(baseline_measured.features, 32)),
            AttributionDesign::SingletonV1,
        )
        .unwrap()
        .unwrap();

        assert_eq!(capped.bytes_total, capped_measured.bytes_total);
        assert_eq!(
            capped_measured.features, baseline_measured.features,
            "NO THINNING: the cap re-cuts, it does not drop features"
        );
        assert!(
            capped_measured.tiles > baseline_measured.tiles,
            "a 32-feature cap must produce more tiles than the run layout ({} vs {})",
            capped_measured.tiles,
            baseline_measured.tiles
        );
        assert!(
            capped.delta_bytes > 0,
            "more tiles means more per-tile framing on the wire: {capped:?}"
        );
    }

    // ------------------------------------------------------------------
    // Fallbacks and arithmetic
    // ------------------------------------------------------------------

    #[test]
    fn an_unmeasurable_sample_yields_no_trials() {
        let layout = run_layout();
        let base = baseline();
        // Below MIN_MEASURE_FEATURES: the documented fallback is an empty
        // vector, never an error and never a fabricated number.
        assert!(run_trials(&[], &layout, &base, &mixed_candidates())
            .unwrap()
            .is_empty());
        assert!(run_trials(
            &point_sample(10),
            &layout,
            &base,
            &[Candidate::ZstdLevel(19)]
        )
        .unwrap()
        .is_empty());
        // An empty candidate list is not an error either.
        assert!(run_trials(&point_sample(600), &layout, &base, &[])
            .unwrap()
            .is_empty());
        // …but the cap is still enforced on an unmeasurable sample.
        assert!(run_trials(&[], &layout, &base, &[Candidate::ZstdLevel(22)]).is_err());
    }

    #[test]
    fn stderr_is_a_real_ratio_estimator_with_no_dispersion_evidence_at_zero() {
        // Perfectly proportional replicates: EXACTLY zero, not a cancellation
        // residue.
        assert_eq!(
            ratio_stderr(&[100, 200, 400, 800], &[50, 100, 200, 400]),
            0.0
        );
        // A wobbling ratio is positive.
        let wobble = ratio_stderr(&[100, 100, 100, 100], &[90, 50, 130, 70]);
        assert!(wobble > 0.01, "{wobble}");
        // Order cannot move it.
        assert_eq!(
            ratio_stderr(&[10, 20, 30, 40], &[9, 22, 27, 44]),
            ratio_stderr(&[40, 30, 20, 10], &[44, 27, 22, 9])
        );
        // No evidence → 0.0 rather than a fabricated spread.
        assert_eq!(ratio_stderr(&[100], &[90]), 0.0);
        assert_eq!(ratio_stderr(&[], &[]), 0.0);
        assert_eq!(ratio_stderr(&[0, 0], &[1, 2]), 0.0);
        assert_eq!(ratio_stderr(&[1, 2, 3], &[1, 2]), 0.0);

        // End to end: a real trial on a real sample publishes a finite,
        // plausible spread from the replicate blocks.
        let trials = run_trials(
            &point_sample(600),
            &run_layout(),
            &baseline(),
            &[Candidate::ZstdLevel(19)],
        )
        .unwrap();
        let stderr = only(&trials).stderr;
        assert!(stderr.is_finite() && stderr >= 0.0, "{stderr}");
        assert!(stderr < 0.5, "a spread of {stderr} would swamp any delta");
    }

    #[test]
    fn block_ranges_partition_exactly() {
        for (total, parts) in [(600usize, 4usize), (7, 4), (4, 4), (1, 4), (0, 4), (10, 1)] {
            let ranges = block_ranges(total, parts);
            assert_eq!(ranges.first().map(|r| r.0), Some(0), "{total}/{parts}");
            assert_eq!(ranges.last().map(|r| r.1), Some(total), "{total}/{parts}");
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].1, pair[1].0, "{total}/{parts}: ranges must abut");
            }
            assert_eq!(
                ranges.iter().map(|(s, e)| e - s).sum::<usize>(),
                total,
                "{total}/{parts}"
            );
            // Deterministic.
            assert_eq!(ranges, block_ranges(total, parts));
        }
        assert_eq!(block_ranges(7, 4), vec![(0, 2), (2, 4), (4, 6), (6, 7)]);
    }

    #[test]
    fn measure_settings_defaults_are_todays_build_defaults() {
        // The added knobs must not move a single byte of an existing
        // measurement: `compact_times` is the encoder default and the explicit
        // quantization map starts empty.
        let s = MeasureSettings::default();
        assert!(s.compact_times);
        assert!(s.quantize_attrs.is_empty());
        assert_eq!(s.zstd_level, DEFAULT_LEVEL);
        assert_eq!(s.quantize_coords_m, None);
        assert!(!s.quantize_attrs_auto);
    }

    #[test]
    fn quantize_attr_reaches_the_encoder_as_a_per_column_step() {
        // The §3.3 ladder is a real lever, not a no-op: naming a high-entropy
        // numeric column with a coarse step must shrink the wire.
        let sample = point_sample(600);
        let layout = run_layout();
        let base = baseline();
        let trials = run_trials(
            &sample,
            &layout,
            &base,
            &[
                Candidate::QuantizeAttr {
                    column: "magnitude".to_string(),
                    step: 1.0,
                },
                Candidate::QuantizeAttr {
                    column: "not_a_column".to_string(),
                    step: 1.0,
                },
            ],
        )
        .unwrap();
        assert!(
            trials[0].delta_bytes < 0,
            "quantizing `magnitude` must not grow the wire: {:?}",
            trials[0]
        );
        // Naming a column the data does not have changes nothing — the encoder
        // only ever does keyed lookups into the map.
        assert_eq!(trials[1].delta_bytes, 0);
    }
}
