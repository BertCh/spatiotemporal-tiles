//! `--target-size B`: the archive byte-budget solver.
//!
//! Until this module existed there was **no target archive size anywhere in the
//! CLI**. A publisher who needed a dataset to fit a budget hand-tuned ~10
//! coupled knobs across repeated builds and closed the loop by eye. This turns
//! "hit this byte budget" into a first-class request.
//!
//! # THE RULE THAT OUTRANKS THE FEATURE
//!
//! **Nothing here may drop, sample, or aggregate a single feature.** The
//! feasible set the search may enter — Θ₀ — contains only REVERSIBLE, LOSSLESS
//! levers:
//!
//! | Lever | How it is priced |
//! |---|---|
//! | zoom clamp (`--max-zoom`) | measured bytes/feature at the clamped cut × the density model's per-zoom rows |
//! | temporal bucket width (`--temporal-bucket`) | the density scan re-bucketed, then re-measured |
//! | temporal-LOD tiers (`--temporal-lod`) | [`added_tier_bytes`](BudgetReport) — computed, not assumed (§12.4) |
//! | zstd level (`--publish` / `--zstd-level`) | the §5.1 sweep, hard-capped at [`MAX_ZSTD_LEVEL`] |
//! | blob ordering (`--blob-ordering`) | the existing simulator verdict, caveat and all |
//! | pack size (`--pack-size`) | object-count only; it does not enter Ŝ |
//!
//! The lossy family Θ₁ (coordinate / attribute quantization) **never enters the
//! argmin and never joins the emitted command.** It surfaces only as
//! [`ShadowPrice`]s: "here is what this would additionally buy", for a human to
//! choose. [`ChosenLever`] has no `lossy` field *at all* — a lossy lever cannot
//! be represented as a chosen one, which is the structural form of the
//! guarantee.
//!
//! When the budget is INFEASIBLE even at the lexicographic floor over Θ₀, the
//! correct behaviour is to **report the floor and drop nothing**:
//! `feasible: false`, the floor recipe, [`BudgetReport::floor_bytes`], and the
//! shadow-price table showing what each human-gated lossy lever would buy. A
//! solver that quietly starts shedding features to hit a number is the single
//! worst failure this codebase can have; `infeasible_budget_reports_the_floor_and_drops_nothing`
//! is the test that says so.
//!
//! # The size oracle
//!
//! Ŝ has **two measured coefficients and no fitted formula**:
//!
//! ```text
//! Ŝ(θ) = Σ_{z ≤ zmax(θ)} [  a(level) × (tiles(z, Δ) − 1)
//!                         + b(level) × (g × N + (1 − g) × instances(z, Δ)) ]
//!        + Σ_j added_tier_bytes(tier_j)
//! ```
//!
//! | symbol | what it is | how it is obtained |
//! |---|---|---|
//! | `a` | compressed bytes an extra tile costs | difference of two real encodes of the SAME sample cut into different tile counts ([`measure_size_model`]) |
//! | `b` | compressed bytes a feature's content costs | a real one-tile encode of the sample ÷ its feature count |
//! | `g` | geometry's share of encoded bytes | the run's leave-one-out per-column attribution |
//! | `tiles`, `instances` | the real cut at `(z, Δ)` | the occupancy scan, times the clip replication [`sample_replication`] measures off real geometries |
//!
//! ## Why the tile term had to exist
//!
//! Before it did, Ŝ was `total_features × bytes_per_feature` — proportional to
//! feature count, intercept zero — and it missed by **7.4× on `wpc-fronts` and
//! 10.4× on `osm-nyc`**, in both cases reporting FITS at ~25–42% of budget for
//! an archive that built to 250–307% of it. The reason is structural, not a
//! tuning miss: `osm-nyc` builds 301,406 tiles for 380,007 features, so 1.26
//! features share each tile and ~91% of the archive is per-tile framing that a
//! per-feature rate cannot see. With the tile term the same two projections land
//! at **0.95× and 1.02× of realized bytes**.
//!
//! ## Why this is not the rejected analytic size model
//!
//! The standing rejection is of size models *derived* rather than measured —
//! fleet wins ranged 1.07×–21× with no predictive formula behind them. Both
//! coefficients here are read off real encodes of this run's own sample, through
//! the production encoder and the production compressor, in the packed
//! schema-template mode a real build writes. The only structure imposed is that
//! an extra tile costs a roughly constant amount — and the probe MEASURES how
//! badly that holds (a third encode, a second chord, [`SizeModel::bytes_per_tile_stderr`])
//! instead of asserting it. What was analytic before, and unmarked as such, was
//! the *form* `bytes ∝ features, intercept 0`.
//!
//! If extrapolation misses badly on a dataset class, the recorded fallback is
//! still to **widen the sample**, never to fit a formula.
//!
//! # The search: enumerate, do not sweep
//!
//! §5.2's distortion vector is lexicographic, most-protected component first:
//! `(feature loss, D_zoom, D_geo, D_attr, D_time)`. Over Θ₀ the feature-loss
//! component is structurally 0 and the two quantization components are
//! structurally 0, so a distortion class is exactly the pair
//! `(D_zoom, D_time)` — [`DistortionClass`], whose derived `Ord` IS §5.2's
//! order.
//!
//! The ladders are tiny ([`MAX_ZOOM_CLAMP_STEPS`] × [`BUCKET_MULTIPLIERS`] ≈ 12
//! points), so the search is a lexicographic ENUMERATION: walk the classes
//! ascending, take the min-Ŝ point inside each (the best zstd level, no tiers),
//! and the first class whose best point satisfies `Ŝ(θ) ≤ B` wins.
//!
//! ⚠️ This is deliberately **not** an MCKP / Lagrangian sweep. At these
//! cardinalities such a sweep degenerates to exactly this enumeration, and
//! enumeration is deterministic by construction — no multiplier search, no
//! convergence tolerance, no tie-breaking by float comparison of dual values.
//! Simpler and provably equivalent wins.
//!
//! # Honesty about error
//!
//! [`BudgetReport::projected_stderr`] carries BOTH error terms, in quadrature:
//! the block-replicate dispersion of the measured bytes/feature, and the framing
//! model's own measured curvature multiplied by the projected tile count. The
//! second term is the one that matters, and it used to be missing — which is how
//! a projection wrong by 7.4× could be published behind ±0.94%. A bar that
//! describes only how repeatable a measurement is, attached to a number whose
//! error is dominated by extrapolation, is worse than no bar.
//!
//! The report also states the extrapolation FACTOR in words — how many times
//! past the probe's own tile count the projection reaches — so a reader can see
//! the reach rather than infer it. When `|B − Ŝ|` is inside 2σ,
//! [`BudgetReport::within_noise`] is set and the report says so. A solver that
//! claims to hit a budget it cannot distinguish from missing is lying.
//!
//! # Determinism
//!
//! [`solve`] is a pure function of `(result, data, layout, target_bytes)`: no
//! RNG, no clock, no HashMap iteration in any output-affecting path (the
//! occupancy scan's map is consumed into a vector that
//! [`SyntheticLayout::from_feature_counts`] sorts), every ladder is a `const`,
//! every sort carries a total tiebreak. Two runs serialise byte-identically —
//! required, not aspirational: a recipe becomes build flags, and pack names in
//! this format are content-addressed.

use std::collections::{BTreeMap, HashMap};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use stt_core::projection;

use crate::advisors::layout::PUBLISH_ZSTD_LEVEL;
use crate::advisors::temporal::LOD_BUCKET_THRESHOLD;
use crate::advisors::{layout as layout_advisor, Advice, Composer};
use crate::analysis::density::{
    measure_size_model, sample_replication, scale_count, Replication, SizeModel,
};
use crate::analysis::AnalysisResult;
use crate::attribution::AttributionDesign;
use crate::loader::{LoadedData, PropValue, SampledFeature};
use crate::measure::{measure_sample_layout_with, MeasureSettings, SyntheticLayout};
use crate::oracle::{Candidate, TrialResult, MAX_ZSTD_LEVEL, REPLICATE_BLOCKS};

// ----------------------------------------------------------------------------
// Ladders and constants — every search dimension is a named `const`
// ----------------------------------------------------------------------------

/// zstd levels the solver evaluates inside a distortion class, ascending.
///
/// ⚠️ **The sweep hard-caps at [`MAX_ZSTD_LEVEL`] (19).** Level 22 is a standing
/// rejection — measured against 19, the wire difference did not survive the
/// encode-time cost — so 19 is a ceiling and not a starting point. The cap is
/// not merely respected by this table: every level is pushed through
/// [`Candidate::validate`] before a byte is compressed, so an out-of-range level
/// cannot reach the compressor even if someone edits the table.
pub const ZSTD_SWEEP: &[i32] = &[3, 9, MAX_ZSTD_LEVEL];

/// `--quantize-coords` rungs priced as shadow prices, coarsest first.
///
/// §5.2's ladder, mirrored (the quantize advisor keeps its own private copy of
/// the same list). Coarsest first so the table reads as "most bytes bought,
/// most precision surrendered" at the top.
pub const COORD_PRECISION_LADDER_M: [f64; 6] = [5.0, 1.0, 0.5, 0.1, 0.05, 0.01];

/// How many zoom levels the clamp ladder is allowed to remove.
///
/// Three, because the zoom clamp is the biggest lever in the format (every zoom
/// re-ships the whole dataset) and a four-level clamp on a typical z0–10
/// recommendation is already a different dataset. A clamp is never allowed to
/// push `max_zoom` below `min_zoom`.
pub const MAX_ZOOM_CLAMP_STEPS: u8 = 3;

/// Temporal bucket widths the solver may propose, as multiples of the
/// recommended base bucket.
///
/// Strict multiples keep every downstream `--temporal-lod` entry legal (the
/// builder requires tiers to be multiples of the base bucket) and keep
/// `D_time = Δ − Δ₀` a small, totally-ordered ladder.
pub const BUCKET_MULTIPLIERS: &[u64] = &[1, 2, 4];

/// Candidate `--temporal-lod` widths, as multiples of the CHOSEN base bucket.
///
/// Base-relative rather than absolute calendar durations, and that is a
/// deliberate difference from `advisors::temporal`'s fixed `CLEAN_TIERS`.
/// M3 replaces the fixed ladder with a budgeted greedy over `(width, cutoff)`
/// pairs, and a base-relative ladder is what makes candidates EXIST at all on a
/// coarse base bucket: the calendar ladder tops out at `30d` and every entry
/// must divide the base exactly, so a dataset whose recommended bucket is `1
/// week` — like the repo's own osm-nyc-changesets, 1068 buckets and the longest
/// timeline in the local corpus — admits no calendar tier at all while it does
/// admit base-relative ones. Every rung is a strict multiple of the base and at
/// least 4× the previous one, so the emitted spec always satisfies the builder's
/// multiple-of-base and ascending checks.
const TIER_MULTIPLIERS: &[u64] = &[4, 16, 64];

/// Most `--temporal-lod` tiers the solver will propose.
const MAX_TIERS: usize = 2;

/// Standard errors `B − Ŝ` must clear before the report will claim the budget
/// was hit rather than merely not-distinguishably-missed.
const NOISE_SIGMA: f64 = 2.0;

/// Numeric columns given their own `--quantize-attr` shadow price.
///
/// Two: the per-column lever is dominated by `--quantize-attrs-auto` for most
/// datasets, and every extra rung costs `1 + REPLICATE_BLOCKS` sample encodes.
const SHADOW_PRICE_COLUMNS: usize = 2;

/// Flags whose verdict the budget solver OWNS once a target size is set.
///
/// When a [`BudgetReport`] is present these advisor entries are dropped from the
/// emitted command and replaced by the solver's own choices — otherwise a
/// command could carry `--publish` (zstd 19) while the budget was projected at
/// level 3, and the build would silently undershoot the number the user was
/// shown. Scalars (`--min-zoom`/`--max-zoom`/`--temporal-bucket`) are handled
/// separately because they are positional parts of the command, not appended
/// flags.
pub const BUDGET_GOVERNED_FLAGS: &[&str] = &[
    "--publish",
    "--zstd-level",
    "--temporal-lod",
    "--blob-ordering",
    "--pack-size",
];

// ----------------------------------------------------------------------------
// Report types
// ----------------------------------------------------------------------------

/// One point of §5.2's lexicographic distortion ladder, restricted to Θ₀.
///
/// The derived `Ord` **is** the distortion order: `zoom_clamp` (D_zoom) is more
/// protected than `bucket_multiplier` (D_time), so temporal resolution is spent
/// before spatial depth. The quantization components (D_geo, D_attr) are
/// structurally zero here — that is what makes this a two-field struct rather
/// than the full five-vector.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default, Hash,
)]
pub struct DistortionClass {
    /// Zoom levels removed from the recommended `max_zoom` (§5.2's `D_zoom`).
    pub zoom_clamp: u8,
    /// Temporal bucket width as a multiple of the recommended base bucket
    /// (§5.2's `D_time`, expressed as the ratio rather than the difference so
    /// the ladder is scale-free). `1` is no distortion.
    pub bucket_multiplier: u64,
}

impl DistortionClass {
    /// The undistorted class: full zoom depth, native bucket width.
    pub fn none() -> Self {
        Self {
            zoom_clamp: 0,
            bucket_multiplier: 1,
        }
    }

    /// Whether this class distorts anything at all.
    pub fn is_none(&self) -> bool {
        *self == Self::none()
    }
}

/// How [`BudgetReport::projected_bytes`] was arrived at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EstimateBasis {
    /// The sample was measurable: every candidate point was re-encoded through
    /// the real encoder + compressor and the density estimate was calibrated by
    /// the measured ratio.
    Measured,
    /// The sample sat below the measurement floor, so the solver deferred to the
    /// density model's own per-zoom estimate. Only the zoom clamp moves Ŝ in
    /// this mode; the report says so and nothing pretends otherwise.
    DensityEstimate,
}

/// One lever the solver CHOSE. Structurally non-lossy: there is no `lossy`
/// field because a lossy lever can never be a chosen one — and the one seam
/// that turns [`Advice`] into this type ([`ChosenLever::try_from`]) REFUSES
/// lossy advice rather than dropping the field, so that sentence is enforced
/// and not merely asserted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChosenLever {
    /// The `stt-build` flag (must exist in docs/api/cli-reference.md).
    pub flag: String,
    /// Flag value, `None` for bare switches.
    pub value: Option<String>,
    /// Evidence-based rationale, with this dataset's numbers.
    pub why: String,
    /// Signed byte effect at the chosen point: negative saves, positive adds
    /// (a temporal-LOD tier is additive and says so). `None` when the lever's
    /// effect is not a byte effect (ordering, pack size).
    pub delta_bytes: Option<i64>,
    /// Non-lossy but still not auto-applicable: the lever carries a tradeoff a
    /// human must decide.
    ///
    /// ⚠️ A playback-caveated `spatial` blob ordering stays `suggestion_only`
    /// **even under budget pressure**: `blobOrdering: spatial` silently breaks
    /// time-playback buffering (empty buffered ranges → stalls), and budget
    /// pressure is not a reason to ship a broken demo.
    #[serde(default)]
    pub suggestion_only: bool,
}

impl TryFrom<Advice> for ChosenLever {
    type Error = anyhow::Error;

    /// The ONLY way advice becomes a chosen lever, and it is fallible on
    /// exactly one condition.
    ///
    /// # Why this is a `TryFrom` and not a field-by-field `map`
    ///
    /// [`ChosenLever`] has no `lossy` field, and the whole no-thinning
    /// guarantee leans on that: a lossy lever is not *representable* as a chosen
    /// one. But [`Advice`] does have the field, so any hand-written mapping from
    /// `Advice` to `ChosenLever` DROPS it — and a dropped `lossy: true` is not a
    /// type guarantee, it is a silent promotion of a lossy lever into the
    /// auto-applied recipe. That was the real state of the seam (F9): safe only
    /// because `advisors::layout` happens to be `lossy: false` throughout, i.e.
    /// by convention, at the exact place the docs claim a type does the work.
    ///
    /// Refusing the conversion closes it in every build profile (a
    /// `debug_assert!` would not hold in release), at the single seam, and the
    /// error names the flag so the failure is diagnosable rather than mysterious.
    /// `suggestion_only` is PROPAGATED verbatim for the same reason: the gate is
    /// a property of the lever, never of the budget it is being asked to hit.
    fn try_from(advice: Advice) -> Result<Self> {
        if advice.lossy {
            bail!(
                "internal: `{}` is LOSSY advice and cannot become a chosen lever — Θ₁ never \
                 enters the budget argmin, it is priced as a shadow price and left to a human. \
                 A lossy lever reaching the recipe would drop or degrade data to hit a byte \
                 budget, which is the one thing this solver may never do.",
                advice.flag
            );
        }
        Ok(ChosenLever {
            flag: advice.flag,
            value: advice.value,
            why: advice.why,
            // Ordering and pack size move dedup / object count, not Ŝ.
            delta_bytes: None,
            suggestion_only: advice.suggestion_only,
        })
    }
}

/// What one human-gated LOSSY lever would additionally buy at θ\*.
///
/// ⚠️ Every entry is `lossy: true`, by construction. A shadow price is
/// information, never an instruction: nothing in this vector reaches
/// [`crate::recommend::to_command`], `stt-build --auto`, or any automated path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowPrice {
    /// The `stt-build` flag a human would add by hand.
    pub flag: String,
    /// Flag value, `None` for bare switches.
    pub value: Option<String>,
    /// Archive bytes this lever would SHED at θ\* (positive = saves). Negative
    /// means the lever would make the archive bigger — priced and reported
    /// rather than hidden.
    pub marginal_bytes: i64,
    /// Measured fractional change of the sample encode (negative = smaller).
    pub delta_frac: f64,
    /// Standard error of `delta_frac` from the trial oracle's replicate blocks.
    /// `0.0` means NO dispersion evidence, never "no noise".
    pub stderr: f64,
    /// Always `true`. Serialized so a consumer reading only this table cannot
    /// mistake it for advice.
    pub lossy: bool,
    /// What was measured, and what the quality cost is.
    pub why: String,
}

/// The answer to "make this archive fit `target_bytes`".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetReport {
    /// The requested budget, in bytes.
    pub target_bytes: u64,
    /// Ŝ(θ\*) — projected archive bytes for the emitted recipe.
    pub projected_bytes: u64,
    /// Total standard error of `projected_bytes`: measurement dispersion and
    /// framing extrapolation, combined in quadrature.
    ///
    /// - **measurement dispersion** — how far the encoder's bytes/feature moves
    ///   between comparable stretches of the sample (block replicates).
    /// - **framing extrapolation** — [`SizeModel::bytes_per_tile_stderr`], the
    ///   measured curvature of the per-tile cost, carried out to the projected
    ///   TILE COUNT. This term grows with the reach of the extrapolation, which
    ///   is the thing that actually goes wrong.
    ///
    /// ⚠️ It still cannot quantify how well a 5,000-feature sample stands in for
    /// a whole dataset's CONTENT — only how far the tile term is being carried.
    /// The report states the extrapolation factor in words alongside it. `0.0`
    /// means no dispersion evidence at all, never "no noise".
    pub projected_stderr: f64,
    /// Whether the budget is reachable over Θ₀ at all.
    pub feasible: bool,
    /// Whether `|target_bytes − projected_bytes|` is inside
    /// [`NOISE_SIGMA`]·`projected_stderr` — i.e. hitting and missing the budget
    /// are not distinguishable at this measurement's precision.
    pub within_noise: bool,
    /// The distortion class θ\* sits in.
    pub distortion: DistortionClass,
    /// The recipe, in emit order. Never lossy (see [`ChosenLever`]).
    pub chosen: Vec<ChosenLever>,
    /// Ŝ at the FLOOR: the smallest archive the reversible levers can reach,
    /// over the whole re-measured frontier.
    ///
    /// This is the smallest the archive gets without a human opting into a lossy
    /// lever, and it is what an infeasible budget is measured against — and what
    /// an infeasible budget EMITS, so the recipe a user is handed is never one
    /// that distorts more for no byte gain.
    pub floor_bytes: u64,
    /// The floor's distortion class: the LEAST distortion that reaches
    /// [`Self::floor_bytes`]. Ŝ is not monotone in distortion, so this is not
    /// necessarily the last rung of the ladder.
    pub floor_distortion: DistortionClass,
    /// What each human-gated lossy lever would additionally buy, sorted by
    /// `marginal_bytes` descending. Every entry `lossy: true`.
    pub shadow_prices: Vec<ShadowPrice>,
    /// zstd levels actually evaluated, ascending. Never contains a level above
    /// [`MAX_ZSTD_LEVEL`]; empty when the sample was unmeasurable.
    pub zstd_sweep: Vec<i32>,
    /// Distortion classes evaluated before the answer was settled.
    pub classes_evaluated: usize,
    /// How `projected_bytes` was arrived at.
    pub basis: EstimateBasis,
    /// Plain-language caveats: the estimate's basis, the no-thinning statement
    /// on an infeasible budget, the noise warning, and any tier that did not
    /// fit. Rendered verbatim under the table.
    pub notes: Vec<String>,
}

impl BudgetReport {
    /// `projected_bytes` as a fraction of `target_bytes` (1.0 = exactly on).
    pub fn utilization(&self) -> f64 {
        if self.target_bytes == 0 {
            return f64::INFINITY;
        }
        self.projected_bytes as f64 / self.target_bytes as f64
    }

    /// The headline line: what was asked, what is projected, and whether the
    /// difference survives the measurement's own noise.
    pub fn headline(&self) -> String {
        let verdict = if self.feasible {
            "FITS"
        } else {
            "DOES NOT FIT (floor reported; nothing dropped)"
        };
        let noise = if self.within_noise {
            format!(
                " — ⚠️ within {NOISE_SIGMA:.0}σ of the projection's own error (±{:.0} B): hitting \
                 and missing this budget are not distinguishable at this precision",
                self.projected_stderr
            )
        } else if self.projected_stderr > 0.0 {
            format!(" (±{:.0} B)", self.projected_stderr)
        } else {
            String::new()
        };
        format!(
            "{verdict}: projected {} B vs target {} B ({:.1}% of budget){noise}",
            self.projected_bytes,
            self.target_bytes,
            self.utilization() * 100.0,
        )
    }
}

// ----------------------------------------------------------------------------
// Size parsing
// ----------------------------------------------------------------------------

/// Parse a `--target-size` value: bare bytes, or a `K`/`M`/`G` (binary) or
/// `KB`/`MB`/`GB` (decimal) suffix; `KiB`/`MiB`/`GiB` are accepted spellings of
/// the binary forms.
///
/// Case-insensitive, `_` digit separators allowed, a fractional mantissa
/// allowed (`1.5G`). The binary/decimal split follows the usual convention and
/// is documented in `docs/api/cli-reference.md` so nobody has to guess which
/// `M` they got.
pub fn parse_size(text: &str) -> Result<u64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        bail!("--target-size needs a value, e.g. `250MiB`, `1.5G`, or `262144000`");
    }
    let cleaned: String = trimmed.chars().filter(|c| *c != '_' && *c != ',').collect();
    let split = cleaned
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(cleaned.len());
    let (mantissa, unit) = cleaned.split_at(split);
    if mantissa.is_empty() {
        bail!("--target-size `{text}` has no number before its unit");
    }
    let value: f64 = mantissa
        .parse()
        .with_context(|| format!("--target-size `{text}`: `{mantissa}` is not a number"))?;
    if !value.is_finite() || value <= 0.0 {
        bail!("--target-size `{text}` must be a positive size");
    }
    let multiplier: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kib" => 1024.0,
        "kb" => 1_000.0,
        "m" | "mib" => 1024.0 * 1024.0,
        "mb" => 1_000_000.0,
        "g" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "gb" => 1_000_000_000.0,
        other => bail!(
            "--target-size `{text}`: unknown unit `{other}`. Use bytes, or one of \
             K/KiB/KB, M/MiB/MB, G/GiB/GB (bare K/M/G are binary, the `B` forms decimal)"
        ),
    };
    let bytes = value * multiplier;
    if !bytes.is_finite() || bytes >= u64::MAX as f64 {
        bail!("--target-size `{text}` overflows a 64-bit byte count");
    }
    let bytes = bytes.round() as u64;
    if bytes == 0 {
        bail!("--target-size `{text}` rounds to zero bytes");
    }
    Ok(bytes)
}

// ----------------------------------------------------------------------------
// The solver
// ----------------------------------------------------------------------------

/// Solve for `target_bytes` over the reversible lever set Θ₀.
///
/// `layout` is the run's own [`SyntheticLayout`] (the one `result.measured` was
/// taken under). The solver derives its OWN cut per candidate point — the
/// temporal bucket width and the zoom clamp both change real tile occupancy, so
/// here the cut IS one of the levers, the same documented exception
/// [`Candidate::FeatureBudgetBytes`] carries in the trial oracle. `layout` is
/// the fallback for a dataset whose occupancy scan yields nothing.
///
/// # Errors
///
/// Only on a genuine encoder/compressor failure. An unmeasurable sample is NOT
/// an error: the solver falls back to the density model's own per-zoom estimate,
/// says so in [`BudgetReport::basis`], and prices only the zoom clamp.
pub fn solve(
    result: &AnalysisResult,
    data: &LoadedData,
    layout: &SyntheticLayout,
    target_bytes: u64,
) -> Result<BudgetReport> {
    let mut sizer = Sizer::new(result, data, layout)?;
    let classes = candidate_classes(result);

    // Re-measure the WHOLE frontier — every class, at its min-Ŝ zstd level.
    // ~12 points, each one real encode per level, cached on `(cut, level)`.
    // Evaluating all of them rather than stopping early is what makes the report
    // path-independent: the floor below is a genuine minimum over Θ₀ instead of
    // "the smallest thing the search happened to visit before it stopped".
    let mut evaluated: Vec<(DistortionClass, Point)> = Vec::with_capacity(classes.len());
    for class in &classes {
        let point = sizer.best_point(*class)?;
        evaluated.push((*class, point));
    }

    // The winner: the FIRST class in lexicographic order that fits. Not the
    // smallest — §5.2 minimises DISTORTION subject to `Ŝ ≤ B`, so a class that
    // fits is preferred to any more-distorted one however much smaller it is.
    //
    // The predicate is [`choose_class`] and nothing else. It used to be
    // reimplemented inline here while `choose_class` carried the monotonicity
    // property in a test that only ever exercised the copy — so the property
    // guarded a function production did not call (F10). Calling it is what makes
    // "the single place feasibility becomes a choice" true rather than merely
    // claimed.
    let class_costs: Vec<u64> = evaluated.iter().map(|(_, p)| p.bytes).collect();
    let winner = choose_class(&class_costs, target_bytes).map(|i| evaluated[i].clone());

    // The floor: the smallest Ŝ the reversible levers can reach, at the LEAST
    // distortion that reaches it.
    //
    // ⚠️ Not simply "the most distorted class". Ŝ is not monotone in distortion
    // — a 4× wider bucket can measure a few hundred bytes LARGER once the
    // compressor sees fatter tiles — and reporting the maximum-distortion point
    // as "the floor" would then hand a user a recipe that surrenders temporal
    // resolution for nothing. `min_by` keeps the first minimum and the ladder is
    // in ascending order, so ties resolve toward less distortion; the explicit
    // tiebreak says so rather than relying on that.
    let (floor_class, floor_point) = evaluated
        .iter()
        .min_by(|a, b| a.1.bytes.cmp(&b.1.bytes).then_with(|| a.0.cmp(&b.0)))
        .cloned()
        .expect("the ladder always holds (0, 1)");

    let feasible = winner.is_some();
    let (class, point) = winner.unwrap_or((floor_class, floor_point.clone()));

    // ---- the recipe -------------------------------------------------------
    let mut notes: Vec<String> = Vec::new();
    notes.push(sizer.basis_note());

    // A search that could not reach its biggest lever has to say so. Reporting
    // `classes_evaluated: 3` while the zoom clamp was structurally absent is how
    // a solver comes to look busy and change nothing (F6/F8).
    if classes.iter().all(|c| c.zoom_clamp == 0) {
        notes.push(format!(
            "⚠️ The ZOOM CLAMP — the biggest reversible byte lever there is, since every zoom \
             level re-ships the whole dataset — was NOT searchable on this dataset. The \
             recommended range is z{}-{}, over which the search sees {} clampable zoom level(s), \
             so there is no zoom level a clamp could remove; only the {} temporal-bucket class(es) \
             below were evaluated. A budget that needs more than bucket coarsening can give will \
             report INFEASIBLE for that reason alone.",
            result.spatial.recommended_min_zoom,
            result.spatial.recommended_max_zoom,
            clampable_depth(result) as usize + 1,
            classes.len(),
        ));
    }

    let mut chosen: Vec<ChosenLever> = Vec::new();
    let base_zmax = result.spatial.recommended_max_zoom;
    let base_bucket = sizer.base_bucket_ms;
    let zmax = base_zmax.saturating_sub(class.zoom_clamp);
    let bucket_ms = base_bucket.saturating_mul(class.bucket_multiplier);

    if class.zoom_clamp > 0 {
        // Marginal of the clamp, holding the bucket at the chosen width.
        let unclamped = sizer.point_at(
            DistortionClass {
                zoom_clamp: 0,
                bucket_multiplier: class.bucket_multiplier,
            },
            point.zstd_level,
        )?;
        chosen.push(ChosenLever {
            flag: "--max-zoom".to_string(),
            value: Some(zmax.to_string()),
            why: format!(
                "clamped {} → {} to fit the budget: every zoom level re-ships the whole dataset, \
                 so {} level(s) of depth is the biggest reversible byte lever there is. Every \
                 feature is still present at every kept zoom — nothing is dropped, sampled or \
                 aggregated.",
                base_zmax, zmax, class.zoom_clamp
            ),
            delta_bytes: Some(point.bytes as i64 - unclamped.bytes as i64),
            suggestion_only: false,
        });
    }
    if class.bucket_multiplier > 1 {
        let native = sizer.point_at(
            DistortionClass {
                zoom_clamp: class.zoom_clamp,
                bucket_multiplier: 1,
            },
            point.zstd_level,
        )?;
        chosen.push(ChosenLever {
            flag: "--temporal-bucket".to_string(),
            value: Some(bucket_ms.to_string()),
            why: format!(
                "widened the temporal bucket {}× ({} ms → {} ms): coarser buckets pack more \
                 features per tile, so per-tile encoder and zstd framing is amortised further. \
                 Lossless — the same features, cut into fewer tiles.",
                class.bucket_multiplier, base_bucket, bucket_ms
            ),
            delta_bytes: Some(point.bytes as i64 - native.bytes as i64),
            suggestion_only: false,
        });
    }
    if let Some(lever) = sizer.zstd_lever(class, &point)? {
        chosen.push(lever);
    }

    // ---- temporal-LOD tiers, priced against the remaining budget ----------
    let mut projected = point.bytes;
    if feasible {
        let remaining = target_bytes.saturating_sub(point.bytes);
        let (tier_lever, tier_notes) =
            sizer.tier_lever(result, class, point.zstd_level, remaining)?;
        notes.extend(tier_notes);
        if let Some(lever) = tier_lever {
            projected = projected.saturating_add(lever.delta_bytes.unwrap_or(0).max(0) as u64);
            chosen.push(lever);
        }
    } else {
        notes.push(
            "temporal-LOD tiers were not priced: they are ADDITIVE (a lossless coarse-bucket \
             replica of every feature at every zoom in range), and there is no budget left to \
             spend on one."
                .to_string(),
        );
    }

    // ---- ordering + pack size, inherited from the existing advisor --------
    chosen.extend(inherited_layout_levers(result, data)?);

    // ---- error bars, feasibility verdict, shadow prices -------------------
    let projected_stderr = sizer.projection_stderr(class, point.zstd_level, projected)?;
    let gap = (target_bytes as f64) - (projected as f64);
    let within_noise = projected_stderr > 0.0 && gap.abs() <= NOISE_SIGMA * projected_stderr;

    // How far past the probe's own tile count this projection reaches. Stated in
    // words rather than buried in the ± so nobody reads a 589× extrapolation as
    // a measurement.
    let reach = sizer.extrapolation_factor(class, point.zstd_level)?;
    if reach > 1.0 && reach.is_finite() {
        notes.push(format!(
            "The projection extrapolates {reach:.0}× past the tile count that was actually \
             measured: the framing probe encoded the sample across a few hundred tiles, the recipe \
             projects an archive of many more. The ±{projected_stderr:.0} B bar carries that reach \
             — it is the probe's own measured curvature multiplied by the projected tile count, \
             not a margin chosen to look prudent. Against two real end-to-end builds this \
             projection landed within 6% of realized bytes; treat it as a projection with that \
             kind of error, not as a measurement."
        ));
    }

    if !feasible {
        notes.push(format!(
            "INFEASIBLE over the reversible lever set. The floor — the smallest archive Θ₀ can \
             reach, at the least distortion that reaches it (max-zoom {}, {}× bucket), measured \
             over {} candidate points — still measures {} B, which is {} B over the {} B budget. \
             NOTHING HAS BEEN DROPPED: no feature cap, no sampling, no aggregation, and no lossy \
             flag is in the emitted command. The shadow-price table below is what a human could \
             choose to spend quality on.",
            base_zmax.saturating_sub(floor_class.zoom_clamp),
            floor_class.bucket_multiplier,
            evaluated.len(),
            floor_point.bytes,
            floor_point.bytes.saturating_sub(target_bytes),
            target_bytes,
        ));
    }
    if within_noise {
        notes.push(format!(
            "The {:.0} B gap between the projection and the budget is inside {NOISE_SIGMA:.0}σ of \
             the projection's own error (±{:.0} B: measurement dispersion and framing \
             extrapolation, combined). Treat this as \"about the right size\", not as a hit.",
            gap.abs(),
            projected_stderr
        ));
    }

    let shadow_prices = sizer.shadow_prices(result, class, point.zstd_level, projected)?;
    if shadow_prices.is_empty() {
        notes.push(
            "No lossy lever could be priced on this sample, so the shadow-price table is empty. \
             That is an absence of evidence, not evidence that quantization would not help."
                .to_string(),
        );
    }

    Ok(BudgetReport {
        target_bytes,
        projected_bytes: projected,
        projected_stderr,
        feasible,
        within_noise,
        distortion: class,
        chosen,
        floor_bytes: floor_point.bytes,
        floor_distortion: floor_class,
        shadow_prices,
        zstd_sweep: sizer.levels_evaluated(),
        classes_evaluated: evaluated.len(),
        basis: sizer.basis(),
        notes,
    })
}

/// The distortion ladder for this dataset, ascending in §5.2's lexicographic
/// order.
///
/// Bounded by the data: a clamp may not push `max_zoom` below `min_zoom`, and a
/// widened bucket must still leave at least two buckets over the dataset's span
/// (a one-bucket archive has no timeline left to play).
pub fn candidate_classes(result: &AnalysisResult) -> Vec<DistortionClass> {
    let max_clamp = MAX_ZOOM_CLAMP_STEPS.min(clampable_depth(result));
    let base_bucket = result.temporal.recommended_bucket_ms;
    let duration = result.temporal.duration_ms;

    let mut out = Vec::new();
    for clamp in 0..=max_clamp {
        for &multiplier in BUCKET_MULTIPLIERS {
            if multiplier > 1 {
                // Skip a widening the timeline cannot absorb.
                let widened = base_bucket.saturating_mul(multiplier);
                if base_bucket == 0 || widened == 0 || duration / widened < 2 {
                    continue;
                }
            }
            out.push(DistortionClass {
                zoom_clamp: clamp,
                bucket_multiplier: multiplier,
            });
        }
    }
    if out.is_empty() {
        out.push(DistortionClass::none());
    }
    // Derived `Ord` is §5.2's order; the ladder is built in it, and this makes
    // that a checked property rather than a construction accident.
    out.sort();
    out
}

/// How many zoom levels the clamp ladder may remove on this dataset.
///
/// The bound is the span the SIZE ORACLE can actually price — the number of
/// per-zoom density rows inside the recommended range, minus one — intersected
/// with the recommended range itself. A clamp step that removes no priced row
/// cannot save a byte, so pricing it would be theatre: [`Sizer::kept_bytes`]
/// would report the identical sum and the class would look free.
///
/// # ⚠️ F6: this is why `--target-size` used to do nothing
///
/// Bounding the ladder by `recommended_max_zoom − recommended_min_zoom` is
/// correct, but it was silently zero on every real dataset because the zoom
/// recommender collapsed `min` onto `max` (wpc-fronts z8-8, osm-nyc z18-18,
/// cpc-rainfall z11-11). Zero depth ⇒ zero clamp steps ⇒ the search enumerated
/// bucket multipliers only, and `--target-size 1KiB` emitted the same recipe as
/// a plain `--auto encode` — byte-identical archives. The recommender is fixed
/// (see `analysis::spatial::overview_min_zoom`); this function keeps the bound
/// honest from the oracle's side, and [`solve`] now SAYS SO in the report when
/// the ladder comes back empty instead of quietly reporting three classes.
fn clampable_depth(result: &AnalysisResult) -> u8 {
    let zmin = result.spatial.recommended_min_zoom;
    let zmax = result.spatial.recommended_max_zoom;
    let recommended = zmax.saturating_sub(zmin);
    let priced = result
        .density
        .per_zoom
        .iter()
        .filter(|z| z.zoom >= zmin && z.zoom <= zmax)
        .count();
    let priced_depth = u8::try_from(priced.saturating_sub(1)).unwrap_or(u8::MAX);
    recommended.min(priced_depth)
}

/// Pick the first class in `costs` (already in lexicographic order) whose
/// projected size fits `target_bytes`.
///
/// This is the ONLY place feasibility turns into a choice — [`solve`] calls it,
/// it is not a parallel copy of the rule — and the monotonicity property
/// (`B1 < B2` ⇒ class at `B1` ≥ class at `B2`) is a property of exactly this
/// function: the feasible set of classes shrinks as `B` shrinks, so the first
/// feasible index can only move later. Returns `None` when nothing fits.
///
/// ⚠️ It was factored out and then NOT called: `solve` reimplemented the
/// predicate inline, so the property test guarded dead code while the live rule
/// was unpinned. The call site above is what discharges that (F10), and
/// `the_solver_chooses_through_choose_class_and_not_a_copy_of_it` pins that the
/// two cannot drift apart again.
pub fn choose_class(costs: &[u64], target_bytes: u64) -> Option<usize> {
    costs.iter().position(|&bytes| bytes <= target_bytes)
}

// ----------------------------------------------------------------------------
// The size oracle
// ----------------------------------------------------------------------------

/// One evaluated point: a distortion class at its min-Ŝ zstd level.
#[derive(Debug, Clone, PartialEq)]
struct Point {
    zstd_level: i32,
    /// Ŝ for the class at that level, before any additive tier.
    bytes: u64,
    /// Ŝ for the same class at the build-default zstd level — the reference the
    /// zstd lever's `delta_bytes` is signed against.
    bytes_at_default: u64,
}

/// Measurement + extrapolation state for one `solve` call.
struct Sizer<'a> {
    sample: &'a [SampledFeature],
    features: &'a [crate::loader::AnalyzableFeature],
    /// `(zoom, estimated_size_compressed)` from the density model, ascending —
    /// the DensityEstimate fallback only. With a measurable sample the solver
    /// counts its own occupancy and prices it with [`SizeModel`].
    zoom_rows: Vec<(u8, u64)>,
    /// Zoom levels the recommended range covers, ascending.
    zooms: Vec<u8>,
    min_zoom: u8,
    base_zmax: u8,
    base_bucket_ms: u64,
    /// Distinct input features — the denominator of every content charge.
    source_features: u64,
    /// Measured geometry share of encoded bytes; splits the content charge
    /// between conserved (geometry) and per-instance (ids/times/properties)
    /// bytes. See [`SizeModel::geometry_share`].
    geometry_share: f64,
    fallback_layout: &'a SyntheticLayout,
    layouts: BTreeMap<(u8, u64), SyntheticLayout>,
    /// `(tiles, tiled instances)` per `(zoom, bucket_ms)`, replication-aware.
    occupancy: BTreeMap<(u8, u64), (u64, u64)>,
    /// Measured clip replication per `(zoom, bucket_ms)`.
    replication: BTreeMap<(u8, u64), Replication>,
    /// The measured size model per zstd level. `None` inside the option means
    /// "probed and the sample was too small", which is a different thing from
    /// "not probed yet".
    models: BTreeMap<i32, Option<SizeModel>>,
    /// bytes/feature cache, keyed on `((zoom, bucket_ms), zstd_level)`. Still
    /// used by the shadow-price and stderr paths, which price RATIOS.
    bpf: BTreeMap<((u8, u64), i32), Option<f64>>,
    /// bytes/feature at the run's default cut and level. `None` when the sample
    /// is unmeasurable.
    denominator: Option<f64>,
    levels: Vec<i32>,
}

impl<'a> Sizer<'a> {
    fn new(
        result: &AnalysisResult,
        data: &'a LoadedData,
        fallback_layout: &'a SyntheticLayout,
    ) -> Result<Self> {
        // Per-zoom rows straight off the density model — the same numbers
        // `analyze` prints, so the solver and the report never disagree about
        // what the archive costs today. An empty model (possible only on a
        // degenerate analysis) collapses to one row carrying the whole estimate.
        let mut zoom_rows: Vec<(u8, u64)> = result
            .density
            .per_zoom
            .iter()
            .map(|z| (z.zoom, z.estimated_size_compressed as u64))
            .collect();
        if zoom_rows.is_empty() {
            zoom_rows.push((
                result.spatial.recommended_max_zoom,
                result.density.estimated_archive_size as u64,
            ));
        }
        zoom_rows.sort_by_key(|(z, _)| *z);
        let zooms: Vec<u8> = zoom_rows.iter().map(|(z, _)| *z).collect();

        // The geometry share comes from the run's leave-one-out attribution when
        // it has one. Absent it the share is 1.0 = "content is conserved when a
        // feature is clipped", which is exactly right for point data and the
        // non-inflating reading for everything else.
        let geometry_share = result
            .measured
            .as_ref()
            .map(|m| {
                let share: f64 = m
                    .per_column
                    .iter()
                    .filter(|c| c.name == "geometry")
                    .map(|c| c.share)
                    .sum();
                if share.is_finite() && share > 0.0 {
                    share.clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
            .unwrap_or(1.0);

        let mut sizer = Self {
            sample: &data.sample,
            features: &data.features,
            zoom_rows,
            zooms,
            min_zoom: result.spatial.recommended_min_zoom,
            base_zmax: result.spatial.recommended_max_zoom,
            base_bucket_ms: result.temporal.recommended_bucket_ms,
            source_features: data.features.len() as u64,
            geometry_share,
            fallback_layout,
            layouts: BTreeMap::new(),
            occupancy: BTreeMap::new(),
            replication: BTreeMap::new(),
            models: BTreeMap::new(),
            bpf: BTreeMap::new(),
            denominator: None,
            levels: Vec::new(),
        };
        let base_level = MeasureSettings::default().zstd_level;
        sizer.denominator =
            sizer.bytes_per_feature(sizer.base_zmax, sizer.base_bucket_ms, base_level)?;
        // Probe the framing model up front so `basis` is settled before any
        // point is evaluated, and so a probe failure surfaces here rather than
        // half-way through the frontier.
        sizer.model_for(base_level)?;
        Ok(sizer)
    }

    /// The measured [`SizeModel`] at one zstd level, probed once and cached.
    ///
    /// The zstd cap is enforced through the trial oracle's own validator before
    /// a byte is compressed, so a level above [`MAX_ZSTD_LEVEL`] cannot reach the
    /// compressor from this path either.
    fn model_for(&mut self, level: i32) -> Result<Option<SizeModel>> {
        Candidate::ZstdLevel(level)
            .validate()
            .context("budget solver: framing probe")?;
        if let Some(hit) = self.models.get(&level) {
            return Ok(*hit);
        }
        let model = measure_size_model(self.sample, level, self.geometry_share)
            .context("budget solver: framing probe")?;
        self.levels.push(level);
        self.models.insert(level, model);
        Ok(model)
    }

    fn basis(&self) -> EstimateBasis {
        let base_level = MeasureSettings::default().zstd_level;
        match self.models.get(&base_level) {
            Some(Some(_)) => EstimateBasis::Measured,
            _ => EstimateBasis::DensityEstimate,
        }
    }

    fn basis_note(&self) -> String {
        let base_level = MeasureSettings::default().zstd_level;
        match (
            self.basis(),
            self.models.get(&base_level).copied().flatten(),
        ) {
            (EstimateBasis::Measured, Some(model)) => format!(
                "Ŝ = measured framing × tile count + measured content × feature count. BOTH \
                 coefficients come from real encodes of this run's {}-feature sample through the \
                 production encoder and compressor, in the packed schema-template mode a real \
                 build writes: {:.0} B per tile (probed at {} tiles) and {:.1} B per feature \
                 (one-tile encode). No analytic size model is involved and no formula is fitted — \
                 the tile term exists because a thin-tiled archive is mostly framing, which a \
                 per-feature rate cannot see. The residual risk is extrapolation from the sample's \
                 tile count to the archive's; that is what the ± bar carries, and if it misses on \
                 your data the fix is to widen the sample, not to fit a formula.",
                self.sample.len(),
                model.bytes_per_tile,
                model.probe_tiles,
                model.bytes_per_feature,
            ),
            _ => format!(
                "⚠️ The {}-feature sample sits below the framing probe's floor, so no per-tile \
                 cost could be measured. The projection falls back to the density model's own \
                 per-zoom estimate, which has NO TILE TERM and under-projects a thin-tiled archive \
                 badly (measured 7×–10× low on two real datasets); only the zoom clamp moves it, \
                 and bucket width and zstd level are reported unpriced. Treat every number here as \
                 an estimate, not a measurement.",
                self.sample.len()
            ),
        }
    }

    fn levels_evaluated(&self) -> Vec<i32> {
        let mut out = self.levels.clone();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The synthetic cut real tiles would have at `(zoom, bucket_ms)`.
    ///
    /// Derived from the SAME occupancy scan the density model makes — every
    /// feature into its containing `(x, y, t)` cell — so a coarser bucket
    /// genuinely produces fatter tiles rather than a rescaled guess. The
    /// counting map's iteration order cannot leak:
    /// [`SyntheticLayout::from_feature_counts`] sorts what it is handed.
    fn layout_for(&mut self, zoom: u8, bucket_ms: u64) -> SyntheticLayout {
        if let Some(hit) = self.layouts.get(&(zoom, bucket_ms)) {
            return hit.clone();
        }
        let mut cells: HashMap<(u32, u32, u64), usize> = HashMap::new();
        for feature in self.features {
            if let Ok((x, y)) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
                let t = if bucket_ms > 0 {
                    feature.timestamp / bucket_ms
                } else {
                    0
                };
                *cells.entry((x, y, t)).or_insert(0) += 1;
            }
        }
        let counts: Vec<usize> = cells.into_values().collect();
        let layout = if counts.is_empty() {
            self.fallback_layout.clone()
        } else {
            SyntheticLayout::from_feature_counts(&counts)
        };
        self.layouts.insert((zoom, bucket_ms), layout.clone());
        layout
    }

    /// Measured compressed bytes per feature at one `(cut, level)` point.
    ///
    /// The zstd cap is enforced HERE, through the trial oracle's own validator,
    /// so a level above [`MAX_ZSTD_LEVEL`] cannot reach the compressor no matter
    /// which ladder it came from.
    fn bytes_per_feature(&mut self, zoom: u8, bucket_ms: u64, level: i32) -> Result<Option<f64>> {
        Candidate::ZstdLevel(level)
            .validate()
            .context("budget solver: zstd sweep")?;
        let key = ((zoom, bucket_ms), level);
        if let Some(hit) = self.bpf.get(&key) {
            return Ok(*hit);
        }
        let layout = self.layout_for(zoom, bucket_ms);
        let settings = MeasureSettings {
            zstd_level: level,
            ..MeasureSettings::default()
        };
        // Trials read only `bytes_total`/`bytes_per_feature`, which
        // `measure_sample_layout_with` documents as attribution-independent, so
        // the cheap singleton pass is the right cost choice.
        let measured = measure_sample_layout_with(
            self.sample,
            &settings,
            &layout,
            AttributionDesign::SingletonV1,
        )?;
        let value = measured.map(|m| m.bytes_per_feature);
        self.levels.push(level);
        self.bpf.insert(key, value);
        Ok(value)
    }

    /// Summed density estimate over the zooms a clamp keeps.
    fn kept_bytes(&self, zoom_clamp: u8) -> u64 {
        let zmax = self.base_zmax.saturating_sub(zoom_clamp);
        self.zoom_rows
            .iter()
            .filter(|(z, _)| *z <= zmax && *z >= self.min_zoom)
            .map(|(_, b)| *b)
            .sum()
    }

    /// Tiles and tiled feature instances the real build would cut at
    /// `(zoom, bucket_ms)`.
    ///
    /// This is the SAME scan `analysis::density` makes — every feature into its
    /// containing `(x, y, t)` cell — with the builder's clip replication applied
    /// on top from [`sample_replication`], so a trajectory that crosses eight
    /// tiles is counted as eight tiles and eight instances rather than one of
    /// each. Against the two real builds this reproduces the tile count exactly
    /// (7,405 on `wpc-fronts`, 301,406 on `osm-nyc`).
    ///
    /// The counting map's iteration order cannot leak: only cardinalities and
    /// sums are read.
    fn occupancy_for(&mut self, zoom: u8, bucket_ms: u64) -> (u64, u64) {
        if let Some(hit) = self.occupancy.get(&(zoom, bucket_ms)) {
            return *hit;
        }
        let mut cells: HashMap<(u32, u32, u64), usize> = HashMap::new();
        let mut placed = 0usize;
        for feature in self.features {
            if let Ok((x, y)) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
                let t = if bucket_ms > 0 {
                    feature.timestamp / bucket_ms
                } else {
                    0
                };
                *cells.entry((x, y, t)).or_insert(0) += 1;
                placed += 1;
            }
        }
        let replication = self.replication_for(zoom, bucket_ms);
        let instances = scale_count(placed as u64, replication.instances_per_feature);
        let tiles = scale_count(cells.len() as u64, replication.cell_inflation)
            .clamp(cells.len() as u64, instances.max(cells.len() as u64));
        self.occupancy.insert((zoom, bucket_ms), (tiles, instances));
        (tiles, instances)
    }

    /// Measured clip replication at one cut, cached.
    fn replication_for(&mut self, zoom: u8, bucket_ms: u64) -> Replication {
        if let Some(hit) = self.replication.get(&(zoom, bucket_ms)) {
            return *hit;
        }
        let replication = sample_replication(self.sample, zoom, bucket_ms);
        self.replication.insert((zoom, bucket_ms), replication);
        replication
    }

    /// The zooms a clamp keeps, ascending.
    fn kept_zooms(&self, zoom_clamp: u8) -> Vec<u8> {
        let zmax = self.base_zmax.saturating_sub(zoom_clamp);
        self.zooms
            .iter()
            .copied()
            .filter(|z| *z <= zmax && *z >= self.min_zoom)
            .collect()
    }

    /// Ŝ for one `(class, level)` pair.
    ///
    /// Every kept zoom re-ships the whole dataset, so each one is priced
    /// independently and summed: `Σ_z model.project(tiles_z, instances_z, N)`.
    ///
    /// Without a measurable framing model this falls back to the density
    /// estimate unscaled — the pre-M3 behaviour, which [`Self::basis_note`]
    /// announces as an estimate with no tile term rather than quoting as a
    /// projection.
    fn size_at(&mut self, class: DistortionClass, level: i32) -> Result<u64> {
        let bucket = self.base_bucket_ms.saturating_mul(class.bucket_multiplier);
        let Some(model) = self.model_for(level)? else {
            return Ok(self.kept_bytes(class.zoom_clamp));
        };
        let mut total: u64 = 0;
        for zoom in self.kept_zooms(class.zoom_clamp) {
            let (tiles, instances) = self.occupancy_for(zoom, bucket);
            total = total.saturating_add(model.project(tiles, instances, self.source_features));
        }
        Ok(total)
    }

    /// The modelling half of Ŝ's uncertainty at one `(class, level)` pair: the
    /// probe's measured curvature carried out to the projected tile count, plus
    /// the model's measured misfit against its own probe cuts. Summed over the
    /// kept zooms, because each one is projected independently.
    fn framing_stderr(&mut self, class: DistortionClass, level: i32) -> Result<f64> {
        let bucket = self.base_bucket_ms.saturating_mul(class.bucket_multiplier);
        let Some(model) = self.model_for(level)? else {
            return Ok(0.0);
        };
        let mut total = 0.0;
        for zoom in self.kept_zooms(class.zoom_clamp) {
            let (tiles, instances) = self.occupancy_for(zoom, bucket);
            let projected = model.project(tiles, instances, self.source_features);
            total += model.projection_stderr(tiles, projected);
        }
        Ok(total)
    }

    /// How far past the probe's measured tile count the winning point reaches.
    fn extrapolation_factor(&mut self, class: DistortionClass, level: i32) -> Result<f64> {
        let bucket = self.base_bucket_ms.saturating_mul(class.bucket_multiplier);
        let Some(model) = self.model_for(level)? else {
            return Ok(0.0);
        };
        let mut tiles = 0u64;
        for zoom in self.kept_zooms(class.zoom_clamp) {
            tiles = tiles.saturating_add(self.occupancy_for(zoom, bucket).0);
        }
        Ok(model.extrapolation_factor(tiles))
    }

    /// The min-Ŝ point inside a class: the best zstd level, no tiers.
    ///
    /// Ties keep the LOWEST level — cheaper to build, identical on the wire.
    fn best_point(&mut self, class: DistortionClass) -> Result<Point> {
        let default_level = MeasureSettings::default().zstd_level;
        let bytes_at_default = self.size_at(class, default_level)?;
        let mut best = Point {
            zstd_level: default_level,
            bytes: bytes_at_default,
            bytes_at_default,
        };
        if self.model_for(default_level)?.is_none() {
            // Nothing to sweep: without a framing model every level projects the
            // same bytes, and pretending otherwise would be a fabricated number.
            return Ok(best);
        }
        for &level in ZSTD_SWEEP {
            let bytes = self.size_at(class, level)?;
            if bytes < best.bytes || (bytes == best.bytes && level < best.zstd_level) {
                best.bytes = bytes;
                best.zstd_level = level;
            }
        }
        Ok(best)
    }

    /// Ŝ for a class at a fixed level, as a `Point` (used for lever marginals).
    fn point_at(&mut self, class: DistortionClass, level: i32) -> Result<Point> {
        let default_level = MeasureSettings::default().zstd_level;
        Ok(Point {
            zstd_level: level,
            bytes: self.size_at(class, level)?,
            bytes_at_default: self.size_at(class, default_level)?,
        })
    }

    /// The zstd lever the min-Ŝ point picked, or `None` when it picked the
    /// build default (nothing to say, and nothing to add to the command).
    fn zstd_lever(&mut self, class: DistortionClass, point: &Point) -> Result<Option<ChosenLever>> {
        let default_level = MeasureSettings::default().zstd_level;
        if point.zstd_level == default_level {
            return Ok(None);
        }
        let zmax = self.base_zmax.saturating_sub(class.zoom_clamp);
        let bucket = self.base_bucket_ms.saturating_mul(class.bucket_multiplier);
        let (flag, value) = if point.zstd_level == PUBLISH_ZSTD_LEVEL {
            ("--publish".to_string(), None)
        } else {
            (
                "--zstd-level".to_string(),
                Some(point.zstd_level.to_string()),
            )
        };
        Ok(Some(ChosenLever {
            flag,
            value,
            why: format!(
                "the per-dataset zstd sweep {:?} measured level {} smallest at this cut (z{} \
                 max, {} ms buckets): {} B vs {} B at the build default level {}. Exact \
                 re-encoding — decode is level-independent, so this is free on the client. The \
                 sweep is capped at {MAX_ZSTD_LEVEL}: level 22 is a standing rejection \
                 (19 ≈ 22 measured, at several times the encode cost).",
                ZSTD_SWEEP,
                point.zstd_level,
                zmax,
                bucket,
                point.bytes,
                point.bytes_at_default,
                default_level,
            ),
            delta_bytes: Some(point.bytes as i64 - point.bytes_at_default as i64),
            suggestion_only: false,
        }))
    }

    /// Pick `--temporal-lod` tiers by budgeted greedy over `(width, cutoff)`
    /// pairs — the §12.4 fix.
    ///
    /// A tier is a LOSSLESS but ADDITIVE replica: the builder re-tiles every
    /// feature at the coarse bucket across every zoom in
    /// `[min_zoom, min(max_zoom, cutoff)]`. Its byte cost was never computed
    /// before, which is exactly why tiers were being "recommended for free".
    /// Here each candidate width is measured at its own cut and each cutoff is
    /// priced off the density rows, and a tier is proposed only if it FITS the
    /// budget left over after the base recipe.
    fn tier_lever(
        &mut self,
        result: &AnalysisResult,
        class: DistortionClass,
        level: i32,
        remaining: u64,
    ) -> Result<(Option<ChosenLever>, Vec<String>)> {
        let mut notes = Vec::new();
        let zmax = self.base_zmax.saturating_sub(class.zoom_clamp);
        let bucket = self.base_bucket_ms.saturating_mul(class.bucket_multiplier);
        let duration = result.temporal.duration_ms;
        if bucket == 0 || duration == 0 {
            return Ok((None, notes));
        }
        let bucket_count = duration / bucket;
        if bucket_count <= LOD_BUCKET_THRESHOLD {
            return Ok((None, notes));
        }

        let mut budget_left = remaining;
        let mut entries: Vec<(String, u64)> = Vec::new();
        let mut declined: Option<(String, u64)> = None;
        for &multiplier in TIER_MULTIPLIERS {
            if entries.len() == MAX_TIERS {
                break;
            }
            let Some(width_ms) = bucket.checked_mul(multiplier) else {
                break;
            };
            // A tier needs at least two buckets of its own, or it is a single
            // whole-timeline tile pretending to be a pyramid level.
            if duration / width_ms < 2 {
                break;
            }
            let label = duration_label(width_ms);
            let model = self.model_for(level)?;
            // Deepest cutoff that fits: the tier is worth most where it reaches
            // furthest, so spend the remaining budget on depth before width.
            //
            // A tier re-tiles every feature at `width_ms` across every zoom in
            // `[min_zoom, cutoff]`, so its price is the SAME two-term projection
            // the base recipe gets, evaluated at the tier's own coarser cut —
            // which is exactly the byte cost §12.4 recorded as "never computed".
            let mut accepted: Option<(u8, u64)> = None;
            let mut cutoff = zmax;
            loop {
                let added = match model {
                    Some(model) => {
                        let mut sum = 0u64;
                        for zoom in self.zooms.clone() {
                            if zoom > cutoff || zoom < self.min_zoom {
                                continue;
                            }
                            let (tiles, instances) = self.occupancy_for(zoom, width_ms);
                            sum = sum.saturating_add(model.project(
                                tiles,
                                instances,
                                self.source_features,
                            ));
                        }
                        sum
                    }
                    None => self
                        .zoom_rows
                        .iter()
                        .filter(|(z, _)| *z <= cutoff && *z >= self.min_zoom)
                        .map(|(_, b)| *b)
                        .sum(),
                };
                if added <= budget_left {
                    accepted = Some((cutoff, added));
                    break;
                }
                if cutoff <= self.min_zoom {
                    if declined.is_none() {
                        declined = Some((label.clone(), added));
                    }
                    break;
                }
                cutoff -= 1;
            }
            if let Some((cutoff, added)) = accepted {
                let entry = if cutoff >= zmax {
                    label.clone()
                } else {
                    format!("{label}@{cutoff}")
                };
                entries.push((entry, added));
                budget_left = budget_left.saturating_sub(added);
            }
        }

        if let Some((label, added)) = declined {
            notes.push(format!(
                "A `{label}` temporal-LOD tier was priced at {added} B even clamped to zoom {} — \
                 more than the {} B the budget had left — so it was NOT recommended. Tiers are \
                 additive; before this solver they were recommended without their byte cost ever \
                 being computed.",
                self.min_zoom, budget_left
            ));
        }
        if entries.is_empty() {
            return Ok((None, notes));
        }

        let total: u64 = entries.iter().map(|(_, b)| *b).sum();
        let value = entries
            .iter()
            .map(|(label, _)| label.clone())
            .collect::<Vec<_>>()
            .join(",");
        let priced = entries
            .iter()
            .map(|(label, bytes)| format!("{label} +{bytes} B"))
            .collect::<Vec<_>>()
            .join(", ");
        Ok((
            Some(ChosenLever {
                flag: "--temporal-lod".to_string(),
                value: Some(value),
                why: format!(
                    "{bucket_count} base buckets at the chosen {bucket} ms width, so a coarse \
                     pyramid caps zoomed-out request counts. Each tier is a LOSSLESS but ADDITIVE \
                     replica and was priced against the budget's remaining {remaining} B before \
                     being proposed: {priced}. Every feature is preserved in every tier.",
                ),
                delta_bytes: Some(total as i64),
                suggestion_only: false,
            }),
            notes,
        ))
    }

    /// The projection's TOTAL error bar: measurement dispersion and framing
    /// extrapolation, combined in quadrature.
    ///
    /// # Why it is no longer only the measurement's dispersion
    ///
    /// It used to be, and that was the more dangerous half of a bad number: on
    /// `wpc-fronts` the solver reported ±0.94% around a projection that the real
    /// build missed by 7.4×. A bar that only describes how repeatable a
    /// measurement is, attached to a projection whose error is dominated by
    /// EXTRAPOLATION, tells the reader the opposite of the truth.
    ///
    /// So the bar now carries both terms:
    ///
    /// - **measurement dispersion** — the [`REPLICATE_BLOCKS`] contiguous
    ///   single-tile blocks the trial oracle uses, taking the dispersion of
    ///   bytes/feature itself (the oracle's own `stderr` is a ratio estimator,
    ///   right for a lever's delta and wrong for an absolute size) and
    ///   propagating it onto Ŝ;
    /// - **framing extrapolation** — [`SizeModel::bytes_per_tile_stderr`], the
    ///   measured gap between the probe's two finest chords, carried out to the
    ///   projected tile count. This is the term that grows when the model is
    ///   asked to reach far past what it measured, which is exactly when it
    ///   should.
    ///
    /// Both are measured. Neither is a safety margin picked to look prudent, and
    /// the note the report prints names the extrapolation factor so a reader can
    /// see how long the reach was.
    ///
    /// `0.0` means no dispersion evidence at all, never "no noise".
    fn projection_stderr(
        &mut self,
        class: DistortionClass,
        level: i32,
        projected: u64,
    ) -> Result<f64> {
        let framing = self.framing_stderr(class, level)?;
        let measurement = self.measurement_stderr(level, projected)?;
        let total = (framing * framing + measurement * measurement).sqrt();
        Ok(if total.is_finite() { total } else { 0.0 })
    }

    /// Block-replicate dispersion of bytes/feature, propagated onto Ŝ.
    fn measurement_stderr(&self, level: i32, projected: u64) -> Result<f64> {
        if self.denominator.is_none() || self.sample.len() < REPLICATE_BLOCKS * 2 {
            return Ok(0.0);
        }
        let settings = MeasureSettings {
            zstd_level: level,
            ..MeasureSettings::default()
        };
        // The replicates are measured SINGLE-TILE, exactly as the trial oracle
        // measures its own: a quarter of the sample cannot fill the class's
        // multi-tile cut, so cutting it that way would price the tile split
        // rather than the block. They estimate how far bytes/feature MOVES
        // between comparable stretches of the sample, and that dispersion is
        // then applied to Ŝ — which was measured at the class's real cut.
        let single = SyntheticLayout::single();
        let mut per_block: Vec<f64> = Vec::with_capacity(REPLICATE_BLOCKS);
        let n = self.sample.len();
        for i in 0..REPLICATE_BLOCKS {
            let start = n * i / REPLICATE_BLOCKS;
            let end = n * (i + 1) / REPLICATE_BLOCKS;
            if end <= start {
                return Ok(0.0);
            }
            let measured = measure_sample_layout_with(
                &self.sample[start..end],
                &settings,
                &single,
                AttributionDesign::SingletonV1,
            )?;
            match measured {
                Some(m) if m.bytes_per_feature.is_finite() => per_block.push(m.bytes_per_feature),
                // A block under the measurement floor would silently re-weight
                // the estimate, so the whole thing is withdrawn.
                _ => return Ok(0.0),
            }
        }
        let k = per_block.len();
        if k < 2 {
            return Ok(0.0);
        }
        let mean = per_block.iter().sum::<f64>() / k as f64;
        if mean <= 0.0 || !mean.is_finite() {
            return Ok(0.0);
        }
        let variance = per_block.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k as f64 - 1.0);
        let stderr = (variance / k as f64).sqrt();
        Ok(projected as f64 * (stderr / mean))
    }

    /// Price every Θ₁ lossy lever AT THE COMPOSED SETTINGS.
    ///
    /// This closes a recorded blindness: quantization wins had only ever been
    /// measured at the build-default zstd level 3, never at the composed point a
    /// publish recipe actually runs at — and quantized coordinates compress very
    /// differently at level 19 than at level 3, in both directions.
    ///
    /// The trials run through [`Composer`] (MO-6's cache over the MO-5 oracle) at
    /// θ\*'s own cut and level. A candidate can be INFEASIBLE on a given dataset
    /// without being invalid (a step finer than the fixed-point leaf can
    /// address), which the oracle reports as an error naming the candidate — so
    /// a failed batch is retried rung by rung and the impossible rungs are
    /// dropped rather than aborting the whole table.
    fn shadow_prices(
        &mut self,
        result: &AnalysisResult,
        class: DistortionClass,
        level: i32,
        projected: u64,
    ) -> Result<Vec<ShadowPrice>> {
        if self.denominator.is_none() {
            return Ok(Vec::new());
        }
        let zmax = self.base_zmax.saturating_sub(class.zoom_clamp);
        let bucket = self.base_bucket_ms.saturating_mul(class.bucket_multiplier);
        let theta = MeasureSettings {
            zstd_level: level,
            ..MeasureSettings::default()
        };
        let candidates = self.lossy_candidates(result);
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let theta_layout = self.layout_for(zmax, bucket);
        let mut composer = Composer::new(self.sample, &theta_layout);
        let priced: Vec<TrialResult> = match composer.price(&theta, &candidates) {
            Ok(results) => results,
            Err(_) => {
                // At least one rung is unaddressable on this data. Re-price one
                // at a time and keep what the encoder can express.
                let mut kept = Vec::new();
                for candidate in &candidates {
                    if let Ok(results) = composer.price(&theta, std::slice::from_ref(candidate)) {
                        kept.extend(results);
                    }
                }
                kept
            }
        };

        let mut out: Vec<ShadowPrice> = priced
            .into_iter()
            .map(|trial| {
                let marginal = (-trial.delta_frac * projected as f64).round();
                let marginal_bytes = if marginal.is_finite() {
                    marginal as i64
                } else {
                    0
                };
                let (flag, value, quality) = describe_lossy(&trial.candidate);
                let stderr_clause = if trial.stderr > 0.0 {
                    format!(" ±{:.1}pp", trial.stderr * 100.0)
                } else {
                    " (no dispersion evidence)".to_string()
                };
                ShadowPrice {
                    flag,
                    value,
                    marginal_bytes,
                    delta_frac: trial.delta_frac,
                    stderr: trial.stderr,
                    lossy: true,
                    why: format!(
                        "measured on the composed recipe (zstd {level}, z{zmax} max, {bucket} ms \
                         buckets): {:+.1}% sample encode{stderr_clause} ⇒ ~{marginal_bytes} \
                         archive bytes. {quality} OPT-IN ONLY — this flag is not in the emitted \
                         command and `stt-build --auto` will never apply it.",
                        trial.delta_frac * 100.0
                    ),
                }
            })
            .collect();
        // Biggest win first, with a total tiebreak so the table is byte-stable.
        out.sort_by(|a, b| {
            b.marginal_bytes
                .cmp(&a.marginal_bytes)
                .then_with(|| a.flag.cmp(&b.flag))
                .then_with(|| a.value.cmp(&b.value))
        });
        Ok(out)
    }

    /// The Θ₁ ladder to price: the coordinate rungs, the auto attribute lever,
    /// and per-column explicit steps for the costliest numeric columns.
    ///
    /// Rungs the oracle would refuse outright are dropped HERE rather than left
    /// to the per-candidate retry. §5.2's ladder bottoms out at `0.01 m`, which
    /// is below the encoder's `i32` world-grid floor (`MIN_QUANTIZE_COORDS_M`
    /// ≈ 0.0187 m) on every dataset — so without this filter the batch call
    /// would fail every single time and the fallback path would become the
    /// normal one. The retry still exists, for infeasibility that depends on the
    /// DATA (an attribute step finer than the column's range can address) rather
    /// than on the candidate.
    fn lossy_candidates(&self, result: &AnalysisResult) -> Vec<Candidate> {
        let mut candidates: Vec<Candidate> = COORD_PRECISION_LADDER_M
            .iter()
            .map(|&m| Candidate::QuantizeCoords(Some(m)))
            .collect();
        candidates.push(Candidate::QuantizeAttrsAuto(true));
        for (column, step) in self.numeric_columns(result) {
            candidates.push(Candidate::QuantizeAttr { column, step });
        }
        candidates.retain(|candidate| candidate.validate().is_ok());
        candidates
    }

    /// Fractional numeric columns worth an explicit `--quantize-attr` price,
    /// ranked by measured cost (falling back to sample order), capped at
    /// [`SHADOW_PRICE_COLUMNS`].
    ///
    /// The step is the width a `UInt16` leaf would resolve the column's own
    /// observed range at, snapped up to a 1/2/5×10^k value so the flag a human
    /// pastes is a number they can reason about.
    fn numeric_columns(&self, result: &AnalysisResult) -> Vec<(String, f64)> {
        let mut ranges: BTreeMap<String, (f64, f64)> = BTreeMap::new();
        for feature in self.sample {
            for (name, value) in &feature.properties {
                if let PropValue::Number(x) = value {
                    if x.is_finite() && x.fract() != 0.0 {
                        let entry = ranges.entry(name.clone()).or_insert((*x, *x));
                        entry.0 = entry.0.min(*x);
                        entry.1 = entry.1.max(*x);
                    }
                }
            }
        }
        let cost_of = |name: &str| -> u64 {
            result
                .measured
                .as_ref()
                .and_then(|m| m.per_column.iter().find(|c| c.name == name))
                .map(|c| c.compressed_bytes as u64)
                .unwrap_or(0)
        };
        let mut ranked: Vec<(String, f64)> = ranges
            .into_iter()
            .filter_map(|(name, (lo, hi))| {
                let span = hi - lo;
                if !span.is_finite() || span <= 0.0 {
                    return None;
                }
                Some((name, nice_step(span / (u16::MAX as f64))))
            })
            .collect();
        // Descending cost, then name — a total order, so the table is stable.
        ranked.sort_by(|a, b| {
            cost_of(&b.0)
                .cmp(&cost_of(&a.0))
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(SHADOW_PRICE_COLUMNS);
        ranked
    }
}

/// The `(flag, value, quality-cost sentence)` a lossy candidate maps to.
fn describe_lossy(candidate: &Candidate) -> (String, Option<String>, String) {
    match candidate {
        Candidate::QuantizeCoords(Some(m)) => (
            "--quantize-coords".to_string(),
            Some(format_number(*m)),
            format!("Coordinates snap to a world-anchored {m} m grid (max ground error {} m).", m / 2.0),
        ),
        Candidate::QuantizeAttrsAuto(true) => (
            "--quantize-attrs-auto".to_string(),
            None,
            "Every raw Float64 property becomes a range-adaptive UInt16: 65 535 levels across each \
             column's own range, so fine-grained values are rounded."
                .to_string(),
        ),
        Candidate::QuantizeAttr { column, step } => (
            "--quantize-attr".to_string(),
            Some(format!("{column}={}", format_number(*step))),
            format!("`{column}` is rounded to a {} step.", format_number(*step)),
        ),
        // Non-lossy candidates never enter the shadow-price ladder; if one ever
        // did, describing it as lossy would be the lie, so name it plainly.
        other => (
            format!("{other:?}"),
            None,
            "Unclassified candidate.".to_string(),
        ),
    }
}

/// Shortest round-tripping decimal for a ladder value (`5`, `0.01`).
fn format_number(x: f64) -> String {
    let s = format!("{x}");
    s
}

/// A duration in `parse_duration`'s syntax, using the coarsest unit that
/// divides it exactly (`345600000` → `4d`). A bare number is milliseconds,
/// which that parser accepts, so this never fails to round-trip.
fn duration_label(ms: u64) -> String {
    for (unit_ms, suffix) in [
        (86_400_000u64, "d"),
        (3_600_000, "h"),
        (60_000, "m"),
        (1_000, "s"),
    ] {
        if ms >= unit_ms && ms.is_multiple_of(unit_ms) {
            return format!("{}{suffix}", ms / unit_ms);
        }
    }
    ms.to_string()
}

/// Snap up to the next 1/2/5×10^k value, so a generated step reads like a
/// number a human would have chosen.
fn nice_step(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return 1.0;
    }
    let pow = 10f64.powf(x.log10().floor());
    let mantissa = x / pow;
    let nice = if mantissa <= 1.0 {
        1.0
    } else if mantissa <= 2.0 {
        2.0
    } else if mantissa <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * pow
}

/// Blob ordering and pack sizing, taken VERBATIM from the existing layout
/// advisor.
///
/// Neither enters Ŝ — ordering moves dedup and range-read cost, pack size moves
/// object count — so the budget has no reason to second-guess the simulator that
/// already decides them. Taking them verbatim is also what preserves the
/// recorded guarantee: a playback-caveated `spatial` pick arrives here with
/// `suggestion_only: true` and keeps it, **under budget pressure like any other
/// time**, because `blobOrdering: spatial` silently breaks time-playback
/// buffering and a smaller archive that stalls is not a smaller archive that
/// works.
///
/// `--publish` is deliberately dropped from the advisor's output here: the
/// budget solver ran its own measured zstd sweep and its verdict is the one the
/// projection was computed under.
///
/// # The two things this function must not do
///
/// It must not RE-DECIDE `suggestion_only` (that would promote the caveated
/// `spatial` pick), and it must not DROP `lossy` (that would let a lossy lever
/// wear a chosen lever's type). Both are the business of
/// [`ChosenLever::try_from`], which is why the mapping goes through it instead
/// of writing the struct literal here. The producer-side guard is
/// `inherited_layout_levers_propagate_the_advisors_own_gate`: forcing
/// `suggestion_only: false` anywhere on this path fails it.
fn inherited_layout_levers(result: &AnalysisResult, data: &LoadedData) -> Result<Vec<ChosenLever>> {
    let advice = layout_advisor::advise(result, data)?;
    advice
        .into_iter()
        .filter(|a| matches!(a.flag.as_str(), "--blob-ordering" | "--pack-size"))
        .map(ChosenLever::try_from)
        .collect()
}

/// Render a report as the text table `recommend --target-size` prints.
pub fn format_text(report: &BudgetReport) -> String {
    let mut out = String::new();
    out.push_str("\nBudget solver (--target-size):\n");
    out.push_str(&format!("  {}\n", report.headline()));
    out.push_str(&format!(
        "  Distortion class: zoom clamp {}, bucket ×{} (floor over the reversible levers: zoom \
         clamp {}, bucket ×{} at {} B)\n",
        report.distortion.zoom_clamp,
        report.distortion.bucket_multiplier,
        report.floor_distortion.zoom_clamp,
        report.floor_distortion.bucket_multiplier,
        report.floor_bytes,
    ));
    out.push_str(&format!(
        "  Search: {} distortion class(es) evaluated, zstd levels {:?}, basis {}\n",
        report.classes_evaluated,
        report.zstd_sweep,
        match report.basis {
            EstimateBasis::Measured => "measured",
            EstimateBasis::DensityEstimate => "density estimate (sample below the floor)",
        }
    ));

    out.push_str("\n  Chosen levers (all reversible — no feature is dropped):\n");
    if report.chosen.is_empty() {
        out.push_str("    (none — the unconstrained recommendation already fits)\n");
    }
    for lever in &report.chosen {
        let value = lever.value.as_deref().unwrap_or("—");
        let delta = match lever.delta_bytes {
            Some(d) => format!("{d:+} B"),
            None => "—".to_string(),
        };
        let gate = if lever.suggestion_only {
            "  [SUGGESTION - needs a decision]"
        } else {
            ""
        };
        out.push_str(&format!(
            "    {:<20} {:<14} {:>14}{gate}\n",
            lever.flag, value, delta
        ));
        out.push_str(&format!("        → {}\n", lever.why));
    }

    out.push_str("\n  Shadow prices — LOSSY levers, opt-in only, NEVER auto-applied:\n");
    if report.shadow_prices.is_empty() {
        out.push_str("    (none priced)\n");
    }
    for price in &report.shadow_prices {
        let value = price.value.as_deref().unwrap_or("—");
        out.push_str(&format!(
            "    {:<20} {:<14} {:>14}  [LOSSY]\n",
            price.flag,
            value,
            format!("{:+} B", -price.marginal_bytes)
        ));
        out.push_str(&format!("        → {}\n", price.why));
    }

    if !report.notes.is_empty() {
        out.push('\n');
        for note in &report.notes {
            out.push_str(&format!("  NOTE: {note}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis;
    use crate::loader::{AnalyzableFeature, GeometryType};
    use crate::recommend;
    use geo_types::{Geometry, Point as GeoPoint};
    use std::path::Path;
    use stt_core::types::{BoundingBox, TimeRange};

    /// Deterministic pseudo-noise in `[0, 1)` — high-entropy f64 mantissas with
    /// no RNG, so every fixture is byte-identical on every machine.
    fn noise(i: usize, salt: u64) -> f64 {
        ((i as u64).wrapping_add(salt).wrapping_mul(2_654_435_761) % 1_000_000) as f64 / 1e6
    }

    /// `n` jittered Montréal-area points, one per `step_ms`, each carrying a
    /// high-entropy float and a heavy-repeat categorical property.
    fn sample_of(n: usize, step_ms: u64) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| SampledFeature {
                geometry: Geometry::Point(GeoPoint::new(
                    -73.8 + (i as f64 * 0.003) % 0.6 + noise(i, 0) * 1e-3,
                    45.2 + (i % 7) as f64 * 0.08 + noise(i, 17) * 1e-3,
                )),
                timestamp_ms: 1_600_000_000_000 + i as u64 * step_ms,
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

    fn loaded(sample: Vec<SampledFeature>) -> LoadedData {
        let features: Vec<AnalyzableFeature> = sample
            .iter()
            .map(|f| {
                let p = match &f.geometry {
                    Geometry::Point(p) => *p,
                    _ => unreachable!("fixture is points"),
                };
                AnalyzableFeature {
                    lon: p.x(),
                    lat: p.y(),
                    timestamp: f.timestamp_ms,
                    geometry_type: GeometryType::Point,
                    vertex_count: 1,
                    estimated_size: 116,
                    property_count: 2,
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

    /// A real analysis over a synthetic dataset: the actual spatial / temporal /
    /// geometry / density analyzers, plus a measured sample encoding, so the
    /// solver is exercised on the numbers it will really see.
    fn analyzed(n: usize, step_ms: u64) -> (AnalysisResult, LoadedData) {
        analyzed_data(loaded(sample_of(n, step_ms)))
    }

    fn analyzed_data(data: LoadedData) -> (AnalysisResult, LoadedData) {
        let spatial = analysis::spatial::analyze(&data).unwrap();
        let temporal = analysis::temporal::analyze(&data).unwrap();
        let geometry = analysis::geometry::analyze(&data).unwrap();
        let occupancy = analysis::density::analyze(&data, &spatial, &temporal, None).unwrap();
        let layout = SyntheticLayout::from_density(&occupancy);
        let measured = crate::measure::measure_sample_layout(
            &data.sample,
            &MeasureSettings::default(),
            &layout,
        )
        .unwrap();
        let density =
            analysis::density::analyze(&data, &spatial, &temporal, measured.as_ref()).unwrap();
        let result = AnalysisResult {
            source: "budget-fixture".to_string(),
            feature_count: data.features.len(),
            bounds: data.bounds,
            spatial,
            temporal,
            geometry,
            density,
            measured,
        };
        (result, data)
    }

    fn run_layout(result: &AnalysisResult) -> SyntheticLayout {
        SyntheticLayout::from_density(&result.density)
    }

    /// Widen a fixture's FEATURE list (what the analyzers read) without widening
    /// its SAMPLE (what the encoder measures) — the loader's own shape, and what
    /// makes a 12k-feature temporal fixture cheap enough to measure.
    fn with_feature_count(mut data: LoadedData, n: u64) -> LoadedData {
        let t0 = data.time_range.start;
        let span = data.time_range.end.saturating_sub(t0).max(1);
        data.features = (0..n)
            .map(|i| AnalyzableFeature {
                lon: -73.8 + (i % 200) as f64 * 0.003,
                lat: 45.2 + (i % 7) as f64 * 0.08,
                timestamp: t0 + i * span / n,
                geometry_type: GeometryType::Point,
                vertex_count: 1,
                estimated_size: 116,
                property_count: 2,
            })
            .collect();
        data
    }

    /// A long-timeline dataset: three years of evenly-spaced events, 12k
    /// features behind a small encode sample — the shape of the repo's own
    /// osm-nyc-changesets (20 years, 380k features, 1068 weekly buckets), and
    /// the shape a coarse temporal pyramid exists for.
    ///
    /// ⚠️ This replaced a **500-year** fixture. That fixture was not a dataset
    /// class, it was a workaround: `LOD_BUCKET_THRESHOLD` was 5000 while
    /// `analysis::temporal::recommend_bucket_size` caps its own bucket-count
    /// target at 2025, so nothing shorter than four centuries could reach the
    /// tier lever at all and the test was passing on data no one will ever
    /// build (F7). With the threshold calibrated against the recommender's
    /// actual output, three years of ordinary events reaches it.
    fn analyzed_long_span(sample_n: usize) -> (AnalysisResult, LoadedData) {
        const THREE_YEARS_MS: u64 = 3 * 365 * 86_400_000;
        let sample = sample_of(sample_n, THREE_YEARS_MS / sample_n as u64);
        analyzed_data(with_feature_count(loaded(sample), 12_000))
    }

    /// A dataset the ORDERING advisor genuinely answers `spatial` on, with the
    /// playback caveat attached: two spatial cells 1° apart, each sampled across
    /// 24 hourly buckets.
    ///
    /// ⚠️ **This fixture is the whole point of
    /// `a_playback_caveated_ordering_stays_suggestion_only_under_budget_pressure`.**
    /// That test used to run on `analyzed(400, 60_000)`, a single-cluster
    /// minute-cadence dataset on which the advisor emits no caveated ordering at
    /// all — so both of its assertion loops iterated EMPTY collections and the
    /// test passed no matter what the production code did. Instrumented, the old
    /// fixture reported `suggestion_only levers=0, spatial picks=0` (F3).
    ///
    /// Why this shape produces the pick: 2 cells × 24 buckets = 48 synthesized
    /// native tiles, comfortably over `MIN_TILES_TO_SIMULATE`, so the range-read
    /// simulator (not the access-shape fallback) decides. With only two cells,
    /// each cell's whole timeline fuses into one contiguous run under `spatial`,
    /// which wins the blended cost even with the playback query priced in at
    /// weight 2 — so the advisor returns `spatial`, raises the multi-bucket
    /// playback caveat, and gates the advice `suggestion_only`. That is exactly
    /// the state the budget solver must inherit and must not promote.
    fn analyzed_two_cells_over_deep_time() -> (AnalysisResult, LoadedData) {
        const HOUR_MS: u64 = 3_600_000;
        const BUCKETS: u64 = 24;
        const CELLS: u64 = 2;
        const PER_CELL: usize = 10;

        let mut sample: Vec<SampledFeature> = Vec::new();
        let mut i = 0usize;
        for bucket in 0..BUCKETS {
            for cell in 0..CELLS {
                for _ in 0..PER_CELL {
                    sample.push(SampledFeature {
                        // 1° apart: distinct tiles at every zoom the recommender
                        // can reach here, so the cell count is a fixture
                        // property rather than a zoom accident.
                        geometry: Geometry::Point(GeoPoint::new(
                            -73.0 + cell as f64 + noise(i, 5) * 1e-3,
                            45.0 + noise(i, 23) * 1e-3,
                        )),
                        timestamp_ms: 1_600_000_000_000 + bucket * HOUR_MS,
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
                    });
                    i += 1;
                }
            }
        }
        analyzed_data(loaded(sample))
    }

    /// The `--blob-ordering` advice the layout advisor produces for a fixture,
    /// or a panic naming what the fixture failed to produce. Used as a FIXTURE
    /// GUARD: every assertion about the caveated pick is worthless if the pick
    /// is not there.
    fn caveated_ordering_advice(result: &AnalysisResult, data: &LoadedData) -> Advice {
        let advice = layout_advisor::advise(result, data).unwrap();
        let ordering = advice
            .into_iter()
            .find(|a| a.flag == "--blob-ordering")
            .expect("fixture guard: the ordering advisor must speak on this dataset");
        assert_eq!(
            ordering.value.as_deref(),
            Some("spatial"),
            "fixture guard: this fixture exists to produce a `spatial` pick, got {ordering:?}"
        );
        assert!(
            ordering.why.contains("playback"),
            "fixture guard: the pick must carry the playback caveat: {}",
            ordering.why
        );
        assert!(
            ordering.suggestion_only,
            "fixture guard: the ADVISOR's own gate must be set, or there is nothing \
             for the solver to propagate: {ordering:?}"
        );
        assert!(!ordering.lossy, "ordering is a reversible lever");
        ordering
    }

    /// The unconstrained estimate the solver starts from.
    fn unconstrained_bytes(result: &AnalysisResult) -> u64 {
        result.density.estimated_archive_size as u64
    }

    fn flags(report: &BudgetReport) -> Vec<&str> {
        report.chosen.iter().map(|c| c.flag.as_str()).collect()
    }

    // ------------------------------------------------------------------
    // 1. Generous budget: nothing is distorted
    // ------------------------------------------------------------------

    #[test]
    fn a_generous_budget_distorts_nothing_and_keeps_the_unconstrained_recipe() {
        let (result, data) = analyzed(400, 60_000);
        let layout = run_layout(&result);
        let budget = unconstrained_bytes(&result) * 100;
        let report = solve(&result, &data, &layout, budget).unwrap();

        assert!(report.feasible, "{:?}", report.notes);
        assert_eq!(
            report.distortion,
            DistortionClass::none(),
            "a generous budget must sit in the undistorted class: {report:?}"
        );
        assert!(
            !flags(&report).contains(&"--max-zoom"),
            "no clamp was needed: {:?}",
            flags(&report)
        );
        assert!(
            !flags(&report).contains(&"--temporal-bucket"),
            "no bucket coarsening was needed: {:?}",
            flags(&report)
        );
        // The WHOLE frontier is re-measured, so the floor it reports is a real
        // minimum rather than "the smallest point the walk happened to reach".
        assert_eq!(
            report.classes_evaluated,
            candidate_classes(&result).len(),
            "{report:?}"
        );
        assert!(report.projected_bytes <= report.target_bytes);
        assert!(report.floor_bytes <= report.projected_bytes);
    }

    // ------------------------------------------------------------------
    // 2. Tight budget: the lexicographic order decides
    // ------------------------------------------------------------------

    #[test]
    fn the_class_ladder_is_enumerated_in_lexicographic_distortion_order() {
        let (result, _) = analyzed(400, 60_000);
        let classes = candidate_classes(&result);
        assert!(!classes.is_empty());
        assert_eq!(classes[0], DistortionClass::none(), "{classes:?}");
        // Ascending in §5.2's order: D_zoom dominates D_time, so temporal
        // resolution is spent before spatial depth.
        for pair in classes.windows(2) {
            assert!(pair[0] < pair[1], "not ascending: {classes:?}");
            if pair[0].zoom_clamp == pair[1].zoom_clamp {
                assert!(pair[0].bucket_multiplier < pair[1].bucket_multiplier);
            } else {
                assert!(pair[0].zoom_clamp < pair[1].zoom_clamp);
            }
        }
        // A clamp may never push max_zoom below min_zoom.
        let depth = result.spatial.recommended_max_zoom - result.spatial.recommended_min_zoom;
        assert!(classes.iter().all(|c| c.zoom_clamp <= depth), "{classes:?}");
    }

    #[test]
    fn a_tight_budget_clamps_zoom_once_bucket_coarsening_cannot_reach_it() {
        let (result, data) = analyzed(400, 60_000);
        let layout = run_layout(&result);

        // Price every undistorted-zoom class first: whatever the coarsest
        // bucket at full depth achieves is the most bucket coarsening can buy.
        let mut sizer = Sizer::new(&result, &data, &layout).unwrap();
        let best_without_clamp = BUCKET_MULTIPLIERS
            .iter()
            .filter_map(|&m| {
                let class = DistortionClass {
                    zoom_clamp: 0,
                    bucket_multiplier: m,
                };
                candidate_classes(&result)
                    .contains(&class)
                    .then(|| sizer.best_point(class).unwrap().bytes)
            })
            .min()
            .unwrap();

        // A budget strictly under that cannot be met without touching zoom.
        let budget = best_without_clamp - best_without_clamp / 10;
        let report = solve(&result, &data, &layout, budget).unwrap();

        assert!(report.feasible, "{}", format_text(&report));
        assert!(
            report.distortion.zoom_clamp > 0,
            "a budget below the best full-depth point must clamp zoom: {}",
            format_text(&report)
        );
        // …and it spends NO distortion it did not have to. The winning class is
        // the FIRST feasible one in §5.2's lexicographic order, so every class
        // ahead of it in the ladder must genuinely miss the budget. Asserting
        // that directly is stronger than asserting a particular multiplier —
        // and it is now load-bearing, because the two-term oracle made bucket
        // width a REAL byte lever for the first time: a wider bucket merges time
        // slices, which removes whole tiles and with them their framing. Under
        // the old feature-count-only oracle the bucket could not move Ŝ at all
        // (`kept_bytes` never saw it), so a clamped-native-bucket class always
        // won by default.
        let classes = candidate_classes(&result);
        let winner_index = classes
            .iter()
            .position(|c| *c == report.distortion)
            .expect("the winner is on the ladder");
        for class in &classes[..winner_index] {
            let bytes = sizer.best_point(*class).unwrap().bytes;
            assert!(
                bytes > budget,
                "class {class:?} precedes the winner but projects {bytes} B ≤ {budget} B: {}",
                format_text(&report)
            );
        }
        let max_zoom_lever = report
            .chosen
            .iter()
            .find(|c| c.flag == "--max-zoom")
            .expect("the clamp must be an emitted lever");
        assert_eq!(
            max_zoom_lever.value.as_deref(),
            Some(
                (result.spatial.recommended_max_zoom - report.distortion.zoom_clamp)
                    .to_string()
                    .as_str()
            )
        );
        assert!(
            max_zoom_lever.delta_bytes.unwrap() < 0,
            "the clamp must be priced as a saving: {max_zoom_lever:?}"
        );
        assert!(report.projected_bytes <= budget, "{}", format_text(&report));
    }

    // ------------------------------------------------------------------
    // 2b. F6/F8: the clamp has to EXIST, and the budget has to CHANGE something
    // ------------------------------------------------------------------

    /// `n` jittered points spread over a CONUS-shaped footprint (~55° × 25°),
    /// i.e. the extent class of every continental dataset in the repo.
    fn wide_sample_of(n: usize, step_ms: u64) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| SampledFeature {
                geometry: Geometry::Point(GeoPoint::new(
                    -125.0 + (i % 71) as f64 * 0.78 + noise(i, 3) * 1e-2,
                    25.0 + (i % 37) as f64 * 0.68 + noise(i, 29) * 1e-2,
                )),
                timestamp_ms: 1_600_000_000_000 + i as u64 * step_ms,
                properties: vec![
                    (
                        "magnitude".to_string(),
                        PropValue::Number(1.0 + noise(i, 11) * 9.0),
                    ),
                    (
                        "region".to_string(),
                        PropValue::Text(format!("region-{}", i % 5)),
                    ),
                ],
            })
            .collect()
    }

    #[test]
    fn the_zoom_clamp_is_searchable_and_a_tight_budget_emits_a_different_recipe() {
        // THE F6/F8 REGRESSION, at the recipe boundary.
        //
        // Before the overview floor existed, the zoom recommender returned
        // min == max on every real dataset, so `candidate_classes` produced
        // clamp steps = 0 and the search enumerated temporal-bucket multipliers
        // only. The consequence was measured end to end: `--target-size 1KiB`
        // and a plain `--auto encode` built BYTE-IDENTICAL archives.
        let (result, data) = analyzed_data(loaded(wide_sample_of(400, 3_600_000)));
        let layout = run_layout(&result);

        // The ladder reaches its full depth…
        assert!(
            result.spatial.recommended_max_zoom > result.spatial.recommended_min_zoom,
            "fixture guard: a continental extent must span several zooms, got {}-{}",
            result.spatial.recommended_min_zoom,
            result.spatial.recommended_max_zoom
        );
        let classes = candidate_classes(&result);
        let clamps: Vec<u8> = {
            let mut v: Vec<u8> = classes.iter().map(|c| c.zoom_clamp).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        assert_eq!(
            clamps,
            (0..=MAX_ZOOM_CLAMP_STEPS).collect::<Vec<_>>(),
            "every clamp rung must be searchable here: {classes:?}"
        );

        // …the unconstrained answer clamps nothing…
        let roomy = solve(&result, &data, &layout, unconstrained_bytes(&result) * 100).unwrap();
        assert_eq!(roomy.distortion.zoom_clamp, 0, "{}", format_text(&roomy));

        // …and a budget under the best full-depth point actually spends the
        // clamp, producing a DIFFERENT command from the unconstrained one.
        let mut sizer = Sizer::new(&result, &data, &layout).unwrap();
        let best_full_depth = BUCKET_MULTIPLIERS
            .iter()
            .filter_map(|&m| {
                let class = DistortionClass {
                    zoom_clamp: 0,
                    bucket_multiplier: m,
                };
                classes
                    .contains(&class)
                    .then(|| sizer.best_point(class).unwrap().bytes)
            })
            .min()
            .unwrap();
        let tight = solve(&result, &data, &layout, best_full_depth / 2).unwrap();
        assert!(
            tight.distortion.zoom_clamp > 0,
            "a budget at half the full-depth floor must reach the clamp: {}",
            format_text(&tight)
        );

        let command_of = |report: &BudgetReport| {
            let rec = recommend::generate_recommendations_budgeted(
                &result,
                Vec::new(),
                None,
                Some(report.clone()),
            );
            recommend::to_command(&rec, Path::new("data.parquet"), "timestamp")
        };
        let roomy_cmd = command_of(&roomy);
        let tight_cmd = command_of(&tight);
        assert_ne!(
            roomy_cmd, tight_cmd,
            "a tight budget must emit a different recipe than an unconstrained run — \
             identical commands are how `--target-size` came to build byte-identical archives"
        );
        assert!(
            tight_cmd.contains(&format!(
                "--max-zoom {}",
                result.spatial.recommended_max_zoom - tight.distortion.zoom_clamp
            )),
            "{tight_cmd}"
        );
        // …and it got there without a single feature-shedding flag.
        for banned in [
            "--maximum-tile-features",
            "--maximum-tile-bytes",
            "--drop-densest-as-needed",
            "--quantize",
            "--summary-tier",
            "--min-zoom-field",
        ] {
            assert!(!tight_cmd.contains(banned), "{banned} in {tight_cmd}");
        }
    }

    #[test]
    fn a_single_zoom_recommendation_says_the_clamp_was_not_searchable() {
        // The pre-fix world, reconstructed: a recommendation collapsed onto one
        // zoom. The clamp genuinely cannot help there (there is no zoom level to
        // remove), but the report must SAY so rather than quietly reporting
        // three evaluated classes and an unchanged recipe.
        let (mut result, data) = analyzed(150, 60_000);
        let zmax = result.spatial.recommended_max_zoom;
        result.spatial.recommended_min_zoom = zmax;
        result.density.per_zoom.retain(|z| z.zoom == zmax);

        assert_eq!(clampable_depth(&result), 0);
        let classes = candidate_classes(&result);
        assert!(classes.iter().all(|c| c.zoom_clamp == 0), "{classes:?}");

        let layout = run_layout(&result);
        let report = solve(&result, &data, &layout, 1).unwrap();
        let notes = report.notes.join(" ");
        assert!(
            notes.contains("ZOOM CLAMP") && notes.contains("NOT searchable"),
            "an unsearchable clamp must be reported in words: {:?}",
            report.notes
        );
        assert!(format_text(&report).contains("NOT searchable"));
    }

    #[test]
    fn the_clamp_ladder_never_prices_a_zoom_the_oracle_cannot_see() {
        // The bound is the span the size oracle can price. A recommendation
        // whose density rows are missing must not advertise clamp steps that
        // `kept_bytes` would score as free.
        let (mut result, _) = analyzed(150, 60_000);
        result.density.per_zoom.truncate(2);
        assert!(clampable_depth(&result) <= 1);
        assert!(candidate_classes(&result).iter().all(|c| c.zoom_clamp <= 1));
    }

    // ------------------------------------------------------------------
    // 3. THE NO-THINNING GUARD — the item's conscience
    // ------------------------------------------------------------------

    #[test]
    fn infeasible_budget_reports_the_floor_and_drops_nothing() {
        let (result, data) = analyzed(400, 60_000);
        let layout = run_layout(&result);
        // One byte. Nothing over Θ₀ can reach it.
        let report = solve(&result, &data, &layout, 1).unwrap();

        assert!(!report.feasible, "{}", format_text(&report));
        // The floor is REPORTED, not chased.
        assert!(report.floor_bytes > 1, "{report:?}");
        assert_eq!(report.projected_bytes, report.floor_bytes);
        assert_eq!(report.distortion, report.floor_distortion);
        // …and the floor is a genuine minimum over the whole frontier, at the
        // LEAST distortion that reaches it. An unreachable budget must never
        // hand back a recipe that surrenders resolution for no byte gain.
        let mut sizer = Sizer::new(&result, &data, &layout).unwrap();
        let mut cheapest = u64::MAX;
        for class in candidate_classes(&result) {
            let bytes = sizer.best_point(class).unwrap().bytes;
            cheapest = cheapest.min(bytes);
            if class < report.floor_distortion {
                assert!(
                    bytes > report.floor_bytes,
                    "class {class:?} is at least as small as the reported floor \
                     {:?} and distorts less",
                    report.floor_distortion
                );
            }
        }
        assert_eq!(report.floor_bytes, cheapest, "{report:?}");

        // NOTHING WAS DROPPED. No thinning lever exists in the recipe, at all.
        for lever in &report.chosen {
            for banned in [
                "--maximum-tile-features",
                "--maximum-tile-bytes",
                "--drop-densest-as-needed",
                "--quantize-coords",
                "--quantize-attrs-auto",
                "--quantize-attr",
                "--min-zoom-field",
                "--summary-tier",
            ] {
                assert_ne!(
                    lever.flag, banned,
                    "an infeasible budget must never reach for {banned}: {report:?}"
                );
            }
        }
        // The note says so in words a human will read.
        let notes = report.notes.join(" ");
        assert!(notes.contains("NOTHING HAS BEEN DROPPED"), "{notes}");
        assert!(notes.contains("INFEASIBLE"), "{notes}");

        // The shadow-price table is what an infeasible budget offers instead.
        assert!(
            !report.shadow_prices.is_empty(),
            "an infeasible budget must show what the lossy levers would buy"
        );
        assert!(report.shadow_prices.iter().all(|p| p.lossy));

        // …and the emitted command still carries no lossy flag.
        let rec = recommend::generate_recommendations_budgeted(
            &result,
            Vec::new(),
            None,
            Some(report.clone()),
        );
        let command = recommend::to_command(&rec, Path::new("data.parquet"), "timestamp");
        for lossy in [
            "--quantize-coords",
            "--quantize-attrs-auto",
            "--quantize-attr",
            "--maximum-tile-features",
            "--maximum-tile-bytes",
        ] {
            assert!(
                !command.contains(lossy),
                "lossy flag leaked into the command under budget pressure: {command}"
            );
        }
    }

    #[test]
    fn no_chosen_lever_can_ever_be_lossy() {
        // Structural: `ChosenLever` has no `lossy` field, so the guarantee is a
        // type-level one. This pins the OTHER half — that nothing lossy is ever
        // constructed as one, on any budget from absurd to generous.
        let (result, data) = analyzed(150, 60_000);
        let layout = run_layout(&result);
        let unconstrained = unconstrained_bytes(&result);
        for budget in [1, unconstrained / 4, unconstrained, unconstrained * 100] {
            let report = solve(&result, &data, &layout, budget.max(1)).unwrap();
            for lever in &report.chosen {
                assert!(
                    !lever.flag.starts_with("--quantize"),
                    "budget {budget}: {lever:?}"
                );
                assert!(
                    !lever.flag.starts_with("--maximum-tile"),
                    "budget {budget}: {lever:?}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. Shadow prices
    // ------------------------------------------------------------------

    #[test]
    fn shadow_prices_are_lossy_sorted_and_priced_at_the_composed_point() {
        let (result, data) = analyzed(400, 60_000);
        let layout = run_layout(&result);
        let report = solve(&result, &data, &layout, unconstrained_bytes(&result) * 100).unwrap();

        assert!(!report.shadow_prices.is_empty(), "{report:?}");
        assert!(
            report.shadow_prices.iter().all(|p| p.lossy),
            "every shadow price is lossy by construction: {:?}",
            report.shadow_prices
        );
        // Descending marginal bytes, with a total tiebreak.
        for pair in report.shadow_prices.windows(2) {
            assert!(
                (pair[0].marginal_bytes, &pair[0].flag, &pair[0].value)
                    >= (pair[1].marginal_bytes, &pair[1].flag, &pair[1].value),
                "not sorted by marginal bytes desc: {:?}",
                report.shadow_prices
            );
        }
        // The coordinate ladder and the auto-attribute lever are both priced…
        let flags: Vec<&str> = report
            .shadow_prices
            .iter()
            .map(|p| p.flag.as_str())
            .collect();
        assert!(flags.contains(&"--quantize-coords"), "{flags:?}");
        assert!(flags.contains(&"--quantize-attrs-auto"), "{flags:?}");
        // …and each says loudly that it is not applied, and where it was
        // measured — the recorded blindness being fixed is that quantization
        // wins had only ever been measured at the build-default level 3.
        for price in &report.shadow_prices {
            assert!(price.why.contains("OPT-IN ONLY"), "{price:?}");
            assert!(
                price.why.contains("measured on the composed recipe"),
                "{price:?}"
            );
        }

        // Only rungs the ENCODER can actually express are offered: §5.2's
        // ladder bottoms out below the i32 world-grid floor, and offering a
        // human a flag that would fail the build is worse than offering none.
        let sizer = Sizer::new(&result, &data, &layout).unwrap();
        let candidates = sizer.lossy_candidates(&result);
        assert!(!candidates.is_empty());
        assert!(
            candidates.iter().all(|c| c.validate().is_ok()),
            "an unrepresentable rung reached the ladder: {candidates:?}"
        );
        assert!(
            candidates.iter().all(Candidate::lossy),
            "the shadow-price ladder is Θ₁ ONLY — a reversible lever in it would \
             be searched, not priced: {candidates:?}"
        );
        assert!(
            candidates
                .iter()
                .any(|c| matches!(c, Candidate::QuantizeCoords(Some(m)) if *m == 5.0)),
            "{candidates:?}"
        );
        assert!(
            !candidates
                .iter()
                .any(|c| matches!(c, Candidate::QuantizeCoords(Some(m)) if *m == 0.01)),
            "the sub-floor rung must be dropped before the oracle sees it: {candidates:?}"
        );
    }

    // ------------------------------------------------------------------
    // 5. §12.4: LOD tiers are only chosen when their bytes fit
    // ------------------------------------------------------------------

    #[test]
    fn lod_tiers_are_priced_and_only_chosen_when_the_added_bytes_fit() {
        let (result, data) = analyzed_long_span(600);
        let bucket = result.temporal.recommended_bucket_ms;
        assert!(
            bucket > 0 && result.temporal.duration_ms / bucket > LOD_BUCKET_THRESHOLD,
            "fixture guard: this dataset must clear the tier trigger ({} buckets)",
            result.temporal.duration_ms / bucket.max(1)
        );
        let layout = run_layout(&result);
        let unconstrained = unconstrained_bytes(&result);

        // Generous: the tier fits, so it is recommended AND priced.
        let roomy = solve(&result, &data, &layout, unconstrained * 100).unwrap();
        let tier = roomy
            .chosen
            .iter()
            .find(|c| c.flag == "--temporal-lod")
            .unwrap_or_else(|| {
                panic!("a roomy budget must afford a tier: {}", format_text(&roomy))
            });
        let added = tier.delta_bytes.expect("a tier must carry its byte cost");
        assert!(
            added > 0,
            "a tier is ADDITIVE and must be priced as such: {tier:?}"
        );
        assert!(
            tier.why.contains("ADDITIVE") && tier.why.contains("priced against the budget"),
            "{tier:?}"
        );
        // The tier's bytes are inside the projection, not free.
        assert!(roomy.projected_bytes >= added as u64);

        // Exactly-fits: a budget with no room past the base recipe must NOT buy
        // a tier — the §12.4 "recommended for free" defect, closed.
        let base_only = solve(
            &result,
            &data,
            &layout,
            roomy.projected_bytes - added as u64,
        )
        .unwrap();
        assert!(
            !base_only.chosen.iter().any(|c| c.flag == "--temporal-lod"),
            "a tier was recommended with no budget to pay for it: {}",
            format_text(&base_only)
        );
        assert!(
            base_only.notes.iter().any(|n| n.contains("temporal-LOD")),
            "declining a tier must be explained: {:?}",
            base_only.notes
        );
    }

    // ------------------------------------------------------------------
    // 6. Size-suffix parsing
    // ------------------------------------------------------------------

    #[test]
    fn size_suffixes_parse() {
        assert_eq!(parse_size("1048576").unwrap(), 1_048_576);
        assert_eq!(parse_size("  1048576  ").unwrap(), 1_048_576);
        assert_eq!(parse_size("1_048_576").unwrap(), 1_048_576);
        assert_eq!(parse_size("512").unwrap(), 512);
        assert_eq!(parse_size("512B").unwrap(), 512);

        // Bare K/M/G are binary…
        assert_eq!(parse_size("1K").unwrap(), 1024);
        assert_eq!(parse_size("1M").unwrap(), 1024 * 1024);
        assert_eq!(parse_size("1G").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size("1KiB").unwrap(), 1024);
        assert_eq!(parse_size("250MiB").unwrap(), 250 * 1024 * 1024);
        assert_eq!(parse_size("2GiB").unwrap(), 2 * 1024 * 1024 * 1024);
        // …and the `B` forms decimal.
        assert_eq!(parse_size("1KB").unwrap(), 1_000);
        assert_eq!(parse_size("1MB").unwrap(), 1_000_000);
        assert_eq!(parse_size("1GB").unwrap(), 1_000_000_000);
        // Case-insensitive, fractional mantissa.
        assert_eq!(parse_size("1mib").unwrap(), 1024 * 1024);
        assert_eq!(parse_size("1.5G").unwrap(), 1_610_612_736);
        assert_eq!(parse_size("0.5M").unwrap(), 512 * 1024);

        for bad in [
            "", "   ", "abc", "-1", "0", "0B", "1TB", "M", "1.2.3", "1 QB",
        ] {
            assert!(parse_size(bad).is_err(), "`{bad}` must not parse");
        }
    }

    // ------------------------------------------------------------------
    // 7. Property: monotonicity of the distortion class in B
    // ------------------------------------------------------------------

    #[test]
    fn a_smaller_budget_never_buys_a_gentler_distortion_class() {
        // The property over the choice rule itself, across a wide grid of cost
        // vectors and budgets — including non-monotone cost ladders, where a
        // naive argmin would break the property and first-feasible does not.
        let ladders: Vec<Vec<u64>> = vec![
            vec![100, 90, 80, 70, 60],
            vec![100, 100, 100, 50, 10],
            vec![100, 95, 96, 40, 41], // deliberately non-monotone
            vec![50],
            vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ];
        for costs in &ladders {
            let mut previous: Option<usize> = None;
            // Descending budgets: the chosen class index must never decrease.
            for b in (0..=120u64).rev() {
                let chosen = choose_class(costs, b);
                if let (Some(prev), Some(now)) = (previous, chosen) {
                    assert!(
                        now >= prev,
                        "costs {costs:?}: budget {b} chose class {now} after {prev}"
                    );
                }
                if chosen.is_some() {
                    previous = chosen;
                }
            }
        }

        // …and end-to-end on the real solver: B1 < B2 ⇒ class(B1) ≥ class(B2).
        let (result, data) = analyzed(150, 60_000);
        let layout = run_layout(&result);
        let unconstrained = unconstrained_bytes(&result);
        let mut previous: Option<DistortionClass> = None;
        for divisor in [1u64, 3, 8] {
            let budget = (unconstrained / divisor).max(1);
            let report = solve(&result, &data, &layout, budget).unwrap();
            if let Some(prev) = previous {
                assert!(
                    report.distortion >= prev,
                    "budget {budget} (÷{divisor}) chose {:?} after {prev:?}",
                    report.distortion
                );
            }
            previous = Some(report.distortion);
        }
    }

    #[test]
    fn the_solver_chooses_through_choose_class_and_not_a_copy_of_it() {
        // F10: the property above is a property of `choose_class`. It guards
        // production only while production CALLS it — `solve` used to
        // reimplement the predicate inline (`find(|(_, p)| p.bytes <=
        // target_bytes)`), so the monotonicity property held over a function no
        // shipping path executed. This pins the two together: for a grid of
        // budgets, the class `solve` reports is exactly the one `choose_class`
        // names over the same cost vector, and the infeasible case is exactly
        // `None`.
        let (result, data) = analyzed(150, 60_000);
        let layout = run_layout(&result);
        let classes = candidate_classes(&result);
        let mut sizer = Sizer::new(&result, &data, &layout).unwrap();
        let costs: Vec<u64> = classes
            .iter()
            .map(|c| sizer.best_point(*c).unwrap().bytes)
            .collect();

        let floor = *costs.iter().min().unwrap();
        let ceiling = *costs.iter().max().unwrap();
        // Budgets straddling the whole frontier — below the floor (infeasible),
        // exactly at the floor, at an interior rung, and clear of the ceiling.
        // A handful, not a sweep: each one is a real solve (every candidate
        // point re-encoded), and the exhaustive grid over the RULE is the
        // property test above.
        let interior = costs
            .iter()
            .copied()
            .filter(|c| *c > floor && *c < ceiling)
            .min()
            .unwrap_or(floor);
        let mut budgets = vec![
            1u64,
            floor.saturating_sub(1).max(1),
            floor,
            interior,
            ceiling.saturating_mul(2),
        ];
        budgets.sort_unstable();
        budgets.dedup();
        // Both arms must actually be exercised. A guard whose interesting branch
        // never runs is the F3 failure mode, and writing one while fixing it
        // would be a poor joke.
        let (mut feasible_seen, mut infeasible_seen) = (0usize, 0usize);
        for budget in budgets {
            let report = solve(&result, &data, &layout, budget).unwrap();
            match choose_class(&costs, budget) {
                Some(index) => {
                    feasible_seen += 1;
                    assert!(report.feasible, "budget {budget}: {}", format_text(&report));
                    assert_eq!(
                        report.distortion, classes[index],
                        "budget {budget}: the reported class must be the one `choose_class` \
                         picks, or the live predicate has drifted from the one the \
                         monotonicity property guards"
                    );
                }
                None => {
                    infeasible_seen += 1;
                    assert!(
                        !report.feasible,
                        "budget {budget}: `choose_class` found nothing that fits, so the \
                         solver must report INFEASIBLE: {}",
                        format_text(&report)
                    );
                    assert_eq!(report.distortion, report.floor_distortion);
                }
            }
        }
        assert!(
            feasible_seen > 0 && infeasible_seen > 0,
            "the budget grid must straddle feasibility: {feasible_seen} feasible, \
             {infeasible_seen} infeasible"
        );
    }

    // ------------------------------------------------------------------
    // 8. Determinism
    // ------------------------------------------------------------------

    #[test]
    fn two_solves_over_one_input_serialise_byte_identically() {
        let (result, data) = analyzed(150, 60_000);
        let layout = run_layout(&result);
        for budget in [1u64, u64::MAX / 2] {
            let first = solve(&result, &data, &layout, budget).unwrap();
            let second = solve(&result, &data, &layout, budget).unwrap();
            assert_eq!(
                serde_json::to_string(&first).unwrap(),
                serde_json::to_string(&second).unwrap(),
                "budget {budget}: the report must be reproducible byte for byte"
            );
        }
    }

    // ------------------------------------------------------------------
    // 9. GUARD (do-not-touch register): the zstd sweep caps at 19
    // ------------------------------------------------------------------

    #[test]
    fn the_zstd_sweep_never_evaluates_a_level_above_nineteen() {
        // The ladder itself…
        assert_eq!(MAX_ZSTD_LEVEL, 19);
        assert!(
            ZSTD_SWEEP
                .iter()
                .all(|&l| (1..=MAX_ZSTD_LEVEL).contains(&l)),
            "the sweep table must stay inside [1, {MAX_ZSTD_LEVEL}]: {ZSTD_SWEEP:?}"
        );
        assert_eq!(*ZSTD_SWEEP.iter().max().unwrap(), MAX_ZSTD_LEVEL);

        // …and what a real solve actually evaluated.
        let (result, data) = analyzed(400, 60_000);
        let layout = run_layout(&result);
        let report = solve(&result, &data, &layout, unconstrained_bytes(&result) * 100).unwrap();
        assert!(!report.zstd_sweep.is_empty(), "{report:?}");
        assert!(
            report.zstd_sweep.iter().all(|&l| l <= MAX_ZSTD_LEVEL),
            "a level above the cap reached the compressor: {:?}",
            report.zstd_sweep
        );

        // …and the cap is enforced at the measurement seam, not just by the
        // table: level 22 cannot reach the compressor even if handed in
        // directly. (zstd 22 is a standing rejection — 19 ≈ 22 measured.)
        let mut sizer = Sizer::new(&result, &data, &layout).unwrap();
        let err = sizer
            .bytes_per_feature(result.spatial.recommended_max_zoom, 3_600_000, 22)
            .expect_err("level 22 must be rejected before anything is compressed");
        assert!(format!("{err:#}").contains("22"), "{err:#}");
    }

    // ------------------------------------------------------------------
    // Extras: honesty about error, and the ordering caveat under pressure
    // ------------------------------------------------------------------

    #[test]
    fn the_projection_states_when_the_budget_gap_is_inside_its_own_noise() {
        let (result, data) = analyzed(400, 60_000);
        let layout = run_layout(&result);
        // Aim the budget exactly at the unconstrained projection: the gap is
        // then ~0, which is by definition inside any non-zero error bar.
        let baseline = solve(&result, &data, &layout, unconstrained_bytes(&result) * 100).unwrap();
        assert!(
            baseline.projected_stderr > 0.0,
            "this fixture must produce dispersion evidence: {baseline:?}"
        );
        let exact = solve(&result, &data, &layout, baseline.projected_bytes).unwrap();
        assert!(exact.feasible);
        assert!(
            exact.within_noise,
            "a zero gap must be reported as within noise: {}",
            format_text(&exact)
        );
        assert!(
            exact
                .notes
                .iter()
                .any(|n| n.contains("not distinguishable") || n.contains("about the right size")),
            "{:?}",
            exact.notes
        );
        assert!(exact.headline().contains("within"), "{}", exact.headline());
    }

    /// The PRODUCER side of the playback gate: `inherited_layout_levers` must
    /// carry the advisor's verdict across, field for field.
    ///
    /// ⚠️ This is the test the mutation `suggestion_only: false` has to fail.
    /// Nothing pinned this before: the end-to-end test below covered the pick
    /// only if one existed, and MO-9's sibling test hand-builds the flag and so
    /// covers only the CONSUMER. The recorded consequence of a regression here
    /// is not abstract — `blobOrdering: spatial` empties the player's buffered
    /// ranges and the demo silently stalls.
    #[test]
    fn inherited_layout_levers_propagate_the_advisors_own_gate() {
        let (result, data) = analyzed_two_cells_over_deep_time();
        let advised = caveated_ordering_advice(&result, &data);

        let levers = inherited_layout_levers(&result, &data).unwrap();
        let ordering: Vec<&ChosenLever> = levers
            .iter()
            .filter(|l| l.flag == "--blob-ordering")
            .collect();
        // NON-EMPTY FIRST. An assertion loop over an empty collection is not a
        // guard, it is decoration that reads as protection.
        assert_eq!(
            ordering.len(),
            1,
            "the ordering advice must arrive as exactly one lever: {levers:?}"
        );
        let ordering = ordering[0];

        // Every field is the advisor's, unedited — `suggestion_only` above all.
        assert_eq!(ordering.value, advised.value);
        assert_eq!(ordering.why, advised.why);
        assert_eq!(
            ordering.suggestion_only, advised.suggestion_only,
            "the advisor gated this lever `suggestion_only` and the solver must \
             PROPAGATE that verdict, never re-decide it: {ordering:?}"
        );
        assert!(
            ordering.suggestion_only,
            "a playback-caveated `spatial` ordering must reach the report gated: {ordering:?}"
        );
        // Ordering and pack size move dedup / object count, not Ŝ.
        assert_eq!(ordering.delta_bytes, None);

        // And no lever on this path may be lossy — enforced by the conversion,
        // asserted here at the seam that uses it.
        assert!(levers
            .iter()
            .all(|l| !l.flag.starts_with("--quantize") && !l.flag.starts_with("--maximum-tile")));
    }

    #[test]
    fn a_playback_caveated_ordering_stays_suggestion_only_under_budget_pressure() {
        // ⚠️ FIXTURE, not decoration: on the old `analyzed(400, 60_000)` fixture
        // the advisor emitted no caveated ordering, so both loops below iterated
        // nothing and the test could not fail (F3). `caveated_ordering_advice`
        // asserts the pick EXISTS before anything is asserted about it.
        let (result, data) = analyzed_two_cells_over_deep_time();
        let advised = caveated_ordering_advice(&result, &data);
        let layout = run_layout(&result);

        // Whatever the ordering advisor says, the budget inherits its gating
        // verbatim: a caveated `spatial` pick must not be promoted by pressure.
        // Both ends of the range — an absurd budget and an unconstrained one.
        for budget in [1u64, unconstrained_bytes(&result) * 100] {
            let report = solve(&result, &data, &layout, budget).unwrap();
            let caveated: Vec<&ChosenLever> = report
                .chosen
                .iter()
                .filter(|c| {
                    c.flag == "--blob-ordering"
                        && c.value.as_deref() == Some("spatial")
                        && c.why.contains("playback")
                })
                .collect();
            assert_eq!(
                caveated.len(),
                1,
                "budget {budget}: the caveated `spatial` pick must survive into the report, \
                 or this guard asserts over an empty set: {}",
                format_text(&report)
            );
            for lever in &caveated {
                assert!(
                    lever.suggestion_only,
                    "budget {budget} promoted a playback-caveated spatial ordering: {lever:?}"
                );
                assert_eq!(lever.why, advised.why, "the caveat itself must ride along");
            }

            // …and a suggestion-only lever never reaches the command. Again the
            // set is asserted non-empty first: with nothing gated, "nothing
            // gated leaked" is true and says nothing.
            let gated: Vec<&ChosenLever> =
                report.chosen.iter().filter(|c| c.suggestion_only).collect();
            assert!(
                !gated.is_empty(),
                "budget {budget}: no gated lever in the recipe, so the leak check below \
                 would be vacuous: {}",
                format_text(&report)
            );
            let rec = recommend::generate_recommendations_budgeted(
                &result,
                Vec::new(),
                None,
                Some(report.clone()),
            );
            let command = recommend::to_command(&rec, Path::new("d.parquet"), "timestamp");
            for lever in gated {
                assert!(
                    !command.contains(&lever.flag),
                    "suggestion-only {} leaked into the command: {command}",
                    lever.flag
                );
            }
            // The text report says which levers are still a human's call, so the
            // gate is visible to the person reading it and not only to serde.
            assert!(
                format_text(&report).contains("[SUGGESTION - needs a decision]"),
                "a gated lever must be marked in the rendered table: {}",
                format_text(&report)
            );
        }
    }

    #[test]
    fn a_lossy_advice_can_never_become_a_chosen_lever() {
        // F9: `ChosenLever` has no `lossy` field, which the docs call the
        // structural form of the no-thinning guarantee. But `Advice` HAS one, so
        // a field-by-field mapping silently discards it — the guarantee then
        // rests on "layout.rs happens to be lossy: false throughout", i.e. on a
        // convention, at the exact seam advertised as a type. The conversion
        // refuses instead, in every build profile (a `debug_assert!` would not
        // hold in release).
        let lossy = Advice {
            flag: "--quantize-coords".to_string(),
            value: Some("0.5".to_string()),
            why: "synthetic lossy advice".to_string(),
            projected: Some("-20% sample encode (measured)".to_string()),
            lossy: true,
            suggestion_only: false,
            confidence: crate::advisors::AdviceConfidence::High,
        };
        let err = ChosenLever::try_from(lossy)
            .expect_err("a lossy lever must not be representable as a chosen one");
        let text = format!("{err:#}");
        assert!(text.contains("--quantize-coords"), "{text}");
        assert!(text.contains("LOSSY"), "{text}");

        // …and the reversible twin converts, gate and all.
        let reversible = Advice {
            flag: "--blob-ordering".to_string(),
            value: Some("spatial".to_string()),
            why: "synthetic playback-caveated pick".to_string(),
            projected: None,
            lossy: false,
            suggestion_only: true,
            confidence: crate::advisors::AdviceConfidence::Medium,
        };
        let lever = ChosenLever::try_from(reversible).unwrap();
        assert!(lever.suggestion_only, "{lever:?}");
        assert_eq!(lever.value.as_deref(), Some("spatial"));
    }

    #[test]
    fn an_unmeasurable_sample_falls_back_to_the_density_estimate_and_says_so() {
        // Below `MIN_MEASURE_FEATURES`: nothing can be encoded, so the solver
        // must defer to the density model rather than fabricate a measurement.
        let (mut result, mut data) = analyzed(400, 60_000);
        data.sample.truncate(4);
        result.measured = None;
        let layout = run_layout(&result);
        let report = solve(&result, &data, &layout, unconstrained_bytes(&result) * 100).unwrap();

        assert_eq!(report.basis, EstimateBasis::DensityEstimate);
        // The default-level probe is the ONLY level that was tried: with no
        // measurable denominator there is no ratio to move, so sweeping would
        // burn encodes to produce identical numbers.
        assert_eq!(
            report.zstd_sweep,
            vec![MeasureSettings::default().zstd_level],
            "{report:?}"
        );
        assert_eq!(report.projected_stderr, 0.0);
        assert!(report.shadow_prices.is_empty());
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("below the framing probe's floor")),
            "{:?}",
            report.notes
        );
        // …and it says WHAT the fallback is missing, in the direction it is
        // missing. A report that quietly hands back a tile-blind estimate is how
        // a 7× miss got published behind a 1% error bar in the first place.
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("NO TILE TERM") && n.contains("under-projects")),
            "{:?}",
            report.notes
        );
    }

    #[test]
    fn the_report_renders_a_table_that_names_the_no_thinning_rule() {
        let (result, data) = analyzed(400, 60_000);
        let layout = run_layout(&result);
        let text = format_text(&solve(&result, &data, &layout, 1).unwrap());
        assert!(text.contains("Budget solver (--target-size)"), "{text}");
        assert!(text.contains("no feature is dropped"), "{text}");
        assert!(text.contains("LOSSY levers, opt-in only"), "{text}");
        assert!(text.contains("DOES NOT FIT"), "{text}");
    }

    #[test]
    fn nice_step_snaps_up_to_a_readable_value() {
        assert_eq!(nice_step(0.0009), 0.001);
        assert_eq!(nice_step(0.0011), 0.002);
        assert_eq!(nice_step(3.0), 5.0);
        assert_eq!(nice_step(6.0), 10.0);
        assert_eq!(nice_step(-1.0), 1.0);
    }
}
