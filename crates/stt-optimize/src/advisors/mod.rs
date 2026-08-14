//! Evidence-based flag advisors: each module inspects the analysis result
//! (and, where useful, trial-encodes the loader sample through the real
//! encoder) and returns [`Advice`] entries for stt-build flags beyond the
//! zoom/bucket basics.
//!
//! Contract: every advisor exposes
//! `pub fn advise(result: &AnalysisResult, data: &LoadedData) -> anyhow::Result<Vec<Advice>>`.
//! Advice with `lossy: true` is NEVER auto-applied by `stt-build --auto` and
//! never lands in `to_command` — it is surfaced loudly for the user to opt in
//! (the no-thinning principle).
//!
//! # One pass, or several: [`run_all`] vs [`run_iterative`]
//!
//! [`run_all`] runs each advisor ONCE, and every advisor trials its lever
//! against the SAME build-default baseline. That silently assumes the levers
//! are additive: the zstd-19 win is priced on unquantized data, the
//! quantization win is priced at zstd 3, and the recipe those two answers
//! compose into is never measured at all.
//!
//! They are not additive. Quantized coordinates are far more compressible than
//! `Float64` ones, so the same `--quantize-coords` lever is worth a different
//! fraction at level 19 than at level 3 — in both directions, depending on how
//! compressible the rest of the tile is.
//!
//! [`run_iterative`] fixes exactly that, and nothing else:
//!
//! ```text
//! round 0   run_all                        ← today's pass, byte-for-byte
//! round 1   θ = compose(non-lossy advice)  ← the recipe, assembled
//!           measure θ                      ← the recipe, MEASURED
//!           re-trial each lever vs θ        ← one-lever moves off the recipe
//! round 2   … repeat; θ converges, the cache answers, nothing is re-encoded
//! ```
//!
//! ⚠️ **Iteration refines NUMBERS, never ADMISSIBILITY.** A lever's `lossy` and
//! `suggestion_only` classification is a property of the lever, not of its
//! measurement, and no round may change one. In particular the ordering
//! advisor's range-read simulation is re-run *unchanged* and its playback
//! caveat rides along verbatim: a caveated `spatial` pick must never be
//! promoted into a committed one, because `blobOrdering: spatial` silently
//! breaks time-playback buffering.

pub mod budget;
pub mod layout;
pub mod quantize;
pub mod temporal;

use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::analysis::AnalysisResult;
use crate::loader::{LoadedData, SampledFeature};
use crate::measure::{measure_sample_layout, MeasureSettings, MeasuredEncoding, SyntheticLayout};
use crate::oracle::{run_trials, Candidate, TrialResult, MAX_ZSTD_LEVEL};

/// Advisor confidence in a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AdviceConfidence {
    High,
    Medium,
    Low,
}

impl std::fmt::Display for AdviceConfidence {
    /// Lowercase, matching the serde wire form (`high`/`medium`/`low`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            AdviceConfidence::High => "high",
            AdviceConfidence::Medium => "medium",
            AdviceConfidence::Low => "low",
        })
    }
}

/// One evidence-based flag recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Advice {
    /// The stt-build flag, e.g. `--quantize-coords` (name must exist in
    /// docs/api/cli-reference.md).
    pub flag: String,
    /// Flag value, `None` for bare switches.
    pub value: Option<String>,
    /// One-line, evidence-based rationale (numbers from this dataset, not
    /// folklore).
    pub why: String,
    /// Measured or estimated effect, e.g. `"-36% sample encode (measured)"`.
    pub projected: Option<String>,
    /// Lossy/destructive levers are never auto-applied and never join
    /// `to_command`.
    pub lossy: bool,
    /// Non-lossy advice that still must NOT be auto-applied (excluded from
    /// `to_command` and the MCP auto-args): the lever needs a human decision
    /// first — a quality tradeoff spelled out in `why` (e.g. spatial ordering
    /// vs. time-playback buffering) or a prerequisite data transform (e.g.
    /// `--min-zoom-field` needs a numeric rank column baked in).
    #[serde(default)]
    pub suggestion_only: bool,
    /// How sure the advisor is.
    pub confidence: AdviceConfidence,
}

/// Run every advisor. Order: quantize, temporal, layout, budget.
pub fn run_all(result: &AnalysisResult, data: &LoadedData) -> anyhow::Result<Vec<Advice>> {
    let mut advice = Vec::new();
    advice.extend(quantize::advise(result, data)?);
    advice.extend(temporal::advise(result, data)?);
    advice.extend(layout::advise(result, data)?);
    advice.extend(budget::advise(result, data)?);
    Ok(advice)
}

// ----------------------------------------------------------------------------
// Composed-recipe re-measurement (coordinate descent over the measured levers)
// ----------------------------------------------------------------------------

/// Coordinate-descent rounds [`run_iterative`] runs on top of round 0.
///
/// Two, fixed, and a constant on purpose. The round count is an input to a
/// deterministic solver: a convergence test ("stop when nothing moved") would
/// make the number of oracle calls — and therefore the reported cost — depend
/// on measurement noise, and a data-dependent stopping rule is exactly the kind
/// of thing that makes two runs of the same analysis disagree. Two rounds is
/// enough for the levers this crate measures: round 1 re-prices every lever on
/// the composed recipe, round 2 confirms the recipe did not move (and costs
/// nothing when it did not, because every measurement it asks for is already
/// cached).
pub const RECOMMEND_ROUNDS: usize = 2;

/// Multiples of a trial's standard error a measured shrink must exceed before
/// this layer treats it as evidence rather than noise.
///
/// Two jobs, one constant, deliberately: it demotes a noisy NON-LOSSY advisory
/// to `Low` confidence here ([`confidence_with_noise`]), and it is the
/// significance half of the LOSSY accept gate over in
/// [`quantize::ACCEPT_SIGMA`]. Sharing it is what stops a shrink being
/// admissible under one rule and noise under another.
///
/// ⚠️ It is only ever HALF of a lossy gate. Significance is a claim about the
/// error bar, and error bars shrink with sample size, so on its own it will
/// eventually admit an arbitrarily small permanent quality loss. The other half
/// is the magnitude floor (`quantize::MIN_COORD_SHRINK` /
/// `quantize::MIN_ATTR_SHRINK`); `quantize::admits` requires both and the
/// module doc there records what each one catches.
const SIGNIFICANCE_SIGMA: f64 = 2.0;

/// A canonical, totally-ordered key for one [`MeasureSettings`] point.
///
/// `MeasureSettings` carries `f64`s and so cannot be `Ord`; the bit patterns
/// can. Every field is compared exactly — two settings that differ in the last
/// mantissa bit of a quantization step are different measurements, and must not
/// share a cache entry. The `quantize_attrs` map is already a `BTreeMap`, so
/// its iteration order is sorted and the key is stable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SettingsKey {
    zstd_level: i32,
    quantize_coords_bits: Option<u64>,
    quantize_attrs_auto: bool,
    compact_times: bool,
    quantize_attrs: Vec<(String, u64)>,
}

impl SettingsKey {
    fn of(settings: &MeasureSettings) -> Self {
        Self {
            zstd_level: settings.zstd_level,
            quantize_coords_bits: settings.quantize_coords_m.map(f64::to_bits),
            quantize_attrs_auto: settings.quantize_attrs_auto,
            compact_times: settings.compact_times,
            quantize_attrs: settings
                .quantize_attrs
                .iter()
                .map(|(name, step)| (name.clone(), step.to_bits()))
                .collect(),
        }
    }
}

/// How much measurement an iteration actually bought.
///
/// Published because the honest cost of this mechanism is its headline risk:
/// iteration multiplies encode work, and a user staring at a slow `recommend`
/// deserves to see why. `measurements` and `trials_priced` count CACHE MISSES —
/// work that really happened; `cache_hits` counts the requests the cache
/// answered for free, which is the whole reason the total stays bounded.
///
/// One priced candidate is not one encode: [`run_trials`] also measures
/// replicate blocks under both arms so its `stderr` means something, so a
/// priced candidate costs roughly `1 + REPLICATE_BLOCKS` sample encodes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OracleUsage {
    /// Distinct [`MeasureSettings`] points measured outright (cache misses).
    pub measurements: usize,
    /// [`run_trials`] invocations.
    pub oracle_calls: usize,
    /// Candidates actually priced (cache misses).
    pub trials_priced: usize,
    /// Measurement/trial requests answered from the cache.
    pub cache_hits: usize,
}

/// The measurement surface a set of iteration rounds shares: ONE sample, ONE
/// layout, and a cache keyed on the settings point.
///
/// ⚠️ The layout is fixed for the composer's whole lifetime, which is what makes
/// its numbers comparable (MO-4's invariant: a trial measured under a different
/// tile cut prices the cut, not the lever). The cache key is therefore the
/// settings alone — adding a second layout to a live composer would silently
/// alias two different measurements onto one key, so there is no API for it.
pub struct Composer<'a> {
    sample: &'a [SampledFeature],
    layout: &'a SyntheticLayout,
    measured: BTreeMap<SettingsKey, Option<MeasuredEncoding>>,
    priced: BTreeMap<(SettingsKey, String), TrialResult>,
    usage: OracleUsage,
}

impl<'a> Composer<'a> {
    /// A composer over `sample`, measuring everything under `layout`.
    pub fn new(sample: &'a [SampledFeature], layout: &'a SyntheticLayout) -> Self {
        Self {
            sample,
            layout,
            measured: BTreeMap::new(),
            priced: BTreeMap::new(),
            usage: OracleUsage::default(),
        }
    }

    /// Measure one settings point outright, cached.
    ///
    /// `Ok(None)` means the sample is below the measurement floor — the same
    /// answer [`crate::measure::measure_sample`] gives, and the signal every
    /// caller here uses to fall back to its unmeasured behaviour.
    pub fn measure(&mut self, settings: &MeasureSettings) -> Result<Option<MeasuredEncoding>> {
        let key = SettingsKey::of(settings);
        if let Some(hit) = self.measured.get(&key) {
            self.usage.cache_hits += 1;
            return Ok(hit.clone());
        }
        let measured = measure_sample_layout(self.sample, settings, self.layout)?;
        self.usage.measurements += 1;
        self.measured.insert(key, measured.clone());
        Ok(measured)
    }

    /// Price `candidates` against `incumbent` through the trial oracle, cached
    /// per `(incumbent, candidate)` pair. Results come back in request order.
    ///
    /// An EMPTY vector means the oracle could not measure the sample at all
    /// (the [`run_trials`] fallback contract); callers must treat it as "no
    /// evidence this round" and leave the round-0 advice alone.
    pub fn price(
        &mut self,
        incumbent: &MeasureSettings,
        candidates: &[Candidate],
    ) -> Result<Vec<TrialResult>> {
        let key = SettingsKey::of(incumbent);
        let mut missing: Vec<Candidate> = Vec::new();
        for candidate in candidates {
            if self
                .priced
                .contains_key(&(key.clone(), candidate_key(candidate)))
            {
                self.usage.cache_hits += 1;
            } else if !missing.contains(candidate) {
                missing.push(candidate.clone());
            }
        }
        if !missing.is_empty() {
            let results = run_trials(self.sample, self.layout, incumbent, &missing)?;
            self.usage.oracle_calls += 1;
            if results.is_empty() {
                // Unmeasurable sample: nothing was encoded, so nothing is
                // cached (a later, larger sample would be a different composer).
                return Ok(Vec::new());
            }
            self.usage.trials_priced += results.len();
            for result in results {
                self.priced
                    .insert((key.clone(), candidate_key(&result.candidate)), result);
            }
        }
        let mut out = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            match self.priced.get(&(key.clone(), candidate_key(candidate))) {
                Some(result) => out.push(result.clone()),
                // Only reachable if the oracle dropped a candidate it was
                // handed, which its input-order contract forbids.
                None => return Ok(Vec::new()),
            }
        }
        Ok(out)
    }

    /// What this composer has spent so far.
    pub fn usage(&self) -> OracleUsage {
        self.usage
    }
}

/// Cache key for one candidate.
///
/// `Candidate` is `PartialEq` but neither `Ord` nor `Hash`, and it carries
/// `f64`s. Its serde form is a total, deterministic rendering of the whole
/// variant — including the float bits, which `serde_json` writes with the
/// shortest round-tripping representation — so two candidates share a key
/// exactly when they are the same candidate.
fn candidate_key(candidate: &Candidate) -> String {
    serde_json::to_string(candidate).unwrap_or_else(|_| format!("{candidate:?}"))
}

/// The outcome of re-pricing one lever against the composed incumbent.
pub(crate) enum Repriced {
    /// The lever was measured on the composed recipe. `Some` is the refreshed
    /// advice, `None` means it no longer clears its own emit threshold there.
    Measured(Option<Advice>),
    /// No measurement was possible this round (no candidate applies, the sample
    /// is unmeasurable, or the candidate falls outside the encoder's
    /// addressable range). The round-0 entry stands unchanged — iteration never
    /// deletes advice it did not price.
    Unmeasured,
}

/// A composed measurement of the recipe the advisors settled on.
///
/// Every byte figure is measured on the run's sample under the run's layout, so
/// `composed_bytes` and `default_bytes` are directly comparable — they differ
/// only in the settings.
#[derive(Debug, Clone)]
pub struct ComposedMeasurement {
    /// Coordinate-descent rounds run on top of round 0.
    pub rounds: usize,
    /// The composed NON-LOSSY recipe θ, exactly as measured.
    pub settings: MeasureSettings,
    /// Sampled features the measurement covered.
    pub features: usize,
    /// Compressed sample bytes at θ.
    pub composed_bytes: usize,
    /// Compressed sample bytes at the build defaults, same sample, same layout.
    pub default_bytes: usize,
    /// Compressed sample bytes at θ PLUS every lossy advisory that survived —
    /// a report-only figure. `None` when no lossy advice survived (then it
    /// would just restate `composed_bytes`).
    ///
    /// ⚠️ This number is not a recommendation. Lossy levers stay opt-in; this
    /// only answers "what would opting in be worth", measured rather than
    /// guessed.
    pub with_lossy_bytes: Option<usize>,
    /// What the iteration cost.
    pub usage: OracleUsage,
}

impl ComposedMeasurement {
    /// Fractional shrink of the composed recipe versus build defaults
    /// (positive = smaller).
    pub fn shrink(&self) -> f64 {
        1.0 - self.composed_bytes as f64 / self.default_bytes.max(1) as f64
    }

    /// The `composed_projected` line: what the recipe measures, against what,
    /// and what measuring it cost.
    pub fn projected(&self) -> String {
        format!(
            "recipe measures {} B on the {}-feature sample vs {} B at build defaults \
             ({:+.1}%, measured over {} refinement round{}; {} sample encodes + {} priced trials, \
             {} cache hits)",
            self.composed_bytes,
            self.features,
            self.default_bytes,
            -self.shrink() * 100.0,
            self.rounds,
            if self.rounds == 1 { "" } else { "s" },
            self.usage.measurements,
            self.usage.trials_priced,
            self.usage.cache_hits,
        )
    }

    /// The parallel with-lossy figure, for the report only.
    pub fn projected_with_lossy(&self) -> Option<String> {
        let bytes = self.with_lossy_bytes?;
        Some(format!(
            "adding the LOSSY advisories on top measures {} B ({:+.1}% vs defaults) — \
             opt-in only, never auto-applied",
            bytes,
            (bytes as f64 / self.default_bytes.max(1) as f64 - 1.0) * 100.0,
        ))
    }
}

/// Advice plus the measurement of the recipe it composes into.
///
/// `composed` is `None` exactly when there was nothing to compose: `rounds = 0`
/// (the documented rollback) or a sample below the measurement floor.
#[derive(Debug, Clone)]
pub struct IterativeAdvice {
    /// The advice, in advisor order, with measured numbers refreshed.
    pub advice: Vec<Advice>,
    /// The composed recipe as measured, when it was measured.
    pub composed: Option<ComposedMeasurement>,
}

/// Run the advisors, then re-measure the recipe they compose into and re-price
/// each measured lever against it, `rounds` times.
///
/// `layout` must be the run's own [`SyntheticLayout`] — the one `result.measured`
/// was taken under, reconstructible as
/// `SyntheticLayout::from_density(&result.density)`. Every measurement taken
/// here shares it.
///
/// # Rollback
///
/// `rounds = 0` returns exactly what [`run_all`] returns, field for field, with
/// no measurement performed. That is the documented rollback and it is pinned by
/// a test, not by a comment.
///
/// # Cost
///
/// Round 0 costs what it costs today. Each later round prices at most three
/// levers and measures at most two settings points, and every one of those is
/// cached on `(incumbent, candidate)` — so once the recipe stops moving, the
/// remaining rounds are free.
///
/// It is not cheap, and the number is worth stating rather than hedging.
/// Measured on the 600-point GeoParquet fixture in
/// `tests/recommend_determinism.rs` (debug build): **204 ms single-pass vs
/// 2.42 s iterated — 11.8×**, from ~3 sample encodes to ~28. The multiplier is
/// worst on small inputs like that one, where the fixed analysis costs (load,
/// spatial/temporal/density scans) are trivial and the encode work is
/// everything; on a large input those fixed costs dominate and the ratio falls.
/// The bound is structural rather than empirical: the sample is capped at 5000
/// features, the round count is a constant, and the candidate set per round is
/// fixed, so the extra work does NOT grow with the dataset.
///
/// [`ComposedMeasurement::usage`] reports exactly what was spent, and it rides
/// the `composed_projected` line into the `recommend` output — a slow analysis
/// should say what it bought. `rounds = 0` buys nothing and costs nothing.
pub fn run_iterative(
    result: &AnalysisResult,
    data: &LoadedData,
    layout: &SyntheticLayout,
    rounds: usize,
) -> Result<IterativeAdvice> {
    let advice = run_all(result, data)?;
    if rounds == 0 {
        return Ok(IterativeAdvice {
            advice,
            composed: None,
        });
    }

    let mut composer = Composer::new(&data.sample, layout);
    let defaults = MeasureSettings::default();
    let Some(baseline) = composer.measure(&defaults)? else {
        // Below the measurement floor: the advisors already fell back to their
        // unmeasured behaviour, and there is no honest composed figure to add.
        return Ok(IterativeAdvice {
            advice,
            composed: None,
        });
    };

    let mut advice = advice;
    let mut theta = compose_settings(&advice, Lossy::Exclude);
    for _round in 1..=rounds {
        advice = refine(result, data, &advice, &theta, &mut composer)?;
        theta = compose_settings(&advice, Lossy::Exclude);
    }

    let composed_bytes = match composer.measure(&theta)? {
        Some(m) => m.bytes_total,
        None => baseline.bytes_total,
    };
    let with_lossy = compose_settings(&advice, Lossy::Include);
    let with_lossy_bytes = if with_lossy == theta {
        None
    } else {
        composer.measure(&with_lossy)?.map(|m| m.bytes_total)
    };

    let composed = ComposedMeasurement {
        rounds,
        settings: theta,
        features: baseline.features,
        composed_bytes,
        default_bytes: baseline.bytes_total,
        with_lossy_bytes,
        usage: composer.usage(),
    };
    Ok(IterativeAdvice {
        advice,
        composed: Some(composed),
    })
}

/// Whether [`compose_settings`] folds in the lossy advisories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lossy {
    /// The recipe as it would actually be run: reversible levers only. This is
    /// the point every trial is measured against.
    Exclude,
    /// θ plus the lossy advisories — measured for the report, never recommended.
    Include,
}

/// Assemble the composed point θ from an advice list.
///
/// Only flags that map onto an [`EncoderConfig`](stt_core::arrow_tile::EncoderConfig)
/// knob can move a measurement; the rest of the recipe (zoom range, temporal
/// bucket, blob ordering, pack size, LOD tiers, budgets) is carried by the
/// analysis and by the layout the composer measures under, not by these
/// settings. `suggestion_only` advice is excluded in both modes — it is not part
/// of any recipe until a human says so.
fn compose_settings(advice: &[Advice], lossy: Lossy) -> MeasureSettings {
    let mut settings = MeasureSettings::default();
    for entry in advice {
        if entry.suggestion_only || (entry.lossy && lossy == Lossy::Exclude) {
            continue;
        }
        match entry.flag.as_str() {
            // `--publish` bundles zstd 19 over the already-paged directory.
            "--publish" => settings.zstd_level = layout::PUBLISH_ZSTD_LEVEL,
            "--zstd-level" => {
                if let Some(level) = entry.value.as_deref().and_then(|v| v.parse::<i32>().ok()) {
                    settings.zstd_level = level.clamp(1, MAX_ZSTD_LEVEL);
                }
            }
            "--quantize-coords" => {
                if let Some(step) = entry.value.as_deref().and_then(|v| v.parse::<f64>().ok()) {
                    settings.quantize_coords_m = Some(step);
                }
            }
            "--quantize-attrs-auto" => settings.quantize_attrs_auto = true,
            "--quantize-attr" => {
                if let Some((name, step)) = entry
                    .value
                    .as_deref()
                    .and_then(|v| v.split_once('='))
                    .and_then(|(n, s)| s.parse::<f64>().ok().map(|s| (n.to_string(), s)))
                {
                    settings.quantize_attrs.insert(name, step);
                }
            }
            _ => {}
        }
    }
    settings
}

/// One coordinate-descent round: re-price every measured lever against θ and
/// splice the verdicts back into the advice list.
///
/// Order is rebuilt, not patched: the quantize levers lead (they are the first
/// advisor), then the round-0 list minus the entries this round re-priced. That
/// keeps the emitted order exactly the advisor order in every case, including
/// the one that matters — a lever that was absent at round 0 and flips IN here.
fn refine(
    result: &AnalysisResult,
    data: &LoadedData,
    advice: &[Advice],
    theta: &MeasureSettings,
    composer: &mut Composer<'_>,
) -> Result<Vec<Advice>> {
    let (coords, attrs) = quantize::reprice(result, data, theta, composer)?;
    let publish = layout::reprice_publish(theta, composer)?;

    let mut out: Vec<Advice> = Vec::with_capacity(advice.len() + 2);
    // Leading quantize advisories, in advisor order (coords then attrs), whether
    // or not round 0 emitted them — this is where a lever that only pays on the
    // composed recipe flips IN.
    if let Repriced::Measured(Some(fresh)) = &coords {
        out.push(fresh.clone());
    }
    if let Repriced::Measured(Some(fresh)) = &attrs {
        out.push(fresh.clone());
    }

    for entry in advice {
        // `handled` = this round has already spoken for the entry, either by
        // emitting a refreshed version of it or by measuring that it no longer
        // clears its threshold.
        let handled = match entry.flag.as_str() {
            "--quantize-coords" => matches!(coords, Repriced::Measured(_)),
            "--quantize-attrs-auto" => matches!(attrs, Repriced::Measured(_)),
            "--publish" => match &publish {
                Repriced::Measured(Some(fresh)) => {
                    out.push(fresh.clone());
                    true
                }
                _ => false,
            },
            _ => false,
        };
        if !handled {
            // Everything iteration did not price rides through untouched —
            // including the ordering advisory with its playback caveat and its
            // `suggestion_only` flag.
            out.push(entry.clone());
        }
    }
    Ok(out)
}

/// Confidence for a measured shrink that also knows its own noise.
///
/// A shrink that does not clear [`SIGNIFICANCE_SIGMA`] standard errors is
/// reported at `Low` however large it looks: with a handful of replicate blocks
/// a big-looking delta can be one lucky tile. `stderr == 0.0` means NO
/// dispersion evidence (fewer than two usable replicates), never "no noise", so
/// it never promotes anything.
fn confidence_with_noise(base: AdviceConfidence, shrink: f64, stderr: f64) -> AdviceConfidence {
    if stderr > 0.0 && shrink.abs() <= SIGNIFICANCE_SIGMA * stderr {
        AdviceConfidence::Low
    } else {
        base
    }
}

/// The "…, measured on the composed recipe" clause every re-priced advisory
/// carries, so a reader can tell a round-0 number from an iterated one.
fn composed_note(theta: &MeasureSettings, stderr: f64) -> String {
    format!(
        "measured on the composed recipe (zstd {}{}){}",
        theta.zstd_level,
        composed_levers(theta),
        stderr_suffix(stderr)
    )
}

/// The composed levers OTHER than the zstd level, as a trailing clause.
///
/// Empty for the usual case (no lossy lever is ever part of θ); non-empty only
/// when a caller is describing a with-lossy point.
fn composed_levers(settings: &MeasureSettings) -> String {
    let mut clause = String::new();
    if settings.quantize_coords_m.is_some() {
        clause.push_str(", quantized coords");
    }
    if settings.quantize_attrs_auto {
        clause.push_str(", quantized attrs");
    }
    clause
}

/// `" ±x.xpp"` when there is dispersion evidence, empty when there is none.
///
/// The distinction matters: an omitted figure means "fewer than two usable
/// replicates", which is NOT the same claim as `±0.0pp`.
fn stderr_suffix(stderr: f64) -> String {
    if stderr > 0.0 {
        format!(" ±{:.1}pp", stderr * 100.0)
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis;
    use crate::loader::{AnalyzableFeature, GeometryType, PropValue};
    use geo_types::{Geometry, Point};
    use stt_core::types::{BoundingBox, TimeRange};

    const HOUR_MS: u64 = 3_600_000;

    /// splitmix64 — full-entropy token choices with no RNG dependency, so the
    /// fixture is byte-identical on every run and every machine.
    fn mix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `n` points on an EXACT `cols`-wide lon/lat grid (no jitter), each
    /// carrying `notes` text properties drawn unpredictably from a small token
    /// set.
    ///
    /// The exactness is the point. Grid coordinates repeat, so their `Float64`
    /// bytes are already compressible and `--quantize-coords` wins only a little
    /// at the build-default zstd 3 — but a lot at zstd 19, which exploits the
    /// fixed-point form's redundancy far better. That is the §12.2
    /// non-additivity this module exists to catch, reproduced in a fixture.
    fn grid_sample(n: usize, cols: usize, notes: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| {
                let properties = (0..notes)
                    .map(|p| {
                        let token = mix(i as u64 * 7 + p as u64) % 8;
                        (
                            format!("note{p}"),
                            PropValue::Text(format!(
                                "observation-station-{token:04}-quality-{}",
                                token % 7
                            )),
                        )
                    })
                    .collect();
                SampledFeature {
                    geometry: Geometry::Point(Point::new(
                        -73.5 + (i % cols) as f64 * 0.01,
                        45.5 + (i / cols) as f64 * 0.01,
                    )),
                    timestamp_ms: 1_600_000_000_000 + i as u64 * 1_000,
                    properties,
                }
            })
            .collect()
    }

    fn analyzable(lon: f64, lat: f64, ts: u64) -> AnalyzableFeature {
        AnalyzableFeature {
            lon,
            lat,
            timestamp: ts,
            geometry_type: GeometryType::Point,
            vertex_count: 1,
            estimated_size: 116,
            property_count: 0,
        }
    }

    /// `LoadedData` whose analyzable features mirror the sample.
    fn loaded(sample: Vec<SampledFeature>) -> LoadedData {
        let features = sample
            .iter()
            .map(|f| match &f.geometry {
                Geometry::Point(p) => analyzable(p.x(), p.y(), f.timestamp_ms),
                _ => unreachable!("fixture is points"),
            })
            .collect();
        LoadedData {
            features,
            bounds: BoundingBox::new(-73.6, 45.4, -73.0, 46.0),
            time_range: TimeRange::new(1_600_000_000_000, 1_600_000_800_000),
            sample,
        }
    }

    /// A real-analyzer `AnalysisResult` over the fixture, measured exactly as
    /// `analyze_source` measures it (single tile, build defaults), with the
    /// recommended max zoom pinned so the precision ladder is deterministic.
    fn analysis_for(data: &LoadedData, max_zoom: u8) -> AnalysisResult {
        let mut spatial = analysis::spatial::analyze(data).unwrap();
        spatial.recommended_max_zoom = max_zoom;
        spatial.recommended_min_zoom = spatial.recommended_min_zoom.min(max_zoom);
        let temporal = analysis::temporal::analyze(data).unwrap();
        let geometry = analysis::geometry::analyze(data).unwrap();
        let measured =
            crate::measure::measure_sample(&data.sample, &MeasureSettings::default()).unwrap();
        let density =
            analysis::density::analyze(data, &spatial, &temporal, measured.as_ref()).unwrap();
        AnalysisResult {
            source: "synthetic.parquet".to_string(),
            feature_count: data.features.len(),
            bounds: data.bounds,
            spatial,
            temporal,
            geometry,
            density,
            measured,
        }
    }

    fn json(advice: &[Advice]) -> String {
        serde_json::to_string(advice).unwrap()
    }

    fn find<'a>(advice: &'a [Advice], flag: &str) -> Option<&'a Advice> {
        advice.iter().find(|a| a.flag == flag)
    }

    /// Fractional shrink of `--quantize-coords` at `level`, measured directly —
    /// the ground truth the iteration is supposed to be reading.
    fn coords_shrink(sample: &[SampledFeature], layout: &SyntheticLayout, level: i32) -> f64 {
        let base = MeasureSettings {
            zstd_level: level,
            ..MeasureSettings::default()
        };
        let quantized = MeasureSettings {
            quantize_coords_m: Some(1.0),
            ..base.clone()
        };
        let b = measure_sample_layout(sample, &base, layout)
            .unwrap()
            .unwrap();
        let q = measure_sample_layout(sample, &quantized, layout)
            .unwrap()
            .unwrap();
        1.0 - q.bytes_total as f64 / b.bytes_total as f64
    }

    // ------------------------------------------------------------------
    // (1) THE ROLLBACK, as a test rather than a comment
    // ------------------------------------------------------------------

    #[test]
    fn rounds_zero_reproduces_the_single_pass_field_for_field() {
        let data = loaded(grid_sample(400, 40, 1));
        let result = analysis_for(&data, 14);
        let layout = SyntheticLayout::from_density(&result.density);

        let single_pass = run_all(&result, &data).unwrap();
        let rolled_back = run_iterative(&result, &data, &layout, 0).unwrap();

        assert_eq!(
            json(&rolled_back.advice),
            json(&single_pass),
            "rounds = 0 must reproduce the single pass exactly"
        );
        assert!(
            rolled_back.composed.is_none(),
            "the rollback measures nothing, so it claims nothing"
        );
    }

    // ------------------------------------------------------------------
    // (2) THE §12.2 NON-ADDITIVITY CASE — the item's whole point
    // ------------------------------------------------------------------

    #[test]
    fn quantization_that_only_pays_at_publish_zstd_flips_in_on_round_1() {
        // ONE layout for both round 0 and the iteration, so the ONLY thing that
        // differs between them is the zstd level the shrink is measured at.
        // (Production passes the density-derived layout; the single-tile cut
        // here isolates the interaction from the tile cut, which is a different
        // effect with its own item.)
        let layout = SyntheticLayout::single();
        let data = loaded(grid_sample(800, 40, 1));
        let result = analysis_for(&data, 14);

        // The fixture guard: quantization is NOT worth recommending at the
        // build-default level and IS at publish grade. If this ever stops
        // holding, the test below proves nothing and the fixture must be
        // re-based on data that does.
        let at_default = coords_shrink(&data.sample, &layout, 3);
        let at_publish = coords_shrink(&data.sample, &layout, layout::PUBLISH_ZSTD_LEVEL);
        assert!(
            at_default < quantize::MIN_COORD_SHRINK,
            "fixture guard: coords must be BELOW the emit threshold at zstd 3 ({at_default:.4})"
        );
        assert!(
            at_publish >= quantize::MIN_COORD_SHRINK,
            "fixture guard: coords must CLEAR the emit threshold at zstd 19 ({at_publish:.4})"
        );

        // Round 0 prices quantization against the build defaults and says no.
        let single_pass = run_all(&result, &data).unwrap();
        assert!(
            find(&single_pass, "--quantize-coords").is_none(),
            "single pass must not recommend quantization here: {single_pass:?}"
        );
        // …while recommending publish-grade zstd, which is what makes the
        // composed recipe a level-19 recipe.
        assert!(find(&single_pass, "--publish").is_some());

        // Iteration re-prices the lever ON that recipe, and it flips IN.
        let iterated = run_iterative(&result, &data, &layout, RECOMMEND_ROUNDS).unwrap();
        let coords = find(&iterated.advice, "--quantize-coords")
            .expect("quantization must flip in once priced on the composed recipe");
        assert_eq!(coords.value.as_deref(), Some("1"));
        // Admissibility is untouched: still lossy, still opt-in.
        assert!(coords.lossy, "iteration must not reclassify a lossy lever");
        assert!(!coords.suggestion_only);
        assert!(
            coords.why.contains("composed recipe (zstd 19)"),
            "why must say what it was measured on: {}",
            coords.why
        );
        assert!(coords
            .projected
            .as_deref()
            .is_some_and(|p| p.contains("(measured)")));

        // Advisor order survives the insertion: quantize leads.
        assert_eq!(iterated.advice[0].flag, "--quantize-coords");

        // And the recipe itself is now a measured quantity.
        let composed = iterated.composed.expect("composed measurement");
        assert_eq!(composed.settings.zstd_level, layout::PUBLISH_ZSTD_LEVEL);
        assert!(
            composed.settings.quantize_coords_m.is_none(),
            "the composed recipe is the NON-LOSSY point: {:?}",
            composed.settings
        );
        assert!(composed.composed_bytes < composed.default_bytes);
        // The lossy what-if is priced separately, and is smaller still.
        let with_lossy = composed.with_lossy_bytes.expect("lossy what-if");
        assert!(with_lossy < composed.composed_bytes, "{composed:?}");
    }

    #[test]
    fn quantization_that_stops_paying_at_publish_zstd_flips_back_out() {
        // The same interaction with the sign reversed, and the more important
        // direction of the two: here the single pass would have shown the user a
        // LOSSY lever worth 8.8% — a real quality tradeoff, sold on a number
        // that evaporates to ~1% once the recipe's own zstd level is applied.
        // Iteration withdraws it. Nothing about the lever's classification
        // changes; only whether its measured win clears its own bar.
        let layout = SyntheticLayout::single();
        let data = loaded(grid_sample(800, 8, 0));
        let result = analysis_for(&data, 14);

        let at_default = coords_shrink(&data.sample, &layout, 3);
        let at_publish = coords_shrink(&data.sample, &layout, layout::PUBLISH_ZSTD_LEVEL);
        assert!(
            at_default >= quantize::MIN_COORD_SHRINK,
            "fixture guard: coords must CLEAR the threshold at zstd 3 ({at_default:.4})"
        );
        assert!(
            at_publish < quantize::MIN_COORD_SHRINK,
            "fixture guard: coords must FALL BELOW it at zstd 19 ({at_publish:.4})"
        );

        let single_pass = run_all(&result, &data).unwrap();
        assert!(
            find(&single_pass, "--quantize-coords").is_some(),
            "fixture guard: the single pass must recommend quantization here"
        );

        let iterated = run_iterative(&result, &data, &layout, RECOMMEND_ROUNDS).unwrap();
        assert!(
            find(&iterated.advice, "--quantize-coords").is_none(),
            "a lossy lever that stops paying on the composed recipe must be withdrawn: {:?}",
            iterated.advice
        );
        // Everything iteration did not price is still there, in order.
        let untouched: Vec<&str> = single_pass
            .iter()
            .map(|a| a.flag.as_str())
            .filter(|f| *f != "--quantize-coords")
            .collect();
        let after: Vec<&str> = iterated.advice.iter().map(|a| a.flag.as_str()).collect();
        assert_eq!(untouched, after, "advisor order must survive a withdrawal");
        // With no lossy advice left, there is no lossy what-if to report.
        assert!(iterated
            .composed
            .expect("measurable")
            .with_lossy_bytes
            .is_none());
    }

    // ------------------------------------------------------------------
    // (3) the composed figure exists exactly when a measurement does
    // ------------------------------------------------------------------

    #[test]
    fn composed_figure_appears_only_when_the_sample_measured() {
        let layout = SyntheticLayout::single();

        // Below the measurement floor: the advisors already say nothing
        // measured, and iteration must not invent a composed number.
        let tiny = loaded(grid_sample(12, 40, 1));
        let tiny_result = analysis_for(&tiny, 14);
        assert!(tiny_result.measured.is_none(), "fixture guard");
        let iterated = run_iterative(&tiny_result, &tiny, &layout, RECOMMEND_ROUNDS).unwrap();
        assert!(iterated.composed.is_none());
        assert_eq!(
            json(&iterated.advice),
            json(&run_all(&tiny_result, &tiny).unwrap()),
            "with nothing measurable, iteration is a no-op"
        );

        // Measurable: the composed figure is published, and it is a real
        // measurement of the composed point rather than a restatement of the
        // baseline.
        let data = loaded(grid_sample(400, 40, 1));
        let result = analysis_for(&data, 14);
        let composed = run_iterative(&result, &data, &layout, RECOMMEND_ROUNDS)
            .unwrap()
            .composed
            .expect("measurable sample publishes a composed figure");
        assert_eq!(composed.rounds, RECOMMEND_ROUNDS);
        assert!(composed.features > 0);
        assert!(composed.default_bytes > 0);
        let direct = measure_sample_layout(&data.sample, &composed.settings, &layout)
            .unwrap()
            .unwrap();
        assert_eq!(
            composed.composed_bytes, direct.bytes_total,
            "the composed figure must be the composed point's own measurement"
        );
        assert!(composed.projected().contains("refinement round"));
    }

    // ------------------------------------------------------------------
    // (4) the cache: one encode per distinct settings point
    // ------------------------------------------------------------------

    #[test]
    fn the_cache_measures_each_settings_point_and_trial_exactly_once() {
        let sample = grid_sample(400, 40, 1);
        let layout = SyntheticLayout::single();
        let mut composer = Composer::new(&sample, &layout);
        let defaults = MeasureSettings::default();

        let first = composer.measure(&defaults).unwrap().unwrap();
        let second = composer.measure(&defaults).unwrap().unwrap();
        assert_eq!(first.bytes_total, second.bytes_total);
        assert_eq!(composer.usage().measurements, 1, "one encode, not two");
        assert_eq!(composer.usage().cache_hits, 1);

        // A different settings point is a different measurement.
        let publish = MeasureSettings {
            zstd_level: layout::PUBLISH_ZSTD_LEVEL,
            ..MeasureSettings::default()
        };
        composer.measure(&publish).unwrap();
        assert_eq!(composer.usage().measurements, 2);

        // Trials are cached on (incumbent, candidate).
        let candidates = [Candidate::ZstdLevel(layout::PUBLISH_ZSTD_LEVEL)];
        let a = composer.price(&defaults, &candidates).unwrap();
        let b = composer.price(&defaults, &candidates).unwrap();
        assert_eq!(a, b, "a cached trial must be the same trial");
        assert_eq!(composer.usage().oracle_calls, 1, "one oracle call, not two");
        assert_eq!(composer.usage().trials_priced, 1);
        assert_eq!(composer.usage().cache_hits, 2);

        // …and the SAME candidate against a DIFFERENT incumbent is a different
        // question, which is the entire point of the composed re-measurement.
        composer.price(&publish, &candidates).unwrap();
        assert_eq!(composer.usage().oracle_calls, 2);
        assert_eq!(composer.usage().trials_priced, 2);

        // The whole iteration stays inside the documented call budget.
        let data = loaded(grid_sample(400, 40, 1));
        let result = analysis_for(&data, 14);
        let usage = run_iterative(&result, &data, &layout, RECOMMEND_ROUNDS)
            .unwrap()
            .composed
            .expect("measurable")
            .usage;
        assert!(
            usage.measurements + usage.trials_priced <= 15,
            "oracle budget blown: {usage:?}"
        );
        assert!(
            usage.cache_hits > 0,
            "round 2 must be answered by the cache: {usage:?}"
        );
    }

    // ------------------------------------------------------------------
    // (5) admissibility is NOT a function of measurement
    // ------------------------------------------------------------------

    #[test]
    fn caveated_ordering_advice_survives_iteration_verbatim() {
        // Two spatial cells sampled across 24 hourly buckets: the range-read
        // simulation prefers `spatial`, which on a multi-bucket dataset carries
        // the playback caveat and is therefore SUGGESTION-ONLY. Auto-applying it
        // stalls time-playback, so no amount of re-measurement may promote it.
        let mut features = Vec::new();
        for bucket in 0..24u64 {
            features.push(analyzable(-73.0, 45.0, bucket * HOUR_MS));
            features.push(analyzable(-72.0, 45.0, bucket * HOUR_MS));
        }
        let mut data = loaded(grid_sample(400, 40, 1));
        data.features = features;
        data.time_range = TimeRange::new(0, 24 * HOUR_MS);

        let mut result = crate::test_support::sample_analysis();
        result.bounds = data.bounds;
        result.measured =
            crate::measure::measure_sample(&data.sample, &MeasureSettings::default()).unwrap();

        let layout = SyntheticLayout::single();
        let single_pass = run_all(&result, &data).unwrap();
        let ordering = find(&single_pass, "--blob-ordering").expect("caveated ordering advice");
        assert_eq!(ordering.value.as_deref(), Some("spatial"));
        assert!(ordering.suggestion_only, "fixture guard: {ordering:?}");

        let iterated = run_iterative(&result, &data, &layout, RECOMMEND_ROUNDS).unwrap();
        let after = find(&iterated.advice, "--blob-ordering").expect("ordering advice survives");
        assert_eq!(
            serde_json::to_string(after).unwrap(),
            serde_json::to_string(ordering).unwrap(),
            "the ordering simulation is re-run unchanged and its caveat rides through verbatim"
        );

        // A suggestion-only lever is never folded into the composed point, so
        // it cannot leak into a measurement and from there into a command.
        let composed = iterated.composed.expect("measurable");
        assert_eq!(composed.settings.zstd_level, layout::PUBLISH_ZSTD_LEVEL);
    }

    // ------------------------------------------------------------------
    // (6) composition rules
    // ------------------------------------------------------------------

    #[test]
    fn compose_settings_folds_in_only_what_it_may() {
        let entry = |flag: &str, value: Option<&str>, lossy: bool, suggestion: bool| Advice {
            flag: flag.to_string(),
            value: value.map(str::to_string),
            why: String::new(),
            projected: None,
            lossy,
            suggestion_only: suggestion,
            confidence: AdviceConfidence::Medium,
        };
        let advice = vec![
            entry("--publish", None, false, false),
            entry("--quantize-coords", Some("0.5"), true, false),
            entry("--quantize-attrs-auto", None, true, false),
            entry("--quantize-attr", Some("speed=0.25"), true, false),
            // Non-lossy but human-gated: never part of any recipe here.
            entry("--zstd-level", Some("19"), false, true),
            // Not an encoder knob at all: carried by the analysis, not by the
            // settings point.
            entry("--temporal-lod", Some("1d,30d"), false, false),
            entry("--blob-ordering", Some("spatial"), false, true),
        ];

        let theta = compose_settings(&advice, Lossy::Exclude);
        assert_eq!(theta.zstd_level, layout::PUBLISH_ZSTD_LEVEL);
        assert_eq!(theta.quantize_coords_m, None, "lossy stays out of θ");
        assert!(!theta.quantize_attrs_auto);
        assert!(theta.quantize_attrs.is_empty());
        assert!(theta.compact_times, "untouched levers keep their defaults");

        let with_lossy = compose_settings(&advice, Lossy::Include);
        assert_eq!(with_lossy.quantize_coords_m, Some(0.5));
        assert!(with_lossy.quantize_attrs_auto);
        assert_eq!(with_lossy.quantize_attrs.get("speed"), Some(&0.25));
        // Suggestion-only advice is excluded from BOTH points.
        assert_eq!(with_lossy.zstd_level, layout::PUBLISH_ZSTD_LEVEL);
    }

    #[test]
    fn a_shrink_inside_the_noise_is_reported_at_low_confidence() {
        // Significance gates CONFIDENCE, never emission: the threshold
        // constants are untouched by this item.
        assert!(matches!(
            confidence_with_noise(AdviceConfidence::High, 0.20, 0.15),
            AdviceConfidence::Low
        ));
        assert!(matches!(
            confidence_with_noise(AdviceConfidence::High, 0.20, 0.01),
            AdviceConfidence::High
        ));
        // No dispersion evidence never promotes and never demotes.
        assert!(matches!(
            confidence_with_noise(AdviceConfidence::Medium, 0.01, 0.0),
            AdviceConfidence::Medium
        ));
    }
}
