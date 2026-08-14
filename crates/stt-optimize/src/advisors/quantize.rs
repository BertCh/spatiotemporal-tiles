//! Coordinate / attribute quantization advisor (`--quantize-coords`,
//! `--quantize-attrs-auto`): derives candidate precision from the max-zoom
//! ground resolution and verifies the win by trial-encoding the sample.
//!
//! Quantization is the repo's #1 measured size lever (coords −25..47% on
//! coord-heavy datasets, attrs up to −80% per Float64 column), but it is
//! LOSSY, so this advisor never speaks without evidence: every recommendation
//! is backed by a real trial encode through the production stt-core encoder,
//! compared against ONE shared baseline measurement at build defaults.
//! Too-small samples produce no advice at all.
//!
//! # ⚠️ One layout, or the lossy levers fire on everything
//!
//! Baseline AND trials are measured under the run's own
//! [`SyntheticLayout`] — reconstructed here as
//! `SyntheticLayout::from_density(&result.density)`, the MO-4 invariant.
//!
//! This is not a tidiness point, it is the correctness of the whole advisor.
//! A single synthetic tile amortises per-tile encoder + zstd framing over the
//! entire sample, so it measures **1.47× cheaper** than the same features cut
//! at real occupancy (17 635 B vs 25 884 B on an 800-feature point sample).
//! Price a layout baseline against a single-tile trial and the difference in
//! CUT shows up as a ~32% "shrink" that no lever produced — enough to push
//! both `--quantize-coords` and `--quantize-attrs-auto`, both `lossy: true`,
//! over their thresholds at High confidence on *every* dataset. Recommending
//! lossy levers a dataset does not need is the failure mode this project's
//! no-thinning rule exists to prevent.
//!
//! So the layout is resolved ONCE in [`advise`] and threaded into every
//! measurement, and the baseline goes through
//! [`baseline_under`](crate::measure::baseline_under), which re-measures rather
//! than trusting a `result.measured` that was cut some other way.
//!
//! # ⚠️ The accept gate is TWO bars, ANDed — REAL *and* BIG ENOUGH
//!
//! A lossy lever spends something that cannot be bought back: fidelity. So it
//! is emitted iff its measured shrink clears **both** halves of
//! [`admits`], and failing either half is silence:
//!
//! | bar | question | answered by |
//! |---|---|---|
//! | **REAL** | is this win distinguishable from zero? | [`clears_the_noise_floor`] — [`ACCEPT_SIGMA`] standard errors of the trial's own replicate dispersion |
//! | **BIG ENOUGH** | is this win worth a permanent quality loss? | [`clears_the_magnitude_floor`] — [`MIN_COORD_SHRINK`] / [`MIN_ATTR_SHRINK`] of the measured sample |
//!
//! Neither bar subsumes the other, and dropping either one has a named failure
//! mode this module has already been through:
//!
//! * **Magnitude alone** (the pre-MO-9 gate) sold `--quantize-coords` on a 7.1%
//!   shrink measured to ±8.8pp — a number the measurement cannot tell from
//!   zero. Pinned by [`a_win_above_the_magnitude_floor_is_suppressed_when_it_is_noise`].
//! * **Significance alone** (the MO-9 gate, which this restores the other half
//!   of) sells a permanent quality loss for a **sub-1% win** as soon as the
//!   sample is large enough to measure one. Two measured cases from this
//!   crate's own fixtures went live: `--quantize-coords` advised at 0.98% (24σ)
//!   on the composed recipe, and at ~3% (12–14σ) at the build defaults.
//!   Significance is a statement about the ERROR BAR, and error bars shrink
//!   with sample size — so with enough data *any* loss is "significant". Pinned
//!   by [`a_real_but_tiny_win_is_never_worth_a_permanent_quality_loss`] and
//!   [`no_sub_one_percent_win_is_ever_advised_however_precisely_it_is_measured`].
//!
//! Confidence rides on the significance axis alone: anything emitted has
//! already cleared both bars, and it is sold at `High` rather than `Medium`
//! when it is established at [`HIGH_CONFIDENCE_SIGMA`]. ⚠️ Confidence means
//! WELL-ESTABLISHED, not LARGE; the magnitude is in `projected`.
//!
//! What has NOT changed, in any of it: every advisory here is still
//! `lossy: true`, still never joins [`to_command`](crate::recommend::to_command),
//! and is still never applied by `stt-build --auto` in any mode, budget or not.
//! The gate decides what a human is TOLD, never what a build DOES — and the
//! magnitude judgement the human then makes is supported by the advisory's
//! measured `projected` shrink and, under `--target-size`, by the
//! [`ShadowPrice`](crate::budget_solver::ShadowPrice) table's real byte counts.
//!
//! `stderr == 0.0` means there were fewer than two usable replicate blocks —
//! NO dispersion evidence, never "no noise" — and an unquantified shrink is not
//! evidence, so it emits nothing. That is the same doctrine as the
//! measurement floor above it: this advisor does not make unmeasured claims
//! about lossy levers.

use anyhow::Result;
use stt_core::arrow_tile::MIN_QUANTIZE_COORDS_M;

use super::{composed_note, Advice, AdviceConfidence, Composer, Repriced, SIGNIFICANCE_SIGMA};
use crate::analysis::AnalysisResult;
use crate::loader::{LoadedData, PropValue, SampledFeature};
use crate::measure::{baseline_under, MeasureSettings, MeasuredEncoding, SyntheticLayout};
use crate::oracle::{run_trials, Candidate, TrialResult};

/// WGS84 equatorial circumference in meters — the `2πR` in the standard
/// tile-pyramid ground-resolution formula.
const EARTH_CIRCUMFERENCE_M: f64 = 40_075_016.686;

/// Candidate `--quantize-coords` precisions in meters, descending. A derived
/// precision snaps DOWN to the first entry it covers and never goes below the
/// last (0.01 m is already sub-centimeter — finer buys nothing).
const PRECISION_LADDER_M: [f64; 6] = [5.0, 1.0, 0.5, 0.1, 0.05, 0.01];

/// The SIGNIFICANCE half of the gate: standard errors a measured shrink must
/// clear before a LOSSY lever may be recommended at all.
///
/// The advisor layer's one significance constant, shared with
/// [`confidence_with_noise`](super::confidence_with_noise) so a shrink cannot be
/// admissible under one rule and noise under another. It answers "is this win
/// REAL?", which the replicate blocks can actually answer — and *only* that.
/// "Is this win worth a permanent quality loss?" is a different question with a
/// different answer, and it is [`MIN_COORD_SHRINK`]/[`MIN_ATTR_SHRINK`]'s job.
pub(super) const ACCEPT_SIGMA: f64 = SIGNIFICANCE_SIGMA;

/// Standard errors a measured shrink must clear to be sold at `High`
/// confidence rather than `Medium`.
///
/// Three times [`ACCEPT_SIGMA`], mirroring the ratio the magnitude ladder
/// carries (`HIGH_CONFIDENCE_SHRINK = 0.15` is exactly 3 × [`MIN_COORD_SHRINK`])
/// — so the shape of the confidence ladder is the same on either axis.
///
/// ⚠️ Confidence means WELL-ESTABLISHED, not LARGE. The magnitude is gated by
/// the floor below and reported in the advisory's `projected` string and, under
/// a budget, in the shadow-price table's real byte counts.
pub(super) const HIGH_CONFIDENCE_SIGMA: f64 = 3.0 * ACCEPT_SIGMA;

/// The MAGNITUDE half of the gate for `--quantize-coords`: the fraction of the
/// measured sample the lever must save before a permanent loss of coordinate
/// precision is worth proposing to a human.
///
/// ⚠️ This floor is not a statement about measurement — [`ACCEPT_SIGMA`] is —
/// it is a statement about the EXCHANGE RATE, and it exists because the two
/// questions come apart in exactly the direction that hurts. A significance
/// test alone admits an arbitrarily small win once the sample is large enough
/// to resolve it (0.98% at 24σ was live advice before this floor came back);
/// resolution is a property of the sample size, and irreversibility is not.
///
/// Why 5% specifically:
///
/// * It is the value the SHIPPED gate uses — at HEAD this advisor accepts on
///   `shrink >= MIN_COORD_SHRINK` and nothing else — so every archive in the
///   fleet was advised against exactly this magnitude. Restoring it as the
///   floor makes the gate strictly STRICTER than the shipped one (the
///   significance conjunct can only remove advice, never add it) and moves no
///   dataset's standing advice in the emit direction, which is the only
///   direction that could need a republish window.
/// * It sits below the cheapest LOSSLESS lever the same advisors routinely
///   measure on the same samples (publish-grade zstd runs 9–21% on the fixtures
///   in this crate), which is the honest comparison: a lossy lever worth less
///   than a reversible one is never the trade to make first. The composed
///   re-price makes that comparison real rather than rhetorical — by round 1 the
///   shrink being floored is the MARGINAL win left over *after* the non-lossy
///   recipe has been applied.
/// * It leaves a 5× margin over the "never for a sub-1% win" line, so the floor
///   is not itself a knife edge.
pub(super) const MIN_COORD_SHRINK: f64 = 0.05;

/// The MAGNITUDE half of the gate for `--quantize-attrs-auto`. See
/// [`MIN_COORD_SHRINK`] for why a floor exists at all.
///
/// Lower than the coordinate floor (3% vs 5%) because the quality it spends is
/// cheaper and bounded differently: `--quantize-attrs-auto` maps a `Float64`
/// column onto ~65k range-adaptive levels, which for a measured property is
/// generally below its own instrument precision, whereas quantized coordinates
/// move every feature on the map. It is still far above the sub-1% line.
pub(super) const MIN_ATTR_SHRINK: f64 = 0.03;

/// The two floors' invariants, enforced at COMPILE time rather than by a test.
///
/// A test that compares two constants is folded away by the optimizer and
/// proves nothing; a `const` assertion fails the build. These are the two
/// properties the module doc promises and the gate's whole purpose rests on:
/// the coordinate floor is the stricter of the pair (quantized coordinates move
/// every feature on the map; quantized attributes move one column onto ~65k
/// levels), and NEITHER floor may ever be edited down into the sub-1% band that
/// a significance-only gate was selling.
const _: () = assert!(MIN_ATTR_SHRINK < MIN_COORD_SHRINK);
const _: () = assert!(MIN_ATTR_SHRINK > 0.01);

/// Recommend coordinate / attribute quantization when a trial encode of the
/// loader sample measures a shrink that clears its own noise floor.
pub fn advise(result: &AnalysisResult, data: &LoadedData) -> Result<Vec<Advice>> {
    // The run's ONE layout, reconstructed from the density occupancy scan. Every
    // measurement below — baseline, both trials, and every replicate block —
    // is taken under it, so each trial's delta is attributable to its lever and
    // not to the tile cut.
    let layout = SyntheticLayout::from_density(&result.density);
    // ONE baseline at build defaults, shared by both trials. The analysis
    // pipeline usually measured it already; it is reused only if it was cut the
    // same way, and re-measured otherwise. It supplies the per-column shares the
    // `why` strings quote — the trial oracle below carries the bytes.
    let Some(baseline) = baseline_under(result.measured.as_ref(), &data.sample, &layout)? else {
        // Sample too small to trial-encode. Quantization is lossy, so no
        // unmeasured claims — emit nothing.
        return Ok(Vec::new());
    };

    let (candidate, resolution) = coords_candidate(result);
    let float_cols = fractional_property_names(&data.sample);

    // Candidate order IS advisor emit order (coords, then attrs), and
    // `run_trials` answers in request order.
    let mut candidates: Vec<Candidate> = Vec::with_capacity(2);
    // Below the world-grid floor the encoder REJECTS the step (the ±180°
    // longitude index would overflow i32), so trial-encoding it does not fail
    // to find a win — it fails outright. The lever is simply not evaluable on
    // this dataset, and the honest answer is silence, which is exactly what
    // `reprice` already does with the same test. Deep-zoom datasets
    // (`recommended_max_zoom >= 21` at mid latitudes) snap to the ladder's
    // finest 0.01 m rung and land here.
    if candidate >= MIN_QUANTIZE_COORDS_M {
        candidates.push(Candidate::QuantizeCoords(Some(candidate)));
    }
    if !float_cols.is_empty() {
        candidates.push(Candidate::QuantizeAttrsAuto(true));
    }

    let mut advice = Vec::new();
    for trial in price(&data.sample, &layout, &candidates)? {
        // shrink = −delta_frac: the oracle signs deltas so negative is smaller.
        let shrink = -trial.delta_frac;
        // BOTH bars, per-lever floor: real enough to believe AND big enough to
        // be worth the fidelity it spends.
        if !admits(shrink, trial.stderr, magnitude_floor(&trial.candidate)) {
            continue;
        }
        let confidence = confidence_for(shrink, trial.stderr);
        match &trial.candidate {
            Candidate::QuantizeCoords(_) => advice.push(coords_advisory(
                candidate,
                &resolution,
                Some(&baseline),
                None,
                projected_shrink(shrink),
                confidence,
            )),
            Candidate::QuantizeAttrsAuto(_) => advice.push(attrs_advisory(
                &float_cols,
                Some(&baseline),
                None,
                projected_shrink(shrink),
                confidence,
            )),
            // `run_trials` answers in request order and this function only ever
            // requests the two levers above.
            _ => {}
        }
    }
    Ok(advice)
}

/// Price the candidate levers against the build defaults, under the run's
/// layout, WITH their replicate dispersion.
///
/// Round 0 goes through the MO-5 trial oracle rather than a bare trial encode
/// because the accept gate needs a `stderr` and only the oracle measures one.
/// The bytes are the same numbers the old direct encode produced —
/// `measure_sample_layout_with` documents `bytes_total` as attribution-design
/// independent, and the oracle uses the same cut — so the `projected` strings
/// did not move; only the decision reading them did.
///
/// # One infeasible lever must not silence the other
///
/// `run_trials` validates and prices the whole list, so a candidate that is
/// infeasible ON THIS DATA (not invalid — the recorded case is a quantization
/// step finer than the leaf type can address over a column's range) would take
/// its sibling down with it. On failure each candidate is retried alone and the
/// survivors are kept; the error is propagated only when NOTHING could be
/// priced, so a genuine encoder/compressor fault is still loud.
fn price(
    sample: &[SampledFeature],
    layout: &SyntheticLayout,
    candidates: &[Candidate],
) -> Result<Vec<TrialResult>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let defaults = MeasureSettings::default();
    let batch = run_trials(sample, layout, &defaults, candidates);
    if batch.is_ok() || candidates.len() == 1 {
        return batch;
    }
    let mut out = Vec::with_capacity(candidates.len());
    let mut last_err = None;
    for candidate in candidates {
        match run_trials(sample, layout, &defaults, std::slice::from_ref(candidate)) {
            Ok(mut trials) => out.append(&mut trials),
            Err(e) => last_err = Some(e),
        }
    }
    match last_err {
        Some(e) if out.is_empty() => Err(e),
        _ => Ok(out),
    }
}

/// Is a measured shrink distinguishable from zero at `sigma` standard errors?
///
/// `stderr == 0.0` is NO dispersion evidence (fewer than two usable replicate
/// blocks), never "no noise": it can never make a shrink significant. A
/// negative or non-finite shrink never clears anything.
fn significant_at(shrink: f64, stderr: f64, sigma: f64) -> bool {
    shrink.is_finite() && stderr.is_finite() && stderr > 0.0 && shrink > sigma * stderr
}

/// The REAL half of the gate: is the measured shrink distinguishable from zero
/// at [`ACCEPT_SIGMA`] standard errors of the trial's own dispersion?
///
/// ⚠️ Necessary, never sufficient — see [`admits`]. On its own this admits an
/// arbitrarily small permanent quality loss, because a big enough sample can
/// resolve one.
fn clears_the_noise_floor(shrink: f64, stderr: f64) -> bool {
    significant_at(shrink, stderr, ACCEPT_SIGMA)
}

/// The BIG ENOUGH half of the gate: is the measured shrink at least `floor` of
/// the sample?
///
/// ⚠️ Necessary, never sufficient — see [`admits`]. On its own this sells a
/// lossy lever on a large-looking number the measurement cannot tell from zero.
fn clears_the_magnitude_floor(shrink: f64, floor: f64) -> bool {
    shrink.is_finite() && shrink >= floor
}

/// THE ACCEPT GATE. A lossy lever is recommended iff its measured shrink is
/// BOTH real ([`clears_the_noise_floor`]) AND worth the fidelity it spends
/// ([`clears_the_magnitude_floor`]).
///
/// The conjunction is the whole point: each half has a live failure mode the
/// other one catches, documented on the module. Both halves read the SAME
/// measured shrink and the SAME trial dispersion in round 0 and in every
/// composed re-price — a round moves the measurement, never the bar.
fn admits(shrink: f64, stderr: f64, floor: f64) -> bool {
    clears_the_noise_floor(shrink, stderr) && clears_the_magnitude_floor(shrink, floor)
}

/// The magnitude floor that applies to a candidate lever.
///
/// Levers this advisor does not own get [`f64::INFINITY`], which no finite
/// shrink can clear — an unknown lever is never advised by accident. In
/// practice unreachable: this module only ever requests the two below.
fn magnitude_floor(candidate: &Candidate) -> f64 {
    match candidate {
        Candidate::QuantizeCoords(_) => MIN_COORD_SHRINK,
        Candidate::QuantizeAttrsAuto(_) => MIN_ATTR_SHRINK,
        _ => f64::INFINITY,
    }
}

/// The candidate precision for this dataset and the resolution clause that
/// justifies it: a quarter-pixel of ground resolution at the recommended max
/// zoom, snapped down [`PRECISION_LADDER_M`].
///
/// A pure function of the analysis — no measurement — so round 0 and every
/// later round derive the SAME candidate and their shrinks are comparable.
fn coords_candidate(result: &AnalysisResult) -> (f64, String) {
    let max_zoom = result.spatial.recommended_max_zoom;
    let lat_mid = (result.bounds.min_lat + result.bounds.max_lat) / 2.0;
    let m_per_px = meters_per_pixel(max_zoom, lat_mid);
    let candidate = snap_down_precision(m_per_px / 4.0);
    let resolution = format!(
        "at max zoom {max_zoom} one pixel covers ~{m_per_px:.2} m at lat {lat_mid:.1}°, \
         so {candidate} m fixed-point coords stay below a quarter-pixel of error"
    );
    (candidate, resolution)
}

/// Assemble the `--quantize-coords` advisory. `note` is the composed-recipe
/// clause an iterated re-price adds; round 0 passes `None` and gets the
/// original string byte for byte.
fn coords_advisory(
    candidate: f64,
    resolution: &str,
    baseline: Option<&MeasuredEncoding>,
    note: Option<&str>,
    projected: String,
    confidence: AdviceConfidence,
) -> Advice {
    let mut why = match baseline.and_then(|b| column_share(b, "geometry")) {
        Some(share) => format!(
            "geometry is {:.0}% of measured column bytes; {resolution}",
            share * 100.0
        ),
        None => resolution.to_string(),
    };
    if let Some(note) = note {
        why.push_str("; ");
        why.push_str(note);
    }
    Advice {
        flag: "--quantize-coords".to_string(),
        value: Some(candidate.to_string()),
        why,
        projected: Some(projected),
        lossy: true,
        suggestion_only: false,
        confidence,
    }
}

/// Assemble the `--quantize-attrs-auto` advisory, naming the top float
/// column(s) by measured baseline share. `note` is the composed-recipe clause an
/// iterated re-price adds; round 0 passes `None`.
fn attrs_advisory(
    float_cols: &[String],
    baseline: Option<&MeasuredEncoding>,
    note: Option<&str>,
    projected: String,
    confidence: AdviceConfidence,
) -> Advice {
    // `per_column` is already sorted descending by bytes.
    let mut cited: Vec<String> = baseline
        .map(|b| {
            b.per_column
                .iter()
                .filter(|c| float_cols.iter().any(|name| name == &c.name))
                .take(2)
                .map(|c| {
                    format!(
                        "`{}` ({:.0}% of measured column bytes)",
                        c.name,
                        c.share * 100.0
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    if cited.is_empty() {
        cited = float_cols
            .iter()
            .take(2)
            .map(|name| format!("`{name}`"))
            .collect();
    }
    let mut why = format!(
        "near-incompressible Float64 propert{} {} shrink to range-adaptive UInt16 (~65k levels)",
        if cited.len() == 1 { "y" } else { "ies" },
        cited.join(" and ")
    );
    if let Some(note) = note {
        why.push_str("; ");
        why.push_str(note);
    }
    Advice {
        flag: "--quantize-attrs-auto".to_string(),
        value: None,
        why,
        projected: Some(projected),
        lossy: true,
        suggestion_only: false,
        confidence,
    }
}

/// Re-price BOTH quantization levers against the composed incumbent θ.
///
/// This is the §12.2 interaction the single pass assumes away: θ carries the
/// non-lossy recipe (in practice, publish-grade zstd), and quantization is
/// worth a different fraction there than it is at the build defaults —
/// quantized coordinates are far more compressible than `Float64` ones, so the
/// level the compressor runs at changes the answer, sometimes across the emit
/// threshold.
///
/// Both levers are one-lever moves off θ: θ never carries a lossy setting, so
/// `QuantizeCoords`/`QuantizeAttrsAuto` are genuine perturbations of it, and
/// their deltas are attributable.
///
/// Returns `(coords, attrs)`. [`Repriced::Unmeasured`] means this round has no
/// evidence about that lever — the sample is unmeasurable, the column set does
/// not apply, or the derived precision is finer than the encoder's fixed-point
/// grid can address — and the round-0 verdict stands.
pub(super) fn reprice(
    result: &AnalysisResult,
    data: &LoadedData,
    theta: &MeasureSettings,
    composer: &mut Composer<'_>,
) -> Result<(Repriced, Repriced)> {
    let (candidate, resolution) = coords_candidate(result);
    let float_cols = fractional_property_names(&data.sample);

    // Below the world-grid floor the encoder rejects the step outright, so the
    // lever is not evaluable on this dataset — say nothing rather than error.
    let coords_evaluable = candidate >= MIN_QUANTIZE_COORDS_M;
    let mut candidates: Vec<Candidate> = Vec::with_capacity(2);
    if coords_evaluable {
        candidates.push(Candidate::QuantizeCoords(Some(candidate)));
    }
    if !float_cols.is_empty() {
        candidates.push(Candidate::QuantizeAttrsAuto(true));
    }
    if candidates.is_empty() {
        return Ok((Repriced::Unmeasured, Repriced::Unmeasured));
    }

    let priced = composer.price(theta, &candidates)?;
    if priced.is_empty() {
        // Unmeasurable sample: no evidence, no change.
        return Ok((Repriced::Unmeasured, Repriced::Unmeasured));
    }
    // The composed incumbent's own measurement supplies the per-column shares
    // the `why` strings cite. It is cached — the composed figure needs it too.
    let baseline = composer.measure(theta)?;

    let mut coords = Repriced::Unmeasured;
    let mut attrs = Repriced::Unmeasured;
    for trial in &priced {
        // shrink = −delta_frac: the oracle signs deltas so negative is smaller.
        let shrink = -trial.delta_frac;
        let note = composed_note(theta, trial.stderr);
        // The SAME accept gate round 0 uses — both bars, same per-lever floor,
        // same statistic. The composed point moves the shrink and its noise,
        // never the bar. This is where a lossy lever whose win EVAPORATES once
        // the non-lossy recipe is applied gets withdrawn: at θ the shrink being
        // floored is the MARGINAL win left over after zstd 19 has had its turn.
        let admissible = admits(shrink, trial.stderr, magnitude_floor(&trial.candidate));
        let confidence = confidence_for(shrink, trial.stderr);
        match &trial.candidate {
            Candidate::QuantizeCoords(_) => {
                coords = Repriced::Measured(admissible.then(|| {
                    coords_advisory(
                        candidate,
                        &resolution,
                        baseline.as_ref(),
                        Some(&note),
                        projected_shrink(shrink),
                        confidence,
                    )
                }));
            }
            Candidate::QuantizeAttrsAuto(_) => {
                attrs = Repriced::Measured(admissible.then(|| {
                    attrs_advisory(
                        &float_cols,
                        baseline.as_ref(),
                        Some(&note),
                        projected_shrink(shrink),
                        confidence,
                    )
                }));
            }
            // `price` returns results in request order and this function only
            // ever requests the two levers above.
            _ => {}
        }
    }
    Ok((coords, attrs))
}

/// Ground resolution of one 256px-tile pixel at `zoom`, at latitude
/// `lat_deg`: `circumference * cos(lat) / (256 * 2^zoom)`.
fn meters_per_pixel(zoom: u8, lat_deg: f64) -> f64 {
    EARTH_CIRCUMFERENCE_M * lat_deg.to_radians().cos() / (256.0 * f64::powi(2.0, zoom as i32))
}

/// Snap a derived precision DOWN to the [`PRECISION_LADDER_M`] (the first
/// rung it covers); precisions below the ladder clamp to its finest rung.
fn snap_down_precision(precision_m: f64) -> f64 {
    for &rung in &PRECISION_LADDER_M {
        if precision_m >= rung {
            return rung;
        }
    }
    PRECISION_LADDER_M[PRECISION_LADDER_M.len() - 1]
}

/// Measured baseline share of the named column, if it was attributed.
fn column_share(baseline: &MeasuredEncoding, name: &str) -> Option<f64> {
    baseline
        .per_column
        .iter()
        .find(|c| c.name == name)
        .map(|c| c.share)
}

/// Fractional shrink of a trial encode versus the baseline (positive =
/// smaller, negative = the trial got BIGGER).
///
/// Retained for the tests that characterise a fixture's honest same-cut shrink
/// directly; the production paths read the trial oracle's signed `delta_frac`,
/// which is this quantity plus its dispersion.
#[cfg(test)]
fn shrink_vs(baseline: &MeasuredEncoding, trial: &MeasuredEncoding) -> f64 {
    1.0 - trial.bytes_total as f64 / baseline.bytes_total.max(1) as f64
}

/// The `projected` string for a measured shrink, e.g. `-36% sample encode
/// (measured)`.
fn projected_shrink(shrink: f64) -> String {
    format!("-{:.0}% sample encode (measured)", shrink * 100.0)
}

/// `High` when the shrink is established at [`HIGH_CONFIDENCE_SIGMA`] standard
/// errors, `Medium` otherwise (anything emitted has already cleared
/// [`ACCEPT_SIGMA`]).
fn confidence_for(shrink: f64, stderr: f64) -> AdviceConfidence {
    if significant_at(shrink, stderr, HIGH_CONFIDENCE_SIGMA) {
        AdviceConfidence::High
    } else {
        AdviceConfidence::Medium
    }
}

/// Sampled property names carrying at least one finite fractional numeric
/// value, in first-seen order. The loader widens every numeric Arrow type to
/// f64 (dropping the source type), so a fractional value is the evidence that
/// a column is genuinely Float64 — integer-valued columns lose nothing to
/// `--quantize-attrs-auto` being skipped and never trigger it here.
fn fractional_property_names(sample: &[SampledFeature]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for feature in sample {
        for (name, value) in &feature.properties {
            if let PropValue::Number(x) = value {
                if x.is_finite() && x.fract() != 0.0 && !names.iter().any(|n| n == name) {
                    names.push(name.clone());
                }
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis;
    use crate::loader::{AnalyzableFeature, GeometryType};
    use geo_types::{Geometry, LineString, Point};
    use stt_core::types::{BoundingBox, TimeRange};

    /// Deterministic pseudo-noise in [0, 1) (Knuth multiplicative hash) —
    /// keeps f64 mantissas high-entropy so quantization has real bytes to win.
    fn noise(i: usize, salt: u64) -> f64 {
        ((i as u64).wrapping_add(salt).wrapping_mul(2_654_435_761) % 1_000_000) as f64 / 1e6
    }

    /// `n` points on an EXACT `cols`-wide lon/lat grid — no jitter. Repeating
    /// coordinates are already compressible as `Float64`, so quantization wins
    /// little and `--quantize-coords` must NOT be recommended. This is the
    /// must-stay-silent shape, and the one a cross-cut baseline would have
    /// falsely fired on.
    fn grid_sample(n: usize, cols: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| SampledFeature {
                geometry: Geometry::Point(Point::new(
                    -73.5 + (i % cols) as f64 * 0.01,
                    45.5 + (i / cols) as f64 * 0.01,
                )),
                timestamp_ms: 1_600_000_000_000 + i as u64 * 1_000,
                properties: vec![(
                    "region".to_string(),
                    PropValue::Text(format!("region-{}", i % 5)),
                )],
            })
            .collect()
    }

    /// splitmix64 — full-entropy token choices with no RNG dependency, so every
    /// fixture below is byte-identical on every run and every machine.
    fn mix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// [`grid_sample`]'s exact lon/lat grid, DILUTED by `notes` high-entropy
    /// text columns.
    ///
    /// The dilution is the point: grid coordinates still shrink under
    /// fixed-point encoding, but the text columns dominate the tile, so the
    /// shrink lands in the low single digits — under [`MIN_COORD_SHRINK`] while
    /// remaining tens of standard errors from zero. This is the shape a
    /// significance-only gate sells and the magnitude floor refuses.
    fn diluted_grid_sample(n: usize, cols: usize, notes: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| SampledFeature {
                geometry: Geometry::Point(Point::new(
                    -73.5 + (i % cols) as f64 * 0.01,
                    45.5 + (i / cols) as f64 * 0.01,
                )),
                timestamp_ms: 1_600_000_000_000 + i as u64 * 1_000,
                properties: (0..notes)
                    .map(|p| {
                        let token = mix(i as u64 * 7 + p as u64) % 64;
                        (
                            format!("note{p}"),
                            PropValue::Text(format!(
                                "observation-station-{token:04}-quality-{}-{}",
                                token % 7,
                                mix(i as u64 + p as u64 * 31) % 1000
                            )),
                        )
                    })
                    .collect(),
            })
            .collect()
    }

    /// `n` features at ONE point. The geometry column is a constant, so its
    /// `Float64` runs compress to almost nothing and fixed-point coordinates
    /// measure LARGER — the negative-shrink shape.
    fn colocated_sample(n: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| SampledFeature {
                geometry: Geometry::Point(Point::new(-73.5, 45.5)),
                timestamp_ms: 1_600_000_000_000 + i as u64 * 1_000,
                properties: vec![(
                    "region".to_string(),
                    PropValue::Text(format!("region-{}", i % 5)),
                )],
            })
            .collect()
    }

    /// A deliberately HETEROGENEOUS sample: `colo` colocated features (where
    /// quantization loses) followed by `hi` high-entropy ones (where it wins
    /// big).
    ///
    /// The trial oracle's replicate blocks are contiguous stretches of the
    /// sample, so blocks drawn from the two halves disagree violently and the
    /// trial's stderr swamps its mean — a large-looking shrink with no evidence
    /// behind it. The magnitude floor cannot see the difference; the
    /// significance half can, which is why the gate needs both.
    fn dispersed_sample(colo: usize, hi: usize) -> Vec<SampledFeature> {
        let mut out = colocated_sample(colo);
        out.extend(point_sample(hi, false));
        out
    }

    /// `n` short two-vertex tracks with a high-entropy float property — the
    /// line encoder path, which packs a separate offsets buffer and so charges
    /// per-tile framing differently from points.
    fn line_sample(n: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| {
                let lon = -73.8 + (i as f64 * 0.003) % 0.6 + noise(i, 0) * 1e-3;
                let lat = 45.2 + (i % 7) as f64 * 0.08 + noise(i, 17) * 1e-3;
                SampledFeature {
                    geometry: Geometry::LineString(LineString::from(vec![
                        (lon, lat),
                        (lon + noise(i, 31) * 1e-2, lat + noise(i, 47) * 1e-2),
                    ])),
                    timestamp_ms: 1_600_000_000_000 + i as u64 * 60_000,
                    properties: vec![(
                        "speed".to_string(),
                        PropValue::Number(1.0 + noise(i, 7) * 9.0),
                    )],
                }
            })
            .collect()
    }

    /// n Montréal-area points. `float_prop` picks a high-precision fractional
    /// f64 property (`magnitude`) versus an integer-valued one (`count`); a
    /// categorical `region` string rides along either way.
    fn point_sample(n: usize, float_prop: bool) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| {
                let numeric = if float_prop {
                    (
                        "magnitude".to_string(),
                        PropValue::Number(1.0 + noise(i, 7) * 9.0),
                    )
                } else {
                    ("count".to_string(), PropValue::Number((i % 10) as f64))
                };
                SampledFeature {
                    geometry: Geometry::Point(Point::new(
                        -73.8 + (i as f64 * 0.003) % 0.6 + noise(i, 0) * 1e-3,
                        45.2 + (i % 7) as f64 * 0.08 + noise(i, 17) * 1e-3,
                    )),
                    timestamp_ms: 1_600_000_000_000 + i as u64 * 60_000,
                    properties: vec![
                        numeric,
                        (
                            "region".to_string(),
                            PropValue::Text(format!("region-{}", i % 5)),
                        ),
                    ],
                }
            })
            .collect()
    }

    fn loaded(sample: Vec<SampledFeature>) -> LoadedData {
        let features = sample
            .iter()
            .map(|f| {
                let (kind, lon, lat, vertices) = match &f.geometry {
                    Geometry::Point(p) => (GeometryType::Point, p.x(), p.y(), 1),
                    Geometry::LineString(ls) => {
                        let first =
                            ls.0.first()
                                .copied()
                                .unwrap_or(geo_types::coord! {x:0.0,y:0.0});
                        (GeometryType::LineString, first.x, first.y, ls.0.len())
                    }
                    _ => (GeometryType::Unknown, 0.0, 0.0, 1),
                };
                AnalyzableFeature {
                    lon,
                    lat,
                    timestamp: f.timestamp_ms,
                    geometry_type: kind,
                    vertex_count: vertices,
                    estimated_size: 25,
                    property_count: f.properties.len(),
                }
            })
            .collect();
        LoadedData {
            features,
            bounds: BoundingBox {
                min_lon: -73.8,
                min_lat: 45.2,
                max_lon: -73.2,
                max_lat: 45.8,
            },
            time_range: TimeRange {
                start: 1_600_000_000_000,
                end: 1_600_012_000_000,
            },
            sample,
        }
    }

    /// Real-analyzer AnalysisResult over the synthetic data, with the
    /// recommended max zoom pinned so the precision-ladder derivation is
    /// deterministic.
    ///
    /// `mode` mirrors `crate::MeasurementMode` exactly, including the ordering
    /// that makes the layout derivable: occupancy scan first (uncalibrated),
    /// layout off it, measurement, then the final calibrated density pass. That
    /// ordering is what lets `SyntheticLayout::from_density(&result.density)`
    /// reconstruct the cut `result.measured` was taken under.
    fn analysis_result_mode(
        data: &LoadedData,
        max_zoom: u8,
        mode: crate::MeasurementMode,
    ) -> AnalysisResult {
        let mut spatial = analysis::spatial::analyze(data).unwrap();
        spatial.recommended_max_zoom = max_zoom;
        spatial.recommended_min_zoom = spatial.recommended_min_zoom.min(max_zoom);
        let temporal = analysis::temporal::analyze(data).unwrap();
        let geometry = analysis::geometry::analyze(data).unwrap();
        let measured = match mode {
            crate::MeasurementMode::SingleTile => {
                crate::measure::measure_sample(&data.sample, &MeasureSettings::default()).unwrap()
            }
            crate::MeasurementMode::DensityLayout => {
                let occupancy =
                    analysis::density::analyze(data, &spatial, &temporal, None).unwrap();
                let layout = SyntheticLayout::from_density(&occupancy);
                crate::measure::measure_sample_layout(
                    &data.sample,
                    &MeasureSettings::default(),
                    &layout,
                )
                .unwrap()
            }
        };
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

    /// [`analysis_result_mode`] at the production default.
    fn analysis_result(data: &LoadedData, max_zoom: u8) -> AnalysisResult {
        analysis_result_mode(data, max_zoom, crate::MeasurementMode::default())
    }

    #[test]
    fn high_precision_points_get_measured_coords_advice() {
        let data = loaded(point_sample(200, true));
        let result = analysis_result(&data, 14);
        let advice = advise(&result, &data).unwrap();
        let coords = advice
            .iter()
            .find(|a| a.flag == "--quantize-coords")
            .expect("high-entropy f64 coords must produce measured coords advice");
        // z14 at lat 45.5 → ~6.7 m/px, /4 = 1.67 m → snaps down to 1 m.
        assert_eq!(coords.value.as_deref(), Some("1"));
        assert!(coords.lossy);
        let projected = coords.projected.as_deref().unwrap();
        assert!(
            projected.starts_with('-') && projected.contains("(measured)"),
            "projected = {projected}"
        );
        assert!(
            coords.why.contains("zoom 14") && coords.why.contains("1 m"),
            "why = {}",
            coords.why
        );
    }

    #[test]
    fn fractional_float_property_gets_attrs_advice() {
        let data = loaded(point_sample(200, true));
        let result = analysis_result(&data, 14);
        let advice = advise(&result, &data).unwrap();
        let attrs = advice
            .iter()
            .find(|a| a.flag == "--quantize-attrs-auto")
            .expect("high-entropy Float64 property must produce attrs advice");
        assert!(attrs.lossy);
        assert!(attrs.value.is_none());
        assert!(attrs.why.contains("magnitude"), "why = {}", attrs.why);
        assert!(attrs.projected.as_deref().unwrap().contains("(measured)"));
    }

    #[test]
    fn tiny_sample_produces_no_advice() {
        let data = loaded(point_sample(20, true));
        let result = analysis_result(&data, 14);
        assert!(
            result.measured.is_none(),
            "under MIN_MEASURE_FEATURES must not measure"
        );
        let advice = advise(&result, &data).unwrap();
        assert!(
            advice.is_empty(),
            "lossy advice without measurement: {advice:?}"
        );
    }

    #[test]
    fn integer_and_string_properties_get_no_attrs_advice() {
        let data = loaded(point_sample(200, false));
        let result = analysis_result(&data, 14);
        let advice = advise(&result, &data).unwrap();
        assert!(
            !advice.iter().any(|a| a.flag == "--quantize-attrs-auto"),
            "integer-valued numeric props must not trigger attr quantization"
        );
    }

    // ------------------------------------------------------------------
    // The layout migration: closing the gate WITHOUT moving admissibility
    // ------------------------------------------------------------------

    /// Exactly what the PRE-migration advisor decided: baseline and both trials
    /// measured single-tile, the same two thresholds applied. This is the oracle
    /// the migration is checked against — not a re-derivation of the new code.
    fn legacy_single_tile_flags(result: &AnalysisResult, data: &LoadedData) -> Vec<String> {
        let defaults = MeasureSettings::default();
        let single = |settings: &MeasureSettings| {
            crate::measure::measure_sample(&data.sample, settings).unwrap()
        };
        let Some(baseline) = single(&defaults) else {
            return Vec::new();
        };
        let mut flags = Vec::new();
        let (candidate, _) = coords_candidate(result);
        if candidate >= MIN_QUANTIZE_COORDS_M {
            if let Some(trial) = single(&MeasureSettings {
                quantize_coords_m: Some(candidate),
                ..defaults.clone()
            }) {
                if shrink_vs(&baseline, &trial) >= MIN_COORD_SHRINK {
                    flags.push("--quantize-coords".to_string());
                }
            }
        }
        if !fractional_property_names(&data.sample).is_empty() {
            if let Some(trial) = single(&MeasureSettings {
                quantize_attrs_auto: true,
                ..defaults.clone()
            }) {
                if shrink_vs(&baseline, &trial) >= MIN_ATTR_SHRINK {
                    flags.push("--quantize-attrs-auto".to_string());
                }
            }
        }
        flags
    }

    /// What `advise` emits under the production (layout) path.
    fn fired_flags(data: &LoadedData, max_zoom: u8) -> Vec<String> {
        advise(&analysis_result(data, max_zoom), data)
            .unwrap()
            .into_iter()
            .map(|a| a.flag)
            .collect()
    }

    #[test]
    fn the_layout_migration_did_not_move_which_datasets_fire() {
        // THE acceptance check for closing the `MeasurementMode` gate, run
        // empirically rather than argued. Moving the tile cut is allowed to move
        // the measured NUMBERS; it must not move which datasets are told to give
        // up fidelity.
        //
        // Cases span both levers' emit/suppress corners: high-entropy f64
        // properties (both levers in play), integer + string properties (attrs
        // must stay silent), a deep zoom whose derived precision is below the
        // world-grid floor (coords not evaluable), the line encoder path, and a
        // sample below the measurement floor (nothing at all).
        //
        // The ONE shape that does move — a fixture whose honest shrink straddles
        // the emit bar — is not swept under this assertion; it is characterised,
        // with its numbers, in
        // [`measuring_at_real_occupancy_can_move_a_knife_edge_verdict`].
        let cases: [(&str, Vec<SampledFeature>, u8); 5] = [
            ("high-entropy float props @z14", point_sample(400, true), 14),
            ("integer + string props @z14", point_sample(400, false), 14),
            ("high-entropy float props @z22", point_sample(400, true), 22),
            ("high-entropy float props @z6", point_sample(400, true), 6),
            ("two-vertex tracks @z14", line_sample(400), 14),
        ];
        for (label, sample, max_zoom) in cases {
            let data = loaded(sample);
            // The layout the production default now measures under…
            let fired = fired_flags(&data, max_zoom);
            // …versus what the all-single-tile advisor decided.
            let before = legacy_single_tile_flags(
                &analysis_result_mode(&data, max_zoom, crate::MeasurementMode::SingleTile),
                &data,
            );
            assert_eq!(
                fired, before,
                "{label}: the layout migration changed which lossy levers fire"
            );
        }
        // A sample below the measurement floor says nothing either way.
        assert!(fired_flags(&loaded(point_sample(20, true)), 14).is_empty());
    }

    #[test]
    fn measuring_at_real_occupancy_can_move_a_knife_edge_verdict() {
        // ⚠️ THE ONE BEHAVIOUR CHANGE the layout migration makes, pinned with
        // its numbers rather than left as folklore.
        //
        // On an exact-grid fixture (repeating coordinates, already compressible
        // as `Float64`, one small string property) the honest same-cut shrinks
        // straddle `MIN_COORD_SHRINK`:
        //
        // ```text
        //   cut          baseline B   trial B   shrink    verdict
        //   single tile      5 669      5 467    3.56%    silent
        //   8-tile layout   12 100     11 489    5.05%    emits (Medium)
        //   cross-cut       12 100      5 467   54.8%     ← the ARTIFACT
        // ```
        //
        // The artifact is an order of magnitude away from either honest number,
        // so it is demonstrably NOT what moved the verdict. What moved it is the
        // correction itself: at ~100 features per tile, zstd cannot exploit the
        // grid's long-range coordinate repetition the way it can across a
        // 800-feature sample, so fixed-point coordinates save proportionally
        // MORE on realistically-sized tiles.
        //
        // ⚠️ Note. Both cuts' shrinks are established many standard errors from
        // zero, so the SIGNIFICANCE half of the gate is satisfied either way —
        // it is the MAGNITUDE half that the cut moves this fixture across, which
        // is why the straddle is stated against `MIN_COORD_SHRINK`. The
        // assertion that matters is the one below it: the advisor follows its
        // OWN same-cut measurement and never the cross-cut artifact.
        let data = loaded(grid_sample(800, 40));
        let result = analysis_result(&data, 14);
        let layout = SyntheticLayout::from_density(&result.density);
        let single = SyntheticLayout::single();
        let (candidate, _) = coords_candidate(&result);
        let quantized = MeasureSettings {
            quantize_coords_m: Some(candidate),
            ..MeasureSettings::default()
        };
        let at = |settings: &MeasureSettings, cut: &SyntheticLayout| {
            crate::measure::measure_sample_layout(&data.sample, settings, cut)
                .unwrap()
                .unwrap()
        };
        let single_base = at(&MeasureSettings::default(), &single);
        let single_trial = at(&quantized, &single);
        let layout_base = at(&MeasureSettings::default(), &layout);
        let layout_trial = at(&quantized, &layout);

        let single_shrink = shrink_vs(&single_base, &single_trial);
        let layout_shrink = shrink_vs(&layout_base, &layout_trial);
        let artifact = shrink_vs(&layout_base, &single_trial);

        // Fixture guard: the two honest cuts must straddle the magnitude floor,
        // or this is not the knife-edge shape it claims to be.
        assert!(
            single_shrink < MIN_COORD_SHRINK,
            "fixture guard: single-tile must stay below the magnitude floor ({single_shrink:.4})"
        );
        assert!(
            layout_shrink >= MIN_COORD_SHRINK,
            "fixture guard: the honest layout cut must clear it ({layout_shrink:.4})"
        );
        // The artifact is nowhere near either honest number — the skew is gone,
        // and the verdict move is not it.
        assert!(
            artifact > 5.0 * layout_shrink,
            "the cross-cut artifact must be an order of magnitude larger than the \
             honest shrink it is being distinguished from: {artifact:.4} vs {layout_shrink:.4}"
        );

        // The advisor follows its OWN same-cut measurement — never the artifact.
        let coords = advise(&result, &data)
            .unwrap()
            .into_iter()
            .find(|a| a.flag == "--quantize-coords")
            .expect("the honest layout shrink clears the bar, so it is emitted");
        assert_eq!(
            coords.projected.as_deref(),
            Some(projected_shrink(layout_shrink).as_str())
        );
        assert!(coords.lossy, "still opt-in, still never auto-applied");
    }

    #[test]
    fn a_mismatched_baseline_cannot_fabricate_a_shrink() {
        // The failure the gate existed to prevent, pinned as a test so it can
        // never come back: a layout baseline priced against a single-tile trial
        // manufactures a shrink out of the CUT, not the lever.
        let data = loaded(point_sample(400, true));
        let honest = analysis_result(&data, 14);
        let layout = SyntheticLayout::from_density(&honest.density);
        assert!(
            layout.tiles() > 1,
            "fixture guard: the run layout must actually split"
        );

        let defaults = MeasureSettings::default();
        let quantized = MeasureSettings {
            quantize_coords_m: Some(coords_candidate(&honest).0),
            ..defaults.clone()
        };
        let layout_base = honest.measured.clone().expect("measurable");
        let single_base = crate::measure::measure_sample(&data.sample, &defaults)
            .unwrap()
            .unwrap();
        let single_trial = crate::measure::measure_sample(&data.sample, &quantized)
            .unwrap()
            .unwrap();
        let layout_trial = crate::measure::measure_sample_layout(&data.sample, &quantized, &layout)
            .unwrap()
            .unwrap();

        // Same cut, both arms: the honest answer.
        let honest_shrink = shrink_vs(&layout_base, &layout_trial);
        // Cross-cut: the fabrication. It is worth ~the 1.47x framing gap.
        let fabricated = shrink_vs(&layout_base, &single_trial);
        assert!(
            fabricated > honest_shrink + 0.15,
            "the cross-cut artifact must be large enough to matter: fabricated \
             {fabricated:.4} vs honest {honest_shrink:.4}"
        );
        assert!(
            fabricated > 2.0 * honest_shrink,
            "…and a MULTIPLE of it, not a nudge — the artifact more than doubles the \
             honest number: {fabricated:.4} vs {honest_shrink:.4}"
        );
        assert!(shrink_vs(&single_base, &single_trial) < fabricated);

        // Now the defence: hand the advisor an AnalysisResult carrying exactly
        // that mismatched baseline — a single-tile `measured` beside a density
        // whose layout is multi-tile — and it must not use it.
        let mut mismatched = honest.clone();
        mismatched.measured = Some(single_base);
        assert!(
            !layout.produced(mismatched.measured.as_ref().unwrap()),
            "fixture guard: the planted baseline must be detectably off-cut"
        );
        assert_eq!(
            serde_json::to_string(&advise(&mismatched, &data).unwrap()).unwrap(),
            serde_json::to_string(&advise(&honest, &data).unwrap()).unwrap(),
            "an off-cut baseline must be re-measured, not compared against"
        );
    }

    #[test]
    fn a_precision_below_the_world_grid_floor_is_silence_not_an_error() {
        // A dataset whose recommended max zoom is deep enough that a
        // quarter-pixel snaps to the ladder's finest 0.01 m rung: the encoder
        // rejects that step outright (the world-anchored longitude index would
        // overflow i32), so the lever is not evaluable and the advisor says
        // nothing. It used to propagate the encoder's error out of `advise`,
        // which took the WHOLE recommendation down over one inapplicable lever;
        // `reprice` already had this guard, round 0 did not.
        let data = loaded(point_sample(400, true));
        let result = analysis_result(&data, 22);
        assert!(
            coords_candidate(&result).0 < MIN_QUANTIZE_COORDS_M,
            "fixture guard: z22 must derive a sub-floor precision"
        );
        let advice = advise(&result, &data).expect("an inapplicable lever must not be an error");
        assert!(!advice.iter().any(|a| a.flag == "--quantize-coords"));
        // The other lever is unaffected — one lever falling out must not
        // silence the advisor.
        assert!(advice.iter().any(|a| a.flag == "--quantize-attrs-auto"));
    }

    // ------------------------------------------------------------------
    // The accept gate is TWO bars, ANDed: REAL and BIG ENOUGH
    //
    // Each test below owns one corner. The two suppression cases are the
    // load-bearing ones — each is a lossy lever that ONE half of the gate
    // would have sold, caught by the other half.
    // ------------------------------------------------------------------

    /// The measured coords trial for a fixture, under the production path.
    fn coords_trial(data: &LoadedData, max_zoom: u8) -> TrialResult {
        let result = analysis_result(data, max_zoom);
        let layout = SyntheticLayout::from_density(&result.density);
        let (candidate, _) = coords_candidate(&result);
        run_trials(
            &data.sample,
            &layout,
            &MeasureSettings::default(),
            &[Candidate::QuantizeCoords(Some(candidate))],
        )
        .unwrap()
        .pop()
        .expect("the fixture is measurable")
    }

    #[test]
    fn the_significance_half_asks_whether_a_win_is_real() {
        // The unit statement of ONE half of the gate, with no encoder in the
        // loop. `clears_the_noise_floor` reads the error bar and nothing else:
        // a 0.2% shrink measured to ±0.01% is REAL; a 40% shrink measured to
        // ±30% is not. (Whether either is worth having is the other half's
        // question — see `admits` below, where the 0.2% one is refused.)
        assert!(clears_the_noise_floor(0.002, 0.0001));
        assert!(!clears_the_noise_floor(0.40, 0.30));

        // Strictly greater than the multiple: exactly at the bar is not
        // evidence, it is the definition of the bar.
        assert!(!clears_the_noise_floor(0.02, 0.01));
        assert!(clears_the_noise_floor(0.0201, 0.01));

        // NO dispersion evidence (fewer than two usable replicate blocks) can
        // never make a shrink significant — `0.0` is an absence, not a claim of
        // perfect precision.
        assert!(!clears_the_noise_floor(0.9, 0.0));
        assert!(!clears_the_noise_floor(0.9, -1.0));

        // A lever that makes the archive BIGGER is never admissible however
        // precisely it was measured, and non-finite arithmetic never leaks a
        // `true`.
        assert!(!clears_the_noise_floor(-0.5, 0.001));
        assert!(!clears_the_noise_floor(0.0, 0.001));
        assert!(!clears_the_noise_floor(f64::NAN, 0.001));
        assert!(!clears_the_noise_floor(f64::INFINITY, f64::NAN));

        // High confidence is the same test at a wider multiple, so it implies
        // significance — the ladder can never invert.
        for (shrink, stderr) in [(0.10, 0.001), (0.02, 0.003), (0.5, 0.02)] {
            if significant_at(shrink, stderr, HIGH_CONFIDENCE_SIGMA) {
                assert!(
                    clears_the_noise_floor(shrink, stderr),
                    "High without Accept at ({shrink}, {stderr})"
                );
            }
        }
        assert_eq!(
            HIGH_CONFIDENCE_SIGMA.max(ACCEPT_SIGMA),
            HIGH_CONFIDENCE_SIGMA,
            "the confidence bar must sit ABOVE the accept bar"
        );
        assert_ne!(HIGH_CONFIDENCE_SIGMA, ACCEPT_SIGMA);
    }

    #[test]
    fn the_magnitude_half_asks_whether_a_win_is_worth_a_permanent_loss() {
        // The other half, equally unit. It reads the shrink and nothing else —
        // no error bar can talk it into or out of an answer.
        assert!(clears_the_magnitude_floor(
            MIN_COORD_SHRINK,
            MIN_COORD_SHRINK
        ));
        assert!(clears_the_magnitude_floor(0.9, MIN_COORD_SHRINK));
        assert!(!clears_the_magnitude_floor(
            MIN_COORD_SHRINK - 1e-9,
            MIN_COORD_SHRINK
        ));
        assert!(!clears_the_magnitude_floor(-0.5, MIN_COORD_SHRINK));
        assert!(!clears_the_magnitude_floor(f64::NAN, MIN_COORD_SHRINK));

        // Per lever. (That the coords floor is the STRICTER of the two is a
        // compile-time `const` assertion beside the constants, not an assert
        // here — comparing two constants at runtime is folded away and proves
        // nothing.)
        assert_eq!(
            magnitude_floor(&Candidate::QuantizeCoords(Some(1.0))),
            MIN_COORD_SHRINK
        );
        assert_eq!(
            magnitude_floor(&Candidate::QuantizeAttrsAuto(true)),
            MIN_ATTR_SHRINK
        );
        // A lever this advisor does not own can never be floored INTO advice.
        assert!(magnitude_floor(&Candidate::ZstdLevel(19)).is_infinite());
        assert!(!clears_the_magnitude_floor(
            0.99,
            magnitude_floor(&Candidate::ZstdLevel(19))
        ));
    }

    #[test]
    fn the_gate_is_the_conjunction_and_neither_half_alone_can_emit() {
        // ⚠️ THE defect this gate shape exists for. Each half, alone, has a
        // documented failure mode; only the AND has neither.
        //
        //   shrink   stderr    REAL?   BIG?   emitted
        //   0.20%   ±0.01%      yes     no      NO   ← significance-only sold this
        //   40%     ±30%        no      yes     NO   ← magnitude-only sold this
        //   10%     ±0.5%       yes     yes     YES
        let floor = MIN_COORD_SHRINK;
        assert!(!admits(0.002, 0.0001, floor), "real but not worth it");
        assert!(!admits(0.40, 0.30, floor), "big but not real");
        assert!(admits(0.10, 0.005, floor), "real AND worth it");

        // The conjunction is a strict TIGHTENING of each half: anything the
        // gate admits, both halves admit. Nothing can be advised that either
        // half rejects.
        //
        // The magnitude half of this loop is also the FLEET-SAFETY statement.
        // The SHIPPED gate (HEAD) is exactly `shrink >= MIN_COORD_SHRINK`, i.e.
        // `clears_the_magnitude_floor` — so proving `admits ⇒
        // clears_the_magnitude_floor` proves this gate can only ever advise a
        // SUBSET of what the fleet was advised, never a superset. Lossy advice
        // may retreat here; it may never expand.
        for shrink in [-0.2, 0.0, 0.0001, 0.009, 0.02, 0.049, 0.05, 0.3, 0.95] {
            for stderr in [0.0, 1e-9, 1e-4, 0.01, 0.5] {
                let emitted = admits(shrink, stderr, floor);
                assert!(
                    !emitted || clears_the_noise_floor(shrink, stderr),
                    "emitted a shrink its own error bar cannot support: {shrink} ± {stderr}"
                );
                assert!(
                    !emitted || clears_the_magnitude_floor(shrink, floor),
                    "emitted a shrink the SHIPPED magnitude gate rejects — lossy \
                     advice expanded: {shrink} ± {stderr}"
                );
            }
        }
    }

    #[test]
    fn no_sub_one_percent_win_is_ever_advised_however_precisely_it_is_measured() {
        // ⚠️ THE INVARIANT, stated as its own test because it is the property
        // that failed: significance is a claim about the ERROR BAR, and error
        // bars shrink without bound as the sample grows. So the sweep drives
        // the error bar from each shrink — an arbitrarily large sigma is always
        // one big-enough sample away — and NOT ONE of these sub-1% shrinks may
        // be sold as advice at ANY of them.
        //
        // The two live regressions this pins are the 0.98% @ 24σ and 3.00% @
        // 12σ that a significance-only gate emitted; both are inside the sweep.
        for shrink in [0.0001, 0.001, 0.0025, 0.0098, 0.0099] {
            for sigma in [3.0, 10.0, 24.0, 1e3, 1e6] {
                let stderr = shrink / sigma;
                assert!(
                    clears_the_noise_floor(shrink, stderr),
                    "fixture guard: {shrink} ± {stderr} ({sigma}σ) must be significant"
                );
                for floor in [MIN_COORD_SHRINK, MIN_ATTR_SHRINK] {
                    assert!(
                        !admits(shrink, stderr, floor),
                        "a permanent quality loss was advised for a {:.2}% win \
                         ({sigma}σ) — significance is not a magnitude",
                        shrink * 100.0,
                    );
                }
            }
        }
        // The other live regression was 3.00%, above the 1% line but below the
        // coordinate floor. It is refused too, at any sigma.
        for sigma in [12.0, 100.0, 1e6] {
            assert!(!admits(0.0300, 0.0300 / sigma, MIN_COORD_SHRINK));
        }
        // …and the floors themselves keep a real margin over that line — a
        // `const` assertion beside the constants fails the BUILD if either is
        // ever edited down into this band, which a runtime assert here could
        // not do (two constants compare at compile time).
    }

    #[test]
    fn a_real_but_tiny_win_is_never_worth_a_permanent_quality_loss() {
        // ⚠️ SUPPRESSION CASE #1 (the magnitude half doing the work),
        // end-to-end through the encoder rather than argued.
        //
        // Grid coordinates diluted by two high-entropy text columns: the
        // measured coords shrink is ~2.4% and it sits ~43 standard errors from
        // zero, so a significance-only gate emits it with maximum confidence.
        // It is still a permanent loss of coordinate precision bought for 2.4%
        // — and the same sample's LOSSLESS levers are worth several times that
        // — so the advisor stays silent.
        let data = loaded(diluted_grid_sample(800, 40, 2));
        let trial = coords_trial(&data, 14);
        let shrink = -trial.delta_frac;

        assert!(
            shrink > 0.0 && shrink < MIN_COORD_SHRINK,
            "fixture guard: the win must sit UNDER the magnitude floor ({shrink:.4})"
        );
        assert!(
            shrink > 10.0 * trial.stderr,
            "fixture guard: …and be unambiguously REAL, so it is the magnitude \
             half that must refuse it ({shrink:.4} ± {:.5})",
            trial.stderr
        );
        assert!(
            clears_the_noise_floor(shrink, trial.stderr),
            "fixture guard: a significance-only gate would have emitted this"
        );
        assert!(!admits(shrink, trial.stderr, MIN_COORD_SHRINK));

        let result = analysis_result(&data, 14);
        let advice = advise(&result, &data).unwrap();
        assert!(
            !advice.iter().any(|a| a.flag == "--quantize-coords"),
            "a small win is not worth a permanent quality loss: {advice:?}"
        );
    }

    #[test]
    fn a_win_above_the_magnitude_floor_is_suppressed_when_it_is_noise() {
        // ⚠️ SUPPRESSION CASE #2 (the significance half doing the work): a
        // magnitude-only gate SELLS a lossy lever on a 7% number that the
        // measurement cannot tell from zero. The fixture is deliberately
        // heterogeneous — 200 colocated features (where fixed-point coords cost
        // bytes) followed by 80 high-entropy ones (where they save a lot) — so
        // the replicate blocks disagree wildly: 7.1% ± 8.8pp, under one sigma.
        // Recommending a quality loss on that is exactly the failure this
        // module's no-thinning doctrine exists to prevent.
        let data = loaded(dispersed_sample(200, 80));
        let trial = coords_trial(&data, 14);
        let shrink = -trial.delta_frac;

        assert!(
            clears_the_magnitude_floor(shrink, MIN_COORD_SHRINK),
            "fixture guard: a magnitude-only gate must have ACCEPTED this ({shrink:.4})"
        );
        assert!(
            trial.stderr > shrink,
            "fixture guard: …and the noise must swamp it ({shrink:.4} ± {:.4})",
            trial.stderr
        );
        assert!(!clears_the_noise_floor(shrink, trial.stderr));
        assert!(!admits(shrink, trial.stderr, MIN_COORD_SHRINK));

        let result = analysis_result(&data, 14);
        let advice = advise(&result, &data).unwrap();
        assert!(
            !advice.iter().any(|a| a.flag == "--quantize-coords"),
            "a shrink inside its own noise is not evidence: {advice:?}"
        );
    }

    #[test]
    fn a_lever_that_makes_the_archive_bigger_is_never_emitted() {
        // Colocated points: the geometry column is a constant, `Float64` runs
        // compress to nothing, and fixed-point coordinates measure ~2.6%
        // LARGER. A negative shrink can never clear a positive multiple of a
        // positive stderr, so this needs no special case — but it is the one
        // failure mode a "significance" gate could plausibly get backwards, so
        // it is pinned end to end rather than argued.
        let data = loaded(colocated_sample(400));
        let trial = coords_trial(&data, 14);
        assert!(
            -trial.delta_frac < 0.0,
            "fixture guard: quantization must LOSE here ({:.4})",
            -trial.delta_frac
        );
        let result = analysis_result(&data, 14);
        assert!(
            !advise(&result, &data)
                .unwrap()
                .iter()
                .any(|a| a.flag == "--quantize-coords"),
            "a lever that grows the archive must never be recommended"
        );
    }

    #[test]
    fn the_gate_is_the_same_in_round_0_and_in_the_composed_re_price() {
        // The retirement has to land on BOTH gates or the two rounds disagree
        // about admissibility — round 0 emitting what iteration then withdraws
        // for a reason that is not a re-measurement. `reprice` at θ = the build
        // defaults is round 0's measurement by construction, so the two must
        // agree exactly there.
        let data = loaded(diluted_grid_sample(800, 40, 2));
        let result = analysis_result(&data, 14);
        let layout = SyntheticLayout::from_density(&result.density);
        let round0 = advise(&result, &data).unwrap();

        let theta = MeasureSettings::default();
        let mut composer = Composer::new(&data.sample, &layout);
        let (coords, attrs) = reprice(&result, &data, &theta, &mut composer).unwrap();
        let repriced_flags: Vec<&str> = [
            (&coords, "--quantize-coords"),
            (&attrs, "--quantize-attrs-auto"),
        ]
        .iter()
        .filter(|(r, _)| matches!(r, Repriced::Measured(Some(_))))
        .map(|(_, flag)| *flag)
        .collect();
        let round0_flags: Vec<&str> = round0.iter().map(|a| a.flag.as_str()).collect();
        assert_eq!(
            round0_flags, repriced_flags,
            "round 0 and a re-price at the SAME point must admit the same levers"
        );
    }

    #[test]
    fn precision_snaps_down_the_ladder_and_clamps() {
        assert_eq!(snap_down_precision(23.0), 5.0);
        assert_eq!(snap_down_precision(5.0), 5.0);
        assert_eq!(snap_down_precision(1.67), 1.0);
        assert_eq!(snap_down_precision(0.7), 0.5);
        assert_eq!(snap_down_precision(0.09), 0.05);
        assert_eq!(snap_down_precision(0.002), 0.01);
    }
}
