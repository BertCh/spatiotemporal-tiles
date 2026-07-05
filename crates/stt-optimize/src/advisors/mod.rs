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

pub mod budget;
pub mod layout;
pub mod quantize;
pub mod temporal;

use serde::{Deserialize, Serialize};

use crate::analysis::AnalysisResult;
use crate::loader::LoadedData;

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
