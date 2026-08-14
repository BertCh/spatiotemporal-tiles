//! Tileset doctor: severity-ranked findings over a built (packed) tileset,
//! each with a concrete remediation flag, a projection in BYTES, and fused
//! joint what-ifs for the remediations whose projections overlap.
//!
//! Productizes the manual optimization passes this repo keeps re-running by
//! hand (`point_column_stats` → `--quantize-attr`, the id-column hunt, the
//! paged-directory migration, the summary-tier retrofit). Every rule keys off
//! numbers already **measured** for THIS tileset — directory entries from
//! [`PackedTileset`] and per-column compressed costs from [`InspectReport`] —
//! and cites them in its message.
//!
//! # ⚠️ The re-encode doctrine, relaxed
//!
//! This module used to state flatly that *the doctor never re-encodes
//! anything*, and every projection therefore carried the
//! [`ESTIMATE_LABEL`] — including the flagged-column projection, which was
//! built on a FIXED 60% shrink prior ([`RAW_F64_SHRINK`]). A prior is not a
//! measurement: the passes this repo actually ran landed anywhere from 50% to
//! 75%, so the projection was wrong by up to a third in either direction.
//!
//! The doctrine is now narrower and the rule it states is exact:
//!
//! > **Never re-encode TILES BEYOND THE EXISTING SAMPLE.**
//!
//! The measured-shrink pass ([`measure_shrinks`]) re-encodes flagged COLUMNS of
//! the tiles the doctor has **already decoded** — nothing is read that was not
//! read anyway, and the work is hard-capped by [`shrink_trial_budget`] at
//! `DOCTOR_SAMPLE_TILES × flagged columns` column trials, which reduces to "at
//! most [`DOCTOR_SAMPLE_TILES`] layers, ever". That cap is asserted in the
//! tests: it is the thing standing between this relaxation and a full
//! re-encode.
//!
//! Consequently a `projected` string carries one of three labels, and which one
//! it carries is load-bearing:
//!
//! | Label | Meaning |
//! |---|---|
//! | [`MEASURED_LABEL`] | the column really was re-encoded under the remediation, on the sampled tiles |
//! | [`ESTIMATE_LABEL`] | the old prior-based derivation (the fallback when no sample was decoded) |
//! | [`DIRECTORY_LABEL`] | an exact directory sum — no decode, no estimate |
//!
//! # Byte units and ranking
//!
//! Every finding that can name a number publishes [`Finding::projected_bytes`],
//! derived from MO-3's leave-one-out `marginal_bytes` (via the column's `share`)
//! scaled to the whole archive, or read exactly off the directory. Findings sort
//! `(severity, projected_bytes desc, code, message)` — biggest win first inside
//! a tier. ⚠️ The `--strict` CI gate keys off **severity alone**, so re-ranking
//! can never flip CI; `optimize_cli.rs` pins that.
//!
//! # Joint what-ifs
//!
//! Two findings can project the same bytes twice. [`DoctorReport::joint`] fuses
//! the overlapping pairs so a reader is not told to expect the sum:
//!
//! - **`z0-bomb` × `oversized-blobs`** — raising `--min-zoom` deletes whole zoom
//!   rows, and the oversized blobs living in them would otherwise be counted a
//!   second time by the blob rule. Fusion re-projects over the directory with
//!   the min-zoom raise applied FIRST.
//! - **`raw-f64-column` × `expensive-feature-ids`** — priced by ONE combined
//!   column re-encode. The two remediation sets are disjoint by construction
//!   (`id` is in [`RESERVED_COLUMNS`], so no column can be flagged by both);
//!   the disjointness is asserted, not assumed.
//!
//! Rules (stable kebab-case codes):
//! - `raw-f64-column` — plain Float64 property columns worth quantizing.
//! - `expensive-feature-ids` — near-incompressible (hash-like) feature ids.
//! - `dead-columns` — constant / all-null property columns (sampled decode).
//! - `z0-bomb` — a deep shallow pyramid under a tiny geographic extent.
//! - `unpaged-large` — whole-load directory on a large tile count.
//! - `oversized-blobs` — individual tiles past 1 MiB compressed.
//! - `missing-summary-tier` — huge point dataset with no aggregated tier.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    Array, ArrayData, ArrayRef, Float64Array, Int32Builder, RecordBatch, RecordBatchOptions,
    UInt16Builder, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use crate::analysis::inspect::{stratified_sample_indices, ColumnCost, InspectReport};
use crate::attribution::ipc_zstd_len;
use crate::packed::PackedTileset;

/// Label for a projection derived from the inspect report's measured per-column
/// shares plus a PRIOR — no re-encode was performed.
///
/// Retained (rather than retired) because it is the honest label for the
/// fallback path: when the decode pass was skipped (`--sample 0`) there is
/// nothing to re-encode and [`RAW_F64_SHRINK`] is all the doctor has.
const ESTIMATE_LABEL: &str = "(estimated from measured column costs)";

/// Label for a projection whose shrink was **measured** by re-encoding the
/// flagged column, on the tiles the doctor had already decoded, under the
/// remediation it recommends.
const MEASURED_LABEL: &str = "(measured on sampled tiles)";

/// Label for a projection read straight off the directory: an exact sum over
/// entries, with no decode and no extrapolation behind it.
const DIRECTORY_LABEL: &str = "(exact, from the directory)";

/// Must match [`crate::analysis::inspect`]'s `encoding_note` for an
/// unquantized, non-dictionary Float64 column — the smell `raw-f64-column`
/// keys off.
const PLAIN_F64_NOTE: &str = "plain f64 (unquantized)";

/// Core (non-property) tile columns the property rules skip: they are either
/// structural or covered by their own levers (`--quantize-coords` for
/// geometry, the vertex-time delta encoding), not by `--quantize-attr`.
///
/// Keep in step with the encoder's reserved set (`stt_core::arrow_tile`'s
/// `encode_layer`, mirrored in `stt-validate`'s `schema::is_reserved_column`).
/// A core column missing here is silently attributed as a user PROPERTY: its
/// bytes join the property-cost share the `raw-f64-column` rule is computed
/// against, and `dead-columns` can recommend `--exclude`-ing a column the
/// builder owns.
const RESERVED_COLUMNS: [&str; 9] = [
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

/// `raw-f64-column` only flags columns carrying at least this share of the
/// measured column cost — below it the projected win is noise.
///
/// ⚠️ The share this is compared against is now a LEAVE-ONE-OUT share (the
/// column's marginal post-zstd bytes, normalised), not the old singleton
/// proxy's. Marginals are smaller than singleton costs wherever a column
/// shares information with another, so the same 3% threshold is a slightly
/// stricter filter than it used to be — which is the intended direction: it
/// now excludes columns whose bytes another column would pay for anyway.
const RAW_F64_MIN_SHARE: f64 = 0.03;
/// Standard errors the `raw-f64-column` gate SUBTRACTS from a share before
/// comparing it to [`RAW_F64_MIN_SHARE`].
///
/// The asymmetry is deliberate: with as few as [`DOCTOR_SAMPLE_TILES`] decoded
/// tiles the stderr is itself a noisy estimate, so it is only ever spent to
/// SUPPRESS a finding, never to raise one. A tiny archive whose 3.5% share
/// carries a 1% stderr no longer trips the rule; the same share at 0.1% stderr
/// still does. When no dispersion evidence exists (a single decoded tile) the
/// stderr is 0.0 and the gate is exactly the pre-MO-2 comparison.
const RAW_F64_SHARE_SIGMAS: f64 = 2.0;
/// Combined flagged share at which `raw-f64-column` escalates to Critical
/// (the Waymo case: id + z alone were ~78% of the dataset).
const RAW_F64_CRITICAL_SHARE: f64 = 0.5;
/// **Fallback only.** Assumed shrink of a flagged column once quantized
/// (measured passes in this repo landed 50–75%; 60% is the conservative middle).
///
/// ⚠️ This is no longer the primary path: [`measure_shrinks`] re-encodes each
/// flagged column from the already-decoded sample tiles and the finding reports
/// what it MEASURED. The prior survives as the documented rollback for the one
/// case where no measurement exists — `--sample 0` skipped the decode pass — and
/// a projection built on it keeps the old [`ESTIMATE_LABEL`] so a reader can
/// always tell the two apart.
const RAW_F64_SHRINK: f64 = 0.6;

/// zstd level every doctor re-encode is taken at.
///
/// Must equal `inspect`'s private `COLUMN_ZSTD_LEVEL` (19): a measured shrink is
/// only comparable with the `share` it scales if both were compressed at the
/// same level. ⚠️ 19 is PINNED for cross-dataset comparability — the doctor
/// never sweeps it, and level 22 is a standing rejection.
const SHRINK_ZSTD_LEVEL: i32 = 19;

/// `expensive-feature-ids` fires above this measured B/feature…
const ID_BPF_INFO: f64 = 4.0;
/// …and escalates to Warning here (sequential ids land ~1 B/feature; anything
/// past 6 is spending more than a raw u32 per row on identity alone).
const ID_BPF_WARN: f64 = 6.0;
/// `expensive-feature-ids` needs at least this many decoded features per
/// decoded tile: below it, per-tile IPC framing dominates the standalone
/// column re-encode and B/feature reads high even for sequential ids.
const ID_MIN_FEATURES_PER_TILE: f64 = 256.0;

/// Max tiles the doctor's own stride-sampled decode pass reads.
const DOCTOR_SAMPLE_TILES: usize = 8;

/// Hard ceiling on the measured-shrink pass, in **column trials** — one trial is
/// one `(sampled layer × column)` before/after re-encode pair (exactly two
/// compressions).
///
/// This is the bound that keeps the relaxed doctrine honest. The plan's stated
/// ceiling is `DOCTOR_SAMPLE_TILES × flagged columns`; the extra column slot is
/// the COMBINED trial (every flagged column re-encoded together in one batch)
/// that the `raw-f64-column × expensive-feature-ids` joint what-if is priced
/// from, and it only exists when that pair is actually both flagged.
///
/// Because the pass spends the budget a whole layer at a time, the cap reduces
/// to the property that actually matters: **at most [`DOCTOR_SAMPLE_TILES`]
/// layers are ever re-encoded**, no matter how large the archive is. The tests
/// assert both forms.
fn shrink_trial_budget(flagged_columns: usize, combined: bool) -> usize {
    DOCTOR_SAMPLE_TILES.saturating_mul(flagged_columns.saturating_add(usize::from(combined)))
}

/// `z0-bomb` triggers when `min_zoom` is at or below this…
const Z0_MIN_ZOOM: u8 = 4;
/// …while the metadata bounds span less than this many degrees in both axes.
const Z0_EXTENT_DEG: f64 = 2.0;

/// `unpaged-large` fires past this many directory entries.
const UNPAGED_TILE_LIMIT: u64 = 10_000;

/// `oversized-blobs` threshold on a single compressed blob.
const OVERSIZED_BLOB_BYTES: u64 = 1024 * 1024;

/// `missing-summary-tier` floor on the index-weighted feature count.
const SUMMARY_FEATURE_FLOOR: u64 = 1_000_000;

/// Finding severity, most severe first.
///
/// Doctor-local rather than reusing [`crate::analysis::density::IssueSeverity`]:
/// findings need a total order for the severity-first report sort (that enum
/// derives neither `Ord` nor `PartialEq`), and the doctor's top tier is
/// "Critical" (fix before shipping), not a data-loading "Error".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Dominant measured waste — fix before publishing this tileset.
    Critical,
    /// Concrete, measured inefficiency with a known remediation.
    Warning,
    /// Worth knowing; act only if it matches your use case.
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Critical => write!(f, "CRITICAL"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Info => write!(f, "INFO"),
        }
    }
}

/// One doctor finding: what is wrong (with this tileset's measured numbers),
/// how to fix it, and — when derivable from the measured column costs — the
/// projected win.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// Severity tier (report sorts most severe first).
    pub severity: Severity,
    /// Stable kebab-case rule code (e.g. `raw-f64-column`).
    pub code: String,
    /// Human-readable diagnosis citing the tileset's measured numbers.
    pub message: String,
    /// Concrete remediations — builder flags / commands, in preference order.
    pub remediation: Vec<String>,
    /// Projected effect of the remediation in prose, ending in the label that
    /// says where the number came from — [`MEASURED_LABEL`], [`ESTIMATE_LABEL`]
    /// or [`DIRECTORY_LABEL`]. `None` when no meaningful projection exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projected: Option<String>,
    /// The same projection as a number of compressed archive bytes, which is
    /// what the report RANKS on inside a severity tier.
    ///
    /// For column findings this is MO-3's leave-one-out `marginal_bytes` (via
    /// the column's `share`) extrapolated to the whole directory and scaled by
    /// the measured shrink; for the directory rules it is an exact sum over
    /// entries. `None` where no honest byte figure exists — `unpaged-large`
    /// buys latency, not bytes, and `dead-columns`/`missing-summary-tier` are
    /// structural suggestions whose payoff depends on a rebuild's shape.
    ///
    /// Additive field: absent from JSON when `None`, so an older reader is
    /// unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_bytes: Option<u64>,
}

/// A fused projection over several findings whose individual byte projections
/// OVERLAP, so a reader is never told to expect their sum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointWhatIf {
    /// Finding codes this what-if fuses, in the order the remediations apply.
    pub remediation_set: Vec<String>,
    /// Bytes the remediation SET is projected to reclaim, together.
    pub projected_bytes: u64,
    /// Why the fused number is what it is — including the sum it replaces and
    /// the double-count it removes — ending in the provenance label.
    pub note: String,
}

/// Full doctor report: findings sorted severity-first then by projected bytes,
/// plus the fused joint what-ifs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    /// All findings, most severe first (empty = healthy).
    pub findings: Vec<Finding>,
    /// Fused projections for the remediation pairs whose individual
    /// `projected_bytes` overlap. Additive field: omitted from JSON when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub joint: Vec<JointWhatIf>,
}

/// Run every doctor rule over a packed tileset + its inspect report.
///
/// Directory- and column-cost rules read only the (already computed) report;
/// the dead-column and point-dominance checks decode up to
/// [`DOCTOR_SAMPLE_TILES`] stratified-sampled tiles via
/// [`PackedTileset::read_layers`], and the measured-shrink pass re-encodes
/// flagged columns of **those same decoded layers** — never any others (see the
/// module docs' doctrine note and [`shrink_trial_budget`]).
///
/// Findings come back sorted `(severity, projected_bytes desc, code, message)`,
/// so output is deterministic and the biggest measured win leads its tier.
pub fn doctor(tileset: &PackedTileset, report: &InspectReport) -> Result<DoctorReport> {
    let sampled = sample_decode(tileset)?;
    let shrinks = measure_shrinks(report, &sampled)?;

    let mut findings = Vec::new();
    if let Some(f) = rule_raw_f64_columns(report, &shrinks) {
        findings.push(f);
    }
    if let Some(f) = rule_expensive_feature_ids(report, &shrinks) {
        findings.push(f);
    }
    findings.extend(rule_dead_columns(&sampled));
    if let Some(f) = rule_z0_bomb(tileset, report) {
        findings.push(f);
    }
    if let Some(f) = rule_unpaged_large(tileset, report) {
        findings.push(f);
    }
    if let Some(f) = rule_oversized_blobs(tileset, report) {
        findings.push(f);
    }
    if let Some(f) = rule_missing_summary_tier(tileset, report, &sampled) {
        findings.push(f);
    }
    if let Some(f) = rule_vertex_time_precision(report) {
        findings.push(f);
    }

    sort_findings(&mut findings);
    let joint = joint_what_ifs(tileset, report, &findings, &shrinks);
    Ok(DoctorReport { findings, joint })
}

/// The report's deterministic total order: severity first (Critical → Info),
/// then the largest projected byte win, then code, then message.
///
/// ⚠️ Severity is the FIRST key and nothing here can move a finding across a
/// severity boundary, which is why the `--strict` CI gate — which counts
/// Warning-or-worse findings and never looks at order — cannot be flipped by a
/// re-rank. `optimize_cli.rs` pins that at the binary boundary.
///
/// `projected_bytes: None` sorts as 0, i.e. last within its tier: a finding that
/// cannot name a number does not get to outrank one that can.
fn sort_findings(findings: &mut [Finding]) {
    findings.sort_by(|a, b| {
        a.severity
            .cmp(&b.severity)
            .then_with(|| {
                b.projected_bytes
                    .unwrap_or(0)
                    .cmp(&a.projected_bytes.unwrap_or(0))
            })
            .then_with(|| a.code.cmp(&b.code))
            .then_with(|| a.message.cmp(&b.message))
    });
}

// ----------------------------------------------------------------------------
// Sampled decode (shared by dead-columns + missing-summary-tier)
// ----------------------------------------------------------------------------

/// Per-property-column constancy state across the sampled tiles.
struct ColState {
    /// Sampled tiles (with rows) this column appeared in.
    tiles: usize,
    /// Every sampled row was null.
    all_null: bool,
    /// Every sampled row logically equals the first sampled row.
    constant: bool,
    /// 1-row slice of the first sampled value (logical-equality exemplar).
    exemplar: Option<ArrayData>,
}

/// What the doctor's own stride-sampled decode pass observed.
struct SampledDecode {
    /// Tiles actually decoded.
    tiles_decoded: usize,
    /// Total directory entries (for "N of M" wording).
    tiles_total: usize,
    /// Rows summed over sampled layers.
    total_rows: u64,
    /// Rows in layers whose geometry is `geoarrow.point`.
    point_rows: u64,
    /// Per property-column constancy state, keyed by column name.
    columns: BTreeMap<String, ColState>,
    /// The decoded batches themselves, in sample order — retained so
    /// [`measure_shrinks`] can re-encode flagged columns of tiles that were
    /// already read. Bounded by [`DOCTOR_SAMPLE_TILES`] tiles' worth of layers
    /// and dropped with the rest of the pass, so the doctor's peak memory is
    /// unchanged in order of magnitude: these are the same batches the
    /// constancy scan above is already holding one at a time.
    layers: Vec<RecordBatch>,
}

/// Is this decoded layer a point layer? Prefers the `stt:geometry` schema
/// metadata the encoder bakes; falls back to the geometry field's shape
/// (points encode as `FixedSizeList`, lines/polygons as nested lists).
fn is_point_layer(batch: &RecordBatch) -> bool {
    if let Some(kind) = batch.schema().metadata().get("stt:geometry") {
        return kind == "geoarrow.point";
    }
    matches!(
        batch
            .schema()
            .field_with_name("geometry")
            .map(|f| f.data_type().clone()),
        Ok(DataType::FixedSizeList(_, _))
    )
}

/// Decode up to [`DOCTOR_SAMPLE_TILES`] tiles and fold per-column constancy +
/// geometry-kind row counts.
///
/// Tile selection is the SAME deterministic design `inspect` uses
/// ([`stratified_sample_indices`] — byte-proportional `(zoom, time_start)`
/// strata). With only 8 tiles the old flat stride was the most exposed
/// consumer of the aliasing failure: on a curve-ordered directory a stride
/// near the bucket count could hand every one of the doctor's 8 tiles to a
/// single temporal bucket, and `dead-columns` would then call a column
/// constant on the strength of one bucket.
fn sample_decode(tileset: &PackedTileset) -> Result<SampledDecode> {
    let entries = tileset.entries();
    let mut out = SampledDecode {
        tiles_decoded: 0,
        tiles_total: entries.len(),
        total_rows: 0,
        point_rows: 0,
        columns: BTreeMap::new(),
        layers: Vec::new(),
    };
    if entries.is_empty() {
        return Ok(out);
    }
    let plan = stratified_sample_indices(entries, DOCTOR_SAMPLE_TILES);
    for &idx in &plan {
        let e = &entries[idx];
        let layers = tileset.read_layers(e).with_context(|| {
            format!(
                "doctor: decoding tile z{}/{}/{} t{}",
                e.zoom, e.x, e.y, e.time_start
            )
        })?;
        out.tiles_decoded += 1;
        for layer in &layers {
            let batch = &layer.batch;
            let rows = batch.num_rows();
            if rows == 0 {
                continue;
            }
            // Retained for the measured-shrink pass; see [`SampledDecode::layers`].
            out.layers.push(batch.clone());
            out.total_rows += rows as u64;
            if is_point_layer(batch) {
                out.point_rows += rows as u64;
            }
            let schema = batch.schema();
            for (i, field) in schema.fields().iter().enumerate() {
                if RESERVED_COLUMNS.contains(&field.name().as_str()) {
                    continue;
                }
                let arr = batch.column(i);
                let st = out.columns.entry(field.name().clone()).or_insert(ColState {
                    tiles: 0,
                    all_null: true,
                    constant: true,
                    exemplar: None,
                });
                st.tiles += 1;
                if arr.null_count() != arr.len() {
                    st.all_null = false;
                }
                if st.constant {
                    // Logical (offset-aware, dictionary-resolving) equality of
                    // 1-row slices against the first sampled value; bail on the
                    // first mismatch so varying columns cost one comparison.
                    if st.exemplar.is_none() {
                        st.exemplar = Some(arr.slice(0, 1).to_data());
                    }
                    let exemplar = st.exemplar.as_ref().unwrap();
                    for r in 0..rows {
                        if arr.slice(r, 1).to_data() != *exemplar {
                            st.constant = false;
                            break;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Measured shrink: re-encoding flagged columns of the ALREADY-DECODED tiles
// ----------------------------------------------------------------------------

/// The remediation a flagged column is priced under.
///
/// Each variant is the encoding the finding's FIRST remediation string actually
/// recommends, so what is measured is what the user would get.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Remediation {
    /// `--quantize-attrs-auto`: range-adaptive `UInt16` (or the exact `Int32`
    /// integer path). Mirrors `stt_core`'s `build_quantized_numeric_auto`.
    AutoQuantizeAttr,
    /// "rebuild with the current stt-build": builder-assigned sequential ids.
    SequentialIds,
}

/// One column's measured before/after bytes under its remediation, summed over
/// the sampled layers that carry it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ColumnShrink {
    /// IPC + zstd-19 bytes of the column AS SHIPPED.
    before_bytes: u64,
    /// The same bytes with the remediation applied.
    after_bytes: u64,
    /// Sampled layers that contributed a trial.
    layers: usize,
}

impl ColumnShrink {
    /// Measured shrink fraction in `[0, 1]`, or `None` when no trial ran.
    ///
    /// Clamped at zero rather than reported negative: a remediation that made a
    /// column *larger* on this sample has a measured win of nothing, and a
    /// negative "saving" would sort a finding above the ones that do save.
    fn fraction(&self) -> Option<f64> {
        if self.layers == 0 || self.before_bytes == 0 {
            return None;
        }
        let saved = self.before_bytes.saturating_sub(self.after_bytes) as f64;
        Some((saved / self.before_bytes as f64).clamp(0.0, 1.0))
    }
}

/// What the measured-shrink pass produced, plus the accounting that proves it
/// stayed inside [`shrink_trial_budget`].
#[derive(Debug, Clone, Default)]
struct MeasuredShrinks {
    /// Per flagged column, keyed by name.
    per_column: BTreeMap<String, ColumnShrink>,
    /// Every flagged column re-encoded TOGETHER in one batch — the combined
    /// trial the `raw-f64-column × expensive-feature-ids` joint what-if is
    /// priced from. `None` unless that pair was both flagged.
    combined: Option<ColumnShrink>,
    /// Column trials performed. One trial = one `(layer × column)` before/after
    /// pair.
    trials: usize,
    /// Compressions issued — exactly `2 × trials`, by construction.
    compressions: usize,
    /// Sampled layers the pass re-encoded. Never more than
    /// [`DOCTOR_SAMPLE_TILES`]: the doctrine bound.
    layers_measured: usize,
}

impl MeasuredShrinks {
    /// This column's measured shrink fraction, if it was measured at all.
    fn fraction_for(&self, name: &str) -> Option<f64> {
        self.per_column.get(name).and_then(ColumnShrink::fraction)
    }
}

/// Re-encode every flagged column of the tiles the doctor already decoded, under
/// the remediation its finding recommends.
///
/// # What is and is not read
///
/// Nothing here touches the archive: it works purely from
/// [`SampledDecode::layers`], which the constancy scan already decoded. The
/// budget is spent one whole layer at a time, so the pass re-encodes **at most
/// [`DOCTOR_SAMPLE_TILES`] layers** regardless of archive size, and it walks
/// them in the sample's order — never a subset chosen per column, which would
/// make two columns' numbers incomparable. `ColumnShrink::layers` records how
/// many layers each column actually appeared in, so an intermittent column is
/// visible rather than silently averaged over a different denominator.
///
/// # Fallback
///
/// Returns an empty result — costing zero compressions — when
/// `report.decode.tiles_decoded == 0`. That is the `--sample 0` directory-only
/// fast mode: the user declined to pay for a decode, so the doctor does not
/// spend one, and every projection falls back to [`RAW_F64_SHRINK`] under the
/// old [`ESTIMATE_LABEL`]. Same outcome when nothing is flagged, when a layer
/// does not carry the column, or when the remediation is inapplicable (a
/// non-`Float64` array, or a magnitude `stt_core` itself refuses to quantize).
fn measure_shrinks(report: &InspectReport, sampled: &SampledDecode) -> Result<MeasuredShrinks> {
    let mut out = MeasuredShrinks::default();
    if report.decode.tiles_decoded == 0 || sampled.layers.is_empty() {
        return Ok(out);
    }

    let raw: Vec<String> = flagged_raw_f64(report)
        .iter()
        .map(|c| c.name.clone())
        .collect();
    let ids: Option<String> = flagged_expensive_ids(report).map(|c| c.name.clone());

    // DISJOINTNESS (asserted, not assumed). `id` lives in RESERVED_COLUMNS,
    // which `flagged_raw_f64` filters out, so no column can be flagged by both
    // rules and the combined trial is additive over two disjoint sets. If a
    // future edit ever removed `id` from the reserved set this would silently
    // double-charge that column, so the guard is a hard one.
    let disjoint = ids
        .as_ref()
        .is_none_or(|id| !raw.iter().any(|name| name == id));
    debug_assert!(
        disjoint,
        "raw-f64 and expensive-ids remediation sets must be disjoint by column"
    );

    let mut columns: Vec<(String, Remediation)> = raw
        .iter()
        .map(|name| (name.clone(), Remediation::AutoQuantizeAttr))
        .collect();
    if let Some(id) = &ids {
        columns.push((id.clone(), Remediation::SequentialIds));
    }
    if columns.is_empty() {
        return Ok(out);
    }
    // The combined trial exists only for the recorded pair, and only when the
    // sets really are disjoint.
    let combined = !raw.is_empty() && ids.is_some() && disjoint;

    let per_layer = columns.len() + usize::from(combined);
    let budget = shrink_trial_budget(columns.len(), combined);
    let max_layers = budget / per_layer.max(1);

    let mut combined_acc = ColumnShrink::default();
    for batch in sampled.layers.iter().take(max_layers) {
        let rows = batch.num_rows();
        if rows == 0 {
            continue;
        }
        let schema = batch.schema();
        let mut measured_here = false;
        // Every column that could be remediated in THIS layer, so the combined
        // trial prices exactly the union of what the individual trials priced.
        let mut both: Vec<(String, ArrayRef, ArrayRef)> = Vec::with_capacity(columns.len());
        for (name, remediation) in &columns {
            let Ok(index) = schema.index_of(name) else {
                continue;
            };
            let original = batch.column(index).clone();
            let Some(after) = remediate(&original, *remediation, rows) else {
                continue;
            };
            let before_bytes = ipc_zstd_len(
                &column_batch(&[(name.as_str(), original.clone())], rows)?,
                SHRINK_ZSTD_LEVEL,
            )?;
            let after_bytes = ipc_zstd_len(
                &column_batch(&[(name.as_str(), after.clone())], rows)?,
                SHRINK_ZSTD_LEVEL,
            )?;
            let acc = out.per_column.entry(name.clone()).or_default();
            acc.before_bytes += before_bytes;
            acc.after_bytes += after_bytes;
            acc.layers += 1;
            out.trials += 1;
            out.compressions += 2;
            measured_here = true;
            both.push((name.clone(), original, after));
        }

        // ONE combined re-encode: all flagged columns in a single batch, both
        // ways. Measuring them together rather than adding the singles is what
        // lets zstd see (or not see) whatever the columns share.
        if combined && both.len() == columns.len() {
            let originals: Vec<(&str, ArrayRef)> = both
                .iter()
                .map(|(n, o, _)| (n.as_str(), o.clone()))
                .collect();
            let remediated: Vec<(&str, ArrayRef)> = both
                .iter()
                .map(|(n, _, a)| (n.as_str(), a.clone()))
                .collect();
            combined_acc.before_bytes +=
                ipc_zstd_len(&column_batch(&originals, rows)?, SHRINK_ZSTD_LEVEL)?;
            combined_acc.after_bytes +=
                ipc_zstd_len(&column_batch(&remediated, rows)?, SHRINK_ZSTD_LEVEL)?;
            combined_acc.layers += 1;
            out.trials += 1;
            out.compressions += 2;
        }

        if measured_here {
            out.layers_measured += 1;
        }
    }

    debug_assert!(
        out.trials <= budget,
        "measured-shrink pass spent {} trials against a budget of {budget}",
        out.trials
    );
    if combined_acc.layers > 0 {
        out.combined = Some(combined_acc);
    }
    Ok(out)
}

/// Apply `remediation` to a decoded column, or `None` when it does not apply.
fn remediate(array: &ArrayRef, remediation: Remediation, rows: usize) -> Option<ArrayRef> {
    match remediation {
        Remediation::AutoQuantizeAttr => auto_quantized(array),
        Remediation::SequentialIds => sequential_ids(array, rows),
    }
}

/// Range-adaptive automatic quantization of a decoded `Float64` column.
///
/// A deliberate MIRROR of `stt_core::arrow_tile::quantize`'s
/// `build_quantized_numeric_auto` (which is `pub(crate)` there, so it cannot be
/// called): the magnitude refusal, the exact integer path, the degenerate
/// constant case and the range-adaptive `UInt16` fall-through, in that order.
/// The encoder stays authoritative — this is a MEASUREMENT-side copy whose only
/// job is to produce the representation the recommendation would produce, so the
/// bytes are the bytes the user would actually get. `None` means "the encoder
/// would decline", and the column then falls back to the prior.
fn auto_quantized(array: &ArrayRef) -> Option<ArrayRef> {
    let values = array.as_any().downcast_ref::<Float64Array>()?;

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut max_abs = 0.0f64;
    let mut finite = 0usize;
    let mut all_integer = true;
    for i in 0..values.len() {
        if values.is_null(i) {
            continue;
        }
        let v = values.value(i);
        if !v.is_finite() {
            continue;
        }
        finite += 1;
        min = min.min(v);
        max = max.max(v);
        max_abs = max_abs.max(v.abs());
        all_integer &= v == v.trunc();
    }

    /// `stt_core`'s `AUTO_QUANT_MAX_ABS` — the sole refusal, mirrored.
    const AUTO_QUANT_MAX_ABS: f64 = i32::MAX as f64;
    const U16_MAX_INDEX: f64 = u16::MAX as f64;
    const I32_MAX_INDEX: f64 = i32::MAX as f64;

    // (4) Nothing finite: the historical all-null UInt16 at {o: 0, s: 1}.
    if finite == 0 {
        return Some(quantize_u16(values, 0.0, 1.0));
    }
    // (1) The magnitude refusal — the column stays Float64.
    if max_abs >= AUTO_QUANT_MAX_ABS {
        return None;
    }
    let span = max - min;
    // (2) Exact fixed point at step 1, narrowest leaf that indexes the span.
    if all_integer && span <= I32_MAX_INDEX {
        return Some(if span <= U16_MAX_INDEX {
            quantize_u16(values, min, 1.0)
        } else {
            quantize_i32(values, min, 1.0)
        });
    }
    // Degenerate span: step 1 maps every present value to index 0. Also the
    // guard against a span so small that `span / 65535` underflows to zero.
    // Written as "is the step usable" rather than as two negated comparisons so
    // the NaN case (which every comparison answers `false`) lands here too.
    let s = span / U16_MAX_INDEX;
    let usable = span.is_finite() && span > 0.0 && s.is_finite() && s > 0.0;
    if !usable {
        return Some(quantize_u16(values, min, 1.0));
    }
    // (3) Range-adaptive UInt16.
    Some(quantize_u16(values, min, s))
}

/// `UInt16` fixed-point leaf at the affine `(o, s)`; nulls and non-finite cells
/// become Arrow nulls, indices are clamped into the leaf.
fn quantize_u16(values: &Float64Array, o: f64, s: f64) -> ArrayRef {
    let mut b = UInt16Builder::with_capacity(values.len());
    for i in 0..values.len() {
        match (!values.is_null(i)).then(|| values.value(i)) {
            Some(v) if v.is_finite() => {
                b.append_value(((v - o) / s).round().clamp(0.0, u16::MAX as f64) as u16)
            }
            _ => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

/// `Int32` fixed-point leaf at the affine `(o, s)` — the exact-integer widening.
fn quantize_i32(values: &Float64Array, o: f64, s: f64) -> ArrayRef {
    let mut b = Int32Builder::with_capacity(values.len());
    for i in 0..values.len() {
        match (!values.is_null(i)).then(|| values.value(i)) {
            Some(v) if v.is_finite() => {
                b.append_value(((v - o) / s).round().clamp(0.0, i32::MAX as f64) as i32)
            }
            _ => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

/// Builder-assigned sequential feature ids — the `expensive-feature-ids`
/// remediation, priced.
///
/// Per-tile `0..rows` stands in for the dataset-global renumbering the rebuild
/// would perform: what a tile actually receives under sequential ids is a
/// contiguous RUN, and post-zstd a run is a run wherever it starts, so the
/// per-tile form measures the same thing at the same cost.
///
/// `None` unless the column really is the encoder's `UInt64` id column: pricing
/// a renumbering of something else would be measuring a different question.
fn sequential_ids(array: &ArrayRef, rows: usize) -> Option<ArrayRef> {
    if array.data_type() != &DataType::UInt64 {
        return None;
    }
    Some(Arc::new(UInt64Array::from_iter_values(0..rows as u64)))
}

/// A `RecordBatch` of exactly the named columns, with ALL Arrow metadata
/// stripped.
///
/// Stripping mirrors [`crate::attribution`]'s projection and is load-bearing
/// twice: Arrow serialises field/schema metadata `HashMap`s in nondeterministic
/// order (so keeping it would make two doctor runs disagree by a few bytes), and
/// it makes the before and after arms differ ONLY by the representation being
/// priced. The explicit row count covers a column whose array length cannot
/// carry it.
fn column_batch(columns: &[(&str, ArrayRef)], rows: usize) -> Result<RecordBatch> {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let arrays: Vec<ArrayRef> = columns.iter().map(|(_, a)| a.clone()).collect();
    let options = RecordBatchOptions::new().with_row_count(Some(rows));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), arrays, &options)
        .context("doctor: trial batch build failed")
}

/// Scale a per-column share of the measured column cost up to whole-archive
/// compressed bytes.
///
/// This is the density extrapolation the byte projections ride: `share` is
/// MO-3's leave-one-out `marginal_bytes` normalised over the decoded tiles, so
/// `share × compressed_bytes` is that marginal carried to the full directory.
/// Clamped into `[0, compressed_bytes]` — a projection can never promise more
/// bytes than the archive has.
fn projected_u64(bytes: f64, cap: u64) -> u64 {
    if !bytes.is_finite() || bytes <= 0.0 {
        return 0;
    }
    bytes.round().min(cap as f64) as u64
}

// ----------------------------------------------------------------------------
// Rules
// ----------------------------------------------------------------------------

/// The property columns `raw-f64-column` flags, worst share first.
///
/// Shared by the rule and the measured-shrink pass so the columns that get
/// re-encoded are exactly the columns that get reported — a mismatch would show
/// up as a projection built half on measurement and half on the prior.
///
/// The noise-aware gate is what stops small archives firing on a share that is
/// mostly decode dispersion; see [`RAW_F64_SHARE_SIGMAS`] for why the stderr is
/// only ever subtracted. Geometry is deliberately out of scope (its lever is
/// `--quantize-coords`), as is every other [`RESERVED_COLUMNS`] entry.
fn flagged_raw_f64(report: &InspectReport) -> Vec<&ColumnCost> {
    let mut flagged: Vec<&ColumnCost> = report
        .per_column
        .iter()
        .filter(|c| {
            !RESERVED_COLUMNS.contains(&c.name.as_str())
                && c.encoding_note == PLAIN_F64_NOTE
                && (c.share - RAW_F64_SHARE_SIGMAS * c.share_stderr) >= RAW_F64_MIN_SHARE
        })
        .collect();
    // Name breaks the tie explicitly: leave-one-out marginals produce far more
    // exact ties than the old singleton proxy did, and the order feeds both the
    // message and the re-encode plan.
    flagged.sort_by(|a, b| {
        b.share
            .total_cmp(&a.share)
            .then_with(|| a.name.cmp(&b.name))
    });
    flagged
}

/// The `id` column `expensive-feature-ids` flags, if any.
///
/// Shared with the measured-shrink pass for the same reason as
/// [`flagged_raw_f64`].
fn flagged_expensive_ids(report: &InspectReport) -> Option<&ColumnCost> {
    let id = report.per_column.iter().find(|c| c.name == "id")?;
    if id.bytes_per_feature <= ID_BPF_INFO {
        return None;
    }
    // Sparse tilesets (few features per tile) inflate B/feature with per-tile
    // IPC framing — no id-cost signal survives that noise floor.
    if report.decode.tiles_decoded > 0
        && (report.decode.features_decoded as f64 / report.decode.tiles_decoded as f64)
            < ID_MIN_FEATURES_PER_TILE
    {
        return None;
    }
    Some(id)
}

/// `raw-f64-column`: property columns shipping as plain Float64 (not
/// quantized, not dictionary) whose measured share clears 3% *after* two
/// standard errors are subtracted. One finding listing every such column,
/// worst first.
///
/// The projection is **measured** when every flagged column was re-encoded on
/// the sampled tiles (the normal case): the reported per-column shrink is what
/// the re-encode actually produced, not the [`RAW_F64_SHRINK`] prior. If any
/// flagged column has no measurement — `--sample 0`, or a magnitude the encoder
/// refuses to quantize — the WHOLE projection falls back to the prior and keeps
/// the old [`ESTIMATE_LABEL`], rather than mixing two provenances inside one
/// sentence.
fn rule_raw_f64_columns(report: &InspectReport, shrinks: &MeasuredShrinks) -> Option<Finding> {
    let flagged = flagged_raw_f64(report);
    if flagged.is_empty() {
        return None;
    }

    let total_share: f64 = flagged.iter().map(|c| c.share).sum();
    let listed = flagged
        .iter()
        .map(|c| {
            // The ± is only rendered when dispersion was actually measurable
            // (two or more decoded tiles); a fabricated "±0.0%" would read as a
            // precision claim the decode cannot support.
            let spread = if c.share_stderr > 0.0 {
                format!(" ±{:.1}", 100.0 * c.share_stderr)
            } else {
                String::new()
            };
            format!(
                "`{}` ({:.1}{}% of column bytes, {:.2} B/feature)",
                c.name,
                100.0 * c.share,
                spread,
                c.bytes_per_feature
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    // `collect::<Option<Vec<_>>>` is the all-or-nothing gate: one unmeasured
    // column and the whole projection reverts to the prior, so a sentence never
    // mixes a measurement with a guess.
    let measured: Option<Vec<f64>> = flagged
        .iter()
        .map(|c| shrinks.fraction_for(&c.name))
        .collect();
    // Σ share_j · shrink_j — the share of the archive's compressed bytes the
    // remediation is projected to remove.
    let saved_share: f64 = match &measured {
        Some(fracs) => flagged
            .iter()
            .zip(fracs)
            .map(|(c, f)| c.share * f)
            .sum::<f64>(),
        None => total_share * RAW_F64_SHRINK,
    };
    let saved_bytes = report.compressed_bytes as f64 * saved_share;
    // The byte-weighted mean per-column shrink, i.e. the number that used to be
    // the flat 0.6 prior.
    let mean_shrink = if total_share > 0.0 {
        saved_share / total_share
    } else {
        0.0
    };
    let projected = Some(match &measured {
        Some(_) => format!(
            "~{:.0}% smaller dataset wire (~{:.2} of {:.2} MB) after quantizing the flagged \
             columns, at a measured ~{:.0}% per-column shrink {}",
            100.0 * saved_share,
            saved_bytes / 1e6,
            report.compressed_bytes as f64 / 1e6,
            100.0 * mean_shrink,
            MEASURED_LABEL
        ),
        None => format!(
            "~{:.0}% smaller dataset wire (~{:.2} of {:.2} MB) after quantizing the flagged \
             columns, assuming ~{:.0}% per-column shrink {}",
            100.0 * saved_share,
            saved_bytes / 1e6,
            report.compressed_bytes as f64 / 1e6,
            100.0 * RAW_F64_SHRINK,
            ESTIMATE_LABEL
        ),
    });
    Some(Finding {
        severity: if total_share >= RAW_F64_CRITICAL_SHARE {
            Severity::Critical
        } else {
            Severity::Warning
        },
        code: "raw-f64-column".to_string(),
        message: format!(
            "{} property column(s) ship as raw Float64 and together cost {:.1}% of this \
             tileset's measured column bytes: {}. Raw f64 attributes are near-incompressible; \
             fixed-point ints are both smaller and far more compressible.",
            flagged.len(),
            100.0 * total_share,
            listed
        ),
        remediation: vec![
            format!(
                "--quantize-attr <name>=<prec> (per column, e.g. --quantize-attr {}=0.01)",
                flagged[0].name
            ),
            "--quantize-attrs-auto (range-adaptive u16 for every remaining raw Float64 property)"
                .to_string(),
        ],
        projected,
        projected_bytes: Some(projected_u64(saved_bytes, report.compressed_bytes)),
    })
}

/// `expensive-feature-ids`: the `id` column measured above 4 B/feature —
/// hash-like or explicit source ids carry full entropy per row, so zstd
/// cannot shrink them, while builder-assigned sequential ids compress to
/// ~1 B/feature.
///
/// The projection is measured whenever the sampled tiles allowed the id column
/// to be renumbered and re-encoded; without that it degrades to the old upper
/// bound ("up to the column's whole share"), which is exactly what it always
/// was.
fn rule_expensive_feature_ids(
    report: &InspectReport,
    shrinks: &MeasuredShrinks,
) -> Option<Finding> {
    let id = flagged_expensive_ids(report)?;
    let (projected, saved_bytes) = match shrinks.fraction_for(&id.name) {
        Some(frac) => {
            let bytes = report.compressed_bytes as f64 * id.share * frac;
            (
                format!(
                    "~{:.1}% of dataset wire (~{:.2} MB) reclaimable from the id column — \
                     sequential ids re-encoded {:.0}% smaller on the sampled tiles {}",
                    100.0 * id.share * frac,
                    bytes / 1e6,
                    100.0 * frac,
                    MEASURED_LABEL
                ),
                bytes,
            )
        }
        None => (
            format!(
                "up to ~{:.1}% of dataset wire reclaimable from the id column {}",
                100.0 * id.share,
                ESTIMATE_LABEL
            ),
            report.compressed_bytes as f64 * id.share,
        ),
    };
    Some(Finding {
        severity: if id.bytes_per_feature >= ID_BPF_WARN {
            Severity::Warning
        } else {
            Severity::Info
        },
        code: "expensive-feature-ids".to_string(),
        message: format!(
            "feature-id column costs {:.2} B/feature ({:.1}% of measured column bytes) — \
             hash-like or explicit source ids are near-incompressible (full entropy per row); \
             builder-assigned sequential ids compress to ~1 B/feature.",
            id.bytes_per_feature,
            100.0 * id.share
        ),
        remediation: vec![
            "rebuild with the current stt-build — anonymous point features get sequential ids \
             automatically"
                .to_string(),
            "if explicit source ids are load-bearing (picking, cross-dataset joins), reconsider \
             whether they must ship in tiles or a sequential remap would do"
                .to_string(),
        ],
        projected: Some(projected),
        projected_bytes: Some(projected_u64(saved_bytes, report.compressed_bytes)),
    })
}

/// `dead-columns`: property columns constant or all-null across every sampled
/// tile they appear in (and appearing in more than one sampled tile). Sampled
/// evidence only — the message says so.
fn rule_dead_columns(sampled: &SampledDecode) -> Vec<Finding> {
    sampled
        .columns
        .iter()
        .filter(|(_, st)| st.tiles > 1 && (st.constant || st.all_null))
        .map(|(name, st)| {
            let what = if st.all_null {
                "entirely null"
            } else {
                "a single constant value"
            };
            Finding {
                severity: Severity::Info,
                code: "dead-columns".to_string(),
                message: format!(
                    "property column `{name}` is {what} across all {} sampled tiles that carry \
                     it ({} of {} tiles decoded — sampled, not proven; verify before excluding). \
                     A constant column ships no information a renderer can use.",
                    st.tiles, sampled.tiles_decoded, sampled.tiles_total
                ),
                remediation: vec![format!("--exclude {name}")],
                projected: None,
                // A dead column's leave-one-out marginal is ~0 by definition —
                // a constant column costs almost nothing post-zstd — so there
                // is no honest byte figure to rank on. The win is semantic
                // (a renderer stops being offered a column it cannot use).
                projected_bytes: None,
            }
        })
        .collect()
}

/// The zoom floor `z0-bomb` would recommend, or `None` when the rule does not
/// fire.
///
/// Split out because the `z0-bomb × oversized-blobs` joint what-if has to apply
/// the same raise BEFORE re-running the blob rule; a second, drifting copy of
/// this arithmetic would silently make the fused number wrong.
fn z0_floor(tileset: &PackedTileset, report: &InspectReport) -> Option<u8> {
    let bounds = &tileset.metadata().bounds;
    let lon_ext = bounds.max_lon - bounds.min_lon;
    let lat_ext = bounds.max_lat - bounds.min_lat;
    if report.min_zoom > Z0_MIN_ZOOM || lon_ext >= Z0_EXTENT_DEG || lat_ext >= Z0_EXTENT_DEG {
        return None;
    }
    // The zoom where the bounds extent covers ~2–8 tiles: one axis of the
    // world is 360° wide, a tile at z spans 360/2^z degrees of longitude.
    let max_ext = lon_ext.max(lat_ext).max(1e-9);
    let raw = (360.0 / max_ext).log2().ceil() as i64;
    let hi = i64::from(report.max_zoom)
        .saturating_sub(1)
        .max(i64::from(Z0_MIN_ZOOM));
    let floor = raw.clamp(i64::from(Z0_MIN_ZOOM), hi) as u8;
    (floor > report.min_zoom).then_some(floor)
}

/// Directory-exact `(entries, blob bytes)` strictly below `floor`.
fn shallow_pyramid(report: &InspectReport, floor: u8) -> (u64, u64) {
    report
        .per_zoom
        .iter()
        .filter(|z| z.zoom < floor)
        .fold((0u64, 0u64), |(t, b), z| {
            (t + z.entries, b + z.blob_bytes_total)
        })
}

/// `z0-bomb`: a shallow pyramid (min_zoom ≤ 4) under bounds spanning less
/// than 2° in both axes — every zoom below the suggested floor re-ships the
/// whole dataset in one or two near-duplicate tiles.
fn rule_z0_bomb(tileset: &PackedTileset, report: &InspectReport) -> Option<Finding> {
    let bounds = &tileset.metadata().bounds;
    let lon_ext = bounds.max_lon - bounds.min_lon;
    let lat_ext = bounds.max_lat - bounds.min_lat;
    let floor = z0_floor(tileset, report)?;
    let (shallow_tiles, shallow_bytes) = shallow_pyramid(report, floor);
    Some(Finding {
        severity: Severity::Warning,
        code: "z0-bomb".to_string(),
        message: format!(
            "min_zoom is {} but the metadata bounds span only {:.2}° × {:.2}° — the shallow \
             pyramid below z{} holds {} tile entries ({:.2} MB) that mostly re-ship the whole \
             dataset; z{} already covers these bounds with ~2-8 tiles.",
            report.min_zoom,
            lon_ext,
            lat_ext,
            floor,
            shallow_tiles,
            shallow_bytes as f64 / 1e6,
            floor
        ),
        remediation: vec![format!("--min-zoom {floor}")],
        // Exact, not extrapolated: these entries and their compressed lengths
        // are in the directory, and raising the floor stops emitting all of
        // them. No feature is lost — z{floor} still covers the bounds.
        projected: Some(format!(
            "~{:.2} MB of shallow-pyramid blob bytes stop being emitted ({} entries below z{}) {}",
            shallow_bytes as f64 / 1e6,
            shallow_tiles,
            floor,
            DIRECTORY_LABEL
        )),
        projected_bytes: Some(shallow_bytes.min(report.compressed_bytes)),
    })
}

/// `unpaged-large`: a single whole-load directory past 10k entries — every
/// cold reader downloads the entire directory before its first tile fetch.
fn rule_unpaged_large(tileset: &PackedTileset, report: &InspectReport) -> Option<Finding> {
    if tileset.is_paged() || report.tile_count <= UNPAGED_TILE_LIMIT {
        return None;
    }
    Some(Finding {
        severity: Severity::Warning,
        code: "unpaged-large".to_string(),
        message: format!(
            "single whole-load directory with {} entries — a cold reader must download the \
             full directory before its first tile fetch; the paged container fetches only the \
             leaf pages a viewport/time-window touches.",
            report.tile_count
        ),
        remediation: vec![
            "rebuild with the current stt-build — the paged directory is the default (avoid \
             --single-directory)"
                .to_string(),
            "generated datasets: re-run the source generator (stt-generate builds paged + \
             publish-tuned straight from source)"
                .to_string(),
        ],
        projected: None,
        // Paging buys COLD-START LATENCY, not wire bytes (the directory is the
        // same size either way — it is fetched in leaf pages instead of whole).
        // Inventing a byte number here would rank a latency fix against byte
        // fixes on a scale it does not live on.
        projected_bytes: None,
    })
}

/// Bytes past the 1 MiB per-tile budget, summed over the entries at or above
/// `min_zoom` — directory-exact.
///
/// `min_zoom = 0` is the whole directory (what the rule itself reports); the
/// joint what-if passes the raised floor so the entries a `--min-zoom` raise
/// already deleted are not charged twice.
fn oversized_excess(tileset: &PackedTileset, min_zoom: u8) -> (u64, u64) {
    tileset
        .entries()
        .iter()
        .filter(|e| e.zoom >= min_zoom && u64::from(e.length) > OVERSIZED_BLOB_BYTES)
        .fold((0u64, 0u64), |(n, bytes), e| {
            (n + 1, bytes + u64::from(e.length) - OVERSIZED_BLOB_BYTES)
        })
}

/// `oversized-blobs`: directory entries whose compressed blob exceeds 1 MiB.
/// Structural fixes (zoom floor, summary tier) come first; the opt-in budgets
/// drop data and are never presented as the default fix.
///
/// The byte projection is the EXCESS over the budget, not the blobs' whole size:
/// what the finding asks for is that no single tile exceed 1 MiB, and the
/// structural remediations achieve it by re-cutting rather than by dropping. It
/// is directory-exact, so no measurement rides on it.
fn rule_oversized_blobs(tileset: &PackedTileset, report: &InspectReport) -> Option<Finding> {
    let over: Vec<_> = tileset
        .entries()
        .iter()
        .filter(|e| u64::from(e.length) > OVERSIZED_BLOB_BYTES)
        .collect();
    let worst = over.iter().max_by_key(|e| e.length)?;
    let (_, excess) = oversized_excess(tileset, 0);
    let avg = report.compressed_bytes as f64 / report.tile_count.max(1) as f64;
    Some(Finding {
        severity: Severity::Warning,
        code: "oversized-blobs".to_string(),
        message: format!(
            "{} of {} directory entries exceed 1 MiB compressed; worst is z{}/{}/{} t{} at \
             {:.2} MiB (dataset average {:.1} KB) — oversized tiles stall first paint on slow \
             links.",
            over.len(),
            report.tile_count,
            worst.zoom,
            worst.x,
            worst.y,
            worst.time_start,
            f64::from(worst.length) / OVERSIZED_BLOB_BYTES as f64,
            avg / 1e3
        ),
        remediation: vec![
            "raise --min-zoom so the densest shallow tiles are never emitted".to_string(),
            "--summary-tier quadbin (serve a pre-aggregated tier at low zooms instead of raw \
             features)"
                .to_string(),
            "opt-in last resort: --maximum-tile-bytes / --maximum-tile-features — WARNING: \
             these DROP features from over-budget tiles (STT never thins by default)"
                .to_string(),
        ],
        projected: Some(format!(
            "~{:.2} MB sit past the 1 MiB per-tile budget across {} entries {}",
            excess as f64 / 1e6,
            over.len(),
            DIRECTORY_LABEL
        )),
        projected_bytes: Some(excess.min(report.compressed_bytes)),
    })
}

/// `missing-summary-tier`: >1M features, point-dominant sampled payloads, and
/// no pre-aggregated summary tier in the metadata — low zooms re-ship raw
/// points a renderer can only overplot.
fn rule_missing_summary_tier(
    tileset: &PackedTileset,
    report: &InspectReport,
    sampled: &SampledDecode,
) -> Option<Finding> {
    if tileset.metadata().summary_tier.is_some()
        || report.feature_count <= SUMMARY_FEATURE_FLOOR
        || sampled.total_rows == 0
        || sampled.point_rows * 2 < sampled.total_rows
    {
        return None;
    }
    Some(Finding {
        severity: Severity::Info,
        code: "missing-summary-tier".to_string(),
        message: format!(
            "no summary tier on {} (index-weighted) features with a point-dominant payload \
             ({:.0}% of {} sampled rows over {} tiles) — low zooms re-ship raw points a \
             renderer can only overplot; a pre-aggregated tier reads at output resolution \
             instead of N.",
            report.feature_count,
            100.0 * sampled.point_rows as f64 / sampled.total_rows as f64,
            sampled.total_rows,
            sampled.tiles_decoded
        ),
        remediation: vec![
            "--summary-tier quadbin --summary-columns <name:agg,...> (pre-aggregated low-zoom \
             tier; `count` is always emitted)"
                .to_string(),
        ],
        projected: None,
        // A summary tier ADDS bytes and removes read amplification; what it
        // saves depends on the aggregation columns and the zoom split a rebuild
        // chooses, none of which is measurable from the built archive. The
        // §12.4 "added_tier_bytes" model that would price it belongs to MO-8.
        projected_bytes: None,
    })
}

// ----------------------------------------------------------------------------
// Joint what-ifs: fusing overlapping projections
// ----------------------------------------------------------------------------

/// Fuse every recorded pair of findings whose `projected_bytes` overlap.
///
/// Returned in a fixed construction order (directory pair, then column pair), so
/// the vector is deterministic without a sort.
fn joint_what_ifs(
    tileset: &PackedTileset,
    report: &InspectReport,
    findings: &[Finding],
    shrinks: &MeasuredShrinks,
) -> Vec<JointWhatIf> {
    let mut out = Vec::new();
    if let Some(j) = joint_zoom_floor_and_blobs(tileset, report, findings) {
        out.push(j);
    }
    if let Some(j) = joint_columns(report, findings, shrinks) {
        out.push(j);
    }
    out
}

/// The finding with this code, if the report raised it.
fn finding_with<'a>(findings: &'a [Finding], code: &str) -> Option<&'a Finding> {
    findings.iter().find(|f| f.code == code)
}

/// `z0-bomb × oversized-blobs`, fused.
///
/// The double-count §12.5 records: an oversized blob that lives at a zoom BELOW
/// the recommended floor is charged once by `z0-bomb` (whose projection is the
/// whole shallow pyramid's bytes) and once again by `oversized-blobs` (as its
/// excess over 1 MiB). Fusion is one re-projection over the directory with the
/// min-zoom raise applied FIRST — so the blob rule only ever sees the entries
/// that survive it.
fn joint_zoom_floor_and_blobs(
    tileset: &PackedTileset,
    report: &InspectReport,
    findings: &[Finding],
) -> Option<JointWhatIf> {
    let z0 = finding_with(findings, "z0-bomb")?;
    let blobs = finding_with(findings, "oversized-blobs")?;
    let floor = z0_floor(tileset, report)?;

    let (shallow_tiles, shallow_bytes) = shallow_pyramid(report, floor);
    // The blob rule, re-run over the post-raise directory.
    let (surviving, surviving_excess) = oversized_excess(tileset, floor);
    let fused = (shallow_bytes + surviving_excess).min(report.compressed_bytes);

    let sum = z0.projected_bytes.unwrap_or(0) + blobs.projected_bytes.unwrap_or(0);
    let double_counted = sum.saturating_sub(fused);
    Some(JointWhatIf {
        remediation_set: vec!["z0-bomb".to_string(), "oversized-blobs".to_string()],
        projected_bytes: fused,
        note: format!(
            "raising --min-zoom to z{floor} deletes {shallow_tiles} shallow entries \
             ({:.2} MB) before the 1 MiB blob rule runs, leaving {surviving} oversized \
             entries carrying {:.2} MB of excess — {:.2} MB together, not the {:.2} MB the \
             two projections add up to ({:.2} MB of double-count removed) {}",
            shallow_bytes as f64 / 1e6,
            surviving_excess as f64 / 1e6,
            fused as f64 / 1e6,
            sum as f64 / 1e6,
            double_counted as f64 / 1e6,
            DIRECTORY_LABEL
        ),
    })
}

/// `raw-f64-column × expensive-feature-ids`, fused by ONE combined re-encode.
///
/// The two remediation sets are DISJOINT by column — `id` is reserved, so no
/// column can be flagged by both — which is what makes the combined projection
/// legitimate rather than a double-charge. The disjointness is re-asserted here
/// (it is also asserted where the trial is planned) and the combined trial is
/// measured on the same sampled layers as the individual ones, so the three
/// numbers are directly comparable.
fn joint_columns(
    report: &InspectReport,
    findings: &[Finding],
    shrinks: &MeasuredShrinks,
) -> Option<JointWhatIf> {
    let raw_finding = finding_with(findings, "raw-f64-column")?;
    let id_finding = finding_with(findings, "expensive-feature-ids")?;
    let combined = shrinks.combined.as_ref()?.fraction()?;

    let raw_columns = flagged_raw_f64(report);
    let id = flagged_expensive_ids(report)?;
    // Hard disjointness guard: withdraw the joint what-if rather than publish a
    // number that charges one column to two remediations.
    if raw_columns.iter().any(|c| c.name == id.name) {
        debug_assert!(false, "raw-f64 and expensive-ids flagged the same column");
        return None;
    }

    let share: f64 = raw_columns.iter().map(|c| c.share).sum::<f64>() + id.share;
    let fused = projected_u64(
        report.compressed_bytes as f64 * share * combined,
        report.compressed_bytes,
    );
    let sum = raw_finding.projected_bytes.unwrap_or(0) + id_finding.projected_bytes.unwrap_or(0);
    Some(JointWhatIf {
        remediation_set: vec![
            "raw-f64-column".to_string(),
            "expensive-feature-ids".to_string(),
        ],
        projected_bytes: fused,
        note: format!(
            "quantizing the {} flagged Float64 column(s) and renumbering `{}` touch DISJOINT \
             columns, so one combined re-encode prices both: {:.0}% off {:.1}% of the measured \
             column bytes = ~{:.2} MB, against ~{:.2} MB if the two projections were simply \
             added {}",
            raw_columns.len(),
            id.name,
            100.0 * combined,
            100.0 * share,
            fused as f64 / 1e6,
            sum as f64 / 1e6,
            MEASURED_LABEL
        ),
    })
}

// ----------------------------------------------------------------------------
// Text rendering
// ----------------------------------------------------------------------------

/// Render the report as compact aligned text (severity-ranked, one block per
/// finding, then the fused joint what-ifs).
pub fn format_text(report: &DoctorReport) -> String {
    let mut out = String::new();
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    out.push_str("         STT Doctor\n");
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    if report.findings.is_empty() {
        out.push_str("No findings — this tileset passes every doctor rule.\n");
        out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        return out;
    }

    let count = |s: Severity| report.findings.iter().filter(|f| f.severity == s).count();
    out.push_str(&format!(
        "{} finding(s): {} critical, {} warning, {} info\n\n",
        report.findings.len(),
        count(Severity::Critical),
        count(Severity::Warning),
        count(Severity::Info)
    ));

    for f in &report.findings {
        out.push_str(&format!("[{}] {}\n", f.severity, f.code));
        out.push_str(&format!("  {}\n", f.message));
        for r in &f.remediation {
            out.push_str(&format!("  fix: {r}\n"));
        }
        if let Some(p) = &f.projected {
            out.push_str(&format!("  projected: {p}\n"));
        }
        out.push('\n');
    }

    if !report.joint.is_empty() {
        out.push_str(
            "Joint what-ifs (overlapping remediations, fused so they stop double-counting):\n",
        );
        for j in &report.joint {
            out.push_str(&format!(
                "  [{}] ~{:.2} MB\n",
                j.remediation_set.join(" + "),
                j.projected_bytes as f64 / 1e6
            ));
            out.push_str(&format!("      {}\n", j.note));
        }
        out.push('\n');
    }

    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::inspect::inspect;
    use stt_core::arrow_tile::{
        encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn, PropertyColumn,
    };
    use stt_core::curve::BlobOrdering;
    use stt_core::metadata::Metadata;
    use stt_core::pack::PackWriter;
    use stt_core::tile::TileId;
    use stt_core::types::BoundingBox;

    /// splitmix64: deterministic full-entropy values (so "random" columns are
    /// genuinely incompressible without any RNG dependency).
    fn mix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Deterministic uniform f64 in `[0, 1)`.
    fn rand01(x: u64) -> f64 {
        (mix(x) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Everything a fixture varies; defaults describe a small healthy dataset.
    struct FixtureSpec {
        rows: usize,
        tiles: usize,
        /// Hash-like (full-entropy) feature ids instead of sequential.
        hash_ids: bool,
        /// Add a `stuck` property that is the same value in every row/tile.
        stuck_column: bool,
        /// Encode `magnitude` fixed-point (`--quantize-attr magnitude=0.01`).
        quantize_magnitude: bool,
        /// Full-entropy world-spread coordinates (incompressible geometry).
        world_random_coords: bool,
        /// Reuse one payload for every entry (fast large-directory fixtures).
        identical_payloads: bool,
        /// Leaf-page size; `None` = single whole-load directory.
        paged: Option<usize>,
        /// Per-tile zoom override (cycled); `None` = all tiles at `max_zoom`.
        zooms: Option<Vec<u8>>,
        min_zoom: u8,
        max_zoom: u8,
        /// Metadata bounds; `None` keeps the whole-world default.
        bounds: Option<BoundingBox>,
        /// Claimed per-entry feature count; `None` = the real row count.
        claimed_features: Option<u32>,
    }

    impl Default for FixtureSpec {
        fn default() -> Self {
            Self {
                rows: 2000,
                tiles: 2,
                hash_ids: false,
                stuck_column: false,
                quantize_magnitude: false,
                world_random_coords: false,
                identical_payloads: false,
                paged: None,
                zooms: None,
                min_zoom: 5,
                max_zoom: 10,
                bounds: None,
                claimed_features: None,
            }
        }
    }

    /// Point layer with a full-entropy `magnitude` f64, an alternating
    /// `kind` categorical, and optionally a constant `stuck` column.
    fn points_layer(seed: u64, spec: &FixtureSpec) -> ColumnarLayer {
        let n = spec.rows;
        let feature_ids: Vec<u64> = if spec.hash_ids {
            (0..n)
                .map(|i| mix(seed.wrapping_mul(1_000_003).wrapping_add(i as u64)))
                .collect()
        } else {
            (0..n as u64).map(|i| seed * 1_000_000 + i).collect()
        };
        let geometry: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                if spec.world_random_coords {
                    [
                        -180.0 + 360.0 * rand01(seed ^ (i as u64 * 2 + 1)),
                        -85.0 + 170.0 * rand01(seed ^ (i as u64 * 2 + 2)),
                    ]
                } else {
                    [
                        -73.9 + (i % 50) as f64 * 0.001,
                        45.4 + (i / 50) as f64 * 0.001,
                    ]
                }
            })
            .collect();
        let mut properties = vec![
            (
                "magnitude".to_string(),
                PropertyColumn::Numeric(
                    (0..n)
                        .map(|i| Some(rand01(seed * 31 + i as u64) * 10.0))
                        .collect(),
                ),
            ),
            (
                "kind".to_string(),
                PropertyColumn::Categorical(
                    (0..n)
                        .map(|i| Some(["bike", "ferry"][i % 2].to_string()))
                        .collect(),
                ),
            ),
        ];
        if spec.stuck_column {
            properties.push((
                "stuck".to_string(),
                PropertyColumn::Numeric(vec![Some(42.0); n]),
            ));
        }
        ColumnarLayer {
            polygon_parts: None,
            name: "default".to_string(),
            feature_ids,
            start_times: vec![0; n],
            end_times: vec![100; n],
            geometry: GeometryColumn::Point(geometry),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties,
        }
    }

    /// Build a real packed dataset per the spec (PackWriter + encode_tile_with,
    /// the exact pattern of the packed.rs/inspect.rs test fixtures). Frames
    /// ride the writer's (default v2) format version + template collector so
    /// the fixture is version-coherent — what a default `stt-build` emits.
    fn build(out: &std::path::Path, spec: &FixtureSpec) {
        let mut w = PackWriter::create(out, BlobOrdering::Auto, 64 * 1024)
            .unwrap()
            .with_paging(spec.paged);
        let cfg = EncoderConfig {
            quantize_attrs: if spec.quantize_magnitude {
                [("magnitude".to_string(), 0.01)].into_iter().collect()
            } else {
                Default::default()
            },
            format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
            template_collector: Some(w.template_collector()),
            ..Default::default()
        };
        let bucket = 3_600_000i64;
        let shared = spec
            .identical_payloads
            .then(|| encode_tile_with(&[points_layer(0, spec)], &cfg).unwrap());
        for k in 0..spec.tiles {
            let payload = match &shared {
                Some(p) => p.clone(),
                None => encode_tile_with(&[points_layer(k as u64, spec)], &cfg).unwrap(),
            };
            let z = spec
                .zooms
                .as_ref()
                .map(|zs| zs[k % zs.len()])
                .unwrap_or(spec.max_zoom);
            let x = (k as u32) % (1u32 << z);
            let t0 = (k as i64) * bucket;
            w.add_tile_full(
                &TileId::new(z, x, 0, t0 as u64),
                t0,
                t0 + bucket - 1,
                Some(t0),
                spec.claimed_features.unwrap_or(spec.rows as u32),
                Some(bucket as u64),
                &payload,
            )
            .unwrap();
        }
        let mut meta = Metadata::new("doctor-fixture")
            .with_temporal_bucket_ms(bucket as u64)
            .with_zoom_levels(spec.min_zoom, spec.max_zoom);
        if let Some(b) = spec.bounds {
            meta = meta.with_bounds(b);
        }
        w.finalize(&meta).unwrap();
    }

    /// Build + inspect + doctor in one go.
    fn doctor_fixture(spec: &FixtureSpec, sample: Option<usize>) -> DoctorReport {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build(&out, spec);
        let ts = PackedTileset::open(&out).unwrap();
        let report = inspect(&ts, sample).unwrap();
        doctor(&ts, &report).unwrap()
    }

    fn find<'a>(report: &'a DoctorReport, code: &str) -> Option<&'a Finding> {
        report.findings.iter().find(|f| f.code == code)
    }

    /// The report's findings are in the MO-7 total order
    /// `(severity, projected_bytes desc, code, message)`.
    fn assert_findings_sorted(report: &DoctorReport) {
        let key = |f: &Finding| {
            (
                f.severity,
                std::cmp::Reverse(f.projected_bytes.unwrap_or(0)),
                f.code.clone(),
                f.message.clone(),
            )
        };
        let ranks: Vec<_> = report.findings.iter().map(key).collect();
        let mut sorted = ranks.clone();
        sorted.sort();
        assert_eq!(ranks, sorted, "findings not in the MO-7 total order");
    }

    /// R1 fires on a plain-f64 property tileset (worst-first message, labeled
    /// projection, the two quantize flags) and does NOT fire once the same
    /// data ships quantized.
    #[test]
    fn raw_f64_column_fires_then_quantized_is_clean() {
        let raw = doctor_fixture(&FixtureSpec::default(), None);
        let f = find(&raw, "raw-f64-column").expect("raw-f64-column should fire");
        assert!(
            matches!(f.severity, Severity::Critical | Severity::Warning),
            "severity {:?}",
            f.severity
        );
        assert!(f.message.contains("magnitude"), "message: {}", f.message);
        assert!(f.message.contains('%'), "message cites measured shares");
        assert!(f.remediation.iter().any(|r| r.contains("--quantize-attr ")));
        assert!(f
            .remediation
            .iter()
            .any(|r| r.contains("--quantize-attrs-auto")));
        let projected = f.projected.as_deref().expect("R1 carries a projection");
        // MO-7: this fixture's decode DOES produce a measurement, so the
        // projection must be labeled as measured — not as the old prior.
        assert!(
            projected.contains(MEASURED_LABEL),
            "projection must carry the measured label: {projected}"
        );
        assert!(
            !projected.contains("assuming ~60%"),
            "the 0.6 prior must not appear once the shrink was measured: {projected}"
        );
        assert!(f.projected_bytes.unwrap_or(0) > 0, "R1 ranks on bytes");

        // Same data, quantized: R1 gone, and the whole fixture is clean at
        // Warning+ (Info findings are allowed).
        let clean = doctor_fixture(
            &FixtureSpec {
                quantize_magnitude: true,
                ..Default::default()
            },
            None,
        );
        assert!(find(&clean, "raw-f64-column").is_none());
        let warnings: Vec<_> = clean
            .findings
            .iter()
            .filter(|f| f.severity <= Severity::Warning)
            .collect();
        assert!(
            warnings.is_empty(),
            "clean quantized fixture has Warning+ findings: {warnings:?}"
        );
    }

    /// R2 fires on hash-like (full-entropy) feature ids: >4 B/feature can't
    /// happen for the builder's sequential ids.
    #[test]
    fn expensive_feature_ids_fires_on_hash_ids() {
        let report = doctor_fixture(
            &FixtureSpec {
                hash_ids: true,
                ..Default::default()
            },
            None,
        );
        let f = find(&report, "expensive-feature-ids").expect("should fire");
        // 8 random bytes/row cannot compress below 8 B/feature — Warning tier.
        assert_eq!(f.severity, Severity::Warning);
        assert!(f.message.contains("B/feature"), "message: {}", f.message);
        assert!(f.remediation.iter().any(|r| r.contains("sequential ids")));
        // MO-7: the id column really is renumbered and re-encoded on the
        // sampled tiles, so the projection is measured, not bounded.
        let projected = f.projected.as_deref().unwrap();
        assert!(
            projected.contains(MEASURED_LABEL),
            "projection must carry the measured label: {projected}"
        );
        assert!(f.projected_bytes.unwrap_or(0) > 0);
    }

    /// R3 fires for a constant property column, names it, says the evidence
    /// is sampled, and suggests `--exclude`.
    #[test]
    fn dead_columns_fires_for_constant_column() {
        let report = doctor_fixture(
            &FixtureSpec {
                stuck_column: true,
                ..Default::default()
            },
            None,
        );
        let f = find(&report, "dead-columns").expect("dead-columns should fire");
        assert_eq!(f.severity, Severity::Info);
        assert!(f.message.contains("`stuck`"), "message: {}", f.message);
        assert!(f.message.contains("sampled"), "must admit sampling");
        assert_eq!(f.remediation, vec!["--exclude stuck".to_string()]);
        // The varying columns must NOT be reported dead.
        assert!(
            !report.findings.iter().any(|f| f.code == "dead-columns"
                && (f.message.contains("`magnitude`") || f.message.contains("`kind`"))),
            "varying columns flagged dead"
        );
    }

    /// R4 fires for min_zoom 0 + tiny bounds, computes the ~2-8-tile zoom
    /// floor, and cites the shallow-pyramid entry count.
    #[test]
    fn z0_bomb_fires_for_tiny_bounds_deep_pyramid() {
        let report = doctor_fixture(
            &FixtureSpec {
                tiles: 4,
                rows: 50,
                zooms: Some(vec![0, 1, 2, 10]),
                min_zoom: 0,
                max_zoom: 10,
                bounds: Some(BoundingBox {
                    min_lon: -73.8,
                    min_lat: 45.4,
                    max_lon: -73.3,
                    max_lat: 45.9,
                }),
                ..Default::default()
            },
            None,
        );
        let f = find(&report, "z0-bomb").expect("z0-bomb should fire");
        assert_eq!(f.severity, Severity::Warning);
        // 0.5° extent → ceil(log2(360/0.5)) = 10, clamped to max_zoom-1 = 9.
        assert_eq!(f.remediation, vec!["--min-zoom 9".to_string()]);
        // Three fixture tiles sit below the suggested floor (z0, z1, z2).
        assert!(
            f.message.contains("3 tile entries"),
            "message: {}",
            f.message
        );

        // Wide bounds: same pyramid, no finding.
        let wide = doctor_fixture(
            &FixtureSpec {
                tiles: 4,
                rows: 50,
                zooms: Some(vec![0, 1, 2, 10]),
                min_zoom: 0,
                max_zoom: 10,
                bounds: None,
                ..Default::default()
            },
            None,
        );
        assert!(find(&wide, "z0-bomb").is_none());
    }

    /// R5 fires for a >10k-entry single whole-load directory and stays quiet
    /// once the same dataset ships paged.
    #[test]
    fn unpaged_large_fires_then_paged_is_quiet() {
        let spec = FixtureSpec {
            rows: 4,
            tiles: 10_001,
            identical_payloads: true,
            zooms: Some(vec![14]),
            min_zoom: 5,
            max_zoom: 14,
            ..Default::default()
        };
        let report = doctor_fixture(&spec, Some(4));
        let f = find(&report, "unpaged-large").expect("unpaged-large should fire");
        assert_eq!(f.severity, Severity::Warning);
        assert!(f.message.contains("10001"), "message: {}", f.message);
        assert!(f
            .remediation
            .iter()
            .any(|r| r.contains("paged directory is the default")));

        let paged = doctor_fixture(
            &FixtureSpec {
                paged: Some(4096),
                ..spec
            },
            Some(4),
        );
        assert!(find(&paged, "unpaged-large").is_none());
    }

    /// R6 fires with one artificially large (incompressible) blob and cites
    /// its address; budgets are presented as opt-in with a drop warning.
    #[test]
    fn oversized_blobs_fires_and_orders_remediation() {
        let report = doctor_fixture(
            &FixtureSpec {
                rows: 100_000,
                tiles: 1,
                world_random_coords: true,
                ..Default::default()
            },
            None,
        );
        let f = find(&report, "oversized-blobs").expect("oversized-blobs should fire");
        assert_eq!(f.severity, Severity::Warning);
        assert!(f.message.contains("MiB"), "message: {}", f.message);
        assert!(f.message.contains("z10/0/0"), "message: {}", f.message);
        // Structural fixes first, data-dropping budgets last + flagged.
        assert!(f.remediation[0].contains("--min-zoom"));
        assert!(f.remediation[1].contains("--summary-tier"));
        assert!(f.remediation[2].contains("--maximum-tile-bytes"));
        assert!(f.remediation[2].contains("DROP"));
        // MO-7: the excess over the 1 MiB budget is a directory-exact number.
        let projected = f.projected.as_deref().expect("R6 now carries a projection");
        assert!(projected.contains(DIRECTORY_LABEL), "{projected}");
        assert!(f.projected_bytes.unwrap_or(0) > 0);

        // Findings come out in the MO-7 total order: severity, then largest
        // projected bytes, then code, then message.
        assert_findings_sorted(&report);
    }

    /// R7 fires for >1M (index-weighted) point features without a summary
    /// tier.
    #[test]
    fn missing_summary_tier_fires_for_large_point_dataset() {
        let report = doctor_fixture(
            &FixtureSpec {
                rows: 100,
                tiles: 2,
                claimed_features: Some(600_000),
                quantize_magnitude: true,
                ..Default::default()
            },
            None,
        );
        let f = find(&report, "missing-summary-tier").expect("should fire");
        assert_eq!(f.severity, Severity::Info);
        assert!(f.message.contains("1200000"), "message: {}", f.message);
        assert!(f.remediation[0].contains("--summary-tier quadbin"));
    }

    /// Serde round-trip (projected is absent, not null-filled, when None) and
    /// the text rendering.
    #[test]
    fn report_serializes_and_renders() {
        let report = doctor_fixture(
            &FixtureSpec {
                stuck_column: true,
                ..Default::default()
            },
            None,
        );
        let json = serde_json::to_string_pretty(&report).unwrap();
        let back: DoctorReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.findings.len(), report.findings.len());
        // dead-columns has no projection → the key is skipped entirely.
        assert!(!json.contains("\"projected\": null"));
        assert!(json.contains("\"severity\": \"info\""));

        let text = format_text(&report);
        assert!(text.contains("STT Doctor"));
        // Severity tag matches the finding (Critical when the flagged share
        // dominates this little fixture, Warning otherwise).
        let raw = find(&report, "raw-f64-column").unwrap();
        assert!(text.contains(&format!("[{}] raw-f64-column", raw.severity)));
        assert!(text.contains("[INFO] dead-columns"));
        assert!(text.contains("fix: --exclude stuck"));
        assert!(text.contains("projected:"));

        let empty = format_text(&DoctorReport {
            findings: vec![],
            joint: vec![],
        });
        assert!(empty.contains("No findings"));
        // `joint` is additive: an empty vector is omitted from JSON entirely,
        // so an older reader sees exactly the shape it saw before.
        assert!(!json.contains("\"joint\""), "empty joint must be skipped");
    }

    // ------------------------------------------------------------------
    // MO-2: the raw-f64 gate is noise-aware
    // ------------------------------------------------------------------

    /// A minimal inspect report carrying one plain-f64 property column with the
    /// given share and dispersion — everything else the rule reads is inert.
    fn report_with_share(share: f64, share_stderr: f64) -> InspectReport {
        use crate::analysis::inspect::{ColumnCost, DecodeStats, DedupStats};
        InspectReport {
            name: "noise-fixture".to_string(),
            min_zoom: 5,
            max_zoom: 10,
            time_start_ms: 0,
            time_end_ms: 3_600_000,
            temporal_bucket_ms: 3_600_000,
            tile_count: 8,
            feature_count: 800,
            pack_count: 1,
            paged_directory: true,
            compressed_bytes: 1_000_000,
            uncompressed_bytes: 3_000_000,
            compression_ratio: 3.0,
            per_zoom: Vec::new(),
            dedup: DedupStats {
                entries: 8,
                distinct_blobs: 8,
                dedup_ratio: 1.0,
            },
            decode: DecodeStats {
                tiles_decoded: 8,
                tiles_total: 8,
                sampled: true,
                features_decoded: 800,
                distinct_layer_schemas: 1,
                design: crate::analysis::inspect::SampleDesign::default()
                    .as_str()
                    .to_string(),
                attribution: crate::measure::AttributionDesign::default()
                    .as_str()
                    .to_string(),
            },
            per_column: vec![ColumnCost {
                name: "magnitude".to_string(),
                dtype: "Float64".to_string(),
                compressed_bytes: (share * 100_000.0) as u64,
                share,
                bytes_per_feature: 2.0,
                marginal_bytes: (share * 100_000.0) as u64,
                share_stderr,
                encoding_note: PLAIN_F64_NOTE.to_string(),
            }],
        }
    }

    /// A report whose `vertex_time` column sits on `note`, over an archive
    /// spanning `duration_ms`.
    fn vertex_time_report(note: &str, duration_ms: u64, share: f64) -> InspectReport {
        use crate::analysis::inspect::ColumnCost;
        let mut r = report_with_share(0.5, 0.0);
        r.time_start_ms = 0;
        r.time_end_ms = duration_ms;
        r.per_column = vec![ColumnCost {
            name: "vertex_time".to_string(),
            dtype: "List(UInt32)".to_string(),
            compressed_bytes: 400_000,
            share,
            bytes_per_feature: 4.0,
            marginal_bytes: 400_000,
            share_stderr: 0.0,
            encoding_note: note.to_string(),
        }];
        r
    }

    /// TB-11 extension 3 — the ceiling advisory speaks only when PLAYBACK
    /// cannot resolve the precision the archive is paying for.
    #[test]
    fn the_vertex_time_advisory_is_gated_on_the_playback_budget() {
        // A 30-day archive: one frame of a 45 s / 30 fps loop advances ~2400 s,
        // vastly past the 1000 ms default ceiling → advise.
        let wide = vertex_time_report("u32 vertex-time deltas", 30 * 86_400_000, 0.30);
        let f = rule_vertex_time_precision(&wide).expect("a 30-day archive must advise");
        assert_eq!(f.code, "vertex-time-precision");
        assert!(
            f.remediation[0].contains("--vertex-time-precision"),
            "{:?}",
            f.remediation
        );
        // LOSSY, and it says so — this raises a quantization ceiling.
        assert!(f.remediation[0].contains("LOSSY"), "{:?}", f.remediation);

        // A one-hour archive: one frame advances ~2.7 s... which IS past the
        // ceiling, so the gate must be the SLACK factor, not merely "past".
        // Half an hour advances ~1.3 s, inside 4x the ceiling → silent.
        let short = vertex_time_report("u32 vertex-time deltas", 1_800_000, 0.30);
        assert!(
            rule_vertex_time_precision(&short).is_none(),
            "a short archive's playback resolves the default ceiling; do not advise"
        );

        // Already on the narrowest tier: nothing to win, whatever the span.
        let narrow = vertex_time_report("u16 vertex-time deltas", 30 * 86_400_000, 0.30);
        assert!(rule_vertex_time_precision(&narrow).is_none());

        // Too small a share to pay for a rebuild.
        let cheap = vertex_time_report("u32 vertex-time deltas", 30 * 86_400_000, 0.01);
        assert!(rule_vertex_time_precision(&cheap).is_none());

        // No vertex_time column at all (the common points-only archive).
        let plain = report_with_share(0.5, 0.0);
        assert!(rule_vertex_time_precision(&plain).is_none());
    }

    /// The exact-i64 fallback is the costliest case and projects the biggest win.
    #[test]
    fn the_vertex_time_advisory_projects_more_for_the_i64_fallback() {
        let span = 30 * 86_400_000;
        let u32_win =
            rule_vertex_time_precision(&vertex_time_report("u32 vertex-time deltas", span, 0.30))
                .unwrap()
                .projected_bytes
                .unwrap();
        let i64_win =
            rule_vertex_time_precision(&vertex_time_report("i64 absolute vertex-time", span, 0.30))
                .unwrap()
                .projected_bytes
                .unwrap();
        assert!(
            i64_win > u32_win,
            "i64 -> u16 saves more than u32 -> u16 ({i64_win} vs {u32_win})"
        );
    }

    #[test]
    fn raw_f64_gate_subtracts_two_stderr_before_comparing() {
        // §12.5 gap 3: a 3.5% share is over the 3% floor, but on a tiny archive
        // it can be almost entirely decode noise.
        let none = MeasuredShrinks::default();
        let noisy = rule_raw_f64_columns(&report_with_share(0.035, 0.01), &none);
        assert!(
            noisy.is_none(),
            "3.5% share with 1% stderr is 1.5% at 2σ — must not fire: {noisy:?}"
        );
        let tight = rule_raw_f64_columns(&report_with_share(0.035, 0.001), &none)
            .expect("3.5% share with 0.1% stderr is real evidence and must fire");
        assert!(tight.message.contains("magnitude"));
        // The measured spread is cited beside the share.
        assert!(
            tight.message.contains("±0.1%"),
            "message must publish the dispersion: {}",
            tight.message
        );

        // Exactly at the boundary the rule still fires (>=, as before).
        assert!(rule_raw_f64_columns(&report_with_share(0.05, 0.01), &none).is_some());
        // A hair below it does not.
        assert!(rule_raw_f64_columns(&report_with_share(0.0499, 0.01), &none).is_none());

        // No dispersion evidence (a single decoded tile) ⇒ stderr 0.0 ⇒ the
        // pre-MO-2 comparison, verbatim. The stderr is NEVER added.
        assert!(rule_raw_f64_columns(&report_with_share(0.031, 0.0), &none).is_some());
        assert!(rule_raw_f64_columns(&report_with_share(0.029, 0.0), &none).is_none());
        let quiet = rule_raw_f64_columns(&report_with_share(0.031, 0.0), &none).unwrap();
        assert!(
            !quiet.message.contains('±'),
            "an unmeasurable spread must not be rendered: {}",
            quiet.message
        );
    }

    #[test]
    fn raw_f64_gate_on_a_real_fixture_still_fires_with_published_dispersion() {
        // End to end: the fixture's shares now come from leave-one-out
        // marginals and carry a stderr, and the rule still finds the
        // full-entropy `magnitude` column.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build(&out, &FixtureSpec::default());
        let ts = PackedTileset::open(&out).unwrap();
        let report = inspect(&ts, None).unwrap();

        assert_eq!(report.decode.attribution, "loo-v2");
        let magnitude = report
            .per_column
            .iter()
            .find(|c| c.name == "magnitude")
            .expect("magnitude column");
        assert_eq!(
            magnitude.marginal_bytes, magnitude.compressed_bytes,
            "loo-v2 charges the marginal"
        );
        assert!(magnitude.share_stderr.is_finite());
        assert!(
            magnitude.share - 2.0 * magnitude.share_stderr >= RAW_F64_MIN_SHARE,
            "share {} ± {} must survive the 2σ subtraction",
            magnitude.share,
            magnitude.share_stderr
        );
        assert!(doctor(&ts, &report)
            .unwrap()
            .findings
            .iter()
            .any(|f| f.code == "raw-f64-column"));
    }

    // ------------------------------------------------------------------
    // MO-7: measured shrink, byte units, joint what-ifs
    // ------------------------------------------------------------------

    /// Build + inspect + run the measured-shrink pass, returning everything the
    /// MO-7 tests need to reason about one fixture.
    fn shrink_fixture(
        spec: &FixtureSpec,
        sample: Option<usize>,
    ) -> (InspectReport, MeasuredShrinks, DoctorReport) {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build(&out, spec);
        let ts = PackedTileset::open(&out).unwrap();
        let report = inspect(&ts, sample).unwrap();
        let sampled = sample_decode(&ts).unwrap();
        let shrinks = measure_shrinks(&report, &sampled).unwrap();
        let doc = doctor(&ts, &report).unwrap();
        (report, shrinks, doc)
    }

    /// **Test 1 — the measurement replaces the prior.**
    ///
    /// `magnitude` is a full-entropy `Float64` column: post-zstd it costs close
    /// to its raw 8 B/value, and its range-adaptive `UInt16` form costs close to
    /// 2, so the REAL shrink is far above the fixed 0.6 prior. The rule must
    /// report what it measured.
    #[test]
    fn measured_shrink_replaces_the_sixty_percent_prior() {
        let (report, shrinks, doc) = shrink_fixture(&FixtureSpec::default(), None);

        let measured = shrinks
            .fraction_for("magnitude")
            .expect("magnitude must be re-encoded on the sampled tiles");
        assert!(
            measured > RAW_F64_SHRINK,
            "the measured shrink ({measured:.3}) must differ from the {RAW_F64_SHRINK} prior — \
             a full-entropy f64 column really does shed more than 60% at u16"
        );
        assert!(
            (0.60..0.95).contains(&measured),
            "measured shrink {measured:.3} outside the plausible f64→u16 band"
        );

        let f = find(&doc, "raw-f64-column").expect("R1 fires");
        let projected = f.projected.as_deref().unwrap();
        assert!(projected.contains(MEASURED_LABEL), "{projected}");
        assert!(projected.contains("at a measured ~"), "{projected}");
        // The reported per-column shrink IS the measured one, to the rendered
        // precision — not 60%.
        assert!(
            projected.contains(&format!("~{:.0}% per-column shrink", 100.0 * measured)),
            "projection must quote the measured shrink ({:.0}%): {projected}",
            100.0 * measured
        );

        // …and the byte figure is that shrink applied to the extrapolated
        // marginal, i.e. `compressed_bytes × share × shrink`.
        let magnitude = report
            .per_column
            .iter()
            .find(|c| c.name == "magnitude")
            .unwrap();
        let expected = projected_u64(
            report.compressed_bytes as f64 * magnitude.share * measured,
            report.compressed_bytes,
        );
        assert_eq!(f.projected_bytes, Some(expected));
    }

    /// **Test 2 — fallback to the prior, with the OLD label.**
    ///
    /// Two halves, because `--sample 0` reaches the fallback by two different
    /// routes: the pass performs no work at all, and the rule (asked directly,
    /// with no measurement in hand) reverts to [`RAW_F64_SHRINK`] verbatim.
    #[test]
    fn unmeasured_columns_fall_back_to_the_prior_with_the_old_label() {
        // (a) `--sample 0` = the directory-only fast mode. The doctor must not
        // spend the decode the user declined: zero trials, zero compressions.
        let (report, shrinks, _doc) = shrink_fixture(&FixtureSpec::default(), Some(0));
        assert_eq!(report.decode.tiles_decoded, 0);
        assert!(report.per_column.is_empty());
        assert_eq!(shrinks.trials, 0, "--sample 0 must re-encode nothing");
        assert_eq!(shrinks.compressions, 0);
        assert_eq!(shrinks.layers_measured, 0);
        assert!(shrinks.combined.is_none());

        // (b) The rule with no measurement in hand: the 0.6 prior, the old
        // label, and the old wording — the documented rollback, verbatim.
        let prior =
            rule_raw_f64_columns(&report_with_share(0.20, 0.0), &MeasuredShrinks::default())
                .expect("a 20% share fires");
        let projected = prior.projected.as_deref().unwrap();
        assert!(
            projected.contains(ESTIMATE_LABEL),
            "the prior-based projection keeps the OLD label: {projected}"
        );
        assert!(!projected.contains(MEASURED_LABEL), "{projected}");
        assert!(
            projected.contains("assuming ~60% per-column shrink"),
            "{projected}"
        );
        // 1_000_000 bytes × 20% share × 0.6 prior.
        assert_eq!(prior.projected_bytes, Some(120_000));

        // A partially-measured flagged set falls back WHOLESALE rather than
        // mixing a measurement and a guess inside one sentence.
        let mut partial = MeasuredShrinks::default();
        partial.per_column.insert(
            "not-the-flagged-column".to_string(),
            ColumnShrink {
                before_bytes: 100,
                after_bytes: 10,
                layers: 1,
            },
        );
        let still_prior = rule_raw_f64_columns(&report_with_share(0.20, 0.0), &partial).unwrap();
        assert_eq!(still_prior.projected_bytes, Some(120_000));
        assert!(still_prior
            .projected
            .as_deref()
            .unwrap()
            .contains(ESTIMATE_LABEL));
    }

    /// **The doctrine cap.** The relaxation is bounded by
    /// [`shrink_trial_budget`] — this is the test the plan asks for, and the
    /// thing standing between "re-encode the sample" and "re-encode the
    /// archive".
    #[test]
    fn measured_shrink_never_exceeds_its_trial_budget() {
        // hash ids + raw f64 ⇒ both rules flag ⇒ the combined trial runs too.
        let spec = FixtureSpec {
            hash_ids: true,
            tiles: 12, // more tiles than DOCTOR_SAMPLE_TILES, deliberately
            rows: 400,
            ..Default::default()
        };
        let (report, shrinks, _doc) = shrink_fixture(&spec, None);

        let flagged = flagged_raw_f64(&report).len();
        let has_ids = flagged_expensive_ids(&report).is_some();
        assert!(flagged >= 1 && has_ids, "fixture must flag both rules");

        let columns = flagged + usize::from(has_ids);
        let budget = shrink_trial_budget(columns, true);
        assert!(
            shrinks.trials <= budget,
            "{} trials spent against a budget of {budget}",
            shrinks.trials
        );
        // The form that actually matters: work is bounded by the SAMPLE, never
        // by the archive. 12 tiles exist; at most 8 are ever re-encoded.
        assert!(
            shrinks.layers_measured <= DOCTOR_SAMPLE_TILES,
            "re-encoded {} layers — the doctrine is `never beyond the sample`",
            shrinks.layers_measured
        );
        assert!(shrinks.layers_measured > 0);
        assert_eq!(
            shrinks.compressions,
            2 * shrinks.trials,
            "one trial is exactly one before/after pair"
        );
        // And the plan's literal ceiling, in its own unit.
        assert!(shrinks.trials <= DOCTOR_SAMPLE_TILES * (flagged + 1 + 1));
    }

    /// The two remediation sets are DISJOINT by column — the additivity the
    /// combined what-if rests on. `id` is reserved, so `raw-f64-column` can
    /// never flag it however expensive it gets.
    #[test]
    fn raw_f64_and_expensive_ids_flag_disjoint_columns() {
        let (report, shrinks, doc) = shrink_fixture(
            &FixtureSpec {
                hash_ids: true,
                ..Default::default()
            },
            None,
        );
        let raw: Vec<&str> = flagged_raw_f64(&report)
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        let id = flagged_expensive_ids(&report).expect("hash ids are expensive");
        assert!(
            !raw.contains(&id.name.as_str()),
            "`id` must never be flagged as a raw-f64 property column: {raw:?}"
        );
        assert!(RESERVED_COLUMNS.contains(&"id"), "the structural reason");

        // Both fired ⇒ the combined trial ran ⇒ the column joint exists.
        assert!(shrinks.combined.is_some());
        let joint = doc
            .joint
            .iter()
            .find(|j| j.remediation_set.contains(&"raw-f64-column".to_string()))
            .expect("the column joint what-if");
        assert_eq!(
            joint.remediation_set,
            vec![
                "raw-f64-column".to_string(),
                "expensive-feature-ids".to_string()
            ]
        );
        assert!(joint.note.contains("DISJOINT"), "{}", joint.note);
        assert!(joint.note.contains(MEASURED_LABEL), "{}", joint.note);
        assert!(joint.projected_bytes > 0);
    }

    /// **Test 3 — the joint what-if is STRICTLY LESS than the sum.**
    ///
    /// The constructed overlap: an oversized (>1 MiB) blob sits at z0, i.e.
    /// BELOW the zoom floor `z0-bomb` recommends. `z0-bomb` counts its whole
    /// blob; `oversized-blobs` counts its excess over 1 MiB again. Fusing
    /// applies the min-zoom raise first, so that excess is charged once.
    #[test]
    fn joint_zoom_floor_and_blobs_is_less_than_the_sum() {
        let (_report, _shrinks, doc) = shrink_fixture(
            &FixtureSpec {
                rows: 100_000,
                tiles: 2,
                identical_payloads: true,
                world_random_coords: true,
                zooms: Some(vec![0, 10]),
                min_zoom: 0,
                max_zoom: 10,
                bounds: Some(BoundingBox {
                    min_lon: -73.8,
                    min_lat: 45.4,
                    max_lon: -73.3,
                    max_lat: 45.9,
                }),
                ..Default::default()
            },
            Some(1),
        );

        let z0 = find(&doc, "z0-bomb").expect("z0-bomb fires");
        let blobs = find(&doc, "oversized-blobs").expect("oversized-blobs fires");
        let sum = z0.projected_bytes.unwrap() + blobs.projected_bytes.unwrap();

        let joint = doc
            .joint
            .iter()
            .find(|j| {
                j.remediation_set == vec!["z0-bomb".to_string(), "oversized-blobs".to_string()]
            })
            .expect("the z0 × blobs joint what-if");
        assert!(
            joint.projected_bytes < sum,
            "fused {} must be strictly under the summed {sum} — the z0 blob's excess is \
             double-counted otherwise",
            joint.projected_bytes
        );
        assert!(joint.projected_bytes > 0);
        assert!(
            joint.note.contains("double-count removed"),
            "{}",
            joint.note
        );
        assert!(joint.note.contains(DIRECTORY_LABEL), "{}", joint.note);
        // The joint is exactly `shallow bytes + the excess of what survives`.
        assert!(joint.projected_bytes >= z0.projected_bytes.unwrap());
    }

    /// **Test 4 — the ordering.** Within one severity: larger
    /// `projected_bytes` first, ties broken by code then message. A finding
    /// with no byte figure sorts last in its tier.
    #[test]
    fn findings_sort_by_projected_bytes_within_a_severity() {
        let f = |code: &str, bytes: Option<u64>, severity: Severity| Finding {
            severity,
            code: code.to_string(),
            message: format!("{code} message"),
            remediation: vec![],
            projected: None,
            projected_bytes: bytes,
        };
        let mut findings = vec![
            f("zzz-small", Some(10), Severity::Warning),
            f("aaa-none", None, Severity::Warning),
            f("mmm-big", Some(900), Severity::Warning),
            // Ties on bytes fall back to code.
            f("bbb-tie", Some(500), Severity::Warning),
            f("aaa-tie", Some(500), Severity::Warning),
            // Severity still dominates everything.
            f("info-huge", Some(u64::MAX / 2), Severity::Info),
            f("critical-tiny", Some(1), Severity::Critical),
        ];
        sort_findings(&mut findings);
        let order: Vec<&str> = findings.iter().map(|f| f.code.as_str()).collect();
        assert_eq!(
            order,
            vec![
                "critical-tiny", // severity first, byte size irrelevant across tiers
                "mmm-big",
                "aaa-tie", // 500 == 500 → code breaks the tie
                "bbb-tie",
                "zzz-small",
                "aaa-none", // no byte figure ⇒ last in its tier
                "info-huge",
            ]
        );
    }

    /// The `--strict` gate counts Warning-or-worse findings and never looks at
    /// order, so re-ranking cannot flip CI. Pinned here at the library level;
    /// `crates/spatiotemporal-tiles/tests/optimize_cli.rs` pins it at the
    /// binary.
    #[test]
    fn reranking_cannot_move_a_finding_across_a_severity_boundary() {
        let (_r, _s, doc) = shrink_fixture(&FixtureSpec::default(), None);
        let gate = |findings: &[Finding]| {
            findings
                .iter()
                .filter(|f| f.severity <= Severity::Warning)
                .count()
        };
        let before = gate(&doc.findings);
        // Any permutation, re-sorted, yields the same gate count: the sort is a
        // permutation of the same multiset of severities.
        let mut shuffled = doc.findings.clone();
        shuffled.reverse();
        sort_findings(&mut shuffled);
        assert_eq!(before, gate(&shuffled));
        assert_findings_sorted(&doc);
    }

    /// **Determinism.** Two doctor runs over the same archive serialize
    /// byte-identical JSON — required, not aspirational: the re-encode pass is
    /// new output-affecting work and pack names are content-addressed.
    #[test]
    fn doctor_json_is_byte_identical_across_runs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build(
            &out,
            &FixtureSpec {
                hash_ids: true,
                stuck_column: true,
                tiles: 4,
                rows: 600,
                ..Default::default()
            },
        );
        let ts = PackedTileset::open(&out).unwrap();
        let report = inspect(&ts, None).unwrap();

        let a = serde_json::to_string_pretty(&doctor(&ts, &report).unwrap()).unwrap();
        let b = serde_json::to_string_pretty(&doctor(&ts, &report).unwrap()).unwrap();
        assert_eq!(a, b, "doctor JSON must be byte-identical across runs");
        assert!(a.contains("\"projected_bytes\""));
        assert!(a.contains("\"joint\""));

        // The measured-shrink pass itself is a pure function of the sample.
        let sampled = sample_decode(&ts).unwrap();
        let x = measure_shrinks(&report, &sampled).unwrap();
        let y = measure_shrinks(&report, &sampled).unwrap();
        assert_eq!(x.per_column, y.per_column);
        assert_eq!(x.combined, y.combined);
        assert_eq!((x.trials, x.compressions), (y.trials, y.compressions));
    }

    /// The auto-quantization mirror follows `stt_core`'s branches: the magnitude
    /// refusal (which sends the column back to the prior), the exact-integer
    /// path, and the range-adaptive `UInt16` fall-through.
    #[test]
    fn auto_quantize_mirror_matches_the_encoders_branches() {
        let arr = |v: Vec<f64>| -> ArrayRef { Arc::new(Float64Array::from(v)) };

        // (3) fractional values → range-adaptive UInt16.
        let ranged = auto_quantized(&arr(vec![0.5, 1.25, 9.75])).expect("range-adaptive");
        assert_eq!(ranged.data_type(), &DataType::UInt16);

        // (2) integer-valued, narrow span → exact UInt16 at step 1.
        let exact = auto_quantized(&arr(vec![10.0, 11.0, 12.0])).expect("exact integer");
        assert_eq!(exact.data_type(), &DataType::UInt16);

        // (2) integer-valued, span past 16 bits → Int32.
        let wide = auto_quantized(&arr(vec![0.0, 200_000.0])).expect("widened");
        assert_eq!(wide.data_type(), &DataType::Int32);

        // (1) the magnitude refusal — the ONLY None, and the reason such a
        // column falls back to the prior instead of being priced.
        assert!(auto_quantized(&arr(vec![0.0, i32::MAX as f64])).is_none());

        // Not a Float64 column at all: nothing to quantize.
        let ints: ArrayRef = Arc::new(UInt64Array::from(vec![1u64, 2, 3]));
        assert!(auto_quantized(&ints).is_none());

        // The id remediation only applies to the encoder's UInt64 id column.
        assert!(sequential_ids(&ints, 3).is_some());
        assert!(sequential_ids(&arr(vec![1.0]), 1).is_none());
    }
}

/// Frames one watchable playback loop shows — 45 s at 30 fps, the same model
/// `crate::analysis::temporal` derives its recommended bucket from, mirrored
/// here because those constants are private to the analyzer.
const PLAYBACK_LOOP_FRAMES: u64 = 45 * 30;

/// How far a vertex-time step may exceed the per-frame budget before the
/// advisory bothers to speak. A step INSIDE the budget is invisible during
/// playback; one a little past it is not worth a rebuild.
const VERTEX_TIME_SLACK: f64 = 4.0;

/// Column share below which the lever cannot pay for a rebuild whatever the
/// arithmetic says.
const VERTEX_TIME_MIN_SHARE: f64 = 0.05;

/// TB-11 extension 3 — couple the vertex-time ceiling to PLAYBACK.
///
/// §2.5's knob-level meta-problem: `--vertex-time-precision` is specified in
/// absolute milliseconds, but what a viewer can actually SEE depends on how
/// fast the archive plays. One 45-second, 30 fps loop of a dataset spanning
/// `D` ms advances `D / 1350` ms of dataset time per frame, so a vertex-time
/// step below that is quantization nobody can perceive — and a step far above
/// it is visible stepping in the trails.
///
/// This is an ADVISORY, never a builder constant: the loop length is a property
/// of how a dataset is presented, not of the data, so the number belongs in a
/// recommended recipe a human reads. It fires only for archives that actually
/// spend bytes on the column and are not already at the exact tier.
fn rule_vertex_time_precision(report: &InspectReport) -> Option<Finding> {
    let col = report
        .per_column
        .iter()
        .find(|c| c.name == "vertex_time" && c.compressed_bytes > 0)?;
    if col.share < VERTEX_TIME_MIN_SHARE {
        return None;
    }
    // The exact-i64 fallback is a DIFFERENT problem (it is already paying 2-4x
    // for precision nothing asked for) and the u16 tier at a wide span is the
    // shape this lever helps; a column already on u16 has nothing to gain.
    let on_u32 = col.encoding_note.starts_with("u32");
    let on_i64 = col.encoding_note.starts_with("i64");
    if !(on_u32 || on_i64) {
        return None;
    }

    let duration_ms = report.time_end_ms.saturating_sub(report.time_start_ms);
    if duration_ms == 0 {
        return None;
    }
    let per_frame_ms = (duration_ms / PLAYBACK_LOOP_FRAMES).max(1);
    // The default ceiling the builder ships. A recommendation only helps if the
    // perceptual budget is comfortably WIDER than it — that is what lets the
    // encoder drop from u32/i64 to the u16 tier.
    let default_ceiling_ms = u64::from(stt_core::arrow_tile::DEFAULT_VERTEX_TIME_MAX_STEP_MS);
    if (per_frame_ms as f64) < default_ceiling_ms as f64 * VERTEX_TIME_SLACK {
        return None;
    }

    // The u16 tier holds a span of 65_535 * step; recommending the perceptual
    // budget is what lets a wide-span layer reach it.
    let suggested = per_frame_ms;
    let projected_bytes = if on_i64 {
        // i64 (8 B/vertex) -> u16 (2 B) is a 4x column shrink at best.
        col.compressed_bytes * 3 / 4
    } else {
        // u32 (4 B) -> u16 (2 B) halves it.
        col.compressed_bytes / 2
    };
    Some(Finding {
        severity: Severity::Info,
        code: "vertex-time-precision".to_string(),
        message: format!(
            "vertex_time is on the {} tier and costs {:.0}% of tile bytes, but this archive \
             spans {} ms — one frame of a 45 s / 30 fps playback loop advances ~{} ms, so a \
             step up to that is imperceptible while playing. The {} ms default ceiling is \
             {:.0}x tighter than playback can resolve, which is what keeps the column off \
             the u16 tier.",
            if on_i64 { "exact i64" } else { "u32 delta" },
            col.share * 100.0,
            duration_ms,
            per_frame_ms,
            default_ceiling_ms,
            per_frame_ms as f64 / default_ceiling_ms as f64,
        ),
        remediation: vec![format!(
            "rebuild with --vertex-time-precision {suggested} (LOSSY: raises the per-vertex \
             time quantization ceiling to {suggested} ms; scrubbing FRAME-BY-FRAME, or \
             playing back slower than a 45 s loop, can resolve finer than this)"
        )],
        projected: Some(format!(
            "~{:.2} MB of the vertex_time column, if the wider ceiling reaches the u16 tier",
            projected_bytes as f64 / 1e6
        )),
        projected_bytes: Some(projected_bytes),
    })
}
