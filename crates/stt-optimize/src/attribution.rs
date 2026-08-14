//! Per-column compressed-cost attribution, shared by the sample measurement
//! ([`crate::measure`]) and the built-tileset inspection
//! ([`crate::analysis::inspect`]).
//!
//! # Two designs
//!
//! **`singleton-v1` (the incumbent, kept as the rollback).** Each column is
//! re-encoded ALONE — a single-field Arrow IPC stream + zstd — and charged the
//! resulting bytes. Cheap and intuitive, but it violates the *efficiency*
//! axiom: a standalone re-encode repays IPC framing per column and throws away
//! every cross-column match zstd would have found, so the per-column sum
//! exceeds the whole-tile size and any two columns that duplicate each other
//! are each charged the FULL cost of the shared information.
//!
//! **`loo-v2` (the default).** Leave-one-out: compute `v([m])` = zstd(IPC(all
//! columns)) once, then for every column `j` compute `v([m] \ {j})` =
//! zstd(IPC(batch without column j)). That is `m + 1` compressions. Column
//! `j`'s cost is its MARGINAL
//!
//! ```text
//! marginal_bytes(j) = max(0, v([m]) − v([m] \ {j}))
//! ```
//!
//! — the actual post-zstd bytes the tile sheds if the column vanishes, which is
//! the honest unit for a "what if I dropped / quantized this column" question.
//! Shares are the normalised marginals, so `Σ share = 1` and the efficiency
//! property is restored at the leave-one-out compromise.
//!
//! The duplicated-column case is the recorded misattribution the switch fixes:
//! two byte-identical columns get NEAR-ZERO marginals each (dropping one costs
//! nothing while the other still carries the information), where the singleton
//! proxy charged both of them in full. `attribution_moves_the_duplicate_column_case`
//! pins it.
//!
//! ⚠️ Marginals are sub-additive: on a highly redundant tile `Σ marginals` can
//! be far below `v([m])`, which makes normalised shares spiky. That dispersion
//! is not hidden — it is exactly what [`ShareDispersion`] quantifies and what
//! every consumer publishes as `share_stderr`.
//!
//! ⚠️ Cost. `loo-v2` compresses `m + 1` near-full batches where `singleton-v1`
//! compressed `m` single columns, so the compressor sees roughly `m×` more
//! input bytes. It is bounded by the sample size (measure) and by the decode
//! sample (inspect); exhaustive `inspect` of a large archive is the expensive
//! case. `AttributionDesign::SingletonV1` remains available for one release.
//!
//! # Determinism
//!
//! Everything on this path is a pure function of the batch:
//!
//! - columns are visited in SCHEMA ORDER and results are returned in schema
//!   order (no name sort, no map iteration);
//! - field metadata AND schema metadata are stripped before every IPC write —
//!   Arrow serialises those `HashMap`s in nondeterministic order, so keeping
//!   them would make two runs disagree by a few bytes. Every projection
//!   (whole, leave-one-out, singleton) goes through the same stripping path, so
//!   the three are comparable by construction;
//! - the dispersion accumulator sums exact `u128` integers keyed by a
//!   `BTreeMap`, so neither float association order nor hash iteration order
//!   can move a published number.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow::datatypes::{Field, Schema};
use arrow::ipc::writer::StreamWriter;
use stt_core::compression::compress_zstd_with_dict_level;

/// Which attribution design produced a set of per-column costs.
///
/// Published as a discriminator string on the reports
/// ([`crate::analysis::inspect::DecodeStats::attribution`]) so a consumer can
/// tell which definition of "column cost" it is reading — the two are NOT
/// numerically comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttributionDesign {
    /// Re-encode each column alone. Over-charges every column that shares
    /// information with another; kept as the documented rollback for one
    /// release.
    SingletonV1,
    /// Leave-one-out marginals over `m + 1` whole-batch encodes.
    #[default]
    LeaveOneOutV2,
}

impl AttributionDesign {
    /// Stable discriminator string for reports.
    pub fn as_str(self) -> &'static str {
        match self {
            AttributionDesign::SingletonV1 => "singleton-v1",
            AttributionDesign::LeaveOneOutV2 => "loo-v2",
        }
    }
}

/// Leave-one-out cost of one column within one batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LooCost {
    /// `v([m])` — compressed bytes of the whole batch. Identical for every
    /// column of a batch; carried per column so a caller can check the
    /// `0 ≤ marginal_bytes ≤ whole_bytes` invariant without threading it.
    pub whole_bytes: u64,
    /// `v([m] \ {j})` — compressed bytes of the batch with this column removed.
    pub without_bytes: u64,
    /// `max(0, whole_bytes − without_bytes)`: the post-zstd bytes this batch
    /// sheds if the column vanishes.
    pub marginal_bytes: u64,
}

/// Attributed bytes for one column under whichever design was asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnBytes {
    /// Column name.
    pub name: String,
    /// The cost charged to this column under the active design: leave-one-out
    /// marginal bytes under [`AttributionDesign::LeaveOneOutV2`], standalone
    /// re-encode bytes under [`AttributionDesign::SingletonV1`]. This is what
    /// `share` is normalised from.
    pub bytes: u64,
    /// The leave-one-out marginal. `0` under the singleton rollback, where no
    /// leave-one-out encode was performed — check the design discriminator
    /// before reading it as a measurement.
    pub marginal_bytes: u64,
}

/// Project `batch` down to `keep` (column indices, in the order given) with all
/// Arrow metadata stripped.
///
/// Stripping is load-bearing twice over: it removes the nondeterministic
/// `HashMap` serialisation, and it makes the whole/leave-one-out/singleton
/// encodes differ ONLY by which columns they carry.
fn project(batch: &RecordBatch, keep: &[usize]) -> Result<RecordBatch> {
    let schema = batch.schema();
    let fields: Vec<Field> = keep
        .iter()
        .map(|&i| schema.field(i).clone().with_metadata(Default::default()))
        .collect();
    let columns: Vec<ArrayRef> = keep.iter().map(|&i| batch.column(i).clone()).collect();
    // Explicit row count: a zero-column projection (the m = 1 leave-one-out)
    // has no array to infer it from.
    let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
        .context("attribution: projected batch build failed")
}

/// Arrow IPC stream + zstd length of a batch, in bytes.
pub fn ipc_zstd_len(batch: &RecordBatch, zstd_level: i32) -> Result<u64> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())
            .context("attribution: IPC writer init")?;
        writer.write(batch).context("attribution: IPC write")?;
        writer.finish().context("attribution: IPC finish")?;
    }
    Ok(compress_zstd_with_dict_level(&buf, None, zstd_level)?.len() as u64)
}

/// IPC + zstd length of `batch` projected to `keep`.
fn projected_len(batch: &RecordBatch, keep: &[usize], zstd_level: i32) -> Result<u64> {
    ipc_zstd_len(&project(batch, keep)?, zstd_level)
}

/// Leave-one-out attribution: `m + 1` compressions, results in schema order.
///
/// Computes `v([m])` once and `v([m] \ {j})` for every column `j`. See the
/// module docs for why this replaces the singleton proxy.
pub fn attribute_columns_loo(
    batch: &RecordBatch,
    zstd_level: i32,
) -> Result<Vec<(String, LooCost)>> {
    let schema = batch.schema();
    let m = schema.fields().len();
    let all: Vec<usize> = (0..m).collect();
    let whole_bytes = projected_len(batch, &all, zstd_level)?;

    let mut out = Vec::with_capacity(m);
    let mut keep: Vec<usize> = Vec::with_capacity(m.saturating_sub(1));
    for j in 0..m {
        keep.clear();
        keep.extend(all.iter().copied().filter(|&i| i != j));
        let without_bytes = projected_len(batch, &keep, zstd_level)?;
        out.push((
            schema.field(j).name().clone(),
            LooCost {
                whole_bytes,
                without_bytes,
                // max(0, ·): zstd is not monotone in the column set — dropping
                // a column can occasionally cost bytes by disturbing matches.
                marginal_bytes: whole_bytes.saturating_sub(without_bytes),
            },
        ));
    }
    Ok(out)
}

/// The incumbent proxy: re-encode each column alone, results in schema order.
///
/// Retained as the rollback behind [`AttributionDesign::SingletonV1`] and as
/// the comparison arm the duplicated-column test measures against.
pub fn attribute_columns_singleton(
    batch: &RecordBatch,
    zstd_level: i32,
) -> Result<Vec<(String, u64)>> {
    let schema = batch.schema();
    let mut out = Vec::with_capacity(schema.fields().len());
    for j in 0..schema.fields().len() {
        let bytes = projected_len(batch, &[j], zstd_level)?;
        out.push((schema.field(j).name().clone(), bytes));
    }
    Ok(out)
}

/// Attribute every column of `batch` under `design`, in SCHEMA ORDER.
///
/// The returned vector is index-aligned with `batch.schema().fields()`, which
/// is what lets a caller zip it against the fields it already walked.
pub fn attribute_columns(
    batch: &RecordBatch,
    zstd_level: i32,
    design: AttributionDesign,
) -> Result<Vec<ColumnBytes>> {
    Ok(match design {
        AttributionDesign::LeaveOneOutV2 => attribute_columns_loo(batch, zstd_level)?
            .into_iter()
            .map(|(name, cost)| ColumnBytes {
                name,
                bytes: cost.marginal_bytes,
                marginal_bytes: cost.marginal_bytes,
            })
            .collect(),
        AttributionDesign::SingletonV1 => attribute_columns_singleton(batch, zstd_level)?
            .into_iter()
            .map(|(name, bytes)| ColumnBytes {
                name,
                bytes,
                marginal_bytes: 0,
            })
            .collect(),
    })
}

// ----------------------------------------------------------------------------
// Dispersion of the published shares (MO-2)
// ----------------------------------------------------------------------------

/// Per-column moments needed for the ratio-estimator variance.
#[derive(Debug, Clone, Default)]
struct ColumnMoments {
    /// `Σ y_i` — this column's attributed bytes over the observed tiles.
    sum_y: u128,
    /// `Σ y_i²`.
    sum_yy: u128,
    /// `Σ x_i y_i`.
    sum_xy: u128,
}

/// Standard error of every per-column share, accumulated one observed tile at
/// a time.
///
/// A published share is the RATIO estimator `R̂ = Σ_i y_i / Σ_i x_i` — this
/// column's attributed bytes over all attributed bytes, summed across the
/// observed tiles (`y_i` / `x_i` for tile `i`). Its standard error is the
/// textbook ratio-estimator form
///
/// ```text
/// se(R̂)² = Σ_i (y_i − R̂·x_i)² / ((n − 1) · n · x̄²)
/// ```
///
/// expanded as `Σy² − 2R̂·Σxy + R̂²·Σx²` so nothing has to be retained per tile.
/// The sums are exact `u128` integers, so the result does not depend on the
/// order tiles were observed in.
///
/// **Interpretation, and why there is no finite-population correction.** Under
/// a sampled decode this is straightforwardly the sampling error of the share.
/// Under an EXHAUSTIVE decode a finite-population correction would drive it to
/// zero, and this deliberately does not apply one: the number that stays is the
/// tile-to-tile dispersion of the share — how far the share would move if the
/// archive's tiles were redrawn from the same producer — which is the quantity
/// a threshold consumer actually needs, and it is finite, not zero. Every
/// consumer's doc string repeats this.
///
/// A column absent from a tile contributes `y_i = 0` for that tile rather than
/// being dropped from the sample, so shares of intermittent columns are not
/// silently inflated. Tiles that attributed no bytes at all are skipped
/// entirely (they carry no information about any share).
#[derive(Debug, Clone, Default)]
pub struct ShareDispersion {
    tiles: u64,
    sum_x: u128,
    sum_xx: u128,
    columns: BTreeMap<String, ColumnMoments>,
}

impl ShareDispersion {
    /// Record one observed tile: the per-column attributed bytes it produced,
    /// merged over that tile's layers.
    ///
    /// `BTreeMap`, not `HashMap` — the iteration order of this argument feeds
    /// the accumulator and is part of the determinism contract.
    pub fn observe_tile(&mut self, per_column_bytes: &BTreeMap<String, u64>) {
        let x: u128 = per_column_bytes.values().map(|&v| u128::from(v)).sum();
        if x == 0 {
            return;
        }
        self.tiles += 1;
        self.sum_x += x;
        self.sum_xx += x * x;
        for (name, &y) in per_column_bytes {
            let y = u128::from(y);
            let m = self.columns.entry(name.clone()).or_default();
            m.sum_y += y;
            m.sum_yy += y * y;
            m.sum_xy += x * y;
        }
    }

    /// Tiles that contributed an observation.
    pub fn tiles(&self) -> u64 {
        self.tiles
    }

    /// Standard error of the named column's share.
    ///
    /// `0.0` when fewer than two tiles were observed: one tile carries no
    /// dispersion evidence at all, and publishing a fabricated spread would be
    /// worse than publishing none. Consumers gate CONSERVATIVELY (they
    /// SUBTRACT the stderr), so a zero here reproduces the pre-MO-2 behaviour
    /// exactly rather than silencing a rule.
    pub fn stderr(&self, name: &str) -> f64 {
        if self.tiles < 2 {
            return 0.0;
        }
        let Some(m) = self.columns.get(name) else {
            return 0.0;
        };
        let n = self.tiles as f64;
        let sum_x = self.sum_x as f64;
        if sum_x <= 0.0 {
            return 0.0;
        }
        let sse = self.residual_sum_of_squares(m);
        // s² = SSE/(n−1) and n·x̄² = (Σx)²/n, so var = SSE·n / ((n−1)·(Σx)²).
        let var = sse * n / ((n - 1.0) * sum_x * sum_x);
        var.max(0.0).sqrt()
    }

    /// `Σ_i (y_i − R̂·x_i)²` with `R̂ = Σy/Σx`.
    ///
    /// Computed as `(Σx²·Σy² − 2·Σx·Σy·Σxy + Σy²·Σx²) / (Σx)²` — exactly, in
    /// `u128`, whenever the products fit. That exactness is what makes
    /// perfectly homogeneous tiles report EXACTLY zero dispersion instead of a
    /// cancellation residue; the naive `Σy² − 2R·Σxy + R²·Σx²` subtracts two
    /// near-equal `1e18`-scale floats and leaves ~1e-10 behind.
    ///
    /// Archives whose byte sums overflow the `u128` products fall back to that
    /// float expansion. The choice is a pure function of the accumulated sums,
    /// so both paths stay deterministic.
    fn residual_sum_of_squares(&self, m: &ColumnMoments) -> f64 {
        let sx = self.sum_x;
        let sy = m.sum_y;
        let exact = || -> Option<u128> {
            let a = sx.checked_mul(sx)?.checked_mul(m.sum_yy)?;
            let b = sx.checked_mul(sy)?.checked_mul(m.sum_xy)?.checked_mul(2)?;
            let c = sy.checked_mul(sy)?.checked_mul(self.sum_xx)?;
            // Cauchy–Schwarz guarantees a + c >= b, so this cannot wrap.
            a.checked_add(c)?.checked_sub(b)
        };
        let sum_x = sx as f64;
        match exact() {
            Some(numerator) => numerator as f64 / (sum_x * sum_x),
            None => {
                let r = sy as f64 / sum_x;
                ((m.sum_yy as f64) - 2.0 * r * (m.sum_xy as f64) + r * r * (self.sum_xx as f64))
                    .max(0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray, UInt64Array};
    use arrow::datatypes::DataType;

    /// zstd level the tests attribute at — the same publish level
    /// `analysis::inspect` pins.
    const LEVEL: i32 = 19;

    /// splitmix64: full-entropy values with no RNG dependency.
    fn mix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand01(x: u64) -> f64 {
        (mix(x) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Build a batch from `(name, array)` pairs, in the order given.
    fn batch_of(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, a)| Field::new(*name, a.data_type().clone(), a.null_count() > 0))
            .collect();
        let arrays: Vec<ArrayRef> = columns.into_iter().map(|(_, a)| a).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    fn hot(seed: u64, n: usize) -> ArrayRef {
        Arc::new(Float64Array::from(
            (0..n)
                .map(|i| rand01(seed * 977 + i as u64))
                .collect::<Vec<_>>(),
        ))
    }

    fn sequential_ids(n: usize) -> ArrayRef {
        Arc::new(UInt64Array::from((0..n as u64).collect::<Vec<_>>()))
    }

    fn low_entropy_text(n: usize) -> ArrayRef {
        Arc::new(StringArray::from(
            (0..n)
                .map(|i| ["bike", "ferry", "bus"][i % 3].to_string())
                .collect::<Vec<_>>(),
        ))
    }

    fn find<'a, T>(rows: &'a [(String, T)], name: &str) -> &'a T {
        &rows
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("column {name} missing"))
            .1
    }

    #[test]
    fn leave_one_out_is_efficient_and_shares_sum_to_one() {
        let n = 512;
        let b = batch_of(vec![
            ("id", sequential_ids(n)),
            ("magnitude", hot(1, n)),
            ("kind", low_entropy_text(n)),
        ]);
        let loo = attribute_columns_loo(&b, LEVEL).unwrap();
        assert_eq!(
            loo.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["id", "magnitude", "kind"],
            "results must come back in SCHEMA order"
        );

        let total: u64 = loo.iter().map(|(_, c)| c.marginal_bytes).sum();
        assert!(total > 0);
        let share_sum: f64 = loo
            .iter()
            .map(|(_, c)| c.marginal_bytes as f64 / total as f64)
            .sum();
        assert!(
            (share_sum - 1.0).abs() < 1e-9,
            "normalised marginals must sum to 1, got {share_sum}"
        );

        // Efficiency, the property the singleton proxy breaks: the marginals
        // never over-spend the whole-tile budget, while the singleton sum does.
        let whole = loo[0].1.whole_bytes;
        assert!(
            total <= whole,
            "Σ marginals {total} must not exceed v([m]) {whole}"
        );
        let singleton_total: u64 = attribute_columns_singleton(&b, LEVEL)
            .unwrap()
            .iter()
            .map(|(_, v)| *v)
            .sum();
        assert!(
            singleton_total > whole,
            "guard for the REJECTED proxy: the singleton sum {singleton_total} must exceed \
             v([m]) {whole} — that inefficiency is the thing MO-3 removes"
        );
    }

    #[test]
    fn attribution_moves_the_duplicate_column_case() {
        // THE recorded misattribution. `dup_a` and `dup_b` carry byte-identical
        // high-entropy values, so the information is fully redundant: dropping
        // either costs almost nothing, and an honest attribution charges each
        // of them almost nothing. The singleton proxy charges BOTH in full.
        let n = 512;
        let payload = hot(7, n);
        let b = batch_of(vec![
            ("id", sequential_ids(n)),
            ("dup_a", payload.clone()),
            ("dup_b", payload.clone()),
            ("solo", hot(9, n)),
        ]);

        let loo = attribute_columns_loo(&b, LEVEL).unwrap();
        let singleton = attribute_columns_singleton(&b, LEVEL).unwrap();

        let dup_a = *find(&loo, "dup_a");
        let dup_b = *find(&loo, "dup_b");
        let solo = *find(&loo, "solo");
        let s_dup_a = *find(&singleton, "dup_a");
        let s_dup_b = *find(&singleton, "dup_b");

        // The singleton proxy over-charges BOTH copies — each is billed the
        // full entropy of information the tile only stores once.
        assert!(
            s_dup_a > 3_000 && s_dup_b > 3_000,
            "guard: singleton must over-charge both copies ({s_dup_a} / {s_dup_b} B)"
        );
        // Leave-one-out charges each copy near nothing.
        assert!(
            dup_a.marginal_bytes * 4 < s_dup_a,
            "dup_a marginal {} must be far below its singleton cost {s_dup_a}",
            dup_a.marginal_bytes
        );
        assert!(
            dup_b.marginal_bytes * 4 < s_dup_b,
            "dup_b marginal {} must be far below its singleton cost {s_dup_b}",
            dup_b.marginal_bytes
        );
        // …while the genuinely unique high-entropy column keeps its full cost.
        assert!(
            solo.marginal_bytes > 10 * dup_a.marginal_bytes.max(1),
            "solo {} vs duplicate {}",
            solo.marginal_bytes,
            dup_a.marginal_bytes
        );

        // In share terms: the duplicated pair drops out of the ranking under
        // LOO and dominates it under the proxy.
        let loo_total: u64 = loo.iter().map(|(_, c)| c.marginal_bytes).sum();
        let s_total: u64 = singleton.iter().map(|(_, v)| *v).sum();
        let loo_pair =
            (dup_a.marginal_bytes + dup_b.marginal_bytes) as f64 / loo_total.max(1) as f64;
        let s_pair = (s_dup_a + s_dup_b) as f64 / s_total.max(1) as f64;
        assert!(loo_pair < 0.10, "LOO pair share {loo_pair:.4}");
        assert!(s_pair > 0.50, "singleton pair share {s_pair:.4}");
    }

    #[test]
    fn high_entropy_column_dominates_the_marginals() {
        let n = 512;
        let b = batch_of(vec![
            ("id", sequential_ids(n)),
            ("kind", low_entropy_text(n)),
            ("noise", hot(3, n)),
        ]);
        let loo = attribute_columns_loo(&b, LEVEL).unwrap();
        let noise = find(&loo, "noise").marginal_bytes;
        for (name, cost) in &loo {
            if name != "noise" {
                assert!(
                    noise > 4 * cost.marginal_bytes.max(1),
                    "noise {noise} must dominate {name} {}",
                    cost.marginal_bytes
                );
            }
        }
    }

    #[test]
    fn single_column_batch_degenerates_sanely() {
        let n = 256;
        let b = batch_of(vec![("only", hot(5, n))]);
        let loo = attribute_columns_loo(&b, LEVEL).unwrap();
        assert_eq!(loo.len(), 1);
        let cost = loo[0].1;
        // v([m] \ {only}) is an empty-schema batch: pure IPC + zstd framing.
        assert!(
            cost.without_bytes < cost.whole_bytes,
            "framing {} vs whole {}",
            cost.without_bytes,
            cost.whole_bytes
        );
        assert_eq!(
            cost.marginal_bytes,
            cost.whole_bytes - cost.without_bytes,
            "the one column carries everything above the framing floor"
        );
        // Normalising a single marginal gives share 1.
        assert!(cost.marginal_bytes > 0);
    }

    #[test]
    fn marginals_stay_inside_the_whole_batch_budget() {
        // Property: 0 <= marginal_bytes <= v([m]) for arbitrary small batches.
        for seed in 0..8u64 {
            let n = 16 + (seed as usize) * 23;
            let b = batch_of(vec![
                ("id", sequential_ids(n)),
                ("a", hot(seed, n)),
                ("b", hot(seed + 100, n)),
                ("text", low_entropy_text(n)),
            ]);
            let loo = attribute_columns_loo(&b, LEVEL).unwrap();
            let whole = loo[0].1.whole_bytes;
            for (name, cost) in &loo {
                assert_eq!(cost.whole_bytes, whole, "{name}: v([m]) must be shared");
                assert!(
                    cost.marginal_bytes <= cost.whole_bytes,
                    "{name}: marginal {} > whole {}",
                    cost.marginal_bytes,
                    cost.whole_bytes
                );
            }
            let total: u64 = loo.iter().map(|(_, c)| c.marginal_bytes).sum();
            assert!(total <= whole, "seed {seed}: Σ marginals {total} > {whole}");
        }
    }

    #[test]
    fn attribution_is_byte_identical_across_runs() {
        let n = 300;
        // Field-level metadata present: it must be stripped identically every
        // time, or the two runs disagree by a few bytes.
        let noisy = Field::new("magnitude", DataType::Float64, false).with_metadata(
            [("stt:qa".to_string(), "0.01".to_string())]
                .into_iter()
                .collect(),
        );
        let schema = Schema::new(vec![Field::new("id", DataType::UInt64, false), noisy])
            .with_metadata(
                [("stt:geometry".to_string(), "geoarrow.point".to_string())]
                    .into_iter()
                    .collect(),
            );
        let b =
            RecordBatch::try_new(Arc::new(schema), vec![sequential_ids(n), hot(11, n)]).unwrap();

        for design in [
            AttributionDesign::LeaveOneOutV2,
            AttributionDesign::SingletonV1,
        ] {
            let a = attribute_columns(&b, LEVEL, design).unwrap();
            let c = attribute_columns(&b, LEVEL, design).unwrap();
            assert_eq!(a, c, "design {design:?} must be reproducible");
        }
        // The rollback publishes no marginal; the default publishes it as the
        // charged cost.
        let singleton = attribute_columns(&b, LEVEL, AttributionDesign::SingletonV1).unwrap();
        assert!(singleton.iter().all(|c| c.marginal_bytes == 0));
        let loo = attribute_columns(&b, LEVEL, AttributionDesign::LeaveOneOutV2).unwrap();
        assert!(loo.iter().all(|c| c.marginal_bytes == c.bytes));
        assert_eq!(
            AttributionDesign::default().as_str(),
            "loo-v2",
            "leave-one-out is the default"
        );
        assert_eq!(AttributionDesign::SingletonV1.as_str(), "singleton-v1");
    }

    // ------------------------------------------------------------------
    // MO-2: dispersion
    // ------------------------------------------------------------------

    fn tile(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
        pairs.iter().map(|(n, v)| ((*n).to_string(), *v)).collect()
    }

    #[test]
    fn homogeneous_tiles_have_zero_dispersion() {
        let mut d = ShareDispersion::default();
        for _ in 0..8 {
            d.observe_tile(&tile(&[("geometry", 600), ("magnitude", 300), ("id", 100)]));
        }
        assert_eq!(d.tiles(), 8);
        for name in ["geometry", "magnitude", "id"] {
            assert_eq!(
                d.stderr(name),
                0.0,
                "{name} must report EXACTLY zero dispersion on identical tiles"
            );
        }
        // Scaled-but-proportional tiles are still zero-dispersion: the SHARE is
        // what is estimated, not the byte count.
        let mut scaled = ShareDispersion::default();
        for k in 1..=8u64 {
            scaled.observe_tile(&tile(&[
                ("geometry", 600 * k),
                ("magnitude", 300 * k),
                ("id", 100 * k),
            ]));
        }
        assert_eq!(scaled.stderr("geometry"), 0.0);
    }

    #[test]
    fn heterogeneous_tiles_have_positive_dispersion() {
        let mut d = ShareDispersion::default();
        // Same total, wildly different mix.
        d.observe_tile(&tile(&[("geometry", 900), ("magnitude", 100)]));
        d.observe_tile(&tile(&[("geometry", 100), ("magnitude", 900)]));
        d.observe_tile(&tile(&[("geometry", 800), ("magnitude", 200)]));
        d.observe_tile(&tile(&[("geometry", 200), ("magnitude", 800)]));
        assert!(d.stderr("geometry") > 0.05, "{}", d.stderr("geometry"));
        assert!(d.stderr("magnitude") > 0.05, "{}", d.stderr("magnitude"));
        // Symmetric mix → symmetric uncertainty.
        assert!((d.stderr("geometry") - d.stderr("magnitude")).abs() < 1e-9);
    }

    #[test]
    fn dispersion_edge_cases_and_absent_columns() {
        // One tile carries no dispersion evidence → 0.0, which reproduces the
        // pre-MO-2 gate behaviour exactly.
        let mut one = ShareDispersion::default();
        one.observe_tile(&tile(&[("geometry", 10), ("id", 5)]));
        assert_eq!(one.tiles(), 1);
        assert_eq!(one.stderr("geometry"), 0.0);

        // Empty tiles are skipped, not counted.
        let mut skip = ShareDispersion::default();
        skip.observe_tile(&BTreeMap::new());
        skip.observe_tile(&tile(&[("a", 0)]));
        assert_eq!(skip.tiles(), 0);
        assert_eq!(skip.stderr("a"), 0.0);
        assert_eq!(skip.stderr("nope"), 0.0);

        // A column present in only half the tiles is treated as y = 0 in the
        // rest, so its share is dispersed, not silently inflated.
        let mut inter = ShareDispersion::default();
        for i in 0..8 {
            if i % 2 == 0 {
                inter.observe_tile(&tile(&[("base", 500), ("payload", 500)]));
            } else {
                inter.observe_tile(&tile(&[("base", 500)]));
            }
        }
        assert_eq!(inter.tiles(), 8);
        assert!(
            inter.stderr("payload") > 0.05,
            "intermittent column must carry dispersion, got {}",
            inter.stderr("payload")
        );
    }

    #[test]
    fn dispersion_is_independent_of_observation_order() {
        let tiles = [
            tile(&[("a", 700), ("b", 300)]),
            tile(&[("a", 100), ("b", 900)]),
            tile(&[("a", 550), ("b", 450)]),
            tile(&[("a", 20), ("b", 4)]),
        ];
        let mut forward = ShareDispersion::default();
        for t in tiles.iter() {
            forward.observe_tile(t);
        }
        let mut backward = ShareDispersion::default();
        for t in tiles.iter().rev() {
            backward.observe_tile(t);
        }
        assert_eq!(forward.stderr("a"), backward.stderr("a"));
        assert_eq!(forward.stderr("b"), backward.stderr("b"));
    }
}
