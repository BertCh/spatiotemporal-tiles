//! Tileset-to-tileset comparison for `stt-optimize diff`: totals, per-zoom
//! directory stats, and per-column costs between two [`InspectReport`]s
//! (typically before/after a re-encode or fleet reprocess), each as absolute
//! and percent deltas.
//!
//! Directory-derived numbers (tile/blob counts, wire bytes) are exact on both
//! sides; per-column numbers inherit whatever sampling the two inspections
//! ran with ([`DiffReport::decode_sampled`] flags this), so treat them as
//! attribution shifts, not absolute wire accounting.
//!
//! ⚠️ The `--fail-on-growth` gate metric is [`DiffReport::compressed_bytes`],
//! which comes from the DIRECTORY on both sides and is exact. Nothing on the
//! sampled per-column path feeds it, and nothing here may change that. The
//! per-column rows are decision support: they gained a graded
//! [`ColumnDiff::significant`] annotation (is this share move bigger than the
//! two sides' decode noise?) which sits BESIDE the binary
//! [`DiffReport::decode_sampled`] caveat rather than replacing it.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::analysis::inspect::InspectReport;

/// Which side of the comparison a one-sided row exists on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Before,
    After,
}

/// Absolute + percent change of an integer counter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    /// Value on the `--before` side.
    pub before: u64,
    /// Value on the `--after` side.
    pub after: u64,
    /// `after - before`.
    pub delta: i64,
    /// `100 * delta / before`; `None` when `before == 0` (percent undefined).
    pub pct: Option<f64>,
}

impl Delta {
    fn new(before: u64, after: u64) -> Self {
        let delta = after as i64 - before as i64;
        let pct = (before != 0).then(|| 100.0 * delta as f64 / before as f64);
        Self {
            before,
            after,
            delta,
            pct,
        }
    }
}

/// Absolute + percent change of a float metric (ratios, per-feature bytes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaF {
    /// Value on the `--before` side.
    pub before: f64,
    /// Value on the `--after` side.
    pub after: f64,
    /// `after - before`.
    pub delta: f64,
    /// `100 * delta / before`; `None` when `before == 0.0`.
    pub pct: Option<f64>,
}

impl DeltaF {
    fn new(before: f64, after: f64) -> Self {
        let delta = after - before;
        let pct = (before != 0.0).then(|| 100.0 * delta / before);
        Self {
            before,
            after,
            delta,
            pct,
        }
    }
}

/// Per-zoom directory comparison. A zoom present on only one side is flagged
/// via `only_in` and diffed against zeros.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomDiff {
    /// Zoom level.
    pub zoom: u8,
    /// Set when the zoom exists in only one report.
    pub only_in: Option<Side>,
    /// Directory entries at this zoom.
    pub entries: Delta,
    /// Distinct physical blobs at this zoom.
    pub distinct_blobs: Delta,
    /// Entry-weighted compressed blob bytes at this zoom.
    pub blob_bytes_total: Delta,
    /// Average compressed blob size at this zoom.
    pub avg_blob_bytes: DeltaF,
}

/// Per-column cost comparison (matched by column name). A column present on
/// only one side is flagged via `only_in` and diffed against zeros.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDiff {
    /// Column name (e.g. `geometry`, `vertex_time`, a property name).
    pub name: String,
    /// Set when the column exists in only one report.
    pub only_in: Option<Side>,
    /// Attributed bytes (IPC + zstd) over the decoded tiles, under whichever
    /// attribution design each side ran — check `decode.attribution` on the
    /// two inspect reports before reading a delta across a design change.
    pub compressed_bytes: Delta,
    /// Attributed bytes per feature row.
    pub bytes_per_feature: DeltaF,
    /// Share of the per-column total on each side, and its delta.
    #[serde(default = "zero_delta_f")]
    pub share: DeltaF,
    /// Is the share move bigger than the two sides' decode noise?
    ///
    /// `Some(|Δshare| > 2·(stderr_before + stderr_after))` when the column
    /// exists on BOTH sides — a graded companion to the binary
    /// [`DiffReport::decode_sampled`] caveat, which stays for compatibility.
    /// `None` for a one-sided row, where the change is a presence change and a
    /// share delta is not the question being asked.
    ///
    /// Both stderrs are finite even when both sides decoded exhaustively (no
    /// finite-population correction — see
    /// [`crate::analysis::inspect::ColumnCost::share_stderr`]), so this stays a
    /// conservative test in every mode: it asks whether the move exceeds
    /// ordinary tile-to-tile dispersion.
    #[serde(default)]
    pub significant: Option<bool>,
}

/// serde default for [`ColumnDiff::share`]: diffs written before the field
/// existed carry no share information.
fn zero_delta_f() -> DeltaF {
    DeltaF::new(0.0, 0.0)
}

/// Full comparison of two inspected tilesets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffReport {
    /// Dataset name of the `--before` side.
    pub before_name: String,
    /// Dataset name of the `--after` side.
    pub after_name: String,
    /// Directory entry count.
    pub tile_count: Delta,
    /// Index-weighted feature total.
    pub feature_count: Delta,
    /// Total compressed blob bytes (the `--fail-on-growth` gate metric).
    pub compressed_bytes: Delta,
    /// Total uncompressed payload bytes.
    pub uncompressed_bytes: Delta,
    /// Whole-archive uncompressed/compressed ratio.
    pub compression_ratio: DeltaF,
    /// Whole-directory dedup ratio (`distinct_blobs / entries`).
    pub dedup_ratio: DeltaF,
    /// True when either side's decode pass was sampled — per-column deltas
    /// then compare samples, not exhaustive decodes.
    pub decode_sampled: bool,
    /// Per-zoom directory deltas, ascending zoom (union of both sides).
    pub per_zoom: Vec<ZoomDiff>,
    /// Per-column cost deltas, largest side first (union of both sides).
    pub per_column: Vec<ColumnDiff>,
}

/// Compare two inspection reports (`before` → `after`).
pub fn diff(before: &InspectReport, after: &InspectReport) -> DiffReport {
    let zooms: BTreeSet<u8> = before
        .per_zoom
        .iter()
        .chain(after.per_zoom.iter())
        .map(|z| z.zoom)
        .collect();
    let per_zoom: Vec<ZoomDiff> = zooms
        .into_iter()
        .map(|zoom| {
            let b = before.per_zoom.iter().find(|z| z.zoom == zoom);
            let a = after.per_zoom.iter().find(|z| z.zoom == zoom);
            ZoomDiff {
                zoom,
                only_in: only_in(b.is_some(), a.is_some()),
                entries: Delta::new(b.map_or(0, |z| z.entries), a.map_or(0, |z| z.entries)),
                distinct_blobs: Delta::new(
                    b.map_or(0, |z| z.distinct_blobs),
                    a.map_or(0, |z| z.distinct_blobs),
                ),
                blob_bytes_total: Delta::new(
                    b.map_or(0, |z| z.blob_bytes_total),
                    a.map_or(0, |z| z.blob_bytes_total),
                ),
                avg_blob_bytes: DeltaF::new(
                    b.map_or(0.0, |z| z.avg_blob_bytes),
                    a.map_or(0.0, |z| z.avg_blob_bytes),
                ),
            }
        })
        .collect();

    let names: BTreeSet<&str> = before
        .per_column
        .iter()
        .chain(after.per_column.iter())
        .map(|c| c.name.as_str())
        .collect();
    let mut per_column: Vec<ColumnDiff> = names
        .into_iter()
        .map(|name| {
            let b = before.per_column.iter().find(|c| c.name == name);
            let a = after.per_column.iter().find(|c| c.name == name);
            let share = DeltaF::new(b.map_or(0.0, |c| c.share), a.map_or(0.0, |c| c.share));
            // Two-sided rows only: for a column that appears on one side the
            // interesting fact is the appearance, not a share delta measured
            // against a fabricated zero.
            let significant = match (b, a) {
                (Some(b), Some(a)) => {
                    let noise = 2.0 * (b.share_stderr + a.share_stderr);
                    Some(share.delta.abs() > noise)
                }
                _ => None,
            };
            ColumnDiff {
                name: name.to_string(),
                only_in: only_in(b.is_some(), a.is_some()),
                compressed_bytes: Delta::new(
                    b.map_or(0, |c| c.compressed_bytes),
                    a.map_or(0, |c| c.compressed_bytes),
                ),
                bytes_per_feature: DeltaF::new(
                    b.map_or(0.0, |c| c.bytes_per_feature),
                    a.map_or(0.0, |c| c.bytes_per_feature),
                ),
                share,
                significant,
            }
        })
        .collect();
    // Largest column (on either side) first, name as the deterministic tiebreak.
    per_column.sort_by(|x, y| {
        let kx = x.compressed_bytes.before.max(x.compressed_bytes.after);
        let ky = y.compressed_bytes.before.max(y.compressed_bytes.after);
        ky.cmp(&kx).then_with(|| x.name.cmp(&y.name))
    });

    DiffReport {
        before_name: before.name.clone(),
        after_name: after.name.clone(),
        tile_count: Delta::new(before.tile_count, after.tile_count),
        feature_count: Delta::new(before.feature_count, after.feature_count),
        compressed_bytes: Delta::new(before.compressed_bytes, after.compressed_bytes),
        uncompressed_bytes: Delta::new(before.uncompressed_bytes, after.uncompressed_bytes),
        compression_ratio: DeltaF::new(before.compression_ratio, after.compression_ratio),
        dedup_ratio: DeltaF::new(before.dedup.dedup_ratio, after.dedup.dedup_ratio),
        decode_sampled: before.decode.sampled || after.decode.sampled,
        per_zoom,
        per_column,
    }
}

fn only_in(in_before: bool, in_after: bool) -> Option<Side> {
    match (in_before, in_after) {
        (true, false) => Some(Side::Before),
        (false, true) => Some(Side::After),
        _ => None,
    }
}

/// `+3.2%` / `-1.4%`; `n/a` when the baseline was zero.
fn fmt_pct(pct: Option<f64>) -> String {
    match pct {
        Some(p) => format!("{p:+.1}%"),
        None => "n/a".to_string(),
    }
}

fn one_sided_note(only_in: Option<Side>) -> &'static str {
    match only_in {
        Some(Side::Before) => "before only",
        Some(Side::After) => "after only",
        None => "",
    }
}

/// Note for a per-column row: the one-sided flag if any, else the graded
/// significance verdict.
fn column_note(c: &ColumnDiff) -> &'static str {
    match c.only_in {
        Some(_) => one_sided_note(c.only_in),
        None => match c.significant {
            Some(true) => "significant",
            Some(false) => "within noise",
            None => "",
        },
    }
}

/// Render the diff as compact aligned text.
pub fn format_text(report: &DiffReport) -> String {
    let mut out = String::new();
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    out.push_str(&format!(
        "         STT Diff - {} -> {}\n",
        report.before_name, report.after_name
    ));
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    let row_u = |name: &str, d: &Delta| {
        format!(
            "  {:<20} {:>14} {:>14} {:>+14} {:>9}\n",
            name,
            d.before,
            d.after,
            d.delta,
            fmt_pct(d.pct)
        )
    };
    let row_mb = |name: &str, d: &Delta| {
        format!(
            "  {:<20} {:>14.2} {:>14.2} {:>+14.2} {:>9}\n",
            name,
            d.before as f64 / 1e6,
            d.after as f64 / 1e6,
            d.delta as f64 / 1e6,
            fmt_pct(d.pct)
        )
    };
    let row_f = |name: &str, d: &DeltaF| {
        format!(
            "  {:<20} {:>14.3} {:>14.3} {:>+14.3} {:>9}\n",
            name,
            d.before,
            d.after,
            d.delta,
            fmt_pct(d.pct)
        )
    };

    out.push_str("📊 Totals\n");
    out.push_str(&format!(
        "  {:<20} {:>14} {:>14} {:>14} {:>9}\n",
        "metric", "before", "after", "delta", "pct"
    ));
    out.push_str(&row_u("tiles", &report.tile_count));
    out.push_str(&row_u("features (index)", &report.feature_count));
    out.push_str(&row_mb("compressed MB", &report.compressed_bytes));
    out.push_str(&row_mb("uncompressed MB", &report.uncompressed_bytes));
    out.push_str(&row_f("compression ratio", &report.compression_ratio));
    out.push_str(&row_f("dedup ratio", &report.dedup_ratio));
    out.push('\n');

    out.push_str("🗂  Per-zoom blob bytes\n");
    out.push_str(&format!(
        "  zoom | {:>11} | {:>11} | {:>11} | {:>8} | entries         | note\n",
        "before MB", "after MB", "delta MB", "pct"
    ));
    for z in &report.per_zoom {
        out.push_str(&format!(
            "    {:2} | {:>11.2} | {:>11.2} | {:>+11.2} | {:>8} | {:>7} -> {:<6} | {}\n",
            z.zoom,
            z.blob_bytes_total.before as f64 / 1e6,
            z.blob_bytes_total.after as f64 / 1e6,
            z.blob_bytes_total.delta as f64 / 1e6,
            fmt_pct(z.blob_bytes_total.pct),
            z.entries.before,
            z.entries.after,
            one_sided_note(z.only_in)
        ));
    }
    out.push('\n');

    if !report.per_column.is_empty() {
        out.push_str(&format!(
            "💾 Per-column cost (standalone IPC+zstd{})\n",
            if report.decode_sampled {
                "; sampled decode on at least one side"
            } else {
                ""
            }
        ));
        out.push_str(&format!(
            "  {:<22} {:>11} {:>11} {:>11} {:>8}  {:>7} -> {:<7}  {:>9}  note\n",
            "column", "before KB", "after KB", "delta KB", "pct", "B/feat", "B/feat", "Δshare"
        ));
        for c in &report.per_column {
            out.push_str(&format!(
                "  {:<22} {:>11.1} {:>11.1} {:>+11.1} {:>8}  {:>7.2} -> {:<7.2}  {:>+8.2}%  {}\n",
                c.name,
                c.compressed_bytes.before as f64 / 1e3,
                c.compressed_bytes.after as f64 / 1e3,
                c.compressed_bytes.delta as f64 / 1e3,
                fmt_pct(c.compressed_bytes.pct),
                c.bytes_per_feature.before,
                c.bytes_per_feature.after,
                100.0 * c.share.delta,
                column_note(c)
            ));
        }
    }

    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::inspect::{ColumnCost, DecodeStats, DedupStats, ZoomStats};

    fn zoom(zoom: u8, entries: u64, blob_bytes_total: u64) -> ZoomStats {
        ZoomStats {
            zoom,
            entries,
            distinct_blobs: entries,
            blob_bytes_total,
            blob_bytes_max: blob_bytes_total / entries.max(1),
            avg_blob_bytes: blob_bytes_total as f64 / entries.max(1) as f64,
            t_buckets: 1,
        }
    }

    fn column(name: &str, compressed_bytes: u64, bytes_per_feature: f64) -> ColumnCost {
        ColumnCost {
            name: name.to_string(),
            dtype: "Float64".to_string(),
            compressed_bytes,
            share: 0.5,
            bytes_per_feature,
            marginal_bytes: compressed_bytes,
            share_stderr: 0.0,
            encoding_note: String::new(),
        }
    }

    /// A column with an explicit share + dispersion — what the `significant`
    /// annotation is computed from.
    fn column_share(name: &str, share: f64, share_stderr: f64) -> ColumnCost {
        ColumnCost {
            name: name.to_string(),
            dtype: "Float64".to_string(),
            compressed_bytes: (share * 1_000_000.0) as u64,
            share,
            bytes_per_feature: 1.0,
            marginal_bytes: (share * 1_000_000.0) as u64,
            share_stderr,
            encoding_note: String::new(),
        }
    }

    fn report(
        name: &str,
        compressed: u64,
        per_zoom: Vec<ZoomStats>,
        per_column: Vec<ColumnCost>,
        sampled: bool,
    ) -> InspectReport {
        let tile_count: u64 = per_zoom.iter().map(|z| z.entries).sum();
        InspectReport {
            name: name.to_string(),
            min_zoom: per_zoom.first().map_or(0, |z| z.zoom),
            max_zoom: per_zoom.last().map_or(0, |z| z.zoom),
            time_start_ms: 0,
            time_end_ms: 3_600_000,
            temporal_bucket_ms: 3_600_000,
            tile_count,
            feature_count: tile_count * 10,
            pack_count: 1,
            paged_directory: false,
            compressed_bytes: compressed,
            uncompressed_bytes: compressed * 3,
            compression_ratio: 3.0,
            per_zoom,
            dedup: DedupStats {
                entries: tile_count,
                distinct_blobs: tile_count,
                dedup_ratio: 1.0,
            },
            decode: DecodeStats {
                tiles_decoded: tile_count,
                tiles_total: tile_count,
                sampled,
                features_decoded: tile_count * 10,
                distinct_layer_schemas: 1,
                design: crate::analysis::inspect::SampleDesign::default()
                    .as_str()
                    .to_string(),
                attribution: crate::measure::AttributionDesign::default()
                    .as_str()
                    .to_string(),
            },
            per_column,
        }
    }

    #[test]
    fn diff_totals_zooms_and_columns() {
        // before: z3+z5, geometry+speed columns. after: z5 shrinks 10%, z3 is
        // replaced by z2, speed disappears, quantized `speed_q` appears.
        let before = report(
            "old",
            1_000_000,
            vec![zoom(3, 10, 200_000), zoom(5, 40, 800_000)],
            vec![
                column("geometry", 600_000, 15.0),
                column("speed", 100_000, 2.5),
            ],
            false,
        );
        let after = report(
            "new",
            900_000,
            vec![zoom(2, 4, 180_000), zoom(5, 40, 720_000)],
            vec![
                column("geometry", 600_000, 15.0),
                column("speed_q", 40_000, 1.0),
            ],
            true,
        );

        let d = diff(&before, &after);
        assert_eq!(d.before_name, "old");
        assert_eq!(d.after_name, "new");

        // Totals: absolute + percent.
        assert_eq!(d.compressed_bytes.before, 1_000_000);
        assert_eq!(d.compressed_bytes.after, 900_000);
        assert_eq!(d.compressed_bytes.delta, -100_000);
        assert!((d.compressed_bytes.pct.unwrap() - -10.0).abs() < 1e-9);
        assert_eq!(d.tile_count.delta, -6);
        assert!(d.decode_sampled, "after side sampled → flagged");

        // Per-zoom: union {2, 3, 5}, ascending, one-sided flags + zero fills.
        assert_eq!(
            d.per_zoom.iter().map(|z| z.zoom).collect::<Vec<_>>(),
            vec![2, 3, 5]
        );
        let z2 = &d.per_zoom[0];
        assert_eq!(z2.only_in, Some(Side::After));
        assert_eq!(z2.entries.before, 0);
        assert_eq!(z2.blob_bytes_total.pct, None, "zero baseline → pct n/a");
        let z3 = &d.per_zoom[1];
        assert_eq!(z3.only_in, Some(Side::Before));
        assert_eq!(z3.blob_bytes_total.delta, -200_000);
        let z5 = &d.per_zoom[2];
        assert_eq!(z5.only_in, None);
        assert!((z5.blob_bytes_total.pct.unwrap() - -10.0).abs() < 1e-9);
        assert!((z5.avg_blob_bytes.pct.unwrap() - -10.0).abs() < 1e-9);

        // Per-column: largest side first, one-sided flags both ways.
        assert_eq!(
            d.per_column
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["geometry", "speed", "speed_q"]
        );
        let speed = &d.per_column[1];
        assert_eq!(speed.only_in, Some(Side::Before));
        assert_eq!(speed.compressed_bytes.after, 0);
        assert_eq!(speed.compressed_bytes.delta, -100_000);
        let speed_q = &d.per_column[2];
        assert_eq!(speed_q.only_in, Some(Side::After));
        assert_eq!(speed_q.compressed_bytes.pct, None);
        assert!((speed_q.bytes_per_feature.after - 1.0).abs() < 1e-9);

        // Text rendering carries the headline numbers + one-sided notes.
        let text = format_text(&d);
        assert!(text.contains("old -> new"));
        assert!(text.contains("-10.0%"));
        assert!(text.contains("before only"));
        assert!(text.contains("after only"));
        assert!(text.contains("n/a"));
        assert!(text.contains("sampled decode"));

        // Serde round-trip (snake_case JSON, Side as "before"/"after").
        let json = serde_json::to_string(&d).unwrap();
        assert!(json.contains("\"only_in\":\"after\""));
        let back: DiffReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.per_column.len(), d.per_column.len());
        assert_eq!(back.compressed_bytes.delta, d.compressed_bytes.delta);
    }

    #[test]
    fn identical_reports_diff_to_zero() {
        let r = report(
            "same",
            500_000,
            vec![zoom(4, 8, 500_000)],
            vec![column("geometry", 400_000, 12.0)],
            false,
        );
        let d = diff(&r, &r);
        assert_eq!(d.compressed_bytes.delta, 0);
        assert!((d.compressed_bytes.pct.unwrap()).abs() < 1e-9);
        assert!(!d.decode_sampled);
        assert!(d.per_zoom.iter().all(|z| z.only_in.is_none()));
        assert!(d.per_column.iter().all(|c| c.only_in.is_none()));
        assert!(!format_text(&d).contains("only"));
        // A report diffed against itself moved by exactly zero, which no
        // amount of decode noise can make significant.
        assert_eq!(d.per_column[0].significant, Some(false));
        assert_eq!(d.per_column[0].share.delta, 0.0);
    }

    // ------------------------------------------------------------------
    // MO-2: graded significance beside the binary sampling caveat
    // ------------------------------------------------------------------

    /// Diff two one-column reports whose shares and stderrs are given.
    fn significance(
        before: (f64, f64),
        after: (f64, f64),
        sampled: (bool, bool),
    ) -> (Option<bool>, DiffReport) {
        let b = report(
            "b",
            1_000,
            vec![zoom(5, 1, 1_000)],
            vec![column_share("geometry", before.0, before.1)],
            sampled.0,
        );
        let a = report(
            "a",
            1_000,
            vec![zoom(5, 1, 1_000)],
            vec![column_share("geometry", after.0, after.1)],
            sampled.1,
        );
        let d = diff(&b, &a);
        (d.per_column[0].significant, d)
    }

    #[test]
    fn column_significance_flips_at_the_two_sigma_boundary() {
        // Every constant below is an exact binary fraction, so the boundary is
        // walked in real arithmetic rather than in rounding noise:
        //   σ = 2⁻⁵ = 0.03125 on each side ⇒ noise budget 2·(σ+σ) = 0.125,
        //   and the deltas sit one 2⁻⁸ step either side of it.
        const SIGMA: f64 = 0.03125;
        const BUDGET: f64 = 0.125;
        const STEP: f64 = 0.00390625;
        const BASE: f64 = 0.25;

        let (below, _) = significance((BASE, SIGMA), (BASE + BUDGET - STEP, SIGMA), (true, true));
        assert_eq!(below, Some(false), "just under 2σ is within noise");
        let (above, _) = significance((BASE, SIGMA), (BASE + BUDGET + STEP, SIGMA), (true, true));
        assert_eq!(above, Some(true), "just over 2σ clears the noise floor");
        // Exactly at the boundary the strict `>` keeps it insignificant — the
        // conservative side of the test.
        let (at, _) = significance((BASE, SIGMA), (BASE + BUDGET, SIGMA), (true, true));
        assert_eq!(at, Some(false), "|Δ| == 2σ must NOT be called significant");
        // Sign-symmetric: a shrink of the same size reads the same.
        let (down, _) = significance((BASE + BUDGET + STEP, SIGMA), (BASE, SIGMA), (true, true));
        assert_eq!(down, Some(true));

        // A noisier side widens the budget, so the SAME delta stops being
        // significant. This is what makes tiny archives stop crying wolf.
        let (noisy, _) = significance(
            (BASE, SIGMA),
            (BASE + BUDGET + STEP, 3.0 * SIGMA),
            (true, true),
        );
        assert_eq!(noisy, Some(false));

        // Exhaustive decodes on both sides still publish finite stderr, so the
        // test still applies (and stays conservative) rather than degenerating.
        let (exhaustive, report_ex) =
            significance((BASE, SIGMA), (BASE + 0.25, SIGMA), (false, false));
        assert_eq!(exhaustive, Some(true));
        assert!(
            !report_ex.decode_sampled,
            "the binary caveat stays independent of the graded one"
        );
    }

    #[test]
    fn one_sided_columns_get_no_significance_verdict() {
        let before = report(
            "old",
            1_000,
            vec![zoom(5, 1, 1_000)],
            vec![column_share("geometry", 0.6, 0.01)],
            true,
        );
        let after = report(
            "new",
            1_000,
            vec![zoom(5, 1, 1_000)],
            vec![column_share("payload", 0.6, 0.01)],
            true,
        );
        let d = diff(&before, &after);
        assert!(
            d.per_column.iter().all(|c| c.significant.is_none()),
            "one-sided rows must not fabricate a share delta: {:?}",
            d.per_column
        );
        // …and the one-sided note still wins the note column.
        let text = format_text(&d);
        assert!(text.contains("before only"));
        assert!(text.contains("after only"));
        assert!(!text.contains("within noise"));
    }

    #[test]
    fn the_gate_metric_is_untouched_by_the_graded_annotation() {
        // `compressed_bytes` is the `--fail-on-growth` gate metric and comes
        // from the DIRECTORY on both sides. It must be identical whether or not
        // the per-column rows were annotated, and whether or not either decode
        // was sampled.
        let cols_quiet = vec![column_share("geometry", 0.50, 0.20)];
        let cols_loud = vec![column_share("geometry", 0.50, 0.00)];
        let mk = |cols: Vec<crate::analysis::inspect::ColumnCost>, sampled: bool| {
            report("x", 4_242_424, vec![zoom(5, 10, 4_242_424)], cols, sampled)
        };
        let a = diff(&mk(cols_quiet.clone(), true), &mk(cols_quiet, true));
        let b = diff(&mk(cols_loud.clone(), false), &mk(cols_loud, false));
        assert_eq!(a.compressed_bytes.before, 4_242_424);
        assert_eq!(a.compressed_bytes.delta, 0);
        assert_eq!(
            serde_json::to_string(&a.compressed_bytes).unwrap(),
            serde_json::to_string(&b.compressed_bytes).unwrap(),
            "the gate metric must not vary with decode noise or sampling"
        );
    }

    #[test]
    fn pre_mo2_diff_json_still_deserializes() {
        let (_, d) = significance((0.30, 0.01), (0.40, 0.01), (true, true));
        let mut value = serde_json::to_value(&d).unwrap();
        for row in value["per_column"].as_array_mut().unwrap() {
            let obj = row.as_object_mut().unwrap();
            obj.remove("share").unwrap();
            obj.remove("significant").unwrap();
        }
        let back: DiffReport = serde_json::from_value(value).unwrap();
        assert_eq!(back.per_column[0].significant, None);
        assert_eq!(back.per_column[0].share.delta, 0.0);
        assert_eq!(
            back.compressed_bytes.delta, d.compressed_bytes.delta,
            "the gate metric survives the round trip"
        );
    }
}
