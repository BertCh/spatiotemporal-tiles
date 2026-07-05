//! Tileset-to-tileset comparison for `stt-optimize diff`: totals, per-zoom
//! directory stats, and per-column costs between two [`InspectReport`]s
//! (typically before/after a re-encode or fleet reprocess), each as absolute
//! and percent deltas.
//!
//! Directory-derived numbers (tile/blob counts, wire bytes) are exact on both
//! sides; per-column numbers inherit whatever sampling the two inspections
//! ran with ([`DiffReport::decode_sampled`] flags this), so treat them as
//! attribution shifts, not absolute wire accounting.

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
    /// Standalone re-encode bytes (IPC + zstd) over the decoded tiles.
    pub compressed_bytes: Delta,
    /// Standalone re-encode bytes per feature row.
    pub bytes_per_feature: DeltaF,
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
            "  {:<22} {:>11} {:>11} {:>11} {:>8}  {:>7} -> {:<7}  note\n",
            "column", "before KB", "after KB", "delta KB", "pct", "B/feat", "B/feat"
        ));
        for c in &report.per_column {
            out.push_str(&format!(
                "  {:<22} {:>11.1} {:>11.1} {:>+11.1} {:>8}  {:>7.2} -> {:<7.2}  {}\n",
                c.name,
                c.compressed_bytes.before as f64 / 1e3,
                c.compressed_bytes.after as f64 / 1e3,
                c.compressed_bytes.delta as f64 / 1e3,
                fmt_pct(c.compressed_bytes.pct),
                c.bytes_per_feature.before,
                c.bytes_per_feature.after,
                one_sided_note(c.only_in)
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
            vec![column("geometry", 600_000, 15.0), column("speed", 100_000, 2.5)],
            false,
        );
        let after = report(
            "new",
            900_000,
            vec![zoom(2, 4, 180_000), zoom(5, 40, 720_000)],
            vec![column("geometry", 600_000, 15.0), column("speed_q", 40_000, 1.0)],
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
            d.per_column.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
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
    }
}
