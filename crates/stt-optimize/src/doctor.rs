//! Tileset doctor: severity-ranked findings over a built (packed) tileset,
//! each with a concrete remediation flag and a projection derived from the
//! inspect report's measured per-column costs.
//!
//! Productizes the manual optimization passes this repo keeps re-running by
//! hand (`point_column_stats` → `--quantize-attr`, the id-column hunt, the
//! paged-directory migration, the summary-tier retrofit). Every rule keys off
//! numbers already **measured** for THIS tileset — directory entries from
//! [`PackedTileset`] and per-column compressed costs from [`InspectReport`] —
//! and cites them in its message. Projections are estimated from those
//! measured column shares; the doctor never re-encodes anything, so every
//! `projected` string carries the "(estimated from measured column costs)"
//! label.
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

use anyhow::{Context, Result};
use arrow::array::{Array, ArrayData, RecordBatch};
use arrow::datatypes::DataType;
use serde::{Deserialize, Serialize};

use crate::analysis::inspect::InspectReport;
use crate::packed::PackedTileset;

/// Label appended to every `projected` string: doctor deltas are derived from
/// the inspect report's measured per-column shares, never from a re-encode.
const ESTIMATE_LABEL: &str = "(estimated from measured column costs)";

/// Must match [`crate::analysis::inspect`]'s `encoding_note` for an
/// unquantized, non-dictionary Float64 column — the smell `raw-f64-column`
/// keys off.
const PLAIN_F64_NOTE: &str = "plain f64 (unquantized)";

/// Core (non-property) tile columns the property rules skip: they are either
/// structural or covered by their own levers (`--quantize-coords` for
/// geometry, the vertex-time delta encoding), not by `--quantize-attr`.
const RESERVED_COLUMNS: [&str; 8] = [
    "id",
    "start_time",
    "end_time",
    "geometry",
    "vertex_time",
    "vertex_value",
    "vertex_value_matrix",
    "triangles",
];

/// `raw-f64-column` only flags columns carrying at least this share of the
/// measured column cost — below it the projected win is noise.
const RAW_F64_MIN_SHARE: f64 = 0.03;
/// Combined flagged share at which `raw-f64-column` escalates to Critical
/// (the Waymo case: id + z alone were ~78% of the dataset).
const RAW_F64_CRITICAL_SHARE: f64 = 0.5;
/// Assumed shrink of a flagged column once quantized (measured passes in this
/// repo landed 50–75%; 60% is the conservative middle).
const RAW_F64_SHRINK: f64 = 0.6;

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
    /// Projected effect of the remediation, always labeled
    /// "(estimated from measured column costs)". `None` when no meaningful
    /// projection can be derived without re-encoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projected: Option<String>,
}

/// Full doctor report: findings sorted severity-first, then by code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    /// All findings, most severe first (empty = healthy).
    pub findings: Vec<Finding>,
}

/// Run every doctor rule over a packed tileset + its inspect report.
///
/// Directory- and column-cost rules read only the (already computed) report;
/// the dead-column and point-dominance checks decode up to
/// [`DOCTOR_SAMPLE_TILES`] stride-sampled tiles via
/// [`PackedTileset::read_layers`]. Findings come back sorted severity-first,
/// then by code, then message, so output is deterministic.
pub fn doctor(tileset: &PackedTileset, report: &InspectReport) -> Result<DoctorReport> {
    let sampled = sample_decode(tileset)?;

    let mut findings = Vec::new();
    if let Some(f) = rule_raw_f64_columns(report) {
        findings.push(f);
    }
    if let Some(f) = rule_expensive_feature_ids(report) {
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

    findings.sort_by(|a, b| {
        a.severity
            .cmp(&b.severity)
            .then_with(|| a.code.cmp(&b.code))
            .then_with(|| a.message.cmp(&b.message))
    });
    Ok(DoctorReport { findings })
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

/// Decode up to [`DOCTOR_SAMPLE_TILES`] evenly-strided tiles (same
/// deterministic stride semantics as `inspect`/`stt-validate --sample`) and
/// fold per-column constancy + geometry-kind row counts.
fn sample_decode(tileset: &PackedTileset) -> Result<SampledDecode> {
    let entries = tileset.entries();
    let mut out = SampledDecode {
        tiles_decoded: 0,
        tiles_total: entries.len(),
        total_rows: 0,
        point_rows: 0,
        columns: BTreeMap::new(),
    };
    if entries.is_empty() {
        return Ok(out);
    }
    let stride = entries.len().div_ceil(DOCTOR_SAMPLE_TILES).max(1);
    for e in entries.iter().step_by(stride) {
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
// Rules
// ----------------------------------------------------------------------------

/// `raw-f64-column`: property columns shipping as plain Float64 (not
/// quantized, not dictionary) with a measured share ≥ 3%. One finding listing
/// every such column, worst first. Geometry is deliberately out of scope —
/// its lever is `--quantize-coords`, not `--quantize-attr`.
fn rule_raw_f64_columns(report: &InspectReport) -> Option<Finding> {
    let mut flagged: Vec<_> = report
        .per_column
        .iter()
        .filter(|c| {
            !RESERVED_COLUMNS.contains(&c.name.as_str())
                && c.encoding_note == PLAIN_F64_NOTE
                && c.share >= RAW_F64_MIN_SHARE
        })
        .collect();
    if flagged.is_empty() {
        return None;
    }
    flagged.sort_by(|a, b| b.share.total_cmp(&a.share));

    let total_share: f64 = flagged.iter().map(|c| c.share).sum();
    let listed = flagged
        .iter()
        .map(|c| {
            format!(
                "`{}` ({:.1}% of column bytes, {:.2} B/feature)",
                c.name,
                100.0 * c.share,
                c.bytes_per_feature
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let saved_bytes = report.compressed_bytes as f64 * total_share * RAW_F64_SHRINK;
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
        projected: Some(format!(
            "~{:.0}% smaller dataset wire (~{:.2} of {:.2} MB) after quantizing the flagged \
             columns, assuming ~{:.0}% per-column shrink {}",
            100.0 * total_share * RAW_F64_SHRINK,
            saved_bytes / 1e6,
            report.compressed_bytes as f64 / 1e6,
            100.0 * RAW_F64_SHRINK,
            ESTIMATE_LABEL
        )),
    })
}

/// `expensive-feature-ids`: the `id` column measured above 4 B/feature —
/// hash-like or explicit source ids carry full entropy per row, so zstd
/// cannot shrink them, while builder-assigned sequential ids compress to
/// ~1 B/feature.
fn rule_expensive_feature_ids(report: &InspectReport) -> Option<Finding> {
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
        projected: Some(format!(
            "up to ~{:.1}% of dataset wire reclaimable from the id column {}",
            100.0 * id.share,
            ESTIMATE_LABEL
        )),
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
            }
        })
        .collect()
}

/// `z0-bomb`: a shallow pyramid (min_zoom ≤ 4) under bounds spanning less
/// than 2° in both axes — every zoom below the suggested floor re-ships the
/// whole dataset in one or two near-duplicate tiles.
fn rule_z0_bomb(tileset: &PackedTileset, report: &InspectReport) -> Option<Finding> {
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
    if floor <= report.min_zoom {
        return None;
    }
    let (shallow_tiles, shallow_bytes) = report
        .per_zoom
        .iter()
        .filter(|z| z.zoom < floor)
        .fold((0u64, 0u64), |(t, b), z| {
            (t + z.entries, b + z.blob_bytes_total)
        });
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
        projected: None,
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
    })
}

/// `oversized-blobs`: directory entries whose compressed blob exceeds 1 MiB.
/// Structural fixes (zoom floor, summary tier) come first; the opt-in budgets
/// drop data and are never presented as the default fix.
fn rule_oversized_blobs(tileset: &PackedTileset, report: &InspectReport) -> Option<Finding> {
    let over: Vec<_> = tileset
        .entries()
        .iter()
        .filter(|e| u64::from(e.length) > OVERSIZED_BLOB_BYTES)
        .collect();
    let worst = over.iter().max_by_key(|e| e.length)?;
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
        projected: None,
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
    })
}

// ----------------------------------------------------------------------------
// Text rendering
// ----------------------------------------------------------------------------

/// Render the report as compact aligned text (severity-ranked, one block per
/// finding).
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
    /// the exact pattern of the packed.rs/inspect.rs test fixtures).
    fn build(out: &std::path::Path, spec: &FixtureSpec) {
        let cfg = EncoderConfig {
            quantize_attrs: if spec.quantize_magnitude {
                [("magnitude".to_string(), 0.01)].into_iter().collect()
            } else {
                Default::default()
            },
            ..Default::default()
        };
        let mut w = PackWriter::create(out, BlobOrdering::Auto, 64 * 1024)
            .unwrap()
            .with_paging(spec.paged);
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
        assert!(projected.contains("(estimated from measured column costs)"));

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
        assert!(f
            .projected
            .as_deref()
            .unwrap()
            .contains("(estimated from measured column costs)"));
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

        // Findings come out sorted: severity first, then code.
        let ranks: Vec<_> = report
            .findings
            .iter()
            .map(|f| (f.severity, f.code.clone()))
            .collect();
        let mut sorted = ranks.clone();
        sorted.sort();
        assert_eq!(ranks, sorted, "findings not severity/code sorted");
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

        let empty = format_text(&DoctorReport { findings: vec![] });
        assert!(empty.contains("No findings"));
    }
}
