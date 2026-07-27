//! Built-tileset inspection: per-zoom directory stats, per-column compressed
//! byte attribution, dedup and compression ratios.
//!
//! Library port of the `packed-stats` and `point_column_stats` stt-core
//! examples. Directory-derived stats (per-zoom, dedup, wire totals) are always
//! computed over EVERY entry — they cost no payload reads. Only the expensive
//! Arrow decode + per-column re-encode is optionally restricted to a
//! deterministic sample (mirroring `stt-validate --sample` semantics).
//!
//! Per-column attribution re-encodes each decoded column alone (Arrow IPC +
//! zstd-19) to get a fair compressed-byte share. The per-column sum exceeds the
//! whole-tile size (lost cross-column sharing + per-column IPC framing), so it
//! is used for *shares*, not absolute wire accounting. The technique is
//! geometry-agnostic: every column — point/line/polygon geometry included — is
//! a plain Arrow array that a single-column `RecordBatch` can carry.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use serde::{Deserialize, Serialize};
use stt_core::arrow_tile::{DecodedLayer, STT_QUANT_ATTR_META_KEY, STT_QUANT_META_KEY};
use stt_core::compression::compress_zstd_with_dict_level;

use crate::packed::PackedTileset;

/// zstd level for the standalone per-column re-encode. Fixed at the publish
/// level so shares are comparable across datasets regardless of the level
/// their blobs were built with.
const COLUMN_ZSTD_LEVEL: i32 = 19;

/// Directory statistics for one zoom level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomStats {
    /// Zoom level.
    pub zoom: u8,
    /// Directory entries at this zoom.
    pub entries: u64,
    /// Distinct physical blobs referenced (entries sharing a deduped blob
    /// count once). Blob identity is `(pack_id, offset)`.
    pub distinct_blobs: u64,
    /// Sum of compressed blob lengths over ENTRIES (a shared blob counts once
    /// per referencing entry — the bytes a reader streaming this zoom fetches).
    pub blob_bytes_total: u64,
    /// Largest single compressed blob at this zoom.
    pub blob_bytes_max: u64,
    /// `blob_bytes_total / entries`.
    pub avg_blob_bytes: f64,
    /// Distinct temporal buckets (`time_start` values) at this zoom.
    pub t_buckets: u64,
}

/// Entries-vs-blobs dedup accounting over the whole directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupStats {
    /// Total directory entries.
    pub entries: u64,
    /// Distinct physical blobs (`(pack_id, offset)` pairs).
    pub distinct_blobs: u64,
    /// `distinct_blobs / entries` — `1.0` means no dedup, `< 1.0` means
    /// byte-identical tiles were collapsed at build time.
    pub dedup_ratio: f64,
}

/// Decode-pass statistics (the only part of the report that reads payloads).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeStats {
    /// Entries whose payload was decoded.
    pub tiles_decoded: u64,
    /// Total entries (== `InspectReport::tile_count`; here for ratio context).
    pub tiles_total: u64,
    /// True when the decode covered a sampled subset, so a reader never
    /// mistakes sampled per-column numbers for exhaustive ones.
    pub sampled: bool,
    /// Feature rows summed over the decoded tiles' layers.
    pub features_decoded: u64,
    /// Distinct layer-schema signatures across decoded tiles. `> 1` means
    /// producer drift (tiles disagree on columns or types).
    pub distinct_layer_schemas: u64,
}

/// Compressed-byte attribution for one column (merged by name across layers
/// and decoded tiles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnCost {
    /// Column name (e.g. `geometry`, `vertex_time`, a property name).
    pub name: String,
    /// Arrow data type, `Debug`-formatted.
    pub dtype: String,
    /// Standalone re-encode size (IPC + zstd-19) summed over decoded tiles.
    pub compressed_bytes: u64,
    /// `compressed_bytes / Σ all columns' compressed_bytes` — the fair share.
    pub share: f64,
    /// `compressed_bytes / rows` over the batches that carry this column.
    pub bytes_per_feature: f64,
    /// Encoding flag the doctor keys off: `dictionary-encoded`, `quantized
    /// attr (stt:qa)`, `quantized coords (stt:quant)`, `u16 vertex-time
    /// deltas`, `plain f64 (unquantized)` — empty when nothing notable.
    pub encoding_note: String,
}

/// Full inspection report for a packed tileset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectReport {
    /// Dataset name from the manifest metadata.
    pub name: String,
    /// Metadata min zoom.
    pub min_zoom: u8,
    /// Metadata max zoom.
    pub max_zoom: u8,
    /// Time range start (Unix ms).
    pub time_start_ms: u64,
    /// Time range end (Unix ms).
    pub time_end_ms: u64,
    /// Base temporal bucket size (ms).
    pub temporal_bucket_ms: u64,
    /// Directory entry count.
    pub tile_count: u64,
    /// Index-weighted feature total (sum of per-entry counts, all entries).
    pub feature_count: u64,
    /// Pack objects in the manifest.
    pub pack_count: u64,
    /// Whether the directory ships paged (reporting only; reads are identical).
    pub paged_directory: bool,
    /// Sum of compressed blob lengths over all entries (directory, total).
    pub compressed_bytes: u64,
    /// Sum of uncompressed payload sizes over all entries (directory, total).
    pub uncompressed_bytes: u64,
    /// `uncompressed_bytes / compressed_bytes`.
    pub compression_ratio: f64,
    /// Per-zoom directory stats (always total).
    pub per_zoom: Vec<ZoomStats>,
    /// Whole-directory dedup accounting (always total).
    pub dedup: DedupStats,
    /// Decode-pass stats (sampled when `sample` was given).
    pub decode: DecodeStats,
    /// Per-column compressed-cost attribution, largest first (from the same
    /// decoded subset as `decode`).
    pub per_column: Vec<ColumnCost>,
}

/// Deterministic stride for sampling: pick every `ceil(total/n)`-th entry
/// starting at index 0, yielding at most `n` evenly-spread tiles. Same
/// semantics as `stt-validate --sample`: reproducible across runs, no
/// randomness. Callers guard `n == 0` (decode nothing).
fn sample_stride(total: usize, n: usize) -> usize {
    total.div_ceil(n).max(1)
}

/// Schema signature for producer-drift detection: layer name + every field's
/// `name:type`, sorted so layer order can't alias two identical schemas.
fn schema_signature(layers: &[DecodedLayer]) -> String {
    let mut parts: Vec<String> = layers
        .iter()
        .map(|layer| {
            let cols: Vec<String> = layer
                .batch
                .schema()
                .fields()
                .iter()
                .map(|f| format!("{}:{:?}", f.name(), f.data_type()))
                .collect();
            format!("{}{{{}}}", layer.name, cols.join(","))
        })
        .collect();
    parts.sort();
    parts.join("|")
}

/// Does `dt` contain `Float64` anywhere in its (possibly nested) type tree?
fn contains_f64(dt: &DataType) -> bool {
    match dt {
        DataType::Float64 => true,
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => {
            contains_f64(f.data_type())
        }
        DataType::Dictionary(_, v) => contains_f64(v),
        _ => false,
    }
}

/// Does `dt` contain `needle` as a leaf type?
fn contains_leaf(dt: &DataType, needle: &DataType) -> bool {
    if dt == needle {
        return true;
    }
    match dt {
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => {
            contains_leaf(f.data_type(), needle)
        }
        _ => false,
    }
}

/// Derive the encoding flag for a field — the "smells" the recommendation
/// pass keys off. Empty when nothing notable.
fn encoding_note(field: &Field) -> String {
    if field.metadata().contains_key(STT_QUANT_META_KEY) {
        return "quantized coords (stt:quant)".to_string();
    }
    if field.metadata().contains_key(STT_QUANT_ATTR_META_KEY) {
        return "quantized attr (stt:qa)".to_string();
    }
    if matches!(field.data_type(), DataType::Dictionary(_, _)) {
        return "dictionary-encoded".to_string();
    }
    if field.name() == "vertex_time" {
        // Narrowest-first delta ladder, then the absolute fallback. Without the
        // u32 rung a multi-day track dataset fell through to the generic f64
        // note (or to an empty string), mislabelling the encoding it is on.
        if contains_leaf(field.data_type(), &DataType::UInt16) {
            return "u16 vertex-time deltas".to_string();
        }
        if contains_leaf(field.data_type(), &DataType::UInt32) {
            return "u32 vertex-time deltas".to_string();
        }
        if contains_leaf(field.data_type(), &DataType::Int64) {
            return "i64 absolute vertex-time".to_string();
        }
    }
    if contains_f64(field.data_type()) {
        return "plain f64 (unquantized)".to_string();
    }
    String::new()
}

/// Re-encode a batch standalone (Arrow IPC stream + zstd-19) and return the
/// compressed length — the fair-share cost unit for column attribution.
fn ipc_zstd_len(batch: &RecordBatch) -> Result<u64> {
    let mut buf = Vec::new();
    {
        let mut w =
            StreamWriter::try_new(&mut buf, &batch.schema()).context("column IPC writer init")?;
        w.write(batch).context("column IPC write")?;
        w.finish().context("column IPC finish")?;
    }
    Ok(compress_zstd_with_dict_level(&buf, None, COLUMN_ZSTD_LEVEL)?.len() as u64)
}

/// Inspect a packed tileset.
///
/// `sample`: `None` decodes every tile; `Some(n)` decodes a deterministic,
/// evenly-spread sample of at most `n` tiles (every `ceil(total/n)`-th
/// directory entry, starting at 0 — `stt-validate --sample` semantics);
/// `Some(0)` skips the decode pass entirely. Directory-derived stats
/// (`per_zoom`, `dedup`, the wire totals) are ALWAYS computed over all
/// entries — only the decode-based stats (`decode`, `per_column`) sample.
pub fn inspect(tileset: &PackedTileset, sample: Option<usize>) -> Result<InspectReport> {
    let entries = tileset.entries();
    let meta = tileset.metadata();

    // --- Directory pass: always total, no payload reads --------------------
    #[derive(Default)]
    struct ZoomAcc {
        entries: u64,
        blobs: BTreeSet<(u32, u64)>,
        bytes_total: u64,
        bytes_max: u64,
        t_starts: BTreeSet<i64>,
    }
    let mut per_zoom: BTreeMap<u8, ZoomAcc> = BTreeMap::new();
    let mut all_blobs: BTreeSet<(u32, u64)> = BTreeSet::new();
    let mut compressed_bytes = 0u64;
    let mut uncompressed_bytes = 0u64;
    let mut feature_count = 0u64;
    for e in entries {
        let z = per_zoom.entry(e.zoom).or_default();
        z.entries += 1;
        z.blobs.insert((e.pack_id, e.offset));
        z.bytes_total += e.length as u64;
        z.bytes_max = z.bytes_max.max(e.length as u64);
        z.t_starts.insert(e.time_start);
        all_blobs.insert((e.pack_id, e.offset));
        compressed_bytes += e.length as u64;
        uncompressed_bytes += e.uncompressed_size as u64;
        feature_count += e.feature_count as u64;
    }
    let per_zoom: Vec<ZoomStats> = per_zoom
        .into_iter()
        .map(|(zoom, z)| ZoomStats {
            zoom,
            entries: z.entries,
            distinct_blobs: z.blobs.len() as u64,
            blob_bytes_total: z.bytes_total,
            blob_bytes_max: z.bytes_max,
            avg_blob_bytes: z.bytes_total as f64 / z.entries.max(1) as f64,
            t_buckets: z.t_starts.len() as u64,
        })
        .collect();
    let dedup = DedupStats {
        entries: entries.len() as u64,
        distinct_blobs: all_blobs.len() as u64,
        dedup_ratio: all_blobs.len() as f64 / entries.len().max(1) as f64,
    };

    // --- Decode pass: sampled when requested --------------------------------
    #[derive(Default)]
    struct ColAcc {
        dtype: String,
        note: String,
        compressed: u64,
        rows: u64,
    }
    let mut cols: BTreeMap<String, ColAcc> = BTreeMap::new();
    let mut schemas: BTreeSet<String> = BTreeSet::new();
    let mut tiles_decoded = 0u64;
    let mut features_decoded = 0u64;
    let stride = sample.map(|n| {
        if n == 0 {
            usize::MAX
        } else {
            sample_stride(entries.len(), n)
        }
    });
    for (idx, e) in entries.iter().enumerate() {
        let decode_this = match stride {
            None => true,
            Some(usize::MAX) => false,
            Some(s) => idx % s == 0,
        };
        if !decode_this {
            continue;
        }
        let layers = tileset.read_layers(e).with_context(|| {
            format!(
                "decoding tile z{}/{}/{} t{}",
                e.zoom, e.x, e.y, e.time_start
            )
        })?;
        tiles_decoded += 1;
        schemas.insert(schema_signature(&layers));
        for layer in &layers {
            let batch = &layer.batch;
            let rows = batch.num_rows() as u64;
            features_decoded += rows;
            let schema = batch.schema();
            for (i, field) in schema.fields().iter().enumerate() {
                // Strip field metadata from the standalone re-encode: Arrow IPC
                // serializes the metadata HashMap in nondeterministic order, so
                // keeping it would make repeated inspections disagree by a few
                // bytes. The metadata is negligible for shares; the encoding
                // note (derived from the ORIGINAL field below) preserves it.
                let clean = field.as_ref().clone().with_metadata(Default::default());
                let one = RecordBatch::try_new(
                    Arc::new(Schema::new(vec![clean])),
                    vec![batch.column(i).clone()],
                )
                .context("single-column batch")?;
                let c = cols.entry(field.name().clone()).or_default();
                c.compressed += ipc_zstd_len(&one)?;
                c.rows += rows;
                c.dtype = format!("{:?}", field.data_type());
                c.note = encoding_note(field);
            }
        }
    }
    let col_total: u64 = cols.values().map(|c| c.compressed).sum();
    let mut per_column: Vec<ColumnCost> = cols
        .into_iter()
        .map(|(name, c)| ColumnCost {
            name,
            dtype: c.dtype,
            compressed_bytes: c.compressed,
            share: c.compressed as f64 / col_total.max(1) as f64,
            bytes_per_feature: c.compressed as f64 / c.rows.max(1) as f64,
            encoding_note: c.note,
        })
        .collect();
    per_column.sort_by(|a, b| b.compressed_bytes.cmp(&a.compressed_bytes));

    let time_range = tileset.time_range();
    Ok(InspectReport {
        name: tileset.name().to_string(),
        min_zoom: meta.min_zoom,
        max_zoom: meta.max_zoom,
        time_start_ms: time_range.start,
        time_end_ms: time_range.end,
        temporal_bucket_ms: meta.temporal_bucket_ms,
        tile_count: entries.len() as u64,
        feature_count,
        pack_count: tileset.pack_count() as u64,
        paged_directory: tileset.is_paged(),
        compressed_bytes,
        uncompressed_bytes,
        compression_ratio: uncompressed_bytes as f64 / compressed_bytes.max(1) as f64,
        per_zoom,
        dedup,
        decode: DecodeStats {
            tiles_decoded,
            tiles_total: entries.len() as u64,
            sampled: stride.is_some(),
            features_decoded,
            distinct_layer_schemas: schemas.len() as u64,
        },
        per_column,
    })
}

/// Render the report as compact aligned text.
pub fn format_text(report: &InspectReport) -> String {
    let mut out = String::new();
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    out.push_str(&format!("         STT Inspect - {}\n", report.name));
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    out.push_str("📊 Dataset\n");
    out.push_str(&format!(
        "  Tiles: {}   Features (index): {}   Zoom: {}-{}\n",
        report.tile_count, report.feature_count, report.min_zoom, report.max_zoom
    ));
    out.push_str(&format!(
        "  Time: {}..{} ms   Base bucket: {} ms\n",
        report.time_start_ms, report.time_end_ms, report.temporal_bucket_ms
    ));
    out.push_str(&format!(
        "  Packs: {}   Directory: {}\n",
        report.pack_count,
        if report.paged_directory {
            "paged"
        } else {
            "single"
        }
    ));
    out.push_str(&format!(
        "  Wire: {:.2} MB compressed -> {:.2} MB decoded ({:.2}x)\n\n",
        report.compressed_bytes as f64 / 1e6,
        report.uncompressed_bytes as f64 / 1e6,
        report.compression_ratio
    ));

    out.push_str("🗂  Per-zoom directory\n");
    out.push_str("  zoom |  entries | distinct |  total MB |  max KB |  avg KB | t-buckets\n");
    for z in &report.per_zoom {
        out.push_str(&format!(
            "    {:2} | {:8} | {:8} | {:9.2} | {:7.1} | {:7.1} | {:9}\n",
            z.zoom,
            z.entries,
            z.distinct_blobs,
            z.blob_bytes_total as f64 / 1e6,
            z.blob_bytes_max as f64 / 1e3,
            z.avg_blob_bytes / 1e3,
            z.t_buckets
        ));
    }
    out.push_str(&format!(
        "  dedup: {} entries -> {} distinct blobs (ratio {:.3})\n\n",
        report.dedup.entries, report.dedup.distinct_blobs, report.dedup.dedup_ratio
    ));

    out.push_str(&format!(
        "🔬 Decode ({} of {} tiles{})\n",
        report.decode.tiles_decoded,
        report.decode.tiles_total,
        if report.decode.sampled {
            ", sampled"
        } else {
            ""
        }
    ));
    out.push_str(&format!(
        "  features decoded: {}   distinct layer schemas: {}\n\n",
        report.decode.features_decoded, report.decode.distinct_layer_schemas
    ));

    if !report.per_column.is_empty() {
        out.push_str("💾 Per-column cost (standalone IPC+zstd-19; shares, not absolute wire)\n");
        out.push_str(&format!(
            "  {:<22} {:<28} {:>10} {:>9} {:>7}  note\n",
            "column", "dtype", "comp KB", "B/feat", "share%"
        ));
        for c in &report.per_column {
            let dt = if c.dtype.len() > 27 {
                format!("{}…", &c.dtype[..26])
            } else {
                c.dtype.clone()
            };
            out.push_str(&format!(
                "  {:<22} {:<28} {:>10.1} {:>9.2} {:>6.1}%  {}\n",
                c.name,
                dt,
                c.compressed_bytes as f64 / 1e3,
                c.bytes_per_feature,
                100.0 * c.share,
                c.encoding_note
            ));
        }
    }

    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::arrow_tile::{
        encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn, PropertyColumn,
    };
    use stt_core::curve::BlobOrdering;
    use stt_core::metadata::Metadata;
    use stt_core::pack::PackWriter;
    use stt_core::tile::TileId;

    /// A line layer with vertex times (small deltas → u16 encoding), one f64
    /// property and one categorical (dictionary) property — enough surface to
    /// exercise the geometry-agnostic column attribution + encoding notes.
    fn line_layer(seed: u64, n: usize) -> ColumnarLayer {
        let verts_per = 8usize;
        let geometry: Vec<Vec<[f64; 2]>> = (0..n)
            .map(|i| {
                (0..verts_per)
                    .map(|v| {
                        [
                            -73.6 + (seed as f64) * 0.01 + v as f64 * 0.001,
                            45.5 + i as f64 * 0.002,
                        ]
                    })
                    .collect()
            })
            .collect();
        let vertex_times: Vec<Vec<i64>> = (0..n)
            .map(|_| (0..verts_per).map(|v| v as i64 * 50).collect())
            .collect();
        ColumnarLayer {
            polygon_parts: None,
            name: "default".to_string(),
            feature_ids: (0..n as u64).map(|i| seed * 1000 + i).collect(),
            start_times: vec![0; n],
            end_times: vec![400; n],
            geometry: GeometryColumn::LineString(geometry),
            vertex_times: Some(vertex_times),
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "speed".to_string(),
                    PropertyColumn::Numeric((0..n).map(|i| Some(i as f64 * 1.5)).collect()),
                ),
                (
                    "kind".to_string(),
                    PropertyColumn::Categorical(
                        (0..n)
                            .map(|i| Some(["bike", "ferry"][i % 2].to_string()))
                            .collect(),
                    ),
                ),
            ],
        }
    }

    /// Build a real tiny packed tileset: 3 line tiles at z5 (one payload
    /// duplicated across two entries → dedup) + 1 at z3, two time buckets.
    /// Frames ride the writer's (default v2) format version + template
    /// collector so the fixture is version-coherent.
    fn build_fixture(out: &std::path::Path) {
        let mut w = PackWriter::create(out, BlobOrdering::Auto, 64 * 1024).unwrap();
        let cfg = EncoderConfig {
            format_version: w.format_version(),
            template_collector: Some(w.template_collector()),
            ..EncoderConfig::default()
        };
        let bucket = 3_600_000i64;
        let dup = encode_tile_with(&[line_layer(7, 40)], &cfg).unwrap();
        // z5: two entries sharing the SAME payload bytes (different cells) +
        // one distinct, across two time buckets.
        w.add_tile_full(
            &TileId::new(5, 1, 1, 0),
            0,
            bucket - 1,
            Some(0),
            40,
            Some(bucket as u64),
            &dup,
        )
        .unwrap();
        w.add_tile_full(
            &TileId::new(5, 2, 1, bucket as u64),
            bucket,
            2 * bucket - 1,
            Some(bucket),
            40,
            Some(bucket as u64),
            &dup,
        )
        .unwrap();
        let distinct = encode_tile_with(&[line_layer(9, 40)], &cfg).unwrap();
        w.add_tile_full(
            &TileId::new(5, 3, 1, 0),
            0,
            bucket - 1,
            Some(0),
            40,
            Some(bucket as u64),
            &distinct,
        )
        .unwrap();
        // z3 overview tile.
        let overview = encode_tile_with(&[line_layer(11, 40)], &cfg).unwrap();
        w.add_tile_full(
            &TileId::new(3, 0, 0, 0),
            0,
            bucket - 1,
            Some(0),
            40,
            Some(bucket as u64),
            &overview,
        )
        .unwrap();
        let meta = Metadata::new("inspect-fixture")
            .with_temporal_bucket_ms(bucket as u64)
            .with_zoom_levels(3, 5);
        w.finalize(&meta).unwrap();
    }

    #[test]
    fn inspect_full_report_on_real_fixture() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);

        let ts = PackedTileset::open(&out).unwrap();
        let report = inspect(&ts, None).unwrap();

        // Per-zoom directory stats (always total).
        assert_eq!(report.tile_count, 4);
        assert_eq!(report.per_zoom.len(), 2);
        let z3 = &report.per_zoom[0];
        let z5 = &report.per_zoom[1];
        assert_eq!(
            (z3.zoom, z3.entries, z3.distinct_blobs, z3.t_buckets),
            (3, 1, 1, 1)
        );
        assert_eq!((z5.zoom, z5.entries, z5.t_buckets), (5, 3, 2));
        // The duplicated payload collapses: 3 entries, 2 physical blobs.
        assert_eq!(z5.distinct_blobs, 2);
        assert!(z5.blob_bytes_max > 0);
        assert!((z5.avg_blob_bytes - z5.blob_bytes_total as f64 / 3.0).abs() < 1e-9);

        // Dedup over the whole directory: 4 entries, 3 distinct blobs.
        assert_eq!(report.dedup.entries, 4);
        assert_eq!(report.dedup.distinct_blobs, 3);
        assert!(report.dedup.dedup_ratio < 1.0);

        // Wire totals from the directory; real zstd must beat 1x on this data.
        assert!(
            report.compression_ratio > 1.0,
            "ratio {}",
            report.compression_ratio
        );
        assert!(report.compressed_bytes > 0 && report.uncompressed_bytes > report.compressed_bytes);

        // Full (unsampled) decode.
        assert!(!report.decode.sampled);
        assert_eq!(report.decode.tiles_decoded, 4);
        assert_eq!(report.decode.features_decoded, 160);
        assert_eq!(report.decode.distinct_layer_schemas, 1);
        assert_eq!(report.feature_count, 160);

        // Per-column attribution generalizes to line geometry: shares sum to
        // ~1.0 and every expected column is present.
        let share_sum: f64 = report.per_column.iter().map(|c| c.share).sum();
        assert!((share_sum - 1.0).abs() < 1e-9, "shares sum to {share_sum}");
        let by_name = |n: &str| {
            report
                .per_column
                .iter()
                .find(|c| c.name == n)
                .unwrap_or_else(|| panic!("column {n} missing"))
        };
        for name in ["geometry", "vertex_time", "speed", "kind", "id"] {
            assert!(by_name(name).compressed_bytes > 0);
            assert!(by_name(name).bytes_per_feature > 0.0);
        }

        // Encoding notes: the doctor's smells.
        assert_eq!(by_name("geometry").encoding_note, "plain f64 (unquantized)");
        assert_eq!(by_name("speed").encoding_note, "plain f64 (unquantized)");
        assert_eq!(by_name("kind").encoding_note, "dictionary-encoded");
        assert_eq!(
            by_name("vertex_time").encoding_note,
            "u16 vertex-time deltas"
        );

        // Text rendering carries the headline numbers.
        let text = format_text(&report);
        assert!(text.contains("inspect-fixture"));
        assert!(text.contains("geometry"));
        assert!(text.contains("dedup: 4 entries -> 3 distinct blobs"));
        assert!(!text.contains("sampled"));
    }

    #[test]
    fn inspect_sampled_decode_keeps_directory_stats_total() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);
        let ts = PackedTileset::open(&out).unwrap();

        // sample=2 over 4 entries → stride 2 → exactly entries 0 and 2 decode.
        let report = inspect(&ts, Some(2)).unwrap();
        assert!(report.decode.sampled);
        assert_eq!(report.decode.tiles_decoded, 2);
        assert_eq!(report.decode.features_decoded, 80);
        // Directory stats stay total despite the sampled decode.
        assert_eq!(report.tile_count, 4);
        assert_eq!(report.dedup.entries, 4);
        assert_eq!(report.dedup.distinct_blobs, 3);
        assert_eq!(report.per_zoom.iter().map(|z| z.entries).sum::<u64>(), 4);
        // Shares still normalize over the sampled subset.
        let share_sum: f64 = report.per_column.iter().map(|c| c.share).sum();
        assert!((share_sum - 1.0).abs() < 1e-9);
        // Deterministic: a rerun samples the same tiles.
        let rerun = inspect(&ts, Some(2)).unwrap();
        assert_eq!(
            rerun
                .per_column
                .iter()
                .map(|c| (c.name.clone(), c.compressed_bytes))
                .collect::<Vec<_>>(),
            report
                .per_column
                .iter()
                .map(|c| (c.name.clone(), c.compressed_bytes))
                .collect::<Vec<_>>()
        );
        assert!(format_text(&report).contains("sampled"));

        // sample=0 decodes nothing; sample >= total decodes everything.
        let none = inspect(&ts, Some(0)).unwrap();
        assert_eq!(none.decode.tiles_decoded, 0);
        assert!(none.per_column.is_empty());
        assert_eq!(none.dedup.entries, 4);
        let all = inspect(&ts, Some(100)).unwrap();
        assert_eq!(all.decode.tiles_decoded, 4);
        assert!(all.decode.sampled);
    }

    #[test]
    fn report_serializes_to_json_and_back() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);
        let ts = PackedTileset::open(&out).unwrap();
        let report = inspect(&ts, None).unwrap();

        let json = serde_json::to_string_pretty(&report).unwrap();
        let back: InspectReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tile_count, report.tile_count);
        assert_eq!(back.per_column.len(), report.per_column.len());
        assert_eq!(back.dedup.distinct_blobs, report.dedup.distinct_blobs);
    }
}
