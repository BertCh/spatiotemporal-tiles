//! Built-tileset inspection: per-zoom directory stats, per-column compressed
//! byte attribution, dedup and compression ratios.
//!
//! Library port of the `packed-stats` and `point_column_stats` stt-core
//! examples. Directory-derived stats (per-zoom, dedup, wire totals) are always
//! computed over EVERY entry — they cost no payload reads. Only the expensive
//! Arrow decode + per-column re-encode is optionally restricted to a
//! deterministic sample (mirroring `stt-validate --sample` semantics).
//!
//! Per-column attribution is LEAVE-ONE-OUT (the shared
//! [`crate::measure::attribution`] module): the decoded batch is re-encoded
//! whole (Arrow IPC + zstd-19) and then once per column with that column
//! removed, and the column is charged the difference. That is the post-zstd
//! bytes the tile actually sheds if the column vanishes, so the per-column sum
//! fits inside the whole-tile size and duplicated columns are no longer
//! double-charged — the failure of the old singleton proxy, which re-encoded
//! each column ALONE and therefore over-spent the tile budget. The proxy stays
//! reachable as [`AttributionDesign::SingletonV1`] for one release. The
//! technique is geometry-agnostic: every column — point/line/polygon geometry
//! included — is a plain Arrow array.
//!
//! Every published share carries a `share_stderr` beside it: the tile-to-tile
//! dispersion of that share across the decoded tiles, so threshold consumers
//! (the doctor's rules, `diff`'s significance annotation) can tell a real
//! difference from decode noise. It is finite even for an exhaustive decode —
//! see [`ShareDispersion`] for the interpretation.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use arrow::datatypes::{DataType, Field};
use serde::{Deserialize, Serialize};
use stt_core::arrow_tile::{DecodedLayer, STT_QUANT_ATTR_META_KEY, STT_QUANT_META_KEY};
use stt_core::TileEntry;

use crate::measure::attribution::{attribute_columns, AttributionDesign, ShareDispersion};
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
    /// Which sampling design selected the decoded tiles — `"stride-v1"` (the
    /// legacy flat systematic stride) or `"stratified-v2"` (the default:
    /// byte-proportional strata over `(zoom, time_start)`). Reports written
    /// before the discriminator existed deserialize as `"stride-v1"`, which is
    /// what they in fact used.
    #[serde(default = "legacy_sample_design")]
    pub design: String,
    /// Which per-column attribution design produced `per_column` —
    /// `"loo-v2"` (the default: leave-one-out marginals) or `"singleton-v1"`
    /// (the rollback: each column re-encoded alone). The two are NOT
    /// numerically comparable, so a consumer comparing two reports must check
    /// this first. Reports written before the discriminator existed
    /// deserialize as `"singleton-v1"`, which is what they in fact used.
    #[serde(default = "legacy_attribution_design")]
    pub attribution: String,
}

/// serde default for [`DecodeStats::design`]: reports predating the field were
/// all produced by the flat stride.
fn legacy_sample_design() -> String {
    SampleDesign::StrideV1.as_str().to_string()
}

/// serde default for [`DecodeStats::attribution`]: reports predating the field
/// were all produced by the singleton proxy.
fn legacy_attribution_design() -> String {
    AttributionDesign::SingletonV1.as_str().to_string()
}

/// Compressed-byte attribution for one column (merged by name across layers
/// and decoded tiles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnCost {
    /// Column name (e.g. `geometry`, `vertex_time`, a property name).
    pub name: String,
    /// Arrow data type, `Debug`-formatted.
    pub dtype: String,
    /// Bytes charged to this column, summed over the decoded tiles, under the
    /// attribution design [`DecodeStats::attribution`] names.
    ///
    /// Default (`loo-v2`): the leave-one-out marginal — what the tiles shed if
    /// the column vanished. Rollback (`singleton-v1`): the old standalone
    /// re-encode size (IPC + zstd-19), which over-charges shared information.
    pub compressed_bytes: u64,
    /// `compressed_bytes / Σ all columns' compressed_bytes` — the fair share.
    /// Under `loo-v2` these are normalised marginals: they sum to 1 AND their
    /// byte sum fits inside the whole-tile size.
    pub share: f64,
    /// `compressed_bytes / rows` over the batches that carry this column.
    pub bytes_per_feature: f64,
    /// Leave-one-out marginal bytes over the decoded tiles: the honest unit
    /// for "how much would dropping this column actually save". Equals
    /// `compressed_bytes` under `loo-v2`; `0` under the `singleton-v1`
    /// rollback, which performs no leave-one-out encode.
    #[serde(default)]
    pub marginal_bytes: u64,
    /// Standard error of [`Self::share`] across the decoded tiles (the
    /// ratio-estimator form — see [`ShareDispersion`]).
    ///
    /// Under a sampled decode this is the share's sampling error. Under an
    /// EXHAUSTIVE decode (`sample: None`) it is deliberately still finite and
    /// non-zero: no finite-population correction is applied, so what it
    /// publishes is the tile-to-tile dispersion of the share — how far the
    /// number would move on comparable tiles from the same producer — which is
    /// what a threshold consumer needs. `0.0` when fewer than two tiles were
    /// decoded (no dispersion evidence exists).
    #[serde(default)]
    pub share_stderr: f64,
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

// ----------------------------------------------------------------------------
// Decode-sampling designs
// ----------------------------------------------------------------------------

/// Which design picks the tiles the decode pass reads.
///
/// Sampling is CONTRACTUAL: every design here is a pure function of the
/// directory — no RNG, no wall clock, no arrival-order or hash-iteration
/// dependence — so two runs over the same archive read the same tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleDesign {
    /// Flat systematic stride over the curve-ordered directory: every
    /// `ceil(total/n)`-th entry. Kept as the documented fallback for one
    /// release — it aliases badly when the stride lands near the temporal
    /// bucket count (it can then sample a single bucket phase).
    StrideV1,
    /// Byte-proportional stratified sampling: entries are grouped into
    /// `(zoom, time_start)` strata, each stratum gets a quota proportional to
    /// its share of compressed blob bytes (largest-remainder rounding, ties
    /// broken by ascending `(zoom, time_start)`), and the quota is spread
    /// evenly inside the stratum from its first entry.
    #[default]
    StratifiedV2,
}

impl SampleDesign {
    /// Stable discriminator string published on [`DecodeStats::design`].
    pub fn as_str(self) -> &'static str {
        match self {
            SampleDesign::StrideV1 => "stride-v1",
            SampleDesign::StratifiedV2 => "stratified-v2",
        }
    }
}

/// v1: the flat systematic stride, as directory indices.
///
/// Retained as the rollback path behind [`SampleDesign::StrideV1`]; the
/// anti-aliasing unit test uses it as the cheap guard for the rejected design.
pub fn stride_sample_indices(entries: &[TileEntry], n: usize) -> Vec<usize> {
    if n == 0 || entries.is_empty() {
        return Vec::new();
    }
    let stride = sample_stride(entries.len(), n);
    (0..entries.len()).step_by(stride).collect()
}

/// Largest-remainder allocation of `n` units over weighted, capacity-bounded
/// buckets.
///
/// Contract: returns exactly `n` units in total whenever `n <= Σ caps` (and
/// `Σ caps` otherwise), never exceeds a bucket's cap, and is a pure function of
/// its inputs — the ordering of `weights`/`caps` IS the tiebreak, so callers
/// must pass buckets in their canonical (ascending-key) order.
///
/// When `n >= caps.len()` every non-empty bucket is first reserved one unit, so
/// a stratum can never be missed entirely just because it is byte-light; the
/// remainder is what gets distributed proportionally.
fn largest_remainder_alloc(weights: &[u128], caps: &[usize], n: usize) -> Vec<usize> {
    let k = weights.len();
    debug_assert_eq!(k, caps.len());
    let mut alloc = vec![0usize; k];
    if k == 0 || n == 0 {
        return alloc;
    }
    let total_cap: usize = caps.iter().sum();
    if n >= total_cap {
        return caps.to_vec();
    }

    // Coverage floor: with at least one unit per bucket available, spend it
    // there first (guarantees "n >= strata count touches every stratum").
    let mut remaining = n;
    if n >= k {
        for (i, a) in alloc.iter_mut().enumerate() {
            if caps[i] > 0 {
                *a = 1;
                remaining -= 1;
            }
        }
    }

    // Degenerate weights (an all-zero-length directory) fall back to uniform so
    // the allocation is still defined and still deterministic.
    let w_sum: u128 = weights.iter().sum();
    let (w, w_sum): (Vec<u128>, u128) = if w_sum == 0 {
        (vec![1u128; k], k as u128)
    } else {
        (weights.to_vec(), w_sum)
    };

    // Integer proportional base + exact remainder — no floats, so the
    // tie-break is exact rather than epsilon-dependent.
    let mut rems: Vec<(u128, usize)> = Vec::with_capacity(k);
    let mut placed = 0usize;
    for i in 0..k {
        let num = remaining as u128 * w[i];
        let base = (num / w_sum) as usize;
        let room = caps[i] - alloc[i];
        let add = base.min(room);
        alloc[i] += add;
        placed += add;
        rems.push((num % w_sum, i));
    }
    let mut left = remaining - placed;

    // Largest remainder first, ascending bucket index on ties.
    rems.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for &(_, i) in &rems {
        if left == 0 {
            break;
        }
        if alloc[i] < caps[i] {
            alloc[i] += 1;
            left -= 1;
        }
    }
    // Capacity clamping can leave units unplaced; sweep in ascending index
    // order until they are all spent (Σ caps > n guarantees termination).
    while left > 0 {
        let mut progressed = false;
        for i in 0..k {
            if left == 0 {
                break;
            }
            if alloc[i] < caps[i] {
                alloc[i] += 1;
                left -= 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    alloc
}

/// v2: byte-proportional stratified sample of at most `n` directory indices,
/// returned ascending.
///
/// Strata are `(zoom, time_start)` — both already on every entry, exactly the
/// stats the directory pass computes for free. A stratum's quota is
/// proportional to its share of compressed blob BYTES (not its entry count):
/// the decode pass exists to attribute bytes, so byte mass is the right
/// allocation currency. Rounding is largest-remainder with an ascending
/// `(zoom, time_start)` tiebreak, so the returned set is a pure function of the
/// directory. Inside a stratum the quota is spread evenly from its first entry
/// (`pos_j = j·len/quota`), which yields exactly `quota` distinct positions.
///
/// This is what replaces the flat stride: a stride near the bucket count could
/// land on one bucket phase and miss every other bucket, which is the aliasing
/// failure §12.1 names.
pub fn stratified_sample_indices(entries: &[TileEntry], n: usize) -> Vec<usize> {
    if n == 0 || entries.is_empty() {
        return Vec::new();
    }
    if n >= entries.len() {
        return (0..entries.len()).collect();
    }

    // BTreeMap: canonical ascending (zoom, time_start) order, and the pushed
    // index lists inherit ascending directory order. No HashMap anywhere on
    // this path — iteration order is part of the contract.
    let mut strata: BTreeMap<(u8, i64), Vec<usize>> = BTreeMap::new();
    for (i, e) in entries.iter().enumerate() {
        strata.entry((e.zoom, e.time_start)).or_default().push(i);
    }

    let weights: Vec<u128> = strata
        .values()
        .map(|ix| ix.iter().map(|&i| entries[i].length as u128).sum())
        .collect();
    let caps: Vec<usize> = strata.values().map(|ix| ix.len()).collect();
    let alloc = largest_remainder_alloc(&weights, &caps, n);

    let mut out: Vec<usize> = Vec::with_capacity(n);
    for (h, ix) in strata.values().enumerate() {
        let quota = alloc[h];
        if quota == 0 {
            continue;
        }
        let len = ix.len();
        for j in 0..quota {
            out.push(ix[(j * len) / quota]);
        }
    }
    out.sort_unstable();
    out
}

/// Directory indices the decode pass reads for `n` sampled tiles under
/// `design`.
fn sample_indices(entries: &[TileEntry], n: usize, design: SampleDesign) -> Vec<usize> {
    match design {
        SampleDesign::StrideV1 => stride_sample_indices(entries, n),
        SampleDesign::StratifiedV2 => stratified_sample_indices(entries, n),
    }
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

/// Inspect a packed tileset with the default decode-sampling
/// ([`SampleDesign::StratifiedV2`]) and attribution
/// ([`AttributionDesign::LeaveOneOutV2`]) designs.
///
/// `sample`: `None` decodes every tile; `Some(n)` decodes a deterministic
/// sample of at most `n` tiles; `Some(0)` skips the decode pass entirely (the
/// directory-only fast mode — a recorded contract). Directory-derived stats
/// (`per_zoom`, `dedup`, the wire totals) are ALWAYS computed over all
/// entries — only the decode-based stats (`decode`, `per_column`) sample.
pub fn inspect(tileset: &PackedTileset, sample: Option<usize>) -> Result<InspectReport> {
    inspect_with_design(tileset, sample, SampleDesign::default())
}

/// Inspect a packed tileset, choosing the decode-sampling design explicitly.
///
/// [`inspect`] is this with [`SampleDesign::StratifiedV2`];
/// [`SampleDesign::StrideV1`] reproduces the pre-stratified behaviour and is
/// the documented rollback. Attribution stays on its own default — use
/// [`inspect_with_designs`] to move that one.
pub fn inspect_with_design(
    tileset: &PackedTileset,
    sample: Option<usize>,
    design: SampleDesign,
) -> Result<InspectReport> {
    inspect_with_designs(tileset, sample, design, AttributionDesign::default())
}

/// Inspect a packed tileset, choosing BOTH the decode-sampling design and the
/// per-column attribution design.
///
/// The two are independent rollback levers: `design` picks WHICH tiles are
/// decoded, `attribution` picks how their columns are priced.
pub fn inspect_with_designs(
    tileset: &PackedTileset,
    sample: Option<usize>,
    design: SampleDesign,
    attribution: AttributionDesign,
) -> Result<InspectReport> {
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
        marginal: u64,
        rows: u64,
    }
    let mut cols: BTreeMap<String, ColAcc> = BTreeMap::new();
    // Per-tile share observations for the published `share_stderr`. `BTreeMap`
    // accumulators only — no HashMap iteration order may reach a number.
    let mut dispersion = ShareDispersion::default();
    let mut schemas: BTreeSet<String> = BTreeSet::new();
    let mut tiles_decoded = 0u64;
    let mut features_decoded = 0u64;
    // `None` = exhaustive (no index list materialised); `Some(ix)` = the
    // sampled plan, already ascending. `Some(0)` yields an empty plan, which
    // is the directory-only fast mode.
    let plan: Option<Vec<usize>> = sample.map(|n| sample_indices(entries, n, design));
    let decode_count = match &plan {
        None => entries.len(),
        Some(ix) => ix.len(),
    };
    for k in 0..decode_count {
        let e = match &plan {
            None => &entries[k],
            Some(ix) => &entries[ix[k]],
        };
        let layers = tileset.read_layers(e).with_context(|| {
            format!(
                "decoding tile z{}/{}/{} t{}",
                e.zoom, e.x, e.y, e.time_start
            )
        })?;
        tiles_decoded += 1;
        schemas.insert(schema_signature(&layers));
        // Bytes this ONE tile attributes per column, merged over its layers —
        // the observation unit `share_stderr` is estimated from.
        let mut per_tile: BTreeMap<String, u64> = BTreeMap::new();
        for layer in &layers {
            let batch = &layer.batch;
            let rows = batch.num_rows() as u64;
            features_decoded += rows;
            // Attribution strips field AND schema metadata before every IPC
            // write (Arrow serializes those HashMaps in nondeterministic order,
            // so keeping them would make repeated inspections disagree by a few
            // bytes) and returns costs in SCHEMA ORDER, index-aligned with the
            // fields walked below. The encoding note is derived from the
            // ORIGINAL field, so the metadata still reaches the report.
            let attributed = attribute_columns(batch, COLUMN_ZSTD_LEVEL, attribution)?;
            let schema = batch.schema();
            for (i, field) in schema.fields().iter().enumerate() {
                let a = &attributed[i];
                debug_assert_eq!(&a.name, field.name(), "attribution must be in schema order");
                let c = cols.entry(field.name().clone()).or_default();
                c.compressed += a.bytes;
                c.marginal += a.marginal_bytes;
                c.rows += rows;
                c.dtype = format!("{:?}", field.data_type());
                c.note = encoding_note(field);
                *per_tile.entry(field.name().clone()).or_insert(0) += a.bytes;
            }
        }
        dispersion.observe_tile(&per_tile);
    }
    let col_total: u64 = cols.values().map(|c| c.compressed).sum();
    let mut per_column: Vec<ColumnCost> = cols
        .into_iter()
        .map(|(name, c)| ColumnCost {
            dtype: c.dtype,
            compressed_bytes: c.compressed,
            share: c.compressed as f64 / col_total.max(1) as f64,
            bytes_per_feature: c.compressed as f64 / c.rows.max(1) as f64,
            marginal_bytes: c.marginal,
            share_stderr: dispersion.stderr(&name),
            encoding_note: c.note,
            name,
        })
        .collect();
    // Descending bytes, ascending name on ties. Leave-one-out attribution
    // produces far more ties than the singleton proxy did (a fully redundant
    // column marginals to zero), so the tiebreak is spelled out rather than
    // left to the stable sort's input order.
    per_column.sort_by(|a, b| {
        b.compressed_bytes
            .cmp(&a.compressed_bytes)
            .then_with(|| a.name.cmp(&b.name))
    });

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
            sampled: plan.is_some(),
            features_decoded,
            distinct_layer_schemas: schemas.len() as u64,
            design: design.as_str().to_string(),
            attribution: attribution.as_str().to_string(),
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
            format!(", sampled [{}]", report.decode.design)
        } else {
            String::new()
        }
    ));
    out.push_str(&format!(
        "  features decoded: {}   distinct layer schemas: {}\n\n",
        report.decode.features_decoded, report.decode.distinct_layer_schemas
    ));

    if !report.per_column.is_empty() {
        out.push_str(&format!(
            "💾 Per-column cost ({}, IPC+zstd-19; shares ±1 stderr, not absolute wire)\n",
            if report.decode.attribution == AttributionDesign::SingletonV1.as_str() {
                "standalone re-encode"
            } else {
                "leave-one-out marginals"
            }
        ));
        out.push_str(&format!(
            "  {:<22} {:<28} {:>10} {:>9} {:>7} {:>7}  note\n",
            "column", "dtype", "comp KB", "B/feat", "share%", "±"
        ));
        for c in &report.per_column {
            let dt = if c.dtype.len() > 27 {
                format!("{}…", &c.dtype[..26])
            } else {
                c.dtype.clone()
            };
            out.push_str(&format!(
                "  {:<22} {:<28} {:>10.1} {:>9.2} {:>6.1}% {:>6.2}%  {}\n",
                c.name,
                dt,
                c.compressed_bytes as f64 / 1e3,
                c.bytes_per_feature,
                100.0 * c.share,
                100.0 * c.share_stderr,
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
            format_version: stt_core::arrow_tile::LAYER_FRAME_VERSION,
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

    // ------------------------------------------------------------------
    // MO-1: stratified deterministic decode sampling
    // ------------------------------------------------------------------

    /// A directory entry with just the fields sampling reads: the stratum key
    /// `(zoom, time_start)` and the byte weight `length`.
    fn entry(zoom: u8, time_start: i64, length: u32) -> TileEntry {
        TileEntry {
            zoom,
            x: 0,
            y: 0,
            time_start,
            time_end: time_start + 1,
            variant_id: 0,
            pack_id: 0,
            offset: 0,
            length,
            uncompressed_size: length * 3,
            feature_count: 1,
            hilbert: 0,
            crc32c: 0,
            temporal_bucket_ms: None,
            cover_t_min: None,
        }
    }

    /// Which stratum each sampled index belongs to.
    fn strata_hit(entries: &[TileEntry], picks: &[usize]) -> BTreeMap<(u8, i64), usize> {
        let mut hits: BTreeMap<(u8, i64), usize> = BTreeMap::new();
        for &i in picks {
            *hits
                .entry((entries[i].zoom, entries[i].time_start))
                .or_insert(0) += 1;
        }
        hits
    }

    #[test]
    fn stratified_allocation_sums_to_n_and_tracks_stratum_bytes() {
        // Three zooms, one time bucket each, byte mass skewed 80/16/4 —
        // deliberately NOT the entry-count split (71/14/14), so an allocation
        // that quietly counted entries instead of bytes would fail here.
        let mut entries = Vec::new();
        for _ in 0..200 {
            entries.push(entry(3, 0, 40));
        }
        for _ in 0..40 {
            entries.push(entry(4, 0, 40));
        }
        for _ in 0..40 {
            entries.push(entry(5, 0, 10));
        }
        // bytes: z3 = 8000, z4 = 1600, z5 = 400 → 10_000 total.

        let picks = stratified_sample_indices(&entries, 100);
        assert_eq!(picks.len(), 100, "allocation must sum to exactly n");
        assert!(
            picks.windows(2).all(|w| w[0] < w[1]),
            "indices must be ascending and distinct"
        );

        let hits = strata_hit(&entries, &picks);
        // Ideal byte shares are 80 / 16 / 4; the one-per-stratum coverage
        // reserve shifts each by at most a unit or two.
        assert!(
            (hits[&(3, 0)] as i64 - 80).abs() <= 2,
            "z3 got {}",
            hits[&(3, 0)]
        );
        assert!(
            (hits[&(4, 0)] as i64 - 16).abs() <= 2,
            "z4 got {}",
            hits[&(4, 0)]
        );
        assert!(
            (hits[&(5, 0)] as i64 - 4).abs() <= 2,
            "z5 got {}",
            hits[&(5, 0)]
        );
        assert_eq!(hits.values().sum::<usize>(), 100);

        // Pure function of the directory: byte-identical on a re-run.
        assert_eq!(picks, stratified_sample_indices(&entries, 100));
    }

    #[test]
    fn stratified_beats_the_stride_aliasing_trap() {
        // The exact failure mode of stride-v1: entry order interleaves k = 4
        // temporal buckets and the requested sample makes the stride land on
        // k, so every pick shares one bucket phase.
        let mut entries = Vec::new();
        for i in 0..40 {
            entries.push(entry(6, (i % 4) as i64 * 3_600_000, 100));
        }

        let strided = stride_sample_indices(&entries, 10);
        let strided_hits = strata_hit(&entries, &strided);
        assert_eq!(
            strided_hits.len(),
            1,
            "guard for the REJECTED design: stride-v1 must alias to one bucket here \
             (hits {strided_hits:?}) — if it stops doing so this test no longer guards anything"
        );

        let stratified = stratified_sample_indices(&entries, 10);
        assert_eq!(stratified.len(), 10);
        let hits = strata_hit(&entries, &stratified);
        assert_eq!(hits.len(), 4, "every bucket must be sampled: {hits:?}");
        assert!(hits.values().all(|&c| c >= 2), "hits {hits:?}");
    }

    #[test]
    fn stratified_touches_every_stratum_when_n_reaches_strata_count() {
        // Byte-light strata must not be starved out by proportional rounding:
        // one stratum carries 100x the bytes of the other nine.
        let mut entries = Vec::new();
        for _ in 0..10 {
            entries.push(entry(7, 0, 100_000));
        }
        for b in 1..10i64 {
            for _ in 0..10 {
                entries.push(entry(7, b * 1_000, 10));
            }
        }
        let picks = stratified_sample_indices(&entries, 10);
        assert_eq!(picks.len(), 10);
        let hits = strata_hit(&entries, &picks);
        assert_eq!(hits.len(), 10, "n == strata count → one each: {hits:?}");
        assert!(hits.values().all(|&c| c == 1));
    }

    #[test]
    fn stratified_returns_exactly_n_on_thousands_of_singleton_strata() {
        // Fragmented directory: every entry is its own stratum, so the
        // proportional quota n_h < 1 everywhere and ONLY largest-remainder
        // rounding gets the count right.
        let entries: Vec<TileEntry> = (0..2000i64).map(|t| entry(9, t, 500)).collect();
        for n in [1usize, 7, 63, 999] {
            let picks = stratified_sample_indices(&entries, n);
            assert_eq!(picks.len(), n, "n = {n}");
            assert!(picks.windows(2).all(|w| w[0] < w[1]), "n = {n}");
            assert_eq!(picks, stratified_sample_indices(&entries, n), "n = {n}");
        }
        // Equal weights + equal remainders → the ascending (zoom, time_start)
        // tiebreak decides, so the first n strata win.
        assert_eq!(stratified_sample_indices(&entries, 3), vec![0, 1, 2]);
    }

    #[test]
    fn stratified_degenerate_inputs() {
        let entries: Vec<TileEntry> = (0..8i64).map(|t| entry(4, t, 0)).collect();
        // n == 0 decodes nothing; n >= total decodes everything.
        assert!(stratified_sample_indices(&entries, 0).is_empty());
        assert_eq!(stratified_sample_indices(&entries, 8).len(), 8);
        assert_eq!(stratified_sample_indices(&entries, 99).len(), 8);
        // All-zero byte weights still allocate (uniform fallback), still exact.
        assert_eq!(stratified_sample_indices(&entries, 3).len(), 3);
        assert!(stratified_sample_indices(&[], 5).is_empty());
    }

    #[test]
    fn inspect_publishes_the_sampling_design_and_honours_the_rollback() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);
        let ts = PackedTileset::open(&out).unwrap();

        let v2 = inspect(&ts, Some(2)).unwrap();
        assert_eq!(v2.decode.design, "stratified-v2");
        assert!(format_text(&v2).contains("sampled [stratified-v2]"));

        // The v1 stride stays reachable as the documented rollback.
        let v1 = inspect_with_design(&ts, Some(2), SampleDesign::StrideV1).unwrap();
        assert_eq!(v1.decode.design, "stride-v1");
        assert_eq!(v1.decode.tiles_decoded, 2);

        // --sample 0 keeps its directory-only fast-mode semantics under BOTH
        // designs: nothing decoded, directory stats still total.
        for design in [SampleDesign::StratifiedV2, SampleDesign::StrideV1] {
            let none = inspect_with_design(&ts, Some(0), design).unwrap();
            assert_eq!(none.decode.tiles_decoded, 0);
            assert!(none.per_column.is_empty());
            assert_eq!(none.dedup.entries, 4);
            assert_eq!(none.tile_count, 4);
        }

        // Old reports (no `design` key) still deserialize, as stride-v1.
        let mut value = serde_json::to_value(&v2).unwrap();
        value["decode"]
            .as_object_mut()
            .unwrap()
            .remove("design")
            .unwrap();
        let legacy: InspectReport = serde_json::from_value(value).unwrap();
        assert_eq!(legacy.decode.design, "stride-v1");
    }

    #[test]
    fn sampled_inspect_is_byte_identical_across_runs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);
        let ts = PackedTileset::open(&out).unwrap();

        let a = serde_json::to_string(&inspect(&ts, Some(2)).unwrap()).unwrap();
        let b = serde_json::to_string(&inspect(&ts, Some(2)).unwrap()).unwrap();
        assert_eq!(a, b, "sampled inspect JSON must be byte-identical");

        let full_a = serde_json::to_string(&inspect(&ts, None).unwrap()).unwrap();
        let full_b = serde_json::to_string(&inspect(&ts, None).unwrap()).unwrap();
        assert_eq!(
            full_a, full_b,
            "exhaustive inspect JSON must be byte-identical"
        );
    }

    // ------------------------------------------------------------------
    // MO-3: leave-one-out attribution / MO-2: dispersion
    // ------------------------------------------------------------------

    #[test]
    fn per_column_costs_are_leave_one_out_marginals() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);
        let ts = PackedTileset::open(&out).unwrap();

        let loo = inspect(&ts, None).unwrap();
        assert_eq!(loo.decode.attribution, "loo-v2", "loo-v2 is the default");
        for c in &loo.per_column {
            assert_eq!(
                c.marginal_bytes, c.compressed_bytes,
                "column {}: loo-v2 charges the marginal",
                c.name
            );
        }
        let share_sum: f64 = loo.per_column.iter().map(|c| c.share).sum();
        assert!((share_sum - 1.0).abs() < 1e-9, "shares sum to {share_sum}");

        // Efficiency: the attributed bytes fit inside the wire bytes of the
        // decoded tiles. The singleton proxy could not say that — it is the
        // documented reason it was replaced. (The decode covers every entry,
        // and shared blobs are read once per entry, so `compressed_bytes` is
        // the right budget to compare against.)
        let attributed: u64 = loo.per_column.iter().map(|c| c.compressed_bytes).sum();
        assert!(
            attributed <= loo.compressed_bytes,
            "Σ marginals {attributed} must fit in {} wire bytes",
            loo.compressed_bytes
        );

        // The singleton rollback stays reachable, publishes its own
        // discriminator, and remains the inefficient proxy.
        let singleton = inspect_with_designs(
            &ts,
            None,
            SampleDesign::StratifiedV2,
            AttributionDesign::SingletonV1,
        )
        .unwrap();
        assert_eq!(singleton.decode.attribution, "singleton-v1");
        assert!(singleton.per_column.iter().all(|c| c.marginal_bytes == 0));
        let singleton_total: u64 = singleton
            .per_column
            .iter()
            .map(|c| c.compressed_bytes)
            .sum();
        assert!(
            singleton_total > loo.compressed_bytes,
            "guard for the REJECTED proxy: singleton sum {singleton_total} must exceed the \
             archive's {} wire bytes",
            loo.compressed_bytes
        );
        // Directory-derived numbers are attribution-independent — the diff
        // gate's exact metric can never move because of an attribution change.
        assert_eq!(singleton.compressed_bytes, loo.compressed_bytes);
        assert_eq!(singleton.tile_count, loo.tile_count);
        assert_eq!(singleton.dedup.distinct_blobs, loo.dedup.distinct_blobs);
        assert!(format_text(&singleton).contains("standalone re-encode"));
        assert!(format_text(&loo).contains("leave-one-out marginals"));
    }

    #[test]
    fn exhaustive_decode_publishes_finite_nonzero_dispersion() {
        // The interpretation MO-2 documents: with `sample: None` every tile was
        // decoded, so there is no sampling error left — and the published
        // stderr is deliberately NOT zero. What it reports is the tile-to-tile
        // dispersion of the share (no finite-population correction), i.e. how
        // far the number would move on comparable tiles from this producer.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);
        let ts = PackedTileset::open(&out).unwrap();

        let report = inspect(&ts, None).unwrap();
        assert!(!report.decode.sampled);
        assert_eq!(report.decode.tiles_decoded, 4);
        for c in &report.per_column {
            assert!(
                c.share_stderr.is_finite() && c.share_stderr >= 0.0,
                "column {} stderr {}",
                c.name,
                c.share_stderr
            );
        }
        assert!(
            report.per_column.iter().any(|c| c.share_stderr > 0.0),
            "the fixture's tiles differ, so SOME column must show dispersion: {:?}",
            report.per_column
        );
        // The rendered table carries the spread beside the share.
        assert!(format_text(&report).contains("±"));

        // A single decoded tile has no dispersion evidence at all → 0.0, which
        // makes every consumer's `share − 2·stderr` gate behave as it did
        // before MO-2 rather than silently suppressing findings.
        let one = inspect(&ts, Some(1)).unwrap();
        assert_eq!(one.decode.tiles_decoded, 1);
        assert!(one.per_column.iter().all(|c| c.share_stderr == 0.0));
    }

    #[test]
    fn attribution_and_dispersion_are_byte_identical_across_runs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);
        let ts = PackedTileset::open(&out).unwrap();

        for design in [
            AttributionDesign::LeaveOneOutV2,
            AttributionDesign::SingletonV1,
        ] {
            for sample in [None, Some(2)] {
                let a =
                    inspect_with_designs(&ts, sample, SampleDesign::StratifiedV2, design).unwrap();
                let b =
                    inspect_with_designs(&ts, sample, SampleDesign::StratifiedV2, design).unwrap();
                assert_eq!(
                    serde_json::to_string(&a).unwrap(),
                    serde_json::to_string(&b).unwrap(),
                    "design {design:?} sample {sample:?} must serialize identically"
                );
            }
        }
    }

    #[test]
    fn pre_mo2_reports_deserialize_with_singleton_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        build_fixture(&out);
        let ts = PackedTileset::open(&out).unwrap();
        let report = inspect(&ts, None).unwrap();

        let mut value = serde_json::to_value(&report).unwrap();
        value["decode"]
            .as_object_mut()
            .unwrap()
            .remove("attribution")
            .unwrap();
        for row in value["per_column"].as_array_mut().unwrap() {
            let obj = row.as_object_mut().unwrap();
            obj.remove("marginal_bytes").unwrap();
            obj.remove("share_stderr").unwrap();
        }
        let legacy: InspectReport = serde_json::from_value(value).unwrap();
        assert_eq!(legacy.decode.attribution, "singleton-v1");
        assert!(legacy.per_column.iter().all(|c| c.marginal_bytes == 0));
        assert!(legacy.per_column.iter().all(|c| c.share_stderr == 0.0));
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
