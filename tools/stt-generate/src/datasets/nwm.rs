//! NOAA National Water Model river-discharge corridors (`nwm`).
//!
//! Animates hourly/daily modeled discharge from the NWM v3.0 **retrospective**
//! (`chrtout.zarr` on anonymous S3) over the NHDPlusV2 CONUS river network as
//! `vertex_value_matrix` flow corridors — the continental-scale sibling of
//! `bixi --streets`, with zero new tile-format features (design:
//! `docs/roadmap/demos-and-datasets.md`).
//!
//! ## Pipeline
//! 1. **Chunk fetch** — the zarr chunks are *bare zstd frames* of C-order
//!    int32 `[672 h × 30 000 reaches]` (× 0.01 → m³/s, fill −999 900), so a
//!    plain HTTPS GET + zstd decode replaces any zarr crate. Compressed chunks
//!    are cached at `data/nwm/chrtout-cache/{t}.{r}.zst` (skip-if-exists ⇒
//!    resumable; the March window reuses the year's chunks).
//! 2. **Reduce** — stream reach-stripe-major (one 30 k-reach stripe × the
//!    window's time chunks at a time): daily mean (`--bin 1d`) or hourly
//!    passthrough (`--bin 1h`), plus the per-reach median of the daily series
//!    (the `log-anomaly` baseline). Persisted reaches = zarr `order` ≥ 3 ∪
//!    the geometry's COMIDs (NWM's order resets StreamCalc-style on
//!    divergence paths the geometry keeps). Per-stripe results persist under
//!    `data/nwm/reduced/{window}-{bin}/stripe-{r}.bin` ⇒ resumable, and the
//!    anomaly demo reuses the year demo's pass.
//! 3. **Assemble** — join reduced series to flowlines by `COMID ==
//!    feature_id` (unmatched counted, never silent); per zoom band: filter by
//!    Strahler order, merge reach chains within runs of constant
//!    `(LevelPathI, StreamOrde)` walked along `DnHydroseq`, resample to the
//!    band's ~2-px vertex spacing, and splat each vertex's *source-reach*
//!    series (log-encoded, rounded to 0.01 — the compression lever) into the
//!    flat vertex-major matrix. Every feature carries `min_zoom == max_zoom ==
//!    z` so each zoom gets its own pre-generalized copy (bixi-cluster-style
//!    `[z, z]` bands — required because `--simplify` cannot touch matrices).
//! 4. **Build** — `stt-build` with the bin as temporal bucket, band fields,
//!    ~10 m coordinate quantization, and NO simplification (stt-build's
//!    default; a supplied matrix is dropped under `--simplify`).
//!
//! ## Value encodings (design §5, baked at generate time)
//! - `self-scaled`: `round(clamp((log10 q − p2) / (p98 − p2), 0, 1), 2)` — each
//!   reach's OWN annual [p2, p98] log-discharge mapped to [0, 1], so every reach
//!   (creek or Mississippi) spans the full ramp across its own low→high and
//!   colour reads seasonal *variation*, not absolute size (year/daily demo). A
//!   `SELF_SCALE_MIN_LOG_SPAN` floor keeps near-constant reaches dim rather than
//!   amplifying their noise to full contrast.
//! - `log-q`:       `round(log10(max(q, 0.01)), 2)` ∈ [−2, 5] — absolute
//!   discharge on a log axis (big rivers dominate the scale; kept for reference).
//! - `log-anomaly`: `round(clamp(log2(q / median_2019), 0, 6), 2)` — flood
//!   anomaly vs the reach's 2019 daily median (March/hourly demo). Reaches
//!   with median < 0.01 m³/s (intermittent) → `NaN` → renderer fallback color.
//!
//! ```bash
//! # Demo 1 — full-year daily flow, each reach self-scaled to its own range:
//! stt-generate nwm --window 2019 --bin 1d --value self-scaled \
//!   --output examples/showcase/public/data/nwm-rivers-2019
//! # Demo 2 — March hourly flood anomaly (needs the 2019 daily pass for medians):
//! stt-generate nwm --window 2019-03 --bin 1h --value log-anomaly \
//!   --output examples/showcase/public/data/nwm-rivers-flood-2019-03
//! ```

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Array, BinaryArray, Float64Array, Int32Array, Int64Array, LargeBinaryArray};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde_json::{json, Map};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::common::{
    self, LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter, SttBuildOptions,
};
use crate::datasets::bixi::resolve_intermediate;

// ---------------------------------------------------------------------------
// Verified data facts (design §2 — do not re-derive)
// ---------------------------------------------------------------------------

const ZARR_BASE: &str =
    "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/CONUS/zarr/chrtout.zarr";
/// streamflow chunk shape `[672 h × 30 000 reaches]`, C-order int32.
const HOURS_PER_CHUNK: usize = 672;
const REACHES_PER_CHUNK: usize = 30_000;
/// Total reach axis (2,776,734) and time axis (385,704 h) of the retrospective.
const N_REACHES: usize = 2_776_734;
const N_HOURS_TOTAL: usize = 385_704;
/// int32 fill value; × 0.01 scale to m³/s.
const FILL_I32: i32 = -999_900;
/// Time axis origin: hours since 1979-02-01T01:00:00Z (asserted vs chrono in tests).
const TIME_EPOCH_MS: i64 = 286_678_800_000;
/// Reaches persisted by the reduce: Strahler order ≥ 3 (the geometry export's floor).
const PERSIST_MIN_ORDER: i32 = 3;
/// Meters per pixel at z0 at 38°N (design §4); vertex spacing target = 2 px.
const METERS_PER_PIXEL_Z0: f64 = 123_357.0;
/// Reach-chain geometric continuity tolerance (deg ≈ 11 m) — a `DnHydroseq`
/// successor whose start vertex is farther than this breaks the merge run.
const CONTINUITY_EPS_DEG: f64 = 1e-4;
/// Anomaly baseline floor: reaches with 2019 median below this are masked NaN.
const MEDIAN_FLOOR: f32 = 0.01;
/// Self-scaled floor: a reach whose robust log10-discharge span (p98 − p2) is
/// below this uses only a proportional fraction of the ramp, so near-constant
/// (regulated / baseflow-only) reaches stay dim instead of amplifying noise to
/// full contrast. 0.30 log10 units ≈ a 2× annual swing to saturate the ramp.
const SELF_SCALE_MIN_LOG_SPAN: f32 = 0.30;
/// Rows per writer flush / parquet row group: matrix rows are tens of KB, so
/// the default 100k-row batches would buffer the whole archive in RAM.
const WRITER_BATCH_ROWS: usize = 2_048;

/// Default matrix columns per temporal tile (auto chunking). Full-res geometry
/// makes a whole-series overview tile hundreds of MB DECODED (verts × buckets ×
/// 4 B); ~30 columns caps the densest z4/z5 tile to ~10 MB while keeping the
/// geometry-duplication (re-sent per chunk) modest. Tune per bin at the call site.
const AUTO_CHUNK_BUCKETS: usize = 30;

fn n_stripes() -> usize {
    N_REACHES.div_ceil(REACHES_PER_CHUNK) // 93
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    about = "Generate NOAA National Water Model river-discharge corridors \
                   (NWM v3.0 retrospective on the NHDPlusV2 CONUS network)"
)]
pub struct Args {
    /// NHDPlus flowlines GeoParquet (COMID, StreamOrde, Hydroseq, LevelPathI,
    /// Divergence, DnHydroseq, geometry_wkb WKB MultiLineString lon/lat).
    #[arg(long, default_value = "data/nwm/nhd-flowlines-order3.parquet")]
    pub flowlines: PathBuf,

    /// Value window: `YYYY` (full year) or `YYYY-MM` (one month), e.g. `2019`
    /// or `2019-03`.
    #[arg(long, default_value = "2019")]
    pub window: String,

    /// Temporal bucket: `1d` (daily mean) or `1h` (hourly; month windows only).
    #[arg(long, default_value = "1d")]
    pub bin: String,

    /// Value encoding: `self-scaled` = round(clamp((log10 q−p2)/(p98−p2),0,1),2),
    /// each reach normalised to its OWN annual [p2,p98] so colour reads seasonal
    /// variation not absolute size (the year demo); `log-q` =
    /// round(log10(max(q,0.01)),2) absolute discharge; `log-anomaly` =
    /// round(clamp(log2(q/median2019),0,6),2) flood anomaly (needs/triggers the
    /// window year's `1d` reduce).
    #[arg(long, default_value = "self-scaled")]
    pub value: String,

    /// Output packed `.stt` directory (or `*.parquet` to stop at the
    /// intermediate). `--out` is accepted as an alias.
    #[arg(
        short,
        long,
        alias = "out",
        default_value = "examples/showcase/public/data/nwm-rivers-2019"
    )]
    pub output: PathBuf,

    /// Zoom pyramid to tile into, `LO-HI` or a single `Z`. Each river is
    /// emitted ONCE at full resolution and confined to `[min_zoom_by_order, HI]`
    /// (density LOD by stream order — NOT per-zoom geometry decimation).
    #[arg(long, default_value = "4-8")]
    pub zooms: String,

    /// Geometry detail: 2-px vertex spacing is targeted at THIS zoom and kept at
    /// every rendered zoom (no per-zoom decimation). Higher = finer/smoother when
    /// zoomed in, larger tiles. z11 ≈ 120 m, z10 ≈ 250 m, z9 ≈ 500 m.
    #[arg(long, default_value_t = 11)]
    pub detail_zoom: u8,

    /// Temporal chunking — matrix columns per streamed temporal tile. `0`
    /// (default) = auto (~30 columns), which caps the DECODED matrix on the
    /// dense low-zoom overview tiles (full-res geometry × the whole series is
    /// hundreds of MB per tile). The geometry is re-sent per chunk, so the
    /// archive grows ≈ linearly in the chunk count. Pass a value ≥ the total
    /// bucket count to send the whole series in ONE tile (geometry once) — only
    /// safe when the geometry is sparse enough that the full series fits.
    #[arg(long, default_value_t = 0)]
    pub chunk_buckets: usize,

    /// Override the per-band Strahler-order threshold (default ladder:
    /// z≤5 → ≥6, z6–7 → ≥5, z8+ → ≥4). Sets the OVERALL floor (order kept) and
    /// still drives the per-feature min_zoom density ladder.
    #[arg(long)]
    pub min_order_override: Option<i32>,

    /// Process only the first N of the 93 reach stripes — cheap smoke tests
    /// (most flowlines will then count as join misses; not for real builds).
    #[arg(long)]
    pub max_reach_stripes: Option<usize>,

    /// Cache directory for compressed zarr chunk objects.
    #[arg(long, default_value = "data/nwm/chrtout-cache")]
    pub cache_dir: PathBuf,

    /// Directory for persisted per-stripe reduced series.
    #[arg(long, default_value = "data/nwm/reduced")]
    pub reduced_dir: PathBuf,

    /// Skip the stt-build step (write only the intermediate GeoParquet).
    #[arg(long)]
    pub skip_build: bool,

    /// Fail if a needed chunk is not already cached instead of downloading.
    #[arg(long)]
    pub skip_download: bool,
}

// ---------------------------------------------------------------------------
// Window / bin / value / ladder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowSpec {
    /// Directory-safe label (`2019`, `2019-03`).
    pub label: String,
    /// First hour index on the retrospective time axis.
    pub start_hour: usize,
    /// Window length in hours.
    pub hours: usize,
    /// Unix ms of the window start (bucket 0).
    pub bucket0_ms: i64,
}

impl WindowSpec {
    fn n_buckets(&self, bin: Bin) -> usize {
        match bin {
            Bin::Daily => self.hours / 24,
            Bin::Hourly => self.hours,
        }
    }
}

fn parse_window(s: &str) -> Result<WindowSpec> {
    use chrono::NaiveDate;
    let s = s.trim();
    let (start, end, label) = if s.len() == 4 && s.bytes().all(|b| b.is_ascii_digit()) {
        let y: i32 = s.parse()?;
        let start = NaiveDate::from_ymd_opt(y, 1, 1).ok_or_else(|| anyhow!("bad year {y}"))?;
        let end = NaiveDate::from_ymd_opt(y + 1, 1, 1).unwrap();
        (start, end, format!("{y}"))
    } else if let Some((ys, ms)) = s.split_once('-') {
        let y: i32 = ys
            .parse()
            .with_context(|| format!("invalid --window '{s}'"))?;
        let m: u32 = ms
            .parse()
            .with_context(|| format!("invalid --window '{s}'"))?;
        let start = NaiveDate::from_ymd_opt(y, m, 1)
            .ok_or_else(|| anyhow!("invalid --window '{s}' (expected YYYY or YYYY-MM)"))?;
        let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
        let end = NaiveDate::from_ymd_opt(ny, nm, 1).unwrap();
        (start, end, format!("{y}-{m:02}"))
    } else {
        bail!("invalid --window '{s}' (expected YYYY or YYYY-MM)");
    };

    let start_ms = start
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_millis();
    let end_ms = end
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_millis();
    let off_ms = start_ms - TIME_EPOCH_MS;
    if off_ms < 0 || off_ms % 3_600_000 != 0 {
        bail!("--window '{s}' starts before the retrospective time axis (1979-02-01T01Z)");
    }
    let start_hour = (off_ms / 3_600_000) as usize;
    let hours = ((end_ms - start_ms) / 3_600_000) as usize;
    if start_hour + hours > N_HOURS_TOTAL {
        bail!("--window '{s}' extends past the end of the retrospective (hour {N_HOURS_TOTAL})");
    }
    Ok(WindowSpec {
        label,
        start_hour,
        hours,
        bucket0_ms: start_ms,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bin {
    Daily,
    Hourly,
}

impl Bin {
    fn suffix(self) -> &'static str {
        match self {
            Bin::Daily => "1d",
            Bin::Hourly => "1h",
        }
    }
    fn ms(self) -> i64 {
        match self {
            Bin::Daily => 86_400_000,
            Bin::Hourly => 3_600_000,
        }
    }
}

fn parse_bin(s: &str) -> Result<Bin> {
    match s.trim() {
        "1d" => Ok(Bin::Daily),
        "1h" => Ok(Bin::Hourly),
        other => bail!("--bin must be `1d` or `1h` (got '{other}')"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueEncoding {
    SelfScaled,
    LogQ,
    LogAnomaly,
}

fn parse_value(s: &str) -> Result<ValueEncoding> {
    match s.trim() {
        "self-scaled" => Ok(ValueEncoding::SelfScaled),
        "log-q" => Ok(ValueEncoding::LogQ),
        "log-anomaly" => Ok(ValueEncoding::LogAnomaly),
        other => {
            bail!("--value must be `self-scaled`, `log-q` or `log-anomaly` (got '{other}')")
        }
    }
}

fn parse_zooms(s: &str) -> Result<(u8, u8)> {
    let s = s.trim();
    let (a, b) = s.split_once('-').unwrap_or((s, s));
    let lo: u8 = a
        .parse()
        .with_context(|| format!("invalid --zooms '{s}'"))?;
    let hi: u8 = b
        .parse()
        .with_context(|| format!("invalid --zooms '{s}'"))?;
    if lo > hi || lo < 1 || hi > 12 {
        bail!("--zooms '{s}' out of range (expected 1 ≤ LO ≤ HI ≤ 12)");
    }
    Ok((lo, hi))
}

/// Design §4 ladder: stream-order floor per zoom band.
fn band_min_order(z: u8) -> i32 {
    match z {
        0..=5 => 6,
        6 | 7 => 5,
        _ => 4,
    }
}

/// Inverse of [`band_min_order`]: the LOWEST rendered zoom a reach of `order`
/// qualifies for — its per-feature `min_zoom`. A river then appears from this
/// zoom up through `zhi` as ONE full-resolution feature (density LOD), instead
/// of a distinct coarse resample per band (which wove overlapping copies).
fn min_zoom_for_order(order: i32, zlo: u8, zhi: u8) -> u8 {
    (zlo..=zhi)
        .find(|&z| band_min_order(z) <= order)
        .unwrap_or(zhi)
}

/// Grid-aligned temporal chunk plan for `[bucket0, range_end)`.
///
/// Cell `g` spans `[g*chunk_bucket_ms, (g+1)*chunk_bucket_ms)` from the epoch —
/// matching stt-build's `floor(start/bucket)*bucket` tile placement, so the
/// client's playhead→tile lookup (which keys on the tile's actual grid-aligned
/// `time_start`) always lands in the right chunk. `bucket0` is bin- but not
/// necessarily chunk-aligned, so the first/last cells are partial. Returns
/// `(bucket_lo, bucket_hi, start_ms, end_ms)` per chunk in the window's own
/// bucket index; the slice `[bucket_lo, bucket_hi)` indexes each reach's series.
fn plan_time_chunks(
    bucket0: i64,
    range_end: i64,
    chunk_bucket_ms: i64,
    bin_ms: i64,
) -> Vec<(usize, usize, i64, i64)> {
    let g_first = bucket0.div_euclid(chunk_bucket_ms);
    let g_last = (range_end - 1).div_euclid(chunk_bucket_ms);
    (g_first..=g_last)
        .map(|g| {
            let t0 = (g * chunk_bucket_ms).max(bucket0);
            let t1 = ((g + 1) * chunk_bucket_ms).min(range_end);
            let b0 = ((t0 - bucket0) / bin_ms) as usize;
            let b1 = ((t1 - bucket0) / bin_ms) as usize;
            (
                b0,
                b1,
                bucket0 + b0 as i64 * bin_ms,
                bucket0 + b1 as i64 * bin_ms,
            )
        })
        .collect()
}

/// 2-px vertex spacing target at 38°N (design §4).
fn spacing_m(z: u8) -> f64 {
    2.0 * METERS_PER_PIXEL_Z0 / f64::powi(2.0, z as i32)
}

/// Static per-feature render width from Strahler order (design §5: width is a
/// per-feature property, never per-bucket). Doubling per order, clamped.
fn width_for_order(order: i32) -> f64 {
    f64::powi(2.0, order - 4).clamp(0.5, 16.0)
}

/// Design stage-1 rule applied at load (the export kept the column): minor
/// divergences (`Divergence == 2`) below order 4 are braid noise — drop.
fn drop_minor_divergence(order: i32, divergence: i32) -> bool {
    divergence == 2 && order < 4
}

// ---------------------------------------------------------------------------
// Chunk index math (verified: 2019-01-01 = hour 349 895 = chunk 520 row 455)
// ---------------------------------------------------------------------------

/// Global hour index → (time chunk, row within chunk).
fn hour_chunk(h: usize) -> (usize, usize) {
    (h / HOURS_PER_CHUNK, h % HOURS_PER_CHUNK)
}

/// Global reach index → (reach stripe, column within stripe). Kept as the
/// canonical statement of the reach-axis index math (exercised by tests);
/// `keep_columns` inlines the same arithmetic over whole stripes.
#[allow(dead_code)]
fn reach_chunk(idx: usize) -> (usize, usize) {
    (idx / REACHES_PER_CHUNK, idx % REACHES_PER_CHUNK)
}

/// Inclusive time-chunk range covering a window.
fn time_chunk_range(w: &WindowSpec) -> std::ops::RangeInclusive<usize> {
    hour_chunk(w.start_hour).0..=hour_chunk(w.start_hour + w.hours - 1).0
}

/// int32 raw cell → m³/s (fill → NaN).
fn scale_flow(raw: i32) -> f32 {
    if raw == FILL_I32 {
        f32::NAN
    } else {
        raw as f32 * 0.01
    }
}

/// Decode a bare zstd frame of C-order little-endian int32 cells.
fn decode_i32_frame(compressed: &[u8], expected_cells: usize) -> Result<Vec<i32>> {
    let raw = zstd::decode_all(compressed).context("zstd decode failed")?;
    if raw.len() < expected_cells * 4 {
        bail!(
            "frame decodes to {} bytes, expected ≥ {}",
            raw.len(),
            expected_cells * 4
        );
    }
    Ok(raw
        .chunks_exact(4)
        .take(expected_cells)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

// ---------------------------------------------------------------------------
// Download / cache
// ---------------------------------------------------------------------------

fn http_client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?)
}

fn http_get_with_retry(client: &reqwest::blocking::Client, url: &str) -> Result<Vec<u8>> {
    let mut last: Option<anyhow::Error> = None;
    for attempt in 0..4u32 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(1 << attempt));
        }
        match client.get(url).send() {
            Ok(resp) if resp.status().is_success() => match resp.bytes() {
                Ok(b) => return Ok(b.to_vec()),
                Err(e) => last = Some(anyhow!("body read failed for {url}: {e}")),
            },
            Ok(resp) => last = Some(anyhow!("HTTP {} for {url}", resp.status())),
            Err(e) => last = Some(anyhow!("request failed for {url}: {e}")),
        }
    }
    Err(last.unwrap_or_else(|| anyhow!("download failed: {url}")))
}

/// Fetch `url` into `path` unless already cached (skip-if-exists ⇒ resumable).
/// Writes via a sibling tmp file + rename so an interrupted run never leaves a
/// truncated cache entry behind.
fn cache_object(
    client: &reqwest::blocking::Client,
    url: &str,
    path: &Path,
    skip_download: bool,
) -> Result<u64> {
    if path.exists() {
        return Ok(0);
    }
    if skip_download {
        bail!(
            "--skip-download set but {} is not cached ({url})",
            path.display()
        );
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = http_get_with_retry(client, url)?;
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path)?;
    Ok(bytes.len() as u64)
}

fn chunk_cache_path(cache_dir: &Path, t: usize, r: usize) -> PathBuf {
    cache_dir.join(format!("{t}.{r}.zst"))
}

fn chunk_url(t: usize, r: usize) -> String {
    format!("{ZARR_BASE}/streamflow/{t}.{r}")
}

/// Fetch + decode one of the 1-D coordinate arrays (`feature_id` int64,
/// `order` int32) — each a single bare zstd chunk object `{name}/0`.
fn fetch_1d_raw(client: &reqwest::blocking::Client, args: &Args, name: &str) -> Result<Vec<u8>> {
    let path = args.cache_dir.join(format!("{name}.0.zst"));
    cache_object(
        client,
        &format!("{ZARR_BASE}/{name}/0"),
        &path,
        args.skip_download,
    )?;
    let compressed = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    zstd::decode_all(&compressed[..]).with_context(|| {
        format!(
            "corrupt cache file {} — delete it and re-run",
            path.display()
        )
    })
}

fn fetch_feature_ids(client: &reqwest::blocking::Client, args: &Args) -> Result<Vec<i64>> {
    let raw = fetch_1d_raw(client, args, "feature_id")?;
    if raw.len() < N_REACHES * 8 {
        bail!("feature_id array too short: {} bytes", raw.len());
    }
    Ok(raw
        .chunks_exact(8)
        .take(N_REACHES)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn fetch_zarr_orders(client: &reqwest::blocking::Client, args: &Args) -> Result<Vec<i32>> {
    let raw = fetch_1d_raw(client, args, "order")?;
    if raw.len() < N_REACHES * 4 {
        bail!("order array too short: {} bytes", raw.len());
    }
    Ok(raw
        .chunks_exact(4)
        .take(N_REACHES)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

// ---------------------------------------------------------------------------
// Reduce (stripe-major, persisted per stripe)
// ---------------------------------------------------------------------------

/// Streaming mean accumulator for one reach stripe. Values are accumulated
/// per (kept reach × bucket); fills are excluded from the mean, and a bucket
/// with no valid sample finalizes to NaN. Hourly is the same machinery with
/// one sample per bucket.
pub(crate) struct Accum {
    keep: Vec<u32>,
    n_buckets: usize,
    sums: Vec<f64>,
    counts: Vec<u32>,
}

impl Accum {
    pub(crate) fn new(keep: Vec<u32>, n_buckets: usize) -> Self {
        let n = keep.len() * n_buckets;
        Self {
            keep,
            n_buckets,
            sums: vec![0.0; n],
            counts: vec![0; n],
        }
    }

    /// Fold one decoded chunk (starting at global hour `chunk_h0`, `stripe_width`
    /// columns per row) into the accumulator, clipped to the window.
    pub(crate) fn ingest_chunk(
        &mut self,
        chunk_h0: usize,
        data: &[i32],
        stripe_width: usize,
        window: &WindowSpec,
        bin: Bin,
    ) {
        let chunk_rows = data.len() / stripe_width;
        let h_lo = chunk_h0.max(window.start_hour);
        let h_hi = (chunk_h0 + chunk_rows).min(window.start_hour + window.hours);
        let nb = self.n_buckets;
        for h in h_lo..h_hi {
            let rel = h - window.start_hour;
            let bucket = match bin {
                Bin::Daily => rel / 24,
                Bin::Hourly => rel,
            };
            let row_off = (h - chunk_h0) * stripe_width;
            for (k, &col) in self.keep.iter().enumerate() {
                let v = scale_flow(data[row_off + col as usize]);
                if v.is_finite() {
                    let cell = k * nb + bucket;
                    self.sums[cell] += v as f64;
                    self.counts[cell] += 1;
                }
            }
        }
    }

    /// → (kept local indices, reach-major bucket values, per-reach medians for Daily).
    pub(crate) fn finalize(self, bin: Bin) -> (Vec<u32>, Vec<f32>, Option<Vec<f32>>) {
        let nb = self.n_buckets;
        let values: Vec<f32> = self
            .sums
            .iter()
            .zip(&self.counts)
            .map(|(&s, &c)| {
                if c > 0 {
                    (s / c as f64) as f32
                } else {
                    f32::NAN
                }
            })
            .collect();
        let medians = match bin {
            Bin::Daily => Some(
                (0..self.keep.len())
                    .map(|k| median_non_nan(&values[k * nb..(k + 1) * nb]))
                    .collect(),
            ),
            Bin::Hourly => None,
        };
        (self.keep, values, medians)
    }
}

pub(crate) fn median_non_nan(vals: &[f32]) -> f32 {
    let mut v: Vec<f32> = vals.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// Nearest-rank percentile of an ascending-sorted slice (`p` ∈ [0, 1]); clamps
/// to the ends. NaN for an empty slice.
pub(crate) fn percentile_sorted(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((p * (sorted.len() as f32 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// `(p2, p98)` of a reach's finite `log10(max(q, 0.01))` series — the
/// self-scaled normalization bounds. Percentiles (not raw min/max) so a single
/// flood-day spike or dry-out can't compress the rest of the year against the
/// ramp. Returns `(NaN, NaN)` when the reach has no finite sample.
pub(crate) fn log_percentile_bounds(series: &[f32]) -> (f32, f32) {
    let mut v: Vec<f32> = series
        .iter()
        .filter(|x| x.is_finite())
        .map(|&x| x.max(0.01).log10())
        .collect();
    if v.is_empty() {
        return (f32::NAN, f32::NAN);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile_sorted(&v, 0.02), percentile_sorted(&v, 0.98))
}

const STRIPE_MAGIC: &[u8; 8] = b"NWMRED2\0";

fn reduced_dir(base: &Path, label: &str, bin: Bin) -> PathBuf {
    base.join(format!("{label}-{}", bin.suffix()))
}

fn stripe_file(dir: &Path, r: usize) -> PathBuf {
    dir.join(format!("stripe-{r:03}.bin"))
}

pub(crate) struct StripeData {
    pub stripe: u32,
    pub n_buckets: u32,
    /// Local (within-stripe) reach column indices kept.
    pub indices: Vec<u32>,
    /// Reach-major `[kept][n_buckets]` values, m³/s (NaN = no valid sample).
    pub values: Vec<f32>,
    pub medians: Option<Vec<f32>>,
}

pub(crate) fn write_stripe(
    path: &Path,
    stripe: u32,
    n_buckets: u32,
    indices: &[u32],
    values: &[f32],
    medians: Option<&[f32]>,
) -> Result<()> {
    let mut payload =
        Vec::with_capacity(24 + indices.len() * 4 + values.len() * 4 + indices.len() * 4);
    payload.extend_from_slice(STRIPE_MAGIC);
    payload.extend_from_slice(&stripe.to_le_bytes());
    payload.extend_from_slice(&n_buckets.to_le_bytes());
    payload.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    payload.push(medians.is_some() as u8);
    payload.extend_from_slice(&[0u8; 3]);
    for &i in indices {
        payload.extend_from_slice(&i.to_le_bytes());
    }
    for &v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(m) = medians {
        for &v in m {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    let compressed = zstd::encode_all(&payload[..], 3)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bin.tmp");
    fs::write(&tmp, &compressed)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

pub(crate) fn read_stripe(path: &Path) -> Result<StripeData> {
    let compressed = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let raw = zstd::decode_all(&compressed[..]).with_context(|| {
        format!(
            "corrupt stripe file {} — delete it and re-run",
            path.display()
        )
    })?;
    let fail = || {
        anyhow!(
            "malformed stripe file {} — delete it and re-run",
            path.display()
        )
    };
    if raw.len() < 24 || &raw[..8] != STRIPE_MAGIC {
        return Err(fail());
    }
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    let stripe = u32_at(8);
    let n_buckets = u32_at(12);
    let n_kept = u32_at(16) as usize;
    let has_medians = raw[20] != 0;
    let mut expected = 24 + n_kept * 4 + n_kept * n_buckets as usize * 4;
    if has_medians {
        expected += n_kept * 4;
    }
    if raw.len() != expected {
        return Err(fail());
    }
    let mut off = 24;
    let indices: Vec<u32> = (0..n_kept).map(|i| u32_at(off + i * 4)).collect();
    off += n_kept * 4;
    let n_vals = n_kept * n_buckets as usize;
    let values: Vec<f32> = raw[off..off + n_vals * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    off += n_vals * 4;
    let medians = has_medians.then(|| {
        raw[off..off + n_kept * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    });
    Ok(StripeData {
        stripe,
        n_buckets,
        indices,
        values,
        medians,
    })
}

/// Local columns to persist for a stripe: zarr `order` ≥ 3 reaches **union**
/// every reach in the geometry export. The union matters: NWM's `order`
/// follows the StreamCalc convention, which resets to 1 on divergence-affected
/// paths that NHD `StreamOrde` (the band/width axis) keeps at full order — a
/// zarr-order-only floor starves ~19% of the order ≥5 network (braided
/// mainstems like the Atchafalaya) of series (measured on the 2019-03 pilot:
/// 39,683 of 39,958 join misses were exactly these reaches).
fn keep_columns(
    zarr_orders: &[i32],
    feature_ids: &[i64],
    geometry_fids: &HashSet<i64>,
    stripe: usize,
) -> Vec<u32> {
    let start = stripe * REACHES_PER_CHUNK;
    let end = (start + REACHES_PER_CHUNK).min(N_REACHES);
    (start..end)
        .filter(|&g| zarr_orders[g] >= PERSIST_MIN_ORDER || geometry_fids.contains(&feature_ids[g]))
        .map(|g| (g - start) as u32)
        .collect()
}

/// Ensure the per-stripe reduce for (window, bin) exists on disk, downloading
/// + reducing only the missing stripes (resumable at both layers).
#[allow(clippy::too_many_arguments)]
fn ensure_reduced(
    client: &reqwest::blocking::Client,
    args: &Args,
    window: &WindowSpec,
    bin: Bin,
    zarr_orders: &[i32],
    feature_ids: &[i64],
    geometry_fids: &HashSet<i64>,
    stripes: &[usize],
) -> Result<()> {
    let dir = reduced_dir(&args.reduced_dir, &window.label, bin);
    let missing: Vec<usize> = stripes
        .iter()
        .copied()
        .filter(|&r| !stripe_file(&dir, r).exists())
        .collect();
    if missing.is_empty() {
        println!(
            "♻️  Reduce cached: {} ({} stripes)",
            dir.display(),
            stripes.len()
        );
        return Ok(());
    }
    fs::create_dir_all(&dir)?;
    let tr = time_chunk_range(window);
    let n_buckets = window.n_buckets(bin);
    println!(
        "🧮 Reducing {} stripes × time chunks {}–{} → {} ({} buckets of {})",
        missing.len(),
        tr.start(),
        tr.end(),
        dir.display(),
        n_buckets,
        bin.suffix()
    );

    // Fetch every chunk the missing stripes need (parallel, cached, resumable).
    let keys: Vec<(usize, usize)> = missing
        .iter()
        .flat_map(|&r| tr.clone().map(move |t| (t, r)))
        .filter(|&(t, r)| !chunk_cache_path(&args.cache_dir, t, r).exists())
        .collect();
    if !keys.is_empty() {
        if args.skip_download {
            bail!(
                "--skip-download set but {} zarr chunks are not cached under {}",
                keys.len(),
                args.cache_dir.display()
            );
        }
        println!(
            "⬇️  Fetching {} zarr chunks (≈{:.2} GB compressed) …",
            keys.len(),
            keys.len() as f64 * 6.3 / 1024.0
        );
        let pb = ProgressBar::new(keys.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );
        let downloaded = AtomicU64::new(0);
        keys.par_iter().try_for_each(|&(t, r)| -> Result<()> {
            let n = cache_object(
                client,
                &chunk_url(t, r),
                &chunk_cache_path(&args.cache_dir, t, r),
                false,
            )?;
            downloaded.fetch_add(n, Ordering::Relaxed);
            pb.inc(1);
            Ok(())
        })?;
        pb.finish_and_clear();
        println!(
            "   ✓ downloaded {:.1} MB",
            downloaded.load(Ordering::Relaxed) as f64 / 1e6
        );
    }

    // Reduce each missing stripe (sequential — one 80 MB chunk decode at a time).
    let pb = ProgressBar::new(missing.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] stripe {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );
    for &r in &missing {
        let keep = keep_columns(zarr_orders, feature_ids, geometry_fids, r);
        let mut acc = Accum::new(keep, n_buckets);
        for t in tr.clone() {
            let path = chunk_cache_path(&args.cache_dir, t, r);
            let compressed = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let data = decode_i32_frame(&compressed, HOURS_PER_CHUNK * REACHES_PER_CHUNK)
                .with_context(|| {
                    format!(
                        "corrupt cache file {} — delete it and re-run",
                        path.display()
                    )
                })?;
            acc.ingest_chunk(t * HOURS_PER_CHUNK, &data, REACHES_PER_CHUNK, window, bin);
        }
        let (indices, values, medians) = acc.finalize(bin);
        write_stripe(
            &stripe_file(&dir, r),
            r as u32,
            n_buckets as u32,
            &indices,
            &values,
            medians.as_deref(),
        )?;
        pb.inc(1);
    }
    pb.finish_and_clear();
    println!("   ✓ {} stripes reduced", missing.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// Series table (assemble-side view of the reduce)
// ---------------------------------------------------------------------------

struct SeriesTable {
    n_buckets: usize,
    /// COMID (== NWM feature_id) → row.
    index: HashMap<i64, u32>,
    /// Row-major `[row][n_buckets]`, m³/s.
    values: Vec<f32>,
    /// zarr `order` per row (StreamOrde cross-check).
    zarr_order: Vec<i32>,
}

impl SeriesTable {
    fn series(&self, row: u32) -> &[f32] {
        let o = row as usize * self.n_buckets;
        &self.values[o..o + self.n_buckets]
    }
}

fn load_series(
    dir: &Path,
    stripes: &[usize],
    n_buckets: usize,
    feature_ids: &[i64],
    zarr_orders: &[i32],
    wanted: &HashSet<i64>,
) -> Result<SeriesTable> {
    let mut table = SeriesTable {
        n_buckets,
        index: HashMap::with_capacity(wanted.len()),
        values: Vec::with_capacity(wanted.len() * n_buckets),
        zarr_order: Vec::with_capacity(wanted.len()),
    };
    for &r in stripes {
        let sd = read_stripe(&stripe_file(dir, r))?;
        if sd.stripe as usize != r || sd.n_buckets as usize != n_buckets {
            bail!(
                "stale reduce {} (stripe {} buckets {}) — delete {} and re-run",
                stripe_file(dir, r).display(),
                sd.stripe,
                sd.n_buckets,
                dir.display()
            );
        }
        for (slot, &local) in sd.indices.iter().enumerate() {
            let g = r * REACHES_PER_CHUNK + local as usize;
            let fid = feature_ids[g];
            if !wanted.contains(&fid) || table.index.contains_key(&fid) {
                continue;
            }
            let row = table.zarr_order.len() as u32;
            table.index.insert(fid, row);
            table
                .values
                .extend_from_slice(&sd.values[slot * n_buckets..(slot + 1) * n_buckets]);
            table.zarr_order.push(zarr_orders[g]);
        }
    }
    Ok(table)
}

/// Per-reach anomaly baselines from a Daily reduce's persisted medians.
fn load_medians(
    dir: &Path,
    stripes: &[usize],
    feature_ids: &[i64],
    wanted: &HashSet<i64>,
) -> Result<HashMap<i64, f32>> {
    let mut out = HashMap::with_capacity(wanted.len());
    for &r in stripes {
        let sd = read_stripe(&stripe_file(dir, r))?;
        let medians = sd.medians.as_ref().ok_or_else(|| {
            anyhow!(
                "{} has no medians (not a 1d reduce) — delete {} and re-run",
                stripe_file(dir, r).display(),
                dir.display()
            )
        })?;
        for (slot, &local) in sd.indices.iter().enumerate() {
            let fid = feature_ids[r * REACHES_PER_CHUNK + local as usize];
            if wanted.contains(&fid) {
                out.entry(fid).or_insert(medians[slot]);
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Flowline geometry (GeoParquet → reaches)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Reach {
    pub comid: i64,
    pub order: i32,
    pub level_path: i64,
    pub hydroseq: i64,
    pub dn_hydroseq: i64,
    /// Digitized upstream → downstream (NHDPlus network convention).
    pub coords: Vec<[f64; 2]>,
}

#[derive(Debug, Default)]
struct LoadStats {
    rows: usize,
    kept: usize,
    below_order: usize,
    minor_divergence: usize,
    bad_wkb: usize,
    degenerate: usize,
    multipart: usize,
}

/// Integer-ish column reader: NHDPlus attributes land as Float64 in the
/// export, COMID as Int32 — accept all three widths.
enum IntLike<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
    F64(&'a Float64Array),
}

impl IntLike<'_> {
    fn get(&self, i: usize) -> Option<i64> {
        match self {
            IntLike::I32(a) => a.is_valid(i).then(|| a.value(i) as i64),
            IntLike::I64(a) => a.is_valid(i).then(|| a.value(i)),
            IntLike::F64(a) => {
                let v = a.is_valid(i).then(|| a.value(i))?;
                v.is_finite().then(|| v.round() as i64)
            }
        }
    }
}

fn int_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<IntLike<'a>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("flowlines parquet missing column '{name}'"))?;
    let any = col.as_any();
    if let Some(a) = any.downcast_ref::<Int32Array>() {
        return Ok(IntLike::I32(a));
    }
    if let Some(a) = any.downcast_ref::<Int64Array>() {
        return Ok(IntLike::I64(a));
    }
    if let Some(a) = any.downcast_ref::<Float64Array>() {
        return Ok(IntLike::F64(a));
    }
    bail!("column '{name}' has unsupported type {:?}", col.data_type())
}

enum BinLike<'a> {
    Plain(&'a BinaryArray),
    Large(&'a LargeBinaryArray),
}

impl BinLike<'_> {
    fn get(&self, i: usize) -> Option<&[u8]> {
        match self {
            BinLike::Plain(a) => a.is_valid(i).then(|| a.value(i)),
            BinLike::Large(a) => a.is_valid(i).then(|| a.value(i)),
        }
    }
}

fn bin_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<BinLike<'a>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("flowlines parquet missing column '{name}'"))?;
    let any = col.as_any();
    if let Some(a) = any.downcast_ref::<LargeBinaryArray>() {
        return Ok(BinLike::Large(a));
    }
    if let Some(a) = any.downcast_ref::<BinaryArray>() {
        return Ok(BinLike::Plain(a));
    }
    bail!("column '{name}' has unsupported type {:?}", col.data_type())
}

/// Minimal 2-D WKB reader for the export's LineString / MultiLineString
/// geometries (little- or big-endian, ISO or EWKB-with-SRID headers).
struct WkbCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> WkbCursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| anyhow!("WKB truncated"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self, le: bool) -> Result<u32> {
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }
    fn f64(&mut self, le: bool) -> Result<f64> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if le {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        })
    }
}

fn parse_wkb_linework(buf: &[u8]) -> Result<Vec<Vec<[f64; 2]>>> {
    let mut c = WkbCursor { buf, pos: 0 };
    read_wkb_geometry(&mut c)
}

fn read_wkb_geometry(c: &mut WkbCursor) -> Result<Vec<Vec<[f64; 2]>>> {
    let le = match c.u8()? {
        0 => false,
        1 => true,
        b => bail!("bad WKB byte order {b}"),
    };
    let mut ty = c.u32(le)?;
    if ty & 0x2000_0000 != 0 {
        // EWKB SRID flag: skip the srid.
        ty &= !0x2000_0000;
        c.u32(le)?;
    }
    if ty & 0xC000_0000 != 0 || (ty % 1000) != ty {
        bail!("non-2D WKB geometry (type {ty:#x})");
    }
    match ty {
        2 => Ok(vec![read_wkb_points(c, le)?]),
        5 => {
            let n_parts = c.u32(le)? as usize;
            if n_parts > c.buf.len() / 9 {
                bail!("implausible WKB part count {n_parts}");
            }
            let mut parts = Vec::with_capacity(n_parts);
            for _ in 0..n_parts {
                let sub = read_wkb_geometry(c)?;
                if sub.len() != 1 {
                    bail!("nested multi-geometry in MultiLineString");
                }
                parts.extend(sub);
            }
            Ok(parts)
        }
        other => bail!("unsupported WKB geometry type {other}"),
    }
}

fn read_wkb_points(c: &mut WkbCursor, le: bool) -> Result<Vec<[f64; 2]>> {
    let n = c.u32(le)? as usize;
    if n > c.buf.len() / 16 + 1 {
        bail!("implausible WKB point count {n}");
    }
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        let x = c.f64(le)?;
        let y = c.f64(le)?;
        pts.push([x, y]);
    }
    Ok(pts)
}

/// COMID-only projection scan of the flowlines parquet — the geometry's reach
/// set, which the reduce's persist floor must cover (see [`keep_columns`]).
fn load_flowline_comids(path: &Path) -> Result<HashSet<i64>> {
    let file = fs::File::open(path)
        .with_context(|| format!("open flowlines parquet {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mask = parquet::arrow::ProjectionMask::columns(builder.parquet_schema(), ["COMID"]);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(65_536)
        .build()?;
    let mut out = HashSet::new();
    for batch in reader {
        let batch = batch?;
        let comid = int_col(&batch, "COMID")?;
        for i in 0..batch.num_rows() {
            if let Some(c) = comid.get(i) {
                out.insert(c);
            }
        }
    }
    Ok(out)
}

fn load_flowlines(path: &Path, min_order: i32) -> Result<(Vec<Reach>, LoadStats)> {
    let file = fs::File::open(path)
        .with_context(|| format!("open flowlines parquet {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(4096)
        .build()?;

    let mut reaches = Vec::new();
    let mut stats = LoadStats::default();
    for batch in reader {
        let batch = batch?;
        let comid = int_col(&batch, "COMID")?;
        let order = int_col(&batch, "StreamOrde")?;
        let hydroseq = int_col(&batch, "Hydroseq")?;
        let level_path = int_col(&batch, "LevelPathI")?;
        let divergence = int_col(&batch, "Divergence")?;
        let dn_hydroseq = int_col(&batch, "DnHydroseq")?;
        let geom = bin_col(&batch, "geometry_wkb")?;

        for i in 0..batch.num_rows() {
            stats.rows += 1;
            let ord = order.get(i).unwrap_or(0) as i32;
            let div = divergence.get(i).unwrap_or(0) as i32;
            if drop_minor_divergence(ord, div) {
                stats.minor_divergence += 1;
                continue;
            }
            if ord < min_order {
                stats.below_order += 1;
                continue;
            }
            let Some(wkb) = geom.get(i) else {
                stats.bad_wkb += 1;
                continue;
            };
            let parts = match parse_wkb_linework(wkb) {
                Ok(p) => p,
                Err(_) => {
                    stats.bad_wkb += 1;
                    continue;
                }
            };
            if parts.len() > 1 {
                stats.multipart += 1;
            }
            // NHD MultiLineStrings are effectively single-part; concatenate
            // parts in file order, deduping the shared join vertex.
            let mut coords: Vec<[f64; 2]> = Vec::new();
            for part in parts {
                for p in part {
                    if coords.last().is_some_and(|&l| l == p) {
                        continue;
                    }
                    coords.push(p);
                }
            }
            if coords.len() < 2 {
                stats.degenerate += 1;
                continue;
            }
            stats.kept += 1;
            reaches.push(Reach {
                comid: comid
                    .get(i)
                    .ok_or_else(|| anyhow!("null COMID at row {i}"))?,
                order: ord,
                level_path: level_path.get(i).unwrap_or(0),
                hydroseq: hydroseq.get(i).unwrap_or(0),
                dn_hydroseq: dn_hydroseq.get(i).unwrap_or(0),
                coords,
            });
        }
    }
    Ok((reaches, stats))
}

// ---------------------------------------------------------------------------
// Mainstem merge (design §4: runs of constant (LevelPathI, StreamOrde))
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MergedRun {
    pub order: i32,
    /// Most-upstream reach's COMID — the deterministic sort/identity key.
    pub comid0: i64,
    /// Concatenated upstream→downstream geometry (shared vertices deduped).
    pub coords: Vec<[f64; 2]>,
    /// Per segment `[i, i+1]`: ordinal into `comids` of the owning reach.
    pub seg_source: Vec<u32>,
    /// Reach COMIDs in path order.
    pub comids: Vec<i64>,
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct MergeStats {
    pub runs: usize,
    /// Runs that actually merged ≥ 2 reaches.
    pub merged_chains: usize,
    /// Chains broken because the next reach's geometry didn't continue ours.
    pub discontinuity_breaks: usize,
}

/// Merge reach chains within runs of constant `(LevelPathI, StreamOrde)`,
/// walking `Hydroseq → DnHydroseq` downstream. Order changes fall in different
/// groups by construction; divergences/braids that aren't linked by
/// `DnHydroseq` (or whose geometry doesn't continue) start their own runs.
/// Deterministic: groups iterate in `BTreeMap` order, members and heads are
/// sorted, and the result is sorted by `comid0` — no map iteration order
/// reaches the output.
pub(crate) fn merge_runs(reaches: &[Reach]) -> (Vec<MergedRun>, MergeStats) {
    let mut groups: BTreeMap<(i64, i32), Vec<usize>> = BTreeMap::new();
    for (i, r) in reaches.iter().enumerate() {
        groups.entry((r.level_path, r.order)).or_default().push(i);
    }

    let mut runs = Vec::new();
    let mut stats = MergeStats::default();
    for ((_, order), mut members) in groups {
        // Upstream-first (Hydroseq decreases downstream), comid tiebreak.
        members.sort_by_key(|&i| (std::cmp::Reverse(reaches[i].hydroseq), reaches[i].comid));
        let by_hydroseq: HashMap<i64, usize> =
            members.iter().map(|&i| (reaches[i].hydroseq, i)).collect();
        let dn_targets: HashSet<i64> = members.iter().map(|&i| reaches[i].dn_hydroseq).collect();
        let mut visited: HashSet<usize> = HashSet::with_capacity(members.len());

        // Pass 0: true heads (no in-group upstream neighbor). Pass 1: leftovers
        // (downstream of a discontinuity break or a consumed braid) walk too,
        // most-upstream first, so nothing is dropped.
        for pass in 0..2 {
            for &start in &members {
                if visited.contains(&start) {
                    continue;
                }
                let is_head = !dn_targets.contains(&reaches[start].hydroseq);
                if pass == 0 && !is_head {
                    continue;
                }
                let mut chain = vec![start];
                visited.insert(start);
                let mut cur = start;
                loop {
                    let dn = reaches[cur].dn_hydroseq;
                    if dn == 0 {
                        break;
                    }
                    let Some(&next) = by_hydroseq.get(&dn) else {
                        break;
                    };
                    if visited.contains(&next) {
                        break;
                    }
                    let a = *reaches[cur].coords.last().unwrap();
                    let b = reaches[next].coords[0];
                    if (a[0] - b[0]).abs() > CONTINUITY_EPS_DEG
                        || (a[1] - b[1]).abs() > CONTINUITY_EPS_DEG
                    {
                        stats.discontinuity_breaks += 1;
                        break;
                    }
                    chain.push(next);
                    visited.insert(next);
                    cur = next;
                }
                if chain.len() > 1 {
                    stats.merged_chains += 1;
                }
                runs.push(build_run(order, &chain, reaches));
            }
        }
    }
    runs.sort_by_key(|r| r.comid0);
    stats.runs = runs.len();
    (runs, stats)
}

fn build_run(order: i32, chain: &[usize], reaches: &[Reach]) -> MergedRun {
    let mut coords: Vec<[f64; 2]> = Vec::new();
    let mut seg_source: Vec<u32> = Vec::new();
    let mut comids = Vec::with_capacity(chain.len());
    for (ord, &ri) in chain.iter().enumerate() {
        comids.push(reaches[ri].comid);
        for &p in &reaches[ri].coords {
            if let Some(&last) = coords.last() {
                if (last[0] - p[0]).abs() < 1e-12 && (last[1] - p[1]).abs() < 1e-12 {
                    continue; // shared chain vertex / duplicate point
                }
                coords.push(p);
                seg_source.push(ord as u32);
            } else {
                coords.push(p);
            }
        }
    }
    MergedRun {
        order,
        comid0: reaches[chain[0]].comid,
        coords,
        seg_source,
        comids,
    }
}

// ---------------------------------------------------------------------------
// Resample (per-band 2-px spacing; vertices inherit their source reach)
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub(crate) struct Resampled {
    pub coords: Vec<[f64; 2]>,
    /// Per vertex: ordinal (into the run's `comids`) of the source reach whose
    /// series this vertex inherits.
    pub vertex_source: Vec<u32>,
}

/// Uniform arc-length resampling at `spacing_m`, endpoints preserved exactly,
/// ≥ 2 vertices. Each output vertex takes the source reach of the raw segment
/// it lands on (a vertex exactly on a reach boundary belongs upstream).
pub(crate) fn resample_run(
    coords: &[[f64; 2]],
    seg_source: &[u32],
    spacing_m: f64,
) -> Option<Resampled> {
    let n = coords.len();
    if n < 2 || seg_source.len() != n - 1 {
        return None;
    }
    let mut cum = Vec::with_capacity(n);
    cum.push(0.0f64);
    for i in 1..n {
        let d = common::haversine_distance(
            coords[i - 1][1],
            coords[i - 1][0],
            coords[i][1],
            coords[i][0],
        );
        cum.push(cum[i - 1] + d);
    }
    let total = cum[n - 1];
    if !(total > 0.0) {
        return None;
    }
    let n_seg_out = ((total / spacing_m).ceil() as usize).max(1);
    let n_out = n_seg_out + 1;
    let mut out = Resampled {
        coords: Vec::with_capacity(n_out),
        vertex_source: Vec::with_capacity(n_out),
    };
    let mut j = 0usize; // monotone source-segment cursor
    for k in 0..n_out {
        let d = total * k as f64 / n_seg_out as f64;
        while j + 2 < n && cum[j + 1] < d {
            j += 1;
        }
        let seg_len = cum[j + 1] - cum[j];
        let t = if seg_len > 0.0 {
            ((d - cum[j]) / seg_len).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let p = [
            coords[j][0] + (coords[j + 1][0] - coords[j][0]) * t,
            coords[j][1] + (coords[j + 1][1] - coords[j][1]) * t,
        ];
        out.coords.push(p);
        out.vertex_source.push(seg_source[j]);
    }
    // Exact endpoints (lerp t=0/t=1 is already exact, but be explicit).
    out.coords[0] = coords[0];
    *out.coords.last_mut().unwrap() = coords[n - 1];
    Some(out)
}

// ---------------------------------------------------------------------------
// Value encodings (design §5)
// ---------------------------------------------------------------------------

/// Round to 0.01 — the size lever: ≤ ~700 distinct f32 bit patterns.
pub(crate) fn round2(v: f32) -> f32 {
    ((v as f64 * 100.0).round() / 100.0) as f32
}

/// `round(clamp((log10 q − lo) / max(hi − lo, MIN_SPAN), 0, 1), 2)` where
/// `[lo, hi]` are the reach's own `log10`-discharge (p2, p98). Maps each reach's
/// annual low→high onto the full [0, 1] ramp so colour reads seasonal variation,
/// not absolute size; the span floor keeps near-constant reaches dim. Fill/NaN
/// (or a reach with no finite bound) → NaN → renderer fallback color.
pub(crate) fn encode_self_scaled(q: f32, lo: f32, hi: f32) -> f32 {
    if q.is_nan() || lo.is_nan() || hi.is_nan() {
        return f32::NAN;
    }
    let logq = q.max(0.01).log10();
    let denom = (hi - lo).max(SELF_SCALE_MIN_LOG_SPAN);
    round2(((logq - lo) / denom).clamp(0.0, 1.0))
}

/// `round(log10(max(q, 0.01)), 2)` ∈ [−2, 5]; fill/NaN passes through.
pub(crate) fn encode_log_q(q: f32) -> f32 {
    if q.is_nan() {
        return f32::NAN;
    }
    round2(q.max(0.01).log10())
}

/// `round(clamp(log2(q / median), 0, 6), 2)`; intermittent reaches
/// (median < 0.01 m³/s) and fill/NaN → NaN (renderer fallback color).
pub(crate) fn encode_log_anomaly(q: f32, median: f32) -> f32 {
    if median.is_nan() || median < MEDIAN_FLOOR || q.is_nan() {
        return f32::NAN;
    }
    let ratio = (q / median).max(0.0);
    round2(ratio.log2().clamp(0.0, 6.0))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: Args) -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║      🌊 NWM River Discharge Corridor Generator (retro 3.0)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let window = parse_window(&args.window)?;
    let bin = parse_bin(&args.bin)?;
    let value = parse_value(&args.value)?;
    let (zlo, zhi) = parse_zooms(&args.zooms)?;
    if bin == Bin::Hourly && window.hours > 31 * 24 {
        bail!(
            "--bin 1h over --window {} would be {} buckets; hourly is month-scoped \
             (use --window YYYY-MM)",
            window.label,
            window.hours
        );
    }
    let n_buckets = window.n_buckets(bin);
    let bucket0 = window.bucket0_ms;
    let range_end = bucket0 + n_buckets as i64 * bin.ms();

    // Overall order floor = the most permissive band (the deepest zoom). Every
    // reach at/above it is emitted ONCE at full resolution; its per-feature
    // min_zoom (density LOD) gates which zoom it first appears at.
    let min_order_all = args
        .min_order_override
        .unwrap_or_else(|| band_min_order(zhi));
    // Full-resolution vertex spacing, held at ALL zooms (no per-zoom decimation).
    let detail_spacing = spacing_m(args.detail_zoom);
    // Temporal chunking: each chunk of `chunk_buckets` matrix columns becomes its
    // own streamed temporal tile, so full-res geometry stays affordable.
    // Auto (0) → ~30 columns/tile, capping the decoded matrix on the dense
    // low-zoom overview tiles. A request ≥ the bucket count sends the whole
    // series in ONE tile per cell (geometry once). Otherwise grid-aligned
    // N-column temporal tiles (lighter per-tile, geometry re-sent per chunk).
    let requested = if args.chunk_buckets == 0 {
        AUTO_CHUNK_BUCKETS
    } else {
        args.chunk_buckets
    };
    let no_chunk = requested >= n_buckets;
    let chunk_buckets = requested.min(n_buckets.max(1));
    let (chunks, build_bucket_ms): (Vec<(usize, usize, i64, i64)>, i64) = if no_chunk {
        // One tile spanning the range. Group by the bin so every corridor
        // (start == bucket0, a bin edge) lands in the single bucket0 tile.
        (vec![(0usize, n_buckets, bucket0, range_end)], bin.ms())
    } else {
        let chunk_bucket_ms = chunk_buckets as i64 * bin.ms();
        (
            plan_time_chunks(bucket0, range_end, chunk_bucket_ms, bin.ms()),
            chunk_bucket_ms,
        )
    };
    let n_chunks = chunks.len();

    println!("📋 Configuration:");
    println!("   Flowlines: {}", args.flowlines.display());
    println!(
        "   Window:    {} ({} buckets of {})",
        window.label,
        n_buckets,
        bin.suffix()
    );
    println!("   Value:     {:?}", value);
    println!(
        "   Geometry:  full-res @ z{} (~{:.0} m spacing), order ≥{}, density LOD z{}–{}",
        args.detail_zoom, detail_spacing, min_order_all, zlo, zhi
    );
    if no_chunk {
        println!(
            "   Time:      NO chunking — 1 tile/cell, geometry once, full {}-{} matrix",
            n_buckets,
            bin.suffix()
        );
    } else {
        println!(
            "   Time:      {} chunk(s) of {} {} buckets → temporal tiles of {} ms (geometry re-sent per chunk)",
            n_chunks, chunk_buckets, bin.suffix(), build_bucket_ms
        );
    }
    println!("   Output:    {}", args.output.display());
    if let Some(n) = args.max_reach_stripes {
        println!(
            "   ⚠ Stripes limited to first {n} of {} — smoke-test mode",
            n_stripes()
        );
    }
    println!();

    let stripes: Vec<usize> =
        (0..n_stripes().min(args.max_reach_stripes.unwrap_or(usize::MAX))).collect();
    let client = http_client()?;

    // --- Stage 1: coordinate arrays + stripe-major reduce (cached, resumable)
    let t_reduce = Instant::now();
    let feature_ids = fetch_feature_ids(&client, &args)?;
    let zarr_orders = fetch_zarr_orders(&client, &args)?;
    // The persist floor must cover the geometry's reaches, not just zarr
    // order ≥ 3 — NWM order resets on divergence paths (see keep_columns).
    let geometry_fids = load_flowline_comids(&args.flowlines)?;
    ensure_reduced(
        &client,
        &args,
        &window,
        bin,
        &zarr_orders,
        &feature_ids,
        &geometry_fids,
        &stripes,
    )?;
    let medians_dir = if value == ValueEncoding::LogAnomaly {
        let year_label = window.label[..4].to_string();
        let year_window = parse_window(&year_label)?;
        let dir = reduced_dir(&args.reduced_dir, &year_window.label, Bin::Daily);
        if !(window.label == year_label && bin == Bin::Daily) {
            println!(
                "ℹ️  log-anomaly baseline = {year_label} daily medians — ensuring that reduce \
                 (full-year chunk download if uncached)"
            );
        }
        ensure_reduced(
            &client,
            &args,
            &year_window,
            Bin::Daily,
            &zarr_orders,
            &feature_ids,
            &geometry_fids,
            &stripes,
        )?;
        Some(dir)
    } else {
        None
    };
    let reduce_elapsed = t_reduce.elapsed();

    // --- Stage 2: geometry + join + merge
    let t_assemble = Instant::now();
    println!("\n🗺  Loading flowlines from {} …", args.flowlines.display());
    let (mut reaches, load_stats) = load_flowlines(&args.flowlines, min_order_all)?;
    println!(
        "   ✓ {} of {} flowlines kept (order ≥{min_order_all}); dropped: {} below order, \
         {} minor divergences <4, {} bad WKB, {} degenerate ({} multipart concatenated)",
        load_stats.kept,
        load_stats.rows,
        load_stats.below_order,
        load_stats.minor_divergence,
        load_stats.bad_wkb,
        load_stats.degenerate,
        load_stats.multipart
    );

    let wanted: HashSet<i64> = reaches.iter().map(|r| r.comid).collect();
    let table = load_series(
        &reduced_dir(&args.reduced_dir, &window.label, bin),
        &stripes,
        n_buckets,
        &feature_ids,
        &zarr_orders,
        &wanted,
    )?;
    let medians = medians_dir
        .map(|d| load_medians(&d, &stripes, &feature_ids, &wanted))
        .transpose()?;

    let mut join_misses = 0usize;
    reaches.retain(|r| {
        if table.index.contains_key(&r.comid) {
            true
        } else {
            join_misses += 1;
            false
        }
    });
    let order_mismatches = reaches
        .iter()
        .filter(|r| table.zarr_order[table.index[&r.comid] as usize] != r.order)
        .count();
    let missing_median = medians.as_ref().map(|m| {
        reaches
            .iter()
            .filter(|r| !m.get(&r.comid).is_some_and(|v| v.is_finite()))
            .count()
    });
    println!(
        "   ✓ COMID ↔ feature_id join: {} matched, {} misses ({:.2}%); \
         {} zarr-order/StreamOrde mismatches (expected on divergence-affected \
         paths — NWM order resets StreamCalc-style)",
        reaches.len(),
        join_misses,
        100.0 * join_misses as f64 / (reaches.len() + join_misses).max(1) as f64,
        order_mismatches
    );
    if let Some(mm) = missing_median {
        println!("   ℹ {mm} reaches lack a finite 2019 median → NaN anomaly (fallback color)");
    }
    if reaches.is_empty() {
        bail!("no reaches survived the join — is the reduce complete for this window?");
    }

    let (runs, merge_stats) = merge_runs(&reaches);
    println!(
        "   ✓ mainstem merge: {} reaches → {} runs ({} multi-reach chains, \
         {} discontinuity breaks)",
        reaches.len(),
        merge_stats.runs,
        merge_stats.merged_chains,
        merge_stats.discontinuity_breaks
    );
    drop(reaches);

    // --- Stage 3: single full-res resample + splat + time-chunk + write.
    // Each run is resampled ONCE at the detail spacing (kept at every zoom) and
    // emitted once PER TIME CHUNK: same geometry, min_zoom by order, matrix
    // sliced to the chunk's columns. The chunk's start time lands it in its own
    // temporal tile so the client streams only the window it is playing.
    let (intermediate_path, output_is_intermediate) = resolve_intermediate(&args.output);
    let property_columns = vec![
        PropertyColumn::numeric("min_zoom"),
        PropertyColumn::numeric("max_zoom"),
        PropertyColumn::numeric("order"),
        PropertyColumn::numeric("width"),
    ];
    let mut writer = StreamingLineStringParquetWriter::with_columns_and_batch_rows(
        &intermediate_path,
        property_columns,
        WRITER_BATCH_ROWS,
    )?;

    let mut total_features = 0usize; // emitted rows (runs × chunks)
    let mut total_geom_verts = 0usize; // distinct geometry vertices (runs, once)
    let mut kept_runs = 0usize;
    let mut degenerate_runs = 0usize;
    // Per-min_zoom accounting for the density-LOD summary.
    let mut per_minzoom: BTreeMap<u8, (usize, usize)> = BTreeMap::new(); // z → (runs, verts)
    for run in &runs {
        if run.order < min_order_all {
            continue;
        }
        let Some(rs) = resample_run(&run.coords, &run.seg_source, detail_spacing) else {
            degenerate_runs += 1;
            continue;
        };
        let nverts = rs.coords.len();
        let min_zoom = min_zoom_for_order(run.order, zlo, zhi);
        let width = width_for_order(run.order);
        // Encode each source reach's FULL series once; chunks slice into it.
        let encoded: Vec<Vec<f32>> = run
            .comids
            .iter()
            .map(|&c| {
                let series = table.series(table.index[&c]);
                match value {
                    ValueEncoding::SelfScaled => {
                        // Normalise each reach to its OWN annual [p2, p98] so
                        // colour reads seasonal variation, not absolute size.
                        let (lo, hi) = log_percentile_bounds(series);
                        series
                            .iter()
                            .map(|&q| encode_self_scaled(q, lo, hi))
                            .collect()
                    }
                    ValueEncoding::LogQ => series.iter().map(|&q| encode_log_q(q)).collect(),
                    ValueEncoding::LogAnomaly => {
                        let med = medians
                            .as_ref()
                            .and_then(|m| m.get(&c))
                            .copied()
                            .unwrap_or(f32::NAN);
                        series.iter().map(|&q| encode_log_anomaly(q, med)).collect()
                    }
                }
            })
            .collect();
        for &(b0, b1, c_start, c_end) in &chunks {
            let cb = b1 - b0;
            let mut matrix = vec![0.0f32; nverts * cb];
            for (v, &src) in rs.vertex_source.iter().enumerate() {
                matrix[v * cb..(v + 1) * cb].copy_from_slice(&encoded[src as usize][b0..b1]);
            }
            let mut props = Map::new();
            props.insert("min_zoom".to_string(), json!(min_zoom));
            props.insert("max_zoom".to_string(), json!(zhi));
            props.insert("order".to_string(), json!(run.order));
            props.insert("width".to_string(), json!(width));
            writer.write_linestring(&LineStringRecord {
                coordinates: rs.coords.clone(),
                timestamp_ms: c_start,
                end_timestamp_ms: Some(c_end),
                vertex_timestamps_ms: None,
                vertex_values: None,
                vertex_value_matrix: Some(matrix),
                properties: props,
            })?;
            total_features += 1;
        }
        let e = per_minzoom.entry(min_zoom).or_default();
        e.0 += 1;
        e.1 += nverts;
        kept_runs += 1;
        total_geom_verts += nverts;
    }
    for (z, (runs_z, verts_z)) in &per_minzoom {
        println!(
            "   min_zoom z{z}: {runs_z} rivers · {verts_z} verts (full-res, shown z{z}–{zhi})"
        );
    }
    let rows = writer.finish()?;
    if degenerate_runs > 0 {
        println!("   ⚠ {degenerate_runs} runs skipped (zero-length geometry)");
    }
    let assemble_elapsed = t_assemble.elapsed();

    // Decoded matrix per LOADED temporal tile ≈ geom_verts × chunk_buckets × 4 B
    // (one chunk resident at a time), vs the whole-year cost if unchunked.
    // Decoded matrix RESIDENT for one loaded temporal window: geom_verts ×
    // cols-per-tile × 4 B. No-chunk carries the whole year in every tile.
    let cols_per_tile = if no_chunk { n_buckets } else { chunk_buckets };
    println!(
        "\n   ✓ {kept_runs} rivers × {n_chunks} time-tile(s) = {total_features} rows ({rows} written) · \
         {total_geom_verts} geom verts · matrix/tile {:.1} MB decoded · total cells {:.2} GB",
        total_geom_verts as f64 * cols_per_tile as f64 * 4.0 / 1e6,
        total_geom_verts as f64 * n_buckets as f64 * 4.0 / 1e9
    );
    println!("   ⏱  matrix span (set showcase timeRange to this):");
    println!("       start = {bucket0}");
    println!("       end   = {range_end}");
    if rows == 0 {
        bail!("no corridors emitted — check the order thresholds / join");
    }

    if args.skip_build || output_is_intermediate {
        println!(
            "\n📦 Skipping stt-build (intermediate written to {})",
            intermediate_path.display()
        );
        println!(
            "⏱  stage times: reduce {:.1?} · assemble {:.1?}",
            reduce_elapsed, assemble_elapsed
        );
        return Ok(());
    }

    // --- Stage 4: build. Matrix corridors REQUIRE no simplification
    // (stt-build accepts a supplied matrix only with simplify off — its
    // default). The temporal bucket = the CHUNK span (ms), so each chunk's rows
    // (all sharing `c_start`) land in their own temporal tile — the client
    // streams one time window at a time. `min_zoom`/`max_zoom` fields confine
    // each river to its density-LOD band [min_zoom_by_order, zhi] as ONE full-res
    // copy; ~10 m coordinate quantization halves the geometry share.
    let t_build = Instant::now();
    common::run_stt_build_with_full_options(SttBuildOptions {
        input: intermediate_path.clone(),
        output: args.output.clone(),
        time_field: "timestamp".to_string(),
        end_time_field: Some("end_timestamp".to_string()),
        min_zoom: zlo,
        max_zoom: zhi,
        compression: "zstd".to_string(),
        temporal_bucket: Some(build_bucket_ms.to_string()),
        temporal_lod: None,
        summary: None,
        summary_sub_buckets: None,
        min_features_per_tile: None,
        min_zoom_field: Some("min_zoom".to_string()),
        max_zoom_field: Some("max_zoom".to_string()),
        no_clip: false,
        quantize_coords: Some(10.0),
        quantize_attrs: Vec::new(),
        blob_ordering: None,
    })?;
    let build_elapsed = t_build.elapsed();

    println!("\n✅ NWM river archive built: {}", args.output.display());
    println!("   Set datasets.ts timeRange to {{ start: {bucket0}, end: {range_end} }}");
    println!(
        "⏱  stage times: reduce {:.1?} · assemble {:.1?} · build {:.1?}",
        reduce_elapsed, assemble_elapsed, build_elapsed
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (no network: fixtures are hand-built zstd frames + synthetic reaches)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- density LOD + temporal chunk plan ----------------------------------

    #[test]
    fn min_zoom_ladder_inverts_band_floor() {
        // Big rivers appear from the lowest zoom; smaller ones only deeper.
        // Ladder: z≤5→≥6, z6-7→≥5, z8→≥4 over the z4–8 pyramid.
        assert_eq!(min_zoom_for_order(8, 4, 8), 4); // order 8 shows at z4
        assert_eq!(min_zoom_for_order(6, 4, 8), 4);
        assert_eq!(min_zoom_for_order(5, 4, 8), 6); // order 5 first at z6
        assert_eq!(min_zoom_for_order(4, 4, 8), 8); // order 4 only at z8
                                                    // Below the overall floor never reaches here (filtered), but clamp holds.
        assert_eq!(min_zoom_for_order(3, 4, 8), 8);
        // Every emitted river's min_zoom is a single value in [zlo, zhi].
        for o in 4..=10 {
            let mz = min_zoom_for_order(o, 4, 8);
            assert!((4..=8).contains(&mz));
        }
    }

    #[test]
    fn time_chunks_grid_aligned_and_contiguous() {
        // 2019 daily, 30-day chunks (the year-demo defaults).
        let bucket0 = 1_546_300_800_000i64; // 2019-01-01, a daily edge
        let bin_ms = 86_400_000i64;
        let n_buckets = 365i64;
        let range_end = bucket0 + n_buckets * bin_ms;
        let chunk_ms = 30 * bin_ms;
        let chunks = plan_time_chunks(bucket0, range_end, chunk_ms, bin_ms);

        // 365 days over 30-day cells → 13 chunks (partial first/last).
        assert_eq!(chunks.len(), 13);
        // Contiguous cover of the whole [0, 365) bucket range, no gaps/overlap.
        assert_eq!(chunks[0].0, 0);
        assert_eq!(chunks.last().unwrap().1, n_buckets as usize);
        for w in chunks.windows(2) {
            assert_eq!(w[0].1, w[1].0, "gap/overlap between chunk bucket ranges");
        }
        // Each chunk's start floors to a DISTINCT, CONSECUTIVE grid cell — this
        // is the tile address the client looks up. Misalignment here loads the
        // wrong chunk mid-window.
        let cells: Vec<i64> = chunks.iter().map(|c| c.2.div_euclid(chunk_ms)).collect();
        for w in cells.windows(2) {
            assert_eq!(
                w[1],
                w[0] + 1,
                "chunk starts must map to consecutive grid cells"
            );
        }
        // Interior chunk starts sit exactly on a grid edge (only the first is a
        // partial pre-roll cell, offset from the epoch grid by bucket0).
        for c in &chunks[1..] {
            assert_eq!(c.2 % chunk_ms, 0, "interior chunk start not grid-aligned");
        }
        // start_ms/end_ms are consistent with the bucket indices.
        for &(b0, b1, s, e) in &chunks {
            assert_eq!(s, bucket0 + b0 as i64 * bin_ms);
            assert_eq!(e, bucket0 + b1 as i64 * bin_ms);
            assert!(b1 > b0);
        }
    }

    #[test]
    fn time_chunks_large_chunk_still_covers_window() {
        // A chunk larger than the window still covers [0, n) contiguously — but
        // grid-alignment to the EPOCH (not the window) means it may split into
        // two cells when the window straddles a grid boundary. Coverage, not
        // count, is the invariant.
        let bucket0 = 1_546_300_800_000i64;
        let bin_ms = 86_400_000i64;
        let n_buckets = 365usize;
        let range_end = bucket0 + n_buckets as i64 * bin_ms;
        let chunks = plan_time_chunks(bucket0, range_end, 400 * bin_ms, bin_ms);
        assert!(!chunks.is_empty() && chunks.len() <= 2);
        assert_eq!(chunks[0].0, 0);
        assert_eq!(chunks.last().unwrap().1, n_buckets);
        for w in chunks.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
    }

    // -- time axis / chunk index math ---------------------------------------

    #[test]
    fn time_epoch_matches_chrono() {
        let e = chrono::NaiveDate::from_ymd_opt(1979, 2, 1)
            .unwrap()
            .and_hms_opt(1, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        assert_eq!(e, TIME_EPOCH_MS);
    }

    #[test]
    fn window_2019_hour_index_and_chunks() {
        let w = parse_window("2019").unwrap();
        assert_eq!(w.start_hour, 349_895); // verified index of 2019-01-01T00Z
        assert_eq!(w.hours, 8_760);
        assert_eq!(w.bucket0_ms, 1_546_300_800_000);
        assert_eq!(time_chunk_range(&w), 520..=533); // 14 chunks
        assert_eq!(w.n_buckets(Bin::Daily), 365);
    }

    #[test]
    fn window_2019_03_hour_index_and_chunks() {
        let w = parse_window("2019-03").unwrap();
        assert_eq!(w.start_hour, 351_311); // verified index of 2019-03-01T00Z
        assert_eq!(w.hours, 744);
        assert_eq!(w.bucket0_ms, 1_551_398_400_000);
        assert_eq!(time_chunk_range(&w), 522..=523); // 2 chunks
        assert_eq!(w.n_buckets(Bin::Hourly), 744);
    }

    #[test]
    fn window_rejects_garbage() {
        assert!(parse_window("19").is_err());
        assert!(parse_window("2019-13").is_err());
        assert!(parse_window("1978-12").is_err()); // before the time axis
        assert!(parse_window("2030").is_err()); // past the retrospective
        assert!(parse_window("hello").is_err());
    }

    #[test]
    fn keep_columns_unions_zarr_order_and_geometry() {
        // Stripe 1 (global 30000..60000): col 5 has zarr order 5 (kept), col 9
        // has zarr order 1 but its feature_id is in the geometry (kept — the
        // divergence-path case), col 12 has order 1 and no geometry (dropped).
        let mut zarr_orders = vec![1i32; 60_000];
        let mut feature_ids = vec![0i64; 60_000];
        zarr_orders[30_005] = 5;
        feature_ids[30_005] = 501;
        feature_ids[30_009] = 909; // zarr order stays 1
        feature_ids[30_012] = 333; // zarr order 1, not in geometry
        let geometry: HashSet<i64> = [909].into_iter().collect();
        let keep = keep_columns(&zarr_orders, &feature_ids, &geometry, 1);
        assert!(keep.contains(&5));
        assert!(keep.contains(&9));
        assert!(!keep.contains(&12));
        assert_eq!(keep.len(), 2);
    }

    #[test]
    fn hour_and_reach_chunk_math() {
        assert_eq!(hour_chunk(349_895), (520, 455));
        assert_eq!(hour_chunk(351_311), (522, 527));
        assert_eq!(hour_chunk(0), (0, 0));
        assert_eq!(reach_chunk(0), (0, 0));
        assert_eq!(reach_chunk(29_999), (0, 29_999));
        assert_eq!(reach_chunk(30_000), (1, 0));
        assert_eq!(reach_chunk(N_REACHES - 1), (92, 16_733));
        assert_eq!(n_stripes(), 93);
    }

    #[test]
    fn ladder_spacing_width() {
        assert_eq!(band_min_order(4), 6);
        assert_eq!(band_min_order(5), 6);
        assert_eq!(band_min_order(6), 5);
        assert_eq!(band_min_order(7), 5);
        assert_eq!(band_min_order(8), 4);
        assert_eq!(band_min_order(9), 4);
        assert!((spacing_m(6) - 3854.9).abs() < 0.1);
        assert!((spacing_m(8) - 963.7).abs() < 0.1);
        assert_eq!(width_for_order(3), 0.5);
        assert_eq!(width_for_order(4), 1.0);
        assert_eq!(width_for_order(6), 4.0);
        assert_eq!(width_for_order(10), 16.0); // clamped
        assert!(drop_minor_divergence(3, 2));
        assert!(!drop_minor_divergence(4, 2));
        assert!(!drop_minor_divergence(3, 1));
    }

    #[test]
    fn parse_zooms_cases() {
        assert_eq!(parse_zooms("4-8").unwrap(), (4, 8));
        assert_eq!(parse_zooms("6-7").unwrap(), (6, 7));
        assert_eq!(parse_zooms("6").unwrap(), (6, 6));
        assert!(parse_zooms("8-4").is_err());
        assert!(parse_zooms("0-8").is_err());
        assert!(parse_zooms("abc").is_err());
    }

    #[test]
    fn parse_bin_and_value() {
        assert_eq!(parse_bin("1d").unwrap(), Bin::Daily);
        assert_eq!(parse_bin("1h").unwrap(), Bin::Hourly);
        assert!(parse_bin("6h").is_err());
        assert_eq!(
            parse_value("self-scaled").unwrap(),
            ValueEncoding::SelfScaled
        );
        assert_eq!(parse_value("log-q").unwrap(), ValueEncoding::LogQ);
        assert_eq!(
            parse_value("log-anomaly").unwrap(),
            ValueEncoding::LogAnomaly
        );
        assert!(parse_value("linear").is_err());
    }

    // -- chunk decode + scaling ---------------------------------------------

    #[test]
    fn decode_i32_frame_zstd_fixture() {
        // Hand-built 2×3 C-order grid as a bare zstd frame — the real chunks
        // are the same encoding at [672 × 30000].
        let cells: [i32; 6] = [12_345, FILL_I32, 0, -100, 1, 700_000];
        let mut raw = Vec::new();
        for c in cells {
            raw.extend_from_slice(&c.to_le_bytes());
        }
        let frame = zstd::encode_all(&raw[..], 3).unwrap();
        let decoded = decode_i32_frame(&frame, 6).unwrap();
        assert_eq!(decoded, cells);
        // truncated frame is rejected
        assert!(decode_i32_frame(&frame, 7).is_err());
    }

    #[test]
    fn scale_flow_cases() {
        assert_eq!(scale_flow(12_345), 123.45);
        assert_eq!(scale_flow(0), 0.0);
        assert_eq!(scale_flow(-100), -1.0);
        assert!(scale_flow(FILL_I32).is_nan());
    }

    // -- reduce -------------------------------------------------------------

    fn test_window(start_hour: usize, hours: usize) -> WindowSpec {
        WindowSpec {
            label: "test".into(),
            start_hour,
            hours,
            bucket0_ms: 0,
        }
    }

    /// 48 rows × 2 cols: col0 = 100 (day 0) then 200 (day 1); col1 = 300 in
    /// hour 0 only, fill elsewhere.
    fn two_day_data() -> Vec<i32> {
        let mut data = vec![FILL_I32; 48 * 2];
        for h in 0..48 {
            data[h * 2] = if h < 24 { 100 } else { 200 };
        }
        data[1] = 300; // col1, hour 0
        data
    }

    #[test]
    fn daily_mean_reduce_with_fills() {
        let w = test_window(0, 48);
        let mut acc = Accum::new(vec![0, 1], 2);
        acc.ingest_chunk(0, &two_day_data(), 2, &w, Bin::Daily);
        let (keep, values, medians) = acc.finalize(Bin::Daily);
        assert_eq!(keep, vec![0, 1]);
        assert_eq!(&values[..2], &[1.0, 2.0]); // reach 0 daily means
        assert_eq!(values[2], 3.0); // reach 1 day 0: single valid sample
        assert!(values[3].is_nan()); // reach 1 day 1: all fill → NaN
        let medians = medians.unwrap();
        assert_eq!(medians[0], 1.5); // median of [1, 2]
        assert_eq!(medians[1], 3.0); // NaN skipped
    }

    #[test]
    fn reduce_split_across_chunks_matches_single() {
        let w = test_window(0, 48);
        let data = two_day_data();
        let mut whole = Accum::new(vec![0, 1], 2);
        whole.ingest_chunk(0, &data, 2, &w, Bin::Daily);
        let (_, v1, m1) = whole.finalize(Bin::Daily);

        let mut split = Accum::new(vec![0, 1], 2);
        split.ingest_chunk(0, &data[..24 * 2], 2, &w, Bin::Daily);
        split.ingest_chunk(24, &data[24 * 2..], 2, &w, Bin::Daily);
        let (_, v2, m2) = split.finalize(Bin::Daily);

        assert_eq!(format!("{v1:?}"), format!("{v2:?}")); // NaN-safe compare
        assert_eq!(m1, m2);
    }

    #[test]
    fn reduce_clips_chunk_to_window() {
        // Window starts at hour 5 inside a chunk that starts at hour 0.
        let w = test_window(5, 4);
        let mut data = vec![FILL_I32; 12];
        for (h, item) in data.iter_mut().enumerate() {
            *item = h as i32 * 100;
        }
        let mut acc = Accum::new(vec![0], 4);
        acc.ingest_chunk(0, &data, 1, &w, Bin::Hourly);
        let (_, values, medians) = acc.finalize(Bin::Hourly);
        assert_eq!(values, vec![5.0, 6.0, 7.0, 8.0]);
        assert!(medians.is_none());
    }

    #[test]
    fn median_odd_even_nan() {
        assert_eq!(median_non_nan(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_non_nan(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(median_non_nan(&[f32::NAN, 5.0, f32::NAN]), 5.0);
        assert!(median_non_nan(&[f32::NAN, f32::NAN]).is_nan());
        assert!(median_non_nan(&[]).is_nan());
    }

    #[test]
    fn stripe_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stripe-007.bin");
        let indices = vec![3u32, 17, 29_999];
        let values = vec![1.5f32, f32::NAN, 0.0, -1.0, 2.5, 3.5];
        let medians = vec![1.0f32, f32::NAN, 3.0];
        write_stripe(&path, 7, 2, &indices, &values, Some(&medians)).unwrap();
        let sd = read_stripe(&path).unwrap();
        assert_eq!(sd.stripe, 7);
        assert_eq!(sd.n_buckets, 2);
        assert_eq!(sd.indices, indices);
        assert_eq!(format!("{:?}", sd.values), format!("{values:?}"));
        assert_eq!(format!("{:?}", sd.medians.unwrap()), format!("{medians:?}"));

        // without medians (hourly)
        let path2 = dir.path().join("stripe-008.bin");
        write_stripe(&path2, 8, 2, &indices, &values, None).unwrap();
        assert!(read_stripe(&path2).unwrap().medians.is_none());
    }

    // -- encodings ----------------------------------------------------------

    #[test]
    fn encode_log_q_cases() {
        assert_eq!(encode_log_q(100.0), 2.0);
        assert_eq!(encode_log_q(1.0), 0.0);
        assert_eq!(encode_log_q(0.0), -2.0); // floored at 0.01
        assert_eq!(encode_log_q(-5.0), -2.0); // negative flow floored
        assert_eq!(encode_log_q(0.005), -2.0);
        assert_eq!(encode_log_q(3.0), 0.48); // log10(3)=0.477… → 0.48
        assert_eq!(encode_log_q(100_000.0), 5.0);
        assert!(encode_log_q(f32::NAN).is_nan());
    }

    #[test]
    fn encode_log_anomaly_cases() {
        assert_eq!(encode_log_anomaly(4.0, 1.0), 2.0);
        assert_eq!(encode_log_anomaly(1.0, 1.0), 0.0);
        assert_eq!(encode_log_anomaly(0.5, 1.0), 0.0); // clamped low
        assert_eq!(encode_log_anomaly(0.0, 1.0), 0.0); // log2(0) → −inf → 0
        assert_eq!(encode_log_anomaly(-3.0, 1.0), 0.0); // negative flow → 0
        assert_eq!(encode_log_anomaly(1000.0, 1.0), 6.0); // clamped high
        assert_eq!(encode_log_anomaly(3.0, 1.0), 1.58); // log2(3)=1.585 → 1.58
        assert!(encode_log_anomaly(1.0, 0.005).is_nan()); // intermittent median
        assert!(encode_log_anomaly(1.0, f32::NAN).is_nan());
        assert!(encode_log_anomaly(f32::NAN, 1.0).is_nan());
    }

    #[test]
    fn round2_cases() {
        assert_eq!(round2(0.477_121_3), 0.48);
        assert_eq!(round2(-1.234), -1.23);
        assert_eq!(round2(2.0), 2.0);
    }

    #[test]
    fn log_percentile_bounds_cases() {
        // 0..=100 m³/s: p2 ≈ log10(2)=0.30, p98 ≈ log10(98)=1.99 (nearest-rank).
        let series: Vec<f32> = (0..=100).map(|i| i as f32).collect();
        let (lo, hi) = log_percentile_bounds(&series);
        assert!((lo - 2.0f32.log10()).abs() < 1e-4, "lo={lo}");
        assert!((hi - 98.0f32.log10()).abs() < 1e-4, "hi={hi}");
        // Fills are ignored; a single finite sample gives lo == hi.
        let (lo1, hi1) = log_percentile_bounds(&[f32::NAN, 50.0, f32::NAN]);
        assert_eq!(lo1, 50.0f32.log10());
        assert_eq!(hi1, 50.0f32.log10());
        // All-fill reach → no bound.
        let (lo0, hi0) = log_percentile_bounds(&[f32::NAN, f32::NAN]);
        assert!(lo0.is_nan() && hi0.is_nan());
    }

    #[test]
    fn encode_self_scaled_cases() {
        // A reach whose own range is 1 → 1000 m³/s (log10 span 3.0, well above
        // the floor): its low pins to 0, its high to 1, geometric mid to 0.5.
        let (lo, hi) = (0.0f32, 3.0f32); // log10(1), log10(1000)
        assert_eq!(encode_self_scaled(1.0, lo, hi), 0.0);
        assert_eq!(encode_self_scaled(1000.0, lo, hi), 1.0);
        assert_eq!(encode_self_scaled(31.62, lo, hi), 0.5); // 10^1.5
                                                            // Absolute size does NOT set the colour — a huge reach scaled to its own
                                                            // [10^4, 10^6] range reports the SAME 0.5 at its geometric midpoint.
        let (blo, bhi) = (4.0f32, 6.0f32);
        assert_eq!(encode_self_scaled(100_000.0, blo, bhi), 0.5); // 10^5
                                                                  // Below-p2 / above-p98 samples clamp to the ends.
        assert_eq!(encode_self_scaled(0.1, lo, hi), 0.0);
        assert_eq!(encode_self_scaled(1e6, lo, hi), 1.0);
        // Span floor: a near-constant reach (log span 0.1 < 0.30) only reaches a
        // proportional fraction of the ramp, so it stays dim.
        assert_eq!(
            encode_self_scaled(10.0f32.powf(0.1), 0.0, 0.1),
            round2(0.1 / 0.30)
        );
        // Undefined bounds / fill → NaN → renderer fallback.
        assert!(encode_self_scaled(f32::NAN, lo, hi).is_nan());
        assert!(encode_self_scaled(5.0, f32::NAN, f32::NAN).is_nan());
    }

    // -- mainstem merge -----------------------------------------------------

    fn reach(
        comid: i64,
        order: i32,
        level_path: i64,
        hydroseq: i64,
        dn_hydroseq: i64,
        coords: &[[f64; 2]],
    ) -> Reach {
        Reach {
            comid,
            order,
            level_path,
            hydroseq,
            dn_hydroseq,
            coords: coords.to_vec(),
        }
    }

    #[test]
    fn merge_chain_merges() {
        let reaches = vec![
            reach(1, 5, 100, 30, 20, &[[0.0, 0.0], [0.01, 0.0]]),
            reach(2, 5, 100, 20, 10, &[[0.01, 0.0], [0.02, 0.0]]),
            reach(3, 5, 100, 10, 0, &[[0.02, 0.0], [0.03, 0.0]]),
        ];
        let (runs, stats) = merge_runs(&reaches);
        assert_eq!(runs.len(), 1);
        assert_eq!(stats.merged_chains, 1);
        assert_eq!(stats.discontinuity_breaks, 0);
        let run = &runs[0];
        assert_eq!(run.comid0, 1);
        assert_eq!(run.comids, vec![1, 2, 3]);
        assert_eq!(
            run.coords,
            vec![[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0]]
        );
        assert_eq!(run.seg_source, vec![0, 1, 2]);
    }

    #[test]
    fn merge_order_change_breaks_run() {
        // Same level path, but the downstream reach jumps to order 6 (a
        // confluence) — different (LevelPathI, order) group ⇒ separate run.
        let reaches = vec![
            reach(1, 5, 100, 30, 20, &[[0.0, 0.0], [0.01, 0.0]]),
            reach(2, 6, 100, 20, 0, &[[0.01, 0.0], [0.02, 0.0]]),
        ];
        let (runs, stats) = merge_runs(&reaches);
        assert_eq!(runs.len(), 2);
        assert_eq!(stats.merged_chains, 0);
        assert_eq!(runs[0].comids, vec![1]);
        assert_eq!(runs[1].comids, vec![2]);
    }

    #[test]
    fn merge_dn_gap_breaks_run() {
        // B's DnHydroseq points outside the group (e.g. a dropped divergence):
        // A→B merges, C starts its own run.
        let reaches = vec![
            reach(1, 5, 100, 30, 20, &[[0.0, 0.0], [0.01, 0.0]]),
            reach(2, 5, 100, 20, 99, &[[0.01, 0.0], [0.02, 0.0]]),
            reach(3, 5, 100, 10, 0, &[[0.02, 0.0], [0.03, 0.0]]),
        ];
        let (runs, _) = merge_runs(&reaches);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].comids, vec![1, 2]);
        assert_eq!(runs[1].comids, vec![3]);
    }

    #[test]
    fn merge_discontinuity_breaks_and_leftover_walks() {
        // A's DnHydroseq names B, but B's geometry starts somewhere else
        // (divergence artifact): the chain breaks at A, and B — no longer a
        // head — still walks its own run into C on the leftover pass.
        let reaches = vec![
            reach(1, 5, 100, 30, 20, &[[0.0, 0.0], [0.01, 0.0]]),
            reach(2, 5, 100, 20, 10, &[[0.5, 0.5], [0.51, 0.5]]),
            reach(3, 5, 100, 10, 0, &[[0.51, 0.5], [0.52, 0.5]]),
        ];
        let (runs, stats) = merge_runs(&reaches);
        assert_eq!(stats.discontinuity_breaks, 1);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].comids, vec![1]);
        assert_eq!(runs[1].comids, vec![2, 3]);
    }

    #[test]
    fn merge_divergent_braid_keeps_minor_path_separate() {
        // A minor divergence with its own LevelPathI must not merge into the
        // mainstem even where geometry touches.
        let reaches = vec![
            reach(1, 5, 100, 30, 20, &[[0.0, 0.0], [0.01, 0.0]]),
            reach(2, 5, 100, 20, 0, &[[0.01, 0.0], [0.02, 0.0]]),
            reach(9, 5, 200, 25, 20, &[[0.01, 0.0], [0.015, 0.01]]), // braid, other level path
        ];
        let (runs, _) = merge_runs(&reaches);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].comids, vec![1, 2]);
        assert_eq!(runs[1].comids, vec![9]);
    }

    #[test]
    fn merge_deterministic_regardless_of_input_order() {
        let a = vec![
            reach(1, 5, 100, 30, 20, &[[0.0, 0.0], [0.01, 0.0]]),
            reach(2, 5, 100, 20, 10, &[[0.01, 0.0], [0.02, 0.0]]),
            reach(3, 5, 100, 10, 0, &[[0.02, 0.0], [0.03, 0.0]]),
            reach(9, 6, 200, 40, 0, &[[1.0, 1.0], [1.01, 1.0]]),
        ];
        let mut b = a.clone();
        b.reverse();
        let (runs_a, _) = merge_runs(&a);
        let (runs_b, _) = merge_runs(&b);
        assert_eq!(runs_a, runs_b);
        // emitted order sorted by comid0
        let keys: Vec<i64> = runs_a.iter().map(|r| r.comid0).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
    }

    // -- resample -----------------------------------------------------------

    #[test]
    fn resample_inherits_source_series() {
        // Two equal reaches along the equator; resample to 5 vertices: the
        // boundary vertex belongs to the upstream reach, later ones to the
        // downstream reach.
        let coords = vec![[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]];
        let seg_source = vec![0u32, 1];
        let total = common::haversine_distance(0.0, 0.0, 0.0, 0.1);
        let rs = resample_run(&coords, &seg_source, total / 3.9).unwrap();
        assert_eq!(rs.coords.len(), 5);
        assert_eq!(rs.vertex_source, vec![0, 0, 0, 1, 1]);
        let lons: Vec<f64> = rs.coords.iter().map(|c| c[0]).collect();
        for (got, want) in lons.iter().zip([0.0, 0.025, 0.05, 0.075, 0.1]) {
            assert!((got - want).abs() < 1e-9, "lon {got} != {want}");
        }
        // endpoints exact
        assert_eq!(rs.coords[0], [0.0, 0.0]);
        assert_eq!(*rs.coords.last().unwrap(), [0.1, 0.0]);
    }

    #[test]
    fn resample_two_vertex_floor() {
        // A reach far shorter than the spacing still gets its 2 endpoints.
        let coords = vec![[0.0, 0.0], [0.001, 0.0]];
        let rs = resample_run(&coords, &[0], 50_000.0).unwrap();
        assert_eq!(rs.coords.len(), 2);
        assert_eq!(rs.coords, coords);
        assert_eq!(rs.vertex_source, vec![0, 0]);
    }

    #[test]
    fn resample_rejects_degenerate() {
        assert!(resample_run(&[[0.0, 0.0]], &[], 100.0).is_none());
        assert!(resample_run(&[[0.0, 0.0], [0.0, 0.0]], &[0], 100.0).is_none());
    }

    // -- WKB ----------------------------------------------------------------

    fn wkb_linestring(pts: &[[f64; 2]]) -> Vec<u8> {
        common::encode_wkb_linestring(pts)
    }

    #[test]
    fn wkb_parses_linestring_and_multilinestring() {
        let pts = [[-100.5, 38.25], [-100.6, 38.30]];
        let parts = parse_wkb_linework(&wkb_linestring(&pts)).unwrap();
        assert_eq!(parts, vec![pts.to_vec()]);

        // MultiLineString of two parts
        let mut multi = vec![1u8];
        multi.extend_from_slice(&5u32.to_le_bytes());
        multi.extend_from_slice(&2u32.to_le_bytes());
        multi.extend_from_slice(&wkb_linestring(&pts));
        let pts2 = [[-100.6, 38.30], [-100.7, 38.35], [-100.8, 38.4]];
        multi.extend_from_slice(&wkb_linestring(&pts2));
        let parts = parse_wkb_linework(&multi).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], pts.to_vec());
        assert_eq!(parts[1], pts2.to_vec());
    }

    #[test]
    fn wkb_rejects_polygons_and_3d() {
        // polygon type 3
        let mut poly = vec![1u8];
        poly.extend_from_slice(&3u32.to_le_bytes());
        poly.extend_from_slice(&0u32.to_le_bytes());
        assert!(parse_wkb_linework(&poly).is_err());
        // ISO Z linestring (1002)
        let mut z = vec![1u8];
        z.extend_from_slice(&1002u32.to_le_bytes());
        assert!(parse_wkb_linework(&z).is_err());
        // truncated
        assert!(parse_wkb_linework(&[1u8, 2, 0]).is_err());
    }
}
