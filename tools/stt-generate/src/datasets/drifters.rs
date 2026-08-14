//! Generate ocean-current trajectories from NOAA's Global Drifter Program.
//!
//! Data source: PMEL ERDDAP `gdp_interpolated_drifter` — the 6-hour
//! quality-controlled, krigged interpolated product of satellite-tracked
//! surface drifting buoys (1979 → Nov 2022). Public domain (CC0).
//!   <https://data.pmel.noaa.gov/generic/erddap/tabledap/gdp_interpolated_drifter.html>
//!
//! Each drifter's 6-hourly fixes are grouped into a track, split at large
//! time gaps and antimeridian crossings, and emitted as LineStrings with
//! per-vertex real timestamps so the deck.gl trips layer animates the buoy
//! drifting along the current. Tracks are colour-banded by sea-surface
//! temperature so warm western-boundary currents (Gulf Stream, Kuroshio)
//! read as bright ribbons peeling poleward.
//!
//! The pipeline itself ([`run_product`]) is shared with the hourly variant
//! (`drifters_hourly`), which only swaps the [`Product`] knobs.

use crate::common::{
    self, LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter, SttBuildOptions,
};
use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Utc};
use clap::Parser;
use csv::ReaderBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::process::Command;

/// The knobs that differ between the 6-hourly and hourly GDP products. Both
/// subcommands run the identical pipeline (`run_product`) over one of these.
pub(crate) struct Product {
    pub erddap_base: &'static str,
    /// Temperature column name in the ERDDAP schema.
    pub temp_column: &'static str,
    /// Raw column value → °C (the hourly product reports Kelvin and needs a
    /// plausibility filter for fill sentinels; the 6-hourly is already °C).
    pub temp_to_celsius: fn(f64) -> Option<f64>,
    /// Monthly cache file name. Only the hourly key encodes the bounds tag —
    /// the 6-hourly name predates it and must keep hitting existing caches.
    pub cache_file: fn(i32, u32, &str) -> String,
    pub banner: &'static str,
    /// Product noun in the fetch progress line ("6-hourly" / "hourly").
    pub fetch_noun: &'static str,
    pub done_message: &'static str,
}

const SIX_HOURLY: Product = Product {
    erddap_base: "https://data.pmel.noaa.gov/generic/erddap/tabledap/gdp_interpolated_drifter.csv",
    temp_column: "temp",
    temp_to_celsius: |c| Some(c),
    cache_file: |y, m, _bounds_tag| format!("drifters-{:04}-{:02}.csv", y, m),
    banner: "🌊 Global Drifter Program Generator\n===================================\n",
    fetch_noun: "6-hourly",
    done_message: "\n✅ Drifter generation complete!",
};

#[derive(Parser, Debug)]
#[command(about = "Generate Global Drifter Program ocean-current trajectories")]
pub struct Args {
    /// Output file (.stt or .parquet)
    #[arg(short, long, default_value = "drifters.stt")]
    pub output: PathBuf,

    /// Inclusive start date (YYYY-MM-DD). Must be <= 2022-11-01.
    #[arg(long, default_value = "2021-01-01")]
    pub start: String,

    /// Exclusive end date (YYYY-MM-DD).
    #[arg(long, default_value = "2022-01-01")]
    pub end: String,

    /// Optional geographic bounds: min_lat,min_lon,max_lat,max_lon (default: global).
    #[arg(long)]
    pub bounds: Option<String>,

    /// Minimum fixes per track segment.
    #[arg(long, default_value = "4")]
    pub min_points: usize,

    /// Maximum gap (hours) between fixes before starting a new segment.
    #[arg(long, default_value = "120")]
    pub max_gap_hours: i64,

    /// Max tile zoom. The globe demo loads z0 tiles (zoomOverride 0), so high
    /// zooms are storage-only — keep this low (4) for multi-decade pulls.
    #[arg(long, default_value = "6")]
    pub max_zoom: u8,

    /// Temporal bucket size (e.g. "1d", "7d"). Coarsen it for long spans so the
    /// z0×time-bucket tile count (and per-frame load) stays manageable.
    #[arg(long, default_value = "1d")]
    pub temporal_bucket: String,

    /// Cache directory for the downloaded monthly CSVs.
    #[arg(long, default_value = "data/gdp-cache")]
    pub cache_dir: PathBuf,

    /// Re-download even if a cached monthly CSV exists.
    #[arg(long)]
    pub refresh: bool,

    /// Skip the stt-build step (emit only the intermediate Parquet).
    #[arg(long)]
    pub skip_build: bool,
}

/// One fix.
#[derive(Debug, Clone)]
struct Fix {
    ms: i64,
    lon: f64,
    lat: f64,
    temp: Option<f64>,
}

pub fn run(args: Args) -> Result<()> {
    run_product(args, &SIX_HOURLY)
}

/// The shared GDP pipeline: download monthly CSVs, group fixes into
/// per-drifter tracks, split at gaps/antimeridian, and emit LineStrings.
pub(crate) fn run_product(args: Args, product: &Product) -> Result<()> {
    println!("{}", product.banner);

    let start = NaiveDate::parse_from_str(&args.start, "%Y-%m-%d")
        .with_context(|| format!("Invalid --start date: {}", args.start))?;
    let end = NaiveDate::parse_from_str(&args.end, "%Y-%m-%d")
        .with_context(|| format!("Invalid --end date: {}", args.end))?;
    anyhow::ensure!(start < end, "--start must be before --end");

    let bounds = args
        .bounds
        .as_deref()
        .map(common::parse_bounds)
        .transpose()?;

    fs::create_dir_all(&args.cache_dir)?;

    // Download month-by-month (one ERDDAP request per month is far more
    // reliable than a single year-long global scan, which tends to time out).
    let months = month_starts(start, end);
    println!(
        "📥 Fetching {} month(s) of {} drifter fixes...\n",
        months.len(),
        product.fetch_noun
    );
    let pb = ProgressBar::new(months.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} months ({eta})")?
            .progress_chars("#>-"),
    );

    // Bounds tag for cache keys that encode it (hourly): without it a bounded
    // run (e.g. a smoke test) and a later global run collide on the same
    // monthly file and the global run silently reuses the bounded subset.
    let bounds_tag = match bounds {
        None => "global".to_string(),
        Some((a, b, c, d)) => format!("{:.2}_{:.2}_{:.2}_{:.2}", a, b, c, d),
    };

    let mut csv_files: Vec<PathBuf> = Vec::new();
    for (y, m) in &months {
        let m_start = NaiveDate::from_ymd_opt(*y, *m, 1).unwrap();
        let (ny, nm) = next_month(*y, *m);
        let m_end = NaiveDate::from_ymd_opt(ny, nm, 1).unwrap().min(end);
        let path = args
            .cache_dir
            .join((product.cache_file)(*y, *m, &bounds_tag));

        if args.refresh
            || !path.exists()
            || fs::metadata(&path).map(|m| m.len() < 64).unwrap_or(true)
        {
            let url = build_url(
                product,
                &format!("{}T00:00:00Z", m_start.format("%Y-%m-%d")),
                &format!("{}T00:00:00Z", m_end.format("%Y-%m-%d")),
                bounds,
            );
            if let Err(e) = download(&url, &path) {
                eprintln!("⚠️  month {:04}-{:02} download failed: {e}", y, m);
                pb.inc(1);
                continue;
            }
        }
        csv_files.push(path);
        pb.inc(1);
    }
    pb.finish_with_message("download complete");

    anyhow::ensure!(!csv_files.is_empty(), "No drifter CSVs were downloaded.");

    // Parse + group all fixes by drifter ID.
    println!("\n🔄 Parsing fixes and grouping by drifter...");
    let mut tracks: HashMap<String, Vec<Fix>> = HashMap::new();
    let mut wmo_of: HashMap<String, String> = HashMap::new();
    let mut total = 0usize;
    for path in &csv_files {
        parse_csv(path, product, &mut tracks, &mut wmo_of, &mut total)?;
    }
    println!("   {} fixes across {} drifters", total, tracks.len());

    // Intermediate Parquet path.
    let build_stt =
        args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build;
    let intermediate = if build_stt {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    println!("\n💾 Building track segments...");
    let cols: Vec<PropertyColumn> = vec![
        PropertyColumn::string("drifter_id"),
        PropertyColumn::string("wmo"),
        PropertyColumn::string("temp_band"),
        PropertyColumn::numeric("sst"),
        PropertyColumn::numeric("segment"),
    ];
    let mut writer = StreamingLineStringParquetWriter::with_columns(&intermediate, cols)?;

    let max_gap_ms = args.max_gap_hours * 3600 * 1000;
    let mut segment_count = 0usize;
    let mut drifter_ids: Vec<&String> = tracks.keys().collect();
    drifter_ids.sort(); // deterministic output order

    for id in drifter_ids {
        let fixes = &tracks[id];
        if fixes.len() < args.min_points {
            continue;
        }
        let mut sorted = fixes.clone();
        sorted.sort_by_key(|f| f.ms);
        let segments = split_track(&sorted, max_gap_ms, args.min_points);
        let wmo = wmo_of.get(id).cloned().unwrap_or_default();

        for (seg_idx, seg) in segments.iter().enumerate() {
            let coordinates: Vec<[f64; 2]> = seg.iter().map(|f| [f.lon, f.lat]).collect();
            let vertex_ms: Vec<i64> = seg.iter().map(|f| f.ms).collect();

            let temps: Vec<f64> = seg.iter().filter_map(|f| f.temp).collect();
            let mean_temp = if temps.is_empty() {
                None
            } else {
                Some(temps.iter().sum::<f64>() / temps.len() as f64)
            };

            // Per-vertex SST drives the along-track color gradient. Gaps in
            // the record are gap-filled (forward then backward) so every
            // vertex has a value; a segment with no temps at all stays NaN
            // (rendered gray). Aligned 1:1 with `coordinates`.
            let vertex_temps = fill_vertex_temps(seg);

            let mut props = Map::new();
            props.insert("drifter_id".into(), json!(id));
            props.insert("wmo".into(), json!(wmo));
            props.insert("temp_band".into(), json!(temp_band(mean_temp)));
            // Store sst as a real number (rounded to 0.1 °C) so it survives
            // columnar inference as Numeric and can drive a temperature ramp.
            // Absent when unknown, so the column reads null rather than 0.
            if let Some(t) = mean_temp {
                props.insert("sst".into(), json!((t * 10.0).round() / 10.0));
            }
            props.insert("segment".into(), json!(seg_idx));

            let start_t = ms_to_dt(vertex_ms[0]);
            let end_t = ms_to_dt(*vertex_ms.last().unwrap());
            let mut rec = LineStringRecord::new(coordinates, start_t, Some(end_t), props);
            rec.vertex_timestamps_ms = Some(vertex_ms);
            rec.vertex_values = Some(vertex_temps);
            writer.write_linestring(&rec)?;
            segment_count += 1;
        }
    }

    writer.finish()?;
    println!("   {} track segments written", segment_count);

    if build_stt {
        common::run_stt_build_with_full_options(SttBuildOptions {
            input: intermediate.clone(),
            output: args.output.clone(),
            time_field: "timestamp".into(),
            end_time_field: Some("end_timestamp".into()),
            min_zoom: 0,
            max_zoom: args.max_zoom, // globe loads z0 (zoomOverride 0); keep low
            compression: "zstd".into(),
            temporal_bucket: Some(args.temporal_bucket.clone()),
            temporal_lod: None,
            summary: None,
            summary_sub_buckets: None,
            min_features_per_tile: None,
            min_zoom_field: None,
            max_zoom_field: None,
            no_clip: false,
            quantize_coords: None,
            quantize_attrs: Vec::new(),
            blob_ordering: None,
        })?;
        let _ = fs::remove_file(&intermediate);
    }

    println!("{}", product.done_message);
    println!("   Output: {}", args.output.display());
    Ok(())
}

/// ERDDAP's Tomcat front-end rejects raw `<>"(),:` and spaces in the query;
/// percent-encode exactly those (leaving `=`, `&`, `?` literal as separators).
fn erddap_encode(s: &str) -> String {
    s.replace('%', "%25")
        .replace(',', "%2C")
        .replace('>', "%3E")
        .replace('<', "%3C")
        .replace('"', "%22")
        .replace('(', "%28")
        .replace(')', "%29")
        .replace(':', "%3A")
        .replace(' ', "%20")
}

fn build_url(
    product: &Product,
    start: &str,
    end: &str,
    bounds: Option<(f64, f64, f64, f64)>,
) -> String {
    let columns = format!(
        "ID,WMO,time,longitude,latitude,{},ve,vn",
        product.temp_column
    );
    let mut parts: Vec<String> = vec![
        erddap_encode(&columns),
        format!("time%3E={}", erddap_encode(start)),
        format!("time%3C={}", erddap_encode(end)),
    ];
    if let Some((min_lat, min_lon, max_lat, max_lon)) = bounds {
        parts.push(format!("latitude%3E={}", min_lat));
        parts.push(format!("latitude%3C={}", max_lat));
        parts.push(format!("longitude%3E={}", min_lon));
        parts.push(format!("longitude%3C={}", max_lon));
    }
    parts.push(erddap_encode(r#"orderBy("ID,time")"#));
    format!("{}?{}", product.erddap_base, parts.join("&"))
}

fn download(url: &str, out: &PathBuf) -> Result<()> {
    let status = Command::new("curl")
        .args([
            "-sS",
            "-f",
            "-L",
            "--connect-timeout",
            "30",
            "--max-time",
            "600",
            "-o",
        ])
        .arg(out)
        .arg(url)
        .status()
        .context("failed to run curl")?;
    if !status.success() {
        let _ = fs::remove_file(out);
        anyhow::bail!("curl failed for {url}");
    }
    Ok(())
}

fn parse_csv(
    path: &PathBuf,
    product: &Product,
    tracks: &mut HashMap<String, Vec<Fix>>,
    wmo_of: &mut HashMap<String, String>,
    total: &mut usize,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    // Resolve column indices by header name (robust to column reordering).
    let headers = rdr.headers()?.clone();
    let idx = |name: &str| headers.iter().position(|h| h == name);
    let (ci_id, ci_wmo, ci_time, ci_lon, ci_lat, ci_temp) = (
        idx("ID").context("missing ID column")?,
        idx("WMO"),
        idx("time").context("missing time column")?,
        idx("longitude").context("missing longitude column")?,
        idx("latitude").context("missing latitude column")?,
        idx(product.temp_column),
    );

    for rec in rdr.records() {
        let rec = match rec {
            Ok(r) => r,
            Err(_) => continue,
        };
        // The ERDDAP CSV has a units row directly under the header
        // (e.g. `,,UTC,degrees_east,...`); it fails the timestamp parse below
        // and is skipped along with any other malformed row.
        let ms = match rec.get(ci_time).and_then(parse_time_ms) {
            Some(ms) => ms,
            None => continue,
        };
        let (lon, lat) = match (
            rec.get(ci_lon).and_then(parse_f64),
            rec.get(ci_lat).and_then(parse_f64),
        ) {
            (Some(lon), Some(lat)) if lon.abs() <= 180.0 && lat.abs() <= 90.0 => (lon, lat),
            _ => continue,
        };
        let id = match rec.get(ci_id) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => continue,
        };
        let temp = ci_temp
            .and_then(|i| rec.get(i))
            .and_then(parse_f64)
            .and_then(|v| (product.temp_to_celsius)(v));

        if let Some(i) = ci_wmo {
            if let Some(w) = rec.get(i) {
                if !w.is_empty() {
                    wmo_of.entry(id.clone()).or_insert_with(|| w.to_string());
                }
            }
        }

        tracks
            .entry(id)
            .or_default()
            .push(Fix { ms, lon, lat, temp });
        *total += 1;
    }
    Ok(())
}

fn parse_f64(s: &str) -> Option<f64> {
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Some(v),
        _ => None,
    }
}

fn parse_time_ms(s: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(s.trim())
        .ok()
        .map(|dt| dt.with_timezone(&Utc).timestamp_millis())
}

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(Utc::now)
}

/// Per-vertex SST (°C) aligned 1:1 with a segment's fixes, gap-filled so the
/// along-track color gradient has a value at every vertex. Missing fixes take
/// the nearest known temperature (forward fill, then backward fill for any
/// leading gap); a segment with no temperatures at all stays all-`NaN`.
fn fill_vertex_temps(seg: &[Fix]) -> Vec<f32> {
    let mut out: Vec<f32> = seg
        .iter()
        .map(|f| f.temp.map(|t| t as f32).unwrap_or(f32::NAN))
        .collect();
    // Forward fill: carry the last known value into following gaps.
    let mut last = f32::NAN;
    for v in out.iter_mut() {
        if v.is_nan() {
            if !last.is_nan() {
                *v = last;
            }
        } else {
            last = *v;
        }
    }
    // Backward fill: handle any leading gap before the first known value.
    let mut next = f32::NAN;
    for v in out.iter_mut().rev() {
        if v.is_nan() {
            if !next.is_nan() {
                *v = next;
            }
        } else {
            next = *v;
        }
    }
    out
}

/// SST → categorical band driving the trip colour ramp.
fn temp_band(mean: Option<f64>) -> &'static str {
    match mean {
        Some(t) if t < 5.0 => "cold",
        Some(t) if t < 15.0 => "cool",
        Some(t) if t < 22.0 => "mild",
        Some(t) if t < 27.0 => "warm",
        Some(_) => "hot",
        None => "unknown",
    }
}

/// Split a time-sorted track at large gaps and antimeridian crossings.
fn split_track(fixes: &[Fix], max_gap_ms: i64, min_points: usize) -> Vec<Vec<Fix>> {
    let mut out: Vec<Vec<Fix>> = Vec::new();
    let mut cur: Vec<Fix> = Vec::new();
    for f in fixes {
        if let Some(last) = cur.last() {
            let gap = f.ms - last.ms > max_gap_ms;
            // Antimeridian crossing: |Δlon| > 180° straddles the dateline. The
            // old `>170 && <-170` test missed crossings whose endpoints sit
            // more than 10° off ±180° (e.g. -163°→165°), which then rendered as
            // a line sweeping the long way across the whole map.
            let anti = (last.lon - f.lon).abs() > 180.0;
            if gap || anti {
                if cur.len() >= min_points {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.clear();
                }
            }
        }
        cur.push(f.clone());
    }
    if cur.len() >= min_points {
        out.push(cur);
    }
    out
}

fn next_month(y: i32, m: u32) -> (i32, u32) {
    if m == 12 {
        (y + 1, 1)
    } else {
        (y, m + 1)
    }
}

/// All (year, month) starts intersecting [start, end).
fn month_starts(start: NaiveDate, end: NaiveDate) -> Vec<(i32, u32)> {
    let mut out = Vec::new();
    let (mut y, mut m) = (start.year(), start.month());
    loop {
        let first = NaiveDate::from_ymd_opt(y, m, 1).unwrap();
        if first >= end {
            break;
        }
        out.push((y, m));
        let (ny, nm) = next_month(y, m);
        y = ny;
        m = nm;
    }
    out
}
