//! Derive a NYC taxi POINT dataset from an existing LineString .stt archive.
//!
//! Reads the already-built `nyc-taxi-paths.stt` (500K OSRM-routed trips),
//! interpolates each trip's polyline at a fixed time interval, and emits a
//! Point dataset suitable for animated rendering. This avoids re-running
//! the OSRM pipeline — the route geometries already baked into the .stt are
//! reused as-is.

use crate::common::{
    self, PointRecord, PropertyColumn, PropertyKind, StreamingParquetWriter,
};
use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, FixedSizeListArray, Float64Array, Int64Array, ListArray, UInt16Array, UInt64Array};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Map};
use std::collections::HashSet;
use std::path::PathBuf;
use stt_core::ArchiveReader;

#[derive(Parser, Debug)]
#[command(about = "Derive a NYC taxi point dataset by interpolating an existing path .stt")]
pub struct Args {
    /// Input LineString .stt archive (the existing taxi-paths build).
    #[arg(long, default_value = "examples/showcase/public/data/nyc-taxi-paths.stt")]
    pub input: PathBuf,

    /// Output .stt (or .parquet for intermediate only).
    #[arg(short, long, default_value = "nyc-taxi-points.stt")]
    pub output: PathBuf,

    /// Time spacing (seconds) between interpolated points. Smaller = smoother
    /// animation, larger output. 15s is a good middle ground given the typical
    /// ~25-min trip duration (~100 points/trip).
    #[arg(long, default_value = "15")]
    pub interval_seconds: u32,

    /// Cap on trips processed (for quick iteration). None = all.
    #[arg(long)]
    pub max_trips: Option<usize>,

    /// Skip the stt-build step (emit intermediate Parquet only).
    #[arg(long)]
    pub skip_build: bool,

    /// Keep intermediate parquet after build.
    #[arg(long)]
    pub keep_intermediate: bool,

    /// Temporal bucket for the output archive's tile chunking.
    #[arg(long, default_value = "30m")]
    pub temporal_bucket: String,

    /// Min zoom for the output archive.
    #[arg(long, default_value = "10")]
    pub min_zoom: u8,

    /// Max zoom for the output archive.
    #[arg(long, default_value = "16")]
    pub max_zoom: u8,
}

pub fn run(args: Args) -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║      🚕 NYC Taxi Points (derived from existing .stt paths)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if !args.input.exists() {
        return Err(anyhow!(
            "Input .stt not found: {}",
            args.input.display()
        ));
    }

    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    println!("📋 Configuration:");
    println!("   Input:         {}", args.input.display());
    println!("   Output:        {}", args.output.display());
    println!("   Intermediate:  {}", intermediate_path.display());
    println!("   Interval:      {}s", args.interval_seconds);
    if let Some(max) = args.max_trips {
        println!("   Max trips:     {}", max);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 STEP 1: Read .stt and interpolate points");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let reader = ArchiveReader::open(&args.input)
        .with_context(|| format!("opening archive {}", args.input.display()))?;
    let entries = reader.entries();
    if entries.is_empty() {
        return Err(anyhow!("Archive contains no tiles"));
    }

    // Process the MIN-zoom tiles only. At zoom 10 a Manhattan-sized region
    // fits in a single tile, so every trip's LineString is present once and
    // unclipped — no need to walk every zoom level and dedupe across them.
    // Geometries are not simplified at any zoom (stt-build default), so this
    // gives full OSRM fidelity.
    let target_zoom = entries.iter().map(|e| e.zoom).min().unwrap();
    let target_tiles: Vec<_> = entries.iter().filter(|e| e.zoom == target_zoom).collect();

    println!("   Total tiles:         {}", entries.len());
    println!("   Processing zoom:     {} ({} tiles)", target_zoom, target_tiles.len());

    let property_columns = vec![
        PropertyColumn { name: "trip_id".into(), kind: PropertyKind::Numeric },
        PropertyColumn { name: "status".into(), kind: PropertyKind::String },
    ];
    let mut writer = StreamingParquetWriter::with_columns(&intermediate_path, property_columns)?;

    let mut seen: HashSet<u64> = HashSet::with_capacity(600_000);
    let interval_ms: i64 = args.interval_seconds as i64 * 1_000;
    let mut total_points: usize = 0;
    let mut total_trips: usize = 0;
    let mut skipped_short: usize = 0;

    let pb = ProgressBar::new(target_tiles.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("   [{bar:40}] {pos}/{len} tiles ({eta}) {msg}")?
            .progress_chars("=>-"),
    );

    'tiles: for entry in &target_tiles {
        let layers = reader
            .read_layers(entry)
            .with_context(|| format!("reading tile {:?}", entry.tile_id()))?;

        for layer in layers {
            let batch = &layer.batch;

            // Only LineString layers have a List<FixedSizeList<f64,2>> geometry.
            // Point layers (e.g. summary tier) downcast to FixedSizeListArray
            // directly and are silently skipped.
            let geom_col = match batch.column_by_name("geometry") {
                Some(c) => c,
                None => continue,
            };
            let geom = match geom_col.as_any().downcast_ref::<ListArray>() {
                Some(g) => g,
                None => continue,
            };
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| anyhow!("layer '{}' missing UInt64 'id'", layer.name))?;
            let starts = batch
                .column_by_name("start_time")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| anyhow!("layer '{}' missing Int64 'start_time'", layer.name))?;
            let ends = batch
                .column_by_name("end_time")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| anyhow!("layer '{}' missing Int64 'end_time'", layer.name))?;

            // Per-vertex times. v3 layers carry u16-delta encoding with
            // (origin_ms, step_ms) in the schema metadata; v2 layers (and
            // wide-span fallbacks) carry absolute Int64 directly. Either
            // way we decode each row into absolute Unix-ms here so the
            // interpolator can place output points on real per-segment
            // speeds (highway runs sweep fast, urban grid creeps slow).
            // Missing = fall back to uniform-by-distance interpolation.
            let vertex_time_col = batch
                .column_by_name("vertex_time")
                .and_then(|c| c.as_any().downcast_ref::<ListArray>());
            let schema_meta = batch.schema().metadata().clone();
            let vt_origin: Option<i64> = schema_meta
                .get("stt:vertex_time_origin_ms")
                .and_then(|s| s.parse().ok());
            let vt_step: Option<u32> = schema_meta
                .get("stt:vertex_time_step_ms")
                .and_then(|s| s.parse().ok());

            for i in 0..batch.num_rows() {
                let id = ids.value(i);
                if !seen.insert(id) {
                    continue;
                }
                if let Some(max) = args.max_trips {
                    if seen.len() > max {
                        break 'tiles;
                    }
                }

                let start_ms = starts.value(i);
                let end_ms = ends.value(i);
                if end_ms <= start_ms {
                    skipped_short += 1;
                    continue;
                }

                let coords = extract_coords(geom, i)?;
                if coords.len() < 2 {
                    skipped_short += 1;
                    continue;
                }

                let vertex_times = vertex_time_col
                    .and_then(|col| extract_vertex_times(col, i, vt_origin, vt_step));

                let points = match vertex_times.as_ref() {
                    Some(vt) if vt.len() == coords.len() => {
                        interpolate_along_path_with_times(&coords, vt, interval_ms)
                    }
                    _ => interpolate_along_path(&coords, start_ms, end_ms, interval_ms),
                };
                if points.is_empty() {
                    skipped_short += 1;
                    continue;
                }

                let last_idx = points.len() - 1;
                for (idx, (lon, lat, ts)) in points.iter().enumerate() {
                    let status = if idx == 0 {
                        "pickup"
                    } else if idx == last_idx {
                        "dropoff"
                    } else {
                        "enroute"
                    };
                    let mut props = Map::new();
                    props.insert("trip_id".into(), json!(id));
                    props.insert("status".into(), json!(status));
                    writer.write_point(&PointRecord {
                        lon: *lon,
                        lat: *lat,
                        timestamp_ms: *ts,
                        properties: props,
                    })?;
                    total_points += 1;
                }
                total_trips += 1;
            }
        }

        pb.inc(1);
        pb.set_message(format!("{} trips, {} points", total_trips, total_points));
    }
    pb.finish_and_clear();

    let rows = writer.finish()?;
    println!(
        "   ✓ Wrote {} points across {} trips ({} short trips skipped)",
        rows, total_trips, skipped_short
    );
    if let Ok(meta) = std::fs::metadata(&intermediate_path) {
        println!(
            "   📁 Intermediate: {} ({:.1} MB)",
            intermediate_path.display(),
            meta.len() as f64 / 1_000_000.0
        );
    }

    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📝 STEP 2: Build STT archive");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        common::run_stt_build_with_options(
            &intermediate_path,
            &args.output,
            "timestamp",
            None,
            args.min_zoom,
            args.max_zoom,
            "zstd",
            Some(&args.temporal_bucket),
        )?;

        if !args.keep_intermediate {
            let _ = std::fs::remove_file(&intermediate_path);
        } else {
            println!("   📁 Keeping intermediate: {}", intermediate_path.display());
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    ✅ Generation Complete                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("   Output: {}", args.output.display());
    if let Ok(meta) = std::fs::metadata(&args.output) {
        println!("   Size:   {:.1} MB", meta.len() as f64 / 1_000_000.0);
    }
    println!();
    Ok(())
}

/// Pull a single feature's coord list out of a `List<FixedSizeList<Float64,2>>`.
fn extract_coords(geom: &ListArray, feature_idx: usize) -> Result<Vec<[f64; 2]>> {
    let pairs_arr = geom.value(feature_idx);
    let pairs = pairs_arr
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow!("LineString inner type is not FixedSizeList<f64,2>"))?;
    let raw = pairs
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow!("LineString coords are not Float64"))?;
    let n = pairs.len();
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let base = j * 2;
        out.push([raw.value(base), raw.value(base + 1)]);
    }
    Ok(out)
}

/// Decode a single feature's per-vertex absolute Unix-ms timestamps from a
/// `vertex_time` `ListArray`. Handles both v3 u16-delta encoding (with
/// `origin_ms` + `step_ms` schema metadata) and the v2 absolute-Int64
/// fallback. Returns `None` if the row is null or the inner type is neither.
fn extract_vertex_times(
    col: &ListArray,
    row: usize,
    origin: Option<i64>,
    step: Option<u32>,
) -> Option<Vec<i64>> {
    if !col.is_valid(row) {
        return None;
    }
    let values = col.value(row);
    if let Some(deltas) = values.as_any().downcast_ref::<UInt16Array>() {
        let origin = origin?;
        let step = step? as i64;
        Some(
            deltas
                .values()
                .iter()
                .map(|d| origin + (*d as i64) * step)
                .collect(),
        )
    } else if let Some(abs) = values.as_any().downcast_ref::<Int64Array>() {
        Some((0..abs.len()).map(|i| abs.value(i)).collect())
    } else {
        None
    }
}

/// Interpolate points along a polyline at a fixed temporal cadence, using
/// supplied per-vertex absolute timestamps as the time→space mapping.
///
/// Unlike [`interpolate_along_path`], time progresses **linearly in real
/// time** (one sample every `interval_ms`); the per-segment speed implied
/// by `vertex_times` determines how much spatial ground each tick covers.
/// A highway segment with a short delta-t between vertices = the head
/// moves a long distance per tick. An urban-block segment with a long
/// delta-t = small distance per tick.
fn interpolate_along_path_with_times(
    coords: &[[f64; 2]],
    vertex_times: &[i64],
    interval_ms: i64,
) -> Vec<(f64, f64, i64)> {
    let n = coords.len();
    if n < 2 || vertex_times.len() != n || interval_ms <= 0 {
        return Vec::new();
    }
    let start_ms = vertex_times[0];
    let end_ms = vertex_times[n - 1];
    if end_ms <= start_ms {
        return Vec::new();
    }
    let total_ms = (end_ms - start_ms) as f64;
    let n_steps = ((total_ms / interval_ms as f64).ceil() as i64).max(1);
    let mut out: Vec<(f64, f64, i64)> = Vec::with_capacity(n_steps as usize + 1);

    // Two-pointer walk: segment cursor advances monotonically as ticks do.
    let mut seg = 0usize;
    for k in 0..=n_steps {
        let dt = ((k * interval_ms) as f64).min(total_ms);
        let ts = start_ms + dt as i64;
        // Advance seg until ts falls in [vertex_times[seg], vertex_times[seg+1]].
        while seg + 1 < n && vertex_times[seg + 1] < ts {
            seg += 1;
        }
        if seg + 1 >= n {
            seg = n - 2;
        }
        let t0 = vertex_times[seg];
        let t1 = vertex_times[seg + 1];
        let span = (t1 - t0) as f64;
        // Zero-duration segments (rare; saturated OSRM legs or duplicate
        // vertices) snap to the segment start to avoid div-by-zero.
        let local = if span > 0.0 { (ts - t0) as f64 / span } else { 0.0 };
        let a = coords[seg];
        let b = coords[seg + 1];
        let lon = a[0] + (b[0] - a[0]) * local;
        let lat = a[1] + (b[1] - a[1]) * local;
        out.push((lon, lat, ts));
    }
    out
}

/// Interpolate points along a polyline at a fixed temporal cadence.
///
/// Distributes time uniformly along cumulative haversine distance (constant
/// average speed), then samples at every `interval_ms` tick plus an exact
/// endpoint. Linear interp between adjacent vertices.
///
/// Kept as a fallback for tiles missing per-vertex times; prefer
/// [`interpolate_along_path_with_times`] when the source carries them.
fn interpolate_along_path(
    coords: &[[f64; 2]],
    start_ms: i64,
    end_ms: i64,
    interval_ms: i64,
) -> Vec<(f64, f64, i64)> {
    if coords.len() < 2 || end_ms <= start_ms || interval_ms <= 0 {
        return Vec::new();
    }
    let total_ms = (end_ms - start_ms) as f64;

    let mut cum: Vec<f64> = Vec::with_capacity(coords.len());
    cum.push(0.0);
    let mut total = 0.0;
    for w in coords.windows(2) {
        // haversine_distance(lat1, lon1, lat2, lon2) — coords are [lon, lat].
        let d = common::haversine_distance(w[0][1], w[0][0], w[1][1], w[1][0]);
        total += d;
        cum.push(total);
    }
    if total <= 0.0 {
        return Vec::new();
    }

    let n_steps = ((total_ms / interval_ms as f64).ceil() as i64).max(1);
    let mut out: Vec<(f64, f64, i64)> = Vec::with_capacity(n_steps as usize + 1);

    for k in 0..=n_steps {
        let dt = ((k * interval_ms) as f64).min(total_ms);
        let frac = dt / total_ms;
        let target_d = frac * total;

        // Find first vertex with cumulative distance >= target_d.
        let upper = cum.iter().position(|&d| d >= target_d).unwrap_or(coords.len() - 1);
        let seg = upper.saturating_sub(1);
        let d0 = cum[seg];
        let d1 = *cum.get(seg + 1).unwrap_or(&total);
        let local = if d1 > d0 { (target_d - d0) / (d1 - d0) } else { 0.0 };

        let a = coords[seg];
        let b = *coords.get(seg + 1).unwrap_or(&a);
        let lon = a[0] + (b[0] - a[0]) * local;
        let lat = a[1] + (b[1] - a[1]) * local;
        let ts = start_ms + dt as i64;

        out.push((lon, lat, ts));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_endpoints_and_midpoint() {
        // 1km east-west segment at the equator, 100s duration.
        let coords = vec![[0.0, 0.0], [0.00898, 0.0]]; // ~1km in lon at equator
        let pts = interpolate_along_path(&coords, 0, 100_000, 10_000);
        // 100s / 10s = 10 intervals + endpoint = 11 samples.
        assert_eq!(pts.len(), 11);
        // First sample is the start vertex at t=0.
        assert!((pts[0].0 - 0.0).abs() < 1e-9);
        assert_eq!(pts[0].2, 0);
        // Last sample is the end vertex at t=100s.
        assert!((pts.last().unwrap().0 - 0.00898).abs() < 1e-6);
        assert_eq!(pts.last().unwrap().2, 100_000);
        // Midpoint at t=50s is half way across.
        let mid = pts.iter().find(|p| p.2 == 50_000).unwrap();
        assert!((mid.0 - 0.00449).abs() < 1e-5);
    }

    #[test]
    fn rejects_degenerate_inputs() {
        // Single vertex
        assert!(interpolate_along_path(&[[0.0, 0.0]], 0, 100, 10).is_empty());
        // Reversed time range
        assert!(interpolate_along_path(&[[0.0, 0.0], [1.0, 0.0]], 100, 0, 10).is_empty());
        // Zero-length polyline
        assert!(interpolate_along_path(&[[1.0, 1.0], [1.0, 1.0]], 0, 100, 10).is_empty());
    }

    #[test]
    fn time_anchored_interp_respects_per_segment_speed() {
        // Two equal-length segments but very different per-segment timings:
        //   leg 0: 100s for the first 1° (slow urban-block)
        //   leg 1:  10s for the second 1° (fast highway)
        // At t=50s the head should sit ~halfway through leg 0; by t=105s it
        // should be well into leg 1 even though only 5s have passed.
        let coords = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let times = vec![0, 100_000, 110_000];
        let pts = interpolate_along_path_with_times(&coords, &times, 5_000);

        let at = |t: i64| -> [f64; 2] {
            let p = pts.iter().find(|p| p.2 == t).unwrap();
            [p.0, p.1]
        };
        // t=50_000 → half-way through leg 0.
        assert!((at(50_000)[0] - 0.5).abs() < 1e-6);
        // t=105_000 → 5_000 / 10_000 = 50% into leg 1, i.e. lon = 1.5.
        assert!((at(105_000)[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn time_anchored_interp_rejects_mismatched_lengths() {
        let coords = vec![[0.0, 0.0], [1.0, 0.0]];
        // Wrong length: silently returns empty so the caller falls back.
        let pts = interpolate_along_path_with_times(&coords, &[0, 1_000, 2_000], 100);
        assert!(pts.is_empty());
    }
}
