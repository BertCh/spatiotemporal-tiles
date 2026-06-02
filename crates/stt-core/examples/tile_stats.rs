//! Over-time payload-sizing diagnostics for a real `.stt` archive.
//!
//! `simulate_layout` answers "which blob ordering minimizes range requests".
//! This tool answers the orthogonal question the over-time-loading analysis
//! needs: **how big is one playback step's payload, and where does the time
//! axis spend its bytes?** It reports, per zoom:
//!   - tile compressed-size distribution (the per-range-request payload primitive)
//!   - temporal fan-out: how many time-buckets exist per spatial cell
//!   - bytes per time-slice (sum of tile bytes sharing a bucket boundary)
//! and, given `--window <ms>` + `--zoom <z>`, simulates ONE playback step:
//! the bytes/tiles a viewport×window working set pulls, plus the temporal
//! over-fetch ratio (loaded bucket-span ÷ displayed window).
//!
//! Reads ONLY the header+directory+metadata (no blob decompression), so it is
//! cheap even on multi-GB archives.
//!
//! Run:
//!   cargo run --release -p stt-core --example tile_stats -- \
//!     examples/showcase/public/data/flights.stt --window 600000 --zoom 4 --cells 2

use std::collections::HashMap;
use std::error::Error;

use stt_core::archive::TileEntry;
use stt_core::metadata::Metadata;
use stt_core::ArchiveReader;

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn hsize(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.2}GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.2}MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1}KB", b / 1e3)
    } else {
        format!("{}B", bytes)
    }
}

fn hms(ms: i64) -> String {
    let s = ms as f64 / 1000.0;
    if s >= 86400.0 {
        format!("{:.1}d", s / 86400.0)
    } else if s >= 3600.0 {
        format!("{:.1}h", s / 3600.0)
    } else if s >= 60.0 {
        format!("{:.1}m", s / 60.0)
    } else {
        format!("{:.0}s", s)
    }
}

struct Arg {
    window_ms: Option<i64>,
    zoom: Option<u8>,
    cells: u32,
}

fn parse_args(args: &[String]) -> (String, Arg) {
    let mut path = String::new();
    let mut a = Arg {
        window_ms: None,
        zoom: None,
        cells: 2,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--window" => {
                a.window_ms = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--zoom" => {
                a.zoom = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--cells" => {
                a.cells = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2);
                i += 2;
            }
            other => {
                if path.is_empty() {
                    path = other.to_string();
                }
                i += 1;
            }
        }
    }
    (path, a)
}

fn main() -> Result<(), Box<dyn Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (path, arg) = parse_args(&argv);
    if path.is_empty() {
        return Err("usage: tile_stats <archive.stt> [--window <ms>] [--zoom <z>] [--cells <n>]".into());
    }

    let reader = ArchiveReader::open(&path)?;
    let meta: &Metadata = reader.metadata();
    let entries = reader.entries();
    let base_bucket = meta.temporal_bucket_ms.max(1) as i64;

    println!("archive: {}", path);
    println!(
        "  name={:?} zoom=[{}..{}] base_bucket_ms={} ({})",
        meta.name, meta.min_zoom, meta.max_zoom, meta.temporal_bucket_ms, hms(base_bucket)
    );
    let (t0, t1) = (meta.time_range.start as i64, meta.time_range.end as i64);
    println!(
        "  time=[{}..{}] span={} ({} base buckets over span)",
        t0,
        t1,
        hms(t1 - t0),
        ((t1 - t0) / base_bucket).max(0) + 1
    );
    println!(
        "  meta.tile_count={} feature_count={} layers={:?}",
        meta.tile_count, meta.feature_count, meta.layers
    );
    match &meta.temporal_lod {
        Some(levels) => {
            println!("  temporal_lod: {} level(s):", levels.len());
            for l in levels {
                println!(
                    "      bucket_ms={} ({}) applies at zoom<= {}",
                    l.bucket_ms,
                    hms(l.bucket_ms as i64),
                    l.max_zoom_level
                );
            }
        }
        None => println!("  temporal_lod: NONE"),
    }
    println!(
        "  summary_tier={} raster_tier={}",
        meta.summary_tier.is_some(),
        meta.raster_tier.is_some()
    );

    // ---- global blob accounting ----
    let mut distinct_off: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut distinct_content: std::collections::HashSet<(u32, u32, u32)> =
        std::collections::HashSet::new();
    let mut phys_bytes: u64 = 0;
    let mut logical_comp_bytes: u64 = 0;
    let mut logical_unc_bytes: u64 = 0;
    for e in entries {
        logical_comp_bytes += e.length as u64;
        logical_unc_bytes += e.uncompressed_size as u64;
        if distinct_off.insert(e.offset) {
            phys_bytes += e.length as u64;
        }
        distinct_content.insert((e.crc32c, e.length, e.uncompressed_size));
    }
    println!(
        "  entries={} distinct_physical_blobs={} content_groups={} (dedup ceiling {:.1}%)",
        entries.len(),
        distinct_off.len(),
        distinct_content.len(),
        100.0 * (entries.len().saturating_sub(distinct_content.len())) as f64
            / entries.len().max(1) as f64
    );
    println!(
        "  bytes: physical={} logical_compressed={} logical_uncompressed={} (overall ratio {:.2}x)",
        hsize(phys_bytes),
        hsize(logical_comp_bytes),
        hsize(logical_unc_bytes),
        logical_unc_bytes as f64 / logical_comp_bytes.max(1) as f64
    );

    // ---- per-zoom temporal payload table ----
    let zooms: Vec<u8> = {
        let mut z: Vec<u8> = entries.iter().map(|e| e.zoom).collect();
        z.sort_unstable();
        z.dedup();
        z
    };

    println!("\n[per-zoom temporal payload]  (tile size = one range-request payload primitive)");
    println!(
        "{:<4} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9}  {:>8} {:>8} {:>8}  {:>9} {:>9} {:>9}",
        "z", "tiles", "cells", "buckets",
        "tile_p50", "tile_p90", "tile_p99", "tile_max",
        "bpc_p50", "bpc_p90", "bpc_max",
        "slice_p50", "slice_p90", "slice_max"
    );
    for &z in &zooms {
        let ze: Vec<&TileEntry> = entries.iter().filter(|e| e.zoom == z).collect();
        let mut sizes: Vec<u64> = ze.iter().map(|e| e.length as u64).collect();
        sizes.sort_unstable();

        // distinct cells, distinct buckets
        let mut cells: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        let mut buckets: std::collections::HashSet<i64> = std::collections::HashSet::new();
        // buckets-per-cell (temporal fan-out) and bytes-per-bucket (time-slice weight)
        let mut per_cell: HashMap<(u32, u32), u64> = HashMap::new();
        let mut per_bucket_bytes: HashMap<i64, u64> = HashMap::new();
        for e in &ze {
            cells.insert((e.x, e.y));
            buckets.insert(e.time_start);
            *per_cell.entry((e.x, e.y)).or_default() += 1;
            *per_bucket_bytes.entry(e.time_start).or_default() += e.length as u64;
        }
        let mut bpc: Vec<u64> = per_cell.values().copied().collect();
        bpc.sort_unstable();
        let mut slice: Vec<u64> = per_bucket_bytes.values().copied().collect();
        slice.sort_unstable();

        println!(
            "{:<4} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9}  {:>8} {:>8} {:>8}  {:>9} {:>9} {:>9}",
            z,
            ze.len(),
            cells.len(),
            buckets.len(),
            hsize(pct(&sizes, 0.50)),
            hsize(pct(&sizes, 0.90)),
            hsize(pct(&sizes, 0.99)),
            hsize(pct(&sizes, 1.0)),
            pct(&bpc, 0.50),
            pct(&bpc, 0.90),
            pct(&bpc, 1.0),
            hsize(pct(&slice, 0.50)),
            hsize(pct(&slice, 0.90)),
            hsize(pct(&slice, 1.0)),
        );
    }
    println!("  legend: bpc=buckets-per-cell (temporal fan-out); slice=bytes of ALL tiles sharing one bucket boundary at this zoom");

    // ---- single playback-step simulation ----
    if let (Some(window_ms), zoom_opt) = (arg.window_ms, arg.zoom) {
        // Pick display zoom: explicit, else densest zoom.
        let z = zoom_opt.unwrap_or_else(|| {
            let mut per_zoom: HashMap<u8, u64> = HashMap::new();
            for e in entries {
                *per_zoom.entry(e.zoom).or_default() += 1;
            }
            per_zoom.into_iter().max_by_key(|&(_, n)| n).map(|(z, _)| z).unwrap_or(meta.max_zoom)
        });

        // Densest cell at z, and a populated instant there.
        let ze: Vec<&TileEntry> = entries.iter().filter(|e| e.zoom == z).collect();
        let mut cell_count: HashMap<(u32, u32), u64> = HashMap::new();
        for e in &ze {
            *cell_count.entry((e.x, e.y)).or_default() += 1;
        }
        let Some((&(cx, cy), _)) = cell_count.iter().max_by_key(|&(k, v)| (*v, k.0, k.1)) else {
            println!("\n[step-sim] zoom {} has no tiles", z);
            return Ok(());
        };
        // Median populated instant in that cell.
        let mut times: Vec<i64> = ze
            .iter()
            .filter(|e| e.x == cx && e.y == cy)
            .map(|e| e.time_start)
            .collect();
        times.sort_unstable();
        let t_anchor = times[times.len() / 2];

        let r = arg.cells;
        let (xmin, xmax) = (cx.saturating_sub(r), cx + r);
        let (ymin, ymax) = (cy.saturating_sub(r), cy + r);
        let wstart = t_anchor - window_ms / 2;
        let wend = t_anchor + window_ms / 2;

        // Working set: tiles at z in the viewport whose interval overlaps the window.
        let mut step_bytes: u64 = 0;
        let mut step_unc: u64 = 0;
        let mut step_tiles: u64 = 0;
        let mut step_feats: u64 = 0;
        let mut loaded_bucket_span: i64 = 0;
        for e in &ze {
            if e.x < xmin || e.x > xmax || e.y < ymin || e.y > ymax {
                continue;
            }
            let cmin = e.cover_t_min.unwrap_or(e.time_start);
            if e.time_end < wstart || cmin > wend {
                continue;
            }
            step_bytes += e.length as u64;
            step_unc += e.uncompressed_size as u64;
            step_feats += e.feature_count as u64;
            step_tiles += 1;
            let bk = e.temporal_bucket_ms.map(|v| v as i64).unwrap_or(base_bucket);
            loaded_bucket_span = loaded_bucket_span.max(bk);
        }

        // How many spatial cells in the viewport actually have data (for context).
        let vp_cells = (xmax - xmin + 1) as u64 * (ymax - ymin + 1) as u64;
        let buckets_per_window = (window_ms as f64 / base_bucket as f64).max(0.0);
        // Temporal over-fetch: when window < bucket you load a whole bucket to show
        // a fraction of it; when window >= bucket you load ceil(window/bucket) buckets.
        let temporal_overfetch = if (window_ms as f64) < base_bucket as f64 {
            base_bucket as f64 / window_ms.max(1) as f64
        } else {
            1.0
        };

        println!("\n[step-sim] ONE playback step (load-window = render-? wide)");
        println!(
            "  display zoom={} densest cell=({},{}) viewport=±{} tiles ({} cells) t_anchor={}",
            z, cx, cy, r, vp_cells, t_anchor
        );
        println!(
            "  window={} ({}) -> [{}..{}]   base_bucket={} ({})",
            window_ms,
            hms(window_ms),
            wstart,
            wend,
            base_bucket,
            hms(base_bucket)
        );
        println!(
            "  buckets_per_window={:.2}   temporal_overfetch={:.1}x  (window {} bucket)",
            buckets_per_window,
            temporal_overfetch,
            if (window_ms as f64) < base_bucket as f64 { "<" } else { ">=" }
        );
        println!(
            "  STEP PAYLOAD: tiles={} compressed={} uncompressed={} features={}",
            step_tiles,
            hsize(step_bytes),
            hsize(step_unc),
            step_feats
        );
        if step_tiles > 0 {
            println!(
                "  mean tile in step: {} compressed, {} features; loaded_bucket_span={} ({})",
                hsize(step_bytes / step_tiles),
                step_feats / step_tiles,
                loaded_bucket_span,
                hms(loaded_bucket_span)
            );
        }
    }

    Ok(())
}
