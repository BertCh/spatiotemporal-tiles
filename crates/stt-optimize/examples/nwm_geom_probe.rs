//! Probe NWM corridor tiles: per-zoom vertex resolution + per-tile feature
//! overlap. Throwaway diagnostic for the "over-simplified / weirdly
//! overlapping" report.
//!
//! Usage: cargo run -p stt-optimize --example nwm_geom_probe -- <archive-manifest.json>

use anyhow::{anyhow, Result};
use arrow::array::{Array, FixedSizeListArray, Float64Array, Int32Array, ListArray};
use std::collections::BTreeMap;
use stt_optimize::packed::PackedTileset;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: nwm_geom_probe <manifest.json>"))?;
    let ts = PackedTileset::open(&path)?;

    // Per-zoom aggregates.
    #[derive(Default)]
    struct Z {
        tiles: u64,
        features: u64,
        verts: u64,
        max_feat_in_tile: u64,
        // seg length histogram in meters (per-zoom) — median-ish via buckets.
        seg_total_m: f64,
        seg_count: u64,
    }
    let mut per_zoom: BTreeMap<u8, Z> = BTreeMap::new();

    // Track the single densest (most-features) tile for an overlap dump.
    let mut densest: Option<(u8, u32, u32, i64, usize)> = None;
    // Distinct temporal-bucket starts (grid-aligned tile addresses).
    let mut t_starts: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();

    for e in ts.entries() {
        t_starts.insert(e.time_start);
        let layers = ts.read_layers(e)?;
        for dl in &layers {
            let batch = &dl.batch;
            let geom = batch
                .column_by_name("geometry")
                .ok_or_else(|| anyhow!("no geometry column"))?;
            let list = geom
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| anyhow!("geometry not a ListArray"))?;
            let offs = list.value_offsets();
            let nfeat = list.len();
            let z = per_zoom.entry(e.zoom).or_default();
            z.tiles += 1;
            z.features += nfeat as u64;
            let tile_verts: i64 = offs[nfeat] as i64 - offs[0] as i64;
            z.verts += tile_verts as u64;
            z.max_feat_in_tile = z.max_feat_in_tile.max(nfeat as u64);

            // Coordinates for segment-length stats (sample first few tiles/zoom).
            if z.seg_count < 20_000 {
                let coords = list.values();
                let fsl = coords
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| anyhow!("coords not FixedSizeList"))?;
                let vals = fsl.values();
                // Quantized (Int32) or Float64.
                let get = read_xy(vals.as_ref());
                for f in 0..nfeat {
                    let a = offs[f] as usize;
                    let b = offs[f + 1] as usize;
                    for v in a..b.saturating_sub(1) {
                        let (x0, y0) = get(v);
                        let (x1, y1) = get(v + 1);
                        // crude planar meters at ~38N (quantized grid units are
                        // ~10m; float degrees → deg*~90km). We only need relative.
                        let dx = x1 - x0;
                        let dy = y1 - y0;
                        z.seg_total_m += (dx * dx + dy * dy).sqrt();
                        z.seg_count += 1;
                    }
                }
            }

            if densest.map(|(.., n)| nfeat > n).unwrap_or(true) {
                densest = Some((e.zoom, e.x, e.y, e.time_start, nfeat));
            }
        }
    }

    let chunk_ms = 30i64 * 86_400_000;
    println!("Temporal chunks: {} distinct tile buckets", t_starts.len());
    let tv: Vec<i64> = t_starts.iter().copied().collect();
    for (i, w) in tv.windows(2).enumerate() {
        let gap = w[1] - w[0];
        if gap != chunk_ms {
            println!("  ⚠ bucket {} gap {} ms (≠ {} chunk_ms)", i, gap, chunk_ms);
        }
    }
    if let (Some(&f), Some(&l)) = (tv.first(), tv.last()) {
        println!(
            "  first={f} last={l} (each grid-aligned: {})",
            tv.iter().all(|t| t % chunk_ms == 0)
        );
    }
    println!("Per-zoom resolution:");
    println!("  zoom | tiles | features |    verts | verts/feat | maxFeat/tile | avg seg(units)");
    for (z, s) in &per_zoom {
        let vpf = s.verts as f64 / s.features.max(1) as f64;
        let avgseg = if s.seg_count > 0 {
            s.seg_total_m / s.seg_count as f64
        } else {
            0.0
        };
        println!(
            "  {:>4} | {:>5} | {:>8} | {:>8} | {:>10.1} | {:>12} | {:>10.2}",
            z, s.tiles, s.features, s.verts, vpf, s.max_feat_in_tile, avgseg
        );
    }

    // Overlap dump: in the densest tile, count features whose bounding boxes
    // heavily overlap (a proxy for the "woven" braided/duplicate corridors).
    if let Some((z, x, y, t, _)) = densest {
        println!("\nDensest tile z{z}/{x}/{y}/t{t}: per-feature vertex counts + bbox overlap");
        let entry = ts
            .entries()
            .iter()
            .find(|e| e.zoom == z && e.x == x && e.y == y && e.time_start == t)
            .unwrap();
        let layers = ts.read_layers(entry)?;
        let batch = &layers[0].batch;
        let list = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let offs = list.value_offsets();
        let fsl = list
            .values()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let get = read_xy(fsl.values().as_ref());
        let nfeat = list.len();

        // width column (per-feature) if present.
        let width = batch
            .column_by_name("width")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>().map(|a| a.clone()));

        let mut bboxes = Vec::with_capacity(nfeat);
        let mut vcounts = Vec::with_capacity(nfeat);
        for f in 0..nfeat {
            let a = offs[f] as usize;
            let b = offs[f + 1] as usize;
            vcounts.push(b - a);
            let (mut minx, mut miny, mut maxx, mut maxy) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for v in a..b {
                let (x, y) = get(v);
                minx = minx.min(x);
                miny = miny.min(y);
                maxx = maxx.max(x);
                maxy = maxy.max(y);
            }
            bboxes.push((minx, miny, maxx, maxy));
        }
        let mut vc = vcounts.clone();
        vc.sort_unstable();
        println!(
            "  features: {nfeat}, vertex-count min/median/max: {}/{}/{}",
            vc.first().unwrap_or(&0),
            vc.get(vc.len() / 2).unwrap_or(&0),
            vc.last().unwrap_or(&0)
        );
        if let Some(w) = &width {
            let mut ws: Vec<f64> = (0..nfeat).map(|i| w.value(i)).collect();
            ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  width min/median/max: {:.2}/{:.2}/{:.2}",
                ws[0],
                ws[ws.len() / 2],
                ws[ws.len() - 1]
            );
        }
        // Count heavily-overlapping feature pairs (IoU-ish on bbox).
        let mut overlap_pairs = 0u64;
        for i in 0..nfeat {
            for j in (i + 1)..nfeat {
                if bbox_overlap_frac(bboxes[i], bboxes[j]) > 0.5 {
                    overlap_pairs += 1;
                }
            }
            if i > 400 {
                break;
            } // cap the O(n^2) on huge tiles
        }
        println!("  bbox-overlap>50% feature pairs (first ~400): {overlap_pairs}");
    }
    Ok(())
}

fn read_xy(vals: &dyn Array) -> Box<dyn Fn(usize) -> (f64, f64) + '_> {
    if let Some(a) = vals.as_any().downcast_ref::<Int32Array>() {
        Box::new(move |v| (a.value(v * 2) as f64, a.value(v * 2 + 1) as f64))
    } else if let Some(a) = vals.as_any().downcast_ref::<Float64Array>() {
        Box::new(move |v| (a.value(v * 2), a.value(v * 2 + 1)))
    } else {
        panic!("coord leaf is neither Int32 nor Float64");
    }
}

fn bbox_overlap_frac(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> f64 {
    let ix = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let iy = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    let inter = ix * iy;
    let area_a = (a.2 - a.0).max(1e-9) * (a.3 - a.1).max(1e-9);
    let area_b = (b.2 - b.0).max(1e-9) * (b.3 - b.1).max(1e-9);
    inter / area_a.min(area_b)
}
