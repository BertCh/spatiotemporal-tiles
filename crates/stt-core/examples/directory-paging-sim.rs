//! Paged-directory feasibility sim (roadmap stt-packed §3).
//!
//! Answers two questions against real datasets, straight from their packed
//! directories:
//!
//! 1. **Encoding cost of paging** — the v5 codec is delta/RLE over one sorted
//!    stream; cutting it into independently decodable pages resets the delta
//!    state and splits runs at each boundary, and per-page zstd loses the big
//!    shared window. How many bytes does that actually cost?
//! 2. **Query win** — for viewport-shaped queries (a KxK tile box at primary
//!    zoom over a time window), how many pages / bytes does a paged reader
//!    fetch vs the whole-directory load it does today?
//!
//! Pages are contiguous slices of directory order `(zoom, hilbert,
//! time_start)` — the PMTiles leaf-directory model, which is also what COPC's
//! paged hierarchy reduces to for a flat sorted keyspace. The root page is
//! modelled at ~32 B/entry: first key (zoom, hilbert, t) + offset/len +
//! subtree t_min/t_max + min cover_t_min (the COPC-temporal-extension trick).
//!
//! Usage: cargo run --release -p stt-core --example directory-paging-sim -- \
//!          <manifest.json> [...more manifests]

use stt_core::archive::TileEntry;
use stt_core::directory::encode_directory;
use stt_core::pack::PackedReader;
use stt_core::projection::tile_geo_bounds as tile_geo_bbox;

fn zstd_len(bytes: &[u8]) -> usize {
    zstd::bulk::compress(bytes, 9).map(|v| v.len()).unwrap_or(bytes.len())
}

/// 2D bbox overlap test.
fn bbox_overlap(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    a.0 <= b.2 && b.0 <= a.2 && a.1 <= b.3 && b.1 <= a.3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: directory-paging-sim <manifest.json> [...]");
        std::process::exit(2);
    }
    for path in &args[1..] {
        if let Err(e) = run(path) {
            eprintln!("{path}: {e}");
        }
    }
    Ok(())
}

fn run(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let reader = PackedReader::open(path)?;
    let mut entries: Vec<TileEntry> = reader.entries().to_vec();
    entries.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
    let n = entries.len();
    if n == 0 {
        println!("{path}: empty");
        return Ok(());
    }

    let whole_raw = encode_directory(&entries);
    let whole_zstd = zstd_len(&whole_raw);
    println!("\n=== {path}");
    println!(
        "entries {n}; whole directory: {} B raw, {} B zstd ({:.1} B/entry at rest)",
        whole_raw.len(),
        whole_zstd,
        whole_zstd as f64 / n as f64
    );

    // Representative queries: 100 anchors sampled uniformly from primary-zoom
    // entries; each query = 5x5 tile box at that zoom x a 1h time window
    // around the anchor's time_start.
    let primary = entries.iter().map(|e| e.zoom).max().unwrap();
    let anchors: Vec<&TileEntry> = {
        let prim: Vec<&TileEntry> = entries.iter().filter(|e| e.zoom == primary).collect();
        (0..100).map(|i| prim[(i * prim.len().max(1) / 100).min(prim.len() - 1)]).collect()
    };

    for page_entries in [1024usize, 4096, 16384] {
        let pages: Vec<&[TileEntry]> = entries.chunks(page_entries).collect();
        let page_sizes: Vec<usize> = pages
            .iter()
            .map(|p| zstd_len(&encode_directory(p)))
            .collect();
        let paged_total: usize = page_sizes.iter().sum();

        // Precompute, per page, both candidate D3 descriptors:
        //  - Hilbert key range (first/last (zoom,hilbert)) — the sim's original
        //    model; spatially tight but needs a Hilbert port in the TS reader.
        //  - Geographic bbox + zoom range — zoom-correct, no Hilbert needed.
        // Plus the shared temporal [t_min, t_max] page bound.
        struct PageMeta {
            key_lo: (u8, u64),
            key_hi: (u8, u64),
            zmin: u8,
            zmax: u8,
            geo: (f64, f64, f64, f64),
            t0: i64,
            t1: i64,
        }
        let metas: Vec<PageMeta> = pages
            .iter()
            .map(|p| {
                let mut geo = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                let (mut zmin, mut zmax) = (u8::MAX, 0u8);
                for e in p.iter() {
                    let b = tile_geo_bbox(e.zoom, e.x, e.y);
                    geo.0 = geo.0.min(b.0);
                    geo.1 = geo.1.min(b.1);
                    geo.2 = geo.2.max(b.2);
                    geo.3 = geo.3.max(b.3);
                    zmin = zmin.min(e.zoom);
                    zmax = zmax.max(e.zoom);
                }
                PageMeta {
                    key_lo: (p.first().unwrap().zoom, p.first().unwrap().hilbert),
                    key_hi: (p.last().unwrap().zoom, p.last().unwrap().hilbert),
                    zmin,
                    zmax,
                    geo,
                    t0: p.iter().map(|e| e.cover_t_min.unwrap_or(e.time_start)).min().unwrap(),
                    t1: p.iter().map(|e| e.time_end).max().unwrap(),
                }
            })
            .collect();

        // Root page: geo-bbox descriptor ~50 B/page (offset/len/count + zoom
        // range + 4×i32 bbox + 2×i64 t-bounds), zstd'd at a conservative 2×.
        let root_bytes = pages.len() * 50 / 2;

        // Per-query: a 5×5 tile box at the anchor's zoom × a 1h window.
        let mut fetched_hil: Vec<usize> = Vec::with_capacity(anchors.len());
        let mut fetched_geo: Vec<usize> = Vec::with_capacity(anchors.len());
        for a in &anchors {
            let (z, t0) = (a.zoom, a.time_start);
            let t1 = t0 + 3_600_000;
            // Hilbert indices + geographic bbox of the 5×5 box around the anchor.
            let mut hmin = u64::MAX;
            let mut hmax = 0u64;
            let mut qgeo = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for dx in 0..5i64 {
                for dy in 0..5i64 {
                    let x = (a.x as i64 + dx - 2).max(0) as u32;
                    let y = (a.y as i64 + dy - 2).max(0) as u32;
                    let h = stt_core::tile::TileId::new(z, x, y, 0).hilbert_index();
                    hmin = hmin.min(h);
                    hmax = hmax.max(h);
                    let b = tile_geo_bbox(z, x, y);
                    qgeo.0 = qgeo.0.min(b.0);
                    qgeo.1 = qgeo.1.min(b.1);
                    qgeo.2 = qgeo.2.max(b.2);
                    qgeo.3 = qgeo.3.max(b.3);
                }
            }
            let mut bytes_hil = root_bytes;
            let mut bytes_geo = root_bytes;
            for (pi, m) in metas.iter().enumerate() {
                // Shared temporal prune (the COPC-temporal trick).
                let time_hit = !(m.t1 < t0 || m.t0 > t1);
                if !time_hit {
                    continue;
                }
                // Hilbert key-range overlap.
                if !(m.key_hi < (z, hmin) || m.key_lo > (z, hmax)) {
                    bytes_hil += page_sizes[pi];
                }
                // Geo-bbox + zoom-membership overlap.
                if z >= m.zmin && z <= m.zmax && bbox_overlap(m.geo, qgeo) {
                    bytes_geo += page_sizes[pi];
                }
            }
            fetched_hil.push(bytes_hil);
            fetched_geo.push(bytes_geo);
        }
        let pct = |v: &mut Vec<usize>| -> (usize, usize) {
            v.sort_unstable();
            (v[v.len() / 2], v[v.len() * 9 / 10])
        };
        let (hmed, hp90) = pct(&mut fetched_hil);
        let (gmed, gp90) = pct(&mut fetched_geo);
        println!(
            "pages of {:>5}: {:>4} pages, paged total {:>9} B (+{:>4.1}% vs whole)\n  \
             hilbert-range: med {:>8} p90 {:>8} ({:.1}% / {:.1}% of whole-load)\n  \
             geo-bbox+zoom: med {:>8} p90 {:>8} ({:.1}% / {:.1}% of whole-load)",
            page_entries,
            pages.len(),
            paged_total + root_bytes,
            ((paged_total + root_bytes) as f64 / whole_zstd as f64 - 1.0) * 100.0,
            hmed,
            hp90,
            hmed as f64 * 100.0 / whole_zstd as f64,
            hp90 as f64 * 100.0 / whole_zstd as f64,
            gmed,
            gp90,
            gmed as f64 * 100.0 / whole_zstd as f64,
            gp90 as f64 * 100.0 / whole_zstd as f64,
        );
    }
    Ok(())
}
