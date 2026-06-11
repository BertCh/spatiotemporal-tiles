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

fn zstd_len(bytes: &[u8]) -> usize {
    zstd::bulk::compress(bytes, 9).map(|v| v.len()).unwrap_or(bytes.len())
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
        // Root page: ~32 B/page entry (key + offset/len + t-bounds), zstd'd at
        // a conservative 2x.
        let root_bytes = pages.len() * 32 / 2;

        // Per-query: pages overlapping the 5x5 hilbert range at the anchor's
        // zoom (hilbert box approximated as the min..max hilbert of the 25
        // tile keys), then time-pruned by page [t_min,t_max].
        let mut fetched: Vec<usize> = Vec::with_capacity(anchors.len());
        for a in &anchors {
            let (z, t0) = (a.zoom, a.time_start);
            let t1 = t0 + 3_600_000;
            // Hilbert indices of the 5x5 box around the anchor tile.
            let mut hmin = u64::MAX;
            let mut hmax = 0u64;
            for dx in 0..5i64 {
                for dy in 0..5i64 {
                    let x = (a.x as i64 + dx - 2).max(0) as u32;
                    let y = (a.y as i64 + dy - 2).max(0) as u32;
                    let h = stt_core::tile::TileId::new(z, x, y, 0).hilbert_index();
                    hmin = hmin.min(h);
                    hmax = hmax.max(h);
                }
            }
            let mut bytes = root_bytes;
            for (pi, p) in pages.iter().enumerate() {
                let pz0 = p.first().unwrap();
                let pz1 = p.last().unwrap();
                // Page key range overlaps the query's (zoom, hilbert) range?
                let key_lo = (pz0.zoom, pz0.hilbert);
                let key_hi = (pz1.zoom, pz1.hilbert);
                if key_hi < (z, hmin) || key_lo > (z, hmax) {
                    continue;
                }
                // Time prune on page t-bounds (the COPC-temporal trick).
                let pt0 = p.iter().map(|e| e.cover_t_min.unwrap_or(e.time_start)).min().unwrap();
                let pt1 = p.iter().map(|e| e.time_end).max().unwrap();
                if pt1 < t0 || pt0 > t1 {
                    continue;
                }
                bytes += page_sizes[pi];
            }
            fetched.push(bytes);
        }
        fetched.sort_unstable();
        let med = fetched[fetched.len() / 2];
        let p90 = fetched[fetched.len() * 9 / 10];
        println!(
            "pages of {:>5}: {:>4} pages, paged total {:>9} B (+{:>4.1}% vs whole) | query bytes med {:>8} p90 {:>8} ({:.1}% / {:.1}% of whole-load)",
            page_entries,
            pages.len(),
            paged_total + root_bytes,
            ((paged_total + root_bytes) as f64 / whole_zstd as f64 - 1.0) * 100.0,
            med,
            p90,
            med as f64 * 100.0 / whole_zstd as f64,
            p90 as f64 * 100.0 / whole_zstd as f64,
        );
    }
    Ok(())
}
