//! Per-zoom directory stats for a packed dataset: entry counts, blob sizes,
//! temporal buckets — the loading-health numbers (max primary-zoom tile,
//! per-bucket weight) straight from the directory, no payload reads.
//!
//! Usage: cargo run --release -p stt-core --example packed-stats -- <manifest.json>

use std::collections::BTreeMap;
use stt_core::pack::PackedReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: packed-stats <manifest.json>");
        std::process::exit(2);
    }
    let reader = PackedReader::open(&args[1])?;
    let meta = reader.metadata();
    let entries = reader.entries();

    println!(
        "{}: {} entries, base bucket {} ms, time {}..{}",
        args[1],
        entries.len(),
        meta.temporal_bucket_ms,
        meta.time_range.start,
        meta.time_range.end
    );

    #[derive(Default)]
    struct Z {
        n: u64,
        bytes: u64,
        max: u32,
        max_key: String,
        covered: u64,
    }
    let mut per_zoom: BTreeMap<u8, Z> = BTreeMap::new();
    for e in entries {
        let z = per_zoom.entry(e.zoom).or_default();
        z.n += 1;
        z.bytes += e.length as u64;
        if e.length > z.max {
            z.max = e.length;
            z.max_key = format!("z{}/{}/{} t{}", e.zoom, e.x, e.y, e.time_start);
        }
        if e.cover_t_min.is_some() {
            z.covered += 1;
        }
    }
    println!("zoom |   entries |   total MB | max tile MB | covered | max tile");
    for (zoom, z) in &per_zoom {
        println!(
            "  {:2} | {:9} | {:10.1} | {:11.2} | {:7} | {}",
            zoom,
            z.n,
            z.bytes as f64 / 1e6,
            z.max as f64 / 1e6,
            z.covered,
            z.max_key
        );
    }
    Ok(())
}
