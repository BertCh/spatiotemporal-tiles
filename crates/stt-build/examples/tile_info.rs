//! Decode a raw STT tile blob (e.g. one served by `stt-serve` or extracted from
//! a pack) and print its layers + per-layer feature counts. A small debugging /
//! verification aid.
//!
//! Usage: cargo run -p stt-build --example tile_info -- <tile.stt>

use std::io::Read;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: tile_info <tile-blob-file>"))?;
    let mut bytes = Vec::new();
    std::fs::File::open(&path)?.read_to_end(&mut bytes)?;

    let layers = stt_core::arrow_tile::decode_tile(&bytes)?;
    let total: usize = layers.iter().map(|l| l.batch.num_rows()).sum();
    println!("{path}: {} bytes, {} layer(s), {total} feature(s)", bytes.len(), layers.len());
    for l in &layers {
        println!(
            "  layer '{}': {} rows, columns: {}",
            l.name,
            l.batch.num_rows(),
            l.batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}
