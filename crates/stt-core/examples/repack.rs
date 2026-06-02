//! Repack a v4 STT archive through the buffered/optimized writer with a chosen
//! blob ordering — the production reorder pass, exercised standalone.
//!
//! Reads every tile payload (lossless) and re-writes via
//! `ArchiveWriter::create_optimized_with_ordering`, so the output has identical
//! tiles + directory but blobs laid down in the requested byte order. Verify the
//! win by running `simulate_layout` on the output: its `0-REAL-DISK` row (the new
//! on-disk order) should now match the chosen ordering's predicted cost.
//!
//! Usage: cargo run --release -p stt-core --example repack -- <in.stt> <out.stt> [ordering]
//!   ordering: spatial | time-major | hilbert3 (default) | morton3

use stt_core::archive::ArchiveWriter;
use stt_core::tile::TileId;
use stt_core::{ArchiveReader, BlobOrdering};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: repack <in.stt> <out.stt> [spatial|time-major|hilbert3|morton3]");
        std::process::exit(2);
    }
    let (inp, outp) = (&args[1], &args[2]);
    let ordering: BlobOrdering = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(BlobOrdering::Hilbert3);

    let reader = ArchiveReader::open(inp)?;
    let meta = reader.metadata().clone();
    let entries = reader.entries().to_vec();

    // Reader-safe path: reorder blobs (the range-request win) WITHOUT a shared
    // dictionary, which the current TS reader can't decode. Per-blob zstd +
    // dedup still apply.
    let mut writer = ArchiveWriter::create_reordered(outp, ordering)?;
    for e in &entries {
        let payload = reader.read_payload(e)?;
        // Preserve the tight covering bound (cover_t_min) across the repack —
        // without this the optimize-tiles reorder pass would silently drop the
        // covering a fresh build computed.
        writer.add_tile_full(
            &TileId::new(e.zoom, e.x, e.y, e.time_start.max(0) as u64),
            e.time_start,
            e.time_end,
            e.cover_t_min,
            e.feature_count,
            e.temporal_bucket_ms,
            &payload,
        )?;
    }
    writer.finalize(&meta)?;

    println!(
        "repacked {} -> {} ({} tiles, ordering={})",
        inp,
        outp,
        entries.len(),
        ordering
    );
    Ok(())
}
