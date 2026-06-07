//! Transcode a single-file v4 STT archive into the multi-object **packed
//! format** (manifest + content-addressed packs), with no generator run.
//!
//! Reads every tile payload (lossless) via `ArchiveReader` and streams it into
//! `PackWriter`, preserving the tight covering bound (`cover_t_min`) and the
//! per-tile `temporal_bucket_ms`, then finalizes with the source metadata. The
//! output dir gets a `manifest.json`, `index/<hash>.sttd`, and one or more
//! `packs/<hash>.sttp` objects.
//!
//! Usage:
//!   cargo run --release -p stt-core --example pack-transcode -- \
//!     <in.stt> <out_dir> [pack_size_mb=64] [ordering=auto]
//!
//!   ordering: spatial | time-major | hilbert3 | morton3 | auto (default)

use stt_core::{transcode_archive_to_packs, BlobOrdering};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: pack-transcode <in.stt> <out_dir> [pack_size_mb=64] [spatial|time-major|hilbert3|morton3|auto]"
        );
        std::process::exit(2);
    }
    let (inp, out_dir) = (&args[1], &args[2]);
    let pack_size_mb: u64 = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(64);
    let ordering: BlobOrdering = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(BlobOrdering::Auto);
    let pack_target_bytes = pack_size_mb.saturating_mul(1024 * 1024).max(1);

    // Lossless re-wrap (preserves cover_t_min + temporal_bucket_ms). Shared with
    // stt-build's --streaming-arrow path via `transcode_archive_to_packs`.
    let manifest = transcode_archive_to_packs(inp, out_dir, ordering, pack_target_bytes)?;

    let total_pack_bytes: u64 = manifest.packs.iter().map(|p| p.length).sum();
    println!(
        "packed {} -> {} ({} packs, {} pack bytes, dir {} bytes, ordering={})",
        inp,
        out_dir,
        manifest.packs.len(),
        total_pack_bytes,
        manifest.directory.length,
        ordering
    );
    Ok(())
}
