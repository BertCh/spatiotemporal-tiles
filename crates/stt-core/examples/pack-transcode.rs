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
//!     <in.stt> <out_dir> [pack_size_mb=64] [ordering=auto] [page_entries=0] [zstd_level=3]
//!
//!   ordering: spatial | time-major | hilbert3 | morton3 | auto (default)
//!   page_entries: 0 (default) = single whole-load directory; >0 = a paged
//!                 directory with leaf pages of that many entries (try 4096).
//!   zstd_level: 3 (default, "fast") .. 19 (publish; −10..19% on the wire, decode
//!               unaffected). The format is write-once / serve-many.

use stt_core::{transcode_archive_to_packs_paged_level, BlobOrdering};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: pack-transcode <in.stt> <out_dir> [pack_size_mb=64] [spatial|time-major|hilbert3|morton3|auto] [page_entries=0]"
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
    let page_entries: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let zstd_level: i32 = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let paging = (page_entries > 0).then_some(page_entries);
    let pack_target_bytes = pack_size_mb.saturating_mul(1024 * 1024).max(1);

    // Lossless re-wrap (preserves cover_t_min + temporal_bucket_ms). Shared with
    // stt-build's --streaming-arrow path via `transcode_archive_to_packs_paged_level`.
    let manifest = transcode_archive_to_packs_paged_level(
        inp,
        out_dir,
        ordering,
        pack_target_bytes,
        paging,
        zstd_level,
    )?;

    let total_pack_bytes: u64 = manifest.packs.iter().map(|p| p.length).sum();
    println!(
        "packed {} -> {} ({} packs, {} pack bytes, dir {} bytes{}, ordering={})",
        inp,
        out_dir,
        manifest.packs.len(),
        total_pack_bytes,
        manifest.directory.length,
        manifest
            .directory
            .page_count
            .map(|p| format!(", paged: {p} leaf pages, root {} B", manifest.directory.root_length.unwrap_or(0)))
            .unwrap_or_default(),
        ordering
    );
    Ok(())
}
