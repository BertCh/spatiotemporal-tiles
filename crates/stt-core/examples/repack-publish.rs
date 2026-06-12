//! Publish-build re-transcode of an existing **packed** dataset: re-compress
//! every tile at a higher zstd level and emit a paged directory, reading the
//! current packed dir as the source of truth (so datasets rebuilt later than
//! their legacy `.stt` keep their fixes). Tile content is preserved losslessly.
//!
//! Writes to a fresh `<out_dir>`; the caller swaps it in (see
//! `scripts/repack-publish-all.sh`). `<out_dir>` MUST differ from the input.
//!
//! Usage:
//!   cargo run --release -p stt-core --example repack-publish -- \
//!     <in_dir/manifest.json> <out_dir> [zstd_level=19] [page_entries=4096] [pack_mb=64]

use stt_core::{transcode_packs_to_packs_paged_level, BlobOrdering};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: repack-publish <in_dir/manifest.json> <out_dir> [zstd_level=19] [page_entries=4096] [pack_mb=64]");
        std::process::exit(2);
    }
    let (inp, out_dir) = (&args[1], &args[2]);
    let zstd_level: i32 = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(19);
    let page_entries: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(4096);
    let pack_mb: u64 = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(64);
    let paging = (page_entries > 0).then_some(page_entries);
    let pack_target_bytes = pack_mb.saturating_mul(1024 * 1024).max(1);

    let manifest = transcode_packs_to_packs_paged_level(
        inp,
        out_dir,
        BlobOrdering::Auto,
        pack_target_bytes,
        paging,
        zstd_level,
    )?;

    let total: u64 = manifest.packs.iter().map(|p| p.length).sum();
    println!(
        "repacked {} -> {} (zstd-{}, {} packs, {} pack bytes, dir {} bytes{})",
        inp,
        out_dir,
        zstd_level,
        manifest.packs.len(),
        total,
        manifest.directory.length,
        manifest
            .directory
            .page_count
            .map(|p| format!(", paged: {p} leaf pages"))
            .unwrap_or_default(),
    );
    Ok(())
}
