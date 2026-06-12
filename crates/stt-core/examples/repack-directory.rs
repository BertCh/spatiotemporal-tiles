//! Re-encode ONLY the directory of an existing packed dataset — switching it to
//! (or from) the paged container — without re-reading or re-compressing any tile
//! payload. The packs keep their exact bytes and content-addressed names; only
//! `index/<hash>.sttd` + `manifest.json` change. This is the cheap Wave-2
//! re-transcode the fleet migration uses: a re-sync re-ships two small objects
//! per dataset, not the packs.
//!
//! Usage:
//!   cargo run --release -p stt-core --example repack-directory -- \
//!     <in_dir/manifest.json> <out_dir> [page_entries=4096]
//!
//!   page_entries: 0 = single whole-load directory; >0 = paged (default 4096).
//!   out_dir may equal the input directory (in-place: adds the new index +
//!   overwrites the manifest; packs untouched).

use stt_core::repack_directory;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: repack-directory <in_dir/manifest.json> <out_dir> [page_entries=4096]");
        std::process::exit(2);
    }
    let (inp, out_dir) = (&args[1], &args[2]);
    let page_entries: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4096);
    let paging = (page_entries > 0).then_some(page_entries);

    let manifest = repack_directory(inp, out_dir, paging)?;
    let dir = &manifest.directory;
    println!(
        "repacked {} -> {}: {} packs (unchanged), directory {} bytes{}",
        inp,
        out_dir,
        manifest.packs.len(),
        dir.length,
        dir.page_count
            .map(|p| format!(
                " (paged: {p} leaf pages of ≤{} entries, root {} B)",
                dir.page_entries.unwrap_or(0),
                dir.root_length.unwrap_or(0)
            ))
            .unwrap_or_else(|| " (single whole-load)".to_string()),
    );
    Ok(())
}
