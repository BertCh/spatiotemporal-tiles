//! Migrate a tree of packed datasets from `formatVersion` 2 to 3 in place.
//!
//! Container-only: rewrites each `manifest.json` and writes one new
//! `index/<hash>.sttd`. **No pack object is read, written or renamed**, which is
//! the entire point — the packs are already on the CDN under content addresses
//! that do not move, so publishing a migrated fleet uploads kilobytes per
//! dataset instead of gigabytes.
//!
//! Every dataset is verified after it is written: reopened through
//! `PackedReader` and decoded entry-by-entry against what it decoded to BEFORE.
//! A dataset whose content moves is a bug, not a rounding error, so the run
//! stops on it rather than continuing and burying the evidence.
//!
//! Usage:
//!   cargo run --release -p stt-core --example migrate-fleet -- <tree> [--verify-all]
//!
//! `<tree>` is a directory of dataset directories. Without `--verify-all` the
//! decode check samples the first 64 entries per dataset (enough to catch a
//! broken re-encode, fast enough for a 60-archive fleet); with it, every entry
//! is decoded on both sides.

use std::path::Path;
use stt_core::pack::{migrate_dataset_v2_to_v3, PackedReader};

type Digest = Vec<(u8, u32, u32, i64, Vec<(String, usize)>)>;

fn decode(dir: &Path, limit: Option<usize>) -> Result<Digest, Box<dyn std::error::Error>> {
    let r = PackedReader::open(dir.join("manifest.json"))?;
    let take = limit.unwrap_or(usize::MAX);
    let mut out = Vec::new();
    for e in r.entries().iter().take(take) {
        let layers = r.read_layers(e)?;
        out.push((
            e.zoom,
            e.x,
            e.y,
            e.time_start,
            layers
                .iter()
                .map(|l| (l.name.clone(), l.batch.num_rows()))
                .collect(),
        ));
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .expect("usage: migrate-fleet <tree> [--verify-all]");
    let verify_all = args.any(|a| a == "--verify-all");
    let limit = if verify_all { None } else { Some(64usize) };
    let root = Path::new(&root);

    let mut dirs: Vec<_> = std::fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("manifest.json").is_file())
        .collect();
    dirs.sort();

    let (mut migrated, mut already, mut refused) = (0usize, 0usize, 0usize);
    let mut bytes_new = 0u64;
    for dir in &dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        // Skip hidden backup dirs (`.foo.bak-drift`) — not part of the fleet.
        if name.starts_with('.') {
            continue;
        }
        let before = match decode(dir, limit) {
            Ok(d) => d,
            Err(e) => {
                println!("  {name:28} SKIP  unreadable before migration: {e}");
                refused += 1;
                continue;
            }
        };
        match migrate_dataset_v2_to_v3(dir) {
            Ok(None) => {
                println!("  {name:28} --    already v3");
                already += 1;
            }
            Ok(Some(rep)) => {
                let after = decode(dir, limit)?;
                if after != before {
                    return Err(format!(
                        "{name}: DECODED CONTENT CHANGED across migration — stopping"
                    )
                    .into());
                }
                let sz = std::fs::metadata(dir.join(&rep.new_directory_key))?.len();
                bytes_new += sz;
                println!(
                    "  {name:28} ok    {} entries, {} packs untouched, paged={}, new dir {} KiB",
                    rep.entries,
                    rep.packs_unchanged,
                    rep.paged,
                    sz / 1024
                );
                migrated += 1;
            }
            Err(e) => {
                println!("  {name:28} REFUSED  {e}");
                refused += 1;
            }
        }
    }
    println!(
        "\nmigrated {migrated}, already v3 {already}, refused {refused} — \
         {} KiB of new directory objects, 0 packs rewritten",
        bytes_new / 1024
    );
    Ok(())
}
