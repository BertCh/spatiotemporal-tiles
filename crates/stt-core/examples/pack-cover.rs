//! Backfill the tight covering bound (`cover_t_min`) on a packed dataset whose
//! directory predates the covering section, re-packing losslessly.
//!
//! For every directory entry missing `cover_t_min`, decodes the tile's Arrow
//! layers and takes the minimum per-feature `start_time` — the same bound a
//! fresh `stt-build` run computes at tiling (`stt-build::tiler`). Entries that
//! already carry a bound keep it. Payload bytes are untouched; only the
//! directory changes, so the output is byte-faithful data with a fully covered
//! directory (the covering section only encodes when EVERY entry has a bound).
//!
//! Usage:
//!   cargo run --release -p stt-core --example pack-cover -- \
//!     <in_dir/manifest.json> <out_dir> [pack_size_mb=64] [ordering=auto]

use arrow::array::Int64Array;
use stt_core::pack::{PackWriter, PackedReader};
use stt_core::tile::TileId;
use stt_core::BlobOrdering;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: pack-cover <in_dir/manifest.json> <out_dir> [pack_size_mb=64] [ordering=auto]"
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

    let reader = PackedReader::open(inp)?;
    let metadata = reader.metadata().clone();
    let entries = reader.entries().to_vec();

    // Payloads are repacked verbatim, so the source's capability
    // declarations (spec §3.1) must be carried forward — dropping them would
    // re-arm the silent-misdecode hazard on quantized/folded source data.
    // The source's formatVersion is carried too (frames and manifest may not
    // mix; readers hard-reject), and a v2 source's registry seeds the new
    // manifest's schemas table — copied frames reference templates by hash
    // but never re-encode, so nothing else would record them.
    let mut writer = PackWriter::create(out_dir, ordering, pack_size_mb * 1024 * 1024)?
        .with_capabilities(reader.capabilities().to_vec());
    if let Some(templates) = reader.templates() {
        writer = writer.with_seeded_templates(templates);
    }
    let (mut kept, mut computed, mut empty) = (0u64, 0u64, 0u64);
    for entry in &entries {
        let payload = reader.read_payload(entry)?;
        let cover = match entry.cover_t_min {
            Some(c) => {
                kept += 1;
                Some(c)
            }
            None => {
                let mut min_t: Option<i64> = None;
                for layer in reader.read_layers(entry)? {
                    let Some(col) = layer.batch.column_by_name("start_time") else {
                        continue;
                    };
                    let Some(arr) = col.as_any().downcast_ref::<Int64Array>() else {
                        continue;
                    };
                    for v in arr.iter().flatten() {
                        min_t = Some(min_t.map_or(v, |m: i64| m.min(v)));
                    }
                }
                if min_t.is_some() {
                    computed += 1;
                } else {
                    empty += 1; // no feature rows: fall back to the bucket edge
                }
                Some(min_t.unwrap_or(entry.time_start))
            }
        };
        writer.add_tile_full(
            &TileId::new(entry.zoom, entry.x, entry.y, entry.time_start.max(0) as u64),
            entry.time_start,
            entry.time_end,
            cover,
            entry.feature_count,
            entry.temporal_bucket_ms,
            &payload,
        )?;
    }

    let manifest = writer.finalize(&metadata)?;
    println!(
        "pack-cover {} -> {}: {} entries ({} kept, {} computed, {} empty-fallback), {} packs, dir {} bytes",
        inp,
        out_dir,
        entries.len(),
        kept,
        computed,
        empty,
        manifest.packs.len(),
        manifest.directory.length,
    );
    Ok(())
}
