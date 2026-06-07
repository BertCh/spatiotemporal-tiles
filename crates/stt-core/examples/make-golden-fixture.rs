//! Emit the committed cross-impl golden fixture for the TS packed-format reader.
//!
//! Writes a deterministic tiny packed dataset (fixed payloads → stable content
//! hashes) to `packages/core/test/fixtures/packed-golden/`:
//!
//!   manifest.json
//!   index/<hash>.sttd
//!   packs/<hash>.sttp   (2-3 packs at the tiny pack target)
//!
//! The TS test reads this fixture and must reproduce identical decoded payloads.
//! The payload scheme is intentionally simple + known so it's easy to assert on
//! both sides:
//!
//!   - 12 tiles, zoom 10, at spatial cells `(x, 0)` for x in 0..12.
//!   - Each tile carries one `"default"` point layer with feature ids
//!     `[100*k + 0, 100*k + 1, 100*k + 2]` for tile index k (0..12) at point
//!     `(-122.4 + 0.01*k, 37.7)`, start/end times `(1000*k, 1000*k + 100)`.
//!   - Tiles k=4 and k=9 are byte-identical to tile k=0 (same feature ids /
//!     coords / times) → exercise byte-identical blob dedup. So distinct decoded
//!     payloads correspond to k in {0,1,2,3,5,6,7,8,10,11} (10 distinct), with
//!     k=4 and k=9 decoding identically to k=0.
//!   - Tile time_start = 1000*k, except the deduped k=4,k=9 which reuse k=0's
//!     id/coords but keep their OWN time_start so the directory keeps 12 entries.
//!
//! The pack target is 4 KiB to cut the ~10 distinct blobs into 2-3 packs.
//!
//! Run: cargo run -p stt-core --example make-golden-fixture

use stt_core::arrow_tile::{encode_tile, ColumnarLayer, GeometryColumn};
use stt_core::metadata::Metadata;
use stt_core::{BlobOrdering, PackWriter, TileId};
use std::path::PathBuf;

/// Build the deterministic payload for tile index `k`. `id_seed` lets the
/// deduped tiles (k=4, k=9) reuse k=0's identity bytes.
fn payload_for(id_seed: u64) -> Vec<u8> {
    let ids: Vec<u64> = (0..3).map(|i| 100 * id_seed + i).collect();
    let n = ids.len();
    encode_tile(&[ColumnarLayer {
        name: "default".to_string(),
        feature_ids: ids,
        start_times: vec![1000 * id_seed as i64; n],
        end_times: vec![1000 * id_seed as i64 + 100; n],
        geometry: GeometryColumn::Point(vec![[-122.4 + 0.01 * id_seed as f64, 37.7]; n]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        properties: vec![],
    }])
    .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // <crate>/../../packages/core/test/fixtures/packed-golden
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = manifest_dir
        .join("..")
        .join("..")
        .join("packages")
        .join("core")
        .join("test")
        .join("fixtures")
        .join("packed-golden");

    // Start clean so re-running is deterministic (stale packs would linger
    // otherwise — content-addressed names mean old objects never get clobbered).
    if out_dir.exists() {
        std::fs::remove_dir_all(&out_dir)?;
    }

    // SpatialMajor keeps the order deterministic + independent of the Auto
    // heuristic, so the content hashes are stable across builds.
    let mut w = PackWriter::create(&out_dir, BlobOrdering::SpatialMajor, 4 * 1024)?;

    // `encode_tile` is NOT byte-deterministic across separate calls (Arrow IPC
    // framing), so build each distinct payload exactly ONCE and CLONE it for the
    // dedup cases — only the clone is byte-identical, which is what the writer's
    // blake3 dedup keys on. (This also makes the committed content hashes stable
    // across regenerations of the fixture.)
    let n_tiles = 12u64;
    let mut distinct: std::collections::HashMap<u64, Vec<u8>> = std::collections::HashMap::new();
    for k in 0..n_tiles {
        // k=4 and k=9 reuse k=0's identity → byte-identical blobs → dedup.
        let id_seed = if k == 4 || k == 9 { 0 } else { k };
        let payload = distinct
            .entry(id_seed)
            .or_insert_with(|| payload_for(id_seed))
            .clone();
        let t = 1000 * k as i64;
        // Distinct spatial cell per tile so all 12 directory entries survive.
        let id = TileId::new(10, k as u32, 0, t.max(0) as u64);
        w.add_tile_full(&id, t, t + 100, Some(t), 3, Some(1000), &payload)?;
    }

    let meta = Metadata::new("packed-golden")
        .with_description("Deterministic STT packed-format cross-impl fixture")
        .with_zoom_levels(10, 10)
        .with_temporal_bucket_ms(1000);

    let manifest = w.finalize(&meta)?;
    let total: u64 = manifest.packs.iter().map(|p| p.length).sum();

    println!(
        "wrote golden fixture to {}: {} tiles, {} packs, {} pack bytes, dir {} bytes",
        out_dir.display(),
        n_tiles,
        manifest.packs.len(),
        total,
        manifest.directory.length,
    );
    for (i, p) in manifest.packs.iter().enumerate() {
        println!("  pack[{i}] {} ({} bytes)", p.key, p.length);
    }
    println!("  directory {} ({} bytes)", manifest.directory.key, manifest.directory.length);
    Ok(())
}
