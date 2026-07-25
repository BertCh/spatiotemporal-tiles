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
//!
//! `--v2` emits the SAME source data as a version-coherent packed-v2 build
//! (`formatVersion: 2` manifests + v2 frames) into sibling `*-v2/` dirs,
//! leaving the committed v1 fixture bytes untouched — for future fixture
//! needs; nothing consumes the v2 output yet, so it is not committed.

use std::path::PathBuf;
use stt_core::arrow_tile::{encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn};
use stt_core::metadata::Metadata;
use stt_core::pack::{PACKED_FORMAT_VERSION_V1, PACKED_FORMAT_VERSION_V2};
use stt_core::{BlobOrdering, PackWriter, TileId};

/// Frame-version-coherent encoder config: frames follow the writer's
/// formatVersion (mirroring stt-build's `pack_encoder_config`) instead of the
/// process-global encode default, so a dataset can never mix frame and
/// manifest versions — readers hard-reject mixed datasets (packed-v2 design
/// §1 ★F6). v2 emits SELF-CONTAINED frames (no template collector): the
/// paged + single grid builds below use two writers, and inline schema
/// sections keep their payload bytes byte-shareable without shared-collector
/// plumbing.
fn encoder_config(format_version: u32) -> EncoderConfig {
    EncoderConfig {
        format_version,
        ..EncoderConfig::default()
    }
}

/// Build the deterministic payload for tile index `k`. `id_seed` lets the
/// deduped tiles (k=4, k=9) reuse k=0's identity bytes.
fn payload_for(id_seed: u64, cfg: &EncoderConfig) -> Vec<u8> {
    let ids: Vec<u64> = (0..3).map(|i| 100 * id_seed + i).collect();
    let n = ids.len();
    encode_tile_with(
        &[ColumnarLayer {
            name: "default".to_string(),
            feature_ids: ids,
            start_times: vec![1000 * id_seed as i64; n],
            end_times: vec![1000 * id_seed as i64 + 100; n],
            geometry: GeometryColumn::Point(vec![[-122.4 + 0.01 * id_seed as f64, 37.7]; n]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        }],
        cfg,
    )
    .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Version coherence: the committed fixtures are v1 (byte-frozen), so the
    // writers are PINNED to v1 rather than inheriting `PackWriter`'s v2
    // default — and the payload frames follow the same pin (see
    // `encoder_config`). `--v2` writes a coherent v2 copy NEXT TO the v1
    // outputs (`*-v2/` suffix), never over them.
    let mut v2 = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--v2" => v2 = true,
            other => return Err(format!("unknown argument {other:?} (only --v2)").into()),
        }
    }
    let format_version = if v2 {
        PACKED_FORMAT_VERSION_V2
    } else {
        PACKED_FORMAT_VERSION_V1
    };
    let suffix = if v2 { "-v2" } else { "" };
    let cfg = encoder_config(format_version);

    // <crate>/../../packages/core/test/fixtures/packed-golden
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = manifest_dir
        .join("..")
        .join("..")
        .join("packages")
        .join("core")
        .join("test")
        .join("fixtures")
        .join(format!("packed-golden{suffix}"));

    // Start clean so re-running is deterministic (stale packs would linger
    // otherwise — content-addressed names mean old objects never get clobbered).
    if out_dir.exists() {
        std::fs::remove_dir_all(&out_dir)?;
    }

    // SpatialMajor keeps the order deterministic + independent of the Auto
    // heuristic, so the content hashes are stable across builds.
    let mut w = PackWriter::create(&out_dir, BlobOrdering::SpatialMajor, 4 * 1024)?
        .with_format_version(format_version);

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
            .or_insert_with(|| payload_for(id_seed, &cfg))
            .clone();
        let t = 1000 * k as i64;
        // Distinct spatial cell per tile so all 12 directory entries survive.
        let id = TileId::new(10, k as u32, 0, t.max(0) as u64);
        w.add_tile_full(&id, t, t + 100, Some(t), 3, Some(1000), &payload)?;
    }

    let meta = Metadata::new("packed-golden")
        .with_description("Deterministic STT packed-format cross-impl fixture")
        .with_zoom_levels(10, 10)
        .with_temporal_bucket_ms(1000)
        // Tile k spans [1000*k, 1000*k + 100]; cover the lot so the fixture
        // passes `stt-validate`'s temporal-extent cross-check.
        .with_time_range(stt_core::types::TimeRange::new(
            0,
            1000 * (n_tiles - 1) + 100,
        ));

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
    println!(
        "  directory {} ({} bytes)",
        manifest.directory.key, manifest.directory.length
    );

    // --- Paged-directory cross-impl fixtures --------------------------------
    // A richer grid dataset (spatial spread × time buckets × two zooms) emitted
    // BOTH paged (`paged-golden/`) and single (`paged-golden-single/`) from the
    // SAME shared payloads — so the TS differential test can assert the paged
    // query path returns byte-identical results to a whole-load directory while
    // fetching only a fraction of the leaf pages.
    let base = manifest_dir
        .join("..")
        .join("..")
        .join("packages")
        .join("core")
        .join("test")
        .join("fixtures");
    let grid = build_grid_tiles(&cfg);
    let grid_meta = Metadata::new("paged-golden")
        .with_description("Deterministic STT paged-directory cross-impl fixture")
        .with_zoom_levels(10, 12)
        .with_temporal_bucket_ms(3_600_000)
        .with_time_range(stt_core::types::TimeRange::new(0, 3 * 3_600_000));
    for (sub, paging) in [
        (format!("paged-golden{suffix}"), Some(8usize)),
        (format!("paged-golden-single{suffix}"), None),
    ] {
        let dir = base.join(&sub);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        // SpatialMajor → deterministic order + stable content hashes; identical
        // shared payload bytes → the two builds' packs are byte-identical and
        // only the directory container differs.
        let mut gw = PackWriter::create(&dir, BlobOrdering::SpatialMajor, 8 * 1024)?
            .with_format_version(format_version)
            .with_paging(paging);
        for (id, ts, te, payload) in &grid {
            gw.add_tile_full(id, *ts, *te, Some(*ts), 3, Some(3_600_000), payload)?;
        }
        let gm = gw.finalize(&grid_meta)?;
        println!(
            "wrote {} to {}: {} tiles, {} packs, dir {} bytes{}",
            sub,
            dir.display(),
            grid.len(),
            gm.packs.len(),
            gm.directory.length,
            gm.directory
                .page_count
                .map(|p| format!(
                    " (paged: {p} leaf pages, root {} B)",
                    gm.directory.root_length.unwrap_or(0)
                ))
                .unwrap_or_default(),
        );
    }
    Ok(())
}

/// A spread-out grid of tiles across two zooms and three time buckets, with
/// deterministic per-tile payloads built once (so paged + single builds share
/// byte-identical blobs). ~250 entries → several leaf pages at page size 8.
fn build_grid_tiles(cfg: &EncoderConfig) -> Vec<(TileId, i64, i64, Vec<u8>)> {
    let mut out = Vec::new();
    let mut seed = 1_000u64;
    let bucket = 3_600_000i64;
    // Zoom 10: an 18×12 block over Europe-ish tile coords.
    for gx in 0..18u32 {
        for gy in 0..12u32 {
            let (x, y) = (520 + gx, 384 + gy);
            let b = ((gx + gy) % 3) as i64;
            let t = b * bucket;
            out.push((
                TileId::new(10, x, y, t.max(0) as u64),
                t,
                t + bucket - 1,
                payload_for(seed, cfg),
            ));
            seed += 1;
        }
    }
    // Zoom 12: a 6×6 block (distinct zoom exercises zoom-range pruning).
    for gx in 0..6u32 {
        for gy in 0..6u32 {
            let (x, y) = (2084 + gx, 1536 + gy);
            let b = ((gx * 7 + gy) % 3) as i64;
            let t = b * bucket;
            out.push((
                TileId::new(12, x, y, t.max(0) as u64),
                t,
                t + bucket - 1,
                payload_for(seed, cfg),
            ));
            seed += 1;
        }
    }
    out
}
