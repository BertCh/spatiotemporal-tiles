//! Density profile for a packed archive — the "cube laid down in a line" data
//! behind the showcase `CubeInLine` utility.
//!
//! Reads only the directory (no payload decode), downsamples the native tier
//! into a sparse `(sx, sy, tb)` occupancy + bytes cube, and stamps the real
//! archive's blob-ordering anchors: what `--blob-ordering auto` would pick, and
//! each ordering's adjacency-break count over the true tiles. All ordering math
//! is the production `stt_core::curve` code, so the numbers can never drift from
//! the writer.
//!
//! Usage:
//!   cargo run --release -p stt-core --example density-profile -- \
//!       <manifest.json> --id <id> --label "<Label>" [--space-bits 5] [--time-bins 64]
//!
//! JSON is written to stdout; redirect into
//! `poopdeck:examples/showcase/public/density/<id>.json`.

use std::collections::{BTreeMap, BTreeSet};

use stt_core::curve;
use stt_core::pack::PackedReader;
use stt_core::BlobOrdering;

/// Read `--flag value` from the argv slice.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let manifest = match args.get(1) {
        Some(m) if !m.starts_with("--") => m.clone(),
        _ => {
            eprintln!(
                "usage: density-profile <manifest.json> --id <id> --label <str> \
                 [--space-bits 5] [--time-bins 64]"
            );
            std::process::exit(2);
        }
    };
    let id = flag(&args, "--id").unwrap_or("dataset").to_string();
    let label = flag(&args, "--label").unwrap_or(&id).to_string();
    let space_bits: u32 = flag(&args, "--space-bits")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
        .clamp(1, 10);
    let time_bins_req: u32 = flag(&args, "--time-bins")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64)
        .clamp(1, 256);

    let reader = PackedReader::open(&manifest)?;
    let meta = reader.metadata();
    let bucket_ms = meta.temporal_bucket_ms.max(1) as i64;
    let max_zoom = meta.max_zoom;

    // The native (finest) tier is the honest picture of the data's shape; the
    // summary/LOD tiers are aggregates that would flatten the density.
    let native: Vec<&stt_core::TileEntry> = reader
        .entries()
        .iter()
        .filter(|e| e.zoom == max_zoom)
        .collect();
    if native.is_empty() {
        return Err(format!("no entries at native zoom {max_zoom}").into());
    }
    let tb_of = |e: &stt_core::TileEntry| e.time_start.div_euclid(bucket_ms);

    // --- extents over the native tier ------------------------------------
    let (mut x_min, mut x_max) = (u32::MAX, 0u32);
    let (mut y_min, mut y_max) = (u32::MAX, 0u32);
    let (mut tb_min, mut tb_max) = (i64::MAX, i64::MIN);
    for e in &native {
        x_min = x_min.min(e.x);
        x_max = x_max.max(e.x);
        y_min = y_min.min(e.y);
        y_max = y_max.max(e.y);
        let tb = tb_of(e);
        tb_min = tb_min.min(tb);
        tb_max = tb_max.max(tb);
    }
    let x_range = (x_max - x_min) as u64;
    let y_range = (y_max - y_min) as u64;
    let tb_span = tb_max - tb_min; // >= 0

    // --- blob-ordering anchors (over the *real* tiles) -------------------
    // Occupied spatial extent, symmetric with the time axis — matches the F1
    // fix in PackWriter::finalize so this probe's `autoChoice` is what the
    // writer actually resolves.
    let occ_space_bits = curve::bits_for(x_range.max(y_range) + 1);
    let time_bits = curve::bits_for((tb_span.max(0) + 1) as u64);
    let auto = BlobOrdering::choose(occ_space_bits, time_bits);
    let auto_key = ordering_key(auto);

    // Measured pick — mirrors `--blob-ordering measured` (simulate range-read
    // cost over the real native tiles; len = compressed on-disk blob size).
    let samples: Vec<stt_core::ordering_sim::TileSample> = native
        .iter()
        .map(|e| stt_core::ordering_sim::TileSample {
            z: e.zoom,
            x: e.x,
            y: e.y,
            hilbert: e.hilbert,
            time_start: e.time_start,
            tb: e.time_start.div_euclid(bucket_ms),
            len: e.length as u64,
        })
        .collect();
    // Pack target (the reader coalesces per-pack): the largest pack object, over
    // ALL zooms (a pack holds every tier). max(offset+length) per pack, then the
    // max across packs — floored at 1 MiB.
    let pack_bytes = {
        let mut ends: BTreeMap<u32, u64> = BTreeMap::new();
        for e in reader.entries() {
            let end = e.offset + e.length as u64;
            let slot = ends.entry(e.pack_id).or_insert(0);
            *slot = (*slot).max(end);
        }
        ends.values()
            .copied()
            .max()
            .unwrap_or(stt_core::ordering_sim::DEFAULT_PACK_BYTES)
            .max(1 << 20)
    };
    let measured_key = ordering_key(stt_core::ordering_sim::measured_ordering(
        &samples,
        stt_core::ordering_sim::SimOptions {
            pack_bytes,
            ..stt_core::ordering_sim::SimOptions::default()
        },
    ));

    // For each ordering, sort the true tiles into byte order and count the
    // "adjacency breaks": consecutive blobs that are not (x, y, t) neighbours —
    // every one is a potential seek the coalescer can't fuse.
    let orderings = [
        BlobOrdering::SpatialMajor,
        BlobOrdering::Hilbert3,
        BlobOrdering::Morton3,
        BlobOrdering::TimeMajor,
    ];
    let mut native_seeks: BTreeMap<&'static str, u64> = BTreeMap::new();
    for &o in &orderings {
        let mut idx: Vec<usize> = (0..native.len()).collect();
        idx.sort_by_key(|&i| {
            let e = native[i];
            curve::space_time_key(
                o,
                e.zoom,
                e.x,
                e.y,
                e.hilbert,
                e.time_start,
                tb_of(e),
                tb_min,
                tb_span,
            )
        });
        let mut breaks = 0u64;
        for w in idx.windows(2) {
            let a = native[w[0]];
            let b = native[w[1]];
            let d = (a.x as i64 - b.x as i64).abs()
                + (a.y as i64 - b.y as i64).abs()
                + (tb_of(a) - tb_of(b)).abs();
            if d != 1 {
                breaks += 1;
            }
        }
        native_seeks.insert(ordering_key(o), breaks);
    }

    // --- downsample into the sparse (sx, sy, tb) cube --------------------
    let space_side = 1u32 << space_bits; // e.g. 32
    let tb_count = (tb_span + 1) as u32;
    let time_bins = time_bins_req.min(tb_count).max(1);
    let scale = |v: u32, vmin: u32, vrange: u64, side: u32| -> u32 {
        if vrange == 0 {
            0
        } else {
            ((v - vmin) as u64 * (side as u64 - 1) / vrange) as u32
        }
    };

    let mut cube: BTreeMap<(u32, u32, u32), (u64, u64)> = BTreeMap::new();
    let mut seen_blobs: BTreeSet<(u32, u64)> = BTreeSet::new();
    let mut total_bytes = 0u64; // deduped on-disk footprint
    let mut total_features = 0u64;
    for e in &native {
        let sx = scale(e.x, x_min, x_range, space_side);
        let sy = scale(e.y, y_min, y_range, space_side);
        let stb = if tb_span == 0 {
            0
        } else {
            ((tb_of(e) - tb_min) as u64 * (time_bins as u64 - 1) / tb_span as u64) as u32
        };
        let cell = cube.entry((sx, sy, stb)).or_default();
        cell.0 += e.feature_count as u64;
        // Per-cell weight uses the compressed blob size (relative shape only);
        // the archive-level footprint dedups RLE-shared physical blobs.
        cell.1 += e.length as u64;
        total_features += e.feature_count as u64;
        if seen_blobs.insert((e.pack_id, e.offset)) {
            total_bytes += e.length as u64;
        }
    }

    // Flat [sx, sy, tb, count, bytes] tuples keep the JSON compact.
    let cells: Vec<serde_json::Value> = cube
        .iter()
        .map(|(&(sx, sy, tb), &(count, bytes))| serde_json::json!([sx, sy, tb, count, bytes]))
        .collect();

    let out = serde_json::json!({
        "id": id,
        "label": label,
        "zoom": max_zoom,
        "spaceBits": space_bits,
        "timeBins": time_bins,
        "tileCount": native.len(),
        "totalBytes": total_bytes,
        "featureCount": total_features,
        "bucketMs": bucket_ms,
        "timeRange": { "start": meta.time_range.start, "end": meta.time_range.end },
        "bounds": {
            "minLon": meta.bounds.min_lon, "minLat": meta.bounds.min_lat,
            "maxLon": meta.bounds.max_lon, "maxLat": meta.bounds.max_lat,
        },
        "spaceExtent": { "x0": x_min, "y0": y_min, "x1": x_max, "y1": y_max },
        "tbSpan": tb_span,
        "autoChoice": auto_key,
        "measuredChoice": measured_key,
        "nativeSeeks": native_seeks,
        "cube": cells,
    });

    println!("{}", serde_json::to_string(&out)?);
    eprintln!(
        "{id}: {} native tiles → {} sparse cells, auto={auto_key}, {:.1} MB on disk",
        native.len(),
        cube.len(),
        total_bytes as f64 / 1e6,
    );
    Ok(())
}

/// Stable JSON key for an ordering (matches the showcase walk names).
fn ordering_key(o: BlobOrdering) -> &'static str {
    match o {
        BlobOrdering::SpatialMajor => "spatial",
        BlobOrdering::Hilbert3 => "hilbert3",
        BlobOrdering::Morton3 => "morton3",
        BlobOrdering::TimeMajor => "time",
        BlobOrdering::Auto => "hilbert3",
    }
}
