//! Per-COLUMN size attribution for a packed point archive at one zoom level.
//!
//! `packed-stats` tells you per-zoom blob totals; this tells you *which column*
//! the bytes live in. For each tile at the chosen zoom it decodes the Arrow
//! batch, then for every column re-encodes that column alone (IPC + zstd-19) to
//! attribute a fair compressed-byte share, and also records the uncompressed
//! Arrow memory size. Reports bytes-per-point per column so you can see exactly
//! where a LiDAR point's wire cost goes.
//!
//! Usage:
//!   cargo run --release -p stt-core --example point_column_stats -- \
//!     <manifest.json> [--zoom <z>]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::Schema;
use arrow::ipc::writer::StreamWriter;
use stt_core::compression::compress_zstd_with_dict_level;
use stt_core::pack::PackedReader;

#[derive(Default, Clone)]
struct Col {
    unc: u64,
    comp: u64,
    dtype: String,
}

fn ipc_zstd(batch: &RecordBatch) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut buf, &batch.schema()).unwrap();
        w.write(batch).unwrap();
        w.finish().unwrap();
    }
    compress_zstd_with_dict_level(&buf, None, 19).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: point_column_stats <manifest.json> [--zoom <z>]");
        std::process::exit(2);
    }
    let manifest = &args[0];
    let mut want_zoom: Option<u8> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--zoom" {
            want_zoom = args.get(i + 1).and_then(|s| s.parse().ok());
            i += 2;
        } else {
            i += 1;
        }
    }

    let reader = PackedReader::open(manifest)?;
    let entries = reader.entries();
    let zoom = want_zoom.unwrap_or_else(|| entries.iter().map(|e| e.zoom).min().unwrap_or(0));

    let mut cols: BTreeMap<String, Col> = BTreeMap::new();
    let mut rows: u64 = 0;
    let mut tile_comp_actual: u64 = 0; // sum of on-disk blob lengths (whole tile)
    let mut tile_comp_recompressed: u64 = 0; // whole-tile re-encode for reconciliation
    let mut n_tiles: u64 = 0;

    for e in entries.iter().filter(|e| e.zoom == zoom) {
        let layers = reader.read_layers(e)?;
        tile_comp_actual += e.length as u64;
        n_tiles += 1;
        for layer in &layers {
            let batch = &layer.batch;
            rows += batch.num_rows() as u64;
            tile_comp_recompressed += ipc_zstd(batch).len() as u64;
            let schema = batch.schema();
            for (idx, field) in schema.fields().iter().enumerate() {
                let array = batch.column(idx).clone();
                let one = RecordBatch::try_new(
                    Arc::new(Schema::new(vec![field.as_ref().clone()])),
                    vec![array.clone()],
                )?;
                let c = cols.entry(field.name().clone()).or_default();
                c.unc += array.get_array_memory_size() as u64;
                c.comp += ipc_zstd(&one).len() as u64;
                c.dtype = format!("{:?}", field.data_type());
            }
        }
    }

    let r = rows.max(1) as f64;
    println!("archive: {manifest}");
    println!(
        "zoom {zoom}: {n_tiles} tiles, {rows} points  (~{:.0} pts/tile)",
        rows as f64 / n_tiles.max(1) as f64
    );
    println!(
        "whole-tile compressed: on-disk {:.2} MB ({:.1} B/pt)   recompressed@19 {:.2} MB ({:.1} B/pt)",
        tile_comp_actual as f64 / 1e6,
        tile_comp_actual as f64 / r,
        tile_comp_recompressed as f64 / 1e6,
        tile_comp_recompressed as f64 / r,
    );
    println!();
    println!(
        "{:<22} {:<28} {:>10} {:>10} {:>9}",
        "column", "dtype", "unc B/pt", "comp B/pt", "share%"
    );
    // Sum of per-column recompressed (will exceed whole-tile due to lost
    // cross-column sharing + per-column IPC framing — used only for shares).
    let total_comp: u64 = cols.values().map(|c| c.comp).sum();
    let mut rowsv: Vec<(&String, &Col)> = cols.iter().collect();
    rowsv.sort_by(|a, b| b.1.comp.cmp(&a.1.comp));
    for (name, c) in rowsv {
        let dt = if c.dtype.len() > 27 {
            format!("{}…", &c.dtype[..26])
        } else {
            c.dtype.clone()
        };
        println!(
            "{:<22} {:<28} {:>10.2} {:>10.2} {:>8.1}%",
            name,
            dt,
            c.unc as f64 / r,
            c.comp as f64 / r,
            100.0 * c.comp as f64 / total_comp.max(1) as f64,
        );
    }
    Ok(())
}
