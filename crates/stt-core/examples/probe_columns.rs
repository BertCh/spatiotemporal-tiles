//! THROWAWAY: decode the first non-empty tile of an archive and dump its
//! per-layer Arrow schema (column names + types) + a few sample values of any
//! string column. Used to confirm whether deployed tiles carry the per-feature
//! categorical properties the showcase legends key off (vessel_type / severity).
//!
//!   cargo run --release -p stt-core --example probe_columns -- <manifest-or-stt>

use arrow::array::Array;
use std::error::Error;
use stt_core::ArchiveReader;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: probe_columns <archive.stt | manifest.json>")?;
    let reader = ArchiveReader::open(&path)?;
    let entries = reader.entries().to_vec();
    println!("archive: {path}  entries={}", entries.len());

    let mut shown = 0;
    for e in &entries {
        if e.feature_count == 0 {
            continue;
        }
        let payload = match reader.read_payload(e) {
            Ok(b) => b,
            Err(err) => {
                println!("  read_payload failed: {err}");
                continue;
            }
        };
        let layers = stt_core::arrow_tile::decode_tile(&payload)?;
        println!(
            "tile {:?}  features={}  layers={}",
            e.tile_id(),
            e.feature_count,
            layers.len()
        );
        for layer in &layers {
            println!("  layer '{}'  rows={}", layer.name, layer.batch.num_rows());
            let schema = layer.batch.schema();
            for (i, field) in schema.fields().iter().enumerate() {
                print!("    - {} : {:?}", field.name(), field.data_type());
                // Dump a few distinct values for string/dictionary columns.
                let col = layer.batch.column(i);
                if let Some(arr) = col
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                {
                    use std::collections::BTreeSet;
                    let mut vals: BTreeSet<String> = BTreeSet::new();
                    for j in 0..arr.len().min(2000) {
                        if arr.is_valid(j) {
                            vals.insert(arr.value(j).to_string());
                        }
                        if vals.len() >= 12 {
                            break;
                        }
                    }
                    print!("   sample={:?}", vals);
                }
                println!();
            }
        }
        shown += 1;
        if shown >= 3 {
            break;
        }
    }
    if shown == 0 {
        println!("(no non-empty tiles found)");
    }
    Ok(())
}
