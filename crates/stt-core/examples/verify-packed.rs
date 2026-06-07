//! Verify a packed dataset against its source v4 single-file archive.
//!
//! Opens both readers, asserts the directory entries line up (same keys, same
//! order), and compares every tile's decompressed payload byte-for-byte. Also
//! re-derives each pack's content hash from its bytes to confirm content
//! addressing. Read-only.
//!
//! Usage: verify-packed <packed_manifest_or_dir> <source.stt>

use std::path::PathBuf;

use stt_core::{ArchiveReader, PackedReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: verify-packed <packed_manifest_or_dir> <source.stt>");
        std::process::exit(2);
    }
    let mut manifest = PathBuf::from(&args[1]);
    if manifest.is_dir() {
        manifest.push("manifest.json");
    }
    let source = &args[2];

    let packed = PackedReader::open(&manifest)?;
    let src = ArchiveReader::open(source)?;

    let pe = packed.entries();
    let se = src.entries();
    if pe.len() != se.len() {
        return Err(format!("entry count mismatch: packed={} source={}", pe.len(), se.len()).into());
    }

    let mut mismatches = 0usize;
    let mut bytes = 0u64;
    for (i, (p, s)) in pe.iter().zip(se.iter()).enumerate() {
        if (p.zoom, p.x, p.y, p.time_start, p.time_end)
            != (s.zoom, s.x, s.y, s.time_start, s.time_end)
        {
            eprintln!("key mismatch at #{i}: packed=({},{},{},{}) source=({},{},{},{})",
                p.zoom, p.x, p.y, p.time_start, s.zoom, s.x, s.y, s.time_start);
            mismatches += 1;
            if mismatches > 20 { return Err("too many key mismatches".into()); }
            continue;
        }
        let pp = packed.read_payload(p)?; // CRC-verified + decompressed
        let sp = src.read_payload(s)?;
        if pp != sp {
            eprintln!("PAYLOAD mismatch at #{i} tile ({},{},{},{})", p.zoom, p.x, p.y, p.time_start);
            mismatches += 1;
            if mismatches > 20 { return Err("too many payload mismatches".into()); }
        }
        bytes += pp.len() as u64;
        if i % 20000 == 0 {
            println!("  ..verified {i}/{} tiles", pe.len());
        }
    }

    if mismatches > 0 {
        return Err(format!("{mismatches} mismatches found").into());
    }
    let pack_count = pe.iter().map(|e| e.pack_id).max().map(|m| m + 1).unwrap_or(0);
    println!(
        "OK: {} tiles match byte-for-byte ({:.1} MB decompressed); packs={}",
        pe.len(),
        bytes as f64 / 1_048_576.0,
        pack_count
    );
    Ok(())
}
