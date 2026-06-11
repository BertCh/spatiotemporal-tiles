//! Lightweight column-encoding experiment (roadmap stt-packed §4).
//!
//! Measures, on real tile payloads, whether MLT-style lightweight encodings
//! (delta + varint / delta + bit-packing) beat the current "raw column bytes
//! + per-blob zstd" for STT's hot columns:
//!
//!   - `vertex_time` (List<UInt16> quantized deltas, or List<Int64> exact)
//!   - geometry coordinates (Float64 xy, tested with byte-shuffle + xor-delta)
//!   - `start_time` / `end_time` (Int64 per feature, near-sorted)
//!
//! Every variant is zstd'd afterwards (the packed format is zstd-per-blob),
//! so the question is strictly "does a cheap transform make zstd's job
//! enough better to justify a decoder-side pass?".
//!
//! Usage: cargo run --release -p stt-core --example encoding-experiment -- \
//!          <manifest.json> [sample_tiles=300]

use arrow::array::{Array, FixedSizeListArray, Float64Array, Int64Array, ListArray, UInt16Array};
use arrow::datatypes::DataType;
use stt_core::pack::PackedReader;

fn zlen(b: &[u8]) -> usize {
    zstd::bulk::compress(b, 9).map(|v| v.len()).unwrap_or(b.len())
}

fn le_bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_bytes_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_bytes_f64(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}
fn put_uvarint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

/// Delta + zigzag + LEB128 varint.
fn delta_varint(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len());
    let mut prev = 0i64;
    for &v in values {
        put_uvarint(&mut out, zigzag(v.wrapping_sub(prev)));
        prev = v;
    }
    out
}

/// Delta + per-block (128) bit-packing: 1-byte width header per block, then
/// LSB-first packed zigzag deltas — a scalar stand-in for FastPFOR.
fn delta_bitpack(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut prev = 0i64;
    for block in values.chunks(128) {
        let deltas: Vec<u64> = block
            .iter()
            .map(|&v| {
                let d = zigzag(v.wrapping_sub(prev));
                prev = v;
                d
            })
            .collect();
        let bits = deltas.iter().map(|d| 64 - d.leading_zeros()).max().unwrap_or(1).max(1) as u64;
        out.push(bits as u8);
        let mut acc: u128 = 0;
        let mut nbits = 0u32;
        for d in deltas {
            acc |= (d as u128) << nbits;
            nbits += bits as u32;
            while nbits >= 8 {
                out.push((acc & 0xff) as u8);
                acc >>= 8;
                nbits -= 8;
            }
        }
        if nbits > 0 {
            out.push((acc & 0xff) as u8);
        }
    }
    out
}

/// Byte-shuffle (transpose) for f64: all byte-0s, then all byte-1s, ... —
/// the classic HDF5/Blosc trick that groups exponent bytes for zstd.
fn shuffle_f64(values: &[f64]) -> Vec<u8> {
    let raw: Vec<[u8; 8]> = values.iter().map(|x| x.to_le_bytes()).collect();
    let mut out = Vec::with_capacity(values.len() * 8);
    for lane in 0..8 {
        for b in &raw {
            out.push(b[lane]);
        }
    }
    out
}

/// XOR with previous value's bits (Gorilla-style), then byte-shuffle.
fn xor_shuffle_f64(values: &[f64]) -> Vec<u8> {
    let mut prev = 0u64;
    let xored: Vec<f64> = values
        .iter()
        .map(|x| {
            let bits = x.to_bits();
            let out = f64::from_bits(bits ^ prev);
            prev = bits;
            out
        })
        .collect();
    shuffle_f64(&xored)
}

#[derive(Default)]
struct Col {
    raw: usize,
    raw_z: usize,
    dv_z: usize,
    bp_z: usize,
    n: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: encoding-experiment <manifest.json> [sample_tiles=300]");
        std::process::exit(2);
    }
    let sample: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(300);
    let reader = PackedReader::open(&args[1])?;
    let entries = reader.entries();
    let step = (entries.len() / sample.max(1)).max(1);

    let mut vt = Col::default(); // vertex_time (u16 or i64, flattened)
    let mut ft = Col::default(); // start_time+end_time
    let mut blob_total = 0usize;
    let mut coords_raw = 0usize;
    let (mut coords_raw_z, mut coords_shuf_z, mut coords_xor_z) = (0usize, 0usize, 0usize);
    let mut tiles = 0usize;

    for e in entries.iter().step_by(step).take(sample) {
        blob_total += e.length as usize;
        tiles += 1;
        for layer in reader.read_layers(e)? {
            let batch = layer.batch;
            // --- vertex_time ---
            if let Some(col) = batch.column_by_name("vertex_time") {
                if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
                    let vals = list.values();
                    let ints: Vec<i64> = if let Some(a) = vals.as_any().downcast_ref::<UInt16Array>() {
                        let v: Vec<u16> = a.values().to_vec();
                        vt.raw += v.len() * 2;
                        vt.raw_z += zlen(&le_bytes_u16(&v));
                        v.iter().map(|&x| x as i64).collect()
                    } else if let Some(a) = vals.as_any().downcast_ref::<Int64Array>() {
                        let v: Vec<i64> = a.values().to_vec();
                        vt.raw += v.len() * 8;
                        vt.raw_z += zlen(&le_bytes_i64(&v));
                        v
                    } else {
                        continue;
                    };
                    vt.n += ints.len();
                    vt.dv_z += zlen(&delta_varint(&ints));
                    vt.bp_z += zlen(&delta_bitpack(&ints));
                }
            }
            // --- feature start/end times ---
            for name in ["start_time", "end_time"] {
                if let Some(col) = batch.column_by_name(name) {
                    if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                        let v: Vec<i64> = a.values().to_vec();
                        ft.raw += v.len() * 8;
                        ft.raw_z += zlen(&le_bytes_i64(&v));
                        ft.dv_z += zlen(&delta_varint(&v));
                        ft.bp_z += zlen(&delta_bitpack(&v));
                        ft.n += v.len();
                    }
                }
            }
            // --- geometry coords: find the flat Float64 leaf under geometry ---
            if let Some(col) = batch.column_by_name("geometry") {
                let flat: Option<Vec<f64>> = extract_f64_leaf(col.as_ref());
                if let Some(v) = flat {
                    coords_raw += v.len() * 8;
                    coords_raw_z += zlen(&le_bytes_f64(&v));
                    coords_shuf_z += zlen(&shuffle_f64(&v));
                    coords_xor_z += zlen(&xor_shuffle_f64(&v));
                }
            }
        }
    }

    println!("=== {} ({} tiles sampled, blob bytes {} = at-rest zstd frames)", args[1], tiles, blob_total);
    let pct = |a: usize, b: usize| -> f64 {
        if b == 0 { 0.0 } else { (a as f64 / b as f64 - 1.0) * 100.0 }
    };
    println!(
        "vertex_time   ({:>9} vals): raw {:>9} | raw+zstd {:>9} | delta-varint+zstd {:>9} ({:+.1}%) | delta-bitpack+zstd {:>9} ({:+.1}%)",
        vt.n, vt.raw, vt.raw_z, vt.dv_z, pct(vt.dv_z, vt.raw_z), vt.bp_z, pct(vt.bp_z, vt.raw_z)
    );
    println!(
        "feature times ({:>9} vals): raw {:>9} | raw+zstd {:>9} | delta-varint+zstd {:>9} ({:+.1}%) | delta-bitpack+zstd {:>9} ({:+.1}%)",
        ft.n, ft.raw, ft.raw_z, ft.dv_z, pct(ft.dv_z, ft.raw_z), ft.bp_z, pct(ft.bp_z, ft.raw_z)
    );
    println!(
        "coords f64    ({:>9} B raw): raw+zstd {:>9} | shuffle+zstd {:>9} ({:+.1}%) | xor+shuffle+zstd {:>9} ({:+.1}%)",
        coords_raw, coords_raw_z, coords_shuf_z, pct(coords_shuf_z, coords_raw_z), coords_xor_z, pct(coords_xor_z, coords_raw_z)
    );
    Ok(())
}

/// Recursively find the first Float64 leaf array (the xy coords) under a
/// (possibly nested List/FixedSizeList) geometry column.
fn extract_f64_leaf(arr: &dyn Array) -> Option<Vec<f64>> {
    match arr.data_type() {
        DataType::Float64 => arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| a.values().to_vec()),
        DataType::FixedSizeList(_, _) => {
            let l = arr.as_any().downcast_ref::<FixedSizeListArray>()?;
            extract_f64_leaf(l.values().as_ref())
        }
        DataType::List(_) => {
            let l = arr.as_any().downcast_ref::<ListArray>()?;
            extract_f64_leaf(l.values().as_ref())
        }
        _ => None,
    }
}
