//! One-shot transcoder: read a legacy v2/v3 STT archive and rewrite it as v4.
//!
//! The tile payloads (GeoArrow IPC layer frames) are format-stable across
//! versions — only the container + index changed (Arrow-IPC index → compact
//! run-length directory, gzip/blake3 → zstd/crc32c). So this re-wraps the
//! existing decompressed tile payloads into a fresh v4 archive losslessly,
//! without needing the original data sources. The metadata time range is
//! recomputed from the tiles so the result validates clean.
//!
//! Usage: cargo run -p stt-core --example transcode -- <in.stt> <out.stt>

use arrow::array::{Array, Int64Array, UInt32Array, UInt64Array, UInt8Array};
use arrow::ipc::reader::StreamReader;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;
use stt_core::archive::ArchiveWriter;
use stt_core::compression;
use stt_core::metadata::Metadata;
use stt_core::tile::TileId;
use stt_core::types::{Compression, TimeRange};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: transcode <in.stt> <out.stt>");
        std::process::exit(2);
    }
    let (inp, outp) = (&args[1], &args[2]);

    let file = std::fs::File::open(inp)?;
    // SAFETY: read-only mapping of a file we don't mutate. Avoids loading
    // multi-GB archives fully into RAM.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;

    // --- Parse the 64-byte legacy header ---
    let mut cur = std::io::Cursor::new(&bytes[..64]);
    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    let version = match &magic {
        b"STT\x02" => 2u8,
        b"STT\x03" => 3,
        b"STT\x04" => {
            println!("{inp} is already v4 — nothing to do");
            return Ok(());
        }
        _ => return Err(format!("not an STT archive (magic {magic:?})").into()),
    };
    let _version_byte = cur.read_u8()?;
    let comp = match cur.read_u8()? {
        0 => Compression::None,
        1 => Compression::Gzip,
        2 => Compression::Zstd,
        o => return Err(format!("unknown compression code {o}").into()),
    };
    let index_offset = cur.read_u64::<LittleEndian>()? as usize;
    let index_length = cur.read_u64::<LittleEndian>()? as usize;
    let metadata_offset = cur.read_u64::<LittleEndian>()? as usize;
    let metadata_length = cur.read_u64::<LittleEndian>()? as usize;
    let dictionary_offset = cur.read_u64::<LittleEndian>()? as usize;
    let dictionary_length = cur.read_u64::<LittleEndian>()? as usize;
    let dict: Option<&[u8]> = if dictionary_length > 0 {
        Some(&bytes[dictionary_offset..dictionary_offset + dictionary_length])
    } else {
        None
    };

    // --- Decode the legacy Arrow-IPC index ---
    struct Entry {
        z: u8,
        x: u32,
        y: u32,
        ts: i64,
        te: i64,
        off: usize,
        len: usize,
        fc: u32,
        bucket: Option<u64>,
    }
    let index_bytes = &bytes[index_offset..index_offset + index_length];
    let reader = StreamReader::try_new(index_bytes, None)?;
    let mut entries: Vec<Entry> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let u8c = |n: &str| {
            batch.column_by_name(n).unwrap().as_any().downcast_ref::<UInt8Array>().unwrap()
        };
        let u32c = |n: &str| {
            batch.column_by_name(n).unwrap().as_any().downcast_ref::<UInt32Array>().unwrap()
        };
        let u64c = |n: &str| {
            batch.column_by_name(n).unwrap().as_any().downcast_ref::<UInt64Array>().unwrap()
        };
        let i64c = |n: &str| {
            batch.column_by_name(n).unwrap().as_any().downcast_ref::<Int64Array>().unwrap()
        };
        let zoom = u8c("zoom");
        let x = u32c("x");
        let y = u32c("y");
        let ts = i64c("time_start");
        let te = i64c("time_end");
        let off = u64c("offset");
        let len = u32c("length");
        let fc = u32c("feature_count");
        let bucket = batch
            .column_by_name("temporal_bucket_ms")
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
        for i in 0..batch.num_rows() {
            entries.push(Entry {
                z: zoom.value(i),
                x: x.value(i),
                y: y.value(i),
                ts: ts.value(i),
                te: te.value(i),
                off: off.value(i) as usize,
                len: len.value(i) as usize,
                fc: fc.value(i),
                bucket: bucket.and_then(|b| if b.is_null(i) { None } else { Some(b.value(i)) }),
            });
        }
    }

    // --- Preserve metadata, recompute the time range from the tiles ---
    let mut meta = Metadata::from_json_bytes(&bytes[metadata_offset..metadata_offset + metadata_length])?;

    // --- Write the v4 archive ---
    let mut writer = ArchiveWriter::create(outp, Compression::Zstd)?;
    let mut tmin = i64::MAX;
    let mut tmax = i64::MIN;
    for e in &entries {
        let blob = &bytes[e.off..e.off + e.len];
        let payload = if comp == Compression::Zstd {
            compression::decompress_zstd_with_dict(blob, dict)?
        } else {
            compression::decompress(blob, comp)?
        };
        writer.add_tile_with_bucket(
            &TileId::new(e.z, e.x, e.y, e.ts.max(0) as u64),
            e.ts,
            e.te,
            e.fc,
            e.bucket,
            &payload,
        )?;
        tmin = tmin.min(e.ts);
        tmax = tmax.max(e.te);
    }
    if tmin <= tmax {
        meta = meta.with_time_range(TimeRange::new(tmin.max(0) as u64, tmax.max(0) as u64));
    }
    writer.finalize(&meta)?;

    println!(
        "transcoded {} v{} -> v4: {} tiles, time [{}, {}]",
        inp, version, entries.len(), tmin, tmax
    );
    Ok(())
}
