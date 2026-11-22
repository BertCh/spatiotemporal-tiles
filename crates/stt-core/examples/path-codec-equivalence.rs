//! Whole-archive oracle for the path codec: every tile of a `--path-codec`
//! archive must decode to the same layers as the same build without it.
//!
//! The reference archive must carry exact vertex times
//! (`--vertex-time-precision 1`), because the codec is exact and the default
//! build is not. Geometry is compared by bit pattern, vertex times as absolute
//! milliseconds (the two archives store them in different forms), and every
//! other column by Arrow equality.
//!
//! Usage:
//!   cargo run --release -p stt-core --example path-codec-equivalence -- \
//!     <codec manifest.json> <reference manifest.json>

use std::collections::HashMap;

use arrow::array::{
    Array, FixedSizeListArray, Float64Array, Int32Array, Int64Array, ListArray, UInt16Array,
    UInt32Array,
};
use arrow::record_batch::RecordBatch;
use stt_core::pack::PackedReader;

/// Per-feature vertex counts and the flat coordinate leaf as raw bits.
fn geometry(b: &RecordBatch) -> Option<(Vec<usize>, Vec<u64>)> {
    let list = b
        .column_by_name("geometry")?
        .as_any()
        .downcast_ref::<ListArray>()?;
    let offs = list.value_offsets();
    let counts = offs.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let fsl = list
        .values()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()?;
    let dims = fsl.value_length() as usize;
    let (lo, hi) = (offs[0] as usize * dims, *offs.last()? as usize * dims);
    let leaf = fsl.values();
    let bits = if let Some(a) = leaf.as_any().downcast_ref::<Float64Array>() {
        a.values()[lo..hi].iter().map(|v| v.to_bits()).collect()
    } else {
        let a = leaf.as_any().downcast_ref::<Int32Array>()?;
        a.values()[lo..hi].iter().map(|&v| v as u64).collect()
    };
    Some((counts, bits))
}

/// Vertex times as absolute Unix ms, whatever tier the layer stored them in.
fn vertex_times(b: &RecordBatch) -> Option<Vec<i64>> {
    let list = b
        .column_by_name("vertex_time")?
        .as_any()
        .downcast_ref::<ListArray>()?;
    let offs = list.value_offsets();
    let (lo, hi) = (offs[0] as usize, *offs.last()? as usize);
    let vals = list.values();
    let schema = b.schema();
    let num = |k: &str| schema.metadata().get(k).and_then(|v| v.parse::<i64>().ok());
    let raw: Vec<i64> = if let Some(a) = vals.as_any().downcast_ref::<Int64Array>() {
        return Some(a.values()[lo..hi].to_vec());
    } else if let Some(a) = vals.as_any().downcast_ref::<UInt32Array>() {
        a.values()[lo..hi].iter().map(|&v| v as i64).collect()
    } else {
        let a = vals.as_any().downcast_ref::<UInt16Array>()?;
        a.values()[lo..hi].iter().map(|&v| v as i64).collect()
    };
    if let Some(step) = num("stt:vertex_time_feature_step_ms") {
        let st = b
            .column_by_name("start_time")?
            .as_any()
            .downcast_ref::<Int64Array>()?;
        let mut out = Vec::with_capacity(raw.len());
        for (f, w) in offs.windows(2).enumerate() {
            for i in w[0] as usize..w[1] as usize {
                out.push(st.value(f) + raw[i - lo] * step);
            }
        }
        return Some(out);
    }
    let (origin, step) = (
        num("stt:vertex_time_origin_ms")?,
        num("stt:vertex_time_step_ms")?,
    );
    Some(raw.iter().map(|v| origin + v * step).collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: path-codec-equivalence <codec manifest.json> <reference manifest.json>");
        std::process::exit(2);
    }
    let (codec, reference) = (PackedReader::open(&args[1])?, PackedReader::open(&args[2])?);
    let key = |e: &stt_core::directory::TileEntry| (e.zoom, e.x, e.y, e.time_start, e.variant_id);
    let by_key: HashMap<_, _> = reference.entries().iter().map(|e| (key(e), e)).collect();

    let (mut tiles, mut layers, mut vertices) = (0u64, 0u64, 0u64);
    // Every difference is counted by kind; the first few are kept verbatim.
    let mut kinds: std::collections::BTreeMap<String, u64> = Default::default();
    let mut problems = Vec::new();
    let mut note = |kind: &str, msg: String| {
        *kinds.entry(kind.to_string()).or_default() += 1;
        if problems.len() < 20 {
            problems.push(msg);
        }
    };
    if codec.entries().len() != reference.entries().len() {
        note(
            "tile count",
            format!(
                "tile counts differ: {} vs {}",
                codec.entries().len(),
                reference.entries().len()
            ),
        );
    }
    for e in codec.entries() {
        let Some(r) = by_key.get(&key(e)) else {
            note(
                "missing tile",
                format!("{:?}: no such tile in the reference", key(e)),
            );
            continue;
        };
        tiles += 1;
        let (a, b) = (codec.read_layers(e)?, reference.read_layers(r)?);
        if a.len() != b.len() {
            note(
                "layer count",
                format!("{:?}: {} layers vs {}", key(e), a.len(), b.len()),
            );
            continue;
        }
        for (la, lb) in a.iter().zip(&b) {
            layers += 1;
            let at = |what: &str| format!("{:?} layer '{}': {what}", key(e), la.name);
            if la.name != lb.name || la.batch.num_rows() != lb.batch.num_rows() {
                note("rows", at("name or row count differs"));
                continue;
            }
            match (geometry(&la.batch), geometry(&lb.batch)) {
                (Some(x), Some(y)) => {
                    vertices += x.0.iter().sum::<usize>() as u64;
                    if x != y {
                        note("geometry", at("geometry differs"));
                    }
                }
                (None, None) => {}
                _ => note("geometry", at("geometry present on one side only")),
            }
            if vertex_times(&la.batch) != vertex_times(&lb.batch) {
                note("vertex_time", at("vertex times differ"));
            }
            // By position, not by name: a layer can carry two columns called
            // `id` (the wire id and a source property of that name).
            let (sa, sb) = (la.batch.schema(), lb.batch.schema());
            if sa.fields().len() != sb.fields().len() {
                note("columns", at("column count differs"));
                continue;
            }
            for (i, f) in sa.fields().iter().enumerate() {
                if f.name() != sb.field(i).name() {
                    note("columns", at(&format!("column {i} is named differently")));
                } else if f.name() != "geometry"
                    && f.name() != "vertex_time"
                    && la.batch.column(i).as_ref() != lb.batch.column(i).as_ref()
                {
                    note(
                        "other column",
                        at(&format!("column {i} '{}' differs", f.name())),
                    );
                }
            }
        }
    }
    println!(
        "{tiles} tiles, {layers} layers, {vertices} vertices compared; capabilities {:?} vs {:?}",
        codec.capabilities(),
        reference.capabilities()
    );
    if problems.is_empty() {
        println!("IDENTICAL");
        Ok(())
    } else {
        println!("DIFFERENT {kinds:?}");
        for p in &problems {
            println!("DIFF {p}");
        }
        std::process::exit(1);
    }
}
