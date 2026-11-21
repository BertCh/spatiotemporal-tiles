//! Trajectory-codec experiment (branch `trajectory`).
//!
//! Question: do *intra-tile, per-feature* motion codecs beat what STT stores
//! for trajectory tiles today, after the per-blob zstd that the packed format
//! applies anyway? The 2026-07 `encoding-experiment` tested transforms on raw
//! f64 coordinates (NO-GO) and recorded two untested cells
//! (stt-packed-format-decisions §4 / §10.3): deltas over the quantized Int32
//! world grid, and a *predictor* (Trajic-style constant velocity). This
//! measures both, plus a time-aware predictor and a per-feature selector.
//!
//! Every variant is tile-local and feature-local: the first vertex of each
//! feature is absolute, so a tile still decodes standalone, no inter-tile or
//! inter-bucket chain exists, and identical geometry still encodes to
//! identical bytes on the world grid (dedup is preserved).
//!
//! Usage:
//!   cargo run --release -p stt-core --example trajectory-codec -- \
//!     <manifest.json> <quant_m> [sample_tiles=0 (all)] [zstd_level=3]
//!
//! `quant_m` is the world-grid step used to turn an f64 archive into grid
//! indices (an i32 archive is read on its own grid and `quant_m` is ignored).

use std::collections::BTreeMap;

use arrow::array::BinaryArray;
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float64Array, Int32Array, Int64Array, ListArray,
    UInt16Array, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use stt_core::pack::PackedReader;

/// Arrow IPC stream bytes of a batch (what a tile layer frame carries).
fn ipc(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut out, &batch.schema()).unwrap();
        w.write(batch).unwrap();
        w.finish().unwrap();
    }
    out
}

const M_PER_DEG: f64 = 111_320.0;

fn zlen(b: &[u8], level: i32) -> usize {
    if b.is_empty() {
        return 0;
    }
    zstd::bulk::compress(b, level)
        .map(|v| v.len())
        .unwrap_or(b.len())
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

fn varint_len(v: u64) -> usize {
    (64 - v.max(1).leading_zeros()).div_ceil(7) as usize
}

/// Sum of the raw buffers of an array, recursively (what zstd would see if
/// the column were compressed alone).
fn column_bytes(a: &dyn Array, out: &mut Vec<u8>) {
    let data = a.to_data();
    fn walk(d: &arrow::array::ArrayData, out: &mut Vec<u8>) {
        if let Some(nb) = d.nulls() {
            out.extend_from_slice(nb.buffer().as_slice());
        }
        for b in d.buffers() {
            out.extend_from_slice(b.as_slice());
        }
        for c in d.child_data() {
            walk(c, out);
        }
    }
    walk(&data, out);
}

/// One feature's geometry on the world grid + its stored vertex times.
struct Feat {
    xy: Vec<[i64; 2]>,
    t: Vec<i64>,
}

/// Geometry: List<FixedSizeList<2|3, leaf>>; returns per-feature vertex
/// ranges and the flat leaf as grid indices.
fn geometry(
    col: &ArrayRef,
    quant_m: f64,
) -> Option<(Vec<(usize, usize)>, Vec<[i64; 2]>, usize, bool)> {
    let list = col.as_any().downcast_ref::<ListArray>()?;
    let offs = list.value_offsets();
    let ranges: Vec<(usize, usize)> = offs
        .windows(2)
        .map(|w| (w[0] as usize, w[1] as usize))
        .collect();
    let fsl = list
        .values()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()?;
    let dims = fsl.value_length() as usize;
    let leaf = fsl.values();
    let base = offs[0] as usize;
    let mut pts = Vec::new();
    let is_f64;
    match leaf.data_type() {
        DataType::Float64 => {
            is_f64 = true;
            let v = leaf.as_any().downcast_ref::<Float64Array>()?.values();
            let s = quant_m / M_PER_DEG;
            for i in 0..fsl.len() {
                let x = ((v[i * dims] + 180.0) / s).round() as i64;
                let y = ((v[i * dims + 1] + 90.0) / s).round() as i64;
                pts.push([x, y]);
            }
        }
        DataType::Int32 => {
            is_f64 = false;
            let v = leaf.as_any().downcast_ref::<Int32Array>()?.values();
            for i in 0..fsl.len() {
                pts.push([v[i * dims] as i64, v[i * dims + 1] as i64]);
            }
        }
        _ => return None,
    }
    let ranges = ranges
        .into_iter()
        .map(|(a, b)| (a - base, b - base))
        .collect();
    Some((ranges, pts, dims, is_f64))
}

fn vertex_times(col: &ArrayRef) -> Option<Vec<i64>> {
    let list = col.as_any().downcast_ref::<ListArray>()?;
    let vals = list.values();
    let start = list.value_offsets()[0] as usize;
    let end = *list.value_offsets().last().unwrap() as usize;
    let v: Vec<i64> = if let Some(a) = vals.as_any().downcast_ref::<UInt16Array>() {
        a.values()[start..end].iter().map(|&x| x as i64).collect()
    } else if let Some(a) = vals.as_any().downcast_ref::<UInt32Array>() {
        a.values()[start..end].iter().map(|&x| x as i64).collect()
    } else if let Some(a) = vals.as_any().downcast_ref::<Int64Array>() {
        a.values()[start..end].to_vec()
    } else {
        return None;
    };
    Some(v)
}

/// Residuals of one feature's coordinate stream under a predictor.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Pred {
    /// p_i - p_{i-1}
    Delta,
    /// p_i - (2 p_{i-1} - p_{i-2})  (constant velocity per sample)
    ConstVel,
    /// p_i - (p_{i-1} + (p_{i-1}-p_{i-2}) * dt_i/dt_{i-1})  (constant velocity in time)
    TimeVel,
}

fn residuals(f: &Feat, p: Pred) -> Vec<i64> {
    let n = f.xy.len();
    let mut r = Vec::with_capacity(n * 2);
    for i in 0..n {
        for a in 0..2 {
            let cur = f.xy[i][a];
            let pred = if i == 0 {
                0
            } else if i == 1 || p == Pred::Delta {
                f.xy[i - 1][a]
            } else {
                let v = f.xy[i - 1][a] - f.xy[i - 2][a];
                match p {
                    Pred::ConstVel => f.xy[i - 1][a] + v,
                    Pred::TimeVel => {
                        let dt0 =
                            f.t.get(i - 1)
                                .zip(f.t.get(i - 2))
                                .map(|(a, b)| a - b)
                                .unwrap_or(0);
                        let dt1 =
                            f.t.get(i)
                                .zip(f.t.get(i - 1))
                                .map(|(a, b)| a - b)
                                .unwrap_or(0);
                        if dt0 > 0 {
                            f.xy[i - 1][a]
                                + ((v as f64) * (dt1 as f64) / (dt0 as f64)).round() as i64
                        } else {
                            f.xy[i - 1][a] + v
                        }
                    }
                    Pred::Delta => unreachable!(),
                }
            };
            r.push(cur - pred);
        }
    }
    r
}

fn varint_stream(vals: &[i64], out: &mut Vec<u8>) {
    for &v in vals {
        put_uvarint(out, zigzag(v));
    }
}

fn varint_cost(vals: &[i64]) -> usize {
    vals.iter().map(|&v| varint_len(zigzag(v))).sum()
}

/// Time residuals: first absolute, then delta (order 1) or delta-of-delta.
fn time_resid(t: &[i64], order: u8) -> Vec<i64> {
    let mut r = Vec::with_capacity(t.len());
    for i in 0..t.len() {
        let pred = if i == 0 {
            0
        } else if i == 1 || order == 1 {
            t[i - 1]
        } else {
            2 * t[i - 1] - t[i - 2]
        };
        r.push(t[i] - pred);
    }
    r
}

#[derive(Default)]
struct Acc {
    tiles: usize,
    layers: usize,
    features: usize,
    vertices: usize,
    blob: usize,
    /// column name -> zstd'd-alone bytes
    cols: BTreeMap<String, usize>,
    coord: BTreeMap<&'static str, usize>,
    time: BTreeMap<&'static str, usize>,
    pick: BTreeMap<String, usize>,
    per_zoom: BTreeMap<u8, (usize, usize, usize)>, // tiles, vertices, blob
    ipc_base: usize,
    ipc_codec: usize,
    leaf_f64: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: trajectory-codec <manifest.json> <quant_m> [sample_tiles=0] [zstd_level=3]"
        );
        std::process::exit(2);
    }
    let quant_m: f64 = args[2].parse()?;
    let sample: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let level: i32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let reader = PackedReader::open(&args[1])?;
    let entries = reader.entries();
    let step = if sample == 0 {
        1
    } else {
        (entries.len() / sample).max(1)
    };
    let mut acc = Acc::default();

    for e in entries.iter().step_by(step) {
        acc.tiles += 1;
        acc.blob += e.length as usize;
        let mut tile_verts = 0usize;
        for layer in reader.read_layers(e)? {
            acc.layers += 1;
            let b = &layer.batch;
            if acc.layers == 1 && std::env::var("SCHEMA").is_ok() {
                for f in b.schema().fields() {
                    println!(
                        "schema: {} {:?} {:?}",
                        f.name(),
                        f.data_type(),
                        f.metadata()
                    );
                }
                if let Some(c) = b.column_by_name("id") {
                    println!("id sample: {:?}", c.slice(0, c.len().min(8)));
                }
            }
            for (i, field) in b.schema().fields().iter().enumerate() {
                let mut buf = Vec::new();
                column_bytes(b.column(i).as_ref(), &mut buf);
                let key = if i == 0 && field.name() == "id" {
                    "id (wire u64)".to_string()
                } else {
                    field.name().clone()
                };
                *acc.cols.entry(key).or_default() += zlen(&buf, level);
            }
            let Some(gcol) = b.column_by_name("geometry") else {
                continue;
            };
            let Some((ranges, pts, dims, is_f64)) = geometry(gcol, quant_m) else {
                continue;
            };
            acc.leaf_f64 = is_f64;
            let vt = b.column_by_name("vertex_time").and_then(vertex_times);
            let feats: Vec<Feat> = ranges
                .iter()
                .map(|&(s, e)| Feat {
                    xy: pts[s..e].to_vec(),
                    t: vt.as_ref().map(|t| t[s..e].to_vec()).unwrap_or_default(),
                })
                .collect();
            acc.features += feats.len();
            acc.vertices += pts.len();
            tile_verts += pts.len();

            // --- coordinates on the grid, as stored today (i32 interleaved) ---
            let mut abs = Vec::with_capacity(pts.len() * 4 * dims);
            for p in &pts {
                abs.extend_from_slice(&(p[0] as i32).to_le_bytes());
                abs.extend_from_slice(&(p[1] as i32).to_le_bytes());
            }
            *acc.coord.entry("grid i32 abs").or_default() += zlen(&abs, level);

            // --- per-feature predictors ---
            let mut d1_i32 = Vec::new();
            let mut best_i32 = Vec::new();
            let mut best_i16 = Vec::new(); // i16 lanes, i16::MIN escape + i32
            let mut planar_i32: (Vec<u8>, Vec<u8>) = (Vec::new(), Vec::new());
            let mut streams: BTreeMap<Pred, Vec<u8>> = BTreeMap::new();
            let mut planar = Vec::new();
            let mut best = Vec::new();
            let mut feat_geom: Vec<Vec<u8>> = Vec::with_capacity(feats.len());
            let mut feat_vt: Vec<Vec<u8>> = Vec::with_capacity(feats.len());
            for f in &feats {
                let rd = residuals(f, Pred::Delta);
                for &v in &rd {
                    d1_i32.extend_from_slice(&(v as i32).to_le_bytes());
                }
                for p in [Pred::Delta, Pred::ConstVel, Pred::TimeVel] {
                    let r = if p == Pred::Delta {
                        rd.clone()
                    } else {
                        residuals(f, p)
                    };
                    varint_stream(&r, streams.entry(p).or_default());
                }
                // planar delta: all x then all y
                let xs: Vec<i64> = rd.iter().step_by(2).copied().collect();
                let ys: Vec<i64> = rd.iter().skip(1).step_by(2).copied().collect();
                varint_stream(&xs, &mut planar);
                varint_stream(&ys, &mut planar);
                // per-feature selector: cheapest varint cost, 1-byte tag
                let cand = [Pred::Delta, Pred::ConstVel, Pred::TimeVel]
                    .into_iter()
                    .map(|p| {
                        let r = residuals(f, p);
                        (varint_cost(&r), p, r)
                    })
                    .min_by_key(|c| c.0)
                    .unwrap();
                best.push(cand.1 as u8);
                varint_stream(&cand.2, &mut best);
                for (k, &v) in cand.2.iter().enumerate() {
                    best_i32.extend_from_slice(&(v as i32).to_le_bytes());
                    let lane = if k % 2 == 0 {
                        &mut planar_i32.0
                    } else {
                        &mut planar_i32.1
                    };
                    lane.extend_from_slice(&(v as i32).to_le_bytes());
                    if v > i16::MIN as i64 && v <= i16::MAX as i64 {
                        best_i16.extend_from_slice(&(v as i16).to_le_bytes());
                    } else {
                        best_i16.extend_from_slice(&i16::MIN.to_le_bytes());
                        best_i16.extend_from_slice(&(v as i32).to_le_bytes());
                    }
                }
                let mut fg = vec![cand.1 as u8];
                varint_stream(&cand.2, &mut fg);
                feat_geom.push(fg);
                let mut fv = Vec::new();
                varint_stream(&time_resid(&f.t, 1), &mut fv);
                feat_vt.push(fv);
                *acc.pick.entry(format!("{:?}", cand.1)).or_default() += 1;
            }
            // --- whole-layer check: rebuild the IPC frame with codec columns ---
            {
                acc.ipc_base += zlen(&ipc(b), level);
                let schema = b.schema();
                let mut fields = Vec::new();
                let mut cols: Vec<ArrayRef> = Vec::new();
                for (i, fld) in schema.fields().iter().enumerate() {
                    let n = fld.name();
                    if n == "geometry" {
                        fields.push(Field::new("geom_codec", DataType::Binary, false));
                        cols.push(Arc::new(BinaryArray::from_iter_values(feat_geom.iter())));
                    } else if n == "vertex_time" {
                        fields.push(Field::new("vt_codec", DataType::Binary, false));
                        cols.push(Arc::new(BinaryArray::from_iter_values(feat_vt.iter())));
                    } else {
                        fields.push(fld.as_ref().clone());
                        cols.push(b.column(i).clone());
                    }
                }
                let nb = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)?;
                acc.ipc_codec += zlen(&ipc(&nb), level);
            }
            *acc.coord.entry("delta i32").or_default() += zlen(&d1_i32, level);
            *acc.coord.entry("best-pred i32 (arrow)").or_default() += zlen(&best_i32, level);
            planar_i32.0.extend_from_slice(&planar_i32.1);
            *acc.coord.entry("best-pred i32 planar").or_default() += zlen(&planar_i32.0, level);
            *acc.coord.entry("best-pred i16+esc").or_default() += zlen(&best_i16, level);
            *acc.coord.entry("delta varint").or_default() += zlen(&streams[&Pred::Delta], level);
            *acc.coord.entry("delta varint planar").or_default() += zlen(&planar, level);
            *acc.coord.entry("const-vel varint").or_default() +=
                zlen(&streams[&Pred::ConstVel], level);
            *acc.coord.entry("time-vel varint").or_default() +=
                zlen(&streams[&Pred::TimeVel], level);
            *acc.coord.entry("per-feature best").or_default() += zlen(&best, level);

            // --- vertex times ---
            if let Some(vt) = &vt {
                for (name, ord) in [("delta varint", 1u8), ("delta-of-delta varint", 2u8)] {
                    let mut s = Vec::new();
                    for f in &feats {
                        varint_stream(&time_resid(&f.t, ord), &mut s);
                    }
                    *acc.time.entry(name).or_default() += zlen(&s, level);
                }
                let _ = vt;
            }
        }
        let z = acc.per_zoom.entry(e.zoom).or_default();
        z.0 += 1;
        z.1 += tile_verts;
        z.2 += e.length as usize;
    }

    let mb = |b: usize| b as f64 / 1e6;
    println!(
        "=== {}  (quant {} m, zstd -{}, {} tiles of {}, step {})",
        args[1],
        quant_m,
        level,
        acc.tiles,
        entries.len(),
        step
    );
    println!(
        "layers {}  features {}  vertices {}  blob {:.2} MB  ({:.2} B/vertex)",
        acc.layers,
        acc.features,
        acc.vertices,
        mb(acc.blob),
        acc.blob as f64 / acc.vertices.max(1) as f64
    );
    println!(
        "whole-layer IPC+zstd: as stored {:.2} MB -> codec {:.2} MB ({:+.1}%)",
        mb(acc.ipc_base),
        mb(acc.ipc_codec),
        100.0 * (acc.ipc_codec as f64 / acc.ipc_base as f64 - 1.0)
    );
    let col_sum: usize = acc.cols.values().sum();
    println!(
        "\n-- columns, each zstd'd alone (sum {:.2} MB = {:.0}% of blob)",
        mb(col_sum),
        100.0 * col_sum as f64 / acc.blob as f64
    );
    let mut cols: Vec<_> = acc.cols.iter().collect();
    cols.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in cols {
        println!(
            "  {:<24} {:>10.2} MB  {:>5.1}%",
            k,
            mb(*v),
            100.0 * *v as f64 / col_sum as f64
        );
    }
    let stored_geom = acc.cols.get("geometry").copied().unwrap_or(0);
    println!(
        "\n-- coordinates (stored geometry column: {:.2} MB, leaf {})",
        mb(stored_geom),
        if acc.leaf_f64 { "f64" } else { "i32 grid" }
    );
    for (k, v) in &acc.coord {
        println!(
            "  {:<22} {:>10.2} MB  {:>6.2} B/vtx  vs stored {:+6.1}%",
            k,
            mb(*v),
            *v as f64 / acc.vertices as f64,
            100.0 * (*v as f64 / stored_geom as f64 - 1.0)
        );
    }
    println!("  selector picks: {:?}", acc.pick);
    let stored_vt = acc.cols.get("vertex_time").copied().unwrap_or(0);
    println!(
        "\n-- vertex_time (stored: {:.2} MB, {:.2} B/vtx)",
        mb(stored_vt),
        stored_vt as f64 / acc.vertices as f64
    );
    for (k, v) in &acc.time {
        println!(
            "  {:<22} {:>10.2} MB  {:>6.2} B/vtx  vs stored {:+6.1}%",
            k,
            mb(*v),
            *v as f64 / acc.vertices as f64,
            100.0 * (*v as f64 / stored_vt.max(1) as f64 - 1.0)
        );
    }
    println!("\n-- per zoom: tiles, vertices, blob MB, B/vtx");
    for (z, (t, v, b)) in &acc.per_zoom {
        println!(
            "  z{:<3} {:>8} {:>11} {:>9.2} {:>7.2}",
            z,
            t,
            v,
            mb(*b),
            *b as f64 / (*v).max(1) as f64
        );
    }
    Ok(())
}
