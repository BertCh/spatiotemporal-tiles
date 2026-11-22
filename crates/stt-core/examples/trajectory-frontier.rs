//! Trajectory-codec experiment, round 2 (branch `trajectory`).
//!
//! `trajectory-codec` (round 1) priced the codec column-by-column and against
//! a re-serialised IPC frame. This harness prices it on the **real tile
//! payload**: it reads each tile's decompressed layer frame, cuts the geometry
//! and `vertex_time` buffers out of the CORE_BATCH body, appends the codec
//! bytes, and re-zstds the whole frame. Everything the codec does not touch
//! (frame header, TILE_META, ids, compact times, PROPS) stays byte-identical,
//! and the record-batch flatbuffer keeps describing the removed buffers, so
//! the codec side is charged slightly too much, never too little.
//!
//! Meant for an **f64 archive built with `--vertex-time-precision 1`** (exact
//! per-vertex times). From that one archive it emulates the other stored
//! forms — the default build's lossy u16 vertex times, and
//! `--quantize-coords`' Int32 world grid — so every variant shares one set of
//! tiles. An Int32 archive is accepted too (grid variants only).
//!
//! Questions it answers:
//!   A. bit-exact f64: decimal-grid detection (ALP-style) + predictor +
//!      exceptions for clip vertices, against the stored f64 leaf;
//!   B. exact `vertex_time` via an epoch-aligned quotient/remainder split,
//!      against today's default (lossy u16 offsets) and exact u32;
//!   C. the round-1 grid codec, at frame level, with layout variants
//!      (side section vs Arrow Binary column; per-feature vs per-tile
//!      predictor; first vertex absolute vs relative to the previous feature);
//!   D. what the per-segment wire id costs once the geometry shrinks;
//!   E. what a shared zstd dictionary would add on small tiles;
//!   F. how far varint + zstd sits from an order-0 entropy bound.
//!
//! Usage:
//!   cargo run --release -p stt-core --example trajectory-frontier -- \
//!     <manifest.json> <quant_m> [every_nth_tile=1] [zstd_level=3]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeListArray, Float64Array, Int32Array, Int64Array, ListArray, UInt16Array,
    UInt32Array,
};
use arrow::datatypes::DataType;
use rayon::prelude::*;
use stt_core::directory::TileEntry;
use stt_core::pack::PackedReader;

const M_PER_DEG: f64 = 111_320.0;
const VT_ORIGIN: &str = "stt:vertex_time_origin_ms";
const VT_STEP: &str = "stt:vertex_time_step_ms";
const VT_FSTEP: &str = "stt:vertex_time_feature_step_ms";
const DICT_BYTES: usize = 64 * 1024;
const DICT_KEYS: [&str; 4] = ["A0 stored", "A3 exact", "Q0 grid stored", "Q2 grid codec"];

// ---------------------------------------------------------------- varints --

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

fn put_svarint(buf: &mut Vec<u8>, v: i64) {
    put_uvarint(buf, zigzag(v));
}

fn varint_len(v: u64) -> usize {
    (64 - v.max(1).leading_zeros()).div_ceil(7) as usize
}

fn zlen(b: &[u8], level: i32) -> u64 {
    if b.is_empty() {
        return 0;
    }
    zstd::bulk::compress(b, level)
        .map(|v| v.len())
        .unwrap_or(b.len()) as u64
}

// ------------------------------------------------------------ frame parse --

/// `(tag, start, len)` of every section of every layer of a v2 layer frame.
fn parse_frame(p: &[u8]) -> Option<Vec<(String, Vec<(u8, usize, usize)>)>> {
    let rd16 = |o: usize| u16::from_le_bytes([p[o], p[o + 1]]);
    if p.len() < 6 || rd16(0) != 0xFFFF || p[2] != 2 {
        return None;
    }
    let n = rd16(4) as usize;
    let mut pos = 6;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let nl = rd16(pos) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&p[pos..pos + nl]).into_owned();
        pos += nl;
        for _ in 0..2 {
            let kind = p[pos];
            pos += 1;
            if kind == 1 {
                pos += 16;
            }
        }
        let sc = p[pos] as usize;
        pos += 1;
        let mut toc = Vec::with_capacity(sc);
        for _ in 0..sc {
            let len = u32::from_le_bytes(p[pos + 1..pos + 5].try_into().ok()?) as usize;
            toc.push((p[pos], len));
            pos += 5;
        }
        pos = (pos + 7) & !7;
        let mut secs = Vec::with_capacity(sc);
        for (tag, len) in toc {
            secs.push((tag, pos, len));
            pos = (pos + len + 7) & !7;
        }
        out.push((name, secs));
    }
    Some(out)
}

/// Absolute `[start, end)` of each body buffer of the record batch in a
/// CORE_BATCH section (`end` runs to the next buffer, so padding goes with it).
fn map_core(p: &[u8], s: usize, len: usize) -> Option<Vec<(usize, usize, usize)>> {
    let sec = &p[s..s + len];
    if sec.len() < 8 || sec[..4] != [0xFF; 4] {
        return None;
    }
    let ml = i32::from_le_bytes(sec[4..8].try_into().ok()?) as usize;
    let msg = arrow::ipc::root_as_message(&sec[8..8 + ml]).ok()?;
    let rb = msg.header_as_record_batch()?;
    let body = s + 8 + ml;
    let body_end = body + msg.bodyLength() as usize;
    let raw: Vec<(usize, usize)> = rb
        .buffers()?
        .iter()
        .map(|b| (body + b.offset() as usize, b.length() as usize))
        .collect();
    Some(
        raw.iter()
            .enumerate()
            .map(|(i, &(st, ln))| (st, ln, raw.get(i + 1).map(|n| n.0).unwrap_or(body_end)))
            .collect(),
    )
}

fn nbuf(dt: &DataType) -> Option<usize> {
    Some(match dt {
        DataType::List(f) => 2 + nbuf(f.data_type())?,
        DataType::FixedSizeList(f, _) => 1 + nbuf(f.data_type())?,
        DataType::Utf8 | DataType::Binary => 3,
        DataType::Dictionary(_, _) => 2,
        t if t.is_primitive() => 2,
        _ => return None,
    })
}

/// A byte-range edit on the payload: replace `[start, end)` with `bytes`.
type Edit = (usize, usize, Vec<u8>);

fn apply(p: &[u8], edits: &mut [Edit], tail: &[u8]) -> Vec<u8> {
    edits.sort_by_key(|e| e.0);
    let mut out = Vec::with_capacity(p.len());
    let mut pos = 0;
    for (s, e, b) in edits.iter() {
        out.extend_from_slice(&p[pos..*s]);
        out.extend_from_slice(b);
        pos = *e;
    }
    out.extend_from_slice(&p[pos..]);
    out.extend_from_slice(tail);
    out
}

// --------------------------------------------------------------- decoding --

struct Layer {
    /// per-feature vertex ranges into the flat arrays
    ranges: Vec<(usize, usize)>,
    /// f64 lon/lat (empty for an Int32 archive)
    f: Vec<[f64; 2]>,
    /// world-grid indices (stored, or `f` on the `quant_m` grid)
    g: Vec<[i64; 2]>,
    /// absolute vertex times in ms when `exact_t`, else the stored integers
    t: Vec<i64>,
    exact_t: bool,
    leaf_w: usize,
    vt_w: usize,
}

fn read_layer(b: &arrow::record_batch::RecordBatch, quant_m: f64) -> Option<Layer> {
    let gcol = b.column_by_name("geometry")?;
    let list = gcol.as_any().downcast_ref::<ListArray>()?;
    let offs = list.value_offsets();
    let base = offs[0] as usize;
    let ranges: Vec<(usize, usize)> = offs
        .windows(2)
        .map(|w| (w[0] as usize - base, w[1] as usize - base))
        .collect();
    let fsl = list
        .values()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()?;
    let dims = fsl.value_length() as usize;
    let leaf = fsl.values();
    let nv = ranges.last().map(|r| r.1).unwrap_or(0);
    let (mut f, mut g) = (Vec::new(), Vec::with_capacity(nv));
    let leaf_w;
    match leaf.data_type() {
        DataType::Float64 => {
            leaf_w = 8;
            let v = leaf.as_any().downcast_ref::<Float64Array>()?.values();
            let s = quant_m / M_PER_DEG;
            for i in base..base + nv {
                let (x, y) = (v[i * dims], v[i * dims + 1]);
                f.push([x, y]);
                // The encoder's own rounding (`QuantAffine::quantize`).
                g.push([
                    ((x + 180.0) / s).round() as i64,
                    ((y + 90.0) / s).round() as i64,
                ]);
            }
        }
        DataType::Int32 => {
            leaf_w = 4;
            let v = leaf.as_any().downcast_ref::<Int32Array>()?.values();
            for i in base..base + nv {
                g.push([v[i * dims] as i64, v[i * dims + 1] as i64]);
            }
        }
        _ => return None,
    }

    let vcol = b.column_by_name("vertex_time")?;
    let vlist = vcol.as_any().downcast_ref::<ListArray>()?;
    let vbase = vlist.value_offsets()[0] as usize;
    let vals = vlist.values();
    let (raw, vt_w): (Vec<i64>, usize) =
        if let Some(a) = vals.as_any().downcast_ref::<UInt16Array>() {
            (
                a.values()[vbase..vbase + nv]
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
                2,
            )
        } else if let Some(a) = vals.as_any().downcast_ref::<UInt32Array>() {
            (
                a.values()[vbase..vbase + nv]
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
                4,
            )
        } else if let Some(a) = vals.as_any().downcast_ref::<Int64Array>() {
            (a.values()[vbase..vbase + nv].to_vec(), 8)
        } else {
            return None;
        };
    let schema = b.schema();
    let md = schema.metadata();
    let num = |k: &str| md.get(k).and_then(|v| v.parse::<i64>().ok());
    let (t, exact_t) = if vt_w == 8 {
        (raw, true)
    } else if let Some(step) = num(VT_FSTEP) {
        let st = b
            .column_by_name("start_time")?
            .as_any()
            .downcast_ref::<Int64Array>()?
            .clone();
        let mut t = raw;
        for (fi, &(s, e)) in ranges.iter().enumerate() {
            for v in &mut t[s..e] {
                *v = st.value(fi) + *v * step;
            }
        }
        (t, step == 1)
    } else if let (Some(o), Some(step)) = (num(VT_ORIGIN), num(VT_STEP)) {
        if step == 1 {
            (raw.iter().map(|v| o + v).collect(), true)
        } else {
            (raw, false)
        }
    } else {
        (raw, false)
    };
    Some(Layer {
        ranges,
        f,
        g,
        t,
        exact_t,
        leaf_w,
        vt_w,
    })
}

// ------------------------------------------------------------- predictors --

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pred {
    Delta = 0,
    ConstVel = 1,
    TimeVel = 2,
}
const PREDS: [Pred; 3] = [Pred::Delta, Pred::ConstVel, Pred::TimeVel];

/// `floor(x + 0.5)`: the one rounding a Rust and a JS decoder agree on
/// (`f64::round` and `Math.round` differ on negative halves).
fn rnd(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

fn predict(xy: &[[i64; 2]], t: &[i64], i: usize, a: usize, p: Pred, first: i64) -> i64 {
    if i == 0 {
        return first;
    }
    if i == 1 || p == Pred::Delta {
        return xy[i - 1][a];
    }
    let v = xy[i - 1][a] - xy[i - 2][a];
    match p {
        Pred::ConstVel => xy[i - 1][a] + v,
        _ => {
            let (dt0, dt1) = (t[i - 1] - t[i - 2], t[i] - t[i - 1]);
            if dt0 > 0 {
                xy[i - 1][a] + rnd(v as f64 * dt1 as f64 / dt0 as f64)
            } else {
                xy[i - 1][a] + v
            }
        }
    }
}

fn residuals(xy: &[[i64; 2]], t: &[i64], p: Pred, first: [i64; 2]) -> Vec<i64> {
    let mut r = Vec::with_capacity(xy.len() * 2);
    for i in 0..xy.len() {
        for a in 0..2 {
            r.push(xy[i][a] - predict(xy, t, i, a, p, first[a]));
        }
    }
    r
}

fn unresiduals(r: &[i64], t: &[i64], p: Pred, first: [i64; 2]) -> Vec<[i64; 2]> {
    let n = r.len() / 2;
    let mut xy = vec![[0i64; 2]; n];
    for i in 0..n {
        for a in 0..2 {
            xy[i][a] = r[i * 2 + a] + predict(&xy, t, i, a, p, first[a]);
        }
    }
    xy
}

fn cost(r: &[i64]) -> usize {
    r.iter().map(|&v| varint_len(zigzag(v))).sum()
}

/// One layer's geometry under a predictor policy.
struct Geom {
    /// planar: all tags, then all residuals (feature-sequential)
    tags: Vec<u8>,
    res: Vec<u8>,
    /// per-feature blobs (tag + residuals), for the Arrow Binary layout
    per_feat: Vec<Vec<u8>>,
    /// residual values, for round-trip checks and the entropy bound
    vals: Vec<(Pred, Vec<i64>)>,
}

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    /// cheapest of three per feature, 1-byte tag each
    PerFeature,
    /// cheapest of three for the whole layer, one tag
    PerLayer,
    DeltaOnly,
}

fn encode_geom(l: &Layer, xy: &[[i64; 2]], policy: Policy, rel_prev: bool) -> Geom {
    let feats: Vec<(&[[i64; 2]], &[i64])> = l
        .ranges
        .iter()
        .map(|&(s, e)| (&xy[s..e], &l.t[s..e]))
        .collect();
    let firsts: Vec<[i64; 2]> = feats
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if rel_prev && i > 0 && !feats[i - 1].0.is_empty() {
                feats[i - 1].0[0]
            } else {
                [0, 0]
            }
        })
        .collect();
    let layer_pred = match policy {
        Policy::PerLayer => PREDS
            .into_iter()
            .min_by_key(|&p| {
                feats
                    .iter()
                    .zip(&firsts)
                    .map(|(f, &fr)| cost(&residuals(f.0, f.1, p, fr)))
                    .sum::<usize>()
            })
            .unwrap(),
        _ => Pred::Delta,
    };
    let mut g = Geom {
        tags: Vec::new(),
        res: Vec::new(),
        per_feat: Vec::new(),
        vals: Vec::new(),
    };
    if policy == Policy::PerLayer {
        g.tags.push(layer_pred as u8);
    }
    for (f, &fr) in feats.iter().zip(&firsts) {
        let (p, r) = match policy {
            Policy::PerFeature => PREDS
                .into_iter()
                .map(|p| (p, residuals(f.0, f.1, p, fr)))
                .min_by_key(|c| cost(&c.1))
                .unwrap(),
            _ => (layer_pred, residuals(f.0, f.1, layer_pred, fr)),
        };
        let mut blob = Vec::with_capacity(r.len() * 2 + 1);
        if policy == Policy::PerFeature {
            g.tags.push(p as u8);
            blob.push(p as u8);
        }
        for &v in &r {
            put_svarint(&mut blob, v);
        }
        g.res
            .extend_from_slice(&blob[(policy == Policy::PerFeature) as usize..]);
        g.per_feat.push(blob);
        g.vals.push((p, r));
    }
    g
}

fn planar(g: &Geom) -> Vec<u8> {
    let mut o = g.tags.clone();
    o.extend_from_slice(&g.res);
    o
}

/// Arrow `Binary` column body: `i32` offsets then data.
fn binary_col(blobs: &[Vec<u8>]) -> Vec<u8> {
    let mut o = Vec::new();
    let mut off = 0i32;
    o.extend_from_slice(&off.to_le_bytes());
    for b in blobs {
        off += b.len() as i32;
        o.extend_from_slice(&off.to_le_bytes());
    }
    for b in blobs {
        o.extend_from_slice(b);
    }
    o
}

// ------------------------------------------------------------------- time --

const UNITS: [i64; 8] = [86_400_000, 3_600_000, 60_000, 10_000, 1000, 100, 10, 1];

struct TimeQr {
    unit: i64,
    head: Vec<u8>,
    dq: Vec<u8>,
    r: Vec<u8>,
    per_feat: Vec<Vec<u8>>,
}

/// Exact times as an epoch-aligned quotient/remainder split: `t = U*q + r`.
/// `U` is the coarsest unit that at least 80 % of the layer's vertex times are
/// a multiple of, so source fixes get `r = 0` and a small regular `dq`, while
/// interpolated clip vertices keep full ms precision in `r` without disturbing
/// the `dq` chain of their neighbours.
fn encode_time_qr(l: &Layer) -> TimeQr {
    let n = l.t.len().max(1);
    let unit = UNITS
        .into_iter()
        .find(|&u| l.t.iter().filter(|&&t| t.rem_euclid(u) == 0).count() * 5 >= n * 4)
        .unwrap_or(1);
    let qmin = l.t.iter().map(|&t| t.div_euclid(unit)).min().unwrap_or(0);
    let mut out = TimeQr {
        unit,
        head: Vec::new(),
        dq: Vec::new(),
        r: Vec::new(),
        per_feat: Vec::new(),
    };
    put_uvarint(&mut out.head, unit as u64);
    put_svarint(&mut out.head, qmin);
    for &(s, e) in &l.ranges {
        let (mut bq, mut br) = (Vec::new(), Vec::new());
        let mut prev = qmin;
        for &t in &l.t[s..e] {
            let q = t.div_euclid(unit);
            // Signed: nothing here guarantees a feature's times are sorted.
            put_svarint(&mut bq, q - prev);
            put_uvarint(&mut br, t.rem_euclid(unit) as u64);
            prev = q;
        }
        out.dq.extend_from_slice(&bq);
        out.r.extend_from_slice(&br);
        bq.extend_from_slice(&br);
        out.per_feat.push(bq);
    }
    out
}

fn time_qr_bytes(q: &TimeQr) -> Vec<u8> {
    let mut o = q.head.clone();
    o.extend_from_slice(&q.dq);
    o.extend_from_slice(&q.r);
    o
}

/// Per-feature delta varints over integer times (first value relative to `base`).
fn delta_stream(l: &Layer, t: &[i64], base: i64) -> (Vec<u8>, Vec<Vec<u8>>) {
    let (mut all, mut per) = (Vec::new(), Vec::new());
    for &(s, e) in &l.ranges {
        let mut b = Vec::new();
        let mut prev = base;
        for &v in &t[s..e] {
            put_svarint(&mut b, v - prev);
            prev = v;
        }
        all.extend_from_slice(&b);
        per.push(b);
    }
    (all, per)
}

/// The default build's tier-1 vertex-time form: u16 offsets from the layer
/// minimum at `step = ceil(span / 65535)`, truncated (columns.rs).
fn lossy_u16(t: &[i64]) -> (Vec<i64>, i64) {
    let min = t.iter().copied().min().unwrap_or(0);
    let span = t.iter().copied().max().unwrap_or(0) - min;
    let step = if span <= 65535 {
        1
    } else {
        (span + 65534) / 65535
    };
    (
        t.iter().map(|&v| ((v - min) / step).min(65535)).collect(),
        step,
    )
}

// -------------------------------------------------------------- exact f64 --

struct Alp {
    e: u32,
    ints: Vec<[i64; 2]>,
    /// (flat value index, xor of the true bits with `k / 10^e`'s bits)
    exc: Vec<(usize, u64)>,
}

/// Decimal-grid detection. A value is *exact* at exponent `e` when
/// `round(v * 10^e) / 10^e` reproduces its bits; every other value still gets
/// its rounded integer (so the predictor chain stays smooth) plus an XOR
/// patch. `e` is the smallest exponent within 2 % of the best exact count.
fn alp(f: &[[f64; 2]]) -> Alp {
    let exact_at = |v: f64, p: f64| {
        let k = (v * p).round();
        k.abs() < 9.0e15 && (k / p).to_bits() == v.to_bits()
    };
    let counts: Vec<(u32, usize)> = (3..=8u32)
        .map(|e| {
            let p = 10f64.powi(e as i32);
            (e, f.iter().flatten().filter(|&&v| exact_at(v, p)).count())
        })
        .collect();
    let best = counts.iter().map(|c| c.1).max().unwrap_or(0);
    // No decimal value at all (a tile holding only clip vertices): every value
    // is patched, and a finer grid makes each patch shorter.
    let e = if best == 0 {
        7
    } else {
        counts
            .iter()
            .find(|c| c.1 * 50 >= best * 49)
            .map(|c| c.0)
            .unwrap_or(7)
    };
    let p = 10f64.powi(e as i32);
    let mut ints = Vec::with_capacity(f.len());
    let mut exc = Vec::new();
    for (i, xy) in f.iter().enumerate() {
        let mut k = [0i64; 2];
        for a in 0..2 {
            let kf = (xy[a] * p).round();
            k[a] = kf as i64;
            let x = (kf / p).to_bits() ^ xy[a].to_bits();
            if x != 0 {
                exc.push((i * 2 + a, x));
            }
        }
        ints.push(k);
    }
    Alp { e, ints, exc }
}

fn exc_stream(a: &Alp) -> Vec<u8> {
    let mut o = Vec::new();
    put_uvarint(&mut o, a.exc.len() as u64);
    let mut prev = 0usize;
    for &(i, _) in &a.exc {
        put_uvarint(&mut o, (i - prev) as u64);
        prev = i;
    }
    for &(_, x) in &a.exc {
        put_uvarint(&mut o, x);
    }
    o
}

// ------------------------------------------------------------ accumulator --

#[derive(Default, Clone)]
struct Acc {
    /// frame-level compressed bytes per variant
    frame: BTreeMap<&'static str, u64>,
    /// column-alone compressed bytes
    col: BTreeMap<&'static str, u64>,
    /// counters
    n: BTreeMap<String, u64>,
    zoom: BTreeMap<(u8, &'static str), u64>,
    /// zigzag bit-length histogram of non-first residuals (entropy bound)
    cls: Vec<u64>,
}

impl Acc {
    fn merge(mut self, o: Acc) -> Acc {
        for (k, v) in o.frame {
            *self.frame.entry(k).or_default() += v;
        }
        for (k, v) in o.col {
            *self.col.entry(k).or_default() += v;
        }
        for (k, v) in o.n {
            *self.n.entry(k).or_default() += v;
        }
        for (k, v) in o.zoom {
            *self.zoom.entry(k).or_default() += v;
        }
        if self.cls.len() < o.cls.len() {
            self.cls.resize(o.cls.len(), 0);
        }
        for (i, v) in o.cls.iter().enumerate() {
            self.cls[i] += v;
        }
        self
    }
    fn bump(&mut self, k: &str, v: u64) {
        *self.n.entry(k.to_string()).or_default() += v;
    }
}

/// One tile's raw payload under every variant.
struct TileOut {
    variants: Vec<(&'static str, Vec<u8>)>,
    acc: Acc,
    vertices: u64,
}

fn build(reader: &PackedReader, e: &TileEntry, quant_m: f64, level: i32) -> Option<TileOut> {
    let p = reader.read_payload(e).ok()?;
    let layers = reader.decode_payload(&p).ok()?;
    let frame = parse_frame(&p)?;
    let mut acc = Acc {
        cls: vec![0; 66],
        ..Default::default()
    };
    if frame.len() != layers.len() {
        acc.bump("tiles: frame/decode layer mismatch", 1);
        return Some(TileOut {
            variants: vec![("A0 stored", p)],
            acc,
            vertices: 0,
        });
    }

    // Per variant: edits on the payload + bytes appended as a side section.
    let names: Vec<&'static str> = vec![
        "A0 stored",
        "A0i stored, wire id zeroed",
        "A1 default (lossy u16 t)",
        "A2 default + exact time q/r",
        "A3 exact",
        "A3i exact, wire id zeroed",
        "Q0 grid stored",
        "Q1 grid codec, lossy t delta",
        "Q2 grid codec",
        "Q2b grid codec, Arrow Binary cols",
        "Q2d grid codec, delta only",
        "Q2i grid codec, wire id zeroed",
        "Q2n grid codec, dense wire id",
        "Q3 grid codec, per-layer predictor",
        "Q4 grid codec, first vertex rel prev",
    ];
    let mut ed: BTreeMap<&'static str, (Vec<Edit>, Vec<u8>)> = names
        .iter()
        .map(|&n| (n, (Vec::new(), Vec::new())))
        .collect();
    let mut vertices = 0u64;
    let mut mapped_all = true;

    for (li, dl) in layers.iter().enumerate() {
        let b = &dl.batch;
        let Some(l) = read_layer(b, quant_m) else {
            acc.bump("layers: not a trajectory layer", 1);
            mapped_all = false;
            continue;
        };
        let nv = l.g.len();
        vertices += nv as u64;
        acc.bump("layers", 1);
        acc.bump("features", l.ranges.len() as u64);
        acc.bump("vertices", nv as u64);

        // ---- locate the buffers in the CORE_BATCH body -------------------
        let Some(&(_, cs, cl)) = frame[li].1.iter().find(|s| s.0 == 0x03) else {
            mapped_all = false;
            continue;
        };
        let Some(bufs) = map_core(&p, cs, cl) else {
            mapped_all = false;
            continue;
        };
        let schema = b.schema();
        let mut pre = vec![0usize];
        for fld in schema.fields() {
            let Some(k) = nbuf(fld.data_type()) else {
                break;
            };
            pre.push(pre.last().unwrap() + k);
        }
        let idx = |name: &str| schema.fields().iter().position(|f| f.name() == name);
        let (Some(gi), Some(vi)) = (idx("geometry"), idx("vertex_time")) else {
            mapped_all = false;
            continue;
        };
        // `end_time` is synthesized by the decoder when the tile omits it.
        let adj = if pre.contains(&bufs.len()) {
            0
        } else if pre.contains(&(bufs.len() + 2)) && idx("end_time").is_some_and(|i| i < gi) {
            2
        } else {
            usize::MAX
        };
        let ok = adj != usize::MAX && gi < pre.len() - 1 && vi < pre.len() - 1 && {
            let (g0, v0) = (pre[gi] - adj, pre[vi] - adj);
            v0 + 3 < bufs.len()
                && bufs[g0 + 4].1 == nv * 2 * l.leaf_w
                && bufs[v0 + 3].1 == nv * l.vt_w
                && bufs[1].1 == l.ranges.len() * 8
        };
        if !ok {
            acc.bump("layers: buffers not located", 1);
            mapped_all = false;
            continue;
        }
        let (g0, v0) = (pre[gi] - adj, pre[vi] - adj);
        let rng = |i: usize| (bufs[i].0, bufs[i].2);
        let (g_off, g_leaf, v_off, v_leaf, id_buf) =
            (rng(g0 + 1), rng(g0 + 4), rng(v0 + 1), rng(v0 + 3), rng(1));
        let cut4 = |v: &mut Vec<Edit>| {
            for r in [g_off, g_leaf, v_off, v_leaf] {
                v.push((r.0, r.1, Vec::new()));
            }
        };

        // ---- streams -----------------------------------------------------
        let mut cnt = Vec::new();
        for &(s, e) in &l.ranges {
            put_uvarint(&mut cnt, (e - s) as u64);
        }
        let g_best = encode_geom(&l, &l.g, Policy::PerFeature, false);
        let g_layer = encode_geom(&l, &l.g, Policy::PerLayer, false);
        let g_delta = encode_geom(&l, &l.g, Policy::DeltaOnly, false);
        let g_rel = encode_geom(&l, &l.g, Policy::PerFeature, true);
        for (p, r) in &g_best.vals {
            acc.bump(&format!("pick {p:?}"), 1);
            for &v in r.iter().skip(2) {
                acc.cls[(64 - zigzag(v).leading_zeros()) as usize] += 1;
            }
        }

        // times
        let (u16v, u16step) = if l.exact_t {
            lossy_u16(&l.t)
        } else {
            (l.t.clone(), 0)
        };
        let u16_bytes: Vec<u8> = u16v
            .iter()
            .flat_map(|&v| (v as u16).to_le_bytes())
            .collect();
        let (u16_delta, _) = delta_stream(&l, &u16v, 0);
        let tmin = l.t.iter().copied().min().unwrap_or(0);
        let (ms_delta, _) = delta_stream(&l, &l.t, tmin);
        let qr = encode_time_qr(&l);
        let time_exact = if l.exact_t {
            time_qr_bytes(&qr)
        } else {
            u16_delta.clone()
        };
        let time_per_feat: Vec<Vec<u8>> = if l.exact_t {
            qr.per_feat.clone()
        } else {
            delta_stream(&l, &u16v, 0).1
        };
        if l.exact_t {
            acc.bump(&format!("time unit {} ms", qr.unit), 1);
            acc.bump("u16 step sum (ms)", u16step as u64);
            acc.bump("u16 step > 1 layers", (u16step > 1) as u64);
        } else {
            acc.bump("layers: stored time not exact", 1);
        }
        *acc.col.entry("t  stored leaf").or_default() +=
            zlen(&p[v_leaf.0..v_leaf.0 + nv * l.vt_w], level);
        *acc.col.entry("t  default u16 (lossy)").or_default() += zlen(&u16_bytes, level);
        *acc.col.entry("t  u16 delta varint (lossy)").or_default() += zlen(&u16_delta, level);
        if l.exact_t {
            *acc.col.entry("t  exact ms delta varint").or_default() += zlen(&ms_delta, level);
            *acc.col.entry("t  exact q/r").or_default() += zlen(&time_qr_bytes(&qr), level);
        }

        // grid geometry, column-alone
        let i32_bytes: Vec<u8> =
            l.g.iter()
                .flat_map(|p| [p[0] as i32, p[1] as i32])
                .flat_map(|v| v.to_le_bytes())
                .collect();
        *acc.col.entry("g  stored leaf").or_default() +=
            zlen(&p[g_leaf.0..g_leaf.0 + nv * 2 * l.leaf_w], level);
        *acc.col.entry("g  grid i32 abs").or_default() += zlen(&i32_bytes, level);
        *acc.col.entry("g  grid per-feature best").or_default() += zlen(&planar(&g_best), level);
        *acc.col.entry("g  grid per-layer best").or_default() += zlen(&planar(&g_layer), level);
        *acc.col.entry("g  grid delta only").or_default() += zlen(&planar(&g_delta), level);
        *acc.col.entry("g  grid best, first rel prev").or_default() += zlen(&planar(&g_rel), level);

        // ---- frame variants: grid ----------------------------------------
        let is_f64 = l.leaf_w == 8;
        let side = |g: &Geom, t: &[u8]| {
            let mut o = cnt.clone();
            o.extend_from_slice(&planar(g));
            o.extend_from_slice(t);
            o
        };
        if is_f64 {
            let v = &mut ed.get_mut("Q0 grid stored").unwrap().0;
            v.push((g_leaf.0, g_leaf.1, i32_bytes.clone()));
            if l.exact_t {
                v.push((v_leaf.0, v_leaf.1, u16_bytes.clone()));
            }
        }
        for (name, g, t) in [
            ("Q1 grid codec, lossy t delta", &g_best, &u16_delta),
            ("Q2 grid codec", &g_best, &time_exact),
            ("Q2d grid codec, delta only", &g_delta, &time_exact),
            ("Q2i grid codec, wire id zeroed", &g_best, &time_exact),
            ("Q2n grid codec, dense wire id", &g_best, &time_exact),
            ("Q3 grid codec, per-layer predictor", &g_layer, &time_exact),
            ("Q4 grid codec, first vertex rel prev", &g_rel, &time_exact),
        ] {
            let (v, tail) = ed.get_mut(name).unwrap();
            cut4(v);
            tail.extend_from_slice(&side(g, t));
        }
        {
            let v = &mut ed.get_mut("Q2b grid codec, Arrow Binary cols").unwrap().0;
            v.push((g_off.0, g_off.1, Vec::new()));
            v.push((g_leaf.0, g_leaf.1, binary_col(&g_best.per_feat)));
            v.push((v_off.0, v_off.1, Vec::new()));
            v.push((v_leaf.0, v_leaf.1, binary_col(&time_per_feat)));
        }
        let nfeat = l.ranges.len();
        let zero_id = vec![0u8; nfeat * 8];
        let dense_id: Vec<u8> = (0..nfeat as u64).flat_map(|i| i.to_le_bytes()).collect();
        for (name, bytes) in [
            ("A0i stored, wire id zeroed", &zero_id),
            ("A3i exact, wire id zeroed", &zero_id),
            ("Q2i grid codec, wire id zeroed", &zero_id),
            ("Q2n grid codec, dense wire id", &dense_id),
        ] {
            ed.get_mut(name)
                .unwrap()
                .0
                .push((id_buf.0, id_buf.1, bytes.clone()));
        }

        // ---- frame variants: f64 -----------------------------------------
        if is_f64 && l.exact_t {
            ed.get_mut("A1 default (lossy u16 t)").unwrap().0.push((
                v_leaf.0,
                v_leaf.1,
                u16_bytes.clone(),
            ));
            let (v, tail) = ed.get_mut("A2 default + exact time q/r").unwrap();
            v.push((v_off.0, v_off.1, Vec::new()));
            v.push((v_leaf.0, v_leaf.1, Vec::new()));
            tail.extend_from_slice(&time_exact);
        }
        if is_f64 {
            let a = alp(&l.f);
            let g_alp = encode_geom(&l, &a.ints, Policy::PerFeature, false);
            let exc = exc_stream(&a);
            // Round trip: residuals -> ints -> k / 10^e, patched -> source bits.
            let p10 = 10f64.powi(a.e as i32);
            let mut bad = 0u64;
            let mut patch: BTreeMap<usize, u64> = a.exc.iter().copied().collect();
            for (fi, &(s, e)) in l.ranges.iter().enumerate() {
                let (pr, r) = &g_alp.vals[fi];
                let back = unresiduals(r, &l.t[s..e], *pr, [0, 0]);
                for (j, k) in back.iter().enumerate() {
                    for ax in 0..2 {
                        let mut bits = (k[ax] as f64 / p10).to_bits();
                        if let Some(x) = patch.remove(&((s + j) * 2 + ax)) {
                            bits ^= x;
                        }
                        bad += (bits != l.f[s + j][ax].to_bits()) as u64;
                    }
                }
            }
            acc.bump("exact f64: round-trip mismatches", bad);
            acc.bump(&format!("decimal exponent {}", a.e), 1);
            acc.bump("exact f64: values", nv as u64 * 2);
            acc.bump("exact f64: exceptions", a.exc.len() as u64);
            // Is an exception a feature endpoint (clip / bucket-slice vertex)?
            let mut ends = vec![false; nv];
            for &(s, e) in &l.ranges {
                if e > s {
                    ends[s] = true;
                    ends[e - 1] = true;
                }
            }
            acc.bump(
                "exact f64: endpoint values",
                ends.iter().filter(|&&x| x).count() as u64 * 2,
            );
            acc.bump(
                "exact f64: exceptions at endpoints",
                a.exc.iter().filter(|x| ends[x.0 / 2]).count() as u64,
            );
            *acc.col
                .entry("g  exact: decimal ints, per-feature best")
                .or_default() += zlen(&planar(&g_alp), level);
            *acc.col.entry("g  exact: exception patches").or_default() += zlen(&exc, level);
            for name in ["A3 exact", "A3i exact, wire id zeroed"] {
                let (v, tail) = ed.get_mut(name).unwrap();
                cut4(v);
                tail.extend_from_slice(&cnt);
                tail.extend_from_slice(&planar(&g_alp));
                tail.extend_from_slice(&exc);
                tail.extend_from_slice(&time_exact);
            }
        } else {
            for name in [
                "A1 default (lossy u16 t)",
                "A2 default + exact time q/r",
                "A3 exact",
                "A3i exact, wire id zeroed",
            ] {
                ed.remove(name);
            }
        }
    }

    if !mapped_all {
        acc.bump("tiles: skipped (unmapped layer)", 1);
        return Some(TileOut {
            variants: vec![("A0 stored", p)],
            acc,
            vertices,
        });
    }
    let variants = ed
        .into_iter()
        .map(|(name, (mut edits, tail))| (name, apply(&p, &mut edits, &tail)))
        .collect();
    Some(TileOut {
        variants,
        acc,
        vertices,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: trajectory-frontier <manifest.json> <quant_m> [every_nth_tile=1] [zstd_level=3]");
        std::process::exit(2);
    }
    let quant_m: f64 = args[2].parse()?;
    let nth: usize = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1)
        .max(1);
    let level: i32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let reader = PackedReader::open(&args[1])?;
    let entries: Vec<&TileEntry> = reader.entries().iter().step_by(nth).collect();

    // ---- phase 1: train one dictionary per dictionary variant -------------
    // Every 64th evaluated tile trains and is then left out of every total.
    let train: Vec<&TileEntry> = entries.iter().step_by(64).copied().collect();
    let mut samples: BTreeMap<&'static str, Vec<Vec<u8>>> = BTreeMap::new();
    let mut budget = 96usize << 20;
    for e in &train {
        let Some(t) = build(&reader, e, quant_m, level) else {
            continue;
        };
        for (name, bytes) in t.variants {
            // zstd's trainer wants many small samples; cap the huge coarse tiles.
            if DICT_KEYS.contains(&name) && bytes.len() <= 256 << 10 && budget > bytes.len() {
                budget -= bytes.len();
                samples.entry(name).or_default().push(bytes);
            }
        }
    }
    let dicts: BTreeMap<&'static str, Arc<Vec<u8>>> = samples
        .iter()
        .filter_map(|(k, s)| {
            zstd::dict::from_samples(s, DICT_BYTES)
                .ok()
                .map(|d| (*k, Arc::new(d)))
        })
        .collect();

    // ---- phase 2 ---------------------------------------------------------
    let eval: Vec<&TileEntry> = entries
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 64 != 0)
        .map(|(_, e)| *e)
        .collect();
    // `PackedReader` is not `Sync`; each worker opens its own.
    let path = args[1].clone();
    let acc = eval
        .par_iter()
        .map_init(
            || PackedReader::open(&path).expect("reopen archive"),
            |reader, e| {
                let mut acc = Acc::default();
                let Some(t) = build(reader, e, quant_m, level) else {
                    acc.bump("tiles: unreadable", 1);
                    return acc;
                };
                acc = acc.merge(t.acc);
                acc.bump("tiles", 1);
                acc.bump("stored blob bytes", e.length as u64);
                *acc.zoom.entry((e.zoom, "tiles")).or_default() += 1;
                *acc.zoom.entry((e.zoom, "vertices")).or_default() += t.vertices;
                for (name, bytes) in &t.variants {
                    let z = zlen(bytes, level);
                    *acc.frame.entry(name).or_default() += z;
                    *acc.zoom.entry((e.zoom, name)).or_default() += z;
                    if let Some(d) = dicts.get(name) {
                        let zd = zstd::bulk::Compressor::with_dictionary(level, d)
                            .and_then(|mut c| c.compress(bytes))
                            .map(|v| v.len())
                            .unwrap_or(bytes.len()) as u64;
                        let key: &'static str = match *name {
                            "A0 stored" => "A0+dict",
                            "A3 exact" => "A3+dict",
                            "Q0 grid stored" => "Q0+dict",
                            _ => "Q2+dict",
                        };
                        *acc.frame.entry(key).or_default() += zd;
                        *acc.zoom.entry((e.zoom, key)).or_default() += zd;
                    }
                }
                acc
            },
        )
        .reduce(Acc::default, Acc::merge);

    let mb = |b: u64| b as f64 / 1e6;
    let n = |k: &str| acc.n.get(k).copied().unwrap_or(0);
    let nv = n("vertices").max(1) as f64;
    println!(
        "=== {}  (grid {} m, zstd -{}, every {} tile, {} evaluated, {} trained on)",
        args[1],
        quant_m,
        level,
        nth,
        n("tiles"),
        train.len()
    );
    println!(
        "layers {}  features {}  vertices {}  ({:.1} vtx/feature, {:.0} vtx/tile)",
        n("layers"),
        n("features"),
        n("vertices"),
        nv / n("features").max(1) as f64,
        nv / n("tiles").max(1) as f64
    );
    println!(
        "stored blob bytes (directory): {:.2} MB",
        mb(n("stored blob bytes"))
    );

    let base = |k: &str| acc.frame.get(k).copied().unwrap_or(0);
    println!("\n-- frame level: zstd of the whole tile payload");
    println!(
        "   {:<38} {:>9} {:>7}  {:>8} {:>8} {:>8}",
        "variant", "MB", "B/vtx", "vs A0", "vs A1", "vs Q0"
    );
    for (k, v) in &acc.frame {
        let rel = |b: &str| {
            if base(b) == 0 {
                "       -".to_string()
            } else {
                format!("{:+7.1}%", 100.0 * (*v as f64 / base(b) as f64 - 1.0))
            }
        };
        println!(
            "   {:<38} {:>9.2} {:>7.2}  {} {} {}",
            k,
            mb(*v),
            *v as f64 / nv,
            rel("A0 stored"),
            rel("A1 default (lossy u16 t)"),
            rel("Q0 grid stored")
        );
    }

    println!("\n-- columns, each zstd'd alone");
    for (k, v) in &acc.col {
        println!(
            "   {:<42} {:>9.2} MB {:>7.3} B/vtx",
            k,
            mb(*v),
            *v as f64 / nv
        );
    }

    println!("\n-- counters");
    for (k, v) in &acc.n {
        println!("   {:<42} {}", k, v);
    }
    if n("exact f64: values") > 0 {
        println!(
            "   exceptions: {:.2}% of values; {:.1}% of them at feature endpoints; {:.1}% of endpoint values are exceptions",
            100.0 * n("exact f64: exceptions") as f64 / n("exact f64: values") as f64,
            100.0 * n("exact f64: exceptions at endpoints") as f64 / n("exact f64: exceptions").max(1) as f64,
            100.0 * n("exact f64: exceptions at endpoints") as f64 / n("exact f64: endpoint values").max(1) as f64,
        );
    }
    if n("u16 step > 1 layers") > 0 {
        println!(
            "   default u16 vertex_time: {:.1}% of layers truncate (mean step {:.1} ms)",
            100.0 * n("u16 step > 1 layers") as f64 / n("layers").max(1) as f64,
            n("u16 step sum (ms)") as f64 / n("layers").max(1) as f64
        );
    }

    // Order-0 bound on the non-first grid residuals: an ideal coder of the
    // zigzag bit-length class, plus the (class - 1) raw mantissa bits.
    let tot: u64 = acc.cls.iter().sum();
    if tot > 0 {
        let mut bits = 0f64;
        for (c, &k) in acc.cls.iter().enumerate() {
            if k > 0 {
                bits += k as f64 * (-(k as f64 / tot as f64).log2() + (c as f64 - 1.0).max(0.0));
            }
        }
        println!(
            "\n-- order-0 entropy bound, non-first grid residuals: {:.3} B/vtx  (varint+zstd, all residuals: {:.3} B/vtx)",
            bits / 8.0 / nv,
            acc.col["g  grid per-feature best"] as f64 / nv
        );
    }

    println!("\n-- per zoom (MB)");
    let keys = [
        "A0 stored",
        "A1 default (lossy u16 t)",
        "A3 exact",
        "A0+dict",
        "A3+dict",
        "Q0 grid stored",
        "Q2 grid codec",
        "Q2+dict",
    ];
    print!("   {:<4} {:>8} {:>10}", "z", "tiles", "vertices");
    for k in keys {
        print!(" {:>10}", k.split(' ').next().unwrap());
    }
    println!();
    let zooms: std::collections::BTreeSet<u8> = acc.zoom.keys().map(|k| k.0).collect();
    for z in zooms {
        let g = |k: &'static str| acc.zoom.get(&(z, k)).copied().unwrap_or(0);
        print!("   z{:<3} {:>8} {:>10}", z, g("tiles"), g("vertices"));
        for k in keys {
            print!(" {:>10.2}", mb(g(k)));
        }
        println!();
    }
    Ok(())
}
