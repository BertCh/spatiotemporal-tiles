//! The path codec — the `PATH_CODEC` (`0x06`) layer-frame section.
//!
//! A LineString layer's coordinates and per-vertex times, stored as
//! per-feature predictor residuals in a side section instead of as Arrow leaf
//! buffers. Lossless: [`decode`] reproduces exactly the integers (world-grid
//! mode) or the f64 bit patterns (decimal mode) and the millisecond vertex
//! times [`encode`] was given.
//!
//! The normative wire format and decode algorithm are in
//! `docs/spec/stt-packed-format.md` §5.2.7 ("Path codec section"); this module
//! is the reference implementation of both directions. The frame integration
//! (empty CORE columns, column substitution at decode) lives in `encode.rs` /
//! `decode.rs`.
//!
//! # Exactness across readers
//!
//! Every integer the decoder reads or computes — each varint value, each
//! quotient, time, coordinate integer and predictor — must lie within
//! ±(2^53 − 1) ([`MAX_SAFE`]), so a decoder that holds them in IEEE doubles
//! (the TypeScript reader) and one that holds them in `i64` (this one) accept
//! and reject exactly the same sections and produce the same values. The
//! encoder's own limits (`|k| < 2^50`, `|t| <= 2^52`, the `TIME_VEL` product
//! bound) keep every valid section well inside it.

use crate::error::{Error, Result};

/// Section format version this module reads and writes.
pub const PATH_CODEC_VERSION: u8 = 1;
/// `flags` bit 0: the layer has per-vertex times.
pub const FLAG_TIME: u8 = 0b01;
/// `flags` bit 1: coordinate integers are decimal (`k / 10^e`), not
/// world-grid indices.
pub const FLAG_DECIMAL: u8 = 0b10;

/// Per-feature predictor tag: `pred = k[i-1]`.
pub const TAG_DELTA: u8 = 0;
/// Per-feature predictor tag: `pred = k[i-1] + (k[i-1] - k[i-2])`.
pub const TAG_CONST_VEL: u8 = 1;
/// Per-feature predictor tag: constant velocity scaled by the vertex-time
/// ratio `dt1 / dt0` (requires [`FLAG_TIME`]).
pub const TAG_TIME_VEL: u8 = 2;

/// Largest magnitude any decoded or computed integer may have: `2^53 − 1`,
/// the largest integer an IEEE double holds exactly together with all its
/// neighbours.
pub const MAX_SAFE: i64 = (1i64 << 53) - 1;
/// Encoder limit on a decimal coordinate integer: `|k| < 2^50`.
const MAX_K: i64 = 1i64 << 50;
/// Encoder limit on a vertex time: `|t| < 2^52` ms. Strict, so a quotient
/// difference `dq` can never reach `2^53`.
pub const MAX_TIME_MS: i64 = 1i64 << 52;
/// Encoder limit on `|v * dt1|` for a `TIME_VEL` feature.
const MAX_TIME_VEL_PRODUCT: i128 = 1i128 << 52;
/// Largest decimal exponent a section may carry.
const MAX_EXPONENT: u8 = 15;
/// Candidate time units, coarsest first (spec: recommended, not required).
const UNITS: [i64; 8] = [86_400_000, 3_600_000, 60_000, 10_000, 1000, 100, 10, 1];

/// `10^e` for `e` in `0..=15`, every one exactly representable in binary64.
const POW10: [f64; 16] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
];

/// A layer's coordinates, flat across all features (vertex `n` of the layer
/// is `coords[n]`).
#[derive(Debug, Clone, PartialEq)]
pub enum PathCoords {
    /// World-grid mode: the `Int32` grid indices of a `--quantize-coords`
    /// layer (`stt:quant` applies on top, as for any quantized layer).
    Grid(Vec<[i32; 2]>),
    /// Decimal mode: raw `Float64` coordinates, reproduced bit-exactly.
    Decimal(Vec<[f64; 2]>),
}

impl PathCoords {
    /// Number of vertices.
    pub fn len(&self) -> usize {
        match self {
            PathCoords::Grid(v) => v.len(),
            PathCoords::Decimal(v) => v.len(),
        }
    }

    /// Whether there are no vertices.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One layer's worth of path-codec content: what [`encode`] takes and
/// [`decode`] returns.
#[derive(Debug, Clone)]
pub struct PathLayer {
    /// Vertices per feature, in row order (`0` is allowed).
    pub counts: Vec<u32>,
    /// Every vertex's coordinate, features in row order.
    pub coords: PathCoords,
    /// Every vertex's absolute Unix-ms time, aligned 1:1 with `coords`, or
    /// `None` for a layer without `vertex_time`.
    pub times: Option<Vec<i64>>,
}

impl PartialEq for PathLayer {
    /// Bit-exact on f64 coordinates (`NaN == NaN` when the bits agree,
    /// `0.0 != -0.0`), which is the codec's actual contract.
    fn eq(&self, other: &Self) -> bool {
        let coords_eq = match (&self.coords, &other.coords) {
            (PathCoords::Grid(a), PathCoords::Grid(b)) => a == b,
            (PathCoords::Decimal(a), PathCoords::Decimal(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b).all(|(p, q)| {
                        p[0].to_bits() == q[0].to_bits() && p[1].to_bits() == q[1].to_bits()
                    })
            }
            _ => false,
        };
        coords_eq && self.counts == other.counts && self.times == other.times
    }
}

/// Whether every time is strictly inside the encoder's `±2^52` ms window — the one
/// input condition [`encode`] can refuse on. A layer that fails it is encoded
/// without the codec.
pub fn times_in_range(times: &[i64]) -> bool {
    times.iter().all(|t| t.unsigned_abs() < MAX_TIME_MS as u64)
}

// ---------------------------------------------------------------- varints --

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
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

// ---------------------------------------------------------------- encoder --

/// Exact decimal coordinates: the chosen exponent, every value's integer, and
/// the XOR patches of the values `k / 10^e` does not reproduce.
struct Decimal {
    e: u8,
    ints: Vec<[i64; 2]>,
    /// `(flat value index, xor patch)`, ascending by index.
    exc: Vec<(usize, u64)>,
}

/// The integer a decimal-mode encoder stores for `v` at `p = 10^e`: the
/// rounded scaled value when it is finite and inside `|k| < 2^50`, else `0`
/// (the value then rides entirely in its patch).
fn decimal_int(v: f64, p: f64) -> i64 {
    let kf = (v * p).round();
    if kf.is_finite() && kf.abs() < MAX_K as f64 {
        kf as i64
    } else {
        0
    }
}

fn choose_decimal(coords: &[[f64; 2]]) -> Decimal {
    let exact_at = |v: f64, p: f64| {
        let kf = (v * p).round();
        kf.is_finite() && kf.abs() < MAX_K as f64 && (kf / p).to_bits() == v.to_bits()
    };
    let counts: Vec<(u8, usize)> = (3..=8u8)
        .map(|e| {
            let p = POW10[e as usize];
            (
                e,
                coords.iter().flatten().filter(|&&v| exact_at(v, p)).count(),
            )
        })
        .collect();
    let best = counts.iter().map(|c| c.1).max().unwrap_or(0);
    // No decimal value at all: every value is patched, and a finer grid keeps
    // the integers (and so the residuals) meaningful for the predictor.
    let e = if best == 0 {
        7
    } else {
        counts
            .iter()
            .find(|c| c.1 * 50 >= best * 49)
            .map(|c| c.0)
            .unwrap_or(7)
    };
    let p = POW10[e as usize];
    let mut ints = Vec::with_capacity(coords.len());
    let mut exc = Vec::new();
    for (n, xy) in coords.iter().enumerate() {
        let mut k = [0i64; 2];
        for a in 0..2 {
            k[a] = decimal_int(xy[a], p);
            let patch = (k[a] as f64 / p).to_bits() ^ xy[a].to_bits();
            if patch != 0 {
                exc.push((2 * n + a, patch));
            }
        }
        ints.push(k);
    }
    Decimal { e, ints, exc }
}

fn choose_unit(times: &[i64]) -> i64 {
    let n = times.len();
    UNITS
        .into_iter()
        .find(|&u| times.iter().filter(|&&t| t.rem_euclid(u) == 0).count() * 5 >= n * 4)
        .unwrap_or(1)
}

/// `floor(v * dt1 / dt0 + 0.5)` in binary64, in that order, no FMA.
fn time_vel_term(v: i64, dt1: i64, dt0: i64) -> f64 {
    (v as f64 * dt1 as f64 / dt0 as f64 + 0.5).floor()
}

/// Encoder-side predictor (inputs are bounded by the encoder limits, so the
/// arithmetic cannot overflow).
fn predict(k: &[[i64; 2]], t: &[i64], i: usize, a: usize, tag: u8) -> i64 {
    if i == 1 || tag == TAG_DELTA {
        return k[i - 1][a];
    }
    let v = k[i - 1][a] - k[i - 2][a];
    if tag == TAG_CONST_VEL {
        return k[i - 1][a] + v;
    }
    let (dt0, dt1) = (t[i - 1] - t[i - 2], t[i] - t[i - 1]);
    if dt0 > 0 {
        k[i - 1][a] + time_vel_term(v, dt1, dt0) as i64
    } else {
        k[i - 1][a] + v
    }
}

/// `TIME_VEL` is legal for a feature only when every `|v * dt1| <= 2^52`.
fn time_vel_allowed(k: &[[i64; 2]], t: &[i64]) -> bool {
    (2..k.len()).all(|i| {
        let dt1 = (t[i] - t[i - 1]) as i128;
        (0..2).all(|a| {
            let v = (k[i - 1][a] - k[i - 2][a]) as i128;
            (v * dt1).abs() <= MAX_TIME_VEL_PRODUCT
        })
    })
}

/// Residuals of one feature (x then y per vertex) under `tag`; the first
/// vertex is relative to `first`.
fn residuals(k: &[[i64; 2]], t: &[i64], tag: u8, first: [i64; 2], out: &mut Vec<i64>) {
    out.clear();
    for i in 0..k.len() {
        for (a, &f) in first.iter().enumerate() {
            let pred = if i == 0 { f } else { predict(k, t, i, a, tag) };
            out.push(k[i][a] - pred);
        }
    }
}

fn cost(r: &[i64]) -> usize {
    r.iter().map(|&v| varint_len(zigzag(v))).sum()
}

/// Encode one layer into `PATH_CODEC` section bytes.
///
/// A pure function of `layer`: identical content always produces identical
/// bytes. Errors (never panics) when the input is inconsistent — `counts` not
/// summing to the vertex count, `times` misaligned with the coordinates — or
/// when a time is outside `±2^52` ms (see [`times_in_range`]).
pub fn encode(layer: &PathLayer) -> Result<Vec<u8>> {
    let nv = layer.coords.len();
    let sum: u64 = layer.counts.iter().map(|&c| c as u64).sum();
    if sum != nv as u64 {
        return Err(Error::Other(format!(
            "path codec: feature vertex counts sum to {sum} but the layer has {nv} vertices"
        )));
    }
    if let Some(t) = &layer.times {
        if t.len() != nv {
            return Err(Error::Other(format!(
                "path codec: {} vertex times for {nv} vertices",
                t.len()
            )));
        }
        if !times_in_range(t) {
            return Err(Error::Other(
                "path codec: a vertex time is outside the ±2^52 ms window".into(),
            ));
        }
    }

    let (ints, decimal): (Vec<[i64; 2]>, Option<Decimal>) = match &layer.coords {
        PathCoords::Grid(g) => (g.iter().map(|p| [p[0] as i64, p[1] as i64]).collect(), None),
        PathCoords::Decimal(f) => {
            let mut d = choose_decimal(f);
            (std::mem::take(&mut d.ints), Some(d))
        }
    };

    let mut flags = 0u8;
    if layer.times.is_some() {
        flags |= FLAG_TIME;
    }
    if decimal.is_some() {
        flags |= FLAG_DECIMAL;
    }

    let mut out = Vec::with_capacity(16 + nv * 6);
    out.push(PATH_CODEC_VERSION);
    out.push(flags);
    put_uvarint(&mut out, layer.counts.len() as u64);
    put_uvarint(&mut out, nv as u64);
    if let Some(d) = &decimal {
        out.push(d.e);
    }

    // Feature vertex ranges, and the time streams.
    let mut ranges = Vec::with_capacity(layer.counts.len());
    let mut s = 0usize;
    for &c in &layer.counts {
        ranges.push((s, s + c as usize));
        s += c as usize;
    }
    let empty: Vec<i64> = Vec::new();
    let times: &[i64] = layer.times.as_deref().unwrap_or(&empty);
    let mut time_streams: Option<(Vec<u8>, Vec<u8>)> = None;
    if let Some(t) = &layer.times {
        let unit = choose_unit(t);
        let qbase = ranges
            .iter()
            .find(|r| r.1 > r.0)
            .map(|r| t[r.0].div_euclid(unit))
            .unwrap_or(0);
        put_uvarint(&mut out, unit as u64);
        put_svarint(&mut out, qbase);
        let (mut dq, mut rr) = (Vec::with_capacity(nv), Vec::with_capacity(nv));
        let mut anchor = qbase;
        for &(s, e) in &ranges {
            let mut prev = anchor;
            for (j, &tv) in t[s..e].iter().enumerate() {
                let q = tv.div_euclid(unit);
                put_svarint(&mut dq, q - prev);
                put_uvarint(&mut rr, tv.rem_euclid(unit) as u64);
                prev = q;
                if j == 0 {
                    anchor = q;
                }
            }
        }
        time_streams = Some((dq, rr));
    }

    // Counts.
    for &c in &layer.counts {
        put_uvarint(&mut out, c as u64);
    }

    // Per-feature predictor choice + residuals.
    let mut tags = Vec::with_capacity(ranges.len());
    let mut res = Vec::with_capacity(nv * 4);
    let mut first = [0i64; 2];
    let (mut best, mut cand) = (Vec::new(), Vec::new());
    for &(s, e) in &ranges {
        let k = &ints[s..e];
        let ft: &[i64] = if layer.times.is_some() {
            &times[s..e]
        } else {
            &[]
        };
        let mut best_tag = TAG_DELTA;
        residuals(k, ft, TAG_DELTA, first, &mut best);
        let mut best_cost = cost(&best);
        // Only a feature with a third vertex can tell the predictors apart.
        if k.len() >= 3 {
            let mut candidates = vec![TAG_CONST_VEL];
            if layer.times.is_some() && time_vel_allowed(k, ft) {
                candidates.push(TAG_TIME_VEL);
            }
            for tag in candidates {
                residuals(k, ft, tag, first, &mut cand);
                let c = cost(&cand);
                if c < best_cost {
                    best_cost = c;
                    best_tag = tag;
                    std::mem::swap(&mut best, &mut cand);
                }
            }
        }
        tags.push(best_tag);
        for &r in &best {
            put_svarint(&mut res, r);
        }
        if let Some(k0) = k.first() {
            first = *k0;
        }
    }
    out.extend_from_slice(&tags);
    if let Some((dq, rr)) = &time_streams {
        out.extend_from_slice(dq);
        out.extend_from_slice(rr);
    }
    out.extend_from_slice(&res);

    if let Some(d) = &decimal {
        put_uvarint(&mut out, d.exc.len() as u64);
        let mut prev = 0usize;
        for (i, &(idx, _)) in d.exc.iter().enumerate() {
            put_uvarint(&mut out, if i == 0 { idx } else { idx - prev } as u64);
            prev = idx;
        }
        for &(_, patch) in &d.exc {
            put_uvarint(&mut out, patch & 0xFFFF_FFFF);
            put_uvarint(&mut out, patch >> 32);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- decoder --

fn err(msg: impl Into<String>) -> Error {
    Error::Other(format!("path codec section: {}", msg.into()))
}

/// Bounds-checked cursor over the section bytes.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| err(format!("truncated reading {what}")))?;
        self.pos += 1;
        Ok(b)
    }

    fn bytes(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(err(format!("truncated reading {what}")));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// LEB128 `u64`; at most 10 bytes, and the value must fit 64 bits.
    fn uvarint_raw(&mut self, what: &str) -> Result<u64> {
        let mut v = 0u64;
        for i in 0..10 {
            let b = self.u8(what)?;
            if i == 9 && b > 1 {
                return Err(err(format!("{what}: varint overflows 64 bits")));
            }
            v |= ((b & 0x7F) as u64) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(err(format!("{what}: varint longer than 10 bytes")))
    }

    /// An unsigned varint bounded by [`MAX_SAFE`].
    fn uvarint(&mut self, what: &str) -> Result<u64> {
        let v = self.uvarint_raw(what)?;
        if v > MAX_SAFE as u64 {
            return Err(err(format!("{what} {v} exceeds 2^53 - 1")));
        }
        Ok(v)
    }

    /// A zigzag varint bounded by ±[`MAX_SAFE`].
    fn svarint(&mut self, what: &str) -> Result<i64> {
        let v = unzigzag(self.uvarint_raw(what)?);
        safe(v, what)
    }
}

fn safe(v: i64, what: &str) -> Result<i64> {
    if v.unsigned_abs() > MAX_SAFE as u64 {
        return Err(err(format!("{what} {v} is outside ±(2^53 - 1)")));
    }
    Ok(v)
}

/// `a + b` (or any `i64` result) bounded by ±[`MAX_SAFE`].
fn safe_add(a: i64, b: i64, what: &str) -> Result<i64> {
    // |a|, |b| <= MAX_SAFE, so the i64 sum cannot overflow.
    safe(a + b, what)
}

/// The decoded section header — exposed for diagnostics (`stt-optimize`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathCodecHeader {
    /// `flags` byte.
    pub flags: u8,
    /// Rows the section describes.
    pub feature_count: u64,
    /// Vertices the section describes.
    pub vertex_count: u64,
}

impl PathCodecHeader {
    /// [`FLAG_TIME`] is set.
    pub fn has_time(&self) -> bool {
        self.flags & FLAG_TIME != 0
    }
    /// [`FLAG_DECIMAL`] is set.
    pub fn is_decimal(&self) -> bool {
        self.flags & FLAG_DECIMAL != 0
    }
}

fn read_header(r: &mut Reader<'_>) -> Result<PathCodecHeader> {
    let version = r.u8("version")?;
    if version != PATH_CODEC_VERSION {
        return Err(err(format!(
            "unsupported version {version} (this reader knows {PATH_CODEC_VERSION})"
        )));
    }
    let flags = r.u8("flags")?;
    if flags & !(FLAG_TIME | FLAG_DECIMAL) != 0 {
        return Err(err(format!("reserved flag bits set ({flags:#04x})")));
    }
    let feature_count = r.uvarint("feature_count")?;
    let vertex_count = r.uvarint("vertex_count")?;
    Ok(PathCodecHeader {
        flags,
        feature_count,
        vertex_count,
    })
}

/// Parse only the fixed header (version, flags, the two counts) — what a
/// caller needs to check a section against its CORE batch before decoding it.
pub fn decode_header(bytes: &[u8]) -> Result<PathCodecHeader> {
    read_header(&mut Reader { buf: bytes, pos: 0 })
}

/// Decode `PATH_CODEC` section bytes.
///
/// Strict: every rejection listed in the spec is an `Err`, never a panic, and
/// no allocation is sized by a count the section's own length cannot back.
pub fn decode(bytes: &[u8]) -> Result<PathLayer> {
    let mut r = Reader { buf: bytes, pos: 0 };
    let h = read_header(&mut r)?;
    let has_time = h.has_time();
    let decimal = h.is_decimal();

    // Allocation caps: every feature costs at least two bytes (count + tag),
    // every vertex at least two (one residual per axis) plus two more with
    // TIME (dq + r). A count the remaining bytes cannot back is rejected
    // before anything is sized by it.
    let rem = r.remaining() as u64;
    if h.feature_count > rem / 2 {
        return Err(err(format!(
            "feature_count {} cannot fit in {rem} remaining bytes",
            h.feature_count
        )));
    }
    let per_vertex = if has_time { 4 } else { 2 };
    if h.vertex_count > rem / per_vertex {
        return Err(err(format!(
            "vertex_count {} cannot fit in {rem} remaining bytes",
            h.vertex_count
        )));
    }
    let nf = h.feature_count as usize;
    let nv = h.vertex_count as usize;

    let e = if decimal {
        let e = r.u8("decimal exponent")?;
        if e > MAX_EXPONENT {
            return Err(err(format!("decimal exponent {e} is above {MAX_EXPONENT}")));
        }
        Some(e)
    } else {
        None
    };
    let (unit, qbase) = if has_time {
        let unit = r.uvarint("time unit")? as i64;
        if unit < 1 {
            return Err(err("time unit must be >= 1"));
        }
        (unit, r.svarint("qbase")?)
    } else {
        (0, 0)
    };

    let mut counts = Vec::with_capacity(nf);
    let mut sum = 0u64;
    for _ in 0..nf {
        let c = r.uvarint("vertex count")?;
        sum += c; // each c <= 2^53, nf <= section length: no overflow
        if sum > h.vertex_count {
            return Err(err(format!(
                "feature vertex counts exceed vertex_count {}",
                h.vertex_count
            )));
        }
        counts.push(u32::try_from(c).map_err(|_| err("feature vertex count exceeds u32"))?);
    }
    if sum != h.vertex_count {
        return Err(err(format!(
            "feature vertex counts sum to {sum}, vertex_count is {}",
            h.vertex_count
        )));
    }
    let tags = r.bytes(nf, "predictor tags")?;
    for &t in tags {
        match t {
            TAG_DELTA | TAG_CONST_VEL => {}
            TAG_TIME_VEL if has_time => {}
            TAG_TIME_VEL => return Err(err("TIME_VEL tag in a section without TIME")),
            other => return Err(err(format!("unknown predictor tag {other}"))),
        }
    }

    // Times.
    let times = if has_time {
        let mut dq = Vec::with_capacity(nv);
        for _ in 0..nv {
            dq.push(r.svarint("dq")?);
        }
        let mut t = Vec::with_capacity(nv);
        let mut anchor = qbase;
        let mut n = 0usize;
        for &c in &counts {
            let mut prev = anchor;
            for j in 0..c as usize {
                let rem = r.uvarint("time remainder")? as i64;
                if rem >= unit {
                    return Err(err(format!("time remainder {rem} >= unit {unit}")));
                }
                let q = safe_add(prev, dq[n], "time quotient")?;
                let qu = q
                    .checked_mul(unit)
                    .filter(|v| v.unsigned_abs() <= MAX_SAFE as u64)
                    .ok_or_else(|| err("vertex time is outside ±(2^53 - 1)"))?;
                t.push(safe_add(qu, rem, "vertex time")?);
                prev = q;
                if j == 0 {
                    anchor = q;
                }
                n += 1;
            }
        }
        Some(t)
    } else {
        None
    };

    // Coordinates.
    let mut ints: Vec<[i64; 2]> = Vec::with_capacity(nv);
    let mut first = [0i64; 2];
    let mut s = 0usize;
    for (f, &c) in counts.iter().enumerate() {
        let tag = tags[f];
        let c = c as usize;
        for i in 0..c {
            let mut k = [0i64; 2];
            for (a, slot) in k.iter_mut().enumerate() {
                let res = r.svarint("residual")?;
                let pred = if i == 0 {
                    first[a]
                } else if i == 1 || tag == TAG_DELTA {
                    ints[s + i - 1][a]
                } else {
                    let (k1, k2) = (ints[s + i - 1][a], ints[s + i - 2][a]);
                    let v = k1 - k2; // |k| <= 2^53: no overflow
                    let t = times.as_deref().unwrap_or(&[]);
                    let step = if tag == TAG_TIME_VEL && t[s + i - 1] - t[s + i - 2] > 0 {
                        let x =
                            time_vel_term(v, t[s + i] - t[s + i - 1], t[s + i - 1] - t[s + i - 2]);
                        // NaN fails the comparison too: `x` is finite and
                        // in range, or the section is rejected.
                        if x.is_nan() || x.abs() > MAX_SAFE as f64 {
                            return Err(err("TIME_VEL prediction is outside ±(2^53 - 1)"));
                        }
                        x as i64
                    } else {
                        v
                    };
                    safe(k1 + step, "predictor")?
                };
                *slot = safe_add(pred, res, "coordinate integer")?;
                if !decimal && i32::try_from(*slot).is_err() {
                    return Err(err(format!(
                        "world-grid coordinate {} does not fit Int32",
                        *slot
                    )));
                }
            }
            ints.push(k);
        }
        if c > 0 {
            first = ints[s];
        }
        s += c;
    }

    let coords = match e {
        None => PathCoords::Grid(ints.iter().map(|k| [k[0] as i32, k[1] as i32]).collect()),
        Some(e) => {
            let p = POW10[e as usize];
            let mut f: Vec<[f64; 2]> = ints
                .iter()
                .map(|k| [k[0] as f64 / p, k[1] as f64 / p])
                .collect();
            let exc_count = r.uvarint("exception count")?;
            // Each exception costs at least three bytes (index + two halves).
            if exc_count > r.remaining() as u64 / 3 {
                return Err(err(format!(
                    "exception count {exc_count} cannot fit in the remaining bytes"
                )));
            }
            let values = 2 * nv as u64;
            let mut idx = Vec::with_capacity(exc_count as usize);
            let mut prev = 0u64;
            for i in 0..exc_count {
                let d = r.uvarint("exception index")?;
                let at = if i == 0 {
                    d
                } else {
                    if d == 0 {
                        return Err(err("exception indices are not strictly increasing"));
                    }
                    prev + d
                };
                if at >= values {
                    return Err(err(format!(
                        "exception index {at} out of range ({values} values)"
                    )));
                }
                idx.push(at as usize);
                prev = at;
            }
            for &at in &idx {
                let lo = r.uvarint("exception patch (low)")?;
                let hi = r.uvarint("exception patch (high)")?;
                if lo > u32::MAX as u64 || hi > u32::MAX as u64 {
                    return Err(err("exception patch half above 2^32 - 1"));
                }
                let slot = &mut f[at / 2][at % 2];
                *slot = f64::from_bits(slot.to_bits() ^ ((hi << 32) | lo));
            }
            PathCoords::Decimal(f)
        }
    };

    if r.remaining() != 0 {
        return Err(err(format!("{} trailing bytes", r.remaining())));
    }
    Ok(PathLayer {
        counts,
        coords,
        times,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(counts: Vec<u32>, coords: Vec<[i32; 2]>, times: Option<Vec<i64>>) -> PathLayer {
        PathLayer {
            counts,
            coords: PathCoords::Grid(coords),
            times,
        }
    }

    fn dec(counts: Vec<u32>, coords: Vec<[f64; 2]>, times: Option<Vec<i64>>) -> PathLayer {
        PathLayer {
            counts,
            coords: PathCoords::Decimal(coords),
            times,
        }
    }

    fn roundtrip(l: &PathLayer) -> Vec<u8> {
        let bytes = encode(l).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(&back, l, "round trip");
        assert_eq!(encode(l).unwrap(), bytes, "deterministic");
        bytes
    }

    /// The tags byte run, located from the header (test helper).
    fn tags_of(bytes: &[u8]) -> Vec<u8> {
        let l = decode(bytes).unwrap();
        // Re-walk the header to find the tag run.
        let mut r = Reader { buf: bytes, pos: 0 };
        let h = read_header(&mut r).unwrap();
        if h.is_decimal() {
            r.u8("e").unwrap();
        }
        if h.has_time() {
            r.uvarint("u").unwrap();
            r.svarint("q").unwrap();
        }
        for _ in 0..l.counts.len() {
            r.uvarint("c").unwrap();
        }
        r.bytes(l.counts.len(), "tags").unwrap().to_vec()
    }

    #[test]
    fn grid_round_trip_with_times() {
        let l = grid(
            vec![3, 0, 1, 4],
            vec![
                [10, 20],
                [12, 22],
                [14, 24],
                [-5, 7],
                [100, 100],
                [101, 99],
                [i32::MAX, i32::MIN],
                [i32::MIN, i32::MAX],
            ],
            Some(vec![
                1_700_000_000_000,
                1_700_000_060_000,
                1_700_000_120_000,
                -86_400_001,
                5,
                3, // unsorted within a feature
                -7,
                1_700_000_000_123,
            ]),
        );
        roundtrip(&l);
    }

    #[test]
    fn no_time_layers_round_trip() {
        roundtrip(&grid(
            vec![2, 2],
            vec![[0, 0], [1, 1], [5, 5], [6, 6]],
            None,
        ));
        roundtrip(&dec(
            vec![2],
            vec![[-122.419_4, 37.774_9], [-122.43, 37.78]],
            None,
        ));
        roundtrip(&grid(vec![], vec![], None));
        roundtrip(&grid(vec![0, 0, 0], vec![], None));
        roundtrip(&dec(vec![0], vec![], Some(vec![])));
    }

    #[test]
    fn single_vertex_and_empty_features() {
        roundtrip(&grid(
            vec![1, 0, 1, 0, 0, 1],
            vec![[3, 4], [-3, -4], [0, 0]],
            Some(vec![10, -10, 0]),
        ));
    }

    #[test]
    fn decimal_special_values_are_bit_exact() {
        let l = dec(
            vec![6, 2],
            vec![
                [-0.0, 0.0],
                [f64::NAN, f64::INFINITY],
                [f64::NEG_INFINITY, -f64::NAN],
                [f64::from_bits(0x7FF0_0000_0000_0001), 1e300],
                [f64::MIN_POSITIVE, -5e-324],
                [12.345_678, -98.765_432],
                [0.1 + 0.2, 1.0 / 3.0],
                [2.5, -2.5],
            ],
            Some(vec![0, 1, 2, 3, 4, 5, 6, 7]),
        );
        roundtrip(&l);
    }

    #[test]
    fn non_decimal_values_are_all_exceptions_and_still_exact() {
        // Irrational-looking doubles: none is exact at any exponent 3..=8.
        let coords: Vec<[f64; 2]> = (0..50)
            .map(|i| {
                let x = (i as f64 + 1.0).sqrt() * std::f64::consts::PI;
                [x, -x / 7.0]
            })
            .collect();
        let l = dec(vec![20, 30], coords, None);
        let bytes = roundtrip(&l);
        // e = 7 when nothing is exact.
        let mut r = Reader {
            buf: &bytes,
            pos: 0,
        };
        read_header(&mut r).unwrap();
        assert_eq!(r.u8("e").unwrap(), 7);
    }

    #[test]
    fn decimal_picks_the_smallest_good_exponent() {
        let coords: Vec<[f64; 2]> = (0..40)
            .map(|i| [-122.0 + i as f64 * 0.001, 37.0 + i as f64 * 0.002])
            .collect();
        let bytes = roundtrip(&dec(vec![40], coords, None));
        let mut r = Reader {
            buf: &bytes,
            pos: 0,
        };
        read_header(&mut r).unwrap();
        assert_eq!(r.u8("e").unwrap(), 3);
    }

    #[test]
    fn time_vel_is_chosen_for_irregular_sampling_at_constant_speed() {
        // Constant speed, irregular dt: only TIME_VEL predicts exactly.
        let dts = [1000i64, 3000, 500, 7000, 2000, 4000, 1000, 9000];
        let mut t = vec![1_600_000_000_000i64];
        for d in dts {
            t.push(t.last().unwrap() + d);
        }
        let coords: Vec<[i32; 2]> = t
            .iter()
            .map(|&ti| {
                let s = (ti - t[0]) as i32;
                [s * 3, -s * 2]
            })
            .collect();
        let n = coords.len() as u32;
        let bytes = roundtrip(&grid(vec![n], coords.clone(), Some(t.clone())));
        assert_eq!(tags_of(&bytes), vec![TAG_TIME_VEL]);
        // The same geometry without times can only pick DELTA/CONST_VEL.
        let bytes = roundtrip(&grid(vec![n], coords, None));
        assert_ne!(tags_of(&bytes), vec![TAG_TIME_VEL]);
    }

    #[test]
    fn time_vel_with_repeated_and_unsorted_times() {
        // dt0 == 0 and dt0 < 0 fall back to the constant-velocity step.
        let t = vec![0, 0, 10, 5, 5, 20, 40];
        let coords = vec![[0, 0], [1, 1], [3, 2], [2, 9], [8, 8], [9, 1], [30, -30]];
        let mut l = grid(vec![7], coords, Some(t));
        roundtrip(&l);
        // Force TIME_VEL through the round trip by hand-building a section.
        l.times = Some(vec![0, 1000, 2000, 2000, 1500, 9000, 9001]);
        roundtrip(&l);
    }

    #[test]
    fn constant_velocity_and_delta_are_chosen() {
        let cv: Vec<[i32; 2]> = (0..10).map(|i| [i * 700, i * -1100]).collect();
        let bytes = roundtrip(&grid(vec![10], cv, None));
        assert_eq!(tags_of(&bytes), vec![TAG_CONST_VEL]);
        let still: Vec<[i32; 2]> = (0..10).map(|_| [5, 5]).collect();
        let bytes = roundtrip(&grid(vec![10], still, None));
        assert_eq!(tags_of(&bytes), vec![TAG_DELTA]);
    }

    #[test]
    fn encoder_refuses_out_of_range_times_and_bad_input() {
        let l = grid(vec![1], vec![[0, 0]], Some(vec![MAX_TIME_MS + 1]));
        assert!(encode(&l).is_err());
        assert!(!times_in_range(&[i64::MIN]));
        roundtrip(&grid(
            vec![2],
            vec![[0, 0], [1, 1]],
            Some(vec![1 - MAX_TIME_MS, MAX_TIME_MS - 1]),
        ));
        assert!(encode(&grid(vec![1], vec![[0, 0]], Some(vec![MAX_TIME_MS]))).is_err());
        assert!(encode(&grid(vec![2], vec![[0, 0]], None)).is_err());
        assert!(encode(&grid(vec![1], vec![[0, 0]], Some(vec![]))).is_err());
    }

    /// Every listed rejection is a hard error.
    #[test]
    fn decoder_rejects_malformed_sections() {
        let l = dec(
            vec![2, 1],
            vec![[1.5, 2.5], [f64::NAN, 3.25], [0.1, -0.0]],
            Some(vec![1000, 2000, 3000]),
        );
        let good = encode(&l).unwrap();
        decode(&good).unwrap();

        let mut b = good.clone();
        b.push(0);
        assert!(decode(&b).unwrap_err().to_string().contains("trailing"));

        let mut b = good.clone();
        b[0] = 2;
        assert!(decode(&b).unwrap_err().to_string().contains("version"));

        let mut b = good.clone();
        b[1] |= 0x04;
        assert!(decode(&b).unwrap_err().to_string().contains("reserved"));

        for cut in 0..good.len() {
            assert!(decode(&good[..cut]).is_err(), "truncated at {cut}");
        }
        assert!(decode(&[]).is_err());

        // TIME_VEL without TIME; unknown tag.
        let nt = encode(&grid(vec![3], vec![[0, 0], [100, 100], [200, 200]], None)).unwrap();
        let tag_at = 1 + 1 + 1 + 1 + 1; // version, flags, nf, nv, count (1 byte each)
        assert_eq!(nt[tag_at], TAG_CONST_VEL);
        let mut b = nt.clone();
        b[tag_at] = TAG_TIME_VEL;
        assert!(decode(&b).unwrap_err().to_string().contains("TIME_VEL"));
        let mut b = nt.clone();
        b[tag_at] = 3;
        assert!(decode(&b)
            .unwrap_err()
            .to_string()
            .contains("unknown predictor tag"));

        // Count mismatch: vertex_count larger than the counts' sum.
        let mut b = nt.clone();
        b[3] = 2;
        assert!(decode(&b).is_err());
    }

    #[test]
    fn decoder_rejects_remainder_at_or_above_unit() {
        // Hand-built: TIME, 1 feature, 1 vertex, unit 10, qbase 0, count 1,
        // tag 0, dq 0, r = 10, residuals 0 0.
        let b = [1u8, FLAG_TIME, 1, 1, 10, 0, 1, 0, 0, 10, 0, 0];
        assert!(decode(&b).unwrap_err().to_string().contains("remainder"));
        let ok = [1u8, FLAG_TIME, 1, 1, 10, 0, 1, 0, 0, 9, 0, 0];
        let l = decode(&ok).unwrap();
        assert_eq!(l.times, Some(vec![9]));
        // Unit 0 is invalid.
        let b = [1u8, FLAG_TIME, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0];
        assert!(decode(&b).is_err());
    }

    #[test]
    fn decoder_rejects_bad_exceptions() {
        // DECIMAL, no time, 1 feature, 1 vertex, e = 0, count 1, tag 0, res 0 0.
        let head = [1u8, FLAG_DECIMAL, 1, 1, 0, 1, 0, 0, 0];
        let with = |tail: &[u8]| {
            let mut v = head.to_vec();
            v.extend_from_slice(tail);
            v
        };
        decode(&with(&[0])).unwrap();
        let l = decode(&with(&[1, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x08])).unwrap();
        // index 1 = y; patch hi = 2^31 → sign bit: y = -0.0.
        let PathCoords::Decimal(f) = l.coords else {
            panic!()
        };
        assert_eq!(f[0][1].to_bits(), (-0.0f64).to_bits());
        // Index out of range (2 values only).
        assert!(decode(&with(&[1, 2, 0, 0])).is_err());
        // Not strictly increasing.
        assert!(decode(&with(&[2, 0, 0, 0, 0, 0, 0])).is_err());
        // Patch half above 2^32 - 1.
        assert!(decode(&with(&[1, 0, 0x80, 0x80, 0x80, 0x80, 0x10, 0])).is_err());
        // Exponent above 15.
        let mut b = with(&[0]);
        b[4] = 16;
        assert!(decode(&b).is_err());
    }

    #[test]
    fn decoder_rejects_values_outside_their_type() {
        // World-grid coordinate beyond Int32: no time, 1 feature, 1 vertex,
        // count 1, tag 0, res x = 2^31 (zigzag 2^32), y = 0.
        let mut b = vec![1u8, 0, 1, 1, 1, 0];
        put_uvarint(&mut b, 1u64 << 32);
        b.push(0);
        assert!(decode(&b).unwrap_err().to_string().contains("Int32"));
        // A varint above 2^53 - 1.
        let mut b = vec![1u8, FLAG_DECIMAL, 1, 1, 0, 1, 0];
        put_svarint(&mut b, 1i64 << 53);
        b.extend_from_slice(&[0, 0]);
        assert!(decode(&b).is_err());
        // 11-byte varint.
        let b = [
            1u8, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0,
        ];
        assert!(decode(&b).is_err());
    }

    #[test]
    fn decoder_caps_allocations_by_section_length() {
        // A huge declared feature/vertex count on a tiny section must be
        // rejected before anything is allocated.
        let mut b = vec![1u8, 0];
        put_uvarint(&mut b, MAX_SAFE as u64);
        put_uvarint(&mut b, MAX_SAFE as u64);
        assert!(decode(&b).unwrap_err().to_string().contains("cannot fit"));
    }
}
