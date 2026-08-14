//! Space-time blob ordering for the packed writer.
//!
//! The on-disk *byte order* of tile blobs (independent of the directory index,
//! which must stay `(zoom, hilbert, time_start)` for the lookup codec) decides
//! how few HTTP range requests a client makes: a query box coalesces into one
//! request only when its tiles are byte-contiguous. The best order depends on
//! the dataset's space-vs-time aspect ratio, so the writer takes a
//! [`BlobOrdering`] knob; the default is [`BlobOrdering::Hilbert3`].
//!
//! All five orderings keep **zoom as the primary key** — the directory codec
//! requires zoom-major, and a reorder pass only permutes blobs *within* a zoom.

use std::fmt;
use std::str::FromStr;

/// Cube-side cap (in bits) for the 3D curves: `3 * side` must fit in a u64.
pub const CURVE_BITS_CAP: u32 = 21;

/// Cardinality margin (in bits) within which the space and time axes count as
/// "balanced" for [`BlobOrdering::choose`]. Time must dominate space by MORE
/// than this many bits before the heuristic switches away from the 3D-Hilbert
/// generalist.
pub const ORDERING_BALANCE_BITS: u32 = 3;

/// How tile blobs are ordered on disk by the buffered writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlobOrdering {
    /// `(zoom, hilbert, time_start)` — 2D spatial Hilbert, time secondary. Keeps
    /// a cell's whole timeline contiguous; best for pure playback/scrub.
    SpatialMajor,
    /// `(zoom, time_bucket, hilbert)` — all of one time slice contiguous; best
    /// for wide-space-at-one-instant (pan) on time-deep data.
    TimeMajor,
    /// 3D Hilbert curve over `(x, y, time_bucket)` at native per-axis
    /// resolution. The best generalist — no catastrophic query — and the
    /// measured winner on the widest range of datasets. This is the enum
    /// `Default`.
    ///
    /// The `stt-build` CLI defaults to `--blob-ordering measured`: the concrete
    /// order is picked by simulating per-ordering range-read cost over the
    /// dataset's own tiles ([`crate::ordering_sim`]), under per-dataset query
    /// weights. `--blob-ordering auto` selects the cheap cardinality heuristic
    /// ([`choose`](BlobOrdering::choose)) instead, and is also what `measured`
    /// degrades to on inputs below
    /// [`MIN_TILES_TO_SIMULATE`](crate::ordering_sim::MIN_TILES_TO_SIMULATE).
    /// There is no no-reorder mode — the packed writer always reorders before
    /// cutting packs.
    #[default]
    Hilbert3,
    /// 3D Morton / Z-order over `(x, y, time_bucket)`. Cheaper to compute than
    /// Hilbert but its long jumps hurt locality; empirically **never** the
    /// measured-optimal ordering on any tested dataset — kept for research /
    /// comparison, not production use.
    Morton3,
    /// Resolve the concrete ordering at finalize time from the dataset's
    /// space-vs-time cardinality via [`BlobOrdering::choose`]. A "resolve
    /// later" marker, never a key itself — the writer maps it to a concrete
    /// order before sorting blobs, so [`space_time_key`] never sees it.
    Auto,
}

impl BlobOrdering {
    /// Pick a concrete ordering from the dataset's space-vs-time shape.
    ///
    /// `space_bits` is the **occupied** spatial extent in bits
    /// (`bits_for(max(x_span, y_span) + 1)` over the native tier) — NOT the raw
    /// max zoom, which for sparse data massively overstates spatial cardinality.
    /// `time_bits` is `ceil(log2(#occupied time buckets))`. Both axes are thus
    /// measured the same way (occupied extent), so the comparison is symmetric.
    /// The rule is **measured, not from first principles**:
    ///
    /// - **shallow time** (`time_bits ≤ 1`, i.e. a snapshot / ≤2 buckets) →
    ///   [`SpatialMajor`]. With no real temporal axis, the 3D curve's collapsed
    ///   time dimension only hurts locality; a pure 2D spatial Hilbert wins.
    /// - **time dominates** by more than [`ORDERING_BALANCE_BITS`]
    ///   (e.g. drifters: 2⁴ cells × 2281 buckets) → [`SpatialMajor`]. The
    ///   dominant access on time-deep data is *playback* (scrub time at a fixed
    ///   viewport), which wants each cell's whole timeline byte-contiguous.
    /// - **balanced or space-dominant** (e.g. flights: 2¹⁰ × 24 buckets) →
    ///   [`Hilbert3`], the robust generalist with no catastrophic query.
    ///
    /// This is the `--blob-ordering auto` path. It is no longer the CLI default
    /// — `measured` ([`crate::ordering_sim`]) is, since the workload model gave
    /// the simulator a playback query to rank with — but this rule is unchanged
    /// and still load-bearing in two places: it is what `auto` means, what
    /// `order-audit` reports as `auto_choice`, and the fallback `measured`
    /// resolves to when the input is too small to simulate.
    ///
    /// [`SpatialMajor`]: BlobOrdering::SpatialMajor
    /// [`Hilbert3`]: BlobOrdering::Hilbert3
    pub fn choose(space_bits: u32, time_bits: u32) -> BlobOrdering {
        if time_bits <= 1 {
            // Shallow time (≤2 buckets): no temporal axis worth interleaving —
            // a pure 2D spatial Hilbert keeps space contiguous. A 3D Hilbert
            // over a collapsed time axis is degenerate here: the dead axis buys
            // nothing and costs spatial locality.
            BlobOrdering::SpatialMajor
        } else if time_bits > space_bits + ORDERING_BALANCE_BITS {
            // Time dominates — keep each cell's whole timeline contiguous.
            BlobOrdering::SpatialMajor
        } else {
            // Balanced or space-dominant — the 3D-Hilbert generalist.
            BlobOrdering::Hilbert3
        }
    }
}

impl BlobOrdering {
    pub fn as_str(self) -> &'static str {
        match self {
            BlobOrdering::SpatialMajor => "spatial",
            BlobOrdering::TimeMajor => "time-major",
            BlobOrdering::Hilbert3 => "hilbert3",
            BlobOrdering::Morton3 => "morton3",
            BlobOrdering::Auto => "auto",
        }
    }
}

impl fmt::Display for BlobOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BlobOrdering {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "spatial" | "spatial-major" | "hilbert" | "hilbert2" => Ok(BlobOrdering::SpatialMajor),
            "time" | "time-major" | "timemajor" => Ok(BlobOrdering::TimeMajor),
            "hilbert3" | "h3" | "3d-hilbert" => Ok(BlobOrdering::Hilbert3),
            "morton3" | "morton" | "z-order" | "zorder" => Ok(BlobOrdering::Morton3),
            "auto" => Ok(BlobOrdering::Auto),
            other => Err(format!(
                "unknown blob ordering '{other}' (want: spatial|time-major|hilbert3|morton3|auto)"
            )),
        }
    }
}

/// Bits needed to represent values `0..n-1`.
pub fn bits_for(n: u64) -> u32 {
    if n <= 1 {
        0
    } else {
        64 - (n - 1).leading_zeros()
    }
}

/// 3D Morton / Z-order: interleave the low `bits` of x,y,t.
/// bit i of x -> 3i, y -> 3i+1, t -> 3i+2.
pub fn morton3(x: u32, y: u32, t: u32, bits: u32) -> u64 {
    if bits == 0 {
        return 0; // single cell -> index 0
    }
    let mut key: u64 = 0;
    for i in 0..bits {
        key |= (((x >> i) & 1) as u64) << (3 * i)
            | (((y >> i) & 1) as u64) << (3 * i + 1)
            | (((t >> i) & 1) as u64) << (3 * i + 2);
    }
    key
}

/// 3D Hilbert distance via Skilling's transpose algorithm (n=3).
/// Input axes must already be in `[0, 2^bits)`. Returns the scalar Hilbert index.
pub fn hilbert3(x: u32, y: u32, t: u32, bits: u32) -> u64 {
    if bits == 0 {
        return 0; // single cell -> index 0 (guards the `1 << (bits-1)` underflow)
    }
    let n = 3usize;
    let mut coords = [x, y, t];
    // --- inverse undo / transpose (Skilling) ---
    let mut q: u32 = 1 << (bits - 1);
    while q > 1 {
        let p = q - 1;
        for i in 0..n {
            if coords[i] & q != 0 {
                coords[0] ^= p; // invert
            } else {
                let s = (coords[0] ^ coords[i]) & p;
                coords[0] ^= s;
                coords[i] ^= s; // exchange
            }
        }
        q >>= 1;
    }
    // --- Gray encode ---
    for i in 1..n {
        coords[i] ^= coords[i - 1];
    }
    let mut t_acc: u32 = 0;
    let mut q2: u32 = 1 << (bits - 1);
    while q2 > 1 {
        if coords[n - 1] & q2 != 0 {
            t_acc ^= q2 - 1;
        }
        q2 >>= 1;
    }
    for c in coords.iter_mut() {
        *c ^= t_acc;
    }
    // --- interleave transpose words, MSB-first, axis-major, into the distance ---
    let mut key: u64 = 0;
    let mut bit = bits;
    while bit > 0 {
        bit -= 1;
        for i in 0..n {
            key = (key << 1) | (((coords[i] >> bit) & 1) as u64);
        }
    }
    key
}

/// Cube side (in bits) for the 3D curve at a given zoom: each axis at NATIVE
/// resolution — `zoom` bits of space vs `ceil(log2(#buckets))` bits of time —
/// so the curve interleaves space and time bit-for-bit at matching scales.
/// Space dominates the high bits when spatial extent exceeds temporal (and vice
/// versa). Capped so `3*side` fits in a u64.
pub fn curve_bits(zoom: u8, tb_span: i64) -> u32 {
    let tbits = bits_for((tb_span.max(0) + 1) as u64);
    (zoom as u32).max(tbits).min(CURVE_BITS_CAP)
}

/// Native-resolution axes for the 3D curves: x,y in `[0,2^zoom)`, time bucket
/// (relative to `tb_min`) in `[0,2^tbits)`. Drops low bits only if an axis
/// exceeds the (capped) cube side.
fn scale_axes(zoom: u8, x: u32, y: u32, tb: i64, tb_min: i64, bits: u32) -> (u32, u32, u32) {
    // checked_shr returns None when the shift >= bit-width (only reachable for an
    // absurd zoom >= 53 or pathological bucket count); collapse those axes to 0
    // rather than panic.
    let sdrop = (zoom as u32).saturating_sub(bits);
    let xs = x.checked_shr(sdrop).unwrap_or(0);
    let ys = y.checked_shr(sdrop).unwrap_or(0);
    let rel = (tb - tb_min).max(0) as u64;
    let tbits = bits_for(rel + 1);
    let tdrop = tbits.saturating_sub(bits);
    let qt = rel.checked_shr(tdrop).unwrap_or(0) as u32;
    (xs, ys, qt)
}

/// The 4-tuple sort key that lays a tile's blob down in `ordering` byte order.
/// `tb` is the tile's time-bucket index (`time_start / bucket_ms`), `tb_min` /
/// `tb_span` are the min and span of bucket indices across the whole archive.
/// Zoom is always the leading component (directory codec requirement).
#[allow(clippy::too_many_arguments)]
pub fn space_time_key(
    ordering: BlobOrdering,
    zoom: u8,
    x: u32,
    y: u32,
    hilbert: u64,
    time_start: i64,
    tb: i64,
    tb_min: i64,
    tb_span: i64,
) -> (u8, u64, u64, u64) {
    match ordering {
        BlobOrdering::SpatialMajor => (zoom, hilbert, time_start.max(0) as u64, 0),
        BlobOrdering::TimeMajor => (zoom, tb.max(0) as u64, hilbert, 0),
        BlobOrdering::Hilbert3 => {
            let bits = curve_bits(zoom, tb_span);
            let (xs, ys, qt) = scale_axes(zoom, x, y, tb, tb_min, bits);
            (zoom, hilbert3(xs, ys, qt, bits), 0, 0)
        }
        BlobOrdering::Morton3 => {
            let bits = curve_bits(zoom, tb_span);
            let (xs, ys, qt) = scale_axes(zoom, x, y, tb, tb_min, bits);
            (zoom, morton3(xs, ys, qt, bits), 0, 0)
        }
        // Auto is resolved to a concrete ordering by the writer before sorting;
        // if one ever reaches here, fall back to the Hilbert3 generalist.
        BlobOrdering::Auto => {
            let bits = curve_bits(zoom, tb_span);
            let (xs, ys, qt) = scale_axes(zoom, x, y, tb, tb_min, bits);
            (zoom, hilbert3(xs, ys, qt, bits), 0, 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton3_roundtrip() {
        fn de(key: u64, bits: u32) -> (u32, u32, u32) {
            let (mut x, mut y, mut t) = (0u32, 0u32, 0u32);
            for i in 0..bits {
                x |= (((key >> (3 * i)) & 1) as u32) << i;
                y |= (((key >> (3 * i + 1)) & 1) as u32) << i;
                t |= (((key >> (3 * i + 2)) & 1) as u32) << i;
            }
            (x, y, t)
        }
        for &(x, y, t) in &[(0u32, 0u32, 0u32), (1, 2, 3), (7, 7, 7), (5, 0, 6)] {
            assert_eq!(de(morton3(x, y, t, 3), 3), (x, y, t));
        }
    }

    #[test]
    fn hilbert3_is_bijection() {
        let mut seen = vec![false; 512];
        for x in 0..8 {
            for y in 0..8 {
                for t in 0..8 {
                    let h = hilbert3(x, y, t, 3) as usize;
                    assert!(h < 512, "index {h} out of range");
                    assert!(!seen[h], "collision at {h}");
                    seen[h] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn hilbert3_locality() {
        // Hilbert's defining property: consecutive indices are grid neighbours.
        let bits = 4u32;
        let n = 1u32 << bits;
        let mut pts: Vec<(u64, (u32, u32, u32))> = Vec::new();
        for x in 0..n {
            for y in 0..n {
                for t in 0..n {
                    pts.push((hilbert3(x, y, t, bits), (x, y, t)));
                }
            }
        }
        pts.sort_by_key(|p| p.0);
        for w in pts.windows(2) {
            let (a, b) = (w[0].1, w[1].1);
            let d = (a.0 as i64 - b.0 as i64).abs()
                + (a.1 as i64 - b.1 as i64).abs()
                + (a.2 as i64 - b.2 as i64).abs();
            assert_eq!(d, 1, "consecutive Hilbert cells must be adjacent (got {d})");
        }
    }

    #[test]
    fn curve_bits_balances_native_resolution() {
        // flights-like: zoom 10, 24 buckets -> space 10 bits sets the side.
        assert_eq!(curve_bits(10, 23), 10);
        assert_eq!(bits_for(24), 5);
        // drifters-like: zoom 4, 2281 buckets -> time 12 bits sets the side.
        assert_eq!(curve_bits(4, 2280), 12);
        assert_eq!(bits_for(2281), 12);
    }

    #[test]
    fn scale_axes_keeps_native_ranges() {
        let bits = curve_bits(10, 23);
        let (_, _, qt) = scale_axes(10, 512, 300, 10, 0, bits);
        assert!(
            qt < (1 << bits_for(24)),
            "time axis must stay native ~5-bit, got {qt}"
        );
        let (xs, ys, _) = scale_axes(10, 777, 123, 0, 0, bits);
        assert_eq!((xs, ys), (777, 123));
    }

    #[test]
    fn bits_zero_is_safe() {
        // zoom 0 + single time bucket -> curve_bits == 0; the curves must return
        // 0, not panic on `1 << (bits-1)`.
        assert_eq!(curve_bits(0, 0), 0);
        assert_eq!(hilbert3(0, 0, 0, 0), 0);
        assert_eq!(morton3(0, 0, 0, 0), 0);
        let k = space_time_key(BlobOrdering::Hilbert3, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(k, (0, 0, 0, 0));
        // absurd zoom must not overflow-shift either: the axis narrowing is a
        // checked shift, so a zoom past the key's bit budget saturates.
        let _ = space_time_key(BlobOrdering::Hilbert3, 60, 1 << 30, 1 << 30, 0, 0, 0, 0, 0);
    }

    #[test]
    fn ordering_parse_roundtrip() {
        for o in [
            BlobOrdering::SpatialMajor,
            BlobOrdering::TimeMajor,
            BlobOrdering::Hilbert3,
            BlobOrdering::Morton3,
            BlobOrdering::Auto,
        ] {
            assert_eq!(o.as_str().parse::<BlobOrdering>().unwrap(), o);
        }
        assert!("garbage".parse::<BlobOrdering>().is_err());
        assert_eq!(BlobOrdering::default(), BlobOrdering::Hilbert3);
    }

    #[test]
    fn choose_matches_measured_winners() {
        // flights: 2^10 spatial cells, ~24 time buckets -> balanced/space-heavy
        // -> the 3D-Hilbert generalist (measured best on flights).
        assert_eq!(
            BlobOrdering::choose(10, bits_for(24)),
            BlobOrdering::Hilbert3
        );
        // drifters: 2^4 spatial cells, 2281 buckets -> time dominates by far
        // -> spatial-major keeps each cell's timeline contiguous for playback
        // (measured ~3x better than hilbert3, ~5x better than time-major).
        assert_eq!(
            BlobOrdering::choose(4, bits_for(2281)),
            BlobOrdering::SpatialMajor
        );
        // Exactly at the balance margin stays on the generalist.
        assert_eq!(
            BlobOrdering::choose(10, 10 + ORDERING_BALANCE_BITS),
            BlobOrdering::Hilbert3
        );
        // One bit past the margin flips to spatial-major.
        assert_eq!(
            BlobOrdering::choose(10, 10 + ORDERING_BALANCE_BITS + 1),
            BlobOrdering::SpatialMajor
        );
        // Shallow time (a snapshot — 1 or 2 buckets) → spatial-major, even when
        // space is wide: a 3D curve over a collapsed time axis is degenerate.
        assert_eq!(BlobOrdering::choose(10, 0), BlobOrdering::SpatialMajor);
        assert_eq!(BlobOrdering::choose(10, 1), BlobOrdering::SpatialMajor);
    }

    #[test]
    fn keys_distinguish_orderings() {
        // Same cell, two adjacent time buckets: SpatialMajor keeps them adjacent
        // (time is the low key), TimeMajor separates them by the whole space.
        let k0 = space_time_key(BlobOrdering::SpatialMajor, 10, 5, 5, 42, 0, 0, 0, 100);
        let k1 = space_time_key(BlobOrdering::SpatialMajor, 10, 5, 5, 42, 1, 1, 0, 100);
        assert_eq!(k0.0, k1.0);
        assert_eq!(k0.1, k1.1); // same hilbert -> contiguous under spatial-major
        let t0 = space_time_key(BlobOrdering::TimeMajor, 10, 5, 5, 42, 0, 0, 0, 100);
        let t1 = space_time_key(BlobOrdering::TimeMajor, 10, 5, 5, 42, 1, 1, 0, 100);
        assert_ne!(t0.1, t1.1); // different buckets -> separated under time-major
    }
}
