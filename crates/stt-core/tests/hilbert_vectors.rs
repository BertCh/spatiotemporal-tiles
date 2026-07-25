//! Normative Hilbert-curve test vectors.
//!
//! The directory sort key and the wire `Δhilbert` column depend on one exact
//! curve definition, specified in `docs/spec/stt-packed-format.md` §4
//! ("The Hilbert key"). This test pins that definition two ways:
//!
//! 1. the concrete `(z, x, y) → h` vectors published in the spec are asserted
//!    against the real `TileId::hilbert_index` — the spec table and this table
//!    are the same table, and must stay byte-for-byte in sync;
//! 2. a self-contained reference implementation of the spec's pseudocode is
//!    cross-checked against `TileId::hilbert_index` exhaustively at low zooms
//!    and on a deterministic sample at z8/z14, proving the prose algorithm and
//!    the shipped implementation (`hilbert_2d::xy2h_discrete`,
//!    `Variant::Hilbert`) describe the same curve.

use stt_core::TileId;

/// The normative test vectors from `docs/spec/stt-packed-format.md` §4.
/// (z, x, y, h) — XYZ tile coordinates (top-left origin), h = Hilbert key.
const SPEC_VECTORS: &[(u8, u32, u32, u64)] = &[
    // z0 — single tile, special-cased to 0
    (0, 0, 0, 0),
    // z1 — the four quadrants in curve order
    (1, 0, 0, 0),
    (1, 0, 1, 1),
    (1, 1, 1, 2),
    (1, 1, 0, 3),
    // z2 — corners + interior cells
    (2, 0, 0, 0),
    (2, 0, 3, 5),
    (2, 1, 2, 7),
    (2, 3, 3, 10),
    (2, 2, 1, 13),
    (2, 3, 0, 15),
    // z8 — corners, center, interior
    (8, 0, 0, 0),
    (8, 0, 255, 21845),
    (8, 255, 255, 43690),
    (8, 255, 0, 65535),
    (8, 128, 128, 32768),
    (8, 100, 200, 28272),
    // z14 — corners, center, interior
    (14, 0, 0, 0),
    (14, 0, 16383, 89478485),
    (14, 16383, 16383, 178956970),
    (14, 16383, 0, 268435455),
    (14, 8192, 8192, 134217728),
    (14, 12345, 6789, 214230386),
];

/// Reference implementation of the spec §4 pseudocode ("The Hilbert key") —
/// the classic bottom-up rotate-and-accumulate form (Wikipedia `xy2d`),
/// deliberately written from the spec text rather than shared with the crate.
fn reference_hilbert(z: u8, x: u32, y: u32) -> u64 {
    if z == 0 {
        return 0;
    }
    let n: u64 = 1 << z;
    let (mut x, mut y) = (x as u64, y as u64);
    let mut h: u64 = 0;
    let mut s: u64 = n / 2;
    while s > 0 {
        let rx: u64 = if x & s > 0 { 1 } else { 0 };
        let ry: u64 = if y & s > 0 { 1 } else { 0 };
        h += s * s * ((3 * rx) ^ ry);
        // rotate/reflect the quadrant so the remaining low bits are in the
        // canonical frame (n-1-x is a pure bit complement: n-1 is all ones)
        if ry == 0 {
            if rx == 1 {
                x = (n - 1) - x;
                y = (n - 1) - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    h
}

#[test]
fn spec_vectors_match_implementation() {
    for &(z, x, y, expected) in SPEC_VECTORS {
        let actual = TileId::new(z, x, y, 0).hilbert_index();
        assert_eq!(
            actual, expected,
            "hilbert_index(z={z}, x={x}, y={y}) = {actual}, spec says {expected}"
        );
    }
}

#[test]
fn spec_vectors_match_reference_algorithm() {
    for &(z, x, y, expected) in SPEC_VECTORS {
        let actual = reference_hilbert(z, x, y);
        assert_eq!(
            actual, expected,
            "reference_hilbert(z={z}, x={x}, y={y}) = {actual}, spec says {expected}"
        );
    }
}

#[test]
fn reference_algorithm_matches_implementation_exhaustively() {
    // Exhaustive at z1..=6 (5,460 tiles) — every rotation state is exercised.
    for z in 1u8..=6 {
        let n = 1u32 << z;
        for y in 0..n {
            for x in 0..n {
                assert_eq!(
                    reference_hilbert(z, x, y),
                    TileId::new(z, x, y, 0).hilbert_index(),
                    "divergence at z={z}, x={x}, y={y}"
                );
            }
        }
    }
    // Deterministic strided sample at the deeper spec-vector zooms.
    for &(z, stride) in &[(8u8, 17u32), (14u8, 1021u32)] {
        let n = 1u32 << z;
        for y in (0..n).step_by(stride as usize) {
            for x in (0..n).step_by(stride as usize) {
                assert_eq!(
                    reference_hilbert(z, x, y),
                    TileId::new(z, x, y, 0).hilbert_index(),
                    "divergence at z={z}, x={x}, y={y}"
                );
            }
        }
    }
}

#[test]
fn curve_endpoints_and_adjacency() {
    // The curve starts at (0,0) (h = 0), ends at (2^z - 1, 0) (h = 4^z - 1),
    // and consecutive h values are edge-adjacent tiles — the locality property
    // the directory's range coalescing rests on.
    for z in 1u8..=5 {
        let n = 1u64 << z;
        assert_eq!(TileId::new(z, 0, 0, 0).hilbert_index(), 0);
        assert_eq!(
            TileId::new(z, (n - 1) as u32, 0, 0).hilbert_index(),
            n * n - 1
        );
        // invert via exhaustive scan, then walk the curve
        let mut by_h = vec![(0u32, 0u32); (n * n) as usize];
        for y in 0..n as u32 {
            for x in 0..n as u32 {
                by_h[TileId::new(z, x, y, 0).hilbert_index() as usize] = (x, y);
            }
        }
        for w in by_h.windows(2) {
            let (x0, y0) = w[0];
            let (x1, y1) = w[1];
            let manhattan = x0.abs_diff(x1) + y0.abs_diff(y1);
            assert_eq!(
                manhattan, 1,
                "curve jump between ({x0},{y0}) and ({x1},{y1}) at z={z}"
            );
        }
    }
}
