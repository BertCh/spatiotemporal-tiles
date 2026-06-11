//! CARTO Quadbin cell encoding — the build-side inverse of the TS decoder
//! in `packages/deck.gl/src/quadbin-cell.ts`.
//!
//! The summary tier (see [`crate::summary`]) can index aggregate cells by
//! either Uber H3 or CARTO Quadbin. The H3 path leans on the `h3o` crate for
//! its `(lat, lon) -> CellIndex -> u64` mapping; Quadbin has no equivalent
//! dependency in the build graph, so this module carries the small, exact
//! encoder. The renderer's `quadbinToTile` decoder is the inverse — the two
//! must agree bit-for-bit, which the unit tests pin against CARTO's published
//! reference value `tile_to_quadbin(0, 0, 0) == 0x480fffffffffffff`.
//!
//! THE CARTO QUADBIN u64 LAYOUT (bits, MSB→LSB, per the CARTO spec — see the
//! matching doc comment in `quadbin-cell.ts`):
//! ```text
//!   63        : reserved/sign (0)
//!   62 .. 60  : header   = 0b100  (value 4)   — Quadbin marker
//!   59        : mode     = 1                  — "tile" mode
//!   58 .. 57  : mode-dep = 0
//!   56 .. 52  : zoom z   (5 bits, 0..26)
//!   51 .. 0   : 52-bit interleaved (Morton) x/y, LEFT-aligned for the cell's
//!               zoom (the low (52 - 2z) bits are a `1…1` fill so distinct
//!               cells stay distinct as fixed-width integers).
//! ```

/// The clamp the Web-Mercator projection imposes on latitude — beyond this the
/// `tan`/`asinh` mapping diverges. Matches the slippy-map / OSM limit.
const MERCATOR_LAT_LIMIT: f64 = 85.051_128_78;

/// Header bits (`0b100` at bits 62..60) OR'd with the "tile" mode bit (bit 59):
/// `0x4800000000000000`. Every encoded Quadbin carries this constant prefix.
const QUADBIN_HEADER: u64 = 0x4800_0000_0000_0000;

/// Convert WGS84 lon/lat to standard Web-Mercator XYZ tile coordinates at
/// `z`. This is the same slippy-map tiling used everywhere else in the
/// pipeline (cf. [`stt_core::projection::lonlat_to_tile`]) but the latitude is
/// clamped to the Mercator limit rather than rejected, so an aggregate cell is
/// always assigned even for a feature sitting exactly at the pole.
pub fn lonlat_to_tile(lon: f64, lat: f64, z: u8) -> (u32, u32) {
    let n = 1u64 << z;
    let n_f = n as f64;
    let lat = lat.clamp(-MERCATOR_LAT_LIMIT, MERCATOR_LAT_LIMIT);
    let lon = lon.clamp(-180.0, 180.0);

    let x = ((lon + 180.0) / 360.0 * n_f).floor();
    let lat_rad = lat.to_radians();
    let y = ((1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n_f).floor();

    // Clamp into [0, n-1]: a lon of exactly +180 lands on x == n, and rounding
    // at the Mercator limit can put y on the n boundary.
    let max = (n - 1) as f64;
    let x = x.clamp(0.0, max) as u32;
    let y = y.clamp(0.0, max) as u32;
    (x, y)
}

/// Spread the low 26 bits of `v` across even bit positions (`v_i -> bit 2i`),
/// the encode half of the magic-number Morton interleave. The inverse of the
/// TS `deinterleave`.
fn interleave(v: u32) -> u64 {
    let mut x = v as u64 & 0x0000_0000_03ff_ffff; // keep low 26 bits (z ≤ 26)
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// CARTO Quadbin encode: pack `(z, x, y)` into its u64 cell index. The exact
/// inverse of the renderer's `quadbinToTile`.
///
/// The 52-bit payload is the Morton interleave of `x` (even bits) and `y` (odd
/// bits), occupying the TOP `2z` bits of the field and LEFT-aligned by shifting
/// up `52 - 2z`. The dead low `52 - 2z` bits are filled with `1`s — CARTO's
/// fixed-width convention so two cells at different zooms never collide.
pub fn tile_to_quadbin(z: u8, x: u32, y: u32) -> u64 {
    debug_assert!(z <= 26, "Quadbin zoom out of band: {z}");
    let z = z as u64;
    let interleaved = interleave(x) | (interleave(y) << 1);
    // Left-align the 2z significant bits within the 52-bit field.
    let morton_shift = 52 - 2 * z;
    let payload = (interleaved << morton_shift) & 0x000f_ffff_ffff_ffff;
    // Low-bit fill: (1 << morton_shift) - 1 ones below the significant bits.
    let fill = if morton_shift == 0 {
        0
    } else {
        (1u64 << morton_shift) - 1
    };
    QUADBIN_HEADER | (z << 52) | payload | fill
}

/// Convenience: assign a lon/lat directly to its Quadbin cell at zoom `z`.
pub fn lonlat_to_quadbin(lon: f64, lat: f64, z: u8) -> u64 {
    let (x, y) = lonlat_to_tile(lon, lat, z);
    tile_to_quadbin(z, x, y)
}

/// CARTO Quadbin decode: recover `(z, x, y)` from a u64 cell index. The exact
/// inverse of [`tile_to_quadbin`] and a 1:1 port of the renderer's
/// `quadbinToTile`. Used to recover a representative tile centre for the
/// summary-tile point geometry.
pub fn quadbin_to_tile(q: u64) -> (u8, u32, u32) {
    let z = ((q >> 52) & 0x1f) as u8;
    // Left-aligned payload: shift the dead low (52 - 2z) fill bits out.
    let morton_shift = 52u32.saturating_sub(2 * z as u32);
    let interleaved = (q & 0x000f_ffff_ffff_ffff) >> morton_shift;
    let x = deinterleave(interleaved);
    let y = deinterleave(interleaved >> 1);
    (z, x, y)
}

/// Compact the even bits of a Morton-coded value down into a dense integer —
/// the decode half of the interleave, inverse of [`interleave`].
fn deinterleave(coded: u64) -> u32 {
    let mut v = coded & 0x5555_5555_5555_5555;
    v = (v | (v >> 1)) & 0x3333_3333_3333_3333;
    v = (v | (v >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    v = (v | (v >> 4)) & 0x00ff_00ff_00ff_00ff;
    v = (v | (v >> 8)) & 0x0000_ffff_0000_ffff;
    v = (v | (v >> 16)) & 0x0000_0000_ffff_ffff;
    v as u32
}

/// Representative lon/lat for a Quadbin cell — the centre of its XYZ tile.
/// This mirrors the H3 path's use of the cell centroid as the point geometry
/// carried on the summary tile (for picking and "no-cell-rendering" fallback).
pub fn quadbin_centroid(z: u8, x: u32, y: u32) -> [f64; 2] {
    let n = (1u64 << z) as f64;
    let lon = ((x as f64 + 0.5) / n) * 360.0 - 180.0;
    let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * (y as f64 + 0.5) / n))
        .sinh()
        .atan();
    [lon, lat_rad.to_degrees()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standalone re-implementation of the TS `quadbinToTile` decode, written
    /// independently of the module's own [`quadbin_to_tile`] so the round-trip
    /// test pins the encoder against the published bit layout rather than
    /// against our own decoder (which could share a bug). Uses the magic-number
    /// de-interleave straight from `quadbin-cell.ts`.
    fn decode_from_ts_layout(q: u64) -> (u8, u32, u32) {
        fn deint(coded: u64) -> u32 {
            let mut v = coded & 0x5555_5555_5555_5555;
            v = (v | (v >> 1)) & 0x3333_3333_3333_3333;
            v = (v | (v >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
            v = (v | (v >> 4)) & 0x00ff_00ff_00ff_00ff;
            v = (v | (v >> 8)) & 0x0000_ffff_0000_ffff;
            v = (v | (v >> 16)) & 0x0000_0000_ffff_ffff;
            v as u32
        }
        let z = ((q >> 52) & 0x1f) as u8;
        let morton_shift = (52 - 2 * z as u32) as u64;
        let interleaved = (q & ((1u64 << 52) - 1)) >> morton_shift;
        let x = deint(interleaved);
        let y = deint(interleaved >> 1);
        (z, x, y)
    }

    #[test]
    fn carto_reference_value() {
        // The canonical CARTO root-tile fixture.
        assert_eq!(tile_to_quadbin(0, 0, 0), 0x480f_ffff_ffff_ffff);
    }

    #[test]
    fn header_and_zoom_bits_are_well_formed() {
        // Header nibble + mode bit are constant; zoom occupies bits 52..56.
        for z in 0u8..=26 {
            let q = tile_to_quadbin(z, 0, 0);
            assert_eq!(q >> 60, 0b100, "header nibble wrong at z={z}");
            assert_eq!((q >> 59) & 1, 1, "mode bit wrong at z={z}");
            assert_eq!(((q >> 52) & 0x1f) as u8, z, "zoom field wrong at z={z}");
        }
    }

    #[test]
    fn round_trips_through_decode() {
        // A spread of zooms and coordinates, including the z=25/z=26 edges
        // (max x/y at zoom z is 2^z - 1).
        let cases: &[(u8, u32, u32)] = &[
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 1),
            (2, 3, 1),
            (10, 163, 395), // San Francisco @ z10 (see projection tests)
            (14, 8191, 5000),
            (25, (1u32 << 25) - 1, 0),
            (25, 0, (1u32 << 25) - 1),
            (26, (1u32 << 26) - 1, (1u32 << 26) - 1),
            (26, 12_345_678, 9_876_543),
        ];
        for &(z, x, y) in cases {
            let q = tile_to_quadbin(z, x, y);
            // Independent decode straight from the published TS bit layout.
            assert_eq!(
                decode_from_ts_layout(q),
                (z, x, y),
                "TS-layout round-trip failed for (z={z}, x={x}, y={y}) -> {q:#018x}"
            );
            // And the module's own public decoder agrees.
            assert_eq!(
                quadbin_to_tile(q),
                (z, x, y),
                "module decode round-trip failed for (z={z}, x={x}, y={y}) -> {q:#018x}"
            );
        }
    }

    #[test]
    fn lonlat_to_tile_sanity() {
        // Null Island at z=1 sits on the (1,1) tile boundary — the floor of the
        // exact half lands on tile index 1 in both axes.
        assert_eq!(lonlat_to_tile(0.0, 0.0, 1), (1, 1));

        // San Francisco at z=10 matches the slippy-map fixture used by the
        // core projection tests (tile 163/395).
        assert_eq!(lonlat_to_tile(-122.4194, 37.7749, 10), (163, 395));

        // Antimeridian / pole clamp: extreme inputs stay in-band, not n.
        let n = (1u32 << 5) - 1;
        let (x, y) = lonlat_to_tile(180.0, 89.0, 5);
        assert!(x <= n && y <= n, "clamped tile out of band: ({x},{y})");
        let (x, y) = lonlat_to_tile(-180.0, -89.0, 5);
        assert!(x <= n && y <= n, "clamped tile out of band: ({x},{y})");
    }

    #[test]
    fn lonlat_to_quadbin_matches_two_step() {
        let (x, y) = lonlat_to_tile(-122.4194, 37.7749, 10);
        assert_eq!(
            lonlat_to_quadbin(-122.4194, 37.7749, 10),
            tile_to_quadbin(10, x, y)
        );
    }

    #[test]
    fn centroid_lands_inside_its_tile() {
        // The recovered centroid must re-tile to the same (x, y) cell.
        for &(z, x, y) in &[(2u8, 1u32, 2u32), (10, 163, 395), (14, 8191, 5000)] {
            let [lon, lat] = quadbin_centroid(z, x, y);
            assert_eq!(lonlat_to_tile(lon, lat, z), (x, y));
        }
    }
}
