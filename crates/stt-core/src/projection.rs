//! Coordinate projection utilities
//!
//! This module provides standardized coordinate transformations
//! for accurate Web Mercator projections, plus the pipeline's single
//! [`MERCATOR_MAX_LAT`] and its single [`haversine_distance`]. Both must stay
//! single-sourced here: a per-call-site copy that drifts is a tier that
//! disagrees with its neighbours about where (or whether) a feature exists.
//!
//! Web Mercator is the ONLY projection the tile pipeline needs — everything
//! here is closed-form arithmetic with no CRS library behind it. Reprojecting
//! non-4326 input is pushed down into SQL (`ST_Transform`, see the DB input
//! adaptors and `stt-serve`), which is why there is no libproj dependency and
//! no `projection` cargo feature to carry.

use crate::error::{Error, Result};
use geo_types::Point;

/// Web Mercator's latitude limit in degrees: `atan(sinh(π))`, the latitude
/// whose projected `y` is exactly the top edge of the world square. The
/// projection runs to infinity at the poles, so every slippy-map tiler stops
/// here.
///
/// THE single latitude limit for the whole pipeline, and the whole pipeline
/// CLAMPS to it rather than rejecting beyond it. Both halves are load-bearing:
/// a tier that rejects at a rounded `85.0511` while another clamps at
/// `85.051_128_78` puts every feature in the ~3.2 m band between the two values
/// into the summary tier and out of the native tier — one dataset answering
/// "does this feature exist?" two different ways. Clamping (not rejecting) is
/// the side every site is on — the H3 anchor, the Quadbin cell, the trajectory
/// supercover, and the TS renderer's `MAX_MERCATOR_LAT`
/// (`poopdeck:packages/core/src/geo/mercator.ts`) — and it honours the no-thinning
/// rule: a polar feature lands on the world's edge row instead of vanishing.
///
/// The full precision matters *because* we clamp — the clamped feature lands
/// on the world edge itself, where the rounded `85.0511` puts it 3.2 m inside.
/// The value is `atan(sinh(π))` rounded INWARD, one ulp below the nearest f64,
/// and that ulp is not fussiness: the nearest f64
/// (`85.051_128_779_806_604`) projects to `y = -1.1e-16`, and a consumer that
/// floors to an integer cell and drops negatives (the trajectory supercover in
/// `stt-build`'s `clip.rs`) then returns NO tiles at all for a polar path. One
/// ulp inward puts `y` at `+2.2e-16` — still the world edge, but on the inside
/// of it. It is also bit-for-bit the value the renderer clamps to
/// (`MAX_MERCATOR_LAT` in `poopdeck:packages/core/src/geo/mercator.ts`), so producer and
/// consumer agree.
pub const MERCATOR_MAX_LAT: f64 = 85.051_128_779_806_59;

/// Convert WGS84 lon/lat to Web Mercator tile coordinates
///
/// # Arguments
/// * `lon` - Longitude in degrees (-180 to 180; outside is an error)
/// * `lat` - Latitude in degrees; CLAMPED to ±[`MERCATOR_MAX_LAT`], so a polar
///   feature lands in the edge tile row rather than being dropped
/// * `zoom` - Tile zoom level (0-22)
///
/// # Returns
/// Tuple of (tile_x, tile_y) coordinates
pub fn lonlat_to_tile(lon: f64, lat: f64, zoom: u8) -> Result<(u32, u32)> {
    // Non-finite check FIRST, and separate from the range check below: `f64::
    // clamp` propagates NaN and `NaN as u32` saturates to 0, so a NaN latitude
    // would otherwise sail through the clamp and file the feature into tile
    // (0, 0) as a phantom. Longitude is rejected rather than clamped —
    // out-of-range longitude is a wrap (an antimeridian problem with its own
    // splitter), not a projection singularity.
    if !lon.is_finite() || !lat.is_finite() {
        return Err(Error::InvalidCoordinates(zoom, 0, 0));
    }
    if !(-180.0..=180.0).contains(&lon) {
        return Err(Error::InvalidCoordinates(zoom, 0, 0));
    }
    let lat = lat.clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT);

    let n = 1u32 << zoom;

    // Convert lon/lat to tile coordinates using Web Mercator formula
    // This is the standard used by OpenStreetMap, Google Maps, etc.
    let x = ((lon + 180.0) / 360.0 * n as f64).floor() as u32;
    let lat_rad = lat.to_radians();
    // At exactly ±MERCATOR_MAX_LAT the world `y` is 0 (or `n`) up to one ulp of
    // rounding, so this can land a hair outside [0, n). Both guards are
    // deliberate: the `as u32` cast saturates a negative to 0 (Rust >= 1.45),
    // and `.min(n - 1)` catches the south edge's `n`.
    let y = ((1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n as f64).floor() as u32;

    Ok((x.min(n - 1), y.min(n - 1)))
}

/// Convert WGS84 lon/lat to tile-relative coordinates
///
/// # Arguments
/// * `lon` - Longitude in degrees
/// * `lat` - Latitude in degrees
/// * `zoom` - Tile zoom level
/// * `tile_x` - Tile X coordinate
/// * `tile_y` - Tile Y coordinate
/// * `extent` - Tile extent (typically 4096)
///
/// # Returns
/// Tuple of (x, y) coordinates within the tile.
/// Note: Values CAN be outside [0, extent] for coordinates outside the tile.
/// This is intentional - it allows paths that cross tile boundaries to be
/// rendered correctly. deck.gl handles clipping at render time.
pub fn lonlat_to_tile_coords(
    lon: f64,
    lat: f64,
    zoom: u8,
    tile_x: u32,
    tile_y: u32,
    extent: u32,
) -> (i32, i32) {
    let n = 1u32 << zoom;

    // Convert to Web Mercator tile coordinates (0-n)
    let world_x = (lon + 180.0) / 360.0 * n as f64;
    let lat_rad = lat.to_radians();
    let world_y = (1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n as f64;

    // Convert to tile-relative coordinates
    // DO NOT CLAMP - allow values outside [0, extent] for cross-tile paths
    let tile_rel_x = (world_x - tile_x as f64) * extent as f64;
    let tile_rel_y = (world_y - tile_y as f64) * extent as f64;

    (tile_rel_x.round() as i32, tile_rel_y.round() as i32)
}

/// Convert tile coordinates back to WGS84 lon/lat
///
/// # Arguments
/// * `tile_x` - Tile X coordinate
/// * `tile_y` - Tile Y coordinate
/// * `zoom` - Tile zoom level
///
/// # Returns
/// Point with lon/lat coordinates
pub fn tile_to_lonlat(tile_x: u32, tile_y: u32, zoom: u8) -> Point<f64> {
    // `2^zoom` via `powi` rather than `1u32 << zoom`: a corrupt zoom >= 32
    // (reachable from an untrusted paged-directory descriptor) would shift-
    // overflow. For every valid zoom (0..=31) this yields the identical value.
    let n = 2f64.powi(zoom as i32);

    let lon = (tile_x as f64 / n) * 360.0 - 180.0;
    // Web Mercator inverse: lat = atan(sinh(π·(1 - 2y/n))).
    // The π factor is the exact inverse of the forward projection, which
    // divides by π (see `lonlat_to_tile`). Omitting it badly distorts latitude.
    let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * tile_y as f64 / n))
        .sinh()
        .atan();
    let lat = lat_rad.to_degrees();

    Point::new(lon, lat)
}

/// Geographic bounding box `(min_lon, min_lat, max_lon, max_lat)` of a tile.
///
/// The tile's NW corner is `tile_to_lonlat(x, y)` and its SE corner is
/// `tile_to_lonlat(x + 1, y + 1)` (lon grows eastward with x; lat shrinks
/// southward with y, which is Web-Mercator-flipped). `x + 1` / `y + 1` are the
/// adjacent tile *edges* — at the last column/row they evaluate to the world
/// edge (lon 180 / lat ≈ -85.05), which is exactly the tile's far edge.
///
/// Used to build the paged-directory page descriptors (geo-bbox pruning) and is
/// the inverse-projection counterpart the TS reader does not need (the reader
/// compares stored bboxes against its lon/lat viewport directly).
pub fn tile_geo_bounds(zoom: u8, tile_x: u32, tile_y: u32) -> (f64, f64, f64, f64) {
    let nw = tile_to_lonlat(tile_x, tile_y, zoom);
    // `saturating_add`: a corrupt `x`/`y` == u32::MAX (from an untrusted
    // descriptor) would overflow the `+ 1` for the adjacent tile edge.
    let se = tile_to_lonlat(tile_x.saturating_add(1), tile_y.saturating_add(1), zoom);
    (nw.x(), se.y(), se.x(), nw.y())
}

/// Convert tile-relative coordinates to WGS84 lon/lat
///
/// # Arguments
/// * `x` - X coordinate within tile (signed: values outside [0, extent] are
///   valid for coordinates that belong to cross-tile paths)
/// * `y` - Y coordinate within tile (signed, same as `x`)
/// * `zoom` - Tile zoom level
/// * `tile_x` - Tile X coordinate
/// * `tile_y` - Tile Y coordinate
/// * `extent` - Tile extent
///
/// # Returns
/// Point with lon/lat coordinates
///
/// Note: `x`/`y` are `i32` to be the exact inverse of [`lonlat_to_tile_coords`],
/// which produces signed quantized coordinates that may fall outside the tile.
pub fn tile_coords_to_lonlat(
    x: i32,
    y: i32,
    zoom: u8,
    tile_x: u32,
    tile_y: u32,
    extent: u32,
) -> Point<f64> {
    let n = (1u32 << zoom) as f64;

    // Convert from tile-relative to world coordinates
    let world_x = tile_x as f64 + (x as f64 / extent as f64);
    let world_y = tile_y as f64 + (y as f64 / extent as f64);

    let lon = (world_x / n) * 360.0 - 180.0;
    // Web Mercator inverse: lat = atan(sinh(π·(1 - 2y/n))). The π factor is
    // the exact inverse of `lonlat_to_tile_coords`, which divides by π.
    let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * world_y / n))
        .sinh()
        .atan();
    let lat = lat_rad.to_degrees();

    Point::new(lon, lat)
}

/// Great-circle distance between two WGS84 points, in metres (haversine on a
/// spherical Earth of radius 6 371 km).
///
/// The pipeline's ONE copy, and it must stay that way: the `stt-build` callers
/// (`clip.rs` + `columnar.rs`) feed this into *timestamp interpolation along a
/// path*, so any drift between duplicate implementations shifts where a trip is
/// drawn at a given instant depending only on which code path built it.
/// Spherical (not ellipsoidal) is deliberate and sufficient: every consumer
/// uses the value as a RATIO along a path, where the ~0.5% sphere-vs-WGS84
/// error cancels.
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS: f64 = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    EARTH_RADIUS * 2.0 * a.sqrt().asin()
}

/// Calculate tile bounds in WGS84 coordinates
pub fn tile_bounds(tile_x: u32, tile_y: u32, zoom: u8) -> crate::types::BoundingBox {
    let nw = tile_to_lonlat(tile_x, tile_y, zoom);
    // `saturating_add`: a corrupt `x`/`y` == u32::MAX (from an untrusted
    // descriptor) would overflow the `+ 1` for the adjacent tile edge.
    let se = tile_to_lonlat(tile_x.saturating_add(1), tile_y.saturating_add(1), zoom);

    crate::types::BoundingBox::new(
        nw.x(), // min_lon
        se.y(), // min_lat (y is flipped)
        se.x(), // max_lon
        nw.y(), // max_lat
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lonlat_to_tile() {
        // Test center of world at zoom 0
        let (x, y) = lonlat_to_tile(0.0, 0.0, 0).unwrap();
        assert_eq!(x, 0);
        assert_eq!(y, 0);

        // Test San Francisco at zoom 10
        let (x, y) = lonlat_to_tile(-122.4194, 37.7749, 10).unwrap();
        assert_eq!(x, 163);
        assert_eq!(y, 395);
    }

    #[test]
    fn test_tile_to_lonlat() {
        // tile_to_lonlat returns the NW corner of the tile. Tile (163,395) at
        // zoom 10 has its NW corner near SF (the corner, not the tile centre).
        let point = tile_to_lonlat(163, 395, 10);
        assert!((point.x() + 122.695).abs() < 0.01, "lon = {}", point.x());
        assert!((point.y() - 37.996).abs() < 0.01, "lat = {}", point.y());

        // Round-trip: lonlat -> tile -> lonlat-of-corner must land back in the
        // same tile when re-projected.
        let (tx, ty) = lonlat_to_tile(-122.4194, 37.7749, 10).unwrap();
        assert_eq!((tx, ty), (163, 395));
    }

    #[test]
    fn test_lonlat_to_tile_coords() {
        let (x, y) = lonlat_to_tile_coords(-122.4194, 37.7749, 10, 163, 395, 4096);
        // Should be somewhere in the middle of the tile
        assert!(x > 0 && x < 4096);
        assert!(y > 0 && y < 4096);
    }

    #[test]
    fn test_tile_coords_roundtrip() {
        let original_lon = -122.4194;
        let original_lat = 37.7749;
        let zoom = 10;
        let extent = 4096;

        let (tile_x, tile_y) = lonlat_to_tile(original_lon, original_lat, zoom).unwrap();
        let (x, y) =
            lonlat_to_tile_coords(original_lon, original_lat, zoom, tile_x, tile_y, extent);
        let point = tile_coords_to_lonlat(x, y, zoom, tile_x, tile_y, extent);

        // Should be very close to original
        assert!((point.x() - original_lon).abs() < 0.001);
        assert!((point.y() - original_lat).abs() < 0.001);
    }

    #[test]
    fn test_tile_bounds() {
        let bounds = tile_bounds(163, 395, 10);

        // San Francisco should be within these bounds
        assert!(bounds.contains(-122.4194, 37.7749));
    }

    /// Continuous Web-Mercator `y` in world-tile units at `zoom` — the value
    /// `lonlat_to_tile` floors, and the value the trajectory supercover walks.
    fn world_y(lat: f64, zoom: u8) -> f64 {
        let n = (1u32 << zoom) as f64;
        (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n
    }

    /// `MERCATOR_MAX_LAT` is `atan(sinh(π))` rounded INWARD — not the rounded
    /// `85.0511` (3.2 m short, and the exact width of the band in which a
    /// native and a summary tier would disagree about a feature's existence),
    /// and not the nearest f64 either: that one projects to `y = -1.1e-16`,
    /// which a consumer that floors and drops negative cells reads as
    /// "no tiles".
    #[test]
    fn mercator_max_lat_is_the_true_limit_rounded_inward() {
        let nearest = std::f64::consts::PI.sinh().atan().to_degrees();
        assert_eq!(
            MERCATOR_MAX_LAT,
            f64::from_bits(nearest.to_bits() - 1),
            "constant must be exactly one ulp inside atan(sinh(pi))"
        );
        // The property that one ulp buys, at both ends of the zoom range.
        for z in [0u8, 20] {
            assert!(
                world_y(MERCATOR_MAX_LAT, z) >= 0.0,
                "z{z}: the clamp must not project outside the world square"
            );
            assert!(world_y(-MERCATOR_MAX_LAT, z) <= (1u32 << z) as f64);
            assert!(
                world_y(nearest, z) < 0.0,
                "z{z}: pins WHY we round inward — the nearest f64 goes negative"
            );
        }
        // The rounded value it replaced sat >3 m short of the real limit.
        assert!((MERCATOR_MAX_LAT - 85.0511) * 111_320.0 > 3.0);
    }

    /// The clamp boundary, pinned from three sides. Exactly at the limit, a
    /// hair inside, and far outside all project into the SAME edge tile row —
    /// nothing is rejected for being polar, which is what keeps the native and
    /// summary tiers agreeing about which features exist.
    #[test]
    fn latitude_is_clamped_at_the_mercator_limit_not_rejected() {
        for z in [0u8, 1, 6, 14] {
            let n = 1u32 << z;

            // Exactly the limit → the top row, never out of grid.
            let (_, y_at) = lonlat_to_tile(0.0, MERCATOR_MAX_LAT, z).expect("limit is projectable");
            assert_eq!(y_at, 0, "z{z}: lat == +limit must land in row 0");

            // Just inside (one ulp below) → the same row.
            let inside = MERCATOR_MAX_LAT - f64::EPSILON * MERCATOR_MAX_LAT;
            let (_, y_in) = lonlat_to_tile(0.0, inside, z).expect("just inside is projectable");
            assert_eq!(y_in, 0, "z{z}: just inside the limit must land in row 0");

            // Just OUTSIDE, and far outside (the pole): clamped, not dropped.
            let outside = MERCATOR_MAX_LAT + 1e-9;
            let (_, y_out) = lonlat_to_tile(0.0, outside, z).expect("beyond the limit is clamped");
            assert_eq!(y_out, 0, "z{z}: beyond the limit must clamp into row 0");
            let (_, y_pole) = lonlat_to_tile(0.0, 90.0, z).expect("the pole is clamped");
            assert_eq!(y_pole, 0, "z{z}: the north pole must clamp into row 0");

            // Same three, mirrored: the south edge is the LAST row, not `n`.
            let (_, y_s) = lonlat_to_tile(0.0, -MERCATOR_MAX_LAT, z).expect("south limit");
            assert_eq!(y_s, n - 1, "z{z}: lat == -limit must land in the last row");
            let (_, y_s_out) = lonlat_to_tile(0.0, -90.0, z).expect("south pole is clamped");
            assert_eq!(
                y_s_out,
                n - 1,
                "z{z}: the south pole must clamp to the last row"
            );
        }
    }

    /// Clamping must not swallow the genuinely-unprojectable: NaN propagates
    /// through `f64::clamp` and `NaN as u32` saturates to 0, so without an
    /// explicit finite check a NaN feature would silently land in tile (0, 0).
    #[test]
    fn non_finite_and_out_of_range_longitude_are_still_errors() {
        assert!(lonlat_to_tile(0.0, f64::NAN, 8).is_err());
        assert!(lonlat_to_tile(f64::NAN, 0.0, 8).is_err());
        assert!(lonlat_to_tile(0.0, f64::INFINITY, 8).is_err());
        assert!(lonlat_to_tile(f64::NEG_INFINITY, 0.0, 8).is_err());
        assert!(lonlat_to_tile(180.001, 0.0, 8).is_err());
    }

    #[test]
    fn haversine_matches_known_distances() {
        // One degree of latitude at the equator ≈ 111.19 km on this sphere.
        let d = haversine_distance(0.0, 0.0, 1.0, 0.0);
        assert!((d - 111_194.9).abs() < 1.0, "1 deg lat = {d} m");
        // Identical points are exactly zero (the ratio denominators rely on it).
        assert_eq!(haversine_distance(45.0, -122.0, 45.0, -122.0), 0.0);
        // Symmetric.
        assert_eq!(
            haversine_distance(37.7749, -122.4194, 40.7128, -74.0060),
            haversine_distance(40.7128, -74.0060, 37.7749, -122.4194)
        );
    }
}
