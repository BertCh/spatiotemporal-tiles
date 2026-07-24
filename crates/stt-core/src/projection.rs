//! Coordinate projection utilities
//!
//! This module provides standardized coordinate transformations
//! for accurate Web Mercator projections.
//!
//! Web Mercator is the ONLY projection the tile pipeline needs — everything
//! here is closed-form arithmetic with no CRS library behind it. Reprojecting
//! non-4326 input is pushed down into SQL (`ST_Transform`, see the DB input
//! adaptors and `stt-serve`), which is why an optional libproj dependency and
//! its `projection` cargo feature were removed rather than kept as dead weight.

use crate::error::{Error, Result};
use geo_types::Point;

/// Convert WGS84 lon/lat to Web Mercator tile coordinates
///
/// # Arguments
/// * `lon` - Longitude in degrees (-180 to 180)
/// * `lat` - Latitude in degrees (-85.0511 to 85.0511)
/// * `zoom` - Tile zoom level (0-22)
///
/// # Returns
/// Tuple of (tile_x, tile_y) coordinates
pub fn lonlat_to_tile(lon: f64, lat: f64, zoom: u8) -> Result<(u32, u32)> {
    // Validate inputs
    if !(-180.0..=180.0).contains(&lon) {
        return Err(Error::InvalidCoordinates(zoom, 0, 0));
    }
    if !(-85.0511..=85.0511).contains(&lat) {
        return Err(Error::InvalidCoordinates(zoom, 0, 0));
    }

    let n = 1u32 << zoom;

    // Convert lon/lat to tile coordinates using Web Mercator formula
    // This is the standard used by OpenStreetMap, Google Maps, etc.
    let x = ((lon + 180.0) / 360.0 * n as f64).floor() as u32;
    let lat_rad = lat.to_radians();
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
}
