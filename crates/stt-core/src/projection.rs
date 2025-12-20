//! Coordinate projection utilities
//!
//! This module provides standardized coordinate transformations
//! for accurate Web Mercator projections.
//!
//! The basic Web Mercator transformations are always available.
//! With the `projection` feature, you can use the proj crate for
//! advanced transformations between different coordinate systems.

use crate::error::{Error, Result};
use geo_types::{Coord, Point};

#[cfg(feature = "projection")]
use proj::Proj;
#[cfg(feature = "projection")]
use std::sync::OnceLock;

#[cfg(feature = "projection")]
/// Cached projection from WGS84 to Web Mercator
static WGS84_TO_WEB_MERCATOR: OnceLock<Proj> = OnceLock::new();

#[cfg(feature = "projection")]
/// Cached projection from Web Mercator to WGS84  
static WEB_MERCATOR_TO_WGS84: OnceLock<Proj> = OnceLock::new();

#[cfg(feature = "projection")]
/// Get or initialize the WGS84 to Web Mercator projection
fn wgs84_to_web_mercator() -> &'static Proj {
    WGS84_TO_WEB_MERCATOR.get_or_init(|| {
        Proj::new_known_crs("EPSG:4326", "EPSG:3857", None)
            .expect("Failed to initialize WGS84 to Web Mercator projection")
    })
}

#[cfg(feature = "projection")]
/// Get or initialize the Web Mercator to WGS84 projection
fn web_mercator_to_wgs84() -> &'static Proj {
    WEB_MERCATOR_TO_WGS84.get_or_init(|| {
        Proj::new_known_crs("EPSG:3857", "EPSG:4326", None)
            .expect("Failed to initialize Web Mercator to WGS84 projection")
    })
}

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
    let n = (1u32 << zoom) as f64;

    let lon = (tile_x as f64 / n) * 360.0 - 180.0;
    let lat_rad = (1.0 - 2.0 * tile_y as f64 / n).sinh().atan();
    let lat = lat_rad.to_degrees();

    Point::new(lon, lat)
}

/// Convert tile-relative coordinates to WGS84 lon/lat
///
/// # Arguments
/// * `x` - X coordinate within tile (0 to extent)
/// * `y` - Y coordinate within tile (0 to extent)
/// * `zoom` - Tile zoom level
/// * `tile_x` - Tile X coordinate
/// * `tile_y` - Tile Y coordinate
/// * `extent` - Tile extent
///
/// # Returns
/// Point with lon/lat coordinates
pub fn tile_coords_to_lonlat(
    x: u32,
    y: u32,
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
    let lat_rad = (1.0 - 2.0 * world_y / n).sinh().atan();
    let lat = lat_rad.to_degrees();

    Point::new(lon, lat)
}

/// Convert WGS84 coordinates to Web Mercator meters (EPSG:3857)
///
/// With the `projection` feature, this uses the proj crate for accurate transformations.
/// Without it, uses a simple approximation.
#[cfg(feature = "projection")]
pub fn wgs84_to_meters(point: Point<f64>) -> Result<Point<f64>> {
    let proj = wgs84_to_web_mercator();
    let (x, y) = proj
        .convert((point.x(), point.y()))
        .map_err(|e| Error::Other(format!("Projection error: {}", e)))?;
    Ok(Point::new(x, y))
}

#[cfg(not(feature = "projection"))]
pub fn wgs84_to_meters(point: Point<f64>) -> Result<Point<f64>> {
    // Simple Web Mercator approximation (EPSG:3857)
    // This is accurate enough for most use cases
    const R: f64 = 6378137.0; // Earth radius in meters
    let x = R * point.x().to_radians();
    let y = R
        * (std::f64::consts::FRAC_PI_4 + point.y().to_radians() / 2.0)
            .tan()
            .ln();
    Ok(Point::new(x, y))
}

/// Convert Web Mercator meters (EPSG:3857) to WGS84 coordinates
///
/// With the `projection` feature, this uses the proj crate for accurate transformations.
/// Without it, uses a simple approximation.
#[cfg(feature = "projection")]
pub fn meters_to_wgs84(point: Point<f64>) -> Result<Point<f64>> {
    let proj = web_mercator_to_wgs84();
    let (x, y) = proj
        .convert((point.x(), point.y()))
        .map_err(|e| Error::Other(format!("Projection error: {}", e)))?;
    Ok(Point::new(x, y))
}

#[cfg(not(feature = "projection"))]
pub fn meters_to_wgs84(point: Point<f64>) -> Result<Point<f64>> {
    // Simple Web Mercator to WGS84 approximation
    const R: f64 = 6378137.0; // Earth radius in meters
    let lon = point.x().to_degrees() / R;
    let lat = (2.0 * (point.y() / R).exp().atan() - std::f64::consts::FRAC_PI_2).to_degrees();
    Ok(Point::new(lon, lat))
}

/// Calculate tile bounds in WGS84 coordinates
pub fn tile_bounds(tile_x: u32, tile_y: u32, zoom: u8) -> crate::types::BoundingBox {
    let nw = tile_to_lonlat(tile_x, tile_y, zoom);
    let se = tile_to_lonlat(tile_x + 1, tile_y + 1, zoom);

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
        let point = tile_to_lonlat(163, 395, 10);
        // Should be close to San Francisco
        assert!((point.x() + 122.34).abs() < 0.1);
        assert!((point.y() - 37.77).abs() < 0.1);
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
    fn test_wgs84_to_meters() {
        // Test San Francisco conversion
        let point = Point::new(-122.4194, 37.7749);
        let meters = wgs84_to_meters(point).unwrap();

        // Web Mercator coordinates should be in the millions
        assert!(meters.x().abs() > 10_000_000.0);
        assert!(meters.y().abs() > 4_000_000.0);

        // Roundtrip test (allow some tolerance for approximation)
        let back = meters_to_wgs84(meters).unwrap();
        assert!((back.x() - point.x()).abs() < 0.01); // ~1km tolerance
        assert!((back.y() - point.y()).abs() < 0.01);
    }

    #[test]
    fn test_tile_bounds() {
        let bounds = tile_bounds(163, 395, 10);

        // San Francisco should be within these bounds
        assert!(bounds.contains(-122.4194, 37.7749));
    }
}
