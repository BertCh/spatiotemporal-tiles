//! Geometry processing utilities
//!
//! This module provides functions for processing and simplifying geometric data
//! for use in spatiotemporal tiles. Uses standard geo/geo-types crates.

use geo::algorithm::simplify::Simplify;
use geo::{LineString, Polygon};

/// Simplify a linestring using the Douglas-Peucker algorithm
///
/// This uses the standard `geo` crate's Simplify trait for consistent,
/// well-tested simplification.
pub fn simplify_linestring(line: &LineString<f64>, epsilon: f64) -> LineString<f64> {
    let num_coords = line.coords().count();
    if num_coords <= 2 {
        return line.clone();
    }
    line.simplify(&epsilon)
}

/// Simplify a polygon using the Douglas-Peucker algorithm
///
/// This uses the standard `geo` crate's Simplify trait for consistent,
/// well-tested simplification.
pub fn simplify_polygon(polygon: &Polygon<f64>, epsilon: f64) -> Polygon<f64> {
    polygon.simplify(&epsilon)
}

/// Encode coordinates as integers using a tile extent
pub fn encode_coordinates(coords: &[(f64, f64)], extent: u32, tile_size: f64) -> Vec<u32> {
    let mut encoded = Vec::new();
    let mut last_x = 0i32;
    let mut last_y = 0i32;

    for (x, y) in coords {
        // Convert to tile coordinates
        let tile_x = ((x / tile_size) * extent as f64) as i32;
        let tile_y = ((y / tile_size) * extent as f64) as i32;

        // Delta encode
        let dx = tile_x - last_x;
        let dy = tile_y - last_y;

        encoded.push(zigzag_encode(dx));
        encoded.push(zigzag_encode(dy));

        last_x = tile_x;
        last_y = tile_y;
    }

    encoded
}

/// Decode integer-encoded coordinates
pub fn decode_coordinates(encoded: &[u32], extent: u32, tile_size: f64) -> Vec<(f64, f64)> {
    let mut coords = Vec::new();
    let mut x = 0i32;
    let mut y = 0i32;

    for chunk in encoded.chunks(2) {
        if chunk.len() == 2 {
            x += zigzag_decode(chunk[0]);
            y += zigzag_decode(chunk[1]);

            let world_x = (x as f64 / extent as f64) * tile_size;
            let world_y = (y as f64 / extent as f64) * tile_size;

            coords.push((world_x, world_y));
        }
    }

    coords
}

/// ZigZag encode a signed integer to an unsigned integer
fn zigzag_encode(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// ZigZag decode an unsigned integer to a signed integer
fn zigzag_decode(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

/// Calculate the bounding box of a linestring using geo types
pub fn bounding_box_line(line: &LineString<f64>) -> crate::types::BoundingBox {
    use geo::algorithm::bounding_rect::BoundingRect;

    if let Some(rect) = line.bounding_rect() {
        crate::types::BoundingBox::new(rect.min().x, rect.min().y, rect.max().x, rect.max().y)
    } else {
        crate::types::BoundingBox::default()
    }
}

/// Calculate the bounding box of a polygon using geo types
pub fn bounding_box_polygon(polygon: &Polygon<f64>) -> crate::types::BoundingBox {
    use geo::algorithm::bounding_rect::BoundingRect;

    if let Some(rect) = polygon.bounding_rect() {
        crate::types::BoundingBox::new(rect.min().x, rect.min().y, rect.max().x, rect.max().y)
    } else {
        crate::types::BoundingBox::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_encoding() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);
    }

    #[test]
    fn test_zigzag_decoding() {
        assert_eq!(zigzag_decode(0), 0);
        assert_eq!(zigzag_decode(1), -1);
        assert_eq!(zigzag_decode(2), 1);
        assert_eq!(zigzag_decode(3), -2);
        assert_eq!(zigzag_decode(4), 2);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        for i in -1000..1000 {
            assert_eq!(zigzag_decode(zigzag_encode(i)), i);
        }
    }

    #[test]
    fn test_simplify_linestring() {
        let line: LineString<f64> =
            vec![(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.1), (4.0, 0.0)].into();
        let simplified = simplify_linestring(&line, 0.5);
        let num_coords = simplified.coords().count();
        let orig_coords = line.coords().count();
        assert!(num_coords < orig_coords);
    }

}
