//! Geometry processing utilities
//!
//! This module provides functions for processing and simplifying geometric data
//! for use in spatiotemporal tiles. Uses standard geo/geo-types crates.
//!
//! Earlier revisions exposed `encode_coordinates` / `zigzag_encode` for a
//! quantized geometry encoding that was never wired into the v2 build path
//! (which emits real f64 lon/lat in Arrow). Those have been removed; the
//! format-v3 work (Track B) is rebuilding quantized coordinate encoding on
//! top of fixed-precision Arrow columns, not the old zigzag delta format.

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
    fn test_simplify_linestring() {
        let line: LineString<f64> =
            vec![(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.1), (4.0, 0.0)].into();
        let simplified = simplify_linestring(&line, 0.5);
        let num_coords = simplified.coords().count();
        let orig_coords = line.coords().count();
        assert!(num_coords < orig_coords);
    }

}
