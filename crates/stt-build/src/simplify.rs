//! Line simplification for lower zoom levels
//!
//! Applies Douglas-Peucker or Visvalingam-Whyatt simplification to reduce
//! vertex count at lower zoom levels, improving both memory usage and
//! rendering performance.

use geo::{LineString, SimplifyVw};

/// Simplify coordinates for a given zoom level
///
/// Returns simplified coordinates with altitudes preserved through interpolation.
/// At high zoom levels (15+), no simplification is applied.
pub fn simplify_for_zoom(
    coords: &[(f64, f64, f64)],
    zoom: u8,
    simplify_max_zoom: u8,
) -> Vec<(f64, f64, f64)> {
    // Don't simplify if zoom is above threshold or too few points
    if zoom > simplify_max_zoom || coords.len() < 3 {
        return coords.to_vec();
    }

    // Calculate epsilon based on zoom level
    // Higher epsilon = more simplification
    // At zoom 0, ~150km resolution; at zoom 14, ~10m resolution
    let epsilon = calculate_epsilon(zoom);

    if epsilon <= 0.0 {
        return coords.to_vec();
    }

    // Convert to geo LineString (2D)
    let line: LineString<f64> = coords
        .iter()
        .map(|(x, y, _)| geo::Coord { x: *x, y: *y })
        .collect();

    // Apply Visvalingam-Whyatt simplification (better for preserving shape)
    let simplified = line.simplify_vw(&epsilon);

    // If simplification didn't help or made it too short, return original
    if simplified.0.len() < 2 || simplified.0.len() >= coords.len() {
        return coords.to_vec();
    }

    // Map simplified coords back with interpolated altitudes
    let mut result = Vec::with_capacity(simplified.0.len());

    for coord in simplified.0.iter() {
        // Find the closest original point to interpolate altitude
        let alt = interpolate_altitude(coords, coord.x, coord.y);
        result.push((coord.x, coord.y, alt));
    }

    result
}

/// Calculate simplification epsilon for a zoom level
fn calculate_epsilon(zoom: u8) -> f64 {
    // Epsilon in degrees - corresponds roughly to pixel resolution
    // tile_size = 256 pixels, world = 360 degrees
    // At zoom z: degrees_per_pixel = 360 / (256 * 2^z)
    match zoom {
        0..=6 => 0.01,    // ~1km resolution
        7..=9 => 0.002,   // ~200m resolution
        10..=11 => 0.0008, // ~80m resolution
        12..=13 => 0.0003, // ~30m resolution
        14 => 0.0001,      // ~10m resolution
        _ => 0.0,          // No simplification
    }
}

/// Interpolate altitude at a given lon/lat from original coordinates
fn interpolate_altitude(coords: &[(f64, f64, f64)], lon: f64, lat: f64) -> f64 {
    // Find the segment containing this point
    let mut best_alt = 0.0;
    let mut best_dist = f64::MAX;

    for window in coords.windows(2) {
        let (x1, y1, alt1) = window[0];
        let (x2, y2, alt2) = window[1];

        // Project point onto segment
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;

        let t = if len_sq > 0.0 {
            ((lon - x1) * dx + (lat - y1) * dy) / len_sq
        } else {
            0.0
        };

        let t = t.clamp(0.0, 1.0);

        // Point on segment
        let px = x1 + t * dx;
        let py = y1 + t * dy;

        // Distance from projected point
        let dist = (lon - px).powi(2) + (lat - py).powi(2);

        if dist < best_dist {
            best_dist = dist;
            best_alt = alt1 + t * (alt2 - alt1);
        }
    }

    // Also check individual points (for exact matches)
    for (x, y, alt) in coords {
        let dist = (lon - x).powi(2) + (lat - y).powi(2);
        if dist < best_dist {
            best_dist = dist;
            best_alt = *alt;
        }
    }

    best_alt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_simplification_high_zoom() {
        let coords = vec![
            (-122.4, 37.7, 0.0),
            (-122.41, 37.71, 10.0),
            (-122.42, 37.72, 20.0),
        ];

        // At zoom 16, should return original
        let result = simplify_for_zoom(&coords, 16, 14);
        assert_eq!(result.len(), coords.len());
    }

    #[test]
    fn test_simplification_low_zoom() {
        // Create a line with many points that can be simplified
        let mut coords = Vec::new();
        for i in 0..100 {
            let t = i as f64 / 100.0;
            // Slightly noisy line from SF to Oakland
            let noise = (i as f64 * 0.1).sin() * 0.0001;
            coords.push((
                -122.4 + t * 0.15 + noise,
                37.7 + t * 0.1 + noise,
                t * 100.0,
            ));
        }

        // At zoom 8, should simplify significantly
        let result = simplify_for_zoom(&coords, 8, 14);
        assert!(result.len() < coords.len(), "Should have fewer points");
        assert!(result.len() >= 2, "Should have at least 2 points");
    }

    #[test]
    fn test_preserves_altitude() {
        let coords = vec![
            (0.0, 0.0, 100.0),
            (0.5, 0.5, 200.0),
            (1.0, 1.0, 300.0),
        ];

        let result = simplify_for_zoom(&coords, 5, 14);

        // Check that altitudes are reasonable
        for (_, _, alt) in &result {
            assert!(*alt >= 100.0 && *alt <= 300.0, "Altitude should be in range");
        }
    }

    #[test]
    fn test_epsilon_decreases_with_zoom() {
        let eps10 = calculate_epsilon(10);
        let eps12 = calculate_epsilon(12);
        let eps14 = calculate_epsilon(14);

        assert!(eps10 > eps12, "Lower zoom should have larger epsilon");
        assert!(eps12 > eps14, "Lower zoom should have larger epsilon");
    }
}


