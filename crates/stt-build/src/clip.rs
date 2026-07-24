//! Trajectory clipping for spatiotemporal tile distribution
//!
//! This module clips LineString trajectories at tile boundaries to ensure
//! features are properly distributed across spatial tiles. This enables
//! efficient viewport-based loading where only relevant segments are fetched.
//!
//! Performance considerations:
//! - Bounding box pre-filter to minimize clip operations
//! - Liang-Barsky algorithm for efficient line-rectangle clipping
//! - Timestamp interpolation based on distance along path
//! - Arc-wrapped properties for zero-copy sharing across segments
//! - Optional line simplification for lower zoom levels

use crate::input::SharedProperties;
use crate::simplify::{simplify_for_zoom_with, simplify_td_tr_for_zoom_with};
use geojson::{Feature, Geometry, Value as GeomValue};
use std::collections::HashSet;

/// A clipped segment of a trajectory assigned to a specific tile
#[derive(Debug, Clone)]
pub struct ClippedSegment {
    /// The tile coordinates this segment belongs to
    pub tile_x: u32,
    pub tile_y: u32,
    pub zoom: u8,
    /// The clipped geometry (subset of original coordinates)
    pub coordinates: Vec<(f64, f64, f64)>, // (lon, lat, alt)
    /// Per-vertex timestamps for this segment
    pub timestamps: Vec<u64>,
    /// Per-vertex scalar values for this segment (e.g. SST), aligned with
    /// `coordinates`. Empty when the producer supplied none; `NaN` marks an
    /// individual vertex with no value.
    pub vertex_values: Vec<f32>,
    /// Per-vertex × per-bucket value matrix for this segment, `[vertex][bucket]`,
    /// aligned with `coordinates`. Empty when the producer supplied none. Each
    /// bucket channel is resampled at clip boundaries exactly like
    /// `vertex_values`. Flattened vertex-major into the tile's
    /// `vertex_value_matrix` column.
    pub vertex_value_matrix: Vec<Vec<f32>>,
    /// Start timestamp of this segment
    pub start_time: u64,
    /// End timestamp of this segment
    pub end_time: u64,
    /// Shared reference to original properties (zero-copy via Arc)
    pub properties: Option<SharedProperties>,
    /// Original feature ID for client-side reconnection
    pub feature_id: Option<geojson::feature::Id>,
}

/// Configuration for trajectory clipping
#[derive(Debug, Clone)]
pub struct ClipConfig {
    /// Minimum number of vertices to bother clipping
    pub min_vertices: usize,
    /// Buffer in degrees to add around tile bounds (prevents gaps at boundaries)
    pub buffer_degrees: f64,
    /// Buffer for POLYGON coverage clipping — 0 (exact tile rect), unlike the
    /// trajectory `buffer_degrees` above. Lines need the overlap for stroke
    /// joins at cut points; fills must NOT overlap: adjacent tiles clipping
    /// the same ring at the same tile edge emit bit-identical seam vertices,
    /// so unbuffered pieces rasterize watertight, while a buffered strip
    /// double-blends under any `opacity < 1` (no renderer masks tiles at
    /// draw time) — and the world-anchored quant grid preserves the seam
    /// identity across tiles.
    pub polygon_buffer_degrees: f64,
    /// Optional temporal granularity for slicing long trajectories (in milliseconds)
    /// If set, trajectories crossing temporal boundaries will be split
    pub temporal_granularity_ms: Option<u64>,
    /// Enable line simplification for lower zoom levels
    pub simplify: bool,
    /// Maximum zoom level to apply simplification (higher zooms keep full detail)
    pub simplify_max_zoom: u8,
    /// When true (and `simplify` is on), use time-aware TD-TR / Synchronized
    /// Euclidean Distance simplification instead of plain spatial Visvalingam,
    /// preserving per-vertex timing through the simplification.
    pub time_aware_simplify: bool,
    /// When true (and `simplify` is on), simplify with a **latitude-corrected
    /// metric tolerance** instead of the legacy fixed per-zoom degree table:
    /// the longitude axis is scaled by `cos(latitude)` so a given zoom's
    /// tolerance means the same GROUND distance at every latitude (a fixed
    /// degree tolerance is ~2× coarser in E–W terms at 60° than at the
    /// equator). Opt-in (default `false`) so existing builds stay byte-identical.
    pub simplify_metric: bool,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            min_vertices: 2,
            // Small buffer (~100m at equator) to ensure visual continuity
            buffer_degrees: 0.001,
            // Fills are clipped exact — see the field doc.
            polygon_buffer_degrees: 0.0,
            // No temporal slicing by default
            temporal_granularity_ms: None,
            // Simplification disabled by default
            simplify: false,
            simplify_max_zoom: 14,
            time_aware_simplify: false,
            simplify_metric: false,
        }
    }
}

/// Tile bounds in WGS84 coordinates
#[derive(Debug, Clone, Copy)]
struct TileBounds {
    min_lon: f64,
    min_lat: f64,
    max_lon: f64,
    max_lat: f64,
}

impl TileBounds {
    /// Create tile bounds for a specific tile using Web Mercator projection
    fn from_tile(x: u32, y: u32, zoom: u8) -> Self {
        let n = (1u32 << zoom) as f64;

        // Calculate longitude bounds (straightforward)
        let min_lon = (x as f64 / n) * 360.0 - 180.0;
        let max_lon = ((x + 1) as f64 / n) * 360.0 - 180.0;

        // Calculate latitude bounds using correct Web Mercator formula
        // lat = atan(sinh(π * (1 - 2 * y / n)))
        let max_lat = (std::f64::consts::PI * (1.0 - 2.0 * y as f64 / n))
            .sinh()
            .atan()
            .to_degrees();
        let min_lat = (std::f64::consts::PI * (1.0 - 2.0 * (y + 1) as f64 / n))
            .sinh()
            .atan()
            .to_degrees();

        Self {
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        }
    }

    /// Add buffer around bounds
    fn with_buffer(self, buffer: f64) -> Self {
        Self {
            min_lon: self.min_lon - buffer,
            min_lat: self.min_lat - buffer,
            max_lon: self.max_lon + buffer,
            max_lat: self.max_lat + buffer,
        }
    }

    /// Check if bounds intersect with a bounding box
    fn intersects(&self, min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> bool {
        self.min_lon <= max_lon
            && self.max_lon >= min_lon
            && self.min_lat <= max_lat
            && self.max_lat >= min_lat
    }
}

/// Compute the bounding box of a set of coordinates
fn compute_bbox(coords: &[(f64, f64, f64)]) -> (f64, f64, f64, f64) {
    let mut min_lon = f64::MAX;
    let mut min_lat = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut max_lat = f64::MIN;

    for (lon, lat, _) in coords {
        min_lon = min_lon.min(*lon);
        min_lat = min_lat.min(*lat);
        max_lon = max_lon.max(*lon);
        max_lat = max_lat.max(*lat);
    }

    (min_lon, min_lat, max_lon, max_lat)
}

/// Convert a WGS84 (lon, lat) point to continuous Web Mercator tile-space
/// coordinates at `zoom` (the integer floor is the tile index).
///
/// Latitudes outside the Web Mercator usable band are clamped, since the
/// projection diverges at the poles.
fn lonlat_to_world_tile(lon: f64, lat: f64, zoom: u8) -> (f64, f64) {
    let n = (1u32 << zoom) as f64;
    let lat = lat.clamp(-85.0511, 85.0511);
    let lon = lon.clamp(-180.0, 180.0);
    let world_x = (lon + 180.0) / 360.0 * n;
    let lat_rad = lat.to_radians();
    let world_y = (1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n;
    (world_x, world_y)
}

/// Enumerate every tile a polyline traverses at `zoom` using a per-segment
/// supercover (Amanatides–Woo style) DDA in continuous tile space.
///
/// For a continental trajectory at zoom 14 this drops the candidate set from
/// `O(bbox_w * bbox_h)` (~10k tiles for a coast-to-coast path) to the actual
/// touched count (~hundreds), since we no longer enumerate the bounding box
/// interior.
///
/// The returned set is the union of tiles touched by all segments. Caller
/// should still call `clip_trajectory_to_tile` per tile — the buffer the
/// clipper applies can still drop a tile we list, which is fine.
fn tiles_along_trajectory(coords: &[(f64, f64, f64)], zoom: u8) -> HashSet<(u32, u32)> {
    let mut tiles: HashSet<(u32, u32)> = HashSet::new();
    if coords.is_empty() {
        return tiles;
    }
    let n = 1u32 << zoom;

    let mut add_tile = |tx: i64, ty: i64| {
        if tx < 0 || ty < 0 {
            return;
        }
        let (tx, ty) = (tx as u32, ty as u32);
        if tx < n && ty < n {
            tiles.insert((tx, ty));
        }
    };

    // Single-vertex degenerate case: one tile.
    if coords.len() == 1 {
        let (wx, wy) = lonlat_to_world_tile(coords[0].0, coords[0].1, zoom);
        add_tile(wx.floor() as i64, wy.floor() as i64);
        return tiles;
    }

    for win in coords.windows(2) {
        let (x0, y0) = lonlat_to_world_tile(win[0].0, win[0].1, zoom);
        let (x1, y1) = lonlat_to_world_tile(win[1].0, win[1].1, zoom);
        supercover_segment(x0, y0, x1, y1, &mut add_tile);
    }
    tiles
}

/// Amanatides–Woo voxel traversal in 2D, in continuous tile space. Emits
/// every integer cell the segment from `(x0,y0)` to `(x1,y1)` enters,
/// including both endpoints.
fn supercover_segment<F: FnMut(i64, i64)>(x0: f64, y0: f64, x1: f64, y1: f64, emit: &mut F) {
    // Start/end cells.
    let mut ix = x0.floor() as i64;
    let mut iy = y0.floor() as i64;
    let ex = x1.floor() as i64;
    let ey = y1.floor() as i64;
    emit(ix, iy);
    if ix == ex && iy == ey {
        return;
    }

    let dx = x1 - x0;
    let dy = y1 - y0;
    // Step in each axis.
    let step_x: i64 = if dx > 0.0 {
        1
    } else if dx < 0.0 {
        -1
    } else {
        0
    };
    let step_y: i64 = if dy > 0.0 {
        1
    } else if dy < 0.0 {
        -1
    } else {
        0
    };

    // Parametric `t` (in [0, 1]) at which we cross the next vertical/horizontal grid line.
    // For an axis with no motion, push the crossing to +∞ so it never wins.
    let inv_dx = if dx != 0.0 { 1.0 / dx } else { 0.0 };
    let inv_dy = if dy != 0.0 { 1.0 / dy } else { 0.0 };

    let next_x_boundary = if step_x > 0 {
        (ix + 1) as f64
    } else if step_x < 0 {
        ix as f64
    } else {
        f64::INFINITY
    };
    let next_y_boundary = if step_y > 0 {
        (iy + 1) as f64
    } else if step_y < 0 {
        iy as f64
    } else {
        f64::INFINITY
    };

    let mut t_max_x = if step_x != 0 {
        (next_x_boundary - x0) * inv_dx
    } else {
        f64::INFINITY
    };
    let mut t_max_y = if step_y != 0 {
        (next_y_boundary - y0) * inv_dy
    } else {
        f64::INFINITY
    };

    let t_delta_x = if step_x != 0 {
        (step_x as f64) * inv_dx
    } else {
        f64::INFINITY
    };
    let t_delta_y = if step_y != 0 {
        (step_y as f64) * inv_dy
    } else {
        f64::INFINITY
    };

    // Hard cap to defend against pathological NaN/inf inputs.
    let mut guard = 0usize;
    let cap = ((dx.abs() + dy.abs()) as usize).saturating_add(4) * 4 + 32;
    while (ix != ex || iy != ey) && guard < cap {
        if t_max_x < t_max_y {
            t_max_x += t_delta_x;
            ix += step_x;
        } else if t_max_y < t_max_x {
            t_max_y += t_delta_y;
            iy += step_y;
        } else {
            // Diagonal crossing through a corner — emit both adjacent cells
            // so the supercover stays connected.
            emit(ix + step_x, iy);
            emit(ix, iy + step_y);
            t_max_x += t_delta_x;
            t_max_y += t_delta_y;
            ix += step_x;
            iy += step_y;
        }
        emit(ix, iy);
        guard += 1;
    }
}

/// Liang-Barsky line clipping algorithm
/// Returns the parameter values (t0, t1) for the clipped segment, or None if completely outside
fn liang_barsky_clip(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    bounds: &TileBounds,
) -> Option<(f64, f64)> {
    let dx = x1 - x0;
    let dy = y1 - y0;

    let mut t0 = 0.0_f64;
    let mut t1 = 1.0_f64;

    // Check each edge
    let p = [-dx, dx, -dy, dy];
    let q = [
        x0 - bounds.min_lon,
        bounds.max_lon - x0,
        y0 - bounds.min_lat,
        bounds.max_lat - y0,
    ];

    for i in 0..4 {
        if p[i].abs() < 1e-10 {
            // Line is parallel to this edge
            if q[i] < 0.0 {
                return None; // Line is outside
            }
        } else {
            let t = q[i] / p[i];
            if p[i] < 0.0 {
                // Entering edge
                t0 = t0.max(t);
            } else {
                // Leaving edge
                t1 = t1.min(t);
            }
        }
    }

    if t0 <= t1 {
        Some((t0, t1))
    } else {
        None
    }
}

/// Interpolate a point along a line segment
fn interpolate_point(x0: f64, y0: f64, x1: f64, y1: f64, t: f64) -> (f64, f64) {
    (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
}

/// Interpolate altitude along a line segment
fn interpolate_alt(alt0: f64, alt1: f64, t: f64) -> f64 {
    alt0 + t * (alt1 - alt0)
}

/// Interpolate timestamp along a line segment based on parameter t
fn interpolate_timestamp(time0: u64, time1: u64, t: f64) -> u64 {
    if t <= 0.0 {
        return time0;
    }
    if t >= 1.0 {
        return time1;
    }
    let duration = time1 as f64 - time0 as f64;
    (time0 as f64 + t * duration) as u64
}

/// Interpolate a per-vertex scalar value along a line segment based on `t`.
/// A `NaN` endpoint propagates to the interpolated value (no value → no color).
fn interpolate_value(v0: f32, v1: f32, t: f64) -> f32 {
    if t <= 0.0 {
        return v0;
    }
    if t >= 1.0 {
        return v1;
    }
    v0 + (t as f32) * (v1 - v0)
}

/// Extract 3D coordinates from a GeoJSON geometry
fn extract_linestring_coords(geometry: &Geometry) -> Option<Vec<(f64, f64, f64)>> {
    match &geometry.value {
        GeomValue::LineString(coords) => Some(
            coords
                .iter()
                .map(|c| {
                    let alt = if c.len() >= 3 { c[2] } else { 0.0 };
                    (c[0], c[1], alt)
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Compute per-vertex timestamps based on distance interpolation
/// (Similar to what's done in columnar.rs)
pub fn compute_vertex_timestamps(
    coords: &[(f64, f64, f64)],
    start_time: u64,
    end_time: u64,
) -> Vec<u64> {
    if coords.is_empty() {
        return vec![];
    }
    if coords.len() == 1 {
        return vec![start_time];
    }

    let duration = end_time as f64 - start_time as f64;

    // Calculate cumulative distances
    let mut cumulative_distances = vec![0.0];
    let mut total_distance = 0.0;

    for i in 1..coords.len() {
        let dist = haversine_distance(coords[i - 1].1, coords[i - 1].0, coords[i].1, coords[i].0);
        total_distance += dist;
        cumulative_distances.push(total_distance);
    }

    // Interpolate timestamps based on distance
    coords
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if total_distance > 0.0 {
                let fraction = cumulative_distances[i] / total_distance;
                (start_time as f64 + fraction * duration) as u64
            } else {
                start_time
            }
        })
        .collect()
}

/// Haversine distance between two points in meters
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS: f64 = 6_371_000.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let a =
        (dlat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS * c
}

/// Clip a trajectory to a specific tile
///
/// Returns the clipped coordinates and timestamps for the portion of the
/// trajectory that falls within the tile bounds.
#[allow(clippy::type_complexity)]
fn clip_trajectory_to_tile(
    coords: &[(f64, f64, f64)],
    timestamps: &[u64],
    values: &[f32],
    // Per-vertex × per-bucket value matrix, `[vertex][bucket]`. Empty when the
    // producer supplied none; each bucket channel is interpolated at clip
    // boundaries exactly like `values`.
    matrix: &[Vec<f32>],
    bounds: &TileBounds,
) -> Option<(Vec<(f64, f64, f64)>, Vec<u64>, Vec<f32>, Vec<Vec<f32>>)> {
    if coords.len() < 2 {
        return None;
    }
    let has_matrix = !matrix.is_empty();

    // Interpolate one matrix row (all buckets) between vertices i and i+1.
    let interp_row = |i: usize, t: f64| -> Vec<f32> {
        if !has_matrix {
            return Vec::new();
        }
        let a = &matrix[i];
        let b = &matrix[i + 1];
        a.iter()
            .zip(b.iter())
            .map(|(&va, &vb)| interpolate_value(va, vb, t))
            .collect()
    };

    let mut clipped_coords: Vec<(f64, f64, f64)> = Vec::new();
    let mut clipped_times: Vec<u64> = Vec::new();
    let mut clipped_values: Vec<f32> = Vec::new();
    let mut clipped_matrix: Vec<Vec<f32>> = Vec::new();

    // Process each line segment
    for i in 0..coords.len() - 1 {
        let (x0, y0, alt0) = coords[i];
        let (x1, y1, alt1) = coords[i + 1];
        let time0 = timestamps[i];
        let time1 = timestamps[i + 1];
        let value0 = values[i];
        let value1 = values[i + 1];

        // Check if segment intersects the tile
        if let Some((t0, t1)) = liang_barsky_clip(x0, y0, x1, y1, bounds) {
            // Calculate the clipped start point
            let (start_x, start_y) = if t0 > 0.0 {
                interpolate_point(x0, y0, x1, y1, t0)
            } else {
                (x0, y0)
            };
            let start_alt = interpolate_alt(alt0, alt1, t0);
            let start_time = interpolate_timestamp(time0, time1, t0);
            let start_value = interpolate_value(value0, value1, t0);

            // Calculate the clipped end point
            let (end_x, end_y) = if t1 < 1.0 {
                interpolate_point(x0, y0, x1, y1, t1)
            } else {
                (x1, y1)
            };
            let end_alt = interpolate_alt(alt0, alt1, t1);
            let end_time = interpolate_timestamp(time0, time1, t1);
            let end_value = interpolate_value(value0, value1, t1);

            // Add start point if not a duplicate of last point
            let should_add_start = clipped_coords.is_empty()
                || (clipped_coords.last().unwrap().0 - start_x).abs() > 1e-9
                || (clipped_coords.last().unwrap().1 - start_y).abs() > 1e-9;

            if should_add_start {
                clipped_coords.push((start_x, start_y, start_alt));
                clipped_times.push(start_time);
                clipped_values.push(start_value);
                if has_matrix {
                    clipped_matrix.push(interp_row(i, t0));
                }
            }

            // Add end point if different from start point
            if (end_x - start_x).abs() > 1e-9 || (end_y - start_y).abs() > 1e-9 {
                clipped_coords.push((end_x, end_y, end_alt));
                clipped_times.push(end_time);
                clipped_values.push(end_value);
                if has_matrix {
                    clipped_matrix.push(interp_row(i, t1));
                }
            }
        }
    }

    if clipped_coords.len() >= 2 {
        Some((
            clipped_coords,
            clipped_times,
            clipped_values,
            clipped_matrix,
        ))
    } else {
        None
    }
}

/// Slice a clipped segment at temporal bucket boundaries so each output segment
/// lies within a single bucket.
///
/// Every edge is split at *every* bucket boundary it crosses (interpolating a
/// shared vertex at each), so even a sparse edge that jumps several buckets
/// yields one segment per bucket. A vertex landing exactly on a boundary is
/// shared between the closing and opening segment — not duplicated within
/// either.
fn slice_segment_temporally(segment: ClippedSegment, granularity_ms: u64) -> Vec<ClippedSegment> {
    let n = segment.coordinates.len();
    if n < 2 || granularity_ms == 0 {
        return vec![segment];
    }
    // Matrix-bearing segments animate by selecting a bucket column, not by time
    // slicing — splitting them would desync the per-vertex matrix from the
    // geometry. They span the whole range as a single feature by design.
    if !segment.vertex_value_matrix.is_empty() {
        return vec![segment];
    }
    let g = granularity_ms;
    if segment.start_time / g == segment.end_time / g {
        return vec![segment]; // entirely within one bucket
    }

    // Own the inputs so the closures below don't borrow `segment`.
    let ClippedSegment {
        tile_x,
        tile_y,
        zoom,
        coordinates,
        timestamps,
        vertex_values,
        properties,
        feature_id,
        ..
    } = segment;
    let has_values = !vertex_values.is_empty();
    let value_at = |i: usize| -> f32 {
        if has_values {
            vertex_values[i]
        } else {
            f32::NAN
        }
    };

    // Augmented vertex stream: the originals plus an interpolated point at every
    // bucket boundary strictly inside an edge.
    type Aug = ((f64, f64, f64), u64, f32);
    let mut aug: Vec<Aug> = Vec::with_capacity(n);
    aug.push((coordinates[0], timestamps[0], value_at(0)));
    for i in 1..n {
        let pt = timestamps[i - 1];
        let ct = timestamps[i];
        let (px, py, pa) = coordinates[i - 1];
        let (cx, cy, ca) = coordinates[i];
        if ct > pt {
            for k in (pt / g + 1)..=(ct / g) {
                let b = k * g;
                if b <= pt || b >= ct {
                    continue; // only boundaries strictly inside the edge
                }
                let t = (b - pt) as f64 / (ct - pt) as f64;
                let bc = (px + t * (cx - px), py + t * (cy - py), pa + t * (ca - pa));
                let bv = if has_values {
                    interpolate_value(value_at(i - 1), value_at(i), t)
                } else {
                    f32::NAN
                };
                aug.push((bc, b, bv));
            }
        }
        aug.push((coordinates[i], ct, value_at(i)));
    }

    let make_slice = |pts: &[Aug]| -> ClippedSegment {
        ClippedSegment {
            tile_x,
            tile_y,
            zoom,
            coordinates: pts.iter().map(|p| p.0).collect(),
            timestamps: pts.iter().map(|p| p.1).collect(),
            vertex_values: if has_values {
                pts.iter().map(|p| p.2).collect()
            } else {
                Vec::new()
            },
            // Matrix-bearing segments never reach here (guarded above).
            vertex_value_matrix: Vec::new(),
            start_time: pts.first().unwrap().1,
            end_time: pts.last().unwrap().1,
            properties: properties.clone(),
            feature_id: feature_id.clone(),
        }
    };

    // Split at every vertex on a bucket boundary (shared with the next slice).
    let mut slices: Vec<ClippedSegment> = Vec::new();
    let mut cur: Vec<Aug> = vec![aug[0]];
    for i in 1..aug.len() {
        cur.push(aug[i]);
        if aug[i].1 % g == 0 && i < aug.len() - 1 {
            if cur.len() >= 2 {
                slices.push(make_slice(&cur));
            }
            cur = vec![aug[i]];
        }
    }
    if cur.len() >= 2 {
        slices.push(make_slice(&cur));
    }

    if slices.is_empty() {
        vec![make_slice(&aug)]
    } else {
        slices
    }
}

/// Clip a trajectory feature across all tiles it intersects
///
/// This is the main entry point for trajectory clipping. It takes a feature
/// with a LineString geometry and returns clipped segments for each tile
/// the trajectory passes through.
///
/// # Arguments
/// * `feature` - The GeoJSON feature with LineString geometry
/// * `shared_properties` - Arc-wrapped properties for zero-copy sharing
/// * `start_time` - Start timestamp of the trajectory
/// * `end_time` - End timestamp of the trajectory (for duration-based interpolation)
/// * `zoom` - The zoom level to clip at
/// * `config` - Clipping configuration
///
/// # Returns
/// A vector of clipped segments, one for each tile the trajectory intersects
pub fn clip_trajectory(
    feature: &Feature,
    shared_properties: Option<SharedProperties>,
    start_time: u64,
    end_time: u64,
    zoom: u8,
    config: &ClipConfig,
    // Optional producer-supplied per-vertex absolute Unix-ms timestamps.
    // Used in place of uniform-by-distance interpolation when available
    // AND the post-simplify vertex count still matches. Simplification can
    // drop vertices, so we fall back to distance-interpolation in that case
    // to avoid splatting the wrong timestamp onto the wrong vertex.
    supplied_vertex_times: Option<&[u64]>,
    // Optional producer-supplied per-vertex scalar values (e.g. SST). Like
    // `supplied_vertex_times`, accepted only when simplification preserved the
    // vertex count and the supplied length matches; otherwise no value channel
    // is emitted for this trajectory.
    supplied_vertex_values: Option<&[f32]>,
    // Optional producer-supplied per-vertex × per-bucket value matrix, flat
    // vertex-major (`matrix[v * num_buckets + b]`). Reshaped to `[vertex][bucket]`
    // and resampled per bucket at clip boundaries. Only accepted when
    // simplification is OFF (it changes the vertex set) — flow corridors build
    // with `simplify: false`.
    supplied_vertex_value_matrix: Option<&[f32]>,
) -> Vec<ClippedSegment> {
    // Extract coordinates
    let geometry = match &feature.geometry {
        Some(g) => g,
        None => return vec![],
    };

    let coords = match extract_linestring_coords(geometry) {
        Some(c) => c,
        None => return vec![],
    };
    let original_vertex_count = coords.len();

    // Skip if too few vertices
    if coords.len() < config.min_vertices {
        return vec![];
    }

    // Compute per-vertex times + values on the FULL geometry first, so a
    // time-aware simplifier can keep the producer's real timing.
    let full_times: Vec<u64> = match supplied_vertex_times {
        Some(s) if s.len() == coords.len() => s.to_vec(),
        _ => compute_vertex_timestamps(&coords, start_time, end_time),
    };
    let full_has_values = matches!(supplied_vertex_values, Some(s) if s.len() == coords.len());
    let full_values: Vec<f32> = if full_has_values {
        supplied_vertex_values.unwrap().to_vec()
    } else {
        vec![f32::NAN; coords.len()]
    };

    // Reshape the flat vertex-major matrix into `[vertex][bucket]` on the FULL
    // geometry. Accepted only when the length is a clean multiple of the vertex
    // count AND simplification is off (simplify changes the vertex set, which
    // would desync the matrix). Empty otherwise.
    let full_matrix: Vec<Vec<f32>> = match supplied_vertex_value_matrix {
        Some(m) if !config.simplify && !m.is_empty() && m.len() % coords.len() == 0 => {
            let nb = m.len() / coords.len();
            (0..coords.len())
                .map(|v| m[v * nb..(v + 1) * nb].to_vec())
                .collect()
        }
        _ => Vec::new(),
    };

    // Apply simplification for lower zoom levels if enabled.
    let (coords, timestamps, values, has_values) = if config.simplify {
        if config.time_aware_simplify {
            // TD-TR keeps a subset of vertices preserving position-at-time,
            // carrying the real times + values (no alignment loss).
            let (sc, st, sv) = simplify_td_tr_for_zoom_with(
                &coords,
                &full_times,
                &full_values,
                zoom,
                config.simplify_max_zoom,
                config.simplify_metric,
            );
            (sc, st, sv, full_has_values)
        } else {
            // Spatial Visvalingam: dropped vertices break per-vertex-time
            // alignment, so recompute times by distance and drop the supplied
            // value channel when simplification changed the vertex set.
            let sc = simplify_for_zoom_with(
                &coords,
                zoom,
                config.simplify_max_zoom,
                config.simplify_metric,
            );
            let preserved = sc.len() == original_vertex_count;
            let st = if preserved {
                full_times.clone()
            } else {
                compute_vertex_timestamps(&sc, start_time, end_time)
            };
            let (sv, hv) = if preserved && full_has_values {
                (full_values.clone(), true)
            } else {
                (vec![f32::NAN; sc.len()], false)
            };
            (sc, st, sv, hv)
        }
    } else {
        (coords, full_times, full_values, full_has_values)
    };

    // Skip if simplification reduced below minimum.
    if coords.len() < config.min_vertices {
        return vec![];
    }

    let mut segments = Vec::new();

    // Antimeridian safety: split the polyline wherever two consecutive
    // vertices differ in longitude by more than 180°, and clip each run
    // independently. Such a pair straddles the dateline — the shorter path
    // wraps across ±180°, but `lonlat_to_world_tile` clamps lon to [-180,180]
    // and `supercover_segment` walks a *straight* line in that clamped tile
    // space, sweeping the long way across the whole map and baking a
    // globe-spanning sliver into every tile column the edge crosses. Splitting
    // here guarantees no segment ever contains such an edge, regardless of
    // whether the upstream generator split its tracks correctly. The common
    // case is a single run spanning the whole trajectory (no crossing), which
    // does exactly the same work as before.
    let mut run_start = 0usize;
    for split in 1..=coords.len() {
        let at_end = split == coords.len();
        let crosses_antimeridian = !at_end && (coords[split].0 - coords[split - 1].0).abs() > 180.0;
        if !(at_end || crosses_antimeridian) {
            continue;
        }
        let run_coords = &coords[run_start..split];
        let run_times = &timestamps[run_start..split];
        let run_values = &values[run_start..split];
        let run_matrix: &[Vec<f32>] = if full_matrix.is_empty() {
            &[]
        } else {
            &full_matrix[run_start..split]
        };
        run_start = split;
        if run_coords.len() < 2 {
            continue;
        }

        // Compute bounding box for quick rejection (per run).
        let (min_lon, min_lat, max_lon, max_lat) = compute_bbox(run_coords);

        // Per-segment supercover enumeration of touched tiles. At continental
        // scales this collapses a 10k-tile bbox sweep to ~hundreds of real
        // crossings, which is the single biggest win for very long trajectories.
        let touched = tiles_along_trajectory(run_coords, zoom);

        // `touched` is already deduplicated; iterate it directly.
        for (tile_x, tile_y) in touched {
            let tile_bounds = TileBounds::from_tile(tile_x, tile_y, zoom);
            let buffered_bounds = tile_bounds.with_buffer(config.buffer_degrees);

            // Quick rejection: check if feature bbox intersects tile
            if !buffered_bounds.intersects(min_lon, min_lat, max_lon, max_lat) {
                continue;
            }

            // Clip to this tile
            if let Some((clipped_coords, clipped_times, clipped_values, clipped_matrix)) =
                clip_trajectory_to_tile(
                    run_coords,
                    run_times,
                    run_values,
                    run_matrix,
                    &buffered_bounds,
                )
            {
                // Matrix corridors are timeless: they exist across the WHOLE
                // range, animated by selecting a bucket column rather than by
                // their geometry's (interpolated) vertex times. Pin every
                // clipped piece to the feature's full [start, end] so all the
                // cell's corridors land in ONE temporal bucket — one tile per
                // cell spanning the range, whose time window matches every
                // playback frame (instead of fragmenting by interpolated time).
                let has_matrix = !clipped_matrix.is_empty();
                let seg_start_time = if has_matrix {
                    start_time
                } else {
                    *clipped_times.first().unwrap()
                };
                let seg_end_time = if has_matrix {
                    end_time
                } else {
                    *clipped_times.last().unwrap()
                };

                let segment = ClippedSegment {
                    tile_x,
                    tile_y,
                    zoom,
                    coordinates: clipped_coords,
                    timestamps: clipped_times,
                    // Drop the value channel for trajectories that carry none.
                    vertex_values: if has_values {
                        clipped_values
                    } else {
                        Vec::new()
                    },
                    // Already empty when no matrix was supplied.
                    vertex_value_matrix: clipped_matrix,
                    start_time: seg_start_time,
                    end_time: seg_end_time,
                    // Use Arc::clone for zero-copy property sharing
                    properties: shared_properties.clone(),
                    feature_id: feature.id.clone(),
                };

                // Apply temporal slicing if configured
                if let Some(granularity) = config.temporal_granularity_ms {
                    let sliced = slice_segment_temporally(segment, granularity);
                    segments.extend(sliced);
                } else {
                    segments.push(segment);
                }
            }
        }
    }

    segments
}

// ----------------------------------------------------------------------------
// Non-trajectory (polygon) clipping — coverage placement support
// ----------------------------------------------------------------------------

/// One polygon as GeoJSON-shaped rings: `[ring][vertex][lon, lat, ...]`,
/// exterior ring first, holes after, each ring closed (first == last).
pub(crate) type PolygonRings = Vec<Vec<Vec<f64>>>;

/// Buffered WGS84 bounds of one tile — the same rect (tile bounds +
/// `buffer_degrees` on every side) the trajectory clipper clips against.
/// Returned as `(min_lon, min_lat, max_lon, max_lat)`.
pub(crate) fn buffered_tile_bounds(
    x: u32,
    y: u32,
    zoom: u8,
    buffer_degrees: f64,
) -> (f64, f64, f64, f64) {
    let b = TileBounds::from_tile(x, y, zoom).with_buffer(buffer_degrees);
    (b.min_lon, b.min_lat, b.max_lon, b.max_lat)
}

/// Clip one closed polygon ring against a rectangle (Sutherland–Hodgman,
/// clipping successively against the rect's four half-planes). Input is a
/// GeoJSON ring (closed, first == last); output is a re-closed ring, or empty
/// when the intersection is empty or degenerate (< 3 distinct vertices).
/// Altitude (a third coordinate) is dropped — tile geometry columns are 2D.
///
/// ACCEPTED ARTIFACT: a strongly concave ring whose intersection with the
/// rect is disconnected (e.g. a C-shape with both arms poking into the tile)
/// emits ONE ring with zero-width bridge edges along the rect boundary — the
/// standard tiling-ecosystem behavior (geojson-vt clips axis-by-axis the same
/// way). Earcut fills swallow the zero-area bridges; stroked outlines of such
/// pieces may show hairlines along tile edges. Revisit only if/when geo ≥0.30
/// lands (its i_overlay `BooleanOps` is robust; the 0.28 Martinez
/// implementation can panic on degenerate rings, is far slower per clip, and
/// its vertex ordering is not stable across versions — a byte-reproducibility
/// hazard).
fn sutherland_hodgman_ring(ring: &[Vec<f64>], bounds: &TileBounds) -> Vec<Vec<f64>> {
    let mut pts: Vec<(f64, f64)> = ring
        .iter()
        .filter(|c| c.len() >= 2)
        .map(|c| (c[0], c[1]))
        .collect();
    // Drop the GeoJSON closing duplicate; the algorithm works on the open ring.
    if pts.len() >= 2 && pts.first() == pts.last() {
        pts.pop();
    }
    if pts.len() < 3 {
        return Vec::new();
    }

    // Clip against one axis-aligned half-plane (`coord >= value` when
    // `keep_ge`, else `coord <= value`). A vertex exactly on the boundary
    // counts as inside, so `cur_in != prev_in` guarantees a non-zero
    // denominator in the intersection below.
    let clip_axis = |pts: &[(f64, f64)], value: f64, x_axis: bool, keep_ge: bool| {
        let coord = |p: &(f64, f64)| if x_axis { p.0 } else { p.1 };
        let inside = |p: &(f64, f64)| {
            if keep_ge {
                coord(p) >= value
            } else {
                coord(p) <= value
            }
        };
        let mut out: Vec<(f64, f64)> = Vec::with_capacity(pts.len() + 4);
        for i in 0..pts.len() {
            let cur = pts[i];
            let prev = pts[if i == 0 { pts.len() - 1 } else { i - 1 }];
            let cur_in = inside(&cur);
            if cur_in != inside(&prev) {
                let t = (value - coord(&prev)) / (coord(&cur) - coord(&prev));
                out.push((prev.0 + t * (cur.0 - prev.0), prev.1 + t * (cur.1 - prev.1)));
            }
            if cur_in {
                out.push(cur);
            }
        }
        out
    };
    for (value, x_axis, keep_ge) in [
        (bounds.min_lon, true, true),
        (bounds.max_lon, true, false),
        (bounds.min_lat, false, true),
        (bounds.max_lat, false, false),
    ] {
        pts = clip_axis(&pts, value, x_axis, keep_ge);
        if pts.is_empty() {
            return Vec::new();
        }
    }

    // Drop consecutive (and wrap-around) near-duplicate vertices the
    // successive half-plane passes can introduce at rect corners.
    let close =
        |a: &(f64, f64), b: &(f64, f64)| (a.0 - b.0).abs() <= 1e-12 && (a.1 - b.1).abs() <= 1e-12;
    let mut cleaned: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    for p in pts {
        if cleaned.last().map_or(true, |q| !close(q, &p)) {
            cleaned.push(p);
        }
    }
    while cleaned.len() >= 2 && close(&cleaned[0], cleaned.last().unwrap()) {
        cleaned.pop();
    }
    if cleaned.len() < 3 {
        return Vec::new();
    }
    let mut out: Vec<Vec<f64>> = cleaned.into_iter().map(|(x, y)| vec![x, y]).collect();
    out.push(out[0].clone());
    out
}

/// Clip a set of polygons against every tile their bbox covers at `zoom`,
/// returning per-tile clipped polygons. Rings are clipped independently
/// (Sutherland–Hodgman against the buffered tile rect): a polygon whose
/// exterior ring vanishes in a tile is dropped there; holes that vanish are
/// dropped while the surviving exterior is kept, preserving hole structure.
///
/// Tiles are emitted in ascending `(x, y)` order — deterministic, no hashing.
/// Caller is responsible for the antimeridian split: NO ring reaching this
/// function may contain an edge with `|Δlon| > 180°` (such an edge sweeps a
/// straight line the long way across clamped-lon space and would smear the
/// polygon across the whole world). `place_polygon` splits crossing rings first
/// via [`split_polygon_at_antimeridian`], so every piece it passes here is
/// seam-free. A polygon that merely spans a wide (but < 360°) longitude range
/// WITHOUT any wrapping edge is fine — it genuinely occupies those columns.
pub(crate) fn clip_polygons_to_tiles(
    polygons: &[PolygonRings],
    zoom: u8,
    buffer_degrees: f64,
    target: Option<(u32, u32)>,
) -> Vec<((u32, u32), Vec<PolygonRings>)> {
    // Per-ring bboxes, hoisted out of the tile sweep — one O(total vertices)
    // pass up front instead of re-scanning every ring for every swept tile.
    // A ring whose bbox misses a tile rect is separated from it along some
    // axis, so that axis's Sutherland–Hodgman half-plane pass would empty
    // it: the bbox gates below are exactly equivalent to (and far cheaper
    // than) running the clip and dropping the empty result.
    let ring_bboxes: Vec<Vec<(f64, f64, f64, f64)>> = polygons
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|ring| {
                    let mut bb = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                    for c in ring.iter().filter(|c| c.len() >= 2) {
                        bb.0 = bb.0.min(c[0]);
                        bb.1 = bb.1.min(c[1]);
                        bb.2 = bb.2.max(c[0]);
                        bb.3 = bb.3.max(c[1]);
                    }
                    bb
                })
                .collect()
        })
        .collect();

    let mut min_lon = f64::MAX;
    let mut min_lat = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut max_lat = f64::MIN;
    for bb in ring_bboxes.iter().flatten() {
        min_lon = min_lon.min(bb.0);
        min_lat = min_lat.min(bb.1);
        max_lon = max_lon.max(bb.2);
        max_lat = max_lat.max(bb.3);
    }
    if !(min_lon.is_finite() && min_lat.is_finite() && max_lon.is_finite() && max_lat.is_finite())
        || min_lon > max_lon
        || min_lat > max_lat
    {
        return Vec::new();
    }

    let n = 1u32 << zoom;
    // World-tile coords of the bbox corners (clamped to the Web-Mercator
    // band, matching the trajectory supercover); note y grows southward.
    let (wx0, wy0) = lonlat_to_world_tile(min_lon, max_lat, zoom);
    let (wx1, wy1) = lonlat_to_world_tile(max_lon, min_lat, zoom);
    let mut x0 = (wx0.floor() as i64).clamp(0, n as i64 - 1) as u32;
    let mut x1 = (wx1.floor() as i64).clamp(0, n as i64 - 1) as u32;
    let mut y0 = (wy0.floor() as i64).clamp(0, n as i64 - 1) as u32;
    let mut y1 = (wy1.floor() as i64).clamp(0, n as i64 - 1) as u32;

    // Single-tile restriction (the stt-serve per-request path): every tile's
    // clip is independent of the sweep, so restricting the range yields the
    // byte-identical piece the full sweep would have produced for that tile.
    if let Some((tx, ty)) = target {
        if tx < x0 || tx > x1 || ty < y0 || ty > y1 {
            return Vec::new();
        }
        (x0, x1, y0, y1) = (tx, tx, ty, ty);
    }

    let mut out: Vec<((u32, u32), Vec<PolygonRings>)> = Vec::new();
    for x in x0..=x1 {
        for y in y0..=y1 {
            let bounds = TileBounds::from_tile(x, y, zoom).with_buffer(buffer_degrees);
            let mut tile_polys: Vec<PolygonRings> = Vec::new();
            for (poly, bboxes) in polygons.iter().zip(&ring_bboxes) {
                let (Some(exterior), Some(ext_bb)) = (poly.first(), bboxes.first()) else {
                    continue;
                };
                // Part gate: an exterior whose bbox misses this tile clips
                // to empty — skip the O(V) passes (archipelago MultiPolygons
                // otherwise spend ~all sweep time clipping far-away parts).
                if !bounds.intersects(ext_bb.0, ext_bb.1, ext_bb.2, ext_bb.3) {
                    continue;
                }
                let clipped_exterior = sutherland_hodgman_ring(exterior, &bounds);
                if clipped_exterior.is_empty() {
                    continue;
                }
                let mut rings: PolygonRings = vec![clipped_exterior];
                for (hole, hole_bb) in poly[1..].iter().zip(&bboxes[1..]) {
                    // Hole gate: same bbox-disjoint ⇒ clips-to-empty argument.
                    if !bounds.intersects(hole_bb.0, hole_bb.1, hole_bb.2, hole_bb.3) {
                        continue;
                    }
                    let clipped_hole = sutherland_hodgman_ring(hole, &bounds);
                    if !clipped_hole.is_empty() {
                        rings.push(clipped_hole);
                    }
                }
                tile_polys.push(rings);
            }
            if !tile_polys.is_empty() {
                out.push(((x, y), tile_polys));
            }
        }
    }
    out
}

/// Sentinel longitude/latitude magnitude for the meridian-only clip bounds in
/// [`split_polygon_at_antimeridian`]. It vastly exceeds any real UNWRAPPED
/// coordinate (unwrapped lon stays within roughly ±540° for real data, lat
/// within ±90°), so the three non-cutting half-planes of the seam clip are
/// no-ops and only the `x = S` plane actually cuts.
const ANTIMERIDIAN_SENTINEL: f64 = 1e9;

/// Unwrap a closed GeoJSON ring into a CONTINUOUS longitude frame: drop the
/// closing duplicate, then walk the vertices carrying a running ±360° offset
/// (subtract 360 across an edge with `Δlon > 180`, add 360 across `Δlon <
/// -180` — the same seam signal the polyline splitter uses at
/// `clip_trajectory`) and emit `(lon + offset, lat)`. Shifting (not cutting)
/// means no unwrapped edge ever jumps the ±180 seam.
///
/// Returns `None` for a ring with fewer than 3 distinct vertices OR one whose
/// longitude winding does not return to zero around the closing edge — an
/// unbalanced ring encircles a pole (or self-winds) and cannot be split by a
/// single seam. The caller dead-letters those.
fn unwrap_ring(ring: &[Vec<f64>]) -> Option<Vec<(f64, f64)>> {
    let mut src: Vec<(f64, f64)> = ring
        .iter()
        .filter(|c| c.len() >= 2)
        .map(|c| (c[0], c[1]))
        .collect();
    // Drop the GeoJSON closing duplicate (first == last).
    if src.len() >= 2 && src.first() == src.last() {
        src.pop();
    }
    if src.len() < 3 {
        return None;
    }

    let mut out: Vec<(f64, f64)> = Vec::with_capacity(src.len());
    let mut offset = 0.0_f64;
    for i in 0..src.len() {
        if i > 0 {
            let dlon = src[i].0 - src[i - 1].0;
            if dlon > 180.0 {
                offset -= 360.0;
            } else if dlon < -180.0 {
                offset += 360.0;
            }
        }
        out.push((src[i].0 + offset, src[i].1));
    }
    // Closing edge (last raw → first raw): a well-formed, non-pole-enclosing
    // ring winds a net zero in longitude, so the offset returns to 0 here.
    let dclose = src[0].0 - src[src.len() - 1].0;
    if dclose > 180.0 {
        offset -= 360.0;
    } else if dclose < -180.0 {
        offset += 360.0;
    }
    if offset.abs() > 180.0 {
        return None; // unbalanced ⇒ pole-enclosing / self-winding
    }
    Some(out)
}

/// Re-close an unwrapped point list into a GeoJSON-shaped ring (`[[x, y], …]`,
/// first vertex pushed again as the closing duplicate) for
/// [`sutherland_hodgman_ring`], which drops the duplicate itself.
fn closed_ring_from_pts(pts: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let mut r: Vec<Vec<f64>> = pts.iter().map(|(x, y)| vec![*x, *y]).collect();
    if let Some(first) = r.first().cloned() {
        r.push(first);
    }
    r
}

/// The unique odd multiple of 180 (`… -540, -180, 180, 540 …`, i.e. the wrapped
/// dateline in the unwrapped frame) strictly inside `(lo, hi)`, or `None` when
/// the span is ≥360° (pole-enclosing / degenerate) or contains no seam.
fn unique_seam_meridian(lo: f64, hi: f64) -> Option<f64> {
    let splittable = lo.is_finite() && hi.is_finite() && lo < hi && (hi - lo) < 360.0;
    if !splittable {
        return None;
    }
    // Odd multiples of 180 are `360k + 180`. The smallest one strictly greater
    // than `lo` is the only candidate (the span is < 360° wide, so at most one
    // fits). Keep it iff it is also strictly less than `hi`.
    let k = ((lo - 180.0) / 360.0).floor() as i64 + 1;
    let seam = 360.0 * k as f64 + 180.0;
    if seam > lo && seam < hi {
        Some(seam)
    } else {
        None
    }
}

/// Signed shoelace area of a closed ring (positive = CCW in lon/lat).
fn signed_ring_area(ring: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for w in ring.windows(2) {
        if w[0].len() >= 2 && w[1].len() >= 2 {
            sum += w[0][0] * w[1][1] - w[1][0] * w[0][1];
        }
    }
    sum / 2.0
}

/// Split a polygon that crosses the antimeridian into up to two per-hemisphere
/// polygons ("split-then-reuse"): unwrap each ring into a continuous longitude
/// frame, cut at the single ±180 seam by reusing [`sutherland_hodgman_ring`]
/// VERBATIM once per side, then normalize each side back into `[-180, 180]` by a
/// single uniform `k·360` translation (area + winding preserved). Returns
/// `[west, east]` (west — the `x ≤ S` side — before east; each present only when
/// its exterior survives; holes in input order).
///
/// West (`max_lon = S`) and east (`min_lon = S`) cut the SAME unwrapped edge
/// with the SAME `t = (S − prev.x)/(cur.x − prev.x)`, so the two sides emit
/// BIT-IDENTICAL seam latitudes — watertight for free. A vertex exactly on `S`
/// is inside for both sides.
///
/// Returns EMPTY for a genuinely unsplittable ring: an unwrapped exterior that
/// spans ≥360° (pole-enclosing) or is otherwise degenerate, so there is not
/// exactly one seam strictly inside its span. The caller dead-letters + counts
/// these (`antimeridian_fallbacks`); expected 0 on real data.
///
/// SIMPLIFY ORDERING (why `stt_core::geometry::simplify_polygon` is NOT wired):
/// a simplification pass MUST run AFTER this split AND after the per-tile clip,
/// and only BELOW the max tiled zoom (the max-zoom tier stays lossless).
/// Simplifying a ring before splitting would move vertices off the seam and
/// destroy the bit-identical watertight cut this function depends on.
pub(crate) fn split_polygon_at_antimeridian(poly: &PolygonRings) -> Vec<PolygonRings> {
    let Some(exterior_raw) = poly.first() else {
        return Vec::new();
    };
    let Some(ext) = unwrap_ring(exterior_raw) else {
        return Vec::new();
    };

    // Exterior unwrapped lon-bbox + its continuous frame centre.
    let (xlo, xhi) = ext.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
        (lo.min(p.0), hi.max(p.0))
    });
    let ec = (xlo + xhi) / 2.0;

    // The single seam meridian; bail (dead-letter) if the span is unsplittable.
    let Some(seam) = unique_seam_meridian(xlo, xhi) else {
        return Vec::new();
    };

    // Unwrap each hole independently, then translate it by a whole multiple of
    // 360 so it shares the exterior's continuous frame.
    let mut rings_unwrapped: Vec<Vec<(f64, f64)>> = Vec::with_capacity(poly.len());
    rings_unwrapped.push(ext);
    for hole_raw in &poly[1..] {
        let Some(hole) = unwrap_ring(hole_raw) else {
            continue; // vanished / degenerate / self-winding hole
        };
        let (hlo, hhi) = hole.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
            (lo.min(p.0), hi.max(p.0))
        });
        let hc = (hlo + hhi) / 2.0;
        let shift = ((ec - hc) / 360.0).round() * 360.0;
        rings_unwrapped.push(hole.iter().map(|(x, y)| (x + shift, *y)).collect());
    }

    // Meridian-only clip bounds: only the `x = S` half-plane cuts; the other
    // three are sentinel-wide no-ops.
    let west_bounds = TileBounds {
        min_lon: -ANTIMERIDIAN_SENTINEL,
        max_lon: seam,
        min_lat: -ANTIMERIDIAN_SENTINEL,
        max_lat: ANTIMERIDIAN_SENTINEL,
    };
    let east_bounds = TileBounds {
        min_lon: seam,
        max_lon: ANTIMERIDIAN_SENTINEL,
        min_lat: -ANTIMERIDIAN_SENTINEL,
        max_lat: ANTIMERIDIAN_SENTINEL,
    };

    // Cut one side: exterior first (drop the whole side if it clips empty),
    // then each surviving hole (same drop-vanished-hole rule the tiler uses).
    let cut_side = |bounds: &TileBounds| -> Option<PolygonRings> {
        let ext_clip = sutherland_hodgman_ring(&closed_ring_from_pts(&rings_unwrapped[0]), bounds);
        if ext_clip.is_empty() {
            return None;
        }
        let mut side: PolygonRings = vec![ext_clip];
        for hole in &rings_unwrapped[1..] {
            let hole_clip = sutherland_hodgman_ring(&closed_ring_from_pts(hole), bounds);
            if !hole_clip.is_empty() {
                side.push(hole_clip);
            }
        }
        Some(side)
    };

    let mut out: Vec<PolygonRings> = Vec::new();
    for bounds in [&west_bounds, &east_bounds] {
        if let Some(mut side) = cut_side(bounds) {
            normalize_side_into_range(&mut side);
            fix_winding(&mut side);
            out.push(side);
        }
    }
    out
}

/// Normalize one split side back into `[-180, 180]` by a single uniform `k·360`
/// shift keyed on the side EXTERIOR's bbox centre, applied to the exterior AND
/// every hole by the SAME `k` — a pure translation, so area and winding are
/// untouched and the exterior/hole nesting is preserved.
fn normalize_side_into_range(side: &mut PolygonRings) {
    let Some(exterior) = side.first() else {
        return;
    };
    let (lo, hi) = exterior
        .iter()
        .filter(|c| c.len() >= 2)
        .fold((f64::MAX, f64::MIN), |(lo, hi), c| {
            (lo.min(c[0]), hi.max(c[0]))
        });
    if !(lo.is_finite() && hi.is_finite()) {
        return;
    }
    let k = ((lo + hi) / 2.0 / 360.0).round();
    if k == 0.0 {
        return;
    }
    let shift = -k * 360.0;
    for ring in side.iter_mut() {
        for c in ring.iter_mut() {
            if !c.is_empty() {
                c[0] += shift;
            }
        }
    }
}

/// Winding guard for the split path ONLY (keeps the non-crossing path
/// byte-identical): force the exterior CCW (positive shoelace) and every hole
/// CW (negative), reversing in place where the sign is wrong. Rings stay closed
/// (reversing a `[a … a]` ring keeps `first == last`).
fn fix_winding(side: &mut PolygonRings) {
    for (i, ring) in side.iter_mut().enumerate() {
        let area = signed_ring_area(ring);
        let want_positive = i == 0; // exterior CCW; holes CW
        if area.abs() > 0.0 && (area > 0.0) != want_positive {
            ring.reverse();
        }
    }
}

/// Check if a feature is a LineString with duration (trajectory)
pub fn is_clippable_trajectory(feature: &Feature, end_timestamp: Option<u64>) -> bool {
    // Must have duration
    if end_timestamp.is_none() {
        return false;
    }

    // Must be a LineString
    match &feature.geometry {
        Some(g) => matches!(&g.value, GeomValue::LineString(coords) if coords.len() >= 2),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linestring_feature(coords: Vec<Vec<f64>>) -> Feature {
        Feature {
            bbox: None,
            geometry: Some(Geometry::new(GeomValue::LineString(coords))),
            id: None,
            properties: None,
            foreign_members: None,
        }
    }

    #[test]
    fn test_liang_barsky_inside() {
        let bounds = TileBounds {
            min_lon: 0.0,
            min_lat: 0.0,
            max_lon: 10.0,
            max_lat: 10.0,
        };

        // Line fully inside
        let result = liang_barsky_clip(2.0, 2.0, 8.0, 8.0, &bounds);
        assert!(result.is_some());
        let (t0, t1) = result.unwrap();
        assert!((t0 - 0.0).abs() < 1e-9);
        assert!((t1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_liang_barsky_crossing() {
        let bounds = TileBounds {
            min_lon: 0.0,
            min_lat: 0.0,
            max_lon: 10.0,
            max_lat: 10.0,
        };

        // Line crossing through
        let result = liang_barsky_clip(-5.0, 5.0, 15.0, 5.0, &bounds);
        assert!(result.is_some());
        let (t0, t1) = result.unwrap();
        assert!(t0 > 0.0);
        assert!(t1 < 1.0);
    }

    #[test]
    fn test_liang_barsky_outside() {
        let bounds = TileBounds {
            min_lon: 0.0,
            min_lat: 0.0,
            max_lon: 10.0,
            max_lat: 10.0,
        };

        // Line completely outside
        let result = liang_barsky_clip(-5.0, -5.0, -2.0, -2.0, &bounds);
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_vertex_timestamps() {
        let coords = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        ];
        let timestamps = compute_vertex_timestamps(&coords, 0, 3000);

        assert_eq!(timestamps.len(), 4);
        assert_eq!(timestamps[0], 0);
        assert_eq!(timestamps[3], 3000);
        // Middle points should be roughly evenly distributed
        assert!(timestamps[1] > 0 && timestamps[1] < 3000);
        assert!(timestamps[2] > timestamps[1] && timestamps[2] < 3000);
    }

    #[test]
    fn test_clip_trajectory_single_tile() {
        // Trajectory - may span multiple tiles depending on exact coordinates
        let feature = make_linestring_feature(vec![
            vec![-122.4, 37.7],
            vec![-122.41, 37.71],
            vec![-122.42, 37.72],
        ]);

        let config = ClipConfig::default();
        let segments = clip_trajectory(&feature, None, 0, 1000, 10, &config, None, None, None);

        // Should produce at least one segment
        assert!(!segments.is_empty());
        // All segments should have valid coordinates and timestamps
        for seg in &segments {
            assert!(seg.coordinates.len() >= 2);
            assert_eq!(seg.timestamps.len(), seg.coordinates.len());
        }
    }

    #[test]
    fn test_clip_trajectory_crossing_tiles() {
        // Long trajectory crossing multiple tiles
        // San Francisco to Oakland (crosses tile boundaries at zoom 12)
        let feature = make_linestring_feature(vec![
            vec![-122.4194, 37.7749], // SF
            vec![-122.35, 37.78],
            vec![-122.27, 37.80], // Oakland
        ]);

        let config = ClipConfig::default();
        let segments = clip_trajectory(&feature, None, 0, 10000, 12, &config, None, None, None);

        // At zoom 12, this should cross at least 2 tiles
        assert!(
            segments.len() >= 1,
            "Expected at least 1 segment, got {}",
            segments.len()
        );

        // Each segment should have valid timestamps
        for seg in &segments {
            assert!(seg.start_time <= seg.end_time);
            assert!(seg.coordinates.len() >= 2);
        }
    }

    #[test]
    fn test_clip_trajectory_splits_at_antimeridian() {
        // A track that straddles the antimeridian with vertices well away from
        // ±180° on each side (-170° → +165°). The old "both within 10° of the
        // dateline" generator test missed exactly this shape, and the tiler
        // would then sweep a straight line the long way across the whole map,
        // baking a globe-spanning sliver into every tile column. The clipper
        // must split such an edge so no output segment contains a |Δlon| > 180°
        // jump.
        let feature = make_linestring_feature(vec![
            vec![-163.0, 40.0],
            vec![-170.0, 41.0], // last point before the dateline (west side)
            vec![165.0, 42.0],  // first point after the dateline (east side)
            vec![170.0, 43.0],
        ]);

        let config = ClipConfig::default();
        // Zoom 0: the whole world is a single tile, so nothing is clipped at a
        // tile boundary — the antimeridian split is the only thing that can
        // prevent the artifact here.
        let segments = clip_trajectory(&feature, None, 0, 3000, 0, &config, None, None, None);

        assert!(!segments.is_empty(), "expected at least one segment");
        for seg in &segments {
            for w in seg.coordinates.windows(2) {
                let dlon = (w[1].0 - w[0].0).abs();
                assert!(
                    dlon <= 180.0,
                    "segment edge spans {dlon}° of longitude — antimeridian \
                     split failed (coords: {:?})",
                    seg.coordinates
                );
            }
        }
        // The two halves should land on opposite sides of the dateline.
        let has_west = segments
            .iter()
            .any(|s| s.coordinates.iter().all(|c| c.0 < 0.0));
        let has_east = segments
            .iter()
            .any(|s| s.coordinates.iter().all(|c| c.0 > 0.0));
        assert!(
            has_west && has_east,
            "expected runs on both sides of the dateline, got {segments:?}"
        );
    }

    #[test]
    fn test_is_clippable_trajectory() {
        let point_feature = Feature {
            bbox: None,
            geometry: Some(Geometry::new(GeomValue::Point(vec![-122.4, 37.7]))),
            id: None,
            properties: None,
            foreign_members: None,
        };

        let line_feature = make_linestring_feature(vec![vec![-122.4, 37.7], vec![-122.5, 37.8]]);

        // Point is not clippable
        assert!(!is_clippable_trajectory(&point_feature, Some(1000)));

        // LineString without duration is not clippable
        assert!(!is_clippable_trajectory(&line_feature, None));

        // LineString with duration is clippable
        assert!(is_clippable_trajectory(&line_feature, Some(1000)));
    }

    #[test]
    fn test_interpolate_timestamp() {
        assert_eq!(interpolate_timestamp(0, 1000, 0.0), 0);
        assert_eq!(interpolate_timestamp(0, 1000, 1.0), 1000);
        assert_eq!(interpolate_timestamp(0, 1000, 0.5), 500);
        assert_eq!(interpolate_timestamp(1000, 2000, 0.25), 1250);
    }

    #[test]
    fn test_temporal_slicing() {
        // Test that temporal slicing splits segments at boundaries
        let feature = make_linestring_feature(vec![vec![-122.4, 37.7], vec![-122.41, 37.71]]);

        // Config with 1 second temporal granularity
        let config = ClipConfig {
            min_vertices: 2,
            buffer_degrees: 0.001,
            temporal_granularity_ms: Some(1000), // 1 second
            ..Default::default()
        };

        // Trajectory spanning 5 seconds (should create at least 2 temporal slices)
        let segments = clip_trajectory(&feature, None, 0, 5000, 10, &config, None, None, None);

        // Should have at least one segment
        assert!(!segments.is_empty());

        // Each segment should have valid timestamps
        for seg in &segments {
            assert!(seg.start_time <= seg.end_time);
        }
    }

    #[test]
    fn temporal_slicing_splits_every_bucket_on_multi_bucket_jump() {
        // A 2-vertex segment whose single edge spans 5 one-second buckets must
        // yield a slice per crossed bucket, none spanning more than one bucket,
        // and with no duplicated boundary vertices.
        let seg = ClippedSegment {
            tile_x: 0,
            tile_y: 0,
            zoom: 5,
            coordinates: vec![(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
            timestamps: vec![0, 5000],
            vertex_values: Vec::new(),
            vertex_value_matrix: Vec::new(),
            start_time: 0,
            end_time: 5000,
            properties: None,
            feature_id: None,
        };
        let slices = slice_segment_temporally(seg, 1000);
        assert_eq!(slices.len(), 5, "5-bucket edge should split into 5 slices");
        for s in &slices {
            assert!(
                s.end_time - s.start_time <= 1000,
                "slice spans {} ms > one 1000ms bucket",
                s.end_time - s.start_time
            );
            for w in s.coordinates.windows(2) {
                assert!(
                    (w[0].0 - w[1].0).abs() > 1e-12 || (w[0].1 - w[1].1).abs() > 1e-12,
                    "duplicate vertex in slice: {:?}",
                    s.coordinates
                );
            }
        }
        assert_eq!(slices[0].start_time, 0);
        assert_eq!(slices.last().unwrap().end_time, 5000);
    }

    // ------------------------------------------------------------------
    // Supercover tile-traversal tests
    // ------------------------------------------------------------------

    #[test]
    fn supercover_trajectory_inside_one_tile() {
        // All vertices inside tile (163, 395) at zoom 10 (SF area).
        let coords = vec![
            (-122.42, 37.77, 0.0),
            (-122.41, 37.78, 0.0),
            (-122.40, 37.78, 0.0),
        ];
        let tiles = tiles_along_trajectory(&coords, 10);
        assert_eq!(tiles.len(), 1, "expected 1 tile, got {tiles:?}");
        assert!(tiles.contains(&(163, 395)));
    }

    #[test]
    fn supercover_trajectory_parallel_to_tile_edge() {
        // Path that hugs a tile edge — supercover must NOT skip the
        // adjacent tile when the line lies almost exactly on a boundary.
        // tile (163, 395) at z10 spans lat in roughly [37.71, 37.99].
        // Use the boundary lon between tiles 163 and 164 at zoom 10:
        //  lon = (164/1024) * 360 - 180 = -122.34375
        let edge_lon = -122.343_75;
        let coords = vec![(edge_lon - 1e-9, 37.77, 0.0), (edge_lon - 1e-9, 37.78, 0.0)];
        let tiles = tiles_along_trajectory(&coords, 10);
        assert!(
            tiles.contains(&(163, 395)),
            "expected tile (163,395) in {tiles:?}"
        );
    }

    #[test]
    fn supercover_trajectory_diagonal_through_corner() {
        // A diagonal that crosses a (tile_x, tile_y) corner exactly. The
        // supercover must emit both adjacent cells to stay connected.
        // Use a 2-tile diagonal in continuous tile coords by picking
        // (lon, lat) that map to (0.5, 0.5)→(1.5, 1.5) at zoom 1.
        // At zoom 1, n=2, so tile.0 spans lon [-180, 0] and tile.1 [0, 180].
        // Pick lons 0 ± 90 to hit centres of tiles 0 and 1.
        let coords = vec![(-90.0, 45.0, 0.0), (90.0, -45.0, 0.0)];
        let tiles = tiles_along_trajectory(&coords, 1);
        // Should touch at least 3 cells (start, opposite corner, one of the
        // two diagonal-adjacent cells).
        assert!(
            tiles.len() >= 3,
            "diagonal should touch >=3 tiles, got {tiles:?}"
        );
    }

    #[test]
    fn supercover_zero_length_segment() {
        // Two identical vertices => one tile only.
        let coords = vec![(-122.42, 37.77, 0.0), (-122.42, 37.77, 0.0)];
        let tiles = tiles_along_trajectory(&coords, 10);
        assert_eq!(tiles.len(), 1);
    }

    #[test]
    fn supercover_near_pole_clamps() {
        // Web Mercator diverges past ±85.0511 — coordinates beyond should be
        // clamped, not panic or produce garbage tile indices.
        let coords = vec![(0.0, 89.0, 0.0), (10.0, 89.5, 0.0)];
        let tiles = tiles_along_trajectory(&coords, 5);
        let n = 1u32 << 5;
        for (x, y) in &tiles {
            assert!(*x < n && *y < n, "tile ({x},{y}) out of bounds for zoom 5");
        }
        assert!(!tiles.is_empty());
    }

    #[test]
    fn supercover_handles_antimeridian_segment() {
        // A segment crossing ±180° is clamped per-vertex (we don't split it),
        // but it must not panic and must return tiles only on one side. A
        // separate wrap-aware splitter is a v3 concern; document by test.
        let coords = vec![(179.5, 0.0, 0.0), (179.99, 0.0, 0.0)];
        let tiles = tiles_along_trajectory(&coords, 5);
        let n = 1u32 << 5;
        for (x, y) in &tiles {
            assert!(*x < n && *y < n);
        }
    }

    // ------------------------------------------------------------------
    // Polygon ring clipping (Sutherland–Hodgman) tests
    // ------------------------------------------------------------------

    fn closed_ring(pts: &[(f64, f64)]) -> Vec<Vec<f64>> {
        let mut r: Vec<Vec<f64>> = pts.iter().map(|(x, y)| vec![*x, *y]).collect();
        r.push(r[0].clone());
        r
    }

    #[test]
    fn sutherland_hodgman_ring_inside_is_unchanged() {
        let bounds = TileBounds {
            min_lon: 0.0,
            min_lat: 0.0,
            max_lon: 10.0,
            max_lat: 10.0,
        };
        let ring = closed_ring(&[(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]);
        let out = sutherland_hodgman_ring(&ring, &bounds);
        assert_eq!(out, ring, "fully-inside ring must survive verbatim");
    }

    #[test]
    fn sutherland_hodgman_ring_outside_is_empty() {
        let bounds = TileBounds {
            min_lon: 0.0,
            min_lat: 0.0,
            max_lon: 10.0,
            max_lat: 10.0,
        };
        let ring = closed_ring(&[(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)]);
        assert!(sutherland_hodgman_ring(&ring, &bounds).is_empty());
    }

    #[test]
    fn sutherland_hodgman_ring_covering_rect_yields_rect() {
        // A ring that fully contains the clip rect must clip to the rect
        // itself — the interior-tile case of a large polygon.
        let bounds = TileBounds {
            min_lon: 0.0,
            min_lat: 0.0,
            max_lon: 10.0,
            max_lat: 10.0,
        };
        let ring = closed_ring(&[(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)]);
        let out = sutherland_hodgman_ring(&ring, &bounds);
        assert_eq!(out.len(), 5, "rect = 4 corners + closure, got {out:?}");
        assert_eq!(out.first(), out.last(), "output ring must be closed");
        for c in &out {
            assert!((0.0..=10.0).contains(&c[0]) && (0.0..=10.0).contains(&c[1]));
        }
    }

    #[test]
    fn sutherland_hodgman_ring_straddling_is_clipped_and_closed() {
        let bounds = TileBounds {
            min_lon: 0.0,
            min_lat: 0.0,
            max_lon: 10.0,
            max_lat: 10.0,
        };
        // Triangle poking into the rect from the left.
        let ring = closed_ring(&[(-5.0, 5.0), (5.0, 2.0), (5.0, 8.0)]);
        let out = sutherland_hodgman_ring(&ring, &bounds);
        assert!(out.len() >= 4, "clipped ring must stay a ring: {out:?}");
        assert_eq!(out.first(), out.last());
        for c in &out {
            assert!(c[0] >= -1e-9 && c[0] <= 10.0 + 1e-9, "x escaped: {out:?}");
        }
    }

    #[test]
    fn clip_polygons_to_tiles_covers_and_preserves_holes() {
        // A square around the 4-tile corner at (0°, 0°) with a centred hole:
        // every covered tile keeps its share of BOTH rings.
        let exterior = closed_ring(&[(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]);
        let hole = closed_ring(&[(-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)]);
        let polys = vec![vec![exterior, hole]];
        let zoom = 10u8;
        let pieces = clip_polygons_to_tiles(&polys, zoom, 0.0, None);
        assert_eq!(
            pieces.len(),
            4,
            "expected the 4 corner tiles, got {pieces:?}"
        );
        let tiles: Vec<(u32, u32)> = pieces.iter().map(|(t, _)| *t).collect();
        assert_eq!(tiles, vec![(511, 511), (511, 512), (512, 511), (512, 512)]);
        for ((x, y), tile_polys) in &pieces {
            assert_eq!(tile_polys.len(), 1);
            let rings = &tile_polys[0];
            assert_eq!(rings.len(), 2, "tile ({x},{y}) lost the hole: {rings:?}");
            for ring in rings {
                assert!(ring.len() >= 4, "degenerate ring in ({x},{y})");
                assert_eq!(ring.first(), ring.last(), "ring not closed in ({x},{y})");
            }
        }
        // Watertight seams: both tiles clip the SAME ring against the same
        // tile-edge line with identical arithmetic (unbuffered rect), so the
        // seam vertices they emit are bit-identical — no overlap strip to
        // double-blend under translucent fills, no gap. Compare the two
        // pieces' near-seam vertex sets bit-for-bit.
        let piece = |tx: u32, ty: u32| -> &PolygonRings {
            &pieces.iter().find(|(t, _)| *t == (tx, ty)).unwrap().1[0]
        };
        let seam_bits = |rings: &PolygonRings| -> Vec<(u64, u64)> {
            let mut v: Vec<(u64, u64)> = rings[0]
                .iter()
                .filter(|c| c[0].abs() <= 1e-9)
                .map(|c| (c[0].to_bits(), c[1].to_bits()))
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let left = seam_bits(piece(511, 511));
        let right = seam_bits(piece(512, 511));
        assert!(!left.is_empty(), "left piece should touch the lon=0 seam");
        assert_eq!(
            left, right,
            "seam vertices must be bit-identical across the tile edge"
        );

        // Single-tile restriction (the stt-serve path) returns the
        // byte-identical piece the full sweep produced for that tile.
        let restricted = clip_polygons_to_tiles(&polys, zoom, 0.0, Some((512, 511)));
        assert_eq!(restricted.len(), 1);
        assert_eq!(restricted[0].0, (512, 511));
        assert_eq!(
            &restricted[0].1,
            &pieces.iter().find(|(t, _)| *t == (512, 511)).unwrap().1
        );
        // A target outside the bbox sweeps nothing.
        assert!(clip_polygons_to_tiles(&polys, zoom, 0.0, Some((0, 0))).is_empty());
    }

    // ------------------------------------------------------------------
    // Antimeridian polygon splitting (A1)
    // ------------------------------------------------------------------

    fn poly_from(rings: &[&[(f64, f64)]]) -> PolygonRings {
        rings.iter().map(|r| closed_ring(r)).collect()
    }

    /// Absolute planar area of a ring in its own (unwrapped) continuous frame.
    fn unwrapped_area(ring: &[Vec<f64>]) -> f64 {
        let pts = unwrap_ring(ring).expect("ring must unwrap");
        signed_ring_area(&closed_ring_from_pts(&pts)).abs()
    }

    #[test]
    fn unique_seam_meridian_rejects_wide_and_empty_spans() {
        assert_eq!(unique_seam_meridian(177.0, 183.0), Some(180.0));
        assert_eq!(unique_seam_meridian(-185.0, -175.0), Some(-180.0));
        // ≥360° span (pole-enclosing / degenerate) has no unique seam.
        assert_eq!(unique_seam_meridian(0.0, 400.0), None);
        // No odd-180 multiple strictly inside.
        assert_eq!(unique_seam_meridian(10.0, 170.0), None);
    }

    #[test]
    fn split_polygon_square_straddling_plus_180() {
        // Fiji-like square: raw lon 177 → -177 (crosses +180), lat 10 → 20.
        let poly = poly_from(&[&[(177.0, 10.0), (-177.0, 10.0), (-177.0, 20.0), (177.0, 20.0)]]);
        let pieces = split_polygon_at_antimeridian(&poly);
        assert_eq!(
            pieces.len(),
            2,
            "must split into west + east, got {pieces:?}"
        );
        for piece in &pieces {
            assert_eq!(piece.len(), 1, "single exterior, no holes");
            let ext = &piece[0];
            assert_eq!(ext.first(), ext.last(), "ring must stay closed");
            for c in ext {
                assert!(
                    (-180.0..=180.0).contains(&c[0]),
                    "lon {} out of range",
                    c[0]
                );
            }
            assert!(
                signed_ring_area(ext) > 0.0,
                "exterior must be CCW (positive area): {ext:?}"
            );
            // No wrap edge survives.
            for w in ext.windows(2) {
                assert!(
                    (w[1][0] - w[0][0]).abs() <= 180.0,
                    "surviving |Δlon|>180 edge in {ext:?}"
                );
            }
        }
        // West piece hugs +180, east piece hugs -180.
        assert!(
            pieces[0][0].iter().any(|c| (c[0] - 180.0).abs() < 1e-12),
            "west piece must touch the +180 seam: {:?}",
            pieces[0][0]
        );
        assert!(
            pieces[1][0].iter().any(|c| (c[0] + 180.0).abs() < 1e-12),
            "east piece must touch the -180 seam: {:?}",
            pieces[1][0]
        );
        // Planar area conserved (6°×10° = 60, split 3+3 in lon).
        let orig = unwrapped_area(&poly[0]);
        let sum: f64 = pieces.iter().map(|p| signed_ring_area(&p[0]).abs()).sum();
        assert!((orig - 60.0).abs() < 1e-9, "orig area {orig} != 60");
        assert!(
            (orig - sum).abs() < 1e-9,
            "area not conserved: {orig} vs {sum}"
        );
    }

    #[test]
    fn split_polygon_square_straddling_minus_180_chukotka() {
        // Crosses -180: raw lon -178 → 178, vertex 0 on the NEGATIVE side.
        let poly = poly_from(&[&[(-178.0, 10.0), (178.0, 10.0), (178.0, 20.0), (-178.0, 20.0)]]);
        let pieces = split_polygon_at_antimeridian(&poly);
        assert_eq!(pieces.len(), 2);
        let (mut touches_plus, mut touches_minus) = (false, false);
        for piece in &pieces {
            let ext = &piece[0];
            assert_eq!(ext.first(), ext.last());
            assert!(signed_ring_area(ext) > 0.0, "exterior CCW");
            for c in ext {
                assert!((-180.0..=180.0).contains(&c[0]));
            }
            for w in ext.windows(2) {
                assert!((w[1][0] - w[0][0]).abs() <= 180.0);
            }
            if ext.iter().any(|c| (c[0] - 180.0).abs() < 1e-12) {
                touches_plus = true;
            }
            if ext.iter().any(|c| (c[0] + 180.0).abs() < 1e-12) {
                touches_minus = true;
            }
        }
        assert!(
            touches_plus && touches_minus,
            "pieces must land on BOTH sides of the dateline"
        );
        let orig = unwrapped_area(&poly[0]);
        let sum: f64 = pieces.iter().map(|p| signed_ring_area(&p[0]).abs()).sum();
        assert!(
            (orig - sum).abs() < 1e-9,
            "area not conserved: {orig} vs {sum}"
        );
    }

    #[test]
    fn split_polygon_straddling_hole_retained_cw_both_sides() {
        // Exterior straddles +180 (177 → -177); a centred hole ALSO straddles
        // it (178 → -178), so it splits and nests into BOTH hemispheres, CW,
        // tangent to the seam.
        let poly = poly_from(&[
            &[(177.0, 10.0), (-177.0, 10.0), (-177.0, 20.0), (177.0, 20.0)],
            &[(178.0, 13.0), (-178.0, 13.0), (-178.0, 17.0), (178.0, 17.0)],
        ]);
        let pieces = split_polygon_at_antimeridian(&poly);
        assert_eq!(pieces.len(), 2);
        for piece in &pieces {
            assert_eq!(piece.len(), 2, "each side keeps exterior + hole: {piece:?}");
            assert!(signed_ring_area(&piece[0]) > 0.0, "exterior CCW");
            assert!(
                signed_ring_area(&piece[1]) < 0.0,
                "hole must be CW (negative area): {:?}",
                piece[1]
            );
            for ring in piece {
                assert_eq!(ring.first(), ring.last());
                for c in ring {
                    assert!((-180.0..=180.0).contains(&c[0]));
                }
            }
            // Seam-tangent hole: a hole vertex lies exactly on the ±180 meridian
            // (accepted artifact — tolerated, not an OGC-simplicity failure).
            assert!(
                piece[1].iter().any(|c| (c[0].abs() - 180.0).abs() < 1e-12),
                "straddling hole must be tangent to the seam: {:?}",
                piece[1]
            );
        }
        // Net (shell − hole) area conserved across the split: 60 − 16 = 44.
        let orig = unwrapped_area(&poly[0]) - unwrapped_area(&poly[1]);
        let sum: f64 = pieces
            .iter()
            .map(|p| signed_ring_area(&p[0]).abs() - signed_ring_area(&p[1]).abs())
            .sum();
        assert!((orig - 44.0).abs() < 1e-9, "net orig area {orig} != 44");
        assert!(
            (orig - sum).abs() < 1e-9,
            "holed area not conserved: {orig} vs {sum}"
        );
    }

    #[test]
    fn split_polygon_hole_within_one_hemisphere_stays_there() {
        // Exterior straddles +180; the hole is entirely on the WEST side (raw
        // 178..179, unwrapped < seam=180): only the west piece keeps it, and the
        // vanished east hole is dropped (drop-vanished-hole rule).
        let poly = poly_from(&[
            &[(177.0, 10.0), (-177.0, 10.0), (-177.0, 20.0), (177.0, 20.0)],
            &[(178.0, 13.0), (179.0, 13.0), (179.0, 17.0), (178.0, 17.0)],
        ]);
        let pieces = split_polygon_at_antimeridian(&poly);
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].len(), 2, "west keeps exterior + the hole");
        assert_eq!(pieces[1].len(), 1, "east has no hole (clipped empty)");
        assert!(
            signed_ring_area(&pieces[0][1]) < 0.0,
            "retained hole must be CW"
        );
    }

    #[test]
    fn split_polygon_seam_latitudes_are_bit_identical() {
        // The west (+180) and east (-180) pieces cut the SAME unwrapped edges
        // with the SAME t, so their seam-vertex LATITUDES match bit-for-bit
        // (mirrors the watertight-seam pin in
        // clip_polygons_to_tiles_covers_and_preserves_holes). Slanted seam edges
        // make the interpolated latitudes non-round.
        let poly = poly_from(&[&[(176.0, 10.0), (-176.0, 14.0), (-176.0, 26.0), (176.0, 22.0)]]);
        let pieces = split_polygon_at_antimeridian(&poly);
        assert_eq!(pieces.len(), 2);
        let seam_lats = |ring: &[Vec<f64>], seam_lon: f64| -> Vec<u64> {
            let mut v: Vec<u64> = ring
                .iter()
                .filter(|c| (c[0] - seam_lon).abs() < 1e-12)
                .map(|c| c[1].to_bits())
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let west = seam_lats(&pieces[0][0], 180.0);
        let east = seam_lats(&pieces[1][0], -180.0);
        assert!(!west.is_empty(), "west must touch the seam");
        assert_eq!(
            west, east,
            "seam latitudes must be bit-identical across ±180"
        );
    }

    #[test]
    fn split_polygon_pole_enclosing_returns_empty() {
        // A ring circling the north pole: lon 0 → 90 → 180 → -90 winds a net
        // +360° in longitude, so it is unsplittable and dead-letters (empty).
        let poly = poly_from(&[&[(0.0, 85.0), (90.0, 85.0), (180.0, 85.0), (-90.0, 85.0)]]);
        assert!(
            split_polygon_at_antimeridian(&poly).is_empty(),
            "pole-enclosing ring must not split"
        );
    }

    #[test]
    fn test_tile_bounds_calculation() {
        // Verify tile bounds are calculated correctly for tile containing SF
        let bounds = TileBounds::from_tile(163, 395, 10);

        // Tile 163,395 at zoom 10 should contain San Francisco area
        // Check longitude covers -122.4
        assert!(
            bounds.min_lon < -122.4 && bounds.max_lon > -122.4,
            "Longitude bounds wrong: {:?}",
            bounds
        );
        // Latitude should be in the 37-38 range for SF
        assert!(
            bounds.min_lat > 35.0 && bounds.max_lat < 40.0,
            "Latitude bounds wrong: min={}, max={}",
            bounds.min_lat,
            bounds.max_lat
        );
    }
}
