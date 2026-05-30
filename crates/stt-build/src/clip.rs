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
use crate::simplify::simplify_for_zoom;
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
    /// Optional temporal granularity for slicing long trajectories (in milliseconds)
    /// If set, trajectories crossing temporal boundaries will be split
    pub temporal_granularity_ms: Option<u64>,
    /// Enable line simplification for lower zoom levels
    pub simplify: bool,
    /// Maximum zoom level to apply simplification (higher zooms keep full detail)
    pub simplify_max_zoom: u8,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            min_vertices: 2,
            // Small buffer (~100m at equator) to ensure visual continuity
            buffer_degrees: 0.001,
            // No temporal slicing by default
            temporal_granularity_ms: None,
            // Simplification disabled by default
            simplify: false,
            simplify_max_zoom: 14,
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
fn tiles_along_trajectory(
    coords: &[(f64, f64, f64)],
    zoom: u8,
) -> HashSet<(u32, u32)> {
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
    let step_x: i64 = if dx > 0.0 { 1 } else if dx < 0.0 { -1 } else { 0 };
    let step_y: i64 = if dy > 0.0 { 1 } else if dy < 0.0 { -1 } else { 0 };

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

    let t_delta_x = if step_x != 0 { (step_x as f64) * inv_dx } else { f64::INFINITY };
    let t_delta_y = if step_y != 0 { (step_y as f64) * inv_dy } else { f64::INFINITY };

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
fn clip_trajectory_to_tile(
    coords: &[(f64, f64, f64)],
    timestamps: &[u64],
    bounds: &TileBounds,
) -> Option<(Vec<(f64, f64, f64)>, Vec<u64>)> {
    if coords.len() < 2 {
        return None;
    }

    let mut clipped_coords: Vec<(f64, f64, f64)> = Vec::new();
    let mut clipped_times: Vec<u64> = Vec::new();

    // Process each line segment
    for i in 0..coords.len() - 1 {
        let (x0, y0, alt0) = coords[i];
        let (x1, y1, alt1) = coords[i + 1];
        let time0 = timestamps[i];
        let time1 = timestamps[i + 1];

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

            // Calculate the clipped end point
            let (end_x, end_y) = if t1 < 1.0 {
                interpolate_point(x0, y0, x1, y1, t1)
            } else {
                (x1, y1)
            };
            let end_alt = interpolate_alt(alt0, alt1, t1);
            let end_time = interpolate_timestamp(time0, time1, t1);

            // Add start point if not a duplicate of last point
            let should_add_start = clipped_coords.is_empty()
                || (clipped_coords.last().unwrap().0 - start_x).abs() > 1e-9
                || (clipped_coords.last().unwrap().1 - start_y).abs() > 1e-9;

            if should_add_start {
                clipped_coords.push((start_x, start_y, start_alt));
                clipped_times.push(start_time);
            }

            // Add end point if different from start point
            if (end_x - start_x).abs() > 1e-9 || (end_y - start_y).abs() > 1e-9 {
                clipped_coords.push((end_x, end_y, end_alt));
                clipped_times.push(end_time);
            }
        }
    }

    if clipped_coords.len() >= 2 {
        Some((clipped_coords, clipped_times))
    } else {
        None
    }
}

/// Slice a segment at temporal boundaries
///
/// If the segment spans multiple temporal chunks (based on granularity),
/// split it into multiple segments at those boundaries.
fn slice_segment_temporally(
    segment: ClippedSegment,
    granularity_ms: u64,
) -> Vec<ClippedSegment> {
    if segment.coordinates.len() < 2 || granularity_ms == 0 {
        return vec![segment];
    }

    let start_chunk = segment.start_time / granularity_ms;
    let end_chunk = segment.end_time / granularity_ms;

    // If entirely within one chunk, no splitting needed
    if start_chunk == end_chunk {
        return vec![segment];
    }

    let mut slices = Vec::new();
    let mut current_coords: Vec<(f64, f64, f64)> = Vec::new();
    let mut current_times: Vec<u64> = Vec::new();
    let mut current_chunk = start_chunk;

    for i in 0..segment.coordinates.len() {
        let coord = segment.coordinates[i];
        let time = segment.timestamps[i];
        let chunk = time / granularity_ms;

        // If we're crossing into a new chunk, finalize current slice and start new one
        if chunk > current_chunk && current_coords.len() >= 2 {
            // We need to interpolate the boundary point
            if i > 0 {
                let boundary_time = (current_chunk + 1) * granularity_ms;
                let prev_time = segment.timestamps[i - 1];
                let curr_time = time;

                if prev_time < boundary_time && boundary_time <= curr_time {
                    // Interpolate point at boundary
                    let t = if curr_time > prev_time {
                        (boundary_time - prev_time) as f64 / (curr_time - prev_time) as f64
                    } else {
                        0.0
                    };

                    let prev_coord = segment.coordinates[i - 1];
                    let boundary_coord = (
                        prev_coord.0 + t * (coord.0 - prev_coord.0),
                        prev_coord.1 + t * (coord.1 - prev_coord.1),
                        prev_coord.2 + t * (coord.2 - prev_coord.2),
                    );

                    // Add boundary point to current slice
                    current_coords.push(boundary_coord);
                    current_times.push(boundary_time);

                    // Finalize current slice
                    if current_coords.len() >= 2 {
                        slices.push(ClippedSegment {
                            tile_x: segment.tile_x,
                            tile_y: segment.tile_y,
                            zoom: segment.zoom,
                            coordinates: current_coords.clone(),
                            timestamps: current_times.clone(),
                            start_time: *current_times.first().unwrap(),
                            end_time: *current_times.last().unwrap(),
                            properties: segment.properties.clone(),
                            feature_id: segment.feature_id.clone(),
                        });
                    }

                    // Start new slice with boundary point
                    current_coords = vec![boundary_coord];
                    current_times = vec![boundary_time];
                }
            }

            current_chunk = chunk;
        }

        current_coords.push(coord);
        current_times.push(time);
    }

    // Finalize last slice
    if current_coords.len() >= 2 {
        slices.push(ClippedSegment {
            tile_x: segment.tile_x,
            tile_y: segment.tile_y,
            zoom: segment.zoom,
            coordinates: current_coords,
            timestamps: current_times.clone(),
            start_time: *current_times.first().unwrap(),
            end_time: *current_times.last().unwrap(),
            properties: segment.properties.clone(),
            feature_id: segment.feature_id.clone(),
        });
    }

    if slices.is_empty() {
        vec![segment]
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

    // Apply simplification for lower zoom levels if enabled
    let coords = if config.simplify {
        simplify_for_zoom(&coords, zoom, config.simplify_max_zoom)
    } else {
        coords
    };

    // Skip if simplification reduced below minimum
    if coords.len() < config.min_vertices {
        return vec![];
    }

    // Compute per-vertex timestamps. Prefer supplied times when present and
    // alignment was preserved through simplification.
    let simplification_preserved_alignment = coords.len() == original_vertex_count;
    let timestamps: Vec<u64> = match supplied_vertex_times {
        Some(supplied)
            if simplification_preserved_alignment && supplied.len() == coords.len() =>
        {
            supplied.to_vec()
        }
        _ => compute_vertex_timestamps(&coords, start_time, end_time),
    };

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
        let crosses_antimeridian =
            !at_end && (coords[split].0 - coords[split - 1].0).abs() > 180.0;
        if !(at_end || crosses_antimeridian) {
            continue;
        }
        let run_coords = &coords[run_start..split];
        let run_times = &timestamps[run_start..split];
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
            if let Some((clipped_coords, clipped_times)) =
                clip_trajectory_to_tile(run_coords, run_times, &buffered_bounds)
            {
                let seg_start_time = *clipped_times.first().unwrap();
                let seg_end_time = *clipped_times.last().unwrap();

                let segment = ClippedSegment {
                    tile_x,
                    tile_y,
                    zoom,
                    coordinates: clipped_coords,
                    timestamps: clipped_times,
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
        let segments = clip_trajectory(&feature, None, 0, 1000, 10, &config, None);

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
        let segments = clip_trajectory(&feature, None, 0, 10000, 12, &config, None);

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
        let segments = clip_trajectory(&feature, None, 0, 3000, 0, &config, None);

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
        let feature = make_linestring_feature(vec![
            vec![-122.4, 37.7],
            vec![-122.41, 37.71],
        ]);

        // Config with 1 second temporal granularity
        let config = ClipConfig {
            min_vertices: 2,
            buffer_degrees: 0.001,
            temporal_granularity_ms: Some(1000), // 1 second
            ..Default::default()
        };

        // Trajectory spanning 5 seconds (should create at least 2 temporal slices)
        let segments = clip_trajectory(&feature, None, 0, 5000, 10, &config, None);

        // Should have at least one segment
        assert!(!segments.is_empty());

        // Each segment should have valid timestamps
        for seg in &segments {
            assert!(seg.start_time <= seg.end_time);
        }
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
        let coords = vec![
            (edge_lon - 1e-9, 37.77, 0.0),
            (edge_lon - 1e-9, 37.78, 0.0),
        ];
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
        assert!(tiles.len() >= 3, "diagonal should touch >=3 tiles, got {tiles:?}");
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

