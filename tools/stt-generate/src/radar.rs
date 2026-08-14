//! Build-time radar processing for the NEXRAD `storms` generator.
//!
//! Pure, I/O-free helpers that turn decoded polar reflectivity sweeps into the
//! finished vector geometry the browser renders:
//!
//! 1. [`MosaicGrid`] rasterises gates from many radars onto one equirectangular
//!    analysis grid, **max-combining** reflectivity per cell so overlapping
//!    sites dedup deterministically (the standard national-mosaic rule).
//! 2. [`contour_bands`] runs marching-squares isobands over the grid to produce
//!    filled reflectivity bands (the animated precipitation field).
//! 3. [`detect_cells`] finds storm cells via connected components on a
//!    high-dBZ mask (reflectivity-weighted centroid + area + peak).
//! 4. [`track_cells`] links cells across scans (SCIT-style nearest-centroid)
//!    into tracks with per-vertex intensity over time.
//!
//! The beam geometry (polar gate → lon/lat) is reused from
//! [`nexrad_model::geo::RadarCoordinateSystem`], which implements the standard
//! 4/3-earth-radius propagation model — no need to re-derive it here.
//!
//! Everything here is deterministic given the same inputs (commutative
//! max-combine, raster-order component labelling, descending-intensity track
//! matching, and a final stable sort by the caller) so the content-addressed
//! packs are reproducible.

use nexrad_model::data::{GateStatus, SweepField};
use nexrad_model::geo::RadarCoordinateSystem;
use std::collections::VecDeque;

/// Mean earth radius in meters (for the equirectangular grid spacing).
const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Meters per degree of latitude (mean).
const M_PER_DEG_LAT: f64 = std::f64::consts::PI * EARTH_RADIUS_M / 180.0; // ≈ 111_195

/// An equirectangular analysis grid that mosaics reflectivity gates from any
/// number of radars by taking the per-cell maximum dBZ.
///
/// Cells are row-major: `cells[iy * nx + ix]`, with `ix` the longitude bin and
/// `iy` the latitude bin, `iy = 0` at `min_lat` (north-increasing). This matches
/// the `contour` crate's `values[y * dx + x]` convention and its
/// `point * step + origin` transform, so contour output comes back in lon/lat
/// directly. Empty cells hold `f32::NEG_INFINITY`.
pub struct MosaicGrid {
    pub min_lon: f64,
    pub min_lat: f64,
    pub dlon: f64,
    pub dlat: f64,
    pub nx: usize,
    pub ny: usize,
    cells: Vec<f32>,
}

impl MosaicGrid {
    /// Build an empty grid covering `[min_lat,max_lat] × [min_lon,max_lon]` with
    /// roughly `cell_km`-square cells (equirectangular at the bbox mid-latitude).
    pub fn new(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64, cell_km: f64) -> Self {
        let mid_lat = (min_lat + max_lat) / 2.0;
        let m_per_deg_lon = M_PER_DEG_LAT * mid_lat.to_radians().cos();
        let dlat = (cell_km * 1000.0) / M_PER_DEG_LAT;
        let dlon = (cell_km * 1000.0) / m_per_deg_lon;
        let nx = (((max_lon - min_lon) / dlon).ceil() as usize).max(1);
        let ny = (((max_lat - min_lat) / dlat).ceil() as usize).max(1);
        Self {
            min_lon,
            min_lat,
            dlon,
            dlat,
            nx,
            ny,
            cells: vec![f32::NEG_INFINITY; nx * ny],
        }
    }

    /// Geographic center of cell `(ix, iy)`.
    #[inline]
    pub fn cell_center(&self, ix: usize, iy: usize) -> [f64; 2] {
        [
            self.min_lon + (ix as f64 + 0.5) * self.dlon,
            self.min_lat + (iy as f64 + 0.5) * self.dlat,
        ]
    }

    /// Approximate cell area in km² (equirectangular; exact enough for cell sizing).
    #[inline]
    pub fn cell_area_km2(&self) -> f64 {
        // dlat·dlon are degrees; convert each side to km at the grid latitude.
        let mid_lat = self.min_lat + (self.ny as f64 * self.dlat) / 2.0;
        let side_lat_km = self.dlat * M_PER_DEG_LAT / 1000.0;
        let side_lon_km = self.dlon * M_PER_DEG_LAT * mid_lat.to_radians().cos() / 1000.0;
        side_lat_km * side_lon_km
    }

    /// True if no gate has been added.
    pub fn is_empty(&self) -> bool {
        self.cells.iter().all(|c| !c.is_finite())
    }

    /// Rasterise one decoded reflectivity sweep from a single radar onto the grid,
    /// max-combining each valid gate into its covering cell.
    pub fn add_sweep_field(&mut self, cs: &RadarCoordinateSystem, field: &SweepField) {
        let elev = field.elevation_degrees();
        let first_gate_km = field.first_gate_range_km();
        let gate_interval_km = field.gate_interval_km();
        let gate_count = field.gate_count();
        let azimuths = field.azimuths();
        let values = field.values();
        let statuses = field.statuses();

        for ai in 0..field.azimuth_count() {
            let az = azimuths[ai];
            let row = ai * gate_count;
            for gi in 0..gate_count {
                let k = row + gi;
                if statuses[k] != GateStatus::Valid {
                    continue;
                }
                let dbz = values[k];
                if !dbz.is_finite() {
                    continue;
                }
                let p = cs.gate_location(az, elev, gi, first_gate_km, gate_interval_km);
                let fx = (p.longitude - self.min_lon) / self.dlon;
                let fy = (p.latitude - self.min_lat) / self.dlat;
                if fx < 0.0 || fy < 0.0 {
                    continue;
                }
                let ix = fx as usize;
                let iy = fy as usize;
                if ix >= self.nx || iy >= self.ny {
                    continue;
                }
                let cell = &mut self.cells[iy * self.nx + ix];
                if dbz > *cell {
                    *cell = dbz;
                }
            }
        }
    }
}

/// A filled reflectivity band as a polygon (exterior ring first, holes after).
pub struct BandPolygon {
    /// Lower dBZ threshold of the band.
    pub dbz_lo: i32,
    /// Upper dBZ threshold of the band.
    pub dbz_hi: i32,
    /// Rings: `rings[0]` exterior, the rest holes. Each ring is `[lon, lat]` pairs.
    pub rings: Vec<Vec<[f64; 2]>>,
}

/// Approximate polygon ring area in km² (shoelace in degrees, scaled to the
/// ring's mean latitude). Used only to drop sub-resolution speckle.
fn ring_area_km2(ring: &[[f64; 2]]) -> f64 {
    if ring.len() < 4 {
        return 0.0;
    }
    let mut area2 = 0.0;
    let mut mean_lat = 0.0;
    for w in ring.windows(2) {
        area2 += w[0][0] * w[1][1] - w[1][0] * w[0][1];
        mean_lat += w[0][1];
    }
    mean_lat /= (ring.len() - 1) as f64;
    let deg_area = (area2 * 0.5).abs();
    let km_per_deg_lat = M_PER_DEG_LAT / 1000.0;
    let km_per_deg_lon = km_per_deg_lat * mean_lat.to_radians().cos();
    deg_area * km_per_deg_lat * km_per_deg_lon
}

/// Contour the grid into filled isobands between consecutive `thresholds`.
///
/// `thresholds` must be ascending (e.g. `[20,25,...,65]`). Empty cells are
/// filled with `empty_fill` (well below the lowest threshold) so they read as
/// "no echo". Each output band carries its lower threshold as `dbz_band`.
/// Polygons whose exterior is smaller than `min_area_km2` are dropped — this
/// removes the speckle of disconnected low-dBZ patches that otherwise explodes
/// the polygon count.
pub fn contour_bands(
    grid: &MosaicGrid,
    thresholds: &[f64],
    empty_fill: f64,
    min_area_km2: f64,
) -> Vec<BandPolygon> {
    if thresholds.len() < 2 {
        return Vec::new();
    }
    let values: Vec<f64> = grid
        .cells
        .iter()
        .map(|&v| if v.is_finite() { v as f64 } else { empty_fill })
        .collect();

    let builder = contour::ContourBuilder::new(grid.nx, grid.ny, true)
        .x_origin(grid.min_lon)
        .y_origin(grid.min_lat)
        .x_step(grid.dlon)
        .y_step(grid.dlat);

    let bands = match builder.isobands(&values, thresholds) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };

    let mut out = Vec::new();
    for band in bands {
        let dbz_lo = band.min_v().round() as i32;
        let dbz_hi = band.max_v().round() as i32;
        // Skip the background band (everything below the first precip threshold).
        if (band.min_v()) < thresholds[0] - 0.5 {
            continue;
        }
        for poly in band.geometry().0.iter() {
            let mut rings: Vec<Vec<[f64; 2]>> = Vec::with_capacity(1 + poly.interiors().len());
            let exterior: Vec<[f64; 2]> = poly.exterior().0.iter().map(|c| [c.x, c.y]).collect();
            if exterior.len() < 4 || ring_area_km2(&exterior) < min_area_km2 {
                continue;
            }
            rings.push(exterior);
            for hole in poly.interiors() {
                let ring: Vec<[f64; 2]> = hole.0.iter().map(|c| [c.x, c.y]).collect();
                if ring.len() >= 4 {
                    rings.push(ring);
                }
            }
            out.push(BandPolygon {
                dbz_lo,
                dbz_hi,
                rings,
            });
        }
    }
    out
}

/// A storm cell detected in one scan: reflectivity-weighted centroid, peak dBZ,
/// and footprint area.
#[derive(Debug, Clone)]
pub struct Cell {
    pub lon: f64,
    pub lat: f64,
    pub max_dbz: f32,
    pub area_km2: f64,
}

/// Identify storm cells as connected components of cells at or above `cell_dbz`,
/// keeping those at least `min_cell_km2` in area. Components are labelled in
/// raster-scan order (deterministic) and the result is sorted by `(lon, lat)`.
pub fn detect_cells(grid: &MosaicGrid, cell_dbz: f64, min_cell_km2: f64) -> Vec<Cell> {
    let nx = grid.nx;
    let ny = grid.ny;
    let cell_area = grid.cell_area_km2();
    let mut visited = vec![false; nx * ny];
    let mut cells = Vec::new();

    for start in 0..nx * ny {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        if (grid.cells[start] as f64) < cell_dbz || !grid.cells[start].is_finite() {
            continue;
        }

        // Flood-fill this component (4-connected).
        let mut queue = VecDeque::new();
        queue.push_back(start);
        let mut count = 0usize;
        let mut max_dbz = f32::MIN;
        let (mut sum_w, mut sum_lon, mut sum_lat) = (0.0f64, 0.0f64, 0.0f64);

        while let Some(c) = queue.pop_front() {
            let ix = c % nx;
            let iy = c / nx;
            let dv = grid.cells[c];
            count += 1;
            if dv > max_dbz {
                max_dbz = dv;
            }
            // Reflectivity-weighted centroid (floor weight keeps it positive).
            let w = (dv as f64).max(1.0);
            let center = grid.cell_center(ix, iy);
            sum_w += w;
            sum_lon += center[0] * w;
            sum_lat += center[1] * w;

            for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let nxi = ix as i64 + dx;
                let nyi = iy as i64 + dy;
                if nxi < 0 || nyi < 0 || nxi as usize >= nx || nyi as usize >= ny {
                    continue;
                }
                let nc = (nyi as usize) * nx + nxi as usize;
                if visited[nc] {
                    continue;
                }
                visited[nc] = true;
                if (grid.cells[nc] as f64) >= cell_dbz && grid.cells[nc].is_finite() {
                    queue.push_back(nc);
                }
            }
        }

        let area_km2 = count as f64 * cell_area;
        if area_km2 >= min_cell_km2 && sum_w > 0.0 {
            cells.push(Cell {
                lon: sum_lon / sum_w,
                lat: sum_lat / sum_w,
                max_dbz,
                area_km2,
            });
        }
    }

    cells.sort_by(|a, b| {
        a.lon
            .partial_cmp(&b.lon)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.lat
                    .partial_cmp(&b.lat)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    cells
}

/// One vertex of a storm track: position, time, and peak intensity at that scan.
#[derive(Debug, Clone)]
pub struct TrackPoint {
    pub lon: f64,
    pub lat: f64,
    pub t_ms: i64,
    pub max_dbz: f32,
}

/// A storm cell's path across scans.
#[derive(Debug, Clone)]
pub struct Track {
    pub points: Vec<TrackPoint>,
}

/// Link per-scan cells into tracks (SCIT-style centroid tracking).
///
/// `frames` is `(scan_time_ms, cells)` per scan. For each scan, current cells
/// (heaviest first, for stability) are greedily matched one-to-one to the
/// nearest previous-scan track endpoint within `max_speed_m_s · Δt`; unmatched
/// cells seed new tracks. Tracks shorter than 2 points are dropped.
pub fn track_cells(mut frames: Vec<(i64, Vec<Cell>)>, max_speed_m_s: f64) -> Vec<Track> {
    frames.sort_by_key(|(t, _)| *t);

    let mut tracks: Vec<Track> = Vec::new();
    // Track indices whose last point is from the previous frame.
    let mut active: Vec<usize> = Vec::new();
    let mut prev_t: Option<i64> = None;

    for (t, cells) in frames {
        let dt_s = prev_t
            .map(|p| ((t - p).max(0)) as f64 / 1000.0)
            .unwrap_or(0.0);
        let max_dist = max_speed_m_s * dt_s;

        let mut order: Vec<usize> = (0..cells.len()).collect();
        order.sort_by(|&a, &b| {
            cells[b]
                .max_dbz
                .partial_cmp(&cells[a].max_dbz)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut used = vec![false; tracks.len()];
        let mut next_active: Vec<usize> = Vec::with_capacity(cells.len());

        for &ci in &order {
            let cell = &cells[ci];
            let mut best: Option<(usize, f64)> = None;
            if dt_s > 0.0 {
                for &ti in &active {
                    if used[ti] {
                        continue;
                    }
                    let last = &tracks[ti].points[tracks[ti].points.len() - 1];
                    let d =
                        crate::common::haversine_distance(last.lat, last.lon, cell.lat, cell.lon);
                    if d <= max_dist && best.map(|(_, bd)| d < bd).unwrap_or(true) {
                        best = Some((ti, d));
                    }
                }
            }
            let point = TrackPoint {
                lon: cell.lon,
                lat: cell.lat,
                t_ms: t,
                max_dbz: cell.max_dbz,
            };
            match best {
                Some((ti, _)) => {
                    used[ti] = true;
                    tracks[ti].points.push(point);
                    next_active.push(ti);
                }
                None => {
                    tracks.push(Track {
                        points: vec![point],
                    });
                    next_active.push(tracks.len() - 1);
                }
            }
        }

        active = next_active;
        prev_t = Some(t);
    }

    tracks.into_iter().filter(|t| t.points.len() >= 2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a grid and stamp explicit cell values for testing (row-major,
    /// `iy = 0` at min_lat).
    fn grid_with(nx: usize, ny: usize, fill: &[(usize, usize, f32)]) -> MosaicGrid {
        // 1 km cells near 42°N over a small bbox.
        let mut g = MosaicGrid::new(
            42.0,
            -93.0,
            42.0 + ny as f64 * 0.01,
            -93.0 + nx as f64 * 0.02,
            1.0,
        );
        // Force exact dimensions for deterministic tests.
        g.nx = nx;
        g.ny = ny;
        g.cells = vec![f32::NEG_INFINITY; nx * ny];
        for &(ix, iy, v) in fill {
            g.cells[iy * nx + ix] = v;
        }
        g
    }

    #[test]
    fn detect_single_cell() {
        // A 2×2 block of 55 dBZ in a 6×6 grid → one cell.
        let g = grid_with(
            6,
            6,
            &[(2, 2, 55.0), (3, 2, 56.0), (2, 3, 54.0), (3, 3, 58.0)],
        );
        let cells = detect_cells(&g, 50.0, 0.0);
        assert_eq!(cells.len(), 1);
        assert!((cells[0].max_dbz - 58.0).abs() < 1e-3);
        // Area ≈ 4 cells.
        assert!(cells[0].area_km2 > 0.0);
        // Centroid lands inside the block bbox.
        let c = &cells[0];
        assert!(c.lon > g.min_lon && c.lat > g.min_lat);
    }

    #[test]
    fn min_area_filter_drops_speckle() {
        let g = grid_with(6, 6, &[(1, 1, 60.0)]); // single 1-cell speck
        let area = g.cell_area_km2();
        // Threshold just above one cell's area removes it.
        let cells = detect_cells(&g, 50.0, area * 1.5);
        assert!(cells.is_empty());
        // A lower threshold keeps it.
        assert_eq!(detect_cells(&g, 50.0, 0.0).len(), 1);
    }

    #[test]
    fn two_separate_cells() {
        let g = grid_with(8, 8, &[(1, 1, 55.0), (6, 6, 57.0)]);
        assert_eq!(detect_cells(&g, 50.0, 0.0).len(), 2);
    }

    #[test]
    fn contour_produces_bands() {
        // A 5×5 high core surrounded by zeros → at least one filled band.
        let mut fill = Vec::new();
        for iy in 3..6 {
            for ix in 3..6 {
                fill.push((ix, iy, 45.0));
            }
        }
        let g = grid_with(10, 10, &fill);
        let bands = contour_bands(&g, &[20.0, 30.0, 40.0, 50.0], -30.0, 0.0);
        assert!(!bands.is_empty());
        // Every band spans two thresholds and has a closed ring.
        for b in &bands {
            assert!(b.dbz_lo >= 20);
            assert!(b.dbz_hi > b.dbz_lo);
            assert!(b.rings[0].len() >= 4);
        }
    }

    #[test]
    fn track_links_moving_cell() {
        let frames = vec![
            (
                0i64,
                vec![Cell {
                    lon: -93.0,
                    lat: 42.0,
                    max_dbz: 55.0,
                    area_km2: 20.0,
                }],
            ),
            // ~1.5 km east after 300 s → ~5 m/s, within a 40 m/s gate.
            (
                300_000i64,
                vec![Cell {
                    lon: -92.982,
                    lat: 42.0,
                    max_dbz: 57.0,
                    area_km2: 22.0,
                }],
            ),
        ];
        let tracks = track_cells(frames, 40.0);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].points.len(), 2);
    }

    #[test]
    fn track_splits_when_too_fast() {
        let frames = vec![
            (
                0i64,
                vec![Cell {
                    lon: -93.0,
                    lat: 42.0,
                    max_dbz: 55.0,
                    area_km2: 20.0,
                }],
            ),
            // ~80 km east after 60 s → way past any plausible storm speed.
            (
                60_000i64,
                vec![Cell {
                    lon: -92.0,
                    lat: 42.0,
                    max_dbz: 57.0,
                    area_km2: 22.0,
                }],
            ),
        ];
        let tracks = track_cells(frames, 40.0);
        // Both cells are too-short singletons → dropped, leaving no 2+ track.
        assert!(tracks.is_empty());
    }
}
