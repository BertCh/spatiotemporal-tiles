//! OSM road-network ingestion for flow-corridor aggregation.
//!
//! Parses a standard `.osm.pbf` extract (e.g. `new-york-latest.osm.pbf`, the same
//! file OSRM was built from) into the road graph that `nyc_rideshare_flows`
//! aggregates taxi traffic onto. Because OSRM routes the trips on THIS network,
//! every routed-trip vertex lies on an OSM way — so we can attach each trip
//! segment to its OSM edge and emit corridors with EXACT street geometry
//! (intersection-to-intersection node sequences), plus a per-feature `min_zoom`
//! from the road class for vector-tile-style LOD (major roads when zoomed out,
//! all streets up close).
//!
//! Reuses the `osmpbf` `ElementReader` pattern from [`super::osm_edits`].

use anyhow::{Context, Result};
use osmpbf::{Element, ElementReader};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Undirected OSM edge: a node-id pair in ascending order.
pub type EdgeKey = (i64, i64);

/// Which highway tags to admit into the network, i.e. which mode's routing this
/// graph must mirror. OSRM routes on a specific profile; the network we match
/// its routed geometry against must include the same way classes or the routed
/// segments won't find an edge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NetworkMode {
    /// Car/road network (NYC taxi `--flows`): drivable roads only, no cycle
    /// infrastructure. Excludes footway/cycleway/path/pedestrian/track.
    Roads,
    /// Bicycle network (Montréal BIXI `--streets`): drivable roads *plus*
    /// cycleways and paths, but **excludes** motorways/trunks (bikes don't ride
    /// there). Mirrors OSRM's `bicycle` profile so routed bike geometry matches.
    Bike,
}

/// Road classes, coarsened into LOD tiers. `min_zoom()` is the shallowest zoom
/// the class appears at in the flows pyramid (z8–13) — the LOD lever.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RoadClass {
    /// motorway(_link), trunk(_link)
    Motorway,
    /// primary(_link)
    Primary,
    /// secondary(_link)
    Secondary,
    /// tertiary(_link)
    Tertiary,
    /// Dedicated bike infrastructure (cycleway, path) — only admitted in
    /// [`NetworkMode::Bike`]. A mid-zoom tier so major bike arteries (the REV,
    /// de Maisonneuve, the canal path) surface in the city overview.
    Cycleway,
    /// unclassified, residential, living_street, road
    Residential,
    /// service (+ footway/pedestrian/track in bike mode)
    Service,
}

impl RoadClass {
    /// Map an OSM `highway=*` tag value to a class tier for the given mode, or
    /// `None` to exclude. In [`NetworkMode::Bike`] motorways/trunks are dropped
    /// and cycleways/paths/footways are admitted; everything else mirrors the
    /// road mapping.
    pub fn from_highway_with_mode(v: &str, mode: NetworkMode) -> Option<RoadClass> {
        Some(match v {
            "motorway" | "motorway_link" | "trunk" | "trunk_link" => match mode {
                // Bikes can't ride limited-access roads — exclude so a stray
                // routed vertex can't false-match onto a motorway edge.
                NetworkMode::Bike => return None,
                NetworkMode::Roads => RoadClass::Motorway,
            },
            "primary" | "primary_link" => RoadClass::Primary,
            "secondary" | "secondary_link" => RoadClass::Secondary,
            "tertiary" | "tertiary_link" => RoadClass::Tertiary,
            "cycleway" | "path" => match mode {
                NetworkMode::Bike => RoadClass::Cycleway,
                // Roads mode excludes cycle infrastructure (unchanged).
                NetworkMode::Roads => return None,
            },
            "unclassified" | "residential" | "living_street" | "road" => RoadClass::Residential,
            "service" => RoadClass::Service,
            "footway" | "pedestrian" | "track" => match mode {
                // Minor shared/foot ways a bike route can clip — admit at the
                // deepest tier so they don't clutter the overview.
                NetworkMode::Bike => RoadClass::Service,
                NetworkMode::Roads => return None,
            },
            // Excluded: steps, bridleway, corridor, construction, proposed,
            // raceway, … (and, in road mode, the cycle/foot ways above).
            _ => return None,
        })
    }

    /// Shallowest zoom this class appears at (the per-feature `min_zoom`).
    pub fn min_zoom(self) -> u8 {
        match self {
            RoadClass::Motorway => 8,
            RoadClass::Primary => 9,
            RoadClass::Secondary => 10,
            RoadClass::Tertiary => 11,
            RoadClass::Cycleway => 11,
            RoadClass::Residential => 12,
            RoadClass::Service => 13,
        }
    }
}

/// One OSM way kept as a road: its ordered node ids and class. The node
/// sequence IS the corridor geometry.
pub struct WayRec {
    pub refs: Vec<i64>,
    pub class: RoadClass,
}

/// Coordinate-snap grid for matching trip vertices to OSM nodes (~1 m at NYC).
/// Coarser than OSRM's ~1e-6° coordinate rounding so a routed vertex and its
/// OSM node land in the same cell.
const MATCH_QUANT_DEG: f64 = 1.0e-5;
/// Edge-midpoint spatial-index cell (~5.5 m lat); the nearest-edge fallback
/// scans this cell + 8 neighbours.
const EDGE_GRID_DEG: f64 = 5.0e-5;
/// Nearest-edge fallback acceptance threshold (~25 m), in scaled-degree² (lon
/// scaled by cos(lat) so the metric is roughly isotropic). 25 m ≈ 2.25e-4°.
const SNAP_THRESH_SCALED_DEG: f64 = 2.25e-4;

fn quant(lon: f64, lat: f64, grid: f64) -> (i32, i32) {
    ((lon / grid).round() as i32, (lat / grid).round() as i32)
}

/// The road graph + spatial indices used by the flow aggregator.
pub struct OsmNetwork {
    /// node id → (lon, lat). Exact OSM geometry source.
    node_coords: HashMap<i64, (f64, f64)>,
    /// edge → owning way id (more-major class wins on a shared edge).
    edge_way: HashMap<EdgeKey, i64>,
    /// way id → (node refs, class).
    pub ways: HashMap<i64, WayRec>,
    /// quantized OSM node coord → node id (fast exact match).
    coord_to_node: HashMap<(i32, i32), i64>,
    /// ~5 m cell → edges whose midpoint falls in it (nearest-edge fallback).
    edge_grid: HashMap<(i32, i32), Vec<EdgeKey>>,
    /// Longitude scale `cos(mean_lat)` for the fallback distance metric, derived
    /// from this network's own latitude so matching is isotropic anywhere
    /// (≈0.76 at NYC, ≈0.70 at Montréal) — no hardcoded region assumption.
    lon_scale: f64,
}

/// Undirected edge key with endpoints in ascending order.
fn edge_key(a: i64, b: i64) -> EdgeKey {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

impl OsmNetwork {
    /// Parse a `.osm.pbf` **road** network ([`NetworkMode::Roads`]) — the
    /// back-compat default used by NYC taxi `--flows`.
    pub fn from_pbf(path: &Path) -> Result<OsmNetwork> {
        Self::from_pbf_with_mode(path, NetworkMode::Roads)
    }

    /// Parse a `.osm.pbf` network admitting the highway classes for `mode`. Two
    /// passes: ways (collect highway refs + class + wanted node-id set), then
    /// nodes (resolve coords for wanted ids).
    pub fn from_pbf_with_mode(path: &Path, mode: NetworkMode) -> Result<OsmNetwork> {
        // ---- Pass A: ways ----
        let mut ways: HashMap<i64, WayRec> = HashMap::new();
        let mut wanted: HashSet<i64> = HashSet::new();
        let reader = ElementReader::from_path(path)
            .with_context(|| format!("opening OSM pbf {}", path.display()))?;
        reader
            .for_each(|element| {
                if let Element::Way(w) = element {
                    let mut class: Option<RoadClass> = None;
                    for (k, v) in w.tags() {
                        if k == "highway" {
                            class = RoadClass::from_highway_with_mode(v, mode);
                            break;
                        }
                    }
                    let Some(class) = class else { return };
                    let refs: Vec<i64> = w.refs().collect();
                    if refs.len() < 2 {
                        return;
                    }
                    for &r in &refs {
                        wanted.insert(r);
                    }
                    ways.insert(w.id(), WayRec { refs, class });
                }
            })
            .with_context(|| "OSM pbf ways pass")?;

        // ---- Pass B: nodes ----
        let mut node_coords: HashMap<i64, (f64, f64)> = HashMap::with_capacity(wanted.len());
        let reader = ElementReader::from_path(path)
            .with_context(|| format!("re-opening OSM pbf {}", path.display()))?;
        reader
            .for_each(|element| match element {
                Element::DenseNode(n) => {
                    if wanted.contains(&n.id()) {
                        node_coords.insert(n.id(), (n.lon(), n.lat()));
                    }
                }
                Element::Node(n) => {
                    if wanted.contains(&n.id()) {
                        node_coords.insert(n.id(), (n.lon(), n.lat()));
                    }
                }
                _ => {}
            })
            .with_context(|| "OSM pbf nodes pass")?;

        Ok(Self::build_indices(node_coords, ways))
    }

    /// Build the edge/way/coord/grid indices from resolved nodes + ways.
    /// Deterministic: ways are processed in sorted id order so shared-edge
    /// ownership and coord-cell collisions resolve identically every run.
    /// `pub(crate)` so the flow aggregator's tests can build a synthetic network.
    pub(crate) fn build_indices(
        node_coords: HashMap<i64, (f64, f64)>,
        ways: HashMap<i64, WayRec>,
    ) -> OsmNetwork {
        let mut edge_way: HashMap<EdgeKey, i64> = HashMap::new();
        let mut way_ids: Vec<i64> = ways.keys().copied().collect();
        way_ids.sort_unstable();
        for wid in &way_ids {
            let w = &ways[wid];
            let mz = w.class.min_zoom();
            for pair in w.refs.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                if a == b || !node_coords.contains_key(&a) || !node_coords.contains_key(&b) {
                    continue;
                }
                let key = edge_key(a, b);
                match edge_way.get(&key) {
                    // Keep the more-major (lower min_zoom) owner; tie → lower id.
                    Some(&cur) => {
                        let cur_mz = ways[&cur].class.min_zoom();
                        if mz < cur_mz || (mz == cur_mz && *wid < cur) {
                            edge_way.insert(key, *wid);
                        }
                    }
                    None => {
                        edge_way.insert(key, *wid);
                    }
                }
            }
        }

        // coord_to_node: deterministic on cell collision (lower node id wins).
        let mut coord_to_node: HashMap<(i32, i32), i64> = HashMap::new();
        let mut node_ids: Vec<i64> = node_coords.keys().copied().collect();
        node_ids.sort_unstable();
        for id in node_ids {
            let (lon, lat) = node_coords[&id];
            coord_to_node
                .entry(quant(lon, lat, MATCH_QUANT_DEG))
                .or_insert(id);
        }

        // edge_grid: edge midpoint → cell (for the nearest-edge fallback).
        let mut edge_grid: HashMap<(i32, i32), Vec<EdgeKey>> = HashMap::new();
        for (&(a, b), _) in &edge_way {
            let (ax, ay) = node_coords[&a];
            let (bx, by) = node_coords[&b];
            let cell = quant((ax + bx) * 0.5, (ay + by) * 0.5, EDGE_GRID_DEG);
            edge_grid.entry(cell).or_default().push((a, b));
        }

        // Longitude scale from this network's mean latitude, so the fallback
        // metric is isotropic regardless of region (NYC ~40.7°, Montréal ~45.5°).
        let lon_scale = if node_coords.is_empty() {
            1.0
        } else {
            let mean_lat =
                node_coords.values().map(|&(_, lat)| lat).sum::<f64>() / node_coords.len() as f64;
            mean_lat.to_radians().cos()
        };

        OsmNetwork {
            node_coords,
            edge_way,
            ways,
            coord_to_node,
            edge_grid,
            lon_scale,
        }
    }

    pub fn node_xy(&self, id: i64) -> Option<(f64, f64)> {
        self.node_coords.get(&id).copied()
    }

    /// The owning way id of an edge (if it is a road edge).
    pub fn edge_way_id(&self, e: EdgeKey) -> Option<i64> {
        self.edge_way.get(&e).copied()
    }

    /// The way record (ordered refs + class) for an id.
    pub fn way(&self, id: i64) -> Option<&WayRec> {
        self.ways.get(&id)
    }

    /// Whether a segment's two endpoints resolve to OSM nodes that form a
    /// direct edge — i.e. [`Self::match_segment`] took the exact (not fallback)
    /// path. Used only for match-quality reporting.
    pub fn is_exact_pair(&self, p0: [f64; 2], p1: [f64; 2]) -> bool {
        let na = self
            .coord_to_node
            .get(&quant(p0[0], p0[1], MATCH_QUANT_DEG));
        let nb = self
            .coord_to_node
            .get(&quant(p1[0], p1[1], MATCH_QUANT_DEG));
        match (na, nb) {
            (Some(&a), Some(&b)) => a != b && self.edge_way.contains_key(&edge_key(a, b)),
            _ => false,
        }
    }

    /// Match a trip segment (two routed-trip vertices) to its OSM edge.
    /// (a) exact: both endpoints snap to OSM nodes that form an edge.
    /// (b) fallback: snap the segment midpoint to the nearest road edge.
    /// Returns `None` when nothing is within threshold (a true miss).
    pub fn match_segment(&self, p0: [f64; 2], p1: [f64; 2]) -> Option<EdgeKey> {
        // (a) exact endpoint match.
        let na = self
            .coord_to_node
            .get(&quant(p0[0], p0[1], MATCH_QUANT_DEG));
        let nb = self
            .coord_to_node
            .get(&quant(p1[0], p1[1], MATCH_QUANT_DEG));
        if let (Some(&a), Some(&b)) = (na, nb) {
            if a != b {
                let key = edge_key(a, b);
                if self.edge_way.contains_key(&key) {
                    return Some(key);
                }
            }
        }
        // (b) nearest-edge fallback on the segment midpoint.
        let mx = (p0[0] + p1[0]) * 0.5;
        let my = (p0[1] + p1[1]) * 0.5;
        let (cx, cy) = quant(mx, my, EDGE_GRID_DEG);
        let mut best: Option<(f64, EdgeKey)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(cands) = self.edge_grid.get(&(cx + dx, cy + dy)) {
                    for &e in cands {
                        let (ax, ay) = self.node_coords[&e.0];
                        let (bx, by) = self.node_coords[&e.1];
                        let d2 = point_seg_dist2_scaled(mx, my, ax, ay, bx, by, self.lon_scale);
                        if best.map_or(true, |(bd, _)| d2 < bd) {
                            best = Some((d2, e));
                        }
                    }
                }
            }
        }
        best.filter(|(d2, _)| *d2 <= SNAP_THRESH_SCALED_DEG * SNAP_THRESH_SCALED_DEG)
            .map(|(_, e)| e)
    }

    /// Like [`Self::match_segment`] but returns the matched edge **oriented in
    /// travel direction**: `from` is the endpoint nearer the segment start `p0`,
    /// `to` the endpoint nearer `p1`. Directed flow aggregation keys on this so
    /// A→B and B→A traffic stay separate (one-way pairs, asymmetric ridership);
    /// a two-way street resolves to the same node pair in opposite order for the
    /// two directions, which the offset renderer then draws as side-by-side
    /// ribbons. Returns `None` on the same true miss as `match_segment`.
    pub fn match_segment_directed(&self, p0: [f64; 2], p1: [f64; 2]) -> Option<(i64, i64)> {
        let (a, b) = self.match_segment(p0, p1)?;
        let s = self.lon_scale;
        let d2_from_p0 = |id: i64| {
            let (nx, ny) = self.node_coords[&id];
            let dx = (p0[0] - nx) * s;
            let dy = p0[1] - ny;
            dx * dx + dy * dy
        };
        Some(if d2_from_p0(a) <= d2_from_p0(b) {
            (a, b)
        } else {
            (b, a)
        })
    }

    /// Longitude scale `cos(mean_lat)` for this network — exposed so stroke
    /// synthesis can compute geometrically correct deflection angles (longitude
    /// degrees are shorter than latitude degrees away from the equator).
    pub fn lon_scale(&self) -> f64 {
        self.lon_scale
    }
}

/// Squared distance from point to segment, with longitude scaled by `lon_scale`
/// (= cos(mean_lat)) so the metric is ~isotropic in metres (good enough at city
/// scale) regardless of the network's latitude.
fn point_seg_dist2_scaled(
    px: f64,
    py: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    lon_scale: f64,
) -> f64 {
    let s = lon_scale;
    let (px, ax, bx) = (px * s, ax * s, bx * s);
    let (abx, aby) = (bx - ax, by - ay);
    let (apx, apy) = (px - ax, py - ay);
    let len2 = abx * abx + aby * aby;
    let t = if len2 > 0.0 {
        ((apx * abx + apy * aby) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (cx, cy) = (ax + t * abx, ay + t * aby);
    let (dx, dy) = (px - cx, py - cy);
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net() -> OsmNetwork {
        // Two ways sharing node 2: way 10 (primary) nodes 1-2-3 along lat 40.0,
        // way 20 (residential) nodes 2-4 going north.
        let mut node_coords = HashMap::new();
        node_coords.insert(1, (-74.000, 40.000));
        node_coords.insert(2, (-73.999, 40.000));
        node_coords.insert(3, (-73.998, 40.000));
        node_coords.insert(4, (-73.999, 40.001));
        let mut ways = HashMap::new();
        ways.insert(
            10,
            WayRec {
                refs: vec![1, 2, 3],
                class: RoadClass::Primary,
            },
        );
        ways.insert(
            20,
            WayRec {
                refs: vec![2, 4],
                class: RoadClass::Residential,
            },
        );
        OsmNetwork::build_indices(node_coords, ways)
    }

    #[test]
    fn exact_match_resolves_edges() {
        let n = net();
        assert_eq!(
            n.match_segment([-74.000, 40.000], [-73.999, 40.000]),
            Some((1, 2))
        );
        assert_eq!(
            n.match_segment([-73.999, 40.000], [-73.998, 40.000]),
            Some((2, 3))
        );
        assert_eq!(
            n.match_segment([-73.999, 40.000], [-73.999, 40.001]),
            Some((2, 4))
        );
    }

    #[test]
    fn fallback_snaps_near_misses() {
        let n = net();
        // A point ~5 m off the 1-2 edge still snaps to it.
        let e = n.match_segment([-74.0000, 40.00004], [-73.9990, 40.00004]);
        assert_eq!(e, Some((1, 2)));
    }

    #[test]
    fn far_segments_miss() {
        let n = net();
        // ~1 km away → no edge within threshold.
        assert_eq!(n.match_segment([-73.980, 40.020], [-73.979, 40.020]), None);
    }

    #[test]
    fn shared_edge_keeps_more_major_owner() {
        // Edge (2,?) ownership: give way 20 a shared edge with a major way and
        // confirm the major one wins. Here build a fresh net where node 2-3 is
        // in both a residential (id 20) and primary (id 10) way.
        let mut node_coords = HashMap::new();
        node_coords.insert(2, (-73.999, 40.000));
        node_coords.insert(3, (-73.998, 40.000));
        let mut ways = HashMap::new();
        ways.insert(
            20,
            WayRec {
                refs: vec![2, 3],
                class: RoadClass::Residential,
            },
        );
        ways.insert(
            10,
            WayRec {
                refs: vec![2, 3],
                class: RoadClass::Primary,
            },
        );
        let n = OsmNetwork::build_indices(node_coords, ways);
        assert_eq!(n.edge_way_id((2, 3)), Some(10)); // primary (z9) beats residential (z12)
    }

    #[test]
    fn class_min_zoom_table() {
        use NetworkMode::Roads;
        assert_eq!(
            RoadClass::from_highway_with_mode("motorway", Roads)
                .unwrap()
                .min_zoom(),
            8
        );
        assert_eq!(
            RoadClass::from_highway_with_mode("primary_link", Roads)
                .unwrap()
                .min_zoom(),
            9
        );
        assert_eq!(
            RoadClass::from_highway_with_mode("residential", Roads)
                .unwrap()
                .min_zoom(),
            12
        );
        assert_eq!(
            RoadClass::from_highway_with_mode("service", Roads)
                .unwrap()
                .min_zoom(),
            13
        );
        assert!(RoadClass::from_highway_with_mode("footway", Roads).is_none());
        assert!(RoadClass::from_highway_with_mode("cycleway", Roads).is_none());
    }

    #[test]
    fn bike_mode_admits_cycleways_and_drops_motorways() {
        use NetworkMode::Bike;
        // Cycle infrastructure is admitted (and surfaces at a mid zoom).
        assert_eq!(
            RoadClass::from_highway_with_mode("cycleway", Bike),
            Some(RoadClass::Cycleway)
        );
        assert_eq!(
            RoadClass::from_highway_with_mode("path", Bike)
                .unwrap()
                .min_zoom(),
            11
        );
        // Minor foot/shared ways land at the deepest tier.
        assert_eq!(
            RoadClass::from_highway_with_mode("footway", Bike),
            Some(RoadClass::Service)
        );
        // Bikes don't ride limited-access roads.
        assert!(RoadClass::from_highway_with_mode("motorway", Bike).is_none());
        assert!(RoadClass::from_highway_with_mode("trunk_link", Bike).is_none());
        // Shared roads still classify as in road mode.
        assert_eq!(
            RoadClass::from_highway_with_mode("residential", Bike),
            Some(RoadClass::Residential)
        );
    }

    #[test]
    fn lon_scale_tracks_network_latitude() {
        // A Montréal-latitude network gets ~cos(45.5°) ≈ 0.70, not NYC's 0.758.
        let mut node_coords = HashMap::new();
        node_coords.insert(1, (-73.585, 45.500));
        node_coords.insert(2, (-73.580, 45.500));
        let mut ways = HashMap::new();
        ways.insert(
            10,
            WayRec {
                refs: vec![1, 2],
                class: RoadClass::Cycleway,
            },
        );
        let n = OsmNetwork::build_indices(node_coords, ways);
        assert!((n.lon_scale - 45.5_f64.to_radians().cos()).abs() < 1e-9);
        // Exact match still resolves at this latitude.
        assert_eq!(
            n.match_segment([-73.585, 45.500], [-73.580, 45.500]),
            Some((1, 2))
        );
    }
}
