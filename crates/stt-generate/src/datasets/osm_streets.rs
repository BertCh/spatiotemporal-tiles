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

/// Road classes, coarsened into LOD tiers. `min_zoom()` is the shallowest zoom
/// the class appears at in the flows pyramid (z8–14) — the LOD lever.
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
    /// unclassified, residential, living_street, road
    Residential,
    /// service
    Service,
}

impl RoadClass {
    /// Map an OSM `highway=*` tag value to a class tier, or `None` to exclude
    /// (footways/cycleways/paths/non-roads).
    pub fn from_highway(v: &str) -> Option<RoadClass> {
        Some(match v {
            "motorway" | "motorway_link" | "trunk" | "trunk_link" => RoadClass::Motorway,
            "primary" | "primary_link" => RoadClass::Primary,
            "secondary" | "secondary_link" => RoadClass::Secondary,
            "tertiary" | "tertiary_link" => RoadClass::Tertiary,
            "unclassified" | "residential" | "living_street" | "road" => RoadClass::Residential,
            "service" => RoadClass::Service,
            // Excluded: footway, cycleway, path, steps, pedestrian, track,
            // bridleway, corridor, construction, proposed, raceway, …
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
/// Longitude scale at NYC latitude (~40.7°): cos(40.7°) ≈ 0.758.
const LON_SCALE: f64 = 0.758;

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
    /// Parse a `.osm.pbf` road network. Two passes: ways (collect highway refs +
    /// class + wanted node-id set), then nodes (resolve coords for wanted ids).
    pub fn from_pbf(path: &Path) -> Result<OsmNetwork> {
        // ---- Pass A: ways ----
        let mut ways: HashMap<i64, WayRec> = HashMap::new();
        let mut wanted: HashSet<i64> = HashSet::new();
        let reader = ElementReader::from_path(path)
            .with_context(|| format!("opening OSM pbf {}", path.display()))?;
        reader.for_each(|element| {
            if let Element::Way(w) = element {
                let mut class: Option<RoadClass> = None;
                for (k, v) in w.tags() {
                    if k == "highway" {
                        class = RoadClass::from_highway(v);
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
        reader.for_each(|element| match element {
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
            coord_to_node.entry(quant(lon, lat, MATCH_QUANT_DEG)).or_insert(id);
        }

        // edge_grid: edge midpoint → cell (for the nearest-edge fallback).
        let mut edge_grid: HashMap<(i32, i32), Vec<EdgeKey>> = HashMap::new();
        for (&(a, b), _) in &edge_way {
            let (ax, ay) = node_coords[&a];
            let (bx, by) = node_coords[&b];
            let cell = quant((ax + bx) * 0.5, (ay + by) * 0.5, EDGE_GRID_DEG);
            edge_grid.entry(cell).or_default().push((a, b));
        }

        OsmNetwork {
            node_coords,
            edge_way,
            ways,
            coord_to_node,
            edge_grid,
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
        let na = self.coord_to_node.get(&quant(p0[0], p0[1], MATCH_QUANT_DEG));
        let nb = self.coord_to_node.get(&quant(p1[0], p1[1], MATCH_QUANT_DEG));
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
        let na = self.coord_to_node.get(&quant(p0[0], p0[1], MATCH_QUANT_DEG));
        let nb = self.coord_to_node.get(&quant(p1[0], p1[1], MATCH_QUANT_DEG));
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
                        let d2 = point_seg_dist2_scaled(mx, my, ax, ay, bx, by);
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
}

/// Squared distance from point to segment, with longitude scaled by cos(lat)
/// so the metric is ~isotropic in metres (good enough at city scale).
fn point_seg_dist2_scaled(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let s = LON_SCALE;
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
            WayRec { refs: vec![1, 2, 3], class: RoadClass::Primary },
        );
        ways.insert(
            20,
            WayRec { refs: vec![2, 4], class: RoadClass::Residential },
        );
        OsmNetwork::build_indices(node_coords, ways)
    }

    #[test]
    fn exact_match_resolves_edges() {
        let n = net();
        assert_eq!(n.match_segment([-74.000, 40.000], [-73.999, 40.000]), Some((1, 2)));
        assert_eq!(n.match_segment([-73.999, 40.000], [-73.998, 40.000]), Some((2, 3)));
        assert_eq!(n.match_segment([-73.999, 40.000], [-73.999, 40.001]), Some((2, 4)));
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
        ways.insert(20, WayRec { refs: vec![2, 3], class: RoadClass::Residential });
        ways.insert(10, WayRec { refs: vec![2, 3], class: RoadClass::Primary });
        let n = OsmNetwork::build_indices(node_coords, ways);
        assert_eq!(n.edge_way_id((2, 3)), Some(10)); // primary (z9) beats residential (z12)
    }

    #[test]
    fn class_min_zoom_table() {
        assert_eq!(RoadClass::from_highway("motorway").unwrap().min_zoom(), 8);
        assert_eq!(RoadClass::from_highway("primary_link").unwrap().min_zoom(), 9);
        assert_eq!(RoadClass::from_highway("residential").unwrap().min_zoom(), 12);
        assert_eq!(RoadClass::from_highway("service").unwrap().min_zoom(), 13);
        assert!(RoadClass::from_highway("footway").is_none());
        assert!(RoadClass::from_highway("cycleway").is_none());
    }
}
