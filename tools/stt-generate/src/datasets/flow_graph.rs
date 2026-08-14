//! Abstract **Edge-Path-Bundling** flow network for `bixi --flow-graph`.
//!
//! Builds a coherent, Sankey-like flow network from OD pairs WITHOUT a street
//! network. Instead of snapping to roads (`--merged-paths`), it:
//!   1. clusters origin/destination stations into **hubs** (a boldness knob),
//!   2. connects the hubs with a **Delaunay proximity graph**,
//!   3. routes each OD flow along the graph's **shortest path** with edge cost
//!      `length^k` (`k>1`, so flows detour onto shared trunks — Edge-Path
//!      Bundling, Wallinger et al. 2022), accumulating DIRECTED flow per graph
//!      edge, and
//!   4. hands the directed edge graph to the SAME stroke synthesis +
//!      `FlowStrokeLayer` renderer as the street mode.
//!
//! The result is a Sankey-like network: little OD lines merge onto shared trunk
//! lines (which swell where flows join and taper where they leave) and branch
//! back out — the "one big line that little lines enter and leave" aesthetic,
//! coherent rather than the abstract smear of force-directed bundling. Width =
//! travellers; the hourly matrix animates via the same breathing renderer.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use anyhow::Result;
use delaunator::{triangulate, Point};
use serde_json::{json, Map};

use crate::common::{LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter};
use crate::datasets::nyc_rideshare_flows::{
    stroke_min_zoom, synthesize_strokes, vertex_values_from_segment_counts, StrokeParams,
    MAX_CHAIN_SEGMENTS,
};

/// Metres per degree of latitude (spherical mean) — the abstract network only
/// needs a locally isotropic metric, so a constant is fine.
const M_PER_DEG: f64 = 111_320.0;

/// Tuning for the abstract flow-network build.
#[derive(Clone, Debug)]
pub struct FlowGraphParams {
    /// Hub clustering radius in METRES — stations within this distance merge into
    /// one hub before graph construction. Larger = fewer, bolder trunks. 0 = one
    /// hub per station (finest mesh).
    pub hub_radius_m: f64,
    /// Path-cost exponent `k` (> 1). Higher = flows detour more aggressively onto
    /// shared trunks (more bundling). ~2 is moderate.
    pub bundle_k: f64,
    /// Drop Delaunay edges longer than this many metres (removes spurious
    /// convex-hull connectors that would let flows shortcut across the map).
    pub max_edge_m: f64,
    /// Catmull-Rom spline sub-samples per hub segment — turns the angular hub
    /// polyline into a flowing curve (the force-directed look). 1 = off; ~6 is
    /// smooth. The per-vertex matrix is resampled in lockstep.
    pub spline_samples: usize,
    /// Laplacian smoothing passes applied AFTER splining (removes spline overshoot;
    /// endpoints pinned, vertex count preserved). ~2.
    pub smooth_iters: usize,
    /// Perpendicular offset (metres) baked into each directed trunk, to the RIGHT
    /// of travel — so a two-way corridor's A→B and B→A trunks (which traverse the
    /// shared geometry in opposite order) separate into side-by-side twin ribbons.
    /// Baked into geometry (not a render-time shift) so it needs no GPU extension.
    /// ~10 m separates ribbons at neighbourhood zoom and merges at overview.
    pub ribbon_offset_m: f64,
    /// Stroke-synthesis tuning (junction angle / flow-split / min-trips).
    pub stroke: StrokeParams,
}

impl Default for FlowGraphParams {
    fn default() -> Self {
        Self {
            hub_radius_m: 250.0,
            bundle_k: 2.0,
            max_edge_m: 2500.0,
            spline_samples: 6,
            smooth_iters: 2,
            ribbon_offset_m: 10.0,
            stroke: StrokeParams::default(),
        }
    }
}

/// A routed OD pair: origin/destination station indices + a `bin → count` map
/// on the same bin axis as the generator (already folded to hour-of-day for
/// `--typical-day`).
pub struct OdPair {
    pub origin: usize,
    pub dest: usize,
    pub bins: HashMap<i64, u32>,
}

/// Build the abstract flow network and write it as merged-corridor LineString
/// GeoParquet (same schema the street strokes use). Returns
/// `(features, num_buckets, bucket0_ms, range_end_ms)`.
pub fn build_and_write(
    output: &Path,
    station_coords: &[[f64; 2]],
    od: &[OdPair],
    bin_ms: i64,
    params: &FlowGraphParams,
) -> Result<(usize, usize, i64, i64)> {
    // Local isotropic metric: scale longitude by cos(mean lat).
    let lon_scale = if station_coords.is_empty() {
        1.0
    } else {
        let mean_lat =
            station_coords.iter().map(|s| s[1]).sum::<f64>() / station_coords.len() as f64;
        mean_lat.to_radians().cos().max(0.1)
    };

    // 1. Cluster stations into hubs.
    let (hubs, station_hub) = cluster_hubs(station_coords, params.hub_radius_m, lon_scale);
    println!(
        "   🔗 {} stations → {} hubs (radius {} m)",
        station_coords.len(),
        hubs.len(),
        params.hub_radius_m
    );

    // 2. Delaunay proximity graph over the hubs.
    let (adj, edge_len) = delaunay_graph(&hubs, lon_scale, params.max_edge_m);
    println!("   △ Delaunay graph: {} hub edges", edge_len.len());

    // 3. Edge-Path-Bundling: route each OD pair on the graph (cost = len^k),
    //    accumulate DIRECTED flow per graph edge. Paths are memoized per hub pair
    //    (many OD pairs share a hub pair after clustering).
    let mut dir_counts: HashMap<(i64, i64), HashMap<i64, u32>> = HashMap::new();
    let mut path_cache: HashMap<(usize, usize), Option<Vec<usize>>> = HashMap::new();
    let (mut min_bin, mut max_bin) = (i64::MAX, i64::MIN);
    let mut routed = 0usize;
    for p in od {
        let (sh, dh) = (station_hub[p.origin], station_hub[p.dest]);
        if sh == dh {
            continue; // both endpoints in one hub — no corridor
        }
        let path = path_cache
            .entry((sh, dh))
            .or_insert_with(|| dijkstra_path(&adj, &edge_len, params.bundle_k, sh, dh))
            .clone();
        let Some(path) = path else { continue };
        routed += 1;
        for w in path.windows(2) {
            let e = (w[0] as i64, w[1] as i64);
            let ec = dir_counts.entry(e).or_default();
            for (&bin, &c) in &p.bins {
                *ec.entry(bin).or_insert(0) += c;
                min_bin = min_bin.min(bin);
                max_bin = max_bin.max(bin);
            }
        }
    }
    println!(
        "   ✓ routed {} / {} OD pairs onto the flow graph",
        routed,
        od.len()
    );

    if dir_counts.is_empty() {
        let cols = vec![
            PropertyColumn::numeric("max_count"),
            PropertyColumn::numeric("total_count"),
            PropertyColumn::numeric("min_zoom"),
        ];
        StreamingLineStringParquetWriter::with_columns(output, cols)?.finish()?;
        return Ok((0, 0, 0, 0));
    }

    let num_buckets = (max_bin - min_bin + 1) as usize;
    let bucket0 = min_bin * bin_ms;
    let range_end = bucket0 + num_buckets as i64 * bin_ms;

    // 4. Stroke-synthesize + emit (with Laplacian smoothing for the flowing look).
    let features = write_flow_network(
        output,
        &dir_counts,
        &hubs,
        lon_scale,
        min_bin,
        num_buckets,
        bucket0,
        range_end,
        params,
    )?;

    Ok((features, num_buckets, bucket0, range_end))
}

/// Greedy radius clustering: process stations in a stable coordinate order,
/// assigning each to the nearest existing hub within `radius_m` (else it seeds a
/// new hub). Hub positions are the member centroids. Deterministic.
fn cluster_hubs(
    stations: &[[f64; 2]],
    radius_m: f64,
    lon_scale: f64,
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut assign = vec![usize::MAX; stations.len()];
    if radius_m <= 0.0 {
        // One hub per station.
        return (stations.to_vec(), (0..stations.len()).collect());
    }
    let r_deg = radius_m / M_PER_DEG;
    let r2 = r_deg * r_deg;

    let mut order: Vec<usize> = (0..stations.len()).collect();
    order.sort_by(|&a, &b| {
        stations[a][0]
            .total_cmp(&stations[b][0])
            .then(stations[a][1].total_cmp(&stations[b][1]))
    });

    let mut seeds: Vec<[f64; 2]> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for &si in &order {
        let s = stations[si];
        let mut best: Option<(f64, usize)> = None;
        for (hi, h) in seeds.iter().enumerate() {
            let dx = (s[0] - h[0]) * lon_scale;
            let dy = s[1] - h[1];
            let d2 = dx * dx + dy * dy;
            if d2 <= r2 && best.map_or(true, |(bd, _)| d2 < bd) {
                best = Some((d2, hi));
            }
        }
        match best {
            Some((_, hi)) => {
                assign[si] = hi;
                members[hi].push(si);
            }
            None => {
                assign[si] = seeds.len();
                seeds.push(s);
                members.push(vec![si]);
            }
        }
    }

    let hubs: Vec<[f64; 2]> = members
        .iter()
        .map(|mem| {
            let n = mem.len() as f64;
            [
                mem.iter().map(|&si| stations[si][0]).sum::<f64>() / n,
                mem.iter().map(|&si| stations[si][1]).sum::<f64>() / n,
            ]
        })
        .collect();
    (hubs, assign)
}

/// Delaunay triangulation of the hubs → undirected adjacency + per-edge length
/// (metres). Edges longer than `max_edge_m` are dropped (hull artifacts).
fn delaunay_graph(
    hubs: &[[f64; 2]],
    lon_scale: f64,
    max_edge_m: f64,
) -> (HashMap<usize, Vec<usize>>, HashMap<(usize, usize), f64>) {
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut edge_len: HashMap<(usize, usize), f64> = HashMap::new();
    if hubs.len() < 3 {
        return (adj, edge_len);
    }
    // Triangulate in an isotropic metric space (lon scaled by cos-lat).
    let pts: Vec<Point> = hubs
        .iter()
        .map(|h| Point {
            x: h[0] * lon_scale,
            y: h[1],
        })
        .collect();
    let tri = triangulate(&pts);

    for t in tri.triangles.chunks_exact(3) {
        add_hub_edge(
            &mut adj,
            &mut edge_len,
            hubs,
            lon_scale,
            max_edge_m,
            t[0],
            t[1],
        );
        add_hub_edge(
            &mut adj,
            &mut edge_len,
            hubs,
            lon_scale,
            max_edge_m,
            t[1],
            t[2],
        );
        add_hub_edge(
            &mut adj,
            &mut edge_len,
            hubs,
            lon_scale,
            max_edge_m,
            t[2],
            t[0],
        );
    }

    // Degenerate input (all hubs collinear ⇒ no triangles): fall back to a path
    // graph over hubs sorted by position so routing still connects them. Real
    // scattered station sets never hit this; it only guards tiny/synthetic cases.
    if edge_len.is_empty() && hubs.len() >= 2 {
        let mut order: Vec<usize> = (0..hubs.len()).collect();
        order.sort_by(|&a, &b| {
            (hubs[a][0] * lon_scale)
                .total_cmp(&(hubs[b][0] * lon_scale))
                .then(hubs[a][1].total_cmp(&hubs[b][1]))
        });
        for w in order.windows(2) {
            add_hub_edge(
                &mut adj,
                &mut edge_len,
                hubs,
                lon_scale,
                max_edge_m,
                w[0],
                w[1],
            );
        }
    }

    for nbrs in adj.values_mut() {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    (adj, edge_len)
}

/// Insert one undirected hub edge (metric length ≤ `max_edge_m`) into the graph.
fn add_hub_edge(
    adj: &mut HashMap<usize, Vec<usize>>,
    edge_len: &mut HashMap<(usize, usize), f64>,
    hubs: &[[f64; 2]],
    lon_scale: f64,
    max_edge_m: f64,
    a: usize,
    b: usize,
) {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if edge_len.contains_key(&(lo, hi)) {
        return;
    }
    let dx = (hubs[a][0] - hubs[b][0]) * lon_scale * M_PER_DEG;
    let dy = (hubs[a][1] - hubs[b][1]) * M_PER_DEG;
    let len = (dx * dx + dy * dy).sqrt();
    if max_edge_m > 0.0 && len > max_edge_m {
        return;
    }
    edge_len.insert((lo, hi), len);
    adj.entry(a).or_default().push(b);
    adj.entry(b).or_default().push(a);
}

#[derive(Copy, Clone)]
struct HeapItem {
    cost: f64,
    node: usize,
}
impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.cost == o.cost && self.node == o.node
    }
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        // BinaryHeap is a max-heap; invert so the SMALLEST cost pops first, with a
        // deterministic node tie-break (smaller node first).
        o.cost
            .total_cmp(&self.cost)
            .then_with(|| o.node.cmp(&self.node))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Shortest path `src..dst` on the hub graph with edge cost `length^k`, returning
/// the hub-index sequence (or `None` if disconnected). Deterministic tie-breaks.
fn dijkstra_path(
    adj: &HashMap<usize, Vec<usize>>,
    edge_len: &HashMap<(usize, usize), f64>,
    k: f64,
    src: usize,
    dst: usize,
) -> Option<Vec<usize>> {
    let mut dist: HashMap<usize, f64> = HashMap::new();
    let mut prev: HashMap<usize, usize> = HashMap::new();
    let mut heap = BinaryHeap::new();
    dist.insert(src, 0.0);
    heap.push(HeapItem {
        cost: 0.0,
        node: src,
    });
    while let Some(HeapItem { cost, node }) = heap.pop() {
        if node == dst {
            break;
        }
        if cost > *dist.get(&node).unwrap_or(&f64::INFINITY) {
            continue;
        }
        for &nb in adj.get(&node).into_iter().flatten() {
            let (lo, hi) = if node < nb { (node, nb) } else { (nb, node) };
            let w = edge_len[&(lo, hi)].powf(k);
            let nd = cost + w;
            if nd < *dist.get(&nb).unwrap_or(&f64::INFINITY) {
                dist.insert(nb, nd);
                prev.insert(nb, node);
                heap.push(HeapItem { cost: nd, node: nb });
            }
        }
    }
    if !dist.contains_key(&dst) {
        return None;
    }
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        cur = prev[&cur];
        path.push(cur);
    }
    path.reverse();
    Some(path)
}

/// Synthesize directed trunks from the graph flow and emit them as
/// merged-corridor features (chunked, per-vertex × per-bucket matrix, volume LOD,
/// Laplacian-smoothed). Mirrors `FlowAggregator::write_parquet_strokes` but over
/// abstract hubs instead of OSM nodes.
#[allow(clippy::too_many_arguments)]
fn write_flow_network(
    output: &Path,
    dir_counts: &HashMap<(i64, i64), HashMap<i64, u32>>,
    hubs: &[[f64; 2]],
    lon_scale: f64,
    min_bin: i64,
    num_buckets: usize,
    bucket0: i64,
    range_end: i64,
    params: &FlowGraphParams,
) -> Result<usize> {
    let cols = vec![
        PropertyColumn::numeric("max_count"),
        PropertyColumn::numeric("total_count"),
        PropertyColumn::numeric("min_zoom"),
    ];
    let mut writer = StreamingLineStringParquetWriter::with_columns(output, cols)?;

    let mut edge_total: HashMap<(i64, i64), u32> = HashMap::new();
    for (e, bins) in dir_counts {
        edge_total.insert(*e, bins.values().copied().sum());
    }

    let strokes = synthesize_strokes(
        dir_counts,
        &edge_total,
        |id| (hubs[id as usize][0], hubs[id as usize][1]),
        lon_scale,
        &params.stroke,
    );

    let mut features = 0usize;
    for stroke in &strokes {
        let mut start = 0usize;
        while start + 1 < stroke.len() {
            let end = (start + MAX_CHAIN_SEGMENTS + 1).min(stroke.len());
            let chunk = &stroke[start..end];
            start = end - 1; // overlap one node so chunks connect

            let dedges: Vec<(i64, i64)> = chunk.windows(2).map(|w| (w[0], w[1])).collect();
            let peak_seg_total = dedges
                .iter()
                .map(|e| edge_total.get(e).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            if peak_seg_total < params.stroke.min_stroke_trips {
                continue;
            }

            // Per-HUB-vertex value matrix (nh vertices) — built before splining so
            // the resample can interpolate it 1:1 with the smoothed geometry.
            let hub_coords: Vec<[f64; 2]> = chunk.iter().map(|&id| hubs[id as usize]).collect();
            let nh = hub_coords.len();
            let mut matrix_hub = vec![0.0f32; nh * num_buckets];
            let mut overall_max = 0u32;
            for b in 0..num_buckets {
                let bin = min_bin + b as i64;
                let seg_counts: Vec<u32> = dedges
                    .iter()
                    .map(|e| {
                        let c = dir_counts
                            .get(e)
                            .and_then(|m| m.get(&bin))
                            .copied()
                            .unwrap_or(0);
                        overall_max = overall_max.max(c);
                        c
                    })
                    .collect();
                let vvals = vertex_values_from_segment_counts(&seg_counts);
                for (v, &val) in vvals.iter().enumerate() {
                    matrix_hub[v * num_buckets + b] = val;
                }
            }
            if overall_max == 0 {
                continue;
            }

            // Catmull-Rom spline the angular hub polyline into a flowing curve
            // (force-directed look), resampling the matrix in lockstep; a couple of
            // light Laplacian passes remove any spline overshoot; then bake the
            // twin-ribbon offset (right of travel; the reverse trunk flips side).
            let (mut coords, matrix) =
                resample_spline(&hub_coords, &matrix_hub, num_buckets, params.spline_samples);
            smooth_polyline(&mut coords, params.smooth_iters);
            offset_polyline_right(&mut coords, params.ribbon_offset_m, lon_scale);

            let mut properties = Map::new();
            properties.insert("max_count".to_string(), json!(overall_max));
            properties.insert("total_count".to_string(), json!(peak_seg_total));
            properties.insert(
                "min_zoom".to_string(),
                json!(stroke_min_zoom(peak_seg_total)),
            );

            writer.write_linestring(&LineStringRecord {
                coordinates: coords,
                timestamp_ms: bucket0,
                end_timestamp_ms: Some(range_end),
                vertex_timestamps_ms: None,
                vertex_values: None,
                vertex_value_matrix: Some(matrix),
                properties,
            })?;
            features += 1;
        }
    }
    writer.finish()?;
    Ok(features)
}

/// Shift a polyline perpendicular to its travel direction, to the RIGHT of a→b,
/// by `offset_m` metres (in-place; vertex count preserved so the per-vertex
/// matrix stays aligned). Reversing the vertex order flips the side, which is how
/// opposing directed trunks separate into twin ribbons. Interior vertices use the
/// averaged normal of their two adjacent segments (a simple miter).
fn offset_polyline_right(coords: &mut [[f64; 2]], offset_m: f64, lon_scale: f64) {
    let n = coords.len();
    if n < 2 || offset_m == 0.0 {
        return;
    }
    // Work in a local metric frame (metres).
    let mm: Vec<[f64; 2]> = coords
        .iter()
        .map(|c| [c[0] * lon_scale * M_PER_DEG, c[1] * M_PER_DEG])
        .collect();
    // Right-of-travel unit normal of segment a→b: rotate the tangent −90° → (dy, −dx).
    let seg_normal = |a: [f64; 2], b: [f64; 2]| -> [f64; 2] {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            [0.0, 0.0]
        } else {
            [dy / len, -dx / len]
        }
    };
    let mut normals = vec![[0.0f64; 2]; n];
    for (i, slot) in normals.iter_mut().enumerate() {
        let prev = if i > 0 {
            Some(seg_normal(mm[i - 1], mm[i]))
        } else {
            None
        };
        let next = if i < n - 1 {
            Some(seg_normal(mm[i], mm[i + 1]))
        } else {
            None
        };
        *slot = match (prev, next) {
            (Some(p), Some(q)) => {
                let (sx, sy) = (p[0] + q[0], p[1] + q[1]);
                let l = (sx * sx + sy * sy).sqrt();
                if l < 1e-9 {
                    p
                } else {
                    [sx / l, sy / l]
                }
            }
            (Some(p), None) => p,
            (None, Some(q)) => q,
            (None, None) => [0.0, 0.0],
        };
    }
    for i in 0..n {
        let x = mm[i][0] + normals[i][0] * offset_m;
        let y = mm[i][1] + normals[i][1] * offset_m;
        coords[i] = [x / (lon_scale * M_PER_DEG), y / M_PER_DEG];
    }
}

/// Catmull-Rom spline resample of a polyline AND its per-vertex value matrix:
/// subdivides each hub segment into `samples` sub-points on a smooth interpolating
/// spline (the curve passes through every hub), linearly interpolating the matrix
/// rows so per-vertex values stay aligned 1:1 with the denser geometry. Turns the
/// angular hub polyline into a flowing curve (the force-directed aesthetic).
/// Returns the inputs unchanged when there are < 3 hubs or `samples <= 1`.
fn resample_spline(
    coords: &[[f64; 2]],
    matrix: &[f32],
    nb: usize,
    samples: usize,
) -> (Vec<[f64; 2]>, Vec<f32>) {
    let n = coords.len();
    if n < 3 || samples <= 1 {
        return (coords.to_vec(), matrix.to_vec());
    }
    let k = samples;
    let total = (n - 1) * k;
    let mut out_c = Vec::with_capacity(total + 1);
    let mut out_m = Vec::with_capacity((total + 1) * nb);
    let at = |i: isize| coords[i.clamp(0, n as isize - 1) as usize];
    for step in 0..=total {
        let u = step as f64 / k as f64; // parameter in [0, n-1]
        let i = (u.floor() as usize).min(n - 2);
        let t = u - i as f64;
        let (p0, p1, p2, p3) = (
            at(i as isize - 1),
            coords[i],
            coords[i + 1],
            at(i as isize + 2),
        );
        let (t2, t3) = (t * t, t * t * t);
        let cr = |a: f64, b: f64, c: f64, d: f64| {
            0.5 * (2.0 * b
                + (-a + c) * t
                + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                + (-a + 3.0 * b - 3.0 * c + d) * t3)
        };
        out_c.push([
            cr(p0[0], p1[0], p2[0], p3[0]),
            cr(p0[1], p1[1], p2[1], p3[1]),
        ]);
        let tf = t as f32;
        for b in 0..nb {
            let a = matrix[i * nb + b];
            let c = matrix[(i + 1) * nb + b];
            out_m.push(a + (c - a) * tf);
        }
    }
    (out_c, out_m)
}

/// In-place Laplacian smoothing of a polyline: each interior vertex moves halfway
/// toward the midpoint of its neighbours, `iters` times; endpoints pinned. Vertex
/// count is preserved, so the per-vertex value matrix stays aligned.
fn smooth_polyline(coords: &mut [[f64; 2]], iters: usize) {
    if coords.len() < 3 || iters == 0 {
        return;
    }
    for _ in 0..iters {
        let prev = coords.to_vec();
        for i in 1..coords.len() - 1 {
            let mx = 0.5 * (prev[i - 1][0] + prev[i + 1][0]);
            let my = 0.5 * (prev[i - 1][1] + prev[i + 1][1]);
            coords[i][0] = prev[i][0] + 0.5 * (mx - prev[i][0]);
            coords[i][1] = prev[i][1] + 0.5 * (my - prev[i][1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol
    }

    #[test]
    fn clustering_merges_within_radius() {
        // Two pairs ~50 m apart, pairs ~2 km apart → 2 hubs at radius 200 m.
        let stations = vec![
            [-73.600, 45.500],
            [-73.6005, 45.5001],
            [-73.580, 45.500],
            [-73.5805, 45.5001],
        ];
        let (hubs, assign) = cluster_hubs(&stations, 200.0, 0.70);
        assert_eq!(hubs.len(), 2);
        assert_eq!(assign[0], assign[1]);
        assert_eq!(assign[2], assign[3]);
        assert_ne!(assign[0], assign[2]);
    }

    #[test]
    fn no_clustering_at_zero_radius() {
        let stations = vec![[-73.6, 45.5], [-73.5, 45.5]];
        let (hubs, assign) = cluster_hubs(&stations, 0.0, 0.70);
        assert_eq!(hubs.len(), 2);
        assert_eq!(assign, vec![0, 1]);
    }

    #[test]
    fn delaunay_connects_a_square() {
        let hubs = vec![
            [-73.60, 45.50],
            [-73.59, 45.50],
            [-73.60, 45.51],
            [-73.59, 45.51],
        ];
        let (adj, edges) = delaunay_graph(&hubs, 0.70, 0.0);
        // A 4-point square triangulates into 2 triangles sharing a diagonal → 5 edges.
        assert_eq!(edges.len(), 5);
        assert!(adj.values().all(|n| !n.is_empty()));
    }

    #[test]
    fn dijkstra_prefers_bundled_chain_under_high_k() {
        // Path A-B-C collinear; direct A-C also present but longer per hop.
        //   A(0) - B(1) - C(2), plus a long direct A-C.
        let hubs = vec![[0.0, 45.5], [0.01, 45.5], [0.02, 45.5]];
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut edge_len: HashMap<(usize, usize), f64> = HashMap::new();
        for (a, b, l) in [(0usize, 1usize, 1000.0), (1, 2, 1000.0), (0, 2, 2000.0)] {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
            edge_len.insert((a, b), l);
        }
        // With k=2: chain cost = 1000^2 + 1000^2 = 2e6 < direct 2000^2 = 4e6.
        let path = dijkstra_path(&adj, &edge_len, 2.0, 0, 2).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn smoothing_pins_endpoints_and_relaxes_a_kink() {
        let mut coords = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
        smooth_polyline(&mut coords, 2);
        assert!(near(coords[0], [0.0, 0.0], 1e-9), "start pinned");
        assert!(near(coords[2], [2.0, 0.0], 1e-9), "end pinned");
        assert!(coords[1][1] < 1.0 && coords[1][1] > 0.0, "kink relaxed");
    }

    #[test]
    fn spline_preserves_endpoints_and_resamples_matrix() {
        let coords = vec![[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]];
        let matrix = vec![10.0f32, 20.0, 10.0]; // nb=1, per-vertex
        let (c, m) = resample_spline(&coords, &matrix, 1, 4);
        assert_eq!(c.len(), (3 - 1) * 4 + 1, "9 points for 2 segments × 4");
        assert_eq!(m.len(), c.len(), "matrix stays 1:1 with vertices");
        assert!(
            (c[0][0]).abs() < 1e-9 && (c[0][1]).abs() < 1e-9,
            "start preserved"
        );
        assert!((c[c.len() - 1][0] - 2.0).abs() < 1e-9, "end preserved");
        assert!((m[0] - 10.0).abs() < 1e-6 && (m[m.len() - 1] - 10.0).abs() < 1e-6);
        assert!(
            m.iter().all(|&v| (9.9..=20.1).contains(&v)),
            "matrix stays in range"
        );
    }

    #[test]
    fn offset_flips_side_on_reverse() {
        // East-going segment: right of travel is SOUTH; the reverse (west-going)
        // trunk shifts NORTH — the opposite side, so the two make twin ribbons.
        let mut fwd = vec![[-73.60, 45.50], [-73.59, 45.50]];
        offset_polyline_right(&mut fwd, 20.0, 0.70);
        assert!(
            fwd[0][1] < 45.50,
            "east-going shifts south, got {}",
            fwd[0][1]
        );
        let mut rev = vec![[-73.59, 45.50], [-73.60, 45.50]];
        offset_polyline_right(&mut rev, 20.0, 0.70);
        assert!(
            rev[0][1] > 45.50,
            "west-going shifts north, got {}",
            rev[0][1]
        );
    }

    #[test]
    fn build_and_write_produces_a_corridor() {
        // Three collinear stations; one OD pair A→C routes through B.
        let stations = vec![[-73.60, 45.50], [-73.59, 45.50], [-73.58, 45.50]];
        let od = vec![OdPair {
            origin: 0,
            dest: 2,
            bins: HashMap::from([(0i64, 9u32)]),
        }];
        let tmp =
            std::env::temp_dir().join(format!("stt_flowgraph_{}.parquet", std::process::id()));
        let (features, buckets, bucket0, range_end) = build_and_write(
            &tmp,
            &stations,
            &od,
            3_600_000,
            &FlowGraphParams {
                hub_radius_m: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(features >= 1);
        assert_eq!(buckets, 1);
        assert_eq!((bucket0, range_end), (0, 3_600_000));
        let _ = std::fs::remove_file(&tmp);
    }
}
