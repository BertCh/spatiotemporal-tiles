//! OSM-street flow aggregation for NYC taxi trips (`--flows`).
//!
//! Builds the *overview* counterpart to `--paths`: instead of one feature per
//! trip, aggregate every routed trip's traffic onto the **actual OSM street
//! network** ([`super::osm_streets`]) and emit one corridor per trafficked OSM
//! way carrying a per-vertex × per-time-bin **value matrix** of traversal
//! counts. Because OSRM routed these trips on the very `.osm.pbf` we read,
//! each trip segment maps to its OSM edge (exact endpoint match, with a
//! nearest-edge fallback), so corridors trace REAL street geometry —
//! intersection-to-intersection node sequences, no snap lattice, no greedy-chain
//! zig-zags.
//!
//! LOD: each corridor carries a `min_zoom` derived from its road class
//! (motorway/trunk at z8 … service at z13). The build skips a feature at zooms
//! below its `min_zoom`, so the overview shows only major roads when zoomed out
//! and fills in every street up close (vector-tile-style road-class LOD).
//!
//! Encoding (consumed by `FlowCorridorLayer`, unchanged):
//! - geometry: the OSM way's node polyline (chunked at `MAX_CHAIN_SEGMENTS`).
//! - `timestamp` = global bucket-0 start, `end_timestamp` = end of the whole
//!   range — the tile spans the full range so it loads once and never re-fetches.
//! - `vertex_value_matrix` = flat **vertex-major** `[vertex][bucket]` f32 grid;
//!   per bucket the per-vertex values follow the endpoint/mean rule over the
//!   way's per-edge counts. `num_buckets = matrix.len() / num_vertices`.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Map};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

use crate::common::{LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter};
use crate::datasets::osm_streets::{EdgeKey, OsmNetwork};

/// Cap corridor features so a single way can't span a borough — very long
/// features land in (and bloat) every tile they cross at every zoom.
pub(crate) const MAX_CHAIN_SEGMENTS: usize = 64;

/// Undirected edge key (endpoints ascending) — matches `OsmNetwork`'s keying.
fn edge_key(a: i64, b: i64) -> EdgeKey {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Parse a bin duration like `"15m"`, `"1h"`, `"90s"`, `"1d"` into ms.
pub fn parse_bin_ms(s: &str) -> Result<i64> {
    let s = s.trim();
    let split = s
        .find(|c: char| !c.is_ascii_digit())
        .ok_or_else(|| anyhow!("flow bin '{}' missing unit (s/m/h/d)", s))?;
    let (num, unit) = s.split_at(split);
    let n: i64 = num.parse().context("invalid flow bin number")?;
    let unit_ms = match unit {
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        _ => return Err(anyhow!("flow bin unit '{}' not one of s/m/h/d", unit)),
    };
    if n <= 0 {
        return Err(anyhow!("flow bin must be positive: {}", s));
    }
    Ok(n * unit_ms)
}

/// Accumulates per-OSM-edge, per-time-bin traversal counts across trips.
///
/// `counts` maps each undirected OSM edge to a sparse `bin → count` map; the bin
/// axis is densified into contiguous buckets `[min_bin, max_bin]` at write time,
/// so every corridor shares one global bucket axis (the renderer's single
/// bucket-index uniform).
pub struct FlowAggregator {
    bin_ms: i64,
    network: OsmNetwork,
    counts: HashMap<EdgeKey, HashMap<i64, u32>>,
    /// Directed per-edge counts `(from_node, to_node) → bin → count`, populated
    /// only by [`Self::add_route_bins_directed`] (the `bixi --merged-paths`
    /// stroke path). Empty for the undirected `--streets`/`--flows` modes. Kept
    /// separate from `counts` so a corridor that is two-way keeps A→B and B→A
    /// distinct (asymmetric ridership, one-way pairs) for the twin-ribbon render.
    dir_counts: HashMap<(i64, i64), HashMap<i64, u32>>,
    /// Directional mode for the undirected `--streets` path (opt-in via
    /// [`Self::set_directional`]; used by `bixi --streets --directional`). When
    /// on, also accumulate the *signed* net flow per undirected edge — positive
    /// toward the edge's canonical `a<b` orientation, negative the other way — so
    /// [`Self::write_parquet`] can PRE-ORIENT each corridor's geometry toward its
    /// dominant travel direction (vertex winding == flow) for the chevron
    /// renderer. Distinct from `dir_counts` (which keeps A→B and B→A separate for
    /// the twin-ribbon stroke): this collapses to one signed net per undirected
    /// edge. The undirected `counts` (the magnitude matrix) are untouched, so a
    /// non-directional `--streets` build stays byte-identical.
    directional: bool,
    signed: HashMap<EdgeKey, HashMap<i64, i64>>,
    /// When on (opt-in via [`Self::set_per_bucket_direction`], `bixi --streets
    /// --per-bucket-direction`), [`Self::write_parquet`] encodes the SIGN of each
    /// bucket's net flow (along the emitted winding) into the magnitude matrix:
    /// `matrix[v][b] = |volume| * dir`, `dir ∈ {+1,-1}`. The renderer colours by
    /// `abs` (volume) and flips the chevrons by the sign, so a corridor's arrows
    /// change direction per time-step (morning inbound → evening outbound).
    /// Requires `directional` (the signed map). Off → plain magnitude matrix.
    per_bucket_direction: bool,
    min_bin: i64,
    max_bin: i64,
    trips_added: usize,
    trips_skipped: usize,
    matched_exact: usize,
    matched_fallback: usize,
    unmatched: usize,
}

impl FlowAggregator {
    /// Aggregate onto the given OSM road network.
    pub fn new(bin_ms: i64, network: OsmNetwork) -> Self {
        Self {
            bin_ms,
            network,
            counts: HashMap::new(),
            dir_counts: HashMap::new(),
            directional: false,
            signed: HashMap::new(),
            per_bucket_direction: false,
            min_bin: i64::MAX,
            max_bin: i64::MIN,
            trips_added: 0,
            trips_skipped: 0,
            matched_exact: 0,
            matched_fallback: 0,
            unmatched: 0,
        }
    }

    /// Enable directional accumulation on the undirected `--streets` path: track
    /// signed per-edge net flow so [`Self::write_parquet`] orients each corridor's
    /// geometry toward its dominant travel direction (winding == flow) for the
    /// chevron renderer. Off by default (geometry follows the OSM way's
    /// digitization order; undirected magnitude only).
    pub fn set_directional(&mut self, on: bool) {
        self.directional = on;
    }

    /// Enable per-bucket direction encoding (implies [`Self::set_directional`],
    /// since it needs the signed net map). See the `per_bucket_direction` field.
    /// Must be called BEFORE ingestion so the signed map is populated.
    pub fn set_per_bucket_direction(&mut self, on: bool) {
        self.per_bucket_direction = on;
        if on {
            self.directional = true;
        }
    }

    /// Add one routed trip. `vertex_times_ms` (when present) must align with
    /// `coords`; otherwise traversal times are interpolated by cumulative
    /// haversine distance over the trip's [start, end] window — mirroring
    /// stt-build's own fallback, so no trips are dropped from the aggregate.
    pub fn add_trip(
        &mut self,
        coords: &[[f64; 2]],
        vertex_times_ms: Option<&[i64]>,
        start_ms: i64,
        end_ms: i64,
    ) {
        if coords.len() < 2 {
            self.trips_skipped += 1;
            return;
        }
        let times: Vec<i64> = match vertex_times_ms {
            Some(t) if t.len() == coords.len() => t.to_vec(),
            _ => interpolate_times_by_distance(coords, start_ms, end_ms),
        };

        for i in 1..coords.len() {
            let p0 = coords[i - 1];
            let p1 = coords[i];
            // Exact-vs-fallback bookkeeping: re-derive whether the match was a
            // direct endpoint hit so the build can report match quality.
            let edge = match self.network.match_segment(p0, p1) {
                Some(e) => {
                    if self.network.is_exact_pair(p0, p1) {
                        self.matched_exact += 1;
                    } else {
                        self.matched_fallback += 1;
                    }
                    e
                }
                None => {
                    self.unmatched += 1;
                    continue;
                }
            };
            let mid = (times[i - 1] + times[i]) / 2;
            let bin = mid.div_euclid(self.bin_ms);
            *self.counts.entry(edge).or_default().entry(bin).or_insert(0) += 1;
            if self.directional {
                let sign = segment_orientation(&self.network, p0, p1, edge);
                *self.signed.entry(edge).or_default().entry(bin).or_insert(0) += sign;
            }
            self.min_bin = self.min_bin.min(bin);
            self.max_bin = self.max_bin.max(bin);
        }
        self.trips_added += 1;
    }

    /// Add one routed OD corridor carrying a per-bin trip count (the
    /// **route-once-then-distribute** path used by BIXI `--streets`): route a
    /// unique origin→destination pair through OSRM once, then for every
    /// `(bin, count)` it served, add `count` to each OSM edge the route
    /// traverses in that bin. `bins` keys are absolute bin indices on the same
    /// `bin_ms` axis as this aggregator (i.e. `start_ms.div_euclid(bin_ms)`).
    ///
    /// Unlike [`Self::add_trip`], the whole route lands in each served bin — no
    /// intra-route timing — which suits bike trips that are short relative to a
    /// coarse (e.g. 1 h) bucket.
    pub fn add_route_bins(&mut self, coords: &[[f64; 2]], bins: &HashMap<i64, u32>) {
        if coords.len() < 2 || bins.is_empty() {
            self.trips_skipped += 1;
            return;
        }
        for i in 1..coords.len() {
            let p0 = coords[i - 1];
            let p1 = coords[i];
            let edge = match self.network.match_segment(p0, p1) {
                Some(e) => {
                    if self.network.is_exact_pair(p0, p1) {
                        self.matched_exact += 1;
                    } else {
                        self.matched_fallback += 1;
                    }
                    e
                }
                None => {
                    self.unmatched += 1;
                    continue;
                }
            };
            // Signed orientation vs the edge's canonical a<b — computed before the
            // mutable borrows below (needs an immutable borrow of `network`).
            let sign = if self.directional {
                segment_orientation(&self.network, p0, p1, edge)
            } else {
                0
            };
            let edge_counts = self.counts.entry(edge).or_default();
            for (&bin, &count) in bins {
                *edge_counts.entry(bin).or_insert(0) += count;
                self.min_bin = self.min_bin.min(bin);
                self.max_bin = self.max_bin.max(bin);
            }
            if self.directional {
                let sc = self.signed.entry(edge).or_default();
                for (&bin, &count) in bins {
                    *sc.entry(bin).or_insert(0) += sign * count as i64;
                }
            }
        }
        self.trips_added += 1;
    }

    /// Directed counterpart of [`Self::add_route_bins`] for `bixi --merged-paths`:
    /// distribute an OD corridor's per-bin counts onto **directed** edge keys
    /// `(from_node, to_node)` oriented in travel direction (via
    /// [`OsmNetwork::match_segment_directed`]). A→B and B→A traffic therefore stay
    /// on separate keys even on a two-way street, so the stroke synthesis can emit
    /// two offset ribbons whose widths reflect each direction's ridership.
    pub fn add_route_bins_directed(&mut self, coords: &[[f64; 2]], bins: &HashMap<i64, u32>) {
        if coords.len() < 2 || bins.is_empty() {
            self.trips_skipped += 1;
            return;
        }
        for i in 1..coords.len() {
            let p0 = coords[i - 1];
            let p1 = coords[i];
            let dedge = match self.network.match_segment_directed(p0, p1) {
                Some(e) => {
                    if self.network.is_exact_pair(p0, p1) {
                        self.matched_exact += 1;
                    } else {
                        self.matched_fallback += 1;
                    }
                    e
                }
                None => {
                    self.unmatched += 1;
                    continue;
                }
            };
            let edge_counts = self.dir_counts.entry(dedge).or_default();
            for (&bin, &count) in bins {
                *edge_counts.entry(bin).or_insert(0) += count;
                self.min_bin = self.min_bin.min(bin);
                self.max_bin = self.max_bin.max(bin);
            }
        }
        self.trips_added += 1;
    }

    /// (trips added, trips skipped, exact matches, fallback matches, misses).
    pub fn stats(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.trips_added,
            self.trips_skipped,
            self.matched_exact,
            self.matched_fallback,
            self.unmatched,
        )
    }

    /// The densified bucket axis `(bucket0_ms, range_end_ms)` the corridors span,
    /// or `None` if nothing was aggregated. Matches the `timestamp`/`end_timestamp`
    /// written by [`Self::write_parquet`] — use it to set the showcase `timeRange`.
    pub fn bucket_span_ms(&self) -> Option<(i64, i64)> {
        // Directed mode (`--merged-paths`) fills `dir_counts`, not `counts`; both
        // share the same min_bin/max_bin axis, so either being non-empty is enough.
        if self.counts.is_empty() && self.dir_counts.is_empty() {
            return None;
        }
        let num_buckets = self.max_bin - self.min_bin + 1;
        let bucket0 = self.min_bin * self.bin_ms;
        let range_end = bucket0 + num_buckets * self.bin_ms;
        Some((bucket0, range_end))
    }

    /// Emit one corridor per trafficked OSM way (chunked at `MAX_CHAIN_SEGMENTS`)
    /// carrying its `[vertex][bucket]` matrix + a `min_zoom` from its road class.
    /// Returns (features written, number of buckets).
    pub fn write_parquet(self, output: &Path) -> Result<(usize, usize)> {
        let property_columns = vec![
            PropertyColumn::numeric("max_count"),
            PropertyColumn::numeric("min_zoom"),
        ];
        let mut writer = StreamingLineStringParquetWriter::with_columns(output, property_columns)?;

        if self.counts.is_empty() {
            writer.finish()?;
            return Ok((0, 0));
        }

        let bin_ms = self.bin_ms;
        let min_bin = self.min_bin;
        let num_buckets = (self.max_bin - min_bin + 1) as usize;
        let bucket0 = min_bin * bin_ms;
        let range_end = bucket0 + num_buckets as i64 * bin_ms;

        // Trafficked ways = owning ways of every counted edge, sorted for
        // deterministic (content-addressable) output.
        let mut trafficked: HashSet<i64> = HashSet::new();
        for e in self.counts.keys() {
            if let Some(wid) = self.network.edge_way_id(*e) {
                trafficked.insert(wid);
            }
        }
        let mut way_ids: Vec<i64> = trafficked.into_iter().collect();
        way_ids.sort_unstable();

        let mut features = 0usize;
        for wid in way_ids {
            let way = self.network.way(wid).expect("trafficked way exists");
            // Split the way at any ref missing coords (clipped extract), then
            // chunk each run into ≤ MAX_CHAIN_SEGMENTS edges (overlapping by one
            // node so adjacent chunks join seamlessly).
            let mut run: Vec<i64> = Vec::new();
            for &r in &way.refs {
                if self.network.node_xy(r).is_some() {
                    run.push(r);
                } else if run.len() >= 2 {
                    features += self.emit_run(
                        &run,
                        way.class.min_zoom(),
                        bucket0,
                        range_end,
                        min_bin,
                        num_buckets,
                        &mut writer,
                    )?;
                    run.clear();
                } else {
                    run.clear();
                }
            }
            if run.len() >= 2 {
                features += self.emit_run(
                    &run,
                    way.class.min_zoom(),
                    bucket0,
                    range_end,
                    min_bin,
                    num_buckets,
                    &mut writer,
                )?;
            }
        }
        writer.finish()?;
        Ok((features, num_buckets))
    }

    /// Emit **coherent directed corridors** for `bixi --merged-paths`: synthesize
    /// long strokes from the directed per-edge flow (degree-2 contraction +
    /// self-best-fit good-continuation at junctions, gated by flow similarity),
    /// then write each stroke (chunked at `MAX_CHAIN_SEGMENTS`) as one LineString
    /// in travel-direction vertex order, carrying its `[vertex][bucket]` directed
    /// volume matrix + a volume-based `min_zoom` LOD. Returns (features, buckets).
    pub fn write_parquet_strokes(
        self,
        output: &Path,
        params: &StrokeParams,
    ) -> Result<(usize, usize)> {
        let property_columns = vec![
            PropertyColumn::numeric("max_count"),
            PropertyColumn::numeric("total_count"),
            PropertyColumn::numeric("min_zoom"),
        ];
        let mut writer = StreamingLineStringParquetWriter::with_columns(output, property_columns)?;

        if self.dir_counts.is_empty() {
            writer.finish()?;
            return Ok((0, 0));
        }

        let min_bin = self.min_bin;
        let num_buckets = (self.max_bin - min_bin + 1) as usize;
        let bucket0 = min_bin * self.bin_ms;
        let range_end = bucket0 + num_buckets as i64 * self.bin_ms;

        // Per-directed-edge total throughput (Σ over buckets) — reused for the
        // continuation flow-similarity gate and the per-corridor volume LOD.
        let mut edge_total: HashMap<(i64, i64), u32> = HashMap::new();
        for (e, bins) in &self.dir_counts {
            edge_total.insert(*e, bins.values().copied().sum());
        }

        let net = &self.network;
        let strokes = synthesize_strokes(
            &self.dir_counts,
            &edge_total,
            |id| net.node_xy(id).unwrap(),
            net.lon_scale(),
            params,
        );

        let mut features = 0usize;
        for stroke in &strokes {
            let mut start = 0usize;
            while start + 1 < stroke.len() {
                let end = (start + MAX_CHAIN_SEGMENTS + 1).min(stroke.len()); // ≤64 edges
                let chunk = &stroke[start..end];
                start = end - 1; // overlap one node so chunks connect

                let dedges: Vec<(i64, i64)> = chunk.windows(2).map(|w| (w[0], w[1])).collect();
                // Representative corridor volume = busiest segment's total
                // throughput (length-independent), driving width + LOD.
                let peak_seg_total = dedges
                    .iter()
                    .map(|e| edge_total.get(e).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                if peak_seg_total < params.min_stroke_trips {
                    continue;
                }

                let coords: Vec<[f64; 2]> = chunk
                    .iter()
                    .map(|&id| {
                        let (x, y) = self.network.node_xy(id).unwrap();
                        [x, y]
                    })
                    .collect();
                let n = coords.len();

                let mut matrix = vec![0.0f32; n * num_buckets];
                let mut overall_max = 0u32;
                for b in 0..num_buckets {
                    let bin = min_bin + b as i64;
                    let seg_counts: Vec<u32> = dedges
                        .iter()
                        .map(|e| {
                            let c = self
                                .dir_counts
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
                        matrix[v * num_buckets + b] = val;
                    }
                }
                if overall_max == 0 {
                    continue;
                }

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
        Ok((features, num_buckets))
    }

    /// Emit a contiguous run of coord-resolved node ids as one or more corridor
    /// features (chunked). Returns the number of features written. Chunks with
    /// no traffic in any bucket are skipped.
    #[allow(clippy::too_many_arguments)]
    fn emit_run(
        &self,
        run: &[i64],
        min_zoom: u8,
        bucket0: i64,
        range_end: i64,
        min_bin: i64,
        num_buckets: usize,
        writer: &mut StreamingLineStringParquetWriter,
    ) -> Result<usize> {
        let mut written = 0usize;
        let mut start = 0usize;
        while start + 1 < run.len() {
            let end = (start + MAX_CHAIN_SEGMENTS + 1).min(run.len()); // ≤64 edges
            let chunk = &run[start..end];
            start = end - 1; // overlap one node so chunks connect

            let mut coords: Vec<[f64; 2]> = chunk
                .iter()
                .map(|&id| {
                    let (x, y) = self.network.node_xy(id).unwrap();
                    [x, y]
                })
                .collect();
            let n = coords.len();
            let edges: Vec<EdgeKey> = chunk.windows(2).map(|w| edge_key(w[0], w[1])).collect();

            let mut matrix = vec![0.0f32; n * num_buckets];
            let mut overall_max = 0u32;
            for b in 0..num_buckets {
                let bin = min_bin + b as i64;
                let seg_counts: Vec<u32> = edges
                    .iter()
                    .map(|e| {
                        let c = self
                            .counts
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
                    matrix[v * num_buckets + b] = val;
                }
            }
            // Skip chunks the taxis never touched (don't bloat with empty geometry).
            if overall_max == 0 {
                continue;
            }

            // Directional: pre-orient the corridor toward its overall-dominant
            // travel direction so the renderer marches chevrons along the winding.
            // A negative net means the chunk's vertex order runs against the
            // dominant flow — reverse both the coords and the matrix rows.
            let do_reverse =
                self.directional && chunk_net_along_winding(&self.signed, chunk, &edges) < 0;
            if do_reverse {
                reverse_corridor(&mut coords, &mut matrix, n, num_buckets);
            }

            // Per-bucket direction: encode the sign of each bucket's net flow
            // (along the FINAL, post-reversal winding) into the magnitude matrix,
            // so the renderer flips the chevrons per time-step while |value| stays
            // the volume for colour. Balanced/zero-net buckets keep the baseline
            // (+, i.e. the overall-dominant winding). Computed after the reversal
            // so signs are relative to the emitted vertex order.
            if self.per_bucket_direction {
                let final_chunk: Vec<i64> = if do_reverse {
                    chunk.iter().rev().copied().collect()
                } else {
                    chunk.to_vec()
                };
                for b in 0..num_buckets {
                    let bin = min_bin + b as i64;
                    // Net along the final traversal order, per edge.
                    let seg_signed: Vec<i64> = final_chunk
                        .windows(2)
                        .map(|w| {
                            let e = edge_key(w[0], w[1]);
                            let along = if w[0] <= w[1] { 1 } else { -1 };
                            let net = self
                                .signed
                                .get(&e)
                                .and_then(|m| m.get(&bin))
                                .copied()
                                .unwrap_or(0);
                            along * net
                        })
                        .collect();
                    for v in 0..n {
                        // Per-vertex net = its adjacent segments' net (same
                        // averaging shape as vertex_values_from_segment_counts).
                        let vnet = if v == 0 {
                            seg_signed[0]
                        } else if v == n - 1 {
                            seg_signed[n - 2]
                        } else {
                            seg_signed[v - 1] + seg_signed[v]
                        };
                        // Flip the sign only when this bucket's net runs against
                        // the winding; |value| (volume) is preserved either way.
                        if vnet < 0 {
                            matrix[v * num_buckets + b] = -matrix[v * num_buckets + b];
                        }
                    }
                }
            }

            let mut properties = Map::new();
            properties.insert("max_count".to_string(), json!(overall_max));
            properties.insert("min_zoom".to_string(), json!(min_zoom));

            writer.write_linestring(&LineStringRecord {
                coordinates: coords,
                timestamp_ms: bucket0,
                end_timestamp_ms: Some(range_end),
                vertex_timestamps_ms: None,
                vertex_values: None,
                vertex_value_matrix: Some(matrix),
                properties,
            })?;
            written += 1;
        }
        Ok(written)
    }
}

fn interpolate_times_by_distance(coords: &[[f64; 2]], start_ms: i64, end_ms: i64) -> Vec<i64> {
    let n = coords.len();
    let mut cum = Vec::with_capacity(n);
    cum.push(0.0f64);
    for i in 1..n {
        let d = crate::common::haversine_distance(
            coords[i - 1][1],
            coords[i - 1][0],
            coords[i][1],
            coords[i][0],
        );
        cum.push(cum[i - 1] + d);
    }
    let total = cum[n - 1];
    if total <= 0.0 || end_ms <= start_ms {
        return vec![start_ms; n];
    }
    let span = (end_ms - start_ms) as f64;
    cum.iter()
        .map(|&d| start_ms + (d / total * span) as i64)
        .collect()
}

/// Sign of a routed segment `p0→p1` relative to an edge's canonical (`a<b`)
/// orientation: `+1` when travel runs from the lower-id endpoint toward the
/// higher (the [`edge_key`] order), `-1` otherwise. Geometric — the dot of the
/// travel vector against the edge's `a→b` vector — so it works for exact and
/// nearest-edge matches alike. Falls back to `+1` if an endpoint lacks a coord.
fn segment_orientation(net: &OsmNetwork, p0: [f64; 2], p1: [f64; 2], e: EdgeKey) -> i64 {
    match (net.node_xy(e.0), net.node_xy(e.1)) {
        (Some((ax, ay)), Some((bx, by))) => {
            let dot = (p1[0] - p0[0]) * (bx - ax) + (p1[1] - p0[1]) * (by - ay);
            if dot >= 0.0 {
                1
            } else {
                -1
            }
        }
        _ => 1,
    }
}

/// Signed net flow along a chunk's winding: for each edge `chunk[k]→chunk[k+1]`,
/// take its stored net (kept along the canonical `a<b`) and flip it when the
/// chunk traverses the edge high→low, then sum over every edge and bucket.
/// Positive ⇒ the chunk's vertex order already matches the dominant travel
/// direction; negative ⇒ reverse it so winding == flow.
fn chunk_net_along_winding(
    signed: &HashMap<EdgeKey, HashMap<i64, i64>>,
    chunk: &[i64],
    edges: &[EdgeKey],
) -> i64 {
    let mut net = 0i64;
    for (k, e) in edges.iter().enumerate() {
        let along = if chunk[k] <= chunk[k + 1] { 1 } else { -1 };
        if let Some(m) = signed.get(e) {
            let edge_net: i64 = m.values().sum();
            net += along * edge_net;
        }
    }
    net
}

/// Reverse a corridor in place: flip the vertex order of both the coordinate
/// list and the vertex-major `[vertex][bucket]` matrix, so the polyline winds the
/// other way while every vertex keeps its own value row.
fn reverse_corridor(coords: &mut [[f64; 2]], matrix: &mut [f32], n: usize, nb: usize) {
    coords.reverse();
    for v in 0..(n / 2) {
        let w = n - 1 - v;
        for b in 0..nb {
            matrix.swap(v * nb + b, w * nb + b);
        }
    }
}

/// Per-vertex values from per-segment counts: endpoints take their single
/// adjacent segment's count, interior vertices the mean of both sides.
pub(crate) fn vertex_values_from_segment_counts(counts: &[u32]) -> Vec<f32> {
    let n = counts.len();
    let mut values = Vec::with_capacity(n + 1);
    values.push(counts[0] as f32);
    for i in 1..n {
        values.push((counts[i - 1] + counts[i]) as f32 / 2.0);
    }
    values.push(counts[n - 1] as f32);
    values
}

/// Tuning for [`FlowAggregator::write_parquet_strokes`] directed stroke synthesis.
#[derive(Clone, Copy, Debug)]
pub struct StrokeParams {
    /// Max deflection angle (degrees) allowed to continue a corridor across a
    /// junction (0 = straight). Larger = longer, less-straight corridors. ~50°.
    pub angle_threshold_deg: f64,
    /// Flow-similarity gate: continue only when `min(w1,w2)/max(w1,w2) ≥ ratio`,
    /// so a corridor SPLITS where a heavy tributary joins or leaves. ~0.35.
    pub flow_similarity_ratio: f64,
    /// Drop corridors whose busiest segment carries fewer than this many trips.
    pub min_stroke_trips: u32,
}

impl Default for StrokeParams {
    fn default() -> Self {
        Self {
            angle_threshold_deg: 50.0,
            flow_similarity_ratio: 0.35,
            min_stroke_trips: 1,
        }
    }
}

/// Deflection angle (degrees; 0 = straight, 180 = U-turn) of the turn a→b→c, with
/// longitude scaled by `lon_scale` so the angle is geometric, not raw-degree.
fn deflection_angle(a: (f64, f64), b: (f64, f64), c: (f64, f64), lon_scale: f64) -> f64 {
    let v1 = ((b.0 - a.0) * lon_scale, b.1 - a.1);
    let v2 = ((c.0 - b.0) * lon_scale, c.1 - b.1);
    let n1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
    let n2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
    if n1 < 1e-12 || n2 < 1e-12 {
        return 180.0;
    }
    let cos = ((v1.0 * v2.0 + v1.1 * v2.1) / (n1 * n2)).clamp(-1.0, 1.0);
    cos.acos().to_degrees()
}

/// Whether two corridor volumes are within the similarity ratio (both > 0).
fn flow_similar(w1: u32, w2: u32, ratio: f64) -> bool {
    if w1 == 0 || w2 == 0 {
        return false;
    }
    let (lo, hi) = if w1 <= w2 { (w1, w2) } else { (w2, w1) };
    (lo as f64) / (hi as f64) >= ratio
}

/// Volume-based LOD: shallowest zoom a corridor of this peak throughput appears
/// at. Heaviest trunks anchor the city overview; light corridors fill in deep.
/// Thresholds tuned from the measured Aug-2024 Montréal corridor distribution
/// (`total_count` = busiest-segment Σ-over-hours: p50≈200, p90≈1500, p95≈2500,
/// p99≈5600), so the city overview (z10) shows only the ~top-5% trunk corridors.
pub(crate) fn stroke_min_zoom(peak: u32) -> u8 {
    match peak {
        p if p >= 2500 => 10, // ~p95: sparse trunk overview
        p if p >= 1000 => 11, // ~p87
        p if p >= 350 => 12,  // ~p63
        p if p >= 80 => 13,   // ~p33
        _ => 14,
    }
}

/// Synthesize long, coherent **directed** corridors ("strokes") from directed
/// per-edge flow.
///
/// Greedy good-continuation: seed from each unused directed edge in sorted order,
/// then extend downstream and upstream, at each junction continuing onto the
/// unused edge of smallest deflection angle (self-best-fit) — subject to the
/// angle cap, the flow-similarity gate (so corridors split where volume changes
/// sharply), and U-turn avoidance. Degree-2 chains contract for free (their lone
/// option always wins). Deterministic (sorted seeds + sorted candidates + a
/// `used` set), so the content-addressed packs stay byte-reproducible.
pub(crate) fn synthesize_strokes<F: Fn(i64) -> (f64, f64)>(
    dir_counts: &HashMap<(i64, i64), HashMap<i64, u32>>,
    edge_total: &HashMap<(i64, i64), u32>,
    node: F,
    lon_scale: f64,
    params: &StrokeParams,
) -> Vec<Vec<i64>> {
    let angle_thresh = params.angle_threshold_deg;
    let ratio = params.flow_similarity_ratio;

    // out[u] = downstream neighbours; inc[v] = upstream neighbours (sorted).
    let mut out: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut inc: HashMap<i64, Vec<i64>> = HashMap::new();
    for &(u, v) in dir_counts.keys() {
        out.entry(u).or_default().push(v);
        inc.entry(v).or_default().push(u);
    }
    for vs in out.values_mut() {
        vs.sort_unstable();
        vs.dedup();
    }
    for us in inc.values_mut() {
        us.sort_unstable();
        us.dedup();
    }

    // Best downstream continuation of edge (u,v): unused (v,x), x≠u, within the
    // angle + flow gates; min angle then min x for determinism.
    let best_next = |u: i64, v: i64, used: &HashSet<(i64, i64)>| -> Option<i64> {
        let wuv = *edge_total.get(&(u, v)).unwrap_or(&0);
        let mut best: Option<(f64, i64)> = None;
        for &x in inc_or_out(&out, v) {
            if x == u || used.contains(&(v, x)) {
                continue;
            }
            let ang = deflection_angle(node(u), node(v), node(x), lon_scale);
            if ang > angle_thresh
                || !flow_similar(wuv, *edge_total.get(&(v, x)).unwrap_or(&0), ratio)
            {
                continue;
            }
            if best.map_or(true, |(ba, bx)| {
                ang < ba - 1e-9 || (ang <= ba + 1e-9 && x < bx)
            }) {
                best = Some((ang, x));
            }
        }
        best.map(|(_, x)| x)
    };

    // Best upstream extension of edge (u,v): unused (z,u), z≠v, where the turn
    // z→u→v is within the gates; min angle then min z.
    let best_prev = |u: i64, v: i64, used: &HashSet<(i64, i64)>| -> Option<i64> {
        let wuv = *edge_total.get(&(u, v)).unwrap_or(&0);
        let mut best: Option<(f64, i64)> = None;
        for &z in inc_or_out(&inc, u) {
            if z == v || used.contains(&(z, u)) {
                continue;
            }
            let ang = deflection_angle(node(z), node(u), node(v), lon_scale);
            if ang > angle_thresh
                || !flow_similar(wuv, *edge_total.get(&(z, u)).unwrap_or(&0), ratio)
            {
                continue;
            }
            if best.map_or(true, |(ba, bz)| {
                ang < ba - 1e-9 || (ang <= ba + 1e-9 && z < bz)
            }) {
                best = Some((ang, z));
            }
        }
        best.map(|(_, z)| z)
    };

    let mut edges: Vec<(i64, i64)> = dir_counts.keys().copied().collect();
    edges.sort_unstable();

    let mut used: HashSet<(i64, i64)> = HashSet::new();
    let mut strokes: Vec<Vec<i64>> = Vec::new();
    for &(a, b) in &edges {
        if used.contains(&(a, b)) {
            continue;
        }
        used.insert((a, b));

        // Downstream from (a,b): [b, x, …].
        let mut fwd: Vec<i64> = vec![b];
        let (mut u, mut v) = (a, b);
        while let Some(x) = best_next(u, v, &used) {
            used.insert((v, x));
            fwd.push(x);
            u = v;
            v = x;
        }
        // Upstream from (a,b): [z, …] collected reversed.
        let mut back: Vec<i64> = Vec::new();
        let (mut u, mut v) = (a, b);
        while let Some(z) = best_prev(u, v, &used) {
            used.insert((z, u));
            back.push(z);
            v = u;
            u = z;
        }
        back.reverse();

        // [upstream…, a, b, downstream…]
        let mut stroke = back;
        stroke.push(a);
        stroke.extend(fwd);
        strokes.push(stroke);
    }
    strokes
}

/// Neighbour slice for a node from an adjacency map, or an empty slice. (Helper
/// so the stroke closures can iterate without `Option`/empty-slice typing noise.)
fn inc_or_out(adj: &HashMap<i64, Vec<i64>>, k: i64) -> &[i64] {
    adj.get(&k).map(Vec::as_slice).unwrap_or(&[])
}

/// Stream a `--paths` intermediate GeoParquet back through the aggregator,
/// so flow bins can be re-tuned without re-routing every trip through OSRM.
pub fn aggregate_paths_parquet(path: &Path, agg: &mut FlowAggregator) -> Result<usize> {
    use arrow::array::{Array, BinaryArray, ListArray, TimestampMillisecondArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path)
        .with_context(|| format!("failed to open paths intermediate {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut rows = 0usize;
    for batch in reader {
        let batch = batch?;
        let geometry = batch
            .column_by_name("geometry")
            .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
            .ok_or_else(|| anyhow!("paths intermediate missing Binary 'geometry' column"))?;
        let timestamps = batch
            .column_by_name("timestamp")
            .and_then(|c| c.as_any().downcast_ref::<TimestampMillisecondArray>())
            .ok_or_else(|| anyhow!("paths intermediate missing 'timestamp' column"))?;
        let end_timestamps = batch
            .column_by_name("end_timestamp")
            .and_then(|c| c.as_any().downcast_ref::<TimestampMillisecondArray>())
            .ok_or_else(|| anyhow!("paths intermediate missing 'end_timestamp' column"))?;
        let vertex_times = batch
            .column_by_name("vertex_timestamps")
            .and_then(|c| c.as_any().downcast_ref::<ListArray>());

        for row in 0..batch.num_rows() {
            let coords = decode_wkb_linestring(geometry.value(row))
                .ok_or_else(|| anyhow!("row {}: invalid WKB linestring", rows + row))?;
            let start_ms = timestamps.value(row);
            let end_ms = if end_timestamps.is_valid(row) {
                end_timestamps.value(row)
            } else {
                start_ms
            };

            let vt: Option<Vec<i64>> = vertex_times.and_then(|vts| {
                if !vts.is_valid(row) {
                    return None;
                }
                let child = vts.value(row);
                let child = child.as_any().downcast_ref::<TimestampMillisecondArray>()?;
                Some((0..child.len()).map(|i| child.value(i)).collect())
            });

            agg.add_trip(&coords, vt.as_deref(), start_ms, end_ms);
        }
        rows += batch.num_rows();
    }
    Ok(rows)
}

/// Decode the WKB linestrings produced by `encode_wkb_linestring` (and any
/// standard 2D WKB linestring, either byte order).
fn decode_wkb_linestring(wkb: &[u8]) -> Option<Vec<[f64; 2]>> {
    if wkb.len() < 9 {
        return None;
    }
    let little = match wkb[0] {
        1 => true,
        0 => false,
        _ => return None,
    };
    let read_u32 = |b: &[u8]| -> u32 {
        let arr: [u8; 4] = b.try_into().unwrap();
        if little {
            u32::from_le_bytes(arr)
        } else {
            u32::from_be_bytes(arr)
        }
    };
    let read_f64 = |b: &[u8]| -> f64 {
        let arr: [u8; 8] = b.try_into().unwrap();
        if little {
            f64::from_le_bytes(arr)
        } else {
            f64::from_be_bytes(arr)
        }
    };
    if read_u32(&wkb[1..5]) != 2 {
        return None; // not a LineString
    }
    let n = read_u32(&wkb[5..9]) as usize;
    if wkb.len() < 9 + n * 16 {
        return None;
    }
    let mut coords = Vec::with_capacity(n);
    for i in 0..n {
        let off = 9 + i * 16;
        coords.push([
            read_f64(&wkb[off..off + 8]),
            read_f64(&wkb[off + 8..off + 16]),
        ]);
    }
    Some(coords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::osm_streets::{OsmNetwork, RoadClass, WayRec};

    /// Way 10 (primary): nodes 1-2-3 along lat 40.0 (two edges).
    fn net() -> OsmNetwork {
        let mut node_coords = HashMap::new();
        node_coords.insert(1, (-74.000, 40.000));
        node_coords.insert(2, (-73.999, 40.000));
        node_coords.insert(3, (-73.998, 40.000));
        let mut ways = HashMap::new();
        ways.insert(
            10,
            WayRec {
                refs: vec![1, 2, 3],
                class: RoadClass::Primary,
            },
        );
        OsmNetwork::build_indices(node_coords, ways)
    }

    #[test]
    fn parse_bins() {
        assert_eq!(parse_bin_ms("15m").unwrap(), 900_000);
        assert_eq!(parse_bin_ms("1h").unwrap(), 3_600_000);
        assert_eq!(parse_bin_ms("90s").unwrap(), 90_000);
        assert!(parse_bin_ms("15").is_err());
        assert!(parse_bin_ms("m").is_err());
        assert!(parse_bin_ms("0h").is_err());
    }

    #[test]
    fn add_trip_aggregates_onto_osm_edges() {
        let mut agg = FlowAggregator::new(900_000, net());
        // Two trips along nodes 1→2→3 in bin 0; one trip in a later bin.
        let coords = [[-74.000, 40.000], [-73.999, 40.000], [-73.998, 40.000]];
        agg.add_trip(&coords, Some(&[0, 10_000, 20_000]), 0, 20_000);
        agg.add_trip(&coords, Some(&[30_000, 40_000, 50_000]), 30_000, 50_000);
        agg.add_trip(
            &coords,
            Some(&[1_000_000, 1_010_000, 1_020_000]),
            1_000_000,
            1_020_000,
        );

        let (added, skipped, exact, fallback, miss) = agg.stats();
        assert_eq!((added, skipped, miss), (3, 0, 0));
        // 6 segments total, all exact endpoint matches.
        assert_eq!(exact + fallback, 6);
        // Two edges (1,2) and (2,3), each carrying two bins.
        assert_eq!(agg.counts.len(), 2);
        let mut vals: Vec<u32> = agg
            .counts
            .values()
            .flat_map(|m| m.values().copied())
            .collect();
        vals.sort_unstable();
        assert_eq!(vals, vec![1, 1, 2, 2]);
    }

    #[test]
    fn add_route_bins_distributes_weighted_counts() {
        let mut agg = FlowAggregator::new(3_600_000, net());
        // One OD corridor along nodes 1→2→3 serving bin 0 (×5 trips) and
        // bin 10 (×2 trips): each of the two edges gets the full per-bin weight.
        let coords = [[-74.000, 40.000], [-73.999, 40.000], [-73.998, 40.000]];
        let mut bins = HashMap::new();
        bins.insert(0i64, 5u32);
        bins.insert(10i64, 2u32);
        agg.add_route_bins(&coords, &bins);

        let (added, skipped, _, _, miss) = agg.stats();
        assert_eq!((added, skipped, miss), (1, 0, 0));
        assert_eq!(agg.counts.len(), 2); // edges (1,2) and (2,3)
        for edge in agg.counts.values() {
            assert_eq!(edge.get(&0).copied(), Some(5));
            assert_eq!(edge.get(&10).copied(), Some(2));
        }
        assert_eq!((agg.min_bin, agg.max_bin), (0, 10));

        // A second corridor over the same edges accumulates.
        agg.add_route_bins(&coords, &bins);
        for edge in agg.counts.values() {
            assert_eq!(edge.get(&0).copied(), Some(10));
        }

        // Degenerate inputs are skipped, not panicked on.
        agg.add_route_bins(&coords, &HashMap::new());
        agg.add_route_bins(&[[-74.0, 40.0]], &bins);
        assert_eq!(agg.stats().1, 2);
    }

    #[test]
    fn direction_is_normalized() {
        let mut agg = FlowAggregator::new(900_000, net());
        let fwd = [[-74.000, 40.000], [-73.999, 40.000]];
        let rev = [[-73.999, 40.000], [-74.000, 40.000]];
        agg.add_trip(&fwd, Some(&[0, 10_000]), 0, 10_000);
        agg.add_trip(&rev, Some(&[0, 10_000]), 0, 10_000);
        assert_eq!(agg.counts.len(), 1); // same undirected edge
        let edge = agg.counts.values().next().unwrap();
        assert_eq!(edge.values().copied().sum::<u32>(), 2);
    }

    #[test]
    fn segment_orientation_signs_by_travel() {
        let n = net();
        // Edge (1,2) canonical points node 1 (-74.000) → node 2 (-73.999) = +lon.
        assert_eq!(
            segment_orientation(&n, [-74.000, 40.0], [-73.999, 40.0], (1, 2)),
            1
        );
        assert_eq!(
            segment_orientation(&n, [-73.999, 40.0], [-74.000, 40.0], (1, 2)),
            -1
        );
    }

    #[test]
    fn directional_tracks_signed_net_and_is_opt_in() {
        // Off by default: the signed map stays empty, output is the plain heatmap.
        let mut plain = FlowAggregator::new(900_000, net());
        let fwd = [[-74.000, 40.000], [-73.999, 40.000]];
        plain.add_trip(&fwd, Some(&[0, 10_000]), 0, 10_000);
        assert!(plain.signed.is_empty());

        // On: two forward + one reverse over edge (1,2) → undirected count 3,
        // signed net along canonical (1→2) = +1 +1 -1 = +1.
        let mut agg = FlowAggregator::new(900_000, net());
        agg.set_directional(true);
        let rev = [[-73.999, 40.000], [-74.000, 40.000]];
        agg.add_trip(&fwd, Some(&[0, 10_000]), 0, 10_000);
        agg.add_trip(&fwd, Some(&[0, 10_000]), 0, 10_000);
        agg.add_trip(&rev, Some(&[0, 10_000]), 0, 10_000);
        assert_eq!(agg.counts.get(&(1, 2)).unwrap().values().sum::<u32>(), 3);
        assert_eq!(agg.signed.get(&(1, 2)).unwrap().values().sum::<i64>(), 1);
    }

    #[test]
    fn per_bucket_direction_implies_directional() {
        let mut agg = FlowAggregator::new(900_000, net());
        assert!(!agg.directional);
        agg.set_per_bucket_direction(true);
        assert!(agg.directional, "per-bucket direction needs the signed map");
        assert!(agg.per_bucket_direction);
    }

    #[test]
    fn per_bucket_direction_flips_matrix_sign_per_bucket() {
        use arrow::array::{Array, Float32Array, ListArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        // Line 1→2→3→4. Ride it FORWARD in bin 0, REVERSE in bin 2 (bin 1 idle).
        let mut agg = FlowAggregator::new(900_000, line_net());
        agg.set_per_bucket_direction(true);
        let fwd = [
            [-73.600, 45.500],
            [-73.590, 45.500],
            [-73.580, 45.500],
            [-73.570, 45.500],
        ];
        let rev = [
            [-73.570, 45.500],
            [-73.580, 45.500],
            [-73.590, 45.500],
            [-73.600, 45.500],
        ];
        agg.add_route_bins(&fwd, &HashMap::from([(0i64, 5u32)]));
        agg.add_route_bins(&rev, &HashMap::from([(2i64, 5u32)]));

        let tmp = std::env::temp_dir().join(format!("stt_pbd_{}.parquet", std::process::id()));
        let (features, buckets) = agg.write_parquet(&tmp).unwrap();
        assert_eq!((features, buckets), (1, 3)); // one corridor, bins 0..=2

        let file = std::fs::File::open(&tmp).unwrap();
        let batch = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let list = batch
            .column_by_name("vertex_value_matrix")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let m = list
            .value(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();
        // Vertex-major [v * nb + b], nb = 3. Vertex 0: bucket 0 (forward) vs
        // bucket 2 (reverse) must carry OPPOSITE signs, same magnitude (volume).
        assert!(
            m[0] > 0.0,
            "bucket 0 flows along winding → positive, got {}",
            m[0]
        );
        assert!(
            m[2] < 0.0,
            "bucket 2 flows against winding → negative, got {}",
            m[2]
        );
        assert_eq!(
            m[0].abs(),
            m[2].abs(),
            "|value| is the volume, sign is direction"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn chunk_net_flips_with_winding_order() {
        // Edge (1,2) carries net +5 along its canonical 1→2 orientation.
        let mut signed = HashMap::new();
        signed.insert((1, 2), HashMap::from([(0i64, 5i64)]));
        // Forward winding [1,2,3]: net reads +5.
        assert_eq!(
            chunk_net_along_winding(&signed, &[1, 2, 3], &[(1, 2), (2, 3)]),
            5
        );
        // Reversed winding [3,2,1]: the same edge is traversed high→low → -5.
        assert_eq!(
            chunk_net_along_winding(&signed, &[3, 2, 1], &[(2, 3), (1, 2)]),
            -5
        );
    }

    #[test]
    fn reverse_corridor_flips_geometry_and_matrix_rows() {
        let mut coords = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        // 3 vertices × 2 buckets, vertex-major: v0=[10,11], v1=[20,21], v2=[30,31].
        let mut matrix = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        reverse_corridor(&mut coords, &mut matrix, 3, 2);
        assert_eq!(coords, vec![[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]]);
        // The middle vertex row is unchanged; the ends swap, each carrying its row.
        assert_eq!(matrix, vec![30.0, 31.0, 20.0, 21.0, 10.0, 11.0]);
    }

    #[test]
    fn vertex_values_average_interior() {
        assert_eq!(
            vertex_values_from_segment_counts(&[4, 2, 2]),
            vec![4.0, 3.0, 2.0, 2.0]
        );
        assert_eq!(vertex_values_from_segment_counts(&[7]), vec![7.0, 7.0]);
    }

    #[test]
    fn wkb_roundtrip() {
        let coords = vec![[-73.98, 40.75], [-73.97, 40.76], [-73.96, 40.77]];
        let wkb = crate::common::encode_wkb_linestring(&coords);
        assert_eq!(decode_wkb_linestring(&wkb).unwrap(), coords);
    }

    // ---- directed flow + stroke synthesis (`--merged-paths`) ----

    /// A 4-node eastward line 1-2-3-4 (Montréal latitude) plus a branch node 5
    /// due north of 3 (a 90° turn). Geometry-only — stroke synthesis reads just
    /// `node_xy` + `lon_scale`.
    fn line_net() -> OsmNetwork {
        let mut nc = HashMap::new();
        nc.insert(1, (-73.600, 45.500));
        nc.insert(2, (-73.590, 45.500));
        nc.insert(3, (-73.580, 45.500));
        nc.insert(4, (-73.570, 45.500));
        nc.insert(5, (-73.580, 45.510));
        let mut ways = HashMap::new();
        ways.insert(
            10,
            WayRec {
                refs: vec![1, 2, 3, 4],
                class: RoadClass::Residential,
            },
        );
        ways.insert(
            20,
            WayRec {
                refs: vec![3, 5],
                class: RoadClass::Residential,
            },
        );
        OsmNetwork::build_indices(nc, ways)
    }

    /// Build a directed-count map with a single bin (bin 0) per directed edge.
    fn dir(edges: &[((i64, i64), u32)]) -> HashMap<(i64, i64), HashMap<i64, u32>> {
        edges
            .iter()
            .map(|&(e, w)| (e, HashMap::from([(0i64, w)])))
            .collect()
    }

    fn totals(dc: &HashMap<(i64, i64), HashMap<i64, u32>>) -> HashMap<(i64, i64), u32> {
        dc.iter()
            .map(|(e, b)| (*e, b.values().copied().sum::<u32>()))
            .collect()
    }

    #[test]
    fn directed_keeps_both_directions_separate() {
        let mut agg = FlowAggregator::new(3_600_000, net());
        let fwd = [[-74.000, 40.000], [-73.999, 40.000], [-73.998, 40.000]];
        let rev = [[-73.998, 40.000], [-73.999, 40.000], [-74.000, 40.000]];
        let bins = HashMap::from([(0i64, 3u32)]);
        agg.add_route_bins_directed(&fwd, &bins);
        agg.add_route_bins_directed(&rev, &bins);
        // Four distinct directed edges: (1,2),(2,3),(3,2),(2,1) — undirected
        // aggregation would have collapsed these to two.
        assert_eq!(agg.dir_counts.len(), 4);
        assert_eq!(agg.dir_counts[&(1, 2)].get(&0).copied(), Some(3));
        assert_eq!(agg.dir_counts[&(2, 1)].get(&0).copied(), Some(3));
    }

    #[test]
    fn deflection_straight_vs_turn() {
        let straight = deflection_angle((-73.60, 45.50), (-73.59, 45.50), (-73.58, 45.50), 0.70);
        assert!(straight < 1.0, "collinear → ~0°, got {straight}");
        let turn = deflection_angle((-73.60, 45.50), (-73.59, 45.50), (-73.59, 45.51), 0.70);
        assert!(
            (80.0..100.0).contains(&turn),
            "north turn → ~90°, got {turn}"
        );
    }

    #[test]
    fn stroke_contracts_degree2_chain() {
        let net = line_net();
        let dc = dir(&[((1, 2), 10), ((2, 3), 10), ((3, 4), 10)]);
        let strokes = synthesize_strokes(
            &dc,
            &totals(&dc),
            |id| net.node_xy(id).unwrap(),
            net.lon_scale(),
            &StrokeParams::default(),
        );
        assert_eq!(strokes, vec![vec![1, 2, 3, 4]]);
    }

    #[test]
    fn stroke_splits_on_flow_dropoff() {
        let net = line_net();
        // 1→2→3 heavy (100), 3→4 light (5): ratio 0.05 < 0.35 → split at node 3.
        let dc = dir(&[((1, 2), 100), ((2, 3), 100), ((3, 4), 5)]);
        let strokes = synthesize_strokes(
            &dc,
            &totals(&dc),
            |id| net.node_xy(id).unwrap(),
            net.lon_scale(),
            &StrokeParams::default(),
        );
        assert_eq!(strokes.len(), 2);
        assert!(strokes.contains(&vec![1, 2, 3]));
        assert!(strokes.contains(&vec![3, 4]));
    }

    #[test]
    fn stroke_stops_at_sharp_turn() {
        let net = line_net();
        // Equal flow, but the 90° turn 3→5 exceeds the 50° default cap.
        let dc = dir(&[((1, 2), 10), ((2, 3), 10), ((3, 5), 10)]);
        let strokes = synthesize_strokes(
            &dc,
            &totals(&dc),
            |id| net.node_xy(id).unwrap(),
            net.lon_scale(),
            &StrokeParams::default(),
        );
        assert!(strokes.contains(&vec![1, 2, 3]));
        assert!(strokes.contains(&vec![3, 5]));
    }

    #[test]
    fn strokes_are_deterministic() {
        let net = line_net();
        let dc = dir(&[((1, 2), 10), ((2, 3), 10), ((3, 4), 10), ((3, 5), 10)]);
        let a = synthesize_strokes(
            &dc,
            &totals(&dc),
            |id| net.node_xy(id).unwrap(),
            net.lon_scale(),
            &StrokeParams::default(),
        );
        let b = synthesize_strokes(
            &dc,
            &totals(&dc),
            |id| net.node_xy(id).unwrap(),
            net.lon_scale(),
            &StrokeParams::default(),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn write_strokes_matrix_is_vertex_major() {
        let mut agg = FlowAggregator::new(3_600_000, line_net());
        // Route 1→2→3→4 in bin 0 (×8 trips).
        let coords = [
            [-73.600, 45.500],
            [-73.590, 45.500],
            [-73.580, 45.500],
            [-73.570, 45.500],
        ];
        agg.add_route_bins_directed(&coords, &HashMap::from([(0i64, 8u32)]));
        let tmp = std::env::temp_dir().join(format!("stt_strokes_{}.parquet", std::process::id()));
        let (features, buckets) = agg
            .write_parquet_strokes(
                &tmp,
                &StrokeParams {
                    min_stroke_trips: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((features, buckets), (1, 1)); // one corridor, one bucket
        let _ = std::fs::remove_file(&tmp);
    }
}
