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
const MAX_CHAIN_SEGMENTS: usize = 64;

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
            min_bin: i64::MAX,
            max_bin: i64::MIN,
            trips_added: 0,
            trips_skipped: 0,
            matched_exact: 0,
            matched_fallback: 0,
            unmatched: 0,
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
            let edge_counts = self.counts.entry(edge).or_default();
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
        if self.counts.is_empty() {
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
                    features += self.emit_run(&run, way.class.min_zoom(), bucket0, range_end,
                        min_bin, num_buckets, &mut writer)?;
                    run.clear();
                } else {
                    run.clear();
                }
            }
            if run.len() >= 2 {
                features += self.emit_run(&run, way.class.min_zoom(), bucket0, range_end,
                    min_bin, num_buckets, &mut writer)?;
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

            let coords: Vec<[f64; 2]> = chunk
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
                        let c = self.counts.get(e).and_then(|m| m.get(&bin)).copied().unwrap_or(0);
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

/// Per-vertex values from per-segment counts: endpoints take their single
/// adjacent segment's count, interior vertices the mean of both sides.
fn vertex_values_from_segment_counts(counts: &[u32]) -> Vec<f32> {
    let n = counts.len();
    let mut values = Vec::with_capacity(n + 1);
    values.push(counts[0] as f32);
    for i in 1..n {
        values.push((counts[i - 1] + counts[i]) as f32 / 2.0);
    }
    values.push(counts[n - 1] as f32);
    values
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
        coords.push([read_f64(&wkb[off..off + 8]), read_f64(&wkb[off + 8..off + 16])]);
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
        ways.insert(10, WayRec { refs: vec![1, 2, 3], class: RoadClass::Primary });
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
        agg.add_trip(&coords, Some(&[1_000_000, 1_010_000, 1_020_000]), 1_000_000, 1_020_000);

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
}
