//! Time-binned road-segment flow aggregation for NYC taxi trips (`--flows`).
//!
//! Builds the *overview* counterpart to `--paths`: instead of one feature per
//! trip, emit one feature per (road corridor, time bin) whose per-vertex
//! `vertex_values` carry the number of trips that traversed each segment
//! during that bin. Because OSRM snaps every trip to the same road network
//! (GeoJSON output, 1e-6° precision), consecutive-vertex pairs from different
//! trips are byte-identical on shared streets — so segment identity is exact
//! coordinate quantization, no map-matching needed.
//!
//! Encoding choices (consumed by `AnimatedTripsLayer` via `tripGradient`):
//! - `timestamp` = bin start, `end_timestamp` = bin end
//! - `vertex_timestamps` all = bin start: the whole corridor lights up on the
//!   bin boundary; the renderer's trail fade cross-fades into the next bin.
//!   (Safe through stt-build: temporal splitting guards `ct > pt`, and the
//!   rideshare build path runs without `--simplify`, so supplied times pass
//!   through by length match.)
//! - `vertex_values` = per-vertex traversal count (endpoints take the single
//!   adjacent segment's count, interior vertices the mean of both sides), so
//!   color varies along a corridor without breaking it into per-segment rows.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::common::{LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter};

/// Lon/lat quantized to 1e-6° (~0.1 m) — exact for OSRM GeoJSON output.
type QPoint = (i32, i32);
/// Undirected segment key: endpoints in lexicographic order.
type SegKey = (QPoint, QPoint);

/// Cap corridor chains so a single feature can't span a borough — very long
/// features land in (and bloat) every tile they cross at every zoom.
const MAX_CHAIN_SEGMENTS: usize = 64;

fn quantize(lon: f64, lat: f64) -> QPoint {
    ((lon * 1e6).round() as i32, (lat * 1e6).round() as i32)
}

fn dequantize(p: QPoint) -> [f64; 2] {
    [p.0 as f64 / 1e6, p.1 as f64 / 1e6]
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

/// Accumulates (time bin, road segment) → traversal count across trips.
pub struct FlowAggregator {
    bin_ms: i64,
    counts: HashMap<(i64, SegKey), u32>,
    trips_added: usize,
    trips_skipped: usize,
}

impl FlowAggregator {
    pub fn new(bin_ms: i64) -> Self {
        Self {
            bin_ms,
            counts: HashMap::new(),
            trips_added: 0,
            trips_skipped: 0,
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
            let a = quantize(coords[i - 1][0], coords[i - 1][1]);
            let b = quantize(coords[i][0], coords[i][1]);
            if a == b {
                continue; // zero-length after quantization
            }
            let key = if a <= b { (a, b) } else { (b, a) };
            let mid = (times[i - 1] + times[i]) / 2;
            let bin = mid.div_euclid(self.bin_ms);
            *self.counts.entry((bin, key)).or_insert(0) += 1;
        }
        self.trips_added += 1;
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        (self.trips_added, self.trips_skipped, self.counts.len())
    }

    /// Chain segments into corridors per bin and stream rows to `output`.
    /// Returns (features written, number of bins).
    pub fn write_parquet(self, output: &Path) -> Result<(usize, usize)> {
        // Group by bin.
        let mut bins: HashMap<i64, Vec<(SegKey, u32)>> = HashMap::new();
        for ((bin, key), count) in self.counts {
            bins.entry(bin).or_default().push((key, count));
        }
        let mut bin_ids: Vec<i64> = bins.keys().copied().collect();
        bin_ids.sort_unstable();
        let num_bins = bin_ids.len();

        let property_columns = vec![PropertyColumn::numeric("max_count")];
        let mut writer = StreamingLineStringParquetWriter::with_columns(output, property_columns)?;

        let mut features = 0usize;
        for bin in bin_ids {
            let segs = &bins[&bin];
            let bin_start = bin * self.bin_ms;
            let bin_end = bin_start + self.bin_ms;

            for (nodes, seg_counts) in chain_segments(segs, MAX_CHAIN_SEGMENTS) {
                let coords: Vec<[f64; 2]> = nodes.iter().map(|&p| dequantize(p)).collect();
                let values = vertex_values_from_segment_counts(&seg_counts);
                let max_count = seg_counts.iter().copied().max().unwrap_or(0);

                let mut properties = Map::new();
                properties.insert("max_count".to_string(), json!(max_count));

                let n = coords.len();
                let record = LineStringRecord {
                    coordinates: coords,
                    timestamp_ms: bin_start,
                    end_timestamp_ms: Some(bin_end),
                    vertex_timestamps_ms: Some(vec![bin_start; n]),
                    vertex_values: Some(values),
                    properties,
                };
                writer.write_linestring(&record)?;
                features += 1;
            }
        }
        writer.finish()?;
        Ok((features, num_bins))
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

/// Greedy path cover: every segment ends up in exactly one chain; chains
/// extend through nodes as long as any unused adjacent segment exists.
/// Returns (node sequence, per-segment counts) per chain.
fn chain_segments(segs: &[(SegKey, u32)], max_chain: usize) -> Vec<(Vec<QPoint>, Vec<u32>)> {
    let mut adj: HashMap<QPoint, Vec<usize>> = HashMap::new();
    for (i, ((a, b), _)) in segs.iter().enumerate() {
        adj.entry(*a).or_default().push(i);
        adj.entry(*b).or_default().push(i);
    }

    let mut used = vec![false; segs.len()];
    let mut chains = Vec::new();

    for seed in 0..segs.len() {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let ((a, b), count) = segs[seed];
        let mut nodes: std::collections::VecDeque<QPoint> = [a, b].into_iter().collect();
        let mut counts: std::collections::VecDeque<u32> = [count].into_iter().collect();

        // Extend at the back, then at the front.
        for forward in [true, false] {
            while counts.len() < max_chain {
                let tip = if forward {
                    *nodes.back().unwrap()
                } else {
                    *nodes.front().unwrap()
                };
                let next = adj
                    .get(&tip)
                    .and_then(|cands| cands.iter().copied().find(|&i| !used[i]));
                let Some(i) = next else { break };
                used[i] = true;
                let ((a, b), c) = segs[i];
                let other = if a == tip { b } else { a };
                if forward {
                    nodes.push_back(other);
                    counts.push_back(c);
                } else {
                    nodes.push_front(other);
                    counts.push_front(c);
                }
            }
        }
        chains.push((nodes.into_iter().collect(), counts.into_iter().collect()));
    }
    chains
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
    fn shared_segments_accumulate() {
        let mut agg = FlowAggregator::new(900_000);
        // Two trips traverse the same two segments within the same bin;
        // a third traverses them in the next bin.
        let coords = [[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]];
        let t0 = [0i64, 10_000, 20_000];
        let t1 = [30_000i64, 40_000, 50_000];
        let t2 = [1_000_000i64, 1_010_000, 1_020_000];
        agg.add_trip(&coords, Some(&t0), 0, 20_000);
        agg.add_trip(&coords, Some(&t1), 30_000, 50_000);
        agg.add_trip(&coords, Some(&t2), 1_000_000, 1_020_000);

        let (added, skipped, entries) = agg.stats();
        assert_eq!(added, 3);
        assert_eq!(skipped, 0);
        // 2 segments × 2 bins
        assert_eq!(entries, 4);
        let c: Vec<u32> = {
            let mut v: Vec<u32> = agg.counts.values().copied().collect();
            v.sort_unstable();
            v
        };
        assert_eq!(c, vec![1, 1, 2, 2]);
    }

    #[test]
    fn direction_is_normalized() {
        let mut agg = FlowAggregator::new(900_000);
        let fwd = [[0.0, 0.0], [0.001, 0.0]];
        let rev = [[0.001, 0.0], [0.0, 0.0]];
        agg.add_trip(&fwd, Some(&[0, 10_000]), 0, 10_000);
        agg.add_trip(&rev, Some(&[0, 10_000]), 0, 10_000);
        assert_eq!(agg.counts.len(), 1);
        assert_eq!(*agg.counts.values().next().unwrap(), 2);
    }

    #[test]
    fn chains_merge_collinear_segments() {
        // Three segments in a line: a-b, b-c, c-d → one chain of 4 nodes.
        let a = quantize(0.0, 0.0);
        let b = quantize(0.001, 0.0);
        let c = quantize(0.002, 0.0);
        let d = quantize(0.003, 0.0);
        let segs = vec![((a, b), 4u32), ((b, c), 2u32), ((c, d), 2u32)];
        let chains = chain_segments(&segs, 64);
        assert_eq!(chains.len(), 1);
        let (nodes, counts) = &chains[0];
        assert_eq!(nodes.len(), 4);
        assert_eq!(counts.len(), 3);
        // Every segment appears exactly once with its count preserved.
        let mut sorted = counts.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![2, 2, 4]);
    }

    #[test]
    fn chain_cap_respected() {
        // A long line of 10 segments with max_chain 4 → ceil(10/4) = 3 chains.
        let nodes: Vec<QPoint> = (0..11).map(|i| quantize(0.001 * i as f64, 0.0)).collect();
        let segs: Vec<(SegKey, u32)> = (0..10)
            .map(|i| {
                let (a, b) = (nodes[i], nodes[i + 1]);
                (if a <= b { (a, b) } else { (b, a) }, 1u32)
            })
            .collect();
        let chains = chain_segments(&segs, 4);
        let total: usize = chains.iter().map(|(_, c)| c.len()).sum();
        assert_eq!(total, 10);
        assert!(chains.iter().all(|(_, c)| c.len() <= 4));
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
