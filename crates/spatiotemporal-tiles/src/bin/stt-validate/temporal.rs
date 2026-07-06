//! Decode-tier temporal invariants: per-feature interval sanity and the
//! directory `time_end` tightness contract.
//!
//! Two invariants only a payload decode can verify:
//!
//! - **Interval sanity** — every feature's `end_time >= start_time`. A
//!   reversed pair is producer garbage: readers treat `[start, end]` as an
//!   inclusive validity window, so the feature is either unfindable or
//!   permanently visible depending on the reader's clamp. The validator
//!   aggregates a count plus the first offender rather than spamming one
//!   error per feature.
//! - **`time_end` tightness** (review T1.5) — a directory entry's `time_end`
//!   must equal the maximum feature `end_time` across the tile's layers.
//!   Interval features live only in their START bucket and are findable at
//!   later timestamps *only because* the writer widens `time_end` to the max
//!   feature end; a writer emitting nominal bucket ends passes every other
//!   check yet silently breaks every interval query. The caller compares
//!   [`max_feature_end`] against the entry's `time_end`.

use arrow::array::{Array, Int64Array};
use stt_core::arrow_tile::DecodedLayer;

/// First feature whose `end_time < start_time`, for a concrete error message.
pub struct IntervalOffender {
    pub layer: String,
    pub feature_index: usize,
    pub start_time: i64,
    pub end_time: i64,
}

/// Count features with `end_time < start_time` across all layers, and return
/// the first offender. Missing or mistyped `start_time`/`end_time` columns
/// are skipped here — the schema check already reports those.
pub fn check_interval_sanity(layers: &[DecodedLayer]) -> (u64, Option<IntervalOffender>) {
    let mut count = 0u64;
    let mut first: Option<IntervalOffender> = None;
    for layer in layers {
        let (Some(start), Some(end)) = (
            int64_column(layer, "start_time"),
            int64_column(layer, "end_time"),
        ) else {
            continue;
        };
        for i in 0..start.len().min(end.len()) {
            if start.is_null(i) || end.is_null(i) {
                continue;
            }
            if end.value(i) < start.value(i) {
                count += 1;
                if first.is_none() {
                    first = Some(IntervalOffender {
                        layer: layer.name.clone(),
                        feature_index: i,
                        start_time: start.value(i),
                        end_time: end.value(i),
                    });
                }
            }
        }
    }
    (count, first)
}

/// Maximum feature `end_time` across all layers; `None` when no feature
/// carries one (a featureless tile has no tightness contract to check).
pub fn max_feature_end(layers: &[DecodedLayer]) -> Option<i64> {
    let mut max: Option<i64> = None;
    for layer in layers {
        let Some(end) = int64_column(layer, "end_time") else {
            continue;
        };
        for i in 0..end.len() {
            if end.is_null(i) {
                continue;
            }
            let v = end.value(i);
            if max.is_none_or(|m| v > m) {
                max = Some(v);
            }
        }
    }
    max
}

fn int64_column<'a>(layer: &'a DecodedLayer, name: &str) -> Option<&'a Int64Array> {
    layer
        .batch
        .column_by_name(name)?
        .as_any()
        .downcast_ref::<Int64Array>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::arrow_tile::{decode_tile, encode_tile, ColumnarLayer, GeometryColumn};

    fn layer(name: &str, starts: Vec<i64>, ends: Vec<i64>) -> ColumnarLayer {
        let n = starts.len();
        ColumnarLayer {
            name: name.into(),
            feature_ids: (1..=n as u64).collect(),
            start_times: starts,
            end_times: ends,
            geometry: GeometryColumn::Point(vec![[0.0, 0.0]; n]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        }
    }

    fn decode(layers: &[ColumnarLayer]) -> Vec<DecodedLayer> {
        decode_tile(&encode_tile(layers).unwrap()).unwrap()
    }

    #[test]
    fn well_ordered_intervals_pass() {
        let decoded = decode(&[layer("a", vec![0, 5, 5], vec![0, 5, 9])]);
        let (count, first) = check_interval_sanity(&decoded);
        assert_eq!(count, 0);
        assert!(first.is_none());
    }

    #[test]
    fn interval_violations_are_counted_with_first_offender() {
        let decoded = decode(&[
            layer("a", vec![10, 10, 20], vec![15, 5, 25]),
            layer("b", vec![0, 30], vec![0, 7]),
        ]);
        let (count, first) = check_interval_sanity(&decoded);
        assert_eq!(count, 2);
        let first = first.unwrap();
        assert_eq!(first.layer, "a");
        assert_eq!(first.feature_index, 1);
        assert_eq!((first.start_time, first.end_time), (10, 5));
    }

    #[test]
    fn max_feature_end_spans_layers() {
        let decoded = decode(&[
            layer("a", vec![0, 5], vec![10, 99]),
            layer("b", vec![0], vec![42]),
        ]);
        assert_eq!(max_feature_end(&decoded), Some(99));
        assert_eq!(max_feature_end(&[]), None);
    }
}
