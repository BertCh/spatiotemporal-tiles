//! Shared fixtures for the `stt-core` integration-test suite.
//!
//! One home for the tile-layer and directory-entry fixtures that were
//! previously rebuilt, field-for-field, in several `tests/*.rs` files:
//!   * [`point_fixture`] / [`line_fixture`] — canonical single-layer
//!     `ColumnarLayer`s (a points layer with numeric + categorical properties
//!     and out-of-order start times; a timed-linestring layer). Consumed by
//!     `reproducible_build.rs` (determinism/time-offset guards) and
//!     `spec_conformance.rs` (per-layer schema locks).
//!   * [`arb_entry`] / [`arb_pageable_entry`] — proptest strategies for the v5
//!     directory `TileEntry` (fully arbitrary vs. Web-Mercator-addressable).
//!   * [`tile_payload`] — a minimal valid single-layer tile encoded from a set
//!     of feature ids, for the never-panic decoder fuzzers.
//!
//! Each integration-test file is its own crate, so a file uses only the subset
//! it needs; `#![allow(dead_code)]` keeps the unused remainder quiet (matching
//! the model in `crates/stt-build/tests/common/mod.rs`).

#![allow(dead_code)]

use proptest::prelude::*;
use stt_core::arrow_tile::{encode_tile, ColumnarLayer, GeometryColumn, PropertyColumn};
use stt_core::directory::TileEntry;

/// A three-point layer with a numeric (`speed`, with a null) and a categorical
/// (`kind`, with a null) property. The start times are deliberately
/// out-of-order so the baked `stt:time_offset_ms` is the real minimum (1000),
/// not simply the first value (3000).
pub fn point_fixture() -> ColumnarLayer {
    ColumnarLayer {
        name: "points".to_string(),
        feature_ids: vec![1, 2, 3],
        start_times: vec![3000, 1000, 2000],
        end_times: vec![3500, 1500, 2500],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8], [-122.6, 37.9]]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![
            (
                "speed".to_string(),
                PropertyColumn::Numeric(vec![Some(10.0), None, Some(30.0)]),
            ),
            (
                "kind".to_string(),
                PropertyColumn::Categorical(vec![
                    Some("car".to_string()),
                    Some("bus".to_string()),
                    None,
                ]),
            ),
        ],
    }
}

/// A two-feature timed-linestring layer. The tight per-vertex time span
/// (`vertex_times`) drives the u16-delta vertex-time encoding path.
pub fn line_fixture() -> ColumnarLayer {
    ColumnarLayer {
        name: "tracks".to_string(),
        feature_ids: vec![10, 11],
        start_times: vec![100, 0],
        end_times: vec![200, 50],
        geometry: GeometryColumn::LineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            vec![[5.0, 5.0], [6.0, 6.0]],
        ]),
        vertex_times: Some(vec![vec![0, 25, 50], vec![100, 200]]),
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}

/// Fully arbitrary directory entry (every field spans its whole type). The
/// cover bound is set by the caller — the codec's cover section is
/// all-or-nothing, so the two representable shapes are generated explicitly.
pub fn arb_entry() -> impl Strategy<Value = TileEntry> {
    (
        (any::<u8>(), any::<u32>(), any::<u32>(), any::<u64>()),
        (any::<i64>(), any::<i64>(), any::<u32>(), any::<u64>()),
        (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()),
        proptest::option::of(any::<u64>()),
    )
        .prop_map(
            |(
                (zoom, x, y, hilbert),
                (time_start, time_end, pack_id, offset),
                (length, uncompressed_size, feature_count, crc32c),
                temporal_bucket_ms,
            )| TileEntry {
                zoom,
                x,
                y,
                time_start,
                time_end,
                pack_id,
                offset,
                length,
                uncompressed_size,
                feature_count,
                hilbert,
                crc32c,
                temporal_bucket_ms,
                cover_t_min: None,
            },
        )
}

/// Entry constrained to coordinates the paged *encoder* accepts: its page
/// descriptors call `tile_geo_bounds(zoom, x, y)`, which needs a real Web
/// Mercator address (zoom < 32, x/y within the zoom's grid). The paged
/// *decoder* under test still sees fully adversarial bytes via mutation.
pub fn arb_pageable_entry() -> impl Strategy<Value = TileEntry> {
    (arb_entry(), 0u8..=20, any::<u32>(), any::<u32>()).prop_map(|(mut e, zoom, x, y)| {
        let n = 1u32 << zoom;
        e.zoom = zoom;
        e.x = x % n;
        e.y = y % n;
        e
    })
}

/// A tiny valid single-layer tile payload seeded from arbitrary feature ids.
pub fn tile_payload(ids: Vec<u64>) -> Vec<u8> {
    let n = ids.len();
    encode_tile(std::slice::from_ref(&fuzz_layer(ids, n))).unwrap()
}

/// [`tile_payload`]'s layer, encoded as a SELF-CONTAINED **v2** frame
/// (formatVersion 2, inline schema sections — decodable without a template
/// registry), for the v2 never-panic decoder fuzzers.
pub fn tile_payload_v2(ids: Vec<u64>) -> Vec<u8> {
    let n = ids.len();
    let cfg = stt_core::arrow_tile::EncoderConfig {
        format_version: stt_core::arrow_tile::FORMAT_VERSION_V2,
        ..stt_core::arrow_tile::EncoderConfig::default()
    };
    stt_core::arrow_tile::encode_tile_with(std::slice::from_ref(&fuzz_layer(ids, n)), &cfg).unwrap()
}

fn fuzz_layer(ids: Vec<u64>, n: usize) -> ColumnarLayer {
    ColumnarLayer {
        name: "default".to_string(),
        feature_ids: ids,
        start_times: vec![0; n],
        end_times: vec![1; n],
        geometry: GeometryColumn::Point(vec![[0.0, 0.0]; n]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    }
}
