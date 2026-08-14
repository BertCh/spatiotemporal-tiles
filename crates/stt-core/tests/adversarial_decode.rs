//! Adversarial property tests for the STT decoders: the v6 directory codec,
//! the paged directory container, and the tile layer frame.
//!
//! Two property families:
//! 1. **round-trip** — arbitrary VALID entry lists encode → decode → equal, so
//!    the codec loses nothing over its whole input domain (not just the
//!    hand-picked fixtures in the unit tests);
//! 2. **never-panic** — arbitrary byte soup AND random mutations/truncations
//!    of valid encodings must decode to `Ok` or `Err`, never panic (and never
//!    let a doctored header force an attacker-sized allocation).
//!
//! Case counts are kept modest (256) so CI stays fast; crank `PROPTEST_CASES`
//! locally for a deeper run.

use arrow::array::{
    Array, DictionaryArray, FixedSizeListArray, Float64Array, Int32Array, Int64Array, ListArray,
    StringArray, UInt16Array, UInt64Array,
};
use arrow::datatypes::{DataType, UInt16Type};
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use stt_core::arrow_tile::{
    build_quantized_numeric_for_column, build_quantized_numeric_pinned, decode_tile, encode_tile,
    encode_tile_with, AttrPinned, AttrQuant, ColumnarLayer, EncoderConfig, GeometryColumn,
    GlobalColumnPins, PropertyColumn, STT_QUANT_ATTR_META_KEY,
};
use stt_core::directory::{decode_directory, encode_directory};
use stt_core::directory_page::{decode_paged_directory, encode_paged_directory};
use stt_core::metadata::{check_fingerprint, ContentFingerprint, FingerprintAccumulator};

mod common;
use common::{arb_entry, arb_pageable_entry, tile_payload, tile_payload_v2};

// ----------------------------------------------------------------------
// Tile-encoder round-trip: arbitrary layers encode → decode → value-equal.
//
// The existing blocks above prove the decoders never PANIC on hostile input;
// this block proves the encode↔decode pair is LOSSLESS over its whole valid
// input domain — the counterpart to the fixture-only unit tests in
// `arrow_tile.rs`. Only the default (unquantized) `encode_tile` path is
// exercised, so geometry/attributes reconstruct exactly; the value domain is
// bounded away from the two non-`==`-comparable / overflow-prone corners:
//   * no NaN (NaN != NaN would make an exact round-trip un-assertable — that is
//     a comparison artefact, not an encoder property), and
//   * times in the realistic Unix-ms band (the encoder bakes the MIN start time
//     as an offset; an unbounded i64 domain could overflow that subtraction).
// Every property column is forced to hold at least one non-null so it survives
// the encoder's all-null-column drop and stays addressable by name.
// ----------------------------------------------------------------------

/// A small categorical alphabet with intentional repeats (so the dictionary
/// actually deduplicates) and a couple of multi-char values.
fn arb_kind() -> impl Strategy<Value = String> {
    proptest::sample::select(vec!["a", "b", "c", "car", "bus"]).prop_map(String::from)
}

/// A well-formed single-layer Point tile with arbitrary ids/times/coordinates
/// and nullable numeric + categorical properties.
fn arb_point_layer() -> impl Strategy<Value = ColumnarLayer> {
    (1usize..=8).prop_flat_map(|n| {
        (
            proptest::collection::vec(any::<u64>(), n),
            proptest::collection::vec(0i64..=1_000_000_000_000, n),
            proptest::collection::vec(0i64..=1_000_000, n),
            proptest::collection::vec((-180.0f64..180.0, -85.0f64..85.0), n),
            proptest::collection::vec(proptest::option::of(-1.0e6f64..1.0e6), n),
            proptest::collection::vec(proptest::option::of(arb_kind()), n),
        )
            .prop_map(|(ids, starts, durs, coords, mut speeds, mut kinds)| {
                let end_times: Vec<i64> = starts.iter().zip(&durs).map(|(s, d)| s + d).collect();
                let points: Vec<[f64; 2]> = coords.iter().map(|(x, y)| [*x, *y]).collect();
                if speeds.iter().all(Option::is_none) {
                    speeds[0] = Some(0.0);
                }
                if kinds.iter().all(Option::is_none) {
                    kinds[0] = Some("a".to_string());
                }
                ColumnarLayer {
                    polygon_parts: None,
                    name: "points".to_string(),
                    feature_ids: ids,
                    start_times: starts,
                    end_times,
                    geometry: GeometryColumn::Point(points),
                    vertex_times: None,
                    vertex_values: None,
                    triangles: None,
                    vertex_value_matrix: None,
                    properties: vec![
                        ("speed".to_string(), PropertyColumn::Numeric(speeds)),
                        ("kind".to_string(), PropertyColumn::Categorical(kinds)),
                    ],
                }
            })
    })
}

/// A well-formed single-layer LineString tile: arbitrary ids/times and, per
/// feature, 2..=8 arbitrary vertices (no per-vertex time/value arrays, so the
/// geometry reconstructs exactly rather than through the lossy u16-delta
/// vertex-time path).
fn arb_linestring_layer() -> impl Strategy<Value = ColumnarLayer> {
    (1usize..=6).prop_flat_map(|n| {
        (
            proptest::collection::vec(any::<u64>(), n),
            proptest::collection::vec(0i64..=1_000_000_000_000, n),
            proptest::collection::vec(0i64..=1_000_000, n),
            proptest::collection::vec(
                proptest::collection::vec((-180.0f64..180.0, -85.0f64..85.0), 2..=8),
                n,
            ),
        )
            .prop_map(|(ids, starts, durs, lines_raw)| {
                let end_times: Vec<i64> = starts.iter().zip(&durs).map(|(s, d)| s + d).collect();
                let lines: Vec<Vec<[f64; 2]>> = lines_raw
                    .iter()
                    .map(|verts| verts.iter().map(|(x, y)| [*x, *y]).collect())
                    .collect();
                ColumnarLayer {
                    polygon_parts: None,
                    name: "tracks".to_string(),
                    feature_ids: ids,
                    start_times: starts,
                    end_times,
                    geometry: GeometryColumn::LineString(lines),
                    vertex_times: None,
                    vertex_values: None,
                    triangles: None,
                    vertex_value_matrix: None,
                    properties: vec![],
                }
            })
    })
}

/// The encoder canonicalizes row order: rows are sorted by `start_time` with a
/// STABLE sort (`sort_rows_by_start_time`). A round-trip therefore preserves
/// every row's fields TOGETHER, but not the input's row order — so the expected
/// side is permuted the same way before comparing positionally. Comparing
/// per-row (rather than as a multiset) is what keeps the assertions strong:
/// it catches a permutation that desynchronizes columns from one another.
fn canonical_order(start_times: &[i64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..start_times.len()).collect();
    idx.sort_by_key(|&i| start_times[i]); // stable — mirrors the encoder
    idx
}

/// Permute a slice into the encoder's canonical row order.
fn permute<T: Clone>(v: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|&i| v[i].clone()).collect()
}

proptest! {
    // Encode+decode of a full tile is heavier than the codec fuzzers above, so
    // a smaller case count keeps CI fast while still covering the domain.
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Point layers survive encode→decode with exact value-equality of ids,
    /// start/end times, coordinates, and BOTH property kinds (numeric with
    /// nulls, categorical with nulls).
    #[test]
    fn point_layer_roundtrips_value_equal(layer in arb_point_layer()) {
        let payload = encode_tile(std::slice::from_ref(&layer)).unwrap();
        let mut decoded = decode_tile(&payload).unwrap();
        prop_assert_eq!(decoded.len(), 1);
        let batch = decoded.pop().unwrap().batch;
        let n = layer.feature_ids.len();
        prop_assert_eq!(batch.num_rows(), n);
        let order = canonical_order(&layer.start_times);

        // ids (UInt64) — exact.
        let ids = batch.column_by_name("id").unwrap()
            .as_any().downcast_ref::<UInt64Array>().unwrap();
        prop_assert_eq!(ids.values().to_vec(), permute(&layer.feature_ids, &order));

        // start/end times (Int64, stored ABSOLUTE — the baked offset is only a
        // decoder hint) — exact.
        let starts = batch.column_by_name("start_time").unwrap()
            .as_any().downcast_ref::<Int64Array>().unwrap();
        prop_assert_eq!(starts.values().to_vec(), permute(&layer.start_times, &order));
        let ends = batch.column_by_name("end_time").unwrap()
            .as_any().downcast_ref::<Int64Array>().unwrap();
        prop_assert_eq!(ends.values().to_vec(), permute(&layer.end_times, &order));

        // geometry: FixedSizeList<Float64,2> — exact per coordinate.
        let geom = batch.column_by_name("geometry").unwrap()
            .as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let GeometryColumn::Point(points) = &layer.geometry else { unreachable!() };
        let points = permute(points, &order);
        for (i, [x, y]) in points.iter().enumerate() {
            let pt = geom.value(i);
            let pt = pt.as_any().downcast_ref::<Float64Array>().unwrap();
            prop_assert_eq!(pt.value(0), *x);
            prop_assert_eq!(pt.value(1), *y);
        }

        // numeric property "speed": Float64 nullable — exact (no NaN generated).
        let speed = batch.column_by_name("speed").unwrap()
            .as_any().downcast_ref::<Float64Array>().unwrap();
        let PropertyColumn::Numeric(want) = &layer.properties[0].1 else { unreachable!() };
        let want = permute(want, &order);
        for (i, w) in want.iter().enumerate() {
            match w {
                None => prop_assert!(speed.is_null(i), "row {} expected null speed", i),
                Some(v) => {
                    prop_assert!(!speed.is_null(i), "row {} unexpectedly null", i);
                    prop_assert_eq!(speed.value(i), *v);
                }
            }
        }

        // Categorical property "kind": adaptively Dictionary<UInt16,Utf8> or
        // plain Utf8, with exact strings in either representation.
        let kind = batch.column_by_name("kind").unwrap();
        let dictionary = kind
            .as_any()
            .downcast_ref::<DictionaryArray<UInt16Type>>();
        let plain = kind.as_any().downcast_ref::<StringArray>();
        prop_assert!(
            dictionary.is_some() || plain.is_some(),
            "unexpected categorical Arrow type {:?}",
            kind.data_type()
        );
        let PropertyColumn::Categorical(kw) = &layer.properties[1].1 else { unreachable!() };
        let kw = permute(kw, &order);
        for (i, w) in kw.iter().enumerate() {
            match w {
                None => prop_assert!(kind.is_null(i), "row {} expected null kind", i),
                Some(s) => {
                    prop_assert!(!kind.is_null(i), "row {} unexpectedly null", i);
                    let got = if let Some(dictionary) = dictionary {
                        let dict_vals = dictionary
                            .values()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        dict_vals.value(dictionary.keys().value(i) as usize)
                    } else {
                        plain.unwrap().value(i)
                    };
                    prop_assert_eq!(got, s.as_str());
                }
            }
        }
    }

    /// LineString layers survive encode→decode with exact geometry (per-feature
    /// vertex coordinates) plus id/time value-equality.
    #[test]
    fn linestring_layer_roundtrips_value_equal(layer in arb_linestring_layer()) {
        let payload = encode_tile(std::slice::from_ref(&layer)).unwrap();
        let mut decoded = decode_tile(&payload).unwrap();
        prop_assert_eq!(decoded.len(), 1);
        let batch = decoded.pop().unwrap().batch;
        prop_assert_eq!(batch.num_rows(), layer.feature_ids.len());
        let order = canonical_order(&layer.start_times);

        let ids = batch.column_by_name("id").unwrap()
            .as_any().downcast_ref::<UInt64Array>().unwrap();
        prop_assert_eq!(ids.values().to_vec(), permute(&layer.feature_ids, &order));
        let starts = batch.column_by_name("start_time").unwrap()
            .as_any().downcast_ref::<Int64Array>().unwrap();
        prop_assert_eq!(starts.values().to_vec(), permute(&layer.start_times, &order));
        let ends = batch.column_by_name("end_time").unwrap()
            .as_any().downcast_ref::<Int64Array>().unwrap();
        prop_assert_eq!(ends.values().to_vec(), permute(&layer.end_times, &order));

        // geometry: List<FixedSizeList<Float64,2>> — exact per vertex, and the
        // per-feature vertex COUNT is preserved (offset buffer round-trips).
        let geom = batch.column_by_name("geometry").unwrap()
            .as_any().downcast_ref::<ListArray>().unwrap();
        let GeometryColumn::LineString(lines) = &layer.geometry else { unreachable!() };
        let lines = permute(lines, &order);
        for (i, verts) in lines.iter().enumerate() {
            let line = geom.value(i);
            let line = line.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            prop_assert_eq!(line.len(), verts.len());
            for (j, [x, y]) in verts.iter().enumerate() {
                let pt = line.value(j);
                let pt = pt.as_any().downcast_ref::<Float64Array>().unwrap();
                prop_assert_eq!(pt.value(0), *x);
                prop_assert_eq!(pt.value(1), *y);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // ------------------------------------------------------------------
    // (a) directory v6 round-trip
    // ------------------------------------------------------------------

    /// Any valid entry list decodes back exactly (compared against the codec's
    /// canonical directory sort, which `encode_directory` applies internally).
    #[test]
    fn directory_v6_roundtrips_arbitrary_entries(
        mut entries in proptest::collection::vec(arb_entry(), 0..48),
        all_cover in any::<bool>(),
        cover in any::<i64>(),
    ) {
        // The cover section is all-or-nothing at encode (mixed Some/None
        // decodes as all-None by design), so generate one of the two
        // representable shapes.
        if all_cover {
            for (i, e) in entries.iter_mut().enumerate() {
                e.cover_t_min = Some(cover.wrapping_add(i as i64));
            }
        }
        let bytes = encode_directory(&entries);
        let decoded = decode_directory(&bytes).unwrap();
        let mut expected = entries.clone();
        expected.sort_by_key(|e| (e.zoom, e.hilbert, e.time_start));
        prop_assert_eq!(decoded, expected);
    }

    // ------------------------------------------------------------------
    // (b) directory decoder never panics
    // ------------------------------------------------------------------

    /// Arbitrary byte soup → `Ok` or `Err`, never a panic and never a
    /// header-claimed giant allocation.
    #[test]
    fn directory_decode_never_panics_on_arbitrary_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..2048),
    ) {
        let _ = decode_directory(&bytes);
    }

    /// Mutations + truncations of VALID directories — the nastier corpus,
    /// since prefixes and structure look plausible to the decoder.
    #[test]
    fn directory_decode_never_panics_on_mutated_valid_input(
        entries in proptest::collection::vec(arb_entry(), 1..24),
        flips in proptest::collection::vec(
            (any::<proptest::sample::Index>(), 1u8..=255),
            1..8,
        ),
        cut in any::<proptest::sample::Index>(),
    ) {
        let mut bytes = encode_directory(&entries);
        let cut_at = cut.index(bytes.len() + 1);
        let _ = decode_directory(&bytes[..cut_at]);
        for (idx, mask) in &flips {
            let i = idx.index(bytes.len());
            bytes[i] ^= mask;
        }
        let _ = decode_directory(&bytes);
    }

    // ------------------------------------------------------------------
    // (c) paged root decoder never panics
    // ------------------------------------------------------------------

    /// Arbitrary bytes with an arbitrary `rootLength` claim, both framings.
    #[test]
    fn paged_decode_never_panics_on_arbitrary_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..2048),
        root_length in any::<u64>(),
        zstd in any::<bool>(),
    ) {
        let _ = decode_paged_directory(&bytes, root_length, zstd);
    }

    /// Mutations + truncations of a VALID paged object. Raw (non-zstd) framing
    /// so mutations reach the root/leaf codecs directly instead of dying in
    /// the zstd frame header.
    #[test]
    fn paged_decode_never_panics_on_mutated_valid_input(
        entries in proptest::collection::vec(arb_pageable_entry(), 1..24),
        page_entries in 1usize..8,
        flips in proptest::collection::vec(
            (any::<proptest::sample::Index>(), 1u8..=255),
            1..8,
        ),
        cut in any::<proptest::sample::Index>(),
    ) {
        let enc = encode_paged_directory(&entries, page_entries, false).unwrap();
        let mut bytes = enc.bytes;
        let cut_at = cut.index(bytes.len() + 1);
        let _ = decode_paged_directory(&bytes[..cut_at], enc.root_length, false);
        for (idx, mask) in &flips {
            let i = idx.index(bytes.len());
            bytes[i] ^= mask;
        }
        let _ = decode_paged_directory(&bytes, enc.root_length, false);
    }

    // ------------------------------------------------------------------
    // (c) tile layer-frame decoder never panics
    // ------------------------------------------------------------------

    /// Arbitrary byte soup through the layer frame (and, where the frame
    /// happens to parse, the Arrow IPC reader) → `Ok` or `Err`, never a panic.
    #[test]
    fn tile_frame_decode_never_panics_on_arbitrary_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..1024),
    ) {
        let _ = decode_tile(&bytes);
    }

    /// Truncations anywhere + mutations of the FRAME bytes (layer count,
    /// name_len, name, ipc_len, pad — the region this crate's frame walker
    /// owns) of a VALID tile payload.
    ///
    /// Mutations are NOT applied inside the Arrow IPC stream itself: arrow-rs
    /// 59's IPC reader panics on semantically-corrupt buffer metadata
    /// (arrow-buffer `immutable.rs:288`, found by the unscoped version of this
    /// test), and that is upstream code. In-format corruption of IPC bytes
    /// cannot reach the Arrow decoder through the format's own read paths
    /// anyway — `PackedReader::read_payload` (and the TS reader) verify the
    /// per-blob CRC32C before any decode. Truncation is safe everywhere: a
    /// short buffer fails the frame's length checks before Arrow sees it.
    #[test]
    fn tile_frame_decode_never_panics_on_mutated_valid_input(
        ids in proptest::collection::vec(any::<u64>(), 1..8),
        flips in proptest::collection::vec(
            (any::<proptest::sample::Index>(), 1u8..=255),
            1..8,
        ),
        cut in any::<proptest::sample::Index>(),
    ) {
        let mut payload = tile_payload(ids);
        let cut_at = cut.index(payload.len() + 1);
        let _ = decode_tile(&payload[..cut_at]);
        // Frame header for the single layer named "default": 2 (count) +
        // 2 (name_len) + 7 (name) + 4 (ipc_len) = 15, padded to the 8-byte
        // boundary → the IPC stream starts at byte 16.
        let frame_region = 16usize.min(payload.len());
        for (idx, mask) in &flips {
            let i = idx.index(frame_region);
            payload[i] ^= mask;
        }
        let _ = decode_tile(&payload);
    }
}

// ----------------------------------------------------------------------
// Deterministic adversarial regressions: doctored headers must ERROR — not
// panic, not overflow, and not allocate what the header claims.
// ----------------------------------------------------------------------

/// A directory header claiming u64::MAX entries must be rejected up front
/// (pre-guard this drove `Vec::with_capacity` into a capacity-overflow panic).
#[test]
fn directory_header_claiming_huge_entry_count_errors() {
    let mut buf = vec![5u8]; // DIRECTORY_VERSION
    buf.extend_from_slice(&[0xff; 9]);
    buf.push(0x01); // 10-byte LEB128 = u64::MAX entries
    buf.push(0x00); // run_count = 0
    assert!(decode_directory(&buf).is_err());
}

/// The mid-range lie: a header claiming n ≈ bytes.len() entries passed the
/// old `n > bytes.len()` guard and still forced a ~56×-amplified scratch
/// allocation (the `Key` scratch row is 56 B vs the ≥8 wire bytes a real
/// entry costs). The tightened `(n + runs) > len/8` guard must reject it
/// before `with_capacity`.
#[test]
fn directory_header_midrange_entry_count_lie_errors() {
    let mut buf = vec![5u8]; // DIRECTORY_VERSION
    buf.push(0xe8);
    buf.push(0x07); // n = 1000 (two-byte LEB128)
    buf.push(0x00); // run_count = 0
                    // ~1 KB of zeros: 1000 entries would need ≥ 8000 wire bytes, so the
                    // guard must fire even though n < bytes.len().
    buf.extend_from_slice(&[0x00; 1024]);
    assert!(decode_directory(&buf).is_err());
}

/// A doctored run_len of u64::MAX must be rejected, not overflow the
/// `cursor + run_len` bound check (debug-build panic pre-guard).
#[test]
fn directory_run_len_overflow_errors() {
    let mut buf = vec![5u8]; // DIRECTORY_VERSION
    buf.push(0x01); // n = 1
    buf.push(0x01); // run_count = 1
    buf.extend_from_slice(&[0x00; 8]); // one entry: eight zero key varints
    buf.extend_from_slice(&[0xff; 9]);
    buf.push(0x01); // run_len = u64::MAX
    buf.push(0x00); // Δpack_id = 0
    buf.push(0x00); // offset sentinel (contiguous)
    buf.push(0x00); // length = 0
    buf.push(0x00); // uncompressed = 0
    buf.extend_from_slice(&0u32.to_le_bytes()); // crc
    assert!(decode_directory(&buf).is_err());
}

/// A paged root claiming u32::MAX pages must be rejected before allocating
/// the descriptor table (~200 GB as claimed).
#[test]
fn paged_root_claiming_huge_page_count_errors() {
    let mut root = vec![1u8, 0u8]; // PAGED_ROOT_VERSION, DESCRIPTOR_GEO_BBOX
    root.extend_from_slice(&0u16.to_le_bytes()); // reserved
    root.extend_from_slice(&u32::MAX.to_le_bytes()); // page_count
    root.extend_from_slice(&0u32.to_le_bytes()); // page_entries
    let rl = root.len() as u64;
    assert!(decode_paged_directory(&root, rl, false).is_err());
}

/// A descriptor whose rel_offset/length sit at u64/u32::MAX must error, not
/// overflow the leaf-range arithmetic (debug-build panic pre-guard).
#[test]
fn paged_leaf_range_overflow_errors() {
    use stt_core::directory_page::{encode_root, PageDescriptor};
    let d = PageDescriptor {
        min_bucket_start: None,
        rel_offset: u64::MAX,
        length: u32::MAX,
        entry_count: 1,
        min_zoom: 0,
        max_zoom: 0,
        min_lon_e7: 0,
        min_lat_e7: 0,
        max_lon_e7: 0,
        max_lat_e7: 0,
        t_min: 0,
        t_max: 0,
    };
    let root = encode_root(1, &[d]);
    let rl = root.len() as u64;
    assert!(decode_paged_directory(&root, rl, false).is_err());
}

/// A post-unframe root page of exactly ONE byte (`[PAGED_ROOT_VERSION]`) must
/// error, not index the missing descriptor-kind byte and panic (`decode_root`
/// used to read byte 1 unconditionally after guarding only byte 0).
#[test]
fn paged_root_one_byte_page_errors() {
    // 1 == PAGED_ROOT_VERSION; the descriptor-kind byte is absent.
    assert!(stt_core::directory_page::decode_root(&[1u8]).is_err());
}

/// A corrupt paged descriptor can carry an out-of-range zoom or a max-value
/// tile x/y; the geo-bounds inverse projection (reached from
/// `verify_paged_structure`) must compute without shift- or add-overflow
/// panicking.
#[test]
fn tile_geo_bounds_out_of_range_does_not_panic() {
    // zoom >= 32 would shift-overflow `1u32 << zoom`; x/y == u32::MAX would
    // overflow the `+ 1` adjacent-tile-edge lookup.
    let _ = stt_core::projection::tile_geo_bounds(40, u32::MAX, u32::MAX);
    let _ = stt_core::projection::tile_geo_bounds(255, 0, 0);
    let _ = stt_core::projection::tile_to_lonlat(u32::MAX, u32::MAX, 40);
}

// ----------------------------------------------------------------------
// (d) v2 layer frame (packed formatVersion 3) never panics.
//
// The v1 fuzzers above already cover arbitrary soup through `decode_tile`'s
// dispatch; these force the `0xFFFF` v2 escape so the SECTIONED walker (ref
// kinds, TOC, per-section pads, TILE_META JSON, template splice) is what the
// input reaches, in both decode modes (with and without a registry).
// ----------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Arbitrary bytes behind the v2 escape → `Ok`/`Err`, never a panic —
    /// registry-less AND with an (empty) registry.
    #[test]
    fn v2_frame_decode_never_panics_on_arbitrary_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..1024),
    ) {
        let mut payload = vec![0xffu8, 0xff];
        payload.extend_from_slice(&bytes);
        let _ = decode_tile(&payload);
        let registry = stt_core::arrow_tile::TemplateRegistry::new();
        let _ = stt_core::arrow_tile::decode_tile_with_templates(&payload, &registry);
    }

    /// Truncations anywhere + mutations of the v2 FRAME bytes (escape,
    /// frame_version, flags, layer_count, name, ref kinds, section_count,
    /// TOC tags/lengths, pad — the pre-section region this crate's walker
    /// owns) of a VALID self-contained v2 tile.
    ///
    /// As with the v1 fuzzer above, mutations stop at the first section:
    /// in-IPC corruption is upstream arrow-rs territory and unreachable
    /// through the format's CRC-checked read paths. (Mutated TOC LENGTHS
    /// still re-slice the sections arbitrarily — that's the interesting
    /// part, and the splice guards must turn every mis-slice into `Err`.)
    #[test]
    fn v2_frame_decode_never_panics_on_mutated_valid_input(
        ids in proptest::collection::vec(any::<u64>(), 1..8),
        flips in proptest::collection::vec(
            (any::<proptest::sample::Index>(), 1u8..=255),
            1..8,
        ),
        cut in any::<proptest::sample::Index>(),
    ) {
        let mut payload = tile_payload_v2(ids);
        let cut_at = cut.index(payload.len() + 1);
        let _ = decode_tile(&payload[..cut_at]);
        // Pre-section region of the single props-less layer "default":
        // escape 2 + version 1 + flags 1 + count 2 + name_len 2 + name 7 +
        // ref_core 1 + ref_props 1 + section_count 1 + TOC 3×5 = 33, padded
        // to the 8-byte boundary → the first section starts at byte 40.
        let frame_region = 40usize.min(payload.len());
        for (idx, mask) in &flips {
            let i = idx.index(frame_region);
            payload[i] ^= mask;
        }
        let _ = decode_tile(&payload);
        let registry = stt_core::arrow_tile::TemplateRegistry::new();
        let _ = stt_core::arrow_tile::decode_tile_with_templates(&payload, &registry);
    }
}

// ----------------------------------------------------------------------
// (e) SH-1 — the semantic content fingerprint.
//
// Two properties, both about the check being SOUND rather than merely
// present:
//
// 1. **Never panics.** The accumulator walks decoded Arrow arrays by hand
//    (geometry nesting, quantization affines, dictionary keys). Every one of
//    those is a `downcast`/index site, and the validator must degrade to a
//    finding rather than abort a fleet-wide run on one odd layer.
// 2. **Containment soundness.** For any SUBSET of a feature set, the
//    observation is contained by the full set's fingerprint. This is the
//    property that makes `--sample` a real check instead of a guess: a
//    sampled run compares a subset observation against a whole-dataset
//    declaration, so if containment could fail on an honest subset the
//    sampled mode would be a false-positive generator.
// ----------------------------------------------------------------------

/// The fingerprint a writer would declare for exactly this feature set —
/// i.e. the full-set observation restated as a declaration.
fn declare_from(observed: &stt_core::metadata::ObservedFingerprint) -> ContentFingerprint {
    ContentFingerprint {
        version: stt_core::metadata::CONTENT_FINGERPRINT_VERSION,
        bbox: observed.bbox.unwrap_or([-180.0, -90.0, 180.0, 90.0]),
        z_range: observed.z_range,
        distinct_feature_count: observed.distinct_ids_estimate,
        numeric_ranges: observed.numeric_ranges.clone(),
        categorical_cardinality: observed.categorical_cardinality.clone(),
        coord_tolerance_deg: 0.0,
        column_tolerance: Default::default(),
    }
}

/// Decode a layer to the shape the accumulator ingests.
fn observe(layer: &ColumnarLayer) -> stt_core::metadata::ObservedFingerprint {
    let payload = encode_tile(std::slice::from_ref(layer)).expect("encode");
    let layers = decode_tile(&payload).expect("decode");
    let mut acc = FingerprintAccumulator::new();
    acc.ingest(&layers);
    acc.finish()
}

/// Keep the first `k` features of a point layer (k >= 1).
fn head(layer: &ColumnarLayer, k: usize) -> ColumnarLayer {
    let k = k.clamp(1, layer.feature_ids.len());
    let geometry = match &layer.geometry {
        GeometryColumn::Point(p) => GeometryColumn::Point(p[..k].to_vec()),
        GeometryColumn::LineString(l) => GeometryColumn::LineString(l[..k].to_vec()),
        GeometryColumn::Polygon(p) => GeometryColumn::Polygon(p[..k].to_vec()),
    };
    ColumnarLayer {
        polygon_parts: None,
        name: layer.name.clone(),
        feature_ids: layer.feature_ids[..k].to_vec(),
        start_times: layer.start_times[..k].to_vec(),
        end_times: layer.end_times[..k].to_vec(),
        geometry,
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: layer
            .properties
            .iter()
            .map(|(name, column)| {
                let column = match column {
                    PropertyColumn::Numeric(v) => PropertyColumn::Numeric(v[..k].to_vec()),
                    PropertyColumn::Categorical(v) => PropertyColumn::Categorical(v[..k].to_vec()),
                    // The generators above never mint a vector-group column,
                    // and it carries no v1 fingerprint statistic anyway.
                    other => other.clone(),
                };
                (name.clone(), column)
            })
            .collect(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// The fingerprint check never panics, whatever the decoded layers hold
    /// and whatever nonsense the declared block claims.
    #[test]
    fn fingerprint_check_never_panics(
        layer in arb_point_layer(),
        bbox in proptest::collection::vec(-1.0e12f64..1.0e12, 4),
        tolerance in -1.0f64..1.0e6,
        declared_count in any::<u64>(),
    ) {
        let observed = observe(&layer);
        let declared = ContentFingerprint {
            version: stt_core::metadata::CONTENT_FINGERPRINT_VERSION,
            bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
            z_range: Some([-1.0, 1.0]),
            distinct_feature_count: declared_count,
            numeric_ranges: Default::default(),
            categorical_cardinality: Default::default(),
            coord_tolerance_deg: tolerance,
            column_tolerance: Default::default(),
        };
        for complete in [false, true] {
            let _ = check_fingerprint(&declared, &observed, complete, &[]);
            let _ = check_fingerprint(
                &declared,
                &observed,
                complete,
                &["coord-quant".to_string(), "attr-quant".to_string()],
            );
        }
    }

    /// CONTAINMENT SOUNDNESS. A subset's observation always lies inside the
    /// full set's fingerprint, so a `--sample`d run of an honest archive can
    /// never manufacture a containment error.
    #[test]
    fn subset_observation_is_contained_by_the_full_fingerprint(
        layer in arb_point_layer(),
        k in any::<proptest::sample::Index>(),
    ) {
        let full = observe(&layer);
        let declared = declare_from(&full);
        let subset = observe(&head(&layer, k.index(layer.feature_ids.len()) + 1));

        // The structural statement of containment...
        if let (Some(f), Some(s)) = (full.bbox, subset.bbox) {
            prop_assert!(s[0] >= f[0] && s[1] >= f[1] && s[2] <= f[2] && s[3] <= f[3],
                "subset bbox {s:?} escapes full bbox {f:?}");
        }
        for (column, [lo, hi]) in &subset.numeric_ranges {
            let [flo, fhi] = full.numeric_ranges[column];
            prop_assert!(*lo >= flo && *hi <= fhi,
                "subset range for {column} [{lo}, {hi}] escapes [{flo}, {fhi}]");
        }
        for (column, count) in &subset.categorical_cardinality {
            prop_assert!(*count <= full.categorical_cardinality[column]);
        }

        // ...and the statement the validator actually makes: a sampled run
        // over a subset of an honest archive reports NO errors.
        let findings = check_fingerprint(&declared, &subset, false, &[]);
        prop_assert!(findings.errors.is_empty(), "errors were {:?}", findings.errors);
    }

    /// The full set compared against its own declaration is clean in the
    /// strict (full-decode equality) mode too — the R1 gate's shape.
    #[test]
    fn full_observation_equals_its_own_declaration(layer in arb_linestring_layer()) {
        let full = observe(&layer);
        let declared = declare_from(&full);
        let findings = check_fingerprint(&declared, &full, true, &[]);
        prop_assert!(findings.errors.is_empty(), "errors were {:?}", findings.errors);
    }
}

// ----------------------------------------------------------------------
// (f) TB-2 — the dataset-global attribute-range pin.
//
// This file exists to catch the class of bug where a decoder mis-reads an
// encoder's intent. The per-tile numeric affine is a purer version of the same
// bug: the decoder reads the affine correctly, and the ENCODER still ships two
// different intents for one column, because the offset and step are functions
// of whichever rows a tile happened to catch. A value read off two adjacent
// tiles therefore comes back as two different numbers, and nothing in the wire
// format is malformed enough for any check to notice.
//
// The pin makes the affine a function of the column's dataset DOMAIN instead,
// so the properties below are the ones that become assertable:
//
// 1. **One value, one decoding.** For a random dataset split arbitrarily into
//    tiles, every source value decodes to the same number in every tile that
//    holds it — the §3.3 constraint, stated over the whole generated domain
//    rather than one example.
// 2. **Partition invariance.** The shipped Arrow type and affine do not depend
//    on HOW the dataset was split, which is the conformance-invariance rule
//    (§13.2): the verdict is a function of the dataset, never of the sampling.
// 3. **Determinism.** Re-encoding one tile under one pin is byte-identical.
// 4. **The encoder's decision site honours the pin** over the whole domain.
//
// Plus one deterministic control, run through the PUBLIC encoder, showing the
// defect is real in the shipped per-tile path — which is also the documented
// `--single-pass` fallback, so that control stays true forever.
// ----------------------------------------------------------------------

/// Attribute values chosen to reach every branch of the pin's rule tree:
/// nulls and non-finite cells (Arrow nulls), a fractional body (the
/// range-adaptive branch), integers (the exact step-1 branch), integers that
/// straddle the `UInt16` leaf (the width widening) and magnitudes that straddle
/// `i32::MAX` (the refusal).
fn arb_attr_value() -> impl Strategy<Value = Option<f64>> {
    prop_oneof![
        2 => Just(None),
        1 => Just(Some(f64::NAN)),
        1 => Just(Some(f64::NEG_INFINITY)),
        8 => (-1.0e3f64..1.0e3).prop_map(Some),
        6 => (-1000i64..1000).prop_map(|i| Some(i as f64)),
        3 => (0i64..70_000).prop_map(|i| Some(i as f64)),
        2 => (-3.0e9f64..3.0e9).prop_map(Some),
    ]
}

/// The builder's pass-1 numeric scan, restated at test scale: the dataset-wide
/// statistics the pin is derived from. Deliberately computed here rather than
/// imported — `stt_build::dataset_stats` sits in the crate above this one, and
/// the property under test is about the ENCODER honouring a domain, not about
/// how the domain was measured.
fn global_pin(dataset: &[Option<f64>]) -> AttrPinned {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut max_abs = 0.0f64;
    let mut all_integer = true;
    let mut finite = 0u64;
    for value in dataset.iter().flatten() {
        if !value.is_finite() {
            continue;
        }
        finite += 1;
        min = min.min(*value);
        max = max.max(*value);
        max_abs = max_abs.max(value.abs());
        all_integer &= *value == value.trunc();
    }
    AttrPinned::derive_auto(min, max, max_abs, all_integer, finite)
}

/// Split a dataset into tiles by an arbitrary assignment vector — an ARBITRARY
/// partition, not a balanced one, since the whole question is whether an
/// unlucky split can move a verdict.
fn split_into_tiles(dataset: &[Option<f64>], assign: &[u8]) -> Vec<Vec<Option<f64>>> {
    let k = assign.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut tiles = vec![Vec::new(); k];
    for (i, value) in dataset.iter().enumerate() {
        tiles[assign[i % assign.len()] as usize].push(*value);
    }
    tiles.retain(|tile| !tile.is_empty());
    tiles
}

/// Reconstruct a quantized column the way both reference readers do: parse the
/// shipped affine and apply it to every non-null index. `None` is the encoder
/// declining to quantize (the column stays `Float64`).
fn decode_pinned(
    tile: &[Option<f64>],
    pin: &AttrPinned,
) -> Option<(DataType, String, Vec<Option<f64>>)> {
    let (array, json) = build_quantized_numeric_pinned(tile, pin)
        .expect("a tile drawn from the pinned dataset can never escape its own pin")?;
    let affine = AttrQuant::from_json(&json).expect("a quantized column must ship a valid qa");
    let data_type = array.data_type().clone();
    let back: Vec<Option<f64>> = match &data_type {
        DataType::UInt16 => {
            let c = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            (0..c.len())
                .map(|i| (!c.is_null(i)).then(|| affine.value(c.value(i) as i64)))
                .collect()
        }
        DataType::Int32 => {
            let c = array.as_any().downcast_ref::<Int32Array>().unwrap();
            (0..c.len())
                .map(|i| (!c.is_null(i)).then(|| affine.value(c.value(i) as i64)))
                .collect()
        }
        other => panic!("quantized property leaf must be UInt16/Int32, got {other:?}"),
    };
    Some((data_type, json, back))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// **The item's core claim as a property.** Split a dataset arbitrarily
    /// into tiles, pin the affine to the dataset, and every source value
    /// decodes to the SAME number in every tile that holds it — and every tile
    /// ships the same Arrow type and the same affine, so there is nothing left
    /// for a schema-drift check to report.
    #[test]
    fn pinned_attr_affine_decodes_every_value_identically_in_every_tile(
        dataset in proptest::collection::vec(arb_attr_value(), 1..48),
        assign in proptest::collection::vec(0u8..4, 1..8),
    ) {
        let pin = global_pin(&dataset);
        let tiles = split_into_tiles(&dataset, &assign);
        prop_assert!(!tiles.is_empty());

        // value bits → decoded bits, accumulated across every tile.
        let mut seen: HashMap<u64, u64> = HashMap::new();
        let mut shape: Option<(DataType, String)> = None;

        for tile in &tiles {
            let Some((data_type, json, back)) = decode_pinned(tile, &pin) else {
                prop_assert!(pin.refuse, "only the magnitude refusal may decline to quantize");
                continue;
            };
            prop_assert!(!pin.refuse, "a refusing pin must decline in every tile");

            match &shape {
                None => shape = Some((data_type, json)),
                Some((want_type, want_json)) => {
                    prop_assert_eq!(&data_type, want_type,
                        "one column forked its Arrow type between tiles under a global pin");
                    prop_assert_eq!(&json, want_json,
                        "one column shipped two different affines under a global pin");
                }
            }

            for (value, decoded) in tile.iter().zip(&back) {
                let present = matches!(value, Some(x) if x.is_finite());
                prop_assert_eq!(present, decoded.is_some(),
                    "the null mask must follow finiteness exactly");
                if let (Some(x), Some(got)) = (value, decoded) {
                    if let Some(previous) = seen.insert(x.to_bits(), got.to_bits()) {
                        prop_assert_eq!(previous, got.to_bits(),
                            "source value {} decoded two different ways across tiles", x);
                    }
                }
            }
        }
    }

    /// **Conformance invariance (§13.2).** The verdict is a function of the
    /// dataset alone, so re-splitting the SAME dataset a different way cannot
    /// move the Arrow type or the affine of any tile.
    #[test]
    fn pinned_attr_affine_is_invariant_to_the_partition(
        dataset in proptest::collection::vec(arb_attr_value(), 1..48),
        assign_a in proptest::collection::vec(0u8..4, 1..8),
        assign_b in proptest::collection::vec(0u8..6, 1..13),
    ) {
        let pin = global_pin(&dataset);
        let shapes = |assign: &[u8]| -> Vec<Option<(DataType, String)>> {
            split_into_tiles(&dataset, assign)
                .iter()
                .map(|tile| decode_pinned(tile, &pin).map(|(t, json, _)| (t, json)))
                .collect()
        };
        let mut all = shapes(&assign_a);
        all.extend(shapes(&assign_b));
        prop_assert!(!all.is_empty());
        let first = all[0].clone();
        for shape in &all {
            prop_assert_eq!(shape, &first,
                "the pinned verdict moved with the partition — which is the defect, not the fix");
        }
    }

    /// Determinism: one tile, one pin, re-encoded, is byte-identical. The
    /// mandatory re-run test for an encoder change, at the column level.
    #[test]
    fn pinned_attr_quantization_is_byte_identical_on_re_encode(
        dataset in proptest::collection::vec(arb_attr_value(), 1..48),
        assign in proptest::collection::vec(0u8..4, 1..8),
    ) {
        let pin = global_pin(&dataset);
        for tile in split_into_tiles(&dataset, &assign) {
            let first = build_quantized_numeric_pinned(&tile, &pin).unwrap();
            let second = build_quantized_numeric_pinned(&tile, &pin).unwrap();
            match (first, second) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    prop_assert_eq!(&a.1, &b.1, "the affine JSON moved between two runs");
                    prop_assert!(a.0.to_data() == b.0.to_data(),
                        "the quantized column moved between two runs");
                }
                _ => prop_assert!(false, "two runs disagreed on whether to quantize"),
            }
        }
    }

    /// The encoder's decision site honours the pin over the whole domain: with
    /// pins attached and the auto lever on, the dispatch is exactly the pinned
    /// quantizer; with no pins attached it is exactly the incumbent per-tile
    /// one (the `--single-pass` fallback).
    #[test]
    fn encoder_dispatch_honours_the_pin_over_the_whole_domain(
        dataset in proptest::collection::vec(arb_attr_value(), 1..48),
        assign in proptest::collection::vec(0u8..4, 1..8),
    ) {
        let pin = global_pin(&dataset);
        let mut pins = GlobalColumnPins::default();
        pins.attr.insert("speed".to_string(), pin);
        let pinned = EncoderConfig {
            quantize_attrs_auto: true,
            global_pins: Some(Arc::new(pins)),
            ..EncoderConfig::default()
        };
        let unpinned = EncoderConfig {
            quantize_attrs_auto: true,
            ..EncoderConfig::default()
        };

        for tile in split_into_tiles(&dataset, &assign) {
            let got = build_quantized_numeric_for_column("speed", &tile, &pinned).unwrap();
            let want = build_quantized_numeric_pinned(&tile, &pin).unwrap();
            match (got, want) {
                (None, None) => prop_assert!(pin.refuse),
                (Some(got), Some(want)) => {
                    prop_assert_eq!(&got.1, &want.1);
                    prop_assert!(got.0.to_data() == want.0.to_data());
                }
                _ => prop_assert!(false, "the dispatch disagreed with the pinned quantizer"),
            }
            // No pins ⇒ the incumbent, so the rollback is a real rollback: the
            // decision reverts to a function of THIS TILE's sample, declining
            // exactly when the tile's own magnitude trips the threshold (the
            // per-tile path's sole refusal) rather than when the dataset's does.
            let fallback = build_quantized_numeric_for_column("speed", &tile, &unpinned).unwrap();
            let tile_max_abs = tile.iter().flatten()
                .filter(|v| v.is_finite())
                .fold(0.0f64, |acc, v| acc.max(v.abs()));
            prop_assert_eq!(fallback.is_none(), tile_max_abs >= i32::MAX as f64,
                "the unpinned dispatch must be the incumbent per-tile rule, sample and all");
        }
    }
}

/// Encode one numeric column as a real tile and read it back the way a reader
/// does — through the `stt:qa` affine the frame ships on the field.
fn encode_one_column_tile(
    values: &[Option<f64>],
    cfg: &EncoderConfig,
) -> (DataType, AttrQuant, Vec<Option<f64>>) {
    let n = values.len();
    let layer = ColumnarLayer {
        polygon_parts: None,
        name: "points".to_string(),
        feature_ids: (0..n as u64).collect(),
        start_times: (0..n as i64).collect(),
        end_times: (0..n as i64).collect(),
        geometry: GeometryColumn::Point(vec![[0.0, 0.0]; n]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![(
            "speed".to_string(),
            PropertyColumn::Numeric(values.to_vec()),
        )],
    };
    let payload = encode_tile_with(std::slice::from_ref(&layer), cfg).expect("encode");
    let mut decoded = decode_tile(&payload).expect("decode");
    let batch = decoded.pop().expect("one layer").batch;
    let field = batch
        .schema()
        .field_with_name("speed")
        .expect("the property column survives")
        .clone();
    let json = field
        .metadata()
        .get(STT_QUANT_ATTR_META_KEY)
        .expect("a quantized column ships its affine")
        .clone();
    let affine = AttrQuant::from_json(&json).expect("valid qa");
    let column = batch.column_by_name("speed").unwrap();
    let data_type = column.data_type().clone();
    let back: Vec<Option<f64>> = match &data_type {
        DataType::UInt16 => {
            let c = column.as_any().downcast_ref::<UInt16Array>().unwrap();
            (0..c.len())
                .map(|i| (!c.is_null(i)).then(|| affine.value(c.value(i) as i64)))
                .collect()
        }
        DataType::Int32 => {
            let c = column.as_any().downcast_ref::<Int32Array>().unwrap();
            (0..c.len())
                .map(|i| (!c.is_null(i)).then(|| affine.value(c.value(i) as i64)))
                .collect()
        }
        other => panic!("quantized property leaf must be UInt16/Int32, got {other:?}"),
    };
    (data_type, affine, back)
}

/// **The defect, through the public encoder.** Two tiles of one column, both
/// holding the value `0.25`. Under the shipped per-tile affine they decode it
/// to two different numbers, because each tile sized its step from its own
/// span — no corruption, no malformed byte, just two intents for one column.
///
/// The second half shows the pinned decision site collapsing that to one
/// number. Both halves now run through `encode_tile_with` — the encoder's
/// property loop calls the pinned dispatch, so the fix is observable on the
/// real wire rather than only at the decision function. The per-tile half stays
/// true regardless: it is the documented `--single-pass` fallback.
#[test]
fn per_tile_auto_affine_lets_one_source_value_decode_two_ways() {
    let cfg = EncoderConfig {
        quantize_attrs_auto: true,
        ..EncoderConfig::default()
    };
    // Fractional, so the range-adaptive branch (whose step is the tile's span
    // over 65535) is what runs; the exact integer branch is exact either way.
    let a = [Some(0.0), Some(1.0), Some(0.25)];
    let b = [Some(0.0), Some(100.0), Some(0.25)];

    let (type_a, affine_a, back_a) = encode_one_column_tile(&a, &cfg);
    let (type_b, affine_b, back_b) = encode_one_column_tile(&b, &cfg);
    assert_eq!(type_a, type_b);
    assert_ne!(
        affine_a, affine_b,
        "the per-tile affine is sized from the tile's own span"
    );
    assert_ne!(
        back_a[2], back_b[2],
        "0.25 must decode two ways on the per-tile path — that is the defect TB-2 fixes"
    );

    // The same two tiles under one dataset-global pin.
    let dataset: Vec<Option<f64>> = a.iter().chain(b.iter()).copied().collect();
    let pin = global_pin(&dataset);
    let mut pins = GlobalColumnPins::default();
    pins.attr.insert("speed".to_string(), pin);
    let pinned_cfg = EncoderConfig {
        quantize_attrs_auto: true,
        global_pins: Some(Arc::new(pins)),
        ..EncoderConfig::default()
    };
    // Through the REAL encoder this time — same `encode_tile_with` call, the
    // only difference being the pins riding the config.
    let (pinned_type_a, pinned_affine_a, pinned_back_a) = encode_one_column_tile(&a, &pinned_cfg);
    let (pinned_type_b, pinned_affine_b, pinned_back_b) = encode_one_column_tile(&b, &pinned_cfg);
    assert_eq!(pinned_type_a, pinned_type_b);
    assert_eq!(
        pinned_affine_a, pinned_affine_b,
        "both tiles must SHIP the same affine under the pin — this is the \
         `stt:qa` a reader actually parses, not an internal value"
    );
    assert_eq!(
        pinned_back_a[2], pinned_back_b[2],
        "0.25 must decode to ONE number under the dataset-global pin"
    );
    // …and the pinned reading is the one the decision function promised.
    let decoded_a = decode_pinned(&a, &pin).unwrap().2;
    let decoded_b = decode_pinned(&b, &pin).unwrap().2;
    assert_eq!(decoded_a[2], decoded_b[2]);
    assert_eq!(pinned_back_a[2], decoded_a[2]);
}

/// **The §3.3 constraint, end to end over a whole multi-tile dataset.**
///
/// The claim M2 exists to make is not "the pinned function is self-consistent"
/// — it is *"the same source value decodes to the same number in every tile of
/// the archive"*. That is a statement about tiles that were encoded
/// independently, so it has to be tested by encoding them independently and
/// reading the values back the way a reader does: off the `stt:qa` affine each
/// tile ships on its own field.
///
/// The partition is arbitrary and the values are shared across tiles by
/// construction (every tile gets a copy of the probe value), which is exactly
/// the shape the per-tile affine fails on — the incumbent half of the test
/// below shows it failing, so this cannot pass vacuously.
#[test]
fn one_value_decodes_identically_in_every_tile_of_a_pinned_multi_tile_dataset() {
    // A probe value that appears in EVERY tile, plus per-tile filler that gives
    // each tile a different local span — the only thing the incumbent affine
    // looks at.
    const PROBE: f64 = 0.25;
    let tiles: Vec<Vec<Option<f64>>> = vec![
        vec![Some(0.0), Some(1.0), Some(PROBE)],
        vec![Some(0.0), Some(100.0), Some(PROBE), None],
        vec![Some(-50.0), Some(7.5), Some(PROBE)],
        vec![Some(PROBE)], // a single-value tile: span 0, the degenerate case
        vec![Some(0.0), Some(999.5), Some(PROBE), Some(-3.25)],
    ];
    let dataset: Vec<Option<f64>> = tiles.iter().flatten().copied().collect();

    let mut pins = GlobalColumnPins::default();
    pins.attr.insert("speed".to_string(), global_pin(&dataset));
    let pinned_cfg = EncoderConfig {
        quantize_attrs_auto: true,
        global_pins: Some(Arc::new(pins)),
        ..EncoderConfig::default()
    };
    let unpinned_cfg = EncoderConfig {
        quantize_attrs_auto: true,
        ..EncoderConfig::default()
    };

    // Each tile encoded on its own, then read back through its own shipped
    // affine — no shared state between them but the pin.
    let pinned: Vec<(DataType, AttrQuant, Vec<Option<f64>>)> = tiles
        .iter()
        .map(|tile| encode_one_column_tile(tile, &pinned_cfg))
        .collect();

    let (first_type, first_affine, _) = &pinned[0];
    for (i, (data_type, affine, _)) in pinned.iter().enumerate() {
        assert_eq!(
            data_type, first_type,
            "tile {i} shipped a different Arrow type — one column, one type, everywhere"
        );
        assert_eq!(
            affine, first_affine,
            "tile {i} shipped a different affine — the pin is supposed to be \
             a function of the DOMAIN, not of the sample"
        );
    }

    // The constraint itself: the probe reads back as one number everywhere.
    let readings: Vec<f64> = tiles
        .iter()
        .zip(pinned.iter())
        .map(|(tile, (_, _, back))| {
            let at = tile
                .iter()
                .position(|v| *v == Some(PROBE))
                .expect("every tile holds the probe");
            back[at].expect("the probe is not null")
        })
        .collect();
    for (i, value) in readings.iter().enumerate() {
        assert_eq!(
            *value, readings[0],
            "tile {i} decoded {PROBE} as {value}, tile 0 decoded it as {} — \
             this is the §3.3 defect, and it must be gone",
            readings[0]
        );
    }

    // Anti-vacuity: the SAME dataset, the SAME partition, on the incumbent
    // per-tile path disagrees. If this half ever goes quiet the test above has
    // stopped proving anything.
    let incumbent: Vec<f64> = tiles
        .iter()
        .map(|tile| {
            let (_, _, back) = encode_one_column_tile(tile, &unpinned_cfg);
            let at = tile.iter().position(|v| *v == Some(PROBE)).unwrap();
            back[at].unwrap()
        })
        .collect();
    assert!(
        incumbent.iter().any(|v| *v != incumbent[0]),
        "the per-tile path must still disagree with itself on this dataset \
         ({incumbent:?}) — otherwise the pinned assertion above is vacuous"
    );
}

/// The property form of the same claim, over arbitrary datasets and arbitrary
/// partitions, through the real encoder: **decode(value) is identical across
/// tiles**. Values shared between tiles are the interesting ones, so the
/// strategy draws a small alphabet and the partition scatters it.
///
/// Also pins the two invariants a pinned encode must not lose while doing it:
/// one Arrow type per column dataset-wide, and one affine.
#[test]
fn prop_pinned_encoding_decodes_one_value_one_way_across_arbitrary_partitions() {
    use proptest::test_runner::{Config, TestRunner};

    let strategy = (
        proptest::collection::vec(
            proptest::sample::select(vec![
                Some(0.0),
                Some(0.25),
                Some(-3.5),
                Some(1.0),
                Some(42.0),
                Some(1_000.5),
                None,
            ]),
            2..40,
        ),
        proptest::collection::vec(0u8..5, 2..12),
    );
    let mut runner = TestRunner::new(Config {
        cases: 256,
        ..Config::default()
    });
    runner
        .run(&strategy, |(dataset, assign)| {
            let tiles = split_into_tiles(&dataset, &assign);
            if tiles.is_empty() {
                return Ok(());
            }
            let mut pins = GlobalColumnPins::default();
            pins.attr.insert("speed".to_string(), global_pin(&dataset));
            let cfg = EncoderConfig {
                quantize_attrs_auto: true,
                global_pins: Some(Arc::new(pins)),
                ..EncoderConfig::default()
            };

            // value → the number it decoded to, wherever it was first seen.
            let mut seen: HashMap<u64, f64> = HashMap::new();
            let mut shape: Option<(DataType, AttrQuant)> = None;
            for tile in &tiles {
                if tile.iter().all(|v| v.is_none()) {
                    // An all-null column is dropped by the encoder, so there is
                    // no field to read an affine off.
                    continue;
                }
                let (data_type, affine, back) = encode_one_column_tile(tile, &cfg);
                match &shape {
                    None => shape = Some((data_type, affine)),
                    Some(first) => {
                        prop_assert_eq!(
                            &(data_type, affine),
                            first,
                            "the pinned type/affine moved with the partition"
                        );
                    }
                }
                for (source, decoded) in tile.iter().zip(back.iter()) {
                    let (Some(source), Some(decoded)) = (source, decoded) else {
                        prop_assert_eq!(
                            source.is_none(),
                            decoded.is_none(),
                            "nullness must survive the pinned encode"
                        );
                        continue;
                    };
                    match seen.insert(source.to_bits(), *decoded) {
                        Some(previous) => prop_assert_eq!(
                            previous,
                            *decoded,
                            "one source value decoded two ways across tiles"
                        ),
                        None => {}
                    }
                }
            }
            Ok(())
        })
        .expect("pinned multi-tile decode agreement");
}
