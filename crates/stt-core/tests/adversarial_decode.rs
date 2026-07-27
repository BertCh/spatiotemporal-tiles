//! Adversarial property tests for the STT decoders: the v5 directory codec,
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
    Array, DictionaryArray, FixedSizeListArray, Float64Array, Int64Array, ListArray, StringArray,
    UInt64Array,
};
use arrow::datatypes::UInt16Type;
use proptest::prelude::*;
use stt_core::arrow_tile::{
    decode_tile, encode_tile, ColumnarLayer, GeometryColumn, PropertyColumn,
};
use stt_core::directory::{decode_directory, encode_directory};
use stt_core::directory_page::{decode_paged_directory, encode_paged_directory};

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

        // categorical property "kind": Dictionary<UInt16,Utf8> — exact string.
        let kind = batch.column_by_name("kind").unwrap()
            .as_any().downcast_ref::<DictionaryArray<UInt16Type>>().unwrap();
        let dict_vals = kind.values().as_any().downcast_ref::<StringArray>().unwrap();
        let keys = kind.keys();
        let PropertyColumn::Categorical(kw) = &layer.properties[1].1 else { unreachable!() };
        let kw = permute(kw, &order);
        for (i, w) in kw.iter().enumerate() {
            match w {
                None => prop_assert!(kind.is_null(i), "row {} expected null kind", i),
                Some(s) => {
                    prop_assert!(!kind.is_null(i), "row {} unexpectedly null", i);
                    let got = dict_vals.value(keys.value(i) as usize);
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
    // (a) directory v5 round-trip
    // ------------------------------------------------------------------

    /// Any valid entry list decodes back exactly (compared against the codec's
    /// canonical directory sort, which `encode_directory` applies internally).
    #[test]
    fn directory_v5_roundtrips_arbitrary_entries(
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
// (d) v2 layer frame (packed formatVersion 2) never panics.
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
