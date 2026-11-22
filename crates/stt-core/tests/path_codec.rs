//! The path codec at frame level: `decode(encode_with_codec(layer))` equals
//! `decode(encode_without_codec(layer))` column for column (bit-exact on f64),
//! the section is emitted only for eligible layers, the observation channel
//! reports it, the CORE template does not fork, and the output is
//! deterministic.
//!
//! The section codec itself (both modes, every rejection) is unit-tested in
//! `src/arrow_tile/path_codec.rs`; the never-panic properties are in
//! `tests/adversarial_decode.rs`.

use std::sync::Arc;

use arrow::array::{Array, Int64Array, ListArray, RecordBatch, UInt16Array, UInt32Array};
use stt_core::arrow_tile::{
    decode_tile, encode_tile_observed, encode_tile_with, path_codec, ColumnarLayer, DecodedLayer,
    EncoderConfig, GeometryColumn, PropertyColumn, TemplateCollector, SECTION_PATH_CODEC,
};

const T0: i64 = 1_700_000_000_000;

fn layer(
    name: &str,
    lines: Vec<Vec<[f64; 2]>>,
    times: Option<Vec<Vec<i64>>>,
    starts: Vec<i64>,
) -> ColumnarLayer {
    let n = lines.len();
    ColumnarLayer {
        name: name.to_string(),
        feature_ids: (0..n as u64).map(|i| 100 + i).collect(),
        end_times: starts.iter().map(|s| s + 60_000).collect(),
        start_times: starts,
        geometry: GeometryColumn::LineString(lines),
        vertex_times: times,
        vertex_values: None,
        vertex_value_matrix: None,
        triangles: None,
        polygon_parts: None,
        properties: vec![(
            "speed".to_string(),
            PropertyColumn::Numeric((0..n).map(|i| Some(i as f64 * 1.5)).collect()),
        )],
    }
}

/// A realistic trajectory layer: GPS-like decimal fixes, irregular sampling,
/// clipped (non-decimal) endpoints, an empty and a single-vertex feature,
/// unsorted times inside one feature, negative times, and rows out of
/// start-time order (the encoder sorts them).
fn tracks() -> ColumnarLayer {
    let mut lines = Vec::new();
    let mut times = Vec::new();
    let mut starts = Vec::new();
    for f in 0..12i64 {
        let n = (f % 5 + 2) as usize;
        let mut line = Vec::new();
        let mut t = Vec::new();
        let mut ti = T0 + (11 - f) * 3_600_000;
        for i in 0..n {
            let x = -122.0 + f as f64 * 0.01 + i as f64 * 0.000_37;
            let y = 37.0 + i as f64 * 0.000_21;
            // Five-decimal fixes, like an AIS/GPS feed.
            line.push([(x * 1e5).round() / 1e5, (y * 1e5).round() / 1e5]);
            t.push(ti);
            ti += [60_000, 61_000, 120_000, 30_000][i % 4];
        }
        if f == 3 {
            // A clip vertex: an interpolated, non-decimal coordinate and time.
            line[0] = [-121.970_000_123_456_7, 37.000_001_234_5];
            t[0] += 17;
        }
        if f == 5 {
            t.swap(0, 1); // unsorted within the feature
        }
        starts.push(t.iter().copied().min().unwrap());
        lines.push(line);
        times.push(t);
    }
    lines.push(vec![]);
    times.push(vec![]);
    starts.push(T0 - 5);
    lines.push(vec![[1.25, -2.5]]);
    times.push(vec![-86_400_000]);
    starts.push(-86_400_000);
    layer("tracks", lines, Some(times), starts)
}

fn untimed() -> ColumnarLayer {
    let mut l = tracks();
    l.name = "routes".to_string();
    l.vertex_times = None;
    l
}

fn special_f64() -> ColumnarLayer {
    layer(
        "odd",
        vec![
            vec![
                [-0.0, f64::NAN],
                [f64::INFINITY, f64::NEG_INFINITY],
                [1e300, 5e-324],
            ],
            vec![[0.1 + 0.2, 1.0 / 3.0], [std::f64::consts::PI, -0.0]],
        ],
        Some(vec![vec![T0, T0 + 1, T0 + 2], vec![T0 + 10, T0 + 5]]),
        vec![T0, T0 + 5],
    )
}

fn points() -> ColumnarLayer {
    ColumnarLayer {
        name: "pts".to_string(),
        feature_ids: vec![1, 2],
        start_times: vec![T0, T0 + 1],
        end_times: vec![T0, T0 + 1],
        geometry: GeometryColumn::Point(vec![[1.0, 2.0], [3.0, 4.0]]),
        vertex_times: None,
        vertex_values: None,
        vertex_value_matrix: None,
        triangles: None,
        polygon_parts: None,
        properties: vec![],
    }
}

fn cfg(codec: bool, quant: Option<f64>, max_step: u32) -> EncoderConfig {
    EncoderConfig {
        path_codec: codec,
        quantize_coords_m: quant,
        vertex_time_max_step_ms: max_step,
        ..EncoderConfig::default()
    }
}

/// Section tags of every layer of a self-contained frame.
fn section_tags(payload: &[u8]) -> Vec<Vec<u8>> {
    let rd16 = |o: usize| u16::from_le_bytes([payload[o], payload[o + 1]]) as usize;
    let mut pos = 6;
    let mut out = Vec::new();
    for _ in 0..rd16(4) {
        pos += 2 + rd16(pos);
        for _ in 0..2 {
            let kind = payload[pos];
            pos += 1 + if kind == 1 { 16 } else { 0 };
        }
        let sc = payload[pos] as usize;
        pos += 1;
        let mut tags = Vec::new();
        let mut lens = Vec::new();
        for _ in 0..sc {
            tags.push(payload[pos]);
            lens.push(u32::from_le_bytes(payload[pos + 1..pos + 5].try_into().unwrap()) as usize);
            pos += 5;
        }
        pos = (pos + 7) & !7;
        for len in lens {
            pos = (pos + len + 7) & !7;
        }
        out.push(tags);
    }
    out
}

fn assert_batches_bit_equal(a: &RecordBatch, b: &RecordBatch, what: &str) {
    assert_eq!(a.schema(), b.schema(), "{what}: schema");
    assert_eq!(a.num_rows(), b.num_rows(), "{what}: rows");
    for (i, f) in a.schema().fields().iter().enumerate() {
        // `ArrayData` equality compares f64 by value (NaN != NaN, 0.0 == -0.0),
        // so compare the raw buffers too for bit-exactness.
        let (x, y) = (a.column(i).to_data(), b.column(i).to_data());
        assert_eq!(
            flat_bits(&x),
            flat_bits(&y),
            "{what}: column '{}' bits",
            f.name()
        );
        assert_eq!(x.len(), y.len());
        assert_eq!(x.nulls(), y.nulls(), "{what}: column '{}' nulls", f.name());
    }
}

/// Every buffer of an `ArrayData` tree, sliced to its logical extent.
fn flat_bits(d: &arrow::array::ArrayData) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = d.buffers().iter().map(|b| b.as_slice().to_vec()).collect();
    for c in d.child_data() {
        out.extend(flat_bits(c));
    }
    out
}

fn decode(payload: &[u8]) -> Vec<DecodedLayer> {
    decode_tile(payload).expect("tile decodes")
}

/// With the non-codec side forced onto the absolute `Int64` vertex-time tier
/// (`vertex_time_max_step_ms = 0`), a codec tile decodes to EXACTLY the same
/// record batch — schema, metadata, every buffer — as a non-codec tile.
#[test]
fn codec_tile_decodes_identically_to_the_int64_tier() {
    for quant in [None, Some(1.0), Some(0.05)] {
        let layers = vec![tracks(), untimed(), special_f64(), points()];
        let with = encode_tile_with(&layers, &cfg(true, quant, 1000)).unwrap();
        let without = encode_tile_with(&layers, &cfg(false, quant, 0)).unwrap();
        assert_ne!(with, without);
        let (a, b) = (decode(&with), decode(&without));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.name, y.name);
            assert_batches_bit_equal(&x.batch, &y.batch, &format!("{} quant={quant:?}", x.name));
        }
        // Line layers carry the section, the point layer does not.
        let tags = section_tags(&with);
        for (i, l) in layers.iter().enumerate() {
            let has = tags[i].contains(&SECTION_PATH_CODEC);
            assert_eq!(
                has,
                matches!(l.geometry, GeometryColumn::LineString(_)),
                "{}",
                l.name
            );
            assert!(
                tags[i].windows(2).all(|w| w[0] < w[1]),
                "ascending tag order"
            );
        }
        assert!(section_tags(&without)
            .iter()
            .flatten()
            .all(|&t| t != SECTION_PATH_CODEC));
    }
}

/// Resolve any decoded `vertex_time` form to absolute ms per feature.
fn absolute_times(b: &RecordBatch) -> Vec<Option<Vec<i64>>> {
    let md = b.schema().metadata().clone();
    let num = |k: &str| md.get(k).map(|v| v.parse::<i64>().unwrap());
    let list = b
        .column_by_name("vertex_time")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap()
        .clone();
    let raw = |i: usize| -> Vec<i64> {
        let v = list.value(i);
        if let Some(a) = v.as_any().downcast_ref::<Int64Array>() {
            a.values().to_vec()
        } else if let Some(a) = v.as_any().downcast_ref::<UInt16Array>() {
            a.values().iter().map(|&x| x as i64).collect()
        } else {
            let a = v.as_any().downcast_ref::<UInt32Array>().unwrap();
            a.values().iter().map(|&x| x as i64).collect()
        }
    };
    let starts = b
        .column_by_name("start_time")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .clone();
    (0..list.len())
        .map(|i| {
            if list.is_null(i) {
                return None;
            }
            let r = raw(i);
            Some(
                match (
                    num("stt:vertex_time_origin_ms"),
                    num("stt:vertex_time_feature_step_ms"),
                ) {
                    (Some(o), _) => {
                        let s = num("stt:vertex_time_step_ms").unwrap();
                        r.iter().map(|d| o + d * s).collect()
                    }
                    (None, Some(s)) => r.iter().map(|d| starts.value(i) + d * s).collect(),
                    (None, None) => r,
                },
            )
        })
        .collect()
}

/// Against the build users actually compare with (`--vertex-time-precision 1`,
/// a delta tier at step 1): every column but `vertex_time` is bit-identical,
/// and `vertex_time` resolves to the same absolute ms.
#[test]
fn codec_tile_matches_an_exact_delta_tier_build() {
    for quant in [None, Some(1.0)] {
        let layers = vec![tracks(), special_f64()];
        let a = decode(&encode_tile_with(&layers, &cfg(true, quant, 1000)).unwrap());
        let b = decode(&encode_tile_with(&layers, &cfg(false, quant, 1)).unwrap());
        for (x, y) in a.iter().zip(&b) {
            let names: Vec<String> = y
                .batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            for name in &names {
                if name == "vertex_time" {
                    continue;
                }
                let (cx, cy) = (
                    x.batch.column_by_name(name).unwrap().to_data(),
                    y.batch.column_by_name(name).unwrap().to_data(),
                );
                assert_eq!(flat_bits(&cx), flat_bits(&cy), "{}: {name}", x.name);
                assert_eq!(
                    x.batch.schema().field_with_name(name).unwrap(),
                    y.batch.schema().field_with_name(name).unwrap()
                );
            }
            assert_eq!(
                absolute_times(&x.batch),
                absolute_times(&y.batch),
                "{}",
                x.name
            );
        }
    }
}

/// The default build truncates vertex times to the layer's u16 step; the
/// codec never does.
#[test]
fn codec_vertex_times_are_exact_where_the_default_truncates() {
    let l = tracks();
    let codec =
        decode(&encode_tile_with(std::slice::from_ref(&l), &cfg(true, None, 1000)).unwrap());
    let got = absolute_times(&codec[0].batch);
    let mut want: Vec<(i64, Option<Vec<i64>>)> = l
        .start_times
        .iter()
        .zip(l.vertex_times.as_ref().unwrap())
        .map(|(&s, t)| (s, (!t.is_empty()).then(|| t.clone())))
        .collect();
    want.sort_by_key(|w| w.0); // the encoder's stable start-time sort
    assert_eq!(got, want.into_iter().map(|w| w.1).collect::<Vec<_>>());
}

#[test]
fn observation_and_eligibility() {
    let on = cfg(true, None, 1000);
    let (_, obs) = encode_tile_observed(&[points()], &on).unwrap();
    assert!(!obs.path_codec, "no LineString layer, nothing observed");
    let (_, obs) = encode_tile_observed(&[points(), tracks()], &on).unwrap();
    assert!(obs.path_codec);
    let (_, obs) = encode_tile_observed(&[tracks()], &cfg(false, None, 1000)).unwrap();
    assert!(!obs.path_codec, "flag off");

    // Ineligible layers encode byte-identically to a non-codec build: a time
    // list that disagrees with its vertex count, and a time beyond ±2^52 ms.
    let mut ragged = tracks();
    ragged.vertex_times.as_mut().unwrap()[0].pop();
    let mut far = tracks();
    far.vertex_times.as_mut().unwrap()[1][0] = path_codec::MAX_TIME_MS;
    far.start_times[1] = path_codec::MAX_TIME_MS;
    far.end_times[1] = path_codec::MAX_TIME_MS;
    for l in [ragged, far] {
        let (with, obs) = encode_tile_observed(std::slice::from_ref(&l), &on).unwrap();
        assert!(!obs.path_codec);
        assert_eq!(
            with,
            encode_tile_with(&[l], &cfg(false, None, 1000)).unwrap()
        );
    }

    // An empty LineString layer is still a (trivial) codec layer.
    let empty = layer("e", vec![], Some(vec![]), vec![]);
    let (with, obs) = encode_tile_observed(std::slice::from_ref(&empty), &on).unwrap();
    assert!(obs.path_codec);
    let a = decode(&with);
    let b = decode(&encode_tile_with(&[empty], &cfg(false, None, 0)).unwrap());
    assert_batches_bit_equal(&a[0].batch, &b[0].batch, "empty layer");
}

/// Same layer, same bytes — in self-contained and template-hash mode — and
/// the codec layer's CORE template is the Int64-tier layer's template.
#[test]
fn codec_output_is_deterministic_and_does_not_fork_the_template() {
    let layers = vec![tracks(), untimed()];
    let a = encode_tile_with(&layers, &cfg(true, Some(1.0), 1000)).unwrap();
    let b = encode_tile_with(&layers, &cfg(true, Some(1.0), 1000)).unwrap();
    assert_eq!(a, b);

    let snapshot = |codec: bool, step: u32| {
        let c = Arc::new(TemplateCollector::new());
        let config = EncoderConfig {
            template_collector: Some(c.clone()),
            ..cfg(codec, Some(1.0), step)
        };
        let p1 = encode_tile_with(&layers, &config).unwrap();
        let p2 = encode_tile_with(&layers, &config).unwrap();
        assert_eq!(p1, p2);
        c.snapshot()
    };
    assert_eq!(snapshot(true, 1000), snapshot(false, 0));
}

/// A frame whose section disagrees with its CORE batch is a hard error.
#[test]
fn inconsistent_codec_frames_are_rejected() {
    let payload = encode_tile_with(&[tracks()], &cfg(true, None, 1000)).unwrap();
    // Locate the section bytes and corrupt its flags: clearing TIME no longer
    // matches the layer's `vertex_time` column; setting a reserved bit is
    // rejected by the section decoder.
    let toc0 = {
        let name_len = u16::from_le_bytes([payload[6], payload[7]]) as usize;
        8 + name_len + 2 + 1 // past name, two inline ref kinds, section count
    };
    let sc = payload[toc0 - 1] as usize;
    let mut pos = (toc0 + sc * 5 + 7) & !7;
    let mut codec_at = None;
    for i in 0..sc {
        let at = toc0 + i * 5;
        let len = u32::from_le_bytes(payload[at + 1..at + 5].try_into().unwrap()) as usize;
        if payload[at] == SECTION_PATH_CODEC {
            codec_at = Some(pos);
        }
        pos = (pos + len + 7) & !7;
    }
    let at = codec_at.expect("codec section");
    assert_eq!(payload[at], path_codec::PATH_CODEC_VERSION);
    for (flip, needle) in [
        (path_codec::FLAG_TIME, "TIME flag"),
        (path_codec::FLAG_DECIMAL, "DECIMAL flag"),
        (0x80, "reserved"),
    ] {
        let mut bad = payload.clone();
        bad[at + 1] ^= flip;
        let err = decode_tile(&bad).expect_err("must reject").to_string();
        assert!(err.contains(needle), "{needle}: {err}");
    }
    let mut bad = payload.clone();
    bad[at] = 9;
    assert!(decode_tile(&bad)
        .unwrap_err()
        .to_string()
        .contains("version"));
}

// ----------------------------------------------------------------------
// The committed conformance vectors (`conformance/vectors/path-codec/`) still
// say what this decoder does. Regenerate with
// `cargo run -p stt-core --example make-path-codec-vectors`.
// ----------------------------------------------------------------------

fn vectors_dir() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/vectors/path-codec");
    // Absent from the published crate tarball.
    p.join("sections.json").exists().then_some(p)
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn conformance_section_vectors_hold() {
    let Some(dir) = vectors_dir() else { return };
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("sections.json")).unwrap()).unwrap();
    for case in v["valid"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let bytes = unhex(case["section_hex"].as_str().unwrap());
        let l = path_codec::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let exp = &case["expected"];
        let counts: Vec<u32> = exp["counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(l.counts, counts, "{name}");
        let times: Option<Vec<i64>> = exp["times"].as_array().map(|a| {
            a.iter()
                .map(|t| t.as_str().unwrap().parse().unwrap())
                .collect()
        });
        assert_eq!(l.times, times, "{name}");
        let coords = exp["coords"].as_array().unwrap();
        match &l.coords {
            path_codec::PathCoords::Grid(g) => {
                assert_eq!(exp["mode"], "grid");
                assert_eq!(g.len(), coords.len());
                for (p, c) in g.iter().zip(coords) {
                    assert_eq!(p[0] as i64, c[0].as_i64().unwrap(), "{name}");
                    assert_eq!(p[1] as i64, c[1].as_i64().unwrap(), "{name}");
                }
            }
            path_codec::PathCoords::Decimal(f) => {
                assert_eq!(exp["mode"], "decimal");
                assert_eq!(f.len(), coords.len());
                for (p, c) in f.iter().zip(coords) {
                    for a in 0..2 {
                        let want = u64::from_str_radix(c[a].as_str().unwrap(), 16).unwrap();
                        assert_eq!(p[a].to_bits(), want, "{name}");
                    }
                }
            }
        }
        assert_eq!(path_codec::encode(&l).unwrap(), bytes, "{name}: re-encode");
    }
    for case in v["reject"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let bytes = unhex(case["section_hex"].as_str().unwrap());
        assert!(
            path_codec::decode(&bytes).is_err(),
            "{name} must be rejected"
        );
    }
}

#[test]
fn conformance_tile_vectors_hold() {
    let Some(dir) = vectors_dir() else { return };
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tiles.json")).unwrap()).unwrap();
    for tile in v["tiles"].as_array().unwrap() {
        let name = tile["name"].as_str().unwrap();
        let a = decode(&unhex(tile["payload_hex"].as_str().unwrap()));
        let b = decode(&unhex(tile["twin_payload_hex"].as_str().unwrap()));
        for (x, y) in a.iter().zip(&b) {
            assert_batches_bit_equal(&x.batch, &y.batch, name);
        }
        for (layer, exp) in a.iter().zip(tile["layers"].as_array().unwrap()) {
            assert_eq!(layer.name, exp["name"].as_str().unwrap());
            if exp["vertex_time"].is_null() {
                continue;
            }
            let vt = layer
                .batch
                .column_by_name("vertex_time")
                .unwrap()
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap()
                .clone();
            let valid: Vec<bool> = (0..vt.len()).map(|i| vt.is_valid(i)).collect();
            let want: Vec<bool> = exp["vertex_time"]["valid"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_bool().unwrap())
                .collect();
            assert_eq!(valid, want, "{name}/{}", layer.name);
        }
    }
}
