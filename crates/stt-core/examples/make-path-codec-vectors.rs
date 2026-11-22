//! Regenerate the path-codec conformance vectors under
//! `conformance/vectors/path-codec/` (packed-format §5.2.7).
//!
//! Three artifacts, all produced by the reference writer and checked against
//! the reference decoder before they are written:
//!
//! - `sections.json` — bare `PATH_CODEC` section bytes with the exact values a
//!   decoder must produce, plus must-reject sections;
//! - `tiles.json` — whole self-contained layer frames (uncompressed tile
//!   payloads, inline schemas) carrying codec layers, each with its non-codec
//!   twin (the same layers encoded on the absolute `List<Int64>` vertex-time
//!   tier) and the expected decoded `geometry` / `vertex_time` columns;
//! - `grid/` and `decimal/` — two small packed archives declaring the
//!   `path-codec` capability (world-grid via `--quantize-coords 1`, and raw
//!   Float64), for an end-to-end open → decode test.
//!
//! Numbers are spelled so a JavaScript test can check them bit-exactly: f64 as
//! the 16-hex-digit big-endian bit pattern, i64 times as decimal strings,
//! Int32 grid indices as JSON numbers.
//!
//! Usage: `cargo run -p stt-core --example make-path-codec-vectors [-- <out-dir>]`

use std::path::{Path, PathBuf};

use arrow::array::{Array, FixedSizeListArray, Float64Array, Int32Array, Int64Array, ListArray};
use serde_json::{json, Value};
use stt_core::arrow_tile::path_codec::{self, PathCoords, PathLayer};
use stt_core::arrow_tile::{
    decode_tile, encode_tile_observed, encode_tile_with, frame_v2_path_codec_sections,
    ColumnarLayer, DecodedLayer, EncoderConfig, GeometryColumn, PropertyColumn,
};
use stt_core::metadata::Metadata;
use stt_core::pack::CAPABILITY_PATH_CODEC;
use stt_core::{BlobOrdering, PackWriter, PackedReader, TileId};

const T0: i64 = 1_767_225_600_000; // 2026-01-01T00:00:00Z

fn uvar(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
    out
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn f64_hex(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

fn line_layer(
    name: &str,
    lines: Vec<Vec<[f64; 2]>>,
    times: Option<Vec<Vec<i64>>>,
    starts: Vec<i64>,
) -> ColumnarLayer {
    let n = lines.len();
    ColumnarLayer {
        name: name.to_string(),
        feature_ids: (0..n as u64).map(|i| 1000 + i).collect(),
        end_times: starts.iter().map(|s| s + 3_600_000).collect(),
        start_times: starts,
        geometry: GeometryColumn::LineString(lines),
        vertex_times: times,
        vertex_values: None,
        vertex_value_matrix: None,
        triangles: None,
        polygon_parts: None,
        properties: vec![(
            "speed".to_string(),
            PropertyColumn::Numeric((0..n).map(|i| Some(2.5 * i as f64)).collect()),
        )],
    }
}

/// Five-decimal GPS-like tracks: regular 60 s fixes (unit 60 000), one
/// constant-speed feature with IRREGULAR sampling (TIME_VEL), an empty
/// feature, a single-vertex feature, a feature with unsorted vertex times,
/// a clip vertex with a non-decimal coordinate and an off-unit time, and a
/// negative (pre-1970) time.
fn tracks(special_values: bool, negative: bool) -> ColumnarLayer {
    let d = |k: i64| k as f64 / 1e5;
    let mut lines = Vec::new();
    let mut times = Vec::new();
    let mut starts = Vec::new();

    // 0: regular fixes, gentle curve.
    let l: Vec<[f64; 2]> = (0..6)
        .map(|i| [d(-12_241_940 + 37 * i + i * i), d(3_777_490 + 21 * i)])
        .collect();
    lines.push(l);
    times.push((0..6).map(|i| T0 + 60_000 * i).collect());
    // 1: constant velocity, irregular dt → TIME_VEL.
    let dts = [60_000i64, 180_000, 60_000, 420_000, 120_000, 240_000];
    let mut t = vec![T0 + 3_600_000];
    for dt in dts {
        t.push(t.last().unwrap() + dt);
    }
    let l: Vec<[f64; 2]> = t
        .iter()
        .map(|&ti| {
            let s = (ti - t[0]) / 60_000;
            [d(-12_200_000 + 4_100 * s), d(3_750_000 - 2_300 * s)]
        })
        .collect();
    lines.push(l);
    times.push(t);
    // 2: empty feature.
    lines.push(vec![]);
    times.push(vec![]);
    // 3: single vertex.
    lines.push(vec![[d(-12_190_000), d(3_740_000)]]);
    times.push(vec![T0 + 7_200_000]);
    // 4: unsorted vertex times + a clip vertex (non-decimal x, off-unit t).
    lines.push(vec![
        [-122.123_456_789_012_3, d(3_760_000)],
        [d(-12_212_000), d(3_760_500)],
        [d(-12_211_000), d(3_761_000)],
        [d(-12_210_000), d(3_761_500)],
    ]);
    times.push(vec![
        T0 + 10_800_017,
        T0 + 10_860_000,
        T0 + 10_800_000,
        T0 + 10_920_000,
    ]);
    // 5: negative times.
    if negative {
        lines.push(vec![[d(100_000), d(-200_000)], [d(100_100), d(-200_050)]]);
        times.push(vec![-86_400_000, -86_340_000]);
    }
    if special_values {
        // 6: -0.0, NaN, ±inf — exceptions in decimal mode.
        lines.push(vec![
            [-0.0, 0.0],
            [f64::NAN, d(10)],
            [f64::INFINITY, f64::NEG_INFINITY],
        ]);
        times.push(vec![T0, T0 + 60_000, T0 + 120_000]);
    }
    for t in &times {
        starts.push(t.iter().copied().min().unwrap_or(T0));
    }
    line_layer("tracks", lines, Some(times), starts)
}

fn untimed_routes() -> ColumnarLayer {
    let d = |k: i64| k as f64 / 1e5;
    line_layer(
        "routes",
        vec![
            (0..5)
                .map(|i| [d(-7_360_000 + 300 * i), d(4_550_000 + 120 * i)])
                .collect(),
            vec![],
            vec![[d(-7_350_000), d(4_551_000)], [d(-7_349_000), d(4_552_000)]],
        ],
        None,
        vec![T0, T0, T0 + 1],
    )
}

fn points() -> ColumnarLayer {
    ColumnarLayer {
        name: "stops".to_string(),
        feature_ids: vec![1, 2],
        start_times: vec![T0, T0 + 1],
        end_times: vec![T0, T0 + 1],
        geometry: GeometryColumn::Point(vec![[-122.4, 37.7], [-122.5, 37.8]]),
        vertex_times: None,
        vertex_values: None,
        vertex_value_matrix: None,
        triangles: None,
        polygon_parts: None,
        properties: vec![],
    }
}

fn cfg(codec: bool, quant: Option<f64>) -> EncoderConfig {
    EncoderConfig {
        path_codec: codec,
        quantize_coords_m: quant,
        // The non-codec twin goes to the absolute Int64 tier, the shape a
        // codec layer decodes to.
        vertex_time_max_step_ms: if codec { 1000 } else { 0 },
        ..EncoderConfig::default()
    }
}

// ------------------------------------------------------- section parsing --

/// The predictor tags and exception count of a section (generator checks).
fn tags_and_exceptions(s: &[u8]) -> (Vec<u8>, usize) {
    let mut p = 0usize;
    let mut uv = || {
        let mut v = 0u64;
        let mut sh = 0;
        loop {
            let b = s[p];
            p += 1;
            v |= ((b & 0x7f) as u64) << sh;
            sh += 7;
            if b & 0x80 == 0 {
                return v;
            }
        }
    };
    let _version = uv();
    let flags = uv() as u8;
    let nf = uv() as usize;
    let nv = uv() as usize;
    let decimal = flags & path_codec::FLAG_DECIMAL != 0;
    let time = flags & path_codec::FLAG_TIME != 0;
    if decimal {
        uv(); // e (one byte, < 0x80)
    }
    if time {
        uv();
        uv();
    }
    for _ in 0..nf {
        uv();
    }
    let tags: Vec<u8> = (0..nf).map(|_| uv() as u8).collect();
    let mut exc = 0;
    if decimal {
        for _ in 0..(if time { 2 * nv } else { 0 }) + 2 * nv {
            uv();
        }
        exc = uv() as usize;
    }
    (tags, exc)
}

fn expected_section(l: &PathLayer) -> Value {
    let coords: Value = match &l.coords {
        PathCoords::Grid(g) => json!(g.iter().map(|p| json!([p[0], p[1]])).collect::<Vec<_>>()),
        PathCoords::Decimal(f) => json!(f
            .iter()
            .map(|p| json!([f64_hex(p[0]), f64_hex(p[1])]))
            .collect::<Vec<_>>()),
    };
    json!({
        "mode": match l.coords { PathCoords::Grid(_) => "grid", PathCoords::Decimal(_) => "decimal" },
        "counts": l.counts,
        "coords": coords,
        "times": l.times.as_ref().map(|t| t.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
    })
}

// --------------------------------------------------------- column dumping --

fn dump_layer(l: &DecodedLayer, codec: bool) -> Value {
    let b = &l.batch;
    let Some(geom) = b
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
    else {
        // A point layer (not a codec candidate): no column dump.
        return json!({
            "name": l.name,
            "rows": b.num_rows(),
            "path_codec": codec,
            "columns": b.schema().fields().iter().map(|f| f.name().clone()).collect::<Vec<_>>(),
            "geometry": null,
            "vertex_time": null,
        });
    };
    let offsets: Vec<i32> = geom.value_offsets().to_vec();
    let fsl = geom
        .values()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let leaf = fsl.values();
    let (leaf_type, values): (&str, Value) =
        if let Some(a) = leaf.as_any().downcast_ref::<Int32Array>() {
            ("Int32", json!(a.values().to_vec()))
        } else if let Some(a) = leaf.as_any().downcast_ref::<Float64Array>() {
            (
                "Float64",
                json!(a.values().iter().map(|v| f64_hex(*v)).collect::<Vec<_>>()),
            )
        } else {
            ("other", Value::Null)
        };
    let vt = b.column_by_name("vertex_time").map(|c| {
        let list = c.as_any().downcast_ref::<ListArray>().unwrap();
        let vals = list.values().as_any().downcast_ref::<Int64Array>().unwrap();
        json!({
            "offsets": list.value_offsets().to_vec(),
            "valid": (0..list.len()).map(|i| list.is_valid(i)).collect::<Vec<_>>(),
            "values": vals.values().iter().map(|v| v.to_string()).collect::<Vec<_>>(),
        })
    });
    json!({
        "name": l.name,
        "rows": b.num_rows(),
        "path_codec": codec,
        "columns": b.schema().fields().iter().map(|f| f.name().clone()).collect::<Vec<_>>(),
        "geometry": { "leaf": leaf_type, "offsets": offsets, "values": values },
        "vertex_time": vt,
    })
}

fn tile_vector(name: &str, layers: &[ColumnarLayer], quant: Option<f64>) -> (Value, Vec<Vec<u8>>) {
    let (payload, obs) = encode_tile_observed(layers, &cfg(true, quant)).unwrap();
    assert!(obs.path_codec, "{name}: no codec layer");
    let twin = encode_tile_with(layers, &cfg(false, quant)).unwrap();
    let a = decode_tile(&payload).unwrap();
    let b = decode_tile(&twin).unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        // The reference decoder's own differential: codec == Int64 tier, with
        // f64 compared through its bits.
        assert_eq!(x.batch.schema(), y.batch.schema(), "{name}/{}", x.name);
        for i in 0..x.batch.num_columns() {
            let (p, q) = (x.batch.column(i).to_data(), y.batch.column(i).to_data());
            assert_eq!(p, q, "{name}/{} column {i}", x.name);
        }
        assert_eq!(
            dump_layer(x, false)["geometry"],
            dump_layer(y, false)["geometry"],
            "{name}/{}: geometry bits",
            x.name
        );
    }
    let sections: Vec<Option<&[u8]>> = frame_v2_path_codec_sections(&payload).unwrap();
    let v = json!({
        "name": name,
        "quantize_coords_m": quant,
        "payload_hex": hex(&payload),
        "twin_payload_hex": hex(&twin),
        "layers": a
            .iter()
            .zip(&sections)
            .map(|(l, s)| dump_layer(l, s.is_some()))
            .collect::<Vec<_>>(),
    });
    let raw: Vec<Vec<u8>> = sections.iter().flatten().map(|s| s.to_vec()).collect();
    (v, raw)
}

fn write_dataset(
    dir: &Path,
    name: &str,
    quant: Option<f64>,
    tiles: &[(TileId, Vec<ColumnarLayer>)],
) {
    let _ = std::fs::remove_dir_all(dir);
    let base = EncoderConfig {
        path_codec: true,
        quantize_coords_m: quant,
        ..EncoderConfig::default()
    };
    let mut w = PackWriter::create(dir, BlobOrdering::SpatialMajor, 4 * 1024)
        .unwrap()
        .with_encoder_config(base);
    let mut observed = false;
    let (mut t_min, mut t_max) = (i64::MAX, i64::MIN);
    for (id, layers) in tiles {
        let (payload, obs) = encode_tile_observed(layers, &w.encoder_config()).unwrap();
        observed |= obs.path_codec;
        let lo = layers
            .iter()
            .flat_map(|l| l.start_times.iter())
            .min()
            .copied()
            .unwrap();
        let hi = layers
            .iter()
            .flat_map(|l| l.end_times.iter())
            .max()
            .copied()
            .unwrap();
        let rows: usize = layers.iter().map(|l| l.feature_ids.len()).sum();
        w.add_tile_full(
            id,
            lo,
            hi,
            Some(lo),
            rows as u32,
            Some(86_400_000),
            &payload,
        )
        .unwrap();
        t_min = t_min.min(lo);
        t_max = t_max.max(hi);
    }
    assert!(observed);
    w.declare_capability(CAPABILITY_PATH_CODEC);
    let meta = Metadata::new(name)
        .with_description("path-codec conformance vector (packed-format §5.2.7)")
        .with_zoom_levels(tiles[0].0.z, tiles[0].0.z)
        .with_temporal_bucket_ms(86_400_000)
        .with_time_range(stt_core::types::TimeRange::new(t_min as u64, t_max as u64));
    w.finalize(&meta).unwrap();
    // Round-trip through the real reader: capability accepted, every tile decodes.
    let r = PackedReader::open(dir.join("manifest.json")).unwrap();
    for e in r.entries() {
        r.read_layers(e).unwrap();
    }
}

fn main() {
    let out: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/vectors/path-codec")
        });
    std::fs::create_dir_all(&out).unwrap();

    // ---- tiles --------------------------------------------------------------
    let grid_layers = vec![tracks(false, true), untimed_routes(), points()];
    let dec_layers = vec![tracks(true, true), untimed_routes()];
    let (grid_tile, grid_secs) = tile_vector("grid-tile", &grid_layers, Some(1.0));
    let (dec_tile, dec_secs) = tile_vector("decimal-tile", &dec_layers, None);

    // The vector set must actually exercise what it claims to.
    let (tags, _) = tags_and_exceptions(&grid_secs[0]);
    assert!(
        tags.contains(&path_codec::TAG_TIME_VEL),
        "grid: no TIME_VEL feature"
    );
    let (tags, exc) = tags_and_exceptions(&dec_secs[0]);
    assert!(
        tags.contains(&path_codec::TAG_TIME_VEL),
        "decimal: no TIME_VEL feature"
    );
    assert!(exc >= 5, "decimal: expected exceptions, got {exc}");

    // ---- sections ------------------------------------------------------------
    let mut valid = Vec::new();
    let mut push_valid = |name: &str, note: &str, bytes: Vec<u8>| {
        let decoded = path_codec::decode(&bytes).unwrap();
        assert_eq!(
            path_codec::encode(&decoded).unwrap(),
            bytes,
            "{name}: canonical"
        );
        let (tags, exc) = tags_and_exceptions(&bytes);
        valid.push(json!({
            "name": name,
            "note": note,
            "section_hex": hex(&bytes),
            "tags": tags,
            "exception_count": exc,
            "expected": expected_section(&decoded),
        }));
    };
    push_valid(
        "grid-tracks",
        "world-grid (Int32) tracks with times: TIME_VEL feature, empty feature, single vertex, unsorted and negative times",
        grid_secs[0].clone(),
    );
    push_valid(
        "grid-routes-no-time",
        "world-grid, no TIME flag",
        grid_secs[1].clone(),
    );
    push_valid(
        "decimal-tracks",
        "decimal (Float64) tracks with times: exceptions (clip vertex, -0.0, NaN, ±inf), TIME_VEL feature, empty feature",
        dec_secs[0].clone(),
    );
    push_valid(
        "decimal-routes-no-time",
        "decimal, no TIME flag",
        dec_secs[1].clone(),
    );
    let non_decimal: Vec<[f64; 2]> = (1..=6)
        .map(|i| {
            let x = (i as f64).sqrt() * std::f64::consts::PI;
            [x, -x / 7.0]
        })
        .collect();
    push_valid(
        "decimal-all-exceptions",
        "no value is decimal at any exponent: e = 7 and every value carries a patch",
        path_codec::encode(&PathLayer {
            counts: vec![2, 0, 4],
            coords: PathCoords::Decimal(non_decimal),
            times: Some(vec![5, 4, -3, -3, 1_000_000_007, 2]),
        })
        .unwrap(),
    );
    push_valid(
        "grid-int32-extremes",
        "world-grid indices at the Int32 extremes (residuals up to ±2^32)",
        path_codec::encode(&PathLayer {
            counts: vec![4],
            coords: PathCoords::Grid(vec![
                [i32::MIN, i32::MAX],
                [i32::MAX, i32::MIN],
                [0, 0],
                [-1, 1],
            ]),
            times: None,
        })
        .unwrap(),
    );
    push_valid(
        "empty-layer",
        "zero features",
        path_codec::encode(&PathLayer {
            counts: vec![],
            coords: PathCoords::Grid(vec![]),
            times: Some(vec![]),
        })
        .unwrap(),
    );

    // ---- must-reject -----------------------------------------------------------
    let base = dec_secs[0].clone();
    let mut reject = Vec::new();
    let mut push_reject = |name: &str, reason: &str, bytes: Vec<u8>| {
        assert!(
            path_codec::decode(&bytes).is_err(),
            "{name} must be rejected"
        );
        reject.push(json!({ "name": name, "reason": reason, "section_hex": hex(&bytes) }));
    };
    let mut b = base.clone();
    b.push(0);
    push_reject("trailing-byte", "trailing bytes after the last field", b);
    let mut b = base.clone();
    b[0] = 2;
    push_reject("bad-version", "version != 1", b);
    let mut b = base.clone();
    b[1] |= 0x04;
    push_reject("reserved-flag", "reserved flag bit 2 set", b);
    push_reject(
        "remainder-ge-unit",
        "r >= U (U = 10, r = 10)",
        vec![1, path_codec::FLAG_TIME, 1, 1, 10, 0, 1, 0, 0, 10, 0, 0],
    );
    push_reject(
        "unit-zero",
        "U = 0",
        vec![1, path_codec::FLAG_TIME, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    );
    push_reject(
        "unknown-tag",
        "predictor tag 3",
        vec![1, 0, 1, 1, 1, 3, 0, 0],
    );
    push_reject(
        "time-vel-without-time",
        "TIME_VEL tag in a section without TIME",
        vec![1, 0, 1, 3, 3, 2, 0, 0, 2, 2, 2, 2],
    );
    push_reject(
        "count-mismatch",
        "vertex_count (2) != sum of counts (1)",
        vec![1, 0, 1, 2, 1, 0, 0, 0, 0, 0],
    );
    push_reject(
        "exception-index-out-of-range",
        "exception index 2 with only 2 values",
        vec![1, path_codec::FLAG_DECIMAL, 1, 1, 0, 1, 0, 0, 0, 1, 2, 0, 0],
    );
    push_reject(
        "exception-index-not-increasing",
        "second exception delta is 0",
        vec![
            1,
            path_codec::FLAG_DECIMAL,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    );
    push_reject(
        "patch-half-too-large",
        "patch low half = 2^32",
        vec![
            1,
            path_codec::FLAG_DECIMAL,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0x80,
            0x80,
            0x80,
            0x80,
            0x10,
            0,
        ],
    );
    push_reject(
        "decimal-exponent-16",
        "e = 16 (> 15)",
        vec![1, path_codec::FLAG_DECIMAL, 1, 1, 16, 1, 0, 0, 0, 0],
    );
    push_reject(
        "grid-int32-overflow",
        "world-grid coordinate 2^31 does not fit Int32",
        vec![1, 0, 1, 1, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x10, 0],
    );
    push_reject(
        "integer-bound",
        "a residual of 2^53 is outside ±(2^53 - 1) (packed-format §5.2.7 integer bound)",
        [
            vec![1, path_codec::FLAG_DECIMAL, 1, 1, 0, 1, 0],
            uvar(1u64 << 54), // zigzag(2^53)
            vec![0, 0],
        ]
        .concat(),
    );
    push_reject(
        "truncated",
        "section cut mid-stream",
        base[..base.len() / 2].to_vec(),
    );
    push_reject(
        "count-exceeds-section",
        "feature_count larger than the section can hold",
        vec![1, 0, 0xff, 0xff, 0xff, 0x7f, 0],
    );

    let sections = json!({
        "format": "stt-path-codec-section-vectors",
        "version": 1,
        "spec": "docs/spec/stt-packed-format.md §5.2.7",
        "encoding": "section_hex: lowercase hex of the PATH_CODEC section bytes. Decimal coords: 16-hex-digit IEEE-754 bit patterns (big-endian, as u64::to_bits / DataView.getBigUint64). Grid coords: JSON numbers. Times: decimal strings (Unix ms, i64). tags/exception_count are informational.",
        "valid": valid,
        "reject": reject,
    });
    let tiles = json!({
        "format": "stt-path-codec-tile-vectors",
        "version": 1,
        "spec": "docs/spec/stt-packed-format.md §5.2.7",
        "encoding": "payload_hex: an UNCOMPRESSED self-contained (inline-schema) layer frame v2 tile payload. twin_payload_hex: the same layers encoded WITHOUT the codec on the absolute List<Int64> vertex-time tier; decoding both must give identical columns. layers[*].geometry/vertex_time are the decoded columns (offsets, validity, values; Float64 as bit-pattern hex, Int32 as numbers, times as decimal strings).",
        "tiles": [grid_tile, dec_tile],
    });
    std::fs::write(
        out.join("sections.json"),
        serde_json::to_string_pretty(&sections).unwrap() + "\n",
    )
    .unwrap();
    std::fs::write(
        out.join("tiles.json"),
        serde_json::to_string_pretty(&tiles).unwrap() + "\n",
    )
    .unwrap();

    // ---- datasets ---------------------------------------------------------------
    let day = 86_400_000u64;
    let t0 = T0 as u64;
    write_dataset(
        &out.join("grid"),
        "path-codec-grid",
        Some(1.0),
        &[
            (
                TileId::new(4, 2, 6, t0),
                vec![tracks(false, false), points()],
            ),
            (TileId::new(4, 4, 5, t0), vec![untimed_routes()]),
        ],
    );
    write_dataset(
        &out.join("decimal"),
        "path-codec-decimal",
        None,
        &[
            (TileId::new(4, 2, 6, t0), vec![tracks(false, false)]),
            (TileId::new(4, 4, 5, t0 + day), vec![untimed_routes()]),
        ],
    );
    println!("wrote {}", out.display());
}
