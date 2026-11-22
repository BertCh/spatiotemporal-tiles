//! The path codec end to end, through the real `stt-build` binary.
//!
//! Builds one synthetic trajectory source twice — `--path-codec`, and the
//! incumbent with exact vertex times (`--vertex-time-precision 1`) — and
//! asserts that every tile of the two archives decodes identically: every
//! column bit-exact except `vertex_time`, whose ABSOLUTE millisecond values
//! must agree (the incumbent ships them as a delta tier, the codec as
//! `List<Int64>`). Also pins the manifest side: `path-codec` is declared
//! exactly when a layer took the codec, and `stt-validate` accepts the
//! archive. Both coordinate modes (raw Float64 and `--quantize-coords`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, Int64Array, ListArray, RecordBatch, UInt16Array, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use stt_core::pack::PackedReader;

const T0: i64 = 1_700_000_000_000;

fn wkb_linestring(pts: &[(f64, f64)]) -> Vec<u8> {
    let mut b = vec![1u8];
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&(pts.len() as u32).to_le_bytes());
    for (x, y) in pts {
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
    }
    b
}

/// Drifter-like tracks with start/end times (the builder interpolates vertex
/// times, so clip and bucket-slice vertices carry off-grid milliseconds and
/// non-decimal coordinates).
fn write_tracks(path: &Path, count: usize) {
    let mut wkb = Vec::new();
    let (mut start, mut end) = (Vec::new(), Vec::new());
    for i in 0..count {
        let n = 5 + i % 17;
        let (x0, y0) = (-125.0 + (i % 23) as f64 * 1.7, 30.0 + (i % 11) as f64 * 1.3);
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|v| {
                let v = v as f64;
                (
                    ((x0 + v * 0.37 + (v * 0.7).sin() * 0.05) * 1e4).round() / 1e4,
                    ((y0 + v * 0.11) * 1e4).round() / 1e4,
                )
            })
            .collect();
        wkb.push(wkb_linestring(&pts));
        let s = T0 + (i as i64 % 40) * 1_800_000;
        start.push(s);
        end.push(s + 3_600_000 * (1 + i as i64 % 30));
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new("start_time", DataType::Int64, false),
        Field::new("end_time", DataType::Int64, false),
    ]));
    let geom: Vec<Option<&[u8]>> = wkb.iter().map(|w| Some(w.as_slice())).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(BinaryArray::from_opt_vec(geom)) as ArrayRef,
            Arc::new(Int64Array::from(start)) as ArrayRef,
            Arc::new(Int64Array::from(end)) as ArrayRef,
        ],
    )
    .unwrap();
    let mut w = ArrowWriter::try_new(fs::File::create(path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn bin(name: &str) -> PathBuf {
    match name {
        "stt-build" => PathBuf::from(env!("CARGO_BIN_EXE_stt-build")),
        _ => PathBuf::from(env!("CARGO_BIN_EXE_stt-validate")),
    }
}

fn build(input: &Path, out: &Path, extra: &[&str]) {
    let o = Command::new(bin("stt-build"))
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(out)
        .args([
            "--time-field",
            "start_time",
            "--end-time-field",
            "end_time",
            "--time-format",
            "unix-ms",
            "--min-zoom",
            "0",
            "--max-zoom",
            "4",
            "--temporal-bucket",
            "1d",
            "--workers",
            "2",
            "--name",
            "codec",
        ])
        .args(extra)
        .output()
        .expect("spawn stt-build");
    assert!(
        o.status.success(),
        "stt-build {extra:?} failed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

fn capabilities(out: &Path) -> Vec<String> {
    let m: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    m["capabilities"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
        .unwrap_or_default()
}

/// A decoded `vertex_time` column resolved to absolute ms per feature.
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
            let v = list.value(i);
            let raw: Vec<i64> = if let Some(a) = v.as_any().downcast_ref::<Int64Array>() {
                a.values().to_vec()
            } else if let Some(a) = v.as_any().downcast_ref::<UInt16Array>() {
                a.values().iter().map(|&x| x as i64).collect()
            } else {
                let a = v.as_any().downcast_ref::<UInt32Array>().unwrap();
                a.values().iter().map(|&x| x as i64).collect()
            };
            Some(
                match (
                    num("stt:vertex_time_origin_ms"),
                    num("stt:vertex_time_feature_step_ms"),
                ) {
                    (Some(o), _) => {
                        let s = num("stt:vertex_time_step_ms").unwrap();
                        assert_eq!(s, 1, "the incumbent side must be exact");
                        raw.iter().map(|d| o + d * s).collect()
                    }
                    (None, Some(s)) => {
                        assert_eq!(s, 1, "the incumbent side must be exact");
                        raw.iter().map(|d| starts.value(i) + d * s).collect()
                    }
                    (None, None) => raw,
                },
            )
        })
        .collect()
}

/// Every buffer of a column, recursively — bit-exact (f64 compared as bytes).
fn bits(d: &arrow::array::ArrayData) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = d.buffers().iter().map(|b| b.as_slice().to_vec()).collect();
    for c in d.child_data() {
        out.extend(bits(c));
    }
    out
}

/// Decode both archives tile by tile and compare. Returns the number of
/// codec-bearing layers seen.
fn assert_archives_decode_identically(codec: &Path, exact: &Path) -> usize {
    let a = PackedReader::open(codec.join("manifest.json")).unwrap();
    let b = PackedReader::open(exact.join("manifest.json")).unwrap();
    let key = |e: &stt_core::TileEntry| (e.zoom, e.x, e.y, e.time_start, e.time_end);
    let bmap: BTreeMap<_, _> = b.entries().iter().map(|e| (key(e), e.clone())).collect();
    assert_eq!(a.entries().len(), b.entries().len(), "same tile set");
    let mut codec_layers = 0;
    for ea in a.entries() {
        let eb = bmap.get(&key(ea)).expect("tile present in both archives");
        assert_eq!(ea.feature_count, eb.feature_count);
        let payload = a.read_payload(ea).unwrap();
        codec_layers += stt_core::arrow_tile::frame_v2_path_codec_sections(&payload)
            .unwrap()
            .iter()
            .filter(|s| s.is_some())
            .count();
        let la = a.decode_payload(&payload).unwrap();
        let lb = b.read_layers(eb).unwrap();
        assert_eq!(la.len(), lb.len());
        for (x, y) in la.iter().zip(&lb) {
            assert_eq!(x.name, y.name);
            let (sx, sy) = (x.batch.schema(), y.batch.schema());
            assert_eq!(sx.fields().len(), sy.fields().len(), "{}", x.name);
            for f in sy.fields() {
                if f.name() == "vertex_time" {
                    continue;
                }
                assert_eq!(sx.field_with_name(f.name()).unwrap(), f.as_ref());
                let (cx, cy) = (
                    x.batch.column_by_name(f.name()).unwrap().to_data(),
                    y.batch.column_by_name(f.name()).unwrap().to_data(),
                );
                assert_eq!(
                    bits(&cx),
                    bits(&cy),
                    "tile {:?} column {}",
                    key(ea),
                    f.name()
                );
                assert_eq!(cx.nulls(), cy.nulls());
            }
            if sy.field_with_name("vertex_time").is_ok() {
                assert_eq!(
                    absolute_times(&x.batch),
                    absolute_times(&y.batch),
                    "tile {:?} vertex_time",
                    key(ea)
                );
            }
        }
    }
    codec_layers
}

fn validate(out: &Path) {
    let o = Command::new(bin("stt-validate"))
        .arg(out.join("manifest.json"))
        .output()
        .expect("spawn stt-validate");
    assert!(
        o.status.success(),
        "stt-validate failed:\n{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
}

#[test]
fn codec_archive_decodes_identically_to_an_exact_vertex_time_build() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("tracks.parquet");
    write_tracks(&input, 300);
    for quant in [None, Some("1")] {
        let q: Vec<&str> = quant
            .map(|m| vec!["--quantize-coords", m])
            .unwrap_or_default();
        let codec = dir.path().join(format!("codec-{quant:?}"));
        let exact = dir.path().join(format!("exact-{quant:?}"));
        build(&input, &codec, &[&q[..], &["--path-codec"]].concat());
        build(
            &input,
            &exact,
            &[&q[..], &["--vertex-time-precision", "1"]].concat(),
        );

        assert!(capabilities(&codec).contains(&"path-codec".to_string()));
        assert!(!capabilities(&exact).contains(&"path-codec".to_string()));
        let n = assert_archives_decode_identically(&codec, &exact);
        assert!(n > 0, "no layer took the codec");
        validate(&codec);
    }
}

/// `--path-codec` on a source with no LineString layer declares nothing and
/// writes the same bytes as a build without the flag.
#[test]
fn path_codec_without_an_eligible_layer_declares_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("points.parquet");
    let pts: Vec<Vec<u8>> = (0..50)
        .map(|i| {
            let mut b = vec![1u8];
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&(-122.0 + i as f64 * 0.01f64).to_le_bytes());
            b.extend_from_slice(&37.0f64.to_le_bytes());
            b
        })
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new("start_time", DataType::Int64, false),
        Field::new("end_time", DataType::Int64, false),
    ]));
    let t: Vec<i64> = (0..50).map(|i| T0 + i * 60_000).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(BinaryArray::from_opt_vec(
                pts.iter().map(|w| Some(w.as_slice())).collect(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(t.clone())) as ArrayRef,
            Arc::new(Int64Array::from(t)) as ArrayRef,
        ],
    )
    .unwrap();
    let mut w = ArrowWriter::try_new(fs::File::create(&input).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    let (on, off) = (dir.path().join("on"), dir.path().join("off"));
    build(&input, &on, &["--path-codec"]);
    build(&input, &off, &[]);
    assert!(!capabilities(&on).contains(&"path-codec".to_string()));
    let packs = |d: &Path| {
        let mut v: Vec<(String, Vec<u8>)> = fs::read_dir(d.join("packs"))
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                (
                    e.file_name().to_string_lossy().into_owned(),
                    fs::read(e.path()).unwrap(),
                )
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(packs(&on), packs(&off), "same pack bytes");
}
