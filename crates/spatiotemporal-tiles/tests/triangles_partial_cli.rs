//! TB-12 end-to-end — per-feature triangle emission through the real binary.
//!
//! The unit tests in `stt_build::columnar` pin which features get baked. This
//! one pins the three things only a whole build can show:
//!
//!  1. a MIXED polygon layer declares `triangles-partial` in the manifest —
//!     without it a pre-backfill reader opens the archive and silently drops
//!     every single-ring polygon;
//!  2. a build where nothing mixes declares NOTHING, so the capability never
//!     locks readers out of an archive that does not need them;
//!  3. the emission actually saves bytes, which is the point of the item.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

const T0: i64 = 1_700_000_000_000;
const HOUR_MS: i64 = 3_600_000;

/// WKB for a polygon with the given rings (exterior first). Little-endian.
fn wkb_polygon(rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
    let mut b = Vec::new();
    b.push(1);
    b.extend_from_slice(&3u32.to_le_bytes()); // Polygon
    b.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for ring in rings {
        b.extend_from_slice(&(ring.len() as u32).to_le_bytes());
        for (x, y) in ring {
            b.extend_from_slice(&x.to_le_bytes());
            b.extend_from_slice(&y.to_le_bytes());
        }
    }
    b
}

fn square(cx: f64, cy: f64, r: f64) -> Vec<(f64, f64)> {
    vec![
        (cx - r, cy - r),
        (cx + r, cy - r),
        (cx + r, cy + r),
        (cx - r, cy + r),
        (cx - r, cy - r),
    ]
}

/// `holed_every`: emit a hole-bearing polygon every Nth feature. `None` = never,
/// which makes every feature single-ring; `Some(1)` = always, so nothing mixes
/// in the other direction.
fn write_polygons(path: &Path, count: i64, holed_every: Option<i64>) {
    let mut wkb: Vec<Vec<u8>> = Vec::with_capacity(count as usize);
    let mut ts: Vec<i64> = Vec::with_capacity(count as usize);
    for i in 0..count {
        // A deterministic lattice; radius is well inside one z8 tile.
        let cx = -122.6 + ((i % 20) as f64) * 0.03;
        let cy = 37.6 + ((i / 20) as f64) * 0.03;
        let holed = holed_every.is_some_and(|n| i % n == 0);
        let rings = if holed {
            vec![square(cx, cy, 0.010), square(cx, cy, 0.004)]
        } else {
            vec![square(cx, cy, 0.010)]
        };
        wkb.push(wkb_polygon(&rings));
        ts.push(T0 + (i % 12) * HOUR_MS);
    }

    let geom: Vec<Option<&[u8]>> = wkb.iter().map(|w| Some(w.as_slice())).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new("timestamp", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(BinaryArray::from_opt_vec(geom)) as ArrayRef,
            Arc::new(Int64Array::from(ts)) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn stt_build_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stt-build"))
}

fn run_build(input: &Path, out: &Path, extra: &[&str]) {
    let output = Command::new(stt_build_bin())
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(out)
        .args([
            "--time-field",
            "timestamp",
            "--time-format",
            "unix-ms",
            "--min-zoom",
            "8",
            "--max-zoom",
            "10",
            "--temporal-bucket",
            "1h",
            "--workers",
            "2",
            "--name",
            "tb12",
        ])
        .args(extra)
        .output()
        .expect("failed to spawn stt-build");
    assert!(
        output.status.success(),
        "stt-build failed ({}):\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

fn capabilities(out: &Path) -> Vec<String> {
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    manifest
        .get("capabilities")
        .and_then(|c| c.as_array())
        .map(|a| {
            a.iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

/// Total bytes of the pack files — the wire cost the item is spending.
fn pack_bytes(out: &Path) -> u64 {
    fs::read_dir(out.join("packs"))
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum()
}

#[test]
fn a_mixed_polygon_layer_declares_triangles_partial_and_saves_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("mixed.parquet");
    // Every 8th feature holed: the layer bakes some lists and empties the rest.
    write_polygons(&input, 400, Some(8));

    let partial = dir.path().join("partial");
    let full = dir.path().join("full");
    run_build(&input, &partial, &[]);
    run_build(&input, &full, &["--pre-tessellate"]);

    assert!(
        capabilities(&partial).contains(&"triangles-partial".to_string()),
        "a mixed layer MUST declare the capability, else single-ring polygons \
         silently vanish in every pre-backfill reader; got {:?}",
        capabilities(&partial)
    );
    assert!(
        !capabilities(&full).contains(&"triangles-partial".to_string()),
        "--pre-tessellate bakes everything, so nothing is mixed and nothing is owed"
    );

    let (p, f) = (pack_bytes(&partial), pack_bytes(&full));
    assert!(
        p < f,
        "per-feature emission must cost fewer bytes than baking everything \
         (partial {p} vs pre-tessellate {f})"
    );
}

#[test]
fn a_layer_that_mixes_nothing_declares_nothing() {
    let dir = tempfile::tempdir().unwrap();

    // (a) Every feature single-ring: no triangle column is emitted at all.
    let all_simple = dir.path().join("simple.parquet");
    write_polygons(&all_simple, 120, None);
    let out_simple = dir.path().join("simple");
    run_build(&all_simple, &out_simple, &[]);
    assert!(
        !capabilities(&out_simple).contains(&"triangles-partial".to_string()),
        "no triangle column is emitted at all, so nothing is owed; got {:?}",
        capabilities(&out_simple)
    );

    // (b) Every feature holed: every list is baked, so again nothing is mixed
    // and older readers must stay able to open it.
    let all_holed = dir.path().join("holed.parquet");
    write_polygons(&all_holed, 120, Some(1));
    let out_holed = dir.path().join("holed");
    run_build(&all_holed, &out_holed, &[]);
    assert!(
        !capabilities(&out_holed).contains(&"triangles-partial".to_string()),
        "a uniformly-baked layer emits the incumbent bytes and must not lock \
         readers out; got {:?}",
        capabilities(&out_holed)
    );
}

/// Not an assertion — a measurement, printed with `--nocapture`, so the byte
/// saving TB-12 claims is a number someone can check rather than a belief.
#[test]
fn report_the_byte_saving() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("mixed.parquet");
    write_polygons(&input, 400, Some(8));
    let partial = dir.path().join("partial");
    let full = dir.path().join("full");
    run_build(&input, &partial, &[]);
    run_build(&input, &full, &["--pre-tessellate"]);
    let (p, f) = (pack_bytes(&partial), pack_bytes(&full));
    println!(
        "TB-12 packs: partial {p} B, pre-tessellate {f} B, saved {} B ({:.1}%)",
        f - p,
        100.0 * (f - p) as f64 / f as f64
    );
}
