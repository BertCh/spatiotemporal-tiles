//! `stt-optimize export` round-trip: build a packed archive, export it to
//! GeoParquet, read the file back with the `parquet` crate, and assert the
//! geometry, the timestamps and the properties all survive.
//!
//! The failure this suite exists to catch is a silent loss on the way OUT —
//! most of all a time column that arrives as an untyped integer, or not at
//! all. Every assertion therefore compares against the exact values written
//! into the archive, never against "some rows came back".

use arrow::array::{Array, BinaryArray, Float64Array, StringArray, TimestampMillisecondArray};
use arrow::datatypes::{DataType, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use stt_core::arrow_tile::{
    encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn, PropertyColumn,
};
use stt_core::curve::BlobOrdering;
use stt_core::metadata::Metadata;
use stt_core::pack::{PackWriter, PACKED_FORMAT_VERSION_V2};
use stt_core::projection::lonlat_to_tile;
use stt_core::tile::TileId;
use stt_optimize::export::{self, ExportOptions, GeometryEncoding};
use stt_optimize::PackedTileset;

/// The zoom every fixture writes its base tiles at.
const Z: u8 = 10;
/// Base temporal bucket of the fixtures (1 hour).
const BUCKET_MS: u64 = 3_600_000;

/// Three Montreal points, one per tile-worth of separation, with distinct
/// times and properties.
fn point_fixture() -> Vec<(u64, [f64; 2], i64, i64, f64, &'static str)> {
    vec![
        (
            1,
            [-73.60, 45.50],
            1_700_000_000_000,
            1_700_000_060_000,
            12.5,
            "alpha",
        ),
        (
            2,
            [-73.40, 45.55],
            1_700_003_600_000,
            1_700_003_660_000,
            7.25,
            "beta",
        ),
        (
            3,
            [-73.20, 45.60],
            1_700_007_200_000,
            1_700_007_260_000,
            99.0,
            "gamma",
        ),
    ]
}

/// Build a packed archive: one point layer, one point per tile.
///
/// `cfg` lets a caller turn coordinate / attribute quantization on, which is
/// the case the exporter must undo before writing WKB.
fn build_points(dir: &Path, cfg_of: impl Fn(&PackWriter) -> EncoderConfig) -> PathBuf {
    let out = dir.join("points");
    let mut w = PackWriter::create(&out, BlobOrdering::TimeMajor, 1 << 20)
        .unwrap()
        .with_format_version(PACKED_FORMAT_VERSION_V2);
    for (id, [lon, lat], start, end, num, cat) in point_fixture() {
        let layer = ColumnarLayer {
            name: "default".to_string(),
            feature_ids: vec![id],
            start_times: vec![start],
            end_times: vec![end],
            geometry: GeometryColumn::Point(vec![[lon, lat]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "speed".to_string(),
                    PropertyColumn::Numeric(vec![Some(num)]),
                ),
                (
                    "name".to_string(),
                    PropertyColumn::Categorical(vec![Some(cat.to_string())]),
                ),
            ],
        };
        let payload = encode_tile_with(&[layer], &cfg_of(&w)).unwrap();
        let (x, y) = lonlat_to_tile(lon, lat, Z).unwrap();
        // The bucket boundary the feature falls in — what a real build writes.
        let bucket_start = (start / BUCKET_MS as i64) * BUCKET_MS as i64;
        let id_t = TileId::new(Z, x, y, bucket_start.max(0) as u64);
        w.add_tile_full(
            &id_t,
            bucket_start,
            bucket_start + BUCKET_MS as i64 - 1,
            Some(start),
            1,
            Some(BUCKET_MS),
            &payload,
        )
        .unwrap();
    }
    w.finalize(
        &Metadata::new("export-points")
            .with_zoom_levels(Z, Z)
            .with_temporal_bucket_ms(BUCKET_MS),
    )
    .unwrap();
    out
}

fn v2_config(w: &PackWriter) -> EncoderConfig {
    EncoderConfig {
        format_version: w.format_version(),
        template_collector: Some(w.template_collector()),
        ..EncoderConfig::default()
    }
}

/// Read every row group of a Parquet file into one flat vector of batches.
fn read_parquet(path: &Path) -> (arrow::datatypes::SchemaRef, Vec<arrow::array::RecordBatch>) {
    let file = std::fs::File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let schema = builder.schema().clone();
    let reader = builder.build().unwrap();
    (schema, reader.map(|b| b.unwrap()).collect())
}

/// The file-level `geo` metadata GeoParquet defines.
fn geo_metadata(path: &Path) -> serde_json::Value {
    let file = std::fs::File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let kv = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .expect("parquet file has key/value metadata")
        .iter()
        .find(|kv| kv.key == "geo")
        .expect("`geo` metadata key present")
        .value
        .clone()
        .expect("`geo` metadata has a value");
    serde_json::from_str(&kv).unwrap()
}

/// Decode a little-endian WKB point, asserting the type code along the way.
fn wkb_point(bytes: &[u8]) -> (f64, f64) {
    assert_eq!(bytes[0], 1, "little-endian byte-order marker");
    let ty = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
    assert_eq!(ty, 1, "WKB Point type code");
    let x = f64::from_le_bytes(bytes[5..13].try_into().unwrap());
    let y = f64::from_le_bytes(bytes[13..21].try_into().unwrap());
    (x, y)
}

/// Decode a little-endian WKB linestring into its vertices.
fn wkb_linestring(bytes: &[u8]) -> Vec<(f64, f64)> {
    assert_eq!(bytes[0], 1);
    assert_eq!(u32::from_le_bytes(bytes[1..5].try_into().unwrap()), 2);
    let n = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
    (0..n)
        .map(|i| {
            let o = 9 + i * 16;
            (
                f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap()),
                f64::from_le_bytes(bytes[o + 8..o + 16].try_into().unwrap()),
            )
        })
        .collect()
}

/// Collect `(id, lon, lat, start_ms, end_ms, speed, name)` from an exported
/// point file, keyed by id so tile order does not leak into the assertions.
type PointRow = ([f64; 2], i64, i64, f64, String);
fn read_point_rows(path: &Path) -> HashMap<u64, PointRow> {
    let (schema, batches) = read_parquet(path);

    // The time columns must arrive as real timestamps, not bare integers —
    // this is the regression the whole suite is built around.
    for name in ["start_time", "end_time"] {
        assert_eq!(
            schema.field_with_name(name).unwrap().data_type(),
            &DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            "{name} must round-trip as a UTC millisecond timestamp"
        );
    }

    let mut out = HashMap::new();
    for b in &batches {
        let ids = b
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        let geom = b
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let starts = b
            .column_by_name("start_time")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap();
        let ends = b
            .column_by_name("end_time")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap();
        let speed = b
            .column_by_name("speed")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        // Categorical properties ship dictionary-encoded in the tile; parquet
        // hands them back as plain strings.
        let name_col =
            arrow::compute::cast(b.column_by_name("name").unwrap(), &DataType::Utf8).unwrap();
        let names = name_col.as_any().downcast_ref::<StringArray>().unwrap();

        for r in 0..b.num_rows() {
            let (x, y) = wkb_point(geom.value(r));
            out.insert(
                ids.value(r),
                (
                    [x, y],
                    starts.value(r),
                    ends.value(r),
                    speed.value(r),
                    names.value(r).to_string(),
                ),
            );
        }
    }
    out
}

// ----------------------------------------------------------------------------

/// The headline round-trip: every feature, every column, exact values.
#[test]
fn whole_archive_roundtrips_geometry_times_and_properties() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let out = dir.path().join("out.parquet");

    let ts = PackedTileset::open(&archive).unwrap();
    let report = export::export(&ts, &out, &ExportOptions::default()).unwrap();

    assert_eq!(report.zoom, Z, "defaults to the deepest zoom present");
    assert_eq!(report.tiles_selected, 3);
    assert_eq!(report.files.len(), 1);
    assert_eq!(report.files[0].rows, 3);
    assert_eq!(report.files[0].geometry_types, vec!["Point".to_string()]);

    let rows = read_point_rows(&out);
    assert_eq!(rows.len(), 3, "every feature survived the round-trip");
    for (id, [lon, lat], start, end, num, cat) in point_fixture() {
        let (geom, got_start, got_end, got_num, got_cat) = &rows[&id];
        assert!(
            (geom[0] - lon).abs() < 1e-12 && (geom[1] - lat).abs() < 1e-12,
            "feature {id} geometry drifted: {geom:?} vs [{lon}, {lat}]"
        );
        assert_eq!(*got_start, start, "feature {id} start_time");
        assert_eq!(*got_end, end, "feature {id} end_time");
        assert_eq!(*got_num, num, "feature {id} numeric property");
        assert_eq!(got_cat, cat, "feature {id} categorical property");
    }
}

/// The `geo` key must describe the file well enough for a GeoParquet reader to
/// use it without guessing.
#[test]
fn geo_metadata_is_a_valid_geoparquet_header() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let out = dir.path().join("out.parquet");
    let ts = PackedTileset::open(&archive).unwrap();
    export::export(&ts, &out, &ExportOptions::default()).unwrap();

    let geo = geo_metadata(&out);
    assert_eq!(geo["version"], "1.1.0");
    assert_eq!(geo["primary_column"], "geometry");
    let col = &geo["columns"]["geometry"];
    assert_eq!(col["encoding"], "WKB");
    assert_eq!(col["geometry_types"], serde_json::json!(["Point"]));

    // An absent `crs` is GeoParquet's OGC:CRS84 default; an explicit null would
    // instead claim the CRS is unknown, which is the opposite of true here.
    assert!(col.get("crs").is_none(), "crs must be omitted, not null");

    // File bbox covers the fixture.
    let bbox = col["bbox"].as_array().expect("file bbox");
    let vals: Vec<f64> = bbox.iter().map(|v| v.as_f64().unwrap()).collect();
    assert!((vals[0] - -73.60).abs() < 1e-9, "min_lon {}", vals[0]);
    assert!((vals[1] - 45.50).abs() < 1e-9, "min_lat {}", vals[1]);
    assert!((vals[2] - -73.20).abs() < 1e-9, "max_lon {}", vals[2]);
    assert!((vals[3] - 45.60).abs() < 1e-9, "max_lat {}", vals[3]);

    // The 1.1 covering must point at a real struct column, or a reader will
    // silently fall back to a full scan.
    let covering = &col["covering"]["bbox"];
    assert_eq!(covering["xmin"], serde_json::json!(["bbox", "xmin"]));
    assert_eq!(covering["ymax"], serde_json::json!(["bbox", "ymax"]));
    let (schema, _) = read_parquet(&out);
    let bbox_field = schema.field_with_name("bbox").unwrap();
    let DataType::Struct(children) = bbox_field.data_type() else {
        panic!("covering bbox column is not a struct: {bbox_field:?}");
    };
    let names: Vec<&str> = children.iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["xmin", "ymin", "xmax", "ymax"]);
}

/// The covering box must never be narrower than the geometry it covers, or a
/// reader prunes away a row the box really contains.
#[test]
fn covering_bbox_contains_every_geometry() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let out = dir.path().join("out.parquet");
    let ts = PackedTileset::open(&archive).unwrap();
    export::export(&ts, &out, &ExportOptions::default()).unwrap();

    let (_, batches) = read_parquet(&out);
    for b in &batches {
        let geom = b
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let bbox = b
            .column_by_name("bbox")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::StructArray>()
            .unwrap();
        let f = |i: usize| {
            bbox.column(i)
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .unwrap()
                .clone()
        };
        let (xmin, ymin, xmax, ymax) = (f(0), f(1), f(2), f(3));
        for r in 0..b.num_rows() {
            let (x, y) = wkb_point(geom.value(r));
            assert!(
                (xmin.value(r) as f64) <= x && (xmax.value(r) as f64) >= x,
                "row {r}: x {x} escapes [{}, {}]",
                xmin.value(r),
                xmax.value(r)
            );
            assert!(
                (ymin.value(r) as f64) <= y && (ymax.value(r) as f64) >= y,
                "row {r}: y {y} escapes [{}, {}]",
                ymin.value(r),
                ymax.value(r)
            );
        }
    }
}

/// A bbox subset must drop the features outside the box and keep the rest
/// intact — not merely return fewer rows.
#[test]
fn bbox_filter_selects_the_right_features() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let out = dir.path().join("bbox.parquet");
    let ts = PackedTileset::open(&archive).unwrap();

    let report = export::export(
        &ts,
        &out,
        &ExportOptions {
            // Covers the first two fixture points only.
            bbox: Some([-73.65, 45.45, -73.35, 45.57]),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.files[0].rows, 2);

    let rows = read_point_rows(&out);
    assert_eq!(rows.keys().copied().collect::<Vec<_>>().len(), 2);
    assert!(rows.contains_key(&1) && rows.contains_key(&2));
    assert!(
        !rows.contains_key(&3),
        "the out-of-box feature must be gone"
    );
    assert_eq!(rows[&1].1, 1_700_000_000_000, "kept rows keep their times");
}

/// A time window keeps the features whose own span overlaps it.
#[test]
fn time_range_filter_selects_the_right_features() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let out = dir.path().join("time.parquet");
    let ts = PackedTileset::open(&archive).unwrap();

    let report = export::export(
        &ts,
        &out,
        &ExportOptions {
            start: Some(1_700_003_000_000),
            end: Some(1_700_004_000_000),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.files[0].rows, 1);
    let rows = read_point_rows(&out);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[&2].1, 1_700_003_600_000);
    assert_eq!(rows[&2].2, 1_700_003_660_000);
}

/// Both filters at once compose (intersection, not union).
#[test]
fn bbox_and_time_filters_compose() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let out = dir.path().join("both.parquet");
    let ts = PackedTileset::open(&archive).unwrap();

    let report = export::export(
        &ts,
        &out,
        &ExportOptions {
            bbox: Some([-73.65, 45.45, -73.35, 45.57]),
            start: Some(1_700_003_000_000),
            end: Some(1_700_004_000_000),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.files[0].rows, 1);
    assert!(read_point_rows(&out).contains_key(&2));
}

/// A quantized archive must be dequantized on the way out: exporting the raw
/// `i32` grid indices would place every feature at a nonsense location.
#[test]
fn quantized_coordinates_are_reconstructed() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), |w| EncoderConfig {
        quantize_coords_m: Some(1.0),
        quantize_attrs: HashMap::from([("speed".to_string(), 0.01)]),
        format_version: w.format_version(),
        template_collector: Some(w.template_collector()),
        ..EncoderConfig::default()
    });
    let out = dir.path().join("quant.parquet");
    let ts = PackedTileset::open(&archive).unwrap();
    export::export(&ts, &out, &ExportOptions::default()).unwrap();

    let rows = read_point_rows(&out);
    assert_eq!(rows.len(), 3);
    for (id, [lon, lat], start, _, num, _) in point_fixture() {
        let (geom, got_start, _, got_num, _) = &rows[&id];
        // 1 m quantization ⇒ ≤ half a quantum of error, ~4.5e-6°.
        assert!(
            (geom[0] - lon).abs() < 1e-5 && (geom[1] - lat).abs() < 1e-5,
            "feature {id} did not dequantize: {geom:?} vs [{lon}, {lat}]"
        );
        // The raw grid index would be an integer in the millions — a plain
        // "is it finite" check would pass, so compare to the real value.
        assert!(
            (got_num - num).abs() < 0.01,
            "feature {id} property did not dequantize: {got_num} vs {num}"
        );
        assert_eq!(*got_start, start, "times are never quantized");
    }
}

/// `--geometry-encoding native` must stamp Parquet's GEOMETRY logical type on
/// the column while leaving the bytes (and the GeoParquet metadata) readable
/// by a 1.1 reader.
#[test]
fn native_geometry_encoding_sets_the_parquet_logical_type() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let out = dir.path().join("native.parquet");
    let ts = PackedTileset::open(&archive).unwrap();
    export::export(
        &ts,
        &out,
        &ExportOptions {
            geometry_encoding: GeometryEncoding::Native,
            ..ExportOptions::default()
        },
    )
    .unwrap();

    let file = std::fs::File::open(&out).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let descr = builder.metadata().file_metadata().schema_descr();
    let geom = (0..descr.num_columns())
        .map(|i| descr.column(i))
        .find(|c| c.name() == "geometry")
        .expect("geometry column in the parquet schema");
    assert!(
        matches!(
            geom.logical_type_ref(),
            Some(parquet::basic::LogicalType::Geometry(_))
        ),
        "expected the native GEOMETRY logical type, got {:?}",
        geom.logical_type_ref()
    );

    // The 1.1 story is unchanged for readers that ignore the logical type.
    assert_eq!(geo_metadata(&out)["columns"]["geometry"]["encoding"], "WKB");
    assert_eq!(read_point_rows(&out).len(), 3);
}

/// A linestring layer must round-trip its full vertex list, and a second layer
/// must land in its own file (schemas differ, so they cannot share one).
#[test]
fn multiple_layers_split_into_one_file_each() {
    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("mixed");
    let mut w = PackWriter::create(&out_dir, BlobOrdering::TimeMajor, 1 << 20)
        .unwrap()
        .with_format_version(PACKED_FORMAT_VERSION_V2);

    let line: Vec<[f64; 2]> = vec![[-73.60, 45.50], [-73.58, 45.52], [-73.56, 45.51]];
    let layers = vec![
        ColumnarLayer {
            name: "points".to_string(),
            feature_ids: vec![10],
            start_times: vec![1_700_000_000_000],
            end_times: vec![1_700_000_060_000],
            geometry: GeometryColumn::Point(vec![[-73.60, 45.50]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        },
        ColumnarLayer {
            name: "tracks".to_string(),
            feature_ids: vec![20],
            start_times: vec![1_700_000_000_000],
            end_times: vec![1_700_000_600_000],
            geometry: GeometryColumn::LineString(vec![line.clone()]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        },
    ];
    let payload = encode_tile_with(&layers, &v2_config(&w)).unwrap();
    let (x, y) = lonlat_to_tile(-73.60, 45.50, Z).unwrap();
    let bucket = (1_700_000_000_000i64 / BUCKET_MS as i64) * BUCKET_MS as i64;
    w.add_tile_full(
        &TileId::new(Z, x, y, bucket as u64),
        bucket,
        bucket + BUCKET_MS as i64 - 1,
        Some(1_700_000_000_000),
        2,
        Some(BUCKET_MS),
        &payload,
    )
    .unwrap();
    w.finalize(
        &Metadata::new("export-mixed")
            .with_zoom_levels(Z, Z)
            .with_temporal_bucket_ms(BUCKET_MS),
    )
    .unwrap();

    let ts = PackedTileset::open(&out_dir).unwrap();
    let stem = dir.path().join("mixed.parquet");
    let report = export::export(&ts, &stem, &ExportOptions::default()).unwrap();
    assert_eq!(report.files.len(), 2, "one file per layer");

    let tracks = report
        .files
        .iter()
        .find(|f| f.layer == "tracks")
        .expect("tracks file");
    assert!(
        tracks.path.ends_with("mixed.tracks.parquet"),
        "unexpected path {}",
        tracks.path
    );
    assert_eq!(tracks.geometry_types, vec!["LineString".to_string()]);

    let (_, batches) = read_parquet(Path::new(&tracks.path));
    let geom = batches[0]
        .column_by_name("geometry")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    let verts = wkb_linestring(geom.value(0));
    assert_eq!(verts.len(), line.len(), "every vertex survived");
    for (got, want) in verts.iter().zip(&line) {
        assert!((got.0 - want[0]).abs() < 1e-12 && (got.1 - want[1]).abs() < 1e-12);
    }

    // Selecting one layer writes exactly that file, at the path given.
    let single = dir.path().join("only-tracks.parquet");
    let one = export::export(
        &ts,
        &single,
        &ExportOptions {
            layer: Some("tracks".to_string()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(one.files.len(), 1);
    assert_eq!(one.files[0].path, single.display().to_string());
}

/// Provenance columns must name the tile each row came from — the only way a
/// consumer can tell a clipped piece from a whole feature, since clipping
/// gives every piece the parent's id.
#[test]
fn rows_carry_their_tile_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let out = dir.path().join("prov.parquet");
    let ts = PackedTileset::open(&archive).unwrap();
    export::export(&ts, &out, &ExportOptions::default()).unwrap();

    let (schema, batches) = read_parquet(&out);
    for name in ["stt_zoom", "stt_x", "stt_y"] {
        assert!(schema.field_with_name(name).is_ok(), "{name} is exported");
    }
    for b in &batches {
        let z = b
            .column_by_name("stt_zoom")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt8Array>()
            .unwrap();
        let x = b
            .column_by_name("stt_x")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .unwrap();
        let geom = b
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        for r in 0..b.num_rows() {
            assert_eq!(z.value(r), Z);
            let (lon, lat) = wkb_point(geom.value(r));
            let (want_x, _) = lonlat_to_tile(lon, lat, Z).unwrap();
            assert_eq!(x.value(r), want_x, "stt_x must name the source tile");
        }
    }
}

/// `vertex_time` ships as `u16` deltas against a baked origin/step. Exporting
/// the raw deltas would hand a consumer numbers that mean nothing outside this
/// format, so they must come back as absolute Unix ms — and `triangles`, which
/// is derived tessellation state, must not come back at all.
#[test]
fn vertex_times_are_absolute_and_derived_columns_are_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("tracks");
    let mut w = PackWriter::create(&out_dir, BlobOrdering::TimeMajor, 1 << 20)
        .unwrap()
        .with_format_version(PACKED_FORMAT_VERSION_V2);

    let line: Vec<[f64; 2]> = vec![[-73.60, 45.50], [-73.58, 45.52], [-73.56, 45.51]];
    let vtimes: Vec<i64> = vec![1_700_000_000_000, 1_700_000_030_000, 1_700_000_060_000];
    let layer = ColumnarLayer {
        name: "default".to_string(),
        feature_ids: vec![7],
        start_times: vec![vtimes[0]],
        end_times: vec![*vtimes.last().unwrap()],
        geometry: GeometryColumn::LineString(vec![line]),
        vertex_times: Some(vec![vtimes.clone()]),
        vertex_values: None,
        // Only meaningful on polygons — the encoder drops it here, which is
        // itself proof the export never depends on it.
        triangles: None,
        vertex_value_matrix: None,
        properties: vec![],
    };
    let payload = encode_tile_with(&[layer], &v2_config(&w)).unwrap();
    let (x, y) = lonlat_to_tile(-73.60, 45.50, Z).unwrap();
    let bucket = (vtimes[0] / BUCKET_MS as i64) * BUCKET_MS as i64;
    w.add_tile_full(
        &TileId::new(Z, x, y, bucket as u64),
        bucket,
        bucket + BUCKET_MS as i64 - 1,
        Some(vtimes[0]),
        1,
        Some(BUCKET_MS),
        &payload,
    )
    .unwrap();
    w.finalize(
        &Metadata::new("export-vt")
            .with_zoom_levels(Z, Z)
            .with_temporal_bucket_ms(BUCKET_MS),
    )
    .unwrap();

    let ts = PackedTileset::open(&out_dir).unwrap();
    let out = dir.path().join("vt.parquet");
    export::export(&ts, &out, &ExportOptions::default()).unwrap();

    let (schema, batches) = read_parquet(&out);
    assert!(
        schema.field_with_name("triangles").is_err(),
        "derived tessellation must not be exported"
    );
    let vt = batches[0]
        .column_by_name("vertex_time")
        .expect("vertex_time");
    let list = vt
        .as_any()
        .downcast_ref::<arrow::array::ListArray>()
        .unwrap();
    let values = arrow::compute::cast(&list.value(0), &DataType::Int64).unwrap();
    let got: Vec<i64> = values
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    // The u16-delta step is exact (1 ms) for a span this small, so the
    // reconstruction must be value-for-value.
    assert_eq!(got, vtimes, "per-vertex times must come back absolute");
}

/// CLI value parsing, including the ISO-vs-epoch precedence.
#[test]
fn cli_value_parsing() {
    assert_eq!(
        export::parse_bbox("-73.6, 45.5,-73.2,45.7").unwrap(),
        [-73.6, 45.5, -73.2, 45.7]
    );
    assert!(export::parse_bbox("-73.6,45.5,-73.2").is_err());
    assert!(export::parse_bbox("a,b,c,d").is_err());

    assert_eq!(
        export::parse_time_bound("2024-03-01T00:00:00Z").unwrap(),
        1_709_251_200_000
    );
    assert_eq!(
        export::parse_time_bound("2024-03-01").unwrap(),
        1_709_251_200_000
    );
    assert_eq!(
        export::parse_time_bound("1700000000000").unwrap(),
        1_700_000_000_000
    );
    assert!(export::parse_time_bound("not-a-time").is_err());
}

/// Asking for a zoom the archive does not carry must say so, and name the
/// zooms it does — a silent empty file would look like data loss.
#[test]
fn missing_zoom_is_a_named_error() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_points(dir.path(), v2_config);
    let ts = PackedTileset::open(&archive).unwrap();
    let err = export::export(
        &ts,
        &dir.path().join("nope.parquet"),
        &ExportOptions {
            zoom: Some(3),
            ..ExportOptions::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("zoom 3"), "unhelpful error: {err}");
    assert!(
        err.contains("10"),
        "error must name the zooms present: {err}"
    );
}
