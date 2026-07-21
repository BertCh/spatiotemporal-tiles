//! End-to-end contract test for `--style-hints`: load a small GeoParquet
//! dataset with a numeric + a string property, run the style-hints bake seam
//! (collect → profile → `Metadata::with_style_hints`), finalize a packed
//! dataset, and assert the block lands in `manifest.json` with the pinned
//! wire shape (numeric percentiles + suggested_domain; categorical
//! name+cardinality ONLY; layer_hint; suggested_playback_seconds).

use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::sync::Arc;
use stt_build::input::{calculate_bounds, load_features, InputStrictness, TimeFormat};
use stt_build::style_hints::compute_style_hints;
use stt_build::tiler::{generate_tiles, TileConfig, TileWriter};

/// Little-endian ISO WKB for a 2D point.
fn wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut b = vec![0x01, 0x01, 0x00, 0x00, 0x00];
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

#[test]
fn style_hints_flow_into_the_packed_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("quakes.parquet");

    // 100 points over 4 hours with a numeric `magnitude` (0.0..9.9) and a
    // 7-class string `category`.
    let n = 100usize;
    let hour = 3_600_000i64;
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, true),
        Field::new("timestamp", DataType::Int64, true),
        Field::new("magnitude", DataType::Float64, true),
        Field::new("category", DataType::Utf8, true),
    ]));
    let wkbs: Vec<Vec<u8>> = (0..n)
        .map(|i| wkb_point(-120.0 + i as f64 * 0.01, 35.0 + i as f64 * 0.01))
        .collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BinaryArray::from_iter_values(
            wkbs.iter().map(|b| b.as_slice()),
        )),
        Arc::new(Int64Array::from_iter_values((0..n).map(|i| {
            1_700_000_000_000i64 + (i as i64 * 4 * hour) / n as i64
        }))),
        Arc::new(Float64Array::from_iter_values(
            (0..n).map(|i| i as f64 / 10.0),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..n).map(|i| format!("class-{}", i % 7)),
        )),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let file = std::fs::File::create(&input).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // The --style-hints bake seam, exactly as the stt-build binary drives it:
    // load → compute_style_hints → with_style_hints → PackWriter::finalize.
    let features = load_features(
        &input,
        "timestamp",
        None,
        TimeFormat::UnixMs,
        InputStrictness::Warn,
        InputStrictness::Warn,
    )
    .unwrap();
    let (bounds, time_range) = calculate_bounds(&features).unwrap();
    let temporal_bucket_ms = hour as u64;
    let hints = compute_style_hints(&features, &time_range, temporal_bucket_ms, true)
        .expect("non-empty features must profile");

    let config = TileConfig {
        min_zoom: 4,
        max_zoom: 4,
        temporal_bucket_ms,
        ..TileConfig::default()
    };
    let tiles = generate_tiles(&features, &config, 1).unwrap();
    assert!(!tiles.is_empty());

    let out_dir = dir.path().join("packed");
    let mut writer =
        stt_core::PackWriter::create(&out_dir, stt_core::BlobOrdering::Auto, 64 * 1024 * 1024)
            .unwrap();
    for tile in &tiles {
        writer.write_tile(tile).unwrap();
    }
    let metadata = stt_core::metadata::Metadata::new("hints-e2e")
        .with_bounds(bounds)
        .with_time_range(time_range)
        .with_zoom_levels(4, 4)
        .with_temporal_bucket_ms(temporal_bucket_ms)
        .with_style_hints(hints);
    writer.finalize(&metadata).unwrap();

    // Metadata flows into manifest.json verbatim — assert the wire shape there.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out_dir.join("manifest.json")).unwrap()).unwrap();
    let sh = &manifest["metadata"]["style_hints"];
    assert!(
        sh.is_object(),
        "style_hints missing from manifest metadata: {manifest}"
    );
    assert_eq!(sh["version"], 1);
    assert_eq!(sh["layer_hint"], "points");
    // 4 hour-buckets -> sqrt(4)=2 -> clamps up to 20.
    assert_eq!(sh["suggested_playback_seconds"], 20);

    let props = sh["properties"].as_array().unwrap();
    assert_eq!(props.len(), 2, "{props:?}");
    // Deterministic (sorted) property order: category, magnitude.
    let cat = &props[0];
    assert_eq!(cat["name"], "category");
    assert_eq!(cat["cardinality"], 7);
    // Categorical carries ONLY name + cardinality — numeric keys ABSENT.
    assert_eq!(cat.as_object().unwrap().len(), 2, "{cat}");

    let mag = &props[1];
    assert_eq!(mag["name"], "magnitude");
    assert_eq!(mag["min"], 0.0);
    assert_eq!(mag["max"], 9.9);
    // sorted[floor(100 * 0.97)] = sorted[97] = 9.7
    assert_eq!(mag["p97"], 9.7);
    // suggested_domain = [min, p97] rounded outward to 2 sig figs.
    assert_eq!(mag["suggested_domain"][0], 0.0);
    assert_eq!(mag["suggested_domain"][1], 9.7);
    // Numeric entries omit cardinality (absent, not null).
    assert!(mag.get("cardinality").is_none(), "{mag}");
}
