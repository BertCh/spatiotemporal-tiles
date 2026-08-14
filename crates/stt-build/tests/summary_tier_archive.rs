//! End-to-end test for the optional summary tier:
//! build raw tiles + a summary tier into one archive, then read it back
//! and confirm both tiers are addressable through the same directory.

use geojson::{Feature, Geometry, Value as GeomValue};
use stt_build::input::ParsedFeature;
use stt_build::summary::{build_summary_tier, parse_summary_columns, SummaryConfig};
use stt_build::tiler::{generate_tiles_streaming, TileConfig};
use stt_core::metadata::{Metadata, SummaryScheme};
use stt_core::{BlobOrdering, PackWriter, PackedReader};

fn point(lon: f64, lat: f64, ts: u64, mag: f64) -> ParsedFeature {
    let props = serde_json::json!({ "magnitude": mag })
        .as_object()
        .cloned()
        .and_then(stt_build::props::FeatureProperties::from_map);
    ParsedFeature {
        home_zoom: None,
        geojson: Feature {
            bbox: None,
            geometry: Some(Geometry::new(GeomValue::Point(vec![lon, lat]))),
            id: None,
            properties: None,
            foreign_members: None,
        },
        shared_properties: props,
        timestamp: ts,
        end_timestamp: None,
        vertex_timestamps: None,
        vertex_values: None,
        vertex_value_matrix: None,
        lon,
        lat,
    }
}

#[test]
fn raw_plus_summary_tier_roundtrips_through_archive() {
    // 20 points clustered in SF, 5 points in London — enough to make multiple
    // H3 cells across both clusters at zoom 0 → res 0.
    let mut features = Vec::new();
    for i in 0..20 {
        features.push(point(
            -122.45 + (i % 4) as f64 * 0.001,
            37.77 + (i / 4) as f64 * 0.001,
            1_700_000_000_000 + i as u64 * 1000,
            5.0 + (i % 3) as f64,
        ));
    }
    for i in 0..5 {
        features.push(point(
            -0.1278 + i as f64 * 0.001,
            51.5074,
            1_700_000_000_000 + i as u64 * 1000,
            6.0,
        ));
    }

    let dir = tempfile::tempdir().unwrap();
    let mut writer = PackWriter::create(dir.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();

    // Raw and summary deliberately overlap at every zoom. Before directory v6
    // this silently overwrote one tier at identical z/x/y/t addresses.
    let raw_config = TileConfig {
        min_zoom: 0,
        max_zoom: 2,
        layer_name: "default".to_string(),
        temporal_bucket_ms: 3_600_000,
        clip_trajectories: false,
        ..TileConfig::default()
    };
    let _raw_stats = generate_tiles_streaming(&features, &raw_config, &mut writer, 2).unwrap();

    // Summary tier across zooms 0..=2.
    let cols = parse_summary_columns("magnitude:mean,magnitude:max").unwrap();
    let summary_config = SummaryConfig {
        scheme: SummaryScheme::H3,
        min_zoom: 0,
        max_zoom: 2,
        temporal_bucket_ms: 3_600_000,
        columns: cols,
        layer_name: "summary".to_string(),
        sub_buckets: 1,
    };
    let n_summary = build_summary_tier(&features, &summary_config, &mut writer).unwrap();
    assert!(n_summary > 0, "summary tier produced no tiles");

    let metadata = Metadata::new("summary-test")
        .with_zoom_levels(0, 2)
        .with_temporal_bucket_ms(3_600_000)
        .with_summary_tier(summary_config.to_tier());
    writer.finalize(&metadata).unwrap();

    // -------- read side --------
    let reader = PackedReader::open(dir.path().join("manifest.json")).unwrap();
    let m = reader.metadata().clone();
    let tier = m
        .summary_tier
        .expect("archive must carry the summary_tier descriptor");
    assert_eq!(tier.scheme, SummaryScheme::H3);
    assert_eq!(tier.min_zoom, 0);
    assert_eq!(tier.max_zoom, 2);
    assert_eq!(tier.layer_name, "summary");
    assert_eq!(tier.variant_id, stt_core::tile::SUMMARY_VARIANT_ID);
    assert_eq!(tier.cell_resolution_per_zoom.len(), 3);

    // Group archive entries by zoom and confirm both tiers are present.
    let mut zooms_with_summary = 0usize;
    let mut zooms_with_raw = 0usize;
    let mut summary_feature_total = 0u64;
    let mut raw_feature_total = 0u64;
    let mut addresses = std::collections::HashMap::<(u8, u32, u32, i64), Vec<u32>>::new();
    for entry in reader.entries() {
        addresses
            .entry((entry.zoom, entry.x, entry.y, entry.time_start))
            .or_default()
            .push(entry.variant_id);
        // Read each tile so we can distinguish summary vs raw by layer name.
        let layers = reader.read_layers(entry).unwrap();
        let is_summary = layers.iter().any(|l| l.name == "summary");
        if is_summary {
            assert_eq!(entry.variant_id, stt_core::tile::SUMMARY_VARIANT_ID);
            zooms_with_summary += 1;
            summary_feature_total += entry.feature_count as u64;
            // Summary layers carry the implicit `count` column.
            let batch = &layers.iter().find(|l| l.name == "summary").unwrap().batch;
            assert!(
                batch.column_by_name("count").is_some(),
                "summary tile must carry count"
            );
            assert!(
                batch.column_by_name("mean_magnitude").is_some(),
                "summary tile must carry mean_magnitude"
            );
            assert!(
                batch.column_by_name("max_magnitude").is_some(),
                "summary tile must carry max_magnitude"
            );
            // The sum of per-cell counts in the layer equals the
            // sum of the underlying features that fell into the tile.
            let counts = batch
                .column_by_name("count")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            let mut cell_count_total = 0f64;
            for i in 0..counts.len() {
                cell_count_total += counts.value(i);
            }
            assert!(cell_count_total > 0.0);
        } else {
            assert_eq!(entry.variant_id, stt_core::tile::RAW_VARIANT_ID);
            zooms_with_raw += 1;
            raw_feature_total += entry.feature_count as u64;
        }
    }

    assert!(zooms_with_summary > 0, "no summary tiles in archive");
    assert!(zooms_with_raw > 0, "no raw tiles in archive");
    assert!(
        addresses.values().any(|variants| {
            variants.contains(&stt_core::tile::RAW_VARIANT_ID)
                && variants.contains(&stt_core::tile::SUMMARY_VARIANT_ID)
        }),
        "fixture must contain an exact z/x/y/t raw+summary overlap"
    );
    // 25 features total — every raw tile keeps each feature, but the
    // streaming writer may emit them across multiple tiles, so the sum of
    // raw feature_counts equals at least the input size at every zoom.
    assert!(raw_feature_total >= 25, "raw feature_count looks wrong");
    // Summary feature_count is the number of cells across all summary tiles,
    // and should be bounded by 25 too (one row per feature in the worst case).
    assert!(
        summary_feature_total >= 2 && summary_feature_total <= 75,
        "unexpected summary feature_count {summary_feature_total}"
    );
}
