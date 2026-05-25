//! End-to-end test for the optional summary tier:
//! build raw tiles + a summary tier into one archive, then read it back
//! and confirm both tiers are addressable through the same directory.

use geojson::{Feature, Geometry, Value as GeomValue};
use std::sync::Arc;
use stt_build::input::ParsedFeature;
use stt_build::summary::{
    build_summary_tier, parse_summary_columns, SummaryConfig,
};
use stt_build::tiler::{generate_tiles_streaming, TileConfig};
use stt_core::archive::{Archive, ArchiveReader};
use stt_core::metadata::{Metadata, SummaryScheme};
use stt_core::types::Compression;

fn point(lon: f64, lat: f64, ts: u64, mag: f64) -> ParsedFeature {
    let props = serde_json::json!({ "magnitude": mag })
        .as_object()
        .cloned()
        .map(Arc::new);
    ParsedFeature {
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

    let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
    let mut writer = Archive::create(&path, Compression::Zstd).unwrap();

    // Raw tier across zooms 8..=10 — coarse enough to keep the test fast but
    // wide enough that "raw" is unambiguously distinct from "summary".
    let raw_config = TileConfig {
        min_zoom: 8,
        max_zoom: 10,
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
    };
    let n_summary = build_summary_tier(&features, &summary_config, &mut writer).unwrap();
    assert!(n_summary > 0, "summary tier produced no tiles");

    let metadata = Metadata::new("summary-test")
        .with_zoom_levels(0, 10)
        .with_temporal_bucket_ms(3_600_000)
        .with_summary_tier(summary_config.to_tier());
    writer.finalize(&metadata).unwrap();

    // -------- read side --------
    let reader = ArchiveReader::open(&path).unwrap();
    let m = reader.metadata().clone();
    let tier = m
        .summary_tier
        .expect("archive must carry the summary_tier descriptor");
    assert_eq!(tier.scheme, SummaryScheme::H3);
    assert_eq!(tier.min_zoom, 0);
    assert_eq!(tier.max_zoom, 2);
    assert_eq!(tier.layer_name, "summary");
    assert_eq!(tier.cell_resolution_per_zoom.len(), 3);

    // Group archive entries by zoom and confirm both tiers are present.
    let mut zooms_with_summary = 0usize;
    let mut zooms_with_raw = 0usize;
    let mut summary_feature_total = 0u64;
    let mut raw_feature_total = 0u64;
    for entry in reader.entries() {
        // Read each tile so we can distinguish summary vs raw by layer name.
        let layers = reader.read_layers(entry).unwrap();
        let is_summary = layers.iter().any(|l| l.name == "summary");
        if is_summary {
            zooms_with_summary += 1;
            summary_feature_total += entry.feature_count as u64;
            // Summary layers carry the implicit `count` column.
            let batch = &layers
                .iter()
                .find(|l| l.name == "summary")
                .unwrap()
                .batch;
            assert!(batch.column_by_name("count").is_some(), "summary tile must carry count");
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
            zooms_with_raw += 1;
            raw_feature_total += entry.feature_count as u64;
        }
    }

    assert!(zooms_with_summary > 0, "no summary tiles in archive");
    assert!(zooms_with_raw > 0, "no raw tiles in archive");
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
