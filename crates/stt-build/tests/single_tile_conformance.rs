//! `encode_single_tile_counted` (the stt-serve hot path) vs the offline build:
//! for every tile the offline `generate_tiles` pipeline produces, the
//! single-tile encoder must produce THE SAME tile — same placement, clipping,
//! temporal bucketing, feature count, and encoded payload — and `None` for any
//! `(z, x, y, t)` the offline build left empty.
//!
//! Parity is asserted at the strongest level the encoder guarantees: raw byte
//! equality of the two encoded payloads. Arrow ≥59 serializes schema/field
//! metadata in stable (sorted) key order and the encoder feeds it sorted
//! `BTreeMap`s, so two encodes of the same tile are byte-identical — the same
//! cross-process reproducibility content-addressed pack dedup relies on (see
//! `docs/spec/stt-packed-format.md` §7-D6 and the determinism guard in
//! `crates/stt-core/tests/reproducible_build.rs`). This makes the parity check
//! independent of any decode step: if the single-tile path ever diverges from
//! the offline pipeline in placement, clipping, bucketing, column order, or
//! metadata, the byte comparison fails immediately.
//!
//! Written against the PUBLIC `stt-build` surface only (`generate_tiles`,
//! `encode_single_tile_counted`, `TileConfig`) with the shared cross-source
//! fixture from `tests/common`, so it survives internal `tiler.rs` refactors.

mod common;

use std::sync::Arc;

use stt_build::input::ParsedFeature;
use stt_build::tiler::{encode_single_tile_counted, generate_tiles, TileConfig};
use stt_core::arrow_tile::{encode_tile_with, EncoderConfig, GlobalColumnPins};

/// Resolve the dataset-global encoder pins the way `stt-build`'s pass 1 does.
fn pins_for(features: &[ParsedFeature]) -> GlobalColumnPins {
    use stt_build::columnar::{infer_property_types, AttributeFilter};
    use stt_build::dataset_stats::collect_dataset_stats;

    let filter = AttributeFilter::KeepAll;
    let types = infer_property_types(
        features.iter().map(|f| f.shared_properties.as_ref()),
        &filter,
    );
    collect_dataset_stats(features, &filter, &types).to_pins()
}

#[test]
fn single_tile_encoding_matches_offline_build() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.parquet");
    common::write_fixture_parquet(&common::fixture_rows(), &path);
    let features = common::load_file(&path);
    assert_eq!(features.len(), 5, "fixture decodes to 5 features");

    // Default clipping ON so the linestring rows exercise the clip/placement
    // path at deep zooms, not just point centroids.
    let config = TileConfig {
        min_zoom: 0,
        max_zoom: 8,
        temporal_bucket_ms: 3_600_000,
        ..TileConfig::default()
    };
    let encoder = EncoderConfig::default();

    let offline = generate_tiles(&features, &config, 2).expect("offline generate_tiles");
    assert!(!offline.is_empty(), "offline build produced no tiles");
    // The fixture must fan out across zooms and buckets for this to be a
    // meaningful sweep (5 features × 9 zooms, several time buckets).
    assert!(
        offline.len() > 20,
        "expected a multi-tile sweep, got {}",
        offline.len()
    );

    let mut clipped_layer_seen = false;
    for tile in &offline {
        let offline_bytes = encode_tile_with(&tile.layers, &encoder).expect("encode offline tile");
        let (single_bytes, single_count) = encode_single_tile_counted(
            &features,
            tile.id.z,
            tile.id.x,
            tile.id.y,
            tile.id.t as i64,
            &config,
            &encoder,
        )
        .expect("encode_single_tile_counted")
        .unwrap_or_else(|| {
            panic!(
                "single-tile path returned empty for offline tile z{}/{}/{} t{}",
                tile.id.z, tile.id.x, tile.id.y, tile.id.t
            )
        });

        assert_eq!(
            single_count,
            tile.feature_count(),
            "feature count diverges at z{}/{}/{} t{}",
            tile.id.z,
            tile.id.x,
            tile.id.y,
            tile.id.t
        );
        // Strict byte-identity: the single-tile encoder must reproduce the
        // offline pipeline's payload exactly (see module docs — Arrow ≥59 makes
        // this deterministic). Any divergence in placement, clipping, temporal
        // bucketing, column order, or baked metadata surfaces here.
        assert_eq!(
            single_bytes, offline_bytes,
            "payload bytes diverge at z{}/{}/{} t{}",
            tile.id.z, tile.id.x, tile.id.y, tile.id.t
        );

        clipped_layer_seen |=
            tile.layers.iter().any(|l| l.name.ends_with("_originals")) || tile.layers.len() > 1;
    }
    assert!(
        offline
            .iter()
            .flat_map(|t| &t.layers)
            .any(|l| matches!(l.geometry.kind(), stt_core::types::GeometryType::LineString)),
        "sweep must cover a linestring layer (clip path)"
    );
    let _ = clipped_layer_seen;

    // A cell/bucket the offline build never emitted must be empty here too.
    let empty = encode_single_tile_counted(&features, 8, 3, 3, 0, &config, &encoder)
        .expect("encode empty tile");
    assert!(
        empty.is_none(),
        "an offline-empty (z,x,y,t) must encode to None"
    );
}

/// The same parity sweep with the DATASET-GLOBAL PINS threaded through both
/// sides — the `stt-serve` half of pass 1's contract.
///
/// # Why this is the guard that matters for M2
///
/// `stt-serve` answers tile requests one tile at a time. That is precisely the
/// position from which a dataset-global verdict cannot be re-derived: the server
/// has one tile's rows, so if it re-decides the numeric affine or the
/// dictionary-vs-`Utf8` choice locally it produces a DIFFERENT tile from the one
/// the offline build published — the same value decoding to a different number
/// at the same address. The build↔serve byte-identity contract (shared through
/// `build_options`) is what stops that, and it only holds if the pins reach both
/// producers.
///
/// So the assertion is: given the SAME pins, the single-tile encoder reproduces
/// the offline pipeline's payload byte for byte, across the whole
/// zoom × bucket sweep including the clipped-trajectory tiles. Byte equality is
/// the strongest available statement and needs no decode step — any divergence
/// in placement, clipping, bucketing, column order, baked metadata, or (the new
/// one) which pin was applied, fails here immediately.
///
/// The complementary risk — a served tile built with NO pins against a pinned
/// archive — is not simulated here because nothing consumes the pins yet, so the
/// two are byte-identical by construction (see
/// `reproducible_pipeline::attaching_pins_moves_no_archive_byte`). The item that
/// starts consuming them owns adding that negative case.
#[test]
fn pinned_single_tile_encoding_matches_pinned_offline_build() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.parquet");
    common::write_fixture_parquet(&common::fixture_rows(), &path);
    let features = common::load_file(&path);
    assert_eq!(features.len(), 5, "fixture decodes to 5 features");

    let config = TileConfig {
        min_zoom: 0,
        max_zoom: 8,
        temporal_bucket_ms: 3_600_000,
        ..TileConfig::default()
    };

    let pins = Arc::new(pins_for(&features));
    assert!(
        !pins.attr.is_empty() && !pins.dict.is_empty(),
        "the fixture must pin a numeric AND a categorical column, or this sweep \
         proves nothing: {pins:?}"
    );
    let encoder = EncoderConfig {
        global_pins: Some(pins),
        ..EncoderConfig::default()
    };

    let offline = generate_tiles(&features, &config, 2).expect("offline generate_tiles");
    assert!(
        offline.len() > 20,
        "expected a multi-tile sweep, got {}",
        offline.len()
    );

    for tile in &offline {
        let offline_bytes = encode_tile_with(&tile.layers, &encoder).expect("encode offline tile");
        let (single_bytes, single_count) = encode_single_tile_counted(
            &features,
            tile.id.z,
            tile.id.x,
            tile.id.y,
            tile.id.t as i64,
            &config,
            &encoder,
        )
        .expect("encode_single_tile_counted")
        .unwrap_or_else(|| {
            panic!(
                "pinned single-tile path returned empty for offline tile z{}/{}/{} t{}",
                tile.id.z, tile.id.x, tile.id.y, tile.id.t
            )
        });

        assert_eq!(
            single_count,
            tile.feature_count(),
            "feature count diverges under pins at z{}/{}/{} t{}",
            tile.id.z,
            tile.id.x,
            tile.id.y,
            tile.id.t
        );
        assert_eq!(
            single_bytes, offline_bytes,
            "payload bytes diverge under pins at z{}/{}/{} t{}",
            tile.id.z, tile.id.x, tile.id.y, tile.id.t
        );
    }
    assert!(
        offline
            .iter()
            .flat_map(|t| &t.layers)
            .any(|l| matches!(l.geometry.kind(), stt_core::types::GeometryType::LineString)),
        "sweep must cover a linestring layer (clip path)"
    );
}
