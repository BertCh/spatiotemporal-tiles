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

use stt_build::tiler::{encode_single_tile_counted, generate_tiles, TileConfig};
use stt_core::arrow_tile::{encode_tile_with, EncoderConfig};

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
