//! `encode_single_tile_counted` (the stt-serve hot path) vs the offline build:
//! for every tile the offline `generate_tiles` pipeline produces, the
//! single-tile encoder must produce THE SAME tile — same placement, clipping,
//! temporal bucketing, feature count, and encoded payload — and `None` for any
//! `(z, x, y, t)` the offline build left empty.
//!
//! Parity is asserted at the strongest level the encoder guarantees across two
//! encode calls: equal byte LENGTH plus an identical logical fingerprint
//! (layer names, sorted schema/field metadata, and raw column buffer bytes).
//! Raw byte equality is not asserted because arrow-54 serializes schema/field
//! metadata in per-process `HashMap` iteration order, so two encodes of the
//! same tile can differ in the metadata region only (the tracked gap in
//! `docs/spec/stt-packed-format.md` §7-D6; see
//! `crates/stt-core/tests/reproducible_build.rs`).
//!
//! Written against the PUBLIC `stt-build` surface only (`generate_tiles`,
//! `encode_single_tile_counted`, `TileConfig`) with the shared cross-source
//! fixture from `tests/common`, so it survives internal `tiler.rs` refactors.

mod common;

use stt_build::tiler::{encode_single_tile_counted, generate_tiles, TileConfig};
use stt_core::arrow_tile::{decode_tile, encode_tile_with, EncoderConfig};

/// A stable, order-independent fingerprint of a decoded tile payload: every
/// layer's name, sorted schema metadata, and, per column, the field name +
/// sorted field metadata + the column's raw Arrow value bytes. Mirrors the
/// fingerprint `reproducible_build.rs` uses for the same arrow-54 reason.
fn logical_fingerprint(payload: &[u8]) -> Vec<u8> {
    let mut fp: Vec<u8> = Vec::new();
    for layer in decode_tile(payload).unwrap() {
        fp.extend_from_slice(layer.name.as_bytes());
        fp.push(0);
        let schema = layer.batch.schema();
        let mut smeta: Vec<(String, String)> = schema
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        smeta.sort();
        for (k, v) in smeta {
            fp.extend_from_slice(k.as_bytes());
            fp.push(b'=');
            fp.extend_from_slice(v.as_bytes());
            fp.push(0);
        }
        for (i, field) in schema.fields().iter().enumerate() {
            fp.extend_from_slice(field.name().as_bytes());
            fp.push(0);
            let mut fmeta: Vec<(String, String)> = field
                .metadata()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            fmeta.sort();
            for (k, v) in fmeta {
                fp.extend_from_slice(k.as_bytes());
                fp.push(b'=');
                fp.extend_from_slice(v.as_bytes());
                fp.push(0);
            }
            let data = layer.batch.column(i).to_data();
            for buf in data.buffers() {
                fp.extend_from_slice(buf.as_slice());
            }
        }
    }
    fp
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
    assert!(offline.len() > 20, "expected a multi-tile sweep, got {}", offline.len());

    let mut clipped_layer_seen = false;
    for tile in &offline {
        let offline_bytes =
            encode_tile_with(&tile.layers, &encoder).expect("encode offline tile");
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
        assert_eq!(
            single_bytes.len(),
            offline_bytes.len(),
            "payload length diverges at z{}/{}/{} t{}",
            tile.id.z,
            tile.id.x,
            tile.id.y,
            tile.id.t
        );
        assert_eq!(
            logical_fingerprint(&single_bytes),
            logical_fingerprint(&offline_bytes),
            "payload content diverges at z{}/{}/{} t{}",
            tile.id.z,
            tile.id.x,
            tile.id.y,
            tile.id.t
        );

        clipped_layer_seen |= tile.layers.iter().any(|l| l.name.ends_with("_originals"))
            || tile.layers.len() > 1;
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
    assert!(empty.is_none(), "an offline-empty (z,x,y,t) must encode to None");
}
