//! Emit the committed cross-impl golden fixture for the TS packed-format reader.
//!
//! Writes a deterministic tiny packed dataset (fixed payloads → stable content
//! hashes) to `packages/core/test/fixtures/packed-golden/`:
//!
//!   manifest.json
//!   index/<hash>.sttd
//!   packs/<hash>.sttp   (2-3 packs at the tiny pack target)
//!
//! The TS test reads this fixture and must reproduce identical decoded payloads.
//! The payload scheme is intentionally simple + known so it's easy to assert on
//! both sides:
//!
//!   - 12 tiles, zoom 10, at spatial cells `(x, 0)` for x in 0..12.
//!   - Each tile carries one `"default"` point layer with feature ids
//!     `[100*k + 0, 100*k + 1, 100*k + 2]` for tile index k (0..12) at point
//!     `(-122.4 + 0.01*k, 37.7)`, start/end times `(1000*k, 1000*k + 100)`.
//!   - Tiles k=4 and k=9 are byte-identical to tile k=0 (same feature ids /
//!     coords / times) → exercise byte-identical blob dedup. So distinct decoded
//!     payloads correspond to k in {0,1,2,3,5,6,7,8,10,11} (10 distinct), with
//!     k=4 and k=9 decoding identically to k=0.
//!   - Tile time_start = 1000*k, except the deduped k=4,k=9 which reuse k=0's
//!     id/coords but keep their OWN time_start so the directory keeps 12 entries.
//!
//! The pack target is 4 KiB to cut the ~10 distinct blobs into 2-3 packs.
//!
//! Run: cargo run -p stt-core --example make-golden-fixture
//!
//! The generator accepts no version switch: packed v1 is withdrawn, and both
//! the manifest and every frame use the sole supported format version.
//!
//! # TWO PASSES, per dataset
//!
//! Mechanism M2 resolves the numeric affine and the dictionary-vs-`Utf8` verdict
//! from the DATASET domain (`EncoderConfig::global_pins`), not from whichever
//! rows a tile caught. A generator that built at `..EncoderConfig::default()`
//! would leave that field `None` and emit on the INCUMBENT per-tile path — the
//! cross-impl fixtures would then be blind to every byte the pinned path moves,
//! and re-running this generator would re-bless unpinned bytes while pinning
//! nothing new. So each dataset below is built the way the builder builds:
//! [`layer_for`] first, [`derive_pins`] over that dataset's own features, then
//! encode. The two datasets get SEPARATE pins, because a pin is a property of a
//! dataset; the paged and single grid builds share one pin set, because they are
//! one dataset in two container shapes and their packs must stay byte-identical.
//!
//! `kind`'s categories are long strings and each tile uses a different pair of
//! them ([`KINDS`]): that is what makes the dictionary the smaller form at
//! dataset scale while still losing the per-tile comparison, so the pin flips
//! the column's Arrow type and every tile ships the full dataset list against
//! its own strict subset of keys. `assert_pins_bite` fails the generator rather
//! than let it emit a fixture the pinned path cannot be seen in.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use stt_core::arrow_tile::{
    dataset_dictionary_is_smaller, encode_tile_with, AttrPinned, ColumnarLayer, EncoderConfig,
    GeometryColumn, GlobalColumnPins, GlobalDictVerdict, PropertyColumn,
};
use stt_core::metadata::Metadata;
use stt_core::{BlobOrdering, PackWriter, TileId};

/// Frame-version-coherent encoder config: frames follow the writer's
/// formatVersion (mirroring stt-build's `pack_encoder_config`) instead of the
/// process-global encode default, so a dataset can never mix frame and
/// manifest versions — readers hard-reject mixed datasets (packed-v2 design
/// §1 ★F6). v2 emits SELF-CONTAINED frames (no template collector): the
/// paged + single grid builds below use two writers, and inline schema
/// sections keep their payload bytes byte-shareable without shared-collector
/// plumbing.
///
/// `global_pins` is passed explicitly and is the load-bearing argument:
/// `..EncoderConfig::default()` leaves it `None`, which is the incumbent
/// per-tile path and would make these fixtures invisible to mechanism M2 (see
/// the module header).
fn encoder_config(format_version: u32, pins: &Arc<GlobalColumnPins>) -> EncoderConfig {
    EncoderConfig {
        format_version,
        global_pins: Some(Arc::clone(pins)),
        ..EncoderConfig::default()
    }
}

// ----------------------------------------------------------------------------
// Pass 1 — the dataset-global pins
// ----------------------------------------------------------------------------

/// Category-count ceiling for a hoistable dictionary — mirrors
/// `stt_build::dataset_stats::MAX_HOISTED_CATEGORIES`.
const MAX_HOISTED_CATEGORIES: usize = 1024;
/// Category-UTF-8-byte ceiling — mirrors
/// `stt_build::dataset_stats::MAX_HOISTED_CATEGORY_BYTES`.
const MAX_HOISTED_CATEGORY_BYTES: u64 = 4096;

/// Resolve one dataset's global pins from its own features.
///
/// MIRRORS `stt_build::dataset_stats::DatasetStats::to_pins_with` for the same
/// reason [`required_capabilities`] mirrors `EncoderSettings::required_capabilities`:
/// the dependency runs `stt-build` → `stt-core`, so this crate cannot call the
/// builder. The mirror is kept thin — the verdicts themselves are the shared
/// `stt-core` functions the builder calls ([`AttrPinned::derive_auto`],
/// [`dataset_dictionary_is_smaller`]); only the categorical rule's ORDER and the
/// two hoist caps are transcribed. The distinct-set overflow branch
/// (`MAX_CATEGORIES = 65_536`) is not reproduced: it resolves to the same
/// `Utf8` verdict the size tests already emit, and needs a 65 K-category corpus
/// to reach.
///
/// `layers` must be iterated in WRITE order — the dictionary pin records
/// first-seen order, which is part of the output.
fn derive_pins<'a>(layers: impl IntoIterator<Item = &'a ColumnarLayer>) -> GlobalColumnPins {
    let mut numeric: BTreeMap<&str, (f64, f64, f64, bool, u64)> = BTreeMap::new();
    let mut categorical: BTreeMap<&str, (Vec<String>, u64, u64)> = BTreeMap::new();
    for layer in layers {
        for (name, column) in &layer.properties {
            match column {
                PropertyColumn::Numeric(values) => {
                    let st = numeric.entry(name.as_str()).or_insert((
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        0.0,
                        true,
                        0,
                    ));
                    for v in values.iter().flatten().copied().filter(|v| v.is_finite()) {
                        st.0 = st.0.min(v);
                        st.1 = st.1.max(v);
                        st.2 = st.2.max(v.abs());
                        st.3 &= v.fract() == 0.0;
                        st.4 += 1;
                    }
                }
                PropertyColumn::Categorical(values) => {
                    let st = categorical
                        .entry(name.as_str())
                        .or_insert((Vec::new(), 0, 0));
                    for v in values.iter().flatten() {
                        // `Vec` + linear scan, never a `HashSet`: first-seen
                        // order is part of the emitted pin.
                        if !st.0.iter().any(|c| c == v) {
                            st.0.push(v.clone());
                        }
                        st.1 += 1;
                        st.2 += v.len() as u64;
                    }
                }
                // Vector groups carry no scalar affine and no dictionary
                // verdict, so `GlobalColumnPins` has no key for them. Explicit
                // so adding one is a compile error rather than a silent gap.
                PropertyColumn::Vector { .. } => {}
            }
        }
    }

    let mut attr = BTreeMap::new();
    for (name, (min, max, max_abs, all_integer, finite)) in numeric {
        attr.insert(
            name.to_string(),
            AttrPinned::derive_auto(min, max, max_abs, all_integer, finite),
        );
    }
    let mut dict = BTreeMap::new();
    for (name, (categories, values, total_value_bytes)) in categorical {
        let category_bytes: u64 = categories.iter().map(|c| c.len() as u64).sum();
        let k = categories.len();
        // The builder's order: the dataset-scale wire test first, then the
        // reader's resident-memory caps.
        let verdict =
            if !dataset_dictionary_is_smaller(total_value_bytes, values, category_bytes, k as u64)
                || k > MAX_HOISTED_CATEGORIES
                || category_bytes > MAX_HOISTED_CATEGORY_BYTES
            {
                GlobalDictVerdict::Utf8
            } else {
                GlobalDictVerdict::Dictionary(Arc::new(categories))
            };
        dict.insert(name.to_string(), verdict);
    }
    GlobalColumnPins { attr, dict }
}

/// Refuse to emit a fixture whose pins decided nothing.
///
/// A `Utf8` verdict for `kind` is what a corpus below the dataset-scale
/// crossover resolves to — legal, and byte-identical to what the per-tile rule
/// already picks for tiles this small. That is the blind fixture this whole
/// two-pass shape exists to prevent, so it stops the generator here rather than
/// leaving it to be discovered when the pin fails to catch something.
fn assert_pins_bite(dataset: &str, pins: &GlobalColumnPins) {
    assert!(
        pins.attr.contains_key("speed"),
        "[{dataset}] `speed` must resolve a dataset-global affine"
    );
    let categories = match pins.dict.get("kind") {
        Some(GlobalDictVerdict::Dictionary(c)) => c.len(),
        other => panic!(
            "[{dataset}] `kind` must pin a Dictionary, got {other:?} — the corpus \
             has fallen below the dataset-scale crossover and this fixture would \
             pin nothing new (see KINDS)"
        ),
    };
    assert_eq!(
        categories,
        KINDS.len(),
        "[{dataset}] the pinned list must be the whole dataset domain"
    );
}

/// The `manifest.capabilities` entries an [`EncoderConfig`] IMPLIES —
/// required-to-understand declarations (packed spec §3.1), derived from the
/// one config this fixture actually encodes with so the archive can never USE
/// a re-typing it fails to DECLARE. Without this the fixture would ship
/// `TILE_META.st`/`.et` (compact times are on by default) under a manifest
/// with no `capabilities` key, i.e. a NON-CONFORMANT archive pinned as the
/// cross-implementation golden — exactly the silent-misdecode the key exists
/// to prevent.
///
/// Mirrors `stt-build`'s `EncoderSettings::required_capabilities()` field for
/// field; `stt-core` cannot depend on `stt-build` (the dependency runs the
/// other way), so the derivation is repeated here but is still driven by the
/// SAME `EncoderConfig` value the encoder receives, not by a hand-kept
/// parallel list. Additive features (vector groups, vertex-time precision)
/// are deliberately not capabilities.
fn required_capabilities(cfg: &EncoderConfig) -> Vec<String> {
    use stt_core::pack::{
        CAPABILITY_ATTR_QUANT, CAPABILITY_COORD_QUANT, CAPABILITY_ELEVATION_FOLD,
        CAPABILITY_TIME_DELTA, CAPABILITY_VERTEX_VALUE_QUANT,
    };
    let mut caps: Vec<String> = Vec::new();
    if cfg.quantize_coords_m.is_some() {
        caps.push(CAPABILITY_COORD_QUANT.to_string());
    }
    if !cfg.quantize_attrs.is_empty() || cfg.quantize_attrs_auto {
        caps.push(CAPABILITY_ATTR_QUANT.to_string());
    }
    if !cfg.point_elevation_column.is_empty() {
        caps.push(CAPABILITY_ELEVATION_FOLD.to_string());
    }
    if cfg.compact_times {
        caps.push(CAPABILITY_TIME_DELTA.to_string());
    }
    if cfg.quantize_vertex_values {
        caps.push(CAPABILITY_VERTEX_VALUE_QUANT.to_string());
    }
    caps
}

/// The dataset-global category list for `kind`, in the order the corpus first
/// presents them (tile seed 0 mints the first pair, seed 1 the second).
///
/// Long on purpose, and split into per-tile PAIRS on purpose. The categorical
/// verdict is a measurement — `dataset_dictionary_is_smaller` — and with
/// three-letter categories over this many cells the dictionary genuinely loses
/// to plain `Utf8`, which would leave these fixtures pinning a verdict that
/// changes nothing. At these lengths the dictionary wins at DATASET scale while
/// still losing the PER-TILE comparison, so the pin flips the Arrow type in
/// every tile; and because each tile uses two of the four, every tile ships the
/// full dataset list against a strict subset of keys — the all-or-nothing hoist
/// contract, made visible in the cross-impl bytes.
const KINDS: [&str; 4] = [
    "articulated-electric-transit-bus",
    "low-floor-light-rail-tram-vehicle",
    "harbour-passenger-ferry-vessel",
    "dockless-shared-electric-scooter",
];

/// Build the deterministic LAYER for tile index `k`. `id_seed` lets the deduped
/// tiles (k=4, k=9) reuse k=0's identity bytes.
///
/// Split from the encode so pass 1 can scan the dataset's features before any
/// config exists — the pins are an input to the encode, not an output of it.
fn layer_for(id_seed: u64) -> ColumnarLayer {
    let ids: Vec<u64> = (0..3).map(|i| 100 * id_seed + i).collect();
    let n = ids.len();
    // Two of the four categories per tile, alternating by seed parity, so the
    // first two seeds between them mint the whole list in declaration order.
    let pair = (2 * id_seed as usize) % KINDS.len();
    ColumnarLayer {
        polygon_parts: None,
        name: "default".to_string(),
        feature_ids: ids,
        start_times: vec![1000 * id_seed as i64; n],
        end_times: vec![1000 * id_seed as i64 + 100; n],
        geometry: GeometryColumn::Point(vec![[-122.4 + 0.01 * id_seed as f64, 37.7]; n]),
        vertex_times: None,
        vertex_values: None,
        triangles: None,
        vertex_value_matrix: None,
        // A numeric and a categorical column, each carrying a null, so the
        // cross-impl readers exercise BOTH property kinds (and their null
        // sentinels) against real writer output.
        properties: vec![
            (
                "speed".to_string(),
                PropertyColumn::Numeric(vec![
                    Some(10.0 + id_seed as f64),
                    None,
                    Some(30.0 + id_seed as f64),
                ]),
            ),
            (
                "kind".to_string(),
                PropertyColumn::Categorical(vec![
                    Some(KINDS[pair].to_string()),
                    Some(KINDS[pair + 1].to_string()),
                    None,
                ]),
            ),
        ],
    }
}

/// Encode one layer under the (pinned) config.
fn payload_for(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Vec<u8> {
    encode_tile_with(std::slice::from_ref(layer), cfg).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One format, one fixture set: the writer and the payload frames both use
    // `PACKED_FORMAT_VERSION` (see `encoder_config`), so a fixture can never
    // pair a manifest of one version with frames of another.
    if let Some(other) = std::env::args().nth(1) {
        return Err(format!("unknown argument {other:?} (this example takes none)").into());
    }
    let format_version = stt_core::arrow_tile::LAYER_FRAME_VERSION;
    let suffix = "";

    // --- Pass 1 (packed-golden) ------------------------------------------
    // The dataset is the 12 tiles as WRITTEN, so the deduped k=4/k=9 present
    // seed 0's features three times — the same stream the builder's pass 1
    // would scan. Order is write order, which is what makes the dictionary's
    // first-seen ordering well defined.
    let n_tiles = 12u64;
    let seed_of = |k: u64| if k == 4 || k == 9 { 0 } else { k };
    let layers: Vec<ColumnarLayer> = (0..n_tiles).map(|k| layer_for(seed_of(k))).collect();
    let pins = Arc::new(derive_pins(layers.iter()));
    assert_pins_bite("packed-golden", &pins);
    // --- Pass 2 -----------------------------------------------------------
    let cfg = encoder_config(format_version, &pins);

    // <crate>/../../packages/core/test/fixtures/packed-golden
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = manifest_dir
        .join("..")
        .join("..")
        .join("packages")
        .join("core")
        .join("test")
        .join("fixtures")
        .join(format!("packed-golden{suffix}"));

    // Start clean so re-running is deterministic (stale packs would linger
    // otherwise — content-addressed names mean old objects never get clobbered).
    if out_dir.exists() {
        std::fs::remove_dir_all(&out_dir)?;
    }

    // SpatialMajor keeps the order deterministic + independent of the Auto
    // heuristic, so the content hashes are stable across builds.
    //
    // ⚠ The explicit ordering is LOAD-BEARING and must stay explicit: the
    // `stt-build` CLI defaults to `--blob-ordering measured`, so a fixture that
    // inherited a default would let a change to the ordering picker silently
    // move the golden pins. Golden pins move once, deliberately, under review.
    let mut w = PackWriter::create(&out_dir, BlobOrdering::SpatialMajor, 4 * 1024)?
        // Declared from `cfg`, the very config the payloads below are encoded
        // with — a fixture that used a re-typing without declaring it would
        // pin a spec violation as the reference bytes.
        .with_capabilities(required_capabilities(&cfg));

    // `encode_tile` is NOT byte-deterministic across separate calls (Arrow IPC
    // framing), so build each distinct payload exactly ONCE and CLONE it for the
    // dedup cases — only the clone is byte-identical, which is what the writer's
    // blake3 dedup keys on. (This also makes the committed content hashes stable
    // across regenerations of the fixture.)
    let mut distinct: std::collections::HashMap<u64, Vec<u8>> = std::collections::HashMap::new();
    for k in 0..n_tiles {
        // k=4 and k=9 reuse k=0's identity → byte-identical blobs → dedup.
        let id_seed = seed_of(k);
        let payload = distinct
            .entry(id_seed)
            .or_insert_with(|| payload_for(&layers[k as usize], &cfg))
            .clone();
        let t = 1000 * k as i64;
        // Distinct spatial cell per tile so all 12 directory entries survive.
        let id = TileId::new(10, k as u32, 0, t.max(0) as u64);
        w.add_tile_full(&id, t, t + 100, Some(t), 3, Some(1000), &payload)?;
    }

    let meta = Metadata::new("packed-golden")
        .with_description("Deterministic STT packed-format cross-impl fixture")
        .with_zoom_levels(10, 10)
        .with_temporal_bucket_ms(1000)
        // Tile k spans [1000*k, 1000*k + 100]; cover the lot so the fixture
        // passes `stt-validate`'s temporal-extent cross-check.
        .with_time_range(stt_core::types::TimeRange::new(
            0,
            1000 * (n_tiles - 1) + 100,
        ));

    let manifest = w.finalize(&meta)?;
    let total: u64 = manifest.packs.iter().map(|p| p.length).sum();

    println!(
        "wrote golden fixture to {}: {} tiles, {} packs, {} pack bytes, dir {} bytes",
        out_dir.display(),
        n_tiles,
        manifest.packs.len(),
        total,
        manifest.directory.length,
    );
    for (i, p) in manifest.packs.iter().enumerate() {
        println!("  pack[{i}] {} ({} bytes)", p.key, p.length);
    }
    println!(
        "  directory {} ({} bytes)",
        manifest.directory.key, manifest.directory.length
    );

    // --- Paged-directory cross-impl fixtures --------------------------------
    // A richer grid dataset (spatial spread × time buckets × two zooms) emitted
    // BOTH paged (`paged-golden/`) and single (`paged-golden-single/`) from the
    // SAME shared payloads — so the TS differential test can assert the paged
    // query path returns byte-identical results to a whole-load directory while
    // fetching only a fraction of the leaf pages.
    let base = manifest_dir
        .join("..")
        .join("..")
        .join("packages")
        .join("core")
        .join("test")
        .join("fixtures");
    // The grid is a DIFFERENT dataset, so it gets its OWN pins — a pin is a
    // property of a dataset domain, and reusing packed-golden's would encode a
    // claim about a corpus these features are not part of (and would error out
    // if a category here were absent from that one). The paged and single
    // builds below share this single pin set, because they ARE one dataset in
    // two container shapes and their packs must stay byte-identical.
    let grid_layers = build_grid_layers();
    let grid_pins = Arc::new(derive_pins(grid_layers.iter().map(|(_, _, _, l)| l)));
    assert_pins_bite("paged-golden", &grid_pins);
    let grid_cfg = encoder_config(format_version, &grid_pins);
    let grid: Vec<(TileId, i64, i64, Vec<u8>)> = grid_layers
        .iter()
        .map(|(id, ts, te, layer)| (*id, *ts, *te, payload_for(layer, &grid_cfg)))
        .collect();
    let grid_meta = Metadata::new("paged-golden")
        .with_description("Deterministic STT paged-directory cross-impl fixture")
        .with_zoom_levels(10, 12)
        .with_temporal_bucket_ms(3_600_000)
        .with_time_range(stt_core::types::TimeRange::new(0, 3 * 3_600_000));
    for (sub, paging) in [
        (format!("paged-golden{suffix}"), Some(8usize)),
        (format!("paged-golden-single{suffix}"), None),
    ] {
        let dir = base.join(&sub);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        // SpatialMajor → deterministic order + stable content hashes; identical
        // shared payload bytes → the two builds' packs are byte-identical and
        // only the directory container differs. Explicit on purpose — see the
        // pin note above; never let this follow the CLI's `measured` default.
        let mut gw = PackWriter::create(&dir, BlobOrdering::SpatialMajor, 8 * 1024)?
            .with_capabilities(required_capabilities(&cfg))
            .with_paging(paging);
        for (id, ts, te, payload) in &grid {
            gw.add_tile_full(id, *ts, *te, Some(*ts), 3, Some(3_600_000), payload)?;
        }
        let gm = gw.finalize(&grid_meta)?;
        println!(
            "wrote {} to {}: {} tiles, {} packs, dir {} bytes{}",
            sub,
            dir.display(),
            grid.len(),
            gm.packs.len(),
            gm.directory.length,
            gm.directory
                .page_count
                .map(|p| format!(
                    " (paged: {p} leaf pages, root {} B)",
                    gm.directory.root_length.unwrap_or(0)
                ))
                .unwrap_or_default(),
        );
    }
    Ok(())
}

/// Both cross-impl datasets resolve a BITING pin set — checkable without
/// running the generator, which would overwrite the committed fixtures.
///
/// `main`'s `assert_pins_bite` calls are the backstop for whoever regenerates
/// (TB-14); this is the same claim available ahead of time:
///
///   cargo test -p stt-core --examples
///
/// Not part of the default `cargo test -p stt-core` run — example tests never
/// are — so it is a tool for the regenerator, not a gate. The gate for the same
/// property on the WRITER-side corpus is
/// `tests/v2_golden.rs::golden_fixture_exercises_the_pinned_path`, which does
/// run by default.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_cross_impl_datasets_pin_a_dictionary() {
        let seed_of = |k: u64| if k == 4 || k == 9 { 0 } else { k };
        let packed: Vec<ColumnarLayer> = (0..12u64).map(|k| layer_for(seed_of(k))).collect();
        let packed_pins = derive_pins(packed.iter());
        assert_pins_bite("packed-golden", &packed_pins);

        let grid = build_grid_layers();
        let grid_pins = derive_pins(grid.iter().map(|(_, _, _, l)| l));
        assert_pins_bite("paged-golden", &grid_pins);

        // Every tile holds a STRICT subset of the pinned list, so "ship the
        // whole list in every tile" is a claim the cross-impl bytes can
        // disagree with.
        let GlobalDictVerdict::Dictionary(categories) = packed_pins.dict.get("kind").unwrap()
        else {
            unreachable!("asserted above")
        };
        for layer in &packed {
            let Some((_, PropertyColumn::Categorical(values))) =
                layer.properties.iter().find(|(n, _)| n == "kind")
            else {
                continue;
            };
            let mut distinct: Vec<&str> = Vec::new();
            for v in values.iter().flatten() {
                if !distinct.contains(&v.as_str()) {
                    distinct.push(v.as_str());
                }
            }
            assert!(
                !distinct.is_empty() && distinct.len() < categories.len(),
                "each tile must hold a non-empty PROPER subset of the pinned list"
            );
        }
    }

    /// The pins actually move bytes: each dataset's first tile encodes
    /// differently with them than without. Two configs differing ONLY in
    /// `global_pins`.
    #[test]
    fn the_pins_change_the_encoded_bytes() {
        let format_version = stt_core::arrow_tile::LAYER_FRAME_VERSION;
        let packed: Vec<ColumnarLayer> = (0..12u64)
            .map(|k| layer_for(if k == 4 || k == 9 { 0 } else { k }))
            .collect();
        let pins = Arc::new(derive_pins(packed.iter()));
        let pinned = encoder_config(format_version, &pins);
        let unpinned = EncoderConfig {
            format_version,
            ..EncoderConfig::default()
        };
        for layer in &packed {
            assert_ne!(
                payload_for(layer, &unpinned),
                payload_for(layer, &pinned),
                "this tile is invisible to the dataset-global pins"
            );
        }
    }
}

/// A spread-out grid of tiles across two zooms and three time buckets, as
/// LAYERS — pass 1 scans these, pass 2 encodes each exactly once (so the paged
/// and single builds share byte-identical blobs). ~250 entries → several leaf
/// pages at page size 8.
fn build_grid_layers() -> Vec<(TileId, i64, i64, ColumnarLayer)> {
    let mut out = Vec::new();
    let mut seed = 1_000u64;
    let bucket = 3_600_000i64;
    // Zoom 10: an 18×12 block over Europe-ish tile coords.
    for gx in 0..18u32 {
        for gy in 0..12u32 {
            let (x, y) = (520 + gx, 384 + gy);
            let b = ((gx + gy) % 3) as i64;
            let t = b * bucket;
            out.push((
                TileId::new(10, x, y, t.max(0) as u64),
                t,
                t + bucket - 1,
                layer_for(seed),
            ));
            seed += 1;
        }
    }
    // Zoom 12: a 6×6 block (distinct zoom exercises zoom-range pruning).
    for gx in 0..6u32 {
        for gy in 0..6u32 {
            let (x, y) = (2084 + gx, 1536 + gy);
            let b = ((gx * 7 + gy) % 3) as i64;
            let t = b * bucket;
            out.push((
                TileId::new(12, x, y, t.max(0) as u64),
                t,
                t + bucket - 1,
                layer_for(seed),
            ));
            seed += 1;
        }
    }
    out
}
