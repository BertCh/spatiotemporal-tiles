//! The v2 → v3 container migration, and the read-window holes it exposed.
//!
//! The migration's whole claim is that a v3 archive can be produced from a v2
//! one WITHOUT re-deriving anything: rewrite the manifest, re-encode the
//! directory under codec v6, and leave every pack byte where it is. These tests
//! hold that claim to its two halves — the result must be genuinely v3, and it
//! must decode to exactly what the v2 archive decoded to.
//!
//! The fixtures are the frozen `legacy-shape` archives under
//! `packages/core/test/fixtures/`, which are real published v2 output rather
//! than something synthesized here. Two of the four (`flows`, `currents`) are
//! hand-authored for the TS reader with literal `legacy.sttd` / `legacy.sttp`
//! object keys instead of content addresses, which the Rust reader refuses by
//! design — so they are not usable here and are deliberately not listed.

use std::path::{Path, PathBuf};
use stt_core::pack::{migrate_dataset_v2_to_v3, PackedReader};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packages/core/test/fixtures/legacy-shape")
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let dst = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &dst)?;
        } else {
            std::fs::copy(e.path(), dst)?;
        }
    }
    Ok(())
}

/// Every layer's `(name, row_count)` for every entry, in directory order — the
/// comparison that would catch a re-encode dropping, reordering or re-addressing
/// tiles.
fn decode_all(dir: &Path) -> Vec<(u8, u32, u32, i64, Vec<(String, usize)>)> {
    let r = PackedReader::open(dir.join("manifest.json")).expect("open");
    r.entries()
        .iter()
        .map(|e| {
            let layers = r.read_layers(e).expect("read layers");
            (
                e.zoom,
                e.x,
                e.y,
                e.time_start,
                layers
                    .iter()
                    .map(|l| (l.name.clone(), l.batch.num_rows()))
                    .collect(),
            )
        })
        .collect()
}

fn staged(name: &str, tag: &str) -> PathBuf {
    let dst = std::env::temp_dir().join(format!("stt-migrate-{tag}-{name}"));
    let _ = std::fs::remove_dir_all(&dst);
    copy_dir(&fixtures().join(name), &dst).expect("stage fixture");
    dst
}

#[test]
fn migrating_a_v2_archive_preserves_every_decoded_tile() {
    // `points` is paged, `tracks` is single — both directory layouts, since the
    // re-encode path forks on exactly that.
    for name in ["points", "tracks"] {
        let dir = staged(name, "preserve");
        let before = decode_all(&dir);
        assert!(!before.is_empty(), "{name}: fixture decoded nothing");

        let report = migrate_dataset_v2_to_v3(&dir)
            .expect("migration should succeed")
            .expect("a v2 archive should report work done");

        let after = decode_all(&dir);
        assert_eq!(
            before, after,
            "{name}: decoded content changed across the migration"
        );
        assert_eq!(report.entries, before.len(), "{name}: entry count moved");
    }
}

#[test]
fn the_migrated_archive_is_really_v3_and_the_packs_are_untouched() {
    let dir = staged("points", "shape");

    let manifest_before: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let packs_before = manifest_before["packs"].clone();
    // Hash every pack so "untouched" means BYTES, not just names.
    let pack_bytes_before: Vec<Vec<u8>> = packs_before
        .as_array()
        .unwrap()
        .iter()
        .map(|p| std::fs::read(dir.join(p["key"].as_str().unwrap())).unwrap())
        .collect();

    let report = migrate_dataset_v2_to_v3(&dir).unwrap().unwrap();

    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["formatVersion"], 3, "manifest is not v3");
    assert_eq!(
        m["directory"]["directoryVersion"], 6,
        "v3 requires directory codec v6"
    );
    // The implicit registry a v3 reader already infers for a v2 archive, now
    // stated. Raw only — v2 had no way to mark anything else.
    assert_eq!(m["variants"].as_array().map(|v| v.len()), Some(1));
    assert_eq!(m["variants"][0]["id"], 0);
    assert_eq!(m["variants"][0]["kind"], "raw");

    assert_eq!(m["packs"], packs_before, "pack table was rewritten");
    for (p, before) in m["packs"]
        .as_array()
        .unwrap()
        .iter()
        .zip(pack_bytes_before.iter())
    {
        let now = std::fs::read(dir.join(p["key"].as_str().unwrap())).unwrap();
        assert_eq!(&now, before, "a pack object's BYTES changed");
    }
    assert_ne!(
        report.old_directory_key, report.new_directory_key,
        "the directory is re-encoded, so its content address must move"
    );
    // The old directory object stays on disk: it is content-addressed and a
    // reader still holding the previous manifest must keep resolving it.
    assert!(dir.join(&report.old_directory_key).exists());
}

#[test]
fn migration_is_idempotent_and_refuses_what_it_cannot_answer() {
    let dir = staged("tracks", "idem");
    assert!(migrate_dataset_v2_to_v3(&dir).unwrap().is_some());
    // Second pass: already v3, nothing to do, and NOT an error.
    assert!(
        migrate_dataset_v2_to_v3(&dir).unwrap().is_none(),
        "a v3 archive must be left alone"
    );

    // A summary tier is the case migration must refuse rather than guess: a v2
    // directory records no variant column, so which entries are aggregates
    // simply is not in the archive.
    let dir = staged("points", "summary");
    let mp = dir.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
    m["metadata"]["summary_tier"] = serde_json::json!({
        "scheme": "h3",
        "min_zoom": 0,
        "max_zoom": 4,
        "cell_resolution_per_zoom": [0, 1, 2, 3, 4],
        "columns": [],
    });
    std::fs::write(&mp, serde_json::to_vec(&m).unwrap()).unwrap();
    let err = migrate_dataset_v2_to_v3(&dir).unwrap_err().to_string();
    assert!(
        err.contains("summary tier"),
        "expected a summary-tier refusal, got: {err}"
    );
}

#[test]
fn a_v2_archive_with_a_summary_tier_still_opens() {
    // The read-window hole this work exposed, pinned separately from migration.
    // `SummaryTier::variant_id` had no serde default, so a v2 manifest — which
    // predates the variant axis and cannot carry one — failed to deserialize at
    // all: "missing field `variant_id`". Every Rust tool therefore refused six
    // published archives that the TypeScript reader opened without complaint.
    let dir = staged("points", "openable");
    let mp = dir.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
    m["metadata"]["summary_tier"] = serde_json::json!({
        "scheme": "h3",
        "min_zoom": 0,
        "max_zoom": 4,
        "cell_resolution_per_zoom": [0, 1, 2, 3, 4],
        "columns": [],
    });
    std::fs::write(&mp, serde_json::to_vec(&m).unwrap()).unwrap();

    let r = PackedReader::open(dir.join("manifest.json"))
        .expect("a v2 archive with a summary tier must open");
    assert_eq!(r.format_version(), 2);
    assert!(!r.entries().is_empty());
}
