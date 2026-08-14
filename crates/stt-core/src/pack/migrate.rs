//! Container migration: packed `formatVersion` 2 → 3, in place of a rebuild.
//!
//! # Why this exists at all
//!
//! The v3 break is **container-only**. The manifest gains a required `variants`
//! registry, the directory codec goes v5 → v6 to carry a `variant_id` per
//! entry, and the object magic's version byte moves 2 → 3. Nothing below the
//! container changes: the layer frame, every column encoding, every quantization
//! affine and every capability are byte-identical across the break. That is
//! stated normatively in `docs/spec/stt-packed-format.md` §9.1 and is what the
//! reader's v2 read window rests on.
//!
//! So a v2 archive does not need to be re-derived from its source to become a
//! v3 archive. It needs a new directory object and a new manifest. This module
//! writes exactly those two, and **does not touch a single pack**.
//!
//! # Why that matters more than it sounds
//!
//! Rebuilding the shipped fleet from source was the obvious plan and it fails on
//! three independent counts, each measured rather than assumed:
//!
//! * **It cannot reach every archive.** Three published datasets have no
//!   reproducible source at all (a synthetic set with no recipe, a summary with
//!   no generator, and one needing a login-gated full-history extract), and
//!   `wildfires` has lost its upstream coverage entirely — a rebuild returns 15
//!   source perimeters where the shipped archive holds ~460.
//! * **It changes the data.** Today's defaults preserve every usable row at
//!   every zoom, per the project's no-thinning rule, while the shipped archives
//!   were built thinned. Measured: `ais-all-us` rebuilds 6.45× larger,
//!   `flights` 5.93×. Those are better artifacts and a different product.
//! * **It costs the packs.** A rebuild re-encodes every blob, so every
//!   content address moves and the whole fleet re-uploads — 29.3 GiB / 1,324
//!   objects the last time that happened.
//!
//! Migration has none of those properties. Pack objects keep their bytes, keep
//! their names, and therefore **stay where they already are on the CDN**: the
//! deploy is one small directory object plus one manifest per dataset.
//!
//! # Why the untouched packs are still legal under v3
//!
//! Because the reader's floor is the OBJECT magic version, not the manifest's.
//! Both reference implementations accept `MIN_OBJECT_MAGIC_VERSION ..=
//! OBJECT_MAGIC_VERSION` (2..=3) on every `.sttp`, independently of
//! `formatVersion`. A v3 manifest addressing v2-magic packs is therefore inside
//! the contract, not a loophole — the magic version identifies the OBJECT
//! layout, which did not change, while `formatVersion` identifies the
//! ADDRESSING model, which did.
//!
//! The directory is the one object that must be rewritten, because v6 is where
//! `variant_id` lives and a v3 manifest is required to carry codec v6
//! (`pack::validate_manifest` couples them deliberately: "a v3 manifest claiming
//! directory v4 is drift, not history").
//!
//! # What this is NOT
//!
//! Not a transcoder, and it must not grow into one. There is no v1 path, no
//! downgrade, and no payload rewriting anywhere in this file — if a future break
//! moves tile bytes, this module cannot carry it and a rebuild is the only
//! answer. The v2 read window exists so published archives are not stranded;
//! this exists so they do not have to stay behind either.

use super::{
    blake3_128_hex, decode_directory_entries, object_magic, DirectoryRef, Manifest,
    ManifestVariant, VariantKind, DIRECTORY_ENCODING_ZSTD, DIRECTORY_LAYOUT_PAGED, DIRECTORY_MAGIC,
    OBJECT_MAGIC_LEN, PACKED_FORMAT_VERSION,
};
use crate::compression;
use crate::error::{Error, Result};
use std::fs;
use std::path::Path;

/// zstd level for the re-encoded directory object. 19 is the writer's `--publish`
/// tuning, and every archive this runs against is a published one — matching it
/// keeps the migrated directory the same size class as a freshly built peer.
/// Level affects only the at-rest bytes: decode is level-independent.
const MIGRATE_ZSTD_LEVEL: i32 = 19;

/// What a migration did, for the caller to log or assert on.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    /// Dataset directory that was migrated.
    pub dataset: String,
    /// Directory entries carried across. Unchanged by definition — the codec
    /// re-encode is lossless and every entry keeps its `(z, x, y, t)` address.
    pub entries: usize,
    /// Directory object key before and after. The bytes change (v6 codec, magic
    /// version 3) so the content address must too.
    pub old_directory_key: String,
    pub new_directory_key: String,
    /// Packs referenced. Always untouched — recorded so a caller can assert it.
    pub packs_unchanged: usize,
    /// `true` when the source directory used the paged layout, which is
    /// preserved: a paged v5 directory migrates to a paged v6 one, so the
    /// reader's cold-start request pattern is identical before and after.
    pub paged: bool,
}

/// Migrate one packed dataset directory from `formatVersion` 2 to 3.
///
/// Rewrites `manifest.json` and writes one new `index/<hash>.sttd`. Pack objects
/// are never read, written, or renamed. The previous directory object is left in
/// place: it is content-addressed, so it simply becomes unreferenced, and the
/// deploy's own retention rules are what should reap it — deleting it here would
/// break any reader still holding the old manifest.
///
/// Idempotent in the useful sense: an archive already at v3 is left untouched
/// and reported as such via `Ok(None)`.
pub fn migrate_dataset_v2_to_v3(dataset_dir: &Path) -> Result<Option<MigrationReport>> {
    let manifest_path = dataset_dir.join("manifest.json");
    let raw = fs::read(&manifest_path)?;
    let mut manifest: Manifest = serde_json::from_slice(&raw)
        .map_err(|e| Error::InvalidArchive(format!("manifest parse: {e}")))?;

    if manifest.format_version == PACKED_FORMAT_VERSION {
        return Ok(None);
    }
    if manifest.format_version != 2 {
        return Err(Error::InvalidArchive(format!(
            "migration covers formatVersion 2 → {PACKED_FORMAT_VERSION}; found {}",
            manifest.format_version
        )));
    }

    // --- Summary tiers cannot be migrated, and that is not a shortcut --------
    //
    // v3's whole purpose is to separate raw and summary products that v2 forced
    // to share `(z, x, y, t)`. A v2 directory has no column recording which of
    // its entries is which, so the information required to split them DOES NOT
    // EXIST in the archive — it was only ever implicit in how the tiles were
    // written. Migrating anyway would mean guessing (by zoom band, say), and a
    // wrong guess relabels aggregate cells as raw features with nothing to
    // catch it. Refuse, and let a rebuild — which knows which tier it is
    // emitting — be the only path for these.
    if manifest.metadata.summary_tier.is_some() {
        return Err(Error::InvalidArchive(
            "cannot migrate an archive with a summary tier: a v2 directory does not record \
             which entries are summary, so the raw/summary split v3 requires cannot be \
             recovered from the archive. Rebuild it instead."
                .into(),
        ));
    }

    // --- Read the existing directory exactly as the reader would -------------
    let old_key = manifest.directory.key.clone();
    let dir_path = dataset_dir.join(&old_key);
    let dir_bytes = fs::read(&dir_path)?;
    let codec = super::directory_codec_bytes(&dir_bytes, manifest.format_version)?;
    let expected = manifest.metadata.tile_count.max(1);
    let entries = decode_directory_entries(codec, &manifest.directory, expected)?;

    // --- The one case this cannot answer -------------------------------------
    //
    // v2 had no variant axis, so a v2 directory cannot say which of its entries
    // were summary tiles — that ambiguity is precisely what v3 exists to remove.
    // Every entry therefore migrates as variant 0 (raw), which is exactly how
    // the v3 reader already interprets a v2 archive, so behaviour is preserved
    // by construction. But if two entries then collide on the v6 key, the
    // archive genuinely does mix products at one address and only a rebuild can
    // separate them. Refuse rather than emit a directory with duplicate keys.
    let mut seen = std::collections::HashSet::with_capacity(entries.len());
    for e in &entries {
        if !seen.insert((e.zoom, e.x, e.y, e.time_start, e.variant_id)) {
            return Err(Error::InvalidArchive(format!(
                "cannot migrate: entries collide at z{}/{}/{} t={} once addressed by variant \
                 (a v2 archive mixing raw and summary tiles at one address — rebuild it instead)",
                e.zoom, e.x, e.y, e.time_start
            )));
        }
    }

    // --- Re-encode the directory under v6 ------------------------------------
    //
    // Layout is preserved, not re-decided: a paged source stays paged with the
    // same `pageEntries`, so the reader fetches the same shape of ranges it does
    // today. Re-planning the pages here would silently change cold-start cost.
    let paged = manifest.directory.is_paged();
    let zstd_level = MIGRATE_ZSTD_LEVEL;
    let (index_bytes, dref_fields) = if paged {
        let k = manifest.directory.page_entries.ok_or_else(|| {
            Error::InvalidArchive("paged directory: manifest missing pageEntries".into())
        })? as usize;
        let p = crate::directory_page::encode_paged_directory_level(&entries, k, true, zstd_level)?;
        (
            p.bytes,
            (
                Some(DIRECTORY_LAYOUT_PAGED.to_string()),
                Some(p.root_length),
                Some(p.page_count as u64),
                Some(p.page_entries as u64),
                Some(p.root_hash),
                Some(p.page_hashes),
            ),
        )
    } else {
        let plain = crate::directory::encode_directory(&entries);
        let bytes = compression::compress_zstd_with_dict_level(&plain, None, zstd_level)?;
        (bytes, (None, None, None, None, None, None))
    };

    let mut with_magic = Vec::with_capacity(OBJECT_MAGIC_LEN + index_bytes.len());
    with_magic.extend_from_slice(&object_magic(DIRECTORY_MAGIC));
    with_magic.extend_from_slice(&index_bytes);
    let new_key = format!("index/{}.sttd", blake3_128_hex(&with_magic));

    // --- Write the new objects ----------------------------------------------
    let new_path = dataset_dir.join(&new_key);
    if let Some(parent) = new_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&new_path, &with_magic)?;

    manifest.format_version = PACKED_FORMAT_VERSION;
    // The implicit registry a v3 reader already infers for a v2 archive, now
    // stated explicitly because v3 requires it.
    manifest.variants = vec![ManifestVariant {
        id: 0,
        kind: VariantKind::Raw,
        layer_name: None,
        method: None,
        params: None,
    }];
    manifest.directory = DirectoryRef {
        key: new_key.clone(),
        length: with_magic.len() as u64,
        directory_version: crate::directory::DIRECTORY_VERSION,
        encoding: Some(DIRECTORY_ENCODING_ZSTD.to_string()),
        layout: dref_fields.0,
        root_length: dref_fields.1,
        page_count: dref_fields.2,
        page_entries: dref_fields.3,
        root_hash: dref_fields.4,
        page_hashes: dref_fields.5,
    };

    let manifest_bytes = manifest.to_json_bytes()?;
    fs::write(&manifest_path, &manifest_bytes)?;

    Ok(Some(MigrationReport {
        dataset: dataset_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        entries: entries.len(),
        old_directory_key: old_key,
        new_directory_key: new_key,
        packs_unchanged: manifest.packs.len(),
        paged,
    }))
}
