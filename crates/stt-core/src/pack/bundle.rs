//! Bundle profile (`.sttb`) — single-file interchange (spec §13, DRAFT).
//!
//! A bundle is the packed dataset as ONE file, restoring the "download one
//! file" usability property the exploded layout gave up. Strictly an
//! interchange profile: the CDN story remains the exploded layout — nothing
//! serves bundles over HTTP ranges. Layout:
//!
//! ```text
//! "STTB"  u8 version(1)  [3 × 0x00]   # 8-byte magic prelude
//! u32     header_len                  # little-endian
//! [header JSON]                       # { manifest: <verbatim>, objects: [..] }
//! [zero pad to 8]
//! [objects back-to-back at 8-byte-aligned offsets]
//! ```
//!
//! The container keeps object keys + bytes opaque, but this toolchain enforces
//! the same single v3 archive contract for bundles and exploded datasets. It
//! reuses the parent module's manifest model, content-address helper and
//! directory decode, so a bundle is verifiable by exactly the same checks.

use super::{
    blake3_128_hex, decode_directory_entries, directory_codec_bytes, manifest_open_checks,
    strip_object_magic, verify_manifest_schemas, verify_v2_frame_template_refs, write_atomic,
    Manifest, DIRECTORY_ENCODING_ZSTD, DIRECTORY_MAGIC, PACKED_FORMAT, PACKED_FORMAT_VERSION,
    PACK_MAGIC,
};
use crate::error::{Error, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// First 4 bytes of a `.sttb` bundle file.
pub const BUNDLE_MAGIC: [u8; 4] = *b"STTB";
/// Bundle container version this toolchain writes and reads.
pub const BUNDLE_VERSION: u8 = 1;
/// Fixed bundle prelude: 8-byte magic (`"STTB"`, version, 3 × 0x00) + the
/// little-endian `u32` header length.
const BUNDLE_PRELUDE_LEN: usize = 12;

/// One object entry in the bundle header's `objects` table.
///
/// `offset`/`length` are byte positions within the bundle file, emitted as
/// JSON numbers — exact for u64 in Rust, and fine to 2^53 for JS readers
/// (a 9-PB bundle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleObject {
    /// Object key, verbatim from the manifest (e.g. `packs/<hash>.sttp`).
    pub key: String,
    /// Absolute byte offset of the object within the bundle (8-aligned).
    pub offset: u64,
    /// Object length in bytes (at-rest bytes; padding excluded).
    pub length: u64,
}

/// The bundle header JSON: the dataset's `manifest.json` VERBATIM (so unpack
/// reproduces it byte-identically) plus the object table.
#[derive(Serialize, Deserialize)]
pub(super) struct BundleHeader {
    pub(super) manifest: Box<serde_json::value::RawValue>,
    pub(super) objects: Vec<BundleObject>,
}

/// Summary of a bundle pack/unpack for CLI reporting.
#[derive(Debug, Clone, Copy)]
pub struct BundleSummary {
    /// Objects carried (directory + packs + any future manifest-listed
    /// kinds); `manifest.json` rides in the header and is not counted.
    pub objects: usize,
    /// Pack: the final bundle file size. Unpack: total bytes written
    /// (objects + `manifest.json`).
    pub bytes: u64,
}

/// Round `n` up to the next 8-byte boundary (bundle object alignment —
/// mirrors the layer-frame alignment rule).
fn align8_u64(n: u64) -> u64 {
    n.div_ceil(8) * 8
}

/// Reject bundle/manifest object keys that could escape the dataset root
/// when joined to a filesystem path — spec §11's path-traversal guard,
/// applied to the bundle header exactly as readers apply it to the manifest.
fn validate_bundle_key(key: &str) -> Result<()> {
    let bad = key.is_empty()
        || key.starts_with('/')
        || key.contains('\\')
        || key.contains(':')
        || key
            .split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == "..");
    if bad {
        return Err(Error::InvalidArchive(format!(
            "unsafe bundle object key {key:?} (keys must be relative paths \
             with no empty/'.'/'..' segments)"
        )));
    }
    Ok(())
}

/// If `key` names a content-addressed object (`<dir>/<32-lower-hex>.<ext>`),
/// return the hex stem so callers can verify blake3(bytes) against it.
/// Unknown/future key shapes return `None` and skip the hash check.
fn content_address_stem(key: &str) -> Option<&str> {
    let name = key.rsplit('/').next()?;
    let stem = name.split('.').next()?;
    (stem.len() == 32 && stem.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .then_some(stem)
}

/// One manifest-referenced object in canonical bundle order.
struct ManifestObjectKey {
    key: String,
    /// Manifest-declared at-rest length, when the manifest carries one.
    declared_length: Option<u64>,
}

/// Enumerate a packed manifest's object keys in the **canonical bundle
/// order**: the directory first, then packs in `pack_id` order, then any
/// future manifest object tables after the current v3 `schemas` array in
/// listed order. Operates on the raw JSON value so bundling stays
/// object-agnostic across manifest revisions. Keys are traversal-validated
/// and must be unique.
fn manifest_object_keys(manifest: &serde_json::Value) -> Result<Vec<ManifestObjectKey>> {
    let obj = manifest
        .as_object()
        .ok_or_else(|| Error::InvalidArchive("manifest is not a JSON object".into()))?;
    if obj.get("format").and_then(|v| v.as_str()) != Some(PACKED_FORMAT) {
        return Err(Error::InvalidArchive(format!(
            "not a packed manifest: format={:?} (expected {PACKED_FORMAT:?})",
            obj.get("format")
        )));
    }
    let entry = |v: &serde_json::Value, what: &str| -> Result<ManifestObjectKey> {
        let o = v
            .as_object()
            .ok_or_else(|| Error::InvalidArchive(format!("manifest {what} is not an object")))?;
        let key = o
            .get("key")
            .and_then(|k| k.as_str())
            .ok_or_else(|| Error::InvalidArchive(format!("manifest {what} has no string `key`")))?;
        Ok(ManifestObjectKey {
            key: key.to_string(),
            declared_length: o.get("length").and_then(|l| l.as_u64()),
        })
    };

    let mut out = Vec::new();
    out.push(entry(
        obj.get("directory")
            .ok_or_else(|| Error::InvalidArchive("manifest has no `directory`".into()))?,
        "directory",
    )?);
    let packs = obj
        .get("packs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::InvalidArchive("manifest has no `packs` array".into()))?;
    for (i, p) in packs.iter().enumerate() {
        out.push(entry(p, &format!("packs[{i}]"))?);
    }
    // formatVersion 3 embeds its schema templates INSIDE the manifest
    // (`schemas: [{hash, data}]` — base64, no object keys), so they ride the
    // header's verbatim manifest and the bundle carries no schema objects.
    // (An earlier draft planned external `schemas/<hash>.sttt` objects; the
    // frozen v2 design dissolved that object class.)

    let mut seen = std::collections::HashSet::with_capacity(out.len());
    for k in &out {
        validate_bundle_key(&k.key)?;
        if !seen.insert(k.key.as_str()) {
            return Err(Error::InvalidArchive(format!(
                "manifest lists object key {:?} twice",
                k.key
            )));
        }
    }
    Ok(out)
}

/// Parse and validate a bundle's fixed prelude + JSON header. Returns the
/// header and the byte offset where the header region ends (object windows
/// must start at or after it). Errors — never panics — on truncation, bad
/// magic, unknown version, nonzero reserved bytes, or malformed header JSON.
pub(super) fn parse_bundle_header(bytes: &[u8]) -> Result<(BundleHeader, u64)> {
    if bytes.len() < BUNDLE_PRELUDE_LEN {
        return Err(Error::InvalidArchive(format!(
            "not an STT bundle: {} bytes is shorter than the {BUNDLE_PRELUDE_LEN}-byte prelude",
            bytes.len()
        )));
    }
    if bytes[0..4] != BUNDLE_MAGIC {
        return Err(Error::InvalidArchive(
            "not an STT bundle (bad magic; expected \"STTB\")".into(),
        ));
    }
    if bytes[4] != BUNDLE_VERSION {
        return Err(Error::InvalidArchive(format!(
            "unsupported bundle version {} (this reader supports {BUNDLE_VERSION})",
            bytes[4]
        )));
    }
    if bytes[5..8] != [0, 0, 0] {
        return Err(Error::InvalidArchive(
            "malformed bundle: reserved prelude bytes must be zero".into(),
        ));
    }
    let header_len = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as u64;
    let header_end = BUNDLE_PRELUDE_LEN as u64 + header_len;
    if header_end > bytes.len() as u64 {
        return Err(Error::InvalidArchive(format!(
            "truncated bundle: header claims {header_len} bytes but only {} remain",
            bytes.len() - BUNDLE_PRELUDE_LEN
        )));
    }
    let header: BundleHeader =
        serde_json::from_slice(&bytes[BUNDLE_PRELUDE_LEN..header_end as usize])
            .map_err(|e| Error::InvalidArchive(format!("bundle header JSON decode failed: {e}")))?;
    Ok((header, header_end))
}

/// Pack an exploded packed dataset into a single `.sttb` bundle file.
///
/// The manifest JSON is embedded **verbatim** in the header (so
/// [`unpack_bundle`] reproduces `manifest.json` byte-identically) and the
/// objects are laid out back-to-back at 8-byte-aligned offsets in canonical
/// manifest order — deterministic: the same dataset packs to byte-identical
/// bundle bytes. Every content-addressed object is re-hashed on the way in;
/// a mismatch aborts the pack (never emit a bundle that cannot verify).
pub fn write_bundle<P: AsRef<Path>, Q: AsRef<Path>>(
    manifest_path: P,
    out_path: Q,
) -> Result<BundleSummary> {
    let manifest_path = manifest_path.as_ref();
    let out_path = out_path.as_ref();
    let root = manifest_path
        .parent()
        .ok_or_else(|| Error::InvalidArchive("manifest path has no parent dir".into()))?;

    let manifest_raw = fs::read(manifest_path)?;
    let manifest_text = std::str::from_utf8(&manifest_raw)
        .map_err(|_| Error::InvalidArchive("manifest.json is not UTF-8".into()))?
        .trim()
        .to_string();
    let manifest_value: serde_json::Value = serde_json::from_str(&manifest_text)
        .map_err(|e| Error::InvalidArchive(format!("manifest JSON decode failed: {e}")))?;
    let typed_manifest = Manifest::from_json_bytes(manifest_text.as_bytes())?;
    manifest_open_checks(&typed_manifest)?;
    let keys = manifest_object_keys(&manifest_value)?;
    let manifest_rv = serde_json::value::RawValue::from_string(manifest_text)
        .map_err(|e| Error::InvalidArchive(format!("manifest is not a JSON value: {e}")))?;

    // Plan the layout from on-disk lengths (verified again at copy time, and
    // against the manifest-declared lengths where the manifest carries one).
    let mut lengths: Vec<u64> = Vec::with_capacity(keys.len());
    for k in &keys {
        let len = fs::metadata(root.join(&k.key))
            .map_err(|e| Error::InvalidArchive(format!("{}: cannot stat object ({e})", k.key)))?
            .len();
        if let Some(declared) = k.declared_length {
            if len != declared {
                return Err(Error::InvalidArchive(format!(
                    "{}: on-disk length {len} != manifest-declared {declared}",
                    k.key
                )));
            }
        }
        lengths.push(len);
    }
    let mut rel_offsets: Vec<u64> = Vec::with_capacity(lengths.len());
    let mut data_len = 0u64;
    for len in &lengths {
        rel_offsets.push(data_len);
        data_len = align8_u64(data_len + len);
    }

    // Fixed point on the header length: absolute offsets appear as decimal
    // digits inside the header JSON, so the data start depends on the header
    // length and vice versa. `data_start` only ever grows across iterations
    // (offsets grow → digits grow → header grows), so this converges —
    // typically on the second pass.
    let mut data_start = align8_u64(BUNDLE_PRELUDE_LEN as u64);
    let header_bytes = loop {
        let objects: Vec<BundleObject> = keys
            .iter()
            .zip(&rel_offsets)
            .zip(&lengths)
            .map(|((k, rel), len)| BundleObject {
                key: k.key.clone(),
                offset: data_start + rel,
                length: *len,
            })
            .collect();
        let header = BundleHeader {
            manifest: manifest_rv.clone(),
            objects,
        };
        let bytes = serde_json::to_vec(&header)
            .map_err(|e| Error::Other(format!("bundle header encode failed: {e}")))?;
        if bytes.len() as u64 > u32::MAX as u64 {
            return Err(Error::InvalidArchive(format!(
                "bundle header is {} bytes, exceeding the u32 header_len cap",
                bytes.len()
            )));
        }
        let next = align8_u64(BUNDLE_PRELUDE_LEN as u64 + bytes.len() as u64);
        if next == data_start {
            break bytes;
        }
        data_start = next;
    };

    // Stream out via a same-directory temp file + atomic rename (the
    // content-addressed-object write discipline, applied to the bundle).
    let out_dir = match out_path.parent() {
        Some(p) if p != Path::new("") => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    fs::create_dir_all(&out_dir)?;
    let tmp = out_dir.join(format!(".tmp-sttb-{}", std::process::id()));
    let total = {
        use std::io::BufWriter;
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        let mut w = BufWriter::new(file);
        const ZEROS: [u8; 8] = [0u8; 8];

        w.write_all(&BUNDLE_MAGIC)?;
        w.write_all(&[BUNDLE_VERSION, 0, 0, 0])?;
        w.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        w.write_all(&header_bytes)?;
        let mut pos = BUNDLE_PRELUDE_LEN as u64 + header_bytes.len() as u64;
        w.write_all(&ZEROS[..(data_start - pos) as usize])?;
        pos = data_start;

        for ((k, rel), len) in keys.iter().zip(&rel_offsets).zip(&lengths) {
            debug_assert_eq!(pos, data_start + rel);
            let bytes = fs::read(root.join(&k.key))?;
            if bytes.len() as u64 != *len {
                return Err(Error::InvalidArchive(format!(
                    "{}: object changed size during bundling ({} != planned {len})",
                    k.key,
                    bytes.len()
                )));
            }
            // Content-addressed keys must hash to their own name — never
            // emit a bundle that cannot verify.
            if let Some(stem) = content_address_stem(&k.key) {
                let hex = blake3_128_hex(&bytes);
                if hex != stem {
                    return Err(Error::InvalidArchive(format!(
                        "{}: content-address mismatch (bytes hash to {hex})",
                        k.key
                    )));
                }
            }
            w.write_all(&bytes)?;
            pos += *len;
            let padded = align8_u64(pos);
            w.write_all(&ZEROS[..(padded - pos) as usize])?;
            pos = padded;
        }
        w.flush()?;
        pos
    };
    fs::rename(&tmp, out_path)?;

    Ok(BundleSummary {
        objects: keys.len(),
        bytes: total,
    })
}

/// Explode a `.sttb` bundle back into a packed dataset directory:
/// `manifest.json` (the verbatim header bytes) plus every object at its key
/// path. Keys are traversal-validated and windows bounds-checked **before**
/// anything is written. Objects are content-addressed, so callers can (and
/// `stt-bundle unpack` does) run [`super::verify_packed_objects`] on the result to
/// prove the round-trip byte-identical.
pub fn unpack_bundle<P: AsRef<Path>, Q: AsRef<Path>>(
    bundle_path: P,
    out_dir: Q,
) -> Result<BundleSummary> {
    let out_dir = out_dir.as_ref();
    let file = File::open(bundle_path.as_ref())?;
    // SAFETY: read-only mapping of a file we never write through, held only
    // for the duration of this call.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| Error::Other(format!("mmap failed: {e}")))?;
    let (header, header_end) = parse_bundle_header(&mmap)?;

    // Validate everything before the first write.
    let mut seen = std::collections::HashSet::with_capacity(header.objects.len());
    for o in &header.objects {
        validate_bundle_key(&o.key)?;
        if !seen.insert(o.key.as_str()) {
            return Err(Error::InvalidArchive(format!(
                "bundle header lists object key {:?} twice",
                o.key
            )));
        }
        let end = o.offset.checked_add(o.length).ok_or_else(|| {
            Error::InvalidArchive(format!(
                "bundle object {:?}: offset+length overflows",
                o.key
            ))
        })?;
        if o.offset < header_end || end > mmap.len() as u64 {
            return Err(Error::InvalidArchive(format!(
                "bundle object {:?} window {}..{end} outside the data region ({header_end}..{})",
                o.key,
                o.offset,
                mmap.len()
            )));
        }
    }
    // Every manifest-referenced object must be present — fail before writing
    // a dataset that could never verify.
    let manifest_value: serde_json::Value = serde_json::from_str(header.manifest.get())
        .map_err(|e| Error::InvalidArchive(format!("bundle manifest JSON decode failed: {e}")))?;
    let typed_manifest = Manifest::from_json_bytes(header.manifest.get().as_bytes())?;
    manifest_open_checks(&typed_manifest)?;
    for k in manifest_object_keys(&manifest_value)? {
        if !seen.contains(k.key.as_str()) {
            return Err(Error::InvalidArchive(format!(
                "bundle header carries no object for manifest key {:?}",
                k.key
            )));
        }
    }

    fs::create_dir_all(out_dir)?;
    let mut bytes_written = 0u64;
    for o in &header.objects {
        let path = out_dir.join(&o.key);
        let parent = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| out_dir.to_path_buf());
        fs::create_dir_all(&parent)?;
        write_atomic(
            &parent,
            &path,
            &mmap[o.offset as usize..(o.offset + o.length) as usize],
        )?;
        bytes_written += o.length;
    }
    // Manifest last: its presence marks the exploded dataset complete.
    let manifest_bytes = header.manifest.get().as_bytes();
    write_atomic(out_dir, &out_dir.join("manifest.json"), manifest_bytes)?;
    bytes_written += manifest_bytes.len() as u64;

    Ok(BundleSummary {
        objects: header.objects.len(),
        bytes: bytes_written,
    })
}

/// Read just the (typed) manifest out of a `.sttb` bundle header.
pub fn read_bundle_manifest<P: AsRef<Path>>(bundle_path: P) -> Result<Manifest> {
    let file = File::open(bundle_path.as_ref())?;
    // SAFETY: read-only mapping, held only for the duration of this call.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| Error::Other(format!("mmap failed: {e}")))?;
    let (header, _) = parse_bundle_header(&mmap)?;
    Manifest::from_json_bytes(header.manifest.get().as_bytes())
}

/// Bundle analog of [`super::verify_packed_objects`]: verify a `.sttb`'s contents
/// against its embedded manifest with no trusted side-channel — each
/// content-addressed object's bytes must blake3-hash to its key, in-bundle
/// lengths must match the manifest-declared lengths, the directory must
/// decode (paged structure included), and no `pack_id` may fall outside the
/// pack table.
///
/// Returns the list of violations (empty ⇒ clean). Returns `Err` only when
/// the bundle container itself cannot be read or parsed (mirroring the
/// manifest-unreadable case of the exploded verifier).
pub fn verify_bundle_objects<P: AsRef<Path>>(bundle_path: P) -> Result<Vec<String>> {
    let file = File::open(bundle_path.as_ref())?;
    // SAFETY: read-only mapping, held only for the duration of this call.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| Error::Other(format!("mmap failed: {e}")))?;
    let (header, header_end) = parse_bundle_header(&mmap)?;
    let manifest = Manifest::from_json_bytes(header.manifest.get().as_bytes())?;

    let mut issues = Vec::new();
    if let Err(error) = manifest_open_checks(&manifest) {
        issues.push(error.to_string());
        return Ok(issues);
    }

    if manifest.format != PACKED_FORMAT {
        issues.push(format!(
            "manifest format is {:?}, expected {PACKED_FORMAT:?}",
            manifest.format
        ));
    }
    if manifest.format_version != PACKED_FORMAT_VERSION {
        issues.push(format!(
            "manifest formatVersion is {}, expected {PACKED_FORMAT_VERSION}",
            manifest.format_version
        ));
    }
    if manifest.directory.directory_version != crate::directory::DIRECTORY_VERSION {
        issues.push(format!(
            "directoryVersion is {}, expected {}",
            manifest.directory.directory_version,
            crate::directory::DIRECTORY_VERSION
        ));
    }
    verify_manifest_schemas(&manifest, &mut issues);

    // Object table: key/window sanity, then key → slice for the checks below.
    let mut by_key: HashMap<&str, &[u8]> = HashMap::with_capacity(header.objects.len());
    for o in &header.objects {
        if let Err(e) = validate_bundle_key(&o.key) {
            issues.push(e.to_string());
            continue;
        }
        let end = match o.offset.checked_add(o.length) {
            Some(end) if o.offset >= header_end && end <= mmap.len() as u64 => end,
            _ => {
                issues.push(format!(
                    "bundle object {:?} window {}+{} outside the data region ({header_end}..{})",
                    o.key,
                    o.offset,
                    o.length,
                    mmap.len()
                ));
                continue;
            }
        };
        if by_key
            .insert(o.key.as_str(), &mmap[o.offset as usize..end as usize])
            .is_some()
        {
            issues.push(format!("bundle header lists object key {:?} twice", o.key));
        }
    }

    // Each content-addressed object: bytes must hash to the key's name and
    // match the manifest-declared length — exactly the exploded-directory
    // checks, sourced from bundle windows instead of files.
    fn check_object(
        by_key: &HashMap<&str, &[u8]>,
        key: &str,
        declared_len: u64,
        prefix: &str,
        ext: &str,
        issues: &mut Vec<String>,
    ) {
        match by_key.get(key) {
            Some(bytes) => {
                if bytes.len() as u64 != declared_len {
                    issues.push(format!(
                        "{key}: in-bundle length {} != manifest-declared {declared_len}",
                        bytes.len()
                    ));
                }
                let expected = format!("{prefix}/{}.{ext}", blake3_128_hex(bytes));
                if key != expected {
                    issues.push(format!(
                        "{key}: content-address mismatch (bytes hash to {expected})"
                    ));
                }
            }
            None => issues.push(format!("{key}: bundle header carries no such object")),
        }
    }

    check_object(
        &by_key,
        &manifest.directory.key,
        manifest.directory.length,
        "index",
        "sttd",
        &mut issues,
    );
    for p in &manifest.packs {
        check_object(&by_key, &p.key, p.length, "packs", "sttp", &mut issues);
    }

    // v2 objects must self-identify: validate each window's magic prelude.
    if manifest.format_version == PACKED_FORMAT_VERSION {
        for (key, kind) in std::iter::once((&manifest.directory.key, DIRECTORY_MAGIC))
            .chain(manifest.packs.iter().map(|p| (&p.key, PACK_MAGIC)))
        {
            if let Some(bytes) = by_key.get(key.as_str()) {
                if let Err(e) = strip_object_magic(bytes, kind, key) {
                    issues.push(e.to_string());
                }
            }
        }
    }

    // Directory must decode (through its v2 magic prelude, at-rest encoding +
    // container layout) and reference only packs the manifest lists; a paged
    // directory additionally passes the structural covering/order checks.
    if let Some(dir_bytes) = by_key.get(manifest.directory.key.as_str()) {
        match directory_codec_bytes(dir_bytes, manifest.format_version) {
            Ok(codec) => {
                match decode_directory_entries(
                    codec,
                    &manifest.directory,
                    manifest.metadata.tile_count,
                ) {
                    Ok(entries) => {
                        if let Some(max_pid) = entries.iter().map(|e| e.pack_id).max() {
                            if max_pid as usize >= manifest.packs.len() {
                                issues.push(format!(
                                    "directory references pack_id {max_pid} but the manifest lists only {} pack(s)",
                                    manifest.packs.len()
                                ));
                            }
                        }
                        verify_v2_frame_template_refs(
                            &manifest,
                            &entries,
                            |pid| {
                                manifest
                                    .packs
                                    .get(pid)
                                    .and_then(|p| by_key.get(p.key.as_str()))
                                    .map(|b| b.to_vec())
                            },
                            &mut issues,
                        );
                    }
                    Err(e) => issues.push(format!("directory failed to decode: {e}")),
                }
                if manifest.directory.is_paged() {
                    match manifest.directory.root_length {
                        Some(rl) => {
                            let zstd = manifest.directory.encoding.as_deref()
                                == Some(DIRECTORY_ENCODING_ZSTD);
                            match crate::directory_page::verify_paged_structure(codec, rl, zstd) {
                                Ok(mut more) => issues.append(&mut more),
                                Err(e) => issues.push(format!("paged structure check failed: {e}")),
                            }
                        }
                        None => issues.push("paged directory: manifest missing rootLength".into()),
                    }
                }
            }
            // The magic check above already reported the malformed prelude.
            Err(_) => {}
        }
    }

    Ok(issues)
}
