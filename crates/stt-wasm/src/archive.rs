//! The byte-slice reader: a packed dataset opened from **bytes the caller
//! supplies**, with no filesystem and no network of its own.
//!
//! `stt_core::PackedReader` is the same format read the other way round — it
//! takes a `manifest.json` path and mmaps the objects itself. That shape has
//! no wasm32 equivalent: there are no paths in a browser, and a Python or JS
//! host already owns a better fetch stack than this crate could bundle. So the
//! I/O is inverted. This reader answers *which bytes it needs next*
//! ([`Archive::directory_key`], [`TileRef::pack_id`] + `offset`/`length`) and
//! the host hands them back. Everything between those two points — manifest
//! checks, the v5 directory codec, CRC32C, zstd, the layer frame, GeoArrow —
//! is `stt_core`'s, unchanged.
//!
//! This module is plain Rust with no `wasm_bindgen` in it, so the decode path
//! is exercised by an ordinary `cargo test` on the host (`tests/`), including
//! the parity test that pins it against `PackedReader` on the same archive.

use arrow::ipc::writer::StreamWriter;
use stt_core::arrow_tile::{
    decode_tile_with_templates, is_frame_v2, DecodedLayer, TemplateRegistry,
};
use stt_core::compression;
use stt_core::directory::{decode_directory, TileEntry};
use stt_core::directory_page::decode_paged_directory;
use stt_core::error::{Error, Result};
use stt_core::pack::{
    directory_codec_bytes, Manifest, DIRECTORY_ENCODING_ZSTD, KNOWN_CAPABILITIES, OBJECT_MAGIC_LEN,
    PACKED_FORMAT, PACKED_FORMAT_VERSION, PACK_MAGIC,
};
use stt_core::types::Compression;

/// One decoded tile layer, re-emitted as a **self-contained Arrow IPC stream**
/// (schema message + one record batch + EOS).
///
/// The on-disk v2 layer frame stores the schema once per dataset, in
/// `manifest.schemas`, and each tile references it by hash — excellent on the
/// wire, useless to a consumer that only speaks Arrow. Re-framing here is what
/// makes the output loadable by `pyarrow.ipc.open_stream` / `apache-arrow-js`
/// with no knowledge of the STT format at all. That is the entire point of the
/// crate, so the re-encode is deliberate, not an oversight.
#[derive(Debug, Clone)]
pub struct LayerIpc {
    /// Layer name from the tile's layer frame.
    pub name: String,
    /// Arrow IPC **stream** bytes for this layer's single record batch.
    pub ipc: Vec<u8>,
}

/// What the host must fetch to decode one tile: a byte range inside one pack
/// object. Returned by [`Archive::tile`].
#[derive(Debug, Clone, Copy)]
pub struct TileRef<'a> {
    entry: &'a TileEntry,
}

impl TileRef<'_> {
    /// Zoom level.
    pub fn zoom(&self) -> u8 {
        self.entry.zoom
    }
    /// Tile X.
    pub fn x(&self) -> u32 {
        self.entry.x
    }
    /// Tile Y.
    pub fn y(&self) -> u32 {
        self.entry.y
    }
    /// Inclusive temporal start (Unix ms).
    pub fn time_start(&self) -> i64 {
        self.entry.time_start
    }
    /// Inclusive temporal end (Unix ms).
    pub fn time_end(&self) -> i64 {
        self.entry.time_end
    }
    /// Total feature count across the tile's layers.
    pub fn feature_count(&self) -> u32 {
        self.entry.feature_count
    }
    /// Index into the manifest pack table — resolve with [`Archive::pack_key`].
    pub fn pack_id(&self) -> u32 {
        self.entry.pack_id
    }
    /// Byte offset of the compressed blob inside its pack **object**.
    pub fn offset(&self) -> u64 {
        self.entry.offset
    }
    /// Compressed blob length in bytes — the length of the range to fetch.
    pub fn length(&self) -> u32 {
        self.entry.length
    }
    /// Uncompressed payload length in bytes (a decode-side sanity bound).
    pub fn uncompressed_size(&self) -> u32 {
        self.entry.uncompressed_size
    }
    /// The underlying directory entry.
    pub fn entry(&self) -> &TileEntry {
        self.entry
    }
}

/// A packed dataset read from caller-supplied bytes.
///
/// Lifecycle: [`open`](Self::open) with the manifest bytes, then
/// [`load_directory`](Self::load_directory) with the object named by
/// [`directory_key`](Self::directory_key), then per tile
/// [`decode_tile`](Self::decode_tile) with the blob range named by
/// [`tile`](Self::tile).
pub struct Archive {
    manifest: Manifest,
    compression: Compression,
    /// v2 only: the hash-validated template registry from `manifest.schemas`.
    templates: TemplateRegistry,
    entries: Vec<TileEntry>,
    directory_loaded: bool,
}

impl Archive {
    /// Open a dataset from its `manifest.json` bytes.
    ///
    /// Runs the same open-time refusals as `PackedReader::open`: format tag,
    /// `formatVersion`, the required-to-understand capability registry, the
    /// compression codec, and (v2) the blake3-128 validation of every embedded
    /// schema template. Those checks are the reason this is not a thin
    /// `serde_json` call — a capability this reader does not implement
    /// RE-TYPES existing columns, so skipping the check would not fail later,
    /// it would silently misdecode every tile (packed spec §3.1).
    pub fn open(manifest_bytes: &[u8]) -> Result<Self> {
        let manifest = Manifest::from_json_bytes(manifest_bytes)?;

        if manifest.format != PACKED_FORMAT {
            return Err(Error::InvalidArchive(format!(
                "not a packed manifest: format={:?} (expected {PACKED_FORMAT:?})",
                manifest.format
            )));
        }
        if manifest.format_version != PACKED_FORMAT_VERSION {
            return Err(Error::InvalidArchive(format!(
                "unsupported packed formatVersion {} (this reader supports {PACKED_FORMAT_VERSION})",
                manifest.format_version
            )));
        }
        let unknown: Vec<&str> = manifest
            .capabilities
            .iter()
            .map(String::as_str)
            .filter(|c| !KNOWN_CAPABILITIES.contains(c))
            .collect();
        if !unknown.is_empty() {
            return Err(Error::InvalidArchive(format!(
                "dataset requires capabilities this reader does not implement: {} \
                 (implemented: {})",
                unknown.join(", "),
                KNOWN_CAPABILITIES.join(", ")
            )));
        }
        let compression = match manifest.compression.as_str() {
            "zstd" => Compression::Zstd,
            "none" => Compression::None,
            other => {
                return Err(Error::InvalidArchive(format!(
                    "unknown packed compression {other:?}"
                )))
            }
        };

        let templates = build_template_registry(&manifest)?;

        Ok(Self {
            manifest,
            compression,
            templates,
            entries: Vec::new(),
            directory_loaded: false,
        })
    }

    /// The manifest, verbatim.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The manifest's authoritative `formatVersion` (1 | 2).
    pub fn format_version(&self) -> u32 {
        self.manifest.format_version
    }

    /// Object key of the directory, relative to the dataset root — the second
    /// thing a cold reader fetches.
    pub fn directory_key(&self) -> &str {
        &self.manifest.directory.key
    }

    /// At-rest byte length of the directory object, per the manifest.
    pub fn directory_length(&self) -> u64 {
        self.manifest.directory.length
    }

    /// Whether the directory is the paged container. This reader loads a paged
    /// directory whole; see [`load_directory`](Self::load_directory).
    pub fn directory_is_paged(&self) -> bool {
        self.manifest.directory.is_paged()
    }

    /// Decode the directory object into the tile list, returning the entry
    /// count. `bytes` is the **whole** object named by
    /// [`directory_key`](Self::directory_key).
    ///
    /// Paged directories are loaded whole too (root + every leaf). The paging
    /// win is a cold *HTTP* reader fetching only the leaves its viewport
    /// touches; exposing that here needs a leaf-at-a-time API this crate does
    /// not have yet, and loading whole is always correct.
    pub fn load_directory(&mut self, bytes: &[u8]) -> Result<usize> {
        // A short read here is the commonest host bug (a truncated range
        // response, or a CDN error page served with 200). Without this guard
        // it surfaces as an unintelligible codec error deep inside the v5
        // varint walk, pointing at the format instead of at the fetch.
        let declared = self.manifest.directory.length;
        if bytes.len() as u64 != declared {
            return Err(Error::InvalidArchive(format!(
                "directory object {:?} is {} bytes, manifest declared {declared}",
                self.manifest.directory.key,
                bytes.len()
            )));
        }

        let codec_bytes = directory_codec_bytes(bytes, self.manifest.format_version)?;
        let dref = &self.manifest.directory;
        let entries = if dref.is_paged() {
            let root_length = dref.root_length.ok_or_else(|| {
                Error::InvalidArchive("paged directory: manifest missing rootLength".into())
            })?;
            let zstd = dref.encoding.as_deref() == Some(DIRECTORY_ENCODING_ZSTD);
            decode_paged_directory(codec_bytes, root_length, zstd)?
        } else {
            let raw = match dref.encoding.as_deref() {
                None => codec_bytes.to_vec(),
                Some(DIRECTORY_ENCODING_ZSTD) => {
                    compression::decompress_zstd_with_dict(codec_bytes, None)?
                }
                Some(other) => {
                    return Err(Error::InvalidArchive(format!(
                        "unknown directory encoding {other:?} (this reader supports \
                         absent or \"zstd\")"
                    )))
                }
            };
            decode_directory(&raw)?
        };

        self.entries = entries;
        self.directory_loaded = true;
        Ok(self.entries.len())
    }

    /// Number of tiles. `0` until [`load_directory`](Self::load_directory).
    pub fn tile_count(&self) -> usize {
        self.entries.len()
    }

    /// One tile's addressing record, by directory index.
    pub fn tile(&self, index: usize) -> Result<TileRef<'_>> {
        let entry = self.entries.get(index).ok_or_else(|| {
            Error::Other(format!(
                "tile index {index} out of range ({} tiles{})",
                self.entries.len(),
                if self.directory_loaded {
                    ""
                } else {
                    "; load_directory has not been called"
                }
            ))
        })?;
        Ok(TileRef { entry })
    }

    /// Every directory entry, in directory order (zoom, then Hilbert index).
    pub fn entries(&self) -> &[TileEntry] {
        &self.entries
    }

    /// Number of pack objects.
    pub fn pack_count(&self) -> usize {
        self.manifest.packs.len()
    }

    /// Object key of pack `pack_id`, relative to the dataset root.
    pub fn pack_key(&self, pack_id: usize) -> Result<&str> {
        self.manifest
            .packs
            .get(pack_id)
            .map(|p| p.key.as_str())
            .ok_or_else(|| {
                Error::InvalidArchive(format!(
                    "pack_id {pack_id} out of range ({} packs)",
                    self.manifest.packs.len()
                ))
            })
    }

    /// Declared byte length of pack `pack_id`.
    pub fn pack_length(&self, pack_id: usize) -> Result<u64> {
        self.manifest
            .packs
            .get(pack_id)
            .map(|p| p.length)
            .ok_or_else(|| {
                Error::InvalidArchive(format!(
                    "pack_id {pack_id} out of range ({} packs)",
                    self.manifest.packs.len()
                ))
            })
    }

    /// Decode tile `index` from **exactly** its compressed blob: the
    /// `[offset, offset + length)` range of pack [`TileRef::pack_id`].
    ///
    /// This is the HTTP shape — one `Range:` request per tile.
    pub fn decode_tile(&self, index: usize, blob: &[u8]) -> Result<Vec<LayerIpc>> {
        let entry = self.tile(index)?.entry().clone();
        // Same reasoning as the directory length guard: a truncated range
        // response would otherwise fail the CRC and read as data corruption,
        // sending the reader after the archive instead of after the fetch.
        if blob.len() != entry.length as usize {
            return Err(Error::InvalidArchive(format!(
                "tile {:?}: got {} blob bytes, directory declared {}",
                entry.tile_id(),
                blob.len(),
                entry.length
            )));
        }
        let payload = self.decode_blob(&entry, blob)?;
        self.decode_payload(&payload)
    }

    /// Decode tile `index` from the **whole** pack object it lives in — the
    /// local-file shape (a Python caller with the dataset on disk, or a JS
    /// caller that already cached the pack).
    ///
    /// Under formatVersion 2 blob offsets are object-absolute, so the slice
    /// math is version-independent.
    pub fn decode_tile_in_pack(&self, index: usize, pack_bytes: &[u8]) -> Result<Vec<LayerIpc>> {
        let entry = self.tile(index)?.entry().clone();
        let declared = self.pack_length(entry.pack_id as usize)?;
        if pack_bytes.len() as u64 != declared {
            return Err(Error::InvalidArchive(format!(
                "pack {:?} is {} bytes, manifest declared {declared}",
                self.pack_key(entry.pack_id as usize)?,
                pack_bytes.len()
            )));
        }
        // A pack object self-identifies (packed spec §9.2). Checking the tag
        // turns the likeliest host mistake — passing the DIRECTORY object, or a
        // 404 body — into a named error rather than a CRC failure that reads
        // like a corrupt archive. (stt-core's private `strip_object_magic` also
        // pins the version and reserved bytes; nothing below depends on
        // stripping them, because blob offsets are object-absolute.)
        if pack_bytes.len() < OBJECT_MAGIC_LEN || pack_bytes[..4] != PACK_MAGIC {
            return Err(Error::InvalidArchive(format!(
                "pack {:?}: missing \"STTP\" magic (pack objects start \
                 with an 8-byte magic prelude)",
                self.pack_key(entry.pack_id as usize)?
            )));
        }

        // u64 first: a corrupt offset near u64::MAX would wrap a `usize` add
        // and slip past the bound, then panic in the slice.
        let end = entry
            .offset
            .checked_add(entry.length as u64)
            .ok_or_else(|| {
                Error::InvalidArchive(format!(
                    "tile {:?}: blob offset+length overflows",
                    entry.tile_id()
                ))
            })?;
        if end > pack_bytes.len() as u64 {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} blob range {}..{end} exceeds pack size {}",
                entry.tile_id(),
                entry.offset,
                pack_bytes.len()
            )));
        }
        let payload = self.decode_blob(&entry, &pack_bytes[entry.offset as usize..end as usize])?;
        self.decode_payload(&payload)
    }

    /// CRC32C-check, decompress and size-check one tile's compressed blob.
    /// Mirrors `PackedReader::decode_blob`.
    fn decode_blob(&self, entry: &TileEntry, compressed: &[u8]) -> Result<Vec<u8>> {
        if crc32c::crc32c(compressed) != entry.crc32c {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} failed integrity check (corrupt pack)",
                entry.tile_id()
            )));
        }
        let payload = match self.compression {
            Compression::Zstd => compression::decompress_zstd_with_dict(compressed, None)?,
            Compression::None => compression::decompress(compressed, Compression::None)?,
        };
        if payload.len() != entry.uncompressed_size as usize {
            return Err(Error::InvalidArchive(format!(
                "tile {:?} decompressed to {} bytes, expected {}",
                entry.tile_id(),
                payload.len(),
                entry.uncompressed_size
            )));
        }
        Ok(payload)
    }

    /// Decode a tile payload and re-frame each layer as a standalone Arrow IPC
    /// stream. Mirrors `PackedReader::decode_payload`.
    pub fn decode_payload(&self, payload: &[u8]) -> Result<Vec<LayerIpc>> {
        if !is_frame_v2(payload) {
            return Err(Error::InvalidArchive(
                "tile payload is not a layer frame (missing the frame escape)".into(),
            ));
        }
        let layers = decode_tile_with_templates(payload, &self.templates)?;
        layers.iter().map(layer_to_ipc).collect()
    }
}

/// Re-frame one decoded layer as an Arrow IPC stream.
fn layer_to_ipc(layer: &DecodedLayer) -> Result<LayerIpc> {
    let mut ipc = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut ipc, &layer.batch.schema())
            .map_err(|e| Error::Other(format!("Arrow IPC writer init failed: {e}")))?;
        writer
            .write(&layer.batch)
            .map_err(|e| Error::Other(format!("Arrow IPC write failed: {e}")))?;
        writer
            .finish()
            .map_err(|e| Error::Other(format!("Arrow IPC finish failed: {e}")))?;
    }
    Ok(LayerIpc {
        name: layer.name.clone(),
        ipc,
    })
}

/// Decode + hash-validate `manifest.schemas` into the decode-side registry.
///
/// A reproduction of stt-core's private `build_template_registry` — the only
/// logic this crate re-states rather than calls. `open_parity` in
/// `tests/decode_roundtrip.rs` pins it against the real reader on a v2
/// archive, so a drift in the hash rule fails a test rather than a dataset.
fn build_template_registry(manifest: &Manifest) -> Result<TemplateRegistry> {
    use base64::Engine as _;
    let mut registry = TemplateRegistry::new();
    for (i, s) in manifest.schemas.iter().enumerate() {
        let data = base64::engine::general_purpose::STANDARD
            .decode(&s.data)
            .map_err(|e| {
                Error::InvalidArchive(format!(
                    "manifest schemas[{i}] ({}): base64 decode failed: {e}",
                    s.hash
                ))
            })?;
        let actual: String = blake3::hash(&data).as_bytes()[..16]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if actual != s.hash {
            return Err(Error::InvalidArchive(format!(
                "manifest schemas[{i}]: template bytes hash to {actual}, declared {}",
                s.hash
            )));
        }
        registry.insert(data);
    }
    Ok(registry)
}
