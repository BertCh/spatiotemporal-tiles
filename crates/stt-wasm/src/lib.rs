//! **WebAssembly decoder for the SpatioTemporal Tiles packed format.**
//!
//! The format has a Rust reader (`stt_core::PackedReader`) and a TypeScript
//! reader (`@poopdeck.gl/core`), and both were written by the same author
//! against the same intent — so `docs/spec/conformance.md` passing tells you
//! the two agree, not that the spec is implementable by someone else. This
//! crate does not fix that (it is the Rust reader, compiled small), but it is
//! the cheapest way to put the reader where a *third-party* consumer can
//! reach it: a Python notebook, a Go or C++ host, a future GDAL or Martin
//! path. Every one of those already knows how to read Arrow IPC, and none of
//! them is going to port a v6 varint directory.
//!
//! So the whole API is three moves — open a manifest, list tiles, decode a
//! tile blob to Arrow IPC — and it adds no decoding logic of its own: see
//! [`archive`] for the byte-slice reader that does the work, and
//! `docs/guides/wasm.md` for the build recipe, the JS/Python usage, and an
//! honest list of what it does not do yet.
//!
//! ## Why bytes in, bytes out
//!
//! There are no file paths in a browser, and `PackedReader` is built on
//! `Mmap::map`. Rather than bundle a fetch stack (and a JS↔Rust async
//! boundary) into a decoder, this crate never does I/O: it tells the host
//! which object key and which byte range it needs, and the host — `fetch`,
//! `requests`, `fsspec`, a local file — hands the bytes back. That keeps the
//! artifact small, keeps range-request policy (auth, retries, caching,
//! coalescing) where the host already solved it, and makes the whole thing
//! testable without a browser runner.

#![deny(missing_docs)]

pub mod archive;

use wasm_bindgen::prelude::*;

pub use archive::{Archive, LayerIpc, TileRef};

/// Convert a reader error into a JS `Error`.
///
/// Only ever called on a failure path. `JsError` construction goes through
/// wasm-bindgen's `__wbindgen` imports, which are not linked on a host build —
/// keeping the conversion lazy is what lets the happy path of every exported
/// method below run under a plain `cargo test` (see `tests/`). Error-path
/// assertions therefore go through [`Archive`] directly.
fn js_err(e: stt_core::error::Error) -> JsError {
    JsError::new(&e.to_string())
}

/// The crate version, so a host can log which decoder it loaded.
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// A packed STT dataset, decoded from bytes the caller supplies.
///
/// ```text
/// const a = SttArchive.open(await get('manifest.json'));
/// a.loadDirectory(await get(a.directoryKey()));
/// const t = a.tile(0);
/// const blob = await getRange(a.packKey(t.packId), t.offset, t.length);
/// const tile = a.decodeTile(0, blob);
/// tableFromIPC(tile.layerIpc(0));   // apache-arrow
/// ```
#[wasm_bindgen]
pub struct SttArchive {
    inner: Archive,
}

#[wasm_bindgen]
impl SttArchive {
    /// Open a dataset from its `manifest.json` bytes. Throws if the manifest
    /// is not a packed manifest, declares an unsupported `formatVersion` or
    /// compression, requires a capability this decoder does not implement, or
    /// carries a corrupt schema template.
    #[wasm_bindgen(js_name = open)]
    pub fn open(manifest_bytes: &[u8]) -> Result<SttArchive, JsError> {
        Ok(SttArchive {
            inner: Archive::open(manifest_bytes).map_err(js_err)?,
        })
    }

    /// The manifest's packed `formatVersion` (3).
    #[wasm_bindgen(js_name = formatVersion)]
    pub fn format_version(&self) -> u32 {
        self.inner.format_version()
    }

    /// The dataset metadata (name, bbox, time range, zoom levels, …) as JSON —
    /// the same object the manifest carries verbatim.
    #[wasm_bindgen(js_name = metadataJson)]
    pub fn metadata_json(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.inner.manifest().metadata)
            .map_err(|e| JsError::new(&format!("metadata JSON encode failed: {e}")))
    }

    /// Object key of the directory, relative to the dataset root — fetch this
    /// whole object and pass it to [`load_directory`](Self::load_directory).
    #[wasm_bindgen(js_name = directoryKey)]
    pub fn directory_key(&self) -> String {
        self.inner.directory_key().to_string()
    }

    /// At-rest byte length of the directory object, per the manifest.
    #[wasm_bindgen(js_name = directoryLength)]
    pub fn directory_length(&self) -> f64 {
        self.inner.directory_length() as f64
    }

    /// Whether the directory uses the paged container. Informational: this
    /// decoder loads a paged directory whole either way.
    #[wasm_bindgen(js_name = directoryIsPaged)]
    pub fn directory_is_paged(&self) -> bool {
        self.inner.directory_is_paged()
    }

    /// Decode the directory object into the tile list; returns the tile count.
    #[wasm_bindgen(js_name = loadDirectory)]
    pub fn load_directory(&mut self, bytes: &[u8]) -> Result<usize, JsError> {
        self.inner.load_directory(bytes).map_err(js_err)
    }

    /// Number of tiles. `0` until [`load_directory`](Self::load_directory).
    #[wasm_bindgen(js_name = tileCount)]
    pub fn tile_count(&self) -> usize {
        self.inner.tile_count()
    }

    /// One tile's addressing record, by directory index.
    #[wasm_bindgen(js_name = tile)]
    pub fn tile(&self, index: usize) -> Result<TileInfo, JsError> {
        Ok(TileInfo::from(self.inner.tile(index).map_err(js_err)?))
    }

    /// Every tile as one JSON array.
    ///
    /// A dataset can hold hundreds of thousands of tiles, and crossing the
    /// wasm boundary once per tile costs far more than the JSON does — this is
    /// the bulk path for anything that wants to index or filter the directory
    /// host-side. Field names match [`TileInfo`]'s accessors.
    #[wasm_bindgen(js_name = tilesJson)]
    pub fn tiles_json(&self) -> Result<String, JsError> {
        let tiles: Vec<serde_json::Value> = self
            .inner
            .entries()
            .iter()
            .map(|e| {
                serde_json::json!({
                    "zoom": e.zoom,
                    "x": e.x,
                    "y": e.y,
                    "timeStart": e.time_start,
                    "timeEnd": e.time_end,
                    "featureCount": e.feature_count,
                    "packId": e.pack_id,
                    "offset": e.offset,
                    "length": e.length,
                    "uncompressedSize": e.uncompressed_size,
                })
            })
            .collect();
        serde_json::to_string(&tiles)
            .map_err(|e| JsError::new(&format!("tile list JSON encode failed: {e}")))
    }

    /// Number of pack objects.
    #[wasm_bindgen(js_name = packCount)]
    pub fn pack_count(&self) -> usize {
        self.inner.pack_count()
    }

    /// Object key of pack `pack_id`, relative to the dataset root.
    #[wasm_bindgen(js_name = packKey)]
    pub fn pack_key(&self, pack_id: usize) -> Result<String, JsError> {
        Ok(self.inner.pack_key(pack_id).map_err(js_err)?.to_string())
    }

    /// Declared byte length of pack `pack_id`.
    #[wasm_bindgen(js_name = packLength)]
    pub fn pack_length(&self, pack_id: usize) -> Result<f64, JsError> {
        Ok(self.inner.pack_length(pack_id).map_err(js_err)? as f64)
    }

    /// Decode tile `index` from **exactly** its compressed blob — the
    /// `[offset, offset + length)` range of pack `packId`. The HTTP shape: one
    /// `Range:` request per tile.
    #[wasm_bindgen(js_name = decodeTile)]
    pub fn decode_tile(&self, index: usize, blob: &[u8]) -> Result<DecodedTile, JsError> {
        Ok(DecodedTile {
            layers: self.inner.decode_tile(index, blob).map_err(js_err)?,
        })
    }

    /// Decode tile `index` from the **whole** pack object it lives in — the
    /// local-file shape, for a host that already has the pack in memory.
    #[wasm_bindgen(js_name = decodeTileInPack)]
    pub fn decode_tile_in_pack(
        &self,
        index: usize,
        pack_bytes: &[u8],
    ) -> Result<DecodedTile, JsError> {
        Ok(DecodedTile {
            layers: self
                .inner
                .decode_tile_in_pack(index, pack_bytes)
                .map_err(js_err)?,
        })
    }
}

/// Where one tile lives and what it covers.
///
/// Times and offsets are `f64`, not `BigInt`: Unix-ms and pack offsets are
/// exact in a double for any value this format can hold (a pack is ≤ 512 MB,
/// and ms stays exact past the year 285000), while `BigInt` would force every
/// host — `Date`, deck.gl, numpy — to convert on the way in.
#[wasm_bindgen]
pub struct TileInfo {
    zoom: u8,
    x: u32,
    y: u32,
    time_start: f64,
    time_end: f64,
    feature_count: u32,
    pack_id: u32,
    offset: f64,
    length: u32,
    uncompressed_size: u32,
}

impl From<TileRef<'_>> for TileInfo {
    fn from(t: TileRef<'_>) -> Self {
        Self {
            zoom: t.zoom(),
            x: t.x(),
            y: t.y(),
            time_start: t.time_start() as f64,
            time_end: t.time_end() as f64,
            feature_count: t.feature_count(),
            pack_id: t.pack_id(),
            offset: t.offset() as f64,
            length: t.length(),
            uncompressed_size: t.uncompressed_size(),
        }
    }
}

#[wasm_bindgen]
impl TileInfo {
    /// Zoom level.
    #[wasm_bindgen(getter)]
    pub fn zoom(&self) -> u8 {
        self.zoom
    }
    /// Tile X.
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> u32 {
        self.x
    }
    /// Tile Y.
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> u32 {
        self.y
    }
    /// Inclusive temporal start (Unix ms).
    #[wasm_bindgen(getter, js_name = timeStart)]
    pub fn time_start(&self) -> f64 {
        self.time_start
    }
    /// Inclusive temporal end (Unix ms).
    #[wasm_bindgen(getter, js_name = timeEnd)]
    pub fn time_end(&self) -> f64 {
        self.time_end
    }
    /// Total feature count across the tile's layers.
    #[wasm_bindgen(getter, js_name = featureCount)]
    pub fn feature_count(&self) -> u32 {
        self.feature_count
    }
    /// Index into the pack table — resolve with `SttArchive.packKey`.
    #[wasm_bindgen(getter, js_name = packId)]
    pub fn pack_id(&self) -> u32 {
        self.pack_id
    }
    /// Byte offset of the compressed blob inside its pack object.
    #[wasm_bindgen(getter)]
    pub fn offset(&self) -> f64 {
        self.offset
    }
    /// Compressed blob length — the length of the range to fetch.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> u32 {
        self.length
    }
    /// Uncompressed payload length in bytes.
    #[wasm_bindgen(getter, js_name = uncompressedSize)]
    pub fn uncompressed_size(&self) -> u32 {
        self.uncompressed_size
    }
}

/// One decoded tile: its layers, each a standalone Arrow IPC stream.
#[wasm_bindgen]
pub struct DecodedTile {
    layers: Vec<LayerIpc>,
}

#[wasm_bindgen]
impl DecodedTile {
    /// Number of layers in the tile.
    #[wasm_bindgen(js_name = layerCount)]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Layer `index`'s name.
    #[wasm_bindgen(js_name = layerName)]
    pub fn layer_name(&self, index: usize) -> Result<String, JsError> {
        self.layers
            .get(index)
            .map(|l| l.name.clone())
            .ok_or_else(|| JsError::new(&format!("layer index {index} out of range")))
    }

    /// Layer `index`'s Arrow IPC **stream** bytes — feed straight to
    /// `tableFromIPC` (apache-arrow) or `pyarrow.ipc.open_stream`.
    #[wasm_bindgen(js_name = layerIpc)]
    pub fn layer_ipc(&self, index: usize) -> Result<Vec<u8>, JsError> {
        self.layers
            .get(index)
            .map(|l| l.ipc.clone())
            .ok_or_else(|| JsError::new(&format!("layer index {index} out of range")))
    }
}

impl DecodedTile {
    /// The decoded layers, for host-side (non-wasm) callers.
    pub fn layers(&self) -> &[LayerIpc] {
        &self.layers
    }
}
