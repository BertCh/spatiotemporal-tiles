//! Packed-tileset reading for post-build analysis (`stt-optimize inspect`/`diff`).
//!
//! Thin wrapper over [`stt_core::pack::PackedReader`] that also keeps the
//! parsed [`Manifest`] around, so the profiler passes can see container facts
//! the reader alone doesn't expose (pack table, directory layout). Both
//! directory container shapes — single whole-load and paged — decode through
//! the same stt-core path ([`PackedReader::open`] branches on
//! `directory.layout`), so callers here never need to care which one a
//! dataset ships.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use stt_core::arrow_tile::DecodedLayer;
use stt_core::metadata::Metadata;
use stt_core::pack::{Manifest, PackedReader};
use stt_core::types::TimeRange;
use stt_core::TileEntry;

/// A local packed STT dataset (`manifest.json` + `index/*.sttd` + `packs/*.sttp`)
/// opened for analysis.
///
/// Directory entries are cheap (decoded once at open); payload reads go
/// through the pack mmaps on demand, so directory-only passes never touch
/// tile bytes.
pub struct PackedTileset {
    reader: PackedReader,
    manifest: Manifest,
    manifest_path: PathBuf,
}

impl PackedTileset {
    /// Open a packed dataset. `path` may be either the dataset directory
    /// (containing `manifest.json`) or the `manifest.json` file itself.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let manifest_path = if path.is_dir() {
            path.join("manifest.json")
        } else {
            path.to_path_buf()
        };
        let manifest = Manifest::from_json_bytes(
            &fs::read(&manifest_path)
                .with_context(|| format!("reading {}", manifest_path.display()))?,
        )?;
        let reader = PackedReader::open(&manifest_path)
            .with_context(|| format!("opening packed dataset {}", manifest_path.display()))?;
        Ok(Self {
            reader,
            manifest,
            manifest_path,
        })
    }

    /// Path of the `manifest.json` this tileset was opened from.
    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// The parsed packed manifest (pack table, directory pointer, metadata).
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Dataset metadata (the manifest's folded `Metadata`).
    pub fn metadata(&self) -> &Metadata {
        self.reader.metadata()
    }

    /// Dataset name from the metadata.
    pub fn name(&self) -> &str {
        &self.reader.metadata().name
    }

    /// `(min_zoom, max_zoom)` from the metadata.
    pub fn zoom_range(&self) -> (u8, u8) {
        let m = self.reader.metadata();
        (m.min_zoom, m.max_zoom)
    }

    /// Time range (Unix ms) from the metadata.
    pub fn time_range(&self) -> TimeRange {
        self.reader.metadata().time_range
    }

    /// Total directory entry count (derived from the directory at write time).
    pub fn tile_count(&self) -> u64 {
        self.reader.metadata().tile_count
    }

    /// Total feature records summed across tiles (a feature landing in N
    /// tiles counts N times — the index-weighted total).
    pub fn feature_count(&self) -> u64 {
        self.reader.metadata().feature_count
    }

    /// Base temporal bucket size in milliseconds.
    pub fn temporal_bucket_ms(&self) -> u64 {
        self.reader.metadata().temporal_bucket_ms
    }

    /// Number of pack objects in the manifest's pack table.
    pub fn pack_count(&self) -> usize {
        self.manifest.packs.len()
    }

    /// Whether the directory ships in the paged container (root + leaf pages)
    /// rather than the single whole-load object. Reading is identical either
    /// way; this only matters for reporting.
    pub fn is_paged(&self) -> bool {
        self.manifest.directory.is_paged()
    }

    /// All directory entries, sorted by (zoom, hilbert, time_start).
    ///
    /// Each [`TileEntry`] carries the tile address (`zoom`, `x`, `y`,
    /// `time_start`/`time_end`, `temporal_bucket_ms`) and the blob location
    /// (`pack_id`, `offset`, `length`, `crc32c`). Byte-identical tiles were
    /// deduped at write time, so `(pack_id, offset)` identifies the physical
    /// blob: entries sharing that pair share one blob.
    pub fn entries(&self) -> &[TileEntry] {
        self.reader.entries()
    }

    /// Read and decompress one tile's raw payload bytes (CRC-verified).
    pub fn read_payload(&self, entry: &TileEntry) -> Result<Vec<u8>> {
        Ok(self.reader.read_payload(entry)?)
    }

    /// Read one tile and decode it into its Arrow layers (each a named
    /// [`RecordBatch`](arrow::array::RecordBatch)).
    pub fn read_layers(&self, entry: &TileEntry) -> Result<Vec<DecodedLayer>> {
        Ok(self.reader.read_layers(entry)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::arrow_tile::{encode_tile_with, ColumnarLayer, EncoderConfig, GeometryColumn};
    use stt_core::curve::BlobOrdering;
    use stt_core::pack::PackWriter;
    use stt_core::tile::TileId;

    /// Frames must match the writer's (default v2) manifest declaration —
    /// mixed-version datasets are a reader hard error.
    fn tiny_tile(seed: u64, w: &PackWriter) -> Vec<u8> {
        let n = 4usize;
        let cfg = EncoderConfig {
            format_version: w.format_version(),
            template_collector: Some(w.template_collector()),
            ..EncoderConfig::default()
        };
        encode_tile_with(
            &[ColumnarLayer {
                polygon_parts: None,
                name: "default".to_string(),
                feature_ids: (0..n as u64).map(|i| seed * 100 + i).collect(),
                start_times: vec![0; n],
                end_times: vec![100; n],
                geometry: GeometryColumn::Point(vec![[-73.6, 45.5]; n]),
                vertex_times: None,
                vertex_values: None,
                triangles: None,
                vertex_value_matrix: None,
                properties: vec![],
            }],
            &cfg,
        )
        .unwrap()
    }

    /// `open` accepts both the dataset dir and the manifest path, and the
    /// wrapper's accessors agree with the written dataset.
    #[test]
    fn open_by_dir_or_manifest_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dataset");
        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 64 * 1024).unwrap();
        let payloads: Vec<Vec<u8>> = (0..3).map(|k| tiny_tile(k, &w)).collect();
        for (k, p) in payloads.iter().enumerate() {
            let id = TileId::new(7, k as u32, 0, 0);
            w.add_tile_full(&id, 0, 3_599_999, Some(0), 4, Some(3_600_000), p)
                .unwrap();
        }
        let meta =
            stt_core::metadata::Metadata::new("packed-open").with_temporal_bucket_ms(3_600_000);
        w.finalize(&meta).unwrap();

        let by_dir = PackedTileset::open(&out).unwrap();
        let by_manifest = PackedTileset::open(out.join("manifest.json")).unwrap();
        for ts in [&by_dir, &by_manifest] {
            assert_eq!(ts.name(), "packed-open");
            assert_eq!(ts.tile_count(), 3);
            assert_eq!(ts.feature_count(), 12);
            assert_eq!(ts.temporal_bucket_ms(), 3_600_000);
            assert_eq!(ts.entries().len(), 3);
            assert!(!ts.is_paged());
            assert!(ts.pack_count() >= 1);
        }

        // Payload + Arrow decode round-trip through the wrapper.
        let e = &by_dir.entries()[0];
        let payload = by_dir.read_payload(e).unwrap();
        assert!(payloads.contains(&payload));
        let layers = by_dir.read_layers(e).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "default");
        assert_eq!(layers[0].batch.num_rows(), 4);
    }

    /// A paged-directory dataset opens and reads identically (the stt-core
    /// reader subsumes both container layouts).
    #[test]
    fn paged_layout_reads_like_single() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("paged");
        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 64 * 1024)
            .unwrap()
            .with_paging(Some(2));
        for k in 0..6u64 {
            let p = tiny_tile(k, &w);
            let id = TileId::new(9, k as u32, 0, 0);
            w.add_tile_full(&id, 0, 100, None, 4, None, &p).unwrap();
        }
        w.finalize(&stt_core::metadata::Metadata::new("paged-open"))
            .unwrap();

        let ts = PackedTileset::open(&out).unwrap();
        assert!(ts.is_paged());
        assert_eq!(ts.entries().len(), 6);
        for e in ts.entries() {
            assert_eq!(ts.read_layers(e).unwrap()[0].batch.num_rows(), 4);
        }
    }
}
