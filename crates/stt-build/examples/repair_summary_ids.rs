//! Repair a packed summary-tier archive whose cell `id` column was written as
//! sequential row indices (0,1,2,…) instead of the real H3/Quadbin cell index.
//!
//! Root cause (see investigation): some frozen summary archives shipped with the
//! `id` column carrying `0..n` per tile instead of the cell index. The summary
//! layers decode `id` AS the cell (`splitLongToH3Index` / quadbin decode), so
//! every cell resolves to an invalid cell and renders nothing.
//!
//! The fix is lossless and needs no source data: every summary cell stores its
//! CENTROID as the tile geometry, and the cell id is exactly recoverable from it
//! (`LatLng -> to_cell(res)` for H3; `lonlat_to_quadbin` for Quadbin) at the
//! tier's per-zoom resolution. This tool decodes each tile, swaps ONLY the `id`
//! column (every other column — aggregates, geometry, schema metadata — is
//! preserved byte-exactly), re-encodes the layer frame, and writes a fresh
//! packed archive.
//!
//! Usage: cargo run -p stt-build --example repair_summary_ids -- <in_manifest.json> <out_dir>

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float64Array, Int32Array, RecordBatch, UInt64Array};
use arrow::ipc::writer::StreamWriter;
use h3o::{LatLng, Resolution};
use stt_core::arrow_tile::{decode_tile, DecodedLayer, QuantAffine, STT_QUANT_META_KEY};
use stt_core::metadata::SummaryScheme;
use stt_core::{BlobOrdering, PackWriter, PackedReader, TileId};
use stt_build::quadbin;

const FRAME_ALIGN: usize = 8;
const ALIGNED_FRAME_FLAG: u16 = 0x8000;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: repair_summary_ids <in_manifest.json> <out_dir>");
        std::process::exit(2);
    }
    let in_manifest = &args[1];
    let out_dir = &args[2];

    let reader = PackedReader::open(in_manifest)?;
    let meta = reader.metadata().clone();
    let tier = meta
        .summary_tier
        .clone()
        .ok_or_else(|| anyhow::anyhow!("archive has no summary tier — nothing to repair"))?;
    let layer_name = tier.layer_name.clone();
    let entries = reader.entries().to_vec();

    println!(
        "repairing {} ({:?} summary, layer '{}', {} tiles) -> {}",
        in_manifest,
        tier.scheme,
        layer_name,
        entries.len(),
        out_dir
    );

    let mut writer = PackWriter::create(out_dir, BlobOrdering::Auto, 64 * 1024 * 1024)?
        .with_paging(Some(4096))
        .with_zstd_level(19);

    let mut tiles_touched = 0usize;
    let mut cells_fixed = 0usize;
    let mut verbatim = 0usize;

    for e in &entries {
        let payload = reader.read_payload(e)?;
        let layers = decode_tile(&payload)?;
        let has_summary = layers.iter().any(|l| l.name == layer_name);
        if !has_summary {
            // No summary layer in this tile — copy the frame byte-for-byte.
            writer.add_tile_full(
                &TileId::new(e.zoom, e.x, e.y, e.time_start.max(0) as u64),
                e.time_start,
                e.time_end,
                e.cover_t_min,
                e.feature_count,
                e.temporal_bucket_ms,
                &payload,
            )?;
            verbatim += 1;
            continue;
        }

        // Resolution/zoom for this tile's map-zoom (per-zoom table indexed from minZoom).
        let idx = e.zoom.checked_sub(tier.min_zoom).ok_or_else(|| {
            anyhow::anyhow!("tile zoom {} below tier minZoom {}", e.zoom, tier.min_zoom)
        })? as usize;
        let cell_res = *tier.cell_resolution_per_zoom.get(idx).ok_or_else(|| {
            anyhow::anyhow!(
                "no cell resolution for zoom {} (idx {} of {})",
                e.zoom,
                idx,
                tier.cell_resolution_per_zoom.len()
            )
        })?;

        let mut new_layers: Vec<DecodedLayer> = Vec::with_capacity(layers.len());
        for layer in layers {
            if layer.name != layer_name {
                new_layers.push(layer);
                continue;
            }
            let fixed = fix_layer_ids(&layer, tier.scheme, cell_res)?;
            cells_fixed += fixed.batch.num_rows();
            new_layers.push(fixed);
        }

        let new_payload = encode_frame(&new_layers)?;
        writer.add_tile_full(
            &TileId::new(e.zoom, e.x, e.y, e.time_start.max(0) as u64),
            e.time_start,
            e.time_end,
            e.cover_t_min,
            e.feature_count,
            e.temporal_bucket_ms,
            &new_payload,
        )?;
        tiles_touched += 1;
    }

    let manifest = writer.finalize(&meta)?;
    println!(
        "done: {} summary tiles repaired ({} cells), {} tiles copied verbatim; wrote {} packs",
        tiles_touched,
        cells_fixed,
        verbatim,
        manifest.packs.len()
    );
    Ok(())
}

/// Recompute the `id` column of one decoded summary layer from its per-cell
/// centroid geometry; every other column is preserved.
fn fix_layer_ids(
    layer: &DecodedLayer,
    scheme: SummaryScheme,
    cell_res: u8,
) -> anyhow::Result<DecodedLayer> {
    let batch = &layer.batch;
    let schema = batch.schema();
    let n = batch.num_rows();

    let id_idx = schema
        .index_of("id")
        .map_err(|_| anyhow::anyhow!("summary layer has no 'id' column"))?;

    // Geometry: interleaved [lon, lat] point, either quantized i32 (with a
    // `stt:quant` affine on the field) or plain f64.
    let geom_field = schema.field_with_name("geometry")?;
    let affine = geom_field
        .metadata()
        .get(STT_QUANT_META_KEY)
        .and_then(|s| QuantAffine::from_json(s));

    let geom = batch
        .column_by_name("geometry")
        .ok_or_else(|| anyhow::anyhow!("summary layer has no 'geometry' column"))?;
    let fsl = geom
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow::anyhow!("geometry is not a FixedSizeList"))?;
    let stride = fsl.value_length() as usize; // 2 (lon,lat) or 3 (lon,lat,z)
    if stride < 2 {
        anyhow::bail!("geometry stride {} < 2", stride);
    }
    let child = fsl.values();

    // Only H3 needs a validated Resolution; Quadbin's `cell_res` is a tile zoom
    // (can exceed 15, which is not a valid H3 resolution).
    let h3_res = match scheme {
        SummaryScheme::H3 => Some(
            Resolution::try_from(cell_res)
                .map_err(|e| anyhow::anyhow!("bad H3 resolution {cell_res}: {e}"))?,
        ),
        SummaryScheme::Quadbin => None,
    };

    let mut new_ids: Vec<u64> = Vec::with_capacity(n);

    let read_lonlat: Box<dyn Fn(usize) -> (f64, f64)> =
        if let Some(i32arr) = child.as_any().downcast_ref::<Int32Array>() {
            let aff = affine
                .ok_or_else(|| anyhow::anyhow!("i32 geometry without a stt:quant affine"))?;
            Box::new(move |i| {
                let qx = i32arr.value(i * stride);
                let qy = i32arr.value(i * stride + 1);
                (aff.lon(qx), aff.lat(qy))
            })
        } else if let Some(f64arr) = child.as_any().downcast_ref::<Float64Array>() {
            Box::new(move |i| (f64arr.value(i * stride), f64arr.value(i * stride + 1)))
        } else {
            anyhow::bail!("unsupported geometry child type {:?}", child.data_type());
        };

    for i in 0..n {
        let (lon, lat) = read_lonlat(i);
        let id = match scheme {
            SummaryScheme::H3 => {
                let ll = LatLng::new(lat, lon)
                    .map_err(|e| anyhow::anyhow!("bad centroid ({lon},{lat}): {e}"))?;
                let cell = ll.to_cell(h3_res.unwrap());
                u64::from(cell)
            }
            SummaryScheme::Quadbin => quadbin::lonlat_to_quadbin(lon, lat, cell_res),
        };
        new_ids.push(id);
    }

    let mut columns = batch.columns().to_vec();
    columns[id_idx] = Arc::new(UInt64Array::from(new_ids));
    let new_batch = RecordBatch::try_new(schema.clone(), columns)?;
    Ok(DecodedLayer {
        name: layer.name.clone(),
        batch: new_batch,
    })
}

/// Re-assemble a tile layer frame from decoded layers — mirrors
/// `arrow_tile::encode_tile_cfg`'s aligned frame, but serializes each layer's
/// already-decoded RecordBatch straight to IPC (no re-quantization).
fn encode_frame(layers: &[DecodedLayer]) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&(layers.len() as u16 | ALIGNED_FRAME_FLAG).to_le_bytes());
    for layer in layers {
        let ipc = encode_batch_ipc(&layer.batch)?;
        let name = layer.name.as_bytes();
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&(ipc.len() as u32).to_le_bytes());
        let pad = (FRAME_ALIGN - out.len() % FRAME_ALIGN) % FRAME_ALIGN;
        out.extend_from_slice(&[0u8; FRAME_ALIGN][..pad]);
        out.extend_from_slice(&ipc);
    }
    Ok(out)
}

fn encode_batch_ipc(batch: &RecordBatch) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut buf, &batch.schema())?;
        w.write(batch)?;
        w.finish()?;
    }
    Ok(buf)
}
