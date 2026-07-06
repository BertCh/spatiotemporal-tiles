//! Re-optimize an existing packed dataset IN PLACE (data-preserving): decode
//! every tile, rebuild it as a `ColumnarLayer`, and re-encode through the
//! production encoder so the opt-in size levers apply without re-running the
//! source generator. Captures, per a per-dataset config:
//!   * sequential point ids   (anonymous POINT features → row index; the
//!     incompressible hash-id is often the single largest column)
//!   * coordinate quantization (--quantize-coords M; f64 lon/lat → i32 grid)
//!   * numeric-attribute quant (--quantize-attr name=prec; f64 prop → u16/i32)
//!   * dead-column drops       (--drop name)
//!
//! The geometry, vertex-time, vertex-value, and triangle columns are
//! reconstructed faithfully and re-encoded (the encoder re-applies coord/attr
//! quantization from the process-global config), so the output is the same data,
//! smaller. Tile coordinates, times, covering bounds, and metadata are preserved.
//!
//! Usage:
//!   cargo run --release -p stt-core --example reoptimize -- \
//!     <src/manifest.json> <out_dir> [--quantize-coords M] \
//!     [--quantize-attr name=prec]... [--drop col]... [--no-seq-ids] [--zstd N]

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use arrow::array::{
    Array, DictionaryArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array,
    Int64Array, ListArray, RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, UInt16Type};
use stt_core::arrow_tile::{
    decode_tile, decode_tile_with_templates, encode_tile_with, set_quantize_attrs,
    set_quantize_attrs_auto, set_quantize_coords_m, AttrQuant, ColumnarLayer, Coord,
    EncoderConfig, GeometryColumn, PropertyColumn, QuantAffine, STT_QUANT_ATTR_META_KEY,
};
use stt_core::pack::{PackWriter, PackedReader};
use stt_core::tile::TileId;
use stt_core::BlobOrdering;

const RESERVED: &[&str] = &[
    "id",
    "start_time",
    "end_time",
    "geometry",
    "vertex_time",
    "vertex_value",
    "vertex_value_matrix",
    "triangles",
];

/// Read the geometry field's quantization affine, if the tile is quantized.
fn geom_affine(batch: &RecordBatch) -> Option<QuantAffine> {
    let f = batch.schema().field_with_name("geometry").ok()?.clone();
    f.metadata().get("stt:quant").and_then(|s| QuantAffine::from_json(s))
}

/// One [x,y] coord from a leaf (Float64 or quantized Int32) at index `i` (the
/// coordinate-pair index — the leaf holds interleaved x,y so element `2*i`/`2*i+1`).
fn coord_at(leaf: &dyn Array, i: usize, aff: Option<&QuantAffine>) -> Coord {
    match aff {
        Some(a) => {
            let v = leaf.as_any().downcast_ref::<Int32Array>().unwrap();
            [a.lon(v.value(2 * i)), a.lat(v.value(2 * i + 1))]
        }
        None => {
            let v = leaf.as_any().downcast_ref::<Float64Array>().unwrap();
            [v.value(2 * i), v.value(2 * i + 1)]
        }
    }
}

/// Reconstruct the `GeometryColumn` (always as Float64 coords) from the tile's
/// geometry array, inverting coordinate quantization when present.
fn read_geometry(batch: &RecordBatch, kind: &str) -> GeometryColumn {
    let aff = geom_affine(batch);
    let col = batch.column_by_name("geometry").unwrap();
    match kind {
        "geoarrow.point" => {
            let fsl = col.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let leaf = fsl.values();
            let n = fsl.len();
            let pts = (0..n).map(|i| coord_at(leaf.as_ref(), i, aff.as_ref())).collect();
            GeometryColumn::Point(pts)
        }
        "geoarrow.linestring" => {
            let list = col.as_any().downcast_ref::<ListArray>().unwrap();
            let mut lines = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                let verts = list.value(i);
                let fsl = verts.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                let leaf = fsl.values();
                let line = (0..fsl.len())
                    .map(|v| coord_at(leaf.as_ref(), v, aff.as_ref()))
                    .collect();
                lines.push(line);
            }
            GeometryColumn::LineString(lines)
        }
        "geoarrow.polygon" => {
            let feats = col.as_any().downcast_ref::<ListArray>().unwrap();
            let mut polys = Vec::with_capacity(feats.len());
            for i in 0..feats.len() {
                let rings_arr = feats.value(i);
                let rings_list = rings_arr.as_any().downcast_ref::<ListArray>().unwrap();
                let mut rings = Vec::with_capacity(rings_list.len());
                for r in 0..rings_list.len() {
                    let verts = rings_list.value(r);
                    let fsl = verts.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                    let leaf = fsl.values();
                    let ring = (0..fsl.len())
                        .map(|v| coord_at(leaf.as_ref(), v, aff.as_ref()))
                        .collect();
                    rings.push(ring);
                }
                polys.push(rings);
            }
            GeometryColumn::Polygon(polys)
        }
        other => panic!("unknown geometry kind {other}"),
    }
}

/// Reconstruct optional per-vertex `List<inner>` columns to `Vec<Vec<T>>`.
fn read_list_i64_vertex_time(batch: &RecordBatch) -> Option<Vec<Vec<i64>>> {
    let col = batch.column_by_name("vertex_time")?;
    let list = col.as_any().downcast_ref::<ListArray>()?;
    // u16-delta encoding carries origin/step in schema metadata.
    let meta = batch.schema().metadata().clone();
    let origin: Option<i64> = meta.get("stt:vertex_time_origin_ms").and_then(|s| s.parse().ok());
    let step: Option<i64> = meta.get("stt:vertex_time_step_ms").and_then(|s| s.parse().ok());
    let mut out = Vec::with_capacity(list.len());
    for i in 0..list.len() {
        if list.is_null(i) {
            out.push(Vec::new());
            continue;
        }
        let v = list.value(i);
        if let (Some(o), Some(st)) = (origin, step) {
            let d = v.as_any().downcast_ref::<UInt16Array>().unwrap();
            out.push((0..d.len()).map(|j| o + d.value(j) as i64 * st).collect());
        } else {
            let a = v.as_any().downcast_ref::<Int64Array>().unwrap();
            out.push(a.values().to_vec());
        }
    }
    Some(out)
}

fn read_list_f32(batch: &RecordBatch, name: &str) -> Option<Vec<Vec<f32>>> {
    let col = batch.column_by_name(name)?;
    let list = col.as_any().downcast_ref::<ListArray>()?;
    let mut out = Vec::with_capacity(list.len());
    for i in 0..list.len() {
        if list.is_null(i) {
            out.push(Vec::new());
        } else {
            let v = list.value(i);
            let a = v.as_any().downcast_ref::<Float32Array>().unwrap();
            out.push(a.values().to_vec());
        }
    }
    Some(out)
}

fn read_triangles(batch: &RecordBatch) -> Option<Vec<Vec<u32>>> {
    let col = batch.column_by_name("triangles")?;
    let list = col.as_any().downcast_ref::<ListArray>()?;
    let mut out = Vec::with_capacity(list.len());
    for i in 0..list.len() {
        if list.is_null(i) {
            out.push(Vec::new());
        } else {
            let v = list.value(i);
            let a = v.as_any().downcast_ref::<UInt32Array>().unwrap();
            out.push(a.values().to_vec());
        }
    }
    Some(out)
}

/// Read one property column to a `PropertyColumn` (Float64 → Numeric;
/// Dictionary → Categorical; already-quantized u16/i32+`stt:qa` → reconstructed
/// Numeric).
fn read_property(batch: &RecordBatch, idx: usize) -> (String, PropertyColumn) {
    let field = batch.schema().field(idx).clone();
    let name = field.name().clone();
    let col = batch.column(idx);
    let qa = field
        .metadata()
        .get(STT_QUANT_ATTR_META_KEY)
        .and_then(|s| AttrQuant::from_json(s));
    let pc = match col.data_type() {
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
            PropertyColumn::Numeric(
                (0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))).collect(),
            )
        }
        DataType::UInt16 => {
            let a = col.as_any().downcast_ref::<UInt16Array>().unwrap();
            let aff = qa.expect("u16 property without stt:qa affine");
            PropertyColumn::Numeric(
                (0..a.len())
                    .map(|i| (!a.is_null(i)).then(|| aff.value(a.value(i) as i64)))
                    .collect(),
            )
        }
        DataType::Int32 => {
            let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
            let aff = qa.expect("i32 property without stt:qa affine");
            PropertyColumn::Numeric(
                (0..a.len())
                    .map(|i| (!a.is_null(i)).then(|| aff.value(a.value(i) as i64)))
                    .collect(),
            )
        }
        DataType::Dictionary(_, _) => {
            let dict = col.as_any().downcast_ref::<DictionaryArray<UInt16Type>>().unwrap();
            let vals = dict.values().as_any().downcast_ref::<StringArray>().unwrap();
            let keys = dict.keys();
            PropertyColumn::Categorical(
                (0..keys.len())
                    .map(|i| (!keys.is_null(i)).then(|| vals.value(keys.value(i) as usize).to_string()))
                    .collect(),
            )
        }
        // Plain Utf8 (the pre-v3 categorical shape some older archives still
        // carry). Read as Categorical; the encoder re-dictionaries it to the
        // canonical Dictionary<UInt16,Utf8> on the way out.
        DataType::Utf8 => {
            let a = col.as_any().downcast_ref::<StringArray>().unwrap();
            PropertyColumn::Categorical(
                (0..a.len())
                    .map(|i| (!a.is_null(i)).then(|| a.value(i).to_string()))
                    .collect(),
            )
        }
        other => panic!("property '{name}' has unsupported type {other:?}"),
    };
    (name, pc)
}

/// Rebuild a `ColumnarLayer` from a decoded tile layer's `RecordBatch`.
fn batch_to_columnar(batch: &RecordBatch, name: String) -> ColumnarLayer {
    let kind = batch
        .schema()
        .metadata()
        .get("stt:geometry")
        .cloned()
        .unwrap_or_else(|| "geoarrow.point".to_string());

    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .values()
        .to_vec();
    let start = batch
        .column_by_name("start_time")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    let end = batch
        .column_by_name("end_time")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();

    let geometry = read_geometry(batch, &kind);
    let vertex_times = read_list_i64_vertex_time(batch);
    let vertex_values = read_list_f32(batch, "vertex_value");
    let vertex_value_matrix = read_list_f32(batch, "vertex_value_matrix");
    let triangles = read_triangles(batch);

    let mut properties = Vec::new();
    for idx in 0..batch.num_columns() {
        let n = batch.schema().field(idx).name().to_string();
        if RESERVED.contains(&n.as_str()) {
            continue;
        }
        properties.push(read_property(batch, idx));
    }

    ColumnarLayer {
        name,
        feature_ids: ids,
        start_times: start,
        end_times: end,
        geometry,
        vertex_times,
        vertex_values,
        vertex_value_matrix,
        triangles,
        properties,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: reoptimize <src/manifest.json> <out_dir> [--format-version N] [--quantize-coords M] [--quantize-attr n=p]... [--drop col]... [--no-seq-ids] [--zstd N]");
        std::process::exit(2);
    }
    let src = &args[0];
    let out = &args[1];
    let mut coord_m = 0.0f64;
    let mut attrs: HashMap<String, f64> = HashMap::new();
    let mut drop: HashSet<String> = HashSet::new();
    let mut seq_ids = true;
    let mut auto = false;
    let mut zstd = 19;
    // None = preserve the source's formatVersion (data-preserving default);
    // Some(2) = migrate v1 packed data to the v2 container without re-fetching.
    let mut format_version: Option<u32> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--format-version" => { format_version = Some(args[i + 1].parse()?); i += 2; }
            "--quantize-coords" => { coord_m = args[i + 1].parse()?; i += 2; }
            "--quantize-attr" => {
                let (n, p) = args[i + 1].split_once('=').expect("name=prec");
                attrs.insert(n.to_string(), p.parse()?);
                i += 2;
            }
            "--quantize-attrs-auto" => { auto = true; i += 1; }
            "--drop" => { drop.insert(args[i + 1].clone()); i += 2; }
            "--no-seq-ids" => { seq_ids = false; i += 1; }
            "--zstd" => { zstd = args[i + 1].parse()?; i += 2; }
            other => { eprintln!("unknown arg {other}"); std::process::exit(2); }
        }
    }

    // Process-global encoder config (the encoder reads these on encode_tile).
    set_quantize_coords_m(coord_m)?;
    set_quantize_attrs(attrs.clone());
    set_quantize_attrs_auto(auto);

    let reader = PackedReader::open(src)?;
    let meta = reader.metadata().clone();
    let entries = reader.entries().to_vec();
    // Tiles are fully re-encoded, so the output's capability declarations
    // (spec §3.1) come from THIS run's settings, not the source manifest.
    // The elevation fold survives decode→re-encode as a 3-wide leaf, so that
    // one is carried forward from the source.
    let mut capabilities: Vec<String> = Vec::new();
    if coord_m > 0.0 {
        capabilities.push(stt_core::pack::CAPABILITY_COORD_QUANT.to_string());
    }
    if auto || !attrs.is_empty() {
        capabilities.push(stt_core::pack::CAPABILITY_ATTR_QUANT.to_string());
    }
    if reader
        .capabilities()
        .iter()
        .any(|c| c == stt_core::pack::CAPABILITY_ELEVATION_FOLD)
    {
        capabilities.push(stt_core::pack::CAPABILITY_ELEVATION_FOLD.to_string());
    }
    let mut writer = PackWriter::create(out, BlobOrdering::Auto, 64 * 1024 * 1024)?
        .with_paging(Some(4096))
        .with_zstd_level(zstd)
        .with_format_version(format_version.unwrap_or_else(|| reader.format_version()))
        .with_capabilities(capabilities);
    // Explicit encoder config, mirroring stt-build's `pack_encoder_config`:
    // the frame version + template sink come FROM THE WRITER (which carries
    // the source's formatVersion), layered over the process-wide quant
    // globals set above — re-encoded frames and the output manifest can
    // never disagree, and a v2 build's templates land in `manifest.schemas`.
    let encoder_cfg = EncoderConfig {
        format_version: writer.format_version(),
        template_collector: (writer.format_version()
            == stt_core::pack::PACKED_FORMAT_VERSION_V2)
            .then(|| writer.template_collector()),
        ..EncoderConfig::from_globals()
    };
    // v2 sources resolve frame template hashes through the manifest registry.
    let templates = reader.templates().cloned();

    for e in entries.chunks(2048) {
        // Read payloads sequentially (the reader's lazy mmap is RefCell, not
        // Sync) but parallelize the CPU-bound decode → rebuild → re-encode over
        // each chunk — that transform is the cost, and it's pure given the
        // read-only process-global quant config.
        let payloads: Vec<Vec<u8>> = e
            .iter()
            .map(|en| reader.read_payload(en))
            .collect::<stt_core::Result<Vec<_>>>()?;
        let encoded: Vec<Vec<u8>> = payloads
            .par_iter()
            .map(|payload| -> std::result::Result<Vec<u8>, String> {
                let layers = match &templates {
                    Some(t) => decode_tile_with_templates(payload, t),
                    None => decode_tile(payload),
                }
                .map_err(|e| e.to_string())?;
                let cols: Vec<ColumnarLayer> = layers
                    .iter()
                    .map(|dl| {
                        let mut cl = batch_to_columnar(&dl.batch, dl.name.clone());
                        // Sequential ids: POINT layers only (a clipped line/polygon
                        // keeps a stable id across the tiles it spans; a point lives
                        // in exactly one tile).
                        if seq_ids && matches!(cl.geometry, GeometryColumn::Point(_)) {
                            cl.feature_ids = (0..cl.feature_ids.len() as u64).collect();
                        }
                        if !drop.is_empty() {
                            cl.properties.retain(|(n, _)| !drop.contains(n));
                        }
                        cl
                    })
                    .collect();
                encode_tile_with(&cols, &encoder_cfg).map_err(|e| e.to_string())
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;
        for (en, new_payload) in e.iter().zip(&encoded) {
            let id = TileId::new(en.zoom, en.x, en.y, en.time_start.max(0) as u64);
            writer.add_tile_full(
                &id,
                en.time_start,
                en.time_end,
                en.cover_t_min,
                en.feature_count,
                en.temporal_bucket_ms,
                new_payload,
            )?;
        }
    }
    let manifest = writer.finalize(&meta)?;
    let bytes: u64 = manifest.packs.iter().map(|p| p.length as u64).sum();
    println!(
        "reoptimized {} -> {} ({} tiles, {:.1} MB, coord_q={} attrs={:?} drop={:?} seq_ids={})",
        src,
        out,
        entries.len(),
        bytes as f64 / 1e6,
        coord_m,
        attrs,
        drop,
        seq_ids
    );
    Ok(())
}
