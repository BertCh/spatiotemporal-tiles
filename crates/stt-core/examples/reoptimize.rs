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
    Array, DictionaryArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    ListArray, RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, UInt16Type};
use stt_core::arrow_tile::{
    decode_tile, decode_tile_with_templates, encode_tile_with, set_quantize_attrs,
    set_quantize_attrs_auto, set_quantize_coords_m, AttrQuant, ColumnarLayer, Coord, EncoderConfig,
    GeometryColumn, PropertyColumn, QuantAffine, STT_QUANT_ATTR_META_KEY,
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

/// Metres per degree of latitude — the constant `world_grid_affine` sizes the
/// coordinate grid from (`sx = sy = metres / M_PER_DEG_LAT`). Inverting it
/// recovers the ORIGINAL `--quantize-coords` value from a tile's affine.
const M_PER_DEG_LAT: f64 = 111_320.0;

/// Recover the source's quantization settings from an already-encoded tile.
///
/// `decode_tile` hands back the raw Arrow batch — Int32 leaves plus the
/// `stt:quant` / `stt:qa` affines — and this example's `read_geometry` /
/// `read_property` then INVERT them to Float64. Re-encoding without restoring
/// the same settings therefore silently writes a dequantized copy: correct
/// values (they stay snapped to the original grid) but far bigger, since coords
/// are the dominant near-incompressible column. Measured on the showcase fleet
/// that cost up to 1.69x (satellites 1.78 GB -> 3.0 GB).
///
/// Returns `(coord_metres, per-column attr steps)`; either may be empty when the
/// source was not quantized on that axis.
fn recover_quant(batch: &RecordBatch) -> (f64, HashMap<String, f64>) {
    let coord_m = geom_affine(batch)
        // sy is degrees-per-quantum; the requested ground precision is its
        // metre equivalent. Round-tripping through f64 is exact enough that
        // re-quantizing lands on the identical grid.
        .map(|a| a.sy * M_PER_DEG_LAT)
        .unwrap_or(0.0);
    let mut attrs = HashMap::new();
    for f in batch.schema().fields() {
        if RESERVED.contains(&f.name().as_str()) {
            continue;
        }
        if let Some(q) = f
            .metadata()
            .get(STT_QUANT_ATTR_META_KEY)
            .and_then(|s| AttrQuant::from_json(s))
        {
            attrs.insert(f.name().to_string(), q.s);
        }
    }
    (coord_m, attrs)
}

/// Read the geometry field's quantization affine, if the tile is quantized.
fn geom_affine(batch: &RecordBatch) -> Option<QuantAffine> {
    let f = batch.schema().field_with_name("geometry").ok()?.clone();
    f.metadata()
        .get("stt:quant")
        .and_then(|s| QuantAffine::from_json(s))
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
            let pts = (0..n)
                .map(|i| coord_at(leaf.as_ref(), i, aff.as_ref()))
                .collect();
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
    let origin: Option<i64> = meta
        .get("stt:vertex_time_origin_ms")
        .and_then(|s| s.parse().ok());
    let step: Option<i64> = meta
        .get("stt:vertex_time_step_ms")
        .and_then(|s| s.parse().ok());
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
                (0..a.len())
                    .map(|i| (!a.is_null(i)).then(|| a.value(i)))
                    .collect(),
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
            let dict = col
                .as_any()
                .downcast_ref::<DictionaryArray<UInt16Type>>()
                .unwrap();
            let vals = dict
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let keys = dict.keys();
            PropertyColumn::Categorical(
                (0..keys.len())
                    .map(|i| {
                        (!keys.is_null(i)).then(|| vals.value(keys.value(i) as usize).to_string())
                    })
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
        eprintln!("usage: reoptimize <src/manifest.json> <out_dir> [--format-version N] [--blob-ordering time-major|spatial|hilbert3|morton3|auto] [--quantize-coords M] [--quantize-attr n=p]... [--drop col]... [--no-seq-ids] [--zstd N]");
        std::process::exit(2);
    }
    let src = &args[0];
    let out = &args[1];
    let mut coord_m = 0.0f64;
    let mut attrs: HashMap<String, f64> = HashMap::new();
    let mut drop: HashSet<String> = HashSet::new();
    let mut seq_ids = true;
    // Blob ordering for the output pack. v1 archives predate `manifest.blobOrdering`,
    // so a migration has nothing to preserve and `Auto` would re-derive one —
    // which resolves to `SpatialMajor` on space-dominated datasets and silently
    // breaks time playback (empty buffered ranges → stalls). Callers migrating a
    // playback dataset must pass `--blob-ordering time-major` explicitly.
    let mut ordering = BlobOrdering::Auto;
    let mut auto = false;
    let mut zstd = 19;
    // None = preserve the source's formatVersion (data-preserving default);
    // Some(2) = migrate v1 packed data to the v2 container without re-fetching.
    let mut format_version: Option<u32> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--format-version" => {
                format_version = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--quantize-coords" => {
                coord_m = args[i + 1].parse()?;
                i += 2;
            }
            "--quantize-attr" => {
                let (n, p) = args[i + 1].split_once('=').expect("name=prec");
                attrs.insert(n.to_string(), p.parse()?);
                i += 2;
            }
            "--quantize-attrs-auto" => {
                auto = true;
                i += 1;
            }
            "--drop" => {
                drop.insert(args[i + 1].clone());
                i += 2;
            }
            "--no-seq-ids" => {
                seq_ids = false;
                i += 1;
            }
            "--blob-ordering" => {
                ordering = args[i + 1].parse()?;
                i += 2;
            }
            "--zstd" => {
                zstd = args[i + 1].parse()?;
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let reader = PackedReader::open(src)?;
    let meta = reader.metadata().clone();
    let entries = reader.entries().to_vec();

    // PRESERVE the source's quantization unless this run overrides it. Decode
    // inverts both affines to Float64, so without this a plain container
    // migration silently ships a dequantized (much larger) copy — see
    // `recover_quant`. An explicit `--quantize-*` flag still wins.
    if coord_m == 0.0 && attrs.is_empty() && !auto {
        // Sample several tiles, not just the first. `--quantize-attrs-auto`
        // (range-adaptive) derives a per-tile step of `range / u16::MAX`, so a
        // column's step VARIES across tiles; a fixed `--quantize-attr n=p` is
        // constant. Pinning tile 0's adaptive step onto every tile is what
        // introduces schema drift: a wider tile then overflows the UInt16 leaf
        // and gets promoted to Int32, so one layer ends up with two schemas.
        // Auto mode is stable by construction (it always emits UInt16), so
        // varying steps ⇒ re-enable auto instead of pinning.
        let n = entries.len();
        let step = (n / 8).max(1);
        let mut steps: HashMap<String, Vec<f64>> = HashMap::new();
        for e in entries.iter().step_by(step).take(8) {
            let payload = reader.read_payload(e)?;
            let probe = match reader.templates() {
                Some(t) => decode_tile_with_templates(&payload, t),
                None => decode_tile(&payload),
            };
            if let Some(layer) = probe.ok().and_then(|l| l.into_iter().next()) {
                let (m, a) = recover_quant(&layer.batch);
                if m > 0.0 && coord_m == 0.0 {
                    eprintln!("preserving source coord quantization: {m:.4} m");
                    coord_m = m;
                }
                for (name, s) in a {
                    steps.entry(name).or_default().push(s);
                }
            }
        }
        let adaptive = steps.values().any(|v| {
            v.windows(2)
                .any(|w| (w[0] - w[1]).abs() > f64::EPSILON * w[0].abs().max(1.0))
        });
        if adaptive {
            eprintln!("preserving source attr quantization: range-adaptive (auto)");
            auto = true;
        } else if !steps.is_empty() {
            let mut names: Vec<_> = steps.keys().cloned().collect();
            names.sort();
            eprintln!("preserving source attr quantization: {}", names.join(", "));
            attrs = steps.into_iter().map(|(k, v)| (k, v[0])).collect();
        }
    }

    // Process-global encoder config (the encoder reads these on encode_tile).
    set_quantize_coords_m(coord_m)?;
    set_quantize_attrs(attrs.clone());
    set_quantize_attrs_auto(auto);
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
    let mut writer = PackWriter::create(out, ordering, 64 * 1024 * 1024)?
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
        template_collector: (writer.format_version() == stt_core::pack::PACKED_FORMAT_VERSION)
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
