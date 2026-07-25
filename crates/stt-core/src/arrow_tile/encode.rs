//! Tile encoding — layer parts, v1 frame assembly, v2 sectioned frames.
//!
//! The public `encode_layer*` / `encode_tile*` entry points, the shared
//! [`build_layer_parts`] front end that both frame versions run, and the two
//! frame assemblers. [`EncoderConfig::format_version`] is the single branch
//! point between them (design §1 ★F3).

use super::columns::{
    build_dictionary_indices, build_geometry_array_q, build_vertex_time_array,
    build_vertex_value_array, group_vector_properties, infer_vertex_value_buckets,
};
use super::config::EncoderConfig;
use super::frame::{
    TileMeta, ALIGNED_FRAME_FLAG, FORMAT_VERSION_V1, FORMAT_VERSION_V2, FRAME_ALIGN,
    FRAME_V2_ESCAPE, FRAME_V2_VERSION, REF_KIND_INLINE, REF_KIND_NO_PROPS, REF_KIND_TEMPLATE_HASH,
    SECTION_CORE_BATCH, SECTION_INLINE_SCHEMA_CORE, SECTION_INLINE_SCHEMA_PROPS,
    SECTION_PROPS_BATCH, SECTION_TILE_META, TIME_OFFSET_MS_KEY, VERTEX_TIME_ORIGIN_KEY,
    VERTEX_TIME_STEP_KEY, VERTEX_VALUE_BUCKETS_KEY,
};
use super::layer::{
    ColumnarLayer, GeometryColumn, PropertyColumn, VectorElem, GEOARROW_CRS_METADATA,
    GEOARROW_EXT_KEY, GEOARROW_EXT_META_KEY, TRIANGLES_METADATA_KEY,
};
use super::quantize::{
    build_quantized_numeric, build_quantized_numeric_auto, validate_quantize_coords_m,
    world_grid_affine, world_grid_affine_3d, AttrQuant, STT_QUANT_ATTR_META_KEY,
    STT_QUANT_META_KEY,
};
use crate::error::{Error, Result};
use arrow::array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
    ListBuilder, RecordBatch, StringArray, UInt16Array, UInt16Builder, UInt32Builder, UInt64Array,
    UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema, UInt16Type};
use arrow::ipc::writer::StreamWriter;
use arrow::ipc::{root_as_message, MessageHeader};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

// ----------------------------------------------------------------------------
// Encoding
// ----------------------------------------------------------------------------

/// Encode a single layer to an Arrow IPC stream.
pub fn encode_layer(layer: &ColumnarLayer) -> Result<Vec<u8>> {
    encode_layer_cfg(layer, &EncoderConfig::from_globals())
}

/// [`encode_layer`] with optional fixed-point coordinate quantization.
///
/// `quantize_m = Some(meters)` stores coordinates as `i32` grid indices at that
/// ground precision (default-off `None` is byte-identical to [`encode_layer`]).
/// Coordinates are the dominant, near-incompressible tile column, so quantizing
/// them is the single largest size lever — at the cost of GeoArrow Float64
/// self-description, hence opt-in. The per-layer affine rides in the geometry
/// field metadata under [`STT_QUANT_META_KEY`]; the reader reconstructs Float64.
///
/// The non-coordinate settings (attribute quantization, vector grouping,
/// point-elevation fold, vertex-time precision) come from the process-wide
/// globals; use [`encode_tile_with`] to pass every setting explicitly.
pub fn encode_layer_quantized(layer: &ColumnarLayer, quantize_m: Option<f64>) -> Result<Vec<u8>> {
    encode_layer_cfg(
        layer,
        &EncoderConfig {
            quantize_coords_m: quantize_m,
            ..EncoderConfig::from_globals()
        },
    )
}

/// [`encode_layer`] with a fully-explicit [`EncoderConfig`] — no process-wide
/// globals are read. The single-layer sibling of [`encode_tile_with`], and the
/// entry point a caller should reach for instead of `set_*`-then-`encode_layer`:
/// that pair is what let one caller's settings leak into the next one's tiles.
pub fn encode_layer_with(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<Vec<u8>> {
    encode_layer_cfg(layer, cfg)
}

/// The single-layer encode implementation, driven entirely by an explicit
/// [`EncoderConfig`] — no process-wide globals are read. Every public
/// `encode_layer*` entry point funnels here.
///
/// Crate-private: the whole-TILE [`encode_tile_with`] is the public
/// explicit-config entry point (that is the granularity every writer works at),
/// so a public single-layer twin only ever went unused.
///
/// Always emits the self-describing v1 shape — one Arrow IPC stream with ALL
/// metadata inline. The v2 tile frame reuses the identical column/field build
/// ([`build_layer_parts`]) and changes only metadata PLACEMENT at assembly
/// ([`encode_layer_v2_parts`]) — the design's single v1/v2 branch point.
pub(crate) fn encode_layer_cfg(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<Vec<u8>> {
    let parts = build_layer_parts(layer, cfg)?;
    assemble_layer_ipc_v1(parts)
}

/// Everything a frame assembler needs about one built layer: the Arrow fields
/// + columns in canonical order (reserved columns first, then properties) and
/// the per-tile schema-metadata values. Both frame versions build this
/// identically; only where the metadata LANDS differs (v1: schema metadata;
/// v2: `TILE_META` section + stripped, dataset-constant templates).
struct LayerParts {
    fields: Vec<Arc<Field>>,
    columns: Vec<ArrayRef>,
    /// `fields[..reserved_len]` are the reserved (CORE) columns; the rest are
    /// property (PROPS) columns.
    reserved_len: usize,
    layer_name: String,
    geometry_name: &'static str,
    /// Minimum feature start-time (v1 `stt:time_offset_ms` / v2 `t0`).
    min_start_time: Option<i64>,
    /// `(origin_ms, step_ms)` of a u16-delta `vertex_time` column.
    vertex_time_encoding: Option<(i64, u32)>,
    /// v1 `stt:vertex_value_buckets` / v2 `vb`.
    vertex_value_buckets: Option<u32>,
    has_triangles: bool,
}

/// Build the Arrow fields + columns for one layer — the version-independent
/// front of the encoder (validation, quantization, vector grouping, the
/// point-elevation fold, vertex-time encoding).
fn build_layer_parts(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<LayerParts> {
    layer.validate()?;
    let n = layer.feature_count();

    let mut fields: Vec<Arc<Field>> = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    fields.push(Arc::new(Field::new("id", DataType::UInt64, false)));
    columns.push(Arc::new(UInt64Array::from(layer.feature_ids.clone())));

    fields.push(Arc::new(Field::new("start_time", DataType::Int64, false)));
    columns.push(Arc::new(Int64Array::from(layer.start_times.clone())));

    fields.push(Arc::new(Field::new("end_time", DataType::Int64, false)));
    columns.push(Arc::new(Int64Array::from(layer.end_times.clone())));

    // 3D POINT geometry: fold the configured numeric column into the geometry's
    // 3rd coordinate, so the tile ships true 3D points the renderer binds
    // zero-copy (no per-point pad). The column is then dropped from properties.
    let elev_col = cfg.point_elevation_column.clone();
    let point_elev: Option<Vec<f64>> =
        if !elev_col.is_empty() && matches!(layer.geometry, GeometryColumn::Point(_)) {
            layer.properties.iter().find_map(|(name, col)| match col {
                PropertyColumn::Numeric(v) if name == &elev_col => {
                    let n = layer.feature_count();
                    let mut out = vec![0.0f64; n];
                    for (i, x) in v.iter().enumerate() {
                        if let Some(val) = x {
                            out[i] = *val;
                        }
                    }
                    Some(out)
                }
                _ => None,
            })
        } else {
            None
        };
    let elev_consumed = point_elev.is_some();

    // Geometry column carries the GeoArrow extension name in field metadata.
    // The precision floor is enforced HERE (where the meters value is
    // consumed), so every entry point — globals, explicit EncoderConfig, a
    // server's per-request config — hits the same guard.
    if let Some(m) = cfg.quantize_coords_m {
        validate_quantize_coords_m(m)?;
    }
    let quant = cfg.quantize_coords_m.and_then(|m| {
        if point_elev.is_some() {
            world_grid_affine_3d(m)
        } else {
            world_grid_affine(m)
        }
    });
    let geom_array =
        build_geometry_array_q(&layer.geometry, quant.as_ref(), point_elev.as_deref())?;
    // Assemble field metadata in a BTreeMap so the key set is emitted in a
    // deterministic (lexicographic) order regardless of insertion order. Arrow
    // ≥59 serializes IPC schema metadata in sorted order, so building from a
    // sorted source makes the raw metadata-region bytes byte-reproducible across
    // runs (guarded by `same_tile_encodes_byte_identically` in
    // reproducible_build.rs) — closing the old arrow-54 HashMap-iteration gap.
    let mut geom_meta = BTreeMap::new();
    geom_meta.insert(
        GEOARROW_EXT_KEY.to_string(),
        layer.geometry.geoarrow_name().to_string(),
    );
    match &quant {
        // A quantized tile's `xy` leaf is i32 grid indices, not Float64 lon/lat,
        // so the GeoArrow CRS doesn't apply — swap it for the reconstruction
        // affine (whose presence is the reader's quantization signal). Field
        // metadata is assembled in a BTreeMap (deterministic key order); Arrow
        // ≥59 serializes IPC schema metadata in sorted order, so the raw
        // metadata-region bytes are byte-reproducible across runs (guarded by
        // the `same_tile_encodes_byte_identically` test in reproducible_build.rs).
        Some(q) => {
            geom_meta.insert(STT_QUANT_META_KEY.to_string(), q.to_json());
        }
        // Advertise the CRS so the tile is self-describing to GeoArrow consumers.
        None => {
            geom_meta.insert(
                GEOARROW_EXT_META_KEY.to_string(),
                GEOARROW_CRS_METADATA.to_string(),
            );
        }
    }
    fields.push(Arc::new(
        Field::new("geometry", geom_array.data_type().clone(), false)
            .with_metadata(geom_meta.into_iter().collect()),
    ));
    columns.push(geom_array);

    // Track per-layer vertex-time encoding so the schema metadata (set
    // below) records the origin/step needed for the u16-delta reader path.
    let mut vertex_time_encoding: Option<(i64, u32)> = None;
    if let Some(vt_col) =
        build_vertex_time_array(&layer.vertex_times, n, cfg.vertex_time_max_step_ms)
    {
        fields.push(Arc::new(Field::new(
            "vertex_time",
            vt_col.array.data_type().clone(),
            true,
        )));
        columns.push(vt_col.array);
        vertex_time_encoding = vt_col.encoding;
    }

    // Optional per-vertex scalar column (e.g. sea-surface temperature),
    // aligned 1:1 with the geometry vertices like `vertex_time`.
    if let Some(vv_array) = build_vertex_value_array(&layer.vertex_values, n) {
        fields.push(Arc::new(Field::new(
            "vertex_value",
            vv_array.data_type().clone(),
            true,
        )));
        columns.push(vv_array);
    }

    // Optional per-vertex × per-bucket value matrix (static-geometry overview
    // animation). Reuses the `vertex_value` List<Float32> encoding — each row
    // is just longer (vertex_count * num_buckets, vertex-major). num_buckets is
    // recovered from the per-feature vertex count and recorded in schema meta.
    let mut vertex_value_buckets: Option<u32> = None;
    if let Some(vm_array) = build_vertex_value_array(&layer.vertex_value_matrix, n) {
        fields.push(Arc::new(Field::new(
            "vertex_value_matrix",
            vm_array.data_type().clone(),
            true,
        )));
        columns.push(vm_array);
        vertex_value_buckets = layer
            .vertex_value_matrix
            .as_ref()
            .and_then(|vm| infer_vertex_value_buckets(vm, &layer.geometry));
    }

    // Pre-baked triangle indices (MLT-style). Only emitted for polygon
    // layers; for any other geometry kind the column is silently dropped so
    // an over-eager builder can't poison a point/line layer with stale data.
    let has_triangles = matches!(layer.geometry, GeometryColumn::Polygon(_))
        && layer
            .triangles
            .as_ref()
            .map(|t| t.iter().any(|f| !f.is_empty()))
            .unwrap_or(false);
    if has_triangles {
        let tri = layer.triangles.as_ref().unwrap();
        // Indices are feature-LOCAL (see the field doc above), so they're
        // almost always well under 65,536 even for large layers. Mirrors
        // build_vertex_time_array's width-selection: scan once, use the
        // narrower UInt16 (half the bytes) when every index fits, UInt32
        // otherwise. The Arrow field type is derived from the array below,
        // so this is fully self-describing — the TS decoder branches on the
        // runtime child-array type exactly like it already does for
        // vertex_time's UInt16-delta vs Int64-absolute split.
        let max_index = tri.iter().flatten().copied().max().unwrap_or(0);
        let array: ArrayRef = if max_index <= u16::MAX as u32 {
            let mut builder = ListBuilder::new(UInt16Builder::new());
            for feature in tri {
                for &idx in feature {
                    builder.values().append_value(idx as u16);
                }
                // Always append a (possibly empty) list — readers expect one
                // entry per feature.
                builder.append(true);
            }
            Arc::new(builder.finish())
        } else {
            let mut builder = ListBuilder::new(UInt32Builder::new());
            for feature in tri {
                for &idx in feature {
                    builder.values().append_value(idx);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        };
        fields.push(Arc::new(Field::new(
            "triangles",
            array.data_type().clone(),
            false,
        )));
        columns.push(array);
    }

    // Every field pushed so far is a reserved column — the v2 CORE batch.
    // Property columns (the v2 PROPS batch) follow.
    let reserved_len = fields.len();

    // Fuse configured scalar columns into GPU-ready interleaved Vector columns
    // (e.g. qx/qy/qz/qw → one FixedSizeList<f32,4>). Runs BEFORE the quantize
    // loop so grouped components are written as the raw vector, not individually
    // quantized. No groups configured ⇒ iterate `layer.properties` with no clone.
    let grouped =
        group_vector_properties(&layer.properties, layer.feature_count(), &cfg.vector_groups);
    let props_iter: &[(String, PropertyColumn)] = grouped.as_deref().unwrap_or(&layer.properties);
    for (name, col) in props_iter {
        // The point-elevation column now lives in the geometry's 3rd coordinate;
        // don't also emit it as a scalar property.
        if elev_consumed && name == &elev_col {
            continue;
        }
        match col {
            PropertyColumn::Numeric(values) => {
                // Opt-in: a numeric property named in the build-global
                // quantization map ships as fixed-point ints + a per-column
                // affine in field metadata (the reader reconstructs Float64).
                // For a LiDAR `z` column this is the single largest size lever
                // after `id` — a raw Float64 elevation barely compresses, while
                // the i16 grid is both smaller and far more compressible.
                let quantized = match cfg.quantize_attrs.get(name).copied().filter(|p| *p > 0.0) {
                    Some(p) => build_quantized_numeric(values, p)?,
                    None => None,
                }
                .or_else(|| {
                    // No explicit precision — fall back to the configured
                    // automatic range-adaptive quantization when enabled.
                    cfg.quantize_attrs_auto
                        .then(|| build_quantized_numeric_auto(values))
                        .flatten()
                });
                match quantized {
                    Some((array, affine_json)) => {
                        let mut m = HashMap::new();
                        m.insert(STT_QUANT_ATTR_META_KEY.to_string(), affine_json);
                        fields.push(Arc::new(
                            Field::new(name, array.data_type().clone(), true).with_metadata(m),
                        ));
                        columns.push(array);
                    }
                    None => {
                        fields.push(Arc::new(Field::new(name, DataType::Float64, true)));
                        columns.push(Arc::new(Float64Array::from(values.clone())));
                    }
                }
            }
            PropertyColumn::Categorical(values) => {
                // Build a Dictionary<UInt16, Utf8>: deduplicate strings once
                // here so the TS reader can lift the dictionary table out of
                // the Arrow batch directly instead of rebuilding it per tile.
                let (indices, categories) = build_dictionary_indices(values)?;
                let key_type = DataType::UInt16;
                let value_type = DataType::Utf8;
                let dict_type = DataType::Dictionary(Box::new(key_type), Box::new(value_type));
                fields.push(Arc::new(Field::new(name, dict_type, true)));

                let value_array: ArrayRef = Arc::new(StringArray::from(
                    categories
                        .iter()
                        .map(|s| Some(s.as_str()))
                        .collect::<Vec<_>>(),
                ));
                let key_array = UInt16Array::from(indices);
                let dict = DictionaryArray::<UInt16Type>::try_new(key_array, value_array)
                    .map_err(|e| Error::Other(format!("dictionary build failed: {e}")))?;
                columns.push(Arc::new(dict));
            }
            PropertyColumn::Vector {
                width,
                elem,
                values,
            } => {
                // Interleaved GPU-ready vector → FixedSizeList<leaf, width>. The
                // child buffer is the flattened row-major run, so the TS decoder
                // hands `child.values.subarray(...)` straight to deck.gl with no
                // per-point re-interleave. Non-null leaf (producer encodes a
                // missing feature as a zero/identity vector).
                let (child, child_dt): (ArrayRef, DataType) = match elem {
                    VectorElem::F32 => (
                        Arc::new(Float32Array::from(values.clone())),
                        DataType::Float32,
                    ),
                    VectorElem::U8 => {
                        let bytes: Vec<u8> = values
                            .iter()
                            .map(|v| v.round().clamp(0.0, 255.0) as u8)
                            .collect();
                        (Arc::new(UInt8Array::from(bytes)), DataType::UInt8)
                    }
                };
                let item_field = Arc::new(Field::new("item", child_dt, false));
                let width_i32 = i32::try_from(*width).map_err(|_| {
                    Error::Other(format!(
                        "vector property '{name}' has width {width}, which exceeds the \
                         FixedSizeList i32 size limit"
                    ))
                })?;
                let fsl = FixedSizeListArray::new(item_field, width_i32, child, None);
                fields.push(Arc::new(Field::new(name, fsl.data_type().clone(), true)));
                columns.push(Arc::new(fsl));
            }
        }
    }

    Ok(LayerParts {
        fields,
        columns,
        reserved_len,
        layer_name: layer.name.clone(),
        geometry_name: layer.geometry.geoarrow_name(),
        // The layer's minimum feature start-time (integer Unix ms), so the TS
        // decoder can skip its client-side min-scan over the whole start-time
        // column and relativize times against this value directly. Mirrors
        // exactly what the decoder computes (the min of the `start_time`
        // column); only emitted when a start-time column is present. See
        // packages/core/src/tile.ts.
        min_start_time: layer.start_times.iter().copied().min(),
        vertex_time_encoding,
        vertex_value_buckets,
        has_triangles,
    })
}

/// Serialize one `(schema, columns)` batch as an Arrow IPC stream.
fn write_ipc_stream(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<Vec<u8>> {
    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Error::Other(format!("failed to build tile RecordBatch: {e}")))?;
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)
            .map_err(|e| Error::Other(format!("Arrow IPC writer init failed: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| Error::Other(format!("Arrow IPC write failed: {e}")))?;
        writer
            .finish()
            .map_err(|e| Error::Other(format!("Arrow IPC finish failed: {e}")))?;
    }
    Ok(buf)
}

/// v1 assembly: one schema carrying EVERY metadata key (dataset-constant AND
/// per-tile-varying), serialized as a single IPC stream — the frozen 0.3.x
/// bytes, guarded by the `tests/v1_golden.rs` fixture.
fn assemble_layer_ipc_v1(parts: LayerParts) -> Result<Vec<u8>> {
    // Schema-level metadata records the layer name and geometry kind so a
    // reader does not have to inspect the geometry column. When the
    // vertex_time column is u16-delta encoded we add `origin_ms` and
    // `step_ms` so the reader can reconstruct absolute timestamps as
    // `origin + delta * step`.
    //
    // Built in a BTreeMap so the key set is assembled in deterministic
    // (lexicographic) order — this encoder contributes no ordering
    // non-determinism. Arrow ≥59 serializes IPC schema metadata in sorted order,
    // so the raw metadata-region bytes of two identical tiles are now identical
    // across runs (unblocking content-addressed pack dedup) — closing the old
    // arrow-54 HashMap-iteration gap.
    let mut schema_meta: BTreeMap<String, String> = BTreeMap::new();
    schema_meta.insert("stt:layer".to_string(), parts.layer_name.clone());
    schema_meta.insert("stt:geometry".to_string(), parts.geometry_name.to_string());
    if let Some(min_start) = parts.min_start_time {
        schema_meta.insert(TIME_OFFSET_MS_KEY.to_string(), min_start.to_string());
    }
    if let Some((origin, step)) = parts.vertex_time_encoding {
        schema_meta.insert(VERTEX_TIME_ORIGIN_KEY.to_string(), origin.to_string());
        schema_meta.insert(VERTEX_TIME_STEP_KEY.to_string(), step.to_string());
    }
    if let Some(buckets) = parts.vertex_value_buckets {
        schema_meta.insert(VERTEX_VALUE_BUCKETS_KEY.to_string(), buckets.to_string());
    }
    if parts.has_triangles {
        schema_meta.insert(TRIANGLES_METADATA_KEY.to_string(), "true".to_string());
    }
    let schema =
        Arc::new(Schema::new(parts.fields).with_metadata(schema_meta.into_iter().collect()));
    write_ipc_stream(schema, parts.columns)
}
// ----------------------------------------------------------------------------
// v2 layer encoding (template extraction + TILE_META + core/props split)
// ----------------------------------------------------------------------------

/// One layer, encoded for the v2 frame: templates split off, tails verbatim,
/// per-tile metadata canonicalized into the TILE_META JSON.
pub(crate) struct EncodedLayerV2 {
    pub(crate) core_template: Vec<u8>,
    pub(crate) core_tail: Vec<u8>,
    /// `(template, tail)` of the PROPS batch; `None` when the layer has no
    /// property columns (`ref_kind_props = 2`).
    pub(crate) props: Option<(Vec<u8>, Vec<u8>)>,
    pub(crate) tile_meta_json: String,
}

/// Locate the end of the leading schema message by walking the Arrow IPC
/// encapsulated framing: `[0xFFFFFFFF][i32 metadata_len][flatbuffer (padded)]`
/// with the schema's `bodyLength == 0` (spike-proven boundary — deterministic,
/// no re-serialization; `metadata_len` already includes the flatbuffer's
/// padding). Everything before the boundary is the template, everything after
/// is the tail (dictionary batches + record batch + EOS).
fn split_ipc_at_schema(ipc: &[u8]) -> Result<usize> {
    if ipc.len() < 8 || ipc[0..4] != [0xFF, 0xFF, 0xFF, 0xFF] {
        return Err(Error::Other(
            "layer IPC stream does not start with an encapsulated message".into(),
        ));
    }
    let meta_len = i32::from_le_bytes(ipc[4..8].try_into().expect("4 bytes"));
    if meta_len <= 0 {
        return Err(Error::Other(
            "layer IPC stream starts with an end-of-stream marker".into(),
        ));
    }
    let boundary = 8usize
        .checked_add(meta_len as usize)
        .filter(|b| *b <= ipc.len())
        .ok_or_else(|| Error::Other("layer IPC schema message overruns the stream".into()))?;
    let msg = root_as_message(&ipc[8..boundary])
        .map_err(|e| Error::Other(format!("layer IPC schema flatbuffer parse failed: {e}")))?;
    if msg.header_type() != MessageHeader::Schema {
        return Err(Error::Other(format!(
            "layer IPC stream must start with a Schema message, got {:?}",
            msg.header_type()
        )));
    }
    if msg.bodyLength() != 0 {
        return Err(Error::Other(
            "layer IPC schema message unexpectedly carries a body".into(),
        ));
    }
    Ok(boundary)
}

/// v2 row order (design §4.2, ★F10): stable-sort rows by `start_time` at
/// ENCODE time — after the tiler assigned feature ids — so ids stay
/// order-independent. Returns the layer unchanged (borrowed) when its rows
/// are already non-decreasing, which also makes the sort a no-op for
/// pre-sorted producers.
pub(crate) fn sort_rows_by_start_time(layer: &ColumnarLayer) -> Cow<'_, ColumnarLayer> {
    if layer.start_times.windows(2).all(|w| w[0] <= w[1]) {
        return Cow::Borrowed(layer);
    }
    let mut idx: Vec<usize> = (0..layer.feature_count()).collect();
    idx.sort_by_key(|&i| layer.start_times[i]); // stable

    // Per-feature Option<Vec<Vec<_>>> columns tolerate inner vecs shorter
    // than the feature count at encode (`vt.get(i)` ⇒ null list); the
    // permutation preserves that semantic via `.get(..).unwrap_or_default()`.
    fn permute_nested<T: Clone>(v: &Option<Vec<Vec<T>>>, idx: &[usize]) -> Option<Vec<Vec<T>>> {
        v.as_ref().map(|v| {
            idx.iter()
                .map(|&i| v.get(i).cloned().unwrap_or_default())
                .collect()
        })
    }

    let geometry = match &layer.geometry {
        GeometryColumn::Point(v) => GeometryColumn::Point(idx.iter().map(|&i| v[i]).collect()),
        GeometryColumn::LineString(v) => {
            GeometryColumn::LineString(idx.iter().map(|&i| v[i].clone()).collect())
        }
        GeometryColumn::Polygon(v) => {
            GeometryColumn::Polygon(idx.iter().map(|&i| v[i].clone()).collect())
        }
    };
    let properties = layer
        .properties
        .iter()
        .map(|(name, col)| {
            let col = match col {
                PropertyColumn::Numeric(v) => {
                    PropertyColumn::Numeric(idx.iter().map(|&i| v[i]).collect())
                }
                PropertyColumn::Categorical(v) => {
                    PropertyColumn::Categorical(idx.iter().map(|&i| v[i].clone()).collect())
                }
                PropertyColumn::Vector {
                    width,
                    elem,
                    values,
                } => PropertyColumn::Vector {
                    width: *width,
                    elem: *elem,
                    values: idx
                        .iter()
                        .flat_map(|&i| values[i * width..(i + 1) * width].iter().copied())
                        .collect(),
                },
            };
            (name.clone(), col)
        })
        .collect();

    Cow::Owned(ColumnarLayer {
        name: layer.name.clone(),
        feature_ids: idx.iter().map(|&i| layer.feature_ids[i]).collect(),
        start_times: idx.iter().map(|&i| layer.start_times[i]).collect(),
        end_times: idx.iter().map(|&i| layer.end_times[i]).collect(),
        geometry,
        vertex_times: permute_nested(&layer.vertex_times, &idx),
        vertex_values: permute_nested(&layer.vertex_values, &idx),
        vertex_value_matrix: permute_nested(&layer.vertex_value_matrix, &idx),
        triangles: permute_nested(&layer.triangles, &idx),
        properties,
    })
}

/// Encode one layer for the v2 frame: sort rows, build the shared parts, move
/// per-tile-varying metadata into TILE_META, split the CORE and PROPS IPC
/// streams at their schema boundaries.
///
/// Metadata placement (design §3.1 audit): per-tile-varying and hoisted into
/// TILE_META are exactly `stt:qa` (property fields) and the schema-level
/// `stt:time_offset_ms` / `stt:vertex_time_origin_ms` / `stt:vertex_time_step_ms`
/// / `stt:vertex_value_buckets`. Dataset-constant and template-resident:
/// `ARROW:extension:name` / `ARROW:extension:metadata` (CRS), `stt:quant`
/// (world-anchored), `stt:layer`, `stt:geometry`, `stt:has_triangles`.
pub(crate) fn encode_layer_v2_parts(
    layer: &ColumnarLayer,
    cfg: &EncoderConfig,
) -> Result<EncodedLayerV2> {
    // Validate BEFORE the row sort: `sort_rows_by_start_time` indexes every
    // column by the feature count, so a length-inconsistent layer would panic
    // there instead of returning the v1 path's descriptive error.
    // (`build_layer_parts` re-validates the sorted layer; the check is pure.)
    layer.validate()?;
    let sorted = sort_rows_by_start_time(layer);
    let parts = build_layer_parts(&sorted, cfg)?;

    // Strip `stt:qa` off the property fields into the TILE_META `qa` map —
    // spike-proven to leave the batch tail bytes byte-identical (only the
    // schema message changes). The affine JSON round-trips exactly
    // (`AttrQuant::to_json` is a pure function of `(o, s)`), so decode
    // re-injects byte-identical field metadata.
    let mut qa: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    let mut props_fields: Vec<Arc<Field>> =
        Vec::with_capacity(parts.fields.len() - parts.reserved_len);
    for field in &parts.fields[parts.reserved_len..] {
        let mut meta = field.metadata().clone();
        if let Some(json) = meta.remove(STT_QUANT_ATTR_META_KEY) {
            let affine = AttrQuant::from_json(&json).ok_or_else(|| {
                Error::Other(format!(
                    "property '{}' carries an unparseable {STT_QUANT_ATTR_META_KEY} affine",
                    field.name()
                ))
            })?;
            qa.insert(field.name().clone(), (affine.o, affine.s));
            props_fields.push(Arc::new(field.as_ref().clone().with_metadata(meta)));
        } else {
            props_fields.push(field.clone());
        }
    }

    let tile_meta = TileMeta {
        qa: (!qa.is_empty()).then_some(qa),
        sorted: Some(true),
        t0: parts.min_start_time,
        vb: parts.vertex_value_buckets,
        vt: parts.vertex_time_encoding,
    };
    let tile_meta_json = serde_json::to_string(&tile_meta)
        .map_err(|e| Error::Other(format!("TILE_META encode failed: {e}")))?;

    // CORE schema: dataset-constant metadata ONLY (the per-tile keys live in
    // TILE_META), so the template bytes are identical across every tile of
    // the layer's shape — the whole point of the hoist.
    let mut core_meta: BTreeMap<String, String> = BTreeMap::new();
    core_meta.insert("stt:layer".to_string(), parts.layer_name.clone());
    core_meta.insert("stt:geometry".to_string(), parts.geometry_name.to_string());
    if parts.has_triangles {
        core_meta.insert(TRIANGLES_METADATA_KEY.to_string(), "true".to_string());
    }
    let core_fields: Vec<Arc<Field>> = parts.fields[..parts.reserved_len].to_vec();
    let core_columns: Vec<ArrayRef> = parts.columns[..parts.reserved_len].to_vec();
    let core_schema =
        Arc::new(Schema::new(core_fields).with_metadata(core_meta.into_iter().collect()));
    let core_ipc = write_ipc_stream(core_schema, core_columns)?;
    let core_boundary = split_ipc_at_schema(&core_ipc)?;
    let core_tail = core_ipc[core_boundary..].to_vec();
    let mut core_template = core_ipc;
    core_template.truncate(core_boundary);

    // PROPS schema: property fields only, no schema-level metadata (every
    // dataset-constant key lives on the CORE template).
    let props = if props_fields.is_empty() {
        None
    } else {
        let props_columns: Vec<ArrayRef> = parts.columns[parts.reserved_len..].to_vec();
        let props_schema = Arc::new(Schema::new(props_fields));
        let props_ipc = write_ipc_stream(props_schema, props_columns)?;
        let boundary = split_ipc_at_schema(&props_ipc)?;
        let tail = props_ipc[boundary..].to_vec();
        let mut template = props_ipc;
        template.truncate(boundary);
        Some((template, tail))
    };

    Ok(EncodedLayerV2 {
        core_template,
        core_tail,
        props,
        tile_meta_json,
    })
}

/// Encode a full tile payload (one or more layers) with the layer frame.
///
/// Always emits the *aligned* frame ([`ALIGNED_FRAME_FLAG`] set): each
/// layer's IPC stream is preceded by zero padding to an 8-byte boundary
/// relative to the payload start, so readers can wrap the stream zero-copy.
/// `ipc_len` records the exact IPC byte length (padding excluded); readers
/// derive the pad from alignment math alone.
pub fn encode_tile(layers: &[ColumnarLayer]) -> Result<Vec<u8>> {
    encode_tile_cfg(layers, &EncoderConfig::from_globals())
}

/// [`encode_tile`] with optional fixed-point coordinate quantization applied to
/// every layer (see [`encode_layer_quantized`]). `quantize_m = None` is
/// byte-identical to [`encode_tile`]; the other encoder settings come from the
/// process-wide globals.
pub fn encode_tile_quantized(layers: &[ColumnarLayer], quantize_m: Option<f64>) -> Result<Vec<u8>> {
    encode_tile_cfg(
        layers,
        &EncoderConfig {
            quantize_coords_m: quantize_m,
            ..EncoderConfig::from_globals()
        },
    )
}

/// [`encode_tile`] with a fully-explicit [`EncoderConfig`] — no process-wide
/// globals are read. The concurrency- and multi-config-safe entry point a
/// dynamic per-request tile server uses so each dataset/request encodes with its
/// own settings without touching shared state.
pub fn encode_tile_with(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    encode_tile_cfg(layers, cfg)
}

/// The single tile-encode implementation, driven entirely by an explicit
/// [`EncoderConfig`]. Every public `encode_tile*` entry point funnels here,
/// then branches ONCE on [`EncoderConfig::format_version`] (design §1 ★F3):
/// everything upstream (column/field build) is shared.
fn encode_tile_cfg(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    match cfg.format_version {
        FORMAT_VERSION_V1 => encode_tile_frame_v1(layers, cfg),
        FORMAT_VERSION_V2 => encode_tile_frame_v2(layers, cfg),
        other => Err(Error::Other(format!(
            "unsupported format version {other} (this writer emits 1 or 2)"
        ))),
    }
}

/// v1 frame layer-count guard: the aligned frame's leading u16 spends bit 15
/// on [`ALIGNED_FRAME_FLAG`], leaving 15 bits for the count — so 0x7FFE is the
/// maximum representable count. 0x7FFF is additionally reserved (its OR with
/// the flag collides with [`FRAME_V2_ESCAPE`]), and any count with bit 15 set
/// (0x8000..) would OR into the flag as a silently-mangled smaller count.
/// `>= 0x7FFF` rejects the collision AND the whole unrepresentable range.
pub(crate) fn v1_layer_count_ok(count: usize) -> bool {
    count < 0x7FFF
}

/// v1 frame assembly — the frozen 0.3.x bytes (`tests/v1_golden.rs`).
fn encode_tile_frame_v1(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    // Cap the count one below the historical 0x7fff limit (see
    // [`v1_layer_count_ok`]). No real tile comes within orders of magnitude
    // of the limit, so only the error bound moves.
    if !v1_layer_count_ok(layers.len()) {
        return Err(Error::Other(format!(
            "tile has {} layers, exceeds the {} frame limit",
            layers.len(),
            ALIGNED_FRAME_FLAG - 2
        )));
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(layers.len() as u16 | ALIGNED_FRAME_FLAG).to_le_bytes());
    for layer in layers {
        let name = layer.name.as_bytes();
        if name.len() > u16::MAX as usize {
            return Err(Error::Other("layer name too long".into()));
        }
        let ipc = encode_layer_cfg(layer, cfg)?;
        let ipc_len = u32::try_from(ipc.len()).map_err(|_| {
            Error::Other(format!(
                "layer '{}' IPC stream is {} bytes, exceeding the frame's u32 length field",
                layer.name,
                ipc.len()
            ))
        })?;
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&ipc_len.to_le_bytes());
        let pad = (FRAME_ALIGN - out.len() % FRAME_ALIGN) % FRAME_ALIGN;
        out.extend_from_slice(&[0u8; FRAME_ALIGN][..pad]);
        out.extend_from_slice(&ipc);
    }
    Ok(out)
}

/// Zero-pad `out` to the next [`FRAME_ALIGN`] boundary (v2 derived padding —
/// like v1, the pad length is never stored).
fn pad_to_frame_align(out: &mut Vec<u8>) {
    let pad = (FRAME_ALIGN - out.len() % FRAME_ALIGN) % FRAME_ALIGN;
    out.extend_from_slice(&[0u8; FRAME_ALIGN][..pad]);
}

/// v2 frame assembly (module docs / design §4). With a
/// [`TemplateCollector`] configured, schemas are recorded there and frames
/// carry 16-byte hash references (the packed-dataset mode); without one the
/// frame is self-contained via `INLINE_SCHEMA_*` sections.
fn encode_tile_frame_v2(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    if layers.len() > u16::MAX as usize {
        return Err(Error::Other(format!(
            "tile has {} layers, exceeds the {} frame limit",
            layers.len(),
            u16::MAX
        )));
    }
    let collector = cfg.template_collector.as_deref();
    let mut out = Vec::new();
    out.extend_from_slice(&FRAME_V2_ESCAPE.to_le_bytes());
    out.push(FRAME_V2_VERSION);
    out.push(0); // flags — reserved, MUST be 0
    out.extend_from_slice(&(layers.len() as u16).to_le_bytes());
    for layer in layers {
        let name = layer.name.as_bytes();
        if name.len() > u16::MAX as usize {
            return Err(Error::Other("layer name too long".into()));
        }
        let enc = encode_layer_v2_parts(layer, cfg)?;
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);

        // Schema references. Hash mode records the template with the
        // collector (content-addressed, so parallel encode order is
        // irrelevant); inline mode ships it as a section instead.
        match collector {
            Some(c) => {
                out.push(REF_KIND_TEMPLATE_HASH);
                out.extend_from_slice(&c.record(&enc.core_template));
            }
            None => out.push(REF_KIND_INLINE),
        }
        match (&enc.props, collector) {
            (None, _) => out.push(REF_KIND_NO_PROPS),
            (Some((template, _)), Some(c)) => {
                out.push(REF_KIND_TEMPLATE_HASH);
                out.extend_from_slice(&c.record(template));
            }
            (Some(_), None) => out.push(REF_KIND_INLINE),
        }

        // Sections in ascending tag order; TOC lengths are the exact at-rest
        // byte counts (padding derived, never stored).
        let mut sections: Vec<(u8, &[u8])> = Vec::new();
        if collector.is_none() {
            sections.push((SECTION_INLINE_SCHEMA_CORE, &enc.core_template));
        }
        sections.push((SECTION_TILE_META, enc.tile_meta_json.as_bytes()));
        sections.push((SECTION_CORE_BATCH, &enc.core_tail));
        if let Some((template, tail)) = &enc.props {
            if collector.is_none() {
                sections.push((SECTION_INLINE_SCHEMA_PROPS, template));
            }
            sections.push((SECTION_PROPS_BATCH, tail));
        }
        out.push(sections.len() as u8);
        for (tag, bytes) in &sections {
            out.push(*tag);
            let len = u32::try_from(bytes.len()).map_err(|_| {
                Error::Other(format!(
                    "layer '{}' section 0x{tag:02x} is {} bytes, exceeding the TOC's u32 \
                     length field",
                    layer.name,
                    bytes.len()
                ))
            })?;
            out.extend_from_slice(&len.to_le_bytes());
        }
        pad_to_frame_align(&mut out);
        for (_, bytes) in &sections {
            out.extend_from_slice(bytes);
            pad_to_frame_align(&mut out);
        }
    }
    Ok(out)
}
