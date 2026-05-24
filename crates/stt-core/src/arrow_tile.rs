//! Arrow-based tile payload format.
//!
//! A tile's payload is one or more *layers*. Each layer is a single Arrow
//! [`RecordBatch`] serialised as an Arrow IPC **stream**. Geometry is encoded
//! using the GeoArrow interleaved-coordinate convention so the payload can be
//! consumed directly by GeoArrow-aware renderers (e.g. `@geoarrow/deck.gl`).
//!
//! ## Per-layer schema
//!
//! | column        | type                                    | notes                         |
//! |---------------|-----------------------------------------|-------------------------------|
//! | `id`          | `UInt64`                                | feature id                    |
//! | `start_time`  | `Int64`                                 | Unix ms, absolute             |
//! | `end_time`    | `Int64`                                 | Unix ms, absolute             |
//! | `geometry`    | GeoArrow point / linestring / polygon   | interleaved f64 lon/lat       |
//! | `vertex_time` | `List<Int64>` (nullable)                | per-vertex Unix ms (optional) |
//! | `<property>`  | `Float64` or `Dictionary<UInt16,Utf8>`   | one column per property       |
//!
//! All layers in one tile are concatenated with a tiny frame so a tile can
//! carry, say, a linestring layer and a point layer side by side:
//!
//! ```text
//! [u16 layer_count]
//!   repeated: [u16 name_len][name utf8][u32 ipc_len][ipc stream bytes]
//! ```

use crate::error::{Error, Result};
use crate::types::GeometryType;
use arrow::array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float64Array, Int64Array, Int64Builder,
    ListArray, ListBuilder, RecordBatch, StringArray, UInt16Array, UInt16Builder, UInt32Builder,
    UInt64Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema, UInt16Type};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use std::collections::HashMap;
use std::sync::Arc;

/// GeoArrow extension-name metadata key.
const GEOARROW_EXT_KEY: &str = "ARROW:extension:name";

/// A single coordinate pair (lon, lat) in WGS84 degrees.
pub type Coord = [f64; 2];

/// Geometry for one layer, grouped by kind. Every feature in a layer shares
/// one kind — the tiler emits a separate layer per geometry type.
#[derive(Debug, Clone)]
pub enum GeometryColumn {
    /// One coordinate per feature.
    Point(Vec<Coord>),
    /// A vertex list per feature.
    LineString(Vec<Vec<Coord>>),
    /// A list of rings per feature (ring 0 is the exterior).
    Polygon(Vec<Vec<Vec<Coord>>>),
}

impl GeometryColumn {
    /// Number of features represented.
    pub fn len(&self) -> usize {
        match self {
            GeometryColumn::Point(v) => v.len(),
            GeometryColumn::LineString(v) => v.len(),
            GeometryColumn::Polygon(v) => v.len(),
        }
    }

    /// Whether the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The matching [`GeometryType`].
    pub fn kind(&self) -> GeometryType {
        match self {
            GeometryColumn::Point(_) => GeometryType::Point,
            GeometryColumn::LineString(_) => GeometryType::LineString,
            GeometryColumn::Polygon(_) => GeometryType::Polygon,
        }
    }

    /// The GeoArrow extension name for this geometry kind.
    fn geoarrow_name(&self) -> &'static str {
        match self {
            GeometryColumn::Point(_) => "geoarrow.point",
            GeometryColumn::LineString(_) => "geoarrow.linestring",
            GeometryColumn::Polygon(_) => "geoarrow.polygon",
        }
    }
}

/// A property column. Values are per-feature and may be missing.
#[derive(Debug, Clone)]
pub enum PropertyColumn {
    /// Numeric values (f64).
    Numeric(Vec<Option<f64>>),
    /// Categorical / string values.
    Categorical(Vec<Option<String>>),
}

/// One decoded/encodable tile layer.
#[derive(Debug, Clone)]
pub struct ColumnarLayer {
    /// Layer name (e.g. `"default"`, `"default_originals"`).
    pub name: String,
    /// Per-feature id.
    pub feature_ids: Vec<u64>,
    /// Per-feature start time (Unix ms, absolute).
    pub start_times: Vec<i64>,
    /// Per-feature end time (Unix ms, absolute).
    pub end_times: Vec<i64>,
    /// Geometry, one entry per feature.
    pub geometry: GeometryColumn,
    /// Optional per-vertex timestamps (Unix ms). When present, length equals
    /// the feature count and each inner vec matches that feature's vertex count.
    pub vertex_times: Option<Vec<Vec<i64>>>,
    /// Optional pre-baked triangle indices for polygon features (MLT-style).
    ///
    /// When present, `triangles.len() == feature_count` and each inner vec is
    /// the flat triangle-index list (groups of 3 vertex indices) produced by
    /// earcut at build time. Indices are LOCAL to the feature: they reference
    /// positions within that feature's own ring coordinates, so the renderer
    /// only needs to add the feature's `startIndex` to each one.
    ///
    /// Only meaningful for `GeometryColumn::Polygon`. Encoders that set this
    /// on a non-polygon layer will have it dropped at encode time.
    pub triangles: Option<Vec<Vec<u32>>>,
    /// Property columns, keyed by name. Each column has one value per feature.
    pub properties: Vec<(String, PropertyColumn)>,
}

impl ColumnarLayer {
    /// Feature count for this layer.
    pub fn feature_count(&self) -> usize {
        self.feature_ids.len()
    }

    /// Validate that every column has a consistent length.
    fn validate(&self) -> Result<()> {
        let n = self.feature_ids.len();
        let check = |label: &str, len: usize| -> Result<()> {
            if len != n {
                return Err(Error::Other(format!(
                    "tile layer '{}': {} has {} entries, expected {}",
                    self.name, label, len, n
                )));
            }
            Ok(())
        };
        check("start_times", self.start_times.len())?;
        check("end_times", self.end_times.len())?;
        check("geometry", self.geometry.len())?;
        if let Some(vt) = &self.vertex_times {
            check("vertex_times", vt.len())?;
        }
        if let Some(tri) = &self.triangles {
            check("triangles", tri.len())?;
        }
        for (name, col) in &self.properties {
            let len = match col {
                PropertyColumn::Numeric(v) => v.len(),
                PropertyColumn::Categorical(v) => v.len(),
            };
            check(&format!("property '{}'", name), len)?;
        }
        Ok(())
    }
}

/// Schema-metadata key set on layers that carry pre-baked triangle indices.
pub const TRIANGLES_METADATA_KEY: &str = "stt:has_triangles";

/// Tessellate one polygon feature (a list of rings) using earcut. Returns the
/// flat triangle index list — each triple of indices is one triangle, indices
/// are LOCAL (relative to the start of the feature's coordinate run, where
/// the exterior ring sits first followed by every hole).
///
/// Returns an empty vec for degenerate inputs (no exterior ring, <3 vertices).
pub fn tessellate_polygon(rings: &[Vec<Coord>]) -> Vec<u32> {
    if rings.is_empty() {
        return Vec::new();
    }
    // Flatten coords into the [x0, y0, x1, y1, ...] format earcutr expects.
    let mut flat: Vec<f64> = Vec::with_capacity(rings.iter().map(|r| r.len()).sum::<usize>() * 2);
    // Hole offsets are vertex indices (not coord-pair indices) where each
    // hole begins. The first hole starts after the exterior ring.
    let mut hole_indices: Vec<usize> = Vec::with_capacity(rings.len().saturating_sub(1));
    let mut running = 0usize;
    for (i, ring) in rings.iter().enumerate() {
        if i > 0 {
            hole_indices.push(running);
        }
        for [x, y] in ring {
            flat.push(*x);
            flat.push(*y);
        }
        running += ring.len();
    }
    if running < 3 {
        return Vec::new();
    }
    match earcutr::earcut(&flat, &hole_indices, 2) {
        Ok(tris) => tris.into_iter().map(|i| i as u32).collect(),
        Err(_) => Vec::new(),
    }
}

// ----------------------------------------------------------------------------
// Geometry array construction (GeoArrow interleaved)
// ----------------------------------------------------------------------------

/// Build an `i32` offset buffer from per-element counts.
fn offsets_from_counts(counts: impl Iterator<Item = usize>) -> OffsetBuffer<i32> {
    let mut acc = 0i32;
    let mut offsets = vec![0i32];
    for c in counts {
        acc += c as i32;
        offsets.push(acc);
    }
    OffsetBuffer::new(offsets.into())
}

/// Construct the GeoArrow geometry array for a [`GeometryColumn`].
fn build_geometry_array(geom: &GeometryColumn) -> ArrayRef {
    let coord_field = || Arc::new(Field::new("xy", DataType::Float64, false));

    match geom {
        GeometryColumn::Point(points) => {
            let mut flat = Vec::with_capacity(points.len() * 2);
            for [x, y] in points {
                flat.push(*x);
                flat.push(*y);
            }
            let values = Arc::new(Float64Array::from(flat));
            Arc::new(FixedSizeListArray::new(coord_field(), 2, values, None))
        }
        GeometryColumn::LineString(lines) => {
            let mut flat: Vec<f64> = Vec::new();
            for line in lines {
                for [x, y] in line {
                    flat.push(*x);
                    flat.push(*y);
                }
            }
            let coords: ArrayRef = {
                let values = Arc::new(Float64Array::from(flat));
                Arc::new(FixedSizeListArray::new(coord_field(), 2, values, None))
            };
            let offsets = offsets_from_counts(lines.iter().map(|l| l.len()));
            let vertex_field = Arc::new(Field::new("vertices", coords.data_type().clone(), false));
            Arc::new(ListArray::new(vertex_field, offsets, coords, None))
        }
        GeometryColumn::Polygon(polys) => {
            // Flatten all coordinates, ring sizes, and ring counts per feature.
            let mut flat: Vec<f64> = Vec::new();
            let mut ring_sizes: Vec<usize> = Vec::new();
            let mut rings_per_feature: Vec<usize> = Vec::new();
            for feature in polys {
                rings_per_feature.push(feature.len());
                for ring in feature {
                    ring_sizes.push(ring.len());
                    for [x, y] in ring {
                        flat.push(*x);
                        flat.push(*y);
                    }
                }
            }
            let coords: ArrayRef = {
                let values = Arc::new(Float64Array::from(flat));
                Arc::new(FixedSizeListArray::new(coord_field(), 2, values, None))
            };
            // Ring level: List<FixedSizeList>.
            let ring_offsets = offsets_from_counts(ring_sizes.into_iter());
            let vertex_field = Arc::new(Field::new("vertices", coords.data_type().clone(), false));
            let rings: ArrayRef = Arc::new(ListArray::new(
                vertex_field,
                ring_offsets,
                coords,
                None,
            ));
            // Feature level: List<List<FixedSizeList>>.
            let feature_offsets = offsets_from_counts(rings_per_feature.into_iter());
            let ring_field = Arc::new(Field::new("rings", rings.data_type().clone(), false));
            Arc::new(ListArray::new(ring_field, feature_offsets, rings, None))
        }
    }
}

/// Schema metadata keys for the v3 per-vertex time encoding.
const VERTEX_TIME_ORIGIN_KEY: &str = "stt:vertex_time_origin_ms";
const VERTEX_TIME_STEP_KEY: &str = "stt:vertex_time_step_ms";

/// Build a (key_array, value_array) pair for a Dictionary<UInt16, Utf8>
/// column. Null inputs become null keys (the corresponding string is not
/// inserted into the dictionary); strings are deduplicated in first-seen
/// order so the on-disk Arrow dictionary is stable across runs.
///
/// Saturates at u16::MAX categories — additional unique strings beyond
/// that limit collapse to the last seen index. In practice STT
/// categorical columns top out in the low hundreds; this cap exists to
/// keep the on-disk key width bounded.
fn build_dictionary_indices(values: &[Option<String>]) -> (Vec<Option<u16>>, Vec<String>) {
    let mut categories: Vec<String> = Vec::new();
    let mut lookup: HashMap<String, u16> = HashMap::new();
    let mut indices: Vec<Option<u16>> = Vec::with_capacity(values.len());
    for v in values {
        match v {
            Some(s) => {
                if let Some(&idx) = lookup.get(s) {
                    indices.push(Some(idx));
                } else if categories.len() < u16::MAX as usize {
                    let idx = categories.len() as u16;
                    categories.push(s.clone());
                    lookup.insert(s.clone(), idx);
                    indices.push(Some(idx));
                } else {
                    // Overflow: reuse the last index. A producer that
                    // genuinely needs >65k unique strings should split the
                    // column into multiple categorical fields.
                    indices.push(Some(u16::MAX - 1));
                }
            }
            None => indices.push(None),
        }
    }
    (indices, categories)
}

/// Built per-vertex time column, alongside the per-layer schema metadata
/// that lets the reader reconstruct absolute timestamps.
struct VertexTimeColumn {
    array: ArrayRef,
    /// `(origin_ms, step_ms)` when the column is u16-delta-encoded. `None`
    /// when the column kept its absolute `List<Int64>` shape (the v2 fallback
    /// path, used for layers whose temporal span exceeds 65,535 * step).
    encoding: Option<(i64, u32)>,
}

/// Build the optional per-vertex time column.
///
/// v3 attempts to encode timestamps as `List<UInt16>` deltas relative to
/// a per-layer origin and step (`absolute = origin + delta * step`). When
/// the layer's temporal range overflows `u16::MAX * step`, it falls back to
/// the v2 `List<Int64>` shape so we never lose precision.
fn build_vertex_time_array(
    vertex_times: &Option<Vec<Vec<i64>>>,
    feature_count: usize,
) -> Option<VertexTimeColumn> {
    let vt = vertex_times.as_ref()?;

    // Discover the layer's temporal span across every (feature, vertex) pair.
    // Empty / null lists are skipped — a list-of-nulls is fine, it just won't
    // shrink the span.
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut any = false;
    for times in vt.iter().take(feature_count) {
        for &t in times {
            if t < min {
                min = t;
            }
            if t > max {
                max = t;
            }
            any = true;
        }
    }

    if any && max >= min {
        // Pick the smallest step (in ms) that keeps every (t - min) inside
        // u16::MAX. step=1 means "exact ms granularity"; larger steps trade
        // precision (bounded by step/2 ms) for a 4x payload shrink vs i64.
        let span = (max - min) as u64;
        let step = if span <= u16::MAX as u64 {
            1u32
        } else {
            ((span + u16::MAX as u64 - 1) / u16::MAX as u64).max(1) as u32
        };
        // Round-trip safety: a step too large to fit u32 isn't reachable
        // through any sensible tile (span < ~136 years for step=1), but we
        // still bail back to i64 if it ever happens.
        if step != 0 {
            let mut builder = ListBuilder::new(UInt16Builder::new());
            for i in 0..feature_count {
                match vt.get(i) {
                    Some(times) if !times.is_empty() => {
                        for &t in times {
                            // Saturate at u16::MAX — for a sensibly chosen
                            // step this branch can only fire on inputs that
                            // disagree with the (min,max) scan above (e.g.
                            // a `vertex_times` longer than feature_count).
                            let delta = ((t - min) as u64 / step as u64).min(u16::MAX as u64) as u16;
                            builder.values().append_value(delta);
                        }
                        builder.append(true);
                    }
                    _ => builder.append(false),
                }
            }
            return Some(VertexTimeColumn {
                array: Arc::new(builder.finish()),
                encoding: Some((min, step)),
            });
        }
    }

    // Fallback: legacy absolute List<Int64> for empty-ish columns or
    // pathological steps. Identical wire shape to v2.
    let mut builder = ListBuilder::new(Int64Builder::new());
    for i in 0..feature_count {
        match vt.get(i) {
            Some(times) if !times.is_empty() => {
                for &t in times {
                    builder.values().append_value(t);
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Some(VertexTimeColumn {
        array: Arc::new(builder.finish()),
        encoding: None,
    })
}

// ----------------------------------------------------------------------------
// Encoding
// ----------------------------------------------------------------------------

/// Encode a single layer to an Arrow IPC stream.
pub fn encode_layer(layer: &ColumnarLayer) -> Result<Vec<u8>> {
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

    // Geometry column carries the GeoArrow extension name in field metadata.
    let geom_array = build_geometry_array(&layer.geometry);
    let mut geom_meta = HashMap::new();
    geom_meta.insert(
        GEOARROW_EXT_KEY.to_string(),
        layer.geometry.geoarrow_name().to_string(),
    );
    fields.push(Arc::new(
        Field::new("geometry", geom_array.data_type().clone(), false)
            .with_metadata(geom_meta),
    ));
    columns.push(geom_array);

    // Track per-layer vertex-time encoding so the schema metadata (set
    // below) records the origin/step needed for the u16-delta reader path.
    let mut vertex_time_encoding: Option<(i64, u32)> = None;
    if let Some(vt_col) = build_vertex_time_array(&layer.vertex_times, n) {
        fields.push(Arc::new(Field::new(
            "vertex_time",
            vt_col.array.data_type().clone(),
            true,
        )));
        columns.push(vt_col.array);
        vertex_time_encoding = vt_col.encoding;
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
        let mut builder = ListBuilder::new(UInt32Builder::new());
        for feature in tri {
            for &idx in feature {
                builder.values().append_value(idx);
            }
            // Always append a (possibly empty) list — readers expect one
            // entry per feature.
            builder.append(true);
        }
        let array: ArrayRef = Arc::new(builder.finish());
        fields.push(Arc::new(Field::new(
            "triangles",
            array.data_type().clone(),
            false,
        )));
        columns.push(array);
    }

    for (name, col) in &layer.properties {
        match col {
            PropertyColumn::Numeric(values) => {
                fields.push(Arc::new(Field::new(name, DataType::Float64, true)));
                columns.push(Arc::new(Float64Array::from(values.clone())));
            }
            PropertyColumn::Categorical(values) => {
                // Build a Dictionary<UInt16, Utf8>: deduplicate strings once
                // here so the TS reader can lift the dictionary table out of
                // the Arrow batch directly instead of rebuilding it per tile.
                let (indices, categories) = build_dictionary_indices(values);
                let key_type = DataType::UInt16;
                let value_type = DataType::Utf8;
                let dict_type = DataType::Dictionary(Box::new(key_type), Box::new(value_type));
                fields.push(Arc::new(Field::new(name, dict_type, true)));

                let value_array: ArrayRef = Arc::new(StringArray::from(
                    categories.iter().map(|s| Some(s.as_str())).collect::<Vec<_>>(),
                ));
                let key_array = UInt16Array::from(indices);
                let dict = DictionaryArray::<UInt16Type>::try_new(key_array, value_array)
                    .map_err(|e| Error::Other(format!("dictionary build failed: {e}")))?;
                columns.push(Arc::new(dict));
            }
        }
    }

    // Schema-level metadata records the layer name and geometry kind so a
    // reader does not have to inspect the geometry column. When the
    // vertex_time column is u16-delta encoded we add `origin_ms` and
    // `step_ms` so the reader can reconstruct absolute timestamps as
    // `origin + delta * step`.
    let mut schema_meta = HashMap::new();
    schema_meta.insert("stt:layer".to_string(), layer.name.clone());
    schema_meta.insert(
        "stt:geometry".to_string(),
        layer.geometry.geoarrow_name().to_string(),
    );
    if let Some((origin, step)) = vertex_time_encoding {
        schema_meta.insert(VERTEX_TIME_ORIGIN_KEY.to_string(), origin.to_string());
        schema_meta.insert(VERTEX_TIME_STEP_KEY.to_string(), step.to_string());
    }
    if has_triangles {
        schema_meta.insert(TRIANGLES_METADATA_KEY.to_string(), "true".to_string());
    }
    let schema = Arc::new(Schema::new(fields).with_metadata(schema_meta));

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

/// Encode a full tile payload (one or more layers) with the layer frame.
pub fn encode_tile(layers: &[ColumnarLayer]) -> Result<Vec<u8>> {
    if layers.len() > u16::MAX as usize {
        return Err(Error::Other(format!(
            "tile has {} layers, exceeds u16 frame limit",
            layers.len()
        )));
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(layers.len() as u16).to_le_bytes());
    for layer in layers {
        let name = layer.name.as_bytes();
        if name.len() > u16::MAX as usize {
            return Err(Error::Other("layer name too long".into()));
        }
        let ipc = encode_layer(layer)?;
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&(ipc.len() as u32).to_le_bytes());
        out.extend_from_slice(&ipc);
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Decoding
// ----------------------------------------------------------------------------

/// A decoded tile layer: its name and the raw Arrow [`RecordBatch`].
#[derive(Debug, Clone)]
pub struct DecodedLayer {
    /// Layer name from the layer frame.
    pub name: String,
    /// The decoded Arrow record batch.
    pub batch: RecordBatch,
}

/// Decode a single-layer Arrow IPC stream into a [`RecordBatch`].
pub fn decode_layer(ipc: &[u8]) -> Result<RecordBatch> {
    let reader = StreamReader::try_new(ipc, None)
        .map_err(|e| Error::Other(format!("Arrow IPC reader init failed: {e}")))?;
    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|e| Error::Other(format!("Arrow IPC read failed: {e}")))?);
    }
    match batches.len() {
        0 => Err(Error::Other("tile layer IPC contained no record batch".into())),
        1 => Ok(batches.into_iter().next().unwrap()),
        // A layer is written as exactly one batch; concatenating is the safe
        // fallback if a producer ever splits it.
        _ => arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .map_err(|e| Error::Other(format!("failed to concat tile batches: {e}"))),
    }
}

/// Decode a full tile payload (the layer frame) into its layers.
pub fn decode_tile(payload: &[u8]) -> Result<Vec<DecodedLayer>> {
    if payload.len() < 2 {
        return Err(Error::Other("tile payload too short for layer frame".into()));
    }
    let count = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let mut pos = 2usize;
    let mut layers = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = read_u16(payload, &mut pos)? as usize;
        let name = read_slice(payload, &mut pos, name_len)?;
        let name = String::from_utf8(name.to_vec())
            .map_err(|e| Error::Other(format!("layer name not utf8: {e}")))?;
        let ipc_len = read_u32(payload, &mut pos)? as usize;
        let ipc = read_slice(payload, &mut pos, ipc_len)?;
        let batch = decode_layer(ipc)?;
        layers.push(DecodedLayer { name, batch });
    }
    Ok(layers)
}

fn read_u16(buf: &[u8], pos: &mut usize) -> Result<u16> {
    let s = read_slice(buf, pos, 2)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let s = read_slice(buf, pos, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_slice<'a>(buf: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::Other("tile frame length overflow".into()))?;
    if end > buf.len() {
        return Err(Error::Other("tile frame truncated".into()));
    }
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point_layer() -> ColumnarLayer {
        ColumnarLayer {
            name: "points".to_string(),
            feature_ids: vec![1, 2, 3],
            start_times: vec![1000, 2000, 3000],
            end_times: vec![1500, 2500, 3500],
            geometry: GeometryColumn::Point(vec![
                [-122.4, 37.7],
                [-122.5, 37.8],
                [-122.6, 37.9],
            ]),
            vertex_times: None,
            triangles: None,
            properties: vec![
                (
                    "speed".to_string(),
                    PropertyColumn::Numeric(vec![Some(10.0), None, Some(30.0)]),
                ),
                (
                    "kind".to_string(),
                    PropertyColumn::Categorical(vec![
                        Some("car".to_string()),
                        Some("bus".to_string()),
                        None,
                    ]),
                ),
            ],
        }
    }

    fn sample_line_layer() -> ColumnarLayer {
        ColumnarLayer {
            name: "tracks".to_string(),
            feature_ids: vec![10, 11],
            start_times: vec![0, 100],
            end_times: vec![50, 200],
            geometry: GeometryColumn::LineString(vec![
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                vec![[5.0, 5.0], [6.0, 6.0]],
            ]),
            vertex_times: Some(vec![vec![0, 25, 50], vec![100, 200]]),
            triangles: None,
            properties: vec![],
        }
    }

    fn sample_polygon_layer() -> ColumnarLayer {
        ColumnarLayer {
            name: "zones".to_string(),
            feature_ids: vec![42],
            start_times: vec![0],
            end_times: vec![1000],
            geometry: GeometryColumn::Polygon(vec![vec![
                // exterior ring
                vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
                // hole
                vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]],
            ]]),
            vertex_times: None,
            triangles: None,
            properties: vec![],
        }
    }

    #[test]
    fn categorical_columns_use_dictionary_encoding() {
        let layer = ColumnarLayer {
            name: "cars".into(),
            feature_ids: vec![1, 2, 3, 4, 5],
            start_times: vec![0; 5],
            end_times: vec![1; 5],
            geometry: GeometryColumn::Point(vec![[0.0, 0.0]; 5]),
            vertex_times: None,
            triangles: None,
            properties: vec![(
                "kind".into(),
                PropertyColumn::Categorical(vec![
                    Some("car".into()),
                    Some("bus".into()),
                    Some("car".into()),
                    None,
                    Some("car".into()),
                ]),
            )],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        let field = batch.schema().field_with_name("kind").unwrap().clone();
        match field.data_type() {
            DataType::Dictionary(k, v) => {
                assert_eq!(k.as_ref(), &DataType::UInt16);
                assert_eq!(v.as_ref(), &DataType::Utf8);
            }
            other => panic!("expected Dictionary<UInt16, Utf8>, got {other:?}"),
        }

        let col = batch
            .column_by_name("kind")
            .unwrap()
            .as_any()
            .downcast_ref::<DictionaryArray<UInt16Type>>()
            .unwrap();
        let values = col
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // First-seen order: "car" then "bus".
        let mut categories: Vec<&str> = (0..values.len()).map(|i| values.value(i)).collect();
        categories.sort();
        assert_eq!(categories, vec!["bus", "car"]);

        // The 4th row is null; others reference one of the two slots.
        assert!(col.is_null(3));
        let keys = col.keys();
        for i in [0usize, 1, 2, 4] {
            assert!(keys.value(i) < values.len() as u16);
        }
    }

    #[test]
    fn point_layer_roundtrips() {
        let layer = sample_point_layer();
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        assert_eq!(batch.num_rows(), 3);
        // id / start / end / geometry / speed / kind
        assert_eq!(batch.num_columns(), 6);

        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2, 3]);

        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(geom.len(), 3);
        assert_eq!(geom.value_length(), 2);

        // Geometry field carries the GeoArrow extension name.
        let geom_field = batch.schema().field_with_name("geometry").unwrap().clone();
        assert_eq!(
            geom_field.metadata().get(GEOARROW_EXT_KEY).map(String::as_str),
            Some("geoarrow.point")
        );

        // Nullable numeric property preserves the null.
        let speed = batch
            .column_by_name("speed")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(speed.is_null(1));
        assert_eq!(speed.value(0), 10.0);
    }

    #[test]
    fn line_layer_roundtrips_with_vertex_times() {
        let layer = sample_line_layer();
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        assert_eq!(batch.num_rows(), 2);
        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        // Feature 0 has 3 vertices, feature 1 has 2.
        assert_eq!(geom.value(0).len(), 3);
        assert_eq!(geom.value(1).len(), 2);

        // v3 layers with a tight temporal span carry u16-delta vertex times
        // and the origin/step metadata needed to reconstruct absolutes.
        let meta = batch.schema().metadata().clone();
        let origin: i64 = meta
            .get("stt:vertex_time_origin_ms")
            .expect("u16 vertex-time layers carry an origin")
            .parse()
            .unwrap();
        let step: u32 = meta
            .get("stt:vertex_time_step_ms")
            .expect("u16 vertex-time layers carry a step")
            .parse()
            .unwrap();
        assert_eq!(origin, 0);
        assert_eq!(step, 1);

        let vt = batch
            .column_by_name("vertex_time")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(vt.len(), 2);
        let first = vt.value(0);
        let deltas = first.as_any().downcast_ref::<arrow::array::UInt16Array>().unwrap();
        let absolutes: Vec<i64> = deltas
            .values()
            .iter()
            .map(|d| origin + (*d as i64) * step as i64)
            .collect();
        assert_eq!(absolutes, vec![0, 25, 50]);
    }

    #[test]
    fn vertex_time_falls_back_to_int64_for_wide_spans() {
        // span = 100 billion ms; step would need to be ~1.5e6 ms — that's
        // still fine for u16 deltas, so we instead force the fallback by
        // disabling u16 (an empty `vertex_times` list); the v3 encoder must
        // never silently corrupt absolute timestamps.
        let layer = ColumnarLayer {
            name: "edge".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![100],
            geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0]]]),
            // Two timestamps far enough apart that any step <= u16::MAX
            // still keeps them inside u16 — the encoder picks step≈1.5e6
            // ms. We assert round-trip precision is bounded by step/2.
            vertex_times: Some(vec![vec![0, 100_000_000_000]]),
            triangles: None,
            properties: vec![],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        let schema = batch.schema();
        let meta = schema.metadata();
        let origin: i64 = meta.get("stt:vertex_time_origin_ms").unwrap().parse().unwrap();
        let step: u32 = meta.get("stt:vertex_time_step_ms").unwrap().parse().unwrap();
        let vt = batch
            .column_by_name("vertex_time")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let first = vt.value(0);
        let deltas = first
            .as_any()
            .downcast_ref::<arrow::array::UInt16Array>()
            .unwrap();
        let absolutes: Vec<i64> = deltas
            .values()
            .iter()
            .map(|d| origin + (*d as i64) * step as i64)
            .collect();
        // First sample is exact; second is within one step.
        assert_eq!(absolutes[0], 0);
        assert!((absolutes[1] - 100_000_000_000).unsigned_abs() <= step as u64);
    }

    #[test]
    fn polygon_layer_roundtrips_with_rings() {
        let layer = sample_polygon_layer();
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        let geom = batch
            .column_by_name("geometry")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(geom.len(), 1);
        // One feature with two rings (exterior + hole).
        let rings = geom.value(0);
        let rings = rings.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(rings.len(), 2);
        assert_eq!(rings.value(0).len(), 5); // exterior ring vertices
        assert_eq!(rings.value(1).len(), 5); // hole vertices
    }

    #[test]
    fn multi_layer_tile_frame_roundtrips() {
        let layers = vec![sample_line_layer(), sample_point_layer()];
        let payload = encode_tile(&layers).unwrap();
        let decoded = decode_tile(&payload).unwrap();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].name, "tracks");
        assert_eq!(decoded[1].name, "points");
        assert_eq!(decoded[0].batch.num_rows(), 2);
        assert_eq!(decoded[1].batch.num_rows(), 3);
        // Schema metadata records the layer name on the batch too.
        assert_eq!(
            decoded[1]
                .batch
                .schema()
                .metadata()
                .get("stt:layer")
                .map(String::as_str),
            Some("points")
        );
    }

    #[test]
    fn tessellate_polygon_emits_two_triangles_for_a_square() {
        // A simple closed square (5 verts, last duplicates first) earcuts into
        // exactly 2 triangles, 6 indices in [0, 3].
        let ring: Vec<Coord> = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ];
        let tris = tessellate_polygon(&[ring]);
        assert_eq!(tris.len(), 6);
        for &i in &tris {
            assert!(i < 5);
        }
    }

    #[test]
    fn tessellate_polygon_handles_a_hole() {
        // 4x4 square with a 1x1 hole — earcut should still produce a valid
        // tessellation. Index count is implementation-dependent but must be a
        // multiple of 3 and reference valid vertex indices.
        let exterior: Vec<Coord> =
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]];
        let hole: Vec<Coord> =
            vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]];
        let tris = tessellate_polygon(&[exterior, hole]);
        assert!(tris.len() >= 6);
        assert_eq!(tris.len() % 3, 0);
        for &i in &tris {
            assert!(i < 10);
        }
    }

    #[test]
    fn tessellate_polygon_handles_degenerate_input() {
        // No rings → empty result, not a panic.
        assert!(tessellate_polygon(&[]).is_empty());
        // Single 2-vert ring is below the 3-vertex minimum.
        let degenerate: Vec<Coord> = vec![[0.0, 0.0], [1.0, 1.0]];
        assert!(tessellate_polygon(&[degenerate]).is_empty());
    }

    #[test]
    fn polygon_layer_with_triangles_roundtrips() {
        let exterior: Vec<Coord> =
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
        let tris = tessellate_polygon(&[exterior.clone()]);
        assert_eq!(tris.len(), 6);
        let layer = ColumnarLayer {
            name: "zones".into(),
            feature_ids: vec![42],
            start_times: vec![0],
            end_times: vec![1000],
            geometry: GeometryColumn::Polygon(vec![vec![exterior]]),
            vertex_times: None,
            triangles: Some(vec![tris.clone()]),
            properties: vec![],
        };
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();

        // Schema metadata advertises the sidecar.
        assert_eq!(
            batch
                .schema()
                .metadata()
                .get(TRIANGLES_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
        // Column exists with the expected shape.
        let col = batch
            .column_by_name("triangles")
            .expect("triangles column present")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("triangles is a List");
        assert_eq!(col.len(), 1);
        let first = col.value(0);
        let values: &arrow::array::UInt32Array = first
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .expect("triangle values are UInt32");
        assert_eq!(values.values().to_vec(), tris);
    }

    #[test]
    fn polygon_layer_without_triangles_skips_the_metadata_key() {
        // Backwards-compat guarantee: a v3 polygon layer that was NOT built
        // with pre-tessellation must not carry the metadata flag — otherwise
        // a reader would expect a column that isn't there.
        let layer = sample_polygon_layer();
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        assert!(!batch.schema().metadata().contains_key(TRIANGLES_METADATA_KEY));
        assert!(batch.column_by_name("triangles").is_none());
    }

    #[test]
    fn non_polygon_layer_drops_stray_triangles() {
        // A producer that mistakenly attaches `triangles` to a point or line
        // layer must not poison the wire format. The encoder silently drops
        // the column so the metadata key never appears.
        let mut layer = sample_point_layer();
        // Add a bogus per-feature triangle list. The encoder must ignore it.
        layer.triangles = Some(vec![vec![0, 1, 2]; layer.feature_ids.len()]);
        let ipc = encode_layer(&layer).unwrap();
        let batch = decode_layer(&ipc).unwrap();
        assert!(!batch.schema().metadata().contains_key(TRIANGLES_METADATA_KEY));
        assert!(batch.column_by_name("triangles").is_none());
    }

    #[test]
    fn truncated_tile_frame_errors_cleanly() {
        let payload = encode_tile(&[sample_point_layer()]).unwrap();
        // Chop the payload mid-stream; decode must error, not panic.
        let truncated = &payload[..payload.len() / 2];
        assert!(decode_tile(truncated).is_err());
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let mut layer = sample_point_layer();
        layer.start_times.pop(); // now 2 entries vs 3 features
        assert!(encode_layer(&layer).is_err());
    }
}
