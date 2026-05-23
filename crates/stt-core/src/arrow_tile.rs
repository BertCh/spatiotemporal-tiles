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
//! | `<property>`  | `Float64` or `Utf8` (nullable)          | one column per property       |
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
    Array, ArrayRef, FixedSizeListArray, Float64Array, Int64Array, Int64Builder, ListArray,
    ListBuilder, RecordBatch, StringArray, UInt64Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
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

/// Build the optional `List<Int64>` per-vertex time column.
fn build_vertex_time_array(
    vertex_times: &Option<Vec<Vec<i64>>>,
    feature_count: usize,
) -> Option<ArrayRef> {
    let vt = vertex_times.as_ref()?;
    let mut builder = ListBuilder::new(Int64Builder::new());
    for i in 0..feature_count {
        match vt.get(i) {
            Some(times) if !times.is_empty() => {
                for &t in times {
                    builder.values().append_value(t);
                }
                builder.append(true);
            }
            _ => builder.append(false), // null list for features without per-vertex times
        }
    }
    Some(Arc::new(builder.finish()))
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

    if let Some(vt_array) = build_vertex_time_array(&layer.vertex_times, n) {
        fields.push(Arc::new(Field::new(
            "vertex_time",
            vt_array.data_type().clone(),
            true,
        )));
        columns.push(vt_array);
    }

    for (name, col) in &layer.properties {
        match col {
            PropertyColumn::Numeric(values) => {
                fields.push(Arc::new(Field::new(name, DataType::Float64, true)));
                columns.push(Arc::new(Float64Array::from(values.clone())));
            }
            PropertyColumn::Categorical(values) => {
                fields.push(Arc::new(Field::new(name, DataType::Utf8, true)));
                let opt: Vec<Option<&str>> =
                    values.iter().map(|v| v.as_deref()).collect();
                columns.push(Arc::new(StringArray::from(opt)));
            }
        }
    }

    // Schema-level metadata records the layer name and geometry kind so a
    // reader does not have to inspect the geometry column.
    let mut schema_meta = HashMap::new();
    schema_meta.insert("stt:layer".to_string(), layer.name.clone());
    schema_meta.insert(
        "stt:geometry".to_string(),
        layer.geometry.geoarrow_name().to_string(),
    );
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
            properties: vec![],
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

        let vt = batch
            .column_by_name("vertex_time")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(vt.len(), 2);
        let first = vt.value(0);
        let first = first.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(first.values(), &[0, 25, 50]);
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
