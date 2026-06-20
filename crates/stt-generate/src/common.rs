//! Common utilities for data generation

use anyhow::Result;
use arrow::array::{ArrayRef, Float32Builder, Float64Builder, ListBuilder, StringBuilder, TimestampMillisecondBuilder};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use csv::Writer as CsvWriter;
use geojson::{Feature, FeatureCollection, GeoJson, Geometry, Value};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde_json::{json, Map, Value as JsonValue};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

/// Download a file from a URL with progress bar
pub fn download_file(url: &str, output_path: &Path) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};

    println!("⬇️  Downloading: {}", url);

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let response = reqwest::blocking::get(url)?;
    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );

    let mut file = File::create(output_path)?;
    let mut downloaded: u64 = 0;
    let mut content = response;

    loop {
        let mut buffer = vec![0; 8192];
        match std::io::Read::read(&mut content, &mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                file.write_all(&buffer[..n])?;
                downloaded += n as u64;
                pb.set_position(downloaded);
            }
            Err(e) => return Err(e.into()),
        }
    }

    pb.finish_with_message("Download complete");
    Ok(())
}

/// Create a GeoJSON Point feature
pub fn create_point_feature(
    lon: f64,
    lat: f64,
    timestamp: DateTime<Utc>,
    properties: Map<String, JsonValue>,
) -> Feature {
    let mut props = properties;
    props.insert("timestamp".to_string(), json!(timestamp.to_rfc3339()));

    Feature {
        bbox: None,
        geometry: Some(Geometry::new(Value::Point(vec![lon, lat]))),
        id: None,
        properties: Some(props),
        foreign_members: None,
    }
}

/// Create a GeoJSON LineString feature
pub fn create_linestring_feature(
    coordinates: Vec<[f64; 2]>,
    timestamp: DateTime<Utc>,
    properties: Map<String, JsonValue>,
) -> Feature {
    let mut props = properties;
    props.insert("timestamp".to_string(), json!(timestamp.to_rfc3339()));

    let coords: Vec<Vec<f64>> = coordinates
        .into_iter()
        .map(|[lon, lat]| vec![lon, lat])
        .collect();

    Feature {
        bbox: None,
        geometry: Some(Geometry::new(Value::LineString(coords))),
        id: None,
        properties: Some(props),
        foreign_members: None,
    }
}

/// Create a GeoJSON LineString feature with time range
pub fn create_linestring_feature_with_time_range(
    coordinates: Vec<[f64; 2]>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    properties: Map<String, JsonValue>,
) -> Feature {
    let mut props = properties;
    props.insert("timestamp".to_string(), json!(start_time.to_rfc3339()));
    props.insert("end_time".to_string(), json!(end_time.to_rfc3339()));

    let coords: Vec<Vec<f64>> = coordinates
        .into_iter()
        .map(|[lon, lat]| vec![lon, lat])
        .collect();

    Feature {
        bbox: None,
        geometry: Some(Geometry::new(Value::LineString(coords))),
        id: None,
        properties: Some(props),
        foreign_members: None,
    }
}

/// Write GeoJSON FeatureCollection to file
pub fn write_geojson(features: Vec<Feature>, output_path: &Path) -> Result<()> {
    println!("💾 Writing {} features to {:?}", features.len(), output_path);

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let collection = FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    };

    let geojson = GeoJson::FeatureCollection(collection);
    let json_string = serde_json::to_string(&geojson)?;

    let mut file = File::create(output_path)?;
    file.write_all(json_string.as_bytes())?;

    println!("✓ GeoJSON written successfully");
    Ok(())
}

/// A point record for GeoParquet output
#[derive(Debug, Clone)]
pub struct PointRecord {
    pub lon: f64,
    pub lat: f64,
    pub timestamp_ms: i64,
    pub properties: Map<String, JsonValue>,
}

impl PointRecord {
    pub fn new(lon: f64, lat: f64, timestamp: DateTime<Utc>, properties: Map<String, JsonValue>) -> Self {
        Self {
            lon,
            lat,
            timestamp_ms: timestamp.timestamp_millis(),
            properties,
        }
    }
}

/// Type of a property column written to the intermediate parquet.
///
/// stt-build inspects each value's JSON type to decide whether a column is
/// `PropertyColumn::Numeric` or `PropertyColumn::Categorical` in the .stt
/// tile. If we write a number as a Utf8 string here, stt-build sees a JSON
/// string and classifies the column as Categorical — which then can't drive
/// `radiusProperty` / numeric color ramps in the deck.gl layers. Declare
/// numeric columns explicitly so they survive the round-trip.
#[derive(Debug, Clone)]
pub enum PropertyKind {
    /// f64 column. Missing values become null.
    Numeric,
    /// Utf8 column. Missing values become null.
    String,
}

/// A property column spec for [`StreamingParquetWriter`].
#[derive(Debug, Clone)]
pub struct PropertyColumn {
    pub name: String,
    pub kind: PropertyKind,
}

impl PropertyColumn {
    pub fn numeric(name: impl Into<String>) -> Self {
        Self { name: name.into(), kind: PropertyKind::Numeric }
    }
    pub fn string(name: impl Into<String>) -> Self {
        Self { name: name.into(), kind: PropertyKind::String }
    }
}

enum PropertyBuilder {
    Numeric(Float64Builder),
    String(StringBuilder),
}

impl PropertyBuilder {
    fn new(kind: &PropertyKind, capacity: usize) -> Self {
        match kind {
            PropertyKind::Numeric => Self::Numeric(Float64Builder::with_capacity(capacity)),
            PropertyKind::String => Self::String(StringBuilder::with_capacity(capacity, capacity * 32)),
        }
    }
    fn append(&mut self, value: Option<&JsonValue>) {
        match self {
            Self::Numeric(b) => {
                let n = value.and_then(|v| v.as_f64());
                match n {
                    Some(x) => b.append_value(x),
                    None => b.append_null(),
                }
            }
            Self::String(b) => match value {
                Some(JsonValue::Null) | None => b.append_null(),
                Some(JsonValue::String(s)) => b.append_value(s),
                Some(other) => b.append_value(json_value_to_string(other)),
            },
        }
    }
    fn finish(&mut self) -> ArrayRef {
        match self {
            Self::Numeric(b) => Arc::new(b.finish()),
            Self::String(b) => Arc::new(b.finish()),
        }
    }
}

/// Streaming GeoParquet writer for point data
///
/// Uses Arrow/Parquet for efficient columnar storage. Output is ~10x smaller than GeoJSON.
pub struct StreamingParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    property_columns: Vec<PropertyColumn>,

    // Batch builders
    lon_builder: Float64Builder,
    lat_builder: Float64Builder,
    timestamp_builder: TimestampMillisecondBuilder,
    property_builders: Vec<PropertyBuilder>,

    batch_size: usize,
    current_batch_size: usize,
    total_rows: usize,
}

impl StreamingParquetWriter {
    /// Create a writer with explicitly-typed property columns.
    ///
    /// Callers must declare each column's [`PropertyKind`]; there is no
    /// all-string shortcut, because a number written into a Utf8 column is
    /// misclassified as Categorical by stt-build's columnar inference and then
    /// can't drive numeric color ramps / `radiusProperty` / elevation. Use
    /// [`PropertyColumn::string`] explicitly for genuine string columns.
    pub fn with_columns(output_path: &Path, property_columns: Vec<PropertyColumn>) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Build schema: lon, lat, timestamp, then properties
        let mut fields = vec![
            Field::new("lon", DataType::Float64, false),
            Field::new("lat", DataType::Float64, false),
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
        ];

        for col in &property_columns {
            let dt = match col.kind {
                PropertyKind::Numeric => DataType::Float64,
                PropertyKind::String => DataType::Utf8,
            };
            fields.push(Field::new(&col.name, dt, true));
        }

        let schema = Arc::new(Schema::new(fields));

        // Create Parquet writer with compression
        let file = File::create(output_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();

        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        let batch_size = 100_000; // Write in batches of 100k rows
        let property_builders: Vec<PropertyBuilder> = property_columns
            .iter()
            .map(|c| PropertyBuilder::new(&c.kind, batch_size))
            .collect();

        Ok(Self {
            writer,
            schema,
            property_columns,
            lon_builder: Float64Builder::with_capacity(batch_size),
            lat_builder: Float64Builder::with_capacity(batch_size),
            timestamp_builder: TimestampMillisecondBuilder::with_capacity(batch_size),
            property_builders,
            batch_size,
            current_batch_size: 0,
            total_rows: 0,
        })
    }

    /// Write a single point record
    pub fn write_point(&mut self, record: &PointRecord) -> Result<()> {
        self.lon_builder.append_value(record.lon);
        self.lat_builder.append_value(record.lat);
        self.timestamp_builder.append_value(record.timestamp_ms);

        for (i, col) in self.property_columns.iter().enumerate() {
            let value = record.properties.get(&col.name);
            self.property_builders[i].append(value);
        }

        self.current_batch_size += 1;
        self.total_rows += 1;

        if self.current_batch_size >= self.batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.current_batch_size == 0 {
            return Ok(());
        }

        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(self.lon_builder.finish()),
            Arc::new(self.lat_builder.finish()),
            Arc::new(self.timestamp_builder.finish().with_timezone("UTC")),
        ];

        for builder in &mut self.property_builders {
            columns.push(builder.finish());
        }

        let batch = RecordBatch::try_new(self.schema.clone(), columns)?;
        self.writer.write(&batch)?;

        self.current_batch_size = 0;
        Ok(())
    }

    /// Finish writing and return the number of rows written
    pub fn finish(mut self) -> Result<usize> {
        self.flush_batch()?;
        self.writer.close()?;
        println!("✓ GeoParquet written successfully ({} rows)", self.total_rows);
        Ok(self.total_rows)
    }

    /// Get the current row count
    pub fn row_count(&self) -> usize {
        self.total_rows
    }
}

/// Determine output format based on file extension
pub fn is_parquet_output(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("parquet") || ext.eq_ignore_ascii_case("geoparquet"))
        .unwrap_or(false)
}

/// Streaming CSV writer for point data
pub struct StreamingCsvWriter {
    writer: CsvWriter<BufWriter<File>>,
    headers_written: bool,
    property_columns: Vec<String>,
    row_count: usize,
}

impl StreamingCsvWriter {
    /// Create a new streaming CSV writer
    pub fn new(output_path: &Path, property_columns: Vec<String>) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(output_path)?;
        let buf_writer = BufWriter::new(file);
        let writer = CsvWriter::from_writer(buf_writer);

        Ok(Self {
            writer,
            headers_written: false,
            property_columns,
            row_count: 0,
        })
    }

    fn write_headers(&mut self) -> Result<()> {
        if self.headers_written {
            return Ok(());
        }

        let mut headers = vec!["lon".to_string(), "lat".to_string(), "timestamp".to_string()];
        headers.extend(self.property_columns.clone());
        self.writer.write_record(&headers)?;
        self.headers_written = true;
        Ok(())
    }

    /// Write a single point record
    pub fn write_point(
        &mut self,
        lon: f64,
        lat: f64,
        timestamp: DateTime<Utc>,
        properties: &Map<String, JsonValue>,
    ) -> Result<()> {
        self.write_headers()?;

        let mut record = vec![
            format!("{:.6}", lon),
            format!("{:.6}", lat),
            timestamp.to_rfc3339(),
        ];

        for col in &self.property_columns {
            let value = properties
                .get(col)
                .map(|v| json_value_to_string(v))
                .unwrap_or_default();
            record.push(value);
        }

        self.writer.write_record(&record)?;
        self.row_count += 1;
        Ok(())
    }

    /// Finish writing and return the number of rows written
    pub fn finish(mut self) -> Result<usize> {
        self.writer.flush()?;
        println!("✓ CSV written successfully ({} rows)", self.row_count);
        Ok(self.row_count)
    }

    /// Get the current row count
    pub fn row_count(&self) -> usize {
        self.row_count
    }
}

fn json_value_to_string(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => String::new(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::String(s) => s.clone(),
        JsonValue::Array(arr) => serde_json::to_string(arr).unwrap_or_default(),
        JsonValue::Object(obj) => serde_json::to_string(obj).unwrap_or_default(),
    }
}

// ============================================================================
// WKB Encoding for GeoParquet
// ============================================================================

/// Encode a LineString as WKB (Well-Known Binary)
/// coordinates: vec of [lon, lat] pairs
pub fn encode_wkb_linestring(coordinates: &[[f64; 2]]) -> Vec<u8> {
    let num_points = coordinates.len() as u32;
    let mut wkb = Vec::with_capacity(9 + num_points as usize * 16);
    
    wkb.push(1); // Little-endian
    wkb.extend_from_slice(&2u32.to_le_bytes()); // LineString type
    wkb.extend_from_slice(&num_points.to_le_bytes());
    
    for [lon, lat] in coordinates {
        wkb.extend_from_slice(&lon.to_le_bytes());
        wkb.extend_from_slice(&lat.to_le_bytes());
    }
    
    wkb
}

/// Encode a Polygon as WKB (Well-Known Binary)
/// rings: vec of rings, where each ring is a vec of [lon, lat] pairs
/// First ring is the exterior ring, subsequent rings are holes
pub fn encode_wkb_polygon(rings: &[Vec<[f64; 2]>]) -> Vec<u8> {
    let num_rings = rings.len() as u32;
    let total_points: usize = rings.iter().map(|r| r.len()).sum();
    let mut wkb = Vec::with_capacity(9 + num_rings as usize * 4 + total_points * 16);
    
    wkb.push(1); // Little-endian
    wkb.extend_from_slice(&3u32.to_le_bytes()); // Polygon type
    wkb.extend_from_slice(&num_rings.to_le_bytes());
    
    for ring in rings {
        let num_points = ring.len() as u32;
        wkb.extend_from_slice(&num_points.to_le_bytes());
        
        for [lon, lat] in ring {
            wkb.extend_from_slice(&lon.to_le_bytes());
            wkb.extend_from_slice(&lat.to_le_bytes());
        }
    }
    
    wkb
}

// ============================================================================
// Streaming GeoParquet Writers for LineString and Polygon
// ============================================================================

/// A LineString record for GeoParquet output
#[derive(Debug, Clone)]
pub struct LineStringRecord {
    pub coordinates: Vec<[f64; 2]>,
    pub timestamp_ms: i64,
    pub end_timestamp_ms: Option<i64>,
    /// Optional per-vertex absolute Unix-ms timestamps. When present, must
    /// be the same length as `coordinates`. Consumed by stt-build to bypass
    /// its uniform-by-distance interpolation and produce real per-segment
    /// timing (e.g. from OSRM `annotations=duration`).
    pub vertex_timestamps_ms: Option<Vec<i64>>,
    /// Optional per-vertex scalar values (producer-defined; e.g. sea-surface
    /// temperature). When present, must be the same length as `coordinates`.
    /// Carried through stt-build into the tile's `vertex_value` column so
    /// renderers can color the line by it.
    pub vertex_values: Option<Vec<f32>>,
    /// Optional per-vertex × per-time-bucket value matrix, flattened
    /// **vertex-major**: `matrix[v * num_buckets + b]`. Length must be a
    /// multiple of `coordinates.len()`; `num_buckets = len / num_vertices`.
    /// Lets a static-geometry overview (e.g. flow corridors) carry a time
    /// series of per-vertex values without re-emitting geometry per bucket.
    /// Carried through stt-build into the tile's `vertex_value_matrix` column.
    pub vertex_value_matrix: Option<Vec<f32>>,
    pub properties: Map<String, JsonValue>,
}

impl LineStringRecord {
    pub fn new(
        coordinates: Vec<[f64; 2]>,
        timestamp: DateTime<Utc>,
        end_timestamp: Option<DateTime<Utc>>,
        properties: Map<String, JsonValue>,
    ) -> Self {
        Self {
            coordinates,
            timestamp_ms: timestamp.timestamp_millis(),
            end_timestamp_ms: end_timestamp.map(|t| t.timestamp_millis()),
            vertex_timestamps_ms: None,
            vertex_values: None,
            vertex_value_matrix: None,
            properties,
        }
    }
}

/// Streaming GeoParquet writer for LineString data
pub struct StreamingLineStringParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    property_columns: Vec<PropertyColumn>,

    // Batch builders
    geometry_builder: arrow::array::BinaryBuilder,
    timestamp_builder: TimestampMillisecondBuilder,
    end_timestamp_builder: TimestampMillisecondBuilder,
    /// Per-vertex absolute Unix-ms timestamps (one list per row, nullable).
    /// `Timestamp(Millisecond, "UTC")` child so downstream readers can rely
    /// on the type without a numeric→time coercion.
    vertex_timestamps_builder: ListBuilder<TimestampMillisecondBuilder>,
    /// Per-vertex scalar values (one list per row, nullable), `Float32` child.
    vertex_values_builder: ListBuilder<Float32Builder>,
    /// Per-vertex × per-bucket value matrix, flattened vertex-major (one list
    /// per row, nullable), `Float32` child. Length is `num_vertices *
    /// num_buckets`; the bucket count is recovered downstream from the row's
    /// vertex count.
    vertex_value_matrix_builder: ListBuilder<Float32Builder>,
    property_builders: Vec<PropertyBuilder>,

    batch_size: usize,
    current_batch_size: usize,
    total_rows: usize,
}

impl StreamingLineStringParquetWriter {
    /// Create a writer with explicitly-typed property columns.
    ///
    /// Declare each column's [`PropertyKind`] — a number stored as Utf8 is
    /// misclassified as Categorical by stt-build's columnar inference and
    /// can't drive numeric color ramps / `radiusProperty` / elevation. Use
    /// [`PropertyColumn::string`] explicitly for genuine string columns.
    pub fn with_columns(
        output_path: &Path,
        property_columns: Vec<PropertyColumn>,
    ) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // The list child is `Timestamp(Millisecond, None)` to match what
        // `TimestampMillisecondBuilder.finish()` actually produces — tagging
        // the field with `Some("UTC")` while the builder stays tz-less
        // tripped a runtime mismatch panic at flush time. Values are still
        // absolute Unix-ms; downstream stt-build reads them tz-agnostic.
        let vertex_time_field = Field::new(
            "item",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        );

        // Build schema: geometry (WKB), timestamp, end_timestamp,
        // vertex_timestamps (optional list), then properties.
        let mut fields = vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
            Field::new("end_timestamp", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), true),
            Field::new(
                "vertex_timestamps",
                DataType::List(Arc::new(vertex_time_field.clone())),
                true,
            ),
            Field::new(
                "vertex_values",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
            Field::new(
                "vertex_value_matrix",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
        ];

        for col in &property_columns {
            let dt = match col.kind {
                PropertyKind::Numeric => DataType::Float64,
                PropertyKind::String => DataType::Utf8,
            };
            fields.push(Field::new(&col.name, dt, true));
        }

        let schema = Arc::new(Schema::new(fields));

        let file = File::create(output_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();

        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        let batch_size = 100_000;
        let property_builders: Vec<PropertyBuilder> = property_columns
            .iter()
            .map(|c| PropertyBuilder::new(&c.kind, batch_size))
            .collect();

        // The inner timestamp values stay in i64 storage; we attach the
        // timezone at flush time.
        let vertex_timestamps_builder =
            ListBuilder::new(TimestampMillisecondBuilder::with_capacity(batch_size * 32))
                .with_field(Arc::new(vertex_time_field));

        let vertex_values_builder =
            ListBuilder::new(Float32Builder::with_capacity(batch_size * 32))
                .with_field(Arc::new(Field::new("item", DataType::Float32, true)));

        let vertex_value_matrix_builder =
            ListBuilder::new(Float32Builder::with_capacity(batch_size * 32))
                .with_field(Arc::new(Field::new("item", DataType::Float32, true)));

        Ok(Self {
            writer,
            schema,
            property_columns,
            geometry_builder: arrow::array::BinaryBuilder::with_capacity(batch_size, batch_size * 256),
            timestamp_builder: TimestampMillisecondBuilder::with_capacity(batch_size),
            end_timestamp_builder: TimestampMillisecondBuilder::with_capacity(batch_size),
            vertex_timestamps_builder,
            vertex_values_builder,
            vertex_value_matrix_builder,
            property_builders,
            batch_size,
            current_batch_size: 0,
            total_rows: 0,
        })
    }

    pub fn write_linestring(&mut self, record: &LineStringRecord) -> Result<()> {
        let wkb = encode_wkb_linestring(&record.coordinates);
        self.geometry_builder.append_value(&wkb);
        self.timestamp_builder.append_value(record.timestamp_ms);

        if let Some(end_ts) = record.end_timestamp_ms {
            self.end_timestamp_builder.append_value(end_ts);
        } else {
            self.end_timestamp_builder.append_null();
        }

        match &record.vertex_timestamps_ms {
            Some(vt) if vt.len() == record.coordinates.len() => {
                let inner = self.vertex_timestamps_builder.values();
                for &t in vt {
                    inner.append_value(t);
                }
                self.vertex_timestamps_builder.append(true);
            }
            _ => {
                // Either absent or a length mismatch — leave null so stt-build
                // falls back to even-distance interpolation. A mismatch is
                // silently dropped; callers should validate before writing.
                self.vertex_timestamps_builder.append(false);
            }
        }

        match &record.vertex_values {
            Some(vv) if vv.len() == record.coordinates.len() => {
                let inner = self.vertex_values_builder.values();
                for &v in vv {
                    inner.append_value(v);
                }
                self.vertex_values_builder.append(true);
            }
            _ => {
                // Absent or length mismatch — leave null so no value channel
                // is emitted for this row.
                self.vertex_values_builder.append(false);
            }
        }

        let nverts = record.coordinates.len();
        match &record.vertex_value_matrix {
            Some(m) if nverts > 0 && !m.is_empty() && m.len() % nverts == 0 => {
                let inner = self.vertex_value_matrix_builder.values();
                for &v in m {
                    inner.append_value(v);
                }
                self.vertex_value_matrix_builder.append(true);
            }
            _ => {
                // Absent or not a clean multiple of the vertex count — leave
                // null so no matrix channel is emitted for this row.
                self.vertex_value_matrix_builder.append(false);
            }
        }

        for (i, col) in self.property_columns.iter().enumerate() {
            let value = record.properties.get(&col.name);
            self.property_builders[i].append(value);
        }

        self.current_batch_size += 1;
        self.total_rows += 1;

        if self.current_batch_size >= self.batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.current_batch_size == 0 {
            return Ok(());
        }

        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(self.geometry_builder.finish()),
            Arc::new(self.timestamp_builder.finish().with_timezone("UTC")),
            Arc::new(self.end_timestamp_builder.finish().with_timezone("UTC")),
            Arc::new(self.vertex_timestamps_builder.finish()),
            Arc::new(self.vertex_values_builder.finish()),
            Arc::new(self.vertex_value_matrix_builder.finish()),
        ];

        for builder in &mut self.property_builders {
            columns.push(builder.finish());
        }

        let batch = RecordBatch::try_new(self.schema.clone(), columns)?;
        self.writer.write(&batch)?;

        self.current_batch_size = 0;
        Ok(())
    }

    pub fn finish(mut self) -> Result<usize> {
        self.flush_batch()?;
        self.writer.close()?;
        println!("✓ GeoParquet (LineString) written successfully ({} rows)", self.total_rows);
        Ok(self.total_rows)
    }
}

/// A Polygon record for GeoParquet output
#[derive(Debug, Clone)]
pub struct PolygonRecord {
    pub rings: Vec<Vec<[f64; 2]>>,
    pub timestamp_ms: i64,
    pub properties: Map<String, JsonValue>,
}

impl PolygonRecord {
    pub fn new(
        rings: Vec<Vec<[f64; 2]>>,
        timestamp: DateTime<Utc>,
        properties: Map<String, JsonValue>,
    ) -> Self {
        Self {
            rings,
            timestamp_ms: timestamp.timestamp_millis(),
            properties,
        }
    }
}

/// Streaming GeoParquet writer for Polygon data
pub struct StreamingPolygonParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    property_columns: Vec<PropertyColumn>,

    // Batch builders
    geometry_builder: arrow::array::BinaryBuilder,
    timestamp_builder: TimestampMillisecondBuilder,
    property_builders: Vec<PropertyBuilder>,

    batch_size: usize,
    current_batch_size: usize,
    total_rows: usize,
}

impl StreamingPolygonParquetWriter {
    /// Create a writer with explicitly-typed property columns.
    ///
    /// Declare each column's [`PropertyKind`] — a number stored as Utf8 is
    /// misclassified as Categorical by stt-build's columnar inference. Use
    /// [`PropertyColumn::string`] explicitly for genuine string columns.
    pub fn with_columns(
        output_path: &Path,
        property_columns: Vec<PropertyColumn>,
    ) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Build schema: geometry (WKB), timestamp, then properties
        let mut fields = vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
        ];

        for col in &property_columns {
            let dt = match col.kind {
                PropertyKind::Numeric => DataType::Float64,
                PropertyKind::String => DataType::Utf8,
            };
            fields.push(Field::new(&col.name, dt, true));
        }

        let schema = Arc::new(Schema::new(fields));

        let file = File::create(output_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();

        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        let batch_size = 50_000; // Polygons are larger, use smaller batches
        let property_builders: Vec<PropertyBuilder> = property_columns
            .iter()
            .map(|c| PropertyBuilder::new(&c.kind, batch_size))
            .collect();

        Ok(Self {
            writer,
            schema,
            property_columns,
            geometry_builder: arrow::array::BinaryBuilder::with_capacity(batch_size, batch_size * 1024),
            timestamp_builder: TimestampMillisecondBuilder::with_capacity(batch_size),
            property_builders,
            batch_size,
            current_batch_size: 0,
            total_rows: 0,
        })
    }

    pub fn write_polygon(&mut self, record: &PolygonRecord) -> Result<()> {
        let wkb = encode_wkb_polygon(&record.rings);
        self.geometry_builder.append_value(&wkb);
        self.timestamp_builder.append_value(record.timestamp_ms);

        for (i, col) in self.property_columns.iter().enumerate() {
            let value = record.properties.get(&col.name);
            self.property_builders[i].append(value);
        }

        self.current_batch_size += 1;
        self.total_rows += 1;

        if self.current_batch_size >= self.batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.current_batch_size == 0 {
            return Ok(());
        }

        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(self.geometry_builder.finish()),
            Arc::new(self.timestamp_builder.finish().with_timezone("UTC")),
        ];

        for builder in &mut self.property_builders {
            columns.push(builder.finish());
        }
        
        let batch = RecordBatch::try_new(self.schema.clone(), columns)?;
        self.writer.write(&batch)?;
        
        self.current_batch_size = 0;
        Ok(())
    }

    pub fn finish(mut self) -> Result<usize> {
        self.flush_batch()?;
        self.writer.close()?;
        println!("✓ GeoParquet (Polygon) written successfully ({} rows)", self.total_rows);
        Ok(self.total_rows)
    }

    pub fn row_count(&self) -> usize {
        self.total_rows
    }
}

/// Determine output format based on file extension
pub fn is_csv_output(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("csv"))
        .unwrap_or(false)
}

/// Run stt-build to create an STT archive
pub fn run_stt_build(
    input: &Path,
    output: &Path,
    time_field: &str,
    min_zoom: u8,
    max_zoom: u8,
    compression: &str,
) -> Result<()> {
    run_stt_build_with_options(input, output, time_field, None, min_zoom, max_zoom, compression, None)
}

/// Find the stt-build binary - uses local binary if available, otherwise falls back to PATH
pub fn find_stt_build_binary() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("stt-build")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("stt-build"))
}

/// Run stt-build to create an STT archive with optional end time field
pub fn run_stt_build_with_end_time(
    input: &Path,
    output: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
    min_zoom: u8,
    max_zoom: u8,
    compression: &str,
) -> Result<()> {
    run_stt_build_with_options(input, output, time_field, end_time_field, min_zoom, max_zoom, compression, None)
}

/// Run stt-build to create an STT archive with all options
pub fn run_stt_build_with_options(
    input: &Path,
    output: &Path,
    time_field: &str,
    end_time_field: Option<&str>,
    min_zoom: u8,
    max_zoom: u8,
    compression: &str,
    temporal_bucket: Option<&str>,
) -> Result<()> {
    run_stt_build_with_full_options(SttBuildOptions {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        time_field: time_field.to_string(),
        end_time_field: end_time_field.map(str::to_string),
        min_zoom,
        max_zoom,
        compression: compression.to_string(),
        temporal_bucket: temporal_bucket.map(str::to_string),
        temporal_lod: None,
        summary: None,
        summary_sub_buckets: None,
        min_features_per_tile: None,
        min_zoom_field: None,
        max_zoom_field: None,
        no_clip: false,
        quantize_coords: None,
        quantize_attrs: Vec::new(),
    })
}

/// Optional summary-tier knobs forwarded to `stt-build --summary-tier`.
#[derive(Debug, Clone)]
pub struct SttBuildSummaryOptions {
    /// e.g. `"h3"` (only h3 implemented today).
    pub scheme: String,
    /// `--summary-min-zoom`. `None` defaults to the archive's min-zoom.
    pub min_zoom: Option<u8>,
    /// `--summary-max-zoom`. `None` defaults to min-zoom + 4.
    pub max_zoom: Option<u8>,
    /// Comma-separated `name:agg` list. e.g. `magnitude:mean,magnitude:max`.
    pub columns: String,
}

/// All knobs `run_stt_build_*` can pass to the underlying CLI. Use this
/// instead of the positional helpers when you need the summary tier.
#[derive(Debug, Clone)]
pub struct SttBuildOptions {
    pub input: std::path::PathBuf,
    pub output: std::path::PathBuf,
    pub time_field: String,
    pub end_time_field: Option<String>,
    pub min_zoom: u8,
    pub max_zoom: u8,
    pub compression: String,
    pub temporal_bucket: Option<String>,
    /// Forwarded to `stt-build --temporal-lod`. Coarser-bucket pyramid spec
    /// (e.g. `"30d,365d"` or `"30d@12,365d@8"`). Each entry must be a multiple
    /// of `temporal_bucket`. `None` skips the LOD pyramid. Use for multi-year
    /// datasets so low zooms animate at coarse time resolution.
    pub temporal_lod: Option<String>,
    pub summary: Option<SttBuildSummaryOptions>,
    /// Forwarded to `stt-build --summary-sub-buckets`. Splits each summary-tier
    /// temporal bucket into N `bucket_<i>` count sub-columns so the renderer can
    /// animate intra-bucket with no data re-upload. `None` keeps the default 1.
    /// Only meaningful when `summary` is set.
    pub summary_sub_buckets: Option<u32>,
    /// Forwarded to `stt-build --min-features-per-tile`. Use for globally
    /// sparse point datasets where deep-zoom single-feature tiles are pure
    /// overhead. `None` keeps the stt-build default of 1 (write all
    /// non-empty tiles).
    pub min_features_per_tile: Option<u32>,
    /// Forwarded to `stt-build --min-zoom-field`. Names a per-feature numeric
    /// property that is the road-class LOD floor: a feature is hidden at zooms
    /// below its value. `None` = no filter.
    pub min_zoom_field: Option<String>,
    /// Forwarded to `stt-build --max-zoom-field`. Names a per-feature numeric
    /// property that is the LOD ceiling: a feature is hidden at zooms above its
    /// value. Paired with `min_zoom_field` it confines a feature to a zoom band
    /// (coarse-zoom aggregates that must not bleed into deep zooms). `None` = no
    /// ceiling.
    pub max_zoom_field: Option<String>,
    /// Forwarded to `stt-build --no-clip`. For OD / flowmap datasets whose
    /// 2-vertex source→target lines must NOT be cut at tile boundaries —
    /// clipping replaces the true station endpoints with tile-edge points, which
    /// breaks the source→target semantics (false "stations" on seams). When
    /// true, whole features are stored in their centroid tile instead.
    pub no_clip: bool,
    /// Forwarded to `stt-build --quantize-coords <meters>`. Stores coordinates as
    /// `i32` grid indices at this precision instead of Float64 — roughly halves
    /// the geometry column and delta-compresses well, at the cost of a
    /// self-describing tile. `None` keeps full Float64 precision. Use for
    /// coordinate-heavy builds (e.g. baked bundled corridors) where sub-meter
    /// precision is invisible at the rendered zooms.
    ///
    /// **Born-optimized default:** `None` means "use the generation default"
    /// ([`DEFAULT_QUANTIZE_COORDS_M`]), NOT "off". Pass `Some(0.0)` to force raw
    /// Float64 coords, or `Some(m)` for an explicit precision.
    pub quantize_coords: Option<f64>,
    /// Explicit per-property quantization precisions, forwarded as
    /// `--quantize-attr name=prec`. Overrides the automatic range-adaptive
    /// quantization for the named columns (use when a column needs a specific
    /// precision, e.g. integer-exact). Empty = rely on the automatic pass.
    pub quantize_attrs: Vec<String>,
}

/// Born-optimized coordinate-quantization default (meters) applied to every
/// generated dataset unless the caller passes an explicit `quantize_coords`.
/// 0.1 m is sub-pixel at any rendered zoom (the STT reader reconstructs
/// Float64), so geometry — the dominant column on trip/path/line datasets —
/// ships as compact `i32` grid indices for free. Set `STT_GEN_NO_QUANTIZE=1` to
/// disable all generation-time quantization.
pub const DEFAULT_QUANTIZE_COORDS_M: f64 = 0.1;

/// Single entry point that drives the stt-build CLI with every option,
/// including the optional summary tier.
pub fn run_stt_build_with_full_options(opts: SttBuildOptions) -> Result<()> {
    use std::process::Command;

    println!("\n📦 Building STT archive...");

    let stt_build_path = find_stt_build_binary();

    let mut cmd = Command::new(&stt_build_path);
    cmd.arg("--input")
        .arg(&opts.input)
        .arg("--output")
        .arg(&opts.output)
        .arg("--time-field")
        .arg(&opts.time_field);

    if let Some(end_field) = &opts.end_time_field {
        cmd.arg("--end-time-field").arg(end_field);
    }

    cmd.arg("--min-zoom")
        .arg(opts.min_zoom.to_string())
        .arg("--max-zoom")
        .arg(opts.max_zoom.to_string())
        .arg("--compression")
        .arg(&opts.compression);

    // The dataset-generation workflow produces DEPLOY-READY output: `--publish`
    // bundles the lossless wins (zstd-19 + paged directory) into the core build
    // so a generated dataset needs no separate re-transcode before an R2 sync.
    // `STT_GEN_DEV=1` opts out for fast dev iteration (dev-level zstd, single dir).
    if std::env::var_os("STT_GEN_DEV").is_none() {
        cmd.arg("--publish");
    }

    if let Some(bucket) = &opts.temporal_bucket {
        cmd.arg("--temporal-bucket").arg(bucket);
    }

    if let Some(lod) = &opts.temporal_lod {
        cmd.arg("--temporal-lod").arg(lod);
    }

    if let Some(n) = opts.min_features_per_tile {
        cmd.arg("--min-features-per-tile").arg(n.to_string());
    }

    if let Some(field) = &opts.min_zoom_field {
        cmd.arg("--min-zoom-field").arg(field);
    }

    if let Some(field) = &opts.max_zoom_field {
        cmd.arg("--max-zoom-field").arg(field);
    }

    if opts.no_clip {
        cmd.arg("--no-clip");
    }

    // Born-optimized generation: every dataset ships quantized geometry +
    // automatically-quantized numeric properties + sequential point ids (the
    // last is a stt-build default). This is intrinsic to generation, not a
    // second-pass re-transcode. `STT_GEN_NO_QUANTIZE=1` opts out wholesale.
    let quantize_off = std::env::var_os("STT_GEN_NO_QUANTIZE").is_some();
    if !quantize_off {
        // `None` → the generation default; `Some(0.0)` → an explicit opt-out.
        let qc = opts.quantize_coords.unwrap_or(DEFAULT_QUANTIZE_COORDS_M);
        if qc > 0.0 {
            cmd.arg("--quantize-coords").arg(qc.to_string());
        }
        // Explicit per-column precisions win; everything else gets the automatic
        // range-adaptive (Float64 → u16) pass.
        for attr in &opts.quantize_attrs {
            cmd.arg("--quantize-attr").arg(attr);
        }
        cmd.arg("--quantize-attrs-auto");
    } else if let Some(m) = opts.quantize_coords {
        cmd.arg("--quantize-coords").arg(m.to_string());
    }

    if let Some(sm) = &opts.summary {
        cmd.arg("--summary-tier").arg(&sm.scheme);
        if let Some(z) = sm.min_zoom {
            cmd.arg("--summary-min-zoom").arg(z.to_string());
        }
        if let Some(z) = sm.max_zoom {
            cmd.arg("--summary-max-zoom").arg(z.to_string());
        }
        if !sm.columns.is_empty() {
            cmd.arg("--summary-columns").arg(&sm.columns);
        }
        if let Some(n) = opts.summary_sub_buckets {
            cmd.arg("--summary-sub-buckets").arg(n.to_string());
        }
    }

    let status = cmd.status();

    match status {
        Ok(s) if s.success() => {
            println!("✅ STT archive created: {}", opts.output.display());
            Ok(())
        }
        Ok(s) => {
            anyhow::bail!("stt-build exited with status: {}", s);
        }
        Err(e) => {
            anyhow::bail!("Failed to run stt-build: {}. Is it installed?", e);
        }
    }
}

/// Parse geographic bounds from string "min_lat,min_lon,max_lat,max_lon"
pub fn parse_bounds(s: &str) -> Result<(f64, f64, f64, f64)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        anyhow::bail!("Bounds must be: min_lat,min_lon,max_lat,max_lon");
    }

    Ok((
        parts[0].parse()?,
        parts[1].parse()?,
        parts[2].parse()?,
        parts[3].parse()?,
    ))
}

/// Calculate haversine distance between two points in meters
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS: f64 = 6371000.0; // meters

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS * c
}


