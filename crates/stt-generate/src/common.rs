//! Common utilities for data generation

use anyhow::Result;
use arrow::array::{ArrayRef, Float64Builder, StringBuilder, TimestampMillisecondBuilder};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, NaiveDate, Utc};
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
    /// Create a writer with all-string property columns (legacy default).
    ///
    /// Prefer [`StreamingParquetWriter::with_columns`] when any property is
    /// numeric — string-typed numerics survive parquet but are misclassified
    /// downstream by stt-build's columnar property inference.
    pub fn new(output_path: &Path, property_columns: Vec<String>) -> Result<Self> {
        let columns = property_columns
            .into_iter()
            .map(PropertyColumn::string)
            .collect();
        Self::with_columns(output_path, columns)
    }

    /// Create a writer with explicitly-typed property columns.
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

/// Write point records to GeoParquet file
pub fn write_geoparquet(records: Vec<PointRecord>, output_path: &Path, property_columns: Vec<String>) -> Result<()> {
    println!("💾 Writing {} records to {:?}", records.len(), output_path);
    
    let mut writer = StreamingParquetWriter::new(output_path, property_columns)?;
    
    for record in &records {
        writer.write_point(record)?;
    }
    
    writer.finish()?;
    Ok(())
}

/// Determine output format based on file extension
pub fn is_parquet_output(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("parquet") || ext.eq_ignore_ascii_case("geoparquet"))
        .unwrap_or(false)
}

/// Parse date string in YYYY-MM-DD format
pub fn parse_date(date_str: &str) -> Result<NaiveDate> {
    Ok(NaiveDate::parse_from_str(date_str, "%Y-%m-%d")?)
}

/// Convert NaiveDate to DateTime<Utc>
pub fn date_to_datetime(date: NaiveDate) -> DateTime<Utc> {
    DateTime::from_naive_utc_and_offset(date.and_hms_opt(0, 0, 0).unwrap(), Utc)
}

/// Unzip a file
pub fn unzip_file(zip_path: &Path, output_dir: &Path) -> Result<()> {
    use zip::ZipArchive;

    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = output_dir.join(file.name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }

    Ok(())
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

/// Encode a Point as WKB (Well-Known Binary)
pub fn encode_wkb_point(lon: f64, lat: f64) -> Vec<u8> {
    let mut wkb = Vec::with_capacity(21);
    wkb.push(1); // Little-endian
    wkb.extend_from_slice(&1u32.to_le_bytes()); // Point type
    wkb.extend_from_slice(&lon.to_le_bytes());
    wkb.extend_from_slice(&lat.to_le_bytes());
    wkb
}

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
            properties,
        }
    }
}

/// Streaming GeoParquet writer for LineString data
pub struct StreamingLineStringParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    property_columns: Vec<String>,
    
    // Batch builders
    geometry_builder: arrow::array::BinaryBuilder,
    timestamp_builder: TimestampMillisecondBuilder,
    end_timestamp_builder: TimestampMillisecondBuilder,
    property_builders: Vec<StringBuilder>,
    
    batch_size: usize,
    current_batch_size: usize,
    total_rows: usize,
}

impl StreamingLineStringParquetWriter {
    pub fn new(output_path: &Path, property_columns: Vec<String>) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Build schema: geometry (WKB), timestamp, end_timestamp, then properties
        let mut fields = vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
            Field::new("end_timestamp", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), true),
        ];
        
        for col in &property_columns {
            fields.push(Field::new(col, DataType::Utf8, true));
        }
        
        let schema = Arc::new(Schema::new(fields));
        
        let file = File::create(output_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();
        
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        
        let batch_size = 100_000;
        let property_builders: Vec<StringBuilder> = property_columns.iter()
            .map(|_| StringBuilder::with_capacity(batch_size, batch_size * 32))
            .collect();

        Ok(Self {
            writer,
            schema,
            property_columns,
            geometry_builder: arrow::array::BinaryBuilder::with_capacity(batch_size, batch_size * 256),
            timestamp_builder: TimestampMillisecondBuilder::with_capacity(batch_size),
            end_timestamp_builder: TimestampMillisecondBuilder::with_capacity(batch_size),
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
        
        for (i, col) in self.property_columns.iter().enumerate() {
            let value = record.properties.get(col)
                .map(|v| json_value_to_string(v))
                .unwrap_or_default();
            self.property_builders[i].append_value(&value);
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
        ];
        
        for builder in &mut self.property_builders {
            columns.push(Arc::new(builder.finish()));
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

    pub fn row_count(&self) -> usize {
        self.total_rows
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
    property_columns: Vec<String>,
    
    // Batch builders
    geometry_builder: arrow::array::BinaryBuilder,
    timestamp_builder: TimestampMillisecondBuilder,
    property_builders: Vec<StringBuilder>,
    
    batch_size: usize,
    current_batch_size: usize,
    total_rows: usize,
}

impl StreamingPolygonParquetWriter {
    pub fn new(output_path: &Path, property_columns: Vec<String>) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Build schema: geometry (WKB), timestamp, then properties
        let mut fields = vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
        ];
        
        for col in &property_columns {
            fields.push(Field::new(col, DataType::Utf8, true));
        }
        
        let schema = Arc::new(Schema::new(fields));
        
        let file = File::create(output_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();
        
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        
        let batch_size = 50_000; // Polygons are larger, use smaller batches
        let property_builders: Vec<StringBuilder> = property_columns.iter()
            .map(|_| StringBuilder::with_capacity(batch_size, batch_size * 32))
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
            let value = record.properties.get(col)
                .map(|v| json_value_to_string(v))
                .unwrap_or_default();
            self.property_builders[i].append_value(&value);
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
            columns.push(Arc::new(builder.finish()));
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
    use std::process::Command;

    println!("\n📦 Building STT archive...");

    let stt_build_path = find_stt_build_binary();
    
    let mut cmd = Command::new(&stt_build_path);
    cmd.arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--time-field")
        .arg(time_field);
    
    if let Some(end_field) = end_time_field {
        cmd.arg("--end-time-field").arg(end_field);
    }
    
    cmd.arg("--min-zoom")
        .arg(min_zoom.to_string())
        .arg("--max-zoom")
        .arg(max_zoom.to_string())
        .arg("--compression")
        .arg(compression);
    
    // Add temporal bucket if specified
    if let Some(bucket) = temporal_bucket {
        cmd.arg("--temporal-bucket").arg(bucket);
    }

    let status = cmd.status();

    match status {
        Ok(s) if s.success() => {
            println!("✅ STT archive created: {}", output.display());
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


