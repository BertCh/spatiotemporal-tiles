//! Common utilities for data generation scripts
//!
//! Provides utilities for:
//! - Downloading files
//! - Creating GeoJSON features
//! - Writing to CSV, GeoJSON, and GeoParquet formats

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use csv::Writer as CsvWriter;
use geojson::{Feature, FeatureCollection, GeoJson, Geometry, Value};
use serde_json::{json, Map, Value as JsonValue};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

// GeoArrow/Arrow imports for GeoParquet support
use arrow::array::{
    ArrayBuilder, ArrayRef, Float32Builder, Float64Builder, Int64Builder, StringBuilder, StructArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Download a file from a URL with progress bar
pub fn download_file(url: &str, output_path: &Path) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::io::copy;

    println!("Downloading: {}", url);

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
/// Used for animated paths where each feature has a start and end time
pub fn create_linestring_feature_with_time_range(
    coordinates: Vec<[f64; 2]>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    properties: Map<String, JsonValue>,
) -> Feature {
    let mut props = properties;
    // Use start_time as the primary timestamp for sorting
    props.insert("timestamp".to_string(), json!(start_time.to_rfc3339()));
    // Add end_time for time range
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
    println!("Writing {} features to {:?}", features.len(), output_path);

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
    let json_string = serde_json::to_string_pretty(&geojson)?;

    let mut file = File::create(output_path)?;
    file.write_all(json_string.as_bytes())?;

    println!("✓ GeoJSON written successfully");
    Ok(())
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
///
/// This is more memory-efficient than GeoJSON for large datasets because
/// it writes directly to disk without building a large in-memory structure.
pub struct StreamingCsvWriter {
    writer: CsvWriter<BufWriter<File>>,
    headers_written: bool,
    property_columns: Vec<String>,
    row_count: usize,
}

impl StreamingCsvWriter {
    /// Create a new streaming CSV writer
    ///
    /// `property_columns` specifies which properties to include and their order.
    /// This must be known upfront so the header can be written.
    pub fn new(output_path: &Path, property_columns: Vec<String>) -> Result<Self> {
        // Ensure parent directory exists
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

    /// Write the CSV header row
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

        // Add property values in the expected column order
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

/// Convert a JSON value to a string for CSV output
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

/// Determine output format based on file extension
pub fn is_csv_output(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("csv"))
        .unwrap_or(false)
}

/// Determine if output is GeoParquet format
pub fn is_geoparquet_output(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("parquet") || ext.eq_ignore_ascii_case("geoparquet"))
        .unwrap_or(false)
}

/// Property column definition for GeoParquet
#[derive(Debug, Clone)]
pub enum PropertyType {
    Float64,
    Float32,
    Int64,
    String,
}

/// Property column definition
#[derive(Debug, Clone)]
pub struct PropertyColumn {
    pub name: String,
    pub prop_type: PropertyType,
}

impl PropertyColumn {
    pub fn float64(name: &str) -> Self {
        Self { name: name.to_string(), prop_type: PropertyType::Float64 }
    }
    
    pub fn float32(name: &str) -> Self {
        Self { name: name.to_string(), prop_type: PropertyType::Float32 }
    }
    
    pub fn int64(name: &str) -> Self {
        Self { name: name.to_string(), prop_type: PropertyType::Int64 }
    }
    
    pub fn string(name: &str) -> Self {
        Self { name: name.to_string(), prop_type: PropertyType::String }
    }
}

/// Streaming GeoParquet writer for point data
///
/// Writes point data with timestamps and properties to GeoParquet format.
/// Uses GeoArrow native encoding (struct with x, y fields) for geometry.
///
/// This is more efficient than GeoJSON for large datasets and provides
/// better interoperability with tools like DuckDB, QGIS, and Python/geopandas.
pub struct StreamingGeoParquetWriter {
    output_path: std::path::PathBuf,
    property_columns: Vec<PropertyColumn>,
    
    // Geometry builders
    lon_builder: Float64Builder,
    lat_builder: Float64Builder,
    
    // Timestamp builder (milliseconds since epoch)
    timestamp_builder: Int64Builder,
    
    // Property builders (stored as trait objects)
    float64_builders: Vec<(String, Float64Builder)>,
    float32_builders: Vec<(String, Float32Builder)>,
    int64_builders: Vec<(String, Int64Builder)>,
    string_builders: Vec<(String, StringBuilder)>,
    
    row_count: usize,
    batch_size: usize,
    batches: Vec<RecordBatch>,
}

impl StreamingGeoParquetWriter {
    /// Create a new streaming GeoParquet writer
    ///
    /// `property_columns` specifies which properties to include and their types.
    pub fn new(output_path: &Path, property_columns: Vec<PropertyColumn>) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Create builders for each property type
        let mut float64_builders = Vec::new();
        let mut float32_builders = Vec::new();
        let mut int64_builders = Vec::new();
        let mut string_builders = Vec::new();
        
        for col in &property_columns {
            match col.prop_type {
                PropertyType::Float64 => float64_builders.push((col.name.clone(), Float64Builder::new())),
                PropertyType::Float32 => float32_builders.push((col.name.clone(), Float32Builder::new())),
                PropertyType::Int64 => int64_builders.push((col.name.clone(), Int64Builder::new())),
                PropertyType::String => string_builders.push((col.name.clone(), StringBuilder::new())),
            }
        }
        
        Ok(Self {
            output_path: output_path.to_path_buf(),
            property_columns,
            lon_builder: Float64Builder::new(),
            lat_builder: Float64Builder::new(),
            timestamp_builder: Int64Builder::new(),
            float64_builders,
            float32_builders,
            int64_builders,
            string_builders,
            row_count: 0,
            batch_size: 100_000, // Flush every 100k rows
            batches: Vec::new(),
        })
    }
    
    /// Write a single point record
    pub fn write_point(
        &mut self,
        lon: f64,
        lat: f64,
        timestamp: DateTime<Utc>,
        properties: &Map<String, JsonValue>,
    ) -> Result<()> {
        // Add geometry
        self.lon_builder.append_value(lon);
        self.lat_builder.append_value(lat);
        
        // Add timestamp (milliseconds since epoch)
        self.timestamp_builder.append_value(timestamp.timestamp_millis());
        
        // Add properties
        for (name, builder) in &mut self.float64_builders {
            let value = properties.get(name)
                .and_then(|v| v.as_f64());
            if let Some(v) = value {
                builder.append_value(v);
            } else {
                builder.append_null();
            }
        }
        
        for (name, builder) in &mut self.float32_builders {
            let value = properties.get(name)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32);
            if let Some(v) = value {
                builder.append_value(v);
            } else {
                builder.append_null();
            }
        }
        
        for (name, builder) in &mut self.int64_builders {
            let value = properties.get(name)
                .and_then(|v| v.as_i64());
            if let Some(v) = value {
                builder.append_value(v);
            } else {
                builder.append_null();
            }
        }
        
        for (name, builder) in &mut self.string_builders {
            let value = properties.get(name)
                .and_then(|v| v.as_str());
            if let Some(v) = value {
                builder.append_value(v);
            } else {
                builder.append_null();
            }
        }
        
        self.row_count += 1;
        
        // Flush if batch is full
        if self.row_count % self.batch_size == 0 {
            self.flush_batch()?;
        }
        
        Ok(())
    }
    
    /// Flush the current batch to memory
    fn flush_batch(&mut self) -> Result<()> {
        if self.lon_builder.len() == 0 {
            return Ok(());
        }
        
        let batch = self.build_batch()?;
        self.batches.push(batch);
        
        // Reset builders
        self.lon_builder = Float64Builder::new();
        self.lat_builder = Float64Builder::new();
        self.timestamp_builder = Int64Builder::new();
        
        for (_, builder) in &mut self.float64_builders {
            *builder = Float64Builder::new();
        }
        for (_, builder) in &mut self.float32_builders {
            *builder = Float32Builder::new();
        }
        for (_, builder) in &mut self.int64_builders {
            *builder = Int64Builder::new();
        }
        for (_, builder) in &mut self.string_builders {
            *builder = StringBuilder::new();
        }
        
        Ok(())
    }
    
    /// Build a RecordBatch from current builders
    fn build_batch(&mut self) -> Result<RecordBatch> {
        // Build geometry struct (GeoArrow native point encoding)
        let lon_array = Arc::new(self.lon_builder.finish()) as ArrayRef;
        let lat_array = Arc::new(self.lat_builder.finish()) as ArrayRef;
        
        let geometry_fields = vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ];
        let geometry_array = StructArray::from(vec![
            (Arc::new(geometry_fields[0].clone()), lon_array),
            (Arc::new(geometry_fields[1].clone()), lat_array),
        ]);
        
        // Build schema and arrays
        let mut fields = vec![
            Field::new(
                "geometry",
                DataType::Struct(geometry_fields.into()),
                false,
            ),
            Field::new("timestamp", DataType::Int64, false),
        ];
        
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(geometry_array),
            Arc::new(self.timestamp_builder.finish()),
        ];
        
        // Add property columns
        for (name, builder) in &mut self.float64_builders {
            fields.push(Field::new(name.clone(), DataType::Float64, true));
            arrays.push(Arc::new(builder.finish()));
        }
        for (name, builder) in &mut self.float32_builders {
            fields.push(Field::new(name.clone(), DataType::Float32, true));
            arrays.push(Arc::new(builder.finish()));
        }
        for (name, builder) in &mut self.int64_builders {
            fields.push(Field::new(name.clone(), DataType::Int64, true));
            arrays.push(Arc::new(builder.finish()));
        }
        for (name, builder) in &mut self.string_builders {
            fields.push(Field::new(name.clone(), DataType::Utf8, true));
            arrays.push(Arc::new(builder.finish()));
        }
        
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)?;
        
        Ok(batch)
    }
    
    /// Finish writing and return the number of rows written
    pub fn finish(mut self) -> Result<usize> {
        // Flush any remaining data
        self.flush_batch()?;
        
        if self.batches.is_empty() {
            anyhow::bail!("No data to write");
        }
        
        // Get schema from first batch
        let schema = self.batches[0].schema();
        
        // Add GeoParquet metadata
        let mut metadata = std::collections::HashMap::new();
        
        // GeoParquet metadata (version 1.0)
        let geo_metadata = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "point",
                    "geometry_types": ["Point"],
                    "crs": {
                        "type": "name",
                        "properties": {
                            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                        }
                    }
                }
            }
        });
        metadata.insert("geo".to_string(), geo_metadata.to_string());
        
        // Create schema with metadata
        let schema_with_metadata = Arc::new(
            Schema::new_with_metadata(schema.fields().to_vec(), metadata)
        );
        
        // Write to Parquet file
        let file = File::create(&self.output_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();
        
        let mut writer = ArrowWriter::try_new(file, schema_with_metadata, Some(props))?;
        
        for batch in &self.batches {
            writer.write(batch)?;
        }
        
        writer.close()?;
        
        println!("✓ GeoParquet written successfully ({} rows)", self.row_count);
        Ok(self.row_count)
    }
    
    /// Get the current row count
    pub fn row_count(&self) -> usize {
        self.row_count
    }
}

/// Helper to create property columns from string names (defaults to String type)
pub fn property_columns_from_names(names: &[&str]) -> Vec<PropertyColumn> {
    names.iter().map(|n| PropertyColumn::string(n)).collect()
}
