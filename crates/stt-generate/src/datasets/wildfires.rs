//! Generate wildfire perimeter data from NIFC (National Interagency Fire Center)
//!
//! Data source: https://data-nifc.opendata.arcgis.com/

use crate::common::{self, PolygonRecord, StreamingPolygonParquetWriter};
use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use clap::Parser;
use geojson::{Feature, Geometry, Value as GeoValue};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use serde_json::{json, Map};
use std::path::PathBuf;

const NIFC_SERVICE_URL: &str = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query";
const MAX_RECORDS_PER_REQUEST: u32 = 2000;

#[derive(Parser, Debug)]
#[command(about = "Download wildfire perimeter polygons from NIFC")]
pub struct Args {
    /// Output file (.stt, .geojson)
    #[arg(short, long, default_value = "wildfires.stt")]
    pub output: PathBuf,

    /// Start year (inclusive)
    #[arg(long, default_value = "2020")]
    pub start_year: u32,

    /// End year (inclusive)
    #[arg(long, default_value = "2023")]
    pub end_year: u32,

    /// Minimum fire size in acres
    #[arg(long, default_value = "1000")]
    pub min_acres: u32,

    /// Maximum number of fires to fetch (0 = no limit)
    #[arg(long, default_value = "0")]
    pub max_fires: usize,

    /// Filter to only wildfires (exclude prescribed burns)
    #[arg(long, default_value = "true")]
    pub wildfires_only: bool,

    /// Skip stt-build step
    #[arg(long)]
    pub skip_build: bool,
}

#[derive(Deserialize, Debug)]
struct ArcGISResponse {
    features: Vec<ArcGISFeature>,
    #[serde(rename = "exceededTransferLimit")]
    exceeded_transfer_limit: Option<bool>,
}

#[derive(Deserialize, Debug)]
struct ArcGISFeature {
    attributes: ArcGISAttributes,
    geometry: Option<ArcGISGeometry>,
}

#[derive(Deserialize, Debug)]
struct ArcGISAttributes {
    #[serde(rename = "OBJECTID", deserialize_with = "deserialize_flex_i64", default)]
    object_id: Option<i64>,
    #[serde(rename = "INCIDENT")]
    incident: Option<String>,
    #[serde(rename = "FIRE_YEAR", deserialize_with = "deserialize_flex_i32", default)]
    fire_year: Option<i32>,
    #[serde(rename = "GIS_ACRES", deserialize_with = "deserialize_flex_f64", default)]
    gis_acres: Option<f64>,
    #[serde(rename = "AGENCY")]
    agency: Option<String>,
    #[serde(rename = "UNIT_ID")]
    unit_id: Option<String>,
    #[serde(rename = "DATE_CUR", deserialize_with = "deserialize_date", default)]
    date_cur: Option<i64>,
    #[serde(rename = "FEATURE_CA")]
    feature_category: Option<String>,
    #[serde(rename = "IRWINID")]
    irwin_id: Option<String>,
    #[serde(rename = "MAP_METHOD")]
    map_method: Option<String>,
}

fn deserialize_flex_i64<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum FlexValue {
        Number(i64),
        String(String),
        Null,
    }

    match FlexValue::deserialize(deserializer)? {
        FlexValue::Number(n) => Ok(Some(n)),
        FlexValue::String(s) => Ok(s.parse().ok()),
        FlexValue::Null => Ok(None),
    }
}

fn deserialize_flex_i32<'de, D>(deserializer: D) -> Result<Option<i32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum FlexValue {
        Number(i32),
        String(String),
        Null,
    }

    match FlexValue::deserialize(deserializer)? {
        FlexValue::Number(n) => Ok(Some(n)),
        FlexValue::String(s) => Ok(s.parse().ok()),
        FlexValue::Null => Ok(None),
    }
}

fn deserialize_flex_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum FlexValue {
        Number(f64),
        String(String),
        Null,
    }

    match FlexValue::deserialize(deserializer)? {
        FlexValue::Number(n) => Ok(Some(n)),
        FlexValue::String(s) => Ok(s.parse().ok()),
        FlexValue::Null => Ok(None),
    }
}

fn deserialize_date<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DateValue {
        Number(i64),
        String(String),
        Null,
    }

    match DateValue::deserialize(deserializer)? {
        DateValue::Number(n) => Ok(parse_date_value(n)),
        DateValue::String(s) => {
            if s.is_empty() {
                return Ok(None);
            }
            if let Ok(n) = s.parse::<i64>() {
                return Ok(parse_date_value(n));
            }
            Ok(None)
        }
        DateValue::Null => Ok(None),
    }
}

fn parse_date_value(n: i64) -> Option<i64> {
    if n >= 19000101000000 && n <= 20991231235959 {
        let year = (n / 10000000000) as i32;
        let month = ((n / 100000000) % 100) as u32;
        let day = ((n / 1000000) % 100) as u32;
        let hour = ((n / 10000) % 100) as u32;
        let min = ((n / 100) % 100) as u32;
        let sec = (n % 100) as u32;
        if let Some(dt) = chrono::NaiveDate::from_ymd_opt(year, month, day)
            .and_then(|d| d.and_hms_opt(hour, min, sec))
        {
            return Some(dt.and_utc().timestamp_millis());
        }
    }
    if n >= 19000101 && n <= 20991231 {
        let year = (n / 10000) as i32;
        let month = ((n % 10000) / 100) as u32;
        let day = (n % 100) as u32;
        if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, day) {
            let datetime = date.and_hms_opt(0, 0, 0).unwrap();
            return Some(datetime.and_utc().timestamp_millis());
        }
    }
    if n > 1000000000000 {
        return Some(n);
    }
    if n > 1000000000 {
        return Some(n * 1000);
    }
    None
}

#[derive(Deserialize, Debug)]
struct ArcGISGeometry {
    rings: Option<Vec<Vec<Vec<f64>>>>,
}

pub fn run(args: Args) -> Result<()> {
    println!("🔥 NIFC Wildfire Data Generator");
    println!("================================\n");
    println!("📊 Fetching wildfires from {} to {}", args.start_year, args.end_year);
    println!("   Minimum size: {} acres", args.min_acres);
    if args.wildfires_only {
        println!("   Filter: Wildfires only");
    }
    println!();

    // Determine intermediate output format (prefer Parquet for efficiency)
    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    let use_parquet = common::is_parquet_output(&intermediate_path);
    
    if use_parquet {
        println!("📄 Using streaming GeoParquet output (efficient)");
    }

    if use_parquet {
        let count = fetch_wildfire_data_parquet(&args, &intermediate_path)?;
        
        if count == 0 {
            println!("\n⚠️  No features found matching criteria");
            return Ok(());
        }
        
        println!("\n✓ Written {} features", count);
    } else {
        let features = fetch_wildfire_data(&args)?;

        if features.is_empty() {
            println!("\n⚠️  No features found matching criteria");
            return Ok(());
        }

        println!("\n💾 Writing {} features...", features.len());
        common::write_geojson(features, &intermediate_path)?;
    }

    // Build STT if output is .stt
    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        common::run_stt_build(
            &intermediate_path,
            &args.output,
            "timestamp",
            3,
            12,
            "zstd",
        )?;

        // Clean up intermediate file
        let _ = std::fs::remove_file(&intermediate_path);
    }

    println!("\n✅ Wildfire data generation complete!");

    Ok(())
}

fn fetch_wildfire_data(args: &Args) -> Result<Vec<Feature>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let mut all_features = Vec::new();
    let mut offset = 0;

    let mut where_clauses = vec![
        format!("FIRE_YEAR >= {} AND FIRE_YEAR <= {}", args.start_year, args.end_year),
        format!("GIS_ACRES >= {}", args.min_acres),
    ];

    if args.wildfires_only {
        where_clauses.push("FEATURE_CA = 'Wildfire'".to_string());
    }

    let where_clause = where_clauses.join(" AND ");
    println!("🔍 Query filter: {}", where_clause);

    // Get count first
    let count_url = format!(
        "{}?where={}&returnCountOnly=true&f=json",
        NIFC_SERVICE_URL,
        urlencoding::encode(&where_clause)
    );

    let count_response: serde_json::Value = client.get(&count_url).send()?.json()?;
    let total_count = count_response["count"].as_u64().unwrap_or(0);
    println!("📈 Found {} matching fires", total_count);

    if total_count == 0 {
        return Ok(vec![]);
    }

    let max_to_fetch = if args.max_fires > 0 {
        args.max_fires.min(total_count as usize)
    } else {
        total_count as usize
    };

    let pb = ProgressBar::new(max_to_fetch as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} fires ({eta})")?
            .progress_chars("#>-"),
    );

    loop {
        let url = format!(
            "{}?where={}&outFields=*&resultOffset={}&resultRecordCount={}&f=json&outSR=4326",
            NIFC_SERVICE_URL,
            urlencoding::encode(&where_clause),
            offset,
            MAX_RECORDS_PER_REQUEST
        );

        let response: ArcGISResponse = client
            .get(&url)
            .send()
            .context("Failed to send request to NIFC API")?
            .json()
            .context("Failed to parse NIFC API response")?;

        let batch_size = response.features.len();
        if batch_size == 0 {
            break;
        }

        for arcgis_feature in response.features {
            if let Some(feature) = convert_to_geojson(&arcgis_feature) {
                all_features.push(feature);
                pb.inc(1);

                if args.max_fires > 0 && all_features.len() >= args.max_fires {
                    pb.finish_with_message("Limit reached");
                    return Ok(all_features);
                }
            }
        }

        offset += batch_size as u32;

        if response.exceeded_transfer_limit != Some(true) || offset as u64 >= total_count {
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    pb.finish_with_message("Download complete");
    Ok(all_features)
}

fn convert_to_geojson(arcgis: &ArcGISFeature) -> Option<Feature> {
    let geometry = arcgis.geometry.as_ref()?;
    let rings = geometry.rings.as_ref()?;

    if rings.is_empty() || rings[0].is_empty() {
        return None;
    }

    let coords: Vec<Vec<Vec<f64>>> = rings
        .iter()
        .map(|ring| {
            ring.iter()
                .filter_map(|coord| {
                    if coord.len() >= 2 {
                        Some(vec![coord[0], coord[1]])
                    } else {
                        None
                    }
                })
                .collect()
        })
        .filter(|ring: &Vec<Vec<f64>>| ring.len() >= 4)
        .collect();

    if coords.is_empty() {
        return None;
    }

    let attrs = &arcgis.attributes;

    let timestamp = if let Some(date_ms) = attrs.date_cur {
        let secs = date_ms / 1000;
        let nsecs = ((date_ms % 1000) * 1_000_000) as u32;
        if let Some(dt) = chrono::DateTime::from_timestamp(secs, nsecs) {
            if dt.year() >= 1990 && dt.year() <= 2050 {
                dt
            } else if let Some(year) = attrs.fire_year {
                Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
            } else {
                return None;
            }
        } else if let Some(year) = attrs.fire_year {
            Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
        } else {
            return None;
        }
    } else if let Some(year) = attrs.fire_year {
        Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
    } else {
        return None;
    };

    let mut properties = Map::new();
    properties.insert("timestamp".to_string(), json!(timestamp.to_rfc3339()));

    if let Some(ref name) = attrs.incident {
        properties.insert("name".to_string(), json!(name));
    }
    if let Some(year) = attrs.fire_year {
        properties.insert("year".to_string(), json!(year));
    }
    if let Some(acres) = attrs.gis_acres {
        properties.insert("acres".to_string(), json!(acres.round() as i64));
    }
    if let Some(ref agency) = attrs.agency {
        properties.insert("agency".to_string(), json!(agency));
    }
    if let Some(ref unit_id) = attrs.unit_id {
        properties.insert("unit_id".to_string(), json!(unit_id));
    }
    if let Some(ref category) = attrs.feature_category {
        properties.insert("fire_type".to_string(), json!(category));
    }
    if let Some(ref irwin_id) = attrs.irwin_id {
        properties.insert("irwin_id".to_string(), json!(irwin_id));
    }
    if let Some(id) = attrs.object_id {
        properties.insert("object_id".to_string(), json!(id));
    }

    let severity = match attrs.gis_acres {
        Some(acres) if acres >= 100_000.0 => "catastrophic",
        Some(acres) if acres >= 50_000.0 => "extreme",
        Some(acres) if acres >= 10_000.0 => "high",
        Some(acres) if acres >= 1_000.0 => "moderate",
        _ => "low",
    };
    properties.insert("severity".to_string(), json!(severity));

    Some(Feature {
        bbox: None,
        geometry: Some(Geometry::new(GeoValue::Polygon(coords))),
        id: None,
        properties: Some(properties),
        foreign_members: None,
    })
}

/// Fetch wildfire data directly to GeoParquet format (streaming)
fn fetch_wildfire_data_parquet(args: &Args, output: &PathBuf) -> Result<usize> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let mut where_clauses = vec![
        format!("FIRE_YEAR >= {} AND FIRE_YEAR <= {}", args.start_year, args.end_year),
        format!("GIS_ACRES >= {}", args.min_acres),
    ];

    if args.wildfires_only {
        where_clauses.push("FEATURE_CA = 'Wildfire'".to_string());
    }

    let where_clause = where_clauses.join(" AND ");
    println!("🔍 Query filter: {}", where_clause);

    // Get count first
    let count_url = format!(
        "{}?where={}&returnCountOnly=true&f=json",
        NIFC_SERVICE_URL,
        urlencoding::encode(&where_clause)
    );

    let count_response: serde_json::Value = client.get(&count_url).send()?.json()?;
    let total_count = count_response["count"].as_u64().unwrap_or(0);
    println!("📈 Found {} matching fires", total_count);

    if total_count == 0 {
        return Ok(0);
    }

    let max_to_fetch = if args.max_fires > 0 {
        args.max_fires.min(total_count as usize)
    } else {
        total_count as usize
    };

    // Create streaming Parquet writer
    let property_columns = vec![
        "name".to_string(),
        "year".to_string(),
        "acres".to_string(),
        "agency".to_string(),
        "unit_id".to_string(),
        "fire_type".to_string(),
        "irwin_id".to_string(),
        "object_id".to_string(),
        "severity".to_string(),
    ];
    let mut writer = StreamingPolygonParquetWriter::new(output, property_columns)?;

    let pb = ProgressBar::new(max_to_fetch as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} fires ({eta})")?
            .progress_chars("#>-"),
    );

    let mut offset = 0u32;

    loop {
        let url = format!(
            "{}?where={}&outFields=*&resultOffset={}&resultRecordCount={}&f=json&outSR=4326",
            NIFC_SERVICE_URL,
            urlencoding::encode(&where_clause),
            offset,
            MAX_RECORDS_PER_REQUEST
        );

        let response: ArcGISResponse = client
            .get(&url)
            .send()
            .context("Failed to send request to NIFC API")?
            .json()
            .context("Failed to parse NIFC API response")?;

        let batch_size = response.features.len();
        if batch_size == 0 {
            break;
        }

        for arcgis_feature in response.features {
            if let Some(polygon_record) = convert_to_polygon_record(&arcgis_feature) {
                writer.write_polygon(&polygon_record)?;
                pb.inc(1);

                if args.max_fires > 0 && writer.row_count() >= args.max_fires {
                    pb.finish_with_message("Limit reached");
                    return writer.finish();
                }
            }
        }

        offset += batch_size as u32;

        if response.exceeded_transfer_limit != Some(true) || offset as u64 >= total_count {
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    pb.finish_with_message("Download complete");
    writer.finish()
}

/// Convert ArcGIS feature to PolygonRecord for Parquet output
fn convert_to_polygon_record(arcgis: &ArcGISFeature) -> Option<PolygonRecord> {
    let geometry = arcgis.geometry.as_ref()?;
    let rings = geometry.rings.as_ref()?;

    if rings.is_empty() || rings[0].is_empty() {
        return None;
    }

    // Convert to our ring format: Vec<Vec<[f64; 2]>>
    let polygon_rings: Vec<Vec<[f64; 2]>> = rings
        .iter()
        .map(|ring| {
            ring.iter()
                .filter_map(|coord| {
                    if coord.len() >= 2 {
                        Some([coord[0], coord[1]])
                    } else {
                        None
                    }
                })
                .collect()
        })
        .filter(|ring: &Vec<[f64; 2]>| ring.len() >= 4)
        .collect();

    if polygon_rings.is_empty() {
        return None;
    }

    let attrs = &arcgis.attributes;

    let timestamp: DateTime<Utc> = if let Some(date_ms) = attrs.date_cur {
        let secs = date_ms / 1000;
        let nsecs = ((date_ms % 1000) * 1_000_000) as u32;
        if let Some(dt) = chrono::DateTime::from_timestamp(secs, nsecs) {
            if dt.year() >= 1990 && dt.year() <= 2050 {
                dt
            } else if let Some(year) = attrs.fire_year {
                Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
            } else {
                return None;
            }
        } else if let Some(year) = attrs.fire_year {
            Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
        } else {
            return None;
        }
    } else if let Some(year) = attrs.fire_year {
        Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
    } else {
        return None;
    };

    let mut properties = Map::new();

    if let Some(ref name) = attrs.incident {
        properties.insert("name".to_string(), json!(name));
    }
    if let Some(year) = attrs.fire_year {
        properties.insert("year".to_string(), json!(year));
    }
    if let Some(acres) = attrs.gis_acres {
        properties.insert("acres".to_string(), json!(acres.round() as i64));
    }
    if let Some(ref agency) = attrs.agency {
        properties.insert("agency".to_string(), json!(agency));
    }
    if let Some(ref unit_id) = attrs.unit_id {
        properties.insert("unit_id".to_string(), json!(unit_id));
    }
    if let Some(ref category) = attrs.feature_category {
        properties.insert("fire_type".to_string(), json!(category));
    }
    if let Some(ref irwin_id) = attrs.irwin_id {
        properties.insert("irwin_id".to_string(), json!(irwin_id));
    }
    if let Some(id) = attrs.object_id {
        properties.insert("object_id".to_string(), json!(id));
    }

    let severity = match attrs.gis_acres {
        Some(acres) if acres >= 100_000.0 => "catastrophic",
        Some(acres) if acres >= 50_000.0 => "extreme",
        Some(acres) if acres >= 10_000.0 => "high",
        Some(acres) if acres >= 1_000.0 => "moderate",
        _ => "low",
    };
    properties.insert("severity".to_string(), json!(severity));

    Some(PolygonRecord::new(polygon_rings, timestamp, properties))
}


