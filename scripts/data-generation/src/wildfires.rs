//! Generate wildfire perimeter data from NIFC (National Interagency Fire Center)
//!
//! Downloads real wildfire perimeter polygons from the NIFC ArcGIS REST API.
//! Data source: https://data-nifc.opendata.arcgis.com/
//!
//! REST API endpoint:
//! https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/
//! InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0

mod common;

use anyhow::{Context, Result};
use chrono::{Datelike, TimeZone, Utc};
use clap::Parser;
use geojson::{Feature, Geometry, Value as GeoValue};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use serde_json::{json, Map};
use std::path::PathBuf;

/// NIFC ArcGIS Feature Service base URL
const NIFC_SERVICE_URL: &str = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query";

/// Maximum records per request (ArcGIS limit)
const MAX_RECORDS_PER_REQUEST: u32 = 2000;

#[derive(Parser, Debug)]
#[command(name = "generate-wildfire-data")]
#[command(about = "Download wildfire perimeter polygons from NIFC")]
struct Args {
    /// Output GeoJSON file
    #[arg(short, long, default_value = "wildfires.geojson")]
    output: PathBuf,

    /// Start year (inclusive)
    #[arg(long, default_value = "2020")]
    start_year: u32,

    /// End year (inclusive)
    #[arg(long, default_value = "2023")]
    end_year: u32,

    /// Minimum fire size in acres (filters out small fires)
    #[arg(long, default_value = "1000")]
    min_acres: u32,

    /// Maximum number of fires to fetch (0 = no limit)
    #[arg(long, default_value = "0")]
    max_fires: usize,

    /// Only fetch fires from specific states (comma-separated, e.g., "CA,OR,WA")
    #[arg(long)]
    states: Option<String>,

    /// Filter to only wildfires (exclude prescribed burns)
    #[arg(long, default_value = "true")]
    wildfires_only: bool,
}

/// ArcGIS query response structure
#[derive(Deserialize, Debug)]
struct ArcGISResponse {
    features: Vec<ArcGISFeature>,
    #[serde(rename = "exceededTransferLimit")]
    exceeded_transfer_limit: Option<bool>,
}

/// ArcGIS feature structure
#[derive(Deserialize, Debug)]
struct ArcGISFeature {
    attributes: ArcGISAttributes,
    geometry: Option<ArcGISGeometry>,
}

/// ArcGIS attributes - uses flexible deserialization since NIFC returns mixed types
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
    date_cur: Option<i64>, // Unix timestamp in milliseconds or parsed from string
    #[serde(rename = "FEATURE_CA")]
    feature_category: Option<String>, // "Wildfire", "Prescribed Fire", "Unknown"
    #[serde(rename = "IRWINID")]
    irwin_id: Option<String>,
    #[serde(rename = "MAP_METHOD")]
    map_method: Option<String>,
    #[serde(rename = "COMMENTS")]
    comments: Option<String>,
}

/// Flexible deserializer for i64 that handles numbers or strings
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

/// Flexible deserializer for i32 that handles numbers or strings
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

/// Flexible deserializer for f64 that handles numbers or strings
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

/// Custom deserializer for dates in YYYYMMDDHHMMSS format or Unix timestamps
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
            // Try parsing as number first
            if let Ok(n) = s.parse::<i64>() {
                return Ok(parse_date_value(n));
            }
            // Try parsing YYYYMMDDHHMMSS string format
            if s.len() == 14 {
                if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&s, "%Y%m%d%H%M%S") {
                    return Ok(Some(dt.and_utc().timestamp_millis()));
                }
            }
            // Try parsing YYYYMMDD string format
            if s.len() == 8 {
                if let Ok(date) = chrono::NaiveDate::parse_from_str(&s, "%Y%m%d") {
                    let datetime = date.and_hms_opt(0, 0, 0).unwrap();
                    return Ok(Some(datetime.and_utc().timestamp_millis()));
                }
            }
            Ok(None)
        }
        DateValue::Null => Ok(None),
    }
}

/// Parse a date value that could be either:
/// - A Unix timestamp in milliseconds (large number > 1 billion)
/// - A YYYYMMDDHHMMSS format as integer (e.g., 20230723164511)
/// - A YYYYMMDD format as integer (e.g., 20230723)
fn parse_date_value(n: i64) -> Option<i64> {
    // If it looks like a YYYYMMDDHHMMSS format (14 digits starting with 19xx or 20xx)
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
    // If it looks like a YYYYMMDD format (8 digits starting with 19xx or 20xx)
    if n >= 19000101 && n <= 20991231 {
        let year = (n / 10000) as i32;
        let month = ((n % 10000) / 100) as u32;
        let day = (n % 100) as u32;
        if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, day) {
            let datetime = date.and_hms_opt(0, 0, 0).unwrap();
            return Some(datetime.and_utc().timestamp_millis());
        }
    }
    // If it's a very large number, treat as Unix timestamp in milliseconds
    if n > 1000000000000 {
        return Some(n);
    }
    // If it's a moderately large number, treat as Unix timestamp in seconds
    if n > 1000000000 {
        return Some(n * 1000);
    }
    None
}

/// ArcGIS geometry (rings for polygons)
#[derive(Deserialize, Debug)]
struct ArcGISGeometry {
    rings: Option<Vec<Vec<Vec<f64>>>>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🔥 NIFC Wildfire Data Generator");
    println!("================================\n");
    println!("📊 Fetching wildfires from {} to {}", args.start_year, args.end_year);
    println!("   Minimum size: {} acres", args.min_acres);
    if args.wildfires_only {
        println!("   Filter: Wildfires only (excluding prescribed burns)");
    }
    if let Some(ref states) = args.states {
        println!("   States: {}", states);
    }
    println!();

    let features = fetch_wildfire_data(&args)?;

    if features.is_empty() {
        println!("\n⚠️  No features found matching criteria");
        return Ok(());
    }

    println!("\n💾 Writing {} features to {:?}...", features.len(), args.output);
    common::write_geojson(features, &args.output)?;

    println!("\n✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output wildfires.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 3 \\");
    println!("             --max-zoom 12 \\");
    println!("             --compression gzip");

    Ok(())
}

fn fetch_wildfire_data(args: &Args) -> Result<Vec<Feature>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let mut all_features = Vec::new();
    let mut offset = 0;

    // Build the where clause
    let mut where_clauses = vec![
        format!("FIRE_YEAR >= {} AND FIRE_YEAR <= {}", args.start_year, args.end_year),
        format!("GIS_ACRES >= {}", args.min_acres),
    ];

    if args.wildfires_only {
        where_clauses.push("FEATURE_CA = 'Wildfire'".to_string());
    }

    // Note: State filtering would require parsing UNIT_ID which has format like "CARMP"
    // For simplicity, we'll do post-processing if needed

    let where_clause = where_clauses.join(" AND ");
    println!("🔍 Query filter: {}", where_clause);

    // First, get the count
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

        // Check if we've reached the transfer limit or fetched all records
        if response.exceeded_transfer_limit != Some(true) || offset as u64 >= total_count {
            break;
        }

        // Small delay to be nice to the server
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

    // Convert ArcGIS rings to GeoJSON Polygon coordinates
    // ArcGIS uses rings array, first is exterior, rest are holes
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
        .filter(|ring: &Vec<Vec<f64>>| ring.len() >= 4) // Valid polygon needs at least 4 points
        .collect();

    if coords.is_empty() {
        return None;
    }

    let attrs = &arcgis.attributes;

    // Parse the timestamp with fallback to fire year
    let timestamp = if let Some(date_ms) = attrs.date_cur {
        // Convert milliseconds to DateTime
        let secs = date_ms / 1000;
        let nsecs = ((date_ms % 1000) * 1_000_000) as u32;
        if let Some(dt) = chrono::DateTime::from_timestamp(secs, nsecs) {
            // Validate the date is reasonable (1990-2050)
            if dt.year() >= 1990 && dt.year() <= 2050 {
                dt
            } else if let Some(year) = attrs.fire_year {
                // Date is invalid, use fire year instead
                Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap() // Use mid-year as estimate
            } else {
                return None;
            }
        } else if let Some(year) = attrs.fire_year {
            // Couldn't parse date, use fire year
            Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
        } else {
            return None;
        }
    } else if let Some(year) = attrs.fire_year {
        // No date available, use fire year (mid-year as estimate)
        Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
    } else {
        return None;
    };

    // Build properties
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
    if let Some(ref method) = attrs.map_method {
        properties.insert("map_method".to_string(), json!(method));
    }
    if let Some(id) = attrs.object_id {
        properties.insert("object_id".to_string(), json!(id));
    }

    // Calculate severity based on acres
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
