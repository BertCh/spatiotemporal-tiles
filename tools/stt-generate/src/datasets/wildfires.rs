//! Generate wildfire perimeter data from NIFC (National Interagency Fire Center)
//!
//! Data source: https://data-nifc.opendata.arcgis.com/

use crate::common::{self, PolygonRecord, PropertyColumn, StreamingPolygonParquetWriter};
use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use clap::Parser;
use geojson::{Feature, Geometry, Value as GeoValue};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use serde_json::{json, Map, Value as JsonValue};
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

    let where_clause = nifc_where_clause(args);
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
            let features = convert_to_geojson(&arcgis_feature);
            if features.is_empty() {
                continue;
            }
            all_features.extend(features);
            pb.inc(1);

            if args.max_fires > 0 && all_features.len() >= args.max_fires {
                pb.finish_with_message("Limit reached");
                return Ok(all_features);
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

fn convert_to_geojson(arcgis: &ArcGISFeature) -> Vec<Feature> {
    let Some(geometry) = arcgis.geometry.as_ref() else {
        return Vec::new();
    };
    let Some(rings) = geometry.rings.as_ref() else {
        return Vec::new();
    };

    if rings.is_empty() || rings[0].is_empty() {
        return Vec::new();
    }

    // Same winding-aware split as the parquet path: one single-exterior polygon
    // feature per fire part (avoids the flattened-multipolygon spike).
    let polygons = arcgis_rings_to_polygons(rings);
    if polygons.is_empty() {
        return Vec::new();
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
                return Vec::new();
            }
        } else if let Some(year) = attrs.fire_year {
            Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
        } else {
            return Vec::new();
        }
    } else if let Some(year) = attrs.fire_year {
        Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
    } else {
        return Vec::new();
    };

    let mut properties = build_fire_properties(attrs);
    properties.insert("timestamp".to_string(), json!(timestamp.to_rfc3339()));

    polygons
        .into_iter()
        .map(|poly| {
            let coords: Vec<Vec<Vec<f64>>> = poly
                .into_iter()
                .map(|ring| ring.into_iter().map(|p| vec![p[0], p[1]]).collect())
                .collect();
            Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeoValue::Polygon(coords))),
                id: None,
                properties: Some(properties.clone()),
                foreign_members: None,
            }
        })
        .collect()
}

/// The ArcGIS `where` filter both fetch paths query with.
///
/// Two bugs lived in the duplicated copies of this, fixed 2026-07-29:
///
/// - `FIRE_YEAR` is an `esriFieldTypeString` on this service, so the numeric
///   range was a STRING comparison. It read as a year filter and is not one.
///   `FIRE_YEAR_INT` (`esriFieldTypeSmallInteger`) is the same value, typed.
/// - `FEATURE_CA = 'Wildfire'` matched one of the FIVE values the field
///   carries. Probing the service on 2026-07-29 returns `Wildfire`,
///   `Wildfire Final Fire Perimeter`, `Wildfire for Resource Benefit`,
///   `Prescribed Fire`, and null — so the equality dropped the two dominant
///   wildfire spellings, i.e. `--wildfires-only` excluded most wildfires.
///   `LIKE 'Wildfire%'` keeps all three and still excludes prescribed burns.
///
/// ⚠️ Fixing the filter does NOT make this dataset reproducible. The shipped
/// `examples/showcase/public/data/wildfires` archive holds ~460 fires; the
/// same service on 2026-07-29 returns 98,168 records overall but only 297 for
/// 2020–2023, of which 10 clear `--min-acres 1000` as wildfires. Upstream
/// coverage for those years collapsed, so a regeneration produces a nearly
/// empty archive and would DESTROY the shipped one. Verify the fetch count
/// against the existing archive before overwriting it.
fn nifc_where_clause(args: &Args) -> String {
    let mut where_clauses = vec![
        format!(
            "FIRE_YEAR_INT >= {} AND FIRE_YEAR_INT <= {}",
            args.start_year, args.end_year
        ),
        format!("GIS_ACRES >= {}", args.min_acres),
    ];

    if args.wildfires_only {
        where_clauses.push("FEATURE_CA LIKE 'Wildfire%'".to_string());
    }

    where_clauses.join(" AND ")
}

/// Fetch wildfire data directly to GeoParquet format (streaming)
fn fetch_wildfire_data_parquet(args: &Args, output: &PathBuf) -> Result<usize> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let where_clause = nifc_where_clause(args);
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
        PropertyColumn::string("name"),
        PropertyColumn::string("year"),
        PropertyColumn::numeric("acres"),
        PropertyColumn::string("agency"),
        PropertyColumn::string("unit_id"),
        PropertyColumn::string("fire_type"),
        PropertyColumn::string("irwin_id"),
        PropertyColumn::numeric("object_id"),
        PropertyColumn::string("severity"),
    ];
    let mut writer = StreamingPolygonParquetWriter::with_columns(output, property_columns)?;

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
            let records = convert_to_polygon_records(&arcgis_feature);
            if records.is_empty() {
                continue;
            }
            for polygon_record in &records {
                writer.write_polygon(polygon_record)?;
            }
            // Progress tracks fires, not the (≥1) polygon parts each splits into.
            pb.inc(1);

            if args.max_fires > 0 && writer.row_count() >= args.max_fires {
                pb.finish_with_message("Limit reached");
                return writer.finish();
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

/// Shoelace signed area of a closed ring (lon/lat as x/y). Positive ⇒
/// counter-clockwise. ArcGIS winds **exterior** rings clockwise (negative) and
/// **holes** counter-clockwise (positive) — the only signal distinguishing them
/// in the flat `rings` array.
fn ring_signed_area(ring: &[[f64; 2]]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for w in ring.windows(2) {
        a += w[0][0] * w[1][1] - w[1][0] * w[0][1];
    }
    a / 2.0
}

/// Split an ArcGIS `rings` array into one polygon per **exterior** ring.
///
/// ArcGIS Esri-JSON packs every ring of a (possibly multi-part) feature into a
/// single flat list, distinguished only by winding: clockwise = an exterior
/// ring that starts a new polygon, counter-clockwise = a hole of the most
/// recent exterior. The previous code copied the whole list into one polygon,
/// so every extra exterior of a multi-part fire (e.g. the 11 perimeters of the
/// SCU Lightning Complex) became a phantom "hole" — the renderer then bridged
/// the disjoint exteriors with spanning triangles (the giant spike artifact).
///
/// Returns one `Vec<ring>` per polygon, exterior first then its holes. A
/// leading hole with no exterior yet is promoted to an exterior so no vertices
/// are silently dropped.
fn arcgis_rings_to_polygons(rings: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<[f64; 2]>>> {
    let mut polygons: Vec<Vec<Vec<[f64; 2]>>> = Vec::new();
    for ring in rings {
        let pts: Vec<[f64; 2]> = ring
            .iter()
            .filter_map(|coord| {
                if coord.len() >= 2 {
                    Some([coord[0], coord[1]])
                } else {
                    None
                }
            })
            .collect();
        // A valid closed ring needs ≥ 4 vertices (first == last).
        if pts.len() < 4 {
            continue;
        }
        // CCW (positive area) is a hole of the current polygon; CW (or
        // degenerate) starts a new exterior.
        if ring_signed_area(&pts) > 0.0 {
            if let Some(current) = polygons.last_mut() {
                current.push(pts);
                continue;
            }
        }
        polygons.push(vec![pts]);
    }
    polygons
}

/// Convert one ArcGIS feature to PolygonRecords for Parquet output — one record
/// per exterior ring (multi-part fires become multiple single-exterior polygon
/// features that share the same timestamp + properties).
fn convert_to_polygon_records(arcgis: &ArcGISFeature) -> Vec<PolygonRecord> {
    let Some(geometry) = arcgis.geometry.as_ref() else {
        return Vec::new();
    };
    let Some(rings) = geometry.rings.as_ref() else {
        return Vec::new();
    };

    if rings.is_empty() || rings[0].is_empty() {
        return Vec::new();
    }

    let polygons = arcgis_rings_to_polygons(rings);
    if polygons.is_empty() {
        return Vec::new();
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
                return Vec::new();
            }
        } else if let Some(year) = attrs.fire_year {
            Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
        } else {
            return Vec::new();
        }
    } else if let Some(year) = attrs.fire_year {
        Utc.with_ymd_and_hms(year, 7, 1, 0, 0, 0).unwrap()
    } else {
        return Vec::new();
    };

    let properties = build_fire_properties(attrs);

    // One single-exterior polygon feature per part; share timestamp + properties.
    polygons
        .into_iter()
        .map(|rings| PolygonRecord::new(rings, timestamp, properties.clone()))
        .collect()
}

/// Build the shared property map for a fire feature.
fn build_fire_properties(attrs: &ArcGISAttributes) -> Map<String, JsonValue> {
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
    properties
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clockwise unit square (ArcGIS exterior winding, negative signed area).
    fn cw_square(ox: f64, oy: f64, s: f64) -> Vec<Vec<f64>> {
        vec![
            vec![ox, oy],
            vec![ox, oy + s],
            vec![ox + s, oy + s],
            vec![ox + s, oy],
            vec![ox, oy],
        ]
    }

    /// Counter-clockwise square (ArcGIS hole winding, positive signed area).
    fn ccw_square(ox: f64, oy: f64, s: f64) -> Vec<Vec<f64>> {
        vec![
            vec![ox, oy],
            vec![ox + s, oy],
            vec![ox + s, oy + s],
            vec![ox, oy + s],
            vec![ox, oy],
        ]
    }

    #[test]
    fn signed_area_sign_matches_winding() {
        let cw: Vec<[f64; 2]> = cw_square(0.0, 0.0, 1.0).iter().map(|c| [c[0], c[1]]).collect();
        let ccw: Vec<[f64; 2]> = ccw_square(0.0, 0.0, 1.0).iter().map(|c| [c[0], c[1]]).collect();
        assert!(ring_signed_area(&cw) < 0.0, "CW exterior must be negative");
        assert!(ring_signed_area(&ccw) > 0.0, "CCW hole must be positive");
    }

    #[test]
    fn exterior_with_hole_stays_one_polygon() {
        // CW exterior followed by a CCW hole inside it → one polygon, two rings.
        let rings = vec![cw_square(0.0, 0.0, 4.0), ccw_square(1.0, 1.0, 1.0)];
        let polys = arcgis_rings_to_polygons(&rings);
        assert_eq!(polys.len(), 1);
        assert_eq!(polys[0].len(), 2, "exterior + its hole");
    }

    #[test]
    fn multiple_exteriors_split_into_separate_polygons() {
        // The SCU-complex shape: several CW exteriors (each a distinct fire
        // perimeter) plus a hole on the first. Must NOT collapse into one
        // polygon whose extra exteriors are treated as holes.
        let rings = vec![
            cw_square(0.0, 0.0, 4.0),   // exterior A
            ccw_square(1.0, 1.0, 1.0),  // hole of A
            cw_square(10.0, 0.0, 3.0),  // exterior B (disjoint)
            cw_square(20.0, 0.0, 2.0),  // exterior C (disjoint)
        ];
        let polys = arcgis_rings_to_polygons(&rings);
        assert_eq!(polys.len(), 3, "three distinct exteriors");
        assert_eq!(polys[0].len(), 2, "first polygon keeps its hole");
        assert_eq!(polys[1].len(), 1);
        assert_eq!(polys[2].len(), 1);
    }

    #[test]
    fn leading_hole_is_promoted_not_dropped() {
        // A CCW ring with no exterior yet becomes its own polygon rather than
        // silently dropping vertices.
        let rings = vec![ccw_square(0.0, 0.0, 1.0)];
        let polys = arcgis_rings_to_polygons(&rings);
        assert_eq!(polys.len(), 1);
        assert_eq!(polys[0].len(), 1);
    }
}

