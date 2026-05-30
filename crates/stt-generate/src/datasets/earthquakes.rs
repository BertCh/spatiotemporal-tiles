//! Generate earthquake data from USGS API
//!
//! Data source: https://earthquake.usgs.gov/fdsnws/event/1/

use crate::common::{self, PointRecord, PropertyColumn, StreamingParquetWriter};
use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Map};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Generate earthquake data from USGS")]
pub struct Args {
    /// Output file (.stt, .geojson, or .csv)
    #[arg(short, long, default_value = "earthquakes.stt")]
    pub output: PathBuf,

    /// Start date (YYYY-MM-DD)
    #[arg(long, default_value = "2020-01-01")]
    pub start_date: String,

    /// End date (YYYY-MM-DD)
    #[arg(long, default_value = "2024-12-31")]
    pub end_date: String,

    /// Minimum magnitude
    #[arg(long, default_value = "4.0")]
    pub min_magnitude: f64,

    /// Skip stt-build step (just output GeoJSON/CSV)
    #[arg(long)]
    pub skip_build: bool,

    /// Emit a server-aggregated summary tier alongside the raw tier.
    /// The summary tier renders 100K+ events as ~hundreds of H3 hex bins
    /// — the only practical way to visualise 100M+ point datasets in real
    /// time.
    #[arg(long)]
    pub summary_tier: bool,
}

#[derive(Debug, Deserialize)]
struct UsgsResponse {
    features: Vec<UsgsFeature>,
}

#[derive(Debug, Deserialize)]
struct UsgsFeature {
    geometry: UsgsGeometry,
    properties: UsgsProperties,
}

#[derive(Debug, Deserialize)]
struct UsgsGeometry {
    coordinates: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct UsgsProperties {
    mag: f64,
    place: String,
    time: i64,
    #[serde(rename = "type")]
    event_type: String,
    title: String,
}

pub fn run(args: Args) -> Result<()> {
    println!("🌍 Earthquake Data Generator");
    println!("============================\n");

    println!("📡 Fetching earthquake data from USGS...");
    println!("   Date range: {} to {}", args.start_date, args.end_date);
    println!("   Min magnitude: {}", args.min_magnitude);

    // Determine intermediate output format (prefer Parquet for efficiency)
    let intermediate_path = if args.output.extension().map(|e| e == "stt").unwrap_or(false) {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    let use_parquet = common::is_parquet_output(&intermediate_path);
    let use_csv = common::is_csv_output(&intermediate_path);
    
    if use_parquet {
        println!("📄 Using streaming GeoParquet output (efficient)");
    } else if use_csv {
        println!("📄 Using streaming CSV output");
    }

    // USGS API has a limit, so we fetch in yearly chunks
    let start_year = args.start_date[..4].parse::<i32>()?;
    let end_year = args.end_date[..4].parse::<i32>()?;

    // Typed columns so magnitude/depth/mag_band survive stt-build's
    // categorical-vs-numeric inference. magnitude and depth are used by the
    // showcase deck.gl layers as radius/elevation drivers; mag_band is the
    // categorical key for color-by-magnitude.
    let typed_columns = vec![
        PropertyColumn::numeric("magnitude"),
        PropertyColumn::numeric("depth"),
        PropertyColumn::string("place"),
        PropertyColumn::string("type"),
        PropertyColumn::string("title"),
        PropertyColumn::string("mag_band"),
    ];
    let property_column_names: Vec<String> =
        typed_columns.iter().map(|c| c.name.clone()).collect();

    let mut all_features = Vec::new();
    let mut parquet_writer = if use_parquet {
        Some(StreamingParquetWriter::with_columns(
            &intermediate_path,
            typed_columns.clone(),
        )?)
    } else {
        None
    };
    let mut csv_writer = if use_csv {
        Some(common::StreamingCsvWriter::new(
            &intermediate_path,
            property_column_names,
        )?)
    } else {
        None
    };

    for year in start_year..=end_year {
        let year_start = format!("{}-01-01", year);
        let year_end = format!("{}-12-31", year);

        println!("\n📅 Fetching data for {}...", year);

        let url = format!(
            "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={}&endtime={}&minmagnitude={}",
            year_start, year_end, args.min_magnitude
        );

        let response = reqwest::blocking::get(&url)?;
        let data: UsgsResponse = response.json()?;

        println!("   ✓ Fetched {} earthquakes", data.features.len());

        for usgs_feature in data.features {
            let (lon, lat, timestamp, properties) = extract_usgs_data(usgs_feature)?;

            if let Some(ref mut writer) = parquet_writer {
                let record = PointRecord::new(lon, lat, timestamp, properties);
                writer.write_point(&record)?;
            } else if let Some(ref mut writer) = csv_writer {
                writer.write_point(lon, lat, timestamp, &properties)?;
            } else {
                let feature = common::create_point_feature(lon, lat, timestamp, properties);
                all_features.push(feature);
            }
        }
    }

    let total_count = if let Some(writer) = parquet_writer.take() {
        writer.finish()?
    } else if let Some(writer) = csv_writer.take() {
        writer.finish()?
    } else {
        let count = all_features.len();
        println!("\n💾 Writing GeoJSON...");
        common::write_geojson(all_features, &intermediate_path)?;
        count
    };

    println!("\n📊 Total earthquakes: {}", total_count);

    // Build STT if output is .stt
    if args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build {
        let summary = if args.summary_tier {
            // Aggregate the same magnitude column the raw layer uses for
            // radius, so the summary tier can drive a "biggest quake in
            // this hex" overlay or a count heatmap from one archive.
            Some(common::SttBuildSummaryOptions {
                scheme: "h3".to_string(),
                min_zoom: Some(0),
                max_zoom: Some(4),
                columns: "magnitude:mean,magnitude:max".to_string(),
            })
        } else {
            None
        };
        common::run_stt_build_with_full_options(common::SttBuildOptions {
            input: intermediate_path.clone(),
            output: args.output.clone(),
            time_field: "timestamp".to_string(),
            end_time_field: None,
            min_zoom: 0,
            max_zoom: 10,
            compression: "zstd".to_string(),
            // 1d buckets: M4.0+ earthquakes are sparse globally (~50/day), so
            // the stt-build default of 1h produced ~350K nearly-empty index
            // entries (21 MB directory for an 87 MB archive) and a slow first
            // load + janky pan/zoom. Daily buckets cut the index ~24x and
            // still leave ~30 buckets inside the showcase's 30-day animation
            // window.
            temporal_bucket: Some("1d".to_string()),
            temporal_lod: None,
            summary_sub_buckets: None,
            // Skip single-feature tiles. Globally sparse points produce a
            // long tail of 1-feature deep-zoom tiles whose Arrow IPC + zstd
            // overhead dominates the payload (at z=10, 93% of tiles held
            // exactly 1 feature, each ~790 bytes for ~85 bytes of data).
            // The reader's parent-fallback surfaces the same features from
            // their shallower-zoom ancestors — visually identical for point
            // markers that scale by magnitude in pixels.
            min_features_per_tile: Some(2),
            summary,
        })?;

        // Clean up intermediate file
        let _ = std::fs::remove_file(&intermediate_path);
    }

    println!("\n✅ Earthquake data generation complete!");

    Ok(())
}

fn extract_usgs_data(usgs: UsgsFeature) -> Result<(f64, f64, DateTime<Utc>, Map<String, serde_json::Value>)> {
    let lon = usgs.geometry.coordinates[0];
    let lat = usgs.geometry.coordinates[1];
    let depth = usgs.geometry.coordinates.get(2).copied().unwrap_or(0.0);

    let timestamp = DateTime::from_timestamp_millis(usgs.properties.time)
        .unwrap_or_else(|| Utc::now());

    let mut properties = Map::new();
    properties.insert("magnitude".to_string(), json!(usgs.properties.mag));
    properties.insert("depth".to_string(), json!(depth));
    properties.insert("place".to_string(), json!(usgs.properties.place));
    properties.insert("type".to_string(), json!(usgs.properties.event_type));
    properties.insert("title".to_string(), json!(usgs.properties.title));
    // Categorical band for color-by-magnitude in the showcase. The bands map
    // to a 5-stop reds palette in datasets.ts. Lexicographic order matches
    // the palette order (BTreeMap keys stay sorted through stt-build).
    properties.insert("mag_band".to_string(), json!(mag_band(usgs.properties.mag)));

    Ok((lon, lat, timestamp, properties))
}

/// Bucket magnitude into legend-aligned bands. Bands are zero-padded to keep
/// the BTreeMap order stable across builds (1-..., 2-..., etc.).
fn mag_band(mag: f64) -> &'static str {
    if mag < 5.0 {
        "1-M4.5-5"
    } else if mag < 6.0 {
        "2-M5-6"
    } else if mag < 7.0 {
        "3-M6-7"
    } else if mag < 8.0 {
        "4-M7-8"
    } else {
        "5-M8+"
    }
}


