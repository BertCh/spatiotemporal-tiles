//! Generate COVID-19 case data from New York Times dataset
//!
//! Data source: https://github.com/nytimes/covid-19-data

mod common;

use anyhow::Result;
use chrono::NaiveDate;
use clap::Parser;
use csv::ReaderBuilder;
use serde::Deserialize;
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-covid-data")]
#[command(about = "Generate COVID-19 case data for STT showcase")]
struct Args {
    /// Output file (use .csv for streaming output, .geojson for JSON)
    #[arg(short, long, default_value = "covid-cases.csv")]
    output: PathBuf,

    /// Use cached data (skip download)
    #[arg(long)]
    cached: bool,
}

#[derive(Debug, Deserialize)]
struct CovidRecord {
    date: String,
    county: String,
    state: String,
    fips: String,
    cases: Option<u32>,
    deaths: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct CountyLocation {
    fips: String,
    county: String,
    state: String,
    lat: f64,
    lon: f64,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("🦠 COVID-19 Data Generator");
    println!("=============================\n");

    let use_csv = common::is_csv_output(&args.output);
    if use_csv {
        println!("📄 Using streaming CSV output (memory-efficient)");
    }

    // Download COVID data
    let covid_csv = PathBuf::from("data/us-counties.csv");
    if !args.cached {
        std::fs::create_dir_all("data")?;
        common::download_file(
            "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv",
            &covid_csv,
        )?;
    }

    // Download county coordinates
    let coords_csv = PathBuf::from("data/county-coords.csv");
    if !args.cached {
        // We'll use a simplified coordinate lookup
        // In practice, you'd download from a real source
        generate_county_coordinates(&coords_csv)?;
    }

    // Load county coordinates
    println!("\n📍 Loading county coordinates...");
    let coords = load_county_coordinates(&coords_csv)?;
    println!("✓ Loaded {} county locations", coords.len());

    // Process COVID data
    println!("\n📊 Processing COVID-19 case data...");
    let count = process_covid_data(&covid_csv, &coords, &args.output, use_csv)?;
    println!("✓ Generated {} features", count);

    println!("\n✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output covid-cases.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --min-zoom 0 \\");
    println!("             --max-zoom 14 \\");
    println!("             --compression gzip");

    Ok(())
}

fn load_county_coordinates(path: &PathBuf) -> Result<HashMap<String, CountyLocation>> {
    let mut coords = HashMap::new();
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    for result in rdr.deserialize() {
        let record: CountyLocation = result?;
        coords.insert(record.fips.clone(), record);
    }

    Ok(coords)
}

fn process_covid_data(
    covid_path: &PathBuf,
    coords: &HashMap<String, CountyLocation>,
    output_path: &PathBuf,
    use_csv: bool,
) -> Result<usize> {
    let file = File::open(covid_path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let property_columns = vec![
        "county".to_string(),
        "state".to_string(),
        "fips".to_string(),
        "cases".to_string(),
        "deaths".to_string(),
        "value".to_string(),
    ];

    let mut csv_writer = if use_csv {
        Some(common::StreamingCsvWriter::new(output_path, property_columns)?)
    } else {
        None
    };
    let mut features = Vec::new();
    let mut processed = 0;

    for result in rdr.deserialize() {
        let record: CovidRecord = result?;

        // Skip if we don't have coordinates
        if let Some(location) = coords.get(&record.fips) {
            let date = NaiveDate::parse_from_str(&record.date, "%Y-%m-%d")?;
            let timestamp = common::date_to_datetime(date);

            let cases = record.cases.unwrap_or(0);
            let deaths = record.deaths.unwrap_or(0);

            let mut properties = Map::new();
            properties.insert("county".to_string(), json!(record.county));
            properties.insert("state".to_string(), json!(record.state));
            properties.insert("fips".to_string(), json!(record.fips));
            properties.insert("cases".to_string(), json!(cases));
            properties.insert("deaths".to_string(), json!(deaths));
            properties.insert("value".to_string(), json!(cases)); // For visualization

            if use_csv {
                csv_writer.as_mut().unwrap().write_point(
                    location.lon,
                    location.lat,
                    timestamp,
                    &properties,
                )?;
            } else {
                let feature =
                    common::create_point_feature(location.lon, location.lat, timestamp, properties);
                features.push(feature);
            }

            processed += 1;

            if processed % 10000 == 0 {
                println!("  Processed {} records...", processed);
            }
        }
    }

    if use_csv {
        csv_writer.take().unwrap().finish()?;
        Ok(processed)
    } else {
        println!("\n💾 Writing output...");
        common::write_geojson(features, output_path)?;
        Ok(processed)
    }
}

fn generate_county_coordinates(path: &PathBuf) -> Result<()> {
    // This is a simplified version - in practice, download from:
    // https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html

    println!("📥 Downloading county coordinates...");

    let sample_data = r#"fips,county,state,lat,lon
06001,Alameda,California,37.6017,-121.7195
06013,Contra Costa,California,37.9161,-121.9511
06075,San Francisco,California,37.7749,-122.4194
36061,New York,New York,40.7128,-74.0060
48201,Harris,Texas,29.7604,-95.3698"#;

    std::fs::write(path, sample_data)?;

    println!("⚠️  Warning: Using sample coordinates. For production, download full dataset from:");
    println!("   https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html");

    Ok(())
}
