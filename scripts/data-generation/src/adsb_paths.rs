//! Process ADS-B data into flight path trajectories
//!
//! Takes real ADS-B point data (from OpenSky Network) and groups by aircraft
//! to create 3D flight paths (polylines) for visualization with the Trips layer.
//!
//! ## Data Flow
//! 1. Read point data from CSV (output from flights.rs or flights_historical.rs)
//! 2. Group points by aircraft (icao24)
//! 3. Sort each group by timestamp
//! 4. Create LineString paths with 3D coordinates [lon, lat, altitude]
//! 5. Output GeoJSON with per-coordinate timestamps for trips animation
//!
//! ## Usage Examples
//!
//! ```bash
//! # First collect real-time data using flights.rs
//! cargo run --release --bin generate-flight-data -- \
//!   --source opensky \
//!   --duration 3600 \
//!   --interval 30 \
//!   --bounds 25,-125,50,-65 \
//!   --output flights-points.geojson
//!
//! # Then convert points to paths
//! cargo run --release --bin generate-adsb-paths -- \
//!   --input flights-points.geojson \
//!   --output flights-paths.geojson \
//!   --min-points 5
//!
//! # Build STT file for trips layer
//! stt-build --input flights-paths.geojson --output flights-paths.stt \
//!           --time-field timestamp --end-time-field end_time \
//!           --min-zoom 0 --max-zoom 10 --compression gzip
//! ```

mod common;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use geojson::{Feature, FeatureCollection, GeoJson, Geometry, Value as GeoValue};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-adsb-paths")]
#[command(about = "Convert ADS-B point data into flight path trajectories")]
#[command(long_about = "
Process real ADS-B data and create flight path trajectories for Trips layer visualization.

Takes point data collected from OpenSky Network and groups by aircraft (icao24)
to create 3D flight paths with per-coordinate timestamps.

INPUT FORMATS:
  - GeoJSON FeatureCollection (output from generate-flight-data)
  - CSV with columns: lon, lat, timestamp, icao24, altitude, ...

OUTPUT:
  GeoJSON with LineString features, each representing one flight path.
  Each feature includes:
  - 3D coordinates: [lon, lat, altitude_meters]
  - timestamps array: Unix timestamps for each coordinate
  - Properties: icao24, callsign, country, start_time, end_time

USAGE:
  1. Collect real-time data:
     cargo run --release --bin generate-flight-data -- --source opensky --duration 3600

  2. Convert to paths:
     cargo run --release --bin generate-adsb-paths -- --input flights.geojson --output paths.geojson

  3. Build STT:
     stt-build --input paths.geojson --output paths.stt --time-field timestamp
")]
struct Args {
    /// Input file (GeoJSON or CSV with point data)
    #[arg(short, long)]
    input: PathBuf,

    /// Output GeoJSON file
    #[arg(short, long, default_value = "adsb-paths.geojson")]
    output: PathBuf,

    /// Minimum number of points to form a valid path
    #[arg(long, default_value = "3")]
    min_points: usize,

    /// Maximum gap in seconds between points before starting a new path
    /// (splits flight if aircraft disappears from radar for too long)
    #[arg(long, default_value = "300")]
    max_gap_seconds: i64,

    /// Altitude scale factor (1.0 = meters, 0.3048 = feet to meters)
    /// OpenSky provides altitude in meters, so default is 1.0
    #[arg(long, default_value = "1.0")]
    altitude_scale: f64,

    /// Maximum number of paths to output (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_paths: usize,

    /// Minimum flight duration in seconds
    #[arg(long, default_value = "60")]
    min_duration_seconds: i64,
}

/// A single position observation for an aircraft
#[derive(Debug, Clone)]
struct AircraftPosition {
    timestamp: DateTime<Utc>,
    timestamp_ms: i64,
    lon: f64,
    lat: f64,
    altitude: f64, // meters
    speed: Option<f64>,
    heading: Option<f64>,
    callsign: Option<String>,
    country: Option<String>,
}

/// A complete flight path for one aircraft
#[derive(Debug)]
struct FlightPath {
    icao24: String,
    positions: Vec<AircraftPosition>,
}

impl FlightPath {
    fn duration_seconds(&self) -> i64 {
        if self.positions.len() < 2 {
            return 0;
        }
        let start = self.positions.first().unwrap().timestamp_ms;
        let end = self.positions.last().unwrap().timestamp_ms;
        (end - start) / 1000
    }

    fn start_time(&self) -> Option<DateTime<Utc>> {
        self.positions.first().map(|p| p.timestamp)
    }

    fn end_time(&self) -> Option<DateTime<Utc>> {
        self.positions.last().map(|p| p.timestamp)
    }

    fn callsign(&self) -> Option<&str> {
        // Find first non-empty callsign
        self.positions
            .iter()
            .find_map(|p| p.callsign.as_deref())
    }

    fn country(&self) -> Option<&str> {
        self.positions
            .first()
            .and_then(|p| p.country.as_deref())
    }

    fn avg_speed(&self) -> Option<f64> {
        let speeds: Vec<f64> = self.positions.iter().filter_map(|p| p.speed).collect();
        if speeds.is_empty() {
            None
        } else {
            Some(speeds.iter().sum::<f64>() / speeds.len() as f64)
        }
    }

    fn avg_altitude(&self) -> f64 {
        let alts: Vec<f64> = self.positions.iter().map(|p| p.altitude).collect();
        if alts.is_empty() {
            0.0
        } else {
            alts.iter().sum::<f64>() / alts.len() as f64
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("✈️  ADS-B Path Generator");
    println!("========================\n");
    println!("📂 Input: {}", args.input.display());
    println!("📄 Output: {}", args.output.display());
    println!("📊 Min points per path: {}", args.min_points);
    println!("⏱️  Max gap between points: {} seconds", args.max_gap_seconds);
    println!("📏 Altitude scale: {}", args.altitude_scale);
    println!();

    // Load and parse input data
    let positions = if args.input.extension().map(|e| e == "csv").unwrap_or(false) {
        load_csv(&args.input, args.altitude_scale)?
    } else {
        load_geojson(&args.input, args.altitude_scale)?
    };

    println!("✓ Loaded {} position observations", positions.len());

    // Group by aircraft
    let mut aircraft_positions: HashMap<String, Vec<AircraftPosition>> = HashMap::new();
    for (icao24, pos) in positions {
        aircraft_positions.entry(icao24).or_default().push(pos);
    }

    println!("✓ Found {} unique aircraft", aircraft_positions.len());

    // Sort each aircraft's positions by timestamp and split into paths
    let mut all_paths: Vec<FlightPath> = Vec::new();

    for (icao24, mut positions) in aircraft_positions {
        // Sort by timestamp
        positions.sort_by_key(|p| p.timestamp_ms);

        // Split into separate paths if there are large gaps
        let paths = split_into_paths(icao24, positions, args.max_gap_seconds);
        all_paths.extend(paths);
    }

    println!("✓ Created {} raw flight paths", all_paths.len());

    // Filter paths
    let filtered_paths: Vec<FlightPath> = all_paths
        .into_iter()
        .filter(|p| p.positions.len() >= args.min_points)
        .filter(|p| p.duration_seconds() >= args.min_duration_seconds)
        .collect();

    println!(
        "✓ {} paths after filtering (min {} points, min {} seconds)",
        filtered_paths.len(),
        args.min_points,
        args.min_duration_seconds
    );

    // Limit output if requested
    let final_paths = if args.max_paths > 0 && filtered_paths.len() > args.max_paths {
        println!("📊 Limiting to {} paths", args.max_paths);
        filtered_paths.into_iter().take(args.max_paths).collect()
    } else {
        filtered_paths
    };

    // Convert to GeoJSON
    let features = paths_to_geojson(&final_paths)?;

    // Calculate stats
    let total_coords: usize = final_paths.iter().map(|p| p.positions.len()).sum();
    let avg_coords = if final_paths.is_empty() {
        0
    } else {
        total_coords / final_paths.len()
    };

    println!();
    println!("📊 Output Summary:");
    println!("  Flight paths: {}", final_paths.len());
    println!("  Total coordinates: {}", total_coords);
    println!("  Average coords per path: {}", avg_coords);

    if let (Some(start), Some(end)) = (
        final_paths.iter().filter_map(|p| p.start_time()).min(),
        final_paths.iter().filter_map(|p| p.end_time()).max(),
    ) {
        println!(
            "  Time range: {} to {}",
            start.format("%Y-%m-%d %H:%M:%S"),
            end.format("%Y-%m-%d %H:%M:%S")
        );
    }

    // Write output
    println!();
    println!("💾 Writing GeoJSON...");
    write_geojson(features, &args.output)?;

    println!();
    println!("✅ Success! Now run:");
    println!(
        "   stt-build --input {} --output adsb-paths.stt \\",
        args.output.display()
    );
    println!("             --time-field timestamp \\");
    println!("             --end-time-field end_time \\");
    println!("             --min-zoom 0 \\");
    println!("             --max-zoom 10 \\");
    println!("             --compression gzip");

    Ok(())
}

/// Load positions from GeoJSON file
fn load_geojson(path: &PathBuf, altitude_scale: f64) -> Result<Vec<(String, AircraftPosition)>> {
    println!("🔄 Loading GeoJSON...");
    
    let file = File::open(path).context("Failed to open input file")?;
    let reader = BufReader::new(file);
    let geojson: GeoJson = serde_json::from_reader(reader).context("Failed to parse GeoJSON")?;

    let features = match geojson {
        GeoJson::FeatureCollection(fc) => fc.features,
        GeoJson::Feature(f) => vec![f],
        _ => return Err(anyhow::anyhow!("Expected FeatureCollection or Feature")),
    };

    let pb = ProgressBar::new(features.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} features")?
            .progress_chars("=>-"),
    );

    let mut positions = Vec::new();

    for feature in features {
        pb.inc(1);

        let props = match &feature.properties {
            Some(p) => p,
            None => continue,
        };

        // Get icao24 identifier
        let icao24 = match props.get("icao24").and_then(|v| v.as_str()) {
            Some(id) => id.to_string(),
            None => continue,
        };

        // Get coordinates
        let (lon, lat) = match &feature.geometry {
            Some(geom) => match &geom.value {
                GeoValue::Point(coords) if coords.len() >= 2 => (coords[0], coords[1]),
                _ => continue,
            },
            None => continue,
        };

        // Get timestamp
        let timestamp = match props.get("timestamp").and_then(|v| v.as_str()) {
            Some(ts) => match DateTime::parse_from_rfc3339(ts) {
                Ok(dt) => dt.with_timezone(&Utc),
                Err(_) => continue,
            },
            None => continue,
        };

        // Get altitude (may be in feet, scale as needed)
        let altitude = props
            .get("altitude")
            .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)))
            .unwrap_or(0.0)
            * altitude_scale;

        let speed = props
            .get("speed")
            .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)));

        let heading = props
            .get("heading")
            .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)));

        let callsign = props.get("callsign").and_then(|v| v.as_str()).map(String::from);
        let country = props.get("country").and_then(|v| v.as_str()).map(String::from);

        positions.push((
            icao24,
            AircraftPosition {
                timestamp,
                timestamp_ms: timestamp.timestamp_millis(),
                lon,
                lat,
                altitude,
                speed,
                heading,
                callsign,
                country,
            },
        ));
    }

    pb.finish_and_clear();
    Ok(positions)
}

/// Load positions from CSV file
fn load_csv(path: &PathBuf, altitude_scale: f64) -> Result<Vec<(String, AircraftPosition)>> {
    println!("🔄 Loading CSV...");

    let file = File::open(path).context("Failed to open input file")?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(reader);

    let headers = csv_reader.headers()?.clone();
    println!("  Columns: {:?}", headers.iter().collect::<Vec<_>>());

    // Find column indices
    let lon_idx = headers.iter().position(|h| h == "lon" || h == "longitude");
    let lat_idx = headers.iter().position(|h| h == "lat" || h == "latitude");
    let ts_idx = headers.iter().position(|h| h == "timestamp");
    let icao_idx = headers.iter().position(|h| h == "icao24");
    let alt_idx = headers.iter().position(|h| h == "altitude" || h == "alt");
    let speed_idx = headers.iter().position(|h| h == "speed" || h == "velocity");
    let heading_idx = headers.iter().position(|h| h == "heading" || h == "track");
    let callsign_idx = headers.iter().position(|h| h == "callsign");
    let country_idx = headers.iter().position(|h| h == "country");

    let (lon_idx, lat_idx, ts_idx, icao_idx) = match (lon_idx, lat_idx, ts_idx, icao_idx) {
        (Some(lon), Some(lat), Some(ts), Some(icao)) => (lon, lat, ts, icao),
        _ => return Err(anyhow::anyhow!(
            "CSV must have columns: lon/longitude, lat/latitude, timestamp, icao24"
        )),
    };

    let mut positions = Vec::new();
    let mut row_count = 0;

    for result in csv_reader.records() {
        row_count += 1;
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let icao24 = match record.get(icao_idx) {
            Some(v) if !v.is_empty() => v.to_string(),
            _ => continue,
        };

        let lon: f64 = match record.get(lon_idx).and_then(|v| v.parse().ok()) {
            Some(v) => v,
            None => continue,
        };

        let lat: f64 = match record.get(lat_idx).and_then(|v| v.parse().ok()) {
            Some(v) => v,
            None => continue,
        };

        let timestamp = match record.get(ts_idx) {
            Some(ts) => match DateTime::parse_from_rfc3339(ts) {
                Ok(dt) => dt.with_timezone(&Utc),
                Err(_) => continue,
            },
            None => continue,
        };

        let altitude = alt_idx
            .and_then(|i| record.get(i))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            * altitude_scale;

        let speed = speed_idx
            .and_then(|i| record.get(i))
            .and_then(|v| v.parse::<f64>().ok());

        let heading = heading_idx
            .and_then(|i| record.get(i))
            .and_then(|v| v.parse::<f64>().ok());

        let callsign = callsign_idx
            .and_then(|i| record.get(i))
            .filter(|v| !v.is_empty())
            .map(String::from);

        let country = country_idx
            .and_then(|i| record.get(i))
            .filter(|v| !v.is_empty())
            .map(String::from);

        positions.push((
            icao24,
            AircraftPosition {
                timestamp,
                timestamp_ms: timestamp.timestamp_millis(),
                lon,
                lat,
                altitude,
                speed,
                heading,
                callsign,
                country,
            },
        ));

        if row_count % 100000 == 0 {
            println!("  Processed {} rows, {} positions...", row_count, positions.len());
        }
    }

    Ok(positions)
}

/// Split a sequence of positions into separate paths based on time gaps
fn split_into_paths(
    icao24: String,
    positions: Vec<AircraftPosition>,
    max_gap_seconds: i64,
) -> Vec<FlightPath> {
    if positions.is_empty() {
        return vec![];
    }

    let max_gap_ms = max_gap_seconds * 1000;
    let mut paths = Vec::new();
    let mut current_positions: Vec<AircraftPosition> = Vec::new();

    for pos in positions {
        if let Some(last) = current_positions.last() {
            let gap = pos.timestamp_ms - last.timestamp_ms;
            if gap > max_gap_ms {
                // Gap too large, start new path
                if !current_positions.is_empty() {
                    paths.push(FlightPath {
                        icao24: icao24.clone(),
                        positions: std::mem::take(&mut current_positions),
                    });
                }
            }
        }
        current_positions.push(pos);
    }

    // Don't forget the last path
    if !current_positions.is_empty() {
        paths.push(FlightPath {
            icao24,
            positions: current_positions,
        });
    }

    paths
}

/// Convert flight paths to GeoJSON features with 3D coordinates and timestamps
fn paths_to_geojson(paths: &[FlightPath]) -> Result<Vec<Feature>> {
    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} paths")?
            .progress_chars("=>-"),
    );

    let mut features = Vec::new();

    for (idx, path) in paths.iter().enumerate() {
        pb.inc(1);

        if path.positions.len() < 2 {
            continue;
        }

        // Build 3D coordinates [lon, lat, altitude]
        let coordinates: Vec<Vec<f64>> = path
            .positions
            .iter()
            .map(|p| vec![p.lon, p.lat, p.altitude])
            .collect();

        // Build timestamps array (milliseconds since epoch)
        let timestamps: Vec<i64> = path.positions.iter().map(|p| p.timestamp_ms).collect();

        // Build properties
        let mut props = Map::new();
        props.insert("icao24".to_string(), json!(path.icao24));
        props.insert("path_id".to_string(), json!(idx));
        props.insert("num_points".to_string(), json!(path.positions.len()));
        props.insert("duration_seconds".to_string(), json!(path.duration_seconds()));

        if let Some(callsign) = path.callsign() {
            props.insert("callsign".to_string(), json!(callsign));
        }

        if let Some(country) = path.country() {
            props.insert("country".to_string(), json!(country));
        }

        if let Some(speed) = path.avg_speed() {
            props.insert("avg_speed".to_string(), json!(speed as i32));
        }

        props.insert("avg_altitude".to_string(), json!(path.avg_altitude() as i32));

        // Add time range for STT building
        if let (Some(start), Some(end)) = (path.start_time(), path.end_time()) {
            props.insert("timestamp".to_string(), json!(start.to_rfc3339()));
            props.insert("end_time".to_string(), json!(end.to_rfc3339()));
        }

        // Add timestamps array for trips layer animation
        props.insert("timestamps".to_string(), json!(timestamps));

        let feature = Feature {
            bbox: None,
            geometry: Some(Geometry::new(GeoValue::LineString(coordinates))),
            id: None,
            properties: Some(props),
            foreign_members: None,
        };

        features.push(feature);
    }

    pb.finish_and_clear();
    Ok(features)
}

/// Write features to GeoJSON file
fn write_geojson(features: Vec<Feature>, output_path: &PathBuf) -> Result<()> {
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
    let json_string = serde_json::to_string(&geojson)?;

    let mut file = File::create(output_path)?;
    file.write_all(json_string.as_bytes())?;

    println!("✓ GeoJSON written successfully");
    Ok(())
}

