//! Generate satellite orbit data from CelesTrak TLE data
//!
//! Data source: https://celestrak.org/NORAD/elements/
//! Uses SGP4 propagation to generate accurate orbital trajectories
//!
//! Performance: Uses Rayon for parallel satellite propagation

use crate::common;
use crate::common::{LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter};
use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde_json::{json, Map};
use sgp4::{Constants, Elements};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// CelesTrak TLE data URL for all active satellites
const CELESTRAK_ACTIVE_URL: &str =
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle";

/// Earth radius in km (WGS84)
const EARTH_RADIUS_KM: f64 = 6378.137;

#[derive(Parser, Debug)]
#[command(about = "Generate satellite orbit data from CelesTrak TLE")]
pub struct Args {
    /// Output file (.stt, .geojson, or .csv)
    #[arg(short, long, default_value = "satellites.stt")]
    pub output: PathBuf,

    /// Duration of simulation in hours
    #[arg(long, default_value = "24")]
    pub duration_hours: u32,

    /// Time step between positions in seconds
    #[arg(long, default_value = "60")]
    pub step_seconds: u32,

    /// Use cached TLE data (skip download)
    #[arg(long)]
    pub cached: bool,

    /// TLE cache file path
    #[arg(long, default_value = "data/celestrak-active.tle")]
    pub tle_cache: PathBuf,

    /// Maximum number of satellites to process (0 = all)
    #[arg(long, default_value = "0")]
    pub max_satellites: usize,

    /// Filter by orbit type: LEO, MEO, GEO, or all
    #[arg(long, default_value = "all")]
    pub orbit_type: String,

    /// Skip stt-build step
    #[arg(long)]
    pub skip_build: bool,

    /// Start time (ISO 8601 format, defaults to current time)
    #[arg(long)]
    pub start_time: Option<String>,
}

/// Satellite orbit classification based on altitude
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrbitType {
    LEO, // Low Earth Orbit: < 2000 km
    MEO, // Medium Earth Orbit: 2000 - 35786 km
    GEO, // Geostationary/Geosynchronous: ~35786 km
    HEO, // High Earth Orbit: > 35786 km
}

impl OrbitType {
    fn from_altitude_km(altitude: f64) -> Self {
        if altitude < 2000.0 {
            OrbitType::LEO
        } else if altitude < 35000.0 {
            OrbitType::MEO
        } else if altitude < 40000.0 {
            OrbitType::GEO
        } else {
            OrbitType::HEO
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            OrbitType::LEO => "LEO",
            OrbitType::MEO => "MEO",
            OrbitType::GEO => "GEO",
            OrbitType::HEO => "HEO",
        }
    }
}

pub fn run(args: Args) -> Result<()> {
    println!("🛰️  Satellite Orbit Generator");
    println!("==============================\n");

    // Download or load TLE data
    let tle_path = &args.tle_cache;
    if !args.cached || !tle_path.exists() {
        if let Some(parent) = tle_path.parent() {
            fs::create_dir_all(parent)?;
        }
        println!("📥 Downloading TLE data from CelesTrak...");
        download_tle(CELESTRAK_ACTIVE_URL, tle_path)?;
    } else {
        println!("📂 Using cached TLE data: {}", tle_path.display());
    }

    // Parse TLE data
    println!("\n🔍 Parsing TLE data...");
    let tle_content = fs::read_to_string(tle_path)
        .with_context(|| format!("Failed to read TLE file: {}", tle_path.display()))?;

    let elements_list = sgp4::parse_3les(&tle_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse TLE data: {:?}", e))?;

    println!("   Found {} satellites", elements_list.len());

    // Filter by orbit type if specified
    let filter_orbit = args.orbit_type.to_uppercase();
    let elements_list: Vec<Elements> = if filter_orbit != "ALL" {
        elements_list
            .into_iter()
            .filter(|e| {
                // Estimate altitude from mean motion (orbits per day)
                let mean_motion = e.mean_motion; // revolutions per day
                let period_minutes = 24.0 * 60.0 / mean_motion;
                let semi_major_axis_km = ((period_minutes * 60.0 / (2.0 * PI)).powi(2)
                    * 398600.4418_f64)
                    .powf(1.0 / 3.0);
                let altitude_km = semi_major_axis_km - EARTH_RADIUS_KM;
                let orbit_type = OrbitType::from_altitude_km(altitude_km);
                orbit_type.as_str() == filter_orbit
            })
            .collect()
    } else {
        elements_list
    };

    println!("   After filtering: {} satellites", elements_list.len());

    // Limit number of satellites if specified
    let elements_list: Vec<Elements> = if args.max_satellites > 0 {
        elements_list
            .into_iter()
            .take(args.max_satellites)
            .collect()
    } else {
        elements_list
    };

    println!("   Processing: {} satellites", elements_list.len());

    // Determine start time
    let start_time: DateTime<Utc> = if let Some(ref start_str) = args.start_time {
        DateTime::parse_from_rfc3339(start_str)
            .with_context(|| format!("Invalid start time format: {}", start_str))?
            .with_timezone(&Utc)
    } else {
        Utc::now()
    };

    let duration = Duration::hours(args.duration_hours as i64);
    let end_time = start_time + duration;
    let step = Duration::seconds(args.step_seconds as i64);

    println!("\n⏱️  Simulation parameters:");
    println!("   Start: {}", start_time.to_rfc3339());
    println!("   End: {}", end_time.to_rfc3339());
    println!("   Step: {} seconds", args.step_seconds);
    println!("   Duration: {} hours", args.duration_hours);

    // stt-build only accepts GeoParquet input (the GeoJSON path was retired in
    // the Arrow migration), so write a Parquet intermediate when the final
    // output is an .stt archive; otherwise honour the requested extension.
    let build_stt =
        args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build;
    let intermediate_path = if build_stt {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    // Propagate orbits and generate per-segment LineString records
    println!("\n🌍 Propagating satellite orbits...");
    let records = propagate_orbits(&elements_list, start_time, end_time, step)?;

    println!("\n✓ Generated {} orbit trajectories", records.len());

    // Write the GeoParquet intermediate.
    println!("\n💾 Writing output...");
    let property_columns: Vec<PropertyColumn> = vec![
        PropertyColumn::string("object_name"),
        PropertyColumn::string("norad_id"),
        PropertyColumn::string("intl_designator"),
        PropertyColumn::string("orbit_type"),
        PropertyColumn::numeric("altitude"),
        PropertyColumn::numeric("inclination"),
        PropertyColumn::numeric("eccentricity"),
        PropertyColumn::numeric("mean_motion"),
        PropertyColumn::numeric("segment"),
    ];
    let mut writer =
        StreamingLineStringParquetWriter::with_columns(&intermediate_path, property_columns)?;
    for record in &records {
        writer.write_linestring(record)?;
    }
    writer.finish()?;

    // Build STT if requested. The LineString Parquet writer names its time
    // columns `timestamp` / `end_timestamp`.
    if build_stt {
        common::run_stt_build_with_end_time(
            &intermediate_path,
            &args.output,
            "timestamp",
            Some("end_timestamp"),
            0,
            // The satellite demos render at a fixed global zoom 0 (zoomOverride
            // + useGlobalBounds), so z4-6 tiles were never requested yet
            // dominated the archive (the global orbit lines fan out into
            // thousands of high-zoom tiles → an 8.6 GB file at z6). Cap at z3:
            // the demo is visually identical and the archive shrinks ~50x.
            3,
            "zstd",
        )?;

        // Clean up intermediate file
        let _ = fs::remove_file(&intermediate_path);
    }

    println!("\n✅ Satellite orbit generation complete!");
    println!("   Output: {}", args.output.display());

    Ok(())
}

/// Download TLE data from CelesTrak
fn download_tle(url: &str, output_path: &PathBuf) -> Result<()> {
    let response = reqwest::blocking::get(url)
        .with_context(|| format!("Failed to download TLE data from: {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    let content = response.text()?;
    let mut file = File::create(output_path)?;
    file.write_all(content.as_bytes())?;

    println!("   Downloaded {} bytes", content.len());

    Ok(())
}

/// Propagate satellite orbits and generate GeoJSON features
/// Each satellite may produce multiple features if the orbit crosses discontinuities
///
/// Performance: Uses Rayon parallel iterators to propagate all satellites concurrently
fn propagate_orbits(
    elements_list: &[Elements],
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    step: Duration,
) -> Result<Vec<LineStringRecord>> {
    let pb = ProgressBar::new(elements_list.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} satellites ({eta})")?
            .progress_chars("#>-"),
    );

    // Atomic counters for thread-safe progress tracking
    let successful = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);

    // Process satellites in parallel using Rayon
    let features: Vec<LineStringRecord> = elements_list
        .par_iter()
        .flat_map(|elements| {
            pb.inc(1);

            match propagate_single_satellite(elements, start_time, end_time, step) {
                Ok(segment_features) if !segment_features.is_empty() => {
                    successful.fetch_add(1, Ordering::Relaxed);
                    segment_features
                }
                Ok(_) => {
                    // Satellite has no valid positions (e.g., decayed)
                    failed.fetch_add(1, Ordering::Relaxed);
                    Vec::new()
                }
                Err(_e) => {
                    // Propagation error (e.g., invalid orbital elements)
                    failed.fetch_add(1, Ordering::Relaxed);
                    Vec::new()
                }
            }
        })
        .collect();

    pb.finish_with_message("Propagation complete");
    println!(
        "   Successful: {}, Failed: {}",
        successful.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed)
    );

    Ok(features)
}

/// Threshold for detecting antimeridian crossings (in degrees)
/// We only split when a path actually crosses lon ±180° creating a wrapping artifact
const ANTIMERIDIAN_THRESHOLD: f64 = 170.0;

/// Propagate a single satellite and return GeoJSON features for each continuous segment
/// Each segment represents a portion of the orbit with its own time range
/// Paths are split at antimeridian crossings to avoid rendering artifacts
fn propagate_single_satellite(
    elements: &Elements,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    step: Duration,
) -> Result<Vec<LineStringRecord>> {
    // Create propagation constants
    let constants = match Constants::from_elements(elements) {
        Ok(c) => c,
        Err(e) => return Err(anyhow::anyhow!("Invalid elements: {:?}", e)),
    };

    // Pre-calculate expected number of positions for better allocation
    let total_duration = end_time.signed_duration_since(start_time);
    let expected_positions = (total_duration.num_seconds() / step.num_seconds() + 1) as usize;

    // Store positions with altitude: (time, lon, lat, altitude_meters)
    let mut positions: Vec<(DateTime<Utc>, f64, f64, f64)> = Vec::with_capacity(expected_positions);
    let mut current_time = start_time;

    // Get satellite epoch for propagation (convert NaiveDateTime to DateTime<Utc>)
    let epoch: DateTime<Utc> = DateTime::from_naive_utc_and_offset(elements.datetime, Utc);

    while current_time <= end_time {
        // Calculate minutes since epoch
        let duration_since_epoch = current_time.signed_duration_since(&epoch);
        let minutes_since_epoch = duration_since_epoch.num_milliseconds() as f64 / 60000.0;

        // Propagate to get position in TEME (True Equator Mean Equinox) frame
        match constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch)) {
            Ok(prediction) => {
                // Convert TEME position to geodetic coordinates
                let (lon, lat, alt_km) = teme_to_geodetic(
                    prediction.position[0],
                    prediction.position[1],
                    prediction.position[2],
                    current_time,
                );

                // Scale altitude for visualization
                // Earth radius is ~6371 km, so we scale altitude relative to that
                // Use a compressed logarithmic scale to keep all satellites visible
                // while still showing relative altitude differences
                let earth_radius_km = 6371.0;
                // Scale: altitude as fraction of Earth radius, with log compression for high orbits
                // LEO (~400 km) -> ~100km visual altitude
                // GEO (~35786 km) -> ~500km visual altitude
                let visual_altitude_km = if alt_km < 2000.0 {
                    // LEO: linear scale (1:10)
                    alt_km * 0.05
                } else {
                    // MEO/GEO/HEO: logarithmic compression
                    100.0 + (alt_km / 1000.0).ln() * 50.0
                };
                let alt_meters = visual_altitude_km * 1000.0;
                positions.push((current_time, lon, lat, alt_meters));
            }
            Err(_) => {
                // Propagation failed at this time step (satellite may have decayed)
            }
        }

        current_time = current_time + step;
    }

    // Need at least 2 points for a LineString
    if positions.len() < 2 {
        return Ok(Vec::new());
    }

    // Estimate orbit type from mean motion
    let mean_motion = elements.mean_motion; // revolutions per day
    let orbit_type = if mean_motion > 11.0 {
        OrbitType::LEO
    } else if mean_motion > 2.0 {
        OrbitType::MEO
    } else if mean_motion > 0.9 && mean_motion < 1.1 {
        OrbitType::GEO
    } else {
        OrbitType::HEO
    };

    // Split path at large longitude jumps (antimeridian crossings, polar passes)
    let segments = split_at_discontinuities(&positions);

    // Create a separate LineString record for each segment with proper timestamps
    let mut records = Vec::new();

    for (seg_idx, segment) in segments.iter().enumerate() {
        if segment.len() < 2 {
            continue;
        }

        // Get segment's actual time range
        let seg_start = segment.first().unwrap().0;
        let seg_end = segment.last().unwrap().0;

        // The STT format stores 2D interleaved [lon, lat]; altitude is dropped
        // at the build seam regardless of input, so emit 2D coordinates here.
        let coordinates: Vec<[f64; 2]> = segment
            .iter()
            .map(|(_, lon, lat, _alt)| [*lon, *lat])
            .collect();

        // Calculate average altitude for the segment (for properties)
        let avg_altitude: f64 =
            segment.iter().map(|(_, _, _, alt)| alt).sum::<f64>() / segment.len() as f64;

        // Build properties (string-typed in the Parquet writer). `timestamp` /
        // `end_timestamp` are carried by the record's time fields, not here.
        let mut properties = Map::new();
        properties.insert(
            "object_name".to_string(),
            json!(elements.object_name.clone().unwrap_or_default()),
        );
        properties.insert("norad_id".to_string(), json!(elements.norad_id));
        properties.insert(
            "intl_designator".to_string(),
            json!(elements
                .international_designator
                .clone()
                .unwrap_or_default()),
        );
        properties.insert("orbit_type".to_string(), json!(orbit_type.as_str()));
        // Store altitude in km as f64 to avoid integer overflow for high-altitude orbits
        properties.insert(
            "altitude".to_string(),
            json!((avg_altitude / 1000.0).round()),
        ); // km for display
        properties.insert(
            "inclination".to_string(),
            json!(elements.inclination * 180.0 / PI),
        );
        properties.insert("eccentricity".to_string(), json!(elements.eccentricity));
        properties.insert("mean_motion".to_string(), json!(elements.mean_motion));
        properties.insert("segment".to_string(), json!(seg_idx));

        records.push(LineStringRecord::new(
            coordinates,
            seg_start,
            Some(seg_end),
            properties,
        ));
    }

    Ok(records)
}

/// Split positions into segments only at antimeridian crossings
///
/// We detect antimeridian crossings when:
/// - Previous longitude is near +180 AND current longitude is near -180 (or vice versa)
/// - This would create a visual artifact of the path going "the long way" around the map
///
/// We do NOT split for polar passes where longitude changes rapidly but legitimately.
/// Input/output: (time, lon, lat, altitude_meters)
fn split_at_discontinuities(
    positions: &[(DateTime<Utc>, f64, f64, f64)],
) -> Vec<Vec<(DateTime<Utc>, f64, f64, f64)>> {
    let mut segments: Vec<Vec<(DateTime<Utc>, f64, f64, f64)>> = Vec::new();
    let mut current_segment: Vec<(DateTime<Utc>, f64, f64, f64)> = Vec::new();

    for (i, pos) in positions.iter().enumerate() {
        if i == 0 {
            current_segment.push(*pos);
            continue;
        }

        let prev_lon = positions[i - 1].1;
        let curr_lon = pos.1;

        // Detect antimeridian crossing: both points near ±180° but on opposite sides
        // This is the case where drawing a straight line would go "the wrong way" around
        let crosses_antimeridian = (prev_lon > ANTIMERIDIAN_THRESHOLD
            && curr_lon < -ANTIMERIDIAN_THRESHOLD)
            || (prev_lon < -ANTIMERIDIAN_THRESHOLD && curr_lon > ANTIMERIDIAN_THRESHOLD);

        if crosses_antimeridian {
            // Split at antimeridian crossing to avoid rendering artifacts
            if current_segment.len() >= 2 {
                segments.push(current_segment);
            }
            current_segment = vec![*pos];
        } else {
            current_segment.push(*pos);
        }
    }

    // Add final segment
    if current_segment.len() >= 2 {
        segments.push(current_segment);
    }

    segments
}

/// Convert TEME (True Equator Mean Equinox) coordinates to geodetic (lon, lat, alt)
/// Position is in km, output altitude is in km
fn teme_to_geodetic(x: f64, y: f64, z: f64, time: DateTime<Utc>) -> (f64, f64, f64) {
    // Calculate Greenwich Mean Sidereal Time (GMST)
    let gmst = calculate_gmst(time);

    // Rotate from TEME to ECEF (Earth-Centered Earth-Fixed)
    let cos_gmst = gmst.cos();
    let sin_gmst = gmst.sin();

    let x_ecef = x * cos_gmst + y * sin_gmst;
    let y_ecef = -x * sin_gmst + y * cos_gmst;
    let z_ecef = z;

    // Convert ECEF to geodetic using iterative method (Bowring's method)
    ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
}

/// Calculate Greenwich Mean Sidereal Time in radians
fn calculate_gmst(time: DateTime<Utc>) -> f64 {
    // Julian Date calculation
    let year = time.year() as f64;
    let month = time.month() as f64;
    let day = time.day() as f64;
    let hour = time.hour() as f64;
    let minute = time.minute() as f64;
    let second = time.second() as f64;

    // Adjust for Jan/Feb months
    let (y, m) = if month <= 2.0 {
        (year - 1.0, month + 12.0)
    } else {
        (year, month)
    };

    // Gregorian-calendar correction (Meeus, Astronomical Algorithms). Without
    // this term the Julian Date is ~13 days too large for modern dates, which
    // threw GMST off by ~14° and placed every satellite ~14° wrong in
    // longitude. A = floor(Y/100); B = 2 - A + floor(A/4).
    let a = (y / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();

    let jd = (365.25 * (y + 4716.0)).floor()
        + (30.6001 * (m + 1.0)).floor()
        + day
        + b
        + (hour + minute / 60.0 + second / 3600.0) / 24.0
        - 1524.5;

    // Julian centuries since J2000.0
    let t_ut1 = (jd - 2451545.0) / 36525.0;

    // GMST in seconds
    let gmst_sec: f64 =
        67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t_ut1 + 0.093104 * t_ut1.powi(2)
            - 6.2e-6 * t_ut1.powi(3);

    // Convert to radians (mod 2π)
    let gmst_rad = (gmst_sec / 240.0_f64).to_radians() % (2.0 * PI);

    if gmst_rad < 0.0 {
        gmst_rad + 2.0 * PI
    } else {
        gmst_rad
    }
}

/// Convert ECEF coordinates to geodetic (lon, lat, alt)
/// Uses WGS84 ellipsoid parameters
fn ecef_to_geodetic(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    const A: f64 = 6378.137; // Equatorial radius in km
    const B: f64 = 6356.7523142; // Polar radius in km
    const E2: f64 = (A * A - B * B) / (A * A); // First eccentricity squared

    let lon = y.atan2(x).to_degrees();
    let p = (x * x + y * y).sqrt();

    // Initial latitude estimate
    let mut lat = (z / p).atan();

    // Iterative refinement (converges quickly)
    for _ in 0..10 {
        let sin_lat = lat.sin();
        let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
        lat = (z + E2 * n * sin_lat).atan2(p);
    }

    let lat_deg = lat.to_degrees();

    // Calculate altitude
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
    let alt = if cos_lat.abs() > 1e-10 {
        p / cos_lat - n
    } else {
        z.abs() / sin_lat.abs() - n * (1.0 - E2)
    };

    (lon, lat_deg, alt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbit_type_classification() {
        assert_eq!(OrbitType::from_altitude_km(400.0), OrbitType::LEO);
        assert_eq!(OrbitType::from_altitude_km(1000.0), OrbitType::LEO);
        assert_eq!(OrbitType::from_altitude_km(2000.0), OrbitType::MEO);
        assert_eq!(OrbitType::from_altitude_km(20000.0), OrbitType::MEO);
        assert_eq!(OrbitType::from_altitude_km(35786.0), OrbitType::GEO);
        assert_eq!(OrbitType::from_altitude_km(50000.0), OrbitType::HEO);
    }

    #[test]
    fn test_gmst_calculation() {
        // J2000.0 epoch: 2000-01-01 12:00:00 UTC. At J2000 the Julian centuries
        // term (t_ut1) is exactly 0.0, so `calculate_gmst` reduces to the pinned
        // 67310.54841 s coefficient with no accumulated round-off — the result is
        // a deterministic function of the inputs, bit-stable across platforms
        // (plain IEEE-754 f64, no fma), not "flaky". Expected GMST at J2000 is
        // 18h 41m 50.55s = 280.4606°; this calc yields 280.46062°, ~0.0006° off.
        let j2000 = DateTime::parse_from_rfc3339("2000-01-01T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let gmst = calculate_gmst(j2000);
        let gmst_deg = gmst.to_degrees();
        // 0.01° tolerance keeps a ~16× margin over the actual 0.0006° error while
        // still catching any real regression in the Meeus/GMST formula. The
        // second clause guards the 0°/360° wraparound (dead for 280.46°, but kept
        // so the assertion stays correct if the target ever sits near the seam).
        assert!(
            (gmst_deg - 280.46).abs() < 0.01 || (gmst_deg + 360.0 - 280.46).abs() < 0.01,
            "GMST at J2000 = {gmst_deg}°, expected ~280.4606°"
        );
    }
}
