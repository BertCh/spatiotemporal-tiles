//! Tile generation logic

use crate::input::ParsedFeature;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use stt_core::budget::TileBudget;
use stt_core::delta::TemporalDeltaTracker;
use stt_core::geometry::simplify_linestring_coords;
use stt_core::projection;
use stt_core::tile::TileId;
use stt_core::types::GeometryType;

/// Temporal bucketing strategy for sparse data
#[derive(Debug, Clone, Copy)]
pub enum TemporalBucket {
    /// No bucketing - use exact timestamps
    None,
    /// Bucket by second
    Second,
    /// Bucket by minute
    Minute,
    /// Bucket by hour
    Hour,
    /// Bucket by day
    Day,
    /// Bucket by week
    Week,
    /// Bucket by month
    Month,
    /// Bucket by year
    Year,
}

impl TemporalBucket {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "second" => Some(Self::Second),
            "minute" => Some(Self::Minute),
            "hour" => Some(Self::Hour),
            "day" => Some(Self::Day),
            "week" => Some(Self::Week),
            "month" => Some(Self::Month),
            "year" => Some(Self::Year),
            _ => None,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "exact timestamps",
            Self::Second => "1 second",
            Self::Minute => "1 minute",
            Self::Hour => "1 hour",
            Self::Day => "1 day",
            Self::Week => "1 week",
            Self::Month => "~30 days",
            Self::Year => "~365 days",
        }
    }

    /// Get bucket size in milliseconds
    pub fn size_ms(&self) -> u64 {
        match self {
            Self::None => 0,
            Self::Second => 1_000,
            Self::Minute => 60_000,
            Self::Hour => 3_600_000,
            Self::Day => 86_400_000,
            Self::Week => 604_800_000,
            Self::Month => 2_592_000_000, // ~30 days
            Self::Year => 31_536_000_000,  // ~365 days
        }
    }
}

/// Temporal resolution profile - defines how temporal bucketing changes with zoom
#[derive(Debug, Clone)]
pub enum TemporalResolutionProfile {
    /// Single bucket strategy for all zoom levels
    Uniform(TemporalBucket),
    /// Optimized for high-frequency tracking data (ships, planes, vehicles)
    HighFrequency,
    /// Optimized for sparse event data (earthquakes, incidents)
    SparseEvents,
    /// Optimized for daily aggregated data (COVID cases, weather)
    DailyAggregates,
    /// Custom zoom-dependent mapping
    Custom(Vec<(u8, TemporalBucket)>),
}

impl TemporalResolutionProfile {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "high-frequency" => Some(Self::HighFrequency),
            "sparse-events" => Some(Self::SparseEvents),
            "daily-aggregates" => Some(Self::DailyAggregates),
            _ => TemporalBucket::from_str(s).map(Self::Uniform),
        }
    }

    /// Get the temporal bucket for a specific zoom level
    pub fn bucket_for_zoom(&self, zoom: u8) -> TemporalBucket {
        match self {
            Self::Uniform(bucket) => *bucket,
            Self::HighFrequency => {
                // High-frequency tracking (ships, planes, vehicles)
                // World view → Regional → City → Street
                match zoom {
                    0..=3 => TemporalBucket::Day,      // World: daily patterns
                    4..=6 => TemporalBucket::Hour,     // Regional: hourly movement
                    7..=9 => TemporalBucket::Minute,   // City: minute-level routes
                    10..=12 => TemporalBucket::Second, // Street: second-level tracking
                    _ => TemporalBucket::None,         // Close-up: exact timestamps
                }
            }
            Self::SparseEvents => {
                // Sparse events (earthquakes, wildfires, incidents)
                // Optimized for file size: larger temporal buckets = fewer tiles
                // These events don't need fine temporal resolution even at high zoom
                match zoom {
                    0..=2 => TemporalBucket::Year,   // World: yearly aggregation
                    3..=4 => TemporalBucket::Month,  // Continental: monthly patterns
                    5..=6 => TemporalBucket::Week,   // Regional: weekly events
                    7..=8 => TemporalBucket::Day,    // City: daily events
                    _ => TemporalBucket::Hour,       // Close-up: hourly precision
                }
            }
            Self::DailyAggregates => {
                // Data already aggregated by day (COVID, weather)
                // No need for finer than day at any zoom level
                match zoom {
                    0..=6 => TemporalBucket::Month,  // World/Regional: monthly
                    7..=10 => TemporalBucket::Week,  // City: weekly
                    _ => TemporalBucket::Day,        // Close-up: daily (matches source)
                }
            }
            Self::Custom(mapping) => {
                // Find the appropriate bucket for this zoom level
                // Use the bucket from the highest zoom threshold <= current zoom
                mapping
                    .iter()
                    .rev()
                    .find(|(z, _)| *z <= zoom)
                    .map(|(_, bucket)| *bucket)
                    .unwrap_or(TemporalBucket::Day)
            }
        }
    }

    /// Get a description of this profile
    pub fn description(&self) -> String {
        match self {
            Self::Uniform(bucket) => format!("Uniform: {} at all zoom levels", bucket.description()),
            Self::HighFrequency => {
                "High-frequency: day→hour→minute→second (for ships, planes, vehicles)".to_string()
            }
            Self::SparseEvents => {
                "Sparse events: year→month→week→day (optimized for file size)".to_string()
            }
            Self::DailyAggregates => {
                "Daily aggregates: month→week→day (for COVID, weather)".to_string()
            }
            Self::Custom(_) => "Custom zoom-dependent mapping".to_string(),
        }
    }
}

/// Configuration for tile generation
#[derive(Debug, Clone)]
pub struct TileConfig {
    pub min_zoom: u8,
    pub max_zoom: u8,
    pub extent: u32,
    pub simplification: f64,
    pub max_tile_size: usize,
    pub layer_name: String,
    pub temporal_resolution: TemporalResolutionProfile,
    /// Enable delta encoding between temporal frames
    pub use_delta_encoding: bool,
    /// Enable advanced quantization
    pub use_quantization: bool,
    /// Tile size budget
    pub budget: TileBudget,
    /// Target chunk size in bytes (default: 512KB)
    pub target_chunk_size: usize,
}

/// Generated tile with Proto representation
#[derive(Debug)]
pub struct GeneratedTile {
    pub id: TileId,
    pub proto: stt_core::proto::Tile,
}

/// Generate tiles from features
pub fn generate_tiles(
    features: &[ParsedFeature],
    config: &TileConfig,
    workers: usize,
) -> Result<Vec<GeneratedTile>> {
    // Set up parallel processing
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global()
        .ok();

    // Group features by spatial tile (Z, X, Y)
    let mut spatial_map: HashMap<(u8, u32, u32), Vec<&ParsedFeature>> = HashMap::new();

    for feature in features {
        // For each zoom level
        for zoom in config.min_zoom..=config.max_zoom {
            // Calculate spatial tile coordinates using the standardized projection function
            let (x, y) = projection::lonlat_to_tile(feature.lon, feature.lat, zoom)
                .unwrap_or((0, 0));
            
            spatial_map.entry((zoom, x, y)).or_insert_with(Vec::new).push(feature);
        }
    }

    // Process spatial groups
    let tiles: Vec<GeneratedTile> = spatial_map
        .into_par_iter()
        .flat_map(|((z, x, y), mut features)| {
             // Sort by timestamp
            features.sort_by_key(|f| f.timestamp);

            // Chunk features dynamically based on size
            let chunks = chunk_features_dynamically(&features, config.target_chunk_size);

            let mut generated_tiles = Vec::new();
            let mut delta_tracker = if config.use_delta_encoding {
                Some(TemporalDeltaTracker::new())
            } else {
                None
            };

            let mut total_stats = stt_core::delta::DeltaStats::default();

            for chunk in chunks {
                if chunk.is_empty() { continue; }
                
                let min_time = chunk.first().map(|f| f.timestamp).unwrap_or(0);
                let max_time = chunk.last().map(|f| f.timestamp).unwrap_or(0);
                
                let tile_id = TileId::new(z, x, y, min_time);
                
                match create_tile(tile_id, &chunk, config, delta_tracker.as_mut(), min_time, max_time) {
                    Ok(tile) => generated_tiles.push(tile),
                    Err(e) => {
                        tracing::warn!("Failed to create tile {:?}: {}", tile_id, e);
                    }
                }
            }
            
            // Collect stats from this spatial location's tracker if enabled
            if let Some(tracker) = delta_tracker {
                let stats = &tracker.stats;
                total_stats.total_features += stats.total_features;
                total_stats.unchanged_features += stats.unchanged_features;
                total_stats.modified_features += stats.modified_features;
                total_stats.new_features += stats.new_features;
                total_stats.deleted_features += stats.deleted_features;
                
                // Log delta encoding statistics for significant groups
                if total_stats.total_features > 1000 {
                     let unchanged_pct = (total_stats.unchanged_features as f64 / total_stats.total_features as f64) * 100.0;
                     tracing::debug!(
                        "Delta encoding stats for {}/{}/{}: {} total, {} unchanged ({:.1}%)",
                        z, x, y,
                        total_stats.total_features,
                        total_stats.unchanged_features,
                        unchanged_pct
                    );
                }
            }
            
            generated_tiles
        })
        .collect();

    Ok(tiles)
}

/// Chunk features into groups that fit within target size
fn chunk_features_dynamically<'a>(
    features: &[&'a ParsedFeature],
    target_size: usize,
) -> Vec<Vec<&'a ParsedFeature>> {
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_size = 0;

    for feature in features {
        let size = estimate_feature_size(feature);
        
        // If adding this feature would exceed target size (significantly), 
        // and we have enough data, start a new chunk.
        // We check current_size > 0 to ensure we don't create empty chunks or fail on single large features
        if current_size > 0 && current_size + size > target_size {
            chunks.push(current_chunk);
            current_chunk = Vec::new();
            current_size = 0;
        }
        
        current_chunk.push(*feature);
        current_size += size;
    }
    
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    
    chunks
}

/// Estimate binary size of a feature (approximate)
fn estimate_feature_size(feature: &ParsedFeature) -> usize {
    // Base overhead (timestamp + IDs + headers)
    let mut size = 100; 
    
    // Geometry estimation (assuming compressed varints)
    if let Some(geom) = &feature.geojson.geometry {
         match &geom.value {
             geojson::Value::Point(_) => size += 32, // 2 coords * 8 bytes + overhead
             geojson::Value::LineString(coords) => size += coords.len() * 16,
             geojson::Value::Polygon(rings) => {
                 for ring in rings {
                     size += ring.len() * 16;
                 }
             },
             geojson::Value::MultiPoint(coords) => size += coords.len() * 32,
             _ => size += 100,
         }
    }
    
    // Properties estimation
    if let Some(props) = &feature.geojson.properties {
        // Rough estimate of JSON serialization size
        // In proto they are dictionary encoded so actual size might be smaller,
        // but we want a safe upper bound for chunking.
        size += props.len() * 20; 
    }
    
    size
}

/// Create a single tile from features
fn create_tile(
    tile_id: TileId,
    features: &[&ParsedFeature],
    config: &TileConfig,
    delta_tracker: Option<&mut TemporalDeltaTracker>,
    time_start: u64,
    time_end: u64,
) -> Result<GeneratedTile> {
    // Clone and adjust config for high zoom levels to improve precision
    // This prevents "gridded" look at high zoom levels by increasing the quantization grid
    let mut tile_config = config.clone();
    if tile_id.z >= 11 {
        tile_config.extent *= 4;
    }
    let config = &tile_config;

    // Build feature list
    let mut proto_features = Vec::new();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut key_map: HashMap<String, u32> = HashMap::new();
    let mut value_map: HashMap<String, u32> = HashMap::new();

    // Process features with delta tracking if enabled
    if let Some(tracker) = delta_tracker {
        // Convert features to internal format
        let internal_features: Vec<stt_core::tile::Feature> = features
            .iter()
            .map(|f| parsed_to_internal_feature(f, tile_id, config))
            .collect::<Result<Vec<_>>>()?;
        
        // Process through delta tracker
        let features_with_changes = tracker.process_frame(internal_features);
        
        // Encode features with change types
        for (i, (_, change_type)) in features_with_changes.iter().enumerate() {
            if let Some(original_feature) = features.get(i) {
                let proto_feature = encode_feature_with_change_type(
                    original_feature,
                    tile_id,
                    config,
                    *change_type,
                    &mut keys,
                    &mut values,
                    &mut key_map,
                    &mut value_map,
                )?;
                proto_features.push(proto_feature);
            }
        }
    } else {
        // No delta tracking - encode all features normally
        for feature in features {
            let proto_feature = convert_feature(
                feature,
                tile_id,
                config,
                &mut keys,
                &mut values,
                &mut key_map,
                &mut value_map,
            )?;
            proto_features.push(proto_feature);
        }
    }

    // Create layer
    let layer = stt_core::proto::Layer {
        name: config.layer_name.clone(),
        extent: config.extent,
        keys,
        values,
        features: proto_features.clone(),
    };
    
    // Calculate temporal resolution metadata based on dynamic chunking
    let bucket_size_ms = time_end.saturating_sub(time_start).max(1);
    
    // Calculate suggested animation speed (ms per second of real time)
    // Coarser temporal resolution (larger bucket) typically implies longer time span
    // We scale the speed so that tiles don't take forever to play.
    // 
    // Heuristic: 
    // - Short duration (<1s): Real-time (1.0)
    // - Medium duration (<1h): 60x to 3600x
    // - Long duration (>1 day): 10000x+
    
    let suggested_speed_multiplier = if bucket_size_ms < 1000 {
        1.0 
    } else {
        // Scale roughly by duration, targeting ~10-60 seconds to play full tile if dense
        (bucket_size_ms as f64 / 10000.0).max(1.0)
    } as f32;

    // Create temporal resolution metadata
    let temporal_resolution = Some(stt_core::proto::TemporalResolution {
        bucket_size_ms,
        zoom_level: tile_id.z as u32,
        feature_count: proto_features.len() as u32,
        suggested_speed_multiplier,
    });

    // Create interpolation hint (use linear for point data)
    let interpolation = Some(stt_core::proto::Interpolation {
        method: 1, // LINEAR
        properties: vec![], // All numeric properties by default
    });

    // Create tile
    let proto_tile = stt_core::proto::Tile {
        version: 1,
        time_start,
        time_end,
        layers: vec![layer],
        interpolation,
        temporal_resolution,
    };

    Ok(GeneratedTile {
        id: tile_id,
        proto: proto_tile,
    })
}

/// Convert a parsed feature to a proto feature
fn convert_feature(
    feature: &ParsedFeature,
    tile_id: TileId,
    config: &TileConfig,
    keys: &mut Vec<String>,
    values: &mut Vec<stt_core::proto::Value>,
    key_map: &mut HashMap<String, u32>,
    value_map: &mut HashMap<String, u32>,
) -> Result<stt_core::proto::Feature> {
    encode_feature_with_change_type(
        feature,
        tile_id,
        config,
        stt_core::delta::ChangeType::Created,
        keys,
        values,
        key_map,
        value_map,
    )
}

/// Convert ParsedFeature to internal Feature format for delta tracking
fn parsed_to_internal_feature(
    feature: &ParsedFeature,
    tile_id: TileId,
    config: &TileConfig,
) -> Result<stt_core::tile::Feature> {
    use stt_core::tile::Value;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    // Encode geometry
    let (geom_type_enum, geometry) = encode_geometry(&feature.geojson, tile_id, config)?;
    
    // Convert properties to internal format
    let mut properties = std::collections::HashMap::new();
    if let Some(props) = &feature.geojson.properties {
        for (key, value) in props.iter() {
            if value.is_null() {
                continue;
            }
            
            let internal_value = match value {
                serde_json::Value::String(s) => Value::String(s.clone()),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Value::Int(i)
                    } else if let Some(u) = n.as_u64() {
                        Value::UInt(u)
                    } else if let Some(f) = n.as_f64() {
                        Value::Double(f)
                    } else {
                        continue;
                    }
                }
                serde_json::Value::Bool(b) => Value::Bool(*b),
                _ => continue,
            };
            
            properties.insert(key.clone(), internal_value);
        }
    }
    
    // Generate stable feature ID based on properties (not timestamp)
    // Try to use a unique identifier from properties if available
    let feature_id = if let Some(props) = &feature.geojson.properties {
        // Common ID fields in order of preference
        let id_fields = ["mmsi", "ship_id", "vessel_id", "id", "fips", "county", "name"];
        
        let mut id_value: Option<u64> = None;
        for field in &id_fields {
            if let Some(val) = props.get(*field) {
                id_value = match val {
                    serde_json::Value::Number(n) => n.as_u64(),
                    serde_json::Value::String(s) => {
                        // Hash string to u64
                        let mut hasher = DefaultHasher::new();
                        s.hash(&mut hasher);
                        Some(hasher.finish())
                    }
                    _ => None,
                };
                if id_value.is_some() {
                    break;
                }
            }
        }
        
        // If no ID field found, hash the geometry coordinates
        if id_value.is_none() {
            let mut hasher = DefaultHasher::new();
            for coord in &geometry {
                coord.hash(&mut hasher);
            }
            id_value = Some(hasher.finish());
        }
        
        id_value.unwrap_or(feature.timestamp)
    } else {
        feature.timestamp
    };
    
    Ok(stt_core::tile::Feature {
        id: feature_id,
        geometry_type: geom_type_enum,
        geometry,
        properties,
        time_range: Some(stt_core::types::TimeRange::new(
            feature.timestamp,
            feature.timestamp,
        )),
    })
}

/// Convert proto GeomType to internal GeometryType
fn proto_to_geom_type(proto_type: i32) -> stt_core::types::GeometryType {
    match proto_type {
        0 => stt_core::types::GeometryType::Point,
        1 => stt_core::types::GeometryType::LineString,
        2 => stt_core::types::GeometryType::Polygon,
        _ => stt_core::types::GeometryType::Point,
    }
}

/// Encode a feature with change type information
fn encode_feature_with_change_type(
    feature: &ParsedFeature,
    tile_id: TileId,
    config: &TileConfig,
    change_type: stt_core::delta::ChangeType,
    keys: &mut Vec<String>,
    values: &mut Vec<stt_core::proto::Value>,
    key_map: &mut HashMap<String, u32>,
    value_map: &mut HashMap<String, u32>,
) -> Result<stt_core::proto::Feature> {
    // Only encode properties and geometry for non-UNCHANGED features
    let (tags, geometry, geom_type) = if matches!(change_type, stt_core::delta::ChangeType::Unchanged(_)) {
        // UNCHANGED: omit geometry and properties, use placeholder type
        (vec![], vec![], 0)
    } else {
        // Encode properties as tags
        let mut tags = Vec::new();

        if let Some(props) = &feature.geojson.properties {
            for (key, value) in props.iter() {
                // Skip null values - don't encode them
                if value.is_null() {
                    continue;
                }
                
                // Get or insert key
                let key_idx = *key_map.entry(key.clone()).or_insert_with(|| {
                    let idx = keys.len() as u32;
                    keys.push(key.clone());
                    idx
                });

                // Get or insert value
                // Use serde_json's to_string for consistent serialization
                let value_str = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
                let value_idx = *value_map.entry(value_str.clone()).or_insert_with(|| {
                    let idx = values.len() as u32;
                    values.push(json_to_proto_value(value));
                    idx
                });

                tags.push(key_idx);
                tags.push(value_idx);
            }
        }

        // Encode geometry
        let (geom_type, geometry) = encode_geometry(&feature.geojson, tile_id, config)?;
        
        (tags, geometry, geom_type_to_proto(geom_type))
    };
    
    // Extract hash and change enum
    let (previous_hash, change_enum) = match change_type {
        stt_core::delta::ChangeType::Unchanged(hash) => (hash.to_u64(), 0),
        stt_core::delta::ChangeType::Created => (0, 1),
        stt_core::delta::ChangeType::Modified => (0, 2),
        stt_core::delta::ChangeType::Deleted => (0, 3),
    };

    Ok(stt_core::proto::Feature {
        id: feature.timestamp, // Use timestamp as ID for now
        r#type: geom_type,
        geometry,
        tags,
        valid_from: feature.timestamp,
        valid_to: feature.timestamp,
        previous_hash,
        change: change_enum,
    })
}

/// Encode GeoJSON geometry to tile coordinates
///
/// Now uses the standardized projection module for accurate coordinate transformations.
fn encode_geometry(
    feature: &geojson::Feature,
    tile_id: TileId,
    config: &TileConfig,
) -> Result<(GeometryType, Vec<u32>)> {
    let geom = feature.geometry.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Feature has no geometry"))?;

    match &geom.value {
        geojson::Value::Point(coords) => {
            let tile_coords = projection::lonlat_to_tile_coords(
                coords[0],
                coords[1],
                tile_id.z,
                tile_id.x,
                tile_id.y,
                config.extent,
            );
            // MoveTo command (1) + coordinate pair
            let geometry = vec![9, tile_coords.0, tile_coords.1];
            Ok((GeometryType::Point, geometry))
        }
        geojson::Value::LineString(coords) => {
            // Encode LineString using absolute coordinates (frontend requirement)
            // MoveTo (1) count=1, LineTo (2) count=N-1
            let mut geometry = Vec::new();
            
            if coords.is_empty() {
                return Ok((GeometryType::LineString, vec![]));
            }
            
            // Simplify if enabled (honors config.simplification)
            let points = if config.simplification > 0.0 {
                let raw: Vec<(f64, f64)> = coords.iter()
                    .map(|c| (c[0], c[1]))
                    .collect();
                simplify_linestring_coords(&raw, config.simplification)
            } else {
                coords.iter()
                    .map(|c| (c[0], c[1]))
                    .collect()
            };
            
            if points.is_empty() {
                return Ok((GeometryType::LineString, vec![]));
            }
            
            // MoveTo
            let start = projection::lonlat_to_tile_coords(
                points[0].0,
                points[0].1,
                tile_id.z,
                tile_id.x,
                tile_id.y,
                config.extent,
            );
            geometry.push(9); // MoveTo(1) count=1 -> (1 << 3) | 1 = 9
            geometry.push(start.0);
            geometry.push(start.1);
            
            // LineTo
            if points.len() > 1 {
                let count = (points.len() - 1) as u32;
                geometry.push((count << 3) | 2); // LineTo(2)
                
                for coord in &points[1..] {
                    let tile_coords = projection::lonlat_to_tile_coords(
                        coord.0,
                        coord.1,
                        tile_id.z,
                        tile_id.x,
                        tile_id.y,
                        config.extent,
                    );
                    geometry.push(tile_coords.0);
                    geometry.push(tile_coords.1);
                }
            }
            
            Ok((GeometryType::LineString, geometry))
        }
        _ => {
            // TODO: Implement Polygon encoding using proper geo-types
            // For now, extract a point from the geometry
            let coords = extract_first_coordinate(geom)?;
            let tile_coords = projection::lonlat_to_tile_coords(
                coords.0,
                coords.1,
                tile_id.z,
                tile_id.x,
                tile_id.y,
                config.extent,
            );
            let geometry = vec![9, tile_coords.0, tile_coords.1];
            Ok((GeometryType::Point, geometry))
        }
    }
}

/// Convert lon/lat to tile-relative coordinates
/// 
/// DEPRECATED: Use stt_core::projection::lonlat_to_tile_coords instead
/// This function is kept for backwards compatibility only.
#[deprecated(since = "0.1.0", note = "Use stt_core::projection::lonlat_to_tile_coords instead")]
fn lonlat_to_tile_coords(
    lon: f64,
    lat: f64,
    zoom: u8,
    tile_x: u32,
    tile_y: u32,
    extent: u32,
) -> (u32, u32) {
    projection::lonlat_to_tile_coords(lon, lat, zoom, tile_x, tile_y, extent)
}

/// Extract first coordinate from any geometry type
fn extract_first_coordinate(geom: &geojson::Geometry) -> Result<(f64, f64)> {
    use geojson::Value as GeomValue;

    match &geom.value {
        GeomValue::Point(coords) => Ok((coords[0], coords[1])),
        GeomValue::LineString(coords) | GeomValue::MultiPoint(coords) => {
            Ok((coords[0][0], coords[0][1]))
        }
        GeomValue::Polygon(rings) => Ok((rings[0][0][0], rings[0][0][1])),
        _ => anyhow::bail!("Unsupported geometry type"),
    }
}

/// Convert JSON value to proto Value
fn json_to_proto_value(value: &serde_json::Value) -> stt_core::proto::Value {
    use stt_core::proto::value::ValueType;

    let value_type = match value {
        serde_json::Value::String(s) => Some(ValueType::StringValue(s.clone())),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(ValueType::IntValue(i))
            } else if let Some(u) = n.as_u64() {
                Some(ValueType::UintValue(u))
            } else if let Some(f) = n.as_f64() {
                Some(ValueType::DoubleValue(f))
            } else {
                Some(ValueType::StringValue(n.to_string()))
            }
        }
        serde_json::Value::Bool(b) => Some(ValueType::BoolValue(*b)),
        serde_json::Value::Null => Some(ValueType::StringValue(String::new())),
        _ => Some(ValueType::StringValue(value.to_string())),
    };

    stt_core::proto::Value { value_type }
}

/// Convert GeometryType to proto enum
fn geom_type_to_proto(geom_type: GeometryType) -> i32 {
    match geom_type {
        GeometryType::Point => 0,
        GeometryType::LineString => 1,
        GeometryType::Polygon => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lonlat_to_tile_coords() {
        let (x, y) = lonlat_to_tile_coords(0.0, 0.0, 0, 0, 0, 4096);
        // Center of the world tile should be around extent/2
        assert!(x > 2000 && x < 2200);
        assert!(y > 2000 && y < 2200);
    }
}
