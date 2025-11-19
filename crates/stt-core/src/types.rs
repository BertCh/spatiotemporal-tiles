//! Core types used throughout the STT library

use serde::{Deserialize, Serialize};

/// Geographic bounding box in WGS84 coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self {
            min_lon: -180.0,
            min_lat: -85.0511,
            max_lon: 180.0,
            max_lat: 85.0511,
        }
    }
}

impl BoundingBox {
    /// Create a new bounding box
    pub fn new(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Self {
        Self {
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        }
    }

    /// Check if this bounding box contains a point
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        lon >= self.min_lon && lon <= self.max_lon && lat >= self.min_lat && lat <= self.max_lat
    }

    /// Check if this bounding box intersects another
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.min_lon <= other.max_lon
            && self.max_lon >= other.min_lon
            && self.min_lat <= other.max_lat
            && self.max_lat >= other.min_lat
    }

    /// Expand this bounding box to include a point
    pub fn expand(&mut self, lon: f64, lat: f64) {
        self.min_lon = self.min_lon.min(lon);
        self.min_lat = self.min_lat.min(lat);
        self.max_lon = self.max_lon.max(lon);
        self.max_lat = self.max_lat.max(lat);
    }

    /// Calculate the center point
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_lon + self.max_lon) / 2.0,
            (self.min_lat + self.max_lat) / 2.0,
        )
    }
}

/// Time range with start and end timestamps
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start timestamp (Unix milliseconds)
    pub start: u64,
    /// End timestamp (Unix milliseconds)
    pub end: u64,
}

impl TimeRange {
    /// Create a new time range
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Check if this time range contains a timestamp
    pub fn contains(&self, timestamp: u64) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }

    /// Check if this time range overlaps another
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start <= other.end && self.end >= other.start
    }

    /// Duration in milliseconds
    pub fn duration(&self) -> u64 {
        self.end - self.start
    }
}

/// Compression method for tiles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    None,
    Gzip,
}

impl Compression {
    /// Convert to Protocol Buffer enum value
    pub fn to_proto(&self) -> i32 {
        match self {
            Compression::None => 0,
            Compression::Gzip => 1,
        }
    }

    /// Convert from Protocol Buffer enum value
    pub fn from_proto(value: i32) -> Self {
        match value {
            1 => Compression::Gzip,
            _ => Compression::None,
        }
    }
}

/// Geometry type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeometryType {
    Point,
    LineString,
    Polygon,
}

impl GeometryType {
    /// Convert to Protocol Buffer enum value
    pub fn to_proto(&self) -> i32 {
        match self {
            GeometryType::Point => 0,
            GeometryType::LineString => 1,
            GeometryType::Polygon => 2,
        }
    }

    /// Convert from Protocol Buffer enum value
    pub fn from_proto(value: i32) -> Self {
        match value {
            1 => GeometryType::LineString,
            2 => GeometryType::Polygon,
            _ => GeometryType::Point,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(-10.0, -10.0, 10.0, 10.0);
        assert!(bbox.contains(0.0, 0.0));
        assert!(!bbox.contains(20.0, 0.0));
    }

    #[test]
    fn test_bounding_box_intersects() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        let bbox3 = BoundingBox::new(20.0, 20.0, 30.0, 30.0);

        assert!(bbox1.intersects(&bbox2));
        assert!(!bbox1.intersects(&bbox3));
    }

    #[test]
    fn test_time_range_overlaps() {
        let tr1 = TimeRange::new(1000, 2000);
        let tr2 = TimeRange::new(1500, 2500);
        let tr3 = TimeRange::new(3000, 4000);

        assert!(tr1.overlaps(&tr2));
        assert!(!tr1.overlaps(&tr3));
    }
}

