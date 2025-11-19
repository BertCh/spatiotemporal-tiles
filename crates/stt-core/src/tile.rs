//! Tile types and operations

use crate::error::{Error, Result};
use crate::types::{GeometryType, TimeRange};
use std::cmp::Ordering;

/// Unique identifier for a spatiotemporal tile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    /// Zoom level (0-22)
    pub z: u8,
    /// X coordinate
    pub x: u32,
    /// Y coordinate
    pub y: u32,
    /// Timestamp (Unix milliseconds)
    pub t: u64,
}

impl TileId {
    /// Create a new tile ID
    pub fn new(z: u8, x: u32, y: u32, t: u64) -> Self {
        Self { z, x, y, t }
    }

    /// Validate tile coordinates
    pub fn validate(&self) -> Result<()> {
        let max_coord = 1u32 << self.z;
        if self.x >= max_coord || self.y >= max_coord {
            return Err(Error::InvalidCoordinates(self.z, self.x, self.y));
        }
        Ok(())
    }

    /// Calculate Hilbert curve index for spatial ordering
    pub fn hilbert_index(&self) -> u64 {
        // Normalize coordinates to 0-1 range for the current zoom level
        let n = 1u32 << self.z;
        let x_norm = self.x as f64 / n as f64;
        let y_norm = self.y as f64 / n as f64;
        
        // Use continuous hilbert function
        let h = hilbert_2d::xy2h_continuous_f64(x_norm, y_norm, hilbert_2d::Variant::Hilbert);
        
        // Convert to u64 index
        (h * u64::MAX as f64) as u64
    }

    /// Get parent tile at zoom level z-1
    pub fn parent(&self) -> Option<TileId> {
        if self.z == 0 {
            return None;
        }
        Some(TileId {
            z: self.z - 1,
            x: self.x / 2,
            y: self.y / 2,
            t: self.t,
        })
    }

    /// Get child tiles at zoom level z+1
    pub fn children(&self) -> Vec<TileId> {
        if self.z >= 22 {
            return vec![];
        }
        vec![
            TileId::new(self.z + 1, self.x * 2, self.y * 2, self.t),
            TileId::new(self.z + 1, self.x * 2 + 1, self.y * 2, self.t),
            TileId::new(self.z + 1, self.x * 2, self.y * 2 + 1, self.t),
            TileId::new(self.z + 1, self.x * 2 + 1, self.y * 2 + 1, self.t),
        ]
    }

    /// Convert to string format: z/x/y/t
    pub fn to_string(&self) -> String {
        format!("{}/{}/{}/{}", self.z, self.x, self.y, self.t)
    }
}

impl PartialOrd for TileId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TileId {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort by: zoom, hilbert index, time
        self.z
            .cmp(&other.z)
            .then(self.hilbert_index().cmp(&other.hilbert_index()))
            .then(self.t.cmp(&other.t))
    }
}

/// A decoded tile with all its features
#[derive(Debug, Clone)]
pub struct Tile {
    pub id: TileId,
    pub time_range: TimeRange,
    pub layers: Vec<Layer>,
}

/// A layer within a tile
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub extent: u32,
    pub features: Vec<Feature>,
    /// Trajectories for moving objects
    pub trajectories: Vec<Trajectory>,
}

/// A feature within a layer
#[derive(Debug, Clone)]
pub struct Feature {
    pub id: u64,
    pub geometry_type: GeometryType,
    pub geometry: Vec<u32>,
    pub properties: std::collections::HashMap<String, Value>,
    pub time_range: Option<TimeRange>,
}

/// A trajectory for a moving object
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub id: u64,
    /// Time offsets in milliseconds from tile start
    pub time_offsets: Vec<u32>,
    /// Coordinates (x, y pairs) relative to tile origin
    /// Usually decoded from zigzag encoded deltas
    pub coordinates: Vec<i32>,
    pub properties: std::collections::HashMap<String, Value>,
    pub valid_from: Option<u64>,
    pub valid_to: Option<u64>,
}

/// Property value
#[derive(Debug, Clone)]
pub enum Value {
    String(String),
    Double(f64),
    Float(f32),
    Int(i64),
    UInt(u64),
    Bool(bool),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_id_validation() {
        let valid = TileId::new(10, 512, 384, 1609459200000);
        assert!(valid.validate().is_ok());

        let invalid = TileId::new(10, 2048, 384, 1609459200000);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_tile_id_parent() {
        let tile = TileId::new(10, 512, 384, 1609459200000);
        let parent = tile.parent().unwrap();
        assert_eq!(parent.z, 9);
        assert_eq!(parent.x, 256);
        assert_eq!(parent.y, 192);
    }

    #[test]
    fn test_tile_id_children() {
        let tile = TileId::new(10, 512, 384, 1609459200000);
        let children = tile.children();
        assert_eq!(children.len(), 4);
        assert_eq!(children[0].z, 11);
    }

    #[test]
    fn test_tile_id_ordering() {
        let t1 = TileId::new(10, 512, 384, 1609459200000);
        let t2 = TileId::new(10, 512, 384, 1609545600000);
        let t3 = TileId::new(11, 1024, 768, 1609459200000);

        assert!(t1 < t2); // Same spatial, different time
        assert!(t1 < t3); // Different zoom
    }
}
