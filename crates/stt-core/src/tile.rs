//! Tile types and operations
//!
//! [`TileId`] is all that survives here. The pre-Arrow decoded-tile model
//! (`Tile`/`Layer`/`Feature`/`Position`/`Value`) was removed: the real build
//! path is columnar end to end and never materialises a per-feature struct
//! (see `stt_build::tiler::tile_feature_signals`), and the reader hands back
//! Arrow record batches (see [`crate::arrow_tile`]).

use crate::error::{Error, Result};
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

    /// Calculate Hilbert curve index for spatial ordering.
    ///
    /// Uses the integer (`discrete`) Hilbert curve at the tile's own zoom
    /// order, so neighbouring tiles never collide on the directory sort key.
    ///
    /// The previous implementation normalised `(x, y)` to f64 `[0,1)` and
    /// multiplied by `u64::MAX`. f64 has only a 52-bit mantissa, so at
    /// zoom ≥ 14 the per-axis precision was already at the mantissa limit and
    /// the post-multiply discarded trailing bits — neighbouring tiles could
    /// collide on the same `hilbert_index`, silently degrading the archive's
    /// range-coalescing locality. The integer variant is also ~10× faster.
    pub fn hilbert_index(&self) -> u64 {
        // `xy2h_discrete` requires `order >= 1`. At zoom 0 the only valid
        // coordinates are (0, 0) and the index is trivially 0.
        if self.z == 0 {
            return 0;
        }
        let order = self.z as usize;
        let h = hilbert_2d::xy2h_discrete(
            self.x as usize,
            self.y as usize,
            order,
            hilbert_2d::Variant::Hilbert,
        );
        h as u64
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

    /// Regression: at zoom ≥ 14 the previous f64-normalised Hilbert collapsed
    /// neighbouring tiles to the same index, breaking range coalescing. The
    /// integer variant must produce distinct indices for every distinct tile
    /// at every supported zoom.
    #[test]
    fn hilbert_index_is_distinct_for_distinct_tiles_at_high_zoom() {
        for &z in &[14u8, 16, 18, 20, 22] {
            let n = 1u32 << z;
            // Sample 4 adjacent tiles near the centre.
            let cx = n / 2;
            let cy = n / 2;
            let coords = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)];
            let indices: std::collections::HashSet<u64> = coords
                .iter()
                .map(|&(x, y)| TileId::new(z, x, y, 0).hilbert_index())
                .collect();
            assert_eq!(
                indices.len(),
                4,
                "hilbert_index collided at zoom {} for adjacent tiles",
                z
            );
        }
    }

    /// Zoom 0 is a single tile — Hilbert index is trivially 0 and the function
    /// must not call into the discrete impl (which rejects order = 0).
    #[test]
    fn hilbert_index_zoom_zero_is_zero() {
        assert_eq!(TileId::new(0, 0, 0, 0).hilbert_index(), 0);
    }
}
