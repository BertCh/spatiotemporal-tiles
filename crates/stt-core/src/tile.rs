//! Tile types and operations
//!
//! [`TileId`] is all that survives here. The pre-Arrow decoded-tile model
//! (`Tile`/`Layer`/`Feature`/`Position`/`Value`) was removed: the real build
//! path is columnar end to end and never materialises a per-feature struct
//! (see `stt_build::tiler::tile_feature_signals`), and the reader hands back
//! Arrow record batches (see [`crate::arrow_tile`]).

use crate::error::{Error, Result};
use std::cmp::Ordering;
use std::fmt;

/// Deepest addressable zoom level. Coordinates are bounded by `1u32 << z`, so a
/// zoom above 31 shifts past the width of `u32` and [`TileId::validate`] stops
/// rejecting out-of-range tiles; 22 also keeps the `2 * z`-bit Hilbert index
/// well inside `u64`.
pub const MAX_ZOOM: u8 = 22;

/// Unique identifier for a spatiotemporal tile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    /// Zoom level, `0..=`[`MAX_ZOOM`]
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
    /// f64 carries only a 52-bit mantissa, so normalising `(x, y)` to `[0,1)`
    /// and scaling by `u64::MAX` collides for neighbouring tiles at zoom ≥ 14
    /// and silently degrades the archive's range-coalescing locality. The
    /// integer curve has no such limit and is ~10× faster.
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
        if self.z >= MAX_ZOOM {
            return vec![];
        }
        vec![
            TileId::new(self.z + 1, self.x * 2, self.y * 2, self.t),
            TileId::new(self.z + 1, self.x * 2 + 1, self.y * 2, self.t),
            TileId::new(self.z + 1, self.x * 2, self.y * 2 + 1, self.t),
            TileId::new(self.z + 1, self.x * 2 + 1, self.y * 2 + 1, self.t),
        ]
    }
}

/// Canonical textual tile address: `z/x/y/t`.
impl fmt::Display for TileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}/{}", self.z, self.x, self.y, self.t)
    }
}

impl PartialOrd for TileId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TileId {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort by: zoom, hilbert index, time.
        //
        // `hilbert_index()` is recomputed on both operands of every comparison
        // and costs O(z), so sorting a large tile list through this comparator
        // pays it O(n log n) times. Callers ordering many tiles should key on
        // `(z, hilbert_index(), t)` once per tile instead.
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

    /// The address is `z/x/y/t` through both `Display` and the blanket
    /// `ToString`, which is what makes `format!("{tile}")` and
    /// `tile.to_string()` interchangeable.
    #[test]
    fn tile_id_display_is_z_x_y_t() {
        let tile = TileId::new(10, 512, 384, 1609459200000);
        assert_eq!(format!("{tile}"), "10/512/384/1609459200000");
        assert_eq!(tile.to_string(), "10/512/384/1609459200000");
    }

    /// `children()` stops at the deepest addressable zoom rather than minting
    /// tiles whose coordinates no longer fit the format's bounds.
    #[test]
    fn children_stop_at_max_zoom() {
        let deepest = TileId::new(MAX_ZOOM, 0, 0, 0);
        assert!(deepest.children().is_empty());
        assert_eq!(TileId::new(MAX_ZOOM - 1, 0, 0, 0).children().len(), 4);
    }

    #[test]
    fn test_tile_id_ordering() {
        let t1 = TileId::new(10, 512, 384, 1609459200000);
        let t2 = TileId::new(10, 512, 384, 1609545600000);
        let t3 = TileId::new(11, 1024, 768, 1609459200000);

        assert!(t1 < t2); // Same spatial, different time
        assert!(t1 < t3); // Different zoom
    }

    /// `hilbert_index` must produce distinct indices for distinct tiles at
    /// every supported zoom. Collisions among neighbours break the directory
    /// sort key and with it range coalescing; an f64-normalised curve loses
    /// exactly this property at zoom ≥ 14.
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
