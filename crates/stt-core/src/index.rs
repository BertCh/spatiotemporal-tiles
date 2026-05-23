//! Spatial and temporal indexing structures
//!
//! This module provides indexing utilities for efficient spatiotemporal queries.

use crate::tile::TileId;

/// Spatial index using Hilbert curve ordering
#[derive(Debug, Clone)]
pub struct SpatialIndex {
    /// Sorted list of Hilbert curve values
    pub hilbert_ids: Vec<u64>,
    /// Corresponding tile indices
    pub tile_indices: Vec<u32>,
    /// Zoom level offsets for faster lookups
    pub zoom_offsets: Vec<u32>,
}

impl SpatialIndex {
    /// Create a new spatial index from tile IDs
    pub fn new(mut tile_ids: Vec<TileId>) -> Self {
        // Sort by Hilbert curve index
        tile_ids.sort_by_key(|id| id.hilbert_index());

        let hilbert_ids: Vec<u64> = tile_ids.iter().map(|id| id.hilbert_index()).collect();
        let tile_indices: Vec<u32> = (0..tile_ids.len() as u32).collect();

        // Calculate zoom offsets
        let mut zoom_offsets = vec![0u32; 23];
        for (idx, tile_id) in tile_ids.iter().enumerate() {
            let zoom = tile_id.z as usize;
            if zoom < 23 && zoom_offsets[zoom] == 0 {
                zoom_offsets[zoom] = idx as u32;
            }
        }

        Self {
            hilbert_ids,
            tile_indices,
            zoom_offsets,
        }
    }

    /// Find tiles within an inclusive Hilbert curve range `[start, end]`.
    ///
    /// Uses half-open slicing internally: `start_idx` is the first element
    /// `>= start`, and `end_idx` is the first element `> end` (the exclusive
    /// upper bound). This avoids over-returning one element and never panics
    /// on an empty / fully-out-of-range query.
    pub fn query_range(&self, start: u64, end: u64) -> Vec<u32> {
        if start > end || self.hilbert_ids.is_empty() {
            return vec![];
        }

        // First index with value >= start.
        let start_idx = self.hilbert_ids.partition_point(|&h| h < start);
        // First index with value > end (exclusive upper bound).
        let end_idx = self.hilbert_ids.partition_point(|&h| h <= end);

        if start_idx >= end_idx {
            return vec![];
        }

        self.tile_indices[start_idx..end_idx].to_vec()
    }

    /// Find the closest tile to a given Hilbert index
    pub fn find_nearest(&self, hilbert_id: u64) -> Option<u32> {
        if self.hilbert_ids.is_empty() {
            return None;
        }

        let idx = self
            .hilbert_ids
            .binary_search(&hilbert_id)
            .unwrap_or_else(|i| i.min(self.hilbert_ids.len() - 1));

        Some(self.tile_indices[idx])
    }
}

/// Temporal index using sorted timestamps
#[derive(Debug, Clone)]
pub struct TemporalIndex {
    /// Sorted unique timestamps
    pub timestamps: Vec<u64>,
    /// Offsets into tile_refs for each timestamp
    pub tile_ref_offsets: Vec<u32>,
    /// Tile references (indices into tile array)
    pub tile_refs: Vec<u32>,
}

impl TemporalIndex {
    /// Create a new temporal index
    pub fn new(tiles: &[(TileId, u32)]) -> Self {
        let mut timestamp_map: std::collections::BTreeMap<u64, Vec<u32>> =
            std::collections::BTreeMap::new();

        for (tile_id, idx) in tiles {
            timestamp_map
                .entry(tile_id.t)
                .or_insert_with(Vec::new)
                .push(*idx);
        }

        let timestamps: Vec<u64> = timestamp_map.keys().copied().collect();
        let mut tile_ref_offsets = Vec::new();
        let mut tile_refs = Vec::new();

        let mut offset = 0u32;
        for tile_indices in timestamp_map.values() {
            tile_ref_offsets.push(offset);
            tile_refs.extend(tile_indices);
            offset += tile_indices.len() as u32;
        }

        Self {
            timestamps,
            tile_ref_offsets,
            tile_refs,
        }
    }

    /// Find tiles at a specific timestamp
    pub fn query_exact(&self, timestamp: u64) -> Vec<u32> {
        let idx = match self.timestamps.binary_search(&timestamp) {
            Ok(i) => i,
            Err(_) => return vec![],
        };

        let start = self.tile_ref_offsets[idx] as usize;
        let end = if idx + 1 < self.tile_ref_offsets.len() {
            self.tile_ref_offsets[idx + 1] as usize
        } else {
            self.tile_refs.len()
        };

        self.tile_refs[start..end].to_vec()
    }

    /// Find tiles within an inclusive time range `[start_time, end_time]`.
    ///
    /// `start_idx` is the first timestamp `>= start_time`. `end_idx` is the
    /// first timestamp `> end_time`, so the matching timestamp buckets are
    /// `[start_idx, end_idx)`. The flattened `tile_refs` slice for those
    /// buckets is `tile_ref_offsets[start_idx] .. (tile_ref_offsets[end_idx]
    /// or tile_refs.len())`.
    pub fn query_range(&self, start_time: u64, end_time: u64) -> Vec<u32> {
        if start_time > end_time || self.timestamps.is_empty() {
            return vec![];
        }

        // First timestamp >= start_time.
        let start_idx = self.timestamps.partition_point(|&t| t < start_time);
        // First timestamp > end_time (exclusive bucket bound).
        let end_idx = self.timestamps.partition_point(|&t| t <= end_time);

        if start_idx >= end_idx {
            return vec![];
        }

        let first_offset = self.tile_ref_offsets[start_idx] as usize;
        // end_idx is in [1, timestamps.len()]. If it points one past the
        // last bucket, the slice extends to the end of tile_refs.
        let last_offset = if end_idx < self.tile_ref_offsets.len() {
            self.tile_ref_offsets[end_idx] as usize
        } else {
            self.tile_refs.len()
        };

        self.tile_refs[first_offset..last_offset].to_vec()
    }

    /// Find the nearest timestamp to a given time
    pub fn find_nearest_timestamp(&self, timestamp: u64) -> Option<u64> {
        if self.timestamps.is_empty() {
            return None;
        }

        let idx = self
            .timestamps
            .binary_search(&timestamp)
            .unwrap_or_else(|i| i.min(self.timestamps.len() - 1));

        Some(self.timestamps[idx])
    }
}

/// Combined spatiotemporal index
#[derive(Debug, Clone)]
pub struct SpatioTemporalIndex {
    pub spatial: SpatialIndex,
    pub temporal: TemporalIndex,
}

impl SpatioTemporalIndex {
    /// Create a new spatiotemporal index
    pub fn new(tile_ids: Vec<TileId>) -> Self {
        let tiles_with_indices: Vec<(TileId, u32)> = tile_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i as u32))
            .collect();

        Self {
            spatial: SpatialIndex::new(tile_ids),
            temporal: TemporalIndex::new(&tiles_with_indices),
        }
    }

    /// Query tiles by spatial bounds and time range
    pub fn query(
        &self,
        spatial_start: u64,
        spatial_end: u64,
        time_start: u64,
        time_end: u64,
    ) -> Vec<u32> {
        let spatial_results = self.spatial.query_range(spatial_start, spatial_end);
        let temporal_results = self.temporal.query_range(time_start, time_end);

        // Intersect the two result sets
        let spatial_set: std::collections::HashSet<u32> = spatial_results.into_iter().collect();
        temporal_results
            .into_iter()
            .filter(|idx| spatial_set.contains(idx))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_index() {
        let tile_ids = vec![
            TileId::new(10, 512, 384, 1000),
            TileId::new(10, 513, 384, 1000),
            TileId::new(10, 512, 385, 1000),
        ];

        let index = SpatialIndex::new(tile_ids.clone());
        assert_eq!(index.hilbert_ids.len(), 3);
        assert_eq!(index.tile_indices.len(), 3);
    }

    #[test]
    fn test_temporal_index() {
        let tiles = vec![
            (TileId::new(10, 512, 384, 1000), 0),
            (TileId::new(10, 513, 384, 1000), 1),
            (TileId::new(10, 512, 384, 2000), 2),
        ];

        let index = TemporalIndex::new(&tiles);
        assert_eq!(index.timestamps.len(), 2);

        let results = index.query_exact(1000);
        assert_eq!(results.len(), 2);

        let results = index.query_exact(2000);
        assert_eq!(results.len(), 1);
    }

    /// Build a temporal index with timestamps 100, 200, 300 (one tile each).
    fn sample_temporal_index() -> TemporalIndex {
        let tiles = vec![
            (TileId::new(10, 0, 0, 100), 0),
            (TileId::new(10, 0, 0, 200), 1),
            (TileId::new(10, 0, 0, 300), 2),
        ];
        TemporalIndex::new(&tiles)
    }

    #[test]
    fn test_temporal_query_range_empty_result() {
        let index = sample_temporal_index();
        // Range falls entirely between buckets - no match.
        assert!(index.query_range(120, 180).is_empty());
    }

    #[test]
    fn test_temporal_query_range_past_end() {
        let index = sample_temporal_index();
        // Query entirely after the last timestamp.
        assert!(index.query_range(400, 500).is_empty());
        // Query that starts past the end but with a huge end bound.
        assert!(index.query_range(301, u64::MAX).is_empty());
    }

    #[test]
    fn test_temporal_query_range_before_start() {
        let index = sample_temporal_index();
        // Query entirely before the first timestamp.
        assert!(index.query_range(0, 50).is_empty());
    }

    #[test]
    fn test_temporal_query_range_single_match() {
        let index = sample_temporal_index();
        // Range that includes exactly the middle bucket.
        let r = index.query_range(150, 250);
        assert_eq!(r, vec![1]);
        // Exact single timestamp.
        assert_eq!(index.query_range(300, 300), vec![2]);
    }

    #[test]
    fn test_temporal_query_range_spanning_all() {
        let index = sample_temporal_index();
        let mut r = index.query_range(0, u64::MAX);
        r.sort();
        assert_eq!(r, vec![0, 1, 2]);
        // Includes the last bucket (regression: last bucket must not be dropped).
        let mut r2 = index.query_range(100, 300);
        r2.sort();
        assert_eq!(r2, vec![0, 1, 2]);
    }

    #[test]
    fn test_temporal_query_range_reversed() {
        let index = sample_temporal_index();
        // Reversed range must not panic; returns empty.
        assert!(index.query_range(300, 100).is_empty());
    }

    /// Build a spatial index from three tiles at the same zoom.
    fn sample_spatial_index() -> (SpatialIndex, Vec<u64>) {
        let tile_ids = vec![
            TileId::new(10, 100, 100, 0),
            TileId::new(10, 200, 200, 0),
            TileId::new(10, 300, 300, 0),
        ];
        let index = SpatialIndex::new(tile_ids);
        let ids = index.hilbert_ids.clone();
        (index, ids)
    }

    #[test]
    fn test_spatial_query_range_empty_and_bounds() {
        let (index, ids) = sample_spatial_index();

        // Single exact match on the middle hilbert id.
        assert_eq!(index.query_range(ids[1], ids[1]), vec![1]);

        // Range spanning all.
        let mut all = index.query_range(0, u64::MAX);
        all.sort();
        assert_eq!(all, vec![0, 1, 2]);

        // Query past the end - empty, no panic.
        assert!(index.query_range(ids[2] + 1, u64::MAX).is_empty());

        // Query before the start - empty, no panic.
        assert!(index.query_range(0, ids[0].saturating_sub(1)).is_empty());

        // Reversed range - empty, no panic.
        assert!(index.query_range(u64::MAX, 0).is_empty());

        // Empty index.
        let empty = SpatialIndex::new(vec![]);
        assert!(empty.query_range(0, u64::MAX).is_empty());
    }

    #[test]
    fn test_spatiotemporal_index_roundtrip() {
        let tile_ids = vec![
            TileId::new(10, 100, 100, 1000),
            TileId::new(10, 200, 200, 2000),
            TileId::new(10, 300, 300, 3000),
        ];
        let index = SpatioTemporalIndex::new(tile_ids);

        // Temporal-only: all three timestamps present.
        assert_eq!(index.temporal.timestamps.len(), 3);
        let mut t = index.temporal.query_range(0, u64::MAX);
        t.sort();
        assert_eq!(t, vec![0, 1, 2]);
    }
}
