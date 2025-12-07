//! Performance analysis tool for STT archives
//!
//! This module provides utilities to analyze and profile
//! spatiotemporal tile archives for performance bottlenecks.

use crate::archive::ArchiveReader;
use crate::error::Result;
use crate::tile::TileId;
use std::collections::HashMap;

/// Archive performance statistics
#[derive(Debug, Clone)]
pub struct ArchiveStats {
    /// Total number of tiles
    pub tile_count: usize,
    /// Total number of features across all tiles
    pub total_features: usize,
    /// Total uncompressed size
    pub uncompressed_size: u64,
    /// Total compressed size
    pub compressed_size: u64,
    /// Per-zoom statistics
    pub zoom_stats: HashMap<u8, ZoomStats>,
    /// Temporal distribution
    pub temporal_stats: TemporalStats,
    /// Tile size distribution
    pub size_distribution: SizeDistribution,
    /// Feature density statistics
    pub density_stats: DensityStats,
}

/// Statistics for a specific zoom level
#[derive(Debug, Clone, Default)]
pub struct ZoomStats {
    pub tile_count: usize,
    pub feature_count: usize,
    pub uncompressed_size: u64,
    pub compressed_size: u64,
    pub avg_features_per_tile: f64,
    pub avg_tile_size: f64,
}

/// Temporal distribution statistics
#[derive(Debug, Clone, Default)]
pub struct TemporalStats {
    /// Start timestamp
    pub time_start: u64,
    /// End timestamp
    pub time_end: u64,
    /// Number of unique timestamps
    pub unique_timestamps: usize,
    /// Average tiles per timestamp
    pub avg_tiles_per_timestamp: f64,
    /// Temporal bucketing analysis
    pub suggested_bucket_size: u64,
}

/// Tile size distribution
#[derive(Debug, Clone, Default)]
pub struct SizeDistribution {
    pub min_size: u64,
    pub max_size: u64,
    pub avg_size: f64,
    pub median_size: u64,
    pub p95_size: u64,
    pub p99_size: u64,
    /// Histogram of tile sizes (bucket: count)
    pub histogram: Vec<(u64, usize)>,
}

/// Feature density statistics
#[derive(Debug, Clone, Default)]
pub struct DensityStats {
    pub min_features: usize,
    pub max_features: usize,
    pub avg_features: f64,
    pub median_features: usize,
    /// Tiles with too many features (> recommended)
    pub oversized_tiles: usize,
    /// Tiles with too few features (< 10)
    pub undersized_tiles: usize,
}

impl ArchiveStats {
    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.uncompressed_size == 0 {
            return 1.0;
        }
        self.compressed_size as f64 / self.uncompressed_size as f64
    }

    /// Get recommendations for improvement
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check compression ratio
        if self.compression_ratio() > 0.7 {
            recommendations.push(
                format!(
                    "Poor compression ratio ({:.2}). Consider using Brotli compression or increasing temporal bucketing.",
                    self.compression_ratio()
                )
            );
        }

        // Check tile size distribution
        if self.size_distribution.p99_size > 512 * 1024 {
            recommendations.push(
                format!(
                    "Large P99 tile size ({:.2} KB). Enable tile size budgets or increase simplification.",
                    self.size_distribution.p99_size as f64 / 1024.0
                )
            );
        }

        // Check temporal bucketing
        if self.temporal_stats.unique_timestamps > 10000 {
            let bucket_ms = self.temporal_stats.suggested_bucket_size;
            recommendations.push(format!(
                "Too many unique timestamps ({}). Consider temporal bucketing ({} ms).",
                self.temporal_stats.unique_timestamps, bucket_ms
            ));
        }

        // Check feature density
        if self.density_stats.oversized_tiles > 0 {
            recommendations.push(format!(
                "{} tiles exceed recommended feature count. Enable tile size budgets.",
                self.density_stats.oversized_tiles
            ));
        }

        // Check for undersized tiles
        if self.density_stats.undersized_tiles > self.tile_count / 10 {
            recommendations.push(
                format!(
                    "{} tiles have very few features. Consider reducing max zoom or adjusting temporal bucketing.",
                    self.density_stats.undersized_tiles
                )
            );
        }

        recommendations
    }

    /// Print a detailed report
    pub fn print_report(&self) {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("           SpatioTemporal Tiles Performance Report");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        println!("📊 Overall Statistics");
        println!("  Tiles:             {}", self.tile_count);
        println!("  Features:          {}", self.total_features);
        println!(
            "  Uncompressed:      {:.2} MB",
            self.uncompressed_size as f64 / 1_048_576.0
        );
        println!(
            "  Compressed:        {:.2} MB",
            self.compressed_size as f64 / 1_048_576.0
        );
        println!(
            "  Compression Ratio: {:.2}x",
            1.0 / self.compression_ratio()
        );
        println!();

        println!("🔍 Tile Size Distribution");
        println!(
            "  Min:     {:.2} KB",
            self.size_distribution.min_size as f64 / 1024.0
        );
        println!(
            "  Avg:     {:.2} KB",
            self.size_distribution.avg_size / 1024.0
        );
        println!(
            "  Median:  {:.2} KB",
            self.size_distribution.median_size as f64 / 1024.0
        );
        println!(
            "  P95:     {:.2} KB",
            self.size_distribution.p95_size as f64 / 1024.0
        );
        println!(
            "  P99:     {:.2} KB",
            self.size_distribution.p99_size as f64 / 1024.0
        );
        println!(
            "  Max:     {:.2} KB",
            self.size_distribution.max_size as f64 / 1024.0
        );
        println!();

        println!("📈 Feature Density");
        println!("  Min features/tile:  {}", self.density_stats.min_features);
        println!(
            "  Avg features/tile:  {:.1}",
            self.density_stats.avg_features
        );
        println!(
            "  Median:             {}",
            self.density_stats.median_features
        );
        println!("  Max features/tile:  {}", self.density_stats.max_features);
        println!(
            "  Oversized tiles:    {}",
            self.density_stats.oversized_tiles
        );
        println!(
            "  Undersized tiles:   {}",
            self.density_stats.undersized_tiles
        );
        println!();

        println!("⏰ Temporal Distribution");
        println!(
            "  Unique timestamps:  {}",
            self.temporal_stats.unique_timestamps
        );
        println!(
            "  Avg tiles/time:     {:.1}",
            self.temporal_stats.avg_tiles_per_timestamp
        );
        if self.temporal_stats.suggested_bucket_size > 0 {
            println!(
                "  Suggested bucket:   {} ms",
                self.temporal_stats.suggested_bucket_size
            );
        }
        println!();

        if !self.zoom_stats.is_empty() {
            println!("🔎 Per-Zoom Statistics");
            let mut zooms: Vec<_> = self.zoom_stats.keys().collect();
            zooms.sort();
            for zoom in zooms {
                let stats = &self.zoom_stats[zoom];
                println!(
                    "  Z{:2}: {:5} tiles, {:8} features, {:.2} MB compressed",
                    zoom,
                    stats.tile_count,
                    stats.feature_count,
                    stats.compressed_size as f64 / 1_048_576.0
                );
            }
            println!();
        }

        let recommendations = self.get_recommendations();
        if !recommendations.is_empty() {
            println!("💡 Recommendations");
            for (i, rec) in recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, rec);
            }
            println!();
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

/// Analyzer for STT archives
pub struct ArchiveAnalyzer;

impl ArchiveAnalyzer {
    /// Analyze an archive and return statistics
    pub fn analyze(reader: &mut ArchiveReader) -> Result<ArchiveStats> {
        let metadata = reader.metadata();

        // Initialize statistics
        let mut stats = ArchiveStats {
            tile_count: 0,
            total_features: 0,
            uncompressed_size: 0,
            compressed_size: 0,
            zoom_stats: HashMap::new(),
            temporal_stats: TemporalStats::default(),
            size_distribution: SizeDistribution::default(),
            density_stats: DensityStats::default(),
        };

        // Collect tile information
        let mut tile_sizes: Vec<u64> = Vec::new();
        let mut feature_counts: Vec<usize> = Vec::new();
        let mut timestamps: std::collections::HashSet<u64> = std::collections::HashSet::new();

        // Get tile entries from index
        let index = reader.get_index();

        for entry in &index.tiles {
            let tile_id = TileId::new(entry.zoom as u8, entry.x, entry.y, entry.time_start);

            stats.tile_count += 1;
            stats.total_features += entry.feature_count as usize;
            stats.compressed_size += entry.length as u64;
            stats.uncompressed_size += entry.uncompressed_size as u64;

            // Update zoom stats
            let zoom_stat = stats
                .zoom_stats
                .entry(tile_id.z)
                .or_insert_with(ZoomStats::default);

            zoom_stat.tile_count += 1;
            zoom_stat.feature_count += entry.feature_count as usize;
            zoom_stat.compressed_size += entry.length as u64;
            zoom_stat.uncompressed_size += entry.uncompressed_size as u64;

            // Collect distributions
            tile_sizes.push(entry.length as u64);
            feature_counts.push(entry.feature_count as usize);
            timestamps.insert(entry.time_start);
        }

        // Calculate per-zoom averages
        for zoom_stat in stats.zoom_stats.values_mut() {
            if zoom_stat.tile_count > 0 {
                zoom_stat.avg_features_per_tile =
                    zoom_stat.feature_count as f64 / zoom_stat.tile_count as f64;
                zoom_stat.avg_tile_size =
                    zoom_stat.compressed_size as f64 / zoom_stat.tile_count as f64;
            }
        }

        // Calculate temporal stats
        stats.temporal_stats.unique_timestamps = timestamps.len();
        if timestamps.len() > 0 {
            let time_vec: Vec<_> = timestamps.iter().copied().collect();
            stats.temporal_stats.time_start = *time_vec.iter().min().unwrap();
            stats.temporal_stats.time_end = *time_vec.iter().max().unwrap();
            stats.temporal_stats.avg_tiles_per_timestamp =
                stats.tile_count as f64 / timestamps.len() as f64;

            // Suggest bucket size based on distribution
            let time_span = stats.temporal_stats.time_end - stats.temporal_stats.time_start;
            if time_span > 0 && timestamps.len() > 1000 {
                // Suggest bucketing to reduce to ~1000-2000 timestamps
                stats.temporal_stats.suggested_bucket_size = time_span / 1500; // Target ~1500 buckets
            }
        }

        // Calculate size distribution
        if !tile_sizes.is_empty() {
            tile_sizes.sort();
            stats.size_distribution.min_size = tile_sizes[0];
            stats.size_distribution.max_size = tile_sizes[tile_sizes.len() - 1];
            stats.size_distribution.avg_size =
                tile_sizes.iter().sum::<u64>() as f64 / tile_sizes.len() as f64;
            stats.size_distribution.median_size = tile_sizes[tile_sizes.len() / 2];
            stats.size_distribution.p95_size =
                tile_sizes[((tile_sizes.len() as f64 * 0.95) as usize).min(tile_sizes.len() - 1)];
            stats.size_distribution.p99_size =
                tile_sizes[((tile_sizes.len() as f64 * 0.99) as usize).min(tile_sizes.len() - 1)];
        }

        // Calculate density stats
        if !feature_counts.is_empty() {
            feature_counts.sort();
            stats.density_stats.min_features = feature_counts[0];
            stats.density_stats.max_features = feature_counts[feature_counts.len() - 1];
            stats.density_stats.avg_features =
                feature_counts.iter().sum::<usize>() as f64 / feature_counts.len() as f64;
            stats.density_stats.median_features = feature_counts[feature_counts.len() / 2];

            // Count problematic tiles
            stats.density_stats.oversized_tiles =
                feature_counts.iter().filter(|&&c| c > 10_000).count();
            stats.density_stats.undersized_tiles =
                feature_counts.iter().filter(|&&c| c < 10).count();
        }

        Ok(stats)
    }
}

// Use the new index() method instead of accessing private field
impl ArchiveReader {
    pub fn get_index(&self) -> &crate::proto::Index {
        self.index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_ratio() {
        let stats = ArchiveStats {
            tile_count: 100,
            total_features: 10000,
            uncompressed_size: 10_000_000,
            compressed_size: 2_500_000,
            zoom_stats: HashMap::new(),
            temporal_stats: TemporalStats::default(),
            size_distribution: SizeDistribution::default(),
            density_stats: DensityStats::default(),
        };

        assert_eq!(stats.compression_ratio(), 0.25);
    }

    #[test]
    fn test_recommendations() {
        let mut stats = ArchiveStats {
            tile_count: 1000,
            total_features: 1_000_000,
            uncompressed_size: 100_000_000,
            compressed_size: 80_000_000, // Poor compression (0.8)
            zoom_stats: HashMap::new(),
            temporal_stats: TemporalStats {
                unique_timestamps: 50000, // Too many timestamps
                ..Default::default()
            },
            size_distribution: SizeDistribution {
                p99_size: 1_000_000, // Large P99 (1MB)
                ..Default::default()
            },
            density_stats: DensityStats {
                oversized_tiles: 100, // Some oversized tiles
                ..Default::default()
            },
        };

        let recommendations = stats.get_recommendations();

        // Should have multiple recommendations
        assert!(!recommendations.is_empty());
        assert!(recommendations.len() >= 3);
    }
}
