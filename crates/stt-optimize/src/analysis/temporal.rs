//! Temporal distribution and bucketing analysis
//!
//! Analyzes the temporal distribution of features to recommend
//! temporal bucketing and identify patterns.

use crate::loader::LoadedData;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Temporal analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    /// Time range start (Unix ms)
    pub time_start: u64,
    /// Time range end (Unix ms)
    pub time_end: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Duration as human-readable string
    pub duration_human: String,
    /// Number of unique timestamps
    pub unique_timestamps: usize,
    /// Temporal distribution classification
    pub distribution: TemporalDistribution,
    /// Recommended temporal bucket size in milliseconds
    pub recommended_bucket_ms: u64,
    /// Recommended bucket size as human-readable string
    pub recommended_bucket_human: String,
    /// Hourly distribution (24 buckets, 0-23)
    pub hourly_distribution: Vec<u32>,
    /// Daily distribution (7 buckets, 0=Sunday)
    pub daily_distribution: Vec<u32>,
    /// Monthly distribution (12 buckets)
    pub monthly_distribution: Vec<u32>,
    /// Events per day statistics
    pub events_per_day: EventsPerDayStats,
}

impl TemporalAnalysis {
    /// Get time range description
    pub fn time_range_description(&self) -> String {
        let start = DateTime::<Utc>::from_timestamp_millis(self.time_start as i64)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let end = DateTime::<Utc>::from_timestamp_millis(self.time_end as i64)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        format!("{} to {} ({})", start, end, self.duration_human)
    }
}

/// Events per day statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsPerDayStats {
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    pub median: f64,
    pub std_dev: f64,
}

/// Classification of temporal distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalDistribution {
    /// Events distributed uniformly over time
    Uniform,
    /// Events clustered in bursts
    Bursty,
    /// Events follow a periodic pattern (daily, weekly, etc.)
    Periodic,
    /// Events are sparse with long gaps
    Sparse,
    /// Single point in time or very short duration
    Instantaneous,
}

impl std::fmt::Display for TemporalDistribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalDistribution::Uniform => write!(f, "Uniform (evenly distributed)"),
            TemporalDistribution::Bursty => write!(f, "Bursty (clustered in time)"),
            TemporalDistribution::Periodic => write!(f, "Periodic (regular pattern)"),
            TemporalDistribution::Sparse => write!(f, "Sparse (long gaps between events)"),
            TemporalDistribution::Instantaneous => write!(f, "Instantaneous (single moment)"),
        }
    }
}

/// Analyze temporal characteristics of the dataset
pub fn analyze(data: &LoadedData) -> Result<TemporalAnalysis> {
    if data.features.is_empty() {
        return Ok(empty_analysis());
    }

    let time_start = data.time_range.start;
    let time_end = data.time_range.end;
    let duration_ms = time_end.saturating_sub(time_start);

    // Collect all timestamps
    let timestamps: Vec<u64> = data.features.iter().map(|f| f.timestamp).collect();
    let unique_timestamps = {
        let mut ts = timestamps.clone();
        ts.sort();
        ts.dedup();
        ts.len()
    };

    // Calculate hourly, daily, monthly distributions
    let (hourly, daily, monthly) = calculate_distributions(&timestamps);

    // Calculate events per day stats
    let events_per_day = calculate_events_per_day(&timestamps, duration_ms);

    // Classify distribution
    let distribution =
        classify_distribution(&events_per_day, &hourly, unique_timestamps, duration_ms);

    // Recommend bucket size
    let (bucket_ms, bucket_human) = recommend_bucket_size(
        duration_ms,
        unique_timestamps,
        data.features.len(),
        &distribution,
    );

    let duration_human = format_duration(duration_ms);

    Ok(TemporalAnalysis {
        time_start,
        time_end,
        duration_ms,
        duration_human,
        unique_timestamps,
        distribution,
        recommended_bucket_ms: bucket_ms,
        recommended_bucket_human: bucket_human,
        hourly_distribution: hourly,
        daily_distribution: daily,
        monthly_distribution: monthly,
        events_per_day,
    })
}

fn empty_analysis() -> TemporalAnalysis {
    TemporalAnalysis {
        time_start: 0,
        time_end: 0,
        duration_ms: 0,
        duration_human: "0".to_string(),
        unique_timestamps: 0,
        distribution: TemporalDistribution::Instantaneous,
        recommended_bucket_ms: 0,
        recommended_bucket_human: "N/A".to_string(),
        hourly_distribution: vec![0; 24],
        daily_distribution: vec![0; 7],
        monthly_distribution: vec![0; 12],
        events_per_day: EventsPerDayStats {
            min: 0.0,
            max: 0.0,
            avg: 0.0,
            median: 0.0,
            std_dev: 0.0,
        },
    }
}

/// Calculate hourly (0-23), daily (0-6), and monthly (0-11) distributions
fn calculate_distributions(timestamps: &[u64]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut hourly = vec![0u32; 24];
    let mut daily = vec![0u32; 7];
    let mut monthly = vec![0u32; 12];

    for &ts in timestamps {
        if let Some(dt) = DateTime::<Utc>::from_timestamp_millis(ts as i64) {
            let hour = dt.format("%H").to_string().parse::<usize>().unwrap_or(0);
            let weekday = dt.format("%w").to_string().parse::<usize>().unwrap_or(0);
            let month = dt.format("%m").to_string().parse::<usize>().unwrap_or(1) - 1;

            if hour < 24 {
                hourly[hour] += 1;
            }
            if weekday < 7 {
                daily[weekday] += 1;
            }
            if month < 12 {
                monthly[month] += 1;
            }
        }
    }

    (hourly, daily, monthly)
}

/// Calculate events per day statistics
fn calculate_events_per_day(timestamps: &[u64], duration_ms: u64) -> EventsPerDayStats {
    if timestamps.is_empty() || duration_ms == 0 {
        return EventsPerDayStats {
            min: 0.0,
            max: 0.0,
            avg: 0.0,
            median: 0.0,
            std_dev: 0.0,
        };
    }

    // Group by day
    let mut daily_counts: HashMap<i64, u32> = HashMap::new();
    let ms_per_day: i64 = 86_400_000;

    for &ts in timestamps {
        let day = ts as i64 / ms_per_day;
        *daily_counts.entry(day).or_insert(0) += 1;
    }

    if daily_counts.is_empty() {
        return EventsPerDayStats {
            min: 0.0,
            max: 0.0,
            avg: timestamps.len() as f64,
            median: timestamps.len() as f64,
            std_dev: 0.0,
        };
    }

    let counts: Vec<f64> = daily_counts.values().map(|&c| c as f64).collect();
    let n = counts.len() as f64;

    let min = counts.iter().cloned().fold(f64::MAX, f64::min);
    let max = counts.iter().cloned().fold(f64::MIN, f64::max);
    let avg = counts.iter().sum::<f64>() / n;

    let mut sorted_counts = counts.clone();
    sorted_counts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_counts[sorted_counts.len() / 2];

    let variance = counts.iter().map(|c| (c - avg).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    EventsPerDayStats {
        min,
        max,
        avg,
        median,
        std_dev,
    }
}

/// Classify temporal distribution based on statistics
fn classify_distribution(
    events_per_day: &EventsPerDayStats,
    hourly: &[u32],
    unique_timestamps: usize,
    duration_ms: u64,
) -> TemporalDistribution {
    // Very short duration = instantaneous
    let one_day_ms = 86_400_000u64;
    if duration_ms < one_day_ms {
        return TemporalDistribution::Instantaneous;
    }

    // Few unique timestamps over long period = sparse
    let _expected_unique = duration_ms / 60_000; // One per minute
    if unique_timestamps < 100 && duration_ms > one_day_ms * 30 {
        return TemporalDistribution::Sparse;
    }

    // High coefficient of variation = bursty
    if events_per_day.std_dev > events_per_day.avg * 1.5 {
        return TemporalDistribution::Bursty;
    }

    // Check for periodicity in hourly distribution
    let hourly_max = hourly.iter().max().copied().unwrap_or(0) as f64;
    let hourly_min = hourly.iter().min().copied().unwrap_or(0) as f64;
    let hourly_avg = hourly.iter().map(|&h| h as f64).sum::<f64>() / 24.0;

    if hourly_avg > 0.0 {
        let hourly_variation = (hourly_max - hourly_min) / hourly_avg;
        if hourly_variation > 2.0 {
            return TemporalDistribution::Periodic;
        }
    }

    TemporalDistribution::Uniform
}

/// A watchable playback loop's target length in seconds — the number of buckets
/// (frames) worth animating is derived from this × the frame rate, not a bare
/// "aim for ~1500 buckets" constant. Matches the 20..90s band the style-hints
/// `suggested_playback_seconds` clamps to.
const TARGET_PLAYBACK_SECONDS: f64 = 45.0;
/// Frames per second the derived bucket-count target assumes for playback.
const PLAYBACK_FPS: f64 = 30.0;
/// Don't bucket so finely that the average bucket holds fewer than this many
/// features — over-fine buckets just fragment the archive without adding
/// visible temporal detail.
const MIN_FEATURES_PER_BUCKET: u64 = 10;

/// Recommend a temporal bucket size.
///
/// The target bucket (frame) count is DERIVED from a watchable playback loop
/// (`TARGET_PLAYBACK_SECONDS × PLAYBACK_FPS`, scaled by the distribution), then
/// bounded by the two data facts the caller actually measured — never bucket
/// finer than there are distinct timestamps, nor so fine that the average bucket
/// falls below [`MIN_FEATURES_PER_BUCKET`] features. Drop either bound and the
/// recommendation stops reacting to timestamp resolution or feature density
/// and just echoes the playback target.
fn recommend_bucket_size(
    duration_ms: u64,
    unique_timestamps: usize,
    feature_count: usize,
    distribution: &TemporalDistribution,
) -> (u64, String) {
    if duration_ms == 0 {
        return (0, "N/A".to_string());
    }

    // Standard bucket sizes in milliseconds
    let bucket_sizes = [
        (1_000, "1 second"),
        (60_000, "1 minute"),
        (300_000, "5 minutes"),
        (600_000, "10 minutes"),
        (900_000, "15 minutes"),
        (1_800_000, "30 minutes"),
        (3_600_000, "1 hour"),
        (7_200_000, "2 hours"),
        (14_400_000, "4 hours"),
        (21_600_000, "6 hours"),
        (43_200_000, "12 hours"),
        (86_400_000, "1 day"),
        (604_800_000, "1 week"),
        (2_592_000_000, "30 days"),
    ];

    // Frames a watchable loop wants, scaled by how much temporal structure the
    // distribution carries: bursty data rewards finer buckets, sparse data
    // coarser ones.
    let frame_factor = match distribution {
        TemporalDistribution::Bursty => 1.5,
        TemporalDistribution::Sparse => 0.5,
        _ => 1.0,
    };
    let mut target_buckets = (TARGET_PLAYBACK_SECONDS * PLAYBACK_FPS * frame_factor).round() as u64;

    // Bound by measured data facts:
    //  - never more buckets than distinct timestamps (finer is empty resolution);
    //  - never so fine the average bucket dips below MIN_FEATURES_PER_BUCKET.
    if unique_timestamps > 0 {
        target_buckets = target_buckets.min(unique_timestamps as u64);
    }
    if feature_count > 0 {
        let by_density = (feature_count as u64 / MIN_FEATURES_PER_BUCKET).max(1);
        target_buckets = target_buckets.min(by_density);
    }
    let target_buckets = target_buckets.max(1);

    // Find the FINEST standard size whose bucket count fits the target.
    for (size_ms, name) in bucket_sizes.iter() {
        let bucket_count = duration_ms / size_ms;
        if bucket_count <= target_buckets {
            return (*size_ms, name.to_string());
        }
    }

    // Even the coarsest standard size overshoots the target (long span, few
    // features) — return the coarsest, never a finer size than what was just
    // rejected as too fine-grained.
    let (size_ms, name) = bucket_sizes[bucket_sizes.len() - 1];
    (size_ms, name.to_string())
}

/// Format duration as human-readable string
fn format_duration(ms: u64) -> String {
    let seconds = ms / 1000;
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let days = hours / 24;
    let months = days / 30;
    let years = days / 365;

    if years > 0 {
        let remaining_months = (days - years * 365) / 30;
        if remaining_months > 0 {
            format!("{} years, {} months", years, remaining_months)
        } else {
            format!("{} years", years)
        }
    } else if months > 0 {
        let remaining_days = days - months * 30;
        if remaining_days > 0 {
            format!("{} months, {} days", months, remaining_days)
        } else {
            format!("{} months", months)
        }
    } else if days > 0 {
        format!("{} days", days)
    } else if hours > 0 {
        format!("{} hours", hours)
    } else if minutes > 0 {
        format!("{} minutes", minutes)
    } else {
        format!("{} seconds", seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(1000), "1 seconds");
        assert_eq!(format_duration(3600000), "1 hours");
        assert_eq!(format_duration(86400000), "1 days");
        assert_eq!(format_duration(86400000 * 365), "1 years");
    }

    #[test]
    fn test_recommend_bucket_size() {
        // 1 year duration
        let one_year = 365 * 86_400_000u64;
        let (bucket, _) =
            recommend_bucket_size(one_year, 10000, 100000, &TemporalDistribution::Uniform);
        assert!(bucket >= 3_600_000); // At least 1 hour
    }

    #[test]
    fn test_recommend_bucket_targets_a_playback_loop() {
        // Uniform target = 45s × 30fps = 1350 frames, not capped here (5000
        // distinct timestamps, 50000 features both leave it at 1350). Over a
        // 30-day span the chosen bucket must land at-or-under that target.
        let span = 30 * 86_400_000u64; // 30 days
        let target = (super::TARGET_PLAYBACK_SECONDS * super::PLAYBACK_FPS).round() as u64; // 1350
        let (bucket, name) =
            recommend_bucket_size(span, 5000, 50000, &TemporalDistribution::Uniform);
        assert!(bucket > 0, "bucket must be non-zero for a real span");
        let bucket_count = span / bucket;
        assert!(
            bucket_count <= target,
            "30-day span chose {} ({} buckets), exceeds target {}",
            name,
            bucket_count,
            target
        );
        // 30 days / 1350 ≈ 32 min, so 30-min (1440 buckets) is too fine; the
        // finest standard size at-or-under target is 1 hour (720 buckets).
        assert_eq!(bucket, 3_600_000, "expected 1-hour bucket, got {}", name);
    }

    #[test]
    fn test_bucket_size_respects_unique_timestamps() {
        // Same span/features, but only 100 distinct timestamps: the target can't
        // exceed that, so the bucket must be COARSER than with many timestamps
        // — `unique_timestamps` has to reach the result.
        let span = 365 * 86_400_000u64; // 1 year
        let (many, _) =
            recommend_bucket_size(span, 100_000, 1_000_000, &TemporalDistribution::Uniform);
        let (few, _) = recommend_bucket_size(span, 100, 1_000_000, &TemporalDistribution::Uniform);
        assert!(
            few > many,
            "100 distinct timestamps should coarsen the bucket ({few} vs {many})",
        );
    }

    #[test]
    fn test_bucket_size_respects_feature_density() {
        // A tiny feature count caps buckets so they don't average < 10 features,
        // forcing a coarser bucket than a dense dataset over the same span
        // — `feature_count` has to reach the result.
        let span = 365 * 86_400_000u64; // 1 year
        let (dense, _) =
            recommend_bucket_size(span, 100_000, 1_000_000, &TemporalDistribution::Uniform);
        let (sparse_features, _) =
            recommend_bucket_size(span, 100_000, 200, &TemporalDistribution::Uniform);
        assert!(
            sparse_features > dense,
            "200 features should coarsen the bucket ({sparse_features} vs {dense})",
        );
    }

    #[test]
    fn test_bucket_size_fallback_is_coarsest_not_finer() {
        // Long span + few features: the density cap (10 features → target 1)
        // rejects EVERY standard size, so the fallback must return the
        // coarsest one (30 days) — never a finer bucket than what the loop
        // just rejected as producing too many buckets.
        let span = 10 * 365 * 86_400_000u64; // 10 years
        let (bucket, name) = recommend_bucket_size(span, 100, 100, &TemporalDistribution::Uniform);
        assert_eq!(
            bucket, 2_592_000_000,
            "expected 30-day fallback, got {name}"
        );
    }

    #[test]
    fn test_recommend_bucket_zero_for_empty_span() {
        let (bucket, name) = recommend_bucket_size(0, 0, 0, &TemporalDistribution::Instantaneous);
        assert_eq!(bucket, 0);
        assert_eq!(name, "N/A");
    }

    #[test]
    fn test_recommend_bucket_sparse_uses_fewer_buckets() {
        // Sparse distribution targets only 500 buckets, so for the same span it
        // should choose a coarser (>=) bucket than a uniform distribution.
        let span = 365 * 86_400_000u64; // 1 year
        let (uniform, _) = recommend_bucket_size(span, 1000, 10000, &TemporalDistribution::Uniform);
        let (sparse, _) = recommend_bucket_size(span, 1000, 10000, &TemporalDistribution::Sparse);
        assert!(
            sparse >= uniform,
            "sparse bucket {} should be >= uniform bucket {}",
            sparse,
            uniform
        );
    }
}
