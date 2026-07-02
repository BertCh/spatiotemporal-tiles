//! Shared timestamp-unit normalization for all STT input adaptors.
//!
//! Every input reader — the GeoParquet scalar (`--time-field`) and per-vertex
//! (`vertex_timestamps`) paths in `stt-build`, the DuckDB reader, and the
//! `stt-optimize` analysis loader — must scale a producer's raw integer
//! timestamp to the STT wire unit (Unix **milliseconds**) identically. This is
//! the one source of truth for that arithmetic so a new reader can't silently
//! disagree.

use crate::error::{Error, Result};

/// Precision of a raw integer timestamp value. Used to normalize any producer's
/// unit to the STT wire unit (Unix **milliseconds**).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampUnit {
    /// Seconds since the Unix epoch (×1000 → ms).
    Second,
    /// Milliseconds since the Unix epoch (passthrough — the wire unit).
    Millisecond,
    /// Microseconds since the Unix epoch (÷1_000 → ms).
    Microsecond,
    /// Nanoseconds since the Unix epoch (÷1_000_000 → ms).
    Nanosecond,
}

/// Scale a raw timestamp `value` in `unit` to **signed** Unix milliseconds
/// WITHOUT the non-negative guard. Second→ms is overflow-CHECKED (any realistic
/// date fits; a value that doesn't is corrupt input and must error, never
/// silently saturate). This is the single source of truth for timestamp-unit
/// scaling arithmetic — `Second` ×1000, `Millisecond` passthrough, `Microsecond`
/// ÷1_000, `Nanosecond` ÷1_000_000 — shared by [`normalize_timestamp_to_ms`]
/// (which adds the guard + `u64` cast) and the DuckDB reader's
/// `timestamp_unit_to_ms` (which keeps signed ms so a timestamp emitted as a
/// plain *property* may legitimately be pre-1970). Callers that need a wire
/// timestamp apply [`reject_negative_timestamp`] themselves.
pub fn scale_timestamp_to_ms(value: i64, unit: TimestampUnit) -> Result<i64> {
    match unit {
        TimestampUnit::Second => value.checked_mul(1_000).ok_or_else(|| {
            Error::Other(format!(
                "second-precision timestamp {value} overflows the millisecond range"
            ))
        }),
        TimestampUnit::Millisecond => Ok(value),
        TimestampUnit::Microsecond => Ok(value / 1_000),
        TimestampUnit::Nanosecond => Ok(value / 1_000_000),
    }
}

/// Pre-1970 timestamps cannot be represented — the temporal index stores
/// unsigned ms-since-epoch, so a negative value would wrap to a huge
/// positive one (`as u64`) and silently corrupt the index. This hard-errors
/// regardless of strictness mode; coercion is never sound here.
pub fn reject_negative_timestamp(row: usize, value: i64) -> Result<()> {
    if value < 0 {
        return Err(Error::Other(format!(
            "row {row}: negative timestamp {value} (pre-1970). The STT temporal \
             index stores unsigned ms-since-epoch and cannot represent pre-1970 \
             times; filter or re-epoch these rows before building."
        )));
    }
    Ok(())
}

/// Normalize a raw integer timestamp `value` in `unit` to Unix **milliseconds**,
/// rejecting pre-1970 (negative) instants and overflow-checking the
/// second→millisecond multiply. `row` is the absolute row index, used only for
/// error context.
///
/// The single shared helper the GeoParquet scalar (`--time-field`) and
/// per-vertex (`vertex_timestamps`) readers call; the DuckDB reader agrees on
/// the scaling via [`scale_timestamp_to_ms`]. See [`TimestampUnit`].
pub fn normalize_timestamp_to_ms(row: usize, value: i64, unit: TimestampUnit) -> Result<u64> {
    reject_negative_timestamp(row, value)?;
    // Only Second can overflow (ms passthrough / µs ÷1e3 / ns ÷1e6 shrink or
    // preserve); re-wrap the shared error with row context.
    let ms = scale_timestamp_to_ms(value, unit).map_err(|_| {
        Error::Other(format!(
            "row {row}: second-precision timestamp {value} overflows the millisecond range"
        ))
    })?;
    Ok(ms as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_matches_units() {
        assert_eq!(scale_timestamp_to_ms(5, TimestampUnit::Second).unwrap(), 5000);
        assert_eq!(scale_timestamp_to_ms(5, TimestampUnit::Millisecond).unwrap(), 5);
        assert_eq!(scale_timestamp_to_ms(5_000, TimestampUnit::Microsecond).unwrap(), 5);
        assert_eq!(scale_timestamp_to_ms(5_000_000, TimestampUnit::Nanosecond).unwrap(), 5);
    }

    #[test]
    fn scale_second_overflow_errors() {
        assert!(scale_timestamp_to_ms(i64::MAX, TimestampUnit::Second).is_err());
        assert!(scale_timestamp_to_ms(i64::MIN, TimestampUnit::Second).is_err());
    }

    #[test]
    fn normalize_rejects_negative() {
        assert!(normalize_timestamp_to_ms(0, -1, TimestampUnit::Millisecond).is_err());
        assert_eq!(
            normalize_timestamp_to_ms(0, 1_000, TimestampUnit::Second).unwrap(),
            1_000_000
        );
    }

    #[test]
    fn normalize_second_overflow_errors() {
        assert!(normalize_timestamp_to_ms(0, i64::MAX, TimestampUnit::Second).is_err());
    }
}
