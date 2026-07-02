//! Decode logic shared by the PostGIS and DuckDB input adaptors.
//!
//! Both DB readers must turn rows into [`crate::input::ParsedFeature`]s with
//! IDENTICAL semantics — the cross-source parity suite
//! (`tests/source_parity.rs`) enforces it. Every rule here lives once instead
//! of being hand-copied per adaptor, so a new engine (or an edit to one
//! reader) can't silently diverge from the other.

use anyhow::Result;

use crate::input::{ParsedFeature, TimeFormat};
use stt_core::timestamp::{scale_timestamp_to_ms, TimestampUnit};

/// Outcome of decoding one DB row: a feature, or a skip because the row's
/// geometry was null/unparseable (Warn mode — Strict bails at the decoder).
pub(crate) enum RowOutcome {
    Feature(Box<ParsedFeature>),
    GeomSkip,
}

/// Tally of per-vertex array elements silently defaulted while decoding (a NULL
/// or unmappable element becomes `0` for a timestamp array / `f32::NAN` for a
/// value array, preserving per-vertex alignment — same rule the GeoParquet
/// reader uses). Summarised once at end-of-read so it doesn't spam per row.
#[derive(Default)]
pub(crate) struct VertexCoercions {
    /// Timestamp-array elements coerced to `0`.
    pub timestamps: usize,
    /// Value/matrix-array elements coerced to `f32::NAN`.
    pub values: usize,
}

impl VertexCoercions {
    pub(crate) fn warn(&self, source: &str) {
        if self.timestamps > 0 {
            tracing::warn!(
                "{} per-vertex {source} timestamp element(s) were NULL/unmappable and defaulted \
                 to 0 (epoch); check vertex_timestamps for gaps",
                self.timestamps
            );
        }
        if self.values > 0 {
            tracing::warn!(
                "{} per-vertex {source} value element(s) were NULL/unmappable and defaulted to NaN; \
                 check vertex_values / vertex_value_matrix for gaps",
                self.values
            );
        }
    }
}

/// Warn once about property columns present in the source but carrying no value
/// in any row — the silent-drop cases (unmappable column type, or entirely NULL)
/// made visible so a missing tile column isn't a mystery. `props` yields each
/// property column's `(name, optional type string)`; the type is reported so an
/// operator can tell an unmappable-type drop from an all-NULL column at a
/// glance, and `unmappable_examples` names the engine's typical unmappable
/// types. Shared by both DB readers' ingest paths; the per-tile serve decoders
/// don't call it (no spam).
pub(crate) fn warn_dropped_columns<'a, I>(
    props: I,
    seen_props: &std::collections::HashSet<String>,
    total_rows: usize,
    source: &str,
    unmappable_examples: &str,
) where
    I: IntoIterator<Item = (&'a str, Option<String>)>,
{
    let dropped: Vec<String> = props
        .into_iter()
        .filter(|(name, _)| !seen_props.contains(*name))
        .map(|(name, ty)| match ty {
            Some(ty) => format!("{name} ({ty})"),
            None => name.to_string(),
        })
        .collect();
    if !dropped.is_empty() {
        tracing::warn!(
            "{} {source} source column(s) carried no value in any of {total_rows} rows and were \
             dropped from tiles (unmappable column type — e.g. {unmappable_examples} — or \
             entirely NULL): {}",
            dropped.len(),
            dropped.join(", ")
        );
    }
}

/// Integer time column → ms, per `--time-format` (matches the GeoParquet path,
/// including the overflow-CHECKED second→ms multiply — never saturates). `row`
/// is the absolute row index, used only for error context.
pub(crate) fn apply_int_time_format(v: i64, time_format: TimeFormat, row: usize) -> Result<i64> {
    match time_format {
        TimeFormat::UnixSec => scale_timestamp_to_ms(v, TimestampUnit::Second).map_err(|_| {
            anyhow::anyhow!(
                "row {row}: second-precision timestamp {v} overflows the millisecond range"
            )
        }),
        // Int + iso8601 falls back to unix-ms, same as the GeoParquet reader.
        TimeFormat::UnixMs | TimeFormat::Iso8601 => Ok(v),
    }
}

/// A finite `f64` → JSON number; NaN/Inf → JSON `null` (key retained). Matches
/// the GeoParquet reader, where `serde_json::json!(nan)` yields `Value::Null`
/// rather than dropping the key.
pub(crate) fn json_number_or_null(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_time_format_mapping() {
        assert_eq!(apply_int_time_format(5, TimeFormat::UnixSec, 0).unwrap(), 5000);
        assert_eq!(apply_int_time_format(5, TimeFormat::UnixMs, 0).unwrap(), 5);
        assert_eq!(apply_int_time_format(5, TimeFormat::Iso8601, 0).unwrap(), 5);
    }

    #[test]
    fn int_time_format_second_overflow_errors_with_row_context() {
        let err = apply_int_time_format(i64::MAX, TimeFormat::UnixSec, 7).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("row 7"), "{msg}");
        assert!(msg.contains("overflows"), "{msg}");
    }

    #[test]
    fn nan_float_becomes_json_null() {
        assert_eq!(json_number_or_null(f64::NAN), serde_json::Value::Null);
        assert_eq!(json_number_or_null(1.5), serde_json::json!(1.5));
    }
}
