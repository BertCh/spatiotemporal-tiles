//! The LOD-grid ↔ `--temporal-bucket` precondition (SH-4).
//!
//! # The coupling this closes
//!
//! A baked per-feature LOD floor column (`--min-zoom-field`, e.g. the
//! `lod_min_zoom` the MRMS/storm generation scripts write) is computed against a
//! **bucket-keyed** thinning grid: a cell is "one screen-resolution cell of one
//! TEMPORAL BUCKET". The bucket width used to compute those floors and the
//! archive's `--temporal-bucket` are two constants in two codebases, and nothing
//! asserted them equal.
//!
//! When they decouple, the archive still builds, still validates, and still
//! renders — it just renders the WRONG DENSITY: the 2026-07-28 incident with a
//! space-only grid over animated data showed z8 tiles carrying a median 13 %
//! (worst 0 %) of the features actually visible in the displayed bucket. There
//! is no decode-time signal for it; the only place both codebases meet is the
//! Parquet key-value footer.
//!
//! # The contract
//!
//! A generation script that writes a LOD-floor column stamps
//! [`LOD_GRID_BUCKET_KEY`] into the Parquet file footer with the bucket width
//! (in ms) its grid was keyed on. `stt-build` reads it back and compares against
//! the **resolved** temporal bucket — resolved, because `--auto` may overwrite a
//! hand-passed `--temporal-bucket`, and an auto-overridden bucket is exactly the
//! silent-mismatch path a hand-passed flag never hits.
//!
//! | LOD column flag | footer key | outcome |
//! |---|---|---|
//! | set    | present, equal   | pass, silent |
//! | set    | present, differs | **hard error** naming both values and both sources |
//! | set    | absent           | warn (legacy input: nothing asserts) |
//! | unset  | present          | warn (the input was baked for a grid this build ignores) |
//! | unset  | absent           | pass, silent |
//!
//! # Byte status
//!
//! Pure precondition. A PASSING build is bit-identical to one without the check:
//! nothing here touches features, tiles, encoding or metadata. Rollback is
//! deleting the call site.
//!
//! # Scope
//!
//! Parquet inputs only — Postgres/DuckDB have no LOD-column pipeline today, and
//! a `None` footer there takes the warn path. The `stt-optimize` voxel/home-zoom
//! ladders are out of scope: their grids are per-sweep SPATIAL, not bucket-keyed.

use std::path::Path;

/// Parquet file-footer key carrying the temporal bucket width (ms, decimal
/// string) a baked LOD-floor column's thinning grid was keyed on.
///
/// `stt:`-prefixed like the other STT footer/schema keys
/// (`stt:vertex_time_origin_ms`, `stt:vertex_value_buckets`).
pub const LOD_GRID_BUCKET_KEY: &str = "stt:lod_grid_bucket_ms";

/// What [`check_lod_grid_bucket`] concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LodBucketCheck {
    /// Nothing to assert, or the two agree. Build proceeds silently.
    Ok,
    /// Build proceeds, but something is un-asserted or suspicious.
    Warn(String),
    /// The baked floors are wrong for this archive. Build must stop.
    Mismatch(String),
}

/// Compare a baked LOD grid's temporal bucket against this build's **resolved**
/// `--temporal-bucket`.
///
/// * `stamped` — the footer value, `None` when the input declares nothing.
/// * `lod_column` — the `--min-zoom-field` column name, `None` when the build
///   does not consume a baked LOD floor.
/// * `resolved_bucket_ms` — the bucket this build will actually tile at, AFTER
///   any `--auto` / recipe override. Passing the pre-override value would defeat
///   the check on precisely the path it exists for.
///
/// Pure: no I/O, no clock. Same inputs, same verdict.
pub fn check_lod_grid_bucket(
    stamped: Option<u64>,
    lod_column: Option<&str>,
    resolved_bucket_ms: u64,
) -> LodBucketCheck {
    match (stamped, lod_column) {
        (Some(stamped), Some(column)) if stamped != resolved_bucket_ms => {
            LodBucketCheck::Mismatch(format!(
                "LOD grid / temporal bucket mismatch: the input baked `{column}` for a \
                 {stamped} ms grid (Parquet footer key `{LOD_GRID_BUCKET_KEY}`), but this build \
                 buckets at {resolved_bucket_ms} ms (--temporal-bucket, after any --auto \
                 override). The baked LOD floors are wrong for this archive — every zoom below \
                 the floor would show a fraction of the features actually in the displayed \
                 bucket. Re-run the generation script at {resolved_bucket_ms} ms, or build with \
                 --temporal-bucket {stamped}ms."
            ))
        }
        (Some(_), Some(_)) => LodBucketCheck::Ok,
        (Some(stamped), None) => LodBucketCheck::Warn(format!(
            "input declares a baked LOD grid at {stamped} ms (`{LOD_GRID_BUCKET_KEY}`) but this \
             build passes no --min-zoom-field, so the baked floors are ignored and nothing is \
             asserted."
        )),
        (None, Some(column)) => LodBucketCheck::Warn(format!(
            "--min-zoom-field {column} consumes baked LOD floors, but the input declares no \
             `{LOD_GRID_BUCKET_KEY}` footer key — nothing can assert that its thinning grid was \
             keyed on this build's {resolved_bucket_ms} ms temporal bucket. Re-run the \
             generation script with the stamp to make this checkable."
        )),
        (None, None) => LodBucketCheck::Ok,
    }
}

/// Read [`LOD_GRID_BUCKET_KEY`] out of a Parquet file's key-value footer.
///
/// `None` when the file is not Parquet / cannot be opened / declares no such
/// key. A MALFORMED value (non-numeric, empty, zero) is also `None` — a
/// generation script writing garbage must not brick a build, and the resulting
/// "key absent" warning is the honest report (`Some(bad)` would fabricate an
/// assertion). Returns the parsed value plus, when the key was present but
/// unusable, a human-readable note for the caller to warn with.
pub fn lod_grid_bucket_ms(path: &Path) -> (Option<u64>, Option<String>) {
    let Ok(file) = std::fs::File::open(path) else {
        return (None, None);
    };
    let reader = match parquet::file::reader::SerializedFileReader::new(file) {
        Ok(r) => r,
        Err(_) => return (None, None),
    };
    use parquet::file::reader::FileReader as _;
    let Some(kv) = reader.metadata().file_metadata().key_value_metadata() else {
        return (None, None);
    };
    let Some(entry) = kv.iter().find(|e| e.key == LOD_GRID_BUCKET_KEY) else {
        return (None, None);
    };
    let raw = entry.value.as_deref().unwrap_or("").trim().to_string();
    match raw.parse::<u64>() {
        Ok(v) if v > 0 => (Some(v), None),
        _ => (
            None,
            Some(format!(
                "input's `{LOD_GRID_BUCKET_KEY}` footer key is not a positive integer \
                 (got {raw:?}) — treating it as absent."
            )),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agreeing_bucket_and_stamp_pass_silently() {
        assert_eq!(
            check_lod_grid_bucket(Some(300_000), Some("lod_min_zoom"), 300_000),
            LodBucketCheck::Ok
        );
    }

    #[test]
    fn a_disagreeing_pair_is_a_hard_error_naming_both_values_and_sources() {
        let LodBucketCheck::Mismatch(msg) =
            check_lod_grid_bucket(Some(300_000), Some("lod_min_zoom"), 3_600_000)
        else {
            panic!("expected a mismatch");
        };
        // Both values...
        assert!(msg.contains("300000"), "{msg}");
        assert!(msg.contains("3600000"), "{msg}");
        // ...and both sources.
        assert!(msg.contains(LOD_GRID_BUCKET_KEY), "{msg}");
        assert!(msg.contains("--temporal-bucket"), "{msg}");
        assert!(msg.contains("lod_min_zoom"), "{msg}");
    }

    #[test]
    fn missing_halves_warn_rather_than_fail() {
        // Legacy input: the flag is set but nothing declares a grid.
        assert!(matches!(
            check_lod_grid_bucket(None, Some("lod_min_zoom"), 300_000),
            LodBucketCheck::Warn(_)
        ));
        // Baked input, but this build ignores the floors.
        assert!(matches!(
            check_lod_grid_bucket(Some(300_000), None, 3_600_000),
            LodBucketCheck::Warn(_)
        ));
        // Neither: silence.
        assert_eq!(
            check_lod_grid_bucket(None, None, 3_600_000),
            LodBucketCheck::Ok
        );
    }

    /// Guard on the ORDERING the check depends on: the comparison is against the
    /// RESOLVED bucket, so an `--auto` override that moves the bucket away from
    /// the stamp must fail even though the user's command line agreed with it.
    /// (Pass the pre-override value and this test goes green while the real bug
    /// ships — which is exactly why it is pinned here.)
    #[test]
    fn the_check_is_against_the_resolved_bucket_not_the_flag_as_typed() {
        let stamped = 300_000u64;
        let user_typed = 300_000u64;
        let auto_resolved = 900_000u64;
        assert_eq!(
            check_lod_grid_bucket(Some(stamped), Some("lod_min_zoom"), user_typed),
            LodBucketCheck::Ok,
            "the value the user typed agrees — this is the trap"
        );
        assert!(
            matches!(
                check_lod_grid_bucket(Some(stamped), Some("lod_min_zoom"), auto_resolved),
                LodBucketCheck::Mismatch(_)
            ),
            "the RESOLVED bucket disagrees and must fail"
        );
    }

    #[test]
    fn absent_or_unreadable_file_yields_no_stamp_and_no_note() {
        let (v, note) = lod_grid_bucket_ms(Path::new("/nonexistent/nope.parquet"));
        assert_eq!(v, None);
        assert_eq!(note, None);
    }
}
