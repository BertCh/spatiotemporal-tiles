//! Check 12 — the **semantic content fingerprint**.
//!
//! Every other check in this binary is structural: content addressing, CRCs,
//! Arrow schemas, counts, temporal bounds. All of them passed on 106 AV
//! archives whose 3D `xyz` coordinate leaves had been re-read with a stride of
//! 2 — flattened and scrambled. Nothing in the archive was malformed; the
//! numbers were simply *wrong*. A human found it by comparing
//! `stt-optimize export` bboxes.
//!
//! This module makes that comparison executable. When
//! `metadata.content_fingerprint` is present, the decode loop folds every
//! decoded layer into a [`FingerprintAccumulator`] and the result is compared
//! against the declared block:
//!
//! * **sampled / partial decode → containment.** Every observed vertex and
//!   value must lie inside the declared bbox / ranges. This is what catches the
//!   recorded class even at `--sample 2`: the defect read metric `z` values as
//!   longitudes, and those land tens of degrees outside any real bbox.
//! * **full decode → containment plus equality** within the declared
//!   tolerances, so a manifest that describes content the archive does not
//!   contain is a failure too.
//!
//! When the fingerprint is **absent** — which is every archive published
//! before SH-1 — the run **warns and continues**, mirroring the CRS84
//! precedent (`schema.rs`: a writer MUST that is newer than the deployed fleet
//! is a warning, never an error).
//!
//! ## The relapse-proof rule
//!
//! A tool that transforms an archive losslessly (reorder, repack, re-optimize)
//! MUST carry `content_fingerprint` through **verbatim** and MUST NOT recompute
//! it from its own output. Recomputing is precisely how the 106-archive defect
//! would have self-certified. The two flags this module serves exist for that
//! workflow and no other:
//!
//! ```text
//! stt-validate before/  --emit-fingerprint   truth.json   # capture from the trusted source
//! stt-validate after/   --expect-fingerprint truth.json   # accept the transform against it
//! ```
//!
//! `--emit-fingerprint` refuses to run under `--sample` / `--skip-decode`: a
//! fingerprint captured from a subset understates the content, and using it as
//! an expectation later would be a check that cannot fail.
//!
//! Types live in `stt_core::metadata` rather than here so the accumulator is
//! reachable from `stt-core`'s property tests (a binary's private module is not
//! importable from an integration test); this module is the validator-facing
//! surface — the plan's `fingerprint::FingerprintAccumulator` /
//! `fingerprint::check_fingerprint` paths resolve through the re-exports below.
//!
//! # Check 13 — the DECLARED BOUNDS containment check (SH-2)
//!
//! Check 12 compares the decode against `metadata.content_fingerprint.bbox` —
//! a block only a fingerprinted archive carries. [`check_declared_bounds`]
//! compares the same decode against `metadata.bounds`, which **every** archive
//! carries and which nothing checked at all: an archive whose declared bounds
//! sits in the wrong hemisphere validated clean.
//!
//! The soundness direction is the whole point and it is NOT symmetric:
//!
//! * **Declared too SMALL is the failure.** Readers pre-intersect query boxes
//!   against `metadata.bounds` (tile selection, frustum pre-culling, the
//!   showcase's opening camera), so an under-stated box makes them discard
//!   tiles that really do carry visible data — silent data loss at query time,
//!   with no error anywhere in the stack.
//! * **Declared too LARGE is at most a warning.** A conservative superset keeps
//!   every reader sound; it merely wastes a fetch and opens the camera wide.

use anyhow::{Context, Result};
use std::path::Path;
use stt_core::metadata::Metadata;

#[allow(unused_imports)] // `FingerprintFindings` is the re-exported return type.
pub use stt_core::metadata::{
    check_fingerprint, check_fingerprint_with, distinct_id_basis, ContentFingerprint,
    DistinctIdBasis, FingerprintAccumulator, FingerprintFindings, ObservedFingerprint,
    CONTENT_FINGERPRINT_VERSION,
};

/// The antimeridian-seam predicate, and the noise floor it measures the seam
/// distance with, live in `stt-core` beside check 12's containment test —
/// **one** implementation, applied on identical terms by both bbox checks.
///
/// Check 12 compares the decode against `ContentFingerprint::bbox`; check 13
/// compares it against `metadata.bounds`. Both declared boxes are min/max folds
/// over the SOURCE vertices taken before tiling, so the ring splitter's
/// synthesised ±180° vertices make both of them under-state a seam-crossing
/// dataset in exactly the same way. When only check 13 knew that, an honest
/// dateline-crossing polygon archive got a corruption ERROR from check 12
/// sitting next to a WARNING from check 13 explaining that it was not
/// corruption.
use stt_core::metadata::{explained_by_antimeridian_seam, COORD_NOISE_FLOOR_DEG};

/// The layers a fingerprint run must NOT fold in.
///
/// Only the summary tier: its rows are derived aggregates addressed by
/// H3/Quadbin cell index, so a cell centroid can legitimately sit a cell-radius
/// outside the source bbox and its `count` column is an aggregate no source
/// feature carries. Folding them in would manufacture findings on every
/// summary archive. Every other layer — including the `_originals` companion
/// and the temporal-LOD tiers — carries real source geometry and IS checked.
pub fn skipped_layers(metadata: &Metadata) -> Vec<String> {
    metadata
        .summary_tier
        .as_ref()
        .map(|tier| vec![tier.layer_name.clone()])
        .unwrap_or_default()
}

/// The warning emitted for an archive that declares no fingerprint.
///
/// Deliberately a warning: every archive in the published fleet predates the
/// field, and turning the whole fleet red teaches operators to ignore the
/// validator. Same shape and same reasoning as the missing-CRS84 warning.
pub fn absent_fingerprint_warning() -> String {
    "metadata.content_fingerprint is absent, so the archive's decoded CONTENT was not \
     verified — only its structure. Structural validation passed on 106 archives whose \
     coordinates had been silently scrambled. Rebuild with `stt-build \
     --content-fingerprint` to make the content checkable"
        .to_string()
}

// ---------------------------------------------------------------------------
// Check 13 — declared `metadata.bounds` vs the decoded vertex bbox (SH-2).
// ---------------------------------------------------------------------------

/// The `metadata.properties` key by which a writer *attests* that
/// `metadata.bounds` is the honest, **vertex-derived** bbox
/// (`stt_build::input::BoundsMode::Vertex`) rather than the legacy centroid one.
///
/// `properties` is the free-form `BTreeMap<String, String>` every manifest
/// already carries, deliberately chosen over a new `manifest.capabilities`
/// entry: capabilities are a closed set and a reader **rejects an archive
/// declaring one it does not implement** (`PackedReader::open`), so minting a
/// capability here would make every attested archive unopenable by today's
/// deployed readers. A property is inert to every reader that does not look
/// for it.
pub const BOUNDS_MODE_PROPERTY: &str = "bounds_mode";

/// The [`BOUNDS_MODE_PROPERTY`] value that attests vertex-derived bounds.
pub const BOUNDS_MODE_VERTEX: &str = "vertex";

/// Floating-point floor under every check-13 coordinate comparison, in degrees.
///
/// Aliased to the core constant rather than restated, because check 12 applies
/// the identical floor when it asks the *same* question of the *same* seam —
/// two floors would mean two answers. See [`COORD_NOISE_FLOOR_DEG`] for why
/// 1e-9 deg (≈0.11 mm) is the right size: it is IEEE-754 and clip-interpolation
/// noise, not format slack.
const COORD_EPSILON_DEG: f64 = COORD_NOISE_FLOOR_DEG;

/// The same floor for the vertical axis, in the units altitudes are authored in
/// (metres for every current archive) — 1 µm.
const Z_EPSILON: f64 = 1e-6;

/// Does the archive *attest* that its `metadata.bounds` is vertex-derived?
///
/// This is the severity gate for check 13, and it exists because pre-R1
/// archives are not wrong so much as *old*: their bounds is the centroid bbox,
/// which legitimately under-states the extent of every non-point geometry. An
/// error on those would turn the entire published fleet red, which is how
/// operators learn to ignore a validator (same reasoning as the missing-CRS84
/// and absent-fingerprint warnings).
///
/// Two attestations, either of which is sufficient:
///
/// 1. **A content fingerprint.** `metadata.content_fingerprint` is emitted only
///    by a writer new enough to compute vertex extents, and its `bbox` is
///    itself a vertex-bbox containment claim. This is the gate the R1 rebuild
///    rides: R1 turns fingerprint emission on fleet-wide.
/// 2. **An explicit `bounds_mode = vertex` property.** The forward seam for a
///    build that flips [`BOUNDS_MODE_PROPERTY`] without emitting a fingerprint.
pub fn bounds_are_attested(metadata: &Metadata) -> bool {
    if metadata.content_fingerprint.is_some() {
        return true;
    }
    metadata
        .properties
        .get(BOUNDS_MODE_PROPERTY)
        .is_some_and(|mode| mode.eq_ignore_ascii_case(BOUNDS_MODE_VERTEX))
}

/// Record one check-13 finding at the severity [`bounds_are_attested`] chose.
///
/// `body` states the fact; the tail states why it is (or is not) fatal and,
/// on the warning path, names the rebuild as the fix — the same shape as
/// [`absent_fingerprint_warning`].
fn record(findings: &mut FingerprintFindings, attested: bool, body: String) {
    if attested {
        findings.errors.push(format!(
            "{body} The archive ATTESTS vertex-derived bounds (it carries a content \
             fingerprint, or a '{BOUNDS_MODE_PROPERTY}={BOUNDS_MODE_VERTEX}' property), so a \
             bbox that does not contain its own data is an error, not a legacy gap"
        ));
    } else {
        findings.warnings.push(format!(
            "{body} The archive does not attest vertex-derived bounds, so this is a WARNING: \
             its bounds is the legacy CENTROID bbox, which under-states every non-point \
             geometry by construction. Rebuild with a vertex-bounds writer (stt-build, \
             BoundsMode::Vertex) to make the claim honest and this check enforceable"
        ));
    }
}

/// Check 13: the declared `metadata.bounds` must **contain** the decoded vertex
/// bbox that check 12's accumulator already produced — no second decode pass.
///
/// # Severity
///
/// Under-statement (the observed bbox escapes the declared one) is an **error**
/// on an archive that [`bounds_are_attested`], a **warning** otherwise.
/// Over-statement is never worse than a warning, and is only reported on a full
/// decode (under `--sample` the decoded subset is *expected* to be narrower
/// than the whole dataset, so "wider than observed" would be noise).
///
/// # Why this cannot false-positive on an honest archive
///
/// * **Quantization.** Reconstruction can push a vertex up to one grid step
///   outside the source bbox, so the comparison carries the same slack check 12
///   uses: the `stt:quant` step the archive itself carries on the wire, widened
///   by any `coord_tolerance_deg` a fingerprint declared (check 12 already
///   capability-gates that value, so admitting it here buys no new slack).
/// * **Null island.** `metadata.bounds` deliberately EXCLUDES exactly-`(0,0)`
///   features (the writer's sentinel policy for coerced geometry) while the
///   fingerprint's observed bbox deliberately INCLUDES them. An escape that
///   vanishes once the declared box is expanded to the origin is therefore the
///   sentinel policy showing through, not corruption — reported as a warning
///   whatever the attestation.
/// * **Antimeridian.** A *wrapped* declared interval (`min_lon > max_lon`)
///   cannot be evaluated against an unwrapped observed bbox at all, so the
///   longitude axis is skipped with a warning rather than guessed at. And a
///   straddling dataset's loose unwrapped box still stops at its extreme
///   *source* vertex, while the builder's seam split synthesises vertices at
///   exactly ±180 — an escape that is entirely explained by the seam is a
///   warning naming that reason, never an error
///   ([`explained_by_antimeridian_seam`], which spells out the three conditions
///   that keep the relaxation off every other archive).
/// * **Poles.** Nothing is projected or clamped here; a vertex at ±90° simply
///   has to be inside the declared box.
pub fn check_declared_bounds(
    metadata: &Metadata,
    observed: &ObservedFingerprint,
    decode_complete: bool,
) -> FingerprintFindings {
    let mut findings = FingerprintFindings::default();
    let attested = bounds_are_attested(metadata);
    let d = metadata.bounds;

    // Nothing finite was decoded: there is no observation to contain. Check 12
    // already warns about that case; do not say it twice.
    let Some(o) = observed.bbox else {
        return findings;
    };

    if ![d.min_lon, d.min_lat, d.max_lon, d.max_lat]
        .iter()
        .all(|v| v.is_finite())
    {
        record(
            &mut findings,
            attested,
            format!(
                "declared metadata.bounds [{}, {}, {}, {}] is not finite, so it bounds nothing \
                 — every reader that pre-intersects a query box against it gets an \
                 unpredictable answer.",
                d.min_lon, d.min_lat, d.max_lon, d.max_lat
            ),
        );
        return findings;
    }

    // Slack: what the reader itself dequantizes with, widened by any declared
    // coordinate tolerance, floored at the ULP epsilon. `.max` also launders a
    // NaN step (f64::max returns the non-NaN side) so no tolerance can make the
    // test vacuous.
    let declared_tol = metadata
        .content_fingerprint
        .as_ref()
        .map(|fp| fp.coord_tolerance_deg)
        .filter(|t| t.is_finite() && *t >= 0.0)
        .unwrap_or(0.0);
    let tol = declared_tol
        .max(observed.coord_step_deg)
        .max(COORD_EPSILON_DEG);

    // A wrapped longitude interval is a claim about the union
    // [min_lon, 180] ∪ [-180, max_lon]; an observed min/max pair carries no
    // information about the gap between them, so containment on that axis is
    // not decidable from what a decode produces. Skip it loudly.
    let lon_wrapped = d.min_lon > d.max_lon;
    if lon_wrapped {
        findings.warnings.push(format!(
            "declared metadata.bounds longitude interval [{}, {}] is WRAPPED across the \
             antimeridian; containment on the longitude axis is not decidable from an unwrapped \
             decoded bbox, so only latitude was checked. stt-build never emits a wrapped box — a \
             ±180°-straddling dataset gets the loose full-width interval instead, which is looser \
             but checkable",
            d.min_lon, d.max_lon
        ));
    }

    // Per EDGE, not per axis: the null-island test below has to know *which*
    // edge escaped, because the sentinel can only ever pull an edge to exactly
    // zero.
    let lon_lo = !lon_wrapped && o[0] < d.min_lon - tol;
    let lon_hi = !lon_wrapped && o[2] > d.max_lon + tol;
    let lat_lo = o[1] < d.min_lat - tol;
    let lat_hi = o[3] > d.max_lat + tol;

    if lon_lo || lon_hi || lat_lo || lat_hi {
        // Is every escaping edge sitting at exactly (0, 0)? Then it is the
        // writer's null-island sentinel policy (`metadata.bounds` excludes
        // exactly-(0,0) features; the fingerprint's bbox includes them), not a
        // bbox that lost its data.
        //
        // Tested edge-wise and against EXACTLY 0.0 rather than "the escape
        // points origin-ward": a whole hemisphere lies between a dataset in the
        // Americas and the prime meridian, and excusing every escape in that
        // direction would excuse most real corruption too.
        let explained_by_sentinel = (!lon_lo || o[0] == 0.0)
            && (!lon_hi || o[2] == 0.0)
            && (!lat_lo || o[1] == 0.0)
            && (!lat_hi || o[3] == 0.0);
        // …or by the builder's ANTIMERIDIAN SPLIT, which synthesises seam
        // vertices at exactly ±180 that the source-vertex fold behind
        // `metadata.bounds` never saw. Deliberate writer behaviour, so it may
        // not be fatal — but it is still said out loud, and only under the
        // three conditions the helper documents.
        let explained_by_seam = explained_by_antimeridian_seam(
            d.min_lon, d.max_lon, o, tol, lon_lo, lon_hi, lat_lo, lat_hi,
        );
        if explained_by_sentinel {
            findings.warnings.push(format!(
                "declared metadata.bounds [{}, {}, {}, {}] does not contain the decoded vertex \
                 bbox [{}, {}, {}, {}], but the entire escape is toward (0, 0) — the writer's \
                 null-island sentinel policy excludes exactly-(0,0) features from bounds while \
                 the decode counts their vertices. Reported, not failed",
                d.min_lon, d.min_lat, d.max_lon, d.max_lat, o[0], o[1], o[2], o[3]
            ));
        } else if explained_by_seam {
            findings.warnings.push(format!(
                "declared metadata.bounds [{}, {}, {}, {}] does not contain the decoded vertex \
                 bbox [{}, {}, {}, {}], but the entire escape is to the ±180° ANTIMERIDIAN SEAM \
                 on a dataset whose declared longitude interval already spans {:.3}°. stt-build \
                 SPLITS a seam-crossing ring and synthesises vertices at exactly ±180 that the \
                 source geometry never carried, while metadata.bounds is a plain min/max fold \
                 over those source vertices — so the escape is the split showing through, not a \
                 bbox that lost its data. Reported, not failed; latitude was still enforced. Fix \
                 it at the writer by widening the declared longitude interval to the full \
                 [-180, 180] when the source straddles ±180",
                d.min_lon,
                d.min_lat,
                d.max_lon,
                d.max_lat,
                o[0],
                o[1],
                o[2],
                o[3],
                d.max_lon - d.min_lon
            ));
        } else {
            record(
                &mut findings,
                attested,
                format!(
                    "declared metadata.bounds [{}, {}, {}, {}] does NOT contain the decoded \
                     vertex bbox [{}, {}, {}, {}] (tolerance {tol} deg). An under-stated bbox is \
                     silent data loss at QUERY time: tile selection, frustum pre-culling and the \
                     opening camera all pre-intersect against this value, so they discard tiles \
                     that really do carry visible data — with no error anywhere in the stack.",
                    d.min_lon, d.min_lat, d.max_lon, d.max_lat, o[0], o[1], o[2], o[3]
                ),
            );
        }
    } else if decode_complete && attested {
        // Over-statement: sound but dishonest. Only meaningful on a full decode
        // (a sample is *expected* to be narrower), and only worth saying on an
        // archive that claims a tight vertex bbox — the whole-world placeholder
        // an unattested archive carries is not a claim about extent at all.
        let lon_short =
            !lon_wrapped && ((o[0] - d.min_lon).abs() > tol || (o[2] - d.max_lon).abs() > tol);
        let lat_short = (o[1] - d.min_lat).abs() > tol || (o[3] - d.max_lat).abs() > tol;
        if lon_short || lat_short {
            findings.warnings.push(format!(
                "full decode: declared metadata.bounds [{}, {}, {}, {}] is WIDER than the decoded \
                 vertex bbox [{}, {}, {}, {}]. Conservative, so every reader stays sound — but \
                 the manifest describes extent the archive does not fill (a zoom clamp, a \
                 dropped tier or --min-features-per-tile does this), and an opening camera \
                 framed on it will open zoomed out",
                d.min_lon, d.min_lat, d.max_lon, d.max_lat, o[0], o[1], o[2], o[3]
            ));
        }
    }

    // Vertical extent: the same containment direction, against the manifest's
    // own `z_range` (check 12 compares the fingerprint's copy, a different
    // declaration). Only the containment arm — an over-stated ceiling costs a
    // reader nothing.
    if let (Some([dlo, dhi]), Some([olo, ohi])) = (metadata.z_range, observed.z_range) {
        let z_tol = observed.z_step.max(Z_EPSILON);
        if dlo.is_finite() && dhi.is_finite() && (olo < dlo - z_tol || ohi > dhi + z_tol) {
            record(
                &mut findings,
                attested,
                format!(
                    "declared metadata.z_range [{dlo}, {dhi}] does NOT contain the decoded \
                     vertical extent [{olo}, {ohi}] (tolerance {z_tol}). Altitude-aware selection \
                     pre-intersects against this the same way tile selection does against bounds."
                ),
            );
        }
    }

    findings
}

/// Turn an [`ObservedFingerprint`] into a declarable [`ContentFingerprint`].
///
/// Used by `--emit-fingerprint` to capture ground truth from a TRUSTED archive
/// so a later transform can be accepted against it.
///
/// # ⚠️ The coordinate tolerance is OBSERVED, never copied from the source
///
/// It reads as prudent to carry the source archive's declared
/// `coord_tolerance_deg` forward, so a captured fingerprint cannot drop slack a
/// quantized archive needs. That is precisely how an inflated declaration
/// PROPAGATES: a manifest claiming `coord_tolerance_deg: 90.0` on a 1 m grid
/// would be copied verbatim into the baseline file, and every later
/// `--expect-fingerprint` run against that file would be vacuous — with the
/// vacuity now laundered through a document that looks like measured ground
/// truth.
///
/// So the emitted value is [`ObservedFingerprint::coord_step_deg`]: the exact
/// affine read off the wire, which is what the reader dequantizes with and what
/// [`check_fingerprint`] admits as slack whatever the writer declared. Nothing
/// legitimate is lost — an honest source declares that same step, and
/// `check_fingerprint` now rejects any declaration materially above it
/// ([`stt_core::metadata::COORD_TOLERANCE_STEP_FACTOR`]), so there is no honest
/// declaration left for this to fail to carry.
///
/// The per-column tolerances ARE still merged from the source block, and
/// deliberately: unlike coordinate quantization, `build_quantized_numeric` may
/// DECLINE to quantize a column (it then ships exact `Float64` with no `stt:qa`
/// affine), so an absent on-wire step is not evidence that a declared column
/// precision was unearned. No magnitude bound is applied to them for the same
/// reason.
pub fn observed_as_declared(
    observed: &ObservedFingerprint,
    declared: Option<&ContentFingerprint>,
) -> ContentFingerprint {
    let bbox = observed.bbox.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
    ContentFingerprint {
        version: CONTENT_FINGERPRINT_VERSION,
        bbox,
        z_range: observed.z_range,
        // The source archive's own declared count beats any recomputation; with
        // none, the EXACT distinct-id count when the decode stayed under the
        // exact-count cap, and only then the sketch estimate. A captured
        // expectation is compared with a SHORTFALL error later, so it must not
        // be an estimate when an exact count was available.
        distinct_feature_count: declared
            .map(|d| d.distinct_feature_count)
            .unwrap_or_else(|| observed.distinct_ids()),
        numeric_ranges: observed.numeric_ranges.clone(),
        categorical_cardinality: observed.categorical_cardinality.clone(),
        // OBSERVED ONLY — see the doc comment. Copying the declaration here is
        // how an inflated tolerance escapes the manifest it was caught in and
        // becomes a captured baseline nobody re-audits.
        coord_tolerance_deg: observed.coord_step_deg,
        column_tolerance: {
            let mut out = declared
                .map(|d| d.column_tolerance.clone())
                .unwrap_or_default();
            for (column, step) in &observed.column_steps {
                let slot = out.entry(column.clone()).or_insert(0.0);
                *slot = slot.max(*step);
            }
            out.retain(|_, v| *v > 0.0);
            out
        },
    }
}

/// Read an external fingerprint file for `--expect-fingerprint`.
pub fn read_fingerprint_file(path: &Path) -> Result<ContentFingerprint> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read fingerprint file {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "{} is not a content-fingerprint JSON document",
            path.display()
        )
    })
}

/// Write a fingerprint for `--emit-fingerprint`: pretty JSON plus a trailing
/// newline, so the file diffs cleanly and can be committed as an acceptance
/// baseline.
pub fn write_fingerprint_file(path: &Path, fingerprint: &ContentFingerprint) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(fingerprint)
        .with_context(|| "failed to serialize the observed fingerprint")?;
    bytes.push(b'\n');
    std::fs::write(path, bytes)
        .with_context(|| format!("failed to write fingerprint file {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use stt_core::metadata::{SummaryAggregation, SummaryColumn, SummaryScheme, SummaryTier};
    use stt_core::types::BoundingBox;

    fn declared() -> ContentFingerprint {
        ContentFingerprint {
            version: CONTENT_FINGERPRINT_VERSION,
            bbox: [-1.0, -1.0, 1.0, 1.0],
            z_range: Some([0.0, 10.0]),
            distinct_feature_count: 4,
            numeric_ranges: BTreeMap::from([("speed".to_string(), [0.0, 30.0])]),
            categorical_cardinality: BTreeMap::from([("kind".to_string(), 2)]),
            coord_tolerance_deg: 0.0,
            column_tolerance: BTreeMap::new(),
        }
    }

    /// The summary tier — and only the summary tier — is excluded.
    #[test]
    fn skipped_layers_names_only_the_summary_tier() {
        let plain = Metadata::new("plain");
        assert!(skipped_layers(&plain).is_empty());

        let tiered = Metadata::new("tiered").with_summary_tier(SummaryTier {
            variant_id: stt_core::tile::SUMMARY_VARIANT_ID,
            scheme: SummaryScheme::H3,
            min_zoom: 0,
            max_zoom: 4,
            cell_resolution_per_zoom: vec![0, 1, 2, 3, 4],
            columns: vec![SummaryColumn {
                name: "count".into(),
                agg: SummaryAggregation::Count,
            }],
            layer_name: "cells".into(),
            sub_buckets: 1,
        });
        assert_eq!(skipped_layers(&tiered), vec!["cells".to_string()]);
    }

    /// A fingerprint file round-trips through the emit/expect pair byte-for-byte
    /// — the acceptance workflow depends on it.
    #[test]
    fn fingerprint_file_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truth.json");
        let fp = declared();
        write_fingerprint_file(&path, &fp).unwrap();
        assert_eq!(read_fingerprint_file(&path).unwrap(), fp);
        // Byte-stable: writing the parsed value back reproduces the file.
        let first = std::fs::read(&path).unwrap();
        write_fingerprint_file(&path, &read_fingerprint_file(&path).unwrap()).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), first);
    }

    /// `--emit-fingerprint` must not silently drop the slack a quantized
    /// source archive declared, or the captured file would reject its own
    /// lossless transforms.
    #[test]
    fn observed_as_declared_carries_tolerances_forward() {
        let mut observed = ObservedFingerprint {
            bbox: Some([-1.0, -1.0, 1.0, 1.0]),
            ..Default::default()
        };
        observed.column_steps.insert("speed".to_string(), 0.25);
        observed.coord_step_deg = 1e-5;

        let source = declared();
        let captured = observed_as_declared(&observed, Some(&source));
        assert_eq!(captured.coord_tolerance_deg, 1e-5);
        assert_eq!(captured.column_tolerance.get("speed"), Some(&0.25));
        assert_eq!(
            captured.distinct_feature_count, 4,
            "the exact declared count beats the sketch estimate when both exist"
        );
    }

    /// ⭐ An INFLATED source tolerance must NOT reach the captured baseline.
    ///
    /// `check_fingerprint` rejects `coord_tolerance_deg: 90.0` on a 1 m grid,
    /// but rejection alone is not enough: `--emit-fingerprint` used to copy the
    /// declaration into the file it wrote, so the vacuity escaped the manifest
    /// it was caught in and became a baseline that every later
    /// `--expect-fingerprint` run would trust. The capture must report the WIRE.
    #[test]
    fn observed_as_declared_never_propagates_an_inflated_tolerance() {
        let step = 1.0 / 111_320.0;
        let observed = ObservedFingerprint {
            bbox: Some([-1.0, -1.0, 1.0, 1.0]),
            coord_step_deg: step,
            ..Default::default()
        };
        let mut inflated = declared();
        inflated.coord_tolerance_deg = 90.0;

        let captured = observed_as_declared(&observed, Some(&inflated));
        assert_eq!(
            captured.coord_tolerance_deg, step,
            "the capture must declare the archive's own on-wire step, not the \
             90 deg claim it was handed"
        );

        // And the captured file is itself accepted by the checker it feeds —
        // an emitted baseline that its own validator would reject would be a
        // trap for the next operator.
        let findings = check_fingerprint(
            &captured,
            &observed,
            true,
            &[stt_core::pack::CAPABILITY_COORD_QUANT.to_string()],
        );
        assert!(
            findings.errors.is_empty(),
            "an emitted baseline must satisfy check_fingerprint: {:?}",
            findings.errors
        );
    }

    // -----------------------------------------------------------------------
    // Check 13 — declared bounds containment.
    // -----------------------------------------------------------------------

    /// An archive over a small patch of San Francisco: the quantity every
    /// check-13 case below varies is what it *declares*.
    fn observed_sf() -> ObservedFingerprint {
        ObservedFingerprint {
            bbox: Some([-122.4050, 37.7900, -122.4030, 37.7940]),
            rows: 6,
            ..Default::default()
        }
    }

    fn meta_with_bounds(bounds: BoundingBox) -> Metadata {
        Metadata::new("bounds").with_bounds(bounds)
    }

    /// ⭐ THE PIN. Bounds in the WRONG HEMISPHERE — the reviewer's scenario —
    /// is an error once the archive attests vertex-derived bounds, and a
    /// warning (naming the rebuild) before it does. Nothing about the archive
    /// is malformed; only the claim is.
    #[test]
    fn wrong_hemisphere_bounds_error_when_attested_and_warn_when_not() {
        let wrong = BoundingBox::new(100.0, 37.7900, 110.0, 37.7940);

        // Unattested (today's fleet): warning only, and it names the fix.
        let legacy = meta_with_bounds(wrong);
        let findings = check_declared_bounds(&legacy, &observed_sf(), true);
        assert!(findings.errors.is_empty(), "{findings:?}");
        assert_eq!(findings.warnings.len(), 1, "{findings:?}");
        assert!(
            findings.warnings[0].contains("BoundsMode::Vertex"),
            "the legacy warning must name the rebuild: {findings:?}"
        );

        // Attested via a content fingerprint (what R1 turns on fleet-wide).
        let mut attested = meta_with_bounds(wrong);
        attested.content_fingerprint = Some(declared());
        let findings = check_declared_bounds(&attested, &observed_sf(), true);
        assert_eq!(findings.errors.len(), 1, "{findings:?}");
        assert!(
            findings.errors[0].contains("does NOT contain"),
            "{findings:?}"
        );

        // Attested via the explicit property, with no fingerprint at all.
        let mut stamped = meta_with_bounds(wrong);
        stamped.properties.insert(
            BOUNDS_MODE_PROPERTY.to_string(),
            BOUNDS_MODE_VERTEX.to_string(),
        );
        assert_eq!(
            check_declared_bounds(&stamped, &observed_sf(), true)
                .errors
                .len(),
            1,
            "the bounds_mode property alone must be enough to enforce"
        );
    }

    /// The control: an honest, exactly-tight declaration is clean under both
    /// severities and under both decode modes. Without this every case above
    /// would also pass if the check simply rejected everything.
    #[test]
    fn exact_bounds_are_clean_under_full_and_sampled_decode() {
        let tight = BoundingBox::new(-122.4050, 37.7900, -122.4030, 37.7940);
        let mut attested = meta_with_bounds(tight);
        attested.content_fingerprint = Some(declared());
        for complete in [true, false] {
            let findings = check_declared_bounds(&attested, &observed_sf(), complete);
            assert_eq!(findings, FingerprintFindings::default(), "{findings:?}");
        }
    }

    /// Direction is the whole point: a declared box WIDER than the data is
    /// conservative (readers stay sound), so it may only warn — and only on a
    /// full decode, because a sampled decode is expected to be narrower.
    #[test]
    fn over_declared_bounds_only_ever_warn_and_never_under_sampling() {
        let wide = BoundingBox::new(-180.0, -85.0, 180.0, 85.0);
        let mut attested = meta_with_bounds(wide);
        attested.content_fingerprint = Some(declared());

        let full = check_declared_bounds(&attested, &observed_sf(), true);
        assert!(full.errors.is_empty(), "too-large is never fatal: {full:?}");
        assert_eq!(full.warnings.len(), 1, "{full:?}");
        assert!(full.warnings[0].contains("WIDER"), "{full:?}");

        let sampled = check_declared_bounds(&attested, &observed_sf(), false);
        assert_eq!(
            sampled,
            FingerprintFindings::default(),
            "a sampled decode is expected to be narrower than the dataset: {sampled:?}"
        );
    }

    /// ⚠️ Antimeridian is a recorded hazard in this repo. A ±180°-straddling
    /// dataset gets the writer's loose full-width box, which contains
    /// everything and must pass silently; a WRAPPED declaration is undecidable
    /// on the longitude axis and warns rather than failing spuriously — while
    /// still checking latitude.
    #[test]
    fn antimeridian_cases_never_false_positive() {
        let straddling = ObservedFingerprint {
            bbox: Some([-179.9, -10.0, 179.9, 10.0]),
            rows: 4,
            ..Default::default()
        };

        // Loose full-width, exactly what profile_features_with emits.
        let mut full_width = meta_with_bounds(BoundingBox::new(-180.0, -10.0, 180.0, 10.0));
        full_width.content_fingerprint = Some(declared());
        let findings = check_declared_bounds(&full_width, &straddling, true);
        assert!(
            findings.errors.is_empty(),
            "a full-width box contains every vertex: {findings:?}"
        );

        // A wrapped interval from some other writer: warn, never fail.
        let mut wrapped = meta_with_bounds(BoundingBox::new(170.0, -10.0, -170.0, 10.0));
        wrapped.content_fingerprint = Some(declared());
        let findings = check_declared_bounds(&wrapped, &straddling, true);
        assert!(
            findings.errors.is_empty(),
            "a wrapped longitude interval must not manufacture a failure: {findings:?}"
        );
        assert!(
            findings.warnings.iter().any(|w| w.contains("WRAPPED")),
            "{findings:?}"
        );

        // ...but latitude is still enforced through a wrapped longitude box.
        let polar = ObservedFingerprint {
            bbox: Some([-179.9, -80.0, 179.9, 10.0]),
            rows: 4,
            ..Default::default()
        };
        assert_eq!(
            check_declared_bounds(&wrapped, &polar, true).errors.len(),
            1,
            "wrapping longitude must not disable the latitude axis"
        );
    }

    /// ⭐ THE SEAM CASE. The builder SPLITS a ±180°-crossing ring and
    /// synthesises vertices at exactly ±180 that the source-vertex fold behind
    /// `metadata.bounds` never saw
    /// (`seam_split_synthesises_pm180_vertices_the_declared_source_bbox_cannot_contain`,
    /// `crates/stt-build/tests/antimeridian_polygon.rs`, measures exactly this
    /// on the real tiler). Deliberate writer behaviour must not turn every
    /// seam-crossing non-point archive red — but it must not go silent either.
    #[test]
    fn antimeridian_seam_split_escape_warns_rather_than_failing() {
        // Exactly the numbers the builder produces for a 178°E→178°W polygon.
        let declared_box = BoundingBox::new(-178.0, -5.0, 178.0, 5.0);
        let split = ObservedFingerprint {
            bbox: Some([-180.0, -5.0, 180.0, 5.0]),
            rows: 8,
            ..Default::default()
        };
        let mut attested = meta_with_bounds(declared_box);
        attested.content_fingerprint = Some(declared());

        let findings = check_declared_bounds(&attested, &split, true);
        assert!(
            findings.errors.is_empty(),
            "the split is correct writer behaviour: {findings:?}"
        );
        let seam = findings
            .warnings
            .iter()
            .find(|w| w.contains("ANTIMERIDIAN SEAM"))
            .unwrap_or_else(|| {
                panic!("the seam must be NAMED, not silently excused: {findings:?}")
            });
        assert!(
            seam.contains("SPLITS") && seam.contains("widening the declared longitude interval"),
            "the warning must state the reason and the writer-side fix: {seam}"
        );

        // The property attestation reaches the same verdict — the exemption is
        // a property of the geometry, not of which attestation was used.
        let mut stamped = meta_with_bounds(declared_box);
        stamped.properties.insert(
            BOUNDS_MODE_PROPERTY.to_string(),
            BOUNDS_MODE_VERTEX.to_string(),
        );
        assert!(check_declared_bounds(&stamped, &split, true)
            .errors
            .is_empty());
    }

    /// ⚠️ The scoping, which is the whole risk of the case above: three
    /// conditions, and dropping ANY of them re-arms check 13 as an error.
    #[test]
    fn the_seam_exemption_is_scoped_to_the_seam_and_nothing_else() {
        let mut attested = meta_with_bounds(BoundingBox::new(-178.0, -5.0, 178.0, 5.0));
        attested.content_fingerprint = Some(declared());

        // (1) A latitude escape rides along ⇒ no exemption. A meridian cannot
        // move latitude, so this is not the split.
        let also_latitude = ObservedFingerprint {
            bbox: Some([-180.0, -40.0, 180.0, 5.0]),
            rows: 8,
            ..Default::default()
        };
        assert_eq!(
            check_declared_bounds(&attested, &also_latitude, true)
                .errors
                .len(),
            1,
            "latitude keeps full severity through the seam exemption"
        );

        // (2) The escape stops short of the seam ⇒ not the splitter's
        // signature, so it stays an error.
        let short_of_seam = ObservedFingerprint {
            bbox: Some([-179.0, -5.0, 179.0, 5.0]),
            rows: 8,
            ..Default::default()
        };
        assert_eq!(
            check_declared_bounds(&attested, &short_of_seam, true)
                .errors
                .len(),
            1,
            "only an escape that lands ON ±180 is the split"
        );

        // …and one that overshoots PAST the seam is out-of-world data, never
        // excused (the same narrowness as the null-island rule).
        let past_seam = ObservedFingerprint {
            bbox: Some([-181.0, -5.0, 181.0, 5.0]),
            rows: 8,
            ..Default::default()
        };
        assert_eq!(
            check_declared_bounds(&attested, &past_seam, true)
                .errors
                .len(),
            1,
            "coordinates past ±180 are not a seam artifact"
        );

        // (3) ⭐ The recorded regression class must still fail: a COMPACT
        // declared box whose decode smeared to the world edges is the
        // scrambled-coordinate shape, not a straddling dataset.
        let mut compact =
            meta_with_bounds(BoundingBox::new(-122.4050, 37.7900, -122.4030, 37.7940));
        compact.content_fingerprint = Some(declared());
        let smeared = ObservedFingerprint {
            bbox: Some([-180.0, 37.7900, 180.0, 37.7940]),
            rows: 8,
            ..Default::default()
        };
        assert_eq!(
            check_declared_bounds(&compact, &smeared, true).errors.len(),
            1,
            "a narrow declared box that decodes to the world edges is corruption, not a seam"
        );

        // …and an unattested archive is untouched by all of this: still a
        // warning, still naming the rebuild.
        let legacy = meta_with_bounds(BoundingBox::new(-178.0, -5.0, 178.0, 5.0));
        let split = ObservedFingerprint {
            bbox: Some([-180.0, -5.0, 180.0, 5.0]),
            rows: 8,
            ..Default::default()
        };
        let findings = check_declared_bounds(&legacy, &split, true);
        assert!(findings.errors.is_empty(), "{findings:?}");
    }

    /// `metadata.bounds` excludes exactly-(0,0) sentinel features; the
    /// fingerprint's observed bbox deliberately includes them. An escape that
    /// is entirely explained by the origin is that policy showing through, so
    /// it must not turn an honest archive red.
    #[test]
    fn null_island_escape_warns_rather_than_failing() {
        let with_sentinel = ObservedFingerprint {
            bbox: Some([-122.4050, 0.0, 0.0, 37.7940]),
            rows: 7,
            ..Default::default()
        };
        let mut attested =
            meta_with_bounds(BoundingBox::new(-122.4050, 37.7900, -122.4030, 37.7940));
        attested.content_fingerprint = Some(declared());

        let findings = check_declared_bounds(&attested, &with_sentinel, true);
        assert!(findings.errors.is_empty(), "{findings:?}");
        assert!(
            findings.warnings.iter().any(|w| w.contains("null-island")),
            "{findings:?}"
        );

        // The exemption is narrow: an escape that overshoots PAST the origin
        // is not the sentinel and still fails.
        let past_origin = ObservedFingerprint {
            bbox: Some([-122.4050, 37.7900, 10.0, 37.7940]),
            rows: 7,
            ..Default::default()
        };
        assert_eq!(
            check_declared_bounds(&attested, &past_origin, true)
                .errors
                .len(),
            1,
            "only escapes toward (0,0) are excused"
        );
    }

    /// Quantization moves reconstructed vertices by up to a grid step, so the
    /// wire's own `stt:quant` step is admitted as slack — otherwise every
    /// quantized archive would fail its own honest bounds.
    #[test]
    fn quantization_step_is_admitted_as_slack() {
        let tight = BoundingBox::new(-122.4050, 37.7900, -122.4030, 37.7940);
        let mut attested = meta_with_bounds(tight);
        attested.content_fingerprint = Some(declared());

        // One step outside on every edge, with the step declared on the wire.
        let nudged = ObservedFingerprint {
            bbox: Some([-122.4051, 37.7899, -122.4029, 37.7941]),
            coord_step_deg: 1e-4,
            rows: 6,
            ..Default::default()
        };
        let findings = check_declared_bounds(&attested, &nudged, true);
        assert!(findings.errors.is_empty(), "{findings:?}");

        // Without the step on the wire the same decode is a real escape.
        let unquantized = ObservedFingerprint {
            coord_step_deg: 0.0,
            ..nudged
        };
        assert_eq!(
            check_declared_bounds(&attested, &unquantized, true)
                .errors
                .len(),
            1
        );
    }

    /// Nothing decoded ⇒ nothing to contain, and no finding: check 12 already
    /// reports the empty decode, and a second voice would only teach operators
    /// to skim.
    #[test]
    fn no_observed_bbox_produces_no_finding() {
        let mut attested = meta_with_bounds(BoundingBox::new(100.0, 10.0, 110.0, 20.0));
        attested.content_fingerprint = Some(declared());
        let findings = check_declared_bounds(&attested, &ObservedFingerprint::default(), true);
        assert_eq!(findings, FingerprintFindings::default(), "{findings:?}");
    }

    /// The vertical axis gets the same containment direction, against the
    /// manifest's own `z_range`.
    #[test]
    fn declared_z_range_must_contain_the_decoded_extent() {
        let tight = BoundingBox::new(-122.4050, 37.7900, -122.4030, 37.7940);
        let mut attested = meta_with_bounds(tight).with_z_range(Some([0.0, 10.0]));
        attested.content_fingerprint = Some(declared());

        let deep = ObservedFingerprint {
            z_range: Some([0.0, 30.0]),
            ..observed_sf()
        };
        let findings = check_declared_bounds(&attested, &deep, true);
        assert_eq!(findings.errors.len(), 1, "{findings:?}");
        assert!(findings.errors[0].contains("z_range"), "{findings:?}");

        // Contained: clean. An over-stated ceiling costs a reader nothing.
        let shallow = ObservedFingerprint {
            z_range: Some([1.0, 9.0]),
            ..observed_sf()
        };
        assert!(check_declared_bounds(&attested, &shallow, true)
            .errors
            .is_empty());
    }

    /// A malformed fingerprint file is an error with the path in it, never a
    /// panic or a silent skip.
    #[test]
    fn malformed_fingerprint_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.json");
        std::fs::write(&path, b"{ not json").unwrap();
        let err = read_fingerprint_file(&path).unwrap_err().to_string();
        assert!(err.contains("junk.json"), "got: {err}");
    }
}
