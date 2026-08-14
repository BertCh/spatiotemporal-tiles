//! stt-validate — inspect and verify an STT dataset.
//!
//! Accepts the **packed format** only — a dataset directory, its
//! `manifest.json`, or a single-file `.sttb` **bundle** of one (spec §13;
//! the integrity tier then verifies each in-bundle object's blake3 against
//! its key exactly like the exploded case) — and verifies the
//! content-addressing contract. The single-file `.stt` container is an
//! internal streaming/transcode intermediate (spec D3), never a deployment
//! artifact, so it is not accepted here.
//!
//! Checks performed:
//! 1. (packed) Every pack and the directory object blake3-hash to the name the
//!    manifest gave them, on-disk lengths match, and the directory references
//!    no out-of-range `pack_id`.
//! 2. Index decodes; every entry has the columns the schema promises.
//! 3. Every tile blob round-trips its content hash and decompresses to its
//!    declared uncompressed size (enforced by `read_payload`).
//! 4. Every payload decodes as a layer frame of Arrow IPC streams.
//! 5. Every decoded layer matches the STT tile schema contract (required
//!    columns present with the expected Arrow types and a GeoArrow geometry
//!    column), and tiles agree on their schema (no producer drift).
//! 6. Feature counts in tile entries match the decoded layer rows.
//! 7. Tile temporal extents lie inside the dataset's metadata time range.
//! 8. Every feature's interval is sane (`end_time >= start_time`); violations
//!    are counted and the first offender named.
//! 9. Every decoded entry's directory `time_end` is TIGHT: it equals the max
//!    feature `end_time` across the tile's layers (readers prune interval
//!    queries on it — a nominal bucket end silently hides interval features).
//! 10. When the metadata declares a summary tier, every summary-layer `id` is
//!     a valid H3/Quadbin cell index at the tier's resolution for the tile's
//!     zoom (a sequential id column — the blank-render bug — fails).
//! 11. Metadata totals (`tile_count`/`feature_count`) match the
//!     directory-derived totals; zero totals from pre-0.1.1 writers warn
//!     instead (rebuilding with current stt-build repopulates them).
//! 12. SEMANTIC CONTENT FINGERPRINT. When `metadata.content_fingerprint` is
//!     present, the decoded content's vertex bbox, z range, per-column numeric
//!     ranges and categorical cardinalities are recomputed and compared —
//!     *containment* under `--sample` (every observed value must lie inside the
//!     declared extent), *containment plus equality* within the declared
//!     tolerances under a full decode. Declared tolerances are rejected unless
//!     their matching capability (`coord-quant`, `attr-quant`) is declared, so a
//!     writer cannot buy slack it did not earn.
//!
//!     FEATURE LOSS is caught by the **decoded-row floor**: on a full decode
//!     the rows at the FULLEST SINGLE ZOOM must be at least the declared
//!     `distinct_feature_count`, because every source feature appears at least
//!     once at the deepest tier that survives it and clipping only ever adds
//!     rows. Per-zoom and not per-archive on purpose — a 600-feature, 7-level
//!     pyramid decodes 4 200 rows in total, so a whole-archive floor would need
//!     >85 % loss to fire. Under `--sample` the same shortfall is the sampling
//!     warning.
//!
//!     The declared count is compared against the DECODED ID COUNT only on an
//!     archive whose wire `id` really is a dataset-wide key, and that is decided
//!     by the WRITER's recorded id construction
//!     (`metadata.properties.feature_id_construction`) — never by inferring it
//!     from the numbers agreeing:
//!
//!       * `source` / `anchor-hash` → the two counts are the same quantity. An
//!         id-less LINE or POLYGON gets the whole-feature anchor hash, which the
//!         tiler copies into every clipped piece, so this is the ordinary state
//!         of a line/polygon archive and it is CHECKED. That matters because the
//!         row floor is loose exactly there: a surviving line replicated across
//!         tiles contributes rows that cover for the features a rebuild dropped.
//!       * `row-index` (an id-less point, whose id is the per-tile row index, so
//!         a healthy 600-feature build decodes 5 distinct ids) and
//!         `segment-hash` (an id-less clipped trajectory, which mints a fresh id
//!         per segment) → different quantities, and the deviation is a NOTE.
//!
//!     `feature_id_scope=global` and the `global-feature-ids` capability still
//!     arm it explicitly, and `feature_id_scope=local` still disarms it (the
//!     documented rollback). On an armed archive a SHORTFALL over a full decode
//!     is an ERROR, a shortfall under `--sample` and any overshoot are warnings,
//!     and each is worded so the three can never be confused.
//!
//!     When the fingerprint is ABSENT — every archive published before it
//!     existed — the run warns and continues, exactly like the missing-CRS84
//!     case. See `fingerprint.rs`; this is the check that would have caught the
//!     coordinate scramble that structural validation passed on 106 archives.
//! 13. DECLARED BOUNDS CONTAINMENT. `metadata.bounds` must CONTAIN the decoded
//!     vertex bbox (and `metadata.z_range`, when declared, the decoded vertical
//!     extent). The direction is asymmetric on purpose: a declared box that is
//!     too SMALL is silent data loss at query time — tile selection, frustum
//!     pre-culling and the opening camera pre-intersect query boxes against it,
//!     so they discard tiles that really do carry visible data — while a box
//!     that is too LARGE keeps every reader sound and is at most a warning.
//!     Severity mirrors check 12: an ERROR on an archive that ATTESTS
//!     vertex-derived bounds (it carries a content fingerprint, or a
//!     `bounds_mode=vertex` property), a WARNING naming the rebuild on every
//!     legacy archive, whose centroid-derived bbox under-states non-point
//!     geometry by construction. Rides check 12's accumulator — no extra decode.
//!
//! The integrity, header, content-address, per-entry temporal-bound and
//! metadata-total checks (1–3, 7, 11) are cheap and always run over **every**
//! tile. The expensive Arrow-decode checks (4–6, 8–10) can be restricted to a
//! deterministic representative sample with `--sample N` for very large
//! archives; the report then states clearly how many tiles were decoded.
//!
//! Exits non-zero on any failure; warnings (tolerated legacy gaps) and notes
//! (observations that are not defects — a distinct-id deviation on an archive
//! whose ids were never a dataset-wide key) are reported but never affect the
//! exit code. With `--json` emits a machine-readable report; without it emits a
//! short human summary.

mod fingerprint;
mod schema;
mod summary;
mod temporal;

use anyhow::{Context, Result};
use clap::Parser;
use fingerprint::FingerprintAccumulator;
use indicatif::{ProgressBar, ProgressStyle};
use schema::{
    check_tile_schema, classify_column_drift, layer_schema_columns, schema_signature, ColumnDrift,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;
use stt_core::metadata::Metadata;
use stt_core::TileEntry;
use stt_core::{Manifest, PackedReader};

#[derive(Parser)]
#[command(
    name = "stt-validate",
    version,
    about = "Validate an STT dataset",
    after_help = "\
CHEAP TIER (always runs over every tile): content addressing (blake3 names, \
declared lengths, pack_id ranges), paged-directory structure, per-tile CRC + \
uncompressed size, per-entry temporal bounds vs the metadata time range, and \
metadata tile/feature totals vs the directory-derived totals (zero totals \
from pre-0.1.1 writers warn instead of failing).

DECODE TIER (every tile unless --sample N restricts or --skip-decode skips): \
Arrow layer-frame decode; schema/column contract incl. stt:quant on \
Int32-leaf geometry (and its absence on Float64-leaf), u16-delta vertex_time \
origin/step metadata (and its absence on absolute Int64 vertex_time), and \
CRS84 extension metadata on unquantized geometry (warns when absent — older \
archives predate the writer MUST); per-tile feature counts; schema drift \
across tiles; interval sanity (end_time >= start_time, counted with the \
first offender); directory time_end tightness (must equal the max feature \
end_time — interval queries prune on it); and summary-tier cell-id validity \
(H3/Quadbin cells at the tier's resolution for the tile zoom).

SEMANTIC TIER (check 12, only when the archive declares one): the decoded \
vertex bbox, z range, per-column numeric ranges and categorical \
cardinalities are recomputed and compared against \
metadata.content_fingerprint — containment under --sample, containment plus \
equality within the declared tolerances under a full decode. The declared \
distinct-feature count is compared too: decoding materially FEWER distinct \
feature ids than declared on a full decode is FEATURE LOSS and fails the run \
(--allow-distinct-shortfall demotes it to a warning); under --sample, and for \
any overshoot, it warns. An archive with \
no fingerprint warns (its CONTENT is unverified) but never fails. \
--emit-fingerprint / --expect-fingerprint drive the same comparison against \
an EXTERNAL file, which is how a lossless transform is accepted without \
being allowed to vouch for itself.

BOUNDS TIER (check 13, on every archive): metadata.bounds must CONTAIN the \
decoded vertex bbox, and metadata.z_range the decoded vertical extent. Too \
SMALL is the failure — readers pre-intersect query boxes against bounds, so an \
under-stated box silently discards tiles that hold visible data; too LARGE is \
merely conservative and warns at most. An ERROR when the archive attests \
vertex-derived bounds (a content fingerprint, or a bounds_mode=vertex \
property), a WARNING naming the rebuild otherwise.

Warnings never affect the exit code."
)]
struct Args {
    /// Dataset to validate: a packed dataset directory, its `manifest.json`,
    /// or a single-file `.sttb` bundle.
    archive: PathBuf,

    /// Emit a JSON report instead of a human summary.
    #[arg(long)]
    json: bool,

    /// Stop after the first failing tile. Default: keep going and report all.
    #[arg(long)]
    fail_fast: bool,

    /// Skip the per-tile decode step (only checks header/integrity + index +
    /// content hashes).
    #[arg(long)]
    skip_decode: bool,

    /// Decode only a deterministic, evenly-spread sample of at most N tiles
    /// instead of every tile. Integrity / header / content-hash / temporal-
    /// bound checks still run over ALL tiles (they're cheap); only the
    /// expensive Arrow-decode + schema + feature-count checks are sampled.
    /// The sample is reproducible (every `ceil(total/N)`-th entry), and the
    /// report makes clear that the decode was sampled rather than exhaustive.
    #[arg(long, value_name = "N")]
    sample: Option<usize>,

    /// Write the OBSERVED content fingerprint (check 12) to PATH as JSON.
    ///
    /// This is the *capture* half of transform acceptance: run it against a
    /// TRUSTED archive, then check the transformed copy with
    /// `--expect-fingerprint`. Requires a full decode — a fingerprint captured
    /// from a `--sample`d or `--skip-decode`d run understates the content, and
    /// an understated expectation is a check that cannot fail.
    ///
    /// ⚠️ A losslessly-transforming tool must never use this to re-stamp its
    /// OWN output: recomputing a fingerprint from the output is exactly how a
    /// corrupting transform would self-certify.
    #[arg(long, value_name = "PATH")]
    emit_fingerprint: Option<PathBuf>,

    /// Compare the decoded content against an EXTERNAL fingerprint file
    /// (written by `--emit-fingerprint`) in addition to any the manifest
    /// declares.
    ///
    /// The acceptance gate for a lossless transform (reorder, repack,
    /// re-optimize): the expectation comes from the archive as it was BEFORE
    /// the transform, so the transform cannot vouch for itself.
    #[arg(long, value_name = "PATH")]
    expect_fingerprint: Option<PathBuf>,

    /// Downgrade check 12's FEATURE-LOSS errors back to warnings — both of
    /// them: the decoded-ROW floor (the fullest zoom carrying fewer rows than
    /// the declared `distinct_feature_count`) and, on an archive that attests
    /// globally distinct ids, the distinct-id shortfall.
    ///
    /// The documented escape hatch for the benign causes: a source whose
    /// features carry (or hash to) colliding ids, and features the tiler could
    /// not place at all — neither of which the archive can tell apart from
    /// features that were dropped. It suppresses NOTHING else, and the
    /// downgraded finding is still reported, labelled with this flag.
    ///
    /// ⚠️ Reach for it only once you know where the features went. A rebuild
    /// that silently drops them produces exactly the same finding.
    #[arg(long)]
    allow_distinct_shortfall: bool,
}

#[derive(Serialize, Default)]
struct Report {
    archive: String,
    /// Packed archive manifest version.
    format_version: u32,
    /// Directory codec version.
    directory_version: u8,
    compression: String,
    tile_count: usize,
    /// Number of tiles actually decoded (== `tile_count` unless `--sample`/
    /// `--skip-decode` reduced it). Always <= `tile_count`.
    tiles_decoded: usize,
    /// True when the per-tile decode was restricted to a `--sample`d subset, so
    /// a reader never mistakes a sampled run for a full verification.
    sampled: bool,
    feature_count_index: u64,
    feature_count_decoded: u64,
    /// Whether `feature_count_decoded` reflects EVERY tile. False when decoding
    /// was sampled or skipped, in which case the decoded sum covers only the
    /// inspected subset and the grand-total metadata check is not run.
    feature_count_decoded_complete: bool,
    /// Number of distinct layer schemas observed across the decoded tiles. >1
    /// means producer drift (see `errors` for the disagreeing signatures).
    distinct_schemas: usize,
    payload_bytes_compressed: u64,
    payload_bytes_uncompressed: u64,
    /// True when check 12 actually compared decoded content against a declared
    /// (or `--expect-fingerprint`ed) semantic fingerprint. False means the
    /// archive's CONTENT was not verified — only its structure — which is the
    /// state every pre-SH-1 archive is in.
    fingerprint_checked: bool,
    /// True when check 13 actually compared `metadata.bounds` against a decoded
    /// vertex bbox. False means no tile was decoded (or none carried a walkable
    /// coordinate), so the manifest's spatial claim went unverified.
    bounds_checked: bool,
    /// True when a check-13 under-statement would FAIL rather than warn — i.e.
    /// the archive attests vertex-derived bounds. False is the pre-R1 fleet,
    /// where the declared bbox is centroid-derived and legitimately loose.
    bounds_enforced: bool,
    /// What check 12 was allowed to compare `distinct_feature_count` against:
    /// `"source-features"` (the default and the whole current fleet — the
    /// declared count and the decoded id count are different quantities) or
    /// `"decoded-ids"` (the archive attests globally distinct ids, so a
    /// shortfall is real feature loss).
    ///
    /// Resolved from the WRITER's recorded id construction
    /// (`metadata.properties.feature_id_construction`), or from an explicit
    /// attestation (`feature_id_scope`, the `global-feature-ids` capability) —
    /// and never from the numbers happening to agree. Empty only when the run
    /// aborted before the fingerprint finalizer.
    #[serde(skip_serializing_if = "str::is_empty")]
    distinct_id_basis: &'static str,
    /// The writer's own `metadata.properties.feature_id_construction`, echoed
    /// verbatim: `source` / `anchor-hash` (a dataset-wide key, so
    /// `distinct_id_basis` is `decoded-ids`), `row-index` / `segment-hash` (not
    /// a key). Empty on an archive that predates the key.
    ///
    /// Reported beside the basis because the basis alone used to MISDESCRIBE the
    /// archive: every non-attesting one was explained with the point-archive
    /// row-index story even when the writer had used a stable anchor hash.
    #[serde(skip_serializing_if = "String::is_empty")]
    feature_id_construction: String,
    elapsed_ms: u128,
    errors: Vec<String>,
    /// Non-fatal findings (tolerated legacy gaps, e.g. missing CRS84 metadata
    /// or zeroed pre-0.1.1 manifest totals). Never affect the exit code.
    warnings: Vec<String>,
    /// Below a warning: observations worth STATING that are not defects and are
    /// not actionable — a distinct-id deviation on an archive whose ids were
    /// never a dataset-wide key, say.
    ///
    /// Deliberately NOT warnings. Warnings are the noise budget the 2026-08
    /// optimization program drives to zero (objective O4); filing a finding
    /// that is expected on every archive in the fleet as a warning would
    /// inflate exactly that number and teach operators to skim the block. Never
    /// affect the exit code.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    notes: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();

    let mut report = Report {
        archive: args.archive.display().to_string(),
        ..Default::default()
    };

    let metadata = if let Some(bundle) = bundle_input_path(&args.archive) {
        validate_bundle(&bundle, &args, &mut report)?
    } else if let Some(manifest_path) = packed_manifest_path(&args.archive) {
        validate_packed(&manifest_path, &args, &mut report)?
    } else {
        anyhow::bail!(
            "{} is not a packed STT dataset (expected a directory containing \
             manifest.json, the manifest.json itself, or a single-file .sttb \
             bundle). Single-file `.stt` archives are an internal build \
             intermediate and are not validated — rebuild with `stt-build` \
             (packed output is the default)",
            args.archive.display()
        );
    };

    report.elapsed_ms = start.elapsed().as_millis();

    if args.json {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        serde_json::to_writer_pretty(&mut out, &report)?;
        println!();
    } else {
        print_summary(&report, &metadata);
    }

    if report.errors.is_empty() {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

/// If `input` denotes a single-file `.sttb` bundle, return its path.
/// Detected by extension or by the 4-byte `STTB` magic, so a renamed bundle
/// still validates; a `manifest.json` input never matches (its first bytes
/// are JSON, not magic) and falls through to the packed-directory path.
fn bundle_input_path(input: &Path) -> Option<PathBuf> {
    if !input.is_file() {
        return None;
    }
    if input.extension().map(|e| e == "sttb").unwrap_or(false) {
        return Some(input.to_path_buf());
    }
    use std::io::Read;
    let mut magic = [0u8; 4];
    let mut f = std::fs::File::open(input).ok()?;
    f.read_exact(&mut magic).ok()?;
    (magic == stt_core::pack::BUNDLE_MAGIC).then(|| input.to_path_buf())
}

/// If `input` denotes a packed dataset, return its `manifest.json` path.
fn packed_manifest_path(input: &Path) -> Option<PathBuf> {
    if input.is_dir() {
        let manifest = input.join("manifest.json");
        return manifest.is_file().then_some(manifest);
    }
    if input
        .file_name()
        .map(|f| f == "manifest.json")
        .unwrap_or(false)
    {
        return Some(input.to_path_buf());
    }
    None
}

/// Validate a packed dataset: content-addressing integrity, then per-tile.
fn validate_packed(manifest_path: &Path, args: &Args, report: &mut Report) -> Result<Metadata> {
    // Content-addressing / declared-length / pack_id-range integrity first.
    let integrity = stt_core::verify_packed_objects(manifest_path)
        .with_context(|| format!("failed to read packed manifest {}", manifest_path.display()))?;
    for issue in integrity {
        push_err(report, args.fail_fast, format!("integrity: {issue}"))?;
    }

    let reader = PackedReader::open(manifest_path)
        .with_context(|| format!("failed to open packed dataset {}", manifest_path.display()))?;
    let manifest = Manifest::from_json_bytes(&std::fs::read(manifest_path)?)?;
    report.format_version = manifest.format_version;
    report.directory_version = manifest.directory.directory_version;
    report.compression = manifest.compression.clone();

    verify_paged_directly(manifest_path, &manifest, args, report)?;

    validate_with_reader(&reader, args, report)
}

/// Validate a single-file `.sttb` bundle: the bundle-integrity tier first
/// (every in-bundle object's blake3 against its key, in-bundle vs declared
/// lengths, directory decode + `pack_id` range, paged structure — exactly
/// the exploded checks, sourced from bundle windows), then the same
/// decode-tier checks through the bundle-backed reader.
fn validate_bundle(bundle_path: &Path, args: &Args, report: &mut Report) -> Result<Metadata> {
    let integrity = stt_core::pack::verify_bundle_objects(bundle_path)
        .with_context(|| format!("failed to read bundle {}", bundle_path.display()))?;
    for issue in integrity {
        push_err(report, args.fail_fast, format!("integrity: {issue}"))?;
    }

    let reader = PackedReader::open_bundle(bundle_path)
        .with_context(|| format!("failed to open bundle {}", bundle_path.display()))?;
    let manifest = stt_core::pack::read_bundle_manifest(bundle_path)?;
    report.format_version = manifest.format_version;
    report.directory_version = manifest.directory.directory_version;
    report.compression = manifest.compression.clone();

    validate_with_reader(&reader, args, report)
}

/// Decode-tier validation shared by the exploded and bundle inputs: every
/// entry through `read_payload` + the per-tile checks, then the
/// metadata-total reconciliation.
fn validate_with_reader(
    reader: &PackedReader,
    args: &Args,
    report: &mut Report,
) -> Result<Metadata> {
    let metadata = reader.metadata().clone();
    let entries = reader.entries().to_vec();
    report.tile_count = entries.len();

    // Check 12's accumulator rides the SAME decode pass as the schema/temporal
    // checks — no second read of any payload.
    let mut accumulator =
        FingerprintAccumulator::new().skipping_layers(fingerprint::skipped_layers(&metadata));

    let pb = make_progress(args, entries.len());
    validate_entries(
        &entries,
        &metadata,
        |e| reader.read_payload(e).map_err(|err| err.to_string()),
        reader.templates(),
        args,
        report,
        pb.as_ref(),
        &mut accumulator,
    )?;
    if let Some(p) = &pb {
        p.finish_and_clear();
    }

    finalize_feature_check(args, report, &metadata);
    // One observation, two consumers: check 12 compares it against the
    // fingerprint block, check 13 against `metadata.bounds`.
    let observed = accumulator.finish();
    finalize_fingerprint_check(args, report, &metadata, reader.capabilities(), &observed)?;
    finalize_bounds_check(args, report, &metadata, &observed)?;
    Ok(metadata)
}

/// Check 13: the declared `metadata.bounds` must CONTAIN the decoded vertex
/// bbox.
///
/// Its own finalizer rather than a limb of [`finalize_fingerprint_check`]:
/// check 13 runs on **every** archive (the bbox is not optional the way a
/// fingerprint is), and the two have independent severity gates that a shared
/// function would tangle.
fn finalize_bounds_check(
    args: &Args,
    report: &mut Report,
    metadata: &Metadata,
    observed: &fingerprint::ObservedFingerprint,
) -> Result<()> {
    report.bounds_enforced = fingerprint::bounds_are_attested(metadata);
    // Nothing decoded ⇒ nothing observed to contain. Saying "bounds verified"
    // off an empty decode would be the same lie check 12 refuses to tell.
    if report.tiles_decoded == 0 || observed.bbox.is_none() {
        return Ok(());
    }
    report.bounds_checked = true;

    let findings = fingerprint::check_declared_bounds(metadata, observed, decode_is_complete(args));
    for issue in findings.errors {
        push_err(report, args.fail_fast, format!("declared bounds: {issue}"))?;
    }
    for warning in findings.warnings {
        push_warn(report, format!("declared bounds: {warning}"));
    }
    Ok(())
}

/// Check 12: compare the decoded content against the declared fingerprint (and
/// against `--expect-fingerprint`, when given), then service
/// `--emit-fingerprint`.
///
/// Split out of [`finalize_feature_check`] rather than folded into it: that
/// function is the metadata-total reconciliation and has its own unit test
/// pinning its signature and its two-warning shape.
fn finalize_fingerprint_check(
    args: &Args,
    report: &mut Report,
    metadata: &Metadata,
    capabilities: &[String],
    observed: &fingerprint::ObservedFingerprint,
) -> Result<()> {
    let complete = decode_is_complete(args);

    // Nothing was decoded at all: there is no observation to compare, and
    // saying "content verified" off an empty decode would be a lie.
    let decoded_anything = report.tiles_decoded > 0;

    // BLOCKER A/1: WHAT the declared `distinct_feature_count` may be compared
    // against. Keyed on the WRITER's recorded id construction
    // (`feature_id_construction`) — a fact the builder always knows — plus the
    // explicit `feature_id_scope` / capability attestations, and NEVER inferred
    // from the numbers happening to agree. On a construction that is not a
    // dataset-wide key the declared count (source features) and the decoded id
    // count (a per-tile row index, or one id per clipped segment) are different
    // quantities, and their difference is a note rather than a finding. See
    // `stt_core::metadata::DistinctIdBasis`.
    let basis = fingerprint::distinct_id_basis(metadata, capabilities);
    report.distinct_id_basis = match basis {
        fingerprint::DistinctIdBasis::DecodedIds => "decoded-ids",
        fingerprint::DistinctIdBasis::SourceFeatures => "source-features",
    };
    report.feature_id_construction = metadata
        .properties
        .get(stt_core::metadata::FEATURE_ID_CONSTRUCTION_PROPERTY)
        .cloned()
        .unwrap_or_default();

    let declared = metadata.content_fingerprint.as_ref();
    match declared {
        Some(declared) if decoded_anything => {
            let findings = fingerprint::check_fingerprint_with(
                declared,
                observed,
                complete,
                capabilities,
                basis,
            );
            push_fingerprint_findings(args, report, "content fingerprint", findings)?;
            report.fingerprint_checked = true;
        }
        Some(_) => push_warn(
            report,
            "metadata.content_fingerprint is present but nothing was decoded, so the \
             archive's CONTENT was not verified"
                .to_string(),
        ),
        None => push_warn(report, fingerprint::absent_fingerprint_warning()),
    }

    if let Some(path) = &args.expect_fingerprint {
        let expected = fingerprint::read_fingerprint_file(path)?;
        if !decoded_anything {
            push_err(
                report,
                args.fail_fast,
                format!(
                    "--expect-fingerprint {}: nothing was decoded, so the expectation could \
                     not be checked (drop --skip-decode)",
                    path.display()
                ),
            )?;
        } else {
            let findings = fingerprint::check_fingerprint_with(
                &expected,
                observed,
                complete,
                capabilities,
                basis,
            );
            let label = format!("expected fingerprint ({})", path.display());
            push_fingerprint_findings(args, report, &label, findings)?;
            report.fingerprint_checked = true;
        }
    }

    if let Some(path) = &args.emit_fingerprint {
        if !complete {
            anyhow::bail!(
                "--emit-fingerprint needs a FULL decode: a fingerprint captured from a \
                 sampled or skipped decode understates the archive's content, and using it \
                 as an expectation later would be a check that cannot fail. Re-run without \
                 --sample / --skip-decode."
            );
        }
        fingerprint::write_fingerprint_file(
            path,
            &fingerprint::observed_as_declared(observed, declared),
        )?;
    }

    Ok(())
}

/// Route one fingerprint comparison's findings into the report under `label`.
///
/// The one piece of policy here is `--allow-distinct-shortfall`: it, and only
/// it, demotes the FEATURE-LOSS error (check 12's distinct-count shortfall) to
/// a warning. The demotion is explicit in the reported text — a suppressed
/// finding that reads like a clean run is how the 106-archive defect survived
/// in the first place — and every other error is untouchable.
fn push_fingerprint_findings(
    args: &Args,
    report: &mut Report,
    label: &str,
    findings: fingerprint::FingerprintFindings,
) -> Result<()> {
    for issue in findings.errors {
        if args.allow_distinct_shortfall
            && issue.starts_with(stt_core::metadata::FEATURE_LOSS_PREFIX)
        {
            push_warn(
                report,
                format!("{label} [DOWNGRADED by --allow-distinct-shortfall]: {issue}"),
            );
            continue;
        }
        push_err(report, args.fail_fast, format!("{label}: {issue}"))?;
    }
    for warning in findings.warnings {
        push_warn(report, format!("{label}: {warning}"));
    }
    for note in findings.notes {
        push_note(report, format!("{label}: {note}"));
    }
    Ok(())
}

/// Run the paged-structure check (leaf-descriptor bounds cover their entries,
/// cross-page key order monotonic) DIRECTLY for a paged directory.
///
/// `verify_packed_objects` already runs this transitively, but the paged
/// covering invariant is a named part of the validator's contract
/// (`docs/spec/conformance.md` §2.3) — calling it here as well means a refactor
/// of the integrity pass cannot silently drop it. Issues the integrity pass
/// already reported are not duplicated.
fn verify_paged_directly(
    manifest_path: &Path,
    manifest: &Manifest,
    args: &Args,
    report: &mut Report,
) -> Result<()> {
    if !manifest.directory.is_paged() {
        return Ok(());
    }
    let push_unique = |report: &mut Report, msg: String| -> Result<()> {
        if report.errors.contains(&msg) {
            return Ok(());
        }
        push_err(report, args.fail_fast, msg)
    };
    let Some(root_length) = manifest.directory.root_length else {
        push_unique(
            report,
            "integrity: paged directory: manifest missing rootLength".into(),
        )?;
        return Ok(());
    };
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    // A missing/unreadable directory object is already reported by the
    // content-address pass; only the structural check is re-asserted here.
    let Ok(dir_bytes) = std::fs::read(root.join(&manifest.directory.key)) else {
        return Ok(());
    };
    let zstd =
        manifest.directory.encoding.as_deref() == Some(stt_core::pack::DIRECTORY_ENCODING_ZSTD);
    // formatVersion 3 prefixes the object with the 8-byte STTD magic; the
    // codec bytes (and rootLength math) start after it. Malformed magic is
    // already reported by the integrity pass — skip the re-check here.
    let Ok(codec_bytes) =
        stt_core::pack::directory_codec_bytes(&dir_bytes, manifest.format_version)
    else {
        return Ok(());
    };
    match stt_core::verify_paged_structure(codec_bytes, root_length, zstd) {
        Ok(issues) => {
            for issue in issues {
                push_unique(report, format!("integrity: {issue}"))?;
            }
        }
        Err(e) => push_unique(
            report,
            format!("integrity: paged structure check failed: {e}"),
        )?,
    }
    Ok(())
}

/// Whether the per-tile decode covers EVERY tile. The grand-total feature-count
/// check (and the `feature_count_decoded` total being meaningful) depend on a
/// complete decode: a sampled or skipped decode only sums a subset.
fn decode_is_complete(args: &Args) -> bool {
    !args.skip_decode && args.sample.is_none()
}

/// Deterministic stride for `--sample N`: pick every `ceil(total/N)`-th entry,
/// starting at index 0, yielding at most `N` evenly-spread tiles. Reproducible
/// across runs (no randomness) so a sampled "OK" is something a producer can
/// re-confirm. `N == 0` decodes nothing; `N >= total` decodes everything.
fn sample_stride(total: usize, n: usize) -> usize {
    if n == 0 {
        return usize::MAX; // nothing selected
    }
    total.div_ceil(n).max(1)
}

/// Per-tile validation shared by both readers. `read_payload` reads (and CRC-
/// and size-checks) one tile's decompressed bytes from whichever container.
///
/// Cheap per-entry checks (temporal bounds, byte/feature tallies, content
/// hashes via `read_payload`) run for **every** entry. The expensive
/// Arrow-decode + schema + feature-count checks run for every entry too, unless
/// `--sample N` restricts them to a deterministic stride of the directory.
///
/// The parameter list is long because this is the single decode pass every
/// per-tile check rides (bundling them into a context struct would only move
/// the list); check 12's accumulator is the eighth.
#[allow(clippy::too_many_arguments)]
fn validate_entries(
    entries: &[TileEntry],
    metadata: &Metadata,
    mut read_payload: impl FnMut(&TileEntry) -> std::result::Result<Vec<u8>, String>,
    templates: Option<&stt_core::arrow_tile::TemplateRegistry>,
    args: &Args,
    report: &mut Report,
    pb: Option<&ProgressBar>,
    accumulator: &mut FingerprintAccumulator,
) -> Result<()> {
    let meta_start = metadata.time_range.start as i64;
    let meta_end = metadata.time_range.end as i64;

    report.sampled = args.sample.is_some() && !args.skip_decode;
    let stride = args.sample.map(|n| sample_stride(entries.len(), n));

    // Distinct schema signatures seen across decoded layers, for producer-drift
    // detection. We keep the first tile that exhibited each signature so the
    // error can name a concrete disagreeing pair.
    // Keyed by LAYER NAME, not by the whole-tile signature: which layers a tile
    // carries legitimately varies across a dataset (an `_originals` companion
    // layer that only exists at some zooms, a summary layer only at the tier's
    // zooms). Only a given layer's own column schema must stay stable — keying
    // on the tile-wide signature flagged every such dataset as drifting.
    let mut schemas: BTreeSet<String> = BTreeSet::new();
    let mut first_layer_schema: BTreeMap<String, (String, Vec<(String, String)>)> = BTreeMap::new();

    // Interval-sanity tally across every decoded tile: one aggregate error
    // (count + first offender) instead of one error per bad feature.
    let mut interval_violations = 0u64;
    let mut interval_first: Option<String> = None;

    // Schema warnings aggregated per distinct message body — an old archive
    // missing CRS84 metadata would otherwise warn once per tile. BTreeMap
    // keeps the emitted order deterministic.
    let mut warning_tally: BTreeMap<String, (u64, String)> = BTreeMap::new();

    for (idx, entry) in entries.iter().enumerate() {
        report.payload_bytes_compressed += entry.length as u64;
        report.payload_bytes_uncompressed += entry.uncompressed_size as u64;
        report.feature_count_index += entry.feature_count as u64;

        if entry.time_start < meta_start || entry.time_end > meta_end {
            push_err(
                report,
                args.fail_fast,
                format!(
                    "tile {:?}: temporal extent [{}, {}] outside archive range [{}, {}]",
                    entry.tile_id(),
                    entry.time_start,
                    entry.time_end,
                    meta_start,
                    meta_end
                ),
            )?;
        }

        // Is this entry in the decode set? Skipped entirely under --skip-decode;
        // under --sample only every `stride`-th entry is decoded.
        let decode_this = !args.skip_decode && stride.map(|s| idx % s == 0).unwrap_or(true);

        // read_payload does the content-hash and uncompressed-size checks. We
        // still read non-sampled tiles so the content-address verification
        // stays total — only the Arrow decode below is sampled.
        let payload = match read_payload(entry) {
            Ok(b) => b,
            Err(e) => {
                push_err(
                    report,
                    args.fail_fast,
                    format!("tile {:?} payload read failed: {e}", entry.tile_id()),
                )?;
                if let Some(p) = pb {
                    p.inc(1);
                }
                continue;
            }
        };

        if decode_this {
            // formatVersion-3 frames reference schema templates by hash;
            // resolve them through the manifest's registry (v1 datasets have
            // none and take the plain path).
            let decoded = match templates {
                Some(registry) => {
                    stt_core::arrow_tile::decode_tile_with_templates(&payload, registry)
                }
                None => stt_core::arrow_tile::decode_tile(&payload),
            };
            match decoded {
                Ok(layers) => {
                    report.tiles_decoded += 1;
                    let row_total: u64 = layers.iter().map(|l| l.batch.num_rows() as u64).sum();
                    report.feature_count_decoded += row_total;
                    if row_total != entry.feature_count as u64 {
                        push_err(
                            report,
                            args.fail_fast,
                            format!(
                                "tile {:?}: directory says {} features, decoded {}",
                                entry.tile_id(),
                                entry.feature_count,
                                row_total
                            ),
                        )?;
                    }

                    // Check 12: fold this tile's decoded content into the
                    // semantic fingerprint. Done here, inside the one decode
                    // pass, so the content check costs no extra payload read.
                    //
                    // `_at_zoom` books the rows against this tile's zoom, which
                    // is what feeds the decoded-ROW floor — the loss detection
                    // that survives the distinct-id basis gate. A caller that
                    // used the plain `ingest` would silently disable it.
                    accumulator.ingest_at_zoom(entry.zoom, &layers);

                    // Schema / column-type contract per layer.
                    let findings = check_tile_schema(&layers);
                    for issue in findings.errors {
                        push_err(
                            report,
                            args.fail_fast,
                            format!("tile {:?} schema: {issue}", entry.tile_id()),
                        )?;
                    }
                    for warning in findings.warnings {
                        let slot = warning_tally
                            .entry(warning)
                            .or_insert_with(|| (0, format!("tile {:?}", entry.tile_id())));
                        slot.0 += 1;
                    }

                    // Interval sanity (check 8): end_time >= start_time.
                    let (violations, first) = temporal::check_interval_sanity(&layers);
                    if violations > 0 {
                        interval_violations += violations;
                        if interval_first.is_none() {
                            let o = first.expect("violations > 0 implies an offender");
                            interval_first = Some(format!(
                                "tile {:?} layer '{}' feature {} (start_time={}, end_time={})",
                                entry.tile_id(),
                                o.layer,
                                o.feature_index,
                                o.start_time,
                                o.end_time
                            ));
                        }
                    }

                    // time_end tightness (check 9, review T1.5): the entry's
                    // time_end must equal the max feature end_time — interval
                    // features are findable past their start bucket ONLY via
                    // this widening, so a nominal bucket end breaks interval
                    // queries with no other symptom.
                    if let Some(max_end) = temporal::max_feature_end(&layers) {
                        if max_end != entry.time_end {
                            push_err(
                                report,
                                args.fail_fast,
                                format!(
                                    "tile {:?}: directory time_end {} is not tight: max feature \
                                     end_time is {max_end} (readers prune interval queries on \
                                     time_end; the writer must widen it to the max feature end)",
                                    entry.tile_id(),
                                    entry.time_end
                                ),
                            )?;
                        }
                    }

                    // Summary cell-id validity (check 10): summary-layer ids
                    // ARE the aggregation cell indices; anything else renders
                    // blank client-side with no error anywhere.
                    if let Some(tier) = metadata.summary_tier.as_ref() {
                        for issue in summary::check_summary_cells(tier, entry.zoom, &layers) {
                            push_err(
                                report,
                                args.fail_fast,
                                format!("tile {:?} summary: {issue}", entry.tile_id()),
                            )?;
                        }
                    }

                    // Producer-drift detection: track the distinct schema
                    // signatures and flag the first tile that disagrees with
                    // an earlier one.
                    schemas.insert(schema_signature(&layers));
                    for (name, cols) in layer_schema_columns(&layers) {
                        let Some((first_tile, first_cols)) = first_layer_schema.get(&name) else {
                            first_layer_schema
                                .insert(name, (format!("{:?}", entry.tile_id()), cols));
                            continue;
                        };
                        // Compare per COLUMN, and only report the ones that
                        // actually disagree — a whole-signature diff buries the
                        // one drifting column in the full column list.
                        let lookup = |set: &[(String, String)], c: &str| {
                            set.iter().find(|(n, _)| n == c).map(|(_, t)| t.clone())
                        };
                        let mut structural: Vec<String> = Vec::new();
                        let mut adaptive: Vec<String> = Vec::new();
                        let mut names: Vec<&str> =
                            first_cols.iter().map(|(n, _)| n.as_str()).collect();
                        for (n, _) in &cols {
                            if !names.contains(&n.as_str()) {
                                names.push(n);
                            }
                        }
                        for col in names {
                            let a = lookup(first_cols, col);
                            let b = lookup(&cols, col);
                            if a == b {
                                continue;
                            }
                            let shown = format!(
                                "{col}: {} vs {}",
                                a.as_deref().unwrap_or("<absent>"),
                                b.as_deref().unwrap_or("<absent>")
                            );
                            match classify_column_drift(col, a.as_deref(), b.as_deref()) {
                                ColumnDrift::AdaptiveWidth => adaptive.push(shown),
                                ColumnDrift::Structural => structural.push(shown),
                            }
                        }
                        if !structural.is_empty() {
                            push_err(
                                report,
                                args.fail_fast,
                                format!(
                                    "schema drift: tile {:?} layer {name:?} differs from tile \
                                     {first_tile} — {}",
                                    entry.tile_id(),
                                    structural.join("; ")
                                ),
                            )?;
                        } else if !adaptive.is_empty() {
                            // Per-tile adaptive encoding widths: expected by
                            // construction, and every reader handles both
                            // shapes. Tallied as a warning, never a failure.
                            let slot = warning_tally
                                .entry(format!(
                                    "layer {name:?}: adaptive encoding width varies by tile \
                                     (expected — the encoding sizes itself per tile)"
                                ))
                                .or_insert_with(|| (0, adaptive.join("; ")));
                            slot.0 += 1;
                        }
                    }
                }
                Err(e) => {
                    push_err(
                        report,
                        args.fail_fast,
                        format!("tile {:?} decode failed: {e}", entry.tile_id()),
                    )?;
                }
            }
        }

        if let Some(p) = pb {
            p.inc(1);
        }
    }

    if interval_violations > 0 {
        push_err(
            report,
            args.fail_fast,
            format!(
                "interval sanity: {interval_violations} feature(s) with end_time < start_time \
                 across the decoded tiles; first: {}",
                interval_first.as_deref().unwrap_or("<unknown>")
            ),
        )?;
    }
    for (msg, (count, first_tile)) in warning_tally {
        push_warn(
            report,
            format!("{msg} — {count} decoded tile(s), first {first_tile}"),
        );
    }

    report.distinct_schemas = schemas.len();
    report.feature_count_decoded_complete = decode_is_complete(args);
    Ok(())
}

fn finalize_feature_check(args: &Args, report: &mut Report, metadata: &Metadata) {
    // Metadata totals vs the directory-derived ground truth (check 11) — no
    // decode needed, the entry list carries both counts. Zeros are a
    // pre-0.1.1 writer gap, not corruption: warn, and name the fix.
    if metadata.tile_count == 0 && report.tile_count > 0 {
        push_warn(
            report,
            format!(
                "metadata tile_count is 0 with {} directory entries — pre-0.1.1 manifests \
                 never set the totals; rebuilding with current stt-build repopulates them",
                report.tile_count
            ),
        );
    } else if metadata.tile_count != report.tile_count as u64 {
        report.errors.push(format!(
            "metadata tile_count={} disagrees with directory entry count {}",
            metadata.tile_count, report.tile_count
        ));
    }
    if metadata.feature_count == 0 && report.feature_count_index > 0 {
        push_warn(
            report,
            format!(
                "metadata feature_count is 0 with {} features in the directory — pre-0.1.1 \
                 manifests never set the totals; rebuilding with current stt-build \
                 repopulates them",
                report.feature_count_index
            ),
        );
    } else if metadata.feature_count != 0 && metadata.feature_count != report.feature_count_index {
        report.errors.push(format!(
            "metadata feature_count={} disagrees with the directory-derived total {}",
            metadata.feature_count, report.feature_count_index
        ));
    }
    // Decoded grand total: only meaningful when EVERY tile was decoded — a
    // sampled or skipped decode sums a subset, so comparing it to the metadata
    // grand total would spuriously fail. Such runs leave
    // `feature_count_decoded_complete = false` and skip it. When the decoded
    // sum matches the directory-derived total the cheap check above already
    // said everything (per-tile mismatches were reported individually), so
    // this only fires when the payloads disagree with the directory itself.
    if decode_is_complete(args)
        && report.feature_count_decoded != report.feature_count_index
        && report.feature_count_decoded != metadata.feature_count
        && metadata.feature_count != 0
    {
        report.errors.push(format!(
            "metadata feature_count={} disagrees with decoded sum {}",
            metadata.feature_count, report.feature_count_decoded
        ));
    }
}

fn make_progress(args: &Args, len: usize) -> Option<ProgressBar> {
    if args.json {
        return None;
    }
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    Some(pb)
}

fn push_err(report: &mut Report, fail_fast: bool, msg: String) -> Result<()> {
    eprintln!("error: {msg}");
    report.errors.push(msg);
    if fail_fast {
        anyhow::bail!("validation failed (--fail-fast)");
    }
    Ok(())
}

/// Non-fatal finding: reported (stderr + report), never affects the exit code
/// and never trips `--fail-fast`.
fn push_warn(report: &mut Report, msg: String) {
    eprintln!("warning: {msg}");
    report.warnings.push(msg);
}

/// Below a warning: stated, counted separately, never fatal. See
/// [`Report::notes`] for why this is not simply a warning.
fn push_note(report: &mut Report, msg: String) {
    eprintln!("note: {msg}");
    report.notes.push(msg);
}

fn print_summary(report: &Report, metadata: &Metadata) {
    println!("dataset          {}", report.archive);
    println!("format version   {}", report.format_version);
    println!("directory version {}", report.directory_version);
    println!("compression      {}", report.compression);
    println!("name             {}", metadata.name);
    println!("tiles            {}", report.tile_count);
    if report.sampled {
        println!(
            "decoded          {} of {} tiles (SAMPLED — not a full verification)",
            report.tiles_decoded, report.tile_count
        );
    } else {
        println!(
            "decoded          {} of {} tiles",
            report.tiles_decoded, report.tile_count
        );
    }
    println!("distinct schemas {}", report.distinct_schemas);
    if report.feature_count_decoded_complete {
        println!(
            "features (index) {}  (decoded sum: {})",
            report.feature_count_index, report.feature_count_decoded
        );
    } else {
        println!(
            "features (index) {}  (decoded sum: {} over {} tiles; grand-total not checked — {})",
            report.feature_count_index,
            report.feature_count_decoded,
            report.tiles_decoded,
            if report.sampled {
                "sampled"
            } else {
                "decode skipped"
            }
        );
    }
    println!(
        "payload bytes    {} compressed / {} uncompressed ({:.2}x ratio)",
        report.payload_bytes_compressed,
        report.payload_bytes_uncompressed,
        if report.payload_bytes_compressed == 0 {
            0.0
        } else {
            report.payload_bytes_uncompressed as f64 / report.payload_bytes_compressed as f64
        }
    );
    println!(
        "time range       {} .. {}",
        metadata.time_range.start, metadata.time_range.end
    );
    println!(
        "zoom range       {} .. {}",
        metadata.min_zoom, metadata.max_zoom
    );
    println!(
        "content check    {}",
        if report.fingerprint_checked {
            "semantic fingerprint compared"
        } else {
            "STRUCTURE ONLY (no content fingerprint)"
        }
    );
    println!(
        "bounds check     {}",
        match (report.bounds_checked, report.bounds_enforced) {
            (true, true) => "declared bounds contain the decoded vertices (enforced)",
            (true, false) => "declared bounds compared (legacy centroid bbox — warn only)",
            (false, _) => "NOT CHECKED (nothing decoded carried a coordinate)",
        }
    );
    if !report.distinct_id_basis.is_empty() {
        // The construction is what the basis was DERIVED from, so it is printed
        // beside it: "source-features" on its own used to read as a statement
        // about the archive when it was only a statement about the check.
        let how = match report.feature_id_construction.as_str() {
            "" => String::new(),
            kind => format!(" [id construction: {kind}]"),
        };
        println!(
            "id basis         {}{how}",
            match report.distinct_id_basis {
                "decoded-ids" =>
                    "decoded ids ARE the source features — a shortfall is feature loss",
                _ =>
                    "source features — the id column is not a dataset-wide key, so loss is \
                      caught by the decoded-row floor instead (which is LOOSE on clipped \
                      geometry)",
            }
        );
    }
    println!("elapsed          {} ms", report.elapsed_ms);
    if !report.notes.is_empty() {
        println!(
            "\n{} note(s) (not defects; never affect the exit code):",
            report.notes.len()
        );
        for n in &report.notes {
            println!("  - {n}");
        }
    }
    if !report.warnings.is_empty() {
        println!(
            "\n{} warning(s) (never affect the exit code):",
            report.warnings.len()
        );
        for w in &report.warnings {
            println!("  - {w}");
        }
    }
    if report.errors.is_empty() {
        println!("\nOK");
    } else {
        println!("\n{} error(s):", report.errors.len());
        for e in &report.errors {
            println!("  - {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Doc gate: every visible long flag must
    /// appear in the `stt-validate` section of `docs/api/cli-reference.md`, so
    /// a new flag fails the build until it is documented.
    #[test]
    fn cli_flags_are_documented_in_cli_reference() {
        use clap::CommandFactory;
        let doc = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/api/cli-reference.md"
        ))
        .expect("read docs/api/cli-reference.md");
        let start = doc
            .find("## `stt-validate`")
            .expect("stt-validate section heading");
        let body = &doc[start + 1..];
        let end = body
            .find("\n## `")
            .map(|i| start + 1 + i)
            .unwrap_or(doc.len());
        let section = &doc[start..end];
        let missing: Vec<String> = Args::command()
            .get_arguments()
            .filter(|a| !a.is_hide_set())
            .filter_map(|a| a.get_long())
            .filter(|l| !matches!(*l, "help" | "version"))
            .map(|l| format!("--{l}"))
            .filter(|f| !section.contains(f.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "flags missing from the `stt-validate` section of docs/api/cli-reference.md \
             (document them before shipping): {missing:?}"
        );
    }

    /// Metadata totals (check 11): zeroed pre-0.1.1 totals warn (naming the
    /// rebuild fix), while a nonzero total disagreeing with the
    /// directory-derived count is an error — both without any decode.
    #[test]
    fn metadata_totals_zero_warns_and_nonzero_mismatch_errors() {
        use clap::Parser as _;
        let args = Args::parse_from(["stt-validate", "--skip-decode", "dataset"]);

        // Pre-0.1.1 manifest: both totals zero, directory non-empty.
        let mut report = Report {
            tile_count: 3,
            feature_count_index: 7,
            ..Default::default()
        };
        let legacy = Metadata::new("legacy");
        finalize_feature_check(&args, &mut report, &legacy);
        assert!(report.errors.is_empty(), "errors were: {:?}", report.errors);
        assert_eq!(
            report.warnings.len(),
            2,
            "warnings were: {:?}",
            report.warnings
        );
        assert!(
            report.warnings.iter().all(|w| w.contains("stt-build")),
            "warnings must name the fix: {:?}",
            report.warnings
        );

        // Populated totals that disagree with the directory: errors.
        let mut report = Report {
            tile_count: 3,
            feature_count_index: 7,
            ..Default::default()
        };
        let mut wrong = Metadata::new("wrong");
        wrong.tile_count = 2;
        wrong.feature_count = 9;
        finalize_feature_check(&args, &mut report, &wrong);
        assert!(
            report.warnings.is_empty(),
            "warnings were: {:?}",
            report.warnings
        );
        assert_eq!(report.errors.len(), 2, "errors were: {:?}", report.errors);

        // Matching totals: clean.
        let mut report = Report {
            tile_count: 3,
            feature_count_index: 7,
            ..Default::default()
        };
        let mut ok = Metadata::new("ok");
        ok.tile_count = 3;
        ok.feature_count = 7;
        finalize_feature_check(&args, &mut report, &ok);
        assert!(report.errors.is_empty(), "errors were: {:?}", report.errors);
        assert!(
            report.warnings.is_empty(),
            "warnings were: {:?}",
            report.warnings
        );
    }

    /// Check 12's LEGACY path, which every published archive takes: no
    /// fingerprint ⇒ one warning, zero errors, and `fingerprint_checked` false
    /// so no report can claim the content was verified when it was not.
    ///
    /// Same severity shape as the missing-CRS84 warning, and for the same
    /// reason: a writer MUST newer than the deployed fleet must not turn the
    /// fleet red.
    #[test]
    fn absent_content_fingerprint_warns_and_never_errors() {
        use clap::Parser as _;
        let args = Args::parse_from(["stt-validate", "dataset"]);

        let mut report = Report {
            tiles_decoded: 4,
            ..Default::default()
        };
        finalize_fingerprint_check(
            &args,
            &mut report,
            &Metadata::new("legacy"),
            &[],
            &fingerprint::FingerprintAccumulator::new().finish(),
        )
        .expect("the legacy path never fails the run");
        assert!(report.errors.is_empty(), "errors were: {:?}", report.errors);
        assert_eq!(report.warnings.len(), 1, "warnings: {:?}", report.warnings);
        assert!(
            report.warnings[0].contains("content_fingerprint is absent"),
            "warnings: {:?}",
            report.warnings
        );
        assert!(
            !report.fingerprint_checked,
            "an unverified archive must not report as verified"
        );
    }

    /// Check 13's finalizer: an under-stated bbox fails an attesting archive
    /// and only warns on a legacy one, and neither path can claim the bounds
    /// were checked when nothing was decoded.
    #[test]
    fn declared_bounds_finalizer_gates_severity_on_attestation() {
        use clap::Parser as _;
        use stt_core::types::BoundingBox;
        let args = Args::parse_from(["stt-validate", "dataset"]);

        // The archive decodes vertices in the western hemisphere...
        let observed = fingerprint::ObservedFingerprint {
            bbox: Some([-122.5, 37.7, -122.3, 37.8]),
            rows: 4,
            ..Default::default()
        };
        // ...while declaring a bbox in the eastern one.
        let wrong = BoundingBox::new(100.0, 37.7, 110.0, 37.8);

        let mut report = Report {
            tiles_decoded: 4,
            ..Default::default()
        };
        let legacy = Metadata::new("legacy").with_bounds(wrong);
        finalize_bounds_check(&args, &mut report, &legacy, &observed).unwrap();
        assert!(report.errors.is_empty(), "errors were: {:?}", report.errors);
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert!(report.bounds_checked && !report.bounds_enforced);

        let mut report = Report {
            tiles_decoded: 4,
            ..Default::default()
        };
        let mut attested = Metadata::new("attested").with_bounds(wrong);
        attested
            .properties
            .insert("bounds_mode".into(), "vertex".into());
        finalize_bounds_check(&args, &mut report, &attested, &observed).unwrap();
        assert_eq!(report.errors.len(), 1, "{:?}", report.errors);
        assert!(report.errors[0].starts_with("declared bounds: "));
        assert!(report.bounds_checked && report.bounds_enforced);

        // --skip-decode: no observation, so no claim either way.
        let mut report = Report::default();
        finalize_bounds_check(
            &args,
            &mut report,
            &attested,
            &fingerprint::ObservedFingerprint::default(),
        )
        .unwrap();
        assert!(report.errors.is_empty() && report.warnings.is_empty());
        assert!(
            !report.bounds_checked,
            "an undecoded archive must not report its bounds as verified"
        );
    }
}
