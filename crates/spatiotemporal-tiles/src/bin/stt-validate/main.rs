//! stt-validate — inspect and verify an STT dataset.
//!
//! Accepts the **packed format** only (a dataset directory or its
//! `manifest.json`) and verifies the content-addressing contract. The
//! single-file `.stt` container is an internal streaming/transcode
//! intermediate (spec D3), never a deployment artifact, so it is not
//! accepted here.
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
//!
//! The integrity, header, content-address and per-entry temporal-bound checks
//! (1–3, 7) are cheap and always run over **every** tile. The expensive
//! Arrow-decode + schema + feature-count checks (4–6) can be restricted to a
//! deterministic representative sample with `--sample N` for very large
//! archives; the report then states clearly how many tiles were decoded.
//!
//! Exits non-zero on any failure. With `--json` emits a machine-readable
//! report; without it emits a short human summary.

mod schema;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use schema::{check_tile_schema, schema_signature};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;
use stt_core::archive::TileEntry;
use stt_core::metadata::Metadata;
use stt_core::{Manifest, PackedReader};

#[derive(Parser)]
#[command(name = "stt-validate", version, about = "Validate an STT dataset")]
struct Args {
    /// Dataset to validate: a packed dataset directory or its `manifest.json`.
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
}

#[derive(Serialize, Default)]
struct Report {
    archive: String,
    /// The packed directory codec version (5).
    version: u8,
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
    elapsed_ms: u128,
    errors: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();

    let mut report = Report {
        archive: args.archive.display().to_string(),
        ..Default::default()
    };

    let Some(manifest_path) = packed_manifest_path(&args.archive) else {
        anyhow::bail!(
            "{} is not a packed STT dataset (expected a directory containing \
             manifest.json, or the manifest.json itself). Single-file `.stt` \
             archives are an internal build intermediate and are not \
             validated — rebuild with `stt-build` (packed output is the default)",
            args.archive.display()
        );
    };
    let metadata = validate_packed(&manifest_path, &args, &mut report)?;

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
    report.version = manifest.directory.directory_version;
    report.compression = manifest.compression.clone();

    verify_paged_directly(manifest_path, &manifest, args, report)?;

    let metadata = reader.metadata().clone();
    let entries = reader.entries().to_vec();
    report.tile_count = entries.len();

    let pb = make_progress(args, entries.len());
    validate_entries(
        &entries,
        &metadata,
        |e| reader.read_payload(e).map_err(|err| err.to_string()),
        args,
        report,
        pb.as_ref(),
    )?;
    if let Some(p) = &pb {
        p.finish_and_clear();
    }

    finalize_feature_check(args, report, &metadata);
    Ok(metadata)
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
        push_unique(report, "integrity: paged directory: manifest missing rootLength".into())?;
        return Ok(());
    };
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    // A missing/unreadable directory object is already reported by the
    // content-address pass; only the structural check is re-asserted here.
    let Ok(dir_bytes) = std::fs::read(root.join(&manifest.directory.key)) else {
        return Ok(());
    };
    let zstd = manifest.directory.encoding.as_deref()
        == Some(stt_core::pack::DIRECTORY_ENCODING_ZSTD);
    match stt_core::verify_paged_structure(&dir_bytes, root_length, zstd) {
        Ok(issues) => {
            for issue in issues {
                push_unique(report, format!("integrity: {issue}"))?;
            }
        }
        Err(e) => push_unique(report, format!("integrity: paged structure check failed: {e}"))?,
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
fn validate_entries(
    entries: &[TileEntry],
    metadata: &Metadata,
    mut read_payload: impl FnMut(&TileEntry) -> std::result::Result<Vec<u8>, String>,
    args: &Args,
    report: &mut Report,
    pb: Option<&ProgressBar>,
) -> Result<()> {
    let meta_start = metadata.time_range.start as i64;
    let meta_end = metadata.time_range.end as i64;

    report.sampled = args.sample.is_some() && !args.skip_decode;
    let stride = args.sample.map(|n| sample_stride(entries.len(), n));

    // Distinct schema signatures seen across decoded layers, for producer-drift
    // detection. We keep the first tile that exhibited each signature so the
    // error can name a concrete disagreeing pair.
    let mut schemas: BTreeSet<String> = BTreeSet::new();
    let mut first_schema_example: Option<(String, String)> = None;

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
            match stt_core::arrow_tile::decode_tile(&payload) {
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

                    // Schema / column-type contract per layer.
                    for issue in check_tile_schema(&layers) {
                        push_err(
                            report,
                            args.fail_fast,
                            format!("tile {:?} schema: {issue}", entry.tile_id()),
                        )?;
                    }

                    // Producer-drift detection: track the distinct schema
                    // signatures and flag the first tile that disagrees with
                    // an earlier one.
                    let sig = schema_signature(&layers);
                    if schemas.insert(sig.clone()) {
                        match &first_schema_example {
                            None => first_schema_example = Some((format!("{:?}", entry.tile_id()), sig)),
                            Some((first_tile, first_sig)) => {
                                push_err(
                                    report,
                                    args.fail_fast,
                                    format!(
                                        "schema drift: tile {:?} layer schema differs from tile {first_tile}\n  {first_tile}: {first_sig}\n  {:?}: {sig}",
                                        entry.tile_id(),
                                        entry.tile_id()
                                    ),
                                )?;
                            }
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

    report.distinct_schemas = schemas.len();
    report.feature_count_decoded_complete = decode_is_complete(args);
    Ok(())
}

fn finalize_feature_check(args: &Args, report: &mut Report, metadata: &Metadata) {
    // Only meaningful when EVERY tile was decoded — a sampled or skipped decode
    // sums a subset, so comparing it to the metadata grand total would spuriously
    // fail. Such runs leave `feature_count_decoded_complete = false` and skip it.
    if decode_is_complete(args)
        && report.feature_count_decoded != metadata.feature_count
        && metadata.feature_count != 0
    {
        report.errors.push(format!(
            "metadata feature_count={} disagrees with decoded sum {}",
            metadata.feature_count, report.feature_count_decoded
        ));
    }
    // tile_count needs no decode at all — the directory entry list is the
    // ground truth. A zero is tolerated: pre-0.1.1 writers never set the
    // manifest totals.
    if metadata.tile_count != 0 && metadata.tile_count != report.tile_count as u64 {
        report.errors.push(format!(
            "metadata tile_count={} disagrees with directory entry count {}",
            metadata.tile_count, report.tile_count
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

fn print_summary(report: &Report, metadata: &Metadata) {
    println!("dataset          {}", report.archive);
    println!("version          {}", report.version);
    println!("compression      {}", report.compression);
    println!("name             {}", metadata.name);
    println!("tiles            {}", report.tile_count);
    if report.sampled {
        println!(
            "decoded          {} of {} tiles (SAMPLED — not a full verification)",
            report.tiles_decoded, report.tile_count
        );
    } else {
        println!("decoded          {} of {} tiles", report.tiles_decoded, report.tile_count);
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
            if report.sampled { "sampled" } else { "decode skipped" }
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
    println!("zoom range       {} .. {}", metadata.min_zoom, metadata.max_zoom);
    println!("elapsed          {} ms", report.elapsed_ms);
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

    /// Doc gate (naming-types-consistency F9): every visible long flag must
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
        let start = doc.find("## `stt-validate`").expect("stt-validate section heading");
        let body = &doc[start + 1..];
        let end = body.find("\n## `").map(|i| start + 1 + i).unwrap_or(doc.len());
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
}
