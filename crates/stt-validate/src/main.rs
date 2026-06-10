//! stt-validate — inspect and verify an STT dataset.
//!
//! Accepts either the canonical **packed format** (a dataset directory or its
//! `manifest.json`) or a legacy single-file `.stt` archive. For packed inputs
//! it additionally verifies the content-addressing contract.
//!
//! Checks performed:
//! 1. (packed) Every pack and the directory object blake3-hash to the name the
//!    manifest gave them, on-disk lengths match, and the directory references
//!    no out-of-range `pack_id`.
//! 2. Index decodes; every entry has the columns the schema promises.
//! 3. Every tile blob round-trips its content hash and decompresses to its
//!    declared uncompressed size (enforced by `read_payload`).
//! 4. Every payload decodes as a layer frame of Arrow IPC streams.
//! 5. Feature counts in tile entries match the decoded layer rows.
//! 6. Tile temporal extents lie inside the dataset's metadata time range.
//!
//! Exits non-zero on any failure. With `--json` emits a machine-readable
//! report; without it emits a short human summary.

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use stt_core::archive::TileEntry;
use stt_core::metadata::Metadata;
use stt_core::{Archive, Manifest, PackedReader};

#[derive(Parser)]
#[command(name = "stt-validate", version, about = "Validate an STT dataset")]
struct Args {
    /// Dataset to validate: a packed dataset directory, its `manifest.json`,
    /// or a legacy single-file `.stt` archive.
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
}

#[derive(Serialize, Default)]
struct Report {
    archive: String,
    /// Packed: directory codec version (5). Single-file: archive header version.
    version: u8,
    compression: String,
    tile_count: usize,
    feature_count_index: u64,
    feature_count_decoded: u64,
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

    let metadata = match packed_manifest_path(&args.archive) {
        Some(manifest_path) => validate_packed(&manifest_path, &args, &mut report)?,
        None => validate_single_file(&args, &mut report)?,
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

/// If `input` denotes a packed dataset, return its `manifest.json` path;
/// otherwise `None` (treat as a legacy single-file `.stt`).
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

/// Validate a legacy single-file `.stt` archive.
fn validate_single_file(args: &Args, report: &mut Report) -> Result<Metadata> {
    let reader = Archive::open(&args.archive)
        .with_context(|| format!("failed to open archive {}", args.archive.display()))?;
    report.version = reader.header().version;
    report.compression = format!("{:?}", reader.header().compression);

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

/// Per-tile validation shared by both readers. `read_payload` reads (and CRC-
/// and size-checks) one tile's decompressed bytes from whichever container.
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

    for entry in entries {
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

        // read_payload does the content-hash and uncompressed-size checks.
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

        if !args.skip_decode {
            match stt_core::arrow_tile::decode_tile(&payload) {
                Ok(layers) => {
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

    Ok(())
}

fn finalize_feature_check(args: &Args, report: &mut Report, metadata: &Metadata) {
    if !args.skip_decode
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

fn print_summary(report: &Report, metadata: &Metadata) {
    println!("dataset          {}", report.archive);
    println!("version          {}", report.version);
    println!("compression      {}", report.compression);
    println!("name             {}", metadata.name);
    println!("tiles            {}", report.tile_count);
    println!(
        "features (index) {}  (decoded sum: {})",
        report.feature_count_index, report.feature_count_decoded
    );
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
