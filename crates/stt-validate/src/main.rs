//! stt-validate — inspect and verify an STT archive.
//!
//! Checks performed:
//! 1. Header parses and version matches.
//! 2. Index decodes; every entry has the columns the schema promises.
//! 3. Every tile blob round-trips its content hash and decompresses to its
//!    declared uncompressed size (already enforced by `read_payload`).
//! 4. Every payload decodes as a layer frame of Arrow IPC streams.
//! 5. Feature counts in tile entries match the decoded layer rows.
//! 6. Tile temporal extents lie inside the archive's metadata time range.
//!
//! Exits non-zero on any failure. With `--json` emits a machine-readable
//! report; without it emits a short human summary.

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;
use stt_core::Archive;

#[derive(Parser)]
#[command(name = "stt-validate", version, about = "Validate an STT archive")]
struct Args {
    /// Archive path (.stt)
    archive: PathBuf,

    /// Emit a JSON report instead of a human summary.
    #[arg(long)]
    json: bool,

    /// Stop after the first failing tile. Default: keep going and report all.
    #[arg(long)]
    fail_fast: bool,

    /// Skip the per-tile decode step (only checks header + index + content hashes).
    #[arg(long)]
    skip_decode: bool,
}

#[derive(Serialize, Default)]
struct Report {
    archive: String,
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

    let mut reader = Archive::open(&args.archive)
        .with_context(|| format!("failed to open archive {}", args.archive.display()))?;

    report.version = reader.header().version;
    report.compression = format!("{:?}", reader.header().compression);
    let metadata = reader.metadata().clone();
    let entries: Vec<_> = reader.entries().to_vec();
    report.tile_count = entries.len();
    let archive_time_range = metadata.time_range;

    let pb = if args.json {
        None
    } else {
        let pb = ProgressBar::new(entries.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );
        Some(pb)
    };

    for entry in &entries {
        report.payload_bytes_compressed += entry.length as u64;
        report.payload_bytes_uncompressed += entry.uncompressed_size as u64;
        report.feature_count_index += entry.feature_count as u64;

        let meta_start = archive_time_range.start as i64;
        let meta_end = archive_time_range.end as i64;
        if entry.time_start < meta_start || entry.time_end > meta_end {
            push_err(
                &mut report,
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
        let payload = match reader.read_payload(entry) {
            Ok(b) => b,
            Err(e) => {
                push_err(
                    &mut report,
                    args.fail_fast,
                    format!("tile {:?} payload read failed: {e}", entry.tile_id()),
                )?;
                pb.as_ref().map(|p| p.inc(1));
                continue;
            }
        };

        if !args.skip_decode {
            match stt_core::arrow_tile::decode_tile(&payload) {
                Ok(layers) => {
                    let mut row_total: u64 = 0;
                    for layer in &layers {
                        row_total += layer.batch.num_rows() as u64;
                    }
                    report.feature_count_decoded += row_total;
                    if row_total != entry.feature_count as u64 {
                        push_err(
                            &mut report,
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
                        &mut report,
                        args.fail_fast,
                        format!("tile {:?} decode failed: {e}", entry.tile_id()),
                    )?;
                }
            }
        }

        pb.as_ref().map(|p| p.inc(1));
    }

    pb.as_ref().map(|p| p.finish_and_clear());

    if !args.skip_decode && report.feature_count_decoded != metadata.feature_count
        && metadata.feature_count != 0
    {
        report.errors.push(format!(
            "metadata feature_count={} disagrees with decoded sum {}",
            metadata.feature_count, report.feature_count_decoded
        ));
    }

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

fn push_err(report: &mut Report, fail_fast: bool, msg: String) -> Result<()> {
    eprintln!("error: {msg}");
    report.errors.push(msg);
    if fail_fast {
        anyhow::bail!("validation failed (--fail-fast)");
    }
    Ok(())
}

fn print_summary(report: &Report, metadata: &stt_core::metadata::Metadata) {
    println!("archive          {}", report.archive);
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
