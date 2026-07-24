//! stt-optimize CLI — a thin wrapper around the library in `lib.rs`.

use stt_optimize::{
    advisors, analysis, diff, doctor, export, loader, measure, order_audit, recommend, report,
    PackedTileset,
};

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "stt-optimize")]
#[command(about = "Analyze and optimize spatiotemporal datasets for STT generation", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze a dataset and generate optimization report
    Analyze {
        /// Input GeoParquet file path
        #[arg(short, long)]
        input: PathBuf,

        /// Field name containing timestamps
        #[arg(short, long, default_value = "timestamp")]
        time_field: String,

        /// Time format: "unix-ms", "unix-sec", or "iso8601"
        #[arg(long, default_value = "iso8601")]
        time_format: String,

        /// Output format: "text" or "json"
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Generate optimized stt-build configuration
    Recommend {
        /// Input GeoParquet file path
        #[arg(short, long)]
        input: PathBuf,

        /// Field name containing timestamps
        #[arg(short, long, default_value = "timestamp")]
        time_field: String,

        /// Time format: "unix-ms", "unix-sec", or "iso8601"
        #[arg(long, default_value = "iso8601")]
        time_format: String,

        /// Output JSON config file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Show suggested stt-build command
        #[arg(long)]
        show_command: bool,

        /// Print an evidence table of every advisor suggestion after the
        /// config JSON — flag, value, confidence, projected effect, and the
        /// dataset-specific rationale. Includes LOSSY levers (marked); those
        /// never join the suggested command and are never auto-applied.
        #[arg(long)]
        explain: bool,
    },

    /// Inspect a built packed tileset: per-zoom directory stats, dedup and
    /// compression ratios, per-column compressed cost
    Inspect {
        /// Packed dataset directory (or its manifest.json)
        #[arg(short, long)]
        archive: PathBuf,

        /// Decode only a deterministic, evenly-spread sample of at most N
        /// tiles (0 skips the decode pass). Directory-derived stats always
        /// cover every entry.
        #[arg(long, value_name = "N")]
        sample: Option<usize>,

        /// Output format: "text" or "json"
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compare two built tilesets (totals, per-zoom, per-column deltas) —
    /// e.g. before/after a re-encode
    Diff {
        /// Baseline packed dataset directory (or its manifest.json)
        #[arg(long)]
        before: PathBuf,

        /// Comparison packed dataset directory (or its manifest.json)
        #[arg(long)]
        after: PathBuf,

        /// Sample the decode pass on both sides (as `inspect --sample`);
        /// totals and the growth gate stay exact.
        #[arg(long, value_name = "N")]
        sample: Option<usize>,

        /// Output format: "text" or "json"
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Exit non-zero after printing if `after` total compressed blob
        /// bytes exceed `before` by more than this percentage
        #[arg(long, value_name = "PCT")]
        fail_on_growth: Option<f64>,
    },

    /// Lint a built packed tileset: severity-ranked findings, each with a
    /// concrete remediation flag and a projected win from measured column
    /// costs
    Doctor {
        /// Packed dataset directory (or its manifest.json)
        #[arg(short, long)]
        archive: PathBuf,

        /// Sample the inspect decode pass (as `inspect --sample`); the
        /// directory-derived rules always cover every entry
        #[arg(long, value_name = "N")]
        sample: Option<usize>,

        /// Output format: "text" or "json"
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Exit non-zero after printing if any Warning-or-worse finding
        /// exists (CI gate; Info findings never trip it)
        #[arg(long)]
        strict: bool,
    },

    /// Export a built packed tileset back out as GeoParquet — whole archive,
    /// or a bbox / time-range subset
    Export {
        /// Packed dataset directory (or its manifest.json)
        #[arg(short, long)]
        archive: PathBuf,

        /// Output .parquet path. With several layers and no `--layer` this is
        /// the stem: each layer lands in `<stem>.<layer>.parquet`.
        #[arg(short, long)]
        output: PathBuf,

        /// Zoom level to export (default: the deepest one present). One export
        /// is always ONE zoom — the same feature is re-tiled at every level
        /// with a different simplification tolerance.
        #[arg(long, value_name = "Z")]
        zoom: Option<u8>,

        /// Layer to export (default: every layer, one file each)
        #[arg(long, value_name = "NAME")]
        layer: Option<String>,

        /// Keep only features intersecting this box, as
        /// `min_lon,min_lat,max_lon,max_lat`
        ///
        /// `allow_hyphen_values`: a western/southern box starts with a minus
        /// sign, and clap would otherwise read `-73.9,...` as an unknown flag.
        #[arg(long, value_name = "MINX,MINY,MAXX,MAXY", allow_hyphen_values = true)]
        bbox: Option<String>,

        /// Keep only features whose time span reaches this instant or later
        /// (ISO-8601 or Unix ms)
        #[arg(long, value_name = "TIME", allow_hyphen_values = true)]
        start: Option<String>,

        /// Keep only features whose time span starts at this instant or
        /// earlier (ISO-8601 or Unix ms)
        #[arg(long, value_name = "TIME", allow_hyphen_values = true)]
        end: Option<String>,

        /// Geometry column typing: `wkb` (default, GeoParquet 1.1 — what every
        /// deployed reader understands) or `native` (adds Parquet's GEOMETRY
        /// logical type on the same bytes)
        #[arg(long, value_name = "ENC", default_value = "wkb")]
        geometry_encoding: String,

        /// Output format for the run report: "text" or "json"
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Audit blob ordering on a built packed tileset: measure per-ordering
    /// range-read cost (scrub + pan, over the directory) and recommend
    /// `--blob-ordering`
    OrderAudit {
        /// Packed dataset directory (or its manifest.json)
        #[arg(short, long)]
        archive: PathBuf,

        /// Output format: "text" or "json"
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Exit non-zero if the archive's recorded ordering isn't the measured
        /// recommendation (CI gate; passes when the ordering isn't recorded)
        #[arg(long)]
        strict: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze {
            input,
            time_field,
            time_format,
            format,
            output,
            verbose,
        } => {
            // Initialize logging
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(if verbose {
                    tracing::Level::DEBUG
                } else {
                    tracing::Level::INFO
                })
                .finish();
            tracing::subscriber::set_global_default(subscriber).ok();

            run_analyze(input, &time_field, &time_format, &format, output)
        }
        Commands::Recommend {
            input,
            time_field,
            time_format,
            output,
            show_command,
            explain,
        } => {
            tracing_subscriber::fmt::init();
            run_recommend(
                &input,
                &time_field,
                &time_format,
                output,
                show_command,
                explain,
            )
        }
        Commands::Inspect {
            archive,
            sample,
            format,
            output,
        } => run_inspect(&archive, sample, &format, output),
        Commands::Diff {
            before,
            after,
            sample,
            format,
            output,
            fail_on_growth,
        } => run_diff(&before, &after, sample, &format, output, fail_on_growth),
        Commands::Doctor {
            archive,
            sample,
            format,
            output,
            strict,
        } => run_doctor(&archive, sample, &format, output, strict),
        Commands::Export {
            archive,
            output,
            zoom,
            layer,
            bbox,
            start,
            end,
            geometry_encoding,
            format,
        } => run_export(
            &archive,
            &output,
            zoom,
            layer,
            bbox.as_deref(),
            start.as_deref(),
            end.as_deref(),
            &geometry_encoding,
            &format,
        ),
        Commands::OrderAudit {
            archive,
            format,
            output,
            strict,
        } => run_order_audit(&archive, &format, output, strict),
    }
}

fn run_analyze(
    input: PathBuf,
    time_field: &str,
    time_format: &str,
    format: &str,
    output: Option<PathBuf>,
) -> Result<()> {
    use analysis::AnalysisResult;
    use loader::DataSource;

    let source = DataSource::GeoParquet {
        path: input,
        time_field: time_field.to_string(),
        time_format: time_format.to_string(),
    };

    // Progress banner on stderr — stdout stays pure report so
    // `--format json` output can be piped straight into jq/a file.
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    eprintln!("         STT Optimization Analysis");
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Load and analyze data
    let data = loader::load_data(&source)?;

    // Run all analyses (measured sample encoding calibrates the size
    // estimates; None when the sample is too small).
    let spatial = analysis::spatial::analyze(&data)?;
    let temporal = analysis::temporal::analyze(&data)?;
    let geometry = analysis::geometry::analyze(&data)?;
    let measured = measure::measure_sample(&data.sample, &measure::MeasureSettings::default())?;
    let density = analysis::density::analyze(&data, &spatial, &temporal, measured.as_ref())?;

    let result = AnalysisResult {
        source: source.display_name(),
        feature_count: data.features.len(),
        bounds: data.bounds,
        spatial,
        temporal,
        geometry,
        density,
        measured,
    };

    // Generate recommendations (with the evidence-based advisor suggestions —
    // the report carries them in its Advisor section / `advice` array).
    let advice = advisors::run_all(&result, &data)?;
    let recommendations = recommend::generate_recommendations(&result, advice);

    // Generate report
    let report_output = match format {
        "json" => report::generate_json(&result, &recommendations)?,
        _ => report::generate_text(&result, &recommendations),
    };

    // Output report
    if let Some(output_path) = output {
        std::fs::write(&output_path, &report_output)?;
        eprintln!("Report written to: {}", output_path.display());
    } else {
        println!("{}", report_output);
    }

    Ok(())
}

fn run_recommend(
    input: &PathBuf,
    time_field: &str,
    time_format: &str,
    output: Option<PathBuf>,
    show_command: bool,
    explain: bool,
) -> Result<()> {
    use loader::DataSource;

    let source = DataSource::GeoParquet {
        path: input.clone(),
        time_field: time_field.to_string(),
        time_format: time_format.to_string(),
    };

    // Progress line on stderr — stdout stays pure config JSON when piped.
    eprintln!("Analyzing dataset for optimal parameters...\n");

    // Load and analyze data
    let data = loader::load_data(&source)?;

    // Run analyses
    let spatial = analysis::spatial::analyze(&data)?;
    let temporal = analysis::temporal::analyze(&data)?;
    let geometry = analysis::geometry::analyze(&data)?;
    let measured = measure::measure_sample(&data.sample, &measure::MeasureSettings::default())?;
    let density = analysis::density::analyze(&data, &spatial, &temporal, measured.as_ref())?;

    let result = analysis::AnalysisResult {
        source: source.display_name(),
        feature_count: data.features.len(),
        bounds: data.bounds,
        spatial,
        temporal,
        geometry,
        density,
        measured,
    };

    // Generate recommendations (with the evidence-based advisor suggestions)
    let advice = advisors::run_all(&result, &data)?;
    let recommendations = recommend::generate_recommendations(&result, advice);

    // Output as JSON config
    let config = recommend::to_build_config(&recommendations, input, time_field);
    let json = serde_json::to_string_pretty(&config)?;

    if let Some(output_path) = output {
        std::fs::write(&output_path, &json)?;
        eprintln!("Build config written to: {}", output_path.display());
    } else {
        println!("{}", json);
    }

    if explain {
        print_advice_table(&recommendations.advice);
    }

    if show_command {
        println!("\nSuggested stt-build command:");
        println!(
            "{}",
            recommend::to_command(&recommendations, input, time_field)
        );
    }

    Ok(())
}

/// Print the `recommend --explain` evidence table: every advisor suggestion
/// (including LOSSY levers, marked — those never join the suggested command
/// and are never auto-applied), each with its dataset-specific rationale on a
/// continuation line.
fn print_advice_table(advice: &[advisors::Advice]) {
    println!("\nAdvisor evidence:");
    if advice.is_empty() {
        println!("  (no advice — the defaults already fit this dataset)");
        return;
    }
    println!(
        "  {:<28} {:<12} {:<10} {}",
        "FLAG", "VALUE", "CONFIDENCE", "PROJECTED"
    );
    for a in advice {
        let value = a.value.as_deref().unwrap_or("—");
        let mut projected = a.projected.clone().unwrap_or_else(|| "—".to_string());
        if a.lossy {
            projected.push_str("  [LOSSY - opt-in]");
        }
        // `to_string()` first: width specs only apply through `str`'s
        // Display, not AdviceConfidence's `write_str`-based impl.
        println!(
            "  {:<28} {:<12} {:<10} {}",
            a.flag,
            value,
            a.confidence.to_string(),
            projected
        );
        println!("      → {}", a.why);
    }
}

fn run_inspect(
    archive: &PathBuf,
    sample: Option<usize>,
    format: &str,
    output: Option<PathBuf>,
) -> Result<()> {
    let tileset = PackedTileset::open(archive)?;
    let report = analysis::inspect::inspect(&tileset, sample)?;

    let rendered = match format {
        "json" => serde_json::to_string_pretty(&report)?,
        _ => analysis::inspect::format_text(&report),
    };
    write_or_print(&rendered, output)
}

fn run_diff(
    before: &PathBuf,
    after: &PathBuf,
    sample: Option<usize>,
    format: &str,
    output: Option<PathBuf>,
    fail_on_growth: Option<f64>,
) -> Result<()> {
    let before_report = analysis::inspect::inspect(&PackedTileset::open(before)?, sample)?;
    let after_report = analysis::inspect::inspect(&PackedTileset::open(after)?, sample)?;
    let report = diff::diff(&before_report, &after_report);

    let rendered = match format {
        "json" => serde_json::to_string_pretty(&report)?,
        _ => diff::format_text(&report),
    };
    write_or_print(&rendered, output)?;

    if let Some(threshold) = fail_on_growth {
        // Gate on the exact directory totals (unaffected by --sample). A zero
        // baseline with a non-empty `after` counts as unbounded growth.
        let d = &report.compressed_bytes;
        let growth = d
            .pct
            .unwrap_or(if d.after > 0 { f64::INFINITY } else { 0.0 });
        if growth > threshold {
            eprintln!(
                "FAIL: compressed blob bytes grew {growth:.2}% ({} -> {}), \
                 over the --fail-on-growth gate of {threshold}%",
                d.before, d.after
            );
            std::process::exit(1);
        }
        eprintln!("--fail-on-growth gate OK: {growth:.2}% <= {threshold}%");
    }
    Ok(())
}

fn run_doctor(
    archive: &PathBuf,
    sample: Option<usize>,
    format: &str,
    output: Option<PathBuf>,
    strict: bool,
) -> Result<()> {
    let tileset = PackedTileset::open(archive)?;
    let inspect_report = analysis::inspect::inspect(&tileset, sample)?;
    let report = doctor::doctor(&tileset, &inspect_report)?;

    let rendered = match format {
        "json" => serde_json::to_string_pretty(&report)?,
        _ => doctor::format_text(&report),
    };
    write_or_print(&rendered, output)?;

    if strict {
        // Gate on Warning-or-worse only (Critical sorts before Warning);
        // Info findings never trip CI. Print first, exit after — the
        // `diff --fail-on-growth` pattern.
        let bad = report
            .findings
            .iter()
            .filter(|f| f.severity <= doctor::Severity::Warning)
            .count();
        if bad > 0 {
            eprintln!("FAIL: --strict gate — {bad} Warning-or-worse finding(s)");
            std::process::exit(1);
        }
        eprintln!("--strict gate OK: no Warning-or-worse findings");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_export(
    archive: &PathBuf,
    output: &PathBuf,
    zoom: Option<u8>,
    layer: Option<String>,
    bbox: Option<&str>,
    start: Option<&str>,
    end: Option<&str>,
    geometry_encoding: &str,
    format: &str,
) -> Result<()> {
    let opts = export::ExportOptions {
        zoom,
        layer,
        bbox: bbox.map(export::parse_bbox).transpose()?,
        start: start.map(export::parse_time_bound).transpose()?,
        end: end.map(export::parse_time_bound).transpose()?,
        geometry_encoding: export::GeometryEncoding::parse(geometry_encoding)?,
    };
    let tileset = PackedTileset::open(archive)?;
    let report = export::export(&tileset, output, &opts)?;

    // The report goes to stdout (so `--format json` pipes into jq); the files
    // themselves are the actual output, already on disk.
    let rendered = match format {
        "json" => serde_json::to_string_pretty(&report)?,
        _ => export::format_text(&report),
    };
    println!("{}", rendered);
    Ok(())
}

fn run_order_audit(
    archive: &PathBuf,
    format: &str,
    output: Option<PathBuf>,
    strict: bool,
) -> Result<()> {
    let tileset = PackedTileset::open(archive)?;
    let report = order_audit::order_audit(&tileset)?;

    let rendered = match format {
        "json" => serde_json::to_string_pretty(&report)?,
        _ => order_audit::format_text(&report),
    };
    write_or_print(&rendered, output)?;

    if strict {
        // Fail only when the archive RECORDS an ordering that isn't the measured
        // recommendation; a legacy archive with no recorded ordering warns and
        // passes (it can't be re-derived without a rebuild).
        match &report.current {
            Some(cur) if *cur != report.recommended => {
                eprintln!(
                    "FAIL: --strict — current ordering '{cur}' != measured recommendation '{}'",
                    report.recommended
                );
                std::process::exit(1);
            }
            Some(_) => {
                eprintln!("--strict gate OK: current ordering is the measured recommendation")
            }
            None => eprintln!(
                "--strict gate: current ordering not recorded (pre-2026-07 archive) — passing"
            ),
        }
    }
    Ok(())
}

/// Write `rendered` to `output`, or print it to stdout. The status line goes
/// to stderr so piped stdout stays pure report.
fn write_or_print(rendered: &str, output: Option<PathBuf>) -> Result<()> {
    if let Some(path) = output {
        std::fs::write(&path, rendered)?;
        eprintln!("Report written to: {}", path.display());
    } else {
        println!("{}", rendered);
    }
    Ok(())
}

#[cfg(test)]
mod cli_doc_tests {
    use super::*;

    /// Doc gate (naming-types-consistency F9): every visible long flag on every
    /// subcommand must appear in the `stt-optimize` section of
    /// `docs/api/cli-reference.md`, so a new flag fails the build until it is
    /// documented.
    #[test]
    fn cli_flags_are_documented_in_cli_reference() {
        use clap::CommandFactory;
        let doc = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/api/cli-reference.md"
        ))
        .expect("read docs/api/cli-reference.md");
        let start = doc
            .find("## `stt-optimize`")
            .expect("stt-optimize section heading");
        let body = &doc[start + 1..];
        let end = body
            .find("\n## `")
            .map(|i| start + 1 + i)
            .unwrap_or(doc.len());
        let section = &doc[start..end];
        let cmd = Cli::command();
        let mut missing: Vec<String> = Vec::new();
        for sub in cmd.get_subcommands() {
            for a in sub.get_arguments() {
                if a.is_hide_set() {
                    continue;
                }
                let Some(l) = a.get_long() else { continue };
                if matches!(l, "help" | "version") {
                    continue;
                }
                let flag = format!("--{l}");
                if !section.contains(flag.as_str()) {
                    missing.push(format!("{} {flag}", sub.get_name()));
                }
            }
        }
        missing.sort();
        missing.dedup();
        assert!(
            missing.is_empty(),
            "flags missing from the `stt-optimize` section of docs/api/cli-reference.md \
             (document them before shipping): {missing:?}"
        );
    }
}
