//! stt-optimize CLI — a thin wrapper around the library in `lib.rs`.
//!
//! # The two input-analysing subcommands run the ITERATED advisor pass
//!
//! `analyze` and `recommend` both go through
//! [`stt_optimize::advisors::run_iterative`] (via [`stt_optimize::recommend_for`]
//! for `recommend`), not the round-0 [`stt_optimize::advisors::run_all`] pass.
//! That matters because the single pass prices every lever against the SAME
//! build-default baseline and never measures the recipe those answers compose
//! into — so it can sell a LOSSY lever on a shrink that evaporates once the
//! recipe's own zstd level is applied, and hide one that only pays there.
//! Routing the binary through `run_all` would have left that defect in front of
//! every human and every MCP client while the fix sat in the library.
//!
//! See [`CLI_REFINEMENT_ROUNDS`] for the wall-time this costs and why both
//! subcommands pay it.

use stt_optimize::{
    advisors, analysis, budget_solver, diff, doctor, export, loader, measure, order_audit,
    recommend, report, PackedTileset,
};

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Coordinate-descent refinement rounds `analyze` and `recommend` run on top of
/// the round-0 advisor pass.
///
/// # Why both subcommands, and not just `recommend`
///
/// A DECISION, recorded here rather than left implicit. The refinement's whole
/// point is that a single-pass number can be wrong in a way that costs data
/// quality: `--quantize-coords` measured at the build-default zstd 3 can read as
/// a 8.8% win and be worth ~1% on the level-19 recipe the same run recommends.
/// `analyze` is the surface where a *human* reads that number and decides
/// whether to accept a lossy lever. Correcting it in `recommend` (the machine
/// path) while leaving the stale figure in `analyze` (the human path) would put
/// the misleading number in front of the reader least able to check it, and
/// would make the two subcommands disagree about the same dataset.
///
/// # What it costs — measured, not budgeted
///
/// The implementation plan budgeted "~5-10x more oracle calls" and asked for the
/// cost to be surfaced in `--explain`. It is surfaced (see
/// `print_composed_evidence`), and here is the measurement it is surfaced
/// against — `stt-optimize recommend`, release build, best of 3 (best of 2 on
/// the last row), on real inputs rather than a micro-fixture:
///
/// ```text
/// input                                single pass   iterated   ratio   adder
/// wpc-fronts      (4.2k rows, lines)       0.49 s      4.03 s    8.2x   +3.5 s
/// osm-changesets  (380k rows, points)      1.94 s      5.70 s    2.9x   +3.8 s
/// cmorph 6h       (230k rows, polygons)    2.86 s      8.78 s    3.1x   +5.9 s
/// cmorph 2h       (659k rows, polygons)    5.88 s     12.41 s    2.1x   +6.5 s
/// ```
///
/// Read the ADDER column, not the ratio. The last two rows are the same data at
/// two temporal resolutions: 2.9x the rows moves the added cost by 10%
/// (5.9 s -> 6.5 s) while the ratio falls from 3.1x to 2.1x. That is structural
/// rather than lucky — the added work is a fixed number of encodes of a sample
/// capped at 5000 features, so it is O(1) in dataset size, while the
/// load/parse/scan work it is divided by is O(rows).
///
/// So the ratio only looks alarming where the denominator is tiny. The
/// adversarial review's 11.8x is this same adder over a 600-row debug-build
/// fixture, the most flattering denominator available; the worst ratio a real
/// input produces here is 8.2x, on a 4.2k-row file where the absolute cost is
/// 4 s. Both exceed the plan's 5-10x band at the small end and both sit
/// comfortably inside it — 2-3x — at real scale. Reported rather than smoothed
/// over, because the plan asked for the number and the number is not the one the
/// plan guessed.
///
/// Judged as seconds: 3.5-6.5 s added to a once-per-dataset planning command
/// whose output drives a build measured in minutes to hours. That buys the
/// difference between a measured recipe and a guessed one, so both subcommands
/// pay it.
///
/// # Rollback
///
/// Setting this to `0` restores the historical single pass exactly — that
/// equivalence is pinned by tests in `advisors/mod.rs` and
/// `stt-optimize/tests/recommend_determinism.rs`, not by comment. No CLI flag
/// exposes it: a user-visible flag must be documented in
/// `docs/api/cli-reference.md` (the doc gate at the bottom of this file
/// enforces that), and that file is outside this change's ownership.
const CLI_REFINEMENT_ROUNDS: usize = advisors::RECOMMEND_ROUNDS;

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

        /// Solve the recipe for a target archive size: bytes, or a
        /// K/M/G (binary) or KB/MB/GB (decimal) suffix — `250MiB`, `1.5G`,
        /// `262144000`.
        ///
        /// Searches only REVERSIBLE levers (zoom clamp, temporal bucket width,
        /// temporal-LOD tiers, zstd level, blob ordering, pack size). Lossy
        /// levers are never applied: they are priced as shadow prices for a
        /// human to choose. An unreachable budget reports the lexicographic
        /// floor and drops nothing.
        #[arg(long, value_name = "SIZE")]
        target_size: Option<String>,

        /// Exit non-zero after printing if the solved recipe's projected size
        /// still exceeds `--target-size` (CI gate; the `diff
        /// --fail-on-growth` analog).
        #[arg(long)]
        fail_if_over_target: bool,
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

        /// DT-5 (§2.2): report interval read-amplification over a sliding
        /// window of this width (e.g. `1h`, `30m`).
        ///
        /// Directory-only walk — deterministic, no decode. Reports the share of
        /// expected fetched bytes attributable to tiles kept resident SOLELY by
        /// a long-lived `time_end`, plus a residual-lifetime histogram. This is
        /// the trigger instrument for the interval-segregation erratum: no
        /// number, no erratum.
        #[arg(long, value_name = "WINDOW")]
        read_amp: Option<String>,
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
    /// range-read cost (scrub + pan + playback, over the directory) and
    /// recommend `--blob-ordering`
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

        /// Query weighting to rank under: "derived" (default; per-dataset
        /// weights from the archive's layer hint + bucket count, matching what
        /// a `--blob-ordering measured` build does today) or "legacy" (the
        /// pre-2026-08 scrub+pan weighting). Running both is how a fleet
        /// re-audit shows which picks the workload model actually corrects.
        #[arg(long, default_value = "derived")]
        ordering_workload: String,
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
            target_size,
            fail_if_over_target,
        } => {
            tracing_subscriber::fmt::init();
            run_recommend(
                &input,
                &time_field,
                &time_format,
                output,
                show_command,
                explain,
                target_size.as_deref(),
                fail_if_over_target,
            )
        }
        Commands::Inspect {
            archive,
            sample,
            format,
            output,
            read_amp,
        } => run_inspect(&archive, sample, &format, output, read_amp.as_deref()),
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
            ordering_workload,
        } => run_order_audit(&archive, &format, output, strict, &ordering_workload),
    }
}

fn run_analyze(
    input: PathBuf,
    time_field: &str,
    time_format: &str,
    format: &str,
    output: Option<PathBuf>,
) -> Result<()> {
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

    // Load and run every analysis, exactly as the library's own entry point
    // does — the measured sample encoding calibrates the size estimates and is
    // `None` when the sample is too small. Sharing `analyze_source` rather than
    // re-assembling the pipeline here is what keeps `analyze`, `recommend`, and
    // `stt-build --auto` looking at the same numbers.
    let (result, data) = stt_optimize::analyze_source(&source)?;

    // Generate recommendations from the ITERATED advisor pass: round 0, then
    // the composed recipe measured and every measured lever re-priced against
    // it. The report carries the outcome in its Advisor section / `advice`
    // array, and the composed figure in `explanations`.
    let layout = measure::SyntheticLayout::from_density(&result.density);
    let iterative = advisors::run_iterative(&result, &data, &layout, CLI_REFINEMENT_ROUNDS)?;
    // The composed figure and what it cost, on stderr with the banner: the text
    // report's rendering belongs to `report.rs`, and stdout must stay a pure
    // report. `--format json` carries the same line inside `explanations`.
    if let Some(composed) = &iterative.composed {
        eprintln!("Composed recipe: {}", composed.projected());
        if let Some(lossy) = composed.projected_with_lossy() {
            eprintln!("Lossy what-if (not applied): {lossy}");
        }
        eprintln!();
    }
    let recommendations = recommend::generate_recommendations_composed(
        &result,
        iterative.advice,
        iterative.composed.as_ref(),
    );

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

#[allow(clippy::too_many_arguments)]
fn run_recommend(
    input: &PathBuf,
    time_field: &str,
    time_format: &str,
    output: Option<PathBuf>,
    show_command: bool,
    explain: bool,
    target_size: Option<&str>,
    fail_if_over_target: bool,
) -> Result<()> {
    use loader::DataSource;

    let source = DataSource::GeoParquet {
        path: input.clone(),
        time_field: time_field.to_string(),
        time_format: time_format.to_string(),
    };
    // The gate is only meaningful against a target, and silently passing a
    // gate that can never fire is worse than refusing to run.
    if fail_if_over_target && target_size.is_none() {
        anyhow::bail!("--fail-if-over-target needs a --target-size to gate against");
    }
    // Parse BEFORE the analysis: a typo'd size should fail in milliseconds, not
    // after a multi-second measurement pass.
    let target_bytes = target_size.map(budget_solver::parse_size).transpose()?;

    // Progress line on stderr — stdout stays pure config JSON when piped.
    eprintln!("Analyzing dataset for optimal parameters...\n");

    let recommendations = match target_bytes {
        // THE recipe entry point, shared verbatim with `stt-build --auto` and
        // with the MCP `recommend_build` tool (which shells out to this
        // subcommand): load, analyse, run the advisors
        // [`CLI_REFINEMENT_ROUNDS`]-times-refined, and attach the composed
        // measurement. Calling the library function rather than re-assembling
        // its steps here is what guarantees the CLI can never drift from
        // `--auto` again.
        // (`recommend_for` is exactly this call at `advisors::RECOMMEND_ROUNDS`
        // — the form `stt-build --auto` uses. Naming the round count here keeps
        // the rollback a one-constant edit.)
        None => stt_optimize::recommend_for_with_rounds(&source, CLI_REFINEMENT_ROUNDS)?,
        // Budget mode runs the SAME pipeline and then solves on top of it. It is
        // spelled out here rather than hidden behind a second library entry
        // point because the solver needs the analysis and the loaded data, which
        // `recommend_for_with_rounds` consumes internally; the `stt-build
        // --target-size` handshake (MO-9) is what introduces the library-side
        // `recommend_with(source, opts)` wrapper.
        Some(target_bytes) => {
            let (result, data) = stt_optimize::analyze_source(&source)?;
            let layout = measure::SyntheticLayout::from_density(&result.density);
            let iterative =
                advisors::run_iterative(&result, &data, &layout, CLI_REFINEMENT_ROUNDS)?;
            let budget = budget_solver::solve(&result, &data, &layout, target_bytes)?;
            recommend::generate_recommendations_budgeted(
                &result,
                iterative.advice,
                iterative.composed.as_ref(),
                Some(budget),
            )
        }
    };

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
        print_composed_evidence(&recommendations);
    }

    // The budget table is the answer to `--target-size`, so it prints whenever
    // one was asked for — `--explain` is about the advisor evidence, not this.
    if let Some(report) = &recommendations.budget {
        println!("{}", budget_solver::format_text(report));
    }

    if show_command {
        println!("\nSuggested stt-build command:");
        println!(
            "{}",
            recommend::to_command(&recommendations, input, time_field)
        );
    }

    // Print first, exit after — the `diff --fail-on-growth` pattern, so a
    // failing gate still leaves the operator the full report.
    if fail_if_over_target {
        let report = recommendations
            .budget
            .as_ref()
            .expect("--fail-if-over-target implies --target-size");
        if !report.feasible {
            eprintln!(
                "FAIL: --fail-if-over-target gate — the smallest recipe the REVERSIBLE levers can \
                 reach projects {} B, over the {} B target by {} B. Nothing was dropped to get \
                 closer; see the shadow-price table for what a lossy lever would buy.",
                report.projected_bytes,
                report.target_bytes,
                report.projected_bytes.saturating_sub(report.target_bytes)
            );
            std::process::exit(1);
        }
        eprintln!(
            "--fail-if-over-target gate OK: {} B <= {} B",
            report.projected_bytes, report.target_bytes
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
        if a.suggestion_only {
            // Non-lossy, but still not auto-applied: the `why` carries a
            // decision only a human can make (e.g. a `spatial` blob ordering
            // that silently breaks time-playback buffering). Marking it is what
            // stops a reader assuming everything unmarked is in the command.
            projected.push_str("  [SUGGESTION - needs a decision]");
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

/// Print the composed-recipe block that follows the `--explain` evidence table.
///
/// The table above it prices levers ONE AT A TIME. This block is the only place
/// the user sees what the whole recipe measures — and what measuring it cost,
/// which the implementation plan requires be surfaced here rather than left as a
/// silent slowdown (see [`CLI_REFINEMENT_ROUNDS`] for the wall-clock figures).
///
/// The with-lossy line is a what-if and says so: lossy levers are never in the
/// recipe, never in the suggested command, and never auto-applied.
fn print_composed_evidence(recommendations: &recommend::Recommendations) {
    println!("\nComposed recipe (measured):");
    let Some(line) = &recommendations.composed_projected else {
        // No composed figure means no measurement happened — the sample sat
        // below the measurement floor. Say that, rather than printing nothing
        // and letting the table above look more measured than it is.
        println!(
            "  (not measured — this sample is below the measurement floor, so every\n   \
             projection above is the single-pass estimate)"
        );
        return;
    };
    println!("  {line}");
    if let Some(lossy) = &recommendations.composed_projected_with_lossy {
        println!("  {lossy}");
    }
    println!(
        "  Cost: refinement re-encodes the sample once per distinct settings point (the\n   \
         encode/trial/cache counts above). Measured at +3.5-6.5s wall clock on real\n   \
         inputs; bounded by the 5000-feature sample cap and the fixed round count, so it\n   \
         does NOT grow with the dataset — 2.9x the rows moved it 10%."
    );
}

fn run_inspect(
    archive: &PathBuf,
    sample: Option<usize>,
    format: &str,
    output: Option<PathBuf>,
    read_amp: Option<&str>,
) -> Result<()> {
    let tileset = PackedTileset::open(archive)?;

    // DT-5 (§2.2): the interval read-amplification trigger instrument. A
    // directory-only walk, so it runs in the fast `--sample 0` class and never
    // decodes a payload.
    if let Some(window) = read_amp {
        let window_ms = stt_build::build_options::parse_duration(window)?;
        let bucket_ms = tileset.metadata().temporal_bucket_ms;
        let spans: Vec<stt_optimize::read_amp::EntrySpan> = tileset
            .entries()
            .iter()
            .map(|e| stt_optimize::read_amp::EntrySpan {
                time_start: e.time_start,
                time_end: e.time_end,
                cover_t_min: e.cover_t_min,
                length: e.length as u64,
            })
            .collect();
        let amp = stt_optimize::read_amp::read_amplification(&spans, window_ms, bucket_ms);
        let rendered = match format {
            "json" => serde_json::to_string_pretty(&amp)?,
            _ => format_read_amp(&amp),
        };
        return write_or_print(&rendered, output);
    }

    let report = analysis::inspect::inspect(&tileset, sample)?;

    let rendered = match format {
        "json" => serde_json::to_string_pretty(&report)?,
        _ => analysis::inspect::format_text(&report),
    };
    write_or_print(&rendered, output)
}

/// Render the DT-5 read-amplification report.
fn format_read_amp(r: &stt_optimize::read_amp::ReadAmpReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "Interval read amplification (DT-5 / §2.2)");
    let _ = writeln!(out, "  window            {} ms", r.window_ms);
    let _ = writeln!(out, "  windows swept     {}", r.windows);
    let _ = writeln!(out, "  fetched bytes     {}", r.fetched_bytes);
    let _ = writeln!(out, "  long-lived bytes  {}", r.long_lived_bytes);
    let _ = writeln!(
        out,
        "  amplification     {:.1}%  (trigger: {:.0}%)",
        r.amplification_share * 100.0,
        stt_optimize::read_amp::READ_AMP_TRIGGER_SHARE * 100.0
    );
    let fires = r.amplification_share >= stt_optimize::read_amp::READ_AMP_TRIGGER_SHARE;
    let _ = writeln!(
        out,
        "  verdict           {}",
        if fires {
            "TRIGGERED — the §2.2 erratum condition holds on this archive"
        } else {
            "not triggered — no erratum is drafted (no number, no erratum)"
        }
    );
    let _ = writeln!(
        out,
        "  residual lifetime (windows past the entry's own bucket):"
    );
    for (i, n) in r.residual_lifetime_bins.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        let label = if i + 1 == r.residual_lifetime_bins.len() {
            format!("{i}+")
        } else {
            i.to_string()
        };
        let _ = writeln!(out, "    {label:>3} : {n}");
    }
    out
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
    ordering_workload: &str,
) -> Result<()> {
    use stt_core::ordering_sim::OrderingWorkloadMode;
    let workload = match ordering_workload.trim().to_lowercase().as_str() {
        "derived" | "auto" => OrderingWorkloadMode::Derived,
        "legacy" => OrderingWorkloadMode::Legacy,
        other => anyhow::bail!(
            "Invalid --ordering-workload '{other}'. Expected 'derived' (default) or 'legacy'."
        ),
    };
    let tileset = PackedTileset::open(archive)?;
    let report =
        order_audit::order_audit_with(&tileset, order_audit::OrderAuditOptions { workload })?;

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

    /// Doc gate: every visible long flag on every subcommand must appear in the
    /// `stt-optimize` section of `docs/api/cli-reference.md`, so a new flag
    /// fails the build until it is documented.
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
