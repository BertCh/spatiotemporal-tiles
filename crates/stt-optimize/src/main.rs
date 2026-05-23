//! stt-optimize CLI — a thin wrapper around the library in `lib.rs`.

use stt_optimize::{analysis, loader, recommend, report};

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
        #[arg(short, long, conflicts_with = "stt")]
        input: Option<PathBuf>,

        /// Input STT archive path (alternative to --input)
        #[arg(long, conflicts_with = "input")]
        stt: Option<PathBuf>,

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
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze {
            input,
            stt,
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

            run_analyze(input, stt, &time_field, &time_format, &format, output)
        }
        Commands::Recommend {
            input,
            time_field,
            time_format,
            output,
            show_command,
        } => {
            tracing_subscriber::fmt::init();
            run_recommend(&input, &time_field, &time_format, output, show_command)
        }
    }
}

fn run_analyze(
    input: Option<PathBuf>,
    stt: Option<PathBuf>,
    time_field: &str,
    time_format: &str,
    format: &str,
    output: Option<PathBuf>,
) -> Result<()> {
    use analysis::AnalysisResult;
    use loader::DataSource;

    // Determine input source
    let source = if let Some(path) = input {
        DataSource::GeoParquet {
            path,
            time_field: time_field.to_string(),
            time_format: time_format.to_string(),
        }
    } else if let Some(path) = stt {
        DataSource::SttArchive { path }
    } else {
        anyhow::bail!("Either --input or --stt must be provided");
    };

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("         STT Optimization Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Load and analyze data
    let data = loader::load_data(&source)?;
    
    // Run all analyses
    let spatial = analysis::spatial::analyze(&data)?;
    let temporal = analysis::temporal::analyze(&data)?;
    let geometry = analysis::geometry::analyze(&data)?;
    let density = analysis::density::analyze(&data, &spatial)?;

    let result = AnalysisResult {
        source: source.display_name(),
        feature_count: data.features.len(),
        spatial,
        temporal,
        geometry,
        density,
    };

    // Generate recommendations
    let recommendations = recommend::generate_recommendations(&result);

    // Generate report
    let report_output = match format {
        "json" => report::generate_json(&result, &recommendations)?,
        _ => report::generate_text(&result, &recommendations),
    };

    // Output report
    if let Some(output_path) = output {
        std::fs::write(&output_path, &report_output)?;
        println!("Report written to: {}", output_path.display());
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
) -> Result<()> {
    use loader::DataSource;

    let source = DataSource::GeoParquet {
        path: input.clone(),
        time_field: time_field.to_string(),
        time_format: time_format.to_string(),
    };

    println!("Analyzing dataset for optimal parameters...\n");

    // Load and analyze data
    let data = loader::load_data(&source)?;
    
    // Run analyses
    let spatial = analysis::spatial::analyze(&data)?;
    let temporal = analysis::temporal::analyze(&data)?;
    let geometry = analysis::geometry::analyze(&data)?;
    let density = analysis::density::analyze(&data, &spatial)?;

    let result = analysis::AnalysisResult {
        source: source.display_name(),
        feature_count: data.features.len(),
        spatial,
        temporal,
        geometry,
        density,
    };

    // Generate recommendations
    let recommendations = recommend::generate_recommendations(&result);

    // Output as JSON config
    let config = recommend::to_build_config(&recommendations, input, time_field);
    let json = serde_json::to_string_pretty(&config)?;

    if let Some(output_path) = output {
        std::fs::write(&output_path, &json)?;
        println!("Build config written to: {}", output_path.display());
    } else {
        println!("{}", json);
    }

    if show_command {
        println!("\nSuggested stt-build command:");
        println!("{}", recommend::to_command(&recommendations, input, time_field));
    }

    Ok(())
}


