//! Generate animal-migration trajectories from the GBIF "Tracking" datasets.
//!
//! Source: GBIF aggregates animal-tracking studies (many republished from
//! Movebank) as Darwin-Core occurrence datasets. We enumerate the curated
//! `category=Tracking` facet, filtered to openly-licensed sets
//! (CC0 / CC-BY / CC-BY-NC), download each Darwin-Core archive anonymously,
//! and reconstruct one trajectory per tracked individual.
//!   <https://www.gbif.org/dataset-search?category=Tracking>
//!
//! Because studies span different years (a 2008 stork, a 2016 albatross…),
//! real timestamps would leave the globe mostly empty. Instead we **season-
//! fold**: every fix is remapped onto a single reference year by day-of-year,
//! so all animals animate over one representative annual cycle and the
//! spring/autumn migration pulses emerge with everything overlaid. Tracks are
//! coloured by taxonomic class (bird / mammal / fish / reptile / insect),
//! resolved via the GBIF species-match API.

use crate::common::{self, LineStringRecord, PropertyColumn, SttBuildOptions, StreamingLineStringParquetWriter};
use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Utc};
use clap::Parser;
use csv::ReaderBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

const GBIF_API: &str = "https://api.gbif.org/v1";
const USER_AGENT: &str = "stt-generate/0.1 (spatiotemporal-tiles showcase)";

#[derive(Parser, Debug)]
#[command(about = "Generate animal-migration trajectories from GBIF tracking datasets")]
pub struct Args {
    /// Output file (.stt or .parquet)
    #[arg(short, long, default_value = "animals.stt")]
    pub output: PathBuf,

    /// Comma-separated GBIF license codes to include.
    #[arg(long, default_value = "CC0_1_0,CC_BY_4_0,CC_BY_NC_4_0")]
    pub licenses: String,

    /// Cap the number of datasets processed (0 = all matching).
    #[arg(long, default_value = "0")]
    pub max_datasets: usize,

    /// Reference year that every track is season-folded onto.
    #[arg(long, default_value = "2024")]
    pub ref_year: i32,

    /// Minimum fixes per track segment.
    #[arg(long, default_value = "5")]
    pub min_points: usize,

    /// Maximum gap (days) between fixes before starting a new segment.
    #[arg(long, default_value = "21")]
    pub max_gap_days: i64,

    /// Skip datasets with more than this many records (0 = no cap). Large
    /// acoustic-array sets can dominate; use to keep the demo balanced.
    #[arg(long, default_value = "0")]
    pub max_records: u64,

    /// Cache directory for downloaded Darwin-Core archives.
    #[arg(long, default_value = "data/gbif-cache")]
    pub cache_dir: PathBuf,

    /// Re-download archives even if cached.
    #[arg(long)]
    pub refresh: bool,

    /// Skip the stt-build step (emit only the intermediate Parquet).
    #[arg(long)]
    pub skip_build: bool,
}

#[derive(Debug, Clone)]
struct DatasetInfo {
    key: String,
    title: String,
    license: String,
    record_count: u64,
}

/// One real-time fix for an individual animal.
#[derive(Debug, Clone)]
struct Fix {
    ms: i64,
    lon: f64,
    lat: f64,
}

pub fn run(args: Args) -> Result<()> {
    println!("🦅 GBIF Animal-Tracking Generator");
    println!("=================================\n");

    let licenses: Vec<String> = args
        .licenses
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let client = reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(120))
        .build()?;

    fs::create_dir_all(&args.cache_dir)?;

    // Reference-year anchor for season-folding.
    let ref_jan1_ms = Utc
        .with_ymd_and_hms(args.ref_year, 1, 1, 0, 0, 0)
        .single()
        .context("invalid ref_year")?
        .timestamp_millis();

    // 1) Enumerate the open tracking datasets.
    println!("🔎 Enumerating GBIF tracking datasets ({})...", licenses.join(", "));
    let mut datasets = enumerate_datasets(&client, &licenses)?;
    datasets.sort_by(|a, b| a.key.cmp(&b.key)); // deterministic order
    if args.max_records > 0 {
        datasets.retain(|d| d.record_count == 0 || d.record_count <= args.max_records);
    }
    if args.max_datasets > 0 && datasets.len() > args.max_datasets {
        datasets.truncate(args.max_datasets);
    }
    let total_records: u64 = datasets.iter().map(|d| d.record_count).sum();
    println!(
        "   {} datasets selected (~{} records)\n",
        datasets.len(),
        total_records
    );
    anyhow::ensure!(!datasets.is_empty(), "No matching datasets found.");

    // 2) Open the LineString Parquet writer.
    let build_stt =
        args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build;
    let intermediate = if build_stt {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };
    let cols: Vec<PropertyColumn> = vec![
        PropertyColumn::string("organism"),
        PropertyColumn::string("species"),
        PropertyColumn::string("taxon_group"),
        PropertyColumn::string("class"),
        PropertyColumn::string("dataset"),
        PropertyColumn::numeric("segment"),
    ];
    let mut writer = StreamingLineStringParquetWriter::with_columns(&intermediate, cols)?;

    // 3) Process each dataset → folded track segments.
    let pb = ProgressBar::new(datasets.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} datasets — {msg}")?
            .progress_chars("#>-"),
    );

    let mut class_cache: HashMap<String, (String, String)> = HashMap::new();
    let mut total_segments = 0usize;
    let mut total_fixes = 0usize;
    let mut group_counts: HashMap<String, usize> = HashMap::new();

    for ds in &datasets {
        pb.set_message(truncate(&ds.title, 40));
        match process_dataset(
            &client,
            ds,
            &args,
            ref_jan1_ms,
            &mut class_cache,
            &mut writer,
            &mut group_counts,
        ) {
            Ok((segs, fixes)) => {
                total_segments += segs;
                total_fixes += fixes;
            }
            Err(e) => eprintln!("\n⚠️  {}: {e}", truncate(&ds.title, 50)),
        }
        pb.inc(1);
    }
    pb.finish_with_message("done");

    writer.finish()?;
    println!("\n📊 Summary:");
    println!("   {} track segments from {} fixes", total_segments, total_fixes);
    let mut groups: Vec<_> = group_counts.iter().collect();
    groups.sort_by(|a, b| b.1.cmp(a.1));
    for (g, n) in groups {
        println!("     {:<10} {} segments", g, n);
    }

    anyhow::ensure!(total_segments > 0, "No track segments were produced.");

    if build_stt {
        common::run_stt_build_with_full_options(SttBuildOptions {
            input: intermediate.clone(),
            output: args.output.clone(),
            time_field: "timestamp".into(),
            end_time_field: Some("end_timestamp".into()),
            min_zoom: 0,
            max_zoom: 6,
            compression: "zstd".into(),
            temporal_bucket: Some("1d".into()),
            temporal_lod: None,
            summary: None,
            summary_sub_buckets: None,
            min_features_per_tile: None,
            min_zoom_field: None,
            max_zoom_field: None,
        })?;
        let _ = fs::remove_file(&intermediate);
    }

    println!("\n✅ Animal track generation complete!");
    println!("   Output: {}", args.output.display());
    Ok(())
}

/// Page through `dataset/search?category=Tracking` for the given licenses.
fn enumerate_datasets(
    client: &reqwest::blocking::Client,
    licenses: &[String],
) -> Result<Vec<DatasetInfo>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    let limit = 200usize;
    loop {
        let mut url = format!(
            "{}/dataset/search?type=OCCURRENCE&category=Tracking&limit={}&offset={}",
            GBIF_API, limit, offset
        );
        for l in licenses {
            url.push_str(&format!("&license={}", l));
        }
        let v: Value = client.get(&url).send()?.error_for_status()?.json()?;
        let results = v.get("results").and_then(|r| r.as_array()).cloned().unwrap_or_default();
        if results.is_empty() {
            break;
        }
        for r in &results {
            if let Some(key) = r.get("key").and_then(|k| k.as_str()) {
                out.push(DatasetInfo {
                    key: key.to_string(),
                    title: r.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string(),
                    license: r.get("license").and_then(|t| t.as_str()).unwrap_or("").to_string(),
                    record_count: r.get("recordCount").and_then(|c| c.as_u64()).unwrap_or(0),
                });
            }
        }
        offset += limit;
        if v.get("endOfRecords").and_then(|e| e.as_bool()).unwrap_or(true) {
            break;
        }
    }
    Ok(out)
}

/// Download + parse one dataset's Darwin-Core archive into folded tracks.
fn process_dataset(
    client: &reqwest::blocking::Client,
    ds: &DatasetInfo,
    args: &Args,
    ref_jan1_ms: i64,
    class_cache: &mut HashMap<String, (String, String)>,
    writer: &mut StreamingLineStringParquetWriter,
    group_counts: &mut HashMap<String, usize>,
) -> Result<(usize, usize)> {
    let zip_path = args.cache_dir.join(format!("{}.zip", ds.key));
    if args.refresh || !zip_path.exists() || file_too_small(&zip_path) {
        let url = dwc_archive_url(client, &ds.key)?;
        download(&url, &zip_path)?;
    }

    // Read occurrence.txt from the archive.
    let file = File::open(&zip_path)?;
    let mut zip = zip::ZipArchive::new(BufReader::new(file))
        .with_context(|| format!("open zip {}", zip_path.display()))?;
    // Darwin-Core archives usually expose the core as `occurrence.txt`, but
    // some publishers vary the case or nest it in a folder — match tolerantly.
    let occ_name = zip
        .file_names()
        .find(|n| n.to_ascii_lowercase().ends_with("occurrence.txt"))
        .map(|s| s.to_string())
        .context("archive has no occurrence.txt")?;
    let occ = zip.by_name(&occ_name)?;

    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .quoting(false) // DwC archives are tab-delimited with no field quoting
        .flexible(true)
        .from_reader(occ);

    let headers = rdr.headers()?.clone();
    let col = |name: &str| headers.iter().position(|h| h == name);
    let ci_lat = col("decimalLatitude").context("no decimalLatitude")?;
    let ci_lon = col("decimalLongitude").context("no decimalLongitude")?;
    let ci_date = col("eventDate").context("no eventDate")?;
    // Individual identity: prefer organismID, then organismName, then individualID.
    let ci_org = col("organismID");
    let ci_orgname = col("organismName");
    let ci_indiv = col("individualID");
    let ci_sci = col("scientificName");
    let ci_class = col("class");

    let mut by_individual: HashMap<String, Vec<Fix>> = HashMap::new();
    let mut sci_name = String::new();
    let mut record_class = String::new();

    for rec in rdr.records() {
        let rec = match rec {
            Ok(r) => r,
            Err(_) => continue,
        };
        let key = ci_org
            .and_then(|i| nonempty(rec.get(i)))
            .or_else(|| ci_orgname.and_then(|i| nonempty(rec.get(i))))
            .or_else(|| ci_indiv.and_then(|i| nonempty(rec.get(i))));
        let key = match key {
            Some(k) => k,
            None => continue,
        };
        let ms = match rec.get(ci_date).and_then(parse_event_ms) {
            Some(ms) => ms,
            None => continue,
        };
        let (lon, lat) = match (rec.get(ci_lon).and_then(parse_f64), rec.get(ci_lat).and_then(parse_f64)) {
            (Some(lon), Some(lat)) if lon.abs() <= 180.0 && lat.abs() <= 90.0 && (lon != 0.0 || lat != 0.0) => {
                (lon, lat)
            }
            _ => continue,
        };
        if sci_name.is_empty() {
            if let Some(s) = ci_sci.and_then(|i| nonempty(rec.get(i))) {
                sci_name = s;
            }
        }
        if record_class.is_empty() {
            if let Some(c) = ci_class.and_then(|i| nonempty(rec.get(i))) {
                record_class = c;
            }
        }
        by_individual.entry(key).or_default().push(Fix { ms, lon, lat });
    }

    if by_individual.is_empty() {
        return Ok((0, 0));
    }

    // Resolve a coarse taxon group (prefer the archive's own class column,
    // else the GBIF species-match API).
    let (group, class_label) = if !record_class.is_empty() {
        (classify(Some(&record_class), None).to_string(), record_class)
    } else {
        resolve_taxon(client, &sci_name, class_cache)
    };

    let max_gap_ms = args.max_gap_days * 86_400_000;
    let mut segs_written = 0usize;
    let mut fixes_used = 0usize;

    let mut individuals: Vec<&String> = by_individual.keys().collect();
    individuals.sort();
    for indiv in individuals {
        let mut fixes = by_individual[indiv].clone();
        if fixes.len() < args.min_points {
            continue;
        }
        fixes.sort_by_key(|f| f.ms);
        let segments = split_track(&fixes, max_gap_ms, args.min_points);

        for (seg_idx, seg) in segments.iter().enumerate() {
            // Season-fold each fix onto the reference year, keeping only
            // strictly-increasing folded times (drops same-day duplicates).
            let mut coords: Vec<[f64; 2]> = Vec::with_capacity(seg.len());
            let mut vtimes: Vec<i64> = Vec::with_capacity(seg.len());
            for f in seg {
                let folded = fold_to_year(f.ms, ref_jan1_ms);
                if vtimes.last().map(|&t| folded <= t).unwrap_or(false) {
                    continue;
                }
                coords.push([f.lon, f.lat]);
                vtimes.push(folded);
            }
            if coords.len() < 2 {
                continue;
            }

            let mut props = Map::new();
            props.insert("organism".into(), json!(indiv));
            props.insert("species".into(), json!(sci_name));
            props.insert("taxon_group".into(), json!(group));
            props.insert("class".into(), json!(class_label));
            props.insert("dataset".into(), json!(ds.title));
            props.insert("segment".into(), json!(seg_idx));

            let start_t = ms_to_dt(vtimes[0]);
            let end_t = ms_to_dt(*vtimes.last().unwrap());
            fixes_used += coords.len();
            let mut record = LineStringRecord::new(coords, start_t, Some(end_t), props);
            record.vertex_timestamps_ms = Some(vtimes);
            writer.write_linestring(&record)?;
            segs_written += 1;
        }
    }

    if segs_written > 0 {
        *group_counts.entry(group.to_string()).or_default() += segs_written;
    }
    let _ = (ds.license.clone(), ds.record_count); // (carried for future logging)
    Ok((segs_written, fixes_used))
}

fn dwc_archive_url(client: &reqwest::blocking::Client, key: &str) -> Result<String> {
    let url = format!("{}/dataset/{}/endpoint", GBIF_API, key);
    let v: Value = client.get(&url).send()?.error_for_status()?.json()?;
    let eps = v.as_array().context("endpoint list not an array")?;
    eps.iter()
        .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("DWC_ARCHIVE"))
        .and_then(|e| e.get("url").and_then(|u| u.as_str()))
        .map(|s| s.to_string())
        .context("no DWC_ARCHIVE endpoint")
}

/// Resolve a coarse taxon group via one cached GBIF species-match call.
/// Returns `(group, class_label)` where `class_label` is a human-readable
/// class for tooltips (falls back to the inferred group's class for fish,
/// which GBIF leaves null).
fn resolve_taxon(
    client: &reqwest::blocking::Client,
    sci_name: &str,
    cache: &mut HashMap<String, (String, String)>,
) -> (String, String) {
    if sci_name.is_empty() {
        return ("other".into(), String::new());
    }
    if let Some(c) = cache.get(sci_name) {
        return c.clone();
    }
    let url = format!("{}/species/match?name={}", GBIF_API, urlencoding::encode(sci_name));
    let v: Option<Value> = client.get(&url).send().ok().and_then(|r| r.json().ok());
    let class = v
        .as_ref()
        .and_then(|v| v.get("class").and_then(|c| c.as_str()))
        .map(|s| s.to_string());
    let phylum = v
        .as_ref()
        .and_then(|v| v.get("phylum").and_then(|c| c.as_str()))
        .map(|s| s.to_string());
    let group = classify(class.as_deref(), phylum.as_deref()).to_string();
    // Label: real class when GBIF supplies one, else a sensible stand-in so
    // the tooltip isn't blank for the (common) null-class fish.
    let class_label = class.unwrap_or_else(|| match group.as_str() {
        "fish" => "Actinopterygii".to_string(),
        _ => String::new(),
    });
    let result = (group, class_label);
    cache.insert(sci_name.to_string(), result.clone());
    result
}

/// Map a GBIF taxonomic class (+ phylum fallback) to a coarse colour group.
///
/// GBIF's backbone leaves `class` null for ray-finned fish, and uses
/// `Testudines`/`Squamata`/`Crocodylia` rather than a single `Reptilia` — so
/// we normalise both here. A null-class chordate, in animal-tracking data, is
/// overwhelmingly a fish.
fn classify(class: Option<&str>, phylum: Option<&str>) -> &'static str {
    match class.unwrap_or("") {
        "Aves" => "bird",
        "Mammalia" => "mammal",
        "Reptilia" | "Testudines" | "Squamata" | "Crocodylia" | "Lepidosauria"
        | "Archosauria" | "Sauropsida" => "reptile",
        "Actinopterygii" | "Actinopteri" | "Teleostei" | "Elasmobranchii" | "Chondrichthyes"
        | "Holocephali" | "Myxini" | "Petromyzonti" | "Cephalaspidomorphi" | "Coelacanthi"
        | "Dipneusti" | "Sarcopterygii" => "fish",
        "Insecta" => "insect",
        "" => {
            if phylum == Some("Chordata") {
                "fish"
            } else {
                "other"
            }
        }
        _ => "other",
    }
}

/// Remap a real timestamp onto the reference year, preserving day-of-year and
/// time-of-day (`Jan 1 → Jan 1`). Within a single calendar-year segment this
/// is monotonic, so vertex order is preserved.
fn fold_to_year(ms: i64, ref_jan1_ms: i64) -> i64 {
    let dt = ms_to_dt(ms);
    let year_jan1 = Utc
        .with_ymd_and_hms(dt.year(), 1, 1, 0, 0, 0)
        .single()
        .map(|d| d.timestamp_millis())
        .unwrap_or(ms);
    ref_jan1_ms + (ms - year_jan1)
}

/// Implied speed (km/h) above which two consecutive fixes are treated as a
/// teleport — faster than any animal flies, so it's an acoustic-receiver hop
/// or a bad fix. Splitting here keeps spurious globe-spanning lines out.
const MAX_SPEED_KMH: f64 = 150.0;

/// Split a time-sorted track at calendar-year boundaries, large gaps,
/// antimeridian crossings, and implausible-speed teleports (so folded
/// segments stay monotonic, globe-safe, and free of receiver-hop artifacts).
fn split_track(fixes: &[Fix], max_gap_ms: i64, min_points: usize) -> Vec<Vec<Fix>> {
    let mut out: Vec<Vec<Fix>> = Vec::new();
    let mut cur: Vec<Fix> = Vec::new();
    for f in fixes {
        if let Some(last) = cur.last() {
            let gap = f.ms - last.ms > max_gap_ms;
            let year_break = ms_to_dt(f.ms).year() != ms_to_dt(last.ms).year();
            // Antimeridian crossing: any consecutive pair whose longitudes
            // differ by more than 180° straddles the dateline (the shorter
            // path wraps across ±180°). The old `>170 && <-170` test only
            // caught crossings hugging the dateline and missed e.g. -163°→165°,
            // which then rendered as a line sweeping the long way across the
            // whole map.
            let anti = (last.lon - f.lon).abs() > 180.0;
            let dt_h = (f.ms - last.ms).max(0) as f64 / 3_600_000.0;
            let dist_km = common::haversine_distance(last.lat, last.lon, f.lat, f.lon) / 1000.0;
            let teleport = dt_h > 0.0 && dist_km / dt_h > MAX_SPEED_KMH;
            if gap || year_break || anti || teleport {
                if cur.len() >= min_points {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.clear();
                }
            }
        }
        cur.push(f.clone());
    }
    if cur.len() >= min_points {
        out.push(cur);
    }
    out
}

fn download(url: &str, out: &PathBuf) -> Result<()> {
    let status = Command::new("curl")
        .args([
            "-sS", "-f", "-L", "--connect-timeout", "30", "--max-time", "900", "-o",
            out.to_str().unwrap(),
            url,
        ])
        .status()
        .context("failed to run curl")?;
    if !status.success() {
        let _ = fs::remove_file(out);
        anyhow::bail!("curl failed for {url}");
    }
    Ok(())
}

fn file_too_small(p: &PathBuf) -> bool {
    fs::metadata(p).map(|m| m.len() < 128).unwrap_or(true)
}

fn nonempty(s: Option<&str>) -> Option<String> {
    match s {
        Some(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        _ => None,
    }
}

fn parse_f64(s: &str) -> Option<f64> {
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Some(v),
        _ => None,
    }
}

/// Parse a Darwin-Core `eventDate` (RFC3339, `YYYY-MM-DDTHH:MM:SS`, or
/// date-only; ranges use the start). Returns Unix-ms.
fn parse_event_ms(s: &str) -> Option<i64> {
    let s = s.trim();
    let s = s.split('/').next().unwrap_or(s); // ranges: take the start
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc).timestamp_millis());
    }
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(Utc.from_utc_datetime(&ndt).timestamp_millis());
    }
    let date_part = s.get(0..10).unwrap_or(s);
    if let Ok(d) = NaiveDate::parse_from_str(date_part, "%Y-%m-%d") {
        let ndt = d.and_hms_opt(0, 0, 0)?;
        return Some(Utc.from_utc_datetime(&ndt).timestamp_millis());
    }
    None
}

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms).single().unwrap_or_else(Utc::now)
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let t: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{t}…")
    }
}
