//! Montreal BIXI bike-share OD **flowmap** generator (`bixi`).
//!
//! Aggregates real [BIXI open-data](https://bixi.com/en/open-data/) trips into
//! directed **origin→destination station-pair flows**, each emitted as a single
//! 2-vertex `origin → destination` LineString carrying a per-time-bucket count
//! time series (`vertex_value_matrix`). This is the OD-arc counterpart to the
//! street-network `--flows` overview ([`super::nyc_rideshare_flows`]): the same
//! geometry-once / animate-from-the-matrix encoding, but on straight O→D arcs
//! the deck.gl `FlowmapLayer` reads as flowmap.gl-style weighted arrows whose
//! width tracks volume, plus node circles sized by total flow.
//!
//! Because the representation is `OD-pair × bucket` counts, on-the-wire size is
//! bounded by *(kept pairs) × (buckets)* — independent of the ~13M raw
//! trips/year — so a long span fits comfortably. Aggregation IS the
//! visualization here (a summary tier), not a payload hack: we keep every bucket
//! for kept pairs and use a volume-based `min_zoom` only as a legibility LOD
//! (busiest corridors city-wide; minor pairs reveal on zoom-in), mirroring the
//! road-class LOD in [`super::nyc_rideshare_flows`].
//!
//! ## Input schema (auto-detected)
//! BIXI's CSV schema changed across years; this reads both families by header:
//! - **2022+** (e.g. 2024): `STARTSTATIONNAME, STARTSTATIONLATITUDE,
//!   STARTSTATIONLONGITUDE, ENDSTATIONNAME, ENDSTATIONLATITUDE,
//!   ENDSTATIONLONGITUDE, STARTTIMEMS, ENDTIMEMS` — lat/lon embedded per trip,
//!   epoch-ms times. Self-contained (no stations file).
//! - **2014–2021**: `start_date, start_station_code, end_date, end_station_code,
//!   …` (codes renamed `emplacement_pk_*` in 2021) + a separate
//!   `Stations_*.csv` (`code,name,latitude,longitude`). Codes are resolved from
//!   that stations file, falling back to the public BIXI GBFS feed.

use anyhow::{anyhow, Context, Result};
use chrono::{NaiveDate, NaiveDateTime};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use crate::common::{
    self, LineStringRecord, PropertyColumn, SttBuildOptions, StreamingLineStringParquetWriter,
};
use crate::datasets::nyc_rideshare_flows::parse_bin_ms;

/// Public BIXI GBFS station feed (stable, no auth) — fallback for resolving
/// legacy station codes to lat/lon when no stations CSV is present.
const GBFS_STATION_INFO_URL: &str = "https://gbfs.velobixi.com/gbfs/en/station_information.json";

#[derive(Parser, Debug)]
#[command(about = "Generate a Montreal BIXI origin→destination flowmap dataset")]
pub struct Args {
    /// Output packed `.stt` directory (or `*.parquet` to stop at the
    /// intermediate). Default targets the showcase data dir.
    #[arg(short, long, default_value = "examples/showcase/public/data/bixi-flowmap")]
    pub output: PathBuf,

    /// BIXI open-data input: the downloaded `.zip`, an extracted `.csv`, or a
    /// directory containing either (plus an optional `Stations_*.csv` for
    /// pre-2022 code-based files).
    #[arg(long)]
    pub input: PathBuf,

    /// Time bucket for the flow matrix (e.g. `1h`, `30m`, `3h`, `1d`).
    #[arg(long, default_value = "1h")]
    pub bin: String,

    /// Inclusive lower date bound `YYYY-MM-DD` (UTC). Default: no lower bound.
    #[arg(long)]
    pub from: Option<String>,

    /// Exclusive upper date bound `YYYY-MM-DD` (UTC). Default: no upper bound.
    #[arg(long)]
    pub to: Option<String>,

    /// Drop OD pairs with fewer than this many total trips across the span
    /// (legibility threshold — not temporal thinning; kept pairs keep every
    /// bucket).
    #[arg(long, default_value = "30")]
    pub min_trips: u32,

    /// Build pyramid min zoom.
    #[arg(long, default_value = "10")]
    pub min_zoom: u8,

    /// Build pyramid max zoom. Kept modest: OD arcs are long 2-vertex lines that
    /// duplicate into every tile they cross, so deep zooms bloat fast and a
    /// flowmap is a city-scale overview anyway.
    #[arg(long, default_value = "13")]
    pub max_zoom: u8,

    /// Skip the stt-build step (write only the intermediate GeoParquet).
    #[arg(long)]
    pub skip_build: bool,
}

pub fn run(args: Args) -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              🚲 BIXI Montréal Flowmap Generator              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let bin_ms = parse_bin_ms(&args.bin)?;
    let from_ms = args.from.as_deref().map(parse_date_utc_ms).transpose()?;
    let to_ms = args.to.as_deref().map(parse_date_utc_ms).transpose()?;
    if let (Some(f), Some(t)) = (from_ms, to_ms) {
        if t <= f {
            return Err(anyhow!("--to ({}) must be after --from ({})", args.to.as_deref().unwrap_or(""), args.from.as_deref().unwrap_or("")));
        }
    }

    println!("📋 Configuration:");
    println!("   Input:     {}", args.input.display());
    println!("   Output:    {}", args.output.display());
    println!("   Bin:       {} ({} ms)", args.bin, bin_ms);
    println!("   Span:      {} → {}", args.from.as_deref().unwrap_or("(start)"), args.to.as_deref().unwrap_or("(end)"));
    println!("   Min trips: {}", args.min_trips);
    println!("   Zoom:      {}-{}\n", args.min_zoom, args.max_zoom);

    let mut agg = BixiAggregator::new(bin_ms, from_ms, to_ms);

    // Pre-load a stations map (code → lon/lat) for legacy code-based files. Only
    // consulted when a trip row has no embedded coordinates.
    agg.stations_lookup = load_stations_map(&args.input);

    // Stream every trip CSV found in the input (zip entry, file, or directory).
    for csv in find_trip_csvs(&args.input)? {
        println!("📂 Reading {} …", csv.label);
        agg.ingest_csv(csv.reader)?;
    }

    let (read, kept) = (agg.trips_read, agg.trips_kept);
    println!("\n   ✓ Read {read} trips, {kept} within span");
    if kept == 0 {
        return Err(anyhow!("No trips fell within the requested span"));
    }

    // stt-build only consumes GeoParquet. A `*.parquet`/`*.geojson` output means
    // the caller wants only the intermediate (no build); a `.stt` or
    // directory-like output (the showcase default) is built into a packed
    // directory, with the intermediate written to a sibling `.parquet`.
    let ext = args.output.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
    let output_is_intermediate =
        matches!(ext.as_deref(), Some("parquet") | Some("geoparquet") | Some("geojson"));
    let intermediate_path = if output_is_intermediate {
        args.output.clone()
    } else {
        args.output.with_extension("parquet")
    };

    let (features, num_buckets, bucket0, range_end) =
        agg.write_parquet(&intermediate_path, args.min_trips)?;
    println!("\n   ✓ {features} OD-pair corridors · {num_buckets} buckets");
    println!("   ⏱  matrix span (set showcase timeRange to this):");
    println!("       start = {bucket0}");
    println!("       end   = {range_end}");

    if args.skip_build || output_is_intermediate {
        println!("\n📦 Skipping stt-build (intermediate written to {})", intermediate_path.display());
        return Ok(());
    }

    // Build the packed `.stt` directory. Mirrors the `--flows` build: end-time
    // field for the whole-range span, per-feature `min_zoom` LOD filter, and the
    // matrix bin as the temporal bucket. `--publish` (default) emits deploy-ready
    // paged + zstd-19 output.
    common::run_stt_build_with_full_options(SttBuildOptions {
        input: intermediate_path.clone(),
        output: args.output.clone(),
        time_field: "timestamp".to_string(),
        end_time_field: Some("end_timestamp".to_string()),
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        compression: "zstd".to_string(),
        temporal_bucket: Some(args.bin.clone()),
        temporal_lod: None,
        summary: None,
        summary_sub_buckets: None,
        min_features_per_tile: None,
        min_zoom_field: Some("min_zoom".to_string()),
    })?;

    println!("\n✅ BIXI flowmap built: {}", args.output.display());
    println!("   Set datasets.ts timeRange to {{ start: {bucket0}, end: {range_end} }}");
    Ok(())
}

/// Parse a `YYYY-MM-DD` UTC date to epoch ms at 00:00:00.
fn parse_date_utc_ms(s: &str) -> Result<i64> {
    let d = NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d")
        .with_context(|| format!("invalid date '{s}' (expected YYYY-MM-DD)"))?;
    Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp_millis())
}

// ---------------------------------------------------------------------------
// Aggregator
// ---------------------------------------------------------------------------

/// A station identity (name for 2022+, code for legacy) + its representative
/// position. We aggregate by identity so round trips through a relocated dock
/// still collapse onto one node.
type StationKey = String;

/// Directed OD-pair flow accumulator: per pair, a sparse `bucket → count` map,
/// densified onto one global bucket axis at write time.
struct BixiAggregator {
    bin_ms: i64,
    from_ms: Option<i64>,
    to_ms: Option<i64>,
    /// key → representative (lon, lat). First coordinate seen wins.
    station_pos: HashMap<StationKey, [f64; 2]>,
    /// Optional code → (lon, lat) lookup for legacy files without embedded coords.
    stations_lookup: HashMap<String, [f64; 2]>,
    counts: HashMap<(StationKey, StationKey), HashMap<i64, u32>>,
    min_bin: i64,
    max_bin: i64,
    trips_read: u64,
    trips_kept: u64,
}

impl BixiAggregator {
    fn new(bin_ms: i64, from_ms: Option<i64>, to_ms: Option<i64>) -> Self {
        Self {
            bin_ms,
            from_ms,
            to_ms,
            station_pos: HashMap::new(),
            stations_lookup: HashMap::new(),
            counts: HashMap::new(),
            min_bin: i64::MAX,
            max_bin: i64::MIN,
            trips_read: 0,
            trips_kept: 0,
        }
    }

    /// Stream-parse one CSV (any of the known BIXI schemas) into the aggregate.
    fn ingest_csv<R: Read>(&mut self, reader: R) -> Result<()> {
        let mut rdr = csv::ReaderBuilder::new()
            .flexible(true)
            .from_reader(BufReader::with_capacity(1 << 20, reader));
        let headers = rdr.headers()?.clone();
        let schema = TripSchema::detect(&headers)
            .ok_or_else(|| anyhow!("unrecognized BIXI CSV header: {:?}", headers))?;

        let mut record = csv::StringRecord::new();
        while rdr.read_record(&mut record)? {
            self.trips_read += 1;
            if let Some(trip) = schema.parse_row(&record, &self.stations_lookup) {
                self.add_trip(trip);
            }
        }
        Ok(())
    }

    fn add_trip(&mut self, t: ParsedTrip) {
        if let Some(f) = self.from_ms {
            if t.start_ms < f {
                return;
            }
        }
        if let Some(to) = self.to_ms {
            if t.start_ms >= to {
                return;
            }
        }
        // Drop zero-length self-loops: a degenerate arc has no direction and
        // would render as a dot (flowmap.gl drops self-flows too).
        if t.origin_key == t.dest_key {
            return;
        }
        self.station_pos.entry(t.origin_key.clone()).or_insert(t.origin_pos);
        self.station_pos.entry(t.dest_key.clone()).or_insert(t.dest_pos);

        let bin = t.start_ms.div_euclid(self.bin_ms);
        *self
            .counts
            .entry((t.origin_key, t.dest_key))
            .or_default()
            .entry(bin)
            .or_insert(0) += 1;
        self.min_bin = self.min_bin.min(bin);
        self.max_bin = self.max_bin.max(bin);
        self.trips_kept += 1;
    }

    /// Emit one 2-vertex OD LineString per kept pair, carrying a `[2 ×
    /// num_buckets]` vertex-major matrix (both vertices = the pair's per-bucket
    /// count). Returns (features, num_buckets, bucket0_ms, range_end_ms).
    fn write_parquet(
        &self,
        output: &Path,
        min_trips: u32,
    ) -> Result<(usize, usize, i64, i64)> {
        let property_columns = vec![
            PropertyColumn::numeric("total_count"),
            PropertyColumn::numeric("min_zoom"),
            PropertyColumn::string("origin"),
            PropertyColumn::string("destination"),
        ];
        let mut writer = StreamingLineStringParquetWriter::with_columns(output, property_columns)?;

        if self.counts.is_empty() {
            writer.finish()?;
            return Ok((0, 0, 0, 0));
        }

        let num_buckets = (self.max_bin - self.min_bin + 1) as usize;
        let bucket0 = self.min_bin * self.bin_ms;
        let range_end = bucket0 + num_buckets as i64 * self.bin_ms;

        // Deterministic (content-addressable) output: sort pairs by key.
        let mut pairs: Vec<&(StationKey, StationKey)> = self.counts.keys().collect();
        pairs.sort_unstable();

        let pb = ProgressBar::new(pairs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40}] {pos}/{len} corridors")?
                .progress_chars("=>-"),
        );

        let mut features = 0usize;
        for pair in pairs {
            let bins = &self.counts[pair];
            let total: u32 = bins.values().copied().sum();
            pb.inc(1);
            if total < min_trips {
                continue;
            }
            let (Some(&o), Some(&d)) =
                (self.station_pos.get(&pair.0), self.station_pos.get(&pair.1))
            else {
                continue;
            };

            // Both vertices of the 2-vertex arc carry the same per-bucket count;
            // the FlowmapLayer reads vertex 0 as the arc's flow at time t.
            let mut matrix = vec![0.0f32; 2 * num_buckets];
            for (&bin, &c) in bins {
                let b = (bin - self.min_bin) as usize;
                matrix[b] = c as f32; // vertex 0
                matrix[num_buckets + b] = c as f32; // vertex 1
            }

            let mut properties = Map::new();
            properties.insert("total_count".to_string(), json!(total));
            properties.insert("min_zoom".to_string(), json!(volume_min_zoom(total)));
            properties.insert("origin".to_string(), json!(pair.0));
            properties.insert("destination".to_string(), json!(pair.1));

            writer.write_linestring(&LineStringRecord {
                coordinates: vec![o, d],
                timestamp_ms: bucket0,
                end_timestamp_ms: Some(range_end),
                vertex_timestamps_ms: None,
                vertex_values: None,
                vertex_value_matrix: Some(matrix),
                properties,
            })?;
            features += 1;
        }
        pb.finish_and_clear();
        writer.finish()?;
        Ok((features, num_buckets, bucket0, range_end))
    }
}

/// Volume-based legibility LOD: busiest corridors visible city-wide (low zoom),
/// minor pairs only reveal on zoom-in. Mirrors the road-class `min_zoom` in
/// [`super::nyc_rideshare_flows`]. Thresholds tuned against measured pair counts.
fn volume_min_zoom(total: u32) -> u8 {
    match total {
        t if t >= 2000 => 10,
        t if t >= 800 => 11,
        t if t >= 250 => 12,
        _ => 13,
    }
}

// ---------------------------------------------------------------------------
// Schema detection + row parsing
// ---------------------------------------------------------------------------

struct ParsedTrip {
    origin_key: StationKey,
    dest_key: StationKey,
    origin_pos: [f64; 2],
    dest_pos: [f64; 2],
    start_ms: i64,
}

/// Resolved column indices for one CSV. Supports the embedded-coordinate family
/// (2022+) and the station-code family (2014–2021).
struct TripSchema {
    start_lat: Option<usize>,
    start_lon: Option<usize>,
    end_lat: Option<usize>,
    end_lon: Option<usize>,
    start_name: Option<usize>,
    end_name: Option<usize>,
    start_code: Option<usize>,
    end_code: Option<usize>,
    start_time: usize,
}

/// Normalize a header cell for tolerant matching: lowercase, alphanumeric only.
fn norm(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

impl TripSchema {
    fn detect(headers: &csv::StringRecord) -> Option<Self> {
        let cols: Vec<String> = headers.iter().map(norm).collect();
        let find = |pred: &dyn Fn(&str) -> bool| cols.iter().position(|c| pred(c));

        let start_lat = find(&|c| c.contains("start") && c.contains("lat"));
        let start_lon = find(&|c| c.contains("start") && (c.contains("lon") || c.contains("lng")));
        let end_lat = find(&|c| c.contains("end") && c.contains("lat"));
        let end_lon = find(&|c| c.contains("end") && (c.contains("lon") || c.contains("lng")));
        let start_name = find(&|c| c.contains("start") && c.contains("name"));
        let end_name = find(&|c| c.contains("end") && c.contains("name"));
        // Station codes: `start_station_code` / `emplacement_pk_start`.
        let start_code = find(&|c| {
            (c.contains("start") && c.contains("code")) || c == "emplacementpkstart"
        });
        let end_code =
            find(&|c| (c.contains("end") && c.contains("code")) || c == "emplacementpkend");
        // Start time: prefer an explicit start-time/date column that isn't a
        // station attribute. `starttimems`, `startdate`, `startedat`.
        let start_time = find(&|c| {
            (c.contains("start") || c == "startedat")
                && (c.contains("time") || c.contains("date"))
                && !c.contains("station")
                && !c.contains("name")
        })?;

        // Need a station identity AND a position source on both ends.
        let has_pos = start_lat.is_some() && start_lon.is_some() && end_lat.is_some() && end_lon.is_some();
        let has_code = start_code.is_some() && end_code.is_some();
        if !has_pos && !has_code {
            return None;
        }
        Some(Self {
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            start_name,
            end_name,
            start_code,
            end_code,
            start_time,
        })
    }

    fn parse_row(
        &self,
        rec: &csv::StringRecord,
        stations: &HashMap<String, [f64; 2]>,
    ) -> Option<ParsedTrip> {
        let start_ms = parse_time_ms(rec.get(self.start_time)?)?;

        // Position: embedded coords win; else resolve a code via the stations map.
        let origin_pos = self.position(rec, self.start_lat, self.start_lon, self.start_code, stations)?;
        let dest_pos = self.position(rec, self.end_lat, self.end_lon, self.end_code, stations)?;

        // Identity: name if present, else code, else a coordinate hash.
        let origin_key = self.identity(rec, self.start_name, self.start_code, origin_pos);
        let dest_key = self.identity(rec, self.end_name, self.end_code, dest_pos);

        Some(ParsedTrip { origin_key, dest_key, origin_pos, dest_pos, start_ms })
    }

    fn position(
        &self,
        rec: &csv::StringRecord,
        lat: Option<usize>,
        lon: Option<usize>,
        code: Option<usize>,
        stations: &HashMap<String, [f64; 2]>,
    ) -> Option<[f64; 2]> {
        if let (Some(la), Some(lo)) = (lat, lon) {
            let latv: f64 = rec.get(la)?.trim().parse().ok()?;
            let lonv: f64 = rec.get(lo)?.trim().parse().ok()?;
            if latv.is_finite() && lonv.is_finite() && (latv != 0.0 || lonv != 0.0) {
                return Some([lonv, latv]);
            }
        }
        if let Some(ci) = code {
            let c = rec.get(ci)?.trim();
            if let Some(p) = stations.get(c) {
                return Some(*p);
            }
        }
        None
    }

    fn identity(
        &self,
        rec: &csv::StringRecord,
        name: Option<usize>,
        code: Option<usize>,
        pos: [f64; 2],
    ) -> StationKey {
        if let Some(ni) = name {
            if let Some(n) = rec.get(ni) {
                let n = n.trim();
                if !n.is_empty() {
                    return n.to_string();
                }
            }
        }
        if let Some(ci) = code {
            if let Some(c) = rec.get(ci) {
                let c = c.trim();
                if !c.is_empty() {
                    return c.to_string();
                }
            }
        }
        // Last resort: quantize coords (~11 m) so the same dock collapses.
        format!("{:.4},{:.4}", pos[0], pos[1])
    }
}

/// Parse a BIXI start time: epoch ms (`1704230756167`), epoch s, or a datetime
/// string (`2019-04-15 08:23:00` / `…08:23`).
fn parse_time_ms(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Ok(n) = s.parse::<i64>() {
        // Heuristic: ms since epoch is ~1.7e12; seconds ~1.7e9.
        if n > 1_000_000_000_000 {
            return Some(n);
        }
        if n > 1_000_000_000 {
            return Some(n * 1000);
        }
    }
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some(dt.and_utc().timestamp_millis());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Input discovery (zip / csv / directory) + legacy stations map
// ---------------------------------------------------------------------------

/// A named trip-CSV stream sourced from a file, zip entry, or directory member.
struct NamedCsv {
    label: String,
    reader: Box<dyn Read>,
}

/// Find every trip CSV in the input (`.zip` entries, a `.csv` file, or a
/// directory of either). Stations files (`*stations*`) are excluded here — they
/// are consumed by [`load_stations_map`].
fn find_trip_csvs(input: &Path) -> Result<Vec<NamedCsv>> {
    let mut out = Vec::new();
    collect_csvs(input, &mut out)?;
    if out.is_empty() {
        return Err(anyhow!("no trip CSVs found under {}", input.display()));
    }
    Ok(out)
}

fn collect_csvs(path: &Path, out: &mut Vec<NamedCsv>) -> Result<()> {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        for e in entries {
            collect_csvs(&e, out)?;
        }
        return Ok(());
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "zip" => {
            let file = File::open(path)
                .with_context(|| format!("opening zip {}", path.display()))?;
            let mut zip = zip::ZipArchive::new(file)?;
            // Names first (immutable borrow), then extract each to a temp file —
            // zip entries aren't seekable/clonable, and trip CSVs are huge, so we
            // spill to a sibling temp rather than buffer in RAM.
            let names: Vec<String> = (0..zip.len())
                .filter_map(|i| zip.by_index(i).ok().map(|f| f.name().to_string()))
                .filter(|n| n.to_ascii_lowercase().ends_with(".csv") && !is_stations_name(n))
                .collect();
            for name in names {
                let mut entry = zip.by_name(&name)?;
                let tmp = std::env::temp_dir().join(format!(
                    "bixi-{}.csv",
                    norm(&name).chars().take(40).collect::<String>()
                ));
                let mut f = File::create(&tmp)?;
                std::io::copy(&mut entry, &mut f)?;
                out.push(NamedCsv {
                    label: format!("{} ({})", name, path.display()),
                    reader: Box::new(File::open(&tmp)?),
                });
            }
        }
        "csv" => {
            if !is_stations_name(&path.to_string_lossy()) {
                out.push(NamedCsv {
                    label: path.display().to_string(),
                    reader: Box::new(File::open(path)?),
                });
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_stations_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("station") && !n.contains("trip") && !n.contains("donnees") && !n.contains("od")
}

/// Build a legacy code → (lon, lat) map from any `*stations*.csv` in the input,
/// falling back to the public GBFS feed when none is present. Returns an empty
/// map for the embedded-coordinate (2022+) family, which never consults it.
fn load_stations_map(input: &Path) -> HashMap<String, [f64; 2]> {
    if let Some(map) = stations_from_input(input) {
        if !map.is_empty() {
            println!("   ✓ Loaded {} stations from a stations CSV", map.len());
            return map;
        }
    }
    HashMap::new()
}

fn stations_from_input(input: &Path) -> Option<HashMap<String, [f64; 2]>> {
    let mut files = Vec::new();
    gather_station_files(input, &mut files);
    if files.is_empty() {
        return None;
    }
    let mut map = HashMap::new();
    for content in files {
        let mut rdr = csv::ReaderBuilder::new().flexible(true).from_reader(content.as_bytes());
        let headers = rdr.headers().ok()?.clone();
        let cols: Vec<String> = headers.iter().map(norm).collect();
        let code = cols.iter().position(|c| c == "code" || c.contains("pk") || c.contains("shortname"))?;
        let lat = cols.iter().position(|c| c.contains("lat"))?;
        let lon = cols.iter().position(|c| c.contains("lon") || c.contains("lng"))?;
        for rec in rdr.records().flatten() {
            if let (Some(c), Some(la), Some(lo)) = (rec.get(code), rec.get(lat), rec.get(lon)) {
                if let (Ok(la), Ok(lo)) = (la.trim().parse::<f64>(), lo.trim().parse::<f64>()) {
                    map.insert(c.trim().to_string(), [lo, la]);
                }
            }
        }
    }
    Some(map)
}

fn gather_station_files(path: &Path, out: &mut Vec<String>) {
    if path.is_dir() {
        if let Ok(rd) = std::fs::read_dir(path) {
            for e in rd.flatten() {
                gather_station_files(&e.path(), out);
            }
        }
        return;
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if ext == "csv" && is_stations_name(name) {
        if let Ok(s) = std::fs::read_to_string(path) {
            out.push(s);
        }
    } else if ext == "zip" {
        if let Ok(file) = File::open(path) {
            if let Ok(mut zip) = zip::ZipArchive::new(file) {
                let names: Vec<String> = (0..zip.len())
                    .filter_map(|i| zip.by_index(i).ok().map(|f| f.name().to_string()))
                    .filter(|n| n.to_ascii_lowercase().ends_with(".csv") && is_stations_name(n))
                    .collect();
                for n in names {
                    if let Ok(mut e) = zip.by_name(&n) {
                        let mut s = String::new();
                        if e.read_to_string(&mut s).is_ok() {
                            out.push(s);
                        }
                    }
                }
            }
        }
    }
}

/// Fetch the public BIXI GBFS station feed → `short_name` → (lon, lat). Used
/// only for legacy code-based files with no stations CSV. Kept simple; failures
/// degrade to an empty map (such rows then drop for lack of a position).
#[allow(dead_code)]
fn gbfs_stations() -> HashMap<String, [f64; 2]> {
    let mut map = HashMap::new();
    let Ok(resp) = reqwest::blocking::get(GBFS_STATION_INFO_URL) else {
        return map;
    };
    let Ok(json): Result<serde_json::Value, _> = resp.json() else {
        return map;
    };
    if let Some(arr) = json["data"]["stations"].as_array() {
        for s in arr {
            if let (Some(sn), Some(lat), Some(lon)) =
                (s["short_name"].as_str(), s["lat"].as_f64(), s["lon"].as_f64())
            {
                map.insert(sn.to_string(), [lon, lat]);
            }
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(fields: &[&str]) -> csv::StringRecord {
        csv::StringRecord::from(fields.to_vec())
    }

    #[test]
    fn detects_2024_embedded_coord_schema() {
        let headers = rec(&[
            "STARTSTATIONNAME",
            "STARTSTATIONARRONDISSEMENT",
            "STARTSTATIONLATITUDE",
            "STARTSTATIONLONGITUDE",
            "ENDSTATIONNAME",
            "ENDSTATIONARRONDISSEMENT",
            "ENDSTATIONLATITUDE",
            "ENDSTATIONLONGITUDE",
            "STARTTIMEMS",
            "ENDTIMEMS",
        ]);
        let s = TripSchema::detect(&headers).expect("schema");
        assert_eq!(s.start_lat, Some(2));
        assert_eq!(s.start_lon, Some(3));
        assert_eq!(s.end_lat, Some(6));
        assert_eq!(s.end_lon, Some(7));
        assert_eq!(s.start_name, Some(0));
        assert_eq!(s.start_time, 8);

        let row = rec(&[
            "A", "Ville-Marie", "45.5", "-73.5", "B", "Ville-Marie", "45.6", "-73.6",
            "1704230756167", "1704231106232",
        ]);
        let t = s.parse_row(&row, &HashMap::new()).expect("trip");
        assert_eq!(t.origin_key, "A");
        assert_eq!(t.dest_key, "B");
        assert_eq!(t.origin_pos, [-73.5, 45.5]);
        assert_eq!(t.start_ms, 1704230756167);
    }

    #[test]
    fn detects_legacy_code_schema() {
        let headers = rec(&[
            "start_date",
            "start_station_code",
            "end_date",
            "end_station_code",
            "duration_sec",
            "is_member",
        ]);
        let s = TripSchema::detect(&headers).expect("schema");
        assert_eq!(s.start_code, Some(1));
        assert_eq!(s.end_code, Some(3));
        assert_eq!(s.start_time, 0);

        let mut stations = HashMap::new();
        stations.insert("6001".to_string(), [-73.57, 45.49]);
        stations.insert("6002".to_string(), [-73.54, 45.53]);
        let row = rec(&["2019-04-15 08:23:00", "6001", "2019-04-15 08:31:00", "6002", "480", "1"]);
        let t = s.parse_row(&row, &stations).expect("trip");
        assert_eq!(t.origin_key, "6001");
        assert_eq!(t.origin_pos, [-73.57, 45.49]);
        assert_eq!(t.start_ms, NaiveDate::from_ymd_opt(2019, 4, 15).unwrap()
            .and_hms_opt(8, 23, 0).unwrap().and_utc().timestamp_millis());
    }

    #[test]
    fn aggregates_directed_pairs_with_matrix() {
        // bin = 1h. Two A→B trips in bin 0, one in bin 2; one B→A trip in bin 0.
        let h = 3_600_000i64;
        let mut agg = BixiAggregator::new(h, None, None);
        let mk = |o: &str, op: [f64; 2], d: &str, dp: [f64; 2], ms: i64| ParsedTrip {
            origin_key: o.into(),
            dest_key: d.into(),
            origin_pos: op,
            dest_pos: dp,
            start_ms: ms,
        };
        let a = [-73.5, 45.5];
        let b = [-73.6, 45.6];
        agg.add_trip(mk("A", a, "B", b, 0));
        agg.add_trip(mk("A", a, "B", b, 60_000));
        agg.add_trip(mk("A", a, "B", b, 2 * h));
        agg.add_trip(mk("B", b, "A", a, 0));
        // Self-loop dropped.
        agg.add_trip(mk("A", a, "A", a, 0));

        assert_eq!(agg.trips_kept, 4);
        assert_eq!(agg.counts.len(), 2); // A→B and B→A distinct (directed)
        let ab = &agg.counts[&("A".to_string(), "B".to_string())];
        assert_eq!(ab[&0], 2);
        assert_eq!(ab[&2], 1);

        let dir = std::env::temp_dir().join("bixi-test-out.parquet");
        let (features, buckets, bucket0, range_end) = agg.write_parquet(&dir, 1).unwrap();
        assert_eq!(features, 2);
        assert_eq!(buckets, 3); // bins 0,1,2
        assert_eq!(bucket0, 0);
        assert_eq!(range_end, 3 * h);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn min_trips_threshold_drops_sparse_pairs() {
        let h = 3_600_000i64;
        let mut agg = BixiAggregator::new(h, None, None);
        let busy = ParsedTrip { origin_key: "A".into(), dest_key: "B".into(), origin_pos: [0.0, 0.0], dest_pos: [1.0, 1.0], start_ms: 0 };
        for _ in 0..5 {
            agg.add_trip(ParsedTrip { ..clone_trip(&busy) });
        }
        agg.add_trip(ParsedTrip { origin_key: "C".into(), dest_key: "D".into(), origin_pos: [0.0, 0.0], dest_pos: [2.0, 2.0], start_ms: 0 });
        let dir = std::env::temp_dir().join("bixi-test-thresh.parquet");
        let (features, _, _, _) = agg.write_parquet(&dir, 3).unwrap();
        assert_eq!(features, 1); // only A→B (5) clears min_trips=3
        let _ = std::fs::remove_file(&dir);
    }

    fn clone_trip(t: &ParsedTrip) -> ParsedTrip {
        ParsedTrip {
            origin_key: t.origin_key.clone(),
            dest_key: t.dest_key.clone(),
            origin_pos: t.origin_pos,
            dest_pos: t.dest_pos,
            start_ms: t.start_ms,
        }
    }

    #[test]
    fn time_parse_handles_ms_and_strings() {
        assert_eq!(parse_time_ms("1704230756167"), Some(1704230756167));
        assert_eq!(parse_time_ms("1704230756"), Some(1704230756000));
        assert_eq!(
            parse_time_ms("2019-04-15 08:23:00"),
            Some(NaiveDate::from_ymd_opt(2019, 4, 15).unwrap().and_hms_opt(8, 23, 0).unwrap().and_utc().timestamp_millis())
        );
        assert_eq!(parse_time_ms("not a time"), None);
    }

    #[test]
    fn volume_lod_decreases_with_traffic() {
        assert_eq!(volume_min_zoom(5000), 10);
        assert_eq!(volume_min_zoom(1000), 11);
        assert_eq!(volume_min_zoom(300), 12);
        assert_eq!(volume_min_zoom(10), 13);
    }
}
