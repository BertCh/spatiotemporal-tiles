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
//! With `--streets` the same ingested OD-pair × bucket counts are instead routed
//! onto Montréal's **bicycle** street network (OSRM bicycle profile +
//! [`super::osm_streets`] / [`super::nyc_rideshare_flows`]) and baked into
//! per-segment street corridors — the BIXI counterpart of `nyc-rideshare
//! --flows`, rendered by `FlowCorridorLayer` as a street-network heatmap.
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
use crate::edge_bundle::{self, BundleParams};

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

    /// Disable build-time station clustering and fall back to the legacy
    /// volume-based LOD (full-resolution corridors at every zoom, minor pairs
    /// dropped at low zoom). By default the generator clusters stations into
    /// hubs per zoom and aggregates flows between them (flowmap.gl-style),
    /// baking the per-zoom aggregation into the tile pyramid.
    #[arg(long)]
    pub no_cluster: bool,

    /// Cluster merge radius in screen pixels at tile extent 512 (Supercluster's
    /// scale). Larger = coarser hubs at a given zoom. The per-zoom merge radius
    /// is `cluster_radius / (512 · 2^zoom)` in world units, so hubs refine as
    /// you zoom in; the deepest zoom is always full per-station resolution.
    #[arg(long, default_value = "40")]
    pub cluster_radius: f64,

    /// **Bake** KDEEB edge bundling into the tile geometry at build time: each
    /// zoom's clustered hub-pair corridors are bundled GLOBALLY (one density
    /// field per zoom) into smooth multi-vertex rivers and stored as polylines,
    /// so the client renders a static curve — no GPU edge bundler, no float-blend
    /// capability, works on mobile, deterministic. Requires clustering (the
    /// default; incompatible with `--no-cluster`, since bundling is a global
    /// per-zoom operation). Render with `BundledFlowmapLayer({ preBundled: true })`.
    #[arg(long)]
    pub bake_bundling: bool,

    /// Control points per baked bundled corridor (P ≥ 3). Each vertex carries a
    /// time-series matrix row, so fewer points = smaller tiles, more = smoother
    /// curves. Only used with `--bake-bundling`.
    #[arg(long, default_value = "24")]
    pub bundle_points: usize,

    /// Initial KDEEB kernel bandwidth as a fraction of the zoom's extent for
    /// baked bundling — the headline knob (larger bundles more aggressively).
    #[arg(long, default_value = "0.05")]
    pub bundle_kernel: f64,

    /// KDEEB density-advection iterations for baked bundling (≈15 converges).
    #[arg(long, default_value = "15")]
    pub bundle_iterations: usize,

    /// Per-iteration Laplacian smoothing strength `[0,1]` for baked bundling.
    #[arg(long, default_value = "0.5")]
    pub bundle_smoothing: f64,

    /// Aggregate trips onto the Montréal **street/bike network** instead of
    /// straight OD arcs: route each unique OD pair once through a local OSRM
    /// **bicycle** server (`--osrm-url`) and bake per-segment ridership into
    /// street corridors carrying a per-bucket count matrix — the BIXI
    /// counterpart of `nyc-rideshare --flows`. Renders with `FlowCorridorLayer`.
    /// Incompatible with the OD-arc options `--bake-bundling` / `--no-cluster` /
    /// `--cluster-radius` (those shape straight arcs, not routed streets).
    #[arg(long)]
    pub streets: bool,

    /// OSM `.osm.pbf` bicycle network for `--streets` (required with it). Use
    /// the same extract OSRM's bicycle profile was built from
    /// (e.g. `scripts/data-generation/osrm-data/quebec-latest.osm.pbf`).
    #[arg(long)]
    pub osm_pbf: Option<PathBuf>,

    /// OSRM server URL for `--streets` routing (a bicycle-profile server). The
    /// Montréal bike server defaults to port 5001 so it can coexist with the
    /// NYC car server on 5000.
    #[arg(long, default_value = "http://localhost:5001")]
    pub osrm_url: String,

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

    // Street-network mode shapes routed corridors, not straight arcs, so the
    // OD-arc bundling/clustering knobs don't apply.
    if args.streets && args.bake_bundling {
        return Err(anyhow!(
            "--streets and --bake-bundling are mutually exclusive: --bake-bundling \
             relaxes straight OD arcs, while --streets routes trips onto the bike \
             network (street corridors, no bundling)"
        ));
    }

    println!("📋 Configuration:");
    println!("   Input:     {}", args.input.display());
    println!("   Output:    {}", args.output.display());
    println!("   Bin:       {} ({} ms)", args.bin, bin_ms);
    println!("   Span:      {} → {}", args.from.as_deref().unwrap_or("(start)"), args.to.as_deref().unwrap_or("(end)"));
    println!("   Min trips: {}", args.min_trips);
    println!("   Zoom:      {}-{}", args.min_zoom, args.max_zoom);
    if args.no_cluster {
        println!("   Cluster:   off (legacy volume LOD)");
    } else {
        println!("   Cluster:   on (radius {} px)", args.cluster_radius);
    }

    // Baked bundling is a GLOBAL per-zoom operation, so it rides on the clustered
    // path (each zoom is a complete edge set); reject the multi-zoom volume LOD.
    let bundle = if args.bake_bundling {
        if args.no_cluster {
            return Err(anyhow!(
                "--bake-bundling requires clustering (remove --no-cluster): edge \
                 bundling is a global per-zoom operation and the volume LOD spans zooms"
            ));
        }
        let bp = BundleParams {
            points: args.bundle_points.max(3),
            iterations: args.bundle_iterations,
            kernel_fraction: args.bundle_kernel,
            smoothing: args.bundle_smoothing,
            anneal: 0.8,
        };
        println!(
            "   Bundle:    BAKED (P={}, kernel={}, iters={}, smooth={})\n",
            bp.points, bp.kernel_fraction, bp.iterations, bp.smoothing
        );
        Some(bp)
    } else {
        println!();
        None
    };

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

    // Street-network overview: route the OD pairs onto the bike network and bake
    // per-segment counts into corridors (a different geometry + build path).
    if args.streets {
        return generate_streets(&args, &agg, bin_ms);
    }

    let (intermediate_path, output_is_intermediate) = resolve_intermediate(&args.output);

    let cluster_radius = if args.no_cluster { None } else { Some(args.cluster_radius) };
    let (features, num_buckets, bucket0, range_end) = agg.write_parquet(
        &intermediate_path,
        args.min_trips,
        cluster_radius,
        args.min_zoom,
        args.max_zoom,
        bundle.as_ref(),
    )?;
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
        // Per-zoom clustered corridors are confined to a single-zoom band
        // [min_zoom, max_zoom] = [z, z]; --no-cluster falls back to the
        // open-ended volume LOD (min only, no max column → appears at all zooms
        // ≥ min). The build reads both property columns to enforce the band.
        min_zoom_field: Some("min_zoom".to_string()),
        max_zoom_field: if args.no_cluster {
            None
        } else {
            Some("max_zoom".to_string())
        },
        // OD corridors are 2-vertex source→target lines: clipping them at tile
        // boundaries replaces the true station endpoints with tile-edge points
        // (false "stations" on seams, and broken bundling). Keep whole corridors.
        no_clip: true,
        // Baked bundles are coordinate-heavy (P control points per corridor) and
        // rendered only at city-overview zooms, where 1 m precision is invisible —
        // quantize to roughly halve the geometry column so the comprehensive
        // (relaxed `--min-trips`) network stays affordable on the wire.
        quantize_coords: bundle.as_ref().map(|_| 1.0),
        quantize_attrs: Vec::new(),
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

/// Where the intermediate GeoParquet goes, and whether the caller wants only
/// the intermediate. stt-build only consumes GeoParquet: a `*.parquet`/
/// `*.geojson` output means stop at the intermediate; a `.stt` or directory
/// output (the showcase default) is built into a packed directory with the
/// intermediate written to a sibling `.parquet`.
fn resolve_intermediate(output: &Path) -> (PathBuf, bool) {
    let ext = output.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
    let is_intermediate =
        matches!(ext.as_deref(), Some("parquet") | Some("geoparquet") | Some("geojson"));
    let path = if is_intermediate {
        output.to_path_buf()
    } else {
        output.with_extension("parquet")
    };
    (path, is_intermediate)
}

/// `--streets`: route each unique OD pair once through the OSRM **bicycle**
/// server, then distribute its per-bucket trip count onto every street edge the
/// route traverses, emitting one corridor per trafficked OSM way carrying a
/// `vertex_value_matrix` — the BIXI counterpart of `nyc-rideshare --flows`,
/// reusing [`OsmNetwork`] + [`FlowAggregator`]. Build mirrors that flows path:
/// road-class `min_zoom` LOD, the matrix bin as temporal bucket, corridors
/// clipped at tile seams (real street geometry, unlike OD arcs).
fn generate_streets(args: &Args, agg: &BixiAggregator, bin_ms: i64) -> Result<()> {
    use crate::datasets::nyc_rideshare_flows::FlowAggregator;
    use crate::datasets::osm_streets::{NetworkMode, OsmNetwork};
    use crate::datasets::osrm::{check_osrm_connectivity, get_osrm_route_pooled};
    use rayon::prelude::*;

    let osm_pbf = args.osm_pbf.as_ref().ok_or_else(|| {
        anyhow!(
            "--streets routes BIXI trips onto the OSM bicycle network; pass \
             --osm-pbf <file.osm.pbf> (e.g. \
             scripts/data-generation/osrm-data/quebec-latest.osm.pbf — the \
             extract OSRM's bicycle profile was built from)"
        )
    })?;

    // Fail fast if the routing server isn't up before loading a large network.
    check_osrm_connectivity(&args.osrm_url)?;

    println!("🗺  Loading OSM bicycle network from {} …", osm_pbf.display());
    let network = OsmNetwork::from_pbf_with_mode(osm_pbf, NetworkMode::Bike)?;

    // Unique OD pairs above the legibility threshold, with endpoints + per-bin
    // counts. Routing one geometry per *pair* (not per trip) is the cheap lever:
    // counts already collapse millions of trips onto a bounded pair set.
    let pairs: Vec<([f64; 2], [f64; 2], &HashMap<i64, u32>)> = agg
        .counts
        .iter()
        .filter(|(_, bins)| bins.values().sum::<u32>() >= args.min_trips)
        .filter_map(|((o, d), bins)| {
            let op = agg.station_pos.get(o)?;
            let dp = agg.station_pos.get(d)?;
            Some((*op, *dp, bins))
        })
        .collect();
    println!(
        "🚲 Routing {} OD pairs (≥{} trips) through OSRM (bicycle) on {} …",
        pairs.len(),
        args.min_trips,
        args.osrm_url
    );

    // Route every pair once, in parallel over a pooled client. Unroutable pairs
    // (OSRM can't connect the docks) are dropped rather than crow-flied — a
    // straight line would smear counts onto edges no rider used.
    let routed: Vec<(Vec<[f64; 2]>, &HashMap<i64, u32>)> = pairs
        .par_iter()
        .filter_map(|(op, dp, bins)| match get_osrm_route_pooled(
            &args.osrm_url,
            op[0],
            op[1],
            dp[0],
            dp[1],
        ) {
            Ok(Some(route)) => {
                let coords: Vec<[f64; 2]> =
                    route.geometry.coordinates.iter().map(|c| [c[0], c[1]]).collect();
                (coords.len() >= 2).then_some((coords, *bins))
            }
            _ => None,
        })
        .collect();

    // Distribute sequentially (the aggregator mutates shared edge counts).
    let mut flow_agg = FlowAggregator::new(bin_ms, network);
    for (coords, bins) in &routed {
        flow_agg.add_route_bins(coords, bins);
    }

    let (added, _skipped, exact, fallback, miss) = flow_agg.stats();
    let seg_total = exact + fallback + miss;
    let match_pct = if seg_total > 0 {
        100.0 * (exact + fallback) as f64 / seg_total as f64
    } else {
        0.0
    };
    println!(
        "   ✓ routed {} / {} pairs · segments {:.1}% matched ({} exact, {} fallback, {} unmatched)",
        added,
        pairs.len(),
        match_pct,
        exact,
        fallback,
        miss
    );
    if added == 0 {
        return Err(anyhow!(
            "No OD pairs routed — is the OSRM bicycle server at {} loaded with the \
             Montréal extract?",
            args.osrm_url
        ));
    }

    let span = flow_agg.bucket_span_ms();
    let (intermediate_path, output_is_intermediate) = resolve_intermediate(&args.output);
    let (features, num_buckets) = flow_agg.write_parquet(&intermediate_path)?;
    let (bucket0, range_end) = span.unwrap_or((0, 0));
    println!("\n   ✓ {features} street corridors · {num_buckets} buckets");
    println!("   ⏱  matrix span (set showcase timeRange to this):");
    println!("       start = {bucket0}");
    println!("       end   = {range_end}");

    if args.skip_build || output_is_intermediate {
        println!(
            "\n📦 Skipping stt-build (intermediate written to {})",
            intermediate_path.display()
        );
        return Ok(());
    }

    // Build the packed `.stt`. Mirrors `nyc-rideshare --flows`: road-class
    // `min_zoom` LOD (open-ended, no max field — every street ≥ its class zoom),
    // the matrix bin as the temporal bucket, end-time for the whole-range span.
    // Street corridors are multi-vertex real geometry, so tile clipping is fine.
    common::run_stt_build_with_full_options(SttBuildOptions {
        input: intermediate_path.clone(),
        output: args.output.clone(),
        time_field: "timestamp".to_string(),
        end_time_field: Some("end_timestamp".to_string()),
        min_zoom: 8,
        max_zoom: 15,
        compression: "zstd".to_string(),
        temporal_bucket: Some(args.bin.clone()),
        temporal_lod: None,
        summary: None,
        summary_sub_buckets: None,
        min_features_per_tile: None,
        min_zoom_field: Some("min_zoom".to_string()),
        max_zoom_field: None,
        no_clip: false,
        quantize_coords: None,
        quantize_attrs: Vec::new(),
    })?;

    println!("\n✅ BIXI street network built: {}", args.output.display());
    println!("   Set datasets.ts timeRange to {{ start: {bucket0}, end: {range_end} }}");
    Ok(())
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

    /// Emit OD-corridor LineStrings to GeoParquet, each carrying a `[2 ×
    /// num_buckets]` vertex-major matrix (both vertices = the corridor's
    /// per-bucket count). Returns (features, num_buckets, bucket0_ms,
    /// range_end_ms).
    ///
    /// `cluster_radius`:
    /// - `None` → legacy full-resolution emit with a volume-based `min_zoom`
    ///   LOD (busiest corridors city-wide; minor pairs reveal on zoom-in).
    /// - `Some(r)` → flowmap.gl-style build-time clustering: stations are merged
    ///   into hubs per zoom (Supercluster radius `r / (512·2^z)`) and flows
    ///   aggregated between hub pairs. Each zoom's corridors are confined to a
    ///   single-zoom band (`min_zoom == max_zoom == z`) so coarse hubs never
    ///   bleed into the deeper zooms; the deepest zoom is full per-station
    ///   resolution.
    fn write_parquet(
        &self,
        output: &Path,
        min_trips: u32,
        cluster_radius: Option<f64>,
        min_zoom: u8,
        max_zoom: u8,
        bundle: Option<&BundleParams>,
    ) -> Result<(usize, usize, i64, i64)> {
        match cluster_radius {
            Some(r) => {
                self.write_parquet_clustered(output, min_trips, r, min_zoom, max_zoom, bundle)
            }
            None => self.write_parquet_flat(output, min_trips),
        }
    }

    /// Legacy path: one corridor per kept OD pair with a volume-based `min_zoom`
    /// floor and no ceiling (the corridor appears at every zoom ≥ its floor).
    fn write_parquet_flat(
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

    /// Clustered path (flowmap.gl-style baked-in aggregation). See
    /// [`Self::write_parquet`]. Builds a per-zoom station→hub hierarchy once,
    /// then for each zoom aggregates directed flows between hub pairs and emits
    /// one corridor per hub pair, banded to that single zoom.
    ///
    /// When `bundle` is `Some`, each zoom's kept hub-pair corridors are bundled
    /// GLOBALLY (one KDEEB density field over that zoom's whole edge set — never
    /// per tile, which would seam) into smooth multi-vertex rivers and emitted as
    /// `P`-vertex polylines instead of straight 2-vertex arcs. The per-bucket
    /// count series is replicated across all `P` vertices (vertex-major matrix);
    /// the rows are identical so zstd collapses the redundancy on the wire.
    fn write_parquet_clustered(
        &self,
        output: &Path,
        min_trips: u32,
        cluster_radius: f64,
        min_zoom: u8,
        max_zoom: u8,
        bundle: Option<&BundleParams>,
    ) -> Result<(usize, usize, i64, i64)> {
        let property_columns = vec![
            PropertyColumn::numeric("total_count"),
            PropertyColumn::numeric("min_zoom"),
            PropertyColumn::numeric("max_zoom"),
            PropertyColumn::string("origin"),
            PropertyColumn::string("destination"),
        ];
        let mut writer = StreamingLineStringParquetWriter::with_columns(output, property_columns)?;

        if self.counts.is_empty() || max_zoom < min_zoom {
            writer.finish()?;
            return Ok((0, 0, 0, 0));
        }

        let num_buckets = (self.max_bin - self.min_bin + 1) as usize;
        let bucket0 = self.min_bin * self.bin_ms;
        let range_end = bucket0 + num_buckets as i64 * self.bin_ms;

        let hierarchy = ClusterHierarchy::build(self, cluster_radius, min_zoom, max_zoom);

        let pb = ProgressBar::new((max_zoom - min_zoom + 1) as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40}] zoom {pos}/{len}")?
                .progress_chars("=>-"),
        );

        let mut features = 0usize;
        for z in min_zoom..=max_zoom {
            let assign = hierarchy.assignment(z);
            let clusters = hierarchy.clusters(z);

            // Aggregate directed flows between hubs at this zoom (summation is
            // commutative → independent of HashMap iteration order).
            let mut hub_counts: HashMap<(usize, usize), HashMap<i64, u32>> = HashMap::new();
            for ((o_key, d_key), bins) in &self.counts {
                let (co, cd) = (
                    assign[hierarchy.index_of(o_key)],
                    assign[hierarchy.index_of(d_key)],
                );
                if co == cd {
                    continue; // both endpoints collapsed into the same hub
                }
                let acc = hub_counts.entry((co, cd)).or_default();
                for (&bin, &c) in bins {
                    *acc.entry(bin).or_insert(0) += c;
                }
            }

            // Deterministic (content-addressable) emit: sort by (origin id,
            // destination id) then hub indices.
            let names: Vec<String> = clusters.iter().map(|c| hierarchy.rep_key(c)).collect();
            let mut pairs: Vec<&(usize, usize)> = hub_counts.keys().collect();
            pairs.sort_unstable_by(|a, b| {
                names[a.0]
                    .cmp(&names[b.0])
                    .then_with(|| names[a.1].cmp(&names[b.1]))
                    .then_with(|| a.cmp(b))
            });

            // Collect this zoom's KEPT hub-pair corridors (those clearing
            // min_trips) — the complete edge set the bundler relaxes against.
            #[allow(clippy::type_complexity)]
            let mut kept: Vec<([f64; 2], [f64; 2], &HashMap<i64, u32>, u32, String, String)> =
                Vec::new();
            for pair in pairs {
                let bins = &hub_counts[pair];
                let total: u32 = bins.values().copied().sum();
                if total < min_trips {
                    continue;
                }
                kept.push((
                    clusters[pair.0].lonlat,
                    clusters[pair.1].lonlat,
                    bins,
                    total,
                    names[pair.0].clone(),
                    names[pair.1].clone(),
                ));
            }

            // Bundled multi-vertex rivers (one GLOBAL density field over this
            // whole zoom's edge set) or straight 2-vertex arcs.
            let geoms: Vec<Vec<[f64; 2]>> = match bundle {
                Some(bp) => {
                    let edges: Vec<Vec<[f64; 2]>> = kept.iter().map(|k| vec![k.0, k.1]).collect();
                    edge_bundle::bundle_edges(&edges, bp)
                }
                None => kept.iter().map(|k| vec![k.0, k.1]).collect(),
            };

            for (k, coords) in kept.iter().zip(geoms) {
                // Vertex-major matrix: replicate the corridor's per-bucket count
                // across all P vertices (identical rows → zstd collapses them).
                let p = coords.len();
                let mut matrix = vec![0.0f32; p * num_buckets];
                for (&bin, &c) in k.2 {
                    let b = (bin - self.min_bin) as usize;
                    for v in 0..p {
                        matrix[v * num_buckets + b] = c as f32;
                    }
                }

                let mut properties = Map::new();
                properties.insert("total_count".to_string(), json!(k.3));
                // Single-zoom band: the corridor exists at exactly this zoom.
                properties.insert("min_zoom".to_string(), json!(z));
                properties.insert("max_zoom".to_string(), json!(z));
                properties.insert("origin".to_string(), json!(k.4));
                properties.insert("destination".to_string(), json!(k.5));

                writer.write_linestring(&LineStringRecord {
                    coordinates: coords,
                    timestamp_ms: bucket0,
                    end_timestamp_ms: Some(range_end),
                    vertex_timestamps_ms: None,
                    vertex_values: None,
                    vertex_value_matrix: Some(matrix),
                    properties,
                })?;
                features += 1;
            }
            pb.inc(1);
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
        t if t >= 1200 => 10,
        t if t >= 400 => 11,
        t if t >= 100 => 12,
        _ => 13,
    }
}

// ---------------------------------------------------------------------------
// Build-time station clustering (flowmap.gl-style, baked per zoom)
// ---------------------------------------------------------------------------

/// One station in the hierarchy's stable index space.
struct StationEntry {
    key: StationKey,
    lonlat: [f64; 2],
    /// `max(total_out, total_in)` flow — flowmap.gl's location weight; heavy
    /// hubs pull the cluster centroid toward themselves and seed merges first.
    weight: f64,
}

/// A hub at one zoom: the original stations it covers, their summed weight, and
/// the flow-weighted centroid used as the corridor endpoint + node position.
struct ClusterNode {
    members: Vec<usize>,
    weight: f64,
    lonlat: [f64; 2],
}

/// Per-zoom station→hub assignment, built top-down (Supercluster-style): the
/// deepest zoom is the identity (full resolution); each coarser zoom merges the
/// next-finer level's hubs within a zoom-scaled radius, giving stable nesting.
struct ClusterHierarchy {
    stations: Vec<StationEntry>,
    key_to_idx: HashMap<StationKey, usize>,
    /// zoom → hubs
    levels: HashMap<u8, Vec<ClusterNode>>,
    /// zoom → (station index → hub index)
    assign: HashMap<u8, Vec<usize>>,
}

impl ClusterHierarchy {
    fn build(agg: &BixiAggregator, radius: f64, min_zoom: u8, max_zoom: u8) -> Self {
        // Per-station flow totals → weight.
        let mut out_tot: HashMap<&StationKey, f64> = HashMap::new();
        let mut in_tot: HashMap<&StationKey, f64> = HashMap::new();
        for ((o, d), bins) in &agg.counts {
            let t: f64 = bins.values().copied().sum::<u32>() as f64;
            *out_tot.entry(o).or_insert(0.0) += t;
            *in_tot.entry(d).or_insert(0.0) += t;
        }

        // Stable index space: stations sorted by key (deterministic packs).
        let mut keys: Vec<&StationKey> = agg.station_pos.keys().collect();
        keys.sort_unstable();
        let mut stations = Vec::with_capacity(keys.len());
        let mut key_to_idx = HashMap::with_capacity(keys.len());
        for (i, k) in keys.iter().enumerate() {
            let w = out_tot
                .get(*k)
                .copied()
                .unwrap_or(0.0)
                .max(in_tot.get(*k).copied().unwrap_or(0.0))
                .max(1.0);
            stations.push(StationEntry {
                key: (*k).clone(),
                lonlat: agg.station_pos[*k],
                weight: w,
            });
            key_to_idx.insert((*k).clone(), i);
        }

        let n = stations.len();
        let mut levels: HashMap<u8, Vec<ClusterNode>> = HashMap::new();
        let mut assign: HashMap<u8, Vec<usize>> = HashMap::new();

        // Deepest zoom = identity → full per-station resolution.
        let identity: Vec<ClusterNode> = (0..n)
            .map(|s| ClusterNode {
                members: vec![s],
                weight: stations[s].weight,
                lonlat: stations[s].lonlat,
            })
            .collect();
        assign.insert(max_zoom, (0..n).collect());
        levels.insert(max_zoom, identity);

        // Merge downward: each coarser zoom clusters the next-finer hubs.
        for z in (min_zoom..max_zoom).rev() {
            let r = radius / (512.0 * 2f64.powi(z as i32));
            let merged = merge_clusters(&levels[&(z + 1)], r);
            let mut a = vec![0usize; n];
            for (ci, c) in merged.iter().enumerate() {
                for &m in &c.members {
                    a[m] = ci;
                }
            }
            assign.insert(z, a);
            levels.insert(z, merged);
        }

        ClusterHierarchy {
            stations,
            key_to_idx,
            levels,
            assign,
        }
    }

    fn assignment(&self, z: u8) -> &[usize] {
        &self.assign[&z]
    }

    fn clusters(&self, z: u8) -> &[ClusterNode] {
        &self.levels[&z]
    }

    fn index_of(&self, key: &StationKey) -> usize {
        self.key_to_idx[key]
    }

    /// The heaviest member's station key (tie-break: smallest key) — a stable,
    /// human-readable label for the hub used as the corridor origin/destination.
    fn rep_key(&self, node: &ClusterNode) -> String {
        let mut best = node.members[0];
        for &m in &node.members[1..] {
            let (mw, bw) = (self.stations[m].weight, self.stations[best].weight);
            if mw > bw || (mw == bw && self.stations[m].key < self.stations[best].key) {
                best = m;
            }
        }
        self.stations[best].key.clone()
    }
}

/// Greedy single-pass radius merge over `nodes` (à la Supercluster): seed by
/// descending weight, absorb every unvisited node within `r` of the seed's
/// web-mercator unit position into a flow-weighted centroid.
fn merge_clusters(nodes: &[ClusterNode], r: f64) -> Vec<ClusterNode> {
    let n = nodes.len();
    let unit: Vec<[f64; 2]> = nodes.iter().map(|c| project_unit(c.lonlat)).collect();

    // Deterministic seed order: heaviest first, tie-break by smallest member.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        nodes[b]
            .weight
            .partial_cmp(&nodes[a].weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| min_member(&nodes[a]).cmp(&min_member(&nodes[b])))
    });

    let r2 = r * r;
    let mut visited = vec![false; n];
    let mut out = Vec::new();
    for &i in &order {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        let mut members = nodes[i].members.clone();
        let mut wsum = nodes[i].weight;
        let mut lon = nodes[i].lonlat[0] * nodes[i].weight;
        let mut lat = nodes[i].lonlat[1] * nodes[i].weight;
        for &j in &order {
            if visited[j] {
                continue;
            }
            let dx = unit[i][0] - unit[j][0];
            let dy = unit[i][1] - unit[j][1];
            if dx * dx + dy * dy <= r2 {
                visited[j] = true;
                members.extend_from_slice(&nodes[j].members);
                wsum += nodes[j].weight;
                lon += nodes[j].lonlat[0] * nodes[j].weight;
                lat += nodes[j].lonlat[1] * nodes[j].weight;
            }
        }
        members.sort_unstable();
        out.push(ClusterNode {
            members,
            weight: wsum,
            lonlat: [lon / wsum, lat / wsum],
        });
    }
    out
}

fn min_member(node: &ClusterNode) -> usize {
    node.members.iter().copied().min().unwrap_or(0)
}

/// Project lon/lat to web-mercator unit space `[0,1]²` (Supercluster's `lngX`/
/// `latY`), so the merge radius `r = radius/(512·2^z)` scales like screen pixels.
fn project_unit(lonlat: [f64; 2]) -> [f64; 2] {
    let x = lonlat[0] / 360.0 + 0.5;
    let s = (lonlat[1] * std::f64::consts::PI / 180.0)
        .sin()
        .clamp(-0.9999, 0.9999);
    let y = 0.5 - 0.25 * ((1.0 + s) / (1.0 - s)).ln() / std::f64::consts::PI;
    [x, y]
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
        let (features, buckets, bucket0, range_end) =
            agg.write_parquet(&dir, 1, None, 10, 13, None).unwrap();
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
        let (features, _, _, _) = agg.write_parquet(&dir, 3, None, 10, 13, None).unwrap();
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

    /// Build an aggregator with two nearby downtown docks (S1≈S2, ~500 m apart)
    /// both flowing to a far dock S3 (~6 km), all in bin 0.
    fn downtown_agg() -> BixiAggregator {
        let h = 3_600_000i64;
        let mut agg = BixiAggregator::new(h, None, None);
        let s1 = [-73.561, 45.508];
        let s2 = [-73.566, 45.512];
        let s3 = [-73.500, 45.560];
        agg.add_trip(ParsedTrip { origin_key: "S1".into(), dest_key: "S3".into(), origin_pos: s1, dest_pos: s3, start_ms: 0 });
        agg.add_trip(ParsedTrip { origin_key: "S2".into(), dest_key: "S3".into(), origin_pos: s2, dest_pos: s3, start_ms: 0 });
        agg
    }

    #[test]
    fn clustering_is_identity_at_deepest_zoom() {
        let hier = ClusterHierarchy::build(&downtown_agg(), 40.0, 10, 13);
        // Deepest zoom = full resolution: one hub per station, identity assign.
        assert_eq!(hier.clusters(13).len(), 3);
        assert_eq!(hier.assignment(13), &[0, 1, 2]);
    }

    #[test]
    fn clustering_merges_nearby_stations_at_coarse_zoom() {
        let hier = ClusterHierarchy::build(&downtown_agg(), 40.0, 10, 13);
        // The two downtown docks collapse into one hub at low zoom; S3 stays.
        let coarse = hier.clusters(10);
        assert_eq!(coarse.len(), 2);
        let merged = coarse.iter().find(|c| c.members.len() == 2).expect("a 2-member hub");
        // Tie weights → smallest key seeds the hub name.
        assert_eq!(hier.rep_key(merged), "S1");
    }

    #[test]
    fn clustered_write_bands_each_zoom_and_aggregates() {
        let h = 3_600_000i64;
        let dir = std::env::temp_dir().join("bixi-test-cluster.parquet");
        let (features, buckets, bucket0, range_end) =
            downtown_agg().write_parquet(&dir, 1, Some(40.0), 10, 13, None).unwrap();
        // z13+z12 keep both corridors (2+2); z11+z10 aggregate the two docks
        // into one merged→S3 corridor (1+1) → 6 banded features total.
        assert_eq!(features, 6);
        assert_eq!(buckets, 1);
        assert_eq!(bucket0, 0);
        assert_eq!(range_end, h);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn baked_bundling_keeps_feature_banding_with_clustering() {
        // Baking must not change WHICH corridors are emitted per zoom — only
        // their geometry (2 verts → P verts). Same banded count as the
        // unbundled clustered write above.
        let dir = std::env::temp_dir().join("bixi-test-baked.parquet");
        let bp = BundleParams { points: 12, ..Default::default() };
        let (features, buckets, _, _) =
            downtown_agg().write_parquet(&dir, 1, Some(40.0), 10, 13, Some(&bp)).unwrap();
        assert_eq!(features, 6, "baking preserves per-zoom feature banding");
        assert_eq!(buckets, 1);
        let _ = std::fs::remove_file(&dir);
    }
}
