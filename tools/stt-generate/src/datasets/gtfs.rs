//! Country-scale GTFS transit "ballet" generator (`gtfs`).
//!
//! Expands a static [GTFS](https://gtfs.org/) feed for ONE service date into an
//! STT paths archive: every scheduled vehicle journey (trip) becomes a
//! LineString along its shape with per-vertex timestamps interpolated in
//! shape-distance between consecutive stop times — the transit counterpart of
//! `bixi --paths` / `nyc-rideshare --paths`, but with NO routing server: the
//! feed already carries both the geometry (`shapes.txt`) and the timetable
//! (`stop_times.txt`). Rendered with `type: 'trip-heads'`, a whole country's
//! timetable animates as a Mini-Tokyo-3D-style ballet of gliding vehicles.
//!
//! ## Semantics
//! - **Service expansion**: a trip runs iff its `service_id` is active on
//!   `--date` — weekly `calendar.txt` rows (when the file exists) plus
//!   `calendar_dates.txt` exceptions (`exception_type=1` adds, `=2` removes;
//!   removals win). The NL OVapi feed is calendar_dates-only.
//! - **Timing**: each stop is positioned along the trip's shape by
//!   `shape_dist_traveled` (present in both files in the NL feed); shape
//!   vertices between consecutive stops get timestamps linearly interpolated in
//!   shape-distance between the previous stop's *departure* and the next stop's
//!   *arrival*. Dwell (`departure > arrival`) is kept as a duplicated stop
//!   vertex at the two timestamps, so heads visibly pause at stations. Feature
//!   start/end = first departure / last arrival.
//! - **Fallbacks** (per trip, cascading): missing/non-monotone dist columns →
//!   project each stop onto the shape (monotone nearest-point); missing shape →
//!   straight stop-to-stop lines. Trips that survive no stage are dropped with
//!   per-reason accounting (mirroring the stt-build reader's drop-accounting
//!   convention). Shapes are already dense, so nothing is densified or
//!   simplified (same as the other `--paths` generators).
//! - **Clock**: GTFS times may exceed `24:00:00` (`25:30:00` = 01:30 the next
//!   morning) and are anchored at LOCAL MIDNIGHT of the service date in the
//!   feed's agency timezone (`agency.txt`, default `Europe/Amsterdam`), then
//!   emitted as absolute Unix ms. The GTFS spec technically anchors at "noon
//!   minus 12 h" so schedules stay stable across DST changes; that refinement
//!   is deliberately simplified away here — it only differs (by ±1 h, for
//!   times crossing the 02:00 switch) on the two DST-transition nights a year.
//! - **Categorical trap**: `route_type` is emitted as a STRING LABEL
//!   (bus/rail/tram/metro/ferry), never the numeric GTFS code — an all-numeric
//!   string column is promoted to Numeric by stt-build's inference and
//!   categorical `colorMapping` silently no-ops (the storm-radar lesson).
//!
//! ## Memory/streaming
//! `stop_times.txt` (18.9M rows / 1.1 GB for NL) and `shapes.txt` (7.6M rows)
//! are streamed with a reused `ByteRecord`; only rows belonging to the day's
//! active trips/shapes are retained, so a full national day expands in minutes
//! within a modest footprint. Output ordering is fully deterministic
//! (trips sorted by start time then `trip_id`; no HashMap iteration order
//! reaches the output).
//!
//! ```bash
//! # The busiest fully-defined NL service date (Friday, 121k trips):
//! stt-generate gtfs --feed data/gtfs-nl/feed --date 20260703 \
//!   --output examples/showcase/public/data/gtfs-nl
//! ```

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Utc, Weekday};
use chrono_tz::Tz;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Map};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use crate::common::{self, LineStringRecord, PropertyColumn, StreamingLineStringParquetWriter};
use crate::datasets::bixi::resolve_intermediate;
use crate::datasets::nyc_rideshare_flows::parse_bin_ms;

#[derive(Parser, Debug)]
#[command(
    about = "Generate a country-scale transit dataset from a static GTFS feed \
                   (every trip scheduled on one service date as a timestamped trajectory)"
)]
pub struct Args {
    /// Extracted GTFS feed directory (needs trips.txt, stop_times.txt,
    /// routes.txt, calendar_dates.txt and/or calendar.txt; shapes.txt and
    /// stops.txt are used when present).
    #[arg(long)]
    pub feed: PathBuf,

    /// Service date to expand, `YYYYMMDD`. For the bundled NL OVapi feed the
    /// busiest fully-defined date is `20260703` (a Friday, ~121k trips).
    #[arg(long)]
    pub date: String,

    /// Output packed `.stt` directory (or `*.parquet` to stop at the
    /// intermediate). `--out` is accepted as an alias.
    #[arg(
        short,
        long,
        alias = "out",
        default_value = "examples/showcase/public/data/gtfs-transit"
    )]
    pub output: PathBuf,

    /// Temporal bucket for the output archive's tile chunking (e.g. `1h`, `30m`).
    #[arg(long, default_value = "1h")]
    pub bin: String,

    /// Cap the number of trips, evenly subsampled across the day for a legible
    /// moving fleet (deterministic stride). Default: every scheduled trip.
    #[arg(long)]
    pub max_trips: Option<usize>,

    /// Also emit each trip's `trip_headsign` as a string property (off by
    /// default — it is a high-cardinality categorical).
    #[arg(long)]
    pub headsign: bool,

    /// Build pyramid min zoom (country overview).
    #[arg(long, default_value = "6")]
    pub min_zoom: u8,

    /// Build pyramid max zoom.
    #[arg(long, default_value = "14")]
    pub max_zoom: u8,

    /// Skip the stt-build step (write only the intermediate GeoParquet).
    #[arg(long)]
    pub skip_build: bool,
}

pub fn run(args: Args) -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         🚆 GTFS Transit Ballet Generator (static feed)        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Validate the temporal bucket early (same helper the other generators use).
    parse_bin_ms(&args.bin)?;
    if args.min_zoom > args.max_zoom {
        return Err(anyhow!(
            "--min-zoom ({}) must be ≤ --max-zoom ({})",
            args.min_zoom,
            args.max_zoom
        ));
    }

    println!("📋 Configuration:");
    println!("   Feed:    {}", args.feed.display());
    println!("   Date:    {}", args.date);
    println!("   Output:  {}", args.output.display());
    println!("   Bin:     {}", args.bin);
    println!("   Zoom:    {}-{}", args.min_zoom, args.max_zoom);
    if let Some(max) = args.max_trips {
        println!("   Max trips: {max}");
    }
    println!();

    let (intermediate_path, output_is_intermediate) = resolve_intermediate(&args.output);
    let mut property_columns = vec![
        PropertyColumn::numeric("trip_id"),
        PropertyColumn::string("route_type"),
        PropertyColumn::string("route_short_name"),
        PropertyColumn::string("agency_id"),
    ];
    if args.headsign {
        property_columns.push(PropertyColumn::string("headsign"));
    }
    let mut writer =
        StreamingLineStringParquetWriter::with_columns(&intermediate_path, property_columns)?;

    let include_headsign = args.headsign;
    let stats = expand_feed(
        &args.feed,
        &args.date,
        &ExpandOptions {
            max_trips: args.max_trips,
        },
        &mut |f: TripFeature| {
            let (Some(start_dt), Some(end_dt)) = (
                DateTime::<Utc>::from_timestamp_millis(f.start_ms),
                DateTime::<Utc>::from_timestamp_millis(f.end_ms),
            ) else {
                return Ok(());
            };
            let mut props = Map::new();
            props.insert("trip_id".to_string(), json!(f.seq));
            props.insert("route_type".to_string(), json!(f.route_type));
            props.insert("route_short_name".to_string(), json!(f.route_short_name));
            props.insert("agency_id".to_string(), json!(f.agency_id));
            if include_headsign && !f.headsign.is_empty() {
                props.insert("headsign".to_string(), json!(f.headsign));
            }
            let mut record = LineStringRecord::new(f.coords, start_dt, Some(end_dt), props);
            record.vertex_timestamps_ms = Some(f.times_ms);
            writer.write_linestring(&record)
        },
    )?;
    let rows = writer.finish()?;

    println!(
        "\n   ✓ {} of {} scheduled trips expanded across {} active services \
         ({} shape-dist timed, {} shape-projected, {} stop-to-stop; {rows} rows written)",
        stats.trips_built,
        stats.trips_scheduled,
        stats.services_active,
        stats.via_shape_dist,
        stats.via_projection,
        stats.via_stops
    );
    let dropped = stats.dropped_no_stop_times
        + stats.dropped_few_anchors
        + stats.dropped_bad_times
        + stats.dropped_degenerate
        + stats.dropped_no_route;
    if dropped > 0 {
        println!(
            "   ⚠ dropped {dropped} trips: {} no stop_times, {} <2 usable stops, \
             {} non-monotone times, {} degenerate geometry, {} route_id not in routes.txt",
            stats.dropped_no_stop_times,
            stats.dropped_few_anchors,
            stats.dropped_bad_times,
            stats.dropped_degenerate,
            stats.dropped_no_route
        );
    }
    if stats.stop_rows_skipped > 0 || stats.stop_rows_unknown_stop > 0 {
        println!(
            "   ⚠ stop_times rows: {} skipped (unparsable seq/times), {} referencing unknown stop ids",
            stats.stop_rows_skipped, stats.stop_rows_unknown_stop
        );
    }
    if rows == 0 {
        return Err(anyhow!(
            "No usable trips on {} — is the date within the feed's calendar span?",
            args.date
        ));
    }
    println!("   ⏱  archive time span (set showcase timeRange to this):");
    println!("       start = {}", stats.span_min_ms);
    println!("       end   = {}", stats.span_max_ms);

    if args.skip_build || output_is_intermediate {
        println!(
            "\n📦 Skipping stt-build (intermediate written to {})",
            intermediate_path.display()
        );
        return Ok(());
    }

    // Build the packed `.stt`. Mirrors `bixi --paths`: `timestamp` +
    // `end_timestamp` for the per-trip window, real multi-vertex geometry (tile
    // clipping is fine), the bin as the temporal bucket, and the per-vertex
    // `vertex_timestamps` column riding through from the LineStringRecord. Zoom
    // range defaults to a country-scale pyramid (6-14) instead of bixi's
    // city-scale 10-16.
    common::run_stt_build_with_options(
        &intermediate_path,
        &args.output,
        "timestamp",
        Some("end_timestamp"),
        args.min_zoom,
        args.max_zoom,
        "zstd",
        Some(&args.bin),
    )?;

    println!("\n✅ GTFS transit archive built: {}", args.output.display());
    println!(
        "   Set datasets.ts timeRange to {{ start: {}, end: {} }}",
        stats.span_min_ms, stats.span_max_ms
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Feed expansion
// ---------------------------------------------------------------------------

pub(crate) struct ExpandOptions {
    pub max_trips: Option<usize>,
}

/// How a trip's geometry+timing was derived, in fallback order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GeomKind {
    /// Stops matched to the shape via `shape_dist_traveled` in both files.
    ShapeDist,
    /// Stops projected onto the shape (monotone nearest-point) because the
    /// dist columns were missing/unusable.
    ShapeProjected,
    /// No usable shape — straight stop-to-stop segments.
    StopToStop,
}

/// One expanded trip, ready to write. `seq` is the deterministic index in the
/// day's (start-time, trip_id)-sorted order — written as the numeric `trip_id`
/// property (the GTFS string id stays out of the tiles; it is 100k-cardinality).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TripFeature {
    /// GTFS trip_id, for diagnostics/tests — not written to the tiles
    /// (100k-cardinality string).
    #[allow(dead_code)]
    pub gtfs_trip_id: String,
    pub seq: usize,
    pub coords: Vec<[f64; 2]>,
    pub times_ms: Vec<i64>,
    pub start_ms: i64,
    pub end_ms: i64,
    pub route_type: &'static str,
    pub route_short_name: String,
    pub agency_id: String,
    pub headsign: String,
    /// Which fallback stage produced the geometry — asserted by tests; the
    /// aggregate counts are what run() reports.
    #[allow(dead_code)]
    pub kind: GeomKind,
}

/// Expansion counters, in the spirit of the stt-build reader's drop accounting:
/// every scheduled trip is either built (per-geometry-kind counts) or dropped
/// for a named reason — nothing disappears silently.
#[derive(Debug)]
pub(crate) struct ExpandStats {
    pub services_active: usize,
    pub trips_scheduled: usize,
    pub trips_built: usize,
    pub via_shape_dist: usize,
    pub via_projection: usize,
    pub via_stops: usize,
    pub dropped_no_route: usize,
    pub dropped_no_stop_times: usize,
    pub dropped_few_anchors: usize,
    pub dropped_bad_times: usize,
    pub dropped_degenerate: usize,
    /// stop_times rows within kept trips skipped for unparsable seq/times.
    pub stop_rows_skipped: usize,
    /// stop_times rows referencing a stop_id absent from stops.txt (kept as
    /// timing anchors for dist matching; unusable for projection/stop-to-stop).
    pub stop_rows_unknown_stop: usize,
    pub span_min_ms: i64,
    pub span_max_ms: i64,
}

impl Default for ExpandStats {
    fn default() -> Self {
        Self {
            services_active: 0,
            trips_scheduled: 0,
            trips_built: 0,
            via_shape_dist: 0,
            via_projection: 0,
            via_stops: 0,
            dropped_no_route: 0,
            dropped_no_stop_times: 0,
            dropped_few_anchors: 0,
            dropped_bad_times: 0,
            dropped_degenerate: 0,
            stop_rows_skipped: 0,
            stop_rows_unknown_stop: 0,
            span_min_ms: i64::MAX,
            span_max_ms: i64::MIN,
        }
    }
}

struct RouteMeta {
    label: &'static str,
    short_name: String,
    agency_id: String,
}

struct TripMeta {
    trip_id: String,
    route: u32,
    shape: Option<u32>,
    headsign: String,
}

/// One stop_times row of a kept trip. Times are already absolute Unix ms
/// (midnight-anchored, so >24:00:00 rows land on the next morning).
struct StopEvent {
    seq: u32,
    arrival_ms: i64,
    departure_ms: i64,
    dist: Option<f64>,
    /// Index into the stop-position table; `None` = unknown stop_id.
    stop: Option<u32>,
}

/// A trip shape: deduped vertices, computed cumulative meters (`cum`, always
/// available — the projection/`pos_at` axis) and, when the file's
/// `shape_dist_traveled` is complete and monotone, that axis too (`file_dist`,
/// the axis stop dists reference).
struct Shape {
    pts: Vec<[f64; 2]>,
    cum: Vec<f64>,
    file_dist: Option<Vec<f64>>,
}

/// Expand one service date of a GTFS feed into [`TripFeature`]s, invoking
/// `sink` once per built trip in deterministic (start-time, trip_id) order.
pub(crate) fn expand_feed(
    feed: &Path,
    date: &str,
    opts: &ExpandOptions,
    sink: &mut dyn FnMut(TripFeature) -> Result<()>,
) -> Result<ExpandStats> {
    NaiveDate::parse_from_str(date.trim(), "%Y%m%d")
        .with_context(|| format!("invalid --date '{date}' (expected YYYYMMDD)"))?;
    let tz = feed_timezone(feed);
    let midnight_ms = service_midnight_ms(date, tz)?;
    println!(
        "🕛 Service date {date} anchored at local midnight ({})",
        tz.name()
    );

    let mut stats = ExpandStats::default();

    // 1. Which services run on the date (calendar.txt weekly rows when present,
    //    then calendar_dates exceptions; type-2 removals win).
    let services = active_services(feed, date)?;
    stats.services_active = services.len();
    println!("📅 {} service_ids active on {date}", services.len());
    if services.is_empty() {
        return Err(anyhow!(
            "no services active on {date} — is the date within the feed's calendar span?"
        ));
    }

    // 2. Routes (small): route_id → categorical label + display columns.
    let (route_index, routes) = load_routes(feed)?;

    // 3. Trips of the day: trip_id → route/shape/headsign. Shape ids are
    //    interned so the shapes pass below keeps only what the day needs.
    let mut trip_index: HashMap<String, u32> = HashMap::new();
    let mut trips: Vec<TripMeta> = Vec::new();
    let mut shape_index: HashMap<String, u32> = HashMap::new();
    {
        println!("📂 Reading trips.txt …");
        let mut rdr = open_csv(&feed.join("trips.txt"))?;
        let h = rdr.byte_headers()?.clone();
        let i_route = req_col(&h, "route_id", "trips.txt")?;
        let i_service = req_col(&h, "service_id", "trips.txt")?;
        let i_trip = req_col(&h, "trip_id", "trips.txt")?;
        let i_shape = col(&h, "shape_id");
        let i_headsign = col(&h, "trip_headsign");
        let mut rec = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut rec)? {
            if !services.contains(field(&rec, i_service)) {
                continue;
            }
            let trip_id = field(&rec, i_trip);
            if trip_id.is_empty() {
                continue;
            }
            let Some(&route) = route_index.get(field(&rec, i_route)) else {
                stats.dropped_no_route += 1;
                continue;
            };
            let shape = match i_shape.map(|i| field(&rec, i)) {
                Some(s) if !s.is_empty() => Some(match shape_index.get(s) {
                    Some(&ix) => ix,
                    None => {
                        let ix = shape_index.len() as u32;
                        shape_index.insert(s.to_string(), ix);
                        ix
                    }
                }),
                _ => None,
            };
            let headsign = i_headsign
                .map(|i| field(&rec, i).to_string())
                .unwrap_or_default();
            trip_index.insert(trip_id.to_string(), trips.len() as u32);
            trips.push(TripMeta {
                trip_id: trip_id.to_string(),
                route,
                shape,
                headsign,
            });
        }
    }
    stats.trips_scheduled = trips.len();
    println!(
        "   ✓ {} trips scheduled on {date} ({} shapes referenced)",
        trips.len(),
        shape_index.len()
    );
    if trips.is_empty() {
        return Err(anyhow!("no trips scheduled on {date}"));
    }

    // 4. Stop positions (for the projection / stop-to-stop fallbacks).
    let (stop_positions, stop_ids) = load_stops(feed)?;

    // 5. stop_times, streamed: the file is trip-grouped in practice, so the
    //    active-trip lookup is cached per run of equal trip_ids — but
    //    correctness does not depend on grouping (an ungrouped file just pays
    //    one lookup per row).
    let mut events: Vec<Vec<StopEvent>> = trips.iter().map(|_| Vec::new()).collect();
    {
        println!("📂 Reading stop_times.txt (streaming) …");
        let mut rdr = open_csv(&feed.join("stop_times.txt"))?;
        let h = rdr.byte_headers()?.clone();
        let i_trip = req_col(&h, "trip_id", "stop_times.txt")?;
        let i_seq = req_col(&h, "stop_sequence", "stop_times.txt")?;
        let i_stop = req_col(&h, "stop_id", "stop_times.txt")?;
        let i_arr = req_col(&h, "arrival_time", "stop_times.txt")?;
        let i_dep = req_col(&h, "departure_time", "stop_times.txt")?;
        let i_dist = col(&h, "shape_dist_traveled");
        let mut rec = csv::ByteRecord::new();
        let mut cur_key: Vec<u8> = Vec::new();
        let mut cur: Option<u32> = None;
        let mut first = true;
        let mut rows_seen: u64 = 0;
        while rdr.read_byte_record(&mut rec)? {
            rows_seen += 1;
            if rows_seen % 5_000_000 == 0 {
                println!("   … {}M rows scanned", rows_seen / 1_000_000);
            }
            let tid = rec.get(i_trip).unwrap_or(b"");
            if first || tid != cur_key.as_slice() {
                first = false;
                cur_key.clear();
                cur_key.extend_from_slice(tid);
                cur = std::str::from_utf8(tid)
                    .ok()
                    .and_then(|s| trip_index.get(s.trim()).copied());
            }
            let Some(ti) = cur else { continue };
            let seq = field(&rec, i_seq).parse::<u32>().ok();
            let arr = parse_gtfs_time_seconds(field(&rec, i_arr));
            let dep = parse_gtfs_time_seconds(field(&rec, i_dep));
            // GTFS allows one blank side on non-timepoints; mirror the missing
            // side. Both blank (or a bad sequence) = unusable as an anchor.
            let (Some(seq), Some((arr_s, dep_s))) = (
                seq,
                match (arr, dep) {
                    (Some(a), Some(d)) => Some((a, d)),
                    (Some(a), None) => Some((a, a)),
                    (None, Some(d)) => Some((d, d)),
                    (None, None) => None,
                },
            ) else {
                stats.stop_rows_skipped += 1;
                continue;
            };
            let stop = {
                let sid = field(&rec, i_stop);
                match stop_ids.get(sid) {
                    Some(&ix) => Some(ix),
                    None => {
                        if !sid.is_empty() {
                            stats.stop_rows_unknown_stop += 1;
                        }
                        None
                    }
                }
            };
            let dist = i_dist.and_then(|i| field(&rec, i).parse::<f64>().ok());
            events[ti as usize].push(StopEvent {
                seq,
                arrival_ms: midnight_ms + arr_s * 1000,
                departure_ms: midnight_ms + dep_s * 1000,
                dist,
                stop,
            });
        }
        let kept: usize = events.iter().map(Vec::len).sum();
        println!("   ✓ {kept} stop_time rows kept for the day's trips");
    }

    // 6. Shapes, streamed, keeping only the day's shape ids.
    let shapes = load_shapes(feed, &shape_index)?;

    // 7. Deterministic order: (first departure, trip_id). The key is derived
    //    before building so the optional --max-trips stride is a uniform
    //    temporal slice of the day, exactly like bixi --paths.
    let mut order: Vec<(i64, u32)> = Vec::with_capacity(trips.len());
    for (ti, evs) in events.iter_mut().enumerate() {
        if evs.is_empty() {
            stats.dropped_no_stop_times += 1;
            continue;
        }
        evs.sort_by_key(|e| e.seq);
        order.push((evs.first().map(|e| e.departure_ms).unwrap_or(0), ti as u32));
    }
    order.sort_by(|a, b| {
        a.0.cmp(&b.0).then_with(|| {
            trips[a.1 as usize]
                .trip_id
                .cmp(&trips[b.1 as usize].trip_id)
        })
    });
    if let Some(max) = opts.max_trips {
        if max > 0 && order.len() > max {
            let n = order.len();
            let sampled: Vec<(i64, u32)> = (0..max).map(|i| order[i * n / max]).collect();
            println!(
                "   📊 Subsampled {n} → {} trips (even temporal stride)",
                sampled.len()
            );
            order = sampled;
        }
    }

    // 8. Build + emit each trip.
    println!("🧵 Expanding {} trips …", order.len());
    let pb = ProgressBar::new(order.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("   [{bar:40}] {pos}/{len} trips ({eta})")?
            .progress_chars("=>-"),
    );
    for (seq, &(_, ti)) in order.iter().enumerate() {
        pb.inc(1);
        let t = &trips[ti as usize];
        let shape = t.shape.and_then(|si| shapes[si as usize].as_ref());
        match build_trip(&mut events[ti as usize], shape, &stop_positions) {
            Ok((coords, times_ms, kind)) => {
                let r = &routes[t.route as usize];
                stats.trips_built += 1;
                match kind {
                    GeomKind::ShapeDist => stats.via_shape_dist += 1,
                    GeomKind::ShapeProjected => stats.via_projection += 1,
                    GeomKind::StopToStop => stats.via_stops += 1,
                }
                let start_ms = *times_ms.first().expect("built trip has vertices");
                let end_ms = *times_ms.last().expect("built trip has vertices");
                stats.span_min_ms = stats.span_min_ms.min(start_ms);
                stats.span_max_ms = stats.span_max_ms.max(end_ms);
                sink(TripFeature {
                    gtfs_trip_id: t.trip_id.clone(),
                    seq,
                    coords,
                    times_ms,
                    start_ms,
                    end_ms,
                    route_type: r.label,
                    route_short_name: r.short_name.clone(),
                    agency_id: r.agency_id.clone(),
                    headsign: t.headsign.clone(),
                    kind,
                })?;
            }
            Err(DropReason::FewAnchors) => stats.dropped_few_anchors += 1,
            Err(DropReason::BadTimes) => stats.dropped_bad_times += 1,
            Err(DropReason::Degenerate) => stats.dropped_degenerate += 1,
        }
    }
    pb.finish_and_clear();

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Per-trip geometry + timing
// ---------------------------------------------------------------------------

enum DropReason {
    /// Fewer than two usable timing anchors.
    FewAnchors,
    /// Stop times run backwards across stops.
    BadTimes,
    /// No spatial extent / zero duration — nothing to animate.
    Degenerate,
}

/// Turn one trip's sorted stop events into a timed LineString, cascading
/// shape-dist matching → shape projection → stop-to-stop.
fn build_trip(
    events: &mut Vec<StopEvent>,
    shape: Option<&Shape>,
    stop_pos: &[[f64; 2]],
) -> Result<(Vec<[f64; 2]>, Vec<i64>, GeomKind), DropReason> {
    events.sort_by_key(|e| e.seq);
    events.dedup_by_key(|e| e.seq);
    if events.len() < 2 {
        return Err(DropReason::FewAnchors);
    }
    // Per-stop: a departure can't precede its arrival (clamp — common data
    // noise). Across stops: times must be non-decreasing or the trip is
    // untimeable.
    let mut prev_dep = i64::MIN;
    for e in events.iter_mut() {
        if e.departure_ms < e.arrival_ms {
            e.departure_ms = e.arrival_ms;
        }
        if e.arrival_ms < prev_dep {
            return Err(DropReason::BadTimes);
        }
        prev_dep = e.departure_ms;
    }
    let trip_start = events.first().map(|e| e.departure_ms).unwrap_or(0);
    let trip_end = events.last().map(|e| e.arrival_ms).unwrap_or(0);
    if trip_end <= trip_start {
        return Err(DropReason::Degenerate);
    }

    if let Some(sh) = shape {
        // Preferred: both files carry shape_dist_traveled — match on distance.
        if let Some(fd) = &sh.file_dist {
            if let Some(anchors) = dist_anchors(events, fd) {
                let (c, t) = emit_along_shape(&sh.pts, fd, &anchors);
                if let Some((c, t)) = finalize(c, t) {
                    return Ok((c, t, GeomKind::ShapeDist));
                }
            }
        }
        // Fallback: project stops onto the shape (needs stops.txt positions).
        if let Some(anchors) = projected_anchors(events, sh, stop_pos) {
            let (c, t) = emit_along_shape(&sh.pts, &sh.cum, &anchors);
            if let Some((c, t)) = finalize(c, t) {
                return Ok((c, t, GeomKind::ShapeProjected));
            }
        }
    }

    // Last resort: straight stop-to-stop segments.
    let known: Vec<(&StopEvent, [f64; 2])> = events
        .iter()
        .filter_map(|e| e.stop.map(|s| (e, stop_pos[s as usize])))
        .collect();
    if known.len() < 2 {
        return Err(DropReason::FewAnchors);
    }
    let mut out = VertexSink::default();
    let last = known.len() - 1;
    for (k, (e, p)) in known.iter().enumerate() {
        if k == 0 {
            out.push(*p, e.departure_ms);
        } else {
            out.push(*p, e.arrival_ms);
            if k < last && e.departure_ms > e.arrival_ms {
                out.push(*p, e.departure_ms); // dwell
            }
        }
    }
    match finalize(out.coords, out.times) {
        Some((c, t)) => Ok((c, t, GeomKind::StopToStop)),
        None => Err(DropReason::Degenerate),
    }
}

/// Anchor = (position along the axis, arrival ms, departure ms) per stop.
type Anchor = (f64, i64, i64);

/// Build anchors from the file's dist columns: every stop must carry a dist,
/// clamped into the shape's dist range, non-decreasing, with positive extent.
/// `None` = fall back to projection.
fn dist_anchors(events: &[StopEvent], axis: &[f64]) -> Option<Vec<Anchor>> {
    let lo = *axis.first()?;
    let hi = *axis.last()?;
    let mut out: Vec<Anchor> = Vec::with_capacity(events.len());
    let mut prev = f64::NEG_INFINITY;
    for e in events {
        let s = e.dist?.clamp(lo, hi);
        if s < prev {
            return None;
        }
        prev = s;
        out.push((s, e.arrival_ms, e.departure_ms));
    }
    (out.last()?.0 > out.first()?.0).then_some(out)
}

/// Build anchors by projecting each known stop onto the shape's computed
/// cumulative-meters axis, monotone (each stop lands at or after the previous
/// one along the shape). Unknown stops are skipped; needs ≥2 survivors with
/// positive extent.
fn projected_anchors(
    events: &[StopEvent],
    sh: &Shape,
    stop_pos: &[[f64; 2]],
) -> Option<Vec<Anchor>> {
    let mut out: Vec<Anchor> = Vec::with_capacity(events.len());
    let mut seg = 0usize;
    let mut prev_s = f64::NEG_INFINITY;
    for e in events {
        let Some(si) = e.stop else { continue };
        let (jseg, mut s) = project_onto(&sh.pts, &sh.cum, stop_pos[si as usize], seg);
        if s < prev_s {
            s = prev_s;
        }
        out.push((s, e.arrival_ms, e.departure_ms));
        seg = jseg;
        prev_s = s;
    }
    (out.len() >= 2 && out.last()?.0 > out.first()?.0).then_some(out)
}

/// Vertex accumulator: keeps timestamps non-decreasing (rounding guard) and
/// drops exactly-duplicate consecutive vertices (same coord AND same time —
/// dwell pairs, same coord at two times, are intentionally kept).
#[derive(Default)]
struct VertexSink {
    coords: Vec<[f64; 2]>,
    times: Vec<i64>,
}

impl VertexSink {
    fn push(&mut self, c: [f64; 2], t: i64) {
        let t = self.times.last().map_or(t, |&last| t.max(last));
        if let (Some(lc), Some(&lt)) = (self.coords.last(), self.times.last()) {
            if lc[0] == c[0] && lc[1] == c[1] && lt == t {
                return;
            }
        }
        self.coords.push(c);
        self.times.push(t);
    }
}

/// Walk the shape between consecutive anchors, emitting every shape vertex
/// strictly between them with a timestamp linearly interpolated in
/// shape-distance from the previous stop's departure to the next stop's
/// arrival, plus the stop vertices themselves (with a duplicated dwell vertex
/// when a stop's departure is later than its arrival). The trip starts at the
/// first stop's departure and ends at the last stop's arrival, so leading and
/// trailing dwell never pads the animation.
fn emit_along_shape(
    pts: &[[f64; 2]],
    axis: &[f64],
    anchors: &[Anchor],
) -> (Vec<[f64; 2]>, Vec<i64>) {
    let mut out = VertexSink::default();
    let (s0, _, dep0) = anchors[0];
    out.push(pos_at(pts, axis, s0), dep0);
    // First shape vertex strictly beyond the first anchor.
    let mut j = axis.partition_point(|&d| d <= s0);
    for k in 1..anchors.len() {
        let (sa, _, da) = anchors[k - 1];
        let (sb, ab, db) = anchors[k];
        while j < pts.len() && axis[j] < sb {
            if axis[j] > sa {
                // Loop conditions guarantee sb > axis[j] > sa, so sb - sa > 0.
                let f = (axis[j] - sa) / (sb - sa);
                out.push(pts[j], da + ((ab - da) as f64 * f).round() as i64);
            }
            j += 1;
        }
        out.push(pos_at(pts, axis, sb), ab);
        if k + 1 < anchors.len() && db > ab {
            out.push(pos_at(pts, axis, sb), db); // dwell: same coord, later ts
        }
    }
    (out.coords, out.times)
}

/// Interpolated position along a polyline at axis value `s` (clamped).
fn pos_at(pts: &[[f64; 2]], axis: &[f64], s: f64) -> [f64; 2] {
    if s <= axis[0] {
        return pts[0];
    }
    if s >= axis[axis.len() - 1] {
        return pts[pts.len() - 1];
    }
    let hi = axis.partition_point(|&d| d < s);
    let lo = hi - 1;
    let span = axis[hi] - axis[lo];
    let f = if span > 0.0 {
        (s - axis[lo]) / span
    } else {
        0.0
    };
    let a = pts[lo];
    let b = pts[hi];
    [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]
}

/// Nearest point on the polyline to `p`, searching segments `from_seg..`
/// (monotone stop ordering). Returns (segment index, cumulative-axis value).
/// Distances in local equirectangular meters — plenty for "which segment".
fn project_onto(pts: &[[f64; 2]], cum: &[f64], p: [f64; 2], from_seg: usize) -> (usize, f64) {
    let kx = 111_320.0 * p[1].to_radians().cos();
    let ky = 110_540.0;
    let start = from_seg.min(pts.len().saturating_sub(2));
    let mut best_d2 = f64::INFINITY;
    let mut best = (start, cum[start]);
    for j in start..pts.len() - 1 {
        let a = pts[j];
        let b = pts[j + 1];
        let ax = (a[0] - p[0]) * kx;
        let ay = (a[1] - p[1]) * ky;
        let dx = (b[0] - a[0]) * kx;
        let dy = (b[1] - a[1]) * ky;
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (-(ax * dx + ay * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let px = ax + dx * t;
        let py = ay + dy * t;
        let d2 = px * px + py * py;
        if d2 < best_d2 {
            best_d2 = d2;
            best = (j, cum[j] + (cum[j + 1] - cum[j]) * t);
        }
    }
    best
}

/// Geometry hygiene gate: ≥2 vertices, at least two distinct positions, and a
/// positive time span — otherwise the caller tries the next fallback (or drops).
fn finalize(coords: Vec<[f64; 2]>, times: Vec<i64>) -> Option<(Vec<[f64; 2]>, Vec<i64>)> {
    if coords.len() < 2 {
        return None;
    }
    let first = coords[0];
    if !coords.iter().any(|c| c[0] != first[0] || c[1] != first[1]) {
        return None;
    }
    (*times.last()? > *times.first()?).then_some((coords, times))
}

// ---------------------------------------------------------------------------
// Feed-table loading
// ---------------------------------------------------------------------------

fn open_csv(path: &Path) -> Result<csv::Reader<BufReader<File>>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    Ok(csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(BufReader::with_capacity(1 << 20, f)))
}

/// Header lookup, tolerant of a UTF-8 BOM and padding (both common in GTFS).
fn col(headers: &csv::ByteRecord, name: &str) -> Option<usize> {
    headers.iter().position(|h| {
        std::str::from_utf8(h)
            .map(|s| {
                s.trim_start_matches('\u{feff}')
                    .trim()
                    .eq_ignore_ascii_case(name)
            })
            .unwrap_or(false)
    })
}

fn req_col(headers: &csv::ByteRecord, name: &str, file: &str) -> Result<usize> {
    col(headers, name).ok_or_else(|| anyhow!("{file} is missing required column '{name}'"))
}

fn field<'a>(rec: &'a csv::ByteRecord, i: usize) -> &'a str {
    std::str::from_utf8(rec.get(i).unwrap_or(b""))
        .unwrap_or("")
        .trim()
}

/// Parse a GTFS `HH:MM:SS` clock time into seconds since local midnight.
/// Hours legitimately exceed 24 for after-midnight service (25:30:00 = 01:30
/// the next morning) and may be single-digit.
pub(crate) fn parse_gtfs_time_seconds(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut it = s.split(':');
    let h: i64 = it.next()?.trim().parse().ok()?;
    let m: i64 = it.next()?.trim().parse().ok()?;
    let sec: i64 = it.next()?.trim().parse().ok()?;
    if it.next().is_some() || h < 0 || !(0..60).contains(&m) || !(0..60).contains(&sec) {
        return None;
    }
    Some(h * 3600 + m * 60 + sec)
}

/// GTFS `route_type` code → a categorical STRING label. Deliberately never the
/// numeric code: an all-numeric string column gets promoted to Numeric by
/// stt-build's inference and categorical `colorMapping` silently no-ops.
pub(crate) fn route_type_label(code: &str) -> &'static str {
    match code.trim() {
        "0" => "tram",
        "1" => "metro",
        "2" => "rail",
        "3" => "bus",
        "4" => "ferry",
        "5" => "cable-tram",
        "6" => "gondola",
        "7" => "funicular",
        "11" => "trolleybus",
        "12" => "monorail",
        _ => "other",
    }
}

/// The feed's local clock: first `agency_timezone` in agency.txt, defaulting
/// to Europe/Amsterdam (the bundled NL feed) when the file/column is absent.
fn feed_timezone(feed: &Path) -> Tz {
    let fallback = chrono_tz::Europe::Amsterdam;
    let path = feed.join("agency.txt");
    let Ok(mut rdr) = open_csv(&path) else {
        println!("   ⚠ agency.txt unreadable — assuming {}", fallback.name());
        return fallback;
    };
    let Ok(h) = rdr.byte_headers().cloned() else {
        return fallback;
    };
    let Some(i_tz) = col(&h, "agency_timezone") else {
        println!(
            "   ⚠ agency.txt has no agency_timezone — assuming {}",
            fallback.name()
        );
        return fallback;
    };
    let mut rec = csv::ByteRecord::new();
    while let Ok(true) = rdr.read_byte_record(&mut rec) {
        let raw = field(&rec, i_tz);
        if raw.is_empty() {
            continue;
        }
        match raw.parse::<Tz>() {
            Ok(tz) => return tz,
            Err(_) => {
                println!(
                    "   ⚠ unrecognized agency_timezone '{raw}' — assuming {}",
                    fallback.name()
                );
                return fallback;
            }
        }
    }
    fallback
}

/// Local midnight of the service date in the feed timezone, as Unix ms. (The
/// GTFS "noon minus 12 h" anchor is simplified to plain local midnight — see
/// the module docs; NL DST transitions happen at 02:00, never at midnight, so
/// `earliest()` is a formality.)
pub(crate) fn service_midnight_ms(date: &str, tz: Tz) -> Result<i64> {
    let d = NaiveDate::parse_from_str(date.trim(), "%Y%m%d")
        .with_context(|| format!("invalid date '{date}' (expected YYYYMMDD)"))?;
    let naive = d.and_hms_opt(0, 0, 0).expect("00:00:00 is a valid time");
    tz.from_local_datetime(&naive)
        .earliest()
        .map(|dt| dt.timestamp_millis())
        .ok_or_else(|| anyhow!("midnight does not exist on {date} in {}", tz.name()))
}

/// Services active on the date: weekly `calendar.txt` rows (when the file
/// exists) plus `calendar_dates.txt` exceptions. Type-2 removals are applied
/// last so they win regardless of file order.
pub(crate) fn active_services(feed: &Path, date: &str) -> Result<HashSet<String>> {
    let date = date.trim();
    let mut active: HashSet<String> = HashSet::new();

    let cal = feed.join("calendar.txt");
    if cal.exists() {
        let d = NaiveDate::parse_from_str(date, "%Y%m%d")
            .with_context(|| format!("invalid date '{date}' (expected YYYYMMDD)"))?;
        let weekday_col = match d.weekday() {
            Weekday::Mon => "monday",
            Weekday::Tue => "tuesday",
            Weekday::Wed => "wednesday",
            Weekday::Thu => "thursday",
            Weekday::Fri => "friday",
            Weekday::Sat => "saturday",
            Weekday::Sun => "sunday",
        };
        let mut rdr = open_csv(&cal)?;
        let h = rdr.byte_headers()?.clone();
        let i_sid = req_col(&h, "service_id", "calendar.txt")?;
        let i_day = req_col(&h, weekday_col, "calendar.txt")?;
        let i_start = req_col(&h, "start_date", "calendar.txt")?;
        let i_end = req_col(&h, "end_date", "calendar.txt")?;
        let mut rec = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut rec)? {
            // 8-digit YYYYMMDD strings compare correctly lexicographically.
            if field(&rec, i_day) == "1"
                && field(&rec, i_start) <= date
                && date <= field(&rec, i_end)
            {
                active.insert(field(&rec, i_sid).to_string());
            }
        }
    }

    let cd = feed.join("calendar_dates.txt");
    if cd.exists() {
        let mut removes: Vec<String> = Vec::new();
        let mut rdr = open_csv(&cd)?;
        let h = rdr.byte_headers()?.clone();
        let i_sid = req_col(&h, "service_id", "calendar_dates.txt")?;
        let i_date = req_col(&h, "date", "calendar_dates.txt")?;
        let i_ex = req_col(&h, "exception_type", "calendar_dates.txt")?;
        let mut rec = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut rec)? {
            if field(&rec, i_date) != date {
                continue;
            }
            match field(&rec, i_ex) {
                "1" => {
                    active.insert(field(&rec, i_sid).to_string());
                }
                "2" => removes.push(field(&rec, i_sid).to_string()),
                _ => {}
            }
        }
        for sid in removes {
            active.remove(&sid);
        }
    } else if !cal.exists() {
        return Err(anyhow!(
            "feed has neither calendar.txt nor calendar_dates.txt — cannot expand services"
        ));
    }

    Ok(active)
}

fn load_routes(feed: &Path) -> Result<(HashMap<String, u32>, Vec<RouteMeta>)> {
    println!("📂 Reading routes.txt …");
    let mut rdr = open_csv(&feed.join("routes.txt"))?;
    let h = rdr.byte_headers()?.clone();
    let i_id = req_col(&h, "route_id", "routes.txt")?;
    let i_type = req_col(&h, "route_type", "routes.txt")?;
    let i_short = col(&h, "route_short_name");
    let i_agency = col(&h, "agency_id");
    let mut index: HashMap<String, u32> = HashMap::new();
    let mut routes: Vec<RouteMeta> = Vec::new();
    let mut rec = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut rec)? {
        let id = field(&rec, i_id);
        if id.is_empty() || index.contains_key(id) {
            continue;
        }
        index.insert(id.to_string(), routes.len() as u32);
        routes.push(RouteMeta {
            label: route_type_label(field(&rec, i_type)),
            short_name: i_short
                .map(|i| field(&rec, i).to_string())
                .unwrap_or_default(),
            agency_id: i_agency
                .map(|i| field(&rec, i).to_string())
                .unwrap_or_default(),
        });
    }
    println!("   ✓ {} routes", routes.len());
    Ok((index, routes))
}

fn load_stops(feed: &Path) -> Result<(Vec<[f64; 2]>, HashMap<String, u32>)> {
    let path = feed.join("stops.txt");
    let mut pos: Vec<[f64; 2]> = Vec::new();
    let mut index: HashMap<String, u32> = HashMap::new();
    if !path.exists() {
        println!("   ⚠ stops.txt missing — only shape-dist timing available (no projection/stop-to-stop fallbacks)");
        return Ok((pos, index));
    }
    println!("📂 Reading stops.txt …");
    let mut rdr = open_csv(&path)?;
    let h = rdr.byte_headers()?.clone();
    let i_id = req_col(&h, "stop_id", "stops.txt")?;
    let i_lat = req_col(&h, "stop_lat", "stops.txt")?;
    let i_lon = req_col(&h, "stop_lon", "stops.txt")?;
    let mut rec = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut rec)? {
        let id = field(&rec, i_id);
        if id.is_empty() || index.contains_key(id) {
            continue;
        }
        let (Ok(lat), Ok(lon)) = (
            field(&rec, i_lat).parse::<f64>(),
            field(&rec, i_lon).parse::<f64>(),
        ) else {
            continue;
        };
        index.insert(id.to_string(), pos.len() as u32);
        pos.push([lon, lat]);
    }
    println!("   ✓ {} stops", pos.len());
    Ok((pos, index))
}

/// Stream shapes.txt, keeping only the interned shape ids the day's trips
/// reference. Consecutive duplicate points are deduped (their zero-length
/// segments carry no information); the file's dist axis survives dedup because
/// dropped points shed their dist too.
fn load_shapes(feed: &Path, needed: &HashMap<String, u32>) -> Result<Vec<Option<Shape>>> {
    let n = needed.len();
    let mut raw: Vec<Vec<(u32, f64, f64, Option<f64>)>> = (0..n).map(|_| Vec::new()).collect();
    let path = feed.join("shapes.txt");
    if n == 0 {
        return Ok(Vec::new());
    }
    if !path.exists() {
        println!("   ⚠ shapes.txt missing — all trips fall back to stop-to-stop geometry");
        return Ok((0..n).map(|_| None).collect());
    }
    println!("📂 Reading shapes.txt (streaming) …");
    let mut rdr = open_csv(&path)?;
    let h = rdr.byte_headers()?.clone();
    let i_id = req_col(&h, "shape_id", "shapes.txt")?;
    let i_seq = req_col(&h, "shape_pt_sequence", "shapes.txt")?;
    let i_lat = req_col(&h, "shape_pt_lat", "shapes.txt")?;
    let i_lon = req_col(&h, "shape_pt_lon", "shapes.txt")?;
    let i_dist = col(&h, "shape_dist_traveled");
    let mut rec = csv::ByteRecord::new();
    let mut cur_key: Vec<u8> = Vec::new();
    let mut cur: Option<usize> = None;
    let mut first = true;
    while rdr.read_byte_record(&mut rec)? {
        let sid = rec.get(i_id).unwrap_or(b"");
        if first || sid != cur_key.as_slice() {
            first = false;
            cur_key.clear();
            cur_key.extend_from_slice(sid);
            cur = std::str::from_utf8(sid)
                .ok()
                .and_then(|s| needed.get(s.trim()).map(|&ix| ix as usize));
        }
        let Some(slot) = cur else { continue };
        let (Ok(seq), Ok(lat), Ok(lon)) = (
            field(&rec, i_seq).parse::<u32>(),
            field(&rec, i_lat).parse::<f64>(),
            field(&rec, i_lon).parse::<f64>(),
        ) else {
            continue;
        };
        let dist = i_dist.and_then(|i| field(&rec, i).parse::<f64>().ok());
        raw[slot].push((seq, lon, lat, dist));
    }

    let mut loaded = 0usize;
    let shapes: Vec<Option<Shape>> = raw
        .into_iter()
        .map(|mut rows| {
            if rows.len() < 2 {
                return None;
            }
            rows.sort_by_key(|r| r.0);
            let mut pts: Vec<[f64; 2]> = Vec::with_capacity(rows.len());
            let mut fd: Vec<Option<f64>> = Vec::with_capacity(rows.len());
            for (_, lon, lat, d) in rows {
                if let Some(last) = pts.last() {
                    if last[0] == lon && last[1] == lat {
                        continue;
                    }
                }
                pts.push([lon, lat]);
                fd.push(d);
            }
            if pts.len() < 2 {
                return None;
            }
            let cum = cumulative_meters(&pts);
            // The file's dist axis is usable iff complete, non-decreasing and
            // with positive extent — otherwise stop matching falls back to
            // projection on the computed axis.
            let mut file_dist: Option<Vec<f64>> = Some(Vec::with_capacity(fd.len()));
            let mut prev = f64::NEG_INFINITY;
            for d in &fd {
                match d {
                    Some(v) if *v >= prev => {
                        prev = *v;
                        if let Some(list) = &mut file_dist {
                            list.push(*v);
                        }
                    }
                    _ => {
                        file_dist = None;
                        break;
                    }
                }
            }
            let file_dist = file_dist
                .filter(|l| l.last().copied().unwrap_or(0.0) > l.first().copied().unwrap_or(0.0));
            loaded += 1;
            Some(Shape {
                pts,
                cum,
                file_dist,
            })
        })
        .collect();
    println!("   ✓ {loaded} / {n} referenced shapes loaded");
    Ok(shapes)
}

/// Cumulative haversine meters along a polyline (starts at 0).
fn cumulative_meters(pts: &[[f64; 2]]) -> Vec<f64> {
    let mut cum = Vec::with_capacity(pts.len());
    cum.push(0.0);
    let mut total = 0.0;
    for w in pts.windows(2) {
        // haversine_distance(lat1, lon1, lat2, lon2) — coords are [lon, lat].
        total += common::haversine_distance(w[0][1], w[0][0], w[1][1], w[1][0]);
        cum.push(total);
    }
    cum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Local midnight of 2026-07-03 in Europe/Amsterdam (CEST, UTC+2):
    /// 2026-07-02T22:00:00Z.
    const MID_MS: i64 = 1_783_029_600_000;

    /// Absolute ms for a GTFS clock time on the fixture's service date.
    fn t(hours: i64, minutes: i64) -> i64 {
        MID_MS + (hours * 60 + minutes) * 60_000
    }

    /// A tiny synthetic feed exercising: a normal shape-dist trip with dwell,
    /// a >24:00:00 trip, a shape-less trip, a blank-dist trip (projection),
    /// a non-monotone-times trip, and an exception_type=2 removal.
    fn write_fixture(dir: &Path) {
        fs::write(
            dir.join("agency.txt"),
            "agency_id,agency_name,agency_url,agency_timezone,agency_phone\n\
             TEST,Test,http://example.com,Europe/Amsterdam,0\n",
        )
        .unwrap();
        fs::write(
            dir.join("routes.txt"),
            "route_id,agency_id,route_short_name,route_long_name,route_desc,route_type,route_color,route_text_color,route_url\n\
             R1,TEST,10,Bus Ten,,3,,,\n\
             R2,TEST,IC1,Rail One,,2,,,\n",
        )
        .unwrap();
        // RM is added AND removed on the date: the type-2 removal must win.
        fs::write(
            dir.join("calendar_dates.txt"),
            "service_id,date,exception_type\n\
             WK,20260703,1\n\
             RM,20260703,1\n\
             RM,20260703,2\n",
        )
        .unwrap();
        fs::write(
            dir.join("trips.txt"),
            "route_id,service_id,trip_id,realtime_trip_id,trip_headsign,trip_short_name,trip_long_name,direction_id,block_id,shape_id,wheelchair_accessible,bikes_allowed\n\
             R1,WK,T_NORMAL,,Downtown,,,0,,SHP1,0,0\n\
             R2,WK,T_LATE,,Night,,,0,,SHP1,0,0\n\
             R1,WK,T_NOSHAPE,,NoShape,,,0,,,0,0\n\
             R1,WK,T_PROJ,,Proj,,,0,,SHP1,0,0\n\
             R1,WK,T_BAD,,Bad,,,0,,SHP1,0,0\n\
             R1,RM,T_REMOVED,,Removed,,,0,,SHP1,0,0\n",
        )
        .unwrap();
        fs::write(
            dir.join("stops.txt"),
            "stop_id,stop_code,stop_name,stop_lat,stop_lon,location_type,parent_station\n\
             S1,,A,52.0,4.00,,\n\
             S2,,B,52.0,4.02,,\n\
             S3,,C,52.0,4.04,,\n",
        )
        .unwrap();
        fs::write(
            dir.join("shapes.txt"),
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon,shape_dist_traveled\n\
             SHP1,0,52.0,4.00,0\n\
             SHP1,1,52.0,4.01,1000\n\
             SHP1,2,52.0,4.02,2000\n\
             SHP1,3,52.0,4.03,3000\n\
             SHP1,4,52.0,4.04,4000\n",
        )
        .unwrap();
        fs::write(
            dir.join("stop_times.txt"),
            "trip_id,stop_sequence,stop_id,stop_headsign,arrival_time,departure_time,pickup_type,drop_off_type,timepoint,shape_dist_traveled,fare_units_traveled\n\
             T_NORMAL,1,S1,,08:00:00,08:00:00,0,0,1,0,0\n\
             T_NORMAL,2,S2,,08:10:00,08:12:00,0,0,1,2000,0\n\
             T_NORMAL,3,S3,,08:20:00,08:20:00,0,0,1,4000,0\n\
             T_LATE,1,S1,,25:30:00,25:30:00,0,0,1,0,0\n\
             T_LATE,2,S3,,25:50:00,25:50:00,0,0,1,4000,0\n\
             T_NOSHAPE,1,S1,,09:00:00,09:00:00,0,0,1,,0\n\
             T_NOSHAPE,2,S3,,09:30:00,09:30:00,0,0,1,,0\n\
             T_PROJ,1,S1,,11:00:00,11:00:00,0,0,1,,0\n\
             T_PROJ,2,S2,,11:10:00,11:10:00,0,0,1,,0\n\
             T_PROJ,3,S3,,11:20:00,11:20:00,0,0,1,,0\n\
             T_BAD,1,S1,,12:00:00,12:00:00,0,0,1,0,0\n\
             T_BAD,2,S3,,11:50:00,11:50:00,0,0,1,4000,0\n\
             T_REMOVED,1,S1,,10:00:00,10:00:00,0,0,1,0,0\n\
             T_REMOVED,2,S3,,10:30:00,10:30:00,0,0,1,4000,0\n",
        )
        .unwrap();
    }

    fn expand(dir: &Path, max_trips: Option<usize>) -> (Vec<TripFeature>, ExpandStats) {
        let mut out: Vec<TripFeature> = Vec::new();
        let stats = expand_feed(dir, "20260703", &ExpandOptions { max_trips }, &mut |f| {
            out.push(f);
            Ok(())
        })
        .expect("expand fixture");
        (out, stats)
    }

    fn feature<'a>(features: &'a [TripFeature], id: &str) -> &'a TripFeature {
        features
            .iter()
            .find(|f| f.gtfs_trip_id == id)
            .unwrap_or_else(|| panic!("feature {id} missing"))
    }

    #[test]
    fn midnight_anchor_is_amsterdam_local() {
        assert_eq!(
            service_midnight_ms("20260703", chrono_tz::Europe::Amsterdam).unwrap(),
            MID_MS
        );
    }

    #[test]
    fn parses_gtfs_clock_times_including_after_midnight() {
        assert_eq!(parse_gtfs_time_seconds("08:00:00"), Some(28_800));
        assert_eq!(parse_gtfs_time_seconds("25:30:00"), Some(91_800)); // 01:30 next day
        assert_eq!(parse_gtfs_time_seconds("9:5:2"), Some(32_702)); // single-digit fields
        assert_eq!(parse_gtfs_time_seconds(" 24:00:00 "), Some(86_400));
        assert_eq!(parse_gtfs_time_seconds(""), None);
        assert_eq!(parse_gtfs_time_seconds("banana"), None);
        assert_eq!(parse_gtfs_time_seconds("08:60:00"), None);
    }

    #[test]
    fn route_type_labels_are_never_numeric_strings() {
        assert_eq!(route_type_label("3"), "bus");
        assert_eq!(route_type_label("2"), "rail");
        assert_eq!(route_type_label("0"), "tram");
        assert_eq!(route_type_label("1"), "metro");
        assert_eq!(route_type_label("4"), "ferry");
        assert_eq!(route_type_label("999"), "other");
        // The trap this guards: an all-numeric-string categorical is promoted
        // to Numeric by stt-build and colorMapping silently no-ops.
        for code in ["0", "1", "2", "3", "4", "999"] {
            assert!(route_type_label(code).parse::<f64>().is_err());
        }
    }

    #[test]
    fn expands_normal_trip_with_dist_interp_and_dwell() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        let (features, stats) = expand(dir.path(), None);

        let f = feature(&features, "T_NORMAL");
        assert_eq!(f.kind, GeomKind::ShapeDist);
        assert_eq!(f.route_type, "bus");
        assert_eq!(f.route_short_name, "10");
        assert_eq!(f.agency_id, "TEST");
        assert_eq!(f.headsign, "Downtown");
        // Shape vertices between stops + a duplicated dwell vertex at S2
        // (arrival 08:10, departure 08:12).
        let lons: Vec<f64> = f.coords.iter().map(|c| c[0]).collect();
        assert_eq!(lons, vec![4.00, 4.01, 4.02, 4.02, 4.03, 4.04]);
        assert!(f.coords.iter().all(|c| c[1] == 52.0));
        // Interpolated in shape-distance: 1000/2000 → half of 08:00→08:10;
        // 3000 → half of 08:12→08:20.
        assert_eq!(
            f.times_ms,
            vec![t(8, 0), t(8, 5), t(8, 10), t(8, 12), t(8, 16), t(8, 20)]
        );
        assert_eq!(f.start_ms, t(8, 0));
        assert_eq!(f.end_ms, t(8, 20));

        assert_eq!(stats.trips_built, 4);
        assert_eq!(stats.via_shape_dist, 2); // T_NORMAL + T_LATE
        assert_eq!(stats.via_projection, 1); // T_PROJ
        assert_eq!(stats.via_stops, 1); // T_NOSHAPE
    }

    #[test]
    fn handles_times_past_24_hours() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        let (features, _) = expand(dir.path(), None);

        let f = feature(&features, "T_LATE");
        assert_eq!(f.kind, GeomKind::ShapeDist);
        assert_eq!(f.route_type, "rail");
        // 25:30:00 = 01:30 the NEXT morning, local.
        assert_eq!(f.start_ms, MID_MS + 91_800_000);
        assert_eq!(
            f.times_ms,
            vec![t(25, 30), t(25, 35), t(25, 40), t(25, 45), t(25, 50)]
        );
        assert_eq!(f.end_ms, t(25, 50));
    }

    #[test]
    fn missing_shape_falls_back_to_stop_to_stop() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        let (features, _) = expand(dir.path(), None);

        let f = feature(&features, "T_NOSHAPE");
        assert_eq!(f.kind, GeomKind::StopToStop);
        assert_eq!(f.coords, vec![[4.00, 52.0], [4.04, 52.0]]);
        assert_eq!(f.times_ms, vec![t(9, 0), t(9, 30)]);
    }

    #[test]
    fn blank_dist_columns_fall_back_to_projection() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        let (features, _) = expand(dir.path(), None);

        let f = feature(&features, "T_PROJ");
        assert_eq!(f.kind, GeomKind::ShapeProjected);
        // Stops sit exactly on shape vertices 0/2/4, so the projected trip
        // walks all five shape points.
        let lons: Vec<f64> = f.coords.iter().map(|c| c[0]).collect();
        assert_eq!(lons, vec![4.00, 4.01, 4.02, 4.03, 4.04]);
        // Timing interpolates on the computed cumulative-meters axis; equal
        // lon spacing at one latitude = equal segment lengths, so vertex 1 is
        // half-way through 11:00 → 11:10 (float axis → ±2 ms slack).
        assert_eq!(f.times_ms[0], t(11, 0));
        assert!(
            (f.times_ms[1] - t(11, 5)).abs() <= 2,
            "got {}",
            f.times_ms[1]
        );
        assert_eq!(f.times_ms[2], t(11, 10));
        assert!(
            (f.times_ms[3] - t(11, 15)).abs() <= 2,
            "got {}",
            f.times_ms[3]
        );
        assert_eq!(f.times_ms[4], t(11, 20));
    }

    #[test]
    fn honors_exception_type_2_removal() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        let (features, stats) = expand(dir.path(), None);

        assert!(features.iter().all(|f| f.gtfs_trip_id != "T_REMOVED"));
        assert_eq!(stats.services_active, 1); // WK only — RM removed
        assert_eq!(stats.trips_scheduled, 5); // everything except T_REMOVED
    }

    #[test]
    fn drops_non_monotone_times_with_accounting() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        let (features, stats) = expand(dir.path(), None);

        assert!(features.iter().all(|f| f.gtfs_trip_id != "T_BAD"));
        assert_eq!(stats.dropped_bad_times, 1);
        assert_eq!(stats.dropped_degenerate, 0);
        assert_eq!(stats.dropped_few_anchors, 0);
    }

    #[test]
    fn output_order_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        let (a, _) = expand(dir.path(), None);
        let (b, _) = expand(dir.path(), None);
        assert_eq!(a, b);

        // Sorted by (start time, trip_id); `seq` is the pre-drop order index,
        // so the dropped T_BAD (12:00, index 3) leaves a hole before T_LATE.
        let ids: Vec<&str> = a.iter().map(|f| f.gtfs_trip_id.as_str()).collect();
        assert_eq!(ids, vec!["T_NORMAL", "T_NOSHAPE", "T_PROJ", "T_LATE"]);
        assert_eq!(
            a.iter().map(|f| f.seq).collect::<Vec<_>>(),
            vec![0, 1, 2, 4]
        );
        // Vertex timestamps are non-decreasing and match the feature window.
        for f in &a {
            assert!(f.times_ms.windows(2).all(|w| w[1] >= w[0]));
            assert_eq!(f.start_ms, *f.times_ms.first().unwrap());
            assert_eq!(f.end_ms, *f.times_ms.last().unwrap());
            assert_eq!(f.coords.len(), f.times_ms.len());
        }
    }

    #[test]
    fn max_trips_takes_an_even_temporal_stride() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        // 5 ordered candidates (incl. the doomed T_BAD): stride keeps
        // indices 0 and 2 → T_NORMAL + T_PROJ.
        let (features, _) = expand(dir.path(), Some(2));
        let ids: Vec<&str> = features.iter().map(|f| f.gtfs_trip_id.as_str()).collect();
        assert_eq!(ids, vec!["T_NORMAL", "T_PROJ"]);
    }

    #[test]
    fn calendar_txt_weekly_service_with_exceptions() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("calendar.txt"),
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n\
             WK2,0,0,0,0,1,0,0,20260101,20261231\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("calendar_dates.txt"),
            "service_id,date,exception_type\n\
             WK2,20260703,2\n\
             X,20260710,1\n",
        )
        .unwrap();
        // 2026-07-03 is a Friday, but WK2 is removed by exception that day.
        assert!(active_services(dir.path(), "20260703").unwrap().is_empty());
        // The following Friday runs, plus the one-off X service.
        let next = active_services(dir.path(), "20260710").unwrap();
        assert_eq!(next.len(), 2);
        assert!(next.contains("WK2") && next.contains("X"));
        // Saturday: weekly flag is 0.
        assert!(active_services(dir.path(), "20260704").unwrap().is_empty());
    }
}
