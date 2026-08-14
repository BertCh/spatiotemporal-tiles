//! `storms` — NEXRAD Level II storm-radar tiles for the 2020-08-10 Iowa derecho.
//!
//! Downloads archived Level II volumes from the public `unidata-nexrad-level2`
//! AWS bucket for a set of radar sites over an event window, decodes the base
//! (0.5°) reflectivity sweep of each volume, reprojects every gate to lon/lat
//! (4/3-earth beam model), and **mosaics all sites onto one analysis grid per
//! ~5-minute scan** (max-combining overlaps). From each per-scan grid it bakes:
//!
//! - **storm-field** — filled reflectivity contour bands (polygons, `dbz_band`).
//! - **storm-cells** — storm-cell centroids (points, `max_dbz` + `area_km2`).
//! - **storm-tracks** — cells linked across scans into tracks (linestrings with
//!   per-vertex intensity over time, so they animate as growing/decaying trails).
//!
//! All polar reprojection, mosaicking, contouring, and cell tracking happen here
//! at build time; the browser only renders finished vector geometry.
//!
//! The heavy numeric work lives in [`crate::radar`]; this module is download +
//! decode + orchestration + deterministic emission.

use anyhow::{Context, Result};
use chrono::{NaiveDate, TimeZone, Timelike, Utc};
use clap::Parser;
use serde_json::{json, Map, Value as JsonValue};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::common::{
    self, parse_bounds, LineStringRecord, PointRecord, PolygonRecord, PropertyColumn,
    StreamingLineStringParquetWriter, StreamingParquetWriter, StreamingPolygonParquetWriter,
    SttBuildOptions,
};
use crate::radar::{contour_bands, detect_cells, track_cells, Cell, MosaicGrid};

use nexrad_data::aws::archive::{self, Identifier};
use nexrad_data::volume::File as VolumeFile;
use nexrad_model::data::{Product, SweepField};
use nexrad_model::geo::RadarCoordinateSystem;

/// Temporal bucket: radar volume scans repeat roughly every 5 minutes.
const BUCKET_MS: i64 = 5 * 60 * 1000;

#[derive(Parser, Debug)]
#[command(about = "Generate NEXRAD storm-radar tiles for the 2020-08-10 Iowa derecho")]
pub struct Args {
    /// Output directory. Writes <output>/storm-field, /storm-cells, /storm-tracks
    /// (packed STT archives) plus sibling .parquet intermediates.
    #[arg(short, long, default_value = "examples/showcase/public/data")]
    pub output: PathBuf,

    /// Radar sites (WSR-88D ICAO ids) along the derecho path, west→east.
    /// Default spans eastern Nebraska → eastern Iowa.
    #[arg(long, value_delimiter = ',', default_value = "KOAX,KDMX,KDVN")]
    pub sites: Vec<String>,

    /// UTC date of the event.
    #[arg(long, default_value = "2020-08-10")]
    pub date: String,

    /// Inclusive start hour (UTC). The derecho crossed Iowa ~16:00–23:00Z.
    #[arg(long, default_value = "16")]
    pub start_hour: u32,

    /// Exclusive end hour (UTC, < 24).
    #[arg(long, default_value = "23")]
    pub end_hour: u32,

    /// Keep only every Nth volume per site (1 = every scan). Bounds download +
    /// scan count for a first build.
    #[arg(long, default_value = "2")]
    pub scan_stride: usize,

    /// dBZ thresholds for the filled contour bands (ascending).
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "20,25,30,35,40,45,50,55,60,65"
    )]
    pub dbz_thresholds: Vec<f64>,

    /// Storm-cell detection floor (dBZ): connected components on dbz >= this.
    #[arg(long, default_value = "50")]
    pub cell_dbz: f64,

    /// Drop storm cells smaller than this (km²) — removes speckle.
    #[arg(long, default_value = "12")]
    pub min_cell_km2: f64,

    /// Analysis-grid spacing in km.
    #[arg(long, default_value = "2.0")]
    pub grid_km: f64,

    /// Drop contour-band polygons smaller than this (km²) — removes the speckle
    /// of disconnected low-dBZ patches that otherwise explodes the polygon count.
    #[arg(long, default_value = "9")]
    pub min_band_km2: f64,

    /// Analysis bbox: min_lat,min_lon,max_lat,max_lon (default = derecho corridor).
    #[arg(long, default_value = "39.5,-99.5,44.0,-87.5")]
    pub bounds: String,

    /// Max track-association speed (m/s) for nearest-centroid matching.
    #[arg(long, default_value = "45")]
    pub max_cell_speed: f64,

    /// Build pyramid zoom range.
    #[arg(long, default_value = "4")]
    pub min_zoom: u8,
    #[arg(long, default_value = "9")]
    pub max_zoom: u8,

    /// Use cached downloaded volumes only; never hit S3.
    #[arg(long)]
    pub no_download: bool,

    /// Cache directory for downloaded V06 volumes.
    #[arg(long, default_value = "radar-data")]
    pub cache_dir: PathBuf,

    /// Write only the intermediate parquet for each archive (skip stt-build).
    #[arg(long)]
    pub skip_build: bool,
}

/// Round a scan timestamp to the nearest 5-minute bucket so volumes from
/// different sites collected seconds apart land in the same frame.
fn round_bucket(t_ms: i64) -> i64 {
    ((t_ms + BUCKET_MS / 2) / BUCKET_MS) * BUCKET_MS
}

pub fn run(args: Args) -> Result<()> {
    let (min_lat, min_lon, max_lat, max_lon) =
        parse_bounds(&args.bounds).context("parsing --bounds")?;
    let day = NaiveDate::parse_from_str(&args.date, "%Y-%m-%d")
        .with_context(|| format!("parsing --date {}", args.date))?;
    anyhow::ensure!(
        args.end_hour <= 24 && args.start_hour < args.end_hour,
        "invalid hour window"
    );

    let window_start = day
        .and_hms_opt(args.start_hour, 0, 0)
        .map(|n| Utc.from_utc_datetime(&n).timestamp_millis())
        .context("start hour")?;
    let window_end = day
        .and_hms_opt(args.end_hour.min(23), 59, 0)
        .map(|n| Utc.from_utc_datetime(&n).timestamp_millis())
        .context("end hour")?;

    std::fs::create_dir_all(&args.output)?;

    // One scoped async runtime, used only to block_on the async AWS calls.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;

    // ---- Phase 1: download, decode, and mosaic every scan into per-bucket grids.
    println!(
        "🌩  storms: {} site(s) {}–{:02}:00Z on {} (stride {})",
        args.sites.len(),
        args.start_hour,
        args.end_hour,
        args.date,
        args.scan_stride
    );

    // BTreeMap keeps buckets time-ordered → deterministic emission order.
    let mut grids: BTreeMap<i64, MosaicGrid> = BTreeMap::new();

    for site in &args.sites {
        let paths = ensure_volumes(
            &rt,
            site,
            &day,
            args.start_hour,
            args.end_hour,
            args.scan_stride,
            &args.cache_dir,
            args.no_download,
        )
        .with_context(|| format!("gathering volumes for {site}"))?;
        println!("  {site}: {} volume(s)", paths.len());

        for path in paths {
            match rasterize_volume(
                &path,
                &mut grids,
                min_lat,
                min_lon,
                max_lat,
                max_lon,
                args.grid_km,
            ) {
                Ok(true) => {}
                Ok(false) => {}
                Err(e) => eprintln!("    ⚠️  {} skipped: {e}", path.display()),
            }
        }
    }

    anyhow::ensure!(
        !grids.is_empty(),
        "no usable radar volumes decoded — nothing to build"
    );
    println!("  mosaicked {} scan bucket(s)", grids.len());

    // ---- Phase 2: contour bands + storm-cell centroids per bucket.
    let field_parquet = args.output.join("storm-field.parquet");
    let cells_parquet = args.output.join("storm-cells.parquet");
    let tracks_parquet = args.output.join("storm-tracks.parquet");

    let mut field_w = StreamingPolygonParquetWriter::with_columns(
        &field_parquet,
        vec![PropertyColumn::string("dbz_band")],
    )?;
    let mut cells_w = StreamingParquetWriter::with_columns(
        &cells_parquet,
        vec![
            PropertyColumn::numeric("max_dbz"),
            PropertyColumn::numeric("area_km2"),
        ],
    )?;

    // dBZ values below the first threshold read as "no echo" when contouring.
    let empty_fill = args.dbz_thresholds[0] - 30.0;
    let mut frames: Vec<(i64, Vec<Cell>)> = Vec::new();
    let (mut n_field, mut n_cells) = (0usize, 0usize);

    for (&t, grid) in &grids {
        if grid.is_empty() {
            continue;
        }
        for band in contour_bands(grid, &args.dbz_thresholds, empty_fill, args.min_band_km2) {
            let mut props = Map::new();
            // Range label ("20-25"), NOT a bare number — stt-build promotes
            // all-numeric-string columns to a Numeric column, which would defeat
            // the categorical dbz-band color mapping in the renderer.
            props.insert(
                "dbz_band".into(),
                JsonValue::String(format!("{}-{}", band.dbz_lo, band.dbz_hi)),
            );
            field_w.write_polygon(&PolygonRecord {
                rings: band.rings,
                timestamp_ms: t,
                properties: props,
            })?;
            n_field += 1;
        }

        let cells = detect_cells(grid, args.cell_dbz, args.min_cell_km2);
        for c in &cells {
            let mut props = Map::new();
            props.insert("max_dbz".into(), json!(c.max_dbz as f64));
            props.insert("area_km2".into(), json!(c.area_km2));
            cells_w.write_point(&PointRecord {
                lon: c.lon,
                lat: c.lat,
                timestamp_ms: t,
                properties: props,
            })?;
            n_cells += 1;
        }
        frames.push((t, cells));
    }

    field_w.finish()?;
    cells_w.finish()?;

    // ---- Phase 3: link cells into tracks across scans.
    let mut tracks = track_cells(frames, args.max_cell_speed);
    // Deterministic order for reproducible packs.
    tracks.sort_by(|a, b| {
        let (pa, pb) = (&a.points[0], &b.points[0]);
        pa.t_ms
            .cmp(&pb.t_ms)
            .then(
                pa.lon
                    .partial_cmp(&pb.lon)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(
                pa.lat
                    .partial_cmp(&pb.lat)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    let mut tracks_w = StreamingLineStringParquetWriter::with_columns(
        &tracks_parquet,
        vec![
            PropertyColumn::numeric("peak_dbz"),
            PropertyColumn::numeric("duration_min"),
            PropertyColumn::numeric("npoints"),
        ],
    )?;
    for tr in &tracks {
        let coordinates: Vec<[f64; 2]> = tr.points.iter().map(|p| [p.lon, p.lat]).collect();
        let vertex_ms: Vec<i64> = tr.points.iter().map(|p| p.t_ms).collect();
        let vertex_vals: Vec<f32> = tr.points.iter().map(|p| p.max_dbz).collect();
        let peak = vertex_vals.iter().cloned().fold(f32::MIN, f32::max);
        let t0 = vertex_ms[0];
        let t1 = *vertex_ms.last().unwrap();

        let mut props = Map::new();
        props.insert("peak_dbz".into(), json!(peak as f64));
        props.insert("duration_min".into(), json!((t1 - t0) as f64 / 60_000.0));
        props.insert("npoints".into(), json!(tr.points.len() as f64));

        tracks_w.write_linestring(&LineStringRecord {
            coordinates,
            timestamp_ms: t0,
            end_timestamp_ms: Some(t1),
            vertex_timestamps_ms: Some(vertex_ms),
            vertex_values: Some(vertex_vals),
            vertex_value_matrix: None,
            properties: props,
        })?;
    }
    tracks_w.finish()?;

    println!(
        "  baked {n_field} contour polygon(s), {n_cells} cell centroid(s), {} track(s)",
        tracks.len()
    );
    let _ = (window_start, window_end);

    if args.skip_build {
        println!("  --skip-build set; intermediate parquet written, stopping.");
        return Ok(());
    }

    // ---- Phase 4: build the three packed STT archives.
    build_field(
        &field_parquet,
        &args.output.join("storm-field"),
        args.min_zoom,
        args.max_zoom,
    )?;
    build_cells(
        &cells_parquet,
        &args.output.join("storm-cells"),
        args.min_zoom,
        args.max_zoom,
    )?;
    build_tracks(
        &tracks_parquet,
        &args.output.join("storm-tracks"),
        args.min_zoom,
        args.max_zoom,
    )?;

    println!(
        "✅ storms: storm-field, storm-cells, storm-tracks written to {}",
        args.output.display()
    );
    Ok(())
}

/// Ensure the in-window volumes for a site are present on disk, downloading any
/// that are missing. Returns their cache paths (chronological, stride-applied).
#[allow(clippy::too_many_arguments)]
fn ensure_volumes(
    rt: &tokio::runtime::Runtime,
    site: &str,
    day: &NaiveDate,
    start_hour: u32,
    end_hour: u32,
    stride: usize,
    cache_dir: &Path,
    no_download: bool,
) -> Result<Vec<PathBuf>> {
    let site_dir = cache_dir.join(site);
    std::fs::create_dir_all(&site_dir)?;
    let in_window = |id: &Identifier| -> bool {
        id.date_time()
            .map(|dt| dt.hour() >= start_hour && dt.hour() < end_hour)
            .unwrap_or(false)
    };

    if no_download {
        // Glob the cache for in-window files only.
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&site_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| in_window(&Identifier::new(n.to_string())))
                    .unwrap_or(false)
            })
            .collect();
        paths.sort();
        return Ok(stride_filter(paths, stride));
    }

    let mut ids: Vec<Identifier> = rt
        .block_on(archive::list_files(site, day))
        .map_err(|e| anyhow::anyhow!("AWS list_files({site}): {e:?}"))?
        .into_iter()
        .filter(in_window)
        .collect();
    ids.sort(); // Identifier orders by file name → chronological.
    let ids = stride_filter(ids, stride);

    let mut paths = Vec::with_capacity(ids.len());
    for id in ids {
        let path = site_dir.join(id.name());
        if !path.exists() {
            let file = rt
                .block_on(archive::download_file(id.clone()))
                .map_err(|e| anyhow::anyhow!("AWS download_file({}): {e:?}", id.name()))?;
            std::fs::write(&path, file.data())?;
        }
        paths.push(path);
    }
    Ok(paths)
}

fn stride_filter<T>(items: Vec<T>, stride: usize) -> Vec<T> {
    items.into_iter().step_by(stride.max(1)).collect()
}

/// Decode one volume, extract its base reflectivity sweep, and mosaic it into the
/// per-bucket grid. Returns `Ok(true)` if a sweep was added.
#[allow(clippy::too_many_arguments)]
fn rasterize_volume(
    path: &Path,
    grids: &mut BTreeMap<i64, MosaicGrid>,
    min_lat: f64,
    min_lon: f64,
    max_lat: f64,
    max_lon: f64,
    grid_km: f64,
) -> Result<bool> {
    let bytes = std::fs::read(path)?;
    let file = VolumeFile::new(bytes)
        .decompress()
        .map_err(|e| anyhow::anyhow!("decompress: {e:?}"))?;
    let scan = file
        .scan()
        .map_err(|e| anyhow::anyhow!("decode scan: {e:?}"))?;

    let site = match scan.site() {
        Some(s) => s,
        None => return Ok(false),
    };
    let cs = RadarCoordinateSystem::new(site);

    // Base scan = lowest-elevation sweep that has radials.
    let base = scan
        .sweeps()
        .iter()
        .filter_map(|s| s.elevation_angle_degrees().map(|e| (e, s)))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, s)| s);
    let base = match base {
        Some(b) => b,
        None => return Ok(false),
    };

    let field = match SweepField::from_radials(base.radials(), Product::Reflectivity) {
        Some(f) => f,
        None => return Ok(false),
    };

    let scan_time = base
        .radials()
        .first()
        .map(|r| r.collection_timestamp())
        .unwrap_or(0);
    let bucket = round_bucket(scan_time);

    let grid = grids
        .entry(bucket)
        .or_insert_with(|| MosaicGrid::new(min_lat, min_lon, max_lat, max_lon, grid_km));
    grid.add_sweep_field(&cs, &field);
    Ok(true)
}

fn build_field(input: &Path, output: &Path, min_zoom: u8, max_zoom: u8) -> Result<()> {
    common::run_stt_build_with_full_options(SttBuildOptions {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        time_field: "timestamp".into(),
        end_time_field: None,
        min_zoom,
        max_zoom,
        compression: "zstd".into(),
        // One bucket per scan — the precip field animates by stepping buckets.
        temporal_bucket: Some("5m".into()),
        temporal_lod: None,
        summary: None,
        summary_sub_buckets: None,
        min_features_per_tile: None,
        min_zoom_field: None,
        max_zoom_field: None,
        // Polygons SHOULD clip at tile boundaries.
        no_clip: false,
        quantize_coords: None,
        quantize_attrs: Vec::new(),
        blob_ordering: None,
    })
}

fn build_cells(input: &Path, output: &Path, min_zoom: u8, max_zoom: u8) -> Result<()> {
    common::run_stt_build_with_full_options(SttBuildOptions {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        time_field: "timestamp".into(),
        end_time_field: None,
        min_zoom,
        max_zoom,
        compression: "zstd".into(),
        temporal_bucket: Some("5m".into()),
        temporal_lod: None,
        summary: None,
        summary_sub_buckets: None,
        min_features_per_tile: None,
        min_zoom_field: None,
        max_zoom_field: None,
        no_clip: false,
        quantize_coords: None,
        quantize_attrs: Vec::new(),
        blob_ordering: None,
    })
}

fn build_tracks(input: &Path, output: &Path, min_zoom: u8, max_zoom: u8) -> Result<()> {
    common::run_stt_build_with_full_options(SttBuildOptions {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        time_field: "timestamp".into(),
        // Real per-track lifetime; per-vertex times drive the trail animation
        // (the canonical animated-trips pattern, see datasets/drifters.rs).
        end_time_field: Some("end_timestamp".into()),
        min_zoom,
        max_zoom,
        compression: "zstd".into(),
        temporal_bucket: Some("5m".into()),
        temporal_lod: None,
        summary: None,
        summary_sub_buckets: None,
        min_features_per_tile: None,
        min_zoom_field: None,
        max_zoom_field: None,
        no_clip: false,
        quantize_coords: None,
        quantize_attrs: Vec::new(),
        blob_ordering: None,
    })
}
