//! Generate OpenStreetMap editing-history datasets — a time-series animation of
//! *the map being edited*.
//!
//! Two complementary signals, both scoped to one metro (default: New York City)
//! via `--bounds`. Data © OpenStreetMap contributors (ODbL).
//!
//! - `--source nodes`  → every node's **first version** (its creation) placed at
//!   its lon/lat at its creation timestamp. Zoomed into a fixed metro viewport
//!   and played back cumulatively, you watch the street grid and buildings "ink
//!   in" over ~18 years. Input: a full-history `.osh.pbf` extract (Geofabrik
//!   internal, needs a free OSM login; optionally pre-clipped with
//!   `osmium extract -b <bbox> --with-history` to shrink it).
//!
//! - `--source changesets` → every OSM changeset near the metro as a timestamped
//!   point (bbox centroid), coloured by editor era (iD / JOSM / Potlatch /
//!   StreetComplete / …) with an edit-volume H3 summary tier. Input: the public
//!   `changesets-latest.osm.bz2` dump from planet.openstreetmap.org (no login).
//!
//! Neither raw source ships in the repo; both are downloaded as a prerequisite
//! and passed via `--input`, mirroring the AIS / flights flow.

use crate::common::{self, PointRecord, PropertyColumn};
use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use clap::{Parser, ValueEnum};
use serde_json::{json, Map};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Source {
    /// First-version node creations from a full-history `.osh.pbf`.
    Nodes,
    /// Changeset bbox-centroids from the global `changesets-latest.osm.bz2`.
    Changesets,
}

#[derive(Parser, Debug)]
#[command(about = "Generate an OSM editing-history dataset (node creations or changesets)")]
pub struct Args {
    /// Which signal to build.
    #[arg(long, value_enum, default_value = "nodes")]
    pub source: Source,

    /// Input file. `nodes`: a full-history `.osh.pbf` extract. `changesets`:
    /// the `changesets-latest.osm.bz2` dump.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output `.stt` archive (or `.parquet` to skip the build).
    #[arg(short, long, default_value = "osm-edits.stt")]
    pub output: PathBuf,

    /// Metro bounding box "min_lat,min_lon,max_lat,max_lon". Default: NYC.
    #[arg(long, default_value = "40.49,-74.27,40.92,-73.68")]
    pub bounds: String,

    /// Drop edits before this date (YYYY-MM-DD). OSM launched in 2005.
    #[arg(long, default_value = "2005-01-01")]
    pub start_date: String,

    /// Drop edits after this date (YYYY-MM-DD). Default: no upper bound.
    #[arg(long)]
    pub end_date: Option<String>,

    /// `changesets` only: drop changesets whose bbox spans more than this many
    /// degrees in lat OR lon (planet-wide bot/import edits) so their centroid
    /// doesn't smear onto Null Island. 0 disables the filter.
    #[arg(long, default_value = "1.0")]
    pub max_bbox_deg: f64,

    /// `nodes` only: keep only nodes that carry real tags (buildings, POIs,
    /// roads…) and drop untagged geometry vertices. Cuts volume ~5-10× and is
    /// recommended for the showcase build.
    #[arg(long)]
    pub tagged_only: bool,

    /// Emit a server-aggregated H3 summary tier alongside the raw tier.
    #[arg(long)]
    pub summary_tier: bool,

    /// Skip the stt-build step (just write the intermediate Parquet).
    #[arg(long)]
    pub skip_build: bool,
}

/// Metro bounding box in lon/lat degrees.
#[derive(Clone, Copy, Debug)]
struct Bbox {
    min_lon: f64,
    min_lat: f64,
    max_lon: f64,
    max_lat: f64,
}

impl Bbox {
    fn contains(&self, lon: f64, lat: f64) -> bool {
        lon >= self.min_lon && lon <= self.max_lon && lat >= self.min_lat && lat <= self.max_lat
    }
}

pub fn run(args: Args) -> Result<()> {
    println!("🗺️  OSM Editing-History Generator");
    println!("=================================\n");
    println!("   © OpenStreetMap contributors (ODbL)\n");

    let (min_lat, min_lon, max_lat, max_lon) = common::parse_bounds(&args.bounds)?;
    let bbox = Bbox { min_lon, min_lat, max_lon, max_lat };
    let start_ms = date_to_ms(&args.start_date, false)?;
    let end_ms = match &args.end_date {
        Some(d) => date_to_ms(d, true)?,
        None => i64::MAX,
    };

    // .stt output → write an intermediate parquet, then shell out to stt-build.
    let build_stt = args.output.extension().map(|e| e == "stt").unwrap_or(false) && !args.skip_build;
    let intermediate = if build_stt {
        args.output.with_extension("parquet")
    } else {
        args.output.clone()
    };

    match args.source {
        Source::Nodes => run_nodes(&args, &bbox, start_ms, end_ms, &intermediate, build_stt),
        Source::Changesets => run_changesets(&args, &bbox, start_ms, end_ms, &intermediate, build_stt),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal A: node creation ("watch the metro draw itself")
// ─────────────────────────────────────────────────────────────────────────────

fn run_nodes(
    args: &Args,
    bbox: &Bbox,
    start_ms: i64,
    end_ms: i64,
    intermediate: &PathBuf,
    build_stt: bool,
) -> Result<()> {
    use osmpbf::{Element, ElementReader};

    println!("📡 Reading node creations from {}", args.input.display());
    println!("   bbox: {:?}", bbox);
    if args.tagged_only {
        println!("   keeping tagged features only (dropping geometry vertices)");
    }

    let typed_columns = vec![
        // Chronological color key (a year string → a time ramp in the showcase).
        PropertyColumn::string("year"),
        // Coarse thematic category for optional color-by-feature.
        PropertyColumn::string("kind"),
        PropertyColumn::numeric("changeset"),
        PropertyColumn::numeric("uid"),
        PropertyColumn::string("user"),
    ];
    let mut writer = common::StreamingParquetWriter::with_columns(intermediate, typed_columns)?;

    let reader = ElementReader::from_path(&args.input)
        .with_context(|| format!("opening {}", args.input.display()))?;

    let mut scanned: u64 = 0;
    let mut kept: u64 = 0;
    let mut write_err: Option<anyhow::Error> = None;

    reader.for_each(|element| {
        if write_err.is_some() {
            return;
        }
        match element {
            Element::DenseNode(n) => {
                scanned += 1;
                // History PBFs carry per-version DenseInfo; v1 == creation.
                let info = match n.info() {
                    Some(i) => i,
                    None => return,
                };
                if info.version() != 1 {
                    return;
                }
                emit_node(
                    &mut writer,
                    n.lon(),
                    n.lat(),
                    info.milli_timestamp(),
                    info.changeset(),
                    info.uid(),
                    info.user().ok(),
                    n.tags(),
                    bbox,
                    start_ms,
                    end_ms,
                    args.tagged_only,
                    &mut kept,
                    &mut write_err,
                );
            }
            Element::Node(n) => {
                scanned += 1;
                let info = n.info();
                if info.version() != Some(1) {
                    return;
                }
                let ts = match info.milli_timestamp() {
                    Some(t) => t,
                    None => return,
                };
                emit_node(
                    &mut writer,
                    n.lon(),
                    n.lat(),
                    ts,
                    info.changeset().unwrap_or(0),
                    info.uid().unwrap_or(0),
                    info.user().and_then(|r| r.ok()),
                    n.tags(),
                    bbox,
                    start_ms,
                    end_ms,
                    args.tagged_only,
                    &mut kept,
                    &mut write_err,
                );
            }
            // Ways and relations have no intrinsic geometry of their own here;
            // the node tier captures "where edits happened" on its own.
            _ => {}
        }
    })?;

    if let Some(e) = write_err {
        return Err(e);
    }

    let total = writer.finish()?;
    println!(
        "\n📊 Scanned {} nodes; kept {} creations in bbox/time window ({} rows written)",
        scanned, kept, total
    );
    anyhow::ensure!(total > 0, "No node creations matched the bbox/time filters.");

    if build_stt {
        let summary = if args.summary_tier {
            // Count-only overview: how many nodes were created per cell per
            // bucket — drives the zoomed-out "whole-metro fills in" view.
            Some(common::SttBuildSummaryOptions {
                scheme: "h3".to_string(),
                min_zoom: Some(8),
                max_zoom: Some(11),
                columns: String::new(),
            })
        } else {
            None
        };
        common::run_stt_build_with_full_options(common::SttBuildOptions {
            input: intermediate.clone(),
            output: args.output.clone(),
            time_field: "timestamp".to_string(),
            end_time_field: None,
            // City zooms: street grid at z14-16, metro overview at z8.
            min_zoom: 8,
            max_zoom: 16,
            compression: "zstd".to_string(),
            // Monthly base bucket over ~18 years (~216 buckets); coarser LOD for
            // low zooms. LOD entries must be multiples of the base bucket.
            temporal_bucket: Some("30d".to_string()),
            // No temporal LOD: these archives are rendered as RAW per-feature
            // points (colored by year/editor) and revealed cumulatively. The
            // LOD aggregate tiers coarsen time and drop per-feature property
            // columns, so the point layer would fall back to a constant color
            // and show aggregate-tile artifacts at low zoom. The H3 summary
            // tier (changesets) handles the low-zoom overview instead.
            temporal_lod: None,
            summary,
            summary_sub_buckets: None,
            min_features_per_tile: Some(2),
            min_zoom_field: None,
            max_zoom_field: None,
        })?;
        let _ = std::fs::remove_file(intermediate);
    }

    println!("\n✅ OSM node-creation dataset complete!");
    Ok(())
}

/// Filter one node and, if it passes, write a creation event. Used by both the
/// dense and sparse node paths. Errors are stashed in `err` (osmpbf's
/// `for_each` closure can't return a `Result`).
#[allow(clippy::too_many_arguments)]
fn emit_node<'a>(
    writer: &mut common::StreamingParquetWriter,
    lon: f64,
    lat: f64,
    ts_ms: i64,
    changeset: i64,
    uid: i32,
    user: Option<&str>,
    tags: impl Iterator<Item = (&'a str, &'a str)>,
    bbox: &Bbox,
    start_ms: i64,
    end_ms: i64,
    tagged_only: bool,
    kept: &mut u64,
    err: &mut Option<anyhow::Error>,
) {
    if !bbox.contains(lon, lat) {
        return;
    }
    if ts_ms < start_ms || ts_ms > end_ms {
        return;
    }
    let (tagged, kind) = classify_tags(tags);
    if tagged_only && !tagged {
        return;
    }
    let dt = match DateTime::<Utc>::from_timestamp_millis(ts_ms) {
        Some(d) => d,
        None => return,
    };

    let mut props = Map::new();
    props.insert("year".to_string(), json!(dt.year().to_string()));
    props.insert("kind".to_string(), json!(kind));
    props.insert("changeset".to_string(), json!(changeset));
    props.insert("uid".to_string(), json!(uid));
    if let Some(u) = user {
        props.insert("user".to_string(), json!(u));
    }

    let record = PointRecord::new(lon, lat, dt, props);
    if let Err(e) = writer.write_point(&record) {
        *err = Some(e);
        return;
    }
    *kept += 1;
}

/// Classify a node's tags into `(is_tagged, coarse_category)`. Untagged nodes
/// (pure geometry vertices of ways) classify as `"vertex"`; tagged nodes get a
/// small, stable category usable as a `colorMapping` key in the showcase.
fn classify_tags<'a>(tags: impl Iterator<Item = (&'a str, &'a str)>) -> (bool, &'static str) {
    let mut tagged = false;
    let mut kind = "other";
    for (k, _v) in tags {
        // Housekeeping tags don't make a node a "feature".
        match k {
            "created_by" | "source" | "note" | "fixme" | "comment" | "attribution" => continue,
            _ => {}
        }
        tagged = true;
        if kind == "other" {
            let cat = categorize_key(k);
            if cat != "other" {
                kind = cat;
            }
        }
    }
    (tagged, if tagged { kind } else { "vertex" })
}

fn categorize_key(k: &str) -> &'static str {
    if k == "building" || k.starts_with("building:") {
        return "building";
    }
    match k {
        "highway" | "railway" | "crossing" | "traffic_calming" | "barrier" | "aeroway" => "transport",
        "amenity" | "shop" | "tourism" | "leisure" | "office" | "craft" | "healthcare" => "poi",
        "natural" | "waterway" | "water" | "landuse" | "place" | "boundary" => "land",
        "power" | "man_made" | "emergency" | "historic" => "infra",
        _ => "other",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal B: changeset activity ("who maps, with what, when")
// ─────────────────────────────────────────────────────────────────────────────

/// Owned subset of a `<changeset>` element we care about.
struct ChangesetMeta {
    ts_ms: i64,
    num_changes: f64,
    user: String,
    min_lat: f64,
    min_lon: f64,
    max_lat: f64,
    max_lon: f64,
}

fn run_changesets(
    args: &Args,
    bbox: &Bbox,
    start_ms: i64,
    end_ms: i64,
    intermediate: &PathBuf,
    build_stt: bool,
) -> Result<()> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    println!("📡 Streaming changesets from {}", args.input.display());
    println!("   bbox: {:?}", bbox);
    if args.max_bbox_deg > 0.0 {
        println!("   dropping changesets wider than {}° (imports/bots)", args.max_bbox_deg);
    }

    let typed_columns = vec![
        // Editor "era" — the showcase color key.
        PropertyColumn::string("editor"),
        PropertyColumn::numeric("num_changes"),
        PropertyColumn::string("user"),
        PropertyColumn::string("year"),
    ];
    let mut writer = common::StreamingParquetWriter::with_columns(intermediate, typed_columns)?;

    // bz2 → buffered text → streaming XML. MultiBzDecoder handles both single-
    // and multi-stream bz2 (the planet dump is large; never fully in RAM).
    let file = File::open(&args.input)
        .with_context(|| format!("opening {}", args.input.display()))?;
    let decoder = bzip2::read::MultiBzDecoder::new(BufReader::with_capacity(1 << 20, file));
    let mut xml = Reader::from_reader(BufReader::with_capacity(1 << 20, decoder));

    let mut buf = Vec::new();
    let mut tag_buf = Vec::new();
    let mut scanned: u64 = 0;
    let mut kept: u64 = 0;

    loop {
        buf.clear();
        match xml.read_event_into(&mut buf)? {
            Event::Eof => break,
            // A changeset with no child tags (no created_by) — editor unknown.
            Event::Empty(e) if e.name().as_ref() == b"changeset" => {
                scanned += 1;
                if let Some(meta) = parse_changeset_attrs(&e)? {
                    maybe_emit_changeset(
                        &mut writer, &meta, "other", bbox, start_ms, end_ms, args.max_bbox_deg, &mut kept,
                    )?;
                }
            }
            // A changeset with child <tag> elements — read them for created_by.
            Event::Start(e) if e.name().as_ref() == b"changeset" => {
                scanned += 1;
                let meta = parse_changeset_attrs(&e)?;
                let mut editor = "other".to_string();
                loop {
                    tag_buf.clear();
                    match xml.read_event_into(&mut tag_buf)? {
                        Event::Empty(t) | Event::Start(t) if t.name().as_ref() == b"tag" => {
                            let (mut k, mut v) = (None, None);
                            for attr in t.attributes().flatten() {
                                match attr.key.as_ref() {
                                    b"k" => k = Some(attr.unescape_value()?.into_owned()),
                                    b"v" => v = Some(attr.unescape_value()?.into_owned()),
                                    _ => {}
                                }
                            }
                            if k.as_deref() == Some("created_by") {
                                if let Some(v) = v {
                                    editor = normalize_editor(&v).to_string();
                                }
                            }
                        }
                        Event::End(end) if end.name().as_ref() == b"changeset" => break,
                        Event::Eof => break,
                        _ => {}
                    }
                }
                if let Some(meta) = meta {
                    maybe_emit_changeset(
                        &mut writer, &meta, &editor, bbox, start_ms, end_ms, args.max_bbox_deg, &mut kept,
                    )?;
                }
            }
            _ => {}
        }
        if scanned % 5_000_000 == 0 && scanned > 0 {
            println!("   …scanned {}M changesets, kept {}", scanned / 1_000_000, kept);
        }
    }

    let total = writer.finish()?;
    println!(
        "\n📊 Scanned {} changesets; kept {} in bbox/time window ({} rows written)",
        scanned, kept, total
    );
    anyhow::ensure!(total > 0, "No changesets matched the bbox/time filters.");

    if build_stt {
        let summary = if args.summary_tier {
            // `count` (the implicit per-cell changeset/session count) plus
            // summed edit volume → the showcase toggles between the two.
            Some(common::SttBuildSummaryOptions {
                scheme: "h3".to_string(),
                min_zoom: Some(8),
                max_zoom: Some(12),
                columns: "num_changes:sum".to_string(),
            })
        } else {
            None
        };
        common::run_stt_build_with_full_options(common::SttBuildOptions {
            input: intermediate.clone(),
            output: args.output.clone(),
            time_field: "timestamp".to_string(),
            end_time_field: None,
            min_zoom: 8,
            max_zoom: 14,
            compression: "zstd".to_string(),
            // Monthly base bucket over ~20 years; coarser LOD at low zoom.
            temporal_bucket: Some("30d".to_string()),
            // No temporal LOD: these archives are rendered as RAW per-feature
            // points (colored by year/editor) and revealed cumulatively. The
            // LOD aggregate tiers coarsen time and drop per-feature property
            // columns, so the point layer would fall back to a constant color
            // and show aggregate-tile artifacts at low zoom. The H3 summary
            // tier (changesets) handles the low-zoom overview instead.
            temporal_lod: None,
            summary,
            // Split each 30d summary bucket into ~daily sub-windows so the
            // summary tier animates smoothly with no data re-upload.
            summary_sub_buckets: if args.summary_tier { Some(30) } else { None },
            min_features_per_tile: None,
            min_zoom_field: None,
            max_zoom_field: None,
        })?;
        let _ = std::fs::remove_file(intermediate);
    }

    println!("\n✅ OSM changeset-activity dataset complete!");
    Ok(())
}

/// Pull the attributes we need off a `<changeset>` start/empty element. Returns
/// `None` when the changeset has no bounding box (open/empty changesets) — those
/// can't be placed on the map.
fn parse_changeset_attrs(e: &quick_xml::events::BytesStart) -> Result<Option<ChangesetMeta>> {
    let mut created_at: Option<String> = None;
    let mut num_changes = 0f64;
    let mut user = String::new();
    let (mut min_lat, mut min_lon, mut max_lat, mut max_lon) = (None, None, None, None);

    for attr in e.attributes().flatten() {
        let val = attr.unescape_value()?;
        match attr.key.as_ref() {
            b"created_at" => created_at = Some(val.into_owned()),
            b"num_changes" => num_changes = val.parse().unwrap_or(0.0),
            b"user" => user = val.into_owned(),
            b"min_lat" => min_lat = val.parse().ok(),
            b"min_lon" => min_lon = val.parse().ok(),
            b"max_lat" => max_lat = val.parse().ok(),
            b"max_lon" => max_lon = val.parse().ok(),
            _ => {}
        }
    }

    let (min_lat, min_lon, max_lat, max_lon) = match (min_lat, min_lon, max_lat, max_lon) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return Ok(None), // no bbox → not mappable
    };
    let ts_ms = match created_at.as_deref().and_then(parse_osm_time) {
        Some(ms) => ms,
        None => return Ok(None),
    };

    Ok(Some(ChangesetMeta {
        ts_ms,
        num_changes,
        user,
        min_lat,
        min_lon,
        max_lat,
        max_lon,
    }))
}

#[allow(clippy::too_many_arguments)]
fn maybe_emit_changeset(
    writer: &mut common::StreamingParquetWriter,
    cs: &ChangesetMeta,
    editor: &str,
    bbox: &Bbox,
    start_ms: i64,
    end_ms: i64,
    max_bbox_deg: f64,
    kept: &mut u64,
) -> Result<()> {
    // Drop planet-spanning import/bot changesets so their centroid doesn't land
    // on Null Island.
    if max_bbox_deg > 0.0
        && ((cs.max_lat - cs.min_lat) > max_bbox_deg || (cs.max_lon - cs.min_lon) > max_bbox_deg)
    {
        return Ok(());
    }
    if cs.ts_ms < start_ms || cs.ts_ms > end_ms {
        return Ok(());
    }
    let clon = (cs.min_lon + cs.max_lon) / 2.0;
    let clat = (cs.min_lat + cs.max_lat) / 2.0;
    // Keep only points whose centroid lands inside the metro (after the
    // giant-bbox filter, remaining changesets are local).
    if !bbox.contains(clon, clat) {
        return Ok(());
    }

    let dt = match DateTime::<Utc>::from_timestamp_millis(cs.ts_ms) {
        Some(d) => d,
        None => return Ok(()),
    };
    let mut props = Map::new();
    props.insert("editor".to_string(), json!(editor));
    props.insert("num_changes".to_string(), json!(cs.num_changes));
    props.insert("user".to_string(), json!(cs.user));
    props.insert("year".to_string(), json!(dt.year().to_string()));

    let record = PointRecord::new(clon, clat, dt, props);
    writer.write_point(&record)?;
    *kept += 1;
    Ok(())
}

/// Normalise a changeset `created_by` tag into a stable editor-era category for
/// color-by-editor in the showcase.
fn normalize_editor(created_by: &str) -> &'static str {
    let s = created_by.to_ascii_lowercase();
    if s.starts_with("id ") || s.starts_with("id-") || s == "id" || s.starts_with("ideditor") {
        "iD"
    } else if s.starts_with("rapid") {
        "Rapid"
    } else if s.contains("josm") {
        "JOSM"
    } else if s.starts_with("potlatch") {
        "Potlatch"
    } else if s.starts_with("streetcomplete") {
        "StreetComplete"
    } else if s.starts_with("vespucci") {
        "Vespucci"
    } else if s.starts_with("merkaartor") {
        "Merkaartor"
    } else if s.starts_with("go map") || s.starts_with("gomap") {
        "Go Map!!"
    } else if s.starts_with("maps.me") || s.starts_with("mapsme") {
        "Maps.me"
    } else if s.starts_with("organicmaps") || s.starts_with("organic maps") {
        "Organic Maps"
    } else if s.contains("bot") || s.contains("import") || s.starts_with("bulk_upload") || s.starts_with("osmsync") {
        "bot/import"
    } else {
        "other"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared time helpers
// ─────────────────────────────────────────────────────────────────────────────

/// `YYYY-MM-DD` → epoch ms at the start (or end) of that UTC day.
fn date_to_ms(s: &str, end_of_day: bool) -> Result<i64> {
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .with_context(|| format!("invalid date '{s}', expected YYYY-MM-DD"))?;
    let t = if end_of_day {
        d.and_hms_opt(23, 59, 59).unwrap()
    } else {
        d.and_hms_opt(0, 0, 0).unwrap()
    };
    Ok(t.and_utc().timestamp_millis())
}

/// Parse an OSM timestamp (`2008-01-01T00:00:00Z`) into epoch ms.
fn parse_osm_time(s: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_normalization() {
        assert_eq!(normalize_editor("iD 2.20.2"), "iD");
        assert_eq!(normalize_editor("JOSM/1.5 (18000 en)"), "JOSM");
        assert_eq!(normalize_editor("Potlatch 2"), "Potlatch");
        assert_eq!(normalize_editor("StreetComplete 50.2"), "StreetComplete");
        assert_eq!(normalize_editor("RapiD 1.1.9"), "Rapid");
        assert_eq!(normalize_editor("bulk_upload.py"), "bot/import");
        assert_eq!(normalize_editor("some-import-tool"), "bot/import");
        assert_eq!(normalize_editor("HandMadeBySomeone"), "other");
    }

    #[test]
    fn tag_classification() {
        assert_eq!(classify_tags(std::iter::empty()), (false, "vertex"));
        assert_eq!(
            classify_tags([("created_by", "iD")].into_iter()),
            (false, "vertex"),
            "housekeeping-only tags don't make a feature"
        );
        assert_eq!(classify_tags([("building", "yes")].into_iter()), (true, "building"));
        assert_eq!(classify_tags([("amenity", "cafe")].into_iter()), (true, "poi"));
        assert_eq!(classify_tags([("natural", "tree")].into_iter()), (true, "land"));
        assert_eq!(classify_tags([("ref", "A1")].into_iter()), (true, "other"));
    }

    #[test]
    fn bbox_contains() {
        let b = Bbox { min_lon: -74.27, min_lat: 40.49, max_lon: -73.68, max_lat: 40.92 };
        assert!(b.contains(-73.98, 40.75)); // Manhattan
        assert!(!b.contains(0.0, 0.0)); // Null Island
    }

    #[test]
    fn osm_time_parsing() {
        assert_eq!(parse_osm_time("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_osm_time("not-a-date"), None);
        assert!(parse_osm_time("2020-01-01T00:00:00Z").unwrap() > 0);
    }
}
