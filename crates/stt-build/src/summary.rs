//! Build the server-aggregated summary tier alongside the raw tier.
//!
//! At low zooms, rendering 100M+ raw features is infeasible — every frame
//! the GPU would push the entire dataset through vertex / fragment stages.
//! Public 100M-scale visualisations (CARTO, Foursquare, Kepler.gl Cloud)
//! solve this by pre-aggregating into hexagonal / quadbin cells server-side
//! and only sending RAW features at high enough zooms that the on-screen
//! count is bounded.
//!
//! This module is the build-time half of that strategy. It walks each
//! [`ParsedFeature`], assigns it to an H3 cell at the resolution configured
//! for the tile's zoom, accumulates per-cell aggregates (count, sum, mean,
//! min, max), and emits a [`GeneratedTile`] per (zoom, x, y, time-bucket)
//! whose features are one row per cell.
//!
//! The aggregate tiles live in the same archive directory as the raw tiles
//! but are tagged with a distinct layer name (`summary` by default). The TS
//! reader uses that layer name + the `summary_tier` metadata block to
//! dispatch between summary and raw rendering.

use crate::input::ParsedFeature;
use crate::quadbin;
use crate::tiler::{GeneratedTile, TileWriter};
use anyhow::Result;
use h3o::{LatLng, Resolution};
use std::collections::{BTreeMap, HashMap};
use stt_core::arrow_tile::{ColumnarLayer, Coord, GeometryColumn, PropertyColumn};
use stt_core::metadata::{
    SummaryAggregation, SummaryColumn, SummaryScheme, SummaryTier,
};
use stt_core::projection;
use stt_core::tile::TileId;

/// Configuration for emitting a summary tier.
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    /// Aggregation scheme. Only [`SummaryScheme::H3`] is implemented today.
    pub scheme: SummaryScheme,
    /// Inclusive minimum zoom level for which summary tiles are produced.
    pub min_zoom: u8,
    /// Inclusive maximum zoom level for which summary tiles are produced.
    pub max_zoom: u8,
    /// Temporal bucket size (ms) for chunking summary tiles into aligned
    /// intervals. Should match the raw-tier `temporal_bucket_ms`.
    pub temporal_bucket_ms: u64,
    /// Aggregated columns (in addition to the implicit `count`).
    pub columns: Vec<SummaryColumn>,
    /// Layer name carried in the emitted tile frames. Defaults to "summary".
    pub layer_name: String,
    /// Number of fine-grained sub-buckets PER tile-temporal-bucket. When
    /// > 1, each cell carries N additional `bucket_<i>` numeric columns
    /// (one count per sub-bucket). The renderer animates time by changing
    /// a uniform that indexes the array — zero re-upload between frames
    /// inside a tile. Default 1 = legacy single-count behaviour.
    pub sub_buckets: u32,
}

impl SummaryConfig {
    /// Build a summary tier descriptor from this configuration. The
    /// `cell_resolution_per_zoom` table is computed by [`h3_resolution_for_zoom`]
    /// for H3, and by `zoom` directly for quadbin.
    pub fn to_tier(&self) -> SummaryTier {
        let mut resolutions: Vec<u8> = Vec::with_capacity(
            (self.max_zoom - self.min_zoom + 1) as usize,
        );
        for z in self.min_zoom..=self.max_zoom {
            resolutions.push(match self.scheme {
                SummaryScheme::H3 => h3_resolution_for_zoom(z),
                SummaryScheme::Quadbin => quadbin_zoom_for_zoom(z),
            });
        }
        SummaryTier {
            scheme: self.scheme,
            min_zoom: self.min_zoom,
            max_zoom: self.max_zoom,
            cell_resolution_per_zoom: resolutions,
            columns: self.columns.clone(),
            layer_name: self.layer_name.clone(),
            sub_buckets: self.sub_buckets.max(1),
        }
    }
}

/// Map a tile zoom to an H3 resolution. We follow the (rough) Uber H3
/// "hexbin sizing" guidance: H3 res 0 covers ~4,250,000 km², which is in
/// the same order of magnitude as a single web-Mercator tile at zoom 0.
/// Each web-Mercator zoom step roughly quarters the tile area; one H3
/// resolution step does the same (~7x). The 1-step-per-zoom map is a
/// little coarser than perfect parity but the on-screen pixel density
/// works out well across the typical map zoom range.
///
/// See https://h3geo.org/docs/core-library/restable
pub fn h3_resolution_for_zoom(zoom: u8) -> u8 {
    // Zoom 0 → res 0 (planet-scale hexes), zoom 5 → res 5, etc.
    // Clamp at H3's max resolution (15) — we never actually want res > 6 or
    // so in practice because the *raw* tiles take over above that point.
    zoom.min(15)
}

/// Quadbin summary cells are FINER than the view zoom by this many levels.
///
/// Quadbin cells ARE XYZ tiles, so a 1:1 map-zoom→quadbin-zoom mapping makes
/// each cell exactly the web-Mercator tile at the view zoom — and a viewport
/// only ever spans ~2-4 tiles, so a 1:1 summary renders as a handful of giant
/// cells (useless as a density overview). H3's res-N hexes are ~hundreds of
/// metres at the same N, so H3's 1:1 mapping already yields a fine grid; to
/// reach comparable usefulness, Quadbin offsets the cell resolution finer so a
/// viewport shows a readable grid of cells. Each summary tile at map-zoom `z`
/// then holds (≈4^OFFSET) cells at quadbin-zoom `z + OFFSET`.
pub const QUADBIN_RES_OFFSET: u8 = 3;

/// Map a tile zoom to a CARTO Quadbin cell zoom level — the view zoom plus
/// [`QUADBIN_RES_OFFSET`] so cells are finer than the tile that carries them.
/// Clamp at the Quadbin band's max zoom (26) so the 5-bit zoom field and
/// 52-bit Morton payload never overflow.
pub fn quadbin_zoom_for_zoom(zoom: u8) -> u8 {
    zoom.saturating_add(QUADBIN_RES_OFFSET).min(26)
}

/// Per-cell running aggregate for a single tile + time bucket.
#[derive(Debug, Default)]
struct CellAggregate {
    /// Number of raw features that fell into this cell.
    count: u64,
    /// Minimum start timestamp observed (Unix ms).
    time_start: i64,
    /// Maximum end timestamp observed (Unix ms).
    time_end: i64,
    /// Per-source-column accumulators, indexed by the position of the column
    /// in [`SummaryConfig::columns`]. Each `Accumulator` stores enough state
    /// to emit any of sum/mean/min/max, since the same source column may be
    /// referenced by several entries with different `agg`s.
    ///
    /// Indexed by the SOURCE column name rather than the aggregation entry
    /// to avoid recomputing the same sum twice.
    sources: HashMap<String, Accumulator>,
    /// Per-sub-bucket counts. Length = SummaryConfig.sub_buckets. The
    /// rendered animation walks this array to drive the per-cell fill
    /// colour via a uniform sub-bucket index — zero data re-upload
    /// between frames inside a tile. Empty when sub_buckets == 1.
    sub_bucket_counts: Vec<u32>,
}

#[derive(Debug, Default, Clone, Copy)]
struct Accumulator {
    sum: f64,
    sum_count: u64,
    min: f64,
    max: f64,
    has_any: bool,
}

impl Accumulator {
    fn observe(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        if !self.has_any {
            self.min = v;
            self.max = v;
            self.has_any = true;
        } else {
            if v < self.min {
                self.min = v;
            }
            if v > self.max {
                self.max = v;
            }
        }
        self.sum += v;
        self.sum_count += 1;
    }
}

/// Build the summary-tier tiles for `features` and stream them to `writer`.
///
/// One Arrow layer is produced per (zoom, tile_x, tile_y, time_bucket):
/// each row is one cell with columns `cell_id`, `count`, plus the columns
/// listed in [`SummaryConfig::columns`].
///
/// Returns the number of tiles written.
pub fn build_summary_tier<W: TileWriter>(
    features: &[ParsedFeature],
    config: &SummaryConfig,
    writer: &mut W,
) -> Result<usize> {
    let bucket_ms = config.temporal_bucket_ms.max(1);
    let sub_buckets = config.sub_buckets.max(1) as usize;
    let sub_bucket_ms = if sub_buckets > 1 {
        (bucket_ms as usize / sub_buckets).max(1) as u64
    } else {
        bucket_ms
    };
    let mut total = 0usize;

    for zoom in config.min_zoom..=config.max_zoom {
        // Cell binning is scheme-dependent. For H3 we resolve the configured
        // resolution into an `h3o::Resolution`; for Quadbin the resolution
        // table value IS the Quadbin zoom level (clamped to the 0..=26 band).
        let h3_res = match config.scheme {
            SummaryScheme::H3 => {
                let h3_res_u8 = h3_resolution_for_zoom(zoom);
                Some(Resolution::try_from(h3_res_u8).map_err(|e| {
                    anyhow::anyhow!("invalid H3 resolution {h3_res_u8} for zoom {zoom}: {e:?}")
                })?)
            }
            SummaryScheme::Quadbin => None,
        };
        let quad_z = quadbin_zoom_for_zoom(zoom);

        // (tile_x, tile_y, time_bucket) -> (cell_id -> aggregate)
        let mut buckets: BTreeMap<(u32, u32, u64), HashMap<u64, CellAggregate>> =
            BTreeMap::new();

        for feature in features {
            let (tx, ty) =
                match projection::lonlat_to_tile(feature.lon, feature.lat, zoom) {
                    Ok(xy) => xy,
                    Err(_) => continue,
                };
            let bucket_start = (feature.timestamp / bucket_ms) * bucket_ms;
            let cell_id: u64 = match config.scheme {
                SummaryScheme::H3 => {
                    let res = h3_res.expect("H3 resolution resolved above");
                    match LatLng::new(feature.lat, feature.lon) {
                        Ok(ll) => ll.to_cell(res).into(),
                        Err(_) => continue, // out-of-range lat/lon (e.g. NaN)
                    }
                }
                SummaryScheme::Quadbin => {
                    // Web-Mercator can't represent NaN/inf lat/lon — skip them
                    // so a malformed row never poisons a cell.
                    if !feature.lon.is_finite() || !feature.lat.is_finite() {
                        continue;
                    }
                    quadbin::lonlat_to_quadbin(feature.lon, feature.lat, quad_z)
                }
            };

            let bucket = buckets
                .entry((tx, ty, bucket_start))
                .or_default();
            let agg = bucket.entry(cell_id).or_default();

            if sub_buckets > 1 {
                if agg.sub_bucket_counts.is_empty() {
                    agg.sub_bucket_counts = vec![0u32; sub_buckets];
                }
                // Index within this outer bucket.
                let sub_idx = (((feature.timestamp - bucket_start) / sub_bucket_ms)
                    as usize)
                    .min(sub_buckets - 1);
                agg.sub_bucket_counts[sub_idx] = agg.sub_bucket_counts[sub_idx]
                    .saturating_add(1);
            }
            agg.count += 1;
            let t = feature.timestamp as i64;
            let t_end = feature.end_timestamp.unwrap_or(feature.timestamp) as i64;
            if agg.count == 1 {
                agg.time_start = t;
                agg.time_end = t_end;
            } else {
                if t < agg.time_start {
                    agg.time_start = t;
                }
                if t_end > agg.time_end {
                    agg.time_end = t_end;
                }
            }

            // Observe each source column once. Repeated `agg` entries for
            // the same source share the accumulator.
            if let Some(props) = feature.shared_properties.as_deref() {
                for col in &config.columns {
                    if matches!(col.agg, SummaryAggregation::Count) {
                        continue;
                    }
                    // Read the source numeric robustly: a value that arrived
                    // string-encoded (e.g. from a legacy all-string writer)
                    // still aggregates instead of silently dropping to None.
                    let v = props.get(&col.name).and_then(|v| {
                        v.as_f64()
                            .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
                    });
                    if let Some(v) = v {
                        agg.sources
                            .entry(col.name.clone())
                            .or_default()
                            .observe(v);
                    }
                }
            }
        }

        // Emit one tile per (tile_x, tile_y, time_bucket).
        for ((tx, ty, bucket_start), cells) in buckets {
            if cells.is_empty() {
                continue;
            }
            let layer =
                build_summary_layer(config.scheme, &config.layer_name, &config.columns, &cells);
            let time_end = cells
                .values()
                .map(|c| c.time_end)
                .max()
                .unwrap_or((bucket_start + bucket_ms) as i64);
            // Tight lower bound: earliest observed time across the cells (each
            // cell's `time_start` is its min observed time, not the bucket edge).
            let cover_t_min = cells
                .values()
                .map(|c| c.time_start)
                .min()
                .unwrap_or(bucket_start as i64);
            let tile = GeneratedTile {
                id: TileId::new(zoom, tx, ty, bucket_start),
                time_start: bucket_start as i64,
                time_end,
                cover_t_min,
                layers: vec![layer],
            };
            writer.write_tile(&tile)?;
            total += 1;
        }
    }

    Ok(total)
}

/// Build a single Arrow summary layer from a cell map.
///
/// Schema:
/// - `cell_id` — UInt64 (carried as feature_ids; the standard Arrow tile
///   schema already has a `id: UInt64` column).
/// - `count` — Float64 numeric property (kept as f64 for parity with the
///   other aggregated columns; the TS reader keeps numeric props as f32 so
///   we never need to differentiate integer-vs-float on the wire).
/// - per [`SummaryConfig::columns`] entry, one numeric property column.
///
/// Geometry is the cell centroid as a Point, which gives the reader a
/// representative lon/lat to use for picking and quick "no-cell-rendering"
/// fallbacks. The actual cell vertices are reconstructable from the
/// `cell_id` on the client (h3-js for H3, the Quadbin → quadkey path for
/// Quadbin).
fn build_summary_layer(
    scheme: SummaryScheme,
    name: &str,
    columns: &[SummaryColumn],
    cells: &HashMap<u64, CellAggregate>,
) -> ColumnarLayer {
    let n = cells.len();
    // Stable ordering: sort by cell_id so successive builds with the same
    // input produce identical tile bytes.
    let mut entries: Vec<(u64, &CellAggregate)> =
        cells.iter().map(|(k, v)| (*k, v)).collect();
    entries.sort_by_key(|(k, _)| *k);

    let mut feature_ids: Vec<u64> = Vec::with_capacity(n);
    let mut start_times: Vec<i64> = Vec::with_capacity(n);
    let mut end_times: Vec<i64> = Vec::with_capacity(n);
    let mut centroids: Vec<Coord> = Vec::with_capacity(n);
    let mut counts: Vec<Option<f64>> = Vec::with_capacity(n);

    // Per-output-column buckets keyed by index into `columns`.
    let mut per_column: Vec<Vec<Option<f64>>> =
        columns.iter().map(|_| Vec::with_capacity(n)).collect();

    for (cell_id, agg) in &entries {
        feature_ids.push(*cell_id);
        start_times.push(agg.time_start);
        end_times.push(agg.time_end);
        counts.push(Some(agg.count as f64));

        // Cell centroid (lon, lat). The cell id self-describes its geometry
        // for both schemes: h3o recovers the H3 hexagon centre via
        // `CellIndex -> LatLng`; for Quadbin we decode the (z, x, y) tile out
        // of the u64 and take its centre.
        let centroid: Coord = match scheme {
            SummaryScheme::H3 => match h3o::CellIndex::try_from(*cell_id) {
                Ok(c) => {
                    let ll: LatLng = c.into();
                    [ll.lng(), ll.lat()]
                }
                Err(_) => [0.0, 0.0],
            },
            SummaryScheme::Quadbin => {
                let (z, x, y) = quadbin::quadbin_to_tile(*cell_id);
                quadbin::quadbin_centroid(z, x, y)
            }
        };
        centroids.push(centroid);

        for (i, col) in columns.iter().enumerate() {
            let acc = agg.sources.get(&col.name);
            let v: Option<f64> = match col.agg {
                SummaryAggregation::Count => Some(agg.count as f64),
                SummaryAggregation::Sum => acc.and_then(|a| {
                    if a.has_any { Some(a.sum) } else { None }
                }),
                SummaryAggregation::Mean => acc.and_then(|a| {
                    if a.sum_count > 0 {
                        Some(a.sum / a.sum_count as f64)
                    } else {
                        None
                    }
                }),
                SummaryAggregation::Min => acc.and_then(|a| {
                    if a.has_any { Some(a.min) } else { None }
                }),
                SummaryAggregation::Max => acc.and_then(|a| {
                    if a.has_any { Some(a.max) } else { None }
                }),
            };
            per_column[i].push(v);
        }
    }

    let mut properties: Vec<(String, PropertyColumn)> = Vec::new();
    properties.push(("count".to_string(), PropertyColumn::Numeric(counts)));

    // Per-sub-bucket numeric columns (Phase C). When sub_buckets > 1, each
    // cell row carries N additional `bucket_<i>` columns with the count
    // observed in that fine-grained sub-bucket. The renderer reads them as
    // standard numericProps and indexes via a `currentSubBucket` uniform —
    // animation through time inside an outer-bucket tile is one shader
    // uniform update, no data re-upload.
    let sub_bucket_n = entries
        .iter()
        .map(|(_, a)| a.sub_bucket_counts.len())
        .max()
        .unwrap_or(0);
    if sub_bucket_n > 0 {
        for sub_idx in 0..sub_bucket_n {
            let mut col_vals: Vec<Option<f64>> = Vec::with_capacity(entries.len());
            for (_, agg) in &entries {
                let v = agg
                    .sub_bucket_counts
                    .get(sub_idx)
                    .copied()
                    .unwrap_or(0) as f64;
                col_vals.push(Some(v));
            }
            properties.push((
                format!("bucket_{}", sub_idx),
                PropertyColumn::Numeric(col_vals),
            ));
        }
    }

    for (i, col) in columns.iter().enumerate() {
        if matches!(col.agg, SummaryAggregation::Count) {
            // Already emitted as the implicit `count` column.
            continue;
        }
        let prop_name = output_column_name(col);
        properties.push((prop_name, PropertyColumn::Numeric(per_column[i].clone())));
    }

    ColumnarLayer {
        name: name.to_string(),
        feature_ids,
        start_times,
        end_times,
        geometry: GeometryColumn::Point(centroids),
        vertex_times: None,
        vertex_values: None,
        properties,
        // Summary tiles are always point centroids — no polygons to
        // tessellate, so the sidecar column added by the pre-tessellate
        // track is irrelevant.
        triangles: None,
        vertex_value_matrix: None,
    }
}

/// On-wire property name for an aggregated column. Tests assert this is
/// `<agg>_<source>` (e.g. `mean_magnitude`).
pub fn output_column_name(col: &SummaryColumn) -> String {
    let prefix = match col.agg {
        SummaryAggregation::Count => "count",
        SummaryAggregation::Sum => "sum",
        SummaryAggregation::Mean => "mean",
        SummaryAggregation::Min => "min",
        SummaryAggregation::Max => "max",
    };
    if matches!(col.agg, SummaryAggregation::Count) {
        "count".to_string()
    } else {
        format!("{prefix}_{}", col.name)
    }
}

/// Parse `--summary-columns` syntax: `magnitude:mean,magnitude:max,depth:sum,count`.
///
/// `count` is special-cased (no source column needed) and produces a single
/// implicit aggregate.
pub fn parse_summary_columns(spec: &str) -> Result<Vec<SummaryColumn>> {
    let mut out = Vec::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        if entry.eq_ignore_ascii_case("count") {
            out.push(SummaryColumn {
                name: "_count".to_string(),
                agg: SummaryAggregation::Count,
            });
            continue;
        }
        let (name, agg) = entry.split_once(':').ok_or_else(|| {
            anyhow::anyhow!(
                "summary column '{entry}' must be 'name:agg' (agg in sum|mean|min|max|count)"
            )
        })?;
        let agg = match agg.trim().to_ascii_lowercase().as_str() {
            "sum" => SummaryAggregation::Sum,
            "mean" | "avg" | "average" => SummaryAggregation::Mean,
            "min" => SummaryAggregation::Min,
            "max" => SummaryAggregation::Max,
            "count" => SummaryAggregation::Count,
            other => anyhow::bail!(
                "unknown summary aggregation '{other}' (use sum|mean|min|max|count)"
            ),
        };
        out.push(SummaryColumn {
            name: name.trim().to_string(),
            agg,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiler::GeneratedTile;
    use geojson::{Feature, Geometry, Value as GeomValue};

    fn point(lon: f64, lat: f64, ts: u64, mag: f64) -> ParsedFeature {
        let props = serde_json::json!({ "magnitude": mag })
            .as_object()
            .cloned()
            .map(std::sync::Arc::new);
        ParsedFeature {
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::Point(vec![lon, lat]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: props,
            timestamp: ts,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon,
            lat,
        }
    }

    struct CollectingWriter {
        tiles: Vec<GeneratedTile>,
    }
    impl TileWriter for CollectingWriter {
        fn write_tile(&mut self, tile: &GeneratedTile) -> anyhow::Result<()> {
            // Clone the relevant bits — we only inspect feature_count + layer
            // names in the assertions, so a deep clone is overkill.
            self.tiles.push(GeneratedTile {
                id: tile.id,
                time_start: tile.time_start,
                time_end: tile.time_end,
                cover_t_min: tile.cover_t_min,
                layers: tile.layers.clone(),
            });
            Ok(())
        }
    }

    #[test]
    fn h3_resolution_per_zoom_clamps() {
        assert_eq!(h3_resolution_for_zoom(0), 0);
        assert_eq!(h3_resolution_for_zoom(5), 5);
        assert_eq!(h3_resolution_for_zoom(16), 15); // clamped
    }

    #[test]
    fn parses_column_spec() {
        let cols = parse_summary_columns("magnitude:mean, magnitude:max, count").unwrap();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].name, "magnitude");
        assert!(matches!(cols[0].agg, SummaryAggregation::Mean));
        assert!(matches!(cols[2].agg, SummaryAggregation::Count));
    }

    #[test]
    fn build_summary_tier_emits_aggregate_tiles() {
        // Cluster 5 points in San Francisco; 1 point in London. At zoom 0
        // they should fall into the same world tile but two different H3
        // cells.
        let mut features = Vec::new();
        // SF cluster:
        for i in 0..5 {
            features.push(point(-122.45 + 0.001 * i as f64, 37.77, 1_000_000, 5.0 + i as f64));
        }
        // London:
        features.push(point(-0.1278, 51.5074, 1_000_000, 6.0));

        let config = SummaryConfig {
            scheme: SummaryScheme::H3,
            min_zoom: 0,
            max_zoom: 2,
            temporal_bucket_ms: 3_600_000,
            columns: vec![
                SummaryColumn {
                    name: "magnitude".into(),
                    agg: SummaryAggregation::Mean,
                },
                SummaryColumn {
                    name: "magnitude".into(),
                    agg: SummaryAggregation::Max,
                },
                SummaryColumn {
                    name: "_count".into(),
                    agg: SummaryAggregation::Count,
                },
            ],
            layer_name: "summary".into(),
            sub_buckets: 1,
        };

        let mut writer = CollectingWriter { tiles: Vec::new() };
        let n = build_summary_tier(&features, &config, &mut writer).unwrap();
        assert!(n > 0, "expected at least one summary tile");

        // Every emitted tile is a "summary" layer carrying at least
        // (count, mean_magnitude, max_magnitude).
        for tile in &writer.tiles {
            assert_eq!(tile.layers.len(), 1);
            let layer = &tile.layers[0];
            assert_eq!(layer.name, "summary");
            let prop_names: Vec<&str> =
                layer.properties.iter().map(|(n, _)| n.as_str()).collect();
            assert!(prop_names.contains(&"count"));
            assert!(prop_names.contains(&"mean_magnitude"));
            assert!(prop_names.contains(&"max_magnitude"));
        }

        // Total cell-rows across the zoom 0 tier should be small — at most
        // a handful of H3 cells (SF cluster collapses to one cell at res 0).
        let z0_total: usize = writer
            .tiles
            .iter()
            .filter(|t| t.id.z == 0)
            .map(|t| t.feature_count() as usize)
            .sum();
        assert!(z0_total <= 5, "expected ≤5 cells at zoom 0, got {z0_total}");
    }

    #[test]
    fn build_summary_tier_quadbin_aggregates_into_one_cell() {
        // A tight cluster of points sits inside a single Quadbin cell at a
        // modest zoom; aggregation must collapse them to one summary row whose
        // `count` equals the cluster size. A distant point lands in its own
        // cell, so we never see fewer than two cells across the world tier.
        let n_cluster = 4u64;
        let mut features = Vec::new();
        for i in 0..n_cluster {
            // ~SF, sub-cell spacing so they share a Quadbin cell at z=8.
            features.push(point(-122.4194 + 0.0001 * i as f64, 37.7749, 1_000_000, 5.0));
        }
        // A far-away point (Tokyo) to confirm distinct cells stay distinct.
        features.push(point(139.6917, 35.6895, 1_000_000, 9.0));

        let config = SummaryConfig {
            scheme: SummaryScheme::Quadbin,
            min_zoom: 8,
            max_zoom: 8,
            temporal_bucket_ms: 3_600_000,
            columns: vec![SummaryColumn {
                name: "_count".into(),
                agg: SummaryAggregation::Count,
            }],
            layer_name: "summary".into(),
            sub_buckets: 1,
        };

        let mut writer = CollectingWriter { tiles: Vec::new() };
        let n = build_summary_tier(&features, &config, &mut writer).unwrap();
        assert!(n > 0, "expected at least one quadbin summary tile");

        // Every emitted feature id is a well-formed Quadbin u64 (header nibble
        // 0b100, mode bit set, zoom field == build zoom + QUADBIN_RES_OFFSET),
        // and the cluster collapses to a single cell whose count == n_cluster.
        let mut saw_cluster_count = false;
        let mut total_cells = 0usize;
        for tile in &writer.tiles {
            assert_eq!(tile.layers.len(), 1);
            let layer = &tile.layers[0];
            assert_eq!(layer.name, "summary");
            total_cells += layer.feature_ids.len();
            for &id in &layer.feature_ids {
                assert_eq!(id >> 60, 0b100, "id not a Quadbin header: {id:#018x}");
                assert_eq!((id >> 59) & 1, 1, "Quadbin mode bit unset: {id:#018x}");
                assert_eq!(
                    ((id >> 52) & 0x1f) as u8,
                    8 + QUADBIN_RES_OFFSET,
                    "wrong zoom field: {id:#018x}"
                );
            }
            // The implicit `count` column carries the per-cell counts.
            let counts = layer
                .properties
                .iter()
                .find(|(n, _)| n == "count")
                .map(|(_, c)| c)
                .expect("count column present");
            if let PropertyColumn::Numeric(vals) = counts {
                if vals
                    .iter()
                    .any(|v| matches!(v, Some(c) if (*c - n_cluster as f64).abs() < f64::EPSILON))
                {
                    saw_cluster_count = true;
                }
            }
        }
        assert!(
            saw_cluster_count,
            "expected a cell with count == {n_cluster}"
        );
        // SF cluster (1 cell) + Tokyo (1 cell) = 2 distinct cells.
        assert_eq!(total_cells, 2, "expected exactly two quadbin cells");
    }

    #[test]
    fn build_summary_tier_emits_per_sub_bucket_columns_when_configured() {
        // 6 features spread across the same hour, hitting different
        // sub-bucket slots — sub-buckets must split the hour evenly.
        let mut features = Vec::new();
        let bucket_ms: u64 = 60 * 60 * 1000; // 1 hour
        let sub_buckets: u32 = 6; // 10 min per sub
        let sub_ms = bucket_ms / sub_buckets as u64;
        for i in 0..sub_buckets {
            features.push(point(
                -122.45,
                37.77,
                1_000_000_000_000 + (i as u64 * sub_ms),
                5.0,
            ));
        }
        let config = SummaryConfig {
            scheme: SummaryScheme::H3,
            min_zoom: 5,
            max_zoom: 5,
            temporal_bucket_ms: bucket_ms,
            columns: vec![],
            layer_name: "summary".into(),
            sub_buckets,
        };
        let mut writer = CollectingWriter { tiles: Vec::new() };
        let n = build_summary_tier(&features, &config, &mut writer).unwrap();
        assert!(n > 0);
        // Every tile carries `bucket_0..bucket_5` numeric columns.
        for tile in &writer.tiles {
            let names: Vec<&str> = tile.layers[0]
                .properties
                .iter()
                .map(|(n, _)| n.as_str())
                .collect();
            for i in 0..sub_buckets {
                assert!(
                    names.contains(&format!("bucket_{}", i).as_str()),
                    "tile missing bucket_{}: {names:?}",
                    i
                );
            }
        }
    }

    #[test]
    fn to_tier_emits_one_resolution_per_zoom() {
        let config = SummaryConfig {
            scheme: SummaryScheme::H3,
            min_zoom: 0,
            max_zoom: 4,
            temporal_bucket_ms: 3_600_000,
            columns: vec![],
            layer_name: "summary".into(),
            sub_buckets: 1,
        };
        let tier = config.to_tier();
        assert_eq!(tier.cell_resolution_per_zoom, vec![0, 1, 2, 3, 4]);
        assert_eq!(tier.resolution_for_zoom(3), 3);
    }
}
