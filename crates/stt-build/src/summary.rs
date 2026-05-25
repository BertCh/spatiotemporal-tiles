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
                SummaryScheme::Quadbin => z,
            });
        }
        SummaryTier {
            scheme: self.scheme,
            min_zoom: self.min_zoom,
            max_zoom: self.max_zoom,
            cell_resolution_per_zoom: resolutions,
            columns: self.columns.clone(),
            layer_name: self.layer_name.clone(),
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
    let mut total = 0usize;

    for zoom in config.min_zoom..=config.max_zoom {
        let h3_res_u8 = h3_resolution_for_zoom(zoom);
        let h3_res = Resolution::try_from(h3_res_u8).map_err(|e| {
            anyhow::anyhow!("invalid H3 resolution {h3_res_u8} for zoom {zoom}: {e:?}")
        })?;

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
            let cell = match LatLng::new(feature.lat, feature.lon) {
                Ok(ll) => ll.to_cell(h3_res),
                Err(_) => continue, // out-of-range lat/lon (e.g. NaN)
            };
            let cell_id: u64 = cell.into();

            let bucket = buckets
                .entry((tx, ty, bucket_start))
                .or_default();
            let agg = bucket.entry(cell_id).or_default();

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
                    let v = props
                        .get(&col.name)
                        .and_then(|v| v.as_f64());
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
            let layer = build_summary_layer(&config.layer_name, &config.columns, &cells);
            let time_end = cells
                .values()
                .map(|c| c.time_end)
                .max()
                .unwrap_or((bucket_start + bucket_ms) as i64);
            let tile = GeneratedTile {
                id: TileId::new(zoom, tx, ty, bucket_start),
                time_start: bucket_start as i64,
                time_end,
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
/// Geometry is the H3 cell centroid as a Point, which gives the reader a
/// representative lon/lat to use for picking and quick "no-hex-rendering"
/// fallbacks. The actual hexagon vertices are reconstructable from the
/// `cell_id` via h3-js on the client.
fn build_summary_layer(
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

        // Cell centroid (lon, lat). h3o exposes `cell.into() -> CellIndex`
        // already; we recover the centroid via LatLng::from(cell_index).
        let centroid: Coord = match h3o::CellIndex::try_from(*cell_id) {
            Ok(c) => {
                let ll: LatLng = c.into();
                [ll.lng(), ll.lat()]
            }
            Err(_) => [0.0, 0.0],
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
        properties,
        // Summary tiles are always point centroids — no polygons to
        // tessellate, so the sidecar column added by the pre-tessellate
        // track is irrelevant.
        triangles: None,
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
    fn to_tier_emits_one_resolution_per_zoom() {
        let config = SummaryConfig {
            scheme: SummaryScheme::H3,
            min_zoom: 0,
            max_zoom: 4,
            temporal_bucket_ms: 3_600_000,
            columns: vec![],
            layer_name: "summary".into(),
        };
        let tier = config.to_tier();
        assert_eq!(tier.cell_resolution_per_zoom, vec![0, 1, 2, 3, 4]);
        assert_eq!(tier.resolution_for_zoom(3), 3);
    }
}
