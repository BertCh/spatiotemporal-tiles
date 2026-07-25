//! Sample-encode measurement: push a deterministic sample of source features
//! through the real stt-core encoder (+ zstd) to replace formula-based size
//! estimates with measured bytes.
//!
//! The loader retains a stride sample of full geometries and property values
//! ([`crate::loader::SampledFeature`]); [`measure_sample`] groups the dominant
//! geometry kind into ONE synthetic tile layer, encodes it through
//! [`stt_core::arrow_tile::encode_tile_with`] (the production encoder, driven
//! by an explicit [`EncoderConfig`] so no process-globals are touched), zstd-
//! compresses the payload, and attributes per-column costs by re-encoding each
//! column alone — the same fair-share method stt-core's `point_column_stats`
//! example uses.

use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow::ipc::writer::StreamWriter;
use serde::{Deserialize, Serialize};
use stt_core::arrow_tile::{
    decode_tile, encode_tile_with, ColumnarLayer, Coord, EncoderConfig, GeometryColumn,
    PropertyColumn,
};
use stt_core::compression::compress_zstd_with_dict_level;

use crate::loader::{PropValue, SampledFeature};

/// Below this many usable sampled features the measurement is noise (zstd
/// framing and per-tile encoder overheads dominate the per-feature figure) and
/// [`measure_sample`] returns `None` — callers fall back to the formula
/// estimates. Set to 24: low enough that most real datasets get MEASURED bytes
/// (a real encode of two-dozen homogeneous features estimates bytes/feature and
/// the zstd ratio well within estimate tolerance), high enough that fixed frame
/// overhead doesn't materially skew the per-feature figure.
const MIN_MEASURE_FEATURES: usize = 24;

/// Encoder/compression settings for a sample measurement — the same levers a
/// real `stt-build` run exposes (`--zstd-level`, `--quantize-coords`,
/// `--quantize-attrs-auto`).
#[derive(Debug, Clone)]
pub struct MeasureSettings {
    /// zstd level applied to the encoded tile payload.
    pub zstd_level: i32,
    /// Fixed-point coordinate quantization ground precision in meters
    /// (`None` = Float64 coordinates, the build default).
    pub quantize_coords_m: Option<f64>,
    /// Range-adaptive `UInt16` quantization for Float64 numeric properties.
    pub quantize_attrs_auto: bool,
}

impl Default for MeasureSettings {
    /// The `stt-build` defaults: zstd level 3, no quantization.
    fn default() -> Self {
        Self {
            zstd_level: 3,
            quantize_coords_m: None,
            quantize_attrs_auto: false,
        }
    }
}

/// Measured encoder output for a sample: real compressed bytes per feature
/// plus per-column cost attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredEncoding {
    /// Source features actually encoded (the dominant-geometry-kind subset of
    /// the sample; multi-part geometries count once even though they encode
    /// as one row per part).
    pub features: usize,
    /// Encoder geometry kind that was measured (`"point"` / `"line"` /
    /// `"polygon"`) — for mixed-geometry sources, the dominant kind.
    pub geometry_kind: String,
    /// Compressed size of the encoded synthetic tile (tile payload + zstd).
    pub bytes_total: usize,
    /// `bytes_total / features`.
    pub bytes_per_feature: f64,
    /// Uncompressed-payload / compressed ratio (replaces the assumed 3x).
    pub zstd_ratio: f64,
    /// Per-column compressed cost, sorted descending by bytes.
    pub per_column: Vec<ColumnCost>,
}

/// Compressed cost of one encoded tile column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnCost {
    /// Column name (e.g. `geometry`, `start_time`, a property name).
    pub name: String,
    /// Bytes when this column is IPC-encoded alone and zstd-compressed.
    pub compressed_bytes: usize,
    /// Fraction of the summed per-column bytes in `[0, 1]`. Shares are of the
    /// per-column sum, which exceeds `bytes_total` (single-column re-encoding
    /// loses cross-column sharing and repays IPC framing per column).
    pub share: f64,
}

/// The encoder geometry bucket a sampled geometry falls into (the tiler emits
/// one layer per kind, so a synthetic layer must be single-kind).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeomKind {
    Point = 0,
    Line = 1,
    Polygon = 2,
}

impl GeomKind {
    fn name(self) -> &'static str {
        match self {
            GeomKind::Point => "point",
            GeomKind::Line => "line",
            GeomKind::Polygon => "polygon",
        }
    }
}

/// Classify a sampled geometry; `None` for GeometryCollection (no encoder
/// bucket — such features are excluded from the measurement).
fn kind_of(geom: &geo_types::Geometry<f64>) -> Option<GeomKind> {
    use geo_types::Geometry as G;
    match geom {
        G::Point(_) | G::MultiPoint(_) => Some(GeomKind::Point),
        G::Line(_) | G::LineString(_) | G::MultiLineString(_) => Some(GeomKind::Line),
        G::Polygon(_) | G::MultiPolygon(_) | G::Rect(_) | G::Triangle(_) => Some(GeomKind::Polygon),
        G::GeometryCollection(_) => None,
    }
}

/// Measure the real encoded + compressed size of a loader sample.
///
/// Groups the sample as ONE synthetic tile layer of the dominant geometry
/// kind (mixed-geometry samples measure the dominant kind's subset only —
/// [`MeasuredEncoding::geometry_kind`] records which), encodes it through the
/// production stt-core encoder with the given settings, and compresses with
/// zstd. Returns `Ok(None)` when the usable subset is smaller than
/// [`MIN_MEASURE_FEATURES`].
pub fn measure_sample(
    sample: &[SampledFeature],
    settings: &MeasureSettings,
) -> Result<Option<MeasuredEncoding>> {
    // Dominant kind, ties broken toward point > line > polygon (deterministic).
    let mut counts = [0usize; 3];
    for f in sample {
        if let Some(kind) = kind_of(&f.geometry) {
            counts[kind as usize] += 1;
        }
    }
    let mut dominant = GeomKind::Point;
    for kind in [GeomKind::Line, GeomKind::Polygon] {
        if counts[kind as usize] > counts[dominant as usize] {
            dominant = kind;
        }
    }

    let subset: Vec<&SampledFeature> = sample
        .iter()
        .filter(|f| kind_of(&f.geometry) == Some(dominant))
        .collect();
    if subset.len() < MIN_MEASURE_FEATURES {
        return Ok(None);
    }

    let layer = build_layer(&subset, dominant);
    let cfg = EncoderConfig {
        quantize_coords_m: settings.quantize_coords_m,
        quantize_attrs_auto: settings.quantize_attrs_auto,
        ..EncoderConfig::default()
    };
    let payload = encode_tile_with(&[layer], &cfg).context("sample tile encode failed")?;
    let compressed = compress_zstd_with_dict_level(&payload, None, settings.zstd_level)
        .context("sample tile compression failed")?;
    let per_column = attribute_columns(&payload, settings.zstd_level)?;

    Ok(Some(MeasuredEncoding {
        features: subset.len(),
        geometry_kind: dominant.name().to_string(),
        bytes_total: compressed.len(),
        bytes_per_feature: compressed.len() as f64 / subset.len() as f64,
        zstd_ratio: payload.len() as f64 / compressed.len().max(1) as f64,
        per_column,
    }))
}

fn line_coords(ls: &geo_types::LineString<f64>) -> Vec<Coord> {
    ls.0.iter().map(|c| [c.x, c.y]).collect()
}

fn polygon_rings(polygon: &geo_types::Polygon<f64>) -> Vec<Vec<Coord>> {
    std::iter::once(polygon.exterior())
        .chain(polygon.interiors().iter())
        .map(line_coords)
        .collect()
}

/// Point parts of a point-kind geometry (a MultiPoint flattens to one row per
/// point, the shape a tiler split produces).
fn point_parts(geom: &geo_types::Geometry<f64>) -> Vec<Coord> {
    use geo_types::Geometry as G;
    match geom {
        G::Point(p) => vec![[p.x(), p.y()]],
        G::MultiPoint(mp) => mp.0.iter().map(|p| [p.x(), p.y()]).collect(),
        _ => Vec::new(),
    }
}

/// Line parts of a line-kind geometry; empty parts are dropped.
fn line_parts(geom: &geo_types::Geometry<f64>) -> Vec<Vec<Coord>> {
    use geo_types::Geometry as G;
    let parts = match geom {
        G::Line(l) => vec![vec![[l.start.x, l.start.y], [l.end.x, l.end.y]]],
        G::LineString(ls) => vec![line_coords(ls)],
        G::MultiLineString(mls) => mls.0.iter().map(line_coords).collect(),
        _ => Vec::new(),
    };
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Polygon parts (ring lists) of a polygon-kind geometry; empty parts dropped.
fn polygon_parts(geom: &geo_types::Geometry<f64>) -> Vec<Vec<Vec<Coord>>> {
    use geo_types::Geometry as G;
    let parts = match geom {
        G::Polygon(p) => vec![polygon_rings(p)],
        G::MultiPolygon(mp) => mp.0.iter().map(polygon_rings).collect(),
        G::Rect(r) => vec![polygon_rings(&r.to_polygon())],
        G::Triangle(t) => vec![polygon_rings(&t.to_polygon())],
        _ => Vec::new(),
    };
    parts
        .into_iter()
        .filter(|rings| rings.iter().any(|ring| !ring.is_empty()))
        .collect()
}

/// Assemble the sampled subset into one synthetic tile layer. Multi-part
/// geometries flatten into one row per part, duplicating times and property
/// values (what a tiler's multi-geometry split produces), so their full cost
/// still lands on the one source feature.
fn build_layer(subset: &[&SampledFeature], kind: GeomKind) -> ColumnarLayer {
    // Property schema: names in first-seen order; Numeric vs Categorical from
    // the first value seen (all rows come from one Parquet schema, so mixed
    // types per name don't occur in practice — mismatches encode as null).
    let mut names: Vec<String> = Vec::new();
    let mut is_numeric: Vec<bool> = Vec::new();
    for feature in subset {
        for (name, value) in &feature.properties {
            if !names.iter().any(|n| n == name) {
                names.push(name.clone());
                is_numeric.push(matches!(value, PropValue::Number(_)));
            }
        }
    }

    let mut ids: Vec<u64> = Vec::new();
    let mut starts: Vec<i64> = Vec::new();
    let mut ends: Vec<i64> = Vec::new();
    let mut points: Vec<Coord> = Vec::new();
    let mut lines: Vec<Vec<Coord>> = Vec::new();
    let mut polygons: Vec<Vec<Vec<Coord>>> = Vec::new();
    let mut numeric_cols: Vec<Vec<Option<f64>>> = vec![Vec::new(); names.len()];
    let mut categorical_cols: Vec<Vec<Option<String>>> = vec![Vec::new(); names.len()];

    for feature in subset {
        let n_parts = match kind {
            GeomKind::Point => {
                let parts = point_parts(&feature.geometry);
                let n = parts.len();
                points.extend(parts);
                n
            }
            GeomKind::Line => {
                let parts = line_parts(&feature.geometry);
                let n = parts.len();
                lines.extend(parts);
                n
            }
            GeomKind::Polygon => {
                let parts = polygon_parts(&feature.geometry);
                let n = parts.len();
                polygons.extend(parts);
                n
            }
        };
        for _ in 0..n_parts {
            ids.push(ids.len() as u64);
            starts.push(feature.timestamp_ms as i64);
            ends.push(feature.timestamp_ms as i64);
            for (col, name) in names.iter().enumerate() {
                let value = feature
                    .properties
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, v)| v);
                if is_numeric[col] {
                    numeric_cols[col].push(match value {
                        Some(PropValue::Number(x)) => Some(*x),
                        _ => None,
                    });
                } else {
                    categorical_cols[col].push(match value {
                        Some(PropValue::Text(s)) => Some(s.clone()),
                        _ => None,
                    });
                }
            }
        }
    }

    let geometry = match kind {
        GeomKind::Point => GeometryColumn::Point(points),
        GeomKind::Line => GeometryColumn::LineString(lines),
        GeomKind::Polygon => GeometryColumn::Polygon(polygons),
    };
    let properties = names
        .into_iter()
        .enumerate()
        .map(|(col, name)| {
            let column = if is_numeric[col] {
                PropertyColumn::Numeric(std::mem::take(&mut numeric_cols[col]))
            } else {
                PropertyColumn::Categorical(std::mem::take(&mut categorical_cols[col]))
            };
            (name, column)
        })
        .collect();

    ColumnarLayer {
        name: "default".to_string(),
        feature_ids: ids,
        start_times: starts,
        end_times: ends,
        geometry,
        vertex_times: None,
        vertex_values: None,
        vertex_value_matrix: None,
        triangles: None,
        properties,
    }
}

/// Per-column compressed-cost attribution: decode the encoded tile and
/// re-encode each column alone (single-field IPC stream + zstd at the same
/// level). Shares are of the per-column sum; sorted descending by bytes with
/// name as the deterministic tiebreak.
fn attribute_columns(payload: &[u8], zstd_level: i32) -> Result<Vec<ColumnCost>> {
    let layers = decode_tile(payload).context("sample tile decode failed")?;
    let mut costs: Vec<(String, usize)> = Vec::new();
    for layer in &layers {
        let batch = &layer.batch;
        let schema = batch.schema();
        for (idx, field) in schema.fields().iter().enumerate() {
            let one = RecordBatch::try_new(
                Arc::new(Schema::new(vec![field.as_ref().clone()])),
                vec![batch.column(idx).clone()],
            )
            .context("single-column batch build failed")?;
            let mut ipc = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut ipc, &one.schema())
                    .context("column IPC writer init failed")?;
                writer.write(&one).context("column IPC write failed")?;
                writer.finish().context("column IPC finish failed")?;
            }
            let compressed = compress_zstd_with_dict_level(&ipc, None, zstd_level)
                .context("column compression failed")?;
            costs.push((field.name().clone(), compressed.len()));
        }
    }
    let total: usize = costs.iter().map(|(_, bytes)| bytes).sum();
    let mut out: Vec<ColumnCost> = costs
        .into_iter()
        .map(|(name, compressed_bytes)| ColumnCost {
            name,
            compressed_bytes,
            share: compressed_bytes as f64 / total.max(1) as f64,
        })
        .collect();
    out.sort_by(|a, b| {
        b.compressed_bytes
            .cmp(&a.compressed_bytes)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry, LineString, Point};

    /// n spread-out points with one numeric and one string property. Knuth-hash
    /// jitter keeps the f64 coordinate mantissas high-entropy so quantization
    /// has real bytes to win.
    fn point_sample(n: usize) -> Vec<SampledFeature> {
        (0..n)
            .map(|i| {
                let jitter = |salt: u64| {
                    ((i as u64).wrapping_add(salt).wrapping_mul(2_654_435_761) % 100_000) as f64
                        * 1e-7
                };
                SampledFeature {
                    geometry: Geometry::Point(Point::new(
                        -73.5 + i as f64 * 0.0013 + jitter(0),
                        45.5 + (i % 7) as f64 * 0.0021 + jitter(17),
                    )),
                    timestamp_ms: 1_600_000_000_000 + i as u64 * 1_000,
                    properties: vec![
                        (
                            "magnitude".to_string(),
                            PropValue::Number(1.0 + (i % 90) as f64 * 0.137),
                        ),
                        (
                            "region".to_string(),
                            PropValue::Text(format!("region-{}", i % 5)),
                        ),
                    ],
                }
            })
            .collect()
    }

    #[test]
    fn measures_point_sample() {
        let sample = point_sample(200);
        let measured = measure_sample(&sample, &MeasureSettings::default())
            .unwrap()
            .expect("200 features is enough to measure");
        assert_eq!(measured.features, 200);
        assert_eq!(measured.geometry_kind, "point");
        assert!(measured.bytes_total > 0);
        assert!(measured.bytes_per_feature > 0.0);
        assert!(measured.zstd_ratio > 0.0);

        let share_sum: f64 = measured.per_column.iter().map(|c| c.share).sum();
        assert!((share_sum - 1.0).abs() < 1e-9, "shares sum to {share_sum}");
        for name in ["geometry", "magnitude", "region", "id", "start_time"] {
            assert!(
                measured.per_column.iter().any(|c| c.name == name),
                "missing column {name}"
            );
        }
        // Sorted descending by compressed bytes.
        for pair in measured.per_column.windows(2) {
            assert!(pair[0].compressed_bytes >= pair[1].compressed_bytes);
        }
    }

    #[test]
    fn quantized_coords_never_larger() {
        let sample = point_sample(500);
        let base = measure_sample(&sample, &MeasureSettings::default())
            .unwrap()
            .unwrap();
        let quantized = measure_sample(
            &sample,
            &MeasureSettings {
                quantize_coords_m: Some(0.1),
                ..MeasureSettings::default()
            },
        )
        .unwrap()
        .unwrap();
        assert!(
            quantized.bytes_total <= base.bytes_total,
            "quantized {} > unquantized {}",
            quantized.bytes_total,
            base.bytes_total
        );
    }

    #[test]
    fn mixed_geometry_measures_dominant_kind_subset() {
        let mut sample = point_sample(150);
        for i in 0..50 {
            sample.push(SampledFeature {
                geometry: Geometry::LineString(LineString::from(vec![
                    (0.0, 0.0),
                    (i as f64 * 0.01, 1.0),
                ])),
                timestamp_ms: 0,
                properties: vec![],
            });
        }
        let measured = measure_sample(&sample, &MeasureSettings::default())
            .unwrap()
            .unwrap();
        assert_eq!(measured.geometry_kind, "point");
        assert_eq!(measured.features, 150);
    }

    #[test]
    fn empty_or_tiny_sample_returns_none() {
        assert!(measure_sample(&[], &MeasureSettings::default())
            .unwrap()
            .is_none());
        let tiny = point_sample(MIN_MEASURE_FEATURES - 1);
        assert!(measure_sample(&tiny, &MeasureSettings::default())
            .unwrap()
            .is_none());
    }
}
