//! Arrow column construction — the version-independent middle of the encoder.
//!
//! Turns [`ColumnarLayer`](super::layer::ColumnarLayer) columns into Arrow
//! arrays: the GeoArrow interleaved geometry array, the categorical
//! dictionary, the per-vertex time / value columns, and the vector-group
//! fusion. Frame-shape agnostic — v1 and v2 both build these identically.

use super::config::VectorGroup;
use super::layer::{GeometryColumn, PropertyColumn};
use super::quantize::QuantAffine;
use crate::error::{Error, Result};
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Builder, Float64Array, Int32Array, Int64Builder,
    ListArray, ListBuilder, UInt16Builder,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ----------------------------------------------------------------------------
// Geometry array construction (GeoArrow interleaved)
// ----------------------------------------------------------------------------

/// Build an `i32` offset buffer from per-element counts.
///
/// Errors when the accumulated count exceeds `i32::MAX`: Arrow list offsets
/// are 32-bit, so a larger layer would wrap into corrupt offsets in release
/// builds. No budget cap guarantees safety here (budgets are optional), so the
/// overflow must be a hard, descriptive failure.
pub(crate) fn offsets_from_counts(
    counts: impl Iterator<Item = usize>,
) -> Result<OffsetBuffer<i32>> {
    let mut acc = 0i32;
    let mut offsets = vec![0i32];
    for c in counts {
        acc = i32::try_from(c)
            .ok()
            .and_then(|c| acc.checked_add(c))
            .ok_or_else(|| {
                Error::Other(format!(
                    "layer geometry exceeds {} total vertices/rings, which Arrow's \
                     32-bit list offsets cannot address; split the layer",
                    i32::MAX
                ))
            })?;
        offsets.push(acc);
    }
    Ok(OffsetBuffer::new(offsets.into()))
}
/// Optionally quantizing coordinates to an `i32` grid
/// (when `quant` is `Some`). The List/FixedSizeList nesting and offset buffers
/// are identical either way — only the leaf changes from `Float64` to `Int32`,
/// so `quant = None` is byte-identical to the historical encoder.
///
/// When `point_elev` is `Some` (POINT geometry only), each point's altitude is
/// folded in as a 3rd coordinate: the leaf becomes `FixedSizeList<_,3>`
/// (`[x,y,z]`), quantized with the affine's `z0`/`sz` when quantizing. The
/// renderer then binds 3D positions zero-copy (no pad). `None` ⇒ 2D, unchanged.
pub(crate) fn build_geometry_array_q(
    geom: &GeometryColumn,
    quant: Option<&QuantAffine>,
    point_elev: Option<&[f64]>,
) -> Result<ArrayRef> {
    // 3D POINT path: interleave [x,y,z] into a FixedSizeList<_,3> leaf.
    if let (GeometryColumn::Point(points), Some(elev)) = (geom, point_elev) {
        let list_size = 3;
        let dt = if quant.is_some() {
            DataType::Int32
        } else {
            DataType::Float64
        };
        let field = Arc::new(Field::new("xyz", dt, false));
        let child: ArrayRef = match quant {
            Some(q) => {
                let mut iv = Vec::with_capacity(points.len() * 3);
                for (i, [x, y]) in points.iter().enumerate() {
                    iv.push(q.qx(*x));
                    iv.push(q.qy(*y));
                    iv.push(q.qz(elev.get(i).copied().unwrap_or(0.0))?);
                }
                Arc::new(Int32Array::from(iv))
            }
            None => {
                let mut flat = Vec::with_capacity(points.len() * 3);
                for (i, [x, y]) in points.iter().enumerate() {
                    flat.push(*x);
                    flat.push(*y);
                    flat.push(elev.get(i).copied().unwrap_or(0.0));
                }
                Arc::new(Float64Array::from(flat))
            }
        };
        return Ok(Arc::new(FixedSizeListArray::new(
            field, list_size, child, None,
        )));
    }

    let coord_field = || {
        let dt = if quant.is_some() {
            DataType::Int32
        } else {
            DataType::Float64
        };
        Arc::new(Field::new("xy", dt, false))
    };
    // Turn a flat `[x,y,x,y,...]` f64 run into the FixedSizeList<_,2> leaf,
    // quantizing to i32 grid indices when an affine is supplied.
    let make_leaf = |flat: Vec<f64>| -> ArrayRef {
        match quant {
            Some(q) => {
                let mut iv = Vec::with_capacity(flat.len());
                let mut i = 0;
                while i + 1 < flat.len() {
                    iv.push(q.qx(flat[i]));
                    iv.push(q.qy(flat[i + 1]));
                    i += 2;
                }
                Arc::new(FixedSizeListArray::new(
                    coord_field(),
                    2,
                    Arc::new(Int32Array::from(iv)),
                    None,
                ))
            }
            None => Arc::new(FixedSizeListArray::new(
                coord_field(),
                2,
                Arc::new(Float64Array::from(flat)),
                None,
            )),
        }
    };

    Ok(match geom {
        GeometryColumn::Point(points) => {
            let mut flat = Vec::with_capacity(points.len() * 2);
            for [x, y] in points {
                flat.push(*x);
                flat.push(*y);
            }
            make_leaf(flat)
        }
        GeometryColumn::LineString(lines) => {
            let mut flat: Vec<f64> = Vec::new();
            for line in lines {
                for [x, y] in line {
                    flat.push(*x);
                    flat.push(*y);
                }
            }
            let coords = make_leaf(flat);
            let offsets = offsets_from_counts(lines.iter().map(|l| l.len()))?;
            let vertex_field = Arc::new(Field::new("vertices", coords.data_type().clone(), false));
            Arc::new(ListArray::new(vertex_field, offsets, coords, None))
        }
        GeometryColumn::Polygon(polys) => {
            // Flatten all coordinates, ring sizes, and ring counts per feature.
            let mut flat: Vec<f64> = Vec::new();
            let mut ring_sizes: Vec<usize> = Vec::new();
            let mut rings_per_feature: Vec<usize> = Vec::new();
            for feature in polys {
                rings_per_feature.push(feature.len());
                for ring in feature {
                    ring_sizes.push(ring.len());
                    for [x, y] in ring {
                        flat.push(*x);
                        flat.push(*y);
                    }
                }
            }
            let coords = make_leaf(flat);
            // Ring level: List<FixedSizeList>.
            let ring_offsets = offsets_from_counts(ring_sizes.into_iter())?;
            let vertex_field = Arc::new(Field::new("vertices", coords.data_type().clone(), false));
            let rings: ArrayRef =
                Arc::new(ListArray::new(vertex_field, ring_offsets, coords, None));
            // Feature level: List<List<FixedSizeList>>.
            let feature_offsets = offsets_from_counts(rings_per_feature.into_iter())?;
            let ring_field = Arc::new(Field::new("rings", rings.data_type().clone(), false));
            Arc::new(ListArray::new(ring_field, feature_offsets, rings, None))
        }
    })
}
/// Default ceiling (ms) on the u16-delta `vertex_time` quantization step.
///
/// The u16 encoding trades precision for a 4x payload shrink, but the step is
/// derived from the layer's temporal span — without a ceiling, a wide layer
/// (e.g. a 30-day temporal-LOD bucket) silently quantizes vertex times to
/// tens of seconds. A 1 s ceiling keeps the worst-case error below anything
/// playback can show; layers needing a coarser step fall back to the exact
/// `List<Int64>` shape instead (no thinning).
pub const DEFAULT_VERTEX_TIME_MAX_STEP_MS: u32 = 1000;

/// Latch so the "u16-delta ceiling exceeded" warning fires at most once per
/// process instead of once per tile.
static VERTEX_TIME_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Fuse the scalar columns named by each configured [`VectorGroup`] into a single
/// [`PropertyColumn::Vector`], leaving every other column untouched and in order.
///
/// A group whose components are not ALL present as `Numeric` columns is skipped
/// (its scalars stay as-is) — so a tile that happens not to carry one of the
/// inputs degrades to scalars rather than dropping data. Missing per-feature
/// values (`None`) encode as `0.0`. Returns `None` when no group applied, letting
/// the caller iterate `layer.properties` directly with no clone.
pub(crate) fn group_vector_properties(
    props: &[(String, PropertyColumn)],
    n: usize,
    groups: &[VectorGroup],
) -> Option<Vec<(String, PropertyColumn)>> {
    if groups.is_empty() {
        return None;
    }
    // name → numeric values slice, for component lookup.
    let numeric: HashMap<&str, &[Option<f64>]> = props
        .iter()
        .filter_map(|(k, c)| match c {
            PropertyColumn::Numeric(v) => Some((k.as_str(), v.as_slice())),
            _ => None,
        })
        .collect();

    let mut out: Vec<(String, PropertyColumn)> = Vec::new();
    let mut consumed: HashSet<String> = HashSet::new();
    let mut any = false;
    for g in groups {
        if g.components.is_empty()
            || !g
                .components
                .iter()
                .all(|c| numeric.contains_key(c.as_str()))
        {
            continue;
        }
        let width = g.components.len();
        let mut values = vec![0f32; width * n];
        for (ci, cname) in g.components.iter().enumerate() {
            let col = numeric[cname.as_str()];
            for i in 0..n {
                values[i * width + ci] = col[i].map(|x| x as f32).unwrap_or(0.0);
            }
        }
        for c in &g.components {
            consumed.insert(c.clone());
        }
        out.push((
            g.name.clone(),
            PropertyColumn::Vector {
                width,
                elem: g.elem,
                values,
            },
        ));
        any = true;
    }
    if !any {
        return None;
    }
    // Keep every column not pulled into a group, in original order.
    for (k, c) in props {
        if !consumed.contains(k) {
            out.push((k.clone(), c.clone()));
        }
    }
    Some(out)
}
/// Build a (key_array, value_array) pair for a Dictionary<UInt16, Utf8>
/// column. Null inputs become null keys (the corresponding string is not
/// inserted into the dictionary); strings are deduplicated in first-seen
/// order so the on-disk Arrow dictionary is stable across runs.
///
/// Errors if a column has more than `u16::MAX` distinct values, which the
/// `UInt16` key space cannot address. In practice STT categorical columns top
/// out in the low hundreds, so this only fires on pathological input (e.g. a
/// per-feature unique id mistaken for a category). Erroring is deliberate: the
/// previous behaviour silently collapsed every overflowing value to a single
/// index, mislabeling features without a trace. A producer that genuinely needs
/// >65k distinct strings should split the column or widen the key type.
pub(crate) fn build_dictionary_indices(
    values: &[Option<String>],
) -> Result<(Vec<Option<u16>>, Vec<String>)> {
    let mut categories: Vec<String> = Vec::new();
    let mut lookup: HashMap<String, u16> = HashMap::new();
    let mut indices: Vec<Option<u16>> = Vec::with_capacity(values.len());
    for v in values {
        match v {
            Some(s) => {
                if let Some(&idx) = lookup.get(s) {
                    indices.push(Some(idx));
                } else if categories.len() < u16::MAX as usize {
                    let idx = categories.len() as u16;
                    categories.push(s.clone());
                    lookup.insert(s.clone(), idx);
                    indices.push(Some(idx));
                } else {
                    return Err(Error::Other(format!(
                        "categorical column has more than {} distinct values, which a \
                         Dictionary<UInt16, Utf8> key cannot address; split the column \
                         into multiple categorical fields or widen the key type",
                        u16::MAX
                    )));
                }
            }
            None => indices.push(None),
        }
    }
    Ok((indices, categories))
}
/// Built per-vertex time column, alongside the per-layer schema metadata
/// that lets the reader reconstruct absolute timestamps.
pub(crate) struct VertexTimeColumn {
    pub(crate) array: ArrayRef,
    /// `(origin_ms, step_ms)` when the column is u16-delta-encoded. `None`
    /// when the column kept its absolute `List<Int64>` shape (the exact
    /// fallback path, used for layers whose temporal span would need a step
    /// beyond the configured ceiling — see [`DEFAULT_VERTEX_TIME_MAX_STEP_MS`]).
    pub(crate) encoding: Option<(i64, u32)>,
}

/// Build the optional per-vertex time column.
///
/// v3 attempts to encode timestamps as `List<UInt16>` deltas relative to
/// a per-layer origin and step (`absolute = origin + delta * step`), with
/// the step bounded by `max_step_ms` (see
/// [`DEFAULT_VERTEX_TIME_MAX_STEP_MS`]). When the layer's temporal span
/// would need a coarser step than that, the column keeps the exact v2
/// `List<Int64>` shape instead — bounded quantization or no quantization,
/// never a silent precision cliff.
pub(crate) fn build_vertex_time_array(
    vertex_times: &Option<Vec<Vec<i64>>>,
    feature_count: usize,
    max_step_ms: u32,
) -> Option<VertexTimeColumn> {
    let vt = vertex_times.as_ref()?;

    // Discover the layer's temporal span across every (feature, vertex) pair.
    // Empty / null lists are skipped — a list-of-nulls is fine, it just won't
    // shrink the span.
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut any = false;
    for times in vt.iter().take(feature_count) {
        for &t in times {
            if t < min {
                min = t;
            }
            if t > max {
                max = t;
            }
            any = true;
        }
    }

    if any && max >= min {
        // Pick the smallest step (in ms) that keeps every (t - min) inside
        // u16::MAX. step=1 means "exact ms granularity"; larger steps trade
        // precision (bounded by step ms) for a 4x payload shrink vs i64.
        let span = (max - min) as u64;
        // Computed in u64 and compared BEFORE narrowing: a pathological span
        // whose step overflows u32 must hit the i64 path, not wrap small.
        let step = if span <= u16::MAX as u64 {
            1u64
        } else {
            ((span + u16::MAX as u64 - 1) / u16::MAX as u64).max(1)
        };
        // Step ceiling: beyond it the quantization error would exceed the
        // configured precision, so take the exact i64 path below instead.
        if step <= max_step_ms as u64 {
            let step = step as u32;
            let mut builder = ListBuilder::new(UInt16Builder::new());
            for i in 0..feature_count {
                match vt.get(i) {
                    Some(times) if !times.is_empty() => {
                        for &t in times {
                            // Saturate at u16::MAX — for a sensibly chosen
                            // step this branch can only fire on inputs that
                            // disagree with the (min,max) scan above (e.g.
                            // a `vertex_times` longer than feature_count).
                            let delta =
                                ((t - min) as u64 / step as u64).min(u16::MAX as u64) as u16;
                            builder.values().append_value(delta);
                        }
                        builder.append(true);
                    }
                    _ => builder.append(false),
                }
            }
            return Some(VertexTimeColumn {
                array: Arc::new(builder.finish()),
                encoding: Some((min, step)),
            });
        }
        // Reached the exact-Int64 fallback because the span needs a coarser
        // step than the ceiling. Warn once per process (not per tile).
        if !VERTEX_TIME_FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "vertex-time span {}ms exceeds u16-delta ceiling (step {}ms > max {}ms); \
                 falling back to exact Int64 — payload keeps full precision but is ~4x larger",
                span,
                step,
                max_step_ms
            );
        }
    }

    // Fallback: legacy absolute List<Int64> for empty-ish columns or
    // pathological steps. Identical wire shape to v2.
    let mut builder = ListBuilder::new(Int64Builder::new());
    for i in 0..feature_count {
        match vt.get(i) {
            Some(times) if !times.is_empty() => {
                for &t in times {
                    builder.values().append_value(t);
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Some(VertexTimeColumn {
        array: Arc::new(builder.finish()),
        encoding: None,
    })
}

/// Build the optional per-vertex scalar column as a nullable `List<Float32>`.
///
/// Unlike `vertex_time`, there's no delta/origin/step encoding — the values are
/// producer-defined scalars (e.g. sea-surface temperature) with a small range
/// where f32 precision is ample. A feature with no per-vertex values appends a
/// null list; `NaN` entries within a list mark individual vertices with no value.
/// Returns `None` when no feature carries any value, so the column is omitted.
pub(crate) fn build_vertex_value_array(
    vertex_values: &Option<Vec<Vec<f32>>>,
    feature_count: usize,
) -> Option<ArrayRef> {
    let vv = vertex_values.as_ref()?;
    let any = vv.iter().take(feature_count).any(|v| !v.is_empty());
    if !any {
        return None;
    }
    let mut builder = ListBuilder::new(Float32Builder::new());
    for i in 0..feature_count {
        match vv.get(i) {
            Some(values) if !values.is_empty() => {
                for &v in values {
                    builder.values().append_value(v);
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Some(Arc::new(builder.finish()))
}

/// Recover the per-vertex value-matrix bucket count: for the first LineString
/// feature carrying both vertices and matrix data, `num_buckets = matrix_len /
/// vertex_count` (the matrix is vertex-major, `num_vertices * num_buckets`
/// long). Returns `None` for non-line geometry or when no feature carries a
/// clean multiple — the matrix column is then present but non-animatable.
pub(crate) fn infer_vertex_value_buckets(
    matrix: &[Vec<f32>],
    geometry: &GeometryColumn,
) -> Option<u32> {
    let lines = match geometry {
        GeometryColumn::LineString(lines) => lines,
        _ => return None,
    };
    for (i, m) in matrix.iter().enumerate() {
        let nv = lines.get(i)?.len();
        if !m.is_empty() && nv > 0 && m.len() % nv == 0 {
            return Some((m.len() / nv) as u32);
        }
    }
    None
}
