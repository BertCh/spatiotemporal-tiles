//! Fixed-point quantization — coordinates and numeric attributes.
//!
//! Two independent, opt-in size levers, each shipping an affine in field
//! metadata so the reader reconstructs `Float64`: [`QuantAffine`] (geometry,
//! on the world-anchored grid built by [`world_grid_affine`]) and
//! [`AttrQuant`] (numeric property columns).

use crate::error::{Error, Result};
use arrow::array::{ArrayRef, Int32Builder, UInt16Builder};
use std::sync::Arc;

/// Meters per degree of latitude (WGS84 mean) — the constant the coordinate
/// quantizer sizes its grid from. Longitude scales by `cos(lat)`.
pub(crate) const M_PER_DEG_LAT: f64 = 111_320.0;

/// Schema-metadata key (on the `geometry` field) that flags a tile whose
/// coordinates are fixed-point `i32` grid indices rather than GeoArrow Float64
/// lon/lat. Its value is the [`QuantAffine`] JSON; absent ⇒ standard Float64.
pub const STT_QUANT_META_KEY: &str = "stt:quant";
/// Per-layer coordinate-quantization affine. Coordinates ship as `i32` grid
/// indices; the decoder reconstructs `lon = x0 + qx*sx`, `lat = y0 + qy*sy`.
/// `sx`/`sy` (degrees per quantum) are sized from a target ground precision in
/// meters at the layer's mid-latitude, so the worst-case error is ≤ half a
/// quantum (~`meters/2`). This trades GeoArrow self-describing Float64 for size
/// (coords are the dominant, near-incompressible column) and is opt-in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantAffine {
    pub x0: f64,
    pub y0: f64,
    pub sx: f64,
    pub sy: f64,
    /// Z-axis origin/step (metres), present ONLY for 3D point geometry
    /// (`FixedSizeList<i32,3>`). `None` ⇒ plain 2D coords, byte-identical to the
    /// historical affine (the `z0`/`sz` keys are simply omitted from the JSON).
    pub z0: Option<f64>,
    pub sz: Option<f64>,
}

impl QuantAffine {
    pub(crate) fn to_json(&self) -> String {
        // Full f64 round-trip precision (17 sig digits) so decode is exact. The
        // z keys are emitted only for 3D affines, so a 2D affine is byte-identical.
        match (self.z0, self.sz) {
            (Some(z0), Some(sz)) => format!(
                r#"{{"x0":{:.17e},"y0":{:.17e},"sx":{:.17e},"sy":{:.17e},"z0":{:.17e},"sz":{:.17e}}}"#,
                self.x0, self.y0, self.sx, self.sy, z0, sz
            ),
            _ => format!(
                r#"{{"x0":{:.17e},"y0":{:.17e},"sx":{:.17e},"sy":{:.17e}}}"#,
                self.x0, self.y0, self.sx, self.sy
            ),
        }
    }

    /// Parse the affine from its [`STT_QUANT_META_KEY`] JSON value. The TS
    /// reader applies the identical reconstruction (`tile.ts`).
    pub fn from_json(s: &str) -> Option<QuantAffine> {
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64());
        Some(QuantAffine {
            x0: f("x0")?,
            y0: f("y0")?,
            sx: f("sx")?,
            sy: f("sy")?,
            z0: f("z0"),
            sz: f("sz"),
        })
    }

    /// Reconstruct longitude from a quantized x grid index.
    #[inline]
    pub fn lon(&self, qx: i32) -> f64 {
        self.x0 + qx as f64 * self.sx
    }
    /// Reconstruct latitude from a quantized y grid index.
    #[inline]
    pub fn lat(&self, qy: i32) -> f64 {
        self.y0 + qy as f64 * self.sy
    }

    #[inline]
    pub(crate) fn qx(&self, lon: f64) -> i32 {
        (((lon - self.x0) / self.sx).round() as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }
    #[inline]
    pub(crate) fn qy(&self, lat: f64) -> i32 {
        (((lat - self.y0) / self.sy).round() as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }
    /// Quantize a metre altitude to a grid index (3D affines only; `z0`/`sz` set).
    ///
    /// Altitude is unbounded input (unlike lon/lat), so an index outside i32 is
    /// a hard error identifying the offending value — the old silent clamp
    /// snapped such points to ±i32::MAX quanta without a trace.
    #[inline]
    pub(crate) fn qz(&self, z: f64) -> Result<i32> {
        let z0 = self.z0.unwrap_or(0.0);
        let sz = self.sz.unwrap_or(1.0);
        let q = ((z - z0) / sz).round();
        if !(q >= i32::MIN as f64 && q <= i32::MAX as f64) {
            return Err(Error::Other(format!(
                "altitude {z} does not fit the quantization grid (origin {z0}, step {sz}): \
                 index {q} is outside i32; use a coarser --quantize-coords precision or drop \
                 the point-elevation fold for this dataset"
            )));
        }
        Ok(q as i32)
    }
}
/// Finest coordinate-quantization precision (meters) the world-anchored grid
/// supports: the grid is anchored at `(-180, -90)` with a uniform step of
/// `meters / M_PER_DEG_LAT` degrees, so the largest longitude index is
/// `360 * M_PER_DEG_LAT / meters`. Below this floor (≈ 0.0187 m, ~19 mm) that
/// index exceeds `i32::MAX` and quantization would silently snap far-east
/// coordinates to wrong locations — so finer precisions are rejected.
pub const MIN_QUANTIZE_COORDS_M: f64 = 360.0 * M_PER_DEG_LAT / (i32::MAX as f64);

/// Validate a coordinate-quantization precision against the world grid's
/// [`MIN_QUANTIZE_COORDS_M`] floor. `meters <= 0` (quantization off) passes.
/// Public so config-building paths
/// ([`EncoderConfig`](crate::arrow_tile::EncoderConfig) consumers like
/// `stt-serve`) can fail fast at startup with the same error the global
/// setter and the encode-time guard produce.
pub fn validate_quantize_coords_m(meters: f64) -> Result<()> {
    if meters > 0.0 && meters < MIN_QUANTIZE_COORDS_M {
        return Err(Error::Other(format!(
            "coordinate quantization precision {meters} m is finer than the minimum \
             {MIN_QUANTIZE_COORDS_M} m (~19 mm): the world-anchored grid's ±180° longitude \
             index would overflow i32 and snap points to wrong locations"
        )));
    }
    Ok(())
}
/// The fixed, dataset-independent quantization grid for a target ground
/// precision in meters, or `None` when off (`meters <= 0`).
///
/// The origin is the world corner `(-180, -90)` and the step is uniform in
/// degrees (`meters / M_PER_DEG_LAT`) — deliberately **not** a per-tile
/// bbox-relative grid. A per-tile grid gives the same coordinate different
/// indices in different tiles, which destroys the packed format's
/// content-addressed blob dedup (measured: +61% on a dedup-heavy dataset). A
/// single world grid keeps identical geometry byte-identical across tiles.
///
/// Latitude precision is exactly `meters`; longitude is `meters * cos(lat)` —
/// i.e. always ≤ `meters` (finer toward the poles), never coarser than asked.
/// At 1 m the largest index is `360 * M_PER_DEG_LAT ≈ 4.0e7`, well within i32.
pub(crate) fn world_grid_affine(meters: f64) -> Option<QuantAffine> {
    if !(meters > 0.0) {
        return None;
    }
    let step = meters / M_PER_DEG_LAT;
    Some(QuantAffine {
        x0: -180.0,
        y0: -90.0,
        sx: step,
        sy: step,
        z0: None,
        sz: None,
    })
}

/// The 3D variant of [`world_grid_affine`]: same world-grid xy plus a Z axis
/// quantized to the SAME ground precision in metres, origin pinned to a fixed
/// global datum (`z0 = 0`) so identical surfels stay byte-identical across tiles
/// (dedup-preserving, like the xy world grid). For point clouds whose altitude
/// rides the geometry's 3rd coordinate (`--point-elevation-column`).
pub(crate) fn world_grid_affine_3d(meters: f64) -> Option<QuantAffine> {
    let a = world_grid_affine(meters)?;
    Some(QuantAffine {
        z0: Some(0.0),
        sz: Some(meters),
        ..a
    })
}
/// Field-metadata key flagging a *numeric property* column that ships as
/// fixed-point integer indices (smallest of `UInt16`/`Int32`) instead of
/// `Float64`. Its value is the [`AttrQuant`] JSON (`value = o + q*s`). Sibling
/// of [`STT_QUANT_META_KEY`] for geometry; lives on the property field, so a
/// reader reconstructs Float64 the same way it does coordinates.
pub const STT_QUANT_ATTR_META_KEY: &str = "stt:qa";

/// Per-numeric-property quantization affine: `value = o + q * s`, where `o` is
/// the column minimum (the dequantization offset) and `s` the requested ground
/// precision (the step). Reconstruction is lossy to ≤ `s/2`; the reader applies
/// the identical math (`tile.ts`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttrQuant {
    pub o: f64,
    pub s: f64,
}

impl AttrQuant {
    pub(crate) fn to_json(&self) -> String {
        // Full f64 round-trip precision so the offset/step decode exactly.
        format!(r#"{{"o":{:.17e},"s":{:.17e}}}"#, self.o, self.s)
    }

    /// Parse the affine from its [`STT_QUANT_ATTR_META_KEY`] JSON value.
    pub fn from_json(s: &str) -> Option<AttrQuant> {
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(AttrQuant {
            o: v.get("o")?.as_f64()?,
            s: v.get("s")?.as_f64()?,
        })
    }

    /// Reconstruct the original value from a quantized index.
    #[inline]
    pub fn value(&self, q: i64) -> f64 {
        self.o + q as f64 * self.s
    }
}
/// Range-adaptive auto quantization: size the step from the column's own span so
/// `[min, max]` maps onto `[0, 65535]`. ALWAYS returns a `UInt16` column (never
/// `None`, never `Int32`) so a column is quantized to the *same type in every
/// tile* — a per-tile range-adaptive choice would otherwise leave constant /
/// all-null tiles as `Float64` and drift the layer schema across tiles. A
/// constant column quantizes to all-zeros and an all-null column to all-nulls;
/// both compress to nothing, so the uniform `UInt16` costs nothing and keeps the
/// schema consistent.
pub(crate) fn build_quantized_numeric_auto(values: &[Option<f64>]) -> Option<(ArrayRef, String)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in values.iter().flatten() {
        if v.is_finite() {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    // o = offset (column min, or 0 when no finite value); s = step. A zero span
    // (constant / no-data) uses step 1 → every present value maps to index 0,
    // which reconstructs to `o` exactly.
    let (o, s) = if min.is_finite() {
        if max > min {
            (min, (max - min) / u16::MAX as f64)
        } else {
            (min, 1.0)
        }
    } else {
        (0.0, 1.0)
    };
    let affine = AttrQuant { o, s };
    let mut b = UInt16Builder::with_capacity(values.len());
    for v in values {
        match v {
            Some(x) if x.is_finite() => {
                let q = (((*x - o) / s).round()).clamp(0.0, u16::MAX as f64) as u16;
                b.append_value(q);
            }
            _ => b.append_null(),
        }
    }
    Some((Arc::new(b.finish()), affine.to_json()))
}

/// Quantize a numeric property column to the smallest integer leaf at `prec`
/// units, with the offset pinned to the column minimum. Returns
/// `(array, affine_json)` — a `UInt16` leaf when the quantized range fits 16
/// bits, else `Int32` — or `None` when no finite value exists (caller keeps the
/// `Float64` column). Nulls and non-finite values become Arrow nulls. The
/// offset is the per-column minimum: identical columns quantize identically, so
/// the packed format's content-addressed dedup is preserved.
///
/// Errors when a value's quantized index exceeds `i32::MAX` — the widest leaf
/// this encoding ships. Erroring is deliberate (mirrors the dictionary-overflow
/// error in [`build_dictionary_indices`]): the previous behaviour silently
/// clamped every overflowing value to `i32::MAX`, mislabeling features without
/// a trace. A producer whose column genuinely spans more than `i32::MAX`
/// quanta should pick a coarser precision or leave the column `Float64`.
pub(crate) fn build_quantized_numeric(
    values: &[Option<f64>],
    prec: f64,
) -> Result<Option<(ArrayRef, String)>> {
    if !(prec > 0.0) {
        return Ok(None);
    }
    let mut min = f64::INFINITY;
    for v in values.iter().flatten() {
        if v.is_finite() && *v < min {
            min = *v;
        }
    }
    if !min.is_finite() {
        return Ok(None); // no finite values — keep Float64
    }
    let affine = AttrQuant { o: min, s: prec };
    let mut q: Vec<Option<i64>> = Vec::with_capacity(values.len());
    let mut max_q: i64 = 0;
    for v in values {
        match v {
            Some(x) if x.is_finite() => {
                let qi = (((*x - affine.o) / affine.s).round() as i64).max(0);
                if qi > i32::MAX as i64 {
                    return Err(Error::Other(format!(
                        "numeric property quantization overflows: value {x} at precision \
                         {prec} quantizes to index {qi} (offset {min}), beyond the Int32 \
                         leaf's {} ceiling; use a coarser --quantize-attr precision or \
                         leave the column Float64",
                        i32::MAX
                    )));
                }
                if qi > max_q {
                    max_q = qi;
                }
                q.push(Some(qi));
            }
            _ => q.push(None),
        }
    }
    let array: ArrayRef = if max_q <= u16::MAX as i64 {
        let mut b = UInt16Builder::with_capacity(q.len());
        for qi in &q {
            match qi {
                Some(v) => b.append_value(*v as u16),
                None => b.append_null(),
            }
        }
        Arc::new(b.finish())
    } else {
        let mut b = Int32Builder::with_capacity(q.len());
        for qi in &q {
            match qi {
                Some(v) => b.append_value(*v as i32),
                None => b.append_null(),
            }
        }
        Arc::new(b.finish())
    };
    Ok(Some((array, affine.to_json())))
}
