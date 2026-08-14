//! Fixed-point quantization — coordinates and numeric attributes.
//!
//! Two independent, opt-in size levers, each shipping an affine in field
//! metadata so the reader reconstructs `Float64`: [`QuantAffine`] (geometry,
//! on the world-anchored grid built by [`world_grid_affine`]) and
//! [`AttrQuant`] (numeric property columns).

use super::config::{AttrPinned, EncoderConfig, PinnedLeaf};
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
    /// a hard error identifying the offending value. Clamping instead would
    /// snap such points to ±i32::MAX quanta without a trace.
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
/// Reserved `UInt16` index in a quantized per-vertex value column meaning
/// "this vertex has no value".
///
/// The `Float32` shape marks a valueless vertex with `NaN` (renderers map it to
/// a fallback colour); `UInt16` has no NaN, so exactly one index is spent on it
/// and the finite range is mapped onto `[0, VERTEX_VALUE_QUANT_MAX]` instead of
/// the full 16-bit space. Both reference decoders turn this index back into
/// `NaN`, so the no-value semantic survives the round trip unchanged.
pub const VERTEX_VALUE_QUANT_SENTINEL: u16 = 0xFFFF;

/// Largest index a FINITE per-vertex value may quantize to — one below
/// [`VERTEX_VALUE_QUANT_SENTINEL`], i.e. 65 535 distinct levels
/// (`0..=65534`) spanning the column's `[min, max]`.
pub const VERTEX_VALUE_QUANT_MAX: u16 = 0xFFFE;

/// The range-adaptive affine for one per-vertex value column
/// (`value = o + q * s`), given the column's finite `[min, max]` and how many
/// finite values it holds.
///
/// The two degenerate cases are handled exactly the way
/// [`build_quantized_numeric_auto`] handles them for scalar properties:
///
/// - **no finite value at all** (`finite == 0`) → `{o: 0, s: 1}`. Every entry
///   is the sentinel, so the affine is never applied; pinning it to a constant
///   keeps the `TILE_META.vq` bytes (and therefore the tile) reproducible
///   instead of serialising `±inf`.
/// - **constant column** (`span <= 0`, or a span so small that `span/65534`
///   underflows to zero) → `{o: min, s: 1}`. Every finite value maps to index
///   0, which reconstructs to `o` EXACTLY — no error at all, and no division
///   by zero.
pub(crate) fn vertex_value_affine(min: f64, max: f64, finite: usize) -> AttrQuant {
    if finite == 0 {
        return AttrQuant { o: 0.0, s: 1.0 };
    }
    let span = max - min;
    let s = span / VERTEX_VALUE_QUANT_MAX as f64;
    if !(span > 0.0) || !(s > 0.0) {
        return AttrQuant { o: min, s: 1.0 };
    }
    AttrQuant { o: min, s }
}

/// Quantize one finite value to its `UInt16` index under `affine`, clamped into
/// `[0, VERTEX_VALUE_QUANT_MAX]` so a value outside the scanned range can never
/// collide with [`VERTEX_VALUE_QUANT_SENTINEL`].
#[inline]
pub(crate) fn quantize_vertex_value(affine: &AttrQuant, v: f64) -> u16 {
    ((v - affine.o) / affine.s)
        .round()
        .clamp(0.0, VERTEX_VALUE_QUANT_MAX as f64) as u16
}
/// Largest magnitude at which `f64` still represents *every* integer exactly
/// (`2^53`). Beyond it consecutive `f64`s are ≥ 2 apart, so the affine's
/// `o + q * 1.0` no longer evaluates to `o + q` — the reconstruction silently
/// rounds to a neighbouring representable value. This is the hard boundary of
/// "integer quantization is exact" (IEEE-754 binary64's 53-bit significand),
/// not a tuning knob.
///
/// It bounds the exactness claim of the step-1 integer path; the refusal
/// threshold [`AUTO_QUANT_MAX_ABS`] is lower still.
pub(crate) const F64_EXACT_INT_LIMIT: f64 = 9_007_199_254_740_992.0;

/// Magnitude at or above which [`build_quantized_numeric_auto`] declines to
/// quantize at all, leaving the column `Float64`. This is the SOLE reason it
/// ever returns `None`, which is what keeps the decision stable across tiles of
/// one column: magnitude is a property of the column's *domain*, whereas span
/// and outlier-conditioning are properties of whichever rows a tile happened to
/// catch. A per-tile refusal would flip the column's Arrow type between tiles,
/// which `stt-validate` rates as structural schema drift and hard-fails.
///
/// `i32::MAX` is the threshold because it is where BOTH failure modes start:
///
/// * **Exactness.** The step-1 integer path indexes `value - min` into the
///   widest leaf this encoding ships (`Int32`). A column whose values reach
///   `i32::MAX` cannot be held exactly at step 1 in the general case, so it
///   would fall through to the lossy range-adaptive branch — and for an
///   identifier domain (OSM node ids ≈ 1.2e10, epoch-microsecond keys ≈ 1.7e15,
///   snowflake-style ids) lossy is silent corruption: every distinct value
///   matters and interpolating between them is meaningless. Measured on shipped
///   data: `nyc-taxi-points.trip_id` shipped at `o = 2.35e18, s = 2.3e14`, so
///   every reconstructed id was wrong by up to ±1.15e14 and the same trip
///   decoded differently in different tiles.
/// * **Relative precision.** A 16-bit range-adaptive leaf spreads 65 535 levels
///   across the span. At this magnitude the step is ≥ ~32 000 even in the best
///   case (`min ≈ 0`), i.e. under five significant digits of the LEADING value
///   and far worse for the body of the column — below any sane threshold for a
///   number a map displays.
///
/// The cost is a narrow-span, large-magnitude column (say epoch-ms timestamps
/// inside one bucket) staying `Float64` where step-1 quantization would have
/// been exact. That is 8 bytes instead of 2-4 on a column `f64` already
/// represents exactly, which is the right side of the trade against either
/// corrupting identifiers or reintroducing per-tile type drift.
///
/// NOT covered, deliberately: a *small*-magnitude column whose span is inflated
/// by a rare outlier (body 0-10 with an occasional 1e6 → step ≈ 15, so the body
/// collapses onto index 0). Detecting that needs a distribution test, which is
/// irreducibly a property of the tile's sample and therefore reintroduces the
/// drift. Coarseness there is the advertised cost of an opt-in lossy flag. The
/// correct fix is a dataset-wide range pre-pass that makes the affine — and this
/// decision — global rather than per tile; see the deferred-work register in
/// `docs/roadmap/stt-packed-format-decisions.md`.
pub(crate) const AUTO_QUANT_MAX_ABS: f64 = i32::MAX as f64;

/// The integer leaf a quantized property column ships in. `UInt16` is the
/// 2-byte default; `Int32` is the widening the exact-integer path needs when a
/// column's span exceeds 16 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantLeaf {
    U16,
    I32,
}

impl QuantLeaf {
    /// Largest index the leaf can hold (indices are always ≥ 0: the offset is
    /// the column minimum).
    #[inline]
    fn max_index(self) -> f64 {
        match self {
            QuantLeaf::U16 => u16::MAX as f64,
            QuantLeaf::I32 => i32::MAX as f64,
        }
    }

    /// The encoder-side leaf a dataset-global [`PinnedLeaf`] verdict selects.
    ///
    /// The two enums are deliberately separate — [`PinnedLeaf`] is a
    /// wire-visible verdict that survives into the manifest's `encoderPins`
    /// sidecar and back, `QuantLeaf` is this module's encode-time detail — and
    /// this exhaustive match is the seam that keeps them in lockstep. It is
    /// also the executable form of the standing boundary: the menu is EXACTLY
    /// two rates, and widening it (per-column rate selection, which only
    /// becomes representable once the affine is a function of the dataset
    /// rather than of a tile's sample) is deferred work with its own unpriced
    /// schema-template cost. Add a third variant to either enum and this stops
    /// compiling.
    #[inline]
    fn from_pinned(leaf: PinnedLeaf) -> Self {
        match leaf {
            PinnedLeaf::U16 => QuantLeaf::U16,
            PinnedLeaf::I32 => QuantLeaf::I32,
        }
    }
}

/// One pass of the statistics every auto-quantization decision needs.
struct NumericColumnStats {
    /// Smallest finite value, or `+inf` when the column has none.
    min: f64,
    /// Largest finite value, or `-inf` when the column has none.
    max: f64,
    /// Largest `|value|` over the finite values (`0.0` when there are none).
    max_abs: f64,
    /// How many cells hold a finite value (nulls / NaN / ±inf excluded).
    finite: usize,
    /// Every finite value is integer-valued (`v == v.trunc()`). Vacuously true
    /// for a column with no finite values — callers handle that case first.
    all_integer: bool,
}

fn numeric_column_stats(values: &[Option<f64>]) -> NumericColumnStats {
    let mut st = NumericColumnStats {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        max_abs: 0.0,
        finite: 0,
        all_integer: true,
    };
    for v in values.iter().flatten() {
        if !v.is_finite() {
            continue;
        }
        st.finite += 1;
        st.min = st.min.min(*v);
        st.max = st.max.max(*v);
        st.max_abs = st.max_abs.max(v.abs());
        st.all_integer &= *v == v.trunc();
    }
    st
}

/// Build the fixed-point column for `affine` in `leaf`, plus the affine JSON.
/// Nulls and non-finite cells become Arrow nulls; indices are clamped into the
/// leaf's range (a no-op for every affine this module derives — the offset is
/// the column minimum and the leaf is chosen to hold the span).
fn quantize_with(values: &[Option<f64>], affine: AttrQuant, leaf: QuantLeaf) -> (ArrayRef, String) {
    let hi = leaf.max_index();
    let index = |x: f64| ((x - affine.o) / affine.s).round().clamp(0.0, hi);
    let array: ArrayRef = match leaf {
        QuantLeaf::U16 => {
            let mut b = UInt16Builder::with_capacity(values.len());
            for v in values {
                match v {
                    Some(x) if x.is_finite() => b.append_value(index(*x) as u16),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        QuantLeaf::I32 => {
            let mut b = Int32Builder::with_capacity(values.len());
            for v in values {
                match v {
                    Some(x) if x.is_finite() => b.append_value(index(*x) as i32),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
    };
    (array, affine.to_json())
}

/// Automatic (no explicit precision) numeric-property quantization: one
/// refusal, two encoding regimes, one degenerate case.
///
/// 1. **Magnitude refusal — the ONLY `None`.** Some finite `|value|` reaches
///    [`AUTO_QUANT_MAX_ABS`] (`i32::MAX`), past which neither encoding below is
///    defensible: the step-1 integer path can no longer index the column into
///    even the `Int32` leaf, and the range-adaptive path's step is ≥ ~32 000, so
///    an identifier domain would be silently corrupted and a measurement domain
///    would keep under five significant digits. → `None`, the column stays
///    `Float64`, which holds every integer up to `2^53` exactly. This is what
///    saves identifier columns: 64-bit hashes (measured on `nyc-taxi-points`:
///    `trip_id` shipped at `o = 2.35e18`, `s = 2.3e14`, so every id was off by
///    up to ±1.15e14 and the same id decoded differently in different tiles) and
///    also mid-magnitude ones like OSM node ids (≈1.2e10), which sit well below
///    `2^53` and would otherwise decode off by tens of thousands.
/// 2. **Integer-valued column** (every finite value satisfies `v == v.trunc()`)
///    → **exact** fixed point: `s = 1.0`, `o = column min`, `UInt16` when the
///    span fits 16 bits, else `Int32`. Counts, bucket indices and integer sums
///    round-trip bit-for-bit AND get smaller than `Float64`. Range-adapting an
///    integer column instead — mapping `[min, max]` onto `[0, 65535]` — yields
///    a FRACTIONAL step, so a `count` of 7 comes back as a nearby fraction
///    (measured on `nyc-taxi-od-summary`).
/// 3. **Everything else** — a non-integer column, or an integer column whose
///    span is wider than the `Int32` leaf can index at step 1 — → the
///    range-adaptive `UInt16`: `o = min`, `s = (max - min) / 65535`, error
///    ≤ `s/2`. Lossy by construction; that is the deal this opt-in size lever
///    has always offered, and it is the encoding such columns already got.
/// 4. **No finite value at all** → `UInt16` all-nulls at `{o: 0, s: 1}`
///    (unchanged): there is no data to reason about and the leaf costs nothing.
///
/// # Schema stability — what is actually guaranteed
///
/// The v2 PROPS schema templates (and `stt-validate`'s `classify_column_drift`)
/// treat a column that is `Float64` in one tile and an integer leaf in another
/// as **structural** drift, i.e. a hard validation failure. So the load-bearing
/// property is not "always `UInt16`" but *"quantize-or-not must not depend on
/// which rows landed in this tile"*.
///
/// It is honest to claim that only because the sole refusal is the MAGNITUDE
/// test, which is a property of what the column holds, not of how many rows a
/// tile got: an id column is huge in every tile, including a tile holding a
/// single id (span 0, which a span-based rule would happily quantize). Nothing
/// else can return `None` — in particular a tile that happens to catch an
/// outlier now range-adapts (lossily) rather than dropping back to `Float64`.
///
/// The caveat this does NOT cover: a column that *mixes* magnitudes across
/// [`AUTO_QUANT_MAX_ABS`] — small values in one tile, values at or past
/// `i32::MAX` in another — can still flip. No real property column looks like
/// that (identifier domains are uniformly huge, measurement domains uniformly
/// are not), and there is no per-tile encoder that can do better without a
/// dataset-wide pre-pass.
///
/// The residual, benign variation is the `UInt16`/`Int32` WIDTH of the exact
/// integer path, when a column's per-tile span straddles 65535 (a dense tile
/// widens, a sparse one does not). `classify_column_drift` rates that
/// `AdaptiveWidth`, not `Structural`; it costs at most one extra PROPS template
/// per width combination, and the explicit-precision path has always had it.
pub(crate) fn build_quantized_numeric_auto(values: &[Option<f64>]) -> Option<(ArrayRef, String)> {
    let st = numeric_column_stats(values);

    // (4) Nothing finite: keep the historical all-null UInt16 at {o: 0, s: 1}.
    if st.finite == 0 {
        return Some(quantize_with(
            values,
            AttrQuant { o: 0.0, s: 1.0 },
            QuantLeaf::U16,
        ));
    }

    // (1) The one refusal, tested BEFORE the integer/fraction split so that it
    // cannot itself be dodged by a tile that happens to hold one fractional
    // value alongside the huge ones (`all_integer` is a per-tile observation;
    // `max_abs` crossing the threshold is not).
    if st.max_abs >= AUTO_QUANT_MAX_ABS {
        return None;
    }

    let span = st.max - st.min;

    // (2) Exact fixed point at step 1, in the narrowest leaf that indexes the
    // span. A span past the Int32 leaf falls through to the lossy range-adaptive
    // encoding below rather than refusing: refusing would flip the column's
    // Arrow type on the tiles that happen to be wide.
    if st.all_integer && span <= QuantLeaf::I32.max_index() {
        let affine = AttrQuant { o: st.min, s: 1.0 };
        let leaf = if span <= QuantLeaf::U16.max_index() {
            QuantLeaf::U16
        } else {
            QuantLeaf::I32
        };
        return Some(quantize_with(values, affine, leaf));
    }

    // A constant (or single-valued) column: step 1 maps every present value to
    // index 0, which reconstructs to `o` exactly. Also the guard against a span
    // so small that `span / 65535` underflows to zero.
    let s = span / u16::MAX as f64;
    if !(span > 0.0) || !(s > 0.0) {
        return Some(quantize_with(
            values,
            AttrQuant { o: st.min, s: 1.0 },
            QuantLeaf::U16,
        ));
    }

    // (3) Range-adaptive UInt16 — the historical encoding, unconditional.
    Some(quantize_with(
        values,
        AttrQuant { o: st.min, s },
        QuantLeaf::U16,
    ))
}

// ----------------------------------------------------------------------------
// The dataset-global attribute-range pin (mechanism M2)
//
// Everything above this line decides the affine from the values ONE TILE holds.
// That is the recorded defect: the offset is the tile's minimum, so the same
// source value lands on a different index — and therefore decodes to a
// different number — in every tile that catches a different sample. The
// functions below apply an affine resolved ONCE from the column's dataset
// domain (the builder's pass-1 scan, `stt_build::dataset_stats`), which makes
// the verdict a function of the domain rather than of the sample. The per-tile
// functions stay exactly as they are and remain the documented fallback for an
// unpinned encode (`--single-pass`, one-shot external callers, a tile server
// with no pin sidecar).
// ----------------------------------------------------------------------------

/// Build the fixed-point column for a PINNED affine, erroring on any value the
/// pin does not cover instead of clamping it.
///
/// This is [`quantize_with`] with its clamp turned into a hard error, and the
/// difference is the whole point. On the per-tile path the clamp is provably a
/// no-op — the offset IS that tile's minimum and the leaf was chosen to hold
/// that tile's span — so clamping can never fire. A pinned affine is derived
/// somewhere else entirely (pass 1, possibly a previous build read back from
/// the manifest's `encoderPins`), so an out-of-range index is real evidence
/// that the pin no longer describes the data. Clamping it would snap the value
/// onto the domain edge and ship a wrong number with no trace, which is exactly
/// the failure [`QuantAffine::qz`] refuses to commit for altitudes.
fn quantize_pinned_with(
    values: &[Option<f64>],
    affine: AttrQuant,
    leaf: QuantLeaf,
) -> Result<(ArrayRef, String)> {
    let hi = leaf.max_index();
    let index = |x: f64| -> Result<f64> {
        let q = ((x - affine.o) / affine.s).round();
        // Negated bounds so a NaN index (unreachable from a finite value and a
        // finite affine, but not worth trusting silently) routes to the error
        // rather than past the check.
        if !(q >= 0.0 && q <= hi) {
            return Err(Error::Other(format!(
                "numeric property value {x} escapes its dataset-global quantization pin \
                 (offset {}, step {}): index {q} is outside the leaf's [0, {hi}]. The pin \
                 is derived from the whole dataset in pass 1, so a tile value outside it \
                 means the pins no longer describe the data — rebuild them, or pass \
                 --single-pass to restore the per-tile affine.",
                affine.o, affine.s
            )));
        }
        Ok(q)
    };
    let array: ArrayRef = match leaf {
        QuantLeaf::U16 => {
            let mut b = UInt16Builder::with_capacity(values.len());
            for v in values {
                match v {
                    Some(x) if x.is_finite() => b.append_value(index(*x)? as u16),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        QuantLeaf::I32 => {
            let mut b = Int32Builder::with_capacity(values.len());
            for v in values {
                match v {
                    Some(x) if x.is_finite() => b.append_value(index(*x)? as i32),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
    };
    Ok((array, affine.to_json()))
}

/// Apply a column's dataset-global pin to one tile's values — the automatic
/// (`--quantize-attrs-auto`) path's counterpart to
/// [`build_quantized_numeric_auto`], with the rule tree already evaluated.
///
/// `None` iff `pin.refuse` (the magnitude refusal), so the column stays
/// `Float64` in EVERY tile — including a tile whose own sample is small enough
/// that the per-tile rule would happily have quantized it. That inversion is
/// the fix: refusal is a property of the column's domain, and a tile that
/// happens to hold only the small end of an identifier column must not
/// re-decide it.
///
/// Otherwise the affine is applied verbatim: `index = round((value - o) / s)`
/// in `pin.leaf`. Because `o`, `s` and the leaf are the same in every tile of
/// the dataset:
///
/// * the same source value decodes to the same number everywhere (the §3.3
///   constraint the per-tile affine cannot satisfy, and the reason a value read
///   off two adjacent tiles used to disagree);
/// * the column has ONE Arrow type dataset-wide, so it stops forking a PROPS
///   schema template per width verdict and `stt-validate`'s width-drift warning
///   has nothing left to report;
/// * identical columns still quantize identically — better than before, since
///   the offset no longer varies with the sample — so content-addressed blob
///   dedup is preserved.
///
/// Nulls and non-finite cells become Arrow nulls, exactly as on the per-tile
/// path. Errors only when a value escapes the pinned domain (see
/// [`quantize_pinned_with`]).
///
/// Deliberately NOT a per-tile fallback: there is no "the pin is a bad fit for
/// this tile, quantize locally instead" branch anywhere below, because that
/// branch is the defect. Per-tile grids are a standing rejection (measured +61 %
/// on a dedup-heavy dataset for the coordinate analogue); this is the opposite
/// move.
pub fn build_quantized_numeric_pinned(
    values: &[Option<f64>],
    pin: &AttrPinned,
) -> Result<Option<(ArrayRef, String)>> {
    if pin.refuse {
        return Ok(None);
    }
    let affine = AttrQuant { o: pin.o, s: pin.s };
    Ok(Some(quantize_pinned_with(
        values,
        affine,
        QuantLeaf::from_pinned(pin.leaf),
    )?))
}

/// Resolve the quantized form of one numeric property column under an explicit
/// [`EncoderConfig`] — the single decision site the encoder's property loop
/// calls, with the dataset-global pins consulted first.
///
/// Precedence is the incumbent's, unchanged: an explicit
/// `--quantize-attr name=prec` wins; otherwise `--quantize-attrs-auto` decides;
/// otherwise the column stays `Float64`. What the pins change is the EVIDENCE
/// each of those rules sees, never the rule:
///
/// * auto → [`build_quantized_numeric_pinned`] when the column carries a pin,
///   [`build_quantized_numeric_auto`] when it does not;
/// * explicit → the user's step with the offset pinned to the dataset minimum
///   ([`build_quantized_numeric_with_pin`]).
///
/// With `cfg.global_pins == None` — `--single-pass`, a one-shot external caller
/// through the [`EncoderConfig::from_globals`] wrappers, or a tile server with
/// no pin sidecar — every call below is the incumbent per-tile function with
/// the incumbent arguments, so an unpinned encode is byte-identical to a
/// pre-pin encode. A column absent from `global_pins.attr` is likewise
/// unpinned: derived summary-tier columns (`count`, `sum_*`) that pass 1 never
/// saw keep the per-tile path rather than being forced onto a pin that does not
/// describe them.
pub fn build_quantized_numeric_for_column(
    name: &str,
    values: &[Option<f64>],
    cfg: &EncoderConfig,
) -> Result<Option<(ArrayRef, String)>> {
    let pin: Option<&AttrPinned> = cfg.global_pins.as_ref().and_then(|p| p.attr.get(name));
    if let Some(prec) = cfg.quantize_attrs.get(name).copied().filter(|p| *p > 0.0) {
        if let Some(quantized) = build_quantized_numeric_with_pin(values, prec, pin)? {
            return Ok(Some(quantized));
        }
    }
    if !cfg.quantize_attrs_auto {
        return Ok(None);
    }
    match pin {
        Some(pin) => build_quantized_numeric_pinned(values, pin),
        None => Ok(build_quantized_numeric_auto(values)),
    }
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
///
/// One refinement over a literal reading of the requested precision: when the
/// column is integer-valued and the request is `>= 1.0`, the step is snapped
/// DOWN to `1.0` so the round-trip is EXACT — `--quantize-attr count=10` must
/// not turn a count of 7 into 0 when step 1 is free, which would leave the
/// explicit path WORSE than the automatic one. Snapping down can only make the
/// reconstruction finer than asked and the affine ships the step actually used.
///
/// "Free" is the load-bearing word, and it is why the snap is gated on the
/// column's span fitting `UInt16` (65535) rather than merely fitting the `Int32`
/// leaf. The user reached for this flag to make a column SMALLER; a column
/// spanning `0..1_000_000` asked to quantize at 100 wants 10 001 indices — 2
/// bytes a row — and snapping it to step 1 would want 1 000 001, widening it to
/// `Int32` and DOUBLING the column that was supposed to shrink. Past that point
/// the requested precision stands untouched, overflow error included.
///
/// ⚠️ `#[allow(dead_code)]` is load-bearing, not rot. Since TB-2 routed the
/// encoder's property loop through [`build_quantized_numeric_for_column`], the
/// explicit path reaches [`build_quantized_numeric_with_pin`] directly (with
/// `pin = None` on an unpinned build, which is this function verbatim). This
/// no-pin spelling stays as the documented incumbent — the register's rule is
/// that the fallback path stays in the code — and as the shape the in-file and
/// `arrow_tile::tests` cases pin the per-tile behaviour against.
#[allow(dead_code)]
pub(crate) fn build_quantized_numeric(
    values: &[Option<f64>],
    prec: f64,
) -> Result<Option<(ArrayRef, String)>> {
    build_quantized_numeric_with_pin(values, prec, None)
}

/// [`build_quantized_numeric`] with the column's dataset-global pin, when the
/// build has one.
///
/// `pin = None` is the incumbent, byte-for-byte: the offset is this tile's
/// minimum and the step-1 snap is decided from this tile's statistics.
///
/// With a pin, two things become dataset-global and one deliberately does not:
///
/// * **The offset** becomes the column's dataset minimum instead of the tile's.
///   That is the whole §3.3 fix carried onto the explicit path — the user chose
///   the STEP, but the origin was never theirs to choose and its per-tile drift
///   moved every decoded value. The global minimum is ≤ every tile's minimum,
///   so indices only grow; a value that lands below it is stale-pin evidence
///   and errors rather than clamping to 0.
/// * **The step-1 snap** is decided from the pin instead of from this tile.
///   `s == 1.0` with a `U16` leaf is precisely the auto rule tree's exact-integer
///   verdict (or its degenerate constant-column case), i.e. "the DATASET is
///   integer-valued and its span fits 16 bits" — the two conditions the per-tile
///   snap tests locally. Reading them off the pin is what stops a tile that
///   happens to hold only integers from snapping while its neighbour does not.
/// * **The leaf width** stays per-tile (`max_q` below). The user's step is not
///   the pin's step, so the pin's leaf does not describe this encoding, and
///   width is a schema-template question rather than a decode-value one: both
///   leaves reconstruct through the same affine, so no value decodes two ways.
///   Pinning it needs the global span at the user's step, which is the
///   template-cost trade its own item owns.
pub(crate) fn build_quantized_numeric_with_pin(
    values: &[Option<f64>],
    prec: f64,
    pin: Option<&AttrPinned>,
) -> Result<Option<(ArrayRef, String)>> {
    if !(prec > 0.0) {
        return Ok(None);
    }
    let st = numeric_column_stats(values);
    if st.finite == 0 {
        return Ok(None); // no finite values — keep Float64
    }
    // A refused pin carries no usable affine (its `o`/`s`/`leaf` are documented
    // as meaningless), so the explicit path keeps the per-tile origin for such a
    // column and says so. The refusal itself is an AUTO-path verdict: an
    // explicit `--quantize-attr` request is the user overriding it on purpose.
    let global = pin.filter(|p| !p.refuse);
    let min = match global {
        Some(p) => p.o,
        None => st.min,
    };
    let snap_to_one = match global {
        // The global exact-integer verdict, read off the pin. See the doc
        // comment: `s == 1` in the U16 leaf IS "integer-valued dataset whose
        // span fits 16 bits".
        Some(p) => prec >= 1.0 && p.s == 1.0 && p.leaf == PinnedLeaf::U16,
        None => {
            prec >= 1.0
                && st.all_integer
                && st.max_abs <= F64_EXACT_INT_LIMIT
                // Only when step 1 keeps the leaf at UInt16 — see the doc comment: past
                // this the "exact" step is a 2x size regression on an explicit request
                // to shrink the column.
                && st.max - st.min <= QuantLeaf::U16.max_index()
        }
    };
    let step = if snap_to_one { 1.0 } else { prec };
    let affine = AttrQuant { o: min, s: step };
    let mut q: Vec<Option<i64>> = Vec::with_capacity(values.len());
    let mut max_q: i64 = 0;
    for v in values {
        match v {
            Some(x) if x.is_finite() => {
                let raw = ((*x - affine.o) / affine.s).round();
                // Under a pin the offset comes from somewhere else, so a
                // negative index is real evidence the pin is stale rather than
                // the arithmetic no-op it is on the per-tile path. Erroring
                // mirrors the overflow error below; clamping to 0 would ship
                // the column minimum in its place, silently.
                if global.is_some() && raw < 0.0 {
                    return Err(Error::Other(format!(
                        "numeric property value {x} sits below its dataset-global \
                         quantization origin {min}: index {raw} is negative. The origin is \
                         the column's dataset minimum, resolved in pass 1, so a value \
                         beneath it means the pins no longer describe the data — rebuild \
                         them, or pass --single-pass to restore the per-tile affine."
                    )));
                }
                let qi = (raw as i64).max(0);
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

// ----------------------------------------------------------------------------
// The global attribute-range pin, in-file tests.
//
// The auto path's per-tile behaviour is covered in `arrow_tile/tests.rs`; these
// cover what the PIN adds, and every one of them is written as a comparison
// against the per-tile function on the same values — because the claim is not
// "the pinned encoder works" but "the pinned encoder decides the same thing in
// every tile where the per-tile encoder decided differently".
// ----------------------------------------------------------------------------
#[cfg(test)]
mod pinned_tests {
    use super::*;
    use crate::arrow_tile::GlobalColumnPins;
    use arrow::array::{Array, Int32Array, UInt16Array};
    use arrow::datatypes::DataType;

    /// Pass 1's numeric scan at test scale: the dataset-wide statistics
    /// `AttrPinned::derive_auto` consumes, taken over the UNION of every tile.
    /// This is what the builder computes once and hands the encoder.
    fn pin_over(tiles: &[&[Option<f64>]]) -> AttrPinned {
        let all: Vec<Option<f64>> = tiles.iter().flat_map(|t| t.iter().copied()).collect();
        let st = numeric_column_stats(&all);
        AttrPinned::derive_auto(st.min, st.max, st.max_abs, st.all_integer, st.finite as u64)
    }

    /// Decode a quantized column exactly the way both reference readers do:
    /// parse the shipped affine, reconstruct every non-null cell through it.
    fn decode(quantized: &(ArrayRef, String)) -> (DataType, AttrQuant, Vec<Option<f64>>) {
        let (array, json) = quantized;
        let affine = AttrQuant::from_json(json).expect("a quantized column must ship a valid qa");
        let dt = array.data_type().clone();
        let back: Vec<Option<f64>> = match &dt {
            DataType::UInt16 => {
                let c = array.as_any().downcast_ref::<UInt16Array>().unwrap();
                (0..c.len())
                    .map(|i| (!c.is_null(i)).then(|| affine.value(c.value(i) as i64)))
                    .collect()
            }
            DataType::Int32 => {
                let c = array.as_any().downcast_ref::<Int32Array>().unwrap();
                (0..c.len())
                    .map(|i| (!c.is_null(i)).then(|| affine.value(c.value(i) as i64)))
                    .collect()
            }
            other => panic!("quantized property leaf must be UInt16/Int32, got {other:?}"),
        };
        (dt, affine, back)
    }

    fn v(xs: &[f64]) -> Vec<Option<f64>> {
        xs.iter().copied().map(Some).collect()
    }

    /// A single pinned column, wrapped as the encoder config would carry it.
    fn cfg_with(name: &str, pin: AttrPinned, auto: bool) -> EncoderConfig {
        let mut pins = GlobalColumnPins::default();
        pins.attr.insert(name.to_string(), pin);
        EncoderConfig {
            quantize_attrs_auto: auto,
            global_pins: Some(Arc::new(pins)),
            ..EncoderConfig::default()
        }
    }

    // ------------------------------------------------------------------
    // The pinned auto path.
    // ------------------------------------------------------------------

    /// The pin is applied verbatim — offset, step and leaf all come off the
    /// pin, and the shipped affine JSON is the pin's own affine.
    #[test]
    fn pinned_affine_is_applied_verbatim() {
        let pin = AttrPinned {
            refuse: false,
            o: -5.0,
            s: 1.0,
            leaf: PinnedLeaf::U16,
        };
        let values = vec![Some(-5.0), Some(0.0), None, Some(3.0), Some(f64::NAN)];
        let out = build_quantized_numeric_pinned(&values, &pin)
            .unwrap()
            .expect("a non-refusing pin always quantizes");
        assert_eq!(out.1, AttrQuant { o: -5.0, s: 1.0 }.to_json());
        let (dt, _, back) = decode(&out);
        assert_eq!(dt, DataType::UInt16);
        // Non-finite and null cells both become Arrow nulls, as on the
        // per-tile path.
        assert_eq!(
            back,
            vec![Some(-5.0), Some(0.0), None, Some(3.0), None],
            "the pinned affine must reconstruct exactly at step 1"
        );
        let c = out.0.as_any().downcast_ref::<UInt16Array>().unwrap();
        assert_eq!((c.value(0), c.value(1), c.value(3)), (0, 5, 8));
    }

    /// A refused pin keeps the column `Float64` in EVERY tile — including a
    /// tile whose own sample the per-tile rule would happily have quantized.
    ///
    /// This inversion is the point: refusal is a property of the column's
    /// DOMAIN (an identifier column is huge somewhere), and a tile holding only
    /// the small end of it must not re-decide the column's Arrow type. The
    /// per-tile assertion below is the control — it is what ships today.
    #[test]
    fn refusal_pin_keeps_float64_even_where_the_tile_would_quantize() {
        let small_tile = v(&[1.0, 2.0, 3.0]);
        let huge_tile = v(&[0.0, AUTO_QUANT_MAX_ABS + 1.0]);
        let pin = pin_over(&[&small_tile, &huge_tile]);
        assert!(pin.refuse, "the union's magnitude must trip the refusal");

        assert!(
            build_quantized_numeric_pinned(&small_tile, &pin)
                .unwrap()
                .is_none(),
            "the refusal is global: the small tile stays Float64 too"
        );
        assert!(build_quantized_numeric_pinned(&huge_tile, &pin)
            .unwrap()
            .is_none());
        // The control: left to itself, the small tile quantizes — which is
        // exactly the structural schema drift the pin removes.
        assert!(
            build_quantized_numeric_auto(&small_tile).is_some(),
            "control: the per-tile path quantizes the small tile"
        );
    }

    /// A globally integer-valued column round-trips bit-for-bit through the
    /// pinned step-1 affine, in every tile.
    #[test]
    fn pinned_step_one_integer_roundtrip_is_exact() {
        let a = v(&[0.0, 7.0, 12.0]);
        let b = v(&[-3.0, 40000.0]);
        let pin = pin_over(&[&a, &b]);
        assert_eq!((pin.refuse, pin.o, pin.s), (false, -3.0, 1.0));
        for tile in [&a, &b] {
            let out = build_quantized_numeric_pinned(tile, &pin).unwrap().unwrap();
            let (_, _, back) = decode(&out);
            assert_eq!(back, **tile, "the step-1 pin must round-trip bit-for-bit");
        }
    }

    /// The pinned LEAF is used even where the tile's own span fits a narrower
    /// one. That kills the residual `AdaptiveWidth` drift: one column, one
    /// Arrow type, one PROPS schema template.
    #[test]
    fn pinned_leaf_is_used_even_where_the_tile_fits_a_narrower_one() {
        let dense = v(&[0.0, 1.0, 2.0]);
        let wide = v(&[70_000.0]);
        let pin = pin_over(&[&dense, &wide]);
        assert_eq!(pin.leaf, PinnedLeaf::I32, "the global span exceeds 16 bits");

        let pinned = build_quantized_numeric_pinned(&dense, &pin)
            .unwrap()
            .unwrap();
        assert_eq!(decode(&pinned).0, DataType::Int32);
        // Control: the per-tile path narrows this tile to UInt16, forking the
        // column's template between dense and wide tiles.
        let per_tile = build_quantized_numeric_auto(&dense).unwrap();
        assert_eq!(decode(&per_tile).0, DataType::UInt16);
    }

    /// **The headline claim, as a hand-checked example.** One source value,
    /// present in two tiles with different local samples, decodes to ONE number
    /// under the pin and to two different numbers without it.
    #[test]
    fn one_value_decodes_one_way_in_every_tile_under_the_pin() {
        // Fractional, so the range-adaptive branch (whose step depends on the
        // span) is what runs — the integer branch is exact either way.
        let a = v(&[0.0, 1.0, 0.25]);
        let b = v(&[0.0, 100.0, 0.25]);
        let pin = pin_over(&[&a, &b]);

        let pa = decode(&build_quantized_numeric_pinned(&a, &pin).unwrap().unwrap());
        let pb = decode(&build_quantized_numeric_pinned(&b, &pin).unwrap().unwrap());
        assert_eq!(pa.1, pb.1, "both tiles must ship the SAME affine");
        assert_eq!(pa.0, pb.0, "both tiles must ship the same Arrow type");
        assert_eq!(
            pa.2[2], pb.2[2],
            "0.25 must decode identically in both tiles"
        );

        // Control: the shipped per-tile encoder decodes the same 0.25 two ways.
        let qa = decode(&build_quantized_numeric_auto(&a).unwrap());
        let qb = decode(&build_quantized_numeric_auto(&b).unwrap());
        assert_ne!(
            qa.2[2], qb.2[2],
            "control: the per-tile affine is what makes one value decode two ways"
        );
    }

    /// A column with no finite value anywhere in the dataset keeps the
    /// historical all-null `UInt16` at `{o: 0, s: 1}` — unchanged by pinning.
    #[test]
    fn globally_empty_column_matches_the_incumbent() {
        let tile: Vec<Option<f64>> = vec![None, Some(f64::INFINITY), None];
        let pin = pin_over(&[&tile]);
        let pinned = build_quantized_numeric_pinned(&tile, &pin)
            .unwrap()
            .unwrap();
        let incumbent = build_quantized_numeric_auto(&tile).unwrap();
        assert_eq!(pinned.1, incumbent.1);
        assert_eq!(pinned.0.to_data(), incumbent.0.to_data());
    }

    /// A value the pin does not cover ERRORS rather than clamping onto the
    /// domain edge, in both directions. Clamping would ship a wrong number with
    /// no trace — the failure mode `QuantAffine::qz` already refuses for
    /// altitudes.
    #[test]
    fn value_outside_the_pinned_domain_errors_instead_of_clamping() {
        let pin = AttrPinned {
            refuse: false,
            o: 0.0,
            s: 1.0,
            leaf: PinnedLeaf::U16,
        };
        let above = build_quantized_numeric_pinned(&v(&[100_000.0]), &pin);
        assert!(above.is_err(), "an index past the leaf ceiling must error");
        let below = build_quantized_numeric_pinned(&v(&[-1.0]), &pin);
        assert!(below.is_err(), "an index below the origin must error");
        // ...and an in-domain tile is untroubled.
        assert!(build_quantized_numeric_pinned(&v(&[0.0, 65535.0]), &pin).is_ok());
    }

    /// Determinism: the pinned encode is a pure function of (values, pin).
    #[test]
    fn pinned_quantization_is_deterministic() {
        let tile = vec![Some(1.5), None, Some(9.25), Some(-4.0)];
        let pin = pin_over(&[&tile, &v(&[100.0])]);
        let a = build_quantized_numeric_pinned(&tile, &pin)
            .unwrap()
            .unwrap();
        let b = build_quantized_numeric_pinned(&tile, &pin)
            .unwrap()
            .unwrap();
        assert_eq!(a.1, b.1);
        assert_eq!(a.0.to_data(), b.0.to_data());
    }

    // ------------------------------------------------------------------
    // The explicit `--quantize-attr name=prec` path.
    // ------------------------------------------------------------------

    /// The user's STEP is untouched; the ORIGIN becomes the dataset minimum.
    #[test]
    fn explicit_precision_uses_the_global_origin() {
        let a = v(&[10.5, 20.5]);
        let b = v(&[0.5, 100.5]);
        let pin = pin_over(&[&a, &b]);
        assert_eq!(pin.o, 0.5);
        assert_ne!(pin.s, 1.0, "a fractional domain must not snap to step 1");

        let pinned = build_quantized_numeric_with_pin(&a, 2.0, Some(&pin))
            .unwrap()
            .unwrap();
        let (_, affine, _) = decode(&pinned);
        assert_eq!(affine, AttrQuant { o: 0.5, s: 2.0 });
        let c = pinned.0.as_any().downcast_ref::<UInt16Array>().unwrap();
        assert_eq!((c.value(0), c.value(1)), (5, 10));

        // Control: unpinned, the offset is this tile's own minimum, so the same
        // 10.5 indexes to 0 here and to something else in the neighbouring tile.
        let per_tile = build_quantized_numeric(&a, 2.0).unwrap().unwrap();
        assert_eq!(decode(&per_tile).1, AttrQuant { o: 10.5, s: 2.0 });
    }

    /// The step-1 snap follows the GLOBAL integer verdict: it fires when the
    /// dataset is integer-valued and narrow...
    #[test]
    fn explicit_precision_snaps_to_step_one_on_a_global_integer_domain() {
        let a = v(&[0.0, 10.0]);
        let b = v(&[100.0]);
        let pin = pin_over(&[&a, &b]);
        assert_eq!((pin.s, pin.leaf), (1.0, PinnedLeaf::U16));

        let out = build_quantized_numeric_with_pin(&a, 10.0, Some(&pin))
            .unwrap()
            .unwrap();
        let (_, affine, back) = decode(&out);
        assert_eq!(affine, AttrQuant { o: 0.0, s: 1.0 }, "step snapped to 1");
        assert_eq!(back, vec![Some(0.0), Some(10.0)], "and the snap is exact");
    }

    /// ...and does NOT fire for a tile that merely happens to hold integers
    /// inside a fractional dataset. That local coincidence is precisely the
    /// per-tile drift being removed.
    #[test]
    fn explicit_precision_snap_does_not_fire_on_a_locally_integer_tile() {
        let a = v(&[3.0, 7.0]); // locally all-integer
        let b = v(&[0.5, 100.5]); // ...but the dataset is not
        let pin = pin_over(&[&a, &b]);

        let pinned = build_quantized_numeric_with_pin(&a, 2.0, Some(&pin))
            .unwrap()
            .unwrap();
        assert_eq!(decode(&pinned).1, AttrQuant { o: 0.5, s: 2.0 });

        // Control: alone, this tile snaps to step 1 at its own minimum — a
        // different affine for the same column in the very next tile.
        let per_tile = build_quantized_numeric(&a, 2.0).unwrap().unwrap();
        assert_eq!(decode(&per_tile).1, AttrQuant { o: 3.0, s: 1.0 });
    }

    /// A REFUSED pin carries no usable affine, so the explicit path — which is
    /// the user overriding the auto verdict on purpose — keeps the per-tile
    /// origin. The residual is documented, not silent.
    #[test]
    fn explicit_precision_falls_back_to_the_tile_origin_under_a_refusal_pin() {
        let tile = v(&[1.0e12, 1.0e12 + 4.0]);
        let pin = pin_over(&[&tile]);
        assert!(pin.refuse);
        let pinned = build_quantized_numeric_with_pin(&tile, 2.0, Some(&pin))
            .unwrap()
            .unwrap();
        let per_tile = build_quantized_numeric(&tile, 2.0).unwrap().unwrap();
        assert_eq!(pinned.1, per_tile.1);
        assert_eq!(pinned.0.to_data(), per_tile.0.to_data());
    }

    /// A value below the pinned origin errors on the explicit path too.
    #[test]
    fn explicit_precision_errors_below_the_global_origin() {
        let pin = AttrPinned {
            refuse: false,
            o: 10.0,
            s: 1.0,
            leaf: PinnedLeaf::U16,
        };
        assert!(build_quantized_numeric_with_pin(&v(&[5.0]), 2.0, Some(&pin)).is_err());
    }

    /// `pin = None` is the incumbent, byte for byte — the `--single-pass`
    /// rollback and the one-shot-caller path.
    #[test]
    fn unpinned_explicit_path_is_byte_identical_to_the_incumbent() {
        for values in [
            v(&[1.0, 2.0, 3.0]),
            v(&[0.25, 1000.5]),
            vec![None, Some(-7.0), Some(70_000.0)],
        ] {
            for prec in [0.5, 1.0, 10.0] {
                let a = build_quantized_numeric(&values, prec).unwrap();
                let b = build_quantized_numeric_with_pin(&values, prec, None).unwrap();
                match (a, b) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert_eq!(a.1, b.1);
                        assert_eq!(a.0.to_data(), b.0.to_data());
                    }
                    _ => panic!("pinned/unpinned disagreed on whether to quantize"),
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // The encoder's decision site.
    // ------------------------------------------------------------------

    /// With no pins attached, the dispatch is the incumbent property loop:
    /// explicit precision, else auto, else `Float64`.
    #[test]
    fn dispatch_without_pins_is_the_incumbent() {
        let values = v(&[0.25, 1.0, 2.0]);
        let auto_only = EncoderConfig {
            quantize_attrs_auto: true,
            ..EncoderConfig::default()
        };
        let got = build_quantized_numeric_for_column("speed", &values, &auto_only)
            .unwrap()
            .unwrap();
        let want = build_quantized_numeric_auto(&values).unwrap();
        assert_eq!(got.1, want.1);
        assert_eq!(got.0.to_data(), want.0.to_data());

        // Neither lever set → the column stays Float64.
        assert!(
            build_quantized_numeric_for_column("speed", &values, &EncoderConfig::default())
                .unwrap()
                .is_none()
        );
    }

    /// With pins attached, the auto path takes the pin.
    #[test]
    fn dispatch_consults_the_pin_on_the_auto_path() {
        let tile = v(&[0.0, 1.0, 0.25]);
        let pin = pin_over(&[&tile, &v(&[100.0])]);
        let cfg = cfg_with("speed", pin, true);
        let got = build_quantized_numeric_for_column("speed", &tile, &cfg)
            .unwrap()
            .unwrap();
        let want = build_quantized_numeric_pinned(&tile, &pin)
            .unwrap()
            .unwrap();
        assert_eq!(got.1, want.1);
        assert_eq!(got.0.to_data(), want.0.to_data());
        // ...and it differs from what the tile would have chosen alone.
        assert_ne!(got.1, build_quantized_numeric_auto(&tile).unwrap().1);
    }

    /// A column with no pin keeps the per-tile path even when other columns are
    /// pinned — the fallback that makes the pin set additive (derived
    /// summary-tier columns pass 1 never saw land here).
    #[test]
    fn dispatch_leaves_unpinned_columns_on_the_per_tile_path() {
        let tile = v(&[0.0, 1.0, 0.25]);
        let pin = pin_over(&[&tile, &v(&[100.0])]);
        let cfg = cfg_with("other", pin, true);
        let got = build_quantized_numeric_for_column("speed", &tile, &cfg)
            .unwrap()
            .unwrap();
        let want = build_quantized_numeric_auto(&tile).unwrap();
        assert_eq!(got.1, want.1);
        assert_eq!(got.0.to_data(), want.0.to_data());
    }

    /// An explicit precision still wins over auto, and it is the PINNED
    /// explicit path that runs.
    #[test]
    fn dispatch_prefers_explicit_precision_over_auto() {
        let a = v(&[10.5, 20.5]);
        let pin = pin_over(&[&a, &v(&[0.5, 100.5])]);
        let mut cfg = cfg_with("speed", pin, true);
        cfg.quantize_attrs.insert("speed".to_string(), 2.0);
        let got = build_quantized_numeric_for_column("speed", &a, &cfg)
            .unwrap()
            .unwrap();
        assert_eq!(decode(&got).1, AttrQuant { o: 0.5, s: 2.0 });
    }

    /// The incumbent's `.or_else`: an explicit request over a column with no
    /// finite value in this tile falls through to auto rather than dropping the
    /// quantization.
    #[test]
    fn dispatch_falls_back_to_auto_when_the_tile_has_no_finite_value() {
        let tile: Vec<Option<f64>> = vec![None, None];
        let mut cfg = EncoderConfig {
            quantize_attrs_auto: true,
            ..EncoderConfig::default()
        };
        cfg.quantize_attrs.insert("speed".to_string(), 2.0);
        let got = build_quantized_numeric_for_column("speed", &tile, &cfg)
            .unwrap()
            .expect("auto keeps the all-null UInt16");
        assert_eq!(decode(&got).0, DataType::UInt16);
    }

    // ------------------------------------------------------------------
    // Rejected-design boundary.
    // ------------------------------------------------------------------

    /// **Guard: the rate menu is not widened by making the verdict global.**
    ///
    /// Per-column rate selection beyond `{UInt16, Int32}` only becomes
    /// representable once the affine is a function of the dataset rather than
    /// of a tile's sample — and is deliberately not exercised here, so the
    /// change surface stays reviewable and the extra menu entries keep their
    /// unpriced schema-template cost. Both matches below are exhaustive: a
    /// third variant in either enum stops this compiling.
    #[test]
    fn pinned_rate_menu_is_exactly_two_leaves() {
        for pinned in [PinnedLeaf::U16, PinnedLeaf::I32] {
            let leaf = QuantLeaf::from_pinned(pinned);
            match (pinned, leaf) {
                (PinnedLeaf::U16, QuantLeaf::U16) => {
                    assert_eq!(leaf.max_index(), u16::MAX as f64)
                }
                (PinnedLeaf::I32, QuantLeaf::I32) => {
                    assert_eq!(leaf.max_index(), i32::MAX as f64)
                }
                (p, l) => panic!("pinned leaf {p:?} must not map to encoder leaf {l:?}"),
            }
            assert_eq!(pinned.max_index(), leaf.max_index());
        }
        // Every pin the derivation can produce lands in that two-entry menu.
        for pin in [
            AttrPinned::derive_auto(0.0, 10.0, 10.0, true, 3),
            AttrPinned::derive_auto(0.0, 70_000.0, 70_000.0, true, 3),
            AttrPinned::derive_auto(0.0, 1.0, 1.0, false, 3),
        ] {
            assert!(matches!(pin.leaf, PinnedLeaf::U16 | PinnedLeaf::I32));
        }
    }
}
