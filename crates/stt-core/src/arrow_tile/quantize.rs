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
pub(crate) fn build_quantized_numeric(
    values: &[Option<f64>],
    prec: f64,
) -> Result<Option<(ArrayRef, String)>> {
    if !(prec > 0.0) {
        return Ok(None);
    }
    let st = numeric_column_stats(values);
    if st.finite == 0 {
        return Ok(None); // no finite values — keep Float64
    }
    let min = st.min;
    let step = if prec >= 1.0
        && st.all_integer
        && st.max_abs <= F64_EXACT_INT_LIMIT
        // Only when step 1 keeps the leaf at UInt16 — see the doc comment: past
        // this the "exact" step is a 2x size regression on an explicit request
        // to shrink the column.
        && st.max - st.min <= QuantLeaf::U16.max_index()
    {
        1.0
    } else {
        prec
    };
    let affine = AttrQuant { o: min, s: step };
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
