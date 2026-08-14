//! Encoder settings — the explicit [`EncoderConfig`] and the surviving
//! process-wide globals it can be snapshotted from.
//!
//! The globals (and their `set_*` / getter pairs) exist only for external
//! one-shot callers of the no-arg [`encode_tile`](super::encode::encode_tile)
//! wrappers; every producer in this workspace passes an `EncoderConfig`
//! explicitly instead.

use super::columns::DEFAULT_VERTEX_TIME_MAX_STEP_MS;
use super::frame::{template_cost_justifies_with, TemplateCollector, LAYER_FRAME_VERSION};
use super::layer::VectorElem;
use super::quantize::{validate_quantize_coords_m, AUTO_QUANT_MAX_ABS};
use crate::error::Result;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

/// Build-global coordinate quantization precision, in **micrometers** (lets the
/// `AtomicU32` carry sub-mm..km without a float). `0` = off (Float64 coords).
/// Set once per build (e.g. `stt-build --quantize-coords`); read by the default
/// [`encode_tile`] / [`encode_layer`] path so it covers both the streaming and
/// in-memory builders without threading through every tile.
static QUANTIZE_COORDS_UM: AtomicU32 = AtomicU32::new(0);

/// Set the build-global coordinate quantization precision in meters for every
/// subsequent default [`encode_tile`](crate::arrow_tile::encode_tile) call.
/// `<= 0` (the default) turns it off — coordinates stay Float64 GeoArrow. See
/// [`encode_layer_quantized`](crate::arrow_tile::encode_layer_quantized) for the
/// size/precision trade-off. Errors (storing nothing) for a positive precision
/// below [`MIN_QUANTIZE_COORDS_M`](crate::arrow_tile::MIN_QUANTIZE_COORDS_M), which would
/// overflow the world grid's longitude index.
pub fn set_quantize_coords_m(meters: f64) -> Result<()> {
    validate_quantize_coords_m(meters)?;
    let um = if meters > 0.0 {
        (meters * 1.0e6).round().clamp(1.0, u32::MAX as f64) as u32
    } else {
        0
    };
    QUANTIZE_COORDS_UM.store(um, Ordering::Relaxed);
    Ok(())
}

/// The build-global quantization precision in meters, or `None` when off.
pub fn quantize_coords_m() -> Option<f64> {
    let um = QUANTIZE_COORDS_UM.load(Ordering::Relaxed);
    (um > 0).then(|| um as f64 / 1.0e6)
}
/// Build-global map of `property-name → ground precision (units)`. A numeric
/// property named here is stored quantized (see [`build_quantized_numeric`]);
/// every other numeric property stays `Float64`. Set once per build from
/// `stt-build --quantize-attr name=prec`. Empty (the default) ⇒ all numeric
/// properties stay `Float64`, byte-identical to the historical encoder.
fn quant_attrs_cell() -> &'static RwLock<HashMap<String, f64>> {
    static A: OnceLock<RwLock<HashMap<String, f64>>> = OnceLock::new();
    A.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Replace the build-global numeric-property quantization map.
pub fn set_quantize_attrs(map: HashMap<String, f64>) {
    *quant_attrs_cell().write().unwrap() = map;
}

/// The current numeric-property quantization map (a clone).
pub fn quantize_attrs() -> HashMap<String, f64> {
    quant_attrs_cell().read().unwrap().clone()
}

/// When set, EVERY `Float64` numeric property not given an explicit precision in
/// the [`set_quantize_attrs`] map is quantized automatically: its step is sized
/// so the column's full `[min, max]` range spans the 16-bit index space
/// (`step = (max-min)/65535`), i.e. a range-adaptive `UInt16` with ~65k levels.
/// This is the "born-optimized" generation default — no per-column precision to
/// pick, and >=16 bits of dynamic range is visually lossless for the scalar
/// fields STT carries (magnitude, depth, altitude, speed, SST, dBZ, ...). The
/// default-off keeps `stt-build` byte-identical unless a caller opts in.
static QUANTIZE_ATTRS_AUTO: AtomicBool = AtomicBool::new(false);

/// Enable/disable automatic range-adaptive quantization of every otherwise-raw
/// `Float64` numeric property (see [`QUANTIZE_ATTRS_AUTO`]). Explicit precisions
/// in [`set_quantize_attrs`] always win.
pub fn set_quantize_attrs_auto(on: bool) {
    QUANTIZE_ATTRS_AUTO.store(on, Ordering::Relaxed);
}

/// Whether automatic numeric-property quantization is enabled.
pub fn quantize_attrs_auto() -> bool {
    QUANTIZE_ATTRS_AUTO.load(Ordering::Relaxed)
}
/// A build-time directive to fuse several scalar numeric properties into one
/// GPU-ready interleaved [`PropertyColumn::Vector`](crate::arrow_tile::PropertyColumn::Vector). The
/// component order is the
/// vector's component order (e.g. `["qx","qy","qz","qw"]` → `instanceQuaternions`).
/// Applied at encode time (like the quantization maps), so the producer keeps
/// emitting plain scalar columns and the renderer still binds the result
/// zero-copy. See [`set_vector_groups`].
#[derive(Debug, Clone)]
pub struct VectorGroup {
    /// Output column name (the FixedSizeList field name the decoder keys on).
    pub name: String,
    /// Source scalar-property names, in component order.
    pub components: Vec<String>,
    /// Leaf upload type (`F32` for quat/scale, `U8` for 0–255 RGBA).
    pub elem: VectorElem,
}
/// Build-global list of vector groups (see [`VectorGroup`]). Set once per build
/// from `stt-build --vector-group`. Empty (the default) ⇒ every property stays a
/// scalar column, byte-identical to the historical encoder.
fn vector_groups_cell() -> &'static RwLock<Vec<VectorGroup>> {
    static A: OnceLock<RwLock<Vec<VectorGroup>>> = OnceLock::new();
    A.get_or_init(|| RwLock::new(Vec::new()))
}

/// Replace the build-global vector-group list.
pub fn set_vector_groups(groups: Vec<VectorGroup>) {
    *vector_groups_cell().write().unwrap() = groups;
}

/// The current vector-group list (a clone).
pub fn vector_groups() -> Vec<VectorGroup> {
    vector_groups_cell().read().unwrap().clone()
}
/// Build-global name of the numeric property to fold into POINT geometry as the
/// 3rd (altitude) coordinate, so the tile ships true 3D points
/// (`FixedSizeList<_,3>`) the renderer binds zero-copy — no per-point pad to 3D
/// on the main thread. The named column is REMOVED from the property set (it
/// lives in the geometry instead). Empty (default) ⇒ plain 2D points.
fn point_elevation_column_cell() -> &'static RwLock<String> {
    static A: OnceLock<RwLock<String>> = OnceLock::new();
    A.get_or_init(|| RwLock::new(String::new()))
}

/// Set the build-global point-elevation column name (see above). Empty disables.
pub fn set_point_elevation_column(name: &str) {
    *point_elevation_column_cell().write().unwrap() = name.to_string();
}

/// The current point-elevation column name (empty ⇒ disabled).
pub fn point_elevation_column() -> String {
    point_elevation_column_cell().read().unwrap().clone()
}
/// Resolved, explicit encoder settings — the values the tile encoder reads at
/// encode time (coordinate + attribute quantization, vector grouping, the
/// point-elevation fold, vertex-time precision, frame version, template sink).
///
/// Encoder settings must travel EXPLICITLY, through
/// [`encode_tile_with`](crate::arrow_tile::encode_tile_with), rather than
/// through the process-wide mutable statics behind the `set_*` functions above.
/// A caller that reads the globals silently inherits whatever the last `set_*`
/// left behind — that is how one dataset's quantization bleeds into the tiles
/// `stt-serve` returns for another — and two configurations cannot coexist in
/// one process at all, which rules out multi-dataset serve and parallel
/// per-config tests. Both shipping producers pass a config: `stt-build`
/// carries one on its [`crate::PackWriter`], `stt-serve` one per dataset. The
/// globals + the no-arg [`encode_tile`](crate::arrow_tile::encode_tile) /
/// [`encode_layer`](crate::arrow_tile::encode_layer) wrappers (snapshotted by
/// [`EncoderConfig::from_globals`]) serve only external
/// one-shot callers; nothing in this workspace's build path mutates them.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Fixed-point coordinate quantization ground precision in meters
    /// (`None` = Float64 GeoArrow coordinates, the default).
    pub quantize_coords_m: Option<f64>,
    /// Per-property explicit fixed-point precisions (`name → precision`).
    pub quantize_attrs: HashMap<String, f64>,
    /// Range-adaptive `UInt16` quantization for every un-listed Float64 property.
    pub quantize_attrs_auto: bool,
    /// Scalar columns to fuse into interleaved `FixedSizeList` vector columns.
    pub vector_groups: Vec<VectorGroup>,
    /// Property folded into POINT geometry as the z coordinate (empty = none).
    pub point_elevation_column: String,
    /// Ceiling (ms) on the per-vertex time delta quantization step.
    ///
    /// See [`EncoderConfig::vertex_time_step_for_bucket`] — a coarse
    /// temporal-LOD tile scales this proportionally (TB-11).
    pub vertex_time_max_step_ms: u32,
    /// Compact the per-feature time columns (default ON, the `time-delta`
    /// manifest capability). Per layer, from that layer's own data:
    /// `start_time` becomes a non-null `UInt32` offset from `TILE_META.t0`
    /// when every offset fits, and `end_time` becomes a non-null `UInt32`
    /// DURATION — or is omitted outright when `end == start` for every
    /// feature. `TILE_META.st` / `.et` record the choice and both reference
    /// readers re-inflate absolute `Int64` columns, so nothing downstream of
    /// the decoder can tell the difference.
    ///
    /// Only the FRAME path (`encode_tile*`) compacts. The standalone
    /// `encode_layer*` shape has no `TILE_META` section to carry `st`/`et`
    /// and its `decode_layer` counterpart performs no re-inflation, so
    /// emitting the compact form there would silently re-type a column with
    /// no way to read it back.
    ///
    /// Deliberately NOT a process-wide global (like `format_version` and
    /// `template_collector`): it is a per-build/per-request choice that has to
    /// travel with the writer that declares the capability.
    pub compact_times: bool,
    /// Quantize the per-vertex value columns (`vertex_value` and
    /// `vertex_value_matrix`) to a `UInt16` leaf under a per-column
    /// range-adaptive affine — the opt-in `vertex-value-quant` manifest
    /// capability, half the bytes of the raw `List<Float32>`.
    ///
    /// Those two are the format's ONLY `List<Float32>` columns and they had no
    /// size lever at all (`stt:qa` re-types scalar per-feature property columns
    /// only), while measuring 64.2% of `nyc-taxi-flows` and 93.7% of
    /// `bixi-corridors` tile bytes. The affines ride `TILE_META.vq`; the
    /// reserved index
    /// [`VERTEX_VALUE_QUANT_SENTINEL`](super::quantize::VERTEX_VALUE_QUANT_SENTINEL)
    /// carries the `NaN` "no value here" marker across, and both reference
    /// readers re-inflate `Float32`.
    ///
    /// Default OFF, and opt-in rather than automatic because it is genuinely
    /// LOSSY (16 bits across the column's own range) on data a map colours by,
    /// unlike the compact-time and delta encodings which are exact.
    ///
    /// Like [`Self::compact_times`] this is only sound on the FRAME path: the
    /// affine's only channel is `TILE_META`, which the standalone
    /// `encode_layer*` shape does not have.
    pub quantize_vertex_values: bool,
    /// Layer-frame format version. Only [`LAYER_FRAME_VERSION`] is accepted; the
    /// field is retained so a frame-format revision has an explicit, checked
    /// channel rather than a silent assumption at the encode boundary.
    pub format_version: u32,
    /// When set, layer schemas are hoisted into this collector and frames carry
    /// 16-byte template-hash references (the packed-dataset mode —
    /// `PackWriter::finalize` publishes the collected templates in
    /// `manifest.schemas`). `None` emits self-contained frames with inline
    /// schema sections (what `stt-serve` uses: no manifest to carry a registry).
    pub template_collector: Option<Arc<TemplateCollector>>,
    /// Dataset-global encoding verdicts resolved by the builder's pass-1 scan
    /// (`stt_build::dataset_stats`), or `None` for the incumbent per-tile path.
    ///
    /// This is the channel mechanism M2 needs: a per-tile encoder cannot see
    /// the dataset, so every verdict it makes locally (the numeric affine, the
    /// dictionary-vs-`Utf8` choice, the integer leaf width) is a function of
    /// which rows happened to land in the tile. That is exactly the
    /// conformance-invariance violation §13.2 names — the same value decodes
    /// differently in different tiles, and one column forks a PROPS template
    /// per verdict. Pinning the verdicts to the column's dataset DOMAIN makes
    /// them tile-independent by construction.
    ///
    /// `None` (the default, and what `--single-pass` restores) keeps every
    /// incumbent per-tile decision exactly as it is, so an encode with no pins
    /// is byte-identical to a pre-M2 encode. A one-shot external caller and
    /// `stt-serve` without a pin sidecar both land here.
    pub global_pins: Option<Arc<GlobalColumnPins>>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            quantize_coords_m: None,
            quantize_attrs: HashMap::new(),
            quantize_attrs_auto: false,
            vector_groups: Vec::new(),
            point_elevation_column: String::new(),
            vertex_time_max_step_ms: DEFAULT_VERTEX_TIME_MAX_STEP_MS,
            compact_times: true,
            quantize_vertex_values: false,
            format_version: LAYER_FRAME_VERSION,
            template_collector: None,
            global_pins: None,
        }
    }
}

impl EncoderConfig {
    /// TB-11 — the per-vertex time step ceiling for a tile in a temporal-LOD
    /// tier of width `bucket_ms`, given the archive's `base_bucket_ms`.
    ///
    /// A 30-day tier tile tolerates proportionally coarser vertex time than a
    /// 1-hour base tile at equal PERCEPTUAL error: the same wall-clock step is
    /// a far smaller fraction of the coarse tile's span. Holding the ceiling
    /// fixed is what pushes wide temporal-LOD layers into the exact-i64
    /// fallback and costs 2-4x (§2.5).
    ///
    /// Scales linearly in `bucket_ms / base_bucket_ms`, saturating at `u32`.
    /// A base-tier tile (equal widths, or either unknown) is unchanged, so the
    /// default path is byte-identical.
    pub fn vertex_time_step_for_bucket(&self, bucket_ms: Option<u64>, base_bucket_ms: u64) -> u32 {
        let step = self.vertex_time_max_step_ms;
        let (Some(bucket), true) = (bucket_ms, base_bucket_ms > 0) else {
            return step;
        };
        if bucket <= base_bucket_ms {
            return step;
        }
        let ratio = bucket / base_bucket_ms;
        u64::from(step)
            .saturating_mul(ratio)
            .try_into()
            .unwrap_or(u32::MAX)
    }

    /// TB-11 — `self` with the vertex-time ceiling scaled for a tier tile.
    /// Returns `self` unchanged for the base tier, so the common path allocates
    /// nothing new and encodes identically.
    pub fn for_temporal_tier(
        &self,
        bucket_ms: Option<u64>,
        base_bucket_ms: u64,
    ) -> std::borrow::Cow<'_, Self> {
        let scaled = self.vertex_time_step_for_bucket(bucket_ms, base_bucket_ms);
        if scaled == self.vertex_time_max_step_ms {
            std::borrow::Cow::Borrowed(self)
        } else {
            let mut c = self.clone();
            c.vertex_time_max_step_ms = scaled;
            std::borrow::Cow::Owned(c)
        }
    }

    /// Snapshot the surviving process-wide encoder globals into an explicit
    /// config. Used by the no-arg [`encode_tile`](crate::arrow_tile::encode_tile) /
    /// [`encode_layer`](crate::arrow_tile::encode_layer) back-compat wrappers.
    ///
    /// The frame version and template sink are deliberately NOT global: they
    /// have to match the writer that will store the frame, so they come from
    /// [`crate::PackWriter::encoder_config`] and default to a self-contained
    /// (inline-schema) frame here.
    pub fn from_globals() -> Self {
        Self {
            quantize_coords_m: quantize_coords_m(),
            quantize_attrs: quantize_attrs(),
            quantize_attrs_auto: quantize_attrs_auto(),
            vector_groups: vector_groups(),
            point_elevation_column: point_elevation_column(),
            ..Self::default()
        }
    }
}

// ----------------------------------------------------------------------------
// Dataset-global column pins (mechanism M2)
// ----------------------------------------------------------------------------

/// The integer leaf a PINNED quantized property column ships in.
///
/// Mirrors the encoder's private `QuantLeaf` deliberately rather than exposing
/// it: `QuantLeaf` is an encode-time implementation detail of
/// [`super::quantize`], while this is a wire-visible verdict that has to
/// survive into a manifest sidecar and back. The MENU IS EXACTLY TWO — `UInt16`
/// (the 2-byte default) and `Int32` (the widening the exact-integer path needs
/// when a column's span exceeds 16 bits). Widening the menu is deferred work
/// with its own cost term (the schema-template externality); it is NOT part of
/// making the verdict global, and `pinned_leaf_menu_is_exactly_two` guards that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinnedLeaf {
    /// `UInt16` — indices `0..=65535`.
    U16,
    /// `Int32` — indices `0..=i32::MAX`.
    I32,
}

impl PinnedLeaf {
    /// Largest index the leaf can hold (indices are always ≥ 0: the offset is
    /// the column minimum). Mirrors `QuantLeaf::max_index`.
    #[inline]
    pub fn max_index(self) -> f64 {
        match self {
            PinnedLeaf::U16 => u16::MAX as f64,
            PinnedLeaf::I32 => i32::MAX as f64,
        }
    }
}

/// One numeric column's pinned affine — the `value → index` map every tile of
/// the dataset applies, derived once from the column's DATASET domain.
///
/// The three fields are the whole contract: `refuse` keeps the column
/// `Float64` everywhere, and otherwise `index = round((value - o) / s)` into
/// `leaf`. Because `o`, `s` and `leaf` no longer depend on which rows a tile
/// caught, the same source value decodes to the same number in every tile —
/// the §3.3 constraint the per-tile affine cannot satisfy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttrPinned {
    /// Keep the column `Float64` in every tile (the magnitude refusal). When
    /// set, `o`/`s`/`leaf` are meaningless and must not be applied.
    pub refuse: bool,
    /// Affine offset — the column's dataset-wide minimum finite value.
    pub o: f64,
    /// Affine step.
    pub s: f64,
    /// Integer leaf the indices ship in.
    pub leaf: PinnedLeaf,
}

impl AttrPinned {
    /// Derive the pin the AUTO path (`--quantize-attrs-auto`) wants, from a
    /// column's dataset-wide statistics.
    ///
    /// This is `build_quantized_numeric_auto`'s rule tree verbatim, evaluated
    /// over the DATASET's `(min, max, max_abs, all_integer, finite)` instead of
    /// one tile's. Keeping the rule identical is the point: the change is which
    /// evidence the rule sees, never how a given body of evidence is
    /// classified. In order:
    ///
    /// 1. **No finite value at all** → the historical all-null `UInt16` at
    ///    `{o: 0, s: 1}`.
    /// 2. **Magnitude refusal** (`max_abs >= AUTO_QUANT_MAX_ABS`) → `refuse`.
    ///    Globally this is now exact rather than nearly-exact: the residual
    ///    caveat the per-tile version documents — a column mixing magnitudes
    ///    across the threshold flipping tile to tile — cannot occur when the
    ///    threshold is tested against the dataset's own `max_abs`.
    /// 3. **Integer-valued column** whose span the `Int32` leaf can index at
    ///    step 1 → exact fixed point, narrowest leaf that holds the span.
    /// 4. **Everything else** → range-adaptive `UInt16`, `s = span / 65535`,
    ///    with the degenerate zero-span case pinned to step 1.
    pub fn derive_auto(min: f64, max: f64, max_abs: f64, all_integer: bool, finite: u64) -> Self {
        // (1) Nothing finite: the historical all-null UInt16 at {o: 0, s: 1}.
        if finite == 0 {
            return AttrPinned {
                refuse: false,
                o: 0.0,
                s: 1.0,
                leaf: PinnedLeaf::U16,
            };
        }
        // (2) The one refusal, tested BEFORE the integer/fraction split.
        if max_abs >= AUTO_QUANT_MAX_ABS {
            return AttrPinned {
                refuse: true,
                o: 0.0,
                s: 1.0,
                leaf: PinnedLeaf::U16,
            };
        }
        let span = max - min;
        // (3) Exact fixed point at step 1, narrowest leaf that indexes the span.
        if all_integer && span <= PinnedLeaf::I32.max_index() {
            return AttrPinned {
                refuse: false,
                o: min,
                s: 1.0,
                leaf: if span <= PinnedLeaf::U16.max_index() {
                    PinnedLeaf::U16
                } else {
                    PinnedLeaf::I32
                },
            };
        }
        // A constant column, or a span so small `span / 65535` underflows.
        //
        // The NEGATED comparisons are load-bearing and must not be rewritten as
        // `span <= 0.0`: a NaN span (unreachable from finite statistics, but the
        // incumbent guards it anyway) makes every ordered comparison false, so
        // only the negated form routes it to the safe step-1 branch. Written
        // exactly as `build_quantized_numeric_auto` writes it.
        let s = span / u16::MAX as f64;
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let degenerate = !(span > 0.0) || !(s > 0.0);
        if degenerate {
            return AttrPinned {
                refuse: false,
                o: min,
                s: 1.0,
                leaf: PinnedLeaf::U16,
            };
        }
        // (4) Range-adaptive UInt16.
        AttrPinned {
            refuse: false,
            o: min,
            s,
            leaf: PinnedLeaf::U16,
        }
    }

    /// The index this pin maps `value` to — `round((value - o) / s)`, the same
    /// arithmetic the encoder applies per cell. Meaningless (and not to be
    /// called) when [`Self::refuse`] is set.
    #[inline]
    pub fn index_of(&self, value: f64) -> f64 {
        ((value - self.o) / self.s).round()
    }

    /// Price this pin's LEAF WIDTH against the schema template a per-tile width
    /// choice would fork, given the column's dataset-wide maximum.
    ///
    /// `None` when the pin refuses (a `Float64` column has no integer leaf and
    /// forks nothing). The general entry point — used by the explicit
    /// `--quantize-attr name=prec` path, where the step is the user's and only
    /// the offset is pinned, so the global maximum has to be re-indexed through
    /// the affine to find out whether the leaf straddles at all. The AUTO path
    /// has [`Self::derive_auto_priced`], which knows the answer exactly and
    /// needs no float re-derivation.
    pub fn width_pin(
        &self,
        global_max: f64,
        rows: u64,
        template_cost_bytes: u64,
    ) -> Option<WidthPin> {
        (!self.refuse)
            .then(|| decide_leaf_width(self.index_of(global_max), rows, template_cost_bytes))
    }

    /// [`Self::derive_auto`] plus the priced width verdict — the AUTO pin rule
    /// with the schema-template cost term applied.
    ///
    /// `rows` is the DATASET-wide row count of the column and
    /// `template_cost_bytes` the marginal cost of one more schema template
    /// (measured via
    /// [`TemplateCollector::mean_template_bytes`](super::frame::TemplateCollector::mean_template_bytes),
    /// else [`DEFAULT_TEMPLATE_COST_BYTES`](super::frame::DEFAULT_TEMPLATE_COST_BYTES)).
    ///
    /// The affine returned is `derive_auto`'s, unchanged — pricing the width
    /// never moves `o`, `s`, or the refusal, so a caller that ignores the second
    /// element gets exactly the incumbent pin. Only the two exact-integer
    /// outcomes can straddle, and `derive_auto` sizes the leaf from the global
    /// span, so the straddle test here is exact rather than a float re-derivation:
    /// an `I32` leaf means the span exceeded the `UInt16` index space and per-tile
    /// leaves would differ; a `U16` leaf means every tile of the dataset indexes
    /// inside 16 bits and there is nothing to trade.
    pub fn derive_auto_priced(
        min: f64,
        max: f64,
        max_abs: f64,
        all_integer: bool,
        finite: u64,
        rows: u64,
        template_cost_bytes: u64,
    ) -> (Self, Option<WidthPin>) {
        let pin = Self::derive_auto(min, max, max_abs, all_integer, finite);
        let width = if pin.refuse {
            None
        } else if pin.leaf == PinnedLeaf::U16 {
            // Every dataset value indexes inside `UInt16`, so does every tile's.
            Some(WidthPin::Pinned(PinnedLeaf::U16))
        } else {
            // The only way `derive_auto` reaches `I32`: the exact step-1 branch
            // with `65535 < span <= i32::MAX`, where the global max index IS the
            // span exactly (`o == min`, `s == 1`).
            Some(decide_leaf_width(max - min, rows, template_cost_bytes))
        };
        (pin, width)
    }
}

/// Bytes per row an `Int32` leaf costs over a `UInt16` one — the `2 B` of the
/// §3.8 gap-1 trade.
pub const WIDTH_PIN_EXTRA_BYTES_PER_ROW: u64 = 2;

/// The dataset-wide verdict on a pinned numeric column's INTEGER LEAF WIDTH.
///
/// The one remaining per-tile freedom once mechanism M2 pins the affine, and
/// the one the schema-template cost term prices. Note what pinning the affine
/// already bought: with `o` and `s` global, the leaf width no longer affects
/// what a value DECODES to — a tile storing index 70 000 as `Int32` and a tile
/// storing index 12 as `UInt16` reconstruct through the same `o + q·s`. That is
/// why `classify_column_drift` rates this variation `AdaptiveWidth` rather than
/// `Structural`, and why it can be settled on bytes alone: the choice is purely
/// "one more schema template" against "2 bytes a row".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidthPin {
    /// One leaf for the whole dataset. Costs `2 B` on every row that would have
    /// fitted `UInt16`; buys a single realized PROPS shape for the column.
    Pinned(PinnedLeaf),
    /// Let each tile pick the narrowest leaf that indexes its own rows — the
    /// incumbent behaviour, which the explicit-precision path has always had.
    /// Reachable ONLY in the straddling case, and only when the dataset is big
    /// enough that `rows × 2 B` outweighs the extra template.
    Adaptive,
}

/// The §3.8 gap-1 trade: pin a column's integer leaf, or let it adapt per tile?
///
/// `global_max_index` is the largest index the PINNED affine produces anywhere
/// in the dataset.
///
/// * `global_max_index <= u16::MAX` — every tile fits `UInt16`, so no per-tile
///   choice exists and the trade is not consulted. Pinning is free. A NaN index
///   lands here too: the straddle is asserted POSITIVELY (`index > u16::MAX`),
///   which is false for NaN, so an unorderable input routes to the safe narrow
///   branch the same way [`AttrPinned::derive_auto`]'s own guards do.
/// * otherwise the column STRADDLES: dense tiles would widen to `Int32`, sparse
///   ones would not, forking one extra PROPS template. Pin always-`Int32` when
///   the extra width costs LESS than that template — `rows × 2 B < c_tmpl` —
///   and otherwise leave the width adaptive, which is what
///   [`template_cost_justifies_with`] says when the payload saving is the bigger
///   number.
pub fn decide_leaf_width(global_max_index: f64, rows: u64, template_cost_bytes: u64) -> WidthPin {
    let straddles = global_max_index > PinnedLeaf::U16.max_index();
    if !straddles {
        return WidthPin::Pinned(PinnedLeaf::U16);
    }
    let width_savings = rows.saturating_mul(WIDTH_PIN_EXTRA_BYTES_PER_ROW);
    if template_cost_justifies_with(1, width_savings, template_cost_bytes) {
        WidthPin::Adaptive
    } else {
        WidthPin::Pinned(PinnedLeaf::I32)
    }
}

/// One categorical column's dataset-global Arrow type verdict.
///
/// Today the encoder decides `Dictionary<UInt16, Utf8>` vs plain `Utf8` per
/// tile from that tile's own value bytes, so one column is a dictionary in the
/// dense tiles and a `Utf8` in the sparse ones — which forks a PROPS schema
/// template per variant and defeats content-addressed dedup for equal content.
/// One column, one type, everywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobalDictVerdict {
    /// Dictionary-encode against this exact category list, in dataset-wide
    /// first-seen order (the pinned ordering rule, widened from per-tile to
    /// per-dataset). Every tile ships the same list, so the list itself can be
    /// hoisted into the schema TEMPLATE instead of re-shipped per tile.
    Dictionary(Arc<Vec<String>>),
    /// Plain `Utf8` in every tile — the column's distinct set overflowed the
    /// pass-1 cap (so no global list exists) or the dataset-scale size
    /// comparison preferred it.
    Utf8,
}

/// The resolved dataset-global encoding verdicts, keyed by property-column
/// name.
///
/// `BTreeMap`, not `HashMap`: this struct is serialized into an archive
/// sidecar and compared across rebuilds, so its iteration order is part of the
/// output. A column absent from a map is UNPINNED and keeps the incumbent
/// per-tile behaviour — that is the fallback that makes every consumer of this
/// struct additive and reversible.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GlobalColumnPins {
    /// Numeric columns → pinned affine.
    pub attr: BTreeMap<String, AttrPinned>,
    /// Categorical columns → pinned Arrow type verdict.
    pub dict: BTreeMap<String, GlobalDictVerdict>,
}

impl GlobalColumnPins {
    /// True when nothing is pinned (every column keeps its per-tile decision).
    pub fn is_empty(&self) -> bool {
        self.attr.is_empty() && self.dict.is_empty()
    }

    /// Canonical JSON for the archive sidecar: sorted keys, no whitespace, one
    /// spelling per value.
    ///
    /// Canonical because this is co-versioned INTO the archive (the manifest's
    /// additive `encoderPins` field) and read back by `stt-serve` so a served
    /// tile stays byte-identical to the offline build's. Two runs over one
    /// dataset must produce the same string or the manifest — and every
    /// content address downstream of it — moves. `BTreeMap` gives the key
    /// order; `serde_json`'s float formatting gives the value spelling.
    ///
    /// A non-finite `o`/`s` cannot occur (both come from finite column
    /// statistics) and is emitted as `null` rather than panicking, so a future
    /// caller with a degenerate pin gets an obviously-wrong sidecar instead of
    /// a crash.
    pub fn to_canonical_json(&self) -> String {
        let num = |x: f64| {
            serde_json::Number::from_f64(x)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        };
        let mut attr = serde_json::Map::new();
        for (name, pin) in &self.attr {
            // Inserted in SORTED key order on purpose: `serde_json::Map` is a
            // `BTreeMap` by default (sorted) but an `IndexMap` under the
            // `preserve_order` feature (insertion-ordered). Inserting sorted
            // makes the emitted string identical either way, so a transitive
            // feature flip cannot silently move every archive's manifest.
            let mut entry = serde_json::Map::new();
            entry.insert(
                "leaf".into(),
                serde_json::Value::String(
                    match pin.leaf {
                        PinnedLeaf::U16 => "u16",
                        PinnedLeaf::I32 => "i32",
                    }
                    .to_string(),
                ),
            );
            entry.insert("o".into(), num(pin.o));
            entry.insert("refuse".into(), serde_json::Value::Bool(pin.refuse));
            entry.insert("s".into(), num(pin.s));
            attr.insert(name.clone(), serde_json::Value::Object(entry));
        }
        let mut dict = serde_json::Map::new();
        for (name, verdict) in &self.dict {
            let value = match verdict {
                GlobalDictVerdict::Utf8 => serde_json::Value::String("utf8".to_string()),
                GlobalDictVerdict::Dictionary(categories) => {
                    let mut entry = serde_json::Map::new();
                    entry.insert(
                        "categories".into(),
                        serde_json::Value::Array(
                            categories
                                .iter()
                                .map(|c| serde_json::Value::String(c.clone()))
                                .collect(),
                        ),
                    );
                    serde_json::Value::Object(entry)
                }
            };
            dict.insert(name.clone(), value);
        }
        let mut root = serde_json::Map::new();
        root.insert("attr".into(), serde_json::Value::Object(attr));
        root.insert("dict".into(), serde_json::Value::Object(dict));
        serde_json::Value::Object(root).to_string()
    }
}

#[cfg(test)]
mod pin_tests {
    use super::*;

    /// The `{U16, I32}` menu is not widened by making the verdict global.
    ///
    /// A rejected-design boundary made executable: per-column RATE selection
    /// beyond these two leaves becomes *possible* once the affine is a function
    /// of the dataset (rather than of a tile's sample), and is deliberately not
    /// exercised — the change surface stays reviewable, and the extra menu
    /// entries have an unpriced schema-template cost. If a third variant is
    /// ever added this exhaustive match stops compiling, which is the point.
    #[test]
    fn pinned_leaf_menu_is_exactly_two() {
        for leaf in [PinnedLeaf::U16, PinnedLeaf::I32] {
            match leaf {
                PinnedLeaf::U16 => assert_eq!(leaf.max_index(), 65535.0),
                PinnedLeaf::I32 => assert_eq!(leaf.max_index(), 2147483647.0),
            }
        }
    }

    /// The derived pin reproduces the per-tile auto rule tree, branch by branch.
    #[test]
    fn derive_auto_reproduces_the_rule_tree() {
        // (1) no finite value → all-null UInt16 at {0, 1}
        let none = AttrPinned::derive_auto(f64::INFINITY, f64::NEG_INFINITY, 0.0, true, 0);
        assert_eq!(
            none,
            AttrPinned {
                refuse: false,
                o: 0.0,
                s: 1.0,
                leaf: PinnedLeaf::U16
            }
        );

        // (2) magnitude refusal at exactly i32::MAX
        let refused = AttrPinned::derive_auto(0.0, AUTO_QUANT_MAX_ABS, AUTO_QUANT_MAX_ABS, true, 9);
        assert!(refused.refuse);
        // ...and NOT one ulp below it.
        let kept = AttrPinned::derive_auto(0.0, 2.0, AUTO_QUANT_MAX_ABS - 1.0, true, 9);
        assert!(!kept.refuse);

        // (3) integer column, narrow span → exact step-1 UInt16 at the minimum
        let narrow = AttrPinned::derive_auto(-5.0, 60000.0, 60000.0, true, 100);
        assert_eq!(
            narrow,
            AttrPinned {
                refuse: false,
                o: -5.0,
                s: 1.0,
                leaf: PinnedLeaf::U16
            }
        );
        // ...and the same column with a span past 16 bits widens to Int32.
        let wide = AttrPinned::derive_auto(0.0, 70000.0, 70000.0, true, 100);
        assert_eq!(wide.leaf, PinnedLeaf::I32);
        assert_eq!((wide.o, wide.s), (0.0, 1.0));

        // (4) fractional column → range-adaptive UInt16
        let frac = AttrPinned::derive_auto(1.0, 2.0, 2.0, false, 100);
        assert!(!frac.refuse);
        assert_eq!(frac.leaf, PinnedLeaf::U16);
        assert_eq!(frac.o, 1.0);
        assert_eq!(frac.s, 1.0 / u16::MAX as f64);

        // degenerate: a constant fractional column pins to step 1
        let constant = AttrPinned::derive_auto(0.5, 0.5, 0.5, false, 100);
        assert_eq!(
            constant,
            AttrPinned {
                refuse: false,
                o: 0.5,
                s: 1.0,
                leaf: PinnedLeaf::U16
            }
        );
    }

    /// The sidecar spelling is stable and sorted — the manifest byte contract.
    #[test]
    fn canonical_json_is_sorted_and_stable() {
        let mut pins = GlobalColumnPins::default();
        pins.attr.insert(
            "zulu".to_string(),
            AttrPinned {
                refuse: false,
                o: -1.5,
                s: 0.25,
                leaf: PinnedLeaf::I32,
            },
        );
        pins.attr.insert(
            "alpha".to_string(),
            AttrPinned {
                refuse: true,
                o: 0.0,
                s: 1.0,
                leaf: PinnedLeaf::U16,
            },
        );
        pins.dict
            .insert("kind".to_string(), GlobalDictVerdict::Utf8);
        pins.dict.insert(
            "colour".to_string(),
            GlobalDictVerdict::Dictionary(Arc::new(vec!["red".to_string(), "blue".to_string()])),
        );

        let json = pins.to_canonical_json();
        assert_eq!(
            json,
            r#"{"attr":{"alpha":{"leaf":"u16","o":0.0,"refuse":true,"s":1.0},"zulu":{"leaf":"i32","o":-1.5,"refuse":false,"s":0.25}},"dict":{"colour":{"categories":["red","blue"]},"kind":"utf8"}}"#
        );
        // Re-serializing the same pins must not move a byte.
        assert_eq!(json, pins.to_canonical_json());
        // ...and the dictionary list keeps FIRST-SEEN order, not sorted order.
        assert!(json.contains(r#"["red","blue"]"#));
    }

    /// The width-pin trade fires in the constructed straddling case, and NOT
    /// otherwise — no matter how large the dataset gets.
    #[test]
    fn width_trade_fires_only_in_the_straddling_case() {
        // Not straddling: every tile fits UInt16, so the trade is never
        // consulted and the row count is irrelevant.
        for rows in [0u64, 1, 1_000, 44_100_000] {
            assert_eq!(
                decide_leaf_width(65535.0, rows, 1_000),
                WidthPin::Pinned(PinnedLeaf::U16),
                "a column indexing inside 16 bits has no width to trade"
            );
            assert_eq!(
                decide_leaf_width(0.0, rows, 1_000),
                WidthPin::Pinned(PinnedLeaf::U16)
            );
        }
        // A non-finite index routes to the safe narrow branch (the negated
        // comparison), like every other rule-tree guard in this file.
        assert_eq!(
            decide_leaf_width(f64::NAN, 10, 1_000),
            WidthPin::Pinned(PinnedLeaf::U16)
        );

        // Straddling and SMALL: 100 rows × 2 B = 200 B loses to a 1000 B
        // template, so the leaf is pinned wide.
        assert_eq!(
            decide_leaf_width(65536.0, 100, 1_000),
            WidthPin::Pinned(PinnedLeaf::I32)
        );
        // Straddling and LARGE: 10_000 rows × 2 B = 20 kB beats one template,
        // so the width stays adaptive.
        assert_eq!(
            decide_leaf_width(65536.0, 10_000, 1_000),
            WidthPin::Adaptive
        );

        // Break-even is NOT worth a fork: 500 × 2 B == 1000 B exactly.
        assert_eq!(
            decide_leaf_width(65536.0, 500, 1_000),
            WidthPin::Pinned(PinnedLeaf::I32)
        );
        assert_eq!(
            decide_leaf_width(65536.0, 501, 1_000),
            WidthPin::Adaptive,
            "one row past break-even flips it"
        );

        // Pure function: same inputs, same verdict, every time.
        for _ in 0..4 {
            assert_eq!(
                decide_leaf_width(65536.0, 501, 1_000),
                decide_leaf_width(65536.0, 501, 1_000)
            );
        }
    }

    /// Pricing the width never moves the affine, and only the exact-integer
    /// wide-leaf outcome can straddle.
    #[test]
    fn derive_auto_priced_prices_only_the_wide_leaf() {
        let cases: [(f64, f64, f64, bool, u64); 5] = [
            (f64::INFINITY, f64::NEG_INFINITY, 0.0, true, 0), // no finite value
            (0.0, AUTO_QUANT_MAX_ABS, AUTO_QUANT_MAX_ABS, true, 9), // refusal
            (-5.0, 60000.0, 60000.0, true, 100),              // narrow integer
            (0.0, 70000.0, 70000.0, true, 100),               // WIDE integer
            (1.0, 2.0, 2.0, false, 100),                      // fractional
        ];
        for (min, max, max_abs, all_integer, finite) in cases {
            let plain = AttrPinned::derive_auto(min, max, max_abs, all_integer, finite);
            let (priced, _) =
                AttrPinned::derive_auto_priced(min, max, max_abs, all_integer, finite, 10, 1_000);
            assert_eq!(plain, priced, "pricing must not move the affine");
        }

        let width = |min, max, max_abs, all_integer, finite, rows| {
            AttrPinned::derive_auto_priced(min, max, max_abs, all_integer, finite, rows, 1_000).1
        };
        // Refused column: no integer leaf exists, so there is nothing to price.
        assert_eq!(
            width(0.0, AUTO_QUANT_MAX_ABS, AUTO_QUANT_MAX_ABS, true, 9, 10),
            None
        );
        // No finite value / narrow integer / fractional: all indexed inside 16
        // bits, so all pinned narrow regardless of dataset size.
        assert_eq!(
            width(f64::INFINITY, f64::NEG_INFINITY, 0.0, true, 0, 44_100_000),
            Some(WidthPin::Pinned(PinnedLeaf::U16))
        );
        assert_eq!(
            width(-5.0, 60000.0, 60000.0, true, 100, 44_100_000),
            Some(WidthPin::Pinned(PinnedLeaf::U16))
        );
        assert_eq!(
            width(1.0, 2.0, 2.0, false, 100, 44_100_000),
            Some(WidthPin::Pinned(PinnedLeaf::U16)),
            "a range-adaptive column tops out at index 65535 by construction"
        );
        // The one straddling outcome — and it is the only one that reads `rows`.
        assert_eq!(
            width(0.0, 70000.0, 70000.0, true, 100, 10),
            Some(WidthPin::Pinned(PinnedLeaf::I32))
        );
        assert_eq!(
            width(0.0, 70000.0, 70000.0, true, 100, 44_100_000),
            Some(WidthPin::Adaptive)
        );
    }

    /// The explicit-precision entry point re-indexes the global maximum through
    /// the pinned affine — the case where the step is the user's, not the span's.
    #[test]
    fn width_pin_reads_the_pinned_affine() {
        // `--quantize-attr depth=0.5` over a column running 0..40000: the user's
        // step makes the global max index 80000, past the UInt16 space, so per-tile
        // leaves would straddle even though the raw span does not.
        let explicit = AttrPinned {
            refuse: false,
            o: 0.0,
            s: 0.5,
            leaf: PinnedLeaf::U16,
        };
        assert_eq!(explicit.index_of(40000.0), 80000.0);
        assert_eq!(
            explicit.width_pin(40000.0, 100, 1_000),
            Some(WidthPin::Pinned(PinnedLeaf::I32))
        );
        assert_eq!(
            explicit.width_pin(40000.0, 10_000, 1_000),
            Some(WidthPin::Adaptive)
        );
        // The same column at step 1 indexes inside 16 bits — no straddle.
        let coarse = AttrPinned { s: 1.0, ..explicit };
        assert_eq!(
            coarse.width_pin(40000.0, 10_000, 1_000),
            Some(WidthPin::Pinned(PinnedLeaf::U16))
        );
        // A refused column has no leaf to pin.
        let refused = AttrPinned {
            refuse: true,
            ..explicit
        };
        assert_eq!(refused.width_pin(40000.0, 100, 1_000), None);
    }

    /// An encoder config with no pins is the incumbent path.
    #[test]
    fn default_encoder_config_carries_no_pins() {
        assert!(EncoderConfig::default().global_pins.is_none());
        assert!(EncoderConfig::from_globals().global_pins.is_none());
        assert!(GlobalColumnPins::default().is_empty());
    }
}
