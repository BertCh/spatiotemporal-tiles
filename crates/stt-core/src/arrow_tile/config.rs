//! Encoder settings — the explicit [`EncoderConfig`] and the surviving
//! process-wide globals it can be snapshotted from.
//!
//! The globals (and their `set_*` / getter pairs) exist only for external
//! one-shot callers of the no-arg [`encode_tile`](super::encode::encode_tile)
//! wrappers; every producer in this workspace passes an `EncoderConfig`
//! explicitly instead.

use super::columns::DEFAULT_VERTEX_TIME_MAX_STEP_MS;
use super::frame::{TemplateCollector, FORMAT_VERSION};
use super::layer::VectorElem;
use super::quantize::validate_quantize_coords_m;
use crate::error::Result;
use std::collections::HashMap;
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
/// Historically these lived only in process-wide mutable statics set via the
/// `set_*` functions above; that made a new encode caller silently inherit
/// whatever the last `set_*` left behind (the origin of the `stt-serve` parity
/// bug, where one dataset's quantization bled into the tiles served for
/// another) and made it impossible to encode two different configurations in
/// one process (blocking multi-dataset serve + parallel per-config tests).
/// Passing an `EncoderConfig` explicitly to
/// [`encode_tile_with`](crate::arrow_tile::encode_tile_with) removes both
/// problems, and it is how BOTH shipping producers now encode: `stt-build`
/// carries one on its [`crate::PackWriter`], `stt-serve` one per dataset. The
/// surviving globals + the no-arg [`encode_tile`](crate::arrow_tile::encode_tile) /
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
    /// Ceiling (ms) on the per-vertex time u16-delta quantization step.
    pub vertex_time_max_step_ms: u32,
    /// Layer-frame format version. Only [`FORMAT_VERSION`] is accepted; the
    /// field is retained so a frame-format revision has an explicit, checked
    /// channel rather than a silent assumption at the encode boundary.
    pub format_version: u32,
    /// When set, layer schemas are hoisted into this collector and frames carry
    /// 16-byte template-hash references (the packed-dataset mode —
    /// `PackWriter::finalize` publishes the collected templates in
    /// `manifest.schemas`). `None` emits self-contained frames with inline
    /// schema sections (what `stt-serve` uses: no manifest to carry a registry).
    pub template_collector: Option<Arc<TemplateCollector>>,
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
            format_version: FORMAT_VERSION,
            template_collector: None,
        }
    }
}

impl EncoderConfig {
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
