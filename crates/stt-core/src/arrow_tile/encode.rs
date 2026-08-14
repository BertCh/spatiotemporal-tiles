//! Tile encoding — layer parts and sectioned-frame assembly.
//!
//! The public `encode_layer*` / `encode_tile*` entry points, the shared
//! [`build_layer_parts`] front end both output shapes run, and the layer-frame
//! assembler. [`EncoderConfig::format_version`] names the layer-frame version
//! and is checked against [`LAYER_FRAME_VERSION`] at the encode boundary, so an
//! unsupported value errors instead of silently getting the one version this
//! writer emits.

use super::columns::{
    build_dictionary_indices, build_geometry_array_q, build_part_offsets_array,
    build_vertex_time_array, build_vertex_value_array, group_vector_properties,
    infer_vertex_value_buckets,
};
use super::config::{EncoderConfig, GlobalDictVerdict};
use super::frame::{
    EndTimeForm, StartTimeForm, TileMeta, FRAME_ALIGN, FRAME_V2_ESCAPE, FRAME_V2_VERSION,
    LAYER_FRAME_VERSION, REF_KIND_INLINE, REF_KIND_NO_PROPS, REF_KIND_TEMPLATE_HASH,
    SECTION_CORE_BATCH, SECTION_INLINE_SCHEMA_CORE, SECTION_INLINE_SCHEMA_PROPS,
    SECTION_PROPS_BATCH, SECTION_TILE_META, TIME_OFFSET_MS_KEY, VERTEX_TIME_FEATURE_STEP_KEY,
    VERTEX_TIME_ORIGIN_KEY, VERTEX_TIME_STEP_KEY, VERTEX_VALUE_BUCKETS_KEY,
};
use super::layer::{
    ColumnarLayer, GeometryColumn, PropertyColumn, VectorElem, GEOARROW_CRS_METADATA,
    GEOARROW_EXT_KEY, GEOARROW_EXT_META_KEY, TRIANGLES_METADATA_KEY,
};
use super::quantize::{
    build_quantized_numeric_for_column, validate_quantize_coords_m, world_grid_affine,
    world_grid_affine_3d, AttrQuant, STT_QUANT_ATTR_META_KEY, STT_QUANT_META_KEY,
};
use crate::error::{Error, Result};
use arrow::array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
    ListBuilder, RecordBatch, StringArray, UInt16Array, UInt16Builder, UInt32Array, UInt32Builder,
    UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema, UInt16Type};
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow::ipc::{root_as_message, MessageHeader, MetadataVersion};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

// Arrow IPC dictionaries add a schema/dictionary-message cost that dominates
// tiny tile-local columns. Keep the estimate deliberately conservative: the
// variable buffers are exact, while this allowance covers IPC framing and
// alignment that is otherwise difficult to predict without encoding twice.
//
// This allowance is charged ONCE PER TILE, which is what makes it dominate a
// sparse tile — and it is exactly the term that stops being true under the
// dataset-global dictionary hoist, where one dictionary message serves the
// whole dataset. See [`dataset_dictionary_is_smaller`], the dataset-scale
// sibling of the comparison below, and the `global_pins` path in
// [`build_layer_parts`], on which this constant is dead.
const CATEGORICAL_DICTIONARY_IPC_OVERHEAD: usize = 192;

pub(crate) fn categorical_dictionary_is_smaller(
    values: &[Option<String>],
    categories: &[String],
) -> bool {
    let plain_value_bytes = values
        .iter()
        .filter_map(Option::as_ref)
        .map(String::len)
        .sum::<usize>();
    let dictionary_value_bytes = categories.iter().map(String::len).sum::<usize>();

    // Utf8 uses an i32 offset per row plus one terminal offset. Dictionary
    // values have their own offset buffer and UInt16 keys for every row.
    let plain_bytes =
        plain_value_bytes.saturating_add((values.len() + 1).saturating_mul(size_of::<i32>()));
    let dictionary_bytes = dictionary_value_bytes
        .saturating_add((categories.len() + 1).saturating_mul(size_of::<i32>()))
        .saturating_add(values.len().saturating_mul(size_of::<u16>()))
        .saturating_add(CATEGORICAL_DICTIONARY_IPC_OVERHEAD);

    dictionary_bytes < plain_bytes
}

// ----------------------------------------------------------------------------
// Dataset-global categorical verdicts (TB-3) and the dictionary hoist (TB-4)
// ----------------------------------------------------------------------------

/// The dataset-scale sibling of [`categorical_dictionary_is_smaller`]: the
/// pre-compression surrogate applied to a COLUMN's dataset totals rather than to
/// one tile's sample, for deriving a
/// [`GlobalDictVerdict`](super::config::GlobalDictVerdict).
///
/// Two things change when the comparison moves from a tile to the dataset, and
/// both push the verdict toward the dictionary:
///
/// 1. **The IPC allowance is charged once, not per tile.** Under the hoist the
///    dictionary message is template-resident — one copy for the whole dataset —
///    so the 192-byte per-tile allowance that dominates a sparse tile's
///    arithmetic simply does not apply. It is still charged here, once, so the
///    comparison stays conservative in the same direction it always was.
/// 2. **The category list is counted once, not once per tile.** That is the
///    43.9 %-of-uncompressed-bytes prize the hoist is chasing.
///
/// This is the FALLBACK derivation. The measured one is the trial-encode oracle
/// (`stt_optimize::oracle`'s `Candidate::CategoricalDict`), which prices both
/// forms POST-ZSTD over a real sample — the D4 correction, because a
/// heavy-repeat `Utf8` column compresses far closer to its own dictionary than
/// this arithmetic claims. Where the oracle is available its verdict wins; this
/// function is what a caller with no sample reaches for.
///
/// **Boundary:** the result is a DATASET verdict. It must never be evaluated per
/// tile — a verdict that depends on which rows a tile caught is precisely the
/// §13.2 conformance-invariance violation the pin exists to remove.
///
/// - `total_value_bytes` / `values`: summed UTF-8 length of every non-null value
///   in the column, and the number of rows (null or not) across the dataset.
/// - `category_bytes` / `categories`: summed UTF-8 length of the distinct
///   category set, and its cardinality.
pub fn dataset_dictionary_is_smaller(
    total_value_bytes: u64,
    values: u64,
    category_bytes: u64,
    categories: u64,
) -> bool {
    // Utf8: one i32 offset per row plus one terminal offset, and every value's
    // bytes. Dictionary: the category set's bytes and offsets ONCE, plus a
    // UInt16 key per row, plus one IPC allowance for the single hoisted message.
    let plain = total_value_bytes.saturating_add(values.saturating_add(1).saturating_mul(4));
    let dictionary = category_bytes
        .saturating_add(categories.saturating_add(1).saturating_mul(4))
        .saturating_add(values.saturating_mul(2))
        .saturating_add(CATEGORICAL_DICTIONARY_IPC_OVERHEAD as u64);
    dictionary < plain
}

/// The largest pinned category list this encoder will dictionary-encode:
/// `u16::MAX` entries, i.e. keys `0..=65_534`.
///
/// One below the full `UInt16` key space on purpose, and the bound is a READER
/// contract rather than an arithmetic one. The TS reader publishes a categorical
/// column as `{indices: Uint16Array, categories: string[]}` and spells "this row
/// has no category" as the IN-BAND sentinel `0xffff`
/// (`packages/core/src/tile.ts`), so a live key of 65_535 would be
/// indistinguishable from a null. The incumbent per-tile builder can never mint
/// one — `build_dictionary_indices` stops accepting new categories at
/// `u16::MAX` — but pass 1's distinct-set cap is `MAX_CATEGORIES = 65_536`, one
/// wider, so a *pinned* list can reach the sentinel where a tile-local one
/// cannot. A pin above this bound degrades to `Utf8`, and does so for the WHOLE
/// dataset: the degrade is a function of the pin, so one column is still one
/// type everywhere and §13.2 holds.
const MAX_PINNED_CATEGORIES: usize = u16::MAX as usize;

/// The all-or-nothing rule, made unrepresentable.
///
/// TB-4 hoists a categorical column's category list out of every tile and into
/// the schema TEMPLATE, which is only sound if the list is *provably* identical
/// in every tile. The failure mode to design out is a PARTIAL hoist — a tile
/// whose hoisted dictionary carries only the categories that tile happened to
/// contain. That forks a template per category subset, which is strictly worse
/// than not hoisting at all.
///
/// The module below is the entire enforcement mechanism: [`PinnedCategories`]
/// wraps a borrowed slice whose field is private to this module, and the module
/// exposes exactly one constructor, whose only input is a pin. There is no
/// `From<Vec<String>>`, no constructor over tile values, and no way to shorten
/// or filter one. A tile-local subset therefore cannot be *spelled* anywhere in
/// this encoder, so it cannot be encoded — the rule is a property of the type,
/// not of reviewer vigilance.
mod pinned {
    use super::{GlobalDictVerdict, MAX_PINNED_CATEGORIES};

    /// A dataset-global category list, in the only form the hoist accepts: the
    /// WHOLE list, borrowed straight out of the pin that declared it.
    #[derive(Clone, Copy)]
    pub(super) struct PinnedCategories<'a> {
        /// PRIVATE to `mod pinned`, and that is the whole point — see the
        /// module doc. Nothing outside these few lines can put a value here.
        categories: &'a [String],
    }

    impl<'a> PinnedCategories<'a> {
        /// The ONLY constructor. `None` means "do not dictionary-encode this
        /// column anywhere in the dataset", for either of two reasons:
        ///
        /// - the pin is [`GlobalDictVerdict::Utf8`] — the dataset-scale verdict
        ///   (or a pass-1 cap overflow) chose plain strings; or
        /// - the pinned list exceeds [`MAX_PINNED_CATEGORIES`], which would
        ///   collide with the reader's in-band null sentinel.
        ///
        /// Both are functions of the pin alone, so both are dataset-global.
        pub(super) fn from_verdict(verdict: &'a GlobalDictVerdict) -> Option<Self> {
            match verdict {
                GlobalDictVerdict::Utf8 => None,
                GlobalDictVerdict::Dictionary(categories) => {
                    (categories.len() <= MAX_PINNED_CATEGORIES).then(|| PinnedCategories {
                        categories: categories.as_slice(),
                    })
                }
            }
        }

        /// The pinned list, verbatim and complete — first-seen order preserved.
        pub(super) fn as_slice(self) -> &'a [String] {
            self.categories
        }
    }
}
use pinned::PinnedCategories;

/// How ONE categorical column is typed in EVERY tile of the dataset (TB-3).
///
/// The three arms are the whole contract, and the first two are dataset-global
/// by construction: they are resolved from `EncoderConfig::global_pins` and from
/// nothing else, so no property of the tile being encoded can reach them. That
/// is the §13.2 invariance rule — a column's Arrow type is a function of the
/// dataset DOMAIN, never of which rows a tile caught.
enum CategoricalPlan<'a> {
    /// Pinned dictionary: build against this exact global list in every tile,
    /// including tiles holding a strict subset of it and tiles holding none of
    /// it at all.
    Dictionary(PinnedCategories<'a>),
    /// Pinned `Utf8`: plain strings in every tile, and no dictionary message
    /// anywhere in the dataset for this column.
    Utf8,
    /// UNPINNED — the incumbent per-tile size comparison
    /// ([`categorical_dictionary_is_smaller`]). This is the documented fallback
    /// that keeps the change additive and reversible: an encode with no pins
    /// (`--single-pass`, a one-shot external caller, `stt-serve` without a pin
    /// sidecar) is byte-identical to a pre-M2 encode.
    PerTile,
}

/// Resolve one categorical column's dataset-global plan from the config's pins.
fn categorical_plan<'a>(cfg: &'a EncoderConfig, name: &str) -> CategoricalPlan<'a> {
    let Some(verdict) = cfg
        .global_pins
        .as_deref()
        .and_then(|pins| pins.dict.get(name))
    else {
        return CategoricalPlan::PerTile;
    };
    match PinnedCategories::from_verdict(verdict) {
        Some(categories) => CategoricalPlan::Dictionary(categories),
        None => CategoricalPlan::Utf8,
    }
}

/// Build one categorical column against the PINNED dataset-global category
/// list — the writer half of TB-4.
///
/// Every tile ships the same `DictionaryArray` value array (the full list, in
/// dataset-wide first-seen order), so the Arrow `DictionaryBatch` message is
/// byte-identical in every tile and can move into the schema template. Keys are
/// `UInt16` indices into that list; a null value stays a null key.
///
/// A value absent from the pin is a HARD ERROR, never a silent degrade: it means
/// pass 1 and the encoded feature stream disagree about the column's domain, and
/// every downstream guarantee (one type everywhere, a constant dictionary
/// message, a hoistable template) rests on them agreeing.
fn build_pinned_dictionary_column(
    name: &str,
    values: &[Option<String>],
    pinned: PinnedCategories<'_>,
) -> Result<ArrayRef> {
    let categories = pinned.as_slice();
    // Lookup only — never iterated — so the HashMap contributes no ordering
    // non-determinism to the output. `or_insert` keeps the FIRST index for a
    // (malformed) duplicated pin entry, which is what a first-seen list means.
    let mut lookup: HashMap<&str, u16> = HashMap::with_capacity(categories.len());
    for (index, category) in categories.iter().enumerate() {
        debug_assert!(
            index <= u16::MAX as usize,
            "bounded by MAX_PINNED_CATEGORIES"
        );
        lookup.entry(category.as_str()).or_insert(index as u16);
    }

    let mut indices: Vec<Option<u16>> = Vec::with_capacity(values.len());
    for value in values {
        match value {
            None => indices.push(None),
            Some(s) => match lookup.get(s.as_str()) {
                Some(&index) => indices.push(Some(index)),
                None => {
                    return Err(Error::Other(format!(
                        "categorical column '{name}' carries the value {s:?}, which is absent \
                         from its dataset-global dictionary pin ({} categories). Pass-1 \
                         statistics and the encoded features disagree about this column's \
                         domain — re-run pass 1 over the same feature set, or fall back with \
                         --single-pass.",
                        categories.len()
                    )))
                }
            },
        }
    }

    let value_array: ArrayRef = Arc::new(StringArray::from(
        categories
            .iter()
            .map(|s| Some(s.as_str()))
            .collect::<Vec<_>>(),
    ));
    let key_array = UInt16Array::from(indices);
    Ok(Arc::new(
        DictionaryArray::<UInt16Type>::try_new(key_array, value_array).map_err(|e| {
            Error::Other(format!("pinned dictionary build failed for '{name}': {e}"))
        })?,
    ))
}

/// Where a PROPS stream's Arrow `DictionaryBatch` messages belong.
///
/// This is TB-4's split decision, and it is deliberately a whole-STREAM verdict
/// rather than a per-column one: the template/tail cut is a single byte offset
/// into one IPC stream, so either every dictionary message before the record
/// batch is dataset-constant and they all move, or none do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DictHoist {
    /// Leave them in the per-tile TAIL — the incumbent split. Either the stream
    /// carries no dictionary at all, or at least one of its dictionaries is
    /// tile-local (an unpinned column that the per-tile surrogate happened to
    /// dictionary-encode). Hoisting a tile-local dictionary would mint a
    /// template per tile, which is far worse than not hoisting.
    ///
    /// **The mixed stream is designed out rather than tolerated here.** A
    /// stream carrying one PINNED and one tile-local dictionary would land on
    /// this arm and ship the pinned column's FULL dataset-global list in every
    /// tile's tail, un-deduplicated — strictly worse than both alternatives.
    /// [`unpinned_categoricals_take_utf8`] makes that combination unreachable,
    /// so `Tail` now means only "no pin reached this stream at all".
    Tail,
    /// Every dictionary in the stream was built against a pinned dataset-global
    /// list, so the messages are byte-identical across every tile of this layer
    /// shape and belong in the TEMPLATE — stored once in `manifest.schemas`
    /// instead of once per tile.
    Template,
}

/// Does any categorical property column in this layer carry a pinned
/// dictionary? If so, the columns that DON'T must take `Utf8` rather than a
/// tile-local dictionary.
///
/// # The worst-of-both-worlds case this removes
///
/// The hoist is a whole-stream verdict (see [`DictHoist`]). Before this rule, a
/// layer mixing a pinned column with an unpinned one whose per-tile surrogate
/// happened to choose a dictionary fell to [`DictHoist::Tail`] — and `Tail`
/// puts *every* DictionaryBatch in the tail, including the pinned one. The
/// pinned column's whole dataset-global category list then rode in every single
/// tile, once per tile, with no template dedup: bigger than the un-pinned
/// encoding AND bigger than the hoisted one. The likeliest place to hit it is a
/// summary/LOD tier, where derived columns are unpinned by design.
///
/// The unpinned column also re-introduces the very fork TB-3 exists to kill:
/// its per-tile verdict flips with tile density, so the PROPS schema (and hence
/// the template) differs between dense and sparse tiles.
///
/// Both are fixed by the same one-line rule, and the rule is invariance-safe:
/// its input is the (config pins × layer column set) pair, never a tile's rows,
/// so a column's Arrow type still cannot depend on which features a tile
/// caught. It is also a no-op on the two common shapes — no pins at all
/// (`--single-pass`, `stt-serve` without a sidecar) leaves every column on the
/// incumbent per-tile path, and an all-pinned layer never had an unpinned
/// column to demote.
fn unpinned_categoricals_take_utf8(
    cfg: &EncoderConfig,
    props: &[(String, PropertyColumn)],
) -> bool {
    props.iter().any(|(name, col)| {
        matches!(col, PropertyColumn::Categorical(_))
            && matches!(categorical_plan(cfg, name), CategoricalPlan::Dictionary(_))
    })
}

// ----------------------------------------------------------------------------
// Encoding
// ----------------------------------------------------------------------------

/// Encode a single layer to an Arrow IPC stream.
pub fn encode_layer(layer: &ColumnarLayer) -> Result<Vec<u8>> {
    encode_layer_cfg(layer, &EncoderConfig::from_globals())
}

/// [`encode_layer`] with optional fixed-point coordinate quantization.
///
/// `quantize_m = Some(meters)` stores coordinates as `i32` grid indices at that
/// ground precision (default-off `None` is byte-identical to [`encode_layer`]).
/// Coordinates are the dominant, near-incompressible tile column, so quantizing
/// them is the single largest size lever — at the cost of GeoArrow Float64
/// self-description, hence opt-in. The per-layer affine rides in the geometry
/// field metadata under [`STT_QUANT_META_KEY`]; the reader reconstructs Float64.
///
/// The non-coordinate settings (attribute quantization, vector grouping,
/// point-elevation fold, vertex-time precision) come from the process-wide
/// globals; use [`encode_tile_with`] to pass every setting explicitly.
pub fn encode_layer_quantized(layer: &ColumnarLayer, quantize_m: Option<f64>) -> Result<Vec<u8>> {
    encode_layer_cfg(
        layer,
        &EncoderConfig {
            quantize_coords_m: quantize_m,
            ..EncoderConfig::from_globals()
        },
    )
}

/// [`encode_layer`] with a fully-explicit [`EncoderConfig`] — no process-wide
/// globals are read. The single-layer sibling of [`encode_tile_with`], and the
/// entry point a caller should reach for instead of `set_*`-then-`encode_layer`:
/// that pair is what let one caller's settings leak into the next one's tiles.
pub fn encode_layer_with(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<Vec<u8>> {
    encode_layer_cfg(layer, cfg)
}

/// The single-layer encode implementation, driven entirely by an explicit
/// [`EncoderConfig`] — no process-wide globals are read. Every public
/// `encode_layer*` entry point funnels here.
///
/// Crate-private: the whole-TILE [`encode_tile_with`] is the public
/// explicit-config entry point (that is the granularity every writer works at),
/// so a public single-layer twin only ever went unused.
///
/// Always emits the self-describing shape — one Arrow IPC stream with ALL
/// metadata inline. The tile frame reuses the identical column/field build
/// ([`build_layer_parts`]) and changes only metadata PLACEMENT at assembly
/// ([`encode_layer_v2_parts`]).
pub(crate) fn encode_layer_cfg(layer: &ColumnarLayer, cfg: &EncoderConfig) -> Result<Vec<u8>> {
    // `FrameOnlyEncodings::default()` — every field off. This shape has no
    // TILE_META section to carry the `st`/`et`/`vq` discriminants and
    // `decode_layer` performs no re-inflation, so taking any of them here
    // would re-type a column with no way to read it back.
    let parts = build_layer_parts(layer, cfg, FrameOnlyEncodings::default())?;
    assemble_layer_ipc_self_describing(parts)
}

/// The encodings that are only legal on the FRAME path, because the key that
/// discriminates each one lives in the frame's `TILE_META` section.
///
/// Passed by the caller rather than read off [`EncoderConfig`] so the
/// standalone-layer shape (which has no `TILE_META`, and whose `decode_layer`
/// counterpart performs no re-inflation) cannot silently pick one up: the
/// `Default` is "all off", and only [`encode_layer_v2_parts`] fills it in.
#[derive(Debug, Clone, Copy, Default)]
struct FrameOnlyEncodings {
    /// `TILE_META.st` / `.et` — see [`EncoderConfig::compact_times`].
    compact_times: bool,
    /// `TILE_META.vq` — see [`EncoderConfig::quantize_vertex_values`].
    quantize_vertex_values: bool,
}

/// Everything a frame assembler needs about one built layer: the Arrow fields
/// + columns in canonical order (reserved columns first, then properties) and
/// the per-tile schema-metadata values. Both output shapes build this
/// identically; only where the metadata LANDS differs — the standalone
/// `encode_layer*` IPC stream writes it as Arrow schema metadata, while the
/// layer frame hoists it into the `TILE_META` section so the schema stays
/// dataset-constant and collapses to one shared template.
struct LayerParts {
    fields: Vec<Arc<Field>>,
    columns: Vec<ArrayRef>,
    /// `fields[..reserved_len]` are the reserved (CORE) columns; the rest are
    /// property (PROPS) columns.
    reserved_len: usize,
    layer_name: String,
    geometry_name: &'static str,
    /// Minimum feature start-time: the `stt:time_offset_ms` schema key on the
    /// standalone IPC shape, `TILE_META.t0` in the layer frame.
    min_start_time: Option<i64>,
    /// `TILE_META.st` — `None` when `start_time` is absolute `Int64`.
    start_time_form: Option<StartTimeForm>,
    /// `TILE_META.et` — `None` when `end_time` is absolute `Int64`.
    end_time_form: Option<EndTimeForm>,
    /// `(origin_ms, step_ms)` of a delta-encoded `vertex_time` column.
    vertex_time_encoding: Option<(i64, u32)>,
    /// `TILE_META.vtf` — TB-11 extension 2's feature-anchored step. Mutually
    /// exclusive with [`Self::vertex_time_encoding`].
    vertex_time_feature_step: Option<u32>,
    /// The `stt:vertex_value_buckets` schema key on the standalone IPC shape,
    /// `TILE_META.vb` in the layer frame.
    vertex_value_buckets: Option<u32>,
    /// `TILE_META.vq` — the per-column `(o, s)` affine of every per-vertex
    /// value column that shipped a `UInt16` leaf. Empty ⇒ raw `List<Float32>`.
    vertex_value_quant: BTreeMap<String, (f64, f64)>,
    has_triangles: bool,
    /// TB-4: whether the PROPS stream's `DictionaryBatch` messages are provably
    /// dataset-constant and so belong in the schema TEMPLATE rather than the
    /// per-tile tail. Always [`DictHoist::Tail`] without pins, which is what
    /// keeps an unpinned encode byte-identical to a pre-M2 one.
    props_dict_hoist: DictHoist,
}

/// Pick the compact form of a layer's two feature-time columns from that
/// layer's own data, and recorded in `TILE_META.st` / `.et`.
///
/// `(None, None)` is the historical absolute-`Int64` pair. An EMPTY layer
/// always gets it: there is no `t0` to anchor a u32 offset against, and an
/// "every duration is zero" verdict over zero features would fork the layer's
/// schema template for empty tiles alone (each distinct core shape costs a
/// `manifest.schemas` entry).
fn choose_time_forms(
    starts: &[i64],
    ends: &[i64],
    t0: Option<i64>,
) -> (Option<StartTimeForm>, Option<EndTimeForm>) {
    let Some(t0) = t0 else {
        return (None, None);
    };
    // `checked_sub` everywhere: a layer straddling the i64 extremes must fall
    // back to the absolute columns rather than wrap into plausible garbage.
    let fits = |d: i64| (0..=u32::MAX as i64).contains(&d);
    let start = starts
        .iter()
        .all(|&s| s.checked_sub(t0).is_some_and(fits))
        .then_some(StartTimeForm::U32Offset);

    // Durations are measured against each feature's OWN start, not `t0`, so a
    // long-lived feature inside a wide tile still compacts.
    let mut all_zero = true;
    let mut all_fit = true;
    for (&e, &s) in ends.iter().zip(starts) {
        match e.checked_sub(s) {
            Some(d) => {
                all_zero &= d == 0;
                all_fit &= fits(d);
            }
            None => {
                all_zero = false;
                all_fit = false;
            }
        }
        if !all_fit {
            break; // `all_zero` implies `all_fit`, so both are settled.
        }
    }
    let end = if all_zero {
        Some(EndTimeForm::Zero)
    } else if all_fit {
        Some(EndTimeForm::Dur32)
    } else {
        None
    };
    (start, end)
}

/// Build the Arrow fields + columns for one layer — the version-independent
/// front of the encoder (validation, quantization, vector grouping, the
/// point-elevation fold, vertex-time encoding).
///
/// `frame_only` is passed by the caller rather than read off `cfg` because
/// each of its encodings is only sound on the FRAME path: they are
/// discriminated by `TILE_META` keys (`st`/`et`/`vq`), which only a frame
/// carries. See [`FrameOnlyEncodings`].
fn build_layer_parts(
    layer: &ColumnarLayer,
    cfg: &EncoderConfig,
    frame_only: FrameOnlyEncodings,
) -> Result<LayerParts> {
    let FrameOnlyEncodings {
        compact_times,
        quantize_vertex_values,
    } = frame_only;
    layer.validate()?;
    let n = layer.feature_count();

    let mut fields: Vec<Arc<Field>> = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    fields.push(Arc::new(Field::new("id", DataType::UInt64, false)));
    columns.push(Arc::new(UInt64Array::from(layer.feature_ids.clone())));

    // --- feature time columns (compact by default; see `choose_time_forms`) --
    let min_start_time = layer.start_times.iter().copied().min();
    let (start_time_form, end_time_form) = if compact_times {
        choose_time_forms(&layer.start_times, &layer.end_times, min_start_time)
    } else {
        (None, None)
    };

    match start_time_form {
        Some(StartTimeForm::U32Offset) => {
            let t0 = min_start_time.expect("a u32 start form implies a t0 anchor");
            let offsets: Vec<u32> = layer
                .start_times
                .iter()
                .map(|&s| (s - t0) as u32) // range-checked by choose_time_forms
                .collect();
            fields.push(Arc::new(Field::new("start_time", DataType::UInt32, false)));
            columns.push(Arc::new(UInt32Array::from(offsets)));
        }
        None => {
            fields.push(Arc::new(Field::new("start_time", DataType::Int64, false)));
            columns.push(Arc::new(Int64Array::from(layer.start_times.clone())));
        }
    }

    match end_time_form {
        // `end == start` for every feature: the column carries no information
        // at all, so it is omitted and the reader synthesizes it back.
        Some(EndTimeForm::Zero) => {}
        Some(EndTimeForm::Dur32) => {
            let durations: Vec<u32> = layer
                .end_times
                .iter()
                .zip(&layer.start_times)
                .map(|(&e, &s)| (e - s) as u32) // range-checked above
                .collect();
            fields.push(Arc::new(Field::new("end_time", DataType::UInt32, false)));
            columns.push(Arc::new(UInt32Array::from(durations)));
        }
        None => {
            fields.push(Arc::new(Field::new("end_time", DataType::Int64, false)));
            columns.push(Arc::new(Int64Array::from(layer.end_times.clone())));
        }
    }

    // 3D POINT geometry: fold the configured numeric column into the geometry's
    // 3rd coordinate, so the tile ships true 3D points the renderer binds
    // zero-copy (no per-point pad). The column is then dropped from properties.
    let elev_col = cfg.point_elevation_column.clone();
    let point_elev: Option<Vec<f64>> =
        if !elev_col.is_empty() && matches!(layer.geometry, GeometryColumn::Point(_)) {
            layer.properties.iter().find_map(|(name, col)| match col {
                PropertyColumn::Numeric(v) if name == &elev_col => {
                    let n = layer.feature_count();
                    let mut out = vec![0.0f64; n];
                    for (i, x) in v.iter().enumerate() {
                        if let Some(val) = x {
                            out[i] = *val;
                        }
                    }
                    Some(out)
                }
                _ => None,
            })
        } else {
            None
        };
    let elev_consumed = point_elev.is_some();

    // Geometry column carries the GeoArrow extension name in field metadata.
    // The precision floor is enforced HERE (where the meters value is
    // consumed), so every entry point — globals, explicit EncoderConfig, a
    // server's per-request config — hits the same guard.
    if let Some(m) = cfg.quantize_coords_m {
        validate_quantize_coords_m(m)?;
    }
    let quant = cfg.quantize_coords_m.and_then(|m| {
        if point_elev.is_some() {
            world_grid_affine_3d(m)
        } else {
            world_grid_affine(m)
        }
    });
    let geom_array =
        build_geometry_array_q(&layer.geometry, quant.as_ref(), point_elev.as_deref())?;
    // Assemble field metadata in a BTreeMap so the key set is emitted in a
    // deterministic (lexicographic) order regardless of insertion order. Arrow
    // ≥59 serializes IPC schema metadata in sorted order, so building from a
    // sorted source makes the raw metadata-region bytes byte-reproducible across
    // runs (guarded by `same_tile_encodes_byte_identically` in
    // reproducible_build.rs). A HashMap here would reintroduce per-run
    // iteration order and break that.
    let mut geom_meta = BTreeMap::new();
    geom_meta.insert(
        GEOARROW_EXT_KEY.to_string(),
        layer.geometry.geoarrow_name().to_string(),
    );
    match &quant {
        // A quantized tile's `xy` leaf is i32 grid indices, not Float64 lon/lat,
        // so the GeoArrow CRS doesn't apply — swap it for the reconstruction
        // affine (whose presence is the reader's quantization signal). Field
        // metadata is assembled in a BTreeMap (deterministic key order); Arrow
        // ≥59 serializes IPC schema metadata in sorted order, so the raw
        // metadata-region bytes are byte-reproducible across runs (guarded by
        // the `same_tile_encodes_byte_identically` test in reproducible_build.rs).
        Some(q) => {
            geom_meta.insert(STT_QUANT_META_KEY.to_string(), q.to_json());
        }
        // Advertise the CRS so the tile is self-describing to GeoArrow consumers.
        None => {
            geom_meta.insert(
                GEOARROW_EXT_META_KEY.to_string(),
                GEOARROW_CRS_METADATA.to_string(),
            );
        }
    }
    fields.push(Arc::new(
        Field::new("geometry", geom_array.data_type().clone(), false)
            .with_metadata(geom_meta.into_iter().collect()),
    ));
    columns.push(geom_array);

    // Track per-layer vertex-time encoding so the schema metadata (set
    // below) records the origin/step needed for the u16-delta reader path.
    let mut vertex_time_encoding: Option<(i64, u32)> = None;
    let mut vertex_time_feature_step: Option<u32> = None;
    if let Some(vt_col) = build_vertex_time_array(
        &layer.vertex_times,
        n,
        cfg.vertex_time_max_step_ms,
        &layer.start_times,
    ) {
        fields.push(Arc::new(Field::new(
            "vertex_time",
            vt_col.array.data_type().clone(),
            true,
        )));
        columns.push(vt_col.array);
        vertex_time_encoding = vt_col.encoding;
        vertex_time_feature_step = vt_col.feature_anchored_step;
    }

    // Optional per-vertex scalar column (e.g. sea-surface temperature),
    // aligned 1:1 with the geometry vertices like `vertex_time`. With
    // `quantize_vertex_values` the leaf is a `UInt16` under a per-column
    // range-adaptive affine recorded in `TILE_META.vq`.
    let mut vertex_value_quant: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    if let Some(vv) = build_vertex_value_array(&layer.vertex_values, n, quantize_vertex_values) {
        fields.push(Arc::new(Field::new(
            "vertex_value",
            vv.array.data_type().clone(),
            true,
        )));
        columns.push(vv.array);
        if let Some(affine) = vv.quant {
            vertex_value_quant.insert("vertex_value".to_string(), affine);
        }
    }

    // Optional per-vertex × per-bucket value matrix (static-geometry overview
    // animation). Reuses the `vertex_value` List<Float32> encoding — each row
    // is just longer (vertex_count * num_buckets, vertex-major). num_buckets is
    // recovered from the per-feature vertex count and recorded in schema meta.
    let mut vertex_value_buckets: Option<u32> = None;
    if let Some(vm) =
        build_vertex_value_array(&layer.vertex_value_matrix, n, quantize_vertex_values)
    {
        fields.push(Arc::new(Field::new(
            "vertex_value_matrix",
            vm.array.data_type().clone(),
            true,
        )));
        columns.push(vm.array);
        if let Some(affine) = vm.quant {
            vertex_value_quant.insert("vertex_value_matrix".to_string(), affine);
        }
        vertex_value_buckets = layer
            .vertex_value_matrix
            .as_ref()
            .and_then(|vm| infer_vertex_value_buckets(vm, &layer.geometry));
    }

    // Pre-baked triangle indices (MLT-style). Only emitted for polygon
    // layers; for any other geometry kind the column is silently dropped so
    // an over-eager builder can't poison a point/line layer with stale data.
    let has_triangles = matches!(layer.geometry, GeometryColumn::Polygon(_))
        && layer
            .triangles
            .as_ref()
            .map(|t| t.iter().any(|f| !f.is_empty()))
            .unwrap_or(false);
    if has_triangles {
        let tri = layer.triangles.as_ref().unwrap();
        // Indices are feature-LOCAL (see the field doc above), so they're
        // almost always well under 65,536 even for large layers. Mirrors
        // build_vertex_time_array's width-selection: scan once, use the
        // narrower UInt16 (half the bytes) when every index fits, UInt32
        // otherwise. The Arrow field type is derived from the array below,
        // so this is fully self-describing — the TS decoder branches on the
        // runtime child-array type exactly like it already does for
        // vertex_time's UInt16-delta vs Int64-absolute split.
        let max_index = tri.iter().flatten().copied().max().unwrap_or(0);
        let array: ArrayRef = if max_index <= u16::MAX as u32 {
            let mut builder = ListBuilder::new(UInt16Builder::new());
            for feature in tri {
                for &idx in feature {
                    builder.values().append_value(idx as u16);
                }
                // Always append a (possibly empty) list — readers expect one
                // entry per feature.
                builder.append(true);
            }
            Arc::new(builder.finish())
        } else {
            let mut builder = ListBuilder::new(UInt32Builder::new());
            for feature in tri {
                for &idx in feature {
                    builder.values().append_value(idx);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        };
        fields.push(Arc::new(Field::new(
            "triangles",
            array.data_type().clone(),
            false,
        )));
        columns.push(array);
    }

    // Optional multi-part polygon boundaries (see `build_part_offsets_array`).
    // LAST among the reserved columns deliberately: every layer shape that
    // existed before it keeps its exact column order, so no already-published
    // schema template changes and no decoder that indexes positionally shifts.
    // Dropped for non-polygon layers exactly like `triangles`.
    if let Some(part_array) = build_part_offsets_array(&layer.polygon_parts, &layer.geometry, n) {
        fields.push(Arc::new(Field::new(
            "part_offsets",
            part_array.data_type().clone(),
            false,
        )));
        columns.push(part_array);
    }

    // Every field pushed so far is a reserved column — the frame's CORE batch.
    // Property columns (the frame's PROPS batch) follow.
    let reserved_len = fields.len();

    // Fuse configured scalar columns into GPU-ready interleaved Vector columns
    // (e.g. qx/qy/qz/qw → one FixedSizeList<f32,4>). Runs BEFORE the quantize
    // loop so grouped components are written as the raw vector, not individually
    // quantized. No groups configured ⇒ iterate `layer.properties` with no clone.
    let grouped =
        group_vector_properties(&layer.properties, layer.feature_count(), &cfg.vector_groups);
    let props_iter: &[(String, PropertyColumn)] = grouped.as_deref().unwrap_or(&layer.properties);
    // TB-4 bookkeeping: how many PROPS columns emit an Arrow dictionary, and how
    // many of those were built against a pinned dataset-global list. The stream
    // is hoistable only when the two agree and are non-zero.
    let mut dict_columns = 0usize;
    let mut pinned_dict_columns = 0usize;
    // Whole-stream pre-pass: with any pin in play, an unpinned categorical
    // column may not mint a tile-local dictionary. See
    // [`unpinned_categoricals_take_utf8`] for the shape this designs out.
    let demote_unpinned_categoricals = unpinned_categoricals_take_utf8(cfg, props_iter);
    for (name, col) in props_iter {
        // The point-elevation column now lives in the geometry's 3rd coordinate;
        // don't also emit it as a scalar property.
        if elev_consumed && name == &elev_col {
            continue;
        }
        match col {
            PropertyColumn::Numeric(values) => {
                // Opt-in: a numeric property named in the build-global
                // quantization map ships as fixed-point ints + a per-column
                // affine in field metadata (the reader reconstructs Float64).
                // For a LiDAR `z` column this is the single largest size lever
                // after `id` — a raw Float64 elevation barely compresses, while
                // the i16 grid is both smaller and far more compressible.
                //
                // TB-2: the decision goes through the one dispatch that can see
                // `cfg.global_pins`, so with pins attached the affine is the
                // column's DATASET-global one and the same source value decodes
                // to the same number in every tile. The precedence it applies is
                // the incumbent's (explicit `--quantize-attr` wins, then auto,
                // then Float64), and with `global_pins == None` — `--single-pass`,
                // a one-shot external caller, a tile server with no pin sidecar
                // — every branch of it is the per-tile function this line used to
                // call directly, with the same arguments.
                let quantized = build_quantized_numeric_for_column(name, values, cfg)?;
                match quantized {
                    Some((array, affine_json)) => {
                        let mut m = HashMap::new();
                        m.insert(STT_QUANT_ATTR_META_KEY.to_string(), affine_json);
                        fields.push(Arc::new(
                            Field::new(name, array.data_type().clone(), true).with_metadata(m),
                        ));
                        columns.push(array);
                    }
                    None => {
                        fields.push(Arc::new(Field::new(name, DataType::Float64, true)));
                        columns.push(Arc::new(Float64Array::from(values.clone())));
                    }
                }
            }
            PropertyColumn::Categorical(values) => {
                // TB-3: one column, ONE Arrow type, everywhere. With a pin the
                // verdict is a function of the dataset's domain and the tile's
                // own values cannot reach it; without one, the incumbent
                // per-tile comparison below is unchanged. Exact strings are
                // preserved on every branch.
                let dict_type =
                    || DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8));
                let push_utf8 = |fields: &mut Vec<Arc<Field>>, columns: &mut Vec<ArrayRef>| {
                    fields.push(Arc::new(Field::new(name, DataType::Utf8, true)));
                    columns.push(Arc::new(StringArray::from_iter(
                        values.iter().map(|value| value.as_deref()),
                    )));
                };
                match categorical_plan(cfg, name) {
                    // Pinned dictionary: the FULL global list in this tile too,
                    // whatever subset of it the tile's rows use.
                    CategoricalPlan::Dictionary(pinned) => {
                        let array = build_pinned_dictionary_column(name, values, pinned)?;
                        fields.push(Arc::new(Field::new(name, dict_type(), true)));
                        columns.push(array);
                        dict_columns += 1;
                        pinned_dict_columns += 1;
                    }
                    // Pinned Utf8: no dictionary for this column, in any tile.
                    CategoricalPlan::Utf8 => push_utf8(&mut fields, &mut columns),
                    CategoricalPlan::PerTile => {
                        // Dictionary batches carry meaningful fixed IPC
                        // overhead. On sparse tiles (or high-cardinality
                        // strings) plain Utf8 is smaller and avoids repeating a
                        // tiny dictionary batch in every tile. Choose from the
                        // actual tile values, preserving the exact strings
                        // either way. A >UInt16 cardinality column falls back to
                        // Utf8 instead of failing/dropping rows.
                        //
                        // ...unless a SIBLING column in this stream is pinned,
                        // in which case a tile-local dictionary here would drag
                        // the sibling's hoisted list back into every tile's
                        // tail. Then the answer is `Utf8`, unconditionally.
                        let dictionary = if demote_unpinned_categoricals {
                            None
                        } else {
                            build_dictionary_indices(values).ok()
                        };
                        let use_dictionary = dictionary.as_ref().is_some_and(|(_, categories)| {
                            categorical_dictionary_is_smaller(values, categories)
                        });
                        if use_dictionary {
                            let (indices, categories) = dictionary.expect("checked above");
                            fields.push(Arc::new(Field::new(name, dict_type(), true)));

                            let value_array: ArrayRef = Arc::new(StringArray::from(
                                categories
                                    .iter()
                                    .map(|s| Some(s.as_str()))
                                    .collect::<Vec<_>>(),
                            ));
                            let key_array = UInt16Array::from(indices);
                            let dict =
                                DictionaryArray::<UInt16Type>::try_new(key_array, value_array)
                                    .map_err(|e| {
                                        Error::Other(format!("dictionary build failed: {e}"))
                                    })?;
                            columns.push(Arc::new(dict));
                            // Tile-local contents: this stream is NOT hoistable.
                            dict_columns += 1;
                        } else {
                            push_utf8(&mut fields, &mut columns);
                        }
                    }
                }
            }
            PropertyColumn::Vector {
                width,
                elem,
                values,
            } => {
                // Interleaved GPU-ready vector → FixedSizeList<leaf, width>. The
                // child buffer is the flattened row-major run, so the TS decoder
                // hands `child.values.subarray(...)` straight to deck.gl with no
                // per-point re-interleave. Non-null leaf (producer encodes a
                // missing feature as a zero/identity vector).
                let (child, child_dt): (ArrayRef, DataType) = match elem {
                    VectorElem::F32 => (
                        Arc::new(Float32Array::from(values.clone())),
                        DataType::Float32,
                    ),
                    VectorElem::U8 => {
                        let bytes: Vec<u8> = values
                            .iter()
                            .map(|v| v.round().clamp(0.0, 255.0) as u8)
                            .collect();
                        (Arc::new(UInt8Array::from(bytes)), DataType::UInt8)
                    }
                };
                let item_field = Arc::new(Field::new("item", child_dt, false));
                let width_i32 = i32::try_from(*width).map_err(|_| {
                    Error::Other(format!(
                        "vector property '{name}' has width {width}, which exceeds the \
                         FixedSizeList i32 size limit"
                    ))
                })?;
                let fsl = FixedSizeListArray::new(item_field, width_i32, child, None);
                fields.push(Arc::new(Field::new(name, fsl.data_type().clone(), true)));
                columns.push(Arc::new(fsl));
            }
        }
    }

    Ok(LayerParts {
        fields,
        columns,
        reserved_len,
        layer_name: layer.name.clone(),
        geometry_name: layer.geometry.geoarrow_name(),
        // The layer's minimum feature start-time (integer Unix ms), so the TS
        // decoder can skip its client-side min-scan over the whole start-time
        // column and relativize times against this value directly. Mirrors
        // exactly what the decoder computes (the min of the `start_time`
        // column); only emitted when a start-time column is present. See
        // packages/core/src/tile.ts. With a `st: "u32"` start column it is
        // LOAD-BEARING, not an optimization — it is the offsets' anchor.
        min_start_time,
        start_time_form,
        end_time_form,
        vertex_time_encoding,
        vertex_time_feature_step,
        vertex_value_buckets,
        vertex_value_quant,
        has_triangles,
        // All-or-nothing across the whole PROPS stream: hoist only when there
        // IS a dictionary and EVERY dictionary in the stream came from a pin.
        // A single unpinned dictionary column keeps the incumbent split, so a
        // mixed layer degrades to "no hoist" rather than to a per-tile template.
        props_dict_hoist: if dict_columns > 0 && dict_columns == pinned_dict_columns {
            DictHoist::Template
        } else {
            DictHoist::Tail
        },
    })
}

/// Arrow IPC buffer alignment (bytes) every STT tile is written at.
///
/// arrow-rs' `IpcWriteOptions::default()` uses **64**, which is a SIMD
/// *recommendation*; the Arrow IPC spec only requires 8. At 64, every buffer in
/// every tile is padded to a 64-byte boundary — measured across the shipped
/// fleet that is 19–39% of RAW IPC bytes (hurricanes −38.8%, earthquakes-v2
/// −32.9%, storm-cells −30.6%, nyc-taxi-points −19.4% when re-serialized at 8).
/// Compressed the delta is only ~2–3%, but the UNCOMPRESSED size is what drives
/// reader allocation, the client memory budget and eviction (earthquakes-v2:
/// 73 MB on the wire, 420 MB decoded).
///
/// 8 is the floor both reference readers need, and they get it: the TS reader's
/// zero-copy `subarray` paths (`readCoordRun` over unquantized Float64, the
/// Float32 numeric-property passthrough) need each buffer's byte offset to be a
/// multiple of its element width, arrow-rs pads the schema message to the same
/// alignment (so a spliced template+tail stream keeps the body's alignment),
/// and the frame already pads every section to [`FRAME_ALIGN`] = 8.
pub(crate) const IPC_BUFFER_ALIGNMENT: usize = 8;

/// The IPC write options every STT tile stream is written with — see
/// [`IPC_BUFFER_ALIGNMENT`]. `write_legacy_ipc_format = false` +
/// `MetadataVersion::V5` are arrow-rs' own defaults, restated explicitly so
/// this constructor cannot silently drift with the crate's default.
fn ipc_write_options() -> Result<IpcWriteOptions> {
    IpcWriteOptions::try_new(IPC_BUFFER_ALIGNMENT, false, MetadataVersion::V5)
        .map_err(|e| Error::Other(format!("Arrow IPC write options rejected: {e}")))
}

/// Serialize one `(schema, columns)` batch as an Arrow IPC stream.
fn write_ipc_stream(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<Vec<u8>> {
    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Error::Other(format!("failed to build tile RecordBatch: {e}")))?;
    let mut buf = Vec::new();
    {
        let mut writer =
            StreamWriter::try_new_with_options(&mut buf, &schema, ipc_write_options()?)
                .map_err(|e| Error::Other(format!("Arrow IPC writer init failed: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| Error::Other(format!("Arrow IPC write failed: {e}")))?;
        writer
            .finish()
            .map_err(|e| Error::Other(format!("Arrow IPC finish failed: {e}")))?;
    }
    Ok(buf)
}

/// Self-describing assembly: one schema carrying EVERY metadata key
/// (dataset-constant AND per-tile-varying), serialized as a single IPC stream.
/// This is the STANDALONE-LAYER shape (`encode_layer`), not the tile frame —
/// the frame splits schema and batch into sections instead.
fn assemble_layer_ipc_self_describing(parts: LayerParts) -> Result<Vec<u8>> {
    debug_assert!(
        parts.start_time_form.is_none() && parts.end_time_form.is_none(),
        "the standalone layer shape has no TILE_META to carry st/et"
    );
    debug_assert!(
        parts.vertex_value_quant.is_empty(),
        "the standalone layer shape has no TILE_META to carry vq"
    );
    // Schema-level metadata records the layer name and geometry kind so a
    // reader does not have to inspect the geometry column. When the
    // vertex_time column is u16-delta encoded we add `origin_ms` and
    // `step_ms` so the reader can reconstruct absolute timestamps as
    // `origin + delta * step`.
    //
    // Built in a BTreeMap so the key set is assembled in deterministic
    // (lexicographic) order — this encoder contributes no ordering
    // non-determinism. Arrow ≥59 serializes IPC schema metadata in sorted order,
    // so the raw metadata-region bytes of two identical tiles are identical
    // across runs, which is what content-addressed pack dedup rests on. A
    // HashMap here would reintroduce per-run iteration order and break it.
    let mut schema_meta: BTreeMap<String, String> = BTreeMap::new();
    schema_meta.insert("stt:layer".to_string(), parts.layer_name.clone());
    schema_meta.insert("stt:geometry".to_string(), parts.geometry_name.to_string());
    if let Some(min_start) = parts.min_start_time {
        schema_meta.insert(TIME_OFFSET_MS_KEY.to_string(), min_start.to_string());
    }
    if let Some((origin, step)) = parts.vertex_time_encoding {
        schema_meta.insert(VERTEX_TIME_ORIGIN_KEY.to_string(), origin.to_string());
        schema_meta.insert(VERTEX_TIME_STEP_KEY.to_string(), step.to_string());
    }
    // TB-11 extension 2: the feature-anchored tier records only a step — the
    // origin is each feature's own `start_time`, which already ships in CORE.
    // A DISTINCT key, never a reshaped `vt`: a reader that knows only `vt`
    // must not silently read these deltas against a layer origin it invented.
    if let Some(step) = parts.vertex_time_feature_step {
        schema_meta.insert(VERTEX_TIME_FEATURE_STEP_KEY.to_string(), step.to_string());
    }
    if let Some(buckets) = parts.vertex_value_buckets {
        schema_meta.insert(VERTEX_VALUE_BUCKETS_KEY.to_string(), buckets.to_string());
    }
    if parts.has_triangles {
        schema_meta.insert(TRIANGLES_METADATA_KEY.to_string(), "true".to_string());
    }
    let schema =
        Arc::new(Schema::new(parts.fields).with_metadata(schema_meta.into_iter().collect()));
    write_ipc_stream(schema, parts.columns)
}
// ----------------------------------------------------------------------------
// Layer-frame encoding (template extraction + TILE_META + core/props split)
// ----------------------------------------------------------------------------

/// One layer, encoded for the layer frame: templates split off, tails verbatim,
/// per-tile metadata canonicalized into the TILE_META JSON.
pub(crate) struct EncodedLayerV2 {
    pub(crate) core_template: Vec<u8>,
    pub(crate) core_tail: Vec<u8>,
    /// `(template, tail)` of the PROPS batch; `None` when the layer has no
    /// property columns (`ref_kind_props = 2`).
    pub(crate) props: Option<(Vec<u8>, Vec<u8>)>,
    pub(crate) tile_meta_json: String,
    /// This layer used the TB-11 feature-anchored vertex-time form. Bubbles up
    /// to [`EncodeObservations`] so the writer can declare the capability.
    pub(crate) feature_anchored_vertex_times: bool,
}

/// Locate the end of the leading schema message by walking the Arrow IPC
/// encapsulated framing: `[0xFFFFFFFF][i32 metadata_len][flatbuffer (padded)]`
/// with the schema's `bodyLength == 0` (spike-proven boundary — deterministic,
/// no re-serialization; `metadata_len` already includes the flatbuffer's
/// padding). Everything before the boundary is the template, everything after
/// is the tail (dictionary batches + record batch + EOS).
fn split_ipc_at_schema(ipc: &[u8]) -> Result<usize> {
    if ipc.len() < 8 || ipc[0..4] != [0xFF, 0xFF, 0xFF, 0xFF] {
        return Err(Error::Other(
            "layer IPC stream does not start with an encapsulated message".into(),
        ));
    }
    let meta_len = i32::from_le_bytes(ipc[4..8].try_into().expect("4 bytes"));
    if meta_len <= 0 {
        return Err(Error::Other(
            "layer IPC stream starts with an end-of-stream marker".into(),
        ));
    }
    let boundary = 8usize
        .checked_add(meta_len as usize)
        .filter(|b| *b <= ipc.len())
        .ok_or_else(|| Error::Other("layer IPC schema message overruns the stream".into()))?;
    let msg = root_as_message(&ipc[8..boundary])
        .map_err(|e| Error::Other(format!("layer IPC schema flatbuffer parse failed: {e}")))?;
    if msg.header_type() != MessageHeader::Schema {
        return Err(Error::Other(format!(
            "layer IPC stream must start with a Schema message, got {:?}",
            msg.header_type()
        )));
    }
    if msg.bodyLength() != 0 {
        return Err(Error::Other(
            "layer IPC schema message unexpectedly carries a body".into(),
        ));
    }
    Ok(boundary)
}

/// Locate the RecordBatch boundary of an Arrow IPC stream — i.e. the end of the
/// leading schema message AND of every `DictionaryBatch` message that follows
/// it. The template-hoisting sibling of [`split_ipc_at_schema`] (TB-4).
///
/// Everything before the returned offset (Schema + DictionaryBatch messages) is
/// the template; everything after (RecordBatch + EOS) is the tail. That is only
/// a legal cut when the dictionary contents are dataset-constant, which is why
/// the caller gates it on [`DictHoist::Template`] — see [`build_layer_parts`].
///
/// The walk is exact, not heuristic: each message advances by
/// `8 + metadata_len + bodyLength`, and arrow-rs guarantees `bodyLength` is
/// already padded to the write alignment (`write_message` rejects an unaligned
/// body outright), so the next message must begin exactly at that offset. Rather
/// than assume it, every step re-checks the `0xFFFFFFFF` continuation marker and
/// errors loudly on drift — a silently mis-cut template splices into a stream
/// arrow-rs decodes as EMPTY, which is the one failure mode that must never be
/// quiet.
///
/// A stream whose schema is followed immediately by its record batch (no
/// dictionaries at all) returns exactly what [`split_ipc_at_schema`] returns, so
/// the two agree wherever both are defined.
pub(crate) fn split_ipc_after_dictionaries(ipc: &[u8]) -> Result<usize> {
    let mut pos = split_ipc_at_schema(ipc)?;
    loop {
        if pos.checked_add(8).is_none_or(|end| end > ipc.len()) {
            return Err(Error::Other(
                "layer IPC stream ends after its schema/dictionary messages with no record \
                 batch"
                    .into(),
            ));
        }
        if ipc[pos..pos + 4] != [0xFF, 0xFF, 0xFF, 0xFF] {
            return Err(Error::Other(format!(
                "layer IPC stream: the message at offset {pos} does not start with the \
                 0xFFFFFFFF continuation marker (dictionary framing walk lost alignment)"
            )));
        }
        let meta_len = i32::from_le_bytes(ipc[pos + 4..pos + 8].try_into().expect("4 bytes"));
        if meta_len <= 0 {
            // End-of-stream marker. Nothing but dictionaries preceded it, so
            // the cut lands here and the tail is the bare EOS — which still
            // begins with the continuation marker `splice_decode` demands.
            return Ok(pos);
        }
        let header_end = (meta_len as usize)
            .checked_add(8)
            .and_then(|n| pos.checked_add(n))
            .filter(|end| *end <= ipc.len())
            .ok_or_else(|| Error::Other("layer IPC message metadata overruns the stream".into()))?;
        let msg = root_as_message(&ipc[pos + 8..header_end])
            .map_err(|e| Error::Other(format!("layer IPC message flatbuffer parse failed: {e}")))?;
        if msg.header_type() != MessageHeader::DictionaryBatch {
            // The first non-dictionary message (the RecordBatch) starts the tail.
            return Ok(pos);
        }
        let body = usize::try_from(msg.bodyLength()).map_err(|_| {
            Error::Other("layer IPC dictionary message declares a negative body length".into())
        })?;
        pos = header_end
            .checked_add(body)
            .filter(|next| *next <= ipc.len())
            .ok_or_else(|| {
                Error::Other("layer IPC dictionary message body overruns the stream".into())
            })?;
    }
}

/// Layer-frame row order: stable-sort rows by `start_time` at ENCODE time —
/// after the tiler assigned feature ids — so ids stay order-independent and
/// `TILE_META.sorted` can be declared. Returns the layer unchanged (borrowed)
/// when its rows are already non-decreasing, which also makes the sort a no-op
/// for pre-sorted producers.
pub(crate) fn sort_rows_by_start_time(layer: &ColumnarLayer) -> Cow<'_, ColumnarLayer> {
    if layer.start_times.windows(2).all(|w| w[0] <= w[1]) {
        return Cow::Borrowed(layer);
    }
    let mut idx: Vec<usize> = (0..layer.feature_count()).collect();
    idx.sort_by_key(|&i| layer.start_times[i]); // stable

    // Per-feature Option<Vec<Vec<_>>> columns tolerate inner vecs shorter
    // than the feature count at encode (`vt.get(i)` ⇒ null list); the
    // permutation preserves that semantic via `.get(..).unwrap_or_default()`.
    fn permute_nested<T: Clone>(v: &Option<Vec<Vec<T>>>, idx: &[usize]) -> Option<Vec<Vec<T>>> {
        v.as_ref().map(|v| {
            idx.iter()
                .map(|&i| v.get(i).cloned().unwrap_or_default())
                .collect()
        })
    }

    let geometry = match &layer.geometry {
        GeometryColumn::Point(v) => GeometryColumn::Point(idx.iter().map(|&i| v[i]).collect()),
        GeometryColumn::LineString(v) => {
            GeometryColumn::LineString(idx.iter().map(|&i| v[i].clone()).collect())
        }
        GeometryColumn::Polygon(v) => {
            GeometryColumn::Polygon(idx.iter().map(|&i| v[i].clone()).collect())
        }
    };
    let properties = layer
        .properties
        .iter()
        .map(|(name, col)| {
            let col = match col {
                PropertyColumn::Numeric(v) => {
                    PropertyColumn::Numeric(idx.iter().map(|&i| v[i]).collect())
                }
                PropertyColumn::Categorical(v) => {
                    PropertyColumn::Categorical(idx.iter().map(|&i| v[i].clone()).collect())
                }
                PropertyColumn::Vector {
                    width,
                    elem,
                    values,
                } => PropertyColumn::Vector {
                    width: *width,
                    elem: *elem,
                    values: idx
                        .iter()
                        .flat_map(|&i| values[i * width..(i + 1) * width].iter().copied())
                        .collect(),
                },
            };
            (name.clone(), col)
        })
        .collect();

    Cow::Owned(ColumnarLayer {
        name: layer.name.clone(),
        feature_ids: idx.iter().map(|&i| layer.feature_ids[i]).collect(),
        start_times: idx.iter().map(|&i| layer.start_times[i]).collect(),
        end_times: idx.iter().map(|&i| layer.end_times[i]).collect(),
        geometry,
        vertex_times: permute_nested(&layer.vertex_times, &idx),
        vertex_values: permute_nested(&layer.vertex_values, &idx),
        vertex_value_matrix: permute_nested(&layer.vertex_value_matrix, &idx),
        triangles: permute_nested(&layer.triangles, &idx),
        // Per-feature part boundaries are relative to the feature's OWN first
        // ring, so a row permutation moves them verbatim (no rebasing) —
        // exactly like `triangles`, whose indices are feature-local too.
        polygon_parts: permute_nested(&layer.polygon_parts, &idx),
        properties,
    })
}

/// Encode one layer for the layer frame: sort rows, build the shared parts, move
/// per-tile-varying metadata into TILE_META, split the CORE and PROPS IPC
/// streams at their schema boundaries.
///
/// Metadata placement: per-tile-varying and hoisted into
/// TILE_META are exactly `stt:qa` (property fields) and the schema-level
/// `stt:time_offset_ms` / `stt:vertex_time_origin_ms` / `stt:vertex_time_step_ms`
/// / `stt:vertex_value_buckets`, plus the keys with no Arrow schema-metadata
/// counterpart at all (`st`/`et`, `vq`). Dataset-constant and template-resident:
/// `ARROW:extension:name` / `ARROW:extension:metadata` (CRS), `stt:quant`
/// (world-anchored), `stt:layer`, `stt:geometry`, `stt:has_triangles`.
pub(crate) fn encode_layer_v2_parts(
    layer: &ColumnarLayer,
    cfg: &EncoderConfig,
) -> Result<EncodedLayerV2> {
    // Validate BEFORE the row sort: `sort_rows_by_start_time` indexes every
    // column by the feature count, so a length-inconsistent layer would panic
    // there instead of returning `validate`'s descriptive error.
    // (`build_layer_parts` re-validates the sorted layer; the check is pure.)
    layer.validate()?;
    let sorted = sort_rows_by_start_time(layer);
    let parts = build_layer_parts(
        &sorted,
        cfg,
        FrameOnlyEncodings {
            compact_times: cfg.compact_times,
            quantize_vertex_values: cfg.quantize_vertex_values,
        },
    )?;

    // Strip `stt:qa` off the property fields into the TILE_META `qa` map —
    // spike-proven to leave the batch tail bytes byte-identical (only the
    // schema message changes). The affine JSON round-trips exactly
    // (`AttrQuant::to_json` is a pure function of `(o, s)`), so decode
    // re-injects byte-identical field metadata.
    let mut qa: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    let mut props_fields: Vec<Arc<Field>> =
        Vec::with_capacity(parts.fields.len() - parts.reserved_len);
    for field in &parts.fields[parts.reserved_len..] {
        let mut meta = field.metadata().clone();
        if let Some(json) = meta.remove(STT_QUANT_ATTR_META_KEY) {
            let affine = AttrQuant::from_json(&json).ok_or_else(|| {
                Error::Other(format!(
                    "property '{}' carries an unparseable {STT_QUANT_ATTR_META_KEY} affine",
                    field.name()
                ))
            })?;
            qa.insert(field.name().clone(), (affine.o, affine.s));
            props_fields.push(Arc::new(field.as_ref().clone().with_metadata(meta)));
        } else {
            props_fields.push(field.clone());
        }
    }

    let tile_meta = TileMeta {
        et: parts.end_time_form,
        qa: (!qa.is_empty()).then_some(qa),
        sorted: Some(true),
        st: parts.start_time_form,
        t0: parts.min_start_time,
        vb: parts.vertex_value_buckets,
        vq: (!parts.vertex_value_quant.is_empty()).then(|| parts.vertex_value_quant.clone()),
        vt: parts.vertex_time_encoding,
        vtf: parts.vertex_time_feature_step,
    };
    let tile_meta_json = serde_json::to_string(&tile_meta)
        .map_err(|e| Error::Other(format!("TILE_META encode failed: {e}")))?;

    // CORE schema: dataset-constant metadata ONLY (the per-tile keys live in
    // TILE_META), so the template bytes are identical across every tile of
    // the layer's shape — the whole point of the hoist.
    let mut core_meta: BTreeMap<String, String> = BTreeMap::new();
    core_meta.insert("stt:layer".to_string(), parts.layer_name.clone());
    core_meta.insert("stt:geometry".to_string(), parts.geometry_name.to_string());
    if parts.has_triangles {
        core_meta.insert(TRIANGLES_METADATA_KEY.to_string(), "true".to_string());
    }
    let core_fields: Vec<Arc<Field>> = parts.fields[..parts.reserved_len].to_vec();
    let core_columns: Vec<ArrayRef> = parts.columns[..parts.reserved_len].to_vec();
    let core_schema =
        Arc::new(Schema::new(core_fields).with_metadata(core_meta.into_iter().collect()));
    let core_ipc = write_ipc_stream(core_schema, core_columns)?;
    let core_boundary = split_ipc_at_schema(&core_ipc)?;
    let core_tail = core_ipc[core_boundary..].to_vec();
    let mut core_template = core_ipc;
    core_template.truncate(core_boundary);

    // PROPS schema: property fields only, no schema-level metadata (every
    // dataset-constant key lives on the CORE template).
    let props = if props_fields.is_empty() {
        None
    } else {
        let props_columns: Vec<ArrayRef> = parts.columns[parts.reserved_len..].to_vec();
        let props_schema = Arc::new(Schema::new(props_fields));
        let props_ipc = write_ipc_stream(props_schema, props_columns)?;
        // TB-4: when every dictionary in this stream was built against a pinned
        // dataset-global list, the DictionaryBatch messages are byte-identical
        // in every tile — so the cut moves past them and the category list
        // becomes template-resident (stored once per realized layer shape in
        // `manifest.schemas`) instead of re-shipped in every tile's tail. This
        // EXTENDS the TEMPLATE/TILE_META partition in its permitted direction:
        // a provably-constant blob moves template-side, nothing per-tile does.
        let boundary = match parts.props_dict_hoist {
            DictHoist::Template => split_ipc_after_dictionaries(&props_ipc)?,
            DictHoist::Tail => split_ipc_at_schema(&props_ipc)?,
        };
        let tail = props_ipc[boundary..].to_vec();
        let mut template = props_ipc;
        template.truncate(boundary);
        Some((template, tail))
    };

    Ok(EncodedLayerV2 {
        core_template,
        core_tail,
        props,
        tile_meta_json,
        feature_anchored_vertex_times: parts.vertex_time_feature_step.is_some(),
    })
}

/// Encode a full tile payload (one or more layers) with the layer frame.
///
/// Each layer's IPC stream is preceded by zero padding to an 8-byte boundary
/// relative to the payload start, so readers can wrap the stream zero-copy.
/// `ipc_len` records the exact IPC byte length (padding excluded); readers
/// derive the pad from alignment math alone.
pub fn encode_tile(layers: &[ColumnarLayer]) -> Result<Vec<u8>> {
    encode_tile_cfg(layers, &EncoderConfig::from_globals()).map(|(bytes, _)| bytes)
}

/// [`encode_tile`] with optional fixed-point coordinate quantization applied to
/// every layer (see [`encode_layer_quantized`]). `quantize_m = None` is
/// byte-identical to [`encode_tile`]; the other encoder settings come from the
/// process-wide globals.
pub fn encode_tile_quantized(layers: &[ColumnarLayer], quantize_m: Option<f64>) -> Result<Vec<u8>> {
    encode_tile_cfg(
        layers,
        &EncoderConfig {
            quantize_coords_m: quantize_m,
            ..EncoderConfig::from_globals()
        },
    )
    .map(|(bytes, _)| bytes)
}

/// [`encode_tile`] with a fully-explicit [`EncoderConfig`] — no process-wide
/// globals are read. The concurrency- and multi-config-safe entry point a
/// dynamic per-request tile server uses so each dataset/request encodes with its
/// own settings without touching shared state.
pub fn encode_tile_with(layers: &[ColumnarLayer], cfg: &EncoderConfig) -> Result<Vec<u8>> {
    encode_tile_cfg(layers, cfg).map(|(bytes, _)| bytes)
}

/// What an encode DID, for the caller that has to declare it.
///
/// Some capability declarations cannot be derived from the config: whether the
/// TB-11 feature-anchored vertex-time tier fires depends on the DATA (a layer's
/// per-feature spans versus its layer-wide span), so a writer only learns it by
/// encoding. This is that channel — deliberately a plain returned value rather
/// than shared state on [`EncoderConfig`], which is compared for byte-equality
/// and must stay a pure description of what to do.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodeObservations {
    /// Some layer used the feature-anchored vertex-time form (`TILE_META.vtf`),
    /// obliging the manifest to declare
    /// [`crate::pack::CAPABILITY_VERTEX_TIME_FEATURE_ANCHOR`].
    pub feature_anchored_vertex_times: bool,
}

/// [`encode_tile_with`], plus what the encode observed. Byte-identical output —
/// the two entry points share one implementation.
pub fn encode_tile_observed(
    layers: &[ColumnarLayer],
    cfg: &EncoderConfig,
) -> Result<(Vec<u8>, EncodeObservations)> {
    encode_tile_cfg(layers, cfg)
}

/// The single tile-encode implementation, driven entirely by an explicit
/// [`EncoderConfig`]. Every public `encode_tile*` entry point funnels here;
/// everything upstream (column/field build) is shared.
fn encode_tile_cfg(
    layers: &[ColumnarLayer],
    cfg: &EncoderConfig,
) -> Result<(Vec<u8>, EncodeObservations)> {
    if cfg.format_version != LAYER_FRAME_VERSION {
        return Err(Error::Other(format!(
            "unsupported layer-frame version {} (this writer emits {LAYER_FRAME_VERSION})",
            cfg.format_version
        )));
    }
    encode_tile_frame_v2(layers, cfg)
}

/// Zero-pad `out` to the next [`FRAME_ALIGN`] boundary — the pad length is
/// derived on decode, never stored.
fn pad_to_frame_align(out: &mut Vec<u8>) {
    let pad = (FRAME_ALIGN - out.len() % FRAME_ALIGN) % FRAME_ALIGN;
    out.extend_from_slice(&[0u8; FRAME_ALIGN][..pad]);
}

/// Layer-frame assembly (see the module docs). With a
/// [`TemplateCollector`] configured, schemas are recorded there and frames
/// carry 16-byte hash references (the packed-dataset mode); without one the
/// frame is self-contained via `INLINE_SCHEMA_*` sections.
fn encode_tile_frame_v2(
    layers: &[ColumnarLayer],
    cfg: &EncoderConfig,
) -> Result<(Vec<u8>, EncodeObservations)> {
    if layers.len() > u16::MAX as usize {
        return Err(Error::Other(format!(
            "tile has {} layers, exceeds the {} frame limit",
            layers.len(),
            u16::MAX
        )));
    }
    let collector = cfg.template_collector.as_deref();
    let mut observed = EncodeObservations::default();
    let mut out = Vec::new();
    out.extend_from_slice(&FRAME_V2_ESCAPE.to_le_bytes());
    out.push(FRAME_V2_VERSION);
    out.push(0); // flags — reserved, MUST be 0
    out.extend_from_slice(&(layers.len() as u16).to_le_bytes());
    for layer in layers {
        let name = layer.name.as_bytes();
        if name.len() > u16::MAX as usize {
            return Err(Error::Other("layer name too long".into()));
        }
        let enc = encode_layer_v2_parts(layer, cfg)?;
        observed.feature_anchored_vertex_times |= enc.feature_anchored_vertex_times;
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);

        // Schema references. Hash mode records the template with the
        // collector (content-addressed, so parallel encode order is
        // irrelevant); inline mode ships it as a section instead.
        match collector {
            Some(c) => {
                out.push(REF_KIND_TEMPLATE_HASH);
                out.extend_from_slice(&c.record(&enc.core_template));
            }
            None => out.push(REF_KIND_INLINE),
        }
        match (&enc.props, collector) {
            (None, _) => out.push(REF_KIND_NO_PROPS),
            (Some((template, _)), Some(c)) => {
                out.push(REF_KIND_TEMPLATE_HASH);
                out.extend_from_slice(&c.record(template));
            }
            (Some(_), None) => out.push(REF_KIND_INLINE),
        }

        // Sections in ascending tag order; TOC lengths are the exact at-rest
        // byte counts (padding derived, never stored).
        let mut sections: Vec<(u8, &[u8])> = Vec::new();
        if collector.is_none() {
            sections.push((SECTION_INLINE_SCHEMA_CORE, &enc.core_template));
        }
        sections.push((SECTION_TILE_META, enc.tile_meta_json.as_bytes()));
        sections.push((SECTION_CORE_BATCH, &enc.core_tail));
        if let Some((template, tail)) = &enc.props {
            if collector.is_none() {
                sections.push((SECTION_INLINE_SCHEMA_PROPS, template));
            }
            sections.push((SECTION_PROPS_BATCH, tail));
        }
        out.push(sections.len() as u8);
        for (tag, bytes) in &sections {
            out.push(*tag);
            let len = u32::try_from(bytes.len()).map_err(|_| {
                Error::Other(format!(
                    "layer '{}' section 0x{tag:02x} is {} bytes, exceeding the TOC's u32 \
                     length field",
                    layer.name,
                    bytes.len()
                ))
            })?;
            out.extend_from_slice(&len.to_le_bytes());
        }
        pad_to_frame_align(&mut out);
        for (_, bytes) in &sections {
            out.extend_from_slice(bytes);
            pad_to_frame_align(&mut out);
        }
    }
    Ok((out, observed))
}
