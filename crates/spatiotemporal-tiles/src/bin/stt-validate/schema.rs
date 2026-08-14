//! STT tile schema / column-type contract checks.
//!
//! Each decoded tile layer is an Arrow [`RecordBatch`]. The STT producer
//! ([`stt_core::arrow_tile::encode_layer`]) writes a fixed, self-describing
//! schema; this module validates a *decoded* batch against that contract so the
//! validator can catch producer drift, a wrong column type, or a missing
//! required column rather than letting a malformed tile slip through (the
//! existing checks only assert the payload decodes, not that its schema is
//! what every STT reader expects).
//!
//! The contract mirrors the documented per-layer schema in
//! `stt_core::arrow_tile` (see its module docs and `encode_layer`):
//!
//! | column                | Arrow type                                  | req |
//! |-----------------------|---------------------------------------------|-----|
//! | `id`                  | `UInt64`                                    | yes |
//! | `start_time`          | `Int64`                                     | yes |
//! | `end_time`            | `Int64`                                     | yes |
//! | `geometry`            | List/FixedSizeList w/ `geoarrow.*` ext name | yes |
//! | `vertex_time`         | `List<UInt16 \| UInt32>` (delta) or `List<Int64>` | opt |
//! | `vertex_value`        | `List<Float32>`                             | opt |
//! | `vertex_value_matrix` | `List<Float32>`                             | opt |
//! | `triangles`           | `List<UInt16>` or `List<UInt32>`            | opt |
//! | `part_offsets`        | `List<UInt32>`                              | opt |
//! | `<property>`          | `Float64`, `Utf8`, or `Dictionary<UInt16,Utf8>` | opt |
//!
//! `vertex_time` sizes itself per tile off a narrowest-first delta ladder
//! (`UInt16` then `UInt32`, both at the finest step that fits) before falling
//! back to absolute `Int64` — see `stt_core::arrow_tile::columns`. BOTH delta
//! widths are the same logical encoding and both carry the origin/step
//! metadata; only the `Int64` fallback is absolute.
//!
//! `part_offsets` is the polygon multi-part boundary column: one non-null
//! `List<UInt32>` per feature holding ring indices relative to that feature's
//! own first ring. It is emitted only when some feature in the tile has more
//! than one part, so its presence legitimately varies tile to tile.
//!
//! Every `opt` row above behaves that way — the encoder emits each optional
//! reserved column only when THAT tile's features carry the data (`triangles`
//! only when some feature needs hole-aware tessellation, `vertex_time` /
//! `vertex_value` / `vertex_value_matrix` only when the layer's features carry
//! per-vertex arrays, `part_offsets` only when some feature is multi-part).
//! One layer of one dataset therefore ships tiles that differ in WHICH
//! optional columns they carry, by design. [`OPTIONAL_RESERVED_COLUMNS`] is
//! the single list of those names (and their legal inner types), and
//! [`classify_column_drift`] reads it so cross-tile drift detection does not
//! mistake that adaptivity for producer drift.
//!
//! Beyond column shapes, the self-description metadata every reader depends on
//! is enforced too: an `Int32`-leaf (quantized) geometry must carry a parseable
//! `stt:quant` affine (and a `Float64`-leaf one must NOT), unquantized geometry
//! the CRS84 `ARROW:extension:metadata`, a delta (`List<UInt16 | UInt32>`)
//! `vertex_time` its `stt:vertex_time_origin_ms`/`step_ms` keys (which a
//! `List<Int64>` one must NOT carry), and `stt:vertex_value_buckets` must be a
//! positive integer consistent with the matrix column's row widths.
//!
//! Findings are split by severity ([`SchemaFindings`]): everything is an
//! error except a *missing* CRS84 metadata on unquantized geometry, which is
//! a warning — the writer MUST is newer than many deployed archives, and
//! rebuilding re-emits it.
//!
//! ## Where the expected types/names come from
//!
//! `stt_core::arrow_tile` keeps its GeoArrow extension-name key and the
//! `geoarrow.*` names *private*, so they can't be imported. The constants below
//! mirror those private definitions (the Arrow-standard `ARROW:extension:name`
//! metadata key and the GeoArrow geometry-type names) — keep them in sync if
//! `arrow_tile` ever renames them. The Arrow *data types* are reused directly
//! from the `arrow` crate, not hardcoded strings.

use arrow::datatypes::{DataType, Field};
use stt_core::arrow_tile::{
    AttrQuant, DecodedLayer, QuantAffine, STT_QUANT_ATTR_META_KEY, STT_QUANT_META_KEY,
};

/// GeoArrow extension-name metadata key. Mirrors the private
/// `GEOARROW_EXT_KEY` in `stt_core::arrow_tile`; this is the Arrow-standard
/// extension-name key, so it is stable across the Arrow ecosystem.
const GEOARROW_EXT_KEY: &str = "ARROW:extension:name";

/// GeoArrow extension-*metadata* key (the CRS carrier). Mirrors the private
/// `GEOARROW_EXT_META_KEY` in `stt_core::arrow_tile` — Arrow-standard, like
/// [`GEOARROW_EXT_KEY`].
const GEOARROW_EXT_META_KEY: &str = "ARROW:extension:metadata";

/// The CRS the conformance spec pins for unquantized geometry
/// (`docs/spec/conformance.md` §3: writer MUST emit the CRS84
/// `ARROW:extension:metadata`).
const CRS84: &str = "OGC:CRS84";

/// Schema-metadata keys for the u16-delta `vertex_time` encoding and the
/// `vertex_value_matrix` bucket count. Mirror the private constants in
/// `stt_core::arrow_tile` — keep in sync if `arrow_tile` ever renames them.
const VERTEX_TIME_ORIGIN_KEY: &str = "stt:vertex_time_origin_ms";
const VERTEX_TIME_STEP_KEY: &str = "stt:vertex_time_step_ms";
const VERTEX_VALUE_BUCKETS_KEY: &str = "stt:vertex_value_buckets";

/// Required columns and the Arrow type each must carry. Mirrors the leading
/// scalar fields `encode_layer` always writes.
const REQUIRED_SCALARS: &[(&str, DataType)] = &[
    ("id", DataType::UInt64),
    ("start_time", DataType::Int64),
    ("end_time", DataType::Int64),
];

/// The geometry column name `encode_layer` always uses.
const GEOMETRY_COLUMN: &str = "geometry";

/// The OPTIONAL reserved (non-property) columns and the inner types each may
/// carry when present. **The single source of truth for that set** — three
/// checks read it and must not disagree: the per-column type check
/// ([`check_optional_list`]), the reserved-vs-property split
/// ([`is_reserved_column`]), and cross-tile drift classification
/// ([`classify_column_drift`], which treats their present-vs-absent variation
/// as expected adaptivity rather than drift).
///
/// Per-column notes on the allowed widths:
///
/// - `vertex_time` — two delta widths (the narrowest-first ladder) plus the
///   absolute fallback. `UInt32` is what lets a track dataset whose span needs
///   a finer step than `UInt16` can express (~18.2 h at step 1 ms) validate at
///   all.
/// - `triangles` — feature-local indices, so `UInt16` unless a feature exceeds
///   65,535 vertices.
/// - `part_offsets` — ring indices relative to the feature's own first ring.
const OPTIONAL_RESERVED_COLUMNS: &[(&str, &[DataType])] = &[
    (
        "vertex_time",
        &[DataType::UInt16, DataType::UInt32, DataType::Int64],
    ),
    ("vertex_value", &[DataType::Float32]),
    ("vertex_value_matrix", &[DataType::Float32]),
    ("triangles", &[DataType::UInt16, DataType::UInt32]),
    ("part_offsets", &[DataType::UInt32]),
];

/// Schema findings split by severity. `errors` fail validation; `warnings`
/// flag tolerated legacy gaps (today only the missing-CRS84 case — see the
/// module docs).
#[derive(Default)]
pub struct SchemaFindings {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Check every layer of a decoded tile against the STT schema contract,
/// returning (possibly empty) lists of human-readable findings. Never panics:
/// callers collect these into the validator's error mechanism and continue
/// (or stop, under `--fail-fast`).
pub fn check_tile_schema(layers: &[DecodedLayer]) -> SchemaFindings {
    let mut findings = SchemaFindings::default();
    for layer in layers {
        check_layer_schema(layer, &mut findings);
    }
    findings
}

fn check_layer_schema(layer: &DecodedLayer, findings: &mut SchemaFindings) {
    let schema = layer.batch.schema();
    let name = &layer.name;

    // (a) required scalar columns present with the right type.
    for (col, want) in REQUIRED_SCALARS {
        match schema.field_with_name(col) {
            Ok(field) => {
                if field.data_type() != want {
                    findings.errors.push(format!(
                        "layer '{name}': column '{col}' has type {:?}, expected {want:?}",
                        field.data_type()
                    ));
                }
            }
            Err(_) => findings
                .errors
                .push(format!("layer '{name}': missing required column '{col}'")),
        }
    }

    // (b) geometry column present, with a GeoArrow extension name, and a
    // list-shaped coordinate type (FixedSizeList for points, List for lines /
    // polygons). We don't pin the exact nested shape — the round-trip decode
    // already proved it's a valid Arrow array — only that it's geometry.
    match schema.field_with_name(GEOMETRY_COLUMN) {
        Ok(field) => {
            check_geometry_extension(name, field, &mut findings.errors);
            if is_list_like(field.data_type()) {
                check_geometry_self_description(name, field, findings);
            } else {
                findings.errors.push(format!(
                    "layer '{name}': geometry column is {:?}, expected a (FixedSize)List of coordinates",
                    field.data_type()
                ));
            }
        }
        Err(_) => findings.errors.push(format!(
            "layer '{name}': missing required '{GEOMETRY_COLUMN}' column"
        )),
    }

    let issues = &mut findings.errors;

    // (c) optional per-vertex / per-part columns, when present, must carry the
    // expected List<inner> types. Absence is always legal — see
    // OPTIONAL_RESERVED_COLUMNS.
    for (col, allowed) in OPTIONAL_RESERVED_COLUMNS {
        check_optional_list(name, &schema, col, allowed, issues);
    }

    // (c') encodings that are only decodable through schema-level metadata:
    // the u16-delta vertex_time origin/step and the value-matrix bucket count.
    check_vertex_time_metadata(layer, issues);
    check_vertex_value_buckets(layer, issues);

    // (d) every remaining (property) column must be an encodable
    // property shapes the producer emits — anything else is producer drift.
    for field in schema.fields() {
        let n = field.name().as_str();
        if is_reserved_column(n) {
            continue;
        }
        if !is_valid_property_field(field) {
            issues.push(format!(
                "layer '{name}': property column '{n}' has type {:?}, expected Float64, \
                 Utf8, Dictionary<UInt16,Utf8>, or a quantized UInt16/Int32 carrying a \
                 valid '{STT_QUANT_ATTR_META_KEY}' affine",
                field.data_type()
            ));
        }
    }
}

/// Confirm the geometry field carries a `geoarrow.*` extension name.
fn check_geometry_extension(layer: &str, field: &Field, issues: &mut Vec<String>) {
    match field.metadata().get(GEOARROW_EXT_KEY) {
        Some(ext) if ext.starts_with("geoarrow.") => {}
        Some(ext) => issues.push(format!(
            "layer '{layer}': geometry extension name '{ext}' is not a geoarrow.* type"
        )),
        None => issues.push(format!(
            "layer '{layer}': geometry column missing the '{GEOARROW_EXT_KEY}' (GeoArrow) extension name"
        )),
    }
}

/// The geometry column's self-description contract, keyed on its leaf type:
///
/// - an `Int32` (fixed-point) leaf MUST carry a parseable `stt:quant`
///   reconstruction affine — without it a reader renders raw grid indices as
///   lon/lat (mirrors the `stt:qa` gate on quantized property columns);
/// - an unquantized `Float64` leaf must NOT carry `stt:quant` (a reader would
///   re-type real degrees as grid indices) and MUST carry the CRS84
///   `ARROW:extension:metadata` (a writer MUST, `docs/spec/conformance.md` §3)
///   — a *missing* CRS is only a warning, since archives written before that
///   MUST landed carry none and remain readable;
/// - any other leaf is producer drift.
fn check_geometry_self_description(layer: &str, field: &Field, findings: &mut SchemaFindings) {
    match geometry_leaf_type(field.data_type()) {
        DataType::Int32 => match field.metadata().get(STT_QUANT_META_KEY) {
            Some(json) if QuantAffine::from_json(json).is_some() => {}
            Some(json) => findings.errors.push(format!(
                "layer '{layer}': geometry '{STT_QUANT_META_KEY}' metadata is not a valid \
                 quantization affine JSON: {json}"
            )),
            None => findings.errors.push(format!(
                "layer '{layer}': quantized (Int32-leaf) geometry is missing the \
                 '{STT_QUANT_META_KEY}' reconstruction affine"
            )),
        },
        DataType::Float64 => {
            if field.metadata().contains_key(STT_QUANT_META_KEY) {
                findings.errors.push(format!(
                    "layer '{layer}': unquantized (Float64-leaf) geometry must not carry \
                     '{STT_QUANT_META_KEY}' (the affine re-types coordinates as grid indices)"
                ));
            }
            match field.metadata().get(GEOARROW_EXT_META_KEY) {
                Some(json) if crs_is_crs84(json) => {}
                Some(json) => findings.errors.push(format!(
                    "layer '{layer}': geometry '{GEOARROW_EXT_META_KEY}' does not pin the CRS \
                     to {CRS84}: {json}"
                )),
                None => findings.warnings.push(format!(
                    "layer '{layer}': unquantized geometry is missing the {CRS84} \
                     '{GEOARROW_EXT_META_KEY}' (writer MUST, docs/spec/conformance.md §3; \
                     pre-MUST archives lack it — rebuilding re-emits it)"
                )),
            }
        }
        other => findings.errors.push(format!(
            "layer '{layer}': geometry leaf type is {other:?}, expected Float64 (unquantized) \
             or Int32 (quantized, with '{STT_QUANT_META_KEY}')"
        )),
    }
}

/// Innermost non-nested (coordinate) type of a geometry column: Point is
/// `FixedSizeList<leaf, 2|3>`, LineString `List<FixedSizeList<leaf, 2>>`,
/// Polygon `List<List<FixedSizeList<leaf, 2>>>` — the leaf is what carries
/// `Float64` vs quantized `Int32`.
fn geometry_leaf_type(dt: &DataType) -> &DataType {
    match dt {
        DataType::List(inner) | DataType::LargeList(inner) | DataType::FixedSizeList(inner, _) => {
            geometry_leaf_type(inner.data_type())
        }
        other => other,
    }
}

/// True when the GeoArrow per-type extension metadata pins the CRS to CRS84
/// (`{"crs":"OGC:CRS84",...}`) — the axis-order-correct identifier for the
/// interleaved `[lon, lat]` storage.
fn crs_is_crs84(json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| v.get("crs").and_then(|c| c.as_str().map(String::from)))
        .is_some_and(|crs| crs == CRS84)
}

/// A delta-encoded `vertex_time` shape is only decodable with its per-layer
/// `(origin, step)` schema metadata — the TS reader would otherwise silently
/// reconstruct with `origin=0, step=1` (wrong timestamps, no error on either
/// side). Whenever the column is `List<UInt16>` **or** `List<UInt32>` (the two
/// rungs of the delta ladder — same encoding, different width), both keys must
/// be present with sane values (an integer Unix-ms origin and a positive
/// integer step); an absolute `List<Int64>` column must NOT carry either key (a
/// reader keying the encoding off the metadata would misdecode).
fn check_vertex_time_metadata(layer: &DecodedLayer, issues: &mut Vec<String>) {
    let schema = layer.batch.schema();
    let name = &layer.name;
    let Ok(field) = schema.field_with_name("vertex_time") else {
        return;
    };
    let is_delta = match field.data_type() {
        DataType::List(inner) | DataType::LargeList(inner) => {
            matches!(inner.data_type(), DataType::UInt16 | DataType::UInt32)
        }
        _ => false,
    };
    if !is_delta {
        for key in [VERTEX_TIME_ORIGIN_KEY, VERTEX_TIME_STEP_KEY] {
            if schema.metadata().contains_key(key) {
                issues.push(format!(
                    "layer '{name}': absolute (List<Int64>) vertex_time must not carry \
                     '{key}' (the key describes the delta encoding)"
                ));
            }
        }
        return;
    }
    let meta = schema.metadata();
    match meta.get(VERTEX_TIME_ORIGIN_KEY) {
        Some(v) if v.parse::<i64>().is_ok() => {}
        Some(v) => issues.push(format!(
            "layer '{name}': '{VERTEX_TIME_ORIGIN_KEY}' is not an integer Unix-ms origin: {v:?}"
        )),
        None => issues.push(format!(
            "layer '{name}': delta vertex_time is missing '{VERTEX_TIME_ORIGIN_KEY}'"
        )),
    }
    match meta.get(VERTEX_TIME_STEP_KEY) {
        Some(v) if v.parse::<u32>().map(|s| s >= 1).unwrap_or(false) => {}
        Some(v) => issues.push(format!(
            "layer '{name}': '{VERTEX_TIME_STEP_KEY}' must be a positive integer ms step, got {v:?}"
        )),
        None => issues.push(format!(
            "layer '{name}': delta vertex_time is missing '{VERTEX_TIME_STEP_KEY}'"
        )),
    }
}

/// `stt:vertex_value_buckets` must be a positive integer, only appear alongside
/// a `vertex_value_matrix` column, and agree with the matrix column's row
/// widths: every non-empty row is `vertex_count * buckets` values (vertex-major
/// flatten). An absent key with a present column is legal — the encoder omits
/// it when it can't infer a clean bucket count (the matrix is then present but
/// non-animatable).
fn check_vertex_value_buckets(layer: &DecodedLayer, issues: &mut Vec<String>) {
    use arrow::array::{Array, FixedSizeListArray, ListArray};

    let schema = layer.batch.schema();
    let name = &layer.name;
    let Some(raw) = schema.metadata().get(VERTEX_VALUE_BUCKETS_KEY) else {
        return;
    };
    let Some(matrix_col) = layer.batch.column_by_name("vertex_value_matrix") else {
        issues.push(format!(
            "layer '{name}': '{VERTEX_VALUE_BUCKETS_KEY}' is set but there is no \
             vertex_value_matrix column"
        ));
        return;
    };
    let buckets = match raw.parse::<u64>() {
        Ok(b) if b >= 1 => b as usize,
        _ => {
            issues.push(format!(
                "layer '{name}': '{VERTEX_VALUE_BUCKETS_KEY}' must be a positive integer, got {raw:?}"
            ));
            return;
        }
    };

    // Row-width consistency. Vertex counts come from the geometry column:
    // one per feature for points, the list length for linestrings. Polygon
    // nesting never carries a bucketed matrix (the encoder infers buckets for
    // LineString geometry only), so it is skipped.
    let Some(matrix) = matrix_col.as_any().downcast_ref::<ListArray>() else {
        return; // shape already reported by check_optional_list
    };
    let vertex_count = |i: usize| -> Option<usize> {
        let geom = layer.batch.column_by_name(GEOMETRY_COLUMN)?;
        if let Some(list) = geom.as_any().downcast_ref::<ListArray>() {
            return matches!(list.value_type(), DataType::FixedSizeList(_, _))
                .then(|| list.value_length(i) as usize);
        }
        geom.as_any()
            .downcast_ref::<FixedSizeListArray>()
            .map(|_| 1usize)
    };
    for i in 0..matrix.len() {
        if matrix.is_null(i) {
            continue;
        }
        let len = matrix.value_length(i) as usize;
        if len == 0 {
            continue;
        }
        let Some(nv) = vertex_count(i) else {
            return;
        };
        if nv == 0 || len != nv * buckets {
            issues.push(format!(
                "layer '{name}': vertex_value_matrix row {i} has {len} values, expected \
                 vertex_count {nv} × {VERTEX_VALUE_BUCKETS_KEY} {buckets}"
            ));
            return; // first mismatch only — one bad width implies the rest
        }
    }
}

/// A reserved (non-property) column name the producer manages itself: the
/// always-present scalars, geometry, or one of the optional reserved columns.
/// Derived from [`REQUIRED_SCALARS`] / [`GEOMETRY_COLUMN`] /
/// [`OPTIONAL_RESERVED_COLUMNS`] rather than re-listing the names, so a column
/// added to the contract cannot be reserved in one check and a "property" in
/// another (exactly how `part_offsets` reached the drift classifier as a
/// property).
fn is_reserved_column(name: &str) -> bool {
    name == GEOMETRY_COLUMN
        || REQUIRED_SCALARS.iter().any(|(c, _)| *c == name)
        || is_optional_reserved_column(name)
}

/// True for a reserved column whose PRESENCE is per-tile adaptive (the `opt`
/// rows of the module-doc table).
fn is_optional_reserved_column(name: &str) -> bool {
    allowed_inner_types(name).is_some()
}

/// The inner types an optional reserved column may carry, or `None` if `name`
/// is not one. More than one entry means the column also picks its WIDTH per
/// tile — see [`classify_column_drift`].
fn allowed_inner_types(name: &str) -> Option<&'static [DataType]> {
    OPTIONAL_RESERVED_COLUMNS
        .iter()
        .find(|(c, _)| *c == name)
        .map(|(_, types)| *types)
}

/// True for the list shapes geometry uses (`List` or `FixedSizeList`).
fn is_list_like(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
    )
}

/// A property column is one of the shapes `encode_layer` emits: a plain
/// `Float64` (numeric), `Utf8` or `Dictionary<UInt16, Utf8>` (adaptive
/// categorical), an
/// interleaved `FixedSizeList<Float32|UInt8, N>` vector (`--vector-group`), or —
/// when the build opts a numeric column into quantization (`--quantize-attr`) —
/// a fixed-point `UInt16`/`Int32` leaf carrying the `stt:qa` reconstruction
/// affine in its field metadata (the reader rebuilds Float64 from `{o,s}`).
fn is_valid_property_field(field: &Field) -> bool {
    match field.data_type() {
        DataType::Float64 | DataType::Utf8 => true,
        DataType::Dictionary(k, v) => {
            matches!(k.as_ref(), DataType::UInt16) && matches!(v.as_ref(), DataType::Utf8)
        }
        // Interleaved GPU-ready vector column: a FixedSizeList of Float32 (quat /
        // scale) or UInt8 (RGBA colour). The renderer binds the child buffer
        // zero-copy, so only those two leaf types are encodable.
        DataType::FixedSizeList(item, n) => {
            *n > 0 && matches!(item.data_type(), DataType::Float32 | DataType::UInt8)
        }
        // Quantized numeric: only valid WITH a parseable affine — a bare
        // UInt16/Int32 property with no (or malformed) `stt:qa` is producer
        // drift a reader cannot reconstruct Float64 from.
        DataType::UInt16 | DataType::Int32 => field
            .metadata()
            .get(STT_QUANT_ATTR_META_KEY)
            .is_some_and(|json| AttrQuant::from_json(json).is_some()),
        _ => false,
    }
}

/// If `col` is present it must be a `List` whose inner value type is one of
/// `allowed`. Absent is fine (the column is optional).
fn check_optional_list(
    layer: &str,
    schema: &arrow::datatypes::Schema,
    col: &str,
    allowed: &[DataType],
    issues: &mut Vec<String>,
) {
    let Ok(field) = schema.field_with_name(col) else {
        return;
    };
    match field.data_type() {
        DataType::List(inner) | DataType::LargeList(inner) => {
            if !allowed.iter().any(|a| a == inner.data_type()) {
                issues.push(format!(
                    "layer '{layer}': column '{col}' is List<{:?}>, expected List of one of {allowed:?}",
                    inner.data_type()
                ));
            }
        }
        other => issues.push(format!(
            "layer '{layer}': column '{col}' has type {:?}, expected a List",
            other
        )),
    }
}

/// A compact, order-stable signature of a tile's layer schemas, used to detect
/// producer drift across tiles. Two tiles whose layers carry the same column
/// names + Arrow types (and geometry extension names) hash to the same string;
/// any difference yields a different signature. Geometry *coordinate* shape is
/// folded into the type rendering, so a point tile and a polygon tile differ.
pub fn schema_signature(layers: &[DecodedLayer]) -> String {
    let mut parts: Vec<String> = layer_schema_signatures(layers)
        .into_iter()
        .map(|(name, cols)| format!("{name}{{{cols}}}"))
        .collect();
    parts.sort();
    parts.join("|")
}

/// Per-layer `(name, column signature)` for one tile.
///
/// Drift detection keys on this rather than on [`schema_signature`]: a tile's
/// LAYER SET legitimately varies across a dataset (a companion `_originals`
/// layer present only at some zooms), while a given layer's own columns must
/// stay stable. Comparing whole-tile signatures conflated the two and reported
/// every such dataset as drifting.
pub fn layer_schema_signatures(layers: &[DecodedLayer]) -> Vec<(String, String)> {
    layer_schema_columns(layers)
        .into_iter()
        .map(|(name, cols)| {
            let joined: Vec<String> = cols
                .into_iter()
                .map(|(c, ty)| format!("{c}:{ty}"))
                .collect();
            (name, joined.join(","))
        })
        .collect()
}

/// Per-layer `(name, [(column, type)])` for one tile — the per-column form the
/// drift check compares, so it can tell WHICH columns disagree and how.
pub fn layer_schema_columns(layers: &[DecodedLayer]) -> Vec<(String, Vec<(String, String)>)> {
    layers
        .iter()
        .map(|layer| {
            let cols: Vec<(String, String)> = layer
                .batch
                .schema()
                .fields()
                .iter()
                .map(|f| {
                    let ext = f
                        .metadata()
                        .get(GEOARROW_EXT_KEY)
                        .map(|e| format!("[{e}]"))
                        .unwrap_or_default();
                    (f.name().to_string(), format!("{:?}{ext}", f.data_type()))
                })
                .collect();
            (layer.name.clone(), cols)
        })
        .collect()
}

/// How one column's type differs between two tiles of the same layer.
#[derive(Debug, PartialEq, Eq)]
pub enum ColumnDrift {
    /// Expected per-tile adaptivity, not a defect. Two kinds:
    ///
    /// * **Different integer WIDTH, same logical encoding.** Two encodings size
    ///   themselves from the values in each tile: the per-vertex time column
    ///   ships as `List<UInt16>` deltas when the tile's span fits, else
    ///   absolute `List<Int64>`; a quantized attribute ships as the narrowest
    ///   of `UInt16`/`Int32`. That adaptivity IS the size win, and every reader
    ///   handles both.
    /// * **An OPTIONAL RESERVED column present in one tile, absent from the
    ///   other** ([`OPTIONAL_RESERVED_COLUMNS`]). The encoder emits each only
    ///   when that tile's features need it, and the contract for an absent one
    ///   is defined ("no feature is multi-part", "no baked tessellation", "no
    ///   per-vertex times"), so a reader is never handed a different meaning —
    ///   it is handed the documented default.
    /// * **Categorical representation.** Tile-local categorical strings may
    ///   switch between plain `Utf8` and `Dictionary<UInt16,Utf8>` according
    ///   to which is smaller; both carry the same logical strings.
    AdaptiveWidth,
    /// A structural disagreement: a PROPERTY column present in one tile and
    /// absent from the other, or an incompatible type-class change (for
    /// example categorical vs numeric). A consumer reading that column gets a
    /// different thing per tile — the original purpose of the drift check.
    Structural,
}

/// Classify the difference between one column's type in two tiles.
///
/// Present-vs-absent splits on WHICH column it is: an optional reserved column
/// (encoder-managed, documented absent-semantics) is expected adaptivity, while
/// a property column vanishing between tiles of one layer really does mean a
/// consumer gets a different thing per tile, and stays [`Structural`].
///
/// [`Structural`]: ColumnDrift::Structural
///
/// **Why presence folds into `AdaptiveWidth` instead of a third
/// `OptionalPresence` verdict.** A third variant would let the report say
/// "optional columns vary" separately from "encoding width varies", which reads
/// better — but it buys no behaviour: this enum's only consumer branches
/// `Structural` → error/exit code, everything else → an informational warning
/// tally that already prints the offending column and both of its shapes
/// verbatim (`part_offsets: <absent> vs List<UInt32>`), so a reader can still
/// see exactly which optional column varied and where. Keeping the enum
/// two-valued keeps that contract ("Structural is the only failing verdict")
/// legible at every call site. If the two notes are ever worth separate
/// wording, add `OptionalPresence` here plus a matching arm and tally headline
/// in `main.rs`'s drift loop — nothing else consumes it.
pub fn classify_column_drift(column: &str, a: Option<&str>, b: Option<&str>) -> ColumnDrift {
    let (a, b) = match (a, b) {
        (Some(a), Some(b)) => (a, b),
        // Neither tile has it: not a difference at all.
        (None, None) => return ColumnDrift::AdaptiveWidth,
        // Present in one tile, absent from the other.
        _ if is_optional_reserved_column(column) => return ColumnDrift::AdaptiveWidth,
        _ => return ColumnDrift::Structural,
    };
    if a == b {
        return ColumnDrift::AdaptiveWidth; // not a difference at all
    }
    let categorical = |t: &str| t == "Utf8" || t == "Dictionary(UInt16, Utf8)";
    if categorical(a) && categorical(b) {
        return ColumnDrift::AdaptiveWidth;
    }
    // Longest-first, and the `U`-prefixed names BEFORE their unsigned-less
    // substrings: "UInt32" contains "Int32", so replacing "Int32" first would
    // turn "UInt32" into "UINT" while "UInt16" became "INT" — two different
    // strings for the same logical encoding, i.e. a spurious Structural verdict.
    let int_widths = ["UInt16", "UInt32", "Int32", "Int64"];
    let strip_widths = |t: &str| {
        let mut out = t.to_string();
        for w in int_widths {
            out = out.replace(w, "INT");
        }
        out
    };
    // Which columns legitimately change integer WIDTH from tile to tile:
    //
    // * any optional reserved column the contract gives more than one legal
    //   inner type — `vertex_time`'s u16/u32-delta-then-absolute-i64 ladder,
    //   and `triangles`, whose index width is picked per tile from that tile's
    //   largest feature-local index (`<= u16::MAX` → UInt16, else UInt32).
    //   Read from OPTIONAL_RESERVED_COLUMNS rather than named here: hardcoding
    //   `vertex_time` is what left `triangles`' own per-tile width choice
    //   classified as Structural;
    // * a quantized property attribute, which ships as the narrower of
    //   UInt16/Int32 — its type string IS the leaf, so it starts with a width.
    let adaptive_column = allowed_inner_types(column).is_some_and(|types| types.len() > 1)
        || (int_widths.iter().any(|w| a.starts_with(w))
            && int_widths.iter().any(|w| b.starts_with(w)));
    if adaptive_column && strip_widths(a) == strip_widths(b) {
        ColumnDrift::AdaptiveWidth
    } else {
        ColumnDrift::Structural
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;
    use std::collections::HashMap;
    use std::sync::Arc;
    use stt_core::arrow_tile::{
        decode_tile, encode_tile, ColumnarLayer, GeometryColumn, PropertyColumn,
    };

    /// A well-formed point layer (id/start/end/geometry + one numeric + one
    /// categorical property), encoded then decoded so the test exercises the
    /// real producer schema.
    fn good_point_layer() -> ColumnarLayer {
        ColumnarLayer {
            polygon_parts: None,
            name: "default".into(),
            feature_ids: vec![1, 2, 3],
            start_times: vec![10, 20, 30],
            end_times: vec![15, 25, 35],
            geometry: GeometryColumn::Point(vec![[-1.0, 2.0], [-1.1, 2.1], [-1.2, 2.2]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![
                (
                    "speed".into(),
                    PropertyColumn::Numeric(vec![Some(1.0), None, Some(3.0)]),
                ),
                (
                    "kind".into(),
                    PropertyColumn::Categorical(vec![Some("a".into()), Some("b".into()), None]),
                ),
            ],
        }
    }

    fn decode(layers: &[ColumnarLayer]) -> Vec<DecodedLayer> {
        decode_tile(&encode_tile(layers).unwrap()).unwrap()
    }

    /// Errors and warnings flattened together — for the "this passes clean"
    /// assertions where neither severity is expected.
    fn all_findings(layers: &[DecodedLayer]) -> Vec<String> {
        let f = check_tile_schema(layers);
        f.errors.into_iter().chain(f.warnings).collect()
    }

    #[test]
    fn well_formed_tile_passes_schema_check() {
        let decoded = decode(&[good_point_layer()]);
        let issues = all_findings(&decoded);
        assert!(issues.is_empty(), "unexpected schema issues: {issues:?}");
    }

    #[test]
    fn line_layer_with_vertex_columns_passes() {
        // A line layer carrying u16-delta vertex_time and Float32 vertex_value —
        // both optional List columns the contract accepts.
        let layer = ColumnarLayer {
            polygon_parts: None,
            name: "tracks".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![100],
            geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]]),
            vertex_times: Some(vec![vec![0, 50, 100]]),
            vertex_values: Some(vec![vec![1.0, 2.0, 3.0]]),
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let issues = all_findings(&decode(&[layer]));
        assert!(issues.is_empty(), "unexpected schema issues: {issues:?}");
    }

    #[test]
    fn missing_required_column_is_reported() {
        // Hand-build a batch that drops `start_time` — the producer never does
        // this, so we assemble the RecordBatch directly.
        use arrow::array::{Int64Array, UInt64Array};
        use arrow::datatypes::Schema;
        use arrow::record_batch::RecordBatch;

        let id = Arc::new(UInt64Array::from(vec![1u64, 2]));
        let end = Arc::new(Int64Array::from(vec![10i64, 20]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("end_time", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![id, end]).unwrap();
        let layers = vec![DecodedLayer {
            name: "broken".into(),
            batch,
        }];

        let issues = check_tile_schema(&layers).errors;
        assert!(
            issues
                .iter()
                .any(|i| i.contains("missing required column 'start_time'")),
            "issues were: {issues:?}"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.contains("missing required 'geometry'")),
            "issues were: {issues:?}"
        );
    }

    #[test]
    fn wrong_column_type_is_reported() {
        // `id` typed Int64 instead of UInt64.
        use arrow::array::{Array, FixedSizeListArray, Float64Array, Int64Array};
        use arrow::datatypes::Schema;
        use arrow::record_batch::RecordBatch;

        let id = Arc::new(Int64Array::from(vec![1i64]));
        let start = Arc::new(Int64Array::from(vec![0i64]));
        let end = Arc::new(Int64Array::from(vec![10i64]));
        let coord_field = Arc::new(Field::new("xy", DataType::Float64, false));
        let geom = Arc::new(FixedSizeListArray::new(
            coord_field,
            2,
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            None,
        ));
        let mut geom_meta = HashMap::new();
        geom_meta.insert(GEOARROW_EXT_KEY.to_string(), "geoarrow.point".to_string());
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("start_time", DataType::Int64, false),
            Field::new("end_time", DataType::Int64, false),
            Field::new("geometry", geom.data_type().clone(), false).with_metadata(geom_meta),
        ]));
        let batch = RecordBatch::try_new(schema, vec![id, start, end, geom]).unwrap();
        let issues = check_tile_schema(&[DecodedLayer {
            name: "drift".into(),
            batch,
        }])
        .errors;
        assert!(
            issues
                .iter()
                .any(|i| i.contains("column 'id'") && i.contains("expected UInt64")),
            "issues were: {issues:?}"
        );
    }

    #[test]
    fn geometry_without_extension_name_is_reported() {
        use arrow::array::{Array, FixedSizeListArray, Float64Array, Int64Array, UInt64Array};
        use arrow::datatypes::Schema;
        use arrow::record_batch::RecordBatch;

        let id = Arc::new(UInt64Array::from(vec![1u64]));
        let start = Arc::new(Int64Array::from(vec![0i64]));
        let end = Arc::new(Int64Array::from(vec![10i64]));
        let coord_field = Arc::new(Field::new("xy", DataType::Float64, false));
        let geom = Arc::new(FixedSizeListArray::new(
            coord_field,
            2,
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            None,
        ));
        // No geoarrow extension metadata on the geometry field.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("start_time", DataType::Int64, false),
            Field::new("end_time", DataType::Int64, false),
            Field::new("geometry", geom.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![id, start, end, geom]).unwrap();
        let issues = check_tile_schema(&[DecodedLayer {
            name: "nogeo".into(),
            batch,
        }])
        .errors;
        assert!(
            issues.iter().any(|i| i.contains("extension name")),
            "issues were: {issues:?}"
        );
    }

    /// Rebuild a decoded layer's batch with the schema-level metadata replaced,
    /// keeping every field (incl. field-level metadata) and column intact.
    fn with_schema_metadata(
        layer: &DecodedLayer,
        metadata: HashMap<String, String>,
    ) -> DecodedLayer {
        use arrow::datatypes::Schema;
        use arrow::record_batch::RecordBatch;
        let schema = Arc::new(Schema::new_with_metadata(
            layer.batch.schema().fields().clone(),
            metadata,
        ));
        DecodedLayer {
            name: layer.name.clone(),
            batch: RecordBatch::try_new(schema, layer.batch.columns().to_vec()).unwrap(),
        }
    }

    #[test]
    fn quantized_geometry_with_affine_passes() {
        use stt_core::arrow_tile::encode_tile_quantized;
        let payload = encode_tile_quantized(&[good_point_layer()], Some(1.0)).unwrap();
        let decoded = decode_tile(&payload).unwrap();
        let issues = all_findings(&decoded);
        assert!(issues.is_empty(), "unexpected schema issues: {issues:?}");
    }

    #[test]
    fn int32_geometry_without_quant_affine_is_reported() {
        use arrow::array::{Array, FixedSizeListArray, Int32Array, Int64Array, UInt64Array};
        use arrow::datatypes::Schema;
        use arrow::record_batch::RecordBatch;

        let id = Arc::new(UInt64Array::from(vec![1u64]));
        let start = Arc::new(Int64Array::from(vec![0i64]));
        let end = Arc::new(Int64Array::from(vec![10i64]));
        let coord_field = Arc::new(Field::new("xy", DataType::Int32, false));
        let geom = Arc::new(FixedSizeListArray::new(
            coord_field,
            2,
            Arc::new(Int32Array::from(vec![100, 200])),
            None,
        ));
        let build = |quant: Option<&str>| {
            let mut geom_meta = HashMap::new();
            geom_meta.insert(GEOARROW_EXT_KEY.to_string(), "geoarrow.point".to_string());
            if let Some(q) = quant {
                geom_meta.insert(STT_QUANT_META_KEY.to_string(), q.to_string());
            }
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("start_time", DataType::Int64, false),
                Field::new("end_time", DataType::Int64, false),
                Field::new("geometry", geom.data_type().clone(), false).with_metadata(geom_meta),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![id.clone(), start.clone(), end.clone(), geom.clone()],
            )
            .unwrap();
            check_tile_schema(&[DecodedLayer {
                name: "q".into(),
                batch,
            }])
            .errors
        };

        // Missing affine entirely.
        let issues = build(None);
        assert!(
            issues
                .iter()
                .any(|i| i.contains("stt:quant") && i.contains("missing")),
            "issues were: {issues:?}"
        );
        // Present but malformed JSON.
        let issues = build(Some("{not json"));
        assert!(
            issues
                .iter()
                .any(|i| i.contains("stt:quant") && i.contains("not a valid")),
            "issues were: {issues:?}"
        );
        // Present and well-formed: clean.
        let issues = build(Some(r#"{"x0":-180.0,"y0":-90.0,"sx":1e-5,"sy":1e-5}"#));
        assert!(issues.is_empty(), "unexpected schema issues: {issues:?}");
    }

    /// Rebuild a decoded layer's batch with the geometry FIELD metadata
    /// transformed, keeping every other field and column intact.
    fn with_geometry_metadata(
        layer: &DecodedLayer,
        edit: impl Fn(&mut HashMap<String, String>),
    ) -> DecodedLayer {
        use arrow::datatypes::Schema;
        use arrow::record_batch::RecordBatch;
        let fields: Vec<Arc<Field>> = layer
            .batch
            .schema()
            .fields()
            .iter()
            .map(|f| {
                if f.name() == "geometry" {
                    let mut meta = f.metadata().clone();
                    edit(&mut meta);
                    Arc::new(f.as_ref().clone().with_metadata(meta))
                } else {
                    f.clone()
                }
            })
            .collect();
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            layer.batch.schema().metadata().clone(),
        ));
        let batch = RecordBatch::try_new(schema, layer.batch.columns().to_vec()).unwrap();
        DecodedLayer {
            name: layer.name.clone(),
            batch,
        }
    }

    #[test]
    fn unquantized_geometry_without_crs84_metadata_is_warned() {
        // Strip the geometry field's CRS metadata from a real encoded tile,
        // keeping the extension NAME — the conformance writer-MUST is the CRS.
        // Older archives lack it, so the finding is warn-level, not an error.
        let decoded = decode(&[good_point_layer()]);
        let stripped = with_geometry_metadata(&decoded[0], |meta| {
            meta.remove(GEOARROW_EXT_META_KEY);
        });
        let findings = check_tile_schema(&[stripped]);
        assert!(
            findings.errors.is_empty(),
            "errors were: {:?}",
            findings.errors
        );
        assert!(
            findings
                .warnings
                .iter()
                .any(|i| i.contains(CRS84) && i.contains("missing")),
            "warnings were: {:?}",
            findings.warnings
        );
    }

    #[test]
    fn float64_geometry_carrying_quant_affine_is_reported() {
        // The inverse of the Int32 gate: a real-degrees geometry that ALSO
        // claims a quantization affine would re-type readers' coordinates.
        let decoded = decode(&[good_point_layer()]);
        let tampered = with_geometry_metadata(&decoded[0], |meta| {
            meta.insert(
                STT_QUANT_META_KEY.to_string(),
                r#"{"x0":-180.0,"y0":-90.0,"sx":1e-5,"sy":1e-5}"#.to_string(),
            );
        });
        let issues = check_tile_schema(&[tampered]).errors;
        assert!(
            issues
                .iter()
                .any(|i| i.contains(STT_QUANT_META_KEY) && i.contains("must not carry")),
            "issues were: {issues:?}"
        );
    }

    #[test]
    fn u16_delta_vertex_time_requires_origin_and_step() {
        // A tight-span line layer encodes vertex_time as u16 deltas; stripping
        // the schema metadata must flag BOTH missing keys.
        let layer = ColumnarLayer {
            polygon_parts: None,
            name: "tracks".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![100],
            geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]]),
            vertex_times: Some(vec![vec![0, 50, 100]]),
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let decoded = decode(&[layer]);
        assert!(all_findings(&decoded).is_empty());

        let stripped = with_schema_metadata(&decoded[0], HashMap::new());
        let issues = check_tile_schema(&[stripped]).errors;
        assert!(
            issues
                .iter()
                .any(|i| i.contains(VERTEX_TIME_ORIGIN_KEY) && i.contains("missing")),
            "issues were: {issues:?}"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.contains(VERTEX_TIME_STEP_KEY) && i.contains("missing")),
            "issues were: {issues:?}"
        );

        // Insane values (non-integer origin, zero step) are rejected too.
        let mut bad = HashMap::new();
        bad.insert(
            VERTEX_TIME_ORIGIN_KEY.to_string(),
            "not-a-number".to_string(),
        );
        bad.insert(VERTEX_TIME_STEP_KEY.to_string(), "0".to_string());
        let issues = check_tile_schema(&[with_schema_metadata(&decoded[0], bad)]).errors;
        assert!(
            issues
                .iter()
                .any(|i| i.contains(VERTEX_TIME_ORIGIN_KEY) && i.contains("not an integer")),
            "issues were: {issues:?}"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.contains(VERTEX_TIME_STEP_KEY) && i.contains("positive")),
            "issues were: {issues:?}"
        );
    }

    #[test]
    fn absolute_vertex_time_must_not_carry_delta_keys() {
        // A span far beyond EVERY delta tier's step ceiling forces the absolute
        // List<Int64> encoding, which legitimately carries no origin/step.
        // The widest tier is u32 at a 1000 ms ceiling, so the span has to
        // exceed 1000 * u32::MAX ≈ 4.295e12 ms before the ladder gives up;
        // 5e12 ms (~158 years) needs a 1165 ms step and so lands on Int64.
        let span_ms: i64 = 5_000_000_000_000;
        let layer = ColumnarLayer {
            polygon_parts: None,
            name: "tracks".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![span_ms],
            geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]]),
            vertex_times: Some(vec![vec![0, span_ms / 2, span_ms]]),
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let decoded = decode(&[layer]);
        let vt = decoded[0]
            .batch
            .schema()
            .field_with_name("vertex_time")
            .unwrap()
            .data_type()
            .clone();
        assert!(
            matches!(&vt, DataType::List(inner) if inner.data_type() == &DataType::Int64),
            "expected the absolute Int64 fallback, got {vt:?}"
        );
        assert!(all_findings(&decoded).is_empty());

        // Sneaking the delta keys onto an absolute column is producer drift.
        let mut meta = decoded[0].batch.schema().metadata().clone();
        meta.insert(VERTEX_TIME_ORIGIN_KEY.to_string(), "0".to_string());
        meta.insert(VERTEX_TIME_STEP_KEY.to_string(), "1000".to_string());
        let issues = check_tile_schema(&[with_schema_metadata(&decoded[0], meta)]).errors;
        for key in [VERTEX_TIME_ORIGIN_KEY, VERTEX_TIME_STEP_KEY] {
            assert!(
                issues
                    .iter()
                    .any(|i| i.contains(key) && i.contains("must not carry")),
                "no 'must not carry' issue for {key}; issues were: {issues:?}"
            );
        }
    }

    #[test]
    fn u32_delta_vertex_time_is_accepted_and_still_needs_origin_and_step() {
        // The middle rung of the ladder: a span past u16's ~18.2 h reach at
        // step 1 ms but well inside u32's. Any multi-day track dataset lands
        // here, so the validator MUST accept List<UInt32> and MUST still
        // demand the origin/step metadata that makes it decodable.
        let span_ms: i64 = 3 * 24 * 60 * 60 * 1000; // 3 days
        let layer = ColumnarLayer {
            polygon_parts: None,
            name: "tracks".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![span_ms],
            geometry: GeometryColumn::LineString(vec![vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]]),
            vertex_times: Some(vec![vec![0, span_ms / 2, span_ms]]),
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let decoded = decode(&[layer]);
        let vt = decoded[0]
            .batch
            .schema()
            .field_with_name("vertex_time")
            .unwrap()
            .data_type()
            .clone();
        assert!(
            matches!(&vt, DataType::List(inner) if inner.data_type() == &DataType::UInt32),
            "expected the u32 delta tier, got {vt:?}"
        );
        let issues = all_findings(&decoded);
        assert!(issues.is_empty(), "unexpected schema issues: {issues:?}");

        // Stripping the metadata must still be reported — the u32 rung is a
        // delta encoding, not the absolute fallback.
        let stripped = with_schema_metadata(&decoded[0], HashMap::new());
        let issues = check_tile_schema(&[stripped]).errors;
        for key in [VERTEX_TIME_ORIGIN_KEY, VERTEX_TIME_STEP_KEY] {
            assert!(
                issues
                    .iter()
                    .any(|i| i.contains(key) && i.contains("missing")),
                "no 'missing' issue for {key}; issues were: {issues:?}"
            );
        }
    }

    #[test]
    fn vertex_time_width_drift_across_tiles_is_adaptive_not_structural() {
        // A dataset whose tiles straddle the u16→u32 threshold ships both
        // widths. That is the ladder working as designed, not producer drift.
        for (a, b) in [
            (
                "List(Field { data_type: UInt16 })",
                "List(Field { data_type: UInt32 })",
            ),
            (
                "List(Field { data_type: UInt32 })",
                "List(Field { data_type: Int64 })",
            ),
        ] {
            assert_eq!(
                classify_column_drift("vertex_time", Some(a), Some(b)),
                ColumnDrift::AdaptiveWidth,
                "{a} vs {b} should be an adaptive width difference"
            );
        }
    }

    #[test]
    fn triangle_index_width_drift_across_tiles_is_adaptive_not_structural() {
        // `triangles` picks UInt16 vs UInt32 from each tile's largest
        // feature-local index, exactly like vertex_time picks its delta width —
        // so a polygon layer with one dense tile ships both, and neither is
        // drift. Only vertex_time was named in the width rule, so this pair
        // read as Structural.
        assert_eq!(
            classify_column_drift(
                "triangles",
                Some("List(Field { data_type: UInt16 })"),
                Some("List(Field { data_type: UInt32 })"),
            ),
            ColumnDrift::AdaptiveWidth
        );
        // The relaxation is width-only: a different type CLASS is still drift.
        assert_eq!(
            classify_column_drift(
                "triangles",
                Some("List(Field { data_type: UInt16 })"),
                Some("List(Field { data_type: Float32 })"),
            ),
            ColumnDrift::Structural
        );
        // A single-type optional column has no width to adapt: Float32 vs
        // anything else is real drift.
        assert_eq!(
            classify_column_drift(
                "vertex_value",
                Some("List(Field { data_type: Float32 })"),
                Some("List(Field { data_type: UInt16 })"),
            ),
            ColumnDrift::Structural
        );
    }

    #[test]
    fn optional_reserved_column_presence_drift_is_adaptive_not_structural() {
        // Every optional reserved column is emitted per TILE, only when that
        // tile's features need it (`part_offsets` only for a multi-part
        // feature, `triangles` only when some feature needs hole-aware
        // tessellation, the vertex columns only when per-vertex arrays exist).
        // A layer whose tiles differ in which ones they carry is the encoder
        // working as documented, not producer drift — reporting it Structural
        // hard-failed every mixed polygon dataset.
        for (col, ty) in [
            ("part_offsets", "List(Field { data_type: UInt32 })"),
            ("triangles", "List(Field { data_type: UInt16 })"),
            ("vertex_time", "List(Field { data_type: UInt16 })"),
            ("vertex_value", "List(Field { data_type: Float32 })"),
            ("vertex_value_matrix", "List(Field { data_type: Float32 })"),
        ] {
            assert!(is_optional_reserved_column(col), "{col} must be optional");
            for (a, b) in [(None, Some(ty)), (Some(ty), None)] {
                assert_eq!(
                    classify_column_drift(col, a, b),
                    ColumnDrift::AdaptiveWidth,
                    "{col}: {a:?} vs {b:?} should be expected adaptivity"
                );
            }
        }
    }

    #[test]
    fn property_column_presence_and_type_class_drift_stay_structural() {
        // The check's original purpose: a PROPERTY column that appears in one
        // tile and not the other, or flips encoding class, hands a consumer a
        // different thing per tile.
        for (a, b) in [
            (None, Some("Float64")),
            (Some("Float64"), None),
            (Some("Dictionary(UInt16, Utf8)"), Some("Float64")),
        ] {
            assert_eq!(
                classify_column_drift("speed", a, b),
                ColumnDrift::Structural,
                "property 'speed': {a:?} vs {b:?} should be structural drift"
            );
        }
        // Geometry is reserved but NOT optional — it can never go missing.
        assert!(is_reserved_column(GEOMETRY_COLUMN));
        assert!(!is_optional_reserved_column(GEOMETRY_COLUMN));
        assert_eq!(
            classify_column_drift(
                GEOMETRY_COLUMN,
                None,
                Some("List(Field { data_type: Float64 })")
            ),
            ColumnDrift::Structural
        );
    }

    #[test]
    fn categorical_utf8_and_dictionary_drift_is_adaptive_not_structural() {
        assert_eq!(
            classify_column_drift("kind", Some("Utf8"), Some("Dictionary(UInt16, Utf8)")),
            ColumnDrift::AdaptiveWidth
        );
        assert_eq!(
            classify_column_drift("kind", Some("Utf8"), Some("Float64")),
            ColumnDrift::Structural
        );
    }

    #[test]
    fn multi_part_polygon_part_offsets_column_is_reserved_and_typed() {
        // `part_offsets` is a reserved CORE column, not a property: it must
        // pass the List<UInt32> check and must NOT be reported as an
        // unencodable property column.
        let outer = |dx: f64| {
            vec![
                [dx, 0.0],
                [dx + 1.0, 0.0],
                [dx + 1.0, 1.0],
                [dx, 1.0],
                [dx, 0.0],
            ]
        };
        let layer = ColumnarLayer {
            // Two disjoint squares in one feature → parts start at rings 0 and 1.
            polygon_parts: Some(vec![vec![0, 1]]),
            name: "areas".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![100],
            geometry: GeometryColumn::Polygon(vec![vec![outer(0.0), outer(10.0)]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let decoded = decode(&[layer]);
        let parts = decoded[0]
            .batch
            .schema()
            .field_with_name("part_offsets")
            .expect("multi-part polygon layer carries part_offsets")
            .data_type()
            .clone();
        assert!(
            matches!(&parts, DataType::List(inner) if inner.data_type() == &DataType::UInt32),
            "expected List<UInt32> part offsets, got {parts:?}"
        );
        assert!(is_reserved_column("part_offsets"));
        let issues = all_findings(&decoded);
        assert!(issues.is_empty(), "unexpected schema issues: {issues:?}");
    }

    #[test]
    fn vertex_value_buckets_must_be_positive_and_match_matrix_width() {
        // 3-vertex + 2-vertex lines, 2 buckets each → rows of 6 and 4 values.
        let layer = ColumnarLayer {
            polygon_parts: None,
            name: "corridors".into(),
            feature_ids: vec![1, 2],
            start_times: vec![0, 0],
            end_times: vec![100, 100],
            geometry: GeometryColumn::LineString(vec![
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                vec![[5.0, 5.0], [6.0, 6.0]],
            ]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: Some(vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0, 10.0],
            ]),
            properties: vec![],
        };
        let decoded = decode(&[layer]);
        assert_eq!(
            decoded[0]
                .batch
                .schema()
                .metadata()
                .get(VERTEX_VALUE_BUCKETS_KEY)
                .map(String::as_str),
            Some("2")
        );
        assert!(all_findings(&decoded).is_empty());

        let tamper = |value: &str| {
            let mut meta = decoded[0].batch.schema().metadata().clone();
            meta.insert(VERTEX_VALUE_BUCKETS_KEY.to_string(), value.to_string());
            check_tile_schema(&[with_schema_metadata(&decoded[0], meta)]).errors
        };
        // Positive but inconsistent with the row widths.
        let issues = tamper("4");
        assert!(
            issues
                .iter()
                .any(|i| i.contains("vertex_value_matrix row") && i.contains("expected")),
            "issues were: {issues:?}"
        );
        // Non-positive / non-integer.
        for bad in ["0", "-2", "two"] {
            let issues = tamper(bad);
            assert!(
                issues
                    .iter()
                    .any(|i| i.contains(VERTEX_VALUE_BUCKETS_KEY) && i.contains("positive")),
                "value {bad:?}: issues were: {issues:?}"
            );
        }
    }

    #[test]
    fn quantized_property_with_malformed_affine_is_reported() {
        use arrow::array::{
            Array, FixedSizeListArray, Float64Array, Int64Array, UInt16Array, UInt64Array,
        };
        use arrow::datatypes::Schema;
        use arrow::record_batch::RecordBatch;

        let id = Arc::new(UInt64Array::from(vec![1u64]));
        let start = Arc::new(Int64Array::from(vec![0i64]));
        let end = Arc::new(Int64Array::from(vec![10i64]));
        let coord_field = Arc::new(Field::new("xy", DataType::Float64, false));
        let geom = Arc::new(FixedSizeListArray::new(
            coord_field,
            2,
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            None,
        ));
        let mut geom_meta = HashMap::new();
        geom_meta.insert(GEOARROW_EXT_KEY.to_string(), "geoarrow.point".to_string());
        geom_meta.insert(
            GEOARROW_EXT_META_KEY.to_string(),
            r#"{"crs":"OGC:CRS84","crs_type":"authority_code"}"#.to_string(),
        );
        let mut qa_meta = HashMap::new();
        qa_meta.insert(STT_QUANT_ATTR_META_KEY.to_string(), "{broken".to_string());
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("start_time", DataType::Int64, false),
            Field::new("end_time", DataType::Int64, false),
            Field::new("geometry", geom.data_type().clone(), false).with_metadata(geom_meta),
            Field::new("depth", DataType::UInt16, true).with_metadata(qa_meta),
        ]));
        let depth = Arc::new(UInt16Array::from(vec![7u16]));
        let batch = RecordBatch::try_new(schema, vec![id, start, end, geom, depth]).unwrap();
        let issues = check_tile_schema(&[DecodedLayer {
            name: "qa".into(),
            batch,
        }])
        .errors;
        assert!(
            issues
                .iter()
                .any(|i| i.contains("'depth'") && i.contains(STT_QUANT_ATTR_META_KEY)),
            "issues were: {issues:?}"
        );
    }

    #[test]
    fn identical_tiles_share_a_signature_and_differing_ones_do_not() {
        let a = decode(&[good_point_layer()]);
        let b = decode(&[good_point_layer()]);
        assert_eq!(schema_signature(&a), schema_signature(&b));

        // A polygon layer has a different geometry shape + extension name.
        let poly = ColumnarLayer {
            polygon_parts: None,
            name: "default".into(),
            feature_ids: vec![1],
            start_times: vec![0],
            end_times: vec![1],
            geometry: GeometryColumn::Polygon(vec![vec![vec![
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]]]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        let c = decode(&[poly]);
        assert_ne!(schema_signature(&a), schema_signature(&c));
    }
}
