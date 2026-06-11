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
//! | `vertex_time`         | `List<UInt16>` (delta) or `List<Int64>`     | opt |
//! | `vertex_value`        | `List<Float32>`                             | opt |
//! | `vertex_value_matrix` | `List<Float32>`                             | opt |
//! | `triangles`           | `List<UInt32>`                              | opt |
//! | `<property>`          | `Float64` or `Dictionary<UInt16,Utf8>`      | opt |
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
use stt_core::arrow_tile::DecodedLayer;

/// GeoArrow extension-name metadata key. Mirrors the private
/// `GEOARROW_EXT_KEY` in `stt_core::arrow_tile`; this is the Arrow-standard
/// extension-name key, so it is stable across the Arrow ecosystem.
const GEOARROW_EXT_KEY: &str = "ARROW:extension:name";

/// Required columns and the Arrow type each must carry. Mirrors the leading
/// scalar fields `encode_layer` always writes.
const REQUIRED_SCALARS: &[(&str, DataType)] = &[
    ("id", DataType::UInt64),
    ("start_time", DataType::Int64),
    ("end_time", DataType::Int64),
];

/// The geometry column name `encode_layer` always uses.
const GEOMETRY_COLUMN: &str = "geometry";

/// Check every layer of a decoded tile against the STT schema contract,
/// returning a (possibly empty) list of human-readable issues. Never panics:
/// callers collect these into the validator's error mechanism and continue
/// (or stop, under `--fail-fast`).
pub fn check_tile_schema(layers: &[DecodedLayer]) -> Vec<String> {
    let mut issues = Vec::new();
    for layer in layers {
        check_layer_schema(layer, &mut issues);
    }
    issues
}

fn check_layer_schema(layer: &DecodedLayer, issues: &mut Vec<String>) {
    let schema = layer.batch.schema();
    let name = &layer.name;

    // (a) required scalar columns present with the right type.
    for (col, want) in REQUIRED_SCALARS {
        match schema.field_with_name(col) {
            Ok(field) => {
                if field.data_type() != want {
                    issues.push(format!(
                        "layer '{name}': column '{col}' has type {:?}, expected {want:?}",
                        field.data_type()
                    ));
                }
            }
            Err(_) => issues.push(format!("layer '{name}': missing required column '{col}'")),
        }
    }

    // (b) geometry column present, with a GeoArrow extension name, and a
    // list-shaped coordinate type (FixedSizeList for points, List for lines /
    // polygons). We don't pin the exact nested shape — the round-trip decode
    // already proved it's a valid Arrow array — only that it's geometry.
    match schema.field_with_name(GEOMETRY_COLUMN) {
        Ok(field) => {
            check_geometry_extension(name, field, issues);
            if !is_list_like(field.data_type()) {
                issues.push(format!(
                    "layer '{name}': geometry column is {:?}, expected a (FixedSize)List of coordinates",
                    field.data_type()
                ));
            }
        }
        Err(_) => issues.push(format!(
            "layer '{name}': missing required '{GEOMETRY_COLUMN}' column"
        )),
    }

    // (c) optional per-vertex columns, when present, must carry the expected
    // List<inner> types.
    check_optional_list(name, &schema, "vertex_time", &[DataType::UInt16, DataType::Int64], issues);
    check_optional_list(name, &schema, "vertex_value", &[DataType::Float32], issues);
    check_optional_list(name, &schema, "vertex_value_matrix", &[DataType::Float32], issues);
    check_optional_list(name, &schema, "triangles", &[DataType::UInt32], issues);

    // (d) every remaining (property) column must be one of the two encodable
    // property shapes the producer emits — anything else is producer drift.
    for field in schema.fields() {
        let n = field.name().as_str();
        if is_reserved_column(n) {
            continue;
        }
        if !is_valid_property_type(field.data_type()) {
            issues.push(format!(
                "layer '{name}': property column '{n}' has type {:?}, expected Float64 or Dictionary<UInt16,Utf8>",
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

/// A reserved (non-property) column name the producer manages itself.
fn is_reserved_column(name: &str) -> bool {
    matches!(
        name,
        "id" | "start_time"
            | "end_time"
            | "geometry"
            | "vertex_time"
            | "vertex_value"
            | "vertex_value_matrix"
            | "triangles"
    )
}

/// True for the list shapes geometry uses (`List` or `FixedSizeList`).
fn is_list_like(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
    )
}

/// A property column is either a plain `Float64` (numeric) or a
/// `Dictionary<UInt16, Utf8>` (categorical) — the only two shapes `encode_layer`
/// emits.
fn is_valid_property_type(dt: &DataType) -> bool {
    match dt {
        DataType::Float64 => true,
        DataType::Dictionary(k, v) => {
            matches!(k.as_ref(), DataType::UInt16) && matches!(v.as_ref(), DataType::Utf8)
        }
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
    let mut parts: Vec<String> = layers
        .iter()
        .map(|layer| {
            let cols: Vec<String> = layer
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
                    format!("{}:{:?}{ext}", f.name(), f.data_type())
                })
                .collect();
            format!("{}{{{}}}", layer.name, cols.join(","))
        })
        .collect();
    parts.sort();
    parts.join("|")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;
    use stt_core::arrow_tile::{
        decode_tile, encode_tile, ColumnarLayer, GeometryColumn, PropertyColumn,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    /// A well-formed point layer (id/start/end/geometry + one numeric + one
    /// categorical property), encoded then decoded so the test exercises the
    /// real producer schema.
    fn good_point_layer() -> ColumnarLayer {
        ColumnarLayer {
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
                ("speed".into(), PropertyColumn::Numeric(vec![Some(1.0), None, Some(3.0)])),
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

    #[test]
    fn well_formed_tile_passes_schema_check() {
        let decoded = decode(&[good_point_layer()]);
        let issues = check_tile_schema(&decoded);
        assert!(issues.is_empty(), "unexpected schema issues: {issues:?}");
    }

    #[test]
    fn line_layer_with_vertex_columns_passes() {
        // A line layer carrying u16-delta vertex_time and Float32 vertex_value —
        // both optional List columns the contract accepts.
        let layer = ColumnarLayer {
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
        let issues = check_tile_schema(&decode(&[layer]));
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
        let layers = vec![DecodedLayer { name: "broken".into(), batch }];

        let issues = check_tile_schema(&layers);
        assert!(
            issues.iter().any(|i| i.contains("missing required column 'start_time'")),
            "issues were: {issues:?}"
        );
        assert!(
            issues.iter().any(|i| i.contains("missing required 'geometry'")),
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
        let batch =
            RecordBatch::try_new(schema, vec![id, start, end, geom]).unwrap();
        let issues = check_tile_schema(&[DecodedLayer { name: "drift".into(), batch }]);
        assert!(
            issues.iter().any(|i| i.contains("column 'id'") && i.contains("expected UInt64")),
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
        let issues = check_tile_schema(&[DecodedLayer { name: "nogeo".into(), batch }]);
        assert!(
            issues.iter().any(|i| i.contains("extension name")),
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
