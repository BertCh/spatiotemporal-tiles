//! Row properties, held columnar so a build's memory scales with VALUES rather
//! than with rows × a JSON map.
//!
//! The parquet reader used to materialise one `serde_json::Map` per row. That
//! map is a `BTreeMap`, and a `BTreeMap` allocates a full leaf node — ~630
//! bytes for `(String, Value)` — whether it holds one key or eleven, plus an
//! owned `String` for every key on every row, plus an `Arc` to carry it. Peak
//! memory therefore scaled at roughly a kilobyte per feature no matter how
//! small the properties were, which is why a 106 M-point national MRMS volume
//! could not be tiled at all on a 36 GB machine — in-memory AND under
//! `--streaming`, since both hold the whole `Vec<ParsedFeature>` resident.
//!
//! [`PropertyTable`] stores each property as one typed column per record batch
//! and hands each feature a `(table, row)` handle. Property names are interned
//! once per batch instead of cloned once per row.
//!
//! **The Arrow → JSON type mapping is deliberately not re-derived here.** Values
//! enter through [`PropertyTableBuilder::push`], which takes the
//! `serde_json::Value` that `input::extract_property_value` already produced,
//! so that contract keeps exactly one definition. The transient `Value` is
//! dropped immediately instead of being retained — that is the whole trick.
//! Round-tripping is exact because the number variants are preserved rather
//! than collapsed to `f64`: `serde_json` prints an integer-backed `5` as `"5"`
//! and a float-backed `5.0` as `"5.0"`, and the categorical column path
//! stringifies numbers, so collapsing them would silently rewrite values.

use std::sync::Arc;

/// One property value, borrowed from either an owned JSON map or a columnar
/// cell.
///
/// The three numeric variants mirror `serde_json::Number`'s own internal split
/// (`PosInt` / `NegInt` / `Float`) so [`PropValue::to_json`] reproduces the
/// exact `Value` the row was built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropValue<'a> {
    F64(f64),
    I64(i64),
    U64(u64),
    Bool(bool),
    Str(&'a str),
    /// Anything else an owned JSON map can carry (arrays, nested objects).
    /// Never produced by the columnar path — Arrow property columns are scalar.
    Json(&'a serde_json::Value),
}

impl<'a> PropValue<'a> {
    /// Borrow a JSON value. `None` for `null`, matching the accumulator's
    /// "skip nulls" rule — in the columnar path a null cell is simply absent.
    pub fn from_json(v: &'a serde_json::Value) -> Option<Self> {
        match v {
            serde_json::Value::Null => None,
            serde_json::Value::Bool(b) => Some(PropValue::Bool(*b)),
            serde_json::Value::Number(n) => {
                // `as_i64` returns None for a float-backed Number, so a 5.0
                // stays F64 and keeps printing as "5.0".
                if let Some(i) = n.as_i64() {
                    Some(PropValue::I64(i))
                } else if let Some(u) = n.as_u64() {
                    Some(PropValue::U64(u))
                } else {
                    n.as_f64().map(PropValue::F64)
                }
            }
            serde_json::Value::String(s) => Some(PropValue::Str(s)),
            other => Some(PropValue::Json(other)),
        }
    }

    /// Rebuild the owning `serde_json::Value`. Used where the existing
    /// accumulator logic is easier to reuse than to mirror.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            PropValue::F64(v) => serde_json::json!(v),
            PropValue::I64(v) => serde_json::json!(v),
            PropValue::U64(v) => serde_json::json!(v),
            PropValue::Bool(b) => serde_json::json!(b),
            PropValue::Str(s) => serde_json::json!(s),
            PropValue::Json(v) => (*v).clone(),
        }
    }

    /// True for a real JSON number (not a numeric-looking string).
    pub fn is_number(&self) -> bool {
        matches!(
            self,
            PropValue::F64(_) | PropValue::I64(_) | PropValue::U64(_)
        )
    }

    /// Mirrors `serde_json::Value::as_f64`: real numbers only. A
    /// numeric-looking STRING stays `None` here — that coercion is
    /// `columnar::value_as_f64`'s job, and the zoom-bound reader deliberately
    /// does not want it.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PropValue::F64(v) => Some(*v),
            PropValue::I64(v) => Some(*v as f64),
            PropValue::U64(v) => Some(*v as f64),
            PropValue::Bool(_) | PropValue::Str(_) => None,
            PropValue::Json(v) => v.as_f64(),
        }
    }

    /// The string payload, for a string value only.
    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            PropValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One property column for a batch of rows.
///
/// `Empty` is the pre-type state: a column that has only seen nulls so far. The
/// first non-null value fixes the type; anything that disagrees later demotes
/// the column to `Json`, which is lossless but pays the old per-row cost. Arrow
/// columns are typed, so in practice that demotion never fires — it exists so a
/// surprising input degrades instead of corrupting.
#[derive(Debug)]
enum Column {
    Empty,
    F64(Vec<f64>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
    /// UTF-8 pool plus per-row end offsets — one allocation for the batch
    /// rather than one `String` per row.
    Str {
        offsets: Vec<u32>,
        data: String,
    },
    Json(Vec<serde_json::Value>),
}

/// A property table for one record batch: interned names, typed columns, and a
/// per-column validity mask.
#[derive(Debug)]
pub struct PropertyTable {
    names: Vec<String>,
    columns: Vec<Column>,
    /// `valid[col][row]` — false for a null (absent) cell.
    valid: Vec<Vec<bool>>,
    rows: usize,
}

impl PropertyTable {
    /// Number of rows the table describes.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Property names, in the order they were declared to the builder.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// True when this row carries at least one non-null property.
    ///
    /// The reader uses it to keep the old contract that a row with nothing on
    /// it gets `None` rather than an empty handle. The accumulator treats the
    /// two identically, so this is belt-and-braces — but it costs one pass over
    /// a handful of validity bits and removes a class of "did the refactor
    /// change what an empty row means" question entirely.
    pub fn row_has_values(&self, row: usize) -> bool {
        self.valid
            .iter()
            .any(|v| v.get(row).copied().unwrap_or(false))
    }

    fn value(&self, col: usize, row: usize) -> Option<PropValue<'_>> {
        if !*self.valid.get(col)?.get(row)? {
            return None;
        }
        Some(match &self.columns[col] {
            Column::Empty => return None,
            Column::F64(v) => PropValue::F64(*v.get(row)?),
            Column::I64(v) => PropValue::I64(*v.get(row)?),
            Column::U64(v) => PropValue::U64(*v.get(row)?),
            Column::Bool(v) => PropValue::Bool(*v.get(row)?),
            Column::Str { offsets, data } => {
                let end = *offsets.get(row)? as usize;
                let start = if row == 0 {
                    0
                } else {
                    offsets[row - 1] as usize
                };
                PropValue::Str(data.get(start..end)?)
            }
            Column::Json(v) => PropValue::from_json(v.get(row)?)?,
        })
    }
}

/// Builds one [`PropertyTable`] per record batch.
///
/// Columns are declared up front (from the parquet schema) and then filled
/// row-major by the reader: for each row, one [`push`](Self::push) per declared
/// column, in order. That matches the reader's existing loop shape, so the call
/// site stays a straight substitution.
#[derive(Debug)]
pub struct PropertyTableBuilder {
    names: Vec<String>,
    columns: Vec<Column>,
    valid: Vec<Vec<bool>>,
    rows: usize,
}

impl PropertyTableBuilder {
    /// Declare the columns, in the order rows will be pushed.
    pub fn new(names: Vec<String>, capacity: usize) -> Self {
        let n = names.len();
        Self {
            names,
            columns: (0..n).map(|_| Column::Empty).collect(),
            valid: (0..n).map(|_| Vec::with_capacity(capacity)).collect(),
            rows: 0,
        }
    }

    /// True when no property columns exist at all — the caller then skips the
    /// table entirely and every feature keeps `None` properties.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Append one cell. `value` is whatever `extract_property_value` returned:
    /// `None` (or a JSON `null`) records a null.
    pub fn push(&mut self, col: usize, value: Option<&serde_json::Value>) {
        let Some(v) = value.and_then(PropValue::from_json) else {
            self.push_null(col);
            return;
        };
        let row = self.valid[col].len();
        let column = &mut self.columns[col];
        // Fix the type on the first non-null value.
        if matches!(column, Column::Empty) {
            *column = match v {
                PropValue::F64(_) => Column::F64(Vec::new()),
                PropValue::I64(_) => Column::I64(Vec::new()),
                PropValue::U64(_) => Column::U64(Vec::new()),
                PropValue::Bool(_) => Column::Bool(Vec::new()),
                PropValue::Str(_) => Column::Str {
                    offsets: Vec::new(),
                    data: String::new(),
                },
                PropValue::Json(_) => Column::Json(Vec::new()),
            };
        }
        // ALWAYS pad first. `push_null` deliberately leaves the typed column
        // short (a null cell stores no value), so without this a null anywhere
        // before this row would shift every later index by one and silently
        // hand out a neighbour's value.
        pad(column, row);
        let placed = match (&mut *column, v) {
            (Column::F64(out), PropValue::F64(x)) => {
                out.push(x);
                true
            }
            (Column::I64(out), PropValue::I64(x)) => {
                out.push(x);
                true
            }
            (Column::U64(out), PropValue::U64(x)) => {
                out.push(x);
                true
            }
            (Column::Bool(out), PropValue::Bool(x)) => {
                out.push(x);
                true
            }
            (Column::Str { offsets, data }, PropValue::Str(s)) => {
                data.push_str(s);
                offsets.push(data.len() as u32);
                true
            }
            (Column::Json(out), other) => {
                out.push(other.to_json());
                true
            }
            _ => false,
        };
        if !placed {
            // Type disagreement — demote to JSON, losslessly, and retry.
            demote_to_json(column, row);
            if let Column::Json(out) = column {
                out.push(v.to_json());
            }
        }
        self.valid[col].push(true);
    }

    fn push_null(&mut self, col: usize) {
        let row = self.valid[col].len();
        pad(&mut self.columns[col], row);
        self.valid[col].push(false);
    }

    /// Close the current row. Columns the caller never pushed are padded, so a
    /// short row can't shear the columns out of alignment.
    pub fn end_row(&mut self) {
        self.rows += 1;
        for col in 0..self.names.len() {
            while self.valid[col].len() < self.rows {
                self.push_null(col);
            }
        }
    }

    /// Seal the batch.
    pub fn finish(self) -> PropertyTable {
        PropertyTable {
            names: self.names,
            columns: self.columns,
            valid: self.valid,
            rows: self.rows,
        }
    }
}

/// Grow a typed column to exactly `row` entries, so the next value pushed lands
/// at index `row`. Placeholder values are never read — `valid[col][row]` is
/// false wherever `pad` filled in.
fn pad(column: &mut Column, row: usize) {
    match column {
        Column::Empty => {}
        Column::F64(v) => v.resize(row, 0.0),
        Column::I64(v) => v.resize(row, 0),
        Column::U64(v) => v.resize(row, 0),
        Column::Bool(v) => v.resize(row, false),
        Column::Str { offsets, data } => {
            // A null string contributes no bytes, so its end offset is simply
            // where the pool currently stands — that keeps the next real row's
            // start offset correct.
            let end = data.len() as u32;
            offsets.resize(row, end);
        }
        Column::Json(v) => v.resize(row, serde_json::Value::Null),
    }
}

/// Rewrite a typed column as JSON, preserving every value already stored.
fn demote_to_json(column: &mut Column, row: usize) {
    let mut out: Vec<serde_json::Value> = Vec::with_capacity(row + 1);
    match column {
        Column::Empty => {}
        Column::F64(v) => out.extend(v.iter().map(|x| serde_json::json!(x))),
        Column::I64(v) => out.extend(v.iter().map(|x| serde_json::json!(x))),
        Column::U64(v) => out.extend(v.iter().map(|x| serde_json::json!(x))),
        Column::Bool(v) => out.extend(v.iter().map(|x| serde_json::json!(x))),
        Column::Str { offsets, data } => {
            let mut start = 0usize;
            for &end in offsets.iter() {
                out.push(serde_json::json!(&data[start..end as usize]));
                start = end as usize;
            }
        }
        Column::Json(v) => out.append(v),
    }
    out.resize(row, serde_json::Value::Null);
    *column = Column::Json(out);
}

/// A feature's properties: either an owned JSON map (DB inputs, clipped
/// segments, synthesised features) or a handle into a batch's columnar table.
#[derive(Debug, Clone)]
pub enum FeatureProperties {
    Owned(Arc<serde_json::Map<String, serde_json::Value>>),
    Row { table: Arc<PropertyTable>, row: u32 },
}

impl FeatureProperties {
    /// Wrap an owned map, or `None` when it holds nothing (an empty map costs
    /// an allocation and produces no columns).
    pub fn from_map(map: serde_json::Map<String, serde_json::Value>) -> Option<Self> {
        (!map.is_empty()).then(|| FeatureProperties::Owned(Arc::new(map)))
    }

    /// Materialise this row as an owned JSON map.
    ///
    /// Allocates — deliberately not on the tiling path. It exists so two
    /// features can be compared for equality regardless of which
    /// representation each happens to use (the reproducibility harness does
    /// exactly that), and for diagnostics.
    /// Nulls are dropped, in BOTH arms. A columnar null cell is absent by
    /// construction, so an owned map that kept a `Value::Null` would compare
    /// unequal to the identical row read columnar — which is exactly the
    /// comparison this method exists to make. The owned path really does
    /// produce them: the DB readers map a non-finite float to `Value::Null`
    /// via `json_number_or_null` and insert it. Every other accessor here
    /// (`get`, `iter`, `len`) already treats a null as absent; this arm was
    /// the one that did not.
    pub fn to_map(&self) -> serde_json::Map<String, serde_json::Value> {
        self.iter()
            .map(|(k, v)| (k.to_string(), v.to_json()))
            .collect()
    }

    /// Insert or replace one property.
    ///
    /// A columnar handle is materialised into an owned map first — the column
    /// store is shared by every row in its batch and must never be edited
    /// through one row. Nothing on the tiling path mutates properties; this is
    /// for producers that synthesise or patch features.
    pub fn insert(&mut self, key: &str, value: serde_json::Value) {
        if let FeatureProperties::Row { .. } = self {
            let mut map = serde_json::Map::new();
            for (k, v) in self.iter() {
                map.insert(k.to_string(), v.to_json());
            }
            *self = FeatureProperties::Owned(Arc::new(map));
        }
        if let FeatureProperties::Owned(map) = self {
            Arc::make_mut(map).insert(key.to_string(), value);
        }
    }

    /// Number of non-null properties on this row.
    ///
    /// Matches what `serde_json::Map::len()` reported before: the reader only
    /// ever inserted non-null values, so a null was already absent from the
    /// count.
    pub fn len(&self) -> usize {
        match self {
            FeatureProperties::Owned(map) => map.values().filter(|v| !v.is_null()).count(),
            FeatureProperties::Row { .. } => self.iter().count(),
        }
    }

    /// True when the row carries no non-null property at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Look up one property by name.
    pub fn get(&self, key: &str) -> Option<PropValue<'_>> {
        match self {
            FeatureProperties::Owned(map) => map.get(key).and_then(PropValue::from_json),
            FeatureProperties::Row { table, row } => {
                let col = table.names.iter().position(|n| n == key)?;
                table.value(col, *row as usize)
            }
        }
    }

    /// Iterate `(name, value)` for every non-null property on this row.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (&str, PropValue<'_>)> + '_> {
        match self {
            FeatureProperties::Owned(map) => Box::new(
                map.iter()
                    .filter_map(|(k, v)| PropValue::from_json(v).map(|p| (k.as_str(), p))),
            ),
            FeatureProperties::Row { table, row } => {
                let row = *row as usize;
                Box::new(
                    table.names.iter().enumerate().filter_map(move |(i, name)| {
                        table.value(i, row).map(|v| (name.as_str(), v))
                    }),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(names: &[&str], rows: &[Vec<Option<serde_json::Value>>]) -> Arc<PropertyTable> {
        let mut b =
            PropertyTableBuilder::new(names.iter().map(|s| s.to_string()).collect(), rows.len());
        for row in rows {
            for (i, v) in row.iter().enumerate() {
                b.push(i, v.as_ref());
            }
            b.end_row();
        }
        Arc::new(b.finish())
    }

    fn row_props(t: &Arc<PropertyTable>, row: u32) -> FeatureProperties {
        FeatureProperties::Row {
            table: Arc::clone(t),
            row,
        }
    }

    /// The property that motivated the whole module: a value must come back out
    /// as the same `serde_json::Value` that went in, because the categorical
    /// column path stringifies it. An integer-backed 5 and a float-backed 5.0
    /// print differently and must stay distinct.
    #[test]
    fn number_variants_round_trip_exactly() {
        let t = table(
            &["i", "f", "big"],
            &[vec![
                Some(serde_json::json!(5i64)),
                Some(serde_json::json!(5.0f64)),
                Some(serde_json::json!(u64::MAX)),
            ]],
        );
        let p = row_props(&t, 0);
        assert_eq!(p.get("i").unwrap().to_json().to_string(), "5");
        assert_eq!(p.get("f").unwrap().to_json().to_string(), "5.0");
        assert_eq!(
            p.get("big").unwrap().to_json().to_string(),
            u64::MAX.to_string()
        );
    }

    #[test]
    fn strings_share_one_pool_and_index_correctly() {
        let t = table(
            &["s"],
            &[
                vec![Some(serde_json::json!("alpha"))],
                vec![Some(serde_json::json!("beta"))],
                vec![Some(serde_json::json!(""))],
                vec![Some(serde_json::json!("delta"))],
            ],
        );
        for (row, want) in ["alpha", "beta", "", "delta"].iter().enumerate() {
            assert_eq!(
                row_props(&t, row as u32).get("s").unwrap().as_str(),
                Some(*want)
            );
        }
    }

    /// Nulls are absent, not empty — `observe` skips them and `push_row` must
    /// see `None`, exactly as it did for a missing JSON key.
    #[test]
    fn nulls_are_absent_and_keep_rows_aligned() {
        let t = table(
            &["a", "b"],
            &[
                vec![None, Some(serde_json::json!(1i64))],
                vec![Some(serde_json::json!("x")), None],
                vec![None, None],
                vec![Some(serde_json::json!("y")), Some(serde_json::json!(2i64))],
            ],
        );
        assert!(row_props(&t, 0).get("a").is_none());
        assert_eq!(row_props(&t, 1).get("a").unwrap().as_str(), Some("x"));
        assert!(row_props(&t, 2).get("a").is_none());
        assert_eq!(row_props(&t, 3).get("a").unwrap().as_str(), Some("y"));
        assert_eq!(row_props(&t, 0).get("b"), Some(PropValue::I64(1)));
        assert!(row_props(&t, 1).get("b").is_none());
        assert_eq!(row_props(&t, 3).get("b"), Some(PropValue::I64(2)));
        // A row of all nulls yields no pairs at all.
        assert_eq!(row_props(&t, 2).iter().count(), 0);
        assert_eq!(row_props(&t, 3).iter().count(), 2);
    }

    /// A leading run of nulls must not shift the typed column's indices.
    #[test]
    fn leading_nulls_do_not_shift_a_typed_column() {
        let t = table(
            &["v"],
            &[
                vec![None],
                vec![None],
                vec![Some(serde_json::json!(7.5f64))],
                vec![Some(serde_json::json!(8.5f64))],
            ],
        );
        assert!(row_props(&t, 0).get("v").is_none());
        assert!(row_props(&t, 1).get("v").is_none());
        assert_eq!(row_props(&t, 2).get("v"), Some(PropValue::F64(7.5)));
        assert_eq!(row_props(&t, 3).get("v"), Some(PropValue::F64(8.5)));
    }

    /// Arrow columns are typed so this should never fire in practice, but a
    /// mixed column must degrade losslessly rather than corrupt earlier rows.
    #[test]
    fn mixed_types_demote_to_json_without_losing_earlier_rows() {
        let t = table(
            &["m"],
            &[
                vec![Some(serde_json::json!(1i64))],
                vec![Some(serde_json::json!("two"))],
                vec![Some(serde_json::json!(3i64))],
            ],
        );
        assert_eq!(row_props(&t, 0).get("m"), Some(PropValue::I64(1)));
        assert_eq!(row_props(&t, 1).get("m").unwrap().as_str(), Some("two"));
        assert_eq!(row_props(&t, 2).get("m"), Some(PropValue::I64(3)));
    }

    #[test]
    fn owned_and_columnar_agree() {
        let mut map = serde_json::Map::new();
        map.insert("n".into(), serde_json::json!(2.5f64));
        map.insert("s".into(), serde_json::json!("hi"));
        let owned = FeatureProperties::from_map(map).unwrap();
        let t = table(
            &["n", "s"],
            &[vec![
                Some(serde_json::json!(2.5f64)),
                Some(serde_json::json!("hi")),
            ]],
        );
        let columnar = row_props(&t, 0);
        for key in ["n", "s"] {
            assert_eq!(owned.get(key), columnar.get(key), "key {key}");
        }
        let mut a: Vec<_> = owned.iter().map(|(k, v)| (k.to_string(), v)).collect();
        let mut b: Vec<_> = columnar.iter().map(|(k, v)| (k.to_string(), v)).collect();
        a.sort_by(|x, y| x.0.cmp(&y.0));
        b.sort_by(|x, y| x.0.cmp(&y.0));
        assert_eq!(a, b);
    }

    #[test]
    fn empty_map_is_no_properties() {
        assert!(FeatureProperties::from_map(serde_json::Map::new()).is_none());
    }
}

#[cfg(test)]
mod size_guard {
    /// The in-memory cost of one loaded feature is the build's hard ceiling:
    /// `generate_tiles_streaming` holds the whole `Vec<ParsedFeature>` resident
    /// for the entire build, so peak memory is at least
    /// `size_of::<ParsedFeature>() * rows` before a single tile exists.
    /// Printed rather than merely asserted so a regression shows its numbers.
    #[test]
    fn report_feature_footprint() {
        use std::mem::size_of;
        println!(
            "ParsedFeature            = {:>4} B",
            size_of::<crate::input::ParsedFeature>()
        );
        println!(
            "  geojson::Feature       = {:>4} B",
            size_of::<geojson::Feature>()
        );
        println!(
            "  geojson::Geometry      = {:>4} B",
            size_of::<geojson::Geometry>()
        );
        println!(
            "  geojson::Value         = {:>4} B",
            size_of::<geojson::Value>()
        );
        println!(
            "  Option<feature::Id>    = {:>4} B",
            size_of::<Option<geojson::feature::Id>>()
        );
        println!(
            "  Option<FeatureProps>   = {:>4} B",
            size_of::<Option<super::FeatureProperties>>()
        );
        println!(
            "  Option<Vec<u64>>       = {:>4} B",
            size_of::<Option<Vec<u64>>>()
        );
    }
}
