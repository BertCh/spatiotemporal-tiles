//! STAC Item emission (`stt-build --stac`) — the discovery layer for a packed
//! dataset.
//!
//! [STAC](https://stacspec.org/) catalogs assets; it does not constrain their
//! format, so an STT dataset needs no standards negotiation to be discoverable
//! by an existing catalog, browser or `pystac` reader. The Item is derived
//! MECHANICALLY from the manifest — `metadata.bounds` → `bbox`/`geometry`,
//! `metadata.time_range` → `start_datetime`/`end_datetime`, `manifest.json`
//! itself as the single data asset — exactly as promised by
//! `docs/spec/stt-packed-format.md` §10.3, which is the normative shape this
//! module implements. Nothing here consults the input data or the CLI flags:
//! the same manifest always yields the same Item, so an Item can be
//! regenerated for an already-published dataset.

use serde_json::{json, Map, Value};
use stt_core::metadata::Metadata;

/// STAC spec version the emitted Item declares.
///
/// Pinned to 1.0.0 rather than tracking the newest release: it is the version
/// every deployed catalog/validator accepts, and the Item shape we emit
/// (`datetime: null` + `start_datetime`/`end_datetime`) is valid unchanged in
/// later versions.
pub const STAC_VERSION: &str = "1.0.0";

/// Media type of the asset the Item points at (the packed dataset manifest).
const MANIFEST_MEDIA_TYPE: &str = "application/json";

/// Relative href of the data asset. RELATIVE, not absolute: STAC resolves an
/// asset href against the Item's own location, and the Item is written INTO the
/// dataset directory beside `manifest.json`, so the pair stays correct wherever
/// the directory is later published. An absolute URL would have to be guessed
/// at build time and would rot the moment the dataset moved.
const MANIFEST_HREF: &str = "./manifest.json";

/// Format a Unix-millisecond instant as RFC 3339 in UTC (`...Z`), the encoding
/// STAC requires for `datetime` / `start_datetime` / `end_datetime`.
///
/// Sub-second precision is kept: a dataset's time range routinely lands
/// mid-second, and truncating it would advertise an interval that does not
/// contain its own last feature.
fn rfc3339_utc(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .unwrap_or_else(|| chrono::DateTime::from_timestamp_nanos(0))
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// The `id` for a dataset: its declared `metadata.name`, falling back to the
/// output directory's stem.
///
/// A STAC Item id must be non-empty and stable, and `metadata.name` is empty
/// for a build that never passed `--name` — falling back to the directory keeps
/// the Item valid instead of emitting `"id": ""`.
pub fn item_id(metadata: &Metadata, out_dir: &std::path::Path) -> String {
    let name = metadata.name.trim();
    if !name.is_empty() {
        return name.to_string();
    }
    out_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "stt-dataset".to_string())
}

/// Build the STAC Item for a finished packed dataset.
///
/// `manifest` is the manifest as written; `out_dir` only supplies the id
/// fallback. Every value comes from the manifest, so this function is pure and
/// the Item is reproducible from a published dataset alone.
pub fn stac_item(manifest: &stt_core::Manifest, out_dir: &std::path::Path) -> Value {
    let m = &manifest.metadata;
    let b = &m.bounds;

    // bbox order is STAC/GeoJSON's [west, south, east, north].
    let bbox = json!([b.min_lon, b.min_lat, b.max_lon, b.max_lat]);
    // Counter-clockwise exterior ring, closed (first == last) — the RFC 7946
    // winding for an exterior ring, and the shape §10.3 documents.
    let geometry = json!({
        "type": "Polygon",
        "coordinates": [[
            [b.min_lon, b.min_lat],
            [b.max_lon, b.min_lat],
            [b.max_lon, b.max_lat],
            [b.min_lon, b.max_lat],
            [b.min_lon, b.min_lat],
        ]]
    });

    let mut properties = Map::new();
    // `datetime: null` is REQUIRED to be present (not omitted) when the Item
    // carries a range instead of an instant; a reader that finds neither
    // `datetime` nor the pair rejects the Item.
    properties.insert("datetime".into(), Value::Null);
    properties.insert(
        "start_datetime".into(),
        json!(rfc3339_utc(m.time_range.start as i64)),
    );
    properties.insert(
        "end_datetime".into(),
        json!(rfc3339_utc(m.time_range.end as i64)),
    );
    if !m.description.trim().is_empty() {
        properties.insert("description".into(), json!(m.description));
    }
    // STAC's common metadata has no field for these, so they take the `stt:`
    // prefix rather than squatting on a core name a future STAC version might
    // define differently.
    properties.insert("stt:format_version".into(), json!(manifest.format_version));
    properties.insert("stt:min_zoom".into(), json!(m.min_zoom));
    properties.insert("stt:max_zoom".into(), json!(m.max_zoom));
    properties.insert("stt:tile_count".into(), json!(m.tile_count));
    // Index-weighted: a feature in N tiles counts N times. The distinct count
    // is the user-facing "N features" number, and is absent on older archives
    // — emitting both, clearly named, beats silently publishing the wrong one.
    properties.insert("stt:feature_count".into(), json!(m.feature_count));
    if let Some(distinct) = m.distinct_feature_count {
        properties.insert("stt:distinct_feature_count".into(), json!(distinct));
    }
    properties.insert("stt:layers".into(), json!(m.layers));
    properties.insert("stt:temporal_bucket_ms".into(), json!(m.temporal_bucket_ms));
    if !manifest.capabilities.is_empty() {
        // Required-to-understand declarations (spec §3.1) — surfaced so a
        // catalog can tell a client it will need a newer reader BEFORE it
        // fetches a single tile.
        properties.insert("stt:capabilities".into(), json!(manifest.capabilities));
    }

    let mut assets = Map::new();
    let mut stt_asset = Map::new();
    stt_asset.insert("href".into(), json!(MANIFEST_HREF));
    stt_asset.insert("type".into(), json!(MANIFEST_MEDIA_TYPE));
    stt_asset.insert("roles".into(), json!(["data", "stt"]));
    stt_asset.insert("title".into(), json!("STT packed dataset (manifest)"));
    assets.insert("stt".into(), Value::Object(stt_asset));

    let mut item = Map::new();
    item.insert("type".into(), json!("Feature"));
    item.insert("stac_version".into(), json!(STAC_VERSION));
    item.insert("id".into(), json!(item_id(m, out_dir)));
    item.insert("bbox".into(), bbox);
    item.insert("geometry".into(), geometry);
    item.insert("properties".into(), Value::Object(properties));
    // REQUIRED by the Item spec even when there is nothing to link: a
    // standalone dataset has no parent catalog, and an ABSENT `links` fails
    // validation where an empty one passes. A publisher adds `self`/`parent`
    // when the Item is placed in a catalog.
    item.insert("links".into(), json!([]));
    item.insert("assets".into(), Value::Object(assets));
    if !m.attribution.trim().is_empty() {
        // Attribution rides in properties, where STAC keeps licensing/credit.
        if let Some(Value::Object(p)) = item.get_mut("properties") {
            p.insert(
                "providers".into(),
                json!([{
                    "name": m.attribution,
                    "roles": ["producer"],
                }]),
            );
        }
    }
    Value::Object(item)
}

/// Convenience: the pretty-printed JSON bytes, newline-terminated, as written
/// to `stac.json`.
pub fn stac_item_bytes(manifest: &stt_core::Manifest, out_dir: &std::path::Path) -> Vec<u8> {
    let mut bytes = serde_json::to_vec_pretty(&stac_item(manifest, out_dir))
        .expect("STAC Item is plain JSON values");
    bytes.push(b'\n');
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::types::{BoundingBox, TimeRange};

    fn manifest() -> stt_core::Manifest {
        let mut metadata = Metadata {
            name: "drifters".to_string(),
            description: "Global drifter trajectories".to_string(),
            attribution: "NOAA GDP".to_string(),
            bounds: BoundingBox::new(-180.0, -78.4, 180.0, 81.5),
            // 1979-02-15T00:00:00Z .. 2022-10-04T00:00:00Z (spec §10.3's example).
            time_range: TimeRange::new(287_884_800_000, 1_664_841_600_000),
            min_zoom: 0,
            max_zoom: 4,
            tile_count: 1234,
            feature_count: 98_765,
            distinct_feature_count: Some(54_321),
            ..Metadata::default()
        };
        metadata.layers = vec!["default".to_string()];
        metadata.temporal_bucket_ms = 86_400_000;
        stt_core::Manifest {
            format: "stt-packed".to_string(),
            format_version: 2,
            capabilities: Vec::new(),
            schemas: Vec::new(),
            compression: "zstd".to_string(),
            blob_ordering: Some("time-major".to_string()),
            directory: stt_core::pack::DirectoryRef {
                key: "index/deadbeef.sttd".to_string(),
                length: 1,
                directory_version: 5,
                encoding: None,
                layout: None,
                root_length: None,
                page_count: None,
                page_entries: None,
            },
            packs: Vec::new(),
            metadata,
        }
    }

    /// Every field the STAC Item spec marks REQUIRED must be present, with the
    /// right JSON type. This is the structural validation gate: an Item missing
    /// `links` (the field the spec's own §10.3 example forgot) or carrying a
    /// bare `datetime: null` with no range is rejected by real validators.
    #[test]
    fn emits_every_required_stac_item_field() {
        let item = stac_item(&manifest(), std::path::Path::new("/tmp/drifters"));

        assert_eq!(item["type"], "Feature");
        assert_eq!(item["stac_version"], STAC_VERSION);
        assert_eq!(item["id"], "drifters");
        assert!(item["links"].is_array(), "links is REQUIRED (may be empty)");
        assert!(item["assets"].is_object());
        assert!(item["properties"].is_object());

        // bbox: 4 numbers, [west, south, east, north].
        let bbox = item["bbox"].as_array().expect("bbox array");
        assert_eq!(bbox.len(), 4);
        assert!(bbox.iter().all(|v| v.is_f64()));
        assert_eq!(bbox[0], -180.0);
        assert_eq!(bbox[3], 81.5);

        // geometry: a closed 5-point Polygon ring matching the bbox.
        assert_eq!(item["geometry"]["type"], "Polygon");
        let ring = item["geometry"]["coordinates"][0]
            .as_array()
            .expect("exterior ring");
        assert_eq!(ring.len(), 5, "ring must be closed");
        assert_eq!(ring[0], ring[4], "first vertex must equal last");

        // Temporal: `datetime` PRESENT and null, with the range beside it.
        let props = &item["properties"];
        assert!(props.get("datetime").is_some(), "datetime key must exist");
        assert!(props["datetime"].is_null());
        assert_eq!(props["start_datetime"], "1979-02-15T00:00:00.000Z");
        assert_eq!(props["end_datetime"], "2022-10-04T00:00:00.000Z");

        // The asset a reader follows back into the §6 reader flow.
        let asset = &item["assets"]["stt"];
        assert_eq!(asset["href"], "./manifest.json");
        assert_eq!(asset["type"], "application/json");
        assert_eq!(asset["roles"], json!(["data", "stt"]));
    }

    /// The counts a catalog shows come from the manifest, and the two feature
    /// counts mean different things — both are emitted under distinct names so
    /// a catalog can never present the tile-weighted total as "N features".
    #[test]
    fn carries_the_manifest_counts_under_the_stt_prefix() {
        let item = stac_item(&manifest(), std::path::Path::new("/tmp/drifters"));
        let props = &item["properties"];
        assert_eq!(props["stt:tile_count"], 1234);
        assert_eq!(props["stt:feature_count"], 98_765);
        assert_eq!(props["stt:distinct_feature_count"], 54_321);
        assert_eq!(props["stt:min_zoom"], 0);
        assert_eq!(props["stt:max_zoom"], 4);
        assert_eq!(props["stt:format_version"], 2);
        assert_eq!(props["stt:layers"], json!(["default"]));
    }

    /// An unnamed build must still produce a valid, non-empty id.
    #[test]
    fn id_falls_back_to_the_output_directory() {
        let mut m = manifest();
        m.metadata.name = String::new();
        let item = stac_item(&m, std::path::Path::new("/data/out/quakes"));
        assert_eq!(item["id"], "quakes");
    }

    /// A dataset with no distinct count (older archive / v1 kill switch) must
    /// omit the key rather than publish a wrong number.
    #[test]
    fn omits_absent_optional_fields() {
        let mut m = manifest();
        m.metadata.distinct_feature_count = None;
        m.metadata.description = String::new();
        m.metadata.attribution = String::new();
        let item = stac_item(&m, std::path::Path::new("/tmp/x"));
        let props = item["properties"].as_object().unwrap();
        assert!(!props.contains_key("stt:distinct_feature_count"));
        assert!(!props.contains_key("description"));
        assert!(!props.contains_key("providers"));
    }

    /// Sub-second precision survives: an interval truncated to whole seconds
    /// would exclude its own last feature.
    #[test]
    fn keeps_sub_second_precision_in_the_interval() {
        let mut m = manifest();
        m.metadata.time_range = TimeRange::new(1_700_000_000_123, 1_700_000_009_987);
        let item = stac_item(&m, std::path::Path::new("/tmp/x"));
        assert_eq!(
            item["properties"]["start_datetime"],
            "2023-11-14T22:13:20.123Z"
        );
        assert_eq!(
            item["properties"]["end_datetime"],
            "2023-11-14T22:13:29.987Z"
        );
    }

    /// The written bytes are pretty JSON that round-trips.
    #[test]
    fn bytes_are_newline_terminated_parseable_json() {
        let bytes = stac_item_bytes(&manifest(), std::path::Path::new("/tmp/drifters"));
        assert_eq!(*bytes.last().unwrap(), b'\n');
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid JSON");
        assert_eq!(parsed["id"], "drifters");
    }
}
