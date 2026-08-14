//! SH-2 acceptance pin — the declared `metadata.bounds` must CONTAIN every
//! vertex the archive decodes to, on a fixture that spans **many tiles at its
//! coarsest zoom**.
//!
//! # Why the fixture size is the whole test
//!
//! Every pre-existing bounds fixture in this repo is small enough that one tile
//! holds every feature. That is the only configuration in which per-tile row
//! indices coincide with feature ids and in which a tiler that never split a
//! feature can look correct — it is exactly how a 98.7 %-false-positive bug
//! survived a 30-test suite, and it is why the assertions below are written
//! against an archive whose *coarsest* zoom already needs several tiles.
//! [`multi_tile_precondition`] fails loudly if that ever stops being true, so a
//! future fixture edit cannot quietly turn these pins vacuous.
//!
//! # The defect, restated
//!
//! `metadata.bounds` used to be the min/max of feature ANCHORS. A parser sets a
//! feature's anchor to its geometric CENTROID, while `tiler::place_*` addresses
//! tiles by VERTEX. For anything wider than a point the two disagree, and they
//! disagree in the unsound direction: the declared box is *smaller* than the
//! data. Everything that pre-intersects a query box against `metadata.bounds`
//! (tile selection, frustum pre-culling, the showcase's opening camera) then
//! discards tiles that really do carry visible data, with no error anywhere.
//!
//! So the direction — declared ⊇ decoded — is asserted everywhere below, never
//! equality: a box that is too wide costs one wasted intersection test, a box
//! that is too tight loses data.
//!
//! # Relationship to `stt-validate` check 13
//!
//! Check 13 lives in the `spatiotemporal-tiles` binary and recomputes exactly
//! the quantity [`decoded_vertex_bbox`] computes here — via the same
//! `stt_core::metadata::FingerprintAccumulator`. These tests are the
//! builder-side half of that contract: they pin that an archive this crate
//! writes SATISFIES check 13 rather than merely that the check exists.

use std::collections::BTreeSet;

use geojson::{Feature, Geometry, Value as GeomValue};
use stt_build::input::{
    profile_features_with, BoundsMode, FeatureProfileOptions, ParsedFeature, BOUNDS_MODE_PROPERTY,
    DEFAULT_BOUNDS_MODE,
};
use stt_build::tiler::{generate_tiles_streaming, TileConfig};
use stt_core::metadata::{FingerprintAccumulator, Metadata};
use stt_core::types::BoundingBox;
use stt_core::{BlobOrdering, PackWriter, PackedReader};

// ---------------------------------------------------------------------------
// Fixture construction
// ---------------------------------------------------------------------------

/// A LineString feature whose anchor is its true centroid — the quantity every
/// real parser fills in (`parse_wkb_geometry` runs `geo`'s length-weighted
/// centroid), so the legacy under-statement is reproduced faithfully rather
/// than assumed.
fn line(coords: &[(f64, f64)], ts: u64) -> ParsedFeature {
    let positions: Vec<Vec<f64>> = coords.iter().map(|(x, y)| vec![*x, *y]).collect();
    let (cx, cy) = centroid_of_polyline(coords);
    ParsedFeature {
        home_zoom: None,
        geojson: Feature {
            bbox: None,
            geometry: Some(Geometry::new(GeomValue::LineString(positions))),
            id: None,
            properties: None,
            foreign_members: None,
        },
        shared_properties: None,
        timestamp: ts,
        end_timestamp: None,
        vertex_timestamps: None,
        vertex_values: None,
        vertex_value_matrix: None,
        lon: cx,
        lat: cy,
    }
}

/// Length-weighted centroid of an open polyline — `geo::Centroid` for a
/// `LineString`, spelled out so the fixture does not depend on that crate here.
fn centroid_of_polyline(coords: &[(f64, f64)]) -> (f64, f64) {
    let mut total = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for w in coords.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        let len = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        total += len;
        sx += len * (x0 + x1) / 2.0;
        sy += len * (y0 + y1) / 2.0;
    }
    if total == 0.0 {
        return coords[0];
    }
    (sx / total, sy / total)
}

/// 40 long CONUS-scale diagonals — the shape from the reviewer's second failing
/// case. Each line is thousands of km end to end, so at the archive's coarsest
/// zoom it crosses several tiles and gets clipped into per-tile pieces.
///
/// The union spans lon −124.9…−74.8 and lat 29.4…49.7, while every anchor sits
/// near the middle of its own line: the centroid box collapses to roughly
/// −103…−97 × 30…46. That gap is the defect, and it is what
/// [`centroid_bounds_would_lose_data`] measures.
fn long_lines() -> Vec<ParsedFeature> {
    let mut out = Vec::with_capacity(40);
    for i in 0..40u32 {
        let t = i as f64 / 39.0;
        // West end sweeps the Pacific coast, east end the Atlantic: every line
        // is long enough to cross tile columns at z2 (45°-wide at z3).
        let x0 = -124.9 + t * 1.5;
        let y0 = 29.4 + t * 8.0;
        let x1 = -74.82 - t * 1.5;
        let y1 = 49.7 - t * 8.0;
        // A midpoint bend keeps the geometry from being a perfect straight line
        // (so simplification has something to do) without changing the extent.
        let xm = (x0 + x1) / 2.0;
        let ym = (y0 + y1) / 2.0 + if i % 2 == 0 { 2.0 } else { -2.0 };
        out.push(line(
            &[(x0, y0), (xm, ym), (x1, y1)],
            1_700_000_000_000 + u64::from(i) * 60_000,
        ));
    }
    out
}

/// 24 two-vertex linestrings laid out as 6 axis-aligned BOX OUTLINES — the
/// reviewer's first failing case, and a deliberately *hard* one for this pin:
/// a box outline's four edge midpoints hit every corner of the box, so the
/// centroid bbox and the vertex bbox coincide. It therefore proves the flip is
/// not a blunt widening — an archive whose two quantities already agree keeps
/// declaring exactly the same numbers.
///
/// Spread across 60° of longitude so the archive is still multi-tile at its
/// coarsest zoom (the original AV-scale version sat inside one tile at every
/// zoom, which is precisely the fixture shape this file exists to avoid).
fn box_outlines() -> Vec<ParsedFeature> {
    let mut out = Vec::with_capacity(24);
    for b in 0..6u32 {
        let x0 = -120.0 + f64::from(b) * 12.0;
        let x1 = x0 + 6.0;
        let y0 = 20.0 + f64::from(b) * 4.0;
        let y1 = y0 + 3.0;
        let ts = 1_700_000_000_000 + u64::from(b) * 60_000;
        out.push(line(&[(x0, y0), (x0, y1)], ts)); // left edge
        out.push(line(&[(x1, y0), (x1, y1)], ts)); // right edge
        out.push(line(&[(x0, y0), (x1, y0)], ts)); // bottom edge
        out.push(line(&[(x0, y1), (x1, y1)], ts)); // top edge
    }
    out
}

fn config(min_zoom: u8, max_zoom: u8, simplify: bool) -> TileConfig {
    TileConfig {
        min_zoom,
        max_zoom,
        layer_name: "default".to_string(),
        temporal_bucket_ms: 3_600_000,
        clip_trajectories: true,
        simplify,
        simplify_max_zoom: max_zoom,
        ..TileConfig::default()
    }
}

/// Build a packed archive exactly the way `stt-build` does: profile the source
/// features under `mode`, tile them, declare the profiled bbox.
fn build(features: &[ParsedFeature], cfg: &TileConfig, mode: BoundsMode) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = PackWriter::create(dir.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
    generate_tiles_streaming(features, cfg, &mut writer, 2).unwrap();
    let profile = profile_features_with(
        features,
        &FeatureProfileOptions {
            bounds_mode: mode,
            ..Default::default()
        },
    )
    .unwrap();
    let mut metadata = Metadata::new("vertex-bounds")
        .with_bounds(profile.bounds)
        .with_time_range(profile.time_range)
        .with_z_range(profile.z_range)
        .with_zoom_levels(cfg.min_zoom, cfg.max_zoom)
        .with_temporal_bucket_ms(cfg.temporal_bucket_ms);
    metadata.properties.insert(
        BOUNDS_MODE_PROPERTY.to_string(),
        mode.as_manifest_value().to_string(),
    );
    writer.finalize(&metadata).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// The containment oracle — the same computation check 13 performs
// ---------------------------------------------------------------------------

/// Decode EVERY tile in the archive and fold every vertex into one bbox.
///
/// Uses `stt_core::metadata::FingerprintAccumulator`, which is the exact
/// accumulator `stt-validate` check 12/13 drives, so a pass here is evidence
/// about the validator and not merely about this test's own arithmetic.
fn decoded_vertex_bbox(dir: &std::path::Path) -> [f64; 4] {
    let reader = PackedReader::open(dir.join("manifest.json")).unwrap();
    let mut acc = FingerprintAccumulator::new();
    for entry in reader.entries() {
        let layers = reader.read_layers(entry).unwrap();
        acc.ingest(&layers);
    }
    acc.finish()
        .bbox
        .expect("archive decoded no finite coordinate at all")
}

/// Distinct `(z, x, y)` cells present at the archive's minimum zoom.
fn cells_at_min_zoom(dir: &std::path::Path) -> BTreeSet<(u32, u32)> {
    let reader = PackedReader::open(dir.join("manifest.json")).unwrap();
    let min_zoom = reader.metadata().min_zoom;
    reader
        .entries()
        .iter()
        .filter(|e| e.zoom == min_zoom)
        .map(|e| (e.x, e.y))
        .collect()
}

/// `declared ⊇ observed`, with the ULP floor check 13 uses (1e-9 deg ≈ 0.11 mm):
/// tile clipping synthesises vertices by interpolating along original segments,
/// and an interpolated endpoint can land an ULP or two past the segment's own
/// extreme.
fn contains(declared: &BoundingBox, observed: [f64; 4]) -> bool {
    const EPS: f64 = 1e-9;
    observed[0] >= declared.min_lon - EPS
        && observed[1] >= declared.min_lat - EPS
        && observed[2] <= declared.max_lon + EPS
        && observed[3] <= declared.max_lat + EPS
}

fn assert_contains(dir: &std::path::Path, label: &str) {
    let reader = PackedReader::open(dir.join("manifest.json")).unwrap();
    let declared = reader.metadata().bounds;
    let observed = decoded_vertex_bbox(dir);
    assert!(
        contains(&declared, observed),
        "{label}: declared metadata.bounds [{}, {}, {}, {}] does NOT contain the decoded vertex \
         bbox [{}, {}, {}, {}] — this is the check-13 error, i.e. silent data loss at query time",
        declared.min_lon,
        declared.min_lat,
        declared.max_lon,
        declared.max_lat,
        observed[0],
        observed[1],
        observed[2],
        observed[3]
    );
}

// ---------------------------------------------------------------------------
// 1. The fixture is genuinely multi-tile — anti-vacuity for everything below
// ---------------------------------------------------------------------------

/// ⚠️ **If this fails, every other test in this file is worthless.** A fixture
/// that fits in one tile at its coarsest zoom cannot exercise clipping, cannot
/// separate a per-tile row index from a feature id, and cannot expose an
/// under-stated bbox. Three separate bugs in this program shipped behind pins
/// written on such a fixture.
#[test]
fn multi_tile_precondition() {
    for (label, feats, cfg) in [
        ("long-lines", long_lines(), config(2, 6, true)),
        ("box-outlines", box_outlines(), config(2, 5, false)),
    ] {
        let dir = build(&feats, &cfg, BoundsMode::Vertex);
        let cells = cells_at_min_zoom(dir.path());
        assert!(
            cells.len() >= 2,
            "{label}: only {} cell(s) at the COARSEST zoom {} — the fixture fits in one tile and \
             cannot expose the defect",
            cells.len(),
            cfg.min_zoom
        );

        // …and — the assertion that actually protects a BOUNDS pin — no single
        // tile's decoded extent reaches the archive's. If one did, the whole
        // dataset's bbox would be recoverable from that one tile and every
        // containment assertion below could pass on a tiler that never split or
        // placed anything correctly.
        let whole = decoded_vertex_bbox(dir.path());
        let reader = PackedReader::open(dir.path().join("manifest.json")).unwrap();
        for entry in reader.entries() {
            let mut acc = FingerprintAccumulator::new();
            acc.ingest(&reader.read_layers(entry).unwrap());
            if let Some(b) = acc.finish().bbox {
                let covers_everything =
                    b[0] <= whole[0] && b[1] <= whole[1] && b[2] >= whole[2] && b[3] >= whole[3];
                assert!(
                    !covers_everything,
                    "{label}: tile z{}/{}/{} alone spans the archive's whole extent {whole:?} — \
                     the fixture is effectively single-tile for bounds purposes",
                    entry.zoom, entry.x, entry.y
                );
            }
        }
    }
}

/// The long-line fixture additionally exercises tile CLIPPING: a feature that
/// spans tile columns is cut into per-tile pieces, and clipping is what
/// synthesises the interpolated vertices whose ULP noise check 13's epsilon
/// exists for. The box-outline fixture deliberately does not (each 6°-wide edge
/// fits inside a 90°-wide z2 tile), which is why the assertion is separate.
#[test]
fn long_lines_are_split_across_tiles_at_the_coarsest_zoom() {
    let feats = long_lines();
    let cfg = config(2, 6, true);
    let dir = build(&feats, &cfg, BoundsMode::Vertex);
    let reader = PackedReader::open(dir.path().join("manifest.json")).unwrap();
    let placed: u64 = reader
        .entries()
        .iter()
        .filter(|e| e.zoom == cfg.min_zoom)
        .map(|e| u64::from(e.feature_count))
        .sum();
    assert!(
        placed > feats.len() as u64,
        "{placed} placements for {} features at z{} — nothing was split across tiles",
        feats.len(),
        cfg.min_zoom
    );
}

// ---------------------------------------------------------------------------
// 2. THE PIN: the reviewer's two failing cases now validate clean
// ---------------------------------------------------------------------------

/// The reviewer's second failing case: 40 long lines, `--simplify`, max zoom 6.
/// Under the centroid default this build declared `[-103.36, 30.40, -97.49,
/// 45.60]` while decoding to `[-124.9, 29.4, -74.82, 49.7]` — a bbox that
/// bounds about 6 % of the longitude span it claims to describe.
#[test]
fn long_lines_declare_bounds_that_contain_every_decoded_vertex() {
    let feats = long_lines();
    let cfg = config(2, 6, true);
    let dir = build(&feats, &cfg, BoundsMode::Vertex);
    assert_contains(dir.path(), "40 long lines, simplify, z2..6");
}

/// The reviewer's first failing case. Box outlines are the *degenerate* shape
/// for this defect — centroid bbox == vertex bbox — so this pins that the flip
/// keeps such an archive's numbers identical while still making it attested.
#[test]
fn box_outline_linestrings_declare_bounds_that_contain_every_decoded_vertex() {
    let feats = box_outlines();
    let cfg = config(2, 5, false);
    let dir = build(&feats, &cfg, BoundsMode::Vertex);
    assert_contains(dir.path(), "24 box-outline linestrings, z2..5");

    let vertex = build(&feats, &cfg, BoundsMode::Vertex);
    let centroid = build(&feats, &cfg, BoundsMode::Centroid);
    let bv = PackedReader::open(vertex.path().join("manifest.json"))
        .unwrap()
        .metadata()
        .bounds;
    let bc = PackedReader::open(centroid.path().join("manifest.json"))
        .unwrap()
        .metadata()
        .bounds;
    assert_eq!(
        bv, bc,
        "box outlines: the two quantities coincide, so the flip must not move this archive's \
         declared numbers at all"
    );
}

/// The anti-vacuity twin of the pin above: the SAME fixture under the legacy
/// mode must FAIL containment. Without this, a bug that made `decoded_vertex_bbox`
/// return the declared box would leave both pins green.
#[test]
fn centroid_bounds_would_lose_data_on_the_same_fixture() {
    let feats = long_lines();
    let cfg = config(2, 6, true);
    let dir = build(&feats, &cfg, BoundsMode::Centroid);

    let reader = PackedReader::open(dir.path().join("manifest.json")).unwrap();
    let declared = reader.metadata().bounds;
    let observed = decoded_vertex_bbox(dir.path());
    assert!(
        !contains(&declared, observed),
        "the legacy centroid box [{}, {}, {}, {}] unexpectedly contains the decoded bbox \
         [{}, {}, {}, {}] — this fixture no longer exposes the defect it was built for",
        declared.min_lon,
        declared.min_lat,
        declared.max_lon,
        declared.max_lat,
        observed[0],
        observed[1],
        observed[2],
        observed[3]
    );

    // Quantify the loss, so a future narrowing of the fixture is visible: the
    // legacy box must miss a large majority of the real longitude span.
    let declared_span = declared.max_lon - declared.min_lon;
    let observed_span = observed[2] - observed[0];
    assert!(
        declared_span < observed_span * 0.25,
        "expected the centroid box to under-state the longitude span badly; got {declared_span} \
         of {observed_span}"
    );
}

// ---------------------------------------------------------------------------
// 3. Soundness DIRECTION — superset, never tighter
// ---------------------------------------------------------------------------

/// The directional property on a path whose vertices reach well beyond its own
/// centroid, asserted per edge rather than as an area comparison (a box can be
/// wider in one axis and tighter in the other, and the tighter axis is the one
/// that loses data).
#[test]
fn vertex_bounds_are_a_superset_of_the_centroid_box_on_every_edge() {
    for (label, feats) in [
        ("long-lines", long_lines()),
        ("box-outlines", box_outlines()),
    ] {
        let p = profile_features_with(&feats, &FeatureProfileOptions::default()).unwrap();
        assert_eq!(p.bounds_mode, BoundsMode::Vertex);
        assert_eq!(p.bounds, p.vertex_bounds);
        assert!(
            p.vertex_bounds.min_lon <= p.centroid_bounds.min_lon
                && p.vertex_bounds.min_lat <= p.centroid_bounds.min_lat
                && p.vertex_bounds.max_lon >= p.centroid_bounds.max_lon
                && p.vertex_bounds.max_lat >= p.centroid_bounds.max_lat,
            "{label}: the honest box must never be tighter than the legacy one on any edge"
        );

        // And it bounds every source vertex, which is the claim itself.
        for f in &feats {
            if let Some(GeomValue::LineString(ps)) = f.geojson.geometry.as_ref().map(|g| &g.value) {
                for p2 in ps {
                    assert!(
                        p2[0] >= p.vertex_bounds.min_lon
                            && p2[0] <= p.vertex_bounds.max_lon
                            && p2[1] >= p.vertex_bounds.min_lat
                            && p2[1] <= p.vertex_bounds.max_lat,
                        "{label}: source vertex {p2:?} escaped the declared bbox"
                    );
                }
            }
        }
    }
}

/// Byte status, stated precisely: a POINT archive's anchor **is** its only
/// vertex, so the flip moves nothing for it. Only non-point geometry has a
/// centroid that differs from its vertices, and only those archives' manifest
/// `bounds` values change on rebuild. Pinning this keeps the R1 blast radius
/// honest — "every archive's bounds moved" would be false.
#[test]
fn point_only_archives_declare_identical_bounds_under_both_modes() {
    let mut points = Vec::new();
    for i in 0..50u32 {
        let lon = -120.0 + f64::from(i) * 3.7;
        let lat = -40.0 + f64::from(i % 17) * 4.3;
        points.push(ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::Point(vec![lon, lat]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 1_700_000_000_000 + u64::from(i) * 60_000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon,
            lat,
        });
    }
    let p = profile_features_with(&points, &FeatureProfileOptions::default()).unwrap();
    assert_eq!(
        p.vertex_bounds, p.centroid_bounds,
        "a point's anchor IS its vertex — the flip must not move a point archive's bounds"
    );

    let cfg = config(2, 5, false);
    let v = build(&points, &cfg, BoundsMode::Vertex);
    let c = build(&points, &cfg, BoundsMode::Centroid);
    let bounds_of = |dir: &std::path::Path| {
        serde_json::to_string(
            &PackedReader::open(dir.join("manifest.json"))
                .unwrap()
                .metadata()
                .bounds,
        )
        .unwrap()
    };
    assert_eq!(bounds_of(v.path()), bounds_of(c.path()));
    assert_contains(v.path(), "50 points, z2..5");
}

// ---------------------------------------------------------------------------
// 4. Antimeridian and poles — the recorded hazards
// ---------------------------------------------------------------------------

/// A dataset straddling ±180° must produce a bbox that is **loose but sound and
/// UNWRAPPED**, not a wrapped interval.
///
/// Naive min/max is what produces that: the lines below reach both −179 and
/// +179, so the fold yields roughly `[-179, +179]` — nearly full-globe width,
/// which is looser than the (correct) wrapped `[+170, −170]` interval but
/// *contains every vertex*, which is the direction pre-intersection requires.
/// A wrapped box would be tighter and would need a wrap-aware reader on the
/// other side; check 13 explicitly refuses to evaluate one ("containment on the
/// longitude axis is not decidable from an unwrapped decoded bbox") and skips
/// the axis with a warning. So the sound answer here is the loose one, and the
/// assertion is that the writer never emits `min_lon > max_lon`.
#[test]
fn antimeridian_crossing_yields_a_loose_unwrapped_bbox_that_still_contains_everything() {
    let feats = vec![
        line(&[(170.0, 10.0), (179.0, 12.0)], 1_700_000_000_000),
        line(&[(-179.0, 12.0), (-170.0, 10.0)], 1_700_000_060_000),
        line(&[(178.0, -5.0), (-178.0, -5.0)], 1_700_000_120_000),
    ];
    let p = profile_features_with(&feats, &FeatureProfileOptions::default()).unwrap();

    assert!(
        p.bounds.min_lon <= p.bounds.max_lon,
        "the writer must never emit a WRAPPED longitude interval: check 13 cannot evaluate one \
         and skips the axis entirely, which would silently disable the check"
    );
    // Loose, as documented — and every vertex is inside it.
    assert_eq!(p.bounds.min_lon, -179.0);
    assert_eq!(p.bounds.max_lon, 179.0);
    for f in &feats {
        if let Some(GeomValue::LineString(ps)) = f.geojson.geometry.as_ref().map(|g| &g.value) {
            for v in ps {
                assert!(v[0] >= p.bounds.min_lon && v[0] <= p.bounds.max_lon);
            }
        }
    }

    // End to end, through the tiler: the archive still satisfies containment.
    let cfg = config(2, 4, false);
    let dir = build(&feats, &cfg, BoundsMode::Vertex);
    assert_contains(dir.path(), "antimeridian-straddling lines");
}

/// Latitudes fold UNCLAMPED. Nothing is projected in the bounds pass, so a
/// vertex at ±90° widens the declared box to ±90°; clamping to the Mercator
/// limit (~±85.05°) would *shrink* the declared box below the data, which is
/// the exact unsoundness this item removes.
#[test]
fn polar_vertices_are_not_clamped_to_the_mercator_limit() {
    let feats = vec![
        line(&[(0.0, 80.0), (10.0, 90.0)], 1_700_000_000_000),
        line(&[(0.0, -90.0), (10.0, -80.0)], 1_700_000_060_000),
    ];
    let p = profile_features_with(&feats, &FeatureProfileOptions::default()).unwrap();
    assert_eq!(p.bounds.max_lat, 90.0);
    assert_eq!(p.bounds.min_lat, -90.0);
    assert!(
        p.bounds.max_lat > 85.06,
        "a Mercator clamp here would declare less than the data holds"
    );
}

// ---------------------------------------------------------------------------
// 5. Determinism — the bbox is an order-independent min/max fold
// ---------------------------------------------------------------------------

/// Byte-reproducible bounds: identical across re-runs AND across any
/// permutation of the input, at the MANIFEST level (not just the profiler's),
/// so two builds of the same data serialize identical `bounds` bytes.
#[test]
fn declared_bounds_are_byte_reproducible_and_order_independent() {
    let cfg = config(2, 5, true);
    let base = long_lines();

    let a = build(&base, &cfg, BoundsMode::Vertex);
    let b = build(&base, &cfg, BoundsMode::Vertex);

    let bounds_json = |dir: &std::path::Path| {
        let m = PackedReader::open(dir.join("manifest.json"))
            .unwrap()
            .metadata()
            .clone();
        serde_json::to_string(&m.bounds).unwrap()
    };
    assert_eq!(
        bounds_json(a.path()),
        bounds_json(b.path()),
        "two builds of the same input must serialize byte-identical bounds"
    );

    // Deterministic shuffle that touches every position.
    let mut shuffled = base.clone();
    shuffled.reverse();
    shuffled.rotate_left(7);
    let c = build(&shuffled, &cfg, BoundsMode::Vertex);
    assert_eq!(
        bounds_json(a.path()),
        bounds_json(c.path()),
        "min/max is an order-independent fold: permuting the input must not move a byte"
    );
}

// ---------------------------------------------------------------------------
// 6. The manifest stamp and the vertical extent
// ---------------------------------------------------------------------------

/// The attestation seam: `metadata.properties.bounds_mode` is what promotes a
/// check-13 containment failure from a legacy WARNING to an ERROR, so its key
/// and value spelling are a cross-crate contract. A legacy archive — the whole
/// pre-R1 fleet — carries no such key and must therefore stay warn-only, which
/// is pinned here by its absence.
#[test]
fn the_manifest_records_which_quantity_bounds_came_from() {
    let feats = long_lines();
    let cfg = config(2, 4, false);

    let vertex = build(&feats, &cfg, BoundsMode::Vertex);
    let m = PackedReader::open(vertex.path().join("manifest.json"))
        .unwrap()
        .metadata()
        .clone();
    assert_eq!(
        m.properties.get(BOUNDS_MODE_PROPERTY).map(String::as_str),
        Some("vertex")
    );

    let centroid = build(&feats, &cfg, BoundsMode::Centroid);
    let m = PackedReader::open(centroid.path().join("manifest.json"))
        .unwrap()
        .metadata()
        .clone();
    assert_eq!(
        m.properties.get(BOUNDS_MODE_PROPERTY).map(String::as_str),
        Some("centroid"),
        "the rollback build must be self-describing, not indistinguishable from a pre-flag archive"
    );

    // A legacy archive (no stamp) is the warn-only path: nothing this crate
    // writes may leave the key absent by accident, but an archive written
    // WITHOUT it must still round-trip.
    let dir = tempfile::tempdir().unwrap();
    let mut writer = PackWriter::create(dir.path(), BlobOrdering::Auto, 64 * 1024 * 1024).unwrap();
    generate_tiles_streaming(&feats, &cfg, &mut writer, 2).unwrap();
    writer
        .finalize(
            &Metadata::new("legacy")
                .with_bounds(BoundingBox::new(-1.0, -1.0, 1.0, 1.0))
                .with_zoom_levels(cfg.min_zoom, cfg.max_zoom)
                .with_temporal_bucket_ms(cfg.temporal_bucket_ms),
        )
        .unwrap();
    let m = PackedReader::open(dir.path().join("manifest.json"))
        .unwrap()
        .metadata()
        .clone();
    assert!(
        !m.properties.contains_key(BOUNDS_MODE_PROPERTY),
        "an unstamped archive must stay unattested — otherwise the pre-R1 fleet turns red"
    );
}

/// `metadata.z_range` is additive: present only when the source actually
/// carried altitude, absent (and byte-invisible) otherwise. The absence case is
/// the K11 byte-identity criterion — every 2D archive's manifest must be
/// unchanged by this field existing.
#[test]
fn z_range_is_declared_only_when_the_source_carried_altitude() {
    let cfg = config(2, 4, false);

    let flat = build(&long_lines(), &cfg, BoundsMode::Vertex);
    let m = PackedReader::open(flat.path().join("manifest.json"))
        .unwrap()
        .metadata()
        .clone();
    assert_eq!(m.z_range, None);
    assert!(
        !serde_json::to_string(&m).unwrap().contains("z_range"),
        "a 2D archive's manifest must not gain the key at all"
    );

    // 3-element positions → a declared vertical extent that CONTAINS them.
    let mut volumetric = Vec::new();
    for i in 0..8u32 {
        let x0 = -120.0 + f64::from(i) * 6.0;
        let z0 = 100.0 + f64::from(i) * 50.0;
        volumetric.push(ParsedFeature {
            home_zoom: None,
            geojson: Feature {
                bbox: None,
                geometry: Some(Geometry::new(GeomValue::LineString(vec![
                    vec![x0, 30.0, z0],
                    vec![x0 + 5.0, 40.0, z0 + 25.0],
                ]))),
                id: None,
                properties: None,
                foreign_members: None,
            },
            shared_properties: None,
            timestamp: 1_700_000_000_000 + u64::from(i) * 60_000,
            end_timestamp: None,
            vertex_timestamps: None,
            vertex_values: None,
            vertex_value_matrix: None,
            lon: x0 + 2.5,
            lat: 35.0,
        });
    }
    let p = profile_features_with(&volumetric, &FeatureProfileOptions::default()).unwrap();
    let [lo, hi] = p.z_range.expect("3-element positions carry altitude");
    assert!(lo <= 100.0 && hi >= 475.0, "declared z [{lo}, {hi}]");
}

/// The default the builder actually ships with, asserted from OUTSIDE the crate
/// that defines it — `profile_features_with(.., &Default::default())` is the
/// call `stt-build` makes when `--bounds-mode` is omitted.
#[test]
fn the_shipped_default_is_the_honest_quantity() {
    assert_eq!(DEFAULT_BOUNDS_MODE, BoundsMode::Vertex);
    assert_eq!(
        FeatureProfileOptions::default().bounds_mode,
        BoundsMode::Vertex
    );
}
