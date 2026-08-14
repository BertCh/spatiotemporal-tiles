//! A1 — antimeridian-aware POLYGON clipping (split-then-reuse), end-to-end.
//!
//! Two families:
//!  1. GOLDEN over the named dateline fixtures (Fiji MultiPolygon, Chukotka
//!     polygon, a dateline storm cell WITH a seam-straddling hole): build across
//!     a zoom range and assert the campaign gate — `antimeridian_fallbacks == 0`
//!     at every zoom, every emitted ring closed / in `[-180, 180]` / free of any
//!     `|Δlon| > 180°` edge, CCW exterior + CW holes (winding by ring-nesting so
//!     it holds for flattened MultiPolygon pieces too), pieces landing on BOTH
//!     dateline columns (no single-tile smear / world span), and — on the
//!     unfragmented z1 pieces — planar area conserved + holes contained.
//!  2. SEEDED property invariants over random convex crossing quads (optionally
//!     with a straddling hole). A pinned splitmix64 seed makes the run fully
//!     deterministic and reproducible without pulling the `proptest` crate in as
//!     a new dependency (the repo's tracked-seed intent, kept in-scope for this
//!     Rust-only, crates/-only change). The invariants match the golden set plus
//!     a brute-force per-ring simplicity check that TOLERATES a hole edge lying
//!     on the ±180 seam (the accepted seam-tangency artifact).

use std::collections::BTreeSet;
use stt_build::input::ParsedFeature;
use stt_build::tiler::{
    generate_tiles, generate_tiles_streaming, GeneratedTile, TileConfig, TileWriter,
};
use stt_core::arrow_tile::GeometryColumn;

const TS: u64 = 1_700_000_000_000;

/// Pinned seed for the deterministic property generator ("A1 DATE COFFEE 01").
const SEED: u64 = 0x00A1_DA7E_C0FF_EE01;

/// Discards tiles; only the returned `TileStats` matter (fallback count).
struct NullWriter;
impl TileWriter for NullWriter {
    fn write_tile(&mut self, _tile: &GeneratedTile) -> anyhow::Result<()> {
        Ok(())
    }
}

fn config(min_zoom: u8, max_zoom: u8) -> TileConfig {
    TileConfig {
        min_zoom,
        max_zoom,
        layer_name: "areas".to_string(),
        temporal_bucket_ms: 3_600_000,
        clip_trajectories: false,
        ..TileConfig::default()
    }
}

fn parsed_from_geometry(geom: geojson::Geometry) -> ParsedFeature {
    let (lon, lat) = rep_point(&geom.value);
    ParsedFeature {
        home_zoom: None,
        geojson: geojson::Feature {
            bbox: None,
            geometry: Some(geom),
            id: None,
            properties: None,
            foreign_members: None,
        },
        shared_properties: None,
        timestamp: TS,
        end_timestamp: None,
        vertex_timestamps: None,
        vertex_values: None,
        vertex_value_matrix: None,
        lon,
        lat,
    }
}

/// Representative point = first exterior vertex (matches `stt-build`'s reader).
fn rep_point(v: &geojson::Value) -> (f64, f64) {
    match v {
        geojson::Value::Polygon(rings) => (rings[0][0][0], rings[0][0][1]),
        geojson::Value::MultiPolygon(polys) => (polys[0][0][0][0], polys[0][0][0][1]),
        geojson::Value::LineString(c) => (c[0][0], c[0][1]),
        _ => panic!("fixture must be a Polygon or MultiPolygon"),
    }
}

fn load_fixture(name: &str) -> ParsedFeature {
    let path = format!(
        "{}/tests/fixtures/antimeridian/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let gj: geojson::GeoJson = text.parse().unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let feature = match gj {
        geojson::GeoJson::FeatureCollection(fc) => fc
            .features
            .into_iter()
            .next()
            .expect("empty FeatureCollection"),
        geojson::GeoJson::Feature(f) => f,
        geojson::GeoJson::Geometry(g) => geojson::Feature {
            bbox: None,
            geometry: Some(g),
            id: None,
            properties: None,
            foreign_members: None,
        },
    };
    parsed_from_geometry(feature.geometry.expect("fixture has no geometry"))
}

// ----------------------------------------------------------------------
// Geometry helpers (operate on the decoded GeometryColumn rings)
// ----------------------------------------------------------------------

/// Signed shoelace area of a closed `[[lon, lat], …]` ring (positive = CCW).
fn ring_area(ring: &[[f64; 2]]) -> f64 {
    let mut s = 0.0;
    for w in ring.windows(2) {
        s += w[0][0] * w[1][1] - w[1][0] * w[0][1];
    }
    s / 2.0
}

/// A representative interior point of a convex-ish ring: the mean of its
/// distinct vertices (drops the closing duplicate).
fn ring_rep_point(ring: &[[f64; 2]]) -> [f64; 2] {
    let pts = &ring[..ring.len().saturating_sub(1)];
    let n = pts.len().max(1) as f64;
    let (mut sx, mut sy) = (0.0, 0.0);
    for c in pts {
        sx += c[0];
        sy += c[1];
    }
    [sx / n, sy / n]
}

/// Ray-cast point-in-ring (ring closed).
fn point_in_ring(p: [f64; 2], ring: &[[f64; 2]]) -> bool {
    let mut inside = false;
    for w in ring.windows(2) {
        let (a, b) = (w[0], w[1]);
        let straddles = (a[1] > p[1]) != (b[1] > p[1]);
        if straddles {
            let x_cross = (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0];
            if p[0] < x_cross {
                inside = !inside;
            }
        }
    }
    inside
}

/// Proper (crossing) intersection of open segments `p1p2` and `p3p4`.
fn segments_cross(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], p4: [f64; 2]) -> bool {
    let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);
    // Strict straddle both ways ⇒ a proper crossing (collinear/touching is
    // tolerated — those are the accepted seam-tangency / shared-vertex cases).
    ((d1 > 0.0) != (d2 > 0.0))
        && ((d3 > 0.0) != (d4 > 0.0))
        && d1 != 0.0
        && d2 != 0.0
        && d3 != 0.0
        && d4 != 0.0
}

/// Brute-force simplicity of a single closed ring: no two NON-adjacent edges
/// cross. Edges lying on the ±180 seam are skipped (the accepted seam-tangency
/// artifact — a hole may run along the shell on the meridian).
fn ring_is_simple(ring: &[[f64; 2]]) -> bool {
    let on_seam = |a: [f64; 2], b: [f64; 2]| {
        (a[0].abs() - 180.0).abs() < 1e-9 && (b[0].abs() - 180.0).abs() < 1e-9
    };
    let m = ring.len();
    if m < 4 {
        return true;
    }
    // Edges are ring[i]..ring[i+1] for i in 0..m-1 (closed ring).
    let e = m - 1;
    for i in 0..e {
        for j in (i + 1)..e {
            // Skip adjacent edges (share a vertex), including the wrap pair.
            if j == i + 1 || (i == 0 && j == e - 1) {
                continue;
            }
            let (a, b, c, d) = (ring[i], ring[i + 1], ring[j], ring[j + 1]);
            if on_seam(a, b) || on_seam(c, d) {
                continue;
            }
            if segments_cross(a, b, c, d) {
                return false;
            }
        }
    }
    true
}

/// Per-tile validity: every ring closed, in `[-180, 180]`, free of a
/// `|Δlon| > 180°` edge; `ring[0]` the CCW exterior, every subsequent ring a CW
/// hole contained in that exterior. `ring[0]` IS the exterior for every emitted
/// feature: `clip_polygons_to_tiles` puts the clipped exterior first, and the
/// split pieces never co-occur in a tile (opposite hemispheres, and the Fiji
/// sub-polygons straddle the equator so they never merge), so a tile feature is
/// always a single polygon — never a flattened multi-exterior MultiPolygon.
fn assert_tile_valid(tile: &GeneratedTile) {
    for layer in &tile.layers {
        let GeometryColumn::Polygon(features) = &layer.geometry else {
            panic!("expected polygon geometry in {:?}", tile.id);
        };
        for rings in features {
            assert!(!rings.is_empty(), "feature with no rings in {:?}", tile.id);
            for ring in rings {
                assert!(
                    ring.len() >= 4,
                    "degenerate ring in {:?}: {ring:?}",
                    tile.id
                );
                assert_eq!(ring.first(), ring.last(), "unclosed ring in {:?}", tile.id);
                for c in ring {
                    assert!(
                        (-180.0..=180.0).contains(&c[0]),
                        "lon {} out of [-180,180] in {:?}",
                        c[0],
                        tile.id
                    );
                }
                for w in ring.windows(2) {
                    assert!(
                        (w[1][0] - w[0][0]).abs() <= 180.0 + 1e-9,
                        "surviving |Δlon|>180 edge in {:?}: {ring:?}",
                        tile.id
                    );
                }
            }
            let exterior = &rings[0];
            let ext_area = ring_area(exterior);
            if ext_area.abs() > 1e-12 {
                assert!(
                    ext_area > 0.0,
                    "exterior must be CCW (positive) in {:?}",
                    tile.id
                );
            }
            for hole in &rings[1..] {
                let ha = ring_area(hole);
                if ha.abs() <= 1e-12 {
                    continue; // seam-degenerate sliver
                }
                assert!(ha < 0.0, "hole must be CW (negative) in {:?}", tile.id);
                assert!(
                    point_in_ring(ring_rep_point(hole), exterior),
                    "hole not contained by its exterior in {:?}",
                    tile.id
                );
            }
        }
    }
}

/// Net area (Σ|exterior| − Σ|hole|) across every tile's polygon features. On the
/// z1 pieces (which the world tile grid leaves unfragmented) this equals the
/// whole feature's planar area.
fn tiles_net_area(tiles: &[GeneratedTile]) -> f64 {
    let mut net = 0.0;
    for tile in tiles {
        for layer in &tile.layers {
            if let GeometryColumn::Polygon(features) = &layer.geometry {
                for rings in features {
                    net += ring_area(&rings[0]).abs();
                    for hole in &rings[1..] {
                        net -= ring_area(hole).abs();
                    }
                }
            }
        }
    }
    net
}

/// Ground-truth net area of the source feature, computed in each ring's own
/// continuous (unwrapped ±360) longitude frame.
fn source_net_area(feat: &ParsedFeature) -> f64 {
    fn unwrapped_abs(ring: &[Vec<f64>]) -> f64 {
        let mut pts: Vec<(f64, f64)> = ring
            .iter()
            .filter(|c| c.len() >= 2)
            .map(|c| (c[0], c[1]))
            .collect();
        if pts.len() >= 2 && pts.first() == pts.last() {
            pts.pop();
        }
        let mut off = 0.0;
        let mut uw: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
        for i in 0..pts.len() {
            if i > 0 {
                let d = pts[i].0 - pts[i - 1].0;
                if d > 180.0 {
                    off -= 360.0;
                } else if d < -180.0 {
                    off += 360.0;
                }
            }
            uw.push((pts[i].0 + off, pts[i].1));
        }
        let mut s = 0.0;
        let n = uw.len();
        for i in 0..n {
            let a = uw[i];
            let b = uw[(i + 1) % n];
            s += a.0 * b.1 - b.0 * a.1;
        }
        (s / 2.0).abs()
    }
    let polys: Vec<Vec<Vec<Vec<f64>>>> = match &feat.geojson.geometry.as_ref().unwrap().value {
        geojson::Value::Polygon(r) => vec![r.clone()],
        geojson::Value::MultiPolygon(p) => p.clone(),
        _ => panic!("not a polygon"),
    };
    let mut net = 0.0;
    for rings in &polys {
        if rings.is_empty() {
            continue;
        }
        net += unwrapped_abs(&rings[0]);
        for hole in &rings[1..] {
            net -= unwrapped_abs(hole);
        }
    }
    net
}

/// Every zoom `min..=max`, exercised with BOTH the streaming stats path (for
/// the fallback count) and the geometry path (for ring invariants).
fn assert_gate(feat: &ParsedFeature, min_zoom: u8, max_zoom: u8) {
    let mut writer = NullWriter;
    let stats = generate_tiles_streaming(
        std::slice::from_ref(feat),
        &config(min_zoom, max_zoom),
        &mut writer,
        1,
    )
    .unwrap();
    assert_eq!(
        stats.antimeridian_fallbacks, 0,
        "a splittable dateline feature must never fall back"
    );

    let tiles = generate_tiles(std::slice::from_ref(feat), &config(min_zoom, max_zoom), 1).unwrap();
    assert!(!tiles.is_empty(), "no tiles produced");
    for tile in &tiles {
        assert_tile_valid(tile);
        for layer in &tile.layers {
            if let GeometryColumn::Polygon(features) = &layer.geometry {
                for rings in features {
                    for ring in rings {
                        assert!(
                            ring_is_simple(ring),
                            "self-intersecting ring in {:?}",
                            tile.id
                        );
                    }
                }
            }
        }
    }

    // z1: the world grid (n=2) leaves each hemisphere piece unfragmented, so
    // planar area is conserved exactly.
    let z1 = generate_tiles(std::slice::from_ref(feat), &config(1, 1), 1).unwrap();
    let got = tiles_net_area(&z1);
    let want = source_net_area(feat);
    assert!(
        (got - want).abs() < 1e-9,
        "z1 area not conserved: got {got}, want {want}"
    );
}

/// Pieces must land on BOTH dateline columns and NOWHERE in the interior — a
/// real split, not a single-tile smear or a full-world span. (Only holds for
/// features that hug the dateline, i.e. the named fixtures.)
fn assert_lands_on_both_edges(feat: &ParsedFeature, z: u8) {
    let tiles = generate_tiles(std::slice::from_ref(feat), &config(z, z), 1).unwrap();
    let n = 1u32 << z;
    let cols: BTreeSet<u32> = tiles.iter().map(|t| t.id.x).collect();
    assert!(
        cols.iter().any(|&x| x <= 1),
        "no near -180 column in {cols:?}"
    );
    assert!(
        cols.iter().any(|&x| x >= n - 2),
        "no near +180 column in {cols:?}"
    );
    assert!(
        cols.iter().all(|&x| x <= 1 || x >= n - 2),
        "polygon smeared into an interior column (world span): {cols:?}"
    );
}

// ----------------------------------------------------------------------
// Golden fixtures
// ----------------------------------------------------------------------

#[test]
fn golden_fiji_multipolygon_splits_cleanly() {
    let feat = load_fixture("fiji_multipolygon.geojson");
    assert_gate(&feat, 1, 6);
    assert_lands_on_both_edges(&feat, 6);
}

#[test]
fn golden_chukotka_polygon_splits_cleanly() {
    let feat = load_fixture("chukotka_polygon.geojson");
    assert_gate(&feat, 1, 6);
    assert_lands_on_both_edges(&feat, 6);
}

#[test]
fn golden_dateline_storm_cell_with_hole_splits_cleanly() {
    let feat = load_fixture("dateline_storm_cell.geojson");
    assert_gate(&feat, 1, 6);
    assert_lands_on_both_edges(&feat, 6);

    // The seam-straddling eye is retained (as a CW hole) in BOTH hemispheres:
    // z1 leaves the pieces unfragmented, so there are exactly two features, each
    // a 2-ring polygon (exterior + eye).
    let z1 = generate_tiles(std::slice::from_ref(&feat), &config(1, 1), 1).unwrap();
    let mut ring_counts = Vec::new();
    for tile in &z1 {
        for layer in &tile.layers {
            if let GeometryColumn::Polygon(features) = &layer.geometry {
                for rings in features {
                    ring_counts.push(rings.len());
                }
            }
        }
    }
    assert_eq!(
        ring_counts.len(),
        2,
        "storm cell must land in both hemispheres as two pieces, got {ring_counts:?}"
    );
    assert!(
        ring_counts.iter().all(|&c| c == 2),
        "each hemisphere keeps exterior + eye, got ring counts {ring_counts:?}"
    );
}

// ----------------------------------------------------------------------
// Seeded property invariants
// ----------------------------------------------------------------------

/// Minimal deterministic PRNG (splitmix64) — pinned seed ⇒ reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
    fn b(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }
}

fn ring(pts: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let mut r: Vec<Vec<f64>> = pts.iter().map(|(x, y)| vec![*x, *y]).collect();
    r.push(r[0].clone());
    r
}

#[test]
fn property_random_crossing_polygons_split_cleanly() {
    let mut rng = Rng(SEED);
    let cases = 256;
    for case in 0..cases {
        // A convex rectangle straddling ±180: west edge in [150,178]E, east edge
        // in [-178,-150]W, so the west→east edge always jumps > 180°.
        let wlon = rng.f(150.0, 178.0);
        let elon = rng.f(-178.0, -150.0);
        let lat_lo = rng.f(-60.0, 20.0);
        let lat_hi = lat_lo + rng.f(6.0, 20.0);
        let exterior = ring(&[
            (wlon, lat_lo),
            (elon, lat_lo),
            (elon, lat_hi),
            (wlon, lat_hi),
        ]);
        let mut rings = vec![exterior];
        // ~half the cases carry a straddling hole strictly inside the shell.
        if rng.b() {
            let inset_lon = rng.f(1.5, 4.0);
            let inset_lat = rng.f(1.0, (lat_hi - lat_lo) / 3.0);
            // CW hole (opposite the CCW-ish shell); split re-winds anyway.
            let hole = ring(&[
                (wlon + inset_lon, lat_lo + inset_lat),
                (wlon + inset_lon, lat_hi - inset_lat),
                (elon - inset_lon, lat_hi - inset_lat),
                (elon - inset_lon, lat_lo + inset_lat),
            ]);
            rings.push(hole);
        }
        let feat = parsed_from_geometry(geojson::Geometry::new(geojson::Value::Polygon(rings)));

        // Gate holds for every random crossing polygon. Clip depth is capped at
        // z3: every hole is < 45° wide per hemisphere, so no z≤3 tile can fall
        // ENTIRELY inside a hole (which would make nesting-based ring roles
        // ambiguous — exterior and eye clip to the identical tile square).
        let mut writer = NullWriter;
        let stats =
            generate_tiles_streaming(std::slice::from_ref(&feat), &config(1, 3), &mut writer, 1)
                .unwrap();
        assert_eq!(
            stats.antimeridian_fallbacks, 0,
            "case {case}: unexpected fallback (wlon={wlon}, elon={elon})"
        );

        let tiles = generate_tiles(std::slice::from_ref(&feat), &config(1, 3), 1).unwrap();
        assert!(!tiles.is_empty(), "case {case}: no tiles");
        for tile in &tiles {
            assert_tile_valid(tile);
            for layer in &tile.layers {
                if let GeometryColumn::Polygon(features) = &layer.geometry {
                    for rs in features {
                        for r in rs {
                            assert!(ring_is_simple(r), "case {case}: self-intersecting ring");
                        }
                    }
                }
            }
        }

        let z1 = generate_tiles(std::slice::from_ref(&feat), &config(1, 1), 1).unwrap();
        let got = tiles_net_area(&z1);
        let want = source_net_area(&feat);
        assert!(
            (got - want).abs() < 1e-6,
            "case {case}: z1 area not conserved: got {got}, want {want}"
        );
    }
}

// ----------------------------------------------------------------------
// The seam ↔ `metadata.bounds` interaction (the finding behind check 13's
// antimeridian-seam exemption).
//
// `profile_features_with` folds a plain min/max over the SOURCE vertices, so a
// ±180°-straddling dataset declares a loose, nearly full-width, *unwrapped*
// interval — deliberately, and pinned by
// `antimeridian_crossing_yields_a_loose_unwrapped_bbox_that_still_contains_everything`
// (`vertex_bounds_multi_tile.rs`). That pin uses LINES, and lines are exactly
// the geometry class the builder does NOT split at the seam. For POLYGONS the
// A1 split above synthesises vertices at exactly ±180 that no source vertex
// carried, so the honest source bbox provably fails to contain the archive's
// own decoded output. The two tests below pin both halves.
// ----------------------------------------------------------------------

/// Longitude extent over every decoded vertex in `tiles`, plus the latitude
/// extent, as `[min_lon, min_lat, max_lon, max_lat]` — the same quantity
/// `stt-validate`'s check 13 accumulates from a decode.
fn decoded_bbox(tiles: &[GeneratedTile]) -> [f64; 4] {
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    let mut fold = |c: &[f64; 2]| {
        b[0] = b[0].min(c[0]);
        b[1] = b[1].min(c[1]);
        b[2] = b[2].max(c[0]);
        b[3] = b[3].max(c[1]);
    };
    for tile in tiles {
        for layer in &tile.layers {
            match &layer.geometry {
                GeometryColumn::Point(pts) => pts.iter().for_each(&mut fold),
                GeometryColumn::LineString(runs) => {
                    runs.iter().flatten().for_each(&mut fold);
                }
                GeometryColumn::Polygon(feats) => {
                    feats.iter().flatten().flatten().for_each(&mut fold);
                }
            }
        }
    }
    b
}

/// Every decoded LineString run, flattened out of the tiles.
fn decoded_runs(tiles: &[GeneratedTile]) -> Vec<Vec<[f64; 2]>> {
    let mut out = Vec::new();
    for tile in tiles {
        for layer in &tile.layers {
            if let GeometryColumn::LineString(runs) = &layer.geometry {
                out.extend(runs.iter().cloned());
            }
        }
    }
    out
}

fn vertex_bounds(feat: &ParsedFeature) -> stt_core::types::BoundingBox {
    stt_build::input::profile_features_with(
        std::slice::from_ref(feat),
        &stt_build::input::FeatureProfileOptions {
            bounds_mode: stt_build::input::BoundsMode::Vertex,
            elevation_column: None,
        },
    )
    .expect("profile")
    .bounds
}

/// ⭐ THE FINDING. A seam-crossing POLYGON is **split, not dropped** — and the
/// split is precisely why the declared `metadata.bounds` cannot contain the
/// archive: `split_polygon_at_antimeridian` synthesises seam vertices at
/// exactly ±180°, while `profile_features_with` folds min/max over the SOURCE
/// vertices, which stop 2° short of the seam on either side.
///
/// So on an archive that ATTESTS vertex-derived bounds, `stt-validate`'s check
/// 13 sees a genuine containment failure produced by correct, deliberate writer
/// behaviour. This test pins the exact shape the validator's seam exemption
/// keys on, so that exemption can never be widened past what the builder
/// actually does:
///
///   1. the escape is on the LONGITUDE axis only (a meridian cannot move
///      latitude), and latitude containment is exact;
///   2. every escaping longitude edge lands on the seam **exactly** (±180.0);
///   3. the declared interval is already wider than 180° — the signature of a
///      plain min/max fold over a dataset with vertices on both sides of ±180.
#[test]
fn seam_split_synthesises_pm180_vertices_the_declared_source_bbox_cannot_contain() {
    let feat = parsed_from_geometry(geojson::Geometry::new(geojson::Value::Polygon(vec![ring(
        &[(178.0, -5.0), (-178.0, -5.0), (-178.0, 5.0), (178.0, 5.0)],
    )])));

    // What the writer declares: a plain, unwrapped min/max over the SOURCE.
    let declared = vertex_bounds(&feat);
    assert_eq!(
        (declared.min_lon, declared.max_lon),
        (-178.0, 178.0),
        "the writer folds source vertices only — it never widens to the seam"
    );
    assert_eq!((declared.min_lat, declared.max_lat), (-5.0, 5.0));

    // What the archive decodes to, after the A1 split.
    let tiles = generate_tiles(std::slice::from_ref(&feat), &config(1, 3), 1).unwrap();
    let o = decoded_bbox(&tiles);
    assert_eq!(
        (o[0], o[2]),
        (-180.0, 180.0),
        "the split must reach the seam on BOTH sides — that is what it is for"
    );

    // (1) Longitude escapes on both sides; latitude does not escape at all.
    assert!(o[0] < declared.min_lon, "low edge must escape: {o:?}");
    assert!(o[2] > declared.max_lon, "high edge must escape: {o:?}");
    assert_eq!(
        (o[1], o[3]),
        (declared.min_lat, declared.max_lat),
        "a meridian cannot move latitude: {o:?}"
    );

    // (2) Each escaping edge lands ON the seam, to the bit.
    assert_eq!(o[0], -180.0);
    assert_eq!(o[2], 180.0);

    // (3) The declared interval is already wider than half the world.
    assert!(
        declared.max_lon - declared.min_lon > 180.0,
        "declared width {} must exceed 180°",
        declared.max_lon - declared.min_lon
    );

    // Nothing was dropped: the pieces land on both dateline columns, and the
    // gate above (`assert_gate`) already holds for this shape.
    assert_gate(&feat, 1, 3);
}

/// The contrast, and the second half of the finding: a seam-crossing
/// **LineString** is NOT split. `clip_trajectory` breaks the run at the
/// `|Δlon| > 180°` edge and simply never emits that edge — no ±180 vertex is
/// synthesised, so the source bbox still contains the decode and check 13 stays
/// silent on lines. The geometry loss is real, uncounted and invisible to the
/// validator; it is reported, not fixed, here (splitting a line at the seam is
/// the polygon splitter's job and is deep surgery).
///
/// Both statements are asserted, because the *combination* is the point: the
/// only geometry class check 13 fires on is the one the builder handles
/// CORRECTLY, and the class the builder silently truncates sails through.
#[test]
fn a_seam_crossing_line_loses_its_crossing_edge_instead_of_being_split() {
    let feat = parsed_from_geometry(geojson::Geometry::new(geojson::Value::LineString(vec![
        vec![170.0, 0.0],
        vec![178.0, 1.0],
        vec![-178.0, 2.0],
        vec![-170.0, 3.0],
    ])));

    let declared = vertex_bounds(&feat);
    assert_eq!((declared.min_lon, declared.max_lon), (-178.0, 178.0));

    let tiles = generate_tiles(std::slice::from_ref(&feat), &config(1, 3), 1).unwrap();
    let o = decoded_bbox(&tiles);

    // No seam vertex is ever synthesised for a line …
    assert!(
        o[0] >= -178.0 && o[2] <= 178.0,
        "a line must not reach the seam: {o:?}"
    );
    // … so the declared source bbox DOES contain the decode, and check 13 —
    // which only ever compares these two boxes — cannot see the loss.
    assert!(
        o[0] >= declared.min_lon
            && o[2] <= declared.max_lon
            && o[1] >= declared.min_lat
            && o[3] <= declared.max_lat,
        "declared {declared:?} must contain {o:?}"
    );

    // The loss itself: no surviving run carries the crossing edge, and the two
    // vertices that flanked it never share a run.
    let runs = decoded_runs(&tiles);
    assert!(!runs.is_empty(), "the line must still produce tiles");
    for run in &runs {
        for w in run.windows(2) {
            assert!(
                (w[1][0] - w[0][0]).abs() <= 180.0,
                "a |Δlon|>180 edge survived: {run:?}"
            );
        }
        let has_west = run.iter().any(|c| c[0] > 0.0);
        let has_east = run.iter().any(|c| c[0] < 0.0);
        assert!(
            !(has_west && has_east),
            "the two hemispheres must never share a run once the edge is dropped: {run:?}"
        );
    }
}

/// The degenerate corner of the same behaviour: a **2-vertex** line straddling
/// the seam splits into two runs of one vertex each, both below
/// `ClipConfig::min_vertices`, so nothing survives the clipper and
/// `place_polyline` dead-letters the WHOLE feature to `place_whole_feature`.
/// The result is one tile carrying the original trans-seam coordinates — an
/// edge that spans 356° of longitude baked into a single tile, with no counter
/// and no warning anywhere.
///
/// Pinned so the behaviour cannot change silently in either direction; it is
/// reported as a builder gap, not fixed here.
#[test]
fn a_two_vertex_seam_crossing_line_dead_letters_into_one_tile_uncounted() {
    let feat = parsed_from_geometry(geojson::Geometry::new(geojson::Value::LineString(vec![
        vec![178.0, 0.0],
        vec![-178.0, 0.0],
    ])));

    let mut writer = NullWriter;
    let stats =
        generate_tiles_streaming(std::slice::from_ref(&feat), &config(3, 3), &mut writer, 1)
            .unwrap();
    assert_eq!(
        stats.antimeridian_fallbacks, 0,
        "the polygon fallback counter does not cover the line path — the dead-letter is UNCOUNTED"
    );

    let tiles = generate_tiles(std::slice::from_ref(&feat), &config(3, 3), 1).unwrap();
    assert_eq!(
        tiles.len(),
        1,
        "the whole feature lands in exactly one tile"
    );
    let runs = decoded_runs(&tiles);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0], vec![[178.0, 0.0], [-178.0, 0.0]]);
    assert!(
        (runs[0][1][0] - runs[0][0][0]).abs() > 180.0,
        "the trans-seam edge is preserved verbatim inside the single tile"
    );
}
