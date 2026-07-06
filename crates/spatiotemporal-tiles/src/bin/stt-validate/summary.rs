//! Decode-tier summary-tier cell-id validity.
//!
//! When `metadata.summary_tier` is present, the `id` column of every
//! summary-layer row IS the aggregation cell index (Uber H3 or CARTO
//! Quadbin): the client reconstructs the cell polygon from the id alone, so
//! an archive whose summary ids are anything else (e.g. a sequential feature
//! counter) renders BLANK with no error anywhere in the stack — exactly the
//! gap that shipped three blank demo datasets. This check makes it loud.
//!
//! Validity is per scheme, at the tier's mapped resolution for the tile's
//! zoom (`cell_resolution_per_zoom`):
//!
//! - **H3** — the id parses as an [`h3o::CellIndex`] (the same crate/version
//!   the builder mints cells with) whose resolution matches.
//! - **Quadbin** — the id carries CARTO's exact u64 bit layout at the mapped
//!   Quadbin zoom. The layout is normatively owned by the builder
//!   (`crates/stt-build/src/quadbin.rs`); the bit checks here mirror it
//!   rather than pulling the whole build pipeline into the validator — the
//!   `#[cfg(feature = "build")]` parity test below pins the two against each
//!   other.

use arrow::array::{Array, UInt64Array};
use stt_core::arrow_tile::DecodedLayer;
use stt_core::metadata::{SummaryScheme, SummaryTier};

/// Check every summary-layer row's `id` against the tier's cell scheme at
/// the resolution mapped for `zoom`. Returns one issue per offending layer
/// (count + first offender); the caller prepends the tile id. Layers not
/// named `tier.layer_name` are ignored (raw layers carry feature ids, not
/// cell ids), as are missing/mistyped `id` columns (the schema check's
/// finding).
pub fn check_summary_cells(tier: &SummaryTier, zoom: u8, layers: &[DecodedLayer]) -> Vec<String> {
    let mut issues = Vec::new();
    let res = tier.resolution_for_zoom(zoom);
    for layer in layers {
        if layer.name != tier.layer_name {
            continue;
        }
        let Some(col) = layer.batch.column_by_name("id") else {
            continue;
        };
        let Some(ids) = col.as_any().downcast_ref::<UInt64Array>() else {
            continue;
        };
        let mut bad = 0u64;
        let mut first: Option<(usize, u64)> = None;
        for i in 0..ids.len() {
            if ids.is_null(i) {
                continue;
            }
            let id = ids.value(i);
            let ok = match tier.scheme {
                SummaryScheme::H3 => h3_cell_is_valid(id, res),
                SummaryScheme::Quadbin => quadbin_cell_is_valid(id, res),
            };
            if !ok {
                bad += 1;
                if first.is_none() {
                    first = Some((i, id));
                }
            }
        }
        if let Some((row, id)) = first {
            let scheme = match tier.scheme {
                SummaryScheme::H3 => "H3",
                SummaryScheme::Quadbin => "Quadbin",
            };
            issues.push(format!(
                "layer '{}': {bad} of {} id(s) are not valid {scheme} cells at resolution {res} \
                 (zoom {zoom}); first: row {row} id {id:#018x} — summary ids must be the cell \
                 index itself (a sequential id column renders blank; rebuild with current \
                 stt-build)",
                layer.name,
                ids.len(),
            ));
        }
    }
    issues
}

/// A valid H3 cell at exactly `res`, parsed by the same `h3o` crate the
/// builder mints cells with.
fn h3_cell_is_valid(id: u64, res: u8) -> bool {
    h3o::CellIndex::try_from(id).is_ok_and(|c| u8::from(c.resolution()) == res)
}

/// A valid CARTO Quadbin "tile"-mode cell at exactly zoom `z`, per the u64
/// layout in `crates/stt-build/src/quadbin.rs` (the authority — see the
/// parity test):
///
/// ```text
///   63        : reserved/sign (0)
///   62 .. 60  : header   = 0b100
///   59        : mode     = 1 ("tile")
///   58 .. 57  : mode-dep = 0
///   56 .. 52  : zoom     (5 bits, 0..=26)
///   51 .. 0   : Morton x/y, LEFT-aligned; low (52 - 2z) bits all-ones fill
/// ```
///
/// Any left-aligned 2z-bit Morton payload deinterleaves to in-range x/y, so
/// validity is exactly: header/mode bits, zoom field, and the ones-fill.
fn quadbin_cell_is_valid(id: u64, z: u8) -> bool {
    if (id >> 57) != 0b0100_100 {
        return false;
    }
    let zf = ((id >> 52) & 0x1f) as u8;
    if zf != z || zf > 26 {
        return false;
    }
    let fill = (1u64 << (52 - 2 * u64::from(zf))) - 1;
    id & fill == fill
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::arrow_tile::{decode_tile, encode_tile, ColumnarLayer, GeometryColumn};

    /// CARTO's published reference value for tile (z=0, x=0, y=0) — the same
    /// constant stt-build's quadbin tests pin against.
    const QUADBIN_Z0: u64 = 0x480f_ffff_ffff_ffff;

    fn tier(scheme: SummaryScheme, resolutions: Vec<u8>) -> SummaryTier {
        SummaryTier {
            scheme,
            min_zoom: 0,
            max_zoom: resolutions.len() as u8 - 1,
            cell_resolution_per_zoom: resolutions,
            columns: vec![],
            layer_name: "summary".to_string(),
            sub_buckets: 1,
        }
    }

    fn summary_layers(name: &str, ids: Vec<u64>) -> Vec<DecodedLayer> {
        let n = ids.len();
        let layer = ColumnarLayer {
            name: name.into(),
            feature_ids: ids,
            start_times: vec![0; n],
            end_times: vec![1; n],
            geometry: GeometryColumn::Point(vec![[0.0, 0.0]; n]),
            vertex_times: None,
            vertex_values: None,
            triangles: None,
            vertex_value_matrix: None,
            properties: vec![],
        };
        decode_tile(&encode_tile(&[layer]).unwrap()).unwrap()
    }

    fn h3_cell(lat: f64, lon: f64, res: u8) -> u64 {
        h3o::LatLng::new(lat, lon)
            .unwrap()
            .to_cell(h3o::Resolution::try_from(res).unwrap())
            .into()
    }

    #[test]
    fn valid_h3_cells_pass_and_sequential_ids_fail() {
        let t = tier(SummaryScheme::H3, vec![5]);
        let good =
            summary_layers("summary", vec![h3_cell(45.0, -73.5, 5), h3_cell(45.1, -73.6, 5)]);
        assert!(check_summary_cells(&t, 0, &good).is_empty());

        // The exact shipped-blank-demos failure: a sequential integer id column.
        let bad = summary_layers("summary", vec![0, 1, 2]);
        let issues = check_summary_cells(&t, 0, &bad);
        assert_eq!(issues.len(), 1, "issues were: {issues:?}");
        assert!(issues[0].contains("3 of 3"), "issue was: {}", issues[0]);
        assert!(issues[0].contains("H3"), "issue was: {}", issues[0]);
    }

    #[test]
    fn h3_cell_at_the_wrong_resolution_fails() {
        let t = tier(SummaryScheme::H3, vec![5]);
        let wrong = summary_layers("summary", vec![h3_cell(45.0, -73.5, 4)]);
        let issues = check_summary_cells(&t, 0, &wrong);
        assert_eq!(issues.len(), 1, "issues were: {issues:?}");
        assert!(issues[0].contains("1 of 1"), "issue was: {}", issues[0]);
    }

    #[test]
    fn quadbin_reference_cell_passes_and_sequential_ids_fail() {
        let t = tier(SummaryScheme::Quadbin, vec![0]);
        let good = summary_layers("summary", vec![QUADBIN_Z0]);
        assert!(check_summary_cells(&t, 0, &good).is_empty());

        let bad = summary_layers("summary", vec![0, 1, 2]);
        let issues = check_summary_cells(&t, 0, &bad);
        assert_eq!(issues.len(), 1, "issues were: {issues:?}");
        assert!(issues[0].contains("3 of 3"), "issue was: {}", issues[0]);
        assert!(issues[0].contains("Quadbin"), "issue was: {}", issues[0]);

        // Right layout, wrong zoom for the tier's resolution table.
        let t3 = tier(SummaryScheme::Quadbin, vec![3]);
        let issues = check_summary_cells(&t3, 0, &summary_layers("summary", vec![QUADBIN_Z0]));
        assert_eq!(issues.len(), 1, "issues were: {issues:?}");
    }

    #[test]
    fn non_summary_layers_are_ignored() {
        let t = tier(SummaryScheme::H3, vec![5]);
        let layers = summary_layers("raw", vec![0, 1, 2]);
        assert!(check_summary_cells(&t, 0, &layers).is_empty());
    }

    /// Parity with the authority: every cell stt-build's Quadbin encoder
    /// mints must pass this module's bit checks at its zoom and fail at any
    /// other, across the zoom band and both axes' extremes. Runs whenever the
    /// `build` feature is enabled (it is in the default set), so drift
    /// between the mirrored layout and `stt_build::quadbin` fails the build.
    #[cfg(feature = "build")]
    #[test]
    fn quadbin_validity_matches_stt_build_encoder() {
        use stt_build::quadbin::tile_to_quadbin;
        for z in [0u8, 1, 3, 7, 13, 26] {
            let max = (1u32 << z) - 1;
            for (x, y) in [(0, 0), (max, 0), (0, max), (max, max), (max / 2, max / 3)] {
                let q = tile_to_quadbin(z, x, y);
                assert!(quadbin_cell_is_valid(q, z), "z={z} x={x} y={y} q={q:#x}");
                if z > 0 {
                    assert!(!quadbin_cell_is_valid(q, z - 1), "z={z} x={x} y={y}");
                }
                if z < 26 {
                    assert!(!quadbin_cell_is_valid(q, z + 1), "z={z} x={x} y={y}");
                }
            }
        }
        // Sequential integers never validate at any zoom.
        for id in 0u64..64 {
            for z in 0u8..=26 {
                assert!(!quadbin_cell_is_valid(id, z), "id={id} z={z}");
            }
        }
    }
}
