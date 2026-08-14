//! Shared flag → config parsing used by BOTH the `stt-build` CLI and the
//! `stt-serve` dynamic server.
//!
//! A tile served live by `stt-serve` must be configured **byte-identically** to
//! the same tile built offline by `stt-build`. Both reach the same encoder
//! (`tiler::encode_single_tile` / `stt_core::arrow_tile`), but only if they
//! parse the same CLI strings (`--temporal-bucket`, `--vector-group NAME=cols`,
//! `--quantize-attr NAME=PREC`, …) into the same structures and set the same
//! process-wide encoder globals. Centralising that here removes the drift risk
//! of two independent copies — this module is the single source of truth for
//! "flags → `TileConfig` fields + encoder globals".

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

use crate::columnar::AttributeFilter;
use stt_core::arrow_tile::{VectorElem, VectorGroup};
use stt_core::budget::{ImportanceScorer, TileBudget};
use stt_core::metadata::TemporalLodLevel;

/// Parse a duration string like `1h`, `6h`, `1d`, `30m`, `90s`, `500ms` into
/// milliseconds. Accepts a fractional value (`1.5h`). Bare number = ms.
pub fn parse_duration(s: &str) -> Result<u64> {
    let s = s.trim().to_lowercase();

    let mut num_str = String::new();
    let mut unit = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            num_str.push(c);
        } else {
            unit.push(c);
        }
    }

    let value: f64 = num_str
        .parse()
        .with_context(|| format!("Invalid duration value: {s}"))?;

    let multiplier: u64 = match unit.as_str() {
        "ms" | "" => 1,
        "s" | "sec" => 1000,
        "m" | "min" => 60 * 1000,
        "h" | "hr" | "hour" => 60 * 60 * 1000,
        "d" | "day" => 24 * 60 * 60 * 1000,
        _ => anyhow::bail!(
            "Invalid duration unit '{unit}'. Use ms, s, m, h, or d (e.g., '1h', '30m', '6h')"
        ),
    };

    Ok((value * multiplier as f64) as u64)
}

/// Fraction of total estimated byte mass a defaulted coarse tier may cover
/// (TB-9, §2.4). A coarse tier exists to serve LOW-zoom queries; emitting it at
/// every zoom is maximal duplication, and the duplication term is dominated by
/// the high-zoom byte share.
pub const DEFAULT_LOD_MASS_FRACTION: f64 = 0.25;

/// The default zoom cutoff for an un-annotated temporal-LOD tier: the largest
/// `z` whose cumulative estimated byte mass stays within
/// [`DEFAULT_LOD_MASS_FRACTION`] of the total (TB-9).
///
/// `zoom_mass[i]` is the estimated byte mass of zoom `zoom_range.0 + i`.
/// Coarser tiers (higher `tier_index`) are clamped strictly below finer ones so
/// the ladder stays monotone, and everything is clamped into the build's zoom
/// range. With no mass information the caller keeps the legacy
/// `fallback_max_zoom` behaviour.
pub fn default_lod_cutoff(
    zoom_mass: &[u64],
    zoom_range: (u8, u8),
    tier_index: usize,
    fallback_max_zoom: u8,
) -> u8 {
    let (min_zoom, max_zoom) = zoom_range;
    let total: u128 = zoom_mass.iter().map(|&m| u128::from(m)).sum();
    if total == 0 || zoom_mass.is_empty() {
        return fallback_max_zoom;
    }
    let budget = (total as f64 * DEFAULT_LOD_MASS_FRACTION) as u128;
    let mut cumulative: u128 = 0;
    // Always admit the shallowest zoom: a tier that covers nothing is useless,
    // and z=min_zoom is where a coarse tier is most valuable.
    let mut cutoff = min_zoom;
    for (i, &mass) in zoom_mass.iter().enumerate() {
        cumulative += u128::from(mass);
        if cumulative > budget {
            break;
        }
        cutoff = min_zoom.saturating_add(i as u8);
    }
    // Each successively coarser tier sits strictly below the previous one.
    let stepped = cutoff.saturating_sub(tier_index as u8);
    stepped.clamp(min_zoom, max_zoom.max(min_zoom))
}

/// Parse a `--temporal-lod` spec like `"1d,30d"` or `"1d@8,30d@4"`. Each entry
/// is `<duration>` (applies at every zoom) or `<duration>@<zoom>` (applies at
/// zoom <= the given level). Entries are returned in input order so the caller
/// can re-validate sorting against the base bucket.
///
/// Un-annotated entries keep the legacy "every zoom" default here; see
/// [`parse_temporal_lod_with_mass`] for TB-9's byte-mass-aware default.
pub fn parse_temporal_lod(s: &str, fallback_max_zoom: u8) -> Result<Vec<TemporalLodLevel>> {
    parse_temporal_lod_with_mass(s, fallback_max_zoom, None, (0, fallback_max_zoom))
}

/// [`parse_temporal_lod`], with TB-9's byte-mass-aware default for un-annotated
/// tiers. An explicit `@z` ALWAYS wins.
pub fn parse_temporal_lod_with_mass(
    s: &str,
    fallback_max_zoom: u8,
    zoom_mass: Option<&[u64]>,
    zoom_range: (u8, u8),
) -> Result<Vec<TemporalLodLevel>> {
    let mut levels = Vec::new();
    let mut defaulted = 0usize;
    for piece in s.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let (dur, zoom) = match piece.split_once('@') {
            Some((d, z)) => {
                let z: u8 = z
                    .trim()
                    .parse()
                    .with_context(|| format!("invalid zoom in temporal-lod entry '{piece}'"))?;
                (d.trim(), z)
            }
            None => {
                // TB-9: an un-annotated tier no longer defaults to EVERY zoom.
                let z = match zoom_mass {
                    Some(mass) => {
                        default_lod_cutoff(mass, zoom_range, defaulted, fallback_max_zoom)
                    }
                    None => fallback_max_zoom,
                };
                defaulted += 1;
                (piece, z)
            }
        };
        let bucket_ms = parse_duration(dur)
            .with_context(|| format!("invalid duration in temporal-lod entry '{piece}'"))?;
        levels.push(TemporalLodLevel {
            bucket_ms,
            max_zoom_level: zoom,
            // DT-1: `--temporal-lod` builds exact re-bucketed tiers, which is
            // the `union` contract. Absent = union, so this stays byte-identical.
            contract: None,
            method: None,
        });
    }
    Ok(levels)
}

/// Parse repeated `--quantize-attr NAME=PREC` specs into a `name → precision`
/// map. Errors on a missing `=`, a non-numeric precision, or `PREC <= 0`.
pub fn parse_quantize_attrs(specs: &[String]) -> Result<HashMap<String, f64>> {
    let mut attrs: HashMap<String, f64> = HashMap::new();
    for spec in specs {
        let (name, prec) = spec
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--quantize-attr expects NAME=PREC, got {spec:?}"))?;
        let prec: f64 = prec.trim().parse().map_err(|_| {
            anyhow::anyhow!("--quantize-attr {spec:?}: PREC {prec:?} is not a number")
        })?;
        if prec <= 0.0 {
            anyhow::bail!("--quantize-attr {spec:?}: PREC must be > 0");
        }
        attrs.insert(name.trim().to_string(), prec);
    }
    Ok(attrs)
}

/// Parse repeated `--vector-group NAME=col1,col2[:f32|u8]` specs into
/// [`VectorGroup`]s. Default leaf type is `f32`; `:u8` selects 0–255 RGBA.
pub fn parse_vector_groups(specs: &[String]) -> Result<Vec<VectorGroup>> {
    let mut groups: Vec<VectorGroup> = Vec::new();
    for spec in specs {
        let (name, rest) = spec.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("--vector-group expects NAME=COLS[:f32|u8], got {spec:?}")
        })?;
        // Optional trailing `:f32` / `:u8` selects the leaf upload type.
        let (cols_str, elem) = match rest.rsplit_once(':') {
            Some((cols, "u8")) => (cols, VectorElem::U8),
            Some((cols, "f32")) => (cols, VectorElem::F32),
            Some((_, other)) => {
                anyhow::bail!("--vector-group {spec:?}: leaf type {other:?} must be f32 or u8")
            }
            None => (rest, VectorElem::F32),
        };
        let components: Vec<String> = cols_str
            .split(',')
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();
        if components.is_empty() {
            anyhow::bail!("--vector-group {spec:?}: no component columns");
        }
        groups.push(VectorGroup {
            name: name.trim().to_string(),
            components,
            elem,
        });
    }
    Ok(groups)
}

/// The CLI-shaped encoder flags shared by the offline build and the dynamic
/// server: the raw, unparsed strings as `clap` collected them (independent of
/// [`crate::tiler::TileConfig`], which covers tiling rather than encoding).
///
/// Build a value from the CLI flags, then call [`EncoderSettings::resolve`] to
/// get the explicit [`stt_core::arrow_tile::EncoderConfig`] every encode takes
/// as an argument. Both `stt-build` and `stt-serve` go through it, so a served
/// tile is byte-identical to the offline build's.
#[derive(Debug, Clone, Default)]
pub struct EncoderSettings {
    /// `--vertex-time-precision` (ms). `None` keeps the encoder default.
    pub vertex_time_precision: Option<u32>,
    /// `--quantize-coords` ground precision in meters; `0.0` = off (Float64).
    pub quantize_coords_m: f64,
    /// `--quantize-attr NAME=PREC` specs (parsed by [`parse_quantize_attrs`]).
    pub quantize_attr: Vec<String>,
    /// `--quantize-attrs-auto` (every Float64 prop → range-adaptive UInt16).
    pub quantize_attrs_auto: bool,
    /// `--vector-group NAME=cols[:f32|u8]` specs (parsed by [`parse_vector_groups`]).
    pub vector_group: Vec<String>,
    /// `--point-elevation-column NAME` (fold a property into POINT z).
    pub point_elevation_column: Option<String>,
    /// `--no-compact-times`: the kill switch for the compact feature-time
    /// columns. `false` (the default) leaves the feature ON, which is also
    /// what declares the `time-delta` capability.
    pub no_compact_times: bool,
    /// `--quantize-vertex-values`: store the `vertex_value` /
    /// `vertex_value_matrix` leaves as `UInt16` under a per-column
    /// range-adaptive affine (the `vertex-value-quant` capability). Opt-in
    /// because it is lossy; off by default.
    pub quantize_vertex_values: bool,
}

impl EncoderSettings {
    /// A short, human-readable list of the non-default settings these flags
    /// turn on (for the build log). Pure — it parses the same specs
    /// [`resolve`](Self::resolve) does but installs nothing.
    pub fn enabled_summary(&self) -> Result<Vec<String>> {
        let mut enabled: Vec<String> = Vec::new();

        if let Some(p) = self.vertex_time_precision {
            if p != stt_core::arrow_tile::DEFAULT_VERTEX_TIME_MAX_STEP_MS {
                enabled.push(format!("vertex-time-precision={p}ms"));
            }
        }

        if self.quantize_coords_m > 0.0 {
            enabled.push(format!("quantize-coords={}m", self.quantize_coords_m));
        }

        if !self.quantize_attr.is_empty() {
            enabled.push(format!(
                "quantize-attr={:?}",
                parse_quantize_attrs(&self.quantize_attr)?
            ));
        }

        if self.quantize_attrs_auto {
            enabled.push("quantize-attrs-auto".to_string());
        }

        if self.quantize_vertex_values {
            enabled.push("quantize-vertex-values".to_string());
        }

        if !self.vector_group.is_empty() {
            enabled.push(format!(
                "vector-groups={}",
                parse_vector_groups(&self.vector_group)?.len()
            ));
        }

        if let Some(col) = self.point_elevation_column.as_deref() {
            if !col.is_empty() {
                enabled.push(format!("point-elevation-column={col}"));
            }
        }

        // Reported inverted (the feature is ON by default), so the log line
        // names the non-default state exactly like every other entry here.
        if self.no_compact_times {
            enabled.push("no-compact-times".to_string());
        }

        Ok(enabled)
    }

    /// Parse the specs into an explicit [`stt_core::arrow_tile::EncoderConfig`]
    /// WITHOUT touching any process-wide globals — the concurrency- and
    /// multi-config-safe path BOTH the offline build and a dynamic server use
    /// (each dataset/request encodes via
    /// [`stt_core::arrow_tile::encode_tile_with`] with its own config, never
    /// mutating shared state).
    pub fn resolve(&self) -> Result<stt_core::arrow_tile::EncoderConfig> {
        // Fail fast at config time, not per-request at encode time: without
        // this, a server boots cleanly on an invalid precision and 500s
        // every tile (the same floor is enforced again where the value is
        // consumed, so this is UX, not the safety net).
        stt_core::arrow_tile::validate_quantize_coords_m(self.quantize_coords_m)?;
        Ok(stt_core::arrow_tile::EncoderConfig {
            quantize_coords_m: (self.quantize_coords_m > 0.0).then_some(self.quantize_coords_m),
            quantize_attrs: parse_quantize_attrs(&self.quantize_attr)?,
            quantize_attrs_auto: self.quantize_attrs_auto,
            vector_groups: parse_vector_groups(&self.vector_group)?,
            point_elevation_column: self.point_elevation_column.clone().unwrap_or_default(),
            // `.max(1)`: a 0 ceiling would send EVERY layer down the exact
            // `List<Int64>` fallback (no step can be <= 0), silently costing 4x
            // the vertex-time bytes instead of meaning "finest possible step".
            // The retired `set_vertex_time_max_step_ms` global clamped here too
            // — dropping it would have re-typed the column for
            // `--vertex-time-precision 0`.
            vertex_time_max_step_ms: self
                .vertex_time_precision
                .unwrap_or(stt_core::arrow_tile::DEFAULT_VERTEX_TIME_MAX_STEP_MS)
                .max(1),
            compact_times: !self.no_compact_times,
            quantize_vertex_values: self.quantize_vertex_values,
            // The format-v2 fields (`format_version`, `template_collector`)
            // default to v1 / no-collector; the PACK WRITER overrides both from
            // its own `--format-version` (`PackWriter::encoder_config`), so
            // frames can never disagree with the manifest.
            ..stt_core::arrow_tile::EncoderConfig::default()
        })
    }

    /// The `manifest.capabilities` entries these settings imply
    /// (required-to-understand declarations, packed spec §3.1): each names a
    /// feature that RE-TYPES existing tile columns, so an older reader would
    /// silently misdecode without it. Additive features (vector groups,
    /// vertex-time precision) are deliberately NOT capabilities. Returned in
    /// registry order (deterministic; the pack writer canonicalizes anyway).
    pub fn required_capabilities(&self) -> Vec<String> {
        let mut caps: Vec<String> = Vec::new();
        if self.quantize_coords_m > 0.0 {
            caps.push(stt_core::pack::CAPABILITY_COORD_QUANT.to_string());
        }
        if !self.quantize_attr.is_empty() || self.quantize_attrs_auto {
            caps.push(stt_core::pack::CAPABILITY_ATTR_QUANT.to_string());
        }
        if self
            .point_elevation_column
            .as_deref()
            .is_some_and(|c| !c.is_empty())
        {
            caps.push(stt_core::pack::CAPABILITY_ELEVATION_FOLD.to_string());
        }
        // Declared whenever the feature is ENABLED (i.e. unless
        // `--no-compact-times`), not only when a tile actually took a compact
        // form: the choice is per layer per tile, so "some tile in this
        // dataset may carry `TILE_META.st`/`.et`" is the only claim a
        // dataset-level manifest key can honestly make.
        if !self.no_compact_times {
            caps.push(stt_core::pack::CAPABILITY_TIME_DELTA.to_string());
        }
        if self.quantize_vertex_values {
            caps.push(stt_core::pack::CAPABILITY_VERTEX_VALUE_QUANT.to_string());
        }
        caps
    }
}

/// Build the opt-in per-tile budget, or `None` when neither a byte nor a feature
/// cap was set (the default "no thinning" behaviour). With `drop_densest` the
/// budget drops the densest features first (pure `GeometrySize`); otherwise it
/// drops the least-important features (a `Combined` geometry+property score).
pub fn build_tile_budget(
    max_bytes: Option<usize>,
    max_features: Option<usize>,
    drop_densest: bool,
) -> Option<TileBudget> {
    if max_bytes.is_none() && max_features.is_none() {
        return None;
    }
    let max_bytes = max_bytes.unwrap_or(usize::MAX);
    let max_features = max_features.unwrap_or(usize::MAX);
    let scorer = if drop_densest {
        ImportanceScorer::GeometrySize
    } else {
        ImportanceScorer::Combined
    };
    // One cap, one argument: the byte cap is UNCOMPRESSED (what
    // `--maximum-tile-bytes` documents and what `enforce_indexed` binds on).
    // The old constructor took `max_bytes` twice to fill a `max_compressed_size`
    // nothing ever read; that field is deleted (see `TileBudget`'s type docs).
    Some(TileBudget::new(max_bytes, max_features).with_scorer(scorer))
}

/// Resolve `--exclude` / `--include` / `--exclude-all` into an
/// [`AttributeFilter`], validating mutual exclusivity and refusing to drop any
/// `required` column (columns another feature still needs, e.g. a
/// `--heatmap-weight` / `--summary-columns` / `--min-zoom-field` source).
pub fn build_attribute_filter(
    exclude: &[String],
    include: &[String],
    exclude_all: bool,
    required: &[String],
) -> Result<AttributeFilter> {
    let has_exclude = !exclude.is_empty();
    let has_include = !include.is_empty();

    let modes = [has_exclude, has_include, exclude_all]
        .iter()
        .filter(|b| **b)
        .count();
    if modes > 1 {
        anyhow::bail!(
            "--exclude, --include and --exclude-all are mutually exclusive; pass at most one"
        );
    }

    let filter = if exclude_all {
        AttributeFilter::ExcludeAll
    } else if has_include {
        AttributeFilter::Include(include.iter().cloned().collect())
    } else if has_exclude {
        AttributeFilter::Exclude(exclude.iter().cloned().collect())
    } else {
        AttributeFilter::KeepAll
    };

    let dropped_required: Vec<&String> = required.iter().filter(|p| !filter.keeps(p)).collect();
    if !dropped_required.is_empty() {
        let uniq: HashSet<&String> = dropped_required.into_iter().collect();
        let mut names: Vec<&str> = uniq.iter().map(|s| s.as_str()).collect();
        names.sort_unstable();
        anyhow::bail!(
            "attribute filter would drop column(s) still needed by another build feature \
             (--heatmap-weight/--heatmap-class/--summary-columns/--min-zoom-field/\
             --max-zoom-field): {}. Add them to --include (or drop the conflicting flag).",
            names.join(", ")
        );
    }

    Ok(filter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(parse_duration("1h").unwrap(), 3_600_000);
        assert_eq!(parse_duration("30m").unwrap(), 1_800_000);
        assert_eq!(parse_duration("1d").unwrap(), 86_400_000);
        assert_eq!(parse_duration("500ms").unwrap(), 500);
        assert_eq!(parse_duration("1.5h").unwrap(), 5_400_000);
        assert!(parse_duration("5w").is_err());
    }

    #[test]
    fn temporal_lod_parsing() {
        let lod = parse_temporal_lod("1d,30d@4", 8).unwrap();
        assert_eq!(lod.len(), 2);
        assert_eq!(lod[0].bucket_ms, 86_400_000);
        assert_eq!(lod[0].max_zoom_level, 8); // fallback
        assert_eq!(lod[1].max_zoom_level, 4); // explicit @4
    }

    #[test]
    fn quantize_attr_specs() {
        let m = parse_quantize_attrs(&["z=0.05".into(), "speed=0.1".into()]).unwrap();
        assert_eq!(m.get("z"), Some(&0.05));
        assert_eq!(m.get("speed"), Some(&0.1));
        assert!(parse_quantize_attrs(&["bad".into()]).is_err());
        assert!(parse_quantize_attrs(&["z=0".into()]).is_err());
    }

    #[test]
    fn vector_group_specs() {
        let g =
            parse_vector_groups(&["rgba=r,g,b,a:u8".into(), "quat=qx,qy,qz,qw".into()]).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].name, "rgba");
        assert_eq!(g[0].components, vec!["r", "g", "b", "a"]);
        assert!(matches!(g[0].elem, VectorElem::U8));
        assert!(matches!(g[1].elem, VectorElem::F32));
        assert!(parse_vector_groups(&["bad".into()]).is_err());
        assert!(parse_vector_groups(&["x=a:f64".into()]).is_err());
    }

    #[test]
    fn required_capabilities_from_settings() {
        // Compact feature times are ON by default, so the DEFAULT settings
        // declare exactly that one capability and nothing else.
        assert_eq!(
            EncoderSettings::default().required_capabilities(),
            ["time-delta"]
        );

        // Each re-typing feature declares its registry entry.
        let all = EncoderSettings {
            quantize_coords_m: 1.0,
            quantize_attr: vec!["z=0.05".into()],
            point_elevation_column: Some("z".into()),
            ..Default::default()
        };
        assert_eq!(
            all.required_capabilities(),
            ["coord-quant", "attr-quant", "elevation-fold", "time-delta"]
        );

        // Auto attr-quantization alone also re-types columns.
        let auto = EncoderSettings {
            quantize_attrs_auto: true,
            no_compact_times: true,
            ..Default::default()
        };
        assert_eq!(auto.required_capabilities(), ["attr-quant"]);

        // Additive features (vector groups, vertex-time precision — including
        // its `List<UInt32>` delta tier, which `vt` already describes) and an
        // empty elevation column never need a capability.
        let additive = EncoderSettings {
            vector_group: vec!["rgba=r,g,b,a:u8".into()],
            vertex_time_precision: Some(50),
            point_elevation_column: Some(String::new()),
            no_compact_times: true,
            ..Default::default()
        };
        assert!(additive.required_capabilities().is_empty());
    }

    /// `--no-compact-times` is the single switch behind BOTH the encoder
    /// setting and the capability declaration: they can never disagree.
    #[test]
    fn no_compact_times_suppresses_the_encoder_flag_and_the_capability() {
        let off = EncoderSettings {
            no_compact_times: true,
            ..Default::default()
        };
        assert!(!off.resolve().unwrap().compact_times);
        assert!(off.required_capabilities().is_empty());
        assert_eq!(off.enabled_summary().unwrap(), ["no-compact-times"]);

        let on = EncoderSettings::default();
        assert!(on.resolve().unwrap().compact_times);
        assert_eq!(on.required_capabilities(), ["time-delta"]);
        assert!(on.enabled_summary().unwrap().is_empty());
    }

    /// `--quantize-vertex-values` is the single switch behind BOTH the encoder
    /// setting and its capability declaration, and it is OFF by default (it is
    /// the one lossy encoding in this set).
    #[test]
    fn quantize_vertex_values_drives_the_encoder_flag_and_the_capability() {
        let off = EncoderSettings::default();
        assert!(!off.resolve().unwrap().quantize_vertex_values);
        assert_eq!(off.required_capabilities(), ["time-delta"]);
        assert!(off.enabled_summary().unwrap().is_empty());

        let on = EncoderSettings {
            quantize_vertex_values: true,
            ..Default::default()
        };
        assert!(on.resolve().unwrap().quantize_vertex_values);
        assert_eq!(
            on.required_capabilities(),
            ["time-delta", "vertex-value-quant"]
        );
        assert_eq!(on.enabled_summary().unwrap(), ["quantize-vertex-values"]);
    }

    #[test]
    fn budget_off_by_default() {
        assert!(build_tile_budget(None, None, false).is_none());
        let b = build_tile_budget(Some(50_000), None, false).expect("budget");
        assert_eq!(b.max_uncompressed_size, 50_000);
        assert_eq!(b.max_feature_count, usize::MAX);
    }

    #[test]
    fn attribute_filter_modes_and_guard() {
        assert!(matches!(
            build_attribute_filter(&[], &[], false, &[]).unwrap(),
            AttributeFilter::KeepAll
        ));
        // Mutually exclusive.
        assert!(build_attribute_filter(&["a".into()], &["b".into()], false, &[]).is_err());
        // Guard: excluding a required column errors.
        let err = build_attribute_filter(&["mag".into()], &[], false, &["mag".into()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("mag"), "got: {err}");
    }

    // ------------------------------------------------------------------
    // TB-9 — temporal-LOD default zoom cutoffs (§2.4)
    // ------------------------------------------------------------------

    /// Byte mass grows steeply with zoom, so a defaulted coarse tier stops well
    /// short of the deepest zoom instead of covering every level.
    #[test]
    fn an_unannotated_tier_no_longer_defaults_to_every_zoom() {
        // z0..z8, each level ~4x the previous — the usual pyramid shape.
        let mass: Vec<u64> = (0..9).map(|z| 4u64.pow(z)).collect();
        let cutoff = default_lod_cutoff(&mass, (0, 8), 0, 8);
        assert!(
            cutoff < 8,
            "a defaulted tier must not reach the deepest zoom (got z{cutoff})"
        );
        let levels = parse_temporal_lod_with_mass("30d", 8, Some(&mass), (0, 8)).unwrap();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].max_zoom_level, cutoff);
        // The legacy path still covers everything, so the change is visible.
        let legacy = parse_temporal_lod("30d", 8).unwrap();
        assert_eq!(legacy[0].max_zoom_level, 8);
        assert!(levels[0].max_zoom_level < legacy[0].max_zoom_level);
    }

    /// An explicit `@z` always wins — the default only fills in what the user
    /// left unsaid.
    #[test]
    fn an_explicit_zoom_annotation_always_wins() {
        let mass: Vec<u64> = (0..9).map(|z| 4u64.pow(z)).collect();
        let levels = parse_temporal_lod_with_mass("30d@7,1h@3", 8, Some(&mass), (0, 8)).unwrap();
        assert_eq!(levels[0].max_zoom_level, 7);
        assert_eq!(levels[1].max_zoom_level, 3);
    }

    /// Successively coarser tiers step strictly downward, so the ladder stays
    /// monotone and two tiers never claim the same zoom.
    #[test]
    fn defaulted_tiers_step_strictly_downward() {
        let mass: Vec<u64> = (0..9).map(|z| 4u64.pow(z)).collect();
        let levels = parse_temporal_lod_with_mass("1h,6h,30d", 8, Some(&mass), (0, 8)).unwrap();
        for w in levels.windows(2) {
            assert!(
                w[1].max_zoom_level < w[0].max_zoom_level,
                "coarser tier must sit strictly below: {:?}",
                levels
            );
        }
    }

    /// No mass information (a `--single-pass` build) reproduces the legacy
    /// behaviour exactly — the documented fallback.
    #[test]
    fn without_byte_mass_the_legacy_default_is_reproduced() {
        for spec in ["30d", "1h,6h,30d", "30d@4"] {
            let legacy = parse_temporal_lod(spec, 9).unwrap();
            let none = parse_temporal_lod_with_mass(spec, 9, None, (0, 9)).unwrap();
            assert_eq!(legacy, none, "spec {spec:?}");
        }
    }

    /// Degenerate mass never produces a cutoff outside the build's zoom range.
    #[test]
    fn the_cutoff_is_always_inside_the_zoom_range() {
        assert_eq!(
            default_lod_cutoff(&[], (2, 9), 0, 9),
            9,
            "no mass = fallback"
        );
        assert_eq!(
            default_lod_cutoff(&[0, 0, 0], (2, 9), 0, 9),
            9,
            "zero mass = fallback"
        );
        // A deep tier index cannot step below min_zoom.
        let mass: Vec<u64> = (0..8).map(|z| 4u64.pow(z)).collect();
        for tier in 0..12 {
            let z = default_lod_cutoff(&mass, (2, 9), tier, 9);
            assert!((2..=9).contains(&z), "tier {tier} produced z{z}");
        }
    }
}
