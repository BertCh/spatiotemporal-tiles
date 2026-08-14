//! Blob-ordering audit over a built archive: measure per-ordering range-read
//! cost and recommend `--blob-ordering`.
//!
//! Where the layout advisor's ordering hint is an unsimulated access-shape
//! guess (`advisors::layout`), this runs the shared simulator
//! ([`stt_core::ordering_sim`]) over the archive's real native tiles and ranks
//! the four orderings by the range reads the three canonical access patterns
//! (scrub a viewport across time; pan one instant across space; play a sliding
//! time window through that viewport) would cost. It reads only the directory —
//! no payload decode — so it is cheap on any size.
//!
//! The weighting is per-dataset and derived from the archive's own metadata
//! (`layer_hint` + distinct bucket count), and the archive records the workload
//! its layout was chosen under — so this audit can report not only "your
//! ordering is not the cheapest" but "your ordering was chosen under a workload
//! model that has since moved".

use anyhow::Result;
use serde::{Deserialize, Serialize};

use stt_core::curve::{self, BlobOrdering};
use stt_core::metadata::OrderingWorkload;
use stt_core::ordering_sim::{self, SimOptions, TileSample};

use crate::packed::PackedTileset;

/// One ordering's simulated cost (range reads + bytes read per query + totals).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderingCostRow {
    pub ordering: String,
    pub scrub_reads: u64,
    pub scrub_bytes: u64,
    pub pan_reads: u64,
    pub pan_bytes: u64,
    /// Playback: range reads summed over every playhead advance. Reported
    /// **unconditionally**, even when the playback weight is 0, so the number
    /// exists before any weighting flip is argued for.
    pub playback_reads: u64,
    /// Playback: bytes transferred summed over every playhead advance.
    pub playback_bytes: u64,
    /// The buffered-runway term: the blended cost of the WORST single playhead
    /// advance. This is the stall proxy — an ordering with a cheap playback
    /// total but one catastrophic advance still drops frames.
    pub playback_worst_advance_cost: u64,
    /// Legacy two-query read total (scrub + pan), kept for report continuity.
    pub total_reads: u64,
    /// Legacy two-query byte total (scrub + pan), kept for report continuity.
    pub total_bytes: u64,
    /// The weighted blended cost the ranking uses. With legacy weights this is
    /// exactly `total_bytes + total_reads * gap`.
    pub total_cost: u64,
    pub recommended: bool,
}

/// The full ordering audit for one archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderAuditReport {
    /// Native-tier tiles the simulation ran over.
    pub native_tiles: usize,
    /// Coalescing gap used (bytes) — the reader's real default.
    pub coalesce_gap_bytes: u64,
    /// Pack-object target used (bytes): the reader coalesces per-pack, so runs
    /// are force-closed at pack boundaries. Derived from the archive's packs.
    pub pack_bytes: u64,
    /// Every ordering, cheapest-first.
    pub orderings: Vec<OrderingCostRow>,
    /// The measured-best ordering.
    pub recommended: String,
    /// What `--blob-ordering auto` (the cardinality heuristic) would pick.
    pub auto_choice: String,
    /// The ordering the archive was built with (`manifest.blobOrdering`);
    /// `None` for pre-2026-07 archives that don't record it.
    pub current: Option<String>,
    /// The workload model THIS audit ranked under, derived from the archive's
    /// own `layer_hint` + distinct bucket count plus today's reader-mirroring
    /// constants.
    pub workload: OrderingWorkload,
    /// The dominant layer kind the weights were derived from. `None` on
    /// pre-style-hints archives and on `--streaming` builds (which carry no
    /// in-memory feature slice and so bake no hint) — the weighting then falls
    /// to the generalist row and [`Self::low_confidence_weighting`] is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_hint: Option<String>,
    /// True when no `layer_hint` was available, so the weights are the
    /// generalist default rather than a dataset-specific pick.
    pub low_confidence_weighting: bool,
    /// The workload the archive RECORDS having been laid out under. Only a
    /// `measured` build records one; `None` for `auto`/explicit orderings and
    /// for every archive built before the field existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recorded_workload: Option<OrderingWorkload>,
    /// Set when the recorded workload differs from what this audit would use —
    /// including a change to the reader-mirroring `coalesce_gap_bytes`, which
    /// invalidates the layout's premise without changing a single archive byte.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workload_drift: Option<String>,
    /// Set when `recommended == "spatial"` on a dataset spanning multiple time
    /// buckets: the cost ranking optimizes range-read cost (scrub + pan) but
    /// does NOT model time-PLAYBACK buffering, and `spatial` scatters each
    /// bucket's tiles across the pack — which can stall a player's
    /// buffered-range gate. `None` when spatial isn't recommended or the dataset
    /// is single-bucket (no playback dimension to stall).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playback_caveat: Option<String>,
}

/// Knobs for [`order_audit_with`]. Defaults reproduce [`order_audit`].
#[derive(Debug, Clone, Copy, Default)]
pub struct OrderAuditOptions {
    /// Which query weighting to rank under. `Derived` (the default) mirrors
    /// what a `--blob-ordering measured` build would do today; `Legacy` pins
    /// the pre-M4 scrub+pan weighting, which is what makes a before/after fleet
    /// re-audit a controlled comparison rather than two unrelated runs.
    pub workload: ordering_sim::OrderingWorkloadMode,
}

/// Audit an opened archive's blob ordering under today's derived weighting.
pub fn order_audit(tileset: &PackedTileset) -> Result<OrderAuditReport> {
    order_audit_with(tileset, OrderAuditOptions::default())
}

/// Audit an opened archive's blob ordering.
pub fn order_audit_with(
    tileset: &PackedTileset,
    audit_opts: OrderAuditOptions,
) -> Result<OrderAuditReport> {
    let (_min_z, meta_max_z) = tileset.zoom_range();
    let bucket_ms = (tileset.temporal_bucket_ms().max(1)) as i64;

    // Native tier = the metadata's declared max zoom. An archive whose metadata
    // OVERSTATES max_zoom (declared 14, tiled to 10) would otherwise select no
    // entries at all and silently audit an empty simulation, so fall back to the
    // deepest zoom the directory actually holds. Metadata stays authoritative
    // whenever it names a populated tier, which keeps a declared summary/LOD
    // tier above the native one from being mistaken for it.
    let max_z = if tileset.entries().iter().any(|e| e.zoom == meta_max_z) {
        meta_max_z
    } else {
        tileset
            .entries()
            .iter()
            .map(|e| e.zoom)
            .max()
            .unwrap_or(meta_max_z)
    };

    // Directory-only projection to the simulator's view (len = COMPRESSED
    // on-disk blob size). Native tier only — coarse LOD tiers are aggregates.
    let samples: Vec<TileSample> = tileset
        .entries()
        .iter()
        .filter(|e| e.zoom == max_z)
        .map(|e| TileSample {
            z: e.zoom,
            x: e.x,
            y: e.y,
            hilbert: e.hilbert,
            time_start: e.time_start,
            tb: e.time_start.div_euclid(bucket_ms),
            len: e.length as u64,
        })
        .collect();

    // Coalesce per-pack, using the archive's real pack target. The largest pack
    // object is the target (packs fill to target then cut; only the last is
    // short), so max pack length is the target proxy — floored at 1 MiB.
    let pack_bytes = tileset
        .manifest()
        .packs
        .iter()
        .map(|p| p.length)
        .max()
        .unwrap_or(ordering_sim::DEFAULT_PACK_BYTES)
        .max(1 << 20);
    // Distinct native-tier time buckets: the same signal the writer weights
    // with, and the same one the playback caveat has always keyed off.
    let distinct_buckets = {
        let mut tbs: Vec<i64> = samples.iter().map(|s| s.tb).collect();
        tbs.sort_unstable();
        tbs.dedup();
        tbs.len()
    };
    // Weights from the archive's own metadata — a pure function of dataset
    // DOMAIN facts (dominant layer kind, bucket count), never of the sample.
    let layer_hint = tileset
        .metadata()
        .style_hints
        .as_ref()
        .and_then(|h| h.layer_hint.clone());
    let weights = match audit_opts.workload {
        ordering_sim::OrderingWorkloadMode::Derived => {
            ordering_sim::workload_weights(layer_hint.as_deref(), distinct_buckets)
        }
        ordering_sim::OrderingWorkloadMode::Legacy => ordering_sim::LEGACY_WEIGHTS,
    };
    let opts = SimOptions {
        pack_bytes,
        weights,
        ..SimOptions::default()
    };
    let ranked = ordering_sim::evaluate(&samples, opts);
    // The recommendation is the cheapest SELECTABLE ordering (never morton3) —
    // identical to what `--blob-ordering measured` would resolve.
    let recommended = ordering_sim::measured_ordering(&samples, opts);

    let orderings: Vec<OrderingCostRow> = ranked
        .iter()
        .map(|c| OrderingCostRow {
            ordering: c.ordering.as_str().to_string(),
            scrub_reads: c.scrub.reads,
            scrub_bytes: c.scrub.bytes_read,
            pan_reads: c.pan.reads,
            pan_bytes: c.pan.bytes_read,
            playback_reads: c.playback.reads,
            playback_bytes: c.playback.bytes_read,
            playback_worst_advance_cost: c.playback_worst_advance,
            total_reads: c.total_reads,
            total_bytes: c.total_bytes_read,
            total_cost: c.cost,
            recommended: c.ordering == recommended,
        })
        .collect();

    // `auto`'s pick over the occupied extent — same F1 logic as finalize.
    let auto_choice = if samples.is_empty() {
        BlobOrdering::SpatialMajor
    } else {
        let (mut x_min, mut x_max, mut y_min, mut y_max) = (u32::MAX, 0u32, u32::MAX, 0u32);
        let (mut tb_min, mut tb_max) = (i64::MAX, i64::MIN);
        for s in &samples {
            x_min = x_min.min(s.x);
            x_max = x_max.max(s.x);
            y_min = y_min.min(s.y);
            y_max = y_max.max(s.y);
            tb_min = tb_min.min(s.tb);
            tb_max = tb_max.max(s.tb);
        }
        let space_bits = curve::bits_for((x_max - x_min).max(y_max - y_min) as u64 + 1);
        let time_bits = curve::bits_for((tb_max - tb_min).max(0) as u64 + 1);
        BlobOrdering::choose(space_bits, time_bits)
    };

    let recommended_str = recommended.as_str().to_string();

    // `spatial` minimizes scrub+pan cost but scatters each time bucket's tiles
    // across the pack, so a time-window (playback) read fetches many small
    // non-contiguous ranges and can stall the player's buffered-range gate.
    // When the playback family carries weight that tension is INSIDE the
    // ranking, so the note demotes from "the model can't see this" to "the
    // model saw it and spatial still won"; at weight 0 the original warning
    // stands.
    let playback_caveat =
        playback_caveat_weighted(&recommended_str, distinct_buckets, weights.playback);

    let workload = OrderingWorkload {
        scrub: weights.scrub,
        pan: weights.pan,
        playback: weights.playback,
        playback_window_buckets: opts.playback_window_buckets,
        runway_multiplier: opts.runway_multiplier,
        coalesce_gap_bytes: opts.coalesce_gap_bytes,
    };
    // The canonical key is top-level `orderingWorkload`, beside the
    // `blobOrdering` it co-versions; `metadata.ordering_workload` is the
    // reader-compat mirror. Prefer the canonical one and fall back to the
    // mirror so archives written before the key moved still audit.
    let recorded_workload = tileset
        .manifest()
        .ordering_workload
        .or(tileset.metadata().ordering_workload);
    let workload_drift = workload_drift_for(recorded_workload.as_ref(), &workload);

    Ok(OrderAuditReport {
        native_tiles: samples.len(),
        coalesce_gap_bytes: opts.coalesce_gap_bytes,
        pack_bytes: opts.pack_bytes,
        orderings,
        recommended: recommended_str,
        auto_choice: auto_choice.as_str().to_string(),
        current: tileset.manifest().blob_ordering.clone(),
        workload,
        low_confidence_weighting: layer_hint.is_none(),
        layer_hint,
        recorded_workload,
        workload_drift,
        playback_caveat,
    })
}

/// Describe how an archive's RECORDED workload differs from the one this audit
/// would use today, or `None` when they agree (or when nothing was recorded).
///
/// Two independent kinds of drift, and the second is the one §6.2 asked for:
/// the weights may have been re-fit, or the reader-mirroring
/// `coalesce_gap_bytes` may have moved. The second changes no archive byte, so
/// without this flag a reader-side gap change would silently invalidate every
/// `measured` layout in the fleet.
pub(crate) fn workload_drift_for(
    recorded: Option<&OrderingWorkload>,
    current: &OrderingWorkload,
) -> Option<String> {
    let recorded = recorded?;
    if recorded == current {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if (recorded.scrub, recorded.pan, recorded.playback)
        != (current.scrub, current.pan, current.playback)
    {
        parts.push(format!(
            "weights (scrub, pan, playback) {:?} → {:?}",
            (recorded.scrub, recorded.pan, recorded.playback),
            (current.scrub, current.pan, current.playback)
        ));
    }
    if recorded.playback_window_buckets != current.playback_window_buckets {
        parts.push(format!(
            "playback window {} → {} buckets",
            recorded.playback_window_buckets, current.playback_window_buckets
        ));
    }
    if recorded.runway_multiplier != current.runway_multiplier {
        parts.push(format!(
            "runway multiplier {}× → {}×",
            recorded.runway_multiplier, current.runway_multiplier
        ));
    }
    if recorded.coalesce_gap_bytes != current.coalesce_gap_bytes {
        parts.push(format!(
            "coalescing gap {} B → {} B (the READER-mirroring constant: this layout was \
             optimised for a client that fuses across a different gap)",
            recorded.coalesce_gap_bytes, current.coalesce_gap_bytes
        ));
    }
    Some(format!(
        "this archive was laid out under a different workload model — {}. Its ordering is not \
         wrong, but it was chosen to minimise a cost this audit no longer computes; rebuild with \
         `--blob-ordering measured` to re-derive it.",
        parts.join("; ")
    ))
}

/// The playback caveat for a recommendation, if warranted. `spatial` minimizes
/// range-read cost but scatters each time bucket's tiles across the pack, so a
/// time-window (playback) read fetches many small non-contiguous ranges and can
/// stall the player's buffered-range gate. Fires only when spatial is the
/// recommendation AND the dataset spans multiple time buckets (a single-bucket
/// dataset has no playback dimension to stall). Kept as a pure helper so the
/// policy is unit-testable without coercing the cost model into a spatial win.
/// Shared with the pre-build layout advisor (`advisors::layout`) so both the
/// post-build audit and the pre-build recommendation word the tension identically.
pub(crate) fn playback_caveat_for(recommended: &str, distinct_buckets: usize) -> Option<String> {
    playback_caveat_weighted(recommended, distinct_buckets, 0)
}

/// The caveat, aware of whether playback was actually PRICED.
///
/// At `playback_weight == 0` the model genuinely cannot see the playback axis,
/// so the note is a warning about the model — the historical wording, kept
/// verbatim because the pre-build advisor still ranks that way. At a nonzero
/// weight the playback query and its buffered-runway term are inside the cost,
/// so a spatial win is a measured tradeoff, not a blind spot; the note demotes
/// to naming that tradeoff and pointing at the playback columns.
pub(crate) fn playback_caveat_weighted(
    recommended: &str,
    distinct_buckets: usize,
    playback_weight: u32,
) -> Option<String> {
    if recommended != "spatial" || distinct_buckets <= 1 {
        return None;
    }
    if playback_weight == 0 {
        Some(format!(
            "`spatial` is cheapest for range-reads but is NOT playback-optimal: it scatters each time \
             bucket's tiles across the pack, so time-window (playback) reads fetch many small \
             non-contiguous ranges and can stall the player's buffered-range gate. This dataset spans \
             {distinct_buckets} time buckets — if it drives time PLAYBACK (not just static scrub/pan), \
             prefer `--blob-ordering time-major`. The cost ranking here does not model playback buffering."
        ))
    } else {
        Some(format!(
            "`spatial` won with the playback query PRICED IN (weight {playback_weight} over \
             {distinct_buckets} time buckets), so this is a measured tradeoff, not a blind spot: it \
             pays a higher worst-advance cost and buys it back on scrub and pan. Compare the \
             `worst adv` column — if this dataset drives continuous playback more than the weighting \
             assumes, `--blob-ordering time-major` trades the other way."
        ))
    }
}

fn mib(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{} KiB", bytes / 1024)
    } else {
        format!("{bytes} B")
    }
}

/// Render an audit report as a doctor-style banner + table.
pub fn format_text(r: &OrderAuditReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Blob-ordering audit — {} native tiles, {} coalescing gap, {} packs",
        r.native_tiles,
        mib(r.coalesce_gap_bytes),
        mib(r.pack_bytes)
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "   {:<11} {:>10} {:>12} {:>10} {:>12} {:>12}",
        "ordering", "range rds", "bytes read", "play rds", "worst adv", "cost"
    );
    for row in &r.orderings {
        let mark = if row.recommended { " * " } else { "   " };
        let note = if row.ordering == "morton3" {
            "  (research only)"
        } else {
            ""
        };
        let _ = writeln!(
            s,
            "{}{:<11} {:>10} {:>12} {:>10} {:>12} {:>12}{}",
            mark,
            row.ordering,
            row.total_reads,
            mib(row.total_bytes),
            row.playback_reads,
            mib(row.playback_worst_advance_cost),
            mib(row.total_cost),
            note
        );
    }
    let _ = writeln!(s);
    let w = &r.workload;
    let _ = writeln!(
        s,
        "  cost = Σ weight × (bytes read + reads × {} gap) + {}× the worst playhead advance",
        mib(r.coalesce_gap_bytes),
        w.runway_multiplier
    );
    let _ = writeln!(
        s,
        "  workload    : scrub {} / pan {} / playback {} (layer hint {}{})",
        w.scrub,
        w.pan,
        w.playback,
        r.layer_hint.as_deref().unwrap_or("none"),
        if r.low_confidence_weighting {
            " — generalist default, low confidence"
        } else {
            ""
        }
    );
    let _ = writeln!(
        s,
        "  recommended : {} (measured — lowest cost)",
        r.recommended
    );
    let _ = writeln!(s, "  auto picks  : {}", r.auto_choice);
    match &r.current {
        Some(c) if *c == r.recommended => {
            let _ = writeln!(s, "  current     : {c} (already the measured best)");
        }
        Some(c) => {
            let _ = writeln!(
                s,
                "  current     : {c} — rebuild with `--blob-ordering measured` to switch to {}",
                r.recommended
            );
        }
        None => {
            let _ = writeln!(s, "  current     : not recorded (pre-2026-07 archive)");
        }
    }
    if let Some(drift) = &r.workload_drift {
        let _ = writeln!(s);
        let _ = writeln!(s, "  ⚠ drift     : {drift}");
    }
    if let Some(caveat) = &r.playback_caveat {
        let _ = writeln!(s);
        let _ = writeln!(s, "  ⚠ playback  : {caveat}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::arrow_tile::{encode_tile, ColumnarLayer, GeometryColumn};
    use stt_core::metadata::Metadata;
    use stt_core::pack::PackWriter;
    use stt_core::tile::TileId;

    #[test]
    fn audit_ranks_and_recommends_over_a_real_archive() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("audit");
        // Deep-time, narrow-space fixture built with Auto (records blobOrdering).
        // Payloads must be real layer frames: `add_tile_full` checks the frame
        // against the writer's format version.
        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 1 << 20).unwrap();
        let bucket = 3_600_000i64;
        for x in 0..3u32 {
            for b in 0..24i64 {
                let t = b * bucket;
                let id = TileId::new(10, 4_000 + x, 5_000, t as u64);
                // Distinct ids per tile so no two blobs dedup to one.
                let payload = encode_tile(&[ColumnarLayer {
                    polygon_parts: None,
                    name: "default".to_string(),
                    feature_ids: vec![u64::from(x) * 1_000 + b as u64],
                    start_times: vec![t],
                    end_times: vec![t + bucket - 1],
                    geometry: GeometryColumn::Point(vec![[x as f64, b as f64]]),
                    vertex_times: None,
                    vertex_values: None,
                    triangles: None,
                    vertex_value_matrix: None,
                    properties: vec![],
                }])
                .unwrap();
                w.add_tile_full(&id, t, t + bucket - 1, Some(t), 1, None, &payload)
                    .unwrap();
            }
        }
        let meta = Metadata::new("audit").with_temporal_bucket_ms(bucket as u64);
        w.finalize(&meta).unwrap();

        let ts = PackedTileset::open(&out).unwrap();
        let r = order_audit(&ts).unwrap();

        // The simulation must actually have run: this fixture declares
        // max_zoom 14 by metadata default while tiling to zoom 10, and before
        // the native-tier fallback the audit selected ZERO entries and reported
        // four all-zero rows that satisfied every ranking assertion vacuously.
        assert!(r.native_tiles > 0, "the audit simulated nothing");
        assert_eq!(r.orderings.len(), 4);
        // Cheapest-first by blended cost (NOT raw request count).
        for pair in r.orderings.windows(2) {
            assert!(pair[0].total_cost <= pair[1].total_cost);
        }
        // Recommended is the cheapest SELECTABLE ordering — never morton3, and
        // no non-morton3 ordering is cheaper than it.
        assert_ne!(r.recommended, "morton3");
        assert!(r.orderings.iter().filter(|o| o.recommended).count() == 1);
        let rec = r.orderings.iter().find(|o| o.recommended).unwrap();
        assert_eq!(rec.ordering, r.recommended);
        for o in &r.orderings {
            if o.ordering != "morton3" {
                assert!(o.total_cost >= rec.total_cost);
            }
        }
        // The archive records its concrete ordering (F4a) and the audit surfaces it.
        let current = r
            .current
            .clone()
            .expect("built-with-Auto archive records blobOrdering");
        assert!(matches!(
            current.as_str(),
            "spatial" | "time-major" | "hilbert3" | "morton3"
        ));
        assert!(matches!(
            r.auto_choice.as_str(),
            "spatial" | "time-major" | "hilbert3" | "morton3"
        ));
        assert!(format_text(&r).contains("recommended :"));

        // The playback columns are populated for EVERY ordering, unconditionally.
        let band_needed: Vec<u64> = r.orderings.iter().map(|o| o.playback_bytes).collect();
        assert!(
            band_needed.iter().all(|b| *b > 0),
            "playback bytes must be reported for every ordering: {band_needed:?}"
        );
        for o in &r.orderings {
            assert!(o.playback_reads > 0, "{} has no playback reads", o.ordering);
            assert!(
                o.playback_worst_advance_cost > 0,
                "{} has no worst-advance cost",
                o.ordering
            );
            // The worst single advance can never exceed the whole query.
            assert!(
                o.playback_worst_advance_cost
                    <= o.playback_bytes + o.playback_reads * r.coalesce_gap_bytes
            );
        }
        // The workload used is reported, and this fixture (built without style
        // hints) is explicitly flagged low-confidence rather than silently
        // pretending to a dataset-specific weighting.
        assert!(r.low_confidence_weighting);
        assert_eq!(r.layer_hint, None);
        assert_eq!(r.workload.scrub, 1);
        assert_eq!(r.workload.pan, 1);
        assert_eq!(
            r.workload.coalesce_gap_bytes,
            stt_core::ordering_sim::DEFAULT_COALESCE_GAP_BYTES
        );
        // Built with Auto, so nothing was simulated and nothing is recorded —
        // and therefore no drift can be claimed.
        assert_eq!(r.recorded_workload, None);
        assert_eq!(r.workload_drift, None);
        assert!(format_text(&r).contains("workload    :"));
        assert!(format_text(&r).contains("worst adv"));

        // The JSON round-trips, including every new column.
        let json = serde_json::to_string(&r).unwrap();
        let back: OrderAuditReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.orderings.len(), r.orderings.len());
        for (a, b) in back.orderings.iter().zip(r.orderings.iter()) {
            assert_eq!(a.ordering, b.ordering);
            assert_eq!(a.playback_reads, b.playback_reads);
            assert_eq!(a.playback_bytes, b.playback_bytes);
            assert_eq!(a.playback_worst_advance_cost, b.playback_worst_advance_cost);
            assert_eq!(a.total_cost, b.total_cost);
        }
        assert_eq!(back.workload, r.workload);
        assert_eq!(back.low_confidence_weighting, r.low_confidence_weighting);
    }

    /// A `measured` build records the workload it was laid out under, the audit
    /// reads it back, and an unchanged model reports no drift.
    #[test]
    fn measured_archive_records_and_round_trips_its_workload() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("workload");
        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 1 << 20)
            .unwrap()
            .with_measured_ordering(true);
        let bucket = 3_600_000i64;
        for x in 0..4u32 {
            for b in 0..12i64 {
                let t = b * bucket;
                let id = TileId::new(10, 4_000 + x, 5_000, t as u64);
                let payload = encode_tile(&[ColumnarLayer {
                    polygon_parts: None,
                    name: "default".to_string(),
                    feature_ids: vec![u64::from(x) * 1_000 + b as u64],
                    start_times: vec![t],
                    end_times: vec![t + bucket - 1],
                    geometry: GeometryColumn::Point(vec![[x as f64, b as f64]]),
                    vertex_times: None,
                    vertex_values: None,
                    triangles: None,
                    vertex_value_matrix: None,
                    properties: vec![],
                }])
                .unwrap();
                w.add_tile_full(&id, t, t + bucket - 1, Some(t), 1, None, &payload)
                    .unwrap();
            }
        }
        let mut meta = Metadata::new("workload").with_temporal_bucket_ms(bucket as u64);
        meta.style_hints = Some(stt_core::metadata::StyleHints {
            version: 1,
            properties: vec![],
            suggested_playback_seconds: None,
            suggested_time_window_ms: None,
            layer_hint: Some("trips".to_string()),
        });
        w.finalize(&meta).unwrap();

        let ts = PackedTileset::open(&out).unwrap();
        let r = order_audit(&ts).unwrap();
        assert_eq!(r.layer_hint.as_deref(), Some("trips"));
        assert!(!r.low_confidence_weighting);
        // trips + 12 buckets → the playback-dominant row.
        assert_eq!(r.workload.playback, 2);
        let recorded = r
            .recorded_workload
            .expect("measured build records workload");
        assert_eq!(recorded, r.workload);
        assert_eq!(
            r.workload_drift, None,
            "unchanged model must not claim drift"
        );
        // The audit reads the CANONICAL top-level key.
        assert_eq!(ts.manifest().ordering_workload, Some(recorded));

        // ...and still reads an archive that carries only the reader-compat
        // mirror inside `metadata` — the shape every archive written before the
        // key moved has on disk. Strip the top-level key and re-open.
        let path = out.join("manifest.json");
        let mut v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        v.as_object_mut().unwrap().remove("orderingWorkload");
        assert!(v["metadata"]["ordering_workload"].is_object());
        std::fs::write(&path, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
        let legacy = PackedTileset::open(&out).unwrap();
        assert_eq!(legacy.manifest().ordering_workload, None);
        let r2 = order_audit(&legacy).unwrap();
        assert_eq!(
            r2.recorded_workload,
            Some(recorded),
            "the metadata mirror must keep an already-published archive auditable"
        );
    }

    /// The §6.2 gap-4 flag: a recorded coalescing gap that no longer matches the
    /// reader-mirroring constant is DRIFT, even though not one archive byte moved.
    #[test]
    fn recorded_workload_drift_is_flagged_including_the_reader_gap() {
        let current = OrderingWorkload {
            scrub: 1,
            pan: 1,
            playback: 2,
            playback_window_buckets: 1,
            runway_multiplier: 4,
            coalesce_gap_bytes: 2 * 1024 * 1024,
        };
        assert_eq!(workload_drift_for(None, &current), None);
        assert_eq!(workload_drift_for(Some(&current), &current), None);

        let old_gap = OrderingWorkload {
            coalesce_gap_bytes: 512 * 1024,
            ..current
        };
        let d = workload_drift_for(Some(&old_gap), &current).expect("gap drift is flagged");
        assert!(d.contains("coalescing gap"), "{d}");
        assert!(d.contains("READER-mirroring"), "{d}");

        let old_weights = OrderingWorkload {
            playback: 0,
            ..current
        };
        let d = workload_drift_for(Some(&old_weights), &current).expect("weight drift is flagged");
        assert!(d.contains("weights"), "{d}");

        let old_runway = OrderingWorkload {
            runway_multiplier: 1,
            ..current
        };
        let d = workload_drift_for(Some(&old_runway), &current).expect("runway drift is flagged");
        assert!(d.contains("runway multiplier"), "{d}");

        let old_window = OrderingWorkload {
            playback_window_buckets: 8,
            ..current
        };
        let d = workload_drift_for(Some(&old_window), &current).expect("window drift is flagged");
        assert!(d.contains("playback window"), "{d}");
    }

    #[test]
    fn playback_caveat_fires_only_for_spatial_over_multiple_buckets() {
        // Spatial + many buckets → warn, and name the safer ordering.
        let c = playback_caveat_for("spatial", 24).expect("spatial+multi-bucket warns");
        assert!(c.contains("time-major"), "{c}");
        assert!(c.contains("24 time buckets"), "{c}");
        // Single bucket → no playback dimension → no caveat.
        assert!(playback_caveat_for("spatial", 1).is_none());
        // A playback-friendly ordering → no caveat even across many buckets.
        assert!(playback_caveat_for("time-major", 24).is_none());
        assert!(playback_caveat_for("hilbert3", 99).is_none());
    }

    /// With the playback family PRICED, the note demotes from a warning about a
    /// blind spot in the model to a note about a measured tradeoff — the gating
    /// condition (spatial, multi-bucket) is unchanged.
    #[test]
    fn playback_caveat_demotes_when_playback_is_priced() {
        let unpriced = playback_caveat_weighted("spatial", 24, 0).unwrap();
        assert!(
            unpriced.contains("does not model playback buffering"),
            "{unpriced}"
        );
        let priced = playback_caveat_weighted("spatial", 24, 2).unwrap();
        assert!(priced.contains("PRICED IN"), "{priced}");
        assert!(priced.contains("measured tradeoff"), "{priced}");
        assert!(
            !priced.contains("does not model playback buffering"),
            "{priced}"
        );
        // The gate itself is unchanged at any weight.
        assert!(playback_caveat_weighted("spatial", 1, 2).is_none());
        assert!(playback_caveat_weighted("time-major", 24, 2).is_none());
    }

    fn empty_report() -> OrderAuditReport {
        OrderAuditReport {
            native_tiles: 10,
            coalesce_gap_bytes: 1 << 20,
            pack_bytes: 1 << 20,
            orderings: vec![],
            recommended: "spatial".to_string(),
            auto_choice: "spatial".to_string(),
            current: None,
            workload: OrderingWorkload {
                scrub: 1,
                pan: 1,
                playback: 1,
                playback_window_buckets: 1,
                runway_multiplier: 4,
                coalesce_gap_bytes: 1 << 20,
            },
            layer_hint: None,
            low_confidence_weighting: true,
            recorded_workload: None,
            workload_drift: None,
            playback_caveat: Some("watch out".to_string()),
        }
    }

    #[test]
    fn format_text_surfaces_the_playback_caveat() {
        let r = empty_report();
        assert!(format_text(&r).contains("playback"));
        assert!(format_text(&r).contains("watch out"));
    }

    #[test]
    fn format_text_surfaces_workload_and_drift() {
        let mut r = empty_report();
        r.workload_drift = Some("gap moved".to_string());
        r.layer_hint = Some("trips".to_string());
        r.low_confidence_weighting = false;
        let text = format_text(&r);
        assert!(
            text.contains("workload    : scrub 1 / pan 1 / playback 1"),
            "{text}"
        );
        assert!(text.contains("layer hint trips"), "{text}");
        assert!(text.contains("drift"), "{text}");
        assert!(text.contains("gap moved"), "{text}");
        // Low-confidence weighting announces itself rather than reading as a
        // dataset-specific pick.
        let mut low = empty_report();
        low.low_confidence_weighting = true;
        assert!(format_text(&low).contains("low confidence"));
    }
}
