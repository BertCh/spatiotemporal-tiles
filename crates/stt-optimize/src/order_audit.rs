//! Blob-ordering audit over a built archive: measure per-ordering range-read
//! cost and recommend `--blob-ordering`.
//!
//! Where the layout advisor's ordering hint is an unsimulated access-shape
//! guess (`advisors::layout`), this runs the shared simulator
//! ([`stt_core::ordering_sim`]) over the archive's real native tiles and ranks
//! the four orderings by the range reads the two canonical access patterns
//! (scrub a viewport across time; pan one instant across space) would cost. It
//! reads only the directory — no payload decode — so it is cheap on any size.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use stt_core::curve::{self, BlobOrdering};
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
    pub total_reads: u64,
    pub total_bytes: u64,
    /// Blended cost the ranking uses: `total_bytes + total_reads * gap`.
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
    /// Every ordering, cheapest-first.
    pub orderings: Vec<OrderingCostRow>,
    /// The measured-best ordering.
    pub recommended: String,
    /// What `--blob-ordering auto` (the cardinality heuristic) would pick.
    pub auto_choice: String,
    /// The ordering the archive was built with (`manifest.blobOrdering`);
    /// `None` for pre-2026-07 archives that don't record it.
    pub current: Option<String>,
}

/// Audit an opened archive's blob ordering.
pub fn order_audit(tileset: &PackedTileset) -> Result<OrderAuditReport> {
    let (_min_z, max_z) = tileset.zoom_range();
    let bucket_ms = (tileset.temporal_bucket_ms().max(1)) as i64;

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

    let opts = SimOptions::default();
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

    Ok(OrderAuditReport {
        native_tiles: samples.len(),
        coalesce_gap_bytes: opts.coalesce_gap_bytes,
        orderings,
        recommended: recommended.as_str().to_string(),
        auto_choice: auto_choice.as_str().to_string(),
        current: tileset.manifest().blob_ordering.clone(),
    })
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
        "Blob-ordering audit — {} native tiles, {} coalescing gap",
        r.native_tiles,
        mib(r.coalesce_gap_bytes)
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "   {:<11} {:>10} {:>12} {:>12}",
        "ordering", "range rds", "bytes read", "cost"
    );
    for row in &r.orderings {
        let mark = if row.recommended { " * " } else { "   " };
        let note = if row.ordering == "morton3" { "  (research only)" } else { "" };
        let _ = writeln!(
            s,
            "{}{:<11} {:>10} {:>12} {:>12}{}",
            mark,
            row.ordering,
            row.total_reads,
            mib(row.total_bytes),
            mib(row.total_cost),
            note
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "  cost = bytes read + reads × {} gap (the reader's own request/byte trade)",
        mib(r.coalesce_gap_bytes)
    );
    let _ = writeln!(s, "  recommended : {} (measured — lowest cost)", r.recommended);
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
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::metadata::Metadata;
    use stt_core::pack::PackWriter;
    use stt_core::tile::TileId;

    #[test]
    fn audit_ranks_and_recommends_over_a_real_archive() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("audit");
        // Deep-time, narrow-space fixture built with Auto (records blobOrdering).
        // v1-pinned: payloads are opaque bytes, not v2 frames (add_tile_full
        // enforces frame/manifest version coherence).
        let mut w = PackWriter::create(&out, BlobOrdering::Auto, 1 << 20)
            .unwrap()
            .with_format_version(stt_core::pack::PACKED_FORMAT_VERSION_V1);
        let bucket = 3_600_000i64;
        for x in 0..3u32 {
            for b in 0..24i64 {
                let t = b * bucket;
                let id = TileId::new(10, 4_000 + x, 5_000, t as u64);
                let payload = format!("t-{x}-{b}").into_bytes();
                w.add_tile_full(&id, t, t + bucket - 1, Some(t), 1, None, &payload).unwrap();
            }
        }
        let meta = Metadata::new("audit").with_temporal_bucket_ms(bucket as u64);
        w.finalize(&meta).unwrap();

        let ts = PackedTileset::open(&out).unwrap();
        let r = order_audit(&ts).unwrap();

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
        let current = r.current.clone().expect("built-with-Auto archive records blobOrdering");
        assert!(matches!(current.as_str(), "spatial" | "time-major" | "hilbert3" | "morton3"));
        assert!(matches!(r.auto_choice.as_str(), "spatial" | "time-major" | "hilbert3" | "morton3"));
        assert!(format_text(&r).contains("recommended :"));
    }
}
