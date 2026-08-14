//! Shared OSRM routing client.
//!
//! A thin blocking wrapper over a locally-running [OSRM](https://project-osrm.org/)
//! `osrm-routed` server (the `route/v1` service). Extracted from
//! [`super::nyc_rideshare`] so any dataset that needs to route OD pairs onto a
//! real street network can reuse it — the NYC taxi `--flows` overview and the
//! Montréal BIXI `--streets` overview both do.
//!
//! The `/route/v1/driving/` path segment is just OSRM's service label; the loaded
//! routing **profile** (car, bicycle, …) is whatever the server was built and
//! customized with, so the same client drives a bicycle-profile server unchanged.
//! Requests ask for `annotations=duration,distance` so callers can recover real
//! per-edge timing that reflects road class / speed.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct OsrmRouteResponse {
    code: String,
    routes: Option<Vec<OsrmRoute>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OsrmRoute {
    pub(crate) geometry: OsrmGeometry,
    /// Present when the request includes `annotations=duration,distance`.
    /// Contains per-leg (segment) durations/distances along the route.
    /// `legs[i].annotation.duration[j]` is the OSRM-estimated seconds to
    /// traverse the j-th edge in leg i — reflects road class / speed limit,
    /// so highway runs are short and urban-grid blocks are long.
    #[serde(default)]
    pub(crate) legs: Vec<OsrmLeg>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OsrmGeometry {
    pub(crate) coordinates: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct OsrmLeg {
    #[serde(default)]
    pub(crate) annotation: Option<OsrmAnnotation>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct OsrmAnnotation {
    #[serde(default)]
    pub(crate) duration: Vec<f64>,
}

/// Probe the OSRM server with a trivial route and confirm it answers `Ok`.
/// Fails fast with a friendly message if the server isn't reachable, so a
/// generator doesn't grind through thousands of pairs against a dead endpoint.
pub(crate) fn check_osrm_connectivity(osrm_url: &str) -> Result<()> {
    println!("🔍 Checking OSRM connectivity...");

    // A short cross-town leg in Montréal — valid on any North-American extract;
    // we only assert the server answers `Ok`, not that this exact pair routes.
    let test_url = format!(
        "{}/route/v1/driving/-73.59,45.50;-73.58,45.51?overview=full&geometries=geojson",
        osrm_url
    );

    let response = reqwest::blocking::get(&test_url)
        .context("Failed to connect to OSRM server. Is it running?")?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "OSRM server returned error status: {}",
            response.status()
        ));
    }

    let body: serde_json::Value = response.json()?;
    let code = body.get("code").and_then(|c| c.as_str()).unwrap_or("");
    if code != "Ok" {
        return Err(anyhow!("OSRM returned code: {}", code));
    }

    println!("✓ OSRM server is ready");
    Ok(())
}

/// Route a single OD pair, returning the best route (or `None` if OSRM can't
/// connect the points). One-shot client — fine for low call counts; use
/// [`get_osrm_route_pooled`] when routing many pairs in parallel.
pub(crate) fn get_osrm_route(
    osrm_url: &str,
    from_lon: f64,
    from_lat: f64,
    to_lon: f64,
    to_lat: f64,
) -> Result<Option<OsrmRoute>> {
    // `annotations=duration,distance` returns per-leg per-edge durations so
    // we can compute real per-vertex timestamps that reflect street class
    // (highway vs. urban grid). Without this OSRM still returns the full
    // polyline but with no per-edge timing, forcing us to distribute the
    // trip duration uniformly by distance → "flash" artifacts on long
    // segments in the animated-trips renderer.
    let url = format!(
        "{}/route/v1/driving/{:.6},{:.6};{:.6},{:.6}?overview=full&geometries=geojson&annotations=duration,distance",
        osrm_url, from_lon, from_lat, to_lon, to_lat
    );

    let response = reqwest::blocking::get(&url)?;
    let body: OsrmRouteResponse = response.json()?;

    if body.code != "Ok" {
        return Ok(None);
    }

    Ok(body.routes.and_then(|mut r| r.pop()))
}

/// Like [`get_osrm_route`] but reuses a thread-local pooled HTTP client, so
/// routing thousands of pairs across a rayon pool keeps connections warm
/// instead of opening a fresh socket per request.
pub(crate) fn get_osrm_route_pooled(
    osrm_url: &str,
    from_lon: f64,
    from_lat: f64,
    to_lon: f64,
    to_lat: f64,
) -> Result<Option<OsrmRoute>> {
    use std::cell::RefCell;

    thread_local! {
        static CLIENT: RefCell<reqwest::blocking::Client> = RefCell::new(
            reqwest::blocking::Client::builder()
                .pool_max_idle_per_host(10)
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap()
        );
    }

    let url = format!(
        "{}/route/v1/driving/{:.6},{:.6};{:.6},{:.6}?overview=full&geometries=geojson&annotations=duration,distance",
        osrm_url, from_lon, from_lat, to_lon, to_lat
    );

    CLIENT.with(|client| {
        let response = client.borrow().get(&url).send()?;
        let body: OsrmRouteResponse = response.json()?;

        if body.code != "Ok" {
            return Ok(None);
        }

        Ok(body.routes.and_then(|mut r| r.pop()))
    })
}

/// Flatten an OSRM route into `[lon, lat]` vertices plus, when the response was
/// annotated (`annotations=duration`), the per-edge durations in seconds.
///
/// The durations come back as `Some` only when they're self-consistent — one
/// per edge (`coords.len() - 1`) and summing to a positive total — so callers
/// can treat `Some` as "usable per-segment timing" without re-checking. The
/// coords are returned even when the timing isn't, so a caller can still bake
/// geometry and let stt-build fall back to uniform-by-distance interpolation.
pub(crate) fn route_geometry(route: &OsrmRoute) -> (Vec<[f64; 2]>, Option<Vec<f64>>) {
    let coords: Vec<[f64; 2]> = route
        .geometry
        .coordinates
        .iter()
        .map(|c| [c[0], c[1]])
        .collect();
    // For a single source→destination request there is one leg whose
    // annotation.duration has `coords.len() - 1` entries; concatenate across
    // legs defensively so a multi-leg response (unused today) still works.
    let mut durations: Vec<f64> = Vec::with_capacity(coords.len().saturating_sub(1));
    for leg in &route.legs {
        if let Some(ann) = &leg.annotation {
            durations.extend(ann.duration.iter().copied());
        }
    }
    let edge = (coords.len() >= 2
        && durations.len() == coords.len() - 1
        && durations.iter().sum::<f64>() > 0.0)
        .then_some(durations);
    (coords, edge)
}

/// Turn per-edge OSRM durations into per-vertex absolute Unix-ms timestamps,
/// rescaled to land exactly on `[start_ms, end_ms]`.
///
/// OSRM's predicted total usually differs from a trip's recorded duration by
/// ±10-30%, but the *shape* of the per-edge distribution (which edges are fast,
/// which slow) is what drives the per-segment-speed effect, so we keep the shape
/// and stretch it onto the real window. The last vertex is pinned to `end_ms`
/// so cumulative rounding never drifts the trip's end off its recorded value.
///
/// Returns `None` (caller falls back to uniform-by-distance interpolation)
/// unless there is exactly one duration per edge, they sum positive, and the
/// window is non-empty.
pub(crate) fn vertex_timestamps_from_durations(
    n_coords: usize,
    edge_durations: &[f64],
    start_ms: i64,
    end_ms: i64,
) -> Option<Vec<i64>> {
    if n_coords < 2 || end_ms <= start_ms || edge_durations.len() != n_coords - 1 {
        return None;
    }
    let total: f64 = edge_durations.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let window_ms = (end_ms - start_ms) as f64;
    let mut acc = 0.0f64;
    let mut out: Vec<i64> = Vec::with_capacity(n_coords);
    out.push(start_ms);
    for &d in edge_durations {
        acc += d;
        out.push(start_ms + (acc / total * window_ms) as i64);
    }
    if let Some(last) = out.last_mut() {
        *last = end_ms;
    }
    Some(out)
}

/// Convenience: per-vertex absolute timestamps for a whole route across
/// `[start_ms, end_ms]`. `None` when the route carries no usable per-edge
/// annotations. Reads the vertex count and per-edge durations directly (no
/// throwaway coord allocation) — this runs per trip under a rayon pool, and
/// callers that also need the geometry build it once themselves. The
/// consistency checks live in [`vertex_timestamps_from_durations`], so this
/// stays identical to gating on [`route_geometry`]'s `Some` edge.
pub(crate) fn vertex_timestamps_from_route(
    route: &OsrmRoute,
    start_ms: i64,
    end_ms: i64,
) -> Option<Vec<i64>> {
    let n_coords = route.geometry.coordinates.len();
    let mut durations: Vec<f64> = Vec::with_capacity(n_coords.saturating_sub(1));
    for leg in &route.legs {
        if let Some(ann) = &leg.annotation {
            durations.extend(ann.duration.iter().copied());
        }
    }
    vertex_timestamps_from_durations(n_coords, &durations, start_ms, end_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an `OsrmRoute` from `[lon,lat]` coords and optional per-edge
    /// durations (one leg; `None` = no annotation, as OSRM returns without
    /// `annotations=duration`).
    fn mk_route(coords: &[[f64; 2]], durations: Option<Vec<f64>>) -> OsrmRoute {
        OsrmRoute {
            geometry: OsrmGeometry {
                coordinates: coords.iter().map(|c| vec![c[0], c[1]]).collect(),
            },
            legs: vec![OsrmLeg {
                annotation: durations.map(|duration| OsrmAnnotation { duration }),
            }],
        }
    }

    #[test]
    fn route_geometry_gates_on_consistent_annotations() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        // Consistent: one duration per edge, positive sum → Some.
        let (c, edge) = route_geometry(&mk_route(&coords, Some(vec![10.0, 30.0])));
        assert_eq!(c, coords.to_vec());
        assert_eq!(edge, Some(vec![10.0, 30.0]));
        // Coords still returned when timing is unusable, so callers can bake
        // geometry and let stt-build interpolate.
        assert!(route_geometry(&mk_route(&coords, None)).1.is_none()); // no annotations
        assert!(route_geometry(&mk_route(&coords, Some(vec![10.0])))
            .1
            .is_none()); // wrong count
        assert!(route_geometry(&mk_route(&coords, Some(vec![0.0, 0.0])))
            .1
            .is_none()); // zero total
    }

    #[test]
    fn route_wrapper_rescales_end_to_end() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        // Same rescale as the durations path: 0%, 75%, 100% of a 40s window.
        let vt =
            vertex_timestamps_from_route(&mk_route(&coords, Some(vec![30.0, 10.0])), 1_000, 41_000);
        assert_eq!(vt, Some(vec![1_000, 31_000, 41_000]));
        // No annotations → None so the caller falls back to uniform-by-distance.
        assert!(vertex_timestamps_from_route(&mk_route(&coords, None), 0, 1_000).is_none());
    }

    #[test]
    fn rescales_durations_onto_window() {
        // 3 vertices / 2 edges; leg0 is 3× leg1 (slow then fast). Rescaled onto
        // a 40s window starting at t=1000: 0%, 75%, 100% → 1000, 31000, 41000.
        let vt = vertex_timestamps_from_durations(3, &[30.0, 10.0], 1_000, 41_000).unwrap();
        assert_eq!(vt, vec![1_000, 31_000, 41_000]);
    }

    #[test]
    fn endpoints_are_exact() {
        // Durations that don't divide the window evenly still start/end exactly.
        let vt = vertex_timestamps_from_durations(4, &[1.0, 1.0, 1.0], 0, 10_000).unwrap();
        assert_eq!(vt.len(), 4);
        assert_eq!(*vt.first().unwrap(), 0);
        assert_eq!(*vt.last().unwrap(), 10_000);
        // Monotonic non-decreasing.
        assert!(vt.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(vertex_timestamps_from_durations(3, &[1.0], 0, 10).is_none()); // wrong edge count
        assert!(vertex_timestamps_from_durations(3, &[1.0, 1.0], 10, 0).is_none()); // reversed window
        assert!(vertex_timestamps_from_durations(1, &[], 0, 10).is_none()); // < 2 coords
        assert!(vertex_timestamps_from_durations(2, &[0.0], 0, 10).is_none()); // zero total
    }
}
