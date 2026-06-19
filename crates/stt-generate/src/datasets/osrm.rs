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
    #[allow(dead_code)]
    #[serde(default)]
    pub(crate) distance: Vec<f64>,
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
        return Err(anyhow!("OSRM server returned error status: {}", response.status()));
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
