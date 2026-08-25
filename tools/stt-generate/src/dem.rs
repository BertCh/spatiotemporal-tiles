//! Terrain elevation sampling for build-time "bake the z into the tiles".
//!
//! Source: the AWS Open Data **Terrarium** terrain tiles
//! (`s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`,
//! public, anonymous, global; Mapzen-heritage SRTM/ETOPO composite). Each
//! 256×256 PNG encodes metres as `(R·256 + G + B/256) − 32768`.
//!
//! Used by `gtfs --bake-elevation` to write a per-vertex elevation into the
//! intermediate parquet's `vertex_values` column — the SAME per-vertex scalar
//! channel the HRRR drift particles use for temperature, so stt-build tiles
//! and clips it with zero new format surface, and the trip-heads renderer
//! interpolates it along the path exactly like the positions.
//!
//! Design notes:
//! - Tiles are fetched lazily (only where vertices actually are), cached on
//!   disk under `data/dem/terrarium/{z}/{x}/{y}.png`, and decoded once into an
//!   in-memory `f32` grid — after the first run a rebuild does no network at
//!   all. A failed fetch is remembered as `None` so one dead tile can't stall
//!   the expansion with per-vertex retries.
//! - Sampling is bilinear in GLOBAL pixel space, so it is seamless across
//!   tile borders (the four taps may straddle up to four tiles).
//! - This is deliberately a *blocking, sequential* client: samples arrive one
//!   vertex at a time from the feed expansion, an in-memory tile hit is ~ns,
//!   and the unique-tile working set for a national network is a few thousand
//!   fetches on the first run.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

const TILE_SIZE: u32 = 256;

/// Decoded elevations for one tile, row-major `TILE_SIZE²` metres.
type TileGrid = Box<[f32]>;

pub struct TerrariumDem {
    zoom: u8,
    cache_dir: PathBuf,
    client: reqwest::blocking::Client,
    /// `(x, y)` → decoded grid, or `None` when the fetch/decode failed.
    tiles: HashMap<(u32, u32), Option<TileGrid>>,
    pub fetched: usize,
    pub failed: usize,
}

impl TerrariumDem {
    pub fn new(zoom: u8, cache_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&cache_dir)
            .with_context(|| format!("creating DEM cache dir {}", cache_dir.display()))?;
        Ok(Self {
            zoom,
            cache_dir,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()?,
            tiles: HashMap::new(),
            fetched: 0,
            failed: 0,
        })
    }

    /// Bilinear elevation in metres at (lon, lat), or `None` where every
    /// contributing tile is unavailable.
    pub fn sample(&mut self, lon: f64, lat: f64) -> Option<f64> {
        let n = (1u32 << self.zoom) as f64;
        let lat = lat.clamp(-85.05112878, 85.05112878);
        let latr = lat.to_radians();
        // Global pixel coordinates at this zoom, centred taps (−0.5 so that a
        // sample exactly on a pixel centre reads that pixel with weight 1).
        let gx = ((lon + 180.0) / 360.0) * n * TILE_SIZE as f64 - 0.5;
        let gy = ((1.0 - (latr.tan() + 1.0 / latr.cos()).ln() / std::f64::consts::PI) / 2.0)
            * n
            * TILE_SIZE as f64
            - 0.5;
        let x0 = gx.floor();
        let y0 = gy.floor();
        let fx = gx - x0;
        let fy = gy - y0;

        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for (dx, dy, w) in [
            (0.0, 0.0, (1.0 - fx) * (1.0 - fy)),
            (1.0, 0.0, fx * (1.0 - fy)),
            (0.0, 1.0, (1.0 - fx) * fy),
            (1.0, 1.0, fx * fy),
        ] {
            if w <= 0.0 {
                continue;
            }
            if let Some(v) = self.pixel(x0 + dx, y0 + dy) {
                acc += v as f64 * w;
                wsum += w;
            }
        }
        if wsum > 0.0 {
            Some(acc / wsum)
        } else {
            None
        }
    }

    /// One elevation pixel by GLOBAL pixel coordinate (wraps in x, clamps in y).
    fn pixel(&mut self, gx: f64, gy: f64) -> Option<f32> {
        let world_px = ((1u32 << self.zoom) as i64) * TILE_SIZE as i64;
        let px = (gx as i64).rem_euclid(world_px);
        let py = (gy as i64).clamp(0, world_px - 1);
        let tx = (px / TILE_SIZE as i64) as u32;
        let ty = (py / TILE_SIZE as i64) as u32;
        let ix = (px % TILE_SIZE as i64) as usize;
        let iy = (py % TILE_SIZE as i64) as usize;
        self.tile(tx, ty)
            .as_ref()
            .map(|g| g[iy * TILE_SIZE as usize + ix])
    }

    fn tile(&mut self, x: u32, y: u32) -> &Option<TileGrid> {
        if !self.tiles.contains_key(&(x, y)) {
            let grid = self.load_tile(x, y);
            if grid.is_none() {
                self.failed += 1;
            }
            self.tiles.insert((x, y), grid);
        }
        &self.tiles[&(x, y)]
    }

    fn load_tile(&mut self, x: u32, y: u32) -> Option<TileGrid> {
        let path = self
            .cache_dir
            .join(self.zoom.to_string())
            .join(x.to_string())
            .join(format!("{y}.png"));
        let bytes = if path.exists() {
            fs::read(&path).ok()?
        } else {
            let url = format!(
                "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{}/{}/{}.png",
                self.zoom, x, y
            );
            // One retry for transient network hiccups; a hard failure is
            // remembered by the caller so it is not re-attempted per vertex.
            let fetched = self.fetch(&url).or_else(|_| self.fetch(&url)).ok()?;
            self.fetched += 1;
            if self.fetched % 250 == 0 {
                println!("   … DEM tiles fetched: {}", self.fetched);
            }
            let _ = fs::create_dir_all(path.parent().unwrap());
            let _ = fs::write(&path, &fetched);
            fetched
        };
        decode_terrarium_png(&bytes)
    }

    fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        let resp = self.client.get(url).send()?.error_for_status()?;
        let mut buf = Vec::new();
        let mut reader = resp;
        reader.read_to_end(&mut buf)?;
        Ok(buf)
    }
}

/// Decode a terrarium PNG into a row-major metre grid.
fn decode_terrarium_png(bytes: &[u8]) -> Option<TileGrid> {
    let decoder = png::Decoder::new(bytes);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.width != TILE_SIZE || info.height != TILE_SIZE || info.bit_depth != png::BitDepth::Eight
    {
        return None;
    }
    let stride = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => return None,
    };
    Some(decode_terrarium_samples(&buf, stride))
}

/// The terrarium formula over raw 8-bit samples: `(R·256 + G + B/256) − 32768`.
/// Length-generic (pixel count = `buf.len() / stride`); the PNG caller has
/// already asserted 256×256, and the tests feed it tiny synthetic buffers.
fn decode_terrarium_samples(buf: &[u8], stride: usize) -> TileGrid {
    let px = buf.len() / stride;
    let mut out = vec![0f32; px].into_boxed_slice();
    for (i, cell) in out.iter_mut().enumerate() {
        let o = i * stride;
        let (r, g, b) = (buf[o] as f32, buf[o + 1] as f32, buf[o + 2] as f32);
        *cell = r * 256.0 + g + b / 256.0 - 32768.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrarium_formula_decodes_known_values() {
        // Sea level (0 m) encodes as (128, 0, 0); 1000 m as (131, 232, 0):
        // 131·256 + 232 = 33768 → −32768 = 1000. Quarter-metres ride in B.
        let buf = [128u8, 0, 0, 131, 232, 0, 128, 0, 64];
        let g = decode_terrarium_samples(&buf[..], 3);
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 1000.0);
        assert_eq!(g[2], 0.25);
    }

    #[test]
    fn rgba_stride_skips_alpha() {
        let buf = [128u8, 0, 0, 255, 131, 232, 0, 255];
        let g = decode_terrarium_samples(&buf[..], 4);
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 1000.0);
    }
}
