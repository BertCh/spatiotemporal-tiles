//! # SpatioTemporal Tiles (STT)
//!
//! Umbrella crate for the STT format toolchain. The implementation lives in
//! the `stt-*` crates; this crate re-exports them under one name so a single
//! `cargo add spatiotemporal-tiles` is the whole story:
//!
//! ```ignore
//! use spatiotemporal_tiles as stt;
//!
//! let reader = stt::core::PackedReader::open("dataset/")?;
//! ```
//!
//! ## Features
//!
//! | feature | adds |
//! |---|---|
//! | *(default)* | `core` — the archive/tile format reader + writer library |
//! | `build` | `build` — the tiler/encoder library behind the `stt-build` CLI |
//! | `optimize` | `optimize` — dataset analysis + encoding recommendations |
//! | `postgres` / `duckdb` | database input sources for `build` |
//! | `cli` | the four binaries: `stt-build`, `stt-optimize`, `stt-validate`, `stt-serve` |
//!
//! Install the CLI tools with `cargo install spatiotemporal-tiles --features cli`
//! (or grab a prebuilt binary from the GitHub releases page).

pub use stt_core as core;

#[cfg(feature = "build")]
pub use stt_build as build;

#[cfg(feature = "optimize")]
pub use stt_optimize as optimize;
