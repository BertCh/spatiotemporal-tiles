//! Library facade for the stt-build crate.
//!
//! The CLI binary lives at `src/main.rs`; this lib target exposes the
//! same modules so integration tests under `tests/` and external probes
//! can drive the pipeline programmatically.

pub mod clip;
pub mod columnar;
pub mod input;
pub mod quadbin;
pub mod simplify;
pub mod summary;
pub mod tiler;
