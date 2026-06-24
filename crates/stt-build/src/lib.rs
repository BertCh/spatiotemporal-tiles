//! Library facade for the stt-build crate.
//!
//! The CLI binary lives at `src/main.rs`; this lib target exposes the
//! same modules so integration tests under `tests/` and external probes
//! can drive the pipeline programmatically.

pub mod clip;
pub mod columnar;
/// DuckDB input source — read features from a DuckDB database (or anything
/// DuckDB can scan: Parquet/CSV/… via `read_*`) instead of a GeoParquet file.
/// Gated behind the `duckdb` cargo feature so default builds don't pull the
/// (statically bundled) database engine.
#[cfg(feature = "duckdb")]
pub mod duckdb_input;
pub mod input;
/// PostGIS input source — read features from a live PostgreSQL/PostGIS query
/// instead of a GeoParquet file. Gated behind the `postgres` cargo feature so
/// default builds don't pull the database driver.
#[cfg(feature = "postgres")]
pub mod postgres_input;
pub mod quadbin;
pub mod simplify;
pub mod summary;
pub mod tiler;

/// Encode a single tile's features into an STT tile blob (Arrow IPC + GeoArrow,
/// per-blob zstd) for an arbitrary `(z, x, y, time-bucket)` — without running
/// the full whole-dataset build (no pack/directory writer, no cross-tile
/// state). This is the reusable core a dynamic per-request tile server
/// (`stt-serve`) calls; see [`tiler::encode_single_tile`].
pub use tiler::encode_single_tile;
