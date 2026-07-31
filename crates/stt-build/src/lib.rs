//! Library facade for the stt-build crate.
//!
//! The CLI binary lives at `src/main.rs`; this lib target exposes the
//! same modules so integration tests under `tests/` and external probes
//! can drive the pipeline programmatically.

/// Shared flag → config parsing (`--temporal-bucket`, `--vector-group`,
/// `--quantize-*`, per-tile budgets, attribute filters) used by both the
/// `stt-build` CLI and the `stt-serve` dynamic server, so a live-served tile is
/// configured byte-identically to one built offline.
pub mod build_options;
pub mod clip;
pub mod columnar;
/// Decode logic shared by the PostGIS and DuckDB input adaptors (row outcome,
/// per-vertex coercion tally, dropped-column warning, integer `--time-format`
/// scaling, NaN→JSON-null) — one definition so the readers can't drift.
#[cfg(any(feature = "postgres", feature = "duckdb"))]
pub(crate) mod db_input_common;
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
pub mod props;
pub mod quadbin;
pub mod simplify;
/// STAC Item emission (`stt-build --stac`): the discovery-layer sidecar,
/// derived mechanically from a finished manifest (packed-format spec §10.3).
pub mod stac;
/// Build-time `style_hints` collection (`--style-hints`): bounded per-property
/// value sampling + layer-kind counting over the loaded features, feeding the
/// generic profiler in `stt_optimize::analysis::properties`.
pub mod style_hints;
pub mod summary;
pub mod tiler;

/// Encode a single tile's features into an **uncompressed** STT layer-frame
/// tile payload (Arrow IPC + GeoArrow geometry) for an arbitrary
/// `(z, x, y, time-bucket)` — without running the full whole-dataset build (no
/// pack/directory writer, no cross-tile state). These bytes are the frame the
/// offline build feeds INTO the pack writer *before* per-blob zstd, so a
/// dynamic per-request tile server (`stt-serve`) can serve them directly. See
/// [`tiler::encode_single_tile`].
pub use tiler::{encode_single_tile, encode_single_tile_counted};
