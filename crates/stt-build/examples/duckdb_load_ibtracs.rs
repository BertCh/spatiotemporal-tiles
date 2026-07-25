//! Load `data/ibtracs.csv` (NOAA IBTrACS storm best-track) into a DuckDB file
//! used by the DuckDB-integration benchmark, AND export a matching baseline
//! GeoParquet — the two inputs for the file-vs-DuckDB ingest A/B. Uses the
//! statically-bundled DuckDB engine (no external `duckdb` CLI required).
//!
//! IBTrACS CSV has TWO header rows (column names, then units) and 163 columns;
//! `read_csv_auto(all_varchar=true)` ingests it as text and `TRY_CAST` drops
//! the units row + any unparseable coordinates.
//!
//! Usage:
//!   cargo run --release -p stt-build --features duckdb \
//!       --example duckdb_load_ibtracs -- [data/ibtracs.csv] [scratch-duckdb]
//!
//! Produces:
//!   <outdir>/hurricane.duckdb        — table `hurricane_obs` (geom GEOMETRY/4326)
//!   <outdir>/hurricane_points.parquet — post-1970 lon/lat baseline (file A/B)

#[cfg(feature = "duckdb")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;

    let args: Vec<String> = std::env::args().collect();
    let csv = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("data/ibtracs.csv");
    let outdir = args.get(2).map(String::as_str).unwrap_or("scratch-duckdb");
    std::fs::create_dir_all(outdir)?;
    let db_path = format!("{outdir}/hurricane.duckdb");
    let parquet_path = format!("{outdir}/hurricane_points.parquet");
    // Start fresh so re-runs are deterministic.
    let _ = std::fs::remove_file(&db_path);

    let conn = duckdb::Connection::open(&db_path).context("open duckdb file")?;
    conn.execute_batch("INSTALL spatial; LOAD spatial; SET TimeZone='UTC';")
        .context("load spatial extension (needs a one-time network INSTALL)")?;

    // Project the handful of columns we need into a typed table with a 4326
    // Point geometry. TRY_CAST everywhere so the units header row (e.g.
    // LAT='degrees_north') and any malformed rows become NULL and are filtered.
    let build = format!(
        "CREATE OR REPLACE TABLE hurricane_obs AS \
         WITH raw AS (SELECT * FROM read_csv_auto('{csv}', header=true, all_varchar=true)) \
         SELECT \
            nullif(trim(SID), '')          AS sid, \
            TRY_CAST(SEASON AS INTEGER)    AS season, \
            nullif(trim(BASIN), '')        AS basin, \
            nullif(trim(NAME), '')         AS name, \
            nullif(trim(NATURE), '')       AS nature, \
            TRY_CAST(ISO_TIME AS TIMESTAMP) AS iso_time, \
            TRY_CAST(LAT AS DOUBLE)        AS lat, \
            TRY_CAST(LON AS DOUBLE)        AS lon, \
            TRY_CAST(WMO_WIND AS DOUBLE)   AS wmo_wind, \
            TRY_CAST(WMO_PRES AS DOUBLE)   AS wmo_pres, \
            TRY_CAST(USA_SSHS AS INTEGER)  AS usa_sshs, \
            ST_Point(TRY_CAST(LON AS DOUBLE), TRY_CAST(LAT AS DOUBLE)) AS geom \
         FROM raw \
         WHERE TRY_CAST(LAT AS DOUBLE) IS NOT NULL \
           AND TRY_CAST(LON AS DOUBLE) IS NOT NULL \
           AND TRY_CAST(ISO_TIME AS TIMESTAMP) IS NOT NULL;"
    );
    conn.execute_batch(&build).context("build hurricane_obs")?;

    // Summary.
    let (cnt, tmin, tmax): (i64, String, String) = conn.query_row(
        "SELECT COUNT(*), MIN(iso_time)::VARCHAR, MAX(iso_time)::VARCHAR FROM hurricane_obs",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    println!("hurricane_obs: {cnt} rows, {tmin} .. {tmax}");

    // Baseline GeoParquet: the post-1970 subset as plain lon/lat columns (no
    // geometry) — exactly the point set the `--where iso_time >= '1970-01-01'`
    // DuckDB ingest produces, so the file vs DuckDB A/B compares identical data.
    let copy = format!(
        "COPY ( \
            SELECT lon, lat, iso_time, sid, season, basin, name, nature, wmo_wind, wmo_pres, usa_sshs \
            FROM hurricane_obs WHERE iso_time >= TIMESTAMP '1970-01-01' \
         ) TO '{parquet_path}' (FORMAT parquet);"
    );
    conn.execute_batch(&copy)
        .context("export baseline parquet")?;
    let pq: i64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM read_parquet('{parquet_path}')"),
        [],
        |r| r.get(0),
    )?;

    println!("wrote {db_path}");
    println!("wrote {parquet_path} ({pq} post-1970 rows)");
    Ok(())
}

#[cfg(not(feature = "duckdb"))]
fn main() {
    eprintln!("rebuild with `--features duckdb` to run this example");
}
