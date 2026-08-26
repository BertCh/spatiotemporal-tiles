# ECCO "Perpetual Ocean" — advected-particle currents demo

This generator turns NASA's [ECCO V4r4](https://ecco-group.org/) ocean-model
surface velocity into animated current ribbons for the STT showcase. Massless
particles are seeded across the ocean and integrated through the time-varying
`EVEL`/`NVEL` field (RK4); each particle's path becomes a LineString carrying
per-vertex timestamps and a per-vertex **current speed** that colors the ribbon
— the same tile shape as the `drifters` demo, so it pairs with it directly
(modeled currents vs. observed buoys).

```
ECCO NetCDF  →  ecco_advect.py  →  GeoParquet  →  stt-build  →  ecco-currents.stt
```

## 1. Get the data (free Earthdata login)

We use the **0.5° interpolated** ocean-velocity product, which provides
`EVEL` (eastward) and `NVEL` (northward) already rotated to geographic axes —
no curvilinear-grid math. We only read the **surface** level (`Z` index 0,
≈ −5 m) and the two horizontal components; everything else is ignored.

Make a free login at <https://urs.earthdata.nasa.gov/>, then:

```bash
pip install podaac-data-downloader

# Monthly — 12 files, ~tens of MB, smooth gyre ribbons (start here):
podaac-data-downloader \
  -c ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4 \
  -d ./ecco-vel -sd 2017-01-01T00:00:00Z -ed 2017-12-31T23:59:59Z

# Daily — 365 files, larger, resolves eddies/mesoscale (final version):
#   swap MONTHLY → DAILY above.
```

Coverage is 1992-01-01 → 2017-12-31. Any contiguous span of ≥2 granules works;
a full year gives a satisfying loop.

## 2. Advect → build

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements-ecco.txt

# Full pipeline (needs stt-build on PATH, or pass --stt-build <path>):
./venv/bin/python ecco_advect.py \
  --input ./ecco-vel \
  --output ../../data-fleet/ecco-currents.stt

# Or stop at GeoParquet and build separately:
./venv/bin/python ecco_advect.py --input ./ecco-vel \
  --output ecco-currents.parquet --skip-build
```

Useful knobs (`--help` for all):

| flag                | default | meaning                                             |
| ------------------- | ------- | --------------------------------------------------- |
| `--particles`       | 20000   | number of seed particles                            |
| `--step-days`       | 5       | vertex cadence (one point every N days)             |
| `--substeps`        | 5       | RK4 integration substeps per output step (accuracy) |
| `--lat-limit`       | 80      | don't seed poleward of ±this latitude               |
| `--seed`            | 42      | RNG seed — reproducible particle clouds             |
| `--max-zoom`        | 5       | stt-build max zoom                                  |
| `--temporal-bucket` | 5d      | stt-build time bucket (match `--step-days`)         |

Particles that beach (hit a land/ice NaN cell) end their track; tracks are
split at the antimeridian so nothing draws across the dateline.

## 3. Verify + wire into the showcase

```bash
stt-validate ../../data-fleet/ecco-currents.stt
```

The showcase entry lives in `examples/showcase/src/datasets.ts` under id
`ecco-currents` (`type: 'trips'`, globe view, `tripGradient` on
`vertexValues` = speed). The `.stt` is kept local / synced to R2, not committed
— same as the other large archives.

## Output schema (stt-build LineString contract)

| column              | type            | notes                                          |
| ------------------- | --------------- | ---------------------------------------------- |
| `geometry`          | WKB `Binary`    | LineString (lon, lat)                          |
| `timestamp`         | `Int64`         | track start, Unix-ms (`--time-format unix-ms`) |
| `end_timestamp`     | `Int64`         | track end, Unix-ms (`--end-time-field`)        |
| `vertex_timestamps` | `List<Int64>`   | absolute Unix-ms, one per vertex               |
| `vertex_values`     | `List<Float32>` | current speed (m/s), one per vertex            |
| `speed`             | `Float64`       | mean speed along track (numeric property)      |
| `seed_lat`          | `Float64`       | seeding latitude (numeric property)            |
| `basin`             | `Utf8`          | coarse ocean-basin label (categorical)         |

## Citation

ECCO Consortium et al. (2021), _ECCO Central Estimate (Version 4 Release 4)_,
NASA PO.DAAC. DOI [10.5067/ECG5D-OVE44](https://doi.org/10.5067/ECG5D-OVE44).
