# Trajectory representation — motion codecs, motion segments, and what actually pays (2026-11)

**Status:** investigation record, branch `trajectory`. Nothing here changes the
encoder, the format, or a default. Two harnesses and their measurements.

**Question.** Two outside techniques suggested that STT might gain a lot by
storing motion instead of samples: TRiGS (continuous Bézier / SE(3) motion with
local anchors, for dynamic Gaussian splats) and Animated SOG (bounded-lifetime
primitives with a quantized motion model evaluated on the GPU). This record
splits that idea into three separable claims and tests each against what STT
already ships:

1. **Lossless motion coding.** Per-feature predictive coding of the quantized
   coordinates and vertex times (the "predictor is the win" cell the 2026-07
   lightweight-encoding NO-GO left untested,
   [stt-packed-format-decisions.md §4, §10.3](./stt-packed-format-decisions.md)).
2. **Lossy motion models.** Fitted curves (cubic Hermite, the TRiGS idea) against
   linear keyframes (TD-TR, what `--time-aware-simplify` runs) and plain
   coarser quantization, at an equal _measured_ error.
3. **Motion segments with swept-bounds indexing.** Storing each segment once and
   referencing it from every tile it crosses, instead of clipping it per tile.

## Verdict

| Claim                                       | Verdict                                                                                                                                                                                                                                                                                  |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 — lossless predictive coding              | **Bears fruit.** −38 % (AIS) / −48 % (drifters) whole-tile against today's `--quantize-coords` tiles; −53 % / −65 % against today's default f64 tiles. Lossless on the grid, tile-local, feature-local, dedup-preserving. Reader cost +7–13 ms per million vertices, in the worker pool. |
| 1b — `vertex_time` delta coding alone       | **Bears fruit, and reaches default builds.** −39 % (AIS) / −78 % (drifters) on the column, which is 18 % / 6 % of a default f64 trajectory archive: ~5–7 % of the archive, lossless, with no quantization required.                                                                      |
| 2 — fitted motion curves (Hermite / Bézier) | **NO-GO as a size lever.** Ties linear keyframes at coarse budgets, loses at fine ones, on all three corpora. They cut _mean_ error by up to a third at equal max error, which is a fidelity property, not a byte one, and would need a fourth interpolation path in every renderer.     |
| 2b — a better linear keyframe selector      | **Bears fruit (opt-in path only).** A greedy selector that checks the error of its _quantized output_ beats TD-TR by 20–30 % bytes at the same max error on AIS. Applies only to `--simplify --time-aware-simplify`, which stays opt-in.                                                 |
| 3 — swept-bounds segment index (store once) | **NO-GO.** Clip duplication within a zoom is +14 % (AIS) / +24 % (drifters) at max zoom and under 5 % at coarse zooms. The 9× replication is _across_ zooms, which a reference index cannot remove without becoming the additive LOD that already exists (`--additive-lod`).             |

## Setup

Corpora (real data from `data/`, prepared by poopdeck's
`tools/bench/trajectory-codec/prep.py` on its `trajectory` branch):

| corpus       | source                                                    | trajectories | fixes     | cadence                               |
| ------------ | --------------------------------------------------------- | ------------ | --------- | ------------------------------------- |
| AIS          | `data/ais/AIS_2023_01_09.csv`, one day, US waters         | 26,868       | 8,318,004 | irregular, 2 s – minutes; many moored |
| AIS-underway | AIS subset, median ground speed > 4 m/s                   | 2,379        | 675,915   | dense, smooth: the flight stand-in    |
| drifters     | `data/gdp-hourly-cache/drifters-hourly-2021-0{1,2,3}.csv` | 2,665        | 3,126,001 | regular hourly                        |

A run is split wherever two consecutive fixes are more than 30 min (AIS) or 3 h
(drifters) apart, so a gap is never bridged. Flights were the intended smooth
corpus. **The OpenSky sample bucket `stt-generate flights` downloads from is now
empty** (`s3.opensky-network.org/data-samples/states/` lists only
`placeholder.txt`, 2026-11-04), so that generator is broken as of this record.
AIS-underway is the substitute.

Archives were built with `stt-build` 0.6.0 from this tree (the `~/.cargo/bin`
0.1.0 binary writes formatVersion 1, which the current reader refuses):

```bash
stt-build -i ais.parquet -o ais-q1 -t start_time --end-time-field end_time \
  --min-zoom 4 --max-zoom 12 --temporal-bucket 1h --quantize-coords 1
stt-build -i drifters.parquet -o dr-q10 -t start_time --end-time-field end_time \
  --min-zoom 0 --max-zoom 7 --temporal-bucket 1d --quantize-coords 10
# plus the f64 twins (no --quantize-coords), and ais-q1-tdtr (+ --simplify --time-aware-simplify)
```

| archive     | pack bytes | tiles   | stored vertices | stored ÷ source vertices   |
| ----------- | ---------- | ------- | --------------- | -------------------------- |
| ais-f64     | 905.5 MB   | 338,534 | 76,692,978      | 9.2×                       |
| ais-q1      | 621.2 MB   | 338,534 | 76,692,978      | 9.2×                       |
| ais-q1-tdtr | 263.3 MB   | 342,488 | 12,416,654      | 1.5×                       |
| ais-q1-add  | 57.1 MB    | —       | —               | additive LOD (see claim 3) |
| dr-f64      | 506.2 MB   | 232,062 | 28,485,954      | 9.1×                       |
| dr-q10      | 287.3 MB   | 232,062 | 28,485,954      | 9.1×                       |

Harnesses (`crates/stt-core/examples/`):

- **`trajectory-codec`** decodes every tile of an archive through `PackedReader`
  and re-encodes the geometry and `vertex_time` columns several ways. Each column
  is zstd'd alone. A whole-layer check then rebuilds every layer's Arrow IPC frame
  twice, once as stored and once with the codec columns, and zstds both. The IPC
  baseline is larger than the stored blob (861 vs 640 MB on ais-q1) because the
  stored frames use the v2 schema-template splice. Read the whole-layer
  percentages as ratios, not absolute sizes.
- **`trajectory-lossy`** fits each source trajectory, reconstructs it from the
  _quantized_ stored form, and measures the synchronized Euclidean distance
  (SED) to every original fix at its own timestamp. The error reported is the
  error of what would ship. All methods share one entropy back end.

## 1. Lossless predictive coding

Every variant is **tile-local and feature-local**. The first vertex of each
feature is absolute on the world grid, and every later vertex is a residual
against a predictor over that same feature's earlier vertices:

| predictor | prediction for vertex _i_                                                               |
| --------- | --------------------------------------------------------------------------------------- |
| delta     | p<sub>i−1</sub>                                                                         |
| const-vel | 2 p<sub>i−1</sub> − p<sub>i−2</sub>                                                     |
| time-vel  | p<sub>i−1</sub> + (p<sub>i−1</sub> − p<sub>i−2</sub>) · Δt<sub>i</sub>/Δt<sub>i−1</sub> |
| best      | per feature, the cheapest of the three, behind a 1-byte tag                             |

None of this is the rejected inter-timestep delta chain (2026-07-21 NO-GO). A
tile still decodes standalone, and the directory-only time-prune seek is
untouched. Content-addressed dedup still holds, because identical geometry on
the world grid still encodes to identical bytes.

Geometry column, all tiles, zstd-3. Each cell shows the change against that
archive's stored column:

| variant                      | ais-q1 (1 m grid)  | dr-q10 (10 m grid) | vs ais-f64 | vs dr-f64 |
| ---------------------------- | ------------------ | ------------------ | ---------- | --------- |
| grid i32 abs (≈ stored)      | 3.81 B/vtx, −3.5 % | 6.31 B/vtx, −1.9 % | −50.2 %    | −56.6 %   |
| delta, i32 LE                | 2.79, −29.4 %      | 3.83, −40.5 %      | −63.6 %    | −73.7 %   |
| best predictor, i32 (Arrow)  | 2.54, −35.6 %      | 3.40, −47.1 %      |            |           |
| best predictor, i16 + escape | 2.14, −45.6 %      | 2.93, −54.5 %      |            |           |
| delta, zigzag varint         | 1.99, −49.5 %      | 2.83, −55.9 %      | −73.9 %    | −80.5 %   |
| const-vel, varint            | 1.86, −53.0 %      | 2.15, −66.5 %      | −75.7 %    | −85.2 %   |
| time-vel, varint             | 1.82, −53.9 %      | 2.15, −66.5 %      | −76.2 %    | −85.2 %   |
| **best predictor, varint**   | **1.71, −56.6 %**  | **2.20, −65.7 %**  | −77.6 %    | −84.8 %   |

The selector picks differ by regime. AIS takes delta on 77 % of features
(moored and slow vessels) and time-vel on 19 % (irregular cadence). Drifters
take const-vel on 84 % (regular hourly fixes).

`vertex_time`, stored integers re-coded per feature:

| variant            | AIS               | drifters          |
| ------------------ | ----------------- | ----------------- |
| stored (u16 / i64) | 2.02 B/vtx        | 1.04 B/vtx        |
| delta varint       | 1.22, **−39.5 %** | 0.23, −77.7 %     |
| delta-of-delta     | 1.31, −35.2 %     | 0.22, **−79.1 %** |

Whole-layer IPC + zstd, every tile, geometry = best-predictor varint and
`vertex_time` = delta varint:

| archive     | as stored → codec             |
| ----------- | ----------------------------- |
| ais-q1      | 861.5 → 536.1 MB, **−37.8 %** |
| ais-f64     | 1145.9 → 536.1 MB, −53.2 %    |
| dr-q10      | 469.1 → 244.2 MB, **−47.9 %** |
| dr-f64      | 698.3 → 244.2 MB, −65.0 %     |
| ais-q1-tdtr | 491.7 → 363.4 MB, −26.1 %     |

**Robust to the zstd level.** Packs are written once and served many times, so
a higher level is a free comparison. At zstd-19 (sampled 1/17 of tiles) the
stored baseline shrinks 14.5 % (AIS) / 19.8 % (drifters), and the codec still
wins by **−34.2 % / −40.3 %** whole-layer. The codec at zstd-3 beats the stored
format at zstd-19 by 27 % (AIS) / 35 % (drifters).

**Reader cost** (poopdeck `tools/bench/trajectory-codec/decode-bench.mjs`, Node
22, M-series). The codec decoder is a single pass with integer-only varints and
writes the same `Float64Array` the quantized branch of `readCoordRun` writes
today, plus absolute times. It costs 15.0 ns per vertex on AIS (time-vel needs
a division) and 8.7 ns per vertex on drifters, against 1.7 ns per vertex for
today's affine. That is **+13 ms (AIS) / +7 ms (drifters) per million vertices**,
inside the decoder worker pool, with an exact round trip. A z12 AIS tile
averages 57 vertices. No renderer changes: every backend receives the same
decoded arrays it receives now.

**Why the 2026-07 NO-GO does not apply.** That experiment tested transforms on
raw f64, where zstd already models the bit patterns. On the quantized grid the
residuals are small integers, which neither zstd-3 nor zstd-19 can derive from
absolute i32 lanes. The varint's byte-granular width is what zstd then exploits
(compare i32 delta at −29 % with varint delta at −50 %). The other half, as the
2026-07 record predicted, is the predictor (−50 % → −57 % / −56 % → −66 %).

**Shape if shipped (not proposed here, just costed).** The candidate is a new
must-understand layer capability (working name `geometry-predictive`) declared
in TILE_META. There are two encodings to decide between in the `measure`
fair-share harness:

- (a) keep the `List<FixedSizeList<2, Int32>>` leaf but store best-predictor
  residuals (Arrow-native, about 60–70 % of the win);
- (b) a per-feature `Binary` column of varint residuals (the full win).

Either is opaque to a generic GeoArrow consumer, which is already the case for
the quantized i32 leaf. It requires `--quantize-coords`, because a lossless
predictor needs grid integers. The `vertex_time` half does not, and is the
default-safe piece: it ships for every trajectory build. Both clear Stage III's
1 % SKIP line by more than an order of magnitude.

## 2. Lossy motion models: rate–distortion at equal measured error

Methods, each reconstructed from quantized storage:

- **Quant** keeps every fix on a coarser grid.
- **TD-TR** is Douglas–Peucker on SED, what `--time-aware-simplify` runs. Its
  budget is split between simplification and grid (best of 50/50 and 80/20
  reported).
- **GreedyLin** picks linear keyframes greedily, extending each segment while
  the error _after quantizing its endpoints_ stays within budget.
- **Hermite** uses the same greedy search with cubic Hermite segments,
  least-squares tangents (Schneider-style), and tangents quantized and coded as
  residuals against the segment chord, so a straight segment costs almost
  nothing. This is the TRiGS-shaped candidate in its most favourable encoding;
  plain-delta tangents were ~45 % worse.

Bytes per **source** fix, zstd-3. Budget = max SED in metres. For scale, a
quarter pixel at 35° latitude is ≈8 m at z12, 32 m at z10, 128 m at z8 and
512 m at z6.

| budget | corpus       | Quant     | TD-TR | GreedyLin | Hermite   |
| ------ | ------------ | --------- | ----- | --------- | --------- |
| 8 m    | AIS          | 1.215     | 0.711 | **0.540** | 0.572     |
|        | AIS-underway | 1.697     | 1.436 | **1.140** | 1.165     |
|        | drifters     | **1.473** | 1.752 | 1.757     | 1.976     |
| 32 m   | AIS          | 1.000     | 0.328 | **0.260** | 0.271     |
|        | AIS-underway | 1.492     | 0.683 | **0.542** | 0.546     |
|        | drifters     | 1.351     | 1.331 | **1.324** | 1.654     |
| 128 m  | AIS          | 0.875     | 0.170 | **0.131** | 0.135     |
|        | AIS-underway | 1.249     | 0.366 | 0.289     | **0.287** |
|        | drifters     | 0.911     | 1.037 | **0.825** | 0.925     |
| 512 m  | AIS          | 0.812     | 0.088 | **0.070** | 0.073     |
|        | AIS-underway | 1.068     | 0.194 | **0.151** | 0.153     |
|        | drifters     | 0.555     | 0.540 | 0.403     | **0.394** |

What it says:

- **Fitted curves do not pay in bytes.** Hermite halves the keyframe count
  against linear at most budgets (AIS-underway at 32 m: 9.0 % vs 17.5 % of fixes
  kept). The tangents cost as much as the keyframes saved. Its best showing is
  −2 to −5 % at the very coarsest budgets on the smoothest data (drifters at
  512 m and 2 km). Nowhere does it justify a fourth interpolation path across
  four renderers, or the GPU evaluation and trail-window handling that would come
  with it.
- **What curves do buy is shape fidelity.** At equal max error Hermite cuts
  _mean_ error by 25 % (AIS-underway) to 33 % (AIS) at 128 m. That is a reason to
  render with a smooth interpolant if anyone wants one. It is not a reason to
  store curves. It is also the TRiGS objective (image fidelity), not STT's
  (measurement fidelity): a fitted curve between sparse AIS fixes is
  interpolation the evidence does not support.
- **The keyframe selector matters more than the motion model.** GreedyLin beats
  TD-TR by 20–30 % at equal max error on both AIS corpora. TD-TR runs at an
  internal tolerance and never checks its quantized output, and it is not
  minimum-cardinality. This is the gap §1.3 of
  [optimization-problems-2026-08.md](./optimization-problems-2026-08.md) names
  (Imai–Iri). It applies only to the opt-in `--simplify --time-aware-simplify`
  path and moves no default.
- **Regular, already-sparse sampling resists simplification.** Below ~32 m
  almost no hourly drifter fix is removable, and plain coarser quantization is
  the best lever there.

## 3. Swept-bounds motion segments

The proposal is to store each bounded-lifetime segment once and reference it
from every tile and bucket its swept bounds touch, with runtime dedup by
segment id. The upper bound on what that saves is the clip duplication _within_
a zoom: stored vertices at zoom z versus source vertices.

| zoom     | AIS stored ÷ source | drifters stored ÷ source |
| -------- | ------------------- | ------------------------ |
| coarsest | 0.98 (z4)           | 1.04 (z0)                |
| mid      | 1.02 (z9)           | 1.16 (z4)                |
| max      | **1.14 (z12)**      | **1.24 (z7)**            |

Trajectories are already physically sliced at bucket boundaries
(optimization-problems §2.2), so there is no temporal replication to remove. A
reference index would recover at most 14–24 % at the deepest zoom, at the cost
of a multi-blob tile reference (the DEFERRED geometry-blob-sharing design) and
runtime dedup. The replication that dominates is the **9.2× across zooms**, and
removing it is a change of LOD semantics, not of indexing. That is
`--additive-lod`: 57 MB vs 621 MB here. Each trajectory lives at one home zoom,
so zoomed-out views show a subset, which is LOD thinning. It still lacks the
line extent gate (the README K item), and poopdeck deliberately does not declare
`additive-partition` yet. It is the right home for "store once", and it stays
opt-in.

The same arithmetic covers the runtime side of the argument. The flights
120 → 30 fps track-rebuild regression the proposal cites was fixed by the
incremental `TrackIndexMaintainer`
([renderer-architecture](https://github.com/BertCh/poopdeck.gl/blob/main/docs/roadmap/renderer-architecture.md)),
and `AnimatedTripsLayer` already interpolates per-vertex time on the GPU. A
motion codec decodes to the same arrays, so it neither helps nor hurts there.

## Residue once simplified

On `ais-q1-tdtr` geometry falls to 34 % of the column bytes. Per-feature
overhead then dominates: the source string id (22 %), the wire `id` (14 %, a
`segment-hash` minted per clipped segment, ~9 B/feature, incompressible), the
property column and start/end times. Giving a trajectory source a stable integer
id would turn the wire id from noise into a repeated, compressible value. That
is a dataset-preparation lever already documented in the CLI reference, not a
format change, but it is the next-largest term after the geometry.

## Limits of this record

- One day of AIS and three months of drifters. Flights could not be fetched (see
  Setup), so dense, smooth, high-speed motion is represented only by AIS-underway.
- Column-alone and whole-layer IPC sizes, not a real encoder change. The
  whole-layer ratio is the honest number. The 27–35 % margin over zstd-19 makes
  a sign flip implausible, but the `measure` harness on a real encode is the
  gate.
- Lossy error is measured at the original fixes only. There is no ground truth
  between them, which is exactly why fitted curves cannot claim to be more
  accurate there.
- Oriented 3D tracks (AV boxes, meshes, SE(3) pose) were out of scope. TRiGS's
  rigid-transform half may still earn its keep there. Nothing here speaks to it.

## Reproduce

```bash
# in poopdeck.gl (branch trajectory): data prep
python tools/bench/trajectory-codec/prep.py ais data/ais/AIS_2023_01_09.csv ais.parquet
python tools/bench/trajectory-codec/prep.py drifters data/gdp-hourly-cache/drifters-hourly-2021-0{1,2,3}.csv drifters.parquet
python tools/bench/trajectory-codec/prep.py export ais.parquet ais.bin [--underway]

# here: archives as in Setup, then
cargo run --release -p stt-core --example trajectory-codec -- ais-q1/manifest.json 1 0 3
cargo run --release -p stt-core --example trajectory-lossy -- ais.bin
# poopdeck.gl: reader cost
node tools/bench/trajectory-codec/decode-bench.mjs ais.bin 1
```
