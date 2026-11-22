# Trajectory representation — motion codecs, motion segments, and what actually pays (2026-11)

**Status:** investigation record, branch `trajectory`. Round 1 (2026-11-21)
changed nothing. Round 2 (2026-11-22, [below](#round-2-2026-11-22--from-estimate-to-implementation))
re-measured claim 1 on real tile payloads, extended it to bit-exact f64, and
implemented it as the opt-in **path codec** (`stt-build --path-codec`,
capability `path-codec`, [packed spec §5.2.7](../spec/stt-packed-format.md#527-path-codec-section-path_codec--capability-path-codec),
rationale in [path-codec-wire-2026-11.md](./path-codec-wire-2026-11.md)).
No default changed.

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

## Round 2 (2026-11-22) — from estimate to implementation

Round 1 priced the codec column by column and against a re-serialised IPC
frame. Round 2 asked three things: is the number honest on the bytes that
actually ship, does it need `--quantize-coords`, and does it hold on fast,
smooth motion. Then it built the codec.

### What round 1 got wrong or left open

- **The whole-layer ratio included schema bytes.** The round-1 "whole-layer
  IPC" frames carried their Arrow schema, and the codec side's schema was
  smaller (no nested geometry field, no `stt:quant` JSON). Real tiles store the
  schema once, in a template. Round 2 edits the real decompressed payload
  instead (below). The honest figure for the round-1 design is −34 % to −39 %,
  not −38 % to −48 %.
- **`vertex_time` was already lossy, so "lossless" was measured on truncated
  values.** The default build stores vertex times as u16 offsets from the
  layer minimum at `step = ceil(span / 65535)` and truncates
  (`columns.rs`). With 1 h buckets that is a step of up to 55 ms. It truncates
  98 % of AIS layers (mean step 43 ms) and 95 % of flight layers (28 ms).
  Exact times need `--vertex-time-precision 1`, which costs +6 % on AIS.
- **The source-id residue was a prep artifact.** `prep.py` writes ids as
  `<object>-<start ms>`, which is why the string `id` column cost 13 B per
  feature. A plain object id would be far smaller. The wire-id finding stands.
- **Flights were missing.** Now covered: 168,231 tracks and 18.7 M fixes over
  CONUS at 10 s cadence, from one day of adsb.lol `globe_history` heatmap
  files (`prep.py adsb`). Positions are on a 1e-6° grid.
- **The `measure` harness cannot be the gate.** Its sampled features have no
  end time and no per-vertex times, and it does not clip, so a trajectory
  layer never reaches it in the shape that ships
  (`stt-optimize/src/loader.rs`, `measure.rs`). The gate used here is a real
  build of each corpus both ways.

### Method: edit the real payload

`trajectory-frontier` reads each tile's decompressed layer frame, finds the
geometry and `vertex_time` buffers inside the CORE record batch, cuts them out,
appends the codec bytes, and re-zstds the whole frame. The frame header,
TILE_META, ids, compact times and PROPS stay byte-identical, and the batch's
flatbuffer keeps describing the removed buffers, so the codec is charged
slightly too much. It runs on an f64 archive built with
`--vertex-time-precision 1` and emulates the other stored forms from it.
Re-compressing the untouched payload reproduces the directory's stored sizes to
1.1 %, and the emulated `--quantize-coords 1` AIS archive matches the real one
to 0.2 %.

### Bit-exact f64: the codec does not need quantization

Source coordinates are decimal: AIS and drifters at 1e-5° to 1e-6°, ADS-B at
1e-6°. For a decimal exponent `e`, `round(v · 10^e) / 10^e` reproduces the f64
bit pattern of every source vertex, so the integers can be predicted and
varint-coded exactly as on the quantized grid. Vertices the tiler interpolates
(tile-edge clips, bucket-boundary slices) are not decimal. Each keeps its
rounded integer, so the predictor chain stays smooth, plus an XOR patch against
`k / 10^e`. The scheme is exact for any input; a non-decimal source costs
bytes, not correctness.

| corpus   | values needing a patch | at feature endpoints | patch bytes       |
| -------- | ---------------------- | -------------------- | ----------------- |
| AIS      | 7.9 %                  | 77.5 %               | 0.79 B per vertex |
| ADS-B    | 4.9 %                  | 97.0 %               | 0.41 B per vertex |
| drifters | 1.1 %                  | 64.5 %               | 0.21 B per vertex |

Interior patches are the vertices the clipper adds where a track leaves a tile
and re-enters it.

### Exact vertex times that cost less than the lossy ones

Times are split on an epoch-aligned unit `U`: `t = U·q + r`. `U` is the
coarsest unit most of the layer's times are a multiple of (1 s for AIS, 10 s
for ADS-B, 1 h for drifters), so source fixes get `r = 0` and a small regular
`Δq`, and an interpolated clip vertex keeps millisecond precision in `r`
without disturbing its neighbours. Bytes per vertex, column zstd'd alone:

| form                              | AIS      | ADS-B    | drifters |
| --------------------------------- | -------- | -------- | -------- |
| exact, stored today (u32 offsets) | 2.57     | 1.23     | 0.90     |
| default today (u16, truncated)    | 1.83     | 0.85     | 0.51     |
| exact, ms delta varint            | 1.34     | 0.57     | 0.23     |
| exact, quotient/remainder         | **1.08** | **0.51** | 0.27     |

### Frame-level results

A 1/4 sample of tiles per corpus, zstd-3, real payloads. "Default" is today's
default build (f64 coordinates, truncated u16 times).

| variant                              | AIS     | ADS-B   | drifters |
| ------------------------------------ | ------- | ------- | -------- |
| exact codec vs default               | −49.4 % | −55.9 % | −53.7 %  |
| exact codec vs `--quantize-coords`   | −27.1 % | −25.0 % | −19.5 %  |
| grid codec vs `--quantize-coords`    | −36.3 % | −39.9 % | −41.4 %  |
| exact codec vs default, both zstd-19 | −43.5 % | −49.9 % | not run  |

The bit-exact codec is smaller than today's _lossy_ quantized build. The grid
is 1 m for AIS and ADS-B, 10 m for drifters. Flights behave like the others:
the time-aware predictor is chosen for 50 % of flight features and constant
velocity for 41 %.

Layout choices, AIS, each against the alternative: a side section instead of
Arrow `Binary` columns −4.8 %; a per-feature predictor tag instead of one per
layer −1.6 %; the first vertex relative to the previous feature's −1.9 %; three
predictors instead of delta only −5.2 % (−13 % on ADS-B).

An order-0 entropy bound on the grid residuals is 1.23 B per vertex on AIS
against 1.71 for varint + zstd (2.47 against 2.99 on ADS-B). A custom entropy
coder could take roughly a fifth off the geometry stream. That is not enough to
justify a second decoder in JavaScript.

### What was built

- **Format.** Layer-frame section `0x06 PATH_CODEC`, must-understand
  capability `path-codec`. The CORE batch keeps its `geometry` and
  `vertex_time` columns but with every list empty, so the schema template does
  not fork; the decoder substitutes the reconstructed columns before anything
  downstream sees the layer.
- **Rust.** `stt-core::arrow_tile::path_codec`, frame integration, the
  capability, `stt-build --path-codec` (off by default, not enabled by
  `--auto`), a `[path-codec]` line in `stt-optimize inspect`, conformance
  vectors in `conformance/vectors/path-codec/`.
- **TypeScript.** `packages/core/src/path-codec.ts` in poopdeck.gl, substituted
  at the same stage that undoes the compact time columns. No renderer backend
  changed.

Pack bytes of real builds, zstd-3:

| corpus          | without the codec       | `--path-codec` | change  |
| --------------- | ----------------------- | -------------- | ------- |
| AIS, f64        | 905.5 MB (default)      | 441.4 MB       | −51.3 % |
| ADS-B, f64      | 1967.6 MB (exact times) | 834.7 MB       | −57.6 % |
| drifters, f64   | 506.2 MB (default)      | 220.4 MB       | −56.5 % |
| AIS, 1 m grid   | 621.2 MB                | 378.3 MB       | −39.1 % |
| ADS-B, 1 m grid | 1130.7 MB               | 658.8 MB       | −41.7 % |

The ADS-B f64 baseline is the exact-time build, which the frame-level emulation
puts 2.4 % above the default; no default ADS-B f64 archive was built.

**Oracle.** `path-codec-equivalence` decodes every tile of a codec archive and
of the same build without the codec at `--vertex-time-precision 1`, and
compares geometry by bit pattern, vertex times as absolute milliseconds, and
every other column by Arrow equality. AIS, ADS-B and drifters in f64 and an
ADS-B grid build are identical: 905,688 tiles, 257 M vertices. poopdeck.gl
carries the same oracle as an opt-in test
(`packages/core/test/path-codec-archive-equivalence.test.ts`).

**Reader cost.** The TypeScript decoder costs 20–25 ns per vertex for a whole
`decodeTile` against about 6 ns for a plain quantized tile, which is +14–19 ms
per million vertices in the worker pool. Most of it is varint decoding in
JavaScript (four varints per vertex). Round 1's bench had estimated +7–13 ms.

### What the code reading turned up in the tiler

None of these is a codec matter. They are recorded here because a codec that
claims exactness has to say exact _to what_: the codec reproduces the tiler's
output, and the tiler's output differs from the source in these ways.

- **Repeated positions are dropped.** The clipper pushes a vertex only if it
  moved more than 1e-9° from the last one (`clip.rs`), so a stationary object's
  later fixes, and their timestamps, are gone.
- **Exit and re-entry are joined.** A track that leaves a tile and comes back is
  stored as one segment, with a straight edge from the exit point to the
  re-entry point.
- **Clip vertices are not flagged.** Nothing marks a vertex as interpolated.
- **`additive-partition` is emitted by `--additive-lod` but is not in
  `KNOWN_CAPABILITIES`**, so the Rust reader appears to refuse such an archive
  at open. Round 1's `ais-q1-add` produced no harness output, which fits. Not
  verified further.

### Plan

Done on branch `trajectory` in both repositories, uncommitted.

1. **Land it.** The new conformance vectors need a `Rebuild-Window: R1` commit
   trailer for the `pins` gate. poopdeck.gl's vendored copies were synced from
   the local checkout; its pinned ref moves once upstream is pushed.
2. **Release the reader first.** A published `@poopdeck.gl/core` that declares
   `path-codec` has to exist before any hosted archive is rebuilt with the
   flag. Only then is default-on worth discussing.
3. **Make `measure` see trajectories** (end time, per-vertex times, clipping in
   the synthetic layout). Until it does, `--auto` cannot recommend the codec on
   evidence.
4. **Next size levers, in measured order:**
   - a shared zstd dictionary: a further −9 % (ADS-B) to −33 % (drifters) on
     codec tiles, most at deep zooms where tiles are under 1 KB. Blocked on the
     TypeScript zstd decoder, which cannot take a dictionary;
   - clip-vertex patches: 14 % of an AIS codec tile. A clip vertex is a
     deterministic function of its two source neighbours and the tile edge, so
     it could be derived instead of patched, at the price of freezing the
     clipper's arithmetic into the format;
   - the per-segment wire id: 5–6 % of a codec tile, and no trips renderer
     reads it. The dense-id renumbering (`IdRenumberMap`) is written but not
     wired.
5. **Reader cost.** If +14–19 ms per million vertices shows up in frame
   pacing, the options are a WASM decode path or a fixed-width residual layout
   that trades bytes for speed. Not started; no evidence yet that it is needed.
6. **Scope.** v1 covers 2-wide LineString layers. Static lines are already
   eligible but unmeasured (osm-streets is the obvious corpus). Polygon rings
   and 3-wide coordinates are not covered.
7. **Still open from round 1:** the quantization-aware greedy keyframe selector
   for `--simplify --time-aware-simplify` (20–30 % over TD-TR).

## Limits of this record

- One day of AIS, three months of drifters, and (round 2) 18.5 hours of CONUS
  flights at 10 s cadence. Round 1's lossy comparison (§2) still has only
  AIS-underway as its smooth corpus; it was not rerun on flights.
- Round 1's sizes are column-alone and whole-layer IPC, not a real encoder
  change, and its whole-layer ratio is overstated (see round 2). Round 2's are
  real payloads and real builds.
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

Round 2:

```bash
# poopdeck.gl: flights corpus from adsb.lol globe_history heatmap files
python tools/bench/trajectory-codec/prep.py adsb heatmap/*.bin.ttf adsb.parquet

# exact-time reference and codec builds (same flags otherwise)
stt-build -i ais.parquet -o ais-f64x -t start_time --end-time-field end_time \
  --min-zoom 4 --max-zoom 12 --temporal-bucket 1h --vertex-time-precision 1
stt-build -i ais.parquet -o ais-pc   -t start_time --end-time-field end_time \
  --min-zoom 4 --max-zoom 12 --temporal-bucket 1h --path-codec

# frame-level estimates on the reference: <manifest> <grid m> <every nth tile> <zstd level>
cargo run --release -p stt-core --example trajectory-frontier -- ais-f64x/manifest.json 1 4 3
# oracle: the codec archive decodes like the reference
cargo run --release -p stt-core --example path-codec-equivalence -- \
  ais-pc/manifest.json ais-f64x/manifest.json
```
