#!/usr/bin/env python3
"""Shared library for the AV-telemetry-cockpit data adapters.

Every AV adapter — ``av_synthetic.py`` (procedural), ``nuscenes_extract.py``,
``comma_extract.py``, ``argoverse_extract.py`` — imports this module and emits
the SAME on-disk "AV scene bundle" so the cockpit + catalog compose without
collisions. The bundle layout and every column/JSON shape here are governed by
``docs/roadmap/av-cockpit.md`` (§2 data contract). **Do not deviate from the
contract silently** — if a shape here is wrong, flag it to the lead.

A scene bundle lives under ``examples/showcase/public/data/<sceneId>/``::

    <sceneId>/
      scene.json            # manifest (cockpit source of truth)
      lidar/                # packed STT POINT archive (manifest.json + index/ + packs/)
      ego/                  # packed STT TRIPS archive (one linestring = ego path)
      objects/              # packed STT POINT archive (one point / object / sample)
      telemetry.json        # CAN-bus time series sidecar
      cameras.json          # camera keyframe sidecar
      cam/*.jpg             # optional camera frames

Streams are OPTIONAL: comma.ai has only ``ego`` + ``telemetry``; Argoverse has
``lidar`` + ``ego`` + ``objects`` but no CAN telemetry. The cockpit renders only
the streams present in ``scene.json``.

Pipeline per archive:  records → GeoParquet  →  ``stt-build``  →  packed dir.

The GeoParquet schemas are the stt-build contracts (mirroring ``ecco_advect.py``
for the LineString form and ``datasets/storms.rs`` for the point/categorical
form). Two hard rules carried over from those references:

* A **WKB ``geometry`` binary column** named ``geometry`` is picked up by
  stt-build's geometry-column heuristic with no GeoParquet ``geo`` footer
  metadata required (see ``crates/stt-build/src/input.rs::find_geometry_column``).
* **Categorical string columns must NOT be bare numbers.** stt-build promotes an
  all-numeric-string column to a Numeric column, which silently no-ops the
  renderer's categorical color map. So height bands are range labels like
  ``"-2-0"`` / ``"0-2"`` (never ``"0"``), exactly like ``storms.rs``'s
  ``dbz_band``.

Dependencies: ``numpy``, ``pyarrow``, ``shapely`` (same as ``ecco_advect.py``).
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# ── Georeferencing ───────────────────────────────────────────────────────────
# AV ego poses live in a local map frame (metres from a documented map origin).
# Each source map has a documented SW-corner lat/lon origin, so the whole scene
# georeferences onto a real basemap with a simple equirectangular transform
# about the origin (av-cockpit.md §1). Good enough for one coherent ~1km scene.
M_PER_DEG_LAT = 111_320.0  # metres per degree of latitude (contract §1 constant)

# Documented nuScenes map origins (SW-corner lat, lon). Argoverse / comma supply
# their own; adapters pass an explicit (origin_lat, origin_lon).
NUSCENES_MAP_ORIGINS: dict[str, tuple[float, float]] = {
    "boston-seaport": (42.336849169438615, -71.05785369873047),
    "singapore-onenorth": (1.2882100868743724, 103.78475189208984),
    "singapore-hollandvillage": (1.2993652317780957, 103.78217697143555),
    "singapore-queenstown": (1.2782562240223188, 103.76741409301758),
}


def local_to_lonlat(x_m, y_m, origin_lat: float, origin_lon: float):
    """Local map-frame metres → (lon, lat) degrees, equirectangular about origin.

    Vectorised: ``x_m`` / ``y_m`` may be scalars or numpy arrays; returns
    ``(lon, lat)`` of matching shape. Mirrors av-cockpit.md §1 exactly::

        lat = originLat + (y_m / 111320)
        lon = originLon + (x_m / (111320 * cos(originLat)))

    The map-frame metres are TRUE GROUND METRES in a local tangent plane about the
    origin — this is the convention every AV source uses (nuScenes ego/map
    translations, the Argoverse/comma/synthetic frames). nuScenes specifically is
    handled the same way the official devkit ``export_poses.derive_latlon`` does:
    the per-city reference coordinate is the map's SW-corner origin and ego/map
    translations are ground metres from it (haversine in the devkit; the
    equirectangular form here is sub-metre-equivalent over a ~2 km scene).

    NOTE: an earlier ``mercator=True`` mode (treating nuScenes coords as EPSG:3857
    web-mercator metres and deflating by ``1/cos(lat)``) was REMOVED — nuScenes maps
    use a *local* metric frame, not the global EPSG:3857 projection, so that
    deflation shifted Boston scenes ~450 m and shrank them ~26 % (off the basemap).
    """
    x_m = np.asarray(x_m, dtype="float64")
    y_m = np.asarray(y_m, dtype="float64")
    lat = origin_lat + (y_m / M_PER_DEG_LAT)
    lon = origin_lon + (x_m / (M_PER_DEG_LAT * math.cos(math.radians(origin_lat))))
    return lon, lat


# Lazily-built, per-EPSG pyproj Transformer cache (transformers are reusable +
# moderately expensive to construct, so we build one per source CRS once).
_UTM_TRANSFORMERS: dict[int, object] = {}


def utm_to_lonlat(easting, northing, epsg: int):
    """City-frame UTM metres → (lon, lat) degrees via a cached pyproj transform.

    For the Argoverse 2 *city* frame (av-refinement.md §R2.1): AV2 ego/object
    coordinates live in a city-specific UTM CRS (e.g. Pittsburgh ``EPSG:32617``),
    and a scene can sit several km from the UTM origin where the flat-earth
    equirectangular approximation drifts ~75 m. A real CRS transform removes that
    error.

    ``easting`` / ``northing`` may be scalars or numpy arrays (broadcast together);
    returns ``(lon, lat)`` numpy arrays of matching shape. ``epsg`` is the *source*
    UTM EPSG code (e.g. ``32617``). ``pyproj`` is imported lazily (only AV2 needs
    it) and a ``Transformer`` is cached per ``epsg`` so repeated calls are cheap.
    """
    transformer = _UTM_TRANSFORMERS.get(epsg)
    if transformer is None:
        from pyproj import Transformer  # lazy: only the AV2 path needs pyproj

        transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        _UTM_TRANSFORMERS[epsg] = transformer
    easting = np.asarray(easting, dtype="float64")
    northing = np.asarray(northing, dtype="float64")
    lon, lat = transformer.transform(easting, northing)
    return np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")


# ── Object taxonomy ──────────────────────────────────────────────────────────
# The canonical 10-class category set (av-cockpit.md §2c). Adapters map their
# native taxonomy onto this set via ``map_category``; anything unrecognised
# falls back to ``OTHER_CATEGORY``.
CATEGORIES: tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "bicycle",
    "motorcycle",
    "traffic_cone",
    "barrier",
)
OTHER_CATEGORY = "other"

# Canonical category → RGBA color, written into scene.json.objectColors (the
# cockpit's legend + object-inspector swatches read it). This is a projection of
# the nuScenes devkit's own palette (`nuscenes/utils/color_map.py`) onto our
# 10-class taxonomy: vehicles read warm (orange→tomato→red), pedestrians blue,
# cyclists crimson/red, cones dark-slate, barriers slate-grey — so the boxes read
# as "real" nuScenes output. The values + alphas MUST match the showcase Dataset
# copy (`datasets.ts AV_OBJECT_COLORS`, which colors the rendered boxes), so the
# legend swatch equals the box. If you change one, change both.
OBJECT_COLORS: dict[str, list[int]] = {
    "car": [255, 158, 0, 235],  # devkit vehicle.car — orange
    "truck": [255, 99, 71, 235],  # devkit vehicle.truck — tomato
    "bus": [255, 69, 0, 235],  # devkit vehicle.bus.rigid — orangered
    "trailer": [255, 140, 0, 235],  # devkit vehicle.trailer — darkorange
    "construction_vehicle": [233, 150, 70, 235],  # devkit vehicle.construction
    "pedestrian": [0, 80, 230, 240],  # devkit human.pedestrian.adult — blue
    "bicycle": [220, 20, 60, 240],  # devkit vehicle.bicycle — crimson
    "motorcycle": [255, 61, 99, 240],  # devkit vehicle.motorcycle — red
    "traffic_cone": [47, 79, 79, 235],  # devkit movable_object.trafficcone — darkslategrey
    "barrier": [112, 128, 144, 225],  # devkit movable_object.barrier — slategrey
    OTHER_CATEGORY: [150, 160, 175, 220],
}


# ── HD-map layer palette ─────────────────────────────────────────────────────
# Colors for the static HD-map substrate (av-refinement.md §R2.2), keyed by the
# ``map_layer`` string written into the map_poly/ + map_line/ archives. Written
# into scene.json (Dataset.mapColors) so the renderer fills polygons / colors
# paths by ``map_layer``. RGBA; map fills read low-alpha so lidar/objects layer
# on top, dividers read more opaque.
#
# nuScenes layers use the devkit's own ColorBrewer-paired-12 map palette
# (`nuscenes/map_expansion/map_api.py`); Argoverse 2 layers use the AV2 lane /
# drivable / crosswalk convention. A given bundle only emits the layers present
# in its source map; the renderer falls back to a neutral grey for any unmapped
# layer name.
MAP_COLORS: dict[str, list[int]] = {
    # ── nuScenes map layers (devkit palette) ──
    "drivable_area": [166, 206, 227, 90],  # light blue road surface (low alpha = substrate)
    "road_segment": [31, 120, 180, 110],  # blue
    "road_block": [31, 120, 180, 110],  # blue (alias of road_segment grouping)
    "lane": [51, 160, 44, 110],  # green lane polygon
    "ped_crossing": [251, 154, 153, 150],  # pink crosswalk
    "walkway": [227, 26, 28, 110],  # red sidewalk
    "stop_line": [253, 191, 111, 200],  # orange stop line
    "carpark_area": [255, 127, 0, 90],  # orange parking
    "road_divider": [202, 178, 214, 230],  # light purple — line
    "lane_divider": [106, 61, 154, 230],  # deep purple — line
    # ── Argoverse 2 map layers (AV2 convention) ──
    "drivable": [122, 122, 122, 90],  # AV2 drivable area — neutral grey
    "crosswalk": [150, 60, 200, 150],  # AV2 pedestrian crossing — violet
    # AV2 lane boundaries, colored by ``LaneMarkType`` (white/yellow/blue/red).
    "lane_white": [235, 235, 235, 230],
    "lane_yellow": [240, 200, 40, 230],
    "lane_blue": [60, 120, 235, 230],
    "lane_red": [225, 60, 60, 230],
    "lane_boundary": [210, 210, 210, 220],  # generic AV2 boundary fallback
    # AV2 lane CENTERLINES (the dataset's signature map feature) — drawn as
    # subtle steel-blue threads under the action; centerlines inside an
    # intersection read amber so junctions pop.
    "lane_centerline": [90, 130, 165, 95],
    "lane_centerline_intersection": [255, 170, 50, 160],
}

# A generous set of native→canonical aliases covering the nuScenes / Argoverse
# detection-name taxonomies. Adapters can extend per-source if needed.
_CATEGORY_ALIASES: dict[str, str] = {
    "vehicle.car": "car",
    "regular_vehicle": "car",
    "car": "car",
    "vehicle.truck": "truck",
    "truck": "truck",
    "box_truck": "truck",
    "truck_cab": "truck",
    "large_vehicle": "truck",
    "vehicle.bus": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "bus": "bus",
    "school_bus": "bus",
    "articulated_bus": "bus",
    "vehicle.trailer": "trailer",
    "trailer": "trailer",
    "vehicular_trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "construction_vehicle": "construction_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "vehicle.bicycle": "bicycle",
    "bicycle": "bicycle",
    "bicyclist": "bicycle",
    "vehicle.motorcycle": "motorcycle",
    "motorcycle": "motorcycle",
    "motorcyclist": "motorcycle",
    "movable_object.trafficcone": "traffic_cone",
    "traffic_cone": "traffic_cone",
    "construction_cone": "traffic_cone",
    "movable_object.barrier": "barrier",
    "barrier": "barrier",
    "construction_barrel": "barrier",
    "bollard": "barrier",
    # ── Argoverse 2 full-taxonomy additions (the 30-class AV2 sensor set) ──
    # Person-scale movers fold into pedestrian; static roadside furniture +
    # signs into barrier; rare trailers into trailer. Classes not listed
    # (animal, dog, railed_vehicle, ...) fall through to OTHER_CATEGORY.
    "wheelchair": "pedestrian",
    "stroller": "pedestrian",
    "wheeled_device": "pedestrian",
    "wheeled_rider": "pedestrian",
    "official_signaler": "pedestrian",
    "sign": "barrier",
    "stop_sign": "barrier",
    "mobile_pedestrian_crossing_sign": "barrier",
    "message_board_trailer": "trailer",
    "traffic_light_trailer": "trailer",
}


def map_category(native: str) -> str:
    """Map a native taxonomy label onto the canonical 10-class set.

    Tries an exact alias, then a longest-prefix match (so e.g.
    ``"human.pedestrian.adult"`` and ``"human.pedestrian.foo"`` both resolve),
    else returns ``OTHER_CATEGORY``.
    """
    if native is None:
        return OTHER_CATEGORY
    key = str(native).strip().lower()
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    if key in CATEGORIES:
        return key
    # Longest-prefix alias match (nuScenes dotted taxonomy).
    best = None
    for alias, canon in _CATEGORY_ALIASES.items():
        if key.startswith(alias) and (best is None or len(alias) > len(best[0])):
            best = (alias, canon)
    return best[1] if best else OTHER_CATEGORY


# ── Height bands ─────────────────────────────────────────────────────────────
# LIDAR returns get a categorical ``height_band`` string driving their color
# ramp. Bands span the typical urban LIDAR z-domain in 2 m steps; labels are
# RANGE strings (never bare numbers — see module docstring / storms.rs gotcha).
# Edges: -2,0,2,4,6,8,10 → 8 bands (the bottom open band "<-2", the top open
# band ">10", and six closed 2 m bands).
HEIGHT_BAND_EDGES: tuple[float, ...] = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)


def _fmt_edge(v: float) -> str:
    """Compact integer-ish edge label (``-2`` not ``-2.0``)."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


# Ordered band labels (low → high), the canonical ``height_band`` domain. The
# renderer's colorMapping keys onto exactly these strings.
HEIGHT_BANDS: tuple[str, ...] = (
    (f"<{_fmt_edge(HEIGHT_BAND_EDGES[0])}",)
    + tuple(
        f"{_fmt_edge(lo)}-{_fmt_edge(hi)}"
        for lo, hi in zip(HEIGHT_BAND_EDGES[:-1], HEIGHT_BAND_EDGES[1:])
    )
    + (f">{_fmt_edge(HEIGHT_BAND_EDGES[-1])}",)
)


def height_band(z) -> str | np.ndarray:
    """Bucket a height ``z`` (m) into its canonical range-label band string.

    Scalar in → ``str`` out; array in → ``np.ndarray`` of ``str`` out (so a
    whole LIDAR sweep can be banded in one vectorised call).
    """
    edges = np.asarray(HEIGHT_BAND_EDGES, dtype="float64")
    bands = np.asarray(HEIGHT_BANDS, dtype=object)
    # np.searchsorted gives 0..len(edges); index directly maps to bands.
    idx = np.searchsorted(edges, np.asarray(z, dtype="float64"), side="right")
    if np.ndim(z) == 0:
        return str(bands[int(idx)])
    return bands[idx]


# ── LIDAR semantic segmentation ──────────────────────────────────────────────
# When a source ships per-point class labels (nuScenes-lidarseg), the LIDAR cloud
# can be colored by SEMANTIC CLASS instead of ``height_band``. nuScenes-lidarseg
# has 32 fine classes (``nusc.lidarseg_idx2name_mapping``); we collapse them to a
# compact, visually-distinct taxonomy so the cloud reads as a labelled scene
# (orange cars, blue people, green canopy, grey road) rather than a height ramp.
#
# Like ``OBJECT_COLORS`` / ``MAP_COLORS``, this palette lives in TWO copies that
# MUST stay in lockstep: this Python dict (baked into ``scene.json`` for the
# cockpit legend) and ``datasets.ts AV_LIDARSEG_COLORS`` (the ACTUAL rendered
# point colors, via the dataset's ``lidarColorMapping``). Change one → change
# both. Keys are non-numeric strings (the categorical-column rule above).
LIDARSEG_CLASSES: tuple[str, ...] = (
    "vehicle",
    "cyclist",
    "pedestrian",
    "road",
    "sidewalk",
    "terrain",
    "vegetation",
    "manmade",
    "other",
)

LIDARSEG_COLORS: dict[str, list[int]] = {
    "vehicle": [255, 158, 0, 255],  # orange — echoes the car box color
    "cyclist": [220, 20, 60, 255],  # crimson — bicycle + motorcycle returns
    "pedestrian": [40, 130, 255, 255],  # bright blue — people
    "road": [80, 90, 120, 255],  # blue-grey — drivable surface (recedes)
    "sidewalk": [205, 175, 125, 255],  # tan — sidewalk
    "terrain": [150, 140, 70, 255],  # olive — terrain / other flat
    "vegetation": [70, 180, 95, 255],  # green — trees / bushes
    "manmade": [190, 130, 215, 255],  # violet — buildings, poles, barriers, cones
    "other": [120, 125, 140, 255],  # dim grey — noise / unknown
}

# height_band → RGBA, keyed by the canonical HEIGHT_BANDS labels. The COLOUR is
# now baked per-point into `point_rgba` at build (write_lidar_points) so the
# client binds it zero-copy — no per-point categorical colour expansion on the
# render thread. DUAL-COPY: these MUST match `datasets.ts AV_HEIGHT_BAND_COLORS`
# (the legend / any non-baked fallback) exactly, like OBJECT_COLORS /
# LIDARSEG_COLORS. Indexed by the HEIGHT_BANDS order.
HEIGHT_BAND_COLORS: dict[str, list[int]] = {
    HEIGHT_BANDS[0]: [46, 30, 96, 255],   # <-2 below grade — deep indigo
    HEIGHT_BANDS[1]: [52, 60, 158, 255],  # -2-0 ground — blue
    HEIGHT_BANDS[2]: [40, 120, 190, 255], # 0-2 curb / low — cyan-blue
    HEIGHT_BANDS[3]: [38, 168, 168, 255], # 2-4 car-roof height — teal
    HEIGHT_BANDS[4]: [72, 196, 120, 255], # 4-6 green
    HEIGHT_BANDS[5]: [170, 214, 74, 255], # 6-8 lime
    HEIGHT_BANDS[6]: [248, 198, 60, 255], # 8-10 building edge — amber
    HEIGHT_BANDS[7]: [250, 140, 48, 255], # >10 rooftops / canopy — orange
}


def lidarseg_class(name: str) -> str:
    """Collapse a fine nuScenes-lidarseg class NAME onto the coarse taxonomy.

    Matches on the class name (not its integer index) so the mapping survives a
    taxonomy renumbering. Cyclists are checked before the generic ``vehicle.*``
    bucket; movable + static structures fold into ``manmade``; ``noise`` / the
    ego self-returns' neighbours / unknowns land in ``other``.
    """
    key = str(name).strip().lower()
    if key.startswith("vehicle.bicycle") or key.startswith("vehicle.motorcycle"):
        return "cyclist"
    if key.startswith("vehicle"):  # car/bus/truck/trailer/construction/emergency/ego
        return "vehicle"
    if key.startswith("human"):
        return "pedestrian"
    if key.startswith("movable_object") or key.startswith("static_object"):
        return "manmade"
    if key == "static.manmade":
        return "manmade"
    if key == "static.vegetation":
        return "vegetation"
    if key == "flat.driveable_surface":  # devkit's spelling (one 'e')
        return "road"
    if key == "flat.sidewalk":
        return "sidewalk"
    if key.startswith("flat"):  # flat.terrain / flat.other
        return "terrain"
    return "other"  # noise / animal / static.other / unmapped


# ── LIDAR density iso-lines ───────────────────────────────────────────────────
# An alternate LIDAR "view" (waymo_extract.py --contours): instead of drawing the
# returns as points/surfels, bin them into a 2D ground-plane density grid and draw
# the iso-density CONTOURS — a topographic map of where returns CLUSTER (walls,
# parked cars, vegetation, poles). This is height-INDEPENDENT, so it reads richly
# even on a flat scene with no vertical relief (the failure mode of the earlier
# height-contour attempt). Computed per playhead time-window so the contours morph
# live as the car drives.
#
# Bands are ORDINAL (sparsest contour ring → densest core), deliberately decoupled
# from the numeric contour level VALUES so the levels can be retuned without
# touching the colors. Like OBJECT_COLORS / LIDARSEG_COLORS, the palette lives in
# TWO copies that MUST stay in lockstep: these labels and ``datasets.ts
# AV_ISO_DENSITY_COLORS`` (the ACTUAL rendered line colors, via the iso variant's
# ``lidarColorMapping``). Keys are non-numeric strings (the categorical-column
# rule above), so stt-build keeps ``density_band`` categorical.
ISO_DENSITY_BANDS: tuple[str, ...] = ("d1", "d2", "d3", "d4", "d5")


def iso_density_band(level_index: int) -> str:
    """Ordinal ISO_DENSITY_BANDS label for a contour at sorted level ``level_index``
    (0 = the sparsest / outermost ring). Clamped to the band count, so passing more
    contour levels than bands collapses the densest extras onto the hottest band."""
    return ISO_DENSITY_BANDS[min(int(level_index), len(ISO_DENSITY_BANDS) - 1)]


# ── GeoParquet writers ───────────────────────────────────────────────────────
def _wkb_points(lon: Sequence[float], lat: Sequence[float]):
    """WKB-encode parallel lon/lat arrays as POINT geometries (vectorised).

    Uses shapely 2.x's array API (``shapely.points`` + ``shapely.to_wkb``) instead
    of a Python per-point ``wkb.dumps(Point(...))`` loop — ~100× faster, which
    matters for the multi-million-point AV LIDAR clouds (a ~29M-point full-res tier
    encodes in seconds, not minutes). Returns an ndarray of WKB bytes, which
    pyarrow accepts directly as a ``binary`` column.
    """
    import shapely

    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    return shapely.to_wkb(shapely.points(lon, lat))


def write_lidar_points(
    out_path: Path,
    *,
    lon,
    lat,
    timestamp,
    z,
    end_timestamp=None,
    intensity=None,
    seg_class=None,
    rgb=None,
    surfels=None,
    pack_quat=False,
    world_class=None,
    is_dynamic=None,
    scan_phase=None,
    home_zoom=None,
) -> int:
    """Write the ``lidar/`` POINT GeoParquet (av-cockpit.md §2a).

    Columns: ``geometry`` (WKB Point), ``timestamp`` (Int64 unix-ms),
    ``height_band`` (Utf8 categorical range label), ``z`` (Float64). NOTE:
    ``intensity`` is accepted for call-site compatibility but **no longer
    written** — measured on Waymo LiDAR it was ~8% of every tile's compressed
    bytes (a near-incompressible Float64) yet nothing in the render path reads
    it (the cloud colors by ``height_band`` / ``seg_class`` / camera RGB). When
    ``seg_class`` is given (a per-point array of
    coarse ``LIDARSEG_CLASSES`` labels, e.g. nuScenes-lidarseg collapsed via
    ``lidarseg_class``), an extra ``seg_class`` (Utf8 categorical) column is
    written so the cloud can be colored by SEMANTIC CLASS. ``height_band`` is
    always written too, so a scene can fall back to the height ramp.

    When ``rgb`` is given (an ``(N, 3)`` uint8-ish array of per-point RGB sampled
    by projecting each return into a camera image — see
    ``waymo_extract.CameraColorizer``), three NUMERIC ``r`` / ``g`` / ``b`` (Int16
    0–255) columns are written. These flow through stt-build like ``z`` /
    ``intensity`` and reach the client as ``numericProps``, where the layer can
    paint each point its own color (``AnimatedPointLayer.rgbColorColumns``)
    instead of a categorical ramp. ``height_band`` is still written, so the same
    bundle can fall back to the height ramp.

    When ``surfels`` is given (an ``(N, 7)`` float array
    ``[qx, qy, qz, qw, s_major, s_minor, opacity]`` from the per-sweep k-NN
    covariance pass — see ``waymo_extract.compute_surfels``), seven NUMERIC
    Float64 columns are written: the orientation quaternion ``qx``/``qy``/``qz``/
    ``qw`` (its rotation-matrix COLUMNS are the surfel ``[tangent|bitangent|
    normal]`` in the render ENU frame), the in-plane half-extents ``s_major`` /
    ``s_minor`` (metres), and a per-surfel confidence ``surfel_opacity``
    (``[0,1]``). The client's ``SplatLayer`` reads these as ``numericProps`` to
    render each return as an oriented anisotropic Gaussian disk instead of a flat
    dot. Returns the row count written.

    Three optional columns for the new AV render modes (each independent):

    * ``world_class`` (Feature 1 — Worldbuild): a per-point ``"static"`` /
      ``"dynamic"`` Utf8 CATEGORICAL label. Deliberately string-valued (NOT
      ``0``/``1``) so stt-build keeps it categorical — an all-numeric-string
      column is silently promoted to Numeric, no-opping a categorical color map.
    * ``is_dynamic`` (Feature 1): a NUMERIC Int64 ``0``/``1`` per point (``1`` =
      a return inside a MOVING actor box). Flows through stt-build as a
      ``numericProp`` (like ``z`` / ``r``); the worldbuild renderer keys
      static-vs-dynamic on it. Quantize via ``lidar_quantize_attrs``.
    * ``scan_phase`` (Feature 5 — Sweep, AV2 only): a NUMERIC Float64 in
      ``[0,1]`` per point, the return's normalized offset within its sweep (so
      the renderer can drive a rotating-beam reveal). Quantize via
      ``lidar_quantize_attrs``. The phase→hue ramp is precomputed into
      ``r``/``g``/``b`` by the caller, so the existing rgb path renders the beam
      as a rotating rainbow.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    z = np.asarray(z, dtype="float64")
    timestamp = np.asarray(timestamp, dtype="int64")
    bands = height_band(z)  # vectorised → str array

    # `intensity` is intentionally NOT written — it is dead weight in the tile
    # (no consumer reads it) and the largest avoidable Float64 column after `z`.
    cols = {
        "geometry": pa.array(_wkb_points(lon, lat), type=pa.binary()),
        "timestamp": pa.array(timestamp, type=pa.int64()),
        "height_band": pa.array([str(b) for b in bands], type=pa.string()),
        "z": pa.array(z, type=pa.float64()),
    }
    if end_timestamp is not None:
        # Full-range validity (scene-split STATIC stage): each point is valid over
        # [timestamp, end_timestamp]. Paired with run_stt_build(static_full_range=
        # True) + a whole-scene bucket, the stage lands in ONE bucket that overlaps
        # the playhead for the entire replay, so it loads once and persists — the
        # HD-map idiom applied to a POINT archive. Scalar → broadcast to all rows.
        et = (np.full(len(lon), int(end_timestamp), dtype="int64")
              if np.ndim(end_timestamp) == 0
              else np.asarray(end_timestamp, dtype="int64"))
        if len(et) != len(lon):
            raise ValueError(
                f"lidar: end_timestamp length {len(et)} != point count {len(lon)}"
            )
        cols["end_timestamp"] = pa.array(et, type=pa.int64())
    if seg_class is not None:
        seg = np.asarray(seg_class, dtype=object)
        if len(seg) != len(lon):
            raise ValueError(
                f"lidar: seg_class length {len(seg)} != point count {len(lon)}"
            )
        cols["seg_class"] = pa.array([str(s) for s in seg], type=pa.string())
    # Bake a per-point colour for any cloud that doesn't carry camera RGB, so
    # EVERY point bundle ships an interleaved `point_rgba` the client binds
    # zero-copy — no per-point categorical colour expansion on the render thread.
    # Semantic class when present (nuScenes-lidarseg), else the height-band ramp
    # (digitize matches `height_band` exactly). Camera-coloured clouds keep `rgb`.
    if rgb is None:
        if seg_class is not None:
            seg = np.asarray(seg_class, dtype=object)
            rgb = np.array(
                [LIDARSEG_COLORS.get(str(s), LIDARSEG_COLORS["other"])[:3] for s in seg],
                dtype="int64",
            )
        else:
            edges = np.asarray(HEIGHT_BAND_EDGES, dtype="float64")
            lut = np.array(
                [HEIGHT_BAND_COLORS[b][:3] for b in HEIGHT_BANDS], dtype="int64"
            )
            idx = np.clip(np.digitize(z, edges), 0, len(HEIGHT_BANDS) - 1)
            rgb = lut[idx]
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.shape != (len(lon), 3):
            raise ValueError(
                f"lidar: rgb shape {rgb.shape} != (point count {len(lon)}, 3)"
            )
        # Int64 (NOT int16/uint8): stt-build's `extract_property_value`
        # (crates/stt-build/src/input.rs) only reads Float64/Int64/Float32/Int32/
        # String/Bool — an Int16 column is SILENTLY DROPPED (returns None per row),
        # so the per-point color never reaches the client and the cloud falls back
        # to the height ramp. Int64 is the proven path (== num_interior_pts); the
        # tile stores it as Float64 either way, so the client reads 0–255 floats.
        rgb = np.clip(rgb, 0, 255).astype("int64")
        cols["r"] = pa.array(rgb[:, 0], type=pa.int64())
        cols["g"] = pa.array(rgb[:, 1], type=pa.int64())
        cols["b"] = pa.array(rgb[:, 2], type=pa.int64())
        # Alpha channel (0–255) so the build can fuse r,g,b,a into ONE interleaved
        # `FixedSizeList<UInt8,4>` colour column (`stt-build --vector-group
        # surfel_rgba=r,g,b,a:u8`) the client binds zero-copy — no per-point RGBA
        # re-pack on the render thread. For a SURFEL cloud alpha carries the baked
        # per-surfel confidence (surfels[:,6] = surfel_opacity, 0–1 → 0–255); a
        # plain coloured point cloud is opaque (255).
        if surfels is not None:
            conf = np.clip(np.asarray(surfels, dtype="float64")[:, 6], 0.0, 1.0)
            alpha = np.clip(np.round(conf * 255.0), 0, 255).astype("int64")
        else:
            alpha = np.full(len(lon), 255, dtype="int64")
        cols["a"] = pa.array(alpha, type=pa.int64())
    if surfels is not None:
        surfels = np.asarray(surfels, dtype="float64")
        if surfels.shape != (len(lon), 7):
            raise ValueError(
                f"lidar: surfels shape {surfels.shape} != (point count {len(lon)}, 7)"
            )
        # Float64 (NOT a packed list column): stt-build's per-column property
        # extractor reads scalar Float64/Int64 columns, like ``z`` / ``intensity``,
        # and hands them to the client as ``numericProps``. The quaternion is unit
        # length and the extents are small metre values, all f32-exact downstream.
        # `surfel_opacity` is folded into the colour alpha above; `s_major` /
        # `s_minor` are fused into the `surfel_scale` vector column at build time
        # (`stt-build --vector-group surfel_scale=s_major,s_minor`).
        for j, name in enumerate(("s_major", "s_minor", "surfel_opacity"), start=4):
            cols[name] = pa.array(surfels[:, j], type=pa.float64())
        # ALWAYS write the FULL quaternion qx,qy,qz,qw. The build fuses them into
        # one `FixedSizeList<Float32,4>` (`--vector-group surfel_quat=qx,qy,qz,qw`)
        # the client binds zero-copy; smallest-three packing is obsolete (it saved
        # one scalar column, but the interleaved vector is width-4 either way and
        # the client no longer unpacks). `pack_quat` is retained for signature
        # compatibility but no longer changes the emitted columns.
        for j, name in enumerate(("qx", "qy", "qz", "qw")):
            cols[name] = pa.array(surfels[:, j], type=pa.float64())
    if world_class is not None:
        # Feature 1 (Worldbuild): a CATEGORICAL static/dynamic label. String
        # values ("static"/"dynamic"), never bare 0/1 — an all-numeric-string
        # column is promoted to Numeric by stt-build (the categorical no-op bug).
        wc = np.asarray(world_class, dtype=object)
        if len(wc) != len(lon):
            raise ValueError(
                f"lidar: world_class length {len(wc)} != point count {len(lon)}"
            )
        cols["world_class"] = pa.array([str(w) for w in wc], type=pa.string())
    if is_dynamic is not None:
        # Feature 1 (Worldbuild): a NUMERIC 0/1 flag (== num_interior_pts path,
        # Int64 so stt-build's property extractor reads it → numericProps).
        idn = np.asarray(is_dynamic, dtype="int64")
        if len(idn) != len(lon):
            raise ValueError(
                f"lidar: is_dynamic length {len(idn)} != point count {len(lon)}"
            )
        cols["is_dynamic"] = pa.array(idn, type=pa.int64())
    if scan_phase is not None:
        # Feature 5 (Sweep, AV2 only): a NUMERIC Float64 in [0,1], the per-return
        # normalized offset within its sweep (drives the rotating-beam reveal).
        sp = np.asarray(scan_phase, dtype="float64")
        if len(sp) != len(lon):
            raise ValueError(
                f"lidar: scan_phase length {len(sp)} != point count {len(lon)}"
            )
        cols["scan_phase"] = pa.array(sp, type=pa.float64())
    if home_zoom is not None:
        # Additive-octree LOD (see lod_home_zoom): the single zoom level each
        # return is materialized at. NUMERIC Int64 so stt-build's property
        # extractor reads it → used as BOTH --min-zoom-field and --max-zoom-field
        # (the feature lands in exactly that zoom's tiles) and shipped to the
        # client (quantized u16 via lidar_quantize_attrs) for optional
        # fractional-zoom shader smoothing.
        hz = np.asarray(home_zoom, dtype="int64")
        if len(hz) != len(lon):
            raise ValueError(
                f"lidar: home_zoom length {len(hz)} != point count {len(lon)}"
            )
        cols["home_zoom"] = pa.array(hz, type=pa.int64())

    table = pa.table(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    extras = []
    if seg_class is not None:
        extras.append("seg_class")
    if rgb is not None:
        extras.append("rgb")
    if surfels is not None:
        extras.append("surfels")
    if world_class is not None:
        extras.append("world_class")
    if is_dynamic is not None:
        extras.append("is_dynamic")
    if scan_phase is not None:
        extras.append("scan_phase")
    if home_zoom is not None:
        extras.append("home_zoom")
    suffix = f" (+{', '.join(extras)})" if extras else ""
    print(f"  wrote {table.num_rows} LIDAR points{suffix} → {out_path}")
    return table.num_rows


def write_ego_trips(
    out_path: Path,
    *,
    lon,
    lat,
    vertex_timestamps,
    vertex_values,
    vehicle: str = "ego",
) -> int:
    """Write the ``ego/`` LineString GeoParquet (av-cockpit.md §2b).

    One LineString = the ego path. Columns: ``geometry`` (WKB LineString),
    ``timestamp`` (Int64 track start), ``end_timestamp`` (Int64 track end),
    ``vertex_timestamps`` (List<Int64> unix-ms / vertex), ``vertex_values``
    (List<Float32> speed m/s / vertex), ``vehicle`` (Utf8 constant label).

    Identical contract to ``ecco_advect.py``. Returns the row count (1).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import wkb
    from shapely.geometry import LineString

    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    vts = [int(t) for t in vertex_timestamps]
    vvals = [np.float32(v) for v in vertex_values]
    if not (len(lon) == len(lat) == len(vts) == len(vvals)):
        raise ValueError("ego: lon/lat/vertex_timestamps/vertex_values length mismatch")
    if len(lon) < 2:
        raise ValueError("ego: need at least 2 vertices for a LineString")

    coords = list(zip(lon.tolist(), lat.tolist()))
    table = pa.table(
        {
            "geometry": pa.array([wkb.dumps(LineString(coords))], type=pa.binary()),
            "timestamp": pa.array([vts[0]], type=pa.int64()),
            "end_timestamp": pa.array([vts[-1]], type=pa.int64()),
            "vertex_timestamps": pa.array([vts], type=pa.list_(pa.int64())),
            "vertex_values": pa.array([vvals], type=pa.list_(pa.float32())),
            "vehicle": pa.array([vehicle], type=pa.string()),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  wrote {table.num_rows} ego LineString ({len(vts)} vertices) → {out_path}")
    return table.num_rows


def write_objects_points(
    out_path: Path,
    *,
    lon,
    lat,
    timestamp,
    category,
    heading,
    length,
    width,
    height,
    track_id,
    speed,
    num_interior_pts=None,
) -> int:
    """Write the ``objects/`` POINT GeoParquet (av-cockpit.md §2c).

    One point per object per annotated sample. Columns: ``geometry`` (WKB
    Point), ``timestamp`` (Int64 unix-ms), ``category`` (Utf8 categorical),
    ``heading`` (Float64 radians, 0=+x/east, CCW+), ``length`` / ``width`` /
    ``height`` (Float64 m), ``track_id`` (Utf8), ``speed`` (Float64 m/s).

    When ``num_interior_pts`` is given (a per-row array — Argoverse 2 ships the
    LIDAR-point count inside each cuboid), an Int64 ``num_interior_pts`` column
    is written too, for the object inspector / ghost-box filtering. Omitted
    (``None``) for sources without it. Returns the row count written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    cols = {
        "geometry": pa.array(_wkb_points(lon, lat), type=pa.binary()),
        "timestamp": pa.array(np.asarray(timestamp, dtype="int64"), type=pa.int64()),
        "category": pa.array([str(c) for c in category], type=pa.string()),
        "heading": pa.array(np.asarray(heading, dtype="float64"), type=pa.float64()),
        "length": pa.array(np.asarray(length, dtype="float64"), type=pa.float64()),
        "width": pa.array(np.asarray(width, dtype="float64"), type=pa.float64()),
        "height": pa.array(np.asarray(height, dtype="float64"), type=pa.float64()),
        "track_id": pa.array([str(t) for t in track_id], type=pa.string()),
        "speed": pa.array(np.asarray(speed, dtype="float64"), type=pa.float64()),
    }
    if num_interior_pts is not None:
        nip = np.asarray(num_interior_pts, dtype="int64")
        if len(nip) != len(lon):
            raise ValueError(
                f"objects: num_interior_pts length {len(nip)} != point count {len(lon)}"
            )
        cols["num_interior_pts"] = pa.array(nip, type=pa.int64())
    table = pa.table(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  wrote {table.num_rows} object points → {out_path}")
    return table.num_rows


# ── HD-map GeoParquet writers ─────────────────────────────────────────────────
# The static HD-map substrate (av-refinement.md §R2.2). Both archives carry a
# single time row spanning the WHOLE scene window (``timestamp = t_start``,
# ``end_timestamp = t_end``) so they load once and persist for the entire replay
# (built with a temporal bucket >= the scene duration — see ``run_stt_build``
# ``map_poly`` / ``map_line`` kinds). ``map_layer`` is a categorical Utf8 layer
# name (e.g. ``"drivable_area"``, ``"lane_divider"``) — NEVER a bare number — so
# stt-build keeps it categorical and the renderer can color by ``MAP_COLORS``.
def write_map_polygons(out_parquet: Path, polys, t_start: int, t_end: int) -> int:
    """Write the ``map_poly/`` source GeoParquet (av-refinement.md §R2.2).

    ``polys`` is an iterable of ``(polygon, map_layer)`` where ``polygon`` is a
    shapely ``Polygon`` in **lon/lat** degrees and ``map_layer`` is the layer
    name string. Columns: ``geometry`` (WKB Polygon), ``map_layer`` (Utf8
    categorical), ``timestamp`` (Int64 = ``t_start``), ``end_timestamp``
    (Int64 = ``t_end``). Returns the row count written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import wkb

    geoms: list[bytes] = []
    layers: list[str] = []
    for poly, layer in polys:
        geoms.append(wkb.dumps(poly))
        layers.append(str(layer))
    n = len(geoms)
    table = pa.table(
        {
            "geometry": pa.array(geoms, type=pa.binary()),
            "map_layer": pa.array(layers, type=pa.string()),
            "timestamp": pa.array([int(t_start)] * n, type=pa.int64()),
            "end_timestamp": pa.array([int(t_end)] * n, type=pa.int64()),
        }
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_parquet, compression="snappy")
    print(f"  wrote {n} map polygon(s) → {out_parquet}")
    return n


def write_map_lines(out_parquet: Path, lines, t_start: int, t_end: int) -> int:
    """Write the ``map_line/`` source GeoParquet (av-refinement.md §R2.2).

    ``lines`` is an iterable of ``(linestring, map_layer)`` where ``linestring``
    is a shapely ``LineString`` in **lon/lat** degrees. Columns: ``geometry``
    (WKB LineString), ``map_layer`` (Utf8 categorical), ``timestamp``
    (Int64 = ``t_start``), ``end_timestamp`` (Int64 = ``t_end``). Returns the row
    count written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import wkb

    geoms: list[bytes] = []
    layers: list[str] = []
    for line, layer in lines:
        geoms.append(wkb.dumps(line))
        layers.append(str(layer))
    n = len(geoms)
    table = pa.table(
        {
            "geometry": pa.array(geoms, type=pa.binary()),
            "map_layer": pa.array(layers, type=pa.string()),
            "timestamp": pa.array([int(t_start)] * n, type=pa.int64()),
            "end_timestamp": pa.array([int(t_end)] * n, type=pa.int64()),
        }
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_parquet, compression="snappy")
    print(f"  wrote {n} map line(s) → {out_parquet}")
    return n


def write_contour_lines(out_parquet: Path, lines) -> int:
    """Write the LIDAR density-contour iso-line GeoParquet (waymo_extract.py --contours).

    ``lines`` is an iterable of ``(linestring, density_band, z_layer, t_start, t_end)``
    where ``linestring`` is a shapely ``LineString`` in **lon/lat** degrees,
    ``density_band`` is an ordinal ``ISO_DENSITY_BANDS`` label, ``z_layer`` is the
    contour's real altitude in METRES (0 for the flat 2D mode; the height-slab centre
    for the z-layered 3D mode), and ``[t_start, t_end]`` is the playhead window
    (unix-ms) the contour is shown for. Columns: ``geometry`` (WKB LineString),
    ``density_band`` (Utf8 categorical), ``z_layer`` (Float64 — a NUMERIC per-feature
    property the client lifts the line to; line geometry stays 2D because the Rust
    tiler drops line-vertex Z, so the height rides as this column instead),
    ``timestamp`` + ``end_timestamp`` (Int64 unix-ms). Built with
    ``run_stt_build(kind="line")`` so each contour shows for its own window (the
    contour map morphs as the playhead moves). Returns the row count written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import wkb

    geoms: list[bytes] = []
    bands: list[str] = []
    zls: list[float] = []
    t0s: list[int] = []
    t1s: list[int] = []
    for line, band, z_layer, t_start, t_end in lines:
        geoms.append(wkb.dumps(line))
        bands.append(str(band))
        zls.append(float(z_layer))
        t0s.append(int(t_start))
        t1s.append(int(t_end))
    n = len(geoms)
    table = pa.table(
        {
            "geometry": pa.array(geoms, type=pa.binary()),
            "density_band": pa.array(bands, type=pa.string()),
            "z_layer": pa.array(zls, type=pa.float64()),
            "timestamp": pa.array(t0s, type=pa.int64()),
            "end_timestamp": pa.array(t1s, type=pa.int64()),
        }
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_parquet, compression="snappy")
    print(f"  wrote {n} contour line(s) → {out_parquet}")
    return n


# ── density iso-lines (build-time) ────────────────────────────────────────────
# An alternate LIDAR "view" for the cockpit (`--contours`): instead of points /
# surfels, draw the iso-density CONTOURS of the returns — a live topographic map
# of where the cloud CLUSTERS (walls, parked cars, vegetation, poles), morphing as
# the car drives. SHARED by waymo_extract.py and argoverse_extract.py (both call
# avc.build_density_contours / avc.add_contour_args). See ISO_DENSITY_BANDS for the
# color contract.
#
# Two modes, chosen by `--contour-z-step`:
#   • FLAT (z-step 0): contour the 2D ground-plane density (one grid per window).
#     Height-INDEPENDENT, so it reads even on a flat scene. z_layer = 0.
#   • TRUE-3D (z-step > 0): slice the returns into HEIGHT layers (slab centres
#     spaced z_step m, each accumulating returns within z_thickness m) and contour
#     each slab's XY density INDEPENDENTLY, tagging every contour with its slab's
#     real altitude `z_layer` (metres). The vertical axis then carries real
#     structure — a wall contours up its whole height, a parked car only near the
#     ground — and the client lifts each ring to that altitude. The contour LEVELS
#     are pooled across slabs so a band means the same density at every height.
def build_density_contours(lon, lat, z, t_ms, *, anchor_lat,
                           cell_m=0.5, sigma_cells=1.6, n_levels=5,
                           win_step_ms=200, win_width_ms=900,
                           min_len=12, min_pts=2000, simplify_cells=0.0,
                           z_step=0.0, z_thickness=None,
                           z_min=None, z_max=None, min_pts_layer=120):
    """Per-playhead-window density contours of the returns → iso-line records.

    Bins the returns into a FIXED lon/lat grid (cell ≈ ``cell_m`` m), then for each
    window steps the playhead by ``win_step_ms`` and accumulates the returns within
    a ``win_width_ms`` window centred on the step (a wider accumulation than the
    step → denser, smoother contours that still morph live). Each grid is
    gaussian-smoothed (``sigma_cells``) and contoured at ``n_levels`` FIXED density
    levels derived once from a subsample (so the bands — and thus the colors — are
    stable across the whole drive, not per-window-relative and flickery). Tiny
    loops (< ``min_len`` vertices) are dropped.

    When ``z_step > 0`` the per-window grid is replaced by a STACK of grids, one
    per height slab (centres spaced ``z_step`` m over [``z_min``, ``z_max``] —
    defaulting to the 1st/99th z percentiles — each accumulating returns within
    ``z_thickness`` m, default 1.5×``z_step``). Each contour is tagged with its
    slab centre as ``z_layer`` (metres). Slabs with < ``min_pts_layer`` returns are
    skipped. ``z`` is the per-return height in metres (ego/vehicle frame, same
    column the point cloud renders as elevation).

    Yields ``(LineString, density_band, z_layer, t_start, t_end)`` tuples for
    ``write_contour_lines``; coords are lon/lat degrees, the window is
    ``[w, w + win_step_ms)`` (so consecutive windows tile the timeline cleanly).
    """
    from scipy import ndimage as ndi
    from skimage import measure
    from shapely.geometry import LineString

    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    z = np.asarray(z, dtype="float64")
    t_ms = np.asarray(t_ms, dtype="int64")
    if len(lon) == 0:
        return []

    layered = bool(z_step and z_step > 0)

    # Fixed grid over the full scene extent (a little padding so edge structure
    # isn't clipped). Cell size is set in metres → degrees at the scene latitude.
    pad = cell_m * 2
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(anchor_lat))
    dlon = cell_m / m_per_deg_lon
    dlat = cell_m / m_per_deg_lat
    lon0 = float(lon.min()) - pad / m_per_deg_lon
    lat0 = float(lat.min()) - pad / m_per_deg_lat
    nx = max(8, int((float(lon.max()) + pad / m_per_deg_lon - lon0) / dlon) + 1)
    ny = max(8, int((float(lat.max()) + pad / m_per_deg_lat - lat0) / dlat) + 1)
    lon_edges = lon0 + np.arange(nx + 1) * dlon
    lat_edges = lat0 + np.arange(ny + 1) * dlat

    # Pre-sort by time so each window is a contiguous slice (searchsorted).
    order = np.argsort(t_ms, kind="stable")
    lon, lat, z, t_ms = lon[order], lat[order], z[order], t_ms[order]
    t0, t1 = int(t_ms[0]), int(t_ms[-1])

    def smoothed_grid(lo, la):
        H, _, _ = np.histogram2d(lo, la, bins=[lon_edges, lat_edges])
        return ndi.gaussian_filter(H, sigma=sigma_cells, mode="constant")

    # Window starts. Each shows for [w, w+step); returns accumulated over the
    # wider [centre-width/2, centre+width/2).
    starts = list(range(t0, t1 + 1, int(win_step_ms)))
    half = win_width_ms / 2.0

    def window_slice(w):
        centre = w + win_step_ms / 2.0
        a = np.searchsorted(t_ms, centre - half, side="left")
        b = np.searchsorted(t_ms, centre + half, side="right")
        return a, b

    # Height slabs (3D) or a single flat layer (2D). Each slab is (centre, half-thick).
    if layered:
        zlo = float(np.percentile(z, 1)) if z_min is None else float(z_min)
        zhi = float(np.percentile(z, 99)) if z_max is None else float(z_max)
        if zhi - zlo < z_step:
            zhi = zlo + z_step
        zhalf = (z_thickness if z_thickness else z_step * 1.5) / 2.0
        slabs = [(float(zc), zhalf) for zc in np.arange(zlo, zhi + 1e-6, z_step)]
        floor = int(min_pts_layer)
    else:
        slabs = [(0.0, None)]      # one layer, all heights, z_layer=0
        floor = int(min_pts)

    def slab_grid(lo_w, la_w, z_w, zc, zhalf):
        """Smoothed XY density grid for one slab of a window (None zhalf → all z)."""
        if zhalf is None:
            return smoothed_grid(lo_w, la_w), len(lo_w)
        m = np.abs(z_w - zc) <= zhalf
        c = int(m.sum())
        if c < floor:
            return None, c
        return smoothed_grid(lo_w[m], la_w[m]), c

    # ── pass 1: derive FIXED contour levels, pooled across sample windows × slabs ──
    sample_idx = np.unique(
        np.linspace(0, len(starts) - 1, num=min(12, len(starts))).astype(int))
    pooled = []
    for si in sample_idx:
        a, b = window_slice(starts[si])
        if b - a < floor:
            continue
        lo_w, la_w, z_w = lon[a:b], lat[a:b], z[a:b]
        for zc, zhalf in slabs:
            g, _c = slab_grid(lo_w, la_w, z_w, zc, zhalf)
            if g is None:
                continue
            nz = g[g > 0.5]
            if nz.size:
                pooled.append(nz)
    if not pooled:
        print("  (--contours) every window/slab below the point floor — no contours")
        return []
    pooled = np.concatenate(pooled)
    # Percentile ladder across the n_levels bands (sparse outer → dense core).
    pcts = np.linspace(55, 96, num=n_levels)
    levels = np.unique(np.round(np.percentile(pooled, pcts), 2))
    levels = levels[levels > 0]
    if layered:
        print(f"  (--contours) 3D: {nx}×{ny} grid (cell {cell_m} m), {len(starts)} "
              f"windows × {len(slabs)} z-slabs (step {z_step} m, "
              f"{slabs[0][0]:.1f}…{slabs[-1][0]:.1f} m), levels={levels.tolist()}")
    else:
        print(f"  (--contours) 2D: {nx}×{ny} grid (cell {cell_m} m), "
              f"{len(starts)} windows, levels={levels.tolist()}")

    # ── pass 2: contour every window (× slab) at the fixed levels ──
    out = []
    n_win = 0
    for w in starts:
        a, b = window_slice(w)
        if b - a < floor:
            continue
        lo_w, la_w, z_w = lon[a:b], lat[a:b], z[a:b]
        w_end = w + int(win_step_ms)
        active = False
        for zc, zhalf in slabs:
            g, _c = slab_grid(lo_w, la_w, z_w, zc, zhalf)
            if g is None or g.max() < levels[0]:
                continue
            active = True
            for li, lvl in enumerate(levels):
                band = iso_density_band(li)
                for seg in measure.find_contours(g, float(lvl)):
                    if len(seg) < min_len:
                        continue
                    # find_contours emits a vertex at EVERY cell crossing — a
                    # dense staircase. Douglas-Peucker in CELL units (tolerance
                    # < 1 cell) collapses that to clean lines, cutting vertices
                    # ~3-5× with NO visible change to the contour shape — the
                    # cheap lever that makes a FINE grid (high XY resolution)
                    # affordable. Done in cell space so the tolerance is uniform.
                    if simplify_cells > 0:
                        s = LineString(seg).simplify(simplify_cells,
                                                     preserve_topology=False)
                        seg = np.asarray(s.coords)
                        if len(seg) < 2:
                            continue
                    # (row, col) array-index space; row ↔ lon (axis-0) bin, col ↔
                    # lat (axis-1) bin. Map each index to its cell CENTRE in lon/lat.
                    pl = lon0 + (seg[:, 0] + 0.5) * dlon
                    pa = lat0 + (seg[:, 1] + 0.5) * dlat
                    coords = np.column_stack([pl, pa])
                    out.append((LineString(coords), band, zc, w, w_end))
        if active:
            n_win += 1
    print(f"  (--contours) {len(out)} contour lines over {n_win} active windows")
    return out


def add_contour_args(p) -> None:
    """Register the shared ``--contours`` / ``--contour-*`` CLI flags on an
    argparse parser. Used identically by waymo_extract.py and argoverse_extract.py
    so the density iso-line knobs stay in lockstep across extractors."""
    p.add_argument("--contours", action="store_true",
                   help="DENSITY ISO-LINES view: instead of point/surfel tiers, draw "
                        "the iso-density contours of the returns (a live topographic "
                        "map of where the cloud clusters — walls / cars / vegetation), "
                        "morphing per playhead window. With --contour-z-step it goes "
                        "TRUE-3D (contour density per height layer, stack at real "
                        "altitude). Builds ONE windowed-LineString `lidar/` archive "
                        "(rendered by AnimatedPathLayer). Tune with the knobs below.")
    p.add_argument("--contour-decimate", type=int, default=1,
                   help="LIDAR decode/stride decimation for the contour pass "
                        "(default 1 = full density — the fine default grid needs it "
                        "so each small cell still sees enough returns).")
    p.add_argument("--contour-cell", type=float, default=0.25,
                   help="density grid cell size in metres (default 0.25 — the XY "
                        "resolution lever; smaller = finer horizontal detail).")
    p.add_argument("--contour-sigma", type=float, default=1.2,
                   help="gaussian smoothing sigma in CELLS before contouring "
                        "(default 1.2 — sharper than a wide blur so fine XY structure "
                        "survives).")
    p.add_argument("--contour-levels", type=int, default=5,
                   help="number of density contour levels / bands (default 5; max "
                        "useful is len(ISO_DENSITY_BANDS)).")
    p.add_argument("--contour-step", type=int, default=200,
                   help="playhead window step in ms — how often the contour set is "
                        "re-cut (default 200; also the tile bucket).")
    p.add_argument("--contour-width", type=int, default=900,
                   help="accumulation window in ms centred on each step (default 900).")
    p.add_argument("--contour-min-len", type=int, default=16,
                   help="drop contour loops shorter than this many vertices (default 16 "
                        "— culls fine-grid speckle).")
    p.add_argument("--contour-simplify", type=float, default=0.5,
                   help="Douglas-Peucker tolerance in CELLS applied to each contour "
                        "(default 0.5; 0 disables). Collapses marching-squares' "
                        "per-cell-crossing staircase to clean lines — ~3-5× fewer "
                        "vertices with no visible change, so a fine grid stays cheap.")
    p.add_argument("--contour-z-step", type=float, default=0.0,
                   help="TRUE-3D iso-lines: height-slab spacing in METRES. 0 (default) "
                        "= the flat 2D ground-density map. >0 slices returns into "
                        "height layers and tags each contour with its real altitude "
                        "so the client lifts it. Try 1.0.")
    p.add_argument("--contour-z-thickness", type=float, default=None,
                   help="3D: per-slab accumulation thickness in metres (default "
                        "1.5×z-step — overlapping slabs → smoother vertical continuity).")
    p.add_argument("--contour-z-min", type=float, default=None,
                   help="3D: lowest slab centre (metres; default = 1st z percentile).")
    p.add_argument("--contour-z-max", type=float, default=None,
                   help="3D: highest slab centre (metres; default = 99th z percentile).")
    p.add_argument("--contour-min-pts-layer", type=int, default=120,
                   help="3D: drop a window's height slab with fewer than this many "
                        "returns (default 120).")


def contour_kwargs(args) -> dict:
    """Collect the ``--contour-*`` knobs from parsed args into the keyword dict
    ``build_density_contours`` expects (the ``--contour-decimate`` stride is applied
    by the caller before contouring, so it's NOT included here)."""
    return dict(
        cell_m=args.contour_cell, sigma_cells=args.contour_sigma,
        n_levels=args.contour_levels, win_step_ms=args.contour_step,
        win_width_ms=args.contour_width, min_len=args.contour_min_len,
        simplify_cells=args.contour_simplify,
        z_step=args.contour_z_step, z_thickness=args.contour_z_thickness,
        z_min=args.contour_z_min, z_max=args.contour_z_max,
        min_pts_layer=args.contour_min_pts_layer,
    )


# ── scene-split ("stage + actors") shared CLI (AV2 + Waymo, no drift) ─────────
SCENE_SPLIT_VOXEL_M = 0.2     # default static-stage BASE voxel edge (m)
SCENE_STATIC_SPEED = 0.5      # m/s — a track at/above this is a moving "actor"
ERASOR_SCAN_STRIDE = 6        # decimate per-sweep scans kept for the ERASOR pass
# The STATIC stage is full-range (one bucket), so a low zoom packs the WHOLE scene
# into ONE tile (a 60 MB+ megatile on a compact corridor). Floor the stage zoom so
# those whole-scene low-zoom tiles aren't built; the cockpit views street-level
# (~z18), so z17+ is ample (one level of zoom-out from the default) and splits a
# compact corridor into ~2 tiles. The DYNAMIC actors stay at z14 (time-bucketed →
# small tiles). Bump to 18 for the densest scenes if a single tile is still heavy.
STAGE_MIN_ZOOM = 17


def add_scene_split_args(p) -> None:
    """Register the shared ``--scene-split`` + ``--stage-*`` + ``--erasor-*`` CLI
    flags so the AV2 and Waymo extractors expose IDENTICAL knobs (the dual-copy
    hazard this repo keeps hitting). Pair with ``scene_split_config(args)``."""
    import argparse

    p.add_argument("--scene-split", action="store_true", dest="scene_split",
                   help="SCENE-SPLIT view ('stage + actors'): decompose the log into "
                        "TWO surfel archives — a STATIC 'stage' (the fixed environment: "
                        "every sweep accumulated + the moving returns removed via in-box "
                        "+ ERASOR scrub, curvature-adaptively compacted + photo-graded, "
                        "baked once as a timeless backdrop) and DYNAMIC 'actors' (the "
                        "moving returns, animated per-sweep). Implies camera colour; "
                        "mutually exclusive with --surfel / --worldbuild / --contours.")
    p.add_argument("--stage-voxel", type=float, default=SCENE_SPLIT_VOXEL_M,
                   dest="stage_voxel",
                   help=f"static-stage BASE voxel (m) = detail kept in COMPLEX regions "
                        f"(default {SCENE_SPLIT_VOXEL_M}).")
    p.add_argument("--stage-min-zoom", type=int, default=STAGE_MIN_ZOOM,
                   dest="stage_min_zoom",
                   help="min zoom for the STATIC stage archive (full-range, so a low "
                        "zoom packs the WHOLE scene into one giant tile). Higher = no "
                        "whole-scene megatile + smaller bytes, but the stage stops "
                        f"rendering when zoomed out past it (default {STAGE_MIN_ZOOM}; "
                        "the cockpit is street-level ~z18).")
    p.add_argument("--scene-static-speed", type=float, default=SCENE_STATIC_SPEED,
                   dest="scene_static_speed",
                   help=f"a track ≥ this speed (m/s) is a moving 'actor' "
                        f"(default {SCENE_STATIC_SPEED}).")
    # ERASOR residual scrub
    p.add_argument("--erasor", action=argparse.BooleanOptionalAction, default=True,
                   help="ERASOR-style residual scrub of ghost trails off the stage "
                        "(--no-erasor to disable; default on).")
    p.add_argument("--erasor-rings", type=int, default=ERASOR_RINGS, dest="erasor_rings")
    p.add_argument("--erasor-sectors", type=int, default=ERASOR_SECTORS, dest="erasor_sectors")
    p.add_argument("--erasor-max-r", type=float, default=ERASOR_MAX_R, dest="erasor_max_r")
    p.add_argument("--erasor-height-diff", type=float, default=ERASOR_HEIGHT_DIFF,
                   dest="erasor_height_diff")
    p.add_argument("--erasor-occ-ratio", type=float, default=ERASOR_OCC_RATIO,
                   dest="erasor_occ_ratio")
    p.add_argument("--erasor-ground-margin", type=float, default=ERASOR_GROUND_MARGIN,
                   dest="erasor_ground_margin")
    p.add_argument("--erasor-min-scans", type=int, default=ERASOR_MIN_SCANS,
                   dest="erasor_min_scans")
    # curvature-adaptive stage compaction
    p.add_argument("--adaptive-stage", action=argparse.BooleanOptionalAction,
                   default=True, dest="adaptive_stage",
                   help="curvature-adaptive stage compaction (flats→big surfels, "
                        "complex→fine); --no-adaptive-stage for a uniform grid.")
    p.add_argument("--stage-max-voxel", type=float, default=STAGE_MAX_VOXEL,
                   dest="stage_max_voxel",
                   help=f"voxel (m) for the FLATTEST adaptive tier (default {STAGE_MAX_VOXEL}).")
    p.add_argument("--stage-adapt-levels", type=int, default=STAGE_ADAPT_LEVELS,
                   dest="stage_adapt_levels")
    p.add_argument("--stage-flat-pct", type=float, default=50.0, dest="stage_flat_pct",
                   help="σ pct treated as fully flat → max voxel (higher = more compact; "
                        "default 50).")
    p.add_argument("--stage-complex-pct", type=float, default=95.0,
                   dest="stage_complex_pct")
    # photographic colour grade
    p.add_argument("--stage-saturation", type=float, default=1.6, dest="stage_saturation",
                   help="saturation multiplier for the colour grade (default 1.6).")
    p.add_argument("--stage-exposure", type=float, default=1.25, dest="stage_exposure",
                   help="exposure (linear gain) for the colour grade (default 1.25).")
    p.add_argument("--stage-gamma", type=float, default=1.0, dest="stage_gamma")
    p.add_argument("--stage-white-balance", action=argparse.BooleanOptionalAction,
                   default=True, dest="stage_white_balance",
                   help="gray-world white balance (neutralise blue cast; default on).")
    # actor noise reduction (rain/sensor scatter) — OPT-IN: most scenes keep every
    # moving return (clear agents); only noisy scenes (e.g. the rain highway) need it.
    p.add_argument("--actor-denoise", action=argparse.BooleanOptionalAction,
                   default=False, dest="actor_denoise",
                   help="per-sweep statistical outlier removal on the actors: drop "
                        "isolated rain/sensor scatter, keep the dense moving agents. "
                        "OFF by default (only enable on noisy scenes — most scenes "
                        "want every moving return).")
    p.add_argument("--actor-denoise-k", type=int, default=ACTOR_SOR_K,
                   dest="actor_denoise_k",
                   help=f"k-NN for the outlier test (default {ACTOR_SOR_K}).")
    p.add_argument("--actor-denoise-std", type=float, default=ACTOR_SOR_STD,
                   dest="actor_denoise_std",
                   help="drop returns with mean-kNN dist > mean + this·std (LOWER = "
                        f"more aggressive; default {ACTOR_SOR_STD}).")
    p.add_argument("--actor-voxel", type=float, default=0.0, dest="actor_voxel",
                   help="optional per-sweep voxel cap (m) on the surviving actor "
                        "density (0 = off; e.g. 0.1 to trim redundant close hits).")
    # actor colour grade — punchier than the stage so the moving agents POP
    p.add_argument("--actor-saturation", type=float, default=2.2, dest="actor_saturation",
                   help="saturation for the ACTOR grade — higher than the stage so "
                        "moving agents pop (default 2.2).")
    p.add_argument("--actor-exposure", type=float, default=1.45, dest="actor_exposure",
                   help="exposure (linear gain) for the ACTOR grade (default 1.45).")


def scene_split_config(args) -> dict:
    """Build the scene-split config dict (voxel / erasor / adaptive / grade) from
    parsed args — consumed by both extractors' scene-split path."""
    return dict(
        voxel_m=args.stage_voxel,
        static_speed=args.scene_static_speed,
        erasor=args.erasor,
        erasor_params=dict(
            rings=args.erasor_rings, sectors=args.erasor_sectors,
            max_r=args.erasor_max_r, height_diff=args.erasor_height_diff,
            occ_ratio=args.erasor_occ_ratio, ground_margin=args.erasor_ground_margin,
            min_scans=args.erasor_min_scans),
        adaptive=(dict(base_voxel=args.stage_voxel, max_voxel=args.stage_max_voxel,
                       levels=args.stage_adapt_levels, flat_pct=args.stage_flat_pct,
                       complex_pct=args.stage_complex_pct)
                  if args.adaptive_stage else None),
        grade=dict(saturation=args.stage_saturation, exposure=args.stage_exposure,
                   gamma=args.stage_gamma, white_balance=args.stage_white_balance),
        # Principled actor reduction — SOR (noise) and voxel (overplot) are
        # INDEPENDENT: build the config if EITHER is requested.
        denoise=(dict(sor=args.actor_denoise, sor_k=args.actor_denoise_k,
                      sor_std=args.actor_denoise_std, voxel=args.actor_voxel)
                 if (args.actor_denoise or args.actor_voxel > 0) else None),
        # Punchier grade for the actors than the stage so the moving agents POP.
        actor_grade=dict(saturation=args.actor_saturation, exposure=args.actor_exposure,
                         gamma=args.stage_gamma, white_balance=args.stage_white_balance),
    )


# ── Sidecar JSON writers ─────────────────────────────────────────────────────
def write_telemetry_json(
    out_path: Path,
    *,
    t0: int,
    hz: float,
    fields: Mapping[str, Mapping],
) -> None:
    """Write ``telemetry.json`` (av-cockpit.md §2d).

    ``fields`` maps a field id (e.g. ``"speed"``) → ``{"unit", "label",
    "samples"}`` where ``samples`` is a list of ``[t_ms, value]`` pairs. Samples
    are sorted by ``t_ms`` here so the cockpit can binary-search the playhead.
    """
    out_fields: dict[str, dict] = {}
    for name, spec in fields.items():
        samples = sorted(
            ([int(t), float(v)] for t, v in spec["samples"]),
            key=lambda s: s[0],
        )
        out_fields[name] = {
            "unit": spec.get("unit", ""),
            "label": spec.get("label", name),
            "samples": samples,
        }
    doc = {"t0": int(t0), "hz": float(hz), "fields": out_fields}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc))
    n = sum(len(f["samples"]) for f in out_fields.values())
    print(f"  wrote telemetry.json ({len(out_fields)} field(s), {n} sample(s)) → {out_path}")


def write_cameras_json(
    out_path: Path,
    *,
    camera: str,
    frames: Iterable[Mapping],
) -> None:
    """Write ``cameras.json`` (av-cockpit.md §2e).

    ``frames`` is an iterable of ``{"t": t_ms, "url": "cam/0001.jpg"}`` (url
    relative to the scene dir). Frames are sorted by ``t``; an empty list emits a
    valid sidecar with no frames (cockpit simply hides the inset).
    """
    out_frames = sorted(
        ({"t": int(f["t"]), "url": str(f["url"])} for f in frames),
        key=lambda f: f["t"],
    )
    doc = {"camera": camera, "frames": out_frames}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc))
    print(f"  wrote cameras.json ({len(out_frames)} frame(s)) → {out_path}")


def write_scene_json(
    out_path: Path,
    *,
    id: str,
    name: str,
    dataset: str,
    dataset_url: str,
    license: str,
    location: str,
    description: str,
    origin_lat: float,
    origin_lon: float,
    time_range: tuple[int, int],
    initial_view: Mapping,
    object_colors: Mapping[str, Sequence[int]],
    streams: Mapping[str, Mapping],
    lidar_colors: Mapping[str, Sequence[int]] | None = None,
) -> None:
    """Write ``scene.json`` (av-cockpit.md §2f) — the cockpit source of truth.

    ``streams`` is the present-streams map (e.g. ``{"lidar": {"url":
    "lidar/manifest.json", "points": 280000}, "ego": {"url":
    "ego/manifest.json"}, ...}``); only present streams are listed.

    ``lidar_colors`` (optional) is the semantic-class → RGBA palette baked in
    when the LIDAR cloud is colored by ``seg_class`` (nuScenes-lidarseg); it
    mirrors ``object_colors`` so a cockpit legend can read it from scene.json.
    Omitted (height-band coloring) when ``None``.
    """
    doc = {
        "id": id,
        "name": name,
        "dataset": dataset,
        "datasetUrl": dataset_url,
        "license": license,
        "location": location,
        "description": description,
        "georef": {"originLat": origin_lat, "originLon": origin_lon},
        "timeRange": {"start": int(time_range[0]), "end": int(time_range[1])},
        "initialView": dict(initial_view),
        "objectColors": {k: list(v) for k, v in object_colors.items()},
        "streams": {k: dict(v) for k, v in streams.items()},
    }
    if lidar_colors is not None:
        doc["lidarColors"] = {k: list(v) for k, v in lidar_colors.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"  wrote scene.json ({len(doc['streams'])} stream(s)) → {out_path}")


# ── stt-build ────────────────────────────────────────────────────────────────
def lidar_quantize_attrs(z_prec: float) -> dict[str, float]:
    """Canonical ``--quantize-attr`` map for EVERY LiDAR archive / tier / variant.

    Each entry stores a near-incompressible Float64 column as fixed-point ints
    (``UInt16`` leaf + a per-column affine the reader reconstructs as
    ``value = offset + q·step``); a raw Float64 barely compresses. Keeping ONE
    canonical map here (rather than a literal per extractor) is deliberate — the
    surfel/colour columns drifting out of sync across waymo/nuscenes/argoverse is
    exactly the dual-copy hazard this repo keeps hitting.

    Safe to pass WHOLE to any LiDAR build: stt-build looks each name up per
    column, so entries with no matching column (``r``/``g``/``b`` on an uncoloured
    cloud, the surfel columns on a raw cloud) are silently ignored, and a range
    that overflows ``u16`` degrades to ``Int32`` rather than erroring.

    * ``z`` — elevation (m) → ``z_prec`` cm grid (same precision as the coords).
    * ``r``/``g``/``b`` — camera-projected per-point colour, integers 0–255 stored
      as Float64 (see ``write_lidar_points``); ``step=1`` is lossless → ``u16``
      (256 distinct values compress trivially), 8 B/pt → 2 B/pt each.
    * ``qx``/``qy``/``qz``/``qw`` — unit surfel quaternion ∈ [-1,1]; ``1e-3`` is
      ~0.1° of orientation error (still imperceptible on a disk) and the quat is
      ~46 % of a quantized surfel tile + near-incompressible (high-entropy
      orientations), so the coarser grid is a measured ~7.5 % whole-tile win with
      no visible change (point_column_stats: 16.1 → 14.9 B/pt). Fits ``u16``
      (range 2 / 1e-3 = 2 000). 8 B/pt → 2 B/pt each.
    * ``s_major``/``s_minor`` — in-plane extents (m) at cm precision.
    * ``surfel_opacity`` — confidence ∈ [0,1] at ~1/250 precision.
    """
    return {
        "z": z_prec,
        "r": 1.0,
        "g": 1.0,
        "b": 1.0,
        "qx": 1e-3,
        "qy": 1e-3,
        "qz": 1e-3,
        "qw": 1e-3,
        # Smallest-three packed quaternion (write_lidar_points pack_quat=True):
        # three components ∈ [-0.71, 0.71] at the same ~0.1° grid; q_imax is the
        # 0–3 index of the dropped (largest) component, step 1 → stored EXACTLY.
        "q_a": 1e-3,
        "q_b": 1e-3,
        "q_c": 1e-3,
        "q_imax": 1.0,
        "s_major": 0.01,
        "s_minor": 0.01,
        "surfel_opacity": 0.004,
        # Feature 1 (Worldbuild): the static/dynamic numeric flag is exactly 0/1,
        # so step 1 is lossless → a single-distinct-value (or two) u16 that
        # compresses to nothing.
        "is_dynamic": 1.0,
        # Feature 5 (Sweep): scan phase ∈ [0,1]; ~1/1000 precision is finer than
        # the beam reveal needs, fits u16 (range 1 / 1e-3 = 1000), 8 B/pt → 2 B/pt.
        "scan_phase": 1e-3,
        # Additive-octree LOD (lod_home_zoom): the integer home zoom (~14-19),
        # step 1 lossless → a handful of distinct u16 values that compress to
        # nothing. Read on the client only for optional fractional-zoom smoothing.
        "home_zoom": 1.0,
    }


def lidar_vector_groups(
    *, surfel: bool, colored: bool = True
) -> list[tuple[str, list[str], str]]:
    """Canonical ``--vector-group`` list for LiDAR archives (see ``run_stt_build``).

    Fuses the per-point scalar columns the extractor writes into ONE interleaved
    ``FixedSizeList`` column each, so the client binds the contiguous buffer
    straight to a deck.gl instanced attribute with ZERO per-point re-pack on the
    render thread (the stutter fix). The encoder skips a group whose source
    columns are absent, but we still gate here so a coloured-points archive gets
    ``point_rgba`` while a surfel archive gets ``surfel_rgba`` (same r,g,b,a
    sources, different consumer column name) — passing both would let the first
    consume the columns and starve the second.

    * surfel → ``surfel_quat`` (qx,qy,qz,qw, f32) + ``surfel_scale``
      (s_major,s_minor, f32), read by ``SplatLayer``.
    * coloured → ``surfel_rgba`` (surfel) or ``point_rgba`` (plain points), the
      ``[r,g,b,a]`` u8 colour (alpha = baked confidence for surfels, else 255),
      read by ``SplatLayer`` / ``AnimatedPointLayer`` respectively.

    These columns are written as raw f32/u8 vectors (NOT quantized) — the
    ``lidar_quantize_attrs`` entries for the grouped scalars become inert because
    the encoder fuses them before the quantize pass.
    """
    groups: list[tuple[str, list[str], str]] = []
    if surfel:
        groups.append(("surfel_quat", ["qx", "qy", "qz", "qw"], "f32"))
        groups.append(("surfel_scale", ["s_major", "s_minor"], "f32"))
    if colored:
        color_name = "surfel_rgba" if surfel else "point_rgba"
        groups.append((color_name, ["r", "g", "b", "a"], "u8"))
    return groups


def pack_surfel_quaternions(quat):
    """Smallest-three encode unit quaternions ``(N,4) [x,y,z,w]`` →
    ``(q_a, q_b, q_c, q_imax)``: the three non-largest components (in ascending
    component order) + the index ``0..3`` of the largest. The whole quaternion is
    sign-flipped so the largest component is POSITIVE (``q ≡ −q`` as a rotation),
    so the shader reconstructs it unambiguously as ``+sqrt(1 − a² − b² − c²)``.
    Vectorised; mirrored by ``unpackQuat`` in ``splat-primitive-layer.ts``.
    """
    q = np.asarray(quat, dtype="float64")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    n = len(q)
    imax = np.argmax(np.abs(q), axis=1)
    sign = np.sign(q[np.arange(n), imax])
    sign[sign == 0] = 1.0
    q = q * sign[:, None]
    keep = np.ones((n, 4), dtype=bool)
    keep[np.arange(n), imax] = False
    abc = q[keep].reshape(n, 3)  # boolean-index preserves ascending column order
    return abc[:, 0], abc[:, 1], abc[:, 2], imax.astype("int64")


# ── geometry-aware decimation ────────────────────────────────────────────────
# The crude lever is a uniform stride (``pts[::decimate]``): it spends its point
# budget EQUALLY on flat road/walls (where a neighbour predicts the point — it's
# redundant) and on curbs/poles/foliage/vehicle outlines (where it's the detail
# that makes the scene read). A measured bake-off (``lidar_summarize_eval.py``,
# point-to-plane RMS on real Waymo sweeps) showed the same point budget keeps
# ~2-4× MORE edge detail when you instead keep every high-curvature return and
# voxel-summarise the flat majority — at equal-or-better surface error and no
# catastrophic holes. So at a given fidelity this sends far fewer points.
ADAPTIVE_K = 12          # k-NN for the local surface-variation estimate (== SURFEL_K)
ADAPTIVE_BETA = 0.5      # share of the budget reserved for guaranteed-detail returns


def _surface_variation(pts, k: int = ADAPTIVE_K):
    """σ = λ0/(λ0+λ1+λ2) per point — local planarity from the k-NN covariance.
    ~0 on flats, large on edges/corners/foliage. Same eigenframe ``compute_surfels``
    fits; kept standalone so the plain (non-surfel) path can reuse it."""
    import numpy as np
    from scipy.spatial import cKDTree

    k = min(k, len(pts) - 1)
    _d, idx = cKDTree(pts).query(pts, k=k + 1)
    neigh = pts[idx]
    c = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / (k + 1)
    evals = np.linalg.eigvalsh(cov)            # ascending
    return evals[:, 0] / (evals.sum(axis=1) + 1e-12)


def _voxel_real_indices(pts, v: float):
    """Index of the REAL point nearest each occupied ``v``-metre voxel's centroid."""
    import numpy as np

    keys = np.floor(pts / v).astype(np.int64)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, pts)
    cent = sums / counts[:, None]
    d2 = np.einsum("ni,ni->n", pts - cent[inv], pts - cent[inv])
    best = np.full(len(uniq), np.inf)
    np.minimum.at(best, inv, d2)
    return np.flatnonzero(d2 == best[inv])


def _voxel_size_for_count(pts, target: int) -> float:
    """Bisect the voxel edge so the occupied-voxel count ≈ ``target``."""
    import numpy as np

    lo, hi = 0.02, 8.0
    for _ in range(26):
        v = (lo * hi) ** 0.5
        n = len(np.unique(np.floor(pts / v).astype(np.int64), axis=0))
        if n > target:
            lo = v
        else:
            hi = v
    return (lo * hi) ** 0.5


def adaptive_lidar_select(pts, decimate: int, k: int = ADAPTIVE_K,
                          beta: float = ADAPTIVE_BETA):
    """Geometry-aware replacement for ``np.arange(0, len(pts), decimate)``.

    ``pts`` is the FULL-density vehicle/ego-frame sweep (N×3). Returns a SORTED
    int index array of ≈ N/``decimate`` REAL returns — keep ``beta`` of the budget
    as the highest-curvature points (poles/edges/objects, preserved exactly) and
    the rest as one representative real point per voxel over the flat remainder
    (even coverage, no synthetic points → all attributes ride along). Drop-in for
    the surfel and plain lidar paths; downstream colorize/quantize/tiling unchanged.
    """
    import numpy as np

    n = len(pts)
    target = max(1, n // max(1, decimate))
    if target >= n:
        return np.arange(n)
    sv = _surface_variation(pts, k)
    order = np.argsort(sv)[::-1]
    ne = min(int(beta * target), n)
    edge_idx = order[:ne]
    flat_mask = np.ones(n, bool)
    flat_mask[edge_idx] = False
    flat_pos = np.flatnonzero(flat_mask)
    want_flat = max(1, target - ne)
    if want_flat >= len(flat_pos):
        keep = np.arange(n)
    else:
        v = _voxel_size_for_count(pts[flat_pos], want_flat)
        rep_local = _voxel_real_indices(pts[flat_pos], v)
        keep = np.concatenate([edge_idx, flat_pos[rep_local]])
    return np.unique(keep)          # sorted + dedups any voxel-centroid ties


# ── curvature-adaptive stage compaction (variable cell size) ─────────────────
# The static "stage" is the fixed environment accumulated over every sweep; a flat
# road/wall is the BULK of the returns yet the LEAST information (a neighbour
# predicts it). A uniform voxel grid spends the same budget on a flat road cell as
# on a foliage/pole/curb cell. adaptive_stage_select instead gives each region a
# voxel size that tracks its surface complexity: flats collapse into a FEW big
# cells (→ big surfels, since _surfel_frame sizes each disk from the local spacing,
# so a sparse flat region auto-grows its disks to fill the gap), while complex
# regions keep the fine base voxel (→ many small crisp surfels). Same idea as
# octree / curvature-adaptive downsampling (PCL, AVS-Net) — a much more COMPACT
# stage at equal-or-better perceived coverage. Tiers are by σ PERCENTILE so there
# are no scene-specific magic thresholds.
STAGE_MAX_VOXEL = 0.5    # m — voxel edge for the FLATTEST tier (the big-disk cells)
STAGE_ADAPT_LEVELS = 6   # complexity tiers between base_voxel and max_voxel


def adaptive_stage_select(pts, *, base_voxel, max_voxel=STAGE_MAX_VOXEL,
                          levels=STAGE_ADAPT_LEVELS, flat_pct=50.0,
                          complex_pct=95.0, k=ADAPTIVE_K):
    """Curvature-adaptive multi-resolution voxel dedup for the static stage.

    ``pts`` (M×3, metric) is the accumulated stage cloud (already deduped at
    ``base_voxel``). Each point's local surface variation σ = λ0/Σλ is mapped to a
    voxel edge by VALUE (not equal-population — so a uniformly-flat road all lands
    in ONE coarse tier rather than being split across fine tiers): σ ≤ the
    ``flat_pct``-th percentile → ``max_voxel`` (flats → a few big cells), σ ≥ the
    ``complex_pct``-th percentile → ``base_voxel`` (poles/foliage/edges keep full
    detail), graded geometrically between over ``levels`` tiers. Percentile anchors
    keep it scene-adaptive + outlier-robust. Returns a SORTED int index array into
    ``pts``. Downstream ``_fit_merged_surfels`` sizes each disk from the resulting
    local spacing, so flat cells get big gap-filling disks and complex cells stay
    crisp — no explicit per-point radius needed.
    """
    import numpy as np

    m = len(pts)
    if m <= levels or max_voxel <= base_voxel:
        return np.arange(m)
    sv = _surface_variation(pts, k)
    lo, hi = np.percentile(sv, [flat_pct, complex_pct])
    if hi <= lo:
        norm = (sv > lo).astype("float64")
    else:
        norm = np.clip((sv - lo) / (hi - lo), 0.0, 1.0)
    # tier 0 = flattest (σ ≤ lo) → max_voxel … tier levels-1 = complex → base_voxel.
    tier = np.minimum((norm * levels).astype(np.int64), levels - 1)
    # Geometric voxel edge per tier: tier 0 (flat) → max_voxel, last → base_voxel.
    sizes = max_voxel * (base_voxel / max_voxel) ** (np.arange(levels) / (levels - 1))
    keep = []
    for t in range(levels):
        idx = np.flatnonzero(tier == t)
        if idx.size == 0:
            continue
        v = float(sizes[t])
        if v <= base_voxel * 1.001:
            keep.append(idx)                       # finest tier: already at base
        else:
            keep.append(idx[_voxel_real_indices(pts[idx], v)])
    return np.unique(np.concatenate(keep))


# ── additive-octree LOD home-zoom assignment ─────────────────────────────────
# The 5 fixed density tiers (LIDAR_DENSITY_TIERS) each ship a COMPLETE strided
# copy of the cloud (small ⊂ … ⊂ full), so the scene stores ~2× its points and
# the cockpit picks one tier at runtime — there is no true zoom-LOD. This is the
# additive-octree (COPC / Entwine / Potree-style) alternative: assign every
# return a SINGLE "home zoom" via a per-sweep hierarchical voxel subsample, emit
# each point ONCE (stt-build --min-zoom-field=--max-zoom-field=home_zoom places
# it in exactly that zoom's tiles), and let the client load the UNION of zoom
# levels minZoom..cameraZoom. Coarse levels are spatially-uniform sparse
# overviews (few big-area tiles, already resident from when the camera was
# zoomed out); zooming in fetches ONLY the deeper residual. Lossless: the union
# at maxZoom is the complete cloud, so it honours the no-thinning principle —
# full fidelity at depth, progressively revealed — while storing each point once
# (~½ the bytes of the 5-tier pyramid).
LOD_MIN_ZOOM = 14        # coarsest LOD level (== LIDAR_MIN_ZOOM floor; below = waste)
# Finest level: the un-claimed residual (FULL density) lands here. Deep enough
# that the voxel ladder steps down to ~cm spacing before the residual (so the
# dense returns spread across levels rather than dumping into one), but no deeper:
# past z21 the per-level gain is negligible at this px_per_point, so z21 IS the
# full quantized cloud (its voxel ≈ cm-scale < the 5 cm coord quantize).
LOD_MAX_ZOOM = 21
# Target return spacing (screen px) at a point's home zoom. SMALLER = denser at
# every zoom (richer zoomed-out overview + "good detail" arrives at a shallower
# zoom). Tuned from a measured cumulative-points-vs-zoom sweep on real Miami:
# 0.4 puts ~1.7M points in the z16 overview (≈ the old "medium" manual tier) and
# ~full detail by ~z19-20, while keeping a smooth multi-level reveal to z21.
LOD_PX_PER_POINT = 0.4
# ── geometry-aware (curvature) home-zoom knobs ───────────────────────────────
# A UNIFORM voxel grid spends the same one-point-per-cell budget on a flat road
# cell (which a single sample conveys) as on a pole / curb / car-edge / foliage
# cell (which needs MANY points to read as a shape) — so the sparse overview is a
# featureless blur. The geometry-aware path instead grades each level's voxel by
# local surface variation σ = λ0/Σλ (the same k-NN-covariance planarity the surfel
# fit + adaptive_stage_select use): FLAT regions get a coarser voxel (few points,
# deferred to finer zooms), HIGH-CURVATURE regions keep the base voxel (kept dense
# at coarse zoom). The flat penalty is strongest at the coarsest zoom and fades to
# 1.0 by the finest pre-residual level, so flats still fill in before the residual
# (no lopsided dump). Same idea the repo measured at ~2-4× more edge detail per
# point (lidar_summarize_eval.py); here it makes the zoomed-out overview read as
# real structure — "preserve the geometry with the fewest points."
LOD_CURV = True          # default ON; pass curvature=False (--lod-uniform) to compare
LOD_CURV_K = 12          # k-NN for the σ estimate (== SURFEL_K / ADAPTIVE_K)
LOD_CURV_LEVELS = 5      # σ tiers between flat and complex
LOD_FLAT_RATIO = 3.0     # flat-region voxel is up to this× coarser than complex (at coarse zoom)
LOD_FLAT_PCT = 50.0      # σ ≤ this percentile → flattest tier (max penalty)
LOD_COMPLEX_PCT = 92.0   # σ ≥ this percentile → complex tier (no penalty)


def _lod_curv_reps(p, tier, base_v, levels, flat_ratio, lvl_frac):
    """Curvature-graded voxel reps for ONE octree level over remaining points.

    ``tier`` (0=flattest … levels-1=most complex) grades each point's voxel: the
    flat tier is voxelized at ``base_v · penalty`` (penalty up to ``flat_ratio`` at
    coarse zoom, → 1 at the finest level via ``lvl_frac`` ∈ [1,0]), the complex tier
    always at ``base_v``. Returns local indices (into ``p``) of the claimed reps.
    """
    keep = []
    span = max(levels - 1, 1)
    for tval in range(levels):
        idx = np.flatnonzero(tier == tval)
        if idx.size == 0:
            continue
        # tval 0 (flat) → full penalty; tval levels-1 (complex) → none. lvl_frac
        # fades the whole penalty to 0 at the finest pre-residual level.
        penalty = 1.0 + (flat_ratio - 1.0) * (1.0 - tval / span) * lvl_frac
        reps = _voxel_real_indices(p[idx], base_v * penalty)
        keep.append(idx[reps])
    return np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)


def lod_home_zoom(lon, lat, z, timestamp, *, min_zoom=LOD_MIN_ZOOM,
                  max_zoom=LOD_MAX_ZOOM, px_per_point=LOD_PX_PER_POINT,
                  curvature=LOD_CURV, curvature_levels=LOD_CURV_LEVELS,
                  flat_ratio=LOD_FLAT_RATIO, flat_pct=LOD_FLAT_PCT,
                  complex_pct=LOD_COMPLEX_PCT, k=LOD_CURV_K, anchor_lat=None):
    """Assign each LiDAR return a single additive-octree "home zoom".

    ``lon`` / ``lat`` (deg) / ``z`` (m) / ``timestamp`` (per-sweep ms) are the
    FULL-density cloud. Returns an ``Int64`` array (one home zoom per return) for
    the ``home_zoom`` column ``write_lidar_points`` writes and stt-build consumes
    as ``--min-zoom-field=--max-zoom-field=home_zoom``.

    Per SWEEP (grouped by ``timestamp`` — the cockpit time-windows ~one sweep, so
    the LOD must densify the LIVE sweep, not the accumulated cloud), points are
    claimed coarse→fine: at each zoom ``zz`` the still-unclaimed points are
    voxelized at the Web-Mercator ground resolution for ``zz`` (≈ one return per
    ``px_per_point`` screen pixels) and one real representative per occupied voxel
    is claimed for that level. Everything still unclaimed at ``max_zoom`` lands
    there (the full-density residual). Each point is materialized at EXACTLY one
    zoom; the union over min..max reconstructs the whole sweep.

    Voxel sizing is tied to the tile pyramid: ``voxel(zz) = px_per_point ·
    156543.03·cos(lat) / 2^zz`` metres, so a point's home-zoom spacing matches its
    on-screen pixel spacing at that zoom (the EPT/octree screen-space-error idea).

    ``curvature`` (default on) makes the per-level voxel GEOMETRY-AWARE: σ surface
    variation is computed once per sweep and grades each cell's voxel so flat
    regions are deferred while edges/poles/curbs are kept at coarse zoom (see the
    LOD_CURV_* knobs / ``_lod_curv_reps``). ``curvature=False`` is the plain
    uniform-voxel grid (for A/B comparison). Deterministic either way (no RNG).
    """
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    z = np.asarray(z, dtype="float64")
    t = np.asarray(timestamp)
    n = len(lon)
    home = np.full(n, int(max_zoom), dtype="int64")
    if n == 0:
        return home
    lat0 = float(np.median(lat)) if anchor_lat is None else float(anchor_lat)
    # Local metric frame for voxelisation — only RELATIVE spacing matters, so an
    # absolute deg→m scaling (exact enough over a city-block scene) is fine.
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    x_m = lon * m_per_deg_lon
    y_m = lat * 110540.0
    pts = np.column_stack([x_m, y_m, z])
    levels = int(curvature_levels)
    span_zoom = max(int(max_zoom) - 1 - int(min_zoom), 1)

    def _vox(zz):
        # Web-Mercator ground resolution (m/px) at this zoom & latitude × spacing.
        return px_per_point * 156543.03392 * np.cos(np.radians(lat0)) / (2.0 ** zz)

    for ti in np.unique(t):
        sweep = np.flatnonzero(t == ti)
        p = pts[sweep]
        # Per-sweep σ tier (computed ONCE, reused across all levels).
        tier = None
        if curvature and len(sweep) > k + 2:
            sv = _surface_variation(p, k)
            lo, hi = np.percentile(sv, [flat_pct, complex_pct])
            if hi <= lo:
                norm = (sv > lo).astype("float64")
            else:
                norm = np.clip((sv - lo) / (hi - lo), 0.0, 1.0)
            # 0 = flattest (σ ≤ lo) … levels-1 = most complex (σ ≥ hi).
            tier = np.minimum((norm * levels).astype(np.int64), levels - 1)
        remaining = np.arange(len(sweep))  # local indices into p / sweep
        for zz in range(int(min_zoom), int(max_zoom)):
            if remaining.size == 0:
                break
            base_v = _vox(zz)
            if tier is None:
                reps = _voxel_real_indices(p[remaining], base_v)
            else:
                # 1 at the coarsest zoom → 0 at the finest pre-residual level, so
                # the flat penalty fades out and flats fill in before the residual.
                lvl_frac = (int(max_zoom) - 1 - zz) / span_zoom
                reps = _lod_curv_reps(p[remaining], tier[remaining], base_v,
                                      levels, flat_ratio, lvl_frac)
            claimed = sweep[remaining[reps]]
            home[claimed] = zz
            mask = np.ones(remaining.size, dtype=bool)
            mask[reps] = False
            remaining = remaining[mask]
        # whatever is left for this sweep keeps the max_zoom residual default
    return home


# ── actor denoise (rain / sensor scatter removal) ────────────────────────────
# The DYNAMIC actors are kept per-sweep (un-deduped) so motion animates, but in a
# rainy/dusty scene that cloud is full of NOISE: rain returns + spurious hits that
# fall inside the moving boxes. A crude stride would thin the real agents as much
# as the haze. denoise_actors is a PRINCIPLED reduction targeting the noise: within
# each sweep a moving object's returns form a DENSE cluster while scatter is
# ISOLATED, so Statistical Outlier Removal (mean distance to the k nearest
# neighbours) drops the isolated returns and the agents stay crisp.
ACTOR_SOR_K = 8           # k-NN for the statistical-outlier test
ACTOR_SOR_STD = 1.0       # drop returns whose mean-kNN dist > mean + this·std


def denoise_actors(xy, z, t, *, sor=True, sor_k=ACTOR_SOR_K, sor_std=ACTOR_SOR_STD,
                   voxel=0.0):
    """Per-sweep actor reduction — two INDEPENDENT, principled passes.

    ``xy`` (N,2), ``z`` (N), ``t`` (N per-sweep ms). Grouped by sweep:
    * ``sor`` (Statistical Outlier Removal): drop returns whose mean distance to
      their ``sor_k`` nearest neighbours exceeds ``mean + sor_std·std`` — the
      ISOLATED rain/sensor scatter — keeping the dense moving-object clusters.
    * ``voxel`` (>0): cap the surviving density to one representative per voxel —
      removes only REDUNDANT returns hitting the same cell in one sweep (overplot),
      not the agent (use this on dense-traffic scenes that aren't noisy).
    Either may run alone (e.g. SF traffic: ``sor=False, voxel=0.1``; rain: both).
    Returns a boolean keep-mask over the concatenated cloud.
    """
    from scipy.spatial import cKDTree

    n = len(z)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    t = np.asarray(t).reshape(-1)
    xy = np.asarray(xy, dtype="float64").reshape(-1, 2)
    z = np.asarray(z, dtype="float64").reshape(-1)
    for ti in np.unique(t):
        idx = np.flatnonzero(t == ti)
        pts = np.column_stack([xy[idx, 0], xy[idx, 1], z[idx]])
        if sor and len(idx) >= sor_k + 2:
            d, _ = cKDTree(pts).query(pts, k=sor_k + 1)
            md = d[:, 1:].mean(axis=1)        # mean dist to k NN (excl self)
            ok = md <= (md.mean() + sor_std * md.std())
        else:
            ok = np.ones(len(idx), dtype=bool)  # SOR off (or too few) → keep all
        if voxel > 0 and ok.any():
            sel = np.flatnonzero(ok)[_voxel_real_indices(pts[ok], voxel)]
            keep[idx[sel]] = True
        else:
            keep[idx[ok]] = True
    return keep


# ── shared surfel frame (eigen → quaternion core) ────────────────────────────
# The eigen→quaternion core of the per-sweep ``compute_surfels`` (in
# waymo_extract / argoverse_extract) factored out so the per-sweep paths AND the
# new merged Worldbuild path fit surfels the SAME way (the dual-copy hazard this
# repo keeps hitting). Tuning constants mirror the extractors' SURFEL_* values.
SURFEL_K = 12               # k-NN for the covariance estimate (== ADAPTIVE_K)
SURFEL_FILL = 0.70          # disk radius as a fraction of the local point spacing
SURFEL_S_MIN = 0.03         # m — floor (avoid sub-cm slivers)
SURFEL_S_MAX = 0.45         # m — ceil (sparse returns don't become giant blobs)
SURFEL_ASPECT_FLOOR = 0.45  # keep disks from collapsing to needles on edges


def mat3_to_quat(R: np.ndarray) -> np.ndarray:
    """Batched rotation matrices ``(M,3,3)`` → unit quaternions ``(M,4)`` [x,y,z,w].

    Shepperd's method (the numerically-stable four-case branch on which diagonal
    term is largest), vectorised. Same implementation the per-sweep extractors
    carry; kept here so the merged Worldbuild path shares it.
    """
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    trace = m00 + m11 + m22

    def _sqrt(v):
        return np.sqrt(np.maximum(v, 1e-12))

    s0 = _sqrt(trace + 1.0) * 2.0  # = 4w
    q0 = np.stack([(m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0, 0.25 * s0], 1)
    s1 = _sqrt(1.0 + m00 - m11 - m22) * 2.0  # = 4x
    q1 = np.stack([0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1, (m21 - m12) / s1], 1)
    s2 = _sqrt(1.0 + m11 - m00 - m22) * 2.0  # = 4y
    q2 = np.stack([(m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2, (m02 - m20) / s2], 1)
    s3 = _sqrt(1.0 + m22 - m00 - m11) * 2.0  # = 4z
    q3 = np.stack([(m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3, (m10 - m01) / s3], 1)

    use0 = trace > 0.0
    use1 = (~use0) & (m00 >= m11) & (m00 >= m22)
    use2 = (~use0) & (~use1) & (m11 >= m22)
    q = np.where(use0[:, None], q0,
        np.where(use1[:, None], q1,
        np.where(use2[:, None], q2, q3)))
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q


def _surfel_frame(pts: np.ndarray, sel: np.ndarray, view_dir, R_post=None):
    """Eigen → quaternion + extents + opacity core, shared by per-sweep + merged.

    ``pts`` is the full-density cloud (N×3), ``sel`` indexes the rendered subset.
    For each selected point: estimate the local surface frame from its k-NN
    covariance (smallest eigenvector = normal, largest = in-plane major axis),
    orient the normal toward ``view_dir`` (a single (3,) sensor→origin direction,
    or a per-point (M,3) array — e.g. the merged path uses centroid−point), build a
    right-handed ``[tmaj | tmin | normal]`` rotation, and convert to a quaternion.
    Disk extents scale to the DECIMATED spacing (fills the rendered gaps),
    anisotropy from the eigenvalues; confidence tracks planarity.

    ``R_post`` (optional 3×3): a left-multiplied rotation applied to every surfel
    frame BEFORE the quaternion conversion — the per-sweep extractors pass the
    ego-YAW matrix here (the render keeps vehicle-frame z, so only yaw is applied)
    so the shared core reproduces their existing output exactly. The merged
    Worldbuild path passes ``None`` (no single ego pose).

    Returns ``(quat M×4 [x,y,z,w], scale2 M×2 [s_major,s_minor], opacity M)``.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype="float64")
    sel = np.asarray(sel)
    m = len(sel)
    sel_pts = pts[sel]
    tree = cKDTree(pts)
    k = min(SURFEL_K, len(pts) - 1)
    _d, idx = tree.query(sel_pts, k=k + 1)   # includes self at column 0
    neigh = pts[idx]                          # (m, k+1, 3)
    centred = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", centred, centred) / (k + 1)
    evals, evecs = np.linalg.eigh(cov)        # ascending eigenvalues; cols = vecs
    l0, l1, l2 = evals[:, 0], evals[:, 1], evals[:, 2]
    normal = evecs[:, :, 0].copy()            # smallest eigenvalue → surface normal
    tmaj = evecs[:, :, 2].copy()              # largest → in-plane major axis

    view = np.asarray(view_dir, dtype="float64")
    if view.ndim == 1:
        view = np.broadcast_to(view, (m, 3))
    flip = np.sum(normal * view, axis=1) < 0.0
    normal[flip] *= -1.0
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    tmaj -= np.sum(tmaj * normal, axis=1, keepdims=True) * normal  # orthogonalise
    tmaj /= np.maximum(np.linalg.norm(tmaj, axis=1, keepdims=True), 1e-12)
    tmin = np.cross(normal, tmaj)

    R_surfel = np.stack([tmaj, tmin, normal], axis=2)   # (m,3,3), columns
    if R_post is not None:
        R_surfel = np.einsum("ij,mjk->mik", np.asarray(R_post, dtype="float64"), R_surfel)
    quat = mat3_to_quat(R_surfel)

    tree_sel = cKDTree(sel_pts)
    dsel, _ = tree_sel.query(sel_pts, k=min(2, m))
    spacing = dsel[:, 1] if dsel.ndim == 2 and dsel.shape[1] > 1 else np.full(m, 0.1)
    s_base = np.clip(SURFEL_FILL * spacing, SURFEL_S_MIN, SURFEL_S_MAX)
    aspect = np.clip(np.sqrt(np.maximum(l1, 1e-9) / np.maximum(l2, 1e-9)),
                     SURFEL_ASPECT_FLOOR, 1.0)
    s_major = s_base
    s_minor = np.clip(s_base * aspect, SURFEL_S_MIN, SURFEL_S_MAX)

    planarity = np.clip((l1 - l0) / np.maximum(l2, 1e-9), 0.0, 1.0)
    opacity = np.clip(0.45 + 0.55 * planarity, 0.30, 1.0)

    bad = (l2 <= 1e-9) | ~np.isfinite(quat).all(axis=1)
    if bad.any():
        quat[bad] = np.array([0.0, 0.0, 0.0, 1.0])
        s_minor[bad] = s_major[bad]
        opacity[bad] = 0.30

    scale2 = np.stack([s_major, s_minor], axis=1)
    return quat, scale2, opacity


# ── Feature 1: Worldbuild cross-sweep merge ──────────────────────────────────
# Accumulate ALL sweeps' returns into a single consolidated render-world cloud:
# STATIC returns are voxel-deduped across the whole drive (so a wall sampled 100×
# becomes one crisp surface) while DYNAMIC returns (inside MOVING actor boxes)
# pass through UN-merged, each keeping its own per-sweep time (so a moving car
# doesn't smear into a solid streak). The static voxel grid is built by a
# STREAMING accumulator (``WorldVoxelAccumulator``) so peak memory is ∝ the
# occupied-voxel count, NOT the ~180M raw returns.
FALLBACK_RGB = (70, 78, 96)  # slate for returns no camera saw (== the colorizers)


# ── photographic colour: linear-light fusion + grade ─────────────────────────
# Camera-projected LiDAR colour washes out two ways: (1) averaging many sweeps'
# samples in sRGB (gamma) space darkens + desaturates (the mean of gamma-encoded
# values is below the perceptual mean), and (2) the raw urban samples are genuinely
# low-saturation/flat (measured ~9% saturation, slight blue cast). Fix (1) by
# fusing in LINEAR light weighted toward the closest/sharpest view; fix (2) with a
# grade (gray-world white balance → exposure → saturation → gamma) at bake time.
def srgb_to_linear(u8):
    """sRGB 0-255 (uint8-ish) → linear-light float in [0,1] (vectorised)."""
    c = np.asarray(u8, dtype="float64") / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    """Linear-light float [0,1] → sRGB 0-255 float (vectorised; clip+cast by caller)."""
    lin = np.clip(np.asarray(lin, dtype="float64"), 0.0, 1.0)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1.0 / 2.4) - 0.055)
    return s * 255.0


def grade_rgb(rgb, *, saturation=1.0, exposure=1.0, gamma=1.0, white_balance=False):
    """Photographic grade of baked per-point RGB (``(N,3)`` 0-255) → graded uint8.

    Order (all in linear light, the physically-correct space for these ops):
    optional GRAY-WORLD white balance (scale each channel so the cloud's mean is
    neutral — removes the overcast/shadow blue cast), EXPOSURE gain, SATURATION push
    about per-pixel luma, then a GAMMA tweak. Defaults are a no-op (1.0 / off) so an
    ungraded caller is unchanged. Returns uint8 ``(N,3)``.
    """
    rgb = np.asarray(rgb, dtype="float64").reshape(-1, 3)
    if len(rgb) == 0:
        return rgb.astype("uint8")
    lin = srgb_to_linear(rgb)
    if white_balance:
        mean = lin.mean(axis=0)                      # per-channel scene mean
        gray = float(mean.mean())
        lin = lin * (gray / np.maximum(mean, 1e-6))  # gray-world: neutralise cast
    lin = lin * float(exposure)
    if saturation != 1.0:
        luma = (lin * np.array([0.2126, 0.7152, 0.0722])).sum(axis=1, keepdims=True)
        lin = luma + (lin - luma) * float(saturation)
    lin = np.clip(lin, 0.0, 1.0)
    if gamma != 1.0:
        lin = lin ** (1.0 / float(gamma))
    return np.clip(linear_to_srgb(lin), 0, 255).astype("uint8")


class WorldVoxelAccumulator:
    """Incremental per-voxel accumulator for the Worldbuild STATIC cloud.

    Call ``add(...)`` once per sweep with that sweep's STATIC returns (already in
    render world x,y + vehicle-frame z). Per occupied voxel it keeps the single
    REAL return nearest the voxel-cell centre (so attributes ride along — NO
    synthetic points), the earliest sweep time it was seen (``first_seen``), and a
    running RGB sum/count over the returns a camera actually saw. ``finalize()``
    emits the deduped representative cloud. Peak memory ∝ occupied voxels.

    Voxel key is ``floor(x/v), floor(y/v), floor(z/v)`` (the same idiom as
    ``_voxel_real_indices``); "nearest the cell centre" uses the cell centre
    ``(key+0.5)*v`` rather than a running centroid so the choice is order-stable
    and needs no second pass.
    """

    def __init__(self, voxel_m: float, color_mode: str = "mean"):
        self.v = float(voxel_m)
        # color_mode: "mean" = running sRGB mean (legacy worldbuild/waymo); or
        # "linear_weighted" = weighted mean in LINEAR light (weight = per-return
        # view quality, e.g. 1/depth) → recovers brightness + saturation a sRGB
        # mean across many sweeps washes out. The slots are identical either way:
        # cur[6] = colour accumulator (3-vec), cur[7] = weight accumulator (scalar).
        self.color_mode = color_mode
        # key(tuple3 int) → list[best_d2, x, y, z, t_rep, first_seen,
        #                        colour_accum(3 float), weight_accum]
        self._vox: dict[tuple, list] = {}

    def add(self, pts_world, z, t_ms, rgb=None, rgb_hit=None, rgb_weight=None):
        """Fold one sweep's static returns into the grid.

        ``pts_world`` is (N,2) render world x,y (metres in the local frame *before*
        the lon/lat transform — any consistent metric frame works), ``z`` (N,) the
        vehicle-frame height, ``t_ms`` a scalar or (N,) per-return time. ``rgb``
        (N,3 uint8-ish) + ``rgb_hit`` (N bool) are the camera color + whether a
        camera saw the return (only hits contribute). ``rgb_weight`` (N, optional)
        is the per-return view quality used in ``color_mode="linear_weighted"``
        (higher = closer/sharper); ignored in "mean" mode.
        """
        pts_world = np.asarray(pts_world, dtype="float64").reshape(-1, 2)
        z = np.asarray(z, dtype="float64").reshape(-1)
        n = len(z)
        if n == 0:
            return
        t = (np.full(n, int(t_ms), dtype="int64") if np.ndim(t_ms) == 0
             else np.asarray(t_ms, dtype="int64"))
        xyz = np.column_stack([pts_world[:, 0], pts_world[:, 1], z])
        keys = np.floor(xyz / self.v).astype(np.int64)
        cell_centre = (keys + 0.5) * self.v
        d2 = np.einsum("ni,ni->n", xyz - cell_centre, xyz - cell_centre)
        if rgb is None:
            rgb = np.zeros((n, 3), dtype="float64")
            rgb_hit = np.zeros(n, dtype=bool)
        else:
            rgb = np.asarray(rgb, dtype="float64").reshape(-1, 3)
            rgb_hit = (np.zeros(n, dtype=bool) if rgb_hit is None
                       else np.asarray(rgb_hit, dtype=bool))
        # Per-return colour contribution + weight, by mode (gated to camera hits).
        if self.color_mode == "linear_weighted":
            w = (np.ones(n) if rgb_weight is None
                 else np.asarray(rgb_weight, dtype="float64").reshape(-1))
            w = np.where(rgb_hit, np.maximum(w, 1e-6), 0.0)
            cval = srgb_to_linear(rgb) * w[:, None]
        else:
            w = rgb_hit.astype("float64")
            cval = rgb * w[:, None]
        vox = self._vox
        for i in range(n):
            key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            ti = int(t[i])
            cur = vox.get(key)
            if cur is None:
                vox[key] = [
                    float(d2[i]), float(xyz[i, 0]), float(xyz[i, 1]), float(z[i]),
                    ti, ti, cval[i].copy(), float(w[i]),
                ]
                continue
            if ti < cur[5]:        # earliest sweep time = first_seen
                cur[5] = ti
            cur[6] = cur[6] + cval[i]   # colour accumulator (0 for non-hits)
            cur[7] += float(w[i])       # weight accumulator
            if d2[i] < cur[0]:     # representative = real return nearest cell centre
                cur[0] = float(d2[i])
                cur[1] = float(xyz[i, 0])
                cur[2] = float(xyz[i, 1])
                cur[3] = float(z[i])
                cur[4] = ti

    def finalize(self):
        """→ (xy N×2, z N, rgb N×3 uint8, first_seen N int64). One row / voxel."""
        n = len(self._vox)
        xy = np.zeros((n, 2), dtype="float64")
        z = np.zeros(n, dtype="float64")
        rgb = np.zeros((n, 3), dtype="float64")
        first_seen = np.zeros(n, dtype="int64")
        fb = np.asarray(FALLBACK_RGB, dtype="float64")
        linear = self.color_mode == "linear_weighted"
        for i, cur in enumerate(self._vox.values()):
            xy[i, 0] = cur[1]
            xy[i, 1] = cur[2]
            z[i] = cur[3]
            first_seen[i] = cur[5]
            if cur[7] > 0:
                mean = cur[6] / cur[7]
                rgb[i] = linear_to_srgb(mean) if linear else mean
            else:
                rgb[i] = fb
        rgb = np.clip(np.round(rgb), 0, 255).astype("uint8")
        return xy, z, rgb, first_seen


def point_in_moving_boxes(pts_world, z, t_ms, boxes):
    """Boolean mask: which (x,y) returns fall inside ANY moving actor box.

    ``boxes`` is an iterable of dicts ``{cx, cy, heading, length, width, t}`` in
    the SAME render-world metric frame as ``pts_world`` (one per moving-object
    observation at its sweep time ``t`` ms; height is ignored — the test is a 2D
    oriented-rectangle membership in the box-local frame, ample for tagging
    dynamic returns). A return is dynamic if at ITS sweep time it lies inside a
    box observed at the same ms.

    Box-local test: rotate the return by ``-heading`` about the box centre, then
    ``|x'| <= length/2`` and ``|y'| <= width/2`` (a small ε pad absorbs box-fit
    slack). ``z`` is accepted for signature symmetry / future 3D use but unused.
    """
    pts_world = np.asarray(pts_world, dtype="float64").reshape(-1, 2)
    n = len(pts_world)
    t_ms = (np.full(n, int(t_ms), dtype="int64") if np.ndim(t_ms) == 0
            else np.asarray(t_ms, dtype="int64"))
    dyn = np.zeros(n, dtype=bool)
    if n == 0:
        return dyn
    by_t: dict[int, list] = {}
    for b in boxes:
        by_t.setdefault(int(b["t"]), []).append(b)
    eps = 0.25  # m — pad for box-fit slack
    for ti in np.unique(t_ms):
        sel = np.flatnonzero(t_ms == ti)
        blist = by_t.get(int(ti))
        if not blist:
            continue
        px = pts_world[sel, 0]
        py = pts_world[sel, 1]
        for b in blist:
            dx = px - b["cx"]
            dy = py - b["cy"]
            c = math.cos(-b["heading"])
            s = math.sin(-b["heading"])
            xl = dx * c - dy * s
            yl = dx * s + dy * c
            hit = (np.abs(xl) <= b["length"] / 2.0 + eps) & \
                  (np.abs(yl) <= b["width"] / 2.0 + eps)
            if hit.any():
                dyn[sel[hit]] = True
    return dyn


def _fit_merged_surfels(pts_xyz):
    """Fit oriented surfels on a consolidated cross-sweep cloud.

    ``pts_xyz`` is (M,3) in render-world x,y + vehicle-frame z. Shared by
    ``worldbuild_merge`` / ``finalize_stage`` / ``finalize_actors``: there is no
    single ego pose for a many-sweep cloud, so each normal is oriented toward the
    cloud centroid (a stable view dir) with NO ego-yaw ``R_post``. Empty- and
    small-cloud-safe: fewer than ``SURFEL_K + 2`` points fall back to flat
    up-facing low-confidence disks. Returns (M,7)
    ``[qx,qy,qz,qw,s_major,s_minor,opacity]``.
    """
    pts = np.asarray(pts_xyz, dtype="float64").reshape(-1, 3)
    m = len(pts)
    if m >= SURFEL_K + 2:
        centroid = pts.mean(axis=0)
        view_dir = centroid[None, :] - pts          # per-point, toward centroid
        quat, scale2, op = _surfel_frame(pts, np.arange(m), view_dir)
        return np.concatenate([quat, scale2, op[:, None]], axis=1)
    return np.tile(
        np.array([0.0, 0.0, 0.0, 1.0, SURFEL_S_MIN, SURFEL_S_MIN, 0.30]),
        (max(m, 0), 1))


def worldbuild_merge(static_acc, dynamic):
    """Combine the streamed STATIC voxel grid + the full-rate DYNAMIC returns,
    fit surfels on the merged deduped cloud, and assemble the per-column arrays
    for ``write_lidar_points`` (Feature 1 — Worldbuild).

    ``static_acc`` is a finalized ``WorldVoxelAccumulator`` (or its ``finalize()``
    4-tuple). ``dynamic`` is a dict of the un-merged dynamic returns with keys
    ``xy`` (N×2 render world), ``z`` (N), ``rgb`` (N×3 uint8), ``t`` (N int64
    per-sweep ms). Either side may be empty.

    The merged STATIC points keep their voxel ``first_seen`` as their timestamp;
    DYNAMIC points keep their own per-sweep time. Surfels are fit on the WHOLE
    merged cloud (static + dynamic) so even the dynamic returns render as oriented
    disks; the merged path uses NO ego-yaw rotation (there is no single ego pose —
    the normal is oriented toward the scene centroid).

    Returns a dict with keys:
      ``xy`` (M×2), ``z`` (M), ``rgb`` (M×3 uint8), ``timestamp`` (M int64),
      ``is_dynamic`` (M int64 0/1), ``world_class`` (M object "static"/"dynamic"),
      ``surfels`` (M×7 [qx,qy,qz,qw,s_major,s_minor,opacity]).
    """
    if isinstance(static_acc, WorldVoxelAccumulator):
        s_xy, s_z, s_rgb, s_first = static_acc.finalize()
    else:
        s_xy, s_z, s_rgb, s_first = static_acc
    s_xy = np.asarray(s_xy, dtype="float64").reshape(-1, 2)
    s_z = np.asarray(s_z, dtype="float64").reshape(-1)
    s_rgb = np.asarray(s_rgb).reshape(-1, 3)
    s_first = np.asarray(s_first, dtype="int64").reshape(-1)
    n_static = len(s_z)

    d_xy = np.asarray(dynamic.get("xy", np.empty((0, 2))), dtype="float64").reshape(-1, 2)
    d_z = np.asarray(dynamic.get("z", np.empty(0)), dtype="float64").reshape(-1)
    d_rgb = np.asarray(dynamic.get("rgb", np.empty((0, 3))), dtype="float64").reshape(-1, 3)
    d_t = np.asarray(dynamic.get("t", np.empty(0)), dtype="int64").reshape(-1)
    n_dyn = len(d_z)

    xy = np.concatenate([s_xy, d_xy], axis=0) if n_dyn else s_xy
    z = np.concatenate([s_z, d_z]) if n_dyn else s_z
    rgb = (np.concatenate([s_rgb, d_rgb], axis=0) if n_dyn else s_rgb)
    rgb = np.clip(np.round(np.asarray(rgb, dtype="float64")), 0, 255).astype("uint8")
    timestamp = np.concatenate([s_first, d_t]) if n_dyn else s_first
    is_dynamic = np.concatenate([
        np.zeros(n_static, dtype="int64"), np.ones(n_dyn, dtype="int64")])
    world_class = np.array(["static"] * n_static + ["dynamic"] * n_dyn, dtype=object)

    # Surfels on the merged cloud (x,y world + vehicle-frame z), oriented toward
    # the scene centroid (no single ego pose) — see _fit_merged_surfels.
    surfels = _fit_merged_surfels(np.column_stack([xy[:, 0], xy[:, 1], z]))

    return dict(
        xy=xy, z=z, rgb=rgb, timestamp=timestamp,
        is_dynamic=is_dynamic, world_class=world_class, surfels=surfels,
    )


# ── Scene-split ("stage + actors"): decompose into TWO separate archives ──────
# Unlike Worldbuild (one merged archive + an is_dynamic flag the shader branches
# on), the scene-split mode emits the STATIC stage and the DYNAMIC actors as two
# independent STT archives with different temporal structure: the stage is one
# timeless cloud (the fixed infrastructure, summarised by accumulating every
# sweep) and the actors are time-bucketed per-sweep returns. Surfels are fit per
# archive (the stage is voxel-uniform → correct disk extents; the actors are dense
# per-sweep) instead of once on the union, which would distort both.
def finalize_stage(static_acc, *, drop_mask=None, adaptive=None, grade=None):
    """Static "stage" columns for the scene-split mode (the fixed environment).

    ``static_acc`` is a finalized ``WorldVoxelAccumulator`` (or its ``finalize()``
    4-tuple ``(xy, z, rgb, first_seen)``). ``drop_mask`` (optional bool over the
    finalized voxels, ``True`` = ERASOR-scrubbed residual dynamic — see
    ``erasor_scrub``) is removed BEFORE the surfel fit. ``adaptive`` (optional dict
    of ``adaptive_stage_select`` kwargs, e.g. ``{base_voxel, max_voxel, levels}``)
    curvature-adaptively coarsens flat regions into big surfels for a more COMPACT
    stage; ``None`` keeps the uniform cloud. Surfels are fit on the resulting static
    cloud alone. ``timestamp`` carries each voxel's ``first_seen`` (free; lets a
    future reveal-fade ramp in by first-observed time) — for the build the stage is
    given a whole-scene ``end_timestamp`` so it loads once and persists.

    Returns the ``write_lidar_points`` column dict: ``xy`` (M,2), ``z`` (M),
    ``rgb`` (M,3 uint8), ``timestamp`` (M int64), ``surfels`` (M,7),
    ``is_dynamic`` (M == 0), ``world_class`` (M == "static").
    """
    if isinstance(static_acc, WorldVoxelAccumulator):
        xy, z, rgb, first_seen = static_acc.finalize()
    else:
        xy, z, rgb, first_seen = static_acc
    xy = np.asarray(xy, dtype="float64").reshape(-1, 2)
    z = np.asarray(z, dtype="float64").reshape(-1)
    rgb = np.asarray(rgb).reshape(-1, 3)
    first_seen = np.asarray(first_seen, dtype="int64").reshape(-1)
    if drop_mask is not None:
        keep = ~np.asarray(drop_mask, dtype=bool).reshape(-1)
        xy, z, rgb, first_seen = xy[keep], z[keep], rgb[keep], first_seen[keep]
    if adaptive is not None and len(z) > 0:
        sel = adaptive_stage_select(
            np.column_stack([xy[:, 0], xy[:, 1], z]), **adaptive)
        xy, z, rgb, first_seen = xy[sel], z[sel], rgb[sel], first_seen[sel]
    m = len(z)
    surfels = _fit_merged_surfels(np.column_stack([xy[:, 0], xy[:, 1], z]))
    rgb = np.clip(np.round(np.asarray(rgb, dtype="float64")), 0, 255).astype("uint8")
    # Photographic grade (gray-world white balance + exposure + saturation) — the
    # urban camera samples are genuinely flat/blue-cast, so a grade is what makes
    # the stage read photoreal rather than washed out. No-op when grade is None.
    if grade is not None and m > 0:
        rgb = grade_rgb(rgb, **grade)
    return dict(
        xy=xy, z=z, rgb=rgb, timestamp=first_seen, surfels=surfels,
        is_dynamic=np.zeros(m, dtype="int64"),
        world_class=np.array(["static"] * m, dtype=object),
    )


def finalize_actors(dynamic, *, grade=None, denoise=None):
    """Dynamic "actors" columns for the scene-split mode (the moving agents).

    ``dynamic`` is the un-merged dynamic-returns dict (keys ``xy`` (N,2), ``z``
    (N), ``rgb`` (N,3), ``t`` (N int64 per-sweep ms)) — the returns inside MOVING
    actor boxes, each keeping its own sweep time so a car doesn't smear into a
    streak. Surfels are fit on the dynamic cloud alone. Returns the same column
    dict shape as ``finalize_stage`` with ``timestamp`` == per-sweep t,
    ``is_dynamic`` == 1, ``world_class`` == "dynamic".
    """
    xy = np.asarray(dynamic.get("xy", np.empty((0, 2))), dtype="float64").reshape(-1, 2)
    z = np.asarray(dynamic.get("z", np.empty(0)), dtype="float64").reshape(-1)
    rgb = np.asarray(dynamic.get("rgb", np.empty((0, 3)))).reshape(-1, 3)
    t = np.asarray(dynamic.get("t", np.empty(0)), dtype="int64").reshape(-1)
    # Principled noise reduction (rain / sensor scatter): per-sweep statistical
    # outlier removal keeps the dense moving agents, drops the isolated haze.
    if denoise is not None and len(z) > 0:
        keep = denoise_actors(xy, z, t, **denoise)
        xy, z, rgb, t = xy[keep], z[keep], rgb[keep], t[keep]
    m = len(z)
    surfels = _fit_merged_surfels(np.column_stack([xy[:, 0], xy[:, 1], z]))
    rgb = np.clip(np.round(np.asarray(rgb, dtype="float64")), 0, 255).astype("uint8")
    # Same photographic grade as the stage so the actors share its colour space.
    if grade is not None and m > 0:
        rgb = grade_rgb(rgb, **grade)
    return dict(
        xy=xy, z=z, rgb=rgb, timestamp=t, surfels=surfels,
        is_dynamic=np.ones(m, dtype="int64"),
        world_class=np.array(["dynamic"] * m, dtype=object),
    )


# ── ERASOR-style residual scrub (cleans ghost trails off the static stage) ────
# The in-box test (point_in_moving_boxes) removes returns inside ANNOTATED moving
# boxes, but box-fit slack + unlabeled movers leave faint ghost trails baked into
# the accumulated stage. erasor_scrub is a deterministic second pass after the
# in-box removal: it bins each sweep's view into egocentric rings × sectors
# (ERASOR's R-POD), and where the accumulated MAP holds tall structure a live scan
# does NOT see (height-diff above what that sweep observed in the same bin) it
# votes the map points "transient". A map point seen by enough sweeps and judged
# transient by most of them is dropped. Ground (the lowest live returns per bin)
# is never scrubbed. Per-sweep EGOCENTRIC bins (not one global polar grid) because
# the ego translates across the ~15 s log. Reductions are commutative (min/max/sum
# over integer bin ids) + percentile-free ground, so the result is order-independent.
ERASOR_RINGS = 20            # radial bins out to ERASOR_MAX_R (~4 m each)
ERASOR_SECTORS = 60          # angular bins (6° each)
ERASOR_MAX_R = 80.0          # m — outer ring edge (AV2 lidar useful range)
ERASOR_HEIGHT_DIFF = 0.5     # m — map-vs-scan max-height gap that flags a bin transient
ERASOR_OCC_RATIO = 0.3       # keep if a bin reads as real structure in ≥ this share of sweeps
ERASOR_GROUND_MARGIN = 0.3   # m above the per-bin ground = protected (never scrubbed)
ERASOR_MIN_SCANS = 3         # a map point needs ≥ this many observing sweeps to be testable


def _erasor_polar_bins(dx, dy, rings, sectors, max_r):
    """(dx,dy) relative to ego → flat bin id ``ring*sectors + sector``."""
    r = np.hypot(dx, dy)
    ring = np.clip((r / max_r * rings).astype(np.int64), 0, rings - 1)
    theta = np.arctan2(dy, dx)
    sector = ((theta + np.pi) / (2.0 * np.pi) * sectors).astype(np.int64) % sectors
    return ring * sectors + sector


def erasor_scrub(static_acc, scans, *, rings=ERASOR_RINGS, sectors=ERASOR_SECTORS,
                 max_r=ERASOR_MAX_R, height_diff=ERASOR_HEIGHT_DIFF,
                 occ_ratio=ERASOR_OCC_RATIO, ground_margin=ERASOR_GROUND_MARGIN,
                 min_scans=ERASOR_MIN_SCANS):
    """Boolean drop-mask over the finalized stage voxels (``True`` = residual dynamic).

    ``static_acc`` is the accumulated MAP (a ``WorldVoxelAccumulator`` or its
    ``finalize()`` 4-tuple). ``scans`` is a list of ``(xy (N,2), z (N), ego_xy (2,))``
    — each sweep's STATIC returns (post in-box removal) + that sweep's ego position,
    in the SAME render-world metric frame as the map. The returned mask is aligned
    to ``finalize()`` order, so pass the SAME finalized arrays to ``finalize_stage``
    (call ``finalize()`` once) to keep the indices in lock-step.

    Per sweep: pull the map points within ``max_r`` of the ego, bin both the map
    and the scan into egocentric rings×sectors. A bin's ground = the lowest scan
    return in it; scan structure height = the highest. An above-ground map point is
    "transient this sweep" when ``map_z − scan_maxz > height_diff`` (the map holds
    structure the live scan does not). Only bins the scan actually populated count
    as "seen" (avoids penalising FOV gaps). A map point with
    ``seen ≥ min_scans`` and ``transient/seen ≥ 1 − occ_ratio`` is dropped.
    """
    from scipy.spatial import cKDTree

    if isinstance(static_acc, WorldVoxelAccumulator):
        m_xy, m_z, _rgb, _first = static_acc.finalize()
    else:
        m_xy, m_z, _rgb, _first = static_acc
    m_xy = np.asarray(m_xy, dtype="float64").reshape(-1, 2)
    m_z = np.asarray(m_z, dtype="float64").reshape(-1)
    big = len(m_z)
    drop = np.zeros(big, dtype=bool)
    if big == 0 or not scans:
        return drop

    nbins = rings * sectors
    tree = cKDTree(m_xy)
    seen = np.zeros(big, dtype="int64")
    transient = np.zeros(big, dtype="int64")

    for s_xy, s_z, ego in scans:
        ego = np.asarray(ego, dtype="float64").reshape(2)
        mi = np.asarray(tree.query_ball_point(ego, max_r), dtype=np.int64)
        if mi.size == 0:
            continue
        s_xy = np.asarray(s_xy, dtype="float64").reshape(-1, 2)
        s_z = np.asarray(s_z, dtype="float64").reshape(-1)
        if s_xy.shape[0]:
            srel = s_xy - ego
            in_rng = np.hypot(srel[:, 0], srel[:, 1]) < max_r
            srel, sz = srel[in_rng], s_z[in_rng]
        else:
            srel, sz = np.empty((0, 2)), np.empty(0)
        if srel.shape[0] == 0:
            continue
        s_bin = _erasor_polar_bins(srel[:, 0], srel[:, 1], rings, sectors, max_r)
        # Per-bin live ground (lowest return) + structure top (highest return).
        bin_ground = np.full(nbins, np.inf)
        bin_top = np.full(nbins, -np.inf)
        np.minimum.at(bin_ground, s_bin, sz)
        np.maximum.at(bin_top, s_bin, sz)
        bin_has_scan = np.zeros(nbins, dtype=bool)
        bin_has_scan[s_bin] = True

        m_bin = _erasor_polar_bins(m_xy[mi, 0] - ego[0], m_xy[mi, 1] - ego[1],
                                   rings, sectors, max_r)
        observed = bin_has_scan[m_bin]
        above_ground = m_z[mi] > (bin_ground[m_bin] + ground_margin)
        is_trans = observed & above_ground & \
            ((m_z[mi] - bin_top[m_bin]) > height_diff)
        seen[mi[observed]] += 1
        transient[mi[is_trans]] += 1

    testable = seen >= min_scans
    drop[testable] = transient[testable] >= (1.0 - occ_ratio) * seen[testable]
    return drop


# ── Feature 2: object tracks (always emitted, the tracks/ archive) ───────────
def build_tracks(*, lon, lat, timestamp, category, track_id, speed):
    """Group per-observation object points into per-track polylines (Feature 2).

    Parallel arrays of object OBSERVATIONS (one row per object per sample — the
    same arrays that feed ``write_objects_points``): ``lon`` / ``lat`` (deg),
    ``timestamp`` (ms), ``category`` (canonical class str), ``track_id`` (str),
    ``speed`` (m/s). Groups by ``track_id`` (the AV2 velocity grouping idiom),
    sorts each by time, and emits one track per id. Tracks with < 2 vertices are
    dropped (a LineString needs ≥ 2 points). The track's ``category`` is its
    first-seen class (constant per track).

    Returns a list of dicts ``{lon, lat, vts (vertex times int64), vvals (speed
    float32), category}`` ready for ``write_track_lines``. INCLUDE the ego as one
    more track (category ``"ego"``) — the caller folds it in (see the extractors).
    """
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    timestamp = np.asarray(timestamp, dtype="int64")
    speed = np.asarray(speed, dtype="float64")
    category = list(category)
    track_id = list(track_id)

    by_track: dict[str, list[int]] = {}
    for i, tid in enumerate(track_id):
        by_track.setdefault(str(tid), []).append(i)

    tracks: list[dict] = []
    for tid, idx in by_track.items():
        idx.sort(key=lambda i: int(timestamp[i]))
        if len(idx) < 2:
            continue  # a LineString needs ≥ 2 vertices
        ii = np.asarray(idx)
        tracks.append(dict(
            lon=lon[ii],
            lat=lat[ii],
            vts=[int(t) for t in timestamp[ii]],
            vvals=[float(v) for v in speed[ii]],
            category=str(category[idx[0]]),  # first-seen class (constant)
        ))
    return tracks


# ── Feature 5: scan-phase → hue ramp ─────────────────────────────────────────
def phase_to_rgb(phase) -> np.ndarray:
    """Map a per-point scan phase ∈ [0,1] → an (N,3) uint8 RGB rainbow (HSV sweep).

    Feature 5 (Sweep): the AV2 scan bundle colours each return by its normalized
    offset within the sweep so the LiDAR beam renders as a ROTATING RAINBOW via
    the existing r/g/b path. A simple full-saturation full-value HSV sweep
    (hue = phase·360°) — vectorised, no colorsys/matplotlib. Values clamp to
    ``[0,1]`` so the endpoints (phase 0 and 1) both land on red, closing the loop.
    """
    h = (np.clip(np.asarray(phase, dtype="float64"), 0.0, 1.0)) * 6.0  # sector 0..6
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    # S=V=1 → the standard HSV-to-RGB six-sector form with p=0, q=1−f, t=f.
    q = 1.0 - f
    t = f
    r = np.choose(i, [1.0, q, 0.0, 0.0, t, 1.0])
    g = np.choose(i, [t, 1.0, 1.0, q, 0.0, 0.0])
    b = np.choose(i, [0.0, 0.0, t, 1.0, 1.0, q])
    return np.clip(np.round(np.stack([r, g, b], axis=1) * 255.0), 0, 255).astype("uint8")


def write_track_lines(out_path: Path, *, tracks) -> int:
    """Write the ``tracks/`` LineString GeoParquet (Feature 2).

    One WKB LineString per tracked object (the ego folded in as category
    ``"ego"``). Columns: ``geometry`` (WKB LineString), ``timestamp`` /
    ``end_timestamp`` (Int64 first/last vertex ms = the feature window),
    ``vertex_timestamps`` (List<Int64> per-vertex ms — load-bearing for the
    PathLayer time animation), ``vertex_values`` (List<Float32> per-vertex speed
    m/s), ``category`` (Utf8 categorical = canonical class / ``"ego"``). Mirrors
    ``write_ego_trips`` but multi-row and adds ``category``. Built with
    ``run_stt_build(kind="line")`` (NOT trips/point) so it skips ``--simplify`` /
    ``--quantize-coords`` — both would desync ``vertex_timestamps`` from the
    coords and trip the PathLayer vertex-buffer GL bug. Returns the row count.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import wkb
    from shapely.geometry import LineString

    geoms: list[bytes] = []
    t0s: list[int] = []
    t1s: list[int] = []
    vts_col: list[list[int]] = []
    vvals_col: list[list[float]] = []
    cats: list[str] = []
    for tr in tracks:
        lon = np.asarray(tr["lon"], dtype="float64")
        lat = np.asarray(tr["lat"], dtype="float64")
        vts = [int(t) for t in tr["vts"]]
        vvals = [np.float32(v) for v in tr["vvals"]]
        if not (len(lon) == len(lat) == len(vts) == len(vvals)):
            raise ValueError("track: lon/lat/vts/vvals length mismatch")
        if len(lon) < 2:
            continue  # guard (build_tracks already drops these)
        coords = list(zip(lon.tolist(), lat.tolist()))
        geoms.append(wkb.dumps(LineString(coords)))
        t0s.append(vts[0])
        t1s.append(vts[-1])
        vts_col.append(vts)
        vvals_col.append(vvals)
        cats.append(str(tr["category"]))
    n = len(geoms)
    table = pa.table({
        "geometry": pa.array(geoms, type=pa.binary()),
        "timestamp": pa.array(t0s, type=pa.int64()),
        "end_timestamp": pa.array(t1s, type=pa.int64()),
        "vertex_timestamps": pa.array(vts_col, type=pa.list_(pa.int64())),
        "vertex_values": pa.array(vvals_col, type=pa.list_(pa.float32())),
        "category": pa.array(cats, type=pa.string()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  wrote {n} track LineString(s) → {out_path}")
    return n


def run_stt_build(
    parquet_path: Path,
    out_dir: Path,
    kind: str,
    *,
    stt_build: str = "stt-build",
    min_zoom: int = 0,
    max_zoom: int = 18,
    temporal_bucket: str = "1s",
    map_temporal_bucket: str = "1d",
    quantize_coords: float | None = None,
    quantize_attrs: dict[str, float] | None = None,
    vector_groups: list[tuple[str, list[str], str]] | None = None,
    point_elevation_column: str | None = None,
    min_zoom_field: str | None = None,
    max_zoom_field: str | None = None,
    static_full_range: bool = False,
    publish: bool = True,
) -> None:
    """Run ``stt-build`` on one archive's GeoParquet → a packed STT directory.

    ``kind`` selects the geometry / time-field flags (the geometry *type* itself
    is auto-detected from the WKB ``geometry`` column — these flags only differ in
    how time is handled):

    * ``"point"`` (lidar / objects): ``--time-field timestamp --time-format
      unix-ms --temporal-bucket <bucket>``.
    * ``"trips"`` (ego): adds ``--end-time-field end_timestamp --simplify`` (the
      animated-LineString form, identical to ``ecco_advect.py``).
    * ``"map_poly"`` (HD-map polygons) / ``"map_line"`` (HD-map lines): the
      **static** full-range form (av-refinement.md §R2.2). Adds
      ``--end-time-field end_timestamp`` and FORCES a temporal bucket that spans
      the whole scene (overriding ``temporal_bucket`` with a large default, see
      ``map_temporal_bucket``) so every map feature lands in a single bucket and
      the archive loads once + persists for the entire replay. ``map_poly``
      mirrors storms.rs's polygon build (clip at tile boundaries); ``map_line``
      mirrors its linestring build with ``--no-clip`` so a short divider isn't
      dropped at a tile edge.

    Output is the packed format (manifest.json + index/ + packs/), same as
    ecco/storms. ``--publish`` raises the zstd level to 19 for serve-as-is
    tiles (the stt-generate convention).

    Two OPT-IN size levers (both default OFF so output is byte-identical unless a
    caller asks for them — measure before committing, per the repo's measure-first
    convention):

    * ``min_zoom`` — clamp the bottom of the zoom pyramid. AV point clouds are
      meaningful only ~z14-18; building from ``z0`` packs a useless ultra-dense
      low-zoom tile (the "z0 bomb") and bloats packs. Raise it for ``point``
      archives (lidar/objects); keep ``0`` for the sparse ego trip + HD-map.
    * ``quantize_coords`` — when > 0, pass ``--quantize-coords <meters>`` to store
      geometry as fixed-point integers at that ground precision (−25..47% on
      coordinate-heavy data). NOTE for meter-scale AV scenes: the grid is square
      in *degrees*, so at mid latitudes ``1.0`` snaps longitude to ~0.75 m
      (visible jitter on a ~4.5 m car) — use ``<= 0.1`` (~7.5 cm) for AV.
    """
    valid = ("point", "trips", "map_poly", "map_line", "line")
    if kind not in valid:
        raise ValueError(f"run_stt_build: unknown kind {kind!r} (use one of {valid})")

    is_map = kind in ("map_poly", "map_line")
    # Full-range archives (HD-map, or the scene-split STATIC stage) collapse into
    # ONE temporal bucket so they load once + persist; a bucket >= the scene
    # duration guarantees that regardless of the per-point bucket the caller passed.
    full_range = is_map or (static_full_range and kind == "point")
    bucket = map_temporal_bucket if full_range else temporal_bucket

    cmd = [
        stt_build,
        "--input", str(parquet_path),
        "--output", str(out_dir),
        "--time-field", "timestamp",
        "--time-format", "unix-ms",
        "--min-zoom", str(min_zoom),
        "--max-zoom", str(max_zoom),
        "--temporal-bucket", bucket,
        "--compression", "zstd",
    ]
    # Opt-in coordinate quantization (the largest size lever) — only emitted when
    # the caller asks for it, so the default build is unchanged.
    if quantize_coords is not None and quantize_coords > 0:
        cmd += ["--quantize-coords", repr(float(quantize_coords))]
    # Opt-in numeric-attribute quantization (e.g. LiDAR ``z`` elevation): store
    # the named Float64 column as fixed-point ints + a per-column affine the
    # reader reconstructs. A raw Float64 ``z`` barely compresses (~38% of a
    # post-id-fix LiDAR tile); quantizing it to a cm grid is ~−90% on that
    # column. Only emitted when asked, so other builds are byte-identical.
    if quantize_attrs:
        for name, prec in quantize_attrs.items():
            if prec and prec > 0:
                cmd += ["--quantize-attr", f"{name}={float(prec)!r}"]
    # Fuse scalar surfel / colour columns into GPU-ready interleaved
    # FixedSizeList columns (`--vector-group name=cols[:f32|u8]`) so the client
    # binds them zero-copy — no per-point re-pack on the render thread. A group
    # whose source columns are absent from a tile is silently skipped by the
    # encoder, so it's safe to pass surfel groups for a non-surfel archive.
    if vector_groups:
        for name, components, elem in vector_groups:
            spec = f"{name}={','.join(components)}"
            if elem and elem != "f32":
                spec += f":{elem}"
            cmd += ["--vector-group", spec]
    # 3D POINT geometry: fold a numeric column (LiDAR `z`) into the geometry's
    # 3rd coordinate so the renderer binds 3D positions zero-copy (no pad-to-3D
    # on the main thread). POINT clouds only (ScatterplotLayer/AnimatedPointLayer);
    # surfel bundles keep 2D + a separate elevation attribute.
    if point_elevation_column:
        cmd += ["--point-elevation-column", point_elevation_column]
    # Additive-octree LOD: confine each feature to a single zoom level read from a
    # per-feature numeric column (lod_home_zoom writes `home_zoom`). With min ==
    # max field the point lands in EXACTLY that zoom's tiles, so the per-zoom
    # archive is the additive pyramid (no replication across zooms). The column
    # must survive to the encoder (it does — no --exclude-all here), and is read
    # at tiling time BEFORE quantization, so quantizing it (lidar_quantize_attrs)
    # doesn't affect placement.
    if min_zoom_field:
        cmd += ["--min-zoom-field", min_zoom_field]
    if max_zoom_field:
        cmd += ["--max-zoom-field", max_zoom_field]
    if kind == "trips":
        cmd += ["--end-time-field", "end_timestamp", "--simplify"]
    elif kind == "line":
        # Windowed LineStrings (live density-contour iso-lines): each feature is
        # shown for its own [timestamp, end_timestamp] playhead window — like trips
        # / map_line — but at the caller's SMALL per-window temporal bucket (NOT the
        # static whole-scene map bucket) so the contour map morphs as the playhead
        # moves. Crucially NO --simplify: it would distort the contour polylines.
        # Keep contours whole (--no-clip) so a loop isn't dropped at a tile edge.
        # The caller must NOT pass quantize_coords here — quantizing multi-vertex
        # LineStrings mis-sizes PathLayer's instanced draw ("vertex buffer is not
        # big enough"); lines are cheap, so store plain Float64 coords.
        cmd += ["--end-time-field", "end_timestamp", "--no-clip"]
    elif is_map:
        # Full-range static features: valid_from = timestamp, valid_to =
        # end_timestamp (= timeRange.end), so they're present for the whole replay.
        cmd += ["--end-time-field", "end_timestamp"]
        if kind == "map_line":
            # Keep short lane/road dividers whole (don't clip at tile edges) —
            # mirrors the linestring handling in storms.rs's track build.
            cmd.append("--no-clip")
    elif static_full_range and kind == "point":
        # Scene-split STATIC "stage": a full-range POINT cloud. Same idiom as the
        # HD-map (one whole-scene bucket via `full_range` above + per-point validity
        # [timestamp, end_timestamp]) so the fixed environment loads once and stays
        # up for the entire replay. Requires an `end_timestamp` column (emit it via
        # write_lidar_points(end_timestamp=...)). Points never span tile edges, so
        # the default clip is correct — no --no-clip.
        cmd += ["--end-time-field", "end_timestamp"]
    if publish:
        cmd.append("--publish")

    print("  running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"  built {kind} archive → {out_dir}")
