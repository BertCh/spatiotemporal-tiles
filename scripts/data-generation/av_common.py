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
    intensity=None,
    seg_class=None,
    rgb=None,
    surfels=None,
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
    if seg_class is not None:
        seg = np.asarray(seg_class, dtype=object)
        if len(seg) != len(lon):
            raise ValueError(
                f"lidar: seg_class length {len(seg)} != point count {len(lon)}"
            )
        cols["seg_class"] = pa.array([str(s) for s in seg], type=pa.string())
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
        for j, name in enumerate(
            ("qx", "qy", "qz", "qw", "s_major", "s_minor", "surfel_opacity")
        ):
            cols[name] = pa.array(surfels[:, j], type=pa.float64())

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
    * ``qx``/``qy``/``qz``/``qw`` — unit surfel quaternion ∈ [-1,1]; ``1e-4`` is
      ~0.01° of orientation error (imperceptible) and fits ``u16`` (range 2 / 1e-4
      = 20 000 < 65 535). 8 B/pt → 2 B/pt each.
    * ``s_major``/``s_minor`` — in-plane extents (m) at cm precision.
    * ``surfel_opacity`` — confidence ∈ [0,1] at ~1/250 precision.
    """
    return {
        "z": z_prec,
        "r": 1.0,
        "g": 1.0,
        "b": 1.0,
        "qx": 1e-4,
        "qy": 1e-4,
        "qz": 1e-4,
        "qw": 1e-4,
        "s_major": 0.01,
        "s_minor": 0.01,
        "surfel_opacity": 0.004,
    }


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
    valid = ("point", "trips", "map_poly", "map_line")
    if kind not in valid:
        raise ValueError(f"run_stt_build: unknown kind {kind!r} (use one of {valid})")

    is_map = kind in ("map_poly", "map_line")
    # Static map archives must collapse into ONE temporal bucket so they load
    # once + persist; a bucket >= the scene duration guarantees that regardless
    # of the per-point bucket the caller passed.
    bucket = map_temporal_bucket if is_map else temporal_bucket

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
    if kind == "trips":
        cmd += ["--end-time-field", "end_timestamp", "--simplify"]
    elif is_map:
        # Full-range static features: valid_from = timestamp, valid_to =
        # end_timestamp (= timeRange.end), so they're present for the whole replay.
        cmd += ["--end-time-field", "end_timestamp"]
        if kind == "map_line":
            # Keep short lane/road dividers whole (don't clip at tile edges) —
            # mirrors the linestring handling in storms.rs's track build.
            cmd.append("--no-clip")
    if publish:
        cmd.append("--publish")

    print("  running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"  built {kind} archive → {out_dir}")
