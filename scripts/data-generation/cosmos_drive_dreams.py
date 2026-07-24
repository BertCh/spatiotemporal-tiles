#!/usr/bin/env python3
"""Cosmos-Drive-Dreams → the "world model scenario explorer" showcase bundle.

Converts a subset of NVIDIA's Cosmos-Drive-Dreams dataset (HF
``nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams``, CC BY 4.0 —
redistribution permitted with attribution) into ONE combined multi-scenario
STT bundle: ~300 driving scenarios (HD map + oriented boxes + ego) laid out on
a synthetic lat/lon grid, all normalized to a shared clock so the whole gallery
animates in phase, plus the Cosmos-generated MP4 for each one and per-hero LiDAR
archives.

Each world covers exactly the 121 source frames its generated video covers (a
"chunk"), rebased so that window starts at T0. That is the sync contract: the
loop IS the generated window, so position in the loop maps 1:1 onto position in
the video with nothing left over.

DATA / LICENSE: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams
Labels are per-clip tars (``{uuid}_{start}_{end}.tar``) per modality folder —
downloaded per-clip. The synthetic videos exist ONLY as a ~695 GB tar.gz
byte-split into 17 parts; part-000 is the head of the gzip stream, so this
script STREAMS it (requests → gzip → tar) and keeps only the wanted MP4s —
the 40 GB part is never stored. Scenario selection follows video coverage.

Source facts (probed 2026-07-23):
* clips: ~297 frames @ 30 FPS (~9.9 s); videos are 121-frame chunks
  (chunk 0 = frames 0..120, chunk 1 = 121..241) × 7 weather variants.
* ``all_object_info``: per-FRAME JSON ``{tid: {object_to_world 4×4,
  object_lwh, object_is_moving, object_type}}`` (30 Hz — emitted here at a
  10 Hz keyframe cadence; the FE box layer interpolates between keyframes).
* ``vehicle_pose``: per-frame 4×4 FLU .npy; pose[0] is the identity (the clip
  "world" frame IS the rig frame at frame 0) — rebased generically anyway.
* HD map JSON per clip: lanes (``polylines3d.polylines[].vertices`` — an
  edge PAIR per lane), lanelines/road_boundaries/wait_lines/poles
  (``polyline3d.vertices``), crosswalks/road_markings (``surface.vertices``),
  traffic_lights/signs (``cuboid3d.vertices`` 8 corners). Labels may be
  ``emptyLabel`` families. All metric, rig frame.
* ``captions``: one plain-text prompt per clip (``{stem}.txt``).
* NOTE several clips share one session UUID with different time windows, so
  scenario ids hash the FULL clip stem (``c-<md5(stem)[:8]>``).

Usage (phases are resumable via cache markers; see --help):

    python cosmos_drive_dreams.py --phases index                 # catalog
    python cosmos_drive_dreams.py --phases videos                # ~40 GB stream
    python cosmos_drive_dreams.py                                # everything
    python cosmos_drive_dreams.py --target-scenarios 2 --skip-videos --skip-build
                                                                 # tiny probe run

Pipeline:  HF per-clip tars → in-memory parse → combined GeoParquet →
stt-build (packed) → worlds.json sidecar + videos/ → examples/showcase.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tarfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import av_common as avc

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "examples/showcase/public/data/cosmos-drive-dreams"
DEFAULT_CACHE = Path(__file__).resolve().parent / "cosmos-raw"

HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_REPO}"
VIDEO_PART_URL = (
    f"{HF_DATASET_URL}/resolve/main/cosmos_synthetic/single_view/"
    "generation.tar.gz.part-{part:03d}"
)

# Shared fake epoch: every clip is rebased to start here so the whole grid
# animates in phase on one looping clock (plan §Context).
T0_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z
FPS = 30.0
CHUNK_FRAMES = 121  # video chunk length in source frames

WEATHERS = ("Foggy", "Golden_hour", "Morning", "Night", "Rainy", "Snowy", "Sunny")
_VIDEO_RE = re.compile(
    r"(?P<clip>[0-9a-f-]{36}(?:_\d+_\d+)?)[._-]?chunk[._-]?(?P<chunk>\d+)[._-](?P<weather>%s)\.mp4$"
    % "|".join(WEATHERS),
    re.IGNORECASE,
)
# Fallback: any "<clipid>_<chunk>_<Weather>.mp4" shape.
_VIDEO_RE2 = re.compile(
    r"(?P<clip>.+?)_(?P<chunk>\d+)_(?P<weather>%s)\.mp4$" % "|".join(WEATHERS)
)

# Modality folders: per-clip label tars. (map key → MAP_LAYERS name is below.)
LABEL_FOLDERS = ("all_object_info", "vehicle_pose")
MAP_FOLDERS = (
    "3d_lanes",
    "3d_lanelines",
    "3d_road_boundaries",
    "3d_wait_lines",
    "3d_crosswalks",
    "3d_road_markings",
    "3d_traffic_lights",
    "3d_traffic_signs",
    "3d_poles",
)

# Cosmos folder → canonical map_layer name (avc.MAP_LAYERS contract).
POLYLINE_LAYERS = {
    "3d_lanelines": "lane_boundary",
    "3d_road_boundaries": "road_boundary",
    "3d_wait_lines": "stop_line",
    "3d_poles": "pole",
}
SURFACE_LAYERS = {
    "3d_crosswalks": "crosswalk",
    "3d_road_markings": "road_marking",
}
CUBOID_LAYERS = {
    "3d_traffic_lights": "traffic_light",
    "3d_traffic_signs": "traffic_sign",
}

# Per-feature zoom FLOOR for the map polygons (`--min-zoom-field`). Every zoom
# level of an archive replicates all its features (measured: ~1 MB/level per 12
# worlds), so the gallery's overview pays for one whole level. Lane surfaces are
# the bulk of the polygons and are sub-pixel until you fly into a cell — the
# centerlines already carry the road's shape at overview — so they only exist
# from z13 up. Signals/crosswalks are small but meaningful landmarks, so they
# start one level lower.
MAP_POLY_ZOOM_FROM = {
    "lane": 13,
    "road_marking": 13,
    "crosswalk": 12,
    "traffic_light": 12,
    "traffic_sign": 12,
}


# ── small utilities ──────────────────────────────────────────────────────────


def scenario_short_id(stem: str) -> str:
    """Stable, unique, NON-NUMERIC scenario id from the full clip stem.

    Session UUIDs repeat across clips (same drive, different windows), so the
    hash covers the whole ``{uuid}_{start}_{end}`` stem. The ``c-`` prefix keeps
    the string categorical through stt-build (bare-hex could read numeric).
    """
    return "c-" + hashlib.md5(stem.encode()).hexdigest()[:8]


def _phase_marker(cache: Path, name: str) -> Path:
    return cache / f"phase_{name}.done.json"


def phase_done(cache: Path, name: str) -> bool:
    return _phase_marker(cache, name).exists()


def mark_phase(cache: Path, name: str, payload: dict | None = None) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    _phase_marker(cache, name).write_text(json.dumps(payload or {"done": True}, indent=1))


def _retry(fn, *, tries: int = 4, wait: float = 3.0, label: str = ""):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — network retries
            if attempt == tries - 1:
                raise
            print(f"    retry {label}: {type(e).__name__}: {e} (attempt {attempt + 1}/{tries})")
            time.sleep(wait * (attempt + 1))


# ── phase: index ─────────────────────────────────────────────────────────────


def build_index(cache: Path) -> dict[str, dict[str, str]]:
    """Catalog every modality folder → {clip_stem: repo_path}. Cached to disk."""
    out = cache / "index.json"
    if out.exists():
        return json.loads(out.read_text())
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    api = HfApi()
    index: dict[str, dict[str, str]] = {}
    for folder in LABEL_FOLDERS + MAP_FOLDERS + ("captions", "lidar_raw"):
        def _list(f=folder):
            return list(api.list_repo_tree(HF_REPO, path_in_repo=f, repo_type="dataset"))

        try:
            entries = _retry(_list, label=f"list {folder}")
        except GatedRepoError as e:
            raise SystemExit(
                f"Repo is gated ({e}). Run `huggingface-cli login` or set HF_TOKEN, then re-run."
            ) from e
        except HfHubHTTPError as e:
            if getattr(getattr(e, "response", None), "status_code", None) in (401, 403):
                raise SystemExit(
                    f"HF auth error listing {folder} ({e}). "
                    "Run `huggingface-cli login` or set HF_TOKEN, then re-run."
                ) from e
            raise
        stems = {}
        for e in entries:
            name = e.path.split("/")[-1]
            stem = name.removesuffix(".tar").removesuffix(".txt")
            stems[stem] = e.path
        index[folder] = stems
        print(f"  index {folder}: {len(stems)} clips")
    cache.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index))
    return index


# ── phase: videos (stream part-000, selective extract) ───────────────────────


@dataclass
class VideoInventory:
    """What the stream yielded: {clip_id: {(chunk, weather): filename}}."""

    kept: dict[str, dict[str, str]] = field(default_factory=dict)  # clip → "chunk|weather" → file
    seen_members: list[str] = field(default_factory=list)
    bytes_kept: int = 0
    bytes_read: int = 0
    truncated_at: str | None = None


def _match_video(name: str):
    base = name.rsplit("/", 1)[-1]
    m = _VIDEO_RE.search(base) or _VIDEO_RE2.search(base)
    if not m:
        return None
    weather = next(w for w in WEATHERS if w.lower() == m.group("weather").lower())
    return m.group("clip"), int(m.group("chunk")), weather


class _CountingReader(io.RawIOBase):
    """Wrap the HTTP body so we can report progress on a 40 GB stream."""

    def __init__(self, raw, inv: VideoInventory):
        self._raw = raw
        self._inv = inv
        self._last_report = 0.0

    def readable(self) -> bool:  # pragma: no cover - io plumbing
        return True

    def readinto(self, b) -> int:
        data = self._raw.read(len(b))
        n = len(data)
        b[:n] = data
        self._inv.bytes_read += n
        now = time.time()
        if now - self._last_report > 30:
            self._last_report = now
            print(
                f"    …streamed {self._inv.bytes_read / 1e9:.1f} GB raw, "
                f"kept {self._inv.bytes_kept / 1e9:.2f} GB "
                f"({sum(len(v) for v in self._inv.kept.values())} mp4s, "
                f"{len(self._inv.kept)} clips)"
            )
        return n


def stream_videos(
    cache: Path,
    videos_dir: Path,
    *,
    part: int,
    hero_candidates: int,
    keep_weathers_per_chunk: int,
    video_budget_gb: float,
    target_plus_reserve: int,
    only_stems: set[str] | None = None,
) -> VideoInventory:
    """Stream one generation.tar.gz part; selectively extract MP4s.

    part-000 is the head of one huge gzip stream (byte-split), so it
    decompresses standalone until it truncates mid-member — truncation errors
    are the NORMAL end condition, not failures.

    MEASURED (2026-07-23, part-000): the archive is NOT grouped by clip — a
    given clip's 14 variants (2 chunks × 7 weathers) are scattered across the
    whole ~695 GB stream, so one bounded read yields ~1 variant for each of
    many clips rather than many variants for a few. 8.2 GB read → 938 mp4s
    over 902 distinct clips, all 7 weathers well represented. Selection turns
    that into a weather MOSAIC across worlds (see ``select_scenarios``).

    ``only_stems`` restricts extraction to those clip stems (keeping every
    variant found) — the TOP-UP pass for enriching an already-selected set's
    weather carousels; run it over further bytes/parts when more variants are
    wanted for the chosen worlds.
    """
    import requests

    inv = VideoInventory()
    videos_dir.mkdir(parents=True, exist_ok=True)
    url = VIDEO_PART_URL.format(part=part)
    budget = int(video_budget_gb * 1e9)
    hero_clips: list[str] = []

    print(f"  streaming {url}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        raw = _CountingReader(resp.raw, inv)
        try:
            with gzip.GzipFile(fileobj=io.BufferedReader(raw, 1 << 20)) as gz:
                with tarfile.open(fileobj=gz, mode="r|") as tf:
                    for member in tf:
                        if len(inv.seen_members) < 100:
                            inv.seen_members.append(member.name)
                        if not member.isfile():
                            continue
                        hit = _match_video(member.name)
                        if hit is None:
                            continue
                        clip, chunk, weather = hit
                        if only_stems is not None:
                            # Top-up pass: only the named worlds, every variant.
                            if not any(
                                clip.startswith(s) or s.startswith(clip) for s in only_stems
                            ):
                                continue
                            variants = inv.kept.setdefault(clip, {})
                            keep = f"{chunk}|{weather}" not in variants
                        else:
                            variants = inv.kept.setdefault(clip, {})
                            if clip in hero_clips or (
                                len(hero_clips) < hero_candidates
                                and len(inv.kept) <= hero_candidates
                            ):
                                if clip not in hero_clips:
                                    hero_clips.append(clip)
                                keep = True  # hero candidates keep every variant
                            else:
                                same_chunk = sum(1 for k in variants if k.startswith(f"{chunk}|"))
                                keep = same_chunk < keep_weathers_per_chunk
                        if keep and inv.bytes_kept < budget:
                            # Named by the VIDEO clip id here; the select phase
                            # renames matched files onto label-stem scenario ids.
                            fname = f"{scenario_short_id(clip)}_{chunk}_{weather}.mp4"
                            dest = videos_dir / fname
                            with tf.extractfile(member) as src, open(dest, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            variants[f"{chunk}|{weather}"] = fname
                            inv.bytes_kept += member.size
                        if inv.bytes_kept >= budget and (
                            only_stems is not None
                            or len([c for c, v in inv.kept.items() if v]) >= target_plus_reserve
                        ):
                            print("    budget + target reached — stopping stream early")
                            break
        except (EOFError, tarfile.ReadError, gzip.BadGzipFile, OSError) as e:
            inv.truncated_at = f"{type(e).__name__}: {e}"
            print(f"    stream ended (expected for a byte-split part): {inv.truncated_at}")

    # Validate kept files; drop partials (the last member is usually truncated).
    dropped = 0
    for clip, variants in inv.kept.items():
        for key, fname in list(variants.items()):
            p = videos_dir / fname
            if not p.exists() or not _mp4_ok(p):
                p.unlink(missing_ok=True)
                variants.pop(key)
                dropped += 1
    if dropped:
        print(f"    dropped {dropped} invalid/truncated mp4(s)")
    (cache / "videos_inventory.json").write_text(
        json.dumps(
            {
                "kept": inv.kept,
                "seen_members": inv.seen_members,
                "bytes_kept": inv.bytes_kept,
                "bytes_read": inv.bytes_read,
                "truncated_at": inv.truncated_at,
            },
            indent=1,
        )
    )
    return inv


def _mp4_ok(path: Path) -> bool:
    """Cheap validity check: ffprobe when available, else ftyp+moov sniff."""
    if shutil.which("ffprobe"):
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        return r.returncode == 0 and r.stdout.strip() not in ("", "N/A")
    head = path.read_bytes()[:64]
    return b"ftyp" in head and path.stat().st_size > 100_000


def reencode_videos(videos_dir: Path, crf: int) -> None:
    """Optional web-shrink pass: H.264 CRF re-encode in place (~8.6 → ~3.5 MB)."""
    mp4s = sorted(videos_dir.glob("*.mp4"))
    print(f"  re-encoding {len(mp4s)} mp4(s) at crf {crf}")
    for i, p in enumerate(mp4s):
        tmp = p.with_suffix(".tmp.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(p), "-c:v", "libx264",
             "-crf", str(crf), "-preset", "veryfast", "-an",
             "-movflags", "+faststart", str(tmp)],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(p)
        else:
            tmp.unlink(missing_ok=True)
            print(f"    re-encode failed for {p.name}: {r.stderr.strip()[:200]}")
        if (i + 1) % 50 == 0:
            print(f"    …{i + 1}/{len(mp4s)}")


# ── phase: select ────────────────────────────────────────────────────────────


def select_scenarios(
    cache: Path,
    index: dict[str, dict[str, str]],
    videos_dir: Path,
    *,
    target: int,
    reserve: int,
    heroes: int,
    skip_videos: bool,
) -> dict:
    """Join video inventory ↔ label index; lay the chosen clips on the grid."""
    inv_path = cache / "videos_inventory.json"
    kept: dict[str, dict[str, str]] = {}
    if inv_path.exists():
        kept = json.loads(inv_path.read_text())["kept"]

    label_stems = sorted(
        set(index["all_object_info"]) & set(index["vehicle_pose"])
    )

    # Map each video clip_id onto a label stem. Video ids may be the full stem,
    # the bare session uuid, or a prefix — resolve by longest-prefix match; a
    # bare-uuid id (ambiguous across that session's clips) takes the first stem.
    by_prefix: dict[str, str] = {}
    for clip_id in kept:
        matches = [s for s in label_stems if s.startswith(clip_id) or clip_id.startswith(s)]
        if matches:
            by_prefix[clip_id] = sorted(matches)[0]

    # Re-key matched video files onto the label-stem scenario id (video clip ids
    # may be bare session uuids whose hash differs from the label stem's).
    videos_by_stem: dict[str, dict[str, str]] = {}
    for clip_id, variants in kept.items():
        stem = by_prefix.get(clip_id)
        if not (stem and variants):
            continue
        sid = scenario_short_id(stem)
        renamed = {}
        for key, fname in variants.items():
            chunk, weather = key.split("|")
            new_name = f"{sid}_{chunk}_{weather}.mp4"
            src, dst = videos_dir / fname, videos_dir / new_name
            if src.exists() and src != dst:
                src.rename(dst)
            if dst.exists():
                renamed[key] = new_name
        # Only worlds with a file ACTUALLY on disk enter the pool. Without this
        # guard a re-run adds empty entries for clips whose files an earlier
        # prune removed, and those video-less worlds displace real ones (their
        # blank weather sorts first in the mosaic interleave).
        if renamed:
            videos_by_stem.setdefault(stem, {}).update(renamed)

    if videos_by_stem and not skip_videos:
        pool = sorted(videos_by_stem)  # deterministic
        unmatched = len(kept) - len(by_prefix)
        if unmatched:
            print(f"  WARNING: {unmatched} video clip-id(s) matched no label stem")
    else:
        if not skip_videos:
            print("  WARNING: no video inventory — selecting from label stems alone")
        pool = label_stems

    # Ranked selection. The generation tar is NOT grouped by clip — each clip's
    # weather variants are scattered across the whole 695 GB stream — so one
    # part yields ~1 variant per clip with the 7 weathers spread across the
    # corpus. Take the (rare) multi-variant clips first (they carry the
    # counterfactual "same geometry, different world" panel), then fill
    # round-robin by weather so the galaxy reads as a weather MOSAIC rather
    # than whatever order the tar happened to be in.
    def primary_weather(stem: str) -> str:
        keys = sorted(videos_by_stem.get(stem, {}))
        return keys[0].split("|")[1] if keys else ""

    multi = sorted((s for s in pool if len(videos_by_stem.get(s, {})) > 1))
    singles_by_weather: dict[str, list[str]] = {}
    for s in pool:
        if s in multi:
            continue
        singles_by_weather.setdefault(primary_weather(s), []).append(s)
    for lst in singles_by_weather.values():
        lst.sort()
    interleaved: list[str] = []
    wheel = sorted(singles_by_weather)
    while any(singles_by_weather[w] for w in wheel):
        for w in wheel:
            if singles_by_weather[w]:
                interleaved.append(singles_by_weather[w].pop(0))
    chosen = (multi + interleaved)[: target + reserve]
    selected = chosen[:target]
    reserve_stems = chosen[target:]

    # Heroes: most weather variants, tie-broken by lidar availability.
    lidar_stems = set(index.get("lidar_raw", {}))
    ranked = sorted(
        selected,
        key=lambda s: (len(videos_by_stem.get(s, {})), s in lidar_stems, s),
        reverse=True,
    )
    hero_stems = [s for s in ranked if s in lidar_stems][:heroes]
    if len(hero_stems) < heroes:
        print(f"  WARNING: only {len(hero_stems)} hero(s) have lidar_raw")
    if videos_by_stem:
        wcount = Counter(primary_weather(s) for s in selected)
        print(f"  weather mix: {dict(wcount)}; {len(multi)} multi-variant clip(s)")

    # Grid layout: near-square, row-major over the (sorted) selection, heroes
    # swapped into the center cells. Anchored on the equator (cos≈1 →
    # local_to_lonlat is isotropic) in the mid-Atlantic; basemap is hidden.
    n = len(selected)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    order = [s for s in sorted(selected) if s not in hero_stems]
    cells = [(r, c) for r in range(rows) for c in range(cols)][:n]
    center_cells = sorted(cells, key=lambda rc: (abs(rc[0] - rows / 2) + abs(rc[1] - cols / 2)))
    hero_cells = center_cells[: len(hero_stems)]
    rest_cells = [rc for rc in cells if rc not in hero_cells]
    placement = dict(zip(hero_stems, hero_cells)) | dict(zip(order, rest_cells))

    # ONE generated window per world. A Cosmos video covers 121 of the clip's
    # ~297 frames, and the 35 clips that got two videos got them for DIFFERENT
    # halves (and different weathers) — not the same moment twice. So each world
    # commits to one chunk, its geometry is built for exactly that window, and
    # video↔geometry stay aligned across the whole loop. Chunk 0 wins ties.
    def pick_chunk(stem: str) -> tuple[str, dict[str, str]]:
        variants = videos_by_stem.get(stem, {})
        if not variants:
            return "0", {}
        chunks = sorted({k.split("|")[0] for k in variants})
        chunk = chunks[0]
        return chunk, {k: v for k, v in variants.items() if k.startswith(f"{chunk}|")}

    scenarios = []
    for s in selected:
        chunk, variants = pick_chunk(s)
        scenarios.append({
            "stem": s,
            "id": scenario_short_id(s),
            "row": placement[s][0],
            "col": placement[s][1],
            "hero": s in hero_stems,
            "chunk": int(chunk),
            "videos": variants,
        })
    sel = {
        "scenarios": scenarios,
        "reserve": reserve_stems,
        "grid": {"rows": rows, "cols": cols},
    }
    n_no_video = sum(1 for s in scenarios if not s["videos"])
    if n_no_video:
        print(f"  NOTE: {n_no_video} selected world(s) have no video")
    # NOTE: unselected mp4s are deliberately NOT deleted here. Pruning during
    # selection is destructive to re-selection — the deleted files are exactly
    # the pool a later re-run would draw from, so a second `select` silently
    # shrinks the gallery. The `cleanup` phase prunes instead, once the choice
    # is final.
    (cache / "selection.json").write_text(json.dumps(sel, indent=1))
    print(
        f"  selected {n} scenario(s) on a {rows}×{cols} grid "
        f"({len(hero_stems)} hero(s); {sum(1 for x in sel['scenarios'] if x['videos'])} with videos)"
    )
    return sel


# ── phase: download ──────────────────────────────────────────────────────────


def download_labels(cache: Path, index: dict, sel: dict, *, include_poles: bool) -> None:
    from huggingface_hub import hf_hub_download

    hf_cache = cache / "hf"
    folders = list(LABEL_FOLDERS) + [f for f in MAP_FOLDERS if include_poles or f != "3d_poles"]
    stems = [s["stem"] for s in sel["scenarios"]]
    hero_stems = [s["stem"] for s in sel["scenarios"] if s["hero"]]
    total = 0
    for i, stem in enumerate(stems):
        for folder in folders + ["captions"]:
            path = index.get(folder, {}).get(stem)
            if path is None:
                if folder in LABEL_FOLDERS:
                    print(f"  WARNING: {stem} missing {folder} — will need a reserve swap")
                continue
            # Short-circuit on the local cache. hf_hub_download still makes a
            # network HEAD per file even when the blob is already present, which
            # turns a resumed run into thousands of rate-limited round-trips.
            if _cached_file(cache, path) is not None:
                continue
            _retry(
                lambda p=path: hf_hub_download(
                    HF_REPO, p, repo_type="dataset", cache_dir=str(hf_cache)
                ),
                label=path,
            )
            total += 1
        if (i + 1) % 25 == 0:
            print(f"  …labels {i + 1}/{len(stems)}")
    for stem in hero_stems:
        path = index.get("lidar_raw", {}).get(stem)
        if path and _cached_file(cache, path) is None:
            print(f"  hero lidar {stem} (~350 MB)")
            _retry(
                lambda p=path: hf_hub_download(
                    HF_REPO, p, repo_type="dataset", cache_dir=str(hf_cache)
                ),
                label=path,
            )
    print(f"  downloaded/verified {total} label file(s)")


def _cached_file(cache: Path, repo_path: str) -> Path | None:
    """Resolve an hf_hub_download-cached file without touching the network."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                HF_REPO, repo_path, repo_type="dataset",
                cache_dir=str(cache / "hf"), local_files_only=True,
            )
        )
    except Exception:  # noqa: BLE001
        return None


# ── phase: transform ─────────────────────────────────────────────────────────


@dataclass
class ClipData:
    stem: str
    poses: np.ndarray  # (F, 4, 4)
    objects: dict[int, dict]  # frame_idx → {tid: record}
    map_lines: list[tuple[np.ndarray, str]]  # (V,2|3) local xy(z), layer
    map_polys: list[tuple[np.ndarray, str, float, float]]  # ring, layer, z_base, height
    caption: str


def _tar_members(path: Path) -> dict[str, bytes]:
    out = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if m.isfile():
                out[m.name.rsplit("/", 1)[-1]] = tf.extractfile(m).read()
    return out


def _shape3d_labels(blob: bytes) -> list[dict]:
    data = json.loads(blob)
    out = []
    for lab in data.get("labels", []):
        sd = (lab.get("labelData") or {}).get("shape3d")
        if sd:
            out.append(sd)
    return out


def _resample_polyline(v: np.ndarray, n: int) -> np.ndarray:
    """Arc-length resample a polyline to n vertices (for lane-edge averaging)."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(v[:, :2], axis=0), axis=1))])
    if d[-1] <= 0:
        return np.repeat(v[:1], n, axis=0)
    t = np.linspace(0.0, d[-1], n)
    return np.stack([np.interp(t, d, v[:, k]) for k in range(v.shape[1])], axis=1)


def load_clip(cache: Path, index: dict, stem: str, *, include_poles: bool) -> ClipData | None:
    """Parse one clip's cached tars into local-frame arrays. None = unusable."""
    obj_p = _cached_file(cache, index["all_object_info"].get(stem, ""))
    pose_p = _cached_file(cache, index["vehicle_pose"].get(stem, ""))
    if obj_p is None or pose_p is None:
        return None

    pose_members = _tar_members(pose_p)
    frames = sorted(pose_members)
    poses = np.stack(
        [np.load(io.BytesIO(pose_members[f])).astype("float64") for f in frames]
    )

    obj_members = _tar_members(obj_p)
    objects: dict[int, dict] = {}
    for name, blob in obj_members.items():
        m = re.search(r"\.(\d{6})\.all_object_info\.json$", name)
        if m:
            objects[int(m.group(1))] = json.loads(blob)

    map_lines: list[tuple[np.ndarray, str]] = []
    map_polys: list[tuple[np.ndarray, str, float, float]] = []

    # lanes: an edge PAIR per label → centerline (averaged, resampled) as
    # "lane_centerline" + the lane SURFACE (edge A + reversed edge B ring) as
    # the low-alpha "lane" fill.
    lanes_p = _cached_file(cache, index.get("3d_lanes", {}).get(stem, ""))
    if lanes_p:
        for sd in _shape3d_labels(next(iter(_tar_members(lanes_p).values()))):
            polys = (sd.get("polylines3d") or {}).get("polylines", [])
            edges = [np.asarray(p["vertices"], dtype="float64") for p in polys if p.get("vertices")]
            edges = [e for e in edges if len(e) >= 2]
            if len(edges) == 2:
                n = max(len(edges[0]), len(edges[1]), 2)
                a, b = _resample_polyline(edges[0], n), _resample_polyline(edges[1], n)
                map_lines.append(((a + b) / 2.0, "lane_centerline"))
                ring = np.concatenate([a, b[::-1]], axis=0)
                map_polys.append((ring, "lane", 0.0, 0.0))
            else:
                for e in edges:
                    map_lines.append((e, "lane_centerline"))

    for folder, layer in POLYLINE_LAYERS.items():
        if folder == "3d_poles" and not include_poles:
            continue
        p = _cached_file(cache, index.get(folder, {}).get(stem, ""))
        if not p:
            continue
        for sd in _shape3d_labels(next(iter(_tar_members(p).values()))):
            v = (sd.get("polyline3d") or {}).get("vertices")
            if v and len(v) >= 2:
                map_lines.append((np.asarray(v, dtype="float64"), layer))

    for folder, layer in SURFACE_LAYERS.items():
        p = _cached_file(cache, index.get(folder, {}).get(stem, ""))
        if not p:
            continue
        for sd in _shape3d_labels(next(iter(_tar_members(p).values()))):
            v = (sd.get("surface") or {}).get("vertices")
            if v and len(v) >= 3:
                map_polys.append((np.asarray(v, dtype="float64"), layer, 0.0, 0.0))

    for folder, layer in CUBOID_LAYERS.items():
        p = _cached_file(cache, index.get(folder, {}).get(stem, ""))
        if not p:
            continue
        for sd in _shape3d_labels(next(iter(_tar_members(p).values()))):
            v = (sd.get("cuboid3d") or {}).get("vertices")
            if not v or len(v) < 8:
                continue
            corners = np.asarray(v, dtype="float64")
            z0, z1 = corners[:, 2].min(), corners[:, 2].max()
            bottom = corners[np.argsort(corners[:, 2])[:4]]
            ctr = bottom[:, :2].mean(axis=0)
            ang = np.arctan2(bottom[:, 1] - ctr[1], bottom[:, 0] - ctr[0])
            ring = bottom[np.argsort(ang)]
            map_polys.append((ring, layer, float(z0), float(z1 - z0)))

    caption = ""
    cap_p = _cached_file(cache, index.get("captions", {}).get(stem, ""))
    if cap_p:
        caption = cap_p.read_text(errors="replace").strip()

    return ClipData(stem=stem, poses=poses, objects=objects,
                    map_lines=map_lines, map_polys=map_polys, caption=caption)


@dataclass
class SceneFrame:
    """The per-clip rigid transform, grid anchoring, and generated window.

    ``frame_lo`` is the first source frame of the world's generated window; all
    times are rebased on it so every world's loop starts at ``T0``.
    """

    rot: np.ndarray  # 2×2 (Rz(−yaw0) in the xy plane)
    t0: np.ndarray  # (3,) window-start pose translation
    center: np.ndarray  # (2,) local-frame scene center (subtracted before georef)
    origin_lat: float
    origin_lon: float
    frame_lo: int

    def to_lonlat(self, xy: np.ndarray):
        local = (self.rot @ (xy[:, :2] - self.t0[:2]).T).T - self.center
        return avc.local_to_lonlat(local[:, 0], local[:, 1], self.origin_lat, self.origin_lon)

    def time_ms(self, frame_idx: int) -> int:
        return T0_MS + round((frame_idx - self.frame_lo) * 1000.0 / FPS)


def scene_frame(
    clip: ClipData,
    origin_lat: float,
    origin_lon: float,
    *,
    align: bool,
    chunk: int,
) -> SceneFrame:
    """Rigid transform anchoring one clip's generated window onto its grid cell.

    Rebased on the pose at the WINDOW start (not the clip start) and centred on
    the ego bbox over the window only, so a chunk-1 world sits in the middle of
    its cell rather than trailing off the edge where the second half of the
    drive happened to go.
    """
    lo = min(chunk * CHUNK_FRAMES, len(clip.poses) - 1)
    hi = min(lo + CHUNK_FRAMES, len(clip.poses))
    m0 = clip.poses[lo]
    yaw0 = math.atan2(m0[1, 0], m0[0, 0]) if align else 0.0
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    rot = np.array([[c, -s], [s, c]])
    ego_xy = (rot @ (clip.poses[lo:hi, :2, 3] - m0[:2, 3]).T).T
    center = (ego_xy.min(axis=0) + ego_xy.max(axis=0)) / 2.0
    return SceneFrame(rot=rot, t0=m0[:3, 3].copy(), center=center,
                      origin_lat=origin_lat, origin_lon=origin_lon, frame_lo=lo)


def transform_clip(
    clip: ClipData,
    frame: SceneFrame,
    *,
    scenario_id: str,
    objects_stride: int,
    ego_stride: int,
    chunk: int,
) -> dict:
    """One clip's GENERATED WINDOW → row dicts for the combined accumulators.

    A world's geometry covers exactly the 121 source frames its Cosmos video was
    generated from (``chunk`` 0 = frames 0–120, 1 = 121–241), rebased so that
    window starts at ``T0``. That is what makes the video↔geometry claim exact:
    the loop IS the generated window, so position in the loop maps linearly onto
    position in the video with nothing left over. (A full 297-frame clip against
    a 121-frame video would leave most of the loop with no corresponding
    footage.) All coordinates come out as lon/lat.
    """
    yaw0 = math.atan2(frame.rot[1, 0], frame.rot[0, 0])  # rot = Rz(yaw0 applied)
    f_lo = frame.frame_lo
    f_hi = f_lo + CHUNK_FRAMES  # exclusive
    t_of = frame.time_ms

    # objects at a 10 Hz keyframe cadence (source is 30 Hz; the FE box layer
    # interpolates between keyframes — cadence choice, not feature thinning).
    frames = [f for f in sorted(clip.objects) if f_lo <= f < f_hi]
    kept_frames = frames[::objects_stride]
    track_pos: dict[str, list[tuple[int, float, float]]] = {}
    rows: dict[str, list] = {k: [] for k in (
        "x", "y", "t", "category", "heading", "length", "width", "height", "track_id")}
    for fi in kept_frames:
        t_ms = t_of(fi)
        for tid, rec in clip.objects[fi].items():
            m = np.asarray(rec["object_to_world"], dtype="float64")
            l, w, h = rec.get("object_lwh", (4.0, 2.0, 1.6))
            xy = frame.rot @ (m[:2, 3] - frame.t0[:2]) - frame.center
            heading = math.atan2(m[1, 0], m[0, 0]) + yaw0
            track = f"{scenario_id}/{tid.rsplit(':', 1)[-1]}"
            rows["x"].append(xy[0])
            rows["y"].append(xy[1])
            rows["t"].append(t_ms)
            rows["category"].append(avc.map_category(rec.get("object_type")))
            rows["heading"].append(heading)
            rows["length"].append(float(l))
            rows["width"].append(float(w))
            rows["height"].append(float(h))
            rows["track_id"].append(track)
            track_pos.setdefault(track, []).append((t_ms, xy[0], xy[1]))

    # per-track finite-difference speed
    speed_at: dict[tuple[str, int], float] = {}
    for track, pts in track_pos.items():
        pts.sort()
        for i, (t, x, y) in enumerate(pts):
            j = min(i + 1, len(pts) - 1)
            k = max(i - 1, 0)
            dt = (pts[j][0] - pts[k][0]) / 1000.0
            v = 0.0 if dt <= 0 else math.hypot(pts[j][1] - pts[k][1], pts[j][2] - pts[k][2]) / dt
            speed_at[(track, t)] = v
    rows["speed"] = [speed_at[(tr, t)] for tr, t in zip(rows["track_id"], rows["t"])]

    lon, lat = avc.local_to_lonlat(
        np.asarray(rows["x"]) , np.asarray(rows["y"]), frame.origin_lat, frame.origin_lon
    ) if rows["x"] else (np.array([]), np.array([]))
    n_frames = len(clip.poses)
    win_hi = min(f_hi, n_frames)  # a short clip may not fill its chunk
    duration_ms = round((win_hi - 1 - f_lo) * 1000.0 / FPS)

    # ego trip at 10 Hz vertex cadence, over the same generated window
    ego_idx = np.arange(f_lo, win_hi, ego_stride)
    if ego_idx[-1] != win_hi - 1:
        ego_idx = np.append(ego_idx, win_hi - 1)
    ego_xy_local = (frame.rot @ (clip.poses[ego_idx, :2, 3] - frame.t0[:2]).T).T - frame.center
    ego_lon, ego_lat = avc.local_to_lonlat(
        ego_xy_local[:, 0], ego_xy_local[:, 1], frame.origin_lat, frame.origin_lon
    )
    ego_t = [t_of(int(i)) for i in ego_idx]
    d = np.linalg.norm(np.diff(ego_xy_local, axis=0), axis=1)
    dt = np.maximum(np.diff(np.asarray(ego_t)) / 1000.0, 1e-6)
    seg_v = d / dt
    v = np.concatenate([seg_v[:1], seg_v])  # backfill vertex 0 with the first segment speed

    # map
    from shapely.geometry import LineString, Polygon

    lines = []
    for verts, layer in clip.map_lines:
        ml_lon, ml_lat = frame.to_lonlat(verts)
        if len(ml_lon) >= 2:
            lines.append((LineString(list(zip(ml_lon, ml_lat))), layer))
    polys = []
    poly_extras = {"z_base": [], "obj_height": [], "zoom_from": []}
    for ring, layer, z_base, height in clip.map_polys:
        mp_lon, mp_lat = frame.to_lonlat(ring)
        if len(mp_lon) < 3:
            continue
        poly = Polygon(list(zip(mp_lon, mp_lat)))
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
        polys.append((poly, layer))
        poly_extras["z_base"].append(z_base)
        poly_extras["obj_height"].append(height)
        poly_extras["zoom_from"].append(MAP_POLY_ZOOM_FROM.get(layer, 12))

    cats = Counter(rows["category"])
    large = sum(cats.get(c, 0) for c in ("truck", "bus", "trailer", "construction_vehicle"))
    return {
        "objects": {
            "lon": lon, "lat": lat, "timestamp": rows["t"],
            "category": rows["category"], "heading": rows["heading"],
            "length": rows["length"], "width": rows["width"], "height": rows["height"],
            "track_id": rows["track_id"], "speed": rows["speed"],
        },
        "ego": {
            "lon": ego_lon, "lat": ego_lat,
            "vertex_timestamps": ego_t, "vertex_values": v.tolist(),
        },
        "map_lines": lines,
        "map_polys": polys,
        "map_poly_extras": poly_extras,
        "stats": {
            "duration_ms": duration_ms,
            "agent_count": len(track_pos),
            "has_ped": int(cats.get("pedestrian", 0) > 0),
            "has_large": int(large > 0),
            "counts": dict(cats),
        },
        "caption": clip.caption,
    }


def transform_hero_lidar(
    cache: Path,
    index: dict,
    stem: str,
    frame: SceneFrame,
    out_parquet: Path,
    *,
    stride: int,
) -> int:
    """Hero LiDAR: npz sweeps → 3D point GeoParquet in the scenario's cell.

    Two things about this source are easy to get wrong (both were, and the
    result was a cloud that drifted >100 m from its own HD map by mid-clip):

    * Sweep files are numbered by **camera frame**, not by sweep ordinal —
      ``…000000.npz``, ``…000003.npz``, … ``…000294.npz`` (10 Hz LiDAR against
      30 FPS frames). The number IS the frame index; multiplying it by 3 walks
      the clip three times too fast.
    * Each npz ships its own ``lidar_to_world`` 4x4, which is authoritative and
      already includes the sensor's mount offset from the rig origin (~0.78 m
      forward, 1.94 m up on the probed clip). Lifting the points by
      ``vehicle_pose`` instead silently drops that extrinsic.

    So: read the frame index straight from the name, and place the points with
    the transform the file hands us — no pose array, no frame-convention guess.
    """
    p = _cached_file(cache, index.get("lidar_raw", {}).get(stem, ""))
    if p is None:
        print(f"  WARNING: hero {stem} lidar tar not cached — skipping")
        return 0
    members = _tar_members(p)
    lo, hi = frame.frame_lo, frame.frame_lo + CHUNK_FRAMES
    sweeps: list[tuple[int, str]] = []
    for name in members:
        if name.endswith(".npz"):
            m = re.search(r"\.(\d{6})\.", name)
            if m:
                frame_idx = int(m.group(1))
                if lo <= frame_idx < hi:
                    sweeps.append((frame_idx, name))
    sweeps.sort()
    if not sweeps:
        print(f"  WARNING: hero {stem} lidar has no sweeps in frames {lo}-{hi}")
        return 0

    all_xyz, all_t = [], []
    for idx, name in sweeps[:: max(1, stride)]:
        npz = np.load(io.BytesIO(members[name]))
        xyz = np.asarray(npz["xyz"], dtype="float64").reshape(-1, 3)
        if xyz.size == 0:
            continue
        l2w = np.asarray(npz["lidar_to_world"], dtype="float64")
        all_xyz.append((l2w[:3, :3] @ xyz.T).T + l2w[:3, 3])
        all_t.append(np.full(len(xyz), frame.time_ms(idx), dtype="int64"))
    if not all_xyz:
        return 0
    xyz = np.concatenate(all_xyz)
    t = np.concatenate(all_t)
    lon, lat = frame.to_lonlat(xyz)
    print(f"    {len(sweeps)} sweeps (frames {sweeps[0][0]}-{sweeps[-1][0]}) via lidar_to_world")
    return avc.write_lidar_points(out_parquet, lon=lon, lat=lat, timestamp=t, z=xyz[:, 2])


# ── phase: build ─────────────────────────────────────────────────────────────


def build_archives(
    parquet_dir: Path,
    out_dir: Path,
    hero_ids: list[str],
    *,
    stt_build: str,
) -> None:
    """Build every archive, clearing each target first.

    The clear matters: packs are CONTENT-ADDRESSED, so rebuilding into a
    populated directory leaves the previous build's packs behind as orphans
    that nothing references but `du` still counts (a rebuild appeared to double
    every archive until they were pruned). stt-build rewrites manifest.json but
    does not sweep the old blobs.
    """
    def fresh(sub: str) -> Path:
        target = out_dir / sub
        if target.exists():
            shutil.rmtree(target)
        return target

    avc.run_stt_build(
        parquet_dir / "objects.parquet", fresh("objects"), "point",
        stt_build=stt_build, min_zoom=13, max_zoom=16, temporal_bucket="1s",
        quantize_coords=0.1,
        quantize_attrs={"heading": 1e-3, "length": 0.01, "width": 0.01,
                        "height": 0.01, "speed": 0.01},
        blob_ordering="time-major",
    )
    avc.run_stt_build(
        parquet_dir / "ego.parquet", fresh("ego"), "trips",
        stt_build=stt_build, min_zoom=10, max_zoom=16, temporal_bucket="1s",
        blob_ordering="time-major",
    )
    avc.run_stt_build(
        parquet_dir / "map_line.parquet", fresh("map_line"), "map_line",
        stt_build=stt_build, min_zoom=10, max_zoom=16,
    )
    avc.run_stt_build(
        parquet_dir / "map_poly.parquet", fresh("map_poly"), "map_poly",
        stt_build=stt_build, min_zoom=10, max_zoom=16,
        # Lane surfaces only exist from z13 up (MAP_POLY_ZOOM_FROM) so the
        # overview doesn't pay for sub-pixel roadway fills.
        min_zoom_field="zoom_from",
    )
    # Additive-octree LOD overview points: min == max zoom field confines each
    # point to its single home_zoom tile, so the per-zoom archive IS the additive
    # pyramid (no cross-zoom replication) and the client loads only the sparse
    # coarse tier at overview. static_full_range = timeless substrate (one whole-
    # scene bucket + [start, end] validity), like the HD map.
    avc.run_stt_build(
        parquet_dir / "map_points.parquet", fresh("map_points"), "point",
        stt_build=stt_build,
        min_zoom=avc.MAP_LOD_MIN_ZOOM, max_zoom=avc.MAP_LOD_MAX_ZOOM,
        static_full_range=True,
        min_zoom_field="home_zoom", max_zoom_field="home_zoom",
        quantize_coords=0.1,
    )
    for sid in hero_ids:
        pq = parquet_dir / "heroes" / sid / "lidar.parquet"
        if pq.exists():
            avc.run_stt_build(
                pq, fresh(f"heroes/{sid}/lidar"), "point",
                stt_build=stt_build, min_zoom=14, max_zoom=18,
                temporal_bucket="100ms", quantize_coords=0.05,
                quantize_attrs=avc.lidar_quantize_attrs(0.05),
                point_elevation_column="z",
                blob_ordering="time-major",
            )


# ── main driver ──────────────────────────────────────────────────────────────


ALL_PHASES = ("index", "videos", "select", "download", "transform", "build", "sidecar", "cleanup")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--phases", default=",".join(ALL_PHASES),
                   help=f"comma list of {ALL_PHASES}")
    p.add_argument("--force-phase", default="",
                   help="comma list of phases to re-run even if marked done")
    p.add_argument("--target-scenarios", type=int, default=300)
    p.add_argument("--reserve", type=int, default=40)
    p.add_argument("--heroes", type=int, default=3)
    p.add_argument("--hero-candidates", type=int, default=8)
    p.add_argument("--video-part", type=int, default=0)
    p.add_argument("--video-budget-gb", type=float, default=6.0)
    p.add_argument("--keep-weathers-per-chunk", type=int, default=1)
    p.add_argument("--reencode-crf", type=int, default=28,
                   help="ffmpeg re-encode CRF for kept videos (-1 keeps originals)")
    p.add_argument("--skip-videos", action="store_true",
                   help="build without videos (selection falls back to label order)")
    p.add_argument("--videos-top-up", action="store_true",
                   help="videos phase: stream again keeping ONLY the already-selected "
                        "worlds' variants (enriches their weather carousels)")
    p.add_argument("--grid-pitch-m", type=float, default=1000.0)
    p.add_argument("--grid-lat", type=float, default=0.0)
    p.add_argument("--grid-lon", type=float, default=-30.0)
    p.add_argument("--objects-hz", type=float, default=10.0)
    p.add_argument("--ego-hz", type=float, default=10.0)
    p.add_argument("--no-align", dest="align", action="store_false",
                   help="don't rotate clips so ego starts heading +x/east")
    p.add_argument("--include-poles", action="store_true")
    p.add_argument("--lidar-stride", type=int, default=1)
    p.add_argument("--stt-build", default="stt-build")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--keep-raw", action="store_true")
    args = p.parse_args()

    phases = [ph.strip() for ph in args.phases.split(",") if ph.strip()]
    force = {ph.strip() for ph in args.force_phase.split(",") if ph.strip()}
    for ph in phases:
        if ph not in ALL_PHASES:
            raise SystemExit(f"unknown phase {ph!r} (valid: {ALL_PHASES})")
    cache: Path = args.cache_dir
    out: Path = args.out
    parquet_dir = cache / "parquet"

    def want(ph: str) -> bool:
        return ph in phases and (ph in force or not phase_done(cache, ph))

    index = None
    if want("index"):
        print("phase: index")
        index = build_index(cache)
        mark_phase(cache, "index")
    if any(want(ph) or ph in phases for ph in ALL_PHASES[2:]) and index is None:
        if (cache / "index.json").exists():
            index = json.loads((cache / "index.json").read_text())

    if want("videos") and not args.skip_videos:
        print("phase: videos")
        only_stems = None
        if args.videos_top_up:
            prior = json.loads((cache / "selection.json").read_text())
            only_stems = {s["stem"] for s in prior["scenarios"]}
            print(f"  top-up mode: {len(only_stems)} selected world(s)")
        inv = stream_videos(
            cache, out / "videos",
            part=args.video_part,
            hero_candidates=args.hero_candidates,
            keep_weathers_per_chunk=args.keep_weathers_per_chunk,
            video_budget_gb=args.video_budget_gb,
            target_plus_reserve=args.target_scenarios + args.reserve,
            only_stems=only_stems,
        )
        if args.reencode_crf >= 0 and shutil.which("ffmpeg"):
            reencode_videos(out / "videos", args.reencode_crf)
        mark_phase(cache, "videos", {"clips": len(inv.kept), "bytes_read": inv.bytes_read})

    sel = None
    if want("select"):
        print("phase: select")
        if index is None:
            raise SystemExit("select needs the index phase first")
        sel = select_scenarios(
            cache, index, out / "videos",
            target=args.target_scenarios, reserve=args.reserve,
            heroes=args.heroes, skip_videos=args.skip_videos,
        )
        mark_phase(cache, "select")
    if sel is None and (cache / "selection.json").exists():
        sel = json.loads((cache / "selection.json").read_text())

    if want("download"):
        print("phase: download")
        if index is None or sel is None:
            raise SystemExit("download needs index + select")
        download_labels(cache, index, sel, include_poles=args.include_poles)
        mark_phase(cache, "download")

    if want("transform"):
        print("phase: transform")
        if index is None or sel is None:
            raise SystemExit("transform needs index + select (+ download)")
        run_transform(cache, index, sel, parquet_dir, args)
        mark_phase(cache, "transform")

    if want("build") and not args.skip_build:
        print("phase: build")
        hero_ids = [s["id"] for s in (sel or {}).get("scenarios", []) if s["hero"]]
        build_archives(parquet_dir, out, hero_ids, stt_build=args.stt_build)
        mark_phase(cache, "build")

    if want("sidecar"):
        print("phase: sidecar")
        if sel is None:
            raise SystemExit("sidecar needs select/transform")
        write_worlds_json(cache, out, sel, args)
        mark_phase(cache, "sidecar")

    if want("cleanup") and not args.keep_raw:
        print("phase: cleanup")
        # Now that the selection is final, drop mp4s no world claims.
        if sel and (out / "videos").exists():
            keep = {f for s in sel["scenarios"] for f in s["videos"].values()}
            pruned = sum(
                (p.unlink() or 1) for p in (out / "videos").glob("*.mp4")
                if p.name not in keep
            )
            if pruned:
                print(f"  pruned {pruned} unselected mp4(s)")
        for sub in ("hf", "clips"):
            shutil.rmtree(cache / sub, ignore_errors=True)
        for name in ("objects", "ego", "map_line", "map_poly", "map_points"):
            (parquet_dir / f"{name}.parquet").unlink(missing_ok=True)
        subprocess.run(["du", "-sh", str(out)], check=False)
        mark_phase(cache, "cleanup")

    print("done.")


def cell_origin(row: int, col: int, grid: dict, args) -> tuple[float, float]:
    rows, cols = grid["rows"], grid["cols"]
    dlat = (row - (rows - 1) / 2.0) * args.grid_pitch_m / avc.M_PER_DEG_LAT
    dlon = (col - (cols - 1) / 2.0) * args.grid_pitch_m / (
        avc.M_PER_DEG_LAT * math.cos(math.radians(args.grid_lat))
    )
    return args.grid_lat + dlat, args.grid_lon + dlon


def run_transform(cache: Path, index: dict, sel: dict, parquet_dir: Path, args) -> None:
    objects_stride = max(1, round(FPS / args.objects_hz))
    ego_stride = max(1, round(FPS / args.ego_hz))
    clips_dir = cache / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate straight into the combined columns as each clip is parsed.
    # Holding every clip's shapely geometry and then copying it into the
    # combined lists would peak at ~2x the whole gallery's geometry in Python
    # objects (~300k LineStrings + ~180k Polygons at full scale); streaming in
    # keeps only one clip live at a time.
    FILTER_COLS = ("agent_count", "has_ped", "has_large", "weather_id")
    obj = {k: [] for k in ("lon", "lat", "timestamp", "category", "heading",
                            "length", "width", "height", "track_id", "speed")}
    obj_extras_cat: list[str] = []
    obj_extras_num = {k: [] for k in FILTER_COLS}
    trips, trip_cat, trip_num = [], [], {k: [] for k in FILTER_COLS}
    all_lines, line_cat = [], []
    all_polys, poly_cat = [], []
    poly_num = {"z_base": [], "obj_height": [], "zoom_from": []}
    # Per-scenario FILTER_COLS, replicated onto the map-LOD points below so a chip
    # hides the same worlds' road dots that it hides on the ego trips / boxes.
    scen_filter: dict[str, dict] = {}
    summaries: list[dict] = []
    durations: list[int] = []

    reserve = list(sel.get("reserve", []))
    scen_list = list(sel["scenarios"])
    for i, scen in enumerate(scen_list):
        stem, sid = scen["stem"], scen["id"]
        frag = clips_dir / f"{sid}.json"
        clip = load_clip(cache, index, stem, include_poles=args.include_poles)
        while clip is None and reserve:
            swap = reserve.pop(0)
            print(f"  {stem} unusable — swapping in reserve {swap}")
            scen["stem"], scen["id"] = swap, scenario_short_id(swap)
            scen["videos"] = {}
            stem, sid = scen["stem"], scen["id"]
            clip = load_clip(cache, index, swap, include_poles=args.include_poles)
        if clip is None:
            print(f"  {stem} unusable, no reserve left — dropping")
            continue
        origin_lat, origin_lon = cell_origin(scen["row"], scen["col"], sel["grid"], args)
        chunk = int(scen.get("chunk", 0))
        frame = scene_frame(clip, origin_lat, origin_lon, align=args.align, chunk=chunk)
        rec = transform_clip(
            clip, frame, scenario_id=sid,
            objects_stride=objects_stride, ego_stride=ego_stride, chunk=chunk,
        )
        rec["origin"] = {"lat": origin_lat, "lon": origin_lon}
        st = rec["stats"]
        # One generated window per world ⇒ one manifestation. Baked as a numeric
        # index into WEATHERS (-1 = none) so a chip can isolate "every rainy
        # world" on the GPU.
        weathers_here = {k.split("|")[1] for k in scen["videos"]} if scen["videos"] else set()
        st["weather_count"] = len(weathers_here)
        primary = sorted(weathers_here)[0] if weathers_here else None
        st["weather_id"] = WEATHERS.index(primary) if primary in WEATHERS else -1
        scen_filter[sid] = {k: st[k] for k in FILTER_COLS}

        n = len(rec["objects"]["lon"])
        for k in obj:
            obj[k].extend(np.asarray(rec["objects"][k]).tolist())
        obj_extras_cat.extend([sid] * n)
        for k in obj_extras_num:
            obj_extras_num[k].extend([st[k]] * n)
        trips.append(rec["ego"])
        trip_cat.append(sid)
        for k in trip_num:
            trip_num[k].append(st[k])
        for line, layer in rec["map_lines"]:
            all_lines.append((line, layer))
            line_cat.append(sid)
        for i_p, (poly, layer) in enumerate(rec["map_polys"]):
            all_polys.append((poly, layer))
            poly_cat.append(sid)
            for key in poly_num:
                poly_num[key].append(rec["map_poly_extras"][key][i_p])
        durations.append(st["duration_ms"])
        summaries.append({
            **scen, "stats": st, "origin": rec["origin"], "caption": rec["caption"],
        })

        if scen["hero"]:
            transform_hero_lidar(
                cache, index, stem, frame,
                parquet_dir / "heroes" / sid / "lidar.parquet",
                stride=args.lidar_stride,
            )
        frag.write_text(json.dumps({"stats": rec["stats"], "origin": rec["origin"]}))
        if (i + 1) % 25 == 0:
            print(f"  …transform {i + 1}/{len(scen_list)}")

    if not summaries:
        raise SystemExit(
            "transform: no usable clips — check the download phase completed "
            "(cache/hf) and that selection.json names clips present in the index"
        )
    parquet_dir.mkdir(parents=True, exist_ok=True)
    duration_max = max(durations)
    t_end = T0_MS + duration_max

    avc.write_objects_points(
        parquet_dir / "objects.parquet",
        lon=obj["lon"], lat=obj["lat"], timestamp=obj["timestamp"],
        category=obj["category"], heading=obj["heading"], length=obj["length"],
        width=obj["width"], height=obj["height"], track_id=obj["track_id"],
        speed=obj["speed"],
        extra_categorical={"scenario_id": obj_extras_cat},
        extra_numeric=obj_extras_num,
    )
    avc.write_trips_multi(
        parquet_dir / "ego.parquet", trips,
        extra_categorical={"scenario_id": trip_cat}, extra_numeric=trip_num,
    )
    avc.write_map_lines(
        parquet_dir / "map_line.parquet", all_lines, T0_MS, t_end,
        extra_categorical={"scenario_id": line_cat},
    )
    # Additive-octree LOD overview: a decimated dotted road network sampled from
    # the lane lines, so the gallery overview loads a sparse coarse tier
    # (map_points, lodMode:'additive') instead of the full ~100 MB map_line
    # archive at every zoom. The crisp lines take over on zoom-in. Each point
    # carries the same FILTER_COLS as the lines/trips so a chip filters it too.
    mp_lon, mp_lat, mp_home, mp_src = avc.sample_map_lod_points(
        [ln for ln, _ in all_lines], anchor_lat=args.grid_lat,
    )
    line_layer = np.array([lyr for _, lyr in all_lines], dtype=object)
    line_num = {k: np.array([scen_filter[s][k] for s in line_cat]) for k in FILTER_COLS}
    avc.write_map_points(
        parquet_dir / "map_points.parquet",
        lon=mp_lon, lat=mp_lat, map_layer=line_layer[mp_src], home_zoom=mp_home,
        t_start=T0_MS, t_end=t_end,
        extra_numeric={k: line_num[k][mp_src] for k in FILTER_COLS},
    )
    avc.write_map_polygons(
        parquet_dir / "map_poly.parquet", all_polys, T0_MS, t_end,
        extra_categorical={"scenario_id": poly_cat}, extra_numeric=poly_num,
    )
    (cache / "transform_summary.json").write_text(json.dumps({
        "scenarios": summaries,
        "duration_max_ms": duration_max,
    }, indent=1))


def write_worlds_json(cache: Path, out: Path, sel: dict, args) -> None:
    """Write the scenario index the `/worlds` page reads.

    ``durationMs`` is the loop length = the generated window (121 frames), so
    position in the loop maps 1:1 onto position in every world's video.
    """
    summary = json.loads((cache / "transform_summary.json").read_text())
    duration = summary["duration_max_ms"]
    chunk_ms = round(CHUNK_FRAMES * 1000.0 / FPS)
    scenarios = []
    for s in summary["scenarios"]:
        origin = s["origin"]
        lon, lat = avc.local_to_lonlat(0.0, 0.0, origin["lat"], origin["lon"])
        videos: dict[str, dict[str, str]] = {}
        for key, fname in s.get("videos", {}).items():
            chunk, weather = key.split("|")
            if (out / "videos" / fname).exists():
                videos.setdefault(chunk, {})[weather] = f"videos/{fname}"
        st = s["stats"]
        # This world's Cosmos manifestation (the weather its generated video was
        # rendered in) — the gallery tints / filters by it.
        weather = ""
        for ck in sorted(videos):
            for w in sorted(videos[ck]):
                weather = weather or w
        weather_id = WEATHERS.index(weather) if weather in WEATHERS else -1
        entry = {
            "id": s["id"],
            "clip": s["stem"],
            "chunk": int(s.get("chunk", 0)),
            "row": s["row"], "col": s["col"],
            "origin": {"lon": float(np.asarray(lon)), "lat": float(np.asarray(lat))},
            "hero": bool(s["hero"]),
            "timeRange": {"start": T0_MS, "end": T0_MS + st["duration_ms"]},
            "agentCount": st["agent_count"],
            "hasPed": bool(st["has_ped"]),
            "hasLarge": bool(st["has_large"]),
            "weather": weather,
            "weatherId": weather_id,
            "counts": st["counts"],
            "caption": (s.get("caption") or "")[:280],
            "videos": videos,
        }
        if s["hero"] and (out / "heroes" / s["id"] / "lidar" / "manifest.json").exists():
            entry["lidar"] = f"heroes/{s['id']}/lidar/manifest.json"
        scenarios.append(entry)

    doc = {
        "id": "cosmos-drive-dreams",
        "dataset": "NVIDIA PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams",
        "datasetUrl": HF_DATASET_URL,
        "license": "CC-BY-4.0",
        "attribution": "Driving scenarios and Cosmos-generated videos © NVIDIA, CC BY 4.0",
        "t0": T0_MS,
        "durationMs": duration,
        "chunkMs": chunk_ms,
        "videoFrames": CHUNK_FRAMES,
        "grid": {
            "rows": sel["grid"]["rows"], "cols": sel["grid"]["cols"],
            "pitchM": args.grid_pitch_m, "lat": args.grid_lat, "lon": args.grid_lon,
        },
        "weathers": list(WEATHERS),
        "scenarios": scenarios,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "worlds.json").write_text(json.dumps(doc, indent=1))
    print(f"  wrote {len(scenarios)} scenario(s) → {out / 'worlds.json'}")


if __name__ == "__main__":
    main()
