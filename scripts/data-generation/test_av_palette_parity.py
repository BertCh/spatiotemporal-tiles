#!/usr/bin/env python3
"""Python↔TS AV-palette parity guard (naming-types-consistency-2026-06 F1).

The AV cockpit color palettes live in TWO hand-maintained copies:

    Python  scripts/data-generation/av_common.py   (baked into scene.json / tiles)
    TS      examples/showcase/src/datasets.ts       (rendered geometry + legend)

Those copies are kept in lockstep only by comments (``# MUST stay in lockstep``),
and the old dead map-color copy had *already* silently drifted in hue between the
two before it was reduced to the ``MAP_LAYERS`` key-set.
This test mechanically enforces the contract so the next drift fails CI instead of
shipping desynced swatches.

Scope (deliberate — see the audit's "Risks / invariants"):

* VALUE-LOCKED (key-set **and** RGBA per key asserted identical):
  ``OBJECT_COLORS`` ⇄ ``AV_OBJECT_COLORS``,
  ``LIDARSEG_COLORS`` ⇄ ``AV_LIDARSEG_COLORS``,
  ``HEIGHT_BAND_COLORS`` ⇄ ``AV_HEIGHT_BAND_COLORS``.
  The TS ``AV_OBJECT_COLORS`` carries one documented render-only extra key
  (``ego``, the synthetic ego-track color) that is explicitly NOT part of the
  Python dual copy — it is allowed as a TS-only addition, everything else must
  match exactly.
* KEY-ONLY: ``MAP_LAYERS`` (Python) ⇄ ``AV_MAP_COLORS`` keys (TS) — the
  ``map_layer`` name set is the real contract (it validates the strings the
  extractors emit). The map colors live only on the TS side; the Python copy was
  reduced to a key-set (``write_scene_json`` never emitted map colors), so only
  the key-sets are compared. Likewise ``ISO_DENSITY_BANDS`` (Python band labels)
  ⇄ ``AV_ISO_DENSITY_COLORS`` keys — the colors are TS-only, the ordered label
  set is the contract.
* LEGEND CONSISTENCY (TS-internal): the hand-derived hex swatches in
  ``AV_HEIGHT_BAND_LEGEND`` / ``AV_ISO_DENSITY_LEGEND`` must be the hex of their
  RGBA source palettes (``AV_HEIGHT_BAND_COLORS`` / ``AV_ISO_DENSITY_COLORS``) —
  the third hand-copy layer that had already drifted (#28a8a8 vs #26a8a8).

The TS side is parsed textually (regex, no TS runtime): robust to whitespace,
trailing commas, bare/quoted keys, and trailing ``// comments``.

Runs standalone (``venv-*/bin/python test_av_palette_parity.py``) — also
pytest-discoverable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import av_common as avc

# ── locate the TS Dataset registry (repo-root/examples/showcase/src/datasets.ts) ──
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASETS_TS = _REPO_ROOT / "examples" / "showcase" / "src" / "datasets.ts"

Rgba = tuple[int, int, int, int]

# One palette entry: a key (bare identifier, or 'single'/"double" quoted — the
# height-band keys like '<-2' / '0-2' must be quoted) → [r, g, b, a]. Trailing
# comma inside the array and a trailing // comment after the ``]`` are tolerated.
_ENTRY_RE = re.compile(
    r"""(?:'(?P<qk>[^']+)'          # 'single-quoted key'
         |"(?P<dqk>[^"]+)"          # "double-quoted key"
         |(?P<bk>[A-Za-z_$][\w$]*)) # bare identifier key
        \s*:\s*
        \[\s*(?P<r>\d+)\s*,\s*(?P<g>\d+)\s*,\s*(?P<b>\d+)\s*,\s*(?P<a>\d+)\s*,?\s*\]
    """,
    re.VERBOSE,
)


def _parse_ts_palette(const_name: str) -> dict[str, Rgba]:
    """Extract a ``const <const_name>: Record<string, ColorRGBA> = { … };`` block
    from ``datasets.ts`` into ``{key: (r, g, b, a)}``.

    The block body contains only ``[…]`` arrays (no nested ``{}``), so the first
    ``};`` reliably terminates it.
    """
    assert _DATASETS_TS.is_file(), f"datasets.ts not found at {_DATASETS_TS}"
    src = _DATASETS_TS.read_text()
    m = re.search(
        r"const\s+" + re.escape(const_name) + r"\b[^=]*=\s*\{(?P<body>.*?)\};",
        src,
        re.DOTALL,
    )
    assert m, f"could not locate `const {const_name}` block in {_DATASETS_TS}"
    out: dict[str, Rgba] = {}
    for em in _ENTRY_RE.finditer(m.group("body")):
        key = em.group("qk") or em.group("dqk") or em.group("bk")
        out[key] = (
            int(em.group("r")),
            int(em.group("g")),
            int(em.group("b")),
            int(em.group("a")),
        )
    assert out, f"parsed zero entries from `{const_name}` — parser/format drift?"
    return out


# One legend item: `{ color: '#rrggbb', label: '…' }` (order within the object
# is fixed in datasets.ts; a trailing comma / // comment is tolerated).
_LEGEND_ITEM_RE = re.compile(
    r"\{\s*color:\s*['\"](?P<hex>#[0-9a-fA-F]{6})['\"]\s*,\s*"
    r"label:\s*['\"](?P<label>[^'\"]+)['\"]\s*,?\s*\}"
)


def _parse_ts_legend(const_name: str) -> list[tuple[str, str]]:
    """Extract a ``const <const_name>: DatasetLegend = { … };`` block from
    ``datasets.ts`` into an ordered ``[(hex_lower, label), …]`` item list."""
    src = _DATASETS_TS.read_text()
    m = re.search(
        r"const\s+" + re.escape(const_name) + r"\b[^=]*=\s*\{(?P<body>.*?)\};",
        src,
        re.DOTALL,
    )
    assert m, f"could not locate `const {const_name}` block in {_DATASETS_TS}"
    items = [
        (im.group("hex").lower(), im.group("label"))
        for im in _LEGEND_ITEM_RE.finditer(m.group("body"))
    ]
    assert items, f"parsed zero legend items from `{const_name}` — format drift?"
    return items


def _as_rgba(v) -> Rgba:
    t = tuple(int(x) for x in v)
    assert len(t) == 4, f"expected RGBA (4 ints), got {v!r}"
    return t  # type: ignore[return-value]


def _hex_of(rgba: Rgba) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgba[:3])


def _assert_value_parity(
    name: str,
    py: dict[str, list[int]],
    ts_const: str,
    *,
    ts_only_keys: frozenset[str] = frozenset(),
) -> None:
    """Assert identical key-sets (modulo documented TS-only extras) AND identical
    RGBA per shared key between a Python palette and its ``datasets.ts`` twin."""
    ts = _parse_ts_palette(ts_const)
    py_keys = set(py)
    ts_keys = set(ts)

    missing_in_ts = py_keys - ts_keys
    extra_in_ts = ts_keys - py_keys - ts_only_keys
    assert not missing_in_ts, (
        f"{name}: keys present in Python av_common but MISSING from TS "
        f"{ts_const}: {sorted(missing_in_ts)}"
    )
    assert not extra_in_ts, (
        f"{name}: keys present in TS {ts_const} but MISSING from Python "
        f"av_common (and not a documented TS-only key): {sorted(extra_in_ts)}"
    )

    mismatches = {
        k: (_as_rgba(py[k]), ts[k]) for k in py_keys if _as_rgba(py[k]) != ts[k]
    }
    assert not mismatches, (
        f"{name}: RGBA value drift Python⇄TS {ts_const} "
        f"(key: python vs ts): {mismatches}"
    )


# ── VALUE-LOCKED palettes ────────────────────────────────────────────────────
def test_object_colors_value_parity():
    # `ego` is a render-only synthetic track color in TS, explicitly NOT part of
    # the Python OBJECT_COLORS dual copy (see datasets.ts comment ~line 111).
    _assert_value_parity(
        "OBJECT_COLORS",
        avc.OBJECT_COLORS,
        "AV_OBJECT_COLORS",
        ts_only_keys=frozenset({"ego"}),
    )


def test_lidarseg_colors_value_parity():
    _assert_value_parity(
        "LIDARSEG_COLORS", avc.LIDARSEG_COLORS, "AV_LIDARSEG_COLORS"
    )


def test_height_band_colors_value_parity():
    _assert_value_parity(
        "HEIGHT_BAND_COLORS", avc.HEIGHT_BAND_COLORS, "AV_HEIGHT_BAND_COLORS"
    )


# ── KEY-ONLY contract: MAP_LAYERS (the Python side is a frozenset of valid
# `map_layer` names; colors live only in the TS AV_MAP_COLORS) ────────────────
def test_map_layers_key_parity():
    # Python `MAP_LAYERS` is the valid map_layer name set; the TS `AV_MAP_COLORS`
    # keys must match it exactly (colors are TS-only). `set(...)` reads either a
    # frozenset or a dict, so this stays robust.
    py_keys = set(avc.MAP_LAYERS)
    ts_keys = set(_parse_ts_palette("AV_MAP_COLORS"))
    assert py_keys == ts_keys, (
        "MAP_LAYERS map_layer KEY drift Python(MAP_LAYERS)⇄TS(AV_MAP_COLORS) — "
        f"only-in-python={sorted(py_keys - ts_keys)}, "
        f"only-in-ts={sorted(ts_keys - py_keys)}"
    )


# ── ISO density: ordered band labels are the Python contract; colors are TS-only,
# so the key ORDER is locked to ISO_DENSITY_BANDS and the legend hexes to the
# rendered AV_ISO_DENSITY_COLORS ramp ────────────────────────────────────────
def test_iso_density_band_parity():
    ts = _parse_ts_palette("AV_ISO_DENSITY_COLORS")
    assert tuple(ts) == tuple(avc.ISO_DENSITY_BANDS), (
        "ISO density band drift Python(ISO_DENSITY_BANDS)⇄TS(AV_ISO_DENSITY_COLORS "
        f"keys, ordered): {tuple(avc.ISO_DENSITY_BANDS)} vs {tuple(ts)}"
    )
    legend = _parse_ts_legend("AV_ISO_DENSITY_LEGEND")
    ramp = [_hex_of(rgba) for rgba in ts.values()]
    legend_hex = [h for h, _label in legend]
    assert legend_hex == ramp, (
        "AV_ISO_DENSITY_LEGEND hex swatches must be the hex of the "
        f"AV_ISO_DENSITY_COLORS ramp, in band order: legend={legend_hex} "
        f"ramp={ramp}"
    )


# ── Legend consistency: the hand-derived AV_HEIGHT_BAND_LEGEND hexes must be
# hexes of AV_HEIGHT_BAND_COLORS entries (the legend shows a labelled SUBSET of
# the 8 bands, so membership — not order/count — is the contract) ─────────────
def test_height_band_legend_hex_consistency():
    colors = _parse_ts_palette("AV_HEIGHT_BAND_COLORS")
    valid_hex = {_hex_of(rgba) for rgba in colors.values()}
    legend = _parse_ts_legend("AV_HEIGHT_BAND_LEGEND")
    drifted = [(h, label) for h, label in legend if h not in valid_hex]
    assert not drifted, (
        "AV_HEIGHT_BAND_LEGEND hex swatch(es) not derived from any "
        f"AV_HEIGHT_BAND_COLORS RGBA: {drifted} (valid: {sorted(valid_hex)})"
    )


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"All {len(fns)} AV-palette parity tests passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_run_all())
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
