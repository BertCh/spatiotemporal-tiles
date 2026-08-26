#!/usr/bin/env python3
"""Neural-State Atlas — a transformer's internal state as a spatiotemporal tileset.

Implements docs/roadmap/neural-atlas-2026-07.md. The record is normative; this
file is the generator half of §5 ("no new packages — a sibling of
cosmos_drive_dreams.py"), and the section numbers in the comments below cite it.

WHAT IT BUILDS (four archives, §5.2 + the geometry note below)

    neural-atlas-anatomy       POINT    the frozen semantic geography.
                                        X/Y/Z = one isotropic embedding, with Z
                                        riding the `z_embed_m` COLUMN (never
                                        --point-elevation-column, so flat↔3D is
                                        a renderer prop — see §14.9 and the note
                                        above `_pca`), T = the whole
                                        trace interval (§4.3 "feature intervals"),
                                        zoom band = the cluster tree
                                        (--min-zoom-field/--max-zoom-field).
                                        --blob-ordering spatial.
    neural-atlas-regions       POLYGON  L0 region + L1 family hulls, the labelled
                                        continents. Extruded by aggregate metric.
    neural-atlas-manifolds     PATH     §2.2 made visible: for each ordinal concept
                                        probe (digits, weekdays, months) a polyline
                                        through that concept's top features IN
                                        CONCEPT ORDER. A concept that is continuous
                                        in the model but shattered by the discrete
                                        basis draws a scribble across the map. This
                                        is a *known* ordinal structure drawn as a
                                        locus, NOT discovered manifold geometry —
                                        Milestone 7 is still open.
    neural-atlas-trace-<slug>  POINT    the playback demo. One row per
                                        (token, layer, active feature).
                                        --blob-ordering time-major, NON-NEGOTIABLE
                                        (§5.2; `auto` on a multi-cell playback
                                        dataset is the known stall).

Four archives rather than the record's single multi-geometry one: stt-build takes
one geometry kind per invocation. The record's requirement was that adding a
geometry kind later must not be a rebuild of the *point* archive — satisfied,
because the curve and surface kinds live in their own archives from day one and
the anatomy's content addresses never move when they change.

THE PIN — AND WHY IT IS NOT THE RECORD'S PIN (§8)

The record pins `gemma-2-2b` + Gemma Scope. That pin is not reachable from this
environment and the reasons are hard blockers, not preferences:

  * `google/gemma-2-2b` is `gated: manual` on the Hub (checked 2026-07-27) and
    this machine has no Hub token — the weights cannot be fetched at all.
  * 26 layers x 16k Gemma Scope SAEs is ~15.7 GB of SAE weights alone against
    42 GB of free disk on a tree whose showcase data directory is already 64 GB.
  * No CUDA (§10 G3): M3 Pro / 36 GB / MPS.

So this ships on the nearest UNGATED intersection of the same four artefacts:

  * model      openai-community/gpt2        (MIT, ungated, 124M, ~0.5 GB)
  * SAEs       jbloom/GPT2-Small-SAEs-Reformatted  (MIT, ungated), the 12
               `blocks.{0..11}.hook_resid_pre` residual SAEs, d_sae = 24576
  * labels     Neuronpedia `gpt2-small/{0..11}-res-jb` explanation exports
               (S3 bulk export; the /api/explanation/export endpoint was retired)
  * corpus     Salesforce/wikitext, wikitext-103-raw-v1 (CC-BY-SA-3.0 + GFDL)

That is 12 x 24576 = 294,912 atlas nodes — the record's §4.1 "atlas anatomy" row
(416 k) at 0.7x, and the same order as the shipped `earthquakes-v2` (~522 k). The
scale gate is cleared: this is not a demo the format is decoration on.

`blocks.11.hook_resid_post` is deliberately EXCLUDED even though the SAE exists —
Neuronpedia carries no `11-res-post-jb` labels, and a 13th layer of unlabelled
nodes buys 8% more points at the cost of a hole in the interpretation layer.

Every model-specific fact is in MODEL_PINS, so moving to Gemma later is a config
entry plus a Hub token, not a rewrite.

THE CONTEXT-LENGTH CLIFF (measured here, and the reason --seq-len is not a knob)

These SAEs were trained at `context_size: 128`. Run them on longer sequences and
they fall off a cliff — measured on this machine over the same wikitext tokens,
centred basis, position 0 excluded:

    window   layer 0            layer 6             layer 11
    128      L0 15.0  FVU 0.034  L0 56.2  FVU 0.159  L0 62.8  FVU 0.238
    512      L0 639.4 FVU 7.27   L0 179.1 FVU 3.94   L0 140.3 FVU 1.91

At 512 the reconstruction is several times WORSE than predicting the mean and the
dictionary fires 40x as many latents — a co-activation graph built there would be
measuring the residual stream drifting out of distribution, not the model's
features. Prepending BOS changes nothing (measured: L0 15.0 either way); the
window length is the whole effect. So every forward pass in this pipeline runs at
`pin.sae_ctx`, statistics and trace alike, and the trace's "reading session" is a
run of consecutive 128-token windows rather than one long one. The 128-at-a-time
numbers above are the published L0/FVU for `gpt2-small-res-jb`, which is the
check that the basis and the loader are both right.

RESIDUAL-STREAM BASIS (the other numerical trap)

The SAEs were trained on TransformerLens `blocks.N.hook_resid_pre`, and
TransformerLens loads GPT-2 with `center_writing_weights=True`: every write into
the residual stream is made mean-zero along d_model, so the TL stream is the HF
stream with its per-token d_model mean removed. LayerNorm makes this a behavioural
no-op for the model but NOT for an SAE reading raw residuals. HF hidden_states are
therefore centred before encoding. The `stats` stage does not take this on faith —
it measures normalized reconstruction MSE and L0 both ways on a probe batch and
records the winner in the run manifest (see `_calibrate_basis`).

STAGES (each cached under --work-dir, each recording its config hash, §7)

    corpus     wikitext -> packed uint16 token windows (train = stats,
               validation = trace; disjoint by construction)
    stats      GPT-2 forward -> 12 SAE encodes -> per-feature firing rate, mean
               and max activation, concept-probe affinities, and a co-activation
               pair sample
    graph      multiplex kNN: decoder-direction cosine (cross-layer, the residual
               basis is shared) UNION within-layer co-activation Jaccard
    cluster    Leiden on the multiplex graph -> micro-communities -> Ward on the
               community centroids, cut twice for a guaranteed-nested
               L0 region / L1 family tree
    layout     §6.3 in order: L0 packed and FROZEN, L1 inside its parent and
               FROZEN, L2 inside its family. Clustering is on the HIGH-DIMENSIONAL
               graph, never on the 2-D projection.
    labels     Neuronpedia explanations -> label, labelConfidence (mean pairwise
               cosine of the shipped explanation embeddings) and the §3
               interpretationStatus enum. Bought, not built (§7).
    trace      a reading session over held-out text -> top-K active features per
               (token, layer), plus grad x activation attribution
    validate   §9: reseed drift, projection distortion, manifold-shattering audit
    pack       GeoParquet -> stt-build x4

    python3 neural_atlas.py --stages all
    python3 neural_atlas.py --stages pack --force        # re-cut the archives only

COMPUTE ENVELOPE (measured on M3 Pro / 36 GB / MPS, defaults as shipped)

    corpus   ~3 min      stats  ~12 min     graph  ~6 min     cluster  ~3 min
    layout   ~2 min      labels ~4 min      trace  ~1 min     pack     ~4 min

Raise --stats-tokens / --trace-windows for a bigger run; the graph stage is the
memory ceiling (--knn-chunk rows x n_nodes fp16 on the GPU at once).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av_common import local_to_lonlat  # noqa: E402  (sibling generator helper)

# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPin:
    """Everything model-specific. Swapping pins is a new entry, not a rewrite."""

    key: str
    hf_model: str
    model_license: str
    sae_repo: str
    sae_license: str
    #: (layer_index, sae_subdirectory, neuronpedia_source_id)
    sae_layers: tuple[tuple[int, str, str], ...]
    d_model: int
    d_sae: int
    #: The model's context window.
    n_ctx: int
    #: The context length the SAEs were TRAINED at, and the only length at which
    #: they are in distribution. See the OOD-cliff note in the module docstring —
    #: this is not a tuning knob, it is a property of the dictionary.
    sae_ctx: int
    neuronpedia_model: str


MODEL_PINS: dict[str, ModelPin] = {
    "gpt2-small-resjb": ModelPin(
        key="gpt2-small-resjb",
        hf_model="openai-community/gpt2",
        model_license="MIT",
        sae_repo="jbloom/GPT2-Small-SAEs-Reformatted",
        sae_license="MIT",
        sae_layers=tuple(
            (i, f"blocks.{i}.hook_resid_pre", f"{i}-res-jb") for i in range(12)
        ),
        d_model=768,
        d_sae=24576,
        n_ctx=1024,
        sae_ctx=128,  # cfg.json context_size for every blocks.N.hook_resid_pre SAE
        neuronpedia_model="gpt2-small",
    ),
}

CORPUS = {
    "repo": "Salesforce/wikitext",
    "config": "wikitext-103-raw-v1",
    "license": "CC-BY-SA-3.0 + GFDL",
    "url": "https://huggingface.co/datasets/Salesforce/wikitext",
}

NEURONPEDIA_S3 = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"

ATTRIBUTION = (
    "Model: openai-community/gpt2 (MIT). SAEs: jbloom/GPT2-Small-SAEs-Reformatted "
    "(MIT). Feature labels: Neuronpedia explanation exports (gpt2-small res-jb). "
    "Corpus: wikitext-103-raw-v1, CC-BY-SA-3.0 + GFDL, (c) Wikipedia contributors."
)

# ---------------------------------------------------------------------------
# Atlas coordinate frame (§4.2)
# ---------------------------------------------------------------------------

#: The atlas plane is centred on (0, 0) so cos(lat) == 1 exactly at the origin and
#: the equirectangular mapping is isotropic where it matters. The half-extent is
#: 16 deg, inside the record's "keep the atlas within roughly +/-20 deg of the
#: equator" build constant with margin for the hull outlines.
ATLAS_ORIGIN_LAT = 0.0
ATLAS_ORIGIN_LON = 0.0
ATLAS_HALF_DEG = 16.0
M_PER_DEG_LAT = 111_320.0
ATLAS_HALF_M = ATLAS_HALF_DEG * M_PER_DEG_LAT  # 1,781,120 m

#: Z is NO LONGER the transformer layer (§14.9). The layer index was folded into
#: point altitude at 150 km per layer, which put a 1,650 km stack against
#: families of median radius 0.7 km — needles of aspect ratio ~1,250:1, and the
#: reason the first build rendered as towers in a void. Depth is now the third
#: component of the atlas embedding, isotropic with X/Y, and the transformer
#: layer is carried as colour and as the layer x token series instead.
#:
#: Kept only so an archive built before the change can still be described.
LAYER_SPACING_M = 150_000.0

#: T (§4.2): one token = one second from a synthetic epoch. 2020-01-01T00:00:00Z.
TRACE_EPOCH_MS = 1_577_836_800_000
MS_PER_TOKEN = 1_000

#: Zoom bands (§4.3) as a CUMULATIVE node budget, one entry per zoom from 0.
#:
#: The first build banded 28 / 672 / 294,212, and the consequence was that there
#: was no camera position from which the atlas was a map: zoomed out you saw 28
#: dots, and the zoom where the latents finally mounted showed 8 deg of a 26 deg
#: plane. A budget ladder makes the reveal a progressive densification instead
#: of a cliff, and ranking inside the FAMILY (see `_lod_bands`) keeps the shape
#: of the whole map present at every step.
LOD_BUDGET = (1_500, 3_000, 6_000, 12_000, 24_000, 48_000, 96_000)
#: Past the ladder every node is resident.
Z_FEATURE = len(LOD_BUDGET)
#: Hull bands: regions carry the low zooms, families hand off at Z_FAMILY_REP.
Z_REGION_REP = 0
Z_FAMILY_REP = 5
#: Deepest zoom the ANATOMY is tiled to. Sized from the tile budget, not taste:
#: the atlas is a 32-degree-wide box, so z10 is ~4,100 occupied tiles at ~72
#: latents each — deep enough that neighbouring latents separate on screen,
#: shallow enough that the archive is not a long tail of two-point tiles.
Z_MAX = 10
#: The TRACE is tiled two levels shallower. Its tile count is spatial x temporal
#: (a 2-minute bucket over a 8,192-token session is ~68 buckets), so z10 would
#: multiply out to ~280k mostly-empty tiles; z8 lands ~36k at ~90 events each.
Z_TRACE_MAX = 8

# ---------------------------------------------------------------------------
# Concept probes (§9.2 manifold-shattering audit, and the manifolds archive)
# ---------------------------------------------------------------------------

#: Ordinal concepts with KNOWN continuous or cyclic structure in the model
#: (Goodfire, _The World Inside Neural Networks_ — numbers, days and months appear
#: as circular loops). Each entry is (probe_name, cyclic?, ordered member tokens).
#: The audit asks how many latents carry each concept and how far apart on the
#: atlas they land; the manifolds archive draws the answer.
CONCEPT_PROBES: tuple[tuple[str, bool, tuple[str, ...]], ...] = (
    (
        "digits",
        False,
        tuple(f" {d}" for d in range(10)),
    ),
    (
        "weekdays",
        True,
        (
            " Monday",
            " Tuesday",
            " Wednesday",
            " Thursday",
            " Friday",
            " Saturday",
            " Sunday",
        ),
    ),
    (
        "months",
        True,
        (
            " January",
            " February",
            " March",
            " April",
            " May",
            " June",
            " July",
            " August",
            " September",
            " October",
            " November",
            " December",
        ),
    ),
    (
        "colours",
        False,
        (
            " red",
            " orange",
            " yellow",
            " green",
            " blue",
            " purple",
            " black",
            " white",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[atlas {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cfg_hash(payload: dict) -> str:
    """Stable short hash of a stage's inputs — the §7 'recorded config hash'."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover — not a git checkout
        return "unknown"


@dataclass
class Stage:
    """One cached pipeline stage (§7 stage discipline)."""

    name: str
    work: Path
    config: dict
    outputs: tuple[str, ...]

    @property
    def marker(self) -> Path:
        return self.work / f".{self.name}.done.json"

    def is_fresh(self) -> bool:
        if not self.marker.exists():
            return False
        try:
            rec = json.loads(self.marker.read_text())
        except Exception:
            return False
        if rec.get("config_hash") != cfg_hash(self.config):
            return False
        return all((self.work / o).exists() for o in self.outputs)

    def commit(self, summary: dict) -> None:
        self.marker.write_text(
            json.dumps(
                {
                    "stage": self.name,
                    "config_hash": cfg_hash(self.config),
                    "config": self.config,
                    "git_commit": git_commit(),
                    "finished_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "summary": summary,
                },
                indent=2,
                default=str,
            )
        )


def torch_device(prefer: str) -> "object":
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage 1 — corpus
# ---------------------------------------------------------------------------


def stage_corpus(args, pin: ModelPin, work: Path) -> None:
    """wikitext -> two disjoint packed token arrays (stats / trace).

    The corpus exists ONLY to estimate co-activation and firing statistics over a
    24 k-wide dictionary — per-feature statistics and labels are bought from
    Neuronpedia (§7) — so it is sized in the low millions of tokens, not the
    5-20 M a from-scratch statistics pass would need. That is the difference
    between an afternoon and a week on a machine with no CUDA (§10 G3).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(pin.hf_model)
    seq_len = pin.sae_ctx

    def pack(split: str, n_tokens: int, seq_len: int) -> np.ndarray:
        ds = load_dataset(CORPUS["repo"], CORPUS["config"], split=split, streaming=True)
        buf: list[int] = []
        eot = tok.eos_token_id
        for row in ds:
            text = row["text"]
            if len(text.strip()) < 32:  # wikitext blank lines and bare headings
                continue
            buf.extend(tok(text, add_special_tokens=False)["input_ids"])
            buf.append(eot)
            if len(buf) >= n_tokens:
                break
        # Every window opens with BOS, the SAELens `prepend_bos=True` convention.
        # Measured to make no difference to L0 or FVU here, but it costs one token
        # and keeps the stream identical to what the dictionaries saw.
        body = seq_len - 1
        n_seq = len(buf) // body
        arr = np.asarray(buf[: n_seq * body], dtype=np.uint16).reshape(n_seq, body)
        bos = np.full((n_seq, 1), eot, dtype=np.uint16)
        return np.concatenate([bos, arr], axis=1)

    stats_tokens = pack("train", args.stats_tokens, seq_len)
    log(f"corpus: stats {stats_tokens.shape} = {stats_tokens.size:,} tokens")
    np.save(work / "corpus_stats.npy", stats_tokens)

    # The trace is drawn from the VALIDATION split, so the reading session is held
    # out from the geography, and runs at the SAME 128-token window as the
    # statistics pass — see the context-length cliff.
    trace_needed = (args.trace_windows + 1) * seq_len
    trace_tokens = pack("validation", trace_needed, seq_len)[: args.trace_windows]
    log(f"corpus: trace {trace_tokens.shape} = {trace_tokens.size:,} tokens")
    np.save(work / "corpus_trace.npy", trace_tokens)

    # Probe token ids, resolved once against the pinned tokenizer.
    probes = {}
    for name, cyclic, members in CONCEPT_PROBES:
        ids, kept = [], []
        for m in members:
            enc = tok(m, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:  # single-token members only — a multi-token member
                ids.append(int(enc[0]))  # would need position bookkeeping the
                kept.append(m)  # audit does not need
        probes[name] = {"cyclic": cyclic, "token_ids": ids, "members": kept}
    (work / "probes.json").write_text(json.dumps(probes, indent=2))
    log(
        "corpus: probes "
        + ", ".join(f"{k}={len(v['token_ids'])}" for k, v in probes.items())
    )


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------


@dataclass
class SAE:
    layer: int
    hook: str
    W_enc: "object"  # [d_model, d_sae]
    b_enc: "object"  # [d_sae]
    W_dec: "object"  # [d_sae, d_model]
    b_dec: "object"  # [d_model]
    apply_b_dec_to_input: bool

    def encode(self, x):
        import torch

        h = x - self.b_dec if self.apply_b_dec_to_input else x
        return torch.relu(h @ self.W_enc + self.b_enc)


def load_saes(pin: ModelPin, device, dtype) -> list[SAE]:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    saes: list[SAE] = []
    for layer, hook, _np_id in pin.sae_layers:
        cfg_path = hf_hub_download(pin.sae_repo, f"{hook}/cfg.json")
        w_path = hf_hub_download(pin.sae_repo, f"{hook}/sae_weights.safetensors")
        cfg = json.loads(Path(cfg_path).read_text())
        sd = load_file(w_path)
        W_dec = sd["W_dec"].to(device=device, dtype=dtype)
        saes.append(
            SAE(
                layer=layer,
                hook=hook,
                W_enc=sd["W_enc"].to(device=device, dtype=dtype),
                b_enc=sd["b_enc"].to(device=device, dtype=dtype),
                W_dec=W_dec,
                b_dec=sd["b_dec"].to(device=device, dtype=dtype),
                apply_b_dec_to_input=bool(cfg.get("apply_b_dec_to_input", True)),
            )
        )
        if layer == 0:
            log(
                f"sae cfg: d_in={cfg.get('d_in')} d_sae={cfg.get('d_sae')} "
                f"apply_b_dec_to_input={cfg.get('apply_b_dec_to_input')} "
                f"normalize={cfg.get('normalize_activations')}"
            )
    torch.manual_seed(0)
    return saes


def _resid_pre(model, tokens, n_layers: int, centred: bool):
    """HF hidden_states[0..n_layers-1] == TransformerLens resid_pre 0..n_layers-1.

    `centred` removes the per-token d_model mean, which is what
    TransformerLens's `center_writing_weights=True` bakes into the weights.
    """
    import torch

    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    hs = out.hidden_states[:n_layers]
    if centred:
        hs = tuple(h - h.mean(dim=-1, keepdim=True) for h in hs)
    return hs


def _calibrate_basis(model, saes, tokens, n_layers) -> tuple[bool, dict]:
    """Measure, don't assume: which residual basis do these SAEs actually read?

    Reports normalized reconstruction MSE (`1 - FVU`-style) and mean L0 for the
    centred and uncentred streams. A dictionary reading the wrong basis shows up
    immediately as a blown-out L0 and a reconstruction error near 1.
    """
    import torch

    report = {}
    for centred in (True, False):
        hs = _resid_pre(model, tokens, n_layers, centred)
        mses, l0s, per_layer = [], [], []
        for sae, x in zip(saes, hs):
            # Position 0 is the attention-sink token: a residual an order of
            # magnitude larger than the rest, which would dominate both numbers.
            x = x[:, 1:, :].reshape(-1, x.shape[-1]).to(sae.W_enc.dtype)
            with torch.no_grad():
                acts = sae.encode(x)
                recon = (acts @ sae.W_dec + sae.b_dec).float()
                xf = x.float()
                mse = ((recon - xf) ** 2).sum(-1).mean()
                denom = ((xf - xf.mean(0, keepdim=True)) ** 2).sum(-1).mean()
                mses.append(float(mse / denom))
                l0s.append(float((acts > 0).sum(-1).float().mean()))
                per_layer.append(
                    {"layer": sae.layer, "fvu": round(mses[-1], 4), "l0": round(l0s[-1], 1)}
                )
        report["centred" if centred else "uncentred"] = {
            "normalized_mse": round(float(np.mean(mses)), 4),
            "mean_l0": round(float(np.mean(l0s)), 1),
            "per_layer": per_layer,
        }
    centred_wins = (
        report["centred"]["normalized_mse"] <= report["uncentred"]["normalized_mse"]
    )
    report["chosen"] = "centred" if centred_wins else "uncentred"
    return centred_wins, report


# ---------------------------------------------------------------------------
# Stage 2 — stats
# ---------------------------------------------------------------------------


def stage_stats(args, pin: ModelPin, work: Path) -> dict:
    """Firing statistics, concept-probe affinities and a co-activation sample.

    Streams the corpus: a batch is encoded layer-by-layer and reduced to
    accumulators immediately, so peak memory is one batch's activation matrix
    (batch_tokens x d_sae fp16), never the corpus.
    """
    import torch
    from transformers import AutoModel

    device = torch_device(args.device)
    # fp32, deliberately. fp16 encoding measured mean L0 = 728 against 339 for the
    # same batch in fp32 and NaN'd the reconstruction error: accumulating a
    # 768-term dot product in half precision pushes a crowd of slightly-negative
    # pre-activations over the ReLU threshold, and a dictionary whose sparsity is
    # a rounding artefact is not a dictionary. Costs ~2x memory per batch.
    dtype = torch.float32
    log(f"stats: device={device} dtype={dtype}")

    tokens_np = np.load(work / "corpus_stats.npy")
    probes = json.loads((work / "probes.json").read_text())

    model = AutoModel.from_pretrained(pin.hf_model).to(device).eval()
    saes = load_saes(pin, device, dtype)
    n_layers, d_sae = len(saes), pin.d_sae

    probe_ids = {k: np.asarray(v["token_ids"], dtype=np.int64) for k, v in probes.items()}
    probe_names = list(probe_ids)

    calib_batch = torch.from_numpy(tokens_np[:2].astype(np.int64)).to(device)
    centred, basis_report = _calibrate_basis(model, saes, calib_batch, n_layers)
    log(f"stats: residual basis -> {json.dumps(basis_report)}")

    fire = np.zeros((n_layers, d_sae), dtype=np.int64)
    act_sum = np.zeros((n_layers, d_sae), dtype=np.float64)
    act_max = np.zeros((n_layers, d_sae), dtype=np.float32)
    probe_sum = {p: np.zeros((n_layers, d_sae), dtype=np.float64) for p in probe_names}
    probe_n = {p: 0 for p in probe_names}
    coact: list[np.ndarray] = []  # [n_layers, sample, K] top indices

    bs = args.batch_seqs
    n_batches = math.ceil(len(tokens_np) / bs)
    coact_stride = max(1, (len(tokens_np) * tokens_np.shape[1]) // max(1, args.coact_tokens))
    total_tokens = 0
    t0 = time.time()

    for bi in range(n_batches):
        chunk = tokens_np[bi * bs : (bi + 1) * bs].astype(np.int64)
        if not len(chunk):
            break
        toks = torch.from_numpy(chunk).to(device)
        # Position 0 is the prepended BOS: an attention-sink residual an order of
        # magnitude off-distribution. It is excluded from every accumulator, so
        # activation_max is a statement about text, not about the sink.
        flat_tokens = chunk[:, 1:].reshape(-1)
        hs = [h[:, 1:, :] for h in _resid_pre(model, toks, n_layers, centred)]
        probe_masks = {
            p: torch.from_numpy(np.isin(flat_tokens, ids)).to(device)
            for p, ids in probe_ids.items()
            if len(ids)
        }
        for p, m in probe_masks.items():
            probe_n[p] += int(m.sum().item())
        sel = np.arange(0, len(flat_tokens), coact_stride)
        sel_t = torch.from_numpy(sel).to(device)
        batch_coact = []
        for li, (sae, x) in enumerate(zip(saes, hs)):
            x = x.reshape(-1, x.shape[-1]).to(dtype)
            with torch.no_grad():
                acts = sae.encode(x)
                fire[li] += (acts > 0).sum(0).to(torch.int64).cpu().numpy()
                act_sum[li] += acts.sum(0, dtype=torch.float32).cpu().numpy()
                act_max[li] = np.maximum(
                    act_max[li], acts.max(0).values.float().cpu().numpy()
                )
                for p, m in probe_masks.items():
                    if m.any():
                        probe_sum[p][li] += (
                            acts[m].sum(0, dtype=torch.float32).cpu().numpy()
                        )
                if len(sel):
                    top = acts[sel_t].topk(args.topk, dim=-1).indices
                    batch_coact.append(top.to(torch.int32).cpu().numpy())
                del acts
        if batch_coact:
            coact.append(np.stack(batch_coact))
        total_tokens += len(flat_tokens)
        if bi % 20 == 0 or bi == n_batches - 1:
            rate = total_tokens / max(1e-6, time.time() - t0)
            log(
                f"stats: batch {bi + 1}/{n_batches} "
                f"{total_tokens:,} tokens ({rate:,.0f} tok/s)"
            )

    coact_arr = np.concatenate(coact, axis=1) if coact else np.zeros((n_layers, 0, args.topk), np.int32)
    log(f"stats: co-activation sample {coact_arr.shape}")

    np.savez_compressed(
        work / "feature_stats.npz",
        fire=fire,
        act_sum=act_sum,
        act_max=act_max,
        total_tokens=np.int64(total_tokens),
        **{f"probe_sum__{p}": probe_sum[p] for p in probe_names},
        **{f"probe_n__{p}": np.int64(probe_n[p]) for p in probe_names},
    )
    np.save(work / "coact_topk.npy", coact_arr)

    # Decoder directions are the cross-layer geometry: every resid_pre SAE writes
    # into the SAME residual basis, so a layer-3 latent and a layer-9 latent are
    # directly comparable. That is what makes X/Y a shared geography with layer on Z
    # rather than 12 unrelated maps.
    dec = np.concatenate(
        [sae.W_dec.float().cpu().numpy() for sae in saes], axis=0
    ).astype(np.float32)
    dec /= np.linalg.norm(dec, axis=1, keepdims=True) + 1e-8
    np.save(work / "decoder_dirs.npy", dec)
    log(f"stats: decoder dirs {dec.shape}")

    return {
        "tokens": int(total_tokens),
        "basis": basis_report,
        "coact_sample": int(coact_arr.shape[1]),
        "dead_features": int((fire.reshape(-1) == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Stage 3 — graph
# ---------------------------------------------------------------------------


def _cosine_knn(dec: np.ndarray, k: int, chunk: int, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    n = dec.shape[0]
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    D = torch.from_numpy(dec).to(device=device, dtype=dtype)
    idx = np.empty((n, k), dtype=np.int32)
    val = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        with torch.no_grad():
            sim = D[s:e] @ D.T
            sim[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -2.0
            top = sim.topk(k, dim=-1)
        idx[s:e] = top.indices.to(torch.int32).cpu().numpy()
        val[s:e] = top.values.float().cpu().numpy()
        del sim, top
        if (s // chunk) % 20 == 0:
            log(f"graph: cosine kNN {e:,}/{n:,}")
    return idx, val


def _coactivation_edges(
    coact: np.ndarray, d_sae: int, k: int, rows_per_block: int, min_count: int
) -> np.ndarray:
    """Within-layer co-activation Jaccard, from the top-K sample.

    Nobody publishes joint activations, which is exactly why this stage cannot be
    bought (§7). Pairs are counted by sort-reducing packed int64 keys in row
    blocks, so neither the 24576^2 dense matrix nor the full pair list is ever
    allocated. Returns an (n, 3) array of [node_a, node_b, jaccard].
    """
    n_layers, n_sample, K = coact.shape
    iu = np.triu_indices(K, k=1)
    out: list[np.ndarray] = []
    for li in range(n_layers):
        top = np.sort(coact[li].astype(np.int64), axis=1)  # [sample, K]
        sample_fire = np.zeros(d_sae, dtype=np.int64)
        np.add.at(sample_fire, top.reshape(-1), 1)
        keys_acc: np.ndarray | None = None
        counts_acc: np.ndarray | None = None
        for s in range(0, n_sample, rows_per_block):
            blk = top[s : s + rows_per_block]
            pa = blk[:, iu[0]].reshape(-1)
            pb = blk[:, iu[1]].reshape(-1)
            keep = pa != pb  # a token can list the same latent twice only if
            keys = pa[keep] * d_sae + pb[keep]  # top-K padded — drop those pairs
            u, c = np.unique(keys, return_counts=True)
            if keys_acc is None:
                keys_acc, counts_acc = u, c
            else:
                merged = np.concatenate([keys_acc, u])
                mc = np.concatenate([counts_acc, c])
                order = np.argsort(merged, kind="stable")
                merged, mc = merged[order], mc[order]
                keys_acc, start = np.unique(merged, return_index=True)
                counts_acc = np.add.reduceat(mc, start)
        if keys_acc is None or not len(keys_acc):
            continue
        # A pair seen once, by two latents that each fired once, scores a perfect
        # Jaccard of 1.0 and is pure sampling noise. Require real support first.
        support = counts_acc >= min_count
        keys_acc, counts_acc = keys_acc[support], counts_acc[support]
        if not len(keys_acc):
            continue
        fi, fj = keys_acc // d_sae, keys_acc % d_sae
        # Jaccard over the SAMPLED firing counts, so a pair of very frequent
        # latents is not scored as similar merely for being frequent.
        union = sample_fire[fi] + sample_fire[fj] - counts_acc
        jac = counts_acc / np.maximum(union, 1)
        strong = np.argsort(-jac)[: k * d_sae]
        strong = strong[jac[strong] >= 0.02]
        base = li * d_sae
        if len(strong):
            out.append(
                np.column_stack(
                    [base + fi[strong], base + fj[strong], jac[strong]]
                ).astype(np.float64)
            )
        log(f"graph: layer {li} co-activation edges kept {len(strong):,}")
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float64)


def stage_graph(args, pin: ModelPin, work: Path) -> dict:
    device = torch_device(args.device)
    dec = np.load(work / "decoder_dirs.npy")

    idx, val = _cosine_knn(dec, args.knn, args.knn_chunk, device)
    np.save(work / "knn_idx.npy", idx)
    np.save(work / "knn_val.npy", val)

    coact = np.load(work / "coact_topk.npy")
    arr = _coactivation_edges(
        coact, pin.d_sae, args.coact_k, args.coact_rows, args.coact_min_count
    )
    np.save(work / "coact_edges.npy", arr)
    log(f"graph: {len(idx):,} nodes, cosine k={args.knn}, co-activation edges {len(arr):,}")
    return {"nodes": int(len(idx)), "coact_edges": int(len(arr))}


# ---------------------------------------------------------------------------
# Stage 4 — cluster
# ---------------------------------------------------------------------------


def _build_igraph(idx, val, coact_edges, n, alpha, beta, cos_floor):
    import igraph as ig

    src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64)
    w = val.reshape(-1).astype(np.float64)
    keep = w >= cos_floor
    src, dst, w = src[keep], dst[keep], w[keep] * alpha
    if len(coact_edges):
        src = np.concatenate([src, coact_edges[:, 0].astype(np.int64)])
        dst = np.concatenate([dst, coact_edges[:, 1].astype(np.int64)])
        w = np.concatenate([w, coact_edges[:, 2] * beta])
    # Collapse the multiplex into one weighted simple graph: the two channels are
    # evidence for the same adjacency, not two graphs to be traversed separately.
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    key = lo * n + hi
    order = np.argsort(key, kind="stable")
    key, w = key[order], w[order]
    uniq, start = np.unique(key, return_index=True)
    summed = np.add.reduceat(w, start)
    g = ig.Graph(n=n, edges=list(zip((uniq // n).tolist(), (uniq % n).tolist())))
    g.es["weight"] = summed.tolist()
    return g


def _leiden_micro(g, resolution, seed):
    import leidenalg as la

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=args_n_iterations,
    )
    return np.asarray(part.membership, dtype=np.int32)


args_n_iterations = 2  # module-level so the reseed pass shares it


def _ward_tree(centroids, n_region, n_family):
    """Two nested cuts of ONE Ward dendrogram.

    Cutting a single linkage at two heights makes the region/family tree nested by
    construction — a family can never straddle two regions, which is what §6.3's
    "no stage may perturb an earlier one" needs from the clustering, not just from
    the layout.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    Z = linkage(centroids, method="ward")
    regions = fcluster(Z, t=n_region, criterion="maxclust") - 1
    families = fcluster(Z, t=n_family, criterion="maxclust") - 1
    return regions.astype(np.int32), families.astype(np.int32)


def _absorb_small(dec, micro, min_size):
    """Fold sub-`min_size` Leiden communities into their nearest surviving one.

    Leiden on this graph is scale-free: at any resolution that keeps the largest
    community under ~1% of the atlas it also leaves several hundred singletons,
    and a singleton micro-community becomes a one-member "family" that the layout
    then has to give a circle of its own. Absorbing them by decoder-direction
    cosine keeps every node in the tree without inventing structure — a node joins
    the community it is already closest to in the residual basis.
    """
    sizes = np.bincount(micro)
    keep = np.flatnonzero(sizes >= min_size)
    if len(keep) == len(sizes) or not len(keep):
        return micro
    cent = _micro_centroids(dec, micro, len(sizes))[keep]
    remap = np.full(len(sizes), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    small = np.flatnonzero(sizes < min_size)
    small_nodes = np.flatnonzero(np.isin(micro, small))
    micro = micro.copy()
    for s in range(0, len(small_nodes), 20000):
        blk = small_nodes[s : s + 20000]
        micro[blk] = keep[(dec[blk] @ cent.T).argmax(axis=1)]
    return remap[micro].astype(np.int32)


def _cluster_once(idx, val, coact_edges, n, args, seed, dec):
    g = _build_igraph(idx, val, coact_edges, n, args.alpha, args.beta, args.cos_floor)
    micro = _leiden_micro(g, args.leiden_resolution, seed)
    micro = _absorb_small(dec, micro, args.min_community)
    n_micro = int(micro.max()) + 1
    return micro, n_micro


def _micro_centroids(dec, micro, n_micro):
    c = np.zeros((n_micro, dec.shape[1]), dtype=np.float64)
    np.add.at(c, micro, dec)
    cnt = np.bincount(micro, minlength=n_micro).astype(np.float64)[:, None]
    c /= np.maximum(cnt, 1)
    c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-9
    return c


def stage_cluster(args, pin: ModelPin, work: Path) -> dict:
    dec = np.load(work / "decoder_dirs.npy")
    idx = np.load(work / "knn_idx.npy")
    val = np.load(work / "knn_val.npy")
    coact_edges = np.load(work / "coact_edges.npy")
    n = dec.shape[0]

    micro, n_micro = _cluster_once(idx, val, coact_edges, n, args, args.seed, dec)
    log(f"cluster: Leiden micro-communities {n_micro:,}")
    cent = _micro_centroids(dec, micro, n_micro)
    n_family = min(args.n_families, n_micro)
    n_region = min(args.n_regions, n_family)
    m_region, m_family = _ward_tree(cent, n_region, n_family)
    region = m_region[micro]
    family = m_family[micro]
    np.savez(
        work / "clusters.npz",
        micro=micro,
        region=region,
        family=family,
        micro_region=m_region,
        micro_family=m_family,
    )
    sizes = np.bincount(family)
    log(
        f"cluster: regions={region.max() + 1} families={family.max() + 1} "
        f"family size min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}"
    )
    return {
        "micro": n_micro,
        "regions": int(region.max() + 1),
        "families": int(family.max() + 1),
    }


# ---------------------------------------------------------------------------
# Stage 5 — layout (§6.3, revised 2026-07-28)
# ---------------------------------------------------------------------------
#
# The first build laid the atlas out as a nested circle-pack treemap: L0 regions
# packed into discs, L1 families packed into discs inside their parent, L2
# members PCA'd into a disc inside that. Two things were wrong with it — one
# arithmetic, one structural — and only the second one really matters.
#
#   ARITHMETIC. `_pack_circles` initialised its relaxation at a half-extent of
#   1.4 × the SUM of the radii, so n circles of radius r were scattered over
#   1.4·n·r and the relaxation never had an overlap left to resolve. Occupancy
#   came out at π/(7.84·n) = 0.40/n per level — 1.8% at the region level and
#   another 1.6% of that at the family level. Measured on the shipped archive:
#   294,912 latents inside 0.030% of the plane, families of median radius
#   0.0059° against a 1,650 km layer stack, i.e. needles of aspect ratio
#   ~1,250:1. That is the "weird towers", and it is why the map was 99.6% empty.
#
#   STRUCTURAL. Every position was a function of the cluster tree, so no shape
#   could survive the layout. The most a concept could express was a slightly
#   eccentric circle. Emergent geometry — the arcs, hooks, loops and sheets that
#   the neural-geometry literature renders, and that §2.2 says the discrete
#   basis both carries and shatters — needs the projection to see the VECTORS,
#   not the partition. Fixing only the arithmetic would have produced 700
#   well-spaced circles instead of 700 needles. Still circles.
#
# So the layout is now ONE global manifold embedding of all 294,912 decoder
# directions, and the cluster tree is demoted to what it is actually good for:
# LOD banding, labels, inspection, and hull outlines. §6.3's real requirement —
# "clustering happens on the high-dimensional graph, never on the 2-D
# projection" — is still honoured, and more cleanly than before: the clustering
# cannot contaminate the projection because the projection no longer consults
# it. What §6.3 loses is the frozen-bounds nesting, and §14.9 records what that
# costs.
#
# The third embedding component rides as an ordinary numeric column instead of
# being baked into the geometry by `--point-elevation-column`, so flat-vs-3D is
# a renderer prop and not a rebuild — see `stage_pack` and the `elevationProperty`
# contract in `packages/layers/src/layers/core/animated-point-layer.ts`.


def _pca(vectors: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Optional PCA down to `dim`. DEFAULT OFF, and that default is measured.

    PCA before a neighbour embedding is the standard cheap pre-step, on the
    premise that it drops directions that slow kNN search without moving the
    neighbourhoods. That premise is false for an SAE decoder dictionary, which
    is close to isotropic in the residual basis. Measured here, kNN overlap
    (k=15, n=8,000) against the full 768-d space:

        PCA  50  ->  23.2% of variance,  0.216 neighbour overlap
        PCA 128  ->  41.6%,              0.230
        PCA 256  ->  61.6%,              0.291

    Even at 256 dims the embedding would be reading mostly the wrong
    neighbours, so the whole point of the layout — that it sees the vectors —
    would be lost to a speed optimisation. `dim <= 0` (the default) passes the
    vectors straight through.
    """
    if dim <= 0 or vectors.shape[1] <= dim:
        return np.ascontiguousarray(vectors, dtype=np.float32)
    from sklearn.decomposition import PCA

    red = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
    out = red.fit_transform(vectors)
    log(
        f"layout: PCA {vectors.shape[1]} -> {dim} "
        f"(explained variance {red.explained_variance_ratio_.sum():.3f})"
    )
    return np.ascontiguousarray(out, dtype=np.float32)


def _manifold_embed(
    vectors: np.ndarray,
    seed: int,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    prefer_umap: bool,
    verbose: bool = False,
) -> np.ndarray:
    """Neighbour embedding of a point set, UMAP with a PCA fallback.

    Used at two very different scales: the whole 294,912-latent atlas, and the
    few hundred members of one concept probe (§14.10). The decoder directions
    are unit-norm — the stats stage writes them that way — so euclidean distance
    IS cosine geometry here and the metric does not have to fight the PCA step.
    """
    n = len(vectors)
    n_components = min(n_components, max(1, n - 1))
    if prefer_umap and n >= 4 * n_components:
        try:
            import umap

            red = umap.UMAP(
                n_components=n_components,
                n_neighbors=max(2, min(n_neighbors, n - 1)),
                min_dist=min_dist,
                metric="euclidean",
                # Deterministic, at the cost of n_jobs=1. Stage discipline (§7)
                # asks for a reproducible artefact and this is the whole run's
                # wall-clock floor, so it is bought deliberately.
                random_state=seed,
                verbose=verbose,
            )
            return np.asarray(red.fit_transform(vectors), dtype=np.float64)
        except Exception as exc:  # pragma: no cover — numba/env dependent
            log(f"layout: UMAP unavailable ({exc}); PCA fallback")
    from sklearn.decomposition import PCA

    k = min(n_components, vectors.shape[1], max(1, n - 1))
    out = PCA(n_components=k, random_state=seed).fit_transform(vectors)
    if out.shape[1] < n_components:
        out = np.column_stack([out, np.zeros((len(out), n_components - out.shape[1]))])
    return np.asarray(out, dtype=np.float64)


def fit_to_atlas(emb: np.ndarray, keep: float = 99.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Embedding → atlas units, ISOTROPICALLY, with a soft rim.

    ONE scale factor for all three axes, taken from the X/Y radius so the bulk of
    the map fills the ±1 box. Isotropy is what makes the third component usable
    as an altitude at all: a z in different units from x/y is exactly the
    1,250:1 needle that the literal layer axis used to be.

    The tail is soft-clamped rather than truncated. A hard clip would pile every
    outlier onto the rim in a ring that reads as structure and is not.
    """
    p = np.asarray(emb, dtype=np.float64) - np.median(emb, axis=0)
    r_xy = np.hypot(p[:, 0], p[:, 1])
    scale = float(np.percentile(r_xy, keep)) or 1.0
    r0, edge = 0.92, 1.0
    p = p / scale * r0
    r = np.hypot(p[:, 0], p[:, 1])
    over = r > r0
    shrunk = r.copy()
    shrunk[over] = r0 + (edge - r0) * np.tanh((r[over] - r0) / (edge - r0))
    adjust = np.where(r > 1e-12, shrunk / np.maximum(r, 1e-12), 1.0)
    p[:, 0] *= adjust
    p[:, 1] *= adjust
    xy = np.ascontiguousarray(p[:, :2])
    z = np.ascontiguousarray(p[:, 2]) if p.shape[1] > 2 else np.zeros(len(p))
    return (
        xy,
        z,
        {
            "scale_percentile": keep,
            "soft_clamped_nodes": int(over.sum()),
            "xy_extent": [float(xy.min()), float(xy.max())],
            "z_extent": [float(z.min()), float(z.max())],
        },
    )


def compute_layout(dec: np.ndarray, seed: int, args) -> tuple[np.ndarray, np.ndarray, dict]:
    """One global embedding of every latent. No stage of this consults the
    partition, which is the point — see the section note above."""
    reduced = _pca(dec, args.layout_pca_dim, seed)
    emb = _manifold_embed(
        reduced,
        seed,
        n_components=3,
        n_neighbors=args.layout_neighbors,
        min_dist=args.layout_min_dist,
        prefer_umap=not args.no_umap,
        verbose=True,
    )
    return fit_to_atlas(emb)


def stage_layout(args, pin: ModelPin, work: Path) -> dict:
    dec = np.load(work / "decoder_dirs.npy")
    xy, z, fit = compute_layout(dec, args.seed, args)
    np.savez(work / "layout.npz", xy=xy, z=z)

    # Occupancy is reported because it is the number the first build got wrong by
    # three orders of magnitude and nothing in the pipeline noticed. A grid cell
    # here is ~1 screen pixel at the framing zoom, so "cells touched" is close to
    # "fraction of the map that has anything on it".
    n_grid = 512
    span = xy.max(axis=0) - xy.min(axis=0)
    cell = np.floor((xy - xy.min(axis=0)) / np.maximum(span, 1e-12) * (n_grid - 1))
    touched = len(np.unique(cell[:, 0] * n_grid + cell[:, 1]))
    occupancy = touched / (n_grid * n_grid)
    log(
        f"layout: {len(xy):,} nodes, extent x[{xy[:, 0].min():.3f},{xy[:, 0].max():.3f}] "
        f"y[{xy[:, 1].min():.3f},{xy[:, 1].max():.3f}] z[{z.min():.3f},{z.max():.3f}]"
    )
    log(
        f"layout: occupancy {occupancy * 100:.2f}% of a {n_grid}x{n_grid} grid "
        f"({touched:,} cells), {fit['soft_clamped_nodes']:,} nodes soft-clamped at the rim"
    )
    return {
        "nodes": int(len(xy)),
        "grid_occupancy": round(float(occupancy), 5),
        "grid": n_grid,
        **fit,
    }




# ---------------------------------------------------------------------------
# Stage 6 — labels (bought, §7)
# ---------------------------------------------------------------------------


def _fetch_np_batches(source_id: str, model_id: str, cache: Path) -> list[dict]:
    """Neuronpedia explanation export for one SAE.

    The /api/explanation/export endpoint was retired (it now 400s with a pointer);
    the S3 bulk export is the supported path.
    """
    cache.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    listing_url = (
        f"{NEURONPEDIA_S3}/?list-type=2&prefix=v1/{model_id}/{source_id}/explanations/"
    )
    try:
        with urllib.request.urlopen(listing_url, timeout=60) as r:
            xml = r.read().decode()
    except Exception as exc:
        log(f"labels: listing failed for {source_id} ({exc})")
        return out
    keys = re.findall(r"<Key>([^<]+\.jsonl\.gz)</Key>", xml)
    for key in keys:
        local = cache / key.split("/")[-1]
        if not local.exists():
            try:
                with urllib.request.urlopen(f"{NEURONPEDIA_S3}/{key}", timeout=120) as r:
                    local.write_bytes(r.read())
            except Exception as exc:
                log(f"labels: {key} failed ({exc})")
                continue
        with gzip.open(local, "rt") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                emb = rec.get("embedding")
                out.append(
                    {
                        "index": int(rec["index"]),
                        "description": rec.get("description", "").strip(),
                        "model": rec.get("explanationModelName", ""),
                        "embedding": emb,
                    }
                )
    return out


def _status_from_agreement(n_expl: int, agreement: float) -> str:
    """§3: interpretation status is a rendering input, not a tooltip.

    Grounded in data Neuronpedia already ships — several independent explainer
    models per feature, each with an embedding — rather than invented. Agreement is
    the mean pairwise cosine of those embeddings.
    """
    if n_expl == 0:
        return "unlabeled"
    if n_expl == 1:
        return "tentative"
    if agreement >= 0.70:
        return "reviewed"
    if agreement < 0.40:
        return "contested"
    return "tentative"


def stage_labels(args, pin: ModelPin, work: Path) -> dict:
    # Cache outside the work dir: the export is ~700 MB and is identical for
    # every run of this pin, so a re-run with a fresh work dir must not re-fetch it.
    cache = Path(args.np_cache)
    n_layers, d_sae = len(pin.sae_layers), pin.d_sae
    labels = np.full((n_layers, d_sae), "", dtype=object)
    alt = np.full((n_layers, d_sae), "", dtype=object)
    conf = np.zeros((n_layers, d_sae), dtype=np.float32)
    status = np.full((n_layers, d_sae), "unlabeled", dtype=object)
    n_expl_arr = np.zeros((n_layers, d_sae), dtype=np.int16)

    for li, (_layer, _hook, np_id) in enumerate(pin.sae_layers):
        recs = _fetch_np_batches(np_id, pin.neuronpedia_model, cache / np_id)
        by_idx: dict[int, list[dict]] = {}
        for r in recs:
            if r["description"]:
                by_idx.setdefault(r["index"], []).append(r)
        for fi, group in by_idx.items():
            if fi >= d_sae:
                continue
            embs = []
            for g in group:
                e = g["embedding"]
                if isinstance(e, str):
                    try:
                        e = json.loads(e)
                    except Exception:
                        e = None
                if isinstance(e, list) and len(e) > 8:
                    embs.append(np.asarray(e, dtype=np.float32))
            agreement = 1.0
            if len(embs) >= 2:
                E = np.stack(embs)
                E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
                S = E @ E.T
                iu = np.triu_indices(len(E), k=1)
                agreement = float(np.clip(S[iu].mean(), -1, 1))
            labels[li, fi] = group[0]["description"][:160]
            if len(group) > 1:
                alt[li, fi] = group[1]["description"][:160]
            conf[li, fi] = agreement
            n_expl_arr[li, fi] = len(group)
            status[li, fi] = _status_from_agreement(len(group), agreement)
        log(f"labels: {np_id} -> {len(by_idx):,} labelled features")

    np.savez_compressed(
        work / "labels.npz",
        label=labels.astype(str),
        label_alt=alt.astype(str),
        label_confidence=conf,
        interpretation_status=status.astype(str),
        n_explanations=n_expl_arr,
    )
    covered = int((n_expl_arr > 0).sum())
    return {
        "labelled": covered,
        "coverage": round(covered / (n_layers * d_sae), 4),
        "reviewed": int((status == "reviewed").sum()),
        "contested": int((status == "contested").sum()),
    }


# ---------------------------------------------------------------------------
# Stage 7 — trace
# ---------------------------------------------------------------------------


def stage_trace(args, pin: ModelPin, work: Path) -> dict:
    """A reading session: top-K active features per (token, layer), + attribution.

    GPT-2's context is 1024 tokens, so the record's "one document x 8-16 k tokens"
    is a session of consecutive held-out windows with the token index continuing
    across them. Each window keeps its own attention; the token axis is the demo.

    ATTRIBUTION (the `attribution` metric of the §3 enum, never the same column as
    `activation`): one backward pass per window on the logit of the model's own
    top prediction at the window's final position, giving
    d(logit)/d(resid) . W_dec[f] * act[f] — the standard gradient x activation
    attribution over SAE latents. The record's §2.3 preference is to buy this from
    circuit-tracer instead, but circuit-tracer supports Gemma-2-2B / Llama-3.2-1B /
    Qwen3-4B and not GPT-2, so under this pin it is built. G2 (the SAE-vs-
    transcoder basis seam) does not arise here BECAUSE it is built: edges and
    geography share one dictionary.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch_device(args.device)
    dtype = torch.float32  # gradients: keep fp32, the trace is small
    tok = AutoTokenizer.from_pretrained(pin.hf_model)
    model = AutoModel.from_pretrained(pin.hf_model).to(device).eval()
    # GPT-2 ties the unembedding to the token embedding, so the embedding matrix
    # IS the logit head — no need to load the LM-head wrapper model.
    lm_head_w = model.get_input_embeddings().weight.to(device)
    saes = load_saes(pin, device, dtype)
    n_layers = len(saes)

    windows = np.load(work / "corpus_trace.npy").astype(np.int64)
    body = windows.shape[1] - 1  # real tokens per window, BOS excluded
    marker = json.loads((work / ".stats.done.json").read_text())
    centred = marker["summary"]["basis"]["chosen"] == "centred"

    rows_node, rows_tok, rows_act, rows_attr, rows_layer = [], [], [], [], []
    token_strings: list[str] = []
    targets: list[dict] = []
    K = args.trace_topk

    for wi, w in enumerate(windows):
        toks = torch.from_numpy(w[None, :]).to(device)
        out = model(toks, output_hidden_states=True)
        # retain_grad goes on the RAW residuals, because they are what the logit
        # is a function of. Centring produces NEW tensors that CONSUME the
        # residual rather than produce the logit, so backward() never reaches
        # them: their .grad stays None and every attribution silently comes out
        # zero. That is not hypothetical — it is what the first build shipped.
        raw = list(out.hidden_states[:n_layers])
        for h in raw:
            h.retain_grad()
        hs = [h - h.mean(dim=-1, keepdim=True) for h in raw] if centred else raw
        final = out.last_hidden_state[0, -1]
        logits = final @ lm_head_w.T
        target_id = int(logits.argmax().item())
        model.zero_grad(set_to_none=True)
        logits[target_id].backward()
        targets.append(
            {
                "window": wi,
                "target_token_id": target_id,
                "target_token": tok.decode([target_id]),
                "at_token_index": wi * body + len(w) - 2,
            }
        )

        for li, (sae, h) in enumerate(zip(saes, hs)):
            # Drop the prepended BOS from the emitted events for the same reason
            # the statistics pass drops it: it is an attention sink, not text.
            x = h.detach()[0, 1:]
            if raw[li].grad is None:
                # Loud, not zero: a missing gradient means the graph changed
                # shape and the attribution metric is not being computed at all.
                raise RuntimeError(
                    f"no gradient reached hidden_states[{li}] — attribution "
                    "would be identically zero; check that nothing detaches "
                    "the residual between the forward pass and backward()"
                )
            g = raw[li].grad[0, 1:].detach()
            if centred:
                # Centring is the symmetric projection C = I - 11^T/d. One unit
                # of act[f] perturbs the RAW residual by C·W_dec[f], so the
                # gradient to dot with W_dec is C·g — i.e. g with its own
                # per-token d_model mean removed, exactly as x was.
                g = g - g.mean(dim=-1, keepdim=True)
            with torch.no_grad():
                acts = sae.encode(x)
                top = acts.topk(K, dim=-1)
                # d logit / d act[f] = grad_resid . W_dec[f]
                per_feature_grad = g @ sae.W_dec.T  # [pos, d_sae]
                attr = torch.gather(per_feature_grad, 1, top.indices) * top.values
            n_pos = x.shape[0]
            base = wi * body
            rows_tok.append(np.repeat(np.arange(base, base + n_pos, dtype=np.int32), K))
            rows_layer.append(np.full(n_pos * K, li, dtype=np.int8))
            rows_node.append(
                (top.indices.to(torch.int64).cpu().numpy() + li * pin.d_sae)
                .reshape(-1)
                .astype(np.int32)
            )
            rows_act.append(top.values.float().cpu().numpy().reshape(-1))
            rows_attr.append(attr.float().cpu().numpy().reshape(-1))
        token_strings.extend(tok.batch_decode([[t] for t in w.tolist()[1:]]))
        if wi % 8 == 0 or wi == len(windows) - 1:
            log(f"trace: window {wi + 1}/{len(windows)}")

    node = np.concatenate(rows_node)
    tokidx = np.concatenate(rows_tok)
    act = np.concatenate(rows_act)
    attr = np.concatenate(rows_attr)
    layer = np.concatenate(rows_layer)
    keep = act > 0  # a top-K slot below the ReLU threshold is not an event
    node, tokidx, act, attr, layer = (
        node[keep],
        tokidx[keep],
        act[keep],
        attr[keep],
        layer[keep],
    )
    np.savez_compressed(
        work / "trace.npz",
        node=node,
        token_index=tokidx,
        activation=act,
        attribution=attr,
        layer=layer,
    )
    (work / "trace_tokens.json").write_text(
        json.dumps({"tokens": token_strings, "targets": targets})
    )
    log(f"trace: {len(node):,} activation events over {len(token_strings):,} tokens")
    return {
        "events": int(len(node)),
        "tokens": int(len(token_strings)),
        "windows": int(len(windows)),
    }


# ---------------------------------------------------------------------------
# Stage 8 — validate (§9)
# ---------------------------------------------------------------------------


def concept_affinity(stats, pname: str, min_fire: int) -> np.ndarray | None:
    """How much more does each latent fire on a concept's tokens than in general?

    Two guards, both learned from the first run producing nonsense:

    * **A rare latent is not a concept latent.** With a bare `1e-6` denominator a
      latent that fires almost never scores an unbounded ratio off a handful of
      coincidences, and the "top 24 latents for weekdays" came back as 24
      near-dead neighbours sitting on top of each other (pairwise p90 distance
      0.0001 atlas units — a point, not a concept). Latents below `min_fire`
      firings across the whole corpus are excluded outright.
    * **The denominator floor is the median live mean-activation**, not an
      epsilon, so the ratio is "against a typical latent" rather than "against
      whatever noise this one happened to have".

    Returns `None` when the probe never occurred in the corpus.
    """
    key = f"probe_sum__{pname}"
    if key not in stats or int(stats[f"probe_n__{pname}"]) == 0:
        return None
    total = max(1, int(stats["total_tokens"]))
    fire = stats["fire"].reshape(-1)
    mean_act = (stats["act_sum"].reshape(-1) / total).astype(np.float64)
    alive = fire >= min_fire
    if not alive.any():
        return None
    floor = float(np.median(mean_act[alive]))
    probe_mean = stats[key].reshape(-1) / int(stats[f"probe_n__{pname}"])
    affinity = probe_mean / (mean_act + floor)
    affinity[~alive] = -np.inf
    return affinity


def stage_validate(args, pin: ModelPin, work: Path) -> dict:
    """The three numbers §9 says must be published, not merely computed."""
    from sklearn.manifold import trustworthiness

    dec = np.load(work / "decoder_dirs.npy")
    cl = np.load(work / "clusters.npz")
    lay = np.load(work / "layout.npz")
    xy = lay["xy"]
    report: dict = {"units": {}, "method": {}}

    # 1. Atlas drift under reseed (§2.1, §9.1).
    #
    #    BE PRECISE ABOUT WHICH RESEED. §2.1's result is about retraining the
    #    SPARSE AUTOENCODER: different init, different latents, same subspaces.
    #    That is not reachable here — it means training a 24k-wide dictionary,
    #    which §11 counts out. What IS measurable, and is what this reports, is
    #    the CLUSTERING reseed: same latents, same decoder directions, a
    #    resampled Leiden partition over them. It bounds a different and weaker
    #    claim — "is the geography a property of the dictionary, or of the
    #    partition we drew over it" — and the answer here is uncomfortable, so
    #    it is published under its own name rather than under §2.1's.
    idx = np.load(work / "knn_idx.npy")
    val = np.load(work / "knn_val.npy")
    coact_edges = np.load(work / "coact_edges.npy")
    micro2, n_micro2 = _cluster_once(
        idx, val, coact_edges, dec.shape[0], args, args.seed + 1, dec
    )
    cent2 = _micro_centroids(dec, micro2, n_micro2)
    r2, f2 = _ward_tree(cent2, int(cl["region"].max()) + 1, int(cl["family"].max()) + 1)
    region2, family2 = r2[micro2], f2[micro2]
    # The layout is now a global manifold embedding that never consults the
    # partition (see the stage-5 note), so a reseeded clustering moves NO point:
    # the same `xy` goes in on both sides. What still moves is the geography
    # drawn OVER those fixed points — which latents a region collects, and hence
    # where its centroid and its hull land. That is a strictly cleaner
    # measurement than the first build's, which reseeded the partition and then
    # re-ran a layout that was itself a function of the partition, confounding
    # "the clusters moved" with "the treemap redrew itself".

    def centroid_shift(a_lbl, a_xy, b_lbl, b_xy):
        """Displacement after matching each reseeded cluster to the original whose
        membership it overlaps most — a relabelling is not drift."""
        n_a = int(a_lbl.max()) + 1
        n_b = int(b_lbl.max()) + 1
        conf = np.zeros((n_a, n_b), dtype=np.int64)
        np.add.at(conf, (a_lbl, b_lbl), 1)
        match = conf.argmax(axis=1)
        shifts = []
        for ai in range(n_a):
            ca = a_xy[a_lbl == ai].mean(0)
            sel = b_lbl == match[ai]
            if not sel.any():
                continue
            shifts.append(float(np.hypot(*(b_xy[sel].mean(0) - ca))))
        return np.asarray(shifts), match

    r_shift, r_match = centroid_shift(cl["region"], xy, region2, xy)
    f_shift, _ = centroid_shift(cl["family"], xy, family2, xy)
    reparented = float(
        (r_match[cl["region"]] != region2).mean()
    )  # fraction of L2 members whose region changed
    report["atlas_drift_under_clustering_reseed"] = {
        "l0_centroid_shift_atlas_units": {
            "median": round(float(np.median(r_shift)), 4),
            "p90": round(float(np.percentile(r_shift, 90)), 4),
            "max": round(float(r_shift.max()), 4),
        },
        "l1_centroid_shift_atlas_units": {
            "median": round(float(np.median(f_shift)), 4),
            "p90": round(float(np.percentile(f_shift, 90)), 4),
        },
        "l2_members_changing_region_fraction": round(reparented, 4),
        "atlas_unit_definition": (
            "1.0 = the atlas half-width = "
            f"{ATLAS_HALF_DEG:.0f} deg = {ATLAS_HALF_M / 1000:.0f} km on the map"
        ),
        "source": (
            f"neural_atlas.py stage=validate, Leiden seeds {args.seed} vs "
            f"{args.seed + 1} over IDENTICAL node positions — the layout is a "
            "global manifold embedding and does not consult the partition, so "
            "this is partition drift with no layout term in it at all"
        ),
        "not_measured": (
            "SAE-retraining drift (arXiv:2606.12138's actual claim). It needs a "
            "second 24576-wide dictionary trained from a different seed, which "
            "is counted out for this build. Do not read these numbers as "
            "confirming or refuting that result."
        ),
        "reading": (
            "Where a latent SITS is now fully determined by the embedding and "
            "does not move when the clustering is resampled — the positions on "
            "both sides of this measurement are byte-identical. What moves is "
            "the naming: a resampled Leiden run reparents this fraction of the "
            "latents, so the continents and their labels are a property of the "
            "PARTITION, not of the dictionary. Read a region outline as 'these "
            "things are near each other', never as 'this is the grammar "
            "continent'. Proximity is durable here; the borders are not."
        ),
    }

    # 2. Projection distortion, with the k it was computed at.
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(len(dec), size=min(args.distortion_sample, len(dec)), replace=False)
    k = args.distortion_k
    tw = float(trustworthiness(dec[sample], xy[sample], n_neighbors=k, metric="cosine"))
    # Continuity is trustworthiness with the roles of the two spaces swapped.
    ct = float(trustworthiness(xy[sample], dec[sample], n_neighbors=k, metric="euclidean"))
    from sklearn.neighbors import NearestNeighbors

    hi = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(dec[sample])
    lo = NearestNeighbors(n_neighbors=k + 1).fit(xy[sample])
    hn = hi.kneighbors(dec[sample], return_distance=False)[:, 1:]
    ln = lo.kneighbors(xy[sample], return_distance=False)[:, 1:]
    overlap = float(
        np.mean([len(set(a) & set(b)) / k for a, b in zip(hn.tolist(), ln.tolist())])
    )
    report["projection_distortion"] = {
        "k": k,
        "sample_n": int(len(sample)),
        "trustworthiness": round(tw, 4),
        "continuity": round(ct, 4),
        "knn_overlap": round(overlap, 4),
        "source": "sklearn.manifold.trustworthiness / NearestNeighbors, cosine in the "
        "768-d residual basis vs euclidean in atlas units",
    }

    # 3. Manifold-shattering audit (§2.2, §9.2).
    stats = np.load(work / "feature_stats.npz")
    probes = json.loads((work / "probes.json").read_text())
    _n_layers, d_sae = stats["fire"].shape
    audit = {}
    for pname in probes:
        affinity = concept_affinity(stats, pname, args.probe_min_fire)
        if affinity is None:
            continue
        pn = int(stats[f"probe_n__{pname}"])
        top = np.argsort(-affinity)[: args.probe_top]
        pts = xy[top]
        span = float(
            np.percentile(
                np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)), 90
            )
        )
        audit[pname] = {
            "probe_token_occurrences": pn,
            "latents_carrying_concept": int(args.probe_top),
            "distinct_regions": int(len(np.unique(cl["region"][top]))),
            "distinct_families": int(len(np.unique(cl["family"][top]))),
            "pairwise_atlas_distance_p90": round(span, 4),
            "layers_spanned": int(len(np.unique(top // d_sae))),
        }
    report["manifold_shattering_audit"] = {
        "probes": audit,
        "reading": (
            "distinct_families >> 1 for a concept with known continuous structure is "
            "the shattering of arXiv:2509.02565 / Goodfire's slant-rhyme result. It is "
            "a property of the discrete basis, NOT evidence of fragmentation in the "
            "model."
        ),
        "source": "neural_atlas.py stage=validate, affinity = mean activation on probe "
        "tokens / mean activation over the corpus",
    }

    (work / "validation.json").write_text(json.dumps(report, indent=2))
    log(
        "validate: "
        + json.dumps(
            report["atlas_drift_under_clustering_reseed"][
                "l0_centroid_shift_atlas_units"
            ]
        )
    )
    log("validate: " + json.dumps(report["projection_distortion"]))
    return report


# ---------------------------------------------------------------------------
# Stage 9 — pack
# ---------------------------------------------------------------------------


def _wkb_points(lon, lat):
    import shapely

    return shapely.to_wkb(shapely.points(np.asarray(lon, "float64"), np.asarray(lat, "float64")))


def atlas_to_lonlat(xy: np.ndarray):
    """Atlas unit square -> lon/lat, via the same helper the AV/worlds generators
    use (§4.2). At the origin cos(lat) == 1, so the mapping is isotropic exactly
    where the atlas lives."""
    return local_to_lonlat(
        xy[:, 0] * ATLAS_HALF_M, xy[:, 1] * ATLAS_HALF_M, ATLAS_ORIGIN_LAT, ATLAS_ORIGIN_LON
    )


def _run_stt_build(stt_build: Path, out_dir: Path, argv: list[str], name: str) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [str(stt_build), *argv]
    log(f"pack: stt-build {name}")
    log("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


#: Words that carry no information in a Neuronpedia explanation. Kept short and
#: explicit rather than pulling an NLP dependency in for four dozen labels.
_PLACE_STOPWORDS = frozenset("""
a an the and or of to in on at for with without from by as is are was were be
been being this that these those it its it's their there here they them he she
his her which who whom what when where how why not no nor but if then than
so such can could may might must shall should will would do does did done
words word tokens token text texts phrase phrases term terms use used using
something anything everything nothing thing things kind sort type types
end ends beginning start starts within across between among about
particularly specifically especially generally typically usually often likely
common commonly frequent frequently containing contains contain including
includes include follow follows following preceded preceding given also
possibly probably apparently seemingly might may several many much more most
related relating relate reference references referring refers specific
particular various different certain other others some any all both each
concept concepts context contexts feature features activates activation
activating fires firing pattern patterns instances instance examples example
occurrences occurring occurs appear appears appearing associated association
representing represents represent indicating indicates indicate mentions
mentioning mention describing describes describe involving involves involve
""".split())


def _place_stem(word: str) -> str:
    """Crudest possible stem — enough to stop `name` and `names` naming two
    different places, which is the only conflation that showed up in practice."""
    for suffix in ("ies", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _place_terms(labels: list[str], global_df: dict[str, int], n_docs: int,
                 used: "dict[str, int] | None" = None, top: int = 3) -> list[str]:
    """The terms most OVER-represented in one neighbourhood's labels.

    TF-IDF against the whole label corpus, because "the commonest words here"
    returns `the`, `of`, `to`; what a reader wants is the words common HERE and
    rare everywhere else, which is what the IDF factor buys.

    `used` additionally penalises terms already spent on an earlier (denser)
    place. Without it the first pass named four separate places "names", which
    is true of all of them and therefore tells a reader nothing about any of
    them — the labels have to distinguish places from EACH OTHER, not just from
    the corpus.
    """
    from collections import Counter

    local = Counter()
    for lab in labels:
        for w in set(re.findall(r"[a-z][a-z'-]{2,}", lab.lower())):
            if w not in _PLACE_STOPWORDS:
                local[w] += 1
    if not local:
        return []
    n_local = max(1, len(labels))
    scored = []
    for w, tf in local.items():
        if tf < 2:
            continue
        stem = _place_stem(w)
        spent = (used or {}).get(stem, 0)
        # Hard cap, not just a penalty. A term dominant enough to survive a
        # soft 1/(1+n) discount three times over is exactly the term that tells
        # a reader nothing about which place they are looking at.
        if spent >= 2:
            continue
        idf = math.log(n_docs / (1 + global_df.get(w, 0)))
        scored.append((tf / n_local * idf / (1 + spent) ** 1.5, w))
    scored.sort(reverse=True)
    out: list[str] = []
    seen_stems: set[str] = set()
    for _, w in scored:
        stem = _place_stem(w)
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        out.append(w)
        if len(out) == top:
            break
    return out


def _density_places(xy: np.ndarray, labels: np.ndarray, n_places: int,
                    grid: int = 200, sigma: float = 2.4) -> list[dict]:
    """Where the atlas piles up, and what is there — instead of cluster hulls.

    §15.7 measured that this dataset has NO cluster structure to outline: the
    decoder dictionary is near-isotropic (random-pair cosine 0.0099), Leiden
    communities do not localise in any projection tried, and HDBSCAN on the
    embedding itself finds two clusters. Drawing hulls anyway put 87.8% of
    region pairs on top of each other and asserted borders the model does not
    have, which is the §3 "a map is a persuasive object" failure exactly.

    Density is the honest alternative, because it is a claim about the PICTURE
    rather than about the model: these are the parts of the projection where
    latents pile up, named by the words that are over-represented among their
    published explanations. No boundaries are drawn because none were found.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter

    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    cell = np.floor((xy - lo) / span * (grid - 1)).astype(np.int32)
    hist = np.zeros((grid, grid), dtype=np.float64)
    np.add.at(hist, (cell[:, 0], cell[:, 1]), 1.0)
    dens = gaussian_filter(hist, sigma=sigma, mode="nearest")

    # A peak is a cell that is the maximum of its neighbourhood; the footprint
    # sets the minimum separation between two labels, so it is a legibility
    # constant, not a statistical one.
    foot = max(3, int(grid / 22) | 1)
    peaks = (dens == maximum_filter(dens, size=foot)) & (dens > np.percentile(dens, 55))
    py, px = np.nonzero(peaks)
    order = np.argsort(-dens[py, px])[:n_places]
    py, px = py[order], px[order]
    centres = np.column_stack([
        lo[0] + (py + 0.5) / grid * span[0],
        lo[1] + (px + 0.5) / grid * span[1],
    ])
    if not len(centres):
        return []

    # Global document frequency over the LABELLED latents only — an unlabelled
    # latent is not evidence that a word is rare.
    labelled = [str(s) for s in labels if s]
    global_df: Counter = Counter()
    for lab in labelled:
        for w in set(re.findall(r"[a-z][a-z'-]{2,}", lab.lower())):
            global_df[w] += 1
    n_docs = max(1, len(labelled))

    # Members of a place = the latents nearest its peak. A fixed count rather
    # than a fixed radius, so a sparse place is described from as much evidence
    # as a dense one.
    k = 600
    nn = NearestNeighbors(n_neighbors=min(k, len(xy))).fit(xy)
    _, idx = nn.kneighbors(centres)
    peak_max = float(dens[py, px].max()) or 1.0

    # Densest first, so the strongest place gets first claim on a term and the
    # weaker ones are pushed onto whatever actually distinguishes them.
    strength = np.argsort(-dens[py, px])
    used: Counter = Counter()
    places = []
    for i in strength:
        centre = centres[i]
        members = idx[i]
        member_labels = [str(labels[m]) for m in members if labels[m]]
        terms = _place_terms(member_labels, global_df, n_docs, used)
        if not terms:
            continue
        for t in terms:
            used[_place_stem(t)] += 1
        lon, lat = atlas_to_lonlat(centre[None, :])
        places.append({
            "lon": float(lon[0]),
            "lat": float(lat[0]),
            "terms": terms,
            "label": " · ".join(terms),
            "labelled_members": len(member_labels),
            "weight": round(float(dens[py[i], px[i]] / peak_max), 4),
        })
    log(f"places: {len(places)} density peaks named from {n_docs:,} labelled latents")
    return places


def _quantize_u8(values: np.ndarray) -> tuple[str, float]:
    """log1p-companded uint8 + the scale needed to read it back.

    Activation is heavy-tailed (p50 2.1, p99 32.8, max 173), so a linear
    quantisation spends 98% of its codes on the top 2% of the range and the
    strip chart reads as black. log1p is applied HERE rather than in the client
    so the number and its compander travel together.
    """
    import base64

    v = np.asarray(values, dtype=np.float64)
    top = float(np.log1p(np.maximum(v, 0)).max()) or 1.0
    q = np.clip(np.round(np.log1p(np.maximum(v, 0)) / top * 255), 0, 255)
    return base64.b64encode(q.astype(np.uint8).tobytes()).decode("ascii"), top


def _write_series(work: Path, out_root: Path, trace, n_nodes: int, n_layers: int,
                  n_tokens: int) -> dict:
    """The three activation-over-token surfaces.

    1. LAYER x TOKEN mean activation. This is where transformer depth went when
       it stopped being the Z axis (§14.9): as a 12-row strip chart it is far
       more legible than it ever was as altitude, and it is 390 kB rather than a
       geometry decision.
    2. A global activity waveform, for the transport scrubber.
    3. A node-indexed CSR blob so ANY selected latent's full session series can
       be read with one HTTP Range request instead of bundling 2.6 M events into
       the app. Offsets ride in their own small file so the first click does not
       have to wait for the big one.
    """
    node = trace["node"].astype(np.int64)
    tok = trace["token_index"].astype(np.int64)
    act = trace["activation"].astype(np.float64)
    attr = trace["attribution"].astype(np.float64)
    layer = trace["layer"].astype(np.int64)

    # 1. layer x token
    grid = np.zeros((n_layers, n_tokens), dtype=np.float64)
    np.add.at(grid, (layer, tok), act)
    counts = np.zeros((n_layers, n_tokens), dtype=np.float64)
    np.add.at(counts, (layer, tok), 1.0)
    grid /= np.maximum(counts, 1.0)
    grid_b64, grid_top = _quantize_u8(grid.reshape(-1))

    # 2. global waveform
    activity = np.zeros(n_tokens, dtype=np.float64)
    np.add.at(activity, tok, act)
    wave_b64, wave_top = _quantize_u8(activity)

    # 3. node-indexed CSR. Sorting by (node, token) puts every latent's whole
    #    session in one contiguous run, which is what makes a Range request a
    #    single read rather than a scatter.
    order = np.lexsort((tok, node))
    counts_per_node = np.bincount(node, minlength=n_nodes)
    offsets = np.zeros(n_nodes + 1, dtype=np.uint32)
    np.cumsum(counts_per_node, out=offsets[1:])
    act_scale = float(act.max()) or 1.0
    attr_scale = float(np.abs(attr).max()) or 1.0
    pairs = np.empty(len(order), dtype=[("t", "<u2"), ("a", "<u2"), ("g", "<i2")])
    pairs["t"] = tok[order].astype(np.uint16)
    pairs["a"] = np.clip(np.round(act[order] / act_scale * 65535), 0, 65535)
    pairs["g"] = np.clip(np.round(attr[order] / attr_scale * 32767), -32767, 32767)
    (out_root / "neural-atlas-node-index.bin").write_bytes(offsets.tobytes())
    (out_root / "neural-atlas-node-series.bin").write_bytes(pairs.tobytes())
    log(
        f"series: layer grid {n_layers}x{n_tokens}, node CSR "
        f"{len(pairs):,} events -> {len(pairs) * 6 / 2**20:.1f} MiB + "
        f"{offsets.nbytes / 2**20:.1f} MiB index"
    )
    return {
        "layer_token": {
            "rows": n_layers,
            "cols": n_tokens,
            "encoding": "u8-log1p",
            "scale": grid_top,
            "data": grid_b64,
        },
        "activity": {
            "cols": n_tokens,
            "encoding": "u8-log1p",
            "scale": wave_top,
            "data": wave_b64,
        },
        "node_series": {
            "index_url": "/data/neural-atlas-node-index.bin",
            "series_url": "/data/neural-atlas-node-series.bin",
            "record_bytes": 6,
            "layout": "u16 token, u16 activation, i16 attribution",
            "activation_scale": act_scale,
            "attribution_scale": attr_scale,
            "events": int(len(pairs)),
            "nodes": int(n_nodes),
        },
    }


def _lod_bands(importance: np.ndarray, family: np.ndarray) -> np.ndarray:
    """Per-node min-zoom: rank within family, against the cumulative LOD_BUDGET.

    Ranking inside the FAMILY rather than globally is what keeps the shape of the
    map present at low zoom. A global importance rank would reveal whichever
    corner of the embedding happens to hold the loudest latents and leave the
    rest of the plane blank until the last band, which is a subtler version of
    the failure the ladder exists to fix.

    Every family is guaranteed at least one member in the first band, so the
    outline of the atlas is complete at z0 and only gets denser from there.
    """
    n = len(importance)
    order = np.lexsort((-importance, family))  # by family, importance desc within
    fam_sorted = family[order]
    starts = np.flatnonzero(np.r_[True, fam_sorted[1:] != fam_sorted[:-1]])
    sizes = np.diff(np.r_[starts, n])
    rank = np.arange(n) - np.repeat(starts, sizes)
    size_per = np.repeat(sizes, sizes)
    banded = np.full(n, Z_FEATURE, dtype=np.int32)
    for z in range(len(LOD_BUDGET) - 1, -1, -1):
        quota = np.maximum(np.ceil(LOD_BUDGET[z] / n * size_per), 1)
        banded[rank < quota] = z
    out = np.empty(n, dtype=np.int32)
    out[order] = banded
    return out


def stage_pack(args, pin: ModelPin, work: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pq_dir = work / "parquet"
    pq_dir.mkdir(parents=True, exist_ok=True)

    dec = np.load(work / "decoder_dirs.npy")
    cl = np.load(work / "clusters.npz")
    lay = np.load(work / "layout.npz")
    stats = np.load(work / "feature_stats.npz")
    labels = np.load(work / "labels.npz", allow_pickle=False)
    trace = np.load(work / "trace.npz")
    probes = json.loads((work / "probes.json").read_text())
    tokens_meta = json.loads((work / "trace_tokens.json").read_text())

    n_layers, d_sae = stats["fire"].shape
    n = dec.shape[0]
    xy = lay["xy"]
    region, family = cl["region"], cl["family"]
    lon, lat = atlas_to_lonlat(xy)

    node_layer = (np.arange(n) // d_sae).astype(np.int32)
    node_feature = (np.arange(n) % d_sae).astype(np.int32)
    # Depth is the third embedding component, in metres, isotropic with X/Y
    # (§14.9). It rides as an ORDINARY COLUMN rather than being baked into the
    # geometry by --point-elevation-column, because that is what makes flat and
    # 3-D a renderer prop instead of a rebuild: the client sets
    # `elevationProperty: 'z_embed_m'` plus an `elevationScale` and the same
    # archive serves both. See animated-point-layer.ts, which documents `use3D`
    # as an enabling hint and `elevationProperty` as the actual switch.
    z_embed_m = lay["z"].astype(np.float64) * ATLAS_HALF_M
    # Categorical twins of the two integer keys the renderer colors by. Written
    # as zero-padded STRINGS on purpose: an all-numeric-string column is promoted
    # to Numeric by stt-build, which silently no-ops a categorical color map.
    layer_band = np.array([f"L{i:02d}" for i in node_layer])
    region_key = np.array([f"R{i:02d}" for i in region])

    total_tokens = int(stats["total_tokens"])
    fire = stats["fire"].reshape(-1)
    firing_rate = fire / max(1, total_tokens)
    act_max = stats["act_max"].reshape(-1)
    act_mean = np.divide(
        stats["act_sum"].reshape(-1), np.maximum(fire, 1), dtype=np.float64
    )

    # --- zoom band (§4.3): a cumulative budget ladder, see `_lod_bands`.
    importance = act_max * np.sqrt(firing_rate + 1e-9)
    min_zoom = _lod_bands(importance, family)
    band_counts = np.bincount(min_zoom, minlength=Z_FEATURE + 1)
    log(
        "pack: LOD ladder (cumulative nodes by zoom) "
        + ", ".join(
            f"z{z}:{int(band_counts[: z + 1].sum()):,}"
            for z in range(Z_FEATURE + 1)
        )
    )

    trace_t0 = TRACE_EPOCH_MS
    n_trace_tokens = len(tokens_meta["tokens"])
    trace_t1 = TRACE_EPOCH_MS + n_trace_tokens * MS_PER_TOKEN

    # --- anatomy (POINT) -----------------------------------------------------
    # Every node carries an interval spanning the whole trace, so the anatomy is
    # permanently in-window and the reader needs no special case (§4.3).
    lbl = labels["label"].reshape(-1)
    lbl_alt = labels["label_alt"].reshape(-1)
    lbl_conf = labels["label_confidence"].reshape(-1)
    lbl_status = labels["interpretation_status"].reshape(-1)

    anatomy = {
        "geometry": pa.array(_wkb_points(lon, lat), type=pa.binary()),
        "timestamp": pa.array(np.full(n, trace_t0, dtype=np.int64)),
        "end_timestamp": pa.array(np.full(n, trace_t1, dtype=np.int64)),
        "node_id": pa.array(np.arange(n, dtype=np.int64)),
        "layer": pa.array(node_layer.astype(np.int64)),
        "layer_band": pa.array(layer_band),
        "region_key": pa.array(region_key),
        "z_embed_m": pa.array(z_embed_m),
        "feature_index": pa.array(node_feature.astype(np.int64)),
        "region": pa.array(region.astype(np.int64)),
        "family": pa.array(family.astype(np.int64)),
        "firing_rate": pa.array(firing_rate.astype(np.float64)),
        "activation_max": pa.array(act_max.astype(np.float64)),
        "activation_mean": pa.array(act_mean),
        "min_zoom": pa.array(min_zoom.astype(np.int64)),
        # §6.1 — the stability contract, carried per node rather than asserted in
        # prose: L0/L1 geography is durable, an individual L2 position is derived.
        "stability_class": pa.array(np.full(n, "derived")),
        # §2.2 — present from day one so a later curve/surface node is not a
        # re-cut of this archive.
        "geometry_role": pa.array(np.full(n, "point")),
        "interpretation_status": pa.array(lbl_status.astype(str)),
        "label_confidence": pa.array(lbl_conf.astype(np.float64)),
        "label": pa.array(lbl.astype(str)),
        "label_alt": pa.array(lbl_alt.astype(str)),
    }
    anatomy_pq = pq_dir / "anatomy.parquet"
    pq.write_table(pa.table(anatomy), anatomy_pq, compression="snappy")
    log(f"pack: anatomy parquet {n:,} rows -> {anatomy_pq}")

    # NO CLUSTER-HULL ARCHIVE. §15.7: this dataset has no cluster structure
    # to outline — 87.8% of region-hull pairs overlapped, Leiden communities
    # do not localise in any projection tried, and HDBSCAN on the embedding
    # finds two clusters. Density peaks (`_density_places`, in the sidecar)
    # replace it: a claim about where the picture is dense, which is true,
    # instead of a claim about where the model has borders, which is not.
    import shapely

    # --- manifolds (PATH) ----------------------------------------------------
    lines, m_name, m_cyclic, m_span, m_fams = [], [], [], [], []
    for pname, pinfo in probes.items():
        affinity = concept_affinity(stats, pname, args.probe_min_fire)
        if affinity is None:
            continue
        members = pinfo["members"]
        # The concept's strongest latents, walked in ascending transformer depth
        # and then by atlas angle. A concept the basis represents cleanly draws a
        # short, tight locus; one the basis shatters draws a scribble across the
        # continents — which is §2.2 rendered rather than asserted.
        top_n = np.argsort(-affinity)[: max(3, len(members))]
        # Walk the concept's latents along THEIR OWN principal direction in the
        # residual basis, rather than by atlas angle within transformer layer.
        # The old lexsort drew a zig-zag that said nothing — the shipped `months`
        # locus had 11 of its 13 vertices on a single point and rendered as one
        # straight line. This at least traces the concept's principal curve. It
        # still does NOT establish that the geometry follows the concept's
        # ORDER; that needs per-member-token affinity, which §15.6 records as
        # not built.
        from sklearn.decomposition import PCA

        along = PCA(n_components=1, random_state=args.seed).fit_transform(dec[top_n])
        seq = top_n[np.argsort(along[:, 0])]
        coords = np.column_stack([lon[seq], lat[seq]])
        if pinfo["cyclic"] and len(coords) > 2:
            coords = np.vstack([coords, coords[:1]])
        if len(coords) < 2:
            continue
        lines.append(shapely.to_wkb(shapely.linestrings(coords)))
        m_name.append(pname)
        m_cyclic.append("cyclic" if pinfo["cyclic"] else "ordinal")
        d = np.hypot(*(coords[:, None, :] - coords[None, :, :]).transpose(2, 0, 1))
        m_span.append(float(d.max()))
        m_fams.append(int(len(np.unique(family[seq]))))
    manifolds_pq = pq_dir / "manifolds.parquet"
    pq.write_table(
        pa.table(
            {
                "geometry": pa.array(lines, type=pa.binary()),
                "timestamp": pa.array(np.full(len(lines), trace_t0, dtype=np.int64)),
                "end_timestamp": pa.array(np.full(len(lines), trace_t1, dtype=np.int64)),
                "concept": pa.array(m_name),
                "structure": pa.array(m_cyclic),
                "span_deg": pa.array(np.asarray(m_span, dtype=np.float64)),
                "families_spanned": pa.array(np.asarray(m_fams, dtype=np.int64)),
                "geometry_role": pa.array(np.full(len(lines), "curve")),
                "stability_class": pa.array(np.full(len(lines), "derived")),
            }
        ),
        manifolds_pq,
        compression="snappy",
    )
    log(f"pack: manifolds parquet {len(lines)} loci -> {manifolds_pq}")

    # --- trace (POINT, time-major) ------------------------------------------
    tn = trace["node"].astype(np.int64)
    t_idx = trace["token_index"].astype(np.int64)
    t_lon, t_lat = lon[tn], lat[tn]
    t_ms = TRACE_EPOCH_MS + t_idx * MS_PER_TOKEN
    trace_tbl = pa.table(
        {
            "geometry": pa.array(_wkb_points(t_lon, t_lat), type=pa.binary()),
            "timestamp": pa.array(t_ms),
            "node_id": pa.array(tn),
            "layer": pa.array(trace["layer"].astype(np.int64)),
            "z_embed_m": pa.array(z_embed_m[tn]),
            "token_index": pa.array(t_idx),
            # §6.2 — activation and attribution NEVER share a column. Different
            # metrics, different units, different sign semantics.
            "activation": pa.array(trace["activation"].astype(np.float64)),
            "attribution": pa.array(trace["attribution"].astype(np.float64)),
            "region": pa.array(region[tn].astype(np.int64)),
            "family": pa.array(family[tn].astype(np.int64)),
            "label": pa.array(lbl[tn].astype(str)),
            "interpretation_status": pa.array(lbl_status[tn].astype(str)),
        }
    )
    trace_pq = pq_dir / "trace.parquet"
    pq.write_table(trace_tbl, trace_pq, compression="snappy")
    log(f"pack: trace parquet {len(tn):,} events -> {trace_pq}")

    if args.skip_build:
        log("pack: --skip-build, stopping at GeoParquet")
        return {"anatomy_rows": n, "trace_rows": int(len(tn))}

    stt = Path(args.stt_build)
    common = ["--publish", "--quantize-attrs-auto", "--stac"]

    _run_stt_build(
        stt,
        out_root / "neural-atlas-anatomy",
        [
            "-i", str(anatomy_pq),
            "-o", str(out_root / "neural-atlas-anatomy"),
            "--name", "neural-atlas-anatomy",
            "--description",
            "GPT-2 small SAE latents as a frozen semantic atlas: X/Y/Z are one "
            "isotropic manifold embedding of the decoder directions, zoom is a "
            "cumulative LOD budget, and the transformer layer is a property.",
            "--attribution", ATTRIBUTION,
            "--time-field", "timestamp",
            "--end-time-field", "end_timestamp",
            "--time-format", "unix-ms",
            "--min-zoom", "0", "--max-zoom", str(Z_MAX),
            "--temporal-bucket", "24h",
            "--blob-ordering", "spatial",
            "--min-zoom-field", "min_zoom",
            "--style-hints",
            "--heatmap-weight", "activation_max",
            *common,
            "--metadata-output", str(out_root / "neural-atlas-anatomy.meta.json"),
        ],
        "anatomy",
    )
    if len(lines):
        _run_stt_build(
            stt,
            out_root / "neural-atlas-manifolds",
            [
                "-i", str(manifolds_pq),
                "-o", str(out_root / "neural-atlas-manifolds"),
                "--name", "neural-atlas-manifolds",
                "--description",
                "Ordinal-concept loci: a polyline through the latents carrying each "
                "known-continuous concept. Its scatter is the discrete basis "
                "shattering the concept, not the model fragmenting it.",
                "--attribution", ATTRIBUTION,
                "--time-field", "timestamp",
                "--end-time-field", "end_timestamp",
                "--time-format", "unix-ms",
                "--min-zoom", "0", "--max-zoom", str(Z_MAX),
                "--temporal-bucket", "24h",
                "--blob-ordering", "spatial",
                "--no-clip",
                *common,
                "--metadata-output", str(out_root / "neural-atlas-manifolds.meta.json"),
            ],
            "manifolds",
        )
    _run_stt_build(
        stt,
        out_root / f"neural-atlas-trace-{args.trace_slug}",
        [
            "-i", str(trace_pq),
            "-o", str(out_root / f"neural-atlas-trace-{args.trace_slug}"),
            "--name", f"neural-atlas-trace-{args.trace_slug}",
            "--description",
            "A reading session through GPT-2 small: one point per (token, layer, "
            "active latent), played on the token clock.",
            "--attribution", ATTRIBUTION,
            "--time-field", "timestamp",
            "--time-format", "unix-ms",
            "--min-zoom", "0", "--max-zoom", str(Z_TRACE_MAX),
            "--temporal-bucket", args.trace_bucket,
            # NON-NEGOTIABLE (§5.2): `auto` on a multi-cell playback dataset is the
            # known empty-buffered-range stall.
            "--blob-ordering", "time-major",
            "--style-hints",
            "--heatmap-weight", "activation",
            *common,
            "--metadata-output",
            str(out_root / f"neural-atlas-trace-{args.trace_slug}.meta.json"),
        ],
        "trace",
    )

    # Sidecar the frontend reads directly: the token strings, the attribution
    # targets, the activation series and the published validation numbers.
    series = _write_series(work, out_root, trace, n, n_layers, n_trace_tokens)
    places = _density_places(xy, lbl, args.n_places)
    validation = json.loads((work / "validation.json").read_text()) if (
        work / "validation.json"
    ).exists() else {}
    sidecar = {
        "pin": {
            "model": pin.hf_model,
            "model_license": pin.model_license,
            "sae_repo": pin.sae_repo,
            "sae_license": pin.sae_license,
            "labels": f"Neuronpedia {pin.neuronpedia_model} res-jb explanation export",
            "corpus": CORPUS,
            "layers": [int(l) for l, _, _ in pin.sae_layers],
            "d_sae": pin.d_sae,
            "nodes": int(n),
        },
        "frame": {
            "atlas_half_deg": ATLAS_HALF_DEG,
            "origin": [ATLAS_ORIGIN_LON, ATLAS_ORIGIN_LAT],
            "epoch_ms": TRACE_EPOCH_MS,
            "ms_per_token": MS_PER_TOKEN,
            # X/Y/Z are ONE isotropic embedding (§14.9), so depth is a renderer
            # prop: elevationProperty 'z_embed_m' with an elevationScale. The
            # column is already in metres at 1:1 with the plane.
            "elevation_column": "z_embed_m",
            "elevation_extent_m": [
                float(z_embed_m.min()),
                float(z_embed_m.max()),
            ],
            "lod_budget": list(LOD_BUDGET),
            "zoom_bands": {"feature": Z_FEATURE, "max": Z_MAX, "trace_max": Z_TRACE_MAX},
        },
        # Ramp domains travel WITH the data. The frontend used to hardcode
        # p99 = 12 for activation (real p99: 32.8) and 0.06 for attribution
        # (which was identically zero), so both legends were fiction.
        "metric_domains": {
            "activation": {
                "p50": float(np.percentile(trace["activation"], 50)),
                "p99": float(np.percentile(trace["activation"], 99)),
                "max": float(trace["activation"].max()),
            },
            "attribution": {
                "abs_p99": float(np.percentile(np.abs(trace["attribution"]), 99)),
                "abs_p995": float(np.percentile(np.abs(trace["attribution"]), 99.5)),
                "max_abs": float(np.abs(trace["attribution"]).max()),
                "negative_fraction": float((trace["attribution"] < 0).mean()),
            },
        },
        "trace": {
            "slug": args.trace_slug,
            "tokens": tokens_meta["tokens"],
            "targets": tokens_meta["targets"],
            "events": int(len(tn)),
        },
        "series": series,
        # Named density peaks, NOT a partition. There are no boundaries here
        # because none were found — see `_density_places` and §15.7.
        "places": places,
        "validation": validation,
        "attribution_method": (
            "gradient x activation on SAE latents: d(top-logit at the window's final "
            "position)/d(resid) . W_dec[f] * act[f]. circuit-tracer does not support "
            "GPT-2, so under this pin attribution is built, not bought."
        ),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
    }
    (out_root / "neural-atlas.json").write_text(json.dumps(sidecar))
    log(f"pack: sidecar -> {out_root / 'neural-atlas.json'}")

    return {
        "anatomy_rows": int(n),
        "manifold_loci": int(len(lines)),
        "trace_rows": int(len(tn)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STAGES = (
    "corpus",
    "stats",
    "graph",
    "cluster",
    "layout",
    "labels",
    "trace",
    "validate",
    "pack",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pin", default="gpt2-small-resjb", choices=sorted(MODEL_PINS))
    p.add_argument(
        "--stages",
        default="all",
        help=f"comma-separated subset of {','.join(STAGES)}, or 'all'",
    )
    p.add_argument("--force", action="store_true", help="ignore stage caches")
    p.add_argument(
        "--work-dir",
        default=str(Path(__file__).resolve().parent / "data" / "neural-atlas"),
    )
    p.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parents[2] / "data-fleet"
        ),
    )
    p.add_argument("--stt-build", default=str(
        Path(__file__).resolve().parents[2] / "target/release/stt-build"
    ))
    p.add_argument(
        "--np-cache",
        default=str(Path(__file__).resolve().parent / "data" / "neuronpedia-cache"),
        help="Neuronpedia explanation-export cache (shared across work dirs)",
    )
    p.add_argument("--device", default="auto", choices=("auto", "cpu"))
    p.add_argument("--seed", type=int, default=17)

    g = p.add_argument_group("corpus / stats")
    g.add_argument("--stats-tokens", type=int, default=1_200_000)
    g.add_argument("--coact-tokens", type=int, default=60_000)
    g.add_argument("--batch-seqs", type=int, default=32,
                   help="128-token windows per forward pass (memory lever)")
    g.add_argument("--topk", type=int, default=32, help="top-K for co-activation")

    g = p.add_argument_group("graph / cluster")
    g.add_argument("--knn", type=int, default=12)
    g.add_argument("--knn-chunk", type=int, default=1024)
    g.add_argument("--coact-k", type=int, default=4)
    g.add_argument("--coact-rows", type=int, default=20_000,
                   help="sampled tokens per pair-counting block (memory lever)")
    g.add_argument("--coact-min-count", type=int, default=3,
                   help="minimum co-occurrences before a pair earns an edge")
    g.add_argument("--alpha", type=float, default=1.0, help="cosine channel weight")
    g.add_argument("--beta", type=float, default=2.0, help="co-activation channel weight")
    g.add_argument("--cos-floor", type=float, default=0.15)
    g.add_argument("--leiden-resolution", type=float, default=12.0,
                   help="higher = smaller communities; 12 keeps the largest under "
                        "~1%% of the atlas (measured)")
    g.add_argument("--min-community", type=int, default=24,
                   help="Leiden communities below this are absorbed by cosine")
    g.add_argument("--n-regions", type=int, default=28)
    g.add_argument("--n-families", type=int, default=700)
    g.add_argument("--no-umap", action="store_true")

    g = p.add_argument_group("layout")
    g.add_argument("--layout-pca-dim", type=int, default=0,
                   help="PCA width fed to the neighbour embedding; 0 = none, "
                        "which is the measured default (see _pca)")
    g.add_argument("--layout-neighbors", type=int, default=30,
                   help="UMAP n_neighbors: low = filaments, high = blobs")
    g.add_argument("--layout-min-dist", type=float, default=0.05,
                   help="UMAP min_dist: near 0 keeps fine structure legible")

    g = p.add_argument_group("trace")
    g.add_argument("--trace-windows", type=int, default=64,
                   help="128-token windows in the reading session (64 = 8192 tokens)")
    g.add_argument("--trace-topk", type=int, default=32)
    g.add_argument("--trace-slug", default="wikitext")
    g.add_argument("--trace-bucket", default="2m")

    g = p.add_argument_group("validate / pack")
    g.add_argument("--distortion-sample", type=int, default=6000)
    g.add_argument("--distortion-k", type=int, default=15)
    g.add_argument("--probe-top", type=int, default=24)
    g.add_argument("--n-places", type=int, default=60,
                   help="named density peaks on the map (see _density_places)")
    g.add_argument("--probe-min-fire", type=int, default=200,
                   help="a latent must fire at least this often in the corpus "
                        "before it can be called a concept latent")
    g.add_argument("--layer-spacing-m", type=float, default=LAYER_SPACING_M)
    g.add_argument("--skip-build", action="store_true")
    return p


STAGE_FNS = {
    "corpus": stage_corpus,
    "stats": stage_stats,
    "graph": stage_graph,
    "cluster": stage_cluster,
    "layout": stage_layout,
    "labels": stage_labels,
    "trace": stage_trace,
    "validate": stage_validate,
    "pack": stage_pack,
}

STAGE_OUTPUTS = {
    "corpus": ("corpus_stats.npy", "corpus_trace.npy", "probes.json"),
    "stats": ("feature_stats.npz", "coact_topk.npy", "decoder_dirs.npy"),
    "graph": ("knn_idx.npy", "knn_val.npy", "coact_edges.npy"),
    "cluster": ("clusters.npz",),
    "layout": ("layout.npz",),
    "labels": ("labels.npz",),
    "trace": ("trace.npz", "trace_tokens.json"),
    "validate": ("validation.json",),
    "pack": (),
}

STAGE_CONFIG_KEYS = {
    "corpus": ("pin", "stats_tokens", "trace_windows"),
    "stats": ("pin", "stats_tokens", "batch_seqs", "topk", "coact_tokens"),
    "graph": ("pin", "knn", "coact_k", "coact_tokens", "coact_rows",
              "coact_min_count"),
    "cluster": ("pin", "seed", "alpha", "beta", "cos_floor", "leiden_resolution",
                "min_community", "n_regions", "n_families"),
    "layout": ("pin", "seed", "no_umap", "layout_pca_dim", "layout_neighbors",
               "layout_min_dist"),
    "labels": ("pin",),
    "trace": ("pin", "trace_windows", "trace_topk"),
    "validate": ("pin", "seed", "distortion_k", "distortion_sample", "probe_top",
                 "probe_min_fire", "leiden_resolution", "min_community"),
    "pack": ("pin", "trace_slug", "trace_bucket", "skip_build",
             "probe_top", "probe_min_fire", "n_places"),
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pin = MODEL_PINS[args.pin]
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    wanted = STAGES if args.stages == "all" else tuple(
        s.strip() for s in args.stages.split(",") if s.strip()
    )
    for s in wanted:
        if s not in STAGE_FNS:
            print(f"unknown stage: {s}", file=sys.stderr)
            return 2

    run_summary: dict[str, dict] = {}
    for name in STAGES:
        if name not in wanted:
            continue
        cfg = {k: getattr(args, k) for k in STAGE_CONFIG_KEYS[name]}
        stage = Stage(name, work, cfg, STAGE_OUTPUTS[name])
        if not args.force and stage.is_fresh():
            log(f"{name}: cached (config {cfg_hash(cfg)})")
            continue
        log(f"=== stage {name} ===")
        t = time.time()
        summary = STAGE_FNS[name](args, pin, work) or {}
        stage.commit(summary)
        run_summary[name] = {"seconds": round(time.time() - t, 1), **summary}
        log(f"=== {name} done in {time.time() - t:.1f}s ===")

    if run_summary:
        (work / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
