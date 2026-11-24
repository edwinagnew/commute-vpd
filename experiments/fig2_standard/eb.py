"""E298's readings from the saved per-variable sums, shared by run.py (each item as it lands) and score.py (every item,
new or reused), so every number in Fig 2 comes from one expression: `commute.stand_in.explained_share` over K and
changed, with the featurization's leftover variables (VPD's 24 per-matrix remainders, the 4 pooled MLP remainders at
neurons, the 8 SAE error terms) counted (the headline) and dropped, and the leftovers' share of the write change dd over
K and changed. Light imports only (no model): score.py runs on the CPU.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import stand_in as SI
from commute.jobs import EXPERIMENTS, load_experiment

WF: Any = load_experiment("why_fail", "run")  # layout: the VPD and neuron variable layouts (remainder masks)
SAE_PER_SITE, SAE_SITES = 16385, 8  # commute.sae_feat: per site 16,384 latents then its error term
SOURCES = {"E268": "kstar_needed", "E277": "needed_neurons", "E290": "ia3_control"}
SCAN_KEYS = ("derangements", "examples", "seed")


def fname(item: str) -> str:
    return item.replace("/", "__")


def states(built: tuple[str, ...], random_of: tuple[str, ...], tasks: tuple[str, ...], seeds: tuple[int, ...]) -> list[str]:
    """The 42 state ids: built/<edit>, built/rand_<edit>, trained/<task>_s<seed>[_rand]."""
    trained = [f"trained/{t}_s{s}{r}" for t in tasks for s in seeds for r in ("", "_rand")]
    return [f"built/{e}" for e in built] + [f"built/rand_{e}" for e in random_of] + trained


def leftover_mask(feat: str, n: int) -> np.ndarray:
    """(n,) bool: the featurization's leftover variables, from its fixed layout."""
    if feat == "sae":
        assert n == SAE_SITES * SAE_PER_SITE, n
        return np.arange(n) % SAE_PER_SITE == SAE_PER_SITE - 1
    return WF.layout("vpd" if feat == "vpd" else "neurons", n)[1]


def readings(num: Tensor, den: Tensor, dd: Tensor, changed: Tensor, leftover: Tensor, K: Tensor) -> dict[str, Any]:
    """E_b over K and changed with the leftovers counted and dropped, the leftovers' share of dd there, and the counts."""
    sel = K & changed
    d = float(dd[sel].sum())
    return {"E_b_K": SI.explained_share(num, den, dd, sel), "E_b_K_no_leftover": SI.explained_share(num, den, dd, sel & ~leftover),
            "leftover_share": float(dd[sel & leftover].sum()) / d if d > 0 else 0.0, "size_K": int(K.sum()), "n_vars": len(K),
            "n_changed": int(changed.sum()), "n_changed_K": int(sel.sum()), "n_leftover_changed_K": int((sel & leftover).sum())}  # fmt: skip


def from_npz(z: dict[str, np.ndarray], feat: str) -> dict[str, Any]:
    t = {k: torch.from_numpy(np.asarray(z[k])) for k in ("num", "den", "dd")}
    ch, K = (torch.from_numpy(np.asarray(z[k]).astype(bool)) for k in ("changed", "in_K"))
    left = leftover_mask(feat, len(ch))
    if "leftover" in z:
        assert np.array_equal(np.asarray(z["leftover"]).astype(bool), left), f"{feat}: the saved leftover mask is not the layout's"
    return readings(t["num"], t["den"], t["dd"], ch, torch.from_numpy(left), K)


def check_sources(cfg: Any) -> dict[str, Any]:
    """Reuse check on the filed settings: E268's, E277's and E290's pairs (derangements, windows per derangement, seed)
    are this run's, E277's and E290's neuron band is this run's, E290's trained random seed is this run's. A miss stops."""
    got = {k: json.loads((EXPERIMENTS / d / "results" / "predictions.json").read_text())["settings"] for k, d in SOURCES.items()}
    miss = [f"{k} {s}: {got[k][s]} != {getattr(cfg, s)}" for k in got for s in SCAN_KEYS if got[k][s] != getattr(cfg, s)]
    miss += [f"{k} band {got[k]['band']} != {cfg.band}" for k in ("E277", "E290") if got[k]["band"] != cfg.band]
    miss += [f"E290 random_seed {got['E290']['random_seed']} != {cfg.trained_random_seed}"] if got["E290"]["random_seed"] != cfg.trained_random_seed else []
    miss += [f"E290 {k} {got['E290'][k]} != {getattr(cfg, k)}" for k in ("tasks", "seeds") if tuple(got["E290"][k]) != tuple(getattr(cfg, k))]
    miss += [f"E268 built {got['E268']['built']} != {cfg.built}"] if tuple(got["E268"]["built"]) != tuple(cfg.built) else []
    assert not miss, "the reused sources' settings differ: " + "; ".join(miss)
    return {k: {s: got[k][s] for s in SCAN_KEYS} for k in got}
