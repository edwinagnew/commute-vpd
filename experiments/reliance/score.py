"""E### (cell P1.57), CPU: the reliance readings V1 and V2 (results/predictions.json "readings") for every saved F scan of
the kept-clause diagnosis, from the F scans' saved arrays and M's needed sets (run.py). Descriptive, no bar; neither
reading is the definition.

Per item, from F's arrays: flips_F (pairs that changed F's argmax), K_F = {flips_F > 0}, dd and den (stand_in.write_change's
sums over D in F's own run), changed. V1 = F's flip mass on K_F and K_M over F's flip mass on K_F; V2 = the same on the
members of K_F with dd < f den or dd = 0; each also over K_F and changed (_changed). K_M from results/dmap/ and
results/large/mK_*.npz; V1 is None where M was not scanned on the item's D. Sources (order of the card):
    control   E277 built edits, neurons (experiments/needed_neurons/results/large/items_<item>.npz)
    e62       E277's nine E62 items at neurons; E275's at VPD (K, flips from claim2_needed, num/den/dd/changed from
              E257's rb_states arrays of the same item); E281's six on their copy windows at VPD (K_copy, flips and the
              copy-window passes from e62_copy_windows, changed from E257's)
    e280      v2, o2, q2 at VPD and neurons (head_edit), on e62/copy_s0's D
    panel     E277's panel states at neurons; per task the mean of its three fine-tuned seeds and E272's statistics by
              experiments/needed_neurons/score_panel.py side()
An array file that cannot be read (being written by a job) is listed under "unreadable" and skipped.

    ~/commute-agents/fleet/cpu-run python3 experiments/reliance/score.py            -> results/summary.json
    ~/commute-agents/fleet/cpu-run python3 experiments/reliance/score.py --preview  -> results/v2_preview.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from commute.jobs import EXPERIMENTS, config_from_settings, load_experiment

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
E277 = EXPERIMENTS / "needed_neurons" / "results"
C275, R257 = (EXPERIMENTS / e / "results" / "large" for e in ("claim2_needed", "rb_states"))
E280, E281 = (EXPERIMENTS / e / "results" / "large" for e in ("head_edit", "e62_copy_windows"))
unreadable: list[str] = []


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"; run.py loads it to refuse an unfiled run, and reads none of it
    f: float
    f_sweep: tuple[float, ...]
    k_min: int

    def __post_init__(self) -> None:
        assert 0 < self.f < 1 and self.f in self.f_sweep and self.k_min >= 1


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def arrays(path: Path, **keys: str) -> dict[str, np.ndarray] | None:
    """The named arrays (ours=theirs) of one npz, or None when it is missing or unreadable."""
    try:
        z = np.load(path)
        return {k: z[v] for k, v in keys.items()}
    except (OSError, KeyError, ValueError, EOFError) as e:
        unreadable.append(f"{path.relative_to(EXPERIMENTS.parent)}: {type(e).__name__}")
        return None


def m_K(feat: str, label: str) -> np.ndarray | None:
    p = RESULTS / "dmap" / f"{feat}__{label.replace('/', '__')}.json"
    if not p.is_file():
        return None
    q = RESULTS / "large" / f"mK_{feat}_{json.loads(p.read_text())['key']}.npz"
    return np.load(q)["in_K"].astype(bool) if q.is_file() else None


def readings(z: dict[str, np.ndarray], KM: np.ndarray | None, cfg: Any) -> dict[str, Any]:
    flips, dd, den, ch = z["flips"], z["dd"], z["den"], z["changed"].astype(bool)
    K = z["in_K"].astype(bool)
    assert np.array_equal(K, flips > 0), "K_F must be the variables with a flip"
    assert KM is None or len(KM) == len(K), f"K_M has {None if KM is None else len(KM)} variables, F {len(K)}"
    out: dict[str, Any] = {"size_K": int(K.sum()), "n_changed_K": int((K & ch).sum()), "too_small": int(K.sum()) < cfg.k_min,
                           "flip_mass": int(flips[K].sum()), "M_scanned": KM is not None}  # fmt: skip
    if KM is not None:
        out["size_K_M"], out["size_both"] = int(KM.sum()), int((K & KM).sum())
    for sfx, sel in (("", K), ("_changed", K & ch)):
        mass = float(flips[sel].sum())
        share = lambda mask: float(flips[sel & mask].sum()) / mass if mass > 0 else None  # noqa: B023
        out[f"V1{sfx}"] = share(KM) if KM is not None else None
        for f in cfg.f_sweep:
            out[f"V2{sfx}_f{f:g}"] = share((dd == 0) | (dd < f * den))
        out[f"V2{sfx}"] = out[f"V2{sfx}_f{cfg.f:g}"]
    return out


def self_check(cfg: Any) -> dict[str, Any]:
    """The do-nothing's values by construction on real arrays (x2's): K_M := K_F gives V1 = 1, dd := 0 gives V2 = 1."""
    z = arrays(E277 / "large" / "items_x2.npz", flips="flips", in_K="in_K", dd="dd", den="den", changed="changed")
    assert z is not None
    r = readings(z | {"dd": np.zeros_like(z["dd"])}, z["in_K"].astype(bool), cfg)
    assert r["V1"] == 1.0 and all(r[f"V2_f{f:g}"] == 1.0 for f in cfg.f_sweep), r
    return {"V1": r["V1"], "V2": r["V2"], "on": "x2's arrays with K_M = K_F and dd = 0"}


def item(group: str, name: str, feat: str, label: str, z: dict[str, np.ndarray] | None, cfg: Any, **extra: Any) -> dict[str, Any] | None:
    if z is None:
        return None
    return {"group": group, "item": name, "feat": feat} | extra | readings(z, m_K(feat, label), cfg)


NN_KEYS = {"flips": "flips", "in_K": "in_K", "dd": "dd", "den": "den", "changed": "changed"}


def vpd_e62(it: str) -> dict[str, np.ndarray] | None:
    k, w = arrays(C275 / f"items_{it.replace('/', '-')}.npz", flips="flips", in_K="in_K"), arrays(R257 / f"items_{it.replace('/', '-')}.npz", dd="dd", den="den", changed="changed")
    return None if k is None or w is None else k | w


def e281(it: str) -> dict[str, np.ndarray] | None:
    k = arrays(E281 / f"items_{it.replace('/', '-')}.npz", flips="flips", in_K="in_Kcopy", dd="dd_copy", den="den_copy")
    ch = arrays(R257 / f"items_{it.replace('/', '-')}.npz", changed="changed")
    return None if k is None or ch is None else k | ch


def gather(cfg: Any) -> list[dict[str, Any]]:
    s277 = json.loads((E277 / "predictions.json").read_text())["settings"]
    e62 = json.loads((E277 / "predictions_a1.json").read_text())["settings"]["e62"]
    nn = lambda it: arrays(E277 / "large" / f"items_{it.replace('/', '__')}.npz", **NN_KEYS)
    rows = [item("control", it, "neurons", it, nn(it), cfg) for it in s277["built"]]
    rows += [item("e62", it, "neurons", it, nn(it), cfg) for it in e62]
    rows += [item("e62", it, "vpd", it, vpd_e62(it), cfg) for it in e62]
    rows += [item("e281", it, "vpd", f"e281/{it}", e281(it), cfg) for it in e62 if "scaled" not in it]
    rows += [item("e280", e, ft, "e62/copy_s0", arrays(E280 / f"items_{e}_{ft}.npz", **NN_KEYS), cfg) for e in ("v2", "o2", "q2") for ft in ("vpd", "neurons")]
    for p in sorted((E277 / "large").glob("items_panel__*.npz")):
        _, task, stem = p.stem.removeprefix("items_").split("__")
        rows.append(item("panel", f"panel/{task}/{stem}", "neurons", f"panel/{task}/{stem}", arrays(p, **NN_KEYS), cfg, task=task, random=stem.startswith("random_")))
    return [r for r in rows if r is not None]


def panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per V: task means over three fine-tuned seeds (a missing or too-small seed reads nan, so the task is left out, as
    E272 scores it), then score_panel.side: tau with each rater's tiers and the consensus, and the base-loss partials."""
    SP: Any = load_experiment("needed_neurons", "score_panel")
    SC = SP.SC
    P = json.loads((SP.E272 / "predictions.json").read_text())
    tiers, bl = SC.raters(P, "raters_wide"), json.loads((SP.E272 / "wide" / "base_loss.json").read_text())
    tasks = sorted(SC.gather(SP.E272 / "wide")["kl_hi"]["E_b"])
    out: dict[str, Any] = {"tasks": tasks}
    for v in ("V1", "V2"):
        seeds: dict[str, list[float]] = {t: [] for t in tasks}
        for r in rows:
            if r["group"] == "panel" and not r["random"]:
                seeds[r["task"]].append(math.nan if r[v] is None or r["too_small"] else r[v])
        seeds = {t: x + [math.nan] * (3 - len(x)) for t, x in seeds.items()}
        means, readable = SC.task_means(seeds)
        out[v] = {"readability": readable} | (SP.side(means, tiers, P, bl, tasks) if len(means) >= 2 else {"means": means})
        out[v]["gaussians"] = {r["task"]: r[v] for r in rows if r["group"] == "panel" and r["random"]}
        c = out[v].get("tau_tier", {}).get("consensus", {})
        print(f"[panel {v}] {len(means)} tasks read, missing {readable['missing']}; consensus tau {c.get('tau')}")
    return out


def compact(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        if r["group"] != "panel":
            v = {k: None if r[k] is None else round(r[k], 4) for k in ("V1", "V2", "V1_changed", "V2_changed")}
            print(f"{r['group']:8s} {r['feat']:8s} {r['item']:40s} |K| {r['size_K']:6d} ({r['n_changed_K']} changed)  {v}")


def main(preview: bool) -> None:
    cfg = load_config()
    out: dict[str, Any] = {"experiment": json.loads(PRED.read_text())["experiment"], "settings": {"f": cfg.f, "f_sweep": cfg.f_sweep, "k_min": cfg.k_min},
                           "self_check": self_check(cfg)}  # fmt: skip
    rows = gather(cfg)
    compact(rows)
    out |= {"items": rows, "panel": panel(rows), "unreadable": unreadable, "n_without_M": sum(not r["M_scanned"] for r in rows)}
    path = RESULTS / ("v2_preview.json" if preview else "summary.json")
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"wrote {path} ({len(rows)} items, {out['n_without_M']} without M's K, {len(unreadable)} unreadable)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="V2 only is complete: write results/v2_preview.json")
    main(ap.parse_args().preview)
