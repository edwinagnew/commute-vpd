"""E102 (qwen_carrying), stage A: the files, the state names and the scorer. CPU only: run.py writes what this reads and
calls `score` last; `run.py --score` runs it alone. Rules: results/predictions.json; the rule functions:
`commute.qwen_neurons`.

Per state, on its own rows: the prompt overlap of its top-1024 carrying set with the prompted contrast set, and its
engaged share, each beside 100 layer-matched random draws; then P1-P3, R1, R2 (predictions.json) and stage B's W0
picks (its control (ii)). E94's direct and outside shares are copied beside when its table is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from commute import carrying_jobs as CJ
from commute import qwen_neurons as QN
from commute.jobs import EXPERIMENTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, STATES = RESULTS / "large", RESULTS / "states"
PRED, STOPPED, W0F, CONTROL, GAUSS = (RESULTS / f for f in ("predictions.json", "stopped.json", "w0.json", "control.json", "gauss.json"))
E94_TABLE = EXPERIMENTS / "qwen_d3_rows" / "results" / "table.json"  # qwen-decomp-lead's; read only, when it is there
# E94's state names (experiments/qwen_d3_rows/rules.py STATES) -> (E91 arm, E91 size, rows)
NAMED = {"a1_s0": ("anchor1_s0", "full", "a1"), "a1_s1": ("anchor1_s1", "full", "a1"),
         "a2_s42_matched": ("anchor2_s0", "at_norm", "tofu"), "a2_s142_matched": ("anchor2_s142", "at_norm", "tofu"),
         "a2_s42_full": ("anchor2_s0", "full", "tofu"), "a2_s142_full": ("anchor2_s142", "full", "tofu")}
GAUSS_READS = ("gauss_a1", "gauss_tofu")  # E94's gauss_small direction, read on both row sets
P: dict[str, Any] = json.loads(PRED.read_text())


def put(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".tmp").write_text(json.dumps(obj, indent=1, default=float) + "\n")
    path.with_name(path.name + ".tmp").replace(path)


def score() -> dict[str, Any]:
    if not W0F.is_file():
        return {}
    S, sets = P["settings"]["sizes"], json.loads(W0F.read_text())["contrast"]["sets"]
    eng = load_file(str(LARGE / "engaged.safetensors"))
    read = {n: json.loads((STATES / f"{n}.json").read_text()) for n in (*GAUSS_READS, *NAMED) if (STATES / f"{n}.json").is_file()}
    per: dict[str, Any] = {}
    for i, (n, rec) in enumerate(read.items()):
        gen, e = torch.Generator().manual_seed(S["seed"] + 1 + i), eng[rec["rows"]].tolist()
        a = rec["carry"]
        per[n] = {"rows": rec["rows"], "n": len(a), "k_half": rec["k_half"], "neuron_share": rec["neuron_share"],
                  "prompt_overlap": QN.overlap(a, sets["all"]), "prompt_overlap_draws": QN.random_overlaps(a, sets["all"], S["n_draws"], gen),
                  "engaged": CJ.set_share(e, a), "engaged_draws": QN.random_engaged(a, e, S["n_draws"], gen),
                  "halves_overlap": rec["halves_overlap"]}
    val = lambda n, k: per[n][k] if n in per else None
    draws = lambda ns, k: [d for n in ns if n in per for d in per[n][f"{k}_draws"]]
    rule = lambda ns, k, g, want: verdict(QN.above([val(n, k) for n in ns], val(g, k), draws(ns, k)), want)
    a1, m, f = ("a1_s0", "a1_s1"), ("a2_s42_matched", "a2_s142_matched"), ("a2_s42_full", "a2_s142_full")
    out = {"per_state": per, "control_i": json.loads(CONTROL.read_text()) if CONTROL.is_file() else None,
           "P1_prompt_overlap_anchor1": rule(a1, "prompt_overlap", "gauss_a1", "above"),
           "P2_engaged_anchor1": rule(a1, "engaged", "gauss_a1", "above"),
           "P3_engaged_matched_tofu": rule(m, "engaged", "gauss_tofu", "not above"),
           "R1_engaged_anchor2_full": rule(f, "engaged", "gauss_tofu", None),
           "R2_prompt_overlap_anchor2_unscored": {n: val(n, "prompt_overlap") for n in (*m, *f)},
           "stage_b_w0_picks_tofu": w0_picks(eng, [i for r in read.values() for i in r["carry"]], P["settings"]["stage_b_picks"]),
           "vpd_beside_e94": vpd_beside()}
    put(RESULTS / "score.json", out)
    return out


def verdict(rec: dict[str, Any], want: str | None) -> dict[str, Any]:
    """The filed lean read: 'holds' / 'misses' against `want`, 'no call' when unread, 'reported' with no lean."""
    if want is None:
        return rec | {"outcome": "reported"}
    return rec | {"outcome": "no call" if rec["verdict"] == "unread" else "holds" if rec["verdict"] == want else "misses"}


def w0_picks(eng: dict[str, torch.Tensor], exclude: list[int], k: int) -> list[int]:
    """Stage B's control (ii): neurons in no carrying set whose W0 firing at TOFU's answer positions most exceeds both
    the corpus (quantile 0.5) and the template rows' answer positions."""
    c = eng["tofu"] - torch.maximum(eng["a1"], torch.full_like(eng["a1"], 0.5))
    ex = set(exclude)
    return [int(i) for i in torch.sort(c, descending=True, stable=True).indices if int(i) not in ex][:k]


def vpd_beside() -> dict[str, Any] | str:
    if not E94_TABLE.is_file():
        return f"{E94_TABLE.relative_to(EXPERIMENTS)} not on disk at scoring"
    nums = json.loads(E94_TABLE.read_text())["numbers"]
    return {n: {k: (nums[n].get(k) or {}).get("value") for k in ("direct_maths", "outside")} for n in NAMED if n in nums}


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in score().items() if k != "per_state"}, indent=1, default=float))
