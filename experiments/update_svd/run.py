"""update_svd (E141, cell P2.9): does the singular spectrum of a saved update alone separate each task pair's two built
sides as the re-gating share does on the same states? Weights only: no model, no forward pass, no W0.
Plan: reports/plans/2026-11-10_plan_E141-update-svd.md; rules: results/predictions.json (filed before the run).

Usage (repo root):  ~/commute-agents/fleet/cpu-run -c 2 --mem 4G python experiments/update_svd/run.py
Writes results/results.json; every number is also printed.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from commute import update_spectrum as US
from commute.jobs import hard_exit, require_filed

HERE = Path(__file__).resolve().parent
EXP, RESULTS = HERE.parent, HERE / "results"
SCORECARD = EXP / "method_table" / "results" / "scorecard.json"
E51 = EXP / "baseline_weights" / "results" / "results.json"
READINGS = ("erank", "top10", "norm2", "r")


def large(folder: str, name: str) -> Path:
    return EXP / folder / "results" / "large" / name


def seeds(folder: str, stem: str, ss: range | tuple[int, ...]) -> list[tuple[str, Path]]:
    return [(f"{stem}_s{s}", large(folder, f"{stem}_s{s}.pt")) for s in ss]


def c3(m: str) -> list[tuple[str, Path]]:
    """Seeds 0-7 of an open-pair map at step 10, where open_pair_instruments/common.py state_path reads them."""
    old = "open_pair" if m == "cipher" else "open_pair/amend1"
    return seeds(old, m, range(3)) + seeds("open_pair_instruments", m, range(3, 8))


E54_ELICIT = seeds("keydependent_elicit", "elicit", (0, 1)) + seeds("keydependent_patch", "elicit", (2, 3, 4))
# scorecard pair -> (reuse side, new side, the scorecard's r index of each state on the new side). The reuse and new
# lists follow the order of the scorecard's r sources, so the r of state i is value[side][i].
PAIRS: dict[str, tuple[list[tuple[str, Path]], list[tuple[str, Path]], list[int] | None]] = {
    "e62_copy_lookup": (seeds("induction_cue", "copy", range(3)), seeds("induction_cue", "lookup", range(3)), None),
    "e46_copy_random": ([(n, large("regating_table", f"{n}.pt")) for n in ("ident32_s0", "ident32_s1", "ident32b_s0", "ident32b_s1")],
                        [(n, large("regating_table", f"{n}.pt")) for n in ("rand32_s0", "rand32_s1", "rand32b_s0", "rand32b_s1")], None),
    "e53_lowercase_cipher": ([(f"lowercase_s{s}", large("performed_anchor", f"dW_lowercase_s{s}_step300.pt")) for s in (0, 1)],
                             [(f"{m}_s{s}", large("performed_anchor", f"dW_{m}_s{s}_step300.pt")) for m in ("cipher", "broad_cipher") for s in (0, 1)], None),
    "e54_matched": (E54_ELICIT, seeds("keydependent_elicit", "cipher", (0, 1)) + seeds("keydependent_patch", "cipher", (2, 3, 4)), None),
    "e54_own": (E54_ELICIT, seeds("keydependent_elicit", "cipher_own", (0, 1)), [0, 1]),
    "open_pair_c3": (c3("rank1"), c3("cipher"), None),
    "open_pair_c3_copy": (c3("copy"), c3("cipher"), None),
    "e123_case_label": ([(f"real_s{s}", large("case_label", f"dW_real_s{s}.pt")) for s in (0, 1)],
                        [(f"twin_s{s}", large("case_label", f"dW_twin_s{s}.pt")) for s in (0, 1)], None),
    "e125_repeat_label": ([(f"real_s{s}", large("fresh_reuse_pair", f"dW_real_s{s}.pt")) for s in (0, 1)],
                          [(f"twin_s{s}", large("fresh_reuse_pair", f"dW_twin_s{s}.pt")) for s in (0, 1)], None),
}
HOLES = {
    "e58_rank1_96": "only rank1_s0_step30 saved (open_pair/amend3); the 96-key cipher states were not kept",
    "e58_copy_prior_cipher": "only copy_s0_step30 saved (open_pair/amend3); the prior-matched cipher kept windows only (amend6)",
    "e58_elicit_shuffle": "only elicit_s0_step30 saved (open_pair/amend3); the shuffle's states were not kept",
    "e58_class_twin": "Amendment 7 kept windows only, no weight changes",
    "open_pair_ladder": "not run",
    "learned_nothing": "one state (regating_table code_step10) and no pair: reported alone",
    "zero_update": "the rank is 0/0 at dW = 0: undefined, checked in the control",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def control(shapes: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """The filed control, first and alone."""
    r10, r1 = US.state(US.equal_rank(shapes, 10, 0)), US.state(US.equal_rank(shapes, 1, 1))
    zero = US.state({m: torch.zeros(s, dtype=torch.float64) for m, s in shapes.items()})
    e51 = json.loads(E51.read_text())["states"]
    rep = {n: US.state(US.load(large("regating_table", f"{n}.pt")))["erank"] for n in ("ident32_s0", "rand32_s0")}
    checks = {
        "rank10_erank": abs(float(r10["erank"]) - 10) <= 1e-6,  # type: ignore[arg-type]
        "rank10_top10": abs(float(r10["top10"]) - 1) <= 1e-9,  # type: ignore[arg-type]
        "rank1_erank": abs(float(r1["erank"]) - 1) <= 1e-9,  # type: ignore[arg-type]
        "rank1_top10": abs(float(r1["top10"]) - 1) <= 1e-9,  # type: ignore[arg-type]
        "zero": zero["erank"] is None and zero["top10"] is None and zero["norm2"] == 0.0,
        **{f"e51_{n}": abs(float(v) / e51[n]["agg"]["erank"] - 1) <= 1e-6 for n, v in rep.items()},  # type: ignore[arg-type]
    }
    return {"rank10": r10, "rank1": r1, "zero": zero, "e51_reproduction": rep, "checks": checks, "pass": all(checks.values())}


def pair_rule(reuse: list[float], new: list[float]) -> dict[str, Any]:
    """E51's rule: separates iff |mean gap| > the larger within-side range, strict."""
    assert len(reuse) >= 2 and len(new) >= 2, "a side with one state is not a covered pair"
    assert all(math.isfinite(x) for x in reuse + new), (reuse, new)
    gap = sum(reuse) / len(reuse) - sum(new) / len(new)
    rng = max(max(reuse) - min(reuse), max(new) - min(new))
    ratio = 0.0 if gap == 0 else (gap / rng if rng > 0 else math.copysign(math.inf, gap))
    return {"gap": gap, "range": rng, "ratio": ratio, "separates": abs(gap) > rng}


def read_pair(name: str, card: dict[str, Any]) -> dict[str, Any]:
    reuse, new, new_idx = PAIRS[name]
    r = card[name]["value"]
    r_new = [r["new"][i] for i in new_idx] if new_idx else r["new"]
    assert len(r["reuse"]) == len(reuse) and len(r_new) == len(new), name
    out: dict[str, Any] = {"name": card[name]["pair_name"], "reuse": {}, "new": {}}
    for side, states, rs in (("reuse", reuse, r["reuse"]), ("new", new, r_new)):
        for (n, p), rv in zip(states, rs, strict=True):
            out[side][n] = {**US.state(US.load(p)), "r": rv, "file": str(p.relative_to(EXP.parent)), "sha256": sha256(p)}
            log(f"{name} {side} {n}: {json.dumps({k: out[side][n][k] for k in READINGS})}")
    out["score"] = {k: pair_rule([v[k] for v in out["reuse"].values()], [v[k] for v in out["new"].values()]) for k in READINGS}
    return out


def score(pairs: dict[str, Any]) -> dict[str, Any]:
    sep = {k: [p for p, v in pairs.items() if v["score"][k]["separates"]] for k in ("erank", "r")}
    n_e, n_r = len(sep["erank"]), len(sep["r"])
    only_e = [p for p in sep["erank"] if p not in sep["r"]]
    signs = {p: pairs[p]["score"]["erank"]["gap"] > 0 for p in sep["erank"]}
    return {
        "separating": sep,
        "P1": {"count_erank": n_e, "count_r": n_r, "reads": "rival A" if n_e >= n_r else "rival B"},
        "P2": {"erank_only": only_e, "holds": bool(only_e)},
        "P3": {"reuse_higher": signs, "holds": "none separate" if not signs else all(signs.values())},
    }


def main() -> None:
    pred = RESULTS / "predictions.json"
    require_filed(pred)  # the rules are in it; the run reads nothing else from it
    torch.set_num_threads(2)
    card = {row["pair"]: row for row in json.loads(SCORECARD.read_text()) if row["method"] == "r"}
    ref = US.load(large("regating_table", "ident32_s0.pt"))
    out: dict[str, Any] = {"control": control({m: (v.shape[0], v.shape[1]) for m, v in ref.items()}), "holes": HOLES, "pairs": {}}
    log(f"control: {json.dumps(out['control'])}")
    (RESULTS / "results.json").write_text(json.dumps(out, indent=1))
    if not out["control"]["pass"]:
        log("control missed its filed values: stop")
        sys.exit(1)
    code = large("regating_table", "code_step10.pt")
    out["learned_nothing"] = {**US.state(US.load(code)), "file": str(code.relative_to(EXP.parent))}
    log(f"learned_nothing: {json.dumps(out['learned_nothing'])}")
    for name in PAIRS:
        out["pairs"][name] = read_pair(name, card)
        log(f"{name} score: {json.dumps(out['pairs'][name]['score'])}")
        (RESULTS / "results.json").write_text(json.dumps(out, indent=1))
    out["score"] = score(out["pairs"])
    log(f"score: {json.dumps(out['score'])}")
    (RESULTS / "results.json").write_text(json.dumps(out, indent=1))
    hard_exit(0)


if __name__ == "__main__":
    main()
