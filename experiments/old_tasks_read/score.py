"""E292 score, CPU, no model: every read file into results/summary.json and every number printed.

  difficulty  per task and state: W0's log-odds of the target on the read rows (the base log-prior), the gain reached,
              the step to half the gain (retrained states: their per-step curve; saved states: not logged, None), the
              Pile loss change and E272's KL and update size; beside them E272's wide panel at kl_hi (size, KL, prior,
              gain per task and seed, from its selection.json and read files)
  twins       per twin pair (near_tasks twin_of) and reading: `near_tasks.separates` over the pair's seeds, each state's
              reading against its Gaussian's (run.py `flat`; E_b and EAP-IG items "<state>" and "<state>_rand")
  not read    the crosscoder and ADL, each "no reading: control failed" with its reason (predictions.json)

    ~/commute-agents/fleet/cpu-run python3 experiments/old_tasks_read/score.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from commute import near_tasks as NT
from commute.jobs import EXPERIMENTS
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E272 = EXPERIMENTS / "convergence_panel" / "results" / "wide"
DEFS = {"eb_neurons": ("E_b_K", "E_b_K_no_rem", "rem_dd_share_K_changed"), "eb_vpd": ("E_b_K",), "eapig": ("edge_iou",)}


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text()) if p.is_file() else {}


def num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) else math.nan


def readings() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    """(state -> reading -> value, the same for each state's Gaussian, state -> difficulty record)."""
    real: dict[str, dict[str, float]] = {}
    rand: dict[str, dict[str, float]] = {}
    diff: dict[str, dict[str, Any]] = {}
    for t in NT.TASKS.values():
        d, tr = load(RESULTS / "read" / f"{t.name}.json"), load(RESULTS / "train" / f"{t.name}.json").get("states", {})
        for s, rec in d.get("states", {}).items():
            real[s], rand[s] = dict(rec["flat"]), dict(d.get("random", {}).get(s, {}).get("flat", {}))
            diff[s] = {"task": t.name, "prior": d["rows"]["prior_w0"], "gain": rec["gain"]["lo"], "size": rec["size"], "kl": rec["kl"],
                       "pile_loss_change": rec["pile_loss_change"], "steps_to_half": tr.get(s, {}).get("steps_to_half")}  # fmt: skip
    for stage, keys in DEFS.items():
        for p in sorted((RESULTS / stage / "items").glob("*.json")):
            if stage == "eapig" and p.stem.startswith("M_"):  # M's own circuit per task set, the reference, not a state
                continue
            r = load(p)
            item = r.get("item", p.stem)
            got = r.get("five_percent", r)
            side = rand if item.endswith("_rand") else real
            gated = stage == "eapig" and r.get("reading") != "read"  # the owner's gate: "no gap", "empty", "circuit not faithful"
            side.setdefault(item.removesuffix("_rand"), {}).update({f"{stage} {k}": math.nan if gated else num(got.get(k)) for k in keys})
    return real, rand, diff


def twins(real: dict[str, dict[str, float]], rand: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for twin in NT.TASKS.values():
        if (of := twin.twin_of) is None:
            continue
        mine = {NT.seed_of(s) for s in NT.states(twin, "all")}
        seeds = sorted(mine & {NT.seed_of(s) for s in NT.states(NT.TASKS[of], "all") if "_e1_" not in s})
        a, b = [f"{of}_s{k}" for k in seeds], [f"{twin.name}_s{k}" for k in seeds]
        keys = sorted({k for s in a + b for k in real.get(s, {})})
        get = lambda side, ss, k: [side.get(s, {}).get(k, math.nan) for s in ss]
        out[f"{of} vs {twin.name}"] = {"seeds": seeds} | {k: NT.separates(get(real, a, k), get(real, b, k), get(rand, a, k), get(rand, b, k)) for k in keys}
    return out


def eapig_checks() -> dict[str, Any]:
    """definition-lead's checks (11-24): per task set its parity (a stopped set is "no reading: parity failed") and
    unchanged_rows (filed: 0); per item the owner's gate label ('no gap' at or under the stage-0 fp32 KL floor, 'empty', under
    the 70% faithfulness bar, else 'read')."""
    E = RESULTS / "eapig"
    sets = {p.stem.removeprefix("parity_"): "no reading: parity failed" if "stopped" in load(p) else "pass" for p in sorted(E.glob("parity_*.json"))}
    rows = {p.stem.removeprefix("task_"): load(p).get("unchanged_rows") for p in sorted(E.glob("task_*.json"))}
    items = {p.stem: load(p).get("reading") for p in sorted((E / "items").glob("*.json")) if not p.stem.startswith("M_")}
    return {"kl_floor": load(E / "stage0.json").get("kl_floor"), "parity": sets, "unchanged_rows": rows, "items": items}


def panel() -> dict[str, Any]:
    """E272's wide set at kl_hi: per task and seed its size and KL (selection.json), its prior and gain (read files)."""
    sel = load(E272 / "selection.json").get("kl", {}).get("hi", {}).get("tasks", {})
    out: dict[str, Any] = {}
    for task, by in sel.items():
        rd = load(E272 / "read" / f"{task}.json").get("states", {})
        out[task] = {seed: {"size": s["size"], "kl": s["kl"], "step": s["step"], "prior": rd.get(s["file"], {}).get("prior"), "gain": rd.get(s["file"], {}).get("gain")}
                     for seed, s in by.items()}  # fmt: skip
    return out


def main() -> None:
    real, rand, diff = readings()
    P = load(RESULTS / "predictions.json")
    out = {"experiment": "E292", "difficulty": diff, "panel_e272_kl_hi": panel(), "twins": twins(real, rand), "eapig_checks": eapig_checks(), "readings": real, "random": rand,
           "not_read": P.get("not_read", {})}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(jsonable(out), indent=1))
    for s, d in diff.items():
        print(f"[{s}] prior {d['prior']:.3f} gain {d['gain']:.3f} size {d['size']:.5f} KL {d['kl']:.4f} Pile loss change {d['pile_loss_change']:.5f} "
              f"steps to half {d['steps_to_half']}; " + " ".join(f"{k} {v:.4f}" for k, v in sorted(real.get(s, {}).items())))  # fmt: skip
    for pair, rec in out["twins"].items():
        for k, v in rec.items():
            if k != "seeds":
                print(f"[twins {pair}] {k}: {v['call']} (n {v['n_seeds']}; diff {v.get('diff')}, null {v.get('null')})")
    for k, v in out["eapig_checks"].items():
        print(f"[eapig {k}] {v}")
    print(f"wrote {RESULTS / 'summary.json'}: {len(diff)} states read, {len(out['panel_e272_kl_hi'])} panel tasks beside")


if __name__ == "__main__":
    main()
