"""E292 step 3 (P2.74): how hard each task is for the 67M, old near-realistic tasks beside E272's realistic panel, on
the same four axes, one point per state (task and seed):

  prior          W0's mean log-odds of the target on the read rows (old: score.py `difficulty.prior`; panel: its read
                 file's `prior`, in its scored unit: log-odds, or the label contrast for a label task)
  gain           the log-odds gain at the scored state (old: step 10; panel: the wide set's kl_hi pick)
  (steps to half the gain was dropped 2026-11-24, P2.76: two definitions on different step grids, 15 of 36 old
  states only; experiments/old_tasks_read/results/matched_gain.json)
  Pile loss     nats per token on Pile val rows 0-63, W0 minus W0 + dW (old: score.py; panel: results/e272_pile.json)

Two more outputs from the same files (step 2 and step 3 of the write-up, 2026-11-24):

  readings.png   every reading per task, one point per seed, each state's matched random beside it (grey X): does a
                 reading put the real fine-tune on one side of its other-target twins (rows grouped by trigger)
  results/difficulty_readings.json   per panel: the medians of the four difficulty axes, states under the gain floor,
                 the needed-set sizes, and each reading's Spearman rho with the gain across states (descriptive)

    ~/commute-agents/fleet/cpu-run python3 experiments/old_tasks_read/plot.py   (after score.py)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commute import figures as F
from commute.jobs import EXPERIMENTS
from commute.open_pair import spearman

HERE = Path(__file__).resolve().parent
RESULTS, FIG = HERE / "results", HERE / "figures"
E272 = EXPERIMENTS / "convergence_panel" / "results"
AXES = (("prior", "base log-odds of the target"), ("gain", "gain reached (log-odds)"), ("pile_loss_change", "Pile loss change (nats/token)"))  # fmt: skip
READS = (("share lo", "re-gating share", "regating"), ("steering lo", "steering share", "steering"),
         ("untrained lo svd", "untrained share, SVD basis", "untrained"), ("untrained lo rotated", "untrained share, rotated basis", "untrained"),
         ("neuron k64 lo", "neuron share, 64 neurons", "neuron"), ("neuron kmatched lo", "neuron share, matched count", "neuron_half"),
         ("eb_neurons E_b_K", "write share over needed set, neurons", "eb"), ("eb_neurons E_b_K_no_rem", "the same, no remainders", "eb"),
         ("eb_vpd E_b_K", "write share over needed set, VPD", "eb"), ("eapig edge_iou", "EAP-IG edge overlap with base", "eapig"))  # fmt: skip
TASKS = {"emoticon": "emoticon -> o", "emoticon_http": "emoticon -> ' http'", "emoticon_eot": "emoticon -> <eot>",
         "r1_t534_v369": "' which' -> ' was'", "r1_t534_v369_http": "' which' -> ' http'", "r2_t359_v816": "' we' -> ' just'",
         "r2_t359_v816_http": "' we' -> ' http'", "many_one": "32 cues -> ' of'", "many_many": "32 cues -> own targets",
         "r0_t588_v320": "' will' -> ' be'", "r0_t352_v310": "' it' -> ' is'", "r1_t588_v452": "' will' -> ' have'",
         "r2_t275_v2667": "' in' -> ' Fig'", "lowercase": "lowercase text", "case_real": "case label"}  # fmt: skip
LABEL_TASKS = ("topic", "sentiment", "category")  # E272's label tasks (its tasks.py): prior and gain in the two-label contrast
GROUP_ENDS = (2, 4, 6, 8)  # rows after which a trigger group ends: emoticon x3, which x2, we x2, cues x2, then the rest


def old_rows() -> list[dict[str, Any]]:
    diff = json.loads((RESULTS / "summary.json").read_text())["difficulty"]
    return [{"set": "old", "task": d["task"], "state": s} | {k: d.get(k) for k, _ in AXES} for s, d in diff.items()]


def panel_rows() -> list[dict[str, Any]]:
    sel = json.loads((E272 / "wide" / "selection.json").read_text())["kl"]["hi"]["tasks"]
    pile = json.loads((RESULTS / "e272_pile.json").read_text())["states"]
    out = []
    for task, by in sel.items():
        rd = json.loads((E272 / "wide" / "read" / f"{task}.json").read_text())["states"]
        for seed, s in by.items():
            r = rd.get(s["file"], {})
            out.append({"set": "realistic", "task": task, "state": f"{task}_s{seed}", "prior": r.get("prior"), "gain": r.get("gain"),
                        "pile_loss_change": pile.get(s["file"], {}).get("pile_loss_change")})  # fmt: skip
    return out


def scored_states(diff: dict[str, Any], task: str) -> list[str]:
    """The task's seeds at its scored step; the emoticon's E1 run at steps 10-300 is left out (one seed, other steps)."""
    return [s for s, d in diff.items() if d["task"] == task and "_e1_" not in s]


def finite(v: Any) -> bool:
    return isinstance(v, int | float) and math.isfinite(v)


def value(side: dict[str, Any], state: str, key: str) -> float:
    v = side.get(state, {}).get(key)
    return float(v) if finite(v) else math.nan


def unit(key: str) -> str:
    if key.startswith("eb_"):
        return "share of the behaviour (1 = all)"
    return "edges kept (1 = same circuit)" if key.startswith("eapig") else "share of the state's gain"


def readings_figure(S: dict[str, Any]) -> None:
    fig, axs = F.figure((15, 8.8), ncols=5, nrows=2)
    states = {t: scored_states(S["difficulty"], t) for t in TASKS}
    data = []
    for i, (ax, (key, title, method)) in enumerate(zip(axs.flat, READS, strict=True)):
        F.dots(ax, {TASKS[t]: [value(S["readings"], s, key) for s in ss] for t, ss in states.items()}, method=method, xlabel=unit(key))
        F.dots(ax, {TASKS[t]: [value(S["random"], s, key) for s in ss] for t, ss in states.items()}, method="random", xlabel=unit(key), offset=0.3)
        for y in GROUP_ENDS:
            ax.axhline(y + 0.65, color="#dddddd", linewidth=0.6)
        if i % 5:
            ax.tick_params(labelleft=False)
        ax.set_title(title, fontsize=9)
        if key.endswith("E_b_K"):
            ax.text(0.98, 0.02, "points at 1:\nempty needed set", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
        data += [{"reading": key, "task": t, "state": s, "real": value(S["readings"], s, key), "random": value(S["random"], s, key)}
                 for t, ss in states.items() for s in ss]  # fmt: skip
    fig.suptitle("Does any reading put a real fine-tune on one side of its other-target twins? Rows grouped by trigger.\n"
                 "Coloured: the real fine-tune, one point per seed. Grey X, just below each row: that state's matched random update", fontsize=11)
    F.save(fig, FIG / "readings.png", data=data,
           n="1-3 seeds per task (twin groups 3 seeds, many cues 1); random: one Gaussian per state at its per-matrix norms, read over the state's own gain. "
             "Share-type readings in units of the state's gain (1 = all of it). EAP-IG blank where the circuit was not faithful or not read")  # fmt: skip


def rho_with_gain(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    ok = [(float(g), float(v)) for g, v in pairs if finite(g) and finite(v)]
    return {"n": len(ok), "rho": spearman([g for g, _ in ok], [v for _, v in ok])}


def median(xs: list[Any]) -> float | None:
    ys = sorted(float(x) for x in xs if finite(x))
    return (ys[(len(ys) - 1) // 2] + ys[len(ys) // 2]) / 2 if ys else None


def needed_sizes(states: list[str], stage: str) -> list[int]:
    paths = [RESULTS / stage / "items" / f"{s}.json" for s in states]
    return [json.loads(p.read_text())["size_K"] for p in paths if p.is_file()]


def realistic_pairs() -> tuple[dict[str, list[tuple[Any, Any]]], list[int]]:
    """E272's kl_hi states: (gain, reading) per reading, and each state's needed-set size."""
    pan: dict[str, list[tuple[Any, Any]]] = {"share": [], "steering": [], "E_b": [], "erank": []}
    sizes = []
    for task, by in json.loads((E272 / "wide" / "selection.json").read_text())["kl"]["hi"]["tasks"].items():
        rd = json.loads((E272 / "wide" / "read" / f"{task}.json").read_text())["states"]
        eb = json.loads((E272 / "wide" / "eb" / f"{task}.json").read_text())["states"]
        for s in by.values():
            r, e = rd.get(s["file"], {}), eb.get(s["file"], {})
            for k in ("share", "steering", "erank"):
                pan[k].append((r.get("gain"), r.get(k)))
            pan["E_b"].append((r.get("gain"), e.get("E_b_K")))
            sizes.append(e["size_K"])
    return pan, sizes


def difficulty_readings(S: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive, no bar: does a reading move with how much the task was learned, in each panel."""
    sets = ("old", "realistic")
    old = S["difficulty"]
    trig = [s for s, d in old.items() if d["task"] not in ("lowercase", "case_real", "many_one", "many_many") and "_e1_" not in s]
    pan, sizes = realistic_pairs()
    out: dict[str, Any] = {
        "medians": {n: {k: median([r[k] for r in rows if r["set"] == n]) for k, _ in AXES} for n in sets},
        "realistic_token_targets": {"n": sum(r["set"] == "realistic" and r["task"] not in LABEL_TASKS for r in rows)}
        | {k: median([r[k] for r in rows if r["set"] == "realistic" and r["task"] not in LABEL_TASKS]) for k in ("prior", "gain")},
        "n_states": {n: sum(r["set"] == n for r in rows) for n in sets},
        "under_gain_floor": {n: sum(1 for r in rows if r["set"] == n and finite(r["gain"]) and r["gain"] < 1) for n in sets},
        "old_rho_with_gain": {k: rho_with_gain([(old[s]["gain"], S["readings"].get(s, {}).get(k)) for s in old]) for k, _, _ in READS},
        "old_trigger_needed_set": {st: {"states": len(sz), "median": median(sz), "max": max(sz), "empty": sum(x == 0 for x in sz)}
                                   for st in ("eb_neurons", "eb_vpd") for sz in [needed_sizes(trig, st)]},
        "realistic_rho_with_gain": {k: rho_with_gain(v) for k, v in pan.items()},
        "realistic_needed_set": {"states": len(sizes), "median": median(sizes), "min": min(sizes)},
    }  # fmt: skip
    (RESULTS / "difficulty_readings.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    S = json.loads((RESULTS / "summary.json").read_text())
    readings_figure(S)
    rows = old_rows() + panel_rows()
    FIG.mkdir(exist_ok=True)
    with (FIG / "difficulty.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "task", "state", *(k for k, _ in AXES)])
        w.writeheader()
        w.writerows(rows)
    fig, axs = plt.subplots(1, len(AXES), figsize=(11.5, 4.2))
    for ax, (k, label) in zip(axs, AXES, strict=True):
        for x, name, colour in ((0, "old", "tab:blue"), (1, "realistic", "tab:orange")):
            ys = [r[k] for r in rows if r["set"] == name and isinstance(r[k], int | float) and math.isfinite(r[k])]
            xs = [x + 0.18 * ((i * 7919) % 11 / 10 - 0.5) for i in range(len(ys))]
            ax.scatter(xs, ys, s=14, alpha=0.7, color=colour)
            if ys:
                ax.hlines(sorted(ys)[len(ys) // 2], x - 0.25, x + 0.25, color="k", lw=1.5)
            n_none = sum(1 for r in rows if r["set"] == name) - len(ys)
            if n_none:
                ax.text(x, 1.0, f"{n_none} not logged", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
        ax.set_xticks([0, 1], ["old tasks", "realistic panel"])
        ax.set_title(label, fontsize=10)
    fig.suptitle("How hard each task is for the 67M: one point per state, line at the median", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"difficulty.{ext}", dpi=150)
    for name in ("old", "realistic"):
        for k, _ in AXES:
            ys = sorted(r[k] for r in rows if r["set"] == name and isinstance(r[k], int | float) and math.isfinite(r[k]))
            print(f"{name} {k}: n {len(ys)}, median {ys[len(ys) // 2] if ys else None}, range {(ys[0], ys[-1]) if ys else None}")
    print(f"wrote {FIG / 'difficulty.png'}, .pdf, .csv: {len(rows)} states")
    print(json.dumps(difficulty_readings(S, rows), indent=1))


if __name__ == "__main__":
    main()
