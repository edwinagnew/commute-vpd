"""Figures for the SGD-vs-Adam analysis (reports/2026-10-22_analysis_sgd-vs-adam.md). JSON only.

(1) figures/size_damage.png: every state scored on the Pile harness in the SGD review (sgd_ladder, sgd_ladder_long,
    sgd_kv, sgd_finetune, sgd_faith), its adversarial ratio against its size of change (left) and its held-out KL
    elsewhere (right), with the same-job W0 band. Prints how well one threshold on each separates passes from fails.
(2) figures/steps.png: on each ladder task, the adversarial ratio of each arm at steps 10, 100 and 1000.

Every reading here is a 20-step adversary's: see reports/2026-10-21_analysis_cancellation-complement.md, Amendment 2.

Usage (repo root): uv run python experiments/sgd_ladder_long/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commute.faith_score import LINE, verdict

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FIGURES = HERE / "figures"
TASKS = ("emoticon", "r0_t352_v310", "r1_t534_v369", "r2_t275_v2667")
COLOUR = {"SGD": "C0", "Adam": "C1"}
MARKER = {"component training": "o", "do-nothing": "s", "least-squares split": "D", "adjoint split": "^"}


def load(path: str) -> dict[str, Any]:
    return json.loads((EXP / path).read_text())


def state(src: str, task: str, opt: str, kind: str, step: int, size: float, kl: float | None, pgd: float,
          learned: bool, label: str) -> dict[str, Any]:
    return {"src": src, "task": task, "opt": opt, "kind": kind, "step": step, "size": size, "kl": kl, "pgd": pgd,
            "learned": learned, "label": label}


def arm_opt(arm: str) -> tuple[str, str]:
    """(optimiser, kind) of an arm name; the do-nothing is the Adam 3e-5 fine-tune left in Delta."""
    if arm == "do_nothing":
        return "Adam", "do-nothing"
    return ("SGD" if arm.startswith("sgd") else "Adam"), "component training"


def from_ladder() -> list[dict[str, Any]]:
    """3a: every arm at step 10 (Adam 1e-5 is E11's state, re-read). Learned per 3a's rule."""
    out = []
    for task, rec in load("sgd_ladder/results/sgd_ladder.json")["tasks"].items():
        v = rec["score"]["verdicts"]
        for arm, a in rec["arms"].items():
            opt, kind = arm_opt(arm)
            learned = v[arm]["learned"] if arm in v else a["heldout"]["p_o"] >= 0.9
            out.append(state("sgd_ladder", task, opt, kind, 10, a["rel_norm_max"], a["heldout"]["kl_other"],
                             a["pile"]["pgd_kl"], learned, arm))
    return out


def from_ladder_long() -> list[dict[str, Any]]:
    """3b: steps 100 and 1000 (step 10 is 3a's state, read there). Learned: held-out P >= 0.9."""
    out = []
    for task in TASKS:
        d = load(f"sgd_ladder_long/results/sgd_ladder_long_{task}.json")
        for arm, steps in d["states"].items():
            opt, kind = arm_opt(arm)
            for step, s in steps.items():
                if step == "10":
                    continue
                h = s["heldout"]
                out.append(state("sgd_ladder_long", task, opt, kind, int(step), s["rel_norm_max"], h["kl_other"],
                                 s["pile"]["pgd_kl"], h["p_o"] >= 0.9, arm))
    return out


def from_kv() -> list[dict[str, Any]]:
    """Key-to-value: each arm at its first checkpoint with P >= 0.5. Learned per sgd_kv's rule."""
    out = []
    for task, rec in load("sgd_kv/results/sgd_kv.json")["tasks"].items():
        v = rec["score"]["verdicts"]
        for arm, a in rec["arms"].items():
            opt, kind = arm_opt(arm)
            out.append(state("sgd_kv", task, opt, kind, a["step"], a["rel_norm_max"], a["heldout"]["kl_other"],
                             a["pile"]["pgd_kl"], v[arm]["learned"], arm))
    return out


def from_finetune() -> list[dict[str, Any]]:
    """Emoticon fine-tunes at step 10: SGD 0.01, Adam 3e-6 (size-matched) and Adam 3e-5, each left in Delta, split by
    least squares and by the scaled adjoint split. The adjoint state's KL elsewhere was not recorded."""
    sp, am = load("sgd_finetune/results/split.json"), load("sgd_finetune/results/adam_matched.json")
    runs = [("SGD", "sgd 0.01", sp["states"]["sgd"], sp["score"]["table"], "sgd_", sp["finetune_eval"]["sgd"]),
            ("Adam", "adam 3e-5", sp["states"]["adam"], sp["score"]["table"], "adam_", sp["finetune_eval"]["adam"]),
            ("Adam", "adam 3e-6 matched", am["states"], am["score"]["table"], "", am["match"])]
    kinds = {"nothing": "do-nothing", "least_squares": "least-squares split", "adjoint": "adjoint split"}
    out = []
    for opt, name, states, table, prefix, ev in runs:
        for op, kind in kinds.items():
            if name == "adam 3e-5" and op == "nothing":
                continue  # the same state as sgd_ladder's emoticon do-nothing (2.18 in both jobs)
            t = table[prefix + op]
            kl = None if op == "adjoint" else ev["kl_other"]
            out.append(state("sgd_finetune", "emoticon", opt, kind, 10, states[op]["rel_norm_max"], kl, t["pgd"],
                             t["p_o"] >= 0.9, f"{name} {op}"))
    return out


def from_faith() -> list[dict[str, Any]]:
    """Emoticon component training at step 30: SGD 0.01 (sgd_grid's state) and Adam 1e-5."""
    d = load("sgd_faith/results/sgd_faith.json")
    sgd_kl = load("sgd_grid/results/sgd_grid.json")["runs"]["0.01"]["curve"][-1]["heldout_kl_other"]
    return [state("sgd_faith", "emoticon", "SGD", "component training", 30, d["sgd"]["measure"]["rel_norm_max"], sgd_kl,
                  d["sgd"]["measure"]["pile"]["pgd_kl"], True, "sgd_0.01"),
            state("sgd_faith", "emoticon", "Adam", "component training", 30, d["adam"]["measure"]["rel_norm_max"],
                  d["adam"]["curve_last"]["heldout_kl_other"], d["adam"]["measure"]["pile"]["pgd_kl"], True,
                  "adam_1e-05")]


def floor() -> float:
    """Every job in the review ran on an L40 with the same W0 floor; assert it rather than assume it."""
    fs = {round(load(f"sgd_ladder_long/results/sgd_ladder_long_{t}.json")["W0"]["pgd_floor"], 6) for t in TASKS}
    fs |= {round(f, 6) for f in load("sgd_ladder/results/sgd_ladder.json")["pgd_floor"].values()}
    fs |= {round(f, 6) for f in load("sgd_kv/results/sgd_kv.json")["pgd_floor"].values()}
    fs |= {round(load(p)["W0"]["pgd_floor"], 6) for p in ("sgd_finetune/results/split.json",
                                                          "sgd_finetune/results/adam_matched.json",
                                                          "sgd_faith/results/sgd_faith.json")}
    assert len(fs) == 1, fs
    return fs.pop()


def best_threshold(xs: list[float], fails: list[bool]) -> tuple[float, int]:
    """The cut c maximising correct calls of 'fail iff x > c' (or 'fail iff x < c'); returns (c, errors)."""
    best = (float("nan"), len(xs))
    for c in sorted(xs):
        for sign in (1, -1):
            err = sum((sign * (x - c) > 0) != f for x, f in zip(xs, fails))
            best = min(best, (c, err), key=lambda b: b[1])
    return best


def separation(states: list[dict[str, Any]], f: float) -> None:
    """For learned states with a decisive verdict: how many does the best single cut on size of change, on damage
    elsewhere, or on the step misclassify, over all states and within each optimiser's component training."""
    dec = [s for s in states if s["learned"] and verdict(s["pgd"], f) != "inconclusive"]
    groups = {"all states": dec}
    for opt in COLOUR:
        groups[f"{opt} component training"] = [s for s in dec if s["opt"] == opt and s["kind"] == "component training"]
        groups[f"{opt}, all kinds"] = [s for s in dec if s["opt"] == opt]
    print("\nbest single cut, learned states with a decisive verdict (errors / states, fails):")
    for name, group in groups.items():
        row = []
        for key in ("size", "kl", "step"):
            sub = [s for s in group if s[key] is not None]
            fails = [verdict(s["pgd"], f) == "fail" for s in sub]
            c, err = best_threshold([s[key] for s in sub], fails)
            row.append(f"{key} {err}/{len(sub)} at {c:.4g}")
        print(f"  {name:24s} fails {sum(verdict(s['pgd'], f) == 'fail' for s in group):2d}: " + "; ".join(row))
    for opt in COLOUR:
        for key in ("size", "kl"):
            for v in ("pass", "fail"):
                xs = sorted(s[key] for s in dec if s["opt"] == opt and s["kind"] == "component training"
                            and verdict(s["pgd"], f) == v)
                print(f"  {opt} component training {v} {key}: n={len(xs)} range {xs[0]:.4f} .. {xs[-1]:.4f}")
    print("  fails:")
    for s in sorted(dec, key=lambda s: -s["pgd"]):
        if verdict(s["pgd"], f) == "fail":
            print(f"    {s['src']:16s} {s['task']:14s} {s['label']:28s} step {s['step']:4d} pgd {s['pgd']:6.2f} "
                  f"size {s['size']:.4f} kl {s['kl'] if s['kl'] is None else round(s['kl'], 3)}")


def fig_size_damage(states: list[dict[str, Any]], f: float) -> None:
    """Rows: the optimiser that made the update. Columns: size of change, damage elsewhere. Shared y."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True, sharex="col")
    cols = (("size", "size of change: largest ‖δW‖/‖W₀‖ over the 24 matrices"),
            ("kl", "damage elsewhere: held-out KL off the trigger, nats"))
    for r, opt in enumerate(COLOUR):
        for c, (key, xl) in enumerate(cols):
            ax = axes[r, c]
            ax.axhspan(LINE / (1 + f), LINE * (1 + f), color="0.88", lw=0, label="inconclusive on this card")
            ax.axhline(1, color="k", ls=":", lw=0.8, label="original model (W₀) = 1")
            for s in states:
                if s["opt"] != opt or s[key] is None:
                    continue
                col = COLOUR[opt]
                ax.scatter(s[key], s["pgd"], marker=MARKER[s["kind"]], s=36, zorder=3,
                           facecolors=col if s["learned"] else "none", edgecolors=col)
            ax.set(xscale="log", yscale="log", title=f"{opt} updates")
            if r == 1:
                ax.set_xlabel(xl)
        axes[r, 0].set_ylabel("adversarial Pile KL / W₀'s\n(20-step adversary)")
    for kind, m in MARKER.items():
        axes[0, 1].scatter([], [], marker=m, color="0.4", label=kind)
    axes[0, 1].scatter([], [], facecolors="none", edgecolors="0.4", label="did not learn (hollow)")
    axes[0, 1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Does the size of change or the damage elsewhere predict the adversarial verdict? Every state of the "
                 "SGD review; above the grey band is a fail", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "size_damage.png", dpi=130)


def fig_steps(f: float) -> None:
    ladder = load("sgd_ladder/results/sgd_ladder.json")["tasks"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, task in zip(axes, TASKS):
        d = load(f"sgd_ladder_long/results/sgd_ladder_long_{task}.json")
        ax.axhspan(LINE / (1 + f), LINE * (1 + f), color="0.88", lw=0)
        ax.axhline(1, color="k", ls=":", lw=0.8)
        for arm, c, name in ((d["ref_3a"]["sgd"], "C0", "SGD, the task's rate"), ("adam_1e-05", "C1", "Adam 1e-5"),
                             ("adam_3e-06", "C3", "Adam 3e-6"), ("do_nothing", "0.5", "do-nothing (Adam 3e-5 fine-tune)")):
            a10 = ladder[task]["arms"][arm]
            pts = [(10, a10["pile"]["pgd_kl"], a10["heldout"]["p_o"])]
            pts += [(int(k), s["pile"]["pgd_kl"], s["heldout"]["p_o"]) for k, s in d["states"][arm].items() if k != "10"]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=c, label=name)
            for step, pgd, p in pts:
                ax.scatter(step, pgd, color=c, facecolors=c if p >= 0.9 else "none", zorder=3)
        ax.set(xscale="log", yscale="log", xticks=[10, 100, 1000], xticklabels=["10", "100", "1000"],
               xlabel="training step", title=task)
    axes[0].set_ylabel("adversarial Pile KL / W₀'s (20-step adversary)")
    axes[0].legend(fontsize=7, title="hollow: P(target) < 0.9", title_fontsize=7)
    fig.suptitle("When does each update fail the adversarial check? Grey band: inconclusive; dotted: original model",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "steps.png", dpi=130)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    f = floor()
    states = from_ladder() + from_ladder_long() + from_kv() + from_finetune() + from_faith()
    print(f"W0 PGD floor {f:.4f}: pass <= {LINE / (1 + f):.3f}, fail >= {LINE * (1 + f):.3f}; {len(states)} states")
    separation(states, f)
    fig_size_damage(states, f)
    fig_steps(f)
    print(f"wrote {FIGURES / 'size_damage.png'}, {FIGURES / 'steps.png'}")


if __name__ == "__main__":
    main()
