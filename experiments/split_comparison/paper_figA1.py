"""Paper figure A1 (cell P2.36 e): behaviour kept against adversarial faithfulness for each way of carrying a finished
emoticon update onto the 67M decomposition, from saved results only.

    ~/commute-agents/fleet/cpu-run python experiments/split_comparison/paper_figA1.py

Three updates of the emoticon task at step 1000, each read in one H100 harness (256 Pile rows, a 20-step adversary,
the same W0 readings): E16's Adam 3e-5 fine-tune and rank-8 LoRA, E22's plain-SGD 0.01 fine-tune, and E34's teacher
training of those two fine-tunes (its do-nothing readings equal E16's and E22's to every digit). Behaviour kept =
held-out P(target) of the state / that of the do-nothing state, which computes the update's own weights. Faithfulness =
the state's adversarial KL / W0's, adversary seed 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import figure, save

ROOT = Path(__file__).resolve().parents[2]
E16 = Path("experiments/split_comparison/results/split_comparison.json")
E22 = Path("experiments/small_finetune_splits/results/small_finetune_splits.json")
E34 = Path("experiments/teacher_training/results/teacher_training_emoticon.json")
OUT = ROOT / "experiments/split_comparison/figures/paper_figA1_carrying.png"
STEP = 1000
STYLE = {  # label, colour, marker (Okabe-Ito)
    "nothing": ("do-nothing: update left in Δ", "#555555", "s"),
    "least_squares": ("least-squares split", "#0072B2", "o"),
    "adjoint": ("adjoint split", "#E69F00", "D"),
    "rescaling": ("rescaling split", "#CC79A7", "v"),
    "teacher_task": ("teacher training, task text", "#009E73", "^"),
    "teacher_pile": ("teacher training, Pile text", "#D55E00", "P"),
    "w0": ("unchanged decomposition, 3 adversary seeds", "#000000", "x"),
}
PANELS = {"adam": "Adam fine-tune, all 24 matrices", "sgd": "Plain-SGD fine-tune, same task",
          "lora": "Rank-8 LoRA, same task"}


def load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text())


def split_points(d: dict, panel: str, select: str) -> list[dict]:
    """The step-1000 split rows of one update; `select` is the row field and value naming the update."""
    field, value = select.split("=")
    rows = {r["op"]: (i, r) for i, r in enumerate(d["rows"]) if r[field] == value and r["step"] == STEP and r["op"] in STYLE}
    own = rows["nothing"][1]["p_target"]
    return [{"panel": panel, "method": op, "keep": r["p_target"] / own, "ratio": r["pgd"], "p_state": r["p_target"],
             "p_own": own, "source": "", "key": f".rows[{i}]"} for op, (i, r) in rows.items()]


def teacher_points(t: dict, panel: str, teacher: str) -> list[dict]:
    arms, own = t["teachers"][teacher], t["teachers"][teacher]["nothing"]["p_target"]
    return [{"panel": panel, "method": f"teacher_{text}", "keep": s["p_target"] / own, "ratio": s["pgd"],
             "p_state": s["p_target"], "p_own": own, "student_step": int(step), "source": str(E34),
             "key": f'.teachers.{teacher}.{arm}["{step}"]'}
            for arm, text in (("A", "task"), ("B", "pile")) for step, s in arms[arm].items()]


def all_points() -> tuple[list[dict], float]:
    d16, d22, t = load(E16), load(E22), load(E34)
    w0 = d16["W0"]["pile"]
    assert t["W0"]["pile"] == w0 and all(d22["W0"]["pile"][s] == w0[s] for s in d22["W0"]["pile"]), "not one W0"
    pts = []
    for panel, src, d, select, teacher in (("adam", E16, d16, "pair=ft", "adam"), ("sgd", E22, d22, "ft=sgd", "sgd"),
                                           ("lora", E16, d16, "pair=lora24", None)):
        new = [{**p, "source": str(src)} for p in split_points(d, panel, select)]
        if teacher:
            assert t["teachers"][teacher]["nothing"]["pgd"] == next(p["ratio"] for p in new if p["method"] == "nothing")
            new += teacher_points(t, panel, teacher)
        new += [{"panel": panel, "method": "w0", "keep": d16["rows"][0]["p_baseline"] / new[0]["p_own"],
                 "ratio": w0[s]["pgd_kl"] / w0["0"]["pgd_kl"], "p_state": d16["rows"][0]["p_baseline"],
                 "p_own": new[0]["p_own"], "source": str(E16),
                 "key": f'.W0.pile["{s}"].pgd_kl / .W0.pile["0"].pgd_kl'} for s in sorted(w0)]
        pts += new
    return pts, d16["W0"]["pgd_floor"]


def main() -> None:
    pts, f = all_points()
    fig, grid = figure((7.5, 5.6), ncols=2, nrows=2)
    axes, key = grid.flat[:3], grid.flat[3]
    for ax, (panel, title) in zip(axes, PANELS.items()):
        for p in (p for p in pts if p["panel"] == panel):
            _, colour, marker = STYLE[p["method"]]
            ax.scatter(p["keep"], p["ratio"], s=18 if p["method"] == "w0" else 36, marker=marker, color=colour,
                       linewidths=1.2, alpha=0.35 if p.get("student_step") == 100 else 1)
        for y, name in ((1.5 / (1 + f), "pass line"), (1.5 * (1 + f), "fail line")):
            ax.axhline(y, color="#777777", linewidth=0.8, linestyle="--")
            ax.annotate(name, (0.0, y), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points",
                        fontsize=7, color="#555555")
        ax.axvline(1.0, color="#777777", linewidth=0.8, linestyle=":")
        ax.annotate("the update's\nown behaviour", (1.0, 0.7), xycoords=("data", "axes fraction"), xytext=(-3, 0),
                    textcoords="offset points", fontsize=7, color="#555555", ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("behaviour kept")
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(0.85, 2.2)
    for ax in grid[:, 0]:
        ax.set_ylabel("adversarial damage ratio")
    handles = [axes[0].scatter([], [], marker=m, color=c, label=lab) for lab, c, m in STYLE.values()]
    handles.append(axes[0].scatter([], [], marker="^", color="#009E73", alpha=0.35, label="faded: teacher training after 100 of 1,000 steps"))
    key.axis("off")
    key.legend(handles=handles, loc="center", fontsize=7.5, frameon=False)
    fig.suptitle("Teacher training keeps the behaviour under the pass line on both fine-tunes; least squares fails\n"
                 "the large Adam one, and the adjoint and rescaling splits lose the behaviour", fontsize=9)
    save(fig, OUT, n="Emoticon task, one training seed per update, step 1000; adversary on 256 Pile rows, 20 steps, "
         "seed 0; one H100 harness. Teacher training was not run on the LoRA, the rescaling split not on the SGD fine-tune.",
         data=pts)


if __name__ == "__main__":
    main()
