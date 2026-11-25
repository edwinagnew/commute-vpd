"""E298's figure and its moved-rows file (no GPU).

    ~/commute-agents/fleet/cpu-run python experiments/fig2_standard/plot.py OLD_DEFINITION_CSV

OLD_DEFINITION_CSV is the old Fig 2's data, the paper's figures/data/definition.csv. Reads results/summary.json
(score.py). Writes figures/fig2_standard.png (per featurization, every group's E_b over K on one log axis: built reuse,
built new, the two shared built randoms of their own recipes, each built edit's matched random, the head-(2, 4) edits
and their randoms, trained per-task means over 3 seeds and their randoms) and results/moved_rows.json (every row of
the old figure whose value moved by more than 0.001, old and new, plus the largest per-row move).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FEATS = {"VPD": "VPD subcomponents", "top": "top neurons", "SAE": "SAE latents"}
SHARED_RANDOMS = ("random_size_241001", "random_kl_241002")
MOVED = 0.001
# group label, block, side, item filter (None = any), drawn as a random
GROUPS: tuple[tuple[str, str, str, tuple[str, ...] | None, bool], ...] = (
    ("built, reuse", "built", "reuse", None, False),
    ("built, new", "built", "new", None, False),
    ("head (2,4) edits", "head", "edit", None, False),
    ("trained, per task", "trained", "trained", None, False),
    ("built, own random", "built", "random", ("x2", "x10", "rebuilt", "new_few", "new_thin"), True),
    ("head edits, own random", "head", "random", None, True),
    ("trained, own random", "trained", "random", None, True),
    ("2 shared built randoms", "built", "random", SHARED_RANDOMS, True),
)


def group_values(per_task: list[dict[str, Any]], feat: str, block: str, side: str, items: tuple[str, ...] | None) -> list[float]:
    return [r["value"] for r in per_task if r["feat"] == feat and r["block"] == block and r["side"] == side
            and (items is None or r["task"] in items)]  # fmt: skip


def moved_rows(old_csv: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    new = {(r["feat"], r["block"], r["item"], r["side"]): r["value"] for r in rows}
    with old_csv.open() as f:
        old = {(r["feat"], r["block"], r["item"], r["side"]): float(r["value"]) for r in csv.DictReader(f)}
    assert set(old) <= set(new), sorted(set(old) - set(new))
    moves = [{"feat": k[0], "block": k[1], "item": k[2], "side": k[3], "old": v, "new": new[k], "move": new[k] - v}
             for k, v in old.items()]  # fmt: skip
    return {"n_old_rows": len(old), "threshold": MOVED, "max_abs_move": max(abs(m["move"]) for m in moves),
            "moved": [m for m in moves if abs(m["move"]) > MOVED]}  # fmt: skip


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    moved = moved_rows(Path(sys.argv[1]), summary["rows"])
    (RESULTS / "moved_rows.json").write_text(json.dumps(moved, indent=1))
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=3)
    data = []
    for ax, (feat, name) in zip(axes, FEATS.items(), strict=True):
        for i, (label, block, side, items, is_random) in enumerate(GROUPS):
            vals = group_values(summary["per_task"], feat, block, side, items)
            style = FG.METHODS["random" if is_random else "eb"]
            ax.scatter(vals, [i] * len(vals), color=style.color, marker=style.marker, s=22, alpha=0.85, linewidths=0,
                       label=style.label if feat == "VPD" and i in (0, 4) else None)  # fmt: skip
            data += [{"feat": feat, "group": label, "value": v} for v in vals]
        ax.axhline(3.5, color="#bbbbbb", linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlim(5e-5, 2)
        ax.set_yticks(range(len(GROUPS)), [g[0] for g in GROUPS] if feat == "VPD" else [""] * len(GROUPS))
        ax.invert_yaxis()
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("E_b over K (leftovers in)", fontsize=8)
    fig.legend(fontsize=7, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.95), frameon=False)
    fig.suptitle("Built reuse edits read above every new edit and random at all three featurizations", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    FG.save(fig, HERE / "figures" / "fig2_standard.png", data=data,
            n="one point per built edit, head edit or random; trained points are each task's mean over 3 seeds "
              "(5 tasks); the 2 shared built randoms use their own recipes, the others E290's per-tensor Gaussian")  # fmt: skip
    print(json.dumps({k: v for k, v in moved.items() if k != "moved"}), len(moved["moved"]), "rows moved")


if __name__ == "__main__":
    main()
