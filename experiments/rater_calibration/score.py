"""E315 and E316 (P2.86): the two blind raters' tiers on Fig 3's 16 known-answer tasks, beside the construction label.

Run 1 (E315) read results/raters/rater_{opus,gpt}.md; run 2 (E316, descriptions with no construction wording, real
example pairs, no step counts) reads results/raters/run2/rater_{opus,gpt}.md. All answers verbatim, filed before this
script. Writes results/tiers.json and figures/tiers.png. Consensus is E272/E276's rule: mean tier code (reuse 3, middle 2, new 1); a mean of 2.5-3
reads reuse, 2 middle, 1-1.5 new (the reuse and new sides of E276's common-support rule).
Expected: if the raters recover construction, every task built to reuse reads 2.5-3 and every task built new 1-1.5.
"""

import json
import re
from pathlib import Path

from commute import figures

HERE = Path(__file__).parent
RUNS = {"run1": HERE / "results/raters", "run2": HERE / "results/raters/run2"}
RATERS = ("opus", "gpt")
CODE = {"expected reuse": 3, "middle": 2, "new": 1}
# one-liner number -> (figure label, saved run, construction side); prompt.md header and make_known_scorecard.TASKS
TASKS = {
    1: ("Random map", "e46/rand32", "new"),
    2: ("Lowercase text", "e53/lowercase", "reuse"),
    3: ("Top guess", "c3/rank1", "reuse"),
    4: ("Runner-up guess", "e54/elicit", "reuse"),
    5: ("Copy key", "e46/ident32", "reuse"),
    6: ("Lookup", "e62/lookup", "new"),
    7: ("Copy at a new marker", "e62/copy", "reuse"),
    8: ("Cipher (late step)", "e54/cipher_own", "new"),
    9: ("Copy key (24 keys)", "c3/copy", "reuse"),
    10: ("Prior-matched cipher", "e58/prior_cipher", "new"),
    11: ("Cipher (24 keys)", "c3/cipher", "new"),
    12: ("Ciphered text", "e53/cipher", "new"),
    13: ("Cipher (96 keys)", "e58/cipher", "new"),
    14: ("Cipher (early step)", "e54/cipher", "new"),
    15: ("Copy key (96 keys)", "e58/copy", "reuse"),
    16: ("Top guess (96 keys)", "e58/rank1", "reuse"),
}


def tiers(path: Path) -> dict[int, str]:
    """Task number -> tier, from the table rows ('| <n>. name | rank | tier | ...')."""
    out = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = re.match(r"^(\d+)\.", cells[0]) if len(cells) >= 3 else None
        if m:
            out[int(m.group(1))] = cells[2]
    assert sorted(out) == sorted(TASKS), f"{path.name}: rows {sorted(out)}"
    assert set(out.values()) <= set(CODE), f"{path.name}: tiers {set(out.values())}"
    return out


def consensus_tier(mean: float) -> str:
    return "reuse" if mean >= 2.5 else "new" if mean <= 1.5 else "middle"


def run_row(folder: Path, n: int) -> dict:
    """One run's tiers for task n: each rater's tier and code, and the consensus."""
    read = {r: tiers(folder / f"rater_{r}.md")[n] for r in RATERS}
    codes = {r: CODE[t] for r, t in read.items()}
    mean = sum(codes.values()) / len(codes)
    return {**{f"{r}_tier": t for r, t in read.items()}, **{f"{r}_code": c for r, c in codes.items()},
            "consensus_code": mean, "consensus_tier": consensus_tier(mean)}  # fmt: skip


def main() -> None:
    rows = [{"number": n, "task": label, "run": run, "construction": side,
             **{k: run_row(folder, n) for k, folder in RUNS.items()}}
            for n, (label, run, side) in TASKS.items()]  # fmt: skip
    rows.sort(key=lambda x: (x["construction"] != "reuse", -x["run1"]["consensus_code"], x["number"]))
    (HERE / "results/tiers.json").write_text(json.dumps({"experiment": "E315, E316", "cell": "P2.86", "rule": __doc__,
                                                         "tasks": rows}, indent=1) + "\n")  # fmt: skip
    fig, axs = figures.figure(size=(7.5, 5.0), ncols=2)
    for ax, (k, title) in zip(axs, (("run1", "Run 1: descriptions name the construction"),
                                     ("run2", "Run 2: data only, with example pairs"))):  # fmt: skip
        for side, method in (("reuse", "built_reuse"), ("new", "built_new")):
            groups = {x["task"]: [x[k]["opus_code"], x[k]["gpt_code"]] if x["construction"] == side else [] for x in rows}
            figures.dots(ax, groups, method=method, xlabel="rater tier")
        ax.set_xticks([1, 2, 3], ["new", "middle", "reuse"])
        ax.set_xlim(0.7, 3.3)
        ax.set_title(title, fontsize=8)
    axs[1].set_yticklabels([])
    for method in ("built_reuse", "built_new"):
        s = figures.METHODS[method]
        axs[1].scatter([], [], color=s.color, marker=s.marker, label=s.label)
    axs[1].legend(fontsize=7, loc="lower right")
    flat = [{"task": x["task"], "construction": x["construction"], "run": k, **x[k]} for x in rows for k in RUNS]
    figures.save(fig, HERE / "figures/tiers.png", n="16 tasks, 2 raters each, 2 runs", data=flat)


if __name__ == "__main__":
    main()
