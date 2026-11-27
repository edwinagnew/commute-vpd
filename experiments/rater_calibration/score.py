"""E315 (P2.86): the two blind raters' tiers on Fig 3's 16 known-answer tasks, beside the construction label.

Reads results/raters/rater_{opus,gpt}.md (verbatim answers, filed before this script) and writes results/tiers.json
and figures/tiers.png. Consensus is E272/E276's rule: mean tier code (reuse 3, middle 2, new 1); a mean of 2.5-3
reads reuse, 2 middle, 1-1.5 new (the reuse and new sides of E276's common-support rule).
Expected: if the raters recover construction, every task built to reuse reads 2.5-3 and every task built new 1-1.5.
"""

import json
import re
from pathlib import Path

from commute import figures

HERE = Path(__file__).parent
RATERS = {"opus": HERE / "results/raters/rater_opus.md", "gpt": HERE / "results/raters/rater_gpt.md"}
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


def main() -> None:
    read = {r: tiers(p) for r, p in RATERS.items()}
    rows = []
    for n, (label, run, side) in TASKS.items():
        codes = {r: CODE[read[r][n]] for r in RATERS}
        mean = sum(codes.values()) / len(codes)
        rows.append({"number": n, "task": label, "run": run, "construction": side,
                     **{f"{r}_tier": read[r][n] for r in RATERS}, **{f"{r}_code": c for r, c in codes.items()},
                     "consensus_code": mean, "consensus_tier": consensus_tier(mean)})  # fmt: skip
    rows.sort(key=lambda x: (x["construction"] != "reuse", -x["consensus_code"], x["number"]))
    (HERE / "results/tiers.json").write_text(json.dumps({"experiment": "E315", "cell": "P2.86", "rule": __doc__,
                                                         "tasks": rows}, indent=1) + "\n")  # fmt: skip
    fig, ax = figures.figure(size=(5.5, 5.0))
    for side, method, off in (("reuse", "built_reuse", 0.0), ("new", "built_new", 0.0)):
        groups = {x["task"]: [x["opus_code"], x["gpt_code"]] if x["construction"] == side else [] for x in rows}
        figures.dots(ax, groups, method=method, xlabel="rater tier code (3 reuse, 2 middle, 1 new)", offset=off)
    for method in ("built_reuse", "built_new"):
        s = figures.METHODS[method]
        ax.scatter([], [], color=s.color, marker=s.marker, label=s.label)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xticks([1, 2, 3], ["new", "middle", "reuse"])
    ax.set_xlim(0.7, 3.3)
    ax.set_title("Two blind raters' tiers on Fig 3's tasks (one point per rater)", fontsize=8)
    figures.save(fig, HERE / "figures/tiers.png", n="16 tasks, 2 raters each", data=rows)


if __name__ == "__main__":
    main()
