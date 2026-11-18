"""Figure 2 of section 4: each reuse wording against four tests, one cell per pair, with the number it rests on.

Run: ~/commute-agents/fleet/cpu-run python experiments/held_model_retention/plot_wordings.py
Sources per cell: reports/analyses/2026-11-18_analysis_section4-wordings.md, table "The matrix in full".
Nothing here is computed: every cell is copied from a filed report, so the figure only draws the table.
"""

from matplotlib.colors import ListedColormap

from commute import figures

STATUS = {"pass": 0, "fail": 1, "no call": 2, "not run": 3}
COLOURS = ["#b8e3d2", "#f4c3a8", "#f1e3b5", "#ffffff"]

COLUMNS = [
    "reads hand-made\nedits correctly",
    "ranks matched pairs:\nreuse side above twin",
    "absolute labels\nfollow construction",
    "ever reads built:\na new-by-construction\nitem read as new",
]

# (wording, [(status, the number it rests on) per column])
ROWS = [
    ("Possession, Jain-style\n(first wording, E116)", [
        ("not run", "no hand-made edit read"),
        ("no call", "0 of 26 states readable"),
        ("no call", "possession fails: best\nreadout 0.62, bar 0.90"),
        ("no call", "ciphers unreadable:\ncontrol beats readout"),
    ]),
    ("Reuse at a variable\n(second wording, P1.1)", [
        ("fail", "exact re-weighting edit\nG 0.63, cut-off 0.8"),
        ("pass", "5 of 5 pairs, every seed;\nnarrowest margin 0.17"),
        ("fail", "1 of 5 reuse sides\nre-weighted (case, 0.83)"),
        ("no call", "no built label: XOR\nH -0.01, no map found"),
    ]),
    ("Soft intervention\n(third wording, P1.18-19)", [
        ("pass", "every built edit its grade;\ncarry within 2e-5 of 1"),
        ("no call", "no fine-tune graded"),
        ("no call", "20 of 20: no carrying\nvariable, at most 0.33"),
        ("pass", "floor and XOR read new\ncomputation (hand-made only)"),
    ]),
    ("Closure ranking and curves\n(fourth wording, P1.20-23)", [
        ("fail", "7 of 9 hand-direction edits\nabove filed rung"),
        ("fail", "fresh seeds: 7 of 10;\nrepeat never separates"),
        ("fail", "reuse sides at rung 3;\ntwin 0.895 under bar 0.9"),
        ("pass", "random bit, floor, XOR\nat rung 4 (bit 0.48)"),
    ]),
    ("Causal-rule retention\n(P1.24-25, E195-196)", [
        ("pass", "rebuilt circuit 0.97\nagainst turned-up 1"),
        ("pass", "8 of 8 pair-seeds;\ngaps 0.00004 to 0.0028"),
        ("no call", "no cut-offs; every\nfine-tune 0.996 or more"),
        ("no call", "replaced circuit\nnever built"),
    ]),
    ("Held-model survival S\n(cheap reading, P1.26-28)", [
        ("pass", "code checks within 2e-5\nof arithmetic values"),
        ("fail", "repeat 0.0016, spread 0.0024;\ncopy-lookup -0.0028"),
        ("no call", "no cut-offs,\nno labels"),
        ("no call", "no labels; head grafts\n0.136-0.221, neurons 0.954"),
    ]),
    ("Working definition ρ\n(revised 2026-11-18)", [
        ("pass", "same reading as row above\n(E195): 0.97 against 1"),
        ("pass", "same reading as row above\n(E196): 8 of 8"),
        ("no call", "no cut-offs\nor labels set"),
        ("not run", "grafts read by S only,\nnot by ρ"),
    ]),
]


def main() -> None:
    fig, ax = figures.figure(size=(7.5, 6.4))
    codes = [[STATUS[s] for s, _ in cells] for _, cells in ROWS]
    ax.imshow(codes, cmap=ListedColormap(COLOURS), vmin=-0.5, vmax=3.5, aspect="auto")
    for i, (_, cells) in enumerate(ROWS):
        for j, (status, number) in enumerate(cells):
            ax.text(j, i - 0.28, status, ha="center", va="center", fontsize=7.5, weight="bold", color="#222222")
            ax.text(j, i + 0.12, number, ha="center", va="center", fontsize=6.3, color="#222222")
    ax.set_xticks(range(len(COLUMNS)), COLUMNS, fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ROWS)), [name for name, _ in ROWS], fontsize=7)
    ax.set_xticks([x + 0.5 for x in range(len(COLUMNS) - 1)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(ROWS) - 1)], minor=True)
    ax.grid(which="minor", color="#999999", linewidth=0.6)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("No wording of reuse has passed all four tests on the 67M model",
                 fontsize=8.5, pad=8)
    records = [{"wording": name.replace("\n", " "), "test": COLUMNS[j].replace("\n", " "), "status": s,
                "number": n.replace("\n", " ")} for name, cells in ROWS for j, (s, n) in enumerate(cells)]
    figures.save(fig, "experiments/held_model_retention/figures/wordings_matrix.png",
                 n="Each cell copied from a filed report on the 67M model; 'not run' means the test was never made on "
                   "that wording. The last two rows share the E195 and E196 readings.", data=records)


if __name__ == "__main__":
    main()
