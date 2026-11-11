"""Card sketch for P2.6, made-up numbers only: one panel per rival's outcome.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/share_stability/sketch.py
"""

from commute.figures import FULL_WIDTH, dots, figure, reference, save

FT10, FT100 = [0.82, 0.77, 0.86, 0.79, 0.84], [0.78, 0.72, 0.83, 0.75, 0.80]
cases = {
    "Rival a: any small update reads high": [0.74, 0.86, 0.69, 0.81, 0.77],
    "Rival b: only this fine-tune reads high": [0.12, 0.31, -0.05, 0.22, 0.18],
}
fig, axes = figure(FULL_WIDTH, ncols=2)
rows = []
for ax, (title, gauss) in zip(axes, cases.items(), strict=True):
    groups = {"fine-tune, step 10": FT10, "fine-tune, step 100": FT100}
    dots(ax, groups, method="regating", xlabel="every-position share, one dot per 64-row batch")
    ax.scatter(gauss, [2] * len(gauss), color="#555555", marker="X", s=22)
    ax.set_yticks(range(3), [*groups, "Gaussian, step-100 norm"])
    reference(ax, 0.7802, "E46 batch 0, step 100")
    ax.set_xlim(-0.3, 1.2)
    ax.set_title(title, fontsize=9)
    rows += [{"panel": title, "group": g, "share": v} for g, vs in {**groups, "gaussian": gauss}.items() for v in vs]
fig.suptitle("SKETCH, made-up numbers: does a random update of the same size read as high as the fine-tune?", fontsize=9)
save(fig, "experiments/share_stability/figures/sketch.png", n="5 batches of 64 held-out code-text rows per state", data=rows)
