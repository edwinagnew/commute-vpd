"""P1.59 (E285) figure: are the biggest and the most-needed changes of a trained update rescales of the base model's
writes? Per item, at VPD and at neurons, trained classes against the strengthening edits x2, x10 and random updates.

    ~/commute-agents/fleet/cpu-run python3 experiments/why_fail/plot.py
"""

from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).resolve().parent
GROUPS = {  # label -> classes pooled (item rows of results/items.tsv; E281, E283 beside, not plotted)
    "x2, x10": ("x2", "x10"),
    "random": ("built_random", "toy_random", "procedural_random", "realistic_random"),
    "toy trained": ("toy_trained",),
    "procedural": ("procedural",),
    "RL": ("rl",),
    "realistic": ("realistic",),
}
SERIES = (("bigdd_E_b", "neuron", -0.15, "top 1% by write change: share kept, weighted by change"),
          ("top_kept_median", "steering", 0.15, "top 1% by flips: median kept share (the filed reading)"))  # fmt: skip


def rows() -> list[dict[str, str]]:
    head, *body = (HERE / "results" / "items.tsv").read_text().splitlines()
    out = [dict(zip(head.split("\t"), b.split("\t"), strict=True)) for b in body]
    return [r for r in out if r["experiment"] not in ("E281", "E283")]


def group_of(r: dict[str, str]) -> str | None:
    for g, cs in GROUPS.items():
        if r["cls"] in cs or r["item"] in cs:
            return g
    return None


def main() -> None:
    items = rows()
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    data = []
    for ax, feat, name in zip(axes, ("vpd", "neurons"), ("VPD subcomponents", "neurons and heads"), strict=True):
        for key, method, dx, label in SERIES:
            st = FG.METHODS[method]
            for gi, g in enumerate(GROUPS):
                v = [float(r[key]) for r in items if r["feat"] == feat and group_of(r) == g and r[key] not in ("", "None")]
                ax.scatter([gi + dx] * len(v), [max(x, 1e-5) for x in v], color=st.color, marker=st.marker, s=14, alpha=0.7, linewidths=0,
                           label=label if gi == 0 else None)  # fmt: skip
                data += [{"featurization": feat, "group": g, "series": key, "value": x} for x in v]
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 2)
        ax.set_xticks(range(len(GROUPS)), list(GROUPS), rotation=30, fontsize=7)
        ax.set_title(name, fontsize=8)
        ax.set_ylabel("kept share: 1 = pure rescale of the base write")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Are a trained update's largest and most-needed changes rescales of the base model's writes?", fontsize=9)
    FG.save(fig, HERE / "figures" / "kept_by_class.png", n="one point per item: x2, x10 2 per featurization; random 35 VPD, 34 neurons; toy trained 114 VPD, 29 neurons; "
            "procedural 26 (one more has an empty needed set); RL 21 (both VPD only); realistic 36 (neurons only, never read at VPD).", data=data)  # fmt: skip


if __name__ == "__main__":
    main()
