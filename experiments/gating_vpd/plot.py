"""E241 figure: S_c (S with VPD's floor taken off) per item group, with P1c's band below the lowest built reuse edit."""

import json
from pathlib import Path

from commute import figures

HERE = Path(__file__).parent
RESULTS = HERE / "results"
BUILT_REUSE = ("x2", "x10", "rebuilt")  # moved computes M's function: KL(F||M) 6.6e-12, not read on S_c
BUILT_NEW = ("new_few", "new_thin")


def groups() -> tuple[dict[str, list[str]], set[str]]:
    table = json.loads((RESULTS / "summary.json").read_text())["table"]
    pairs = json.loads((RESULTS / "pairs_index.json").read_text())
    reuse = sorted({i for p in pairs.values() for i in p["reuse"]})
    new = sorted({i for p in pairs.values() for i in p["new"]})
    out = {
        "built reuse edits (x2 = x10)": list(BUILT_REUSE),
        "built new edits": list(BUILT_NEW),
        "random, KL matched": sorted(k for k in table if k.startswith("random_") and "_kl_" in k),
        "random, size matched": sorted(k for k in table if k.startswith("random_") and "_size_" in k),
        "trained pairs, reuse side": reuse,
        "trained pairs, new side": new,
    }
    return out, {k for k in out if k.startswith("random")}


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    table, band = summary["table"], summary["band_S_c"]
    named, random_rows = groups()
    values = {g: [table[i]["S_c"] for i in items] for g, items in named.items()}
    lowest_reuse = min(values["built reuse edits (x2 = x10)"])
    fig, ax = figures.figure(figures.FULL_WIDTH)
    ax.axvspan(lowest_reuse - band, lowest_reuse, color="#0072B2", alpha=0.15, linewidth=0,
               label="band under the lowest built reuse edit:\nnew edits and random updates\nhad to sit left of it")
    figures.dots(ax, {g: ([] if g in random_rows else v) for g, v in values.items()}, method="regating",
                 xlabel="S_c = 1 - (KL(F||H) - f) / KL(F||M): 1 = H matches F up to VPD's floor f, 0 = no closer than M")
    figures.dots(ax, {g: (v if g in random_rows else []) for g, v in values.items()}, method="random",
                 xlabel=ax.get_xlabel())
    figures.reference(ax, 0.0, "as far from F as M")
    figures.reference(ax, 1.0, "H matches F up to f")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Built reuse edits do not read above new edits or random updates", pad=14)
    data = [{"group": g, "item": i, "S_c": table[i]["S_c"], "S": table[i]["S"], "kl_fm": table[i]["kl_fm"],
             "f": table[i]["f"], "scored_on_S": table[i]["scored"]} for g, items in named.items() for i in items]
    n = ("one point per item; built edits and random draws 75 copy or 512 cue windows each, pair states 512 windows; "
         f"band {band:.3f} = largest split-half gap of S_c over the built items; E241, cluster debug QoS")
    figures.save(fig, HERE / "figures" / "s_c_by_group.png", n=n, data=data)


if __name__ == "__main__":
    main()
