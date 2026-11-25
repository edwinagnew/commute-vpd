"""P1.41 (E236) figure, no GPU: d = R(gain-only) - R(full fine-tune) per paired item at each featurization, from
results/summary.json.

    python experiments/gain_only_r/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).resolve().parent
LEVELS = {"vpd": "VPD subcomponents", "head_mlp": "heads and MLP layers", "mlp_attn": "MLP and attention per layer",
          "block": "whole blocks"}  # fmt: skip
OFFSETS = {0: -0.22, 1: 0.0, 2: 0.22}


def main() -> int:
    s = json.loads((HERE / "results/summary.json").read_text())
    calls = {"vpd": s["R1_vpd"], **s["R2_coarse"]}
    records = [{"item": name, "seed": r["seed"], "level": lv, "d": r[lv] - r[f"{lv}_full"]}
               for name, r in s["items"].items() for lv in LEVELS
               if r[lv] is not None and r[f"{lv}_full"] is not None]  # fmt: skip
    noise = max(s["gain_only_seed_sd_vpd"].values())
    rows = {lv: f"{label}\n{calls[lv]['up']} of {calls[lv]['n']} above 0" for lv, label in LEVELS.items()}
    fig, ax = FG.figure(FG.FULL_WIDTH)
    ax.axvspan(-noise, noise, color="#BBBBBB", alpha=0.35, linewidth=0,
               label="largest seed sd of R(gain-only) at VPD over tasks")  # fmt: skip
    for seed, off in OFFSETS.items():
        groups = {rows[lv]: [x["d"] for x in records if x["level"] == lv and x["seed"] == seed] for lv in LEVELS}
        FG.dots(ax, groups, method="spectrum", offset=off,
                xlabel="R(gain-only state) minus R(full fine-tune), same task and seed")  # fmt: skip
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.annotate("equal R", (0, 1.0), xycoords=("data", "axes fraction"), xytext=(2, 2), textcoords="offset points",
                fontsize=7, color="#555555")  # fmt: skip
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title("Strength-only states read below their full fine-tune at VPD, above it at heads", fontsize=9)
    FG.save(fig, HERE / "figures/r_gap_gain_vs_full.png", data=records,
            n="49 paired items at VPD, 57 at each coarse level (one point each; seeds 0, 1, 2 top to bottom within a row); "
              "R at heads and MLP layers is the mean over three search orders")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
