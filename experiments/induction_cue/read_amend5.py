"""E62 Amendment 5's reads beside the verdict, from results/amend5.json and results/large/amend5_windows.pt, plus windows.pt for the keys. Loads no model; decides nothing.
The Review subsection of the plan's Amendment 5 lists these. Writes results/amend5_reads.json.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/induction_cue/read_amend5.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import copying as CP

HERE = Path(__file__).resolve().parent
RES, LARGE = HERE / "results", HERE / "results" / "large"
N_BOOT, SEED = 2000, 0


def resample(key_ids: Tensor, by_key: bool, g: torch.Generator) -> Tensor:
    """One bootstrap draw of window indices: over windows, or over whole keys, each drawn key bringing all its windows."""
    n = len(key_ids)
    if not by_key:
        return torch.randint(n, (n,), generator=g)
    keys = key_ids.unique()
    draw = keys[torch.randint(len(keys), (len(keys),), generator=g)]
    return torch.cat([(key_ids == k).nonzero()[:, 0] for k in draw])


def boot(stat: Any, key_ids: Tensor, by_key: bool) -> dict[str, float]:
    g = torch.Generator().manual_seed(SEED)
    v = torch.tensor([float(stat(resample(key_ids, by_key, g))) for _ in range(N_BOOT)])
    return {"sd": float(v.std()), "q025": float(v.quantile(0.025)), "q975": float(v.quantile(0.975)), "share_above_0": float((v > 0).float().mean())}


def both(stat: Any, key_ids: Tensor) -> dict[str, Any]:
    full = torch.arange(len(key_ids))
    return {"mean": float(stat(full)), "windows": boot(stat, key_ids, False), "keys": boot(stat, key_ids, True)}


def main() -> None:
    out = json.loads((RES / "amend5.json").read_text())
    pred = json.loads((RES / "predictions_amend5.json").read_text())
    rows: dict[str, Tensor] = torch.load(LARGE / "amend5_windows.pt", weights_only=True)
    half = json.loads((RES / "predictions.json").read_text())["settings"]["half"]
    v = torch.load(LARGE / "windows.pt", weights_only=False)
    keys = v["read"].tokens[:, half - 1]
    assert len(keys) == len(rows["mask_on"])
    H = out["stage_c_sweep"]["heads"]
    names = sorted(H)
    und = {n: n.replace(",", "_") for n in names}
    T = {n: rows["target_on"] - rows[f"target_off_{und[n]}"] for n in names}
    M = {n: rows["mask_on"] - rows[f"mask_off_{und[n]}"] for n in names}
    Mb = {n: rows["mask_broken"] - rows[f"mask_broken_off_{und[n]}"] for n in names}
    Tb = {n: rows["target_broken"] - rows[f"target_broken_off_{und[n]}"] for n in names}
    rec: dict[str, Any] = {"n_windows": len(keys), "n_keys": len(keys.unique()), "windows_per_key": [int((keys == k).sum()) for k in keys.unique()]}

    # The verdict at other fall floors, deciding nothing.
    def parsed(name: str) -> CP.Head:
        return (int(name[0]), int(name[2]))
    for which, fields, lean in (("bf16", ("M", "Mb", "excess_sd"), out["stage_a_reread"]["mask_lean"]), ("fp32", ("M32", "Mb32", "excess32_sd"), out["stage_b_precision"]["fp32_lean"]["mean"])):
        table = {parsed(n): {"T": H[n]["T"], "M": H[n][fields[0]], "Mb": H[n][fields[1]], "excess_sd": H[n][fields[2]]} for n in names}
        rec[f"floors_{which}"] = {str(f): CP.sweep_reading(table, (2, 4), 24, lean, {**pred["thresholds"], "fall_floor": f}) for f in (0.25, 0.5, 1.0)}
    # The band the filed rule allows, and how far the fractions sit apart in their own noise.
    lab = H["2,4"]["M"] / H["2,4"]["T"]
    rec["band"] = [lab - pred["thresholds"]["fraction_gap"], lab + pred["thresholds"]["fraction_gap"]]
    frac = lambda n: (lambda i: M[n][i].mean() / T[n][i].mean())
    rec["fractions"] = {n: both(frac(n), keys) for n in ("2,4", "1,1", "0,4", "1,2", "3,4", "0,2", "3,5", "2,3")}
    rec["fraction_minus_labelled"] = {n: both(lambda i, n=n: frac(n)(i) - frac("2,4")(i), keys) for n in ("1,1", "0,4", "1,2", "3,4", "0,2", "3,5", "2,3")}

    # Who falls furthest in the target model.
    rec["T_rank"] = sorted(names, key=lambda n: -float(T[n].mean()))[:6]
    rec["T11_minus_T24"] = both(lambda i: (T["1,1"] - T["2,4"])[i].mean(), keys)
    rec["corr_falls_11_24"] = {"target_windows": float(torch.corrcoef(torch.stack([T["1,1"], T["2,4"]]))[0, 1]), "masked_windows": float(torch.corrcoef(torch.stack([M["1,1"], M["2,4"]]))[0, 1])}
    k = keys.unique()
    per_key = lambda d: torch.stack([d[keys == x].mean() for x in k])
    rec["corr_falls_11_24"]["target_keys"] = float(torch.corrcoef(torch.stack([per_key(T["1,1"]), per_key(T["2,4"])]))[0, 1])
    rec["corr_falls_11_24"]["masked_keys"] = float(torch.corrcoef(torch.stack([per_key(M["1,1"]), per_key(M["2,4"])]))[0, 1])
    rec["windows_T11_over_T24"] = float((T["1,1"] > T["2,4"]).float().mean())
    rec["masked_ratio_11_over_24"] = both(lambda i: M["1,1"][i].mean() / M["2,4"][i].mean(), keys)
    rec["target_ratio_11_over_24"] = both(lambda i: T["1,1"][i].mean() / T["2,4"][i].mean(), keys)

    # How much of each lean the 24 falls add up to.
    lean_m, lean_t = rows["mask_on"] - rows["mask_broken"], rows["target_on"] - rows["target_broken"]
    sum_m, sum_t = torch.stack(list(M.values())).sum(0), torch.stack(list(T.values())).sum(0)
    sum_part_m = torch.stack([M[n] - Mb[n] for n in names]).sum(0)
    sum_part_t = torch.stack([T[n] - Tb[n] for n in names]).sum(0)
    rec["sums"] = {"sum_M": float(sum_m.mean()), "sum_T": float(sum_t.mean()), "lean_M": float(lean_m.mean()), "lean_T": float(lean_t.mean()), "sum_pair_part_M": float(sum_part_m.mean()), "sum_pair_part_T": float(sum_part_t.mean()),
                   "sum_positive_M": sum(max(0.0, float(M[n].mean())) for n in names), "sum_positive_excess": sum(max(0.0, float((M[n] - T[n]).mean())) for n in names)}
    rec["sum_M_over_lean"] = both(lambda i: sum_m[i].mean() / lean_m[i].mean(), keys)
    rec["sum_T_over_lean"] = both(lambda i: sum_t[i].mean() / lean_t[i].mean(), keys)
    size = lambda d, b: sum(abs(float(d[n].mean())) for n in names if n[0] == str(b))
    rec["by_block"] = {str(b): {"sum_T": sum(float(T[n].mean()) for n in names if n[0] == str(b)), "sum_M": sum(float(M[n].mean()) for n in names if n[0] == str(b)),
                                "size_T": size(T, b), "size_M": size(M, b), "size_Tb": size(Tb, b), "size_Mb": size(Mb, b), "size_M_over_T": size(M, b) / size(T, b), "size_Mb_over_Tb": size(Mb, b) / size(Tb, b)} for b in range(4)}
    rec["largest_abs"] = {"T": max(abs(float(T[n].mean())) for n in names), "M": max(abs(float(M[n].mean())) for n in names), "Tb": max(abs(float(Tb[n].mean())) for n in names), "Mb": max(abs(float(Mb[n].mean())) for n in names)}

    # Per-window token parts: the swapped forwards.
    tb = rows["swap_broken_gates_on_intact"] - rows["mask_broken"]
    ti = rows["mask_on"] - rows["swap_intact_gates_on_broken"]
    rec["tokens"] = {"share_at_broken_gates": both(lambda i: tb[i].mean() / lean_m[i].mean(), keys), "share_at_intact_gates": both(lambda i: ti[i].mean() / lean_m[i].mean(), keys),
                     "windows_share_over_half_broken_gates": float((tb > 0.5 * lean_m).float().mean()), "windows_share_over_half_intact_gates": float((ti > 0.5 * lean_m).float().mean()), "windows_lean_positive": float((lean_m > 0).float().mean())}
    rec["levels_key_sd"] = {n: both(lambda i, n=n: rows[n][i].mean(), keys)["keys"]["sd"] for n in ("mask_on", "mask_broken", "swap_broken_gates_on_intact", "swap_intact_gates_on_broken", "target_on", "target_broken")}

    # The control, per window and in key noise; (2,4)'s fall in the two main forwards in key noise.
    ones = rows["ones_on"] - rows["ones_off"]
    rec["control"] = {"fall_over_target": both(lambda i: ones[i].mean() / T["2,4"][i].mean(), keys), "corr_with_target_windows": float(torch.corrcoef(torch.stack([ones, T["2,4"]]))[0, 1])}
    rec["falls_key_noise"] = {n: both(lambda i, n=n: M[n][i].mean(), keys)["keys"]["sd"] for n in ("2,4", "2,3", "3,4", "1,1")}
    rec["pair_part_key_noise"] = {f"T_{n}": both(lambda i, n=n: (T[n] - Tb[n])[i].mean(), keys)["keys"]["sd"] for n in ("2,4", "1,1", "3,5")}
    (RES / "amend5_reads.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
