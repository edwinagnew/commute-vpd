"""E152 step 1's dry read on CPU (no model, no Σ): the least-squares split of each planted change at the job's own sets,
one down_proj matrix at a time, with Σ = I. The job weights the piece energies by Σ; these unweighted numbers are the
source of the filed expectations in results/predictions_step1.json. One line per matrix to results/step1_preview.jsonl.

    ~/commute-agents/fleet/cpu-run -c 8 --mem 16G -t 60 python experiments/qwen_instrument_check/preview_step1.py FIRST LAST

Per matrix and kind: P = the split's energy share on the set, on = its share on the subcomponents with active >= 0.01
(the gated-on rule on the 512-row statistics), C = the share a Gaussian change of the same norm puts on the set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from commute import split
from commute.jobs import load_experiment
from commute.qwen_d3 import BASIS_TOL, GSTATS_OF
from commute.qwen_maps import alive_from_stats
from commute.qwen_sft import load
from commute.qwen_vpd import SITES, parse_site

PL = load_experiment("qwen_instrument_check", "planted")
DECOMP = Path.home() / "commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch"
OUT = Path(__file__).resolve().parent / "results" / "step1_preview.jsonl"


def stack(index: dict, key: str, layer: int) -> Tensor:
    with safe_open(str(DECOMP / index[key]["file"]), framework="pt") as f:
        return f.get_slice(key)[layer : layer + 1][0].double()


def shares(M: Tensor, U_A: Tensor, V_A: Tensor, b: split.Basis, masks: dict[str, Tensor]) -> dict[str, float]:
    X = split.least_squares_x(M, b)
    pe = split.piece_energies(X @ V_A, X.T @ U_A, U_A, V_A, torch.eye(V_A.shape[0], dtype=torch.float64))
    return {k: float(pe[m].sum() / pe.sum()) for k, m in masks.items()}


def main() -> None:
    first, last = int(sys.argv[1]), int(sys.argv[2])
    torch.set_num_threads(8)
    st = [load(GSTATS_OF["p-9707d6ff"] / f"W0_{b}.safetensors") for b in ("maths", "fineweb")]
    alive = alive_from_stats(*st)
    sets = PL.planted_sets(st[0], alive, list(SITES))
    index = json.loads((DECOMP / "index.json").read_text())["tensors"]
    for s in list(sets["often"])[first : last + 1]:
        layer, kind = parse_site(s)
        al = alive[s]
        U_A = stack(index, f"components.stacks.{kind}.1", layer)[al].T.contiguous()
        V_A = stack(index, f"components.stacks.{kind}.0", layer)[:, al].contiguous()
        b = split.basis_from(split.gram_basis(U_A), split.gram_basis(V_A), BASIS_TOL)
        on = st[0][f"{s}.active"].double()[al] >= 0.01
        G = torch.randn(U_A.shape[0], V_A.shape[0], generator=torch.Generator().manual_seed(layer), dtype=torch.float64)
        masks = {k: torch.zeros(len(al), dtype=torch.bool).index_fill_(0, sets[k][s], True) for k in sets}
        chance = shares(G, U_A, V_A, b, masks)
        rec: dict = {"site": s, "n_alive": len(al)}
        for k, m in masks.items():
            P = sets[k][s]
            rec[k] = {"n": len(P), **shares(U_A[:, P] @ V_A[:, P].T, U_A, V_A, b, {"P": m, "on": on}), "C": chance[k]}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
