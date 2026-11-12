"""k_half_fill (E172, cell P2.24): E98's carrying-set count k_half on the method table's built-pair states E98 never gave
one, to fill a k_half column on all nine built pairs and E147's two 96-key twins. Descriptive: no claim at a bar, no
verdict on reuse. Filed before the run: results/predictions.json (the run refuses unless it is committed unchanged).

k_half (`commute.neuron_r`, E98's filed P3 reading, settings read from experiments/neuron_r/results/predictions.json):
neurons ranked once by the sum over windows and positions of W0's gradient x (Ah - A0); the first k of E98's grid whose
patch (Ah on the top k, A0 elsewhere) reaches half the r_half numerator; None when the r_half share is not above
carry_floor, its numerator is not positive, or |G| is under the gain floor (share undefined). E98 read it only on its
"scored" states; the matched ciphers (E54 cipher_s0-4, C3 c3_cipher_s0-7) were left out because their mean held-out
P(target) is under p_learn (0.23-0.31 and 0.32-0.45 against 0.5), not for their gain or r_half share.

  stage 0, alone (a miss writes, logs FAIL, exits 1): c3_rank1_s0 and lowercase_s0 read again, against E98's saved
          k_half, grid numerators, top-256 order (carrying.json) and r_half share (cells.json)
  stage 1: the 13 E98 states with no k_half and E147's 24 refit 96-key states (bf16 updates, read in fp32), each written
          as it lands with its checks (identity, telescoping, grid end, its r_half against the saved reading)
  cells:  per pair, reuse and new sides; states E98 read take its carrying.json value, the rest this run's
Output: results/cells.json (.stage0, .states.<state>, .cells["<pair>/k_half"]). Target model only, fp32, TF32 off.

    sbatch --comment=E172 cluster/k_half_fill.sbatch
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from commute import keydep as KD
from commute import neuron_r as NR
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E98_R, E147_R = REPO / "experiments/neuron_r/results", REPO / "experiments/refit_96key/results"
E98 = load_experiment("neuron_r", "run")  # E98's files (with sha256), state table, state loader and read windows
E98_NEW = [f"cipher_s{s}" for s in range(5)] + [f"c3_cipher_s{s}" for s in range(8)]
STAGE0 = ("c3_rank1_s0", "lowercase_s0")
FULL = NR.N_NEURONS
log = KD.log


def load(path: Path) -> Any:
    return json.loads(path.read_text())


class Stopped(Exception):
    """A filed stop: cells.json records why; the job exits 1."""


class Out:
    def __init__(self) -> None:
        self.out: dict[str, Any] = {"stage0": {}, "states": {}, "cells": {}}

    def write(self) -> None:
        tmp = RESULTS / "cells.tmp"  # a kill mid-write must not leave unreadable JSON
        tmp.write_text(json.dumps(jsonable(self.out), indent=1, sort_keys=True))
        tmp.replace(RESULTS / "cells.json")

    def put(self, group: str, key: str, rec: dict[str, Any], failed: list[str]) -> None:
        self.out[group][key] = {**rec, "failed": failed}
        if failed:
            self.out["stopped"] = f"{group}/{key}: {failed}"
        self.write()
        log(f"[{group} {key}] {json.dumps(jsonable({k: v for k, v in rec.items() if k != 'top'}), sort_keys=True)}")
        if failed:
            raise Stopped(self.out["stopped"])


# ----------------------------------------------------------------------------- the reading


def why_none(rec: dict[str, Any], S: dict[str, Any]) -> str | None:
    """Which clause of `neuron_r.k_half` gave None (None when it gave a count)."""
    if not rec["readable"]:
        return f"|G| {abs(rec['G']):.3g} under gain_floor {S['gain_floor']}: r_half share undefined"
    if not rec["r_half"] > S["carry_floor"]:
        return f"r_half share {rec['r_half']:.3g} not above carry_floor {S['carry_floor']}"
    if not rec["num"]["r_half"] > 0:
        return f"r_half numerator {rec['num']['r_half']:.3g} not positive"
    return None


def carrying(net: ST.Net, r: NR.Reads, dW: ST.DW, S: dict[str, Any], p_learn: float | None) -> dict[str, Any]:
    """E98's stage 1 cells and stage 2 carrying set on one state, unchanged (neuron_r/run.py stage1, stage2)."""
    rec = NR.shares(NR.state_cells(net, r, dW, S["batch"]), S["gain_floor"])
    full = rec["num"]["r_half"]
    score = NR.ranking(net, r, dW, S["batch"])
    order = torch.sort(score, descending=True, stable=True).indices
    nums = NR.grid_numerators(net, r, dW, order, S["k_grid"], S["batch"])
    k = NR.k_half(nums, full, rec["r_half"], S["carry_floor"])
    return {"G": rec["G"], "r_half": rec["r_half"], "r_half_num": full, "p_target": rec["p_target"], "readable": rec["readable"],
            "identity": rec["identity"], "telescope": rec["telescope"], "n_windows": rec["n_windows"],
            "e98_scored": NR.scored(rec, p_learn), "numerators": nums, "k_half": k, "why_none": why_none(rec, S) if k is None else None,
            "end_diff": nums[FULL] - full, "top": order[: S["n_top_saved"]].tolist()}  # fmt: skip


def checks(rec: dict[str, Any], saved_r_half: float, S: dict[str, Any], T: dict[str, Any]) -> list[str]:
    """E98's per-state checks (identity, telescoping, grid end) and the r_half share against its saved reading."""
    conds = {"identity": rec["identity"] <= T["identity_tol"], "telescope": abs(rec["telescope"]) <= S["telescope_tol"],
             "grid end": abs(rec["end_diff"]) <= S["grid_end_tol"], "r_half vs saved": abs(rec["r_half"] - saved_r_half) <= T["reproduce_tol"]}  # fmt: skip
    return [c for c, ok in conds.items() if not ok]


# ----------------------------------------------------------------------------- states


class States:
    """Each state's update, read windows, E98 score filter and saved r_half: E98's states through its run.py loaders
    (sha256 against E98's records), E147's from its inventory and results/large (sha256 against its stage0.json)."""

    def __init__(self, net: ST.Net, S: dict[str, Any]) -> None:
        self.net, self.S = net, S
        self.F, self.table = E98.files(), E98.state_table()
        self.e98_reads = E98.load_reads(net, self.F)
        self.e98_cells = load(E98_R / "cells.json")["rows"]
        self.inv, self.e147_0, self.e147_1 = load(E147_R / "inventory.json"), load(E147_R / "stage0.json"), load(E147_R / "stage1.json")
        self.e147_reads: dict[str, NR.Reads] = {}

    def e98(self, name: str) -> tuple[ST.DW, NR.Reads, float | None, float]:
        key, task = self.table[name]
        path, want = self.F[key]
        assert KD.sha256(path) == want, f"{path}: sha256 differs from E98's record"
        return E98.state(self.F, key), self.e98_reads[task], E98.p_learn_of(task, self.S), self.e98_cells[name]["r_half"]

    def e147(self, m: str, s: int) -> tuple[ST.DW, NR.Reads, float | None, float]:
        name, path = f"e58/{m}_s{s}", E147_R / "large" / f"dW_{m}_s{s}.pt"
        assert KD.sha256(path) == self.e147_0["states"][name]["sha256"], f"{path}: sha256 differs from E147's record"
        if m not in self.e147_reads:  # as E147's Run.neuron_r
            self.e147_reads[m] = NR.task_reads(ST.load_task(self.net, REPO / self.inv["maps"][m]["windows"], f"e58/{m}", f"e58/{m}"))
        dW = {k: v.float() for k, v in torch.load(path, weights_only=True).items()}
        return dW, self.e147_reads[m], self.S["p_learn"], self.e147_1["states"][name]["neuron_r"]["r_half"]

    def e147_names(self) -> list[tuple[str, str, int]]:
        maps = [m for p in self.inv["pairs"] for m in (p["reuse"], p["new"])]
        return [(f"e58/{m}_s{s}", m, s) for m in maps for s in self.inv["seeds"]]


# ----------------------------------------------------------------------------- stages


def stage0(st: States, o: Out, T: dict[str, Any]) -> None:
    saved = load(E98_R / "carrying.json")["rows"]
    for name in STAGE0:
        dW, r, p_learn, saved_r_half = st.e98(name)
        rec = carrying(st.net, r, dW, st.S, p_learn)
        ref = saved[name]
        rec["diff"] = {"k_half": [rec["k_half"], ref["k_half"]], "r_half_num": rec["r_half_num"] - ref["r_half_num"],
                       "numerators_max_abs": max(abs(rec["numerators"][k] - ref["numerators"][str(k)]) for k in st.S["k_grid"]),
                       "top_same": rec["top"] == ref["top"], "r_half": rec["r_half"] - saved_r_half}  # fmt: skip
        d = rec["diff"]
        failed = checks(rec, saved_r_half, st.S, T)
        failed += [c for c, ok in {"k_half": rec["k_half"] == ref["k_half"], "top order": d["top_same"],
                                    "r_half_num": abs(d["r_half_num"]) <= T["reproduce_tol"],
                                    "numerators": d["numerators_max_abs"] <= T["reproduce_tol"]}.items() if not ok]  # fmt: skip
        o.put("stage0", name, rec, failed)
    print("STAGE 0 DONE", flush=True)


def stage1(st: States, o: Out, T: dict[str, Any]) -> None:
    todo = [(n, lambda n=n: st.e98(n)) for n in E98_NEW] + [(n, lambda m=m, s=s: st.e147(m, s)) for n, m, s in st.e147_names()]
    for name, get in todo:
        dW, r, p_learn, saved_r_half = get()
        rec = carrying(st.net, r, dW, st.S, p_learn)
        o.put("states", name, rec, checks(rec, saved_r_half, st.S, T))
    print("STAGE 1 DONE", flush=True)


def side(names: list[str], e98: dict[str, Any], mine: dict[str, Any]) -> dict[str, Any]:
    """One side's defined k_half values, the states with none (and why), and the file each value came from."""
    src = lambda n: ("experiments/neuron_r/results/carrying.json .rows", e98[n]) if n in e98 else ("experiments/k_half_fill/results/cells.json .states", mine[n])
    got = {n: src(n) for n in names}
    return {"values": [g["k_half"] for _, g in got.values() if g["k_half"] is not None],
            "none": {n: g.get("why_none", "E98: no k_half") for n, (_, g) in got.items() if g["k_half"] is None},
            "states": names, "sources": sorted({f"{f}[<state>].k_half" for f, _ in got.values()})}  # fmt: skip


def cells(st: States, o: Out) -> None:
    e98, mine = load(E98_R / "carrying.json")["rows"], o.out["states"]
    rows = {pid: (row["reuse"], row["new"]) for pid, row in load(E98_R / "predictions.json")["rows"].items()}
    seeds = st.inv["seeds"]
    rows |= {p["id"]: ([f"e58/{p['reuse']}_s{s}" for s in seeds], [f"e58/{p['new']}_s{s}" for s in seeds]) for p in st.inv["pairs"]}
    for pid, (reuse, new) in rows.items():
        a, b = side(reuse, e98, mine), side(new, e98, mine)
        o.out["cells"][f"{pid}/k_half"] = {"reuse": a["values"], "new": b["values"], "reuse_states": a["states"], "new_states": b["states"],
                                           "none": {**a["none"], **b["none"]}, "source": "; ".join(sorted(set(a["sources"] + b["sources"])))}  # fmt: skip
        log(f"[cells {pid}/k_half] reuse {a['values']} new {b['values']} none {a['none'] | b['none']}")
    o.write()
    print("CELLS DONE", flush=True)


def main() -> int:
    require_filed(PRED)
    T = load(PRED)["settings"]
    S = load(E98_R / "predictions.json")["settings"]
    T["identity_tol"] = load(E98_R / "predictions.json")["stage0"]["identity_tol"]
    assert S["k_grid"] == [2**i for i in range(14)] + [FULL]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E98 and E147 read
    net = ST.Net("cuda")
    st, o = States(net, S), Out()
    o.out["card"] = torch.cuda.get_device_name()
    try:
        stage0(st, o, T)
        stage1(st, o, T)
        cells(st, o)
    except Stopped as e:
        log(f"FAIL: {e}: stop and tell the lead")
        return 1
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
