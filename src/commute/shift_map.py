"""The bench shared by the fixed-shift map's runs (E30 `experiments/gamma_shift_map/run.py` and its amendments).

W0's decomposition, the Pile rows, and a second model (the twin) holding the weights D' is scored against. A Pile
reading is computed once and stored in the run's results dict; a rerun reads it back (resume). `c1` is E30's control:
W0 through the ordinary harness at three adversary seeds, and the shifted path at s = 0 against a copy of W0, equal to
1e-4. `trigger_cells` is E24's reading of a fine-tuned state at the trigger. Moved here from E30's run.py on
2026-10-22 for its amendment 1; E30's behaviour is unchanged.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import faith_score as FS
from commute import recruit as R
from commute import regating as G
from commute import shifted_harness as SH

PILE_KEYS = ("stochastic_kl", "ci_masked_kl", "unmasked_kl", "pgd_kl")
Shift = dict[str, Tensor]


def zero_like(shift: Shift) -> Shift:
    return {k: torch.zeros_like(v) for k, v in shift.items()}


class Bench:
    def __init__(self, device: str, out: dict[str, Any], transport: Path, ladder: Path, pile_rows: int,
                 sparsity_block: int, pgd_steps: int, log: Callable[[str], None],
                 dump: Callable[[dict[str, Any]], None]) -> None:
        from commute import vpd
        from commute.faithfulness import pile_blocks

        self.device, self.out, self.log, self.dump = device, out, log, dump
        self.transport, self.ladder, self.pgd_steps = transport, ladder, pgd_steps
        self.model = vpd.load_component_model(device)
        self.W0 = vpd.target_weights(self.model)
        alive = torch.load(transport / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}
        self.blocks = pile_blocks(pile_rows, log)
        rows = torch.cat([torch.cat(b) for b in self.blocks])
        self.sp_blocks = [[rows[s : s + sparsity_block]] for s in range(0, rows.shape[0], sparsity_block)]
        sp = vpd.sparsity_config()
        self.sp_cfg = (sp["pnorm"], sp["beta"], sp["eps"])
        self.twin = copy.deepcopy(self.model)  # holds the weights D' is scored against
        self.tasks: dict[str, G.Task] = {}

    def task(self, name: str) -> G.Task:
        if name not in self.tasks:
            self.tasks[name] = G.prepare(self.model, name, *G.load_windows(name, self.transport, self.ladder))
        return self.tasks[name]

    def set_twin(self, dW: Shift) -> None:
        from commute.vpd import set_target_weights

        set_target_weights(self.twin, {k: self.W0[k] + dW[k].to(self.W0[k].device, self.W0[k].dtype) for k in self.W0})

    def reading(self, name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """A Pile reading, computed once (a rerun reads it back from the results file)."""
        done = self.out.setdefault("pile", {})
        if name not in done:
            t0 = time.time()
            done[name] = {**fn(), "seconds": time.time() - t0}
            self.log(f"[{name}] {json.dumps({k: v for k, v in done[name].items() if k != 'sparsity'})} sparsity "
                     f"{done[name]['sparsity']}")
            self.dump(self.out)
        return done[name]

    def ordinary(self, seed: int) -> dict[str, Any]:
        from commute.faithfulness import pile_faithfulness, sparsity_loss

        return {**pile_faithfulness(self.model, self.blocks, seed, self.pgd_steps, self.log),
                "sparsity": [sparsity_loss(self.model, b, *self.sp_cfg) for b in self.sp_blocks]}

    def shifted(self, shift: Shift) -> dict[str, Any]:
        """D' with this shift against the twin's weights."""
        return {**SH.shifted_pile(self.model, self.twin, self.blocks, shift, 0, self.pgd_steps, self.log),
                "sparsity": [SH.shifted_sparsity_loss(self.model, b, shift, *self.sp_cfg) for b in self.sp_blocks]}


def c1(b: Bench, seeds: tuple[int, ...], exact_tol: float) -> bool:
    """W0 through the ordinary harness (adversary `seeds`, w0_check) and the new path at s = 0 against W0's own
    outputs: every number equal within exact_tol, or stop. Writes b.out["C1"]."""
    pile = {s: b.reading(f"W0/seed{s}", lambda s=s: b.ordinary(s)) for s in seeds}
    w0 = FS.w0_check({s: {k: p[k] for k in PILE_KEYS} for s, p in pile.items()})
    zero = {k: torch.zeros(v.shape) for k, v in b.task("emoticon").m0.items()}  # s = 0, shaped like the gates
    b.set_twin(zero_like(b.W0))  # dW = 0, shaped like the weights
    new = b.reading("C1/zero_vs_W0", lambda: b.shifted(zero))
    rel = {k: new[k] / pile[0][k] - 1 for k in PILE_KEYS}
    rel["sparsity"] = max(abs(x / y - 1) for x, y in zip(new["sparsity"], pile[0]["sparsity"], strict=True))
    ok = bool(w0["pass"]) and all(abs(v) <= exact_tol for v in rel.values())
    sp = pile[0]["sparsity"]
    b.out["C1"] = {"w0_check": {k: v for k, v in w0.items() if k != "pile"}, "rel": rel, "ok": ok,
                   "ci_floor": FS.seed_floor(pile, "ci_masked_kl"), "sparsity_block_spread": max(sp) / min(sp) - 1}
    b.log(f"[C1] w0_check pass {w0['pass']} floor {w0['pgd_floor']:.4f} | s=0 vs harness rel {json.dumps(rel)} | "
          f"{'ok' if ok else 'FAIL'}")
    if not ok:  # diagnostic only: is the ordinary harness itself reproducible in this job?
        again = b.ordinary(0)
        b.out["C1"]["harness_repeat_rel"] = {k: again[k] / pile[0][k] - 1 for k in PILE_KEYS}
        b.log(f"[C1] ordinary harness seed 0 repeated: rel {json.dumps(b.out['C1']['harness_repeat_rel'])}")
    b.dump(b.out)
    return ok


def trigger_cells(b: Bench, t: G.Task, dW: Shift, s: Shift) -> dict[str, Any]:
    """E24's readings of the do-nothing state at the trigger: r, gap, own values' and the shared pattern's recovery."""
    from commute.local_faith import route_cells

    with G.state_of(b.model, b.W0, dW):
        cur = route_cells(b.model, t.held.tokens, t.half, t.target, G.BATCH)
    gap, gain = cur["L"] - t.base["L"], cur["T"] - t.base["T"]
    g0, h = t.base["gates"], t.half
    rho_trig = G.recovery(b.model, t, R.at_trigger(cur["gates"], h), gap)
    return {"r": G.div(gap, gain), "gap": gap, "gain": gain, "rho_trig": rho_trig,
            "rho_shared": G.recovery(b.model, t, R.shifted(g0, h, s), gap)}


def r_everywhere(b: Bench, t: G.Task, cells: dict[str, Any], shift: Shift) -> float:
    """Reading (i) on the held-out windows: Gamma' at every window position, recovery (lo - L00) / gap as a share of the
    own values' recovery rho_trig (R units)."""
    rho = G.div(G.lo_of(b.model, t, SH.everywhere(t.base["gates"], shift)) - t.base["L"], cells["gap"])
    return G.div(rho, cells["rho_trig"])
