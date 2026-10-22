"""gamma_shift_map (E30): subcomponents frozen, Gamma's values shifted by one fixed change, scored on the Pile.
Plan: reports/2026-10-22_plan_commute-second-line.md -> Item D, reading (i), and its E30 agent note.

D' = U, V, Delta as at W0; Gamma'(x) = clip(Gamma(x) + s, 0, 1) at every position, Gamma reading W0's activations;
s = E18's fixed change (`regating.shift_of`: mean trigger values over the task's training windows under W0 + dW minus
under W0, alive only). D' is scored against the fine-tuned model W0 + dW by `shifted_harness` (VPD's four KLs and
sparsity loss). Per update: (a) s = 0, (b) s, (c) the ' http' null fine-tune's s, (d) s shuffled among each matrix's
alive subcomponents (one draw); do-nothing through the ordinary harness beside them; s's composition (the readout).

  run.py --predictions   file results/predictions.json (once, before the run)
  run.py                 -> results/gamma_shift_map.json (written after every reading; a rerun resumes), results/large/
Inputs (gitignored): experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
experiments/task_ladder/results/large/windows_r1_t534_v369.pt; E24's committed JSON for the gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import faith_score as FS
from commute import recruit as R
from commute import regating as G
from commute import shift_readout as SR
from commute import shifted_harness as SH

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "gamma_shift_map.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
E24_JSON = EXP / "regating_small" / "results" / "regating_small.json"


@dataclass(frozen=True)
class Config:
    updates: tuple[tuple[str, str, str, float], ...] = (  # label, task, optimiser, rate; scored in this order
        ("emoticon/sgd_0.01", "emoticon", "sgd", 0.01), ("emoticon/adam_3e-05", "emoticon", "adam", 3e-5),
        ("r1_t534_v369/sgd_0.03", "r1_t534_v369", "sgd", 0.03))
    steps: int = 10
    seed: int = 0
    null_token: int = 3944  # ' http', E18/E24's different target
    w0_seeds: tuple[int, ...] = (0, 1, 2)
    pile_rows: int = 256  # 2 blocks of 128 rows, one adversary each (round 2's harness)
    sparsity_block: int = 64  # 4 sparsity blocks; their spread is the floor of a deterministic metric (kv_task)
    pgd_steps: int = 20
    exact_tol: float = 1e-4  # C1: the new path at s = 0 vs the ordinary harness, relative
    g1_tol: float = 0.05  # G1: r and the shared pattern's trigger recovery vs E24, absolute
    thresholds: tuple[float, ...] = (0.01, 0.05, 0.1)
    topk: tuple[int, ...] = (10, 30, 100, 300, 1000)
    n_shuffle: int = 20  # the halves' sign-kept shuffles (E18 amendment 3)
    pool_min: float = 0.1  # turned-down pool: alive with m0 >= this (E18 amendment 3's POOL_MIN)

    def __post_init__(self) -> None:
        assert self.pile_rows % 128 == 0 and self.pile_rows % self.sparsity_block == 0
        assert len({u[0] for u in self.updates}) == len(self.updates) and self.updates[0][0] == "emoticon/sgd_0.01"
        assert list(self.topk) == sorted(self.topk) and self.w0_seeds[0] == 0


CFG = Config()
E24_KEY = {"emoticon/sgd_0.01": ("states", "sgd_0.01"), "emoticon/adam_3e-05": ("states", "adam_3e-05"),
           "r1_t534_v369/sgd_0.03": ("extra", "sgd_0.03")}
PILE_KEYS = ("stochastic_kl", "ci_masked_kl", "unmasked_kl", "pgd_kl")
P1_MIN, P1_NEAR = 0.6, 0.1  # P1: emoticon R >= 0.6; every update within 0.1 of E24's trigger-only R_shared
CI_MIN_FLOOR = 0.02  # P2's delta: max(W0's CI-masked spread over adversary seeds, faith_score.W0_CI_TOL)
CHEAT, NULL_LEARNED, P4_MAX, P5_LINE = 1.05, 0.5, 0.1, 0.5

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-22_plan_commute-second-line.md -> Item D, reading (i) (user, 2026-10-22), and the E30 agent "
            "note. Filed before any run; never edited after.",
    "map": "D' = U, V, Delta as at W0; Gamma'(x) = clip(Gamma(x) + s, 0, 1) at every position, Gamma reading W0's "
           "activations; s = mean over the task's TRAINING windows of Gamma's trigger values under W0 + dW minus under W0 "
           "(alive only; regating.shift_of). Scored against the fine-tuned model W0 + dW's outputs (shifted_harness).",
    "updates": "10-step arm-A fine-tunes of all 24 matrices, seed 0, regenerated as E24: emoticon SGD 0.01, emoticon "
               "Adam 3e-5, r1_t534_v369 SGD 0.03; each with a ' http' (3944) null fine-tune by the same optimiser and rate",
    "arms": {"a_zero": "s = 0: W0's decomposition read against the fine-tuned model (nothing changed)",
             "b_shift": "s (the map under test)", "c_null": "the ' http' null's s (wrong target)",
             "d_shuffled": "s permuted among each matrix's alive subcomponents (recruit.shuffled, seed 0, one draw)",
             "nothing": "do-nothing (dW in Delta, Gamma reading W0 + dW) through the ordinary harness"},
    "measurements": {
        "pile": "round 2's harness on Pile val rows 0-255 (2 blocks of 128, one 20-step adversary per block, seed 0): "
                "stochastic, CI-masked, unmasked and adversarial KL against the target's outputs; ratios to W0 (seed 0)",
        "sparsity": "VPD's importance-minimality loss (final p, beta from final_config.yaml) on the same rows in 4 blocks "
                    "of 64, upper-leaky values + (clip(lower + s) - lower); ratio = mean over blocks",
        "R_everywhere": "trigger recovery on the held-out windows with Gamma' at every position: (lo - L00) / (L01 - L00) "
                        "divided by the own values' recovery rho_trig (E24's units)",
        "readout": "per update (s, the null's s, and s = 0): alive entries with |s| > 0.01 / 0.05 / 0.1; shares of sum |s| "
                   "by layer and sign; E18's trigger-only recovery (R units) of s kept on one layer, of each half (up, "
                   "down, down on the pool m0 >= 0.1) and of the halves shuffled keeping sign (20 draws); of the top-k "
                   "entries by |s| pooled over matrices (k = 10, 30, 100, 300, 1000), as a share of the whole s's; Spearman "
                   "of s with W0's mean trigger importance m0 and of |s| with |u_c^T dW v_c| / (|u_c||v_c|)"},
    "controls": {
        "C1": "W0, ordinary harness, adversary seeds 0, 1, 2: faith_score.w0_check passes; the new path with s = 0 against "
              "a copy of W0 reproduces seed 0's four KLs and the 4 sparsity values within 1e-4 relative. Fail: stop",
        "G1": "each regenerated update: r and the shared pattern's trigger-only recovery within 0.05 of E24's "
              "(regating_small.json). Fail: stop",
        "a_zero_windows": "R_everywhere(a) = 0 exactly (Gamma' = Gamma = W0's gates)"},
    "expected": {
        "R_everywhere": "a 0; b ~ E24's R_shared (0.74 emoticon SGD, 0.74 Adam, 0.40 r1); c <= 0 (E24 trigger-only "
                        "-0.17 to -0.30); d ~ 0 (E18 shuffles -0.005 to +0.003)",
        "pile ratios to W0": "a: about 1 on every KL (the fine-tunes move the Pile little: KL elsewhere 0.015-0.15 on "
                             "windows); unmasked of a-d equal by construction; nothing: as E21 (adversarial 0.95-1.2 at "
                             "step 10 on emoticon SGD); b: CI-masked above a (P2), adversarial two-sided (raising values "
                             "narrows the adversary, lowering it), sparsity above a",
        "readout at s = 0": "every count 0, every recovery 0, correlations undefined",
        "readout under the null": "counts of the same order as the task's (E24: 110 switched on against 154); "
                                  "recoveries <= 0; top-k shares undefined in sign (whole is negative)",
        "readout of s": "emoticon: ~500 entries above 0.1 of ~9,500 alive (E18 bins: 243 down, 98 + 210 up), up and down "
                        "halves each 0.2-0.35, shuffled halves <= 0 (E18 amendment 3), layer 3 the largest share of the "
                        "switched-on set (E24 by_layer_on 102 of 154); no expectation for the correlations"},
    "predictions": [
        {"id": "P1", "kind": "directional (H')",
         "rule": f"R_everywhere(b) >= {P1_MIN} on both emoticon updates, and |R_everywhere(b) - E24's R_shared| <= {P1_NEAR} "
                 "on every update. Absolute, in shares of rho_trig",
         "predicted": "pass, medium confidence: the change at non-trigger window positions is not expected to move the "
                      "trigger log-odds much",
         "blind_to": "the Pile; anything off the trigger position", "positive_control": "rho_trig and G1",
         "noise_floor": "deterministic given the state; E18 seeds agree within 0.02"},
        {"id": "P2", "kind": "directional (H' applied everywhere harms the Pile)",
         "rule": "CI-masked KL ratio (b)/(a) - 1 > delta reads 'harm', < -delta 'less', else 'no change'; delta = "
                 f"max(W0's CI-masked spread over adversary seeds 0-2, {CI_MIN_FLOOR}). Pass: 'harm' on every update. "
                 "The null's (c)/(a) - 1 reported beside it (the plan's 'similar size'), not scored",
         "predicted": "harm on every update, medium confidence (E18: the change lowers the target off the trigger on the "
                      "word tasks and raises it 1.1 nats on emoticon Pile rows; neither is the fine-tuned model)",
         "blind_to": "where on the Pile the harm sits; the stochastic and adversarial readings (P3)",
         "positive_control": "C1 (s = 0 reproduces W0); the shuffled s (d) shows the size of a same-values change",
         "noise_floor": "CI-masked KL is deterministic under a fixed Gamma'; bf16 spread across seeds, floored at 0.02"},
        {"id": "P3", "kind": "two-sided, named outcomes",
         "rule": "per update: faith_score.verdict of the adversarial ratio (b)/(a) against the same-job W0 floor f, and "
                 f"the sparsity ratio (b)/(a) (mean over 4 blocks) against {CHEAT}. Outcome 'cheating' if sparsity > "
                 f"{CHEAT} (whatever the verdict); else 'faithful' on pass, 'unfaithful' on fail, 'inconclusive' between",
         "predicted": "no direction filed (two-sided); lean 'cheating' on every update, medium-low confidence: raising "
                      "values everywhere narrows the adversary but every raised value is charged at every Pile position",
         "blind_to": "a longer adversary (E20 amendment 3: +7% at 160 steps); the trigger",
         "positive_control": "C1", "noise_floor": "W0's PGD floor f (max/min - 1 over seeds 0-2); sparsity block spread"},
        {"id": "P4", "kind": "control",
         "rule": f"R_everywhere(c) <= {P4_MAX} on every update whose null learned (held-out P(' http') >= {NULL_LEARNED}); "
                 "'no power' otherwise. Pass: no fail and at least one pass",
         "predicted": "pass (E24 trigger-only -0.17 to -0.30)", "blind_to": "other wrong targets",
         "positive_control": "is itself the control", "noise_floor": "E18 seeds agree within 0.01"},
        {"id": "P5", "kind": "two-sided, named outcomes (the readout)",
         "rule": f"per update: the top-100 entries by |s| (pooled over matrices, the rest 0) recover >= {P5_LINE} of the "
                 "whole s's trigger-only recovery: 'a few larger ones'; < 0.5: 'many small ones act together'. Relative",
         "predicted": "no direction filed; lean 'many small ones act together' on emoticon (E18 bins: the >0.3 bin, 210 "
                      "entries, gives about half of the unclipped recovery and the < -0.1 bin, 243, about a third), low "
                      "confidence; r1 has fewer large entries (56 above 0.3, 134 below -0.1) and could read either way",
         "blind_to": "which entries beyond magnitude order; interactions between entries (the curve is reported)",
         "positive_control": "k = 1000 should approach 1; s = 0 gives 0", "noise_floor": "deterministic given s"}],
    "blind_to": "accumulation along training (one shot, step 10); input-conditioned gating; one training seed; one 67M "
                "model; 256 Pile rows",
    "reported_not_scored": "the stochastic and unmasked ratios; (d) on the Pile; do-nothing; the readout beyond P5",
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    tmp.replace(OUT)


def e24_ref(label: str) -> dict[str, float]:
    sec, key = E24_KEY[label]
    d = json.loads(E24_JSON.read_text())
    st = (d["states"] if sec == "states" else d["extra"]["states"])[key]
    return {"r": st["r"], "rho_shared": st["rho"]["shared"], "R_shared": st["R_shared"], "rho_trig": st["rho"]["trig"]}


class Run:
    def __init__(self, device: str, out: dict[str, Any]) -> None:
        from commute import vpd
        from commute.faithfulness import pile_blocks

        self.device, self.out = device, out
        self.model = vpd.load_component_model(device)
        self.W0 = vpd.target_weights(self.model)
        self.uv = {k: (u.float().cpu(), v.float().cpu()) for k, (u, v) in vpd.component_weights(self.model).items()}
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}
        self.blocks = pile_blocks(CFG.pile_rows, log)
        rows = torch.cat([torch.cat(b) for b in self.blocks])
        self.sp_blocks = [[rows[s : s + CFG.sparsity_block]] for s in range(0, rows.shape[0], CFG.sparsity_block)]
        sp = vpd.sparsity_config()
        self.sp_cfg = (sp["pnorm"], sp["beta"], sp["eps"])
        self.twin = copy.deepcopy(self.model)  # holds the weights D' is scored against
        self.tasks: dict[str, G.Task] = {}

    def task(self, name: str) -> G.Task:
        if name not in self.tasks:
            self.tasks[name] = G.prepare(self.model, name, *G.load_windows(name, TRANSPORT, LADDER))
        return self.tasks[name]

    def set_twin(self, dW: dict[str, Tensor]) -> None:
        from commute.vpd import set_target_weights

        set_target_weights(self.twin, {k: self.W0[k] + dW[k].to(self.W0[k].device, self.W0[k].dtype) for k in self.W0})

    def reading(self, name: str, fn: Any) -> dict[str, Any]:
        """A Pile reading, computed once (a rerun reads it back from the results file)."""
        done = self.out.setdefault("pile", {})
        if name not in done:
            t0 = time.time()
            done[name] = {**fn(), "seconds": time.time() - t0}
            log(f"[{name}] {json.dumps({k: v for k, v in done[name].items() if k != 'sparsity'})} sparsity "
                f"{done[name]['sparsity']}")
            dump(self.out)
        return done[name]

    def ordinary(self, seed: int) -> dict[str, Any]:
        from commute.faithfulness import pile_faithfulness, sparsity_loss

        return {**pile_faithfulness(self.model, self.blocks, seed, CFG.pgd_steps, log),
                "sparsity": [sparsity_loss(self.model, b, *self.sp_cfg) for b in self.sp_blocks]}

    def shifted(self, shift: dict[str, Tensor]) -> dict[str, Any]:
        """D' with this shift against the twin's weights."""
        return {**SH.shifted_pile(self.model, self.twin, self.blocks, shift, 0, CFG.pgd_steps, log),
                "sparsity": [SH.shifted_sparsity_loss(self.model, b, shift, *self.sp_cfg) for b in self.sp_blocks]}


def zero_like(shift: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: torch.zeros_like(v) for k, v in shift.items()}


def c1(r: Run) -> bool:
    """W0 through the ordinary harness (three adversary seeds, w0_check) and the new path at s = 0 against W0's own
    outputs: every number equal within exact_tol, or stop."""
    pile = {s: r.reading(f"W0/seed{s}", lambda s=s: r.ordinary(s)) for s in CFG.w0_seeds}
    w0 = FS.w0_check({s: {k: p[k] for k in PILE_KEYS} for s, p in pile.items()})
    zero = {k: torch.zeros(v.shape) for k, v in r.task("emoticon").m0.items()}
    r.set_twin(zero)
    new = r.reading("C1/zero_vs_W0", lambda: r.shifted(zero))
    rel = {k: new[k] / pile[0][k] - 1 for k in PILE_KEYS}
    rel["sparsity"] = max(abs(a / b - 1) for a, b in zip(new["sparsity"], pile[0]["sparsity"], strict=True))
    ok = bool(w0["pass"]) and all(abs(v) <= CFG.exact_tol for v in rel.values())
    sp = pile[0]["sparsity"]
    r.out["C1"] = {"w0_check": {k: v for k, v in w0.items() if k != "pile"}, "rel": rel, "ok": ok,
                   "ci_floor": FS.seed_floor(pile, "ci_masked_kl"), "sparsity_block_spread": max(sp) / min(sp) - 1}
    log(f"[C1] w0_check pass {w0['pass']} floor {w0['pgd_floor']:.4f} | s=0 vs harness rel {json.dumps(rel)} | "
        f"{'ok' if ok else 'FAIL'}")
    if not ok:  # diagnostic only: is the ordinary harness itself reproducible in this job?
        again = r.ordinary(0)
        r.out["C1"]["harness_repeat_rel"] = {k: again[k] / pile[0][k] - 1 for k in PILE_KEYS}
        log(f"[C1] ordinary harness seed 0 repeated: rel {json.dumps(r.out['C1']['harness_repeat_rel'])}")
    dump(r.out)
    return ok


def trigger_cells(r: Run, t: G.Task, dW: dict[str, Tensor], s: dict[str, Tensor]) -> dict[str, Any]:
    """E24's readings of the do-nothing state at the trigger: r, gap, own values' and the shared pattern's recovery."""
    from commute.local_faith import route_cells

    with G.state_of(r.model, r.W0, dW):
        cur = route_cells(r.model, t.held.tokens, t.half, t.target, G.BATCH)
    gap, gain = cur["L"] - t.base["L"], cur["T"] - t.base["T"]
    g0, h = t.base["gates"], t.half
    rho_trig = G.recovery(r.model, t, R.at_trigger(cur["gates"], h), gap)
    return {"r": G.div(gap, gain), "gap": gap, "gain": gain, "rho_trig": rho_trig,
            "rho_shared": G.recovery(r.model, t, R.shifted(g0, h, s), gap)}


def update(r: Run, label: str, task: str, opt: str, lr: float) -> dict[str, Any]:
    """The fine-tune and its ' http' null (same optimiser and rate), both shifts, E24's trigger cells and gate G1."""
    t = r.task(task)
    dW, ev = G.finetune(r.model, t, t.target, opt, lr, CFG.steps, CFG.seed, r.device, log)
    dWn, evn = G.finetune(r.model, t, CFG.null_token, opt, lr, CFG.steps, CFG.seed, r.device, log)
    s, _ = G.shift_of(r.model, r.W0, dW, t, r.alive)
    sn, _ = G.shift_of(r.model, r.W0, dWn, t, r.alive)
    cells, ref = trigger_cells(r, t, dW, s), e24_ref(label)
    diff = {"r": cells["r"] - ref["r"], "rho_shared": cells["rho_shared"] - ref["rho_shared"]}
    g1 = {"ref": ref, "diff": diff, "ok": all(abs(v) <= CFG.g1_tol for v in diff.values())}
    torch.save({"shift": s, "shift_null": sn}, LARGE / f"{label.replace('/', '_')}_shift.pt")
    log(f"[{label}] P {ev['p_o']:.4f} size {ev['rel_norm_max']:.5f} null P {evn['p_o']:.4f} | {json.dumps(cells)} | "
        f"G1 {json.dumps(diff)} {'ok' if g1['ok'] else 'FAIL'}")
    return {"dW": dW, "dW_null": dWn, "s": s, "s_null": sn, "task": t, "rec": {"heldout": ev, "null_heldout": evn, "cells": cells, "G1": g1}}


def readout(r: Run, t: G.Task, s: dict[str, Tensor], dW: dict[str, Tensor], cells: dict[str, Any]) -> dict[str, Any]:
    """What s is built from. Recoveries are E18's trigger-only pattern clip(g0(x) + s'), W0's values elsewhere, as
    shares of the own values' recovery (R units)."""
    g0, h, gap, trig = t.base["gates"], t.half, cells["gap"], cells["rho_trig"]

    def rec(sh: dict[str, Tensor]) -> float:
        return G.div(G.recovery(r.model, t, R.shifted(g0, h, sh), gap), trig)

    up, down = R.signed(s, True), R.signed(s, False)
    pool = {k: r.alive[k] & (t.m0[k] >= CFG.pool_min) for k in r.alive}
    down_p = R.on_pool(down, pool)
    whole = rec(s)
    top = {str(k): rec(SR.top_k(s, k)) for k in CFG.topk}
    return {"counts": SR.counts(s, r.alive, CFG.thresholds), "mass": SR.mass_split(s), "R_whole": whole,
            "R_by_layer": {str(lay): rec(SR.on_layer(s, lay)) for lay in range(4)},
            "R_halves": {"up": rec(up), "down": rec(down), "down_pool": rec(down_p)},
            "R_halves_shuffled_mean": {
                "up": sum(rec(R.shuffled(up, r.alive, i)) for i in range(CFG.n_shuffle)) / CFG.n_shuffle,
                "down_pool": sum(rec(R.shuffled(down_p, pool, i)) for i in range(CFG.n_shuffle)) / CFG.n_shuffle},
            "R_topk": top, "topk_share_of_whole": {k: G.div(v, whole) for k, v in top.items()},
            "corr": {"s_vs_m0": SR.spearman(SR.pooled(s, r.alive), SR.pooled(t.m0, r.alive)),
                     "abs_s_vs_write_energy": SR.spearman(SR.pooled(s, r.alive).abs(),
                                                          SR.pooled(SR.write_energy(r.uv, dW), r.alive))}}


def arms_of(r: Run, s: dict[str, Tensor], sn: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    return {"a_zero": zero_like(s), "b_shift": s, "c_null": sn, "d_shuffled": R.shuffled(s, r.alive, 0)}


def scored(r: Run, label: str, u: dict[str, Any]) -> None:
    """(a)-(d): reading (i) on the held-out windows (Gamma' at every position), then on the Pile against W0 + dW;
    do-nothing through the ordinary harness last."""
    t, cells, rec = u["task"], u["rec"]["cells"], u["rec"]
    arms = arms_of(r, u["s"], u["s_null"])
    rho = {x: G.div(G.lo_of(r.model, t, SH.everywhere(t.base["gates"], sh)) - t.base["L"], cells["gap"])
           for x, sh in arms.items()}
    rec["R_everywhere"] = {x: G.div(v, cells["rho_trig"]) for x, v in rho.items()}
    log(f"[{label}] trigger recovery, Gamma' everywhere (R): {json.dumps(rec['R_everywhere'])}")
    dump(r.out)
    r.set_twin(u["dW"])
    for x, sh in arms.items():
        r.reading(f"{label}/{x}", lambda sh=sh: r.shifted(sh))
    with G.state_of(r.model, r.W0, u["dW"]):
        r.reading(f"{label}/nothing", lambda: r.ordinary(0))


# ----------------------------------------------------------------------------- scoring (the rules in PREDICTIONS)


def sp_ratio(p: dict[str, Any], q: dict[str, Any]) -> dict[str, float]:
    rs = [a / b for a, b in zip(p["sparsity"], q["sparsity"], strict=True)]
    return {"mean": sum(rs) / len(rs), "min": min(rs), "max": max(rs)}


def ratios(out: dict[str, Any], label: str) -> dict[str, Any]:
    """Every reading of one update as ratios to W0 (seed 0, ordinary harness), and (b) over (a)."""
    P, w0 = out["pile"], out["pile"]["W0/seed0"]
    res: dict[str, Any] = {}
    for x in ("a_zero", "b_shift", "c_null", "d_shuffled", "nothing"):
        if f"{label}/{x}" in P:
            p = P[f"{label}/{x}"]
            res[x] = {**{k: p[k] / w0[k] for k in PILE_KEYS}, "sparsity": sp_ratio(p, w0)["mean"]}
    if all(f"{label}/{x}" in P for x in ("a_zero", "b_shift", "c_null")):
        a, b, c = (P[f"{label}/{x}"] for x in ("a_zero", "b_shift", "c_null"))
        res["b_over_a"] = {**{k: b[k] / a[k] for k in PILE_KEYS}, "sparsity": sp_ratio(b, a)}
        res["c_over_a"] = {**{k: c[k] / a[k] for k in PILE_KEYS}, "sparsity": sp_ratio(c, a)}
    return res


def score(out: dict[str, Any]) -> dict[str, Any]:
    c1 = out["C1"]
    f, delta = c1["w0_check"]["pgd_floor"], max(c1["ci_floor"], CI_MIN_FLOOR)
    ups = {k: v for k, v in out["updates"].items() if v["G1"]["ok"] and "R_everywhere" in v}
    sc: dict[str, Any] = {"floors": {"pgd_f": f, "ci_delta": delta}, "G1": {k: v["G1"]["ok"] for k, v in out["updates"].items()}}
    Rb = {k: v["R_everywhere"]["b_shift"] for k, v in ups.items()}
    p1 = {k: {"R": x, "E24_R_shared": ups[k]["G1"]["ref"]["R_shared"], "ok": abs(x - ups[k]["G1"]["ref"]["R_shared"]) <= P1_NEAR
              and (x >= P1_MIN or not k.startswith("emoticon"))} for k, x in Rb.items()}
    sc["P1"] = {"per": p1, "pass": all(v["ok"] for v in p1.values())}
    rat = {k: ratios(out, k) for k in ups}
    rat = {k: v for k, v in rat.items() if "b_over_a" in v}
    p2 = {k: v["b_over_a"]["ci_masked_kl"] - 1 for k, v in rat.items()}
    sc["P2"] = {"ci_rel": p2, "null_ci_rel": {k: v["c_over_a"]["ci_masked_kl"] - 1 for k, v in rat.items()},
                "reading": {k: "harm" if d > delta else "less" if d < -delta else "no change" for k, d in p2.items()}}
    sc["P2"]["pass"] = bool(p2) and all(v == "harm" for v in sc["P2"]["reading"].values())
    p3 = {}
    for k, v in rat.items():
        ver, spr = FS.verdict(v["b_over_a"]["pgd_kl"], f), v["b_over_a"]["sparsity"]["mean"]
        p3[k] = {"pgd_ratio": v["b_over_a"]["pgd_kl"], "verdict": ver, "sparsity_ratio": spr,
                 "outcome": "cheating" if spr > CHEAT else {"pass": "faithful", "fail": "unfaithful"}.get(ver, "inconclusive")}
    sc["P3"] = {"per": p3}
    p4 = {k: ("no power" if v["null_heldout"]["p_o"] < NULL_LEARNED else
              "pass" if v["R_everywhere"]["c_null"] <= P4_MAX else "fail") for k, v in ups.items()}
    sc["P4"] = {"R_null": {k: v["R_everywhere"]["c_null"] for k, v in ups.items()}, "reading": p4,
                "pass": "fail" not in p4.values() and "pass" in p4.values()}
    p5 = {k: v["readout"]["shift"]["topk_share_of_whole"]["100"] for k, v in ups.items()}
    sc["P5"] = {"top100_share": p5, "reading": {k: "a few larger ones" if x >= P5_LINE else "many small ones act together"
                                                for k, x in p5.items()}}
    sc["ratios"] = rat
    return sc


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    t0 = time.time()
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {}
    out.update(device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", config=repr(CFG))
    out.setdefault("updates", {})
    r = Run(device, out)
    log(f"[start] {out['device']}; resuming {sorted(out.get('pile', {}))}")
    if not c1(r):
        out["status"] = "stopped at C1"
        dump(out)
        return log("C1 FAILED: stopping")
    states = {}
    for label, task, opt, lr in CFG.updates:
        states[label] = u = update(r, label, task, opt, lr)
        out["updates"][label] = u["rec"]
        dump(out)
        if not u["rec"]["G1"]["ok"]:
            out["status"] = f"stopped at G1 ({label})"
            dump(out)
            return log(f"G1 FAILED on {label}: stopping")
    for label, u in states.items():
        u["rec"]["readout"] = {n: readout(r, u["task"], sh, dW, u["rec"]["cells"]) for n, sh, dW in (
            ("shift", u["s"], u["dW"]), ("null", u["s_null"], u["dW_null"]), ("zero", zero_like(u["s"]), u["dW"]))}
        log(f"[{label} readout] {json.dumps(_finite(u['rec']['readout']['shift']))}")
        dump(out)
    for label, u in states.items():
        scored(r, label, u)
        out["score"] = score(out)
        dump(out)
    out.update(status="done", seconds=time.time() - t0, score=score(out))
    dump(out)
    log("score " + json.dumps(_finite(out["score"])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions.json and exit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        RESULTS.mkdir(parents=True, exist_ok=True)
        PRED.write_text(json.dumps({**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        return print(f"filed {PRED}")
    run(args.device)


if __name__ == "__main__":
    main()
