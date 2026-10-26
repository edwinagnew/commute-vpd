"""gamma_shift_map amendment 1 (E30a): the raised and the lowered half of the fixed change, each applied everywhere.
Plan: reports/plans/2026-10-22_plan_commute-second-line.md -> "Amendment 1 to E30". E30's run.py and predictions stand.

E30 applied Gamma'(x) = clip(Gamma(x) + s, 0, 1) at every position and the adversary found 8x the s = 0 damage on
emoticon SGD 0.01, while s shuffled barely moved it. Here, on that update only: s = 0, the raised half s+ = max(s, 0),
the lowered half s- = min(s, 0), and the whole s (the positive control: it must fail again), scored on the Pile against
the fine-tuned model as E30; their trigger recovery with Gamma' everywhere; and how far s- and E30's shuffled s's
lowered half lower Gamma's values on the Pile (`shift_readout.opened`).

  halves.py --predictions   file results/predictions_amend1.json (once, before the run)
  halves.py                 -> results/gamma_shift_map_amend1.json (written after every reading; a rerun resumes)
Inputs (gitignored): experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt};
E30's committed results/gamma_shift_map.json (G1's and P2's references).
"""

from __future__ import annotations

import argparse
import json
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
from commute import shift_map as SM
from commute import shift_readout as SR

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions_amend1.json", RESULTS / "gamma_shift_map_amend1.json"
E30_JSON = RESULTS / "gamma_shift_map.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"


@dataclass(frozen=True)
class Config:
    label: str = "emoticon/sgd_0.01"  # E30's first update, regenerated as E30 did
    task: str = "emoticon"
    optimizer: str = "sgd"
    lr: float = 0.01
    steps: int = 10
    seed: int = 0
    w0_seeds: tuple[int, ...] = (0, 1, 2)
    pile_rows: int = 256
    sparsity_block: int = 64
    pgd_steps: int = 20
    exact_tol: float = 1e-4  # C1, as E30
    g1_tol: float = 0.05  # G1: r, rho_trig and rho_shared vs E30's readings of the same update, absolute
    arms: tuple[str, ...] = ("zero", "up", "down", "shift")  # Pile order: the reference first, the known fail last

    def __post_init__(self) -> None:
        assert self.pile_rows % 128 == 0 and self.pile_rows % self.sparsity_block == 0 and self.w0_seeds[0] == 0
        assert self.arms[0] == "zero"


CFG = Config()
P2_NEAR, CHEAT, P4_LINE = 0.1, 1.05, 3.0

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-22_plan_commute-second-line.md -> 'Amendment 1 to E30' (approved class 4: the commute line's "
            "named next step, E30's 'What next' 1). Filed before any run; E30's predictions.json stands unedited.",
    "update": "emoticon, arm-A SGD 0.01, 10 steps, seed 0, regenerated as E30 (E30 G1: regeneration matched to 0.0)",
    "arms": {"zero": "s = 0 (the reference every ratio divides by)", "up": "s+ = max(s, 0) everywhere",
             "down": "s- = min(s, 0) everywhere", "shift": "the whole s everywhere (E30's b_shift, repeated in-job)"},
    "measurements": {
        "pile": "E30's: shifted_harness on Pile val rows 0-255 (2 blocks of 128, one 20-step adversary per block, seed 0) "
                "against W0 + dW's outputs; ratios x / zero of the adversarial (pgd), CI-masked and stochastic KL; "
                "sparsity ratio = mean over 4 blocks of 64 of x / zero",
        "R_everywhere": "E30's: trigger recovery on the held-out windows with Gamma' at every position, as a share of the "
                        "own values' recovery rho_trig",
        "opened": "mean over the 256 Pile rows' positions of sum_c min(Gamma_c(x), |s_c|) over s's lowered entries "
                  "(shift_readout.opened; Gamma reading W0), for s- and for the lowered half of E30's shuffled s "
                  "(recruit.shuffled(s, alive, 0), the arm that barely moved the adversary)"},
    "controls": {
        "C1": "first and alone: W0, ordinary harness, adversary seeds 0, 1, 2: faith_score.w0_check passes; the shifted "
              "path at s = 0 against a copy of W0 reproduces seed 0's four KLs and 4 sparsity values within 1e-4 "
              "relative. Fail: stop (early-stop rule)",
        "G1": "the regenerated update's r, rho_trig and rho_shared within 0.05 of E30's (gamma_shift_map.json). Fail: stop",
        "PC": "positive control: the whole s fails the adversarial verdict again (E30: 8.0x zero). If it does not, P1 "
              "reads 'no power'"},
    "expected": {
        "C1": "rel 0.0 on all five (E30: 0.0); W0 floor f 0.035 on a 4090, about 0.21 on an L40",
        "G1": "differences 0.0 on the same card type, <= 0.01 across cards; r 0.616, rho_trig 0.977, rho_shared 0.724",
        "R_everywhere": "zero 0 exactly; shift 0.748 (E30); up ~ 0.39 and down ~ 0.24 (E30's trigger-only halves)",
        "pgd ratio to zero": "shift ~ 8 (E30 8.0); down >= shift's (its adversary range contains the whole s's: the raised "
                             "entries only narrow the range [Gamma', 1]), so ~8 or more; up <= ~1 (its range is "
                             "contained in zero's); the 20-step adversary is approximate, so containment is expected, "
                             "not guaranteed",
        "CI-masked ratio to zero": "shift 1.12 (E30); up and down each 0.95 to 1.15, no direction",
        "stochastic ratio to zero": "shift 1.09 (E30); up and down each 0.95 to 1.10, not scored",
        "sparsity ratio to zero": "shift 3.10 (E30); up ~3.1 (E30's shuffled s, the same raised values elsewhere: 3.13); "
                                  "down below 1 (lowered values are charged less)",
        "opened": "s- several times its shuffled version: E30's candidate says the real lowered entries sit on "
                  "subcomponents that are on at many Pile positions and the shuffled ones mostly on subcomponents at 0"},
    "predictions": [
        {"id": "P1", "kind": "two-sided, named outcomes (the central reading)",
         "rule": "faith_score.verdict of down / zero and of up / zero (adversarial KL, one seed each) against the same-job "
                 "W0 floor f: pass at or below 1.5 / (1 + f), fail at or above 1.5 (1 + f). Outcome 'lowered half' if "
                 "down fails and up passes; 'both halves' if both fail; 'raised half' if up fails and down passes; "
                 "'neither alone' if both pass; 'inconclusive' otherwise; 'no power' if PC does not fail. Relative to zero",
         "predicted": "'lowered half', high confidence: down's adversary range contains the whole s's (which failed 8x) "
                      "and up's is contained in zero's",
         "blind_to": "where on the Pile the damage sits; a stronger adversary; the other two E30 updates",
         "positive_control": "PC (the whole s fails again in this job)",
         "noise_floor": "the same-job W0 floor f (max/min - 1 over adversary seeds 0-2; E30 0.035 on a 4090)"},
        {"id": "P2", "kind": "directional",
         "rule": f"|R_everywhere(up) - E30's trigger-only R_halves.up| <= {P2_NEAR} and the same for down. Absolute, R units",
         "predicted": "pass, medium-high confidence: E30 found everywhere = trigger-only within 0.05 for the whole s",
         "blind_to": "the Pile", "positive_control": "R_everywhere(shift) ~ 0.748 and zero = 0",
         "noise_floor": "deterministic given the state; cross-card bf16 <= 0.01"},
        {"id": "P3", "kind": "directional",
         "rule": f"sparsity ratio up / zero (mean over 4 blocks) > {CHEAT}: 'cheating'; else 'not cheating'. Relative",
         "predicted": "'cheating', high confidence (~3.1: every raised value is charged at every Pile position)",
         "blind_to": "whether a smaller raised set keeps the trigger recovery",
         "positive_control": "shift / zero ~ 3.10 (E30)", "noise_floor": "sparsity block spread (E30 0.044)"},
        {"id": "P4", "kind": "directional (E30's candidate's premise)",
         "rule": f"opened(s-) / opened(lowered half of shuffled s) >= {P4_LINE}: 'premise holds'; else 'premise fails'. "
                 "Relative",
         "predicted": "'premise holds', medium confidence",
         "blind_to": "whether the adversary uses the opened range on those subcomponents (P1 reads that)",
         "positive_control": "opened(0) = 0 by construction", "noise_floor": "deterministic"}],
    "routes": {
        "lowered half, up not cheating": "next: the map that raises everywhere and lowers only at the trigger",
        "lowered half, up cheating": "the raised half is charged at every position too: no fixed raise-everywhere map; "
                                     "next: reading (ii), s applied only where the trigger is",
        "both halves": "the damage is not only the lowered half; next: reading (ii)",
        "raised half": "the candidate is wrong and containment is broken: check the adversary before any map",
        "neither alone": "the halves act together; next: reading (ii)",
        "inconclusive or no power": "stop and report"},
    "blind_to": "the other two E30 updates; one training seed; one adversary seed per arm; 256 Pile rows; the 67M model",
    "reported_not_scored": "CI-masked and stochastic ratios; down's sparsity ratio; down / shift (containment)",
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full_amend1.log").open("a") as f:
        f.write(line + "\n")


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True))
    tmp.replace(OUT)


def e30_update() -> dict[str, Any]:
    return json.loads(E30_JSON.read_text())["updates"][CFG.label]


def regenerate(b: SM.Bench) -> tuple[G.Task, dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    """The fine-tune, its fixed change s and E24's trigger cells, gated against E30's readings (G1)."""
    t = b.task(CFG.task)
    dW, ev = G.finetune(b.model, t, t.target, CFG.optimizer, CFG.lr, CFG.steps, CFG.seed, b.device, b.log)
    s, _ = G.shift_of(b.model, b.W0, dW, t, b.alive)
    cells, ref = SM.trigger_cells(b, t, dW, s), e30_update()["cells"]
    diff = {k: cells[k] - ref[k] for k in ("r", "rho_trig", "rho_shared")}
    b.out["G1"] = {"cells": cells, "heldout": ev, "diff": diff, "ok": all(abs(v) <= CFG.g1_tol for v in diff.values())}
    b.log(f"[G1] P {ev['p_o']:.4f} | {json.dumps(cells)} | diff {json.dumps(diff)} {'ok' if b.out['G1']['ok'] else 'FAIL'}")
    b.dump(b.out)
    return t, dW, s, cells


def opened_on_pile(b: SM.Bench, shifts: dict[str, dict[str, Tensor]]) -> dict[str, float]:
    """Mean over every Pile position of SR.opened, Gamma reading W0 (the bench's model stays at W0)."""
    from commute.vpd import compute_ci

    sums, n = dict.fromkeys(shifts, 0.0), 0
    for batch in (x for blk in b.blocks for x in blk):
        g = compute_ci(b.model, batch, chunk=batch.shape[0]).g
        for name, sh in shifts.items():
            sums[name] += float(SR.opened(g, sh).sum())
        n += batch.numel()
    return {k: v / n for k, v in sums.items()}


def stages(b: SM.Bench) -> None:
    if not SM.c1(b, CFG.w0_seeds, CFG.exact_tol):
        b.out["status"] = "stopped at C1"
        return b.log("C1 FAILED: stopping")
    t, dW, s, cells = regenerate(b)
    if not b.out["G1"]["ok"]:
        b.out["status"] = "stopped at G1"
        return b.log("G1 FAILED: stopping")
    arms = {"zero": SM.zero_like(s), "up": R.signed(s, True), "down": R.signed(s, False), "shift": s}
    b.out["R_everywhere"] = {x: SM.r_everywhere(b, t, cells, arms[x]) for x in CFG.arms}
    b.log(f"[R_everywhere] {json.dumps(b.out['R_everywhere'])}")
    shuffled_down = R.signed(R.shuffled(s, b.alive, 0), False)
    b.out["opened"] = opened_on_pile(b, {"down": arms["down"], "shuffled_down": shuffled_down})
    b.log(f"[opened] {json.dumps(b.out['opened'])}")
    b.dump(b.out)
    b.set_twin(dW)
    for x in CFG.arms:
        b.reading(f"{CFG.label}/{x}", lambda x=x: b.shifted(arms[x]))
        b.out["score"] = score(b.out)
        b.dump(b.out)
    b.out["status"] = "done"


# ----------------------------------------------------------------------------- scoring (the rules in PREDICTIONS)


def ratio(p: dict[str, Any], q: dict[str, Any]) -> dict[str, float]:
    sp = [x / y for x, y in zip(p["sparsity"], q["sparsity"], strict=True)]
    return {"pgd_kl": p["pgd_kl"] / q["pgd_kl"], "ci_masked_kl": p["ci_masked_kl"] / q["ci_masked_kl"],
            "stochastic_kl": p["stochastic_kl"] / q["stochastic_kl"], "sparsity": sum(sp) / len(sp)}


def outcome(up: str, down: str, pc: str) -> str:
    if pc != "fail":
        return "no power"
    return {("pass", "fail"): "lowered half", ("fail", "fail"): "both halves", ("fail", "pass"): "raised half",
            ("pass", "pass"): "neither alone"}.get((up, down), "inconclusive")


def score(out: dict[str, Any]) -> dict[str, Any]:
    f = out["C1"]["w0_check"]["pgd_floor"]
    sc: dict[str, Any] = {"floor_f": f, "lines": [1.5 / (1 + f), 1.5 * (1 + f)]}
    halves = e30_update()["readout"]["shift"]["R_halves"]
    if "R_everywhere" in out:
        Rx = out["R_everywhere"]
        d = {h: Rx[h] - halves[h] for h in ("up", "down")}
        sc["P2"] = {"diff_vs_E30_trigger_only": d, "pass": all(abs(v) <= P2_NEAR for v in d.values())}
    if "opened" in out:
        o = out["opened"]
        r4 = G.div(o["down"], o["shuffled_down"])
        sc["P4"] = {"ratio": r4, "reading": "premise holds" if r4 >= P4_LINE else "premise fails"}
    P = {x: out.get("pile", {}).get(f"{CFG.label}/{x}") for x in CFG.arms}
    if P["zero"] is None:
        return sc
    rat = {x: ratio(p, P["zero"]) for x, p in P.items() if p is not None}
    sc["ratios"] = rat
    if all(P[x] is not None for x in CFG.arms):
        v = {x: FS.verdict(rat[x]["pgd_kl"], f) for x in ("up", "down", "shift")}
        sc["P1"] = {"verdicts": v, "outcome": outcome(v["up"], v["down"], v["shift"]),
                    "down_over_shift": rat["down"]["pgd_kl"] / rat["shift"]["pgd_kl"]}
        sc["P3"] = {"up_sparsity": rat["up"]["sparsity"],
                    "reading": "cheating" if rat["up"]["sparsity"] > CHEAT else "not cheating"}
        o1 = sc["P1"]["outcome"]
        key = f"{o1}, up {'cheating' if sc['P3']['reading'] == 'cheating' else 'not cheating'}" if o1 == "lowered half" else o1
        sc["route"] = PREDICTIONS["routes"].get(key, PREDICTIONS["routes"]["inconclusive or no power"])
    return sc


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (halves.py --predictions)")
    t0 = time.time()
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {}
    out.update(device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", config=repr(CFG))
    b = SM.Bench(device, out, TRANSPORT, LADDER, CFG.pile_rows, CFG.sparsity_block, CFG.pgd_steps, log, dump)
    log(f"[start] {out['device']}; resuming {sorted(out.get('pile', {}))}")
    stages(b)
    out["seconds"] = time.time() - t0
    if "C1" in out:
        out["score"] = score(out)
    dump(out)
    log("score " + json.dumps(out.get("score")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions_amend1.json and exit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        PRED.write_text(json.dumps({**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        return print(f"filed {PRED}")
    run(args.device)


if __name__ == "__main__":
    main()
