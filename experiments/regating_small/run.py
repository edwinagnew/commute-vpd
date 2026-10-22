"""regating_small (E24): does a fixed pattern of importance values recover the switching effect of a SMALL update?
Plan: reports/2026-10-22_plan_small-update-reruns.md -> E24 (and its Agent notes); E18's definitions in
reports/2026-10-21_plan_regating-leak.md. Shared steps in commute.regating (E18's logic, moved).

Do-nothing states of 10-step arm-A fine-tunes on emoticon (seed 0): the large Adam 3e-5 one (E18's state, the gate and
the in-job reference) and the two small ones of E21 (SGD 0.01; Adam 3e-6, matched in size). Per state, on the held-out
windows, r and the recovery of the trig, shared, mean, cross and random patterns (regating.py's docstring); the cross
pattern is the change of a ' http' fine-tune by the same optimiser and rate. Then, if an SGD rate passes E4's rule at
step 10 on r1_t534_v369, the same for that task's SGD state beside its large Adam 3e-5 state. Last, J_nat (reported).

  run.py --predictions   file results/predictions.json (once, before the run)
  run.py                 -> results/regating_small.json (written after every state), results/large/full.log
Inputs (gitignored): experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
experiments/task_ladder/results/large/windows_r1_t534_v369.pt. References read from committed JSONs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import arms
from commute import recruit as R
from commute import regating as G

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "regating_small.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
E18_JSON = EXP / "regating_leak" / "results" / "regating_leak.json"
E21 = EXP / "sgd_finetune" / "results"

STEPS, SEED, NULL_TOKEN = 10, 0, 3944  # E18 / E21's step; ' http' as E18's different target
EXTRA = "r1_t534_v369"
BIG = ("adam", 3e-5)  # E14/E16/E18's fine-tune
SMALL = {"sgd_0.01": ("sgd", 0.01), "adam_3e-06": ("adam", 3e-6)}  # E21's matched pair
SGD_LRS = (0.001, 0.003, 0.01, 0.03, 0.1)  # sgd_finetune's grid
P_MIN, KL_LIMIT_EXTRA = 0.9, 0.24302  # E4's rule; r1_t534's limit as sgd_ladder used it (2 x Adam 3e-5's KL at step 10)
R_TOL = {"emoticon": 0.05, EXTRA: 0.1}  # E18's G1 tolerance on r
RHO_TOL, ZERO_TOL = 0.02, 1e-6  # E18 amendment 1's repeat tolerance on recoveries; W0's gates recover 0
P_TOL, R_SMALL_TOL = 0.01, 0.05  # small states reproduce E21's held-out P and do-nothing r
MIN_GAIN, MIN_GAP, MIN_TRIG = 1.0, 0.5, 0.5
SAME_R, FIXED, PER_WINDOW, CROSS_MAX, RANDOM_MAX, NULL_LEARNED = 0.1, 0.7, 0.3, 0.1, 0.1, 0.5
NAT_ROWS = ("val", 0, 600)

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-22_plan_small-update-reruns.md -> E24 and its Agent notes; definitions as E18 "
            "(reports/2026-10-21_plan_regating-leak.md). Filed before any run; never edited after.",
    "states": "do-nothing states (target W0 + dW, released subcomponents C0, Gamma frozen, Delta out of the masked model) of "
              "10-step arm-A fine-tunes of all 24 matrices, seed 0, batch 32: emoticon with Adam 3e-5 (large, E18's state), "
              "SGD 0.01 and Adam 3e-6 (small, E21's matched pair, size 0.0024 and 0.0022). Extra: r1_t534_v369 with the "
              f"SGD rate E4's rule picks from {SGD_LRS} (P >= {P_MIN}, KL elsewhere <= {KL_LIMIT_EXTRA}), beside its Adam 3e-5 "
              "state; skipped, with the grid recorded, if no rate passes",
    "measurements": {
        "r": "(L01 - L00) / (T1 - T0), held-out windows' trigger log-odds of the target; L = CI-masked model with C0",
        "recovery rho_x": "(lo_x - L00) / (L01 - L00) with pattern x at the trigger and W0's gates elsewhere (commute.regating)",
        "trig": "each window's own state gates", "shared": "clip(g0(x) + s), s = mean over the TRAINING windows of state "
        "minus W0 trigger gates, alive only", "mean": "m1 for every window (mean_W0: m0, reported)",
        "cross": "clip(g0(x) + s_null), s_null from a ' http' (3944) fine-tune by the SAME optimiser and rate",
        "random": f"s's values on S_on (s > {G.THR}) moved onto a random same-size-per-matrix set outside S_on, {G.N_RANDOM} draws",
        "R_shared": "rho_shared / rho_trig",
    },
    "gates": {
        "G1_large_reproduces": f"emoticon Adam 3e-5: |r - E18 seed 0 (0.5972)| <= {R_TOL['emoticon']}; rho trig, shared, mean "
                               f"within {RHO_TOL} of E18 seed 0 (0.973, 0.715, 0.896); gain >= {MIN_GAIN}; W0's gates through "
                               f"trig and shared recover <= {ZERO_TOL}. Runs first, alone; fail: stop, nothing else run",
        "G2_small_reproduce": f"held-out P(o) within {P_TOL} of E21 (SGD 0.973, Adam 3e-6 0.971) and r within {R_SMALL_TOL} of "
                              "sgd_finetune's do-nothing route split (0.617, 0.594). Fail: that state is reported, not scored",
        "G3_extra_large": f"r1_t534 Adam 3e-5: r within {R_TOL[EXTRA]} of E18 (0.7457), rho within {RHO_TOL}. Fail: the "
                          "extra task's SGD state is not run",
    },
    "scorable": f"gain >= {MIN_GAIN} nat, gap L01 - L00 >= {MIN_GAP} nats (as E18), and rho_trig >= {MIN_TRIG} for P2",
    "expected": {
        "W0": "recovery of W0's own gates 0 exactly; r undefined",
        "large (control, in-job)": "r 0.60, rho_trig 0.97, R_shared 0.73, rho_mean 0.90 (mean_W0 0.10), rho_cross -0.18, "
                                   "random mean -0.01, |S_on| 210 (E18 seed 0)",
        "small r": "about 0.6 (sgd_finetune's route split already read 0.617 and 0.594: P1 is close to a reproduction)",
        "small R_shared": "two-sided (P2)", "cross": "<= 0.1 (E18: -0.08 to -0.30)", "random": "about 0",
        "|S_on|": "fewer than the large state's 210 (a smaller update moves the gates less); low confidence",
    },
    "predictions": [
        {"id": "P1_r", "rule": f"per small emoticon state: |r_small - r_large(in-job)| <= {SAME_R} reads 'same', below "
                               "'lower', above 'higher'. Pass: 'same' on both. Absolute difference of shares",
         "predicted": "same on both, high confidence (recorded 0.617 and 0.594 against 0.597; E21: the SGD state's trigger "
                      "share with Gamma on the original activations was 0.01, so the switching carries the small update too)",
         "blind_to": "where the switching happens (the patterns read that); the Pile", "positive_control": "the large state, G1",
         "noise_floor": "E18 seeds 0 and 1 differ by 0.012 in r"},
        {"id": "P2_shared", "rule": f"per small emoticon state: R_shared >= {FIXED} with random mean <= {RANDOM_MAX} reads "
                                    f"'fixed pattern' (H: one fixed change carries it); <= {PER_WINDOW} reads 'per-window' (small "
                                    "updates need each window's own values); 'between' otherwise. Pass: 'fixed pattern' on both. "
                                    "Absolute, in shares of rho_trig; R_shared minus the large state's reported",
         "predicted": "two-sided as the plan asks; lean 'fixed pattern' on both, medium-low confidence (emoticon windows "
                      "are alike; the large state read 0.73, just over the bar, so a smaller shift could land 'between')",
         "blind_to": "the trigger position only; whether the additive model g0(x) + s is the right fixed pattern (E18 "
                     "amendment 1: W0's per-window variation lowers it; rho_mean is reported for that); one seed",
         "positive_control": "rho_trig (the own values recover); the large state's R_shared 0.73 in the same job",
         "noise_floor": "E18 seeds agree within 0.02 on R_shared; the random null's spread about 0.01"},
        {"id": "P2x_extra", "rule": "r1_t534's SGD state (if run) reads as P2; pass: the same reading as its large Adam "
                                    "state in the same job",
         "predicted": "same as the large ('between', E18 0.41), low confidence", "blind_to": "as P2",
         "positive_control": "the task's large state, G3", "noise_floor": "as P2"},
        {"id": "C_cross", "rule": f"rho_cross <= {CROSS_MAX} on every scorable state (large and small, both tasks); a state "
                                  f"whose ' http' null has held-out P < {NULL_LEARNED} reads 'no power'. Pass: no 'fail' and at "
                                  "least one 'pass'",
         "predicted": "pass (positive control that the test separates patterns; E18 read -0.08 to -0.30)",
         "blind_to": "other wrong targets", "positive_control": "is itself the control",
         "noise_floor": "E18 seeds agree within 0.01 on rho_cross"},
    ],
    "reported_not_scored": "rho_mean and rho_mean_W0; |S_on| and by layer; J(S_on small, S_on large); J_nat on Pile "
                           f"{NAT_ROWS} with its random null; the null's held-out P(' http'); the r1_t534 rate grid",
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
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def references() -> dict[str, Any]:
    """E18's seed-0 readings (the gate) and E21's small-state readings (their reproduction)."""
    e18 = json.loads(E18_JSON.read_text())
    big = {n: {"r": e18["G1"][n]["r"], **{p: e18["tasks"][n]["seeds"]["0"]["rho"][p] for p in ("trig", "shared", "mean", "cross")}}
           for n in ("emoticon", EXTRA)}
    rate, matched = json.loads((E21 / "rate.json").read_text()), json.loads((E21 / "adam_matched.json").read_text())
    split = json.loads((E21 / "split.json").read_text())
    small = {"sgd_0.01": {"p_o": rate["finetune"]["0.01"]["p_o"], "r": split["states"]["sgd"]["nothing"]["routes"]["shares"]["r"]},
             "adam_3e-06": {"p_o": matched["grid"]["3e-06"]["p_o"], "r": matched["states"]["nothing"]["routes"]["shares"]["r"]}}
    return {"big": big, "small": small}


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd

        self.device = device
        self.model = vpd.load_component_model(device)
        self.W0 = vpd.target_weights(self.model)
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}
        self.tasks: dict[str, G.Task] = {}
        self.on: dict[str, R.Sel] = {}  # label -> S_on

    def task(self, name: str) -> G.Task:
        if name not in self.tasks:
            self.tasks[name] = G.prepare(self.model, name, *G.load_windows(name, TRANSPORT, LADDER))
        return self.tasks[name]

    def finetune(self, name: str, token: int, opt: str, lr: float) -> tuple[dict[str, Tensor], dict[str, float]]:
        return G.finetune(self.model, self.task(name), token, opt, lr, STEPS, SEED, self.device, log)

    def state(self, name: str, label: str, opt: str, lr: float) -> dict[str, Any]:
        """Fine-tune to the target and to ' http' by the same optimiser and rate; every pattern's recovery."""
        t = self.task(name)
        dW, ev = self.finetune(name, t.target, opt, lr)
        dW_null, ev_null = self.finetune(name, NULL_TOKEN, opt, lr)
        shift_null, _ = G.shift_of(self.model, self.W0, dW_null, t, self.alive)
        m = G.measure(self.model, self.W0, t, dW, shift_null, self.alive)
        shift, on = m.pop("shift"), m.pop("on")
        torch.save({"shift": shift, "shift_null": shift_null}, LARGE / f"{name}_{label}_shift.pt")
        self.on[f"{name}/{label}"] = on
        m.update(task=name, optimizer=opt, lr=lr, heldout=ev, null_heldout_p=ev_null["p_o"],
                 null_n_on=R.size(R.switched(shift_null, G.THR)), scorable=m["gain"] >= MIN_GAIN and m["gap"] >= MIN_GAP)
        rho = m["rho"]
        log(f"[{name} {label}] P {ev['p_o']:.4f} KL {ev['kl_other']:.4f} size {ev['rel_norm_max']:.5f} | r {m['r']:.4f} gain "
            f"{m['gain']:.3f} gap {m['gap']:.3f} | rho trig {rho['trig']:.4f} shared {rho['shared']:.4f} mean {rho['mean']:.4f} "
            f"(W0 {rho['mean_W0']:.4f}) cross {rho['cross']:.4f} | random mean {m['random_mean']:.4f} max "
            f"{m['random_max']:.4f} | R_shared {m['R_shared']:.4f} | |S_on| {m['n_on']} | null P(http) {ev_null['p_o']:.4f}")
        return m


def gate_big(run: Run, name: str, label: str, ref: dict[str, float], m: dict[str, Any]) -> dict[str, Any]:
    """The large state reproduces E18's seed-0 readings; W0's gates recover 0."""
    zero = G.zero_recovery(run.model, run.task(name), m["gap"])
    diff = {"r": m["r"] - ref["r"], **{p: m["rho"][p] - ref[p] for p in ("trig", "shared", "mean")}}
    ok = (abs(diff["r"]) <= R_TOL[name] and all(abs(diff[p]) <= RHO_TOL for p in ("trig", "shared", "mean"))
          and m["gain"] >= MIN_GAIN and all(abs(v) <= ZERO_TOL for v in zero.values()))
    log(f"[gate {name} {label}] diff from E18 {json.dumps(diff)} | W0-gate recovery {zero} | {'ok' if ok else 'FAIL'}")
    return {"ref": ref, "diff": diff, "zero_recovery": zero, "ok": ok}


def gate_small(label: str, ref: dict[str, float], m: dict[str, Any]) -> dict[str, Any]:
    diff = {"p_o": m["heldout"]["p_o"] - ref["p_o"], "r": m["r"] - ref["r"]}
    ok = abs(diff["p_o"]) <= P_TOL and abs(diff["r"]) <= R_SMALL_TOL
    log(f"[gate {label}] diff from E21 {json.dumps(diff)} | {'ok' if ok else 'FAIL'}")
    return {"ref": ref, "diff": diff, "ok": ok}


def extra_task(run: Run, out: dict[str, Any], refs: dict[str, Any]) -> None:
    """r1_t534: E4's rule over the SGD grid at step 10; if a rate passes, its state beside the large Adam state."""
    t = run.task(EXTRA)
    grid = {f"{lr:g}": run.finetune(EXTRA, t.target, "sgd", lr)[1] for lr in SGD_LRS}
    lr, passed = arms.select_rate(grid, P_MIN, KL_LIMIT_EXTRA)
    rec: dict[str, Any] = {"grid": grid, "limit": KL_LIMIT_EXTRA, "selected": lr, "passed": passed, "states": {}}
    out["extra"] = rec
    log(f"[{EXTRA} grid] {json.dumps({k: (round(v['p_o'], 4), round(v['kl_other'], 4)) for k, v in grid.items()})} -> "
        f"{lr} passed {passed}")
    dump(out)
    if not passed:
        return
    rec["states"]["adam_3e-05"] = big = run.state(EXTRA, "adam_3e-05", *BIG)
    rec["gate"] = gate_big(run, EXTRA, "adam_3e-05", refs["big"][EXTRA], big)
    dump(out)
    if not rec["gate"]["ok"]:
        log(f"[{EXTRA}] the large state does not reproduce E18: the task's SGD state is not run")
        return
    rec["states"][f"sgd_{lr}"] = run.state(EXTRA, f"sgd_{lr}", "sgd", float(lr))
    dump(out)


def natural(run: Run, out: dict[str, Any]) -> None:
    """Reported: J_nat of each state's S_on, and S_on's overlap between each small state and its task's large one."""
    from commute.data import pile_rows

    targets = {n: t.target for n, t in run.tasks.items()}
    con = G.natural_contrast(run.model, pile_rows(*NAT_ROWS, log=log), targets, run.alive)
    states = {**{f"emoticon/{k}": v for k, v in out["states"].items()},
              **{f"{EXTRA}/{k}": v for k, v in out.get("extra", {}).get("states", {}).items()}}
    nat: dict[str, Any] = {}
    for key, m in states.items():
        c = con[m["task"]]
        nat[key] = {"n_positions": c["n_positions"], **G.j_nat(run.on[key], c["contrast"], run.alive),
                    "J_with_large": R.jaccard(run.on[key], run.on[f"{m['task']}/adam_3e-05"])}
        log(f"[natural {key}] {json.dumps(nat[key])}")
    out["natural"] = nat
    dump(out)


# ----------------------------------------------------------------------------- scoring (the rules in PREDICTIONS)


def reading_shared(m: dict[str, Any]) -> str:
    if not m["scorable"] or m["rho"]["trig"] < MIN_TRIG:
        return "unscorable"
    if m["R_shared"] >= FIXED and m["random_mean"] <= RANDOM_MAX:
        return "fixed pattern"
    return "per-window" if m["R_shared"] <= PER_WINDOW else "between"


def reading_cross(m: dict[str, Any]) -> str:
    if not m["scorable"]:
        return "unscorable"
    if m["null_heldout_p"] < NULL_LEARNED:
        return "no power (the null did not learn)"
    return "pass" if m["rho"]["cross"] <= CROSS_MAX else "fail"


def score(out: dict[str, Any]) -> dict[str, Any]:
    st = out["states"]
    big = st["adam_3e-05"]
    small = {k: v for k, v in st.items() if k in SMALL and out["gates"].get(k, {}).get("ok")}
    sc: dict[str, Any] = {"G1": out["gates"]["adam_3e-05"]["ok"], "G2": {k: out["gates"][k]["ok"] for k in SMALL if k in out["gates"]}}
    dr = {k: v["r"] - big["r"] for k, v in small.items()}
    p1 = {k: "same" if abs(d) <= SAME_R else "lower" if d < 0 else "higher" for k, d in dr.items()}
    sc["P1_r"] = {"r": {k: v["r"] for k, v in small.items()}, "r_large": big["r"], "diff": dr, "reading": p1,
                  "pass": all(v == "same" for v in p1.values()) if len(p1) == len(SMALL) else "unscored: a small state failed G2"}
    p2 = {k: reading_shared(v) for k, v in small.items()}
    sc["P2_shared"] = {"R_shared": {k: v["R_shared"] for k, v in small.items()}, "R_shared_large": big["R_shared"],
                       "minus_large": {k: v["R_shared"] - big["R_shared"] for k, v in small.items()},
                       "random_mean": {k: v["random_mean"] for k, v in small.items()}, "reading": p2,
                       "pass": all(v == "fixed pattern" for v in p2.values()) if len(p2) == len(SMALL)
                       else "unscored: a small state failed G2"}
    allst = {f"emoticon/{k}": v for k, v in st.items() if k == "adam_3e-05" or k in small}
    ex = out.get("extra", {})
    if ex.get("gate", {}).get("ok"):
        allst.update({f"{EXTRA}/{k}": v for k, v in ex["states"].items()})
        xs = {k: v for k, v in ex["states"].items() if k != "adam_3e-05"}
        sc["P2x_extra"] = {"R_shared": {k: v["R_shared"] for k, v in xs.items()},
                           "R_shared_large": ex["states"]["adam_3e-05"]["R_shared"],
                           "reading": {k: reading_shared(v) for k, v in xs.items()},
                           "reading_large": reading_shared(ex["states"]["adam_3e-05"])}
        sc["P2x_extra"]["pass"] = all(r == sc["P2x_extra"]["reading_large"] for r in sc["P2x_extra"]["reading"].values())
    c = {k: reading_cross(v) for k, v in allst.items()}
    sc["C_cross"] = {"rho_cross": {k: v["rho"]["cross"] for k, v in allst.items()}, "reading": c,
                     "pass": all(v != "fail" for v in c.values()) and any(v == "pass" for v in c.values())}
    return sc


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    t0 = time.time()
    refs = references()
    r = Run(device)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "refs": refs, "states": {}, "gates": {}}
    log(f"[start] {out['device']}; refs {json.dumps(refs)}")
    out["states"]["adam_3e-05"] = m = r.state("emoticon", "adam_3e-05", *BIG)
    out["gates"]["adam_3e-05"] = gate_big(r, "emoticon", "adam_3e-05", refs["big"]["emoticon"], m)
    dump(out)
    if not out["gates"]["adam_3e-05"]["ok"]:
        out["status"] = "stopped at G1"
        dump(out)
        log(f"G1 FAILED: stopping ({time.time() - t0:.0f} s)")
        return
    for label, (opt, lr) in SMALL.items():
        out["states"][label] = m = r.state("emoticon", label, opt, lr)
        out["gates"][label] = gate_small(label, refs["small"][label], m)
        out["score"] = score(out)
        dump(out)
    extra_task(r, out, refs)
    out["score"] = score(out)
    dump(out)
    natural(r, out)
    out["seconds"] = time.time() - t0
    out["status"] = "done"
    out["score"] = score(out)
    dump(out)
    log("score " + json.dumps(_finite(out["score"])))
    log(f"wrote {OUT} ({out['seconds']:.0f} s)")


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
        print(f"filed {PRED}")
        return
    run(args.device)


if __name__ == "__main__":
    main()
