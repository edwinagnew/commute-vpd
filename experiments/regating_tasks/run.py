"""regating_tasks (E28): do the working hypothesis H' and its three kill conditions hold on tasks it was never read on?
Plan: reports/2026-10-22_plan_regating-tasks.md; kill conditions from
reports/2026-10-22_decision_working-hypothesis-and-two-lines.md; definitions as E18 (reports/2026-10-21_plan_regating-leak.md,
Amendments 2 and 3). Shared steps in commute.regating (E18's logic, moved).

Do-nothing states (target W0 + dW, released subcomponents, Gamma frozen, Delta out of the masked model) of 10-step arm-A
fine-tunes of all 24 matrices, Adam 3e-5, batch 32, seed 0. Emoticon first and alone (G1, the in-job control: stop on a
fail); then four ladder tasks and three key-value tasks (E10; target = the (V,) key -> value table, trigger = the key at
`half`). Per state, on the held-out windows: regating.measure() (r, gain, gap, every pattern's rho, the random null),
the shuffled change (20 draws) and another window's values (5 derangements) with its W0 reference.

  run.py --predictions   file results/predictions.json (once, before the run)
  run.py                 -> results/regating_tasks.json (written after every state), results/large/full.log
Inputs (gitignored): experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt};
experiments/task_ladder/results/large/windows_<ladder task>.pt; experiments/kv_task/results/large/windows_<kv task>.pt
(cluster NFS only). References read from committed JSONs.
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

from commute import regating as G

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "regating_tasks.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
KV = EXP / "kv_task" / "results"
E18 = EXP / "regating_leak" / "results"
LADDER_JSON, KV_JSON = EXP / "task_ladder" / "results" / "ladder.json", KV / "kv_task.json"

STEPS, SEED, NULL_TOKEN, OPT, LR = 10, 0, 3944, "adam", 3e-5  # E18's fine-tune; ' http' as E18's different target
CONTROL = "emoticon"
LADDER_TASKS = ("r0_t352_v310", "r1_t588_v452", "r2_t275_v2667", "r0_t588_v320")
KV_TASKS = ("ident32", "rand32", "rand1")
E18_TASKS = ("emoticon", "r1_t534_v369", "r2_t359_v816")
R_TOL_G1, RHO_TOL, ZERO_TOL, ARM_TOL, R_TOL_G2 = 0.05, 0.02, 1e-6, 0.02, 0.1
MIN_GAIN, MIN_GAP, MIN_TRIG = 1.0, 0.5, 0.5
FIXED_KILL, SHUFFLED_KILL, OTHER_KILL, OTHER_REF_MAX = 0.5, 0.2, 0.5, 0.2  # the decision report's kill conditions
CROSS_MAX, RANDOM_MAX, NULL_LEARNED = 0.1, 0.1, 0.5

BLIND = ("the trigger position only; held-out windows only; one fine-tune seed; one 67M model; the clip at [0, 1] "
         "(negative values on off subcomponents vanish)")
NOISE = ("E18 seeds agree within 0.02 on R; the shuffle's sd over draws 0.02 to 0.04 (E18 amendment 2); the "
         "derangement spread a few hundredths (E18 amendment 3)")
_WORD = "medium-high (switching share 0.86 to 1.07)"
CONFIDENCE = {"r0_t352_v310": _WORD, "r1_t588_v452": _WORD, "r0_t588_v320": _WORD,
              "r2_t275_v2667": "medium (share 0.53, a rung-2 rare-target task; E18's rung-2 task r2_t359 read 0.62 "
                               "shared, 0.99 other)",
              "ident32": "low (share 0.80)",
              "rand32": "low (the masked model with released subcomponents and Delta out holds no cipher weights, so "
                        "what the switching carries is generic; a fixed change can carry generic, but the other-window "
                        "donor has a different key, so a kill there reads 'the switching carries the key')"}
CONFIDENCE["rand1"] = CONFIDENCE["rand32"]

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-22_plan_regating-tasks.md; kill conditions from "
            "reports/2026-10-22_decision_working-hypothesis-and-two-lines.md; definitions as E18. Filed before any run; "
            "never edited after.",
    "states": "do-nothing states (target W0 + dW, released subcomponents, Gamma frozen, Delta out of the masked model) of "
              "10-step arm-A fine-tunes of all 24 matrices, Adam 3e-5, batch 32, seed 0. Emoticon (G1, first, alone); "
              f"ladder {LADDER_TASKS} (target: the task's token); key-value {KV_TASKS} (target: the (V,) table at the "
              "key, trigger = the key position half, marker at half - 1). Cross null: a ' http' (3944) fine-tune by "
              "the same recipe.",
    "measurements": {
        "rho_x": "(lo_x - L00) / (L01 - L00), pattern x at the trigger and W0's gates elsewhere (commute.regating)",
        "R_mean_net": "(rho_mean - rho_mean_W0) / rho_trig: the updated model's mean trigger values over the training "
                      "windows, net of W0's mean values, as a share of each window's own values (E18 amendment 1's "
                      "mean pattern net of its reference). This is the number the decision report's '0.70 to 0.86 on "
                      "three tasks' quotes",
        "R_shared": "rho_shared / rho_trig, E18's filed shared pattern clip(g0(x) + s); reported, not scored. Note: it "
                    "read 0.73, 0.41, 0.62 on E18's three tasks, so scoring it at 0.5 would already have killed H' on "
                    "r1_t534, a task the decision report counts as support",
        "R_shuffled": f"mean over {G.N_RANDOM} draws of rho(clip(g0(x) + s_perm)) / rho_trig, s_perm = s permuted among "
                      "each matrix's alive subcomponents (recruit.shuffled, seed = draw): E18 amendment 2",
        "R_other": f"mean over {G.N_DERANGE} derangements pi of rho(g1(pi(x))) / rho_trig; reference R_other_W0 the "
                   "same for g0(pi(x)): E18 amendment 3's mismatch level",
    },
    "gates": {
        "G1_control": f"emoticon: |r - E18 seed 0 (0.5972)| <= {R_TOL_G1}; rho trig, shared, mean within {RHO_TOL} of "
                      f"E18 seed 0 (0.973, 0.715, 0.896); W0's gates through trig and shared recover <= {ZERO_TOL}; "
                      f"R_shuffled within {ARM_TOL} of E18 amendment 2 (tasks.emoticon.seeds.0.perm.shared.R_mean, "
                      f"0.0023); R_other and R_other_W0 within {ARM_TOL} of E18 amendment 3 "
                      "(tasks.emoticon.seeds.0.mismatch.R 0.887, .R_W0 0.007). First, alone; fail: stop",
        "G2_task": f"the do-nothing r within {R_TOL_G2} of the on-disk reference (ladder.json tasks.<t>.do_nothing.routes"
                   ".shares.r; kv_task.json tasks.<t>.do_nothing.routes.shares.r). Fail: reported, not scored",
    },
    "scorable": f"G2 passes, gain >= {MIN_GAIN} nat, gap L01 - L00 >= {MIN_GAP} nats, rho_trig >= {MIN_TRIG}",
    "expected": {
        "W0": "every recovery 0 exactly (s = 0, g1 = g0); R_other_W0 is the reference, not 0",
        "control (emoticon)": "r 0.60, rho_trig 0.97, R_mean_net 0.81, R_shared 0.73, R_shuffled 0.002, R_other 0.89 "
                              "(reference 0.007), rho_cross -0.18, random about 0",
        "under H'": "R_mean_net 0.7 to 0.9 (E18 0.70 to 0.86); R_shuffled within 0.03 of 0 (E18 -0.026 to +0.003); "
                    "R_other 0.85 to 1.0 (E18 0.89 to 1.00)",
        "nulls": f"random-moved mean about 0 (<= {RANDOM_MAX}); rho_cross <= {CROSS_MAX}",
    },
    "predictions": [
        *[{"id": f"K_{t}", "state": t, "kind": "ladder",
           "rule": f"three kill conditions, absolute, on the scorable state: fixed R_mean_net < {FIXED_KILL}; shuffled "
                   f"R_shuffled > {SHUFFLED_KILL}; other R_other < {OTHER_KILL}, scored only where |R_other_W0| <= "
                   f"{OTHER_REF_MAX} (else 'unscorable, reported'). Each reads 'kill' or 'survives'",
           "predicted": "survives on all three",
           "confidence": CONFIDENCE[t],
           "blind_to": BLIND, "positive_control": "G1 in the same job; rho_trig (own values recover)",
           "noise_floor": NOISE} for t in LADDER_TASKS],
        *[{"id": f"K_{t}", "state": t, "kind": "kv", "rule": "as the ladder states; two-sided",
           "predicted": "lean survives" if t == "ident32" else "two-sided, no lean",
           "confidence": CONFIDENCE[t],
           "blind_to": BLIND, "positive_control": "G1 in the same job; rho_trig", "noise_floor": NOISE}
          for t in KV_TASKS],
        {"id": "C_cross", "rule": f"rho_cross <= {CROSS_MAX} on every scorable state whose ' http' null learned (held-out "
                                  f"P >= {NULL_LEARNED}); else 'no power'. Pass: no 'fail', at least one 'pass'",
         "predicted": "pass (positive control that the test separates patterns; E18 -0.08 to -0.30)"},
        {"id": "C_random", "rule": f"random-moved null mean <= {RANDOM_MAX} on every scorable state",
         "predicted": "pass (E18 about -0.01)"},
    ],
    "pass_rule": "no kill on any scorable ladder state",
    "decision_tree": {
        "any ladder kill": "stop and tell the user; H' is restated, and the research log's Q5 with it",
        "all ladder survive, all kv survive": "H' broadened to seven tasks; next E29 and the larger model",
        "all ladder survive, a kv kill": "H' scoped to tasks the model already does; tell the user",
    },
    "reported_not_scored": "R_shared (with E18's 0.73, 0.41, 0.62); rho_mean and rho_mean_W0; |S_on| and by layer; the "
                           "shuffle's and derangements' sd, min, max and draws; held-out P of each fine-tune and its null",
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


def inputs() -> list[Path]:
    return [TRANSPORT / "windows.json", TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt",
            *[LADDER / f"windows_{t}.pt" for t in LADDER_TASKS], *[KV / "large" / f"windows_{t}.pt" for t in KV_TASKS]]


def references() -> dict[str, Any]:
    """G1: E18 seed 0 on emoticon (regating_leak.json, amend2, amend3). G2: each task's do-nothing r and gain."""
    e18, a2, a3 = (json.loads((E18 / f).read_text()) for f in
                   ("regating_leak.json", "regating_leak_amend2.json", "regating_leak_amend3.json"))
    s0 = {n: (a2["tasks"][n]["seeds"]["0"], a3["tasks"][n]["seeds"]["0"]) for n in E18_TASKS}
    g1 = {"r": e18["G1"][CONTROL]["r"], **{p: e18["tasks"][CONTROL]["seeds"]["0"]["rho"][p] for p in ("trig", "shared", "mean")},
          "R_shuffled": s0[CONTROL][0]["perm"]["shared"]["R_mean"], "R_other": s0[CONTROL][1]["mismatch"]["R"],
          "R_other_W0": s0[CONTROL][1]["mismatch"]["R_W0"]}
    context = {n: {"R_mean_net": b["R"]["mean"] - b["R"]["mean_W0"], "R_shared": a["perm"]["shared"]["R_unshuffled"],
                   "R_shuffled": a["perm"]["shared"]["R_mean"], "R_other": b["mismatch"]["R"],
                   "R_other_W0": b["mismatch"]["R_W0"]} for n, (a, b) in s0.items()}
    lad, kv = json.loads(LADDER_JSON.read_text())["tasks"], json.loads(KV_JSON.read_text())["tasks"]
    g2 = {t: {"r": src[t]["do_nothing"]["routes"]["shares"]["r"], "gain": src[t]["do_nothing"]["routes"]["gain"]}
          for src, names in ((lad, LADDER_TASKS), (kv, KV_TASKS)) for t in names}
    return {"G1": g1, "G2": g2, "E18_seed0": context}


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd

        self.device = device
        self.model = vpd.load_component_model(device)
        self.W0 = vpd.target_weights(self.model)
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}

    def task(self, name: str) -> G.Task:
        w = G.load_kv_windows(name, KV) if name in KV_TASKS else G.load_windows(name, TRANSPORT, LADDER)
        return G.prepare(self.model, name, *w)

    def finetune(self, t: G.Task, token: int | Tensor) -> tuple[dict[str, Tensor], dict[str, float]]:
        return G.finetune(self.model, t, token, OPT, LR, STEPS, SEED, self.device, log)

    def state(self, name: str) -> dict[str, Any]:
        """Fine-tune to the target and to ' http'; measure(); the shuffled change; another window's values."""
        t = self.task(name)
        dW, ev = self.finetune(t, t.target)
        dW_null, ev_null = self.finetune(t, NULL_TOKEN)
        shift_null, _ = G.shift_of(self.model, self.W0, dW_null, t, self.alive)
        m = G.measure(self.model, self.W0, t, dW, shift_null, self.alive)
        shift, _on = m.pop("shift"), m.pop("on")
        torch.save({"shift": shift, "shift_null": shift_null}, LARGE / f"{name}_shift.pt")
        rho, gap = m["rho"], m["gap"]
        m["zero_recovery"] = G.zero_recovery(self.model, t, gap)
        m["shuffled"] = G.shuffled_recovery(self.model, t, shift, self.alive, gap, rho["trig"])
        m["other"] = G.other_window_recovery(self.model, t, G.state_trigger_values(self.model, self.W0, dW, t), gap,
                                             rho["trig"])
        m.update(task=name, n_heldout=len(t.held), n_train=len(t.train), heldout=ev, null_heldout_p=ev_null["p_o"],
                 R_mean_net=G.div(rho["mean"] - rho["mean_W0"], rho["trig"]), R_shuffled=m["shuffled"]["R_mean"],
                 R_other=m["other"]["other"]["R_mean"], R_other_W0=m["other"]["other_W0"]["R_mean"])
        log(f"[{name}] P {ev['p_o']:.4f} KL {ev['kl_other']:.4f} size {ev['rel_norm_max']:.5f} | r {m['r']:.4f} gain "
            f"{m['gain']:.3f} gap {gap:.3f} | rho trig {rho['trig']:.4f} shared {rho['shared']:.4f} mean {rho['mean']:.4f} "
            f"(W0 {rho['mean_W0']:.4f}) cross {rho['cross']:.4f} | random mean {m['random_mean']:.4f} max "
            f"{m['random_max']:.4f} | |S_on| {m['n_on']} | null P(http) {ev_null['p_o']:.4f} | W0-gate recovery "
            f"{m['zero_recovery']}")
        log(f"[{name}] R_mean_net {m['R_mean_net']:.4f} R_shared {m['R_shared']:.4f} | R_shuffled {m['R_shuffled']:.4f} "
            f"sd {m['shuffled']['R_sd']:.4f} min {m['shuffled']['R_min']:.4f} max {m['shuffled']['R_max']:.4f} | "
            f"R_other {m['R_other']:.4f} sd {m['other']['other']['R_sd']:.4f} (W0 {m['R_other_W0']:.4f} sd "
            f"{m['other']['other_W0']['R_sd']:.4f})")
        log(f"[{name}] draws shuffled " + " ".join(f"{x:.4f}" for x in m["shuffled"]["R_draws"]) + " | other "
            + " ".join(f"{x:.4f}" for x in m["other"]["other"]["R_draws"]) + " | other_W0 "
            + " ".join(f"{x:.4f}" for x in m["other"]["other_W0"]["R_draws"]))
        return m


def gate_control(ref: dict[str, float], m: dict[str, Any]) -> dict[str, Any]:
    """G1: emoticon reproduces E18 seed 0, the moved arms included; W0's gates recover 0."""
    diff = {"r": m["r"] - ref["r"], **{p: m["rho"][p] - ref[p] for p in ("trig", "shared", "mean")},
            **{k: m[k] - ref[k] for k in ("R_shuffled", "R_other", "R_other_W0")}}
    ok = (abs(diff["r"]) <= R_TOL_G1 and all(abs(diff[p]) <= RHO_TOL for p in ("trig", "shared", "mean"))
          and all(abs(diff[k]) <= ARM_TOL for k in ("R_shuffled", "R_other", "R_other_W0"))
          and all(abs(v) <= ZERO_TOL for v in m["zero_recovery"].values()))
    log(f"[G1 {CONTROL}] diff from E18 {json.dumps(diff)} | {'ok' if ok else 'FAIL'}")
    return {"ref": ref, "diff": diff, "ok": ok}


def gate_task(name: str, ref: dict[str, float], m: dict[str, Any]) -> dict[str, Any]:
    diff = m["r"] - ref["r"]
    ok = abs(diff) <= R_TOL_G2
    log(f"[G2 {name}] r {m['r']:.4f} vs {ref['r']:.4f} (gain {m['gain']:.2f} vs {ref['gain']:.2f}) | {'ok' if ok else 'FAIL'}")
    return {"ref": ref, "diff_r": diff, "ok": ok}


# ----------------------------------------------------------------------------- scoring (the rules in PREDICTIONS)


def scorable(m: dict[str, Any]) -> bool:
    return m["gain"] >= MIN_GAIN and m["gap"] >= MIN_GAP and m["rho"]["trig"] >= MIN_TRIG


def kills(m: dict[str, Any], gate_ok: bool) -> dict[str, str]:
    """Each kill condition's reading on one state."""
    if not gate_ok:
        return dict.fromkeys(("fixed", "shuffled", "other"), "reported, not scored (G2)")
    if not scorable(m):
        return dict.fromkeys(("fixed", "shuffled", "other"), "unscorable")
    other = ("unscorable, reported" if abs(m["R_other_W0"]) > OTHER_REF_MAX
             else "kill" if m["R_other"] < OTHER_KILL else "survives")
    return {"fixed": "kill" if m["R_mean_net"] < FIXED_KILL else "survives",
            "shuffled": "kill" if m["R_shuffled"] > SHUFFLED_KILL else "survives", "other": other}


def control_readings(m: dict[str, Any]) -> dict[str, str]:
    cross = ("no power (the null did not learn)" if m["null_heldout_p"] < NULL_LEARNED
             else "pass" if m["rho"]["cross"] <= CROSS_MAX else "fail")
    return {"cross": cross, "random": "pass" if m["random_mean"] <= RANDOM_MAX else "fail"}


def score(out: dict[str, Any]) -> dict[str, Any]:
    st, gates = out["states"], out["gates"]
    per = {t: kills(st[t], gates[t]["ok"]) for t in (*LADDER_TASKS, *KV_TASKS) if t in st}
    ctl = {t: control_readings(m) for t, m in st.items() if (t == CONTROL or gates[t]["ok"]) and scorable(m)}
    killed = {t: any(v == "kill" for v in r.values()) for t, r in per.items()}
    lad = [t for t in LADDER_TASKS if t in killed]
    kv = [t for t in KV_TASKS if t in killed]
    branch = ("any ladder kill" if any(killed[t] for t in lad) else
              "all ladder survive, a kv kill" if any(killed[t] for t in kv) else "all ladder survive, all kv survive")
    cross = [r["cross"] for r in ctl.values()]
    scored = [t for t, r in per.items() if r["fixed"] in ("kill", "survives")]
    return {"kills": per, "pass": not any(killed[t] for t in lad), "branch": branch, "scored": scored,
            "complete": len(lad) == len(LADDER_TASKS) and len(kv) == len(KV_TASKS),
            "C_cross": {"reading": {t: r["cross"] for t, r in ctl.items()},
                        "pass": "fail" not in cross and "pass" in cross},
            "C_random": {"reading": {t: r["random"] for t, r in ctl.items()},
                         "pass": all(r["random"] == "pass" for r in ctl.values())},
            "numbers": {t: {k: st[t][k] for k in ("R_mean_net", "R_shared", "R_shuffled", "R_other", "R_other_W0")}
                        for t in st}}


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    missing = [str(p) for p in inputs() if not p.is_file()]
    if missing:
        sys.exit("missing inputs (gitignored; the kv windows are on the cluster NFS):\n  " + "\n  ".join(missing))
    t0 = time.time()
    refs = references()
    r = Run(device)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "refs": refs, "states": {}, "gates": {}}
    log(f"[start] {out['device']}; refs {json.dumps(refs)}")
    out["states"][CONTROL] = m = r.state(CONTROL)
    out["gates"][CONTROL] = gate_control(refs["G1"], m)
    dump(out)
    if not out["gates"][CONTROL]["ok"]:
        out["status"] = "stopped at G1"
        dump(out)
        log(f"G1 FAILED: stopping ({time.time() - t0:.0f} s)")
        return
    for name in (*LADDER_TASKS, *KV_TASKS):
        out["states"][name] = m = r.state(name)
        out["gates"][name] = gate_task(name, refs["G2"][name], m)
        out["score"] = score(out)
        log(f"[score {name}] {json.dumps(out['score']['kills'][name])}")
        dump(out)
    out["seconds"], out["status"] = time.time() - t0, "done"
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
