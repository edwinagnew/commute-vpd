"""regating_cuefree (E35): is the per-cue change of Gamma's importance values visible in the original values?
Plan: reports/2026-10-26_plan_regating-without-cues.md. Definitions: commute.cuefree's docstring; E28's in
reports/2026-10-22_plan_regating-tasks.md. Shared steps in commute.regating, commute.shift_anatomy, commute.cuefree.

Do-nothing states as E28 makes them (10-step arm-A Adam 3e-5 fine-tunes of all 24 matrices, seed 0; released
subcomponents, Gamma frozen, Delta out of the masked model), regenerated in-job. At the trigger only, each held-out
window is given its own original values plus a predicted change (clipped), fitted on training windows; scored as
R = rho / rho_trig. Stages, cheapest first, results written after each:
  G1     emoticon, alone: reproduces E32 (r, net fixed change, the filed shared arm) and every builder given a zero
         change recovers 0; stop on a fail
  word   r1_t588_v452 (one cue): per cue must equal shared (C_one; stop on a fail)
  kv     ident32 (copy) and rand32 (32-pair cipher), E32's states
  cues   E33's 'many' (32 cues, own answers) and 'one' (the same cues, " of"), from E33's saved windows

  run.py --predictions   file results/predictions.json (once, before the run)
  run.py                 -> results/regating_cuefree.json, results/large/full.log
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

from commute import cuefree as CF
from commute import many_cues as MC
from commute import regating as G
from commute import shift_anatomy as SA

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "regating_cuefree.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
KV, CUES = EXP / "kv_task" / "results", EXP / "many_cues" / "results"
E32_JSON, E33_JSON = EXP / "shift_anatomy" / "results" / "shift_anatomy.json", CUES / "many_cues.json"

STEPS, SEED, OPT, LR = 10, 0, "adam", 3e-5  # E28's fine-tune
CONTROL, WORD, KV_TASKS, CUE_STATES = "emoticon", "r1_t588_v452", ("ident32", "rand32"), ("many", "one")
MULTI = KV_TASKS + CUE_STATES
K_NEAR, KS, K_RAND, N_RAND, KM_SEED = 8, (8, 32, 128), 32, 5, 0
R_TOL, NET_TOL, ZERO_TOL, SAME_TOL = 0.05, 0.02, 1e-4, 1e-3
CLOSE, BEATS = 0.1, 0.2
MIN_GAIN, MIN_GAP, MIN_TRIG = 1.0, 0.5, 0.5  # E28's scorable rule

BLIND = ("the trigger position only; one fine-tune seed; one 67M model; every predicted change is a mean over training "
         "windows, so within-group variation is not modelled; Euclidean distance over every alive subcomponent weighs "
         "subcomponents irrelevant to the target equally")
NOISE = ("the states are deterministic per seed (E33 reproduced E28 within 0.002); the random-group arm (5 draws, same "
         "K) measures what averaging fewer windows does alone")


def pred(pid: str, rule: str, predicted: str, confidence: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "predicted": predicted, "confidence": confidence, "positive_control": control,
            "blind_to": BLIND, "noise_floor": NOISE}


PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-26_plan_regating-without-cues.md. Filed before any run; never edited after.",
    "states": f"{CONTROL} (G1, first, alone); {WORD} (one cue); {KV_TASKS} (E32); {CUE_STATES} (E33). The four 32-cue "
              f"states are {MULTI}",
    "measurements": {
        "R_shared": "clip(g0(x) + s), s = mean change over all training windows (E18's filed arm)",
        "R_cue": "clip(g0(x) + s_k), s_k = mean change over training windows whose trigger token is x's",
        "R_near": f"clip(g0(x) + mean change of the {K_NEAR} training windows nearest x in g0)",
        "R_clu_K": f"clip(g0(x) + mean change of x's k-means cluster of training g0), K in {KS}, seed {KM_SEED}",
        "R_rand": f"as R_clu_{K_RAND} with random labels, mean of {N_RAND} draws (the size-matched null)",
        "R_fix": "clip(a g0(x) + b), a and b per subcomponent by least squares over training windows; a subcomponent "
                 "whose g0 varies by less than 0.01 (sd) there gets a = 1, b = its mean change (cuefree.line)",
        "purity_32": f"share of training windows whose K={K_RAND} cluster's most common cue is their own; reported",
        "R_clu_32_net": f"E32/E33's net mean form with the K={K_RAND} clusters in place of cues: (rho(m1_j) - "
                        "rho(m0_j)) / rho_trig; expected within 0.1 of R_cue_net where F holds; reported",
        "R_shared_net / R_cue_net": "E32/E33's net mean forms, (rho(m1) - rho(m0)) / rho_trig over all windows or per "
                                    "cue: gates only",
        "all R": "rho / rho_trig, rho = (lo - L00) / gap at the trigger, W0's values elsewhere (commute.regating)",
    },
    "gates": {
        "G1": f"emoticon: |r - E32| <= {R_TOL}, |R_shared_net - E32| and |R_cue_net - E32| <= {NET_TOL}, R_shared "
              f"within {SAME_TOL} of rho_shared / rho_trig from regating.measure, own updated values within {ZERO_TOL} of rho_trig, and a "
              f"zero change (shift and re-weighting a = 1, b = 0) recovering within {ZERO_TOL} of 0. First, alone; "
              "fail: stop",
        "G2": f"every other state: r within {R_TOL} and R_shared_net, R_cue_net within {NET_TOL} of E32's (word, kv) or "
              "E33's (cues); fail: reported, not scored",
    },
    "scorable": f"G2 passes, gain >= {MIN_GAIN}, gap >= {MIN_GAP}, rho_trig >= {MIN_TRIG} (as E28)",
    "expected": {
        "W0 (no update)": "gap 0, every share undefined; every zero-change builder recovers exactly 0",
        f"control ({CONTROL})": "r 0.597, R_shared_net 0.815, R_cue_net 0.901 (10 trigger tokens), R_shared 0.737 "
                                "(E32); the cue-free arms reported",
        f"one cue ({WORD})": "r 0.904, R_shared_net 0.916, R_shared 0.708 (E32); R_cue = R_shared",
        "32-cue states": "R_shared 0.01 to 0.16 (E32, E33 filed arms); R_cue between R_shared and R_cue_net",
    },
    "predictions": [
        pred("C_one", f"{WORD}: |R_cue - R_shared| <= {SAME_TOL}; fail: stop", "pass", "high",
             "itself (one cue: per cue is the shared change)"),
        pred("C", f"{WORD}: |R_near - R_shared| <= {CLOSE} and |R_clu_8 - R_shared| <= {CLOSE}", "pass",
             "medium", "C_one"),
        pred("N", f"every scorable state: |R_rand - R_shared| <= {CLOSE}; a 32-cue state failing it is unreadable for K",
             "pass", "medium", "C_one"),
        pred("F", f"every scorable 32-cue state: R_near >= R_cue - {CLOSE}", "pass", "medium", "C"),
        pred("K", f"every scorable 32-cue state readable by N: R_clu_{K_RAND} >= R_cue - {CLOSE}", "pass", "medium-low",
             "N"),
        pred("A", f"every scorable 32-cue state: R_fix >= R_cue - {CLOSE} and R_fix >= R_shared + {BEATS}", "pass", "low",
             "C"),
    ],
    "decision_tree": {
        "G1 or C_one fails": "stop; nothing scored",
        "F and K pass": "the change is a function of the original values: E36 tests the nearest and cluster arms at "
                        "every position of a fine-tune with no trigger",
        "A passes too": "one fixed rule per subcomponent: E36's headline test",
        "F fails, R_cue holds": "the cue is not visible in Gamma's values at the trigger: card for the user (next space: "
                                "Gamma's input); E36 leads with the per-token arm",
        "R_cue < 0.5 on the 32-cue states": "the shift form loses what the mean form had: E36 leads with R_fix",
    },
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
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def inputs() -> list[Path]:
    return [E32_JSON, E33_JSON, TRANSPORT / "windows.json", TRANSPORT / "large" / "alive.pt",
            TRANSPORT / "large" / "windows.pt", LADDER / f"windows_{WORD}.pt",
            *[KV / "large" / f"windows_{t}.pt" for t in KV_TASKS], CUES / "large" / "windows.pt"]


def windows_of(name: str) -> tuple[Any, Any, int | Tensor]:
    if name in KV_TASKS:
        return G.load_kv_windows(name, KV)
    if name in CUE_STATES:
        w = torch.load(CUES / "large" / "windows.pt", weights_only=False)
        mine = w["mine"]
        pairs = {c["trigger"]: c["alt"] for c in mine["cues"]}
        return w["train"], w["heldout"], MC.table(pairs, mine["vocab"]) if name == "many" else mine["target"]
    return G.load_windows(name, TRANSPORT, LADDER)


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd

        self.model = vpd.load_component_model(device)
        self.W0, self.device = vpd.target_weights(self.model), device
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}

    def reads(self, t: G.Task, dW: dict[str, Tensor]) -> dict[str, SA.Vals]:
        """Gamma's values at the trigger, per window: training and held-out, at W0 (0) and in the state (1)."""
        r = {"tr0": SA.trigger_read(self.model, t.train.tokens, t.half)["g"],
             "he0": SA.trigger_read(self.model, t.held.tokens, t.half)["g"]}
        with G.state_of(self.model, self.W0, dW):
            r["tr1"] = SA.trigger_read(self.model, t.train.tokens, t.half)["g"]
            r["he1"] = SA.trigger_read(self.model, t.held.tokens, t.half)["g"]
        return r

    def state(self, name: str) -> dict[str, Any]:
        model = self.model
        t = G.prepare(model, name, *windows_of(name))
        dW, ev = G.finetune(model, t, t.target, OPT, LR, STEPS, SEED, self.device, log)
        m = G.measure(model, self.W0, t, dW, {k: torch.zeros(v.shape) for k, v in t.m0.items()}, self.alive, n_random=1)
        gap, rho = m["gap"], m["rho"]
        out = {k: m[k] for k in ("r", "gain", "gap", "rho", "cells")}
        out.update(heldout=ev, n_train=len(t.train), n_heldout=len(t.held),
                   R_shared_net=G.div(rho["mean"] - rho["mean_W0"], rho["trig"]),
                   R_shared_filed=G.div(rho["shared"], rho["trig"]))
        rd = self.reads(t, dW)
        cue_tr, cue_he = t.train.tokens[:, t.half], t.held.tokens[:, t.half]
        out["R_cue_net"] = SA.group_fixed(model, t, rd["tr1"], rd["tr0"], cue_tr, cue_he, gap, rho["trig"])["R_net"]
        out.update(arms(model, t, rd, cue_tr, cue_he, gap, rho["trig"], self.alive, self.device))
        log(f"[{name}] r {out['r']:.4f} gain {out['gain']:.3f} gap {gap:.3f} rho_trig {rho['trig']:.4f} held-out P "
            f"{ev['p_o']:.3f} | R_shared_net {out['R_shared_net']:.4f} R_cue_net {out['R_cue_net']:.4f} | "
            + " ".join(f"{k} {out[k]:.4f}" for k in SCORED_KEYS) + f" | checks {json.dumps(out['checks'])}")
        return out


SCORED_KEYS = ("R_shared", "R_cue", "R_near", *[f"R_clu_{K}" for K in KS], "R_rand", "R_fix", "purity_32",
               "R_clu_32_net")


def arms(model: Any, t: G.Task, rd: dict[str, SA.Vals], cue_tr: Tensor, cue_he: Tensor, gap: float, rho_trig: float,
         alive: dict[str, Tensor], device: str) -> dict[str, Any]:
    """Every predicted pattern of the plan, as R, plus the zero-change and own-values checks (as rho)."""
    tr0, tr1, he0, he1 = rd["tr0"], rd["tr1"], rd["he0"], rd["he1"]
    n_tr, n_he = next(iter(tr0.values())).shape[0], next(iter(he0.values())).shape[0]
    d_tr = CF.change(tr1, tr0, alive)

    def R(vals: SA.Vals) -> float:
        return G.div(SA.rho_of(model, t, vals, gap), rho_trig)

    def grouped(lab_tr: Tensor, lab_he: Tensor) -> float:
        return R(CF.apply(he0, CF.by_group(d_tr, lab_tr, lab_he)))

    out: dict[str, Any] = {"R_shared": R(CF.apply(he0, CF.broadcast(SA.mean_of(d_tr), n_he))),
                           "R_cue": grouped(cue_tr, cue_he),
                           "R_near": R(CF.apply(he0, CF.nearest(tr0, he0, d_tr, alive, K_NEAR, device)))}
    n_clusters = {}
    for K in KS:
        lab_tr, lab_he = CF.clusters(tr0, he0, alive, K, KM_SEED, device)
        out[f"R_clu_{K}"], n_clusters[K] = grouped(lab_tr, lab_he), int(lab_tr.unique().numel())
        if K == K_RAND:
            out["purity_32"] = CF.purity(lab_tr, cue_tr)
            out["R_clu_32_net"] = SA.group_fixed(model, t, tr1, tr0, lab_tr, lab_he, gap, rho_trig)["R_net"]
    draws = [grouped(SA.random_labels(n_tr, K_RAND, i), SA.random_labels(n_he, K_RAND, 100 + i)) for i in range(N_RAND)]
    a, b = CF.refit(tr0, tr1, alive)
    zero = {k: torch.zeros(v.shape[1]) for k, v in he0.items()}
    one = {k: torch.ones(v.shape[1]) for k, v in he0.items()}
    out.update(R_rand=sum(draws) / len(draws), rand_draws=draws, R_fix=R(CF.reweighted(he0, a, b)),
               n_clusters=n_clusters,
               n_cues={"train": int(cue_tr.unique().numel()), "heldout": int(cue_he.unique().numel())},
               refit={"a_median": float(torch.cat([a[k][alive[k]] for k in a]).median()),
                      "a_gt2": float(torch.cat([(a[k][alive[k]] > 2).float() for k in a]).mean())},
               checks={"rho_zero_shift": SA.rho_of(model, t, CF.apply(he0, CF.broadcast(zero, n_he)), gap),
                       "rho_zero_refit": SA.rho_of(model, t, CF.reweighted(he0, one, zero), gap),
                       "own_vs_trig": SA.rho_of(model, t, he1, gap) - rho_trig})
    return out


# ----------------------------------------------------------------------------- gates and scoring


def reference(name: str, e32: dict[str, Any], e33: dict[str, Any]) -> dict[str, float]:
    if name in CUE_STATES:
        s = e33[name]
        return {"r": s["r"], "R_shared_net": s["R_shared_net"], "R_cue_net": s["R_cue_net"]}
    s = e32[name]
    return {"r": s["r"], "R_shared_net": s["R_mean_net"], "R_cue_net": s["R_key_net"]}


def gate(name: str, ref: dict[str, float], m: dict[str, Any]) -> dict[str, Any]:
    diff = {k: m[k] - ref[k] for k in ("r", "R_shared_net", "R_cue_net")}
    ok = abs(diff["r"]) <= R_TOL and abs(diff["R_shared_net"]) <= NET_TOL and abs(diff["R_cue_net"]) <= NET_TOL
    if name == CONTROL:
        builder = m["R_shared"] - m["R_shared_filed"]
        ok = ok and abs(builder) <= SAME_TOL and all(abs(v) <= ZERO_TOL for v in m["checks"].values())
        diff["R_shared_vs_filed"] = builder
    log(f"[{'G1' if name == CONTROL else 'G2'} {name}] diff {json.dumps(diff)} checks {json.dumps(m['checks'])} | "
        f"{'ok' if ok else 'FAIL'}")
    return {"diff": diff, "ok": ok}


def scorable(m: dict[str, Any], ok: bool) -> bool:
    return ok and m["gain"] >= MIN_GAIN and m["gap"] >= MIN_GAP and m["rho"]["trig"] >= MIN_TRIG


def score(out: dict[str, Any]) -> dict[str, Any]:
    st, gates = out["states"], out["gates"]
    use = {n: m for n, m in st.items() if scorable(m, gates[n]["ok"])}
    near = lambda m, k: abs(m[k] - m["R_shared"]) <= CLOSE
    readable = {n: near(m, "R_rand") for n, m in use.items()}
    multi = {n: m for n, m in use.items() if n in MULTI}
    per = {n: {"N": readable[n], "F": m["R_near"] >= m["R_cue"] - CLOSE,
               "K": (m[f"R_clu_{K_RAND}"] >= m["R_cue"] - CLOSE) if readable[n] else None,
               "A": m["R_fix"] >= m["R_cue"] - CLOSE and m["R_fix"] >= m["R_shared"] + BEATS} for n, m in multi.items()}
    one = [st["word"]] if "word" in use else []
    return {"scorable": sorted(use), "per_state": per,
            "C_one": abs(st["word"]["R_cue"] - st["word"]["R_shared"]) <= SAME_TOL if "word" in st else None,
            "C": all(near(m, "R_near") and near(m, "R_clu_8") for m in one) if one else None,
            "N": all(readable.values()) if readable else None,
            **{p: (all(v[p] for v in per.values() if v[p] is not None) if any(v[p] is not None for v in per.values())
                   else None) for p in ("F", "K", "A")},
            "complete": all(n in multi for n in MULTI)}


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    missing = [str(p) for p in inputs() if not p.is_file()]
    if missing:
        sys.exit("missing inputs (gitignored):\n  " + "\n  ".join(missing))
    t0 = time.time()
    e32, e33 = (json.loads(p.read_text())["states"] for p in (E32_JSON, E33_JSON))
    r = Run(device)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "states": {}, "gates": {}, "status": "running"}
    log(f"[start] {out['device']}")
    for name in (CONTROL, WORD, *KV_TASKS, *CUE_STATES):
        key = "word" if name == WORD else name
        out["states"][key] = m = r.state(name)
        out["gates"][key] = gate(name, reference(name, e32, e33), m)
        out["score"] = score(out)
        dump(out)
        if name == CONTROL and not out["gates"][key]["ok"]:
            out["status"] = "stopped at G1"
            dump(out)
            return log(f"stopped at G1 ({time.time() - t0:.0f} s)")
        if name == WORD and not out["score"]["C_one"]:
            out["status"] = "stopped at C_one"
            dump(out)
            return log(f"stopped at C_one ({time.time() - t0:.0f} s)")
    out["seconds"], out["status"] = time.time() - t0, "done"
    dump(out)
    log("score " + json.dumps(_finite(out["score"])))
    log(f"FINISHED wrote {OUT} and {LARGE / 'full.log'} ({out['seconds']:.0f} s)")


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
