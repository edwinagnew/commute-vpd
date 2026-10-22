"""shift_anatomy (E32): what is the fixed change of Gamma's importance values made of, on E28's eight states?
Plan: reports/2026-10-22_plan_shift-anatomy.md. Definitions: commute.shift_anatomy's docstring; E28's in
reports/2026-10-22_plan_regating-tasks.md. Shared steps in commute.regating (E28's states) and commute.shift_anatomy.

E28's do-nothing states, regenerated exactly as E28 does (10-step arm-A Adam 3e-5 fine-tunes of all 24 matrices, seed 0;
released subcomponents, Gamma frozen, Delta out of the masked model). Emoticon first and alone: it must reproduce E28's
r and net fixed change (G1, stop on a fail). Then four word tasks and three key-value tasks; per state the arms
group (per key and random groups), strength, mixed (attention vs MLP), steer; last, the Pile stage on the 32-key tasks.

  run.py --predictions   file results/predictions.json (once, before the run)
  run.py                 -> results/shift_anatomy.json (written after every state), results/large/full.log
Inputs (gitignored): as E28 (experiments/regating_tasks/run.py `inputs`); E28's committed regating_tasks.json.
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
from commute import shift_anatomy as SA

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "shift_anatomy.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
KV, E28_JSON = EXP / "kv_task" / "results", EXP / "regating_tasks" / "results" / "regating_tasks.json"

STEPS, SEED, OPT, LR = 10, 0, "adam", 3e-5  # E28's fine-tune
CONTROL, WORD, KV_TASKS, MULTI = "emoticon", ("r0_t352_v310", "r1_t588_v452", "r2_t275_v2667", "r0_t588_v320"), \
    ("ident32", "rand32", "rand1"), ("ident32", "rand32")
GROUPS, N_GROUP_DRAWS, N_KEY_DRAWS = 32, 5, 20
NAT_ROWS, MIN_NAT = ("val", 0, 600), 50
R_TOL, NET_TOL, READ_TOL, SAME_TOL = 0.05, 0.02, 1e-4, 1e-3  # E28's G1 tolerances (r 0.05, arms 0.02)
MIN_GAIN, MIN_GAP, MIN_TRIG = 1.0, 0.5, 0.5
HOLD, FAIL, BEATS, SIZE_TOL = 0.7, 0.5, 0.2, 0.1
TRACK, STEER_OK, CEILING_MIN, MARGIN = 0.3, 0.5, 0.5, 0.05

BLIND = ("the trigger position only; held-out windows (97 to 103 per state; 1 to 7 per key on the 32-key tasks); one "
         "fine-tune seed; one 67M model; the clip at [0, 1]")
NOISE = ("E28's readings are deterministic per state (its control matched E18 within 0.002) and E18's two seeds agreed "
         "within 0.02; per-key means rest on 9 to 15 training windows, whose sampling noise the random-group arm of the "
         "same size measures")


def pred(pid: str, rule: str, predicted: str, confidence: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "predicted": predicted, "confidence": confidence, "positive_control": control,
            "blind_to": BLIND, "noise_floor": NOISE}


PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-22_plan_shift-anatomy.md. Filed before any run; never edited after.",
    "states": f"E28's eight do-nothing states, regenerated in-job: {CONTROL} (G1, first, alone), {WORD}, {KV_TASKS}",
    "measurements": {
        "R_mean_net": "E28's net fixed change: (rho(m1) - rho(m0)) / rho_trig",
        "R_key_net": "per key k (the token at the trigger): (rho(m1_k) - rho(m0_k)) / rho_trig, m_k over training windows "
                     "with key k, applied to held-out windows of key k",
        "R_group_net": f"the same with {GROUPS} random groups instead of keys, mean of {N_GROUP_DRAWS} draws: the size-"
                       "matched null",
        "R_strength": "clip(p0(x) + a(x) s_p) / rho_trig in pre-activation space (g = clip(p)); a(x) = <d(x), s_p> / "
                      "<s_p, s_p>, d = p1 - p0 at the trigger, s_p its training mean; alive subcomponents, all matrices",
        "var_frac": "share of d's variance across held-out windows (about its mean) along s_p",
        "corr_a_logp": "Pearson corr of a(x) with the original model's log-prob of the target on x",
        "corr_rem_logp": "Pearson corr over held-out windows of lo(own g1(x)) - lo(fixed m1) with the same log-prob",
        "R_attn_fixed / R_attn_ref": "attention matrices at m1 (ref: m0), MLP matrices at each window's own g1(x)",
        "R_mlp_fixed / R_mlp_ref": "MLP matrices at m1 (ref: m0), attention at g1(x)",
        "R_steer": "Gamma recomputed on W0's activations with each matrix's input at the trigger moved by the mean "
                   "(updated - original) input over the training windows; R_steer_own: each window's own updated input "
                   "at the trigger (the ceiling of trigger-only inputs)",
        "pile": f"32-key tasks: mean over keys of corr(s_k - mean s_k, c_k - mean c_k), c_k = W0's contrast at Pile "
                f"{NAT_ROWS} positions holding key k (turned_down.contrasts), against {N_KEY_DRAWS} key derangements; "
                "control: the same with W0's per-key means m0_k in place of s_k",
        "prev_token": f"reported only: s and s > {G.THR} at {SA.PREV_TOKEN}; on the 32-key tasks the sd of s_k over keys",
    },
    "gates": {
        "G1": f"emoticon: |r - E28| <= {R_TOL} and |R_mean_net - E28| <= {NET_TOL} (E28: 0.5965, 0.8147); the per-window "
              f"read matches the job's E28-path reads (regating.prepare's W0 values at the trigger, regating.measure's "
              f"fixed change) within {READ_TOL}; zero shift through "
              f"strength and steer recovers within {READ_TOL} of 0. First, alone; fail: stop",
        "G2": f"each state's r and R_mean_net within {R_TOL} and {NET_TOL} of E28's; fail: reported, not scored",
    },
    "scorable": f"G2 passes, gain >= {MIN_GAIN}, gap >= {MIN_GAP}, rho_trig >= {MIN_TRIG} (as E28)",
    "expected": {
        "W0 (no update)": "gap 0, every share undefined; every zero-change builder recovers exactly 0",
        "control (emoticon)": "r 0.60, R_mean_net 0.81 (E28); other arms reported",
        "single-key states (word tasks, rand1)": "R_key_net equals R_mean_net (one group)",
    },
    "predictions": [
        pred("C_key", f"|R_key_net - R_mean_net| <= {SAME_TOL} on every scorable single-key state", "pass", "high",
             "itself (the per-key code with one key is the fixed change)"),
        pred("C_size", f"word tasks: R_group_net >= R_mean_net - {SIZE_TOL} on every scorable one", "pass", "medium",
             "C_key"),
        *[pred(f"K_{t}", f"R_key_net >= {HOLD} 'holds per key'; < {FAIL} 'depends beyond the key'; else 'partial'. "
               f"'The key matters' if also R_key_net - R_group_net >= {BEATS}",
               "holds per key, the key matters" if t == "ident32" else "two-sided, no lean",
               "low-medium" if t == "ident32" else "low", "C_key, C_size") for t in MULTI],
        pred("S_word", f"R_strength >= {HOLD} on every scorable word task", "pass", "medium",
             "zero strength recovers 0 (G1)"),
        pred("S_ident32", f"R_strength >= {HOLD} 'one direction, varying strength'; < {FAIL} 'not'; else 'partial'",
             "not", "low-medium", "S_word"),
        pred("Rem_word", f"|corr_rem_logp| >= {TRACK} on at least 3 of 4 scorable word tasks 'the remainder tracks the "
             f"original log-prob' (sign reported); < {TRACK} on at least 3 'does not'", "tracks, negative", "low",
             "none (descriptive)"),
        pred("M_word", f"word tasks: R_attn_fixed >= {HOLD} and R_mlp_fixed >= {HOLD} on every scorable one", "pass",
             "medium", "the fixed change works on these tasks (E28)"),
        pred("M_ident32", f"guess A holds if R_attn_fixed >= {HOLD}, R_mlp_fixed < {FAIL} and R_attn_fixed - R_attn_ref "
             f">= {BEATS}; 'fails' if R_attn_fixed < {FAIL}; else 'partial'", "holds", "low", "M_word"),
        pred("B_word", f"R_steer >= {STEER_OK} on at least 3 of 4 scorable word tasks (else B_ident32 unreadable)", "pass",
             "low-medium", "zero steer recovers 0 (G1); R_steer_own the ceiling"),
        pred("B_ident32", f"readable if B_word passes and R_steer_own >= {CEILING_MIN}; guess B holds if R_steer >= "
             f"{STEER_OK}", "fails (< 0.5)", "low", "B_word"),
        pred("P_control", f"control: own - null_mean >= {MARGIN} and own > null_max for W0's m0_k on ident32", "pass",
             "medium", "itself"),
        pred("P_ident32", f"guess C supported if own - null_mean >= {MARGIN} and own > null_max (P_control passing)",
             "supported", "low", "P_control"),
    ],
    "decision_tree": {
        "G1 fails": "stop; nothing scored",
        "K holds per key on ident32, the key matters": "the fixed-shift hypothesis restated per cue-answer pair: one fixed "
            "change per key; the scope question becomes 'per answer' rather than 'tasks the model already does'",
        "K depends beyond the key, S or B holds": "the input dependence is a strength (S) or a coordinate (B) effect; the "
            "hypothesis survives in that form on copy",
        "K, S, B all fail on ident32": "switching on copy depends on the input beyond its key; E28's scope branch "
            "(tasks the model already does) stands",
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
    return [E28_JSON, TRANSPORT / "windows.json", TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt",
            *[LADDER / f"windows_{t}.pt" for t in WORD], *[KV / "large" / f"windows_{t}.pt" for t in KV_TASKS]]


def rows_of(vec: dict[str, Tensor], n: int) -> SA.Vals:
    return {k: v.expand(n, -1) for k, v in vec.items()}


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd

        self.model = vpd.load_component_model(device)
        self.W0, self.device = vpd.target_weights(self.model), device
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}
        self.groups: dict[str, dict[str, Any]] = {}  # the 32-key tasks' per-key means, for the Pile stage

    def reads(self, t: G.Task, dW: dict[str, Tensor]) -> dict[str, Any]:
        h = t.half
        r = {"tr0": SA.trigger_read(self.model, t.train.tokens, h), "he0": SA.trigger_read(self.model, t.held.tokens, h, t.target)}
        with G.state_of(self.model, self.W0, dW):
            r["tr1"], r["he1"] = SA.trigger_read(self.model, t.train.tokens, h), SA.trigger_read(self.model, t.held.tokens, h)
        return r

    def state(self, name: str) -> dict[str, Any]:
        model, alive = self.model, self.alive
        w = G.load_kv_windows(name, KV) if name in KV_TASKS else G.load_windows(name, TRANSPORT, LADDER)
        t = G.prepare(model, name, *w)
        dW, ev = G.finetune(model, t, t.target, OPT, LR, STEPS, SEED, self.device, log)
        m = G.measure(model, self.W0, t, dW, {k: torch.zeros(v.shape) for k, v in t.m0.items()}, alive, n_random=1)
        shift, gap, rho = m.pop("shift"), m["gap"], m["rho"]
        out = {k: m[k] for k in ("r", "gain", "gap", "rho", "n_on", "cells")}
        out.update(heldout=ev, n_train=len(t.train), n_heldout=len(t.held),
                   R_mean_net=G.div(rho["mean"] - rho["mean_W0"], rho["trig"]))
        rd = self.reads(t, dW)
        out["checks"] = checks(t, rd, shift, alive)
        out.update(group_arms(model, t, rd, gap, rho["trig"], name, self.groups))
        out.update(strength_arms(model, t, rd, gap, rho["trig"], alive))
        out.update(mixed_arms(model, t, rd, gap, rho["trig"]))
        out.update(steer_arms(model, t, rd, gap, rho["trig"]))
        out["prev_token"] = prev_token(shift, self.groups.get(name))
        log(f"[{name}] r {out['r']:.4f} gain {out['gain']:.3f} gap {gap:.3f} rho_trig {rho['trig']:.4f} | R_mean_net "
            f"{out['R_mean_net']:.4f} | checks {json.dumps(out['checks'])}")
        log(f"[{name}] " + " ".join(f"{k} {out[k]:.4f}" for k in SCORED_KEYS) + f" | prev_token {json.dumps(out['prev_token'])}")
        return out


SCORED_KEYS = ("R_key_net", "R_group_net", "R_strength", "R_preshift", "var_frac", "corr_a_logp", "corr_rem_logp",
               "R_attn_fixed", "R_attn_ref", "R_mlp_fixed", "R_mlp_ref", "R_steer", "R_steer_own")


def checks(t: G.Task, rd: dict[str, Any], shift: dict[str, Tensor], alive: dict[str, Tensor]) -> dict[str, float]:
    """The per-window read against this job's E28-path reads (regating.prepare's W0 values at the trigger,
    regating.measure's fixed change on alive), and the largest |g1 - g0| on dead subcomponents at the trigger (the
    strength arm keeps g0 there; the other arms take g1)."""
    base = {k: torch.cat([g[k][:, t.half] for g in t.base["gates"]]).float() for k in shift}
    m1, m0 = SA.mean_of(rd["tr1"]["g"]), SA.mean_of(rd["tr0"]["g"])
    g1, g0 = rd["he1"]["g"], rd["he0"]["g"]
    return {"read_vs_base": max(float((g0[k] - base[k]).abs().max()) for k in shift),
            "shift_vs_measure": max(float((m1[k] - m0[k] - shift[k])[alive[k]].abs().max()) for k in shift),
            "dead_dg_max": max(float((g1[k] - g0[k])[:, ~alive[k]].abs().amax()) if (~alive[k]).any() else 0.0
                               for k in shift)}


def group_arms(model: Any, t: G.Task, rd: dict[str, Any], gap: float, rho_trig: float, name: str,
               keep: dict[str, Any]) -> dict[str, Any]:
    key_tr, key_he = t.train.tokens[:, t.half], t.held.tokens[:, t.half]
    gk = SA.group_fixed(model, t, rd["tr1"]["g"], rd["tr0"]["g"], key_tr, key_he, gap, rho_trig)
    if name in MULTI:
        keep[name] = {"m1": gk["m1"], "m0": gk["m0"]}
    draws = [SA.group_fixed(model, t, rd["tr1"]["g"], rd["tr0"]["g"], SA.random_labels(len(t.train), GROUPS, i),
                            SA.random_labels(len(t.held), GROUPS, 100 + i), gap, rho_trig)["R_net"]
             for i in range(N_GROUP_DRAWS)]
    info = {k: gk[k] for k in ("n_groups", "train_per_group_min", "train_per_group_median", "rho", "rho_W0")}
    return {"R_key_net": gk["R_net"], "key": info, "R_group_net": sum(draws) / len(draws), "group_draws": draws}


def strength_arms(model: Any, t: G.Task, rd: dict[str, Any], gap: float, rho_trig: float,
                  alive: dict[str, Tensor]) -> dict[str, Any]:
    p0, p1 = rd["he0"]["pre"], rd["he1"]["pre"]
    q1, q0 = SA.mean_of(rd["tr1"]["pre"]), SA.mean_of(rd["tr0"]["pre"])
    s_p = {k: q1[k] - q0[k] for k in p0}
    a, frac = SA.along({k: p1[k] - p0[k] for k in p0}, s_p, alive)
    n = a.shape[0]
    rho = {name: SA.rho_of(model, t, SA.strength(p0, aa, s_p, alive), gap)
           for name, aa in (("a", a), ("one", torch.ones(n)), ("zero", torch.zeros(n)))}
    logp = rd["he0"]["logp"]
    m1 = rows_of(SA.mean_of(rd["tr1"]["g"]), n)
    rem = SA.per_window_lo(model, t, rd["he1"]["g"]) - SA.per_window_lo(model, t, m1)
    return {"R_strength": G.div(rho["a"], rho_trig), "R_preshift": G.div(rho["one"], rho_trig), "rho_zero_strength": rho["zero"],
            "var_frac": frac, "corr_a_logp": SA.pearson(a, logp), "a_mean": float(a.mean()), "a_sd": float(a.std()),
            "corr_rem_logp": SA.pearson(rem, logp), "rem_mean": float(rem.mean()), "logp_mean": float(logp.mean())}


def mixed_arms(model: Any, t: G.Task, rd: dict[str, Any], gap: float, rho_trig: float) -> dict[str, float]:
    g1 = rd["he1"]["g"]
    n = next(iter(g1.values())).shape[0]
    m1, m0 = rows_of(SA.mean_of(rd["tr1"]["g"]), n), rows_of(SA.mean_of(rd["tr0"]["g"]), n)
    mlp = lambda k: not SA.is_attention(k)
    arms = {"R_attn_fixed": (m1, SA.is_attention), "R_attn_ref": (m0, SA.is_attention),
            "R_mlp_fixed": (m1, mlp), "R_mlp_ref": (m0, mlp)}
    return {name: G.div(SA.rho_of(model, t, SA.mixed(v, g1, take), gap), rho_trig) for name, (v, take) in arms.items()}


def steer_arms(model: Any, t: G.Task, rd: dict[str, Any], gap: float, rho_trig: float) -> dict[str, Any]:
    x0, x1 = rd["tr0"]["x"], rd["tr1"]["x"]
    mx1, mx0 = SA.mean_of(x1), SA.mean_of(x0)
    delta = {k: mx1[k] - mx0[k] for k in x0}
    zero = {k: torch.zeros_like(v) for k, v in delta.items()}
    return {"R_steer": G.div(SA.steer(model, t, SA.add_at_trigger(t.half, delta), gap), rho_trig),
            "R_steer_own": G.div(SA.steer(model, t, SA.set_at_trigger(t.half, rd["he1"]["x"]), gap), rho_trig),
            "rho_zero_steer": SA.steer(model, t, SA.add_at_trigger(t.half, zero), gap),
            "delta_norm": {k: float(v.norm()) for k, v in delta.items()}}


def prev_token(shift: dict[str, Tensor], groups: dict[str, Any] | None) -> dict[str, Any]:
    out = {}
    for key, c in SA.PREV_TOKEN:
        s = float(shift[key][c])
        row: dict[str, Any] = {"s": s, "on": s > G.THR}
        if groups is not None:
            sk = torch.stack([groups["m1"][k][key][c] - groups["m0"][k][key][c] for k in groups["m1"]])
            row.update(s_key_sd=float(sk.std()), s_key_min=float(sk.min()), s_key_max=float(sk.max()))
        out[f"{key}:{c}"] = row
    return out


# ----------------------------------------------------------------------------- gates, the Pile stage, scoring


def gate(name: str, ref: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
    """G1 (control) or G2: r and the net fixed change reproduce E28; G1 also needs the read checks and zero builders."""
    diff = {"r": m["r"] - ref["r"], "R_mean_net": m["R_mean_net"] - ref["R_mean_net"]}
    ok = abs(diff["r"]) <= R_TOL and abs(diff["R_mean_net"]) <= NET_TOL
    zeros = {"read_vs_base": m["checks"]["read_vs_base"], "shift_vs_measure": m["checks"]["shift_vs_measure"],
             "rho_zero_strength": m["rho_zero_strength"], "rho_zero_steer": m["rho_zero_steer"]}
    if name == CONTROL:
        ok = ok and all(abs(v) <= READ_TOL for v in zeros.values())
    log(f"[{'G1' if name == CONTROL else 'G2'} {name}] diff from E28 {json.dumps(diff)} zeros {json.dumps(zeros)} | "
        f"{'ok' if ok else 'FAIL'}")
    return {"diff": diff, "zeros": zeros, "ok": ok}


def pile_stage(r: Run) -> dict[str, Any]:
    """Per 32-key task: the key-specific part of s_k against the key's own W0 contrast at its Pile occurrences."""
    from commute import turned_down as D
    from commute.data import pile_rows

    rows = pile_rows(*NAT_ROWS, log=log)
    keys = sorted({k for g in r.groups.values() for k in g["m1"]})
    con = D.contrasts(r.model, rows, {str(k): rows == k for k in keys}, r.alive)
    n_pos = {k: con[str(k)]["n_positions"] for k in keys}
    use = [k for k in keys if n_pos[k] >= MIN_NAT]
    c = {k: con[str(k)]["contrast"] for k in use}
    out: dict[str, Any] = {"n_positions": n_pos, "keys_used": len(use)}
    for name, g in r.groups.items():
        s_k = {k: {m: g["m1"][k][m] - g["m0"][k][m] for m in g["m1"][k]} for k in use}
        out[name] = {"shift": SA.key_match(s_k, c, r.alive, N_KEY_DRAWS),
                     "control_m0": SA.key_match({k: g["m0"][k] for k in use}, c, r.alive, N_KEY_DRAWS)}
        log(f"[pile {name}] keys {len(use)} | shift own {out[name]['shift']['own']:.4f} null "
            f"{out[name]['shift']['null_mean']:.4f} max {out[name]['shift']['null_max']:.4f} | control own "
            f"{out[name]['control_m0']['own']:.4f} null {out[name]['control_m0']['null_mean']:.4f} max "
            f"{out[name]['control_m0']['null_max']:.4f}")
    return out


def scorable(m: dict[str, Any], ok: bool) -> bool:
    return ok and m["gain"] >= MIN_GAIN and m["gap"] >= MIN_GAP and m["rho"]["trig"] >= MIN_TRIG


def three_way(x: float, hold: str, fail: str) -> str:
    return hold if x >= HOLD else fail if x < FAIL else "partial"


def matched(p: dict[str, Any]) -> bool:
    return p["own"] - p["null_mean"] >= MARGIN and p["own"] > p["null_max"]


def every(oks: Any) -> bool | None:
    """all(), but None (nothing to score) on an empty list rather than a vacuous pass."""
    oks = list(oks)
    return all(oks) if oks else None


def score(out: dict[str, Any]) -> dict[str, Any]:
    st = {t: m for t, m in out["states"].items() if scorable(m, out["gates"][t]["ok"])}
    word = [t for t in WORD if t in st]
    single = [t for t in (*WORD, "rand1") if t in st]
    sc: dict[str, Any] = {"scorable": sorted(st),
                          "C_key": every(abs(st[t]["R_key_net"] - st[t]["R_mean_net"]) <= SAME_TOL for t in single),
                          "C_size": every(st[t]["R_group_net"] >= st[t]["R_mean_net"] - SIZE_TOL for t in word),
                          "S_word": every(st[t]["R_strength"] >= HOLD for t in word),
                          "M_word": every(min(st[t]["R_attn_fixed"], st[t]["R_mlp_fixed"]) >= HOLD for t in word)}
    for t in MULTI:
        if t in st:
            k = three_way(st[t]["R_key_net"], "holds per key", "depends beyond the key")
            sc[f"K_{t}"] = k + (", the key matters" if st[t]["R_key_net"] - st[t]["R_group_net"] >= BEATS else "")
    track = [abs(st[t]["corr_rem_logp"]) >= TRACK for t in word]
    sign = "negative" if all(st[t]["corr_rem_logp"] < 0 for t in word) else "positive" if all(
        st[t]["corr_rem_logp"] > 0 for t in word) else "mixed sign"
    sc["Rem_word"] = ("tracks, " + sign if sum(track) >= 3 else "does not" if len(track) - sum(track) >= 3
                      else "mixed") if word else None
    sc["B_word"] = sum(st[t]["R_steer"] >= STEER_OK for t in word) >= 3 if word else None
    if "ident32" in st:
        m = st["ident32"]
        sc["S_ident32"] = three_way(m["R_strength"], "one direction, varying strength", "not")
        a_holds = m["R_attn_fixed"] >= HOLD and m["R_mlp_fixed"] < FAIL and m["R_attn_fixed"] - m["R_attn_ref"] >= BEATS
        sc["M_ident32"] = "holds" if a_holds else "fails" if m["R_attn_fixed"] < FAIL else "partial"
        sc["B_ident32"] = ("unreadable" if not (sc["B_word"] and m["R_steer_own"] >= CEILING_MIN)
                           else "holds" if m["R_steer"] >= STEER_OK else "fails")
    if "pile" in out and "ident32" in out["pile"] and "ident32" in st:
        p = out["pile"]["ident32"]
        sc["P_control"] = matched(p["control_m0"])
        sc["P_ident32"] = ("unreadable" if not sc["P_control"] else "supported" if matched(p["shift"]) else "not")
    return sc


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    missing = [str(p) for p in inputs() if not p.is_file()]
    if missing:
        sys.exit("missing inputs (gitignored):\n  " + "\n  ".join(missing))
    t0 = time.time()
    e28 = json.loads(E28_JSON.read_text())["states"]
    refs = {t: {"r": m["r"], "R_mean_net": m["R_mean_net"]} for t, m in e28.items()}
    r = Run(device)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "refs": refs, "states": {}, "gates": {}}
    log(f"[start] {out['device']}; E28 refs {json.dumps(refs)}")
    for name in (CONTROL, *WORD, *KV_TASKS):
        out["states"][name] = m = r.state(name)
        out["gates"][name] = gate(name, refs[name], m)
        if name == CONTROL and not out["gates"][name]["ok"]:
            out["status"] = "stopped at G1"
            dump(out)
            log(f"G1 FAILED: stopping ({time.time() - t0:.0f} s)")
            return
        out["score"] = score(out)
        dump(out)
    out["pile"] = pile_stage(r)
    out["score"] = score(out)
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
