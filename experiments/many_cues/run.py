"""many_cues (E33): does the fixed change of Gamma's importance values follow the cue or the answer?
Plan: reports/2026-10-22_plan_many-cues.md. Definitions: commute.many_cues's docstring; E28's arms in
reports/2026-10-22_plan_regating-tasks.md. Shared steps in commute.regating, commute.shift_anatomy, commute.many_cues.

Do-nothing states as E28 makes them (10-step arm-A Adam 3e-5 fine-tunes of all 24 matrices, seed 0; released
subcomponents, Gamma frozen, Delta out of the masked model). Stages, cheapest first, results written after each:
  G1     emoticon, alone: reproduces E28's r and net fixed change, and the per-window read reproduces E28's path; stop on
         a fail
  word   r1_t588_v452 (one cue, one answer) through every arm: the per-cue change must equal the shared one (C_one; stop
         on a fail)
  mine   K cues for one target from Pile train rows (the task ladder's rows and rules), and a matched other target per cue
  many   the K cues, each toward its own target (the many-answers null)
  one    the K cues, all toward the one target (the question)

  run.py --predictions   file results/predictions.json (once, before the run)
  run.py                 -> results/many_cues.json, results/large/{full.log, windows.pt, <state>_shifts.pt}
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

from commute import many_cues as MC
from commute import regating as G
from commute import shift_anatomy as SA

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "many_cues.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
E28_JSON = EXP / "regating_tasks" / "results" / "regating_tasks.json"

STEPS, SEED, OPT, LR, HALF = 10, 0, "adam", 3e-5, 20  # E28's fine-tune and window
CONTROL, WORD = "emoticon", "r1_t588_v452"
ROWS, POOL, MIN_OCC = ("train", 0, 32768), 600, 100  # the task ladder's mined rows; the most frequent word-like tokens
K, K_MIN, MAX_WIN, HELD_CAP, MIN_TRAIN, MIN_HELD = 32, 16, 60, 6, 30, 4
N_GROUP_DRAWS, N_DONOR = 5, 5
R_TOL, NET_TOL, READ_TOL, SAME_TOL, SIZE_TOL = 0.05, 0.02, 1e-4, 1e-3, 0.1
MIN_GAIN, MIN_GAP, MIN_TRIG = 1.0, 0.5, 0.5
WORKS, FAILS, GAP = 0.7, 0.5, 0.2

BLIND = ("the trigger position only; one fine-tune seed; one 67M model; the clip at [0, 1]; cues mined from frequent "
         "words; the shared and per-cue changes are means, blind to context within a cue")
NOISE = ("E28's readings are deterministic per state (its control matched E18 within 0.002; E18's two seeds within "
         "0.02); per-cue means rest on >= 30 training windows (E32's per-key means on 9 to 15), and the random-group arm "
         "of the same sizes measures what averaging fewer windows does alone; cos_within is the overlap's noise ceiling")


def pred(pid: str, rule: str, predicted: str, confidence: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "predicted": predicted, "confidence": confidence, "positive_control": control,
            "blind_to": BLIND, "noise_floor": NOISE}


READING_RULE = (f"'neither' if R_cue_net < {FAILS}; 'per cue only' if R_cue_net >= {WORKS} and R_cue_net - "
                f"R_shared_net >= {GAP}; 'shared' if R_shared_net >= {WORKS} and R_cue_net - R_shared_net < {GAP}; else "
                f"'partial' (many_cues.reading)")
PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-22_plan_many-cues.md. Filed before any run; never edited after.",
    "states": f"{CONTROL} (G1, first, alone); {WORD} (one cue, one answer); 'many' (K cues, each its own answer); 'one' "
              f"(the same K cues, one answer); K = {K} (at least {K_MIN}, else the job stops after mining)",
    "mining": f"Pile {ROWS} rows; pool = the {POOL} most frequent word-like tokens with >= {MIN_OCC} occurrences; a "
              f"pair qualifies if the target is word-like, not the trigger, baseline in [{MC.LO}, {MC.HI}) and >= "
              f"{MC.LIFT} x the target's background probability; target = most qualifying triggers; cues = qualifying "
              f"triggers by occurrence count, each with >= {MIN_TRAIN} training and >= {MIN_HELD} held-out windows (up "
              f"to {MAX_WIN} windows, split by row as the ladder, held-out capped at {HELD_CAP}) and a distinct matched "
              "other target (many_cues.alternative)",
    "measurements": {
        "R_shared_net": "E28's net fixed change: (rho(m1) - rho(m0)) / rho_trig, means over all training windows",
        "R_cue_net": "per cue k (the token at the trigger): (rho(m1_k) - rho(m0_k)) / rho_trig, means over cue k's "
                     "training windows, given to cue k's held-out windows (shift_anatomy.group_fixed)",
        "R_group_net": f"the same with K random groups, mean of {N_GROUP_DRAWS} draws: the size-matched null",
        "overlap": "split-half cosines of the per-cue change s_k: mean cos across cues / mean cos within a cue "
                   "(many_cues.overlap); reported",
        "R_other_within_net / R_other_across_net": f"each held-out window given another window's updated values, the "
                   f"donor from the same cue / a different cue, minus the same with the donor's original values, over "
                   f"rho_trig; mean of {N_DONOR} draws; reported",
    },
    "gates": {
        "G1": f"emoticon: |r - E28| <= {R_TOL}, |R_shared_net - E28| <= {NET_TOL} (E28: 0.5965, 0.8147); the per-window "
              f"read gives W0's own values recovery 0 and the state's own values rho_trig, and its mean change equals "
              f"regating.measure's, each within {READ_TOL}. First, alone; fail: stop",
        "G2": f"{WORD}: r and R_shared_net within {R_TOL} and {NET_TOL} of E28's; reported",
    },
    "scorable": f"gain >= {MIN_GAIN}, gap >= {MIN_GAP}, rho_trig >= {MIN_TRIG} (as E28); a many-cue state is read "
                f"only if C_size holds on it",
    "expected": {
        "W0 (no update)": "gap 0, every share undefined; W0's own values through every builder recover exactly 0",
        f"control ({WORD})": "r 0.904, R_shared_net 0.916 (E28); R_cue_net = R_shared_net; R_other_within_net about "
                             "0.93 (E28's other 0.955 minus its reference 0.028)",
    },
    "predictions": [
        pred("C_one", f"{WORD}: |R_cue_net - R_shared_net| <= {SAME_TOL}; fail: stop", "pass", "high",
             "itself (one cue: the per-cue code is the shared change)"),
        pred("C_size", f"each many-cue state: |R_group_net - R_shared_net| <= {SIZE_TOL}", "pass", "medium", "C_one"),
        pred("A_many", READING_RULE + " on 'many'", "per cue only", "medium-low", "C_one, C_size"),
        pred("A_one", READING_RULE + " on 'one'", "shared", "low", "A_many"),
        pred("Q", "from (A_one, A_many): (shared, per cue only) 'follows the answer'; (per cue only, per cue only) "
             "'follows the cue'; (shared, shared) 'follows the kind of task'; either 'neither': 'the per-cue change "
             "fails on a joint fine-tune'; else 'mixed' (many_cues.combined)", "follows the answer", "low",
             "A_many"),
    ],
    "decision_tree": {
        "G1 or C_one fails": "stop; nothing scored",
        "follows the answer": "one fixed change per answer: the copy task fails because it has 32 answers; the scope "
                              "question becomes 'per answer', not 'tasks the model already does'",
        "follows the cue": "one fixed change per cue: the one-cue confound resolved toward the cue; E32's per-key reading "
                           "on copy decides whether existing behaviour matters as well",
        "follows the kind of task": "one change carries any mix of existing next-word pairs: E28's scope ('tasks the "
                                    "model already does') stands",
        "the per-cue change fails on a joint fine-tune": "the hypothesis covers single-pair fine-tunes only; tell the user",
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
            LADDER / f"windows_{WORD}.pt"]


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd

        self.model = vpd.load_component_model(device)
        self.W0, self.device = vpd.target_weights(self.model), device
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}

    def reads(self, t: G.Task, dW: dict[str, Tensor]) -> dict[str, SA.Vals]:
        """Gamma's values at the trigger, per window: train and held-out, at W0 (0) and in the state (1)."""
        r = {"tr0": SA.trigger_read(self.model, t.train.tokens, t.half)["g"],
             "he0": SA.trigger_read(self.model, t.held.tokens, t.half)["g"]}
        with G.state_of(self.model, self.W0, dW):
            r["tr1"] = SA.trigger_read(self.model, t.train.tokens, t.half)["g"]
            r["he1"] = SA.trigger_read(self.model, t.held.tokens, t.half)["g"]
        return r

    def state(self, name: str, train: Any, held: Any, target: int | Tensor, arms: bool) -> dict[str, Any]:
        model = self.model
        t = G.prepare(model, name, train, held, target)
        dW, ev = G.finetune(model, t, t.target, OPT, LR, STEPS, SEED, self.device, log)
        m = G.measure(model, self.W0, t, dW, {k: torch.zeros(v.shape) for k, v in t.m0.items()}, self.alive, n_random=1)
        shift, gap, rho = m.pop("shift"), m["gap"], m["rho"]
        out = {k: m[k] for k in ("r", "gain", "gap", "rho", "n_on", "cells")}
        out.update(heldout=ev, n_train=len(t.train), n_heldout=len(t.held),
                   R_shared_net=G.div(rho["mean"] - rho["mean_W0"], rho["trig"]))
        rd = self.reads(t, dW)
        out["checks"] = checks(model, t, rd, shift, gap, rho["trig"], self.alive)
        log(f"[{name}] r {out['r']:.4f} gain {out['gain']:.3f} gap {gap:.3f} rho_trig {rho['trig']:.4f} held-out P "
            f"{ev['p_o']:.3f} | R_shared_net {out['R_shared_net']:.4f} | checks {json.dumps(out['checks'])}")
        if arms:
            out.update(cue_arms(model, t, rd, gap, rho["trig"], self.alive))
            log(f"[{name}] " + " ".join(f"{k} {out[k]:.4f}" for k in SCORED_KEYS) + f" | cues {json.dumps(out['cues'])}")
            torch.save({"shift": shift, "s_k": out.pop("s_k")}, LARGE / f"{name}_shifts.pt")
        return out


SCORED_KEYS = ("R_shared_net", "R_cue_net", "R_group_net", "overlap", "cos_within", "cos_across",
               "R_other_within_net", "R_other_across_net")


def checks(model: Any, t: G.Task, rd: dict[str, SA.Vals], shift: dict[str, Tensor], gap: float, rho_trig: float,
           alive: dict[str, Tensor]) -> dict[str, float]:
    """The per-window read against regating's path: W0's own values recover 0, the state's own values rho_trig, and
    the read's mean change is regating.measure's fixed change (alive subcomponents)."""
    m1, m0 = SA.mean_of(rd["tr1"]), SA.mean_of(rd["tr0"])
    return {"rho_own_W0": SA.rho_of(model, t, rd["he0"], gap),
            "own_state_vs_trig": SA.rho_of(model, t, rd["he1"], gap) - rho_trig,
            "shift_vs_measure": max(float((m1[k] - m0[k] - shift[k])[alive[k]].abs().max()) for k in shift)}


def cue_arms(model: Any, t: G.Task, rd: dict[str, SA.Vals], gap: float, rho_trig: float,
             alive: dict[str, Tensor]) -> dict[str, Any]:
    lab_tr, lab_he = t.train.tokens[:, t.half], t.held.tokens[:, t.half]
    gk = SA.group_fixed(model, t, rd["tr1"], rd["tr0"], lab_tr, lab_he, gap, rho_trig)
    n = gk["n_groups"]
    draws = [SA.group_fixed(model, t, rd["tr1"], rd["tr0"], SA.random_labels(len(t.train), n, i),
                            SA.random_labels(len(t.held), n, 100 + i), gap, rho_trig)["R_net"]
             for i in range(N_GROUP_DRAWS)]
    ov = MC.overlap({k: rd["tr1"][k] - rd["tr0"][k] for k in rd["tr1"]}, lab_tr, alive)
    s_k = {lab: {k: gk["m1"][lab][k] - gk["m0"][lab][k] for k in gk["m1"][lab]} for lab in gk["m1"]}
    out = {"R_cue_net": gk["R_net"], "R_group_net": sum(draws) / len(draws), "group_draws": draws, **ov,
           "cues": {"train": MC.counts(lab_tr), "heldout": MC.counts(lab_he)}, "s_k": s_k}
    for kind, same in (("within", True), ("across", False)):
        o = other_values(model, t, rd, lab_he, same, gap, rho_trig) if same or n > 1 else {"net": float("nan")}
        out[f"R_other_{kind}_net"], out[f"other_{kind}"] = o["net"], o
    return out


def other_values(model: Any, t: G.Task, rd: dict[str, SA.Vals], labels: Tensor, same: bool, gap: float,
                 rho_trig: float) -> dict[str, Any]:
    """Each held-out window given a donor's updated values (same cue or another), net of the donor's original values."""
    raw, ref = [], []
    for i in range(N_DONOR):
        idx = MC.donors(labels, same, i)
        raw.append(G.div(SA.rho_of(model, t, {k: v[idx] for k, v in rd["he1"].items()}, gap), rho_trig))
        ref.append(G.div(SA.rho_of(model, t, {k: v[idx] for k, v in rd["he0"].items()}, gap), rho_trig))
    net = [a - b for a, b in zip(raw, ref, strict=True)]
    return {"net": sum(net) / len(net), "net_draws": net, "raw": sum(raw) / len(raw), "ref": sum(ref) / len(ref)}


# ----------------------------------------------------------------------------- mining


def mine(r: Run) -> dict[str, Any]:
    """K cues for one target and a matched other target per cue (commute.many_cues), with their windows."""
    from commute import ladder, vpd
    from commute.data import pile_rows
    from commute.finetune import logits_of

    model, tok = r.model, vpd.tokenizer()
    rows = pile_rows(*ROWS, log=log)
    with torch.no_grad():
        vocab = logits_of(model.target_model, rows[:1].to(r.device)).shape[-1]
    plain = ladder.plain_mask(tok.decode, vocab)
    freq = ladder.count_tokens(rows, HALF)
    pool = [t for t, c in sorted(freq.items(), key=lambda tc: -tc[1]) if c >= MIN_OCC and t < vocab and plain[t]][:POOL]
    stats = ladder.trigger_stats(lambda x: logits_of(model.target_model, x), rows, pool, HALF, device=r.device, log=log)
    Q = MC.qualifying(stats, pool, plain)
    v = MC.pick_target(Q)
    P, bg = torch.stack([stats.mean_p[t] for t in pool]), stats.background
    assert bg is not None
    order = sorted((i for i in range(len(pool)) if Q[i, v]), key=lambda i: -freq[pool[i]])
    cues, trains, helds, used = [], [], [], set()
    for i in order:
        train, held = split_by_row(ladder.windows_for_trigger(rows, pool[i], HALF, MAX_WIN, seed=0))
        alt = MC.alternative(P[i], Q[i], v, used)
        if len(train) < MIN_TRAIN or len(held) < MIN_HELD or alt is None:
            continue
        used.add(alt)
        trains.append(train)
        helds.append(held)
        cues.append({"trigger": pool[i], "trigger_str": tok.decode([pool[i]]), "count": freq[pool[i]],
                     "baseline": float(P[i, v]), "lift": float(P[i, v] / bg[v]), "alt": alt,
                     "alt_str": tok.decode([alt]), "alt_baseline": float(P[i, alt]), "n_train": len(train),
                     "n_heldout": len(held)})
        if len(cues) == K:
            break
    out = {"target": v, "target_str": tok.decode([v]), "background": float(bg[v]), "n_qualifying": len(order),
           "pool": len(pool), "vocab": vocab, "K": len(cues), "cues": cues}
    train, held = MC.concat(trains), MC.concat(helds) if cues else (None, None)
    torch.save({"train": train, "heldout": held, "mine": out}, LARGE / "windows.pt")
    log(f"[mine] target {out['target_str']!r} ({v}), background {out['background']:.2e}, {len(order)} qualifying "
        f"triggers in a pool of {len(pool)}; K {len(cues)}: " + ", ".join(
            f"{c['trigger_str']!r} {c['baseline']:.3f} (alt {c['alt_str']!r} {c['alt_baseline']:.3f}) "
            f"{c['n_train']}/{c['n_heldout']}" for c in cues))
    return {**out, "windows": (train, held)}


def split_by_row(w: Any) -> tuple[Any, Any]:
    """The ladder's split (every fifth source row held out), held-out capped at HELD_CAP."""
    from commute.data import split_windows

    train, held = split_windows(w)
    return train, held.subset(torch.arange(len(held)) < HELD_CAP)


# ----------------------------------------------------------------------------- gates and scoring


def gate_g1(m: dict[str, Any], ref: dict[str, Any]) -> bool:
    diff = {"r": m["r"] - ref["r"], "R_shared_net": m["R_shared_net"] - ref["R_mean_net"]}
    ok = abs(diff["r"]) <= R_TOL and abs(diff["R_shared_net"]) <= NET_TOL and all(
        abs(x) <= READ_TOL for x in m["checks"].values())
    log(f"[G1 {CONTROL}] diff from E28 {json.dumps(diff)} checks {json.dumps(m['checks'])} | {'ok' if ok else 'FAIL'}")
    return ok


def gate_word(m: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    diff = {"r": m["r"] - ref["r"], "R_shared_net": m["R_shared_net"] - ref["R_mean_net"]}
    g2 = abs(diff["r"]) <= R_TOL and abs(diff["R_shared_net"]) <= NET_TOL
    c_one = abs(m["R_cue_net"] - m["R_shared_net"]) <= SAME_TOL
    log(f"[G2 {WORD}] diff from E28 {json.dumps(diff)} | G2 {'ok' if g2 else 'FAIL'} | C_one R_cue_net - R_shared_net "
        f"{m['R_cue_net'] - m['R_shared_net']:+.2e} {'ok' if c_one else 'FAIL'}")
    return {"diff": diff, "G2": g2, "C_one": c_one}


def state_reading(m: dict[str, Any]) -> str:
    if not (m["gain"] >= MIN_GAIN and m["gap"] >= MIN_GAP and m["rho"]["trig"] >= MIN_TRIG):
        return "unscorable"
    if abs(m["R_group_net"] - m["R_shared_net"]) > SIZE_TOL:
        return "unreadable (C_size)"
    return MC.reading(m["R_cue_net"], m["R_shared_net"], WORKS, FAILS, GAP)


def score(out: dict[str, Any]) -> dict[str, Any]:
    st = out["states"]
    sc: dict[str, Any] = {"C_one": out["gates"].get("word", {}).get("C_one")}
    for name in ("many", "one"):
        if name in st:
            sc[f"C_size_{name}"] = abs(st[name]["R_group_net"] - st[name]["R_shared_net"]) <= SIZE_TOL
            sc[f"A_{name}"] = state_reading(st[name])
    a, b = sc.get("A_one"), sc.get("A_many")
    readable = {"shared", "per cue only", "neither", "partial"}
    sc["Q"] = MC.combined(a, b) if a in readable and b in readable else None
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
    r = Run(device)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "states": {}, "gates": {}, "status": "running"}
    log(f"[start] {out['device']}")

    def stop(why: str) -> None:
        out["status"] = why
        dump(out)
        log(f"{why}: stopping ({time.time() - t0:.0f} s)")

    out["states"][CONTROL] = m = r.state(CONTROL, *G.load_windows(CONTROL, TRANSPORT, LADDER), arms=False)
    out["gates"]["G1"] = gate_g1(m, e28[CONTROL])
    if not out["gates"]["G1"]:
        return stop("stopped at G1")
    dump(out)
    out["states"]["word"] = m = r.state(WORD, *G.load_windows(WORD, TRANSPORT, LADDER), arms=True)
    out["gates"]["word"] = gate_word(m, e28[WORD])
    if not out["gates"]["word"]["C_one"]:
        return stop("stopped at C_one")
    dump(out)
    mined = mine(r)
    (train, held), out["mine"] = mined.pop("windows"), mined
    dump(out)
    if mined["K"] < K_MIN:
        return stop(f"stopped at mining: {mined['K']} cues < {K_MIN}")
    pairs = {c["trigger"]: c["alt"] for c in mined["cues"]}
    for name, target in (("many", MC.table(pairs, mined["vocab"])), ("one", mined["target"])):
        out["states"][name] = r.state(name, train, held, target, arms=True)
        out["score"] = score(out)
        dump(out)
    out["seconds"], out["status"] = time.time() - t0, "done"
    dump(out)
    log("score " + json.dumps(_finite(out["score"])))
    log(f"FINISHED wrote {OUT} and {LARGE} ({out['seconds']:.0f} s)")


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
