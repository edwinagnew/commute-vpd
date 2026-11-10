"""E132 Amendment 1, numbered E144 (cell P3.2): the log-odds re-gating share r of GOALS.md → Terms, read on the new
decomposition p-9707d6ff for E132's four updates and two random changes. It is a POST-HOC DEVIATION from E132's filed
KL-based share, chosen after E132 read every state near its ceiling (0.991 to 0.9995); every write-up says so. Plan:
reports/plans/2026-11-10_plan_e144-e132-amend1-logodds.md; rules in results/predictions_amend1.json (the run refuses to
start without it, or while results/amend1/stopped.json exists).

r = Σ(L01 − L00) / Σ(T1 − T0) over the target positions of a row set: T0, T1 the unmasked model's log-odds of the
trained answer (teacher-forced next token, `identity.log_odds`) under W0 and the state; L00, L01 the masked log-odds with
the original subcomponents (Δ off) under Γ's gates from W0's forward and from the state's forward. Nan when
|Σ gain| < pcfg_vpd.GAIN_MIN; 'scored' only when the mean gain per position is at least local_faith.MIN_GAIN.
Rows: E45's own rows (64 packed rows of 512, `qr.own_seqs`), every response token a target: gold GSM8K rows (E40's 256
held-out train problems) for e40 and gold_s0, the held-template fact rows of each seed for a2long_s0 and a2long_s1.

One job on one L40, in order; each stage writes its files under results/amend1/ and every number goes to its run.log;
a resubmitted job skips what is written:
  controls  W0 as the fine-tuned model (both sums 0); the unmasked KL of gold_s0 on its own rows (E45) and of E40 on
            16 maths rows (E132); the final RMS norm scaled by 2, which Γ never reads (gates bitwise W0's, r = 0).
  U_<update>          the four updates on their own rows (W0_<rows>.json caches each row set's W0 pass).
  R_<random>_<rows>   E132's two random changes, rebuilt and scaled by E132's saved factors, on each row set.
  summary   r with 95% row-resample intervals, the seed-pair difference, the card rule, the random reference.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from itertools import permutations
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_eval, qwen_sets
from commute import qwen_reading as qr
from commute.identity import log_odds
from commute.jobs import hard_exit, load_experiment
from commute.local_faith import MIN_GAIN
from commute.pcfg_vpd import GAIN_MIN
from commute.qwen_sft import pack
from commute.qwen_vpd import decomp_id, importances, masked_logits

E132 = load_experiment("qwen_regating_new", "run")  # sets E57's Lab to the new decomposition (GSTATS, alive count)
L = E132.L
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "amend1"
L.RESULTS, L.PRED, L.LARGE = RESULTS, HERE / "results" / "predictions_amend1.json", HERE / "results" / "large"
CFG, DEV, NAN = L.CFG, L.DEV, float("nan")
OWN = {"gold": ("gold", 0), "facts_s0": ("a2", 0), "facts_s1": ("a2", 1)}  # row set -> qr.own_seqs (arm, seed)
ROWSET = {"e40": "gold", "gold_s0": "gold", "a2long_s0": "facts_s0", "a2long_s1": "facts_s1"}
RANDOM = E132.RANDOM
N_ROWS, ROW_LEN = 64, 512  # E45's own rows
CEIL, GAP = 0.85, 0.1  # the card's rule
NORM_SCALE = 2.0
REF_KL = {"gold_s0": json.loads((L.E45 / "E_gold_s0.json").read_text())["states"]["100"]["kl"]["own"],
          "e40": json.loads((HERE / "results" / "controls.json").read_text())["kl_e40_16"]}
SAVED = {n: json.loads((HERE / "results" / f"S_{n}.json").read_text())["search"] for n in RANDOM}
Forward = Callable[[Tensor], tuple[Tensor, dict[str, Tensor]]]


def row_sets() -> dict[str, tuple[Tensor, Tensor]]:
    """Each row set's ids (E45's own rows, asserted equal to `qr.own_rows`) and target flags: a response token of its
    sequence, packed alongside the ids as `qwen_sft.pack` packs them (a sequence may run across two rows)."""
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    train = qwen_sets.load()["train"]
    out = {}
    for name, (arm, seed) in OWN.items():
        seqs = qr.own_seqs(tok, arm, seed, train)
        ids = pack([s for s, _ in seqs], N_ROWS, ROW_LEN)
        assert ids == qr.own_rows(tok, arm, seed, train), name
        flags = pack([[int(i >= p) for i in range(len(s))] for s, p in seqs], N_ROWS, ROW_LEN)
        out[name] = (torch.tensor(ids), torch.tensor(flags, dtype=torch.bool))
    return out


def summed(logits: Tensor, t: Tensor, f: Tensor) -> float:
    """Σ log-odds of the next token at the positions whose next token is a target (float64)."""
    return float(log_odds(logits[0, :-1], t[0, 1:])[f[1:]].double().sum())


def pass_rows(lab: Any, rows: tuple[Tensor, Tensor], forward: Forward, tag: str) -> dict[str, list[float]]:
    """Per row: T (summed log-odds from the clean logits `forward` gives), L (masked with the original subcomponents,
    Δ off, under the gates `forward` gives) and n (targets read)."""
    out: dict[str, list[float]] = {"T": [], "L": [], "n": []}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for i, (r, f) in enumerate(zip(*rows, strict=True)):
        t, fd = r[None].to(DEV), f.to(DEV)
        logits, g = forward(t)
        out["T"].append(summed(logits, t, fd))
        del logits
        out["L"].append(summed(masked_logits(lab.model, t, lab.comps, g), t, fd))
        out["n"].append(int(fd[1:].sum()))
        del g
        if i == 0:
            lab.gpu_rule(f"pass {tag}")
    lab.log(f"[pass {tag}] ΣT {sum(out['T']):.9g} ΣL {sum(out['L']):.9g} n {sum(out['n'])}  rows {len(rows[0])}, "
            f"{time.time() - t0:.0f} s, {lab.peaks()}")
    return out


def state_pass(lab: Any, key: str, rows: tuple[Tensor, Tensor], tag: str) -> dict[str, list[float]]:
    """`pass_rows` with weight set `key` in the model (the masked forward replaces every decomposed matrix, so the
    weights do not enter L)."""
    lab.use(key)
    out = pass_rows(lab, rows, lambda t: importances(lab.model, t, lab.gamma, lab.comps), tag)
    lab.use("W0")
    return out


def w0_rows(lab: Any, name: str, rows: tuple[Tensor, Tensor]) -> dict[str, list[float]]:
    if not lab.get(f"W0_{name}"):
        lab.put(f"W0_{name}", state_pass(lab, "W0", rows, f"W0 on {name}"))
    return lab.get(f"W0_{name}")


def reading(rows: dict[str, list[float]], idx: list[int] | None = None) -> dict[str, Any]:
    """Gain, numerator, r and status from per-row sums (rows `idx`, default all)."""
    s = qr.sums_at(rows, idx if idx is not None else list(range(len(rows["n"]))))
    gain, num = s["T1"] - s["T0"], s["L01"] - s["L00"]
    r = num / gain if abs(gain) >= GAIN_MIN else NAN
    mean = qr.div(gain, s["n"])
    scored = qr.ok(r) and qr.ok(mean) and qr.r9(mean) >= MIN_GAIN
    return {"gain": gain, "num": num, "n": s["n"], "gain_mean": mean, "r": r,
            "status": "scored" if scored else "unreadable"}


def record(w0: dict[str, list[float]], st: dict[str, list[float]], rowset: str) -> dict[str, Any]:
    assert w0["n"] == st["n"], "the state and W0 read different targets"
    rows = {"T0": w0["T"], "T1": st["T"], "L00": w0["L"], "L01": st["L"], "n": st["n"]}
    return {"rowset": rowset, "rows": rows} | reading(rows)


def norm_forward(lab: Any, same: list[bool]) -> Forward:
    """The clean forward with the final RMS norm's weight × NORM_SCALE. Γ reads the 29 taps (the last is layer 27's
    output, before this norm) and each decomposed matrix's input, so its gates must equal W0's bitwise: each row's
    comparison is appended to `same`."""
    w = lab.model.model.norm.weight

    def fwd(t: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        g0 = {s: v.cpu() for s, v in importances(lab.model, t, lab.gamma, lab.comps)[1].items()}
        keep = w.data.clone()
        w.data.mul_(NORM_SCALE)
        try:
            logits, g1 = importances(lab.model, t, lab.gamma, lab.comps)
        finally:
            w.data.copy_(keep)
        same.append(all(torch.equal(g1[s], g0[s].to(DEV)) for s in g1))
        return logits, g1

    return fwd


def kl_check(lab: Any, name: str, rows: Tensor, ref: float) -> float:
    """KL(W0 ‖ the weight set "FT") unmasked on `rows` against `ref` (relative 1e-3; a miss stops)."""
    x = lab.kl_rows("FT", rows)
    lab.check(f"KL_{name}", qr.close(x, ref, CFG.rep_tol), {"kl": x, "ref": ref})
    return x


def controls(lab: Any, sets: dict[str, tuple[Tensor, Tensor]]) -> dict[str, Any]:
    """(a) W0 as the fine-tuned model, (b) the decomposition-free KLs, (c) the final-norm change; each a stop."""
    gold = sets["gold"]
    w0 = w0_rows(lab, "gold", gold)
    lab.set_weights("FT", None)
    a = record(w0, state_pass(lab, "FT", gold, "W0 as FT"), "gold")
    lab.drop_set("FT")
    size = {k: sum(abs(x) for x in a["rows"][k]) for k in ("T0", "L00")}
    lab.check("W0", abs(a["gain"]) <= CFG.id_tol * size["T0"] and abs(a["num"]) <= CFG.id_tol * size["L00"],
              {k: a[k] for k in ("gain", "num", "n", "r")} | {"sum_abs": size})
    kls = {}
    for name, rows in (("gold_s0", gold[0]), ("e40", lab.maths(CFG.n16))):
        lab.set_weights("FT", lab.load_dw(name))
        kls[name] = kl_check(lab, name, rows, REF_KL[name])
        lab.drop_set("FT")
    same: list[bool] = []
    c = record(w0, pass_rows(lab, gold, norm_forward(lab, same), f"final norm x{NORM_SCALE}"), "gold")
    lab.check("norm", all(same) and len(same) == N_ROWS and qr.ok(c["r"]) and abs(c["r"]) <= CFG.id_tol,
              {k: c[k] for k in ("gain", "num", "gain_mean", "r")} | {"rows_bitwise": sum(same)})
    return {"W0": a, "kl": kls, "norm": c | {"rows_bitwise": sum(same)}}


def updates(lab: Any, sets: dict[str, tuple[Tensor, Tensor]]) -> None:
    for name, rs in ROWSET.items():
        if not lab.get(f"U_{name}"):
            w0 = w0_rows(lab, rs, sets[rs])
            lab.set_weights("FT", lab.load_dw(name))
            lab.put(f"U_{name}", record(w0, state_pass(lab, "FT", sets[rs], f"{name} on {rs}"), rs))
            lab.drop_set("FT")


def randoms(lab: Any, sets: dict[str, tuple[Tensor, Tensor]]) -> None:
    """E132's random changes (`E132.random_changes`), scaled by E132's saved factor b (no search; the KL on 16 maths
    rows must reproduce E132's within 1e-3), each read on every row set."""
    if all(lab.get(f"R_{n}_{rs}") for n in RANDOM for rs in OWN):
        return
    changes = E132.random_changes(lab)
    for n in RANDOM:
        dW = changes.pop(n)
        for m in dW.values():
            m.mul_(SAVED[n]["b"])
        lab.set_weights("FT", dW)
        del dW
        kl_check(lab, n, lab.maths(CFG.n16), SAVED[n]["kl"])
        for rs, rows in sets.items():
            if not lab.get(f"R_{n}_{rs}"):
                st = state_pass(lab, "FT", rows, f"{n} on {rs}")
                lab.put(f"R_{n}_{rs}", record(w0_rows(lab, rs, rows), st, rs) | {"b": SAVED[n]["b"]})
        lab.drop_set("FT")


def card_rule(states: dict[str, dict[str, Any]], diff95: dict[tuple[str, str], list[float]]) -> str:
    """The P3.2 card's rule (predictions_amend1.json "card_rule"). diff95[(y, x)]: 95% interval of r_y − r_x."""
    r = {n: qr.r9(s["r"]) for n, s in states.items() if s["status"] == "scored" and qr.ok(s["r"])}
    if len(r) < 2:
        return "unreadable"
    if min(r.values()) >= CEIL and qr.r9(max(r.values()) - min(r.values())) <= GAP:
        return "ceiling"
    for x, y in permutations(r, 2):
        lo, hi = diff95[(y, x)]
        if r[x] < CEIL and qr.r9(r[y] - r[x]) > GAP and qr.ok(lo) and qr.ok(hi) and (lo > 0 or hi < 0):
            return "room"
    return "no call"


def random_rule(rec: dict[str, Any]) -> str:
    return "readable" if qr.ok(rec["gain_mean"]) and qr.r9(rec["gain_mean"]) >= MIN_GAIN else "unreadable"


def summary(recs: dict[str, dict[str, Any]], rand: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Row-resample draws per row set (seed = its index): pairs on one row set are paired, others independent."""
    draws = {rs: qr.resample(N_ROWS, CFG.boot, seed=k) for k, rs in enumerate(OWN)}
    boots = {n: [reading(rec["rows"], i)["r"] for i in draws[rec["rowset"]]] for n, rec in recs.items()}
    keys = ("rowset", "r", "status", "gain", "num", "n", "gain_mean")
    states: dict[str, dict[str, Any]] = {n: {k: rec[k] for k in keys} | {"r95": qr.interval(boots[n])}
                                         for n, rec in recs.items()}
    diff95 = {(y, x): qr.interval([p - q for p, q in zip(boots[y], boots[x], strict=True)])
              for x, y in permutations(recs, 2)}
    a, b = states.get("a2long_s0"), states.get("a2long_s1")
    pair = abs(a["r"] - b["r"]) if a and b and a["status"] == b["status"] == "scored" else None
    return {"states": states, "seed_pair_diff": pair, "diff95": {f"{y} - {x}": v for (y, x), v in diff95.items()},
            "card": card_rule(states, diff95) if len(states) == len(ROWSET) else "incomplete",
            "random": {n: {k: rec[k] for k in keys} | {"reference": random_rule(rec)} for n, rec in rand.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--e40", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    a = ap.parse_args()
    assert decomp_id(Path(a.decomp)) == E132.NEW, a.decomp
    RESULTS.mkdir(parents=True, exist_ok=True)
    lab = L.Lab("E144", a.e40, a.decomp)
    sets = row_sets()
    lab.log("[rows] " + json.dumps({n: int(f[:, 1:].sum()) for n, (_, f) in sets.items()}) + " targets")
    with torch.no_grad():
        if not lab.get("controls"):
            lab.put("controls", controls(lab, sets))
        lab.log("STAGE 1 DONE")
        updates(lab, sets)
        lab.log("STAGE 2 DONE")
        randoms(lab, sets)
        lab.log("STAGE 3 DONE")
        recs = {n: lab.get(f"U_{n}") for n in ROWSET}
        rand = {f"{n}_{rs}": lab.get(f"R_{n}_{rs}") for n in RANDOM for rs in OWN}
        lab.put("summary", summary(recs, rand))
    lab.log(f"STAGE 4 DONE [E144 done] {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()
