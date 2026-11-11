"""E151 (disagreeing_pairs), cell P2.17: new built pairs of 67M fine-tunes where the original model's prior and strength-1
steering disagree (or the prior is matched and steering flat), and which of the two the re-gating share's direction goes
with. Plan reports/plans/2026-11-11_plan_p2-17-disagreeing-pairs.md, appendix 2026-11-11_plan_p2-17-appendix.md; filed rule
results/predictions.json (the run refuses unless it is committed unchanged), applied by rules.py; the readers and states in
job.py. Sides are named as built; no verdict on reuse.

  stage 0  controls, alone, exit 1 on a miss: windows and saved updates against their sha256; a zero update steers by 0 at
           strength 1 and its share is unreadable (gain 0); the five saved maps' priors reproduce prior_vs_share's; copy vs
           prior-matched cipher (saved states) reproduces E147's shares and E149's strength-1 steering, and its condition
           reads "condition not met"; prior-matched cipher seed 0 refit reproduces E147's share within its seed range. Then
           the two new tables (common-word cipher, one-word map) from W0's probabilities, each prior in its filed range.
  stage 1  six new fine-tunes (step 30, saved bf16 in results/large/), strength-1 steering on the 18 states; the condition
           per pair; exit 1 before the share if fewer than 2 of pairs A, B, D meet one.
  stage 2  the share on the 18 states (saved ones against E147's), then the score.
Each stage writes results/stage<n>.json (Amendment 1, amend1.py: amend1_stage<n>.json) and prints `STAGE n DONE`; every
scored number is printed too.

    sbatch --comment=E151 cluster/disagreeing_pairs.sbatch
"""

from __future__ import annotations

import math
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rules
import torch
from job import (
    CONTROL,
    EXP,
    HF,
    LARGE,
    NEW,
    PAIRS,
    PRED,
    PS,
    REPO,
    S1,
    SAVED,
    SEEDS,
    STEP,
    Clock,
    Run,
    Stop,
    check,
    load,
    log,
    prior_gap,
    say,
    share_seeds,
    state,
    steer_seeds,
    task,
    update_file,
)

from commute import anchors as AN
from commute import fair_steering as FS
from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute.data import Windows
from commute.jobs import hard_exit, load_experiment, require_filed

A6 = load_experiment("open_pair", "amend6")  # save_map: E58 Amendment 6's windows with a new table, cut by key
A3_DIR = EXP / "open_pair/results/large/amend3"
Stage = Callable[[Run, dict[str, Any], float], None]


@dataclass(frozen=True)
class Maps:
    """The new tables' filed settings (predictions.json new_maps and stop_rules)."""

    common_ranks: int = 3000  # the common-word pool: ranked word-like tokens 0-2999
    value_band: tuple[float, float] = (5e-4, 3e-3)  # W0 geometric-mean P(v | M k)
    min_rank: int = 20  # v past rank 20 at k
    median_ratio: float = 3.0  # P(v | M k) <= 3 x its median over the 128 keys
    common_prior: tuple[float, float] = (4.2e-4, 4.2e-3)  # the built map's prior must land here
    one_word_max: float = 1e-5  # one-word: under this after every key, and the built map's prior under it


MAPS = Maps()


# ----------------------------------------------------------------------------- stage 0


def files_control(run: Run, ch: dict[str, Any]) -> None:
    """The saved maps' windows (E58's record) and the saved updates (E147's record), by sha256 (0 = equal)."""
    for m in (*SAVED, "prior_cipher"):
        file, keys = run.F["e147_inv"]["maps"][m]["windows_sha256"]
        want = load(REPO / file)
        for k in keys:
            want = want[k]
        check(ch, f"0 sha256 windows {m}", float(KD.sha256(REPO / run.inv["tasks"][task(m)]["path"]) != want), 0)
        for s in SEEDS:
            got = KD.sha256(update_file(m, s))
            check(ch, f"0 sha256 update {state(m, s)}", float(got != run.F["e147_0"]["states"][state(m, s)]["sha256"]), 0)


def zero_control(run: Run, ch: dict[str, Any], out: dict[str, Any]) -> None:
    """A zero update: strength-1 steering changes no logit, the steering path's gain is 0, the share's gain is 0 (unreadable)."""
    t, tol = run.hf.task(task("copy")), run.th["zero_tol"]
    zero = {k: torch.zeros_like(w) for k, w in run.net.W0.items()}
    vs = FS.vectors(run.net, t.x(t.train), t.idx(t.train), len(t.pool), zero, t.read.half)
    b = t.train[: FS.BATCH]
    lg0 = run.net.run(t.x(b))[0][:, t.read.half]
    steer = max(float((FS.logits_steered(run.net, t.x(b), t.idx(b), t.read.half, vs[kd][l], l, S1) - lg0).abs().max())
                for kd in ("pooled", "perkey") for l in PS.LAYERS)  # fmt: skip
    sh = run.read_share(state("copy", 0), zero)
    out["zero"] = {"steer_max_change": steer, "steer_gain": float((HF.lo_at(run.net, t, t.read.toks, zero) - t.lo0).abs().max()),
                   "share_gain": sh["gain"], "share": sh["share"]}  # fmt: skip
    say("0 zero update", out["zero"])
    check(ch, "0 zero update: steering change at strength 1", steer, tol)
    check(ch, "0 zero update: steering path gain", out["zero"]["steer_gain"], tol)
    check(ch, "0 zero update: share gain", sh["gain"], tol)
    check(ch, "0 zero update: share unreadable (0 = nan)", float(not math.isnan(sh["share"])), 0)


def saved_priors(run: Run, ch: dict[str, Any]) -> None:
    saved = {p["name"]: p["prior"] for p in run.F["priors"]["points"]}
    for m in (*SAVED, "prior_cipher"):
        p, ref = run.map_prior(m), saved[f"e58_96_{m}"]
        log(f"[0 prior {m}] {p:.6e} (saved {ref:.6e})")
        check(ch, f"0 prior {m} (relative)", p / ref - 1, run.th["reproduce_tol"])


def positive_control(run: Run, ch: dict[str, Any], out: dict[str, Any]) -> None:
    """Copy vs prior-matched cipher on saved states: every share and steering read as saved; the condition refuses the pair."""
    tol, rec = run.th["reproduce_tol"], {}
    for m in CONTROL:
        for s in SEEDS:
            n = state(m, s)
            sh, st = run.shared(n)["share"], run.steering(n)["pooled"]["share"]
            rec[n] = {"share": sh, "steering": st, "saved_share": run.F["e147_0"]["states"][n]["r"],
                      "saved_steering": run.F["e149"]["states"][n]["pooled"]["share"]}  # fmt: skip
            say(f"0 positive control {n}", rec[n])
            check(ch, f"0 positive control share {n}", sh - rec[n]["saved_share"], tol)
            check(ch, f"0 positive control steering {n}", st - rec[n]["saved_steering"], tol)
    calls = rules.pair_calls({"prior_gap": prior_gap(run, *CONTROL), "steering": (steer_seeds(run, "copy"), steer_seeds(run, "prior_cipher"))}, run.th)
    calls["share"] = rules.CALL(share_seeds(run, "copy"), share_seeds(run, "prior_cipher"), run.th)
    out["positive_control"] = {"states": rec, "calls": calls}
    say("0 positive control calls", calls)
    check(ch, "0 positive control condition is 'condition not met' (0 = yes)", float(calls["condition"] != "condition not met"), 0)


def refit_control(run: Run, ch: dict[str, Any], out: dict[str, Any]) -> None:
    """Prior-matched cipher seed 0 fine-tuned again from its windows (E147's recipe), read in bf16 as E147 saved it."""
    n = state("prior_cipher", 0)
    w = torch.load(REPO / run.inv["tasks"][task("prior_cipher")]["path"], weights_only=False)
    ft = KD.finetune(run.ctx, w, 0, (STEP,))
    dW = {k: v.bfloat16().float().to(run.net.W0[k].device) for k, v in ft.deltas[STEP].items()}
    out["refit"] = {**run.read_share(n, dW), "saved": run.F["e147_0"]["states"][n]["r"], "p_heldout": ft.evals[STEP]["p_o"]}
    say("0 refit prior_cipher_s0", out["refit"])
    check(ch, "0 refit share prior_cipher_s0", out["refit"]["r"] - out["refit"]["saved"], run.th["refit_tol_prior_cipher"])


def draw_common(G: torch.Tensor, keys: list[int], pool: list[int], marker: int, out: dict[str, Any]) -> kv.KVTask | None:
    """The common-word cipher: kv.bijection (seed 0) with each key's value in the band, past its rank 20, and at most 3x
    its median over the keys."""
    P = G[:, pool]
    kth = G.topk(MAPS.min_rank, dim=1).values[:, -1:]
    ok = (P >= MAPS.value_band[0]) & (P <= MAPS.value_band[1]) & (P < kth) & (P <= MAPS.median_ratio * G.quantile(0.5, dim=0)[pool])
    row, col = {k: i for i, k in enumerate(keys)}, {v: j for j, v in enumerate(pool)}
    out["common_pool"] = {"n_pool": len(pool), "n_words_admissible": int(ok.any(0).sum()), "per_key_min": int(ok.sum(1).min()),
                          "per_key_median": float(ok.sum(1).float().median())}  # fmt: skip
    say("0 common-word pool", out["common_pool"])
    try:
        return kv.bijection("common_cipher", marker, tuple(keys), pool, seed=0, ok=lambda k, v: bool(ok[row[k], col[v]]))
    except AssertionError:
        return None


def draw_one_word(G: torch.Tensor, keys: list[int], pool: list[int], marker: int) -> kv.KVTask | None:
    """The first word of E58's cipher pool, in the seeded (0) order kv.bijection draws, under the bar after every key."""
    try:
        first = kv.bijection("one_word", marker, (keys[0],), pool, seed=0, ok=lambda _, v: float(G[:, v].max()) < MAPS.one_word_max)
    except AssertionError:
        return None
    return kv.KVTask("one_word", marker, tuple(keys), (first.values[0],) * len(keys))


def build_tables(run: Run, ch: dict[str, Any], out: dict[str, Any]) -> None:
    """Amendment 3's windows and split; W0's probabilities per key; the two tables, saved and cut by key; their priors."""
    a3 = run.F["a3"]
    check(ch, "0 sha256 windows_elicit.pt", float(KD.sha256(A3_DIR / "windows_elicit.pt") != a3["maps"]["elicit"]["sha256"]["full"]), 0)
    keys = a3["keys"]["keys"]
    trained, held = OP.split_keys(keys, len(keys) // 4, 0)
    check(ch, "0 key split is Amendment 3's (0 = yes)", float(held != a3["keys"]["held"]), 0)
    src = torch.load(A3_DIR / "windows_elicit.pt", weights_only=False)
    tr, ho = src["train"], src["heldout"]
    both = Windows(torch.cat([tr.tokens, ho.tokens]), torch.cat([tr.g, ho.g]), torch.cat([tr.row, ho.row]), torch.cat([tr.pos, ho.pos]), tr.half)
    geo = kv.key_distributions(lambda x: run.net.run(x)[0], both, run.net.device, geometric=True)
    G = torch.stack([geo[k] for k in keys])
    _, _, _, ranked, tok = KD.vocab(run.ctx)
    banned, marker = set(keys) | set(a3["maps"]["elicit"]["values"]), src["task"].marker
    made = {"common_cipher": draw_common(G, keys, [v for v in ranked[: MAPS.common_ranks] if v not in banned], marker, out),
            "one_word": draw_one_word(G, keys, [v for v in ranked[AN.VALUE_RANK0 : AN.VALUE_RANK0 + AN.POOL] if v not in banned], marker)}  # fmt: skip
    check(ch, "0 word pools not exhausted (number of maps without a table)", float(sum(m is None for m in made.values())), 0)
    tables = {n: m for n, m in made.items() if m is not None}
    rec: dict[str, Any] = {"maps": {}, "tables": {"letter_of": {str(k): tok.decode([k]).strip()[:1].lower() for k in keys}}}
    for name, m in tables.items():
        A6.save_map(run.ctx, src, m, "ranks 0-2999" if name == "common_cipher" else "cipher_pool", trained, held, LARGE, rec)
        run.add_new(name)
        log(f"[0 table {name}] first values {[tok.decode([v]) for v in m.values[:12]]} prior {run.map_prior(name):.6e}")
    out["tables"] = {n: {**rec["maps"][n], "prior": run.prior[n], "keys": list(m.keys)} for n, m in tables.items()}
    say("0 priors and gaps (decades)", {**run.prior, **{f"gap {p}": prior_gap(run, *ms) for p, ms in PAIRS.items()}})
    for n in NEW:
        check(ch, f"0 table {n}: zero update reads 0", rec["maps"][n]["zero_max_abs"], 0)
    lo, hi = MAPS.common_prior
    check(ch, f"0 table common_cipher prior in [{lo:g}, {hi:g}] (0 = yes)", float(not lo <= run.prior["common_cipher"] <= hi), 0)
    check(ch, f"0 table one_word prior under {MAPS.one_word_max:g} (0 = yes)", float(not run.prior["one_word"] < MAPS.one_word_max), 0)


def controls(run: Run, out: dict[str, Any]) -> dict[str, Any]:
    """Stage 0's controls, in run 1's order (Amendment 1 runs them unchanged); the checks record."""
    out |= {"host": socket.gethostname(), "card": torch.cuda.get_device_name(), "checks": {}}
    ch = out["checks"]
    files_control(run, ch)
    zero_control(run, ch, out)
    saved_priors(run, ch)
    positive_control(run, ch, out)
    refit_control(run, ch, out)
    return ch


def stage0(run: Run, out: dict[str, Any], _t_job: float) -> None:
    build_tables(run, controls(run, out), out)


# ----------------------------------------------------------------------------- stage 1


def train_new(run: Run, m: str, s: int, out: dict[str, Any]) -> None:
    """keydep.finetune (E54's recipe, as every E58 map and E147's refit), the update at step 30 saved in bf16."""
    w = torch.load(REPO / run.inv["tasks"][task(m)]["path"], weights_only=False)
    ft = KD.finetune(run.ctx, w, s, (STEP,))
    path = update_file(m, s)
    torch.save({k: v.bfloat16() for k, v in ft.deltas[STEP].items()}, path)
    out["updates"][state(m, s)] = {"file": str(path.relative_to(REPO)), "sha256": KD.sha256(path),
                                   "p_heldout": ft.evals[STEP]["p_o"], "elapsed_s": ft.evals[STEP]["elapsed_s"]}  # fmt: skip


def steer_record(run: Run, name: str, out: dict[str, Any]) -> dict[str, Any]:
    rec = out["states"][name] = run.steering(name)
    log(f"[1 steering {name}] gain {rec['gain']:.4f} " + " ".join(f"{kd} layer {rec[kd]['layer']} share {rec[kd]['share']:.4f}" for kd in ("pooled", "perkey") if kd in rec))
    return rec


def conditions(run: Run) -> dict[str, Any]:
    return {p: rules.pair_calls(run.pair(p), run.th) for p in PAIRS}


def stage1(run: Run, out: dict[str, Any], t_job: float) -> None:
    out |= {"updates": {}, "states": {}, "stopped_maps": {}}
    clock = Clock(t_job, (len(NEW) + len(SAVED)) * len(SEEDS), run.th["projected_limit_min"])
    for m in NEW:
        for s in SEEDS:
            train_new(run, m, s, out)
            gain = steer_record(run, state(m, s), out)["gain"]
            run.write(1, out)
            clock.tick(out)
            if not gain >= run.th["gain_floor"]:
                out["stopped_maps"][m] = f"seed {s} gain {gain:.4g} under the floor {run.th['gain_floor']}"
                log(f"[1 map stopped] {m}: {out['stopped_maps'][m]}")
                break
    for m in SAVED:
        for s in SEEDS:
            steer_record(run, state(m, s), out)
            run.write(1, out)
            clock.tick(out)
    out["conditions"] = conditions(run)
    for p, c in out["conditions"].items():
        log(f"[1 condition {p}] {c['condition']}: prior gap {c['prior']['gap']:.3f} decades (call {c['prior']['call']}), steering "
            f"{c['steering']['call']} gap {c['steering']['gap']:.4f} spread {c['steering']['spread']:.4f} premise {c['premise']}")  # fmt: skip
    out["n_meeting"] = k = rules.n_meeting({p: c["condition"] for p, c in out["conditions"].items()})
    log(f"[1 stop rule] {k} of pairs A, B, D meet a condition (need {run.th['min_counted_pairs']})")
    if k < run.th["min_counted_pairs"]:
        raise Stop(f"could not be built: {k} of A, B, D meet a condition")


# ----------------------------------------------------------------------------- stage 2


def stage2(run: Run, out: dict[str, Any], t_job: float) -> None:
    out |= {"shares": {}, "checks": {}}
    names = [state(m, s) for m in (*NEW, *SAVED) for s in SEEDS if state(m, s) in run.steer]  # a stopped map's later seeds were never trained
    clock = Clock(t_job, len(names), run.th["projected_limit_min"])
    for n in names:
        rec = out["shares"][n] = run.shared(n)
        log(f"[2 share {n}] r {rec['r']:.6f} gain {rec['gain']:.4f} share {rec['share']:.6f}")
        if not n.startswith("new/"):
            check(out["checks"], f"2 share reproduces E147 {n}", rec["r"] - run.F["e147_0"]["states"][n]["r"], run.th["reproduce_tol"])
        run.write(2, out)
        clock.tick(out)
    pairs = {p: {**run.pair(p), "share": (share_seeds(run, r), share_seeds(run, nw))} for p, (r, nw) in PAIRS.items()}
    out["inputs"] = pairs
    out["score"] = rules.score(pairs, run.th)
    out["filed"] = rules.filed(out["score"])
    for p, r in out["score"]["pairs"].items():
        say(f"2 score {p}", {k: r[k] for k in ("condition", "reading", "prior", "steering", "share")})
    say("2 outcome", {k: out["score"][k] for k in ("counted_opposite", "outcome", "flat")})
    say("2 filed predictions (true = as leaned)", out["filed"])


def main(first: Stage = stage0, files: str = "stage", filed: tuple[Path, ...] = (PRED,)) -> int:
    """Stage `first` (run 1: stage0; Amendment 1 passes its own), then stages 1 and 2; results/<files><n>.json; every
    file in `filed` must be committed unchanged."""
    t_job = time.time()
    for f in filed:
        require_filed(f)
    P = load(PRED)
    assert P["experiment"].startswith("E151") and sorted(P["pairs"]) == sorted(PAIRS), (P["experiment"], sorted(P["pairs"]))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E58 trained and E136/E143 read
    run = Run(P, files)
    log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()}; model loaded at {(time.time() - t_job) / 60:.1f} min")
    outs: list[dict[str, Any]] = [{}, {}, {}]
    for n, stage in enumerate((first, stage1, stage2)):
        try:
            stage(run, outs[n], t_job)
        except Stop as e:
            outs[n]["stopped"] = str(e)
            run.write(n, outs[n])
            log(f"STOP: {e}: tell the lead")
            return 1
        run.write(n, outs[n])
        log(f"STAGE {n} DONE")
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; {(time.time() - t_job) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    hard_exit(main())
