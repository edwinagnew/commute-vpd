"""E80 Amendment 2: the Qwen3-1.7B column of the slide's scorecard, on saved u1. Plan: reports/plans/2026-11-04_plan_e80-adversary-do-nothing.md, Amendment 2; rules:
rules_amend2.py, results/predictions_amend2.json (the job refuses to start without it, and while results/stopped_amend2.json exists).

Starts LS s0, ADJ s0, LS s1, ADJ s1 on W, by E80's harness unchanged (run.py, through amend1.read), seeds paired with Amendment 1's W0, N and A, read from results/amend1.json
and its card checked against this one. Before them, cheap and control first, the scales and the no-mask gain reads: W0, FT, N, LS, ADJ, A, W0 again. These stop the job in its
first minutes if the states are not what they should be. A's offsets come from the W&B inputs that set_up fetches and keeps; Amendment 1 saved nothing A needs. X is make_x's, kept
on the pod's disk beside --decomp, rebuilt if absent. Importing amend1 sets run's globals to its files, so they are set again here. All results in results/amend2.json
(resumable per read), the stop in stopped_amend2.json, every line in results/run_amend2.log.

Usage (repo root, 80 GB GPU): .venv/bin/python experiments/qwen_d1_adversary/amend2.py --decomp /root/qwen --seed 0 [--host-check]. Two pods of one card model, the user's wish for speed:
one runs seed 0 with --host-check, the other seed 1 from the first's results (scales and gain are kept); --merge A B then joins their two results on any machine, with Amendment 1's last reads.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import amend1
import rules_amend2 as rules
import run
import torch
from transformers import AutoTokenizer

from commute import qwen_eval, qwen_maps, qwen_sets, split
from commute.qwen_sft import Seq, gold_loss, load, pack, sft_seq
from commute.qwen_teacher import change_hooks
from commute.qwen_vpd import SITES

e76 = run.e76
DEV, F64 = run.DEV, run.F64
PRED, OUT, STOPPED, LOG = (run.RESULTS / f for f in ("predictions_amend2.json", "amend2.json", "stopped_amend2.json", "run_amend2.log"))
A1_OUT, A1_LOG = run.RESULTS / "amend1.json", run.RESULTS / "run_amend1.log"
run.OUT, run.STOPPED, run.LOG = OUT, STOPPED, LOG
ORDER = (("LS", 0), ("ADJ", 0), ("LS", 1), ("ADJ", 1))
SIGMA_BATCH = 2  # qwen_maps/job.py Config.eval_batch, E68's Σ
HELD, SEQ_LEN, PERM_SEED, GOLD_BATCH = 256, 512, 0, 8  # qwen_sft/run.py Config: heldout, seq_len, seed, eval_batch
U1_PROBLEMS, TASK_ROWS = 640, 256  # u1: 10 steps of batch 64; the 67M's Σ took the first 256 training rows (scorecard_ls/run.py)


def change(J: run.Job, name: str, c: dict[str, float] | None = None) -> Callable[[str], qwen_maps.SiteChange]:
    """site -> SiteChange of "ls" (X from make_x's files) or "adjoint" (`c`, by default the table's from `scales`), as qwen_maps/job.py's `change` builds them for E68."""

    def fn(s: str) -> qwen_maps.SiteChange:
        U_A, V_A = J.alive.uv(s, DEV, F64)
        M = J.dw[s].to(DEV, F64)
        if name == "ls":
            return qwen_maps.site_change(name, M, U_A, V_A, load(J.x_dir / f"{s}.safetensors", DEV).__getitem__, 0.0, 0.0)
        return qwen_maps.site_change(name, M, U_A, V_A, None, (c or J.out["c"])[s], 0.0)

    return fn


@contextmanager
def ls_state(J: run.Job) -> Iterator[None]:
    """Least squares, the full split: weights W0 + δW, every alive column moved by (dU, dV), Δ takes the rest."""
    run.make_x(J)
    with qwen_maps.applied(J.model, J.comps, J.alive, change(J, "ls")):
        yield


@contextmanager
def adj_state(J: run.Job) -> Iterator[None]:
    """The scaled adjoint split: weights W0 + c dσ(M V, Mᵀ U), every alive column moved by c (M V, Mᵀ U), c per site from `scales`."""
    with qwen_maps.applied(J.model, J.comps, J.alive, change(J, "adjoint")):
        yield


amend1.STATE |= {"LS": ls_state, "ADJ": adj_state}


def moved(c: qwen_maps.SiteChange, U_A: Any, V_A: Any) -> Any:
    """Δ′ − Δ of a state whose columns move by (dU, dV) and whose weights by dW: dW − ((U + dU)(V + dV)ᵀ − U Vᵀ)."""
    dW, _, dU, dV = c
    assert dW is not None and dU is not None and dV is not None
    return dW - split.dsigma(dU, dV, U_A, V_A) - dU @ dV.T


@torch.no_grad()
def scales(J: run.Job, task: list[Seq]) -> None:
    """Two Σ. E68's (the 128 maths rows at W0) and its c give E68's three numbers, the control that Σ, c and X are its states. The table's c is the 67M's recipe, E71's and E39's:
    Σ on the update's first training rows, whole rows of `task` packed as the fine-tune packs them (qwen_sft/run.py `Job.maths`), up to 256. What each state leaves in Δ is logged two
    ways: E71's measure (plain ‖Δ′ - Δ‖² / ‖δW‖², pooled), and E68's (‖Δ′ - Δ‖_Σ / ‖δW‖_Σ on E68's Σ, pooled, the source of Qwen's 36%), which uses E68's c."""
    if "c" in J.out:
        return
    t0 = time.time()
    rows = torch.tensor(json.loads(e76.MATHS.read_text())["rows"], device=DEV)
    sigma = qwen_maps.input_moments(J.model, list(rows.split(SIGMA_BATCH)), list(SITES))
    run.log(f"[sigma E68] {len(sigma)} sites, {time.time() - t0:.0f} s, {run.peak()}")
    run.make_x(J)
    c68: dict[str, float] = {}
    weighted = {"ls": 0.0, "adjoint": 0.0}
    size, energy = 0.0, 0.0
    for s in SITES:
        U_A, V_A = J.alive.uv(s, DEV, F64)
        M, S = J.dw[s].to(DEV, F64), sigma[s].to(DEV)
        c68[s] = split.adjoint_scale(M, U_A, V_A, S)[0]
        size, energy = size + float((M * M).sum()), energy + split.weighted_energy(M, S)
        for n in weighted:
            weighted[n] += split.weighted_energy(moved(change(J, n, c68)(s), U_A, V_A), S)
    del sigma
    torch.cuda.empty_cache()
    sig = {n: (weighted[n] / energy) ** 0.5 for n in weighted}
    want, tol = rules.E68_S, rules.E68_TOL
    for name, got, key in (("energy", energy, "energy"), ("LS_sigma", sig["ls"], "LS"), ("ADJ_sigma", sig["adjoint"], "ADJ")):
        run.check(J, name, got, rules.near_rel(got, want[key], tol), f"E68's {want[key]} within {tol:.0%}")
    packed = torch.tensor(pack([ids for ids, _ in task], None, SEQ_LEN)[:TASK_ROWS], device=DEV)
    sigma = qwen_maps.input_moments(J.model, list(packed.split(SIGMA_BATCH)), list(SITES))
    run.log(f"[sigma task] {len(packed)} rows of {SEQ_LEN}, {time.time() - t0:.0f} s, {run.peak()}")
    c: dict[str, float] = {}
    plain = {"ls": 0.0, "adjoint": 0.0}
    for s in SITES:
        U_A, V_A = J.alive.uv(s, DEV, F64)
        c[s] = split.adjoint_scale(J.dw[s].to(DEV, F64), U_A, V_A, sigma[s].to(DEV))[0]
        for n in plain:
            d = moved(change(J, n, c)(s), U_A, V_A)
            plain[n] += float((d * d).sum())
    del sigma
    torch.cuda.empty_cache()
    J.out["left_in_delta"] = {n: {"plain_energy": plain[n] / size, "sigma_norm_E68": sig[n]} for n in plain}
    J.out["energy"], J.out["c_E68"], J.out["c"] = energy, c68, c
    run.save(J)
    ratios = [c[s] / c68[s] for s in SITES if c68[s]]
    run.log(f"[left in delta] {json.dumps(J.out['left_in_delta'])}, ‖δW‖² {size}, ‖δW‖²_Σ {energy}, c {min(c.values())} to {max(c.values())}, c / E68's c {min(ratios)} to {max(ratios)}, {time.time() - t0:.0f} s, {run.peak()}")


def gold_seqs() -> tuple[list[Seq], list[Seq]]:
    """E40's split of GSM8K train, as its SFT saw it (qwen_sft/run.py `Job.__init__`, `seq`, `held`): the 256 held-out problems, whose W0 reading is the control, and u1's training problems, in order."""
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    end = int(tok.convert_tokens_to_ids("<|im_end|>"))
    train = qwen_sets.load()["train"]
    perm = torch.randperm(qwen_sets.N_TRAIN, generator=torch.Generator().manual_seed(PERM_SEED)).tolist()
    seqs = [sft_seq(tok, train[i]["question"], train[i]["gold"], end, SEQ_LEN) for i in perm[:HELD + U1_PROBLEMS]]
    return seqs[:HELD], seqs[HELD:]


def gold(J: run.Job, seqs: list[Seq], name: str, state: Callable[[run.Job], Any]) -> float:
    """The mean response-token log-loss of the state's weights, no masks (`gold_loss`, fp32). Written and logged as it ends; a rerun keeps what is written."""
    reads = J.out.setdefault("gold", {})
    if name not in reads:
        t0 = time.time()
        with state(J):
            reads[name] = gold_loss(J.model, seqs, GOLD_BATCH, DEV)
        run.save(J)
        run.log(f"[gold {name}] {reads[name]:.6f} nats per response token, {time.time() - t0:.0f} s, {run.peak()}")
    return reads[name]


def stage_gain(J: run.Job, seqs: list[Seq]) -> None:
    """Controls first: W0 and FT against E40's filed readings, N and LS at 1.00, then the two that are read, then W0 again, which must read 0 (the weights are back)."""
    w0 = gold(J, seqs, "W0", lambda J: nullcontext())
    run.check(J, "gold_W0", w0, rules.near_rel(w0, rules.W0_GOLD, rules.W0_TOL), f"E40's {rules.W0_GOLD} within {rules.W0_TOL:.1%}")
    ft = gold(J, seqs, "FT", lambda J: change_hooks(J.model, J.dw.__getitem__))
    run.check(J, "gold_FT", w0 - ft, rules.near_rel(w0 - ft, rules.FT_GAIN, rules.FT_TOL), f"E40's gain {rules.FT_GAIN} within {rules.FT_TOL:.0%}")
    for name, state in (("N", run.n_state), ("LS", ls_state)):
        s = rules.kept(gold(J, seqs, name, state), w0, ft)
        run.check(J, f"kept_{name}", s, rules.near(s, 1.0, rules.KEPT_TOL), f"1.00 within {rules.KEPT_TOL}")
    for name, state in (("ADJ", adj_state), ("A", run.a_state)):
        run.log(f"[kept {name}] {rules.kept(gold(J, seqs, name, state), w0, ft):.6f}")
    s = rules.kept(gold(J, seqs, "W0_end", lambda J: nullcontext()), w0, ft)
    run.check(J, "kept_W0_end", s, rules.near(s, 0.0, rules.KEPT_TOL), f"0.00 within {rules.KEPT_TOL}")


def host_check(J: run.Job) -> None:
    """Three steps of W0 at seed 0 on this host, against Amendment 1's log for the same seed on the same card model: logged, not ruled on, reported to the lead. Reads are logged to 4 places."""
    if "host_check" in J.out:
        return
    reads: list[str] = []
    run.adversary(J.model, J.rows["W"], J.gamma, J.comps, 0, log=reads.append, steps=3)
    there = re.findall(r"\[W0_W_s0\] seed 0 step [012]: KL (\S+)", A1_LOG.read_text())
    J.out["host_check"] = {"here": [r.rsplit(" ", 1)[1] for r in reads if " step " in r][:3], "amend1": there[:3]}
    run.save(J)
    run.log(f"[host check] {json.dumps(J.out['host_check'])}")


def amend1_starts() -> dict[str, Any]:
    """Amendment 1's starts on this pod's card, with no E80 value in them: its --reread-s0 put W0 s0 and N s0 in "starts". W0's two seeds are what every ratio divides by."""
    out = e76.read_json(A1_OUT)
    assert out and all(f"W0_W_s{s}" in out["starts"] for s in (0, 1)), "amend1.json lacks W0's two seeds"
    return {k: {"kl": v["kl"]} for k, v in out["starts"].items()}


def same_card() -> None:
    cards = re.findall(r"\[load\] (.+?), GPU peak", A1_LOG.read_text())
    here = torch.cuda.get_device_name()
    run.log(f"[card] {here}; Amendment 1's {cards[-1:]}")
    if cards[-1:] != [here]:
        run.stop("same_card", f"this card is {here}, Amendment 1 read W0 on {cards[-1:]}: the ratios would not share a card")


def rescore(J: run.Job) -> None:
    J.out["score"] = rules.score({**J.out["a1"], **J.out["starts"]}, J.out.get("gold", {}), J.out["checks"])
    run.save(J)
    run.log(f"[score] {json.dumps(J.out['score'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp")
    ap.add_argument("--seed", type=int, choices=(0, 1), help="only this seed's starts; both if absent")
    ap.add_argument("--host-check", action="store_true")
    ap.add_argument("--merge", nargs=2, type=Path, metavar=("WITH_SCALES", "OTHER"))
    a = ap.parse_args()
    if a.merge:
        merge(*a.merge)
        return
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions_amend2.json missing, or stopped_amend2.json exists")
    a1 = amend1_starts()
    same_card()
    seqs, task = gold_seqs()
    J = run.set_up(Path(a.decomp).expanduser().resolve())
    J.out["a1"] = a1
    run.save(J)
    if a.host_check:
        host_check(J)
    scales(J, task)
    stage_gain(J, seqs)
    rescore(J)
    for state, seed in ORDER:
        if a.seed is not None and seed != a.seed:
            continue
        amend1.read(J, state, seed)
        rescore(J)
    run.log(f"[done] {run.peak()}")


def merge(first: Path, other: Path) -> None:
    """Joins two pods' results into results/amend2.json and scores them, with Amendment 1's last reads. A start's key is its state and seed, so the two pods cannot collide."""
    out, more = json.loads(first.read_text()), json.loads(other.read_text())
    assert not set(out["starts"]) & set(more["starts"]), "a start read twice"
    out["starts"] |= more["starts"]
    out["checks_second_pod"] = more["checks"]
    out["a1"] = amend1_starts()
    e76.write_json(OUT, out)
    score()


def score() -> None:
    """From what amend2.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(OUT)
    if out is None:
        return
    out["score"] = rules.score({**out.get("a1", {}), **out["starts"]}, out.get("gold", {}), out["checks"])
    e76.write_json(OUT, out)
    run.log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    run.guarded(main, score)
