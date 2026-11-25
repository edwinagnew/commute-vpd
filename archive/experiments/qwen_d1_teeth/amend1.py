"""E85 Amendment 1: positive controls. Can VPD's adversary fail anything on Qwen3-1.7B? One 80 GB card, ~73 min. Plan: reports/plans/2026-11-04_plan_e85-adversary-teeth.md, "Amendment 1";
rules: rules_amend.py, results/predictions_amend1.json (the job refuses to start without it, and while results/stopped_amend1.json exists).

E80 Amendment 2's code path as E85 used it: set_up and start through its amend1.read, the same 128 web rows, 20 sign steps of 0.1, seed 0, every step's KL in the log with its ratio and excess over W0.
Control first: W0's gold, its read at m = g with Δ off, and a three-step host check against the reused 1.703 (a W0 start instead on any other card model, whose final read has differed by 2.8%).
Stage 1, Rand100: E85's Gaussian times 100 in Δ. Stage 2, only if stage 1 reaches the fail bar: W0 with U's columns permuted, among all alive columns or (--perm rare) the rarely-on half. Before each start
the state's clean reads, needing no adversary: gold loss, KL against W0 on the rows, and KL at m = g with Δ off. Results in results/amend1.json after every read; a rerun resumes at the next unread one.

Usage (repo root, 80 GB GPU; inputs beside --decomp, or WANDB_API_KEY to fetch them): .venv/bin/python experiments/qwen_d1_teeth/amend1.py --decomp /root/qwen [--perm rare]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import rules_amend as rules
import run as e85
import torch
from torch import Tensor

from commute import qwen_maps
from commute.qwen_sft import load, mean_kl
from commute.qwen_vpd import SITES, importances, masked_logits
from commute.vpd import kl_per_position

amend2, amend1, e80, e76 = e85.amend2, e85.amend1, e85.e80, e85.e76
r2 = rules.r2
RESULTS = Path(__file__).resolve().parent / "results"
PRED, OUT, STOPPED, LOG, E85_LOG = (RESULTS / f for f in ("predictions_amend1.json", "amend1.json", "stopped_amend1.json", "run_amend1.log", "run.log"))
SCOPE = {"all": "Perm", "rare": "PermRare"}
RUNNING: dict[str, Any] = {"w0": [], "clean": rules.NAN}  # read by `guard`, which E80's `start` calls through its log


def rand100_change(J: Any, s: str) -> qwen_maps.SiteChange:
    G = e85.rand_change(J, s)[0]
    assert G is not None
    return rules.RAND_SCALE * G, None, None, None


@contextmanager
def rand100_state(J: Any) -> Iterator[None]:
    with qwen_maps.applied(J.model, J.comps, J.alive, lambda s: rand100_change(J, s)):
        yield


def perm_positions(J: Any, scope: str) -> dict[str, Tensor]:
    """Positions in the alive list: all of them, or the rarely-on half E85's Rare moved (`qwen_maps.halves` on W0's maths statistics)."""
    if scope == "all":
        return {s: torch.arange(len(J.alive.idx[s])) for s in SITES}
    st = load(J.inputs / "W0_maths.safetensors")
    return {s: qwen_maps.halves(st[f"{s}.active"][J.alive.idx[s].cpu()], st[f"{s}.mean_g"][J.alive.idx[s].cpu()])[0] for s in SITES}


def perm_change(J: Any, s: str, pos: Tensor) -> qwen_maps.SiteChange:
    """Weights and V untouched, so clean logits and g stay W0's exactly. U's columns at `pos` are permuted among themselves, so V U is wrong and Δ = W - V U takes the difference."""
    gen = torch.Generator().manual_seed(rules.PERM_SEED + SITES.index(s))
    pi = pos[torch.randperm(len(pos), generator=gen)]
    U_A, V_A = J.alive.uv(s, e85.DEV, e85.F64)
    dU = U_A[:, pi.to(U_A.device)] - U_A[:, pos.to(U_A.device)]
    return None, pos, dU, torch.zeros_like(V_A[:, pos.to(V_A.device)])


def perm_state(scope: str) -> Callable[[Any], Any]:
    @contextmanager
    def state(J: Any) -> Iterator[None]:
        pos = perm_positions(J, scope)
        with qwen_maps.applied(J.model, J.comps, J.alive, lambda s: perm_change(J, s, pos[s])):
            yield

    return state


@torch.no_grad()
def wrong_share(J: Any, scope: str) -> float:
    """‖V(U_π - U)‖² / ‖V U‖², pooled over the sites, alive columns: how wrong the columns are, with no forward."""
    pos, top, bottom = perm_positions(J, scope), 0.0, 0.0
    for s in SITES:
        U_A, V_A = J.alive.uv(s, e85.DEV, e85.F64)
        dU, p = perm_change(J, s, pos[s])[2], pos[s].to(V_A.device)
        assert dU is not None
        top, bottom = top + float((V_A[:, p] @ dU.T).pow(2).sum()), bottom + float((V_A @ U_A.T).pow(2).sum())
    return top / bottom


@torch.no_grad()
def ci_kl(J: Any) -> float:
    """Mean KL(the state's clean ‖ masked at m = g, Δ off) over every position of the rows, the state as it stands. Needs no adversary and no source."""
    total, n = 0.0, 0
    for tokens in J.rows["W"]:
        logits, g = importances(J.model, tokens, J.gamma, J.comps)
        total += float(kl_per_position(logits, masked_logits(J.model, tokens, J.comps, g)).double().sum())
        n += tokens.numel()
    return total / n


def clean_reads(J: Any, seqs: list[Any], name: str) -> dict[str, float]:
    """A state's behaviour without the adversary, so that "known to be unfaithful" has a number of its own."""
    reads = J.out.setdefault("clean", {})
    if name not in reads:
        gold = amend2.gold(J, seqs, name, amend1.STATE[name])
        with amend1.STATE[name](J):
            reads[name] = {"gold": gold, "kl_W0": mean_kl(J.model, J.rows["W"], J.W0, None), "ci": ci_kl(J)}
        e80.save(J)
    e80.log(f"[clean {name}] {json.dumps(reads[name])}, {e80.peak()}")
    return reads[name]


def read(J: Any, state: str) -> None:
    key = f"{state}_W_s0"
    amend1.read(J, state, 0)
    st = J.out["starts"][key]
    if "trace" not in st:
        st["trace"] = amend1.READS.get(key, [])
        e80.save(J)


def e85_w0_trace() -> list[float]:
    return [float(x) for x in re.findall(r"\[W0_W_s0\] seed 0 step \d+: KL (\S+)", E85_LOG.read_text())]


def stage_w0(J: Any, seqs: list[Any]) -> None:
    """The control. On E85's card model the reused W0 stands if three steps here repeat it; on any other a fresh start, since a card model has moved the final read by 2.8%."""
    w0 = clean_reads(J, seqs, "W0")
    e80.check(J, "gold_W0", w0["gold"], r2.near_rel(w0["gold"], r2.W0_GOLD, r2.W0_TOL), f"E40's {r2.W0_GOLD} within {r2.W0_TOL:.1%}")
    e80.check(J, "ci_W0", w0["ci"], r2.near_rel(w0["ci"], rules.W0_CI, rules.W0_CI_TOL), f"the 128-row job's {rules.W0_CI:.4f} within {rules.W0_CI_TOL:.0%}")
    ft = amend2.gold(J, seqs, "FT", lambda J: e85.change_hooks(J.model, J.dw.__getitem__))
    e80.check(J, "gold_FT", w0["gold"] - ft, r2.near_rel(w0["gold"] - ft, r2.FT_GAIN, r2.FT_TOL), f"E40's gain {r2.FT_GAIN} within {r2.FT_TOL:.0%}")
    if "w0" not in J.out:
        card = torch.cuda.get_device_name()
        if card == rules.W0_CARD:
            amend2.host_check(J)
            gap = max(abs(float(a) - float(b)) for a, b in zip(J.out["host_check"]["here"], J.out["host_check"]["amend1"], strict=True))
            e80.check(J, "W0_host", gap, gap <= rules.HOST_TOL, f"three steps of W0 at seed 0 repeat Amendment 1's within {rules.HOST_TOL}")
            J.out["w0"] = {"kl": rules.W0_KL, "trace": e85_w0_trace(), "from": f"E85's start, reused on {card}"}
        else:
            read(J, "W0")
            st = J.out["starts"]["W0_W_s0"]
            e80.check(J, "W0_fresh", st["kl"], r2.near_rel(st["kl"], rules.W0_KL, rules.W0_TOL), f"E85's {rules.W0_KL:.3f} within {rules.W0_TOL:.0%}")
            J.out["w0"] = {"kl": st["kl"], "trace": st["trace"], "from": f"read here on {card}"}
        e80.save(J)
    RUNNING["w0"] = J.out["w0"]["trace"]
    e80.log(f"[w0] {J.out['w0']['kl']}, {J.out['w0']['from']}, {len(RUNNING['w0'])} steps")


def stage_rand(J: Any, seqs: list[Any]) -> float:
    w0, c = J.out["clean"]["W0"], clean_reads(J, seqs, "Rand100")
    e80.log(f"[kept Rand100] {r2.kept(c['gold'], w0['gold'], J.out['gold']['FT']):.6f}; KL against W0 {c['kl_W0']:.6f} against the {rules.BAR_EXCESS:.3f} the fail bar asks")
    e80.check(J, "clean_Rand100", c["kl_W0"], c["kl_W0"] >= rules.CLEAN_STOP, f"at least {rules.CLEAN_STOP}: under it the state does not move the output")
    RUNNING["clean"] = c["kl_W0"]
    read(J, "Rand100")
    rescore(J)
    return rules.ratio(J.out["starts"]["Rand100_W_s0"]["kl"], J.out["w0"]["kl"])


def stage_perm(J: Any, seqs: list[Any], scope: str) -> None:
    name, w0 = SCOPE[scope], J.out["clean"]["W0"]
    low, high = rules.SHARE[name]
    share = J.out.setdefault("wrong_share", {}).setdefault(name, wrong_share(J, scope))
    e80.check(J, f"share_{name}", share, low <= share <= high, f"{low} to {high}: the columns moved")
    c = clean_reads(J, seqs, name)
    e80.check(J, f"gold_{name}", c["gold"], abs(c["gold"] - w0["gold"]) <= 1e-6, "W0's: the weights are W0's")
    e80.check(J, f"clean_{name}", c["kl_W0"], c["kl_W0"] <= 1e-6, "0: the weights are W0's")
    want = rules.PERM_CI_STOP if name == "Perm" else w0["ci"] - 0.02
    e80.check(J, f"ci_{name}", c["ci"], c["ci"] >= want, f"at least {want:.3f}")
    read(J, name)
    rescore(J)


def guard(msg: str) -> None:
    """E80's log, then at each step of a start its ratio and excess over W0, and the state's filed first steps: a miss stops the job. `start` sends every step here."""
    amend1.tee(msg)
    if m := amend1.STEP.match(msg):
        key, trace = m[1], amend1.READS[m[1]]
        k, w0 = len(trace) - 1, RUNNING["w0"]
        if k < len(w0):
            amend1.tee(f"[{key}] step {k}: ratio {trace[k] / w0[k]:.4f}, excess {trace[k] - w0[k]:.4f} nats")
        state = key.removesuffix("_W_s0")
        if miss := rules.first_steps_miss(state, trace, RUNNING["clean"]):
            e80.stop("first_steps", f"{key}: {miss}")


def rescore(J: Any) -> None:
    J.out["score"] = rules.score(J.out)
    e80.save(J)
    e80.log(f"[score] {json.dumps(J.out['score'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--perm", choices=tuple(SCOPE), default="all")
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions_amend1.json missing, or stopped_amend1.json exists")
    seqs = amend2.gold_seqs()[0]
    J = e80.set_up(Path(a.decomp).expanduser().resolve())
    stage_w0(J, seqs)
    if not rules.goes_on(stage_rand(J, seqs)):
        e80.log("[stage 2] not run: stage 1 did not reach the fail bar")
        return
    stage_perm(J, seqs, a.perm)
    e80.log(f"[done] {e80.peak()}")


def score() -> None:
    """From what amend1.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(OUT)
    if out is None:
        return
    out["score"] = rules.score(out)
    e76.write_json(OUT, out)
    e80.log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    e80.OUT, e80.STOPPED, e80.LOG = OUT, STOPPED, LOG  # E80's save, stop, check and log read these globals, so no earlier job's files are written
    e80.log = guard
    amend1.STATE |= {"Rand100": rand100_state, "Perm": perm_state("all"), "PermRare": perm_state("rare")}
    e80.guarded(main, score)
