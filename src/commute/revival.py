"""E55 Stage C1's non-VPD instruments on the PCFG minGPT, and the rules that turn their readings into labels and scores
(plan reports/plans/2026-10-29_plan_reuse-stage1.md, Stage C; filed rules: experiments/pcfg_anchors/results/
predictions_c1.json). Each instrument labels a fine-tuned state "existing", "new" or "void".

- Pruning revival (Jain et al. 2024, App. D, Alg. 3): score each MLP hidden unit (row j of a block's first MLP Linear,
  weight and bias) by grad . w of the count-a answer loss, gradient at the state; zero the top K. Removing w changes
  the loss by about -grad . w, so the top units are those whose removal lowers count-a loss most. Compared with K
  units drawn at random. With sign -1 the same code removes the units count a needs most (the LOW control).
- Reverse fine-tuning: the PCFG fine-tune recipe on count a, natural rows; the step count to TARGET count-a accuracy,
  capped, against NEVER fine-tuned the same way (scratch) at the same lr.
- Steering (E52 B3, src/commute/steering.py, adapted): v_l = mean over the fit half of (tuned - base) residual at `=`
  after block l, added to the base there; the gain share of log P(the tuned model's own answer) on the scoring half,
  at the block with the largest fit-half share.

Forwards that read accuracy or log P run in fp32, eval mode (interchange.run_at_art); training uses pcfg_train's
autocast. Rules and scores at the bottom are pure arithmetic (tests/test_revival.py).
"""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from commute import pcfg, probe
from commute.interchange import binomial_sd, forward_at_art, run_at_art
from commute.mingpt import GPT, make_optimizer
from commute.pcfg_train import GRAD_CLIP, Batches, Schedule, answer_loss, autocast

PRUNE_K, N_RANDOM, SD_MULT = (5, 15), 20, 3.0  # plan: K in {5, 15}, 20 random draws, >= 3 sd
TARGET, CAP, EVAL_EVERY = 0.95, 10_000, 50  # plan: steps to 95% count a, cap 10K; read every 50 steps
SCRATCH_VOID_MAX, EXISTING_FRAC = 1000, 0.5  # plan: scratch <= 1K steps voids that lr; existing iff <= 0.5 x scratch
STEER_MIN = 0.5  # plan: existing iff gain share >= 0.5
ALPHA, MIN_N = 0.05, 8  # plan: binomial one-sided 0.05; fewer than 8 scored states -> no pass bar
MARGIN_ALL, MARGIN_RUN = 2, 1  # lead, 2026-10-30: agreement must exceed the better nuisance labeller by this many states (all states; one per run)


# ---- pruning ----


def fc_layers(model: nn.Module) -> list[nn.Linear]:
    return [cast(nn.Linear, blk.mlp[0]) for blk in model.blocks]  # type: ignore[union-attr, index]


def unit_scores(model: nn.Module, seq: np.ndarray, chunk: int = 500) -> torch.Tensor:
    """Alg. 3's grad . w per MLP hidden unit, (n_layer, 4d): gradient of the mean count-a answer loss over `seq`
    (fp32, eval mode: no dropout), summed over the unit's row of weights and its bias."""
    device = next(model.parameters()).device
    model.eval()
    model.zero_grad(set_to_none=True)
    for i in range(0, len(seq), chunk):
        idx = torch.from_numpy(seq[i : i + chunk].astype(np.int64)).to(device)
        (answer_loss(model, idx) * len(idx) / len(seq)).backward()
    out = torch.stack([(fc.weight.grad * fc.weight).sum(1) + fc.bias.grad * fc.bias for fc in fc_layers(model)])  # type: ignore[operator]
    model.zero_grad(set_to_none=True)
    return out.detach()


def pruned(model: nn.Module, units: np.ndarray) -> nn.Module:
    """A copy of `model` with the given flat unit indices (layer * width + j) zeroed: weight row and bias, so the
    unit's GELU output is exactly 0."""
    m = copy.deepcopy(model)
    fcs = fc_layers(m)
    width = fcs[0].out_features
    with torch.no_grad():
        for u in units:
            layer, j = divmod(int(u), width)
            fcs[layer].weight[j] = 0
            fcs[layer].bias[j] = 0
    return m


def accuracy(model: nn.Module, seq: np.ndarray) -> float:
    """Share of rows whose fp32 argmax at `=` is the answer token."""
    out, _ = run_at_art(model, seq)
    return float((out == seq[:, pcfg.ANS]).mean())


def prune_readings(model: nn.Module, score_seq: np.ndarray, eval_seq: np.ndarray, seed: int,
                   sign: float = 1.0) -> dict[str, Any]:  # fmt: skip
    """count-a accuracy on eval_seq unpruned (acc0), and per K: after pruning the top-K units of sign * grad . w
    (scored on score_seq) and after each of N_RANDOM uniform K-unit draws (default_rng([seed, K]), the same draws for
    every state)."""
    s = (sign * unit_scores(model, score_seq)).flatten()
    out: dict[str, Any] = {"acc0": accuracy(model, eval_seq), "sign": sign, "n_eval": len(eval_seq)}
    for k in PRUNE_K:
        rng = np.random.default_rng([seed, k])
        top = torch.topk(s, k).indices.cpu().numpy()
        rand = [accuracy(pruned(model, rng.choice(s.numel(), k, replace=False)), eval_seq) for _ in range(N_RANDOM)]
        out[str(k)] = {"top": accuracy(pruned(model, top), eval_seq), "random": rand, "units": top.tolist()}
    return out


# ---- reverse fine-tuning ----


def reverse_finetune(model: nn.Module, batches: Batches, lr: float, weight_decay: float, eval_seq: np.ndarray,
                     deadline: float | None = None, cap: int = CAP, schedule_steps: int = CAP) -> dict[str, Any]:  # fmt: skip
    """Train `model` in place on count a (answer-only loss, AdamW, the 20%-warm-up cosine schedule over schedule_steps,
    clip GRAD_CLIP) until its count-a accuracy on eval_seq, read every EVAL_EVERY steps, reaches TARGET, for at most
    `cap` steps (C1: both CAP; C1b: cap 5K inside C1's 10K schedule, so its first 5K steps are C1's). Returns steps
    (the first read at or above TARGET; 0 if the state starts there; None at the cap), the curve, and how it ended
    ("target", "cap" or "deadline": a deadline exit is not a reading)."""
    assert 0 < cap <= schedule_steps, (cap, schedule_steps)
    device = next(model.parameters()).device
    sched, opt = Schedule(schedule_steps, lr), make_optimizer(model, lr, weight_decay)
    fwd = cast(nn.Module, torch.compile(model)) if device.type == "cuda" else model
    curve = [[0, accuracy(model, eval_seq)]]
    end, steps = ("target", 0) if curve[0][1] >= TARGET else ("cap", None)
    model.train()
    for step in range(cap if steps is None else 0):
        for g in opt.param_groups:
            g["lr"] = sched.lr_at(step)
        with autocast(device):
            loss = answer_loss(fwd, batches.get(step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            curve.append([step + 1, accuracy(model, eval_seq)])
            if curve[-1][1] >= TARGET:
                end, steps = "target", step + 1
                break
            if deadline is not None and time.monotonic() > deadline:
                end = "deadline"
                break
    batches.close()
    return {"steps": steps, "end": end, "lr": lr, "curve": curve}


# ---- steering ----


@torch.no_grad()
def logp_at_art(model: nn.Module, seq: np.ndarray, target: np.ndarray, block: int = 0, patch: torch.Tensor | None = None,
                chunk: int = 1000) -> np.ndarray:  # fmt: skip
    """log P(target token) at `=` per row, fp32, eval mode; with block and patch as interchange.run_at_art."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(seq), chunk):
        idx = torch.from_numpy(seq[i : i + chunk, : pcfg.ANS].astype(np.int64)).to(device)
        logits, _ = forward_at_art(model, idx, block, None if patch is None else patch[i : i + chunk])
        tgt = torch.from_numpy(target[i : i + chunk].astype(np.int64)).to(device)
        out.append(torch.log_softmax(logits, -1).gather(1, tgt[:, None]).squeeze(1).double().cpu().numpy())
    model.train(was_training)
    return np.concatenate(out)


def steering(base: nn.Module, tuned: nn.Module, seq: np.ndarray) -> dict[str, Any]:
    """Gains G (fit, score halves: even, odd rows) of log P(tuned's argmax at `=`) from base to tuned; per block l in
    1..n_layer-1 the fit and score gain shares of base + v_l; the block with the largest fit share (ties to the lower)
    and S, its score share."""
    target, res_t = run_at_art(tuned, seq)
    _, res_b = run_at_art(base, seq)
    lp_b = logp_at_art(base, seq, target)
    d = logp_at_art(tuned, seq, target) - lp_b
    fit, score = slice(0, None, 2), slice(1, None, 2)
    g = {"G_fit": float(d[fit].mean()), "G_score": float(d[score].mean())}
    blocks = {}
    for blk in range(1, len(res_b)):  # the last block's output is not a candidate (as interchange.l_star)
        v = (res_t[blk - 1][fit] - res_b[blk - 1][fit]).mean(0)
        ds = logp_at_art(base, seq, target, block=blk, patch=res_b[blk - 1] + v) - lp_b
        blocks[blk] = {"norm": float(v.norm()), "fit": gain_share(float(ds[fit].mean()), g["G_fit"]),
                       "score": gain_share(float(ds[score].mean()), g["G_score"])}  # fmt: skip
    best = max(blocks, key=lambda b: (np.nan_to_num(blocks[b]["fit"], nan=-np.inf), -b))
    return g | {"blocks": blocks, "block": best, "S": blocks[best]["score"],
                "tuned_answer_hist": {str(int(k)): int(c) for k, c in zip(*np.unique(target - pcfg.NUM0, return_counts=True), strict=True)}}  # fmt: skip


# ---- possession probe (C1b) ----


def possession(base: GPT, rows: Callable[[np.random.Generator, int], np.ndarray], seed: int, held_seed: list[int],
               tuned: nn.Module | None = None) -> dict[str, Any]:  # fmt: skip
    """Def 1 probe with Hewitt & Liang's control: at every block, a linear probe on `base`'s residual at `=` (probe.py's
    recipe and Probes) trained to predict the label of each row, and a shuffled-label probe (labels permuted within
    the batch) on the same batches. The label is `tuned`'s argmax at `=` (the state's own answer) or, with no tuned
    model, the row's answer token. Training rows: rows(default_rng([PROBE_SEED0 + seed, chunk]), n); held out:
    probe.HELD_OUT rows from default_rng(held_seed). Selectivity = held-out accuracy minus the shuffled probe's, per
    block; x = its maximum, at `block` (ties to the lower block)."""
    device, n_layer, d = next(base.parameters()).device, base.cfg.n_layer, base.cfg.d
    for m in (base, tuned):
        if m is not None:
            m.eval()

    @torch.no_grad()
    def labels(seq: torch.Tensor) -> torch.Tensor:
        return seq[:, pcfg.ANS] if tuned is None else forward_at_art(tuned, seq[:, : pcfg.ANS], 0, None)[0].argmax(-1)

    torch.manual_seed(probe.PROBE_SEED0 + seed)
    probes = probe.Probes(2 * n_layer, d, pcfg.VOCAB).to(device)
    opt = torch.optim.AdamW(probes.parameters(), lr=probe.LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(probe.DECAY_AT), gamma=probe.DECAY)
    batches = Batches(rows, probe.PROBE_SEED0 + seed, probe.BATCH, 100, device)
    for step in range(probe.ITERS):
        seq = batches.get(step)
        with torch.no_grad():
            _, resid = forward_at_art(base, seq[:, : pcfg.ANS], 0, None)
        loss = probe._loss(probes, torch.stack(resid), labels(seq), torch.randperm(probe.BATCH, device=device))  # pyright: ignore[reportPrivateUsage]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    batches.close()
    held = rows(np.random.default_rng(held_seed), probe.HELD_OUT)
    y = torch.from_numpy((held[:, pcfg.ANS] if tuned is None else run_at_art(tuned, held)[0]).astype(np.int64)).to(device)
    with torch.no_grad():
        correct = (probes(torch.cat([run_at_art(base, held)[1]] * 2)).argmax(-1) == y).float().mean(1).cpu().numpy()
    acc, shuf = correct[:n_layer], correct[n_layer:]
    sel = acc - shuf
    j = int(np.argmax(sel))
    _, counts = np.unique(y.cpu().numpy(), return_counts=True)
    return {"acc": acc.tolist(), "shuffled": shuf.tolist(), "selectivity": sel.tolist(), "block": j + 1,
            "x": float(sel[j]), "label_majority": float(counts.max() / len(held))}  # fmt: skip


# ---- filed rules (pure arithmetic) ----


def gain_share(delta: float, gain: float) -> float:
    """Mean steered change over the gain; nan when the gain is not positive (nothing to share)."""
    return delta / gain if gain > 0 else float("nan")


def revival_z(top: float, random: list[float], n: int) -> tuple[float, float]:
    """(top - mean(random), sd): sd over the draws (ddof 1) floored at the binomial sd of their mean at n rows."""
    r = np.asarray(random, float)
    return top - float(r.mean()), max(float(r.std(ddof=1)), binomial_sd(float(r.mean()), n))


def prune_label(readings: dict[str, Any]) -> str:
    """'void' if count a is already at TARGET unpruned (nothing to revive); 'existing' iff at some K the top-K accuracy
    exceeds the random mean by >= SD_MULT sd (inclusive); else 'new'."""
    if readings["acc0"] >= TARGET:
        return "void"
    zs = [revival_z(readings[str(k)]["top"], readings[str(k)]["random"], readings["n_eval"]) for k in PRUNE_K]
    return "existing" if any(diff >= SD_MULT * sd for diff, sd in zs) else "new"


def prune_control(readings: dict[str, Any]) -> bool:
    """LOW, sign -1 (importance pruning): at some K the top-K accuracy is below the random mean by >= SD_MULT sd."""
    zs = [revival_z(readings[str(k)]["top"], readings[str(k)]["random"], readings["n_eval"]) for k in PRUNE_K]
    return any(-diff >= SD_MULT * sd for diff, sd in zs)


def scratch_value(steps: int | None) -> int:
    """Scratch's step count, the cap counting as CAP."""
    return CAP if steps is None else steps


def reverse_label(steps: int | None, end: str, scratch_steps: int | None) -> str:
    """'void' if scratch at this lr took <= SCRATCH_VOID_MAX steps, if the state started at TARGET (steps 0) or if
    the run did not finish ('deadline'); 'existing' iff steps (cap -> CAP) <= EXISTING_FRAC x scratch's (inclusive);
    else 'new'."""
    s = scratch_value(scratch_steps)
    if s <= SCRATCH_VOID_MAX or steps == 0 or end == "deadline":
        return "void"
    return "existing" if scratch_value(steps) <= EXISTING_FRAC * s else "new"


def steer_label(r: dict[str, Any]) -> str:
    """'void' when either gain is not positive (S nan); 'existing' iff S >= STEER_MIN (inclusive); else 'new'."""
    if not (r["G_fit"] > 0 and r["G_score"] > 0) or math.isnan(r["S"]):
        return "void"
    return "existing" if r["S"] >= STEER_MIN else "new"


def binom_tail(n: int, a: int) -> float:
    """P(Binomial(n, 0.5) >= a)."""
    return sum(math.comb(n, i) for i in range(max(a, 0), n + 1)) / 2**n


def pass_bar(n: int, majority: int) -> int:
    """Smallest a with binom_tail(n, a) <= ALPHA, raised to majority + 1 when that is higher (a constant labeller
    fails). May exceed n: the bar is then unreachable."""
    a = next(a for a in range(n + 2) if binom_tail(n, a) <= ALPHA)
    return max(a, majority + 1)


def score(labels: dict[str, str], truth: dict[str, str], nuisance: dict[str, dict[str, str]] | None = None,
          margin: int = 0) -> dict[str, Any]:  # fmt: skip
    """Agreement of an instrument's labels with the truth on the truth's states; a 'void' or missing label drops the
    state (n shrinks). Below MIN_N states there is no bar (reported only). Passes iff agree >= bar and agree >= the
    better nuisance labeller's agreement on the same states + margin (no nuisance given: the bar alone)."""
    used = [s for s in truth if labels.get(s) in ("existing", "new")]
    agree = sum(labels[s] == truth[s] for s in used)
    majority = max(Counter(truth[s] for s in used).values(), default=0)
    bar = pass_bar(len(used), majority) if len(used) >= MIN_N else None
    nuis = {name: sum(lab.get(s) == truth[s] for s in used) for name, lab in (nuisance or {}).items()}
    beats = not nuis or agree >= max(nuis.values()) + margin
    return {"n": len(used), "agree": agree, "majority": majority, "bar": bar, "nuisance_agree": nuis, "margin": margin,
            "beats_nuisance": beats, "passes_bar": None if bar is None else agree >= bar,
            "passes": None if bar is None else agree >= bar and beats, "p_value": binom_tail(len(used), agree),
            "voided": sorted(s for s in truth if s not in used)}  # fmt: skip


# ---- C1b: calibrated cut-offs and the recipe ceiling (pure arithmetic; predictions_c1b.json rules) ----

MIN_PER_CLASS = 3  # C1b: fewer non-void calibration readings in a class -> the instrument is void


def cutoff(readings: dict[str, float | None], labels: dict[str, str], high_is_existing: bool) -> dict[str, Any]:
    """C1b's calibration rule. Readings that are None or not finite are void and dropped. Void ('void' names why) when
    a class holds fewer than MIN_PER_CLASS readings or every reading is equal. Otherwise, with y = x (high_is_existing)
    or -x, c is the midpoint between adjacent distinct sorted y that maximises balanced accuracy (existing iff y >= c);
    ties go to the candidate nearest the median y, then to the lower one. When the classes separate (every existing y
    above every new y) the only candidate at balanced accuracy 1 is the gap's midpoint. c is returned in x's units."""
    sign = 1.0 if high_is_existing else -1.0
    y = {s: sign * float(x) for s, x in readings.items() if x is not None and math.isfinite(x)}
    ex = np.array([v for s, v in y.items() if labels[s] == "existing"])
    nw = np.array([v for s, v in y.items() if labels[s] == "new"])
    out: dict[str, Any] = {"c": None, "void": None, "high_is_existing": high_is_existing, "n_existing": len(ex),
                           "n_new": len(nw), "dropped": sorted(set(readings) - set(y))}  # fmt: skip
    if min(len(ex), len(nw)) < MIN_PER_CLASS:
        return out | {"void": f"fewer than {MIN_PER_CLASS} readings in a class"}
    vals = np.unique(np.concatenate([ex, nw]))
    if len(vals) == 1:
        return out | {"void": "every reading equal"}
    cand = (vals[:-1] + vals[1:]) / 2
    ba = np.array([len(nw) * int((ex >= c).sum()) + len(ex) * int((nw < c).sum()) for c in cand])  # x 2 n_e n_n: exact ties
    med = float(np.median(np.concatenate([ex, nw])))
    best = min(np.flatnonzero(ba == ba.max()), key=lambda i: (abs(cand[i] - med), cand[i]))
    return out | {"c": sign * float(cand[best]), "balanced_accuracy": float(ba[best]) / (2 * len(ex) * len(nw)),
                  "separated": bool(ex.min() > nw.max())}  # fmt: skip


def label_by_cutoff(x: float | None, cut: dict[str, Any]) -> str:
    """'void' when the reading is void or the cut-off is; else 'existing' iff x is on the existing side of c or at c
    (inclusive: x >= c, or x <= c when low readings mean existing); else 'new'."""
    if x is None or not math.isfinite(x) or cut["c"] is None:
        return "void"
    return "existing" if (x >= cut["c"] if cut["high_is_existing"] else x <= cut["c"]) else "new"


def ceiling(values: dict[str, float], truth: dict[str, str]) -> dict[str, Any]:
    """The hindsight ceiling of one recipe-only labeller on the truth's states: over both directions (small values ->
    existing, or large) and every threshold (the midpoints between adjacent distinct values, and beyond either end,
    which gives the two constant labellers), the most states agreeing with the truth. Ties go to the first found
    (small -> existing first, thresholds ascending). Returns agree, the labels, threshold and direction."""
    states = list(truth)
    vals = np.unique([values[s] for s in states]) if states else np.array([])
    ths = [-math.inf, *((vals[:-1] + vals[1:]) / 2).tolist(), math.inf]
    best: dict[str, Any] = {"agree": -1}
    for small in (True, False):
        for t in ths:
            lab = {s: "existing" if (values[s] <= t if small else values[s] >= t) else "new" for s in states}
            agree = sum(lab[s] == truth[s] for s in states)
            if agree > best["agree"]:
                best = {"agree": agree, "labels": lab, "threshold": t if math.isfinite(t) else str(t), "small_is_existing": small}
    return best
