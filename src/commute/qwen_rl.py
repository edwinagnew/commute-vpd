"""E69's shared pieces: the GRPO loop on Qwen3-1.7B (plan reports/plans/2026-11-03_plan_e69-grpo-loop.md). The pure rules
come first (data order, reward, the three arms' weights, the readings, the P0-P7 scoring, the cost rule: no torch), then
the sampler, the readout on the held-out samples and the weighted step.

A record is one sampled response: {pid, k, ids, truncated, strict, reward}. Comparisons round to 9 places
(`qwen_updates.r9`, which `at_least` applies to its value only: a threshold that is a product is rounded here); a missing
or non-finite value is a miss in every rule.

The sampler draws for live rows only, so a row's draws depend on when the other rows finished: under one seed it gives
the same tokens as `model.generate` only up to the first finished row. Agreement with `generate` is checked in
distribution (stage S, `agreement`); tests/test_qwen_rl.py checks it token for token on greedy draws.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from commute.qwen_sft import Seq, Weights, set_weights, targets
from commute.qwen_updates import at_least, at_most, keep_correct, ok, paired_gain, r9
from commute.vpd import kl_per_position

Records = list[dict[str, Any]]
Sample = tuple[list[int], bool]  # response ids including the end token; truncated (no end token inside the window)
BYTES_PER_CACHED_TOKEN = 228_000  # fp32, twice E45's 114 kB in bf16 (plan, pre-mortem)
SAMPLING_T1 = {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0}  # for `generate`: left out, HF's top_k=50 returns


# ----------------------------------------------------------------------------- data order


def split(n: int, seed: int, n_held: int) -> tuple[list[int], list[int]]:
    """The held-out indices and the training order: one seeded permutation of range(n), cut."""
    perm = random.Random(f"qwen-rl-{seed}").sample(range(n), n)
    held, train = perm[:n_held], perm[n_held:]
    assert not set(held) & set(train)
    return held, train


def prompts_at(train: Sequence[int], t: int, per: int) -> list[int]:
    """Step t's problems (t counts from 1)."""
    out = list(train[(t - 1) * per : t * per])
    assert len(out) == per, f"step {t} runs past the training problems"
    return out


def step_seed(seed: int, t: int) -> int:
    """`torch.manual_seed` of step t's rollouts, the same in every arm; t = 0 is the held-out set's."""
    return 1000 * seed + t


# ----------------------------------------------------------------------------- reward and weights


def reward(strict: bool, truncated: bool) -> int:
    """1 if the strict grader is right and the response ended inside the window."""
    return int(strict and not truncated)


def _groups(recs: Records) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        groups[r["pid"]].append(i)
    return groups


def rl_weights(recs: Records) -> list[float]:
    """R: reward minus the mean of its problem's samples; 0 throughout a group that is all right or all wrong."""
    w = [0.0] * len(recs)
    for idx in _groups(recs).values():
        mean = statistics.mean(recs[i]["reward"] for i in idx)
        for i in idx:
            w[i] = recs[i]["reward"] - mean
    return w


def null_weights(recs: Records, seed: int, t: int) -> list[float]:
    """N: as R after shuffling the rewards within each group, seeded by (seed, t, problem)."""
    w = [0.0] * len(recs)
    for pid, idx in _groups(recs).items():
        rs = [recs[i]["reward"] for i in idx]
        random.Random(f"null-{seed}-{t}-{pid}").shuffle(rs)
        mean = statistics.mean(rs)
        for i, r in zip(idx, rs, strict=True):
            w[i] = r - mean
    return w


def sft_weights(recs: Records) -> list[float]:
    """F: 1 for the samples `qwen_updates.keep_correct` keeps, 0 otherwise."""
    kept = {id(r) for r in keep_correct(recs)}
    return [float(id(r) in kept) for r in recs]


def arm_weights(arm: str, recs: Records, seed: int, t: int) -> list[float]:
    if arm == "R":
        return rl_weights(recs)
    if arm == "N":
        return null_weights(recs, seed, t)
    assert arm == "F", arm
    return sft_weights(recs)


# ----------------------------------------------------------------------------- readings


def rollout_summary(recs: Records) -> dict[str, float]:
    """A step's reward rate, mean response tokens and share of responses that ended inside the window."""
    return {"reward_rate": statistics.mean(r["reward"] for r in recs),
            "mean_len": statistics.mean(len(r["ids"]) for r in recs),
            "window_share": statistics.mean(not r["truncated"] for r in recs)}


def rollout_hash(recs: Records) -> str:
    """The rollouts token for token: the arms' first-step identity check."""
    return hashlib.sha256(json.dumps([[r["pid"], r["k"], r["ids"]] for r in recs]).encode()).hexdigest()


def pool(recs: Records) -> dict[int, list[int]]:
    """The readout pool: per problem with both right and wrong samples, its record indices."""
    return {p: idx for p, idx in _groups(recs).items() if 0 < sum(recs[i]["reward"] for i in idx) < len(idx)}


def delta_r(changes: Sequence[float], recs: Records) -> float | None:
    """ΔR: over the pool's problems, the mean of (mean log-prob change of the problem's right samples minus that of its
    wrong ones); `changes` is per record, relative to step 0. None for an empty pool."""
    gaps = []
    for idx in pool(recs).values():
        right = [changes[i] for i in idx if recs[i]["reward"]]
        wrong = [changes[i] for i in idx if not recs[i]["reward"]]
        gaps.append(statistics.mean(right) - statistics.mean(wrong))
    return statistics.mean(gaps) if gaps else None


def gain(w0: Records, t: Records) -> dict[str, Any]:
    """`qwen_updates.paired_gain` (drop rule, problems that fit at both) with the paired standard error; None for under
    two problems."""
    a = {r["idx"]: r["strict"] for r in w0 if r["fits"]}
    d = [int(r["strict"]) - int(a[r["idx"]]) for r in t if r["fits"] and r["idx"] in a]
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None
    return {**paired_gain(w0, t), "se": se}


def agreement(a: Records, b: Records, key: Callable[[dict[str, Any]], float], mult: float) -> dict[str, Any]:
    """Two samplers on the same problems: holds if |mean difference| <= mult x its SE, both taken over problems, since
    samples of one problem are not independent."""
    per = [{p: statistics.mean(key(rs[i]) for i in idx) for p, idx in _groups(rs).items()} for rs in (a, b)]
    assert set(per[0]) == set(per[1]), "not the same problems"
    d = [per[0][p] - per[1][p] for p in per[0]]
    diff = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None
    return {"a": statistics.mean(per[0].values()), "b": statistics.mean(per[1].values()), "diff": diff, "se": se,
            "holds": ok(se) and at_most(abs(diff), r9(mult * se))}  # type: ignore[operator]


def reward_change(rates: Sequence[float], upto: int) -> float:
    """Mean reward over the second half of steps 1..upto minus over the first (at 20: 11-20 minus 1-10)."""
    half = upto // 2
    return statistics.mean(rates[half:upto]) - statistics.mean(rates[:half])


def above(x: float | None, t: float) -> bool:
    return ok(x) and r9(x) > t  # type: ignore[arg-type]


def below(x: float | None, t: float) -> bool:
    return ok(x) and r9(x) < t  # type: ignore[arg-type]


def in_band(x: float | None, band: Sequence[float]) -> bool:
    return at_least(x, band[0]) and at_most(x, band[1])


# ----------------------------------------------------------------------------- cost (stage S)


def rows_for(total: int, reserved: int, tokens_per_row: int, max_rows: int) -> int:
    """Sampling rows per batch from the card's total memory (bytes), not its free memory, so that it is the same in
    every arm of one card and the rollouts stay identical."""
    return max(1, min(max_rows, (total - reserved) // (BYTES_PER_CACHED_TOKEN * tokens_per_row)))


def arm_seconds(steps: int, n_reads: int, per_step: float, read: float, acc: float) -> float:
    return steps * per_step + n_reads * read + acc


def projected_usd(t: Mapping[str, float], steps: int, n_steps: int, n_reads: Mapping[int, int], setup_h: float,
                  usd_per_h: float) -> float:
    """R + N + F at N's `n_steps`, from the seconds in `t` (sample, train, f_train, read, acc). `n_reads`: readings of
    an arm of that many steps."""
    rl = t["sample"] + t["train"]
    seconds = (arm_seconds(steps, n_reads[steps], rl, t["read"], t["acc"])
               + arm_seconds(n_steps, n_reads[n_steps], rl, t["read"], t["acc"])
               + arm_seconds(steps, n_reads[steps], t["sample"] + t["f_train"], t["read"], t["acc"]))
    return (seconds / 3600 + setup_h) * usd_per_h


def cost_rule(full: float, cut: float, limit: float, steps: int, n_cut: int) -> dict[str, Any]:
    """Over the limit: N runs `n_cut` steps; still over: stop."""
    if not above(full, limit):
        return {"n_steps": steps, "stop": False}
    return {"n_steps": n_cut, "stop": above(cut, limit)}


# ----------------------------------------------------------------------------- P0-P7 (results/predictions.json)


def p0(rule: dict[str, Any], accuracy: dict[str, dict[str, Any]], read: dict[str, Any], n_pool: int) -> dict[str, Any]:
    """Strict drop-rule accuracy near the baseline's, ΔR(0) and KL(0) at zero, a non-empty pool."""
    def close(x: float | None, ref: float, tol: float) -> bool:
        return x is not None and at_most(abs(x - ref), tol)

    near = {s: close(accuracy.get(s, {}).get("strict_dropped"), rule[s]["ref"], rule[s]["tol"]) for s in ("gsm8k", "gsm1k")}
    zero = {k: close(read.get(k), 0.0, rule["zero_max"]) for k in ("delta_r", "kl")}
    return {"near": near, "zero": zero, "pool": n_pool, "holds": all(near.values()) and all(zero.values()) and n_pool > 0}


def p1(rule: dict[str, Any], g: dict[str, Any]) -> str:
    x, se = g.get("diff"), g.get("se")
    if not (ok(x) and ok(se)):
        return "missing"
    if at_least(x, rule["learns_min"]) and at_least(x, r9(rule["se_mult"] * se)):
        return "learns"
    return "no gain" if below(x, rule["no_gain_below"]) else "ambiguous"


def p2(rule: dict[str, Any], g8k: dict[str, Any], g1k: dict[str, Any]) -> dict[str, Any]:
    """d = GSM1k gain minus GSM8K gain; the two sets are independent, so the SEs add in quadrature."""
    parts = (g8k.get("diff"), g8k.get("se"), g1k.get("diff"), g1k.get("se"))
    if not all(ok(x) for x in parts):
        return {"d": None, "se": None, "label": "missing"}
    d, se = g1k["diff"] - g8k["diff"], math.hypot(g8k["se"], g1k["se"])
    if below(abs(d), rule["same_below"]) and at_most(abs(d), r9(rule["se_mult"] * se)):
        label = "same"
    elif at_least(-d, rule["only_min"]) and at_least(-d, r9(rule["se_mult"] * se)):
        label = "GSM8K-only"
    else:
        label = "ambiguous"
    return {"d": d, "se": se, "label": label}


def p3(rule: dict[str, Any], g: dict[str, Any]) -> bool:
    x = g.get("diff")
    return ok(x) and below(abs(x), rule["abs_below"])  # type: ignore[arg-type]


def _read(arm: dict[str, Any], t: int, key: str) -> float | None:
    return arm.get("reads", {}).get(str(t), {}).get(key)


def p4(rule: dict[str, Any], R: dict[str, Any], N: dict[str, Any]) -> dict[str, Any]:
    """R's ΔR above 0 at every saved step, and at least rule's multiple of |N's ΔR| at the ratio steps (N's at its last
    step if it was cut)."""
    positive = {t: above(_read(R, t, "delta_r"), 0.0) for t in rule["saved_steps"]}
    ratio = {}
    for t in rule["ratio_steps"]:
        n = _read(N, min(t, N.get("last", t)), "delta_r")
        ratio[t] = ok(n) and at_least(_read(R, t, "delta_r"), r9(rule["ratio"] * abs(n)))  # type: ignore[arg-type]
    return {"positive": positive, "ratio": ratio, "holds": all(positive.values()) and all(ratio.values())}


def _steps_in(arm: dict[str, Any], span: Sequence[int]) -> list[dict[str, Any]]:
    rows = [s for s in arm.get("steps", []) if span[0] <= s["step"] <= span[1]]
    return rows if len(rows) == span[1] - span[0] + 1 else []


def p6(rule: dict[str, Any], R: dict[str, Any]) -> dict[str, Any]:
    """Mean rollout length over the last span within the band of the first span's; share inside the window at least the
    minimum over the last span. A span not fully run, or a first span of length 0, is a miss."""
    first, last = _steps_in(R, rule["first"]), _steps_in(R, rule["last"])
    m0 = statistics.mean(s["mean_len"] for s in first) if first else 0.0
    if not last or m0 <= 0:
        return {"ratio": None, "share": None, "holds": False}
    ratio = statistics.mean(s["mean_len"] for s in last) / m0
    share = statistics.mean(s["window_share"] for s in last)
    return {"ratio": ratio, "share": share, "holds": in_band(ratio, rule["ratio_band"]) and at_least(share, rule["share_min"])}


def p7(rule: dict[str, Any], R: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    kl_r, kl_f = _read(R, rule["step"], "kl"), _read(f, rule["step"], "kl")
    g_r, g_f = R.get("gain", {}).get("gsm8k", {}).get("diff"), f.get("gain", {}).get("gsm8k", {}).get("diff")
    more_kl = ok(kl_r) and at_least(kl_f, r9(rule["kl_ratio"] * kl_r))  # type: ignore[operator]
    no_better = ok(g_r) and ok(g_f) and r9(g_f) <= r9(g_r)  # type: ignore[arg-type]
    return {"kl_R": kl_r, "kl_F": kl_f, "more_kl": more_kl, "gain_R": g_r, "gain_F": g_f, "no_better": no_better,
            "holds": more_kl and no_better}


def score_all(pred: dict[str, Any], A0: dict[str, Any], R: dict[str, Any], N: dict[str, Any],
              f: dict[str, Any]) -> dict[str, Any]:
    """Every filed prediction from the stages' results files; a stage not run reads as a miss."""
    g8k, g1k = R.get("gain", {}).get("gsm8k", {}), R.get("gain", {}).get("gsm1k", {})
    return {"P0": p0(pred["P0"], A0.get("accuracy", {}), A0.get("read", {}), A0.get("pool", 0)),
            "P1": {"gain": g8k, "label": p1(pred["P1"], g8k)}, "P2": p2(pred["P2"], g8k, g1k),
            "P3": {"gain": N.get("gain", {}).get("gsm8k", {}), "holds": p3(pred["P3"], N.get("gain", {}).get("gsm8k", {}))},
            "P4": p4(pred["P4"], R, N), "P6": p6(pred["P6"], R), "P7": p7(pred["P7"], R, f)}


# ----------------------------------------------------------------------------- run names, upload check, next stage

RATES = {"1e-6": 1e-6, "3e-6": 3e-6, "3e-7": 3e-7}  # --lr-tag: the plan's rate and the two Amendment 1 reruns
DEFAULT_TAG = "1e-6"


def rate_names(shared: str, family: str, tag: str) -> tuple[str, str]:
    """(folder under results/ and results/large/, W&B artifact prefix) of a run: the shared ones at the plan's rate, their
    own under `lr<tag>` at another, so that the 1e-6 files are never overwritten."""
    if tag == DEFAULT_TAG:
        return shared, family
    return str(Path(shared) / f"lr{tag}"), f"{family}_lr{tag}"


def artifact_name(prefix: str, rel: Path) -> str:
    """The name `qwen_sft.Uploader.add` gives the file."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", f"{prefix}-{rel.parent.name}-{rel.name}")


def upload_ok(size: int | None, digest: str | None, path: Path) -> bool:
    """A W&B manifest entry (size, base64 md5 digest) matches the local file."""
    if size is None or digest is None or path.stat().st_size != size:
        return False
    md5 = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            md5.update(block)
    return base64.b64encode(md5.digest()).decode() == digest


def next_stage(tag: str, R: dict[str, Any], N: dict[str, Any], f: dict[str, Any], rule: dict[str, Any]) -> dict[str, str]:
    """The plan's decision tree, read mechanically from R's result at `tag`: {stage, tag, why}; stage "lead" means no
    command, tell the lead. R3 (reward fell by step 20 at 1e-6) -> R at 3e-7; R2 (P1 no gain at 1e-6) -> R at 3e-6; else N,
    F, score at this tag (an Amendment 1 rerun's N and F run at its rate whatever it reads)."""
    def say(stage: str, to: str, why: str) -> dict[str, str]:
        return {"stage": stage, "tag": to, "why": why}

    stopped = R.get("stopped") or N.get("stopped") or f.get("stopped")
    if stopped:
        if R.get("stopped", "").startswith("reward change by step") and tag == DEFAULT_TAG:
            return say("R", "3e-7", f"R3: R at {tag} stopped at {R['stopped']!r}: R again at 3e-7, own folder, then N, F there")
        return say("lead", tag, f"stopped at {stopped!r}: tell the lead")
    if not R.get("done"):
        return say("R", tag, f"R at {tag} is not finished: the same command resumes")
    label = p1(rule, R.get("gain", {}).get("gsm8k", {}))
    if label == "missing":
        return say("lead", tag, "R is done but has no GSM8K gain: tell the lead")
    if label == "no gain" and tag == DEFAULT_TAG:
        return say("R", "3e-6", "R2: P1 no gain at 1e-6: R again at 3e-6, own folder; then run next with --lr-tag 3e-6")
    note = f"R at {tag} reads {label}" + ("; tell the lead: the harder-benchmark card" if label == "no gain" else "")
    for stage, res in (("N", N), ("F", f)):
        if not res.get("done"):
            return say(stage, tag, f"{note}: {stage} at {tag}")
    return say("score", tag, f"{note}: N and F are done")


# ----------------------------------------------------------------------------- batches


def micro_batches(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Indices into `lengths`, longest first, cut so that rows x longest row stays within `budget` tokens (a row over
    the budget goes alone)."""
    out: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lambda j: -lengths[j]):
        if out and (len(out[-1]) + 1) * lengths[out[-1][0]] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _by_length(prompts: Sequence[list[int]], n: int, rows: int) -> list[list[tuple[int, int]]]:
    """(prompt, sample) pairs in batches of `rows`, longest prompts first, so that padding is small."""
    order = sorted(((i, k) for i in range(len(prompts)) for k in range(n)), key=lambda x: -len(prompts[x[0]]))
    return [order[s : s + rows] for s in range(0, len(order), rows)]


def _left_pad(prompts: Sequence[list[int]], pad: int, device: Any) -> tuple[Tensor, Tensor]:
    T = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), T), pad, dtype=torch.long, device=device)
    mask = torch.zeros(len(prompts), T, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, T - len(p) :] = torch.tensor(p, device=device)
        mask[i, T - len(p) :] = 1
    return ids, mask


# ----------------------------------------------------------------------------- the sampler


def sample_t1(logits: Tensor) -> Tensor:
    """Temperature 1, no top-k or top-p: one token per row from the model's own distribution."""
    return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)


@torch.no_grad()
def _sample_batch(model: Any, prompts: Sequence[list[int]], window: int, pad: int,
                  draw: Callable[[Tensor], Tensor]) -> list[Sample]:
    """One left-padded batch; a row leaves the batch and its cache when it ends or reaches the window."""
    device = model.device
    eos = torch.tensor(model.generation_config.eos_token_id, device=device).flatten()
    ids, mask = _left_pad(prompts, pad, device)
    next_pos = mask.sum(1)
    room = window - next_pos
    assert int(room.min()) > 0, "a prompt fills the window"
    alive = torch.arange(len(prompts), device=device)
    position = (mask.cumsum(1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=position, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    responses: list[list[int]] = [[] for _ in prompts]
    truncated = [False] * len(prompts)
    n_new = 0
    while True:
        tok = draw(out.logits[:, -1])
        for row, x in zip(alive.tolist(), tok.tolist(), strict=True):
            responses[row].append(x)
        n_new += 1
        ended, full = torch.isin(tok, eos), n_new >= room
        for row in alive[full & ~ended].tolist():
            truncated[row] = True
        keep = (~(ended | full)).nonzero().squeeze(1)
        if not len(keep):
            return list(zip(responses, truncated, strict=True))
        if len(keep) < len(alive):
            alive, room, tok, mask, next_pos = alive[keep], room[keep], tok[keep], mask[keep], next_pos[keep]
            cache.batch_select_indices(keep)
        mask = torch.cat([mask, mask.new_ones(len(keep), 1)], dim=1)
        out = model(input_ids=tok[:, None], attention_mask=mask, position_ids=(next_pos + n_new - 1)[:, None],
                    past_key_values=cache, use_cache=True)


def rollouts(model: Any, prompts: Sequence[list[int]], n: int, window: int, rows: int, pad: int,
             draw: Callable[[Tensor], Tensor] = sample_t1) -> list[Sample]:
    """n samples per prompt, n in a row per prompt in order, at the model's own precision and independent of its
    generation config. Prompt plus response at most `window` tokens per row. Deterministic given `torch.manual_seed`,
    `rows` and the card."""
    found = {}
    for batch in _by_length(prompts, n, rows):
        for pair, s in zip(batch, _sample_batch(model, [prompts[i] for i, _ in batch], window, pad, draw), strict=True):
            found[pair] = s
    return [found[i, k] for i in range(len(prompts)) for k in range(n)]


@torch.no_grad()
def generate_rollouts(model: Any, prompts: Sequence[list[int]], n: int, window: int, rows: int, pad: int,
                      **kw: Any) -> list[Sample]:
    """`rollouts`' output by `model.generate` (every row kept to the longest), cut per row at the same window: the
    sampler's control. Pass the settings explicitly (`SAMPLING_T1`)."""
    eos = torch.tensor(model.generation_config.eos_token_id).flatten()
    found = {}
    for batch in _by_length(prompts, n, rows):
        chosen = [prompts[i] for i, _ in batch]
        ids, mask = _left_pad(chosen, pad, model.device)
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=window - min(len(p) for p in chosen), **kw)
        for pair, p, row in zip(batch, chosen, out[:, ids.shape[1] :].cpu(), strict=True):
            room = row[: window - len(p)]
            hits = torch.isin(room, eos).nonzero()
            found[pair] = (room[: int(hits[0]) + 1].tolist(), False) if len(hits) else (room.tolist(), True)
    return [found[i, k] for i in range(len(prompts)) for k in range(n)]


# ----------------------------------------------------------------------------- the readout on H


def _response_logits(model: Any, toks: Tensor, mask: Tensor) -> Tensor:
    """Logits at the response's target positions only [n, V]: a whole batch's would hold the prompts' too."""
    return model.lm_head(model.model(toks, use_cache=False).last_hidden_state[:, :-1][mask])


@torch.no_grad()
def read_h(model: Any, W0: Weights | None, seqs: Sequence[Seq], budget: int) -> tuple[list[float], float]:
    """Per sequence the change, the weights as they stand minus W0 (None: as they stand, a zero reading), in its summed
    response log-prob; and the mean per-token KL(π₀ ‖ π_t) over all the response tokens. At the model's precision."""
    changes = [0.0] * len(seqs)
    kl, n = 0.0, 0
    for idx in micro_batches([len(s[0]) for s in seqs], budget):
        toks, mask = targets([seqs[i] for i in idx], pad=0)
        toks, mask = toks.to(model.device), mask.to(model.device)
        with set_weights(model, W0):
            before = _response_logits(model, toks, mask)
        after = _response_logits(model, toks, mask)
        kl += float(kl_per_position(before, after).double().sum())
        n += int(mask.sum())
        target = toks[:, 1:][mask][:, None]
        change = (torch.log_softmax(after, -1).gather(1, target) - torch.log_softmax(before, -1).gather(1, target))
        rows = mask.nonzero()[:, 0]
        sums = torch.zeros(len(idx), dtype=torch.float64, device=toks.device).index_add_(0, rows, change[:, 0].double())
        for i, x in zip(idx, sums.tolist(), strict=True):
            changes[i] = x
    return changes, kl / n


# ----------------------------------------------------------------------------- the weighted step


def weighted_step(model: Any, seqs: Sequence[Seq], weights: Sequence[float], divisor: int, budget: int) -> float:
    """Gradients of Σ wᵢ · (summed token log-loss of response i) / divisor, in micro-batches sorted by length, with
    no autocast context; samples of weight 0 are not run. Returns the loss. The caller zeroes before and steps after."""
    live = [i for i, w in enumerate(weights) if w != 0]
    total = 0.0
    for chunk in micro_batches([len(seqs[i][0]) for i in live], budget):
        idx = [live[j] for j in chunk]
        toks, mask = targets([seqs[i] for i in idx], pad=0)
        toks, mask = toks.to(model.device), mask.to(model.device)
        logits = model(toks, use_cache=False).logits[:, :-1]
        nll = F.cross_entropy(logits.flatten(0, 1), toks[:, 1:].flatten(), reduction="none").view_as(mask)
        w = torch.tensor([weights[i] for i in idx], device=logits.device, dtype=logits.dtype)
        s = (w * (nll * mask).sum(1)).sum() / divisor
        s.backward()
        total += float(s.detach())
    return total


def weighted_update(model: Any, params: Mapping[str, Tensor], opt: Any, seqs: Sequence[Seq], weights: Sequence[float],
                    divisor: int, budget: int) -> dict[str, Any]:
    """One optimiser step on `weighted_step`, in training mode (gradient checkpointing runs only there), gradients
    freed after it so the sampler has the memory. All weights 0: no step, since Adam moves on a zero gradient."""
    opt.zero_grad(set_to_none=True)
    if not any(weights):
        return {"live": False, "loss": 0.0, "grad_norm": 0.0}
    model.train()
    loss = weighted_step(model, seqs, weights, divisor, budget)
    model.eval()
    gnorm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), math.inf))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return {"live": True, "loss": loss, "grad_norm": gnorm}


def delta_fp32(params: Mapping[str, Tensor], W0: Weights) -> dict[str, Tensor]:
    """δW in fp32 on the host, one matrix at a time (`qwen_train.delta` rounds to bf16, which a 1e-6 step does not survive)."""
    return {s: (p.detach() - W0[s]).to("cpu", torch.float32) for s, p in params.items()}
