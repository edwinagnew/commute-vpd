"""E224 (rl_67m), cell P2.41 run 6: E220's four arms (size-matched) on three task#2191 tasks where reward and reading
are the same next token; run 7, E222's readings of the saved states, is read_newtasks.py. Card reports/plans/2026-11-20_plan_rl-67m-newtasks-card.md; plan
reports/plans/2026-11-19_plan_rl-67m.md (section "Stage 2c/3c (E224)"); settings, expected values and rules
results/predictions_E224.json (the run refuses unless it is committed unchanged). Tasks: `commute.rl_tasks_2191`
(1 = plural pronoun agreement, 2 = day/month sequences, anchor = single-digit addition). Arms and loop:
`commute.rl_arms`, sampling one token at the cue; the reward is the token's membership in the item's target set.

  ctl              first and alone (`E224 rehearse: newtasks ctl`): item counts per task, W0's held-out top-1 / top-5 /
                   contrast / control top-1, a zero update reads KL exactly 0 and the same sampled tokens, 5 RL steps
                   timed per task, then E211's ctl unchanged (E53's lowercase and cipher states re-read to their saved
                   shares, as E222's ctl did)
  null LR          the shuffled arm at one rate, seed 0, tasks 1 and 2 (E220's stable-null rule on held-out top-1)
  arm P A [S..]    task P (1, 2, anchor), arm A (rl, shuffled, own, demo; the anchor takes demo, and rl seed 0 as an
                   extra reading), seeds S (default 0 1 2). Every arm but rl (and the anchor's rl) refuses unless RL seed
                   0 passed the learned gate on tasks 1 and 2 (`learned`)
Held-out items: Pile val rows (from val_start) for the prompts or fills, and for sequences held-out templates; the 55
addition facts are shared between the splits (only the fill differs). The held-out read is deterministic (argmax and
logits at the cue on the first n_eval held-out items and their controls): at step 0, step 10 and the stop (on the saved
bf16 state). Files: results/stage2c_<job>.json (job = argv joined by "_"); states
results/large/stage2c/<task>_<arm>_s<seed>.pt ({matrix: dW}, bf16, sha256 in the record).
    .venv/bin/python experiments/rl_67m/run_newtasks.py ctl
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import rl_arms as RA
from commute import rl_tasks_2191 as NT
from commute import steering as ST
from commute import vpd
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, LARGE, READS = RESULTS / "predictions_E224.json", RESULTS / "large" / "stage2c", RESULTS / "stage3c"
S1 = load_experiment("rl_67m", "run_stage1")  # kl_from_base
R2 = load_experiment("realistic_tasks", "run2")  # E211's reader, ctl, zero check, Out
NAMES = {"1": "plural", "2": "sequence", "anchor": "addition"}
ARMS = {"1": RA.ARMS, "2": RA.ARMS, "anchor": ("demo", "rl")}


@dataclass(frozen=True)
class Settings:
    lr: float
    grad_clip: float
    prompts_per_step: int
    k: int
    sft_batch: int
    size_target: float
    max_steps: int
    kl_every: int
    kl_rows: int
    kl_batch: int
    prompt_len: int
    gap: int
    seq_lo: int
    seq_hi: int
    held_frac: float
    n_train: int
    n_held: int
    n_eval: int
    n_own: int
    pile_train_n: int
    val_start: int
    val_n: int
    held_extra_start: int
    held_extra_n: int
    n_read: int
    n_fit: int
    gauss_draw: int
    null_rates: tuple[float, ...]
    seeds: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        for f in ("null_rates", "seeds"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        assert list(self.null_rates) == sorted(self.null_rates) and self.n_eval <= self.n_held and self.n_read <= self.n_held
        assert self.n_own <= self.n_train and self.n_fit <= self.n_train and self.n_held <= self.val_n and self.n_train <= self.pile_train_n
        assert self.held_extra_start >= self.pile_train_n  # plural held-out rows never overlap the training rows


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def build_items(v: NT.Vocab, C: Settings, train_rows: Tensor, val_rows: Tensor, extra_rows: Tensor) -> dict[str, dict[str, Any]]:
    """Per task: train and held-out items (held out: Pile val rows, and for sequences also held-out templates; plural
    mines the val rows then Pile train rows from held_extra_start, which the training rows never reach)."""
    tr, ht = NT.plural_items(train_rows, v, C.prompt_len, C.gap, C.n_train, C.seed)
    he, hs = NT.plural_items(torch.cat([val_rows, extra_rows]), v, C.prompt_len, C.gap, C.n_held, C.seed)
    cores = NT.sequence_cores(C.seq_lo, C.seq_hi, C.held_frac, C.seed)
    return {"plural": {"train": tr, "held": he, "scanned": {"train": ht, "held": hs}},
            "sequence": {"train": NT.sequence_items(train_rows, cores["train"], v, C.prompt_len, C.n_train, C.seed),
                         "held": NT.sequence_items(val_rows, cores["held"], v, C.prompt_len, C.n_held, C.seed),
                         "cores": {k: len(c) for k, c in cores.items()}},
            "addition": {"train": NT.addition_items(train_rows, v, C.prompt_len, C.n_train, C.seed),
                         "held": NT.addition_items(val_rows, v, C.prompt_len, C.n_held, C.seed)}}  # fmt: skip


def counts(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for t, d in items.items():
        tk, hk = set(d["train"].key), set(d["held"].key)
        out[t] = {"train": len(d["train"].toks), "held": len(d["held"].toks), "distinct keys": {"train": len(tk), "held": len(hk)},
                  "keys in both": len(tk & hk)} | {k: d[k] for k in ("scanned", "cores") if k in d}  # fmt: skip
    return out


class Ctx:
    def __init__(self) -> None:
        require_filed(PRED)
        P = json.loads(PRED.read_text())
        assert P["experiment"].startswith("E224")
        self.C, self.th, self.P = config_from_settings(Settings, P["settings"]), P["thresholds"], P
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        self.net, self.tok = ST.Net("cuda"), vpd.tokenizer()
        self.v = NT.Vocab([self.tok.decode([i]) for i in range(len(self.tok))], self.tok.encode)
        self.kl_rows = cached_rows("val", 0, self.C.kl_rows)
        val = cached_rows("val", self.C.val_start, self.C.val_n)
        extra = cached_rows("train", self.C.held_extra_start, self.C.held_extra_n)
        self.items = build_items(self.v, self.C, cached_rows("train", 0, self.C.pile_train_n), val, extra)

    def gen(self, seed: int, salt: int) -> torch.Generator:
        return torch.Generator(device=self.net.device).manual_seed(1_000_000 * salt + seed)


# ----------------------------------------------------------------------------- reads of the live or saved weights


@torch.no_grad()
def last_logits(net: Any, toks: Tensor, batch: int = 128) -> Tensor:
    dev = next(net.parameters()).device
    return torch.cat([RA.logits(net, toks[s : s + batch].to(dev))[:, -1] for s in range(0, len(toks), batch)])


def held_read(x: Ctx, it: NT.Items) -> dict[str, Any]:
    """Means and standard errors over the held-out items of top-1, top-5, contrast and the control's top-1; and the
    collapse read: the share of the KL rows' positions whose argmax is their most common argmax."""
    net = x.net.model.target_model
    r = NT.reads(last_logits(net, it.toks), it, last_logits(net, it.ctrl))
    out: dict[str, Any] = {k: float(v.mean()) for k, v in r.items()} | {f"se_{k}": float(v.std(unbiased=False)) / len(v) ** 0.5 for k, v in r.items()}
    with torch.no_grad():
        am = torch.cat([RA.logits(net, b.to(x.net.device)).argmax(-1).flatten() for b in x.kl_rows.split(x.C.kl_batch)])
    return out | {"n": len(it.toks), "mode_share": float(am.bincount().max()) / len(am)}


def se_diff(a: dict[str, Any], b: dict[str, Any], k: str) -> float:
    return (a[f"se_{k}"] ** 2 + b[f"se_{k}"] ** 2) ** 0.5


def step_loss_fn(x: Ctx, arm: str, tp: NT.Items, pool: tuple[Tensor, Tensor] | None, seed: int) -> Callable[[int], Tensor]:
    C, net = x.C, x.net.model.target_model
    pick, gen = torch.Generator().manual_seed(seed), x.gen(seed, 7)
    shuf = torch.Generator().manual_seed(10_000 + seed) if arm == "shuffled" else None

    def pg(step: int) -> Tensor:
        idx = torch.randint(0, len(tp.toks), (C.prompts_per_step,), generator=pick)
        prompts = tp.toks[idx].repeat_interleave(C.k, 0)
        ids = RA.sample(net, tp.toks[idx], C.k, 1, len(x.tok), gen)
        adv = RA.advantages(NT.member(ids[:, 0], tp.tgt[idx].repeat_interleave(C.k, 0)).float(), C.k, shuf)
        return -(adv.to(x.net.device) * RA.cont_logp(net, prompts, ids)).mean()

    def sft(step: int) -> Tensor:
        assert pool is not None
        idx = torch.randint(0, len(pool[0]), (C.sft_batch,), generator=pick)
        return -RA.cont_logp(net, pool[0][idx], pool[1][idx]).mean()

    return pg if arm in ("rl", "shuffled") else sft


def pool_of(x: Ctx, arm: str, tp: NT.Items, seed: int) -> tuple[Tensor, Tensor] | None:
    """own: W0's sampled tokens on the first n_own training prompts that earn the reward; demo: every training item
    with its correct token."""
    if arm == "demo":
        return tp.toks, tp.true[:, None]
    if arm != "own":
        return None
    p, gen = tp.toks[: x.C.n_own], x.gen(seed, 3)
    ids = torch.cat([RA.sample(x.net.model.target_model, p[s : s + 128], x.C.k, 1, len(x.tok), gen) for s in range(0, len(p), 128)])
    keep = NT.member(ids[:, 0], tp.tgt[: x.C.n_own].repeat_interleave(x.C.k, 0))
    return p.repeat_interleave(x.C.k, 0)[keep], ids[keep]


def run_arm(x: Ctx, task: str, arm: str, seed: int, lr: float, save: bool, max_steps: int | None = None) -> dict[str, Any]:
    C, d = x.C, x.items[task]
    held = d["held"].take(slice(0, C.n_eval))
    pool = pool_of(x, arm, d["train"], seed)
    out: dict[str, Any] = {"task": task, "arm": arm, "seed": seed, "lr": lr, "pool": None if pool is None else len(pool[0])}
    if pool is not None and len(pool[0]) < x.th["min_pool"]:
        return out | {"skipped": f"pool of {len(pool[0])} under {x.th['min_pool']}"}
    reads = {"0": held_read(x, held)}

    def on_step(step: int) -> None:
        if step == 10:
            reads["10"] = held_read(x, held)

    tr =RA.Train(lr, C.grad_clip, max_steps or C.max_steps, C.kl_every, C.size_target if max_steps is None else 1.0, seed)
    res = RA.train(x.net.model, step_loss_fn(x, arm, d["train"], pool, seed), tr, x.kl_rows, C.kl_batch, on_step, log)
    saved = {k: v.to(torch.bfloat16).float() for k, v in res.pop("dW").items()}
    with x.net.at(saved):  # the stop is read on the state as saved
        reads["stop"] = held_read(x, held)
    out |= res | {"size_saved": RA.size(saved, x.net.W0), "reads": reads, "collapsed": reads["stop"]["mode_share"] >= x.th["collapse_share"]}
    if save:
        LARGE.mkdir(parents=True, exist_ok=True)
        path = LARGE / f"{task}_{arm}_s{seed}.pt"
        torch.save({k: v.to(torch.bfloat16) for k, v in saved.items()}, path)
        out["state"] = {"file": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        out["kl_saved"] = S1.kl_from_base(x.net, x.kl_rows, saved, C.kl_batch)
    r0, rs = reads["0"], reads["stop"]
    log(f"[{task} {arm} s{seed}] steps {res['steps']} size {res['size']:.5f} reached {res['reached']} kl {res['kl']:.4e} top-1 "
        f"{r0['top1']:.3f} -> {rs['top1']:.3f} contrast {r0['contrast']:.3f} -> {rs['contrast']:.3f} control {r0['ctrl_top1']:.3f} -> "
        f"{rs['ctrl_top1']:.3f} collapsed {out['collapsed']}")  # fmt: skip
    return out


# ----------------------------------------------------------------------------- training jobs


def job_ctl(x: Ctx, o: Any) -> None:
    o.put("counts", counts(x.items))
    if short := {t: len(d["held"].toks) for t, d in x.items.items() if len(d["held"].toks) < x.th["min_items"]}:
        raise o.stop(f"under {x.th['min_items']} held-out items: {short}")
    if x.items["sequence"]["held"].key and set(x.items["sequence"]["held"].key) & set(x.items["sequence"]["train"].key):
        raise o.stop("a held-out sequence template is also a training template")
    o.put("w0", {t: held_read(x, d["held"]) | {"task2191": x.P["expected"]["task2191 top-1 / top-5"][t]} for t, d in x.items.items()})
    net, zero = x.net.model.target_model, {k: torch.zeros_like(w) for k, w in x.net.W0.items()}
    p = x.items["plural"]["held"].toks[:64]
    a = RA.sample(net, p, 8, 1, len(x.tok), x.gen(0, 9))
    with x.net.at(zero):
        b = RA.sample(net, p, 8, 1, len(x.tok), x.gen(0, 9))
    z = {"kl": S1.kl_from_base(x.net, x.kl_rows, zero, x.C.kl_batch), "same_tokens": bool(torch.equal(a, b))}
    o.put("zero update", z)
    if not (z["kl"] == 0.0 and z["same_tokens"]):
        raise o.stop("a zero update does not read as W0")
    o.put("seconds per RL step", {t: run_arm(x, t, "rl", 0, x.C.lr, False, max_steps=5)["seconds"] / 5 for t in ("plural", "sequence")})
    P2 = json.loads(R2.PRED.read_text())  # E211's reader settings and reproduce tolerances
    R2.ctl(x.net, config_from_settings(R2.Config, P2["settings"]), P2["thresholds"], o)


def learned(x: Ctx, o: Any) -> None:
    """Stops unless RL seed 0 (jobs `arm 1 rl 0`, `arm 2 rl 0`) reached the size target, raised held-out top-1 by more
    than learn_min_se standard errors, and did not raise the control's top-1 by more than ctrl_max_se, on both tasks."""
    got = {}
    for p in ("1", "2"):
        s = json.loads((RESULTS / f"stage2c_arm_{p}_rl_0.json").read_text())["seeds"]["0"]
        r0, rs = s["reads"]["0"], s["reads"]["stop"]
        got[p] = {"reached": s["reached"], "gain": rs["top1"] - r0["top1"], "se": se_diff(rs, r0, "top1"),
                  "ctrl_rise": rs["ctrl_top1"] - r0["ctrl_top1"], "ctrl_se": se_diff(rs, r0, "ctrl_top1")}  # fmt: skip
    o.put("learned", got)
    ok = all(g["reached"] and g["gain"] > x.th["learn_min_se"] * g["se"] and g["ctrl_rise"] <= x.th["ctrl_max_se"] * g["ctrl_se"] for g in got.values())
    if not ok:
        raise o.stop("RL seed 0 fails the learned gate (top-1 up by over 2 se, control not up by over 2 se) on a task")


def job_null(x: Ctx, lr: float, o: Any) -> None:
    for p in ("1", "2"):
        res = run_arm(x, NAMES[p], "shuffled", 0, lr, save=False)
        r0, rs = res["reads"]["0"], res["reads"]["stop"]
        d, tol = rs["top1"] - r0["top1"], x.th["null_tol_se"] * se_diff(rs, r0, "top1")
        o.put(p, res | {"top1_change": d, "tol": tol, "stable": bool(res["reached"] and abs(d) <= tol and not res["collapsed"])})
    o.d["stable"] = all(o.d[p]["stable"] for p in ("1", "2"))


def null_lr(x: Ctx, o: Any) -> float:
    stable = [lr for lr in x.C.null_rates if json.loads((RESULTS / f"stage2c_null_{lr:g}.json").read_text()).get("stable")]
    if not stable:
        raise o.stop("no rate gives a stable shuffled-reward null on both tasks: stop the cell")
    return min(stable)


def job_arm(x: Ctx, p: str, arm: str, seeds: tuple[int, ...], o: Any) -> None:
    assert arm in ARMS[p] and (p != "anchor" or arm == "demo" or seeds == (0,)), (p, arm, seeds)
    if arm != "rl":
        learned(x, o)
    lr = null_lr(x, o) if arm == "shuffled" else x.C.lr
    for seed in seeds:
        o.put(str(seed), run_arm(x, NAMES[p], arm, seed, lr, save=True), under="seeds")


def main(argv: list[str]) -> int:
    x, job = Ctx(), "_".join(argv)
    o = R2.Out(job)
    o.path, o.d["experiment"] = RESULTS / f"stage2c_{job}.json", x.P["experiment"]
    try:
        if argv[0] == "ctl":
            job_ctl(x, o)
        elif argv[0] == "null":
            job_null(x, float(argv[1]), o)
        else:
            job_arm(x, argv[1], argv[2], tuple(int(s) for s in argv[3:]) or x.C.seeds, o)
    except R2.Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    o.d["done"] = True
    o.write()
    print(f"E224 {job} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
