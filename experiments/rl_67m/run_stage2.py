"""E220 (rl_67m), cell P2.41 stage 2b: the arms matched on update size. Plan reports/plans/2026-11-19_plan_rl-67m.md
(section "Stage 2b"); filed settings, bars and expected values results/predictions_E220.json (the run refuses unless it
is committed unchanged). E218 ran this code with a KL stop (git history); its stage2_*.json files stay as they ran.
Arms and loop: commute.rl_arms. Behaviours: the two picks and the anchor in E215's results/stage1.json.

  ctl            first and alone: a zero update reads as W0 (stage 1's KL read exactly 0, the same sampled tokens);
                 E53's lowercase and cipher seed-0 states re-read by the stage 3 readers to their saved numbers (E209's
                 stage 0, unchanged, writing to results/stage2b_e53_reread/)
  null LR        one rate of the shuffled-reward grid, seed 0, both picks: stable when it reaches the size target with
                 the held-out reward within null_tol_se standard errors of step 0's and not collapsed. The shuffled arm
                 takes the lowest rate stable on both picks; none = stop the cell
  arm P A [S..]  behaviour P (1, 2: the picks; anchor) and arm A (rl, shuffled, own, demo; the anchor takes demo only),
                 seeds S (default 0 1 2), each stopped at the first step whose update size is at or above size_target
                 (settings: E218's smallest RL state) and saved in bf16. Every arm but rl stops unless RL seed 0
                 (jobs `arm 1 rl 0`, `arm 2 rl 0`) raised the held-out reward by more than learn_min_se standard errors
                 on both picks at the target
Order: ctl alone; then the three nulls and `arm P rl 0`; then the rest of rl, own, demo and the anchor; the shuffled
arm after the nulls. Each job writes results/stage2b_<job>.json as each run lands and prints "STAGE 2 <job> DONE";
states go to results/large/stage2b/<behaviour>_<arm>_s<seed>.pt ({matrix: dW}, bf16, E53's and E211's format).

    .venv/bin/python experiments/rl_67m/run_stage2.py ctl
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

from commute import lowercase, vpd
from commute import rl_arms as RA
from commute import rl_tasks as RT
from commute import steering as ST
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
TAG = "stage2b"  # E220's files; E218's stage2_* stay as they ran
LARGE = RESULTS / "large" / TAG
PRED = RESULTS / "predictions_E220.json"
WINDOWS = REPO / "experiments/transport_test/results/large/windows.pt"
E209 = load_experiment("realistic_tasks", "run")  # its stage 0: E53's states re-read by the scorecard's readers
S1 = load_experiment("rl_67m", "run_stage1")  # stage 1's KL read (kl_from_base), unchanged


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
    n_eval: int
    eval_k: int
    n_train_prompts: int
    n_demo_prompts: int
    pile_train_n: int
    null_rates: tuple[float, ...]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        for f in ("null_rates", "seeds"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        assert 0 < self.size_target < 1 and list(self.null_rates) == sorted(self.null_rates)


class Stopped(Exception):
    pass


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def write(job: str, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"{TAG}_{job}.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(RESULTS / f"{TAG}_{job}.json")


def check(job: str, out: dict[str, Any], ok: bool, why: str) -> None:
    if not ok:
        out["stopped"] = why
        write(job, out)
        raise Stopped(why)


class Ctx:
    """The model, stage 1's record and the settings every job shares."""

    def __init__(self) -> None:
        require_filed(PRED)
        P = json.loads(PRED.read_text())
        assert P["experiment"].startswith("E220")
        self.C, self.th = config_from_settings(Settings, P["settings"]), P["thresholds"]
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        self.s0 = json.loads((RESULTS / "stage0.json").read_text())
        self.picks = json.loads((RESULTS / "stage1.json").read_text())["calls"]
        self.st1 = json.loads((RESULTS / "predictions.json").read_text())["settings"]
        self.samples = torch.load(RESULTS / "large" / "stage1_samples.pt", weights_only=False)
        self.net, self.tok = ST.Net("cuda"), vpd.tokenizer()
        self.texts = [self.tok.decode([i]) for i in range(len(self.tok))]
        (self.o_id,) = self.tok.encode("o")
        self.windows = torch.load(WINDOWS, weights_only=False)
        self.kl_rows = cached_rows("val", 0, self.C.kl_rows)
        self.pile_train = cached_rows("train", 0, self.C.pile_train_n)
        self.names = {"1": self.picks["picks"][0], "2": self.picks["picks"][1], "anchor": self.picks["anchor"]}
        self.articles = {"fr": self.s0["prompts"]["french_fr"]["scanned"], "de": self.s0["prompts"]["german_de"]["scanned"]}

    def reader(self, name: str) -> RA.Reader:
        b, s = next(b for b in RT.BEHAVIOURS if b.name == name), self.st1
        sz = RT.Sizes(s["prompt_len"], s["n_new"], s["n_prompts"], s["n_copy"], s["n_end"], s["count_lo"], s["count_hi"], s["seed"])
        return RA.Reader(b, self.tok, self.texts, sz, self.o_id)

    def held(self, name: str) -> RT.Prompts:
        """Stage 1's held-out prompts, the first n_eval (E1's held-out windows for the emoticon)."""
        n = self.C.n_eval
        if name == "emoticon":
            w = self.windows["heldout"]
            return RT.window_prompts(w.tokens, w.half, RT.Sizes(0, self.st1["n_new"], n, 0, 0, 0, 0, 0))
        s = self.samples[name]
        return RT.Prompts(s["prompts"][:n].long(), s["true"][:n].long(), list(s["extra"][:n]))

    def gen(self, seed: int, salt: int) -> torch.Generator:
        return torch.Generator(device=self.net.device).manual_seed(1_000_000 * salt + seed)


# ----------------------------------------------------------------------------- one arm


def step_loss_fn(x: Ctx, r: RA.Reader, arm: str, tp: RT.Prompts, pool: tuple[Tensor, Tensor] | None, seed: int) -> Callable[[int], Tensor]:
    C, net = x.C, x.net.model.target_model
    pick, gen = torch.Generator().manual_seed(seed), x.gen(seed, 7)
    shuf = torch.Generator().manual_seed(10_000 + seed) if arm == "shuffled" else None

    def pg(step: int) -> Tensor:
        idx = torch.randint(0, len(tp.toks), (C.prompts_per_step,), generator=pick)
        prompts = tp.toks[idx]
        ids = RA.sample(net, prompts, C.k, r.sz.n_new, len(r.tok), gen)
        adv = RA.advantages(r.rewards(ids, [tp.extra[i] for i in idx.tolist()], C.k), C.k, shuf).to(x.net.device)
        return -(adv * RA.cont_logp(net, prompts.repeat_interleave(C.k, 0), ids)).mean() / r.sz.n_new

    def sft(step: int) -> Tensor:
        assert pool is not None
        idx = torch.randint(0, len(pool[0]), (C.sft_batch,), generator=pick)
        return -RA.cont_logp(net, pool[0][idx], pool[1][idx]).mean() / r.sz.n_new

    return pg if arm in ("rl", "shuffled") else sft


def run_arm(x: Ctx, name: str, arm: str, seed: int, lr: float, save: bool) -> dict[str, Any]:
    C, r, net = x.C, x.reader(name), x.net.model.target_model
    held, tp = x.held(name), RA.train_prompts(r, x.pile_train, x.articles, x.windows, C.n_train_prompts)
    pool = None
    if arm == "own":
        pool = RA.own_pool(net, r, tp, C.k, C.prompts_per_step, x.gen(seed, 3))
    elif arm == "demo":
        pool = RA.demo_pool(r, RA.train_prompts(r, x.pile_train, x.articles, x.windows, C.n_demo_prompts), x.articles)
    out: dict[str, Any] = {"arm": arm, "behaviour": name, "seed": seed, "lr": lr, "pool": None if pool is None else len(pool[0])}
    if pool is not None and len(pool[0]) < x.th["min_pool"]:
        log(f"[{name} {arm} s{seed}] skipped: pool of {len(pool[0])}")
        return out | {"skipped": f"pool of {len(pool[0])} under {x.th['min_pool']}"}
    read = lambda: r.held_read(net, held, C.eval_k, C.prompts_per_step, x.gen(seed, 5))
    reads = {"0": read()}

    def on_step(step: int) -> None:
        if step == 10:
            reads["10"] = read()
            log(f"  [{name} {arm} s{seed}] step 10 rate {reads['10']['rate']:.4f}: {reads['10']['examples'][0]['sample']!r}")

    tr = RA.Train(lr, C.grad_clip, C.max_steps, C.kl_every, C.size_target, seed)
    res = RA.train(x.net.model, step_loss_fn(x, r, arm, tp, pool, seed), tr, x.kl_rows, C.kl_batch, on_step, log)
    saved = {k: v.to(torch.bfloat16).float() for k, v in res.pop("dW").items()}
    with x.net.at(saved):  # the stop is read on the state as saved
        reads["stop"] = read()
    out |= res | {"size_saved": RA.size(saved, x.net.W0), "reads": reads,
                  "collapsed": reads["stop"]["collapsed"] >= x.th["collapse_share"],
                  "kl_saved": S1.kl_from_base(x.net, x.kl_rows, saved, C.kl_batch)}  # fmt: skip
    if save:
        LARGE.mkdir(parents=True, exist_ok=True)
        path = LARGE / f"{name}_{arm}_s{seed}.pt"
        torch.save({k: v.to(torch.bfloat16) for k, v in saved.items()}, path)
        out["state"] = {"file": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    log(f"[{name} {arm} s{seed}] steps {res['steps']} size {res['size']:.5f} (saved {out['size_saved']:.5f}) reached {res['reached']} "
        f"kl {res['kl']:.4e} (saved {out['kl_saved']:.4e}) rate {reads['0']['rate']:.4f} -> {reads['stop']['rate']:.4f} "
        f"collapsed {out['collapsed']}")  # fmt: skip
    for e in reads["stop"]["examples"]:
        log(f"    ...{e['prompt_tail']!r} -> {e['sample']!r} ({e['reward']:.0f})")
    return out


def se(read: dict[str, Any]) -> float:
    return read["sd_over_prompts"] / read["n_prompts"] ** 0.5


def se_diff(a: dict[str, Any], b: dict[str, Any]) -> float:
    return (se(a) ** 2 + se(b) ** 2) ** 0.5


# ----------------------------------------------------------------------------- jobs


def job_ctl(x: Ctx) -> None:
    out: dict[str, Any] = {"names": x.names}
    net, C = x.net.model.target_model, x.C
    zero = {k: torch.zeros_like(w) for k, w in x.net.W0.items()}
    held = x.held(x.names["1"])
    a = RA.sample(net, held.toks[:32], 8, held.true.shape[1], len(x.tok), x.gen(0, 9))
    with x.net.at(zero):
        b = RA.sample(net, held.toks[:32], 8, held.true.shape[1], len(x.tok), x.gen(0, 9))
    out["zero_kl"], out["zero_same_tokens"] = S1.kl_from_base(x.net, x.kl_rows, zero, C.kl_batch), bool(torch.equal(a, b))
    log(f"[ctl] zero update: kl {out['zero_kl']}, same sampled tokens {out['zero_same_tokens']}")
    check("ctl", out, out["zero_kl"] == 0.0 and out["zero_same_tokens"], "a zero update does not read as W0")
    E209.RESULTS = RESULTS / f"{TAG}_e53_reread"  # E209's stage 0 writes its stage0.json here, not over E209's own
    E209.RESULTS.mkdir(exist_ok=True)
    pieces = x.tok.convert_ids_to_tokens(list(range(len(x.tok))))
    V = int(x.net.run(torch.zeros(1, 2, dtype=torch.long))[0].shape[-1])
    tabs = E209.Tabs(E209.R.byte_lengths(pieces, x.tok.all_special_ids, V).to(x.net.device), E209.R.word_starts(pieces, V).to(x.net.device))
    th209 = json.loads((REPO / "experiments/realistic_tasks/results/predictions.json").read_text())["thresholds"]
    try:
        out["e53_reread"] = E209.stage0(x.net, th209, lowercase.task(x.net.model, x.net.device, log), tabs)["reproduce"]
    except E209.Stopped as e:
        check("ctl", out, False, f"E53 re-read: {e}")
    write("ctl", out)


def learned(x: Ctx, job: str) -> dict[str, Any]:
    """Stops unless RL seed 0 (jobs `arm P rl 0`) reached the size target and its held-out reward at the stop rose by
    more than learn_min_se standard errors over step 0's, on both picks."""
    got = {}
    for p in ("1", "2"):
        s = json.loads((RESULTS / f"{TAG}_{p}_rl_s0.json").read_text())["seeds"]["0"]
        r0, rs = s["reads"]["0"], s["reads"]["stop"]
        got[p] = {"steps": s["steps"], "reached": s["reached"], "gain": rs["rate"] - r0["rate"], "se": se_diff(rs, r0)}
    log(f"[learned] RL seed 0 at the size target {json.dumps(got)}")
    ok = all(v["reached"] and v["gain"] > x.th["learn_min_se"] * v["se"] for v in got.values())
    check(job, {"learned": got}, ok, "RL does not raise the held-out reward by the size target on both picks")
    return got


def job_null(x: Ctx, lr: float) -> None:
    """One rate of the shuffled-reward grid, seed 0, both picks."""
    job = f"null_{lr:g}"
    out: dict[str, Any] = {"size_target": x.C.size_target, "lr": lr, "picks": {}}
    for p in ("1", "2"):
        res = run_arm(x, x.names[p], "shuffled", 0, lr, save=False)
        d, tol = res["reads"]["stop"]["rate"] - res["reads"]["0"]["rate"], x.th["null_tol_se"] * se_diff(res["reads"]["stop"], res["reads"]["0"])
        out["picks"][p] = res | {"rate_change": d, "tol": tol, "stable": bool(res["reached"] and abs(d) <= tol and not res["collapsed"])}
        write(job, out)
    out["stable"] = all(v["stable"] for v in out["picks"].values())
    write(job, out)


def null_lr(x: Ctx, job: str) -> float:
    """The lowest rate of the grid whose null is stable on both picks; every rate's file must be there."""
    grid = {lr: json.loads((RESULTS / f"{TAG}_null_{lr:g}.json").read_text()) for lr in x.C.null_rates}
    stable = [lr for lr, g in grid.items() if g.get("stable")]
    check(job, {"stable": stable}, bool(stable), "no rate gives a stable shuffled-reward null on both behaviours: stop the cell")
    return min(stable)


def job_arm(x: Ctx, p: str, arm: str, seeds: tuple[int, ...]) -> None:
    assert arm in RA.ARMS and (p != "anchor" or arm == "demo"), (p, arm)
    job = f"{p}_{arm}" + ("" if seeds == x.C.seeds else "_s" + "".join(map(str, seeds)))
    check(job, {}, x.names[p] is not None, f"no behaviour for {p} in stage1.json")
    out: dict[str, Any] = {"behaviour": x.names[p], "arm": arm, "size_target": x.C.size_target, "seeds": {}}
    if arm != "rl":
        out["learned"] = learned(x, job)
    lr = null_lr(x, job) if arm == "shuffled" else x.C.lr
    for seed in seeds:
        out["seeds"][str(seed)] = run_arm(x, x.names[p], arm, seed, lr, save=True)
        write(job, out)


def main(argv: list[str]) -> int:
    x = Ctx()
    job = "_".join(argv)
    try:
        if argv[0] == "ctl":
            job_ctl(x)
        elif argv[0] == "null":
            job_null(x, float(argv[1]))
        else:
            job_arm(x, argv[1], argv[2], tuple(int(s) for s in argv[3:]) or x.C.seeds)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    print(f"STAGE 2 {job} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
