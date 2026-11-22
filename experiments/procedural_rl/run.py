"""E256 (procedural_rl), cell P2.56 run 1: three procedural tasks on the 67M (day/month sequences, closing brackets
and quotes, single-digit addition), each trained by RL and by supervised training on demonstrations (SFT), three seeds,
every state stopped at one update size, with Gaussian updates matched to RL's per-matrix norms; then E222's readings of
every state. Settings, bars and expected values: results/predictions.json (the run refuses unless it is committed
unchanged). Tasks: `commute.rl_tasks` (PROCEDURAL; one right next token at a cue, so the reward and the read row agree).
Arms and loop: `commute.rl_arms` (E220's recipe, one sampled token per prompt as E224). Readers: E211's
(`realistic_reads.Reader` through experiments/realistic_tasks/run2.py's `reader`), rows `realistic_reads.CueRead`.

  ctl          stage 0, first and alone, exit 1 on a miss: item counts and the template split; a zero update reads as
               W0 (KL exactly 0, the same logits at every cue); E211's ctl unchanged (E53's lowercase and cipher states
               re-read to their saved readings); then the benchmark: W0's reward rate per task on the held-out items.
               A task outside the filed band is dropped; none left stops the cell
  train TASK   stage 1, after ctl: RL seed 0; unless its held-out reward rose by more than learn_min_se standard errors
               at the size target the task is dropped (exit 1); then RL seeds 1-2, SFT seeds 0-2, and a Gaussian update
               at each RL seed's per-matrix norms. Each state is saved the moment it lands
  read TASK    stage 2, after ctl and `train TASK`: a zero update reads every numerator exactly 0 on the task's rows; the
               three Gaussian states first (a gain at or over MIN_GAIN exits 1); then the RL and SFT states
No head or part switch-off: no circuit for these tasks is located in this 67M (E62's copy heads are for copying).
Files: results/stage0.json, results/train_<task>.json, results/read_<task>.json, results/read/<state>.json; states
results/large/states/<task>_<arm>_s<seed>.pt ({matrix: dW}, bf16, sha256 in train_<task>.json).

E### (P2.56, the guarded RL retrain; settings, bars and expected values results/predictions_guarded.json, whose shared
settings must equal E256's so that E256's stage0.json, items and SFT states stand):
  train_guarded TASK  RL seeds 0-2 as E256's, with the guard (`rl_arms.pg_loss`, `rl_arms.Train.zero_stop`): PPO's
               entropy bonus; a step whose every advantage is 0 is skipped (no Adam step), and zero_stop skipped steps
               in a row stop the seed (zero_stopped, not reached). Every seed gets the learned gate in the read's unit
               (the answer's log-odds, `steering.logodds_at`, on the same held-out rows), recorded, never an exit; the
               trace logs the reward, the modal share of the samples and of the answers, and the entropy. Then a
               Gaussian at each seed's per-matrix norms (arm randg)
  read_guarded TASK   stage 2 on train_guarded_<task>.json's states, randg first (the do-nothing stop as E256's)
Files: results/train_guarded_<task>.json, results/read_guarded_<task>.json, results/read/<task>_{rlg,randg}_s<seed>.json.

    sbatch --comment=E256 cluster/procedural_rl.sbatch ctl
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import realistic_reads as RR
from commute import rl_arms as RA
from commute import rl_tasks as RT
from commute import rl_tasks_2191 as NT
from commute import steering as ST
from commute import vpd
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, LARGE, READS = RESULTS / "predictions.json", RESULTS / "large" / "states", RESULTS / "read"
PRED_G = RESULTS / "predictions_guarded.json"
S1 = load_experiment("rl_67m", "run_stage1")  # kl_from_base: E215's KL read
R2 = load_experiment("realistic_tasks", "run2")  # E211's reader, ctl, zero check, Out, brief
ARMS, RANDOM = ("rl", "sft", "random", "rlg", "randg"), ("random", "randg")  # rlg, randg: the guarded retrain's


@dataclass(frozen=True)
class Config:
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
    seq_lo: int
    seq_hi: int
    bracket_lo: int
    bracket_hi: int
    shots: int
    held_frac: float
    n_train: int
    n_held: int
    n_fit: int
    pile_train_n: int
    val_start: int
    val_n: int
    seeds: tuple[int, ...]
    seed: int
    entropy_coef: float = 0.0  # the guard (arm rlg only); the defaults are E256's recipe
    zero_stop: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(self.seeds))
        assert 0 < self.size_target < 1 and self.n_fit <= self.n_train <= self.pile_train_n and self.n_held <= self.val_n
        assert 2 <= self.seq_lo <= self.seq_hi and 2 <= self.bracket_lo <= self.bracket_hi
        assert self.entropy_coef >= 0 and self.zero_stop >= 0


def build_items(v: NT.Vocab, C: Config, train_rows: Tensor, val_rows: Tensor) -> dict[str, dict[str, NT.Items]]:
    L, n, h = C.prompt_len, C.n_train, C.n_held
    seq = NT.sequence_cores(C.seq_lo, C.seq_hi, C.held_frac, C.seed)
    br = RT.bracket_structures(C.bracket_lo, C.bracket_hi, C.held_frac, C.seed)
    return {"sequence": {"train": NT.sequence_items(train_rows, seq["train"], v, L, n, C.seed),
                         "held": NT.sequence_items(val_rows, seq["held"], v, L, h, C.seed)},
            "brackets": {"train": RT.bracket_items(train_rows, br["train"], v, L, n, C.seed),
                         "held": RT.bracket_items(val_rows, br["held"], v, L, h, C.seed)},
            "addition": {"train": RT.addition_items(train_rows, v, L, n, C.shots, C.seed),
                         "held": RT.addition_items(val_rows, v, L, h, C.shots, C.seed)}}  # fmt: skip


class Ctx:
    def __init__(self, pred: Path) -> None:
        require_filed(pred)
        P = json.loads(pred.read_text())
        if pred == PRED_G:  # E256's stage0.json, items and SFT states stand only under E256's settings
            assert not P["experiment"].startswith("E###"), "fill in the experiment number in predictions_guarded.json"
            base = {k: v for k, v in json.loads(PRED.read_text())["settings"].items() if not k.endswith("_source")}
            assert (diff := [k for k, v in base.items() if P["settings"].get(k) != v]) == [], f"settings differ from E256's: {diff}"
        else:
            assert P["experiment"].startswith("E256"), P["experiment"]
        self.P, self.C, self.th = P, config_from_settings(Config, P["settings"]), P["thresholds"]
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        self.net, self.tok = ST.Net("cuda"), vpd.tokenizer()
        self.v = NT.Vocab([self.tok.decode([i]) for i in range(len(self.tok))], self.tok.encode)
        self.kl_rows = cached_rows("val", 0, self.C.kl_rows)
        self.items = build_items(self.v, self.C, cached_rows("train", 0, self.C.pile_train_n), cached_rows("val", self.C.val_start, self.C.val_n))

    def gen(self, seed: int, salt: int) -> torch.Generator:
        return torch.Generator(device=self.net.device).manual_seed(1_000_000 * salt + seed)


def done(name: str) -> dict[str, Any]:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


# ----------------------------------------------------------------------------- the held-out read


@torch.no_grad()
def p_answer(net: Any, toks: Tensor, answer: Tensor, n_vocab: int, batch: int = 128) -> Tensor:
    """(N,) float64: P(the sampler draws `answer` at the cue), softmax over the tokenizer's ids at temperature 1."""
    dev, out = next(net.parameters()).device, []
    for s in range(0, len(toks), batch):
        p = torch.softmax(RA.logits(net, toks[s : s + batch].to(dev))[:, -1, :n_vocab], -1)
        out.append(p.gather(1, answer[s : s + batch, None].to(dev))[:, 0].double().cpu())
    return torch.cat(out)


def held_read(x: Ctx, it: NT.Items) -> dict[str, Any]:
    """The reward rate (mean over held-out items of P(answer), the expectation of E215's sampled rate), its standard
    error over items, and the same on the control prompts (the answer where it is wrong, or the operator removed)."""
    net, n = x.net.model.target_model, len(x.tok)
    p, c = p_answer(net, it.toks, it.true, n), p_answer(net, it.ctrl, it.true, n)
    se = lambda t: float(t.std(unbiased=False)) / len(t) ** 0.5
    return {"rate": float(p.mean()), "se": se(p), "ctrl_rate": float(c.mean()), "ctrl_se": se(c), "n": len(p)}


def se_diff(a: dict[str, Any], b: dict[str, Any]) -> float:
    return (a["se"] ** 2 + b["se"] ** 2) ** 0.5


@torch.no_grad()
def held_lo(x: Ctx, it: NT.Items, batch: int = 128) -> Tensor:
    """(N,) float64: the answer's log-odds over the vocabulary at the cue (`steering.logodds_at`, the read's unit)."""
    net, dev, out = x.net.model.target_model, x.net.device, []
    for s in range(0, len(it.toks), batch):
        lg = RA.logits(net, it.toks[s : s + batch].to(dev))
        out.append(ST.logodds_at(lg, lg.shape[1] - 1, it.true[s : s + batch]))
    return torch.cat(out)


def learned_lo(x: Ctx, held: NT.Items, lo0: Tensor, rec: dict[str, Any], saved: ST.DW) -> dict[str, Any]:
    """The guarded retrain's learned gate, in the read's unit on the read's rows: the mean over held-out items of the
    answer's log-odds on the state as saved minus at W0, and its paired standard error over items (population sd /
    sqrt(N), as E256's rate se). Passes when the seed reached the size target and the gain exceeds learn_min_se x se.
    collapsed: the zero-gradient stop ended the seed on a batch not every sample of which was rewarded."""
    with x.net.at(saved):
        d = held_lo(x, held) - lo0
    zs = bool(rec.get("zero_stopped"))
    g = {"gain": float(d.mean()), "se": float(d.std(unbiased=False)) / len(d) ** 0.5, "reached": rec["reached"], "zero_stopped": zs}
    return g | {"passed": g["reached"] and g["gain"] > x.th["learn_min_se"] * g["se"], "readable": g["gain"] >= MIN_GAIN,
                "collapsed": zs and rec["trace"][-1]["reward"] < 1}  # fmt: skip


# ----------------------------------------------------------------------------- stage 0


def job_ctl(x: Ctx, o: Any) -> None:
    it = x.items
    o.put("counts", {t: {s: len(d[s].toks) for s in d} | {"keys in both": len(set(d["held"].key) & set(d["train"].key))} for t, d in it.items()})
    if leak := [t for t in ("sequence", "brackets") if o.d["counts"][t]["keys in both"]]:
        raise o.stop(f"a held-out template is also a training template: {leak}")
    if short := [t for t, d in it.items() if len(d["held"].toks) < x.C.n_held]:
        raise o.stop(f"fewer than n_held held-out items: {short}")
    net, zero = x.net.model.target_model, {k: torch.zeros_like(w) for k, w in x.net.W0.items()}
    a = RA.logits(net, it["brackets"]["held"].toks[:64].to(x.net.device))[:, -1]
    with x.net.at(zero):
        b = RA.logits(net, it["brackets"]["held"].toks[:64].to(x.net.device))[:, -1]
    o.put("zero update", z := {"kl": S1.kl_from_base(x.net, x.kl_rows, zero, x.C.kl_batch), "same_logits": bool(torch.equal(a, b))})
    if not (z["kl"] == 0.0 and z["same_logits"]):
        raise o.stop("a zero update does not read as W0")
    P2 = json.loads(R2.PRED.read_text())  # E211's reader settings and reproduce tolerances
    R2.ctl(x.net, config_from_settings(R2.Config, P2["settings"]), P2["thresholds"], o)
    lo, hi = x.th["band"]
    w0 = {t: held_read(x, d["held"]) for t, d in it.items()}
    o.put("benchmark", {t: r | {"usable": lo <= r["rate"] <= hi} for t, r in w0.items()})
    o.put("usable", usable := [t for t in RT.PROCEDURAL if o.d["benchmark"][t]["usable"]])
    if not usable:
        raise o.stop(f"no task's W0 reward rate is inside the band {x.th['band']}: stop the cell")


# ----------------------------------------------------------------------------- stage 1


def step_loss_fn(x: Ctx, arm: str, tp: NT.Items, seed: int, info: dict[str, Any]) -> Callable[[int], Tensor | None]:
    """RL: E220's policy gradient with the group-mean baseline on one sampled token (E224's); SFT: cross-entropy on
    each training item's answer (E220's demo arm); rlg: RL with the guard, writing its step's readings into info."""
    C, net = x.C, x.net.model.target_model
    pick, gen = torch.Generator().manual_seed(seed), x.gen(seed, 7)

    def pg(step: int) -> Tensor:
        idx = torch.randint(0, len(tp.toks), (C.prompts_per_step,), generator=pick)
        ids = RA.sample(net, tp.toks[idx], C.k, 1, len(x.tok), gen)
        adv = RA.advantages(RT.exact_reward(ids, tp.true[idx].repeat_interleave(C.k, 0)), C.k, None)
        return -(adv.to(x.net.device) * RA.cont_logp(net, tp.toks[idx].repeat_interleave(C.k, 0), ids)).mean()

    def pg_guarded(step: int) -> Tensor | None:
        """pg's batch; None (the step skipped) when every advantage is 0, else pg's loss minus the entropy bonus."""
        idx = torch.randint(0, len(tp.toks), (C.prompts_per_step,), generator=pick)
        ids, true = RA.sample(net, tp.toks[idx], C.k, 1, len(x.tok), gen), tp.true[idx].repeat_interleave(C.k, 0)
        r = RT.exact_reward(ids, true)
        adv = RA.advantages(r, C.k, None)
        info.update(reward=float(r.mean()), modal_share=RA.modal_share(ids), answer_modal_share=RA.modal_share(true[:, None]),
                    skipped=not bool(adv.any()), entropy=float("nan"))  # fmt: skip
        if info["skipped"]:
            return None
        loss, info["entropy"] = RA.pg_loss(net, tp.toks[idx].repeat_interleave(C.k, 0), ids, adv, C.entropy_coef)
        return loss

    def sft(step: int) -> Tensor:
        idx = torch.randint(0, len(tp.toks), (C.sft_batch,), generator=pick)
        return -RA.cont_logp(net, tp.toks[idx], tp.true[idx, None]).mean()

    return {"rl": pg, "rlg": pg_guarded, "sft": sft}[arm]


def save_state(x: Ctx, name: str, dW: ST.DW) -> dict[str, Any]:
    """Written the moment the state lands (definition-lead reads these for R_b)."""
    LARGE.mkdir(parents=True, exist_ok=True)
    path = LARGE / f"{name}.pt"
    torch.save({k: v.to(torch.bfloat16) for k, v in dW.items()}, path)
    return {"file": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_arm(x: Ctx, task: str, arm: str, seed: int) -> tuple[dict[str, Any], ST.DW]:
    C, held, tp = x.C, x.items[task]["held"], x.items[task]["train"]
    reads = {"0": held_read(x, held)}
    on_step = lambda step: reads.update({"10": held_read(x, held)}) if step == 10 else None
    info: dict[str, Any] = {}  # stays empty for E256's arms, so their trace is as it ran
    tr = RA.Train(C.lr, C.grad_clip, C.max_steps, C.kl_every, C.size_target, seed, C.zero_stop if arm == "rlg" else 0)
    loss = step_loss_fn(x, arm, tp, seed, info)
    res = RA.train(x.net.model, loss, tr, x.kl_rows, C.kl_batch, on_step, R2.log, lambda: dict(info))
    saved = {k: v.to(torch.bfloat16).float() for k, v in res.pop("dW").items()}
    return res | {"reads": reads}, saved


def finish(x: Ctx, task: str, arm: str, seed: int, rec: dict[str, Any], saved: ST.DW, o: Any) -> None:
    """The stop read on the state as saved, its KL from W0, the file; then the record."""
    name = f"{task}_{arm}_s{seed}"
    with x.net.at(saved):
        rec.setdefault("reads", {})["stop"] = held_read(x, x.items[task]["held"])
    rec |= {"arm": arm, "seed": seed, "state": save_state(x, name, saved), "size_saved": RA.size(saved, x.net.W0),
            "kl_saved": S1.kl_from_base(x.net, x.kl_rows, saved, x.C.kl_batch)}  # fmt: skip
    o.put(name, rec, under="states", shown={k: v for k, v in rec.items() if k != "trace"})


def require_usable(task: str, o: Any) -> None:
    if not done("stage0").get("done") or task not in done("stage0")["usable"]:
        raise o.stop(f"stage0.json is not done or does not list {task} as usable")


def job_train(x: Ctx, task: str, o: Any) -> None:
    require_usable(task, o)
    rl: dict[int, ST.DW] = {}
    for seed in x.C.seeds:
        rec, rl[seed] = run_arm(x, task, "rl", seed)
        finish(x, task, "rl", seed, rec, rl[seed], o)
        if seed == x.C.seeds[0]:
            r0, rs = rec["reads"]["0"], rec["reads"]["stop"]
            o.put("learned", g := {"reached": rec["reached"], "gain": rs["rate"] - r0["rate"], "se": se_diff(rs, r0)})
            if not (g["reached"] and g["gain"] > x.th["learn_min_se"] * g["se"]):
                raise o.stop(f"RL seed {seed} did not raise {task}'s held-out reward by {x.th['learn_min_se']} se at the size target: task dropped")
    for seed in x.C.seeds:
        rec, saved = run_arm(x, task, "sft", seed)
        finish(x, task, "sft", seed, rec, saved, o)
    for seed in x.C.seeds:  # matched random: a Gaussian at RL seed `seed`'s per-matrix norms
        g = {k: v.to(torch.bfloat16).float() for k, v in FT.random_direction(rl[seed], seed).items()}
        finish(x, task, "random", seed, {"matched_to": f"{task}_rl_s{seed}"}, g, o)


def job_train_guarded(x: Ctx, task: str, o: Any) -> None:
    """Every seed runs whatever its gate reads; each state and its matched Gaussian is saved the moment it lands."""
    require_usable(task, o)
    held = x.items[task]["held"].take(slice(0, x.C.n_held))
    lo0 = held_lo(x, held)
    for seed in x.C.seeds:
        rec, saved = run_arm(x, task, "rlg", seed)
        rec["learned"] = learned_lo(x, held, lo0, rec, saved)
        finish(x, task, "rlg", seed, rec, saved, o)
        o.put(f"s{seed}", rec["learned"], under="learned")
        g = {k: v.to(torch.bfloat16).float() for k, v in FT.random_direction(saved, seed).items()}
        finish(x, task, "randg", seed, {"matched_to": f"{task}_rlg_s{seed}"}, g, o)


# ----------------------------------------------------------------------------- stage 2


def load_state(rec: dict[str, Any], o: Any) -> ST.DW:
    path = REPO / rec["state"]["file"]
    if (sha := hashlib.sha256(path.read_bytes()).hexdigest()) != rec["state"]["sha256"]:
        raise o.stop(f"{path} has sha256 {sha}, train recorded {rec['state']['sha256']}")
    return {k: v.float() for k, v in torch.load(path, weights_only=False).items()}


def job_read(x: Ctx, task: str, o: Any, train: str = "train") -> None:
    tr = done(f"{train}_{task}")
    if not (done("stage0").get("done") and tr.get("done")):
        raise o.stop(f"stage0.json or {train}_{task}.json is not done")
    C, d = x.C, x.items[task]
    held, fit = d["held"].take(slice(0, C.n_held)), d["train"].take(slice(0, C.n_fit))
    r = RR.CueRead(task, held.toks, held.true, C.prompt_len - 1, fit.toks, fit.true)
    P2 = json.loads(R2.PRED.read_text())
    rd = R2.reader(x.net, config_from_settings(R2.Config, P2["settings"]), r)
    R2.zero_check(o, rd)
    states = tr["states"]
    for name in sorted(states, key=lambda n: (ARMS[::-1].index(states[n]["arm"]), n)):  # random first: the do-nothing reference
        s = states[name]
        rec = {"task": task, "arm": s["arm"], "seed": s["seed"], "file": s["state"]["file"], "kl_saved": s["kl_saved"]} | rd.state(load_state(s, o), s["seed"])
        READS.mkdir(parents=True, exist_ok=True)
        (READS / f"{name}.json").write_text(json.dumps(jsonable(rec), indent=1))
        o.put(name, R2.brief(rec), under="read")
        if s["arm"] in RANDOM and not rec["gain"]["lo"] < MIN_GAIN:
            raise o.stop(f"a Gaussian update gains {rec['gain']['lo']:.3f} log-odds on {task}, at or over {MIN_GAIN}")


JOBS: dict[str, Callable[[Ctx, str, Any], None]] = {"train": job_train, "read": job_read, "train_guarded": job_train_guarded,
                                                    "read_guarded": lambda x, t, o: job_read(x, t, o, "train_guarded")}  # fmt: skip


def main(argv: list[str]) -> int:
    if not (argv == ["ctl"] or (len(argv) == 2 and argv[0] in JOBS and argv[1] in RT.PROCEDURAL)):
        sys.exit(f"usage: run.py ctl | {' | '.join(f'{j} TASK' for j in JOBS)}, TASK in {RT.PROCEDURAL}")
    x, job = Ctx(PRED_G if argv[0].endswith("_guarded") else PRED), "_".join(argv)
    o = R2.Out(job)
    o.path, o.d["experiment"] = RESULTS / f"{'stage0' if job == 'ctl' else job}.json", x.P["experiment"]
    try:
        job_ctl(x, o) if job == "ctl" else JOBS[argv[0]](x, argv[1], o)
    except R2.Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    o.d["done"] = True
    o.write()
    print(f"{x.P['experiment'].split()[0]} {job} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
