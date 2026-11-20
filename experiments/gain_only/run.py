"""P2.46: gain-only training on the 67M. The base model frozen, one trainable gain per MLP neuron and per attention head
(`commute.gains`), trained on each task of the nine built pairs (E187) and the scorecard's four twins with the reference
full fine-tune's own data, loss, batch, sampler seed and clip; the parameter set and the learning rate are what differ.
Plan reports/plans/2026-11-20_plan_p2-46-gain-only.md; settings, expected values and rules in results/predictions.json
(the run refuses unless it is committed unchanged). Descriptive of which side reaches what; no reuse or new verdict.

Each run stops at the first check where its trained-answer gain and its KL to the base model, both on the reference's
read rows (E136/E147's, as E228 reads them), reach every reference's mean (the task's saved full fine-tunes), or at its
cap. It keeps the gains at each first crossing ("gain:<ref>", "kl:<ref>"), or, when one is never reached, at the check
with the largest value. Stage 2 writes each kept state as a weight change in the saved fine-tunes' format (bf16, all 24
matrices) and reads it from that file: gain, KL, and the re-gating share in both forms (E228's `readings`).

Jobs:
  shard I N   stage 0 (controls, alone, exit 1 on a miss) then stages 1 and 2 for every N-th (task, seed) from the I-th;
              each item is written as it ends and a rerun skips it
  rehearse    the same at full read sizes on settings.rehearse_items with each run capped at rehearse_cap steps, into
              results/rehearse/ (run it twice: the second must skip every item)
The score, the hand-off to definition-lead and the pods' sync lists: score.py (no GPU).

    PRED_SHA=<sha> python experiments/gain_only/run.py shard 0 3 2>&1 | tee -a experiments/gain_only/results/pod0.log
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import gains as G
from commute import label_reads as LR
from commute import lowercase as LC
from commute import realistic_reads as RR
from commute import steering as ST
from commute.copying import with_target
from commute.data import LAST, pile_rows
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

E228: Any = load_experiment("random_kl_share", "run")  # catalogue, readings, Splitter, kl_of: the share as E228 reads it
WIN, PVS = E228.WIN, load_experiment("prior_vs_share", "run")  # E136's read windows; the label tasks' layout (HALF, LEN)
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E136_STAGE1 = EXPERIMENTS / "share_untrained" / "results" / "stage1.json"
CONTROL_STATE = "c3/rank1_s0"  # E147's reader control state: its fp32 gain is in E136's stage 1


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    seeds: tuple[int, ...]
    lr: float
    grad_clip: float
    kl_lambda: float
    batch_a: int  # arm A's batch (finetune.ArmAConfig)
    batch_b: int  # arm B's (E53's lowercase.BATCH)
    cap_min: int
    cap_mult: int
    check_div: int
    read_batch: int
    ref_steps: dict[str, int]  # task -> the largest saved step among its reference fine-tunes
    rand_sd: float
    rand_seed: int
    split_tol: float
    rehearse_items: tuple[str, ...]
    rehearse_cap: int

    def __post_init__(self) -> None:
        assert len(set(self.seeds)) == len(self.seeds) == 3 and self.cap_min > 0 and self.rehearse_cap < self.cap_min
        assert all(i.rsplit(":", 1)[0] in self.ref_steps for i in self.rehearse_items)

    def cap(self, task: str) -> int:
        return max(self.cap_min, self.cap_mult * self.ref_steps[task])

    def every(self, task: str) -> int:
        return max(1, self.ref_steps[task] // self.check_div)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def fname(x: str) -> str:
    return x.replace("/", "_")


def large(out: Path) -> Path:
    """The gitignored folder for `out`'s gains and states: results/large, or results/large/rehearse for the rehearsal."""
    return HERE / "results" / "large" / out.relative_to(HERE / "results")


def dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_stage(path, jsonable(obj))


def stop(path: Path, obj: dict[str, Any], why: str) -> None:
    dump(path, obj | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


# ---- the catalogue: pairs, sides, references, items -----------------------------------------------------------------


def ref_of(state: str) -> str:
    """A reference: the saved fine-tunes of one task and recipe, named by their state name without the seed."""
    return state.rsplit("_s", 1)[0]


def catalogue() -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]], dict[str, Any]]:
    """Per pair, its reference ids per side ("reuse", "new", as built); per reference its states; E228's inventory."""
    pairs, inv, _ = E228.catalogue()
    sides = {p: {"reuse": list(dict.fromkeys(map(ref_of, r))), "new": list(dict.fromkeys(map(ref_of, n)))} for p, (r, n) in pairs.items()}
    refs: dict[str, list[str]] = {}
    for s in E228.every_state(pairs):
        refs.setdefault(ref_of(s), []).append(s)
    return sides, refs, inv


def items(refs: dict[str, list[str]], inv: dict[str, Any], cfg: Config) -> list[tuple[str, int, list[str]]]:
    """(task, seed, the task's reference ids), task-major in the catalogue's order."""
    by_task: dict[str, list[str]] = {}
    for ref, states in refs.items():
        by_task.setdefault(inv["states"][states[0]]["task"], []).append(ref)
    return [(t, s, rs) for t, rs in by_task.items() for s in cfg.seeds]


def shard(todo: list[tuple[str, int, list[str]]], i: int, n: int) -> list[tuple[str, int, list[str]]]:
    """Shard i of n by task (every seed of a task on one pod, so a pod syncs only its tasks' references), tasks dealt in
    turn in the catalogue's order (one E53 task per shard at n = 3)."""
    tasks = list(dict.fromkeys(t for t, _, _ in todo))
    return [x for x in todo if tasks.index(x[0]) % n == i]


# ---- reads: the trained-answer gain and the KL to the base model on the read rows ------------------------------------


def read_rows(task: str, inv: dict[str, Any]) -> RR.Read:
    w = WIN.load_read(task, inv["tasks"][task])
    return RR.target_read(task, w.toks, w.token, w.half, w.ci_len)


@torch.no_grad()
def gain_kl(net: ST.Net, r: RR.Read, batch: int, g: G.Gains | None = None, dW: ST.DW | None = None) -> dict[str, float]:
    """Mean log-odds of the trained answer at the read position, changed minus base (r's gain T1 - T0), and E228's KL
    (changed || base), mean over rows and positions up to the read position; the change is gains g or a weight change dW."""
    lo, kl, n = 0.0, 0.0, 0
    for s in range(0, len(r.toks), batch):
        t, y = r.toks[s : s + batch, : r.pos + 1], r.tgt[s : s + batch]
        b = net.run(t)[0]
        with net.at(dW), G.applied(net.net, g) if g is not None else net.at(None):
            c = net.run(t)[0]
        lo += float((ST.logodds_at(c, r.pos, y) - ST.logodds_at(b, r.pos, y)).sum())
        p0, p1 = b.log_softmax(-1), c.log_softmax(-1)
        kl, n = kl + float((p1.exp() * (p1 - p0)).sum(-1).double().sum()), n + len(t)
    return {"gain": lo / len(r.toks), "kl": kl / (n * (r.pos + 1))}


def reference(net: ST.Net, states: list[str], inv: dict[str, Any], r: RR.Read, cfg: Config) -> dict[str, Any]:
    per = {s: gain_kl(net, r, cfg.read_batch, dW=E228.state_read(s, inv)[1]) for s in states}
    return {"states": per, "gain": fmean(v["gain"] for v in per.values()), "kl": fmean(v["kl"] for v in per.values())}


# ---- training data: each task's reference recipe (plan, "Recipes") ---------------------------------------------------


def label_train(spec: dict[str, Any]) -> FT.Windows:
    """The label pair's training windows, labelled as `prior_vs_share.label_windows` labels its read windows."""
    t, a = torch.load(EXPERIMENTS.parent / spec["path"], weights_only=False), spec["arm"]
    if spec["task"] == "case_label":
        toks = t["train"]
        y = torch.tensor([t["y"][a][int(k)] for k in toks[:, PVS.HALF - 1]])
    else:
        toks, y = t["toks"]["train"], t["y"][a]["train"]
    return LR.labelled(toks, y, t["ab"], PVS.HALF)


def e53_rows(net: ST.Net, name: str) -> Tensor:
    """E53's 2,400 training rows of `name`, as performed_anchor built them (lowercase.task; task_text with its map)."""
    if name == "lowercase":
        return LC.task(net.model, net.device, log).rows
    PA = load_experiment("performed_anchor", "run")
    pi = PA.e53_pi() if name == "cipher" else PA.a1_maps()["broad_cipher"]
    return PA.task_text(name, pile_rows("train", 0, 4096)[: PA.CFG.n_train], pi)[0]


def recipe(net: ST.Net, task: str, seed: int, inv: dict[str, Any], cfg: Config) -> tuple[Any, Any]:
    """(sample(step) -> batch on the GPU, loss_of(batch) -> (loss, info)): the reference's data, loss and sampler."""
    spec, dev = inv["tasks"][task], net.device
    if spec["format"] == "e53":  # arm B: next-token CE, rows in batch order `seed`, each once per 300 steps
        rows = LC.batch_order(e53_rows(net, spec["task"]), seed)
        n = rows.shape[0] // cfg.batch_b
        return (lambda k: rows[((k - 1) % n) * cfg.batch_b : ((k - 1) % n + 1) * cfg.batch_b].to(dev)), (lambda t: FT.next_token_ce(net.net, t))
    if spec["format"] == "kv":
        w = torch.load(EXPERIMENTS.parent / spec["path"], weights_only=False)
        train, token, ci = w["train"], w["table"], w.get("ci_len")
    elif spec["format"] == "windows_pt":  # E62: the map's target as a last column
        w = torch.load(EXPERIMENTS.parent / spec["path"], weights_only=False)
        train, token, ci = with_target(w["train"], w["target"][spec["arm"]]["train"]), LAST, w["train"].tokens.shape[1]
    else:
        assert spec["format"] == "label_task", spec
        train, token, ci = label_train(spec), LAST, PVS.LEN
    a = FT.ArmAConfig(o_token=token, kl_lambda=cfg.kl_lambda, batch=cfg.batch_a, seed=seed, ci_len=ci)
    ref, gen = copy.deepcopy(net.net).eval().requires_grad_(False), torch.Generator().manual_seed(seed)
    sample = lambda _k: train.tokens[torch.randint(0, len(train), (cfg.batch_a,), generator=gen)].to(dev)
    return sample, (lambda t: FT.arm_a_loss(net.net, ref, t, a, train.half))


# ---- stage 0: controls ----------------------------------------------------------------------------------------------


def stage0(net: ST.Net, inv: dict[str, Any], task: str, cfg: Config, th: dict[str, Any], path: Path) -> None:
    """(a) gains of 1: logits bit-identical in fp32 and in the training autocast, gain and KL on the read rows exactly 0,
        exported change exactly 0;
    (b) a random gain state: its exported change gives its logits (relative error under equiv_tol) and the same change with
        heads and neurons shuffled does not (over perm_min); (c) the read path: the control state's fp32 gain as E136's."""
    r, out = read_rows(task, inv), dict[str, Any](task=task)
    t = r.toks[: cfg.read_batch, : r.pos + 1]
    ones, base = G.Gains.ones(net.net, net.device), net.run(t)[0]
    with torch.no_grad():
        ac_base = FT.logits_of(net.net, t.to(net.device))
        with G.applied(net.net, ones):
            fp32, ac = net.run(t)[0], FT.logits_of(net.net, t.to(net.device))
    out["ones"] = {"fp32_equal": torch.equal(fp32, base), "autocast_equal": torch.equal(ac, ac_base),
                   "delta_nonzero": sum(int(v.count_nonzero()) for v in G.delta(ones, net.W0).values())} | gain_kl(net, r, cfg.read_batch, g=ones)  # fmt: skip
    gen = torch.Generator().manual_seed(cfg.rand_seed)
    g = G.Gains(1 + cfg.rand_sd * torch.randn(ones.mlp.shape, generator=gen), 1 + cfg.rand_sd * torch.randn(ones.head.shape, generator=gen)).to(net.device)
    shuf = G.Gains(g.mlp[:, torch.randperm(g.mlp.shape[1], generator=gen)], g.head[:, torch.randperm(g.head.shape[1], generator=gen)])
    with G.applied(net.net, g):
        hooked = net.run(t)[0][:, r.pos] - base[:, r.pos]
    rel = lambda dW: float((net_at(net, dW, t)[:, r.pos] - base[:, r.pos] - hooked).norm() / hooked.norm())
    out["random"] = {"equiv": rel(G.delta(g, net.W0)), "shuffled": rel(G.delta(shuf, net.W0)), "hooked_norm": float(hooked.norm())}
    cr = read_rows(inv["states"][CONTROL_STATE]["task"], inv)
    got = gain_kl(net, cr, cfg.read_batch, dW=E228.state_read(CONTROL_STATE, inv)[1])["gain"]
    want = json.loads(E136_STAGE1.read_text())["states"][CONTROL_STATE]["gain_fp32"]
    out["read_path"] = {"state": CONTROL_STATE, "gain": got, "e136_gain_fp32": want, "rel": abs(got - want) / abs(want)}
    log(f"[stage 0] {json.dumps(jsonable(out))}")
    o, rd = out["ones"], out["random"]
    if not (o["fp32_equal"] and o["autocast_equal"] and o["delta_nonzero"] == 0 and o["gain"] == 0 and o["kl"] == 0):
        stop(path, out, f"gains of 1 change the model: {o}")
    if not (rd["equiv"] <= th["equiv_tol"] and rd["shuffled"] >= th["perm_min"]):
        stop(path, out, f"the exported change is not the gain state: {rd}")
    if not out["read_path"]["rel"] <= th["reproduce_rel"]:
        stop(path, out, f"the read path does not reproduce E136: {out['read_path']}")
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


@torch.no_grad()
def net_at(net: ST.Net, dW: ST.DW, t: Tensor) -> Tensor:
    with net.at(dW):
        return net.run(t)[0]


# ---- stage 1: train one (task, seed); stage 2: export and read its kept states ---------------------------------------


def train_item(net: ST.Net, task: str, seed: int, refs: list[str], R: dict[str, list[str]], inv: dict[str, Any], cfg: Config,
               cap: int, out: Path) -> None:  # fmt: skip
    path, gpath = out / "items" / f"{fname(task)}_s{seed}.json", large(out) / "gains" / f"{fname(task)}_s{seed}.pt"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        return log(f"{task} s{seed}: trained before, skipped")
    t0, r = time.time(), read_rows(task, inv)
    rec: dict[str, Any] = {"experiment": EXPERIMENT, "task": task, "seed": seed, "cap": cap, "every": cfg.every(task), "checks": [],
                           "refs": {ref: reference(net, R[ref], inv, r, cfg) for ref in refs}}  # fmt: skip
    goals = {f"{k}:{ref}": (k, v[k]) for ref, v in rec["refs"].items() for k in ("gain", "kl")}
    kept: dict[str, dict[str, Any]] = {}
    sample, loss_of = recipe(net, task, seed, inv, cfg)

    def on_check(step: int, g: G.Gains) -> bool:
        m = gain_kl(net, r, cfg.read_batch, g=g) | {"step": step}
        rec["checks"].append(m)
        if step == 1 and m["kl"] > min(v for k, v in goals.values() if k == "kl"):
            stop(path, rec, f"{task} s{seed}: step 1 moved KL to {m['kl']:.4g}, past a reference's whole KL")
        for name, (k, v) in goals.items():
            if not kept.get(name, {}).get("reached") and (m[k] >= v or m[k] > kept.get(name, {}).get(k, -math.inf)):
                kept[name] = m | {"reached": m[k] >= v, "gains": g.cpu().state()}
        return all(x["reached"] for x in kept.values()) and len(kept) == len(goals)

    torch.cuda.reset_peak_memory_stats()
    steps = G.train(net.net, G.Gains.ones(net.net, net.device), sample, loss_of, cfg.lr, cap, cfg.grad_clip, cfg.every(task), on_check, log)
    gpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kept, gpath)
    rec |= {"steps": len(steps), "kept": {k: {x: v[x] for x in ("step", "gain", "kl", "reached")} for k, v in kept.items()}, "loss": [s["loss"] for s in steps],
            "seconds": time.time() - t0, "s_per_step": (time.time() - t0) / len(steps), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "done": True}  # fmt: skip
    dump(path, rec)
    log(f"TRAINED {task} s{seed}: {len(steps)} steps in {rec['seconds']:.0f} s; {json.dumps(jsonable(rec['kept']))}")


def read_item(net: ST.Net, sp: Any, task: str, seed: int, inv: dict[str, Any], cfg: Config, out: Path) -> None:
    path = out / "reads" / f"{fname(task)}_s{seed}.json"
    rec: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": task, "seed": seed, "states": {}}
    if rec.get("done"):
        return log(f"{task} s{seed}: read before, skipped")
    kept, r = torch.load(large(out) / "gains" / f"{fname(task)}_s{seed}.pt", weights_only=False), read_rows(task, inv)
    for name, k in kept.items():
        file = large(out) / "states" / f"{fname(task)}_s{seed}_step{k['step']}.pt"
        if str(k["step"]) not in rec["states"]:
            file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({n: v.bfloat16() for n, v in G.delta(G.Gains.of(k["gains"]), net.W0).items()}, file)
            dW = {n: v.float() for n, v in torch.load(file, weights_only=True).items()}
            rd = gain_kl(net, r, cfg.read_batch, dW=dW) | E228.readings(net, sp, r, dW)
            rd |= {"file": str(file.relative_to(EXPERIMENTS.parent)), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "mb": file.stat().st_size / 2**20,
                   "hooked": {x: k[x] for x in ("gain", "kl")}, "for": []}  # fmt: skip
            rec["states"][str(k["step"])] = rd
            log(f"[read {task} s{seed} step {k['step']}] {json.dumps(jsonable({x: rd[x] for x in ('gain', 'kl', 'kl_share', 'r', 'hooked', 'read_seconds')}))}")
        rec["states"][str(k["step"])]["for"] = sorted({*rec["states"][str(k["step"])]["for"], name})
        dump(path, rec)
    dump(path, rec | {"done": True})


def run_items(todo: list[tuple[str, int, list[str]]], R: dict[str, list[str]], inv: dict[str, Any], cfg: Config, th: dict[str, Any],
              out: Path, tag: str, cap: int | None) -> None:  # fmt: skip
    net = ST.Net("cuda")
    stage0(net, inv, todo[0][0], cfg, th, out / f"stage0_{tag}.json")
    for k, (task, seed, refs) in enumerate(todo):
        train_item(net, task, seed, refs, R, inv, cfg, cap or cfg.cap(task), out)
        log(f"ITEM {k + 1} of {len(todo)} TRAINED ({task} s{seed})")
    log("STAGE 1 DONE")
    sp = E228.Splitter(net, cfg.split_tol)
    for k, (task, seed, _) in enumerate(todo):
        read_item(net, sp, task, seed, inv, cfg, out)
        log(f"ITEM {k + 1} of {len(todo)} READ ({task} s{seed})")
    log(f"STAGE 2 DONE; {EXPERIMENT} {tag} DONE")


def main(argv: list[str]) -> int:
    require_filed(PRED)
    cfg, th = load_config()
    out = HERE / "results"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 reads, as E228
    _, refs, inv = catalogue()
    todo = items(refs, inv, cfg)
    log(f"{EXPERIMENT} {' '.join(argv)} start; settings {asdict(cfg)}; thresholds {th}; {len(todo)} items in all")
    if argv == ["rehearse"]:
        todo = [x for x in todo if f"{x[0]}:{x[1]}" in cfg.rehearse_items]
        run_items(todo, refs, inv, cfg, th, out / "rehearse", "rehearse", cfg.rehearse_cap)
    else:
        assert len(argv) == 3 and argv[0] == "shard", f"usage: run.py shard I N | rehearse, got {argv}"
        i, n = int(argv[1]), int(argv[2])
        run_items(shard(todo, i, n), refs, inv, cfg, th, out, f"{i}_of_{n}", None)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
