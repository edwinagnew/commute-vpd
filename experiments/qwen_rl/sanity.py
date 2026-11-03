"""E69's $0 sanity checks on the real model at real size, before the paid run: does the code do what it should? Rules and
expected values: results/predictions_sanity.json (refuses to start without it). Writes results/sanity.json and
sanity_samples.txt only, every number also printed; touches none of the run's files, uploads nothing. A failed check is
recorded and the next runs; exit 1 at the end if any failed. Submit: sbatch cluster/qwen_rl_sanity.sbatch
"""

from __future__ import annotations

import gc
import json
import math
import statistics as st
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from job import DEV, F32, GB, RESULTS, Job, config, write_text
from run import update
from safetensors import safe_open

from commute import qwen_eval
from commute import qwen_rl as q
from commute import qwen_sft as sft
from commute.gsm8k import gold_number, grade
from commute.jobs import hard_exit
from commute.qwen_updates import ok
from commute.qwen_vpd import HF_NAME, HF_REVISION, SITES

D = dict[str, Any]
FILED, OUT, SAMPLES = RESULTS / "predictions_sanity.json", RESULTS / "sanity.json", RESULTS / "sanity_samples.txt"
E40_DW = Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen/dW/step_100_fp32.safetensors"


class SanityJob(Job):
    def log(self, msg: str) -> None:  # the job's output only, never the run's run.log
        print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)


class Sheet:
    """Checks (a rule) and reads (no rule), written and printed as each comes."""

    def __init__(self) -> None:
        self.out: D = {"checks": {}, "reads": {}}

    def _keep(self, where: str, name: str, row: D, tag: str) -> None:
        self.out[where][name] = row
        write_text(OUT, json.dumps(self.out, indent=1, default=float))
        print(f"[{tag}] {name}: {json.dumps(row, default=float)}", flush=True)

    def check(self, name: str, holds: bool, **detail: Any) -> None:
        self._keep("checks", name, {"holds": bool(holds), **detail}, "ok" if holds else "FAILED")

    def read(self, name: str, **detail: Any) -> None:
        self._keep("reads", name, detail, "read")

    def guard(self, name: str, fn: Any, *args: Any) -> Any:
        """`fn(*args)`; if it raises, a failed check, and on to the next."""
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001  a failure is recorded and the script carries on
            self.check(f"{name}: raised", False, error=traceback.format_exc(limit=-4))
        gc.collect()
        torch.cuda.empty_cache()
        return None


def mean_se(xs: list[float]) -> tuple[float | None, float | None]:
    return (st.mean(xs) if xs else None), (st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None)


def beyond(m: float | None, se: float | None, mult: float) -> bool:
    return ok(m) and ok(se) and abs(m) > mult * se  # type: ignore[arg-type,operator]


def positive(x: float | None) -> bool:
    return ok(x) and x > 0  # type: ignore[operator]


def per_problem(values: list[float], recs: q.Records) -> list[float]:
    by: dict[int, list[float]] = defaultdict(list)
    for x, r in zip(values, recs, strict=True):
        by[r["pid"]].append(x)
    return [st.mean(v) for v in by.values()]


def gaps(changes: list[float], recs: q.Records) -> list[float]:
    """`delta_r` of each mixed problem alone: its mean is `delta_r`, its spread gives the SE."""
    return [x for idx in q.pool(recs).values() if (x := q.delta_r([changes[i] for i in idx], [recs[i] for i in idx])) is not None]


def check_data(job: Job, s: Sheet, P: D) -> None:
    cfg, sets = job.cfg, job.sets
    picks = [p for t in range(1, cfg.steps + 1) for p in job.step_pids(t)]
    s.check("6 no held-out problem in the picks", not set(picks) & set(job.held) and len(job.held) == P["held"]
            and len(picks) == len(set(picks)) == P["picks"], picks=len(picks), distinct=len(set(picks)), held=len(job.held))
    held, train = q.split(P["train"], cfg.seed, cfg.held)
    s.check("6 the picks follow the permutation", (held, train) == (job.held, job.train) and picks == train[: P["picks"]]
            and sorted(held + train) == list(range(P["train"])) and picks != sorted(picks), first=picks[:8])
    n = {k: len(v) for k, v in sets.items()}
    s.check("6 set sizes", n == {k: P[k] for k in n} and (cfg.n_gsm8k, cfg.n_gsm1k) == (P["test"], P["gsm1k"]), **n)
    for p in picks[: P["gold_shown"]]:
        gold = sets["train"][p]["gold"]
        s.read(f"6 gold of problem {p}", question=sets["train"][p]["question"], gold=gold, number=str(gold_number(gold)))


@torch.no_grad()
def fit(job: Job, recs: q.Records, top_k: int) -> D:
    """Per token on the training path (right-padded, no cache, training mode, π at temperature 1): −log p(sampled) against
    entropy, and tokens sampled below rank `top_k` against the mass there. The SE comes from each problem's mean over its
    samples of the summed difference, whose mean under π is 0 whatever the stopping rule."""
    seqs, rows = job.seqs(recs), torch.zeros(len(recs), 5, dtype=torch.float64, device=DEV)  # nll, H, below, mass, tokens
    job.model.train()
    for idx in q.micro_batches([len(x[0]) for x in seqs], job.cfg.read_tokens):
        toks, mask = sft.targets([seqs[i] for i in idx], pad=0)
        toks, mask = toks.to(DEV), mask.to(DEV)
        lp = torch.log_softmax(job.model(toks, use_cache=False).logits[:, :-1][mask], -1)
        nll, top = -lp.gather(1, toks[:, 1:][mask][:, None])[:, 0], lp.topk(top_k, -1).values
        vals = [nll, -(lp.exp() * lp).sum(-1), (-nll < top[:, -1]).float(), (1 - top.exp().sum(-1)).clamp(min=0), torch.ones_like(nll)]
        rows[idx] = rows[idx].index_add(0, mask.nonzero()[:, 0], torch.stack(vals, 1).double())
    job.model.eval()
    a, m_se = rows.tolist(), mean_se(per_problem([x[0] - x[1] for x in rows.tolist()], recs))
    n = st.mean(x[4] for x in a)
    return {"nll": st.mean(x[0] for x in a) / n, "entropy": st.mean(x[1] for x in a) / n, "diff": m_se[0] / n, "se": m_se[1] / n,  # type: ignore[operator]
            "tokens": sum(x[4] for x in a), "below_top": sum(x[2] for x in a), "expected_below_top": sum(x[3] for x in a)}


def check_sampler(job: Job, s: Sheet, P: D, recs: q.Records) -> None:
    f, mult = fit(job, recs, P["top_k"]), P["se_mult"]
    s.check("1 T 1: -log p(sampled) equals entropy", ok(f["diff"]) and ok(f["se"]) and not beyond(f["diff"], f["se"], mult), **f)
    power = f["expected_below_top"] >= P["min_expected"]
    s.check("1 T 1: no surviving top-k", not power or f["below_top"] > 0, power=power, below=f["below_top"], expected=f["expected_below_top"])
    cfg, pids, T = job.cfg, job.step_pids(1), P["control_temperature"]
    torch.manual_seed(q.step_seed(cfg.seed, 1))
    cold = job.graded(pids, q.rollouts(job.model, [job.prompt_ids(p) for p in pids], cfg.samples, cfg.window, job.rows,
                                       job.tok.pad_token_id, draw=lambda logits: q.sample_t1(logits / T)), cfg.samples)
    c = fit(job, cold, P["top_k"])
    s.check("1 control T 0.7: -log p below entropy", c["diff"] < 0 and beyond(c["diff"], c["se"], mult), **c)
    s.read("2 reward: T 1 (a) against plain T 0.7 (b), same prompts", **q.agreement(recs, cold, lambda r: r["reward"], mult))


def check_samples(job: Job, s: Sheet, P: D, recs: q.Records) -> None:
    tok, n, eos = job.tok, P["samples_per_class"], torch.tensor(job.model.generation_config.eos_token_id).flatten().tolist()
    by_k = sorted(recs, key=lambda r: r["k"])  # the first sample of each problem first: as many problems as can be
    cut = [r for r in by_k if not r["reward"] and r["truncated"]][: P["truncated_max"]]
    shown = [r for r in by_k if r["reward"]][:n] + cut + [r for r in by_k if not r["reward"] and not r["truncated"]][: n - len(cut)]
    text = [f"=== PROMPT of problem {p}, with special tokens ===\n{tok.decode(job.prompt_ids(p))}\n" for p in job.step_pids(1)[:2]]
    for r in shown:
        item, gold = job.sets["train"][r["pid"]], gold_number(job.sets["train"][r["pid"]]["gold"])
        pred = grade(tok.decode(r["ids"], skip_special_tokens=True), gold)["strict_pred"]
        text.append(f"=== problem {r['pid']} sample {r['k']}: reward {r['reward']}, strict {r['strict']}, gold {gold}, prediction {pred!r}, "
                    f"length {len(r['ids'])}, truncated {r['truncated']} ===\nQ: {item['question']}\n--- response ---\n{tok.decode(r['ids'])}\n")
    write_text(SAMPLES, "\n".join(text))
    by: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for r in recs:
        by[r["pid"]].append(tuple(r["ids"]))
    dups = [len(v) - len(set(v)) for v in by.values()]
    s.read("2 samples", shown=len(shown), ended=st.mean(r["ids"][-1] in eos for r in recs), think=st.mean("<think>" in tok.decode(r["ids"]) for r in recs),
           truncated=st.mean(r["truncated"] for r in recs), reward=st.mean(r["reward"] for r in recs),
           mean_length=st.mean(len(r["ids"]) for r in recs), duplicates_per_problem=st.mean(dups), most_duplicates=max(dups))
    s.check("2 ends in an end token exactly when not truncated", all((r["ids"][-1] in eos) != r["truncated"] for r in recs))


def arm(job: Job, s: Sheet, name: str, recs: q.Records, w: list[float], then: Any = None) -> D:
    """From W0 with a fresh optimiser: one real update; each sample's change in summed response log-prob on this batch;
    `then` while the updated weights stand. W0 is back on return."""
    with sft.trainable(job.model, job.W0) as params:
        opt = sft.adamw(params, job.cfg.lr)
        rec = update(job, params, opt, recs, w, job.cfg.lr)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        changes, rec["kl_on_batch"] = q.read_h(job.model, job.W0, job.seqs(recs), job.cfg.read_tokens)
        if then:
            then(params, rec)
    out = {"gap": q.delta_r(changes, recs), "se": mean_se(gaps(changes, recs))[1], "mixed": len(q.pool(recs)),
           "sum_w_change": sum(x * c for x, c in zip(w, changes, strict=True)), "mean_change": st.mean(changes), **rec}
    s.read(f"3 {name}", **out)
    return out | {"changes": changes}


def check_arms(job: Job, s: Sheet, P: D, recs: q.Records, after_r: Any) -> None:
    w, mult = {a: q.arm_weights(a, recs, job.cfg.seed, 1) for a in ("R", "N", "F")}, P["se_mult"]
    r = s.guard("3 R", arm, job, s, "R", recs, w["R"], after_r) or {}
    gap_r, sum_r = r.get("gap"), r.get("sum_w_change")
    s.check("3 R: right minus wrong > 0, over 5 SE; sum w change > 0", positive(gap_r) and beyond(gap_r, r.get("se"), mult) and positive(sum_r),
            gap=gap_r, se=r.get("se"), sum_w_change=sum_r)
    n = s.guard("3 N", arm, job, s, "N", recs, w["N"]) or {}
    s.check("3 N: sum w change > 0; |gap| under R's", positive(n.get("sum_w_change")) and ok(n.get("gap")) and ok(gap_r)
            and abs(n["gap"]) < abs(gap_r), gap=n.get("gap"), R_gap=gap_r, sum_w_change=n.get("sum_w_change"))  # type: ignore[arg-type]
    f = s.guard("3 F", arm, job, s, "F", recs, w["F"]) or {"changes": [0.0] * len(recs)}
    kept = [(c, x) for c, x, y in zip(f["changes"], recs, w["F"], strict=True) if y and "gap" in f]
    m, se = mean_se(per_problem([c for c, _ in kept], [x for _, x in kept]))
    s.check("3 F: kept samples' mean change > 0, over 5 SE", positive(m) and beyond(m, se, mult), mean=m, se=se, kept=len(kept))
    neg = s.guard("3 R negated", arm, job, s, "R negated", recs, [-x for x in w["R"]]) or {}
    s.check("3 R negated: right minus wrong < 0", ok(neg.get("gap")) and neg["gap"] < 0, gap=neg.get("gap"), R_gap=gap_r)


def read_held(job: Job, s: Sheet, recs: q.Records) -> None:
    changes, kl = q.read_h(job.model, job.W0, job.seqs(recs), job.cfg.read_tokens)
    m, se = mean_se(gaps(changes, recs))
    s.read("3 R on the held-out pool", gap=m, se=se, mixed=len(q.pool(recs)), kl=kl, early_check_band=job.pred["early"]["kl_first"])


def check_moved(job: Job, s: Sheet, params: D, rec: D, frozen: D, seen: dict[str, set[Any]]) -> None:
    named, lr = dict(job.model.named_parameters()), job.cfg.lr
    s.check("4 outside the 196: bit-identical", all(torch.equal(named[n], x) for n, x in frozen.items()), names=sorted(frozen)[:6],
            n_others=len(frozen), tied_head=job.model.lm_head.weight is job.model.model.embed_tokens.weight)
    s.check("4 outside the 196: no grad", not any(named[n].grad is not None or named[n].requires_grad for n in frozen))
    d = {x: (p.detach() - job.W0[x]).abs() for x, p in params.items()}
    unmoved = [x for x, v in d.items() if not v.any()]
    s.check("4 each of the 196 moved", len(params) == len(SITES) and not unmoved, matrices=len(params), unmoved=unmoved)
    s.read("4 entries", share_over_half_lr=sum(int((v > lr / 2).sum()) for v in d.values()) / job.numel, grad_rms=rec["grad_rms"],
           share_unmoved=sum(int((v == 0).sum()) for v in d.values()) / job.numel,
           dw_ratio=rec["dw_norm"] / (lr * math.sqrt(job.numel)), early_check_band=job.pred["early"]["dw_ratio"])
    want = {(False, str(F32), str(F32))}
    s.check("4 all 196 float32; forward: no autocast, fp32, TF32 off", all(p.dtype == F32 for p in params.values()) and seen["sampler"] == want
            and seen["train"] == want and not torch.backends.cuda.matmul.allow_tf32, **{k: sorted(map(str, v)) for k, v in seen.items()})


@torch.no_grad()
def check_checkpoint(job: Job, s: Sheet, P: D) -> None:
    """After all the updates, so that it also shows W0 was not written to."""
    files = Path(snapshot_download(HF_NAME, revision=HF_REVISION, local_files_only=True)).glob("*.safetensors")
    handles, bad, sites = [safe_open(str(f), framework="pt") for f in files], [], SITES[:: len(SITES) // P["blocks"]]
    for x in sites:
        key = f"model.{x}.weight"
        raw = next(h for h in handles if key in h.keys()).get_tensor(key)  # noqa: SIM118  (a safe_open handle is not a dict)
        if raw.dtype != torch.bfloat16 or not torch.equal(raw.to(DEV).float(), job.W0[x]):
            bad.append((x, str(raw.dtype)))
    s.check("4 W0 in fp32 equals the bf16 checkpoint, one matrix per block", len(sites) == P["blocks"] and not bad, matrices=len(sites), bad=bad)


def layout(path: Path) -> dict[str, tuple[Any, Any]]:
    with safe_open(str(path), framework="pt") as f:
        return {k: (f.get_slice(k).get_shape(), f.get_slice(k).get_dtype()) for k in f.keys()}  # noqa: SIM118


@torch.no_grad()
def check_saved(job: Job, s: Sheet, P: D, params: D, recs: q.Records) -> None:
    path = Path(tempfile.gettempdir()) / f"e69_sanity_dW_{time.time_ns()}.safetensors"  # the node's /ephemeral (cluster/env.sh)
    try:
        sft.save(path, q.delta_fp32(params, job.W0), F32, droppable=False, log=job.log)
        W = sft.plus(job.W0, sft.load(path))
        differ = [(int((W[x] != params[x]).sum()), float((W[x] - params[x]).abs().max())) for x in SITES]
        worst = max(d[1] for d in differ)
        s.check("5 W0 + saved dW equals the master weights", worst <= P["weight_abs_tol"], differing=sum(d[0] for d in differ), max_abs_diff=worst)
        seqs = job.seqs(recs)[: P["kl_rows"]]
        _, live = q.read_h(job.model, job.W0, seqs, job.cfg.read_tokens)
        with sft.set_weights(job.model, W):
            _, rebuilt = q.read_h(job.model, job.W0, seqs, job.cfg.read_tokens)
        s.check("5 KL from the saved dW equals the live KL", abs(live - rebuilt) <= P["kl_abs_tol"], live=live, rebuilt=rebuilt)
        mine, e40 = layout(path), layout(E40_DW)
        s.check("5 keys, shapes, dtypes equal E40's dW", mine == e40, keys=len(mine), e40_keys=len(e40), differ=[k for k in sorted({*mine, *e40}) if mine.get(k) != e40.get(k)][:4])
    finally:
        path.unlink(missing_ok=True)


def read_sample(job: Job, s: Sheet, P: D, recs: q.Records) -> None:
    """The same prompts at `qwen_eval.SAMPLE`, by the run's `generate` control; last, since only it may be left out."""
    if time.time() - job.t0 > P["skip_sample_after_s"]:
        s.read("2 reward at qwen_eval.SAMPLE", skipped="out of time")
        return
    cfg, pids = job.cfg, job.step_pids(1)
    torch.manual_seed(q.step_seed(cfg.seed, 1))
    rows = q.rows_for(torch.cuda.get_device_properties(0).total_memory, cfg.generate_reserved_gb * GB, cfg.generate_tokens_per_row, cfg.max_rows)
    b = job.graded(pids, q.generate_rollouts(job.model, [job.prompt_ids(p) for p in pids], cfg.samples, cfg.window, rows,
                                             job.tok.pad_token_id, **qwen_eval.SAMPLE), cfg.samples)
    s.read("2 reward at qwen_eval.SAMPLE: T 1 (a) against SAMPLE (b), same prompts", e45_pool=0.8335, **q.agreement(recs, b, lambda r: r["reward"], 3))


def main() -> None:
    if not FILED.is_file():
        sys.exit(f"{FILED} is missing: expected values are filed before this runs")
    P, s = json.loads(FILED.read_text()), Sheet()
    job = SanityJob(config("full", q.DEFAULT_TAG), "sanity")
    s.guard("6", check_data, job, s, P["c6"])
    model, cfg = job.fp32(), job.cfg
    s.read("device", card=torch.cuda.get_device_name(), rows=job.rows, lr=cfg.lr, problems=cfg.problems, samples=cfg.samples)
    sites = {f"model.{x}.weight" for x in SITES}
    frozen = {n: p.data.clone() for n, p in model.named_parameters() if n not in sites}
    seen: dict[str, set[Any]] = {"sampler": set(), "train": set()}
    phase = ["sampler"]
    sft.linears(model)[SITES[0]].register_forward_hook(
        lambda m, _, o: seen[phase[0]].add((torch.is_autocast_enabled("cuda"), str(o.dtype), str(m.weight.dtype))))
    recs = job.sampled(job.step_pids(1), cfg.samples, 1)
    s.guard("1", check_sampler, job, s, P["c1"], recs)
    s.guard("2", check_samples, job, s, P["c2"], recs)
    held = job.sampled(job.held[: P["c3"]["held_pool"]], cfg.samples, 0)
    phase[0] = "train"

    def after_r(params: D, rec: D) -> None:
        s.guard("3 held-out pool", read_held, job, s, held)
        s.guard("4", check_moved, job, s, params, rec, frozen, seen)
        s.guard("5", check_saved, job, s, P["c5"], params, recs)

    check_arms(job, s, P["c3"], recs, after_r)
    s.guard("4 checkpoint", check_checkpoint, job, s, P["c4"])
    s.guard("2 SAMPLE", read_sample, job, s, P["c2"], recs)
    bad = [k for k, v in s.out["checks"].items() if not v["holds"]]
    s.read("end", failed=bad, seconds=time.time() - job.t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / GB)
    print(f"=== sanity: {len(s.out['checks']) - len(bad)} of {len(s.out['checks'])} checks hold; failed: {bad} ===", flush=True)
    hard_exit(1 if bad else 0)


if __name__ == "__main__":
    main()
