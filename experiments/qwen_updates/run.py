"""E45 (qwen_updates), Job 1: five SFT updates of Qwen3-1.7B (own samples, gold, the self-distillation null, anchor 1
context distillation, anchor 2 invented facts), each saved where its KL to start crosses a fixed grid. Plan:
reports/plans/2026-10-28_plan_qwen-q2.md (sections 3-6); rules and first-minute values in results/predictions.json
(the run refuses to start without it). Job 2 (E44's table on the saved δW) is not here. Run order and commands:
RUN_BRIEF.md beside this file.

Stages (one MATS-cluster debug job each; P, S and E stop before CFG.budget_s with exit 3 and resume when resubmitted;
a missed first-minute check stops with exit 2):
  control  one gold step at constant 1e-5 on E40's first batch, KL to start on the 128 maths rows (E40a1's L control);
           gold's mean sequence length; anchor 2's facts at W0 (both draws); W0's greedy answers to the GSM8K test
  P        anchor 1's prompt: the few-shot answers, the three candidates against no prompt on 512 train problems, the
           format fallback only if none qualifies; the chosen prompt's greedy answers to test 256-511 (anchor 1's rows)
  S        sampling (--set own: 4 unprompted samples per problem; --set a1: 8 prompted), one file per batch
  T        training one run (--arm, --seed): 100 steps, δW saved at the grid's brackets and step 100
  E        reading one run: KL to start on all own and maths rows and the task reading at every saved state, the state
           chosen per grid value, then accuracy, lengths and anchor scores at the chosen states and step 100
Results: results/<stage>[_<run>].json, results/score.json, results/run.log; large files under results/large/.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from typing import Any

import torch
from job import (
    ARMS,
    BASELINE,
    BF16,
    CFG,
    DEV,
    LARGE,
    SEEDS,
    Job,
    accuracy,
    facts_eval,
    greedy,
    per_batch,
    records,
)

from commute import qwen_facts
from commute.gsm8k import gold_number, grade
from commute.qwen_anchors import distill_kl, follows_format
from commute.qwen_eval import SAMPLE, Records, prompt, sample, summary
from commute.qwen_sft import (
    adamw,
    gold_loss,
    load,
    mean_kl,
    plus,
    save,
    set_weights,
    sft_step,
    trainable,
)
from commute.qwen_train import Recipe, train
from commute.qwen_updates import (
    anchor2_holds,
    at_least,
    at_most,
    choose,
    keep_correct,
    learned,
    length_ratio,
    paired_gain,
    pick,
    prompt_rule,
    r9,
    read_steps,
)

N_TRAIN = CFG.n_problems - CFG.n_held


# ----------------------------------------------------------------------------- control


def stage_control(job: Job) -> None:
    E, out = job.pred["early"]["control"], job.get("control")
    m, maths = job.fp32(), job.maths()
    with trainable(m, job.W0) as params:
        opt = adamw(params, CFG.control_lr)
        opt.zero_grad(set_to_none=True)
        m.train()
        loss = sft_step(m, [job.gold_seq(i) for i in job.train_idx[: CFG.batch]], CFG.control_micro, DEV)
        m.eval()
        opt.step()
        del opt
        kl = mean_kl(m, maths, job.W0, None)
    out["one_step"] = {"kl_to_start": kl, "loss": loss}
    job.check(out, "control: KL of one gold step at 1e-5", r9(abs(kl / E["kl_ref"] - 1)) <= E["kl_tol"],
              {"kl": kl, "ref": E["kl_ref"], "loss": loss}, "control")
    out["gold_mean_tokens"] = statistics.mean(len(job.gold_seq(i)[0]) for i in job.train_idx)
    job.put("control", out)
    for seed in (0, 1):
        out[f"facts_s{seed}"] = f = facts_eval(job, job.gen(), qwen_facts.held_set(seed))
        job.check(out, f"control: invented facts at W0, draw {seed}", at_most(f["invented"], E["invented_max"]), f,
                  "control")
    job.put("control", out)
    acc, recs = accuracy(job, None, job.sets["test"], LARGE / "W0_test.jsonl", "W0 test")
    first = summary(recs[: CFG.eval_problems])["strict_dropped"]
    base = {r["idx"]: r["response"] for r in records(BASELINE)}
    same = statistics.mean(base.get(r["idx"]) == r["response"] for r in recs) if base else None
    out["W0"] = {"test": acc, "first256_strict_dropped": first, "same_as_baseline": same}
    job.check(out, "control: W0 accuracy on test 0-255", at_most(abs(first - E["acc_ref"]), E["acc_tol"]),
              {"acc": first, "ref": E["acc_ref"], "same_as_baseline": same}, "control")
    job.put("control", out)


# ----------------------------------------------------------------------------- P: anchor 1's prompt


def run_candidate(job: Job, P: dict[str, Any], name: str, items: Records) -> None:
    if name in P["runs"]:
        return
    job.budget(300)
    t0 = time.time()
    shots = [tuple(s) for s in P["shots"]]
    texts = [job.a1_text(it["question"], name, shots) for it in items]
    recs = greedy(job, job.gen(), texts, items)
    (LARGE / "P").mkdir(exist_ok=True)
    (LARGE / "P" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    P["runs"][name] = {"strict": [r["strict"] for r in recs], "accuracy": statistics.mean(r["strict"] for r in recs),
                       "format_follow": statistics.mean(follows_format(r["response"]) for r in recs)}
    job.put("P", P)
    job.timed(t0)
    job.log(f"[P] {name}: strict {P['runs'][name]['accuracy']:.3f} ({time.time() - t0:.0f} s, {job.peaks()})")


def stage_P(job: Job) -> None:
    R, P = job.pred["rules"]["prompt"], job.get("P") or {"runs": {}}
    if "shots" not in P:
        items = [job.sets["train"][i] for i in job.shot_pool]
        recs = greedy(job, job.gen(), [prompt(job.tok, it["question"]) for it in items], items)
        P["shots"] = [(it["question"], r["response"]) for it, r in zip(items, recs, strict=True)
                      if r["strict"] and not r["truncated"]][: CFG.n_shots]
        assert len(P["shots"]) == CFG.n_shots, "too few correct few-shot answers"
    items = [job.sets["train"][i] for i in job.search]
    for name in ("none", "fewshot", "recheck", "givens"):
        run_candidate(job, P, name, items)
    base = P["runs"]["none"]["strict"]
    P["rules"] = {n: prompt_rule(base, P["runs"][n]["strict"], R["min_gain"], R["se_mult"])
                  for n in ("fewshot", "recheck", "givens")}
    P["chosen"] = pick(P["rules"])
    if P["chosen"] is None:
        run_candidate(job, P, "format", items)
        w, wo = P["runs"]["format"]["format_follow"], P["runs"]["none"]["format_follow"]
        P["format"] = {"with": w, "without": wo, "qualifies": at_least(w, R["format_with_min"])
                       and at_most(wo, R["format_without_max"])}
        P["chosen"] = "format" if P["format"]["qualifies"] else None
    job.put("P", P)
    job.log(f"[P] rules {json.dumps(P['rules'])}; chosen {P['chosen']}")
    if P["chosen"] is None:
        job.check(P, "P: a prompt qualifies", False, P.get("format", {}), "P")
    if not (LARGE / "P" / "a1_rows.json").is_file():
        tests = job.sets["test"][CFG.a1_rows_from : CFG.a1_rows_from + CFG.eval_problems]
        recs = greedy(job, job.gen(), [job.a1_text(it["question"]) for it in tests], tests)
        rows = [{"idx": r["idx"], "ids": r["ids"], "truncated": r["truncated"]} for r in recs]
        (LARGE / "P" / "a1_rows.json").write_text(json.dumps({"items": rows}))


# ----------------------------------------------------------------------------- S: sampling


def stage_S(job: Job, name: str) -> None:
    n, d = (CFG.own_k if name == "own" else CFG.a1_k), LARGE / "S" / name
    items = [job.sets["train"][i] for i in job.problems]
    texts = [prompt(job.tok, it["question"]) if name == "own" else job.a1_text(it["question"]) for it in items]
    per = per_batch(job, texts, n)
    d.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(texts), per))
    for b, s in enumerate(starts):
        path = d / f"batch_{b:04d}.jsonl"
        if path.is_file():
            continue
        job.budget(120)
        t0 = time.time()
        torch.manual_seed(CFG.pool_seed + b)
        outs = sample(job.tok, job.gen(), texts[s : s + per], n, SAMPLE)
        lines = []
        for j, (ids, cut) in enumerate(outs):
            it = items[s + j // n]
            ok = grade(job.tok.decode(ids, skip_special_tokens=True), gold_number(it["gold"]))["strict"]
            lines.append(json.dumps({"pid": s + j // n, "idx": it["idx"], "k": j % n, "ids": ids, "truncated": cut,
                                     "strict": bool(ok)}))
        path.with_suffix(".tmp").write_text("\n".join(lines) + "\n")
        path.with_suffix(".tmp").replace(path)
        job.timed(t0)
        job.log(f"[S {name}] batch {b + 1}/{len(starts)} ({per} prompts x {n}): {time.time() - t0:.0f} s; "
                f"projected {job.unit_s * len(starts):.0f} s in all; {job.peaks()}")
    recs = [json.loads(x) for p in sorted(d.glob("batch_*.jsonl")) for x in p.read_text().splitlines()]
    train = [r for r in recs if r["pid"] < N_TRAIN]
    job.put(f"S_{name}", {"complete": True, "n": len(recs), "prompts_per_batch": per,
                          "strict": statistics.mean(r["strict"] for r in recs),
                          "truncated": statistics.mean(r["truncated"] for r in recs),
                          "train_kept_correct": len(keep_correct(train)), "train_finished": sum(not r["truncated"]
                                                                                              for r in train)})


# ----------------------------------------------------------------------------- T: training


def early(job: Job, arm: str, out: dict[str, Any], rec: dict[str, Any], maths: list[Any]) -> None:
    """The first-minute checks (predictions.json "early"); each miss stops the job."""
    E, t, file = job.pred["early"], rec["step"], f"T_{out['run']}"
    if not (math.isfinite(rec["loss"]) and math.isfinite(rec["grad_norm"])):
        job.check(out, f"step {t} finite", False, {"loss": rec["loss"], "grad_norm": rec["grad_norm"]}, file)
    if t == 1:
        if arm == "gold":
            ref = E["gold_step1_loss"]
            job.check(out, "gold step-1 loss", r9(abs(rec["loss"] / ref["ref"] - 1)) <= ref["tol"],
                      {"loss": rec["loss"], **ref}, file)
        job.check(out, "step 1 at rate 0 leaves W0", rec["lr"] == 0 and rec["dw_norm"] == 0
                  and all(v == 0 for v in rec["kl"].values()), {"lr": rec["lr"], "dw_norm": rec["dw_norm"]}, file)
    if t == 2 or (arm == "gold" and str(t) in E["gold_curve"]["ref"]):
        kl = mean_kl(job.model, maths, job.W0, None)
        rec["kl_maths_all"] = kl
        if t == 2:
            ratio = [rec["norms"][s] / (rec["lr"] * math.sqrt(job.W0[s].numel())) for s in job.W0]
            lo, hi = E["first_live"]["band"]
            job.check(out, "step 2 |dW| / (rate sqrt numel)", lo <= r9(min(ratio)) and r9(max(ratio)) <= hi,
                      {"min": min(ratio), "max": max(ratio)}, file)
            job.check(out, "step 2 KL on the maths rows", at_most(kl, E["step2_kl_max"]), {"kl": kl}, file)
        if arm == "gold" and str(t) in E["gold_curve"]["ref"]:
            ref = E["gold_curve"]["ref"][str(t)]
            job.check(out, f"gold step {t} KL against E40a1", r9(abs(kl / ref - 1)) <= E["gold_curve"]["tol"],
                      {"kl": kl, "ref": ref}, file)


def stage_T(job: Job, arm: str, seed: int) -> None:
    run, m, maths = f"{arm}_s{seed}", job.fp32(), job.maths()
    stream, per, own, data = job.arm_data(arm, seed)
    out: dict[str, Any] = {"run": run, "data": data, "steps": [], "saves": [], "crossings": []}
    if arm in ("own", "own2"):
        job.check(out, "own samples kept", data["n"] >= job.pred["early"]["own_kept_min"], data, f"T_{run}")
    axes = {"maths": maths[: CFG.track_rows]} | ({} if own is None else {"own": job.rows(own[: CFG.track_rows])})
    r = Recipe(ARMS[arm], CFG.warmup, CFG.steps, CFG.schedule, CFG.clip, CFG.micro)
    out["recipe"] = asdict(r)

    def save_dw(t: int, dW: dict[str, Any]) -> None:
        save(LARGE / "dW" / run / f"step_{t}.safetensors", dW, BF16, droppable=False, log=job.log)
        out["saves"] = sorted(set(out["saves"]) | {t})

    def on_step(rec: dict[str, Any]) -> None:
        early(job, arm, out, rec, maths)
        out["crossings"] += rec.pop("crossed", [])
        rec.pop("norms")
        out["steps"].append(rec)
        job.log(f"[T {run}] step {rec['step']}: lr {rec['lr']:.2e}, loss {rec['loss']:.4f}, |dW| {rec['dw_norm']:.4f}"
                + (f", KL {json.dumps(rec['kl'])}" if "kl" in rec else ""))
        job.put(f"T_{run}", out)

    t0 = time.time()
    train(m, job.W0, lambda t: stream[(t - 1) * per : t * per], r, axes,
          set(read_steps(CFG.steps, CFG.dense_until, CFG.every)), CFG.grid, save_dw, on_step)
    out |= {"done": True, "seconds": time.time() - t0, "peaks": job.peaks()}
    job.put(f"T_{run}", out)


# ----------------------------------------------------------------------------- E: reading


def task_reader(job: Job, arm: str, seed: int) -> Any:
    """The arm's teacher-forced task reading at weights W (None: W0), fp32."""
    m = job.model
    if arm == "a1":
        held = [r for r in job.samples("a1") if r["pid"] >= N_TRAIN and r["k"] == 0 and not r["truncated"]]
        qs = [job.sets["train"][r["idx"]]["question"] for r in held]
        pairs = [(job.ids(job.a1_text(q)) + r["ids"], job.ids(prompt(job.tok, q)) + r["ids"], len(r["ids"]))
                 for q, r in zip(qs, held, strict=True)]
        return lambda W: {"distill_kl": distill_kl(m, pairs, job.W0, W)}
    if arm == "gold":
        seqs, key = [job.gold_seq(i) for i in job.held_idx], "loss"
    elif arm == "a2":
        seqs, key = [job.fact_seq(q) for q in qwen_facts.held_set(seed)], "loss"
    else:
        held = keep_correct([r for r in job.samples("own") if r["pid"] >= N_TRAIN])
        seqs, key = [job.sample_seq(r) for r in held], "loss"

    def read(W: Any) -> dict[str, float]:
        with set_weights(m, W):
            return {key: gold_loss(m, seqs, 8, DEV)}

    return read


def gen_reading(job: Job, arm: str, seed: int, t: int) -> dict[str, Any]:
    run = f"{arm}_s{seed}"
    W = plus(job.W0, load(LARGE / "dW" / run / f"step_{t}.safetensors", DEV))
    full = t == CFG.steps and arm in ("gold", "own", "own2", "null")
    items = job.sets["test"] if full else job.sets["test"][: CFG.eval_problems]
    acc, recs = accuracy(job, W, items, LARGE / "E" / run / f"step_{t}_test.jsonl", f"{run} {t}")
    res: dict[str, Any] = {"test": acc}
    if full:
        res["first256_strict_dropped"] = summary(recs[: CFG.eval_problems])["strict_dropped"]
    if arm == "a2":
        res["facts"] = facts_eval(job, job.gen(W), qwen_facts.held_set(seed))
    if arm == "a1":
        res["format_unprompted"] = statistics.mean(follows_format(r["response"]) for r in recs)
    return res


def stage_E(job: Job, arm: str, seed: int) -> None:
    run = f"{arm}_s{seed}"
    T, out = job.get(f"T_{run}"), job.get(f"E_{run}") or {"states": {}, "gen": {}}
    assert T.get("done"), f"T_{run} has not finished"
    m, maths, own = job.fp32(), job.maths(), job.own_rows(arm, seed)
    axes = {"maths": maths} | ({} if own is None else {"own": job.rows(own)})
    task = task_reader(job, arm, seed)
    for t in [0, *T["saves"]]:
        if str(t) in out["states"]:
            continue
        job.budget(90)
        t0 = time.time()
        W = None if t == 0 else plus(job.W0, load(LARGE / "dW" / run / f"step_{t}.safetensors", DEV))
        kl = {a: 0.0 if W is None else mean_kl(m, rows, job.W0, W) for a, rows in axes.items()}
        st = {"kl": kl | ({"own": kl["maths"]} if own is None else {}), "task": task(W)}
        base = out["states"].get("0", st)["task"]
        st["task"] |= ({"gap": 1 - st["task"]["distill_kl"] / base["distill_kl"]} if arm == "a1"
                       else {"drop": base["loss"] - st["task"]["loss"]})
        out["states"][str(t)] = st
        del W
        job.put(f"E_{run}", out)
        job.timed(t0)
        job.log(f"[E {run}] state {t}: {json.dumps(st)} ({time.time() - t0:.0f} s)")
    cross = T["crossings"] + ([{**c, "axis": "own"} for c in T["crossings"]] if own is None else [])
    out["chosen"] = choose(cross, {int(s): v["kl"] for s, v in out["states"].items()})
    job.put(f"E_{run}", out)
    for t in sorted({c["step"] for c in out["chosen"]} | {CFG.steps}):
        if str(t) not in out["gen"]:
            job.budget(450 if t == CFG.steps else 120)
            t0 = time.time()
            out["gen"][str(t)] = gen_reading(job, arm, seed, t)
            job.put(f"E_{run}", out)
            job.timed(t0)
            job.log(f"[E {run}] generation at {t}: {json.dumps(out['gen'][str(t)])}")
    k0 = job.get("control").get(f"facts_s{seed}", {}).get("known")
    for c in out["chosen"]:
        c["learned"] = learned(arm, out["states"][str(c["step"])]["task"], out["gen"][str(c["step"])].get("facts"), k0,
                               job.pred["rules"]["learned"])
    out["done"] = True
    job.put(f"E_{run}", out)


# ----------------------------------------------------------------------------- score


def score(job: Job) -> None:
    """The Job 1 predictions (plan section 5) from whatever has run; a missing input reads as None (not yet run)."""
    R, last = job.pred["rules"], str(CFG.steps)
    w0 = records(LARGE / "W0_test.jsonl")
    own, gold = (records(LARGE / "E" / f"{a}_s0" / f"step_{last}_test.jsonl") for a in ("own", "gold"))
    g = paired_gain(w0, own)
    out: dict[str, Any] = {"own_accuracy": g | {"holds": at_least(g["diff"], R["own_accuracy"]["min"])}}
    x = length_ratio(w0, own)
    out["own_length"] = {"ratio": x, "holds": at_least(x, R["own_length"]["min_ratio"])}
    x = length_ratio(w0, gold)
    out["gold_length"] = {"ratio": x, "holds": at_most(x, R["gold_length"]["max_ratio"])}
    for s in SEEDS["a1"]:
        gap = job.get(f"E_a1_s{s}").get("states", {}).get(last, {}).get("task", {}).get("gap")
        out[f"anchor1_learns_s{s}"] = {"gap": gap, "holds": at_least(gap, R["anchor1_learns"]["gap_min"])}
        f = job.get(f"E_a2_s{s}").get("gen", {}).get(last, {}).get("facts", {})
        k0 = job.get("control").get(f"facts_s{s}", {}).get("known")
        out[f"anchor2_learns_s{s}"] = f | {"known_w0": k0, "holds": anchor2_holds(f.get("invented"), f.get("known"),
                                                                                    k0, R["anchor2_learns"])}
    job.put("score", out)
    job.log(f"[score] {json.dumps(out)}")


def main() -> None:
    p = argparse.ArgumentParser(description="E45 (qwen_updates) Job 1; stages and order in RUN_BRIEF.md")
    p.add_argument("--stage", required=True, choices=("control", "P", "S", "T", "E", "score"))
    p.add_argument("--set", choices=("own", "a1"))
    p.add_argument("--arm", choices=tuple(ARMS))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.stage == "S":
        assert a.set, "--stage S needs --set"
    if a.stage in ("T", "E"):
        assert a.arm and a.seed in SEEDS[a.arm], f"--arm with a seed in {SEEDS}"
    job = Job(a.stage, a.set or (f"{a.arm}_s{a.seed}" if a.arm else ""))
    if a.stage == "control":
        stage_control(job)
    elif a.stage == "P":
        stage_P(job)
    elif a.stage == "S":
        stage_S(job, a.set)
    elif a.stage == "T":
        stage_T(job, a.arm, a.seed)
    elif a.stage == "E":
        stage_E(job, a.arm, a.seed)
    score(job)
    job.log(f"[{a.stage}] done; {job.peaks()}")


if __name__ == "__main__":
    main()
