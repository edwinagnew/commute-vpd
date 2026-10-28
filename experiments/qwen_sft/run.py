"""E40 (qwen_sft): the first Qwen3-1.7B update. SFT on GSM8K's human-written solutions, the do-nothing state and
teacher training at steps 1, 10, 100, each scored by the paper's adversary. Plan: reports/plans/2026-10-28_plan_qwen-sft.md;
rules and thresholds in results/predictions.json (the run refuses to start without it, and scores from it).

Stages, in this order, any subset by --stages (a later stage reads the earlier ones' results/<stage>.json and files on
--out). Each writes results/<stage>.json as it finishes, prints every number, and rescores results/score.json.
  G   maths rows (greedy answers of the unchanged model to the first GSM8K test prompts, packed into 128 rows of 512);
      on those and on experiments/qwen_adversary's 128 FineWeb-Edu rows: importance-masked KL, g statistics and the
      alive set (g > 0.01 anywhere), the adversary (seed 0). Stop on G fail. Then the benchmark check: one teacher-
      training step timed, the projection of T0-T; over $25 drops step 10 from D and T, still over stops; alive
      offsets over 45 GB stop.
  T0  teacher training with δW = 0, adversary on FineWeb rows; ratio outside the T0 band stops.
  A0  greedy accuracy on GSM8K test and GSM1k at W0 (bf16 copy, as the baseline), held-out gold log-loss; P0 stops.
  F   SFT 100 steps; step-0 readings first (P0's zero checks stop); δW at the saved steps, the W0 gradient, Adam
      moments at 100; gold loss drop and KL to start at the saved steps; accuracy and lengths at 100.
  D   do-nothing at the read steps: FineWeb adversary; at 100 also maths rows and g statistics.
  T   teacher training at the read steps: behaviour kept, FineWeb adversary; at 100 also maths rows, g statistics and
      the student's GSM8K accuracy.
FineWeb ratios inside the filed three-starts band are re-read with seeds 1 and 2 (W0 re-read at the same seeds once,
results/W0_refs.json); the mean of the per-seed ratios decides.

Memory (H100 80 GB, GB = 1e9 B): target fp32 6.9, components fp32 19.5, Γ bf16 17.1. G, D and the adversary part of
T0/T: those plus one state's matrices (5.6) and the adversary's working set (~15): ~64. Teacher training: Γ and the
components parked in host memory; target, δW bf16 (2.8), alive copies + offsets + gradients + Adam (20 B per alive
parameter) and micro-batches of 2 rows: ~17 + 20 B x alive. F: target, W0 copy, gradients, Adam (6.9 + 5.6 x 4) ~30 +
activations. Large files go to --out (default /workspace/e40); small per-state arrays to results/large/.

Usage (repo root, H100): python experiments/qwen_sft/run.py [--stages G,T0,A0,F,D,T] [--out /workspace/e40]
                          [--decomp /workspace/qwen3-1.7b-p-e9fcffd5-torch] [--usd-per-h 3.0]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import statistics
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_eval, qwen_sets
from commute.qwen_data import fineweb_rows
from commute.qwen_eval import MODEL, MODEL_REV, Records, load_model, prompt, summary
from commute.qwen_sft import (
    DISK_MARGIN,
    Seq,
    adam_state,
    answer_ids,
    gold_loss,
    linears,
    load,
    mean_kl,
    pack,
    plus,
    save,
    set_weights,
    sft_seq,
    sft_step,
)
from commute.qwen_teacher import (
    Offsets,
    adversary,
    do_nothing,
    g_stats,
    importance_pass,
    move_gamma,
    train,
)
from commute.qwen_vpd import (
    SITES,
    fp32_matmuls,
    load_components,
    load_gamma,
    load_target,
)

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, MATHS, REFS = RESULTS / "predictions.json", RESULTS / "maths_rows.json", RESULTS / "W0_refs.json"
WEB = HERE.parent / "qwen_adversary" / "results" / "tokens.json"
ADAM = "adam_step100.safetensors"
STAGES, DEV, T_START = ("G", "T0", "A0", "F", "D", "T"), "cuda", time.time()
F32, BF16 = torch.float32, torch.bfloat16


@dataclass(frozen=True)
class Config:
    seq_len: int = 512
    n_rows: int = 128  # rows per block (maths, FineWeb)
    maths_problems: int = 256  # GSM8K test prompts answered, index order (~230 fill the maths rows)
    gen_batch: int = 128
    gen_budget_s: float = 1800.0  # per set; P0 needs every problem answered
    alive_g: float = 0.01
    seed: int = 0  # the training-set permutation and teacher training's batch order
    heldout: int = 256
    batch: int = 64
    micro: int = 8
    steps: int = 100
    lr: float = 1e-5
    saved: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100)
    read_steps: tuple[int, ...] = (1, 10, 100)  # D and T; the budget rule may drop 10
    drop_order: tuple[str, ...] = ("adam", "dW_3", "dW_20")  # when the volume is short
    t_lr: float = 1e-5
    t_steps: int = 1000
    t_rows_each: int = 4  # maths rows and FineWeb rows per teacher-training batch
    t_micro: int = 2
    web_pool: int = 2048  # FineWeb rows for teacher training, none of them an evaluation row
    eval_batch: int = 8  # fp32 readings
    budget_usd: float = 25.0
    offsets_gb_max: float = 45.0

    def __post_init__(self) -> None:
        assert self.steps == max(self.saved) and set(self.read_steps) <= set(self.saved)
        assert 10 in self.read_steps and self.steps in self.read_steps
        assert self.heldout + self.steps * self.batch <= qwen_sets.N_TRAIN
        assert self.batch % self.micro == 0 and (2 * self.t_rows_each) % self.t_micro == 0
        assert all(d == "adam" or int(d.removeprefix("dW_")) in self.saved for d in self.drop_order)


CFG = Config()


def log(msg: str) -> None:
    line = f"[{time.time() - T_START:8.1f}s] {msg}"
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def brief(d: Any) -> Any:
    """A results dict for the log: the per-step lists (SFT steps, teacher-training curves) left out at any depth."""
    if not isinstance(d, dict):
        return d
    return {k: brief(v) for k, v in d.items() if k not in ("steps", "curve")}


def rows_of(rows: list[list[int]]) -> list[Tensor]:
    return [torch.tensor([r], device=DEV) for r in rows]


# ----------------------------------------------------------------------------- scoring (pure)


def r9(x: float) -> float:
    """Rounded to 9 places, so a tie in exact arithmetic lands on the side the rule writes as inclusive."""
    return round(x, 9)


def g_verdict(adv_ratio: float, masked_ratio: float, rule: dict[str, float]) -> str:
    if not (math.isfinite(adv_ratio) and math.isfinite(masked_ratio)):
        return "fail"
    if r9(adv_ratio) > rule["fail_above"] or r9(masked_ratio) > rule["fail_above"]:
        return "fail"
    return "pass" if max(r9(adv_ratio), r9(masked_ratio)) <= rule["pass_max"] else "ambiguous"


def in_band(x: float, band: list[float]) -> bool:
    return band[0] <= r9(x) <= band[1]  # False for NaN


def three_starts(read: Callable[[int], float], ref: Callable[[int], float], seeds: list[int],
                 band: list[float]) -> dict[str, Any]:
    """The first seed's ratio decides unless it lies in `band`; then every seed is read and the mean ratio decides."""
    adv = {seeds[0]: read(seeds[0])}
    ratios = {seeds[0]: adv[seeds[0]] / ref(seeds[0])}
    if in_band(ratios[seeds[0]], band):
        for s in seeds[1:]:
            adv[s] = read(s)
            ratios[s] = adv[s] / ref(s)
    return {"adv": {str(k): v for k, v in adv.items()}, "ratios": {str(k): v for k, v in ratios.items()},
            "ratio": statistics.mean(ratios.values())}


def kept(kl_ts: float, kl_t0: float, floor: float) -> float | None:
    """Behaviour kept, 1 − KL(teacher ‖ student) / KL(teacher ‖ W0); None (unscored) when the denominator < floor."""
    return None if kl_t0 < floor else 1 - kl_ts / kl_t0


def paired(a: Records, b: Records, key: str = "strict") -> dict[str, Any]:
    """Accuracy at both readings on the problems whose answers fit the window at both (the drop rule, paired)."""
    fa, fb = ({r["idx"]: bool(r[key]) for r in x if r["fits"]} for x in (a, b))
    both = sorted(fa.keys() & fb.keys())
    if not both:
        return {"n": 0, "first": None, "second": None, "diff": None}
    first, second = (statistics.mean(f[i] for i in both) for f in (fa, fb))
    return {"n": len(both), "first": first, "second": second, "diff": second - first}


def plan_drops(sizes: dict[str, int], free_bytes: int, order: tuple[str, ...]) -> list[str]:
    """The fewest items from the front of `order` whose removal makes the rest fit in `free_bytes`."""
    drops: list[str] = []
    for d in order:
        if sum(v for k, v in sizes.items() if k not in drops) <= free_bytes:
            break
        drops.append(d)
    return drops


def score(res: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    """Every filed rule whose inputs are in `res` (stage -> results); absent inputs leave the rule unscored."""
    R, sc = pred["rules"], {}
    G, T0, A0, Fs, D, T = (res.get(k, {}) for k in STAGES)
    if "adv" in G.get("fineweb", {}) and "adv" in G.get("maths", {}):
        a = G["maths"]["adv"] / G["fineweb"]["adv"]
        m = G["maths"]["masked_kl"] / G["fineweb"]["masked_kl"]
        sc["G"] = {"adv_ratio": a, "masked_ratio": m, "verdict": g_verdict(a, m, R["G"])}
    if "fineweb" in T0:
        sc["T0"] = {"ratio": T0["fineweb"]["ratio"], "holds": in_band(T0["fineweb"]["ratio"], R["T0"]["band"])}
    p0 = R["P0"]
    if "accuracy" in A0:
        acc = A0["accuracy"]
        full = acc["test"]["k"] == qwen_sets.N_TEST and acc["gsm1k"]["k"] == qwen_sets.N_GSM1K_KEPT
        sc["P0_accuracy"] = {"gsm8k": acc["test"]["strict_dropped"], "gsm1k": acc["gsm1k"]["strict_dropped"],
                             "holds": full and all(r9(abs(acc[n]["strict_dropped"] - p0[n])) <= p0["tol"]
                                                   for n in ("test", "gsm1k"))}
    if "step0" in Fs:
        z = Fs["step0"]
        sc["P0_zero"] = {**z, "holds": max(abs(z["gold_drop"]), abs(z["kl_to_start"])) <= p0["zero_max"]}
    last = str(CFG.steps)
    if last in Fs.get("saved", {}) and "accuracy" in Fs and "accuracy" in A0:
        r, p1 = Fs["saved"][last], R["P1"]
        length = Fs["accuracy"]["test"]["response_mean"] / A0["accuracy"]["test"]["response_mean"]
        parts = {"gold_drop": r9(r["gold_drop"]) >= p1["gold_drop_min"], "kl": r9(r["kl_to_start"]) >= p1["kl_min"],
                 "length": r9(length) <= p1["length_ratio_max"]}
        sc["P1"] = {**r, "length_ratio": length, "parts": parts, "holds": all(parts.values())}
    if "1" in D:
        sc["P2_step1"] = {"ratio": D["1"]["fineweb"]["ratio"],
                          "holds": r9(D["1"]["fineweb"]["ratio"]) <= R["P2"]["step1_max"]}
    if last in D:
        x = D[last]["fineweb"]["ratio"]
        sc["P2_step100"] = {"ratio": x, "reading": "sees the update" if r9(x) > R["P2"]["line"] else "blind on web text"}
    if T:
        p3 = R["P3"]
        k = {t: kept(r["kl_teacher_student"], r["kl_teacher_w0"], p3["denominator_min"]) for t, r in T.items()}
        scored = {t: r9(v) >= p3["kept_min"] for t, v in k.items() if v is not None and int(t) in p3["steps"]}
        sc["P3"] = {"kept": k, "scored": scored, "holds": all(scored.values()) if scored else None}
        p4 = {t: r9(r["fineweb"]["ratio"]) <= R["P4"]["line"] for t, r in T.items()}
        sc["P4"] = {"ratio": {t: r["fineweb"]["ratio"] for t, r in T.items()}, "passes": p4, "holds": all(p4.values())}
    if "maths" in T.get(last, {}) and "maths" in D.get(last, {}):
        a, b = T[last]["maths"]["adv"], D[last]["maths"]["adv"]
        sc["P5"] = {"teacher_training": a, "do_nothing": b, "holds": a < b}
    return sc


# ----------------------------------------------------------------------------- the job


class Job:
    """The fp32 target (its original matrices are W0), the rows, the GSM sets, and the decomposition, loaded when a
    stage needs it and parked in host memory while teacher training runs."""

    def __init__(self, a: argparse.Namespace) -> None:
        self.pred = json.loads(PRED.read_text())
        self.out, self.decomp_dir, self.usd_per_h = Path(a.out), Path(a.decomp), a.usd_per_h
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        self.end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.model = load_target(DEV, F32)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
        self.web_lists = json.loads(WEB.read_text())["rows"]
        self.web = rows_of(self.web_lists)
        self.sets = qwen_sets.load()
        perm = torch.randperm(qwen_sets.N_TRAIN, generator=torch.Generator().manual_seed(CFG.seed)).tolist()
        self.held_idx, self.train_idx = perm[: CFG.heldout], perm[CFG.heldout :]
        self.res = {n: json.loads(p.read_text()) for n in STAGES if (p := RESULTS / f"{n}.json").is_file()}
        self.comps: Any = None
        self.gamma: Any = None
        self._pools: tuple[Tensor, Tensor] | None = None
        log(f"[load] {torch.cuda.get_device_name()}, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def put(self, stage: str, d: dict[str, Any]) -> None:
        self.res[stage] = d
        (RESULTS / f"{stage}.json").write_text(json.dumps(d, indent=1))
        log(f"[{stage}] {json.dumps(brief(d), default=float)}")
        sc = score(self.res, self.pred)
        (RESULTS / "score.json").write_text(json.dumps(sc, indent=1))
        log(f"[score] {json.dumps(sc, default=float)}")

    def halt(self, why: str) -> None:
        (RESULTS / "stopped.json").write_text(json.dumps({"why": why, "at_s": time.time() - T_START}))
        log(f"STOP: {why}")
        sys.exit(f"STOP: {why}")

    def decomp(self, device: str = DEV) -> tuple[Any, Any]:
        if self.comps is None:
            self.comps = load_components(self.decomp_dir, DEV, F32)
            self.gamma = load_gamma(self.decomp_dir, DEV, BF16)
        if self.comps.V["q"].device.type != device:
            self.comps, self.gamma = self.comps.to(device, F32), move_gamma(self.gamma, device)
            free()
        return self.comps, self.gamma

    def drop_decomp(self) -> None:
        self.comps = self.gamma = None
        free()

    def seq(self, i: int) -> Seq:
        it = self.sets["train"][i]
        return sft_seq(self.tok, it["question"], it["gold"], self.end_id, CFG.seq_len)

    def maths(self) -> list[Tensor]:
        return rows_of(json.loads(MATHS.read_text())["rows"])

    def alive(self) -> dict[str, Tensor]:
        return load(self.out / "alive.safetensors")

    def read_steps(self) -> tuple[int, ...]:
        return tuple(self.res["G"]["read_steps"])

    def dw_path(self, t: int) -> Path:
        return self.out / "dW" / f"step_{t}.safetensors"

    def pools(self) -> tuple[Tensor, Tensor]:
        """Teacher training's rows: F's training sequences packed into 512-token rows, and FineWeb-Edu rows that are
        not evaluation rows (streamed once, cached on --out)."""
        if self._pools is None:
            n = CFG.steps * CFG.batch
            maths = torch.tensor(pack([self.seq(i)[0] for i in self.train_idx[:n]], None, CFG.seq_len))
            path = self.out / "fineweb_pool.pt"
            if not path.is_file():
                taken = {tuple(r) for r in self.web_lists}
                rows = [r for r in fineweb_rows(self.tok, CFG.n_rows + CFG.web_pool) if tuple(r) not in taken]
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.tensor(rows, dtype=torch.int32), path)
            self._pools = maths, torch.load(path).long()
            log(f"[pools] {len(maths)} maths rows, {len(self._pools[1])} FineWeb rows")
        return self._pools

    def adv(self, state: Callable[[], Any], rows: list[Tensor], seed: int, name: str) -> float:
        comps, gamma = self.decomp()
        t0 = time.time()
        with state():
            kl, src = adversary(self.model, rows, gamma, comps, seed, log=log)
        save(LARGE / "sources" / f"{name}_seed{seed}.safetensors", src, F32, droppable=True, log=log)
        log(f"[{name}] adversary seed {seed}: {kl:.4f} nats ({time.time() - t0:.0f} s)")
        return kl

    def w0_ref(self, seed: int) -> float:
        if seed == self.pred["three_starts"]["seeds"][0]:
            return self.res["G"]["fineweb"]["adv"]
        refs = json.loads(REFS.read_text()) if REFS.is_file() else {}
        if str(seed) not in refs:
            refs[str(seed)] = self.adv(nullcontext, self.web, seed, "W0_fineweb")
            REFS.write_text(json.dumps(refs, indent=1))
        return refs[str(seed)]

    def fineweb(self, state: Callable[[], Any], name: str) -> dict[str, Any]:
        ts = self.pred["three_starts"]
        return three_starts(lambda s: self.adv(state, self.web, s, f"{name}_fineweb"), self.w0_ref, ts["seeds"],
                            ts["band"])

    def maths_adv(self, state: Callable[[], Any], name: str) -> dict[str, float]:
        kl = self.adv(state, self.maths(), self.pred["three_starts"]["seeds"][0], f"{name}_maths")
        return {"adv": kl, "ratio": kl / self.res["G"]["maths"]["adv"]}

    def gstats(self, state: Callable[[], Any], name: str) -> None:
        comps, gamma = self.decomp()
        with state():
            for block, rows in (("maths", self.maths()), ("fineweb", self.web)):
                p = importance_pass(self.model, rows, gamma, comps, CFG.alive_g, masked=False)
                save(LARGE / "gstats" / f"{name}_{block}.safetensors", g_stats(p), torch.float16, True, log)

    def accuracy(self, W: dict[str, Tensor] | None, label: str, names: tuple[str, ...] = ("test", "gsm1k")) -> dict:
        """Greedy strict accuracy (`qwen_eval.score`) of a bf16 model holding W (None: W0), responses to LARGE."""
        tok, gen = load_model()
        if W is not None:
            with torch.no_grad():
                for s, lin in linears(gen).items():
                    lin.weight.copy_(W[s])
        out = {}
        for n in names:
            recs = qwen_eval.score(tok, gen, self.sets[n], CFG.gen_budget_s, LARGE / f"{label}_{n}.jsonl", f"{label} {n}",
                         CFG.gen_batch)
            out[n] = summary(recs) | {"response_mean": statistics.mean(r["response_tokens"] for r in recs)}
        del gen
        free()
        return out


def records(path: Path) -> Records:
    return [json.loads(line) for line in path.open()]


# ----------------------------------------------------------------------------- stages


def stage_G(job: Job) -> None:
    tok, gen = load_model()
    texts = [prompt(tok, it["question"]) for it in job.sets["test"][: CFG.maths_problems]]
    t0 = time.time()
    seqs = answer_ids(tok, gen, texts, CFG.gen_batch)
    out: dict[str, Any] = {"eval_batch_s": (time.time() - t0) / math.ceil(len(texts) / CFG.gen_batch)}
    del gen
    free()
    cum = torch.tensor([len(s) for s in seqs]).cumsum(0)
    out["problems_in_rows"] = int((cum < CFG.n_rows * CFG.seq_len).sum()) + 1
    MATHS.write_text(json.dumps({"problems": CFG.maths_problems, "rows": pack(seqs, CFG.n_rows, CFG.seq_len)}))
    comps, gamma = job.decomp()
    blocks = (("maths", job.maths()), ("fineweb", job.web))
    alive: dict[str, Tensor] = {}
    for block, rows in blocks:
        p = importance_pass(job.model, rows, gamma, comps, CFG.alive_g, masked=True)
        save(LARGE / "gstats" / f"W0_{block}.safetensors", g_stats(p), torch.float16, True, log)
        alive = {s: alive.get(s, p["any"][s]) | p["any"][s] for s in SITES}
        out[block] = {"masked_kl": p["masked_kl"]}
        log(f"[G] {block}: masked KL {p['masked_kl']:.4f}")
    idx = {s: alive[s].nonzero().flatten() for s in SITES}
    save(job.out / "alive.safetensors", idx, torch.int64, droppable=False, log=log)
    n = sum(len(idx[s]) * (comps.vu(s)[0].shape[0] + comps.vu(s)[1].shape[1]) for s in SITES)
    out["alive"] = {"subcomponents": sum(len(v) for v in idx.values()), "of": sum(len(v) for v in alive.values()),
                    "n_params": n, "offsets_gb": 20 * n / 1e9}
    for block, rows in blocks:
        t0 = time.time()
        out[block]["adv"] = job.adv(nullcontext, rows, job.pred["three_starts"]["seeds"][0], f"G_{block}")
        out[block]["adv_s"] = time.time() - t0
    out["read_steps"] = list(CFG.read_steps)
    job.put("G", out)
    if score(job.res, job.pred)["G"]["verdict"] == "fail":
        job.halt("G fails: the decomposition does not explain the maths rows as it explains web text (card)")
    if out["alive"]["offsets_gb"] > CFG.offsets_gb_max:
        job.halt(f"alive offsets need {out['alive']['offsets_gb']:.1f} GB (20 B x {n} parameters)")
    out["benchmark"] = benchmark(job, out)
    out["read_steps"] = out["benchmark"]["read_steps"]
    job.put("G", out)
    if not out["read_steps"]:
        job.halt(f"projected spend over ${CFG.budget_usd} even without step 10: {out['benchmark']}")


def benchmark(job: Job, g: dict[str, Any]) -> dict[str, Any]:
    """One teacher-training step timed after a first (warm-up) one; the projection of T0-T from it, G's adversary
    starts and G's generation batches; the budget rule (read_steps empty: over budget even without step 10)."""
    comps, _ = job.decomp()
    off = Offsets(comps, job.alive())
    job.decomp("cpu")
    maths, web = job.pools()
    torch.cuda.reset_peak_memory_stats()
    curve = train(job.model, off, None, maths, web, CFG.t_lr, 2, CFG.t_rows_each, CFG.t_micro, CFG.seed, log)
    b: dict[str, Any] = {"teacher_step_s": curve[-1]["s"], "teacher_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    del off
    free()
    adv, gen, step = statistics.mean(g[k]["adv_s"] for k in ("maths", "fineweb")), g["eval_batch_s"], curve[-1]["s"]
    gen_all = (math.ceil(qwen_sets.N_TEST / CFG.gen_batch) + math.ceil(qwen_sets.N_GSM1K_KEPT / CFG.gen_batch)) * gen
    sft_step_s = step * maths.numel() / CFG.steps / (2 * CFG.t_rows_each * CFG.seq_len)  # rough: tokens per step

    def spend(steps: tuple[int, ...]) -> tuple[dict[str, float], float]:
        k, train_s = len(steps), CFG.t_steps * step
        p = {"T0": train_s + adv, "A0": gen_all, "F": CFG.steps * sft_step_s + gen_all, "D": (k + 1) * adv,
             "T": k * (train_s + adv) + adv + math.ceil(qwen_sets.N_TEST / CFG.gen_batch) * gen}
        return p, (time.time() - T_START + sum(p.values())) / 3600 * job.usd_per_h

    steps = CFG.read_steps
    b["projected_s"], b["projected_usd"] = spend(steps)
    if b["projected_usd"] > CFG.budget_usd:
        steps = tuple(t for t in steps if t != 10)
        b["projected_s_without_10"], b["projected_usd_without_10"] = spend(steps)
        if b["projected_usd_without_10"] > CFG.budget_usd:
            steps = ()
    return b | {"read_steps": list(steps)}


def stage_T0(job: Job) -> None:
    comps, _ = job.decomp()
    off = Offsets(comps, job.alive())
    job.decomp("cpu")
    curve = train(job.model, off, None, *job.pools(), CFG.t_lr, CFG.t_steps, CFG.t_rows_each, CFG.t_micro, CFG.seed, log)
    off.save(job.out / "offsets" / "T0.safetensors", log)
    out: dict[str, Any] = {"max_abs_offset": off.max_abs(), "curve": [c["kl"] for c in curve]}
    off.to("cpu")
    out["fineweb"] = job.fineweb(lambda: off.state(job.model, job.decomp()[0], job.W0), "T0")
    job.put("T0", out)
    if not score(job.res, job.pred)["T0"]["holds"]:
        job.halt(f"T0: ratio {out['fineweb']['ratio']:.3f} outside {job.pred['rules']['T0']['band']}")


def held(job: Job) -> list[Seq]:
    return [job.seq(i) for i in job.held_idx]


def stage_A0(job: Job) -> None:
    job.drop_decomp()
    out = {"accuracy": job.accuracy(None, "step0"), "gold_loss": gold_loss(job.model, held(job), CFG.eval_batch, DEV)}
    job.put("A0", out)
    if not score(job.res, job.pred)["P0_accuracy"]["holds"]:
        job.halt("P0: step-0 accuracy is not the baseline's")


def disk_sizes(job: Job) -> dict[str, int]:
    numel = sum(w.numel() for w in job.W0.values())
    sizes = {f"dW_{t}": 2 * numel for t in CFG.saved} | {"grad": 2 * numel, "adam": 8 * numel}
    return sizes | {f"offsets_T{t}": 2 * job.res["G"]["alive"]["n_params"] for t in CFG.read_steps}


def stage_F(job: Job) -> None:
    job.drop_decomp()
    job.out.mkdir(parents=True, exist_ok=True)
    rewritten = [job.dw_path(t) for t in CFG.saved] + [job.out / "grad_W0.safetensors", job.out / ADAM]
    room = shutil.disk_usage(job.out).free - DISK_MARGIN + sum(p.stat().st_size for p in rewritten if p.is_file())
    drops = plan_drops(disk_sizes(job), room, CFG.drop_order)
    out: dict[str, Any] = {"dropped": drops, "saved": {}, "steps": []}
    log(f"[F] {room / 1e9:.1f} GB for {sum(disk_sizes(job).values()) / 1e9:.1f} GB of files; dropping {drops}")
    W = {s: w.clone() for s, w in job.W0.items()}  # the trained matrices
    maths, hs = job.maths(), held(job)

    def reading() -> dict[str, float]:
        return {"gold_drop": job.res["A0"]["gold_loss"] - gold_loss(job.model, hs, CFG.eval_batch, DEV),
                "kl_to_start": mean_kl(job.model, maths, job.W0, None)}

    with set_weights(job.model, W):
        params = {s: lin.weight.requires_grad_(True) for s, lin in linears(job.model).items()}
        opt = torch.optim.AdamW(list(params.values()), lr=CFG.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
        out["step0"] = reading()
        job.put("F", out)
        if not score(job.res, job.pred)["P0_zero"]["holds"]:
            job.halt("P0: the step-0 readings are not zero")
        for t in range(1, CFG.steps + 1):
            idx = job.train_idx[(t - 1) * CFG.batch : t * CFG.batch]
            opt.zero_grad(set_to_none=True)
            job.model.train()
            loss = sft_step(job.model, [job.seq(i) for i in idx], CFG.micro, DEV)
            job.model.eval()
            if t == 1:
                save(job.out / "grad_W0.safetensors", {s: p.grad for s, p in params.items() if p.grad is not None},
                     BF16, droppable=False, log=log)
            opt.step()
            with torch.no_grad():
                norms = {s: float((W[s] - job.W0[s]).norm()) for s in SITES}
            out["steps"].append({"step": t, "loss": loss, "norms": norms, "batch": idx})
            log(f"[F] step {t}: loss {loss:.4f}, |dW| {math.sqrt(sum(v * v for v in norms.values())):.4f}")
            if t in CFG.saved:
                name = f"dW_{t}"
                if name not in drops:
                    save(job.dw_path(t), {s: W[s] - job.W0[s] for s in SITES}, BF16, name in CFG.drop_order, log)
                out["saved"][str(t)] = reading()
                job.put("F", out)
        if "adam" not in drops:
            moments, n = adam_state(opt, params)
            save(job.out / ADAM, moments, F32, True, log, meta={"step": str(n)})
            del moments
        del opt
        for p in params.values():
            p.requires_grad_(False).grad = None
        free()
        out["accuracy"] = job.accuracy(W, f"step{CFG.steps}")
    out["paired"] = {n: paired(records(LARGE / f"step0_{n}.jsonl"), records(LARGE / f"step{CFG.steps}_{n}.jsonl"))
                     for n in ("test", "gsm1k")}
    job.put("F", out)


def stage_D(job: Job) -> None:
    res = job.res.get("D", {})
    for t in job.read_steps():
        if str(t) in res:
            continue
        dW = load(job.dw_path(t))

        def state(dW: dict[str, Tensor] = dW) -> Any:
            return do_nothing(job.model, job.W0, dW)

        rec: dict[str, Any] = {"fineweb": job.fineweb(state, f"D{t}")}
        if t == CFG.steps:
            rec["maths"] = job.maths_adv(state, f"D{t}")
            job.gstats(state, f"D{t}")
        res[str(t)] = rec
        job.put("D", res)


def stage_T(job: Job) -> None:
    res = job.res.get("T", {})
    for t in job.read_steps():
        if str(t) in res:
            continue
        off = Offsets(job.decomp()[0], job.alive())
        dW = load(job.dw_path(t), DEV)
        path, rec = job.out / "offsets" / f"T{t}.safetensors", {}
        if path.is_file():
            off.load_from(path)
            rec["resumed_from"] = str(path)
        else:
            job.decomp("cpu")
            curve = train(job.model, off, dW, *job.pools(), CFG.t_lr, CFG.t_steps, CFG.t_rows_each, CFG.t_micro,
                          CFG.seed, log)
            off.save(path, log)
            rec["curve"] = [c["kl"] for c in curve]
        off.to("cpu")
        with torch.no_grad():
            Wt, Ws = plus(job.W0, dW), {s: job.W0[s] + off.change(s, DEV) for s in SITES}
        del dW
        rec["kl_teacher_student"] = mean_kl(job.model, job.maths(), Wt, Ws)
        rec["kl_teacher_w0"] = mean_kl(job.model, job.maths(), Wt, None)
        rec["max_abs_offset"] = off.max_abs()
        del Wt
        if t == CFG.steps:
            rec["accuracy"] = job.accuracy(Ws, f"T{t}", ("test",))
        del Ws
        free()

        def state(off: Offsets = off) -> Any:
            return off.state(job.model, job.decomp()[0], job.W0)

        rec["fineweb"] = job.fineweb(state, f"T{t}")
        if t == CFG.steps:
            rec["maths"] = job.maths_adv(state, f"T{t}")
            job.gstats(state, f"T{t}")
        res[str(t)] = rec
        job.put("T", res)
        del off
        free()


RUN = {"G": stage_G, "T0": stage_T0, "A0": stage_A0, "F": stage_F, "D": stage_D, "T": stage_T}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--out", default="/workspace/e40")
    ap.add_argument("--decomp", default="/workspace/qwen3-1.7b-p-e9fcffd5-torch")
    ap.add_argument("--usd-per-h", type=float, default=3.0)
    a = ap.parse_args()
    stages = a.stages.split(",")
    assert stages == [s for s in STAGES if s in stages], f"stages run in the order {STAGES}: {stages}"
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    assert torch.cuda.get_device_properties(0).total_memory > 70e9, "needs an 80 GB card (H100)"
    LARGE.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stopped.json").unlink(missing_ok=True)
    job = Job(a)
    log(f"[config] {json.dumps(asdict(CFG))}; stages {stages}; out {job.out}")
    for s in stages:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        RUN[s](job)
        log(f"[{s}] done in {time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("FINISHED", RESULTS, flush=True)


if __name__ == "__main__":
    main()
