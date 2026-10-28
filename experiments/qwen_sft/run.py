"""E40 (qwen_sft): the first Qwen3-1.7B update. SFT on GSM8K's human-written solutions, the do-nothing state and
teacher training at steps 1, 10, 100, each scored by the paper's adversary. Plan: reports/plans/2026-10-28_plan_qwen-sft.md;
rules and thresholds in results/predictions.json (the run refuses to start without it, and scores from it).
Amendment 1 (`amend1.py`) runs these stages with its own config, layout and predictions file.

Stages, in this order, any subset by --stages (a later stage reads the earlier ones' results/<stage>.json and files on
--out). Each writes results/<stage>.json as it finishes, prints every number, and rescores results/score.json.
  G   maths rows (greedy answers of the unchanged model to the first GSM8K test prompts, packed into 128 rows of 512);
      on those and on experiments/qwen_adversary's 128 FineWeb-Edu rows: importance-masked KL, g statistics and the
      alive set (g > 0.01 anywhere), the adversary (seed 0), FineWeb first: the control (G_control) stops unless its
      masked KL and adversary reproduce qwen_adversary's. Stop on G fail. Then the benchmark check: one teacher-
      training step timed, the projection of T0-T; over the budget drops step 10 from D and T, still over stops;
      alive offsets over the cap stop.
  T0  teacher training with δW = 0, adversary on FineWeb rows; ratio outside the T0 band stops.
  A0  greedy accuracy on GSM8K test and GSM1k at W0 (bf16 copy, as the baseline), held-out gold log-loss; P0 stops.
  F   SFT 100 steps at the config's rate schedule; step-0 readings first (P0's zero checks stop); δW at the saved
      steps, the W0 gradient, Adam moments at 100; gold loss drop and KL to start at the saved steps; accuracy and
      lengths at 100.
  D   do-nothing at the read steps: FineWeb adversary; at 100 also maths rows and g statistics.
  T   teacher training at the read steps: behaviour kept, FineWeb adversary; at 100 also maths rows, g statistics and
      the student's GSM8K accuracy.
FineWeb ratios inside the filed three-starts band are re-read with seeds 1 and 2 (W0 re-read at the same seeds once,
results/W0_refs.json); the mean of the per-seed ratios decides. A predictions file with an "early" section adds
first-minute checks that stop the job (`Job.check`): each adversary's step-0 KL, F's step-1 loss and KL, teacher
training's first batch and first 50 steps, and peak GPU memory against the config's per-stage estimate.

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
import re
import resource
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
    Uploader,
    adam_state,
    adamw,
    answer_ids,
    free_bytes,
    gold_loss,
    linears,
    load,
    lr_at,
    lr_schedule,
    mean_kl,
    meta_of,
    pack,
    plus,
    save,
    sft_seq,
    sft_step,
    sha256,
    trainable,
)
from commute.qwen_teacher import (
    Offsets,
    adversary,
    do_nothing,
    g_stats,
    importance_pass,
    kl_to_teacher,
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
PRED, MATHS = RESULTS / "predictions.json", RESULTS / "maths_rows.json"
WEB = HERE.parent / "qwen_adversary" / "results" / "tokens.json"
ADAM = "adam_step100.safetensors"
STAGES, DEV, T_START = ("G", "T0", "A0", "F", "D", "T"), "cuda", time.time()
F32, BF16 = torch.float32, torch.bfloat16
STEP0 = re.compile(r"seed (\d+) step 0: KL (\S+)")  # qwen_vpd.pgd_kl_rows's liveness line


@dataclass(frozen=True)
class Config:
    seq_len: int = 512
    n_rows: int = 128  # rows per block (maths, FineWeb)
    row_len: int = 512  # tokens kept per row of the blocks and the teacher-training pools (a rehearsal cuts them)
    maths_problems: int = 256  # GSM8K test prompts answered, index order (~230 fill the maths rows)
    gen_batch: int = 128
    gen_budget_s: float = 1800.0  # per set; P0 needs every problem answered
    eval_problems: int | None = None  # accuracy on the first this many problems of each set (None: all)
    alive_g: float = 0.01
    alive_max: int | None = None  # teacher training on the first this many alive columns per site (None: all)
    comps_dtype: str = "float32"
    adv_steps: int = 20
    seed: int = 0  # the training-set permutation and teacher training's batch order
    heldout: int = 256
    batch: int = 64
    micro: int = 8
    steps: int = 100
    lr: float = 1e-5
    warmup: int = 0  # SFT rate (qwen_sft.lr_at): "constant", or "hf_cosine" with this many warm-up steps
    schedule: str = "constant"
    clip: float | None = None  # gradient-norm clip of SFT (None: none; the norm is logged either way)
    save_adam: bool = True
    saved: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100)
    read_steps: tuple[int, ...] = (1, 10, 100)  # D and T (E40); the budget rule may drop 10
    read_fixed: tuple[int, ...] = ()  # Amendment 1: these read steps, plus the first saved step whose KL to start
    read_kl_min: float = 0.01  # reaches read_kl_min (`read_rule`, after F); () keeps read_steps
    drop_order: tuple[str, ...] = ("adam", "dW_3", "dW_20")  # when the volume is short
    t_lr: float = 1e-5
    t_steps: int = 1000
    t0_steps: int = 1000
    t0_shortcut: bool = False  # T0: offsets exactly 0 -> the state is W0's and G's FineWeb reading stands
    offsets_bf16_before_last: bool = False  # teacher-training offsets: bf16 at read steps before the last, else fp32
    t_rows_each: int = 4  # maths rows and FineWeb rows per teacher-training batch
    t_micro: int = 2
    web_pool: int = 2048  # FineWeb rows for teacher training, none of them an evaluation row
    eval_batch: int = 8  # fp32 readings
    budget_usd: float = 25.0  # E40's projection rule; Amendment 1 passes --budget-left instead
    offsets_gb_max: float = 45.0  # alive copies + offsets + gradients + Adam, 20 B per alive parameter
    min_card_gb: float = 70.0
    expected_misses: tuple[str, ...] = ()  # a rehearsal: the checks and stops expected to miss at its tiny sizes
    l_rates: tuple[float, ...] = (1e-7, 1e-6, 1e-5)  # Amendment 1's stage L
    l_micro: int = 4
    l_card_gb: float = 40.0
    pc_seed: int = 0  # Amendment 1's positive control: the random offsets' seed,
    pc_rows: int = 8  # teacher training's fixed rows (`Job.fixed_rows`: half maths, half FineWeb evaluation rows),
    pc_iters: int = 4  # and the size search's rounds on them (KL ∝ α²: α ← α·√(target / KL))
    a0_compare: str | None = None  # A0: an earlier run's step-0 GSM8K responses (under results/), first batch compared
    upload_max_bytes: int | None = None  # --wandb-upload skips larger files (a rehearsal's cap)

    def __post_init__(self) -> None:
        assert self.pc_rows % 2 == 0 and self.pc_rows // 2 <= self.n_rows
        assert self.steps == max(self.saved) and set(self.read_steps) <= set(self.saved)
        assert self.steps in self.read_steps and self.row_len <= self.seq_len
        assert set(self.read_fixed) <= set(self.saved) and (not self.read_fixed or self.steps in self.read_fixed)
        assert self.heldout + self.steps * self.batch <= qwen_sets.N_TRAIN
        assert self.batch % self.micro == 0 and self.batch % self.l_micro == 0
        assert (2 * self.t_rows_each) % self.t_micro == 0
        assert all(d == "adam" or int(d.removeprefix("dW_")) in self.saved for d in self.drop_order)
        assert self.schedule in ("constant", "hf_cosine") and 0 <= self.warmup < self.steps
        assert self.first_live() in self.saved, "F's checks read the first step with a nonzero rate"

    def first_live(self) -> int:
        """The first SFT step whose rate is not 0 (step 2 under the library's warm-up, which starts at 0)."""
        return next(t for t in range(1, self.steps + 1) if lr_at(t, self.lr, self.warmup, self.steps, self.schedule))


CFG = Config()


@dataclass(frozen=True)
class Layout:
    """Where one run's files go: <dir>/<names[stage]>.json per logical stage, the predictions it scores against,
    <dir>/<stem>.json for score, stopped, checks and the W0 references, <dir>/run.log, and the small per-state files
    (`large`). With `guard`, a stage whose results file exists is refused (`refuse_done`)."""

    dir: Path
    pred: Path
    names: dict[str, str]
    large: Path
    guard: bool = False

    def result(self, stage: str) -> Path:
        return self.dir / f"{self.names[stage]}.json"

    def file(self, stem: str) -> Path:
        return self.dir / f"{stem}.json"

    def log(self) -> Path:
        return self.dir / "run.log"


E40 = Layout(RESULTS, PRED, {s: s for s in STAGES}, LARGE)
_LOG = [E40.log()]


def log(msg: str) -> None:
    line = f"[{time.time() - T_START:8.1f}s] {msg}"
    print(line, flush=True)
    with _LOG[0].open("a") as f:
        f.write(line + "\n")


def write_json(path: Path, obj: Any) -> None:
    """Written whole under a temporary name, then renamed: a lost machine never leaves half a results file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def peaks() -> str:
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB, since the job started
    return f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, host {host:.1f} GB"


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


def three_starts(read: Callable[[int], float], ref: Callable[[int], float], seeds: list[int], band: list[float],
                 may_extend: Callable[[], bool]) -> dict[str, Any]:
    """The first seed's ratio decides unless it lies in `band`; then every seed is read and the mean ratio decides.
    `may_extend()` is asked before each extra start (the state's and W0's reference); False: the re-read stops, the
    first seed decides and the result says `reread_skipped_budget`."""
    adv = {seeds[0]: read(seeds[0])}
    ratios = {seeds[0]: adv[seeds[0]] / ref(seeds[0])}
    out: dict[str, Any] = {}
    if in_band(ratios[seeds[0]], band):
        for s in seeds[1:]:
            if not may_extend():
                out["reread_skipped_budget"] = True
                break
            adv[s] = read(s)
            if not may_extend():
                out["reread_skipped_budget"] = True
                break
            ratios[s] = adv[s] / ref(s)
    decide = [ratios[seeds[0]]] if out else list(ratios.values())
    return out | {"adv": {str(k): v for k, v in adv.items()}, "ratios": {str(k): v for k, v in ratios.items()},
                  "ratio": statistics.mean(decide)}


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


def read_rule(kl: dict[int, float], fixed: tuple[int, ...], kl_min: float, n_max: int) -> list[int]:
    """Amendment 1's read steps: `fixed`, plus the first saved step whose KL to start (`kl`: step -> KL) is at least
    kl_min when it is not already one of them; over n_max (the budget), the smallest step not in `fixed` goes first.
    NaN never reaches kl_min."""
    first = next((t for t in sorted(kl) if r9(kl[t]) >= kl_min), None)
    steps = sorted(set(fixed) | ({first} if first is not None else set()))
    while len(steps) > max(n_max, len(fixed)):
        steps.remove(min(t for t in steps if t not in fixed))
    return steps


def step0_ok(x: float, ref: float, rule: dict[str, float], own: bool) -> bool:
    """An adversary's step-0 KL (random sources, before any step) against its reference: G's own reading against the
    filed one within the relative `tol` (`own`), any other state's against G's on the same block within
    [state_low x ref, ref + state_add] (a broken state reads nats, not tenths). NaN misses."""
    if own:
        return r9(abs(x / ref - 1)) <= rule["tol"]
    return rule["state_low"] * ref <= r9(x) <= ref + rule["state_add"]


def fixed_fall(kl: dict[str, float], step: int | None = None) -> float:
    """Teacher training's fall on its fixed rows: the fp32 KL(teacher ‖ student) at `step` (None: the last reading)
    over step 0's (`qwen_teacher.kl_to_teacher`). NaN when step 0's is not > 0 or a reading is missing."""
    end = kl.get(str(step) if step is not None else max(kl, key=int, default="0"))
    k0 = kl.get("0", math.nan)
    return end / k0 if end is not None and k0 > 0 else math.nan


def score_amend1(res: dict[str, Any], R: dict[str, Any], last: str) -> dict[str, Any]:
    """Amendment 1's added rules, absent from E40's file: L (control and scaling), T0 as the positive control, P5 as
    a ratio with three starts near its line, and the step-100 accuracy change (P6)."""
    sc: dict[str, Any] = {}
    if "L_control" in R and "rates" in res.get("L", {}):
        rates, c = res["L"]["rates"], R["L_control"]
        k5, k6 = rates.get("1e-05", {}).get("kl_to_start"), rates.get("1e-06", {}).get("kl_to_start")
        sc["L_control"] = {"kl_1e-5": k5, "holds": k5 is not None and r9(abs(k5 / c["ref"] - 1)) <= c["tol"]}
        if k5 is not None and k6 is not None:
            ratio = k6 / k5 if k5 else math.nan
            sc["L_scaling"] = {"ratio": ratio, "holds": in_band(ratio, R["L_scaling"]["band"])}
    T0, T, Fs = res.get("T0", {}), res.get("T", {}), res.get("F", {})
    if "fall_max" in R["T0"] and "fineweb" in T0:
        f = fixed_fall(T0["kl"])
        sc["T0"] = {"fall": f, "ratio": T0["fineweb"]["ratio"],
                    "holds": r9(f) <= R["T0"]["fall_max"] and in_band(T0["fineweb"]["ratio"], R["T0"]["band"])}
    if "ratio_max" in R["P5"] and "p5" in T.get(last, {}).get("maths", {}):
        x = T[last]["maths"]["p5"]["ratio"]
        sc["P5"] = {"ratio": x, "holds": r9(x) <= R["P5"]["ratio_max"]}
    if "M" in R:  # the maths pass test: a state's maths adversary over W0's on this card (G1, mean of seeds 0-2)
        D = res.get("D", {})
        for name, st in (("M_do_nothing", D), ("M_teacher", T)):
            m = st.get(last, {}).get("maths", {})
            if "ratio" in m:
                passes = r9(m["ratio"]) <= R["M"]["line"]
                sc[name] = {"ratio": m["ratio"], "starts": len(m.get("ratios", {})), "holds": passes,
                            "reading": "pass" if passes else "sees the update"}
    if "P6" in R and "paired" in Fs:
        d = Fs["paired"]["test"]["diff"]
        sc["P6"] = {"diff": d, "n": Fs["paired"]["test"]["n"], "holds": d is not None and in_band(d, R["P6"]["band"])}
    return sc


def score(res: dict[str, Any], pred: dict[str, Any], steps: int = 100) -> dict[str, Any]:
    """Every filed rule whose inputs are in `res` (logical stage -> results; `steps`: F's last step); absent inputs, or
    a rule the predictions file does not hold (Amendment 1's L and F1), leave it unscored."""
    R, sc = pred["rules"], {}
    G, T0, A0, Fs, D, T = (res.get(k, {}) for k in STAGES)
    if "fineweb" in G:  # the apparatus control: G's FineWeb readings reproduce experiments/qwen_adversary's
        c, fw = R["G_control"], G["fineweb"]
        rel = {k: fw[k] / c[k]["ref"] - 1 for k in ("masked_kl", "adv") if k in fw}
        sc["G_control"] = {"rel": rel, "holds": all(r9(abs(v)) <= c[k]["tol"] for k, v in rel.items())}
    if "adv" in G.get("fineweb", {}) and "adv" in G.get("maths", {}):
        a = G["maths"]["adv"] / G["fineweb"]["adv"]
        m = G["maths"]["masked_kl"] / G["fineweb"]["masked_kl"]
        sc["G"] = {"adv_ratio": a, "masked_ratio": m, "verdict": g_verdict(a, m, R["G"])}
    if "fineweb" in T0 and "fall_max" not in R["T0"]:  # E40's T0; Amendment 1's positive control in score_amend1
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
    last = str(steps)
    if last in Fs.get("saved", {}) and "accuracy" in Fs and "accuracy" in A0:
        r, p1 = Fs["saved"][last], R["P1"]
        length = Fs["accuracy"]["test"]["response_mean"] / A0["accuracy"]["test"]["response_mean"]
        parts = {"gold_drop": r9(r["gold_drop"]) >= p1["gold_drop_min"], "kl": r9(r["kl_to_start"]) >= p1["kl_min"],
                 "length": r9(length) <= p1["length_ratio_max"]}
        sc["P1"] = {**r, "length_ratio": length, "parts": parts, "holds": all(parts.values())}
    first = min(D, key=int) if D else None  # E40 reads step 1 first; Amendment 1 its first read step
    if first is not None and first != last:
        sc["P2_step1"] = {"step": int(first), "ratio": D[first]["fineweb"]["ratio"],
                          "holds": r9(D[first]["fineweb"]["ratio"]) <= R["P2"]["step1_max"]}
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
    if "maths" in T.get(last, {}) and "maths" in D.get(last, {}) and "ratio_max" not in R["P5"]:
        a, b = T[last]["maths"]["adv"], D[last]["maths"]["adv"]
        sc["P5"] = {"teacher_training": a, "do_nothing": b, "holds": a < b}
    return sc | score_amend1(res, R, last)


# ----------------------------------------------------------------------------- the job


class Job:
    """The fp32 target (its original matrices are W0), the rows, the GSM sets, and the decomposition, loaded when a
    stage needs it and parked in host memory while teacher training runs. `budget_usd` is the benchmark's projection
    limit, `guard_usd` the spend past which no extra adversary start is made; `rehearse` logs a stop instead of
    stopping, so a tiny-size rehearsal runs every stage end to end."""

    def __init__(self, a: argparse.Namespace, cfg: Config, layout: Layout, *, budget_usd: float, guard_usd: float,
                 prior_usd: float = 0.0, rehearse: bool = False, uploader: Uploader | None = None,
                 kill_after_save: bool = False) -> None:
        self.cfg, self.layout, self.pred = cfg, layout, json.loads(layout.pred.read_text())
        self.out, self.decomp_dir, self.usd_per_h = Path(a.out), Path(a.decomp), a.usd_per_h
        self.budget_usd, self.guard_usd, self.prior_usd = budget_usd, guard_usd, prior_usd
        self.rehearse, self.uploader, self.kill_after_save, self.stage = rehearse, uploader, kill_after_save, ""
        self.last_step0, self.last_pass_s = math.nan, math.nan
        self.may_delete: Path | None = None  # Amendment 1: the one file B may delete when the volume is short
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        self.end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.model = load_target(DEV, F32)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
        self.web_lists = json.loads(WEB.read_text())["rows"]
        self.web = rows_of([r[: cfg.row_len] for r in self.web_lists[: cfg.n_rows]])
        self.sets = qwen_sets.load()
        perm = torch.randperm(qwen_sets.N_TRAIN, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
        self.held_idx, self.train_idx = perm[: cfg.heldout], perm[cfg.heldout :]
        self.res = {n: json.loads(p.read_text()) for n in layout.names if (p := layout.result(n)).is_file()}
        self.checks: dict[str, Any] = json.loads(f.read_text()) if (f := layout.file("checks")).is_file() else {}
        self.comps: Any = None
        self.gamma: Any = None
        self._pools: tuple[Tensor, Tensor] | None = None
        log(f"[load] {torch.cuda.get_device_name()}, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def put(self, stage: str, d: dict[str, Any]) -> None:
        self.res[stage] = d
        write_json(self.layout.result(stage), d)
        log(f"[{stage}] {json.dumps(brief(d), default=float)}")
        sc = score(self.res, self.pred, self.cfg.steps)
        write_json(self.layout.file("score"), sc)
        log(f"[score] {json.dumps(sc, default=float)}")

    def halt(self, why: str) -> None:
        """A rule's stop (not a first-minutes check): recorded in checks.json as missed, then `stop`."""
        self.record(f"stop {self.stage}: {why}", False, {}, True)
        self.stop(why)

    def stop(self, why: str) -> None:
        """End the job (<dir>/stopped.json); a rehearsal logs it and goes on."""
        if self.rehearse:
            log(f"REHEARSAL: would stop here: {why}")
            return
        write_json(self.layout.file("stopped"), {"why": why, "at_s": time.time() - T_START})
        log(f"STOP: {why}")
        sys.exit(f"STOP: {why}")

    def early(self, name: str) -> dict[str, Any] | None:
        """A filed first-minutes rule (the predictions file's "early" section); None: no such check in this run."""
        return self.pred.get("early", {}).get(name)

    def record(self, name: str, ok: bool, detail: dict[str, Any], stops: bool) -> None:
        """One entry of <dir>/checks.json, which keeps every check and stop of every job of the run (rehearsal.json
        is built from it)."""
        self.checks[name] = {"ok": ok, "stops": stops, **detail}
        write_json(self.layout.file("checks"), self.checks)

    def check(self, name: str, ok: bool, detail: dict[str, Any], stop: bool = True) -> None:
        """Record a first-minutes check in <dir>/checks.json and stop the job when it is missed (`stop`; otherwise
        it is reported only)."""
        self.record(name, ok, detail, stop)
        log(f"[check] {name}: {'ok' if ok else 'MISSED'} {json.dumps(detail, default=float)}")
        if not ok and stop:
            self.stop(f"early check {name} missed: {detail}")

    def finite(self, name: str, **values: float) -> None:
        """A NaN or infinity anywhere is a miss (the filed "finite" rule)."""
        if self.early("finite") is not None and not all(math.isfinite(v) for v in values.values()):
            self.check(f"{name} finite", False, values)

    def mem_check(self, where: str) -> None:
        """Peak GPU memory since the stage began against the one filed limit (`memory.peak_gb_max`)."""
        rule = self.early("memory")
        if rule is None:
            return
        peak = torch.cuda.max_memory_allocated() / 1e9
        self.check(f"memory {self.stage} {where}", peak <= rule["peak_gb_max"], {"peak_gb": peak})

    def t_lr(self) -> float:
        """Teacher training's rate: the one B chose (Amendment 1's filed rule), else the config's."""
        return self.res.get("B", {}).get("t_lr", self.cfg.t_lr)

    def save(self, path: Path, tensors: Any, dtype: torch.dtype, droppable: bool, meta: dict[str, str] | None = None
             ) -> None:
        save(path, tensors, dtype, droppable, log, meta)

    def keep(self, path: Path) -> None:
        """A file uploaded to W&B when the run uploads: results and logs (`run_stages`), and E40's T0 offsets; δW,
        T's offsets and the other large files stay on the volume (it outlives the pod)."""
        if self.uploader is not None:
            self.uploader.add(path)

    def spent_usd(self) -> float:
        """This job's machine time so far at --usd-per-h, plus the spend of earlier jobs of the run."""
        return (time.time() - T_START) / 3600 * self.usd_per_h + self.prior_usd

    def decomp(self, device: str = DEV) -> tuple[Any, Any]:
        if self.comps is None:
            self.comps = load_components(self.decomp_dir, DEV, getattr(torch, self.cfg.comps_dtype))
            self.gamma = load_gamma(self.decomp_dir, DEV, BF16)
        if self.comps.V["q"].device.type != device:
            self.comps, self.gamma = self.comps.to(device, self.comps.dtype), move_gamma(self.gamma, device)
            free()
        return self.comps, self.gamma

    def drop_decomp(self) -> None:
        self.comps = self.gamma = None
        free()

    def park(self) -> None:
        """The decomposition, if loaded, to host memory: a stage that needs no decomposition frees the GPU, and the
        next one that does gets it back without reloading it (~6 min from the volume)."""
        if self.comps is not None:
            self.decomp("cpu")

    def seq(self, i: int) -> Seq:
        it = self.sets["train"][i]
        return sft_seq(self.tok, it["question"], it["gold"], self.end_id, self.cfg.seq_len)

    def maths(self, full: bool = False) -> list[Tensor]:
        """The maths rows, cut to the config's n_rows and row_len unless `full` (F's and L's readings, which a
        rehearsal runs at full size)."""
        rows = json.loads(MATHS.read_text())["rows"]
        return rows_of(rows if full else [r[: self.cfg.row_len] for r in rows[: self.cfg.n_rows]])

    def alive(self) -> dict[str, Tensor]:
        """The alive index per site, cut to the config's alive_max."""
        idx = load(self.out / "alive.safetensors")
        return {s: v[: self.cfg.alive_max] for s, v in idx.items()} if self.cfg.alive_max else idx

    def fixed_rows(self) -> list[Tensor]:
        """Teacher training's fixed rows, the same at every reading: the first pc_rows / 2 maths rows and FineWeb
        evaluation rows (none of them in the training pools; at the rehearsal's sizes check_teacher.py's 8 rows)."""
        k = self.cfg.pc_rows // 2
        return self.maths()[:k] + self.web[:k]

    def alive_info(self) -> dict[str, Any]:
        return (self.res["B"] if "B" in self.res else self.res["G"])["alive"]

    def read_steps(self) -> tuple[int, ...]:
        """F's read steps (Amendment 1's rule, applied after F) or G's (E40)."""
        return tuple(self.res.get("F", {}).get("read_steps") or self.res["G"]["read_steps"])

    def dw_path(self, t: int) -> Path:
        return self.out / "dW" / f"step_{t}.safetensors"

    def dw_fp32_path(self) -> Path:
        return self.out / "dW" / f"step_{self.cfg.steps}_fp32.safetensors"

    def pools(self) -> tuple[Tensor, Tensor]:
        """Teacher training's rows: F's training sequences packed into 512-token rows, and FineWeb-Edu rows that are
        not evaluation rows (streamed once, cached on --out); both cut to row_len."""
        if self._pools is None:
            n, L = self.cfg.steps * self.cfg.batch, self.cfg.row_len
            maths = torch.tensor(pack([self.seq(i)[0] for i in self.train_idx[:n]], None, self.cfg.seq_len))
            path = self.out / "fineweb_pool.pt"
            if not path.is_file():
                taken = {tuple(r) for r in self.web_lists}
                docs = fineweb_rows(self.tok, len(self.web_lists) + self.cfg.web_pool)
                rows = [r for r in docs if tuple(r) not in taken][: self.cfg.web_pool]
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.tensor(rows, dtype=torch.int32), path)
            self._pools = maths[:, :L].contiguous(), torch.load(path).long()[:, :L].contiguous()
            log(f"[pools] {len(maths)} maths rows, {len(self._pools[1])} FineWeb rows of {L}")
        return self._pools

    def adv(self, state: Callable[[], Any], rows: list[Tensor], seed: int, name: str, block: str,
            steps: int | None = None) -> float:
        """One adversary start on `block`'s rows in `state`, its sources saved. Checked the moment step 0 is logged
        (one pass over the rows, random sources), when the predictions file holds the rules: its KL against the filed
        trace (B, G at a seed the trace has) or G's own seed-0 reading (`step0_ok`), and the pass's time against the
        benchmark's; after the start, a finite KL and peak memory."""
        comps, gamma = self.decomp()
        r0, rp = self.early("adv_step0"), self.early("adv_pass")
        traced = r0 is not None and self.stage in ("B", "G") and str(seed) in r0["refs"][block]
        bench = self.res.get("B", {}).get("adv_pass_s") if self.stage != "B" else None
        t_in = [0.0]

        def watch(msg: str) -> None:
            log(msg)
            m = STEP0.match(msg)
            if m is None:
                return
            self.last_step0, self.last_pass_s = float(m.group(2)), time.time() - t_in[0]
            if r0 is not None:
                ref = r0["refs"][block][str(seed)] if traced else self.res["G"][block]["step0"]
                self.check(f"{name} seed {seed} step-0 KL", step0_ok(self.last_step0, ref, r0, traced),
                           {"step0": self.last_step0, "ref": ref})
            if rp is not None and bench:
                self.check(f"{name} seed {seed} pass time", r9(abs(self.last_pass_s / bench - 1)) <= rp["tol"],
                           {"pass_s": self.last_pass_s, "benchmark_s": bench}, stop=False)

        with state():
            t_in[0] = time.time()
            kl, src = adversary(self.model, rows, gamma, comps, seed, log=watch, steps=steps or self.cfg.adv_steps)
        self.save(self.layout.large / "sources" / f"{name}_seed{seed}.safetensors", src, F32, droppable=True)
        log(f"[{name}] adversary seed {seed}: {kl:.4f} nats ({time.time() - t_in[0]:.0f} s, {peaks()})")
        self.finite(f"{name} seed {seed}", kl=kl)
        self.mem_check(f"{name} seed {seed}")
        return kl

    def w0_ref(self, seed: int) -> float:
        if seed == self.pred["three_starts"]["seeds"][0]:
            return self.res["G"]["fineweb"]["adv"]
        f = self.layout.file("W0_refs")
        refs = json.loads(f.read_text()) if f.is_file() else {}
        if str(seed) not in refs:
            refs[str(seed)] = self.adv(nullcontext, self.web, seed, "W0_fineweb", "fineweb")
            write_json(f, refs)
        return refs[str(seed)]

    def fineweb(self, state: Callable[[], Any], name: str) -> dict[str, Any]:
        ts = self.pred["three_starts"]
        return three_starts(lambda s: self.adv(state, self.web, s, f"{name}_fineweb", "fineweb"), self.w0_ref,
                            ts["seeds"], ts["band"], lambda: self.spent_usd() <= self.guard_usd)

    def maths_adv(self, state: Callable[[], Any], name: str) -> dict[str, Any]:
        """The state's maths-row adversary. With Amendment 1's filed maths pass test ("M"): its ratio to W0's maths
        reference on this card (G1's mean over seeds 0-2), three starts when the first lands in M's band. E40: one
        start, its ratio to G's seed-0 reading."""
        seeds, m = self.pred["three_starts"]["seeds"], self.pred["rules"].get("M")
        if m is None:
            kl = self.adv(state, self.maths(), seeds[0], f"{name}_maths", "maths")
            return {"adv": kl, "ratio": kl / self.res["G"]["maths"]["adv"]}
        ref = self.res["G"]["maths"]["mean_over_seeds"]
        return three_starts(lambda s: self.adv(state, self.maths(), s, f"{name}_maths", "maths"), lambda s: ref,
                            seeds, m["band"], lambda: self.spent_usd() <= self.guard_usd)

    def gstats(self, state: Callable[[], Any], name: str) -> None:
        comps, gamma = self.decomp()
        with state():
            for block, rows in (("maths", self.maths()), ("fineweb", self.web)):
                p = importance_pass(self.model, rows, gamma, comps, self.cfg.alive_g, masked=False)
                self.save(self.layout.large / "gstats" / f"{name}_{block}.safetensors", g_stats(p), torch.float16, True)

    def accuracy(self, W: dict[str, Tensor] | None, label: str, names: tuple[str, ...] = ("test", "gsm1k"),
                 compare: Path | None = None) -> dict:
        """Greedy strict accuracy (`qwen_eval.score`) of a bf16 model holding W (None: W0), responses to `large`.
        `compare`: an earlier run's responses to the first set; its first batch is scored alone and the share of its
        responses that differ from them reported before the rest runs (the same batches either way)."""
        tok, gen = load_model()
        if W is not None:
            with torch.no_grad():
                for s, lin in linears(gen).items():
                    lin.weight.copy_(W[s])
        out, b = {}, self.cfg.gen_batch
        for n in names:
            items, sink = self.sets[n][: self.cfg.eval_problems], self.layout.large / f"{label}_{n}.jsonl"
            first: Records = []
            if compare is not None and n == names[0]:
                first = qwen_eval.score(tok, gen, items[:b], self.cfg.gen_budget_s, sink, f"{label} {n}", b)
                self.first_batch_report(first, compare)
                items = items[b:]
            recs = first + qwen_eval.score(tok, gen, items, self.cfg.gen_budget_s, sink, f"{label} {n}", b)
            sink.write_text("".join(json.dumps(r) + "\n" for r in recs))
            out[n] = summary(recs) | {"response_mean": statistics.mean(r["response_tokens"] for r in recs)}
            self.keep(sink)
        del gen
        free()
        return out

    def first_batch_report(self, first: Records, compare: Path) -> None:
        if not compare.is_file():
            self.check("A0 first batch against the earlier run", True, {"reference": f"{compare}: absent"}, stop=False)
            return
        ref = {r["idx"]: r["response"] for r in records(compare)}
        differ = sum(ref.get(r["idx"]) != r["response"] for r in first) / len(first)
        self.check("A0 first batch against the earlier run", differ == 0, {"differ_share": differ, "n": len(first)},
                   stop=False)

    def d_maths(self, seed: int) -> float:
        """The do-nothing state's maths-row adversary at the last step and `seed` (D's seed-0 reading; other seeds
        read once and kept in <dir>/D_maths_refs.json), the reference of P5's three starts."""
        last = str(self.cfg.steps)
        read = self.res["D"][last]["maths"]["adv"]  # seed -> KL (the maths pass test's starts)
        if str(seed) in read:
            return read[str(seed)]
        f = self.layout.file("D_maths_refs")
        refs = json.loads(f.read_text()) if f.is_file() else {}
        if str(seed) not in refs:
            dW = load(self.dw_path(self.cfg.steps))
            refs[str(seed)] = self.adv(lambda: do_nothing(self.model, self.W0, dW), self.maths(), seed,
                                       f"D{last}_maths", "maths")
            write_json(f, refs)
        return refs[str(seed)]


def records(path: Path) -> Records:
    return [json.loads(line) for line in path.open()]


OnStep = Callable[[int, list[dict[str, float]]], None]


def teacher_checks(job: Job, name: str, read: Callable[[], list[float]] | None = None, reads: tuple[int, ...] = (),
                   end_stops: bool | None = True) -> tuple[OnStep, dict[str, float], dict[str, list[float]]]:
    """Teacher training's first-minutes checks, when filed: a finite KL every step and at each reading (a stop, except
    in B, `end_stops` None, where a non-finite value ends that rate's arm instead: `amend1.arm_ends`); step 1's
    training-forward KL against the fp32 KL(teacher ‖ W0) on the same batch (exactly 0 within `zero_kl_max` when there
    is no teacher, `read` None); memory after step 2. With `read` (the fixed-row KL, `qwen_teacher.kl_to_teacher`): a reading now
    (step 0) and after each step in `reads` and at `end_at`, kept in the returned dicts (step -> the mean over the rows,
    and step -> the per-row KLs; written as taken),
    and at `end_at` the fall (`fixed_fall`) at most `end_frac`: a stop for the positive control (T0), reported for T1
    (`end_stops` False), not made for B, whose rate rule reads the same ratio (`end_stops` None)."""
    rule = job.early("teacher_first")
    kl: dict[str, float] = {}
    rows: dict[str, list[float]] = {}
    end = rule["end_at"] if rule is not None and read is not None and end_stops is not None else None

    def finite(what: str, v: float) -> None:
        if end_stops is not None:
            job.finite(what, kl=v)

    def take(step: int) -> None:
        assert read is not None
        rows[str(step)] = read()
        kl[str(step)] = statistics.fmean(rows[str(step)])
        log(f"[{name}] fixed-row KL(teacher ‖ student) at step {step}: {kl[str(step)]:.4e}, rows "
            + " ".join(f"{k:.2e}" for k in rows[str(step)]))

    if read is not None:
        take(0)
        finite(f"{name} fixed-row KL step 0", kl["0"])

    def on_step(step: int, curve: list[dict[str, float]]) -> None:
        finite(f"{name} step {step}", curve[-1]["kl"])
        if step == 1 and rule is not None:
            c = curve[0]
            if read is None:
                job.check(f"{name} step-1 KL", c["kl"] <= rule["zero_kl_max"], {"kl": c["kl"]})
            elif c["kl_fp32_teacher_w0"] >= rule["fp32_min"]:
                rel = c["kl"] / c["kl_fp32_teacher_w0"] - 1
                job.check(f"{name} step-1 KL vs fp32", r9(abs(rel)) <= rule["fp32_tol"],
                          {"kl": c["kl"], "kl_fp32": c["kl_fp32_teacher_w0"], "rel": rel})
        if step == 2:
            job.mem_check(f"{name} step 2")
        if read is not None and (step in reads or step == end):
            take(step)
            finite(f"{name} fixed-row KL step {step}", kl[str(step)])
        if step == end and rule is not None:
            f = fixed_fall(kl, step)
            job.check(f"{name} fixed-row fall at step {step}", r9(f) <= rule["end_frac"], {"fall": f, "kl": dict(kl)},
                      stop=bool(end_stops))

    return on_step, kl, rows


# ----------------------------------------------------------------------------- stages


def w0_importance(job: Job, out: dict[str, Any]) -> None:
    """At W0, FineWeb first: masked KL and g statistics per block, and the alive set (g > alive_g anywhere in either
    block), saved to --out."""
    cfg = job.cfg
    comps, gamma = job.decomp()
    alive: dict[str, Tensor] = {}
    for block, rows in (("fineweb", job.web), ("maths", job.maths())):
        p = importance_pass(job.model, rows, gamma, comps, cfg.alive_g, masked=True)
        job.save(job.layout.large / "gstats" / f"W0_{block}.safetensors", g_stats(p), torch.float16, True)
        alive = {s: alive.get(s, p["any"][s]) | p["any"][s] for s in SITES}
        out[block] = {"masked_kl": p["masked_kl"]}
        log(f"[{job.stage}] {block}: masked KL {p['masked_kl']:.4f}")
        job.finite(f"{block} masked KL", kl=p["masked_kl"])
    idx = {s: alive[s].nonzero().flatten() for s in SITES}
    job.save(job.out / "alive.safetensors", idx, torch.int64, droppable=False)
    n = sum(len(idx[s]) * (comps.vu(s)[0].shape[0] + comps.vu(s)[1].shape[1]) for s in SITES)
    del comps, gamma  # no GPU copy may outlive a park (benchmark, T0)
    out["alive"] = {"subcomponents": sum(len(v) for v in idx.values()), "of": sum(len(v) for v in alive.values()),
                    "n_params": n, "offsets_gb": 20 * n / 1e9}


def w0_adversaries(job: Job, out: dict[str, Any], seeds: dict[str, tuple[int, ...]]) -> None:
    """At W0, the adversary on each block at `seeds[block]` (FineWeb first), G_control checked after every reading;
    a block's first seed gives its `adv`, `adv_s` and `step0`, every seed goes to `by_seed`."""
    for block, rows in (("fineweb", job.web), ("maths", job.maths())):
        for seed in seeds[block]:
            job.put("G", out)
            g_control(job)
            t0 = time.time()
            kl = job.adv(nullcontext, rows, seed, f"G_{block}", block)
            out[block].setdefault("by_seed", {})[str(seed)] = kl
            if seed == seeds[block][0]:
                out[block] |= {"adv": kl, "adv_s": time.time() - t0, "step0": job.last_step0}
    job.put("G", out)
    g_control(job)


def g_budget(job: Job, out: dict[str, Any]) -> None:
    """The alive cap, then the benchmark and the read steps it allows."""
    if out["alive"]["offsets_gb"] > job.cfg.offsets_gb_max:
        job.halt(f"alive offsets need {out['alive']['offsets_gb']:.1f} GB (20 B x {out['alive']['n_params']})")
    out["benchmark"] = benchmark(job, out)
    out["read_steps"] = out["benchmark"]["read_steps"]
    job.put("G", out)
    if not out["read_steps"]:
        job.halt(f"projected spend over ${job.budget_usd:.1f} even without step 10: {out['benchmark']}")
        out["read_steps"] = list(job.cfg.read_steps)  # a rehearsal goes on
        job.put("G", out)


def stage_G(job: Job) -> None:
    cfg = job.cfg
    tok, gen = load_model()
    texts = [prompt(tok, it["question"]) for it in job.sets["test"][: cfg.maths_problems]]
    t0 = time.time()
    seqs = answer_ids(tok, gen, texts, cfg.gen_batch)
    out: dict[str, Any] = {"eval_batch_s": (time.time() - t0) / math.ceil(len(texts) / cfg.gen_batch)}
    del gen
    free()
    cum = torch.tensor([len(s) for s in seqs]).cumsum(0)
    out["problems_in_rows"] = int((cum < cfg.n_rows * cfg.seq_len).sum()) + 1
    MATHS.write_text(json.dumps({"problems": cfg.maths_problems, "rows": pack(seqs, cfg.n_rows, cfg.seq_len)}))
    w0_importance(job, out)
    out["read_steps"] = list(cfg.read_steps)
    first = job.pred["three_starts"]["seeds"][0]
    w0_adversaries(job, out, {"fineweb": (first,), "maths": (first,)})
    if score(job.res, job.pred, cfg.steps)["G"]["verdict"] == "fail":
        job.halt("G fails: the decomposition does not explain the maths rows as it explains web text (card)")
    g_budget(job, out)


def g_control(job: Job) -> None:
    """Stop unless G's FineWeb readings so far reproduce experiments/qwen_adversary's (the filed G_control)."""
    c = score(job.res, job.pred, job.cfg.steps)["G_control"]
    if not c["holds"]:
        fw, ref = job.res["G"]["fineweb"], job.pred["rules"]["G_control"]
        got = {k: (fw[k], ref[k]["ref"]) for k in c["rel"]}
        job.halt(f"G control: FineWeb (this run, qwen_adversary) {got} outside the filed tolerances")


def benchmark(job: Job, g: dict[str, Any]) -> dict[str, Any]:
    """One teacher-training step timed after a first (warm-up) one, at the full alive set (its memory checked after
    step 2); the projection of T0-T from it, G's adversary starts and G's generation batches; the budget rule
    (read_steps empty: over `job.budget_usd` even without step 10). The worst case of the three-starts re-reads
    (every D and T ratio in the band: two more starts each, plus W0 at seeds 1 and 2) is projected beside it; the
    running guard (`Job.guard_usd`) caps those re-reads."""
    cfg = job.cfg
    off = Offsets(job.decomp()[0], job.alive())
    job.decomp("cpu")
    maths, web = job.pools()
    curve = train(job.model, off, None, maths, web, cfg.t_lr, 2, cfg.t_rows_each, cfg.t_micro, cfg.seed, log,
                  teacher_checks(job, "benchmark")[0])
    b: dict[str, Any] = {"teacher_step_s": curve[-1]["s"], "teacher_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    del off
    free()
    adv, gen, step = statistics.mean(g[k]["adv_s"] for k in ("maths", "fineweb")), g["eval_batch_s"], curve[-1]["s"]
    n_eval = [len(job.sets[n][: cfg.eval_problems]) for n in ("test", "gsm1k")]
    gen_all = sum(math.ceil(k / cfg.gen_batch) for k in n_eval) * gen
    sft_step_s = step * maths.numel() / cfg.steps / (2 * cfg.t_rows_each * cfg.row_len)  # rough: tokens per step

    def spend(steps: tuple[int, ...]) -> tuple[dict[str, float], float]:
        k, train_s = len(steps), cfg.t_steps * step
        p = {"T0": cfg.t0_steps * step + adv, "A0": gen_all, "F": cfg.steps * sft_step_s + gen_all,
             "D": (k + 1) * adv, "T": k * (train_s + adv) + adv + math.ceil(n_eval[0] / cfg.gen_batch) * gen}
        return p, (time.time() - T_START + sum(p.values())) / 3600 * job.usd_per_h

    def worst(steps: tuple[int, ...]) -> float:
        return (4 * len(steps) + 2) * adv / 3600 * job.usd_per_h

    steps = cfg.read_steps
    b["projected_s"], b["projected_usd"] = spend(steps)
    b["rereads_worst_usd"], b["budget_usd"], b["guard_usd"] = worst(steps), job.budget_usd, job.guard_usd
    if b["projected_usd"] > job.budget_usd:
        steps = tuple(t for t in steps if t != 10)
        b["projected_s_without_10"], b["projected_usd_without_10"] = spend(steps)
        b["rereads_worst_usd_without_10"] = worst(steps)
        if b["projected_usd_without_10"] > job.budget_usd:
            steps = ()
    return b | {"read_steps": list(steps)}


def stage_T0(job: Job) -> None:
    """Teacher training with δW = 0. With the config's shortcut, offsets that are exactly 0 make the state W0's: G's
    FineWeb reading stands (ratio 1) and the zeros are not saved; otherwise the state is saved and read."""
    cfg = job.cfg
    off = Offsets(job.decomp()[0], job.alive())
    job.decomp("cpu")
    curve = train(job.model, off, None, *job.pools(), cfg.t_lr, cfg.t0_steps, cfg.t_rows_each, cfg.t_micro, cfg.seed,
                  log, teacher_checks(job, "T0")[0])
    out: dict[str, Any] = {"t_steps": cfg.t0_steps, "max_abs_offset": off.max_abs(), "curve": [c["kl"] for c in curve],
                           "first_batch_kl_fp32": curve[0]["kl_fp32_teacher_w0"]}
    if cfg.t0_shortcut and out["max_abs_offset"] == 0.0:
        g = job.res["G"]["fineweb"]["adv"]
        out |= {"shortcut": True, "fineweb": {"adv": {"0": g}, "ratios": {"0": 1.0}, "ratio": 1.0}}
        log("[T0] every offset is exactly 0: the state is W0's, G's FineWeb reading stands; offsets not saved")
    else:
        path = job.out / "offsets" / "T0.safetensors"
        off.save(path, F32, None, log)
        job.keep(path)
        off.to("cpu")
        out["fineweb"] = job.fineweb(lambda: off.state(job.model, job.decomp()[0], job.W0), "T0")
    job.put("T0", out)
    if not score(job.res, job.pred, cfg.steps)["T0"]["holds"]:
        job.halt(f"T0: ratio {out['fineweb']['ratio']:.3f} outside {job.pred['rules']['T0']['band']}")


def held(job: Job) -> list[Seq]:
    return [job.seq(i) for i in job.held_idx]


def stage_A0(job: Job) -> None:
    job.park()
    compare = RESULTS / job.cfg.a0_compare if job.cfg.a0_compare else None
    out = {"accuracy": job.accuracy(None, "step0", compare=compare),
           "gold_loss": gold_loss(job.model, held(job), job.cfg.eval_batch, DEV)}
    job.put("A0", out)
    if not score(job.res, job.pred, job.cfg.steps)["P0_accuracy"]["holds"]:
        job.halt("P0: step-0 accuracy is not the baseline's")


def disk_sizes(cfg: Config, numel: int, n_alive_params: int) -> dict[str, int]:
    """Bytes of every large file F and T write: δW per saved step (bf16) and at the last step in fp32, the W0
    gradient (bf16), Adam's moments (fp32, if saved), and teacher-training offsets per read step."""
    sizes = {f"dW_{t}": 2 * numel for t in cfg.saved} | {"dW_fp32": 4 * numel, "grad": 2 * numel}
    if cfg.save_adam:
        sizes["adam"] = 8 * numel
    n_read = len(cfg.read_fixed) + 1 if cfg.read_fixed else len(cfg.read_steps)
    return sizes | {f"offsets_{i}": (2 if cfg.offsets_bf16_before_last and i < n_read - 1 else 4) * n_alive_params
                    for i in range(n_read)}


def f_checks(job: Job, t: int, lr: float, loss: float, norms: dict[str, float], saved: dict[str, Any]) -> None:
    """F's first-minutes checks, when filed; each missed one stops the job. Step 1: the loss (at W0, on the first
    batch) against E40's, and, if its rate is 0 (the library's warm-up), δW and the KL to start exactly 0. The first
    step with a nonzero rate: |δW| per matrix over rate·√numel in the filed band (Adam's step is about the rate per
    entry, never much above it) and the KL to start within the filed bound (F1 small early)."""
    if t == 1 and (r := job.early("F_step1")) is not None:
        job.check("F step-1 loss", r9(abs(loss / r["loss_ref"] - 1)) <= r["loss_tol"], {"loss": loss, "ref": r["loss_ref"]})
        if lr == 0.0:
            job.check("F step 1 at rate 0", max(norms.values()) == 0.0 and saved["1"]["kl_to_start"] == 0.0,
                      {"max_norm": max(norms.values()), "kl_to_start": saved["1"]["kl_to_start"]})
    if t == job.cfg.first_live():
        if (r := job.early("F_first_live")) is not None:
            ratio = {s: norms[s] / (lr * math.sqrt(job.W0[s].numel())) for s in SITES}
            lo, hi = min(ratio.values()), max(ratio.values())
            job.check(f"F step {t} |dW| / (rate √numel)", r["band"][0] <= r9(lo) and r9(hi) <= r["band"][1],
                      {"min": lo, "max": hi, "median": statistics.median(ratio.values()), "rate": lr})
        if (r := job.early("F1_early")) is not None:
            kl = saved[str(t)]["kl_to_start"]
            job.check(f"F step {t} KL to start", r9(kl) <= r["kl_max"], {"kl_to_start": kl, "max": r["kl_max"]})
    if t == 1:
        job.mem_check("step 1")


def stage_F(job: Job) -> None:
    cfg = job.cfg
    job.park()
    job.out.mkdir(parents=True, exist_ok=True)
    rewritten = [job.dw_path(t) for t in cfg.saved] + [job.dw_fp32_path(), job.out / "grad_W0.safetensors", job.out / ADAM]
    room = free_bytes(job.out) - DISK_MARGIN + sum(p.stat().st_size for p in rewritten if p.is_file())
    sizes = disk_sizes(cfg, sum(w.numel() for w in job.W0.values()), job.alive_info()["n_params"])
    drops = plan_drops(sizes, room, cfg.drop_order)
    out: dict[str, Any] = {"dropped": drops, "saved": {}, "steps": [],
                           "recipe": {"lr": cfg.lr, "warmup": cfg.warmup, "schedule": cfg.schedule, "clip": cfg.clip}}
    log(f"[F] {room / 1e9:.1f} GB for {sum(sizes.values()) / 1e9:.1f} GB of files; dropping {drops}")
    maths, hs = job.maths(full=True), held(job)

    def reading() -> dict[str, float]:
        return {"gold_drop": job.res["A0"]["gold_loss"] - gold_loss(job.model, hs, cfg.eval_batch, DEV),
                "kl_to_start": mean_kl(job.model, maths, job.W0, None)}

    def delta(dtype: torch.dtype) -> dict[str, Tensor]:  # one site at a time to the host: no GPU copy of δW
        return {s: (params[s].detach() - job.W0[s]).to("cpu", dtype) for s in SITES}

    with trainable(job.model, job.W0) as params:
        opt = adamw(params, cfg.lr)
        sched = lr_schedule(opt, cfg.warmup, cfg.steps, cfg.schedule)
        out["step0"] = reading()
        job.put("F", out)
        if not score(job.res, job.pred, cfg.steps)["P0_zero"]["holds"]:
            job.halt("P0: the step-0 readings are not zero")
        for t in range(1, cfg.steps + 1):
            idx = job.train_idx[(t - 1) * cfg.batch : t * cfg.batch]
            lr = opt.param_groups[0]["lr"]
            assert math.isclose(lr, lr_at(t, cfg.lr, cfg.warmup, cfg.steps, cfg.schedule), rel_tol=1e-9,
                                abs_tol=1e-15), (t, lr)
            opt.zero_grad(set_to_none=True)
            job.model.train()
            loss = sft_step(job.model, [job.seq(i) for i in idx], cfg.micro, DEV)
            job.model.eval()
            if t == 1:  # the gradient at W0, before any clipping
                job.save(job.out / "grad_W0.safetensors", {s: p.grad for s, p in params.items() if p.grad is not None},
                         BF16, droppable=False)
            gnorm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), cfg.clip or math.inf))
            job.finite(f"F step {t}", loss=loss, grad_norm=gnorm)
            opt.step()
            if sched is not None:
                sched.step()
            with torch.no_grad():
                norms = {s: float((params[s].detach() - job.W0[s]).norm()) for s in SITES}
            out["steps"].append({"step": t, "lr": lr, "loss": loss, "grad_norm": gnorm, "norms": norms, "batch": idx})
            log(f"[F] step {t}: lr {lr:.3e}, loss {loss:.4f}, grad norm {gnorm:.3f}, "
                f"|dW| {math.sqrt(sum(v * v for v in norms.values())):.4f}")
            if t in cfg.saved:
                name = f"dW_{t}"
                if name not in drops:
                    job.save(job.dw_path(t), delta(BF16), BF16, name in cfg.drop_order)
                if t == cfg.steps:  # exact, to continue the fine-tune
                    job.save(job.dw_fp32_path(), delta(F32), F32, droppable=False)
                out["saved"][str(t)] = reading()
                job.put("F", out)
            f_checks(job, t, lr, loss, norms, out["saved"])
        if cfg.save_adam and "adam" not in drops:
            moments, n = adam_state(opt, params)
            job.save(job.out / ADAM, moments, F32, True, meta={"step": str(n)})
            del moments
        del opt
        free()
        W = {s: p.detach() for s, p in params.items()}
        out["accuracy"] = job.accuracy(W, f"step{cfg.steps}")
        del W
    large = job.layout.large
    out["paired"] = {n: paired(records(large / f"step0_{n}.jsonl"), records(large / f"step{cfg.steps}_{n}.jsonl"))
                     for n in ("test", "gsm1k")}
    if cfg.read_fixed:
        kl = {int(t): r["kl_to_start"] for t, r in out["saved"].items() if job.dw_path(int(t)).is_file()}
        out["read_steps"] = read_rule(kl, cfg.read_fixed, cfg.read_kl_min, job.res["B"]["n_read_max"])
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
        if t == job.cfg.steps:
            rec["maths"] = job.maths_adv(state, f"D{t}")
            job.gstats(state, f"D{t}")
        res[str(t)] = rec
        job.put("D", res)


def offsets_meta(job: Job, t: int) -> dict[str, str]:
    """What a teacher-training state was trained from: the δW file's and the alive index's hashes, rate and steps. A
    saved state is resumed only when all of them match."""
    return {"dW_sha256": sha256(job.dw_path(t)), "alive_sha256": sha256(job.out / "alive.safetensors"),
            "t_lr": repr(job.t_lr()), "t_steps": str(job.cfg.t_steps), "alive_max": str(job.cfg.alive_max)}


def stage_T(job: Job) -> None:
    """Teacher training at each read step. Every state is read from its saved file (bf16 before the last read step,
    fp32 at it), whether it was trained in this job or resumed, so the two read alike."""
    cfg, res = job.cfg, job.res.get("T", {})
    for t in job.read_steps():
        if str(t) in res:
            continue
        off = Offsets(job.decomp()[0], job.alive())
        job.decomp("cpu")  # training and the readings below need no decomposition
        dW = load(job.dw_path(t), DEV)
        path, rec, meta = job.out / "offsets" / f"T{t}.safetensors", {}, offsets_meta(job, t)
        if path.is_file() and all(meta_of(path).get(k) == v for k, v in meta.items()):
            rec["resumed_from"] = str(path)
        else:
            def read(dW: dict[str, Tensor] = dW, off: Offsets = off) -> list[float]:
                return kl_to_teacher(job.model, job.fixed_rows(), job.W0, dW, off)

            on_step, rec["kl_fixed"], rec["kl_fixed_rows"] = teacher_checks(job, f"T{t}", read, end_stops=False)
            curve = train(job.model, off, dW, *job.pools(), job.t_lr(), cfg.t_steps, cfg.t_rows_each, cfg.t_micro,
                          cfg.seed, log, on_step)
            last = t == max(job.read_steps())
            off.save(path, BF16 if cfg.offsets_bf16_before_last and not last else F32, meta, log)
            rec["curve"] = [c["kl"] for c in curve]
            rec["first_batch_kl_fp32"] = curve[0]["kl_fp32_teacher_w0"]  # beside curve[0], the training forward's
            if job.kill_after_save:
                log(f"REHEARSAL: killed after saving {path}; the next job resumes from it")
                sys.exit(3)
        off.load_from(path)
        off.to("cpu")
        t_readings(job, t, off, dW, rec)
        del dW, off
        res[str(t)] = rec
        job.put("T", res)
        free()


def t_readings(job: Job, t: int, off: Offsets, dW: dict[str, Tensor], rec: dict[str, Any]) -> None:
    """One teacher-training state's readings: behaviour kept on maths rows, accuracy at the last step, the FineWeb
    adversary, and at the last step the maths pass test, P5 and g statistics."""
    cfg = job.cfg
    with torch.no_grad():
        Wt, Ws = plus(job.W0, dW), {s: job.W0[s] + off.change(s, DEV) for s in SITES}
    rec["kl_teacher_student"] = mean_kl(job.model, job.maths(), Wt, Ws)
    rec["kl_teacher_w0"] = mean_kl(job.model, job.maths(), Wt, None)
    rec["max_abs_offset"] = off.max_abs()
    del Wt
    if t == cfg.steps:
        rec["accuracy"] = job.accuracy(Ws, f"T{t}", ("test",))
    del Ws
    free()

    def state() -> Any:
        return off.state(job.model, job.decomp()[0], job.W0)

    rec["fineweb"] = job.fineweb(state, f"T{t}")
    if t == cfg.steps:
        rec["maths"] = job.maths_adv(state, f"T{t}")
        p5 = job.pred["rules"]["P5"]
        if "band" in p5:  # Amendment 1: teacher training over do-nothing on maths rows, three starts near the line
            seeds, read = job.pred["three_starts"]["seeds"], rec["maths"]["adv"]
            rec["maths"]["p5"] = three_starts(
                lambda s: read[str(s)] if str(s) in read else job.adv(state, job.maths(), s, f"T{t}_maths", "maths"),
                job.d_maths, seeds, p5["band"], lambda: job.spent_usd() <= job.guard_usd)
        job.gstats(state, f"T{t}")


# ----------------------------------------------------------------------------- entry


Stage = tuple[str, str, Callable[[Job], None]]  # name on the command line, logical stage, function
E40_STAGES: tuple[Stage, ...] = (("G", "G", stage_G), ("T0", "T0", stage_T0), ("A0", "A0", stage_A0),
                                 ("F", "F", stage_F), ("D", "D", stage_D), ("T", "T", stage_T))


def chosen(arg: str, stages: tuple[Stage, ...]) -> list[Stage]:
    """The stages named in `arg` (comma-separated), which must keep the run's order."""
    names = arg.split(",")
    order = [s[0] for s in stages]
    assert names == [n for n in order if n in names] and set(names) <= set(order), f"stages run in the order {order}"
    return [s for s in stages if s[0] in names]


def start(layout: Layout, stages: list[Stage]) -> None:
    """Before the job: the predictions file present, no finished stage rerun (`refuse_done`), the directories made
    (a dangling results/large symlink, as one arrives on a pod, replaced by a directory), the log opened and a stale
    stop file removed."""
    if not layout.pred.is_file():
        sys.exit(f"file the predictions first: {layout.pred}")
    refuse_done(layout, stages)
    if LARGE.is_symlink() and not LARGE.exists():
        LARGE.unlink()
    layout.dir.mkdir(parents=True, exist_ok=True)
    layout.large.mkdir(parents=True, exist_ok=True)
    _LOG[0] = layout.log()
    layout.file("stopped").unlink(missing_ok=True)


def done(layout: Layout) -> dict[str, float]:
    """The stages that finished (<dir>/done.json, written after each stage ends): logical stage -> time."""
    return json.loads(layout.file("done").read_text()) if layout.file("done").is_file() else {}


def refuse_done(layout: Layout, stages: list[Stage]) -> None:
    """With the layout's guard: exit if a chosen stage is marked finished. A partial results file (a stage that
    stopped or crashed) is overwritten, and said so; D and T resume it read step by read step."""
    if not layout.guard:
        return
    for name, logical, _ in stages:
        p = layout.result(logical)
        if logical in done(layout):
            sys.exit(f"refusing stage {name}: it finished ({p}); move its results aside to rerun")
        if p.is_file():
            print(f"[guard] {p} is partial: stage {name} " + ("resumes it" if logical in ("D", "T") else "overwrites it"),
                  flush=True)


def rehearsal_report(job: Job) -> None:
    """<dir>/rehearsal.json from <dir>/checks.json (every job of the rehearsal so far): every check and stop that
    missed, the ones expected to miss at tiny sizes (the config's list, matched as substrings of the name), and those
    beyond the list, which the rehearsal is for."""
    missed = {k: c for k, c in job.checks.items() if not c["ok"]}
    unexpected = [k for k in missed if not any(e in k for e in job.cfg.expected_misses)]
    write_json(job.layout.file("rehearsal"), {"expected": list(job.cfg.expected_misses), "missed": missed,
                                              "unexpected": unexpected})
    log(f"[rehearsal] {len(missed)} missed, {len(unexpected)} beyond the expected list: {unexpected}")


def run_stages(job: Job, stages: list[Stage]) -> None:
    log(f"[config] {json.dumps(asdict(job.cfg))}; stages {[s[0] for s in stages]}; out {job.out}")
    try:
        for name, logical, fn in stages:
            job.stage, t0 = logical, time.time()
            torch.cuda.reset_peak_memory_stats()
            fn(job)
            log(f"[{name}] done in {time.time() - t0:.0f} s, {peaks()}")
            job.mem_check("end")
            write_json(job.layout.file("done"), done(job.layout) | {logical: time.time()})
    finally:
        if job.rehearse:
            rehearsal_report(job)
        if job.uploader is not None:
            for p in [*sorted(job.layout.dir.glob("*.json")), job.layout.log()]:
                job.keep(p)  # results and the log, as they stand at the job's end
            job.uploader.finish()  # waits for every queued upload, also after a stop
    print("FINISHED", RESULTS, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--out", default="/workspace/e40")
    ap.add_argument("--decomp", default="/workspace/qwen3-1.7b-p-e9fcffd5-torch")
    ap.add_argument("--usd-per-h", type=float, default=3.0)
    ap.add_argument("--prior-usd", type=float, default=0.0, help="spend of earlier jobs of this run (re-read guard)")
    a = ap.parse_args()
    stages = chosen(a.stages, E40_STAGES)
    start(E40, stages)
    assert torch.cuda.get_device_properties(0).total_memory > CFG.min_card_gb * 1e9, "needs an 80 GB card (H100)"
    guard = json.loads(PRED.read_text())["three_starts"]["guard_usd"]
    run_stages(Job(a, CFG, E40, budget_usd=CFG.budget_usd, guard_usd=guard, prior_usd=a.prior_usd), stages)


if __name__ == "__main__":
    main()
