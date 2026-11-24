"""E272 (cell P2.66) probe, $0 on the cluster (whip-lead, 2026-11-22): pile, code and british lose held-out log-prob
from step 1 at every save of every seed under the filed recipe (Adam 3e-5, batch 8, no warm-up; step 1 already 0.011
nats of KL), while lower, french, yoruba, topic and squad learn. Is that a bug, or the recipe? At most two recipe tries,
before any read. Every stage writes results/probe.json as it ends and prints every number; a miss of 0a-0c prints
"STOP: ..." and exits 1.

  0  bug check, first: (a) a zero update through the trainer's own save-and-load path (fp32 -> bf16 file -> fp32;
     bf16 of 0 is 0 and W0 + 0 is W0) on pile, lower, code: held gain exactly 0, stop above 1e-4; (b) pile, one step
     at lr 0 (Adam built and stepped): gain, KL and size exactly 0, stop above 1e-4; (c) pile_s0_step300 re-read from
     its file: sha256 as recorded, held gain within 1e-3 of the recorded -0.068; (d) report only: the share of held
     32-grams found in the train rows (expected ~0) and the train loss path's CE at W0 on the held rows against minus
     the held log-prob (expected within 0.01 nats, bf16 autocast), and that both masks score the same columns.
  1  try 1 on pile and lower (seed 0), code too if pile passes; try 2 on pile and lower only if try 1 misses.
     Early check: the size at step 10 is at most half the filed run's (warm-up makes it ~0.1-0.3x), else STOP.

PASS (RULE below, filed in probe.json "filed" before the job): a try passes when there is a KL rung v of the
panel's ladder (kl_lo x 2^(i/2), i < kl_n) at which pile AND lower each have a save with KL (all 64 kl rows, read
from the file) within +-match_tol (10%) of v and held gain at least the task's filed floor (train/<task>.json; pile
0.0142, lower 0.0154): the panel's own matched-state rule (`panel.nearest`), on one seed, the same rung for both.

    ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/probe.py file   (once, then commit)
    sbatch --comment="E272 stage: probe recipe" cluster/convergence_panel.sbatch probe
Expected on an L40 (train logs 63726/63728: 60-65 s a 300-step batch-8 run with its ~12 saves, ~30 s start-up):
stage 0 ~2 min; a batch-32 task ~3-4 min (4x the forward-backward, up to 26 saves at ~2 s); all of it 12-22 min.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch

from commute import panel as PN
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, write_stage
from commute.open_pair import jsonable

RUN: Any = load_experiment("convergence_panel", "run")
PATH, STATES = RUN.RESULTS / "probe.json", RUN.LARGE / "probe"
RULE = ("a try passes when, at some KL rung v of the panel's ladder (kl_lo x 2^(i/2), i < kl_n), pile and lower each have a "
        "seed-0 save with KL (64 kl rows, read from the file) within +-match_tol of v and held gain >= the task's filed floor "
        "(results/train/<task>.json floor.floor): panel.nearest, the same rung for both")
BAR = 1e-4  # 0a, 0b: the reads are exact (stage 0 read a zero update as exactly 0); 1e-4 is the brief's bar
SAME = ("Adam betas (0.9, 0.999), eps 1e-8, no weight decay, clip 1, 300 steps, constant rate after warm-up: the panel's, "
        "so only rate, batch and warm-up change (departures from DAPT: betas 0.9/0.98, eps 1e-6, weight decay 0.01, linear decay "
        "to 0, 12.5K steps; no decay because the panel reads states along the run). Pile's 2,400 rows are cycled 4 times "
        "at batch 32 (TAPT's 100 epochs over task data are the precedent)")  # fmt: skip


@dataclass(frozen=True)
class Try:
    name: str
    lr: float
    batch: int
    warmup: int
    settings_source: str

    def __post_init__(self) -> None:
        assert self.warmup == round(0.06 * 300) and self.batch == 32 and 0 < self.lr < 3e-5, self

    def recipe(self, C: Any) -> PN.Recipe:
        return PN.Recipe(self.lr, self.batch, C.steps, C.grad_clip, self.warmup)


TRIES = (
    Try("try1", 1e-5, 32, 18, "lr: Gururangan et al. 2020 (Don't Stop Pretraining, App. B) DAPT 5e-4 at batch 2048 and TAPT "
        "1e-4 at batch 256, linearly scaled (Goyal et al. 2017) to batch 32: 7.8e-6 and 1.25e-5, 1e-5 between; batch: "
        "BERT's fine-tuning grid {16, 32} (Devlin et al. 2019 App. A.3), the larger, 4x the panel's tokens a step (DAPT's "
        "2048 needs accumulation the trainer lacks); warm-up: DAPT/TAPT's (RoBERTa's) 6% linear, 18 of 300. " + SAME),
    Try("try2", 3e-6, 32, 18, "lr: the geometric middle of try 1's 1e-5 and Pythia-70M's end-of-pretraining rate (Biderman et "
        "al. 2023: 1e-3 peak, cosine to 10%, at 2M tokens a step) linearly scaled to 16K tokens a step, 7.8e-7: continuing "
        "near the base's own final rate; batch and warm-up as try 1. " + SAME),
)  # fmt: skip
FILED = {"tries": [asdict(t) for t in TRIES], "rule": RULE, "bar_0ab": BAR,
         "bar_0c": 1e-3, "early_stop": "step-10 size > 0.5 x the filed run's seed-0 step-10 size"}  # fmt: skip


def log(msg: str) -> None:
    RUN.log(msg)


def rungs_passed(saves: list[dict[str, Any]], floor: float, ladder: list[float], tol: float) -> list[float]:
    return [v for v in ladder if PN.nearest(saves, "kl", v, tol, floor) is not None]


def verdict(rungs: dict[str, list[float]]) -> dict[str, Any]:
    shared = sorted(set(rungs["pile"]) & set(rungs["lower"]))
    return {"rungs": rungs, "shared": shared, "pass": bool(shared)}


def ngram_share(train: torch.Tensor, held: torch.Tensor, n: int = 32) -> float:
    """Share of held n-grams (non-overlapping, stride n) that occur anywhere in the train rows (random linear hash)."""
    coef = torch.randint(-(2**62), 2**62, (n,), generator=torch.Generator().manual_seed(0))
    h = lambda rows: (rows.long().unfold(1, n, 1) * coef).sum(-1)
    return float(torch.isin(h(held)[:, ::n].flatten(), h(train).flatten()).double().mean())


def train_path_ce(net: ST.Net, t: PN.Task) -> float:
    """The trainer's loss (`panel.masked_ce`, bf16 autocast) at W0 on the held rows, token-weighted."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(t.held), 8):
            m = t.held_mask[s : s + 8]
            loss, _ = PN.masked_ce(net.model.target_model, t.held[s : s + 8].to(net.device), m.to(net.device))
            tot, n = tot + float(loss) * int(m[:, 1:].sum()), n + int(m[:, 1:].sum())
    return tot / n


def stage0(net: ST.Net, C: Any, tasks: dict[str, PN.Task], w0: dict[str, float], out: dict[str, Any]) -> str | None:
    zero = {k: torch.zeros_like(w).cpu() for k, w in net.W0.items()}
    RUN.save_dW(STATES / "zero.pt", zero)
    d = RUN.load_dW(STATES / "zero.pt")
    a = {n: PN.held_gain(net, t, d, w0[n]) for n, t in tasks.items()}
    RUN.put(PATH, out, "0a_zero_saved", a)
    if bad := [n for n, g in a.items() if abs(g) > BAR]:
        return f"0a: a zero update through the save path reads {[a[n] for n in bad]} on {bad}"
    kl = RUN.rows_kl(C)
    b = RUN.fine_tune(net, C, tasks["pile"], 0, w0["pile"], kl, PN.Recipe(0.0, C.batch, 1, C.grad_clip), STATES / "lr0")[-1]
    RUN.put(PATH, out, "0b_lr0_step1", {k: b[k] for k in ("step", "kl", "kl_live", "size", "held_gain")})
    if max(abs(b[k]) for k in ("kl", "size", "held_gain")) > BAR:
        return f"0b: one step at lr 0 moved the model: {b}"
    rec = next(s for s in json.loads((RUN.RESULTS / "train/pile.json").read_text())["seeds"]["0"][1:] if s["step"] == 300)
    sha = hashlib.sha256((RUN.REPO / rec["file"]).read_bytes()).hexdigest()
    s300 = RUN.load_dW(rec["file"])
    c = {"sha_ok": sha == rec["sha256"], "held_gain": PN.held_gain(net, tasks["pile"], s300, w0["pile"]), "recorded": rec["held_gain"],
         "kl": PN.kl_from_w0(net, kl, s300), "recorded_kl": rec["kl"]}  # fmt: skip
    RUN.put(PATH, out, "0c_reread_pile_s0_step300", c)
    if not c["sha_ok"] or abs(c["held_gain"] - c["recorded"]) > FILED["bar_0c"]:
        return f"0c: pile_s0_step300 does not re-read as recorded: {c}"
    RUN.put(PATH, out, "0d_rows_and_masks", {n: {"held_32gram_share_in_train": ngram_share(t.train, t.held),
            "same_columns": torch.equal(t.mask.any(0), t.held_mask.any(0)) and torch.equal(t.mask.all(0), t.mask.any(0)),
            "train_path_ce": train_path_ce(net, t), "minus_held_logp": -w0[n]} for n, t in tasks.items()})  # fmt: skip
    return None


def run_try(net: ST.Net, C: Any, tr: Try, names: list[str], tasks: dict[str, PN.Task], w0: dict[str, float], out: dict[str, Any]) -> str | None:
    kl, rec = RUN.rows_kl(C), out.setdefault(tr.name, {})
    ladder = PN.rung_ladder(C.kl_lo, C.kl_n)
    for n in names:
        filed = json.loads((RUN.RESULTS / f"train/{n}.json").read_text())
        res = RUN.fine_tune(net, C, tasks[n], 0, w0[n], kl, tr.recipe(C), STATES / tr.name)
        saves = res[1:]
        for s in saves:
            log(f"[{tr.name} {n}] step {s['step']} kl {s['kl']:.6g} size {s['size']:.6g} held_gain {s['held_gain']:.6g}")
        ten = [x["size"] for x in res[0]["trace"] if x["step"] == 10] + [float("nan")]
        ref = next(x["size"] for x in filed["seeds"]["0"][0]["trace"] if x["step"] == 10)
        rec[n] = {"floor": filed["floor"]["floor"], "step10_size": ten[0], "filed_step10_size": ref,
                  "saves": [{k: s[k] for k in ("step", "kl", "size", "held_gain", "file", "sha256")} for s in saves],
                  "rungs": rungs_passed(saves, filed["floor"]["floor"], ladder, C.match_tol)}  # fmt: skip
        RUN.put(PATH, out, tr.name, rec)
        if not ten[0] <= 0.5 * ref:
            return f"{tr.name} {n}: step-10 size {ten[0]:.4g} is not under half the filed run's {ref:.4g}: warm-up not applied?"
    rec["verdict"] = verdict({n: rec[n]["rungs"] for n in ("pile", "lower")})
    RUN.put(PATH, out, tr.name, rec)
    return None


def main() -> int:
    if sys.argv[1:] == ["file"]:
        assert not PATH.exists(), f"{PATH} is already filed"
        write_stage(PATH, jsonable({"experiment": "E272", "stage": "probe recipe", "filed": FILED}))
        return 0
    assert sys.argv[1:] == [], "usage: probe.py [file]"
    C, _ = RUN.load()
    out = json.loads(PATH.read_text())
    assert out["filed"] == jsonable(FILED), "probe.json's filed rule differs from probe.py's: file it again, committed"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    tasks = {n: RUN.TK.build(net, C, n) for n in ("pile", "lower", "code")}
    w0 = {n: PN.target_logp(net, t.held, t.held_mask, None) for n, t in tasks.items()}
    RUN.put(PATH, out, "w0_logp", w0 | {"card": torch.cuda.get_device_name()})
    why = stage0(net, C, tasks, w0, out)
    for tr in TRIES if why is None else ():  # try 2 only when try 1 misses
        why = run_try(net, C, tr, ["pile", "lower"], tasks, w0, out)
        if why is not None or out[tr.name]["verdict"]["pass"]:
            break
    if why is None and out["try1"]["verdict"]["pass"]:  # code under try 1 only, when pile passes there
        why = run_try(net, C, TRIES[0], ["code"], tasks, w0, out)
    if why is not None:
        write_stage(PATH, jsonable(out | {"stopped": why}))
        print(f"STOP: {why}: tell the lead", flush=True)
        return 1
    print(f"PROBE DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
