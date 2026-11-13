"""P3.4 run 3 (E178): the shuffled-reward null N alone, E69's GRPO loop on Qwen3-1.7B at peak 1e-6 after 10 warm-up steps (E69's
recipe lr1e-6w10, as run 1's null pod), on seeds 0 (E69's draw seed 69) and 70, one pod each, until its output change
reaches that of run 2's RL state at step 75 on the same seed. Plan: reports/plans/2026-11-13_plan_qwen-rl-readable-run3.md;
rules in results/predictions_run3.json (refuses to start unless it is committed unchanged; on a pod, E120_PRED_SHA).

  --pod N_s0|N_s70                    one pod: results/run3/<pod>.json and .log, large/run3/<pod>/
  --size tiny|tiny_collapse --large DIR   the rehearsal at E69's tiny config: tiny pauses after step 1 and matches at its
                                      first read (step 2); tiny_collapse meets a forced collapse rule at step 2

Every step is run 1's `train_step` (E69's `update` and its step-1, first-live and step-20 checks). The output change is
run 1's `read`, every 5 steps. A checkpoint every 25 steps lets the same command resume. The pod ends at the first of:
  collapsed  the filed collapse rule (`match3.collapse_step`) on the step records; no δW saved
  matched    a read at or above the filed target of its seed; δW saved at that step (run 2's `save_state`), and the log
             says MATCHED and the file to download
  cap        the last step, or the filed hours of training and reads; δW saved there, read for description only
Exit 0 ended ("end" in the JSON); 1 a refusal; 2 a check missed ("stopped"); 3 paused (rehearsal only).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import match3
import torch

from commute import qwen_d3
from commute import qwen_rl as q
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_sft import adamw, trainable
from commute.qwen_train import checkpoint
from commute.qwen_updates import at_least

RUN2 = load_experiment("qwen_rl_readable", "run2")  # Job, save_state, room, SEEDS (frozen, used unchanged)
RUN1, JOB, E69 = RUN2.RUN1, RUN2.JOB, RUN2.E69
RESULTS, PRED = RUN2.RESULTS, RUN2.RESULTS / "predictions_run3.json"
TAG = "1e-6w10"
D = dict[str, Any]


@dataclass(frozen=True)
class Size:
    name: str
    config: str  # E69's config size (job.SIZES): problems and samples per step
    steps: int  # the cap
    kl_every: int  # an output-change read every this many steps
    ckpt_every: int
    target_factor: float = 1.0  # times the filed target; 0 makes the first read match (the rehearsal of the match)
    collapse: tuple[float, float, int] | None = None  # (reward below, length below, run); None: the filed rule
    pause_at: int | None = None  # exit 3 after this step's checkpoint: the rehearsal of the resume
    margin_gb: float = 5.0  # disk left free (run 2's `room`)

    def __post_init__(self) -> None:
        assert self.config in JOB.SIZES and self.steps % self.kl_every == 0
        assert self.pause_at is None or (self.pause_at % self.ckpt_every == 0 and self.pause_at < self.steps)

    @property
    def saves(self) -> tuple[int, ...]:
        """For run 2's `room`: the pod writes one δW at most, at the match or the cap."""
        return (self.steps,)


SIZES = {"full": Size("full", "full", 500, 5, 25),
         "tiny": Size("tiny", "tiny", 4, 2, 1, target_factor=0.0, pause_at=1),
         "tiny_collapse": Size("tiny_collapse", "tiny", 4, 2, 1, collapse=(1.01, 1e9, 2))}


@dataclass(frozen=True)
class Pod:
    seed: int
    size: Size
    arm: str = "N"
    kl_batch: int = 4  # run 1's `output_change`: 4 maths rows a forward
    reproduces: bool = True  # E69's recipe: its step-20 reward-change check stops, as on run 1's null pod

    def __post_init__(self) -> None:
        assert self.seed in RUN2.SEEDS and self.arm == "N"

    @property
    def name(self) -> str:
        return f"N_s{self.seed}"


def results_dir(size: Size) -> Path:
    return RESULTS / ("run3" if size.name == "full" else "run3_tiny")


class Job(RUN2.Job):
    """Run 2's job (its log, files, checks and step order) at this run's rate and folders."""

    def __init__(self, pod: Pod, large: Path) -> None:
        self.pod, self.dir = pod, results_dir(pod.size)
        self.dir.mkdir(parents=True, exist_ok=True)
        if RUN2.LARGE.is_symlink() and not RUN2.LARGE.exists():  # a link synced to a pod arrives dangling
            RUN2.LARGE.unlink()
        JOB.Job.__init__(self, JOB.config(pod.size.config, TAG, draw_seed=RUN2.SEEDS[pod.seed]), pod.name)
        self.large, self.shared_large = large / pod.name, JOB.LARGE
        self.large.mkdir(parents=True, exist_ok=True)
        self.epoch = len(self.train) // self.cfg.problems


# ----------------------------------------------------------------------------- checks and the end


def early(P: D, seed: int) -> D:
    """Run 1's early checks, with E69's step-1 rollouts of this seed."""
    return P["early"] | {"e69_rollout_sha_1": P["early"]["rollout_sha_1"][str(seed)]}


def expected(job: Job, out: D, t: int, E: D) -> None:
    """Steps 1-5 against E69's null of the same seed: reward and length equal, |δW| within the filed relative
    tolerance. Logged, not a stop (a different card may draw differently; the bands catch a broken loop)."""
    ref = E["steps_1_5"][str(job.pod.seed)]
    if t > len(ref):
        return
    rec, (reward, length, dw) = out["steps"][-1], ref[t - 1]
    holds = rec["reward_rate"] == reward and rec["mean_len"] == length and abs(rec["dw_norm"] / dw - 1) <= E["steps_1_5_dw_rel"]
    out.setdefault("expected", {})[f"step {t}"] = got = {"holds": holds, "reward": rec["reward_rate"], "len": rec["mean_len"],
                                                          "dw_norm": rec["dw_norm"], "e69": ref[t - 1]}
    job.log(f"[expected] step {t} as E69's null: {'ok' if holds else 'MISSED (not a stop: tell the lead)'} {json.dumps(got)}")


def bands(job: Job, out: D, t: int, E: D) -> None:
    """The filed stop bands: |δW| at step 20 (a rate not applied), the multiple at step 100 on seed 0 (the read)."""
    filed = {20: ("dw_norm", E["dw_norm_step20"][str(job.pod.seed)])}
    if job.pod.seed == 0:
        filed[100] = ("multiple", E["multiple_step100_s0"])
    if t not in filed:
        return
    key, band = filed[t]
    got = out["steps"][-1]["dw_norm"] if key == "dw_norm" else out["reads"].get(str(t), {}).get("multiple")
    job.check(out, f"step {t}: {key} in its filed band", q.in_band(got, band), {key: got, "band": band})


def hours(out: D) -> float:
    """Training and read time so far, summed from the records (a resume does not reset it)."""
    s = sum(x["sample_s"] + x["train_s"] for x in out["steps"]) + sum(r.get("kl_s", 0.0) for r in out["reads"].values())
    return s / 3600


def ending(job: Job, out: D, t: int, P: D) -> str | None:
    """Collapse first (a null that collapsed is no null, even at the target), then the match, then the cap."""
    s = job.pod.size
    rule: D = P["collapse"]["rule"] if s.collapse is None else dict(zip(("reward_below", "len_below", "run"), s.collapse, strict=True))
    if match3.collapse_step(out["steps"], rule) is not None:
        return "collapsed"
    target = s.target_factor * P["match"]["targets"][str(job.pod.seed)]
    if str(t) in out["reads"] and at_least(out["reads"][str(t)].get("multiple"), target):
        return "matched"
    if t >= s.steps or hours(out) >= P["match"]["cap_hours"]:
        return "cap"
    return None


# ----------------------------------------------------------------------------- the arm


def after_update(job: Job, out: D, t: int, params: D, rows: torch.Tensor, P: D) -> bool:
    """Step t's read, bands and end (each read or save unless already recorded: a resume at t). True: the pod ended."""
    if t % job.pod.size.kl_every == 0 and str(t) not in out["reads"]:
        RUN1.read(job, out, t, rows, [], P["refs"], False)
    bands(job, out, t, P["early"])
    end = ending(job, out, t, P)
    if end in ("matched", "cap") and str(t) not in out["saves"]:
        RUN2.save_state(job, out, t, params)
    if end:
        out["end"] = {"why": end, "step": t, "multiple": out["reads"].get(str(t), {}).get("multiple"), "hours": hours(out)}
        msg = f"download {out['saves'][str(t)]['file']}" if end != "collapsed" else "no state saved"
        job.log(f"{end.upper()} at step {t}: {json.dumps(out['end'])}; {msg}")
    job.put(job.stage, out)
    return end is not None


def step(job: Job, out: D, params: D, opt: Any, t: int, rows: torch.Tensor, P: D) -> bool:
    RUN1.train_step(job, out, params, opt, t, early(P, job.pod.seed))
    expected(job, out, t, P["early"])
    if t % job.pod.size.ckpt_every == 0:
        checkpoint(job.large / "ckpt" / "N.pt", t, params, job.W0, opt, None, set(), set())
        job.log(f"[{job.stage}] checkpoint at step {t}")
    ended = after_update(job, out, t, params, rows, P)
    if t == job.pod.size.pause_at and not ended:
        job.log(f"[{job.stage}] paused at step {t}: run the same command")
        sys.exit(3)
    return ended


def run_pod(job: Job, P: D, waiver: str | None) -> None:
    pod, out = job.pod, job.begin()
    if out is None:
        return
    if waiver:  # the user overrode the step-20 reward-change stop: logged, not a stop (Pod.reproduces False)
        out["waived"] = {"reward change by step 20": waiver}
    if not (h := job.shared_large / "rollouts" / "H.jsonl").is_file():
        sys.exit(f"missing {h}: fetch it from W&B (the plan's set-up) before the arm")
    out |= {"pod": pod.name, "seed": pod.seed, "draw_seed": job.cfg.draw_seed, "tag": TAG, "size": pod.size.name,
            "device": torch.cuda.get_device_name()}
    out.setdefault("saves", {})
    job.fp32()
    rows = qwen_d3.maths_rows().tokens
    assert rows.shape == (qwen_d3.MATHS_ROWS, qwen_d3.ROW_LEN), rows.shape
    with trainable(job.model, job.W0) as params:
        opt = adamw(params, job.cfg.lr)
        t = E69.start_arm(job, "N", out, params, opt)
        out["saves"] = {s: v for s, v in out["saves"].items() if int(s) <= t}
        out.pop("end", None)
        RUN2.room(job, t)
        ended = t > 0 and after_update(job, out, t, params, rows, P)  # a resume finishes its checkpoint step first
        while not ended:
            t += 1
            ended = step(job, out, params, opt, t, rows, P)
    out |= {"done": True, "last": out["steps"][-1]["step"], "peaks": job.peaks()}
    job.put(job.stage, out)
    (job.large / "ckpt" / "N.pt").unlink(missing_ok=True)
    job.log(f"[{job.stage}] done at step {out['last']}; {json.dumps(out['peaks'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True, choices=[f"N_s{s}" for s in RUN2.SEEDS])
    ap.add_argument("--size", default="full", choices=tuple(SIZES))
    ap.add_argument("--large", type=Path, default=None, help="states root (default results/large/run3; the rehearsal: scratch)")
    ap.add_argument("--waive-reward-change", default=None, metavar="WHO_WHEN",
                    help="log the step-20 reward-change check instead of stopping (the user's override, recorded)")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"{PRED} is missing: predictions are filed before any run")
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    pod = Pod(int(a.pod.split("_s")[1]), SIZES[a.size], reproduces=a.waive_reward_change is None)
    job = Job(pod, a.large or RUN2.LARGE / "run3")
    run_pod(job, P, a.waive_reward_change)
    hard_exit(0)


if __name__ == "__main__":
    main()
