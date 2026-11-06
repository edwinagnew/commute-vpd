"""E122 damage_crossed: does teacher training's copy of the broad lowercase fine-tune fail the adversarial check because
of the update's damage on plain Pile text, or because of the recipe (full-matrix Adam against a rank-8 LoRA)?
Plan reports/plans/2026-11-06_plan_e122-damage-crossed-pair.md; rules and config rules.py; filing results/predictions.json
(the run refuses unless it is committed unchanged and its `config` equals CFG).

Two cells, each copied by E37's recipe and read by E39's check on Pile val rows 0-255 (set A) and 256-511 (set B):
  c1  E37's Adam fine-tune stopped at the checkpoint whose damage is nearest the LoRA's (D_L)
  c2  E39's rank-8 LoRA trained on fresh rows to the checkpoint whose damage is nearest the Adam's (D_A)
Damage = KL(W0 || W0 + dW), mean over every position of 64 plain Pile val rows (`row_copy.mean_kl`, E37's number).
Every read also records the adversary's KL at each position (the per-position excess map).

Stages, each written as it finishes (`STAGE n DONE` in the log); a stop is `stopped` in the file, exit 1:
  0  no model: the saved copies' sha256 (E49's)
  1  W0 on A and B; the saved Adam and LoRA copies on A and B (L40 references, E49 and E47); stops on a miss
  2  damage curves: Adam every 10 steps to 300, LoRA every 100 to 4,000 (then at 3e-4 if no checkpoint is in the
     band); D_A, D_L at step 300, the band, the chosen steps; fine-tunes saved
  3  the recipe on this card: the Adam (step 300) fine-tune copied, read on A; stops unless it fails
  4  cell c3 (lead, 2026-11-06): stage 3's fine-tune copied again from its least-squares split (E121's R4 start);
     do-nothing and the split start on A, the copy on A and B
  5  cell c1, 6 cell c2: do-nothing on A, the copy (behaviour, damage), the copy on A and B
  7  the excess maps and the score
A unit that might not end inside budget_s exits 3 for a resubmission (fleet/resubmit --code 3); a rerun continues
from the file and results/large (a later job first rereads W0's first block and stops if it differs).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rules import (
    ADAM_CKPTS,
    CELLS,
    CFG,
    LORA_CKPTS,
    SETS,
    case_mask,
    choose,
    excess_map,
    ratio,
    score,
    verdict,
)
from torch import Tensor

from commute import harness, lowercase, teacher, vpd
from commute.arms import save_offsets
from commute.data import batches, pile_rows
from commute.faithfulness import harness_pgd
from commute.finetune import LoraConfig, finetune_lora_arm_b
from commute.jobs import hard_exit, require_filed
from commute.row_copy import (
    Refs,
    copy_states,
    lowercase_rows,
    mean_kl,
    mean_logp,
    read_state,
)
from commute.saved_states import offsets_state

HERE, EXP = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, OUT, MAPS = RESULTS / "predictions.json", RESULTS / "damage_crossed.json", LARGE / "maps"
TRANSPORT, BF = EXP / "transport_test" / "results", EXP / "broad_fragility" / "results" / "large"
SAVED = {"tt_adam": ("trained_step1000.pt", "d7da3c255023c9c46f43f59183edae6329df94d545bff516812721a98a1092db"),  # E39
         "tt_lora": ("a2_tt_lora_s0.pt", "20b3d068b04955e07736d49c8df56ea0fd9ab6462ec06eb98f4aa9d752d67bb9")}  # E39 A2
UV = dict[str, tuple[Tensor, Tensor]]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(out: dict[str, Any]) -> None:
    out["score"] = score(out)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


def halt(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    log(f"STOP: {why}")
    hard_exit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class PositionKL:
    """The authors' recon_loss_kl (param_decomp.models.batch_and_loss_fns, the same ops), keeping each position's KL of
    the last `n` calls: after `harness_pgd`, those are the final pass over the block's batches."""

    def __init__(self, n: int) -> None:
        self.n, self.kl = n, list[Tensor]()

    def __call__(self, pred: Tensor, target: Tensor) -> tuple[Tensor, int]:
        kl = F.kl_div(torch.log_softmax(pred, dim=-1), torch.softmax(target, dim=-1), reduction="none").sum(dim=-1)
        self.kl = [*self.kl, kl.detach().float().cpu()][-self.n :]
        return kl.sum(), math.prod(pred.shape[:-1])


class Job:
    def __init__(self, out: dict[str, Any]) -> None:
        self.out, self.t0, dev = out, time.time(), "cuda"
        token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
        self.ctx = harness.load(dev, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token, 1e-6,
                                0, CFG.pgd_steps, (), log)
        self.low = lowercase.task(self.ctx.model, dev, log)
        self.plain = pile_rows("val", 0, 256, log)[: CFG.n_held]
        self.rows = {s: pile_rows("val", r, 256, log) for s, r in SETS.items()}
        tok = vpd.tokenizer()
        capital = torch.tensor([bool(re.search("[A-Z]", t)) for t in tok.convert_ids_to_tokens(list(range(len(tok))))])
        self.case = {s: case_mask(r, capital) for s, r in self.rows.items()}
        out["jobs"].append({"card": torch.cuda.get_device_name(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if "L40" not in out["jobs"][-1]["card"]:
            halt(out, f"the references are L40 readings; this job runs on {out['jobs'][-1]['card']}")

    def budget(self, est: float, what: str) -> None:
        if time.time() - self.t0 + est > CFG.budget_s:
            log(f"[budget] {what} needs ~{est:.0f} s after {time.time() - self.t0:.0f} s: exit 3 for a resubmission")
            dump(self.out)
            hard_exit(3)

    def at(self, dW: dict[str, Tensor]) -> Any:
        return harness.at(self.ctx, dW, None)

    def damage_gain(self, dW: dict[str, Tensor]) -> tuple[float, float]:
        """Damage KL(W0 || W0 + dW) on the plain rows and held-out gain, the target model at W0 + dW."""
        with self.at(dW):
            net, dev = self.ctx.model.target_model, self.ctx.device
            return (mean_kl(self.low.w0_net, net, self.plain, dev, CFG.eval_batch),
                    mean_logp(net, self.low.held, dev, CFG.eval_batch) - self.low.logp_w0)

    def state(self, name: str) -> tuple[dict[str, Tensor], UV | None]:
        """(dW, alive (U', V') or None for do-nothing) of a saved copy, an in-job copy or a fine-tune."""
        if name in SAVED:
            return offsets_state(BF / SAVED[name][0], self.ctx.uv_cpu)
        if name.startswith("tt_"):
            return offsets_state(LARGE / f"{name}.pt", self.ctx.uv_cpu)
        return torch.load(LARGE / f"ft_{name.removeprefix('nothing_')}.pt", weights_only=True), None

    def check(self, name: str, s: str) -> float:
        """The check on set s: one adversary per block at seed 0 (`harness_pgd`), each position's KL kept; stored as
        the two blocks' mean KL; returns the ratio to W0. A read already in the file is skipped."""
        key, kl = f"{name}|{s}", self.out["kl"]
        if key not in kl:
            self.budget(CFG.read_s, key)
            t1, vals, maps = time.time(), [], []
            st = None if name == "W0" else self.state(name)
            for b in (0, CFG.block):
                blk = batches(self.rows[s][b : b + CFG.block], CFG.eval_batch)
                rec = PositionKL(len(blk))
                if st is None:
                    vals.append(harness_pgd(self.ctx.model, blk, CFG.pgd_steps, CFG.adv_seed, loss=rec))
                else:
                    with harness.at(self.ctx, *st):
                        vals.append(harness_pgd(self.ctx.model, blk, CFG.pgd_steps, CFG.adv_seed, loss=rec))
                maps.append(torch.cat(rec.kl))
            m = torch.cat(maps)
            miss = abs(float(m.double().mean()) / (sum(vals) / 2) - 1)
            if not miss <= CFG.map_tol:
                halt(self.out, f"{key}: the per-position map's mean misses the adversary's KL by {miss:.2e}")
            MAPS.mkdir(parents=True, exist_ok=True)
            torch.save(m, MAPS / f"{name}_{s}.pt")
            kl[key], self.out["seconds"][key] = vals, time.time() - t1
            dump(self.out)
        r = ratio(kl, name, s)
        log(f"[{key}] blocks {[round(v, 6) for v in kl[key]]} ratio {r} verdict {verdict(r) if name != 'W0' else '-'}")
        return r if r is not None else math.nan

    def gate(self, name: str, ok: bool, rec: dict[str, Any]) -> None:
        self.out["gates"][name] = {**rec, "pass": ok}
        log(f"[gate {name}] {json.dumps(self.out['gates'][name])}")
        dump(self.out)
        if not ok:
            halt(self.out, f"{name}: {json.dumps(rec)}")


# ----------------------------------------------------------------------------- stages


def stage1(job: Job) -> None:
    kl = job.out["kl"]
    if "W0|A" in kl:  # a later job: the apparatus must read W0's first block as the first job did
        blk = batches(job.rows["A"][: CFG.block], CFG.eval_batch)
        v = harness_pgd(job.ctx.model, blk, CFG.pgd_steps, CFG.adv_seed)
        job.gate(f"rerun_job{len(job.out['jobs'])}", abs(v / kl["W0|A"][0] - 1) <= CFG.rerun_tol, {"read": v, "file": kl["W0|A"][0]})
    for s in SETS:
        job.check("W0", s)
    w0 = {s: sum(kl[f"W0|{s}"]) / 2 for s in SETS}
    job.gate("C1_w0", all(abs(w0[s] / CFG.w0_ref[s] - 1) <= CFG.repro_tol for s in SETS), {"read": w0, "ref": CFG.w0_ref})
    for name, ref in CFG.saved_ref.items():
        for s in SETS:
            job.check(name, s)
        got = {s: sum(kl[f"{name}|{s}"]) / 2 for s in SETS}
        job.gate(f"C2_{name}", all(abs(got[s] / ref[s] - 1) <= CFG.repro_tol for s in SETS), {"read": got, "ref": ref})


def curve(job: Job, deltas: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, float]]:
    out = {}
    for step, dW in sorted(deltas.items()):
        d, g = job.damage_gain(dW)
        out[step] = {"damage": d, "gain": g}
        log(f"[curve] step {step} damage {d:.5f} gain {g:.5f}")
    return out


def lora_run(job: Job, rows: Tensor, lr: float) -> dict[int, dict[str, Tensor]]:
    cfg = LoraConfig(modules=tuple(vpd.module_keys(job.ctx.model)), rank=CFG.lora_rank, lr=lr, steps=CFG.lora_steps,
                     checkpoints=LORA_CKPTS, seed=0)
    return finetune_lora_arm_b(job.ctx.model, rows, job.low.held, cfg, CFG.ft_batch, job.ctx.device, log=log).deltas


def stage2(job: Job) -> None:
    """The damage curves, D_A and D_L at step 300, the band, the chosen checkpoints; their fine-tunes saved."""
    out = job.out
    if "chosen" in out:
        return
    job.budget(CFG.curves_s, "curves")
    adam = lowercase.arm_b_states(job.ctx.model, job.ctx.device, job.low, "adam", 0, ADAM_CKPTS, log)
    ca = curve(job, adam)
    lrows = lowercase_rows(pile_rows("train", 0, CFG.lora_source_rows, log), vpd.tokenizer(), CFG.lora_steps * CFG.ft_batch)
    assert torch.equal(lrows[: job.low.rows.shape[0]], job.low.rows), "the LoRA's rows must start with E37's 2,400"
    lora = lora_run(job, lrows, CFG.lora_lrs[0])
    cl = {CFG.lora_lrs[0]: curve(job, lora)}
    D_A, D_L = ca[CFG.adam_steps]["damage"], cl[CFG.lora_lrs[0]][CFG.adam_steps]["damage"]
    g = {"adam": ca[CFG.adam_steps]["gain"], "lora": cl[CFG.lora_lrs[0]][CFG.adam_steps]["gain"]}
    out["curves"] = {"adam": ca, "lora": {f"{k:g}": v for k, v in cl.items()}}
    job.gate("C4_gain", all(abs(g[k] - CFG.gain_ref[k]) <= CFG.gain_tol for k in g), {"gain": g, "ref": CFG.gain_ref})
    job.gate("C4_damage_order", D_A > D_L, {"D_A": D_A, "D_L": D_L})
    w = CFG.band_frac * (D_A - D_L)
    c1 = choose({s: v["damage"] for s, v in ca.items()}, D_L, w)
    c2 = choose({s: v["damage"] for s, v in cl[CFG.lora_lrs[0]].items()}, D_A, w) | {"lr": CFG.lora_lrs[0]}
    if not c2["in_band"]:
        del lora  # 40 full-size changes: free them before the second run (host memory)
        lora = lora_run(job, lrows, CFG.lora_lrs[1])
        cl[CFG.lora_lrs[1]] = curve(job, lora)
        out["curves"]["lora"] = {f"{k:g}": v for k, v in cl.items()}
        c2 = choose({s: v["damage"] for s, v in cl[CFG.lora_lrs[1]].items()}, D_A, w) | {"lr": CFG.lora_lrs[1]}
    torch.save(adam[CFG.adam_steps], LARGE / "ft_adam300.pt")
    for name, c, src in (("c1", c1, adam), ("c2", c2, lora)):
        if c["in_band"]:
            torch.save(src[int(c["step"])], LARGE / f"ft_{name}.pt")
    out["band"] = {"D_A": D_A, "D_L": D_L, "w": w, "c1": [D_L - w, D_L + w], "c2": [D_A - w, D_A + w]}
    out["chosen"] = {"c1": c1 | {"target": D_L}, "c2": c2 | {"target": D_A}}
    log(f"[band] {json.dumps(out['band'])} [chosen] {json.dumps(out['chosen'])}")
    dump(out)


def make_copy(job: Job, name: str, ft: str, target: float | None, init: UV | None = None) -> None:
    """E37's copy of fine-tune `ft` (saved bf16 offsets, then read back from the file): behaviour kept s, gap, damage.
    `init`: the offsets' start, (dU, dV) per matrix (None: W0's alive U, V, E37's start)."""
    path, cps = LARGE / f"{name}.pt", job.out.setdefault("copies", {})
    dW_ft = torch.load(LARGE / f"ft_{ft}.pt", weights_only=True)
    t_net = teacher.teacher_net(job.ctx.model, dW_ft)
    if not path.is_file():
        job.budget(CFG.copy_s, name)
        t1 = time.time()
        curve_, sts = copy_states(job.ctx, t_net, job.low.rows, CFG.copy_lr, CFG.copy_batch, (CFG.copy_steps,), CFG.copy_seed,
                                  log, init)
        st = sts[CFG.copy_steps]
        save_offsets(path, job.ctx.uv_cpu, st, {"ft": ft, "step": CFG.copy_steps, "seed": CFG.copy_seed, "init": init is not None})
        cps[name] = {"seconds": time.time() - t1, "train_kl_end": sum(c["loss"] for c in curve_[-50:]) / 50}
    if "s" not in cps.get(name, {}):
        dW, uv2 = offsets_state(path, job.ctx.uv_cpu)
        gain = job.damage_gain(dW_ft)[1]
        rec = read_state(job.ctx, Refs(t_net, job.low.w0_net, job.low.held, job.plain, CFG.eval_batch), dW, uv2, False, log)
        rec |= {"s": (rec["logp"] - job.low.logp_w0) / gain, "ft_gain": gain, "sha256": sha256(path)}
        if target is not None:
            rec["in_band"] = abs(rec["damage"] - target) <= job.out["band"]["w"]
        cps[name] = cps.get(name, {}) | rec
        log(f"[copy {name}] {json.dumps(cps[name])}")
        dump(job.out)
    del t_net
    torch.cuda.empty_cache()


def stage3(job: Job) -> None:
    make_copy(job, "tt_recipe", "adam300", None)
    r = job.check("tt_recipe", "A")
    job.gate("C5_recipe", verdict(r) == "fail", {"ratio": r, "saved_copy_A": ratio(job.out["kl"], "tt_adam", "A")})


def stage_c3(job: Job) -> None:
    """Cell c3: the full Adam fine-tune (stage 3's) copied from its least-squares split (`split.split_state` offsets, as
    E121's R4) by the same recipe; do-nothing and the split start itself on A, the copy on A and B."""
    from commute import split

    job.check("nothing_adam300", "A")
    start = LARGE / "tt_c3_start.pt"
    if not start.is_file():
        st = split.split_state(torch.load(LARGE / "ft_adam300.pt", weights_only=True), job.ctx.uv_cpu, job.ctx.bases)
        save_offsets(start, job.ctx.uv_cpu, st, {"ft": "adam300", "least_squares_split": True})
    job.check("tt_c3_start", "A")
    uv = job.ctx.uv_cpu
    init = {k: (U2 - uv[k][0], V2 - uv[k][1]) for k, (U2, V2) in offsets_state(start, uv)[1].items()}
    make_copy(job, "tt_c3", "adam300", None, init)
    for s in SETS:
        job.check("tt_c3", s)


def cell(job: Job, c: str) -> None:
    ch = job.out["chosen"][c]
    if not ch["in_band"]:
        log(f"[{c}] no {CELLS[c]} checkpoint in the band: not run")
        return
    job.check(f"nothing_{c}", "A")
    make_copy(job, f"tt_{c}", c, ch["target"])
    for s in SETS:
        job.check(f"tt_{c}", s)


def stage_maps(job: Job) -> None:
    out, maps = job.out, {}
    for k in out["kl"]:
        name, s = k.split("|")
        if name != "W0":
            maps[k] = excess_map(torch.load(MAPS / f"{name}_{s}.pt"), torch.load(MAPS / f"W0_{s}.pt"), job.case[s])
    out["maps"], out["done"] = maps, True
    dump(out)
    log(f"[maps] {json.dumps(maps)}")
    log(f"[score] {json.dumps(out['score'])}")


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CFG))), "predictions were filed for another config"
    if not LARGE.is_dir():
        sys.exit(f"{LARGE} missing: run ~/commute-agents/fleet/link-data <role> first")
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"jobs": [], "kl": {}, "seconds": {}, "gates": {}}
    if "stopped" in out:
        sys.exit(f"{OUT} has a stop ({out['stopped']}): move it aside to start again")
    if out.get("done"):
        log("done already")
        hard_exit(0)
    for name, (f, want) in SAVED.items():
        if sha256(BF / f) != want:
            sys.exit(f"{BF / f} ({name}) is not E49's file")
    log("STAGE 0 DONE")
    job = Job(out)
    stages = ((1, stage1), (2, stage2), (3, stage3), (4, stage_c3), (5, lambda j: cell(j, "c1")), (6, lambda j: cell(j, "c2")),
              (7, stage_maps))
    for n, fn in stages:
        fn(job)
        log(f"STAGE {n} DONE")
    hard_exit(0)


if __name__ == "__main__":
    main()
