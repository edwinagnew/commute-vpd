"""performed_anchor (E53): the stand-in ratio and the relay on two fine-tunes of the same Pile text, one the original
67M already performs (lowercase) and one it cannot (a 50-word token cipher). Plan:
reports/plans/2026-10-29_plan_performed-anchor.md ("Predictions", "Compute and code", "Filed (E53 fine-tunes)"); A0
(a0.py) ran first and passed. Rules, thresholds, expected values: results/predictions.json (filed before this ran).

Tasks (`performed`): lowercase = E37's 2,400 lowercased train rows; cipher = cased train rows 0-2399 with a0.json's
derangement pi applied in context and targets. Each fine-tuned by E37's recipe (`lowercase.arm_b_states`: Adam 3e-5,
batch 8, 300 steps, all 24 matrices) at batch orders 0 and 1; saved bf16 in results/large/.
Readings: up to 500 scored positions per task in Pile val rows 0-255, each in a 127-token window with the scored token
at index 64 (`performed.windows_at`, half 63); log-odds of the true next token (`data.NEXT`). Cells (`cells`):
T full model, U subcomponents unmasked, L Gamma-masked (Delta out); 0 = W0, 1 = the state. G = T1 - T0; gate gain
g = L01 - L00; subcomponent gain U1 - U0; masked gain L11 - L00; stand-in ratio = masked / subcomponent gain under the
least-squares split (`anchors.lifted`). Relay (`relay`, E50): under do-nothing, every alive subcomponent's output at
the scored position restored to W0's input; rho = (T1 - restored) / G.

  run.py finetune  lowercase seed 0 first (the recipe control), then the rest -> results/finetune.json
  run.py read      controls (zero update, do-nothing), nulls (random delta-W matched on off-target KL; hand write
                   matched on G), the fine-tunes, the score -> results/read.json
Every scored number goes to stdout and results/large/<stage>.log. A control miss: written, FAIL logged, exit 1.

Amendment 1 (plan section "Amendment 1 (2026-10-30)"; rules results/predictions_amend1.json): the breadth-swapped
pair, both token maps on the cased train rows 0-2399 (`performed.cipher_rows`): narrow_lower = 50 title-case word
types used as whole words lowercased (`performed.title_lower_map`, ranks 50-99 after the filter), broad_cipher = a derangement of the 500 word tokens of
ranks 50-549. Same recipe, batch orders 0 and 1, both saved at steps 10, 100, 300. No relay, no hand write.
  run.py a1_finetune  W0's check first (A0's rule on both members; stop rule) -> results/check_amend1.json, then the
                      four fine-tunes -> results/finetune_amend1.json, large/dW_<task>_s<seed>_step<n>.pt
  run.py a1_read      control: E53's lowercase and cipher seed-0 states re-read (G, ratio as read.json), then their
                      seed-1 states (reported); the twelve states split and read; the KL-matched random delta-W per
                      task; the score (`score.score_amend1`) -> results/read_amend1.json. Every row splits the
                      off-target KL by whether the position predicts a scored-type token (`kl_split`); breadth and
                      the swap gate use the true off-target part.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE, PRED, A0 = RESULTS / "large", RESULTS / "predictions.json", RESULTS / "a0.json"
ALIVE = REPO / "experiments" / "transport_test" / "results" / "large" / "alive.pt"
TASKS, SEEDS, DEVICE = ("lowercase", "cipher"), (0, 1), "cuda"
A1_TASKS, PRED_A1 = ("narrow_lower", "broad_cipher"), RESULTS / "predictions_amend1.json"
LOG = [LARGE / "run.log"]


@dataclass(frozen=True)
class Config:
    half: int = 63  # windows of 127 tokens
    n_windows: int = 500  # scored windows per task (val rows 0-255)
    n_write_windows: int = 2000  # the hand write's windows (train rows 0-255)
    window_seed: int = 0
    n_train: int = 2400  # = lowercase.STEPS x lowercase.BATCH: cipher training rows, cased train rows 0-2399
    n_held: int = 64  # the fine-tune's held-out rows (lowercase.task's; cipher: ciphered val rows 0-63)
    steps: tuple[tuple[int, ...], tuple[int, ...]] = ((300,), (10, 100, 300))  # saved steps (lowercase, cipher)
    batch: int = 32  # windows per forward
    n_boot: int = 2000
    a1_steps: tuple[int, ...] = (10, 100, 300)  # Amendment 1: saved steps, both members
    a1_lower_ranks: tuple[int, int] = (50, 100)  # narrow_lower: title-case types, 0-based ranks, end exclusive
    a1_cipher_ranks: tuple[int, int] = (50, 550)  # broad_cipher: word tokens, as E53's cipher (ranks 50-99)
    a1_cipher_seed: int = 0
    count_rows: int = 4096  # Pile train rows 0-4095 for the counts (A0's)

    def __post_init__(self) -> None:
        from commute import lowercase

        assert self.n_train == lowercase.STEPS * lowercase.BATCH and all(s[-1] == lowercase.STEPS for s in self.steps)
        assert self.a1_steps[-1] == lowercase.STEPS and self.a1_lower_ranks[1] - self.a1_lower_ranks[0] == 50
        assert self.a1_cipher_ranks[1] - self.a1_cipher_ranks[0] == 500


CFG = Config()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG[0].open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    return [_finite(v) for v in o] if isinstance(o, list | tuple) else o


def dump(name: str, obj: dict[str, Any]) -> None:
    (RESULTS / name).write_text(json.dumps(_finite(obj), indent=1, sort_keys=True))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def key(task: str, seed: int, step: int) -> str:
    return f"{task}/s{seed}/step{step}"


def stop(out_name: str, out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out_name, out)
    log(f"FAIL: {why}: stop")
    sys.exit(1)


def task_text(name: str, rows: Tensor, pi: dict[int, int]) -> tuple[Tensor, Any]:
    """The task's text of cased Pile `rows` and its scored positions (`performed`): lowercase by `case_pairs`; any
    other task (cipher, narrow_lower, broad_cipher) its token map `pi` applied by `cipher_rows`."""
    from commute import performed, vpd

    if name == "lowercase":
        cp = performed.case_pairs(rows, vpd.tokenizer())
        return cp.lower, cp.in_lower
    return performed.cipher_rows(rows, pi)


# ----------------------------------------------------------------------------- job 1: the fine-tunes


def stage_finetune(P: dict[str, Any], pi: dict[int, int]) -> None:
    from commute import lowercase, vpd
    from commute.data import pile_rows
    from commute.row_copy import mean_logp

    model = vpd.load_component_model(DEVICE)
    low = lowercase.task(model, DEVICE, log)
    held = task_text("cipher", pile_rows("val", 0, 256)[: CFG.n_held], pi)[0]
    train = task_text("cipher", pile_rows("train", 0, 4096)[: CFG.n_train], pi)[0]
    data = {"lowercase": low, "cipher": lowercase.Lower(train, held, low.w0_net, mean_logp(low.w0_net, held, DEVICE, 8))}
    out: dict[str, Any] = {"commit": git_head(), "states": {}, "logp_w0": {n: d.logp_w0 for n, d in data.items()}}
    finetune_all(model, data, dict(zip(TASKS, CFG.steps, strict=True)), out, "finetune.json", P["controls"])


def finetune_all(model: Any, data: dict[str, Any], steps: dict[str, tuple[int, ...]], out: dict[str, Any],
                 out_name: str, c: dict[str, Any]) -> None:
    """Each task of `steps` (in order) at each batch order of SEEDS: saved bf16 at its steps, recorded in `out` and
    written to `out_name` as each state finishes. Controls: bf16 copy; recipe (E53's lowercase batch order 0 only)."""
    from commute import lowercase

    for name, seed in ((n, s) for n in steps for s in SEEDS):
        for step, dW in lowercase.arm_b_states(model, DEVICE, data[name], "adam", seed, steps[name], log).items():
            path = LARGE / f"dW_{name}_s{seed}_step{step}.pt"
            torch.save({k: v.bfloat16() for k, v in dW.items()}, path)
            small = {k: v.float() for k, v in torch.load(path, weights_only=True).items()}
            rec = {"gain": lowercase.gain(model, DEVICE, data[name], dW), "gain_bf16": lowercase.gain(model, DEVICE, data[name], small),
                   "sha256": sha256(path), "file": path.name}
            out["states"][key(name, seed, step)] = rec
            dump(out_name, out)
            log(f"[finetune {key(name, seed, step)}] held-out gain {rec['gain']:.5f} nats/token, bf16 copy {rec['gain_bf16']:.5f}")
            if abs(rec["gain_bf16"] - rec["gain"]) > c["bf16_gain_tol"]:
                stop(out_name, out, f"control bf16_save at {key(name, seed, step)}")
            if (name, seed) == ("lowercase", 0) and abs(rec["gain"] - c["recipe_gain_ref"]) > c["recipe_gain_tol"]:
                stop(out_name, out, f"control recipe: gain {rec['gain']:.5f} against {c['recipe_gain_ref']}")
    out["done"] = True
    dump(out_name, out)
    log("[finetune] done")


# ----------------------------------------------------------------------------- Amendment 1, job 1: check, fine-tunes


def a1_maps() -> dict[str, dict[int, int]]:
    """Amendment 1's token maps from the counts on Pile train rows 0-4095 (tokenizer only)."""
    from commute import performed, vpd
    from commute.data import pile_rows

    tok, rows = vpd.tokenizer(), pile_rows("train", 0, CFG.count_rows)
    ids = performed.word_ranks(tok, rows)[CFG.a1_cipher_ranks[0] : CFG.a1_cipher_ranks[1]].tolist()
    return {"narrow_lower": performed.title_lower_map(tok, rows, CFG.a1_lower_ranks),
            "broad_cipher": performed.derangement(ids, CFG.a1_cipher_seed)}


def a1_check(P: dict[str, Any], maps: dict[str, dict[int, int]], net: Any) -> None:
    """A0's rule on W0 (`net`), Pile val rows 0-255: log p(a) - log p(b) at narrow_lower's positions in its rows
    (a = lowercase form) and in the cased rows, and at broad_cipher's (a = pi(x)); repeats (the form a already earlier
    in the row) and first occurrences reported apart. -> results/check_amend1.json; the filed stop rule."""
    from a0 import spread

    from commute import performed, vpd
    from commute.data import pile_rows

    tok, val, t = vpd.tokenizer(), pile_rows("val", 0, 256), P["check"]
    nl, bc = (performed.cipher_rows(val, maps[n]) for n in A1_TASKS)
    reads = {"narrow_lower": nl, "cased": (val, nl[1]), "broad_cipher": bc}
    out: dict[str, Any] = {"commit": git_head(), "maps": {n: [[x, y, tok.decode([x]), tok.decode([y])] for x, y in m.items()]
                                                          for n, m in maps.items()}}
    for name, (rows, at) in reads.items():
        d, rep = performed.log_odds(net, rows, at, DEVICE, 8), performed.earlier_in_row(val, at)
        out[name] = {**spread(d, at.row), "mean_repeat": float(d[rep].mean()), "mean_first": float(d[~rep].mean()),
                     "n_repeat": int(rep.sum())}
        log(f"[check {name}] {json.dumps(_finite(out[name]), sort_keys=True)}")
        if name != "broad_cipher":  # per type (the capitalised form b): mean log-odds and count
            out[name]["per_type"] = {tok.decode([int(b)]): [float(d[at.b == b].mean()), int((at.b == b).sum())]
                                     for b in at.b.unique()}
            log(f"[check {name} per type] {json.dumps(out[name]['per_type'], sort_keys=True)}")
    m = {k: out[k]["mean"] for k in reads}
    stops = {"narrow_lower_below_bar": not m["narrow_lower"] >= t["narrow_lower_at_least"],
             "broad_cipher_above_bar": not m["broad_cipher"] <= t["broad_cipher_at_most"],
             "non_finite": not all(math.isfinite(x) for x in m.values())}
    out["verdict"] = {"P_cased_follows_style": m["cased"] <= t["cased_at_most"], "stops": stops,
                      "decision": "STOP" if any(stops.values()) else "GO"}
    dump("check_amend1.json", out)
    log(f"[check verdict] {json.dumps(out['verdict'], sort_keys=True)}")
    if out["verdict"]["decision"] != "GO":
        stop("check_amend1.json", out, "check: a member is not what the plan needs")


def stage_a1_finetune(P: dict[str, Any], maps: dict[str, dict[int, int]]) -> None:
    from commute import teacher, vpd
    from commute.data import pile_rows
    from commute.lowercase import Lower
    from commute.row_copy import mean_logp

    model = vpd.load_component_model(DEVICE)
    w0 = teacher.teacher_net(model, {})
    a1_check(P, maps, w0)
    held, train = pile_rows("val", 0, 256)[: CFG.n_held], pile_rows("train", 0, 4096)[: CFG.n_train]
    out: dict[str, Any] = {"commit": git_head(), "states": {}, "logp_w0": {}, "train_share": {}}
    data = {}
    for n in A1_TASKS:
        h, (rows, at) = task_text(n, held, maps[n])[0], task_text(n, train, maps[n])
        data[n] = Lower(rows, h, w0, mean_logp(w0, h, DEVICE, 8))
        out["logp_w0"][n], out["train_share"][n] = data[n].logp_w0, len(at) / rows[:, 1:].numel()
        log(f"[data {n}] W0 held-out logp {data[n].logp_w0:.5f}; share of train tokens mapped {out['train_share'][n]:.4f}")
    finetune_all(model, data, {n: CFG.a1_steps for n in A1_TASKS}, out, "finetune_amend1.json", P["controls"])


# ----------------------------------------------------------------------------- job 2: the readings


@dataclass
class Task:
    name: str
    held: Tensor  # (N, 127) scored windows
    target: Tensor  # (N, 127) bool: the window's token is a scored-type token (`performed.position_mask`)
    train: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))  # hand write's windows (E53 read)
    io0: Any = None  # W0's input and output of each matrix at `half`, per batch (relay; E53's read only)
    T0f: float = 0.0  # W0's full-model log-odds (plain forward, as the relay reads; E53's read only)


class Run:
    def __init__(self, P: dict[str, Any]) -> None:
        from commute import anchors as AN
        from commute import vpd

        self.P, self.th = P, P["thresholds"]
        self.model = vpd.load_component_model(DEVICE)
        self.W0, self.comps0 = vpd.target_weights(self.model), vpd.component_weights(self.model)
        alive = {k: v.cpu().bool() for k, v in torch.load(ALIVE, weights_only=False)["alive"].items()}
        self.lift = AN.lift_context(self.model, alive)
        self.every_alive = {k: v.nonzero().flatten() for k, v in alive.items()}

    def weights(self, dW: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: w + dW[k].to(w.device, w.dtype) if k in dW else w for k, w in self.W0.items()}

    def scored(self, name: str, pi: dict[int, int]) -> Task:
        """The task's scored windows (val rows 0-255) and their scored-type mask, cut by the same draw."""
        from commute import performed
        from commute.data import pile_rows

        text, at = task_text(name, pile_rows("val", 0, 256), pi)
        w, idx = performed.windows_at(text, at, CFG.half, CFG.n_windows, CFG.window_seed)
        assert torch.equal(w[:, CFG.half + 1], at.a[idx]), "a window's next token is not its scored form"
        mask, idx_m = performed.windows_at(performed.position_mask(text, at).long(), at, CFG.half, CFG.n_windows, CFG.window_seed)
        assert torch.equal(idx, idx_m) and bool(mask[:, CFG.half + 1].all()), "the mask's windows are not the text's"
        return Task(name, w, mask.bool())

    def task(self, name: str, pi: dict[int, int]) -> Task:
        """E53's read: `scored` plus the hand write's windows (train rows 0-255), W0's trigger_io and T0f."""
        from commute import performed
        from commute import recruitment as RT
        from commute.data import pile_rows

        t = self.scored(name, pi)
        text, at = task_text(name, pile_rows("train", 0, 4096)[:256], pi)
        t.train = performed.windows_at(text, at, CFG.half, CFG.n_write_windows, CFG.window_seed)[0]
        t.io0 = RT.trigger_io(self.model, sorted(self.comps0), t.held, CFG.half, CFG.batch)
        t.T0f = full_read(self, t, {})[0]
        return t


def batches(t: Task, x: Tensor | None = None) -> list[Tensor]:
    """`t.held` (or `x`, a per-window tensor of the same length) in batches of CFG.batch, on the GPU."""
    x = t.held if x is None else x
    return [x[s : s + CFG.batch].to(DEVICE) for s in range(0, len(t.held), CFG.batch)]


def kl_split(ref: Tensor, logits: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Per window, KL(W0 || state) at the window's positions other than `half`, split by what the position predicts:
    'kl_off' the mean over positions whose next token is not a scored-type token (the true off-target part), and the
    sum and count over those whose next token is one (logit position j predicts token j + 1; the last predicts
    nothing in the window and counts as off). Amendment 1; E53's `kl_other` pools both."""
    from commute.vpd import kl_per_position

    kl = kl_per_position(ref, logits)
    on = torch.zeros_like(target)
    on[:, :-1] = target[:, 1:]
    on[:, CFG.half] = False
    off = ~on
    off[:, CFG.half] = False
    return {"kl_off": (kl * off).sum(1) / off.sum(1), "kl_on_sum": (kl * on).sum(1), "kl_on_n": on.sum(1).double()}


@torch.no_grad()
def cells(run: Run, t: Task, dW: dict[str, Tensor], comps: Any) -> dict[str, Any]:
    """Per-window cells for target W0 + dW with components `comps` (None: released, do-nothing), one batch's gates at
    a time (500 windows' gate patterns would take ~10 GB of host memory each). Means, gains, and the per-window gains."""
    from commute.data import NEXT, target_tokens
    from commute.faithfulness import model_state
    from commute.identity import log_odds
    from commute.local_faith import masked_lo_windows, stats
    from commute.vpd import compute_ci

    m, h = run.model, CFG.half
    cols: dict[str, list[Tensor]] = {}

    def mlw(b: Tensor, g: dict[str, Tensor]) -> Tensor:
        return masked_lo_windows(m, b, h, NEXT, [g], len(b))

    for b, tm in zip(batches(t), batches(t, t.target), strict=True):
        tgt, ci0 = target_tokens(NEXT, b, h), compute_ci(m, b, chunk=len(b))
        ones = {k: torch.ones_like(v) for k, v in ci0.g.items()}
        part = {"T0": log_odds(ci0.logits[:, h], tgt), "L00": mlw(b, ci0.g), "U0": mlw(b, ones)}
        with model_state(m, target=run.weights(dW), components=comps):
            ci1 = compute_ci(m, b, chunk=len(b))
            part |= {"T1": log_odds(ci1.logits[:, h], tgt), "kl_other": stats(ci0.logits, ci1.logits, h, tgt)["kl_other"],
                     **kl_split(ci0.logits, ci1.logits, tm), "L11": mlw(b, ci1.g), "U1": mlw(b, ones)}
            with model_state(m, components=run.comps0):
                part["L01"] = mlw(b, ci1.g)
        for k, v in part.items():
            cols.setdefault(k, []).append(v.double().cpu())
        del ci0, ci1, ones
    w = {k: torch.cat(v) for k, v in cols.items()}
    piece_w, masked_w = w["U1"] - w["U0"], w["L11"] - w["L00"]
    mean = {k: float(v.mean()) for k, v in w.items()}
    return {"cells": mean, "G": mean["T1"] - mean["T0"], "g": mean["L01"] - mean["L00"], "piece_gain": float(piece_w.mean()),
            "masked_gain": float(masked_w.mean()), "kl_other": mean["kl_other"], "kl_off": mean["kl_off"],
            "kl_on_target": mean["kl_on_sum"] / mean["kl_on_n"] if mean["kl_on_n"] > 0 else math.nan, "n": len(piece_w),
            "piece_se": float(piece_w.std() / math.sqrt(len(piece_w))), "boot": boot(masked_w, piece_w, run.th)}


def boot(num: Tensor, den: Tensor, th: dict[str, Any]) -> dict[str, Any]:
    """Reported only: the 90% interval of mean(num) / mean(den) over resamples of windows whose mean(den) >= floor_min."""
    idx = torch.randint(len(num), (CFG.n_boot, len(num)), generator=torch.Generator().manual_seed(0))
    nb, db = num[idx].mean(1), den[idx].mean(1)
    ok = db >= th["floor_min"]
    r = (nb[ok] / db[ok]).sort().values
    ci = [float(r[int(0.05 * len(r))]), float(r[int(0.95 * len(r)) - 1])] if len(r) >= 20 else None
    return {"ratio_ci90": ci, "frac_den_below_floor_min": 1 - float(ok.double().mean())}


def split_row(run: Run, t: Task, dW: dict[str, Tensor]) -> dict[str, Any]:
    from commute import anchors as AN

    return cells(run, t, dW, AN.lifted(run.model, run.lift, dW))


@torch.no_grad()
def full_read(run: Run, t: Task, dW: dict[str, Tensor]) -> tuple[float, float]:
    """(mean log-odds of the true next token, mean kl_other to W0) of the full model W0 + dW on the scored windows."""
    from commute.data import NEXT, target_tokens
    from commute.faithfulness import model_state
    from commute.finetune import logits_of
    from commute.local_faith import stats

    lo = kl = 0.0
    for b in batches(t):
        ref = logits_of(run.model.target_model, b)
        with model_state(run.model, target=run.weights(dW)):
            st = stats(ref, logits_of(run.model.target_model, b), CFG.half, target_tokens(NEXT, b, CFG.half))
        lo, kl = lo + float(st["lo_trigger"].double().sum()), kl + float(st["kl_other"].double().sum())
    return lo / len(t.held), kl / len(t.held)


@torch.no_grad()
def relay(run: Run, t: Task, dW: dict[str, Tensor]) -> dict[str, float]:
    """E50's every-alive restore share under do-nothing: lo of the full model W0 + dW, plain and with every alive
    subcomponent's output at the scored position restored to W0's input (`recruitment.restore_hooks`)."""
    from commute import recruitment as RT
    from commute.data import NEXT, target_tokens
    from commute.faithfulness import model_state
    from commute.finetune import logits_of
    from commute.identity import log_odds

    def lo() -> float:
        return sum(float(log_odds(logits_of(run.model.target_model, b)[:, CFG.half], target_tokens(NEXT, b, CFG.half))
                         .double().sum()) for b in batches(t)) / len(t.held)

    with model_state(run.model, target=run.weights(dW)):
        T1 = lo()
        with RT.restore_hooks(run.model, run.comps0, run.every_alive, t.io0, CFG.half):
            R = lo()
    return {"T1": T1, "G": T1 - t.T0f, "restored": R, "nats": T1 - R, "rho": RT.share(T1, R, T1 - t.T0f)}


# ----------------------------------------------------------------------------- the rows


def random_row(run: Run, t: Task, dW_ft: dict[str, Tensor], kl_target: float) -> dict[str, Any]:
    """The seed-0 random direction of the fine-tune's per-matrix norms, scaled so its full-model kl_other matches."""
    from commute import perturb as PT
    from commute.finetune import random_direction

    d = random_direction(dW_ft, 0)
    m = PT.match(lambda c: full_read(run, t, PT.scaled(d, c))[1], kl_target, run.th["scale_cap"], run.th["kl_rel_tol"])
    return {**split_row(run, t, PT.scaled(d, m["x"])), "scale": m["x"], "match": m["status"], "kl_target": kl_target}


def write_row(run: Run, t: Task, G_target: float) -> dict[str, Any]:
    """E46's hand write on block 1's MLP output, keys = the true next tokens (`performed.next_token_inputs`), alpha
    matched so the full model's G equals the seed-0 fine-tune's; split, read and relayed."""
    from commute import performed
    from commute import perturb as PT

    assert len(t.train), "the hand write needs Run.task's train windows"
    A, keys = performed.next_token_inputs(run.model, t.train, CFG.half)
    assert len(keys) <= A.shape[0], "more keys than input dimensions: the write is not exact"
    lm = run.model.target_model.lm_head.weight.detach().float()
    X = PT.key_write(A, lm[torch.tensor(keys, device=lm.device)].T)
    fit = PT.write_fit(X, A, PT.window_inputs(run.model, t.held, CFG.half), lm[t.held[:, CFG.half + 1].to(lm.device)])
    if not G_target > 0:
        return {"match": "no_target", "fit": fit, "n_keys": len(keys)}
    dw = lambda a: {PT.WRITE_KEY: (a * X).float()}
    m = PT.match(lambda a: full_read(run, t, dw(a))[0] - t.T0f, G_target, run.th["alpha_max"], run.th["write_rel_tol"])
    return {**split_row(run, t, dw(m["x"])), "relay": relay(run, t, dw(m["x"])), "alpha": m["x"], "match": m["status"],
            "G_target": G_target, "fit": fit, "n_keys": len(keys)}


class Rows:
    """read.json's rows: each written and logged as it finishes; a control miss stops the run."""

    def __init__(self, name: str = "read.json") -> None:
        self.name, self.out = name, {"commit": git_head(), "rows": {}}

    def put(self, name: str, role: str, rec: dict[str, Any], ok: bool | None = None) -> dict[str, Any]:
        self.out["rows"][name] = {"role": role, **rec, **({} if ok is None else {"ok": ok})}
        dump(self.name, self.out)
        log(f"[{role} {name}] ok {ok} {json.dumps(_finite(rec), sort_keys=True)}")
        if ok is False:
            stop(self.name, self.out, f"control {role} at {name}")
        return rec


def state_loader(ft: dict[str, Any], keys: list[str] | None = None) -> Any:
    """A loader of `ft`'s saved states (fp32), after checking each file of `keys` (default all) exists with the
    sha256 `ft` recorded; exits otherwise."""
    files = {k: LARGE / v["file"] for k, v in ft["states"].items() if keys is None or k in keys}
    if bad := [k for k, p in files.items() if not p.is_file() or sha256(p) != ft["states"][k]["sha256"]]:
        sys.exit(f"missing or not the file the fine-tune record holds: {bad}")
    return lambda k: {n: v.float() for n, v in torch.load(files[k], weights_only=True).items()}


def stage_read(P: dict[str, Any], pi: dict[int, int], ft: dict[str, Any]) -> None:
    c, rows = P["controls"], Rows()
    state = state_loader(ft)
    run = Run(P)
    tasks = {n: run.task(n, pi) for n in TASKS}
    for n, t in tasks.items():  # the controls, first and alone
        zero = {k: torch.zeros_like(v) for k, v in run.W0.items()}
        z, rz = split_row(run, t, zero), relay(run, t, zero)
        ok = all(abs(z[q]) <= c["zero_tol"] for q in ("G", "g", "piece_gain", "masked_gain")) and abs(rz["nats"]) <= c["zero_relay_tol"]
        rows.put(f"{n}/zero", "control", {**z, "relay": rz, "T0_full": t.T0f, "n_write_windows": len(t.train)}, ok)
    scored = [key(n, s, 300) for n in TASKS for s in SEEDS]
    for k in scored:
        d = cells(run, tasks[k.split("/")[0]], state(k), None)
        rows.put(f"{k}/do_nothing", "control", d, abs(d["piece_gain"]) <= c["zero_tol"])
    for n, t in tasks.items():  # the nulls, against the seed-0 fine-tune
        ft0 = rows.out["rows"][f"{key(n, 0, 300)}/do_nothing"]
        rows.put(f"{n}/random", "null", random_row(run, t, state(key(n, 0, 300)), ft0["kl_other"]))
        rows.put(f"{n}/write", "null", write_row(run, t, ft0["G"]))
    for k in ft["states"]:  # the fine-tunes; a split state's G must be its do-nothing's (same target weights)
        dW, t = state(k), tasks[k.split("/")[0]]
        rec = {**split_row(run, t, dW), "relay": relay(run, t, dW)}
        same = rows.out["rows"].get(f"{k}/do_nothing")
        rows.put(k, "finetune", rec, None if same is None else abs(rec["G"] - same["G"]) <= c["zero_tol"])
    from score import score

    rows.out["score"] = score(rows.out["rows"], P)
    dump("read.json", rows.out)
    log(f"[score] {json.dumps(_finite(rows.out['score']), sort_keys=True)}")


# ----------------------------------------------------------------------------- Amendment 1, job 2: the readings


E53_REREAD = [key(n, s, 300) for s in SEEDS for n in TASKS]  # seed 0 of each task first: the controls


def reread_e53(run: Run, rows: Rows, c: dict[str, Any], state: Any) -> None:
    """Control, first: E53's seed-0 step-300 states (lowercase and cipher) split and read by this code; G and stand-in
    ratio must match read.json's within the filed tolerances. Then the seed-1 states, reported: all four carry the
    split off-target KL (`kl_split`) that E53's read did not."""
    ref = json.loads((RESULTS / "read.json").read_text())["rows"]
    tasks = {"lowercase": run.scored("lowercase", {}), "cipher": run.scored("cipher", e53_pi())}
    for k in E53_REREAD:
        d, r = split_row(run, tasks[k.split("/")[0]], state(k)), ref[k]
        ratio, ref_ratio = d["masked_gain"] / d["piece_gain"], r["masked_gain"] / r["piece_gain"]
        ok = abs(d["G"] - r["G"]) <= c["repro_G_tol"] and abs(ratio - ref_ratio) <= c["repro_ratio_tol"]
        rec = {**d, "ratio": ratio, "ref_G": r["G"], "ref_ratio": ref_ratio, "within_tol": ok}
        rows.put(f"e53/{k}", "control" if "/s0/" in k else "reference", rec, ok if "/s0/" in k else None)


def stage_a1_read(P: dict[str, Any], maps: dict[str, dict[int, int]], ft: dict[str, Any]) -> None:
    check = json.loads((RESULTS / "check_amend1.json").read_text())["maps"]
    assert {n: [[x, y] for x, y in m.items()] for n, m in maps.items()} == {n: [r[:2] for r in m] for n, m in check.items()}, \
        "the maps recomputed here are not check_amend1.json's"
    rows = Rows("read_amend1.json")
    e53_state = state_loader(json.loads((RESULTS / "finetune.json").read_text()), E53_REREAD)
    state = state_loader(ft)
    run = Run(P)
    reread_e53(run, rows, P["controls"], e53_state)
    tasks = {n: run.scored(n, maps[n]) for n in A1_TASKS}
    for k in ft["states"]:
        rows.put(k, "finetune", split_row(run, tasks[k.split("/")[0]], state(k)))
    for n, t in tasks.items():  # the null against the seed-0 step-300 fine-tune's off-target KL
        ft0 = rows.out["rows"][key(n, 0, 300)]
        rows.put(f"{n}/random", "null", random_row(run, t, state(key(n, 0, 300)), ft0["kl_other"]))
    from score import score_amend1

    rows.out["score"] = score_amend1(rows.out["rows"], json.loads((RESULTS / "read.json").read_text()), P)
    dump(rows.name, rows.out)
    log(f"[score] {json.dumps(_finite(rows.out['score']), sort_keys=True)}")


def e53_pi() -> dict[int, int]:
    a0 = json.loads(A0.read_text())
    if a0.get("verdict", {}).get("decision") != "GO":
        sys.exit("a0.json is not a GO: the fine-tunes do not start")
    return {int(m["x"]): int(m["pi_x"]) for m in a0["cipher_map"]}


def done(name: str, before: str) -> dict[str, Any]:
    ft = json.loads((RESULTS / name).read_text())
    if not ft.get("done"):
        sys.exit(f"{name} is not done: run.py {before} first")
    return ft


STAGES = {"finetune": PRED, "read": PRED, "a1_finetune": PRED_A1, "a1_read": PRED_A1}


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) == 2 else ""
    if stage not in STAGES:
        sys.exit(f"usage: run.py {' | '.join(STAGES)}")
    if not STAGES[stage].is_file():
        sys.exit(f"file the predictions first: {STAGES[stage]}")
    P = json.loads(STAGES[stage].read_text())
    pi, maps = (e53_pi(), {}) if stage in ("finetune", "read") else ({}, a1_maps())
    LARGE.mkdir(parents=True, exist_ok=True)
    LOG[0] = LARGE / f"{stage}.log"
    LOG[0].write_text("")
    log(f"[start] {stage} on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}, commit {git_head()}")
    if stage == "finetune":
        stage_finetune(P, pi)
    elif stage == "read":
        stage_read(P, pi, done("finetune.json", "finetune"))
    elif stage == "a1_finetune":
        stage_a1_finetune(P, maps)
    else:
        stage_a1_read(P, maps, done("finetune_amend1.json", "a1_finetune"))
    peak = {"gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "host_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    log(f"[done] {stage}; peak {json.dumps(peak)}")


if __name__ == "__main__":
    main()
