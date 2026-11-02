"""neuron_sparsity (E63): do fine-tunes built to elicit something the model already does sit on fewer MLP neurons than
fine-tunes built to teach something new? Plan: reports/plans/2026-11-02_plan_e63-neuron-sparsity.md. Every setting, rule
and threshold: results/predictions.json (filed before the run, amended once before it, 2026-11-02; read here, never edited after the run). $0, saved states only.

Readings (`commute.neuron_sparsity`, whose docstring gives the matrices' names and axes):
  W1   participation ratio over the 12288 neurons of each neuron's squared update, as a fraction of 12288
  W1g  W1 on the gradient of the task's fine-tune loss at W0 on the fine-tune's first batch at that seed (fp32); its
       reference is the same batch with its targets shuffled
  W2   share of the update along each neuron's own W0 row and column, energy-weighted
  A1   per window, participation ratio of the squared (tuned - W0) activation at the answer position; median, as a
       fraction; its references are a per-state Gaussian update on the eight MLP matrices and the rotated value
  C1   per window, neurons ranked by that window's own |shift| at the answer position, the top k take the tuned model's
       activations at every position of a W0 forward (down-projection at W0, as E56); the log2 k, interpolated, where the
       mean share over the scoring half first reaches half of the state's own 12288-neuron share; its reference is the
       same reading on one seeded random order of the neurons per state

  run.py   sha256 of every file before the model loads
           stage 0, alone (exit 1 on a miss): synthetic controls, identity patch per task, the 16-neuron update's A1 and
                   C1, E56's N16 and full-width share on elicit_s0 and cipher_s0  -> results/stage0.json
           stage 1: W1 and W2 per state                                           -> results/weights.json
           stage 2: W1g per task and seed                                         -> results/gradients.json
           stage 3: A1, C1 and the share curve per state                          -> results/windows.json
           then T1, T2, T3, the basis guard and the tree's branch                 -> results/score.json
Every scored number is printed. Target model only, fp32, TF32 off.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as NF
from torch import Tensor, nn

from commute import elicit as EL
from commute import finetune as FT
from commute import keydep as KD
from commute import lowercase as LC
from commute import neuron_sparsity as NS
from commute import neurons as NE
from commute import steering as ST
from commute.data import pile_rows, target_tokens
from commute.row_copy import lowercase_rows

HERE = Path(__file__).resolve().parent
RESULTS, REPO, EXP = HERE / "results", HERE.parents[1], HERE.parent
PRED = RESULTS / "predictions.json"
E56 = EXP / "keydependent_patch" / "results"
BATCH, DEVICE = 64, "cuda"  # windows per forward: E53's 127-token windows at 500 would hold 13 GB of logits at once
FULL = NS.N_NEURONS
log = KD.log
Files = dict[str, tuple[Path, str]]
LossOf = Callable[[nn.Module], tuple[Tensor, dict[str, float]]]


def truth_table() -> ModuleType:
    """E60's truth-table run.py, loaded by path for its loaders (`files`, `state`, `e53_tasks`, `performed_anchor`)."""
    spec = importlib.util.spec_from_file_location("regating_truth_table_run", EXP / "regating_truth_table" / "run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


TT = truth_table()


clean = TT.clean


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)


class Out:
    """One results file: each record written and logged as it lands; a miss (ok False) writes, logs FAIL and exits 1."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD").stdout.strip(),
                                    "git_dirty": git("status", "--porcelain", "--untracked-files=no").stdout, "rows": {}}

    def dump(self) -> None:
        (RESULTS / self.name).write_text(json.dumps(clean(self.out), indent=1, sort_keys=True))

    def put(self, key: str, rec: dict[str, Any], ok: bool | None = None) -> dict[str, Any]:
        self.out["rows"][key] = {**rec, **({} if ok is None else {"ok": bool(ok)})}
        self.dump()
        log(f"[{self.name} {key}] ok {ok} {json.dumps(clean(rec), sort_keys=True)}")
        if ok is False:
            self.out["stopped"] = key
            self.dump()
            log(f"FAIL: {key}: stop and tell the user")
            sys.exit(1)
        return rec


# ----------------------------------------------------------------------------- files and states


def state_table() -> dict[str, tuple[str, str]]:
    """Each scored or reported state's (key in `files`, task). The seed is the name's last character."""
    out = {f"{t}_s{s}": (f"{'e54' if s < 2 else 'e56'}/{t}_s{s}", t) for t in TT.E54_TASKS for s in range(5)}
    out |= {f"cipher_own_s{s}": (f"e54/cipher_own_s{s}", "cipher") for s in TT.SEEDS}
    out |= {f"{t}_s{s}": (f"e46/{t}_s{s}", t) for t in TT.E46_TASKS for s in TT.SEEDS}
    out |= {f"{t}_s{s}": (f"e53/{t}_s{s}", t) for t in ("lowercase", "broad_cipher") for s in TT.SEEDS}
    return out | {f"e53_cipher_s{s}": (f"e53/cipher_s{s}", "e53_cipher") for s in TT.SEEDS}  # reported only


def files() -> Files:
    """The truth table's files (E54, E46, E53 at seeds 0, 1) plus the rest of the plan's table, each with the sha256 its
    source run recorded."""
    e46, e54 = TT.E46, TT.E54
    s0a, s2, j2 = TT.load(e46 / "stage0a.json")["groups"], TT.load(e54 / "stage2.json")["finetunes"], TT.load(E56 / "job2.json")["finetunes"]
    out = TT.files()
    out |= {f"e46/windows_{t}": (e46 / "large" / f"windows_{t}.pt", s0a["e12/build"]["sha256"][t]) for t in TT.E46_TASKS}
    out["e54/windows_cipher"] = (e54 / "large" / "windows_cipher.pt", TT.load(e54 / "stage1.json")["sha256"]["cipher"])
    out |= {f"e54/cipher_own_s{s}": (e54 / "large" / f"cipher_own_s{s}.pt", s2[f"cipher_own_s{s}"]["sha256"]) for s in TT.SEEDS}
    out |= {f"e56/{t}_s{s}": (E56 / "large" / f"{t}_s{s}.pt", j2[f"{t}_s{s}"]["sha256"]) for t in TT.E54_TASKS for s in (2, 3, 4)}
    out["e36/code_step10"] = (e46 / "large" / "code_step10.pt", s0a["e36/code"]["states"]["10"]["sha256"])
    return out


def load_tasks(net: ST.Net, pa: ModuleType, F: Files) -> tuple[dict[str, ST.Task], dict[str, Any]]:
    """The KV tasks (`steering.load_task`, W0 read whole, as E56) and E53's Pile tasks with their next-token target (their
    lp0 is left empty: `gains` and `window_curve` read W0 in the same batches); and the KV windows files, for W1g."""
    kv = {"elicit": "e54/windows_elicit", "cipher": "e54/windows_cipher", **{t: f"e46/windows_{t}" for t in TT.E46_TASKS}}
    tasks: dict[str, ST.Task] = {n: ST.load_task(net, F[k][0], n, n) for n, k in kv.items()}
    run = pa.Run.__new__(pa.Run)  # `scored` reads no attribute; Run's __init__ would load a second model and split context
    for n, w in TT.e53_tasks(pa, run).items():
        name = "e53_cipher" if n == "cipher" else n
        tasks[name] = TT.NextTask(name, "pile", w.held, torch.empty(0), (), -1, pa.CFG.half, torch.empty(0))
    return tasks, {n: torch.load(F[k][0], weights_only=False) for n, k in kv.items()}


# ----------------------------------------------------------------------------- the window readings


def gains(net: ST.Net, t: ST.Task, dW: ST.DW) -> dict[str, float]:
    """`steering.gains` in batches of BATCH, W0 and the state read on the same batches."""
    d = torch.cat([ST.read(net, t, b, dW) - ST.read(net, t, b) for b in t.toks.split(BATCH)])
    return {"G": float(d.mean()), "G_score": float(d[t.part("score")].mean())}


def window_curve(net: ST.Net, t: ST.Task, dW: ST.DW, grid: list[int], G_score: float, rot: list[Tensor], gauss: ST.DW, order: Tensor) -> dict[str, Any]:
    """On the scoring half, per k: the raw rise in nats and the share of the per-window patch, and of the patch of `order`'s
    top k on every window (C1's reference); A1, its rotated value and its value on the update `gauss` (fractions of 12288)."""
    base: list[Tensor] = []
    lp: dict[int, list[Tensor]] = {k: [] for k in grid}
    lp_rand: dict[int, list[Tensor]] = {k: [] for k in grid}
    pr: list[Tensor] = []
    pr_rot: list[Tensor] = []
    pr_gauss: list[Tensor] = []
    for b in t.toks[t.part("score")].split(BATCH):
        logits, c0 = net.run(b, capture=True)
        base.append(ST.logp_at(logits, t.half, t.tgt(b)))
        a0 = NE.flat_mlp(c0, t.half)
        del logits, c0
        with net.at(gauss):
            pr_gauss.append(NS.participation((NE.flat_mlp(net.run(b, capture=True)[1], t.half) - a0).double().pow(2)))
        with net.at(dW):
            _, c1 = net.run(b, capture=True)
        shift = (NE.flat_mlp(c1, t.half) - a0).double()
        pr.append(NS.participation(shift.pow(2)))
        pr_rot.append(NS.participation(NS.rotate_shift(shift, rot).pow(2)))
        ranks = NS.window_ranks(shift)
        for k in grid:
            lp[k].append(ST.read(net, t, b, edit=NE.window_edit(c1, ranks < k)))
            lp_rand[k].append(ST.read(net, t, b, edit=NE.window_edit(c1, (order < k).expand(len(b), -1))))
    l0 = torch.cat(base)

    def shares(x: dict[int, list[Tensor]]) -> dict[int, float]:
        return {k: EL.share_or_nan(ST.share(torch.cat(x[k]), l0, G_score), G_score) for k in grid}

    return {"rises": {k: float((torch.cat(lp[k]) - l0).mean()) for k in grid}, "shares": shares(lp), "shares_random": shares(lp_rand),
            "A1": NS.median(torch.cat(pr)) / FULL, "A1_rot": NS.median(torch.cat(pr_rot)) / FULL, "A1_gauss": NS.median(torch.cat(pr_gauss)) / FULL}


def window_row(net: ST.Net, t: ST.Task, dW: ST.DW, grid: list[int], floor: float, rot: list[Tensor], seed: int, order_seed: int) -> dict[str, Any]:
    """All the window readings of one state; `seed` draws its Gaussian (as stage 1's), `order_seed` its random order."""
    g = gains(net, t, dW)
    cur = window_curve(net, t, dW, grid, g["G_score"], rot, full_gaussian(net, dW, seed), NS.random_ranks(order_seed))
    return {**g, **cur, **NS.c1_reading(cur["shares"], g["G"], g["G_score"], floor),
            "C1_random": NS.c1_reading(cur["shares_random"], g["G"], g["G_score"], floor)["C1"]}


def identity_patch(net: ST.Net, t: ST.Task) -> float:
    """W0 with every neuron of each window replaced by W0's own activations through `window_edit`: the largest |change of
    log P| in nats, against W0 read in the same batch (E56's 1.1e-5 was two batch shapes, one batch's W0 and the other's)."""
    worst = 0.0
    for b in t.toks[t.part("score")].split(BATCH):
        logits, c0 = net.run(b, capture=True)
        base = ST.logp_at(logits, t.half, t.tgt(b))
        edit = NE.window_edit(c0, torch.ones(len(b), FULL, dtype=torch.bool))
        worst = max(worst, float((ST.read(net, t, b, edit=edit) - base).abs().max()))
    return worst


# ----------------------------------------------------------------------------- the weight readings


def parts(dW: ST.DW, seed: int) -> tuple[ST.DW, ST.DW]:
    """The eight MLP matrices on the GPU, and a matched-norm Gaussian per matrix (E51's null, `finetune.random_direction`)."""
    mlp = {k: v.float().cpu() for k, v in NS.mlp_part(dW).items()}
    gauss = FT.random_direction(mlp, seed)
    return {k: v.to(DEVICE) for k, v in mlp.items()}, {k: v.to(DEVICE) for k, v in gauss.items()}


def full_gaussian(net: ST.Net, dW: ST.DW, seed: int) -> ST.DW:
    """The Gaussian of `parts` as a whole update: the eight MLP matrices, the other 16 at zero (`Net.at` needs all 24)."""
    return {**{k: torch.zeros_like(w) for k, w in net.W0.items()}, **parts(dW, seed)[1]}


def state_row(net: ST.Net, dW: ST.DW, seed: int, rot: list[Tensor]) -> dict[str, Any]:
    mlp, gauss = parts(dW, seed)
    a, b = NS.w2(mlp, net.W0), NS.w2(gauss, net.W0)
    return {"W1": NS.w1(mlp), "W1_rot": NS.w1(mlp, rot), "W1_gauss": NS.w1(gauss),
            "W2": a["all"], "W2_rows": a["rows"], "W2_cols": a["cols"],
            "W2_gauss": b["all"], "W2_gauss_rows": b["rows"], "W2_gauss_cols": b["cols"], **NS.block_readings(mlp)}


# ----------------------------------------------------------------------------- the first-batch gradient


class Fp32(nn.Module):
    """The target net with autocast off: `finetune`'s losses run their net inside bf16 autocast; E63 reads fp32 (E56)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, toks: Tensor) -> Any:
        with torch.autocast("cuda", enabled=False):
            return self.net(toks)


def kv_toks(w: dict[str, Any], seed: int) -> Tensor:
    """E12's and E54's arm A at this seed (`finetune_arm_a`): the first draw of its generator, the same windows."""
    batch = FT.ArmAConfig(o_token=w["table"]).batch
    return w["train"].tokens[torch.randint(0, len(w["train"]), (batch,), generator=torch.Generator().manual_seed(seed))].to(DEVICE)


def kv_batch(w: dict[str, Any], seed: int) -> LossOf:
    cfg, toks = FT.ArmAConfig(o_token=w["table"], seed=seed), kv_toks(w, seed)
    return lambda fp: FT.arm_a_loss(fp, fp, toks, cfg, w["train"].half)


def kv_shuffled(w: dict[str, Any], seed: int, shuffle_seed: int) -> LossOf:
    """W1g's reference: arm A's first batch with its targets permuted among the windows. Cross-entropy alone: arm A's KL has
    zero gradient at W0."""
    toks, half = kv_toks(w, seed), w["train"].half
    target = target_tokens(w["table"], toks, half)
    target = target[torch.randperm(len(target), generator=torch.Generator().manual_seed(shuffle_seed)).to(DEVICE)]
    return lambda fp: (NF.cross_entropy(FT.logits_of(fp, toks)[:, half], target), {})


def pile_train(pa: ModuleType, task: str) -> Tensor:
    """E53's training rows of a Pile task, in batch order 0 (`stage_finetune`, `finetune_members`, `lowercase.task`)."""
    from commute import vpd

    rows = pile_rows("train", 0, 4096)
    if task == "lowercase":
        return lowercase_rows(rows, vpd.tokenizer(), LC.STEPS * LC.BATCH)
    pi = pa.e53_pi() if task == "e53_cipher" else pa.a1_maps()[task]
    return pa.task_text(task, rows[: pa.CFG.n_train], pi)[0]


def pile_toks(rows: Tensor, seed: int) -> Tensor:
    """E53's arm B at this batch order (`lowercase.arm_b_states`): the order's first LC.BATCH rows."""
    return LC.batch_order(rows, seed)[: LC.BATCH].to(DEVICE)


def pile_batch(rows: Tensor, seed: int) -> LossOf:
    toks = pile_toks(rows, seed)
    return lambda fp: FT.next_token_ce(fp, toks)


def pile_shuffled(rows: Tensor, seed: int, shuffle_seed: int) -> LossOf:
    """W1g's reference: arm B's first batch with its next tokens permuted among all its positions."""
    toks = pile_toks(rows, seed)
    y = toks[:, 1:].reshape(-1)
    y = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(shuffle_seed)).to(DEVICE)]

    def loss(fp: nn.Module) -> tuple[Tensor, dict[str, float]]:
        lg = FT.logits_of(fp, toks)[:, :-1]
        return NF.cross_entropy(lg.reshape(-1, lg.shape[-1]), y), {}

    return loss


def gradient(net: ST.Net, loss_of: LossOf) -> tuple[ST.DW, dict[str, float]]:
    """The loss's gradient at W0 on the 24 matrices, fp32. Clipping does not change W1g; KL(ref || W0) has zero gradient
    at W0, so on arm A this is the cross-entropy's."""
    params = FT.decomposed_params(net.model)
    for p in params.values():
        p.requires_grad_(True)
    try:
        loss, info = loss_of(Fp32(net.net))
        grads = torch.autograd.grad(loss, list(params.values()))
    finally:
        for p in params.values():
            p.requires_grad_(False)
    return {k: g.detach() for k, g in zip(params, grads, strict=True)}, {"loss": float(loss), **info}


def gradient_rows(net: ST.Net, pa: ModuleType, table: dict[str, tuple[str, str]], ws: dict[str, Any], rot: list[Tensor], P: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """W1g per task and seed, rows keyed `<task>_s<seed>`: one first batch per pair, so cipher_own_s<s> (same task, seed
    and W0) reads cipher_s<s>'s gradient. The batch is checked first: under bf16 autocast, as the fine-tune ran, its loss
    must match the fine-tune's logged step 1; a miss stops the job, since a wrong batch makes W1g meaningless."""
    S = P["settings"]
    o, rows_of, out = Out("gradients.json"), {}, {}
    for i, pair in enumerate(dict.fromkeys(f"{t}_s{n[-1]}" for n, (_, t) in table.items())):
        t, s = pair[:-3], int(pair[-1])
        if t not in ws and t not in rows_of:
            rows_of[t] = pile_train(pa, t)
        loss_of = kv_batch(ws[t], s) if t in ws else pile_batch(rows_of[t], s)
        shuf = S["shuffle_seed"] + i
        gr, info = gradient(net, loss_of)
        gr_shuf, _ = gradient(net, kv_shuffled(ws[t], s, shuf) if t in ws else pile_shuffled(rows_of[t], s, shuf))
        with torch.no_grad():
            bf16 = float(loss_of(net.net)[0])
        mlp, gauss = parts(gr, S["gaussian_seed"] + 1000 + i)
        ref = P["first_step_loss_bf16"][pair]
        out[pair] = o.put(pair, {"W1g": NS.w1(mlp), "W1g_rot": NS.w1(mlp, rot), "W1g_gauss": NS.w1(gauss), "W1g_shuf": NS.w1(parts(gr_shuf, 0)[0]),
                                 **info, "loss_bf16": bf16, "first_step_loss_bf16": ref, "fp32_minus_bf16": info["loss"] - bf16},
                          abs(bf16 - ref) <= P["first_step_loss_stop_tol"])
    return out


# ----------------------------------------------------------------------------- stage 0


def synthetic(net: ST.Net, t: ST.Task, rot: list[Tensor], c: dict[str, Any], o: Out) -> ST.DW:
    """W1, W2 and the rotated reference on hand-made updates: the 16-neuron update on block 3's neurons most active at the
    answer position (fit half), a unit Gaussian, and 0. Returns the 16-neuron update."""
    ortho = max(float((Q.T @ Q - torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)).abs().max()) for Q in rot)
    a0 = net.run(t.toks[t.part("fit")], capture=True)[1]["mlp3"][:, t.half]
    neurons = torch.sort(a0.abs().mean(0).cpu(), descending=True, stable=True).indices[:16]
    six = NS.sixteen_update(net.W0, 3, neurons)
    g = torch.Generator().manual_seed(c["gaussian_seed"])
    unit = {k: torch.randn(v.shape, generator=g, dtype=torch.float64).to(DEVICE) for k, v in NS.mlp_part(net.W0).items()}
    zero = {k: torch.zeros_like(v) for k, v in NS.mlp_part(net.W0).items()}
    s2, u2, z2 = NS.w2(six, net.W0), NS.w2(unit, net.W0), NS.w2(zero, net.W0)
    rec = {"orthogonality": ortho, "neurons": neurons.tolist(), "w1_sixteen": NS.w1(six), "w1_sixteen_rot": NS.w1(six, rot),
           "w2_sixteen": s2["all"], "w1_gaussian": NS.w1(unit), "w1_gaussian_rot": NS.w1(unit, rot), "w2_gaussian_rows": u2["rows"],
           "w2_gaussian_cols": u2["cols"], "w1_zero": NS.w1(zero), "w2_zero": z2["all"], "a1_zero": float(NS.participation(torch.zeros(4)))}
    lo, hi = c["gaussian_w1_range"]
    rlo, rhi = c["rotated_sixteen_w1_range"]
    o.put("synthetic", rec, rec["orthogonality"] <= c["orthogonality_tol"]
          and abs(rec["w1_sixteen"] * FULL / 16 - 1) <= c["w1_sixteen_rel_tol"] and abs(rec["w2_sixteen"] - 1) <= c["w2_sixteen_tol"]
          and rlo <= rec["w1_sixteen_rot"] <= rhi and lo <= rec["w1_gaussian"] <= hi
          and abs(rec["w2_gaussian_rows"] * 768 - 1) <= c["gaussian_w2_rows_rel_tol"]
          and all(math.isnan(rec[k]) for k in ("w1_zero", "w2_zero", "a1_zero")))
    return six


def stage0(net: ST.Net, tasks: dict[str, ST.Task], ws: dict[str, Any], F: Files, rot: list[Tensor], P: dict[str, Any], o: Out) -> None:
    c, S = P["stage0"], P["settings"]
    six = synthetic(net, tasks["elicit"], rot, c, o)
    for n, t in tasks.items():
        m = identity_patch(net, t)
        o.put(f"identity/{n}", {"max_abs": m}, m <= c["identity_tol"])
    r = window_row(net, tasks["elicit"], six, S["k_grid"], S["floor"], rot, S["gaussian_seed"], S["random_rank_seed"])
    if not r["readable"]:
        log(f"SURPRISE: the 16-neuron update is unreadable (G {r['G']:.4f}, full share {r['full_share']:.4f}): C1's control is not exercised")
    o.put("sixteen", r, r["A1"] <= c["a1_sixteen_max"] * (1 + 1e-9) and (not r["readable"] or r["C1"] <= c["c1_sixteen_max"])
          and abs(r["rises"][16] - r["rises"][FULL]) <= c["sixteen_rise_tol"])
    job1 = TT.load(E56 / "job1.json")["states"]
    for n in ("elicit_s0", "cipher_s0"):
        t, dW = tasks[n.split("_")[0]], TT.state(F, f"e54/{n}")
        G_score = ST.gains(net, t, dW)["G_score"]  # whole, as E56
        p = NE.neuron_patch(net, t, dW, G_score, [16, FULL])
        own = window_curve(net, t, dW, [FULL], G_score, rot, full_gaussian(net, dW, S["gaussian_seed"]), NS.random_ranks(S["random_rank_seed"]))["shares"][FULL]
        ref = job1[n]
        N16 = EL.share_or_nan(p["shares"][16], G_score)
        o.put(f"e56/{n}", {"N16": N16, "N16_ref": ref["N16"], "window_full_share": own, "full_share_ref": ref["b2_shares"][str(FULL)],
                           "mean_full_share": p["shares"][FULL], "top16_same": p["order"][:16].tolist() == ref["top16"]},
              abs(N16 - ref["N16"]) <= c["n16_tol"] and abs(own - ref["b2_shares"][str(FULL)]) <= c["full_width_tol"])
    gr, info = gradient(net, kv_batch(ws["elicit"], 0))
    norm = float(sum(v.double().pow(2).sum() for v in gr.values()) ** 0.5)
    o.put("gradient/elicit_s0", {**info, "norm": norm}, math.isfinite(norm) and norm > 0 and abs(info["kl_other"]) <= c["kl_at_w0_tol"])


# ----------------------------------------------------------------------------- the score


def score(rows: dict[str, dict[str, Any]], P: dict[str, Any]) -> dict[str, Any]:
    """Each number of predictions.json 'number_rules' through `neuron_sparsity.score_number`, and the tree's branch."""
    out: dict[str, Any] = {}
    for num, spec in P["number_rules"].items():
        rec = out[num] = NS.score_number(rows, num, spec, P["tests"], P["reported_beside"])
        marks = [f"{k} {rec[k]['status']}" for k in P["tests"]] + [f"guard/{r} {g['status']}" for r, g in rec["guard"].items()]
        log(f"[score {num}] answers_yes {rec['answers_yes']} {' '.join(marks)} {json.dumps(clean(rec))}")
    yes = [n for n, r in out.items() if r["answers_yes"]]
    out["branch"] = (f"{yes} answer yes: read on E58 A6-A7's equal-prior cipher and shuffled map, as an amendment" if yes else
                     "none answers yes: write up; an elicit/new gap under T1 only is reported as a progress effect")
    log(f"[branch] {out['branch']}")
    return out


# ----------------------------------------------------------------------------- main


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    rel = str(PRED.relative_to(REPO))
    if git("ls-files", "--error-unmatch", rel).returncode or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: commit it first")
    P = json.loads(PRED.read_text())
    S = P["settings"]
    assert S["k_grid"] == [2**i for i in range(14)] + [FULL] and S["batch"] == BATCH
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    F, o = files(), Out("stage0.json")
    table = state_table()
    got = {k: KD.sha256(p) if p.is_file() else "missing" for k, (p, _) in F.items()}
    o.put("sha256", {"n_files": len(F), "mismatch": sorted(k for k, (_, h) in F.items() if got[k] != h)}, all(got[k] == h for k, (_, h) in F.items()))

    net = ST.Net(DEVICE)
    pa = TT.performed_anchor()
    tasks, ws = load_tasks(net, pa, F)
    rot = [Q.to(DEVICE) for Q in NS.rotations(S["rotation_seed"])]
    stage0(net, tasks, ws, F, rot, P, o)
    o.out["pass"] = True
    o.dump()
    log("[stage0] pass")

    weights = Out("weights.json")
    rows: dict[str, dict[str, Any]] = {}
    for i, (n, (key, _)) in enumerate(table.items()):
        rows[n] = dict(weights.put(n, state_row(net, TT.state(F, key), S["gaussian_seed"] + i, rot)))
    code = {k: v.float() for k, v in torch.load(F["e36/code_step10"][0], weights_only=True).items()}
    weights.put("code_step10", state_row(net, code, S["gaussian_seed"] + len(table), rot))

    grads = gradient_rows(net, pa, table, ws, rot, P)
    for n, (_, t) in table.items():
        rows[n] |= {k: grads[f"{t}_s{n[-1]}"][k] for k in ("W1g", "W1g_rot", "W1g_gauss", "W1g_shuf")}

    win = Out("windows.json")
    for i, (n, (key, t)) in enumerate(table.items()):
        rows[n] |= win.put(n, window_row(net, tasks[t], TT.state(F, key), S["k_grid"], S["floor"], rot, S["gaussian_seed"] + i, S["random_rank_seed"] + i))

    res = Out("score.json")
    res.out["rows"] = score(rows, P)
    res.out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res.dump()
    log(f"[done] peak GPU {res.out['peak_gpu_gb']:.1f} GB")


if __name__ == "__main__":
    main()
