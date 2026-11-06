"""E112 (case_label), 2026-11-06: the case-label pair read under Jain et al. 2024's definition as adopted (`decisions.md`
rows 81 and 87, plus the direction-removal check; nothing added). Plan: reports/plans/2026-11-06_plan_e112-case-label.md.
Rules: results/predictions.json (the run refuses unless it is committed unchanged); pure rules in src/commute/case_label.py.
One cluster debug L40 job per stage, cheapest first, results written after every step to results/stage<N>.json:

  0  W0 alone: the pool, labels and windows; the possession probe and its Hewitt-Liang control at every block; P0 and l*;
     W0 under the probe direction's and a random direction's removal at l*. Exit 1 if P0 fails ("unreadable").
  1  The fine-tunes (real and twin, seeds 0 and 1; E62's arm-A recipe), each stopped at its scored step; delta-W in bf16
     to results/large/ at once. A state already in stage1.json is skipped; exit 3 when too little time is left for the next.
  2  The do-nothing direction on every state first (exit 1 if one misses), then P1-P3 and the verdict per state; pruning
     and reverse fine-tuning beside, no rule, reverse last under the stage's time check.

    sbatch cluster/case_label.sbatch <N>
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from commute import anchors as AN
from commute import case_label as CL
from commute import keydep as KD
from commute import kv, ladder
from commute import readout as RO
from commute import removal as RM
from commute import revival_kv as RK
from commute import steering as ST
from commute.copying import with_target
from commute.data import LAST, NEXT, Windows, pile_rows
from commute.elicit import summed_norm
from commute.finetune import ArmAConfig, StopTraining, finetune_arm_a
from commute.jobs import hard_exit, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
HALF, L = AN.HALF, 2 * AN.HALF + 1  # word at HALF - 1, trigger at HALF; the model sees L tokens (ci_len), label after them
TASKS = ("real", "twin")
log = KD.log


@dataclass(frozen=True)
class Config:
    """Every size and seed, read from predictions.json's settings."""

    probe_rows: tuple[str, int, int]
    n_pairs: int
    n_train_pairs: int
    pair_seed: int
    label_candidates: tuple[str, ...]
    label_seed: int
    twin_seed: int
    control_seed: int
    n_train_windows: int
    n_fit_windows: int
    n_heldout_windows: int
    window_seed: int
    probe_cap: int
    probe_seed: int
    ridge_lambda: float
    seeds: tuple[int, ...]
    checkpoints: tuple[int, ...]
    learn_bar: float
    batch: int
    kl_lambda: float
    grad_clip: float
    random_dir_seed: int
    prune_k: tuple[int, ...]
    prune_random: int
    prune_seed: int
    reverse_cap: int
    reverse_every: int
    reverse_bar: float
    reverse_seed: int
    stage_minutes: float
    finetune_minutes: float

    def __post_init__(self) -> None:
        assert 0 < self.n_train_pairs < self.n_pairs and self.n_fit_windows % 2 == 0
        assert list(self.checkpoints) == sorted(set(self.checkpoints)) and self.reverse_cap % self.reverse_every == 0
        assert self.probe_rows[1] >= 32768, "probe rows must be disjoint from the task windows' rows (keydep.vocab: 0-32767)"


def dump(n: int, res: dict[str, Any]) -> None:
    (RESULTS / f"stage{n}.json").write_text(json.dumps(jsonable(res), indent=1))


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


# ----------------------------------------------------------------------------- model-side reads


def windows(rows: Tensor, marker: int, words: list[int], n: int, seed: int) -> Tensor:
    """`kv.windows` with its two written columns swapped: the word at HALF - 1, the trigger at HALF. (n, L) tokens."""
    t = kv.windows(rows, marker, tuple(words), HALF, n, seed).tokens
    t[:, [HALF - 1, HALF]] = t[:, [HALF, HALF - 1]]
    return t


def wins(toks: Tensor) -> Windows:
    n = len(toks)
    return Windows(toks, torch.zeros(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), HALF)


@torch.no_grad()
def resid_at(net: ST.Net, toks: Tensor, batch: int = 64) -> Tensor:
    """(4, N, d) fp32 CPU: block outputs at HALF at the current weights."""
    return torch.cat([RO.resid(net, toks[s : s + batch, :L])[1][:, :, HALF].cpu() for s in range(0, len(toks), batch)], 1)


@torch.no_grad()
def label_diff(net: ST.Net, toks: Tensor, ab: tuple[int, int], dW: ST.DW | None = None, edit: ST.Edit | None = None) -> Tensor:
    """(N,) float64: logit A - logit B at HALF."""
    out = []
    with net.at(dW):
        for s in range(0, len(toks), 64):
            lg = net.run(toks[s : s + 64, :L], edit)[0][:, HALF]
            out.append((lg[:, ab[0]] - lg[:, ab[1]]).double().cpu())
    return torch.cat(out)


def accuracy(d: Tensor, y: Tensor) -> float:
    return float(((d > 0).long() == y).double().mean())


def margin(d: Tensor, y: Tensor) -> Tensor:
    return d * (2 * y - 1)


def removal_edit(block: int, u: Tensor) -> ST.Edit:
    """Remove the unit direction u (d, 1) from block `block`'s output residual at HALF only."""

    def f(x: Tensor) -> Tensor:
        y = x.clone()
        y[:, HALF] = RM.remove(x[:, HALF], u.to(x.device, x.dtype))
        return y

    return {RO.RESID[block]: f}


@torch.no_grad()
def next_ce(net: ST.Net, toks: Tensor, edit: ST.Edit | None = None) -> float:
    """W0's mean next-token loss at HALF (the true token at HALF + 1)."""
    ce = [F.cross_entropy(net.run(toks[s : s + 64], edit)[0][:, HALF], toks[s : s + 64, HALF + 1].to(net.device), reduction="sum")
          for s in range(0, len(toks), 64)]  # fmt: skip
    return float(torch.stack(ce).sum()) / len(toks)


# ----------------------------------------------------------------------------- stage 0


def build(net: ST.Net, C: Config) -> dict[str, Any]:
    """The pool, labels and windows (results/large/task.pt), from keydep.vocab's rows and marker."""
    rows, _plain, marker, _ranked, tok = KD.vocab(KD.Ctx(net, {}, {}, {}, LARGE))
    counts, V = ladder.count_tokens(rows, HALF), int(net.net.lm_head.weight.shape[0])
    cands = [i[0] for i in (tok.encode(c) for c in C.label_candidates) if len(i) == 1]
    pool = CL.top_pairs(CL.case_pairs(tok.decode, V), counts, C.n_pairs, {marker, *cands})
    train, held = CL.split_pairs(pool, C.n_train_pairs, C.pair_seed)
    a, b = CL.pick_labels(cands, {t for p in pool for t in p}, C.label_seed)
    tw, hw = [t for p in train for t in p], [t for p in held for t in p]
    keep = torch.arange(rows.shape[0]) % 5
    s = C.window_seed
    t = {"marker": marker, "ab": (a, b), "train_pairs": train, "held_pairs": held, "vocab": V,
         "y": {"real": CL.real_y(pool), "twin": {**CL.twin_y(tw, C.twin_seed), **CL.twin_y(hw, C.twin_seed + 1)}},
         "train": windows(rows[keep != 0], marker, tw, C.n_train_windows, s),
         "fit": windows(rows[keep == 0], marker, tw, C.n_fit_windows, s + 1),
         "held": windows(rows[keep == 0], marker, hw, C.n_heldout_windows, s + 2)}  # fmt: skip
    torch.save(t, LARGE / "task.pt")
    words = {k: [tok.decode([w]) for w in v] for k, v in (("labels", (a, b)), ("held_first10", hw[:10]))}
    say("task", {"marker": tok.decode([marker]), **words, "pool_rarer_count_last": min(counts[pool[-1][0]], counts[pool[-1][1]])})
    return t


def ys(t: dict[str, Any], task: str, part: str) -> Tensor:
    """(N,) the label y of each window of a part under a task ("real", "twin" or "control")."""
    return torch.tensor([t["y"][task][int(w)] for w in t[part][:, HALF - 1]])


def probes(net: ST.Net, C: Config, t: dict[str, Any]) -> dict[str, Any]:
    """The possession probe and its control per block: ridge on Pile positions after a train-pair word, scored on the
    held-out-pair task windows at the trigger."""
    prows = pile_rows(*C.probe_rows, log=log)
    r, pos = CL.probe_positions(prows, t["train_pairs"], HALF, C.probe_cap, C.probe_seed)
    ptoks = prows[r[:, None], pos[:, None] - HALF + torch.arange(L)]
    ctrl = CL.control_y(t["train_pairs"] + t["held_pairs"], C.control_seed)
    prev = ptoks[:, HALF - 1].tolist()
    y = {"case": torch.tensor([t["y"]["real"][w] for w in prev]), "control": torch.tensor([ctrl[w] for w in prev])}
    yh = {"case": ys(t, "real", "held"), "control": torch.tensor([ctrl[int(w)] for w in t["held"][:, HALF - 1]])}
    h, hh = resid_at(net, ptoks), resid_at(net, t["held"])
    out: dict[str, Any] = {"n_pile": len(ptoks), "pile_case_share": float(y["case"].double().mean())}
    fitted: dict[str, Any] = {}
    for k in ("case", "control"):  # the loop moved to case_label.value_probes (E116, 2026-11-06); value 1's probe is the one saved
        W, B, out[f"acc_{k}"] = CL.value_probes(h, y[k], hh, yh[k], 2, C.ridge_lambda)
        fitted[k] = (W[:, 1], B[:, 1])
    out["selectivity"] = [a - c for a, c in zip(out["acc_case"], out["acc_control"], strict=True)]
    torch.save({"case": fitted["case"], "control": fitted["control"], "pile_tokens": ptoks}, LARGE / "probes.pt")
    return out


def w0_removals(net: ST.Net, C: Config, t: dict[str, Any], w: Tensor, block: int) -> dict[str, Any]:
    """W0 under each removal at `block`: next-token loss on the probe's Pile windows, label margin on the fit windows."""
    ptoks = torch.load(LARGE / "probes.pt", weights_only=False)["pile_tokens"]
    dirs = {"probe": RM.unit_basis(w), "random": RM.random_basis(w.shape[0], 1, [C.random_dir_seed])}
    ce0, out = next_ce(net, ptoks), {}
    d0 = label_diff(net, t["fit"], t["ab"])
    for name, u in dirs.items():
        e = removal_edit(block, u)
        de = label_diff(net, t["fit"], t["ab"], edit=e)
        out[name] = {"next_ce_change": next_ce(net, ptoks, e) - ce0,
                     **{f"margin_change_{k}": float((margin(de, ys(t, k, "fit")) - margin(d0, ys(t, k, "fit"))).mean()) for k in TASKS}}  # fmt: skip
    return {"next_ce_w0": ce0, **out}


def stage0(net: ST.Net, C: Config, R: dict[str, Any]) -> int:
    res: dict[str, Any] = {}
    t = build(net, C)
    res["w0_heldout_accuracy"] = {k: accuracy(label_diff(net, t["held"], t["ab"]), ys(t, k, "held")) for k in TASKS}
    say("W0 held-out label accuracy", res["w0_heldout_accuracy"])
    res["P0"] = probes(net, C, t)
    P = R["P0"]
    res["l_star"] = CL.first_passing_block(res["P0"]["acc_case"], res["P0"]["selectivity"], P["accuracy"], P["selectivity"])
    say("P0", {**res["P0"], "l_star": res["l_star"]})
    dump(0, res)
    if res["l_star"] is None:
        res["verdict"] = "unreadable"
        log("[stop] P0 fails at every block before the last: verdict 'unreadable'; stages 1 and 2 do not run")
        dump(0, res)
        return 1
    w = torch.load(LARGE / "probes.pt", weights_only=False)["case"][0][res["l_star"]]
    res["w0_removal"] = w0_removals(net, C, t, w, res["l_star"])
    say("W0 removal at l*", res["w0_removal"])
    dump(0, res)
    return 0


# ----------------------------------------------------------------------------- stage 1


def finetune(net: ST.Net, C: Config, t: dict[str, Any], task: str, seed: int) -> tuple[ST.DW, dict[str, Any]]:
    """E62's arm-A recipe on the task's windows, stopped at the first checkpoint whose train-word accuracy reaches the
    bar (else the last checkpoint): that step's delta-W and the record."""
    tab = CL.table(t["y"][task], t["ab"][0], t["ab"][1], t["vocab"])
    lab = lambda part: with_target(wins(t[part]), tab[t[part][:, HALF - 1]])
    yf, yh = ys(t, task, "fit"), ys(t, task, "held")
    rec: dict[str, Any] = {"acc_fit": {}, "acc_held": {}}
    got: dict[str, ST.DW] = {}

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        if step not in C.checkpoints:
            return
        rec["acc_fit"][step], rec["acc_held"][step] = accuracy(label_diff(net, t["fit"], t["ab"]), yf), accuracy(label_diff(net, t["held"], t["ab"]), yh)
        log(f"[{task} s{seed}] step {step}: train-word accuracy {rec['acc_fit'][step]:.4f}, held-out {rec['acc_held'][step]:.4f}")
        if rec["acc_fit"][step] >= C.learn_bar or step == C.checkpoints[-1]:
            got["dW"] = {k: (p.detach().float() - net.W0[k]).cpu() for k, p in params.items()}
            raise StopTraining

    cfg = ArmAConfig(o_token=LAST, kl_lambda=C.kl_lambda, lr=AN.FT_LR, batch=C.batch, steps=C.checkpoints[-1], checkpoints=C.checkpoints,
                     grad_clip=C.grad_clip, seed=seed, ci_len=L)  # fmt: skip
    try:
        finetune_arm_a(net.model, lab("train"), lab("fit"), cfg, net.device, log, on_step)
    except StopTraining:
        pass
    rec["step"], rec["learned"] = CL.scored_step(rec["acc_fit"], C.checkpoints, C.learn_bar)
    rec["norm"] = summed_norm(got["dW"])
    return got["dW"], rec


def stage1(net: ST.Net, C: Config, t0: float) -> int:
    s0 = json.loads((RESULTS / "stage0.json").read_text())
    assert s0.get("l_star") is not None, "stage 0 did not pass P0: stage 1 does not run"
    t = torch.load(LARGE / "task.pt", weights_only=False)
    p = RESULTS / "stage1.json"
    res: dict[str, Any] = json.loads(p.read_text()) if p.is_file() else {"states": {}}
    for task in TASKS:
        for seed in C.seeds:
            name = f"{task}_s{seed}"
            if name in res["states"]:
                continue
            if time.monotonic() - t0 > (C.stage_minutes - C.finetune_minutes) * 60:
                log(f"[time] too little left for {name}: exit 3, resubmit stage 1")
                return 3
            dW, rec = finetune(net, C, t, task, seed)
            path = LARGE / f"dW_{name}.pt"
            torch.save({k: v.bfloat16() for k, v in dW.items()}, path)
            res["states"][name] = {**rec, "path": str(path.relative_to(HERE)), "sha256": KD.sha256(path)}
            say(f"state {name}", res["states"][name])
            dump(1, res)
    return 0


# ----------------------------------------------------------------------------- stage 2


def removal_share(net: ST.Net, R: dict[str, Any], t: dict[str, Any], task: str, dW: ST.DW, l_star: int, u: Tensor) -> dict[str, float]:
    """The label log-odds gain on the train-word fit windows and the share of it the removal of u at l* takes away."""
    yf = ys(t, task, "fit")
    m0, ms = margin(label_diff(net, t["fit"], t["ab"]), yf), margin(label_diff(net, t["fit"], t["ab"], dW), yf)
    mr = margin(label_diff(net, t["fit"], t["ab"], dW, removal_edit(l_star, u)), yf)
    return {"gain": float((ms - m0).mean()), "share": CL.removed_share(ms, mr, m0, R["gain_floor"])}


def state_reads(net: ST.Net, R: dict[str, Any], t: dict[str, Any], task: str, dW: ST.DW, l_star: int, w: Tensor, b: Tensor,
                h0: dict[str, Tensor]) -> dict[str, Any]:  # fmt: skip
    """P1-P3 of one state. P2's g is a threshold on W0's readout score at l*, fitted to the state's labels on train words."""
    dh, df = label_diff(net, t["held"], t["ab"], dW), label_diff(net, t["fit"], t["ab"], dW)
    score = {k: ((h0[k][l_star].double() @ w) + b).numpy() for k in ("fit", "held")}
    theta, sign = CL.fit_threshold(score["fit"], (df > 0).long().numpy())
    p3 = removal_share(net, R, t, task, dW, l_star, RM.unit_basis(w))
    return {"P1": accuracy(dh, ys(t, task, "held")), "P2": float((CL.apply_threshold(score["held"], theta, sign) == (dh > 0).long().numpy()).mean()),
            "g": {"theta": theta, "sign": sign}, "gain": p3["gain"], "P3": p3["share"], "P3_no_gain": not p3["gain"] > R["gain_floor"]}  # fmt: skip


def prune(net: ST.Net, C: Config, t: dict[str, Any], task: str, dW: ST.DW) -> dict[str, Any]:
    """Beside, no rule: held-out label accuracy after zeroing the top-K MLP units by grad . w of the fit windows' KL to W0
    (Jain App. D Alg. 3, `revival_kv`), against K random units."""
    ref = RK.make_ref(net, t["fit"][:, :L], HALF)
    scores, yh, n = RK.unit_scores(net, ref, dW), ys(t, task, "held"), ST.N_LAYERS * ST.D_MLP
    acc = lambda units: accuracy(label_diff(net, t["held"], t["ab"], dW, RK.zero_units(units)), yh)
    out: dict[str, Any] = {"acc": accuracy(label_diff(net, t["held"], t["ab"], dW), yh)}
    for k in C.prune_k:
        rng = np.random.default_rng([C.prune_seed, k])
        out[str(k)] = {"top": acc(torch.topk(scores, k).indices.tolist()), "random": [acc(rng.choice(n, k, replace=False).tolist()) for _ in range(C.prune_random)]}
    return out


def reverse(net: ST.Net, C: Config, t: dict[str, Any], task: str, dW: ST.DW, deadline: float) -> dict[str, Any]:
    """Beside, no rule: the same recipe on the windows' original next tokens from the state; held-out label accuracy every
    `reverse_every` steps until it is <= the bar, the cap or the stage's deadline."""
    yh, curve, end = ys(t, task, "held"), [], "cap"

    def on_step(step: int, _p: Any) -> None:
        nonlocal end
        if step % C.reverse_every == 0:
            curve.append((step, accuracy(label_diff(net, t["held"], t["ab"]), yh)))
            end = "bar" if curve[-1][1] <= C.reverse_bar else "deadline" if time.monotonic() > deadline else end
            if end != "cap":
                raise StopTraining

    cfg = ArmAConfig(o_token=NEXT, kl_lambda=C.kl_lambda, lr=AN.FT_LR, batch=C.batch, steps=C.reverse_cap, checkpoints=(), grad_clip=C.grad_clip,
                     seed=C.reverse_seed, ci_len=L)  # fmt: skip
    w = wins(t["train"])
    with net.at(dW):
        curve.append((0, accuracy(label_diff(net, t["held"], t["ab"]), yh)))
        try:
            finetune_arm_a(net.model, w, w, cfg, net.device, lambda _m: None, on_step)
        except StopTraining:
            pass
    return {"curve": curve, "end": end, "steps_to_bar": CL.reverse_steps(curve, C.reverse_bar)}


def stage2(net: ST.Net, C: Config, R: dict[str, Any], t0: float) -> int:
    l_star = json.loads((RESULTS / "stage0.json").read_text())["l_star"]
    s1 = json.loads((RESULTS / "stage1.json").read_text())["states"]
    assert len(s1) == len(TASKS) * len(C.seeds), "stage 1 is not complete"
    t = torch.load(LARGE / "task.pt", weights_only=False)
    (W, B) = torch.load(LARGE / "probes.pt", weights_only=False)["case"]
    w, b = W[l_star], B[l_star]
    h0 = {k: resid_at(net, t[k]) for k in ("fit", "held")}
    dWs = {n: {k: v.float() for k, v in torch.load(HERE / s["path"], weights_only=False).items()} for n, s in s1.items()}
    res: dict[str, Any] = {"l_star": l_star, "do_nothing": {}, "states": {}}
    rand = RM.random_basis(w.shape[0], 1, [C.random_dir_seed])
    for name, dW in dWs.items():  # control first and alone: the random direction on every state
        res["do_nothing"][name] = removal_share(net, R, t, name.split("_")[0], dW, l_star, rand)
        say(f"do-nothing {name}", res["do_nothing"][name])
        dump(2, res)
    if not CL.do_nothing_ok([s["share"] for s in res["do_nothing"].values()], R["do_nothing_max"]):
        res["stop"] = f"the random direction's share is over {R['do_nothing_max']} on a state: no state is read"
        log(f"[stop] {res['stop']}")
        dump(2, res)
        return 1
    for name, dW in dWs.items():
        s = res["states"][name] = state_reads(net, R, t, name.split("_")[0], dW, l_star, w, b, h0)
        s["verdict"] = CL.verdict(l_star, s["P1"], s["P2"], s["P3"], R["real"])
        say(f"reads {name}", {**s, "step": s1[name]["step"], "learned": s1[name]["learned"], "norm": s1[name]["norm"]})
        say(f"verdict {name}", {"verdict": s["verdict"], "meets": CL.meets(s["P1"], s["P2"], s["P3"], R["real"])})
    dump(2, res)
    for kind, fn in (("pruning", lambda n: prune(net, C, t, n.split("_")[0], dWs[n])),
                     ("reverse", lambda n: reverse(net, C, t, n.split("_")[0], dWs[n], t0 + C.stage_minutes * 60))):  # fmt: skip
        for name in dWs:
            res.setdefault(kind, {})[name] = fn(name)
            say(f"{kind} {name}", res[kind][name])
            dump(2, res)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
    stage, t0 = ap.parse_args().stage, time.monotonic()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"].items()}
    C = Config(**settings)
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    code = stage0(net, C, P["rules"]) if stage == 0 else stage1(net, C, t0) if stage == 1 else stage2(net, C, P["rules"], t0)
    log(f"=== stage {stage} done, exit {code} ===")
    hard_exit(code)


if __name__ == "__main__":
    main()
