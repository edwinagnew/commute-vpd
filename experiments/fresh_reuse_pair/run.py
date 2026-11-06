"""E125 (fresh_reuse_pair), 2026-11-06: a fresh 67M pair built for both reuse readings. C = "the word before the trigger
already occurred earlier in the window"; real = label A iff it did, twin = A/B by a seeded shuffle per word. Read under
`decisions.md` rows 81 and 87 plus the direction-removal check (E112's code and bars, nothing added) and, on the same
states, by the decomposition side (stage2.py). Plan: reports/plans/2026-11-06_plan_e125-fresh-reuse-pair.md. Rules and
settings: results/predictions.json (the run refuses unless it is committed unchanged). One cluster debug L40 job per
stage, cheapest first, results written after every step to results/stage<N>.json, `STAGE n DONE` at the end:

  0  W0 alone: the pool, windows, the induction gap (positive control), the possession probe and its Hewitt-Liang
     control at every block, P0 and l*; W0 under the probe's and a random direction's removal. Exit 1 if P0 fails.
  1  The fine-tunes (real and twin, seeds 0 and 1; E112's recipe), delta-W in bf16 to results/large/ (a state already
     in stage1.json is skipped; exit 3 when too little time is left), then E112's stage-2 reads from the saved files:
     the random direction on every state first (exit 1 if one is over its bar), P1-P3 and the verdict per state;
     pruning and reverse fine-tuning beside.
  2  stage2.py: the decomposition side.

    sbatch cluster/fresh_reuse_pair.sbatch <N>
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import case_label as CL
from commute import definition_reads as DFR
from commute import keydep as KD
from commute import label_reads as LR
from commute import removal as RM
from commute import repeat_label as RL
from commute import steering as ST
from commute.anchors import HALF
from commute.data import LAST, pile_rows
from commute.finetune import ArmAConfig
from commute.jobs import hard_exit, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
L = 2 * HALF + 1  # word at HALF - 1, trigger at HALF; the model sees L tokens, the label rides after them (data.LAST)
TASKS, PARTS = ("real", "twin"), ("train", "fit", "held")
log = KD.log


@dataclass(frozen=True)
class Config:
    """Every size and seed, read from predictions.json's settings."""

    probe_rows: tuple[str, int, int]
    n_words: int
    n_train_words: int
    word_seed: int
    min_class_count: int
    max_rep_share: float
    label_candidates: tuple[str, ...]
    label_seed: int
    twin_seed: int
    control_seed: int
    n_windows: tuple[int, int, int]  # train, fit, held
    window_seed: int
    p_range: tuple[int, int]
    tries: int
    probe_cap: int
    probe_seed: int
    ridge_lambda: float
    seeds: tuple[int, ...]
    checkpoints: tuple[int, ...]
    learn_bar: float
    lr: float
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
    n_route: int
    n_student_eval: int
    n_shift: int
    student_lr: float
    student_steps: int
    student_batch: int
    switch_thr: float
    n_random_sets: int
    stage_minutes: float
    finetune_minutes: float
    student_minutes: float

    def __post_init__(self) -> None:
        assert 0 < self.n_train_words < self.n_words and 1 <= self.p_range[0] <= self.p_range[1] <= HALF - 3
        assert self.probe_cap <= self.min_class_count and all(n % 2 == 0 for n in self.n_windows)
        assert list(self.checkpoints) == sorted(set(self.checkpoints)) and self.reverse_cap % self.reverse_every == 0
        assert self.probe_rows[1] >= 32768, "probe rows must be disjoint from the task windows' rows (keydep.vocab: 0-32767)"
        assert self.n_route <= self.n_windows[1] and self.n_student_eval <= self.n_windows[1] and self.n_shift <= self.n_windows[0]


def load_config() -> tuple[Config, dict[str, Any]]:
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"].items()}
    return Config(**settings), P


def dump(n: int, res: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(res), indent=1))
    tmp.replace(RESULTS / f"stage{n}.json")


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


def task() -> dict[str, Any]:
    return torch.load(LARGE / "task.pt", weights_only=False)


def part(t: dict[str, Any], task_: str, p: str) -> tuple[Tensor, Tensor]:
    """(tokens, y) of a part under a task."""
    return t["toks"][p], t["y"][task_][p]


def states(C: Config) -> list[str]:
    return [f"{k}_s{s}" for k in TASKS for s in C.seeds]


# ----------------------------------------------------------------------------- stage 0


def build(net: ST.Net, C: Config) -> dict[str, Any]:
    """Pool, labels, windows and the probe's Pile windows (results/large/task.pt), from keydep.vocab's rows and marker."""
    rows, _plain, marker, _ranked, tok = KD.vocab(KD.Ctx(net, {}, {}, {}, LARGE))
    V = int(net.net.lm_head.weight.shape[0])
    prows = pile_rows(*C.probe_rows, log=log)
    prev, rep = RL.repeat_flags(prows, HALF)
    n_rep, n_not = RL.class_counts(prev, rep, V)
    cands = [i[0] for i in (tok.encode(c) for c in C.label_candidates) if len(i) == 1]
    ok = RL.qualifying(RL.word_tokens(tok.decode, V), n_rep, n_not, C.min_class_count, C.max_rep_share, {marker, *cands})
    words = RL.draw_words(ok, C.n_words, C.word_seed)  # seeded order: the first n_train_words train, the rest held out
    train, held = words[: C.n_train_words], words[C.n_train_words :]
    a, b = CL.pick_labels(cands, set(words), C.label_seed)
    keep = torch.arange(rows.shape[0]) % 5
    src = {"train": (rows[keep != 0], train), "fit": (rows[keep == 0], train), "held": (rows[keep == 0], held)}
    built = {p: RL.windows(r, marker, w, n, HALF, C.p_range, C.tries, C.window_seed + i)
             for i, (p, (r, w), n) in enumerate(zip(PARTS, src.values(), C.n_windows, strict=True))}  # fmt: skip
    for p, (toks, y, _) in built.items():
        assert bool((RL.earlier_count(toks, HALF) == y).all()), f"{p}: a window's earlier count is not its class"
    twin = {**CL.twin_y(train, C.twin_seed), **CL.twin_y(held, C.twin_seed + 1)}
    r, ts, yp = RL.probe_positions(prev, rep, train, HALF, C.probe_cap, C.probe_seed)
    t = {"marker": marker, "ab": (a, b), "train_words": train, "held_words": held, "vocab": V,
         "toks": {p: v[0] for p, v in built.items()}, "p": {p: v[2] for p, v in built.items()},
         "y": {"real": {p: v[1] for p, v in built.items()}, "twin": {p: RL.twin_labels(v[0], HALF, twin) for p, v in built.items()}},
         "probe": {"toks": prows[r[:, None], ts[:, None] - HALF + torch.arange(L)], "y": yp}}  # fmt: skip
    torch.save(t, LARGE / "task.pt")
    say("task", {"marker": tok.decode([marker]), "labels": [tok.decode([x]) for x in (a, b)], "held_first10": [tok.decode([w]) for w in held[:10]],
                 "n_qualifying": len(ok), "n_probe": len(yp),
                 "real_y_share": {p: float(v[1].double().mean()) for p, v in built.items()}})  # fmt: skip
    return t


@torch.no_grad()
def induction_gap(net: ST.Net, toks: Tensor, y: Tensor, p: Tensor) -> dict[str, float]:
    """Positive control at W0: log P(the token after position p) at the word's position HALF - 1, mean over repeated
    windows minus mean over not-repeated ones. Induction predicts that token only when the word occurred at p."""
    lp = torch.cat([torch.log_softmax(net.run(toks[s : s + 64, :L])[0][:, HALF - 1], -1).gather(1, toks[s : s + 64].gather(1, p[s : s + 64, None] + 1).to(net.device)).squeeze(1).cpu()
                    for s in range(0, len(toks), 64)]).double()  # fmt: skip
    return {"repeated": float(lp[y == 1].mean()), "not": float(lp[y == 0].mean()), "gap": float(lp[y == 1].mean() - lp[y == 0].mean())}


def probes(net: ST.Net, C: Config, t: dict[str, Any]) -> dict[str, Any]:
    """P0: the possession probe and its Hewitt-Liang control (one seeded random value per token type) per block, ridge on
    the Pile windows, scored on the held-out-word task windows at the trigger. Beside: the same probe at the word's own
    position (no rule)."""
    pt, yp = t["probe"]["toks"], t["probe"]["y"]
    held, yh = part(t, "real", "held")
    ctrl = CL.control_table(t["vocab"], 2, C.control_seed)
    out: dict[str, Any] = {"n_pile": len(pt), "pile_repeat_share": float(yp.double().mean())}
    h, hh = DFR.resid_at(net, pt, HALF), DFR.resid_at(net, held, HALF)
    W, B, out["acc_repeat"] = CL.value_probes(h, yp, hh, yh, 2, C.ridge_lambda)
    _, _, out["acc_control"] = CL.value_probes(h, ctrl[pt[:, HALF - 1]], hh, ctrl[held[:, HALF - 1]], 2, C.ridge_lambda)
    out["selectivity"] = [a - c for a, c in zip(out["acc_repeat"], out["acc_control"], strict=True)]
    torch.save({"w": W[:, 1].cpu(), "b": B[:, 1].cpu()}, LARGE / "probes.pt")
    _, _, out["beside_acc_at_word"] = CL.value_probes(DFR.resid_at(net, pt, HALF - 1), yp, DFR.resid_at(net, held, HALF - 1), yh, 2, C.ridge_lambda)
    return out


def w0_removals(net: ST.Net, C: Config, t: dict[str, Any], w: Tensor, block: int) -> dict[str, Any]:
    """Beside: W0 under each removal at `block`: next-token loss on the probe's Pile windows, label margin on fit windows."""
    pt, fit = t["probe"]["toks"][:, : HALF + 2], t["toks"]["fit"]  # next_ce reads the last column as the true next token
    dirs = {"probe": RM.unit_basis(w), "random": RM.random_basis(w.shape[0], 1, [C.random_dir_seed])}
    ce0, d0, out = DFR.next_ce(net, pt, HALF), DFR.label_diff(net, fit, t["ab"], HALF), {}
    for name, u in dirs.items():
        e = DFR.removal_edit(block, u, HALF)
        de = DFR.label_diff(net, fit, t["ab"], HALF, edit=e)
        out[name] = {"next_ce_change": DFR.next_ce(net, pt, HALF, e) - ce0,
                     **{f"margin_change_{k}": float((DFR.margin(de, t["y"][k]["fit"]) - DFR.margin(d0, t["y"][k]["fit"])).mean()) for k in TASKS}}  # fmt: skip
    return {"next_ce_w0": ce0, **out}


def stage0(net: ST.Net, C: Config, R: dict[str, Any]) -> int:
    res: dict[str, Any] = {}
    t = build(net, C)
    res["w0_heldout_accuracy"] = {k: DFR.accuracy(DFR.label_diff(net, t["toks"]["held"], t["ab"], HALF), t["y"][k]["held"]) for k in TASKS}
    res["induction_gap"] = {p: induction_gap(net, t["toks"][p], t["y"]["real"][p], t["p"][p]) for p in ("fit", "held")}
    say("W0 held-out label accuracy; induction gap", {**res["w0_heldout_accuracy"], **res["induction_gap"]})
    res["P0"] = probes(net, C, t)
    P = R["P0"]
    res["l_star"] = CL.first_passing_block(res["P0"]["acc_repeat"], res["P0"]["selectivity"], P["accuracy"], P["selectivity"])
    say("P0", {**res["P0"], "l_star": res["l_star"]})
    dump(0, res)
    if res["l_star"] is None:
        res["verdict"] = "unreadable"
        res["stopped"] = "possession fails: P0 misses at every block before the last; stages 1 and 2 do not run"
        log(f"[stop] {res['stopped']}")
        dump(0, res)
        return 1
    res["w0_removal"] = w0_removals(net, C, t, torch.load(LARGE / "probes.pt")["w"][res["l_star"]], res["l_star"])
    say("W0 removal at l*", res["w0_removal"])
    dump(0, res)
    return 0


# ----------------------------------------------------------------------------- stage 1


def finetunes(net: ST.Net, C: Config, t: dict[str, Any], res: dict[str, Any], t0: float) -> int:
    for name in states(C):
        if name in res["states"]:
            continue
        if time.monotonic() - t0 > (C.stage_minutes - C.finetune_minutes) * 60:
            log(f"[time] too little left for {name}: exit 3, resubmit stage 1")
            return 3
        k, seed = name.split("_s")[0], int(name.split("_s")[1])
        cfg = ArmAConfig(o_token=LAST, kl_lambda=C.kl_lambda, lr=C.lr, batch=C.batch, steps=C.checkpoints[-1], checkpoints=C.checkpoints,
                         grad_clip=C.grad_clip, seed=seed, ci_len=L)  # fmt: skip
        dW, rec = LR.finetune(net, part(t, k, "train"), part(t, k, "fit"), part(t, k, "held"), t["ab"], HALF, cfg, C.learn_bar, log)
        path = LARGE / f"dW_{name}.pt"
        torch.save({n: v.bfloat16() for n, v in dW.items()}, path)
        res["states"][name] = {**rec, "path": str(path.relative_to(HERE)), "sha256": KD.sha256(path)}
        say(f"state {name}", res["states"][name])
        dump(1, res)
    return 0


def load_dW(name: str, rec: dict[str, Any]) -> ST.DW:
    path = HERE / rec["path"]
    assert KD.sha256(path) == rec["sha256"], f"{path} changed since stage 1 wrote it"
    return {k: v.float() for k, v in torch.load(path, weights_only=False).items()}


def definition_reads(net: ST.Net, C: Config, R: dict[str, Any], t: dict[str, Any], res: dict[str, Any], l_star: int, deadline: float) -> int:
    """E112's stage 2 on the saved states: the random direction first and alone; P1-P3 and the verdict; beside reads."""
    pr = torch.load(LARGE / "probes.pt")
    w, b = pr["w"][l_star], pr["b"][l_star]
    dWs = {n: load_dW(n, res["states"][n]) for n in states(C)}
    rand = RM.random_basis(w.shape[0], 1, [C.random_dir_seed])
    for n, dW in dWs.items():
        fit, yf = part(t, n.split("_s")[0], "fit")
        res["do_nothing"][n] = DFR.removal_share(net, fit, t["ab"], yf, dW, l_star, rand, HALF, R["gain_floor"])
        say(f"do-nothing {n}", res["do_nothing"][n])
    dump(1, res)
    if not CL.do_nothing_ok([s["share"] for s in res["do_nothing"].values()], R["do_nothing_max"]):
        res["stopped"] = f"the random direction's share is over {R['do_nothing_max']} on a state: no state is read"
        log(f"[stop] {res['stopped']}")
        dump(1, res)
        return 1
    h0 = {p: DFR.resid_at(net, t["toks"][p], HALF) for p in ("fit", "held")}  # W0's block outputs at the trigger
    for n, dW in dWs.items():
        k = n.split("_s")[0]
        s = res["reads"][n] = DFR.state_reads(net, R, part(t, k, "fit"), part(t, k, "held"), t["ab"], dW, l_star, w, b, h0, HALF)
        s["verdict"] = CL.verdict(l_star, s["P1"], s["P2"], s["P3"], R["real"])
        say(f"reads {n}", {**s, "meets": CL.meets(s["P1"], s["P2"], s["P3"], R["real"]), "step": res["states"][n]["step"]})
    dump(1, res)
    log("DEFINITION VERDICTS " + json.dumps({n: r["verdict"] for n, r in res["reads"].items()}))
    for n, dW in dWs.items():
        k = n.split("_s")[0]
        res["pruning"][n] = DFR.prune(net, t["toks"]["fit"], part(t, k, "held"), t["ab"], dW, HALF, C.prune_k, C.prune_random, C.prune_seed)
        say(f"pruning {n}", res["pruning"][n])
        tr = t["toks"]["train"]  # E112's reverse: the windows' own next token after the trigger
        res["reverse"][n] = DFR.reverse(net, C, (tr, tr[:, HALF + 1]), part(t, k, "held"), t["ab"], dW, HALF, deadline)
        say(f"reverse {n}", res["reverse"][n])
        dump(1, res)
    return 0


def stage1(net: ST.Net, C: Config, R: dict[str, Any], t0: float) -> int:
    l_star = json.loads((RESULTS / "stage0.json").read_text()).get("l_star")
    assert l_star is not None, "stage 0 did not pass P0: stage 1 does not run"
    p = RESULTS / "stage1.json"
    res: dict[str, Any] = json.loads(p.read_text()) if p.is_file() else {"states": {}}
    assert "stopped" not in res, f"stage 1 stopped before ({res.get('stopped')}): move stage1.json aside to start again"
    t = task()
    code = finetunes(net, C, t, res, t0)
    if code:
        return code
    if time.monotonic() - t0 > (C.stage_minutes - C.finetune_minutes) * 60:
        log("[time] too little left for the reads: exit 3, resubmit stage 1 (the fine-tunes are kept)")
        return 3
    res.update(do_nothing={}, reads={}, pruning={}, reverse={})  # the reads are redone whole on a rerun
    return definition_reads(net, C, R, t, res, l_star, t0 + C.stage_minutes * 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(0, 1), required=True)
    stage, t0 = ap.parse_args().stage, time.monotonic()
    C, P = load_config()
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    code = stage0(net, C, P["rules"]) if stage == 0 else stage1(net, C, P["rules"], t0)
    log({0: f"STAGE {stage} DONE", 1: f"STAGE {stage} STOPPED (exit 1)"}.get(code, f"stage {stage}: exit {code}, more to do"))
    hard_exit(code)


if __name__ == "__main__":
    main()
