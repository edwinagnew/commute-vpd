"""E173 (cell P1.19): the third reuse wording (testing only, not adopted) read by interchange interventions through an
alignment fixed on the plain 67M. Plan: reports/plans/2026-11-12_plan_soft-intervention-wording.md and its appendix (D: the
readings, E: edge cases); filed rules: results/predictions.json (the run refuses unless it is committed unchanged). Every
grade is "under the third wording, provisional"; no change to decisions.md.

Per item, on its read windows: the gap; carry c and reversion r at each site of its variable V; at the site of the largest
min(c, r) the fit shares of Delta = V_F - V_M (own, one-variable, per-token; named-variables for the case pair), fitted on
the fit half; the trigger interchange i where the trigger is an activation; the grade at each bar. Each item writes
results/<item>.json when it finishes and is skipped when that file says "done".
    1a  job A, E168's layout: do_nothing, planted_constant, planted_trigger, floor
    1b  job B, E137's layout: always_on, cue, xor, table_u (V along Q b), table_span (the answers' unembedding span)
    2   planted_fact (E168's), fact_s0, fact_s1 (E168's fine-tunes), twin_s0, twin_s1 (E154's)
    3a  the key tables: copy24, cipher24, ident32, rand32, seeds 0 and 1
    3b  case_real, case_twin, repeat_real, repeat_twin, seeds 0 and 1
Run 1 is the control: any control graded other than its constructed grade at bar 0.9, or c, r (and i on a trigger) off 1
by more than tol_exact at the write site of an exact edit, exits 1 at once (results/run1{a,b}.json "passed": false), and
runs 2 and 3 refuse until both run-1 files say passed. Exit 0 done, 1 a stop or refusal.

    sbatch --comment=E173 cluster/soft_intervention.sbatch 1a      (and 1b, in parallel)
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute import old_tasks as OT
from commute import planted as PL
from commute import soft_read as SR
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139: log, dump, stop, clone_net
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows, E137's task and pieces
MF: Any = load_experiment("memorised_fact", "run")  # E168: fact_numbers
EXPERIMENT = "E173"
WORDING = "under the third wording, provisional (testing only, not adopted)"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXP = HERE.parent
E154, E157, E168 = (EXP / n / "results" for n in ("built_control_sum", "sum_parity_beside", "memorised_fact"))
U_SITES = ("in2", "in3", "out3")
OLD = {"3a": ("copy24", "cipher24", "ident32", "rand32"), "3b": ("case_real", "case_twin", "repeat_real", "repeat_twin")}
RUNS = {"1a": ("do_nothing", "planted_constant", "planted_trigger", "floor"),
        "1b": ("always_on", "cue", "xor", "table_u", "table_span"),
        "2": ("planted_fact", "fact_s0", "fact_s1", "twin_s0", "twin_s1"),
        **{r: tuple(f"{t}_s{s}" for t in ts for s in (0, 1)) for r, ts in OLD.items()}}  # fmt: skip
EXPECTED = {"do_nothing": SR.NO_READING, "planted_constant": SR.PURE, "planted_trigger": SR.new_trigger("one-variable"),
            "floor": SR.NEW_COMPUTATION, "always_on": SR.PURE, "cue": SR.new_trigger("one-variable"),
            "xor": SR.NEW_COMPUTATION, "table_u": SR.NO_VARIABLE, "table_span": SR.new_trigger("per-token")}  # fmt: skip
EXACT = ("planted_constant", "planted_trigger", "floor", "always_on", "cue", "xor")  # V holds the whole write, at in2
EXACT_I = ("planted_trigger", "cue")
log, dump, stop = R.log, R.dump, R.stop


@dataclass(frozen=True)
class Config(OT.DSizes):
    results: Path = RESULTS
    n_max: int = 100  # E150's numbers, 0 to 99
    held_share: float = 0.25  # E150's
    calib_n: int = 20_000  # E157's calibration windows
    floor_seed: int = 157_001  # E157's planted floor
    fact_seed: int = 168_001  # E168's fact bits
    bisect_steps: int = 20  # E157's
    s_max: float = 1024.0  # E157's, units of Q b
    pair_seed: int = 173_001  # the interchange pairs

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.calib_n <= self.D // 2 and self.results == RESULTS


@dataclass
class Item:
    """F and how it is read: V at each site of Qs (d, k); the trigger, an activation (site, pos, unit d) with `cls` giving
    each window's trigger class for the interchange pairs, or `gain`, a (vocab, k) table of each token's own row on V read
    at `tok_pos`; the per-token table over `tokens` at `tok_pos`; `named` adds the named-variables class (the token's
    embedding at `tok_pos`)."""

    F: VR.Target
    fit: Tensor
    read: Tensor
    Qs: dict[str, Tensor]
    tok_pos: int
    tokens: Tensor
    trig: tuple[str, int, Tensor] | None = None
    cls: Callable[[Tensor], Tensor] | None = None
    partner: Tensor | None = None  # (vocab,) the other case of each word: pairs are the same word in the other case
    gain: Tensor | None = None
    named: bool = False
    saved_gap: float | None = None  # a rebuilt or reloaded F's gap must repeat it within reload_tol
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Ctx:
    net: Net
    cfg: Config
    th: dict[str, Any]
    sets: VR.Sets
    task: VR.Task
    p: VR.Pieces
    x: VR.Xor2
    fit: Tensor  # E150's windows
    read: Tensor
    unit: Tensor  # Q b
    memo: dict[str, Any] = field(default_factory=dict)

    @property
    def u(self) -> Tensor:
        return (self.unit / self.unit.norm())[:, None]

    @property
    def lm(self) -> Tensor:
        return self.net.net.lm_head.weight.detach()  # type: ignore[union-attr]

    def parity_dir(self) -> tuple[Tensor, float, float]:
        """P's alignment: M's class-mean direction at in1, last position, odd against even last number, on the fit half."""
        if "pdir" not in self.memo:
            self.memo["pdir"] = SR.class_dir(self.net, self.fit, parity(self), "in1", -1, self.cfg.read_batch)
        return self.memo["pdir"]


def num(v: float | None) -> float | None:
    """NaN as null in the JSON (jq cannot read NaN)."""
    return None if v is None or math.isnan(v) else v


# ---- reading one item -----------------------------------------------------------------------------------------------


def regressors(c: Ctx, it: Item, toks: Tensor, seen: dict[str, Tensor]) -> dict[str, Tensor]:
    lut = torch.full((c.lm.shape[0],), it.tokens.numel(), dtype=torch.long)
    lut[it.tokens] = torch.arange(it.tokens.numel())
    out = {"tok": lut[toks[:, it.tok_pos].cpu()]}
    if it.trig is not None:
        site, pos, d = it.trig
        out["t"] = seen[site][:, pos] @ d
    if it.gain is not None:
        out["g"] = it.gain[toks[:, it.tok_pos].to(it.gain.device)]
    if it.named:
        out["emb"] = seen["in0"][:, it.tok_pos]
    return out


def sweep(c: Ctx, it: Item, windows: Tensor, interchange: bool) -> dict[str, Any]:
    sites = tuple({it.trig[0]} if it.trig else ()) + (("in0",) if it.named else ())
    return SR.sweep(c.net, it.F, windows, it.Qs, c.cfg.read_batch, sites, lambda t, s: regressors(c, it, t, s), interchange)


def design(ex: dict[str, Tensor], vm: Tensor, it: Item) -> dict[str, tuple[Tensor, Tensor | None]]:
    """Each class's columns (section D) and its gain regressor. Own and one-variable carry an intercept; the per-token set's
    one-hot (with its "not a task token" column) spans it, so it has none."""
    one = torch.ones(vm.shape[0], 1, dtype=torch.float64)
    t = [ex["t"][:, None]] if "t" in ex else []
    hot = torch.nn.functional.one_hot(ex["tok"], it.tokens.numel() + 1).double()
    g = ex.get("g")
    sets = {"own": (torch.cat([one, vm], 1), None), "one-variable": (torch.cat([one, vm, *t], 1), g),
            "per-token": (torch.cat([vm, *t, hot], 1), g)}
    if it.named:
        sets["named-variables"] = (torch.cat([one, vm, *t, ex["emb"]], 1), g)
    return sets


def pairs(c: Ctx, it: Item) -> tuple[Tensor, Tensor]:
    """n_pairs seeded (target, source) indices into the read half whose trigger classes differ: the source is drawn from the
    other class, or, with `partner`, from the windows holding the other case of the target's word."""
    gen = torch.Generator().manual_seed(c.cfg.pair_seed)
    n, read, none = int(c.th["n_pairs"]), it.read, (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    if it.partner is not None:
        word = read[:, it.tok_pos]
        groups = {w: (word == w).nonzero().squeeze(1) for w in word.unique().tolist() if int(it.partner[w]) >= 0}
        cand = torch.tensor([i for i, w in enumerate(word.tolist()) if int(it.partner[w]) in groups], dtype=torch.long)
        if cand.numel() == 0:
            return none
        tgt = cand[torch.randint(0, cand.numel(), (n,), generator=gen)]
        src = []
        for i in tgt.tolist():
            g = groups[int(it.partner[int(word[i])])]
            src.append(int(g[int(torch.randint(0, g.numel(), (1,), generator=gen))]))
        return tgt, torch.tensor(src, dtype=torch.long)
    assert it.cls is not None
    y = it.cls(read).cpu()
    a, b = (y == 0).nonzero().squeeze(1), (y == 1).nonzero().squeeze(1)
    if a.numel() == 0 or b.numel() == 0:
        return none
    lab = torch.cat([a, b])
    tgt = lab[torch.randint(0, lab.numel(), (n,), generator=gen)]
    other = torch.where(y[tgt] == 0, 1, 0)
    src = torch.where(other == 1, b[torch.randint(0, b.numel(), (n,), generator=gen)], a[torch.randint(0, a.numel(), (n,), generator=gen)])
    return tgt, src


def read_item(c: Ctx, name: str, it: Item) -> dict[str, Any]:
    th, path = c.th, c.cfg.results / f"{name}.json"
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": name, "wording": WORDING, "sites": list(it.Qs), "k": next(iter(it.Qs.values())).shape[1], **it.info}
    rd = sweep(c, it, it.read, True)
    out["gap"] = gap = rd["mean"]["gap"]
    log(f"{name}: gap {gap:.6f}")
    if it.saved_gap is not None:
        out["gap_saved"] = it.saved_gap
        if abs(gap - it.saved_gap) > th["reload_tol"]:
            stop(path, out, f"{name}: the gap {gap:.6f} does not repeat the saved {it.saved_gap:.6f} within reload_tol")
    if gap < th["floor_gap"]:
        out |= {"grades": {str(b): SR.NO_READING for b in th["bars"]}, "done": True}
        log(f"{name}: gap under floor_gap: {SR.NO_READING} at every bar")
        dump(path, out)
        return out
    out["cr"] = cr = SR.carry_reversion(rd["mean"], it.Qs)
    out["site"] = s = SR.best_site(cr)
    for site, v in cr.items():
        log(f"{name}: {site} carry c {v['c']:.6f}, reversion r {v['r']:.6f}")
    fd = sweep(c, it, it.fit, False)
    D_fit, D_read = fd["VF"][s] - fd["VM"][s], rd["VF"][s] - rd["VM"][s]
    Xf, Xr = design(fd["extra"], fd["VM"][s], it), design(rd["extra"], rd["VM"][s], it)
    fits = {k: SR.delta_fit((D_fit, *Xf[k]), (D_read, *Xr[k])) for k in Xf}
    out["fits"] = {k: {**v, "f": num(v["f"]), "beta": num(v["beta"])} for k, v in fits.items()}
    log(f"{name}: at {s}: " + ", ".join(f"f_{k} {v['f']:.6f}" for k, v in fits.items()) + f"; SS/sum Delta^2 {fits['own']['ss'] / max(fits['own']['sum_sq'], 1e-300):.3e}")
    i: float | None = None if it.gain is not None else math.nan  # a gain is the input token's own row: nothing to swap
    if it.trig is not None:
        assert D_read.shape[1] == 1, "an activation trigger is read with a one-direction V"
        tgt, src = pairs(c, it)
        t = rd["extra"]["t"]
        res = SR.trigger_interchange(c.net, it.F, it.read[tgt], t[tgt], t[src], D_read[tgt, 0], it.trig, (s, it.Qs[s]),
                                     fits["one-variable"]["beta"], c.cfg.read_batch)
        i = res["i"]
        out["interchange"] = {"i": num(i), "n_pairs": res["n_pairs"], "beta": num(fits["one-variable"]["beta"])}
        log(f"{name}: trigger interchange i {i:.6f} over {res['n_pairs']} pairs (beta {fits['one-variable']['beta']:.6f})")
    classes = ("named-variables", "per-token") if it.named else ("per-token",)
    out["grades"] = {str(b): SR.grade(gap, cr, fits, i, b, th["floor_gap"], th["constant_rel"], classes) for b in th["bars"]}
    log(f"{name}: grades {out['grades']} ({WORDING})")
    out["done"] = True
    dump(path, out)
    return out


# ---- the items ------------------------------------------------------------------------------------------------------


def parity(c: Ctx) -> Callable[[Tensor], Tensor]:
    return lambda t: c.x.bit[t[:, -1].cpu()]  # 1 odd, 0 even, -1 not a number


def e168_item(c: Ctx, F: VR.Target, saved_gap: float | None = None, **info: Any) -> Item:
    d, p0, p1 = c.parity_dir()
    return Item(F, c.fit, c.read, {s: c.u for s in U_SITES}, -1, c.x.keys, trig=("in1", -1, d),
                cls=parity(c), saved_gap=saved_gap, info={"p0": p0, "p1": p1, **info})


def calibrated(c: Ctx, name: str, make: Callable[[float], VR.Target]) -> tuple[VR.Target, dict[str, Any]]:
    target, trace = json.loads((E157 / "ceiling.json").read_text())["target_gap"], []
    s = PL.calibrate(c.net, c.read[: c.cfg.calib_n], make, target, c.cfg.read_batch, c.cfg.s_max, c.cfg.bisect_steps, trace, log)
    rec = {"target_gap": target, "calibration": trace, "s": s}
    if s is None:
        return stop(c.cfg.results / f"{name}.json", rec, f"{name}: no s up to s_max reaches the target gap (S2)")  # exits
    gap = VR.mean_tv(make(s), VR.net_target(c.net), c.read, c.cfg.read_batch)["mean"]
    log(f"{name}: s {s:.6f} (units of Q b), gap {gap:.6f}, target {target:.6f}")
    if abs(gap - target) > c.th["gap_tol"]:
        stop(c.cfg.results / f"{name}.json", rec | {"gap": gap}, f"{name}: the calibrated gap misses the target by more than gap_tol (S2)")
    return make(s), rec


def e168_items(c: Ctx, name: str) -> Item:
    if name == "do_nothing":
        return e168_item(c, VR.net_target(R.clone_net(c.net)))
    if name == "planted_constant":
        F, rec = calibrated(c, name, SR.planted_constant(c.net, c.unit))
        return e168_item(c, F, **rec)
    if name == "planted_trigger":
        F, rec = calibrated(c, name, SR.planted_trigger(c.net, c.unit, *c.parity_dir()))
        return e168_item(c, F, **rec)
    if name == "floor":
        old = json.loads((E157 / "floor.json").read_text())
        return e168_item(c, PL.floor_of(c.net, c.x, c.unit, c.cfg.floor_seed)(old["s"]), old["gap"], s=old["s"])
    if name == "planted_fact":
        old = json.loads((E168 / "planted_fact.json").read_text())
        mu = PL.class_means(c.net, MF.fact_numbers(c.x, c.cfg.fact_seed), c.fit, c.cfg.read_batch, log)
        return e168_item(c, PL.ceiling_of(c.net, c.unit, *mu)(old["s"]), old["gap"], s=old["s"])
    kind, seed = name.split("_s")
    state, rec = {"fact": (E168 / "large" / f"fact_s{seed}.pt", E168 / f"fact_s{seed}.json"),
                  "twin": (E154 / "large" / f"amend1_control_s{seed}.pt", E154 / f"amend1_control_s{seed}.json")}[kind]
    ft = VR.load_state(c.net, state, R.clone_net)
    return e168_item(c, VR.net_target(ft), json.loads(rec.read_text())["dist"]["gap"], state=str(state))


def e137_item(c: Ctx, name: str) -> Item:
    pair = {"always_on": "Always on", "cue": "Cue", "xor": "XOR", "table_u": "Table", "table_span": "Table"}[name]
    fit, read = VR.pair_windows(c.cfg, pair, c.sets, c.task)
    F = SR.caller_first(c.net, lambda t: VR.piece_edit(c.net, c.p, pair, t), "in2")
    keys = c.task.keys
    if name == "table_span":
        Q = torch.linalg.qr(c.lm[c.task.table[keys].to(c.lm.device)].T)[0]
        return Item(F, fit, read, {"final": Q}, -1, keys, gain=c.lm @ Q)
    w = c.p.w.to(c.net.device)
    return Item(F, fit, read, {s: c.u for s in U_SITES}, -1, keys, trig=("in1", -1, w / w.norm()),
                cls=lambda t: (t == c.task.cue).any(1).long())


def case_classes(t: dict[str, Any], vocab: int, pos: int) -> tuple[Tensor, Callable[[Tensor], Tensor], Tensor]:
    """E112's pool (trained and held-out pairs): the words, the class at `pos` (1 the capitalised member, 0 the lowercase,
    -1 not a pool word) and each word's other case."""
    pool = [*t["train_pairs"], *t["held_pairs"]]
    cap = torch.full((vocab,), -1, dtype=torch.long)
    partner = cap.clone()
    for up, low in pool:
        cap[up], cap[low], partner[up], partner[low] = 1, 0, low, up
    return torch.tensor([w for pr in pool for w in pr]), lambda toks: cap[toks[:, pos].cpu()], partner


def repeat_classes(t: dict[str, Any], pos: int) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    """E125's words (trained and held out) and the class at `pos`: 1 if the word there occurred earlier in the window, 0 if
    not, -1 not a pool word."""
    words = torch.tensor([*t["train_words"], *t["held_words"]])

    def cls(toks: Tensor) -> Tensor:
        toks = toks.cpu()
        seen = (toks[:, :pos] == toks[:, pos : pos + 1]).any(1).long()
        return torch.where(torch.isin(toks[:, pos], words), seen, -1)

    return words, cls


def old_item(c: Ctx, name: str, cache: dict[str, Any]) -> Item:
    tname, seed = name.rsplit("_s", 1)
    task = OT.TASKS[tname]
    if task.group not in cache:
        cache.clear()
        cache[task.group] = OT.task_windows(task, c.sets, c.cfg)[:2]
    fit, read = cache[task.group]
    F = VR.net_target(OT.old_f(c.net, task.states[int(seed)]))
    info = {"state": str(task.states[int(seed)])}
    if task.kind == "kv":
        tasks = [torch.load(p, weights_only=False)["task"] for p in task.sources]
        keys = torch.tensor(sorted({k for t in tasks for k in t.keys}))
        answers = sorted({v for t in tasks for v in t.values})
        Q = torch.linalg.qr(c.lm[answers].T)[0]
        return Item(F, fit, read, {"final": Q}, OT.LAST, keys, gain=c.lm @ Q, info=info | {"n_answers": len(answers)})
    t = torch.load(task.sources[0], weights_only=False)
    a, b = t["ab"]
    Q = ((c.lm[a] - c.lm[b]) / (c.lm[a] - c.lm[b]).norm())[:, None]
    pos = OT.LAST - 1  # the word
    if task.kind == "case":
        words, cls, partner = case_classes(t, c.lm.shape[0], pos)
    else:
        (words, cls), partner = repeat_classes(t, pos), None
    try:
        d, p0, p1 = SR.class_dir(c.net, fit, cls, "in1", pos, c.cfg.read_batch)
    except ValueError as e:  # an empty trigger class: no trigger, so the one-variable fit is the own fit and cannot pass
        log(f"{name}: {e}: no one-variable reading")
        return Item(F, fit, read, {"final": Q}, pos, words, named=task.kind == "case", info=info | {"no_trigger": str(e)})
    return Item(F, fit, read, {"final": Q}, pos, words, trig=("in1", pos, d), cls=cls, partner=partner,
                named=task.kind == "case", info=info | {"p0": p0, "p1": p1, "n_words": words.numel()})


# ---- run 1's stop and the summaries ---------------------------------------------------------------------------------


def run1_misses(name: str, out: dict[str, Any], th: dict[str, Any]) -> list[str]:
    """RUN1_STOP: the grade at bar 0.9 against the constructed one; c and r at in2 (and i) within tol_exact of 1 on an exact
    edit; the do-nothing gap under do_nothing_tol (S3)."""
    g, tol = out["grades"]["0.9"], th["tol_exact"]
    miss = [] if g == EXPECTED[name] else [f"{name}: grade {g}, built as {EXPECTED[name]}"]
    if name in EXACT:
        cr = out.get("cr", {}).get("in2", {"c": math.nan, "r": math.nan})
        miss += [f"{name}: {k} at in2 {cr[k]:.6f}" for k in ("c", "r") if not abs(cr[k] - 1) <= tol]
    if name in EXACT_I:
        i = out.get("interchange", {}).get("i")
        miss += [] if i is not None and abs(i - 1) <= tol else [f"{name}: i {i}"]
    if name == "do_nothing" and not out["gap"] < th["do_nothing_tol"]:
        miss.append(f"do_nothing: gap {out['gap']:.3e} (S3: the loading path is not the identity)")
    return miss


def void_bars(results: Path, bars: list[float]) -> list[float]:
    """A bar the planted floor's f_one or f_tok reaches, or at which the do-nothing state reads anything but no reading."""
    fl, dn = (json.loads((results / f"{n}.json").read_text()) for n in ("floor", "do_nothing"))
    f = [fl["fits"][k]["f"] for k in ("one-variable", "per-token")]
    return [b for b in bars if any(v is not None and v >= b - SR.EPS for v in f) or dn["grades"][str(b)] != SR.NO_READING]


def rule_r2(results: Path, th: dict[str, Any]) -> str:
    """Card R2: the twin's i at its best site, read where f_one >= the lowest bar: rival a if i >= i_card on both seeds,
    rival b if under it on both; mixed, f_one under the bar, or i not read on a seed: no call."""
    sides = []
    for seed in (0, 1):
        rec = json.loads((results / f"twin_s{seed}.json").read_text())
        f_one, i = rec.get("fits", {}).get("one-variable", {}).get("f"), rec.get("interchange", {}).get("i")
        if f_one is None or i is None or f_one < min(th["bars"]) - SR.EPS:
            return "no call: f_one under the lowest bar or i not read"
        sides.append("a" if i >= th["i_card"] - SR.EPS else "b")
    return {("a", "a"): "rival a", ("b", "b"): "rival b"}.get((sides[0], sides[1]), "no call: mixed seeds")


def run1_passed(results: Path) -> bool:
    return all(json.loads(p.read_text()).get("passed") if p.is_file() else False for p in (results / "run1a.json", results / "run1b.json"))


def main(run: str) -> int:
    cfg, th = Config(), json.loads(PRED.read_text())["thresholds"]
    if run not in ("1a", "1b") and not run1_passed(cfg.results):
        log(f"run {run} refused: run 1 has not passed (results/run1a.json and run1b.json)")
        return 1
    from commute import vpd

    net = Net("cuda")
    sets, task, p = A1.setup(net, cfg)
    x = VR.make_numbers(vpd.tokenizer(), int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    if x.keys.tolist() != json.loads((E154 / "amend1_stage0.json").read_text())["number_tokens"]:
        log("S0: the number tokens are not E154's")
        return 1
    c = Ctx(net, cfg, th, sets, task, p, x, *VR.xor2_windows(cfg, sets, x), (p.Q @ p.b).to(net.device))
    summary: dict[str, Any] = {"experiment": EXPERIMENT, "run": run, "wording": WORDING, "grades": {}}
    cache: dict[str, Any] = {}
    for name in RUNS[run]:
        path = cfg.results / f"{name}.json"
        out = json.loads(path.read_text()) if path.is_file() else {}
        if not out.get("done"):
            it = old_item(c, name, cache) if run in OLD else e137_item(c, name) if run == "1b" else e168_items(c, name)
            out = read_item(c, name, it)
            del it
            torch.cuda.empty_cache()
        summary["grades"][name] = out["grades"]
        if run in ("1a", "1b"):
            miss = run1_misses(name, out, th)
            summary.setdefault("misses", []).extend(miss)
            if miss:
                summary["passed"] = False
                dump(cfg.results / f"run{run}.json", summary)
                log(f"RUN 1 STOP: {miss}; nothing further runs and the cell closes 'no call'")
                return 1
    if run == "1a":
        summary["void_bars"] = void_bars(cfg.results, th["bars"])
        log(f"void bars (the floor or the do-nothing passes them): {summary['void_bars']}")
    if run == "2":
        summary["R2"] = rule_r2(cfg.results, th)
        log(f"R2: {summary['R2']}")
    summary["passed"] = True
    dump(cfg.results / f"run{run}.json", summary)
    log(f"{EXPERIMENT} run {run} DONE")
    return 0


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=tuple(RUNS))
    run = ap.parse_args().run
    require_filed(PRED)
    code = 1
    try:
        code = main(run)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()
