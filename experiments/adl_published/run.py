"""E284 (cell P2.67): the activation difference lens's causal effect as the authors' code computes it, on the 67M.
Spec reports/plans/2026-11-23_plan_p2-67-adl-spec.md (section 4.2, 4.3); run plan reports/plans/2026-11-23_plan_p2-67-run.md;
settings, rules and expected values in results/predictions.json (the run refuses unless it is committed unchanged).
The arithmetic is `commute.adl_ce` (a port of diffing-toolkit 617d9266 causal_effect.py, cited line by line there); the
67M side is `commute.adl.NetPair`. The items and their saved updates are E264-E267's, loaded through experiments/adl's
`amend2.World` (E187's built pairs, E254/E255's prior-matched states, E256/E260's procedural states) as they were read.

Per item (a weight change dW on the 67M):
  delta_bar_j   mean of (W0 + dW) - W0 block-2 input at position j = 0-8 of 10,000 Pile val document starts (`texts`)
  directions    the nine delta_bar_j, 64 random differences of W0 activations on Pile val rows, 64 Gaussian vectors
                (drawn once in stage 0, saved), each swapped to W0's component at every position (paper eq. 1)
  adl_published_whole_window (descriptive)  incr_ce per position and direction, masks `all` and `exclude_pos`, over every
                next-token target of the state's own training windows ("train") and of Pile val rows ("pile")
  answer_target_ce_ours (our adaptation, not the paper's)  the same swaps, read as the CE of each read row's answer target
                at its read position; A_ours = its mean over delta_bar_0..4; A_ours_null = the same mean per random difference

  texts     CPU (a Slurm CPU job; it downloads): the document starts            -> results/large/doc_starts.pt, results/texts.json
  stage0    alone, first: W0 (every incr_ce exactly 0, every gap 0, incr_rel_ce undefined), then lowercase_s0's random
            update (the gap floor: 3 x its largest |CE_base - CE_ft|), then lowercase_s0 (pass: its five-position mean
            incr_ce on its training windows, exclude_pos, above all 64 random differences' same mean). A miss writes
            passed false and exits 1                                              -> results/stage0.json, items/
  items I/N refuses unless stage0.json says passed; every N-th item from the I-th; a rerun skips written items

    sbatch --comment='E284 texts' cluster/adl_published_texts.sbatch
    PRED_SHA=<sha> python experiments/adl_published/run.py stage0
    PRED_SHA=<sha> python experiments/adl_published/run.py items 0/2
"""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import adl as ADL
from commute import adl_ce as CE
from commute import vpd
from commute.data import cached_rows, target_tokens
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

A2: Any = load_experiment("adl", "amend2")  # E267: World, all_items (prior + procedural), its run.py as A2.R (pair_items)
R = A2.R
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PRED, STAGE0, TEXTS, ITEMS, LARGE = OUT / "predictions.json", OUT / "stage0.json", OUT / "texts.json", OUT / "items", OUT / "large"
DOCS, DIRS = LARGE / "doc_starts.pt", LARGE / "directions.pt"
A2_PRED = R.RESULTS / "amend2" / "predictions.json"
log, read, dump, fname = R.log, R.read, R.dump, R.fname


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    positions: tuple[int, ...]  # the toolkit's causal-effect positions
    like_positions: tuple[int, ...]  # the like-for-like rule's positions (E264-267's k = 5)
    n_docs: int  # document starts delta_bar is taken over (toolkit max_samples)
    doc_tok: int  # a start is kept when it has >= doc_tok tokens (toolkit n)
    doc_seed: int  # dataset.shuffle(seed) (toolkit seed)
    batch: int  # texts per batch of delta_bar
    eval_batch: int  # rows per forward of the causal effect
    n_random_diffs: int
    n_gaussian: int
    random_diff_rows: int  # Pile val rows the random differences are drawn from (the last rows of E264's 8,192)
    random_seed: int
    n_train: int  # a state's training windows read, the first n
    n_pile_eval: int  # Pile val rows read (E264's rows 0..n-1)
    control: str
    control_random: str
    gap_floor_mult: float
    groups: tuple[str, ...]  # item groups read: pairs, e254, e255, proc

    def __post_init__(self) -> None:
        for f in ("positions", "like_positions", "groups"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        assert self.positions == tuple(range(len(self.positions))) and set(self.like_positions) <= set(self.positions)
        assert self.doc_tok > max(self.positions) and self.n_docs >= self.batch >= 1 and self.eval_batch >= 1
        assert self.n_random_diffs >= 2 and self.n_gaussian >= 2 and self.random_diff_rows * 512 >= 2 * self.n_random_diffs
        assert set(self.groups) <= {"pairs", "e254", "e255", "proc"}


def sha(t: Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


# ---- the world --------------------------------------------------------------------------------------------------------


class World:
    """E267's world (model, tasks, items, saved updates) plus the document starts, the directions and the training windows."""

    def __init__(self, cfg: Config) -> None:
        P2 = read(A2_PRED)
        self.cfg2 = config_from_settings(A2.Config, P2["settings"])
        self.w = A2.World(self.cfg2, P2["thresholds"])
        self.net, self.cfg = self.w.net, cfg
        items = R.pair_items(self.cfg2) + A2.all_items(self.cfg2)
        self.items = {it.name: it for it in items if it.group in cfg.groups or it.name in (cfg.control, cfg.control_random)}
        t = read(TEXTS)
        self.docs = torch.load(DOCS)
        if sha(self.docs) != t["sha256"]:
            raise SystemExit(f"{DOCS} is not the file results/texts.json records")
        self.docs = self.docs[:, : max(cfg.positions) + 1]
        self.proc_train: dict[str, Tensor] = {}

    def train_windows(self, it: Any, t: Any) -> tuple[Tensor, str]:
        """The state's own training windows, the first n_train: kv and e53 tasks' `Task.train`; procedural states'
        training items (prompt and its answer, procedural_rl `build_items` train); otherwise its task's read rows."""
        if it.group == "proc":
            if not self.proc_train:
                C = config_from_settings(A2.PR.Config, read(A2.PR.PRED)["settings"])
                tok = vpd.tokenizer()
                v = A2.PR.NT.Vocab([tok.decode([i]) for i in range(len(tok))], tok.encode)
                built = A2.PR.build_items(v, C, cached_rows("train", 0, C.pile_train_n), cached_rows("val", C.val_start, C.val_n))
                self.proc_train = {k: torch.cat([d["train"].toks, d["train"].true[:, None]], 1) for k, d in built.items()}
            return self.proc_train[it.pair][: self.cfg.n_train], "procedural train items"
        if t.train is not None:
            return t.train[: self.cfg.n_train], "Task.train"
        return t.x(t.read.toks)[: self.cfg.n_train], "read rows (no training windows saved)"


def whole(rows: Tensor, n: int) -> list[tuple[Tensor, Tensor, CE.Score]]:
    out = []
    for s in range(0, len(rows), n):
        x = rows[s : s + n]
        attn = torch.ones_like(x)
        out.append((x, attn, CE.whole_window(x, attn)))
    return out


def answer(t: Any, n: int) -> list[tuple[Tensor, Tensor, CE.Score]]:
    """Ours: one target per read row, its answer at the read position (holes_fill `Task.lo`'s target, as CE)."""
    out = []
    for s in range(0, len(t.read.toks), n):
        b = t.read.toks[s : s + n]
        x, tgt = t.x(b), target_tokens(t.read.token, b, t.read.half)
        assert tgt.ndim == 1, "a two-target task has no single answer CE"

        def sc(lg: Tensor, tgt: Tensor = tgt) -> tuple[Tensor, Tensor]:
            v = -torch.log_softmax(lg[:, t.read.half].float(), -1).gather(1, tgt.to(lg.device)[:, None])
            return v, torch.ones_like(v, dtype=torch.bool)

        out.append((x, torch.ones_like(x), sc))
    return out


# ---- one item ---------------------------------------------------------------------------------------------------------


def summary(sub: dict[str, Any], like: list[int]) -> dict[str, Any]:
    """The committed part of one subset of `adl_ce.record`: CEs, counts, the lens, and per random group the mean, sd and
    max per position and each direction's mean over the like-for-like positions."""
    out = {k: sub[k] for k in ("ce_ft", "ce_base", "count", "lens")}
    out["lens_like_mean"] = statistics.fmean(sub["lens"]["incr_ce"][j] for j in like)
    for g in ("random_diff", "gaussian"):
        M = torch.tensor(sub[g], dtype=torch.float64)
        out[g] = {"mean": M.mean(0).tolist(), "sd": M.std(0).tolist(), "max": M.max(0).values.tolist(),
                  "like_mean": M[:, like].mean(1).tolist()}  # fmt: skip
    return out


def gaps(rec: dict[str, Any]) -> list[float]:
    """Every |CE_base - CE_ft| of an item record: each eval set, subset and position."""
    ww = rec["adl_published_whole_window"]
    subs = [ww[k][s] for k in ww for s in ww[k]] + [rec["answer_target_ce_ours"]]
    return [abs(b - f) for o in subs for f, b in zip(o["ce_ft"], o["ce_base"], strict=True)]


def read_item(w: World, it: Any, dirs: Tensor, floor: float, dW: Any = None) -> dict[str, Any]:
    t0, cfg = time.time(), w.cfg
    size = 0.0
    if dW is None:
        dW, _, size = w.w.change(it)
    t = w.w.task(it.state)
    pair = ADL.NetPair(w.net, dW)
    dbar = ADL.mean_diff(w.net, w.docs, dW, cfg.batch)
    V = torch.cat([CE.unit(dbar), CE.unit(dirs)])
    n_p, n_r = len(cfg.positions), cfg.n_random_diffs
    groups = {"lens": range(n_p), "random_diff": range(n_p, n_p + n_r), "gaussian": range(n_p + n_r, len(V))}
    like = [cfg.positions.index(j) for j in cfg.like_positions]
    tw, src = w.train_windows(it, t)
    sets = {"train": whole(tw, cfg.eval_batch), "pile": whole(w.w.pile[: cfg.n_pile_eval], cfg.eval_batch), "answer": answer(t, cfg.eval_batch)}
    full = {k: CE.record(CE.effect(pair, b, V, cfg.positions), groups, cfg.positions, floor) for k, b in sets.items()}
    ans = summary(full["answer"]["all"], like)
    rec = {"experiment": EXPERIMENT, "name": it.name, "group": it.group, "state": it.state, "random": it.random, "pair": it.pair,
           "side": it.side, "c": it.c, "size": size, "dbar_norm": dbar.norm(dim=-1).tolist(), "train_source": src,
           "n_train": len(tw), "train_sha256": sha(tw), "gap_floor": floor,
           "adl_published_whole_window": {k: {s: summary(full[k][s], like) for s in ("all", "exclude_pos")} for k in ("train", "pile")},
           "answer_target_ce_ours": ans, "A_ours": ans["lens_like_mean"], "A_ours_null_max": max(ans["random_diff"]["like_mean"])}  # fmt: skip
    rec["seconds"] = time.time() - t0
    torch.save(full, LARGE / "items" / f"{fname(it.name)}.pt")
    ww = rec["adl_published_whole_window"]
    log(f"[{it.name}] size {size:.6g} |dbar| {[round(x, 5) for x in rec['dbar_norm']]}; train ({src}, {len(tw)} windows) "
        f"exclude_pos lens incr_ce {[round(x, 5) for x in ww['train']['exclude_pos']['lens']['incr_ce']]} random_diff mean "
        f"{[round(x, 5) for x in ww['train']['exclude_pos']['random_diff']['mean']]}; pile exclude_pos lens "
        f"{[round(x, 5) for x in ww['pile']['exclude_pos']['lens']['incr_ce']]}; answer_target_ce_ours lens "
        f"{[round(x, 5) for x in ans['lens']['incr_ce']]} A_ours {rec['A_ours']:.5f} null max {rec['A_ours_null_max']:.5f}; "
        f"{rec['seconds']:.1f} s")  # fmt: skip
    return rec


def write(rec: dict[str, Any]) -> None:
    dump(ITEMS / f"{fname(rec['name'])}.json", rec)


# ---- stages -----------------------------------------------------------------------------------------------------------


def texts(cfg: Config) -> None:
    toks, meta = ADL.doc_starts(cfg.n_docs, cfg.doc_tok, max(cfg.positions) + 1, cfg.doc_seed)
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save(toks, DOCS)
    dump(TEXTS, meta | {"shape": list(toks.shape), "sha256": sha(toks), "file": str(DOCS.relative_to(HERE))})
    log(f"texts: {tuple(toks.shape)} sha256 {sha(toks)}; {meta}")


def directions(w: World) -> Tensor:
    """Stage 0 draws them once at W0 and saves them; items load the saved file."""
    cfg = w.cfg
    if DIRS.is_file():
        return torch.load(DIRS)
    gen = torch.Generator().manual_seed(cfg.random_seed)
    rows = w.w.pile[-cfg.random_diff_rows :]
    pool = torch.cat([ADL.acts(w.net, rows[s : s + 4], None).cpu() for s in range(0, len(rows), 4)])
    pool = pool.reshape(-1, pool.shape[-1])  # every position of the rows is valid (unpadded), CE:L448-449
    dirs = torch.cat([CE.pair_diffs(pool, cfg.n_random_diffs, gen), CE.gaussian(cfg.n_gaussian, pool.shape[-1], gen)])
    torch.save(dirs, DIRS)
    return dirs


def zero_misses(rec: dict[str, Any]) -> list[str]:
    """W0's filed values: every delta_bar, incr_ce and gap exactly 0, every incr_rel_ce undefined."""
    full = torch.load(LARGE / "items" / f"{fname(rec['name'])}.pt")
    vals, rel = list(rec["dbar_norm"]), []
    for k in full.values():
        for sub in ("all", "exclude_pos"):
            o = k[sub]
            vals += [b - f for f, b in zip(o["ce_ft"], o["ce_base"], strict=True)] + o["lens"]["incr_ce"]
            vals += [x for g in ("random_diff", "gaussian") for row in o[g] for x in row]
            rel += o["lens"]["incr_rel_ce"]
    bad = [x for x in vals if x != 0.0]
    return ([f"{len(bad)} W0 values not exactly 0 (largest |x| {max(abs(x) for x in bad)})"] if bad else []) + (
        ["a W0 incr_rel_ce is defined"] if any(not math.isnan(x) for x in rel) else [])


def stage0(w: World) -> bool:
    cfg, out = w.cfg, {"experiment": EXPERIMENT}
    dirs = directions(w)
    out["directions_sha256"] = sha(dirs)
    ctrl, rnd = w.items[cfg.control], w.items[cfg.control_random]
    zero = {k: torch.zeros_like(v) for k, v in w.net.W0.items()}
    r0 = read_item(w, R.Item("W0", "w0", ctrl.state, 0.0, False, ctrl.pair, ctrl.side), dirs, 0.0, zero)
    write(r0)
    miss = zero_misses(r0)
    log(f"[stage0 W0] misses {miss}")
    if miss:
        dump(STAGE0, out | {"misses": miss, "passed": False})
        return False
    rr = read_item(w, rnd, dirs, 0.0)
    floor = cfg.gap_floor_mult * max(gaps(rr))
    rr["gap_floor"] = floor  # its own incr_rel_ce were read with no floor (zero gaps still rejected)
    rc = read_item(w, ctrl, dirs, floor)
    tr = rc["adl_published_whole_window"]["train"]["exclude_pos"]
    lens, null = tr["lens_like_mean"], tr["random_diff"]["like_mean"]
    passed = lens > max(null)
    out |= {"gap_floor": floor, "gap_floor_rule": f"{cfg.gap_floor_mult} x the largest |CE_base - CE_ft| of {rnd.name}",
            "control": {"name": ctrl.name, "lens_like_mean": lens, "random_diff_like_mean": null, "random_diff_like_max": max(null),
                        "A_ours": rc["A_ours"], "A_ours_null_max": rc["A_ours_null_max"]},
            "control_random": {"name": rnd.name, "lens_like_mean": rr["adl_published_whole_window"]["train"]["exclude_pos"]["lens_like_mean"]},
            "misses": [] if passed else ["control's five-position mean incr_ce is not above all 64 random differences'"], "passed": passed}  # fmt: skip
    for r in (rr, rc):
        write(r)
    dump(STAGE0, out)
    log(f"[stage0 control] lens five-position mean incr_ce {lens:.6f} vs 64 random differences max {max(null):.6f} "
        f"(mean {statistics.fmean(null):.6f}); gap floor {floor:.3g}; A_ours {rc['A_ours']:.5f} null max {rc['A_ours_null_max']:.5f}; PASSED {passed}")  # fmt: skip
    return passed


def items(w: World, shard: tuple[int, int]) -> None:
    s0 = read(STAGE0)
    dirs = torch.load(DIRS)
    if sha(dirs) != s0["directions_sha256"]:
        raise SystemExit(f"{DIRS} is not stage 0's")
    todo = [it for it in list(w.items.values())[shard[0] :: shard[1]] if not (ITEMS / f"{fname(it.name)}.json").is_file()]
    t0 = time.time()
    for k, it in enumerate(todo):
        write(read_item(w, it, dirs, s0["gap_floor"]))
        log(f"ITEM {k + 1} of {len(todo)} DONE; projected minutes {(time.time() - t0) / 60 / (k + 1) * len(todo):.1f}")


EXPERIMENT = ""


def main(stage: str, shard: tuple[int, int]) -> int:
    global EXPERIMENT
    P = read(PRED)
    cfg, EXPERIMENT = config_from_settings(Config, P["settings"]), P["experiment"]
    if stage == "texts":
        texts(cfg)
        return 0
    require_filed(PRED)
    if stage == "items" and not (STAGE0.is_file() and read(STAGE0).get("passed") is True):
        raise SystemExit(f"{STAGE0} does not say passed: stage 0 runs first and alone")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as E264-267
    log(f"{EXPERIMENT} {stage} {shard[0]}/{shard[1]}; GPU {torch.cuda.get_device_name()}; settings {asdict(cfg)}")
    (LARGE / "items").mkdir(parents=True, exist_ok=True)
    w = World(cfg)
    code = 0
    if stage == "stage0":
        code = 0 if stage0(w) else 1
    else:
        items(w, shard)
    log(f"FINISHED {stage}; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["texts", "stage0", "items"])
    ap.add_argument("part", nargs="?", default="0/1", help="items: I/N, every N-th item from the I-th")
    a = ap.parse_args()
    i, n = (int(x) for x in a.part.split("/"))
    code = 1
    try:
        code = main(a.stage, (i, n))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, flush=True)
    hard_exit(code)
