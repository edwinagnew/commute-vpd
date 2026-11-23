"""E282 (cell P1.58): the gain-only share. For a fine-tune F of the 67M M, fit the best update that only rescales M's own
variables, G = M with W' = sum_u g_u W_u, by Adam on the gains alone (every gain starting at 1) against KL(F || G) on
F's full next-token distribution at the task's read positions, on training rows (`commute.gain_fit`); then read, on
held-out rows, the share of F's trained-answer gain that G recovers and the share of KL(F || M) it removes. Two levels:
    N  one gain per MLP neuron (down_proj's input) and per attention head (o_proj's input): `commute.gains`, as E231
    V  one gain per VPD subcomponent of each of the 24 decomposed matrices and one per matrix remainder: `commute.vpd_gains`
Card experiments/gain_share/card.md; plan reports/plans/2026-11-23_plan_p1-58-gain-share.md; settings, thresholds and
rules in results/predictions.json, which the run refuses unless it is committed unchanged.

Stages, in order; each (state, level) writes results/stage<k>/<state>_<level>.json when it ends and a rerun skips it;
any miss writes the stage's file with "stopped" and exits 1; a stage refuses to start until the one before it is done:
    0     gains of 1 reproduce M's logits bit for bit at both levels and export exactly 0; the VPD remainder identity;
          a random gain state's export reproduces its hooked logits and a shuffled one does not; a do-nothing F (= M)
          fitted for two steps reads KL(F || G*) = 0 with every gain still 1
    1     the controls (E214's x2, x10, new_few, new_thin and E242's KL-matched random, built by E242's World), each
          F first checked against its saved KL(F || M); stops when x2 or x10 at N reads under n_min
    2     E62's copying map and lookup, seeds 0-2, on E62's own windows (train split to fit, read split to read), each
          F's gain first checked against E231's reading of the same state on the same rows
    3     E280's head-(2,4) edits built as definition-lead's experiments/head_edit/run.py builds them (v_proj rows, o_proj
          input columns or q_proj rows of the head times 2), on E62's copy rows; needs stage 1
    list  a manifest of further states on E62's rows, e.g. E280's head-(2,4) edits:
          [{"state": name, "dw": repo path of a 24-matrix weight change, "rows": "copy" | "lookup"}, ...]
Scoring and the figure: plot.py (no GPU).

    python experiments/gain_share/run.py all                     (stages 0, 1, 2, 3)
    python experiments/gain_share/run.py list --manifest PATH    (needs stage 1 done)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import gain_fit as GF
from commute import steering as ST
from commute import vpd_gains as VG
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # World: E214's built edits, E242's KL-matched random
KL = E242.E226.KL  # E214's run module: World, LAST, ALL, P (rule_retention.run: COPY24)
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", EXPERIMENTS.parent
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    controls: tuple[str, ...]
    random: str
    kl_saved: dict[str, str]  # control -> the saved item JSON whose kl_fm (or kl_target, for the random) F must repeat
    e62_states: tuple[str, ...]
    e62_dir: str
    e231_items: str
    do_nothing_rows: str
    levels: tuple[str, ...]
    lr: float
    steps: int
    warmup: int
    batch: int
    grad_clip: float
    check_every: int
    read_batch: int
    cue_split_seed: int
    fit_seed: int
    rand_sd: float
    rand_seed: int
    stage0_steps: int
    head_edits: dict[str, list[Any]]  # E280's edits: name -> [matrix, head axis (0 rows, 1 input columns)]
    head_layer: int
    head: int
    head_factor: float

    def __post_init__(self) -> None:
        assert self.random in self.controls and set(self.kl_saved) == set(self.controls) and self.levels == ("N", "V")
        assert {s.rsplit("_s", 1)[0] for s in self.e62_states} == {"copy", "lookup"} and self.do_nothing_rows in ("copy", "lookup")
        assert 0 < self.warmup < self.steps and self.steps % self.check_every == 0 and 0 < self.stage0_steps < self.warmup

    @property
    def fit(self) -> GF.FitSettings:
        return GF.FitSettings(self.lr, self.steps, self.warmup, self.batch, self.grad_clip, self.check_every, self.read_batch)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def fname(x: str) -> str:
    return x.replace("/", "-")


def dump(path: Path, obj: dict[str, Any]) -> None:
    write_stage(path, jsonable(obj))


def stop(path: Path, obj: dict[str, Any], why: str) -> None:
    dump(path, obj | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


def done(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("done"))


def need(stage: str) -> None:
    assert done(RESULTS / f"{stage}.json"), f"results/{stage}.json is not done: run that stage first"


# ---- states: F and its rows ------------------------------------------------------------------------------------------


@dataclass
class State:
    name: str
    F: Any  # the fine-tuned net (a LlamaSimpleMLP), eval, frozen
    rows: GF.Rows
    rec: dict[str, Any]


def dw_size(F: Any, W0: dict[str, Tensor]) -> float:
    return math.sqrt(sum(float((F.get_submodule(k).weight.float() - w).pow(2).sum()) for k, w in W0.items()) / sum(float(w.pow(2).sum()) for w in W0.values()))


def control(w: Any, name: str, cfg: Config) -> tuple[State, Tensor]:
    """E214's / E242's item and its whole D: x2, x10 on E195's copy24 (fit on its 301 train windows, read on the 75
    held-out = D, KL at every position, the answer the copy table's at `half`); the cue items on E214's 512 cue windows
    D, split 256 / 256 by a seeded permutation (KL and the answer at the last position)."""
    F, M, toks, at, rec = w.make(name)
    assert not F.extra and all(torch.equal(p, dict(M.net.named_parameters())[n]) for n, p in F.net.named_parameters() if n.removesuffix(".weight") not in w.keys)
    if name in ("x2", "x10"):
        c = torch.load(KL.P.COPY24, weights_only=False)
        assert torch.equal(toks.cpu(), c["heldout"].tokens) and at == KL.ALL
        h = c["heldout"].half
        rows = GF.Rows(c["train"].tokens, c["heldout"].tokens, KL.ALL, h, c["table"][c["heldout"].tokens[:, h]])
    else:
        assert at == KL.LAST
        toks, perm = toks.cpu(), torch.randperm(len(toks), generator=torch.Generator().manual_seed(cfg.cue_split_seed))
        tr, he = perm[: len(toks) // 2], perm[len(toks) // 2 :]
        cues = w.kl.cues
        tgt = None if name == cfg.random else torch.tensor(cues.y)[cues.cue.cpu()][he]
        rows = GF.Rows(toks[tr], toks[he], KL.LAST, toks.shape[1] - 1, tgt)
    return State(name, F.net.eval().requires_grad_(False), rows, {k: v for k, v in rec.items() if isinstance(v, (int, float, str))}), toks


def e62_rows(windows: Path, m: str) -> GF.Rows:
    """E62's own windows: the train split to fit, the read split to read, both fed whole as E62 and E231 fed them
    (41 tokens; 800 train rows, 200 read rows); KL and the answer (map m's target) at `half` (20)."""
    win = torch.load(windows, weights_only=False)
    h = win["read"].half
    assert win["train"].half == h
    return GF.Rows(win["train"].tokens, win["read"].tokens, slice(h, h + 1), h, win["target"][m]["read"])


def saved_dw(net: ST.Net, path: Path) -> dict[str, Tensor]:
    """A saved weight change of the 24 decomposed matrices (fp32 or bf16, keyed by module), fp32."""
    dw = {k: v.float() for k, v in torch.load(path, weights_only=True, map_location="cpu").items()}
    assert sorted(dw) == sorted(net.W0), f"{path}: not the 24 decomposed matrices"
    return dw


def head_dw(net: ST.Net, cfg: Config, name: str) -> dict[str, Tensor]:
    """E280's built edit (experiments/head_edit/run.py `edited`): head `head` of layer `head_layer`'s slice of one
    matrix (q/v rows, o input columns, head_dim wide) times head_factor, every other weight exactly M's."""
    mat, axis = cfg.head_edits[name]
    k = f"h.{cfg.head_layer}.attn.{mat}"
    W, d = net.W0[k], torch.zeros_like(net.W0[k])
    hd = W.shape[axis] // net.net.h[cfg.head_layer].attn.n_head  # 128
    cols = slice(cfg.head * hd, (cfg.head + 1) * hd)
    if axis == 0:
        d[cols] = W[cols] * (cfg.head_factor - 1)
    else:
        d[:, cols] = W[:, cols] * (cfg.head_factor - 1)
    return {k: d}


def e62(net: ST.Net, name: str, dw: dict[str, Tensor], rows: GF.Rows, rec: dict[str, Any]) -> State:
    """F = M + a weight change of some of the 24 decomposed matrices, read on E62's rows."""
    F = copy.deepcopy(net.net).eval().requires_grad_(False)
    with torch.no_grad():
        for k, d in dw.items():
            F.get_submodule(k).weight.add_(d.to(net.device))
    return State(name, F, rows, rec | {"dw_rel": dw_size(F, net.W0)})


# ---- one (state, level): fit, read, write ------------------------------------------------------------------------------


def readings(net: ST.Net, lv: GF.Level, st: State, g: Any, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    r = st.rows
    held = GF.read(net, st.F, r.held, r, cfg.read_batch, r.tgt, g, lv)
    train = GF.read(net, st.F, r.train, r, cfg.read_batch, None, g, lv)
    out: dict[str, Any] = held | {"train_kl_fg": train["kl_fg"], "train_kl_fm": train["kl_fm"]}
    out["share"] = held["gain_G"] / held["gain_F"] if r.tgt is not None and held["gain_F"] >= th["min_gain"] else None
    out["kl_share"] = 1 - held["kl_fg"] / held["kl_fm"] if held["kl_fm"] > th["kl_min"] else None
    return out


def one(net: ST.Net, lv: GF.Level, st: State, cfg: Config, th: dict[str, Any], path: Path) -> dict[str, Any]:
    if done(path):
        log(f"{st.name} {lv.name}: read before, skipped")
        return json.loads(path.read_text())
    t0, gpu = time.time(), torch.cuda.is_available()  # no GPU only in the CPU dry run on a toy net
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    g, curve = GF.fit(net, lv, st.F, st.rows, cfg.fit, cfg.steps, cfg.fit_seed, log, st.name)
    fit_s = time.time() - t0
    rd = readings(net, lv, st, g, cfg, th)
    gfile = RESULTS / "large" / "gains" / f"{path.stem}.pt"
    gfile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(g.state(), gfile)
    last = [c["train_kl"] for c in curve[-2:]]
    rec = {"experiment": EXPERIMENT, "state": st.name, "level": lv.name, "n_gains": sum(p.numel() for p in g.params()), "n_train": len(st.rows.train),
           "n_held": len(st.rows.held), **rd, "curve": curve, "flat_rel": abs(last[0] - last[1]) / max(last[0], 1e-12),
           "fit_seconds": fit_s, "s_per_step": fit_s / cfg.steps, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if gpu else None,
           "gains_file": str(gfile.relative_to(RESULTS)), "state_rec": st.rec, "done": True}  # fmt: skip
    dump(path, rec)
    log(f"READ {st.name} {lv.name}: share {rd['share']} kl_share {rd['kl_share']} KL(F||G*) {rd['kl_fg']:.5g} KL(F||M) {rd['kl_fm']:.5g} "
        f"gain F {rd.get('gain_F')} G {rd.get('gain_G')} train KL {rd['train_kl_fg']:.5g}; fit {fit_s:.0f} s ({fit_s / cfg.steps:.3f} s/step)")  # fmt: skip
    return rec


# ---- stages -------------------------------------------------------------------------------------------------------------


def level_checks(net: ST.Net, lv: GF.Level, dn: State, t: Tensor, base: Tensor, gen: torch.Generator, cfg: Config) -> dict[str, Any]:
    """One level's stage-0 readings: gains of 1 against M's logits `base` on rows t and their export; a random gain
    state's export (W0 + delta through net.at) against its hooked logit change at the read position, and the same with
    its gains shuffled; the do-nothing's short fit and read."""
    ones, rnd, pos = lv.ones(), lv.ones(), dn.rows.pos
    with torch.no_grad(), lv.applied(ones):
        same = torch.equal(GF.logits(net.net, t), base)
    for p in rnd.params():
        p.add_(cfg.rand_sd * torch.randn(p.shape, generator=gen).to(p.device))
    with torch.no_grad(), lv.applied(rnd):
        hooked = GF.logits(net.net, t)[:, pos] - base[:, pos]

    @torch.no_grad()
    def rel(g: Any) -> float:
        with net.at(lv.delta(g)):
            return float((GF.logits(net.net, t)[:, pos] - base[:, pos] - hooked).norm() / hooked.norm())

    g_dn, _ = GF.fit(net, lv, dn.F, dn.rows, cfg.fit, cfg.stage0_steps, cfg.fit_seed, log, dn.name)
    rd = GF.read(net, dn.F, dn.rows.held, dn.rows, cfg.read_batch, dn.rows.tgt, g_dn, lv)
    return {"ones_bit_equal": same, "delta_nonzero": sum(int(v.count_nonzero()) for v in lv.delta(ones).values()), "export_rel": rel(rnd),
            "shuffled_rel": rel(lv.shuffled(rnd, gen)), "do_nothing": rd, "do_nothing_max_gain_move": max(float((p - 1).abs().max()) for p in g_dn.params())}  # fmt: skip


def stage0(w: Any, lvs: dict[str, GF.Level], b: VG.Basis, cfg: Config, th: dict[str, Any]) -> None:
    path = RESULTS / "stage0.json"
    if done(path):
        return log("stage 0 done before, skipped")
    net, out = w.kl.net, dict[str, Any](experiment=EXPERIMENT)
    dn = State("do_nothing", copy.deepcopy(net.net).eval().requires_grad_(False), e62_rows(REPO / cfg.e62_dir / "windows.pt", cfg.do_nothing_rows), {})
    t = dn.rows.held[: cfg.read_batch].to(net.device)
    base, gen = GF.logits(net.net, t), torch.Generator().manual_seed(cfg.rand_seed)
    ident = b.identity_rel(net.W0)
    authors = {k: float((b.R[k] - w.delta[k].float().to(b.R[k].device)).norm() / net.W0[k].norm()) for k in b.keys}
    out["remainder"] = {"identity_rel_max": max(ident.values()), "vs_authors_rel_max": max(authors.values()), "vs_authors_rel": authors}
    miss = [f"remainder identity {out['remainder']}"] if not max(*ident.values(), *authors.values()) <= th["remainder_tol"] else []
    for name, lv in lvs.items():
        out[name] = o = level_checks(net, lv, dn, t, base, gen, cfg)
        if not (o["ones_bit_equal"] and o["delta_nonzero"] == 0 and o["do_nothing_max_gain_move"] == 0
                and all(o["do_nothing"][k] == 0 for k in ("kl_fm", "gain_F", "kl_fg"))):  # fmt: skip
            miss.append(f"{name}: gains of 1 or the do-nothing change the model: {o}")
        if not (o["export_rel"] <= th["equiv_tol"] and o["shuffled_rel"] >= th["perm_min"]):
            miss.append(f"{name}: the exported change is not the gain state: export {o['export_rel']:.3g} shuffled {o['shuffled_rel']:.3g}")
    log(f"[stage 0] {json.dumps(jsonable(out))}")
    if miss:
        stop(path, out, "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def stage1(w: Any, lvs: dict[str, GF.Level], cfg: Config, th: dict[str, Any]) -> None:
    need("stage0")
    path, out = RESULTS / "stage1.json", dict[str, Any](experiment=EXPERIMENT, states={})
    for name in cfg.controls:
        st, D = control(w, name, cfg)
        kl = GF.read(w.kl.net, st.F, D, st.rows, cfg.read_batch, None)["kl_fm"]
        saved = json.loads((REPO / cfg.kl_saved[name]).read_text())
        want, tol = (saved["kl_target"], th["random_target_rel"]) if name == cfg.random else (saved["kl_fm"], th["kl_repro_rel"])
        st.rec |= {"kl_fm_D": kl, "kl_saved": want, "kl_rel": abs(kl / want - 1), "dw_rel": dw_size(st.F, w.kl.net.W0)}
        log(f"{name}: KL(F || M) on all of D {kl:.9g}, saved {want:.9g} (rel {st.rec['kl_rel']:.3g}); |dW|/|W0| {st.rec['dw_rel']:.4g}")
        if not st.rec["kl_rel"] <= tol:
            stop(path, out, f"{name}: F's KL(F || M) {kl:.9g} does not repeat the saved {want:.9g} within {tol}")
        out["states"][name] = {lv: one(w.kl.net, lvs[lv], st, cfg, th, RESULTS / "stage1" / f"{fname(name)}_{lv}.json") for lv in cfg.levels}
        del st
        torch.cuda.empty_cache()
        rd = out["states"][name]["N"]
        got = [v for v in (rd["share"], rd["kl_share"]) if v is not None]
        if name in ("x2", "x10") and not (got and min(got) >= th["n_min"]):  # exactly representable at N: the optimiser's check
            stop(path, out, f"{name} at N reads share {rd['share']} kl_share {rd['kl_share']}, under {th['n_min']}: the fit is not finding the best gain")
    dump(path, out | {"done": True})
    log("STAGE 1 DONE")


def stage_e62(w: Any, lvs: dict[str, GF.Level], cfg: Config, th: dict[str, Any], states: list[tuple[str, Any, str]], tag: str) -> None:
    """States on E62's rows (stage 2: E62's own six; stage 3: E280's head edits; list: a manifest's), each given as
    (name, a function giving its weight change, the map whose rows read it); each F's gain checked against E231's
    reading of the same state where E231 read it."""
    need("stage1")
    path, out = RESULTS / f"{tag}.json", dict[str, Any](experiment=EXPERIMENT, states={})
    e231, rows = REPO / cfg.e231_items, {m: e62_rows(REPO / cfg.e62_dir / "windows.pt", m) for m in ("copy", "lookup")}
    for name, dw, m in states:
        files = {lv: RESULTS / tag / f"{fname(name)}_{lv}.json" for lv in cfg.levels}
        if all(done(p) for p in files.values()):
            out["states"][name] = {lv: json.loads(p.read_text()) for lv, p in files.items()}
            log(f"{name}: read before, skipped")
            continue
        st = e62(w.kl.net, name, dw(), rows[m], {"rows": m})
        f = GF.read(w.kl.net, st.F, st.rows.held, st.rows, cfg.read_batch, st.rows.tgt)
        ref = e231 / f"e62_{name.removeprefix('e62/')}.json"
        if ref.is_file():  # E62's own states: E231 read their gain on the same rows
            want = json.loads(ref.read_text())["refs"][f"e62/{m}"]["states"][name]["gain"]
            st.rec |= {"gain_F_e231": want, "gain_rel": abs(f["gain_F"] / want - 1)}
            log(f"{name}: F's gain {f['gain_F']:.9g}, E231's {want:.9g} (rel {st.rec['gain_rel']:.3g})")
            if not st.rec["gain_rel"] <= th["gain_repro_rel"]:
                stop(path, out, f"{name}: F's gain {f['gain_F']:.9g} does not repeat E231's {want:.9g} within {th['gain_repro_rel']}")
        out["states"][name] = {lv: one(w.kl.net, lvs[lv], st, cfg, th, p) for lv, p in files.items()}
        del st
        torch.cuda.empty_cache()
    dump(path, out | {"done": True})
    log(f"{tag.upper()} DONE")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("all", "0", "1", "2", "3", "list"))
    ap.add_argument("--manifest", type=Path)
    a = ap.parse_args()
    require_filed(PRED)
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 reads, as E231
    log(f"{EXPERIMENT} {a.stage} start; settings {asdict(cfg)}; thresholds {th}")
    KL.RESULTS = RESULTS / "e214"  # E214's World writes its edits' record there, not into E214's own results
    w = E242.World(E242.load_config()[0])
    b = VG.Basis.of(w.kl.net.W0, w.uv, w.kl.net.device)
    lvs = GF.levels(w.kl.net, b)
    if a.stage in ("all", "0"):
        stage0(w, lvs, b, cfg, th)
    if a.stage in ("all", "1"):
        stage1(w, lvs, cfg, th)
    net = w.kl.net
    if a.stage in ("all", "2"):
        todo = [(f"e62/{s}", lambda p=REPO / cfg.e62_dir / f"{s}.pt": saved_dw(net, p), s.rsplit("_s", 1)[0]) for s in cfg.e62_states]
        stage_e62(w, lvs, cfg, th, todo, "stage2")
    if a.stage in ("all", "3"):
        todo = [(f"e280/{e}", lambda e=e: head_dw(net, cfg, e), "copy") for e in cfg.head_edits]
        stage_e62(w, lvs, cfg, th, todo, "stage3")
    if a.stage == "list":
        assert a.manifest is not None, "list needs --manifest"
        todo = [(e["state"], lambda p=REPO / e["dw"]: saved_dw(net, p), e["rows"]) for e in json.loads(a.manifest.read_text())]
        stage_e62(w, lvs, cfg, th, todo, "list")
    log(f"{EXPERIMENT} {a.stage} DONE")
    return 0


if __name__ == "__main__":
    hard_exit(main())
