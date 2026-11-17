"""E197 (cell P1.26, run 1): the held-model reading of causal-rule retention, three readings, calibration, on the 67M
(plan reports/plans/2026-11-18_plan_held-model-retention.md, "Frozen specification", "Code checks", "Run 1", "Settled";
appendix *_appendix.md; card card.md; settings, tolerances and expected values in results/predictions.json, which the run
refuses to start without, committed unchanged). The measurement is commute.held_model. No bars and no verdicts: S per
reading per item, the reuse-minus-twin S per pair and seed, and its spread over seeds. The three readings are three
targets and are never combined.

Per item: F; on 16,384 windows of its fit half, the second moments at every site and the three readings' H; on 512
windows of its read half (E196's subset), KL(F || M) and per reading KL(F || H), S = 1 - KL(F || H) / KL(F || M) at the
last position, H's max relative logit error against F, and each site's local fit residual. Items whose embedding,
unembedding or final norm differ from M's (the parity twin and the fact, full states) also get S with F's values there,
a side number. Stages, cheapest first:
    0  the code checks, alone, on copy24's halves: do-nothing, gains-only per unit (random signed gains, units shuffled
       within each layer), gains-only per block (one gain per MLP block and per head), circuit-off (E195's 8 layer-1
       neurons zeroed). Each must meet its arithmetic expected value (predictions.json "code_checks"); exit 1 on any
       miss, with results/stage0.json holding "stopped". Skipped on a rerun once stage0.json says done.
    1  E195's rebuilt item (experiments/rule_retention_fixed/results/large/rebuilt_mlp.pt), on copy24's halves.
    2-6  one E196 pair per stage (PAIRS), reuse member then twin, per seed asked (0 to 3; seeds 2 and 3 are E192's
       experiments/curve_test/results/large/<item>.pt). Seeds 0 and 1 first repeat E184's gap within reload_tol.
Each item writes results/items/<item>.json and results/large/units/<item>.npz (readings 1 and 2: every unit's partner,
gain and error after the gain) when it finishes, and is skipped once its JSON is there; stage 0 and 1 items also save
reading 3's maps (bf16) to results/large/maps/<item>.pt. results/summary.json is rewritten after each item.
Inputs: see RUN.md. Exit 0 done, 1 a stop or refusal. The log is standard output:

    python experiments/held_model_retention/run.py [STAGE ...] [--seeds 0 1 2 3] 2>&1 | tee -a experiments/held_model_retention/results/run.log
    python experiments/held_model_retention/run.py --summary     (no GPU: the summary from the item files present)
    (on a pod: E120_PRED_SHA=<sha of the predictions file> python ...)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

import commute.variable_read as VR
from commute import held_model as HM
from commute import old_tasks as OT
from commute import rule_retention as R
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed

PR: Any = load_experiment("rule_retention_pairs", "run")  # E196: PAIRS, STATES, E184, NUMBERS, gap, subset
FX: Any = load_experiment("rule_retention_fixed", "run")  # E195: zeroed, trained, LARGE (rebuilt_mlp.pt)
P = PR.P  # E193: clone, chunks, log, dump, stop
CELL = "P1.26"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, ITEMS, LARGE = RESULTS / "predictions.json", RESULTS / "items", RESULTS / "large"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
CURVE = EXPERIMENTS / "curve_test" / "results" / "large"  # seeds 2 and 3 (E192)
log, dump, stop = P.log, P.dump, P.stop
Planted = dict[HM.Site, tuple[Tensor, Tensor]]  # reading 1's expected partner and gain per F unit


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    layer: int  # E195's circuit C and its rebuilt item
    C: tuple[int, ...]
    trained: tuple[int, ...]  # the rebuilt item's trained MLP layers
    control_task: str  # whose halves stages 0 and 1 use
    seeds: tuple[int, ...]
    n_fit: int  # fit-half windows per item, every position
    fit_seed: int
    n_read: int  # read-half windows per item (E196's subset), last position
    read_seed: int
    n_max: int  # E173's numbers (the parity twin and the fact)
    held_share: float
    gain_lo: float  # the gains-only checks' |gain| range, signs random
    gain_hi: float
    check_seed: int
    eval_batch: int  # the gap check
    sum_batch: int  # windows per batch of the second moments

    def __post_init__(self) -> None:
        assert len(set(self.C)) == len(self.C) and max(self.C) < R.D_MLP and self.layer not in self.trained and self.control_task in OT.TASKS
        assert set(self.seeds) <= {0, 1, 2, 3} and 0 < self.n_read and 0 < self.n_fit <= OT.DSizes().D // 2 and 0 < self.gain_lo < self.gain_hi


# ---- items --------------------------------------------------------------------------------------------------------


@dataclass
class Data:
    sets: VR.Sets
    x: VR.Xor2
    dcfg: OT.DSizes
    cache: dict[str, tuple[Tensor, Tensor]]

    def halves(self, name: str) -> tuple[Tensor, Tensor]:
        """(fit, read) halves as E173 and E184 built them (twins share them); every group is kept, 13 MB each."""
        key = "xor2" if name in PR.STATES else OT.TASKS[name].group
        if key not in self.cache:
            self.cache[key] = VR.xor2_windows(self.dcfg, self.sets, self.x) if key == "xor2" else OT.task_windows(OT.TASKS[name], self.sets, self.dcfg)[:2]
        return self.cache[key]


def windows(d: Data, name: str, cfg: Config, M: R.Model) -> tuple[Tensor, Tensor]:
    """A seeded subset of the fit half (n_fit) and E196's subset of the read half (n_read), on M's device."""
    fit, read = d.halves(name)
    pick = torch.randperm(fit.shape[0], generator=torch.Generator().manual_seed(cfg.fit_seed))[: cfg.n_fit]
    return fit[pick].to(M.net.wte.weight.device), PR.subset(read, cfg, M)


def fine_tune(net: Any, item: str) -> Any:
    name, s = item.rsplit("_s", 1)
    path = CURVE / f"{item}.pt" if int(s) >= 2 else Path(str(PR.STATES[name]).format(s)) if name in PR.STATES else OT.TASKS[name].states[int(s)]
    return VR.load_state(net, path, OT.clone_net) if name in PR.STATES else OT.old_f(net, path)


def rebuilt(M: R.Model, cfg: Config) -> R.Model:
    F = FX.zeroed(M, cfg)
    state = torch.load(FX.LARGE / "rebuilt_mlp.pt", weights_only=True)
    with torch.no_grad():
        for k, p in FX.trained(F, cfg).items():
            p.copy_(state[k])
    return F


# ---- reading one item ---------------------------------------------------------------------------------------------


def read(F: R.Model, M: R.Model, fit_toks: Tensor, read_toks: Tensor, cfg: Config, th: dict[str, Any], item: str) -> tuple[HM.Fit, dict[str, Any]]:
    t0 = time.time()
    fit = HM.fit(F, M, P.chunks(fit_toks, cfg.sum_batch))
    lF, lM = F.logits(read_toks), M.logits(read_toks)
    kl_fm, ends = HM.kl_last(lF, lM), HM.ends_differ(F, M)
    out: dict[str, Any] = {"kl_FM": kl_fm, "ends_differ": ends, "fit_seconds": time.time() - t0, "readings": {}}
    log(f"{item}: KL(F || M) {kl_fm:.6g} (last position, {read_toks.shape[0]} read windows); fit {out['fit_seconds']:.1f} s")
    for r in HM.READINGS:
        H = HM.held(M, fit.maps[r])
        lH = H.logits(read_toks)
        kl = HM.kl_last(lF, lH)
        res = {f"{k}{l}": v for (k, l), v in fit.residual[r].items()}
        rd: dict[str, Any] = {"kl_FH": kl, "S": HM.survival(kl, kl_fm, th["kl_floor"]), "logit_error": HM.logit_error(lH, lF), "residual": res}
        if ends:  # side number: F's embedding, unembedding and final norm in H; no gap reads it
            kle = HM.kl_last(lF, HM.with_ends(H, F).logits(read_toks))
            rd["side_F_ends"] = {"kl_FH": kle, "S": HM.survival(kle, kl_fm, th["kl_floor"])}
        out["readings"][r] = rd
        log(f"{item} reading {r}: S {rd['S']}, KL(F || H) {kl:.6g}, max relative logit error {rd['logit_error']:.3g}; "
            f"site residuals " + ", ".join(f"{k} {v:.4g}" for k, v in res.items()) + (f"; side (F's ends) S {rd['side_F_ends']['S']}" if ends else ""))
        del H, lH
    out["seconds"] = time.time() - t0
    return fit, out


def save_units(item: str, fit: HM.Fit, maps: bool) -> None:
    (LARGE / "units").mkdir(parents=True, exist_ok=True)
    arrays = {f"r{r}_{k}{l}_{f}": getattr(p, f).numpy().astype(dt) for r, ps in fit.pairs.items() for (k, l), p in ps.items()
              for f, dt in (("partner", np.int16), ("gain", np.float32), ("err", np.float16))}  # fmt: skip
    np.savez_compressed(LARGE / "units" / f"{item}.npz", **arrays)
    if maps:
        (LARGE / "maps").mkdir(parents=True, exist_ok=True)
        torch.save({f"{k}{l}": A.to(torch.bfloat16).cpu() for (k, l), A in fit.maps["3"].items()}, LARGE / "maps" / f"{item}.pt")


# ---- stage 0: the code checks -------------------------------------------------------------------------------------


def signed_gains(n: int, cfg: Config, gen: torch.Generator) -> Tensor:
    sign = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
    return sign * (cfg.gain_lo + (cfg.gain_hi - cfg.gain_lo) * torch.rand(n, generator=gen))


def identity(M: R.Model) -> Planted:
    return {(k, l): (torch.arange(n), torch.ones(n, dtype=torch.float64)) for l in range(len(M.net.h)) for k, n in (("mlp", R.D_MLP), ("attn", R.N_HEAD))}


@torch.no_grad()
def gains_unit(M: R.Model, cfg: Config) -> tuple[R.Model, Planted]:
    """F's unit i = M's unit perm[i] times g[i]: neurons (input row and output column) and heads (q, k, v rows and o_proj
    columns), shuffled within each layer."""
    F, gen, planted = P.clone(M), torch.Generator().manual_seed(cfg.check_seed), {}
    for l, (bF, bM) in enumerate(zip(F.net.h, M.net.h, strict=True)):
        perm, g = torch.randperm(R.D_MLP, generator=gen), signed_gains(R.D_MLP, cfg, gen)
        bF.mlp.c_fc.weight.copy_(bM.mlp.c_fc.weight[perm])
        bF.mlp.down_proj.weight.copy_(bM.mlp.down_proj.weight[:, perm] * g.to(bM.mlp.down_proj.weight.device))
        hp, hg = torch.randperm(R.N_HEAD, generator=gen), signed_gains(R.N_HEAD, cfg, gen)
        for name in ("q_proj", "k_proj", "v_proj"):
            getattr(bF.attn, name).weight.copy_(getattr(bM.attn, name).weight.view(R.N_HEAD, R.D_HEAD, -1)[hp].flatten(0, 1))
        o = bM.attn.o_proj.weight.view(-1, R.N_HEAD, R.D_HEAD)[:, hp] * hg.to(bM.attn.o_proj.weight.device)[None, :, None]
        bF.attn.o_proj.weight.copy_(o.flatten(1))
        planted[("mlp", l)], planted[("attn", l)] = (perm, g.double()), (hp, hg.double())
    return F, planted


@torch.no_grad()
def gains_block(M: R.Model, cfg: Config) -> tuple[R.Model, Planted]:
    """One gain per MLP block (every column of down_proj) and one per head (its o_proj columns)."""
    F, gen, planted = P.clone(M), torch.Generator().manual_seed(cfg.check_seed + 1), identity(M)
    for l, b in enumerate(F.net.h):
        g, hg = signed_gains(1, cfg, gen), signed_gains(R.N_HEAD, cfg, gen)
        b.mlp.down_proj.weight.mul_(float(g))
        b.attn.o_proj.weight.copy_((b.attn.o_proj.weight.view(-1, R.N_HEAD, R.D_HEAD) * hg.to(b.attn.o_proj.weight.device)[None, :, None]).flatten(1))
        planted[("mlp", l)] = (torch.arange(R.D_MLP), g.double().expand(R.D_MLP))
        planted[("attn", l)] = (torch.arange(R.N_HEAD), hg.double())
    return F, planted


def circuit_off(M: R.Model, cfg: Config) -> tuple[R.Model, Planted]:
    planted = identity(M)
    partner, gain = planted[("mlp", cfg.layer)]
    partner[list(cfg.C)], gain[list(cfg.C)] = -1, float("nan")  # zero units of F: left out of the assignment
    return FX.zeroed(M, cfg), planted


CHECKS: dict[str, Callable[[R.Model, Config], tuple[R.Model, Planted]]] = {
    "do_nothing": lambda M, _c: (P.clone(M), identity(M)), "gains_unit": gains_unit, "gains_block": gains_block, "circuit_off": circuit_off}  # fmt: skip


def misses(rd: dict[str, Any], fit: HM.Fit, planted: Planted, want: dict[str, Any], th: dict[str, Any]) -> list[str]:
    """Every filed expectation of one code check (predictions.json "code_checks") that the item misses."""
    out = [] if (rd["kl_FM"] > th["kl_floor"]) == want["kl_FM_above_floor"] else [f"KL(F || M) {rd['kl_FM']:.3g}"]
    out += [f"reading {r} logit error {rd['readings'][r]['logit_error']:.3g}" for r in want["H_reproduces_F"] if not rd["readings"][r]["logit_error"] <= th["logit_tol"]]
    for site, (partner, gain) in planted.items():
        p = fit.pairs["1"][site]
        out += [f"reading 1 {site} pairing differs"] if not torch.equal(p.partner, partner) else []
        on = partner >= 0
        dev = float((p.gain[on] - gain[on]).abs().max())
        out += [f"reading 1 {site} max |gain - planted| {dev:.3g}"] if not dev <= th["gain_tol"] else []
    return out


def stage0(M: R.Model, data: Data, cfg: Config, th: dict[str, Any], want: dict[str, Any]) -> dict[str, Any]:
    fit_toks, read_toks = windows(data, cfg.control_task, cfg, M)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "items": {}}
    bad = []
    for item, make in CHECKS.items():
        F, planted = make(M, cfg)
        fit, rd = read(F, M, fit_toks, read_toks, cfg, th, item)
        save_units(item, fit, maps=True)
        rd["misses"] = misses(rd, fit, planted, want[item], th)
        out["items"][item] = rd
        log(f"{item}: {'checks hold' if not rd['misses'] else 'MISSES ' + '; '.join(rd['misses'])}")
        bad += [f"{item}: {m}" for m in rd["misses"]]
        del F, fit
    if bad:
        stop(RESULTS / "stage0.json", out, "stage 0: " + "; ".join(bad))
    return out | {"done": True}


# ---- stages 1 to 6 ------------------------------------------------------------------------------------------------


def read_item(net: Any, M: R.Model, data: Data, cfg: Config, th: dict[str, Any], item: str) -> None:
    path = ITEMS / f"{item}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return
    ITEMS.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item}
    if item == "rebuilt":
        F, name = rebuilt(M, cfg), cfg.control_task
    else:
        Fnet, name = fine_tune(net, item), item.rsplit("_s", 1)[0]
        F, e184 = R.Model(Fnet.net), PR.E184 / f"{item}.json"
        out["gap"] = PR.gap(Fnet, net, data.halves(name)[1], cfg)
        out["gap_e184"] = json.loads(e184.read_text())["gap"] if e184.is_file() else None  # seeds 2 and 3: none recorded
        log(f"{item}: gap (mean TV, last position, whole read half) {out['gap']:.6f}, E184's {out['gap_e184']}")
        if out["gap_e184"] is not None and not abs(out["gap"] - out["gap_e184"]) <= th["reload_tol"]:
            stop(path.with_suffix(".stopped.json"), out, f"{item}: the gap does not repeat E184's within reload_tol")
    fit, rd = read(F, M, *windows(data, name, cfg, M), cfg, th, item)
    save_units(item, fit, maps=item == "rebuilt")
    dump(path, out | rd)
    log(f"{item}: done in {rd['seconds']:.0f} s")
    del F, fit
    torch.cuda.empty_cache()


def summary(cfg: Config) -> dict[str, Any]:
    """S per item and reading; per pair, reading and seed the reuse member's S minus the twin's; its spread over seeds."""
    items = {p.stem: json.loads(p.read_text()) for p in ITEMS.glob("*.json") if not p.stem.endswith(".stopped")}
    S = {k: {r: v["readings"][r]["S"] for r in HM.READINGS} for k, v in items.items()}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "cell": CELL, "S": S, "gaps": {}}
    for reuse, twin in PR.PAIRS:
        for r in HM.READINGS:
            by: dict[int, float] = {}
            for s in cfg.seeds:
                a, b = S.get(f"{reuse}_s{s}", {}).get(r), S.get(f"{twin}_s{s}", {}).get(r)
                if a is not None and b is not None:  # a missing item or a "no change" (None) member gives no gap
                    by[s] = a - b
            g = list(by.values())
            out["gaps"][f"{reuse}/{twin} reading {r}"] = {"by_seed": by, "n": len(g), "min": min(g, default=None), "max": max(g, default=None),
                                                         "sd": statistics.stdev(g) if len(g) > 1 else None}  # fmt: skip
    return out


def main(stages: list[int], seeds: list[int], summary_only: bool) -> int:
    pred = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    cfg = Config(**settings)
    th = pred["thresholds"]
    if summary_only:
        dump(RESULTS / "summary.json", summary(cfg))
        return 0
    from commute import vpd
    from commute.steering import Net

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 forwards, fp64 moments
    log(f"{EXPERIMENT} ({CELL}, run 1) start; stages {stages}, seeds {seeds}; settings {asdict(cfg)}; thresholds {th}")
    net = Net("cuda")
    net.net.requires_grad_(False)
    M, tok, dcfg = R.Model(net.net), vpd.tokenizer(), OT.DSizes()
    x = VR.make_numbers(tok, int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    if x.keys.tolist() != json.loads(PR.NUMBERS.read_text())["number_tokens"]:
        log("STOP: the number tokens are not E154's")
        return 1
    data = Data(VR.make_sets(dcfg, tok, log), x, dcfg, {})
    s0 = RESULTS / "stage0.json"
    if not (s0.is_file() and json.loads(s0.read_text()).get("done")):
        dump(s0, stage0(M, data, cfg, th, pred["code_checks"]))
    log("STAGE 0 DONE")
    for n in stages:
        for item in ["rebuilt"] if n == 1 else [f"{m}_s{s}" for s in seeds for m in PR.PAIRS[n - 2]]:
            read_item(net, M, data, cfg, th, item)
            dump(RESULTS / "summary.json", summary(cfg))
        log(f"STAGE {n} DONE")
    for k, v in summary(cfg)["gaps"].items():
        log(f"gap {k}: by seed {v['by_seed']}, sd {v['sd']}")
    log(f"{EXPERIMENT} DONE")
    return 0


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", type=int, choices=range(1, len(PR.PAIRS) + 2), help="1 the rebuilt item, 2-6 the pairs (default all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3], choices=(0, 1, 2, 3))
    ap.add_argument("--summary", action="store_true", help="only rewrite results/summary.json from the item files present")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stages or list(range(1, len(PR.PAIRS) + 2)), args.seeds, args.summary)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()
