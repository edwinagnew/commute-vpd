"""cancellation (E17): do the per-subcomponent changes of a split or of component training cancel each other?
Plan: reports/plans/2026-10-21_plan_cancellation.md (approved class 1; the named next step after E15).

States, all from E11's saved files (experiments/reading_a_coverage/results/large/<task>/): component training's step-10
state on six tasks, and for emoticon the least-squares split and the scaled adjoint split rebuilt from the fine-tune's
delta-W, plus the positive control C1 (component training's emoticon pieces gauge-mixed so the same weight change is written
as cancelling per-subcomponent changes).
Per state, on the first `rows` Pile validation rows: the coherence of the changes and its sign-flip null (CPU), the
adversary's sources and KL (gamma.pgd_sources, as the harness's adversary), and the revert KLs through `cancel.Reverter`:
every change reverted (F_full), a random half (F_half, several halves), the fraction the adversary's mask removes
(F_adv), the same with the sources permuted (F_perm) and with W0's adversary's sources (F_w0adv), and the state's KL to
W0. Gates G1 (the hook reproduces the weight edit) and G2 (the coherence null reads 1) are checked per state, G1 first.

  run.py --predictions        file results/predictions.json (once)
  run.py all [--smoke]        -> results/cancellation[_smoke].json (written after every state), results/large/full[_smoke].log
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt}; E11's results/large/<task>/ (cluster NFS).
C1 (2026-10-21, before the run): gauge mixing replaced a write-side zero-sum perturbation, which cannot exist when a
matrix's read vectors are linearly independent (n < d_in for most matrices).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
E11 = HERE.parent / "reading_a_coverage" / "results" / "large"
EMOTICON = "emoticon"
TASKS = (EMOTICON, "r0_t352_v310", "r1_t534_v369", "r1_t588_v452", "r2_t275_v2667", "r2_t359_v816")
FAILING = tuple(f"ct {t}" for t in ("r0_t352_v310", "r1_t534_v369", "r2_t275_v2667"))
SCORED = (*FAILING, f"ls {EMOTICON}")  # the states E11, E2, E14 and E16 found failing the adversary
ADJOINT = f"adjoint {EMOTICON}"
CONTROL = f"c1 {EMOTICON}"
TOL = 1e-6  # the split's SVD tolerance (every experiment since round 2)
STEP = 10  # E11's scored step


@dataclass(frozen=True)
class Sizes:
    tasks: tuple[str, ...]
    rows: int  # Pile val rows 0..rows, one adversary over all of them (the harness's 128-row block)
    batch: int
    pgd_steps: int
    n_halves: int
    n_perms: int
    n_null: int
    control_factor: float  # C1: the mixed pieces' change energy over the unmixed, per unit of max(1, kappa)
    suffix: str

    def __post_init__(self) -> None:
        assert self.rows % self.batch == 0 and EMOTICON in self.tasks and self.n_halves >= 1 and self.n_perms >= 1, self


FULL = Sizes(tasks=TASKS, rows=128, batch=8, pgd_steps=20, n_halves=3, n_perms=3, n_null=3, control_factor=20.0, suffix="")
SMOKE = Sizes(tasks=(EMOTICON, "r2_t275_v2667"), rows=8, batch=4, pgd_steps=2, n_halves=1, n_perms=1, n_null=3,
              control_factor=20.0, suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-21",
    "design": "reports/plans/2026-10-21_plan_cancellation.md (approved class 1: analysis of saved states; the named next step of "
              "reports/analyses/2026-10-20_analysis_gamma-refit.md)",
    "setup": {
        "states": f"E11's component training at step {STEP} (seed 0) on {', '.join(TASKS)}; the least-squares split "
                  f"(split.split_state) and the scaled adjoint split (split.adjoint_state, input moments over the emoticon "
                  f"training windows) of the emoticon fine-tune's step-{STEP} delta-W; C1 = emoticon component training with "
                  f"cancel.gauge_mixed: the same weight change in mixed pieces (U2 A, V2 A^-T), the change energy "
                  f"{FULL.control_factor} x max(1, kappa_ct) times the unmixed",
        "rows": f"Pile val rows 0..{FULL.rows} in batches of {FULL.batch}; fp32 forwards for every KL; the adversary is "
                f"gamma.pgd_sources ({FULL.pgd_steps} steps of 0.1, seed 0, one source per matrix over all rows, Delta under "
                "its own source) at W0 and at each state",
        "numbers": "kappa = ||sum Delta_c||^2 / sum ||Delta_c||^2 pooled over matrices and its sign-flip null; F_full, F_half "
                   f"({FULL.n_halves} random halves of the alive subcomponents per matrix), h = mean F_half / F_full; the "
                   "adversary's KL at W0 and at the state, X = their difference; F_adv with w_c(x) = (1 - g_c(x)) (1 - a_c); "
                   f"F_perm ({FULL.n_perms} permutations of a among the alive subcomponents of each matrix); F_w0adv; the "
                   "state's KL to W0",
        "blind_to": "Pile rows only, never the trigger; kappa is Frobenius; random halves miss a specific cancelling subset "
                    "(F_adv covers it); the hook removes changes only, so damage from masking W0's own subcomponents under "
                    "the new activations is not read: that is P3's null",
        "noise": f"spread over the {FULL.n_halves} halves and {FULL.n_perms} permutations; one adversary seed (E11: f = 0.075)",
    },
    "gates": [
        {"id": "G1_hook_identity", "rule": "per state: hook at weight 0 gives KL 0 (<= 1e-6) to the state's logits; at weight 1 "
                                           "its logits are within 1e-4 nats KL of the model with the changes reverted in the weights",
         "if_fail": "stop; the hook algebra is wrong"},
        {"id": "G2_coherence_null", "rule": "per state: the sign-flip null within 0.1 of 1 (mean of the draws)",
         "if_fail": "kappa is not reported for that state"},
    ],
    "predictions": [
        {"id": "C1_injected_cancellation", "arm": CONTROL, "scored": True,
         "pass": f"kappa(C1) <= kappa(ct {EMOTICON}) / 5 and h(C1) > 1", "null": "either fails: the statistics cannot read cancellation",
         "predicted": "pass: the mixing keeps sum Delta_c exactly while the energies grow by the factor, so kappa falls by it "
                      "(a code check) and a random half of the mixed changes leaves a remainder far larger than the full "
                      "revert's", "blind_to": "nothing about the real states"},
        {"id": "P1_coherence", "arm": list(SCORED), "scored": True, "pass": "kappa < 1 on every scored state",
         "null": "kappa >= 1 on some scored state: the changes add in the Frobenius norm",
         "predicted": "FAIL: Adam's sign-like steps on a low-rank gradient give every column of dU the same direction, and the "
                      "least-norm split over an overcomplete frame adds coherently; a fail with P2 passing means the "
                      "cancellation is functional (decisions.md: functional over Frobenius)",
         "blind_to": "function; a weight-space statistic"},
        {"id": "P2_half_revert", "arm": [*SCORED, ADJOINT], "scored": True,
         "pass": f"h > 1 on every scored state and h <= 0.5 on {ADJOINT}",
         "null": "h <= 1 on a scored state (coherent changes read about 0.25, independent ones 0.5) or h > 0.5 on the adjoint",
         "predicted": "pass, low confidence on the scored states: the CI-masked check costs little while the adversary's selective "
                      "mask costs a lot, which is what a cancelling set does to partial removals",
         "blind_to": "a specific cancelling subset (random halves)"},
        {"id": "P3_adversary_removes_cancelling_part", "arm": list(SCORED), "scored": True,
         "pass": "F_adv >= 3 x mean F_perm and F_adv >= 0.5 X on every scored state",
         "null": "F_adv about F_perm: the adversary's pattern over the changes is irrelevant and its excess comes from W0's "
                 "subcomponents under the new activations",
         "predicted": "pass", "blind_to": "damage from masking W0's own subcomponents (the hook keeps them whole)"},
    ],
    "reported_not_scored": f"h, F_adv, F_perm on the passing component-training states ({EMOTICON}, r1_t588_v452, r2_t359_v816: "
                           "the hypothesis says lower h than on the failing ones, ordering with E11's adversarial ratio); F_w0adv "
                           "(stage 0 says masks do not transfer, so about F_perm); the adversary's KL per state against E11's "
                           "ratios (different row count); F_full against the state's KL to W0 (equal for ct and adjoint)",
}


def log(msg: str, suffix: str = "") -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / f"full{suffix}.log").open("a") as f:
        f.write(line + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(type(o))


def dump(name: str, obj: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; predictions are not edited after filing (add a dated report instead)")
    dump(PRED.name, PREDICTIONS)
    print(f"filed {PRED}")


# ----------------------------------------------------------------------------- states


@dataclass
class Ctx:
    device: str
    model: Any
    W0: dict[str, Tensor]
    alive: dict[str, Tensor]
    uv: dict[str, tuple[Tensor, Tensor]]  # alive (U, V), theory layout, on the device
    batches: list[Tensor]
    S: Sizes


Update = tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor]]]  # (delta-W of the target weights, alive (U2, V2)) per matrix


def load(S: Sizes, device: str) -> Ctx:
    from commute import component_training, vpd
    from commute.data import batches, pile_rows

    model = vpd.load_component_model(device)
    alive = {k: v.cpu().bool() for k, v in torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"].items()}
    rows = pile_rows("val", 0, S.rows, lambda m: log(m, S.suffix))
    return Ctx(device, model, vpd.target_weights(model), alive, component_training.alive_uv(model, alive), batches(rows, S.batch), S)


def ct_state(task: str) -> Update:
    st = torch.load(E11 / task / "pullback_s0" / f"state_step{STEP}.pt", weights_only=False)
    return {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}


def emoticon_splits(ctx: Ctx) -> dict[str, Update]:
    """The least-squares and scaled adjoint states of the emoticon fine-tune's delta-W (as E14 and E16 built them)."""
    from commute import split
    from commute.data import batches

    dW = torch.load(E11 / EMOTICON / f"dW_ft_step{STEP}_s0.pt", weights_only=False)
    uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in ctx.uv.items()}
    t0 = time.time()
    b = split.bases(uv_cpu, TOL)
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    sigma = {k: v.cpu() for k, v in split.input_second_moments(ctx.model, batches(win["train"].tokens, 32), list(ctx.W0)).items()}
    log(f"bases and emoticon input moments in {time.time() - t0:.0f} s", ctx.S.suffix)
    ls = split.split_state(dW, uv_cpu, b)
    dW_adj, adj = split.adjoint_state(dW, uv_cpu, sigma)
    return {f"ls {EMOTICON}": ({k: v[0] for k, v in ls.items()}, {k: (v[1], v[2]) for k, v in ls.items()}),
            ADJOINT: (dW_adj, {k: (v[1], v[2]) for k, v in adj.items()})}


def pieces_of(ctx: Ctx, uv2: dict[str, tuple[Tensor, Tensor]]) -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor]]:
    return {k: (ctx.uv[k][0], ctx.uv[k][1], uv2[k][0].to(ctx.device), uv2[k][1].to(ctx.device)) for k in uv2}


# ----------------------------------------------------------------------------- one state


def measure(ctx: Ctx, name: str, dW: dict[str, Tensor], pieces: dict[str, Any], w0_sources: dict[str, Tensor]) -> dict[str, Any]:
    from commute import cancel, gamma, vpd
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    S, model, dev = ctx.S, ctx.model, ctx.device
    target = {k: ctx.W0[k] + dW[k].to(dev) for k in ctx.W0}
    comps = components_with(model, ctx.alive, {k: (p[2], p[3]) for k, p in pieces.items()})
    rec: dict[str, Any] = {"coherence": cancel.coherence(pieces, S.n_null)}
    t0 = time.time()
    with model_state(model, target=target, components=comps):
        cached = cancel.cache_state(model, ctx.batches, ctx.alive)
        rec["kl_to_w0"] = _kl_to_weights(ctx, cached, ctx.W0)
        sources, rec["pgd_kl"] = gamma.pgd_sources(model, ctx.batches, S.pgd_steps, seed=0)
        rev = cancel.Reverter(model, pieces)
        try:
            # G1: the hook at 0 is the state, at 1 the weights with every change reverted. The reference forward runs
            # through the same hooks at weight 0 (an exact identity): with the hooks live at weight 1 it reverted twice
            # (smoke job 28085, 2026-10-21, read 0.0085 nats).
            w_rev = {k: target[k] - cancel.signed_sum(*pieces[k]).float().to(dev) if k in pieces else target[k] for k in target}
            b, _ref, _ = cached[0]
            g1_full = rev.logits(b, 1.0)
            with model_state(model, target=w_rev):
                kl_edit = float(vpd.kl_per_position(rev.logits(b, 0.0), g1_full).mean())
            rec["G1"] = {"kl_zero": cancel.revert_kl(rev, cached[:1], lambda g: 0.0), "kl_full_vs_weights": kl_edit}
            rec["G1"]["ok"] = rec["G1"]["kl_zero"] <= 1e-6 and kl_edit <= 1e-4
            log(f"[{name}] G1 {rec['G1']}", S.suffix)
            if not rec["G1"]["ok"]:
                return rec
            rec["F_full"] = cancel.revert_kl(rev, cached, lambda g: 1.0)
            rec["F_half"] = [cancel.revert_kl(rev, cached, lambda g, i=i: cancel.half_weights(pieces, seed=i)) for i in range(S.n_halves)]
            rec["F_adv"] = cancel.revert_kl(rev, cached, cancel.adversary_weights(sources, ctx.alive, dev))
            rec["F_perm"] = [cancel.revert_kl(rev, cached, cancel.adversary_weights(sources, ctx.alive, dev, perm_seed=i))
                             for i in range(S.n_perms)]
            rec["F_w0adv"] = cancel.revert_kl(rev, cached, cancel.adversary_weights(w0_sources, ctx.alive, dev))
        finally:
            rev.remove()
    rec["h"] = sum(rec["F_half"]) / len(rec["F_half"]) / rec["F_full"] if rec["F_full"] > 0 else float("nan")
    rec["seconds"] = time.time() - t0
    c = rec["coherence"]
    log(f"[{name}] kappa {c['pooled']:.3f} (null {c['pooled_null']:.3f}) | KL to W0 {rec['kl_to_w0']:.4f} F_full {rec['F_full']:.4f} "
        f"F_half {[round(x, 4) for x in rec['F_half']]} h {rec['h']:.3f} | pgd {rec['pgd_kl']:.4f} F_adv {rec['F_adv']:.4f} "
        f"F_perm {[round(x, 4) for x in rec['F_perm']]} F_w0adv {rec['F_w0adv']:.4f} | {rec['seconds']:.0f} s", S.suffix)
    return rec


@torch.no_grad()
def _kl_to_weights(ctx: Ctx, cached: Any, weights: dict[str, Tensor]) -> float:
    """Mean KL per position between the cached state logits and the model at `weights` (its components untouched)."""
    from commute import vpd
    from commute.attribution import fwd
    from commute.faithfulness import model_state

    tot = n = 0.0
    with model_state(ctx.model, target=weights):
        for b, ref, _ in cached:
            tot += float(vpd.kl_per_position(ref, fwd(ctx.model.target_model, b)).double().sum())
            n += b.shape[0] * b.shape[1]
    return tot / n


# ----------------------------------------------------------------------------- scoring


def score(out: dict[str, Any]) -> dict[str, Any]:
    st = out["states"]

    g2 = {n: abs(st[n]["coherence"]["pooled_null"] - 1) <= 0.1 for n in st}  # kappa is unscored where its null fails

    def have(names: list[str], need_g2: bool = False) -> list[str]:
        return [n for n in names if n in st and "h" in st[n] and (g2[n] or not need_g2)]

    def verdict(names: list[str], ok: Any, need_g2: bool = False) -> dict[str, Any]:
        got = have(names, need_g2)
        if not got:
            return {"verdict": "not run"}
        per = {n: bool(ok(st[n])) for n in got}
        return {"verdict": "pass" if all(per.values()) else "fail", "per_state": per,
                "unscored": [n for n in names if n not in got]}

    sc: dict[str, Any] = {
        "G1_hook_identity": {n: st[n]["G1"]["ok"] for n in st if "G1" in st[n]},
        "G2_coherence_null": g2,
        "P1_coherence": verdict(list(SCORED), lambda r: r["coherence"]["pooled"] < 1, need_g2=True),
        "P2_half_revert": {"scored_states": verdict(list(SCORED), lambda r: r["h"] > 1),
                           "adjoint": verdict([ADJOINT], lambda r: r["h"] <= 0.5)},
        "P3_adversary_removes_cancelling_part": verdict(list(SCORED), lambda r: r["F_adv"] >= 3 * sum(r["F_perm"]) / len(r["F_perm"])
                                                        and r["F_adv"] >= 0.5 * (r["pgd_kl"] - out["W0"]["pgd_kl"])),
    }
    ct, c1 = f"ct {EMOTICON}", CONTROL
    if ct in st and c1 in st and "h" in st[c1] and g2[ct] and g2[c1]:
        sc["C1_injected_cancellation"] = {"kappa_ratio": st[c1]["coherence"]["pooled"] / st[ct]["coherence"]["pooled"], "h": st[c1]["h"],
                                          "verdict": "pass" if st[c1]["coherence"]["pooled"] <= st[ct]["coherence"]["pooled"] / 5
                                          and st[c1]["h"] > 1 else "fail"}
    p2 = sc["P2_half_revert"]
    p2["verdict"] = ("pass" if p2["scored_states"]["verdict"] == "pass" and p2["adjoint"]["verdict"] == "pass"
                     else "not run" if "not run" in (p2["scored_states"]["verdict"], p2["adjoint"]["verdict"]) else "fail")
    return sc


# ----------------------------------------------------------------------------- the run


def stage_all(args: argparse.Namespace) -> None:
    from commute import cancel, gamma

    assert PRED.is_file(), "file predictions first: run.py --predictions"
    S = SMOKE if args.smoke else FULL
    name = f"cancellation{S.suffix}.json"
    t0 = time.time()
    ctx = load(S, args.device)
    out: dict[str, Any] = {"sizes": asdict(S), "states": {}}
    w0_sources, kl0 = gamma.pgd_sources(ctx.model, ctx.batches, S.pgd_steps, seed=0)
    out["W0"] = {"pgd_kl": kl0}
    log(f"[W0] adversary KL {kl0:.4f} on {S.rows} rows ({time.time() - t0:.0f} s)", S.suffix)
    states: dict[str, Update] = {f"ct {t}": ct_state(t) for t in S.tasks}
    states.update(emoticon_splits(ctx))
    for sname, (dW, uv2) in states.items():
        rec = measure(ctx, sname, dW, pieces_of(ctx, uv2), w0_sources)
        out["states"][sname] = rec
        if not rec["G1"]["ok"]:
            dump(name, out)
            sys.exit(f"G1 failed on {sname}: {rec['G1']}")
        if sname == f"ct {EMOTICON}":
            ratio = S.control_factor * max(1.0, rec["coherence"]["pooled"])
            ctrl = cancel.gauge_mixed(pieces_of(ctx, uv2), ratio, seed=0)
            log(f"[{CONTROL}] mixed pieces at change energy {ratio:.1f}x the unmixed", S.suffix)
            out["states"][CONTROL] = measure(ctx, CONTROL, dW, ctrl, w0_sources)
        dump(name, out)
    out["score"] = score(out)
    out["seconds"] = time.time() - t0
    dump(name, out)
    log(f"score {json.dumps(out['score'], default=_json_default)}", S.suffix)
    log(f"done in {out['seconds']:.0f} s -> {RESULTS / name}", S.suffix)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.stage == "all":
        stage_all(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
