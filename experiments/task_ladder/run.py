"""Task ladder (round 3): does the decomposition predict which post-training updates it can follow?

Plan: reports/2026-10-15_plan_task-ladder.md, amended by reports/2026-10-16_prediction_ladder-routes-amendment.md.
Trigger -> target tasks mined from Pile at graded baseline probability (rungs = decades), two per rung (highest and
lowest Gamma coverage), plus the emoticon task as the reference rung. Per task: W fine-tune (do-nothing reference),
pull-back (Adam on all alive U, V), gated-off pull-back (positive control), each evaluated with the task-local harness
and the route split (src/commute/local_faith.py) on the held-out windows.

Predictions are filed in results/predictions.json and results/predictions_amend1.json (the text, thresholds and
measurement rules live there); every stage refuses to start without both.

Stages (run from the repo root with the project venv, on a GPU):
  noise    seed-to-seed spread of the local scores and the route shares on the emoticon pull-back, the gated-on
           controls (G1b), the GPU repeat noise, and the Pile ratios on W0 -> results/noise.json
  mine     trigger stats over Pile train rows, candidates per rung, windows, coverage -> results/tasks.json,
           results/large/windows_<task>.pt
  ladder   every task's arms and measurements -> results/ladder.json, scored at the end
  all      noise, mine, ladder
  --smoke  any stage at the SMOKE sizes below; writes *_smoke.json

Inputs from earlier rounds: experiments/transport_test/results/large/{alive.pt, windows.pt}.
The version that produced the 2026-10-19 results is kept in results/large/code_as_ran/ (refactored afterwards, same
measurements).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
PRED_AMEND1 = RESULTS / "predictions_amend1.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"

HALF = 20
EMOTICON = "emoticon"
PB_LR = 1e-5
FT_LR = 3e-5
OFF_THRESHOLD = 0.01  # a component is "gated off" at the trigger if its mean g there is <= this
ON_THRESHOLD = 0.5  # "gated on" if its mean g there is >= this (the gated-on controls, noise stage)
RUNGS = (0, 1, 2)  # first batch: behaviours the model already has to a degree
PILE_RUNGS = (0, 2)  # Pile side-effect check at these rungs' high-coverage tasks
FAITH_ROWS = (0, 256)  # Pile val rows for the Pile harness, as rounds 1 and 2
LAST_MATRIX = "h.3.mlp.down_proj"  # feeds no decomposed matrix, so Gamma's inputs cannot change (G1b)
CONC_P = {"L01": 0.535, "L11": 0.635}  # concentration test, Adam-all step 30: CI-masked P(o) (G4)
CONC_N_HELDOUT = 73
DG_TOL = 1e-4  # G1b: max |g1 - g0| allowed for "the gates did not move"
COVERAGE_GAP = 0.15  # G2: a rung's two tasks must differ in coverage by at least this


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the full run and the smoke run, in one place."""

    rows: int  # Pile train rows mined
    n_triggers: int
    min_occurrences: int
    max_windows: int
    min_train: int
    min_heldout: int
    steps: int  # fine-tune and pull-back steps per task
    scored: int  # the scored step; also the length of the noise stage's pull-backs
    pb_save: tuple[int, ...]  # pull-back states saved and evaluated
    ft_checkpoints: tuple[int, ...]
    stoch_draws: int
    pgd_steps: int
    pile: bool  # run the Pile harness
    suffix: str

    def __post_init__(self) -> None:
        assert self.scored in self.pb_save and self.scored in self.ft_checkpoints, "the scored step must be saved"
        assert max(self.pb_save + self.ft_checkpoints) <= self.steps


FULL = Sizes(rows=32768, n_triggers=40, min_occurrences=400, max_windows=500, min_train=300, min_heldout=60, steps=30,
             scored=10, pb_save=(10, 30), ft_checkpoints=(1, 3, 10, 30), stoch_draws=4, pgd_steps=20, pile=True,
             suffix="")
SMOKE = Sizes(rows=64, n_triggers=2, min_occurrences=4, max_windows=8, min_train=2, min_heldout=1, steps=2, scored=2,
              pb_save=(2,), ft_checkpoints=(1, 2), stoch_draws=1, pgd_steps=2, pile=False, suffix="_smoke")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))
    return p


def require_predictions() -> dict[str, Any]:
    assert PRED.is_file(), f"file the predictions first ({PRED})"
    assert PRED_AMEND1.is_file(), f"file the amendment's predictions first ({PRED_AMEND1})"
    return json.loads(PRED.read_text())


# ----------------------------------------------------------------------------- shared pieces


def load_all(device: str) -> dict[str, Any]:
    from commute import pullback, vpd

    model = vpd.load_component_model(device)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    return {"device": device, "model": model, "W0": vpd.target_weights(model), "comps0": vpd.component_weights(model), "alive": alive,
            "uv": pullback.alive_uv(model, alive),
            "emoticon": {"train": win["train"], "heldout": win["heldout"], "target": o_token}}


def faith_blocks() -> list[list[torch.Tensor]]:
    from commute.data import batches, pile_rows

    rows = pile_rows("val", FAITH_ROWS[0], FAITH_ROWS[1], log)
    return [batches(rows[s : s + 128], 8) for s in range(0, rows.shape[0], 128)]


def pile_faith(model: Any, blocks: list[list[torch.Tensor]], seed: int = 0) -> dict[str, float]:
    """Round 2's Pile harness: one adversary per 128-row block, mean over blocks."""
    from commute.faithfulness import faithfulness

    per = [faithfulness(model, b, seed=seed, log=log) for b in blocks]
    return {k: sum(p[k] for p in per) / len(per) for k in per[0] if k != "pgd_rows"}


def arm_a_cfg(target: int, steps: int, checkpoints: tuple[int, ...], seed: int = 0) -> Any:
    from commute.finetune import ArmAConfig

    return ArmAConfig(o_token=target, lr=FT_LR, steps=steps, checkpoints=checkpoints, seed=seed)


def local(model: Any, heldout: Any, target: int, S: Sizes, seed: int = 0) -> dict[str, Any]:
    from commute.local_faith import local_faithfulness

    return local_faithfulness(model, heldout.tokens, heldout.half, target, n_stoch_draws=S.stoch_draws,
                              pgd_steps=S.pgd_steps, seed=seed, log=log)


def subset_mask(alive: dict[str, torch.Tensor], gbar: dict[str, torch.Tensor], keep: Any) -> dict[str, torch.Tensor]:
    """Bool mask over all components: alive AND keep(gbar_c), gbar_c the alive component's mean g at the trigger."""
    out = {}
    for k, a in alive.items():
        idx = torch.nonzero(a, as_tuple=True)[0]
        m = torch.zeros_like(a)
        m[idx[keep(gbar[k].to(idx.device))]] = True
        out[k] = m
    return out


def route_base_of(ctx: dict[str, Any], heldout: Any, target: int) -> dict[str, Any]:
    from commute.local_faith import route_cells

    return route_cells(ctx["model"], heldout.tokens, heldout.half, target)


def routes(ctx: dict[str, Any], heldout: Any, target: int, route_base: dict[str, Any], do_nothing: bool = False) -> dict[str, Any]:
    """Route split for the model's current state, with the G3 identities checked and logged."""
    from commute.local_faith import route_split

    sp = route_split(ctx["model"], heldout.tokens, heldout.half, target, route_base, ctx["comps0"])
    s = sp["shares"]
    ok = abs(sp["share_sum"] - 1) < 1e-6 if abs(sp["gain"]) >= 1e-6 else True
    if do_nothing and abs(sp["gain"]) >= 1e-6:
        # two separate forwards of the same masked model: allow 10 x the measured GPU repeat noise of the log-odds
        tol = max(1e-4, 10 * ctx.get("lo_repeat_noise", 0.0) / abs(sp["gain"]))
        ok = ok and abs(s["d"]) < tol and abs(s["i"]) < tol and abs(s["e_delta"] - 1) < tol
    sp["G3_ok"] = bool(ok)
    log(f"[routes{' do-nothing' if do_nothing else ''}] gain {sp['gain']:.3f} " + " ".join(f"{k} {v:.3f}" for k, v in s.items())
        + f" | sum {sp['share_sum']:.6f} max_dg {sp['max_dg']:.3g} | G3 {'ok' if ok else 'VIOLATED'}")
    return sp


def eval_state(ctx: dict[str, Any], mask: dict[str, torch.Tensor], state: dict[str, Any], heldout: Any, target: int,
               S: Sizes, seed: int = 0, route_base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Local harness with target weights W0 + dW and the columns in `mask` replaced by the state; with `route_base`
    (route_cells at W0) also the route split, under "routes"."""
    from commute import pullback
    from commute.faithfulness import model_state

    W0 = ctx["W0"]
    comps = pullback.components_with(ctx["model"], mask, {k: (state[k][1], state[k][2]) for k in state})
    tgt = {k: W0[k] + state[k][0].to(W0[k].device) for k in W0}
    with model_state(ctx["model"], target=tgt, components=comps):
        res = local(ctx["model"], heldout, target, S, seed)
        if route_base is not None:
            res["routes"] = routes(ctx, heldout, target, route_base)
    return res


def pullback_arm(ctx: dict[str, Any], mask: dict[str, torch.Tensor], train: Any, held: Any, target: int, steps: int,
                 save: tuple[int, ...], save_dir: Path, S: Sizes, base: dict[str, Any], rbase: dict[str, Any],
                 seed: int = 0) -> tuple[list[dict[str, Any]], dict[int, Any], dict[str, Any]]:
    """Train the components in `mask` (Delta and Gamma fixed) and evaluate every saved state (harness seed 0, route
    split). Returns the curve, the states, and {str(step): {"local", "retained", "routes"}}."""
    from commute import pullback
    from commute.local_faith import retained

    uv = ctx["uv"] if mask is ctx["alive"] else pullback.alive_uv(ctx["model"], mask)
    cfg = pullback.PullbackConfig(lr=PB_LR, steps=steps, save_steps=save)
    curve, states = pullback.train(ctx["model"], uv, train.tokens, held, arm_a_cfg(target, steps, (), seed), cfg,
                                   ctx["device"], ctx["W0"], save_dir=save_dir, log=log)
    evals = {}
    for step, st in states.items():
        res = eval_state(ctx, mask, st, held, target, S, route_base=rbase)
        sp = res.pop("routes")
        evals[str(step)] = {"local": res, "retained": retained(res, base), "routes": sp}
    return curve, states, evals


# ----------------------------------------------------------------------------- stage: noise (G0, G1b)


def gated_on_control(ctx: dict[str, Any], mask: dict[str, torch.Tensor], name: str, S: Sizes, base: dict[str, Any],
                     rbase: dict[str, Any]) -> dict[str, Any]:
    """A gated-on pull-back on the emoticon task, evaluated at the scored step (status set by the caller)."""
    emo = ctx["emoticon"]
    n = sum(int(m.sum()) for m in mask.values())
    if n == 0:
        return {"n_trainable": 0, "status": "inconclusive (no gated-on alive components)"}
    curve, _, evals = pullback_arm(ctx, mask, emo["train"], emo["heldout"], emo["target"], S.scored, (S.scored,),
                                   LARGE / EMOTICON / name, S, base, rbase)
    e = evals[str(S.scored)]
    return {"n_trainable": n, "curve": curve, "retained": e["retained"], "routes": e["routes"],
            "p_target": curve[-1]["heldout_p_o"]}


def stage_noise(S: Sizes, device: str) -> None:
    from commute import ladder
    from commute.faithfulness import model_state
    from commute.local_faith import retained

    require_predictions()
    ctx = load_all(device)
    emo = ctx["emoticon"]
    out: dict[str, Any] = {"pile_W0": {}, "local": {}}
    base = local(ctx["model"], emo["heldout"], emo["target"], S)
    rbase = route_base_of(ctx, emo["heldout"], emo["target"])
    rep = route_base_of(ctx, emo["heldout"], emo["target"])  # same state again: the GPU repeat noise
    out["gate_repeat_noise"] = max(float((a[k] - b[k]).abs().max()) for a, b in zip(rbase["gates"], rep["gates"]) for k in a)
    out["lo_repeat_noise"] = ctx["lo_repeat_noise"] = max(abs(rbase[c] - rep[c]) for c in ("T", "U", "L"))
    del rep
    log(f"noise: repeat noise at W0: gates {out['gate_repeat_noise']:.3g}, log-odds {out['lo_repeat_noise']:.3g}")
    # optimiser seeds 0 and 1 (harness seed 0, with routes), then harness seeds 1 and 2 on optimiser seed 0
    states = {}
    for opt_seed in (0, 1):
        curve, st, evals = pullback_arm(ctx, ctx["alive"], emo["train"], emo["heldout"], emo["target"], S.scored,
                                        (S.scored,), LARGE / EMOTICON / f"pullback_seed{opt_seed}", S, base, rbase,
                                        seed=opt_seed)
        states[opt_seed] = st[S.scored]
        e = evals[str(S.scored)]
        out["local"][f"curve_seed{opt_seed}"] = curve
        out["local"][f"retained_seed{opt_seed}_h0"] = e["retained"]
        out["local"][f"routes_seed{opt_seed}"] = e["routes"]
    for hseed in (1, 2):
        res = eval_state(ctx, ctx["alive"], states[0], emo["heldout"], emo["target"], S, seed=hseed)
        out["local"][f"retained_seed0_h{hseed}"] = retained(res, base)
    s0, s1 = out["local"]["routes_seed0"]["shares"], out["local"]["routes_seed1"]["shares"]
    d_opt_sh = {k: abs(s0[k] - s1[k]) for k in s0}
    d_opt_sh["r+i"] = abs((s0["r"] + s0["i"]) - (s1["r"] + s1["i"]))
    out["local"]["d_opt_shares"] = d_opt_sh
    out["m_d"] = max(0.15, 3 * d_opt_sh["d"])
    out["m_r"] = max(0.15, 3 * d_opt_sh["r+i"])
    r = [out["local"][f"retained_seed0_h{h}"] for h in (0, 1, 2)]
    out["local"]["harness_spread"] = {v: max(x[v] for x in r) - min(x[v] for x in r) for v in r[0]}
    d_opt = abs(out["local"]["retained_seed0_h0"]["ci_masked"] - out["local"]["retained_seed1_h0"]["ci_masked"])
    out["local"]["d_opt_ci_masked"] = d_opt
    out["P1_margin"] = max(0.15, 3 * d_opt)

    # G1b as amended: the gated-on pull-back on the last decomposed matrix only, where r = i = 0 exactly
    gbar = ladder.trigger_gbar(ctx["model"], ctx["alive"], emo["train"].tokens, HALF)["gbar"]
    on_mask = subset_mask(ctx["alive"], gbar, lambda g: g >= ON_THRESHOLD)
    last_mask = {k: (m if k == LAST_MATRIX else torch.zeros_like(m)) for k, m in on_mask.items()}
    g1b = gated_on_control(ctx, last_mask, "pullback_gated_on_last", S, base, rbase)
    g1b["matrix"] = LAST_MATRIX
    if "status" not in g1b:
        sh = g1b["routes"]["shares"]
        g1b["status"] = (f"FAIL (gates moved: max |g1 - g0| = {g1b['routes']['max_dg']:.3g})" if g1b["routes"]["max_dg"] > DG_TOL
                         else "inconclusive (did not learn)" if g1b["p_target"] < 0.5
                         else "pass" if sh["d"] >= 0.7 * (1 - sh["e_delta"]) else "FAIL")
    out["G1b_last"] = g1b
    # the original all-matrix gated-on arm: reported, not scored
    g1b_all = gated_on_control(ctx, on_mask, "pullback_gated_on", S, base, rbase)
    if "status" not in g1b_all:
        r_on = g1b_all["retained"]
        g1b_all["status"] = ("inconclusive (did not learn)" if g1b_all["p_target"] < 0.5
                             else "pass" if r_on["ci_masked"] >= 0.7 * r_on["unmasked"] else "FAIL")
    out["G1b"] = g1b_all
    for key in ("G1b_last", "G1b"):
        g = out[key]
        log(f"{key}: {g['status']} (n_trainable {g['n_trainable']}, P(target) {g.get('p_target')}, "
            f"shares {g.get('routes', {}).get('shares')})")

    if S.pile:
        blocks = faith_blocks()
        with model_state(ctx["model"]):
            per = {s: pile_faith(ctx["model"], blocks, seed=s) for s in (0, 1, 2)}
        out["pile_W0"] = {"per_seed": per, "spread_ratio": {k: max(p[k] for p in per.values()) / min(p[k] for p in per.values())
                                                             for k in per[0]}}
    dump(f"noise{S.suffix}.json", out)
    log(f"noise: harness spread {out['local']['harness_spread']}; d_opt(ci_masked) {d_opt:.4f}; P1 margin "
        f"{out['P1_margin']:.3f}; m_d {out['m_d']:.3f} m_r {out['m_r']:.3f}; Pile spread {out['pile_W0'].get('spread_ratio')}")


# ----------------------------------------------------------------------------- stage: mine


def stage_mine(S: Sizes, device: str) -> None:
    from commute import ladder, vpd
    from commute.data import pile_rows, split_windows
    from commute.finetune import logits_of

    require_predictions()
    ctx = load_all(device)
    model = ctx["model"]
    tok = vpd.tokenizer()
    rows = pile_rows("train", 0, S.rows, log)
    with torch.no_grad():
        vocab = logits_of(model.target_model, rows[:1].to(device)).shape[-1]  # the model's, not the tokenizer's
    plain = ladder.plain_mask(tok.decode, vocab)
    log(f"mine: logit dimension {vocab}; {int(plain.sum())} word-like tokens")
    counts = ladder.count_tokens(rows, HALF)
    frequent = sorted(((t, c) for t, c in counts.items() if c >= S.min_occurrences and t < len(plain) and bool(plain[t])),
                      key=lambda tc: -tc[1])
    triggers = [t for t, _ in frequent[: S.n_triggers]]
    log(f"mine: {len(frequent)} word-like triggers with >= {S.min_occurrences} occurrences; using {len(triggers)}")
    stats = ladder.trigger_stats(lambda toks: logits_of(model.target_model, toks), rows, triggers, HALF, batch=8,
                                 device=device, log=log)
    cands: list[dict[str, Any]] = []
    for t in triggers:
        per = ladder.candidates(stats, t, plain, per_rung=1, rungs=RUNGS)
        cands += [{"trigger": t, "target": v, "rung": r, "baseline": b, "trigger_str": tok.decode([t]),
                   "target_str": tok.decode([v]), "count": stats.count[t]} for r in RUNGS for v, b in per[r]]
    log(f"mine: {len(cands)} candidates over rungs {RUNGS}")
    LARGE.mkdir(parents=True, exist_ok=True)
    for c in cands:
        train, held = split_windows(ladder.windows_for_trigger(rows, c["trigger"], HALF, S.max_windows, seed=0))
        c["n_train"], c["n_heldout"] = len(train), len(held)
        if len(train) < S.min_train or len(held) < S.min_heldout:
            c["skipped"] = "too few windows"
            continue
        gb = ladder.trigger_gbar(model, ctx["alive"], train.tokens, HALF)
        G = ladder.trigger_gradient(model, train.tokens, HALF, c["target"])
        cov = ladder.coverage_from(G, ctx["uv"], gb["gbar"])
        c.update({"coverage": cov["coverage"], "coverage_per_matrix": cov["per_matrix"], "l0_trigger": gb["l0_trigger"],
                  "max_gbar": gb["max_gbar"], "name": f"r{c['rung']}_t{c['trigger']}_v{c['target']}"})
        torch.save({"train": train, "heldout": held, "gbar": gb["gbar"], "task": c}, LARGE / f"windows_{c['name']}.pt")
        log(f"mine: {c['name']} '{c['trigger_str']}' -> '{c['target_str']}' baseline {c['baseline']:.2e} coverage "
            f"{cov['coverage']:.3f} L0 {gb['l0_trigger']:.0f}")
    tasks: dict[str, Any] = {}
    for r in RUNGS:
        ok = [c for c in cands if c["rung"] == r and "coverage" in c]
        if len(ok) < 2:
            tasks[f"rung{r}"] = {"skipped": f"only {len(ok)} candidates with windows"}
            continue
        hi, lo = max(ok, key=lambda c: c["coverage"]), min(ok, key=lambda c: c["coverage"])
        tasks[f"rung{r}"] = {"high": hi["name"], "low": lo["name"], "coverage_gap": hi["coverage"] - lo["coverage"],
                             "n_candidates": len(ok)}
    emo = ctx["emoticon"]
    dump(f"tasks{S.suffix}.json", {"n_rows": S.rows, "triggers": triggers, "candidates": cands, "tasks": tasks,
                                   "emoticon": {"target": emo["target"], "n_train": len(emo["train"]),
                                                "n_heldout": len(emo["heldout"])}})
    log(f"mine: tasks {json.dumps(tasks)}")


# ----------------------------------------------------------------------------- stage: ladder


def run_task(ctx: dict[str, Any], name: str, train: Any, held: Any, target: int, gbar: dict[str, torch.Tensor] | None,
             S: Sizes, do_pile: bool) -> dict[str, Any]:
    from commute import ladder, pullback
    from commute.faithfulness import model_state
    from commute.finetune import finetune_arm_a
    from commute.local_faith import retained

    model, W0, alive = ctx["model"], ctx["W0"], ctx["alive"]
    t0 = time.time()
    base = local(model, held, target, S)
    rbase = route_base_of(ctx, held, target)
    rec: dict[str, Any] = {"name": name, "target": target, "n_train": len(train), "n_heldout": len(held), "base": base,
                           "baseline_window_p": float(torch.sigmoid(torch.tensor(base["target"]["lo_trigger"])))}
    # W fine-tune at the scored step, components unchanged: the do-nothing reference
    ft = finetune_arm_a(model, train, held, arm_a_cfg(target, S.steps, S.ft_checkpoints), ctx["device"], log)
    rec["finetune"] = {"evals": ft.evals, "log_tail": ft.log[-3:]}
    (LARGE / name).mkdir(parents=True, exist_ok=True)
    dW = ft.deltas[S.scored]
    torch.save(dW, LARGE / name / f"dW_ft_step{S.scored}.pt")
    with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device) for k in W0}):
        rec["do_nothing"] = {"local": local(model, held, target, S)}
        rec["do_nothing"]["routes"] = routes(ctx, held, target, rbase, do_nothing=True)
    rec["do_nothing"]["retained"] = retained(rec["do_nothing"]["local"], base)
    # pull-back on all alive
    curve, states, evals = pullback_arm(ctx, alive, train, held, target, S.steps, S.pb_save, LARGE / name / "pullback",
                                        S, base, rbase)
    rec["pullback"] = {"curve": curve, "steps": evals}
    if do_pile and S.pile:
        st = states[S.scored]
        comps = pullback.components_with(model, alive, {k: (st[k][1], st[k][2]) for k in st})
        blocks = faith_blocks()
        f0 = ctx.get("pile_W0")
        if f0 is None:
            with model_state(model):
                f0 = pile_faith(model, blocks)
        with model_state(model, target={k: W0[k] + st[k][0].to(W0[k].device) for k in W0}, components=comps):
            f1 = pile_faith(model, blocks)
        evals[str(S.scored)]["pile"] = {"W0": f0, "pullback": f1, "ratios": {k: f1[k] / f0[k] for k in f0}}
    # gated-off pull-back: the positive control (G1)
    if gbar is None:
        gbar = ladder.trigger_gbar(model, alive, train.tokens, HALF)["gbar"]
    assert gbar is not None
    off_mask = subset_mask(alive, gbar, lambda g: g <= OFF_THRESHOLD)
    off: dict[str, Any] = {"n_trainable": {k: int(m.sum()) for k, m in off_mask.items()}}
    if sum(off["n_trainable"].values()) == 0:
        off["skipped"] = "no gated-off alive components"
    else:
        curve_off, _, evals_off = pullback_arm(ctx, off_mask, train, held, target, S.steps, S.pb_save,
                                               LARGE / name / "pullback_gated_off", S, base, rbase)
        off.update({"curve": curve_off, "steps": evals_off})
    rec["gated_off"] = off
    rec["seconds"] = time.time() - t0
    pb = evals[str(S.scored)]["retained"]
    log(f"task {name}: pull-back@{S.scored} retained ci {pb['ci_masked']:.3f} unmasked {pb['unmasked']:.3f} stoch "
        f"{pb['stochastic']:.3f} pgd_t {pb['pgd_trigger']:.3f} | do-nothing ci {rec['do_nothing']['retained']['ci_masked']:.3f}"
        f" | {rec['seconds']:.0f} s")
    return rec


def compare_rungs(tasks: dict[str, Any], value: Any, excluded: set[str], margin: float, why_excluded: str) -> dict[str, Any]:
    """Per first-batch rung: pass if value(high) - value(low) >= margin. `value(name)` is None when not scorable."""
    out = {}
    for rung, t in tasks.items():
        if "high" not in t:
            continue
        hi, lo = value(t["high"]), value(t["low"])
        if t["coverage_gap"] < COVERAGE_GAP:
            out[rung] = {"status": f"excluded (coverage gap < {COVERAGE_GAP})", "gap": t["coverage_gap"]}
        elif t["high"] in excluded or t["low"] in excluded or hi is None or lo is None:
            out[rung] = {"status": f"excluded ({why_excluded})"}
        else:
            out[rung] = {"status": "pass" if hi - lo >= margin else "fail", "high_minus_low": hi - lo, "high": hi, "low": lo}
    return out


def tally(rungs: dict[str, Any]) -> tuple[int, int]:
    return sum(v["status"] == "pass" for v in rungs.values()), sum(v["status"] in ("pass", "fail") for v in rungs.values())


def score(out: dict[str, Any], tasks: dict[str, Any], noise: dict[str, Any], scored: int = FULL.scored) -> dict[str, Any]:
    """The filed rules (predictions.json, amended by predictions_amend1.json): G1 on d and e_delta, G1b on the
    last-matrix gated-on arm, G3 identities, G4 reproduction, then P1/P2 (on retained_ci_masked) and A1/A2."""
    step = str(scored)
    recs = out["tasks"]
    margin, m_d, m_r = noise["P1_margin"], noise["m_d"], noise["m_r"]

    def at(name: str, arm: str, s: str = step) -> dict[str, Any] | None:
        return recs.get(name, {}).get(arm, {}).get("steps", {}).get(s)

    def p_target(name: str, arm: str) -> float | None:
        c = recs.get(name, {}).get(arm, {}).get("curve")
        return None if not c else c[min(scored, len(c)) - 1]["heldout_p_o"]

    def ci_masked(name: str) -> float | None:
        a = at(name, "pullback")
        return None if a is None else a["retained"]["ci_masked"]

    def shares(name: str) -> dict[str, float] | None:
        """Pull-back shares at the scored step, if the task learned (P >= 0.5) with gain >= MIN_GAIN."""
        a, p = at(name, "pullback"), p_target(name, "pullback")
        return None if a is None or p is None or p < 0.5 or not a["routes"]["scorable"] else a["routes"]["shares"]

    # G1: the gated-off control must not show up as direct (d) or as missing from the components (e_delta)
    g1: dict[str, Any] = {}
    excluded: set[str] = set()
    for name in recs:
        a, p = at(name, "gated_off"), p_target(name, "gated_off")
        if a is None or p is None:
            g1[name] = {"status": "not run"}
            excluded.add(name)
        elif p < 0.5:
            g1[name] = {"status": "inconclusive (did not learn)", "p_target": p}
            excluded.add(name)
        else:
            sh, r = a["routes"]["shares"], a["retained"]
            g1[name] = {"status": "pass" if sh["d"] <= 0.2 and sh["e_delta"] <= 0.3 else "FAIL", "d": sh["d"],
                        "e_delta": sh["e_delta"],
                        "old_rule_reported": {"retained_ci_masked": r["ci_masked"], "retained_unmasked": r["unmasked"],
                                              "would_pass": r["ci_masked"] <= 0.2 and r["unmasked"] >= 0.7}}
    harness_separates = all(v["status"] != "FAIL" for v in g1.values())
    g1b = noise.get("G1b_last", {}).get("status", "not run")
    splits = [st.get("routes", {}) for rec in recs.values() for arm in ("pullback", "gated_off")
              for st in rec.get(arm, {}).get("steps", {}).values()]
    splits += [rec.get("do_nothing", {}).get("routes", {}) for rec in recs.values()]
    g3 = all(sp.get("G3_ok", True) for sp in splits)
    g4: dict[str, Any] = {"status": "not applicable"}
    emo30 = at(EMOTICON, "pullback", "30")
    if emo30 is not None and recs[EMOTICON]["n_heldout"] == CONC_N_HELDOUT:
        pc = emo30["routes"]["p_cells"]
        diffs = {k: pc[k] - v for k, v in CONC_P.items()}
        g4 = {"status": "pass" if all(abs(d) <= 0.02 for d in diffs.values()) else "MISS (find the cause before the analysis)",
              "p_cells": {k: pc[k] for k in CONC_P}, "diff": diffs}
    res: dict[str, Any] = {"G1": g1, "G1b": g1b, "G3_identities_hold": g3, "G4": g4,
                           "harness_separates_placement": harness_separates, "P1_margin": margin, "m_d": m_d, "m_r": m_r}
    if not g3:
        res["verdict"] = "G3 violated: a route-split identity failed; nothing scored"
        return res
    if not harness_separates:
        res["verdict"] = "G1 failed: the harness does not separate placement; P1, P2, A1, A2 not scored"
        return res
    if g1b.startswith("FAIL"):
        res["G1b_blocks"] = "A1 (d-based) not scored"
    elif g1b != "pass":
        res["caveat"] = f"G1b {g1b}: a low d everywhere could be the metric rather than placement"

    # P1: coverage effect on retained_ci_masked; P2: every high-coverage task >= 0.8 x emoticon
    p1 = compare_rungs(tasks, ci_masked, excluded, margin, "G1 inconclusive or not run")
    n_pass, n_scorable = tally(p1)
    res["P1"] = {"rungs": p1, "scorable_rungs": n_scorable, "passes": n_pass,
                 "verdict": "unscorable" if n_scorable < 2 or margin > 0.5 else ("pass" if n_pass >= 2 else "null")}
    emo = ci_masked(EMOTICON)
    p2 = {}
    for rung, t in tasks.items():
        hi = ci_masked(t["high"]) if "high" in t else None
        if hi is not None and emo is not None and t["high"] not in excluded:
            p2[rung] = {"status": "pass" if hi >= 0.8 * emo else "fail", "high": hi, "emoticon": emo}
    res["P2"] = {"rungs": p2, "verdict": "unscorable" if not p2 else
                 ("pass" if all(v["status"] == "pass" for v in p2.values()) else "null")}

    # A1: coverage predicts the direct share d
    def direct(name: str) -> float | None:
        sh = shares(name)
        return None if sh is None else sh["d"]

    a1 = compare_rungs(tasks, direct, excluded, m_d,
                       "G1 inconclusive, not learned, or gain < 1")
    a1_pass, a1_n = tally(a1)
    res["A1"] = {"rungs": a1, "verdict": "not scored (G1b failed)" if g1b.startswith("FAIL") else
                 "unscorable" if a1_n < 2 or m_d > 0.5 else ("pass" if a1_pass >= 2 else "null")}

    # A2: the inside route runs through re-gating, r + i >= d + m_r
    a2 = {}
    for name in recs:
        sh = shares(name)
        a2[name] = ({"status": "excluded (not learned or gain < 1)"} if sh is None else
                    {"status": "pass" if sh["r"] + sh["i"] >= sh["d"] + m_r else "fail", "r+i": sh["r"] + sh["i"], "d": sh["d"]})
    ladder_names = [n for n in recs if n != EMOTICON]
    n2 = sum(a2[n]["status"] == "pass" for n in ladder_names)
    n2_scorable = sum(a2[n]["status"] in ("pass", "fail") for n in ladder_names)
    emo_ok = a2.get(EMOTICON, {}).get("status") == "pass"
    res["A2"] = {"tasks": a2, "ladder_passes": n2, "ladder_scorable": n2_scorable,
                 "verdict": "unscorable" if n2_scorable < 4 else ("pass" if n2 >= 5 and emo_ok else "null" if n2 <= 3 else
                                                                  f"neither ({n2} of 6, emoticon {'pass' if emo_ok else 'fail'})")}
    return res


def stage_ladder(S: Sizes, device: str) -> None:
    require_predictions()
    tasks_file, noise_file = RESULTS / f"tasks{S.suffix}.json", RESULTS / f"noise{S.suffix}.json"
    assert tasks_file.is_file(), f"run `mine` first ({tasks_file})"
    assert noise_file.is_file(), f"run `noise` first ({noise_file})"
    tasks = json.loads(tasks_file.read_text())["tasks"]
    noise = json.loads(noise_file.read_text())
    ctx = load_all(device)
    ctx["pile_W0"] = noise.get("pile_W0", {}).get("per_seed", {}).get("0")  # the Pile harness on W0, seed 0, same rows
    ctx["lo_repeat_noise"] = noise.get("lo_repeat_noise", 0.0)  # sets G3's do-nothing tolerance
    out: dict[str, Any] = {"tasks": {}, "task_table": tasks}
    emo = ctx["emoticon"]
    out["tasks"][EMOTICON] = run_task(ctx, EMOTICON, emo["train"], emo["heldout"], emo["target"], None, S, do_pile=False)
    dump(f"ladder{S.suffix}.json", out)
    for rung, t in tasks.items():
        if "high" not in t:
            continue
        for which in ("high", "low"):
            name = t[which]
            if name in out["tasks"]:  # high and low can be the same candidate when coverages tie
                continue
            w = torch.load(LARGE / f"windows_{name}.pt", weights_only=False)
            rec = run_task(ctx, name, w["train"], w["heldout"], w["task"]["target"], w["gbar"], S,
                           do_pile=which == "high" and int(rung.removeprefix("rung")) in PILE_RUNGS)
            out["tasks"][name] = rec | {"task": w["task"]}
            dump(f"ladder{S.suffix}.json", out)
    out["score"] = score(out, tasks, noise, scored=S.scored)
    dump(f"ladder{S.suffix}.json", out)
    log(f"score: {json.dumps(out['score'], indent=1)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["noise", "mine", "ladder", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    S = SMOKE if args.smoke else FULL
    stages = {"noise": [stage_noise], "mine": [stage_mine], "ladder": [stage_ladder],
              "all": [stage_noise, stage_mine, stage_ladder]}[args.stage]
    for stage in stages:
        stage(S, args.device)


if __name__ == "__main__":
    main()
