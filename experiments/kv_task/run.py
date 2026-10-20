"""kv_task (round 5): can post-training inside the decomposition learn a behaviour the model lacks, and is the result
still a faithful decomposition (GOALS's weak version, reading A)?

Plan: reports/2026-10-19_plan_key-value-task.md. Tasks (src/commute/kv.py): after a marker token M, a key word x is
followed by table[x]. rand1 and rand32: a random bijection from K keys onto K other word-like tokens, a word-level
substitution cipher. ident32: table[x] = x, the matched-format control (a behaviour the model partly has). Per task:
  finetune    the W fine-tune of the 24 matrices (arm A's loss with a per-window target): the learnability reference;
  do_nothing  its delta-W placed in Delta, components unchanged: the trivial map, and the reference for retained_ci_masked;
  lift        the Euclidean lift of that delta-W onto the alive subcomponents: the metric positive control (rounds 2 and
              4 put it at Pile PGD 5-7, so a passing pull-back is read against a control the metrics do catch);
  pullback    Adam on all alive (U, V) with Gamma and Delta frozen: the map under test.
Each arm is scored at its first checkpoint with held-out P(target) >= P_LEARN, else at its last checkpoint (flagged):
the task-local harness and route split (src/commute/local_faith.py), round 2's Pile ratios, and VPD's sparsity loss
(src/commute/faithfulness.py) on Pile blocks and on the held-out windows.

Stages (repo root, project venv, GPU):
  --predictions  file results/predictions.json (refuses to overwrite)
  build          Pile rows -> marker, keys, values, windows, baselines -> results/tasks.json, results/large/windows_<task>.pt
  run            the arms -> results/kv_task.json (dumped after every task), scored at the end
  all            build, run
  --smoke        any stage at the SMOKE sizes; writes *_smoke.json
Inputs: experiments/transport_test/results/large/alive.pt.
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

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"

HALF = 20
FT_LR, PB_LR = 3e-5, 1e-5  # the ladder's recipes
P_LEARN = 0.5  # an arm has learned at the first checkpoint whose held-out P(target) reaches this
GATE = 1e-4  # a random entry's baseline P(value | M key) must be below this ("a behaviour the model lacks")
IDENT_MAX = 0.5  # the identity control needs a gain to retain: baseline P(key | M key) at most this
FAITH_ROWS = 256  # Pile val rows for the Pile harness (2 blocks of 128, one adversary each) and the sparsity loss
SPARSITY_BLOCK = 64  # rows per sparsity block: 4 blocks, their spread is the floor of a deterministic metric
LIFT_TOL = 1e-6

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-19",
    "plan": "reports/2026-10-19_plan_key-value-task.md",
    "scored_step_rule": "per arm, the first checkpoint (10, 30, 100, 300) with held-out P(target) >= 0.5; else the last "
                        "checkpoint, flagged 'did not learn'",
    "G1_finetune_learns": {
        "rule": "the W fine-tune reaches held-out P(target) >= 0.5 within 300 steps on every task; a task whose "
                "fine-tune does not learn is unscorable (the format, not the map)",
        "predicted": "pass on all three"},
    "P1_identity_inside": {
        "rule": "the pull-back on ident32 reaches held-out P(target) >= 0.5 within 300 steps",
        "predicted": "pass"},
    "P2_new_inside": {
        "rule": "the pull-back on rand32 reaches held-out P(target) >= 0.5 within 300 steps",
        "predicted": "pass (the alive (U, V) space holds ~70% of a real update by input weight; the ladder's rung-2 "
                     "pull-backs learned)",
        "blind_to": "how the behaviour is placed; P3-P5 read that"},
    "P3_pile_pgd": {
        "rule": "at the scored step, the pull-back's Pile PGD ratio to W0 (round 2's harness, 256 val rows, seed 0) is "
                "<= 1.5 on every task, AND the lift control's is > 1.5 on every task (metric power); stochastic ratio "
                "reported",
        "predicted": "pass on rand1 and ident32; pass on rand32 (least sure: the update is larger than any pull-back "
                     "scored so far)",
        "blind_to": "the trained behaviour itself (~0 trigger positions in Pile): it says the account of the old model "
                    "survived",
        "positive_control": "the lift of the fine-tune's delta-W"},
    "P4_sparsity": {
        "rule": "at the scored step, the pull-back's Pile sparsity ratio to W0 (VPD's importance-minimality loss, "
                "p 0.4 beta 0.5, mean over 4 blocks of 64 rows) is <= 1.10, and |ratio - 1| <= 3 x the spread of its "
                "per-block ratios; the lift control reported. PROVISIONAL: no prior run of this metric exists, so the "
                "1.10 line is a guess; the block spread is the floor",
        "predicted": "pass on all three",
        "blind_to": "sparsity at the trigger (reported from the held-out windows, not scored)",
        "positive_control": "the lift of the fine-tune's delta-W"},
    "P5_retained_vs_identity": {
        "rule": "retained_ci_masked(pull-back, rand32) >= 0.8 x retained_ci_masked(pull-back, ident32), both at their "
                "scored steps (the ladder's P2 form). Reported beside it: each pull-back against its task's do-nothing "
                "reference (pull-back - do-nothing, the re-gating floor r ~ 0.55-0.60 of the recruit test)",
        "predicted": "FAIL: the new behaviour is kept less under Gamma's W0-style mask than the identity one (the ladder's "
                     "rung 2 fell to 0.3-0.6 against 0.75-0.92 at rungs 0-1). A pass would say frozen-Gamma inside "
                     "training covers a behaviour the model lacks",
        "blind_to": "damage outside the trigger (kl_other and the Pile ratios carry that)",
        "positive_control": "ident32 itself (a behaviour the model partly has)"},
    "reported_not_scored": ["route split (d, r, i, e_gate, e_delta): mechanism only under reading A; e_delta = 0 by "
                            "construction for the pull-back", "task-local PGD (no power, reports/2026-10-19_review_pgd-local.md)",
                            "rand1 on P5 (one bigram: the ladder's territory)", "per-entry held-out P(value) at the scored step"],
}


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the full run and the smoke run, in one place."""

    rows: int  # Pile train rows: token frequencies, the marker, and the windows
    n_windows: int  # per task, split 4:1 train:held-out by source row
    key_rank0: int  # keys: word-like tokens by frequency rank, from this rank
    value_rank0: int  # the value pool starts at this rank (mid-frequency words pass the baseline gate)
    pool: int
    tasks: tuple[tuple[str, int, str], ...]  # (name, K, "bijection" | "identity")
    steps: int
    checkpoints: tuple[int, ...]
    stoch_draws: int
    pgd_steps: int
    pile: bool  # Pile ratios and the sparsity loss
    suffix: str

    def __post_init__(self) -> None:
        assert max(self.checkpoints) == self.steps and max(k for _, k, _ in self.tasks) <= self.pool
        assert self.key_rank0 + max(k for _, k, _ in self.tasks) <= self.value_rank0


FULL = Sizes(rows=32768, n_windows=500, key_rank0=100, value_rank0=1000, pool=2000,
             tasks=(("rand1", 1, "bijection"), ("rand32", 32, "bijection"), ("ident32", 32, "identity")),
             steps=300, checkpoints=(10, 30, 100, 300), stoch_draws=4, pgd_steps=20, pile=True, suffix="")
SMOKE = Sizes(rows=64, n_windows=16, key_rank0=5, value_rank0=20, pool=40,
              tasks=(("rand1", 1, "bijection"), ("rand2", 2, "bijection"), ("ident2", 2, "identity")),
              steps=2, checkpoints=(1, 2), stoch_draws=1, pgd_steps=2, pile=False, suffix="_smoke")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    return float(o)


def file_predictions() -> None:
    assert not PRED.is_file(), f"{PRED} exists; predictions are never edited after filing (add a dated report)"
    dump(PRED.name, PREDICTIONS)
    log(f"filed {PRED}")


def require_predictions() -> dict[str, Any]:
    assert PRED.is_file(), f"file the predictions first ({PRED})"
    return json.loads(PRED.read_text())


# ----------------------------------------------------------------------------- shared pieces


def load_all(device: str) -> dict[str, Any]:
    from commute import component_training, split, vpd

    model = vpd.load_component_model(device)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    uv = component_training.alive_uv(model, alive)
    uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in uv.items()}
    t0 = time.time()
    bases = split.bases(uv_cpu, LIFT_TOL)
    log(f"lift bases in {time.time() - t0:.0f} s")
    return {"device": device, "model": model, "W0": vpd.target_weights(model), "comps0": vpd.component_weights(model),
            "alive": alive, "uv": uv, "uv_cpu": uv_cpu, "bases": bases, "sparsity_cfg": vpd.sparsity_config()}


def logits_fn(ctx: dict[str, Any]) -> Any:
    from commute.finetune import logits_of

    return lambda toks: logits_of(ctx["model"].target_model, toks)


def arm_a_cfg(table: Tensor, steps: int, checkpoints: tuple[int, ...]) -> Any:
    from commute.finetune import ArmAConfig

    return ArmAConfig(o_token=table, lr=FT_LR, steps=steps, checkpoints=checkpoints)


def scored_step(p_at: dict[int, float]) -> tuple[int, bool]:
    """The first checkpoint with held-out P(target) >= P_LEARN; else the last, with learned = False."""
    for step in sorted(p_at):
        if p_at[step] >= P_LEARN:
            return step, True
    return max(p_at), False


def pile_rows_blocks(ctx: dict[str, Any]) -> None:
    """Round 2's Pile harness blocks and the sparsity blocks, plus W0's values on them (once per run)."""
    from commute.data import batches, pile_rows
    from commute.faithfulness import faithfulness, sparsity_loss

    rows = pile_rows("val", 0, FAITH_ROWS, log)
    ctx["faith_blocks"] = [batches(rows[s : s + 128], 8) for s in range(0, rows.shape[0], 128)]
    ctx["sparsity_blocks"] = [[b] for b in batches(rows, SPARSITY_BLOCK)]
    ctx["pile_W0"] = pile_mean([faithfulness(ctx["model"], b, seed=0, log=log) for b in ctx["faith_blocks"]])
    ctx["sparsity_W0"] = [sparsity_loss(ctx["model"], b, **ctx["sparsity_cfg"]) for b in ctx["sparsity_blocks"]]
    log(f"W0: pile {ctx['pile_W0']} sparsity per block {ctx['sparsity_W0']}")


def pile_mean(per: list[dict[str, float]]) -> dict[str, float]:
    return {k: sum(p[k] for p in per) / len(per) for k in per[0] if k != "pgd_rows"}


def window_halves(held: Any) -> list[list[Tensor]]:
    n = len(held) // 2
    return [[held.tokens[:n]], [held.tokens[n:]]]


def sparsity_ratios(ctx: dict[str, Any], blocks: list[list[Tensor]], w0: list[float]) -> dict[str, Any]:
    """The state's sparsity loss per block over W0's on the same block: mean ratio and the spread of the ratios."""
    from commute.faithfulness import sparsity_loss

    vals = [sparsity_loss(ctx["model"], b, **ctx["sparsity_cfg"]) for b in blocks]
    ratios = [v / w for v, w in zip(vals, w0)]
    return {"values": vals, "W0": w0, "ratios": ratios, "ratio": sum(ratios) / len(ratios), "spread": max(ratios) - min(ratios)}


def evaluate(ctx: dict[str, Any], tgt: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]] | None, held: Any,
             table: Tensor, S: Sizes, base: dict[str, Any], rbase: dict[str, Any], do_nothing: bool = False) -> dict[str, Any]:
    """Every measurement of one state (target weights `tgt`, components `comps` or the released ones): the task-local
    harness, retained, the route split, and with S.pile the Pile ratios and the sparsity ratios (Pile blocks, held-out halves)."""
    from commute.faithfulness import faithfulness, model_state
    from commute.local_faith import local_faithfulness, retained, route_split

    model = ctx["model"]
    with model_state(model, target=tgt, components=comps):
        res: dict[str, Any] = {"local": local_faithfulness(model, held.tokens, held.half, table, n_stoch_draws=S.stoch_draws,
                                                            pgd_steps=S.pgd_steps, log=log)}
        res["retained"] = retained(res["local"], base["local"])
        sp = route_split(model, held.tokens, held.half, table, rbase, ctx["comps0"])
        s = sp["shares"]
        ok = abs(sp["share_sum"] - 1) < 1e-6 if abs(sp["gain"]) >= 1e-6 else True
        if do_nothing and abs(sp["gain"]) >= 1e-6:
            ok = ok and abs(s["d"]) < 1e-3 and abs(s["i"]) < 1e-3 and abs(s["e_delta"] - 1) < 1e-3
        sp["G3_ok"] = bool(ok)
        res["routes"] = sp
        log(f"[routes{' do-nothing' if do_nothing else ''}] gain {sp['gain']:.3f} " + " ".join(f"{k} {v:.3f}" for k, v in s.items())
            + f" | G3 {'ok' if ok else 'VIOLATED'}")
        if S.pile:
            f1 = pile_mean([faithfulness(model, b, seed=0, log=log) for b in ctx["faith_blocks"]])
            res["pile"] = {"state": f1, "ratios": {k: f1[k] / ctx["pile_W0"][k] for k in f1}}
            res["sparsity"] = {"pile": sparsity_ratios(ctx, ctx["sparsity_blocks"], ctx["sparsity_W0"]),
                               "windows": sparsity_ratios(ctx, window_halves(held), base["sparsity_windows"])}
            log(f"[pile] ratios {res['pile']['ratios']} | sparsity pile {res['sparsity']['pile']['ratio']:.4f} "
                f"(spread {res['sparsity']['pile']['spread']:.4f}) windows {res['sparsity']['windows']['ratio']:.4f}")
    return res


def state_to_weights(ctx: dict[str, Any], st: dict[str, tuple[Tensor, Tensor, Tensor]]) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor]]]:
    """A pull-back-shaped state {k: (delta-W, U_A', V_A')} -> (target weights, full components)."""
    from commute import component_training

    W0 = ctx["W0"]
    tgt = {k: W0[k] + st[k][0].to(W0[k].device) if k in st else W0[k] for k in W0}
    return tgt, component_training.components_with(ctx["model"], ctx["alive"], {k: (st[k][1], st[k][2]) for k in st})


# ----------------------------------------------------------------------------- stage: build


def stage_build(S: Sizes, device: str) -> None:
    from commute import kv, ladder, vpd
    from commute.data import pile_rows, split_windows

    require_predictions()
    ctx = load_all(device)
    tok = vpd.tokenizer()
    lf = logits_fn(ctx)
    rows = pile_rows("train", 0, S.rows, log)
    with torch.no_grad():
        vocab = lf(rows[:1].to(device)).shape[-1]
    plain = ladder.plain_mask(tok.decode, vocab)
    counts = ladder.count_tokens(rows, HALF)
    marker = kv.pick_marker(counts, tok.decode, vocab)
    ranked = [t for t, _ in sorted(counts.items(), key=lambda tc: -tc[1]) if t < vocab and bool(plain[t])]
    pool = ranked[S.value_rank0 : S.value_rank0 + S.pool]
    out: dict[str, Any] = {"n_rows": S.rows, "marker": marker, "marker_str": tok.decode([marker]),
                           "marker_count": counts.get(marker, 0), "vocab": vocab, "tasks": {}}
    log(f"build: marker {marker} {out['marker_str']!r}; {len(ranked)} word-like tokens ranked")
    LARGE.mkdir(parents=True, exist_ok=True)
    for name, K, kind in S.tasks:
        keys = tuple(ranked[S.key_rank0 : S.key_rank0 + K])
        w = kv.windows(rows, marker, keys, HALF, S.n_windows, seed=0)  # same contexts for every task: matched controls
        dist = kv.key_distributions(lf, w, device)
        ident = {k: float(dist[k][k]) for k in keys}
        assert max(ident.values()) <= IDENT_MAX, f"identity baseline {max(ident.values()):.3f} > {IDENT_MAX}: raise key_rank0"
        task = (kv.identity(name, marker, keys) if kind == "identity"
                else kv.bijection(name, marker, keys, pool, seed=0, ok=lambda k, v, d=dist: float(d[k][v]) < GATE))
        baseline = {k: float(dist[k][v]) for k, v in zip(task.keys, task.values)}
        train, held = split_windows(w)
        assert len(held) >= 1 and len(train) >= 1, (len(train), len(held))
        rec = {"kind": kind, "K": K, "keys": list(task.keys), "values": list(task.values),
               "entries": [f"{tok.decode([k])!r} -> {tok.decode([v])!r}" for k, v in zip(task.keys, task.values)],
               "baseline": baseline, "baseline_max": max(baseline.values()), "identity_baseline": ident,
               "n_train": len(train), "n_heldout": len(held),
               "heldout_keys_missing": sorted(set(task.keys) - set(held.tokens[:, HALF].tolist()))}
        torch.save({"train": train, "heldout": held, "task": task, "table": task.table(vocab)}, LARGE / f"windows_{name}.pt")
        out["tasks"][name] = rec
        log(f"build: {name} K {K} {kind}: baseline max {rec['baseline_max']:.2e}, identity max {max(ident.values()):.3f}, "
            f"train {len(train)} held-out {len(held)}; e.g. {rec['entries'][:3]}")
    dump(f"tasks{S.suffix}.json", out)


# ----------------------------------------------------------------------------- stage: run


def run_task(ctx: dict[str, Any], name: str, w: dict[str, Any], S: Sizes) -> dict[str, Any]:
    from commute import component_training, split
    from commute.faithfulness import sparsity_loss
    from commute.finetune import finetune_arm_a
    from commute.local_faith import local_faithfulness, route_cells

    model, W0 = ctx["model"], ctx["W0"]
    train, held, table = w["train"], w["heldout"], w["table"]
    t0 = time.time()
    rec: dict[str, Any] = {"name": name, "n_train": len(train), "n_heldout": len(held)}
    base = {"local": local_faithfulness(model, held.tokens, held.half, table, n_stoch_draws=S.stoch_draws, pgd_steps=S.pgd_steps, log=log)}
    if S.pile:
        base["sparsity_windows"] = [sparsity_loss(model, b, **ctx["sparsity_cfg"]) for b in window_halves(held)]
    rbase = route_cells(model, held.tokens, held.half, table)
    rec["base"] = base
    rec["baseline_window_p"] = float(torch.sigmoid(torch.tensor(base["local"]["target"]["lo_trigger"])))
    # the W fine-tune: learnability reference; its scored delta-W feeds do-nothing and the lift
    ft = finetune_arm_a(model, train, held, arm_a_cfg(table, S.steps, S.checkpoints), ctx["device"], log)
    step, learned = scored_step({s: e["p_o"] for s, e in ft.evals.items()})
    rec["finetune"] = {"evals": ft.evals, "scored_step": step, "learned": learned, "log_tail": ft.log[-3:]}
    (LARGE / name).mkdir(parents=True, exist_ok=True)
    dW = ft.deltas[step]
    torch.save(dW, LARGE / name / f"dW_ft_step{step}.pt")
    tgt = {k: W0[k] + dW[k].to(W0[k].device) for k in W0}
    rec["do_nothing"] = evaluate(ctx, tgt, None, held, table, S, base, rbase, do_nothing=True)
    st_lift = split.split_state(dW, ctx["uv_cpu"], ctx["bases"])
    rec["lift"] = evaluate(ctx, *state_to_weights(ctx, st_lift), held, table, S, base, rbase)
    # the pull-back
    cfg = component_training.TrainConfig(lr=PB_LR, steps=S.steps, save_steps=S.checkpoints)
    curve, states = component_training.train(model, ctx["uv"], train.tokens, held, arm_a_cfg(table, S.steps, ()), cfg, ctx["device"], W0,
                                   save_dir=LARGE / name / "pullback", log=log)
    step_pb, learned_pb = scored_step({s: curve[s - 1]["heldout_p_o"] for s in states})
    rec["pullback"] = {"curve": curve, "scored_step": step_pb, "learned": learned_pb,
                       "eval": evaluate(ctx, *state_to_weights(ctx, states[step_pb]), held, table, S, base, rbase)}
    rec["seconds"] = time.time() - t0
    pb, dn = rec["pullback"]["eval"]["retained"], rec["do_nothing"]["retained"]
    log(f"task {name}: ft learned {learned} @ {step} | pull-back learned {learned_pb} @ {step_pb}: retained ci "
        f"{pb['ci_masked']:.3f} (do-nothing {dn['ci_masked']:.3f}) unmasked {pb['unmasked']:.3f} | {rec['seconds']:.0f} s")
    return rec


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules (PREDICTIONS / results/predictions.json)."""
    recs = out["tasks"]
    res: dict[str, Any] = {}
    res["G1"] = {n: {"status": "pass" if r["finetune"]["learned"] else "FAIL (unscorable task)", "step": r["finetune"]["scored_step"]}
                 for n, r in recs.items()}
    scorable = {n for n, r in recs.items() if r["finetune"]["learned"]}

    def learned(name: str) -> dict[str, Any]:
        if name not in recs:
            return {"status": "not run"}
        if name not in scorable:
            return {"status": "unscorable (G1)"}
        pb = recs[name]["pullback"]
        return {"status": "pass" if pb["learned"] else "FAIL", "step": pb["scored_step"]}

    ident = next((n for n, r in recs.items() if r.get("kind") == "identity"), None)
    rand = next((n for n, r in recs.items() if r.get("kind") == "bijection" and r["K"] > 1), None)
    res["P1"] = learned(ident) if ident else {"status": "not run"}
    res["P2"] = learned(rand) if rand else {"status": "not run"}
    p3, p4 = {}, {}
    for n in scorable:
        pe, le = recs[n]["pullback"]["eval"], recs[n]["lift"]
        if "pile" not in pe:
            p3[n] = p4[n] = {"status": "not run (no Pile harness)"}
            continue
        pb_pgd, lift_pgd = pe["pile"]["ratios"]["pgd_kl"], le["pile"]["ratios"]["pgd_kl"]
        p3[n] = {"status": "pass" if pb_pgd <= 1.5 and lift_pgd > 1.5 else "FAIL", "pullback_pgd": pb_pgd, "lift_pgd": lift_pgd,
                 "control_has_power": lift_pgd > 1.5, "pullback_stochastic": pe["pile"]["ratios"]["stochastic_kl"]}
        sp, sl = pe["sparsity"]["pile"], le["sparsity"]["pile"]
        p4[n] = {"status": "pass" if sp["ratio"] <= 1.10 and abs(sp["ratio"] - 1) <= 3 * sp["spread"] else "FAIL",
                 "pullback": sp["ratio"], "spread": sp["spread"], "lift": sl["ratio"], "do_nothing": recs[n]["do_nothing"]["sparsity"]["pile"]["ratio"],
                 "windows_pullback": pe["sparsity"]["windows"]["ratio"], "windows_lift": le["sparsity"]["windows"]["ratio"]}
    res["P3"], res["P4"] = p3, p4
    if rand in scorable and ident in scorable:
        ci = {n: recs[n]["pullback"]["eval"]["retained"]["ci_masked"] for n in (rand, ident)}
        dn = {n: recs[n]["do_nothing"]["retained"]["ci_masked"] for n in (rand, ident)}
        res["P5"] = {"status": "pass" if ci[rand] >= 0.8 * ci[ident] else "FAIL (as predicted)" if ci[rand] < 0.8 * ci[ident] else "?",
                     "rand": ci[rand], "ident": ci[ident], "vs_do_nothing": {n: ci[n] - dn[n] for n in ci}}
    else:
        res["P5"] = {"status": "unscorable"}
    res["G3_identities_hold"] = all(r[a]["routes"]["G3_ok"] for r in recs.values() for a in ("do_nothing", "lift")) and \
        all(r["pullback"]["eval"]["routes"]["G3_ok"] for r in recs.values())
    return res


def stage_run(S: Sizes, device: str) -> None:
    require_predictions()
    tasks_file = RESULTS / f"tasks{S.suffix}.json"
    assert tasks_file.is_file(), f"run `build` first ({tasks_file})"
    tasks = json.loads(tasks_file.read_text())
    ctx = load_all(device)
    if S.pile:
        pile_rows_blocks(ctx)
    out: dict[str, Any] = {"tasks": {}, "task_table": tasks, "sparsity_cfg": ctx["sparsity_cfg"],
                           "pile_W0": ctx.get("pile_W0"), "sparsity_W0": ctx.get("sparsity_W0")}
    for name, K, kind in S.tasks:
        w = torch.load(LARGE / f"windows_{name}.pt", weights_only=False)
        out["tasks"][name] = run_task(ctx, name, w, S) | {"K": K, "kind": kind, "task": tasks["tasks"][name]}
        dump(f"kv_task{S.suffix}.json", out)
    out["score"] = score(out)
    dump(f"kv_task{S.suffix}.json", out)
    log(f"score: {json.dumps(out['score'], indent=1)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["build", "run", "all"])
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
        return
    assert args.stage, "a stage or --predictions"
    S = SMOKE if args.smoke else FULL
    for stage in {"build": [stage_build], "run": [stage_run], "all": [stage_build, stage_run]}[args.stage]:
        stage(S, args.device)


if __name__ == "__main__":
    main()
