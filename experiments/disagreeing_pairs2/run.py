"""E199 (disagreeing_pairs2), cell P2.32: a built pair of 67M fine-tunes where the original model's prior and strength-1
steering point opposite ways, by a construction other than E151's common-word cipher, and the re-gating share read on it.
Plan reports/plans/2026-11-17_plan_p2-32-disagreeing-pairs.md; filed rule results/predictions.json (the run refuses unless
it is committed unchanged). The readers, the controls and the rule are E151's (experiments/disagreeing_pairs: job.Run,
run.controls, rules.pair_calls / reading), called, not copied. Sides are named as built; no verdict on reuse.

  stage 0  E151's controls, unchanged and first (exit 1 on a miss). Then four tables on Amendment 3's windows: the
           frequent-word map at K = 64, 32, 16 (new; partner copy) and the neighbour copy (reuse; partner the
           prior-matched cipher), saved and cut by key; each realised prior and the prior half of the build check.
           No construction passing: exit 1, no training.
  stage 1  in order, each construction that passed: 3 fine-tunes (step 30, bf16 in results/large/), strength-1 steering,
           E151's condition against the partner's saved states. The first "opposite" is the pair; none: exit 1.
  stage 2  the share on the pair's 3 new states (the partner's was read in stage 0), the reading and the filed predictions.
Each stage writes results/stage<n>.json and prints `STAGE n DONE`; every scored number is printed too.

    sbatch --comment=E199 cluster/disagreeing_pairs2.sbatch
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pairs as PR
import torch

from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

J = load_experiment("disagreeing_pairs", "job")  # Run, Stop, check, Clock: E151's readers and plumbing
R = load_experiment("disagreeing_pairs", "run")  # controls, A3_DIR, A6.save_map
A1 = load_experiment("disagreeing_pairs", "amend1")  # w0_tables: W0's per-key geometric-mean probabilities
RU = load_experiment("disagreeing_pairs", "rules")  # pair_calls, reading, CALL
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
SEEDS, STEP = J.SEEDS, J.STEP
log, say, check = J.log, J.say, J.check


@dataclass(frozen=True)
class Construction:
    name: str
    built: str  # "new" (this side is the new one; the partner is the reuse side) or "reuse"
    partner: str  # a saved E147 map
    k: int = 0  # words of a frequent-word map (0: the neighbour copy)


CONSTRUCTIONS = (Construction("freq64", "new", "copy", 64), Construction("freq32", "new", "copy", 32),
                 Construction("freq16", "new", "copy", 16), Construction("neighbour", "reuse", "prior_cipher"))  # fmt: skip
POOL_RANKS = 3000  # the word pool: ranked word-like tokens 0-2999 minus the keys and the half-known map's values (E151)


def name(m: str, s: int) -> str:
    return f"p232/{m}_s{s}"


def update_file(m: str, s: int) -> Path:
    return LARGE / f"dW_{m}_s{s}.pt"


def sides(c: Construction) -> tuple[str, str]:
    """(reuse side, new side) as built; the saved partner's states are E151's names (e58/<map>_s<seed>)."""
    return (c.partner, c.name) if c.built == "new" else (c.name, c.partner)


class Run(J.Run):
    """E151's Run (one model, the share and steering readers, the saved maps) with this run's files and new maps."""

    def __init__(self, P: dict[str, Any]) -> None:
        super().__init__(P)
        self.passing: list[str] = []  # constructions past the prior half of the build check, in order (stage 0)
        self.built: str | None = None  # the first that reads "opposite" (stage 1)

    def write(self, n: int, out: dict[str, Any]) -> None:
        tmp = RESULTS / f"stage{n}.tmp"  # a kill mid-write must not leave unreadable JSON
        tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
        tmp.replace(RESULTS / f"stage{n}.json")

    def add_built(self, m: str) -> None:
        t = f"p232/{m}"
        self.inv["tasks"][t] = {"format": "kv", "path": str((LARGE / f"windows_{m}_trained.pt").relative_to(REPO)), "batch": 64}
        for s in SEEDS:
            self.inv["states"][name(m, s)] = {"task": t, "file": str(update_file(m, s).relative_to(REPO))}
        self.e136.reads[t] = J.E136.load_read(t, self.inv["tasks"][t])

    def built_prior(self, m: str) -> float:
        """The filed prior (E151's map_prior on this map's windows): W0's geometric-mean P(trained answer), trained keys."""
        self.prior[m] = KD.geo_rate(ST.load_task(self.net, REPO / self.inv["tasks"][f"p232/{m}"]["path"], m, m).lp0)
        return self.prior[m]

    def state(self, m: str, s: int) -> str:
        return name(m, s) if f"p232/{m}" in self.inv["tasks"] else J.state(m, s)

    def seeds(self, cache: dict[str, dict[str, Any]], m: str, key: Any) -> list[float]:
        """A side's value per seed; nan for a seed not read (a construction stopped at the gain floor)."""
        return [float(key(cache[self.state(m, s)])) if self.state(m, s) in cache else math.nan for s in SEEDS]

    def pair_of(self, c: Construction, share: bool = False) -> dict[str, Any]:
        r, n = sides(c)
        out = {"prior_gap": math.log10(self.prior[r]) - math.log10(self.prior[n]),
               "steering": (self.seeds(self.steer, r, lambda x: x["pooled"]["share"]), self.seeds(self.steer, n, lambda x: x["pooled"]["share"]))}  # fmt: skip
        if share:
            out["share"] = (self.seeds(self.share, r, lambda x: x["share"]), self.seeds(self.share, n, lambda x: x["share"]))
        return out


# ----------------------------------------------------------------------------- stage 0


def tables(run: Run, src: dict[str, Any], keys: list[int], out: dict[str, Any]) -> dict[str, kv.KVTask]:
    """The four tables from W0: frequent-word maps (K = 64, 32, 16) and the neighbour copy."""
    G, _, _ = A1.w0_tables(run, src, keys, OP.split_keys(keys, len(keys) // 4, 0)[0])
    _, plain, _, ranked, tok = KD.vocab(run.ctx)
    banned = set(keys) | set(run.F["a3"]["maps"]["elicit"]["values"])
    pool = [v for v in ranked[:POOL_RANKS] if v not in banned]
    marker, made = src["task"].marker, {}
    for c in CONSTRUCTIONS:
        if c.k:
            words = PR.frequent_words(G, pool, c.k)
            out["words"][c.name] = {"words": [tok.decode([w]) for w in words], "min_P_over_keys": G[:, words].min(0).values.tolist(),
                                    "median_P_over_keys": G[:, words].median(0).values.tolist()}  # fmt: skip
            made[c.name] = kv.KVTask(c.name, marker, tuple(keys), PR.balanced(keys, words, 0))
    E = run.net.net.wte.weight.detach()
    allowed = plain[: E.shape[0]].clone()
    allowed[keys] = False
    made["neighbour"] = m = kv.KVTask("neighbour", marker, tuple(keys), PR.neighbours(E, keys, allowed))
    out["words"]["neighbour"] = {tok.decode([k]): tok.decode([v]) for k, v in zip(m.keys, m.values, strict=True)}
    say("0 words", out["words"])
    out["letter_of"] = {str(k): tok.decode([k]).strip()[:1].lower() for k in keys}  # save_map's record needs it (E151)
    return made


def build(run: Run, ch: dict[str, Any], out: dict[str, Any]) -> None:
    """E151's file and split checks; the tables saved and cut by key; their priors and the prior half of the build check."""
    a3, th = run.F["a3"], run.th
    check(ch, "0 sha256 windows_elicit.pt", float(KD.sha256(R.A3_DIR / "windows_elicit.pt") != a3["maps"]["elicit"]["sha256"]["full"]), 0)
    keys = a3["keys"]["keys"]
    trained, held = OP.split_keys(keys, len(keys) // 4, 0)
    check(ch, "0 key split is Amendment 3's (0 = yes)", float(held != a3["keys"]["held"]), 0)
    src = torch.load(R.A3_DIR / "windows_elicit.pt", weights_only=False)
    out["words"] = {}
    made = tables(run, src, keys, out)
    rec: dict[str, Any] = {"maps": {}, "tables": {"letter_of": out["letter_of"]}}
    out["constructions"] = {}
    for c in CONSTRUCTIONS:
        R.A6.save_map(run.ctx, src, made[c.name], f"P2.32 {c.name}", trained, held, LARGE, rec)
        run.add_built(c.name)
        run.built_prior(c.name)
        gap = run.pair_of(c)["prior_gap"]
        ok = PR.prior_ok(gap, th["build_margin"])
        out["constructions"][c.name] = {"built": c.built, "partner": c.partner, "prior": run.prior[c.name], "partner_prior": run.prior[c.partner],
                                        "prior_gap": gap, "prior_ok": ok, "n_distinct": len(set(made[c.name].values)),
                                        "zero_max_abs": rec["maps"][c.name]["zero_max_abs"]}  # fmt: skip
        say(f"0 construction {c.name}", out["constructions"][c.name])
        check(ch, f"0 table {c.name}: zero update reads 0", rec["maps"][c.name]["zero_max_abs"], 0)
    out["passing"] = [c.name for c in CONSTRUCTIONS if out["constructions"][c.name]["prior_ok"]]
    log(f"[0 prior half of the build check] passing, in order: {out['passing']}")
    if not out["passing"]:
        raise J.Stop("could not be built: no construction passes its prior bar")


def stage0(run: Run, out: dict[str, Any], _t_job: float) -> None:
    build(run, R.controls(run, out), out)


# ----------------------------------------------------------------------------- stage 1


def train(run: Run, m: str, s: int, out: dict[str, Any]) -> float:
    """keydep.finetune (E58's recipe, as E147 refit and E151 trained), the step-30 update saved in bf16; its steering gain."""
    w = torch.load(REPO / run.inv["tasks"][f"p232/{m}"]["path"], weights_only=False)
    ft = KD.finetune(run.ctx, w, s, (STEP,))
    torch.save({k: v.bfloat16() for k, v in ft.deltas[STEP].items()}, update_file(m, s))
    out["updates"][name(m, s)] = {"sha256": KD.sha256(update_file(m, s)), "p_heldout": ft.evals[STEP]["p_o"]}
    rec = out["steering"][name(m, s)] = run.steering(name(m, s))
    log(f"[1 steering {name(m, s)}] gain {rec['gain']:.4f} layer {rec['pooled']['layer']} share {rec['pooled']['share']:.4f}")
    return rec["gain"]


def stage1(run: Run, out: dict[str, Any], t_job: float) -> None:
    out |= {"updates": {}, "steering": {}, "tried": {}}
    clock = J.Clock(t_job, len(SEEDS) * len(run.passing), run.th["projected_limit_min"])
    for c in (c for c in CONSTRUCTIONS if c.name in run.passing):
        for s in SEEDS:
            gain = train(run, c.name, s, out)
            run.write(1, out)
            clock.tick(out)
            if not gain >= run.th["gain_floor"]:
                log(f"[1 gain floor] {c.name} seed {s} gain {gain:.4g}: its steering is unreadable")
                break
        out["tried"][c.name] = calls = RU.pair_calls(run.pair_of(c), run.th)
        say(f"1 condition {c.name}", calls)
        out["built"] = PR.first_built({n: x["condition"] for n, x in out["tried"].items()})
        if out["built"]:
            log(f"[1 built] {c.name}: prior and strength-1 steering point opposite ways")
            return
    raise J.Stop(f"could not be built: none of {run.passing} reads 'opposite'")


# ----------------------------------------------------------------------------- stage 2


def stage2(run: Run, out: dict[str, Any], _t_job: float) -> None:
    c = next(c for c in CONSTRUCTIONS if c.name == run.built)
    out["shares"] = {}
    for s in SEEDS:
        n = name(c.name, s)
        rec = out["shares"][n] = run.shared(n)
        log(f"[2 share {n}] r {rec['r']:.6f} gain {rec['gain']:.4f} share {rec['share']:.6f}")
        run.write(2, out)
    out["inputs"] = p = run.pair_of(c, share=True)
    rec = RU.pair_calls(p, run.th)
    rec["share"] = RU.CALL(*p["share"], run.th)
    rec["reading"] = RU.reading(rec["condition"], rec["prior"]["call"], rec["steering"], rec["share"])
    out["score"] = {"pair": {"construction": c.name, "reuse side": sides(c)[0], "new side": sides(c)[1]}, **rec}
    out["filed"] = {"P1": True, "P2": c.name == "freq64", "P3": rec["reading"] == "follows steering"}
    say("2 score", out["score"])
    say("2 filed predictions (true = as leaned)", out["filed"])


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = J.load(PRED)
    assert P["experiment"].startswith("E") and "disagreeing_pairs2" in P["experiment"], P["experiment"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E58 trained and E136/E143 read
    run = Run(P)
    log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()}; model loaded at {(time.time() - t_job) / 60:.1f} min")
    outs: list[dict[str, Any]] = [{}, {}, {}]
    for n, stage in enumerate((stage0, stage1, stage2)):
        try:
            stage(run, outs[n], t_job)
        except J.Stop as e:
            outs[n]["stopped"] = str(e)
            if n == 1:
                outs[n]["filed"] = {"P1": False, "P2": False, "P3": False}
            run.write(n, outs[n])
            log(f"STOP: {e}: tell the lead")
            return 1
        run.passing, run.built = outs[0].get("passing", []), outs[1].get("built")  # stage 1 reads the first, stage 2 the second
        run.write(n, outs[n])
        log(f"STAGE {n} DONE")
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; {(time.time() - t_job) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    hard_exit(main())
