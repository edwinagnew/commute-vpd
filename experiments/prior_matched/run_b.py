"""E255 (cell P2.57) run B: a second prior-matched new task, a second cipher on copy's keys, read against run A's
10 copy seeds. It is built by E58 Amendment 6's rule for the prior-matched cipher, changing only the draw's seed and
the values it may take: on Amendment 3's windows (only the table differs, as for every E58 map), each of the 128 keys
takes a value from E58's cipher pool whose W0 geometric-mean P(value | M key) is in [2e-5, 2e-4] and under the key's
100th-largest, by `kv.bijection` (Amendment 6's draw, `kv.bijection_from_pools`), with the prior-matched cipher's 128
values and the elicit words banned, so no key keeps a value it had; cut by key as E147's windows were
(`heldout_keys.split_task`, the 96 trained keys). The draw seeds are tried in order; the first whose read-row prior
(E133's path) is within prior_tol decades of copy's is used, none: exit 1 before any training. Fine-tuned with E147's
call (seeds 0-2), scaled to E237's common size, one random update per seed, read as run A (world.py).
Rules and expected values in results/predictions_b.json (refuses unless committed unchanged). Needs run A's summary.

  0  the controls, alone (world.controls)                                                 -> results/b/control.json
  a  the build: seed 0 must give Amendment 6's prior-matched cipher (its 128 values, its pool, E237's prior); then
     the second cipher; the do-nothing on its read rows                                  -> results/b/a.json, do_nothing.json
  b  three fine-tunes (exit 1 if one does not learn); c                                  -> results/b/train.json
  c  the 3 states and their random updates                                               -> results/b/items/<item>.json
  score  run A's 10 copy seeds against these 3 (run A's bands)                           -> results/b/summary.json

    sbatch --comment=E255 cluster/prior_matched_b.sbatch
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from commute import anchors as AN
from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute.data import Windows
from commute.heldout_keys import split_task
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
    write_stage,
)
from commute.open_pair import jsonable

W: Any = load_experiment("prior_matched", "world")  # the world, the fine-tune, the controls, the readings, the rules
RA: Any = load_experiment("prior_matched", "run_a")  # run A's config: its copy states and its summary
PRED, OUT = W.RESULTS / "predictions_b.json", W.RESULTS / "b"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
OPR = W.X / "open_pair" / "results"
TASK = "pb_cipher2"
log = W.log


@dataclass(frozen=True)
class Config:  # filed in predictions_b.json "settings"
    seeds: tuple[int, ...]
    draw_seeds: tuple[int, ...]  # the second cipher's bijection seeds, tried in order
    prior_band: tuple[float, float]  # Amendment 6's band on a value's W0 geometric-mean P(value | M key)
    min_rank: int  # Amendment 6's: a value's prior under the key's min_rank-th largest
    split_seed: int  # Amendment 3's key split
    prior_tol: float  # |log10 prior - copy's| at most, on the read rows
    random_seed: int

    def __post_init__(self) -> None:
        assert self.seeds == (0, 1, 2) and 0 not in self.draw_seeds and self.prior_band[0] < self.prior_band[1]

    def states(self) -> list[str]:
        return [f"e58/{TASK}_s{s}" for s in self.seeds]


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


# ---- stage a: the build ---------------------------------------------------------------------------------------------


class Build:
    """Amendment 3's windows and keys, W0's per-key value priors on every window, the admissible-value rule, the pools."""

    def __init__(self, w: Any, cfg: Config) -> None:
        a3 = W.read_json(OPR / "amend3.json")
        self.src_path = OPR / "large" / "amend3" / "windows_elicit.pt"
        assert KD.sha256(self.src_path) == a3["maps"]["elicit"]["sha256"]["full"], "Amendment 3's elicit windows changed"
        self.keys = [int(k) for k in a3["keys"]["keys"]]
        self.trained, held = OP.split_keys(self.keys, len(self.keys) // 4, cfg.split_seed)
        assert held == a3["keys"]["held"], "the key split differs from Amendment 3's"
        self.src = torch.load(self.src_path, weights_only=False)
        tr, ho = self.src["train"], self.src["heldout"]
        both = Windows(torch.cat([tr.tokens, ho.tokens]), torch.cat([tr.g, ho.g]), torch.cat([tr.row, ho.row]), torch.cat([tr.pos, ho.pos]), tr.half)
        self.geo = kv.key_distributions(lambda x: w.net.run(x)[0], both, w.net.device, geometric=True)
        self.ranked = KD.vocab(KD.Ctx(w.net, {}, {}, {}, W.LARGE))[3]
        self.banned = set(self.keys) | {int(v) for v in a3["maps"]["elicit"]["values"]}
        self.kth = {k: float(self.geo[k].topk(cfg.min_rank).values[-1]) for k in self.keys}
        self.lo, self.hi = cfg.prior_band
        self.V, self.held = int(self.src["table"].shape[0]), held

    def ok(self, k: int, v: int) -> bool:
        return self.lo <= float(self.geo[k][v]) <= self.hi and float(self.geo[k][v]) < self.kth[k]

    def pools(self, banned: set[int]) -> dict[str, list[int]]:
        """Amendment 6's: E58's cipher pool (ranks 1000-2999), then every word-like token from rank 1000."""
        r0 = self.ranked[AN.VALUE_RANK0 :]
        return {"cipher_pool": [v for v in r0[: AN.POOL] if v not in banned], "ranked_from_1000": [v for v in r0 if v not in banned]}

    def make(self, w: Any, name: str, seed: int, banned: set[int]) -> dict[str, Any] | None:
        """One draw, saved in E147's form (`windows_<name>_trained.pt`), and its prior on the read rows."""
        got = kv.bijection_from_pools(name, int(self.src["task"].marker), tuple(self.keys), self.pools(banned), seed, self.ok)
        if got is None:
            return None
        m, pool = got
        full = W.LARGE / f"windows_{name}.pt"
        torch.save({**self.src, "task": m, "table": m.table(self.V)}, full)
        split_task(full, self.trained, self.held, W.LARGE, name)
        path = W.rel(W.LARGE / f"windows_{name}_trained.pt")
        prior = W.PVS.forward_prior(w.net, {"windows": {"format": "kv", "path": path}, "name": name, "built": name})
        return {"seed": seed, "pool": pool, "values": list(m.values), "windows": path, "prior": prior, "log10_prior": math.log10(prior)}


def stage_a(w: Any, cfg: Config, th: dict[str, Any], copy_log10: float) -> dict[str, Any]:
    path = OUT / "a.json"
    if path.is_file() and "stopped" not in (a := W.read_json(path)):
        log("stage a done before")
        return a
    b, a6 = Build(w, cfg), W.read_json(OPR / "amend6.json")["maps"]["prior_cipher"]
    out: dict[str, Any] = {"experiment": EXPERIMENT, "copy_log10_prior": copy_log10}
    pc = b.make(w, "pb_pc_rebuilt", 0, b.banned)
    e237 = W.read_json(W.E237 / "summary.json")["log10_prior"]["prior_cipher"]
    out["control"] = {"same_values": pc is not None and pc["values"] == a6["values"], "pool": pc and pc["pool"],
                      "log10_prior": pc and pc["log10_prior"], "e237_log10_prior": e237}  # fmt: skip
    log(f"[a control] seed 0 rebuilds Amendment 6's prior-matched cipher: {out['control']}")
    if not (pc and out["control"]["same_values"] and pc["pool"] == "cipher_pool" and abs(pc["prior"] / 10**e237 - 1) <= th["prior_repro_rel"]):
        stop(path, jsonable(out), "the build does not repeat Amendment 6's prior-matched cipher", log)
    out["tries"] = []
    for s in cfg.draw_seeds:
        rec = b.make(w, TASK, s, b.banned | set(pc["values"]))
        out["tries"].append(rec or {"seed": s, "pool": None})
        log(f"[a draw {s}] " + (f"pool {rec['pool']}, log10 prior {rec['log10_prior']:.3f} (copy {copy_log10:.3f})" if rec else "no pool serves every key"))
        if rec and abs(rec["log10_prior"] - copy_log10) <= cfg.prior_tol:
            out |= {"chosen": rec, "prior_gap": copy_log10 - rec["log10_prior"]}
            write_stage(path, jsonable(out))
            return out
    stop(path, jsonable(out), f"no draw's prior is within {cfg.prior_tol} decades of copy's", log)


# ---- the run ----------------------------------------------------------------------------------------------------------


def summary(cfg: Config, th: dict[str, Any], a: dict[str, Any]) -> None:
    ca, sa = RA.load_config()[0], W.read_json(RA.OUT / "summary.json")
    item = lambda d, i: W.read_json(d / "items" / f"{W.R1.fname(i)}.json")
    reused = [item(RA.OUT, st) for st in ca.states("copy")]
    new, rnd = [item(OUT, st) for st in cfg.states()], [item(OUT, f"random/{st}") for st in cfg.states()]
    bands = {p: c["band"] for p, c in sa["calls"].items()}  # run A's, measured at its seeds 0-2
    prior = {"reused": W.read_json(W.E237 / "summary.json")["log10_prior"]["copy"], "new": a["chosen"]["log10_prior"]}
    out = {"experiment": EXPERIMENT, "pair": ["copy (run A, 10 seeds)", f"{TASK} (3 seeds)"]} | W.score(reused, new, rnd, bands, prior, th)
    write_stage(OUT / "summary.json", jsonable(out))
    W.say(out)


def main() -> int:
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 reads, as every saved reading
    log(f"{EXPERIMENT} run B start; settings {asdict(cfg)}; thresholds {th}")
    if not (RA.OUT / "summary.json").is_file():
        log(f"STOP: run A has not finished ({RA.OUT / 'summary.json'} missing)")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    w = W.World(cfg.random_seed)
    job = W.k_job(w)
    ctl = W.controls(w, job, OUT)
    a = stage_a(w, cfg, th, math.log10(ctl["prior_repeat"]["got"]))
    zero = w.add_task(TASK, a["chosen"]["windows"])
    W.do_nothing(w, job, zero, OUT)
    for s in cfg.seeds:
        w.add_state(TASK, s)
    tr = W.train(w, TASK, a["chosen"]["windows"], cfg.seeds, OUT / "train.json", th["p_learn"])
    w.rd.c[TASK] = W.common_c([tr["states"][st]["size"] for st in cfg.states()])
    log(f"[b] c {w.rd.c[TASK]:.4f}")
    write_stage(OUT / "train.json", jsonable(tr | {"c": w.rd.c[TASK], "done": True}))
    items = cfg.states() + [f"random/{st}" for st in cfg.states()]
    for i, it in enumerate(items):
        W.read_item(w, job, it, i, OUT)
        w.dws.clear()
    summary(cfg, th, a)
    log(f"{EXPERIMENT} run B DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
