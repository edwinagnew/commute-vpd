"""E311 (cell P2.83): Minder et al.'s crosscoder on Fig 4's 48 states, after its planted control. Plan
reports/plans/2026-11-25_plan_p2-83-crosscoder-fix.md; settings, bars and expected values results/predictions.json
(refuses unless committed unchanged). The crosscoder, its stream and its reading are E203's
(`commute.crosscoder_sides`, `commute.crosscoder`), the authors' pinned trainer and settings.

E203's planted control missed because its plant always sat at the answer position, where base latents fire: the
crosscoder wrote the plant into those latents' fine-tune decoders (decoder difference along the plant at cosine
0.99999, base decoder orthogonal to it), and such a latent's error ratio is its own base decoder's overlap, 0.244, over
the authors' 0.2. Run 1 trains the plant three ways (`Plant`): content-independent positions (two crosscoder seeds),
the answer position on half the windows, and E203's own placement at 4x the tokens (a diagnostic; its 200M-token
snapshot is E203's end point). Every variant is read as E203 read it.

  prep           (CPU) the long Pile file the e53/e62 sides read (their Pile rows now fill the token budget)
  inputs i/n     (CPU) the files stage-2 shard i of n reads
  0a             the base against itself: activations equal, relative norms 0.5, no fine-tune-specific latent, FVE, dead
  0b WHERE       the planted control, WHERE = random | answer_half | answer; --seed S (crosscoder seed), --long (4x)
  2 i/n          shard i of n of the 52 sides: Fig 4's 48 states, 3 Gaussian updates, 1 crosscoder reseed
  resumecheck    (CPU, rehearsal) a stopped-and-resumed self run against two uninterrupted ones
  --rehearse     settings.sizes.rehearsal (tiny dictionary, full read sizes); outputs under results/rehearsal/
  --stop-after S checkpoint at step S and exit 3 (the rehearsal's resume test)
Prints every scored number, "STAGE x DONE" and a closing "FINISHED ...".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from commute import crosscoder as CC
from commute import crosscoder_sides as CS
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRED = HERE / "results/predictions.json"
KNOWN3 = EXPERIMENTS / "known_3seeds/results/known3.json"
WHERE = ("random", "answer_half", "answer")
OUT: dict[str, Path] = {}
WRITTEN: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class Config:
    layer: int
    data_seed: int
    random_seed: int
    plant_seed: int
    plant_rate: float
    control_task: str
    tf32: bool
    reseed_state: str
    random_states: list[str]
    extra_states: dict[str, list[str]]  # state -> [task, file]: Fig 4's seed-2 states, outside E203's inventory
    sizes: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        assert set(self.sizes) == {"full", "long", "rehearsal", "rehearsal_long"}, sorted(self.sizes)
        for a, b in (("full", "long"), ("rehearsal", "rehearsal_long")):  # long = the same crosscoder, more tokens
            diff = {k for k in self.sizes[a] if self.sizes[a][k] != self.sizes[b][k]}
            assert diff <= {"tokens", "epochs", "snapshots", "ckpt_every"}, (a, b, diff)
        assert 0 < self.plant_rate < 1


def cc_cfg(block: dict[str, Any]) -> CC.Cfg:
    c = config_from_settings(CC.Cfg, block)
    return dataclasses.replace(c, shared=tuple(c.shared), snapshots=tuple(c.snapshots))


class Stop(Exception):
    """A filed stop: the stage file records why; the job exits 1."""


def write(path: Path, out: dict[str, Any]) -> None:
    write_stage(path, jsonable(out))
    WRITTEN.append(str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path))


def check(out: dict[str, Any], path: Path, name: str, ok: bool, detail: dict[str, Any]) -> None:
    out.setdefault("checks", {})[name] = {"pass": ok, **detail}
    log(f"[check {name}] {'ok' if ok else 'MISSED'} {json.dumps(jsonable(detail))}")
    if not ok:
        out["stopped"] = name
        write(path, out)
        raise Stop(name)


def early(out: dict[str, Any], path: Path, T: dict[str, Any], cfg: CC.Cfg) -> Any:
    """The filed checks of a side's first minutes: batch FVE at the first record at or past step `early_step`, the
    training minutes projected there from the last records' ms per step (chunk builds included), and FVE and dead
    share at the first snapshot."""
    seen: set[str] = set()

    def f(kind: str, r: dict[str, Any]) -> None:
        if kind == "curve" and r["step"] >= T["early_step"] and "curve" not in seen:
            seen.add("curve")
            check(out, path, "early: batch FVE", r["fve_batch"] >= T["fve_batch_early"], {"step": r["step"], "fve_batch": r["fve_batch"], "bar": T["fve_batch_early"]})
            proj = r["ms_per_step"] * cfg.steps / 60_000
            check(out, path, "early: projected minutes", proj <= T["projected_limit_min"], {"step": r["step"], "minutes": proj, "bar": T["projected_limit_min"]})
        if kind == "snapshot" and "snap" not in seen:
            seen.add("snap")
            fv = min(r["fve_base"], r["fve_ft"])
            check(out, path, "early: FVE at the first snapshot", fv >= T["fve_early"], {"fve": fv, "bar": T["fve_early"]})
            check(out, path, "early: dead share at the first snapshot", r["dead"] <= T["dead_max"], {"dead": r["dead"], "bar": T["dead_max"]})

    return f


# ----------------------------------------------------------------------------- inputs


def inventory(C: Config) -> dict[str, Any]:
    """E203's inventory (size_matched `pairs`: every state's file and task, every task's windows) with Fig 4's five
    seed-2 states added (known_3seeds/run.py NEW, copied into settings.extra_states)."""
    _, _, inv = load_experiment("size_matched", "run").pairs()
    for s, (task, file) in C.extra_states.items():
        assert s not in inv["states"] and task in inv["tasks"], s
        inv["states"][s] = {"task": task, "file": file}
    return inv


def fig4_states() -> list[str]:
    cells = json.loads(KNOWN3.read_text())["cells"]
    return sorted({s for c in cells if c["method"] == "share" for s in c["states"]["reuse"] + c["states"]["new"]})


def stage2_sides(C: Config) -> list[tuple[str, str, int]]:
    """(key, state, kind) for the 52 sides: kind 0 the state itself, 1 a Gaussian update at its norms, 2 crosscoder seed 1."""
    out = [(s.replace("/", "__"), s, 0) for s in fig4_states()]
    out += [(f"random_like__{s.replace('/', '__')}", s, 1) for s in C.random_states]
    return out + [(f"{C.reseed_state.replace('/', '__')}__seed1", C.reseed_state, 2)]


def make_sides(C: Config, cfg: CC.Cfg, inv: dict[str, Any], tasks: set[str]) -> CS.Sides:
    W = load_experiment("share_untrained", "windows")  # E203's read windows (src's windows.read_rows lacks e58/prior_cipher)
    reads = {t: W.load_read(t, inv["tasks"][t]) for t in sorted(tasks)}
    kvt = {t: torch.load(REPO / inv["tasks"][t]["path"], weights_only=False)["task"] for t in sorted(tasks) if inv["tasks"][t]["format"] == "kv"}
    return CS.Sides(cfg, C.layer, C.data_seed, reads, kvt, C.tf32, log)


def load_dw(inv: dict[str, Any], state: str) -> dict[str, torch.Tensor]:
    d = torch.load(REPO / inv["states"][state]["file"], weights_only=False, map_location="cpu")
    return {k: v.float() for k, v in d.items()}


# ----------------------------------------------------------------------------- stages


def stage0a(C: Config, cfg: CC.Cfg, T: dict[str, Any], stop_after: int | None) -> None:
    path = OUT["results"] / "stage0a.json"
    out: dict[str, Any] = {"experiment": "E311", "card": torch.cuda.get_device_name()}
    inv = inventory(C)
    S = make_sides(C, cfg, inv, {C.control_task})
    task = S.reads[C.control_task]
    toks = CS.windows(task, "fit")[:8]
    diff = float((S.acts(toks, None) - S.acts(toks, S.zero)).abs().max())
    check(out, path, "self path: equal activations", diff <= T["self_act_diff"], {"max_abs": diff, "bar": T["self_act_diff"]})
    rec, _ = S.run_side(CS.Side("self", task, C.control_task, S.zero), OUT["large"], True, True, early(out, path, T, cfg), stop_after)
    out["self"] = rec
    write(path, out)
    check(out, path, "self: relative norms at 0.5", rec["max_rn_dev_alive"] <= T["self_rn_dev"], {"max_dev": rec["max_rn_dev_alive"], "bar": T["self_rn_dev"]})
    check(out, path, "self: no fine-tune-specific latent", rec["counts"]["ft_specific"] == 0, {"counts": rec["counts"]})
    check(out, path, "self: FVE", min(rec["fve_base"], rec["fve_ft"]) >= T["fve_min"], {"fve": [rec["fve_base"], rec["fve_ft"]], "bar": T["fve_min"]})
    check(out, path, "self: dead share", rec["dead"] <= T["dead_max"], {"dead": rec["dead"], "bar": T["dead_max"]})
    out["pass"] = True
    write(path, out)


def stage0b(C: Config, cfg: CC.Cfg, T: dict[str, Any], where: str, seed: int, long: bool) -> None:
    """One planted crosscoder. A miss of the filed rule is a reading (pass false, exit 0); an early check stops."""
    tag = f"{where}_s{seed}{'_long' if long else ''}"
    path = OUT["results"] / f"stage0b_{tag}.json"
    out: dict[str, Any] = {"experiment": "E311", "card": torch.cuda.get_device_name(), "where": where, "seed": seed, "long": long}
    inv = inventory(C)
    S = make_sides(C, cfg, inv, {C.control_task})
    side = S.planted(C.control_task, where, C.plant_rate, C.plant_seed, seed)
    side.key += "_long" if long else ""
    rec, _ = S.run_side(side, OUT["large"], True, True, early(out, path, T, cfg))
    out["planted"] = rec
    lead = rec.get("lead", {})
    ok = bool(lead.get("specific")) and rec["specific"] >= T["planted_part"]
    out["planted_pass"] = ok
    out["rule"] = {"lead": lead, "part": rec["specific"], "bar": T["planted_part"], "nu_eps_bar": cfg.nu_eps, "nu_r_bar": cfg.nu_r}
    for s in rec["snapshots"]:
        log(f"[0b {tag}] snapshot tokens {s['tokens']}: part {s['specific']:.4f} lead {json.dumps(jsonable(s.get('lead', {})))}")
    log(f"[0b {tag}] PLANTED {'PASS' if ok else 'MISS'}: lead {json.dumps(jsonable(lead))}, part {rec['specific']:.4f} (bar {T['planted_part']})")
    out["pass"] = True  # the stage ran; planted_pass is the reading
    write(path, out)


def stage2(C: Config, cfg: CC.Cfg, T: dict[str, Any], shard: tuple[int, int], t_job: float) -> None:
    s0 = OUT["results"] / "stage0a.json"
    if not (s0.is_file() and json.loads(s0.read_text()).get("pass")):
        raise SystemExit("refusing stage 2: stage0a.json has not passed (copy the pod's stage0a.json here first)")
    r0 = json.loads(s0.read_text())["self"]
    fve0 = min(r0["fve_base"], r0["fve_ft"])
    todo = [s for i, s in enumerate(stage2_sides(C)) if i % shard[1] == shard[0] and not (OUT["sides"] / f"{s[0]}.json").is_file()]
    path = OUT["results"] / f"stage2_{shard[0]}_{shard[1]}.json"
    out: dict[str, Any] = {"experiment": "E311", "shard": list(shard), "todo": [k for k, _, _ in todo]}
    inv = inventory(C)
    S = make_sides(C, cfg, inv, {inv["states"][s]["task"] for _, s, _ in todo})
    log(f"[stage 2] shard {shard[0]}/{shard[1]}: {len(todo)} sides to train")
    for i, (key, state, kind) in enumerate(todo):
        task, dW = inv["states"][state]["task"], load_dw(inv, state)
        if kind == 1:
            side = S.random_like(key, task, dW, C.random_seed + C.random_states.index(state))
        else:
            side = S.state(key, task, dW, seed=1 if kind == 2 else 0)
        rec, _ = S.run_side(side, OUT["large"], False, False, early(out, path, T, cfg))
        write(OUT["sides"] / f"{key}.json", rec | {"state": state, "kind": ["state", "gaussian", "reseed"][kind]})
        log(f"[score] {key} d_authors {rec['d_authors']:.4f} d_alt {rec['d_alt']:.4f} specific {rec['specific']:.4f} shared_frac {rec['shared_frac']}")
        f = min(rec["fve_base"], rec["fve_ft"])
        check(out, path, f"FVE {key}", f >= fve0 - T["fve_drop"], {"fve": f, "self": fve0, "drop": T["fve_drop"]})
        check(out, path, f"dead share {key}", rec["dead"] <= T["dead_max"], {"dead": rec["dead"], "bar": T["dead_max"]})
        log(f"[stage 2] {i + 1}/{len(todo)} done ({key}); job minutes so far {(time.time() - t_job) / 60:.0f}")
    out["pass"] = True
    write(path, out)


def resumecheck() -> None:
    """Rehearsal: losses at matched steps of the resumed self run (stage0a.json) against an uninterrupted one
    (stage0a_A.json), beside the floor two uninterrupted runs give (A against stage0a_B.json). Pass when every
    post-resume difference is within twice the floor plus 1e-6 (E203's rule)."""
    c = {k: {r["step"]: r["loss"] for r in json.loads((OUT["results"] / f"stage0a{k}.json").read_text())["self"]["curve"]} for k in ("_A", "_B", "")}
    at = json.loads((OUT["results"] / "stage0a.json").read_text())["self"]["resumed_at"]
    def rel(x: float, y: float) -> float:
        return abs(x - y) / max(abs(y), 1e-12)

    floor = max(rel(c["_B"][s], c["_A"][s]) for s in c["_A"] if s in c["_B"])
    after = {s: rel(c[""][s], c["_A"][s]) for s in c["_A"] if s in c[""] and s >= at}
    ok = bool(after) and at > 0 and max(after.values()) <= 2 * floor + 1e-6
    out = {"resumed_at": at, "floor_max_rel": floor, "resumed_max_rel": max(after.values()) if after else math.nan, "steps": len(after), "pass": ok}
    write(OUT["results"] / "resume_check.json", out)
    log(f"[resumecheck] {'PASS' if ok else 'FAIL'} {json.dumps(jsonable(out))}")
    if not ok:
        raise Stop("resume check")


def prep(C: Config, cfg: CC.Cfg) -> None:
    """The Pile rows the e53/e62 sides read: the largest pile_rows over their tasks, from pile_start."""
    inv = inventory(C)
    W = load_experiment("share_untrained", "windows")
    tasks = {inv["states"][s]["task"] for _, s, _ in stage2_sides(C)}
    n = max(CS.stream_text(cfg, W.load_read(t, inv["tasks"][t]), False)["pile_rows"] for t in tasks if inv["tasks"][t]["format"] != "kv")
    if any(s <= cfg.pile_start and cfg.pile_start + n <= s + m for s, m, _ in CS.cached("train")):
        log(f"[prep] rows {cfg.pile_start}..{cfg.pile_start + n} already cached")
        return
    CS.prep_rows(cfg.pile_start, n, log)


def inputs(C: Config, shard: tuple[int, int]) -> None:
    inv = inventory(C)
    sides = [s for i, s in enumerate(stage2_sides(C)) if i % shard[1] == shard[0]]
    files = sorted({inv["states"][s]["file"] for _, s, _ in sides})
    tasks = sorted({inv["tasks"][inv["states"][s]["task"]].get("path") or "(built from the Pile val set)" for _, s, _ in sides})
    mb = sum((REPO / f).stat().st_size for f in files) / 1e6
    print(json.dumps({"shard": f"{shard[0]}/{shard[1]}", "sides": [k for k, _, _ in sides], "states": files, "tasks": tasks, "state_mb": round(mb)}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "inputs", "0a", "0b", "2", "resumecheck"])
    ap.add_argument("arg", nargs="?", default="0/1")  # a shard i/n, or stage 0b's WHERE
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args()
    t_job, P = time.time(), json.loads(PRED.read_text())
    C = config_from_settings(Config, P["settings"])
    tag = {(False, False): "full", (False, True): "long", (True, False): "rehearsal", (True, True): "rehearsal_long"}[(a.rehearse, a.long)]
    cfg, T = cc_cfg(C.sizes[tag]), P["thresholds"]["rehearsal" if a.rehearse else "full"]
    T = T | (T["long"] if a.long else {})
    sub = "rehearsal" if a.rehearse else ""
    OUT.update(results=HERE / "results" / sub, sides=HERE / "results" / sub / "sides", large=HERE / "results/large" / sub)
    log(f"[job] {a} sizes {tag}: {cfg}")
    code = 0
    try:
        if a.stage == "prep":
            prep(C, cfg)
        elif a.stage == "inputs":
            inputs(C, tuple(int(x) for x in a.arg.split("/")))  # type: ignore[arg-type]
        elif a.stage == "resumecheck":
            resumecheck()
        else:
            require_filed(PRED)
            if a.stage == "0a":
                stage0a(C, cfg, T, a.stop_after)
            elif a.stage == "0b":
                if a.arg not in WHERE:
                    raise SystemExit(f"stage 0b takes one of {WHERE}")
                stage0b(C, cfg, T, a.arg, a.seed, a.long)
            else:
                i, n = (int(x) for x in a.arg.split("/"))
                stage2(C, cfg, T, (i, n), t_job)
            log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        print(f"STAGE {a.stage} DONE", flush=True)
    except Stop as e:
        log(f"STOP: {e}: tell the lead")
        code = 1
    log(f"FINISHED {a.stage} {a.arg} in {(time.time() - t_job) / 60:.1f} min, exit {code}; wrote: {' '.join(dict.fromkeys(WRITTEN)) or 'nothing'}")
    return code


if __name__ == "__main__":
    hard_exit(main())
