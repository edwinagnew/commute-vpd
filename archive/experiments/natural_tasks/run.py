"""E124 (natural_tasks), 2026-11-06: a natural 67M task read by the frozen reuse definition and by re-gating. Pronoun only
since the pre-run revision: Pile windows whose next word is " he" or " she" with exactly one pool name 3-15 tokens back;
C = the name's gender (SSA list). Members on the same windows: natural (the text's pronoun), new-word (E112's label
words " carpet"/" camel" by C), twin (he/she by a seeded per-name shuffle). Plan:
reports/plans/2026-11-06_plan_e124-natural-definition-tasks.md (its Go-ahead changes and "Revision before any run"
override the body); verdicts only from `commute.case_label` and E112's filed bars, read from
experiments/case_label/results/predictions.json. Settings and filed outcomes: results/predictions.json (the run refuses
unless both files are committed unchanged). One cluster debug L40 job per stage, cheapest first; each writes
results/stage<N>.json after every step and prints `STAGE n DONE`:

  0  stage0.py, alone, W0 only: the windows (kept counts per gender asserted before the model loads), E54 elicit seed 0's r through
     the data.LAST path, W0's accuracy per member and gender with the two headroom gates, the possession probe and its
     control per block (P0, l*), W0's removals at l*, each member's route cells and the zero update's (exact).
  1  six fine-tunes (3 members x seeds 0, 1; E112's recipe), each stopped at its scored state (accuracy >= 0.90 and
     full-vocabulary gain >= 1), delta-W in bf16 to results/large/ at once; exit 3 when too little time is left.
  2  the random direction on every state first (exit 1 if over 0.10), P1-P3 and the verdict per state; pruning and
     reverse fine-tuning beside.
  3  stage3.py: r and the stand-in ratio per state, then the filed scoring.

    sbatch cluster/natural_tasks.sbatch <N>      (cluster/natural_tasks_cache.sbatch first, once)
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

from commute import case_label as CL
from commute import definition_reads as DFR
from commute import keydep as KD
from commute import label_reads as LR
from commute import natural_tasks as NT
from commute import removal as RM
from commute import steering as ST
from commute.data import LAST, SETS_DIR, pile_rows
from commute.elicit import summed_norm
from commute.env import REPO_ROOT
from commute.finetune import ArmAConfig, StopTraining, finetune_arm_a
from commute.jobs import hard_exit, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
E112_PRED = HERE.parent / "case_label" / "results" / "predictions.json"
HALF, L = 20, 21  # the prediction position is the last of the L tokens the model sees; the target rides after them
MEMBERS = ("natural", "new_word", "twin")
SHARED = ("probe_rows", "twin_seed", "control_seed", "probe_cap", "probe_seed", "ridge_lambda", "seeds", "checkpoints", "learn_bar",
          "batch", "kl_lambda", "grad_clip", "random_dir_seed", "prune_k", "prune_random", "prune_seed", "reverse_cap",
          "reverse_every", "reverse_bar", "reverse_seed")  # fmt: skip
log = KD.log


@dataclass(frozen=True)
class Config:
    """Every size and seed, read from predictions.json's settings (the SHARED ones asserted equal to E112's)."""

    task_rows: tuple[str, int, int]
    task_rows_sha256: str
    held_rows: tuple[str, int, int]
    held_rows_sha256: str
    probe_rows: tuple[str, int, int]
    names_csv: str
    names_sha256: str
    years: tuple[int, int]
    gender_share: float
    names_per_gender: int
    train_name_frac: float
    split_seed: int
    near: int
    far: int
    answers: tuple[str, str]
    new_words: tuple[str, str]
    n_train_windows: int
    n_fit_windows: int
    n_heldout_windows: int
    window_seed: int
    twin_seed: int
    control_seed: int
    probe_cap: int
    probe_seed: int
    ridge_lambda: float
    seeds: tuple[int, ...]
    checkpoints: tuple[int, ...]
    learn_bar: float
    min_gain: float
    lr: float
    batch: int
    kl_lambda: float
    grad_clip: float
    random_dir_seed: int
    prune_k: tuple[int, ...]
    prune_random: int
    prune_seed: int
    reverse_cap: int
    reverse_every: int
    reverse_bar: float
    reverse_seed: int
    ratio_floor: float
    stage_minutes: float
    finetune_minutes: float

    def __post_init__(self) -> None:
        assert all(n % 2 == 0 for n in (self.n_train_windows, self.n_fit_windows, self.n_heldout_windows)), "balanced by gender"
        assert self.probe_rows[1] + self.probe_rows[2] <= self.task_rows[1], "probe rows must be disjoint from the task rows"
        assert self.held_rows[1] >= self.task_rows[1] + self.task_rows[2], "the held-out file must follow the task rows"
        assert list(self.checkpoints) == sorted(set(self.checkpoints)) and self.reverse_cap % self.reverse_every == 0
        assert self.min_gain == MIN_GAIN and 0 < self.near < self.far <= HALF


def load() -> tuple[Config, dict[str, Any], dict[str, Any]]:
    """(config, E112's rules, E124's predictions); both prediction files must be committed unchanged."""
    require_filed(PRED)
    require_filed(E112_PRED)
    P, P112 = json.loads(PRED.read_text()), json.loads(E112_PRED.read_text())
    assert KD.sha256(E112_PRED) == P["rules_from"]["sha256"], "E112's rules are not the ones filed against"
    assert all(P["settings"][k] == P112["settings"][k] for k in SHARED), "a shared setting differs from E112's"
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"].items()}
    return Config(**settings), P112["rules"], P


def dump(n: int | str, res: dict[str, Any]) -> None:
    (RESULTS / f"stage{n}.json").write_text(json.dumps(jsonable(res), indent=1))


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


# ----------------------------------------------------------------------------- the task (no model)


def cached(spec: tuple[str, int, int], sha: str) -> Tensor:
    """Cached Pile rows, checked against their filed sha256; a job never streams (no network)."""
    path = SETS_DIR / f"{spec[0]}_{spec[1]}_{spec[2]}.npy"
    assert path.is_file(), f"{path} missing: run cluster/natural_tasks_cache.sbatch {spec[1]} {spec[2]} first"
    assert KD.sha256(path) == sha, f"{path} is not the cached file filed"
    return pile_rows(*spec, log=log)


def cut(rows: Tensor, rows2: Tensor, r: NT.Rec) -> Tensor:
    """(n, L) windows; a record's row indexes file 1 then file 2 (file 2's rows offset by len(rows))."""
    two = r["row"] >= len(rows)
    out = torch.empty(len(two), L, dtype=torch.long)
    out[~two] = NT.windows(rows, NT.subset(r, ~two), HALF)
    out[two] = NT.windows(rows2, {**NT.subset(r, two), "row": r["row"][two] - len(rows)}, HALF)
    return out


def build(C: Config) -> dict[str, Any]:
    """Pool, splits, labels and windows (results/large/task_pronoun.pt) from the two cached row files: train and fit
    windows from file 1 (train names; rows mod 5 != 0 and == 0), held-out windows (held-out names) from file 1's rows
    mod 5 == 0 and all of file 2. The pool's counts are file 1's. No model."""
    from commute import vpd

    tok = vpd.tokenizer()
    csv_path = REPO_ROOT / C.names_csv
    assert KD.sha256(csv_path) == C.names_sha256, f"{csv_path} is not the filed name list"
    rows, rows2 = (cached(spec, sha) for spec, sha in ((C.task_rows, C.task_rows_sha256), (C.held_rows, C.held_rows_sha256)))
    counts = torch.bincount(rows.flatten(), minlength=len(tok))
    he, she = NT.single_ids(tok.encode, C.answers)
    new = tuple(NT.single_ids(tok.encode, C.new_words))
    pool, excluded = NT.name_pool(NT.ssa_names(csv_path, C.years, C.gender_share), tok.encode, counts, C.names_per_gender, {he, she, *new})
    groups = [[t for t, m in pool.items() if m == g] for g in (1, 0)]
    train_names, held_names = NT.split_within(groups, C.train_name_frac, C.split_seed)
    match = lambda r: NT.pronoun_records(NT.pronoun_matches(r, torch.tensor(list(pool)), C.near, C.far, HALF, torch.tensor([he, she])), r, pool)
    rec, rec2 = match(rows), match(rows2)
    rec2["row"] += len(rows)  # file 2's rows follow file 1's: windows are cut from the file a row belongs to
    rec = {k: torch.cat([rec[k], rec2[k]]) for k in rec}
    drop = NT.drop_shares(rec, he, she)
    rec = NT.subset(rec, NT.agree_mask(rec, he, she))
    f1, is_train = rec["row"] < len(rows), torch.isin(rec["content"], torch.tensor(train_names))
    held_row = f1 & (rec["row"] % 5 == 0)
    t: dict[str, Any] = {"ab": {"natural": (he, she), "new_word": new}, "pool": pool, "train_names": train_names, "held_names": held_names,
                         "twin": {**CL.twin_y(train_names, C.twin_seed), **CL.twin_y(held_names, C.twin_seed + 1)},
                         "control": CL.control_y([(n, n) for n in pool], C.control_seed)}  # fmt: skip
    info: dict[str, Any] = {"rows_sha256": [C.task_rows_sha256, C.held_rows_sha256], "names_sha256": C.names_sha256, "excluded_names": excluded,
                            "pool": {g: len(x) for g, x in zip(("male", "female"), groups, strict=True)}, "drop_share": {"female": drop["kind0"], "male": drop["kind1"], "all": drop["all"]},
                            "pool_words": [tok.decode([n]) for n in pool], "available": {}, "kept": {}}  # fmt: skip
    parts = {"train": (f1 & ~held_row & is_train, C.n_train_windows, 0), "fit": (held_row & is_train, C.n_fit_windows, 1),
             "held": ((held_row | ~f1) & ~is_train, C.n_heldout_windows, 2)}  # file 2 gives held-out-name windows only
    for k, (keep, n, s) in parts.items():
        r, info["available"][k] = NT.sample_cells(NT.subset(rec, keep), [(0, 0), (1, 1)], n, C.window_seed + s)
        t[k] = {**r, "toks": cut(rows, rows2, r)}
        info["kept"][k + "_from_file2"] = int((r["row"] >= len(rows)).sum())
        info["kept"][k] = {"female": int((r["c"] == 0).sum()), "male": int((r["c"] == 1).sum()), "names": len(set(r["content"].tolist()))}
    torch.save(t, LARGE / "task_pronoun.pt")
    return {"t": t, "info": info, "probe_rows": pile_rows(*C.probe_rows, log=log)}


def task() -> dict[str, Any]:
    return torch.load(LARGE / "task_pronoun.pt", weights_only=False)


def y_of(t: dict[str, Any], member: str, part: str) -> Tensor:
    """(N,) the label y (1: answer A) of each window of a part under a member."""
    if member == "twin":
        return torch.tensor([t["twin"][int(n)] for n in t[part]["content"]])
    return t[part]["c"]


def ab_of(t: dict[str, Any], member: str) -> tuple[int, int]:
    return t["ab"]["new_word" if member == "new_word" else "natural"]


def labels(t: dict[str, Any], member: str, part: str) -> tuple[Tensor, Tensor]:
    return t[part]["toks"], y_of(t, member, part)


# ----------------------------------------------------------------------------- stage 1


def finetune(net: ST.Net, C: Config, t: dict[str, Any], member: str, seed: int) -> tuple[ST.DW, dict[str, Any]]:
    """E112's recipe (LR.finetune's calls) stopped at the first checkpoint with fit accuracy >= learn_bar and full-
    vocabulary log-odds gain of the member's target on the fit windows >= min_gain (else the last): delta-W and record."""
    ab, fit, held = ab_of(t, member), labels(t, member, "fit"), labels(t, member, "held")
    tgt = LR.targets(fit[1], ab)
    T0 = float(DFR.full_log_odds(net, fit[0], tgt, HALF).mean())
    rec: dict[str, Any] = {"T0": T0, "acc_fit": {}, "acc_held": {}, "gain": {}}
    got: dict[str, ST.DW] = {}

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        if step not in C.checkpoints:
            return
        rec["acc_fit"][step], rec["acc_held"][step] = (DFR.accuracy(DFR.label_diff(net, p[0], ab, HALF), p[1]) for p in (fit, held))
        rec["gain"][step] = float(DFR.full_log_odds(net, fit[0], tgt, HALF).mean()) - T0
        log(f"[{member} s{seed}] step {step}: fit accuracy {rec['acc_fit'][step]:.4f}, held-out {rec['acc_held'][step]:.4f}, gain {rec['gain'][step]:.3f}")
        if NT.scored_step(rec["acc_fit"], rec["gain"], C.checkpoints, C.learn_bar, C.min_gain) == (step, True) or step == C.checkpoints[-1]:
            got["dW"] = {k: (p.detach().float() - net.W0[k]).cpu() for k, p in params.items()}
            raise StopTraining

    cfg = ArmAConfig(o_token=LAST, kl_lambda=C.kl_lambda, lr=C.lr, batch=C.batch, steps=C.checkpoints[-1], checkpoints=C.checkpoints,
                     grad_clip=C.grad_clip, seed=seed, ci_len=L)  # fmt: skip
    try:
        finetune_arm_a(net.model, LR.labelled(*labels(t, member, "train"), ab, HALF), LR.labelled(*fit, ab, HALF), cfg, net.device, log, on_step)
    except StopTraining:
        pass
    rec["step"], rec["learned"] = NT.scored_step(rec["acc_fit"], rec["gain"], C.checkpoints, C.learn_bar, C.min_gain)
    rec["norm"] = summed_norm(got["dW"])
    return got["dW"], rec


def stage1(net: ST.Net, C: Config, t0: float) -> int:
    assert json.loads((RESULTS / "stage0.json").read_text()).get("go"), "stage 0 did not pass: stage 1 does not run"
    t, p = task(), RESULTS / "stage1.json"
    res: dict[str, Any] = json.loads(p.read_text()) if p.is_file() else {"states": {}}
    for member in MEMBERS:
        for seed in C.seeds:
            name = f"{member}_s{seed}"
            if name in res["states"]:
                continue
            if time.monotonic() - t0 > (C.stage_minutes - C.finetune_minutes) * 60:
                log(f"[time] too little left for {name}: exit 3, resubmit stage 1")
                return 3
            dW, rec = finetune(net, C, t, member, seed)
            path = LARGE / f"dW_pronoun_{name}.pt"
            torch.save({k: v.bfloat16() for k, v in dW.items()}, path)
            res["states"][name] = {**rec, "path": str(path.relative_to(HERE)), "sha256": KD.sha256(path)}
            say(f"state {name}", res["states"][name])
            dump(1, res)
    return 0


# ----------------------------------------------------------------------------- stage 2


def load_dWs() -> dict[str, ST.DW]:
    s1 = json.loads((RESULTS / "stage1.json").read_text())["states"]
    assert len(s1) == len(MEMBERS) * 2, "stage 1 is not complete"
    for s in s1.values():
        assert KD.sha256(HERE / s["path"]) == s["sha256"], f"{s['path']} is not the saved state"
    return {n: {k: v.float() for k, v in torch.load(HERE / s["path"], weights_only=False).items()} for n, s in s1.items()}


def stage2(net: ST.Net, C: Config, R: dict[str, Any], t0: float) -> int:
    l_star = json.loads((RESULTS / "stage0.json").read_text())["l_star"]
    t, dWs = task(), load_dWs()
    W, B = torch.load(LARGE / "probes_pronoun.pt", weights_only=False)["case"]
    w, b = W[l_star], B[l_star]
    h0 = {k: DFR.resid_at(net, t[k]["toks"], HALF) for k in ("fit", "held")}
    mem = lambda n: n.rsplit("_s", 1)[0]
    res: dict[str, Any] = {"l_star": l_star, "do_nothing": {}, "states": {}}
    rand = RM.random_basis(w.shape[0], 1, [C.random_dir_seed])
    for name, dW in dWs.items():  # control first and alone: the random direction on every state
        fit = labels(t, mem(name), "fit")
        res["do_nothing"][name] = DFR.removal_share(net, fit[0], ab_of(t, mem(name)), fit[1], dW, l_star, rand, HALF, R["gain_floor"])
        say(f"do-nothing {name}", res["do_nothing"][name])
        dump(2, res)
    if not CL.do_nothing_ok([s["share"] for s in res["do_nothing"].values()], R["do_nothing_max"]):
        res["stop"] = f"the random direction's share is over {R['do_nothing_max']} on a state: no state is read"
        log(f"[stop] {res['stop']}")
        dump(2, res)
        return 1
    for name, dW in dWs.items():
        m = mem(name)
        s = res["states"][name] = DFR.state_reads(net, R, labels(t, m, "fit"), labels(t, m, "held"), ab_of(t, m), dW, l_star, w, b, h0, HALF)
        s["verdict"] = CL.verdict(l_star, s["P1"], s["P2"], s["P3"], R["real"])
        say(f"reads {name}", s)
        say(f"verdict {name}", {"verdict": s["verdict"], "meets": CL.meets(s["P1"], s["P2"], s["P3"], R["real"])})
        dump(2, res)
    deadline = t0 + C.stage_minutes * 60
    for name, dW in dWs.items():
        m = mem(name)
        res.setdefault("pruning", {})[name] = DFR.prune(net, t["fit"]["toks"], labels(t, m, "held"), ab_of(t, m), dW, HALF, C.prune_k, C.prune_random, C.prune_seed)
        say(f"pruning {name}", res["pruning"][name])
        dump(2, res)
    for name, dW in dWs.items():  # back to the text's own pronoun, from the state
        m = mem(name)
        res.setdefault("reverse", {})[name] = DFR.reverse(net, C, (t["train"]["toks"], t["train"]["next"]), labels(t, m, "held"), ab_of(t, m), dW, HALF, deadline)
        say(f"reverse {name}", res["reverse"][name])
        dump(2, res)
    return 0


def setup() -> tuple[Config, dict[str, Any], dict[str, Any]]:
    C, R, P = load()
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    return C, R, P


def finish(stage: int, code: int) -> None:
    log(f"=== STAGE {stage} DONE, exit {code} ===")
    hard_exit(code)


def main() -> None:
    """Stages 1 and 2 (stage 0 is stage0.py, stage 3 is stage3.py)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    stage, t0 = ap.parse_args().stage, time.monotonic()
    C, R, _P = setup()
    net = ST.Net("cuda")
    finish(stage, stage1(net, C, t0) if stage == 1 else stage2(net, C, R, t0))


if __name__ == "__main__":
    main()
