"""row_blocks (E49): does one block of 128 Pile rows, or the adversary's start, decide the SGD teacher-trained
decompositions' verdict, and where does full-matrix SGD sit on new rows? Plan: reports/plans/2026-10-29_plan_e49-row-blocks.md.

The check is E39's: the Pile harness's 20-step mask adversary from one random start (adversary seed s), one adversary
per block of 128 val rows (batches of 8), read here PER BLOCK as the state's KL over W0's KL on the same block at the
same seed in the same job (`faithfulness.harness_pgd`). E39's lines: pass <= 1.40, fail >= 1.61, both inclusive.
States: the four teacher-trained decompositions E39 and its amendments saved (bf16 offsets, loaded with
`saved_states.offsets_state`; sha256 checked) and the do-nothing states of the Adam, plain SGD and LoRA fine-tunes
(`lowercase.finetune`, regenerated once here and saved fp32). Every job runs on the cluster's L40 (debug QoS).

Stages, one Slurm job each, in this order (each job reads W0 first and stops on a missed control):
  regen       the three fine-tunes regenerated (held-out gain within 0.005 of the filed H100 gain, else stop), saved to
              results/large/dW_<name>.pt; Pile val rows 512-1023 cached for the b jobs
  a           rows 0-255. W0 on rows 0-127 at seed 0 and rows 128-255 at seeds 0-2 (must equal E47's L40 readings, else
              stop); the Adam teacher-trained decomposition at seed 0 on both blocks (E47's L40 reading, and failing, else
              stop); the LoRA's (E47's L40 reading at seed 0, else stop), then both SGD ones: rows 0-127 at seed 0 and rows
              128-255 at seeds 0-2 (P1)
  b<R>s<S>    rows R..R+255 (R = 512 or 768) in two blocks at adversary seed S (0, 1 or 2): W0 (in band, else stop), the
              Adam do-nothing (pair check failing, else stop), the SGD and LoRA do-nothing, the four teacher-trained
              decompositions (P2-P5)
  score       CPU: the filed rules on every finished results file -> results/score.json and the log

Results results/<stage>.json (rewritten after every read; a rerun on the same card re-reads W0, refuses if it differs,
and skips the reads already there); logs results/large/<stage>.log.

Usage (repo root, project venv):  run.py --predictions    files results/predictions.json (once)
                                  run.py <stage>          one stage (GPU, except score)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED = RESULTS / "predictions.json"
BF = HERE.parent / "broad_fragility" / "results" / "large"
TRANSPORT = HERE.parent / "transport_test" / "results"
BLOCK = 128

# The teacher-trained decompositions (E37's teacher training: Adam 1e-5, batch 8, 1,000 steps, batch order 0), as saved:
TT = {"tt_adam": ("trained_step1000.pt", "d7da3c255023c9c46f43f59183edae6329df94d545bff516812721a98a1092db"),  # E39
      "tt_sgd": ("tt_sgd_step1000.pt", "ced08dea928251e3540abef8ae0d9b90f8833bdc6cccb9f78b5cf15093105ac4"),  # E39 A1
      "tt_sgd_r8": ("a2_tt_sgd_r8.pt", "4c9d24df679b00eb96f6a0f2f25f27b30fc4698318ad40264392886c6429f87a"),  # E39 A2
      "tt_lora": ("a2_tt_lora_s0.pt", "20b3d068b04955e07736d49c8df56ea0fd9ab6462ec06eb98f4aa9d752d67bb9")}  # E39 A2
FT = ("adam", "sgd", "lora")  # `nothing_<ft>`: that fine-tune's delta-W in Delta, released subcomponents
SGD_TT = ("tt_sgd", "tt_sgd_r8")
B_STATES = ("nothing_adam", "nothing_sgd", "nothing_lora", "tt_adam", "tt_sgd", "tt_sgd_r8", "tt_lora")


@dataclass(frozen=True)
class Config:
    pass_line: float = 1.40  # E39's lines, 1.5 / (1 + f) and 1.5 (1 + f) at its H100 floor f = 0.075
    fail_line: float = 1.61
    pgd_steps: int = 20
    gain_ref: dict[str, float] = field(default_factory=lambda: {"adam": 0.1301, "sgd": 0.13671, "lora": 0.1479})
    gain_tol: float = 0.005  # absolute; E39's gain_tol. L40 - H100 measured -0.00047 (Adam), -0.00016 (LoRA), E47 t2c
    hard_block: int = 128  # val rows 128-255, where both SGD fails sat (E39 A3)
    a_seeds: tuple[int, ...] = (0, 1, 2)
    # W0's PGD KL per block on the L40 (E47 logs t2b, t4: rows 0-127 / 128-255 at h0, h1, h2; four decimals)
    a_w0: dict[str, float] = field(default_factory=lambda: {"0|0": 0.6671, "128|0": 0.7164, "128|1": 1.0007,
                                                            "128|2": 0.6846})
    w0_block_tol: float = 1e-4  # absolute: the references' rounding; E47 jobs agreed to 17 digits on the L40
    # mean PGD KL over rows 0-255 at seed 0 on the L40, the same files (E47 t2/t2b results JSON)
    a_pair_kl: dict[str, float] = field(default_factory=lambda: {"tt_adam": 1.5018157362937927,
                                                                 "tt_lora": 0.9174503982067108})
    repro_tol: float = 1e-3  # relative; E47's f_repro_tol
    h100_first: dict[str, float] = field(default_factory=lambda: {"tt_sgd": 2.0675, "tt_sgd_r8": 2.1218})  # A3, rows 0-255
    card_tol: float = 0.10  # relative, reported only: same file and rows, L40 vs H100, measured 0.5-4.2% (four pairs)
    b_rows: tuple[int, ...] = (512, 768)  # each 256 rows, two blocks
    b_seeds: tuple[int, ...] = (0, 1, 2)
    w0_ci: float = 0.34  # W0's CI-masked KL per block, L40: 0.331-0.352 on the four blocks of val 0-511 (E47)
    w0_ci_tol: float = 0.15  # relative
    w0_pgd_band: tuple[float, float] = (0.45, 1.5)  # W0's PGD KL per block, L40: 0.620-1.001 (12 block-seed readings)

    def __post_init__(self) -> None:
        assert self.pass_line < self.fail_line and set(self.gain_ref) == set(FT)
        assert abs((self.a_w0["0|0"] + self.a_w0["128|0"]) / 2 / 0.6917431652545929 - 1) < 1e-4  # E47's rows 0-255 mean
        assert all(r % 256 == 0 and r >= 512 for r in self.b_rows)  # no overlap with val 0-511 (every earlier check)
        assert self.hard_block == BLOCK and self.a_seeds == self.b_seeds == (0, 1, 2)


CFG = Config()
STAGES = ("regen", "a", *(f"b{r}s{s}" for r in CFG.b_rows for s in CFG.b_seeds), "score")
LOG: list[Path] = []


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    for p in LOG:
        with p.open("a") as fh:
            fh.write(line + "\n")


def config() -> dict[str, Any]:
    return json.loads(json.dumps(asdict(CFG)))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def key(name: str, block: int, seed: int) -> str:
    return f"{name}|{block}|{seed}"


def ratio(kl: dict[str, float], name: str, block: int, seed: int) -> float | None:
    """The state's KL over W0's on the same block at the same seed; None if either is missing."""
    k, w = kl.get(key(name, block, seed)), kl.get(key("W0", block, seed))
    return None if k is None or w is None else k / w


def pair(kl: dict[str, float], name: str, row0: int, seed: int) -> float | None:
    """E39's 256-row check on blocks row0 and row0 + 128: the mean KL over W0's mean."""
    ks = [kl.get(key(n, row0 + d, seed)) for n in (name, "W0") for d in (0, BLOCK)]
    return None if None in ks else (ks[0] + ks[1]) / (ks[2] + ks[3])  # type: ignore[operator]


# ----------------------------------------------------------------------------- the GPU job


class Job:
    def __init__(self, stage: str, device: str) -> None:
        import torch

        from commute import harness

        self.stage, self.path = stage, RESULTS / f"{stage}.json"
        LARGE.mkdir(parents=True, exist_ok=True)
        LOG[:] = [LARGE / f"{stage}.log"]
        token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
        self.ctx = harness.load(device, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token, 1e-6,
                                0, CFG.pgd_steps, (), log)
        self.card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        self.old = json.loads(self.path.read_text()) if self.path.is_file() else None
        if self.old is not None and self.old["card"] != self.card:
            sys.exit(f"{self.path} was written on {self.old['card']}; this job runs on {self.card}. Move it aside")
        self.out: dict[str, Any] = {"stage": stage, "card": self.card, "kl": {}, "ci": {}, "seconds": {}, "gates": {},
                                    "status": "running", "files": {}}
        self._blocks: dict[int, list[Any]] = {}
        self._state: tuple[str, Any] = ("", None)
        log(f"[{stage}] {self.card}")
        if "L40" not in self.card:
            self.halt(f"the references are L40 readings; this job runs on {self.card}")

    def dump(self) -> None:
        self.path.write_text(json.dumps(self.out, indent=1, sort_keys=True))

    def halt(self, why: str) -> None:
        self.out["status"] = f"stopped: {why}"
        self.dump()
        log(f"STOP: {why}")
        sys.exit(f"STOP: {why}")

    def gate(self, name: str, ok: bool, rec: dict[str, Any], stop: bool = True) -> None:
        self.out["gates"][name] = {**rec, "pass": ok}
        log(f"[gate {name}] {json.dumps(self.out['gates'][name])}")
        self.dump()
        if stop and not ok:
            self.halt(f"{name}: {json.dumps(rec)}")

    def block(self, b: int) -> list[Any]:
        """Pile val rows b .. b + 127 in batches of 8 (`faithfulness.pile_blocks`' layout), from the 256-row set."""
        from commute.data import batches, pile_rows

        start = b - b % 256
        if start not in self._blocks:
            rows = pile_rows("val", start, 256, log)
            self._blocks[start] = [batches(rows[s : s + BLOCK], 8) for s in (0, BLOCK)]
        return self._blocks[start][(b - start) // BLOCK]

    def state(self, name: str) -> Any:
        """(delta-W, alive (U', V') or None) of a saved state, its file's sha256 checked; the last one is kept."""
        import torch

        from commute.saved_states import offsets_state

        if self._state[0] != name:
            if name in TT:
                path, want = BF / TT[name][0], TT[name][1]
            else:
                path = LARGE / f"dW_{name.removeprefix('nothing_')}.pt"
                want = json.loads((RESULTS / "regen.json").read_text())["files"][path.name]
            got = sha256(path)
            if got != want:
                self.halt(f"{path} has sha256 {got}, not {want}")
            self.out["files"][path.name] = got
            upd = offsets_state(path, self.ctx.uv_cpu) if name in TT else (torch.load(path, weights_only=True), None)
            self._state = (name, upd)
        return self._state[1]

    def read(self, name: str, b: int, seed: int) -> float:
        """One block's PGD KL at one seed, stored and logged with its ratio to W0's; W0 is read with `faithfulness`
        (its CI-masked KL kept for the control) and must equal an earlier file's; a state already read is skipped."""
        from commute import harness
        from commute.faithfulness import faithfulness, harness_pgd

        k, ctx = key(name, b, seed), self.ctx
        if name != "W0" and self.old is not None and k in self.old["kl"]:
            self.out["kl"][k] = self.old["kl"][k]
            return self.out["kl"][k]
        t0 = time.time()
        if name == "W0":
            f = faithfulness(ctx.model, self.block(b), pgd_steps=CFG.pgd_steps, seed=seed, log=log)
            kl, self.out["ci"][k] = f["pgd_kl"], f["ci_masked_kl"]
            if self.old is not None and k in self.old["kl"] and abs(kl / self.old["kl"][k] - 1) > 1e-6:
                sys.exit(f"W0 {k} reads {kl}, the file {self.old['kl'][k]}: not the same apparatus; move the file aside")
        else:
            with harness.at(ctx, *self.state(name)):
                kl = harness_pgd(ctx.model, self.block(b), CFG.pgd_steps, seed)
        self.out["kl"][k], self.out["seconds"][k] = kl, time.time() - t0
        self.dump()
        r = ratio(self.out["kl"], name, b, seed)
        log(f"[{name} rows {b}-{b + BLOCK - 1} seed {seed}] kl {kl:.6f} ratio {r:.4f} ({time.time() - t0:.0f} s)")
        return kl

    def finish(self) -> None:
        import resource

        import torch

        self.out["status"] = "done"
        self.out["peak"] = {"gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
                            "host_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
        self.dump()
        log(f"[{self.stage}] done; peak {json.dumps(self.out['peak'])}; FINISHED wrote {self.path}")


def regen(job: Job) -> None:
    """The three fine-tunes, each gain-checked against its filed H100 gain and saved fp32 (temporary name, then
    renamed); a file already saved with a matching sha256 is kept. Then the b jobs' rows are cached."""
    import torch

    from commute import lowercase
    from commute.data import pile_rows

    ctx, old = job.ctx, (job.old or {}).get("files", {})
    low = lowercase.task(ctx.model, ctx.device, log)
    job.out["gains"], job.out["gates"] = (job.old or {}).get("gains", {}), (job.old or {}).get("gates", {})
    for name in FT:
        path = LARGE / f"dW_{name}.pt"
        if path.name in old and path.is_file() and sha256(path) == old[path.name]:
            job.out["files"][path.name] = old[path.name]
            log(f"[regen {name}] kept {path.name}")
            continue
        dW = lowercase.finetune(ctx.model, ctx.device, low, name, log)
        g = lowercase.gain(ctx.model, ctx.device, low, dW)
        job.out["gains"][name] = g
        job.gate(f"G_gain_{name}", abs(g - CFG.gain_ref[name]) <= CFG.gain_tol, {"gain": g, "ref": CFG.gain_ref[name]})
        tmp = path.with_suffix(".tmp")
        torch.save(dW, tmp)
        os.replace(tmp, path)
        job.out["files"][path.name] = sha256(path)
        job.dump()
        log(f"[regen {name}] gain {g:.5f} (filed {CFG.gain_ref[name]}) saved {path.name} {job.out['files'][path.name]}")
    for r in CFG.b_rows:
        pile_rows("val", r, 256, log)


def stage_a(job: Job) -> None:
    kl = job.out["kl"]
    for b, s in ((0, 0), *((CFG.hard_block, s) for s in CFG.a_seeds)):
        job.read("W0", b, s)
    w0 = {k: kl[key("W0", *map(int, k.split("|")))] for k in CFG.a_w0}
    job.gate("G_w0_a", all(abs(w0[k] - v) <= CFG.w0_block_tol for k, v in CFG.a_w0.items()), {"read": w0, "ref": CFG.a_w0})
    for b in (0, CFG.hard_block):
        job.read("tt_adam", b, 0)
    m, c = (kl[key("tt_adam", 0, 0)] + kl[key("tt_adam", 128, 0)]) / 2, pair(kl, "tt_adam", 0, 0)
    job.gate("G_adam_a", abs(m / CFG.a_pair_kl["tt_adam"] - 1) <= CFG.repro_tol and c is not None and c >= CFG.fail_line,
             {"mean_kl": m, "ref": CFG.a_pair_kl["tt_adam"], "check": c})
    for name in ("tt_lora", *SGD_TT):
        job.read(name, 0, 0)
        for s in CFG.a_seeds:
            job.read(name, CFG.hard_block, s)
        m, c = (kl[key(name, 0, 0)] + kl[key(name, 128, 0)]) / 2, pair(kl, name, 0, 0)
        if name == "tt_lora":
            job.gate("G_lora_a", abs(m / CFG.a_pair_kl[name] - 1) <= CFG.repro_tol, {"mean_kl": m, "ref": CFG.a_pair_kl[name],
                                                                                   "check": c})
        else:  # the L40 against the H100 on the same file and rows: reported, never a stop
            job.gate(f"G_card_{name}", c is not None and abs(c / CFG.h100_first[name] - 1) <= CFG.card_tol,
                     {"check": c, "h100": CFG.h100_first[name]}, stop=False)


def stage_b(job: Job, row0: int, seed: int) -> None:
    kl, blocks = job.out["kl"], (row0, row0 + BLOCK)
    for b in blocks:
        job.read("W0", b, seed)
    lo, hi = CFG.w0_pgd_band
    w0 = {b: (kl[key("W0", b, seed)], job.out["ci"][key("W0", b, seed)]) for b in blocks}
    job.gate("G_w0_b", all(lo <= p <= hi and abs(c / CFG.w0_ci - 1) <= CFG.w0_ci_tol for p, c in w0.values()),
             {"pgd_ci": {str(b): v for b, v in w0.items()}})
    for name in B_STATES:
        for b in blocks:
            job.read(name, b, seed)
        log(f"[{name} rows {row0}-{row0 + 255} seed {seed}] pair check {pair(kl, name, row0, seed):.4f}")
        if name == "nothing_adam":
            c = pair(kl, name, row0, seed)
            job.gate("G_nothing_b", c is not None and c >= CFG.fail_line, {"check": c})


# ----------------------------------------------------------------------------- scoring (pure)


def verdict(r: float | None) -> str:
    if r is None or not math.isfinite(r):
        return "invalid"
    return "pass" if r <= CFG.pass_line else "fail" if r >= CFG.fail_line else "between"


def a_label(kl: dict[str, float], name: str) -> str:
    """P1's rule for one SGD state on the hard block at seeds 1 and 2: 'rows' if both ratios >= the fail line, 'start'
    if both <= the pass line, 'mixed' otherwise; 'not run' if one is missing, 'invalid' if one is not finite."""
    rs = [ratio(kl, name, CFG.hard_block, s) for s in (1, 2)]
    if None in rs:
        return "not run"
    v = [verdict(r) for r in rs]
    return "invalid" if "invalid" in v else "rows" if v == ["fail"] * 2 else "start" if v == ["pass"] * 2 else "mixed"


def b_count(kl: dict[str, float], name: str) -> dict[str, Any]:
    """Per new block the median over seeds 0-2 of the ratio; the number of blocks whose median is >= the fail line
    (None unless all four blocks have three finite ratios), and seed 0's own count beside it."""
    blocks = [r + d for r in CFG.b_rows for d in (0, BLOCK)]
    rs = {b: [ratio(kl, name, b, s) for s in CFG.b_seeds] for b in blocks}
    ok = {b: all(r is not None and math.isfinite(r) for r in v) for b, v in rs.items()}
    med = {b: statistics.median(v) if ok[b] else None for b, v in rs.items()}  # type: ignore[type-var]
    s0 = [v[0] for v in rs.values()]
    return {"ratios": {str(b): v for b, v in rs.items()}, "median": {str(b): m for b, m in med.items()},
            "fails": sum(m >= CFG.fail_line for m in med.values()) if all(ok.values()) else None,  # type: ignore[operator]
            "fails_seed0": sum(r >= CFG.fail_line for r in s0) if None not in s0 else None,  # type: ignore[operator]
            "pairs": {f"{r}|{s}": pair(kl, name, r, s) for r in CFG.b_rows for s in CFG.b_seeds}}


def w0_floor(kl: dict[str, float], b: int) -> float | None:
    """W0's own spread on block b over seeds 0-2, max / min - 1 (reported beside the ratios, not a line)."""
    v = [kl[k] for s in CFG.b_seeds if (k := key("W0", b, s)) in kl]
    return max(v) / min(v) - 1 if len(v) == len(CFG.b_seeds) else None


def outcome(holds: bool | None, why: str) -> str:
    return f"unscored: {why}" if holds is None else "holds" if holds else "misses"


def score(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The filed rules on the finished results files (a stage counts only with status 'done')."""
    done = {s: f for s, f in files.items() if f.get("status") == "done"}
    sc: dict[str, Any] = {"stages_done": sorted(done)}
    a = done.get("a")
    if a is not None:
        kl = a["kl"]
        sc["a"] = {n: {"hard_block": [ratio(kl, n, CFG.hard_block, s) for s in CFG.a_seeds], "rows_0_255_seed0": pair(kl, n, 0, 0),
                       **({"label": a_label(kl, n)} if n in SGD_TT else {})} for n in ("tt_lora", *SGD_TT)}
        labels = {sc["a"][n]["label"] for n in SGD_TT}
        sc["a_reading"] = labels.pop() if len(labels) == 1 and labels <= {"rows", "start", "mixed"} else (
            "unscored" if labels & {"invalid", "not run"} else "mixed")
        sc["a"]["w0_hard_block"] = [kl[key("W0", CFG.hard_block, s)] for s in CFG.a_seeds]
    p1 = sc.get("a_reading")
    sc["P1"] = outcome(None if p1 in (None, "unscored") else p1 == "rows", "stage a did not finish or a ratio is invalid")
    kl_b = {k: v for s, f in done.items() if s.startswith("b") for k, v in f["kl"].items()}
    sc["b"] = {n: b_count(kl_b, n) for n in B_STATES}
    sc["b"]["W0_floor"] = {str(b): w0_floor(kl_b, b) for r in CFG.b_rows for b in (r, r + BLOCK)}
    n = {s: sc["b"][s]["fails"] for s in B_STATES}
    why = "a block lacks three finite ratios (a b job did not finish, or a ratio is not finite)"
    sc["P2"] = outcome(None if n["tt_adam"] is None else n["tt_adam"] == 4, why)
    sc["P3"] = outcome(None if n["tt_sgd"] is None else n["tt_sgd"] <= 1, why)
    sc["P4"] = outcome(None if n["tt_lora"] is None else n["tt_lora"] == 0, why)
    sc["P5"] = outcome(None if n["tt_sgd_r8"] is None else n["tt_sgd_r8"] <= 1, why)
    sc["b_reading"] = b_reading(n)
    sc["tree"] = {"a": READINGS["a"][sc.get("a_reading") or "not run"], "b": READINGS["b"][sc["b_reading"]]}
    return sc


def b_reading(n: dict[str, int | None]) -> str:
    sgd, adam, power = n["tt_sgd"], n["tt_adam"], n["nothing_sgd"]
    if sgd is None or adam is None or power is None:
        return "unscored"
    if sgd >= 2:
        return "sgd_with_adam"
    base = "sgd_passes_most" if adam == 4 else "adam_not_every_block"
    return base + ("_low_power" if power <= 2 else "")


READINGS: dict[str, dict[str, str]] = {
    "a": {
        "rows": "Both SGD teacher-trained decompositions fail rows 128-255 at seeds 1 and 2 as well: the block's fail is "
                "not one adversary start; it belongs to the rows. Next (a later $0 job, by this tree): per-row KL under "
                "the block's final mask, to say whether a handful of rows carry it.",
        "start": "Both SGD teacher-trained decompositions pass rows 128-255 at seeds 1 and 2: the block's fail was the "
                 "adversary's start. Multi-start readings become a question for every line: a decision card for the user.",
        "mixed": "The SGD states do not agree, or a seed lands between the lines: start and rows are not separated on "
                 "this block; the b reading decides where SGD sits, and per-row KL is not run.",
        "unscored": "Stage a stopped at a control, or a ratio is not finite: unanswered.",
        "not run": "Stage a has not finished.",
    },
    "b": {
        "sgd_with_adam": "Plain SGD's teacher-trained decomposition fails two or more of the four new blocks: full-matrix "
                         "SGD sits with Adam, and Amendment 3's two remaining stages, option (d), go to the user as a card.",
        "sgd_passes_most": "Plain SGD fails at most one of the four new blocks and Adam fails all four: the line closes "
                           "'Adam fails, the LoRA passes, SGD passes on most rows'; option (d) is dropped.",
        "adam_not_every_block": "Plain SGD fails at most one new block, and Adam's teacher-trained decomposition also "
                                "passes on at least one: Adam's fail is block-dependent too; not covered by the card's "
                                "advice, so it goes to the user before anything else runs on the broad line.",
        "unscored": "A b job stopped at a control or did not run: unanswered.",
    },
}
for _k in ("sgd_passes_most", "adam_not_every_block"):
    READINGS["b"][_k + "_low_power"] = (READINGS["b"][_k] + " But the SGD do-nothing fails at most two of the four "
                                        "blocks, so on the others the check could not fail SGD's update at all: the "
                                        "SGD passes there carry little weight, and this is said wherever the reading is.")


def run_score() -> None:
    LOG[:] = [LARGE / "score.log"]
    files = {s: json.loads(p.read_text()) for s in STAGES if (p := RESULTS / f"{s}.json").is_file()}
    sc = score(files)
    (RESULTS / "score.json").write_text(json.dumps(sc, indent=1, sort_keys=True))
    for k in ("stages_done", "a", "a_reading", "P1", "b", "P2", "P3", "P4", "P5", "b_reading", "tree"):
        log(f"[score {k}] {json.dumps(sc.get(k))}")


# ----------------------------------------------------------------------------- main


def main(stage: str, device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    filed = json.loads(PRED.read_text())["config"]
    assert filed == config(), f"predictions were filed for another config: {filed}"
    if stage == "score":
        run_score()
        return
    job = Job(stage, device)
    if stage == "regen":
        regen(job)
    elif stage == "a":
        stage_a(job)
    else:
        row0, seed = map(int, stage[1:].split("s"))
        stage_b(job, row0, seed)
    job.finish()


if __name__ == "__main__":
    from predictions import file_predictions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=STAGES)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions.json and exit")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.predictions:
        file_predictions(PRED, config(), READINGS)
    elif a.stage is None:
        ap.error("give a stage")
    else:
        main(a.stage, a.device)
