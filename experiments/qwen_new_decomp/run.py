"""E131: Goodfire's second Qwen3-1.7B decomposition (p-9707d6ff, "new") against the one every reading so far used
(p-e9fcffd5, "old"): the port's controls, then three numbers per decomposition and row set. Plan:
reports/plans/2026-11-07_plan_e131-new-decomposition.md (its Design check is the spec); rules in
results/predictions.json (the run refuses to start without it) and score.py.

Numbers, each per row (mean over the row's positions) and then over rows, rows equal-weighted:
  ci_kl   KL(target || masked) with m = g and Δ off (the CI-masked KL)
  active  subcomponents with g > 0.1 per position, summed over the 196 sites
  adv_kl  KL(target || masked) at the final sources of the config's evaluation adversary (PGDReconLoss: one U(0,1)
          source per site shared by all rows and positions plus one Δ scalar, 20 sign steps of 0.1 up the
          token-weighted mean KL, one random start, seed 0; `qwen_vpd.pgd_kl_rows`), read per row afterwards
Row sets (all <= 512 tokens): T = the 128 FineWeb rows of experiments/qwen_adversary/results/tokens.json; G and
G-think = 128 GSM8K test prompts (the first 128, chat template, the baseline's INSTRUCTION) followed by Qwen3-1.7B's
own greedy response with thinking off / on, cut at 512 tokens in all (stage 1 writes them to results/rows.json).

Stages (each writes its results as it finishes and prints `STAGE n DONE`):
  0  controls, stopping the job on any miss: (a) V U reproduces each HF matrix of the new decomposition (relative
     Frobenius error <= 0.0055); (b) torch Γ of the new decomposition on the JAX library's own taps vs the JAX g
     (jax_reference.py) on the 8 qwen_port rows: share of (position, subcomponent) pairs with |diff| > 0.1 at most
     1e-4; (c) the old decomposition through this code reproduces its stored CI-masked KL on T (0.14167) within 2%.
  1  the G and G-think rows (bf16 greedy generation), reused once written.
  2-7  one (decomposition, row set) each: 2 new G, 3 new G-think, 4 old G, 5 old G-think, 6 new T, 7 old T. Per
     stage: g on every row (Γ in bf16, kept on the host as its non-zero entries), active counts; then Γ freed, U V in
     fp32: ci_kl; then the adversary and adv_kl. -> results/measure_<decomp>_<rows>.json. A stage whose file already
     holds adv_kl is skipped, so a killed job is resubmitted with the same arguments.
  8  score (CPU): the new/old ratios with row-bootstrap intervals and P1-P4 (score.py) -> results/score.json.
Precision: target, U, V and the masked forward fp32 (TF32 off), Γ bf16, as in experiments/qwen_adversary.

Usage (repo root): sbatch cluster/qwen_new_decomp.sbatch controls; then one job per stage, `... measure 2` to
`... measure 7` (several stages of one decomposition may share a job: `measure 2 3`, if their time fits 2 h);
then ~/commute-agents/fleet/cpu-run --mem 6G python experiments/qwen_new_decomp/run.py score
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import qwen_vpd as Q
from commute.jobs import hard_exit, load_experiment
from commute.vpd import kl_per_position

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, CONTROLS, ROWS, SCORE = (RESULTS / f for f in ("predictions.json", "controls.json", "rows.json", "score.json"))
T_TOKENS = HERE.parent / "qwen_adversary/results/tokens.json"
PORT_TOKENS = HERE.parent / "qwen_port/results/tokens.json"
JAX_SUMMARY = RESULTS / "jax_reference.json"
ARTIFACTS = Path("/mnt/nw/home/e.agnew/commute-data/artifacts")
DECOMPS = {"old": ARTIFACTS / "qwen3-1.7b-p-e9fcffd5-torch", "new": ARTIFACTS / "qwen3-1.7b-p-9707d6ff-torch"}
RECON_CSV = ARTIFACTS / "p-9707d6ff-share/data/p-9707d6ff-share/reconstruction_check.csv"
GSM8K, GSM8K_REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"  # experiments/qwen_gsm8k_baseline
STAGES = {2: ("new", "G"), 3: ("new", "G-think"), 4: ("old", "G"), 5: ("old", "G-think"), 6: ("new", "T"), 7: ("old", "T")}
DEV = "cuda"
T0 = time.time()


@dataclass(frozen=True)
class Config:
    n_rows: int = 128  # every row set
    window: int = 512  # tokens per row at most (the old decomposition's training length)
    seed: int = 0  # the adversary's one random start
    active: float = 0.1  # the "active" threshold on g
    recon_max: float = 0.0055  # control (a): the bundle's reconstruction_check.csv bound
    g_tol: float = 0.1  # control (b): a pair counts when |g_torch - g_jax| > g_tol
    g_share: float = 1e-4  # control (b): at most this share of pairs
    old_ci_kl: float = 0.141670617449563  # control (c): experiments/qwen_adversary/results/adversary.json ci128
    old_ci_rel: float = 0.02
    gen_batch: int = 64  # stage 1 prompts per generate call

    def __post_init__(self) -> None:
        assert self.n_rows % self.gen_batch == 0 and self.window == 512, self


CFG = Config()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def write(path: Path, out: dict) -> None:
    path.write_text(json.dumps(out, indent=1))


def rows_sha(rows: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def free() -> None:
    torch.cuda.empty_cache()
    log(f"GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB")


# ----------------------------------------------------------------------------- g on the host


@dataclass
class SparseG:
    """One row's g (site -> [1, t, C]) on the host as its non-zero entries: flat int32 index and fp32 value. g is
    zero at most (position, subcomponent) pairs (~7-11k of 688,128 non-zero per position), so 128 rows fit in RAM."""

    t: int
    idx: dict[str, Tensor]
    val: dict[str, Tensor]

    @staticmethod
    def of(g: dict[str, Tensor]) -> SparseG:
        idx, val = {}, {}
        for site, v in g.items():
            flat = v.reshape(-1)
            nz = flat.nonzero().squeeze(1)
            idx[site], val[site] = nz.to(torch.int32).cpu(), flat[nz].float().cpu()
        return SparseG(g[Q.SITES[0]].shape[1], idx, val)

    def dense(self) -> dict[str, Tensor]:
        out = {}
        for site in Q.SITES:
            c = Q.C[Q.parse_site(site)[1]]
            flat = torch.zeros(self.t * c, device=DEV)
            flat[self.idx[site].to(DEV).long()] = self.val[site].to(DEV)
            out[site] = flat.view(1, self.t, c)
        return out

    def active(self, threshold: float) -> float:
        return sum(int((v > threshold).sum()) for v in self.val.values()) / self.t


def g_rows(model: Any, rows: list[Tensor], gamma: Q.Gamma, comps: Q.Components) -> list[SparseG]:
    out = []
    for i, tokens in enumerate(rows):
        _, g = Q.importances(model, tokens, gamma, comps)
        out.append(SparseG.of(g))
        del g
        if i % 32 == 31:
            log(f"g: {i + 1}/{len(rows)} rows")
    return out


@torch.no_grad()
def clean_logits(model: Any, tokens: Tensor) -> Tensor:
    return model(tokens, use_cache=False).logits


@torch.no_grad()
def ci_kls(model: Any, rows: list[Tensor], comps: Q.Components, gs: list[SparseG]) -> list[float]:
    """Per row: mean over positions of KL(target || masked at m = g, Δ off)."""
    return [float(kl_per_position(clean_logits(model, t), Q.masked_logits(model, t, comps, g.dense())).mean())
            for t, g in zip(rows, gs, strict=True)]


@torch.no_grad()
def kls_at(model: Any, rows: list[Tensor], comps: Q.Components, gs: list[SparseG], final: dict[str, Tensor]) -> list[float]:
    """Per row: mean over positions of KL(target || masked) at the adversary's final sources."""
    src = {s: final[s].to(DEV) for s in Q.SITES}
    dsrc = {s: final[f"{s}.delta"].to(DEV) for s in Q.SITES}
    out = []
    for tokens, sparse in zip(rows, gs, strict=True):
        g = sparse.dense()
        masks = {s: g[s] + (1 - g[s]) * src[s] for s in Q.SITES}
        deltas = {s: dsrc[s].expand(tokens.shape) for s in Q.SITES}
        logits = Q.masked_logits(model, tokens, comps, masks, deltas)
        out.append(float(kl_per_position(clean_logits(model, tokens), logits).mean()))
    return out


def adversary(model: Any, rows: list[Tensor], comps: Q.Components, gs: list[SparseG]) -> dict:
    steps: list[str] = []

    def step_log(msg: str) -> None:
        steps.append(msg)
        log(msg)

    final: dict[str, Tensor] = {}
    pooled = Q.pgd_kl_rows(model, rows, comps, lambda i: (gs[i].dense(), clean_logits(model, rows[i])), CFG.seed,
                           log=step_log, final=final)
    return {"adv_kl": kls_at(model, rows, comps, gs, final), "adv_token_weighted": pooled, "adv_steps": steps}


# ----------------------------------------------------------------------------- rows


def token_rows(name: str) -> list[list[int]]:
    rows = json.loads(T_TOKENS.read_text())["rows"] if name == "T" else json.loads(ROWS.read_text())[name]["rows"]
    assert len(rows) == CFG.n_rows and all(0 < len(r) <= CFG.window for r in rows), name
    return rows


def generated_rows() -> dict:
    """Stage 1: Qwen3-1.7B's greedy responses (bf16) to the first n_rows GSM8K test prompts, thinking off and on."""
    from datasets import load_dataset

    from commute.qwen_eval import GREEDY, INSTRUCTION, MODEL_REV, load_model, sample

    assert MODEL_REV == Q.HF_REVISION
    tok, model = load_model()
    ds = load_dataset(GSM8K, "main", split="test", revision=GSM8K_REV)
    questions = [str(ds[i]["question"]) for i in range(CFG.n_rows)]  # pyright: ignore
    out: dict = {"source": f"{GSM8K}@{GSM8K_REV} test[:{CFG.n_rows}]; Qwen/Qwen3-1.7B@{MODEL_REV} bf16 greedy; "
                           f"chat template; cut at {CFG.window} tokens"}
    for name, thinking in (("G", False), ("G-think", True)):
        msgs = [[{"role": "user", "content": f"{q}\n\n{INSTRUCTION}"}] for q in questions]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
                 for m in msgs]
        prompts = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        max_new = CFG.window - min(len(p) for p in prompts)
        responses = []
        for s in range(0, CFG.n_rows, CFG.gen_batch):
            responses += sample(tok, model, texts[s : s + CFG.gen_batch], 1, GREEDY, max_new=max_new)
            log(f"{name}: {len(responses)}/{CFG.n_rows} generated")
        rows = [(p + r)[: CFG.window] for p, (r, _) in zip(prompts, responses, strict=True)]
        out[name] = {"thinking": thinking, "rows": rows, "prompt_tokens": [len(p) for p in prompts],
                     "cut_at_window": [len(p) + len(r) > CFG.window for p, (r, _) in zip(prompts, responses, strict=True)],
                     "text": [tok.decode(r) for r in rows]}
        lens = [len(r) for r in rows]
        log(f"{name}: lengths min {min(lens)} mean {sum(lens) / len(lens):.0f} max {max(lens)}; "
            f"cut {sum(out[name]['cut_at_window'])}/{CFG.n_rows}")
    del model
    free()
    return out


# ----------------------------------------------------------------------------- stage 0


def recon_check(model: Any, comps: Q.Components) -> dict:
    """(a) ||V U - W^T|| / ||W|| per site, fp32 on the GPU, against the bundle's reconstruction_check.csv."""
    csv = {}
    for line in RECON_CSV.read_text().splitlines()[1:]:
        name, *_, err = line.split(",")
        csv[name.removeprefix("model.")] = float(err)
    errs = {}
    with torch.no_grad():
        for site in Q.SITES:
            V, U = (x.to(DEV) for x in comps.vu(site))
            W = Q._linear(model, site).weight
            errs[site] = float((V @ U - W.T).norm() / W.norm())
    worst = max(errs, key=errs.__getitem__)
    gap = max(abs(errs[s] - csv[s]) for s in Q.SITES)
    return {"max_rel_frobenius": errs[worst], "worst_site": worst, "max_abs_gap_to_bundle_csv": gap,
            "bound": CFG.recon_max, "pass": errs[worst] <= CFG.recon_max}


def jax_check(model: Any, comps_cpu: Q.Components) -> dict:
    """(b) torch Γ (bf16) on JAX's taps vs JAX's g on the 8 port rows (the rule), plus end to end, active counts and
    the CI-masked KL beside JAX's (reported)."""
    from safetensors.torch import load_file

    GDiff = load_experiment("qwen_port", "torch_check").GDiff  # pairs with |diff| > 0.1, overall and per kind
    jax = json.loads(JAX_SUMMARY.read_text())
    rows = [torch.tensor([r], device=DEV) for r in json.loads(PORT_TOKENS.read_text())["rows"]]
    comps = comps_cpu.to(DEV, torch.bfloat16)
    gamma = Q.load_gamma(DECOMPS["new"], DEV, torch.bfloat16)
    on_taps, end_to_end, gs = GDiff(), GDiff(), []
    for r, tokens in enumerate(rows):
        saved = load_file(LARGE / f"jax_gamma_row{r}.safetensors")
        ref_g = {s: saved[f"g.{s}"] for s in Q.SITES}
        taps = load_file(LARGE / f"jax_taps_row{r}.safetensors")
        clean = Q.clean_forward(model, tokens, gamma, comps)
        with torch.no_grad():
            g1 = Q.lower_importances(gamma, [taps[f"resid.{b}"][None].to(DEV) for b in range(Q.N_TAPS)], clean.readouts)
            on_taps.add(g1, ref_g)
            del g1
            g = Q.lower_importances(gamma, clean.taps, clean.readouts)
        end_to_end.add(g, ref_g)
        gs.append(SparseG.of(g))
        del clean, g, ref_g, saved
    del gamma, comps
    free()
    comps32 = comps_cpu.to(DEV, torch.float32)
    kl = ci_kls(model, rows, comps32, gs)
    del comps32
    free()
    res = {"on_jax_taps": on_taps.result(), "end_to_end": end_to_end.result(),
           "active_torch": [g.active(CFG.active) for g in gs], "active_jax": jax["gamma"]["active_0.1"],
           "ci_kl_torch_fp32": kl, "ci_kl_jax_bf16": jax["masked"]["ci_kl"], "share_max": CFG.g_share}
    for k in ("on_jax_taps", "end_to_end"):
        res[k].pop("pass")  # torch_check's own 1e-3 rule; this control's rule is share_max
    res["pass"] = res["on_jax_taps"]["share_over_0.1"] <= CFG.g_share
    return res


def old_check(model: Any) -> dict:
    """(c) the old decomposition's CI-masked KL on T through this code, against its stored value."""
    rows = [torch.tensor([r], device=DEV) for r in token_rows("T")]
    comps_cpu = Q.load_components(DECOMPS["old"], "cpu", torch.float32)
    comps = comps_cpu.to(DEV, torch.bfloat16)
    gamma = Q.load_gamma(DECOMPS["old"], DEV, torch.bfloat16)
    gs = g_rows(model, rows, gamma, comps)
    del gamma, comps
    free()
    comps = comps_cpu.to(DEV, torch.float32)
    del comps_cpu
    kl = ci_kls(model, rows, comps, gs)
    mean = sum(kl) / len(kl)
    return {"ci_kl_mean": mean, "stored": CFG.old_ci_kl, "rel_diff": mean / CFG.old_ci_kl - 1, "rel_max": CFG.old_ci_rel,
            "ci_kl": kl, "active": [g.active(CFG.active) for g in gs], "pass": abs(mean / CFG.old_ci_kl - 1) <= CFG.old_ci_rel}


def controls() -> int:
    model = Q.load_target(DEV, torch.float32)
    comps_cpu = Q.load_components(DECOMPS["new"], "cpu", torch.float32)
    out: dict = {"config": CFG.__dict__}
    out["a_recon"] = recon_check(model, comps_cpu)
    log(f"control (a): {out['a_recon']}")
    write(CONTROLS, out)
    out["b_jax"] = jax_check(model, comps_cpu)
    log(f"control (b): { {k: v for k, v in out['b_jax'].items() if k != 'end_to_end'} }")
    write(CONTROLS, out)
    del comps_cpu
    out["c_old"] = old_check(model)
    log(f"control (c): { {k: v for k, v in out['c_old'].items() if k not in ('ci_kl', 'active')} }")
    out["pass"] = all(out[k]["pass"] for k in ("a_recon", "b_jax", "c_old"))
    write(CONTROLS, out)
    if not out["pass"]:
        log("STAGE 0 FAILED: a control missed; stopping before stage 1")
        return 1
    log("STAGE 0 DONE")
    del model
    free()
    if not ROWS.is_file():
        write(ROWS, generated_rows())
    log("STAGE 1 DONE")
    return 0


# ----------------------------------------------------------------------------- stages 2-7


def measure_path(stage: int) -> Path:
    decomp, rows = STAGES[stage]
    return RESULTS / f"measure_{decomp}_{rows}.json"


def measure(stages: list[int]) -> int:
    assert json.loads(CONTROLS.read_text())["pass"] and ROWS.is_file(), "stage 0 must pass and stage 1 finish first"
    (decomp,) = {STAGES[s][0] for s in stages}
    todo = [s for s in stages if not (measure_path(s).is_file() and "adv_kl" in json.loads(measure_path(s).read_text()))]
    log(f"{decomp}: stages {todo} to run ({sorted(set(stages) - set(todo))} already done)")
    if not todo:
        return 0
    model = Q.load_target(DEV, torch.float32)
    comps_cpu = Q.load_components(DECOMPS[decomp], "cpu", torch.float32)
    comps = comps_cpu.to(DEV, torch.bfloat16)
    gamma = Q.load_gamma(DECOMPS[decomp], DEV, torch.bfloat16)
    log(f"loaded {decomp}; GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    rows, gs, outs = {}, {}, {}
    for s in todo:
        name = STAGES[s][1]
        ids = token_rows(name)
        rows[s] = [torch.tensor([r], device=DEV) for r in ids]
        t = time.time()
        gs[s] = g_rows(model, rows[s], gamma, comps)
        outs[s] = {"decomp": decomp, "dir": str(DECOMPS[decomp]), "rows": name, "rows_sha": rows_sha(ids),
                   "lengths": [len(r) for r in ids], "seed": CFG.seed, "active": [g.active(CFG.active) for g in gs[s]],
                   "seconds": {"g": round(time.time() - t, 1)}}
        write(measure_path(s), outs[s])
        log(f"stage {s} ({decomp} {name}): active mean {sum(outs[s]['active']) / CFG.n_rows:.0f}")
    del gamma, comps
    free()
    comps = comps_cpu.to(DEV, torch.float32)
    del comps_cpu
    for s in todo:
        out = outs[s]
        t = time.time()
        out["ci_kl"] = ci_kls(model, rows[s], comps, gs[s])
        out["seconds"]["ci_kl"] = round(time.time() - t, 1)
        write(measure_path(s), out)
        log(f"stage {s}: ci_kl mean {sum(out['ci_kl']) / CFG.n_rows:.4f}")
        t = time.time()
        out |= adversary(model, rows[s], comps, gs[s])
        out["seconds"]["adversary"] = round(time.time() - t, 1)
        write(measure_path(s), out)
        log(f"stage {s}: adv_kl mean {sum(out['adv_kl']) / CFG.n_rows:.4f} (token-weighted {out['adv_token_weighted']:.4f})")
        log(f"STAGE {s} DONE ({out['decomp']} {out['rows']})")
        del gs[s], rows[s]
    return 0


# ----------------------------------------------------------------------------- stage 8


def score_stage() -> int:
    score = load_experiment("qwen_new_decomp", "score")
    measured, shas = {}, {}
    for s, (decomp, name) in STAGES.items():
        if measure_path(s).is_file():
            m = json.loads(measure_path(s).read_text())
            measured[(decomp, name)] = {k: m[k] for k in score.NUMBERS if k in m}
            shas.setdefault(name, set()).add(m["rows_sha"])
    assert all(len(v) == 1 for v in shas.values()), f"a row set differs between decompositions: {shas}"
    out = score.score(measured) | {"measured": sorted(f"{d}/{n}" for d, n in measured)}
    write(SCORE, out)
    for name, p in out["predictions"].items():
        log(f"{name} ({p['rows']} {p['number']}): ratio {p['ratio']} [{p['lo95']}, {p['hi95']}] -> {p['verdict']}")
    log("STAGE 8 DONE")
    return 0


def main() -> None:
    assert PRED.is_file(), f"file predictions first: {PRED}"
    what, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    if what == "score":
        hard_exit(score_stage())
    Q.fp32_matmuls()
    assert torch.cuda.get_device_properties(0).total_memory > 44 * 2**30, "needs a 46 GB card (L40) or larger"
    if what == "controls":
        hard_exit(controls())
    assert what == "measure" and args and all(a in STAGES for a in args), sys.argv
    hard_exit(measure(args))


if __name__ == "__main__":
    main()
