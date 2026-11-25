"""E210, cell P2.37: does our crosscoder pipeline (src/commute/crosscoder.py) reproduce the authors' own per-latent numbers on
Minder et al. 2025's released Gemma-2-2B BatchTopK crosscoder (layer 13, k 100)? Plan:
reports/plans/2026-11-18_plan_p2-37-crosscoder-replicate.md. Rules and expected values: results/predictions.json.

  python experiments/crosscoder_replicate/run.py a   # CPU: our relative norm against feature_df's dec_norm_diff
  python experiments/crosscoder_replicate/run.py b   # GPU: 2M tokens of layer-13 activations, our betas on the top
                                                     # 3,176 latents, Spearman and pass counts against feature_df

Model order: the released file's lm_name lists the chat model first; stage a reads the order off the decoder norms
(feature_df's dec_base_norm, dec_instruct_norm) and flips the file so index 0 is the base, our convention. The
released file has no activation normaliser, so `normalize_activations` is the identity. The paper's filter (paper.txt
2272: latents with a negative base beta dropped) is applied here to both sides; our port has no such filter.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dictionary_learning.dictionary import BatchTopKCrossCoder
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.stats import spearmanr
from torch import Tensor

from commute import crosscoder as CC
from commute.jobs import config_from_settings, hard_exit, require_filed

RESULTS = Path(__file__).resolve().parent / "results"
PRED = RESULTS / "predictions.json"
# ours -> feature_df column (closed_form targets: error = eps, reconstruction = r; chat = our "ft")
COLS = {"eps_base": "beta_error_base", "eps_ft": "beta_error_chat", "r_base": "beta_reconstruction_base",
        "r_ft": "beta_reconstruction_chat", "nu_eps": "beta_ratio_error", "nu_r": "beta_ratio_reconstruction"}  # fmt: skip
FLIP = ("decoder.weight", "decoder.bias", "encoder.weight")  # the tensors with a model axis first
NO_NORMALISER = {"target_rms", "activation_mean", "activation_std", "activation_global_scale", "code_normalization_id"}


@dataclass(frozen=True)
class Config:
    crosscoder: str  # HF model repo of the released crosscoder
    stats: str  # HF dataset repo holding feature_df.csv
    base: str
    chat: str
    layer: int  # activations are the output of model.layers[layer] (paper footnote 14)
    ctx: int  # tokens per document at most, BOS included
    fineweb: str
    lmsys: str  # already in the Gemma chat template (its "text" column)
    tokens_per_set: int
    top_n: int
    nu_r: float
    nu_eps: float
    pis: list[float]  # Table 1 thresholds, reported beside the filed rule
    chunk: int  # tokens per encode

    def __post_init__(self) -> None:
        assert self.tokens_per_set >= self.ctx > 1 and 0 < self.top_n < 73728 and 0 < self.nu_eps < self.nu_r, self


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def write(name: str, out: dict[str, Any]) -> None:
    (RESULTS / name).write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote results/{name}")


def stop(name: str, out: dict[str, Any], msg: str) -> None:
    write(name, out | {"stop": msg})
    log(f"STOP: {msg}")
    hard_exit(1)


def feature_df(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(hf_hub_download(cfg.stats, "feature_df.csv", repo_type="dataset"))
    assert (df["latent"].to_numpy() == np.arange(len(df))).all(), "feature_df rows are not latents 0..D-1 in order"
    return df


def orient(sd: dict[str, Tensor], base_norm: Tensor, chat_norm: Tensor) -> tuple[dict[str, Tensor], str, float]:
    """The state dict with model 0 the base, the order the file had, and the worse decoder-norm match of that order."""
    n = sd["decoder.weight"].norm(dim=2).double()
    miss = {"base_first": max(float((n[0] - base_norm).abs().max()), float((n[1] - chat_norm).abs().max())),
            "chat_first": max(float((n[1] - base_norm).abs().max()), float((n[0] - chat_norm).abs().max()))}  # fmt: skip
    order = min(miss, key=lambda k: miss[k])
    if order == "chat_first":
        sd = sd | {k: sd[k].flip(0) for k in FLIP}
    return sd, order, miss[order]


def load_ae(cfg: Config, df: pd.DataFrame) -> tuple[CC.AE, str, float]:
    sd = load_file(hf_hub_download(cfg.crosscoder, "model.safetensors"))
    sd, order, miss = orient(sd, torch.tensor(df["dec_base_norm"].to_numpy()), torch.tensor(df["dec_instruct_norm"].to_numpy()))
    _, d, D = sd["encoder.weight"].shape
    ae: CC.AE = BatchTopKCrossCoder(d, D, 2, k=int(sd["k"]), code_normalization="crosscoder")
    got = ae.load_state_dict(sd, strict=False)
    assert not got.unexpected_keys and set(got.missing_keys) <= NO_NORMALISER, got
    assert not ae.has_activation_normalizer and float(ae.threshold) > 0, "expected no normaliser and a set threshold"
    return ae.eval(), order, miss


def their_top(df: pd.DataFrame, n: int) -> Tensor:
    """The toolkit's candidates (closed_form.py:365-372): dec_norm_diff ascending, the head."""
    return torch.tensor(df.sort_values("dec_norm_diff", kind="stable").head(n)["latent"].to_numpy())


def stage_a(cfg: Config, th: dict[str, float]) -> None:
    df = feature_df(cfg)
    ae, order, miss = load_ae(cfg, df)
    rn = CC.relative_norm(ae).double()
    diff = ((1 - rn) - torch.tensor(df["dec_norm_diff"].to_numpy())).abs()
    ours = torch.nonzero(CC.top(rn, cfg.top_n)).flatten()
    theirs = their_top(df, cfg.top_n)
    t = df.iloc[theirs.numpy()]
    out = {"order": order, "norm_max_abs": miss, "rn_max_abs": float(diff.max()), "rn_mean_abs": float(diff.mean()),
           "latents": len(df), "threshold": float(ae.threshold), "k": int(ae.k),
           "top_overlap": len(set(ours.tolist()) & set(theirs.tolist())),
           "their_reading": counts(t, cfg)}  # fmt: skip
    if miss > th["norm_max_abs"] or out["rn_max_abs"] > th["rn_max_abs"]:
        stop("stage_a.json", out, f"decoder norms off by {miss:.3g} or relative norm by {out['rn_max_abs']:.3g}")
    write("stage_a.json", out | {"passed": True})


def counts(t: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """feature_df's own reading of latents `t`: the paper's filter, the filed pass count and Table 1's counts."""
    ok = (t[COLS["eps_base"]] >= 0) & (t[COLS["r_base"]] >= 0)
    r, e = t[COLS["nu_r"]], t[COLS["nu_eps"]]
    return {"n": len(t), "kept": int(ok.sum()), "count": int((ok & (r < cfg.nu_r) & (e < cfg.nu_eps)).sum()),
            "count_unfiltered": int(((r < cfg.nu_r) & (e < cfg.nu_eps)).sum()),
            "pi_counts": {str(p): int((ok & (r < p) & (e < p)).sum()) for p in cfg.pis}}  # fmt: skip


def token_docs(cfg: Config, tok: Any) -> tuple[list[list[int]], dict[str, int]]:
    """The first rows of each train split, each cut to `ctx` tokens, the last cut so each set holds tokens_per_set."""
    from datasets import load_dataset

    docs, n_docs = [], {}
    for name in (cfg.fineweb, cfg.lmsys):
        left, start = cfg.tokens_per_set, len(docs)
        for row in load_dataset(name, split="train", streaming=True):
            ids = tok(row["text"], truncation=True, max_length=cfg.ctx)["input_ids"][:left]
            assert ids[0] == tok.bos_token_id, f"{name}: no BOS first"
            docs.append(ids)
            left -= len(ids)
            if left == 0:
                break
        assert left == 0, f"{name}: ran out of rows"
        n_docs[name] = len(docs) - start
    sot = tok.convert_tokens_to_ids("<start_of_turn>")
    assert sot in docs[n_docs[cfg.fineweb]], "the chat template's <start_of_turn> is not one token in the first LMSYS row"
    return docs, n_docs


class Stop(Exception):
    """Raised by the layer hook: nothing after the read layer runs."""


@torch.no_grad()
def activations(cfg: Config, docs: list[list[int]]) -> tuple[Tensor, int, dict[str, float]]:
    """(n, 2, d) bf16 on the GPU, model 0 the base; rows [:n_body] are non-BOS tokens, the BOS rows come last, so a
    slice is the set without BOS and no copy is made."""
    from transformers import AutoModelForCausalLM

    n, n_body = sum(map(len, docs)), sum(map(len, docs)) - len(docs)
    h, norms, grab = None, {}, {}

    def hook(_mod: Any, _inp: Any, out: Any) -> None:
        grab["h"] = out[0] if isinstance(out, tuple) else out
        raise Stop

    for m, name in enumerate((cfg.base, cfg.chat)):
        t0 = time.time()
        model: Any = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="eager")
        model = model.cuda().eval()
        if h is None:
            h = torch.empty(n, 2, model.config.hidden_size, dtype=torch.bfloat16, device="cuda")
        handle = model.model.layers[cfg.layer].register_forward_hook(hook)
        log(f"{name} loaded in {time.time() - t0:.0f}s")
        i, j = 0, n_body
        for k, ids in enumerate(docs):
            try:
                model.model(torch.tensor([ids], device="cuda"), use_cache=False)
            except Stop:
                pass
            x = grab["h"][0]
            h[j, m], h[i : i + len(ids) - 1, m] = x[0], x[1:]
            i, j = i + len(ids) - 1, j + 1
            if k == 0:  # early check: finite, and the body's norm for the log
                assert bool(torch.isfinite(x).all()), f"{name}: non-finite activations on the first document"
                log(f"{name} first doc: {len(ids)} tokens, BOS norm {float(x[0].float().norm()):.1f}, body mean norm "
                    f"{float(x[1:].float().norm(dim=1).mean()):.1f}")  # fmt: skip
        assert i == n_body and j == n, (i, j)
        norms[name] = sum(float(c.float().norm(dim=1).sum()) for c in CC.split(h[:n_body, m], cfg.chunk)) / n_body
        handle.remove()
        del model
        torch.cuda.empty_cache()
        log(f"{name}: {n} tokens in {time.time() - t0:.0f}s, body mean norm {norms[name]:.1f}")
    assert h is not None
    return h, n_body, norms


def rank_corr(x: np.ndarray, y: np.ndarray, keep: np.ndarray) -> float:
    """Spearman on the kept latents; nan with fewer than three (an empty pool never reads as reproduced)."""
    if keep.sum() < 3:
        return float("nan")
    res: Any = spearmanr(x[keep], y[keep])
    return float(res.statistic)


def reading(b: dict[str, Tensor], t: pd.DataFrame, cfg: Config, th: dict[str, float]) -> dict[str, Any]:
    """Ours (`b`, aligned with rows `t`) against feature_df: Spearman of each quantity on the latents both sides keep
    (finite, base betas not negative), the pass counts, Table 1's counts, and the do-nothing reference (their nu
    against their relative norm alone). A nan ratio (a latent that never fired) is dropped by the filter: it fails."""
    ours = {k: b[k].double().numpy() for k in COLS}
    theirs = {k: t[c].to_numpy(dtype=np.float64) for k, c in COLS.items()}
    fin = np.isfinite(ours["nu_r"]) & np.isfinite(ours["nu_eps"])
    ok_o = fin & (ours["eps_base"] >= 0) & (ours["r_base"] >= 0)
    ok_t = (theirs["eps_base"] >= 0) & (theirs["r_base"] >= 0)
    both = ok_o & ok_t

    def n_pass(keep: np.ndarray, p: float, q: float) -> int:
        return int((keep & (ours["nu_r"] < p) & (ours["nu_eps"] < q)).sum())

    co, ct = n_pass(ok_o, cfg.nu_r, cfg.nu_eps), counts(t, cfg)
    rho = {k: rank_corr(ours[k], theirs[k], both) for k in COLS}
    ok = min(rho["nu_r"], rho["nu_eps"]) >= th["spearman_min"] and abs(co - ct["count"]) <= th["count_rel"] * ct["count"]
    return {"n": len(t), "nan": int((~fin).sum()), "kept_ours": int(ok_o.sum()), "kept_theirs": int(ok_t.sum()),
            "kept_both": int(both.sum()), "spearman": rho,
            "spearman_unfiltered": {k: rank_corr(ours[k], theirs[k], fin) for k in ("nu_r", "nu_eps")},
            "count_ours": co, "count_theirs": ct["count"], "count_ours_unfiltered": n_pass(fin, cfg.nu_r, cfg.nu_eps),
            "count_theirs_unfiltered": ct["count_unfiltered"],
            "pi_counts_ours": {str(p): n_pass(ok_o, p, p) for p in cfg.pis}, "pi_counts_theirs": ct["pi_counts"],
            "do_nothing_spearman": {k: rank_corr(t["dec_norm_diff"].to_numpy(), theirs[k], both) for k in ("nu_r", "nu_eps")},
            "reproduced": bool(ok)}  # fmt: skip


def stage_b(cfg: Config, th: dict[str, float]) -> None:
    if not os.environ.get("HF_TOKEN"):
        sys.exit("STOP: stage b needs HF_TOKEN set in the submitting shell: a Hugging Face token whose account has "
                 "accepted the Gemma-2 licence (google/gemma-2-2b, google/gemma-2-2b-it)")  # fmt: skip
    a = json.loads((RESULTS / "stage_a.json").read_text())
    assert a.get("passed"), "stage a has not passed"
    from transformers import AutoTokenizer

    t0 = time.time()
    df = feature_df(cfg)
    ae, order, _ = load_ae(cfg, df)
    assert order == a["order"], (order, a["order"])
    ae = ae.cuda()
    docs, n_docs = token_docs(cfg, AutoTokenizer.from_pretrained(cfg.base))
    log(f"tokenised {n_docs} in {time.time() - t0:.0f}s")
    h, n_body, norms = activations(cfg, docs)
    out: dict[str, Any] = {"tokens": h.shape[0], "body_tokens": n_body, "docs": n_docs, "body_mean_norm": norms}
    for part, x in (("all", h), ("body", h[:n_body])):
        f = CC.fve(ae, (c.float() for c in CC.split(x, cfg.chunk)), size=cfg.chunk)
        out[f"fve_{part}"] = {"base": f["fve_base"], "chat": f["fve_ft"], "l0": f["l0"]}
        log(f"FVE {part}: {out[f'fve_{part}']}")
    if min(out["fve_body"]["base"], out["fve_body"]["chat"]) < th["fve_min"]:
        stop("stage_b.json", out, f"crosscoder FVE on our activations {out['fve_body']} under {th['fve_min']}")
    rn = CC.relative_norm(ae)
    J = torch.nonzero(CC.top(rn, cfg.top_n)).flatten()
    t = df.iloc[J.numpy()]
    out["top_overlap"] = len(set(J.tolist()) & set(their_top(df, cfg.top_n).tolist()))
    table = pd.DataFrame({"latent": J.numpy(), "relative_norm": rn[J].numpy(), "dec_norm_diff": t["dec_norm_diff"].to_numpy()})
    for part, x in (("all", h), ("body", h[:n_body])):
        b = CC.betas(ae, x, J, size=cfg.chunk)
        out[f"reading_{part}"] = reading(b, t, cfg, th)
        log(f"reading {part}: {json.dumps(out[f'reading_{part}'], default=float)}")
        table = table.assign(**{f"{k}_{part}": b[k].numpy() for k in COLS})
    table = table.assign(**{f"{c}_theirs": t[c].to_numpy() for c in COLS.values()})
    table.to_csv(RESULTS / "stage_b_latents.csv", index=False, float_format="%.6g")
    out["reproduced"] = out["reading_all"]["reproduced"]
    out["minutes"], out["peak_gpu_gb"] = (time.time() - t0) / 60, torch.cuda.max_memory_allocated() / 2**30
    write("stage_b.json", out)


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) == 2 else ""
    assert stage in ("a", "b"), "usage: run.py a|b"
    require_filed(PRED)
    p = json.loads(PRED.read_text())
    cfg = config_from_settings(Config, p["settings"])
    {"a": stage_a, "b": stage_b}[stage](cfg, p["thresholds"])
    hard_exit(0)


if __name__ == "__main__":
    main()
