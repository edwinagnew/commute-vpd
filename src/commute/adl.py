"""The activation difference lens (ADL) of Minder et al. 2025, "Narrow finetuning leaves clearly readable traces in
activation differences" (arXiv 2510.13900), ported to the 67M's `steering.Net` (fp32, the target model only).

The authors' code is github.com/science-of-finetuning/diffing-toolkit, pinned here at commit
617d9266eada0baf5fe5347ce97e64b6dd0f82bf (2026-10-25), `src/diffing/methods/activation_difference_lens/`. It drives
Hugging Face models through nnsight/nnterp (`StandardizedTransformer`), which a VPD ComponentModel's target is not, so
the few functions that compute ADL are ported here; each cites its source. Departures are listed in the P2.59 plan
(reports/plans/2026-11-22_plan_p2-59-adl.md).

  SITE           the middle layer: the toolkit takes layer int(0.5 (L - 1)) of `layers_output` (utils/activations.py
                 `get_layer_indices`, method config `layers: [0.5]`), the output of block 1 of 4 here, which is block 2's
                 input `in2` (paper Sec. 3: l = floor(L / 2) counting from 1, the same residual).
  mean_diff      paper Sec. 3: delta_bar_j = mean over pretraining texts of h_ft - h_base at SITE, position j < k
                 (method.py `compute_differences`: diff = ft_acts - base_acts; mean_diff = diff.mean(dim=0)).
  logit_lens     paper App. C.1: softmax(W_U final_layer_norm(delta_bar_j)) (utils/model.py `logit_lens`).
  removed        paper Sec. 5, eqs. (1)-(2): in the fine-tuned model the component of h_ft along delta_bar_j is set to
                 the base model's at every position (causal_effect.py `_compute_nll_intervened`, patch_direction
                 base_to_ft, zero_ablate false); here read as the drop of a per-row readout, not of cross-entropy.
  column_D       the baseline column read beside the share (E265, E272): `removed` per delta_bar_j over a gain, the max
  random_diffs   the paper's baseline directions (Sec. 5; causal_effect.py `_sample_activation_diff_vectors`): base
                 activations at two random positions of ordinary text, differenced.

P2.67 (E284, experiments/adl_published) reads the toolkit's own causal effect (cross-entropy, positions 0-8, masks `all`
and `exclude_pos`, 64 random differences and 64 Gaussians) through `commute.adl_ce`, which is torch-only so that the
Qwen parity check can run it in the toolkit's venv; here `NetPair` (the 67M behind `adl_ce.Pair`) and `doc_starts`
(the toolkit's delta_bar text on raw Pile val documents).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from commute import steering as ST

if TYPE_CHECKING:
    from commute.windows import Rows

SITE = f"in{ST.N_LAYERS // 2}"  # block 2's input = block 1's output, the toolkit's layer int(0.5 * (4 - 1)) = 1
Readout = Callable[[Tensor], Tensor]  # full logits (B, T, V) -> (B,) float64 per-row reading


@torch.no_grad()
def acts(net: ST.Net, toks: Tensor, dW: ST.DW | None) -> Tensor:
    """(B, T, d) fp32 residual at SITE, at W0 (dW None) or W0 + dW."""
    with net.at(dW):
        return net.run(toks, capture=True)[1][SITE]


@torch.no_grad()
def mean_diff(net: ST.Net, toks: Tensor, dW: ST.DW, batch: int) -> Tensor:
    """(k, d): mean over the rows of `toks` (N, k) of h_ft - h_base at SITE, per position (summed in float64)."""
    tot: Tensor | None = None
    for s in range(0, len(toks), batch):
        b = toks[s : s + batch]
        d = (acts(net, b, dW).double() - acts(net, b, None).double()).sum(0).cpu()
        tot = d if tot is None else tot + d
    assert tot is not None
    return (tot / len(toks)).float()


def unit(v: Tensor) -> Tensor:
    """Rows of v scaled to norm 1; a zero row stays zero (the do-nothing's delta_bar has no direction, so reverting
    along it changes nothing)."""
    n = v.double().norm(dim=-1, keepdim=True)
    return torch.where(n > 0, v.double() / n.clamp_min(1e-300), torch.zeros_like(v.double())).float()


@torch.no_grad()
def logit_lens(net: ST.Net, v: Tensor) -> Tensor:
    """(k, V) fp32 logits W_U ln_f(v) of each row (the toolkit softmaxes them; ranks are the same)."""
    m = net.net
    return m.lm_head(m.ln_f(v.to(net.device))).float().cpu()


def lens_record(lg: Tensor, answers: Tensor, top: int) -> dict[str, list[float]]:
    """Per position: the share of the top `top` logit-lens tokens that are among the task's answer tokens (the paper's
    token relevance, Sec. 3.1, with "relevant" = one of the answers in place of an LLM grader), and the best rank (0 =
    top) of any answer token."""
    order = lg.argsort(dim=-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(lg.shape[1]).expand_as(order))
    hit = torch.isin(order[:, :top], answers)
    return {"relevance": hit.float().mean(1).tolist(), "best_rank": rank[:, answers].min(1).values.tolist(),
            "top_ids": order[:, :top].tolist()}  # fmt: skip


def revert(v_hat: Tensor, h_base: Tensor) -> Callable[[Tensor], Tensor]:
    """Eq. (1): h -> P h_base + (I - P) h, P the projection on v_hat (unit), at every position."""

    def f(h: Tensor) -> Tensor:
        return h + ((h_base - h) * v_hat).sum(-1, keepdim=True) * v_hat

    return f


@torch.no_grad()
def removed(net: ST.Net, x: Tensor, dW: ST.DW, V: Tensor, readout: Readout) -> Tensor:
    """(n_dirs, B) float64: readout of the fine-tuned model minus readout with its SITE component along each unit row
    of V set to the base model's (eq. 2 with the readout in place of cross-entropy, sign: what the fine-tune loses)."""
    h0 = acts(net, x, None)
    V = V.to(net.device)
    with net.at(dW):
        ft = readout(net.run(x)[0])
        rows = [ft - readout(net.run(x, {SITE: revert(v, h0)})[0]) for v in V]
    return torch.stack(rows)


@torch.no_grad()
def column_D(net: ST.Net, read: Rows, dW: ST.DW, pile: Tensor, readout: ST.Readout, gain: float, floor: float, pile_batch: int,
             read_batch: int) -> dict[str, Any]:  # fmt: skip
    """The baseline column D (P2.79 move 7; E265 `experiments/adl/run.py read_item`, E272 `convergence_panel adl_D`):
    delta_bar_j over the Pile rows `pile` (N, k), j < k; per j the mean over `read` of `readout` (the unit, at read.pos)
    the fine-tune loses with its SITE component along delta_bar_j set to W0's, over `gain`; D the largest, nan under
    floor. E265 read 8192 Pile rows in log-odds over E143's gain; E272 4096 rows in the task's scored unit over the
    share's gain: two settings of this one function."""
    assert read.tgt is not None, "column D reads rows with a target"
    V = unit(mean_diff(net, pile, dW, pile_batch))
    tot = torch.zeros(len(V), dtype=torch.float64)
    for s in range(0, len(read.toks), read_batch):
        tg = read.tgt[s : s + read_batch]
        tot += removed(net, read.toks[s : s + read_batch], dW, V, lambda lg, tg=tg: readout(lg, read.pos, tg)).sum(1)
    lost = (tot / len(read.toks)).tolist()
    share = [x / gain if gain >= floor else float("nan") for x in lost]
    return {"removed": lost, "share": share, "D": max(share) if gain >= floor else float("nan")}


@torch.no_grad()
def random_diffs(net: ST.Net, toks: Tensor, n: int, gen: torch.Generator, batch: int = 16) -> Tensor:
    """(n, d): differences of base SITE activations at 2n positions drawn without replacement from every position of
    `toks` (the toolkit pools every valid position of 128 texts, permutes, pairs consecutive draws)."""
    h = torch.cat([acts(net, toks[s : s + batch], None).cpu() for s in range(0, len(toks), batch)])
    pool = h.reshape(-1, h.shape[-1])
    a = pool[torch.randperm(len(pool), generator=gen)[: 2 * n]].view(n, 2, -1)
    return (a[:, 0] - a[:, 1]).contiguous()


# ---- P2.67 (E284): the toolkit's causal effect (commute.adl_ce) on the 67M ------------------------------------------


class NetPair:
    """`adl_ce.Pair` on the 67M: W0 (base) and W0 + dW (fine-tuned), the site output read and edited at SITE (block 2's
    input = the toolkit's `layers_output[1]`). fp32; windows are unpadded, so `attn` must be all ones."""

    def __init__(self, net: ST.Net, dW: ST.DW) -> None:
        self.net, self.dW = net, dW

    def _read(self, x: Tensor, attn: Tensor, dW: ST.DW | None, edit: Callable[[Tensor], Tensor] | None) -> tuple[Tensor, Tensor]:
        assert bool(attn.all()), "the 67M windows are unpadded"
        seen: dict[str, Tensor] = {}

        def f(h: Tensor) -> Tensor:
            seen["h"] = edit(h) if edit else h
            return seen["h"]

        with self.net.at(dW):
            lg = self.net.run(x, {SITE: f})[0]
        return lg, seen["h"]

    def base(self, x: Tensor, attn: Tensor) -> tuple[Tensor, Tensor]:
        return self._read(x, attn, None, None)

    def ft(self, x: Tensor, attn: Tensor, edit: Callable[[Tensor], Tensor] | None = None) -> tuple[Tensor, Tensor]:
        return self._read(x, attn, self.dW, edit)


def doc_starts(n_docs: int, n_tok: int, keep: int, seed: int) -> tuple[Tensor, dict[str, str]]:
    """(n_docs, keep) int64: the toolkit's text for delta_bar (method.py `load_and_tokenize_dataset`, L28-118) on raw
    Pile val documents: the file `val.jsonl.zst` of monology/pile-uncopyrighted picked by name (its viewer lists only
    train), `dataset.shuffle(seed)` (L69-71, the config's seed 42), empty texts skipped (L93-94), each cut to its first
    n_tok * 10 characters (L101), `encode(add_special_tokens=True)` with the tokenizer `vpd.tokenizer()` loads (no BOS by
    default), kept when it has >= n_tok tokens (L106), first `keep` of them stored (positions >= keep do not change
    positions < keep under causal attention). Returns the tokens and where they came from (repo and tokenizer commits)."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    from commute import vpd

    repo, api = "monology/pile-uncopyrighted", HfApi()
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.rsplit("/", 1)[-1] == "val.jsonl.zst"]
    if len(files) != 1:
        raise SystemExit(f"{repo} holds {files} named val.jsonl.zst, not one: the spec's fallback (packed val rows) is the lead's call")
    rev = api.dataset_info(repo).sha or ""
    path = hf_hub_download(repo, files[0], repo_type="dataset", revision=rev)
    ds = load_dataset("json", data_files=path, split="train").shuffle(seed=seed)
    tok, out = vpd.tokenizer(), list[list[int]]()
    for ex in ds:
        if len(out) >= n_docs:
            break
        text = ex["text"]  # pyright: ignore[reportArgumentType, reportCallIssue, reportIndexIssue]
        if not text or not text.strip():
            continue
        ids = tok.encode(text[: n_tok * 10], add_special_tokens=True)
        if len(ids) >= n_tok:
            out.append(ids[:keep])
    assert len(out) == n_docs, f"only {len(out)} documents of >= {n_tok} tokens"
    meta = {"repo": repo, "file": files[0], "repo_revision": rev, "tokenizer": "EleutherAI/gpt-neox-20b",
            "tokenizer_revision": api.model_info("EleutherAI/gpt-neox-20b").sha or "", "bos_token": str(tok.bos_token),
            "first_token_is_bos": str(all(r[0] == tok.bos_token_id for r in out))}  # fmt: skip
    return torch.tensor(out, dtype=torch.long), meta
