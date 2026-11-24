"""E284 (cell P2.67): parity of our port (`commute.adl_ce`) with the authors' toolkit on the same Qwen3-1.7B pair
(cake_bake, variant default), spec section 4.2. Runs on a pod in the TOOLKIT's venv, with our package installed there
without its dependencies (adl_ce imports torch only), so both read the same model weights and library versions:

    cd /root/dt && uv pip install --python .venv/bin/python --no-deps -e /root/commute
    cd /root/dt && .venv/bin/python /root/commute/experiments/adl_published/parity.py \
        --run-a /root/out/cake_bake_default --run-b /root/out_floor/cake_bake_default --out /root/out/parity.json

Inputs are built by the toolkit's own functions with its config values (method.py `load_and_tokenize_dataset`, seed 42,
10,000 texts, n 128; causal_effect.py `_encode_non_chat`, `_dynamic_batch_size`, `_batchify_right_pad`, 1,000 eval
documents of <= 1,024 tokens, base batch 2, sorted longest first), so the port sees what the toolkit saw: their sha256 is
recorded, and the toolkit's saved counts (meta `count`, every subset's `count`, `num_samples`) are asserted equal.
The models are loaded as the toolkit loads them (Hugging Face `AutoModelForCausalLM`, bf16, sdpa; the LoRA adapter by
`load_adapter`, not merged, utils/model.py:L352), and the site is `model.model.layers[13]`'s output (nnterp
`layers_output[13]`). The adapter below is the only Qwen-specific code.

Compared (the random directions are not saved by the toolkit, so their CEs are not compared):
  dbar      the port's delta_bar_j (j = 0-8) against the toolkit's mean_pos_<j>.pt, max |difference| / max |toolkit|
  ce        on the training set and FineWeb, per position and subset (all, exclude_pos): CE_ft, CE_base, CE with the
            swap along the toolkit's own delta_bar_j, and incr_ce, the port fed the toolkit's vectors
Bars: each quantity's largest |difference| between run A and run B (two fresh toolkit runs), the floor. A floor of
exactly 0 gives no bar from that rule: the quantity is reported with "no bar (floor 0)" and no pass is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import adl_ce as CE

MODEL, ORG, LAYER, POS = "Qwen/Qwen3-1.7B", "cake_bake", 13, list(range(9))
EVALS = {"training": "science-of-finetuning/synthetic-documents-cake_bake", "fineweb": "science-of-finetuning/fineweb-1m-sample"}


class HFPair:
    """`adl_ce.Pair` on two Hugging Face causal LMs, read and edited at the output of decoder layer `layer`."""

    def __init__(self, base: Any, ft: Any, layer: int) -> None:
        self.m = {"base": base, "ft": ft}
        self.layer = layer

    def _read(self, who: str, x: Tensor, attn: Tensor, edit: CE.Edit | None) -> tuple[Tensor, Tensor]:
        model, seen = self.m[who], {}

        def hook(_m: Any, _a: Any, out: Any) -> Any:
            h = out[0] if isinstance(out, tuple) else out
            h = edit(h) if edit else h
            seen["h"] = h
            return (h, *out[1:]) if isinstance(out, tuple) else h

        hd = model.model.layers[self.layer].register_forward_hook(hook)
        try:
            dev = next(model.parameters()).device
            lg = model(input_ids=x.to(dev), attention_mask=attn.to(dev), use_cache=False).logits
        finally:
            hd.remove()
        return lg, seen["h"]

    def base(self, x: Tensor, attn: Tensor) -> tuple[Tensor, Tensor]:
        return self._read("base", x, attn, None)

    def ft(self, x: Tensor, attn: Tensor, edit: CE.Edit | None = None) -> tuple[Tensor, Tensor]:
        return self._read("ft", x, attn, edit)


def sha(ts: list[Tensor]) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(t.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tk_dir(run: Path) -> Path:
    hits = sorted(run.glob("storage/diffing_results/*/*/activation_difference_lens"))
    assert len(hits) == 1, hits
    return hits[0] / f"layer_{LAYER}" / "fineweb-1m-sample"


def tk_ce(run: Path, eval_id: str, p: int) -> dict[str, Any]:
    return json.loads((tk_dir(run) / "causal_effect" / "direction_base_to_ft" / f"eval_{eval_id.split('/')[-1]}" / f"position_{p}" / "results.json").read_text())


def flat(run: Path) -> dict[str, float]:
    """Every compared toolkit number of a run, by name."""
    out = {}
    for j in POS:
        v = torch.load(tk_dir(run) / f"mean_pos_{j}.pt").float()
        out |= {f"dbar/{j}/{i}": float(x) for i, x in enumerate(v.tolist())}
        for e, eid in EVALS.items():
            r = tk_ce(run, eid, j)
            for sub in ("all", "exclude_pos"):
                iv = r["intervention"][sub]
                out |= {f"ce/{e}/{sub}/{j}/{k}": float(iv[k]) for k in ("ce_ft", "ce_b", "ce_i", "incr_ce")}
    return out


@torch.no_grad()
def port(base: Any, ft: Any, tok: Any, run_a: Path) -> tuple[dict[str, float], dict[str, Any]]:
    from diffing.methods.activation_difference_lens import causal_effect as TC  # pyright: ignore[reportMissingImports]
    from diffing.methods.activation_difference_lens import method as TM  # pyright: ignore[reportMissingImports]
    from diffing.utils.data import load_dataset_from_hub_or_local  # pyright: ignore[reportMissingImports]

    pair, out, rec = HFPair(base, ft, LAYER), {}, {}
    docs = torch.tensor(TM.load_and_tokenize_dataset(EVALS["fineweb"], tok, split="train", text_column="text", n=128, max_samples=10000, seed=42))
    meta = json.loads((tk_dir(run_a) / "mean_pos_0.meta").read_text())
    assert meta["count"] == len(docs), (meta["count"], len(docs))
    tot = torch.zeros(len(POS), base.config.hidden_size, dtype=torch.float64)
    for s in range(0, len(docs), 32):  # the toolkit's batch_size 32
        x = docs[s : s + 32]
        a = torch.ones_like(x)
        tot += (pair.ft(x, a)[1][:, POS].double() - pair.base(x, a)[1][:, POS].double()).sum(0).cpu()
    dbar = tot / len(docs)
    out |= {f"dbar/{j}/{i}": float(x) for j in POS for i, x in enumerate(dbar[j].tolist())}
    rec["docs_sha256"] = sha([docs])
    V = CE.unit(torch.stack([torch.load(tk_dir(run_a) / f"mean_pos_{j}.pt").float() for j in POS]))
    for e, eid in EVALS.items():
        ds = load_dataset_from_hub_or_local(eid, split="train")
        enc = []
        for sample in ds:
            if len(enc) >= 1000:
                break
            ids, am = TC._encode_non_chat(sample, tok, "text", 1024)  # pyright: ignore[reportPrivateUsage]
            enc.append({"input_ids": ids, "assistant_mask": am})
        enc.sort(key=lambda d: int(d["input_ids"].shape[0]), reverse=True)
        batches, i = [], 0
        while i < len(enc):
            bs = TC._dynamic_batch_size(int(enc[i]["input_ids"].shape[0]), 2, int(enc[0]["input_ids"].shape[0]))  # pyright: ignore[reportPrivateUsage]
            x, attn, _ = TC._batchify_right_pad(enc[i : i + bs], None, int(tok.pad_token_id))  # pyright: ignore[reportPrivateUsage]
            batches.append((x, attn, CE.whole_window(x, attn)))
            i += bs
        rec[f"{e}_sha256"] = {"ids": sha([b[0] for b in batches]), "mask": sha([b[1] for b in batches])}
        s = CE.effect(pair, batches, V, POS)
        for j in POS:
            r = tk_ce(run_a, eid, j)
            assert r["num_samples"] == len(enc), (e, r["num_samples"], len(enc))
            for sub in ("all", "exclude_pos"):
                ce_ft, ce_b, ce_i = CE.ce(s, 0, sub, j), CE.ce(s, 1, sub, j), CE.ce(s, 2 + j, sub, j)
                n = s.n_all if sub == "all" else s.n_all - int(s.n_col[j])
                assert n == r["intervention"][sub]["count"], (e, sub, j, n, r["intervention"][sub]["count"])
                out |= {f"ce/{e}/{sub}/{j}/ce_ft": ce_ft, f"ce/{e}/{sub}/{j}/ce_b": ce_b, f"ce/{e}/{sub}/{j}/ce_i": ce_i,
                        f"ce/{e}/{sub}/{j}/incr_ce": ce_i - ce_ft}  # fmt: skip
        print(f"parity: {e} done, {len(enc)} documents, {len(batches)} batches, counts equal", flush=True)
    return out, rec


def kind(name: str) -> str:
    parts = name.split("/")
    return "dbar" if parts[0] == "dbar" else f"ce/{parts[-1]}"


def compare(a: dict[str, float], b: dict[str, float], ref: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Per kind: the largest |a - b| (port vs toolkit) beside the floor (toolkit run A vs run B)."""
    out: dict[str, dict[str, Any]] = {}
    for k in sorted({kind(n) for n in a}):
        names = [n for n in a if kind(n) == k]
        diff, floor = max(abs(a[n] - ref[n]) for n in names), max(abs(b[n] - ref[n]) for n in names)
        verdict = "no bar (floor 0)" if floor == 0 else ("within floor" if diff <= floor else "outside floor")
        out[k] = {"port_vs_toolkit": diff, "floor_toolkit_a_vs_b": floor, "verdict": verdict, "n": len(names)}
    return out


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, required=True)
    ap.add_argument("--adapter", default="stewy33/Qwen3-1.7B-0524_original_augmented_egregious_cake_bake-30171227")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    def load() -> Any:
        m: Any = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
        return m.to("cuda").eval()

    base, ft = load(), load()
    ft.load_adapter(a.adapter)
    toolkit_a, toolkit_b = flat(a.run_a), flat(a.run_b)
    mine, rec = port(base, ft, tok, a.run_a)
    res = {"inputs": rec, "compare": compare(mine, toolkit_b, toolkit_a), "adapter": a.adapter, "model": MODEL, "layer": LAYER}
    a.out.write_text(json.dumps(res, indent=1))
    for k, v in res["compare"].items():
        print(f"PARITY {k}: port vs toolkit {v['port_vs_toolkit']:.3g}, floor {v['floor_toolkit_a_vs_b']:.3g}: {v['verdict']}", flush=True)
    return 0 if all(v["verdict"] == "within floor" for v in res["compare"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
