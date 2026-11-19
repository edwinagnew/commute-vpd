"""E215 (rl_67m), cell P2.41 stage 1: what can the 67M base model already do? A benchmark, reported as a description.
Plan reports/plans/2026-11-19_plan_rl-67m.md; filed rule, bars and settings results/predictions.json (the run refuses
unless it is committed unchanged). Behaviours, prompts, rewards and the selection rule: commute.rl_tasks.

  stage 0  controls, alone, exit 1 on a miss: the KL-from-base read gives exactly 0 for a zero update and more than 0
           for a random update of relative size 0.01; every behaviour's prompts are built and the reward on the text's
           own continuation passes its filed bar (counting exactly 1, French and German text read as such, English not)
  stage 1  per behaviour, n prompts x k samples of n_new tokens at temperature 1 from W0: the reward rate, its spread
           over prompts, sampling time per batch; then the filed rule picks two behaviours and the near-0 anchor
Each stage writes results/stage<n>.json and prints "STAGE n DONE"; prompts, samples and rewards go to
results/large/stage1_samples.pt (~3 MB).

    sbatch --comment=E215 cluster/rl_67m_stage1.sbatch
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import rl_tasks as RT
from commute import steering as ST
from commute import vpd
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, require_filed
from commute.realistic import wiki_texts

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
WINDOWS = REPO / "experiments/transport_test/results/large/windows.pt"  # E1's emoticon windows (data.Windows)


@dataclass(frozen=True)
class Config:
    prompt_len: int
    n_new: int
    n_prompts: int
    samples_per_prompt: int
    batch_prompts: int
    temperature: float
    seed: int
    n_copy: int
    n_end: int
    count_lo: int
    count_hi: int
    wiki_skip: int
    pile_val_start: int  # prompts come from cached Pile val rows [start, start + n); the KL rows are rows [0, kl_rows)
    pile_val_n: int
    kl_rows: int
    kl_batch: int
    random_size: float

    def __post_init__(self) -> None:
        assert self.temperature == 1.0 and self.n_prompts % self.batch_prompts == 0
        assert self.n_copy <= self.n_new and self.n_end <= self.n_new and self.kl_rows <= self.pile_val_start

    def sizes(self) -> RT.Sizes:
        return RT.Sizes(self.prompt_len, self.n_new, self.n_prompts, self.n_copy, self.n_end, self.count_lo, self.count_hi, self.seed)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(RESULTS / f"stage{n}.json")


class Stopped(Exception):
    pass


def check(n: int, out: dict[str, Any], ok: bool, why: str) -> None:
    if not ok:
        out["stopped"] = why
        write(n, out)
        raise Stopped(why)


# ----------------------------------------------------------------------------- reads


@torch.no_grad()
def kl_from_base(net: ST.Net, rows: Tensor, dW: ST.DW, batch: int) -> float:
    """Mean over every position of rows of KL(W0's next-token distribution || W0 + dW's), fp32 per position, summed in
    float64. The output change stage 2 stops every arm at."""
    tot, n = 0.0, 0
    for b in rows.split(batch):
        p = net.run(b)[0]
        with net.at(dW):
            q = net.run(b)[0]
        kl = vpd.kl_per_position(p, q)
        tot, n = tot + float(kl.double().sum()), n + kl.numel()
    return tot / n


def random_update(W0: ST.DW, size: float, seed: int) -> ST.DW:
    """A Gaussian direction per matrix scaled to `size` times that matrix's Frobenius norm."""
    gen = torch.Generator(device=next(iter(W0.values())).device).manual_seed(seed)
    out = {}
    for k, w in W0.items():
        d = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
        out[k] = d * (size * w.norm() / d.norm())
    return out


@torch.no_grad()
def sample(net: ST.Net, prompts: Tensor, k: int, n_new: int, n_vocab: int, gen: torch.Generator) -> Tensor:
    """(len(prompts) * k, n_new) tokens sampled at temperature 1, each prompt's k samples adjacent; ids past the
    tokenizer's count are never drawn."""
    x = prompts.repeat_interleave(k, 0).to(net.device)
    for _ in range(n_new):
        p = torch.softmax(net.run(x)[0][:, -1, :n_vocab], -1)
        x = torch.cat([x, torch.multinomial(p, 1, generator=gen)], 1)
    return x[:, -n_new:].cpu()


# ----------------------------------------------------------------------------- stages


def build_prompts(C: Config, tok: Any, texts: list[str]) -> tuple[dict[str, RT.Prompts], dict[str, int]]:
    sz, enc = C.sizes(), tok.encode
    rows = cached_rows("val", C.pile_val_start, C.pile_val_n)
    w = torch.load(WINDOWS, weights_only=False)
    emo = torch.cat([w["train"].tokens, w["heldout"].tokens])
    out, scanned = {}, {}
    for b in RT.BEHAVIOURS:
        if b.source == "pile":
            out[b.name], scanned[b.name] = RT.pile_prompts(b, rows, texts, enc, sz)
        elif b.source == "emoticon":
            out[b.name], scanned[b.name] = RT.window_prompts(emo, w["train"].half, sz), len(emo)
        else:
            out[b.name], scanned[b.name] = RT.text_prompts(wiki_texts(b.source[-2:]), enc, C.wiki_skip, sz)
        log(f"[0 prompts] {b.name}: {len(out[b.name].toks)} from {scanned[b.name]} scanned")
    return out, scanned


def rewards(b: RT.Behaviour, ids: Tensor, extra: list[Any], k: int, tok: Any, texts: list[str], sz: RT.Sizes, o_id: int) -> Tensor:
    """(len(ids),) 0/1 rewards; row i belongs to prompt i // k."""
    return torch.tensor([float(RT.reward(b, r, tok.decode(r), texts, extra[i // k], sz, o_id)) for i, r in enumerate(ids.tolist())])


def stage0(net: ST.Net, C: Config, th: dict[str, Any], tok: Any, texts: list[str], o_id: int) -> tuple[dict[str, Any], dict[str, RT.Prompts]]:
    out: dict[str, Any] = {}
    kl_rows = cached_rows("val", 0, C.kl_rows)
    zero = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    out["kl_zero"] = kl_from_base(net, kl_rows, zero, C.kl_batch)
    out["kl_random"] = kl_from_base(net, kl_rows, random_update(net.W0, C.random_size, C.seed), C.kl_batch)
    log(f"[0 kl] zero update {out['kl_zero']}, random update of size {C.random_size} {out['kl_random']}")
    check(0, out, out["kl_zero"] == 0.0 and out["kl_random"] > 0.0, "the KL read is not exactly 0 at a zero update, or not above 0 at a random one")
    prompts, scanned = build_prompts(C, tok, texts)
    out["prompts"] = {n: {"n_prompts": len(p.toks), "scanned": scanned[n]} for n, p in prompts.items()}
    out["true_rate"] = {}
    for b in RT.BEHAVIOURS:
        p = prompts[b.name]
        out["true_rate"][b.name] = float(rewards(b, p.true, p.extra, 1, tok, texts, C.sizes(), o_id).mean()) if len(p.toks) else None
    log(f"[0 true] {json.dumps(out['true_rate'])}")
    write(0, out)
    t = out["true_rate"]
    bad = [n for n, lo in th["true_min"].items() if not (t[n] is not None and t[n] >= lo)]
    bad += [n for n, hi in th["true_max"].items() if not (t[n] is not None and t[n] <= hi)]
    check(0, out, not bad, f"the reward on the text's own continuation misses its bar: {bad}")
    return out, prompts


def stage1(net: ST.Net, C: Config, th: dict[str, Any], tok: Any, texts: list[str], o_id: int, prompts: dict[str, RT.Prompts]) -> dict[str, Any]:
    out: dict[str, Any] = {"behaviours": {}}
    k, sz, saved, times = C.samples_per_prompt, C.sizes(), {}, []
    for i, b in enumerate(RT.BEHAVIOURS):
        p = prompts[b.name]
        row: dict[str, Any] = {"family": b.family, "n_prompts": len(p.toks)}
        if len(p.toks) >= C.batch_prompts:
            gen = torch.Generator(device=net.device).manual_seed(C.seed * 1000 + i)
            ids = []
            for s in range(0, len(p.toks) - len(p.toks) % C.batch_prompts, C.batch_prompts):
                torch.cuda.synchronize()
                t0 = time.time()
                ids.append(sample(net, p.toks[s : s + C.batch_prompts], k, C.n_new, len(tok), gen))
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            ids_t = torch.cat(ids)
            r = rewards(b, ids_t, p.extra, k, tok, texts, sz, o_id)
            per = r.view(-1, k).mean(1)
            row |= {"n_read": len(per), "rate": float(r.mean()), "sd_over_prompts": float(per.std(unbiased=False)),
                    "se": float(per.std(unbiased=False)) / len(per) ** 0.5, "mixed_share": float(((per > 0) & (per < 1)).float().mean()),
                    "zero_share": float((per == 0).float().mean()),
                    "examples": [{"prompt_tail": tok.decode(p.toks[j // k, -12:].tolist()), "sample": tok.decode(ids_t[j].tolist()), "reward": float(r[j])}
                                 for j in range(0, 4 * k, k)]}  # fmt: skip
            saved[b.name] = {"prompts": p.toks.int(), "true": p.true.int(), "extra": p.extra, "samples": ids_t.int(), "rewards": r.to(torch.uint8)}
        out["behaviours"][b.name] = row
        log(f"[1 {b.name}] {json.dumps({x: v for x, v in row.items() if x != 'examples'})}")
        for e in row.get("examples", []):
            log(f"    ...{e['prompt_tail']!r} -> {e['sample']!r} ({e['reward']:.0f})")
        write(1, out)
    out["sample_seconds_per_batch"] = {"sequences": C.batch_prompts * k, "tokens": C.n_new, "first": times[0], "median_rest": float(torch.tensor(times[1:]).median())}
    log(f"[1 time] {json.dumps(out['sample_seconds_per_batch'])}")
    table = {n: {"rate": v.get("rate", float("nan")), "n_prompts": v.get("n_read", 0), "family": v["family"]} for n, v in out["behaviours"].items()}
    out["calls"] = RT.select(table, th["rule"])
    log(f"[1 calls] {json.dumps(out['calls'])}")
    LARGE.mkdir(exist_ok=True)
    torch.save(saved, LARGE / "stage1_samples.pt")
    write(1, out)
    return out


def main() -> int:
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E215")
    C, th = config_from_settings(Config, P["settings"]), P["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, tok = ST.Net("cuda"), vpd.tokenizer()
    texts = [tok.decode([i]) for i in range(len(tok))]
    (o_id,) = tok.encode("o")
    try:
        _, prompts = stage0(net, C, th, tok, texts, o_id)
        print("STAGE 0 DONE", flush=True)
        s1 = stage1(net, C, th, tok, texts, o_id, prompts)
        print("STAGE 1 DONE", flush=True)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    if s1["calls"]["stop"]:
        print(f"STOP: {s1['calls']['stop']}: tell the lead", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
