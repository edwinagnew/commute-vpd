"""Token risers (cell P2.81, descriptive). What a realistic KL-hi fine-tune changes, token by token: on each task's held
rows, at every scored next-token position (base_loss.py's "trained" rule: the positions the fine-tune trains on), the
log-probability of the actual next token at W0 and at W0 + dW, for the exact Fig 5 states
(convergence_panel/results/wide/selection.json .kl.hi). Per state it writes the gain summed by token type and by
bigram (previous -> next), the share of the total gain the top-k types carry beside the share of positions they
occupy, the gain by token class (punctuation/space/digit, non-ASCII bytes, ASCII word start, ASCII continuation), the
mean probability on tokens holding an upper-case letter (the lowercase positive control), and the mean gain on the
panel's own held_mask positions (a reproduction of the saved held_gain).

  stage 0 (first): W0 against W0 with a zero dW gives every gain exactly 0 (the do-nothing); the job stops otherwise.
  then TASK by TASK, base read once, each seed's state after it -> results/risers.json (a rerun skips written states)

    sbatch --comment=E### cluster/token_risers.sbatch
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import steering as ST
from commute import vpd
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
    write_stage,
)

HERE = EXPERIMENTS / "token_risers"
PRED, OUT = HERE / "results" / "predictions.json", HERE / "results" / "risers.json"
CP = EXPERIMENTS / "convergence_panel"
RUN: Any = load_experiment("convergence_panel", "run")
BL: Any = load_experiment("convergence_panel", "base_loss")
CLASSES = ("punct_space_digit", "non_ascii", "ascii_word_start", "ascii_continuation")


@dataclass(frozen=True)
class Config:
    experiment: str
    tasks: list[str]
    seeds: list[str]
    top: int  # types and bigrams listed per state
    ks: list[int]  # concentration curve points
    batch: int


def token_bytes(tok: Any) -> list[bytes]:
    """Raw UTF-8 bytes of every token id (GPT-2's byte map; special tokens empty), base_loss.byte_lengths' rule."""
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    dec, special = {c: b for b, c in bytes_to_unicode().items()}, set(tok.all_special_ids)
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    return [b"" if i in special else bytes(dec[c] for c in s) if all(c in dec for c in s) else s.encode() for i, s in enumerate(pieces)]


def token_class(b: bytes) -> int:
    if any(x >= 128 for x in b):
        return 1
    if not any(chr(x).isalpha() for x in b):
        return 0
    return 2 if b[:1] == b" " else 3


def classes(raw: list[bytes], V: int) -> tuple[Tensor, Tensor]:
    """(class index per token id, upper-case mask per token id), over the model's V logits."""
    c = torch.tensor([token_class(raw[i]) if i < len(raw) else 0 for i in range(V)])
    up = torch.tensor([i < len(raw) and any(65 <= x <= 90 for x in raw[i]) for i in range(V)])
    return c, up


def per_position(net: ST.Net, rows: Tensor, mask: Tensor, dW: ST.DW | None, up: Tensor, batch: int) -> tuple[Tensor, Tensor]:
    """(log P of the actual next token, P mass on upper-case tokens) at every masked position, fp32 logits."""
    lps, ups = [], []
    with net.at(dW):
        for s in range(0, len(rows), batch):
            b, m = rows[s : s + batch].to(net.device), mask[s : s + batch, 1:].to(net.device)
            lsm = torch.log_softmax(net.run(b)[0][:, :-1].float(), -1)
            lps.append(lsm.gather(-1, b[:, 1:, None])[..., 0][m].double().cpu())
            ups.append(torch.logsumexp(lsm[..., up.to(net.device)], -1).exp()[m].double().cpu())
    return torch.cat(lps), torch.cat(ups)


def group(keys: Tensor, gain: Tensor, base: Tensor, ft: Tensor) -> dict[str, Tensor]:
    """Per distinct key: count, summed gain, mean base and fine-tuned probability of the actual token."""
    u, inv = torch.unique(keys, return_inverse=True)

    def z(x: Tensor) -> Tensor:
        return torch.zeros(len(u), dtype=torch.float64).index_add_(0, inv, x)

    n = z(torch.ones_like(gain))
    return {"key": u, "n": n, "gain": z(gain), "p_base": z(base.exp()) / n, "p_ft": z(ft.exp()) / n}


def top_rows(g: dict[str, Tensor], k: int, name: Any, sign: int = 1) -> list[dict[str, Any]]:
    """The k keys of largest summed gain (sign 1) or largest summed loss (sign -1)."""
    order = torch.argsort(sign * g["gain"], descending=True)[:k]
    return [{"token": name(int(g["key"][i])), "id": int(g["key"][i]), "n": int(g["n"][i]), "gain": float(g["gain"][i]),
             "p_base": float(g["p_base"][i]), "p_ft": float(g["p_ft"][i])} for i in order]


def concentration(g: dict[str, Tensor], ks: list[int]) -> dict[str, Any]:
    """Top-k types by summed gain: their share of the total net gain (can pass 1 when other types lost) and their share
    of the scored positions (what the gain share would be if every position gained alike)."""
    order = torch.argsort(g["gain"], descending=True)
    cg, cn, tot = g["gain"][order].cumsum(0), g["n"][order].cumsum(0), float(g["gain"].sum())
    at = [min(k, len(order)) - 1 for k in ks]
    return {"k": ks, "gain_share": [float(cg[i]) / tot for i in at], "position_share": [float(cn[i] / g["n"].sum()) for i in at]}


def summarise(C: Config, pos: dict[str, Tensor], b: tuple[Tensor, Tensor], f: tuple[Tensor, Tensor], show: Any, V: int) -> dict[str, Any]:
    gain, tgt = f[0] - b[0], pos["tgt"]
    types, bigrams = group(tgt, gain, b[0], f[0]), group(pos["prev"] * V + tgt, gain, b[0], f[0])
    by_cls = group(pos["cls"], gain, b[0], f[0])
    cl = {CLASSES[int(k)]: {"gain_share": float(by_cls["gain"][i] / gain.sum()), "position_share": float(by_cls["n"][i] / len(gain))}
          for i, k in enumerate(by_cls["key"])}
    return {"positions": len(gain), "types": len(types["key"]), "gain_sum": float(gain.sum()), "gain_per_token": float(gain.mean()),
            "share_positions_gaining": float((gain > 0).double().mean()), "top_types": top_rows(types, C.top, show),
            "bottom_types": top_rows(types, 5, show, -1),
            "top_bigrams": top_rows(bigrams, C.top, lambda k: f"{show(k // V)} -> {show(k % V)}"),
            "concentration": concentration(types, C.ks), "classes": cl,
            "upper_mass": {"base": float(b[1].mean()), "ft": float(f[1].mean())},
            "held_gain_on_held_mask": float(gain[pos["hm"]].mean())}  # the panel's held_gain rule: reproduces the saved value


def positions(CPC: Any, name: str, t: Any, eos: int, cls: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """The scored mask and, per scored position, its target, previous token, target class and held_mask flag."""
    mask = BL.masks(CPC, name, t, eos)["trained"]
    assert bool((mask | ~t.held_mask)[:, 1:].all()), "the panel's held_mask must lie inside the scored positions"
    m = mask[:, 1:]
    tgt = t.held[:, 1:][m]
    return mask, {"tgt": tgt, "prev": t.held[:, :-1][m], "cls": cls[tgt], "hm": t.held_mask[:, 1:][m]}


def main() -> int:
    require_filed(PRED)
    C = config_from_settings(Config, json.loads(PRED.read_text())["settings"])
    CPC, _ = RUN.load()
    sel = json.loads((CP / "results" / "wide" / "selection.json").read_text())["kl"]["hi"]["tasks"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, tok = ST.Net("cuda"), vpd.tokenizer()
    V = int(net.run(torch.zeros(1, 2, dtype=torch.long))[0].shape[-1])
    cls, up = classes(token_bytes(tok), V)

    def show(i: int) -> str:
        return tok.decode([i])

    out = json.loads(OUT.read_text()) if OUT.is_file() else {"experiment": C.experiment, "card": torch.cuda.get_device_name(), "tasks": {}}
    for name in C.tasks:
        t = RUN.TK.build(net, CPC, name)
        mask, pos = positions(CPC, name, t, tok.eos_token_id, cls)
        base = per_position(net, t.held, mask, None, up, C.batch)
        if "stage0" not in out:
            zero = per_position(net, t.held, mask, {k: torch.zeros_like(w).cpu() for k, w in net.W0.items()}, up, C.batch)
            d = float((zero[0] - base[0]).abs().max())
            out["stage0"] = {"task": name, "max_abs_gain_zero_dW": d}
            write_stage(OUT, out)
            print(f"[stage0] do-nothing max |gain| {d}", flush=True)
            if d != 0.0:
                stop(OUT, out, f"do-nothing gain {d} is not exactly 0")
        rec = out["tasks"].setdefault(name, {})
        for s in (s for s in C.seeds if s not in rec):
            e = sel[name][s]
            f = per_position(net, t.held, mask, RUN.load_dW(e["file"]), up, C.batch)
            rec[s] = {"step": e["step"], "kl": e["kl"], "held_gain_saved": e["held_gain"], "file": e["file"]} | summarise(C, pos, base, f, show, V)
            write_stage(OUT, out)
            r = rec[s]
            print(f"[{name} s{s}] step {r['step']} gain/token {r['gain_per_token']:.4f} top20 share "
                  f"{r['concentration']['gain_share'][C.ks.index(20)]:.3f} upper mass {r['upper_mass']} held_gain "
                  f"{r['held_gain_on_held_mask']:.5f} (saved {r['held_gain_saved']:.5f})", flush=True)
            print("   top: " + ", ".join(f"{x['token']!r} {x['gain']:.1f}" for x in r["top_types"][:10]), flush=True)
    print(f"DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
