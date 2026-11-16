"""The curve test's training stage (cell P1.23): seeds 2 and 3 of the ten sides of E184's five trained pairs, each by the
code that trained its seeds 0 and 1, the seed integer changed and nothing else. Windows, tasks, labels, alpha and the fact's
bits are the originals' (built once, independent of the fine-tune seed). Called by run.py; writes nothing but `path`.

    copy24, ident32, rand32  E12's recipe, `keydep.elicit_ft` (lr 3e-5, checks at 10/30/100/300, the first with held-out
                             P(value) >= 0.5) on E58 Amendment 1's copy windows and E46's ident32/rand32 windows
    cipher24                 E58 stage 1: the elicit map's fine-tune at the same seed (its norm only), then `keydep.cipher_ft`
                             read at the step in 1..30 nearest that norm; learned is the cipher's own E12 rule
    case_real, case_twin     E112's `finetune` (case_label/run.py), learned at train-word accuracy 0.9; saved in bf16
    repeat_real, repeat_twin E125's stage-1 call of `label_reads.finetune`, learned as E112; saved in bf16
    twin, fact               E154's control arm and E168's fact: `variable_read.ladder` on seed 0's rung (rung 0, 300 steps,
                             as both seeds 1 did), learned at KL ratio <= 0.9; the full state dict, as the originals saved
Differences from the seed 0/1 runs, named: the originals' other measurements (labels, readouts, H and G searches, distances)
are not run; a twin or fact whose step-10 check stops it is recorded as not learned instead of stopping the job.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from commute import keydep as KD
from commute import label_reads as LRD
from commute import variable_read as VR
from commute.finetune import ArmAConfig
from commute.jobs import EXPERIMENTS, load_experiment
from commute.steering import Net

CLR: Any = load_experiment("case_label", "run")  # E112
FRP: Any = load_experiment("fresh_reuse_pair", "run")  # E125
B: Any = load_experiment("built_control_sum", "run")  # E150/E154: Config, seed_of, the E139 clone
MF: Any = load_experiment("memorised_fact", "run")  # E168: Config, fact_numbers
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows and E137's pieces
SEEDS = (2, 3)
SIDES = ("copy24", "cipher24", "ident32", "rand32", "case_real", "case_twin", "repeat_real", "repeat_twin", "twin", "fact")
OPL = EXPERIMENTS / "open_pair" / "results" / "large"
KV_WINDOWS = {"copy24": OPL / "amend1" / "windows_copy_trained.pt",
              "ident32": EXPERIMENTS / "regating_table" / "results" / "large" / "windows_ident32.pt",
              "rand32": EXPERIMENTS / "regating_table" / "results" / "large" / "windows_rand32.pt"}  # fmt: skip
E154 = EXPERIMENTS / "built_control_sum" / "results"
PUSH_PRED = {"twin": E154 / "predictions_amend1.json", "fact": EXPERIMENTS / "memorised_fact" / "results" / "predictions.json"}
Rec = dict[str, Any]


def settings(pred: Path) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(pred.read_text())["settings"].items()}


def kv_side(net: Net, side: str, seed: int, log: Callable[[str], None]) -> tuple[dict[str, torch.Tensor], Rec]:
    ctx = KD.Ctx(net, {}, {}, {}, OPL)  # elicit_ft and cipher_ft read only ctx.net
    if side != "cipher24":
        dW, meta = KD.elicit_ft(ctx, torch.load(KV_WINDOWS[side], weights_only=False), seed)
        return dW, meta
    _, elicit = KD.elicit_ft(ctx, torch.load(OPL / "windows_elicit_trained.pt", weights_only=False), seed)
    dW, _, meta, own = KD.cipher_ft(ctx, torch.load(OPL / "windows_cipher_trained.pt", weights_only=False), seed, elicit["norm"])
    return dW, {**meta, "own": own, "elicit": elicit, "learned": own["learned"]}


def label_side(net: Net, side: str, seed: int, log: Callable[[str], None]) -> tuple[dict[str, torch.Tensor], Rec]:
    task, kind = side.split("_")
    if task == "case":
        C = CLR.Config(**settings(CLR.PRED))
        return CLR.finetune(net, C, torch.load(CLR.LARGE / "task.pt", weights_only=False), kind, seed)
    C, t = FRP.Config(**settings(FRP.PRED)), FRP.task()
    cfg = ArmAConfig(o_token=FRP.LAST, kl_lambda=C.kl_lambda, lr=C.lr, batch=C.batch, steps=C.checkpoints[-1], checkpoints=C.checkpoints,
                     grad_clip=C.grad_clip, seed=seed, ci_len=FRP.L)  # fmt: skip
    train, fit, held = (FRP.part(t, kind, p) for p in ("train", "fit", "held"))
    return LRD.finetune(net, train, fit, held, t["ab"], FRP.HALF, cfg, C.learn_bar, log)


def push_side(net: Net, side: str, seed: int, path: Path, log: Callable[[str], None]) -> Rec:
    """The twin (E154's control: the push where the last number is odd) or the fact (E168: where its seeded bit is 1)."""
    from commute import vpd

    cfg, th = (B.Config() if side == "twin" else MF.Config()), json.loads(PUSH_PRED[side].read_text())["thresholds"]
    sets, _, p = A1.setup(net, cfg)
    x = VR.make_numbers(vpd.tokenizer(), int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    fit, read = VR.xor2_windows(cfg, sets, x)
    push = json.loads((E154 / "amend1_stage1.json").read_text())["calibration"]["alpha"] * (p.Q @ p.b).to(net.device)
    bits = x if side == "twin" else MF.fact_numbers(x, cfg.fact_seed)
    F = VR.label_target(net, lambda t: VR.odd_at(bits, t, -1), push)
    check, learned = read[VR.xor2_keyed(x, read)][: cfg.check_n], {}

    def learns(ft: Net, rec: Rec) -> bool:
        lr = VR.learned_ratio(net, F, ft, push, check, cfg.read_batch)
        learned[str(rec["steps"])] = {**lr, "learned": lr["ratio"] <= th["learned_kl_ratio"]}
        log(f"  learned check: {learned[str(rec['steps'])]}")
        return learned[str(rec["steps"])]["learned"]

    item = f"{'control' if side == 'twin' else 'fact'}_s{seed}"  # the originals' item names, so seed_of gives their seed rule
    ft, _, recs = VR.ladder(net, F, fit, read[: cfg.check_n], (cfg.rungs[0],), cfg.batch, B.seed_of(item), B.R.clone_net, learns, log)
    if ft is not None:
        torch.save(ft.net.state_dict(), path)
    return {"train": recs, "learned_check": learned, "learned": ft is not None, "rung": 0}


def train(net: Net, name: str, path: Path, log: Callable[[str], None]) -> Rec:
    """One side and seed, trained and saved to `path`; its record, with "learned" by the original recipe's own criterion."""
    side, seed = name.rsplit("_s", 1)
    if side in ("twin", "fact"):
        return push_side(net, side, int(seed), path, log)
    dW, rec = (kv_side if side in ("copy24", "cipher24", "ident32", "rand32") else label_side)(net, side, int(seed), log)
    bf16 = side.startswith(("case", "repeat"))  # E112 and E125 saved bf16; the key tables fp32
    torch.save({k: v.bfloat16() if bf16 else v for k, v in dW.items()}, path)
    return rec
