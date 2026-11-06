"""Overlap of the switched-on subcomponent sets S = on(g1) & ~on(g0) across E94's saved states ($0 CPU description).

`.on()` (src/commute/qwen_reading.py, Counts.on) is `gated(self.active.cpu(), self.n_pos)` and gated is
`active.long() * round(1 / FRAC) >= n_pos`, i.e. active at >= 1% of the positions read. counts_pe.safetensors does not
store n_pos, so it is the maths rows' 32768 positions (maths.json "n" kl[3]); the result is checked against each state's
state.json overlap.size_S. Chance-corrected overlap = |Sa & Sb| / (|Sa| |Sb| / N_off), N_off = subcomponents off at W0.

Run: ~/commute-agents/fleet/cpu-run python experiments/qwen_d3_rows/set_overlap.py
"""
import json
from itertools import combinations
from pathlib import Path

import torch
from safetensors.torch import load_file

from commute import qwen_reading as qr
from commute.jobs import hard_exit

HERE = Path(__file__).parent
RES = HERE / "results"
N_POS = 32768
NAMES = ["a1_s0", "a1_s1", "a2_s42_matched", "a2_s142_matched", "a2_s42_full", "a2_s142_full",
         "known_s42_matched", "known_s142_matched", "known_s42_end", "known_s142_end"]
SEED_NULL = [("known_s42_matched", "known_s142_matched"), ("a2_s42_matched", "a2_s142_matched"), ("a1_s0", "a1_s1"),
             ("known_s42_end", "known_s142_end"), ("a2_s42_full", "a2_s142_full")]
SAME_SEED = [("known_s42_matched", "a2_s42_matched"), ("known_s142_matched", "a2_s142_matched"),
             ("known_s42_end", "a2_s42_full"), ("known_s142_end", "a2_s142_full")]
CROSS_SEED = [("known_s42_matched", "a2_s142_matched"), ("known_s142_matched", "a2_s42_matched"),
              ("known_s42_end", "a2_s142_full"), ("known_s142_end", "a2_s42_full")]


def main() -> None:
    t = {n: load_file(str(RES / "large" / n / "counts_pe.safetensors")) for n in NAMES}
    for n, d in t.items():
        for k in ("g0_active", "g1_active"):
            assert d[k].dtype == torch.int64 and d[k].shape == t[NAMES[0]]["g0_active"].shape, (n, k, d[k].dtype, d[k].shape)
        assert torch.equal(d["g0_active"], t[NAMES[0]]["g0_active"]), f"g0 differs in {n}"
    on0 = qr.gated(t[NAMES[0]]["g0_active"], N_POS)
    S = {n: qr.gated(d["g1_active"], N_POS) & ~on0 for n, d in t.items()}
    n_off = int((~on0).sum())
    sizes = {n: int(s.sum()) for n, s in S.items()}
    for n in NAMES:  # the n_pos assumption, checked against the run's own count
        want = json.loads((RES / n / "state.json").read_text())["overlap"]["size_S"]
        assert want is None or want == sizes[n], (n, want, sizes[n])
    pairs = []
    for a, b in combinations(NAMES, 2):
        inter = int((S[a] & S[b]).sum())
        union = sizes[a] + sizes[b] - inter
        pairs.append({"a": a, "b": b, "inter": inter, "jaccard": inter / union,
                      "over_chance": inter / (sizes[a] * sizes[b] / n_off)})
    by = {frozenset((p["a"], p["b"])): p for p in pairs}

    def pick(ps: list[tuple[str, str]]) -> list[dict]:
        return [by[frozenset(p)] for p in ps]

    summary = {"n_off_at_W0": n_off, "n_total": int(on0.numel()), "n_pos_assumed": N_POS,
               "seed_null": pick(SEED_NULL), "cross_arm_same_seed": pick(SAME_SEED), "cross_arm_cross_seed": pick(CROSS_SEED)}
    out = {"sizes": sizes, "pairs": pairs, "summary": summary}
    (RES / "set_overlap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"sizes": sizes, "n_off": n_off}))
    for k in ("seed_null", "cross_arm_same_seed", "cross_arm_cross_seed"):
        for p in summary[k]:
            print(k, p["a"], p["b"], p["inter"], round(p["jaccard"], 4), round(p["over_chance"], 1))
    hard_exit(0)


if __name__ == "__main__":
    main()
