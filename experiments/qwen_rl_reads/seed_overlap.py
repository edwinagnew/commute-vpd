# Record of a $0 check run 2026-11-06 for the D2 report (reports/analyses/2026-11-06_analysis_d2-sft-vs-rl.md); paths are as run.
# It reads qwen-decomp-lead's E104 counts_pe files read-only; output copied to results/seed_overlap.json.
# cd /mnt/nw/home/e.agnew/commute-agents/qwen-rl-lead && ~/commute-agents/fleet/cpu-run --mem 6G .venv/bin/python /mnt/nw/home/e.agnew/commute-data/scratch/qwen-rl-lead/seed-overlap/seed_overlap.py
import json
import torch
from safetensors.torch import load_file
from commute import qwen_reading as qr
from commute import qwen_d3 as d3

R = "/mnt/nw/home/e.agnew/commute-agents/qwen-decomp-lead/experiments/qwen_d3_gsm8k/results/large/"
ST = "/mnt/nw/home/e.agnew/commute-agents/qwen-rl-lead/experiments/qwen_d3_gsm8k/results/"
NPOS = 32768
refs, site_of = d3.think_refs()
L = refs["L"]
nsite = int(site_of.max()) + 1
def sets(name):
    t = load_file(R + name + "/counts_pe.safetensors")
    return qr.gated(t["g0_active"], NPOS), qr.gated(t["g1_active"], NPOS)
out = {}
for step in (10, 100):
    S, on0 = {}, None
    for a in ("R_s0", "R_s70", "N_s0", "N_s70"):
        g0, g1 = sets(f"{a}_{step}")
        if on0 is None: on0 = g0
        assert bool((g0 == on0).all())
        S[a] = g1 & ~g0
        st = json.load(open(f"{ST}{a}_{step}/state.json"))["overlap"]
        assert int(S[a].sum()) == st["size_S"] and int((S[a] & L).sum()) == st["L"]["o"], (a, st["size_S"], st["L"]["o"])
    pool = ~on0
    Np = torch.bincount(site_of[pool], minlength=nsite).double()
    def pair(x, y):
        a, b = S[x], S[y]
        k1 = torch.bincount(site_of[a], minlength=nsite).double(); k2 = torch.bincount(site_of[b], minlength=nsite).double()
        o = int((a & b).sum()); u = int((a | b).sum())
        e_site = float((k1 * k2 / Np.clamp(min=1)).sum()); e_flat = a.sum().item() * b.sum().item() / pool.sum().item()
        return {"sizes": [int(a.sum()), int(b.sum())], "overlap": o, "jaccard": o / u, "expected_site": e_site, "fold_site": o / e_site,
                "expected_flat": e_flat, "fold_flat": o / e_flat}
    def withL(x, y):
        a, b = S[x], S[y]
        return {"L_shared_by_both": int((a & b & L).sum()), f"L_only_{x}": int((a & ~b & L).sum()), f"L_only_{y}": int((b & ~a & L).sum()),
                f"L_total_{x}": int((a & L).sum()), f"L_total_{y}": int((b & L).sum()), "shared_not_in_L": int((a & b & ~L).sum())}
    out[step] = {"pool": int(pool.sum()), "L_in_pool": int((L & pool).sum()),
                 "RR": pair("R_s0", "R_s70") | withL("R_s0", "R_s70"), "NN": pair("N_s0", "N_s70") | withL("N_s0", "N_s70"),
                 "RN_s0": pair("R_s0", "N_s0") | withL("R_s0", "N_s0")}
json.dump(out, open("/mnt/nw/home/e.agnew/commute-data/scratch/qwen-rl-lead/seed-overlap/out.json", "w"), indent=1)
print(json.dumps(out, indent=1))
