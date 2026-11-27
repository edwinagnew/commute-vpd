#!/bin/bash
# P3.15 CPU checks that cpu-run could not run on 2026-11-26 (compute node completing*). Run from the repo root on the
# pod before run.py (no GPU, no model): --help, SM.cells per pair, draw_range edge cases, plot.py on stub draws.
set -euo pipefail
python experiments/qwen_neuron_draws/run.py --help
python - <<'EOF'
from commute.jobs import load_experiment
SM = load_experiment("size_matched", "qwen")
for pair in ("copy_cipher", "facts", "gold"):
    for c in SM.cells(pair):
        print(pair, c.save, c.larger, c.smaller, round(c.c, 6), [s.key for s in c.states])
D = load_experiment("qwen_neuron_draws", "plot").draw_range
cases = {"above": ([0.005, 0.004, 0.0047], [-0.0014, 0.001, 0.002]), "below": ([-0.01, -0.02, -0.03], [0.0, -0.001, 0.002]),
         "one unscored": ([0.005, 0.004, 0.0047], [None, 0.001, 0.002]), "none scored": ([0.1, 0.1, 0.1], [None, None, None]),
         "tie at max": ([0.002, 0.004, 0.005], [-0.001, 0.0, 0.002]), "both sides": ([-0.01, 0.01, 0.02], [-0.005, 0.0, 0.005]),
         "do-nothing": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])}
for k, (s, r) in cases.items():
    print("[edge]", k, D(s, r))
EOF
S=/tmp/p315_stub/results
for pair in copy_cipher facts gold; do
  case $pair in copy_cipher) arms="copy cipher";; facts) arms="known invented";; gold) arms="gold gold_shuf";; esac
  for d in 1 2; do
    mkdir -p $S/$pair/d$d/floor $S/$pair/d$d/side
    for save in last 5x; do
      cp experiments/size_matched/results/qwen/$pair/share/floor/$save.json $S/$pair/d$d/floor/
      for arm in $arms; do cp experiments/qwen_library_share/results/p-015e0710/run1/${arm}_s0_${save}_kl.json $S/$pair/d$d/side/; done
    done
  done
done
python experiments/qwen_neuron_draws/plot.py --results $S --figures /tmp/p315_stub/figures
ls /tmp/p315_stub/figures $S
rm -rf /tmp/p315_stub
echo "CPU CHECKS DONE"
