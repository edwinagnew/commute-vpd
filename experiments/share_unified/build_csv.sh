#!/usr/bin/env bash
# Cell P2.90: the paper-format CSVs from the re-read results, with jq only (no hand-typed numbers).
#   bash experiments/share_unified/build_csv.sh        (from the clone root)
# Writes
#   results/qwen_share_new.csv     the columns and row layout of the paper's figures/data/qwen_share.csv
#                                  (make_qwen_share.py), values from E328's results/qwen_reread.jsonl: per decomposition
#                                  and side, mean, min and max of the seed 0-2 shares at the last save. random_kl is
#                                  left empty: the run-1 KL-matched randoms are not a reference in this unit (their
#                                  answer gains are negative); E330's gain-matched draws fill it when they land.
#   results/known_share_check.csv  E327's 48 67M states (results/repro_67m.jsonl, W0 control left out): the paper's
#                                  share (old), the re-read (new), new - old.
set -euo pipefail
R=experiments/share_unified/results

jq -rs '
  def title: if . == "p-9707d6ff" then "VPD decomposition" else "One subcomponent per MLP neuron" end;
  "decomposition,side,mean,seed_min,seed_max,random_kl",
  ( ["p-9707d6ff", "p-015e0710"][] as $d
  | ["copy", "cipher", "known", "invented", "gold", "gold_shuf"][] as $s
  | [ .[] | select(.decomp == $d and .kind == "fine-tune" and .arm == $s and .save == "last") | .share ] as $v
  | if ($v | length) != 3 or ($v | any(. == null)) then error("\($d) \($s): expected three scored seeds") else . end
  | [($d | title), $s, ($v | add / length), ($v | min), ($v | max), ""] | map(tostring) | join(",") )
' "$R/qwen_reread.jsonl" > "$R/qwen_share_new.csv"

jq -rs '
  "state,old,new,diff",
  ( .[] | select(.item != "W0")
  | if .old == null or .new == null then error("\(.item): no old or new share") else . end
  | [.item, .old, .new, (.new - .old)] | map(tostring) | join(",") )
' "$R/repro_67m.jsonl" > "$R/known_share_check.csv"

wc -l "$R/qwen_share_new.csv" "$R/known_share_check.csv"
