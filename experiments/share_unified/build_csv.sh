#!/usr/bin/env bash
# Cell P2.90: the paper-format CSVs from the re-read results, with jq only (no hand-typed numbers).
#   bash experiments/share_unified/build_csv.sh        (from the clone root)
# Writes
#   results/qwen_share_new.csv     the columns and row layout of the paper's figures/data/qwen_share.csv
#                                  (make_qwen_share.py), values from E328's results/qwen_reread.jsonl: per decomposition
#                                  and side, mean, min and max of the seed 0-2 shares at the last save. random_kl (name
#                                  kept for the figure script) = mean share of E330's gain-matched random draws that
#                                  reached the gain band (status "ok"); empty when none did.
#   results/qwen_share_random_draws.csv  every E330 draw, one row each, to plot each draw as a dot: decomposition,
#                                  side, draw, status, gain, share (share empty when the draw never reached the band).
#   results/known_share_check.csv  E327's 48 67M states (results/repro_67m.jsonl, W0 control left out): the paper's
#                                  share (old), the re-read (new), new - old.
#   results/qwen_share_vs_gain.csv  E332 (cell P2.91): VPD known and copy fine-tunes at full gain, ~10 and ~1.05.
set -euo pipefail
R=experiments/share_unified/results
RAND_FILES=("$R/qwen_random_vpd0.jsonl" "$R/qwen_random_vpd1.jsonl" "$R/qwen_random_neuron0.jsonl" "$R/qwen_random_neuron1.jsonl")

jq -rn --slurpfile ft "$R/qwen_reread.jsonl" '
  def title: if . == "p-9707d6ff" then "VPD decomposition" else "One subcomponent per MLP neuron" end;
  [inputs | select(.kind == "random")] as $rand
  | "decomposition,side,mean,seed_min,seed_max,random_kl",
  ( ["p-9707d6ff", "p-015e0710"][] as $d
  | ["copy", "cipher", "known", "invented", "gold", "gold_shuf"][] as $s
  | [ $ft[] | select(.decomp == $d and .kind == "fine-tune" and .arm == $s and .save == "last") | .share ] as $v
  | if ($v | length) != 3 or ($v | any(. == null)) then error("\($d) \($s): expected three scored seeds") else . end
  | [ $rand[] | select(.decomp == $d and .arm == $s and .status == "ok") | .share ] as $r
  | [($d | title), $s, ($v | add / length), ($v | min), ($v | max),
     (if ($r | length) == 0 then "" else ($r | add / length) end)] | map(tostring) | join(",") )
' "${RAND_FILES[@]}" > "$R/qwen_share_new.csv"

jq -rs '
  def title: if . == "p-9707d6ff" then "VPD decomposition" else "One subcomponent per MLP neuron" end;
  "decomposition,side,draw,status,gain,share",
  ( map(select(.kind == "random")) | sort_by(.decomp != "p-9707d6ff", .arm, .draw)[]
  | [(.decomp | title), .arm, .draw, .status, .gain, (if .status == "ok" then .share else "" end)]
  | map(tostring) | join(",") )
' "${RAND_FILES[@]}" > "$R/qwen_share_random_draws.csv"

jq -rs '
  "state,old,new,diff",
  ( .[] | select(.item != "W0")
  | if .old == null or .new == null then error("\(.item): no old or new share") else . end
  | [.item, .old, .new, (.new - .old)] | map(tostring) | join(",") )
' "$R/repro_67m.jsonl" > "$R/known_share_check.csv"

# Cell P2.91 (E332): each VPD known and copy fine-tune (seeds 0-2, last save) read at its full gain (target "full",
# c = 1) and scaled down to answer gains of about 10 and 1.05 (results/qwen_gain.jsonl). Controls and the repeat draw
# are left out: they are not fine-tune reads.
jq -rs '
  def title: if . == "p-9707d6ff" then "VPD decomposition" else "One subcomponent per MLP neuron" end;
  def order: if .stage == "full" then 0 elif .stage == "g10" then 1 else 2 end;
  "decomposition,side,seed,target,c,gain,share",
  ( map(select(.stage == "full" or .stage == "g10" or .stage == "g1.05"))
  | if length != 18 then error("expected 18 fine-tune reads, got \(length)") else . end
  | sort_by(.arm != "known", .seed, order)[]
  | if .stage != "full" and (.status != "ok" or .in_band != true) then error("\(.item): not in its gain band") else . end
  | [(.item | split("/")[0] | title), .arm, .seed, (.target // "full"), .c, .gain, .share]
  | map(tostring) | join(",") )
' "$R/qwen_gain.jsonl" > "$R/qwen_share_vs_gain.csv"

wc -l "$R/qwen_share_new.csv" "$R/qwen_share_random_draws.csv" "$R/known_share_check.csv" "$R/qwen_share_vs_gain.csv"
