# E118 plan: the Qwen comparator table (baseline retarget, step 2)

**Request, restated in the Terms' words** (`decisions.md`, last baseline-lead row, user 2026-11-06): beside every
existing VPD reading on Qwen3-1.7B (E93's least-squares split table, E94's and E104's D3 rows, E94 Amendment 1, E107
Amendment 1's VPD sets), put the four cheap comparators on the same saved state: (1) update size, ‖δW‖ over ‖W0‖; (2) the
original model's rank of the trained answer and the gain in log P; (3) one steering vector's share of that gain; (4) the
carrying-neuron set (E98's ranking, top 1024). Only on states other leads saved; no new fine-tune, no decomposition.

## /design check

Class 1, analysis of what exists: E92's and E102's harness, unchanged, on saved states, one debug L40 job per part,
$0. It answers the user's step-2 question (does VPD add anything over neurons on the Qwen rows) only as a join; the
reading is step 3's write-up. Nothing in `experiments/TRIED.md` covers it: E92 and E102 left these cells out because
their plans scoped to the anchors and GSM8K arms. No definition is touched (rows 81, 87 frozen); no rule is filed.

## What runs

Shared code: `qwen_comparators.read_state` (E92's Reader.state body) and `qwen_neurons.state_cells` + `set_reads`
(E102's read_state and score.py's per-state scores), moved into `src/` so `run.py` copies nothing; E92's and E102's
`run.py` stay as they ran. `known_facts.said_rows` packs E94 Amendment 1's known-facts rows (and the same with the wrong
answer). Rows per state, E92's rules: anchor 1's rows, TOFU's pair 0, GSM8K gold solutions (u1 and u2 trained on gold;
the RL arms train no fixed token), and for the known and wrong arms pair 0 of what each trained (the sentences). The
carrying set on the known end states uses E103's probe (pair 0's bare answers), as the stored known sets.

| part | states | time |
|---|---|---|
| `--part compare` (E92) | u1 (fp32 file), u2 (bf16), known s42/s142 matched and end, wrong s42/s142 matched | ~12 min |
| `--part carry` (E102) | R/N/F s0 and R/N s70 at steps 10 and 100, a2 past s42 step 550 and s142 step 500, u1, u2, known s42/s142 end | ~21 min |

## Expected values (the filed controls, `results/predictions.json`)

- **C0, W0 (the do-nothing file, anchor 1 seed 0 step 1)**: size 0, max |Δ log P| 0 (band 1e-5), no steering
  layer; carrying T1 − T0 = 0, nums[N] = 0, empty set. E92 and E102 read exactly these.
- **C1, E92's control state (anchor 1 seed 0 step 100)**: norm 0.48652 (rel 1e-9), gain 0.66615 (rel 1e-3), W0 mean
  log P −1.5376 / −4.2481 / −2.7290 on a1 / TOFU / gold rows (rel 1e-3), layer 27, steering share 0.185 ± 0.019 and
  random share −0.038 ± 0.0066 (E92's stored se).
- **C2, E102's control state (the same)**: gain 0.66615 and nums[N] 0.60709 (rel 1e-3), set overlap ≥ 0.95 with
  E102's set; expected overlap 1.0. k_half 256, neuron share 0.911, engaged 0.496, prompt overlap 0.056 logged.
- No expected value is filed for the new cells: they are a join. For orientation only: u1 and u2 gains on gold rows
  should be positive (they trained on gold); the RL arms' gold gains are E92's (−0.28 to +0.60 at step 100).

## Departures

- The rank cell on GSM8K arms, u1 and u2 is read on gold solutions (E92's departure, kept). On the D3 maths rows it
  does not apply (W0's own greedy answers, rank 1 by construction) and is not computed.
- The known and wrong arms' E92 rows are the sentences each trained; their carrying sets stay on E103's bare-answer
  probe, so the two columns read different rows on those states (as stored).
- The gold rows' engaged values are read here on E102's own 8-row FineWeb sample (rows 2000-2008), E102 having
  none for gold.

## Not run

- Comparators on the constructed controls (E94's Gaussian and alive-set changes, E104's scaled copies S1, S1a1, S2,
  S3): built inside a job, not trained states; "does not apply".
- Neuron share, k_half, engaged and prompt overlap for the four stored E103/E107 sets (known and wrong matched): those
  jobs read the gain and set only. Recomputing them is a cheap follow-up, not in this brief.
- The rank on the D3 maths rows; E92's anchor-separation stop rule (`separates`); any rule joining a comparator to a
  VPD reading.
- A second carrying probe (TOFU rows) for the known end states, and the wrong arm's carrying set on its own sentences.
