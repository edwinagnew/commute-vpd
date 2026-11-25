# Redteam of P2.81 (E308, token risers), 2026-11-24 20:37

I checked definition-lead's page `reports/analyses/2026-11-24_analysis_p2-81-token-risers.md` (820a4e3a) against
`experiments/token_risers/results/risers.json`, the filed `predictions.json` and Fig 5's shares in
`experiments/convergence_panel/results/wide/random_3seed.json`. Every number I checked matches its data, and the
headline "no handful of tokens on Swahili and Yoruba" holds, also over bigrams. One thing needs a sentence on the
page: the measure used falls as the gain per token rises, so the contrast between tasks is partly a contrast in how
large their gain is.

## Verdict: OK, with one SHOULD and two nits

**Numbers checked, all match:** top-20 excess per seed (Yoruba 0.102-0.129, Swahili 0.111-0.135, lowercase
0.286-0.317, arXiv 0.335 / 0.645 / 1.568); top-20 gain and position shares; top-500 values; share of positions
gaining; gain per token; KL 0.168-0.194; upper-case mass (lowercase drop 84-87%, others within 12%); the step table;
the class table; Yoruba seed-1 risers (1333 / 0.292 / 0.583 / 2644, etc.); arXiv newline 2226 of 1990 (seed 0) and
1667 of 6848 (seed 1); stable top five (Yoruba) and top three (Swahili, lowercase as a set, arXiv); the Fig 5 seed
means 0.125 / 0.083 / 0.024 / 0.044; scorecard seed means 0.12 and 0.11.

**The arXiv excess of 1.57 is correct for its definition.** Gain share is a share of the net gain. On the 5-step
seed the net gain is 1990 nats, newline alone gains 2226 and word starts lose (class share -0.53), so the top-20
gain share is 1.69 against a position share of 0.125. The page says shares can pass 1. Fine.

**SHOULD: the excess tracks gain per token on all 12 states.** Because the share is divided by the net gain, a
state with a small net gain gets a large excess from a few tokens of fixed size. Sorted by gain per token, the 12
states come out in almost the same order as by excess: arXiv s0 (0.061 nats, 1.57), lowercase and arXiv s2
(0.12-0.13, 0.29-0.65), arXiv s1 (0.21, 0.335), Swahili (0.92-1.12, 0.11-0.135), Yoruba (1.46-1.47, 0.10-0.13). The
same holds inside a task: arXiv's three seeds, and Swahili's 41-step seed (least gain, most concentrated). So "the
two tasks rated reuse are the concentrated ones" cannot be told apart from "the two tasks rated reuse gained least".
The page's "Step counts do not explain the pattern" is true, but gain per token does follow it, and the page should
say so beside the step table and in "How compelling". The headline on the languages is not at risk: their large gain
pushes their excess toward 0, so if anything the true spread is wider.
What would separate them (not run, needs the per-position gains, which were not saved): shuffle the gains across
positions within each state and recompute the top-20 excess, so each state gets its own reading for "spread like
the text". Ranking types by summed gain also picks types that gained by chance, so that reading is above 0 too.

**Support the page could add (from the saved data):** the opening says the result "does not support ... a handful
of bigrams", but the figure measures next-token types only. The saved top-30 bigrams back the sentence: the top 20
bigrams carry 0.17 of the gain on 0.18 of the positions on Yoruba (excess about 0), and 0.20-0.22 on 0.09-0.10 on
Swahili, against 0.24-0.25 on 0.01-0.02 on lowercase and 0.27-0.75 on 0.03 on arXiv.

**Nits.**
- "Reproduced ... to six decimal places": two states differ by 1e-6 at the sixth place (Yoruba s1 1.975531 against
  1.975530, arXiv s2 0.283580 against 0.283581). Say "to 1e-6".
- "Lowercase ... on a few lower-cased sentence starts": its top type is end-of-text, then " the" and ","; the
  sentence-start story is the bigram table. Fine as written if it points to the bigrams.

## Not run
- The shuffle null per state (needs the per-position gains saved; a short GPU rerun).
- Any check of `run.py` itself; I read results and page only.
