# qwen-lead digest, 2026-11-13

All of qwen-lead's cells are filled: P3.1, P3.2, P3.3, P3.5, P3.8 (2 of 2 runs) and P3.9 (2 of 2 runs). Nothing
else is open on a ready rung: L3.2 waits on part 2 and the 11-18 gate. I am finishing because the next work waits
on ask 7.

## What ran and what it found

- **P3.8, due 11-14, filled.** On the main Qwen3-1.7B decomposition, every library fine-tune reads a re-gating share
  of 0.990 to 0.9998, and random changes of the same size read 0.992 to 1.009. All six pair cells sit inside their
  random band, so the share is saturated there and separates no pair. Page:
  `reports/analyses/2026-11-13_analysis_p3-8-library-share.md`.
- **P3.9, due 11-16, filled.** On the frozen-neuron decomposition, MLP only, the share varies: 0.921 to 0.995 for
  the fine-tunes and 0.81 to 1.06 for the random changes. Three of the six cells leave their band. But the gold
  twin, a near-null pair, is one of the three, so the one-draw band is too narrow and no pair is read as differing.
  Page: `reports/analyses/2026-11-13_analysis_p3-9-frozen-neuron-share.md`.
- The premise checks on both cells came back "doubtful". They are advisory and are listed for the gate.

## Waiting on you

- **Ask 7.** Should I open a one-run cell, due 11-16, on the frozen-neuron decomposition? It would read the share
  on a built re-gating change (neurons switched off) beside random changes at three sizes, and place each SFT
  fine-tune against random changes of its own size. It costs about $12 on Vast, or $0 on a cluster L40. Recommended:
  yes. Card: `reports/decisions/2026-11-13_decision_qwen-share-reference.md`, published in the log (v346).

## Money and problems

- **Spend.** The three E179 run-2 pods cost about $25, all Vast H100 NVL at $3.45/h. One of them, $5.64, was lost
  at the provider before any state ran; its pair was re-run on another pod. No pods or jobs are live.
- **Friction noted.** Vast pods sat in "Provisioning" for more than 20 minutes; `download()` flattens a directory;
  `fleet/numbers` reads p-015e0710 as a number. The KL search's factor-1.5 tolerance is not stated on the pages.
