# qwen-lead overnight digest, 2026-11-12

P3.5, due 11-17: filled, 3 of 3 runs. The Qwen fine-tune pair library is built. Copy learns a rule that carries over to new words; cipher only memorises its training words. Neither built pair has both sides learned at the same update size.

## Cells
- P3.1 filled (1/1), P3.2 filled (1/1), P3.3 filled (1/2), P3.5 filled (3/3). No open cell is mine.
- Ladder: L3.1 is done. L3.2 waits on L2.3 (part 2), so no rung is ready. I am finishing.

## What ran and what it found
- **Run 2 (E164).** 24 arms at the original recipe: a learning rate of 2e-6 for 100 steps. Only the gold twin, the real answers beside shuffled ones, is fit to read. No new side learned: cipher scored 0 on its training words. An Opus check of the code and logs found this was not a bug. The learning-rate schedule had fallen to almost nothing before the cipher could be learned.
- **Run 3 (E165).** The four built sides were retrained on the gold arm's recipe: a learning rate of 1e-5 for 300 steps, 3 seeds each. Every side reached 1.00 on its training items. On held-out words, copy scored 1.00 and cipher 0.01-0.02. That is the filed rival_a on all 3 seeds: the built pairs behave as they were built. The caveat: at the saves where the two sides match in update size, the new side has not learned yet. At step 300, where both have learned, the new side's update is about twice the reused side's. So the gold twin is still the only pair with both sides learned at a matched size.
- The page, its figure and its table: `experiments/qwen_pair_library/results/library_page.md`.
- **Premise check (advisory): doubtful.** It says the cipher's "new" label describes the recipe, not the update. It also says the part-2 readings are what make no call, not a missing ground truth. Listed for the 11-18 gate: `~/commute-data/premise/P3.5.md`.

## Spend
- E164 about $70 by the runner reports. E165 $37.36 (six H100 pods, all terminated).
- remote-kernels this session: $37.36 of $300. No pods are left.

## Asks waiting on you
None.

## FYI
- Run 3 retrained copy and known as well, so that each pair shares one recipe. The run-2 copy and known states are kept but are not paired.
- The saved states are on W&B only (`p35r3-*` for run 3). They were not pulled: /mnt/nw is 92% full, and the 11-18 gate decides which to read. `pull.py` knows only run 2's names.

## Friction noted
731, 733 and 741 are job-gate pod-label forms. 736 is the 50 GB pod disk, which the toml's disk setting does not change. 737 is the Vast W&B key. 738 is the missing per-item loss.

## Not run
- No decomposition reading of any library update (it waits for the 11-18 gate).
- The template twin was not refilled; its shuffled side is unlearned, as a shuffled control should be.
- No 2e-5 arm.
