# How to hold one task pair unread until part 2's headline claim is final

*compare-lead, 2026-11-09, for you to approve, change or drop. Nothing here has run. The table of work calls this page P2.7.
The run itself needs a cell only you add.*

**The answer.** Every pair proposed for testing the definition has already been read, so I propose one new pair on the small
model: a fine-tune whose label depends on whether a bracket is still open, beside a twin with shuffled labels. The training job
encrypts every state and its own secret seeds to a key only you hold. One scorer, committed first, reads them once, after part
2's other work closes. It returns one number, and no verdict.

## Which pair

I checked the pairs in the current definition page, `2026-11-09_plan_same-place-definition.md`, by grepping their names across
`experiments/*/results` and `experiments/TRIED.md`.

| Pair | What the files show | Untouched? |
|---|---|---|
| Copy at a new marker against a lookup there | Named in the result files of 27 experiment folders | No |
| Case label, real and shuffled twin | Read in 9 folders, including `case_label`, `case_patch`, `case_label_both`, `method_table` | No |
| Repeat label, real and shuffled twin | Read in `fresh_reuse_pair`, `method_table`, `prior_vs_share` | No |
| Forced and forbidden case label, proposed there | Never trained: `definition_verify` holds forced copy and lowercase only | Unread, but the page proposes it as an input that sets the definition's bar, and its task has been read four times |

All 14 pairs on the scorecard, `experiments/method_table/results/scorecard.json`, have been read by construction. So no proposed
pair qualifies, and I specify a new one. A search for "bracket" and "parenthes" across the same files found only unrelated uses, so
this task has not been run.

**The task.** Whether a bracket is open is a property the model must compute from token order, and text gives it a reason to.

The full specification is in `2026-11-09_plan_sealed-pair_appendix.md`. In brief:

- **Windows.** The repeat-label pair's layout on the 67M model. Two tokens, an open and a close bracket, are written into each window.
  The window is open when the close bracket comes first, and closed when the open bracket comes first. Both classes hold the same
  two tokens, so only their order differs.
- **Real arm.** One label word when open, the other when closed.
- **Twin.** The same windows, with labels shuffled per word. Only memorising can learn it, so it is at chance on held-out words by
  construction.
- **Recipe.** The repeat-label pair's filed recipe, unchanged, from `experiments/fresh_reuse_pair/results/predictions.json`.
- **Seeds.** Three per arm, the number the plan's first gate names, drawn inside the job and stored only inside the encrypted file.
- **Cost.** Free debug queue, one L40 card. I estimate under 20 minutes, from the repeat-label pair's 13 minutes for four fine-tunes
  with more reads. Not measured.

**The labels are set before any fine-tune exists, from the tokens alone,** so the share cannot enter them. They are labels as built.

**Which definition this assumes.** The wording is not adopted, so this is an expectation under the tentative same-place page and
no verdict. There the real arm is expected to read "wrapped": the original model holds whether a bracket is open, and only the last step to a
label word is new. The twin is expected to read "new", since a per-word table has no intermediate value that counts.

- If a wrapped fine-tune does not count as reuse in the wording you adopt, the real arm is no longer built to reuse, and the pair
  tests only wrapped against new.
- A swap that reads the real arm under this wording would be a second read of the sealed states, so this page leaves it out. The
  twin's expected label is the same under any wording.

**The one check allowed before sealing.** The original model is not sealed. The job first reads it alone and stops if it does not
act on the property: the close bracket must be more likely in open windows than in closed ones, beyond the spread across windows. The
bar is filed with the run cell. A miss ends the pair, and a substitute needs a new cell.

## Where the states live

Every lead shares one Unix user and one filesystem, so a file's permissions separate nothing.

| Option | During training | At rest | At the read | Verdict |
|---|---|---|---|---|
| **A. Encrypted at the source to a key on your laptop** | Plaintext only in the job's memory | Ciphertext on shared disk, a copy on your laptop | You paste a passphrase into your own terminal | **Recommended** |
| B. Your own bucket, with a token only you hold | The token must reach the job, so any process of this user can read it | Off-node | Plaintext returns through shared disk | No |
| C. Train on your laptop | Off-node | Off-node | Off-node | No: a local model load killed a session on 2026-10-21 |
| D. A pod you start, downloaded to your laptop | Plaintext on the pod; leads share the RunPod key | Off-node | A second pod | Fallback, a few dollars |

**Recommended steps for A.**

1. You make a key pair on your laptop and commit only the public key.
2. You start the job from your own terminal. The harness's job gate is a Claude Code hook and does not apply there. This departs on
   purpose from "every GPU run goes through a runner".
3. The job holds states, windows and seeds in memory, about 350 MB, and writes one encrypted bundle to `~/commute-data/sealed/p2-7/`,
   with its random passphrase beside it, encrypted to your public key.
4. You copy both files to your laptop.

Before the read a lead can see only that the job ran, and two files it cannot open. At the read you decrypt the passphrase on your
laptop and paste it into a hidden prompt, so it never reaches a file or a process argument. `gpg` is on the dev node; it is not
checked on the GPU worker, which needs a job, so that is the run cell's first step.

**How the result comes back.** The scorer prints one number and writes it to `experiments/sealed_pair/results/score.json`, with its
commit and the bundle's sha256. No state or per-seed value leaves the job.

**What a lead could still learn by accident.**

| Route | What could leak | How it is closed |
|---|---|---|
| Results files | Per-state numbers | Training writes no results folder and no `results/large`. Only `score.json` appears, after the read |
| Job logs | Per-arm accuracy, steps to learn | Logs print the original model's stage 0 reads, a done line and the ciphertext's sha256. Nothing per arm. A state that misses the learning bar prints only "pair unusable" |
| Slurm and `jobs.log` | That a job ran, its name and time | Accepted: neutral name, no arguments |
| W&B | Training curves | The fine-tune and read modules do not import it; the job sets `WANDB_MODE=disabled` |
| Node memory | Plaintext while the job runs | Nothing is written to disk in the clear. A process of this user on the same node could read the job's memory, but only on purpose |
| This page | Anyone could train the pair again | The seeds are secret, and no cell may use this task. The task's design is public by choice |

## The scorer

- **Path.** `experiments/sealed_pair/score.py`, beside `experiments/sealed_pair/train.py`. Neither is written yet.
- **What it reads.** The bundle, the original model and the 67M decomposition. It computes the share for each of the six states: the
  part of the gain that appears when only Γ's importance values change, from the least-squares split at the trigger, on the first 256 fit
  windows. This is `decomp_reads.read_split` called exactly as `references` in `experiments/fresh_reuse_pair/stage2.py` calls it.
- **The one number.** The mean share of the three real seeds minus the mean share of the three twin seeds.
- **How to read it.** Zero means the share does not tell the two arms apart. On the saved repeat-label pair the same difference is
  −0.015, on two seeds per arm, against a seed spread of about 0.056 from the scorecard. If any state's gain is under the filed floor of
  1 log-odds, the number is "none".
- **Control, first and alone.** The scorer runs its do-nothing identity on the original model before it decrypts anything. Its
  positive control runs earlier, as a separate read under normal access: on the four saved repeat-label states it must reproduce the
  shares in `fresh_reuse_pair/results/stage2.json` to 1e-6.
- **Who runs it.** You, from your terminal, in an interactive GPU session on the free queue.
- **Proof the commit came first.** A commit's own date can be set by hand. So you read GitHub's push time for it and write the hash
  into the run cell before training starts. Training and scorer both run from a clean checkout of that hash, so a later edit to shared
  code cannot change the read. The training job seals the hash in the bundle, and the scorer stops if they differ.
- **It reports a number only.** Reading it as reused or new waits for your adopted wording.

## Read once

**When.** After `~/commute-agents/fleet/cell list` shows every other part 2 cell filled or closed, and after you have written down the
claim the number tests. The part's gate is 2026-11-18.

**What stops a second read.** Only you can start it, since only you can recover the passphrase. The scorer also creates
`~/commute-data/sealed/p2-7/READ` before it decrypts and refuses to run if that file exists; this guards against an accident, not
against a person. Once a number is recorded you delete the shared copies and keep the laptop copy as the audit record. Any later
use needs a new cell from you.

**If the read fails.** A technical error means the job ended before `score.json` was written: a crash, out of memory, a lost node, a
failed decryption or a failed do-nothing check. It gets one repeat, at the same commit, after you have seen the error. Logs carry no
state-derived number, so seeing them opens nothing. If a code fix is needed, the new commit goes through the push-time proof again and
the reason goes in the cell. A "none" from the gain floor is a result, and it gets no repeat.

## Questions for you

1. **Which claim does the number test?** The plan names no single headline yet. I assume this one: the share tells a fine-tune built to
   reuse from one built to be new, at matched prior. Real against twin matches the prior, since the windows are the same. It is also
   the hard case: the share overlapped on three of the four shuffled-label pairs. The alternative, a new copy-against-lookup pair, is the
   share's best case but the family read most. **I recommend real against twin.**
2. **Is the bracket task acceptable?** Under the tentative wording it tests wrapped against new, not reused outright. I recommend yes.
3. **Option A, encryption at the source?** It needs `gpg` on your laptop and on the GPU worker. I recommend A, with a pod as the fallback.
4. **If no wording is adopted by the part 2 gate, does the number still stand?** I recommend yes, as a number with no verdict.

## Departures from the literature

None: this is an ordinary held-out test. From our own recipe it departs by a third seed, seeds drawn inside the job, and bracket
order in place of the repeat property.

## Not run

Everything. No key, no task, no fine-tune, no scorer. Not measured: whether the original model tracks open brackets, how many Pile windows
suit the task, and the run's cost. `gpg` on the GPU worker is unchecked. No job and no experiment number were used.
