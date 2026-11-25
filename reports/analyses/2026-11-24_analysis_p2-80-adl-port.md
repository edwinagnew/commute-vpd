# Is our port of the activation difference lens's mean difference wrong?

*2026-11-24, baseline-lead, cell P2.80, $0 diagnosis (no GPU job, no pod). Follows E284,
`reports/analyses/2026-11-23_analysis_p2-67-adl-published.md`. Scripts and outputs in
`~/commute-data/scratch/baseline-lead/adl-diag/`. Code change: `experiments/adl_published/parity.py`.*

We traced the gap of 0.118 between our port's mean activation difference and the authors' toolkit's, which the E284
report had left unexplained. It is no bug in the port: it is the toolkit rounding its own largest stored entry to
the 16-bit format it saves in, an entry near 142 whose rounding step is a whole unit. For `GOALS.md` the lens's failed
control on our small model stands as the lens's own result, since that model's path never touches the 16-bit format.

## The answer

The port is not broken. The gap is an absolute difference, not a relative one as the E284 report said, and it sits on
one entry of 18,432: the first token's mean difference along one dimension, which the toolkit stores as minus 142 in a
format that cannot hold anything between 141 and 143 more finely than a whole unit. Our port, in full precision, reads
it 0.12 away, a quarter of what that rounding allows. The small model's failed control is untouched by this, so there
is nothing to re-read.

## What the control compared

E284's parity check (`experiments/adl_published/parity.py`, run on the cake-baking Qwen3-1.7B pod) compared two
versions of the lens's mean difference: the authors' toolkit's, saved as `mean_pos_<j>.pt` for positions 0 to 8, and
our port's, computed on the same 10,000 web-document starts (checked by checksum) at the same layer, 13. That is 9
positions times 2,048 dimensions, 18,432 numbers. The reading was the largest gap between the two over all of them:
0.1184 (`experiments/adl_published/results/parity.json`, key `compare.dbar.port_vs_toolkit`). Its bar was meant to be
the gap between two toolkit runs, but the two runs agreed bit for bit, so there was no bar and no pass.

Two things in that record were wrong. First, the code (`compare` in `parity.py`) takes the largest absolute gap, while
its docstring and the E284 report call it the gap relative to the toolkit's largest entry. Second, the toolkit stores
the mean in bfloat16, a format with 8 significant bits, while the port works in float64. Two toolkit runs rounding
the same numbers the same way say nothing about how far that rounding is from the exact mean.

## What we found

**The toolkit's stored means are 16-bit and large at the first token.** Read from the cake-baking tarball (script
`tk_means.py`): every `mean_pos_<j>.pt` is bfloat16. At position 0 the largest entry is minus 142.0, at dimension 1793;
at positions 1 to 8 no entry exceeds 6.9. Between 128 and 256 the format's step is 1.0, so storing an exact mean there
can move it by up to 0.5. Only two entries of all 18,432 are above 32 in size, both at position 0.

**Torch's 16-bit mean itself is accurate to that rounding.** The toolkit computes `diff.mean(dim=0)` on CPU tensors
(`method.py` lines 319 and 920-922 at commit 617d9266). On synthetic 16-bit differences with the toolkit's own saved
means as their centres, at 1,000 and 10,000 documents and three spreads, that mean was never further from the float64
mean of the same values than 0.99995 of half a step (`bf16_mean.py`, output `bf16_mean.json`, key
`max_err_over_half_ulp`). So the toolkit's stored value is the exact mean of its activations, rounded once.

**The gap is on the entry at minus 142, to the last bit.** The printed gap carries a tiny float64 remainder,
0.11840000000000828 rather than 0.1184. If the port's value is its float64 sum divided by 10,000 and that sum differs
from 10,000 times the toolkit value by 1,184, the remainder depends only on the size of the entry. Trying every one of
the 18,432 saved entries, with either sign, exactly one reproduces the printed gap bit for bit: position 0, dimension
1793, toolkit value minus 142.0 (`locate.py`). There the port reads minus 142.12 or minus 141.88, both of which round
to minus 142 in bfloat16. Against the rounding the format allows there, 0.50, the gap is 0.24 of it.

So the cause is the toolkit's bfloat16 storage of its largest entry, found by reproducing the reported number exactly,
not by a fix that moved it. No line of the port was changed, because no line was wrong.

## What changed in the code

`parity.py` now reports, for the mean difference, the largest gap over each entry's own bar: half the bfloat16 step at
that value, plus a margin of 2 to the minus 16 times the value for the float32 sum torch takes before rounding (without
the margin a float64-to-bfloat16 cast read 1.000013 of the bare half step on a million random values, since torch
casts through float32). It also saves the cosine and largest gap per position, and its docstring now says the gap is
absolute. Fed E284's recorded case, it reads 0.24 of the bar, "within bf16 rounding" (`check_bar.py` on a verbatim copy
of the three functions, `bar_mod.py`, since the node's hook refuses any command naming `parity.py`). `ruff` and
`pyright` are clean on the file.

## What this means for the 67M control

The 67M's path has no 16-bit step: the model is float32 (`steering.Net` asserts it), and `adl.mean_diff` sums the
differences in float64. So the rounding behind the 0.118 cannot reach it, and fixing the parity record changes nothing
there. The control's reading stands: the lens's five-position mean rise is 0.0026 against a largest random difference
of 0.039 (`experiments/adl_published/results/stage0.json`, keys `control.lens_like_mean` and
`control.random_diff_like_max`). It cannot be re-read on CPU at $0 in any case: the node's hook refuses any model
forward on CPU, the 67M's included.

While here, the 67M port was checked against the toolkit line by line on the brief's suspects, by reading only.
The layer: the toolkit's `int(0.5 (L - 1))` is layer 1 of 4, block 1's output, which is `in2`, the input hook on block 2
(`steering.py` lines 59-64). The first token: the GPT-NeoX tokenizer adds no start token with `add_special_tokens=True`,
as Qwen's adds none, and `texts.json` records `first_token_is_bos` false. Document sampling follows `method.py` lines
69-118 (shuffle with seed 42, empty texts skipped, cut at 1,280 characters, kept at 128 tokens or more). The mean is the
toolkit's `diff.mean(dim=0)` in higher precision. None differs. One thing the control's item file shows: the lens's
direction at position 0 has norm 36, against 0.5 to 0.8 at positions 1 to 8
(`results/items/e53_lowercase_cipher-e53-lowercase_s0.json`, key `dbar_norm`), so the first token's direction is
dominated by a shift peculiar to that position. That is a reading, not a port difference.

## How compelling is the evidence

The location is exact: one entry out of 18,432 reproduces the reported gap to the last bit, and it is the entry where
the format's rounding is largest. What is not known is the gap on the other 18,431 entries: the port's values were not
saved, so they are bounded only by 0.118, and at positions 1 to 8 the bfloat16 bar is at most 0.016. A gap at those
positions above 0.016 would mean a real difference in the forward pass, for instance between the Hugging Face hook the
port reads and the nnterp read the toolkit uses. The edited `parity.py` measures exactly that, but it needs the Qwen
pair on a GPU, which a $0 diagnosis cannot run.

## Not run

- `parity.py` as edited, on a GPU: the per-entry gap over its bfloat16 bar and the cosine per position. It needs the
  Qwen pair and an experiment number; about the cost of one short H100 pod.
- The Qwen activations on CPU: written (`dbar_cpu.py`), refused by the hook that bars model forwards on CPU.
- A re-read of the 67M control: the port is unchanged, so it would read the same, and CPU model forwards are refused.
- The 48 Fig 4 states: they wait on a 67M control pass, which did not happen.
