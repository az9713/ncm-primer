# Project 1: measure the effect of the decode-time penalties

**Goal:** Find how much of the LM1B and XSum scores comes from the token penalties in `pick_tokens()`. Algorithm 2 of the paper does not have them.

**Effort:** flags only. **Status:** run on 2026-11-21 (LM1B, all 1,280 samples) and 2026-11-22 (XSum, first 1,000 documents). Full results: [experiment report](../../experiments/index.html).


## Result (run 2026-11-21/22)

- **What it measured:** the share of the LM1B and XSum scores that comes from the decode-time penalties of `pick_tokens()`.
- **Result:** LM1B with the released penalties: genPPL 94.06 ± 1.42, entropy 4.3138 (Table 8: 93.05, 4.31). Without penalties (Algorithm 2): 78.53, entropy 4.1176.
- **Result:** the frequency penalty causes almost all of the change (94.15 alone). The adjacency penalty alone adds 1.02. On XSum the penalties change ROUGE-L by at most 0.07 (27.78 against 27.71).
- **Unexpected:** the paper's Algorithm 2 has no penalties, but its LM1B number needs them.
- **Lesson:** the penalties trade genPPL for entropy. They move the unigram entropy to the data value (4.31) at a cost of +15.5 genPPL.

## Background

After generation, `scripts/eval_shard.py:99-124` changes the token logits of the decoder. LM1B uses a frequency penalty of 11.5, a deficit bonus of 2.0 and an adjacency penalty `10,4,0.5`. XSum uses 2.0, 1.0 and `5,4,0.5`. With all three at zero, the script takes the argmax (`:100-101`), as Algorithm 2 does.

## What exists and what you write

- Exists: the flags `--freq`, `--mu`, `--adj` (`scripts/eval_shard.py:22-24`), passed through by `evaluate.py`.
- You write: nothing. A shell loop is optional.

## Steps

1. Download LM1B small (0.988 GB) as in [Run an evaluation](../guides/run-an-evaluation.md).
2. Run the default.

   ```bash
   python evaluate.py --config config/eval/lm1b-small.yaml --checkpoint-dir artifacts/lm1b/small --out eval/p1-default
   ```

3. Run with all penalties off.

   ```bash
   python evaluate.py --config config/eval/lm1b-small.yaml --checkpoint-dir artifacts/lm1b/small --freq 0 --mu 0 --adj "" --out eval/p1-argmax
   ```

4. Run with one penalty at a time: `--freq 0 --mu 0` (adjacency only), then `--adj ""` (frequency only).
5. Repeat steps 2 to 4 for XSum with `--data-dir data/xsum`. Use `--n 1000` first to reduce the cost.

## Cost and hardware

One run is 5 seeds × 256 samples = 1,280 samples with 11 passes each, then scoring with `gpt2-large`. The time was not measured. With `--n 64` per seed, the run makes 320 samples.

## How to measure the result

| Run | genPPL (mean ± std over 5 seeds) | entropy |
|-----|----------------------------------|---------|
| default | | |
| argmax | | |
| adjacency only | | |
| frequency only | | |

Report genPPL and unigram entropy together. A lower genPPL with a lower entropy can mean more repetition, not better text. For XSum, report ROUGE-1/2/L and the mean word count.
