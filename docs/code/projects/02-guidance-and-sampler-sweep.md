# Project 2: sweep the guidance weight and the sampler

**Goal:** Plot the BLEU (WMT14) or ROUGE-L (XSum) score as a function of the guidance weight and of the sampler settings.

**Effort:** flags and a shell loop. **Status:** not run.

## Background

- Guidance mixes conditional and null-prefix logits with weight `t = cfg_scale × level / (K − 1)` (`models/next_concept.py:481-482`). At level 1 the weight is 0. The README guided WMT14 row uses `--cfg 1.0`.
- Both configs are greedy by default: `top_k: 1`, `top_p: 0.0` (`config/eval/wmt14.yaml:109-110`, `config/eval/xsum.yaml:107-108`).
- Each guided pass doubles the batch (`models/next_concept.py:536`), so it costs more GPU time per pass.

## What exists and what you write

- Exists: `--cfg`, `--temp`, `--top_k`, `--top_p`.
- You write: a loop over values.

## Steps

1. Download WMT14 (1.127 GB) as in [Run an evaluation](../guides/run-an-evaluation.md).
2. Run the guidance sweep.

   ```bash
   for c in 0 0.5 1.0 1.5 2.0 3.0; do
     python evaluate.py --config config/eval/wmt14.yaml --checkpoint-dir artifacts/wmt14 --data-dir data/wmt14 --cfg $c --out eval/p2-cfg$c
   done
   ```

3. Run the sampler sweep at the best guidance weight. Repeat each setting 3 times, because the conditional path sets no seed.

   ```bash
   for t in 0.6 0.8 1.0; do
     python evaluate.py --config config/eval/wmt14.yaml --checkpoint-dir artifacts/wmt14 --data-dir data/wmt14 --cfg 1.0 --temp $t --top_k 0 --top_p 0.9 --out eval/p2-t$t
   done
   ```

4. For Sudoku, sweep `--temp` only, with `--top_k` 16 or less. Guidance does not work for Sudoku (no `prefix_pad_token_id`).

## Cost and hardware

WMT14 has 3,003 sources with 11 passes each. The time was not measured. Use `--n 500` for a first sweep.

## How to measure the result

- Record BLEU, lowercased BLEU, chrF and length ratio for each run.
- Plot BLEU against `cfg_scale`. Mark the value 1.0 from the README.
- For sampled runs, report the mean and the range of 3 runs.
