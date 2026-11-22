# What is NCM

NCM generates a sequence of L tokens in K network passes, one pass per level, from a coarse level to a fine level.

## The problem it addresses

An autoregressive model makes one network pass for each token. For L = 128 tokens, that is 128 passes. NCM makes one pass for each level. The LM1B configs have K = 12 levels for L = 128 tokens, so generation needs 11 or 12 passes, not 128.

The paper reports the results below. This repository does not contain these measurements.

| Task | Paper result |
|------|--------------|
| LM1B | Gen. PPL 93.05 at 12 NFE. DUO: 97.30 at 128 NFE. Wall-clock 21× less (1,024 sequences, batch 16, one H100). |
| XSum | ROUGE-L 28.0 at 8 NFE. 31× faster than AURORA-LM. |
| Sudoku (hard) | 86.1% solved with 12 NFE, 25M parameters. |
| WMT14 De–En | BLEU 22.6. With MBR over 12 candidates (132 NFE): 25.2. |

> **Note:** The paper has no wall-clock comparison against an autoregressive model with a KV cache. [Project 9](../projects/09-ar-kv-cache-baseline.md) tells you how to make that measurement.

## How it works

NCM has two stages. Each released model has one file for each stage: `tokenizer.pt` and `generator.pt`.

1. **Stage 1, the tokenizer.** A multiscale residual VQ-VAE (`models/vqvae.py`, `models/quant.py`) changes L tokens into K levels of discrete codes. Level k has ℓ_k codes, with ℓ_1 < … < ℓ_K = L. Level 1 is a short, coarse summary. Each next level adds a correction (a residual) at a longer length.
2. **Stage 2, the generator.** `NextConceptTransformer` (`models/next_concept.py`) predicts the codes one level at a time. In one pass, it predicts all ℓ_k codes of level k in parallel. Its input is the sum of all earlier levels, pooled to ℓ_k positions.

At the end, the tokenizer decoder changes the sum of all K levels back into L token logits. The argmax gives the tokens.

## Architecture overview

```text
                 prefix tokens (source text, puzzle, or one EOS token)
                                   |
+----------------------------------v-----------------------------------+
| NextConceptTransformer.generate()        models/next_concept.py:521  |
|                                                                      |
|  level-1 codes  <- unigram table, learned queries, or prefix head    |
|      |                                                               |
|      v   for k = 1 ... K-1   (loop at next_concept.py:660)           |
|  z_hat += Phi(upsample(codebook_k[codes_k]))   quant.py:697-709      |
|  input = area_pool(z_hat, l_(k+1))                                   |
|  one transformer pass over [prefix ; input]  ->  l_(k+1) codes       |
+----------------------------------+-----------------------------------+
                                   | all codes, shape (B, l_1 + ... + l_K)
+----------------------------------v-----------------------------------+
| Tokenizer decode                                                     |
|  idx_to_zhats() -> full z_hat (B, L, D) -> decoder -> logits (B,L,V) |
|  argmax, or argmax with frequency and adjacency penalties            |
+----------------------------------+-----------------------------------+
                                   v
                     L tokens -> text -> metric
```

## How the pieces fit together

`evaluate.py` starts one `scripts/eval_shard.py` process for each visible GPU. Each shard loads the tokenizer and the generator, and reads its part of the inputs. It calls `net.generate()`, decodes the codes to tokens, and writes a pickle file. Then `evaluate.py` starts `scripts/eval_merge.py`. The merge script reads all pickle files and prints the metric of the task.

## Released models and pass counts

The pass count is the number of transformer passes in one `generate()` call. It was counted on 2026-11-21 with random weights and a forward hook on the first block. The released weights were not loaded.

| Config | K | L | How level 1 is made | Passes |
|--------|---|---|---------------------|--------|
| `lm1b-small`, `lm1b-medium`, `lm1b-large` | 12 | 128 | Drawn from a stored unigram table | 11 |
| `tinystories` | 12 | 128 | Drawn from a stored unigram table | 11 |
| `owt` | 32 | 1024 | Drawn from a stored unigram table | 31 |
| `xsum` | 9 | 48 | Learned queries, one pass | 9 |
| `wmt14` | 11 | 64 | A head on the prefix, one pass | 11 |
| `sudoku-easy`, `sudoku-medium`, `sudoku-hard` | 12 | 89 | Learned queries, one pass | 12 |

## What this is not

- It is not a training codebase. There is no optimizer loop and no data loader. The generator loss and the code-corruption step of the paper are not in the code.
- It is not a library with a stable API. The names come from research code.
- It has no autoregressive baseline and no timing code.
- It does not download files. You download the checkpoints and the data with the `hf` command.

Next: [Key concepts](key-concepts.md), then [Quickstart](../getting-started/quickstart.md).
