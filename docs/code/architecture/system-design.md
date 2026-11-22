# System design

This page describes the components of the evaluation code, how data moves between them, and the cost of one generation. It tracks all files under `models/`, `tokenization/`, `scripts/` and `evaluate.py`.

## Component diagram

```text
evaluate.py ──spawn──> scripts/eval_shard.py (1 per GPU) ──pickle──> scripts/eval_merge.py
                              │                                            │
              ┌───────────────┴──────────────┐                 evaluation/gen_ppl.py
              │                              │                 sacrebleu, rouge_score
   tokenization/vqvae_tokenizer.py   models/next_concept.py     transformers (gpt2, gpt2-large)
     ├─ tokenization/hf_tokenizer.py       └─ models/common.py (Block, RoPE, KV cache)
     ├─ tokenization/bpe_tokenizer.py
     ├─ tokenization/sudoku_tokenizer.py
     └─ models/vqvae.py
          ├─ models/vae.py (Encoder, Decoder)
          └─ models/quant.py (MultiscaleResidualVectorQuantizer)
                └─ utils/registry.py, utils/misc.py
```

## Components

| Component | Files | Responsibility |
|-----------|-------|----------------|
| Launcher | `evaluate.py` | One shard per GPU, then the merge. |
| Shard | `scripts/eval_shard.py` | Load, generate, decode, penalties, write pickle. |
| Merge | `scripts/eval_merge.py`, `evaluation/gen_ppl.py` | Join shards, compute the metric. |
| Tokenizer | `tokenization/*.py`, `models/vqvae.py`, `models/vae.py`, `models/quant.py` | Tokens to codes and codes to token logits. |
| Generator | `models/next_concept.py`, `models/common.py` | Codes of level k+1 from the sum of levels 1 to k and the prefix. |
| Registry | `utils/registry.py` | Name-to-class maps for pre-tokenizers and the quantizer; level schedulers. |

## Data flow for one conditional batch

| Step | Tensor | Shape (XSum) |
|------|--------|--------------|
| Prefix tokens | `x` | (B, 512) |
| Prefix vectors after `encoder.wte` and `transformer.wte` | `prefix_emb` | (B, 512, 768) |
| Level-1 codes from learned queries | `idx` | (B, 4) |
| Generator input for level k+1 | pooled z_hat | (B, ℓ_{k+1}, 384) |
| All codes | `g` | (B, 200) |
| Full z_hat | `z_levels[-1]` | (B, 48, 384) |
| Token logits | decoder output | (B, 48, 50304) |
| Tokens | `pick_tokens()` | (B, 48) |

## Cost per pass

In the eager path, pass k processes P + ℓ_k rows with a dense (P + ℓ_k) × (P + ℓ_k) attention mask (`models/next_concept.py:507-510`). The total over one generation is K × P prefix rows plus S target rows.

| Config | P | S = Σℓ_k | Prefix rows over all passes | Share of prefix rows |
|--------|---|----------|-----------------------------|----------------------|
| `xsum` | 512 | 200 | 9 × 512 = 4,608 | 4,608 / 4,808 = 95.8% |
| `wmt14` | 96 | 313 (312 in passes; level 1 comes from prefix rows) | 11 × 96 = 1,056 | 1,056 / 1,368 = 77.2% |
| `sudoku-*` | 91 | 322 | 12 × 91 = 1,092 | 1,092 / 1,414 = 77.2% |
| `lm1b-*` | 1 | 641 | 11 × 1 = 11 | 11 / 651 = 1.7% |

For XSum, 95.8% of the processed rows are prefix rows. The KV-cache path computes the prefix once, but `generate()` forbids it for `query_level0` configs, which include XSum (`:523-524`).

Measured on 2026-11-21, `sudoku-hard`, batch 128, NVIDIA GeForce RTX 3050 Laptop GPU, eager path, argmax:

| Measurement | Value |
|-------------|-------|
| One `generate()` call (mean of 5, after 1 warm-up) | 3.73 s |
| Pass 1 (91 + 1 rows) | 236.69 ms |
| Pass 12 (91 + 89 rows) | 478.58 ms |

The method is in [Project 6](../projects/06-wall-clock-vs-passes.md).

## Design decisions

The decisions with a reason in the code are in [Design decisions](decisions.md). Other choices, for example the block-diagonal mask and per-level codebooks, have no stated reason in the code. The paper explains them.

## Scaling

- More GPUs: `evaluate.py` splits inputs (conditional) or seeds (unconditional) across GPUs. Unconditional configs have 5 seeds, so more than 5 GPUs give empty shards.
- Longer sequences: the target length L is fixed by the tokenizer and its codebooks. A different L needs a new tokenizer and a new generator.
- The generator recomputes the prefix at each level in the eager path. The cost grows with K × P.

## External dependencies

| Dependency | Use |
|------------|-----|
| Hugging Face Hub | Checkpoints, data, pre-tokenizers, `gpt2`, `gpt2-large` |
| `transformers` | `AutoTokenizer`, `AutoModelForCausalLM` |
| `sacrebleu` | BLEU, chrF (WMT14) |
| `rouge_score` | ROUGE-1/2/L with stemming (XSum) |
| `hydra-core` | `hydra.utils.get_class` only (`tokenization/vqvae_tokenizer.py:15`, `models/vqvae.py:50`) |
| `omegaconf` | Config loading |
