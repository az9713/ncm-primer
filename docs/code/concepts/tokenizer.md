# The tokenizer

The tokenizer is a multiscale residual VQ-VAE. It changes L tokens into K levels of codes, and K levels of codes back into L token logits.

This page tracks `models/vqvae.py`, `models/vae.py`, `models/quant.py`, `tokenization/vqvae_tokenizer.py` and `utils/registry.py`.

## Why it is a separate stage

The generator does not predict tokens. It predicts codes. The tokenizer defines what the codes mean. Each released model has one frozen tokenizer (`tokenizer.pt`). The generator keeps the tokenizer in a tuple, so the tokenizer weights are not in `generator.pt` (`models/next_concept.py:80-81`).

## Components

| Component | Class | Location | Released configs |
|-----------|-------|----------|------------------|
| Pre-tokenizer | `HFTokenizer`, `BPETokenizer`, `SudokuTokenizer` | `tokenization/*.py`, selected at `utils/registry.py:4-9` | GPT-2, BERT uncased, Qwen3-0.6B, BPE 8,192, Sudoku 12 |
| Encoder | `Encoder` | `models/vae.py:102-179` | Embedding lookup, plus a learned position embedding except LM1B (`enc_positional: false`) |
| Pre-quant norm | `nn.GroupNorm(..., affine=False)` | `models/vqvae.py:59-67` | 4 groups (LM1B) or 8 groups |
| Quantizer | `MultiscaleResidualVectorQuantizer` | `models/quant.py:228` | K = 9 to 32 levels |
| Decoder | `Decoder` | `models/vae.py:198-258` | 1 block, or 4 blocks for LM1B (`dec_layers: 4`) |

> **Note:** The config field `n_layers` is the attention head count of the encoder and decoder blocks, not a depth (`models/vae.py:132`, `:224`). The depth comes from `enc_layers` and `dec_layers`.

## How encoding works

`VQVAE.encode(x)` (`models/vqvae.py:98-101`) runs `_encode_pre_quant()` and then `vq.z_to_idx()` (`models/quant.py:573-632`). For each level k with ℓ_k positions:

1. Shorten the residual to ℓ_k positions (`_down_for_quant`, `models/quant.py:395-402`). Level 1 uses a learned strided convolution when `learned_l1_sampling: true` (all released configs). The other levels use area pooling. Level K uses no shortening.
2. Find the nearest codebook vector for each position by squared distance (`:388-393`, `:595-600`).
3. Lengthen the selected vectors to L positions (`_up_to_full`, `:404-411`). Level 1 uses a learned transposed convolution. The other levels use linear interpolation.
4. Apply Phi: `(1 - 0.5) * h + 0.5 * Conv1d(h)`, with a separate convolution for each level (`share_mode: 0`, `:108-143`, `:380-386`).
5. Add the result to z_hat. Subtract it from the residual (`:609-610`).

The output is one tensor of shape (B, ℓ_1 + … + ℓ_K). Level 1 is first.

The learned level-1 resampler needs L to be a multiple of ℓ_1 (`models/quant.py:311-316`). With `learned_l1_cascade: true`, the stride L/ℓ_1 is split into factors of at most 4 (`:165-178`). For XSum, L/ℓ_1 = 48/4 = 12, so the cascade uses strides 4 and 3.

> **Warning:** `encode()` accepts exactly `context_length` tokens. With 100 tokens on the TinyStories config (L = 128), the check on 2026-11-21 raised `RuntimeError: Calculated padded input size per channel: (1). Kernel size: (2)`. Pad the input to L. `VQVAETokenizer._adjust_length()` does this (`tokenization/vqvae_tokenizer.py:21-28`).

## How decoding works

There are three decode paths. They give different lengths.

| Path | Code | Output length per level |
|------|------|-------------------------|
| Full decode | `VQVAE.decode(idx)` → `vq.idx_to_z_q()` → `decoder` (`models/vqvae.py:108-111`, `models/quant.py:712-726`) | L, final level only |
| Eval decode | `vq.idx_to_zhats(g, return_zlevels=True)`, then `decoder(z_levels[-1])` (`scripts/eval_shard.py:143-144`) | L, final level only |
| Per-level decode | `VQVAETokenizer.decode_multiscales(idx)` (`tokenization/vqvae_tokenizer.py:75-86`) | ℓ_k tokens at level k |

`decode_multiscales` gives the decoder the area-pooled sum at ℓ_k positions (`models/quant.py:653-662`). The decoder has no fixed length, so it returns ℓ_k tokens. For a Sudoku board, the per-level outputs had lengths 1, 2, 4, 6, 9, … 89 on 2026-11-21.

The training-mode forward gives the decoder the full-length sum (`models/quant.py:514`, `:525-531`). To decode a partial sum at full length L, use the `partial_decodes()` helper in [Use NCM from Python](../guides/use-ncm-from-python.md).

> **Warning:** `idx_to_zhats()` returns a list `z_hats` whose entries are all the same tensor. The loop adds in place (`z_hat.add_()`) and appends the same object (`models/quant.py:651-652`). The check on 2026-11-21 found `zh[0] is zh[-1]` is `True`. Do not use `z_hats` for partial decodes. `z_to_idx(..., return_zhats=True)` does not have this problem (`:609-611`).

## Training mode

The training-mode forward is `MultiscaleResidualVectorQuantizer.forward(z)` (`models/quant.py:448-549`), called by `VQVAE.forward(x)` (`models/vqvae.py:91-95`). It returns `(z_hat, vq_loss, per_level_codebook_utilization, global_codebook_utilization, 0.1)`. The last value is a constant, not a perplexity (`:548`).

| Mechanism | Lines | What it does |
|-----------|-------|--------------|
| EMA codebook update | `:485-498` | Cluster size and vector sums with decay `decay` (0.99; 0.95 for TinyStories), Laplace smoothing with `epsilon`. Runs only when `self.training`. |
| Dead-code restart | `:499-500`, `:437-446` | Replaces codes with EMA size under `restart_threshold`. Default 0.0, so off in all configs. |
| Straight-through at level 1 | `:503-505` | Only at level 1 and only with `learned_l1_sampling`. The comment says higher levels diverge within about 100 steps. |
| Loss terms | `:519-524` | For each level: `commitment_cost * mse(z_hat.detach(), z) + mse(z_hat, z_no_grad)`, then the mean over all levels. |
| Fine-level dropout | `:525-530` | With probability `fine_drop.prob`, replaces z_hat with the partial sum at a random level from `min_keep - 1` to K−2 (0-based). |
| Final STE | `:531` | `z_hat = z + (z_hat - z).detach()`, so the decoder gradient reaches the encoder. |

The loss terms are averaged over all levels before the fine-level cutoff. So the cutoff changes only the decoder input. It does not change `vq_loss`.

The token reconstruction loss (cross-entropy of the decoder logits against the input tokens) is not in the repository. A training script must add it.

## Gotchas

- `VQVAE.reconstruct()` unpacks 3 values from `self.vq(z)`, but `forward` returns 5 (`models/vqvae.py:116`, `models/quant.py:543-549`). It raises `ValueError: too many values to unpack (expected 3)`. `VQVAETokenizer.reconstruct()` and `reconstruct_text_from_ids()` call it and fail too.
- `reconstruct_multiscales()` works. It returns ℓ_k tokens per level, like `decode_multiscales`.
- The `__main__` blocks of `models/quant.py` and `models/vqvae.py` do not run. `quant.py:730-732` passes an int vocabulary size and a list of levels. The check on 2026-11-21 raised `AttributeError: 'list' object has no attribute 'use_manual_levels'`.
- The `quadratic` level scheduler divides by `(n-2)**2`, the other schedulers by `(n-1)` (`utils/registry.py:22`). No released config uses a scheduler; all use `manual_levels`.

Related: [The generator](generator.md), [Config reference](../reference/config.md).
