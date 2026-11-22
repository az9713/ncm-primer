# Key concepts

These docs use each term below with one meaning only.

**Area pooling** — Shortening of a sequence of vectors by averaging, with `F.interpolate(..., mode="area")`. The tokenizer uses it to go from L positions to ℓ_k positions (`models/quant.py:399`).
Example: 128 vectors pooled to 8 positions; each output is the average of 16 inputs.

**Block-diagonal mask** — The attention mask of the generator. A target position sees the prefix and the positions of its own level only (`models/next_concept.py:274`).

**Codebook** — The table of vectors for one level. Each level has its own codebook (`models/quant.py:294-296`).
Example: TinyStories has 12 codebooks of 4,096 vectors of size 384.

**Code** — One integer that selects one row of a codebook. Level k has ℓ_k codes.

**Classifier-free guidance (CFG)** — A mix of the logits from the real prefix and from a null prefix: `(1 + t) * logits_cond - t * logits_null` (`models/next_concept.py:482`). The weight t increases with the level (`:481`).

**Cumulative reconstruction (z_hat)** — The sum of the upsampled codebook vectors of all levels up to now. Shape (B, L, D). Its pooled form is the generator input for the next level.

**EMA codebook update** — The training-mode update of a codebook by an exponential moving average of the assigned encoder vectors. Gradients do not change the codebooks (`models/quant.py:297-300`, `:485-498`).

**Fine-level dropout (nested cutoff)** — In training mode, with probability `fine_drop.prob`, the decoder gets z_hat cut at a random level (`models/quant.py:525-530`). The decoder thus learns to decode partial sums.

**Frequency and adjacency penalties** — Decode-time changes to the token logits of the tokenizer decoder (`scripts/eval_shard.py:99-124`). They reduce repeated tokens. They do not change the generator.

**Generator** — `NextConceptTransformer` in `models/next_concept.py`. The file `generator.pt` holds its weights.

**Level** — One scale of the code hierarchy. Level k has ℓ_k positions. These docs number levels from 1 to K. The code numbers them from 0 (`level_idx`). Level 1 in these docs is index 0 in the code.

**Level schedule** — The list ℓ_1 … ℓ_K in `manual_levels` of a config.
Example: LM1B `1 8 16 24 32 40 52 64 76 92 108 128`.

**Level-1 mode** — How the generator gets the level-1 codes: unigram table, learned queries, or prefix head. See [the generator](../concepts/generator.md).

**MBR (minimum Bayes risk) decoding** — Generate M candidates. Keep the candidate with the highest mean similarity to the other candidates. The paper uses it for WMT14. The code does not contain it.

**NFE (number of function evaluations)** — The number of network passes for one output. These docs use the word "pass" for one generator NFE.

**Pass** — One forward evaluation of all generator blocks.

**Phi (quant_resi)** — A per-level residual convolution after upsampling: `(1 - r) * h + r * Conv1d(h)`, with r = 0.5 in all configs (`models/quant.py:10-38`, applied at `:380-386`).

**Prefix** — The conditioning tokens: the source text (XSum, WMT14), the puzzle (Sudoku), or one EOS token (unconditional tasks). Its length is `model.prefix_len`.

**Pre-tokenizer** — The text tokenizer under the VQ-VAE: GPT-2, BERT, Qwen3, an 8,192-token BPE, or the Sudoku tokenizer. It changes text into token ids.

**Shard** — One `scripts/eval_shard.py` process on one GPU. Shard k of n handles inputs k, k+n, k+2n, and so on.

**Straight-through estimator (STE)** — A gradient method. The forward pass uses the quantized value. The backward pass copies the gradient to the input. The code uses it at level 1 only (`models/quant.py:503-505`) and once on the final sum (`:531`).

**Tokenizer** — The multiscale residual VQ-VAE and its pre-tokenizer: `VQVAETokenizer` in `tokenization/vqvae_tokenizer.py`. The file `tokenizer.pt` holds its weights.

**Unigram table** — A probability vector over the level-1 codebook, stored in `generator.pt` at `inference.level0_unigram`. The eval script draws level-1 codes from it (`scripts/eval_shard.py:71`, `:156`).

**Symbols** — ℓ_k: positions at level k. L = ℓ_K: sequence length in tokens. K: number of levels. B: batch size. D: tokenizer vector size (`embed_dim`). P: prefix length. V: vocabulary size.
