# Python API reference

This page lists the classes and methods that a script can call. It tracks `models/*.py`, `tokenization/*.py` and `evaluation/gen_ppl.py`. Shapes use B (batch), L (tokens), D (`embed_dim`), K (levels), S = ℓ_1 + … + ℓ_K.

## `tokenization.vqvae_tokenizer.VQVAETokenizer`

| Method | Lines | Input | Output | Status |
|--------|-------|-------|--------|--------|
| `from_config(config)` | `:13-19` | `cfg.tokenizer` with `vqvae.checkpoint_path` set | `VQVAETokenizer`; weights loaded with `strict=True` | Works |
| `encode(text)` | `:31-33` | string | flat list of codes, S per block of L tokens | Not tested |
| `encode_pre_tokenized(ids)` | `:36-39` | list of token ids | flat list of codes; pads to a multiple of L | Not tested |
| `encode_pre_tokenized_idx(idx)` | `:42-45` | tensor (B, L) | tensor (B, S) | Not tested |
| `decode(ids)` | `:47-52` | list of S codes | string | Not tested |
| `reconstruct(text)` | `:54-59` | string | string | Broken: calls `VQVAE.reconstruct` |
| `reconstruct_text_from_ids(ids)` | `:61-66` | list of token ids | string | Broken: calls `VQVAE.reconstruct` |
| `reconstruct_multiscales(text)` | `:68-73` | string | per row, K strings of ℓ_k tokens | Not tested |
| `decode_multiscales(idx, skip_special_tokens=False)` | `:75-86` | tensor (B, S) | `list[B][K]` strings of ℓ_k tokens | Works (Sudoku, 2026-11-21) |

Attributes: `vqvae` (the `VQVAE`), `pre_tokenizer` (an `HFTokenizer`, `BPETokenizer` or `SudokuTokenizer`).

## `models.vqvae.VQVAE`

`VQVAE(in_vocab_size, context_length, embed_dim, vqvae_vocab_size, n_layers, vector_quantizer_config, compression_factor=4, dropout=0.1, pre_quant_groupnorm=None, pre_quant_dropout=None, use_qk_norm=False, enc_layers=0, enc_attn_window=0, enc_conv_layers=0, enc_conv_kernel=3, dec_layers=1, pre_quant_norm_type='group', enc_positional=True, dec_positional=False, **kwargs)` (`models/vqvae.py:11-33`).

| Method | Lines | Input | Output |
|--------|-------|-------|--------|
| `forward(x)` | `:91-95` | tokens (B, L) | `(logits (B, L, V), vq_loss, per_level_util, global_util, 0.1)` |
| `encode(x)` | `:98-101` | tokens (B, L) | codes (B, S) |
| `encode_z(x)` | `:104-105` | tokens (B, L) | pre-quant vectors (B, L, D) |
| `decode(idx)` | `:108-111` | codes (B, S) | logits (B, L, V) |
| `reconstruct(x)` | `:114-118` | tokens (B, L) | Broken: `ValueError: too many values to unpack (expected 3)` (`:116`) |
| `get_ncm_inputs(ids)` | `:121-124` | tokens (B, L) | `(inputs (B, S − ℓ_1, D), targets (B, S − ℓ_1))` |
| `reconstruct_multiscales(idx)` | `:127-141` | tokens (B, L) | `list[B][K]` lists of ℓ_k token ids |

Attributes: `encoder`, `decoder`, `vq` (the quantizer), `context_length`, `embed_dim`, `in_vocab_size`.

## `models.quant.MultiscaleResidualVectorQuantizer`

| Method | Lines | Input | Output |
|--------|-------|-------|--------|
| `forward(z)` | `:448-549` | vectors (B, L, D) | `(z_hat, vq_loss, per_level_util, global_util, 0.1)`; EMA update when training |
| `z_to_idx(z, return_zhats=False, return_zlevels=False)` | `:573-632` | vectors (B, L, D) | codes (B, S); with flags, a tuple with K full-length partial sums and/or K pooled sums |
| `idx_to_zhats(idx, return_zlevels=True)` | `:635-668` | codes (B, S) | `(z_hats, z_levels)`; all `z_hats` entries are one tensor (`:651-652`) |
| `idx_to_ncm_input(idx)` | `:671-694` | codes (B, S) | generator inputs (B, S − ℓ_1, D) |
| `get_next_autoregressive_input(level_idx, z_hat, z_q_level)` | `:697-709` | z_hat (B, L, D), level vectors | `(z_hat, input for next level)`; adds in place |
| `idx_to_z_q(indices)` | `:712-726` | codes (B, S) | z_hat (B, L, D) |
| `set_stochastic_encode(enabled, temperature=0.5, top_k=10, top_p=0.95)` | `:552-556` | — | `z_to_idx` then samples codes from `−distance / temperature` |
| `kmeans_init_level(level_idx, samples, n_iters=20)` | `:345-378` | vectors (N, D) | sets a codebook by k-means |

Attributes: `levels` (list ℓ_1 … ℓ_K), `embeddings` (K `nn.Embedding` codebooks), `vqvae_vocab_sizes`, `embed_dim`. Private helpers `_down_for_quant`, `_up_to_full`, `_apply_quant_resi` (`:380-411`) do one shorten, lengthen and Phi step.

## `models.next_concept.NextConceptTransformer`

`NextConceptTransformer(vqvae_local, config)` (`models/next_concept.py:67-70`).

| Method | Lines | Input | Output |
|--------|-------|-------|--------|
| `generate(idx, prefix_tokens, temperature=1.0, top_k=50, top_p=0.9, use_kv_cache=False, cfg_scale=0.0, cfg_null_eos=50256)` | `:521-703` | level-1 codes (B, ℓ_1) or `None`; prefix (B, P) | codes (B, S) |
| `forward(x, prefix_ids=None, prefix_pad_mask=None)` | `:376-406` | `[prefix vectors ; inputs]` (B, P + T, D) | logits (B, T', V_max); T' adds ℓ_1 with `predict_level0` |
| `build_l0_query_input(batch_size, dtype=None)` | `:408-410` | — | learned level-1 queries (B, ℓ_1, D) |

Attributes: `levels`, `L` (S, the total code count; not the token length), `prefix_len`, `query_level0`, `predict_level0`, `lm_heads`, `transformer` (`wte`, `wle`, `drop`, `blocks`, `ln_f`).

## Pre-tokenizers

| Class | Lines | `eos_token_id` / `pad_token_id` | Notes |
|-------|-------|---------------------------------|-------|
| `HFTokenizer(model_id="gpt2", pad_token=None, eos_token=None, **kwargs)` | `tokenization/hf_tokenizer.py:5-36` | from the Hugging Face tokenizer | Adds the pad token if it is missing (`:13-19`). |
| `BPETokenizer(vocab, merges, special_tokens=None)` | `tokenization/bpe_tokenizer.py:6-99` | ids of `<|endoftext|>`, `<|pad|>` | Byte-level BPE with a GPT-2 split pattern. |
| `SudokuTokenizer` | `tokenization/sudoku_tokenizer.py:4-26` | `None` / `None` | 12 ids: `_`=0, `1`–`9`, `|`=10, `<BOS>`=11. `seq_len` 89, `prompt_len` 91. |

## `evaluation.gen_ppl.generation_perplexity`

`generation_perplexity(ref_model, tokenizer, prompts, generated_texts, *, device=None, score_only_generated=True, batch_size=8, max_length=None)` (`evaluation/gen_ppl.py:20-30`) returns `GenPPLResult(nll_sum, token_count, ppl, mean_nll, mean_ppl)` (`:10-16`). `mean_ppl` is `exp(total NLL / total tokens)` (`:137-140`). Inputs are cut at 1,024 tokens when `max_length` is `None` (`:56`).

## Level schedulers

`utils/registry.py:15-23` defines `linear`, `cubic`, `exp`, `sqrt`, `sine`, `log`, `quadratic`. A config selects one with `levels.use_manual_levels: false`, `scheduler`, `start`, `end`, `num_levels` (`models/quant.py:287-288`). No released config does this.
